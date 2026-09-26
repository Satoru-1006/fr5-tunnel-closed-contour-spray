"""Run the D49 continuous self-collision verifier in isolated processes.

The verifier itself is deliberately serial because the installed FCL 0.7
continuous BVH traversal is not re-entrant in this environment.  This driver
parallelizes only at process boundaries, preserves one evidence directory per
case, and aggregates only completed machine-readable summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "outputs" / "D48_STAGE4C_EXECUTION_FORM_V1" / "C3_ULTRA_SLOW_005" / "native_manifest.csv"
URDF = ROOT / "outputs" / "stage3_h4_5_robot_fixture_geometry_codesign_20260809T000000Z" / "derived_robot_model.urdf"
SRDF = ROOT / "ros2_moveit_bridge" / "config" / "fairino5_v6_spray_tcp.srdf"
BINARY = ROOT / "tmp" / "stage4d_install" / "lib" / "stage4d_continuous_self_collision" / "stage4d_continuous_self_collision"
DEFAULT_OUTPUT = ROOT / "outputs" / "D49_STAGE4D_SHADOW" / "c2_fcl_isolated_v2"


@dataclass(frozen=True)
class Case:
    case_id: str
    trajectory_csv: str
    family: str


def wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    return f"/mnt/{drive}{resolved.as_posix()[2:]}"


def read_manifest(path: Path = MANIFEST) -> list[Case]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"case_id", "trajectory_csv", "family"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"manifest missing required columns: {path}")
    return [Case(row["case_id"], row["trajectory_csv"], row["family"]) for row in rows]


def shell_command(case: Case, output: Path, max_segments: int) -> str:
    env = (
        "source /opt/ros/jazzy/setup.bash; "
        f"source {shlex.quote(wsl_path(ROOT / 'install' / 'setup.bash'))}; "
        "export LD_LIBRARY_PATH="
        f"{shlex.quote(wsl_path(BINARY.parent.parent))}:"
        f"{shlex.quote(wsl_path(ROOT / 'install' / 'lib'))}:"
        f"{shlex.quote(wsl_path(ROOT / 'tmp' / 'd41_install' / 'lib'))}:"
        "/opt/ros/jazzy/lib:/opt/ros/jazzy/lib/x86_64-linux-gnu:"
        "/opt/ros/jazzy/opt/sdformat_vendor/lib:/opt/ros/jazzy/opt/gz_math_vendor/lib:"
        "/opt/ros/jazzy/opt/gz_utils_vendor/lib:/opt/ros/jazzy/opt/gz_tools_vendor/lib:"
        "/usr/lib/x86_64-linux-gnu; "
    )
    arguments = [
        wsl_path(BINARY),
        "--manifest", wsl_path(MANIFEST),
        "--urdf", wsl_path(URDF),
        "--srdf", wsl_path(SRDF),
        "--output", wsl_path(output),
        "--case", case.case_id,
    ]
    if max_segments:
        arguments.extend(["--max-segments", str(max_segments)])
    return env + "mkdir -p " + shlex.quote(wsl_path(output)) + "; exec " + " ".join(shlex.quote(arg) for arg in arguments)


def run_one(case: Case, output_root: Path, max_segments: int, timeout_s: int) -> dict[str, Any]:
    output = output_root / case.case_id
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "continuous_self_collision_launch.log"
    command = ["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", shell_command(case, output, max_segments)]
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        try:
            completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=timeout_s)
            return_code = completed.returncode
        except subprocess.TimeoutExpired:
            return_code = 124
            log.write("\nD49_DRIVER_TIMEOUT\n")
    summary_path = output / "continuous_self_collision_case_summary.jsonl"
    summary: dict[str, Any] | None = None
    if summary_path.is_file():
        with summary_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    summary = json.loads(line)
                    break
    return {
        "case_id": case.case_id,
        "family": case.family,
        "return_code": return_code,
        "output": str(output),
        "complete": bool(summary and summary.get("complete_trajectory")),
        "status": summary.get("status") if summary else "MISSING_SUMMARY",
        "summary": summary,
    }


def aggregate(results: list[dict[str, Any]], output_root: Path, max_workers: int, max_segments: int) -> dict[str, Any]:
    summaries = [result["summary"] for result in results if result.get("summary")]
    complete_pass = all(
        result["return_code"] == 0
        and result["complete"]
        and result["status"] == "PASS"
        for result in results
    )
    totals = {
        key: sum(int(summary.get(key, 0)) for summary in summaries)
        for key in (
            "swept_interval_count",
            "swept_pair_calls",
            "broadphase_rejected_pairs",
            "continuous_collision_count",
            "continuous_api_error_count",
            "endpoint_contact_count",
        )
    }
    checked_pairs = sorted({pair for summary in summaries for pair in summary.get("checked_link_pairs", [])})
    payload = {
        "schema_version": "d49-c2-fcl-isolated-campaign-v1",
        "status": "PASS" if complete_pass else "INCOMPLETE_OR_CONTACT",
        "continuous_self_collision_status": "PASS_ZERO_CONTACTS" if complete_pass else "NOT_CERTIFIED",
        "measurement_backend": "FCL_0.7_continuousCollide",
        "process_isolation": True,
        "max_workers": max_workers,
        "max_segments_per_case": max_segments if max_segments else None,
        "case_count": len(results),
        "completed_case_count": sum(bool(result["complete"]) for result in results),
        "passing_case_count": sum(result["status"] == "PASS" and result["complete"] for result in results),
        **totals,
        "checked_link_pairs": checked_pairs,
        "distance_m": None,
        "penetration_depth_m": None,
        "distance_and_penetration_status": "not_available",
        "scope": "Stage 0/1 ON-state open-arch only; persisted D48 post-Ruckig trajectories",
        "cases": results,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "continuous_self_collision_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--case", action="append", dest="case_ids")
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--max-segments", type=int, default=0)
    parser.add_argument("--timeout-s", type=int, default=10 * 60 * 60)
    args = parser.parse_args(argv)
    for required in (MANIFEST, URDF, SRDF, BINARY):
        if not required.is_file():
            raise FileNotFoundError(required)
    cases = read_manifest()
    if args.case_ids:
        selected = set(args.case_ids)
        cases = [case for case in cases if case.case_id in selected]
    if not cases:
        raise ValueError("no cases selected")
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
        futures = [executor.submit(run_one, case, args.output_root, args.max_segments, args.timeout_s) for case in cases]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"{result['case_id']}: return_code={result['return_code']} status={result['status']}", flush=True)
    results.sort(key=lambda item: item["case_id"])
    payload = aggregate(results, args.output_root, max(1, args.max_workers), args.max_segments)
    print(json.dumps({key: payload[key] for key in ("status", "case_count", "completed_case_count", "continuous_collision_count", "continuous_api_error_count")}, indent=2))
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
