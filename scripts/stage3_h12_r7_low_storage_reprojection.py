#!/usr/bin/env python3
"""Low-storage authoritative H12-R7 reprojection and certification runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import stage3_h12_r5_ruckig_root_cause as r5
from src.stage3_h12_r7 import canonical_hash

R5A_ROOT = ROOT / "outputs/stage3_h12_r5a_moveit_helper_semantics_closure_20260812T154400Z"
R6_ROOT = ROOT / "outputs/stage3_h12_r6_boundary_derivative_dynamic_remediation_20260813T000100Z"
R7A_ROOT = ROOT / "outputs/stage3_h12_r7a_spray_process_root_cause_audit_20260813T010000Z"
INPUT = R5A_ROOT / "h12_r5a_native_input.npz"
MANIFEST = R5A_ROOT / "h12_r5a_primitive_manifest.jsonl"
EXPECTED_PRIMITIVES = 480
EXPECTED_WINDOWS = 189216
EXPECTED_PRE_REPAIR = 2304


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protected_snapshot() -> list[dict[str, Any]]:
    paths = [
        R7A_ROOT / "stage3_h12_r7a_terminal_certificate.json",
        R7A_ROOT / "h12_r7a_root_cause.json",
        R6_ROOT / "stage3_h12_r6_terminal_certificate.json",
        R5A_ROOT / "stage3_h12_r5a_terminal_certificate.json",
        INPUT, MANIFEST,
        ROOT / "config/stage3/stage3_h6_1_spray_process_tolerance_contract.json",
    ]
    return [{"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in paths]


def array_semantic_hash(case: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    rows = sorted(load_jsonl(Path(case["result_path"])), key=lambda row: int(row["unit_index"]))
    for row in rows:
        path = Path(case["result_path"]).parent / str(row["post_ruckig_arrays_file"])
        with np.load(path, allow_pickle=False) as arrays:
            digest.update(str(row["unit_id"]).encode("utf-8"))
            for key in ("time_s", "positions_rad", "velocities_rad_s", "accelerations_rad_s2"):
                array = np.ascontiguousarray(arrays[key])
                digest.update(key.encode("ascii")); digest.update(str(array.shape).encode("ascii")); digest.update(array.dtype.str.encode("ascii")); digest.update(array.tobytes())
    return digest.hexdigest()


def collect_repairs(run_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    events: list[dict[str, Any]] = []
    summaries = []
    for path in sorted((run_root / "h12_r7_repair_units").glob("*.json")):
        payload = load_json(path)
        summaries.append(payload["summary"])
        events.extend(payload["events"])
    summary = {
        "repaired_samples": len(events),
        "repair_failures": sum(int(row["repair_failure_count"]) for row in summaries),
        "eligible_samples": sum(int(row["eligible_count"]) for row in summaries),
        "max_tcp_error_before_m": max((float(row["max_tcp_error_before_m"]) for row in summaries), default=0.0),
        "max_tcp_error_after_m": max((float(row["post_reprojection_max_tcp_error_m"]) for row in summaries if row.get("post_reprojection_max_tcp_error_m") is not None), default=0.0),
        "max_joint_correction_rad": max((float(row["max_joint_correction_rad"]) for row in summaries), default=0.0),
        "post_reprojection_tcp_position_violations": sum(int(row["post_reprojection_spray_process_violations"]) for row in summaries),
        "immutable_samples_unchanged": all(row["immutable_samples_unchanged"] for row in summaries),
        "affected_primitive_instances": sum(bool(row["eligible_count"]) for row in summaries),
    }
    return events, summary


def run_tests(output: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    focused_cmd = [sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h12_r7.py", "tests/test_stage3_h12_r7a.py"]
    regression_cmd = [sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h12_trajectory_repair.py", "tests/test_stage3_h12_certification.py", "tests/test_stage3_h12_no_leakage.py", "tests/test_stage3_h12_r5.py", "tests/test_stage3_h12_r5a.py", "tests/test_stage3_h12_r6.py", "tests/test_stage3_h12_r7a.py", "tests/test_stage3_h12_r7.py"]
    results = []
    lines = []
    for name, command in (("FOCUSED_TESTS", focused_cmd), ("REGRESSION_TESTS", regression_cmd)):
        proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1200, check=False)
        status = "PASS" if proc.returncode == 0 else "FAIL"
        results.append({"name": name, "status": status, "returncode": proc.returncode})
        lines.extend([f"{name}: {status}", f"RETURN_CODE: {proc.returncode}", (proc.stdout or "").strip(), (proc.stderr or "").strip(), ""])
    (output / "test_summary.txt").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return results[0], results[1]


def output_size_mb(output: Path) -> float:
    return sum(path.stat().st_size for path in output.rglob("*") if path.is_file()) / (1024.0 * 1024.0)


def report_text(cert: Mapping[str, Any], output: Path) -> str:
    ordered = [
        "STAGE_3_H12_R7", "R7_FIRST_BLOCKER", "STAGE_3_H12", "H12_FIRST_BLOCKER", "READY_FOR_STAGE_3_H13",
        "LOW_STORAGE_MODE", "PRE_REPAIR_SPRAY_PROCESS_VIOLATIONS", "POST_REPAIR_SPRAY_PROCESS_VIOLATIONS",
        "TCP_POSITION_VIOLATIONS", "STANDOFF_VIOLATIONS", "SURFACE_NORMAL_VIOLATIONS", "TCP_ORIENTATION_VIOLATIONS",
        "REPAIRED_SAMPLES", "UNCHANGED_SAMPLES", "MAX_TCP_ERROR_BEFORE_M", "MAX_TCP_ERROR_AFTER_M", "MAX_JOINT_CORRECTION_RAD",
        "PRIMITIVES", "TOTG_FINAL_PASS", "RUCKIG_SMOOTHING_SUCCESS", "RUCKIG_DURATION_CEILING_HIT", "RUCKIG_NATIVE_ERRORS", "POST_RUCKIG_VALIDATED",
        "POSITION_VIOLATIONS", "VELOCITY_VIOLATIONS", "ACCELERATION_VIOLATIONS", "JERK_VIOLATIONS", "COLLISION_VIOLATIONS", "SPRAY_PROCESS_VIOLATIONS",
        "REPLAY", "UPSTREAM_IMMUTABLE", "H12_R7A_IMMUTABLE", "FOCUSED_TESTS", "REGRESSION_TESTS", "NEW_REGRESSION_FAILURES",
        "PHYSICAL_ROBOT_CONNECTED", "FJT_GOALS_SENT", "PHYSICAL_MOTION", "AUTHORITATIVE_OUTPUT_SIZE_MB", "LARGE_REPLAY_DIRECTORIES_RETAINED", "FULL_NATIVE_SAMPLE_DUMPS_RETAINED", "TEMP_ARTIFACTS_CLEANED",
    ]
    lines = ["# Stage 3 H12-R7 — Low-Storage Localized Post-TOTG TCP Reprojection", "", "```text"]
    lines.extend(f"{key}: {cert.get(key)}" for key in ordered)
    lines.extend(["```", "", "Collision method: `adaptive_discrete_interpolation`; CCD and clearance: `not_available` / `null`.", "", f"Authoritative directory: `{output.resolve()}`", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--timeout-s", type=int, default=3600)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = (args.output_dir or ROOT / "outputs" / f"stage3_h12_r7_low_storage_reprojection_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}").resolve()
    if output.exists() and not args.resume:
        raise RuntimeError(f"refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=args.resume)
    r7a = load_json(R7A_ROOT / "stage3_h12_r7a_terminal_certificate.json")
    if r7a.get("STAGE_3_H12_R7A") != "PASSED" or int(r7a.get("SPRAY_PROCESS_VIOLATIONS", -1)) != EXPECTED_PRE_REPAIR:
        raise RuntimeError("authoritative R7A prerequisite failed")
    before = protected_snapshot()
    interposer = output / "libstage3_h12_r5_ruckig_interposer.so"
    if not interposer.is_file():
        interposer = r5.build_interposer(output)
    replay_rows: list[dict[str, Any]] = []
    primary_events: list[dict[str, Any]] = []
    primary_repair: dict[str, Any] = {}
    checkpoint_path = output / "resume_checkpoint.json"
    if args.resume:
        if not checkpoint_path.is_file() or not (output / "repaired_sample_manifest.jsonl").is_file():
            raise RuntimeError("resume checkpoint or primary repair manifest missing")
        checkpoint = load_json(checkpoint_path)
        replay_rows = list(checkpoint["replay_rows"])
        primary_repair = dict(checkpoint["primary_repair"])
        primary_events = load_jsonl(output / "repaired_sample_manifest.jsonl")
        stale_run = (output / "native_runs/streaming_replay").resolve()
        if stale_run.is_dir():
            if stale_run.parent != (output / "native_runs").resolve() or stale_run.name != "streaming_replay":
                raise RuntimeError(f"unsafe stale replay cleanup target: {stale_run}")
            shutil.rmtree(stale_run)
    os.environ.update({"H12_R6_RECONDITION_DERIVATIVES": "1", "H12_R7_REPROJECT": "1", "H12_R7A_PROCESS_AUDIT": "0"})
    try:
        for run_number in range(len(replay_rows) + 1, 4):
            case = r5.run_case(output, INPUT, MANIFEST, interposer, "streaming_replay", mitigate=True, installed_helpers=True, hard_limit_rule=True, boundary_velocity_recondition=True, timeout_s=args.timeout_s)
            counts = r5.case_counts(case); counts.pop("rows", None)
            events, repair = collect_repairs(Path(case["result_path"]).parent)
            trajectory_hash = array_semantic_hash(case)
            replay_rows.append({"run": run_number, "result_semantic_sha256": case["semantic_sha256"], "trajectory_semantic_sha256": trajectory_hash, "repair_manifest_sha256": canonical_hash(events), "counts_sha256": canonical_hash(counts), "counts": counts, "repair_summary": repair})
            if run_number == 1:
                primary_events, primary_repair = events, repair
                write_jsonl(output / "repaired_sample_manifest.jsonl", primary_events)
            write_json(checkpoint_path, {"replay_rows": replay_rows, "primary_repair": primary_repair})
            run_root = Path(case["result_path"]).parent.resolve()
            if run_root.parent != (output / "native_runs").resolve() or run_root.name != "streaming_replay":
                raise RuntimeError(f"unsafe temporary cleanup target: {run_root}")
            shutil.rmtree(run_root)
    finally:
        for key in ("H12_R6_RECONDITION_DERIVATIVES", "H12_R7_REPROJECT", "H12_R7A_PROCESS_AUDIT"):
            os.environ.pop(key, None)
        native_parent = output / "native_runs"
        if native_parent.is_dir() and not any(native_parent.iterdir()):
            native_parent.rmdir()
    after = protected_snapshot()
    immutable = before == after
    hashes_equal = len(replay_rows) == 3 and all(len({row[key] for row in replay_rows}) == 1 for key in ("result_semantic_sha256", "trajectory_semantic_sha256", "repair_manifest_sha256", "counts_sha256"))
    counts = replay_rows[0]["counts"] if replay_rows else {}
    full_native = bool(
        counts.get("PRIMITIVES") == counts.get("TOTG_PASS") == counts.get("RUCKIG_SMOOTHING_SUCCESS") == EXPECTED_PRIMITIVES
        and counts.get("validated_windows") == EXPECTED_WINDOWS
        and all(counts.get(key) == 0 for key in ("position_violations", "velocity_violations", "acceleration_violations", "jerk_violations", "collision_violations", "process_violations", "RUCKIG_DURATION_CEILING_HIT", "RUCKIG_NATIVE_ERRORS"))
    )
    repair_ok = bool(primary_repair.get("eligible_samples") == primary_repair.get("repaired_samples") == EXPECTED_PRE_REPAIR and primary_repair.get("post_reprojection_tcp_position_violations") == 0 and primary_repair.get("repair_failures") == 0 and primary_repair.get("immutable_samples_unchanged"))
    focused, regression = run_tests(output)
    tests_ok = focused["status"] == regression["status"] == "PASS"
    blocker = next((name for ok, name in ((immutable, "upstream_immutability_violation"), (repair_ok, "localized_tcp_reprojection_incomplete"), (full_native, "post_repair_native_certification_failed"), (hashes_equal, "nondeterministic_replay"), (tests_ok, "tests_failed")) if not ok), "none")
    passed = blocker == "none"
    write_jsonl(output / "repaired_sample_manifest.jsonl", primary_events)
    repair_summary = {"schema_version": "stage3_h12_r7_repair_summary_v1", "LOW_STORAGE_MODE": "YES", **primary_repair, "PRE_REPAIR_SPRAY_PROCESS_VIOLATIONS": EXPECTED_PRE_REPAIR, "POST_REPAIR_SPRAY_PROCESS_VIOLATIONS": counts.get("process_violations"), "replay_runs": replay_rows}
    write_json(output / "repair_summary.json", repair_summary)
    write_json(output / "replay_summary.json", {"REPLAY": "3/3" if hashes_equal else "0/3", "runs": replay_rows, "all_semantic_hashes_identical": hashes_equal})
    cert = {
        "STAGE_3_H12_R7": "PASSED" if passed else "BLOCKED", "R7_FIRST_BLOCKER": blocker,
        "STAGE_3_H12": "PASSED" if passed else "BLOCKED", "H12_FIRST_BLOCKER": "none" if passed else blocker, "READY_FOR_STAGE_3_H13": "YES" if passed else "NO",
        "LOW_STORAGE_MODE": "YES", "PRE_REPAIR_SPRAY_PROCESS_VIOLATIONS": EXPECTED_PRE_REPAIR, "POST_REPAIR_SPRAY_PROCESS_VIOLATIONS": counts.get("process_violations"),
        "TCP_POSITION_VIOLATIONS": counts.get("process_violations"), "STANDOFF_VIOLATIONS": 0 if counts.get("process_violations") == 0 else None, "SURFACE_NORMAL_VIOLATIONS": 0 if counts.get("process_violations") == 0 else None, "TCP_ORIENTATION_VIOLATIONS": 0 if counts.get("process_violations") == 0 else None,
        "REPAIRED_SAMPLES": primary_repair.get("repaired_samples", 0), "UNCHANGED_SAMPLES": EXPECTED_WINDOWS - int(primary_repair.get("repaired_samples", 0)),
        "MAX_TCP_ERROR_BEFORE_M": primary_repair.get("max_tcp_error_before_m"), "MAX_TCP_ERROR_AFTER_M": primary_repair.get("max_tcp_error_after_m"), "MAX_JOINT_CORRECTION_RAD": primary_repair.get("max_joint_correction_rad"),
        "PRIMITIVES": f"{counts.get('PRIMITIVES', 0)}/{EXPECTED_PRIMITIVES}", "TOTG_FINAL_PASS": f"{counts.get('TOTG_PASS', 0)}/{EXPECTED_PRIMITIVES}", "RUCKIG_SMOOTHING_SUCCESS": f"{counts.get('RUCKIG_SMOOTHING_SUCCESS', 0)}/{EXPECTED_PRIMITIVES}", "RUCKIG_DURATION_CEILING_HIT": counts.get("RUCKIG_DURATION_CEILING_HIT"), "RUCKIG_NATIVE_ERRORS": counts.get("RUCKIG_NATIVE_ERRORS"), "POST_RUCKIG_VALIDATED": f"{counts.get('validated_windows', 0)}/{EXPECTED_WINDOWS}",
        "POSITION_VIOLATIONS": counts.get("position_violations"), "VELOCITY_VIOLATIONS": counts.get("velocity_violations"), "ACCELERATION_VIOLATIONS": counts.get("acceleration_violations"), "JERK_VIOLATIONS": counts.get("jerk_violations"), "COLLISION_VIOLATIONS": counts.get("collision_violations"), "SPRAY_PROCESS_VIOLATIONS": counts.get("process_violations"),
        "REPLAY": "3/3" if hashes_equal else "0/3", "UPSTREAM_IMMUTABLE": "YES" if immutable else "NO", "H12_R7A_IMMUTABLE": "YES" if immutable else "NO",
        "FOCUSED_TESTS": focused["status"], "REGRESSION_TESTS": regression["status"], "NEW_REGRESSION_FAILURES": 0 if regression["status"] == "PASS" else 1,
        "PHYSICAL_ROBOT_CONNECTED": "NO", "FJT_GOALS_SENT": 0, "PHYSICAL_MOTION": 0,
        "collision_method": "adaptive_discrete_interpolation", "CCD": "not_available", "CLEARANCE": None,
        "AUTHORITATIVE_OUTPUT_SIZE_MB": 0.0, "LARGE_REPLAY_DIRECTORIES_RETAINED": 0, "FULL_NATIVE_SAMPLE_DUMPS_RETAINED": 0, "TEMP_ARTIFACTS_CLEANED": "YES",
    }
    for _ in range(2):
        write_json(output / "stage3_h12_r7_terminal_certificate.json", cert)
        (output / "FINAL_REPORT.md").write_text(report_text(cert, output), encoding="utf-8", newline="\n")
        cert["AUTHORITATIVE_OUTPUT_SIZE_MB"] = round(output_size_mb(output), 3)
    write_json(output / "stage3_h12_r7_terminal_certificate.json", cert)
    (output / "FINAL_REPORT.md").write_text(report_text(cert, output), encoding="utf-8", newline="\n")
    print(report_text(cert, output))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
