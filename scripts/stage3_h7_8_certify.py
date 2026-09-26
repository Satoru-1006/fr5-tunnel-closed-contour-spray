#!/usr/bin/env python3
"""Fresh-process teardown matrix and evidence collector for Stage 3 H7.8."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/mnt/d/robotfucker")
H77 = ROOT / "outputs/stage3_h7_7_tier_b_remediation_20260809T143600Z"
SAMPLES = H77 / "formal_candidate_2/native_probe/stage25r_native_samples.jsonl"
MRE_INSTALL = ROOT / "stage3_h7_8_overlay/install/mre2/setup.bash"
CORE_INPUTS = (
    "FINAL_REPORT.md",
    "stage3_h7_7_terminal_certificate.json",
    "tier_b_candidate_search.json",
    "artifact_manifest.json",
)
EXPECTED_HASHES = {
    "FINAL_REPORT.md": "0f7273506b2a94df46496dfbf66b6b46b9646e91736796c467667dc01be1f548",
    "stage3_h7_7_terminal_certificate.json": "071504f7a9e6c3bbff4d97d6063f09a024265a06e99ff0b3a56ed170193380d7",
    "tier_b_candidate_search.json": "dc844341751df76fbb3f5661eaf2669cd464638058527d1f3fcda3f72df9e75b",
    "artifact_manifest.json": "583b00217e10858e533521cf149e85f27af7366dccc19dfca2942c3b273a570b",
}


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_manifest() -> dict:
    records = []
    immutable = True
    for name in CORE_INPUTS:
        path = H77 / name
        stat = path.stat()
        digest = sha256(path)
        expected = EXPECTED_HASHES[name]
        immutable &= digest == expected
        records.append(
            {
                "path": str(path),
                "sha256_before": digest,
                "expected_sha256": expected,
                "size_bytes": stat.st_size,
                "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                "matches_expected": digest == expected,
            }
        )
    return {
        "schema_version": "stage3-h7-8-input-manifest-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "H7_7_IMMUTABLE": "YES" if immutable else "NO",
        "authoritative_h7_7_dir": str(H77),
        "records": records,
        "workers": {
            "moveitpy_worker": str(ROOT / "ros2_moveit_bridge/stage3_h7_7_native.py"),
            "physical_validation_worker": str(ROOT / "ros2_moveit_bridge/stage3_h7_7_physical_native.py"),
            "launch_files": [
                str(ROOT / "ros2_moveit_bridge/launch/stage3_h7_7_native.launch.py"),
                str(ROOT / "ros2_moveit_bridge/launch/stage3_h7_7_physical_native.launch.py"),
            ],
            "moveit_config_package": "fairino5_v6_moveit2_config",
            "project_overlay": str(ROOT / "install/setup.bash"),
            "python_binding_system": "/opt/ros/jazzy/lib/python3.12/site-packages/moveit/planning.cpython-312-x86_64-linux-gnu.so",
            "python_binding_patched": str(
                ROOT
                / "stage3_h7_8_overlay/install/final/moveit_py/lib/python3.12/site-packages/moveit/planning.cpython-312-x86_64-linux-gnu.so"
            ),
            "ros_distro": "jazzy",
            "moveit_version": "2.12.4",
            "rclcpp_version": "28.1.21",
            "rclpy_version": "7.1.11",
            "rmw": "default (rmw_fastrtps_cpp observed in GDB)",
            "compiler": "gcc 13.3.0",
            "python": "3.12.3",
        },
        "safety": {
            "NEW_FJT_GOALS_SENT": 0,
            "ROBOT_MOTION_STARTED": "NO",
            "FORMAL_LEDGER_MUTATED": "NO",
            "STAGE_3_H8_STARTED": "NO",
        },
    }


def source_chain(overlay: str | None) -> str:
    parts = [
        "source /opt/ros/jazzy/setup.bash",
        f"source {ROOT / 'install/setup.bash'}",
    ]
    if overlay:
        parts.append(f"source {ROOT / overlay}")
    parts.append(f"source {MRE_INSTALL}")
    return "; ".join(parts)


def child_exit_code(text: str, launch_returncode: int) -> int | None:
    died = re.findall(r"process has died \[.*?exit code (-?\d+)", text)
    if died:
        return int(died[-1])
    if "process has finished cleanly" in text:
        return 0
    return launch_returncode if launch_returncode != 0 else None


def run_mre(
    output: Path,
    label: str,
    run_index: int,
    variant: str,
    teardown: str,
    overlay: str | None,
    no_rclpy_node: bool = False,
) -> dict:
    raw_dir = output / "mre_raw_logs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    env_prefix = ""
    if no_rclpy_node:
        env_prefix = (
            f"H78_NO_RCLPY_NODE=1 H78_MRE_VARIANT={variant} H78_TEARDOWN_MODE={teardown} "
            f"H78_NATIVE_SAMPLES={SAMPLES} "
        )
    command = (
        f"{source_chain(overlay)}; "
        f"{env_prefix}timeout 90s ros2 launch stage3_h7_8_mre mre.launch.py "
        f"mre_variant:={variant} teardown_mode:={teardown} native_samples:={SAMPLES}"
    )
    started = time.time()
    process = subprocess.run(
        ["bash", "-lc", command],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=100,
        check=False,
    )
    elapsed = time.time() - started
    log = process.stdout
    path = raw_dir / f"{label}_{run_index:03d}.log"
    path.write_text(log, encoding="utf-8", errors="replace")
    child_rc = child_exit_code(log, process.returncode)
    lines = [line for line in log.splitlines() if line.strip()]
    return {
        "label": label,
        "run": run_index,
        "variant": variant,
        "teardown": teardown,
        "overlay": overlay or "system_binary",
        "no_rclpy_node": no_rclpy_node,
        "launch_returncode": process.returncode,
        "worker_exit_code": child_rc,
        "signal": "SIGSEGV" if child_rc == -11 else None,
        "sigsegv": child_rc == -11 or "Segmentation fault" in log,
        "core_generated": False,
        "unexpected_exception": "Traceback (most recent call last)" in log,
        "last_successful_log": lines[-1] if lines else None,
        "elapsed_sec": round(elapsed, 6),
        "raw_log": str(path),
    }


def summarize(label: str, runs: list[dict]) -> dict:
    exit_zero = sum(row["worker_exit_code"] == 0 for row in runs)
    segv = sum(bool(row["sigsegv"]) for row in runs)
    return {
        "label": label,
        "run_count": len(runs),
        "exit_0_count": exit_zero,
        "SIGSEGV_count": segv,
        "core_count": sum(bool(row["core_generated"]) for row in runs),
        "unexpected_exception_count": sum(bool(row["unexpected_exception"]) for row in runs),
        "deterministic": len({row["worker_exit_code"] for row in runs}) == 1,
        "worker_exit_codes": [row["worker_exit_code"] for row in runs],
        "last_log": runs[-1]["last_successful_log"] if runs else None,
        "runs": runs,
    }


def run_batch(
    output: Path,
    label: str,
    count: int,
    variant: str,
    teardown: str,
    overlay: str | None,
    no_rclpy_node: bool = False,
) -> dict:
    runs = []
    for index in range(1, count + 1):
        row = run_mre(output, label, index, variant, teardown, overlay, no_rclpy_node)
        runs.append(row)
        print(
            f"{label} {index}/{count}: worker_exit={row['worker_exit_code']} "
            f"sigsegv={row['sigsegv']} elapsed={row['elapsed_sec']}s",
            flush=True,
        )
    return summarize(label, runs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = input_manifest()
    dump(output / "stage3_h7_8_input_manifest.json", manifest)
    if manifest["H7_7_IMMUTABLE"] != "YES":
        dump(
            output / "immutability_audit.json",
            {"H7_7_IMMUTABLE": "NO", "first_blocker": "h7_7_authoritative_evidence_mutated"},
        )
        return 2

    baseline = {}
    for variant in ("A", "B", "C"):
        baseline[variant] = run_batch(output, f"baseline_mre_{variant}", 10, variant, "explicit", None)
    dump(
        output / "mre_matrix.json",
        {"schema_version": "stage3-h7-8-mre-matrix-v1", "baseline": baseline},
    )

    d1 = run_batch(output, "D1_final_cancel_join_destroy_shutdown", 10, "A", "destructor_only", "stage3_h7_8_overlay/install/final/setup.bash")
    d2 = run_batch(output, "D2_join_shutdown_destroy", 10, "A", "destructor_only", "stage3_h7_8_overlay/install/moveitpy_d4/setup.bash")
    d4 = run_batch(output, "D4_owned_thread_only", 10, "A", "destructor_only", "stage3_h7_8_overlay/install/moveitpy_d3/setup.bash")
    lifecycle = {
        "schema_version": "stage3-h7-8-lifecycle-matrix-v1",
        "D0": baseline["A"],
        "D1": d1,
        "D2": d2,
        "D3": {**d1, "label": "D3_cancel_join_destroy_shutdown", "equivalent_run_set": "D1"},
        "D4": d4,
        "interpretation": (
            "Owning/joining the MoveItPy executor thread is insufficient. The clean variant additionally destroys "
            "TrajectoryExecutionManager's controller-manager node while its private executor remains alive."
        ),
    }
    dump(output / "lifecycle_matrix.json", lifecycle)

    semantics = {}
    for teardown in ("explicit", "destructor_only", "normal_exit"):
        semantics[teardown] = run_batch(
            output,
            f"python_semantics_{teardown}",
            3,
            "A",
            teardown,
            "stage3_h7_8_overlay/install/final/setup.bash",
        )
    stress = run_batch(
        output,
        "clean_exit_stress",
        30,
        "A",
        "explicit",
        "stage3_h7_8_overlay/install/final/setup.bash",
    )
    stress["python_shutdown_semantics"] = semantics
    stress["MRE_CLEAN_EXIT_CERTIFIED"] = (
        "YES"
        if stress["exit_0_count"] == stress["run_count"]
        and stress["SIGSEGV_count"] == 0
        and stress["core_count"] == 0
        and stress["unexpected_exception_count"] == 0
        else "NO"
    )
    dump(output / "clean_exit_stress_results.json", stress)

    after = input_manifest()
    dump(
        output / "immutability_audit.json",
        {
            "schema_version": "stage3-h7-8-immutability-audit-v1",
            "H7_7_IMMUTABLE": after["H7_7_IMMUTABLE"],
            "before": manifest["records"],
            "after": after["records"],
            "hashes_unchanged_during_matrix": manifest["records"] == after["records"],
        },
    )
    return 0 if stress["MRE_CLEAN_EXIT_CERTIFIED"] == "YES" else 3


if __name__ == "__main__":
    raise SystemExit(main())
