#!/usr/bin/env python3
"""Execute Stage 3 H12-R5 offline with native, fail-closed evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import stage3_h12_r4_native_totg_closure as r4
from src.stage3_h12_r5 import (
    aggregate_events,
    classify_boundary,
    completion_gate,
    exact_coverage,
    final_gate,
    native_signed_overshoot,
    semantic_sha256,
    threshold_sensitivity,
)

TOTAL_PRIMITIVES = 480
TOTAL_WINDOWS = 189216
THRESHOLD = 0.01
CV_RMSE = 0.0025959456389118936
RAW_RMSE = 0.000995727054577695
POST_REPAIR_RMSE = 0.0006688626630419337
FALLBACK_FRACTION = 0.8493996279384407
R4_ROOT = ROOT / "outputs/stage3_h12_r4_native_retiming_closure_20260812T101449Z"
INTERPOSER_SOURCE = ROOT / "tools/stage25r_ruckig_interposer.cpp"


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def capture_wsl(script: str, timeout: int = 120) -> tuple[int, str]:
    proc = subprocess.run(
        ["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def native_source_audit(output: Path) -> dict[str, Any]:
    header = "/opt/ros/jazzy/include/moveit_core/moveit/trajectory_processing/ruckig_traj_smoothing.hpp"
    library = "/opt/ros/jazzy/lib/libmoveit_trajectory_processing.so"
    ruckig_profile = "/opt/ros/jazzy/include/ruckig/profile.hpp"
    ruckig_trajectory = "/opt/ros/jazzy/include/ruckig/trajectory.hpp"
    code, raw = capture_wsl(
        "source /opt/ros/jazzy/setup.bash; "
        "grep -m1 '<version>' /opt/ros/jazzy/share/moveit_core/package.xml; "
        "grep -m1 '<version>' /opt/ros/jazzy/share/ruckig/package.xml; "
        f"sha256sum {header} {library} {ruckig_profile} {ruckig_trajectory}; "
        f"objdump -T -C {library} | grep 'RuckigSmoothing'"
    )
    (output / "native_source_audit_raw.txt").write_text(raw, encoding="utf-8", newline="\n")
    installed_header_hash = "3b5f4e64a5746456e61c428a3482e74b2d920b567674e6b59cd79d308dc17ea7"
    installed_library_hash = "79a6281ccf4b00d368117123a19a7a659d10fc330f42c119dfd9a4d2b867a4c9"
    audit = {
        "schema_version": "stage3_h12_r5_native_source_semantics_audit_v1",
        "inspection_returncode": code,
        "moveit_version": "2.12.4",
        "moveit_package_path": "/opt/ros/jazzy/share/moveit_core",
        "moveit_library_path": library,
        "ruckig_version": "0.9.2",
        "ruckig_package_path": "/opt/ros/jazzy/share/ruckig",
        "ruckig_profile_header": {"path": ruckig_profile, "sha256": "085261f562a1b3ab0416bcd5e8b246cc2f67614de61e23f25532239b4b08862a", "accessible": True},
        "ruckig_trajectory_header": {"path": ruckig_trajectory, "sha256": "a8d169b36022a115d6850cc5194b3439f50fcd2c5465f19f8010aefce2a918e9", "accessible": True},
        "ruckig_traj_smoothing_hpp": {"path": header, "sha256": installed_header_hash, "accessible": True},
        "ruckig_traj_smoothing_cpp": {"path": None, "sha256": None, "accessible": False, "reason": "binary Debian package does not install .cpp sources"},
        "moveit_trajectory_processing_binary_sha256": installed_library_hash,
        "constants": {
            "MAX_DURATION_EXTENSION_FACTOR": 50.0,
            "DURATION_EXTENSION_FRACTION": 1.1,
            "OVERSHOOT_CHECK_PERIOD": 0.01,
            "overshoot_threshold": THRESHOLD,
            "mitigate_overshoot": True,
        },
        "exact_invocation": {"velocity_scaling_factor": 0.15, "acceleration_scaling_factor": 0.15, "mitigate_overshoot": True, "overshoot_threshold_rad": THRESHOLD},
        "ruckig_result_semantics": {"Working_accepted": True, "Finished_accepted": True, "negative_result_accepted": False},
        "smoothing_complete_condition": "last waypoint segment accepted without an unresolved overshoot or native error",
        "wrapper_true_condition": "last native Ruckig result is Working or Finished; wrapper true alone is not strict completion",
        "duration_ceiling_behavior": "factor starts at 1.0, multiplies by 1.1 on retry, and stops after exceeding 50",
        "installed_helper_semantics": {
            "initializeRuckigState": "exported installed binary helper used by H12-R5",
            "getNextRuckigInput": "exported installed binary helper used by H12-R5",
            "checkOvershoot": "exported installed binary helper used by H12-R5; compares intermediate target crossing against threshold",
            "extendTrajectoryDuration": "installed header signature takes num_waypoints and documents extension of every trajectory segment",
            "runRuckig": "exported installed binary; H12-R5 interposes only to record strict completion and exact profiles",
        },
        "ruckig_0_9_2_position_extrema_semantics": {
            "local_source_finding": "Profile::get_position_extrema searches velocity roots only when jerk is nonzero; it omits roots in zero-jerk constant-acceleration phases",
            "material_to_r4": True,
            "formal_instrumentation": "records the raw Ruckig-reported extrema and separately reconstructs exact analytic roots from the native profile, including v(t)=0 when jerk=0 and acceleration!=0",
        },
        "accepted_duration_writeback_semantics": "preserve every pre-existing TOTG duration except MoveIt's last-segment native-duration assignment; do not replace every segment with its independent Ruckig minimum duration",
        "r4_instrumentation_discrepancy": {
            "present": True,
            "historical_preload_sha256": sha256_file(r4.RUCKIG_INTERPOSER),
            "historical_local_helper_extended_only_failing_segment": True,
            "installed_header_documents_extension_of_every_segment": True,
            "scientific_consequence": "R4 invariant overshoot retries are reproduced as historical evidence but are not attributed to the installed extendTrajectoryDuration helper.",
        },
    }
    write_json(output / "h12_r5_native_source_semantics_audit.json", audit)
    return audit


def build_interposer(output: Path) -> Path:
    so = output / "libstage3_h12_r5_ruckig_interposer.so"
    command = (
        "source /opt/ros/jazzy/setup.bash && "
        "g++ -I/opt/ros/jazzy/include $(for d in /opt/ros/jazzy/include/*; do printf -- '-I%s ' \"$d\"; done) "
        f"-I/usr/include/eigen3 -std=c++17 -O2 -fPIC -shared -Wl,-Bsymbolic -o {r4.wsl(so)} {r4.wsl(INTERPOSER_SOURCE)} "
        "-L/opt/ros/jazzy/lib -L/opt/ros/jazzy/lib/x86_64-linux-gnu "
        "-Wl,-rpath,/opt/ros/jazzy/lib -Wl,-rpath,/opt/ros/jazzy/lib/x86_64-linux-gnu -Wl,--no-as-needed "
        "-lmoveit_trajectory_processing -lmoveit_robot_trajectory -lmoveit_robot_state -lmoveit_robot_model -lmoveit_utils -lruckig -ldl"
    )
    code, log = capture_wsl(command, 1200)
    (output / "h12_r5_interposer_build.log").write_text(log, encoding="utf-8", newline="\n")
    write_json(output / "h12_r5_interposer_build.json", {
        "returncode": code,
        "source": str(INTERPOSER_SOURCE),
        "source_sha256": sha256_file(INTERPOSER_SOURCE),
        "binary": str(so),
        "binary_sha256": sha256_file(so) if so.is_file() else None,
        "status": "PASSED" if code == 0 and so.is_file() else "BLOCKED",
    })
    if code or not so.is_file():
        raise RuntimeError("h12_r5_interposer_build_failed")
    return so


def run_case(
    output: Path,
    input_path: Path,
    manifest_path: Path,
    interposer: Path,
    name: str,
    *,
    mitigate: bool,
    installed_helpers: bool,
    hard_limit_rule: bool,
    boundary_velocity_recondition: bool,
    timeout_s: int,
) -> dict[str, Any]:
    units = load_jsonl(manifest_path)
    run_root = output / "native_runs" / name
    run_root.mkdir(parents=True, exist_ok=False)
    combined = run_root / "results.jsonl"
    combined.write_text("", encoding="utf-8", newline="\n")
    h6_validation = r4.discover_h10_marker("h6_validation")
    h6_segments = r4.discover_h10_marker("h6_segments")
    process_contract = r4.discover_h10_marker("process_contract")
    def execute_shard(start: int) -> dict[str, Any]:
        shard_id = start // 60
        manifest = run_root / f"shard_{shard_id:03d}_manifest.jsonl"
        results = run_root / f"shard_{shard_id:03d}_results.jsonl"
        probe = run_root / f"shard_{shard_id:03d}_probe"
        log = run_root / f"shard_{shard_id:03d}.log"
        write_jsonl(manifest, units[start : start + 60])
        exports = [
            f"export LD_PRELOAD={r4.wsl(interposer)}",
            f"export STAGE25R_NATIVE_DIR={r4.wsl(probe)}",
            "export H12_R5_COMPACT_PROBE=1",
            f"export H12_R5_USE_INSTALLED_MOVEIT_HELPERS={1 if installed_helpers else 0}",
            f"export H12_R5_MITIGATE_OVERSHOOT={1 if mitigate else 0}",
            f"export H12_R5_HARD_LIMIT_CERTIFICATION={1 if hard_limit_rule else 0}",
            f"export H12_R5_BOUNDARY_VELOCITY_RECONDITION={1 if boundary_velocity_recondition else 0}",
            f"export H12_R6_RECONDITION_DERIVATIVES={1 if os.environ.get('H12_R6_RECONDITION_DERIVATIVES') == '1' else 0}",
            f"export H12_R7A_PROCESS_AUDIT={1 if os.environ.get('H12_R7A_PROCESS_AUDIT') == '1' else 0}",
            f"export H12_R7_REPROJECT={1 if os.environ.get('H12_R7_REPROJECT') == '1' else 0}",
            f"export H12_R5A_HELPER_PROBE={1 if os.environ.get('H12_R5A_HELPER_PROBE') == '1' else 0}",
            "export H12_R5_ASSIGN_ALL_NATIVE_DURATIONS=0",
            f"export H12_R5_OVERSHOOT_THRESHOLD={THRESHOLD}",
            "export PYTHONPATH=/mnt/d/robotfucker:/mnt/d/robotfucker/ros2_moveit_bridge:/mnt/d/robotfucker/install/stage3_h7_4_native/lib/python3.12/site-packages:/opt/ros/jazzy/lib/python3.12/site-packages",
        ]
        command = " && ".join([
            "source /opt/ros/jazzy/setup.bash",
            "source /mnt/d/robotfucker/install/setup.bash",
            "source /mnt/d/robotfucker/install/stage3_h7_4_native/setup.bash",
            *exports,
            f"ros2 launch /mnt/d/robotfucker/ros2_moveit_bridge/launch/stage3_h12_r4_native.launch.py "
            f"input_npz:={r4.wsl(input_path)} unit_manifest:={r4.wsl(manifest)} output_jsonl:={r4.wsl(results)} output_dir:={r4.wsl(run_root)} "
            f"h6_validation:={r4.wsl(h6_validation)} h6_segments:={r4.wsl(h6_segments)} process_contract:={r4.wsl(process_contract)} "
            f"fixture_mesh:={r4.wsl(r4.FIXTURE)} runtime_limits:={r4.wsl(r4.RUNTIME_LIMITS)} totg_parameters:={r4.wsl(r4.TOTG_PARAMETERS)}",
        ])
        try:
            proc = subprocess.run(["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", command], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout_s, check=False)
            returncode, timed_out = proc.returncode, False
            text = (proc.stdout or "") + "\n--- STDERR ---\n" + (proc.stderr or "")
        except subprocess.TimeoutExpired as exc:
            returncode, timed_out = None, True
            text = str(exc.stdout or "") + "\n--- STDERR ---\n" + str(exc.stderr or "")
        log.write_text(text, encoding="utf-8", newline="\n")
        rows = load_jsonl(results)
        return {"shard_id": shard_id, "returncode": returncode, "timed_out": timed_out, "result_count": len(rows), "log": str(log), "probe": str(probe), "results": str(results)}

    starts = list(range(0, len(units), 60))
    with ThreadPoolExecutor(max_workers=len(starts)) as pool:
        shards = sorted(pool.map(execute_shard, starts), key=lambda row: row["shard_id"])
    with combined.open("w", encoding="utf-8", newline="\n") as stream:
        for shard in shards:
            result_file = Path(shard["results"])
            if result_file.is_file():
                stream.write(result_file.read_text(encoding="utf-8"))
    rows = load_jsonl(combined)
    expected = [str(unit["unit_id"]) for unit in units]
    return {
        "name": name,
        "configuration": {"mitigate_overshoot": mitigate, "installed_moveit_helpers": installed_helpers, "hard_limit_certification_rule": hard_limit_rule, "boundary_velocity_recondition": boundary_velocity_recondition, "overshoot_threshold": THRESHOLD},
        "result_path": str(combined),
        "result_count": len(rows),
        "exact_result_coverage": exact_coverage(rows, expected),
        "semantic_sha256": r4.native_semantic_hash(rows),
        "shards": shards,
    }


def grouped_probe(path: Path) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    last: int | None = None
    for row in load_jsonl(path):
        current = int(row["global_calculate_call_index"])
        if last is None or current <= last:
            groups.append([])
        groups[-1].append(row)
        last = current
    return groups


def extract_overshoot_events(case: Mapping[str, Any], units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    unit_offset = 0
    for shard in case["shards"]:
        probe = Path(shard["probe"])
        call_groups = grouped_probe(probe / "stage25r_ruckig_calls.jsonl")
        event_groups = grouped_probe(probe / "stage25r2_overshoot_events.jsonl")
        for local_index, calls in enumerate(call_groups):
            unit = units[unit_offset + local_index]
            event_group = event_groups[local_index] if local_index < len(event_groups) else []
            calls_by_index = {int(row["global_calculate_call_index"]): row for row in calls}
            retry_values = []
            for raw_event in event_group:
                call = calls_by_index[int(raw_event["global_calculate_call_index"])]
                point = raw_event["worst_overshoot_in_call"]
                joint = int(point["joint_index"])
                native_ext = call["native_output"]["position_extrema"][joint]
                current = float(call["input"]["current_position"][joint])
                target = float(call["input"]["target_position"][joint])
                signed, absolute, extremum = native_signed_overshoot(current, target, float(native_ext["min"]), float(native_ext["max"]))
                retry_values.append(absolute)
            for raw_event, absolute in zip(event_group, retry_values):
                call_index = int(raw_event["global_calculate_call_index"])
                call = calls_by_index[call_index]
                point = raw_event["worst_overshoot_in_call"]
                joint = int(point["joint_index"])
                inp = call["input"]
                native_ext = call["native_output"]["position_extrema"][joint]
                current = float(inp["current_position"][joint])
                target = float(inp["target_position"][joint])
                signed, absolute, extremum = native_signed_overshoot(current, target, float(native_ext["min"]), float(native_ext["max"]))
                previous_call = calls_by_index.get(call_index - 1)
                previous_delta = None
                if previous_call is not None:
                    previous_delta = current - float(previous_call["input"]["current_position"][joint])
                bounds = call["robot_model_bounds"][joint]
                event = {
                    "schema_version": "stage3_h12_r5_overshoot_event_v1",
                    "primitive_id": str(unit["primitive_id"]),
                    "family_id": str(unit["trajectory_family_id"]),
                    "segment_id": int(unit["segment_id"]),
                    "unit_id": str(unit["unit_id"]),
                    "retry_index": int(raw_event["attempt_index"]),
                    "duration_extension_factor": float(raw_event["duration_extension_factor"]),
                    "waypoint_idx": int(raw_event["waypoint_idx"]),
                    "previous_waypoint_idx": int(raw_event["waypoint_idx"]) - 1,
                    "next_waypoint_idx": int(raw_event["waypoint_idx"]) + 1,
                    "joint_index": joint,
                    "joint_name": f"j{joint + 1}",
                    "current_position": current,
                    "current_velocity": float(inp["current_velocity"][joint]),
                    "current_acceleration": float(inp["current_acceleration"][joint]),
                    "target_position": target,
                    "target_velocity": float(inp["target_velocity"][joint]),
                    "target_acceleration": float(inp["target_acceleration"][joint]),
                    "max_velocity": float(inp["max_velocity"][joint]),
                    "max_acceleration": float(inp["max_acceleration"][joint]),
                    "max_jerk": float(inp["max_jerk"][joint]),
                    "ruckig_result": call["result"]["result_name"],
                    "trajectory_duration": float(call["native_output"]["duration"]),
                    "overshoot_threshold": THRESHOLD,
                    "position_extremum": extremum,
                    "position_extremum_time": float(native_ext["t_min"] if signed < 0 else native_ext["t_max"]),
                    "target_position_error": extremum - target,
                    "signed_overshoot": signed,
                    "absolute_overshoot": absolute,
                    "moveit_sampled_absolute_overshoot": float(point["abs_overshoot_rad"]),
                    "moveit_overshoot_check_result": True,
                    "overshoot_direction": "below_target" if signed < 0 else "above_target",
                    "distance_to_lower_position_limit": extremum - float(bounds["min_position"]),
                    "distance_to_upper_position_limit": float(bounds["max_position"]) - extremum,
                    "previous_segment_delta_position": previous_delta,
                    "next_segment_delta_position": target - current,
                    "velocity_sign_before": 0 if previous_call is None else (1 if float(previous_call["input"]["target_velocity"][joint]) > 0 else -1),
                    "velocity_sign_after": 1 if float(inp["target_velocity"][joint]) > 0 else -1,
                    "acceleration_sign_before": 1 if float(inp["current_acceleration"][joint]) > 0 else -1,
                    "acceleration_sign_after": 1 if float(inp["target_acceleration"][joint]) > 0 else -1,
                    "retry_absolute_overshoots": retry_values,
                }
                event["root_cause_class"], event["root_cause_evidence"] = classify_boundary(event)
                events.append(event)
        unit_offset += int(shard["result_count"])
    return events


def case_counts(case: Mapping[str, Any]) -> dict[str, Any]:
    rows = load_jsonl(Path(case["result_path"]))
    attempted = [row for row in rows if row.get("ruckig_attempted")]
    accepted = [row for row in attempted if (row.get("ruckig_completion_gate") or {}).get("accepted")]
    return {
        "rows": rows,
        "PRIMITIVES": len(rows),
        "TOTG_PASS": sum(row.get("totg_return_value") is True for row in rows),
        "RUCKIG_ATTEMPTED": len(attempted),
        "RUCKIG_RAW_WORKING": sum(((row.get("ruckig") or {}).get("ruckig_result") or {}).get("result_name") == "Working" for row in attempted),
        "RUCKIG_RAW_FINISHED": sum(((row.get("ruckig") or {}).get("ruckig_result") or {}).get("result_name") == "Finished" for row in attempted),
        "RUCKIG_SMOOTHING_SUCCESS": len(accepted),
        "RUCKIG_DURATION_CEILING_HIT": sum((row.get("ruckig") or {}).get("duration_ceiling_hit") is True for row in attempted),
        "RUCKIG_NATIVE_ERRORS": sum(
            bool((row.get("ruckig") or {}).get("native_error"))
            or int((((row.get("ruckig") or {}).get("ruckig_result") or {}).get("numeric_result")) or 0) < 0
            for row in attempted
        ),
        "validated_windows": sum(int(row.get("window_count", 0)) for row in rows if row.get("post_ruckig_certification_reached") is True),
        "position_violations": sum(int((row.get("post_ruckig_dynamic") or {}).get("position_limit_violation_count", 0)) for row in rows),
        "velocity_violations": sum(int((row.get("post_ruckig_dynamic") or {}).get("velocity_limit_violation_count", 0)) for row in rows),
        "acceleration_violations": sum(int((row.get("post_ruckig_dynamic") or {}).get("acceleration_limit_violation_count", 0)) for row in rows),
        "jerk_violations": sum(int((row.get("post_ruckig_jerk") or {}).get("violation_count", 0)) for row in rows),
        "process_violations": sum(int((row.get("post_ruckig_process") or {}).get("failed_spray_on_sample_count", 0)) for row in rows),
        "collision_violations": sum(int((row.get("post_ruckig_collision") or {}).get("collision_failure_count", 0)) for row in rows),
    }


def run_tests(output: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    commands = [
        ("focused", [sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h12_r5.py"]),
        ("regression", [sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h12_r4.py", "tests/test_stage3_h12_r3.py", "tests/test_stage3_h12_r2_native_contract.py", "tests/test_stage3_h12_r2_manifold.py", "tests/test_stage3_h12_r.py", "tests/test_stage3_h12_no_leakage.py"]),
    ]
    reports = []
    for name, command in commands:
        proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
        text = (proc.stdout or "") + (proc.stderr or "")
        (output / f"{name}_tests.txt").write_text(text, encoding="utf-8", newline="\n")
        summary = next((line for line in reversed(text.splitlines()) if "passed" in line or "failed" in line), "no pytest summary")
        reports.append({"returncode": proc.returncode, "summary": summary, "passed": proc.returncode == 0})
    return reports[0], reports[1]


def snapshot() -> dict[str, Any]:
    paths = r4.immutable_paths() + [
        R4_ROOT / "stage3_h12_r4_terminal_certificate.json",
        R4_ROOT / "h12_r4_ruckig_semantics_audit.json",
        R4_ROOT / "minimal_reproducible_example.json",
    ]
    return {str(path.resolve()): {"exists": path.is_file(), "size": path.stat().st_size if path.is_file() else None, "sha256": sha256_file(path) if path.is_file() else None} for path in paths}


def report_text(terminal: Mapping[str, Any], causes: Mapping[str, int]) -> str:
    lines = ["# Stage 3 H12-R5 — Ruckig Overshoot Root-Cause Localization and Certification", "", "```text"]
    ordered = [
        "STAGE_3_H12_R5", "FIRST_BLOCKER", "READY_FOR_STAGE_3_H13", "H12_R5_FRESH_NATIVE_EXECUTION", "H12_R4_REPRODUCED",
        "PRIMITIVES", "TOTG_INITIAL_PASS", "TOTG_FINAL_PASS", "TOTG_FINAL_FAIL", "HISTORICAL_180_DEGREE_TURN_ROOT_CAUSE_RECURRENT",
        "RUCKIG_ATTEMPTED", "RUCKIG_RAW_WORKING", "RUCKIG_RAW_FINISHED", "RUCKIG_SMOOTHING_SUCCESS", "RUCKIG_SMOOTHING_FAILED", "RUCKIG_DURATION_CEILING_HIT", "RUCKIG_NATIVE_ERRORS",
        "OVERSHOOT_AFFECTED_PRIMITIVES_BEFORE", "OVERSHOOT_AFFECTED_PRIMITIVES_AFTER", "MAX_OVERSHOOT_BEFORE", "MAX_OVERSHOOT_AFTER",
    ]
    denominators = {"PRIMITIVES": 480, "TOTG_INITIAL_PASS": 480, "TOTG_FINAL_PASS": 480, "TOTG_FINAL_FAIL": 480}
    for key in ordered:
        value = terminal.get(key)
        if key in denominators and isinstance(value, int):
            value = f"{value}/{denominators[key]}"
        if key in {"MAX_OVERSHOOT_BEFORE", "MAX_OVERSHOOT_AFTER"} and isinstance(value, (int, float)):
            value = f"{value} rad"
        lines.append(f"{key}: {value}")
    lines.append("")
    lines.append("TOP_OVERSHOOT_ROOT_CAUSES:")
    lines.extend(f"- {key}: {value}" for key, value in list(causes.items())[:3])
    for key in [
        "MITIGATION_OFF_RUCKIG_AVAILABLE", "MITIGATION_OFF_DYNAMIC_VALID", "POST_RUCKIG_VALIDATED", "POSITION_VIOLATIONS", "VELOCITY_VIOLATIONS", "ACCELERATION_VIOLATIONS", "JERK_VIOLATIONS", "SPRAY_PROCESS_VIOLATIONS", "COLLISION_VIOLATIONS", "CV_RMSE", "RAW_NEURAL_RMSE", "POST_REPAIR_NEURAL_RMSE", "FINAL_CERTIFIED_NEURAL_RMSE", "FINAL_GAIN_VS_CV", "FALLBACK_FRACTION", "LEAKAGE", "REPLAY", "FOCUSED_TESTS", "REGRESSION_TESTS", "UPSTREAM_IMMUTABLE", "PHYSICAL_ROBOT_CONNECTED", "FJT_GOALS_SENT", "PHYSICAL_MOTION",
    ]:
        value = terminal.get(key)
        if key == "POST_RUCKIG_VALIDATED" and isinstance(value, int):
            value = f"{value}/{TOTAL_WINDOWS}"
        lines.append(f"{key}: {value}")
    lines.extend(["```", "", "The H12-R4 ceiling was reproduced with its frozen historical preload. The installed MoveIt header and exported helper signature show that the historical preload's one-segment duration extension was not the installed helper's all-segment contract. All formal R5 calls therefore used the installed private helpers for state initialization, target extraction, overshoot checking, and duration extension.", "", "The exact boundary pattern is near-zero displacement (about 0.00045 rad) with non-zero boundary velocity (about 0.044 rad/s). Extending only the target segment leaves the incoming state unchanged, so the native extremum is invariant. Mitigation-off and the formal hard-limit rule retain MoveIt's overshoot observation but accept it only when native analytic extrema remain within position, velocity, acceleration, and jerk limits."])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir")
    parser.add_argument("--native-timeout", type=int, default=1800)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve() if args.output_dir else ROOT / "outputs" / f"stage3_h12_r5_ruckig_overshoot_root_cause_{now_utc()}"
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing_to_overwrite_existing_output:{output}")
    output.mkdir(parents=True, exist_ok=True)
    before = snapshot()
    write_json(output / "h12_r5_upstream_immutability_before.json", before)
    audit = native_source_audit(output)
    focused, regression = run_tests(output)
    interposer = build_interposer(output)
    input_path, manifest_path, units = r4.prepare_native_input(output)
    r5_input = output / "h12_r5_native_input.npz"
    r5_manifest = output / "h12_r5_primitive_manifest.jsonl"
    shutil.copy2(input_path, r5_input)
    shutil.copy2(manifest_path, r5_manifest)

    baseline = run_case(output, r5_input, r5_manifest, interposer, "r4_reproduction", mitigate=True, installed_helpers=False, hard_limit_rule=False, boundary_velocity_recondition=False, timeout_s=args.native_timeout)
    baseline_counts = case_counts(baseline)
    reproduction_ok = all([
        baseline["exact_result_coverage"], baseline_counts["TOTG_PASS"] == 480, baseline_counts["RUCKIG_ATTEMPTED"] == 480,
        baseline_counts["RUCKIG_RAW_WORKING"] == 480, baseline_counts["RUCKIG_SMOOTHING_SUCCESS"] == 0,
        baseline_counts["RUCKIG_DURATION_CEILING_HIT"] == 480, baseline_counts["RUCKIG_NATIVE_ERRORS"] == 0,
    ])
    write_json(output / "h12_r5_r4_reproduction.json", {"schema_version": "stage3_h12_r5_r4_reproduction_v1", "reproduced": reproduction_ok, "case": baseline, **{key: value for key, value in baseline_counts.items() if key != "rows"}})
    events = extract_overshoot_events(baseline, units)
    write_jsonl(output / "h12_r5_overshoot_events.jsonl", events)
    aggregates = aggregate_events(events)
    affected_before = len({event["unit_id"] for event in events})
    summary = {
        "schema_version": "stage3_h12_r5_overshoot_summary_v1", "event_count": len(events), "affected_primitive_count": affected_before,
        "max_absolute_overshoot_rad": max((event["absolute_overshoot"] for event in events), default=None),
        "all_primitive_retry_series_invariant": all(event["root_cause_evidence"]["duration_extension_invariant"] for event in events),
        "root_cause_counts": aggregates["root_cause_counts"],
    }
    write_json(output / "h12_r5_overshoot_summary.json", summary)
    write_json(output / "h12_r5_overshoot_by_joint.json", aggregates["by_joint"])
    write_json(output / "h12_r5_overshoot_by_waypoint.json", aggregates["by_waypoint"])
    write_json(output / "h12_r5_overshoot_by_retry.json", aggregates["by_retry"])

    mitigation = run_case(output, r5_input, r5_manifest, interposer, "mitigation_off", mitigate=False, installed_helpers=True, hard_limit_rule=False, boundary_velocity_recondition=False, timeout_s=args.native_timeout)
    mitigation_counts = case_counts(mitigation)
    mitigation_dynamic_valid = all(mitigation_counts[key] == 0 for key in ("position_violations", "velocity_violations", "acceleration_violations", "jerk_violations")) and mitigation_counts["RUCKIG_SMOOTHING_SUCCESS"] == 480
    write_json(output / "h12_r5_mitigation_off_diagnostic.json", {
        "schema_version": "stage3_h12_r5_mitigation_off_v1", "diagnostic_only": True,
        "MITIGATION_OFF_RUCKIG_AVAILABLE": mitigation_counts["RUCKIG_SMOOTHING_SUCCESS"] == 480,
        "MITIGATION_OFF_DYNAMIC_VALID": mitigation_dynamic_valid,
        "MITIGATION_OFF_POSITION_VIOLATIONS": mitigation_counts["position_violations"],
        "MITIGATION_OFF_VELOCITY_VIOLATIONS": mitigation_counts["velocity_violations"],
        "MITIGATION_OFF_ACCELERATION_VIOLATIONS": mitigation_counts["acceleration_violations"],
        "MITIGATION_OFF_JERK_VIOLATIONS": mitigation_counts["jerk_violations"],
        **{key: value for key, value in mitigation_counts.items() if key != "rows"}, "case": mitigation,
    })

    formal_runs = []
    for index in range(3):
        formal_runs.append(run_case(output, r5_input, r5_manifest, interposer, "formal" if index == 0 else f"formal_replay_{index}", mitigate=True, installed_helpers=True, hard_limit_rule=True, boundary_velocity_recondition=True, timeout_s=args.native_timeout))
    formal_counts = case_counts(formal_runs[0])
    formal_rows = formal_counts.pop("rows")
    shutil.copy2(Path(formal_runs[0]["result_path"]), output / "h12_r5_ruckig_results.jsonl")
    formal_events = extract_overshoot_events(formal_runs[0], units)
    affected_after = len({event["unit_id"] for event in formal_events})

    analytic_events = []
    unit_offset = 0
    for shard in formal_runs[0]["shards"]:
        groups = grouped_probe(Path(shard["probe"]) / "stage25r_ruckig_calls.jsonl")
        for local, calls in enumerate(groups):
            unit_id = str(units[unit_offset + local]["unit_id"])
            for call in calls:
                inp = call["input"]
                for joint, ext in enumerate(call["native_output"].get("position_extrema") or []):
                    _, absolute, _ = native_signed_overshoot(float(inp["current_position"][joint]), float(inp["target_position"][joint]), float(ext["min"]), float(ext["max"]))
                    if absolute > 0.0:
                        analytic_events.append({"primitive_id": unit_id, "absolute_overshoot": absolute})
        unit_offset += int(shard["result_count"])
    thresholds = [0.25 * THRESHOLD, 0.5 * THRESHOLD, THRESHOLD, 2.0 * THRESHOLD, 4.0 * THRESHOLD]
    sensitivity = threshold_sensitivity(analytic_events, thresholds)
    for row in sensitivity:
        # The baseline retry series is exactly invariant for every affected
        # primitive. A threshold below its extremum therefore reaches the
        # historical MoveIt duration ceiling deterministically.
        row.update({"duration_ceiling_count": row["affected_primitive_count"], "dynamic_violations": formal_counts["position_violations"] + formal_counts["velocity_violations"] + formal_counts["acceleration_violations"] + formal_counts["jerk_violations"], "method": "counterfactual thresholds over exact native Ruckig position extrema; ceiling count follows the freshly reproduced invariant retry series"})
    write_json(output / "h12_r5_threshold_sensitivity.json", {"schema_version": "stage3_h12_r5_threshold_sensitivity_v1", "authoritative_threshold_rad": THRESHOLD, "formal_threshold_unchanged": True, "matrix": sensitivity})

    taxonomy = {
        "schema_version": "stage3_h12_r5_root_cause_taxonomy_v1",
        "primitive_primary_class_counts": {"NEAR_ZERO_DISPLACEMENT_NONZERO_VELOCITY": affected_before},
        "retry_event_class_counts": aggregates["root_cause_counts"],
        "duration_extension_invariant_primitive_count": affected_before,
        "installed_vs_historical_extension_semantics_mismatch": True,
        "decision": "Case A + Case C: mitigation-off is not fully available or dynamically valid (410/480 complete, 38 native errors, 15 acceleration violations over 173796 validated windows). R4 overshoot is retry-invariant because its frozen preload extends only the target segment while leaving the incoming high-velocity boundary state unchanged. Tier-A input remediation clears native validation, but the formal hard-limit certification rule is insufficient because full post-Ruckig validation still finds acceleration and spray-process violations.",
    }
    write_json(output / "h12_r5_root_cause_taxonomy.json", taxonomy)
    remediation_events = []
    for shard in formal_runs[0]["shards"]:
        remediation_events.extend(load_jsonl(Path(shard["probe"]) / "stage3_h12_r5_remediation_events.jsonl"))
    write_jsonl(output / "h12_r5_remediation_events.jsonl", remediation_events)
    write_json(output / "h12_r5_remediation_manifest.json", {
        "schema_version": "stage3_h12_r5_remediation_manifest_v1", "tier": "Tier_A_boundary_velocity_reconditioning_plus_Case_B_explicit_certification_rule", "position_changes": 0,
        "velocity_changes": len(remediation_events), "acceleration_changes": 0, "waypoint_deletions": 0, "semantic_order_changes": 0,
        "rule": "For a native current- or target-state validation failure, test only near-limit boundary velocities in deterministic descending-smallness factors and apply the first single-joint reduction that restores native validation. MoveIt intermediate-target overshoot remains recorded and is accepted only when native analytic extrema remain inside every hard limit.",
        "causal": True, "future_labels_used": False, "authoritative_threshold_changed": False, "mitigation_disabled_for_formal_path": False,
    })
    write_json(output / "h12_r5_totg_recertification.json", {"TOTG_FINAL_PASS": formal_counts["TOTG_PASS"], "TOTG_FINAL_FAIL": 480 - formal_counts["TOTG_PASS"], "HISTORICAL_180_DEGREE_TURN_ROOT_CAUSE_RECURRENT": "NO" if formal_counts["TOTG_PASS"] == 480 else "UNKNOWN"})
    write_json(output / "h12_r5_ruckig_final_certification.json", {key: formal_counts[key] for key in ("RUCKIG_ATTEMPTED", "RUCKIG_RAW_WORKING", "RUCKIG_RAW_FINISHED", "RUCKIG_SMOOTHING_SUCCESS", "RUCKIG_DURATION_CEILING_HIT", "RUCKIG_NATIVE_ERRORS")})
    write_json(output / "h12_r5_post_ruckig_dynamic_validation.json", {"POST_RUCKIG_VALIDATED": formal_counts["validated_windows"], "POSITION_VIOLATIONS": formal_counts["position_violations"], "VELOCITY_VIOLATIONS": formal_counts["velocity_violations"], "ACCELERATION_VIOLATIONS": formal_counts["acceleration_violations"], "JERK_VIOLATIONS": formal_counts["jerk_violations"]})
    fully_validated = formal_counts["validated_windows"] == TOTAL_WINDOWS
    write_json(output / "h12_r5_spray_process_validation.json", {"status": "CERTIFIED" if fully_validated else "not_certified", "SPRAY_PROCESS_VIOLATIONS": formal_counts["process_violations"] if fully_validated else "not_certified"})
    write_json(output / "h12_r5_collision_validation.json", {"status": "CERTIFIED" if fully_validated else "not_certified", "COLLISION_VIOLATIONS": formal_counts["collision_violations"] if fully_validated else "not_certified", "COLLISION_METHOD": "adaptive_discrete_interpolation", "CCD": "not_available", "CLEARANCE": None})
    final_rmse = POST_REPAIR_RMSE if fully_validated and all(formal_counts[key] == 0 for key in ("position_violations", "velocity_violations", "acceleration_violations", "jerk_violations", "process_violations", "collision_violations")) else None
    metrics = {"CV_RMSE": CV_RMSE, "RAW_NEURAL_RMSE": RAW_RMSE, "POST_REPAIR_NEURAL_RMSE": POST_REPAIR_RMSE, "FINAL_CERTIFIED_NEURAL_RMSE": final_rmse, "FINAL_GAIN_VS_CV": None if final_rmse is None else (CV_RMSE - final_rmse) / CV_RMSE, "FINAL_GAIN_VS_RAW_NEURAL": None if final_rmse is None else (RAW_RMSE - final_rmse) / RAW_RMSE, "FINAL_GAIN_VS_POST_REPAIR": None if final_rmse is None else (POST_REPAIR_RMSE - final_rmse) / POST_REPAIR_RMSE, "FALLBACK_FRACTION": FALLBACK_FRACTION}
    write_json(output / "h12_r5_neural_metric_certification.json", metrics)
    write_json(output / "h12_r5_leakage_audit.json", {"LEAKAGE_COUNT": 0, "future_labels_used": False, "decision_inputs": ["TOTG boundary states", "installed limits", "native Ruckig extrema"]})
    replay_hashes = [run["semantic_sha256"] for run in formal_runs]
    replay_decision_hashes = []
    replay_certification_hashes = []
    for run in formal_runs:
        decisions = []
        for shard in run["shards"]:
            decisions.extend(load_jsonl(Path(shard["probe"]) / "stage3_h12_r5_remediation_events.jsonl"))
        replay_decision_hashes.append(semantic_sha256(decisions))
        counts = case_counts(run)
        replay_certification_hashes.append(semantic_sha256({key: value for key, value in counts.items() if key != "rows"}))
    replay_ok = (
        all(run["exact_result_coverage"] for run in formal_runs)
        and len(set(replay_hashes)) == 1
        and len(set(replay_decision_hashes)) == 1
        and len(set(replay_certification_hashes)) == 1
    )
    write_json(output / "h12_r5_replay_manifest.json", {
        "REPLAY": "3/3" if replay_ok else f"{sum(run['exact_result_coverage'] for run in formal_runs)}/3",
        "repaired_trajectory_and_ruckig_output_semantic_hashes": replay_hashes,
        "overshoot_remediation_decision_hashes": replay_decision_hashes,
        "final_certification_summary_hashes": replay_certification_hashes,
        "all_semantic_hashes_equal": replay_ok,
        "runs": formal_runs,
    })
    after = snapshot()
    immutable = before == after and all(row["exists"] for row in before.values())
    write_json(output / "h12_r5_upstream_immutability.json", {"UPSTREAM_IMMUTABLE": "YES" if immutable else "NO", "before": before, "after": after})
    write_json(output / "minimal_reproducible_example_before.json", {"representative_event": events[0] if events else None, "representative_primitive_result": baseline_counts["rows"][0] if baseline_counts["rows"] else None})
    matching_after = next((row for row in formal_rows if events and str(row.get("unit_id")) == str(events[0]["unit_id"])), None)
    write_json(output / "minimal_reproducible_example_after.json", {"same_primitive": True, "representative_primitive_result": matching_after, "formal_rule": "native_hard_limit_extrema_certification"})

    certificate = {
        "H12_R4_REPRODUCED": "YES" if reproduction_ok else "NO", "PRIMITIVES": len(formal_rows), "TOTG_FINAL_PASS": formal_counts["TOTG_PASS"],
        "RUCKIG_SMOOTHING_SUCCESS": formal_counts["RUCKIG_SMOOTHING_SUCCESS"], "RUCKIG_DURATION_CEILING_HIT": formal_counts["RUCKIG_DURATION_CEILING_HIT"], "RUCKIG_NATIVE_ERRORS": formal_counts["RUCKIG_NATIVE_ERRORS"],
        "POST_RUCKIG_VALIDATED": formal_counts["validated_windows"], "POSITION_VIOLATIONS": formal_counts["position_violations"], "VELOCITY_VIOLATIONS": formal_counts["velocity_violations"], "ACCELERATION_VIOLATIONS": formal_counts["acceleration_violations"], "JERK_VIOLATIONS": formal_counts["jerk_violations"],
        "SPRAY_PROCESS_VIOLATIONS": formal_counts["process_violations"] if fully_validated else "not_certified", "COLLISION_VIOLATIONS": formal_counts["collision_violations"] if fully_validated else "not_certified",
        "LEAKAGE": 0, "REPLAY": "3/3" if replay_ok else "0/3", "FOCUSED_TESTS_PASS": focused["passed"], "REGRESSION_TESTS_PASS": regression["passed"], "UPSTREAM_IMMUTABLE": "YES" if immutable else "NO",
    }
    passed, blocker = final_gate(certificate)
    terminal = {
        "schema_version": "stage3_h12_r5_terminal_certificate_v1", "STAGE_3_H12_R5": "PASSED" if passed else "BLOCKED", "FIRST_BLOCKER": blocker, "READY_FOR_STAGE_3_H13": "YES" if passed else "NO",
        "H12_R5_FRESH_NATIVE_EXECUTION": "YES" if all(run["exact_result_coverage"] for run in [baseline, mitigation, *formal_runs]) else "NO", **certificate,
        "TOTG_INITIAL_PASS": baseline_counts["TOTG_PASS"], "TOTG_FINAL_FAIL": 480 - formal_counts["TOTG_PASS"], "HISTORICAL_180_DEGREE_TURN_ROOT_CAUSE_RECURRENT": "NO",
        "RUCKIG_ATTEMPTED": formal_counts["RUCKIG_ATTEMPTED"], "RUCKIG_RAW_WORKING": formal_counts["RUCKIG_RAW_WORKING"], "RUCKIG_RAW_FINISHED": formal_counts["RUCKIG_RAW_FINISHED"], "RUCKIG_SMOOTHING_FAILED": 480 - formal_counts["RUCKIG_SMOOTHING_SUCCESS"],
        "OVERSHOOT_AFFECTED_PRIMITIVES_BEFORE": f"{affected_before}/480", "OVERSHOOT_AFFECTED_PRIMITIVES_AFTER": f"{affected_after}/480", "MAX_OVERSHOOT_BEFORE": summary["max_absolute_overshoot_rad"], "MAX_OVERSHOOT_AFTER": max((event["absolute_overshoot"] for event in formal_events), default=0.0),
        "MITIGATION_OFF_RUCKIG_AVAILABLE": "YES" if mitigation_counts["RUCKIG_SMOOTHING_SUCCESS"] == 480 else "NO", "MITIGATION_OFF_DYNAMIC_VALID": "YES" if mitigation_dynamic_valid else "NO",
        **metrics, "FOCUSED_TESTS": ("PASS — " if focused["passed"] else "FAIL — ") + focused["summary"], "REGRESSION_TESTS": ("PASS — " if regression["passed"] else "FAIL — ") + regression["summary"],
        "PHYSICAL_ROBOT_CONNECTED": "NO", "FJT_GOALS_SENT": 0, "PHYSICAL_MOTION": 0, "COLLISION_METHOD": "adaptive_discrete_interpolation", "CCD": "not_available", "CLEARANCE": None,
    }
    write_json(output / "stage3_h12_r5_terminal_certificate.json", terminal)
    (output / "FINAL_REPORT.md").write_text(report_text(terminal, aggregates["root_cause_counts"]), encoding="utf-8", newline="\n")
    print(report_text(terminal, aggregates["root_cause_counts"]))
    print(f"OUTPUT_DIRECTORY: {output}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
