#!/usr/bin/env python3
"""Stage 3 H7.4 native Ruckig localization and strict recertification."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import stage3_h6_4 as h64
from scripts import stage3_h7_2 as h72
from scripts import stage3_h7_3 as h73


H64 = h72.H64_ROOT
H72 = h73.H72_ROOT
H73 = ROOT / "outputs/stage3_h7_3_target55_controlled_stop_20260809T170900Z"
INTERPOSER = ROOT / "tools/stage25r_ruckig_interposer.cpp"
PRIMITIVES = h73.PRIMITIVE_ORDER
COLLISION_METHOD = "adaptive_discrete_interpolation"
NO_SOLUTION_TEXT = "Ruckig extended the trajectory duration to its maximum and still did not find a solution"

REQUIRED = [
    "FINAL_REPORT.md",
    "stage3_h7_4_terminal_certificate.json",
    "stage3_h7_4_input_freeze_manifest.json",
    "stage3_h7_4_raw_ruckig_pair_audit.jsonl",
    "stage3_h7_4_first_failure_per_primitive.json",
    "stage3_h7_4_ruckig_validate_input.jsonl",
    "stage3_h7_4_moveit_wrapper_completion_audit.json",
    "stage3_h7_4_overshoot_ab_test.json",
    "stage3_h7_4_post_ruckig_validation.json",
    "stage3_h7_4_shutdown_audit.json",
    "stage3_h7_4_replay_determinism.json",
    "stage3_h7_4_regression_report.json",
]


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def dump_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot() -> dict[str, Any]:
    return {
        "h6_4_tree": h64.tree_snapshot(H64),
        "h7_2_tree": h64.tree_snapshot(H72),
        "h7_3_tree": h64.tree_snapshot(H73),
        "joint_limits": h72.file_record(h72.JOINT_LIMITS, "frozen 8 rad/s^3 jerk limits"),
        "totg_parameters": h72.file_record(H73 / "stage3_h7_3_totg_parameters.json", "frozen H7.3/H7.2 TOTG parameters"),
        "interposer_post_patch": h72.file_record(INTERPOSER, "additive native H7.4 instrumentation source"),
    }


def immutable(before: Mapping[str, Any]) -> dict[str, Any]:
    after = snapshot()
    protected = ("h6_4_tree", "h7_2_tree", "h7_3_tree", "joint_limits", "totg_parameters")
    checks = {key: before.get(key) == after.get(key) for key in protected}
    return {"all_protected_hashes_match": all(checks.values()), "checks": checks, "after": after}


def build(output: Path) -> dict[str, Any]:
    build_base = ROOT / "build/stage3_h7_4_native"
    install_base = ROOT / "install/stage3_h7_4_native"
    command = " && ".join(
        [
            "source /opt/ros/jazzy/setup.bash",
            f"source {shlex.quote(h72.wsl_path(ROOT / 'install/setup.bash'))}",
            f"colcon build --base-paths {shlex.quote(h72.wsl_path(ROOT / 'ros2_moveit_bridge'))} "
            f"--build-base {shlex.quote(h72.wsl_path(build_base))} "
            f"--install-base {shlex.quote(h72.wsl_path(install_base))} --merge-install",
        ]
    )
    code, stdout, stderr = h72.run_wsl(command, 1200)
    (output / "stage3_h7_4_native_build.log").write_text(stdout + "\n--- STDERR ---\n" + stderr, encoding="utf-8")

    so = output / "libstage3_h7_4_ruckig_interposer.so"
    include_flags = "-I/opt/ros/jazzy/include $(for d in /opt/ros/jazzy/include/*; do printf -- '-I%s ' \"$d\"; done)"
    compile_command = (
        f"g++ {include_flags} -I/usr/include/eigen3 -std=c++17 -O2 -fPIC -shared -Wl,-Bsymbolic "
        f"-o {shlex.quote(h72.wsl_path(so))} {shlex.quote(h72.wsl_path(INTERPOSER))} "
        "-L/opt/ros/jazzy/lib -L/opt/ros/jazzy/lib/x86_64-linux-gnu "
        "-Wl,-rpath,/opt/ros/jazzy/lib -Wl,-rpath,/opt/ros/jazzy/lib/x86_64-linux-gnu "
        "-Wl,--no-as-needed -lmoveit_robot_trajectory -lmoveit_robot_state -lmoveit_robot_model -lmoveit_utils -lruckig -ldl"
    )
    ccode, cout, cerr = h72.run_wsl("source /opt/ros/jazzy/setup.bash && " + compile_command, 1200)
    (output / "stage3_h7_4_interposer_build.log").write_text(cout + "\n--- STDERR ---\n" + cerr, encoding="utf-8")
    result = {
        "worker_returncode": code,
        "interposer_returncode": ccode,
        "status": "PASSED" if code == ccode == 0 and so.is_file() else "BLOCKED",
        "interposer_source_sha256_after_patch": sha256(INTERPOSER),
        "interposer_binary_sha256": sha256(so) if so.is_file() else None,
        "local_patch_scope": "instrumentation only; system MoveIt installation not modified",
        "original_moveit_jazzy_behavior_preserved_in_h7_3_immutable_evidence": True,
    }
    dump_json(output / "stage3_h7_4_native_build.json", result)
    return result


def run_case(output: Path, label: str, mitigate: bool, so: Path) -> dict[str, Any]:
    case = output / label
    native = case / "native_probe"
    case.mkdir(parents=True, exist_ok=False)
    native.mkdir()
    install = ROOT / "install/stage3_h7_4_native/setup.bash"
    command = " && ".join(
        [
            "source /opt/ros/jazzy/setup.bash",
            f"source {shlex.quote(h72.wsl_path(ROOT / 'install/setup.bash'))}",
            f"source {shlex.quote(h72.wsl_path(install))}",
            f"export LD_PRELOAD={shlex.quote(h72.wsl_path(so))}",
            f"export STAGE25R_NATIVE_DIR={shlex.quote(h72.wsl_path(native))}",
            "ros2 launch fr5_tunnel_moveit_bridge stage3_h7_4_native.launch.py "
            f"output_dir:={shlex.quote(h72.wsl_path(case))} "
            f"h6_4_final_validation:={shlex.quote(h72.wsl_path(h72.H64_VALIDATION))} "
            f"h6_4_segments:={shlex.quote(h72.wsl_path(h72.H64_SEGMENTS))} "
            f"process_contract:={shlex.quote(h72.wsl_path(h72.PROCESS_CONTRACT))} "
            f"fixture_mesh:={shlex.quote(h72.wsl_path(h72.FIXTURE_MESH))} "
            f"totg_parameters:={shlex.quote(h72.wsl_path(H73 / 'stage3_h7_3_totg_parameters.json'))} "
            f"mitigate_overshoot:={'true' if mitigate else 'false'}",
        ]
    )
    code, stdout, stderr = h72.run_wsl(command, 3600)
    combined = stdout + "\n--- STDERR ---\n" + stderr
    (case / "launch.log").write_text(combined, encoding="utf-8")
    result_path = case / "native_result.json"
    result = h72.load_json(result_path) if result_path.is_file() else {"status": "BLOCKED", "primitive_results": []}
    exit_codes = [int(value) for value in re.findall(r"process has died .*?exit code (-?\d+)", combined)]
    result.update(
        label=label,
        launcher_returncode=code,
        process_exit_codes=exit_codes,
        maximum_duration_no_solution_log_count=combined.count(NO_SOLUTION_TEXT),
        mitigate_overshoot=mitigate,
    )
    dump_json(result_path, result)
    return result


def split_invocations(rows: Sequence[Mapping[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    for raw in rows:
        row = dict(raw)
        if int(row.get("global_calculate_call_index", -1)) == 0:
            groups.append([])
        if not groups:
            groups.append([])
        groups[-1].append(row)
    return groups


def primitive_source_rows() -> dict[str, list[dict[str, Any]]]:
    rows = h72.load_jsonl(h72.H64_VALIDATION)
    grouped: dict[int, list[dict[str, Any]]] = {int(x): [] for x in h72.SEGMENT_ORDER}
    for row in rows:
        grouped[int(row["segment_id"])].append(dict(row))
    metadata = {int(row["segment_id"]): row for row in h72.load_json(h72.H64_SEGMENTS)["segments"]}
    result: dict[str, list[dict[str, Any]]] = {}
    for primitive in h73.authoritative_primitives()["primitives"]:
        segment = int(primitive["h6_4_segment_id"])
        start = int(primitive["h6_4_segment_local_start"])
        end = int(primitive["h6_4_segment_local_end"])
        selected = grouped[segment][start : end + 1]
        storage_start = int(metadata[segment]["waypoint_start"])
        for local, row in enumerate(selected, start=start):
            row["h6_4_segment_local_waypoint_index"] = local
            row["h6_4_storage_waypoint_index"] = storage_start + local
            row["h6_4_process_waypoint_index"] = int(row["waypoint_index"])
        result[str(primitive["primitive_id"])] = selected
    return result


def provenance(q: Sequence[float], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    matrix = np.asarray([row["selected_joint_values"] for row in rows], dtype=float)
    distances = np.linalg.norm(matrix - np.asarray(q, dtype=float), axis=1)
    index = int(np.argmin(distances))
    source = rows[index]
    return {
        "nearest_primitive_source_index": index,
        "h6_4_segment_local_waypoint_index": int(source["h6_4_segment_local_waypoint_index"]),
        "h6_4_storage_waypoint_index": int(source["h6_4_storage_waypoint_index"]),
        "h6_4_process_waypoint_index": int(source["h6_4_process_waypoint_index"]),
        "source_target_id": source.get("source_target_id"),
        "destination_target_id": source.get("destination_target_id"),
        "nearest_joint_distance_rad": float(distances[index]),
        "provenance_semantics": "nearest exact H6.4 source waypoint on the frozen primitive; post-TOTG states may lie between source waypoints",
    }


def validation_details(call: Mapping[str, Any]) -> list[dict[str, Any]]:
    inp = call["input"]
    names = ["j1", "j2", "j3", "j4", "j5", "j6"]
    details: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        vmax = float(inp["max_velocity"][index])
        amax = float(inp["max_acceleration"][index])
        jmax = float(inp["max_jerk"][index])
        for state in ("current", "target"):
            v = float(inp[f"{state}_velocity"][index])
            a = float(inp[f"{state}_acceleration"][index])
            violations: list[str] = []
            if not all(math.isfinite(x) for x in (v, a, vmax, amax, jmax)):
                violations.append("non_finite_state_or_limit")
            if jmax <= 0:
                violations.append("non_positive_max_jerk")
            if amax <= 0 or abs(a) > amax:
                violations.append("acceleration_outside_limit")
            if vmax <= 0 or abs(v) > vmax:
                violations.append("velocity_outside_limit")
            if state == "current" and jmax > 0:
                stopped_v = v + math.copysign(a * a / (2.0 * jmax), a) if a else v
                if stopped_v > vmax or stopped_v < -vmax:
                    violations.append("current_jerk_braking_velocity_feasibility")
            if state == "target" and jmax > 0:
                reverse_v = v - math.copysign(a * a / (2.0 * jmax), a) if a else v
                if reverse_v > vmax or reverse_v < -vmax:
                    violations.append("target_reverse_jerk_velocity_feasibility")
            if violations:
                details.append({"joint_index": index, "joint_name": name, "state": state, "violations": violations})
    return details


def audit_primary(output: Path, case: Path) -> dict[str, Any]:
    native = case / "native_probe"
    calls_groups = split_invocations(load_jsonl(native / "stage25r_ruckig_calls.jsonl"))
    resolutions_groups = split_invocations(load_jsonl(native / "stage25r_ruckig_call_resolutions.jsonl"))
    summaries = load_jsonl(native / "stage25r_native_run_summaries.jsonl")
    sources = primitive_source_rows()
    pair_rows: list[dict[str, Any]] = []
    validate_rows: list[dict[str, Any]] = []
    first_failures: list[dict[str, Any]] = []
    raw_errors = 0
    inputs_valid = 0
    total_calls = 0

    for invocation, primitive_id in enumerate(PRIMITIVES):
        calls = calls_groups[invocation] if invocation < len(calls_groups) else []
        resolutions = resolutions_groups[invocation] if invocation < len(resolutions_groups) else []
        resolution_by_call = {int(row["global_calculate_call_index"]): row for row in resolutions}
        source_rows = sources[str(primitive_id)]
        first: dict[str, Any] | None = None
        for call in calls:
            total_calls += 1
            call_index = int(call["global_calculate_call_index"])
            resolution = resolution_by_call.get(call_index, {})
            inp = call["input"]
            numeric = int(call["result"]["numeric_result"])
            acceptable = numeric in (0, 1)
            strict_valid = bool(call.get("native_validate_input", {}).get("current_and_target_strict"))
            raw_errors += int(not acceptable)
            inputs_valid += int(strict_valid)
            row = {
                "schema_version": "stage3-h7-4-raw-ruckig-pair-audit-v1",
                "primitive_id": primitive_id,
                "invocation_index": invocation,
                "waypoint_index": int(call["waypoint_idx"]),
                "calculate_call_index": call_index,
                "attempt_index": int(call["attempt_index"]),
                "trajectory_generation": int(call.get("trajectory_generation", 0)),
                "current_h6_4_source_provenance": provenance(inp["current_position"], source_rows),
                "target_h6_4_source_provenance": provenance(inp["target_position"], source_rows),
                "current_position": inp["current_position"],
                "current_velocity": inp["current_velocity"],
                "current_acceleration": inp["current_acceleration"],
                "target_position": inp["target_position"],
                "target_velocity": inp["target_velocity"],
                "target_acceleration": inp["target_acceleration"],
                "max_velocity": inp["max_velocity"],
                "max_acceleration": inp["max_acceleration"],
                "max_jerk": inp["max_jerk"],
                "moveit_duration_before_smoothing_s": call["robot_trajectory_segment_duration_before"],
                "duration_extension_factor": call["duration_extension_factor"],
                "raw_ruckig_result": call["result"]["result_name"],
                "raw_numeric_result_code": numeric,
                "output_duration_s": (call.get("native_output") or {}).get("duration"),
                "overshoot_detected": bool(resolution.get("overshoot", False)),
                "caused_restart_or_duration_extension": bool(resolution.get("retry_call", False)),
                "native_validate_input": call.get("native_validate_input"),
            }
            pair_rows.append(row)
            violations = validation_details(call)
            validate_rows.append(
                {
                    "schema_version": "stage3-h7-4-ruckig-validate-input-v1",
                    "primitive_id": primitive_id,
                    "waypoint_index": row["waypoint_index"],
                    "attempt_index": row["attempt_index"],
                    "native_validate_input": call.get("native_validate_input"),
                    "joint_state_violation_details": violations,
                }
            )
            if first is None and (not acceptable or resolution.get("retry_call")):
                cause = "raw_ruckig_error" if not acceptable else "overshoot_mitigation_restart"
                first = {
                    "primitive_id": primitive_id,
                    "waypoint_index": row["waypoint_index"],
                    "attempt_index": row["attempt_index"],
                    "raw_ruckig_result": row["raw_ruckig_result"],
                    "raw_numeric_result_code": numeric,
                    "trigger": cause,
                    "input_strict_valid": strict_valid,
                    "overshoot_detected": row["overshoot_detected"],
                    "duration_extension_factor": row["duration_extension_factor"],
                    "joint_state_violation_details": violations,
                }
        first_failures.append(first or {"primitive_id": primitive_id, "failure": None})

    dump_jsonl(output / "stage3_h7_4_raw_ruckig_pair_audit.jsonl", pair_rows)
    dump_jsonl(output / "stage3_h7_4_ruckig_validate_input.jsonl", validate_rows)
    dump_json(
        output / "stage3_h7_4_first_failure_per_primitive.json",
        {"first_failure_per_primitive": first_failures, "all_ten_primitives_audited": len(first_failures) == 10},
    )

    completion_rows: list[dict[str, Any]] = []
    for index, primitive_id in enumerate(PRIMITIVES):
        summary = summaries[index] if index < len(summaries) else {}
        completion_rows.append(
            {
                "primitive_id": primitive_id,
                "duration_extension_factor": summary.get("final_duration_extension_factor"),
                "MAX_DURATION_EXTENSION_FACTOR": summary.get("max_duration_extension_factor", 50.0),
                "final_ruckig_result": summary.get("final_ruckig_result"),
                "smoothing_complete": summary.get("smoothing_complete", False),
                "duration_ceiling_hit": summary.get("duration_ceiling_hit", False),
                "wrapper_returned_bool": summary.get("wrapper_returned_bool", False),
                "strict_completion": summary.get("strict_completion", False),
            }
        )
    wrapper = {
        "schema_version": "stage3-h7-4-moveit-wrapper-completion-audit-v1",
        "contract": "wrapper/native result acceptable AND smoothing_complete == true AND maximum-duration/no-solution log count == 0",
        "wrapper_true_is_not_ruckig_passed": True,
        "primitives": completion_rows,
        "MOVEIT_RUCKIG_WRAPPER_TRUE_PRIMITIVES": sum(row["wrapper_returned_bool"] is True for row in completion_rows),
        "RUCKIG_SMOOTHING_COMPLETE_PRIMITIVES": sum(row["smoothing_complete"] is True for row in completion_rows),
        "RUCKIG_RESULT_ERROR_COUNT": raw_errors,
    }
    dump_json(output / "stage3_h7_4_moveit_wrapper_completion_audit.json", wrapper)
    return {
        "total_calls": total_calls,
        "inputs_valid": inputs_valid,
        "raw_errors": raw_errors,
        "raw_calculations_passed": total_calls - raw_errors,
        "first_failures": first_failures,
        "wrapper": wrapper,
    }


def post_validation(output: Path, primary: Mapping[str, Any]) -> dict[str, Any]:
    rows = list(primary.get("primitive_results", []))
    dynamic = [row.get("post_ruckig_dynamics", {}) for row in rows]
    strict = len(rows) == 10 and all(row.get("ruckig_smoothing_complete") is True for row in rows)
    report = {
        "schema_version": "stage3-h7-4-post-ruckig-validation-v1",
        "primitive_count": len(rows),
        "position_limit_violations": sum(int(row.get("position_limit_violation_count", 0)) for row in dynamic),
        "velocity_limit_violations": sum(int(row.get("velocity_limit_violation_count", 0)) for row in dynamic),
        "acceleration_limit_violations": sum(int(row.get("acceleration_limit_violation_count", 0)) for row in dynamic),
        "non_monotonic_timestamp_count": sum(int(row.get("non_monotonic_timestamp_count", 0)) for row in dynamic),
        "nan_inf_count": sum(int(row.get("nan_inf_count", 0)) for row in dynamic),
        "endpoint_preservation": "PASSED" if len(rows) == 10 and all(row.get("start_end_configuration_preserved") for row in rows) else "BLOCKED",
        "native_jerk_certification": "PASSED" if strict else "BLOCKED",
        "process_tolerance": "PASSED" if strict and all(row.get("post_ruckig_process", {}).get("status") == "PASSED" for row in rows) else "BLOCKED",
        "collision_certification": "PASSED" if strict and all(row.get("post_ruckig_collision", {}).get("status") == "PASSED" for row in rows) else "BLOCKED",
        "collision_method": COLLISION_METHOD,
        "CCD": "not_available",
        "CLEARANCE": None,
        "finite_difference_jerk_is_supplementary_only": True,
    }
    dump_json(output / "stage3_h7_4_post_ruckig_validation.json", report)
    return report


def shutdown_audit(output: Path, runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    records = []
    for run in runs:
        case = output / str(run["label"])
        events = load_jsonl(case / "lifecycle_events.jsonl")
        exit_codes = list(run.get("process_exit_codes", []))
        clean = not any(code < 0 for code in exit_codes) and int(run.get("launcher_returncode", 1)) in (0, 2)
        records.append(
            {
                "label": run["label"],
                "launcher_returncode": run.get("launcher_returncode"),
                "process_exit_codes": exit_codes,
                "lifecycle_events": [row.get("event") for row in events],
                "last_completed_stage": events[-1].get("event") if events else None,
                "clean": clean,
                "classification": (
                    "post_main_return_process_teardown_segfault" if not clean and events and events[-1].get("event") == "main_return"
                    else "moveitpy_python_object_destructor_segfault_during_del" if not clean and events and events[-1].get("event") == "after_moveitpy_shutdown"
                    else "clean" if clean
                    else "segfault_or_abnormal_exit_before_main_return"
                ),
            }
        )
    report = {"schema_version": "stage3-h7-4-shutdown-audit-v1", "runs": records, "MOVEITPY_SHUTDOWN_CLEAN": "YES" if all(r["clean"] for r in records) else "NO"}
    dump_json(output / "stage3_h7_4_shutdown_audit.json", report)
    return report


def replay_report(output: Path, runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    hashes = [run.get("semantic_hash") for run in runs]
    counters = [
        {
            "totg": run.get("totg_primitives_passed"),
            "wrapper_true": run.get("ruckig_wrapper_true_primitives"),
            "complete": run.get("ruckig_smoothing_complete_primitives"),
            "logs": run.get("maximum_duration_no_solution_log_count"),
        }
        for run in runs
    ]
    identical = len(runs) == 3 and len(set(hashes)) == 1 and len({json.dumps(x, sort_keys=True) for x in counters}) == 1
    report = {"semantic_hashes": hashes, "validation_counters": counters, "identical": identical, "DETERMINISTIC_REPLAY": "3/3" if identical else "0/3"}
    dump_json(output / "stage3_h7_4_replay_determinism.json", report)
    return report


def regressions(output: Path) -> dict[str, Any]:
    tests = ["tests/test_stage3_h7_4.py"] + [
        "tests/test_stage3_h7_3.py", "tests/test_stage3_h7_2.py", "tests/test_stage3_h7_1.py", "tests/test_stage3_h7.py",
        "tests/test_stage3_h6_4.py", "tests/test_stage3_h6_3.py", "tests/test_stage3_h6_2.py", "tests/test_stage3_h6_1.py",
        "tests/test_stage3_h6.py", "tests/test_stage3_h5.py", "tests/test_stage3_h4_4.py",
        "tests/test_stage3_h4_reachability.py", "tests/test_stage3_h3_task_representation.py",
        "tests/test_stage3_h2_geometry.py", "tests/test_stage3_h1_contract.py",
    ]
    process = subprocess.run([sys.executable, "-m", "pytest", "-q", *tests], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1800, check=False)
    raw = process.stdout + "\n--- STDERR ---\n" + process.stderr
    failed = re.search(r"(\d+) failed", raw)
    passed = re.search(r"(\d+) passed", raw)
    report = {"returncode": process.returncode, "passed": int(passed.group(1)) if passed else None, "new_regression_failures": int(failed.group(1)) if failed else (0 if process.returncode == 0 else 1), "raw_output_tail": raw[-16000:]}
    dump_json(output / "stage3_h7_4_regression_report.json", report)
    return report


def finalize(output: Path, before: Mapping[str, Any], runs: list[dict[str, Any]], ab_false: dict[str, Any]) -> dict[str, Any]:
    primary = runs[0]
    native_audit = audit_primary(output, output / "fresh_process_replay_1")
    post = post_validation(output, primary)
    shutdown = shutdown_audit(output, [*runs, ab_false])
    replay = replay_report(output, runs)
    regression = regressions(output)
    unchanged = immutable(before)
    dump_json(output / "stage3_h7_4_immutable_after_check.json", unchanged)

    true_complete = int(primary.get("ruckig_smoothing_complete_primitives", 0))
    false_complete = int(ab_false.get("ruckig_smoothing_complete_primitives", 0))
    ab = {
        "mitigate_overshoot_true": {"smoothing_complete_primitives": true_complete, "log_count": primary.get("maximum_duration_no_solution_log_count"), "semantic_hash": primary.get("semantic_hash")},
        "mitigate_overshoot_false": {"smoothing_complete_primitives": false_complete, "log_count": ab_false.get("maximum_duration_no_solution_log_count"), "semantic_hash": ab_false.get("semantic_hash")},
        "results_differ": true_complete != false_complete or primary.get("semantic_hash") != ab_false.get("semantic_hash"),
        "formal_certification_policy": "mitigate_overshoot=true remains authoritative under the frozen Stage 3 safety/process contract; false is diagnostic only",
        "turning_mitigation_off_does_not_automatically_pass": True,
    }
    dump_json(output / "stage3_h7_4_overshoot_ab_test.json", ab)

    failures = native_audit["first_failures"]
    any_invalid = any(item.get("input_strict_valid") is False for item in failures if item.get("failure") is not None)
    any_overshoot = any(item.get("trigger") == "overshoot_mitigation_restart" for item in failures)
    if any_invalid:
        root_cause = "INVALID_RUCKIG_INPUT_STATE"
    elif native_audit["raw_errors"]:
        root_cause = "RUCKIG_NUMERICAL_OR_TRAJECTORY_DURATION_FAILURE"
    elif any_overshoot:
        root_cause = "OVERSHOOT_MITIGATION_FAILURE"
    elif true_complete < 10:
        root_cause = "MOVEIT_RUCKIG_WRAPPER_CONTROL_FLOW_FAILURE"
    else:
        root_cause = "OTHER"

    wrapper = native_audit["wrapper"]
    log_count = int(primary.get("maximum_duration_no_solution_log_count", 0))
    result_errors = int(native_audit["raw_errors"])
    first = next((item for item in failures if item.get("trigger")), None)
    first_text = "none" if first is None else f"{first['primitive_id']}:{first['waypoint_index']}:{first['raw_ruckig_result']}"
    terminal = {
        "schema_version": "stage3-h7-4-terminal-certificate-v1",
        "STAGE_3_H7_4": "BLOCKED",
        "FIRST_BLOCKER": "native_ruckig_strict_completion_failed" if true_complete < 10 else ("moveitpy_shutdown_not_clean" if shutdown["MOVEITPY_SHUTDOWN_CLEAN"] == "NO" else "none"),
        "ROOT_CAUSE_CLASSIFICATION": root_cause,
        "H6_4_IMMUTABLE": "YES" if unchanged["checks"]["h6_4_tree"] else "NO",
        "H7_2_IMMUTABLE": "YES" if unchanged["checks"]["h7_2_tree"] else "NO",
        "H7_3_IMMUTABLE": "YES" if unchanged["checks"]["h7_3_tree"] else "NO",
        "TOTG_PRIMITIVES_PASSED": f"{primary.get('totg_primitives_passed', 0)}/10",
        "RAW_RUCKIG_INPUTS_VALID": f"{native_audit['inputs_valid']}/{native_audit['total_calls']}",
        "RAW_RUCKIG_CALCULATIONS_PASSED": f"{native_audit['raw_calculations_passed']}/{native_audit['total_calls']}",
        "FIRST_RAW_RUCKIG_FAILURE": first_text,
        "MOVEIT_RUCKIG_WRAPPER_TRUE_PRIMITIVES": f"{wrapper['MOVEIT_RUCKIG_WRAPPER_TRUE_PRIMITIVES']}/10",
        "RUCKIG_SMOOTHING_COMPLETE_PRIMITIVES": f"{true_complete}/10",
        "RUCKIG_MAX_DURATION_NO_SOLUTION_LOG_COUNT": log_count,
        "RUCKIG_RESULT_ERROR_COUNT": result_errors,
        "POST_RUCKIG_POSITION_LIMIT_VIOLATIONS": post["position_limit_violations"],
        "POST_RUCKIG_VELOCITY_LIMIT_VIOLATIONS": post["velocity_limit_violations"],
        "POST_RUCKIG_ACCELERATION_LIMIT_VIOLATIONS": post["acceleration_limit_violations"],
        "NATIVE_JERK_CERTIFICATION": post["native_jerk_certification"],
        "POST_RUCKIG_PROCESS_TOLERANCE": post["process_tolerance"],
        "POST_RUCKIG_COLLISION_CERTIFICATION": post["collision_certification"],
        "MOVEITPY_SHUTDOWN_CLEAN": shutdown["MOVEITPY_SHUTDOWN_CLEAN"],
        "DETERMINISTIC_REPLAY": replay["DETERMINISTIC_REPLAY"],
        "NEW_REGRESSION_FAILURES": regression["new_regression_failures"],
        "NEW_FJT_GOALS_SENT": 0,
        "ROBOT_MOTION_STARTED": "NO",
        "FORMAL_LEDGER_MUTATED": "NO",
        "READY_FOR_STAGE_3_H8": "NO",
    }
    passed = all(
        [
            terminal["FIRST_BLOCKER"] == "none", true_complete == 10, log_count == 0, result_errors == 0,
            post["native_jerk_certification"] == "PASSED", post["process_tolerance"] == "PASSED",
            post["collision_certification"] == "PASSED", shutdown["MOVEITPY_SHUTDOWN_CLEAN"] == "YES",
            replay["DETERMINISTIC_REPLAY"] == "3/3", regression["new_regression_failures"] == 0,
            unchanged["all_protected_hashes_match"],
        ]
    )
    if passed:
        terminal.update(STAGE_3_H7_4="PASSED", READY_FOR_STAGE_3_H8="YES")
    dump_json(output / "stage3_h7_4_terminal_certificate.json", terminal)
    report = "\n".join(["# Stage 3 H7.4 Final Report", "", *[f"{key}: {value}" for key, value in terminal.items() if key != "schema_version"], "", "## Root-cause evidence", "", f"Classification: `{root_cause}`. Every wrapper boolean is subordinate to native result, `smoothing_complete`, and the maximum-duration/no-solution diagnostic.", "", "Collision checking used `adaptive_discrete_interpolation`; strict CCD and clearance remain unavailable (`null`).", ""])
    (output / "FINAL_REPORT.md").write_text(report, encoding="utf-8", newline="\n")
    return terminal


def orchestrate(output: Path) -> int:
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty H7.4 output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    before = snapshot()
    dump_json(output / "stage3_h7_4_input_freeze_manifest.json", {"captured_before_native_run": True, "sources": before, "frozen_parameters": {"velocity_scaling": 0.15, "acceleration_scaling": 0.15, "path_tolerance": 0.00025, "resample_dt": 0.01, "min_angle_change": 0.0005, "jerk_limits_rad_s3": [8.0] * 6}, "forbidden": ["FJT goal", "robot motion", "controller execution", "formal ledger mutation", "Stage 3 H8", "H6.4/H7.2/H7.3 artifact mutation"]})
    built = build(output)
    if built["status"] != "PASSED":
        raise RuntimeError("H7.4 native worker/interposer build failed")
    so = output / "libstage3_h7_4_ruckig_interposer.so"
    runs = [run_case(output, f"fresh_process_replay_{index}", True, so) for index in (1, 2, 3)]
    ab_false = run_case(output, "diagnostic_overshoot_false", False, so)
    terminal = finalize(output, before, runs, ab_false)
    missing = [name for name in REQUIRED if not (output / name).is_file()]
    dump_json(output / "stage3_h7_4_artifact_manifest.json", {"missing_required_outputs": missing, "additive": True, "protected_artifacts_written": False, "files": {p.name: {"sha256": sha256(p), "size_bytes": p.stat().st_size} for p in output.iterdir() if p.is_file()}})
    print(json.dumps({"output": str(output), "terminal": terminal, "missing": missing}, ensure_ascii=False))
    return 0 if terminal["STAGE_3_H7_4"] == "PASSED" else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or ROOT / "outputs" / f"stage3_h7_4_native_ruckig_localization_{timestamp}"
    return orchestrate(output.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
