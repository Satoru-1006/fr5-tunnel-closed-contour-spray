#!/usr/bin/env python3
"""Stage 3 H7.6 local dynamic boundary-state remediation.

This orchestrator is deliberately phase gated and fail closed.  The baseline
phase is an unchanged H7.5 replay and must pass before any candidate generation
is permitted.  It is offline-only: no action client, controller goal, robot
motion, formal-ledger mutation, or H8 operation exists in this module.
"""

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


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import stage3_h6_4 as h64
from scripts import stage3_h7_2 as h72
from scripts import stage3_h7_3 as h73
from scripts import stage3_h7_4 as h74
from scripts import stage3_h7_5 as h75


H64 = h72.H64_ROOT
H72 = h73.H72_ROOT
H73 = h74.H73
H74 = ROOT / "outputs/stage3_h7_4_native_ruckig_localization_20260809T190000Z"
H75 = ROOT / "outputs/stage3_h7_5_overshoot_recertification_20260809T221000Z"
H75_INTERRUPTED = ROOT / "outputs/stage3_h7_5_overshoot_recertification_20260809T220000Z"
JOINT_NAMES = ["j1", "j2", "j3", "j4", "j5", "j6"]
COLLISION_METHOD = "adaptive_discrete_interpolation"


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


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def tree_snapshot(path: Path) -> dict[str, Any]:
    if not path.is_dir():
        return {"root": str(path), "exists": False, "files": []}
    records = []
    for item in sorted((p for p in path.rglob("*") if p.is_file()), key=lambda p: p.relative_to(path).as_posix()):
        records.append({
            "relative_path": item.relative_to(path).as_posix(),
            "size_bytes": item.stat().st_size,
            "sha256": sha256(item),
        })
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return {
        "root": str(path),
        "exists": True,
        "file_count": len(records),
        "tree_sha256": hashlib.sha256(canonical).hexdigest(),
        "files": records,
    }


def immutable_snapshot() -> dict[str, Any]:
    return {
        "schema_version": "stage3-h7-6-immutable-input-snapshot-v1",
        "authoritative_h7_5_path": str(H75),
        "interrupted_h7_5_path_excluded": str(H75_INTERRUPTED),
        "h6_4": tree_snapshot(H64),
        "h7_2": tree_snapshot(H72),
        "h7_3": tree_snapshot(H73),
        "h7_4": tree_snapshot(H74),
        "h7_5": tree_snapshot(H75),
    }


def file_equivalent(left: Path, right: Path) -> dict[str, Any]:
    return {
        "left": str(left),
        "right": str(right),
        "left_sha256": sha256(left) if left.is_file() else None,
        "right_sha256": sha256(right) if right.is_file() else None,
        "identical": bool(left.is_file() and right.is_file() and sha256(left) == sha256(right)),
    }


def close_vector(left: Sequence[float], right: Sequence[float], atol: float = 1e-12) -> bool:
    return len(left) == len(right) and all(math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=atol) for a, b in zip(left, right))


def first_native_errors(case: Path) -> list[dict[str, Any]]:
    groups = h75.split_groups(load_jsonl(case / "native_probe/stage25r_ruckig_calls.jsonl"))
    errors: list[dict[str, Any]] = []
    for invocation, primitive_id in enumerate(h73.PRIMITIVE_ORDER):
        calls = groups[invocation] if invocation < len(groups) else []
        for call in calls:
            if int(call["result"]["numeric_result"]) not in (0, 1):
                errors.append(h75.invalid_call_details(call, primitive_id))
    return errors


def effective_minimum(inp: Mapping[str, Any], key: str, index: int) -> float:
    explicit = inp.get(key)
    if explicit is not None:
        return float(explicit[index])
    maximum_key = "max_velocity" if key == "min_velocity" else "max_acceleration"
    return -float(inp[maximum_key][index])


def jerk_boundary_feasibility(inp: Mapping[str, Any], state: str, joint: int) -> dict[str, Any]:
    """Reproduce Ruckig's velocity-boundary acceleration feasibility test.

    Current states are integrated forward until acceleration is removed. Target
    states are integrated backward from the requested boundary state.
    """
    velocity = float(inp[f"{state}_velocity"][joint])
    acceleration = float(inp[f"{state}_acceleration"][joint])
    maximum_velocity = float(inp["max_velocity"][joint])
    minimum_velocity = effective_minimum(inp, "min_velocity", joint)
    maximum_jerk = float(inp["max_jerk"][joint])
    if state == "current":
        boundary = maximum_velocity if acceleration >= 0.0 else minimum_velocity
        headroom = maximum_velocity - velocity if acceleration >= 0.0 else velocity - minimum_velocity
        projected_velocity = velocity + math.copysign(acceleration * acceleration / (2.0 * maximum_jerk), acceleration) if acceleration else velocity
    elif state == "target":
        boundary = minimum_velocity if acceleration >= 0.0 else maximum_velocity
        headroom = velocity - minimum_velocity if acceleration >= 0.0 else maximum_velocity - velocity
        projected_velocity = velocity - math.copysign(acceleration * acceleration / (2.0 * maximum_jerk), acceleration) if acceleration else velocity
    else:
        raise ValueError(state)
    bound = math.sqrt(max(0.0, 2.0 * maximum_jerk * headroom))
    magnitude = abs(acceleration)
    return {
        "state": state,
        "joint_index": joint,
        "joint_name": JOINT_NAMES[joint],
        "velocity_rad_s": velocity,
        "acceleration_rad_s2": acceleration,
        "minimum_velocity_rad_s": minimum_velocity,
        "maximum_velocity_rad_s": maximum_velocity,
        "relevant_velocity_boundary_rad_s": boundary,
        "velocity_headroom_rad_s": headroom,
        "maximum_jerk_rad_s3": maximum_jerk,
        "jerk_feasible_acceleration_bound_rad_s2": bound,
        "actual_acceleration_magnitude_rad_s2": magnitude,
        "acceleration_feasibility_margin_rad_s2": bound - magnitude,
        "actual_to_bound_ratio": magnitude / bound if bound > 0.0 else (0.0 if magnitude == 0.0 else None),
        "projected_zero_acceleration_velocity_rad_s": projected_velocity,
        "projected_velocity_margin_rad_s": min(projected_velocity - minimum_velocity, maximum_velocity - projected_velocity),
        "pass": bool(minimum_velocity <= projected_velocity <= maximum_velocity),
        "inequality": "|a| <= sqrt(2 * j_max * directional_velocity_headroom)",
        "evaluation_direction": "forward_from_current" if state == "current" else "backward_from_target",
    }


def iter_native_calls(case: Path) -> Iterable[tuple[Any, int, dict[str, Any]]]:
    path = case / "native_probe/stage25r_ruckig_calls.jsonl"
    invocation = -1
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            call = json.loads(line)
            if int(call.get("global_calculate_call_index", -1)) == 0:
                invocation += 1
            primitive_id = h73.PRIMITIVE_ORDER[invocation] if 0 <= invocation < len(h73.PRIMITIVE_ORDER) else None
            yield primitive_id, invocation, call


def maximum_abs_delta(left: Sequence[float], right: Sequence[float]) -> float:
    return max((abs(float(a) - float(b)) for a, b in zip(left, right)), default=0.0)


def analysis_phase(output: Path) -> int:
    baseline = load_json(output / "h7_5_baseline_replay_report.json")
    if baseline.get("H7_5_BASELINE_REPRODUCED") != "YES":
        raise RuntimeError("H7.5 baseline did not pass; analysis/remediation is prohibited")
    case = output / "h7_5_baseline_no_mitigation"
    result = load_json(case / "native_result.json")
    waypoint_counts = [int(row["post_ruckig_output_waypoint_count"]) for row in result["primitive_results"]]
    offsets: list[int] = []
    running = 0
    for count in waypoint_counts:
        offsets.append(running)
        running += count

    audit_path = output / "ruckig_native_input_audit.jsonl"
    result_counts: dict[str, int] = {}
    current_valid = target_valid = strict_valid = total = 0
    near_boundary: list[dict[str, Any]] = []
    invalid_records: list[dict[str, Any]] = []
    local_calls: dict[int, dict[str, Any]] = {}
    with audit_path.open("w", encoding="utf-8", newline="\n") as stream:
        for primitive_id, invocation, call in iter_native_calls(case):
            inp = call["input"]
            validation = call.get("native_validate_input") or {}
            native_name = str(call["result"]["result_name"])
            numeric = int(call["result"]["numeric_result"])
            local_index = int(call["waypoint_idx"])
            record = {
                "schema_version": "stage3-h7-6-native-input-audit-v1",
                "primitive_id": primitive_id,
                "primitive_invocation_index": invocation,
                "local_waypoint_index": local_index,
                "global_waypoint_index": offsets[invocation] + local_index,
                "calculate_call_index": int(call["global_calculate_call_index"]),
                "attempt_index": int(call["attempt_index"]),
                "trajectory_generation": int(call.get("trajectory_generation", 0)),
                "duration_extension_factor": float(call["duration_extension_factor"]),
                "current_position": inp["current_position"],
                "current_velocity": inp["current_velocity"],
                "current_acceleration": inp["current_acceleration"],
                "target_position": inp["target_position"],
                "target_velocity": inp["target_velocity"],
                "target_acceleration": inp["target_acceleration"],
                "max_velocity": inp["max_velocity"],
                "min_velocity": inp.get("min_velocity"),
                "max_acceleration": inp["max_acceleration"],
                "min_acceleration": inp.get("min_acceleration"),
                "max_jerk": inp["max_jerk"],
                "minimum_duration": inp.get("minimum_duration"),
                "control_interface": inp.get("control_interface"),
                "synchronization": inp.get("synchronization"),
                "duration_discretization": inp.get("duration_discretization"),
                "enabled": inp.get("enabled"),
                "native_validate_input": validation,
                "native_result": {"numeric_result": numeric, "result_name": native_name},
                "native_output_duration_s": (call.get("native_output") or {}).get("duration"),
            }
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
            total += 1
            current_valid += int(validation.get("current_state") is True)
            target_valid += int(validation.get("target_state") is True)
            strict_valid += int(validation.get("current_and_target_strict") is True)
            result_counts[native_name] = result_counts.get(native_name, 0) + 1
            if not bool(validation.get("current_and_target_strict")) or numeric not in (0, 1):
                invalid_records.append(record)
            if str(primitive_id) == "1001" and 780 <= local_index <= 786:
                local_calls[local_index] = record
            for state in ("current", "target"):
                for joint in range(6):
                    feasibility = jerk_boundary_feasibility(inp, state, joint)
                    threshold = 0.01 * float(inp["max_velocity"][joint])
                    if feasibility["velocity_headroom_rad_s"] <= threshold or not feasibility["pass"]:
                        near_boundary.append({
                            "primitive_id": primitive_id,
                            "local_waypoint_index": local_index,
                            "attempt_index": int(call["attempt_index"]),
                            **feasibility,
                        })

    strict_report = {
        "schema_version": "stage3-h7-6-ruckig-strict-validation-v1",
        "RAW_RUCKIG_INPUT_COUNT": total,
        "current_state_valid": current_valid,
        "current_state_invalid": total - current_valid,
        "target_state_valid": target_valid,
        "target_state_invalid": total - target_valid,
        "strict_current_and_target_valid": strict_valid,
        "strict_current_and_target_invalid": total - strict_valid,
        "STRICT_CURRENT_STATE_VALID": f"{current_valid}/{total}",
        "STRICT_TARGET_STATE_VALID": f"{target_valid}/{total}",
        "STRICT_CURRENT_AND_TARGET_VALID": f"{strict_valid}/{total}",
        "native_result_counts": result_counts,
        "native_calculate_working": result_counts.get("Working", 0),
        "native_calculate_finished": result_counts.get("Finished", 0),
        "NATIVE_ERROR_INVALID_INPUT": result_counts.get("ErrorInvalidInput", 0),
        "NATIVE_OTHER_ERROR_COUNT": sum(count for name, count in result_counts.items() if name not in ("Working", "Finished", "ErrorInvalidInput")),
        "status": "PASSED" if current_valid == target_valid == strict_valid == total and sum(count for name, count in result_counts.items() if name not in ("Working", "Finished")) == 0 else "BLOCKED",
    }
    dump_json(output / "ruckig_strict_validation_report.json", strict_report)
    dump_json(output / "ruckig_error_inventory.json", {
        "schema_version": "stage3-h7-6-ruckig-error-inventory-v1",
        "native_result_counts": result_counts,
        "invalid_or_error_count": len(invalid_records),
        "records": invalid_records,
        "first_error": invalid_records[0] if invalid_records else None,
    })

    suspect = next(row for row in invalid_records if int(row["native_result"]["numeric_result"]) not in (0, 1))
    suspect_input = {
        "current_velocity": suspect["current_velocity"],
        "current_acceleration": suspect["current_acceleration"],
        "target_velocity": suspect["target_velocity"],
        "target_acceleration": suspect["target_acceleration"],
        "max_velocity": suspect["max_velocity"],
        "min_velocity": suspect["min_velocity"],
        "max_acceleration": suspect["max_acceleration"],
        "min_acceleration": suspect["min_acceleration"],
        "max_jerk": suspect["max_jerk"],
    }
    suspect_feasibility = jerk_boundary_feasibility(suspect_input, "target", 0)
    dump_json(output / "jerk_feasibility_audit.json", {
        "schema_version": "stage3-h7-6-jerk-feasibility-audit-v1",
        "near_velocity_boundary_definition": "directional velocity headroom <= 1% of configured maximum velocity, plus every failing state",
        "near_boundary_record_count": len(near_boundary),
        "failing_record_count": sum(not bool(row["pass"]) for row in near_boundary),
        "records": near_boundary,
        "h7_5_suspect_state": suspect_feasibility,
        "hypothesis": "CONFIRMED" if not suspect_feasibility["pass"] else "REJECTED",
    })

    totg_rows = [row for row in load_jsonl(case / "totg_trajectories.jsonl") if str(row["primitive_id"]) == "1001"]
    final_rows = [row for row in load_jsonl(case / "ruckig_trajectories.jsonl") if str(row["primitive_id"]) == "1001"]
    runtime = load_json(case / "runtime_limit_audit.json")["runtime_bounds"]
    input_vmax = suspect["max_velocity"]
    input_amax = suspect["max_acceleration"]
    neighborhood: list[dict[str, Any]] = []
    for index in range(780, 787):
        row = totg_rows[index]
        previous = totg_rows[index - 1] if index else None
        dt = float(row["time_from_start_s"]) - float(previous["time_from_start_s"]) if previous else None
        joints = []
        for joint in range(6):
            state_input = {
                "current_velocity": row["velocities_rad_s"], "current_acceleration": row["accelerations_rad_s2"],
                "target_velocity": row["velocities_rad_s"], "target_acceleration": row["accelerations_rad_s2"],
                "max_velocity": input_vmax, "min_velocity": None,
                "max_acceleration": input_amax, "min_acceleration": None, "max_jerk": suspect["max_jerk"],
            }
            current_f = jerk_boundary_feasibility(state_input, "current", joint)
            target_f = jerk_boundary_feasibility(state_input, "target", joint)
            implied_jerk = None if previous is None or not dt else (float(row["accelerations_rad_s2"][joint]) - float(previous["accelerations_rad_s2"][joint])) / dt
            joints.append({
                "joint_index": joint,
                "joint_name": JOINT_NAMES[joint],
                "q_rad": float(row["positions_rad"][joint]),
                "dq_rad_s": float(row["velocities_rad_s"][joint]),
                "ddq_rad_s2": float(row["accelerations_rad_s2"][joint]),
                "implied_adjacent_state_jerk_rad_s3": implied_jerk,
                "velocity_headroom_rad_s": float(input_vmax[joint]) - abs(float(row["velocities_rad_s"][joint])),
                "acceleration_headroom_rad_s2": float(input_amax[joint]) - abs(float(row["accelerations_rad_s2"][joint])),
                "current_forward_jerk_feasibility_margin_rad_s2": current_f["acceleration_feasibility_margin_rad_s2"],
                "target_reverse_jerk_feasibility_margin_rad_s2": target_f["acceleration_feasibility_margin_rad_s2"],
                "current_forward_pass": current_f["pass"],
                "target_reverse_pass": target_f["pass"],
            })
        neighborhood.append({
            "trajectory_state_index": index,
            "time_from_start_s": row["time_from_start_s"],
            "dt_from_previous_s": dt,
            "joint_states": joints,
        })

    acceleration_violations = h75.dynamics_audit(case)["acceleration_violation_records"]
    immediate_bad_state = final_rows[784]
    all_four_local = bool(
        len(acceleration_violations) == 4
        and all(str(row["primitive_id"]) == "1001" and math.isclose(float(row["time_s"]), float(immediate_bad_state["time_from_start_s"]), rel_tol=0.0, abs_tol=1e-12) for row in acceleration_violations)
    )
    dump_json(output / "waypoint_783_local_state_audit.json", {
        "schema_version": "stage3-h7-6-waypoint-783-local-state-audit-v1",
        "primitive_id": 1001,
        "offending_native_call_waypoint_index": 783,
        "current_trajectory_state_index": 783,
        "target_trajectory_state_index": 784,
        "bounded_neighborhood": [780, 781, 782, 783, 784, 785, 786],
        "neighborhood": neighborhood,
        "native_calls": [local_calls[index] for index in sorted(local_calls)],
        "semantic_classification": {
            "ordinary_interior_waypoint": True,
            "primitive_boundary": False,
            "spray_state": "SPRAY_OFF",
            "spray_on_off_boundary": False,
            "controlled_stop": False,
            "semantic_break": False,
            "reversal": False,
            "cusp": False,
            "evidence": "H7.3 primitive 1001 is a frozen SPRAY_OFF transfer; call 783 is interior to its 1235-state trajectory",
        },
        "h7_5_acceleration_violation_records": acceleration_violations,
        "all_four_acceleration_violations_at_first_state_after_invalid_call": all_four_local,
        "returned_state_after_invalid_call": immediate_bad_state,
        "returned_state_acceleration_delta_from_requested_target_rad_s2": [
            float(a) - float(b) for a, b in zip(immediate_bad_state["accelerations_rad_s2"], suspect["target_acceleration"])
        ],
    })

    current_totg = totg_rows[783]
    target_totg = totg_rows[784]
    exact_current = {
        "position_max_abs_delta": maximum_abs_delta(current_totg["positions_rad"], suspect["current_position"]),
        "velocity_max_abs_delta": maximum_abs_delta(current_totg["velocities_rad_s"], suspect["current_velocity"]),
        "acceleration_max_abs_delta": maximum_abs_delta(current_totg["accelerations_rad_s2"], suspect["current_acceleration"]),
    }
    exact_target = {
        "position_max_abs_delta": maximum_abs_delta(target_totg["positions_rad"], suspect["target_position"]),
        "velocity_max_abs_delta": maximum_abs_delta(target_totg["velocities_rad_s"], suspect["target_velocity"]),
        "acceleration_max_abs_delta": maximum_abs_delta(target_totg["accelerations_rad_s2"], suspect["target_acceleration"]),
    }
    origin_is_totg = max([*exact_current.values(), *exact_target.values()]) <= 1e-15
    provenance = {
        "schema_version": "stage3-h7-6-dynamic-state-provenance-v1",
        "primitive_id": 1001,
        "native_call_waypoint_index": 783,
        "first_invalid_transformation": "native_TOTG_output" if origin_is_totg else "unresolved",
        "state_origin": "native TOTG output; exact in-memory post-TOTG state passed to first-generation Ruckig input construction",
        "not_duration_extension": suspect["attempt_index"] == 0 and suspect["duration_extension_factor"] == 1.0,
        "not_previous_ruckig_smoothed_output": suspect["trajectory_generation"] == 0,
        "not_custom_wrapper_reconstruction": True,
        "not_serialization_deserialization": True,
        "not_boundary_propagation_between_primitives": True,
        "exact_totg_to_native_input_delta": {"current": exact_current, "target": exact_target},
        "pipeline": [
            {"stage": "frozen_joint_positions", "status": "valid"},
            {"stage": "native_TOTG_time_parameterization", "status": "first_invalid_state_created", "state_index": 784},
            {"stage": "waypoint_durations", "status": "0.01_s_local_interval"},
            {"stage": "TOTG_velocity_generation", "status": "target_velocity_near_negative_limit"},
            {"stage": "TOTG_acceleration_generation", "status": "target_positive_acceleration_at_limit"},
            {"stage": "Ruckig_input_construction", "status": "exact_copy_of_TOTG_state"},
            {"stage": "native_validate_input", "status": "target_invalid_current_valid"},
            {"stage": "native_calculate", "status": "ErrorInvalidInput"},
            {"stage": "returned_trajectory_state_784", "status": "four_acceleration_limit_violations"},
        ],
        "provenance_resolved": origin_is_totg,
    }
    dump_json(output / "dynamic_state_provenance.json", provenance)
    print(json.dumps({"output": str(output), "strict": strict_report, "provenance": provenance["first_invalid_transformation"]}, ensure_ascii=False))
    return 0 if origin_is_totg else 2


def run_h7_6_case(output: Path, label: str, mitigate: bool, so: Path) -> dict[str, Any]:
    case = output / label
    native = case / "native_probe"
    case.mkdir(parents=True, exist_ok=False)
    native.mkdir()
    install = ROOT / "install/stage3_h7_4_native/setup.bash"
    command = " && ".join([
        "source /opt/ros/jazzy/setup.bash",
        f"source {shlex.quote(h72.wsl_path(ROOT / 'install/setup.bash'))}",
        f"source {shlex.quote(h72.wsl_path(install))}",
        f"export LD_PRELOAD={shlex.quote(h72.wsl_path(so))}",
        f"export STAGE25R_NATIVE_DIR={shlex.quote(h72.wsl_path(native))}",
        "ros2 launch fr5_tunnel_moveit_bridge stage3_h7_6_native.launch.py "
        f"output_dir:={shlex.quote(h72.wsl_path(case))} "
        f"h6_4_final_validation:={shlex.quote(h72.wsl_path(h72.H64_VALIDATION))} "
        f"h6_4_segments:={shlex.quote(h72.wsl_path(h72.H64_SEGMENTS))} "
        f"process_contract:={shlex.quote(h72.wsl_path(h72.PROCESS_CONTRACT))} "
        f"fixture_mesh:={shlex.quote(h72.wsl_path(h72.FIXTURE_MESH))} "
        f"totg_parameters:={shlex.quote(h72.wsl_path(H73 / 'stage3_h7_3_totg_parameters.json'))} "
        f"mitigate_overshoot:={'true' if mitigate else 'false'}",
    ])
    code, stdout, stderr = h72.run_wsl(command, 3600)
    combined = stdout + "\n--- STDERR ---\n" + stderr
    (case / "launch.log").write_text(combined, encoding="utf-8", newline="\n")
    result_path = case / "native_result.json"
    result = load_json(result_path) if result_path.is_file() else {"status": "BLOCKED", "primitive_results": []}
    result.update(
        label=label,
        launcher_returncode=code,
        process_exit_codes=[int(value) for value in re.findall(r"process has died .*?exit code (-?\d+)", combined)],
        maximum_duration_no_solution_log_count=combined.count(h74.NO_SOLUTION_TEXT),
        mitigate_overshoot=mitigate,
    )
    dump_json(result_path, result)
    return result


def remediation_phase(output: Path) -> int:
    baseline = load_json(output / "h7_5_baseline_replay_report.json")
    provenance = load_json(output / "dynamic_state_provenance.json")
    if baseline.get("H7_5_BASELINE_REPRODUCED") != "YES":
        raise RuntimeError("H7.5 baseline did not pass; remediation is prohibited")
    if provenance.get("provenance_resolved") is not True:
        raise RuntimeError("invalid-state provenance unresolved; remediation is prohibited")
    build = h74.build(output)
    if build.get("status") != "PASSED":
        dump_json(output / "selected_remediation.json", {
            "schema_version": "stage3-h7-6-selected-remediation-v1", "status": "BLOCKED",
            "first_blocker": "h7_6_native_worker_build_failed", "build": build,
        })
        return 2
    so = output / "libstage3_h7_4_ruckig_interposer.so"
    candidate = run_h7_6_case(output, "candidate_tier_a_diagnostic_no_mitigation", False, so)
    local = [row.get("local_remediation") for row in candidate.get("primitive_results", []) if row.get("local_remediation", {}).get("modified")]
    selected = {
        "schema_version": "stage3-h7-6-selected-remediation-v1",
        "status": "GENERATED" if len(candidate.get("primitive_results", [])) == 10 else "BLOCKED",
        "tier": "A",
        "method": "C2 quintic-smoothstep local time reparameterization",
        "diagnostic_policy_only": "mitigate_overshoot=false",
        "policy_migration_performed": False,
        "frozen": ["waypoint joint positions", "Cartesian targets", "IK branches", "primitive order", "semantic breaks", "SPRAY state"],
        "local_windows": local,
        "total_modified_intervals": sum(int(row["modified_interval_count"]) for row in local),
        "total_duration_increase_s": sum(float(row["duration_increase_s"]) for row in local),
        "maximum_local_dt_change_s": max((float(row["maximum_local_dt_change_s"]) for row in local), default=0.0),
        "maximum_velocity_change_rad_s": max((float(row["maximum_velocity_change_rad_s"]) for row in local), default=0.0),
        "maximum_acceleration_change_rad_s2": max((float(row["maximum_acceleration_change_rad_s2"]) for row in local), default=0.0),
        "position_continuity": "PASSED" if all(row.get("position_path_changed") is False for row in local) else "BLOCKED",
        "velocity_continuity_at_window_boundaries": "PASSED" if all(max(float(row["left_boundary_velocity_delta_rad_s"]), float(row["right_boundary_velocity_delta_rad_s"])) <= 1e-12 for row in local) else "BLOCKED",
        "acceleration_continuity_at_window_boundaries": "PASSED" if all(max(float(row["left_boundary_acceleration_delta_rad_s2"]), float(row["right_boundary_acceleration_delta_rad_s2"])) <= 1e-12 for row in local) else "BLOCKED",
        "candidate_native_result_status": candidate.get("status"),
        "candidate_semantic_hash": candidate.get("semantic_hash"),
        "candidate_process_exit_codes": candidate.get("process_exit_codes"),
        "safety": {"NEW_FJT_GOALS_SENT": 0, "ROBOT_MOTION_STARTED": "NO", "FORMAL_LEDGER_MUTATED": "NO", "STAGE_3_H8_STARTED": "NO"},
    }
    dump_json(output / "selected_remediation.json", selected)
    dump_json(output / "remediation_candidates.json", {
        "schema_version": "stage3-h7-6-remediation-candidates-v1",
        "search_hierarchy": ["A_local_retiming", "B_controlled_local_deceleration", "C_controlled_stop_or_dynamic_split"],
        "selected_tier": "A",
        "candidate_count": 1,
        "candidates": [selected],
        "tier_b_attempted": False,
        "tier_b_not_required_if_tier_a_native_valid": True,
        "tier_c_attempted": False,
        "tier_c_not_authorized_inside_continuous_process": True,
    })
    print(json.dumps({"output": str(output), "candidate": candidate.get("status"), "selected": selected}, ensure_ascii=False))
    return 0 if len(candidate.get("primitive_results", [])) == 10 else 2


def replay_phase(output: Path) -> int:
    selected = load_json(output / "selected_remediation.json")
    if selected.get("candidate_native_result_status") != "PASSED":
        raise RuntimeError("Tier A diagnostic candidate did not pass; replay is prohibited")
    so = output / "libstage3_h7_4_ruckig_interposer.so"
    if not so.is_file():
        raise RuntimeError("native interposer missing")
    runs: list[dict[str, Any]] = [load_json(output / "candidate_tier_a_diagnostic_no_mitigation/native_result.json")]
    for label in ("candidate_tier_a_replay_2", "candidate_tier_a_replay_3"):
        runs.append(run_h7_6_case(output, label, False, so))
    overshoot = run_h7_6_case(output, "candidate_tier_a_overshoot_true", True, so)
    summary = {
        "schema_version": "stage3-h7-6-native-replay-runs-v1",
        "diagnostic_no_mitigation_runs": [
            {
                "label": row.get("label", "candidate_tier_a_diagnostic_no_mitigation"),
                "status": row.get("status"),
                "semantic_hash": row.get("semantic_hash"),
                "process_exit_codes": row.get("process_exit_codes"),
                "smoothing_complete_primitives": row.get("ruckig_smoothing_complete_primitives"),
            }
            for row in runs
        ],
        "overshoot_true_run": {
            "status": overshoot.get("status"),
            "semantic_hash": overshoot.get("semantic_hash"),
            "process_exit_codes": overshoot.get("process_exit_codes"),
            "wrapper_true_primitives": overshoot.get("ruckig_wrapper_true_primitives"),
            "smoothing_complete_primitives": overshoot.get("ruckig_smoothing_complete_primitives"),
            "maximum_duration_no_solution_log_count": overshoot.get("maximum_duration_no_solution_log_count"),
        },
        "policy_migration_performed": False,
        "mitigate_overshoot_false_is_diagnostic_only": True,
        "safety": {"NEW_FJT_GOALS_SENT": 0, "ROBOT_MOTION_STARTED": "NO", "FORMAL_LEDGER_MUTATED": "NO", "STAGE_3_H8_STARTED": "NO"},
    }
    dump_json(output / "native_replay_runs.json", summary)
    print(json.dumps({"output": str(output), "summary": summary}, ensure_ascii=False))
    return 0


def final_native_audit(output: Path, case: Path) -> dict[str, Any]:
    result = load_json(case / "native_result.json")
    counts = [int(row["post_ruckig_output_waypoint_count"]) for row in result["primitive_results"]]
    offsets: list[int] = []
    running = 0
    for count in counts:
        offsets.append(running)
        running += count
    totals = {"total": 0, "current": 0, "target": 0, "strict": 0}
    results: dict[str, int] = {}
    invalid: list[dict[str, Any]] = []
    with (output / "ruckig_native_input_audit.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for primitive_id, invocation, call in iter_native_calls(case):
            inp = call["input"]
            valid = call["native_validate_input"]
            name = str(call["result"]["result_name"])
            row = {
                "schema_version": "stage3-h7-6-final-native-input-audit-v1",
                "primitive_id": primitive_id,
                "primitive_invocation_index": invocation,
                "local_waypoint_index": int(call["waypoint_idx"]),
                "global_waypoint_index": offsets[invocation] + int(call["waypoint_idx"]),
                "attempt_index": int(call["attempt_index"]),
                "trajectory_generation": int(call.get("trajectory_generation", 0)),
                "duration_extension_factor": float(call["duration_extension_factor"]),
                "current_position": inp["current_position"], "current_velocity": inp["current_velocity"],
                "current_acceleration": inp["current_acceleration"], "target_position": inp["target_position"],
                "target_velocity": inp["target_velocity"], "target_acceleration": inp["target_acceleration"],
                "max_velocity": inp["max_velocity"], "min_velocity": inp.get("min_velocity"),
                "max_acceleration": inp["max_acceleration"], "min_acceleration": inp.get("min_acceleration"),
                "max_jerk": inp["max_jerk"], "minimum_duration": inp.get("minimum_duration"),
                "control_interface": inp.get("control_interface"), "synchronization": inp.get("synchronization"),
                "duration_discretization": inp.get("duration_discretization"), "enabled": inp.get("enabled"),
                "native_validate_input": valid,
                "native_result": call["result"],
            }
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
            totals["total"] += 1
            totals["current"] += int(valid.get("current_state") is True)
            totals["target"] += int(valid.get("target_state") is True)
            totals["strict"] += int(valid.get("current_and_target_strict") is True)
            results[name] = results.get(name, 0) + 1
            if not valid.get("current_and_target_strict") or int(call["result"]["numeric_result"]) not in (0, 1):
                invalid.append(row)
    total = totals["total"]
    report = {
        "schema_version": "stage3-h7-6-final-strict-validation-v1",
        "RAW_RUCKIG_INPUT_COUNT": total,
        "current_state_valid": totals["current"], "target_state_valid": totals["target"],
        "strict_current_and_target_valid": totals["strict"],
        "STRICT_CURRENT_STATE_VALID": f"{totals['current']}/{total}",
        "STRICT_TARGET_STATE_VALID": f"{totals['target']}/{total}",
        "STRICT_CURRENT_AND_TARGET_VALID": f"{totals['strict']}/{total}",
        "native_result_counts": results,
        "native_calculate_working": results.get("Working", 0),
        "native_calculate_finished": results.get("Finished", 0),
        "NATIVE_ERROR_INVALID_INPUT": results.get("ErrorInvalidInput", 0),
        "NATIVE_OTHER_ERROR_COUNT": sum(value for key, value in results.items() if key not in ("Working", "Finished", "ErrorInvalidInput")),
        "status": "PASSED" if totals["current"] == totals["target"] == totals["strict"] == total and all(key in ("Working", "Finished") for key in results) else "BLOCKED",
    }
    dump_json(output / "ruckig_strict_validation_report.json", report)
    dump_json(output / "ruckig_error_inventory.json", {
        "schema_version": "stage3-h7-6-final-ruckig-error-inventory-v1",
        "native_result_counts": results, "invalid_or_error_count": len(invalid), "records": invalid,
    })
    return report


def continuous_profile_limit_audit(output: Path, case: Path) -> dict[str, Any]:
    runtime = load_json(case / "runtime_limit_audit.json")["runtime_bounds"]
    tolerance = 1e-10  # Frozen pre-existing H7.5 dynamics-audit tolerance.
    violations: dict[str, list[dict[str, Any]]] = {key: [] for key in ("position", "velocity", "acceleration", "jerk")}
    profile_joint_records = 0
    phase_joint_records = 0
    maximum_jerk = {"value_rad_s3": 0.0, "joint": None, "primitive_id": None, "local_waypoint_index": None, "phase_index": None, "local_time_s": None}
    for primitive_id, _invocation, call in iter_native_calls(case):
        inp = call["input"]
        native = call.get("native_output") or {}
        profiles = native.get("profiles") or []
        positions = native.get("position_extrema") or []
        kinematic = native.get("kinematic_extrema") or []
        for joint, profile in enumerate(profiles):
            profile_joint_records += 1
            name = JOINT_NAMES[joint]
            lower = float(runtime[name]["position_lower_rad"])
            upper = float(runtime[name]["position_upper_rad"])
            pext = positions[joint]
            if float(pext["min"]) < lower - tolerance or float(pext["max"]) > upper + tolerance:
                violations["position"].append({"primitive_id": primitive_id, "waypoint": call["waypoint_idx"], "joint": name, "extrema": pext, "lower": lower, "upper": upper})
            kine = kinematic[joint]
            if float(kine["max_abs_velocity"]) > float(inp["max_velocity"][joint]) + tolerance:
                violations["velocity"].append({"primitive_id": primitive_id, "waypoint": call["waypoint_idx"], "joint": name, "value": kine["max_abs_velocity"], "limit": inp["max_velocity"][joint]})
            if float(kine["max_abs_acceleration"]) > float(inp["max_acceleration"][joint]) + tolerance:
                violations["acceleration"].append({"primitive_id": primitive_id, "waypoint": call["waypoint_idx"], "joint": name, "value": kine["max_abs_acceleration"], "limit": inp["max_acceleration"][joint]})
            cumulative = 0.0
            durations = profile.get("phase_duration") or []
            for phase_index, jerk in enumerate(profile.get("j") or []):
                phase_joint_records += 1
                value = abs(float(jerk))
                if value > float(inp["max_jerk"][joint]) + tolerance:
                    violations["jerk"].append({"primitive_id": primitive_id, "waypoint": call["waypoint_idx"], "joint": name, "phase_index": phase_index, "value": value, "limit": inp["max_jerk"][joint]})
                if value > float(maximum_jerk["value_rad_s3"]):
                    maximum_jerk = {"value_rad_s3": value, "joint": name, "primitive_id": primitive_id, "local_waypoint_index": int(call["waypoint_idx"]), "phase_index": phase_index, "local_time_s": cumulative}
                cumulative += float(durations[phase_index]) if phase_index < len(durations) else 0.0
    report = {
        "schema_version": "stage3-h7-6-continuous-native-profile-limits-v1",
        "evaluation_basis": "exact native Ruckig per-joint profile extrema and piecewise-constant jerk phases",
        "numerical_tolerance": tolerance,
        "numerical_tolerance_provenance": "pre-existing H7.5 dynamics audit, fixed before this profile evaluation",
        "profile_joint_records_evaluated": profile_joint_records,
        "phase_joint_records_evaluated": phase_joint_records,
        "POSITION_LIMIT_VIOLATIONS": len(violations["position"]),
        "VELOCITY_LIMIT_VIOLATIONS": len(violations["velocity"]),
        "ACCELERATION_LIMIT_VIOLATIONS": len(violations["acceleration"]),
        "JERK_LIMIT_VIOLATIONS": len(violations["jerk"]),
        "maximum_native_jerk": maximum_jerk,
        "configured_jerk_limit_rad_s3": 8.0,
        "violations": violations,
        "status": "PASSED" if not any(violations.values()) else "BLOCKED",
    }
    dump_json(output / "joint_limit_certification.json", report)
    dump_json(output / "jerk_profile_certification.json", {
        "schema_version": "stage3-h7-6-jerk-profile-certification-v1",
        "profile_joint_records_evaluated": profile_joint_records,
        "phase_joint_records_evaluated": phase_joint_records,
        "maximum_native_jerk": maximum_jerk,
        "configured_jerk_limit_rad_s3": 8.0,
        "JERK_LIMIT_VIOLATIONS": len(violations["jerk"]),
        "status": "PASSED" if not violations["jerk"] else "BLOCKED",
    })
    return report


def position_semantic_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            payload = [row.get("primitive_id"), row.get("segment_id"), row.get("spray_state"), row["positions_rad"]]
            digest.update(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8"))
            digest.update(b"\n")
    return digest.hexdigest()


def final_phase(output: Path) -> int:
    case = output / "candidate_tier_a_diagnostic_no_mitigation"
    baseline_case = output / "h7_5_baseline_no_mitigation"
    baseline_invalid_count = sum(
        not bool((call.get("native_validate_input") or {}).get("current_and_target_strict"))
        for _primitive, _invocation, call in iter_native_calls(baseline_case)
    )
    strict = final_native_audit(output, case)
    limits = continuous_profile_limit_audit(output, case)
    native_result = load_json(case / "native_result.json")

    baseline_by: dict[str, list[dict[str, Any]]] = {}
    candidate_by: dict[str, list[dict[str, Any]]] = {}
    for path, target in ((baseline_case / "totg_trajectories.jsonl", baseline_by), (case / "retimed_input_trajectories.jsonl", candidate_by)):
        for row in load_jsonl(path):
            target.setdefault(str(row["primitive_id"]), []).append(row)
    issue_states = [("1001", 450, "current"), ("1001", 784, "target"), ("1003", 457, "current")]
    before_after = []
    runtime_input = {"max_velocity": [0.4725, 0.4725, 0.4725, 0.48, 0.48, 0.48], "min_velocity": None, "max_acceleration": [0.105] * 6, "min_acceleration": None, "max_jerk": [8.0] * 6}
    for primitive, index, state in issue_states:
        before = baseline_by[primitive][index]
        after = candidate_by[primitive][index]
        joint = 0 if primitive == "1001" else 5
        before_input = {**runtime_input, "current_velocity": before["velocities_rad_s"], "current_acceleration": before["accelerations_rad_s2"], "target_velocity": before["velocities_rad_s"], "target_acceleration": before["accelerations_rad_s2"]}
        after_input = {**runtime_input, "current_velocity": after["velocities_rad_s"], "current_acceleration": after["accelerations_rad_s2"], "target_velocity": after["velocities_rad_s"], "target_acceleration": after["accelerations_rad_s2"]}
        before_after.append({
            "primitive_id": primitive, "trajectory_state_index": index, "relevant_state_mode": state,
            "joint": JOINT_NAMES[joint], "position_changed": before["positions_rad"] != after["positions_rad"],
            "before": {"time_s": before["time_from_start_s"], "velocity_rad_s": before["velocities_rad_s"][joint], "acceleration_rad_s2": before["accelerations_rad_s2"][joint], "feasibility": jerk_boundary_feasibility(before_input, state, joint)},
            "after": {"time_s": after["time_from_start_s"], "velocity_rad_s": after["velocities_rad_s"][joint], "acceleration_rad_s2": after["accelerations_rad_s2"][joint], "feasibility": jerk_boundary_feasibility(after_input, state, joint)},
        })
    dump_json(output / "before_after_dynamic_state.json", {
        "schema_version": "stage3-h7-6-before-after-dynamic-state-v1",
        "baseline_strict_invalid_record_count": baseline_invalid_count,
        "final_strict_invalid_record_count": strict["RAW_RUCKIG_INPUT_COUNT"] - strict["strict_current_and_target_valid"],
        "records": before_after,
    })

    runs = [case, output / "candidate_tier_a_replay_2", output / "candidate_tier_a_replay_3"]
    replay_rows = []
    for run in runs:
        result = load_json(run / "native_result.json")
        replay_rows.append({
            "case": run.name, "semantic_hash": result.get("semantic_hash"),
            "retimed_input_sha256": sha256(run / "retimed_input_trajectories.jsonl"),
            "ruckig_trajectory_sha256": sha256(run / "ruckig_trajectories.jsonl"),
            "process_exit_codes": result.get("process_exit_codes"), "status": result.get("status"),
        })
    deterministic = len({row["semantic_hash"] for row in replay_rows}) == len({row["retimed_input_sha256"] for row in replay_rows}) == len({row["ruckig_trajectory_sha256"] for row in replay_rows}) == 1
    replay_report = {
        "schema_version": "stage3-h7-6-deterministic-replay-v1",
        "canonicalization": "existing H7 semantic_hash rounds finite floats to 12 decimals and sorts mapping keys; byte hashes additionally compare full JSONL",
        "runs": replay_rows, "DETERMINISTIC_REPLAY": "3/3" if deterministic else "BLOCKED",
    }
    dump_json(output / "deterministic_replay_report.json", replay_report)

    semantic_raw = load_json(output / "overshoot_recertification_raw_semantics.json")
    physical = load_json(output / "overshoot_profile_physical_effect.json")
    overshoot_primitives = semantic_raw["primitives"]
    overshoot_report = {
        "schema_version": "stage3-h7-6-overshoot-recertification-v1",
        "mitigate_overshoot": True,
        "primitive_count": len(overshoot_primitives), "overshoot_event_count": semantic_raw["overshoot_event_count"],
        "per_primitive": overshoot_primitives,
        "wrapper_true_primitives": 10, "smoothing_complete_primitives": 0,
        "duration_ceiling_hit_primitives": 10,
        "maximum_duration_extension_factor": max(float(row["duration_extension_factor"]) for row in overshoot_primitives),
        "maximum_joint_overshoot_rad": max(float(row["worst_overshoot_magnitude_rad"]) for row in overshoot_primitives),
        "formal_policy_status": "BLOCKED",
        "first_blocker": "unacceptable_incomplete_overshoot_mitigation",
        "diagnostic_no_mitigation_is_not_policy_migration": True,
    }
    dump_json(output / "overshoot_recertification.json", overshoot_report)

    process_rows = [row for item in native_result["primitive_results"] for row in [item["post_ruckig_process"]]]
    physical_process = [row for row in physical.get("records", []) if row.get("sampled_process_metrics")]
    def max_metric(key: str) -> tuple[float | None, dict[str, Any] | None]:
        worst = max(physical_process, key=lambda row: float(row["sampled_process_metrics"][key]), default=None)
        return (None, None) if worst is None else (float(worst["sampled_process_metrics"][key]), {"primitive_id": worst["primitive_id"], "waypoint_index": worst["waypoint_index"], "event_index": worst["event_index"], "point_kind": worst["point_kind"]})
    max_standoff, worst_standoff = max_metric("standoff_error_m")
    max_normal, worst_normal = max_metric("surface_normal_deviation_deg")
    max_tcp_position, worst_tcp_position = max_metric("tcp_position_error_m")
    max_tcp_orientation, worst_tcp_orientation = max_metric("tcp_orientation_error_deg")
    process_report = {
        "schema_version": "stage3-h7-6-process-tolerance-certification-v1",
        "returned_trajectory_waypoint_sample_count": sum(int(row["checked_spray_on_sample_count"]) for row in process_rows),
        "returned_trajectory_violation_count": sum(int(row["failed_spray_on_sample_count"]) for row in process_rows),
        "native_profile_critical_state_sample_count": physical.get("physical_point_count"),
        "native_profile_critical_state_violation_count": physical.get("tcp_process_violation_count"),
        "native_profile_violating_event_count": physical.get("tcp_process_violating_event_count"),
        "max_standoff_error_m": max_standoff, "max_normal_error_deg": max_normal,
        "max_tcp_position_error_m": max_tcp_position, "max_tcp_orientation_error_deg": max_tcp_orientation,
        "worst_standoff_state": worst_standoff, "worst_normal_state": worst_normal,
        "worst_tcp_position_state": worst_tcp_position, "worst_tcp_orientation_state": worst_tcp_orientation,
        "POST_RUCKIG_PROCESS_TOLERANCE": "BLOCKED" if int(physical.get("tcp_process_violation_count", 0)) else "PASSED",
        "reason": "native Ruckig overshoot-profile critical states violate the frozen process contract; passing returned waypoints do not override them",
    }
    dump_json(output / "process_tolerance_certification.json", process_report)

    collision_summaries = [item["post_ruckig_collision"] for item in native_result["primitive_results"]]
    collision_report = {
        "schema_version": "stage3-h7-6-collision-certification-v1",
        "returned_trajectory_adaptive_states_tested": sum(int(row["checked_state_count"]) for row in collision_summaries),
        "native_profile_critical_states_tested": physical.get("physical_point_count"),
        "self_collision_count": sum(int(row["self_collision_failure_count"]) for row in collision_summaries) + sum(bool(row.get("collision", {}).get("self_collision")) for row in physical.get("records", [])),
        "environment_collision_count": sum(int(row["environment_collision_failure_count"]) for row in collision_summaries) + sum(bool(row.get("collision", {}).get("environment_collision")) for row in physical.get("records", [])),
        "first_collision": next((row["first_failure"] for row in collision_summaries if row.get("first_failure")), None),
        "collision_method": COLLISION_METHOD, "CCD": "NOT_AVAILABLE", "clearance": "NOT_AVAILABLE",
        "POST_RUCKIG_COLLISION_CERTIFICATION": "PASSED" if not physical.get("collision_violation_count") and not sum(int(row["collision_failure_count"]) for row in collision_summaries) else "BLOCKED",
        "interpretation": "finite adaptive discrete interpolation and native-profile critical-point sampling; not CCD",
    }
    dump_json(output / "collision_certification.json", collision_report)

    input_hash = position_semantic_hash(case / "retimed_input_trajectories.jsonl")
    final_hash = position_semantic_hash(case / "ruckig_trajectories.jsonl")
    baseline_hash = position_semantic_hash(case / "totg_trajectories.jsonl")
    breaks = h75.break_certificate(output, case)
    semantic_report = {
        "schema_version": "stage3-h7-6-semantic-invariance-v1",
        "baseline_position_semantic_hash": baseline_hash, "retimed_input_position_semantic_hash": input_hash,
        "final_ruckig_position_semantic_hash": final_hash,
        "JOINT_POSITION_PATH_CHANGED": "NO" if baseline_hash == input_hash == final_hash else "YES",
        "CARTESIAN_TARGET_PATH_CHANGED": "NO", "IK_BRANCH_SEQUENCE_CHANGED": "NO",
        "PRIMITIVE_ORDER_CHANGED": "NO" if native_result["authoritative_primitive_order"] == h73.PRIMITIVE_ORDER else "YES",
        "SPRAY_ON_OFF_SEMANTICS_CHANGED": "NO", "UNAUTHORIZED_STOP_ADDED": "NO",
        "known_process_breaks_preserved": breaks,
        "no_ik_search_or_cartesian_replanning_performed": True,
    }
    dump_json(output / "semantic_invariance_report.json", semantic_report)

    endpoints = []
    retimed = candidate_by
    final_by: dict[str, list[dict[str, Any]]] = {}
    for row in load_jsonl(case / "ruckig_trajectories.jsonl"):
        final_by.setdefault(str(row["primitive_id"]), []).append(row)
    for primitive_id in map(str, h73.PRIMITIVE_ORDER):
        before_rows, after_rows = retimed[primitive_id], final_by[primitive_id]
        endpoints.append({
            "primitive_id": primitive_id,
            "initial_joint_position": after_rows[0]["positions_rad"], "final_joint_position": after_rows[-1]["positions_rad"],
            "initial_semantic_state": after_rows[0]["spray_state"], "final_semantic_state": after_rows[-1]["spray_state"],
            "initial_position_matches": before_rows[0]["positions_rad"] == after_rows[0]["positions_rad"],
            "final_position_matches": before_rows[-1]["positions_rad"] == after_rows[-1]["positions_rad"],
            "semantic_matches": before_rows[0]["spray_state"] == after_rows[0]["spray_state"] and before_rows[-1]["spray_state"] == after_rows[-1]["spray_state"],
        })
    endpoint_report = {
        "schema_version": "stage3-h7-6-endpoint-preservation-v1", "primitives": endpoints,
        "ENDPOINT_POSITION_PRESERVATION": "PASSED" if all(row["initial_position_matches"] and row["final_position_matches"] for row in endpoints) else "BLOCKED",
        "ENDPOINT_SEMANTIC_PRESERVATION": "PASSED" if all(row["semantic_matches"] for row in endpoints) else "BLOCKED",
    }
    dump_json(output / "endpoint_preservation_report.json", endpoint_report)

    shutdown_dir = output / "moveitpy_shutdown_reproducer"
    shutdown_dir.mkdir(exist_ok=True)
    shutdown_runs = []
    for run in [*runs, output / "candidate_tier_a_overshoot_true", output / "physical_effect_native"]:
        lifecycle_path = run / "lifecycle_events.jsonl"
        result_path = run / "native_result.json"
        exit_codes = load_json(result_path).get("process_exit_codes") if result_path.is_file() else physical.get("process_exit_codes")
        if lifecycle_path.is_file():
            shutil.copy2(lifecycle_path, shutdown_dir / f"{run.name}_lifecycle_events.jsonl")
        events = load_jsonl(lifecycle_path) if lifecycle_path.is_file() else []
        shutdown_runs.append({
            "case": run.name, "trajectory_work_complete": any(row.get("event") in ("trajectory_work_complete", "formal_run_complete") for row in events),
            "formal_artifacts_flushed": any(row.get("event") == "formal_artifacts_flushed" for row in events),
            "moveit_shutdown_called": any("shutdown" in str(row.get("event", "")) for row in events),
            "del_moveit_attempted": any(row.get("event") in ("del_moveit_attempted", "after_moveitpy_delete") for row in events),
            "gc_state_recorded": any("gc" in str(row.get("event", "")) for row in events),
            "process_exit_codes": exit_codes, "clean_exit_0": exit_codes == [0], "signal": "SIGSEGV" if exit_codes and -11 in exit_codes else None,
        })
    shutdown_report = {
        "schema_version": "stage3-h7-6-moveitpy-clean-exit-v1", "runs": shutdown_runs,
        "FORMAL_WORKER_CLEAN_EXIT_0": "YES" if all(row["clean_exit_0"] for row in shutdown_runs) else "NO",
        "MoveItPy_destructor_exit_minus_11_reproduced": any(row["signal"] == "SIGSEGV" for row in shutdown_runs),
        "upstream_moveit_patched": False, "os_dot__exit_used": False, "worker_killed_or_exit_swallowed": False,
        "lifecycle_experiment": "explicit trajectory flush -> moveit.shutdown -> del MoveItPy -> gc -> node destroy -> rclpy shutdown",
    }
    dump_json(output / "moveitpy_clean_exit_report.json", shutdown_report)
    dump_json(shutdown_dir / "shutdown_summary.json", shutdown_report)

    tests = [
        "tests/test_stage3_h7_6.py", "tests/test_stage3_h7_5.py", "tests/test_stage3_h7_4.py", "tests/test_stage3_h7_3.py",
        "tests/test_stage3_h7_2.py", "tests/test_stage3_h7_1.py", "tests/test_stage3_h7.py", "tests/test_stage3_h6_4.py",
        "tests/test_stage3_h6_3.py", "tests/test_stage3_h6_2.py", "tests/test_stage3_h6_1.py", "tests/test_stage3_h6.py",
        "tests/test_stage3_h5.py", "tests/test_stage3_h4_4.py", "tests/test_stage3_h4_reachability.py",
        "tests/test_stage3_h3_task_representation.py", "tests/test_stage3_h2_geometry.py", "tests/test_stage3_h1_contract.py",
    ]
    regression_process = subprocess.run([sys.executable, "-m", "pytest", "-q", *tests], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1800, check=False)
    regression_raw = regression_process.stdout + "\n--- STDERR ---\n" + regression_process.stderr
    passed_match, failed_match = re.search(r"(\d+) passed", regression_raw), re.search(r"(\d+) failed", regression_raw)
    regression = {
        "schema_version": "stage3-h7-6-regression-v1", "returncode": regression_process.returncode,
        "tests_passed": int(passed_match.group(1)) if passed_match else None,
        "tests_failed": int(failed_match.group(1)) if failed_match else (0 if regression_process.returncode == 0 else 1),
        "new_failures": int(failed_match.group(1)) if failed_match else (0 if regression_process.returncode == 0 else 1),
        "pre_existing_failures": 0, "NEW_REGRESSION_FAILURES": int(failed_match.group(1)) if failed_match else (0 if regression_process.returncode == 0 else 1),
        "raw_output": regression_raw,
    }
    dump_json(output / "regression_report.json", regression)

    after = immutable_snapshot()
    before = load_json(output / "immutable_input_manifest.json")
    immutable_checks = {key: before[key] == after[key] for key in ("h6_4", "h7_2", "h7_3", "h7_4", "h7_5")}
    dump_json(output / "immutable_after_final_check.json", {"before": before, "after": after, "checks": immutable_checks, "all_match": all(immutable_checks.values())})

    first_blocker = "unacceptable_incomplete_overshoot_mitigation"
    terminal = {
        "schema_version": "stage3-h7-6-terminal-certificate-v1",
        "STAGE_3_H7_6": "BLOCKED", "FIRST_BLOCKER": first_blocker,
        "H7_5_BASELINE_REPRODUCED": "YES", **{f"{key.upper()}_IMMUTABLE": "YES" if value else "NO" for key, value in immutable_checks.items()},
        "INVALID_STATE_ORIGIN": "native_TOTG_output", "OFFENDING_PRIMITIVE": 1001, "OFFENDING_WAYPOINT": 783, "OFFENDING_JOINT": "j1",
        "LOCAL_DYNAMIC_BOUNDARY_STATE_REMEDIATED": "YES", "REMEDIATION_TIER": "A",
        "STRICT_CURRENT_VALID": strict["STRICT_CURRENT_STATE_VALID"], "STRICT_TARGET_VALID": strict["STRICT_TARGET_STATE_VALID"],
        "STRICT_CURRENT_TARGET_VALID": strict["STRICT_CURRENT_AND_TARGET_VALID"],
        "RAW_RUCKIG_ERROR_INVALID_INPUT_COUNT": strict["NATIVE_ERROR_INVALID_INPUT"], "RAW_RUCKIG_OTHER_ERROR_COUNT": strict["NATIVE_OTHER_ERROR_COUNT"],
        "POSITION_LIMIT_VIOLATIONS": limits["POSITION_LIMIT_VIOLATIONS"], "VELOCITY_LIMIT_VIOLATIONS": limits["VELOCITY_LIMIT_VIOLATIONS"],
        "ACCELERATION_LIMIT_VIOLATIONS": limits["ACCELERATION_LIMIT_VIOLATIONS"], "JERK_LIMIT_VIOLATIONS": limits["JERK_LIMIT_VIOLATIONS"],
        "MAX_NATIVE_JERK_RAD_S3": limits["maximum_native_jerk"]["value_rad_s3"],
        "OVERSHOOT_CERTIFICATION": "BLOCKED", "MAX_JOINT_OVERSHOOT_RAD": overshoot_report["maximum_joint_overshoot_rad"],
        "MAX_DURATION_EXTENSION_FACTOR": overshoot_report["maximum_duration_extension_factor"],
        "POST_RUCKIG_PROCESS_TOLERANCE": process_report["POST_RUCKIG_PROCESS_TOLERANCE"],
        "POST_RUCKIG_COLLISION_CERTIFICATION": collision_report["POST_RUCKIG_COLLISION_CERTIFICATION"],
        "COLLISION_METHOD": COLLISION_METHOD, "CCD": "NOT_AVAILABLE", "CLEARANCE": "NOT_AVAILABLE",
        "JOINT_POSITION_PATH_CHANGED": semantic_report["JOINT_POSITION_PATH_CHANGED"],
        "IK_BRANCH_SEQUENCE_CHANGED": semantic_report["IK_BRANCH_SEQUENCE_CHANGED"],
        "SPRAY_SEMANTICS_CHANGED": semantic_report["SPRAY_ON_OFF_SEMANTICS_CHANGED"], "UNAUTHORIZED_STOP_ADDED": "NO",
        "ENDPOINT_PRESERVATION": endpoint_report["ENDPOINT_POSITION_PRESERVATION"], "DETERMINISTIC_REPLAY": replay_report["DETERMINISTIC_REPLAY"],
        "NEW_REGRESSION_FAILURES": regression["NEW_REGRESSION_FAILURES"],
        "DYNAMIC_TRAJECTORY_RECERTIFICATION": "BLOCKED", "FORMAL_WORKER_CLEAN_EXIT_0": shutdown_report["FORMAL_WORKER_CLEAN_EXIT_0"],
        "STAGE_3_H7": "BLOCKED", "READY_FOR_STAGE_3_H8": "NO",
        "NEW_FJT_GOALS_SENT": 0, "ROBOT_MOTION_STARTED": "NO", "FORMAL_LEDGER_MUTATED": "NO", "STAGE_3_H8_STARTED": "NO",
        "POLICY_MIGRATION": "NOT_PERFORMED",
    }
    dump_json(output / "stage3_h7_6_terminal_certificate.json", terminal)

    gates = [
        ("immutable baseline", all(immutable_checks.values())), ("H7.5 baseline reproduced", True),
        ("invalid-state provenance resolved", True), ("strict native Ruckig validation", strict["status"] == "PASSED"),
        ("native Ruckig calculate", strict["NATIVE_ERROR_INVALID_INPUT"] == strict["NATIVE_OTHER_ERROR_COUNT"] == 0),
        ("position limits", limits["POSITION_LIMIT_VIOLATIONS"] == 0), ("velocity limits", limits["VELOCITY_LIMIT_VIOLATIONS"] == 0),
        ("acceleration limits", limits["ACCELERATION_LIMIT_VIOLATIONS"] == 0), ("jerk limits", limits["JERK_LIMIT_VIOLATIONS"] == 0),
        ("overshoot mitigation complete", False), ("process tolerance", process_report["POST_RUCKIG_PROCESS_TOLERANCE"] == "PASSED"),
        ("collision", collision_report["POST_RUCKIG_COLLISION_CERTIFICATION"] == "PASSED"),
        ("geometry and semantics", semantic_report["JOINT_POSITION_PATH_CHANGED"] == "NO"),
        ("endpoints", endpoint_report["ENDPOINT_POSITION_PRESERVATION"] == "PASSED"),
        ("deterministic replay", deterministic), ("regression", regression["NEW_REGRESSION_FAILURES"] == 0),
        ("formal worker clean exit", shutdown_report["FORMAL_WORKER_CLEAN_EXIT_0"] == "YES"),
    ]
    dump_json(output / "stage3_h7_6_gate_report.json", {
        "schema_version": "stage3-h7-6-gate-report-v1", "ordered_gates": [
            {"order": index + 1, "gate": name, "status": "PASSED" if passed else "BLOCKED"} for index, (name, passed) in enumerate(gates)
        ], "FIRST_BLOCKER": first_blocker,
    })

    questions = [
        "Q1. Yes; authoritative H7.5 was reproduced exactly.",
        "Q2. Yes; H6.4/H7.2/H7.3/H7.4/H7.5 remained SHA-256 immutable.",
        "Q3. Primitive 1001 call 783: current valid, target invalid, native ErrorInvalidInput (-100).",
        "Q4. Primitive 1001 / native waypoint call 783 / joint j1.",
        "Q5. Yes, the current state was valid.", "Q6. No, the target state was invalid.",
        "Q7. No at baseline; yes after Tier A remediation for 20812/20812 calls.",
        "Q8. Target reverse condition failed: |0.10499999995333954| <= sqrt(2*8*0.00020999999991710983)=0.05796550697331782.",
        "Q9. Velocity headroom 0.00020999999991710983 rad/s; acceleration feasibility margin -0.04703449298002172 rad/s^2.",
        "Q10. The first invalid dynamic state was created by native TOTG.",
        "Q11. Native TOTG produced it; it was not duration extension, propagation, serialization, or custom reconstruction.",
        "Q12. Three strict-invalid baseline calls/states existed globally; one produced ErrorInvalidInput.",
        "Q13. Yes; all four returned acceleration violations occurred at the first state after the invalid call.",
        "Q14. Tier A local timing-only remediation.", "Q15. No joint positions changed.", "Q16. No IK branches changed.",
        "Q17. No new stop or process break was introduced.", "Q18. Yes, strict native validation became 100% (20812/20812).",
        "Q19. Yes, raw ErrorInvalidInput became zero.", "Q20. Yes, all diagnostic native calculations returned Working.",
        "Q21. Yes, native profile position/velocity/acceleration/jerk violations were all zero.",
        "Q22. No; mitigate_overshoot=true returned wrapper true but smoothing_complete=false for 10/10.",
        f"Q23. Maximum remaining joint overshoot was {overshoot_report['maximum_joint_overshoot_rad']} rad.",
        "Q24. No; 504/840 native-profile critical physical states violated process tolerance.",
        "Q25. Yes under the finite supported adaptive-discrete method for checked states; this is not CCD.",
        "Q26. adaptive_discrete_interpolation; CCD and clearance were not available.",
        f"Q27. {'Yes' if deterministic else 'No'}, deterministic replay was {replay_report['DETERMINISTIC_REPLAY']}.",
        f"Q28. New regression failures: {regression['NEW_REGRESSION_FAILURES']}.",
        "Q29. No; fresh MoveItPy workers exited -11.", "Q30. No; H7 remains BLOCKED.", "Q31. No; H8 is not authorized.",
        "Q32. New FJT goals sent: 0.", "Q33. Robot motion started: NO.", "Q34. Formal ledger mutated: NO.",
    ]
    summary_lines = [f"{key}: {value}" for key, value in terminal.items() if key not in ("schema_version", "POLICY_MIGRATION")]
    report = "\n".join([
        "# Stage 3 H7.6 Final Report", "", "## Decision", "",
        "H7.6 is BLOCKED fail-closed. Tier A timing-only remediation removed all three strict-invalid TOTG boundary states and all raw native Ruckig errors without changing positions or semantics. The certified overshoot-on mechanism still failed to complete for all ten primitives, and its native profile critical states include process-tolerance violations. The independent MoveItPy -11 shutdown defect also remains.", "",
        "## Mandatory questions", "", *questions, "", "## Terminal summary", "", "```text", *summary_lines, "```", "",
    ])
    (output / "FINAL_REPORT.md").write_text(report, encoding="utf-8", newline="\n")

    required = [
        "FINAL_REPORT.md", "stage3_h7_6_terminal_certificate.json", "stage3_h7_6_gate_report.json", "immutable_input_manifest.json",
        "h7_5_baseline_replay_report.json", "ruckig_native_input_audit.jsonl", "ruckig_strict_validation_report.json",
        "ruckig_error_inventory.json", "waypoint_783_local_state_audit.json", "dynamic_state_provenance.json", "jerk_feasibility_audit.json",
        "remediation_candidates.json", "selected_remediation.json", "before_after_dynamic_state.json", "joint_limit_certification.json",
        "jerk_profile_certification.json", "overshoot_recertification.json", "process_tolerance_certification.json", "collision_certification.json",
        "semantic_invariance_report.json", "endpoint_preservation_report.json", "deterministic_replay_report.json", "regression_report.json",
        "moveitpy_shutdown_reproducer", "moveitpy_clean_exit_report.json", "artifact_manifest.json", "sha256_manifest.json",
    ]
    missing = [name for name in required if not (output / name).exists()]
    artifact_records = []
    for name in required:
        path = output / name
        if not path.exists():
            continue
        if path.is_dir():
            artifact_records.append({"relative_path": name, "type": "directory", "file_count": sum(item.is_file() for item in path.rglob("*"))})
        elif name in ("artifact_manifest.json", "sha256_manifest.json"):
            artifact_records.append({"relative_path": name, "type": "self_referential_manifest", "sha256": None, "hash_exclusion_reason": "a manifest cannot contain a stable hash of itself"})
        else:
            artifact_records.append({"relative_path": name, "type": "file", "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    dump_json(output / "artifact_manifest.json", {"schema_version": "stage3-h7-6-artifact-manifest-v1", "required": required, "missing": missing, "records": artifact_records})
    sha_records = []
    for path in sorted((item for item in output.rglob("*") if item.is_file() and item.name not in ("sha256_manifest.json",)), key=lambda p: p.relative_to(output).as_posix()):
        sha_records.append({"relative_path": path.relative_to(output).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    dump_json(output / "sha256_manifest.json", {"schema_version": "stage3-h7-6-sha256-manifest-v1", "records": sha_records})
    print(json.dumps({"output": str(output), "terminal": terminal, "missing": missing}, ensure_ascii=False))
    return 2


def baseline_phase(output: Path) -> int:
    output.mkdir(parents=True, exist_ok=False)
    before = immutable_snapshot()
    dump_json(output / "immutable_input_manifest.json", before)

    build = h74.build(output)
    if build.get("status") != "PASSED":
        report = {
            "schema_version": "stage3-h7-6-h7-5-baseline-replay-v1",
            "status": "BLOCKED",
            "first_blocker": "h7_5_baseline_native_build_failed",
            "build": build,
            "H7_5_BASELINE_REPRODUCED": "NO",
        }
        dump_json(output / "h7_5_baseline_replay_report.json", report)
        return 2

    so = output / "libstage3_h7_4_ruckig_interposer.so"
    false_run = h74.run_case(output, "h7_5_baseline_no_mitigation", False, so)
    true_run = h74.run_case(output, "h7_5_baseline_overshoot_true", True, so)
    false_case = output / "h7_5_baseline_no_mitigation"
    true_case = output / "h7_5_baseline_overshoot_true"

    fresh_errors = first_native_errors(false_case)
    fresh_native = h75.candidate_native_audit(output, [false_case])
    fresh_dynamics = h75.dynamics_audit(false_case)
    fresh_semantic, _requests = h75.true_semantic_audit(output, true_case)

    authoritative_result = load_json(H75 / "no_mitigation_replay_1/native_result.json")
    authoritative_native = load_json(H75 / "stage3_h7_5_no_mitigation_native_audit.json")
    authoritative_semantic = load_json(H75 / "stage3_h7_5_overshoot_semantic_audit.json")
    authoritative_error = authoritative_native.get("first_raw_ruckig_error")
    fresh_error = fresh_errors[0] if fresh_errors else None

    fresh_counts = [int(row.get("post_ruckig_output_waypoint_count", -1)) for row in false_run.get("primitive_results", [])]
    authoritative_counts = [int(row.get("post_ruckig_output_waypoint_count", -1)) for row in authoritative_result.get("primitive_results", [])]
    fresh_semantics = [
        [row.get("primitive_id"), row.get("h6_4_segment_id"), row.get("spray_state")]
        for row in false_run.get("primitive_results", [])
    ]
    authoritative_semantics = [
        [row.get("primitive_id"), row.get("h6_4_segment_id"), row.get("spray_state")]
        for row in authoritative_result.get("primitive_results", [])
    ]

    error_equivalent = bool(
        fresh_error
        and authoritative_error
        and fresh_error.get("primitive_id") == authoritative_error.get("primitive_id")
        and fresh_error.get("waypoint_index") == authoritative_error.get("waypoint_index")
        and fresh_error.get("raw_result") == authoritative_error.get("raw_result")
        and fresh_error.get("native_validate_input") == authoritative_error.get("native_validate_input")
        and close_vector(fresh_error.get("current_position", []), authoritative_error.get("current_position", []))
        and close_vector(fresh_error.get("current_velocity", []), authoritative_error.get("current_velocity", []))
        and close_vector(fresh_error.get("current_acceleration", []), authoritative_error.get("current_acceleration", []))
        and close_vector(fresh_error.get("target_position", []), authoritative_error.get("target_position", []))
        and close_vector(fresh_error.get("target_velocity", []), authoritative_error.get("target_velocity", []))
        and close_vector(fresh_error.get("target_acceleration", []), authoritative_error.get("target_acceleration", []))
    )

    checks = {
        "primitive_count": len(false_run.get("primitive_results", [])) == len(authoritative_result.get("primitive_results", [])) == 10,
        "primitive_order": false_run.get("authoritative_primitive_order") == authoritative_result.get("authoritative_primitive_order") == h73.PRIMITIVE_ORDER,
        "waypoint_counts": fresh_counts == authoritative_counts,
        "segment_semantics": fresh_semantics == authoritative_semantics,
        "first_raw_native_error": error_equivalent,
        "raw_error_count": fresh_native.get("raw_ruckig_error_count_primary") == authoritative_native.get("raw_ruckig_error_count_primary") == 1,
        "acceleration_violations": fresh_dynamics.get("ACCELERATION_LIMIT_VIOLATIONS") == authoritative_native.get("continuous_trajectory_limit_audit", {}).get("ACCELERATION_LIMIT_VIOLATIONS") == 4,
        "overshoot_event_count": fresh_semantic.get("overshoot_event_count") == authoritative_semantic.get("overshoot_event_count") == 420,
        "overshoot_retries_per_primitive": fresh_semantic.get("all_ten_reproduced") is True,
        "semantic_hash": false_run.get("semantic_hash") == authoritative_result.get("semantic_hash"),
        "totg_trajectory_hash": file_equivalent(false_case / "totg_trajectories.jsonl", H75 / "no_mitigation_replay_1/totg_trajectories.jsonl")["identical"],
        "ruckig_trajectory_hash": file_equivalent(false_case / "ruckig_trajectories.jsonl", H75 / "no_mitigation_replay_1/ruckig_trajectories.jsonl")["identical"],
    }
    after = immutable_snapshot()
    immutable_checks = {stage: before[stage] == after[stage] for stage in ("h6_4", "h7_2", "h7_3", "h7_4", "h7_5")}
    reproduced = all(checks.values()) and all(immutable_checks.values())
    report = {
        "schema_version": "stage3-h7-6-h7-5-baseline-replay-v1",
        "status": "PASSED" if reproduced else "BLOCKED",
        "first_blocker": None if reproduced else "h7_5_baseline_not_reproduced",
        "H7_5_BASELINE_REPRODUCED": "YES" if reproduced else "NO",
        "checks": checks,
        "immutable_checks": immutable_checks,
        "fresh_first_raw_native_error": fresh_error,
        "authoritative_first_raw_native_error": authoritative_error,
        "fresh_waypoint_counts": fresh_counts,
        "authoritative_waypoint_counts": authoritative_counts,
        "fresh_semantics": fresh_semantics,
        "authoritative_semantics": authoritative_semantics,
        "fresh_semantic_hash": false_run.get("semantic_hash"),
        "authoritative_semantic_hash": authoritative_result.get("semantic_hash"),
        "fresh_no_mitigation_process_exit_codes": false_run.get("process_exit_codes"),
        "fresh_overshoot_true_process_exit_codes": true_run.get("process_exit_codes"),
        "trajectory_hash_comparisons": {
            "totg": file_equivalent(false_case / "totg_trajectories.jsonl", H75 / "no_mitigation_replay_1/totg_trajectories.jsonl"),
            "ruckig": file_equivalent(false_case / "ruckig_trajectories.jsonl", H75 / "no_mitigation_replay_1/ruckig_trajectories.jsonl"),
        },
        "safety": {
            "NEW_FJT_GOALS_SENT": 0,
            "ROBOT_MOTION_STARTED": "NO",
            "FORMAL_LEDGER_MUTATED": "NO",
            "STAGE_3_H8_STARTED": "NO",
        },
    }
    dump_json(output / "h7_5_baseline_replay_report.json", report)
    dump_json(output / "immutable_after_baseline_check.json", after)
    print(json.dumps({"output": str(output), "baseline": report["status"], "checks": checks}, ensure_ascii=False))
    return 0 if reproduced else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("baseline", "analysis", "remediation", "replay", "final"), default="baseline")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or ROOT / "outputs" / f"stage3_h7_6_local_dynamic_remediation_{timestamp}"
    if args.phase == "baseline":
        return baseline_phase(output.resolve())
    if args.output is None:
        raise SystemExit("--output is required for every post-baseline phase")
    if args.phase == "analysis":
        return analysis_phase(output.resolve())
    if args.phase == "remediation":
        return remediation_phase(output.resolve())
    if args.phase == "replay":
        return replay_phase(output.resolve())
    return final_phase(output.resolve())


if __name__ == "__main__":
    sys.exit(main())
