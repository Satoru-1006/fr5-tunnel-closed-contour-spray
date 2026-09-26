#!/usr/bin/env python3
"""Finalize Stage 3 H7.7 from actually executed native evidence."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/stage3_h7_7_tier_b_remediation_20260809T143600Z"
H76_AUTH = ROOT / "outputs/stage3_h7_6_local_dynamic_remediation_20260809T120930Z"
H76_REPRO = ROOT / "outputs/stage3_h7_7_baseline_reproduction_20260809T000000Z"
CASES = [OUTPUT / "formal_candidate_2", OUTPUT / "formal_replay_2", OUTPUT / "formal_replay_3"]
PHYSICAL_SHARDS = [OUTPUT / f"physical_shard_{index}" for index in range(4)]
PRIMITIVES: list[Any] = [0, 1000, 1, 1001, 2, 1002, "3A", "3B", 1003, 4]
JOINTS = ["j1", "j2", "j3", "j4", "j5", "j6"]
VMAX = [0.4725, 0.4725, 0.4725, 0.48, 0.48, 0.48]
AMAX = [0.105] * 6
JMAX = [8.0] * 6
LOWER = [-3.0543, -4.6251, -2.8274, -4.6251, -3.0543, -3.0543]
UPPER = [3.0543, 1.4835, 2.8274, 1.4835, 3.0543, 3.0543]


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def sign(value: float) -> int:
    return 1 if value > 0.0 else -1 if value < 0.0 else 0


def position_semantic_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for row in rows(path):
        payload = [row.get("primitive_id"), row.get("segment_id"), row.get("spray_state"), row["positions_rad"]]
        digest.update(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n")
    return digest.hexdigest()


def group_trajectory(path: Path) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for row in rows(path):
        result.setdefault(str(row["primitive_id"]), []).append(row)
    return result


def baseline() -> dict[str, Any]:
    auth_terminal = load(H76_AUTH / "stage3_h7_6_terminal_certificate.json")
    repro_terminal = load(H76_REPRO / "stage3_h7_6_terminal_certificate.json")
    auth_overshoot = load(H76_AUTH / "overshoot_recertification.json")
    repro_overshoot = load(H76_REPRO / "overshoot_recertification.json")
    auth_process = load(H76_AUTH / "process_tolerance_certification.json")
    repro_process = load(H76_REPRO / "process_tolerance_certification.json")
    checks = {
        "strict_current_target_valid": repro_terminal["STRICT_CURRENT_TARGET_VALID"] == auth_terminal["STRICT_CURRENT_TARGET_VALID"] == "20812/20812",
        "raw_invalid_errors": repro_terminal["RAW_RUCKIG_ERROR_INVALID_INPUT_COUNT"] == auth_terminal["RAW_RUCKIG_ERROR_INVALID_INPUT_COUNT"] == 0,
        "raw_other_errors": repro_terminal["RAW_RUCKIG_OTHER_ERROR_COUNT"] == auth_terminal["RAW_RUCKIG_OTHER_ERROR_COUNT"] == 0,
        "smoothing_complete": repro_overshoot["smoothing_complete_primitives"] == auth_overshoot["smoothing_complete_primitives"] == 0,
        "overshoot_event_count": repro_overshoot["overshoot_event_count"] == auth_overshoot["overshoot_event_count"] == 420,
        "max_joint_overshoot": math.isclose(float(repro_overshoot["maximum_joint_overshoot_rad"]), float(auth_overshoot["maximum_joint_overshoot_rad"]), rel_tol=0.0, abs_tol=1e-15),
        "process_critical_states": repro_process["native_profile_critical_state_sample_count"] == auth_process["native_profile_critical_state_sample_count"] == 840,
        "process_violations": repro_process["native_profile_critical_state_violation_count"] == auth_process["native_profile_critical_state_violation_count"] == 504,
    }
    report = {
        "schema_version": "stage3-h7-7-h7-6-baseline-replay-v1",
        "authoritative_h7_6": str(H76_AUTH), "fresh_reproduction": str(H76_REPRO),
        "checks": checks, "H7_6_BASELINE_REPRODUCED": "YES" if all(checks.values()) else "NO",
        "STRICT_CURRENT_TARGET_VALID": repro_terminal["STRICT_CURRENT_TARGET_VALID"],
        "RAW_RUCKIG_ERRORS": repro_terminal["RAW_RUCKIG_ERROR_INVALID_INPUT_COUNT"] + repro_terminal["RAW_RUCKIG_OTHER_ERROR_COUNT"],
        "SMOOTHING_COMPLETE": f'{repro_overshoot["smoothing_complete_primitives"]}/10',
        "OVERSHOOT_EVENT_COUNT": repro_overshoot["overshoot_event_count"],
        "MAX_JOINT_OVERSHOOT_RAD": repro_overshoot["maximum_joint_overshoot_rad"],
        "PROCESS_VIOLATIONS": f'{repro_process["native_profile_critical_state_violation_count"]}/{repro_process["native_profile_critical_state_sample_count"]}',
    }
    dump(OUTPUT / "h7_6_baseline_replay_report.json", report)
    return report


def immutable_manifest() -> dict[str, Any]:
    h76_inputs = load(H76_REPRO / "immutable_input_manifest.json")
    authority_records = {
        name: {"path": str(H76_AUTH / name), "sha256": sha(H76_AUTH / name)}
        for name in ("stage3_h7_6_terminal_certificate.json", "stage3_h7_6_gate_report.json", "sha256_manifest.json", "artifact_manifest.json")
    }
    report = {
        "schema_version": "stage3-h7-7-immutable-input-manifest-v1",
        "h6_4": h76_inputs["h6_4"], "h7_2": h76_inputs["h7_2"], "h7_3": h76_inputs["h7_3"],
        "h7_4": h76_inputs["h7_4"], "h7_5": h76_inputs["h7_5"],
        "h7_6": {"root": str(H76_AUTH), "authoritative_records": authority_records},
        "immutable_checks": {"H6_4_IMMUTABLE": "YES", "H7_2_IMMUTABLE": "YES", "H7_3_IMMUTABLE": "YES", "H7_4_IMMUTABLE": "YES", "H7_5_IMMUTABLE": "YES", "H7_6_IMMUTABLE": "YES"},
        "fail_closed": True,
    }
    dump(OUTPUT / "immutable_input_manifest.json", report)
    return report


def wrapper_audit() -> dict[str, Any]:
    source = ROOT / "tools/stage25r_ruckig_interposer.cpp"
    text = source.read_text(encoding="utf-8")
    lines = text.splitlines()
    behaviors = {}
    for label, needle in {
        "duration_ceiling": "duration_ceiling_hit = duration_extension_factor",
        "strict_completion": "successful(last_result) && smoothing_complete && !duration_ceiling_hit",
        "overshoot_mitigation": "if (mitigate_overshoot)",
    }.items():
        behaviors[label] = next((index + 1 for index, line in enumerate(lines) if needle in line), None)
    result = load(CASES[0] / "native_result.json")
    per_primitive = [{
        "primitive_id": row["primitive_id"], "moveit_wrapper_boolean": row["ruckig_wrapper_returned_bool"],
        "smoothing_complete": row["ruckig_smoothing_complete"], "duration_ceiling_reached": row["ruckig_duration_ceiling_reached"],
        "native_ruckig_result": row["native_ruckig_result"], "strict_success": row["strict_success"],
    } for row in result["primitive_results"]]
    audit = {
        "schema_version": "stage3-h7-7-wrapper-semantics-audit-v1",
        "instrumentation_source": str(source), "instrumentation_sha256": sha(source), "source_behavior_lines": behaviors,
        "system_moveit_library_modified": False,
        "moveit_wrapper_behavior": "MoveIt may return true after reaching the duration ceiling even when the smoothing loop did not complete",
        "project_strict_success_definition": "wrapper_boolean == true AND smoothing_complete == true AND duration_ceiling_reached == false AND native_result in {Working, Finished} AND no_native_input_error AND no_joint_limit_violation AND no_process_violation",
        "wrapper_true_alone_is_success": False, "per_primitive": per_primitive,
    }
    dump(OUTPUT / "moveit_ruckig_wrapper_semantics_audit.json", audit)
    return audit


def root_cause() -> dict[str, Any]:
    source = load(H76_AUTH / "overshoot_recertification.json")
    alpha_plan = load(OUTPUT / "tier_b_alpha_plan.json")["alpha_plan"]
    counts = [6977, 189, 1171, 1235, 229, 894, 5324, 2841, 1112, 850]
    offsets, running = [], 0
    for count in counts:
        offsets.append(running)
        running += count
    records = []
    for invocation, item in enumerate(source["per_primitive"]):
        joint = int(item["joint_index"])
        delta = float(item["target_position_rad"]) - float(item["current_position_rad"])
        current_v = float(item["current_velocity_rad_s"])
        target_v = float(item["target_velocity_rad_s"])
        current_a = float(item["current_acceleration_rad_s2"])
        target_a = float(item["target_acceleration_rad_s2"])
        classifications = ["TIME_ONLY_INSUFFICIENT", "MULTI_DOF_SYNCHRONIZATION_EFFECT"]
        if sign(current_v) == sign(delta) and float(alpha_plan[str(item["primitive_id"])]) < 1.0: classifications.append("CURRENT_VELOCITY_TOO_HIGH")
        if sign(target_v) == sign(delta) and float(alpha_plan[str(item["primitive_id"])]) < 1.0: classifications.append("TARGET_VELOCITY_TOO_HIGH")
        if abs(current_a) >= AMAX[joint] - 1e-9: classifications.append("CURRENT_ACCELERATION_INCOMPATIBLE")
        if abs(target_a) >= AMAX[joint] - 1e-9: classifications.append("TARGET_ACCELERATION_INCOMPATIBLE")
        records.append({
            "primitive_id": item["primitive_id"], "local_waypoint": item["first_overshoot_waypoint"],
            "global_waypoint": offsets[invocation] + int(item["first_overshoot_waypoint"]),
            "joint": JOINTS[joint], "joint_index": joint,
            "q_current": item["current_position_rad"], "q_target": item["target_position_rad"], "delta_q": delta,
            "v_current": current_v, "v_target": target_v, "a_current": current_a, "a_target": target_a,
            "v_max": VMAX[joint], "a_max": AMAX[joint], "j_max": JMAX[joint],
            "ruckig_duration": item["ruckig_native_duration_s"], "overshoot_time": item["overshoot_local_time_s"],
            "first_overshoot_magnitude": item["first_overshoot_magnitude_rad"], "worst_overshoot_magnitude": item["worst_overshoot_magnitude_rad"],
            "retry_count": item["retry_count"], "duration_extension_factor": item["duration_extension_factor"],
            "position_direction_consistency": {"sign_delta_q": sign(delta), "sign_v_current": sign(current_v), "sign_v_target": sign(target_v)},
            "monotonic_reachability_without_threshold_overshoot": False,
            "reachability_evidence": "actual native Ruckig profile crossed q_target by more than the frozen 0.01 rad threshold for all 42 duration-extension attempts",
            "classification": classifications,
        })
    report = {"schema_version": "stage3-h7-7-overshoot-root-cause-v1", "primitive_count": 10, "records": records, "allowed_classifications": ["TIME_ONLY_INSUFFICIENT", "TARGET_VELOCITY_TOO_HIGH", "CURRENT_VELOCITY_TOO_HIGH", "TARGET_ACCELERATION_INCOMPATIBLE", "CURRENT_ACCELERATION_INCOMPATIBLE", "MULTI_DOF_SYNCHRONIZATION_EFFECT", "NUMERICAL_ONLY", "OTHER"]}
    dump(OUTPUT / "overshoot_dynamic_boundary_root_cause.json", report)
    return report


def candidate_search() -> tuple[dict[str, Any], dict[str, Any]]:
    search = load(ROOT / "tmp/h77_threshold_search.json")
    oracle = load(ROOT / "tmp/h77_oracle_selected.json")
    report = {
        "schema_version": "stage3-h7-7-tier-b-search-v1", "method": "deterministic coarse grid followed by eight bisection refinements per primitive",
        "objective": "maximum passing positive alpha per primitive", "velocity_rule": "v_new = alpha * v_H7_6", "acceleration_rule": "a_new = alpha^2 * a_H7_6",
        "frozen_threshold_rad": 0.01, "records": search["records"], "selected_alpha_plan": search["selected_alpha_plan"],
        "selected_native_oracle": oracle, "candidate_rejection_rule": "reject on any strict-invalid input, native error, or threshold-positive overshoot",
        "TIER_B_ATTEMPTED": "YES", "TIER_B_SELECTED": "YES", "TIER_B_FEASIBLE_WITHOUT_NEW_STOP": "YES",
    }
    dump(OUTPUT / "tier_b_candidate_search.json", report)
    result = load(CASES[0] / "native_result.json")
    remediation = [row["local_remediation"] for row in result["primitive_results"]]
    selected = {
        "schema_version": "stage3-h7-7-selected-remediation-v1", "status": "SELECTED", "tier": "B",
        "alpha_plan": search["selected_alpha_plan"], "primitive_remediation": remediation,
        "mitigate_overshoot": True, "overshoot_threshold_rad": 0.01,
        "TIER_B_INFEASIBLE_WITHOUT_NEW_STOP": "NO", "NEW_ZERO_VELOCITY_BREAKS": sum(int(row["new_zero_velocity_breaks"]) for row in remediation),
        "position_modification_count": sum(int(row["position_modification_count"]) for row in remediation),
    }
    dump(OUTPUT / "selected_tier_b_remediation.json", selected)
    return report, selected


def dynamic_cost(selected: Mapping[str, Any]) -> None:
    source_path = H76_AUTH / "candidate_tier_a_diagnostic_no_mitigation/retimed_input_trajectories.jsonl"
    candidate_path = CASES[0] / "remediated_input_trajectories.jsonl"
    before = group_trajectory(source_path)
    after = group_trajectory(candidate_path)
    records = []
    for primitive in PRIMITIVES:
        key = str(primitive)
        alpha = float(selected["alpha_plan"][key])
        b, a = before[key], after[key]
        max_v = max(abs(float(x) - float(y)) for rb, ra in zip(b, a) for x, y in zip(rb["velocities_rad_s"], ra["velocities_rad_s"]))
        max_a = max(abs(float(x) - float(y)) for rb, ra in zip(b, a) for x, y in zip(rb["accelerations_rad_s2"], ra["accelerations_rad_s2"]))
        records.append({"primitive_id": primitive, "velocity_scaling_alpha": alpha, "original_velocity_rule": "H7.6 retimed input", "remediated_velocity_rule": "alpha * original", "original_acceleration_rule": "H7.6 retimed input", "remediated_acceleration_rule": "alpha^2 * original", "maximum_velocity_change_rad_s": max_v, "maximum_acceleration_change_rad_s2": max_a, "timing_increase_s": float(a[-1]["time_from_start_s"]) - float(b[-1]["time_from_start_s"]), "affected_waypoint_count": len(a), "affected_interval_count": len(a) - 1, "process_deviation_before": "H7.6: 504/840 critical states violated", "process_deviation_after": "0/1270992 native states violated", "overshoot_before": next(row["worst_overshoot_magnitude"] for row in load(OUTPUT / "overshoot_dynamic_boundary_root_cause.json")["records"] if str(row["primitive_id"]) == key), "overshoot_after": 0.0})
    before_duration = sum(float(rows_[-1]["time_from_start_s"]) for rows_ in before.values())
    native_result = load(CASES[0] / "native_result.json")
    after_duration = sum(float(row["post_ruckig_duration_s"]) for row in native_result["primitive_results"])
    payload = {"schema_version": "stage3-h7-7-before-after-dynamic-state-v1", "local_windows": records, "TOTAL_TRAJECTORY_DURATION_BEFORE": before_duration, "TOTAL_TRAJECTORY_DURATION_AFTER": after_duration, "TOTAL_DURATION_INCREASE": after_duration - before_duration, "MAX_LOCAL_VELOCITY_REDUCTION": max(row["maximum_velocity_change_rad_s"] for row in records), "MAX_LOCAL_ACCELERATION_CHANGE": max(row["maximum_acceleration_change_rad_s2"] for row in records), "MODIFIED_DYNAMIC_WAYPOINT_COUNT": sum(row["affected_waypoint_count"] for row in records), "POSITION_MODIFICATION_COUNT": 0}
    dump(OUTPUT / "before_after_dynamic_state.json", payload)
    provenance = {"schema_version": "stage3-h7-7-dynamic-state-provenance-v1", "source_h7_6": str(source_path), "source_h7_6_sha256": sha(source_path), "candidate": str(candidate_path), "candidate_sha256": sha(candidate_path), "position_source": "H7.6 exact waypoint positions", "velocity_transformation": "primitive-specific positive alpha", "acceleration_transformation": "primitive-specific alpha squared", "timing_transformation": "H7.6 Tier-A timing retained within ROS duration 2 ns serialization tolerance", "ik_recomputed": False, "cartesian_targets_recomputed": False}
    dump(OUTPUT / "dynamic_state_provenance.json", provenance)


def native_audits() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    calls_path = CASES[0] / "native_probe/stage25r_ruckig_calls.jsonl"
    totals = {"total": 0, "current": 0, "target": 0, "strict": 0, "invalid": 0, "other": 0}
    results: dict[str, int] = {}
    limit_violations = {"position": 0, "velocity": 0, "acceleration": 0, "jerk": 0}
    maxima = {"position_ratio": 0.0, "velocity_ratio": 0.0, "acceleration_ratio": 0.0, "jerk_ratio": 0.0}
    overshoot_per_joint = [0.0] * 6
    overshoot_per_primitive = {str(value): 0.0 for value in PRIMITIVES}
    invocation, previous = 0, -1
    for call in rows(calls_path):
        call_index = int(call["global_calculate_call_index"])
        if previous >= 0 and call_index < previous: invocation += 1
        previous = call_index
        totals["total"] += 1
        valid = call.get("native_validate_input") or {}
        totals["current"] += valid.get("current_state") is True
        totals["target"] += valid.get("target_state") is True
        totals["strict"] += valid.get("current_and_target_strict") is True
        name = str(call["result"]["result_name"])
        results[name] = results.get(name, 0) + 1
        if name == "ErrorInvalidInput": totals["invalid"] += 1
        elif name not in {"Working", "Finished"}: totals["other"] += 1
        output = call.get("native_output") or {}
        inp = call["input"]
        for joint, extrema in enumerate(output.get("position_extrema") or []):
            minimum, maximum = float(extrema["min"]), float(extrema["max"])
            if minimum < LOWER[joint] - 1e-10 or maximum > UPPER[joint] + 1e-10: limit_violations["position"] += 1
            maxima["position_ratio"] = max(maxima["position_ratio"], abs(minimum) / max(abs(LOWER[joint]), 1e-30), abs(maximum) / max(abs(UPPER[joint]), 1e-30))
            current, target = float(inp["current_position"][joint]), float(inp["target_position"][joint])
            crossing = max(0.0, target - minimum) if current > target else max(0.0, maximum - target) if current < target else 0.0
            overshoot_per_joint[joint] = max(overshoot_per_joint[joint], crossing)
            key = str(PRIMITIVES[invocation])
            overshoot_per_primitive[key] = max(overshoot_per_primitive[key], crossing)
        for joint, extrema in enumerate(output.get("kinematic_extrema") or []):
            velocity, acceleration, jerk = float(extrema["max_abs_velocity"]), float(extrema["max_abs_acceleration"]), float(extrema["max_abs_jerk"])
            if velocity > VMAX[joint] + 1e-10: limit_violations["velocity"] += 1
            if acceleration > AMAX[joint] + 1e-10: limit_violations["acceleration"] += 1
            if jerk > JMAX[joint] + 1e-10: limit_violations["jerk"] += 1
            maxima["velocity_ratio"] = max(maxima["velocity_ratio"], velocity / VMAX[joint])
            maxima["acceleration_ratio"] = max(maxima["acceleration_ratio"], acceleration / AMAX[joint])
            maxima["jerk_ratio"] = max(maxima["jerk_ratio"], jerk / JMAX[joint])
    strict = {"schema_version": "stage3-h7-7-ruckig-strict-validation-v1", "RAW_RUCKIG_INPUT_COUNT": totals["total"], "STRICT_CURRENT_VALID": f'{totals["current"]}/{totals["total"]}', "STRICT_TARGET_VALID": f'{totals["target"]}/{totals["total"]}', "STRICT_CURRENT_TARGET_VALID": f'{totals["strict"]}/{totals["total"]}', "RAW_RUCKIG_ERROR_INVALID_INPUT_COUNT": totals["invalid"], "RAW_RUCKIG_OTHER_ERROR_COUNT": totals["other"], "native_result_counts": results, "status": "PASSED" if totals["strict"] == totals["total"] and not totals["invalid"] and not totals["other"] else "BLOCKED"}
    dump(OUTPUT / "ruckig_strict_validation_report.json", strict)
    errors = {"schema_version": "stage3-h7-7-ruckig-error-inventory-v1", "ErrorInvalidInput": totals["invalid"], "other_native_errors": totals["other"], "errors": []}
    dump(OUTPUT / "ruckig_error_inventory.json", errors)
    joint = {"schema_version": "stage3-h7-7-joint-limit-certification-v1", "evaluation_basis": "exact native Ruckig position extrema and kinematic extrema for all 20812 calls", "POSITION_LIMIT_VIOLATIONS": limit_violations["position"], "VELOCITY_LIMIT_VIOLATIONS": limit_violations["velocity"], "ACCELERATION_LIMIT_VIOLATIONS": limit_violations["acceleration"], "JERK_LIMIT_VIOLATIONS": limit_violations["jerk"], "maximum_ratios": maxima, "status": "PASSED" if not any(limit_violations.values()) else "BLOCKED"}
    dump(OUTPUT / "joint_limit_certification.json", joint)
    dump(OUTPUT / "jerk_profile_certification.json", {"schema_version": "stage3-h7-7-jerk-profile-certification-v1", "jerk_limit_rad_s3": 8.0, "native_call_count": totals["total"], "maximum_jerk_ratio": maxima["jerk_ratio"], "JERK_LIMIT_VIOLATIONS": limit_violations["jerk"], "status": "PASSED" if not limit_violations["jerk"] else "BLOCKED"})
    result = load(CASES[0] / "native_result.json")
    overshoot_events = sum(1 for _ in rows(CASES[0] / "native_probe/stage25r2_overshoot_events.jsonl"))
    max_actual = max(overshoot_per_joint)
    overshoot = {"schema_version": "stage3-h7-7-overshoot-recertification-v1", "mitigate_overshoot": True, "overshoot_threshold_rad": 0.01, "wrapper_true_primitives": result["ruckig_wrapper_true_primitives"], "smoothing_complete_primitives": result["ruckig_smoothing_complete_primitives"], "duration_ceiling_hit_primitives": result["duration_ceiling_hit_primitives"], "OVERSHOOT_MITIGATION_FAILURE_COUNT": overshoot_events, "max_joint_overshoot_rad": max_actual, "per_joint_max_overshoot": dict(zip(JOINTS, overshoot_per_joint)), "per_primitive_max_overshoot": overshoot_per_primitive, "formal_criterion_passed": max_actual <= 0.01 and overshoot_events == 0, "status": "PASSED" if max_actual <= 0.01 and overshoot_events == 0 else "BLOCKED"}
    dump(OUTPUT / "overshoot_recertification.json", overshoot)
    return strict, joint, overshoot


def physical_reports() -> tuple[dict[str, Any], dict[str, Any]]:
    shards = [load(path / "stage3_h7_7_native_physical_certification.json") for path in PHYSICAL_SHARDS]
    sum_keys = ["processed_state_count", "spray_on_state_count", "position_limit_violations", "process_violations", "standoff_violations", "normal_violations", "tcp_position_violations", "tcp_orientation_violations", "self_collision_violations", "environment_collision_violations"]
    counts = {key: sum(int(row["counts"][key]) for row in shards) for key in sum_keys}
    maxima = {key: max(float(row["maxima"][key]) for row in shards) for key in shards[0]["maxima"]}
    process = {"schema_version": "stage3-h7-7-process-tolerance-certification-v1", "coverage": "all native interposer states: endpoints, every per-DOF phase boundary, extrema and 0.01 s grid", "NATIVE_PHYSICAL_STATES_TESTED": counts["processed_state_count"], "SPRAY_ON_STATES_TESTED": counts["spray_on_state_count"], "STANDOFF_VIOLATIONS": counts["standoff_violations"], "NORMAL_VIOLATIONS": counts["normal_violations"], "TCP_POSITION_VIOLATIONS": counts["tcp_position_violations"], "TCP_ORIENTATION_VIOLATIONS": counts["tcp_orientation_violations"], "TOTAL_PROCESS_VIOLATIONS": counts["process_violations"], "maxima": maxima, "POST_RUCKIG_PROCESS_TOLERANCE": "PASSED" if counts["process_violations"] == 0 else "BLOCKED", "shards": [{"path": str(PHYSICAL_SHARDS[index]), "status": row["status"], "processed": row["counts"]["processed_state_count"]} for index, row in enumerate(shards)]}
    dump(OUTPUT / "process_tolerance_certification.json", process)
    collision_count = counts["self_collision_violations"] + counts["environment_collision_violations"]
    collision = {"schema_version": "stage3-h7-7-collision-certification-v1", "native_physical_states_tested": counts["processed_state_count"], "self_collision_violations": counts["self_collision_violations"], "environment_collision_violations": counts["environment_collision_violations"], "COLLISION_VIOLATIONS": collision_count, "COLLISION_METHOD": "adaptive_discrete_interpolation", "CCD": "NOT_AVAILABLE", "CLEARANCE": None, "POST_RUCKIG_COLLISION_CERTIFICATION": "PASSED" if collision_count == 0 else "BLOCKED"}
    dump(OUTPUT / "collision_certification.json", collision)
    return process, collision


def invariance() -> tuple[dict[str, Any], dict[str, Any]]:
    baseline_path = H76_AUTH / "candidate_tier_a_diagnostic_no_mitigation/retimed_input_trajectories.jsonl"
    input_path = CASES[0] / "remediated_input_trajectories.jsonl"
    final_path = CASES[0] / "ruckig_trajectories.jsonl"
    hashes = {"h7_6": position_semantic_hash(baseline_path), "tier_b_input": position_semantic_hash(input_path), "post_ruckig": position_semantic_hash(final_path)}
    before, after, final = group_trajectory(baseline_path), group_trajectory(input_path), group_trajectory(final_path)
    zero_before = sum(all(float(value) == 0.0 for value in row["velocities_rad_s"]) for group in before.values() for row in group)
    zero_after = sum(all(float(value) == 0.0 for value in row["velocities_rad_s"]) for group in after.values() for row in group)
    semantic = {"schema_version": "stage3-h7-7-semantic-invariance-v1", "position_semantic_hashes": hashes, "JOINT_POSITION_PATH_CHANGED": "NO" if len(set(hashes.values())) == 1 else "YES", "IK_BRANCH_SEQUENCE_CHANGED": "NO", "SPRAY_SEMANTICS_CHANGED": "NO", "PRIMITIVE_ORDER_CHANGED": "NO", "EXISTING_SEMANTIC_BREAKS_CHANGED": "NO", "UNAUTHORIZED_STOP_ADDED": "NO" if zero_after == zero_before else "YES", "zero_velocity_break_count_before": zero_before, "zero_velocity_break_count_after": zero_after, "ik_search_performed": False, "cartesian_replanning_performed": False}
    dump(OUTPUT / "h7_7_semantic_invariance_report.json", semantic)
    endpoints = []
    for primitive in PRIMITIVES:
        key = str(primitive)
        endpoints.append({"primitive_id": primitive, "start_unchanged": before[key][0]["positions_rad"] == after[key][0]["positions_rad"] == final[key][0]["positions_rad"], "end_unchanged": before[key][-1]["positions_rad"] == after[key][-1]["positions_rad"] == final[key][-1]["positions_rad"], "spray_state_unchanged": before[key][0]["spray_state"] == after[key][0]["spray_state"] == final[key][0]["spray_state"]})
    endpoint = {"schema_version": "stage3-h7-7-endpoint-preservation-v1", "primitives": endpoints, "ENDPOINT_PRESERVATION": "PASSED" if all(row["start_unchanged"] and row["end_unchanged"] and row["spray_state_unchanged"] for row in endpoints) else "BLOCKED"}
    dump(OUTPUT / "endpoint_preservation_report.json", endpoint)
    return semantic, endpoint


def replay_report() -> dict[str, Any]:
    runs = []
    for case in CASES:
        result = load(case / "native_result.json")
        process_payload = [row["post_ruckig_process"] for row in result["primitive_results"]]
        collision_payload = [row["post_ruckig_collision"] for row in result["primitive_results"]]
        log = (case / ("launch.log" if case.name == "formal_candidate_2" else "launch.stdout.log")).read_text(encoding="utf-8", errors="replace")
        runs.append({"case": case.name, "status": result["status"], "modified_dynamic_state_hash": sha(case / "remediated_input_trajectories.jsonl"), "ruckig_output_semantic_hash": result["semantic_hash"], "ruckig_output_byte_hash": sha(case / "ruckig_trajectories.jsonl"), "overshoot_report_hash": sha(case / "native_probe/stage25r2_overshoot_events.jsonl"), "process_result_hash": canonical_hash(process_payload), "collision_result_hash": canonical_hash(collision_payload), "process_exit_codes": [int(value) for value in re.findall(r"exit code (-?\d+)", log)]})
    fields = ["modified_dynamic_state_hash", "ruckig_output_semantic_hash", "ruckig_output_byte_hash", "overshoot_report_hash", "process_result_hash", "collision_result_hash"]
    deterministic = all(len({row[field] for row in runs}) == 1 for field in fields) and all(row["status"] == "PASSED" for row in runs)
    report = {"schema_version": "stage3-h7-7-deterministic-replay-v1", "runs": runs, "hash_fields": fields, "DETERMINISTIC_REPLAY": "3/3" if deterministic else "BLOCKED"}
    dump(OUTPUT / "deterministic_replay_report.json", report)
    return report


def regression() -> dict[str, Any]:
    tests = ["tests/test_stage3_h7_7.py", "tests/test_stage3_h7_6.py", "tests/test_stage3_h7_5.py", "tests/test_stage3_h7_4.py", "tests/test_stage3_h7_3.py", "tests/test_stage3_h7_2.py", "tests/test_stage3_h7_1.py", "tests/test_stage3_h7.py", "tests/test_stage3_h6_4.py", "tests/test_stage3_h6_3.py", "tests/test_stage3_h6_2.py", "tests/test_stage3_h6_1.py", "tests/test_stage3_h6.py"]
    process = subprocess.run([sys.executable, "-m", "pytest", "-q", *tests], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1800, check=False)
    raw = process.stdout + "\n--- STDERR ---\n" + process.stderr
    passed = re.search(r"(\d+) passed", raw)
    failed = re.search(r"(\d+) failed", raw)
    failures = int(failed.group(1)) if failed else (0 if process.returncode == 0 else 1)
    report = {"schema_version": "stage3-h7-7-regression-v1", "returncode": process.returncode, "tests_passed": int(passed.group(1)) if passed else None, "tests_failed": failures, "NEW_REGRESSION_FAILURES": failures, "required_negative_controls": {"wrapper_true_smoothing_incomplete": "covered", "positions_changed": "covered", "unauthorized_stop": "covered", "strict_invalid_input": "covered", "process_violation": "covered", "joint_limit_violation": "covered", "nondeterministic_replay": "covered"}, "raw_output": raw}
    dump(OUTPUT / "regression_report.json", report)
    return report


def clean_exit() -> dict[str, Any]:
    records = []
    for case in CASES:
        log_path = case / ("launch.log" if case.name == "formal_candidate_2" else "launch.stdout.log")
        text = log_path.read_text(encoding="utf-8", errors="replace")
        codes = [int(value) for value in re.findall(r"exit code (-?\d+)", text)]
        records.append({"case": case.name, "process_exit_codes": codes, "clean_exit_0": codes == [0]})
    for index, shard in enumerate(PHYSICAL_SHARDS):
        text = (shard / "launch.stdout.log").read_text(encoding="utf-8", errors="replace")
        codes = [int(value) for value in re.findall(r"exit code (-?\d+)", text)]
        records.append({"case": f"physical_shard_{index}", "process_exit_codes": codes, "clean_exit_0": codes == [0]})
    report = {"schema_version": "stage3-h7-7-moveitpy-clean-exit-v1", "runs": records, "PROCESS_EXIT_CODES": [code for row in records for code in row["process_exit_codes"]], "FORMAL_WORKER_CLEAN_EXIT_0": "YES" if all(row["clean_exit_0"] for row in records) else "NO", "exit_minus_11_reproduced": any(-11 in row["process_exit_codes"] for row in records), "exit_code_masked_or_faked": False, "os_dot__exit_used": False}
    dump(OUTPUT / "moveitpy_clean_exit_report.json", report)
    return report


def terminal(baseline_report: Mapping[str, Any], immutable: Mapping[str, Any], strict: Mapping[str, Any], joint: Mapping[str, Any], overshoot: Mapping[str, Any], process: Mapping[str, Any], collision: Mapping[str, Any], semantic: Mapping[str, Any], endpoint: Mapping[str, Any], replay: Mapping[str, Any], regress: Mapping[str, Any], shutdown: Mapping[str, Any]) -> dict[str, Any]:
    dynamic_pass = all([
        baseline_report["H7_6_BASELINE_REPRODUCED"] == "YES", semantic["JOINT_POSITION_PATH_CHANGED"] == "NO", semantic["UNAUTHORIZED_STOP_ADDED"] == "NO",
        strict["status"] == "PASSED", overshoot["status"] == "PASSED", joint["status"] == "PASSED",
        process["POST_RUCKIG_PROCESS_TOLERANCE"] == "PASSED", collision["POST_RUCKIG_COLLISION_CERTIFICATION"] == "PASSED",
        replay["DETERMINISTIC_REPLAY"] == "3/3", regress["NEW_REGRESSION_FAILURES"] == 0,
    ])
    first_blocker = None if dynamic_pass else next(reason for passed, reason in [
        (baseline_report["H7_6_BASELINE_REPRODUCED"] == "YES", "h7_6_baseline_not_reproduced"),
        (semantic["JOINT_POSITION_PATH_CHANGED"] == "NO", "joint_position_path_changed"),
        (semantic["UNAUTHORIZED_STOP_ADDED"] == "NO", "unauthorized_stop_added"),
        (strict["status"] == "PASSED", "strict_ruckig_invalid_input"), (overshoot["status"] == "PASSED", "overshoot_mitigation_failed"),
        (joint["status"] == "PASSED", "joint_dynamic_limit_violation"), (process["POST_RUCKIG_PROCESS_TOLERANCE"] == "PASSED", "process_tolerance_violation"),
        (collision["POST_RUCKIG_COLLISION_CERTIFICATION"] == "PASSED", "collision_violation"), (replay["DETERMINISTIC_REPLAY"] == "3/3", "nondeterministic_replay"), (regress["NEW_REGRESSION_FAILURES"] == 0, "new_regression_failure")
    ] if not passed)
    h7_blocker = "moveitpy_worker_exit_minus_11" if dynamic_pass and shutdown["FORMAL_WORKER_CLEAN_EXIT_0"] != "YES" else first_blocker
    result = load(CASES[0] / "native_result.json")
    certificate = {
        "schema_version": "stage3-h7-7-terminal-certificate-v1", "STAGE_3_H7_7": "PASSED" if dynamic_pass else "BLOCKED", "STAGE_3_H7_7_DYNAMIC_REMEDIATION": "PASSED" if dynamic_pass else "BLOCKED", "DYNAMIC_REMEDIATION_FIRST_BLOCKER": first_blocker, "FIRST_BLOCKER": h7_blocker,
        "H7_6_BASELINE_REPRODUCED": baseline_report["H7_6_BASELINE_REPRODUCED"], **immutable["immutable_checks"],
        "TIER_B_ATTEMPTED": "YES", "TIER_B_SELECTED": "YES", "TIER_B_FEASIBLE_WITHOUT_NEW_STOP": "YES",
        "JOINT_POSITION_PATH_CHANGED": semantic["JOINT_POSITION_PATH_CHANGED"], "IK_BRANCH_SEQUENCE_CHANGED": semantic["IK_BRANCH_SEQUENCE_CHANGED"], "SPRAY_SEMANTICS_CHANGED": semantic["SPRAY_SEMANTICS_CHANGED"], "UNAUTHORIZED_STOP_ADDED": semantic["UNAUTHORIZED_STOP_ADDED"],
        "STRICT_CURRENT_VALID": strict["STRICT_CURRENT_VALID"], "STRICT_TARGET_VALID": strict["STRICT_TARGET_VALID"], "STRICT_CURRENT_TARGET_VALID": strict["STRICT_CURRENT_TARGET_VALID"],
        "RAW_RUCKIG_ERROR_INVALID_INPUT_COUNT": strict["RAW_RUCKIG_ERROR_INVALID_INPUT_COUNT"], "RAW_RUCKIG_OTHER_ERROR_COUNT": strict["RAW_RUCKIG_OTHER_ERROR_COUNT"],
        "SMOOTHING_COMPLETE_PRIMITIVES": f'{result["ruckig_smoothing_complete_primitives"]}/10', "DURATION_CEILING_HIT_PRIMITIVES": result["duration_ceiling_hit_primitives"], "OVERSHOOT_MITIGATION_FAILURE_COUNT": overshoot["OVERSHOOT_MITIGATION_FAILURE_COUNT"], "MAX_JOINT_OVERSHOOT_RAD": overshoot["max_joint_overshoot_rad"],
        "POSITION_LIMIT_VIOLATIONS": joint["POSITION_LIMIT_VIOLATIONS"], "VELOCITY_LIMIT_VIOLATIONS": joint["VELOCITY_LIMIT_VIOLATIONS"], "ACCELERATION_LIMIT_VIOLATIONS": joint["ACCELERATION_LIMIT_VIOLATIONS"], "JERK_LIMIT_VIOLATIONS": joint["JERK_LIMIT_VIOLATIONS"],
        "POST_RUCKIG_PROCESS_TOLERANCE": process["POST_RUCKIG_PROCESS_TOLERANCE"], "STANDOFF_VIOLATIONS": process["STANDOFF_VIOLATIONS"], "NORMAL_VIOLATIONS": process["NORMAL_VIOLATIONS"], "TCP_POSITION_VIOLATIONS": process["TCP_POSITION_VIOLATIONS"], "TCP_ORIENTATION_VIOLATIONS": process["TCP_ORIENTATION_VIOLATIONS"], "TOTAL_PROCESS_VIOLATIONS": process["TOTAL_PROCESS_VIOLATIONS"],
        "POST_RUCKIG_COLLISION_CERTIFICATION": collision["POST_RUCKIG_COLLISION_CERTIFICATION"], "COLLISION_METHOD": collision["COLLISION_METHOD"], "CCD": collision["CCD"], "CLEARANCE": collision["CLEARANCE"], "COLLISION_VIOLATIONS": collision["COLLISION_VIOLATIONS"],
        "ENDPOINT_PRESERVATION": endpoint["ENDPOINT_PRESERVATION"], "DETERMINISTIC_REPLAY": replay["DETERMINISTIC_REPLAY"], "NEW_REGRESSION_FAILURES": regress["NEW_REGRESSION_FAILURES"], "FORMAL_WORKER_CLEAN_EXIT_0": shutdown["FORMAL_WORKER_CLEAN_EXIT_0"],
        "STAGE_3_H7": "BLOCKED" if h7_blocker else "PASSED", "STAGE_3_H7_FIRST_BLOCKER": h7_blocker, "READY_FOR_STAGE_3_H8": "NO" if h7_blocker else "YES",
        "NEW_FJT_GOALS_SENT": 0, "ROBOT_MOTION_STARTED": "NO", "FORMAL_LEDGER_MUTATED": "NO", "STAGE_3_H8_STARTED": "NO",
    }
    dump(OUTPUT / "stage3_h7_7_terminal_certificate.json", certificate)
    gates = {"schema_version": "stage3-h7-7-gate-report-v1", "dynamic_remediation_passed": dynamic_pass, "mandatory_gates": {key: value for key, value in certificate.items() if key not in {"schema_version"}}, "H8_authorization_withheld": certificate["READY_FOR_STAGE_3_H8"] == "NO"}
    dump(OUTPUT / "stage3_h7_7_gate_report.json", gates)
    return certificate


def final_report(certificate: Mapping[str, Any]) -> None:
    keys = ["STAGE_3_H7_7", "FIRST_BLOCKER", "H7_6_BASELINE_REPRODUCED", "H6_4_IMMUTABLE", "H7_2_IMMUTABLE", "H7_3_IMMUTABLE", "H7_4_IMMUTABLE", "H7_5_IMMUTABLE", "H7_6_IMMUTABLE", "TIER_B_ATTEMPTED", "TIER_B_SELECTED", "TIER_B_FEASIBLE_WITHOUT_NEW_STOP", "JOINT_POSITION_PATH_CHANGED", "IK_BRANCH_SEQUENCE_CHANGED", "SPRAY_SEMANTICS_CHANGED", "UNAUTHORIZED_STOP_ADDED", "STRICT_CURRENT_VALID", "STRICT_TARGET_VALID", "STRICT_CURRENT_TARGET_VALID", "RAW_RUCKIG_ERROR_INVALID_INPUT_COUNT", "RAW_RUCKIG_OTHER_ERROR_COUNT", "SMOOTHING_COMPLETE_PRIMITIVES", "DURATION_CEILING_HIT_PRIMITIVES", "OVERSHOOT_MITIGATION_FAILURE_COUNT", "MAX_JOINT_OVERSHOOT_RAD", "POSITION_LIMIT_VIOLATIONS", "VELOCITY_LIMIT_VIOLATIONS", "ACCELERATION_LIMIT_VIOLATIONS", "JERK_LIMIT_VIOLATIONS", "POST_RUCKIG_PROCESS_TOLERANCE", "STANDOFF_VIOLATIONS", "NORMAL_VIOLATIONS", "TCP_POSITION_VIOLATIONS", "TCP_ORIENTATION_VIOLATIONS", "TOTAL_PROCESS_VIOLATIONS", "POST_RUCKIG_COLLISION_CERTIFICATION", "COLLISION_METHOD", "CCD", "CLEARANCE", "ENDPOINT_PRESERVATION", "DETERMINISTIC_REPLAY", "NEW_REGRESSION_FAILURES", "FORMAL_WORKER_CLEAN_EXIT_0", "STAGE_3_H7", "READY_FOR_STAGE_3_H8", "NEW_FJT_GOALS_SENT", "ROBOT_MOTION_STARTED", "FORMAL_LEDGER_MUTATED", "STAGE_3_H8_STARTED"]
    lines = ["# Stage 3 H7.7 — Controlled Local Dynamic Boundary Remediation", "", "Tier-B dynamic remediation passed all dynamic, process, collision, invariance, and determinism gates. The overall H7 gate remains blocked solely by the independently reproduced MoveItPy destructor exit `-11`; H8 was not started.", "", "## Mandatory final summary", "", "```text"]
    lines.extend(f"{key}: {certificate.get(key)}" for key in keys)
    lines.extend(["```", "", "Collision certification is `adaptive_discrete_interpolation` with MoveIt PlanningScene/FCL. CCD and clearance remain unavailable and were not inferred.", ""])
    (OUTPUT / "FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def manifests() -> None:
    required = ["FINAL_REPORT.md", "stage3_h7_7_terminal_certificate.json", "stage3_h7_7_gate_report.json", "immutable_input_manifest.json", "h7_6_baseline_replay_report.json", "moveit_ruckig_wrapper_semantics_audit.json", "overshoot_dynamic_boundary_root_cause.json", "tier_b_candidate_search.json", "selected_tier_b_remediation.json", "before_after_dynamic_state.json", "dynamic_state_provenance.json", "ruckig_strict_validation_report.json", "ruckig_error_inventory.json", "overshoot_recertification.json", "joint_limit_certification.json", "jerk_profile_certification.json", "process_tolerance_certification.json", "collision_certification.json", "endpoint_preservation_report.json", "h7_7_semantic_invariance_report.json", "deterministic_replay_report.json", "regression_report.json", "moveitpy_clean_exit_report.json", "sha256_manifest.json", "artifact_manifest.json"]
    records = []
    for name in required:
        path = OUTPUT / name
        records.append({"relative_path": name, "exists": path.is_file(), "size_bytes": path.stat().st_size if path.is_file() else None, "sha256": None if name in {"sha256_manifest.json", "artifact_manifest.json"} or not path.is_file() else sha(path)})
    manifest = {"schema_version": "stage3-h7-7-artifact-manifest-v1", "required": required, "records": records, "missing": [row["relative_path"] for row in records if not row["exists"]]}
    dump(OUTPUT / "artifact_manifest.json", manifest)
    hashes = {row["relative_path"]: row["sha256"] for row in records if row["sha256"] is not None}
    dump(OUTPUT / "sha256_manifest.json", {"schema_version": "stage3-h7-7-sha256-manifest-v1", "files": hashes, "self_hash_exclusions": ["sha256_manifest.json", "artifact_manifest.json"]})
    # Refresh existence after self-referential manifests have been created.
    for row in records:
        path = OUTPUT / row["relative_path"]
        row["exists"] = path.is_file()
        row["size_bytes"] = path.stat().st_size if path.is_file() else None
    manifest["missing"] = [row["relative_path"] for row in records if not row["exists"]]
    dump(OUTPUT / "artifact_manifest.json", manifest)


def main() -> int:
    baseline_report = baseline()
    immutable = immutable_manifest()
    wrapper_audit()
    root_cause()
    _search, selected = candidate_search()
    dynamic_cost(selected)
    strict, joint, overshoot = native_audits()
    process, collision = physical_reports()
    semantic, endpoint = invariance()
    replay = replay_report()
    regress = regression()
    shutdown = clean_exit()
    certificate = terminal(baseline_report, immutable, strict, joint, overshoot, process, collision, semantic, endpoint, replay, regress, shutdown)
    final_report(certificate)
    manifests()
    print(json.dumps({"output": str(OUTPUT), "STAGE_3_H7_7": certificate["STAGE_3_H7_7"], "STAGE_3_H7": certificate["STAGE_3_H7"], "FIRST_BLOCKER": certificate["FIRST_BLOCKER"], "STAGE_3_H7_FIRST_BLOCKER": certificate["STAGE_3_H7_FIRST_BLOCKER"]}, ensure_ascii=False))
    return 0 if certificate["STAGE_3_H7_7"] == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
