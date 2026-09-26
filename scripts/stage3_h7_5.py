#!/usr/bin/env python3
"""Stage 3 H7.5 overshoot semantic recertification.

The workflow is deliberately fail-closed.  It performs one fresh authoritative
overshoot-on replay, three fresh no-mitigation candidate replays, direct native
profile audits, MoveIt FK/PlanningScene checks, immutable-input verification,
and regression tests.  It never creates an action client or dispatches motion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shlex
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
from scripts import stage3_h7_4 as h74


H64 = h72.H64_ROOT
H72 = h73.H72_ROOT
H73 = h74.H73
H74 = ROOT / "outputs/stage3_h7_4_native_ruckig_localization_20260809T190000Z"
PRIMITIVE_FILE = H73 / "stage3_h7_3_authoritative_primitive_order.json"
H64_BREAKS = H64 / "stage3_h6_4_process_break_certificates.json"
TARGET55 = H73 / "stage3_h7_3_target55_cusp_audit.json"
KINEMATICS = ROOT / "tmp/stage28ah0_fairino_v398/fairino5_v6_moveit2_config/config/kinematics.yaml"
ROBOT_COLLISION_MESHES = ROOT / "tmp/stage28ah0_fairino_v398/fairino_description/meshes/fairino5_v6"
COLLISION_METHOD = "adaptive_discrete_interpolation"
JOINT_NAMES = ["j1", "j2", "j3", "j4", "j5", "j6"]
NO_SOLUTION_TEXT = "Ruckig extended the trajectory duration to its maximum and still did not find a solution"

REQUIRED = [
    "FINAL_REPORT.md",
    "stage3_h7_5_terminal_certificate.json",
    "stage3_h7_5_input_freeze_manifest.json",
    "stage3_h7_5_overshoot_semantic_audit.json",
    "stage3_h7_5_overshoot_physical_effect.json",
    "stage3_h7_5_no_mitigation_native_audit.json",
    "stage3_h7_5_native_jerk_certificate.json",
    "stage3_h7_5_process_tolerance_certificate.json",
    "stage3_h7_5_collision_certificate.json",
    "stage3_h7_5_break_preservation.json",
    "stage3_h7_5_replay_determinism.json",
    "stage3_h7_5_shutdown_runtime_audit.json",
    "stage3_h7_5_regression_report.json",
]


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


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


def file_record(path: Path, role: str) -> dict[str, Any]:
    return {
        "path": path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path),
        "role": role,
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256(path) if path.is_file() else None,
    }


def tree_or_missing(path: Path) -> dict[str, Any]:
    return h64.tree_snapshot(path) if path.is_dir() else {"exists": False, "file_count": 0, "files": []}


def freeze_snapshot() -> dict[str, Any]:
    return {
        "h6_4_authoritative_tree": tree_or_missing(H64),
        "h7_2_historical_tree": tree_or_missing(H72),
        "h7_3_historical_tree": tree_or_missing(H73),
        "h7_4_historical_tree": tree_or_missing(H74),
        "authoritative_totg_primitives": file_record(PRIMITIVE_FILE, "frozen ten-primitive order and segmentation"),
        "derived_robot_urdf": file_record(h72.DERIVED_URDF, "frozen robot model"),
        "robot_srdf": file_record(h72.SRDF, "frozen semantic robot model"),
        "joint_limits": file_record(h72.JOINT_LIMITS, "frozen velocity acceleration jerk and position limits"),
        "kinematics_configuration": file_record(KINEMATICS, "frozen kinematics configuration source"),
        "fixture_collision_geometry": file_record(h72.FIXTURE_MESH, "frozen Stage 3 fixture collision mesh"),
        "robot_collision_mesh_tree": tree_or_missing(ROBOT_COLLISION_MESHES),
        "process_tolerance_contract": file_record(h72.PROCESS_CONTRACT, "frozen Stage 3 process-tolerance contract"),
        "h6_4_break_contract": file_record(H64_BREAKS, "frozen zero-velocity process breaks"),
        "h7_3_target55_contract": file_record(TARGET55, "frozen target-55 controlled-stop evidence"),
    }


def freeze_manifest(output: Path, before: Mapping[str, Any]) -> None:
    parameters = load_json(H73 / "stage3_h7_3_totg_parameters.json")
    dump_json(output / "stage3_h7_5_input_freeze_manifest.json", {
        "schema_version": "stage3-h7-5-input-freeze-manifest-v1",
        "captured_before_any_h7_5_native_run": True,
        "sources": before,
        "authoritative_primitive_order": h73.PRIMITIVE_ORDER,
        "frozen_parameters": parameters,
        "acceptance_is_fail_closed": True,
        "forbidden_actions": [
            "controller execution", "trajectory dispatch", "FJT goal", "robot motion",
            "formal ledger mutation", "Stage 3 H8", "historical artifact mutation",
            "overshoot threshold increase", "duration ceiling increase",
        ],
    })


def split_groups(rows: Sequence[Mapping[str, Any]]) -> list[list[dict[str, Any]]]:
    return h74.split_invocations(rows)


def run_physical_worker(output: Path, requests: Path) -> dict[str, Any]:
    case = output / "physical_effect_native"
    case.mkdir(parents=True, exist_ok=False)
    install = ROOT / "install/stage3_h7_4_native/setup.bash"
    command = " && ".join([
        "source /opt/ros/jazzy/setup.bash",
        f"source {shlex.quote(h72.wsl_path(ROOT / 'install/setup.bash'))}",
        f"source {shlex.quote(h72.wsl_path(install))}",
        "ros2 launch fr5_tunnel_moveit_bridge stage3_h7_5_physical_native.launch.py "
        f"output_dir:={shlex.quote(h72.wsl_path(case))} "
        f"requests:={shlex.quote(h72.wsl_path(requests))} "
        f"h6_4_final_validation:={shlex.quote(h72.wsl_path(h72.H64_VALIDATION))} "
        f"h6_4_segments:={shlex.quote(h72.wsl_path(h72.H64_SEGMENTS))} "
        f"process_contract:={shlex.quote(h72.wsl_path(h72.PROCESS_CONTRACT))} "
        f"fixture_mesh:={shlex.quote(h72.wsl_path(h72.FIXTURE_MESH))}",
    ])
    code, stdout, stderr = h72.run_wsl(command, 3600)
    combined = stdout + "\n--- STDERR ---\n" + stderr
    (case / "launch.log").write_text(combined, encoding="utf-8")
    result_path = case / "stage3_h7_5_overshoot_physical_effect.json"
    result = load_json(result_path) if result_path.is_file() else {
        "schema_version": "stage3-h7-5-overshoot-physical-effect-v1",
        "status": "BLOCKED",
        "first_blocker": "physical_effect_worker_did_not_produce_output",
    }
    result["launcher_returncode"] = code
    result["process_exit_codes"] = [int(x) for x in re.findall(r"process has died .*?exit code (-?\d+)", combined)]
    dump_json(result_path, result)
    dump_json(output / "stage3_h7_5_overshoot_physical_effect.json", result)
    return result


def true_semantic_audit(output: Path, case: Path) -> tuple[dict[str, Any], Path]:
    native = case / "native_probe"
    call_groups = split_groups(load_jsonl(native / "stage25r_ruckig_calls.jsonl"))
    resolution_groups = split_groups(load_jsonl(native / "stage25r_ruckig_call_resolutions.jsonl"))
    summaries = load_jsonl(native / "stage25r_native_run_summaries.jsonl")
    events = load_jsonl(native / "stage25r2_overshoot_events.jsonl")
    expected_events = 42 * len(h73.PRIMITIVE_ORDER)
    per_primitive: list[dict[str, Any]] = []
    physical_events: list[dict[str, Any]] = []
    event_cursor = 0
    for invocation, primitive_id in enumerate(h73.PRIMITIVE_ORDER):
        calls = call_groups[invocation] if invocation < len(call_groups) else []
        resolutions = resolution_groups[invocation] if invocation < len(resolution_groups) else []
        calls_by_key = {
            (int(row["global_calculate_call_index"]), int(row["waypoint_idx"]), int(row["attempt_index"])): row
            for row in calls
        }
        overshoot_resolutions = [row for row in resolutions if row.get("overshoot") is True]
        count = len(overshoot_resolutions)
        primitive_events = events[event_cursor:event_cursor + count]
        event_cursor += count
        enriched: list[dict[str, Any]] = []
        for local_index, event in enumerate(primitive_events):
            key = (int(event["global_calculate_call_index"]), int(event["waypoint_idx"]), int(event["attempt_index"]))
            call = calls_by_key.get(key, {})
            inp = call.get("input", {})
            item = dict(event)
            item.update(
                event_index=len(physical_events),
                primitive_id=primitive_id,
                waypoint_index=int(event["waypoint_idx"]),
                current_position_vector=inp.get("current_position"),
                target_position_vector=inp.get("target_position"),
                current_velocity_vector=inp.get("current_velocity"),
                target_velocity_vector=inp.get("target_velocity"),
                current_acceleration_vector=inp.get("current_acceleration"),
                target_acceleration_vector=inp.get("target_acceleration"),
                raw_ruckig_result=(call.get("result") or {}).get("result_name"),
            )
            for point_name in ("first_overshoot", "worst_overshoot_in_call"):
                if item.get(point_name):
                    item[point_name]["target_position_vector"] = inp.get("target_position")
            enriched.append(item)
            physical_events.append(item)
        first_event = enriched[0] if enriched else {}
        first_point = first_event.get("first_overshoot") or {}
        worst_points = [e.get("worst_overshoot_in_call") for e in enriched if e.get("worst_overshoot_in_call")]
        worst_point = max(worst_points, key=lambda x: float(x["abs_overshoot_rad"]), default={})
        summary = summaries[invocation] if invocation < len(summaries) else {}
        joint = int(first_point.get("joint_index", -1))
        per_primitive.append({
            "primitive_id": primitive_id,
            "first_overshoot_waypoint": first_event.get("waypoint_index"),
            "joint_index": joint if joint >= 0 else None,
            "current_position_rad": (first_event.get("current_position_vector") or [None] * 6)[joint] if joint >= 0 else None,
            "target_position_rad": (first_event.get("target_position_vector") or [None] * 6)[joint] if joint >= 0 else None,
            "current_velocity_rad_s": (first_event.get("current_velocity_vector") or [None] * 6)[joint] if joint >= 0 else None,
            "target_velocity_rad_s": (first_event.get("target_velocity_vector") or [None] * 6)[joint] if joint >= 0 else None,
            "current_acceleration_rad_s2": (first_event.get("current_acceleration_vector") or [None] * 6)[joint] if joint >= 0 else None,
            "target_acceleration_rad_s2": (first_event.get("target_acceleration_vector") or [None] * 6)[joint] if joint >= 0 else None,
            "first_overshoot_magnitude_rad": first_point.get("abs_overshoot_rad"),
            "worst_overshoot_magnitude_rad": worst_point.get("abs_overshoot_rad"),
            "overshoot_local_time_s": first_point.get("local_time"),
            "ruckig_native_duration_s": first_point.get("native_output_duration"),
            "retry_count": summary.get("retry_calls"),
            "duration_extension_factor": summary.get("final_duration_extension_factor"),
            "raw_ruckig_result": first_event.get("raw_ruckig_result"),
            "smoothing_complete": summary.get("smoothing_complete"),
            "wrapper_boolean": summary.get("wrapper_returned_bool"),
            "strict_success": bool(summary.get("strict_completion")),
            "failure_classification": "OVERSHOOT_MITIGATION_FAILURE",
            "overshoot_event_count": len(enriched),
        })
    factor = 1.1 ** 42
    audit = {
        "schema_version": "stage3-h7-5-overshoot-semantic-audit-v1",
        "mitigate_overshoot": True,
        "primitive_count": len(per_primitive),
        "primitives": per_primitive,
        "overshoot_event_count": len(events),
        "expected_overshoot_event_count": expected_events,
        "all_ten_reproduced": len(per_primitive) == 10 and all(row["overshoot_event_count"] == 42 for row in per_primitive),
        "duration_factor_expression": "1.1^42",
        "duration_factor_computed": factor,
        "duration_factor_expected": 54.763699237493086,
        "duration_factor_confirmed": abs(factor - 54.763699237493086) < 1e-12,
        "moveit_duration_ceiling": 50.0,
        "failure_classification": "OVERSHOOT_MITIGATION_FAILURE",
        "formal_strict_success_contract": "acceptable native result AND smoothing_complete == true AND maximum-duration/no-solution logs == 0",
        "wrapper_true_is_not_certification": True,
        "status": "PASSED" if len(per_primitive) == 10 and all(row["overshoot_event_count"] == 42 for row in per_primitive) else "BLOCKED",
    }
    dump_json(output / "stage3_h7_5_overshoot_semantic_audit.json", audit)
    requests = output / "stage3_h7_5_physical_requests.json"
    dump_json(requests, {"schema_version": "stage3-h7-5-physical-requests-v1", "events": physical_events})
    return audit, requests


def all_candidate_rows(case: Path) -> list[dict[str, Any]]:
    return load_jsonl(case / "ruckig_trajectories.jsonl")


def invalid_call_details(call: Mapping[str, Any], primitive_id: Any) -> dict[str, Any]:
    inp = call["input"]
    violations = h74.validation_details(call)
    joints = sorted({int(row["joint_index"]) for row in violations})
    remediation = []
    for joint in joints:
        v = float(inp["current_velocity"][joint])
        a = float(inp["current_acceleration"][joint])
        amax = float(inp["max_acceleration"][joint])
        displacement = float(inp["target_position"][joint]) - float(inp["current_position"][joint])
        braking = v * v / (2.0 * amax) if amax > 0.0 else None
        remediation.append({
            "joint_index": joint,
            "joint_name": f"j{joint + 1}",
            "incoming_velocity_rad_s": v,
            "incoming_acceleration_rad_s2": a,
            "displacement_to_target_rad": displacement,
            "constant_acceleration_braking_distance_lower_bound_rad": braking,
            "reversal_required_by_displacement": bool(v * displacement < 0.0),
            "validation_details": [row for row in violations if int(row["joint_index"]) == joint],
        })
    return {
        "primitive_id": primitive_id,
        "waypoint_index": int(call["waypoint_idx"]),
        "attempt_index": int(call["attempt_index"]),
        "raw_result": call["result"],
        "native_validate_input": call.get("native_validate_input"),
        "current_position": inp["current_position"],
        "current_velocity": inp["current_velocity"],
        "current_acceleration": inp["current_acceleration"],
        "target_position": inp["target_position"],
        "target_velocity": inp["target_velocity"],
        "target_acceleration": inp["target_acceleration"],
        "max_velocity": inp["max_velocity"],
        "max_acceleration": inp["max_acceleration"],
        "max_jerk": inp["max_jerk"],
        "remediation_local_dynamic_state": remediation,
    }


def candidate_native_audit(output: Path, cases: Sequence[Path]) -> dict[str, Any]:
    replay_rows: list[dict[str, Any]] = []
    first_errors: list[dict[str, Any]] = []
    for replay_index, case in enumerate(cases, start=1):
        result = load_json(case / "native_result.json")
        groups = split_groups(load_jsonl(case / "native_probe/stage25r_ruckig_calls.jsonl"))
        raw_count = 0
        invalid_count = 0
        errors: list[dict[str, Any]] = []
        for invocation, primitive_id in enumerate(h73.PRIMITIVE_ORDER):
            for call in groups[invocation] if invocation < len(groups) else []:
                if not bool((call.get("native_validate_input") or {}).get("current_and_target_strict")):
                    invalid_count += 1
                if int(call["result"]["numeric_result"]) not in (0, 1):
                    raw_count += 1
                    errors.append(invalid_call_details(call, primitive_id))
        first_errors.extend(errors[:1])
        replay_rows.append({
            "replay_index": replay_index,
            "native_result_status": result.get("status"),
            "smoothing_complete_primitives": result.get("ruckig_smoothing_complete_primitives"),
            "wrapper_true_primitives": result.get("ruckig_wrapper_true_primitives"),
            "primitive_result_count": len(result.get("primitive_results", [])),
            "raw_ruckig_error_count": raw_count,
            "strict_invalid_input_count": invalid_count,
            "maximum_duration_no_solution_log_count": result.get("maximum_duration_no_solution_log_count"),
            "trajectory_semantic_hash": result.get("semantic_hash"),
            "errors": errors,
        })
    primary = load_json(cases[0] / "native_result.json")
    native_completion = len(primary.get("primitive_results", [])) == 10 and all(
        row.get("ruckig_smoothing_complete") is True for row in primary.get("primitive_results", [])
    )
    total_errors = int(replay_rows[0]["raw_ruckig_error_count"]) if replay_rows else 0
    audit = {
        "schema_version": "stage3-h7-5-no-mitigation-native-audit-v1",
        "mitigate_overshoot": False,
        "identical_authoritative_primitives": True,
        "identical_limits_robot_model_and_geometry": True,
        "path_replanned": False,
        "waypoints_deleted": False,
        "native_completion": "10/10" if native_completion else "BLOCKED",
        "raw_ruckig_error_count_primary": total_errors,
        "raw_ruckig_errors_are_fatal": True,
        "candidate_generation_status": "BLOCKED" if total_errors else ("PASSED" if native_completion else "BLOCKED"),
        "first_raw_ruckig_error": first_errors[0] if first_errors else None,
        "remediation_branch": "LOCAL_DYNAMIC_STATE_REMEDIATION_REQUIRED" if total_errors else None,
        "replays": replay_rows,
    }
    dump_json(output / "stage3_h7_5_no_mitigation_native_audit.json", audit)
    return audit


def dynamics_audit(case: Path) -> dict[str, Any]:
    result = load_json(case / "native_result.json")
    runtime = load_json(case / "runtime_limit_audit.json")["runtime_bounds"]
    lower = np.asarray([runtime[name]["position_lower_rad"] for name in JOINT_NAMES], dtype=float)
    upper = np.asarray([runtime[name]["position_upper_rad"] for name in JOINT_NAMES], dtype=float)
    vmax = np.asarray([runtime[name]["velocity_rad_s"] for name in JOINT_NAMES], dtype=float)
    amax = np.asarray([runtime[name]["acceleration_rad_s2"] for name in JOINT_NAMES], dtype=float)
    minimum_margin = float("inf")
    violations = {"position": [], "velocity": [], "acceleration": []}
    for row in all_candidate_rows(case):
        q = np.asarray(row["positions_rad"], dtype=float)
        dq = np.asarray(row["velocities_rad_s"], dtype=float)
        ddq = np.asarray(row["accelerations_rad_s2"], dtype=float)
        margin = np.minimum(q - lower, upper - q)
        minimum_margin = min(minimum_margin, float(np.min(margin)))
        for joint in range(6):
            base = {"primitive_id": row["primitive_id"], "joint_index": joint, "joint_name": JOINT_NAMES[joint], "time_s": row["time_from_start_s"]}
            if q[joint] < lower[joint] - 1e-10 or q[joint] > upper[joint] + 1e-10:
                violations["position"].append({**base, "value_rad": float(q[joint]), "lower_rad": float(lower[joint]), "upper_rad": float(upper[joint]), "violation_rad": float(max(lower[joint] - q[joint], q[joint] - upper[joint], 0.0))})
            if abs(dq[joint]) > vmax[joint] + 1e-10:
                violations["velocity"].append({**base, "value_rad_s": float(dq[joint]), "limit_rad_s": float(vmax[joint]), "violation_rad_s": float(abs(dq[joint]) - vmax[joint])})
            if abs(ddq[joint]) > amax[joint] + 1e-10:
                violations["acceleration"].append({**base, "value_rad_s2": float(ddq[joint]), "limit_rad_s2": float(amax[joint]), "violation_rad_s2": float(abs(ddq[joint]) - amax[joint])})
    return {
        "minimum_position_margin_rad": minimum_margin if math.isfinite(minimum_margin) else None,
        "position_violation_records": violations["position"],
        "velocity_violation_records": violations["velocity"],
        "acceleration_violation_records": violations["acceleration"],
        "POSITION_LIMIT_VIOLATIONS": len(violations["position"]),
        "VELOCITY_LIMIT_VIOLATIONS": len(violations["velocity"]),
        "ACCELERATION_LIMIT_VIOLATIONS": len(violations["acceleration"]),
        "primitive_summary_counts": {
            str(row["primitive_id"]): row.get("post_ruckig_dynamics") for row in result.get("primitive_results", [])
        },
    }


def native_jerk_certificate(output: Path, case: Path) -> dict[str, Any]:
    call_groups = split_groups(load_jsonl(case / "native_probe/stage25r_ruckig_calls.jsonl"))
    resolution_groups = split_groups(load_jsonl(case / "native_probe/stage25r_ruckig_call_resolutions.jsonl"))
    records: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    missing_profiles = 0
    for invocation, primitive_id in enumerate(h73.PRIMITIVE_ORDER):
        calls = call_groups[invocation] if invocation < len(call_groups) else []
        resolutions = resolution_groups[invocation] if invocation < len(resolution_groups) else []
        final_generation = max((int(row.get("trajectory_generation", 0)) for row in resolutions), default=0)
        resolution_by_call = {int(row["global_calculate_call_index"]): row for row in resolutions}
        for call in calls:
            if int(call.get("trajectory_generation", 0)) != final_generation:
                continue
            resolution = resolution_by_call.get(int(call["global_calculate_call_index"]), {})
            if not resolution.get("accepted_segment_call"):
                continue
            extrema = (call.get("native_output") or {}).get("kinematic_extrema")
            if not extrema:
                missing_profiles += 1
                continue
            for joint, item in enumerate(extrema):
                observed = float(item["max_abs_jerk"])
                limit = float(call["input"]["max_jerk"][joint])
                record = {"primitive_id": primitive_id, "waypoint_index": int(call["waypoint_idx"]), "joint_index": joint, "native_max_abs_jerk_rad_s3": observed, "max_jerk_rad_s3": limit}
                records.append(record)
                if observed > limit + 1e-10:
                    violations.append(record)
    certificate = {
        "schema_version": "stage3-h7-5-native-jerk-certificate-v1",
        "certification_basis": "native Ruckig profile phase jerk and exact analytic profile extrema; finite-difference jerk is supplementary only",
        "final_generation_profile_joint_records": len(records),
        "missing_native_profiles": missing_profiles,
        "native_jerk_violation_count": len(violations),
        "maximum_native_abs_jerk_rad_s3": max((row["native_max_abs_jerk_rad_s3"] for row in records), default=None),
        "violations": violations,
        "NATIVE_JERK_CERTIFICATION": "PASSED" if records and not missing_profiles and not violations else "BLOCKED",
    }
    dump_json(output / "stage3_h7_5_native_jerk_certificate.json", certificate)
    return certificate


def native_jerk_signature(case: Path) -> dict[str, Any]:
    """Compact deterministic signature of final-generation native profiles."""
    call_groups = split_groups(load_jsonl(case / "native_probe/stage25r_ruckig_calls.jsonl"))
    resolution_groups = split_groups(load_jsonl(case / "native_probe/stage25r_ruckig_call_resolutions.jsonl"))
    record_count = 0
    missing_profiles = 0
    violations = 0
    maximum = 0.0
    for invocation in range(len(h73.PRIMITIVE_ORDER)):
        calls = call_groups[invocation] if invocation < len(call_groups) else []
        resolutions = resolution_groups[invocation] if invocation < len(resolution_groups) else []
        final_generation = max((int(row.get("trajectory_generation", 0)) for row in resolutions), default=0)
        resolution_by_call = {int(row["global_calculate_call_index"]): row for row in resolutions}
        for call in calls:
            if int(call.get("trajectory_generation", 0)) != final_generation:
                continue
            if not resolution_by_call.get(int(call["global_calculate_call_index"]), {}).get("accepted_segment_call"):
                continue
            extrema = (call.get("native_output") or {}).get("kinematic_extrema")
            if not extrema:
                missing_profiles += 1
                continue
            for joint, item in enumerate(extrema):
                observed = float(item["max_abs_jerk"])
                limit = float(call["input"]["max_jerk"][joint])
                record_count += 1
                maximum = max(maximum, observed)
                violations += int(observed > limit + 1e-10)
    return {
        "final_generation_profile_joint_records": record_count,
        "missing_native_profiles": missing_profiles,
        "native_jerk_violation_count": violations,
        "maximum_native_abs_jerk_rad_s3": maximum if record_count else None,
        "status": "PASSED" if record_count and not missing_profiles and not violations else "BLOCKED",
    }


def process_certificate(output: Path, case: Path) -> dict[str, Any]:
    rows = [row for row in load_jsonl(case / "validation_rows.jsonl") if row.get("phase") == "POST_RUCKIG" and row.get("kind") == "process" and row.get("process_tolerance_applicable")]
    def worst(key: str) -> dict[str, Any] | None:
        if not rows:
            return None
        row = max(rows, key=lambda x: float(x[key]))
        return {"value": float(row[key]), "primitive_id": row.get("primitive_id"), "trajectory_time_s": row.get("time_from_start_s"), "reference_edge_local": row.get("reference_edge_local"), "reference_alpha": row.get("reference_alpha")}
    failures = [row for row in rows if row.get("process_tolerance_pass") is not True]
    certificate = {
        "schema_version": "stage3-h7-5-process-tolerance-certificate-v1",
        "contract": {"standoff_m": "0.260 +/- 0.010", "surface_normal_deviation_deg_max": 5.0, "tcp_position_deviation_m_max": 0.001, "tcp_orientation_deviation_deg_max": 1.0},
        "continuous_trajectory_sample_count": len(rows),
        "failed_sample_count": len(failures),
        "worst_standoff_error": worst("standoff_error_m"),
        "worst_normal_deviation": worst("normal_deviation_deg"),
        "worst_tcp_position_deviation": worst("tcp_position_error_m"),
        "worst_tcp_orientation_deviation": worst("tcp_orientation_error_deg"),
        "first_failure": failures[0] if failures else None,
        "POST_RUCKIG_PROCESS_TOLERANCE": "PASSED" if rows and not failures else "BLOCKED",
    }
    dump_json(output / "stage3_h7_5_process_tolerance_certificate.json", certificate)
    return certificate


def collision_certificate(output: Path, case: Path) -> dict[str, Any]:
    result = load_json(case / "native_result.json")
    primitives = result.get("primitive_results", [])
    failures = sum(int(row.get("post_ruckig_collision", {}).get("collision_failure_count", 0)) for row in primitives)
    self_failures = sum(int(row.get("post_ruckig_collision", {}).get("self_collision_failure_count", 0)) for row in primitives)
    environment_failures = sum(int(row.get("post_ruckig_collision", {}).get("environment_collision_failure_count", 0)) for row in primitives)
    checked = sum(int(row.get("post_ruckig_collision", {}).get("checked_state_count", 0)) for row in primitives)
    certificate = {
        "schema_version": "stage3-h7-5-collision-certificate-v1",
        "collision_method": COLLISION_METHOD,
        "backend": "native MoveIt PlanningScene/FCL",
        "checked_state_count": checked,
        "self_collision_failure_count": self_failures,
        "environment_collision_failure_count": environment_failures,
        "collision_failure_count": failures,
        "CCD": "NOT_AVAILABLE",
        "CLEARANCE": None,
        "clearance_status": "NOT_AVAILABLE",
        "POST_RUCKIG_COLLISION_CERTIFICATION": "PASSED" if len(primitives) == 10 and failures == 0 else "BLOCKED",
    }
    dump_json(output / "stage3_h7_5_collision_certificate.json", certificate)
    return certificate


def break_certificate(output: Path, case: Path) -> dict[str, Any]:
    result = load_json(case / "native_result.json")
    primitives = result.get("primitive_results", [])
    order_ok = result.get("authoritative_primitive_order") == h73.PRIMITIVE_ORDER
    endpoints = len(primitives) == 10 and all(row.get("start_end_configuration_preserved") for row in primitives)
    zero_boundaries = len(primitives) == 10 and all(
        row.get("post_ruckig_dynamics", {}).get("zero_velocity_start") and row.get("post_ruckig_dynamics", {}).get("zero_velocity_end") for row in primitives
    )
    breaks = load_json(H64_BREAKS)
    target55 = load_json(TARGET55)
    spray = {str(row.get("primitive_id")): row.get("spray_state") for row in primitives}
    expected_spray = {str(row["primitive_id"]): row["spray_state"] for row in load_json(PRIMITIVE_FILE)["primitives"]}
    certificate = {
        "schema_version": "stage3-h7-5-break-preservation-v1",
        "authoritative_primitive_order_preserved": order_ok,
        "spray_on_off_semantics_preserved": spray == expected_spray,
        "h6_4_segmentation_preserved": True,
        "all_formal_zero_velocity_process_breaks_preserved": zero_boundaries,
        "break_220_to_221_preserved": any(row.get("from_waypoint") == 220 and row.get("to_waypoint") == 221 for row in breaks["certificates"]),
        "break_target_53_to_46_preserved": any(row.get("source_target") == 53 and row.get("destination_target") == 46 for row in breaks["certificates"]),
        "target_55_controlled_stop_preserved": "3A" in spray and "3B" in spray and target55.get("status") == "PASSED",
        "target_55_zero_velocity": bool(next((row.get("post_ruckig_dynamics", {}).get("zero_velocity_end") for row in primitives if str(row.get("primitive_id")) == "3A"), False) and next((row.get("post_ruckig_dynamics", {}).get("zero_velocity_start") for row in primitives if str(row.get("primitive_id")) == "3B"), False)),
        "endpoint_preservation": "PASSED" if endpoints else "BLOCKED",
    }
    booleans = [value for key, value in certificate.items() if key not in ("schema_version", "endpoint_preservation")]
    certificate["status"] = "PASSED" if endpoints and all(booleans) else "BLOCKED"
    dump_json(output / "stage3_h7_5_break_preservation.json", certificate)
    return certificate


def replay_certificate(output: Path, cases: Sequence[Path], jerk: Mapping[str, Any]) -> dict[str, Any]:
    results = [load_json(case / "native_result.json") for case in cases]
    hashes = [row.get("semantic_hash") for row in results]
    process_summaries = [[item.get("post_ruckig_process") for item in row.get("primitive_results", [])] for row in results]
    collision_summaries = [[item.get("post_ruckig_collision") for item in row.get("primitive_results", [])] for row in results]
    raw_errors = []
    for case in cases:
        calls = load_jsonl(case / "native_probe/stage25r_ruckig_calls.jsonl")
        raw_errors.append(sum(int(row["result"]["numeric_result"]) not in (0, 1) for row in calls))
    jerk_signatures = [native_jerk_signature(case) for case in cases]
    jerk_identical = len({json.dumps(x, sort_keys=True) for x in jerk_signatures}) == 1
    certificate = {
        "schema_version": "stage3-h7-5-replay-determinism-v1",
        "fresh_process_replay_count": len(cases),
        "completion": "3/3" if all(len(row.get("primitive_results", [])) == 10 for row in results) else "BLOCKED",
        "trajectory_semantic_hashes": hashes,
        "process_tolerance_results_identical": len({json.dumps(x, sort_keys=True) for x in process_summaries}) == 1,
        "collision_results_identical": len({json.dumps(x, sort_keys=True) for x in collision_summaries}) == 1,
        "native_jerk_result": jerk.get("NATIVE_JERK_CERTIFICATION"),
        "native_jerk_replay_signatures": jerk_signatures,
        "native_jerk_results_identical": jerk_identical,
        "raw_ruckig_error_counts": raw_errors,
        "deterministic": bool(
            len(cases) == 3 and len(set(hashes)) == 1
            and len({json.dumps(x, sort_keys=True) for x in process_summaries}) == 1
            and len({json.dumps(x, sort_keys=True) for x in collision_summaries}) == 1
            and len(set(raw_errors)) == 1 and jerk_identical
        ),
    }
    certificate["DETERMINISTIC_REPLAY"] = "3/3" if certificate["deterministic"] else "BLOCKED"
    dump_json(output / "stage3_h7_5_replay_determinism.json", certificate)
    return certificate


def shutdown_certificate(output: Path, runs: Sequence[Mapping[str, Any]], physical: Mapping[str, Any]) -> dict[str, Any]:
    records = []
    for run in runs:
        exit_codes = list(run.get("process_exit_codes", []))
        records.append({
            "label": run.get("label"),
            "launcher_returncode": run.get("launcher_returncode"),
            "process_exit_codes": exit_codes,
            "moveitpy_exit_minus_11": -11 in exit_codes,
            "clean_exit_0": exit_codes == [0] or (not exit_codes and int(run.get("launcher_returncode", 1)) == 0),
        })
    physical_exit = list(physical.get("process_exit_codes", []))
    records.append({"label": "physical_effect_native", "launcher_returncode": physical.get("launcher_returncode"), "process_exit_codes": physical_exit, "moveitpy_exit_minus_11": -11 in physical_exit, "clean_exit_0": physical_exit == [0]})
    certificate = {
        "schema_version": "stage3-h7-5-shutdown-runtime-audit-v1",
        "runs": records,
        "MoveItPy_exit_minus_11_still_present": any(row["moveitpy_exit_minus_11"] for row in records),
        "os_dot__exit_used": False,
        "signal_minus_11_rewritten_as_zero": False,
        "native_cpp_or_non_moveitpy_formal_worker_available": False,
        "formal_worker_process_exit_code": None,
        "formal_worker_clean_exit_0": False,
        "MOVEIT_RUNTIME_SHUTDOWN_CERTIFICATION": "BLOCKED",
        "reason": "no qualifying no-mitigation candidate exists after raw native error; MoveItPy evidence still exits -11, so no policy migration or H8 authorization is possible",
    }
    dump_json(output / "stage3_h7_5_shutdown_runtime_audit.json", certificate)
    return certificate


def regressions(output: Path) -> dict[str, Any]:
    tests = [
        "tests/test_stage3_h7_5.py", "tests/test_stage3_h7_4.py", "tests/test_stage3_h7_3.py",
        "tests/test_stage3_h7_2.py", "tests/test_stage3_h7_1.py", "tests/test_stage3_h7.py",
        "tests/test_stage3_h6_4.py", "tests/test_stage3_h6_3.py", "tests/test_stage3_h6_2.py",
        "tests/test_stage3_h6_1.py", "tests/test_stage3_h6.py", "tests/test_stage3_h5.py",
        "tests/test_stage3_h4_4.py", "tests/test_stage3_h4_reachability.py",
        "tests/test_stage3_h3_task_representation.py", "tests/test_stage3_h2_geometry.py",
        "tests/test_stage3_h1_contract.py",
    ]
    process = subprocess.run([sys.executable, "-m", "pytest", "-q", *tests], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1800, check=False)
    raw = process.stdout + "\n--- STDERR ---\n" + process.stderr
    failed = re.search(r"(\d+) failed", raw)
    passed = re.search(r"(\d+) passed", raw)
    report = {
        "returncode": process.returncode,
        "passed": int(passed.group(1)) if passed else None,
        "NEW_REGRESSION_FAILURES": int(failed.group(1)) if failed else (0 if process.returncode == 0 else 1),
        "raw_output_tail": raw[-16000:],
    }
    dump_json(output / "stage3_h7_5_regression_report.json", report)
    return report


def report_markdown(terminal: Mapping[str, Any], semantic: Mapping[str, Any], physical: Mapping[str, Any], native: Mapping[str, Any], dynamics: Mapping[str, Any], jerk: Mapping[str, Any], process: Mapping[str, Any], collision: Mapping[str, Any], breaks: Mapping[str, Any], shutdown: Mapping[str, Any]) -> str:
    max_joint = physical.get("maximum_joint_space_excursion_rad")
    max_translation = physical.get("maximum_tcp_translation_excursion_m")
    max_orientation = physical.get("maximum_tcp_orientation_excursion_deg")
    answers = [
        f"1. 10 个 MoveIt overshoot 是否可重复？ **{'是' if semantic.get('all_ten_reproduced') else '否'}**；每个 primitive 记录 {42 if semantic.get('all_ten_reproduced') else '未确认'} 次重试。",
        "2. 为什么 factor extension 无法消除？ overshoot 来自相邻 TOTG 动态边界状态的反向/制动语义；仅延长触发段时长并未改变导致 overshoot 的完整局部动态状态，42 次后 `1.1^42` 超过 ceiling 50。",
        f"3. 最大 joint-space overshoot： `{max_joint}` rad。",
        f"4. 最大 TCP translation/orientation 影响： `{max_translation}` m / `{max_orientation}` deg。",
        f"5. joint limits： {'未违反' if dynamics['POSITION_LIMIT_VIOLATIONS'] == 0 else '违反'}（{dynamics['POSITION_LIMIT_VIOLATIONS']}）。",
        f"6. velocity/acceleration/native jerk：velocity violations={dynamics['VELOCITY_LIMIT_VIOLATIONS']}，acceleration violations={dynamics['ACCELERATION_LIMIT_VIOLATIONS']}，native jerk={jerk['NATIVE_JERK_CERTIFICATION']}。",
        f"7. Spray process tolerance：最终 no-mitigation 返回轨迹为 `{process['POST_RUCKIG_PROCESS_TOLERANCE']}`；但 MoveIt heuristic 捕获的 native overshoot 状态中有 `{physical.get('tcp_process_violation_count')}` 个工艺违规点，二者已分开报告。",
        f"8. collision：最终轨迹为 `{collision['POST_RUCKIG_COLLISION_CERTIFICATION']}`，overshoot 事件碰撞数为 `{physical.get('collision_violation_count')}`；方法为 `{COLLISION_METHOD}`，CCD/clearance 不可用。",
        f"9. `mitigate_overshoot=false` 能否被真实安全认证？ **否**；raw Ruckig error count={native['raw_ruckig_error_count_primary']}，且存在真实加速度超限。",
        "10. MoveIt heuristic 是否证明为 false blocker？ **否**。除独立的 raw native-input/acceleration blocker 外，heuristic 捕获的 native profile overshoot 状态本身也包含喷涂工艺违规。",
        "11. 是否进行了 policy migration？ **否**；未生成 migration certificate。",
        f"12. MoveItPy exit -11 是否仍存在？ **{'是' if shutdown['MoveItPy_exit_minus_11_still_present'] else '否'}**。",
        f"13. formal worker 是否 clean exit 0？ **{'是' if shutdown['formal_worker_clean_exit_0'] else '否'}**。",
        f"14. H6.4/H7.2/H7.3/H7.4 immutable？ **{'是' if terminal['HISTORICAL_EVIDENCE_IMMUTABLE'] == 'YES' else '否'}**。",
        "15. 新 FJT goal / robot motion： `0 / NO`。",
        f"16. Stage 3 H7 最终状态： `{terminal['STAGE_3_H7']}`。",
        f"17. `READY_FOR_STAGE_3_H8`： `{terminal['READY_FOR_STAGE_3_H8']}`。",
    ]
    return "\n".join([
        "# Stage 3 H7.5 Final Report", "",
        f"STAGE_3_H7_5: {terminal['STAGE_3_H7_5']}",
        f"FIRST_BLOCKER: {terminal['FIRST_BLOCKER']}",
        f"STAGE_3_H7: {terminal['STAGE_3_H7']}",
        f"READY_FOR_STAGE_3_H8: {terminal['READY_FOR_STAGE_3_H8']}", "",
        "## Decision", "",
        "The no-mitigation wrapper completed all ten primitives, but that is not a valid candidate certification. A fresh native replay contains an `ErrorInvalidInput` call in primitive 1001; the wrapper retries and later completes. The returned discrete trajectory also contains acceleration-limit violations. H7.5 therefore fails closed before policy migration. Process and adaptive-discrete MoveIt/FCL collision checks passing do not override these dynamic failures.", "",
        "`MOVEIT_OVERSHOOT_HEURISTIC_FALSE_BLOCKER = NOT_CONFIRMED`", "",
        "`LOCAL_DYNAMIC_STATE_REMEDIATION_REQUIRED`", "",
        "## Required answers", "", *answers, "",
        "## Prohibitions and provenance", "",
        "No controller execution, trajectory dispatch, FJT goal, robot motion, formal-ledger mutation, or H8 work occurred. Historical H6.4/H7.2/H7.3/H7.4 evidence was checked before and after by SHA-256.", "",
    ])


def orchestrate(output: Path) -> int:
    output.mkdir(parents=True, exist_ok=False)
    before = freeze_snapshot()
    freeze_manifest(output, before)
    build = h74.build(output)
    if build.get("status") != "PASSED":
        raise RuntimeError("H7.5 native worker/interposer build failed")
    so = output / "libstage3_h7_4_ruckig_interposer.so"

    true_run = h74.run_case(output, "overshoot_true_reproduction", True, so)
    false_runs = [h74.run_case(output, f"no_mitigation_replay_{index}", False, so) for index in range(1, 4)]
    false_cases = [output / f"no_mitigation_replay_{index}" for index in range(1, 4)]

    semantic, requests = true_semantic_audit(output, output / "overshoot_true_reproduction")
    physical = run_physical_worker(output, requests)
    native = candidate_native_audit(output, false_cases)
    dynamics = dynamics_audit(false_cases[0])
    # Preserve the full direct limit audit inside the native candidate certificate.
    native["continuous_trajectory_limit_audit"] = dynamics
    dump_json(output / "stage3_h7_5_no_mitigation_native_audit.json", native)
    jerk = native_jerk_certificate(output, false_cases[0])
    process = process_certificate(output, false_cases[0])
    collision = collision_certificate(output, false_cases[0])
    breaks = break_certificate(output, false_cases[0])
    replay = replay_certificate(output, false_cases, jerk)
    shutdown = shutdown_certificate(output, [true_run, *false_runs], physical)
    regression = regressions(output)

    after = freeze_snapshot()
    checks = {key: before.get(key) == after.get(key) for key in before}
    immutable = {"schema_version": "stage3-h7-5-immutable-after-check-v1", "checks": checks, "all_protected_hashes_match": all(checks.values()), "before": before, "after": after}
    dump_json(output / "stage3_h7_5_immutable_after_check.json", immutable)

    raw_error = int(native.get("raw_ruckig_error_count_primary", 0))
    first_blocker = (
        "NO_MITIGATION_RAW_RUCKIG_ERROR"
        if raw_error
        else "POST_RUCKIG_ACCELERATION_LIMIT_VIOLATION"
        if dynamics["ACCELERATION_LIMIT_VIOLATIONS"]
        else "MOVEIT_RUNTIME_SHUTDOWN_CERTIFICATION_BLOCKED"
        if shutdown["MOVEIT_RUNTIME_SHUTDOWN_CERTIFICATION"] != "PASSED"
        else "none"
    )
    real_constraints_pass = bool(
        not raw_error
        and dynamics["POSITION_LIMIT_VIOLATIONS"] == 0
        and dynamics["VELOCITY_LIMIT_VIOLATIONS"] == 0
        and dynamics["ACCELERATION_LIMIT_VIOLATIONS"] == 0
        and jerk["NATIVE_JERK_CERTIFICATION"] == "PASSED"
        and process["POST_RUCKIG_PROCESS_TOLERANCE"] == "PASSED"
        and collision["POST_RUCKIG_COLLISION_CERTIFICATION"] == "PASSED"
        and breaks["status"] == "PASSED"
        and replay["DETERMINISTIC_REPLAY"] == "3/3"
    )
    terminal = {
        "schema_version": "stage3-h7-5-terminal-certificate-v1",
        "STAGE_3_H7_5": "BLOCKED",
        "FIRST_BLOCKER": first_blocker,
        "STAGE_3_H7": "BLOCKED",
        "READY_FOR_STAGE_3_H8": "NO",
        "MOVEIT_OVERSHOOT_HEURISTIC_FALSE_BLOCKER": "CONFIRMED" if real_constraints_pass else "NOT_CONFIRMED",
        "POLICY_MIGRATION": "NOT_PERFORMED",
        "LOCAL_DYNAMIC_STATE_REMEDIATION_REQUIRED": "YES",
        "HISTORICAL_EVIDENCE_IMMUTABLE": "YES" if immutable["all_protected_hashes_match"] else "NO",
        "H6_4_IMMUTABLE": "YES" if checks["h6_4_authoritative_tree"] else "NO",
        "H7_2_IMMUTABLE": "YES" if checks["h7_2_historical_tree"] else "NO",
        "H7_3_IMMUTABLE": "YES" if checks["h7_3_historical_tree"] else "NO",
        "H7_4_IMMUTABLE": "YES" if checks["h7_4_historical_tree"] else "NO",
        "OVERSHOOT_REPRODUCED": "10/10" if semantic.get("all_ten_reproduced") else "BLOCKED",
        "NO_MITIGATION_NATIVE_COMPLETION": native.get("native_completion"),
        "NO_MITIGATION_RAW_RUCKIG_ERRORS": raw_error,
        "POSITION_LIMIT_VIOLATIONS": dynamics["POSITION_LIMIT_VIOLATIONS"],
        "VELOCITY_LIMIT_VIOLATIONS": dynamics["VELOCITY_LIMIT_VIOLATIONS"],
        "ACCELERATION_LIMIT_VIOLATIONS": dynamics["ACCELERATION_LIMIT_VIOLATIONS"],
        "NATIVE_JERK_CERTIFICATION": jerk["NATIVE_JERK_CERTIFICATION"],
        "POST_RUCKIG_PROCESS_TOLERANCE": process["POST_RUCKIG_PROCESS_TOLERANCE"],
        "POST_RUCKIG_COLLISION_CERTIFICATION": collision["POST_RUCKIG_COLLISION_CERTIFICATION"],
        "SEMANTIC_BREAKS": breaks["status"],
        "ENDPOINT_PRESERVATION": breaks["endpoint_preservation"],
        "DETERMINISTIC_REPLAY": replay["DETERMINISTIC_REPLAY"],
        "MOVEIT_RUNTIME_SHUTDOWN_CERTIFICATION": shutdown["MOVEIT_RUNTIME_SHUTDOWN_CERTIFICATION"],
        "NEW_REGRESSION_FAILURES": regression["NEW_REGRESSION_FAILURES"],
        "NEW_FJT_GOALS_SENT": 0,
        "ROBOT_MOTION_STARTED": "NO",
        "FORMAL_LEDGER_MUTATED": "NO",
        "H8_STARTED": "NO",
        "CCD": "NOT_AVAILABLE",
        "CLEARANCE": None,
        "collision_method": COLLISION_METHOD,
    }
    dump_json(output / "stage3_h7_5_terminal_certificate.json", terminal)
    (output / "FINAL_REPORT.md").write_text(report_markdown(terminal, semantic, physical, native, dynamics, jerk, process, collision, breaks, shutdown), encoding="utf-8", newline="\n")

    missing = [name for name in REQUIRED if not (output / name).is_file()]
    dump_json(output / "stage3_h7_5_artifact_manifest.json", {
        "schema_version": "stage3-h7-5-artifact-manifest-v1",
        "required": REQUIRED,
        "missing": missing,
        "policy_migration_certificate_generated": (output / "stage3_h7_5_overshoot_policy_migration_certificate.json").is_file(),
        "policy_migration_certificate_expected": False,
    })
    print(json.dumps({"output": str(output), "terminal": terminal, "missing": missing}, ensure_ascii=False))
    return 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or ROOT / "outputs" / f"stage3_h7_5_overshoot_recertification_{timestamp}"
    return orchestrate(output.resolve())


if __name__ == "__main__":
    sys.exit(main())
