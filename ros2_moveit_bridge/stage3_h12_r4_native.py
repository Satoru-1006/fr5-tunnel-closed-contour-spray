#!/usr/bin/env python3
"""Fresh software-only H12-R4 MoveIt2 TOTG/Ruckig primitive worker.

No controller, driver, action client, trajectory action goal, or physical
robot API is imported. Historical timestamps are diagnostics only; TOTG is
given the reconstructed joint-space geometry and creates new timing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from geometry_msgs.msg import Pose

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import rclpy
from moveit.core.robot_state import RobotState
from moveit.core.robot_trajectory import RobotTrajectory
from moveit.planning import MoveItPy

import stage3_h7_2_native as base
import stage3_h7_3_native as h73
from src.stage3_h12_r4 import (
    analyze_geometric_path,
    classify_totg_false,
    finite_difference_diagnostics_only,
    joint_position_gate,
    numeric_tolerance,
    ruckig_completion_gate,
)
from src.stage3_h12_r6 import reconstruct_local_derivative_state
from src.stage3_h12_r7 import bounded_seeds, candidate_score, immutable_sample_hash, repair_eligible

SCHEMA_VERSION = "stage3_h12_r4_native_primitive_v1"
JOINT_NAMES = ["j1", "j2", "j3", "j4", "j5", "j6"]
PROBE_FILES = {
    "summaries": "stage25r_native_run_summaries.jsonl",
    "calls": "stage25r_ruckig_calls.jsonl",
    "overshoots": "stage25r2_overshoot_events.jsonl",
    "samples": "stage25r_native_samples.jsonl",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def probe_directory() -> Path | None:
    raw = os.environ.get("STAGE25R_NATIVE_DIR")
    return Path(raw) if raw else None


def probe_offsets() -> dict[str, int]:
    directory = probe_directory()
    if directory is None:
        return {key: 0 for key in PROBE_FILES}
    return {key: (directory / name).stat().st_size if (directory / name).is_file() else 0 for key, name in PROBE_FILES.items()}


def read_probe_delta(key: str, offset: int, *, parse: bool = True) -> list[Any]:
    directory = probe_directory()
    if directory is None:
        return []
    path = directory / PROBE_FILES[key]
    if not path.is_file() or path.stat().st_size <= offset:
        return []
    with path.open("rb") as stream:
        stream.seek(offset)
        raw = stream.read().decode("utf-8", errors="replace")
    lines = [line for line in raw.splitlines() if line.strip()]
    return [json.loads(line) for line in lines] if parse else lines


def make_trajectory(moveit: MoveItPy, positions: np.ndarray) -> RobotTrajectory:
    start_state = RobotState(moveit.get_robot_model())
    trajectory = RobotTrajectory(moveit.get_robot_model())
    trajectory.joint_model_group_name = base.GROUP_NAME
    trajectory.set_robot_trajectory_msg(start_state, base.make_trajectory_message([{"joint_values": row.tolist()} for row in positions]))
    trajectory.unwind()
    return trajectory


def _pose(position: np.ndarray, quaternion: np.ndarray) -> Pose:
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = (float(value) for value in position)
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = (float(value) for value in quaternion)
    return pose


def localized_tcp_reprojection(
    moveit: MoveItPy,
    primitive: Mapping[str, Any],
    item: Mapping[str, Any],
    positions: np.ndarray,
    times: np.ndarray,
    process_rows: list[Mapping[str, Any]],
    bounds: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    """Repair only failed post-TOTG SPRAY_ON samples with bounded local IK."""
    original = np.asarray(positions, dtype=float)
    repaired = original.copy()
    state = RobotState(moveit.get_robot_model())
    lower = np.asarray([bounds[name]["position_lower_rad"] for name in JOINT_NAMES], dtype=float)
    upper = np.asarray([bounds[name]["position_upper_rad"] for name in JOINT_NAMES], dtype=float)
    eligible = [row for row in process_rows if repair_eligible(
        spray_state=str(item["spray_state"]), segment_id=int(item["segment_id"]),
        tcp_position_error_m=float(row.get("tcp_position_error_m", 0.0)),
    )]
    events: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    max_before = max((float(row["tcp_position_error_m"]) for row in eligible), default=0.0)
    for failure in eligible:
        index = int(failure["trajectory_index"])
        q_before = repaired[index].copy()
        target_position = np.asarray(failure["reference_tcp_position_xyz_m"], dtype=float)
        # Freeze the current authoritative FK orientation while correcting position.
        target_orientation = np.asarray(failure["tcp_orientation_xyzw"], dtype=float)
        candidates: list[tuple[tuple[Any, ...], str, np.ndarray, Mapping[str, Any]]] = []
        attempt_records: list[dict[str, Any]] = []
        for seed_name, seed in bounded_seeds(repaired, index):
            state.set_joint_group_positions(base.GROUP_NAME, seed.tolist())
            state.update()
            try:
                solved = bool(state.set_from_ik(base.GROUP_NAME, _pose(target_position, target_orientation), base.EE_LINK, 0.005))
                error = None
            except Exception as exc:
                solved = False
                error = f"{type(exc).__name__}:{exc}"
            attempt = {"seed": seed_name, "ik_solved": solved, "error": error}
            if not solved:
                attempt_records.append(attempt)
                continue
            state.update()
            candidate = np.asarray(state.get_joint_group_positions(base.GROUP_NAME), dtype=float)
            joint_valid = bool(np.all(np.isfinite(candidate)) and np.all(candidate >= lower - 1e-10) and np.all(candidate <= upper + 1e-10))
            candidate_rows, candidate_summary = base.process_validation(moveit, primitive, candidate[None, :], np.asarray([times[index]]), contract)
            candidate_process = candidate_rows[0]
            collision_rows, collision_summary = base.collision_validate(moveit, np.asarray([times[index]]), candidate[None, :])
            collision_free = bool(collision_summary.get("status") == "PASSED")
            valid = bool(joint_valid and collision_free and candidate_process.get("process_tolerance_pass") is True)
            attempt.update({
                "joint_valid": joint_valid, "collision_free": collision_free,
                "tcp_position_error_m": candidate_process.get("tcp_position_error_m"),
                "tcp_orientation_error_deg": candidate_process.get("tcp_orientation_error_deg"),
                "valid": valid,
            })
            attempt_records.append(attempt)
            if valid:
                previous = repaired[index - 1] if index > 0 else None
                following = repaired[index + 1] if index + 1 < len(repaired) else None
                candidates.append((candidate_score(q_before, candidate, previous, following), seed_name, candidate, candidate_process))
        if not candidates:
            failures.append({"trajectory_index": index, "tcp_error_before_m": float(failure["tcp_position_error_m"]), "attempts": attempt_records})
            continue
        _, seed_name, q_after, after_process = min(candidates, key=lambda row: row[0])
        repaired[index] = q_after
        events.append({
            "schema_version": "stage3_h12_r7_repaired_sample_v1",
            "family": str(item["trajectory_family_id"]), "primitive": str(item["primitive_id"]),
            "primitive_instance": str(item["unit_id"]), "segment": int(item["segment_id"]),
            "sample_identity": {"trajectory_index": index, "time_from_start_s": float(times[index])},
            "q_before": q_before.tolist(), "q_after": q_after.tolist(), "selected_seed": seed_name,
            "tcp_error_before": float(failure["tcp_position_error_m"]),
            "tcp_error_after": float(after_process["tcp_position_error_m"]),
            "joint_delta_norm": float(np.linalg.norm(q_after - q_before)),
        })
    repaired_indices = [int(row["sample_identity"]["trajectory_index"]) for row in events]
    summary = {
        "schema_version": "stage3_h12_r7_reprojection_summary_v1", "repair_layer": "POST_TOTG_PRE_RUCKIG",
        "eligible_count": len(eligible), "repaired_count": len(events), "repair_failure_count": len(failures),
        "repair_failures": failures, "max_tcp_error_before_m": max_before,
        "max_tcp_error_after_m": max((float(row["tcp_error_after"]) for row in events), default=0.0),
        "max_joint_correction_rad": max((float(row["joint_delta_norm"]) for row in events), default=0.0),
        "immutable_sample_hash_before": immutable_sample_hash(original, repaired_indices),
        "immutable_sample_hash_after": immutable_sample_hash(repaired, repaired_indices),
        "immutable_samples_unchanged": immutable_sample_hash(original, repaired_indices) == immutable_sample_hash(repaired, repaired_indices),
    }
    return repaired, summary, events


def execute_unit(
    moveit: MoveItPy,
    primitive: Mapping[str, Any],
    item: Mapping[str, Any],
    positions: np.ndarray,
    old_times: np.ndarray,
    runtime_audit: Mapping[str, Any],
    contract: Mapping[str, Any],
    totg_parameters: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    bounds = runtime_audit["runtime_bounds"]
    lower = [bounds[name]["position_lower_rad"] for name in JOINT_NAMES]
    upper = [bounds[name]["position_upper_rad"] for name in JOINT_NAMES]
    velocity_limits = [bounds[name]["velocity_rad_s"] * float(totg_parameters["velocity_scaling_factor"]) for name in JOINT_NAMES]
    acceleration_limits = [bounds[name]["acceleration_rad_s2"] * float(totg_parameters["acceleration_scaling_factor"]) for name in JOINT_NAMES]
    geometry = analyze_geometric_path(positions, min_angle_change=float(totg_parameters["min_angle_change"]))
    position_gate = joint_position_gate(positions, lower, upper)
    old_diagnostics = finite_difference_diagnostics_only(old_times, positions, {"velocity": velocity_limits, "acceleration": acceleration_limits})
    group_valid = bool(moveit.get_robot_model().get_joint_model_group(base.GROUP_NAME))
    velocity_valid = bool(all(math.isfinite(value) and value > 0.0 for value in velocity_limits))
    acceleration_valid = bool(all(math.isfinite(value) and value > 0.0 for value in acceleration_limits))
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "execution_unit": "trajectory_primitive",
        "unit_index": int(item["unit_index"]),
        "unit_id": str(item["unit_id"]),
        "primitive_id": str(item["primitive_id"]),
        "segment_id": int(item["segment_id"]),
        "segment_order": int(item["segment_order"]),
        "segment_key": str(item["segment_key"]),
        "trajectory_family_id": str(item["trajectory_family_id"]),
        "spray_mode": str(item["spray_state"]),
        "source_window_ids": [int(value) for value in item["window_ids"]],
        "window_count": int(item["window_count"]),
        "waypoint_count": int(len(positions)),
        "joint_names": JOINT_NAMES,
        **geometry,
        **position_gate,
        "old_timestamp_diagnostics_only": old_diagnostics,
        "geometric_path_validity": bool(geometry.get("geometric_shape_valid") and geometry.get("input_position_finite") and position_gate["input_joint_limit_valid"]),
        "totg_called": False,
        "totg_return_value": False,
        "totg_exception": None,
        "totg_wall_time": None,
        "native_log_reference": os.environ.get("H12_R4_NATIVE_LOG_REFERENCE"),
        "repair_attempted": False,
        "repair_class": None,
        "repair_reason": "no evidence-supported deterministic geometric repair selected",
        "CCD": "not_available",
        "clearance": None,
        "collision_method": base.COLLISION_METHOD,
    }
    trajectory = make_trajectory(moveit, positions)
    r7a_audit = os.environ.get("H12_R7A_PROCESS_AUDIT", "0").strip().lower() in {"1", "true", "yes"}
    audit_stages: dict[str, Any] = {}
    if r7a_audit:
        pre_rows, pre_summary = base.process_validation(moveit, primitive, positions, old_times - old_times[0], contract)
        audit_stages["PRE_TOTG"] = {"summary": pre_summary, "failures": [row for row in pre_rows if row.get("process_tolerance_pass") is False]}
    before_totg = time.perf_counter()
    record["totg_called"] = True
    try:
        totg_ok = bool(trajectory.apply_totg_time_parameterization(
            float(totg_parameters["velocity_scaling_factor"]),
            float(totg_parameters["acceleration_scaling_factor"]),
            path_tolerance=float(totg_parameters["path_tolerance"]),
            resample_dt=float(totg_parameters["resample_dt"]),
            min_angle_change=float(totg_parameters["min_angle_change"]),
        ))
    except Exception as exc:
        totg_ok = False
        record["totg_exception"] = f"{type(exc).__name__}:{exc}"
    record["totg_wall_time"] = time.perf_counter() - before_totg
    record["totg_return_value"] = totg_ok
    if not totg_ok:
        failure_class, failure_detail = classify_totg_false(
            geometry,
            group_valid=group_valid,
            velocity_limits_valid=velocity_valid,
            acceleration_limits_valid=acceleration_valid,
            path_tolerance=float(totg_parameters["path_tolerance"]),
            exception=record["totg_exception"],
        )
        record.update({
            "output_waypoint_count": 0,
            "output_duration": None,
            "output_positions_finite": None,
            "output_velocities_finite": None,
            "output_accelerations_finite": None,
            "failure_class": failure_class,
            "failure_detail": failure_detail,
            "status": "BLOCKED",
            "first_blocker": failure_class,
            "ruckig_attempted": False,
        })
        return record

    totg_t, totg_q, totg_dq, totg_ddq = base.message_arrays(trajectory.get_robot_trajectory_msg())
    r7_reproject = os.environ.get("H12_R7_REPROJECT", "0").strip().lower() in {"1", "true", "yes"}
    totg_process_rows: list[dict[str, Any]] = []
    totg_process_summary: dict[str, Any] = {}
    if r7a_audit or r7_reproject:
        totg_process_rows, totg_process_summary = base.process_validation(moveit, primitive, totg_q, totg_t, contract)
    if r7a_audit:
        audit_stages["POST_TOTG_PRE_RUCKIG"] = {"summary": totg_process_summary, "failures": [row for row in totg_process_rows if row.get("process_tolerance_pass") is False]}
    r7_summary = None
    r7_events: list[dict[str, Any]] = []
    if r7_reproject:
        repaired_q, r7_summary, r7_events = localized_tcp_reprojection(
            moveit, primitive, item, totg_q, totg_t, totg_process_rows, bounds, contract,
        )
        repaired_rows, repaired_process = base.process_validation(moveit, primitive, repaired_q, totg_t, contract)
        r7_summary["post_reprojection_spray_process_violations"] = int(repaired_process["failed_spray_on_sample_count"])
        r7_summary["post_reprojection_max_tcp_error_m"] = repaired_process["max_tcp_position_error_m"]
        r7_summary["status"] = "PASSED" if (
            r7_summary["repair_failure_count"] == 0
            and r7_summary["immutable_samples_unchanged"]
            and repaired_process["failed_spray_on_sample_count"] == 0
        ) else "BLOCKED"
        if r7a_audit:
            audit_stages["POST_REPROJECTION_PRE_RUCKIG"] = {
                "summary": repaired_process,
                "failures": [row for row in repaired_rows if row.get("process_tolerance_pass") is False],
            }
        repair_dir = output_dir.parent / "h12_r7_repair_units"
        repair_dir.mkdir(parents=True, exist_ok=True)
        (repair_dir / f"{int(item['unit_index']):05d}.json").write_text(
            json.dumps(json_safe({"summary": r7_summary, "events": r7_events}), ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
            encoding="utf-8", newline="\n",
        )
        if r7_summary["status"] != "PASSED":
            record.update({
                "output_waypoint_count": int(len(totg_t)), "output_duration": float(totg_t[-1]),
                "post_totg_dynamic": base.dynamics_validation(totg_q, totg_dq, totg_ddq, totg_t, bounds, "post_totg"),
                "h12_r7_reprojection": r7_summary, "status": "BLOCKED",
                "first_blocker": "h12_r7_local_reprojection_failed", "ruckig_attempted": False,
            })
            return record
        # Preserve the native TOTG timing/derivative message and change only
        # the explicitly repaired post-TOTG joint positions before Ruckig.
        repaired_message = trajectory.get_robot_trajectory_msg()
        for point, q in zip(repaired_message.joint_trajectory.points, repaired_q):
            point.positions = q.tolist()
        repaired_start = RobotState(moveit.get_robot_model())
        trajectory.set_robot_trajectory_msg(repaired_start, repaired_message)
        totg_q = repaired_q
    record.update({
        "output_waypoint_count": int(len(totg_t)),
        "output_duration": float(totg_t[-1]),
        "output_positions_finite": bool(np.all(np.isfinite(totg_q))),
        "output_velocities_finite": bool(np.all(np.isfinite(totg_dq))),
        "output_accelerations_finite": bool(np.all(np.isfinite(totg_ddq))),
        "failure_class": None,
        "failure_detail": None,
        "post_totg_dynamic": base.dynamics_validation(totg_q, totg_dq, totg_ddq, totg_t, bounds, "post_totg"),
        "h12_r7_reprojection": r7_summary,
        "ruckig_attempted": True,
    })
    offsets = probe_offsets()
    before_ruckig = time.perf_counter()
    mitigate_overshoot = os.environ.get("H12_R5_MITIGATE_OVERSHOOT", "1").strip().lower() not in {"0", "false", "no"}
    overshoot_threshold = float(os.environ.get("H12_R5_OVERSHOOT_THRESHOLD", "0.01"))
    try:
        smoothing_bool = bool(trajectory.apply_ruckig_smoothing(
            float(totg_parameters["velocity_scaling_factor"]),
            float(totg_parameters["acceleration_scaling_factor"]),
            mitigate_overshoot=mitigate_overshoot,
            overshoot_threshold=overshoot_threshold,
        ))
        native_exception = None
    except Exception as exc:
        smoothing_bool = False
        native_exception = f"{type(exc).__name__}:{exc}"
    native_compute_time = time.perf_counter() - before_ruckig
    summaries = read_probe_delta("summaries", offsets["summaries"])
    calls = read_probe_delta("calls", offsets["calls"])
    overshoots = read_probe_delta("overshoots", offsets["overshoots"], parse=False)
    summary = summaries[-1] if summaries else {}
    validations = [(row.get("native_validate_input") or {}).get("current_and_target_strict") for row in calls]
    input_validation_passed = bool(calls and all(value is True for value in validations))
    raw_result = summary.get("final_ruckig_result") or {}
    smoothing_complete = summary.get("smoothing_complete") is True
    last_segment_completed = bool(smoothing_complete and int(summary.get("last_called_waypoint_idx", -1)) == len(totg_t) - 2)
    try:
        final_t, final_q, final_dq, final_ddq = base.message_arrays(trajectory.get_robot_trajectory_msg())
        output_finite = bool(np.all(np.isfinite(final_t)) and np.all(np.isfinite(final_q)) and np.all(np.isfinite(final_dq)) and np.all(np.isfinite(final_ddq)))
    except Exception:
        final_t = final_q = final_dq = final_ddq = None
        output_finite = False
    ruckig_record = {
        "native_smoothing_returned_success": smoothing_bool,
        "ruckig_result": raw_result,
        "last_segment_completed": last_segment_completed,
        "smoothing_complete": smoothing_complete,
        "overshoot_gate_passed": bool(smoothing_complete),
        "duration_ceiling_hit": summary.get("duration_ceiling_hit") is True,
        "iteration_limit_hit": False,
        "wall_clock_timeout": False,
        "input_validation_passed": input_validation_passed,
        "native_error": native_exception,
        "output_finite": output_finite,
        "iteration_count": int(summary.get("calls_total", len(calls))),
        "accepted_segment_calls": int(summary.get("accepted_segment_calls", 0)),
        "retry_count": int(summary.get("retry_calls", 0)),
        "overshoot_retry_event_count": len(overshoots),
        "simulated_trajectory_duration": None if final_t is None else float(final_t[-1]),
        "native_compute_time": native_compute_time,
        "requested_duration": float(totg_t[-1]),
        "final_duration": None if final_t is None else float(final_t[-1]),
        "last_called_waypoint_idx": summary.get("last_called_waypoint_idx"),
        "probe_summary": summary or None,
        "probe_call_count": len(calls),
        "mitigate_overshoot": mitigate_overshoot,
        "overshoot_threshold": overshoot_threshold,
    }
    completion = ruckig_completion_gate(ruckig_record)
    record.update({"ruckig": ruckig_record, "ruckig_completion_gate": completion})
    if not completion["accepted"]:
        if ruckig_record["duration_ceiling_hit"] and ruckig_record["overshoot_retry_event_count"]:
            blocker = "ruckig_overshoot_mitigation_duration_ceiling"
        elif ruckig_record["duration_ceiling_hit"]:
            blocker = "ruckig_duration_ceiling"
        elif native_exception:
            blocker = "ruckig_native_error"
        elif not input_validation_passed:
            blocker = "ruckig_input_invalid"
        else:
            blocker = "ruckig_smoothing_incomplete"
        record.update({"status": "BLOCKED", "first_blocker": blocker, "post_ruckig_certification_reached": False})
        return record

    assert final_t is not None and final_q is not None and final_dq is not None and final_ddq is not None
    derivative_reconditioning = None
    if os.environ.get("H12_R6_RECONDITION_DERIVATIVES", "0").strip().lower() in {"1", "true", "yes"}:
        try:
            reconstructed, derivative_reconditioning = reconstruct_local_derivative_state(
                final_t,
                final_dq,
                final_ddq,
                np.asarray([bounds[name]["acceleration_rad_s2"] for name in JOINT_NAMES], dtype=np.float64),
            )
            final_ddq = reconstructed
        except Exception as exc:
            record.update({
                "status": "BLOCKED",
                "first_blocker": f"h12_r6_derivative_reconditioning_failed:{type(exc).__name__}:{exc}",
                "post_ruckig_certification_reached": True,
                "h12_r6_derivative_reconditioning": {"status": "BLOCKED", "error": f"{type(exc).__name__}:{exc}"},
            })
            return record
    dynamic = base.dynamics_validation(final_q, final_dq, final_ddq, final_t, bounds, "post_ruckig")
    analytic_position_violations = 0
    analytic_velocity_violations = 0
    analytic_acceleration_violations = 0
    analytic_jerk_violations = 0
    for call in calls:
        native_output = call.get("native_output") or {}
        position_extrema = native_output.get("position_extrema") or []
        kinematic_extrema = native_output.get("kinematic_extrema") or []
        input_limits = call.get("input") or {}
        for joint, name in enumerate(JOINT_NAMES):
            if joint < len(position_extrema):
                ext = position_extrema[joint]
                tolerance = numeric_tolerance(float(ext["max"]), float(bounds[name]["position_upper_rad"]))
                analytic_position_violations += int(float(ext["max"]) > float(bounds[name]["position_upper_rad"]) + tolerance)
                tolerance = numeric_tolerance(float(ext["min"]), float(bounds[name]["position_lower_rad"]))
                analytic_position_violations += int(float(ext["min"]) < float(bounds[name]["position_lower_rad"]) - tolerance)
            if joint < len(kinematic_extrema):
                ext = kinematic_extrema[joint]
                for key, limits_key, counter_name in (
                    ("max_abs_velocity", "max_velocity", "velocity"),
                    ("max_abs_acceleration", "max_acceleration", "acceleration"),
                    ("max_abs_jerk", "max_jerk", "jerk"),
                ):
                    value = float(ext[key])
                    limit = float((input_limits.get(limits_key) or [math.nan] * 6)[joint])
                    violation = int(not math.isfinite(limit) or value > limit + numeric_tolerance(value, limit))
                    if counter_name == "velocity": analytic_velocity_violations += violation
                    elif counter_name == "acceleration": analytic_acceleration_violations += violation
                    else: analytic_jerk_violations += violation
    dynamic["native_analytic_position_violation_count"] = analytic_position_violations
    dynamic["native_analytic_velocity_violation_count"] = analytic_velocity_violations
    dynamic["native_analytic_acceleration_violation_count"] = analytic_acceleration_violations
    dynamic["position_limit_violation_count"] = int(dynamic.get("position_limit_violation_count", 0)) + analytic_position_violations
    dynamic["velocity_limit_violation_count"] = int(dynamic.get("velocity_limit_violation_count", 0)) + analytic_velocity_violations
    dynamic["acceleration_limit_violation_count"] = int(dynamic.get("acceleration_limit_violation_count", 0)) + analytic_acceleration_violations
    if analytic_position_violations or analytic_velocity_violations or analytic_acceleration_violations:
        dynamic["status"] = "BLOCKED"
    process_rows, process = base.process_validation(moveit, primitive, final_q, final_t, contract)
    if r7a_audit:
        audit_stages["POST_RUCKIG"] = {"summary": process, "failures": [row for row in process_rows if row.get("process_tolerance_pass") is False]}
        audit_path = output_dir.parent / "h12_r7a_process_stages" / f"{int(item['unit_index']):05d}.json"
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_payload = {
            "schema_version": "stage3_h12_r7a_native_process_stage_audit_v1",
            "unit_index": int(item["unit_index"]), "unit_id": str(item["unit_id"]),
            "trajectory_family_id": str(item["trajectory_family_id"]), "primitive_id": str(item["primitive_id"]),
            "segment_id": int(item["segment_id"]), "segment_order": int(item["segment_order"]),
            "spray_state": str(item["spray_state"]), "stages": audit_stages,
        }
        audit_path.write_text(json.dumps(json_safe(atomic_payload), ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    collision_rows, collision = base.collision_validate(moveit, final_t, final_q)
    collision["self_collision_failure_count"] = sum(bool(row.get("self_collision")) for row in collision_rows)
    collision["environment_collision_failure_count"] = sum(bool(row.get("environment_collision")) for row in collision_rows)
    # Native Ruckig samples are read only for a completed invocation.
    sample_rows = read_probe_delta("samples", offsets["samples"])
    jerk_limits = np.asarray([bounds[name]["jerk_rad_s3"] for name in JOINT_NAMES], dtype=float)
    jerk_violation_count = analytic_jerk_violations
    for row in sample_rows:
        native_jerk = np.asarray(row.get("native_jerk", []), dtype=float)
        if native_jerk.shape == (6,):
            jerk_violation_count += int(np.count_nonzero(np.abs(native_jerk) > jerk_limits + np.asarray([numeric_tolerance(value, limit) for value, limit in zip(native_jerk, jerk_limits)])))
    jerk = {"validation_method": "native_Ruckig_analytic_profile_extrema_and_optional_phase_samples", "sample_count": len(sample_rows), "profile_count": len(calls), "violation_count": jerk_violation_count, "status": "PASSED" if calls and jerk_violation_count == 0 else "BLOCKED"}
    endpoint_delta = float(max(np.max(np.abs(final_q[0] - totg_q[0])), np.max(np.abs(final_q[-1] - totg_q[-1]))))
    passed = bool(dynamic["status"] == process["status"] == collision["status"] == jerk["status"] == "PASSED" and endpoint_delta <= 1.0e-10)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / f"{int(item['unit_index']):05d}.npz", time_s=final_t, positions_rad=final_q, velocities_rad_s=final_dq, accelerations_rad_s2=final_ddq)
    record.update({
        "post_ruckig_certification_reached": True,
        "post_ruckig_dynamic": dynamic,
        "post_ruckig_jerk": jerk,
        "post_ruckig_process": process,
        "post_ruckig_collision": collision,
        "post_ruckig_endpoint_delta_rad": endpoint_delta,
        "h12_r6_derivative_reconditioning": derivative_reconditioning,
        "post_ruckig_arrays_file": f"post_ruckig_units/{int(item['unit_index']):05d}.npz",
        "status": "PASSED" if passed else "BLOCKED",
        "first_blocker": None if passed else "post_ruckig_certification_failed",
    })
    return record


def run(args: argparse.Namespace, moveit: MoveItPy) -> int:
    bundle = np.load(Path(args.input_npz), allow_pickle=False)
    positions = np.asarray(bundle["positions_rad"], dtype=np.float64)
    times = np.asarray(bundle["times_s"], dtype=np.float64)
    lengths = np.asarray(bundle["lengths"], dtype=np.int64)
    units = load_jsonl(Path(args.unit_manifest))
    validation = load_jsonl(Path(args.h6_validation))
    segments = load_json(Path(args.h6_segments))
    primitive_map = {str(item["primitive_id"]): item for item in h73.derive_primitives(validation, segments)}
    contract = load_json(Path(args.process_contract))
    totg = load_json(Path(args.totg_parameters))
    base.apply_fixture(moveit, Path(args.fixture_mesh))
    runtime_audit = base.runtime_limits(moveit, Path(args.runtime_limits))
    if runtime_audit.get("status") != "PASSED":
        raise RuntimeError("runtime_joint_limit_audit_failed")
    output = Path(args.output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for unit in units:
            index = int(unit["unit_index"])
            primitive = primitive_map.get(str(unit["primitive_id"]))
            try:
                if primitive is None:
                    raise RuntimeError("primitive_missing")
                length = int(lengths[index])
                record = execute_unit(moveit, primitive, unit, positions[index, :length], times[index, :length], runtime_audit, contract, totg, Path(args.output_dir) / "post_ruckig_units")
            except Exception as exc:
                record = {**unit, "schema_version": SCHEMA_VERSION, "totg_called": False, "totg_return_value": False, "status": "BLOCKED", "first_blocker": f"native_exception:{type(exc).__name__}:{exc}", "failure_class": "native_exception", "failure_detail": f"{type(exc).__name__}:{exc}", "native_runtime_error": True}
            stream.write(json.dumps(json_safe(record), ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
            stream.flush()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("input_npz", "unit_manifest", "output_jsonl", "output_dir", "h6_validation", "h6_segments", "process_contract", "fixture_mesh", "runtime_limits", "totg_parameters"):
        parser.add_argument(f"--{name.replace('_', '-')}", required=True)
    args, _ = parser.parse_known_args()
    rclpy.init()
    moveit = None
    try:
        moveit = MoveItPy(node_name="stage3_h12_r4_native")
        return run(args, moveit)
    except Exception as exc:
        Path(args.output_jsonl).write_text(json.dumps({"schema_version": SCHEMA_VERSION, "status": "BLOCKED", "first_blocker": f"native_exception:{type(exc).__name__}:{exc}", "failure_class": "native_exception"}, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        return 2
    finally:
        if moveit is not None:
            moveit.shutdown()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
