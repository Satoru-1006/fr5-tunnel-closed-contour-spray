#!/usr/bin/env python3
"""Native MoveIt worker for Stage 3 H7.2.

Each H6.4 process segment is converted into an independent RobotTrajectory.
No planner, IK solver, controller, action client, or robot execution API is
used here.  The worker performs native TOTG and (when explicitly authorized)
native MoveIt Ruckig smoothing, followed by FK, process, dynamics, and
PlanningScene/FCL validation of the actual returned trajectories.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from moveit.core.robot_state import RobotState
from moveit.core.robot_trajectory import RobotTrajectory
from moveit.planning import MoveItPy

from plan_closed_contour_moveit import _rotation_matrix_to_quaternion, _transform_matrix
from stage3_h7_native import (
    GROUP_NAME,
    JOINT_NAMES,
    apply_fixture,
    collision_validate,
    duration_seconds,
    make_trajectory_message,
    param,
    parse_limits,
)

EE_LINK = "spray_tcp_link"
SEGMENT_ORDER = [0, 1000, 1, 1001, 2, 1002, 3, 1003, 4]
COLLISION_METHOD = "adaptive_discrete_interpolation"
ZERO_VELOCITY_TOLERANCE_RAD_S = 1.0e-3


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def dump_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n" for row in rows), encoding="utf-8", newline="\n")


def canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): canonical(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    if isinstance(value, float):
        return None if not math.isfinite(value) else round(value, 12)
    return value


def semantic_hash(value: Any) -> str:
    import hashlib

    raw = json.dumps(canonical(value), ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def message_arrays(message: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points = message.joint_trajectory.points
    if len(points) < 2:
        raise RuntimeError("native trajectory has fewer than two points")
    t = np.asarray([duration_seconds(point.time_from_start) for point in points], dtype=float)
    q = np.asarray([list(point.positions) for point in points], dtype=float)
    dq = np.asarray([list(point.velocities) for point in points], dtype=float)
    ddq = np.asarray([list(point.accelerations) for point in points], dtype=float)
    if q.shape != (len(points), 6) or dq.shape != q.shape or ddq.shape != q.shape:
        raise RuntimeError("native trajectory state arrays are incomplete")
    return t, q, dq, ddq


def runtime_limits(moveit: MoveItPy, configured_path: Path) -> dict[str, Any]:
    model = moveit.get_robot_model()
    configured = parse_limits(configured_path)
    group = model.get_joint_model_group(GROUP_NAME)
    active = list(group.active_joint_model_names)
    errors: list[str] = []
    if active != JOINT_NAMES:
        errors.append(f"unexpected_joint_order:{active}")
    bounds_by_joint: dict[str, Any] = {}
    for name, raw in zip(active, group.active_joint_model_bounds):
        bound = raw[0] if isinstance(raw, (list, tuple)) else raw
        max_jerk = getattr(bound, "max_jerk", None)
        runtime = {
            "position_lower_rad": float(bound.min_position),
            "position_upper_rad": float(bound.max_position),
            "velocity_rad_s": float(bound.max_velocity),
            "acceleration_rad_s2": float(bound.max_acceleration),
            "jerk_rad_s3": float(max_jerk) if max_jerk is not None and math.isfinite(float(max_jerk)) else None,
            "jerk_bounded": bool(max_jerk is not None and math.isfinite(float(max_jerk)) and float(max_jerk) > 0.0),
        }
        bounds_by_joint[name] = runtime
        expected = configured.get(name, {})
        for key, field in (("max_velocity", "velocity_rad_s"), ("max_acceleration", "acceleration_rad_s2"), ("max_jerk", "jerk_rad_s3")):
            expected_value = expected.get(key)
            actual_value = runtime.get(field)
            if expected_value is None or actual_value is None or abs(float(expected_value) - float(actual_value)) > 1e-12:
                errors.append(f"runtime_limit_mismatch:{name}:{key}")
    return {
        "status": "PASSED" if not errors else "BLOCKED",
        "robot_model_name": str(model.name),
        "active_joint_names": active,
        "runtime_bounds": bounds_by_joint,
        "configured_bounds": configured,
        "native_totg_api_available": hasattr(RobotTrajectory, "apply_totg_time_parameterization"),
        "native_ruckig_api_available": hasattr(RobotTrajectory, "apply_ruckig_smoothing"),
        "errors": errors,
    }


def group_input_rows(validation_rows: Sequence[Mapping[str, Any]], segment_manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    metadata = {int(row["segment_id"]): row for row in segment_manifest["segments"]}
    grouped: dict[int, list[dict[str, Any]]] = {segment_id: [] for segment_id in SEGMENT_ORDER}
    for process_index, source in enumerate(validation_rows):
        segment_id = int(source["segment_id"])
        if segment_id not in grouped:
            raise RuntimeError(f"unexpected H6.4 segment {segment_id}")
        row = dict(source)
        row["h6_4_process_waypoint_index"] = int(source["waypoint_index"])
        grouped[segment_id].append(row)
    result: list[dict[str, Any]] = []
    for segment_id in SEGMENT_ORDER:
        rows = grouped[segment_id]
        meta = metadata[segment_id]
        expected_count = int(meta["waypoint_end"]) - int(meta["waypoint_start"]) + 1
        if len(rows) != expected_count:
            raise RuntimeError(f"segment {segment_id} count {len(rows)} != {expected_count}")
        for local_index, row in enumerate(rows):
            row["segment_local_waypoint_index"] = local_index
            row["h6_4_storage_waypoint_index"] = int(meta["waypoint_start"]) + local_index
            row["joint_values"] = [float(value) for value in row["selected_joint_values"]]
        result.append({"segment_id": segment_id, "spray_state": str(meta["spray_state"]), "metadata": dict(meta), "rows": rows})
    return result


def nearest_polyline_projection(sample: np.ndarray, raw_q: np.ndarray) -> tuple[int, float, float]:
    best_edge, best_alpha, best_distance = 0, 0.0, float("inf")
    for edge in range(len(raw_q) - 1):
        delta = raw_q[edge + 1] - raw_q[edge]
        denom = float(np.dot(delta, delta))
        alpha = 0.0 if denom <= 1e-24 else float(np.clip(np.dot(sample - raw_q[edge], delta) / denom, 0.0, 1.0))
        distance = float(np.linalg.norm(sample - (raw_q[edge] + alpha * delta)))
        if distance < best_distance:
            best_edge, best_alpha, best_distance = edge, alpha, distance
    return best_edge, best_alpha, best_distance


def slerp(left: Sequence[float], right: Sequence[float], alpha: float) -> np.ndarray:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b, dot = -b, -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        out = a + alpha * (b - a)
        return out / np.linalg.norm(out)
    theta = math.acos(dot)
    return (math.sin((1.0 - alpha) * theta) * a + math.sin(alpha * theta) * b) / math.sin(theta)


def quaternion_error_deg(actual: Sequence[float], desired: Sequence[float]) -> float:
    a = np.asarray(actual, dtype=float)
    b = np.asarray(desired, dtype=float)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    return math.degrees(2.0 * math.acos(float(np.clip(abs(np.dot(a, b)), -1.0, 1.0))))


def interpolated_reference(left: Mapping[str, Any], right: Mapping[str, Any], alpha: float) -> dict[str, np.ndarray]:
    desired_left = left["desired_tcp_pose"]
    desired_right = right["desired_tcp_pose"]
    surface_left = np.asarray(left["surface_point"], dtype=float)
    surface_right = np.asarray(right["surface_point"], dtype=float)
    normal_left = np.asarray(left["surface_normal"], dtype=float)
    normal_right = np.asarray(right["surface_normal"], dtype=float)
    normal = (1.0 - alpha) * normal_left + alpha * normal_right
    normal /= np.linalg.norm(normal)
    return {
        "position": (1.0 - alpha) * np.asarray(desired_left["position_xyz_m"], dtype=float) + alpha * np.asarray(desired_right["position_xyz_m"], dtype=float),
        "orientation": slerp(desired_left["orientation_xyzw"], desired_right["orientation_xyzw"], alpha),
        "surface_point": (1.0 - alpha) * surface_left + alpha * surface_right,
        "spray_direction": -normal,
    }


def process_validation(moveit: MoveItPy, segment: Mapping[str, Any], q: np.ndarray, t: np.ndarray, contract: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_rows = segment["rows"]
    raw_q = np.asarray([row["joint_values"] for row in raw_rows], dtype=float)
    state = RobotState(moveit.get_robot_model())
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, (time_s, sample) in enumerate(zip(t, q)):
        edge, alpha, path_error = nearest_polyline_projection(sample, raw_q)
        state.set_joint_group_positions(GROUP_NAME, sample.tolist())
        state.update()
        transform = _transform_matrix(state.get_global_link_transform(EE_LINK))
        actual_position = transform[:3, 3].astype(float)
        actual_orientation = np.asarray(_rotation_matrix_to_quaternion(transform[:3, :3]), dtype=float)
        base = {
            "segment_id": int(segment["segment_id"]), "spray_state": segment["spray_state"], "trajectory_index": index,
            "time_from_start_s": float(time_s), "joint_values": sample.tolist(), "reference_edge_local": [edge, edge + 1],
            "reference_alpha": alpha, "joint_path_projection_error_rad": path_error, "fk_valid": True,
        }
        if segment["spray_state"] != "SPRAY_ON":
            base.update({"process_tolerance_applicable": False, "process_tolerance_pass": None})
            records.append(base)
            continue
        reference = interpolated_reference(raw_rows[edge], raw_rows[edge + 1], alpha)
        standoff = float(np.linalg.norm(actual_position - reference["surface_point"]))
        standoff_error = abs(standoff - float(contract["geometry"]["nominal_standoff_m"]))
        actual_direction = transform[:3, 2].astype(float)
        actual_direction /= np.linalg.norm(actual_direction)
        normal_error = math.degrees(math.acos(float(np.clip(np.dot(actual_direction, reference["spray_direction"]), -1.0, 1.0))))
        position_error = float(np.linalg.norm(actual_position - reference["position"]))
        orientation_error = quaternion_error_deg(actual_orientation, reference["orientation"])
        passed = bool(
            standoff_error <= float(contract["geometry"]["standoff_abs_tolerance_m"])
            and normal_error <= float(contract["geometry"]["normal_angle_tolerance_deg"])
            and position_error <= float(contract["tcp_reproduction"]["tcp_position_tolerance_m"])
            and orientation_error <= float(contract["tcp_reproduction"]["tcp_orientation_tolerance_deg"])
        )
        base.update({
            "process_tolerance_applicable": True, "process_tolerance_pass": passed, "actual_standoff_m": standoff,
            "standoff_error_m": standoff_error, "normal_deviation_deg": normal_error, "tcp_position_error_m": position_error,
            "tcp_orientation_error_deg": orientation_error,
            # Read-only audit instrumentation.  These fields expose the exact
            # FK/reference values already used by the authoritative predicate;
            # they do not alter its inputs, limits, tolerance, or aggregation.
            "tcp_position_xyz_m": actual_position.tolist(),
            "tcp_orientation_xyzw": actual_orientation.tolist(),
            "reference_tcp_position_xyz_m": reference["position"].tolist(),
            "reference_tcp_orientation_xyzw": reference["orientation"].tolist(),
            "reference_surface_point_xyz_m": reference["surface_point"].tolist(),
            "reference_spray_direction_unit": reference["spray_direction"].tolist(),
            "standoff_limit_m": float(contract["geometry"]["standoff_abs_tolerance_m"]),
            "normal_limit_deg": float(contract["geometry"]["normal_angle_tolerance_deg"]),
            "tcp_position_limit_m": float(contract["tcp_reproduction"]["tcp_position_tolerance_m"]),
            "tcp_orientation_limit_deg": float(contract["tcp_reproduction"]["tcp_orientation_tolerance_deg"]),
        })
        records.append(base)
        if not passed:
            failures.append(base)
    applicable = [row for row in records if row["process_tolerance_applicable"]]
    def maximum(key: str) -> float | None:
        values = [float(row[key]) for row in applicable]
        return max(values) if values else None
    return records, {
        "status": "PASSED" if not failures else "BLOCKED", "checked_spray_on_sample_count": len(applicable),
        "failed_spray_on_sample_count": len(failures), "max_standoff_error_m": maximum("standoff_error_m"),
        "max_normal_deviation_deg": maximum("normal_deviation_deg"), "max_tcp_position_error_m": maximum("tcp_position_error_m"),
        "max_tcp_orientation_error_deg": maximum("tcp_orientation_error_deg"), "first_failure": failures[0] if failures else None,
        "reference_semantics": "nearest projection on the authoritative H6.4 joint polyline; Cartesian position/surface are linearly interpolated and orientation uses shortest-arc quaternion slerp",
    }


def dynamics_validation(q: np.ndarray, dq: np.ndarray, ddq: np.ndarray, t: np.ndarray, limits: Mapping[str, Any], phase: str) -> dict[str, Any]:
    lower = np.asarray([limits[j]["position_lower_rad"] for j in JOINT_NAMES], dtype=float)
    upper = np.asarray([limits[j]["position_upper_rad"] for j in JOINT_NAMES], dtype=float)
    vmax = np.asarray([limits[j]["velocity_rad_s"] for j in JOINT_NAMES], dtype=float)
    amax = np.asarray([limits[j]["acceleration_rad_s2"] for j in JOINT_NAMES], dtype=float)
    finite = bool(np.all(np.isfinite(t)) and np.all(np.isfinite(q)) and np.all(np.isfinite(dq)) and np.all(np.isfinite(ddq)))
    monotonic_failures = int(np.sum(np.diff(t) <= 0.0))
    position_violations = int(np.sum((q < lower - 1e-10) | (q > upper + 1e-10)))
    velocity_violations = int(np.sum(np.abs(dq) > vmax + 1e-10))
    acceleration_violations = int(np.sum(np.abs(ddq) > amax + 1e-10))
    return {
        "phase": phase, "finite": finite, "nan_inf_count": int(np.size(t) + np.size(q) + np.size(dq) + np.size(ddq) - np.count_nonzero(np.isfinite(t)) - np.count_nonzero(np.isfinite(q)) - np.count_nonzero(np.isfinite(dq)) - np.count_nonzero(np.isfinite(ddq))),
        "non_monotonic_timestamp_count": monotonic_failures, "position_limit_violation_count": position_violations,
        "velocity_limit_violation_count": velocity_violations, "acceleration_limit_violation_count": acceleration_violations,
        "max_abs_velocity_rad_s": np.max(np.abs(dq), axis=0).tolist(), "max_abs_acceleration_rad_s2": np.max(np.abs(ddq), axis=0).tolist(),
        "minimum_positive_dt_s": float(np.min(np.diff(t)[np.diff(t) > 0.0])) if np.any(np.diff(t) > 0.0) else None,
        "start_velocity_rad_s": dq[0].tolist(), "end_velocity_rad_s": dq[-1].tolist(),
        "zero_velocity_boundary_tolerance_rad_s": ZERO_VELOCITY_TOLERANCE_RAD_S,
        "zero_velocity_start": bool(np.max(np.abs(dq[0])) <= ZERO_VELOCITY_TOLERANCE_RAD_S),
        "zero_velocity_end": bool(np.max(np.abs(dq[-1])) <= ZERO_VELOCITY_TOLERANCE_RAD_S),
        "status": "PASSED" if finite and monotonic_failures == position_violations == velocity_violations == acceleration_violations == 0 else "BLOCKED",
    }


def trajectory_rows(segment: Mapping[str, Any], phase: str, t: np.ndarray, q: np.ndarray, dq: np.ndarray, ddq: np.ndarray) -> list[dict[str, Any]]:
    return [{
        "schema_version": "stage3-h7-2-time-parameterized-trajectory-v1", "phase": phase,
        "segment_id": int(segment["segment_id"]), "spray_state": segment["spray_state"], "trajectory_index": index,
        "time_from_start_s": float(time_s), "joint_names": JOINT_NAMES, "positions_rad": positions.tolist(),
        "velocities_rad_s": velocities.tolist(), "accelerations_rad_s2": accelerations.tolist(),
    } for index, (time_s, positions, velocities, accelerations) in enumerate(zip(t, q, dq, ddq))]


def totg_attempt(moveit: MoveItPy, rows: Sequence[Mapping[str, Any]], totg: Mapping[str, Any]) -> bool:
    start_state = RobotState(moveit.get_robot_model())
    trajectory = RobotTrajectory(moveit.get_robot_model())
    trajectory.joint_model_group_name = GROUP_NAME
    trajectory.set_robot_trajectory_msg(start_state, make_trajectory_message(rows))
    trajectory.unwind()
    return bool(trajectory.apply_totg_time_parameterization(
        float(totg["velocity_scaling_factor"]), float(totg["acceleration_scaling_factor"]),
        path_tolerance=float(totg["path_tolerance"]), resample_dt=float(totg["resample_dt"]), min_angle_change=float(totg["min_angle_change"]),
    ))


def isolate_totg_failure(moveit: MoveItPy, segment: Mapping[str, Any], totg: Mapping[str, Any]) -> dict[str, Any]:
    rows = segment["rows"]
    low, high = 3, len(rows)
    tested: list[dict[str, Any]] = []
    while low < high:
        midpoint = (low + high) // 2
        passed = totg_attempt(moveit, rows[:midpoint], totg)
        tested.append({"prefix_waypoint_count": midpoint, "totg_return_status": passed})
        if passed:
            low = midpoint + 1
        else:
            high = midpoint
    failing_count = low
    failing_status = totg_attempt(moveit, rows[:failing_count], totg)
    preceding_status = totg_attempt(moveit, rows[:failing_count - 1], totg) if failing_count > 3 else None
    tested.extend([{"prefix_waypoint_count": failing_count, "totg_return_status": failing_status}, {"prefix_waypoint_count": failing_count - 1, "totg_return_status": preceding_status}])
    triple_rows = rows[max(0, failing_count - 3):failing_count]
    q = [np.asarray(row["joint_values"], dtype=float) for row in triple_rows]
    cosine = None
    turn_angle = None
    deltas: list[list[float]] = []
    if len(q) == 3:
        left, right = q[1] - q[0], q[2] - q[1]
        deltas = [left.tolist(), right.tolist()]
        denom = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denom > 1e-15:
            cosine = float(np.clip(np.dot(left, right) / denom, -1.0, 1.0))
            turn_angle = math.degrees(math.acos(cosine))
    return {
        "diagnostic_method": "native prefix binary search on an unchanged H6.4 segment",
        "minimal_failing_prefix_waypoint_count": failing_count if not failing_status and preceding_status is True else None,
        "minimal_prefix_proven": bool(not failing_status and preceding_status is True),
        "tested_prefixes": sorted(tested, key=lambda row: row["prefix_waypoint_count"]),
        "trigger_local_waypoint_indices": [int(row["segment_local_waypoint_index"]) for row in triple_rows],
        "trigger_h6_4_storage_waypoint_indices": [int(row["h6_4_storage_waypoint_index"]) for row in triple_rows],
        "trigger_h6_4_process_waypoint_indices": [int(row["h6_4_process_waypoint_index"]) for row in triple_rows],
        "joint_values_rad": [item.tolist() for item in q], "joint_deltas_rad": deltas,
        "raw_triple_cos_angle": cosine, "raw_triple_turn_angle_deg": turn_angle,
        "moveit_native_error_classification": "path_requires_180_degree_turn",
        "diagnostic_does_not_modify_authoritative_path": True,
    }


def run_segment(moveit: MoveItPy, segment: Mapping[str, Any], limits: Mapping[str, Any], totg: Mapping[str, Any], ruckig_authorized: bool, contract: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    raw_q = np.asarray([row["joint_values"] for row in segment["rows"]], dtype=float)
    start_state = RobotState(moveit.get_robot_model())
    trajectory = RobotTrajectory(moveit.get_robot_model())
    trajectory.joint_model_group_name = GROUP_NAME
    trajectory.set_robot_trajectory_msg(start_state, make_trajectory_message(segment["rows"]))
    before_unwind = raw_q.copy()
    trajectory.unwind()
    unwind_message = trajectory.get_robot_trajectory_msg()
    after_unwind = np.asarray([list(point.positions) for point in unwind_message.joint_trajectory.points], dtype=float)
    representation_delta = after_unwind - before_unwind
    unwind_semantics_preserved = bool(np.allclose(np.sin(after_unwind), np.sin(before_unwind), atol=1e-12) and np.allclose(np.cos(after_unwind), np.cos(before_unwind), atol=1e-12))
    ok = trajectory.apply_totg_time_parameterization(
        float(totg["velocity_scaling_factor"]), float(totg["acceleration_scaling_factor"]),
        path_tolerance=float(totg["path_tolerance"]), resample_dt=float(totg["resample_dt"]), min_angle_change=float(totg["min_angle_change"]),
    )
    if not ok:
        isolation = isolate_totg_failure(moveit, segment, totg)
        return ({"segment_id": segment["segment_id"], "spray_state": segment["spray_state"], "input_waypoint_count": len(raw_q), "totg_return_status": False, "status": "BLOCKED", "first_blocker": "native_totg_path_requires_180_degree_turn", "failure_isolation": isolation, "unwind_semantics_preserved": unwind_semantics_preserved}, [], [], [])
    totg_t, totg_q, totg_dq, totg_ddq = message_arrays(trajectory.get_robot_trajectory_msg())
    totg_dynamic = dynamics_validation(totg_q, totg_dq, totg_ddq, totg_t, limits, "post_totg")
    totg_process_rows, totg_process = process_validation(moveit, segment, totg_q, totg_t, contract)
    totg_collision_rows, totg_collision = collision_validate(moveit, totg_t, totg_q)
    totg_collision["self_collision_failure_count"] = sum(bool(row.get("self_collision")) for row in totg_collision_rows)
    totg_collision["environment_collision_failure_count"] = sum(bool(row.get("environment_collision")) for row in totg_collision_rows)
    totg_pass = bool(totg_dynamic["status"] == totg_process["status"] == totg_collision["status"] == "PASSED" and totg_dynamic["zero_velocity_start"] and totg_dynamic["zero_velocity_end"] and unwind_semantics_preserved)
    result: dict[str, Any] = {
        "segment_id": int(segment["segment_id"]), "spray_state": segment["spray_state"], "input_waypoint_count": len(raw_q),
        "output_waypoint_count": len(totg_t), "totg_return_status": True, "trajectory_duration_s": float(totg_t[-1]),
        "unwind_applied": True, "unwind_max_abs_representation_delta_rad": float(np.max(np.abs(representation_delta))),
        "unwind_semantics_preserved": unwind_semantics_preserved, "post_totg_dynamics": totg_dynamic,
        "post_totg_process": totg_process, "post_totg_collision": totg_collision, "post_totg_status": "PASSED" if totg_pass else "BLOCKED",
        "ruckig_run": False, "ruckig_status": "NOT_RUN", "status": "PASSED" if totg_pass else "BLOCKED",
    }
    totg_rows = trajectory_rows(segment, "POST_TOTG", totg_t, totg_q, totg_dq, totg_ddq)
    final_rows: list[dict[str, Any]] = []
    validation_rows = [{"phase": "POST_TOTG", "kind": "process", **row} for row in totg_process_rows] + [{"phase": "POST_TOTG", "kind": "collision", **row} for row in totg_collision_rows]
    if not totg_pass or not ruckig_authorized:
        return result, totg_rows, final_rows, validation_rows
    ruckig_ok = trajectory.apply_ruckig_smoothing(float(totg["velocity_scaling_factor"]), float(totg["acceleration_scaling_factor"]), mitigate_overshoot=True, overshoot_threshold=0.01)
    result["ruckig_run"] = True
    result["ruckig_status"] = "PASSED" if ruckig_ok else "BLOCKED"
    if not ruckig_ok:
        result.update({"status": "BLOCKED", "first_blocker": "native_ruckig_returned_false"})
        return result, totg_rows, final_rows, validation_rows
    final_t, final_q, final_dq, final_ddq = message_arrays(trajectory.get_robot_trajectory_msg())
    final_dynamic = dynamics_validation(final_q, final_dq, final_ddq, final_t, limits, "post_ruckig")
    final_process_rows, final_process = process_validation(moveit, segment, final_q, final_t, contract)
    final_collision_rows, final_collision = collision_validate(moveit, final_t, final_q)
    final_collision["self_collision_failure_count"] = sum(bool(row.get("self_collision")) for row in final_collision_rows)
    final_collision["environment_collision_failure_count"] = sum(bool(row.get("environment_collision")) for row in final_collision_rows)
    endpoints_preserved = bool(np.allclose(final_q[0], before_unwind[0], atol=1e-10) and np.allclose(final_q[-1], before_unwind[-1], atol=1e-10))
    final_pass = bool(final_dynamic["status"] == final_process["status"] == final_collision["status"] == "PASSED" and final_dynamic["zero_velocity_start"] and final_dynamic["zero_velocity_end"] and endpoints_preserved)
    result.update({
        "post_ruckig_duration_s": float(final_t[-1]), "post_ruckig_output_waypoint_count": len(final_t),
        "post_ruckig_dynamics": final_dynamic, "post_ruckig_process": final_process, "post_ruckig_collision": final_collision,
        "start_end_configuration_preserved": endpoints_preserved, "ruckig_certification_semantics": "native MoveIt RuckigSmoothing success with frozen jerk-bounded RobotModel; no finite-difference jerk is substituted for the native algorithmic bound",
        "jerk_limit_violation_count": 0 if final_pass else None, "status": "PASSED" if final_pass else "BLOCKED",
    })
    if not final_pass:
        result["first_blocker"] = "post_ruckig_validation_failed"
    final_rows = trajectory_rows(segment, "POST_RUCKIG", final_t, final_q, final_dq, final_ddq)
    validation_rows.extend([{"phase": "POST_RUCKIG", "kind": "process", **row} for row in final_process_rows])
    validation_rows.extend([{"phase": "POST_RUCKIG", "kind": "collision", **row} for row in final_collision_rows])
    return result, totg_rows, final_rows, validation_rows


def formal_run(node: Any, moveit: MoveItPy) -> dict[str, Any]:
    output = Path(str(param(node, "output_dir"))).resolve()
    validation_path = Path(str(param(node, "h6_4_final_validation"))).resolve()
    segments_path = Path(str(param(node, "h6_4_segments"))).resolve()
    contract_path = Path(str(param(node, "process_contract"))).resolve()
    mesh_path = Path(str(param(node, "fixture_mesh"))).resolve()
    totg_path = Path(str(param(node, "totg_parameters"))).resolve()
    ruckig_authorized = bool(param(node, "ruckig_authorized", False))
    output.mkdir(parents=True, exist_ok=True)
    configured_limits = Path(get_package_share_directory("fr5_tunnel_moveit_bridge")) / "config/joint_limits_with_jerk.yaml"
    limit_audit = runtime_limits(moveit, configured_limits)
    dump_json(output / "runtime_limit_audit.json", limit_audit)
    if limit_audit["status"] != "PASSED":
        result = {"status": "BLOCKED", "first_blocker": "runtime_joint_limit_audit_failed", "segment_results": [], "NEW_FJT_GOALS_SENT": 0, "ROBOT_MOTION_STARTED": "NO"}
        dump_json(output / "native_result.json", result)
        return result
    apply_fixture(moveit, mesh_path)
    segments = group_input_rows(load_jsonl(validation_path), load_json(segments_path))
    contract = load_json(contract_path)
    totg = load_json(totg_path)
    results: list[dict[str, Any]] = []
    all_totg_rows: list[dict[str, Any]] = []
    all_final_rows: list[dict[str, Any]] = []
    all_validation_rows: list[dict[str, Any]] = []
    # Mandatory precondition: complete native TOTG and post-TOTG validation
    # for all nine segments before any Ruckig call is authorized.
    for segment in segments:
        result, totg_rows, final_rows, validation_rows = run_segment(moveit, segment, limit_audit["runtime_bounds"], totg, False, contract)
        results.append(result)
        all_totg_rows.extend(totg_rows)
        all_validation_rows.extend(validation_rows)
        if result["status"] != "PASSED":
            break
    totg_precondition_passed = len(results) == 9 and all(row.get("post_totg_status") == "PASSED" for row in results)
    if totg_precondition_passed and ruckig_authorized:
        results = []
        all_totg_rows = []
        all_validation_rows = []
        for segment in segments:
            result, totg_rows, final_rows, validation_rows = run_segment(moveit, segment, limit_audit["runtime_bounds"], totg, True, contract)
            results.append(result)
            all_totg_rows.extend(totg_rows)
            all_final_rows.extend(final_rows)
            all_validation_rows.extend(validation_rows)
            if result["status"] != "PASSED":
                break
    dump_jsonl(output / "totg_trajectories.jsonl", all_totg_rows)
    dump_jsonl(output / "ruckig_trajectories.jsonl", all_final_rows)
    dump_jsonl(output / "validation_rows.jsonl", all_validation_rows)
    passed = len(results) == 9 and all(row["status"] == "PASSED" for row in results)
    summary = {
        "schema_version": "stage3-h7-2-native-result-v1", "status": "PASSED" if passed else "BLOCKED",
        "first_blocker": next((f"segment_{row.get('segment_id')}:{row.get('first_blocker', 'segment_validation_failed')}" for row in results if row["status"] != "PASSED"), None),
        "authoritative_segment_order": SEGMENT_ORDER, "segment_results": results,
        "totg_segments_passed": sum(row.get("post_totg_status") == "PASSED" for row in results),
        "totg_global_precondition_passed_before_ruckig": totg_precondition_passed,
        "ruckig_authorized": ruckig_authorized, "ruckig_segments_passed": sum(row.get("ruckig_status") == "PASSED" for row in results),
        "semantic_hash": semantic_hash({"segments": results, "totg_trajectories": all_totg_rows, "ruckig_trajectories": all_final_rows}),
        "NEW_FJT_GOALS_SENT": 0, "ROBOT_MOTION_STARTED": "NO", "FORMAL_LEDGER_MUTATED": "NO",
    }
    dump_json(output / "native_result.json", summary)
    return summary


def main() -> int:
    rclpy.init()
    node = rclpy.create_node("stage3_h7_2_native")
    output = Path(str(param(node, "output_dir"))).resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        moveit = MoveItPy(node_name="stage3_h7_2_native")
        result = formal_run(node, moveit)
        return 0 if result["status"] == "PASSED" else 2
    except Exception as exc:
        result = {"schema_version": "stage3-h7-2-native-result-v1", "status": "BLOCKED", "first_blocker": f"native_exception:{type(exc).__name__}:{exc}", "segment_results": [], "NEW_FJT_GOALS_SENT": 0, "ROBOT_MOTION_STARTED": "NO", "FORMAL_LEDGER_MUTATED": "NO"}
        dump_json(output / "native_result.json", result)
        return 2
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
