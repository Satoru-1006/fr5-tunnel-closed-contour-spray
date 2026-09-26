"""Native MoveIt2 TOTG -> Ruckig timing and FK audit for Stage 2.5.

This runner consumes only the frozen Stage 2.4T joint path.  It does not run
IK, graph search, planning, segmentation recovery, or controller execution.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage25_contract import (  # noqa: E402
    FORMAL_RUCKIG_PARAMETERS,
    FORMAL_TOTG_PARAMETERS,
    REPAIR_REASON,
    validate_ruckig_parameters,
    validate_totg_parameters,
)

import rclpy  # noqa: E402
import yaml  # noqa: E402
from moveit.core.robot_state import RobotState  # noqa: E402
from moveit.core.robot_trajectory import RobotTrajectory  # noqa: E402
from moveit.planning import MoveItPy  # noqa: E402
from moveit_msgs.msg import RobotTrajectory as RobotTrajectoryMsg  # noqa: E402
from trajectory_msgs.msg import JointTrajectoryPoint  # noqa: E402

from ros2_moveit_bridge.plan_closed_contour_moveit import (  # noqa: E402
    _apply_ruckig_smoothing_without_known_false_error,
    _trajectory_arrays,
    as_robot_trajectory_message,
    moveit_group_limits,
    write_joint_trajectory_csv,
)


JOINT_NAMES = ["j1", "j2", "j3", "j4", "j5", "j6"]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def finite(value: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(value)))


def read_path(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append([float(row[f"q{i}"]) for i in range(1, 7)])
    q = np.asarray(rows, dtype=float)
    if q.ndim != 2 or q.shape[1] != 6 or len(q) < 2:
        raise RuntimeError(f"Stage 2.5 input path is not a 6-DOF trajectory: {path}")
    if not finite(q):
        raise RuntimeError("Stage 2.5 pre-timing path contains non-finite joint values.")
    return q


def build_message(q: np.ndarray) -> RobotTrajectoryMsg:
    message = RobotTrajectoryMsg()
    message.joint_trajectory.joint_names = list(JOINT_NAMES)
    for row in q:
        point = JointTrajectoryPoint()
        point.positions = [float(value) for value in row]
        message.joint_trajectory.points.append(point)
    return message


def duration_seconds(duration) -> float:
    return float(duration.sec) + float(duration.nanosec) * 1e-9


def trajectory_basic_audit(trajectory, limits: tuple[np.ndarray, np.ndarray, np.ndarray], position_bounds: tuple[np.ndarray, np.ndarray], stage: str) -> dict:
    velocity_limits, acceleration_limits, jerk_limits = limits
    position_lower, position_upper = position_bounds
    t, q, dq, ddq, jerk = _trajectory_arrays(trajectory)
    finite_flags = {
        "positions_finite": finite(q),
        "velocities_finite": finite(dq),
        "accelerations_finite": finite(ddq),
        "jerks_finite": finite(jerk),
        "timestamps_present": bool(np.all(t >= 0.0)),
        "timestamps_strictly_increasing": bool(np.all(np.diff(t) > 0.0)),
    }
    position_limits_passed = bool(finite(q) and np.all(q >= position_lower[None, :] - 1.0e-12) and np.all(q <= position_upper[None, :] + 1.0e-12))
    velocity_ratio = np.max(np.abs(dq), axis=0) / np.maximum(velocity_limits, 1e-12)
    acceleration_ratio = np.max(np.abs(ddq), axis=0) / np.maximum(acceleration_limits, 1e-12)
    jerk_ratio = np.max(np.abs(jerk), axis=0) / np.maximum(jerk_limits, 1e-12)
    rows = []
    for index, name in enumerate(JOINT_NAMES):
        rows.append(
            {
                "joint": name,
                "max_abs_velocity": float(np.max(np.abs(dq[:, index]))),
                "velocity_limit": float(velocity_limits[index]),
                "velocity_ratio": float(velocity_ratio[index]),
                "margin_to_velocity_limit": float(velocity_limits[index] - np.max(np.abs(dq[:, index]))),
                "max_abs_acceleration": float(np.max(np.abs(ddq[:, index]))),
                "acceleration_limit": float(acceleration_limits[index]),
                "acceleration_ratio": float(acceleration_ratio[index]),
                "margin_to_acceleration_limit": float(acceleration_limits[index] - np.max(np.abs(ddq[:, index]))),
                "max_abs_jerk": float(np.max(np.abs(jerk[:, index]))),
                "jerk_limit": float(jerk_limits[index]),
                "jerk_ratio": float(jerk_ratio[index]),
                "margin_to_jerk_limit": float(jerk_limits[index] - np.max(np.abs(jerk[:, index]))),
            }
        )
    result = {
        "stage": stage,
        "sample_count": int(len(t)),
        "trajectory_duration_s": float(t[-1]),
        "timestamp_first_s": float(t[0]),
        "timestamp_last_s": float(t[-1]),
        "timestamps_present": finite_flags["timestamps_present"],
        "timestamps_strictly_increasing": finite_flags["timestamps_strictly_increasing"],
        "positions_finite": finite_flags["positions_finite"],
        "position_limits_passed": position_limits_passed,
        "position_lower_rad": position_lower.tolist(),
        "position_upper_rad": position_upper.tolist(),
        "velocities_finite": finite_flags["velocities_finite"],
        "accelerations_finite": finite_flags["accelerations_finite"],
        "jerks_finite": finite_flags["jerks_finite"],
        "max_velocity_ratio": float(np.max(velocity_ratio)),
        "max_acceleration_ratio": float(np.max(acceleration_ratio)),
        "max_jerk_ratio": float(np.max(jerk_ratio)),
        "velocity_limits_passed": bool(finite_flags["velocities_finite"] and np.all(velocity_ratio <= 1.0 + 1e-9)),
        "acceleration_limits_passed": bool(finite_flags["accelerations_finite"] and np.all(acceleration_ratio <= 1.0 + 1e-9)),
        "jerk_limits_passed": bool(finite_flags["jerks_finite"] and np.all(jerk_ratio <= 1.0 + 1e-9)),
        "per_joint": rows,
        "trajectory_start_velocity": dq[0].tolist(),
        "trajectory_start_acceleration": ddq[0].tolist(),
        "trajectory_end_velocity": dq[-1].tolist(),
        "trajectory_end_acceleration": ddq[-1].tolist(),
    }
    result["numeric_integrity_passed"] = bool(all(finite_flags.values()) and np.all(np.diff(t) > 0.0))
    return result


def transform_matrix(value) -> np.ndarray:
    if hasattr(value, "matrix"):
        return np.asarray(value.matrix(), dtype=float)
    return np.asarray(value, dtype=float)


def read_waypoints(path: Path) -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            index = int(row["waypoint_id"])
            rows[index] = {key: float(row[key]) for key in ("x", "y", "z", "qx", "qy", "qz", "qw", "nx", "ny", "nz", "surface_x_m", "surface_y_m", "surface_z_m")}
    if len(rows) != 720 or sorted(rows) != list(range(720)):
        raise RuntimeError(f"Stage 2.5 expected 720 nominal waypoints, found {len(rows)}.")
    return rows


def map_final_samples(final_q: np.ndarray, source_q: np.ndarray, source_ref: dict) -> list[int]:
    if len(final_q) == len(source_q) and np.allclose(final_q, source_q, atol=1e-10, rtol=0.0):
        return list(range(len(source_q)))
    mapped: list[int] = []
    cursor = 0
    for row in final_q:
        global_index = int(np.argmin(np.linalg.norm(source_q - row[None, :], axis=1)))
        end = min(len(source_q), cursor + 500)
        local_distances = np.linalg.norm(source_q[cursor:end] - row[None, :], axis=1)
        local_index = cursor + int(np.argmin(local_distances))
        chosen = global_index if cursor <= global_index <= end - 1 else local_index
        cursor = max(cursor, chosen)
        mapped.append(cursor)
    return mapped


def source_boundary_indices(final_q: np.ndarray, source_q: np.ndarray) -> list[int]:
    """Locate each frozen source state in the dense native output in order."""

    result: list[int] = []
    cursor = 0
    for source_row in source_q:
        distances = np.linalg.norm(final_q[cursor:] - source_row[None, :], axis=1)
        local = int(np.argmin(distances)) + cursor
        result.append(local)
        cursor = local
    return result


def pose_validation(moveit, trajectory, group_name: str, ee_link: str, waypoints_path: Path, source_ref_path: Path, output_dir: Path) -> dict:
    waypoints = read_waypoints(waypoints_path)
    source_ref = json.loads(source_ref_path.read_text(encoding="utf-8"))
    source_q = np.asarray(source_ref["source_q"], dtype=float)
    source_map = source_ref["path_samples"]
    final_t, final_q, _, _, _ = _trajectory_arrays(trajectory)
    mapped = map_final_samples(final_q, source_q, source_ref)
    boundary_indices = source_boundary_indices(final_q, source_q)
    state = RobotState(moveit.get_robot_model())
    base_xyz = np.asarray(source_ref["robot_base_xyz_m"], dtype=float)
    nominal_standoff = float(source_ref["standoff_nominal_m"])
    on_records = []
    off_count = 0
    joint_path_deviations_deg = []
    orientation_errors_deg = []
    trace_path = output_dir / "stage25_fk_trace.csv"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["final_index", "t_s", "source_path_index", "spray_state", "source_waypoint", "tcp_x", "tcp_y", "tcp_z", "position_error_mm", "normal_error_deg", "spray_distance_m", "spray_distance_error_mm"])
        for final_index, (time_value, q) in enumerate(zip(final_t, final_q)):
            # Use the ordered locations of frozen source states in the dense
            # output.  Nearest-q lookup alone can remain on an ON endpoint
            # while Ruckig traverses an OFF transition with a large joint
            # excursion.
            source_index = int(np.searchsorted(boundary_indices, final_index, side="right") - 1)
            source_index = min(max(source_index, 0), len(source_map) - 1)
            metadata = source_map[source_index]
            at_frozen_boundary = final_index == boundary_indices[source_index]
            next_index = min(source_index + 1, len(source_q) - 1)
            next_meta = source_map[next_index]
            source_direction = source_q[next_index] - source_q[source_index]
            source_denominator = float(np.dot(source_direction, source_direction))
            source_alpha = float(np.clip(np.dot(q - source_q[source_index], source_direction) / max(source_denominator, 1e-15), 0.0, 1.0))
            source_projection = (1.0 - source_alpha) * source_q[source_index] + source_alpha * source_q[next_index]
            joint_path_deviations_deg.append(float(np.max(np.abs(q - source_projection)) * 180.0 / math.pi))
            state.set_joint_group_positions(group_name, q)
            state.update()
            transform = transform_matrix(state.get_global_link_transform(ee_link))
            actual = transform[:3, 3]
            tool_z = transform[:3, 2] / max(float(np.linalg.norm(transform[:3, 2])), 1e-12)
            # At an exact shared boundary the source waypoint remains a valid
            # ON sample.  Between source states, the next interval determines
            # the state; this prevents the OFF reorientation after 402 from
            # being treated as ON while its q still lies near q(402).
            interval_is_on = bool(metadata.get("source_waypoints")) and bool(next_meta.get("source_waypoints")) and bool(metadata.get("segment_ids")) and bool(next_meta.get("segment_ids")) and bool(set(metadata.get("segment_ids", [])) & set(next_meta.get("segment_ids", [])))
            is_on_sample = bool(metadata.get("source_waypoints")) if at_frozen_boundary else interval_is_on
            if not is_on_sample:
                off_count += 1
                writer.writerow([final_index, float(time_value), source_index, "OFF", "", *actual.tolist(), "", "", "", ""])
                continue
            metadata_waypoints = metadata.get("source_waypoints") or []
            if not metadata_waypoints and metadata.get("source_waypoint") is not None:
                metadata_waypoints = [metadata["source_waypoint"]]
            if not metadata_waypoints:
                raise RuntimeError(f"ON sample has no source waypoint metadata at final index {final_index}, source index {source_index}.")
            wp0_id = int(metadata_waypoints[0])
            # Ruckig returns dense samples between the frozen path states.
            # Compare each timed sample with the linearly projected frozen
            # joint-space segment, rather than incorrectly comparing it with
            # the preceding endpoint waypoint.
            if next_meta.get("spray_state") == "ON" and next_meta.get("segment_ids") and set(next_meta.get("segment_ids", [])) & set(metadata.get("segment_ids", [])):
                q0 = source_q[source_index]
                q1 = source_q[next_index]
                direction = q1 - q0
                denominator = float(np.dot(direction, direction))
                alpha = float(np.clip(np.dot(q - q0, direction) / max(denominator, 1e-15), 0.0, 1.0))
                next_waypoints = next_meta.get("source_waypoints") or [wp0_id]
                next_wp = int(next_waypoints[0])
            else:
                alpha = 0.0
                next_wp = wp0_id
            wp1_id = next_wp if next_meta.get("source_waypoints") else wp0_id
            wp0 = waypoints[wp0_id]
            wp1 = waypoints[wp1_id]
            target = (1.0 - alpha) * np.asarray([wp0["x"], wp0["y"], wp0["z"]]) + alpha * np.asarray([wp1["x"], wp1["y"], wp1["z"]])
            normal = (1.0 - alpha) * np.asarray([wp0["nx"], wp0["ny"], wp0["nz"]]) + alpha * np.asarray([wp1["nx"], wp1["ny"], wp1["nz"]])
            normal /= max(float(np.linalg.norm(normal)), 1e-12)
            surface_base = (1.0 - alpha) * (np.asarray([wp0["surface_x_m"], wp0["surface_y_m"], wp0["surface_z_m"]]) - base_xyz) + alpha * (np.asarray([wp1["surface_x_m"], wp1["surface_y_m"], wp1["surface_z_m"]]) - base_xyz)
            position_error = float(np.linalg.norm(actual - target))
            # The frozen spray TCP points toward the surface: source q0 has
            # tool-Z = -surface normal.  Preserve that formal sign convention.
            normal_error = float(math.degrees(math.acos(float(np.clip(np.dot(-tool_z, normal), -1.0, 1.0)))))
            spray_distance = float(np.dot(actual - surface_base, normal))
            spray_error = spray_distance - nominal_standoff
            target_quaternion = np.asarray([wp0["qx"], wp0["qy"], wp0["qz"], wp0["qw"]], dtype=float)
            qx, qy, qz, qw = target_quaternion / max(float(np.linalg.norm(target_quaternion)), 1e-15)
            target_rotation = np.asarray(
                [
                    [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
                    [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
                    [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
                ],
                dtype=float,
            )
            orientation_error = float(math.degrees(math.acos(float(np.clip((np.trace(target_rotation.T @ transform[:3, :3]) - 1.0) * 0.5, -1.0, 1.0)))))
            orientation_errors_deg.append(orientation_error)
            on_records.append({"final_index": final_index, "time_s": float(time_value), "source_path_index": source_index, "source_waypoint": wp0_id, "position_error_mm": position_error * 1000.0, "normal_error_deg": normal_error, "orientation_error_deg": orientation_error, "spray_distance_m": spray_distance, "spray_distance_error_mm": spray_error * 1000.0})
            writer.writerow([final_index, float(time_value), source_index, "ON", wp0_id, *actual.tolist(), position_error * 1000.0, normal_error, spray_distance, spray_error * 1000.0])
    position_errors = np.asarray([row["position_error_mm"] for row in on_records], dtype=float)
    normal_errors = np.asarray([row["normal_error_deg"] for row in on_records], dtype=float)
    spray_errors = np.asarray([row["spray_distance_error_mm"] for row in on_records], dtype=float)
    sorted_position_errors = sorted((float(row["position_error_mm"]) for row in on_records), reverse=True)
    max_position_record = max(on_records, key=lambda row: float(row["position_error_mm"])) if on_records else None
    samples_over_6mm = int(np.sum(position_errors > 6.0)) if len(position_errors) else 0
    pose_constraints_passed = bool(len(on_records) and len({row["source_waypoint"] for row in on_records}) == 720 and np.all(position_errors <= 6.0) and np.all(normal_errors <= 10.0) and np.all(np.abs(spray_errors) <= 5.0))
    result = {
        "status": "pass" if pose_constraints_passed else ("blocked_no_spray_on_samples" if not on_records else "fail:post_timing_pose_constraints"),
        "source_waypoints_checked": len({row["source_waypoint"] for row in on_records}),
        "timed_spray_on_samples_checked": len(on_records),
        "timed_spray_off_samples_not_pose_gated": off_count,
        "spray_on_waypoint_coverage": f"{len({row['source_waypoint'] for row in on_records})}/720",
        "max_tcp_position_deviation_mm": float(np.max(position_errors)) if len(position_errors) else None,
        "max_tcp_position_deviation_waypoint": int(max_position_record["source_waypoint"]) if max_position_record else None,
        "max_tcp_position_deviation_time_s": float(max_position_record["time_s"]) if max_position_record else None,
        "second_largest_tcp_position_deviation_mm": sorted_position_errors[1] if len(sorted_position_errors) > 1 else None,
        "samples_over_6mm": samples_over_6mm,
        "max_joint_path_deviation_deg": float(np.max(joint_path_deviations_deg)) if joint_path_deviations_deg else None,
        "max_joint_path_deviation_rad": float(np.deg2rad(np.max(joint_path_deviations_deg))) if joint_path_deviations_deg else None,
        "max_tcp_orientation_deviation_deg": float(np.max(orientation_errors_deg)) if orientation_errors_deg else None,
        "max_spray_normal_error_deg": float(np.max(normal_errors)) if len(normal_errors) else None,
        "max_spray_distance_error_mm": float(np.max(np.abs(spray_errors))) if len(spray_errors) else None,
        "position_constraint_limit_mm": 6.0,
        "normal_constraint_limit_deg": 10.0,
        "spray_distance_tolerance_mm": 5.0,
        "pose_constraints_passed": pose_constraints_passed,
        "orientation_deviation_contract": "diagnostic_full_quaternion_error; formal spray orientation gate is tool-Z normal error with roll freedom",
        "spray_axis_convention": "tool_Z_points_to_surface; compare -tool_Z with frozen surface normal",
        "fk_trace": str(trace_path.resolve()),
        "new_timed_samples_included": True,
        "mapping": "monotonic_nearest_frozen_stage24t_path_sample",
    }
    failing_records = [row for row in on_records if float(row["position_error_mm"]) > 6.0 or float(row["normal_error_deg"]) > 10.0 or abs(float(row["spray_distance_error_mm"])) > 5.0]
    if failing_records:
        first_failure = failing_records[0]
        result["first_failing_segment"] = next((int(x) for x in source_map[first_failure["source_path_index"]].get("segment_ids", [])), None)
        result["first_failing_transition"] = next((int(x) for x in source_map[first_failure["source_path_index"]].get("transition_ids", [])), None)
        result["first_failing_timestamp_s"] = first_failure["time_s"]
        result["first_failing_source_waypoint"] = first_failure["source_waypoint"]
        result["first_failing_measured_position_error_mm"] = first_failure["position_error_mm"]
        result["first_failing_formal_position_limit_mm"] = 6.0
    write_json(output_dir / "stage25_pose_validation.json", result)
    return result


def main() -> int:
    rclpy.init()
    node = rclpy.create_node("stage25_moveit_timing_runner")
    try:
        input_csv = Path(str(node.declare_parameter("input_csv", "").value))
        source_reference_json = Path(str(node.declare_parameter("source_reference_json", "").value))
        waypoints_csv = Path(str(node.declare_parameter("waypoints_csv", "").value))
        output_dir = Path(str(node.declare_parameter("output_dir", "").value))
        group_name = str(node.declare_parameter("group_name", "fairino5_v6_group").value)
        ee_link = str(node.declare_parameter("ee_link", "spray_tcp_link").value)
        velocity_scaling = float(node.declare_parameter("velocity_scaling", FORMAL_TOTG_PARAMETERS["velocity_scaling_factor"]).value)
        acceleration_scaling = float(node.declare_parameter("acceleration_scaling", FORMAL_TOTG_PARAMETERS["acceleration_scaling_factor"]).value)
        path_tolerance = float(node.declare_parameter("path_tolerance", FORMAL_TOTG_PARAMETERS["path_tolerance"]).value)
        resample_dt = float(node.declare_parameter("resample_dt", FORMAL_TOTG_PARAMETERS["resample_dt"]).value)
        min_angle_change = float(node.declare_parameter("min_angle_change", FORMAL_TOTG_PARAMETERS["min_angle_change"]).value)
        raw_historical_replay = node.declare_parameter("stage25r_historical_replay", False).value
        historical_replay = raw_historical_replay if isinstance(raw_historical_replay, bool) else str(raw_historical_replay).strip().lower() in {"true", "1", "yes"}
        raw_stage25r2_shadow = node.declare_parameter("stage25r2_shadow_sweep", False).value
        stage25r2_shadow = raw_stage25r2_shadow if isinstance(raw_stage25r2_shadow, bool) else str(raw_stage25r2_shadow).strip().lower() in {"true", "1", "yes"}
        if not historical_replay:
            validate_totg_parameters({"velocity_scaling_factor": velocity_scaling, "acceleration_scaling_factor": acceleration_scaling, "path_tolerance": path_tolerance, "resample_dt": resample_dt, "min_angle_change": min_angle_change})
        raw_mitigate_overshoot = node.declare_parameter("ruckig_mitigate_overshoot", FORMAL_RUCKIG_PARAMETERS["mitigate_overshoot"]).value
        mitigate_overshoot = raw_mitigate_overshoot if isinstance(raw_mitigate_overshoot, bool) else str(raw_mitigate_overshoot).strip().lower() in {"true", "1", "yes"}
        overshoot_threshold = float(node.declare_parameter("ruckig_overshoot_threshold", FORMAL_RUCKIG_PARAMETERS["overshoot_threshold"]).value)
        raw_zero_target_acceleration = node.declare_parameter("stage25r_zero_target_acceleration", True).value
        zero_target_acceleration = raw_zero_target_acceleration if isinstance(raw_zero_target_acceleration, bool) else str(raw_zero_target_acceleration).strip().lower() in {"true", "1", "yes"}
        if historical_replay:
            expected_historical_totg = {"velocity_scaling_factor": 0.15, "acceleration_scaling_factor": 0.15, "path_tolerance": 0.01, "resample_dt": 0.01, "min_angle_change": 0.0005}
            expected_historical_ruckig = {"velocity_scaling_factor": 0.15, "acceleration_scaling_factor": 0.15, "mitigate_overshoot": True, "overshoot_threshold": 0.005}
            if any(abs(float({"velocity_scaling_factor": velocity_scaling, "acceleration_scaling_factor": acceleration_scaling, "path_tolerance": path_tolerance, "resample_dt": resample_dt, "min_angle_change": min_angle_change}[key]) - value) > 1e-12 for key, value in expected_historical_totg.items()):
                raise ValueError("Stage 2.5R historical replay TOTG parameter mismatch")
            if mitigate_overshoot is not True or abs(overshoot_threshold - expected_historical_ruckig["overshoot_threshold"]) > 1e-12:
                raise ValueError("Stage 2.5R historical replay Ruckig parameter mismatch")
        elif not stage25r2_shadow:
            validate_ruckig_parameters({"velocity_scaling_factor": velocity_scaling, "acceleration_scaling_factor": acceleration_scaling, "mitigate_overshoot": mitigate_overshoot, "overshoot_threshold": overshoot_threshold})
        elif not math.isfinite(overshoot_threshold) or overshoot_threshold < 0.0:
            raise ValueError("Stage 2.5R2 shadow overshoot threshold must be finite and non-negative")
        node.get_logger().info(f"Stage 2.5 inputs: input_csv={input_csv} exists={input_csv.exists()}, source_reference_json={source_reference_json} exists={source_reference_json.exists()}, waypoints_csv={waypoints_csv} exists={waypoints_csv.exists()}, output_dir={output_dir}")
        if not input_csv.exists() or not source_reference_json.exists() or not waypoints_csv.exists():
            raise RuntimeError("Stage 2.5 MoveIt runner input file is missing.")
        output_dir.mkdir(parents=True, exist_ok=True)
        source_q = read_path(input_csv)
        source_ref = json.loads(source_reference_json.read_text(encoding="utf-8"))
        if len(source_q) != len(source_ref["source_q"]):
            raise RuntimeError("Stage 2.5 source path/reference mapping length mismatch.")

        moveit = MoveItPy(node_name="stage25_moveit_timing")
        model = moveit.get_robot_model()
        group = model.get_joint_model_group(group_name)
        if group is None or list(group.active_joint_model_names) != JOINT_NAMES:
            raise RuntimeError(f"Unexpected MoveIt active joint group: {list(group.active_joint_model_names) if group else None}")
        if not hasattr(RobotTrajectory, "apply_ruckig_smoothing"):
            raise RuntimeError("MoveItPy RobotTrajectory.apply_ruckig_smoothing is unavailable.")
        joint_names, velocity_limits, acceleration_limits, jerk_limits = moveit_group_limits(moveit, group_name)
        if joint_names != JOINT_NAMES:
            raise RuntimeError(f"MoveIt joint order mismatch: {joint_names}")
        position_lower = []
        position_upper = []
        for bounds in group.active_joint_model_bounds:
            if isinstance(bounds, (list, tuple)):
                bounds = bounds[0]
            position_lower.append(float(bounds.min_position))
            position_upper.append(float(bounds.max_position))
        position_bounds = (np.asarray(position_lower), np.asarray(position_upper))
        limits = (velocity_limits, acceleration_limits, jerk_limits)

        start_state = RobotState(model)
        start_state.set_joint_group_positions(group_name, source_q[0])
        start_state.update()
        raw_message = build_message(source_q)
        trajectory = RobotTrajectory(model)
        trajectory.joint_model_group_name = group_name
        trajectory.set_robot_trajectory_msg(start_state, raw_message)
        trajectory.unwind()
        totg_ok = bool(trajectory.apply_totg_time_parameterization(velocity_scaling, acceleration_scaling, path_tolerance=path_tolerance, resample_dt=resample_dt, min_angle_change=min_angle_change))
        if not totg_ok:
            raise RuntimeError("MoveIt2 native TOTG returned false.")
        totg_audit = trajectory_basic_audit(trajectory, limits, position_bounds, "TOTG")
        write_joint_trajectory_csv(trajectory, output_dir / "stage25_totg_trajectory.csv")
        write_json(output_dir / "stage25_totg_audit.json", {"totg_success": totg_ok, "parameters": FORMAL_TOTG_PARAMETERS, "formal_totg_parameters": FORMAL_TOTG_PARAMETERS, "repair_reason": REPAIR_REASON, "audit": totg_audit})

        # MoveIt2 Jazzy's native Ruckig adapter consumes the acceleration state
        # stored at every input waypoint.  TOTG may leave different endpoint
        # accelerations on adjacent path pieces; feeding those directly to
        # Ruckig creates an acceleration discontinuity at the waypoint and a
        # finite-difference jerk spike.  Keep the frozen q path and the TOTG
        # velocity states, but use the continuous-process C2 contract of zero
        # target acceleration at each waypoint.  This is a timing-state
        # preparation only: no joint position, waypoint, branch, or segment is
        # changed, and internal velocities are not force-zeroed.
        ruckig_input_message = as_robot_trajectory_message(trajectory)
        if zero_target_acceleration:
            for point in ruckig_input_message.joint_trajectory.points:
                point.accelerations = [0.0] * len(JOINT_NAMES)
            prepared_trajectory = RobotTrajectory(model)
            prepared_trajectory.joint_model_group_name = group_name
            prepared_trajectory.set_robot_trajectory_msg(start_state, ruckig_input_message)
            prepared_trajectory.unwind()
            trajectory = prepared_trajectory
            write_joint_trajectory_csv(trajectory, output_dir / "stage25_ruckig_input_prepared.csv")
            write_json(
                output_dir / "stage25_ruckig_input_preparation.json",
                {
                    "status": "passed",
                    "policy": "zero_target_acceleration_at_waypoints_for_continuous_C2_Ruckig_input",
                    "positions_changed": False,
                    "velocities_changed": False,
                    "accelerations_changed": True,
                    "internal_velocity_zeroing": False,
                    "reason": "prevent TOTG waypoint acceleration discontinuities from becoming jerk spikes",
                    "waypoint_count": len(ruckig_input_message.joint_trajectory.points),
                },
            )
        else:
            write_json(
                output_dir / "stage25_ruckig_input_preparation.json",
                {
                    "status": "not_applied",
                    "policy": "preserve_native_TOTG_waypoint_accelerations",
                    "positions_changed": False,
                    "velocities_changed": False,
                    "accelerations_changed": False,
                    "internal_velocity_zeroing": False,
                    "reason": "Stage 2.5R historical replay of the frozen formal MoveIt execution model",
                    "waypoint_count": len(ruckig_input_message.joint_trajectory.points),
                },
            )
        # Keep the project contract's overshoot mitigation enabled while
        # using the smallest positive native Ruckig threshold that still
        # exercises the adapter's documented parameter path.  This is a
        # timing-only parameter; the frozen Stage 2.4T joint path is untouched.
        ruckig_ok = _apply_ruckig_smoothing_without_known_false_error(trajectory, velocity_scaling, acceleration_scaling, mitigate_overshoot=mitigate_overshoot, overshoot_threshold=overshoot_threshold)
        if not ruckig_ok:
            raise RuntimeError("MoveIt2 native Ruckig smoothing returned false.")
        ruckig_audit = trajectory_basic_audit(trajectory, limits, position_bounds, "Ruckig")
        write_joint_trajectory_csv(trajectory, output_dir / "stage25_ruckig_trajectory.csv")
        effective_ruckig_parameters = {"velocity_scaling_factor": velocity_scaling, "acceleration_scaling_factor": acceleration_scaling, "mitigate_overshoot": mitigate_overshoot, "overshoot_threshold": overshoot_threshold}
        write_json(output_dir / "stage25_ruckig_audit.json", {"ruckig_success": bool(ruckig_ok), "parameters": effective_ruckig_parameters, "formal_ruckig_parameters": FORMAL_RUCKIG_PARAMETERS, "repair_reason": REPAIR_REASON, "audit": ruckig_audit})

        pose = pose_validation(moveit, trajectory, group_name, ee_link, waypoints_csv, source_reference_json, output_dir)
        runtime = {
            "status": "passed_native_moveit_timing" if totg_ok and ruckig_ok else "blocked",
            "ros_distribution": "jazzy",
            "moveit_runtime": "MoveItPy RobotTrajectory",
            "moveit_api": {"totg": "RobotTrajectory.apply_totg_time_parameterization", "ruckig": "RobotTrajectory.apply_ruckig_smoothing"},
            "totg_executed": totg_ok,
            "ruckig_executed": bool(ruckig_ok),
            "group_name": group_name,
            "ee_link": ee_link,
            "joint_names": joint_names,
            "effective_runtime_limits": {"velocity_rad_s": velocity_limits.tolist(), "acceleration_rad_s2": acceleration_limits.tolist(), "jerk_rad_s3": jerk_limits.tolist(), "position_source": "URDF/MoveIt RobotModel bounds"},
            "continuous_process_strategy": "single_continuous_timed_trajectory",
            "internal_boundary_zeroing": False,
            "controller_execution": "not_requested_offline_certification",
            "start_state_initialized_from_frozen_stage24t_q0": True,
            "totg_audit": totg_audit,
            "ruckig_audit": ruckig_audit,
            "ruckig_input_preparation": "stage25_ruckig_input_preparation.json",
            "formal_totg_parameters": FORMAL_TOTG_PARAMETERS,
            "formal_ruckig_parameters": FORMAL_RUCKIG_PARAMETERS,
            "repair_reason": REPAIR_REASON,
            "pose_validation": pose,
        }
        write_json(output_dir / "stage25_native_moveit_runtime.json", runtime)
        node.get_logger().info(f"Stage 2.5 native timing complete: TOTG={totg_ok}, Ruckig={ruckig_ok}, samples={ruckig_audit['sample_count']}")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    except Exception as exc:
        output_dir = locals().get("output_dir", Path("."))
        write_json(Path(output_dir) / "stage25_native_moveit_runtime.json", {"status": "blocked_native_moveit_runtime", "error": repr(exc)})
        print(f"stage25_moveit_runner error: {exc}", file=sys.stderr)
        sys.stderr.flush()
        return 10
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
