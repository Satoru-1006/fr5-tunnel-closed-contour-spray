"""Shadow-only native MoveIt TOTG -> Ruckig retiming for a Stage5AC path."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def read_path(path: Path) -> np.ndarray:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    q = np.asarray([[float(row[f"j{i}_q"]) for i in range(1, 7)] for row in rows], dtype=float)
    if q.shape[0] < 2 or q.shape[1:] != (6,) or not np.isfinite(q).all():
        raise RuntimeError("stage5ac_native_retime_invalid_path")
    return q


def main() -> int:  # pragma: no cover - ROS 2 runtime
    import rclpy
    from moveit.core.robot_state import RobotState
    from moveit.core.robot_trajectory import RobotTrajectory
    from moveit.planning import MoveItPy
    from moveit_msgs.msg import RobotTrajectory as RobotTrajectoryMsg
    from rclpy.node import Node
    from trajectory_msgs.msg import JointTrajectoryPoint

    from ros2_moveit_bridge.plan_closed_contour_moveit import (
        _apply_ruckig_smoothing_without_known_false_error,
        _trajectory_arrays,
        as_robot_trajectory_message,
        moveit_group_limits,
        validate_joint_dynamics,
        write_joint_trajectory_csv,
    )
    from stage25_moveit_runner import trajectory_basic_audit

    rclpy.init()
    node = Node("stage5ac_native_retime")
    for name, default in (("source_path", ""), ("output_dir", ""), ("velocity_scaling", 0.25), ("acceleration_scaling", 0.25), ("path_tolerance", 1.0e-6), ("resample_dt", 0.01), ("min_angle_change", 0.0005), ("manual_time_step_s", 0.0), ("consistent_fd_time_dilation", False), ("fd_time_scale_multiplier", 1.0), ("zero_target_acceleration", True), ("mitigate_overshoot", True), ("overshoot_threshold", 0.01)):
        node.declare_parameter(name, default)
    source_path = Path(str(node.get_parameter("source_path").value)).resolve()
    output = Path(str(node.get_parameter("output_dir").value)).resolve()
    velocity_scaling = float(node.get_parameter("velocity_scaling").value)
    acceleration_scaling = float(node.get_parameter("acceleration_scaling").value)
    path_tolerance = float(node.get_parameter("path_tolerance").value)
    resample_dt = float(node.get_parameter("resample_dt").value)
    min_angle_change = float(node.get_parameter("min_angle_change").value)
    manual_time_step_s = float(node.get_parameter("manual_time_step_s").value)
    consistent_fd_time_dilation = bool(node.get_parameter("consistent_fd_time_dilation").value)
    fd_time_scale_multiplier = float(node.get_parameter("fd_time_scale_multiplier").value)
    zero_target_acceleration = bool(node.get_parameter("zero_target_acceleration").value)
    mitigate_overshoot = bool(node.get_parameter("mitigate_overshoot").value)
    overshoot_threshold = float(node.get_parameter("overshoot_threshold").value)
    q = read_path(source_path)
    names = [f"j{i}" for i in range(1, 7)]
    moveit = MoveItPy(node_name="stage5ac_native_retime_moveit")
    model = moveit.get_robot_model()
    group_name = "fairino5_v6_group"
    group = model.get_joint_model_group(group_name)
    if group is None or list(group.active_joint_model_names) != names:
        raise RuntimeError("stage5ac_native_retime_joint_mapping_mismatch")
    joint_names, velocity_limits, acceleration_limits, jerk_limits = moveit_group_limits(moveit, group_name)
    if not hasattr(RobotTrajectory, "apply_ruckig_smoothing"):
        raise RuntimeError("stage5ac_native_retime_ruckig_unavailable")
    start = RobotState(model)
    start.set_joint_group_positions(group_name, q[0].tolist())
    start.update()
    message = RobotTrajectoryMsg()
    message.joint_trajectory.joint_names = names
    for row in q:
        point = JointTrajectoryPoint()
        point.positions = row.tolist()
        message.joint_trajectory.points.append(point)
    trajectory = RobotTrajectory(model)
    trajectory.joint_model_group_name = group_name
    trajectory.set_robot_trajectory_msg(start, message)
    trajectory.unwind()
    input_point_count = len(as_robot_trajectory_message(trajectory).joint_trajectory.points)
    if consistent_fd_time_dilation:
        if manual_time_step_s > 0.0:
            raise RuntimeError("stage5ac_native_retime_time_modes_are_mutually_exclusive")
        if fd_time_scale_multiplier <= 0.0:
            raise RuntimeError("stage5ac_native_retime_fd_time_scale_multiplier_must_be_positive")
        # Shadow-only, state-consistent timing repair. Derivatives are derived
        # from the exact selected q path, then one uniform dilation is applied
        # to q(t), dq(t), ddq(t), and the implied jerk. This avoids the false
        # zero-derivative audit of the exploratory manual mode while preserving
        # every selected waypoint exactly.
        from builtin_interfaces.msg import Duration
        base_time = np.arange(len(q), dtype=float)
        dq_base = np.gradient(q, base_time, axis=0, edge_order=1)
        ddq_base = np.gradient(dq_base, base_time, axis=0, edge_order=1)
        jerk_base = np.gradient(ddq_base, base_time, axis=0, edge_order=1)
        velocity_ratio = float(np.max(np.abs(dq_base) / np.maximum(velocity_limits * velocity_scaling, 1e-12)))
        acceleration_ratio = float(np.max(np.abs(ddq_base) / np.maximum(acceleration_limits * acceleration_scaling, 1e-12)))
        jerk_ratio = float(np.max(np.abs(jerk_base) / np.maximum(jerk_limits, 1e-12)))
        dilation = max(1.0, 1.05 * velocity_ratio, 1.05 * np.sqrt(max(0.0, acceleration_ratio)), 1.05 * max(0.0, jerk_ratio) ** (1.0 / 3.0)) * fd_time_scale_multiplier
        timed = as_robot_trajectory_message(trajectory)
        for index, point in enumerate(timed.joint_trajectory.points):
            seconds = float(index * dilation)
            whole = int(seconds)
            point.time_from_start = Duration(sec=whole, nanosec=int(round((seconds - whole) * 1e9)))
            point.velocities = (dq_base[index] / dilation).tolist()
            point.accelerations = (ddq_base[index] / (dilation * dilation)).tolist()
        consistent = RobotTrajectory(model)
        consistent.joint_model_group_name = group_name
        consistent.set_robot_trajectory_msg(start, timed)
        consistent.unwind()
        trajectory = consistent
        totg_ok = True
        time_parameterization = "consistent_q_fd_uniform_time_dilation"
        fd_timing = {"base_time_step_s": 1.0, "uniform_dilation": float(dilation), "time_scale_multiplier": fd_time_scale_multiplier, "source_max_velocity_ratio": velocity_ratio, "source_max_acceleration_ratio": acceleration_ratio, "source_max_jerk_ratio": jerk_ratio}
    elif manual_time_step_s > 0.0:
        from builtin_interfaces.msg import Duration
        timed = as_robot_trajectory_message(trajectory)
        for index, point in enumerate(timed.joint_trajectory.points):
            point.time_from_start = Duration(sec=int(index * manual_time_step_s), nanosec=int(round((index * manual_time_step_s - int(index * manual_time_step_s)) * 1e9)))
            point.velocities = [0.0] * 6
            point.accelerations = [0.0] * 6
        manual = RobotTrajectory(model)
        manual.joint_model_group_name = group_name
        manual.set_robot_trajectory_msg(start, timed)
        manual.unwind()
        trajectory = manual
        totg_ok = True
        time_parameterization = "manual_uniform_zero_boundary"
        fd_timing = {"base_time_step_s": None, "uniform_dilation": None, "time_scale_multiplier": None, "source_max_velocity_ratio": None, "source_max_acceleration_ratio": None, "source_max_jerk_ratio": None}
    else:
        totg_ok = bool(trajectory.apply_totg_time_parameterization(velocity_scaling, acceleration_scaling, path_tolerance=path_tolerance, resample_dt=resample_dt, min_angle_change=min_angle_change))
        time_parameterization = "native_totg"
        fd_timing = {"base_time_step_s": None, "uniform_dilation": None, "time_scale_multiplier": None, "source_max_velocity_ratio": None, "source_max_acceleration_ratio": None, "source_max_jerk_ratio": None}
    if not totg_ok:
        raise RuntimeError("stage5ac_native_retime_totg_failed")
    output.mkdir(parents=True, exist_ok=True)
    totg_point_count = len(as_robot_trajectory_message(trajectory).joint_trajectory.points)
    if totg_point_count < 2:
        write_json(output / "STAGE5AC_NATIVE_RETIME.json", {"schema_version": "stage5ac-native-retime-v1", "source_path": str(source_path), "input_point_count": input_point_count, "totg_point_count": totg_point_count, "totg_returned": totg_ok, "error": "native_totg_returned_true_without_output_points", "promotion": "NO_PROMOTION"})
        raise RuntimeError("native_totg_returned_true_without_output_points")
    write_joint_trajectory_csv(trajectory, output / "stage5ac_totg_trajectory.csv")
    prepared = as_robot_trajectory_message(trajectory)
    if zero_target_acceleration and not consistent_fd_time_dilation:
        for point in prepared.joint_trajectory.points:
            point.accelerations = [0.0] * 6
        adjusted = RobotTrajectory(model)
        adjusted.joint_model_group_name = group_name
        adjusted.set_robot_trajectory_msg(start, prepared)
        adjusted.unwind()
        trajectory = adjusted
    ruckig_ok = bool(_apply_ruckig_smoothing_without_known_false_error(trajectory, velocity_scaling, acceleration_scaling, mitigate_overshoot=mitigate_overshoot, overshoot_threshold=overshoot_threshold))
    write_joint_trajectory_csv(trajectory, output / "STAGE5AC_CANDIDATE_TRAJECTORY.csv")
    position_lower = np.asarray([float((b[0] if isinstance(b, (list, tuple)) else b).min_position) for b in group.active_joint_model_bounds])
    position_upper = np.asarray([float((b[0] if isinstance(b, (list, tuple)) else b).max_position) for b in group.active_joint_model_bounds])
    limits = (velocity_limits, acceleration_limits, jerk_limits)
    arrays = _trajectory_arrays(trajectory)
    audit = trajectory_basic_audit(trajectory, limits, (position_lower, position_upper), "Stage5AC_Ruckig")
    try:
        dynamics = validate_joint_dynamics(moveit, trajectory, group_name, output / "STAGE5AC_DYNAMICS_AUDIT.csv", limit_margin=1.0)
    except RuntimeError as exc:
        dynamics = {"status": "fail", "error": str(exc)}
    write_json(output / "STAGE5AC_NATIVE_RETIME.json", {
        "schema_version": "stage5ac-native-retime-v1",
        "source_path": str(source_path),
        "source_waypoint_count": int(len(q)),
        "output_state_count": int(len(arrays[0])),
        "totg_executed": True,
        "totg_returned": totg_ok,
        "time_parameterization": time_parameterization,
        "manual_time_step_s": manual_time_step_s,
        "consistent_fd_time_dilation": consistent_fd_time_dilation,
        "fd_timing": fd_timing,
        "ruckig_executed": True,
        "ruckig_returned": ruckig_ok,
        "zero_target_acceleration": zero_target_acceleration,
        "velocity_scaling": velocity_scaling,
        "acceleration_scaling": acceleration_scaling,
        "mitigate_overshoot": mitigate_overshoot,
        "overshoot_threshold": overshoot_threshold,
        "totg_parameters": {"path_tolerance": path_tolerance, "resample_dt": resample_dt, "min_angle_change": min_angle_change},
        "audit": audit,
        "dynamics_audit": dynamics,
        "candidate_csv": str((output / "STAGE5AC_CANDIDATE_TRAJECTORY.csv").resolve()),
        "promotion": "NO_PROMOTION",
    })
    result_code = 0 if ruckig_ok and dynamics.get("status") == "pass" else 2
    print(json.dumps({"output": str(output), "totg": totg_ok, "ruckig": ruckig_ok, "dynamics": dynamics.get("status"), "states": int(len(arrays[0])), "duration_s": float(arrays[0][-1])}, ensure_ascii=False), flush=True)
    node.destroy_node()
    rclpy.shutdown()
    return result_code


if __name__ == "__main__":
    raise SystemExit(main())
