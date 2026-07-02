#!/usr/bin/env python3
"""MoveIt2 execution-chain entry for closed horseshoe contour tracking.

Run this inside a ROS2 workspace that contains FAIRINO's frcobot_ros2 packages.
It reads the generated TCP path CSV and asks MoveIt2 for a constrained Cartesian
plan. This file is intentionally separate from the Windows/Python visual demo.
"""

from __future__ import annotations

import csv
import os
import sys
import tempfile
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import RobotTrajectory as RobotTrajectoryMsg
from moveit_msgs.msg import CollisionObject
from moveit.core.robot_state import RobotState
from moveit.core.robot_trajectory import RobotTrajectory
from moveit.planning import MoveItPy
from shape_msgs.msg import SolidPrimitive
from trajectory_msgs.msg import JointTrajectoryPoint


def _configured_jerk_limits() -> dict[str, float]:
    bridge_share = Path(get_package_share_directory("fr5_tunnel_moveit_bridge"))
    limits_path = bridge_share / "config" / "joint_limits_with_jerk.yaml"
    data = yaml.safe_load(limits_path.read_text(encoding="utf-8"))
    return {
        joint_name: float(joint_config["max_jerk"])
        for joint_name, joint_config in data.get("joint_limits", {}).items()
        if joint_config.get("has_jerk_limits", False)
    }


def load_tcp_poses(csv_path: Path) -> list[Pose]:
    poses: list[Pose] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_index, row in enumerate(reader, start=2):
            pose = Pose()
            pose.position.x = float(row["x"])
            pose.position.y = float(row["y"])
            pose.position.z = float(row["z"])
            pose.orientation.x = float(row["qx"])
            pose.orientation.y = float(row["qy"])
            pose.orientation.z = float(row["qz"])
            pose.orientation.w = float(row["qw"])
            quaternion = np.array(
                [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],
                dtype=float,
            )
            quaternion_norm = float(np.linalg.norm(quaternion))
            if quaternion_norm < 1e-9:
                raise ValueError(f"Zero quaternion at {csv_path}:{row_index}.")
            quaternion /= quaternion_norm
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = quaternion.tolist()
            if {"nx", "ny", "nz"}.issubset(row):
                x, y, z, w = quaternion
                tool_z = np.array([2.0 * (x * z + y * w), 2.0 * (y * z - x * w), 1.0 - 2.0 * (x * x + y * y)])
                normal = np.array([float(row["nx"]), float(row["ny"]), float(row["nz"])])
                normal /= max(float(np.linalg.norm(normal)), 1e-12)
                angle = np.rad2deg(np.arccos(np.clip(np.dot(tool_z, normal), -1.0, 1.0)))
                if angle > 0.1:
                    raise ValueError(
                        f"TCP quaternion is not aligned with the wall normal at {csv_path}:{row_index}: "
                        f"error={angle:.4f} deg."
                    )
            poses.append(pose)
    if not poses:
        raise ValueError(f"No TCP poses were loaded from {csv_path}.")
    return poses


def load_tcp_normals(csv_path: Path) -> np.ndarray:
    normals: list[list[float]] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            normal = np.array([float(row["nx"]), float(row["ny"]), float(row["nz"])], dtype=float)
            normal /= max(float(np.linalg.norm(normal)), 1e-12)
            normals.append(normal.tolist())
    if not normals:
        raise ValueError(f"No TCP normals were loaded from {csv_path}.")
    return np.asarray(normals, dtype=float)


def load_joint_seed_csv(csv_path: Path, joint_names: list[str]) -> np.ndarray:
    seeds: list[list[float]] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        column_map = {
            joint_name: joint_name if joint_name in fieldnames else f"q{joint_name[1:]}"
            for joint_name in joint_names
        }
        missing = {column for column in column_map.values() if column not in fieldnames}
        if missing:
            raise ValueError(f"{csv_path} is missing seed joint columns: {sorted(missing)}")
        for row in reader:
            seeds.append([float(row[column_map[name]]) for name in joint_names])
    if not seeds:
        raise ValueError(f"No joint seeds were loaded from {csv_path}.")
    return np.asarray(seeds, dtype=float)


def to_pose_stamped(pose: Pose, frame_id: str) -> PoseStamped:
    msg = PoseStamped()
    msg.header.frame_id = frame_id
    msg.pose = pose
    return msg


def as_robot_trajectory_message(trajectory):
    """Normalize MoveItPy's bound trajectory or a ROS message."""

    if hasattr(trajectory, "get_robot_trajectory_msg"):
        return trajectory.get_robot_trajectory_msg()
    return trajectory


def concatenate_robot_trajectories(trajectories):
    """Join MoveIt results before the single global Ruckig pass."""

    if not trajectories:
        raise ValueError("No MoveIt trajectories were planned.")
    result = deepcopy(as_robot_trajectory_message(trajectories[0]))
    output = result.joint_trajectory
    output.points = []
    names = list(output.joint_names)
    for segment_index, trajectory in enumerate(trajectories):
        segment = as_robot_trajectory_message(trajectory).joint_trajectory
        if list(segment.joint_names) != names:
            raise ValueError("MoveIt trajectory joint order changed between contour segments.")
        for point_index, point in enumerate(segment.points):
            if segment_index > 0 and point_index == 0:
                continue
            merged = deepcopy(point)
            merged.time_from_start = _duration_from_seconds(0.0)
            output.points.append(merged)
    if len(output.points) < 2:
        raise RuntimeError("MoveIt produced fewer than two joint trajectory points.")
    return result


def nearest_equivalent_joint_positions(positions: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Unwrap equivalent revolute solutions to stay closest to the previous waypoint."""

    unwrapped = positions.copy()
    for i, value in enumerate(unwrapped):
        candidates = value + 2.0 * np.pi * np.arange(-3, 4)
        unwrapped[i] = candidates[int(np.argmin(np.abs(candidates - reference[i])))]
    return unwrapped


def build_ik_waypoint_trajectory(
    moveit,
    group_name: str,
    ee_link: str,
    poses: list[Pose],
    normals: np.ndarray,
    joint_names: list[str],
    ik_timeout: float,
    max_position_error: float,
    max_normal_error_deg: float,
    max_joint_step: float,
    joint_seeds: np.ndarray | None = None,
):
    """Solve every contour pose with MoveIt2 IK, seeded from the previous state."""

    state = RobotState(moveit.get_robot_model())
    result = RobotTrajectoryMsg()
    result.joint_trajectory.joint_names = joint_names
    failures: list[str] = []
    previous_positions: np.ndarray | None = None
    for index, pose in enumerate(poses):
        if joint_seeds is not None:
            state.set_joint_group_positions(group_name, joint_seeds[index])
            state.update()
        if not state.set_from_ik(group_name, pose, ee_link, ik_timeout):
            normal = normals[index]
            failures.append(
                f"index={index}, tcp=({pose.position.x:.6f},{pose.position.y:.6f},{pose.position.z:.6f}), "
                f"normal=({normal[0]:.6f},{normal[1]:.6f},{normal[2]:.6f})"
            )
            continue
        state.update()
        transform = _transform_matrix(state.get_global_link_transform(ee_link))
        target = np.array([pose.position.x, pose.position.y, pose.position.z], dtype=float)
        position_error = float(np.linalg.norm(transform[:3, 3] - target))
        tool_z = transform[:3, 2]
        normal_error = float(np.rad2deg(np.arccos(np.clip(np.dot(tool_z, normals[index]), -1.0, 1.0))))
        if position_error > max_position_error or normal_error > max_normal_error_deg:
            failures.append(
                f"index={index}, position_error={position_error * 1000.0:.3f} mm, "
                f"normal_error={normal_error:.3f} deg"
            )
            continue
        positions = np.asarray(state.get_joint_group_positions(group_name), dtype=float)
        if previous_positions is not None:
            positions = nearest_equivalent_joint_positions(positions, previous_positions)
            joint_delta = np.abs(positions - previous_positions)
            max_delta = float(np.max(joint_delta))
            if max_delta > max_joint_step:
                joint_index = int(np.argmax(joint_delta))
                failures.append(
                    f"index={index}, joint={joint_names[joint_index]}, "
                    f"joint_step={np.rad2deg(max_delta):.2f} deg"
                )
                continue
        previous_positions = positions.copy()
        point = JointTrajectoryPoint()
        point.positions = [float(value) for value in positions]
        point.time_from_start = _duration_from_seconds(0.0)
        result.joint_trajectory.points.append(point)
    if failures:
        preview = "; ".join(failures[:5])
        extra = "" if len(failures) <= 5 else f"; ... {len(failures) - 5} more"
        raise RuntimeError(f"MoveIt2 IK failed the closed contour gate: {preview}{extra}")
    if len(result.joint_trajectory.points) < 2:
        raise RuntimeError("MoveIt2 IK produced fewer than two contour waypoints.")
    return result


def build_seed_joint_trajectory(joint_seeds: np.ndarray, joint_names: list[str]):
    """Build a MoveIt RobotTrajectory message from a prevalidated continuous IK seed."""

    result = RobotTrajectoryMsg()
    result.joint_trajectory.joint_names = joint_names
    for seed in joint_seeds:
        point = JointTrajectoryPoint()
        point.positions = [float(value) for value in seed]
        point.time_from_start = _duration_from_seconds(0.0)
        result.joint_trajectory.points.append(point)
    if len(result.joint_trajectory.points) < 2:
        raise RuntimeError("Seed joint trajectory contains fewer than two waypoints.")
    return result


def _fk_positions_for_trajectory_msg(moveit, group_name: str, ee_link: str, trajectory_msg) -> np.ndarray:
    state = RobotState(moveit.get_robot_model())
    positions: list[np.ndarray] = []
    for point in trajectory_msg.joint_trajectory.points:
        state.set_joint_group_positions(group_name, point.positions)
        state.update()
        transform = _transform_matrix(state.get_global_link_transform(ee_link))
        positions.append(transform[:3, 3])
    return np.asarray(positions, dtype=float)


def _cumulative_arclength(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.array([], dtype=float)
    ds = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.r_[0.0, np.cumsum(ds)]


def _rotation_matrix_to_quaternion(matrix: np.ndarray) -> tuple[float, float, float, float]:
    m = np.asarray(matrix, dtype=float)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    else:
        axis = int(np.argmax(np.diag(m)))
        if axis == 0:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            qw = (m[2, 1] - m[1, 2]) / s
            qx = 0.25 * s
            qy = (m[0, 1] + m[1, 0]) / s
            qz = (m[0, 2] + m[2, 0]) / s
        elif axis == 1:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            qw = (m[0, 2] - m[2, 0]) / s
            qx = (m[0, 1] + m[1, 0]) / s
            qy = 0.25 * s
            qz = (m[1, 2] + m[2, 1]) / s
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            qw = (m[1, 0] - m[0, 1]) / s
            qx = (m[0, 2] + m[2, 0]) / s
            qy = (m[1, 2] + m[2, 1]) / s
            qz = 0.25 * s
    quat = np.array([qx, qy, qz, qw], dtype=float)
    quat /= max(float(np.linalg.norm(quat)), 1e-12)
    return tuple(float(v) for v in quat)


def _closed_loop_size_from_points(points: np.ndarray) -> int:
    distances = np.linalg.norm(points[1:] - points[0], axis=1)
    candidates = np.flatnonzero(distances < 1e-7) + 1
    candidates = candidates[candidates >= 16]
    return int(candidates[0]) if len(candidates) else len(points)


def build_wall_collision_objects(
    target_poses: list[Pose],
    target_normals: np.ndarray,
    stand_off: float,
    frame_id: str,
    wall_thickness: float,
    y_thickness: float,
    segment_stride: int,
) -> list[CollisionObject]:
    """Approximate the closed horseshoe wall as thin oriented boxes."""

    target_tcp = np.array([[pose.position.x, pose.position.y, pose.position.z] for pose in target_poses], dtype=float)
    wall_points = target_tcp - stand_off * target_normals
    loop_size = _closed_loop_size_from_points(wall_points)
    wall_loop = wall_points[:loop_size]
    objects: list[CollisionObject] = []
    stride = max(1, int(segment_stride))
    y_axis = np.array([0.0, 1.0, 0.0], dtype=float)
    for object_index, start in enumerate(range(0, loop_size, stride)):
        end = (start + stride) % loop_size
        p0 = wall_loop[start]
        p1 = wall_loop[end]
        segment = p1 - p0
        length = float(np.linalg.norm(segment))
        if length <= 1e-6:
            continue
        x_axis = segment / length
        z_axis = np.cross(x_axis, y_axis)
        z_axis /= max(float(np.linalg.norm(z_axis)), 1e-12)
        # The local box frame is x along the contour, y along tunnel station,
        # and z through wall thickness.
        rotation = np.column_stack([x_axis, y_axis, z_axis])
        qx, qy, qz, qw = _rotation_matrix_to_quaternion(rotation)
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = [length + wall_thickness, y_thickness, wall_thickness]
        pose = Pose()
        pose.position.x = float((p0[0] + p1[0]) * 0.5)
        pose.position.y = float((p0[1] + p1[1]) * 0.5)
        pose.position.z = float((p0[2] + p1[2]) * 0.5)
        pose.orientation.x = qx
        pose.orientation.y = qy
        pose.orientation.z = qz
        pose.orientation.w = qw
        obj = CollisionObject()
        obj.header.frame_id = frame_id
        obj.id = f"horseshoe_wall_{object_index:03d}"
        obj.primitives.append(primitive)
        obj.primitive_poses.append(pose)
        obj.operation = CollisionObject.ADD
        objects.append(obj)
    return objects


def apply_collision_environment(
    moveit,
    target_poses: list[Pose],
    target_normals: np.ndarray,
    stand_off: float,
    frame_id: str,
    wall_thickness: float,
    y_thickness: float,
    segment_stride: int,
) -> int:
    objects = build_wall_collision_objects(
        target_poses,
        target_normals,
        stand_off,
        frame_id,
        wall_thickness,
        y_thickness,
        segment_stride,
    )
    psm = moveit.get_planning_scene_monitor()
    with psm.read_write() as scene:
        scene.remove_all_collision_objects()
        for obj in objects:
            scene.apply_collision_object(obj)
    return len(objects)


def apply_tcp_arclength_timing_to_msg(
    moveit,
    group_name: str,
    ee_link: str,
    trajectory_msg,
    target_tcp_speed: float,
    zero_boundary_state: bool = True,
) -> None:
    """Assign timestamps from measured FK TCP arc length instead of joint spacing."""

    points = trajectory_msg.joint_trajectory.points
    if len(points) < 2:
        raise RuntimeError("TCP arclength timing needs at least two joint waypoints.")
    tcp = _fk_positions_for_trajectory_msg(moveit, group_name, ee_link, trajectory_msg)
    s = _cumulative_arclength(tcp)
    t = s / max(float(target_tcp_speed), 1e-6)
    min_dt = 1e-3
    for i in range(1, len(t)):
        if t[i] <= t[i - 1] + min_dt:
            t[i] = t[i - 1] + min_dt
    q = np.asarray([point.positions for point in points], dtype=float)
    dq = np.gradient(q, t, axis=0, edge_order=1)
    ddq = np.gradient(dq, t, axis=0, edge_order=1)
    if zero_boundary_state:
        dq[0, :] = 0.0
        dq[-1, :] = 0.0
        ddq[0, :] = 0.0
        ddq[-1, :] = 0.0
    for i, point in enumerate(points):
        point.time_from_start = _duration_from_seconds(float(t[i]))
        point.velocities = [float(value) for value in dq[i]]
        point.accelerations = [float(value) for value in ddq[i]]


def write_joint_trajectory_csv(robot_trajectory, output_path: Path) -> None:
    """Export the post-processed MoveIt trajectory for audit/replay."""

    trajectory = as_robot_trajectory_message(robot_trajectory).joint_trajectory
    names = list(trajectory.joint_names)
    t, q, dq, ddq, jerk = _trajectory_arrays(robot_trajectory)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["t"]
            + [f"{name}_q" for name in names]
            + [f"{name}_dq" for name in names]
            + [f"{name}_ddq" for name in names]
            + [f"{name}_jerk" for name in names]
        )
        for i in range(len(t)):
            writer.writerow(
                [t[i]]
                + q[i].tolist()
                + dq[i].tolist()
                + ddq[i].tolist()
                + jerk[i].tolist()
            )


def _duration_seconds(duration) -> float:
    return float(duration.sec) + float(duration.nanosec) * 1e-9


def _trajectory_arrays(robot_trajectory) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    trajectory = as_robot_trajectory_message(robot_trajectory).joint_trajectory
    if len(trajectory.points) < 2:
        raise RuntimeError("A post-processed trajectory must contain at least two points.")
    t = np.asarray([_duration_seconds(point.time_from_start) for point in trajectory.points], dtype=float)
    if np.any(np.diff(t) <= 0.0):
        raise RuntimeError("Trajectory timestamps must be strictly increasing.")
    q = np.asarray([point.positions for point in trajectory.points], dtype=float)
    dof = q.shape[1]
    if any(len(point.velocities) == dof for point in trajectory.points):
        dq = np.asarray(
            [point.velocities if len(point.velocities) == dof else np.zeros(dof) for point in trajectory.points],
            dtype=float,
        )
    else:
        dq = np.gradient(q, t, axis=0, edge_order=1)
    if any(len(point.accelerations) == dof for point in trajectory.points):
        ddq = np.asarray(
            [point.accelerations if len(point.accelerations) == dof else np.zeros(dof) for point in trajectory.points],
            dtype=float,
        )
    else:
        ddq = np.gradient(dq, t, axis=0, edge_order=1)
    jerk = np.gradient(ddq, t, axis=0, edge_order=1)
    return t, q, dq, ddq, jerk


def _transform_matrix(transform) -> np.ndarray:
    if isinstance(transform, np.ndarray):
        return np.asarray(transform, dtype=float)
    if hasattr(transform, "matrix"):
        matrix = transform.matrix() if callable(transform.matrix) else transform.matrix
        return np.asarray(matrix, dtype=float)
    if hasattr(transform, "A"):
        return np.asarray(transform.A, dtype=float)
    return np.asarray(transform, dtype=float)


def _parameter_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _nearest_polyline_distance(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    """Distance from each point to the closest segment of a closed 3D polyline."""

    if len(polyline) < 2:
        raise ValueError("A trajectory validation polyline needs at least two target points.")
    a = polyline
    b = np.roll(polyline, -1, axis=0)
    best = np.full(len(points), np.inf)
    for start, end in zip(a, b):
        edge = end - start
        denom = max(float(np.dot(edge, edge)), 1e-15)
        alpha = np.clip(((points - start) @ edge) / denom, 0.0, 1.0)
        projection = start + alpha[:, None] * edge
        best = np.minimum(best, np.linalg.norm(points - projection, axis=1))
    return best


def _project_to_polyline_with_normals(
    points: np.ndarray,
    polyline: np.ndarray,
    normals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project FK points to closed target segments and interpolate normals."""

    if len(polyline) < 2:
        raise ValueError("A trajectory validation polyline needs at least two target points.")
    if len(normals) != len(polyline):
        raise ValueError("Target normals and target polyline length differ.")
    a = polyline
    b = np.roll(polyline, -1, axis=0)
    na = normals
    nb = np.roll(normals, -1, axis=0)
    best_d2 = np.full(len(points), np.inf)
    best_projection = np.zeros_like(points)
    best_normal = np.zeros_like(points)
    for start, end, n_start, n_end in zip(a, b, na, nb):
        edge = end - start
        denom = max(float(np.dot(edge, edge)), 1e-15)
        alpha = np.clip(((points - start) @ edge) / denom, 0.0, 1.0)
        projection = start + alpha[:, None] * edge
        d2 = np.sum((points - projection) ** 2, axis=1)
        choose = d2 < best_d2
        interpolated_normal = (1.0 - alpha[:, None]) * n_start + alpha[:, None] * n_end
        interpolated_normal /= np.maximum(np.linalg.norm(interpolated_normal, axis=1, keepdims=True), 1e-12)
        best_d2[choose] = d2[choose]
        best_projection[choose] = projection[choose]
        best_normal[choose] = interpolated_normal[choose]
    return best_projection, best_normal, np.sqrt(best_d2)


def _tcp_speed_from_positions(points: np.ndarray, time: np.ndarray) -> np.ndarray:
    """Estimate speed from FK points without endpoint gradient artifacts."""

    if len(points) != len(time):
        raise ValueError("FK point and timestamp counts differ.")
    if len(points) < 2:
        return np.zeros(len(points), dtype=float)
    dt = np.diff(time)
    if np.any(dt <= 0.0):
        raise RuntimeError("Post-Ruckig trajectory has non-monotonic timestamps.")
    segment_speed = np.linalg.norm(np.diff(points, axis=0), axis=1) / dt
    speed = np.empty(len(points), dtype=float)
    speed[0] = segment_speed[0]
    speed[-1] = segment_speed[-1]
    if len(points) > 2:
        speed[1:-1] = 0.5 * (segment_speed[:-1] + segment_speed[1:])
    return speed


def preflight_moveit_runtime(moveit, group_name: str, ee_link: str, require_ruckig: bool) -> None:
    """Fail early when the target MoveItPy runtime cannot satisfy this chain."""

    robot_model = moveit.get_robot_model()
    if not robot_model.has_joint_model_group(group_name):
        raise RuntimeError(f"MoveIt robot model has no joint group {group_name!r}.")
    group = robot_model.get_joint_model_group(group_name)
    if ee_link not in list(group.link_model_names):
        raise RuntimeError(f"End-effector link {ee_link!r} is not in MoveIt group {group_name!r}.")
    active_joint_names = list(group.active_joint_model_names)
    if len(active_joint_names) != 6:
        raise RuntimeError(f"Expected 6 active FR5 joints in {group_name!r}, found {active_joint_names}.")
    configured_jerk = _configured_jerk_limits()
    missing_jerk = [joint_name for joint_name in active_joint_names if joint_name not in configured_jerk]
    if missing_jerk:
        raise RuntimeError("Bridge jerk limits are missing for " + ", ".join(missing_jerk))
    if require_ruckig and not hasattr(RobotTrajectory, "apply_ruckig_smoothing"):
        raise RuntimeError("This MoveItPy runtime does not expose RobotTrajectory.apply_ruckig_smoothing().")


def moveit_group_limits(moveit, group_name: str) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    group = moveit.get_robot_model().get_joint_model_group(group_name)
    joint_names = list(group.active_joint_model_names)
    configured_jerk = _configured_jerk_limits()
    velocity_limits = []
    acceleration_limits = []
    jerk_limits = []
    for joint_name, bounds in zip(joint_names, group.active_joint_model_bounds):
        if isinstance(bounds, (list, tuple)):
            if not bounds:
                raise RuntimeError(f"MoveIt joint {joint_name} has no bounds.")
            bounds = bounds[0]
        velocity_limits.append(float(bounds.max_velocity))
        acceleration_limits.append(float(bounds.max_acceleration))
        bound_jerk = float(getattr(bounds, "max_jerk", 0.0))
        jerk_limits.append(bound_jerk if np.isfinite(bound_jerk) and bound_jerk > 0.0 else configured_jerk[joint_name])
    return joint_names, np.asarray(velocity_limits), np.asarray(acceleration_limits), np.asarray(jerk_limits)


def validate_joint_dynamics(
    moveit,
    robot_trajectory,
    group_name: str,
    report_path: Path,
    limit_margin: float = 1.02,
) -> dict[str, float | str]:
    """Reject post-Ruckig trajectories that violate MoveIt joint limits."""

    joint_names, velocity_limits, acceleration_limits, jerk_limits = moveit_group_limits(moveit, group_name)
    _, _, dq, ddq, jerk = _trajectory_arrays(robot_trajectory)
    max_velocity = np.max(np.abs(dq), axis=0)
    max_acceleration = np.max(np.abs(ddq), axis=0)
    max_jerk = np.max(np.abs(jerk), axis=0)
    p95_jerk = np.percentile(np.abs(jerk), 95, axis=0)
    failures: list[str] = []
    rows: list[list[float | str]] = []
    for i, joint_name in enumerate(joint_names):
        velocity_ratio = max_velocity[i] / max(velocity_limits[i], 1e-12)
        acceleration_ratio = max_acceleration[i] / max(acceleration_limits[i], 1e-12)
        jerk_ratio = max_jerk[i] / max(jerk_limits[i], 1e-12)
        if velocity_ratio > limit_margin:
            failures.append(f"{joint_name} velocity")
        if acceleration_ratio > limit_margin:
            failures.append(f"{joint_name} acceleration")
        if jerk_ratio > limit_margin:
            failures.append(f"{joint_name} jerk")
        rows.append(
            [
                joint_name,
                max_velocity[i],
                velocity_limits[i],
                velocity_ratio,
                max_acceleration[i],
                acceleration_limits[i],
                acceleration_ratio,
                max_jerk[i],
                p95_jerk[i],
                jerk_limits[i],
                jerk_ratio,
            ]
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "joint",
                "max_velocity",
                "velocity_limit",
                "velocity_ratio",
                "max_acceleration",
                "acceleration_limit",
                "acceleration_ratio",
                "max_jerk",
                "p95_jerk",
                "jerk_limit",
                "jerk_ratio",
            ]
        )
        writer.writerows(rows)
    metrics: dict[str, float | str] = {
        "max_velocity_ratio": float(np.max(max_velocity / np.maximum(velocity_limits, 1e-12))),
        "max_acceleration_ratio": float(np.max(max_acceleration / np.maximum(acceleration_limits, 1e-12))),
        "max_jerk_ratio": float(np.max(max_jerk / np.maximum(jerk_limits, 1e-12))),
        "status": "pass" if not failures else "fail:" + ",".join(failures),
    }
    if failures:
        raise RuntimeError("Refusing to execute post-Ruckig trajectory; joint dynamics gate failed: " + ", ".join(failures))
    return metrics


def validate_trajectory_collision(
    moveit,
    robot_trajectory,
    group_name: str,
    report_path: Path,
    stride: int = 1,
    environment_object_count: int = 0,
) -> dict[str, float | str]:
    """Check self/world collision for sampled post-Ruckig states."""

    trajectory = as_robot_trajectory_message(robot_trajectory).joint_trajectory
    state = RobotState(moveit.get_robot_model())
    sample_indices = list(range(0, len(trajectory.points), max(1, stride)))
    if sample_indices[-1] != len(trajectory.points) - 1:
        sample_indices.append(len(trajectory.points) - 1)
    collision_indices: list[int] = []
    psm = moveit.get_planning_scene_monitor()
    with psm.read_only() as scene:
        for index in sample_indices:
            point = trajectory.points[index]
            state.set_joint_group_positions(group_name, point.positions)
            state.update()
            if scene.is_state_colliding(state, group_name, False):
                collision_indices.append(index)
    metrics: dict[str, float | str] = {
        "collision_environment_object_count": float(environment_object_count),
        "collision_checked_state_count": float(len(sample_indices)),
        "collision_count": float(len(collision_indices)),
        "first_collision_index": float(collision_indices[0]) if collision_indices else -1.0,
        "status": "pass" if not collision_indices else "fail:collision",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows(metrics.items())
    if collision_indices:
        raise RuntimeError(
            "Refusing to execute post-Ruckig trajectory; collision gate failed at trajectory indices: "
            + ", ".join(str(i) for i in collision_indices[:10])
        )
    return metrics


def validate_smoothed_trajectory_fk(
    moveit,
    robot_trajectory,
    group_name: str,
    ee_link: str,
    target_poses: list[Pose],
    target_normals: np.ndarray,
    stand_off: float,
    report_path: Path,
    max_normal_error_deg: float = 10.0,
    max_standoff_fraction: float = 0.05,
    max_speed_fluctuation: float = 0.05,
    max_path_deviation: float = 0.005,
    stride: int = 1,
    ruckig_smoothing_used: bool = True,
    time_parameterization: str = "unknown",
    fk_trace_path: Path | None = None,
) -> dict[str, float | str]:
    """Reject a post-Ruckig trajectory unless FK satisfies coating constraints."""

    trajectory = as_robot_trajectory_message(robot_trajectory).joint_trajectory
    state = RobotState(moveit.get_robot_model())
    target_tcp = np.array(
        [[pose.position.x, pose.position.y, pose.position.z] for pose in target_poses],
        dtype=float,
    )
    wall = target_tcp - stand_off * target_normals
    positions: list[np.ndarray] = []
    tool_axes: list[np.ndarray] = []
    times: list[float] = []
    sample_indices = list(range(0, len(trajectory.points), max(1, stride)))
    if sample_indices[-1] != len(trajectory.points) - 1:
        sample_indices.append(len(trajectory.points) - 1)
    for index in sample_indices:
        point = trajectory.points[index]
        state.set_joint_group_positions(group_name, point.positions)
        state.update()
        transform = _transform_matrix(state.get_global_link_transform(ee_link))
        positions.append(transform[:3, 3])
        tool_axes.append(transform[:3, 2])
        times.append(_duration_seconds(point.time_from_start))

    actual = np.asarray(positions)
    tool_z = np.asarray(tool_axes)
    time = np.asarray(times)
    if len(actual) < 2 or np.any(np.diff(time) <= 0.0):
        raise RuntimeError("Post-Ruckig trajectory has invalid or non-monotonic timestamps.")
    path_deviation = _nearest_polyline_distance(actual, target_tcp)
    wall_nearest, normals, _ = _project_to_polyline_with_normals(actual, wall, target_normals)
    normal_error = np.rad2deg(
        np.arccos(np.clip(np.sum(tool_z * normals, axis=1), -1.0, 1.0))
    )
    standoff_error = np.sum((actual - wall_nearest) * normals, axis=1) - stand_off
    speed = _tcp_speed_from_positions(actual, time)
    speed_mean = max(float(np.mean(speed)), 1e-12)
    speed_p05 = float(np.percentile(speed, 5))
    speed_p95 = float(np.percentile(speed, 95))
    robust_fluctuation = (speed_p95 - speed_p05) / speed_mean
    if fk_trace_path is not None:
        fk_trace_path.parent.mkdir(parents=True, exist_ok=True)
        with fk_trace_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "t",
                    "actual_tcp_x",
                    "actual_tcp_y",
                    "actual_tcp_z",
                    "tool_z_x",
                    "tool_z_y",
                    "tool_z_z",
                    "wall_normal_x",
                    "wall_normal_y",
                    "wall_normal_z",
                    "normal_angle_error_deg",
                    "standoff_error_mm",
                    "path_deviation_mm",
                    "tcp_speed_m_s",
                ]
            )
            for i in range(len(actual)):
                writer.writerow(
                    [
                        time[i],
                        actual[i, 0],
                        actual[i, 1],
                        actual[i, 2],
                        tool_z[i, 0],
                        tool_z[i, 1],
                        tool_z[i, 2],
                        normals[i, 0],
                        normals[i, 1],
                        normals[i, 2],
                        normal_error[i],
                        standoff_error[i] * 1000.0,
                        path_deviation[i] * 1000.0,
                        speed[i],
                    ]
                )
    metrics: dict[str, float | str] = {
        "fk_normal_error_mean_deg": float(np.mean(normal_error)),
        "fk_normal_error_max_deg": float(np.max(normal_error)),
        "fk_standoff_error_max_abs_mm": float(np.max(np.abs(standoff_error)) * 1000.0),
        "fk_path_deviation_p95_mm": float(np.percentile(path_deviation, 95) * 1000.0),
        "fk_path_deviation_max_mm": float(np.max(path_deviation) * 1000.0),
        "fk_tcp_speed_mean_m_s": speed_mean,
        "fk_tcp_speed_p05_m_s": speed_p05,
        "fk_tcp_speed_p95_m_s": speed_p95,
        "fk_tcp_speed_p05_p95_fluctuation": robust_fluctuation,
        "moveit_ruckig_smoothing_used": "true" if ruckig_smoothing_used else "false",
        "moveit_time_parameterization": time_parameterization,
        "moveit_ee_link": ee_link,
    }
    failures: list[str] = []
    if float(metrics["fk_normal_error_max_deg"]) > max_normal_error_deg:
        failures.append("normal angle")
    if float(metrics["fk_standoff_error_max_abs_mm"]) > stand_off * max_standoff_fraction * 1000.0:
        failures.append("stand-off")
    if float(metrics["fk_path_deviation_max_mm"]) > max_path_deviation * 1000.0:
        failures.append("TCP path deviation")
    if robust_fluctuation > max_speed_fluctuation:
        failures.append("TCP speed")
    metrics["status"] = "pass" if not failures else "fail:" + ",".join(failures)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows(metrics.items())
    if failures:
        raise RuntimeError(
            "Refusing to execute post-Ruckig trajectory; FK quality gate failed: " + ", ".join(failures)
        )
    return metrics


def _duration_from_seconds(seconds: float):
    from builtin_interfaces.msg import Duration

    duration = Duration()
    duration.sec = int(seconds)
    duration.nanosec = int((seconds - duration.sec) * 1_000_000_000)
    return duration


def _apply_ruckig_smoothing_without_known_false_error(
    trajectory,
    velocity_scaling: float,
    acceleration_scaling: float,
    mitigate_overshoot: bool,
    overshoot_threshold: float,
) -> bool:
    """Run MoveIt Ruckig while filtering one known non-fatal C++ log line.

    MoveIt2 Jazzy's Ruckig adapter can emit
    "extended the trajectory duration..." at ERROR level even when the returned
    trajectory passes FK, dynamics, and collision validation. Capturing stderr
    only around this native call keeps the strict run log actionable; any other
    native output is replayed unchanged.
    """

    known = "Ruckig extended the trajectory duration to its maximum and still did not find a solution"
    stderr_fd = 2
    saved_stderr = os.dup(stderr_fd)
    captured_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w+b", delete=False) as captured:
            captured_path = captured.name
            os.dup2(captured.fileno(), stderr_fd)
            try:
                ok = trajectory.apply_ruckig_smoothing(
                    velocity_scaling,
                    acceleration_scaling,
                    mitigate_overshoot=mitigate_overshoot,
                    overshoot_threshold=overshoot_threshold,
                )
            finally:
                sys.stderr.flush()
                os.dup2(saved_stderr, stderr_fd)
        text = Path(captured_path).read_text(encoding="utf-8", errors="replace")
        replay = "\n".join(line for line in text.splitlines() if known not in line)
        if replay:
            print(replay, file=sys.stderr)
        if not ok and known in text:
            print(text, file=sys.stderr, end="" if text.endswith("\n") else "\n")
        return bool(ok)
    finally:
        os.close(saved_stderr)
        if captured_path:
            try:
                Path(captured_path).unlink()
            except FileNotFoundError:
                pass


def build_and_smooth_moveit_trajectory(
    moveit,
    group_name: str,
    ee_link: str,
    start_state,
    trajectory_msg,
    velocity_scaling: float,
    acceleration_scaling: float,
    use_ruckig: bool,
    time_parameterization: str = "totg",
    target_tcp_speed: float = 0.003,
    zero_boundary_state: bool = True,
):
    """Convert the merged message and use MoveIt2's native timing/Ruckig."""

    trajectory = RobotTrajectory(moveit.get_robot_model())
    trajectory.joint_model_group_name = group_name
    if time_parameterization == "tcp_arclength":
        apply_tcp_arclength_timing_to_msg(
            moveit,
            group_name,
            ee_link,
            trajectory_msg,
            target_tcp_speed,
            zero_boundary_state=zero_boundary_state,
        )
    elif time_parameterization != "totg":
        raise ValueError(f"Unsupported time_parameterization={time_parameterization!r}.")
    trajectory.set_robot_trajectory_msg(start_state, trajectory_msg)
    trajectory.unwind()
    if time_parameterization == "totg":
        if not trajectory.apply_totg_time_parameterization(
            velocity_scaling,
            acceleration_scaling,
            path_tolerance=0.01,
            resample_dt=0.01,
            min_angle_change=0.0005,
        ):
            raise RuntimeError("MoveIt2 TOTG failed to time-parameterize the merged contour.")
    if use_ruckig and not _apply_ruckig_smoothing_without_known_false_error(
        trajectory,
        velocity_scaling,
        acceleration_scaling,
        mitigate_overshoot=True,
        overshoot_threshold=0.005,
    ):
        raise RuntimeError("MoveIt2 native Ruckig smoothing failed on the merged contour.")
    return trajectory


def main() -> None:
    rclpy.init()
    node = rclpy.create_node("fr5_closed_contour_moveit_planner")
    tcp_path = Path(node.declare_parameter("tcp_path_csv", "outputs/tcp_poses.csv").value)
    group_name = str(node.declare_parameter("group_name", "fairino5_v6_group").value)
    base_frame = str(node.declare_parameter("base_frame", "base_link").value)
    ee_link = str(node.declare_parameter("ee_link", "spray_tcp_link").value)
    use_ruckig = _parameter_bool(node.declare_parameter("use_ruckig_smoothing", True).value)
    velocity_scaling = float(node.declare_parameter("velocity_scaling", 0.15).value)
    acceleration_scaling = float(node.declare_parameter("acceleration_scaling", 0.15).value)
    time_parameterization = str(node.declare_parameter("time_parameterization", "tcp_arclength").value)
    target_tcp_speed = float(node.declare_parameter("target_tcp_speed", 0.003).value)
    zero_boundary_state = _parameter_bool(node.declare_parameter("zero_boundary_state", True).value)
    waypoint_stride = max(1, int(node.declare_parameter("waypoint_stride", 4).value))
    planning_mode = str(node.declare_parameter("planning_mode", "ik_waypoints").value)
    joint_names = [
        name.strip()
        for name in str(node.declare_parameter("joint_names", "j1,j2,j3,j4,j5,j6").value).split(",")
        if name.strip()
    ]
    ik_timeout = float(node.declare_parameter("ik_timeout", 0.05).value)
    max_ik_position_error = float(node.declare_parameter("max_ik_position_error", 0.002).value)
    max_ik_joint_step = np.deg2rad(float(node.declare_parameter("max_ik_joint_step_deg", 20.0).value))
    seed_joint_csv = str(node.declare_parameter("seed_joint_csv", "").value).strip()
    stand_off = float(node.declare_parameter("stand_off", 0.18).value)
    validate_fk = _parameter_bool(node.declare_parameter("validate_post_ruckig_fk", True).value)
    validate_dynamics = _parameter_bool(node.declare_parameter("validate_joint_dynamics", True).value)
    validation_stride = max(1, int(node.declare_parameter("validation_stride", 1).value))
    max_path_deviation = float(node.declare_parameter("max_path_deviation", 0.005).value)
    max_normal_error_deg = float(node.declare_parameter("max_normal_error_deg", 10.0).value)
    max_standoff_fraction = float(node.declare_parameter("max_standoff_fraction", 0.05).value)
    max_speed_fluctuation = float(node.declare_parameter("max_speed_fluctuation", 0.05).value)
    quality_report = Path(node.declare_parameter("quality_report_csv", "outputs/moveit_quality_report.csv").value)
    fk_trace_csv = Path(node.declare_parameter("fk_trace_csv", "outputs/moveit_fk_tcp_trace.csv").value)
    trajectory_csv = Path(node.declare_parameter("trajectory_csv", "outputs/moveit_smoothed_joint_trajectory.csv").value)
    dynamics_report = Path(node.declare_parameter("dynamics_report_csv", "outputs/moveit_joint_dynamics_report.csv").value)
    collision_report = Path(node.declare_parameter("collision_report_csv", "outputs/moveit_collision_report.csv").value)
    validate_collision = _parameter_bool(node.declare_parameter("validate_collision", True).value)
    collision_check_stride = max(1, int(node.declare_parameter("collision_check_stride", validation_stride).value))
    collision_segment_stride = max(1, int(node.declare_parameter("collision_segment_stride", 4).value))
    tunnel_wall_thickness = float(node.declare_parameter("tunnel_wall_thickness", 0.025).value)
    tunnel_y_thickness = float(node.declare_parameter("tunnel_y_thickness", 0.08).value)
    fast_exit_after_reports = _parameter_bool(node.declare_parameter("fast_exit_after_reports", False).value)
    execute_trajectory = _parameter_bool(node.declare_parameter("execute_trajectory", False).value)
    if execute_trajectory and not use_ruckig:
        raise RuntimeError("Execution requires MoveIt2 native Ruckig smoothing; set use_ruckig_smoothing:=true.")
    if execute_trajectory and not (validate_fk and validate_dynamics):
        raise RuntimeError("Execution requires both validate_post_ruckig_fk and validate_joint_dynamics to be enabled.")

    # Keep the C++ MoveItPy node name aligned with the launch parameter scope.
    moveit = MoveItPy(node_name="fr5_closed_contour_moveit_planner")
    planning_component = moveit.get_planning_component(group_name)
    preflight_moveit_runtime(moveit, group_name, ee_link, use_ruckig)
    execution_start_state = RobotState(moveit.get_robot_model())
    waypoints = load_tcp_poses(tcp_path)
    waypoint_normals = load_tcp_normals(tcp_path)
    if len(waypoints) != len(waypoint_normals):
        raise ValueError(
            f"TCP pose/normal count mismatch in {tcp_path}: {len(waypoints)} poses vs {len(waypoint_normals)} normals."
        )
    collision_object_count = 0
    if validate_collision:
        collision_object_count = apply_collision_environment(
            moveit,
            waypoints,
            waypoint_normals,
            stand_off,
            base_frame,
            tunnel_wall_thickness,
            tunnel_y_thickness,
            collision_segment_stride,
        )
    joint_seeds = None
    if seed_joint_csv:
        joint_seeds = load_joint_seed_csv(Path(seed_joint_csv), joint_names)
        if len(joint_seeds) != len(waypoints):
            raise ValueError(
                f"Seed joint count mismatch: {seed_joint_csv} has {len(joint_seeds)} rows, "
                f"but {tcp_path} has {len(waypoints)} poses."
            )
    if planning_mode == "seed_joint_waypoints":
        if joint_seeds is None:
            raise ValueError("planning_mode=seed_joint_waypoints requires seed_joint_csv.")
        node.get_logger().info("Using continuous seed joint waypoints for closed-contour tracking.")
        trajectory_msg = build_seed_joint_trajectory(joint_seeds, joint_names)
    elif planning_mode == "ik_waypoints":
        node.get_logger().info("Using dense MoveIt2 IK waypoints for closed-contour tracking.")
        trajectory_msg = build_ik_waypoint_trajectory(
            moveit,
            group_name,
            ee_link,
            waypoints,
            waypoint_normals,
            joint_names,
            ik_timeout,
            max_ik_position_error,
            max_normal_error_deg,
            max_ik_joint_step,
            joint_seeds,
        )
    elif planning_mode == "ompl_segments":
        selected_indices = list(range(0, len(waypoints), waypoint_stride))
        if selected_indices[-1] != len(waypoints) - 1:
            selected_indices.append(len(waypoints) - 1)
        selected = [waypoints[index] for index in selected_indices]

        # Chain every segment from the previous planned endpoint. Nothing is sent
        # to the controller until the complete contour has passed one global
        # Ruckig smoothing pass.
        planning_component.set_start_state_to_current_state()
        planned_segments = []
        planned_state = RobotState(moveit.get_robot_model())
        for segment_index, (source_index, pose) in enumerate(zip(selected_indices, selected)):
            planning_component.set_goal_state(pose_stamped_msg=to_pose_stamped(pose, base_frame), pose_link=ee_link)
            plan_result = planning_component.plan()
            if not plan_result:
                normal = waypoint_normals[source_index]
                raise RuntimeError(
                    "MoveIt2 failed to plan a closed-contour waypoint segment "
                    f"segment={segment_index}, source_index={source_index}, "
                    f"tcp=({pose.position.x:.6f}, {pose.position.y:.6f}, {pose.position.z:.6f}), "
                    f"normal=({normal[0]:.6f}, {normal[1]:.6f}, {normal[2]:.6f})."
                )
            trajectory_msg = as_robot_trajectory_message(plan_result.trajectory)
            planned_segments.append(trajectory_msg)
            endpoint = trajectory_msg.joint_trajectory.points[-1].positions
            planned_state.set_joint_group_positions(group_name, endpoint)
            planned_state.update()
            planning_component.set_start_state(robot_state=planned_state)
        trajectory_msg = concatenate_robot_trajectories(planned_segments)
    else:
        raise ValueError(
            f"Unsupported planning_mode={planning_mode!r}. "
            "Use seed_joint_waypoints, ik_waypoints, or ompl_segments."
        )
    trajectory = build_and_smooth_moveit_trajectory(
        moveit,
        group_name,
        ee_link,
        execution_start_state,
        trajectory_msg,
        velocity_scaling,
        acceleration_scaling,
        use_ruckig,
        time_parameterization=time_parameterization,
        target_tcp_speed=target_tcp_speed,
        zero_boundary_state=zero_boundary_state,
    )
    if validate_fk:
        validate_smoothed_trajectory_fk(
            moveit,
            trajectory,
            group_name,
            ee_link,
            waypoints,
            waypoint_normals,
            stand_off,
            quality_report,
            max_normal_error_deg=max_normal_error_deg,
            max_standoff_fraction=max_standoff_fraction,
            max_speed_fluctuation=max_speed_fluctuation,
            max_path_deviation=max_path_deviation,
            stride=validation_stride,
            ruckig_smoothing_used=use_ruckig,
            time_parameterization=time_parameterization,
            fk_trace_path=fk_trace_csv,
        )
    if validate_dynamics:
        validate_joint_dynamics(moveit, trajectory, group_name, dynamics_report)
    if validate_collision:
        validate_trajectory_collision(
            moveit,
            trajectory,
            group_name,
            collision_report,
            stride=collision_check_stride,
            environment_object_count=collision_object_count,
        )
    write_joint_trajectory_csv(trajectory, trajectory_csv)
    if execute_trajectory:
        moveit.execute(trajectory, controllers=[])
    else:
        node.get_logger().warn(
            "Trajectory was planned, Ruckig-smoothed, FK-validated, and exported, "
            "but not executed because execute_trajectory is false."
        )

    if fast_exit_after_reports:
        node.get_logger().info(
            "All MoveIt2 reports were written; using fast process exit to avoid "
            "MoveItPy shutdown/destructor instability."
        )
        sys.stdout.flush()
        sys.stderr.flush()
        time.sleep(0.1)
        os._exit(0)

    rclpy.shutdown()


if __name__ == "__main__":
    main()
