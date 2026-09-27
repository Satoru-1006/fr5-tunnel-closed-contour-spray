#!/usr/bin/env python3
"""MoveIt2 execution-chain entry for closed horseshoe contour tracking.

Run this inside a ROS2 workspace that contains FAIRINO's frcobot_ros2 packages.
It reads the generated TCP path CSV and asks MoveIt2 for a constrained Cartesian
plan. This file is intentionally separate from the Windows/Python visual demo.
"""

from __future__ import annotations

import csv
import json
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
from rcl_interfaces.msg import ParameterDescriptor
from moveit_msgs.msg import RobotTrajectory as RobotTrajectoryMsg
from moveit_msgs.msg import CollisionObject
from moveit.core.robot_state import RobotState
from moveit.core.robot_trajectory import RobotTrajectory
from moveit.core.collision_detection import CollisionRequest
from moveit.core.collision_detection import CollisionResult
from moveit.planning import MoveItPy
from shape_msgs.msg import SolidPrimitive
from trajectory_msgs.msg import JointTrajectoryPoint
from p2b1_solver_policy_ablation import (
    append_pressure_record as _append_p2b1_pressure_record,
    dls_primary_command as _p2b1_dls_primary_command,
    joint_centering_objective as _p2b1_joint_centering_objective,
    nullspace_metrics as _p2b1_nullspace_metrics,
    process_jacobian as _p2b1_process_jacobian,
    process_residual as _p2b1_process_residual,
    project_joint_limits as _p2b1_project_joint_limits,
    secondary_command as _p2b1_secondary_command,
    tangent_bases as _p2b1_tangent_bases,
    weighted_task as _p2b1_weighted_task,
)
from p2b2_redundancy_objectives import (
    rank_aware_secondary_command as _p2b2_rank_aware_secondary_command,
    secondary_objective as _p2b2_secondary_objective,
)


def _configured_jerk_limits() -> dict[str, float]:
    bridge_share = Path(get_package_share_directory("fr5_tunnel_moveit_bridge"))
    limits_path = bridge_share / "config" / "joint_limits_with_jerk.yaml"
    data = yaml.safe_load(limits_path.read_text(encoding="utf-8"))
    return {
        joint_name: float(joint_config["max_jerk"])
        for joint_name, joint_config in data.get("joint_limits", {}).items()
        if joint_config.get("has_jerk_limits", False)
    }


def validate_p2b1_moveit_fd_consistency(
    moveit, group_name: str, ee_link: str, target_poses: list[Pose], target_normals: np.ndarray,
    frozen_q_csv: str | Path, output_json: str | Path,
) -> dict[str, object]:
    """Finite-difference the five process residuals at frozen D41 states."""
    q_columns = [f"j{index}_q" for index in range(1, 7)]
    with Path(frozen_q_csv).open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(target_poses) or len(rows) != len(target_normals):
        raise RuntimeError("p2b1_fd_frozen_waypoint_count_mismatch")
    frozen_q = np.asarray([[float(row[name]) for name in q_columns] for row in rows], dtype=np.float64)
    if frozen_q.shape != (len(target_poses), 6) or not np.isfinite(frozen_q).all():
        raise RuntimeError("p2b1_fd_frozen_joint_matrix_invalid")
    indices = sorted(set(index for index in (80, 81, 86, 87, 95) if index < len(frozen_q)))
    if len(indices) < 3:
        raise RuntimeError("p2b1_fd_requires_multiple_frozen_waypoints")
    patterns = (
        np.asarray([1, -1, 1, -1, 1, -1], dtype=np.float64),
        np.asarray([0, 0, 0, 0, 0, 1], dtype=np.float64),
        np.asarray([1, 1, -1, -1, 1, 1], dtype=np.float64),
    )
    steps = (1.0e-5, 5.0e-5)
    state = RobotState(moveit.get_robot_model())
    records: list[dict[str, object]] = []
    for index in indices:
        q = frozen_q[index]
        normal, basis, _angular_basis = _p2b1_tangent_bases(target_normals[index])
        state.set_joint_group_positions(group_name, q)
        state.update()
        transform = _transform_matrix(state.get_global_link_transform(ee_link))
        jacobian = np.asarray(state.get_jacobian(group_name, ee_link, np.zeros(3, dtype=float), False), dtype=np.float64)
        if jacobian.shape != (6, 6) or not np.isfinite(jacobian).all():
            raise RuntimeError(f"p2b1_fd_invalid_moveit_jacobian:index={index}")
        tool_z = transform[:3, 2] / max(float(np.linalg.norm(transform[:3, 2])), 1e-12)
        analytic = _p2b1_process_jacobian(jacobian[:3], jacobian[3:], tool_z, normal, basis)
        target_position = np.asarray([
            target_poses[index].position.x, target_poses[index].position.y, target_poses[index].position.z,
        ], dtype=np.float64)
        baseline_residual = _p2b1_process_residual(transform[:3, 3], target_position, tool_z, normal, basis)
        for pattern_index, raw_pattern in enumerate(patterns):
            direction = raw_pattern / np.linalg.norm(raw_pattern)
            for epsilon in steps:
                delta_q = direction * epsilon
                q_plus = q + delta_q
                state.set_joint_group_positions(group_name, q_plus)
                state.update()
                transform_plus = _transform_matrix(state.get_global_link_transform(ee_link))
                tool_z_plus = transform_plus[:3, 2] / max(float(np.linalg.norm(transform_plus[:3, 2])), 1e-12)
                residual_plus = _p2b1_process_residual(
                    transform_plus[:3, 3], target_position, tool_z_plus, normal, basis,
                )
                observed = residual_plus - baseline_residual
                predicted = analytic @ delta_q
                error = float(np.linalg.norm(observed - predicted))
                scale = float(np.linalg.norm(delta_q))
                error_limit = 2.0e-8 + 5.0 * scale * scale
                records.append({
                    "waypoint": index, "pattern": pattern_index, "step_rad": epsilon,
                    "delta_norm_rad": scale, "first_order_prediction_error": error,
                    "second_order_allowance": 5.0 * scale * scale,
                    "absolute_error_tolerance": 2.0e-8, "status": "PASS" if error <= error_limit else "FAIL",
                })
    result = {
        "schema": "p2b1-process-jacobian-fd-v1", "status": "PASS" if all(row["status"] == "PASS" for row in records) else "FAIL",
        "waypoint_indices_zero_based": indices, "perturbation_directions_per_waypoint": len(patterns),
        "step_sizes_rad": list(steps), "finite_difference_case_count": len(records),
        "residual": "[p(q)-p_target; B_n^T(z(q)-n)]",
        "jacobian": "[J_v; -B_n^T[z(q)]x J_omega]",
        "aligned_state_identity": "U_n=[n]xB_n; lower block equals U_n^T J_omega when z=n",
        "convention": "MoveIt2 RobotState global-link transform and spatial angular Jacobian; checked numerically",
        "maximum_first_order_prediction_error": max((float(row["first_order_prediction_error"]) for row in records), default=None),
        "cases": records,
    }
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(output_json).write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    if result["status"] != "PASS":
        raise RuntimeError("p2b1_process_residual_jacobian_fd_consistency_failed")
    return result


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


def circular_joint_delta(delta: np.ndarray) -> np.ndarray:
    """Return shortest equivalent revolute-joint deltas for continuity diagnostics."""

    return (np.asarray(delta, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def _select_diverse_ik_beams(
    sorted_beams: list[tuple[float, float, float, list[np.ndarray]]],
    beam_width: int,
    diversity_threshold: float,
) -> list[tuple[float, float, float, list[np.ndarray]]]:
    """Keep low-cost IK beams while avoiding premature collapse to one branch."""

    if beam_width <= 0 or not sorted_beams:
        return []
    if diversity_threshold <= 0.0:
        return sorted_beams[:beam_width]

    selected: list[tuple[float, float, float, list[np.ndarray]]] = []
    deferred: list[tuple[float, float, float, list[np.ndarray]]] = []
    for beam in sorted_beams:
        positions = beam[3][-1] if beam[3] else None
        if positions is None:
            selected.append(beam)
        else:
            too_close = False
            for existing in selected:
                existing_positions = existing[3][-1] if existing[3] else None
                if existing_positions is None:
                    continue
                separation = float(np.max(np.abs(circular_joint_delta(positions - existing_positions))))
                if separation < diversity_threshold:
                    too_close = True
                    break
            if too_close:
                deferred.append(beam)
            else:
                selected.append(beam)
        if len(selected) >= beam_width:
            break
    if len(selected) < beam_width:
        selected.extend(deferred[: beam_width - len(selected)])
    return selected[:beam_width]


def _ik_seed_candidates(
    primary_seed: np.ndarray,
    previous_positions: np.ndarray | None,
) -> list[np.ndarray]:
    """Generate branch seeds for collision-aware continuous IK retry."""

    seeds: list[np.ndarray] = []

    def add(seed: np.ndarray) -> None:
        if not any(np.allclose(seed, existing, atol=1e-9, rtol=0.0) for existing in seeds):
            seeds.append(seed.copy())

    add(np.asarray(primary_seed, dtype=float))
    if previous_positions is not None:
        previous = np.asarray(previous_positions, dtype=float)
        add(previous)
        for joint_index in range(len(previous)):
            for offset in (-2.0 * np.pi, -np.pi, np.pi, 2.0 * np.pi):
                seed = previous.copy()
                seed[joint_index] += offset
                add(seed)
        branch_pairs = ((0, 1), (1, 2), (2, 3), (0, 5), (3, 5), (4, 5))
        for joint_a, joint_b in branch_pairs:
            if joint_a >= len(previous) or joint_b >= len(previous):
                continue
            for offset_a in (-np.pi, np.pi):
                for offset_b in (-np.pi, np.pi):
                    seed = previous.copy()
                    seed[joint_a] += offset_a
                    seed[joint_b] += offset_b
                    add(seed)
    return seeds


def _candidate_collection_seed_candidates(primary_seed: np.ndarray) -> list[np.ndarray]:
    """Return a bounded branch-seed set for offline IK candidate collection.

    The production beam search deliberately explores a much larger seed set.
    Candidate-bank generation must be repeatable and bounded, so it starts from
    the verified continuous seed and samples one-joint shoulder/elbow/wrist
    branches without allowing 2*pi duplicates to dominate the solve budget.
    """

    base = np.asarray(primary_seed, dtype=float)
    seeds: list[np.ndarray] = [base.copy()]
    for joint_index in range(len(base)):
        for offset in (-np.pi, np.pi):
            candidate = base.copy()
            candidate[joint_index] += offset
            if not any(np.allclose(candidate, existing, atol=1e-9, rtol=0.0) for existing in seeds):
                seeds.append(candidate)
    return seeds


def _write_ik_search_diagnostics(path: Path | None, records: list[dict[str, float | int | str]]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "index",
        "elapsed_s",
        "input_beam_count",
        "candidate_failure_count",
        "colliding_candidate_count",
        "next_beam_count",
        "best_bottleneck_step_deg",
        "best_step_sum_deg",
        "best_roll_sum_deg",
        "status",
        "detail",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _write_ik_candidate_bank(path: Path | None, records: list[dict[str, float | int | str]]) -> None:
    """Export collision-aware IK candidates for the offline RL selector."""

    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "waypoint",
        "candidate",
        "valid",
        "colliding",
        "max_joint_step_deg",
        "roll_deg",
        "roll_curve_deg",
        "roll_bias_deg",
        "tilt_x_deg",
        "tilt_y_deg",
        "reparameterization_window",
        "backtrack_depth",
        "collision_summary",
        "source",
        "position_error_mm",
        "normal_error_deg",
        "edge_prev_valid",
        "edge_prev_collision_free",
        "edge_prev_max_joint_step_deg",
        "reachable_in",
        "can_reach_end",
        "path_selected",
        *[f"q{i}" for i in range(1, 7)],
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def _write_ik_candidate_collection_report(path: Path | None, report: dict[str, object]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def collect_ik_candidate_bank_global_roll_backtracking(
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
    ik_candidate_bank_csv: Path,
    ik_candidate_collection_report_json: Path | None,
    joint_seeds: np.ndarray,
    global_roll_step_deg: float = 5.0,
    global_roll_bias_limit_deg: float = 45.0,
    global_tilt_step_deg: float = 0.5,
    global_tilt_limit_deg: float = 2.0,
    global_max_backtracks: int = 1200,
    global_reparameterization_window: int = 5,
) -> dict[str, object]:
    """Solve one complete path with global roll-curve optimization and DFS backtracking."""

    if len(joint_seeds) != len(poses) or len(normals) != len(poses):
        raise ValueError("Global roll search requires one seed and one normal for every TCP waypoint.")
    if not poses:
        raise ValueError("Global roll search requires at least one TCP pose.")

    state = RobotState(moveit.get_robot_model())
    psm = moveit.get_planning_scene_monitor()
    start_time = time.monotonic()
    reparam_window = max(1, int(global_reparameterization_window))
    # A control-point correction is linearly ramped over this many contour
    # samples.  The search state therefore stores a curve value, not a
    # pointwise pose jump.
    roll_step = np.deg2rad(max(0.25, float(global_roll_step_deg))) / reparam_window
    roll_limit = np.deg2rad(max(float(global_roll_bias_limit_deg), np.rad2deg(roll_step)))
    tilt_step = max(0.1, float(global_tilt_step_deg)) / reparam_window
    tilt_limit = max(tilt_step, float(global_tilt_limit_deg))
    max_backtracks = max(1, int(global_max_backtracks))
    base_roll_curve = _continuous_curve_step_limit(
        _continuous_tcp_roll_curve(poses), max(np.deg2rad(90.0), 4.0 * roll_step)
    )

    def evaluate(q: np.ndarray, target_pose: Pose, normal: np.ndarray) -> dict[str, object]:
        positions = np.asarray(q, dtype=float).copy()
        state.set_joint_group_positions(group_name, positions)
        state.update()
        transform = _transform_matrix(state.get_global_link_transform(ee_link))
        target = np.array([target_pose.position.x, target_pose.position.y, target_pose.position.z], dtype=float)
        position_error = float(np.linalg.norm(transform[:3, 3] - target))
        normal_error = float(
            np.rad2deg(np.arccos(np.clip(np.dot(transform[:3, 2], normal), -1.0, 1.0)))
        )
        with psm.read_only() as scene:
            colliding, collision_summary = _state_collision_summary(scene, state, group_name)
        return {
            "positions": positions,
            "position_error_mm": position_error * 1000.0,
            "normal_error_deg": normal_error,
            "colliding": int(colliding),
            "collision_summary": collision_summary,
            "node_valid": int(
                np.all(np.isfinite(positions))
                and position_error <= max_position_error
                and normal_error <= max_normal_error_deg
                and not colliding
            ),
        }

    def check_edge(previous: np.ndarray, current: np.ndarray) -> tuple[bool, float, str]:
        aligned = nearest_equivalent_joint_positions(np.asarray(current, dtype=float), previous)
        delta = aligned - previous
        max_delta = float(np.max(np.abs(delta)))
        if max_delta > max_joint_step:
            return False, max_delta, "joint_step_gate"
        sample_count = max(2, int(np.ceil(max_delta / np.deg2rad(1.0))) + 1)
        for fraction in np.linspace(0.0, 1.0, sample_count):
            state.set_joint_group_positions(group_name, previous + fraction * delta)
            state.update()
            with psm.read_only() as scene:
                colliding, _ = _state_collision_summary(scene, state, group_name)
            if colliding:
                return False, max_delta, "interpolated_collision"
        return True, max_delta, "safe"

    def proposals(previous: dict[str, object] | None) -> list[tuple[float, float, float]]:
        if previous is None:
            return [(0.0, 0.0, 0.0)]
        previous_roll = float(previous["roll_bias_rad"])
        previous_tx = float(previous["tilt_x_deg"])
        previous_ty = float(previous["tilt_y_deg"])
        roll_deltas = (0.0, -roll_step, roll_step, -2.0 * roll_step, 2.0 * roll_step)
        tilt_deltas = ((0.0, 0.0), (-tilt_step, 0.0), (tilt_step, 0.0), (0.0, -tilt_step), (0.0, tilt_step))
        q5_deg = abs(float(np.rad2deg(np.asarray(previous["q"], dtype=float)[4])))
        if abs(q5_deg - 90.0) <= 8.0:
            tilt_deltas = tilt_deltas + (
                (-2.0 * tilt_step, 0.0),
                (2.0 * tilt_step, 0.0),
                (0.0, -2.0 * tilt_step),
                (0.0, 2.0 * tilt_step),
            )
        result: list[tuple[float, float, float]] = []
        for delta_roll in roll_deltas:
            roll_bias = previous_roll + delta_roll
            if abs(roll_bias) > roll_limit + 1e-9:
                continue
            for delta_x, delta_y in tilt_deltas:
                tilt_x, tilt_y = previous_tx + delta_x, previous_ty + delta_y
                if max(abs(tilt_x), abs(tilt_y)) > tilt_limit + 1e-9:
                    continue
                item = (float(roll_bias), float(tilt_x), float(tilt_y))
                if item not in result:
                    result.append(item)
        # A later failure can reveal that the whole preceding section needs a
        # different roll branch.  These are global curve control-point values,
        # ordered from the current value outward; they are tried only after
        # local continuation proposals fail, and are revisited by backtracking.
        coarse_step = np.deg2rad(max(1.0, float(global_roll_step_deg)))
        absolute_biases = np.arange(-roll_limit, roll_limit + 0.5 * coarse_step, coarse_step)
        absolute_biases = sorted(absolute_biases.tolist(), key=lambda value: abs(float(value) - previous_roll))
        for absolute_bias in absolute_biases:
            for delta_x, delta_y in tilt_deltas[:1]:
                tilt_x, tilt_y = previous_tx + delta_x, previous_ty + delta_y
                if max(abs(tilt_x), abs(tilt_y)) > tilt_limit + 1e-9:
                    continue
                item = (float(absolute_bias), float(tilt_x), float(tilt_y))
                if item not in result:
                    result.append(item)
        return result

    def candidate_pose(index: int, roll_bias: float, tilt_x: float, tilt_y: float) -> Pose:
        return _global_roll_reparameterized_pose(
            poses[index], float(base_roll_curve[index]), roll_bias, tilt_x, tilt_y
        )

    path_nodes: list[dict[str, object]] = []
    failure_counts: list[dict[str, int]] = [
        {"ik": 0, "node": 0, "joint_step_gate": 0, "interpolated_collision": 0} for _ in poses
    ]
    visited_failures: set[tuple[int, tuple[float, ...], int, int, int]] = set()
    attempts = 0
    backtracks = 0
    first_failure: dict[str, object] | None = None

    first_eval = evaluate(np.asarray(joint_seeds[0], dtype=float), poses[0], normals[0])
    if int(first_eval["node_valid"]):
        path_nodes.append(
            {
                "q": np.asarray(joint_seeds[0], dtype=float).copy(),
                "roll_bias_rad": 0.0,
                "tilt_x_deg": 0.0,
                "tilt_y_deg": 0.0,
                "position_error_mm": float(first_eval["position_error_mm"]),
                "normal_error_deg": float(first_eval["normal_error_deg"]),
                "colliding": int(first_eval["colliding"]),
                "collision_summary": str(first_eval["collision_summary"]),
                "max_joint_step_deg": 0.0,
                "source": "verified_seed_anchor",
            }
        )
    else:
        first_failure = {"waypoint": 0, "reason": "verified_seed_invalid", **first_eval}

    index = 1
    while path_nodes and index < len(poses) and backtracks <= max_backtracks:
        previous = path_nodes[-1]
        found = False
        for proposal_index, (roll_bias, tilt_x, tilt_y) in enumerate(proposals(previous)):
            q_previous = np.asarray(previous["q"], dtype=float)
            key = (
                index,
                tuple(np.round(q_previous, 6)),
                int(round(roll_bias * 10000.0)),
                int(round(tilt_x * 1000.0)),
                int(round(tilt_y * 1000.0)),
            )
            if key in visited_failures:
                continue
            visited_failures.add(key)
            attempts += 1
            state.set_joint_group_positions(group_name, q_previous)
            state.update()
            if not state.set_from_ik(
                group_name, candidate_pose(index, roll_bias, tilt_x, tilt_y), ee_link, ik_timeout
            ):
                failure_counts[index]["ik"] += 1
                continue
            state.update()
            q_current = nearest_equivalent_joint_positions(
                np.asarray(state.get_joint_group_positions(group_name), dtype=float), q_previous
            )
            evaluated = evaluate(q_current, poses[index], normals[index])
            if not int(evaluated["node_valid"]):
                failure_counts[index]["node"] += 1
                continue
            safe, max_delta, reason = check_edge(q_previous, q_current)
            if not safe:
                failure_counts[index][reason] += 1
                continue
            path_nodes.append(
                {
                    "q": q_current.copy(),
                    "roll_bias_rad": roll_bias,
                    "tilt_x_deg": tilt_x,
                    "tilt_y_deg": tilt_y,
                    "position_error_mm": float(evaluated["position_error_mm"]),
                    "normal_error_deg": float(evaluated["normal_error_deg"]),
                    "colliding": int(evaluated["colliding"]),
                    "collision_summary": str(evaluated["collision_summary"]),
                    "max_joint_step_deg": float(np.rad2deg(max_delta)),
                    "proposal_index": proposal_index,
                    "attempt_index": attempts,
                }
            )
            index += 1
            found = True
            break
        if found:
            continue
        # The verified seed route is a safety anchor, not a larger candidate
        # bank.  If the optimized TCP frame was referenced differently from
        # the seed frame, retain that already validated configuration and carry
        # its measured roll as the next curve state.  This lets the global
        # optimizer recover its roll reference without accepting a joint jump.
        seed_q = np.asarray(joint_seeds[index], dtype=float)
        seed_eval = evaluate(seed_q, poses[index], normals[index])
        if int(seed_eval["node_valid"]):
            safe, max_delta, reason = check_edge(np.asarray(previous["q"], dtype=float), seed_q)
            if safe:
                path_nodes.append(
                    {
                        "q": nearest_equivalent_joint_positions(seed_q, np.asarray(previous["q"], dtype=float)),
                        "roll_bias_rad": float(previous["roll_bias_rad"]),
                        "tilt_x_deg": float(previous["tilt_x_deg"]),
                        "tilt_y_deg": float(previous["tilt_y_deg"]),
                        "position_error_mm": float(seed_eval["position_error_mm"]),
                        "normal_error_deg": float(seed_eval["normal_error_deg"]),
                        "colliding": int(seed_eval["colliding"]),
                        "collision_summary": str(seed_eval["collision_summary"]),
                        "max_joint_step_deg": float(np.rad2deg(max_delta)),
                        "source": "verified_seed_curve_anchor",
                    }
                )
                index += 1
                continue
            failure_counts[index][reason] += 1
        failed_index = index
        backtracks += 1
        if len(path_nodes) <= 1:
            first_failure = {
                "waypoint": failed_index,
                "reason": "no_safe_continuation_from_start",
                "failure_counts": failure_counts[min(failed_index, len(poses) - 1)],
            }
            break
        path_nodes.pop()
        index -= 1
        if first_failure is None:
            first_failure = {
                "waypoint": failed_index,
                "reason": "backtracking",
                "failure_counts": failure_counts[failed_index],
            }

    success = bool(path_nodes) and len(path_nodes) == len(poses)
    if not success and first_failure is None:
        first_failure = {
            "waypoint": min(index, len(poses) - 1),
            "reason": "backtrack_limit",
            "backtrack_count": backtracks,
        }
    records: list[dict[str, float | int | str]] = []
    roll_curve: list[float] = []
    for waypoint, node in enumerate(path_nodes):
        q = np.asarray(node["q"], dtype=float)
        roll_deg = float(np.rad2deg(base_roll_curve[waypoint] + float(node["roll_bias_rad"])))
        roll_curve.append(roll_deg)
        records.append(
            {
                "waypoint": waypoint,
                "candidate": 0,
                "valid": int(success),
                "colliding": int(node["colliding"]),
                "max_joint_step_deg": float(node.get("max_joint_step_deg", 0.0)),
                "roll_deg": roll_deg,
                "roll_curve_deg": roll_deg,
                "roll_bias_deg": float(np.rad2deg(float(node["roll_bias_rad"]))),
                "tilt_x_deg": float(node["tilt_x_deg"]),
                "tilt_y_deg": float(node["tilt_y_deg"]),
                "reparameterization_window": reparam_window,
                "backtrack_depth": 0,
                "collision_summary": str(node["collision_summary"]),
                "source": str(node.get("source", "global_roll_curve_backtracking")),
                "position_error_mm": float(node["position_error_mm"]),
                "normal_error_deg": float(node["normal_error_deg"]),
                "edge_prev_valid": int(waypoint == 0 or "max_joint_step_deg" in node),
                "edge_prev_collision_free": int(waypoint == 0 or "max_joint_step_deg" in node),
                "edge_prev_max_joint_step_deg": float(node.get("max_joint_step_deg", 0.0)),
                "reachable_in": int(success or waypoint == 0),
                "can_reach_end": int(success),
                "path_selected": int(success),
                **{f"q{i}": float(value) for i, value in enumerate(q, start=1)},
            }
        )
    _write_ik_candidate_bank(ik_candidate_bank_csv, records)
    roll_steps = np.abs(np.diff(roll_curve)) if len(roll_curve) > 1 else np.zeros(0)
    max_tilt = max(
        (max(abs(float(node["tilt_x_deg"])), abs(float(node["tilt_y_deg"]))) for node in path_nodes),
        default=0.0,
    )
    report: dict[str, object] = {
        "status": "pass" if success else "fail:no_complete_safe_path",
        "algorithm": "global_roll_curve_backtracking_tcp_reparameterization",
        "waypoint_count": len(poses),
        "solved_waypoint_count": len(path_nodes),
        "candidate_count_min": 1 if path_nodes else 0,
        "candidate_count_max": 1 if path_nodes else 0,
        "complete_path_candidate_count_min": 1 if success else 0,
        "complete_path_candidate_count_max": 1 if success else 0,
        "global_roll_step_deg": float(global_roll_step_deg),
        "global_roll_bias_limit_deg": float(global_roll_bias_limit_deg),
        "global_roll_curve_max_step_deg": float(np.max(roll_steps)) if len(roll_steps) else 0.0,
        "global_roll_curve_total_unwrapped_deg": float(roll_curve[-1] - roll_curve[0]) if roll_curve else 0.0,
        "tcp_reparameterization_max_tilt_deg": float(max_tilt),
        "tcp_reparameterization_window": reparam_window,
        "backtrack_count": backtracks,
        "max_backtracks": max_backtracks,
        "ik_attempt_count": attempts,
        "first_failure": first_failure,
        "failure_counts_by_waypoint": failure_counts,
        "candidate_bank_csv": str(ik_candidate_bank_csv),
        "interpretation": (
            "One globally transported roll curve is optimized with bounded TCP tangent-plane reparameterization. "
            "Backtracking revisits prior curve decisions when a later hard MoveIt edge gate blocks continuation; "
            "candidate count is not the recovery mechanism."
        ),
        "elapsed_s": time.monotonic() - start_time,
    }
    _write_ik_candidate_collection_report(ik_candidate_collection_report_json, report)
    return report


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
    ik_roll_sample_count: int = 1,
    ik_beam_width: int = 1,
    ik_candidates_per_beam: int = 8,
    ik_beam_diversity_joint_deg: float = 0.0,
    ik_search_diagnostics_csv: Path | None = None,
    ik_candidate_bank_csv: Path | None = None,
    ik_max_solve_seconds: float = 0.0,
    joint_seeds: np.ndarray | None = None,
):
    """Solve every contour pose with MoveIt2 IK, retaining optional branch beams."""

    state = RobotState(moveit.get_robot_model())
    result = RobotTrajectoryMsg()
    result.joint_trajectory.joint_names = joint_names
    failures: list[str] = []
    target_tcp = np.array([[pose.position.x, pose.position.y, pose.position.z] for pose in poses], dtype=float)
    loop_size = _repeated_closed_loop_size(target_tcp, normals)
    solve_poses = poses[:loop_size]
    solve_normals = normals[:loop_size]
    solve_joint_seeds = joint_seeds[:loop_size] if joint_seeds is not None else None
    psm = moveit.get_planning_scene_monitor()
    verbose_collision_emitted = False
    beam_width = max(1, int(ik_beam_width))
    candidates_per_beam = max(1, int(ik_candidates_per_beam))
    beam_diversity_joint = max(0.0, np.deg2rad(float(ik_beam_diversity_joint_deg)))
    max_solve_seconds = max(0.0, float(ik_max_solve_seconds))
    solve_start = time.monotonic()
    diagnostic_records: list[dict[str, float | int | str]] = []
    candidate_records: list[dict[str, float | int | str]] = []
    beams: list[tuple[float, float, float, list[np.ndarray]]] = [(0.0, 0.0, 0.0, [])]
    for index, pose in enumerate(solve_poses):
        elapsed_s = time.monotonic() - solve_start
        if max_solve_seconds > 0.0 and elapsed_s > max_solve_seconds:
            diagnostic_records.append(
                {
                    "index": index,
                    "elapsed_s": elapsed_s,
                    "input_beam_count": len(beams),
                    "candidate_failure_count": 0,
                    "colliding_candidate_count": 0,
                    "next_beam_count": 0,
                    "best_bottleneck_step_deg": "",
                    "best_step_sum_deg": "",
                    "best_roll_sum_deg": "",
                    "status": "timeout",
                    "detail": f"ik_max_solve_seconds={max_solve_seconds}",
                }
            )
            _write_ik_search_diagnostics(ik_search_diagnostics_csv, diagnostic_records)
            raise RuntimeError(
                f"MoveIt2 IK exceeded ik_max_solve_seconds={max_solve_seconds:.3f} before index={index}."
            )
        candidate_failures: list[str] = []
        next_beams: list[tuple[float, float, float, list[np.ndarray]]] = []
        colliding_candidates: list[tuple[float, float, str, np.ndarray]] = []
        candidate_rank = 0
        input_beam_count = len(beams)
        for beam_bottleneck_cost, beam_step_sum_cost, beam_roll_cost, beam_path in beams:
            previous_positions = beam_path[-1] if beam_path else None
            if solve_joint_seeds is not None:
                primary_seed = np.asarray(solve_joint_seeds[index], dtype=float)
            elif previous_positions is not None:
                primary_seed = previous_positions.copy()
            else:
                primary_seed = np.zeros(len(joint_names), dtype=float)
            valid_candidates: list[tuple[bool, float, float, str, np.ndarray]] = []
            for roll, candidate_pose in _tool_axis_roll_pose_candidates(pose, ik_roll_sample_count):
                for seed_index, seed in enumerate(_ik_seed_candidates(primary_seed, previous_positions)):
                    state.set_joint_group_positions(group_name, seed)
                    state.update()
                    if not state.set_from_ik(group_name, candidate_pose, ee_link, ik_timeout):
                        candidate_failures.append(f"roll={np.rad2deg(roll):.1f}deg,seed={seed_index}:no_ik")
                        continue
                    state.update()
                    transform = _transform_matrix(state.get_global_link_transform(ee_link))
                    target = np.array([pose.position.x, pose.position.y, pose.position.z], dtype=float)
                    position_error = float(np.linalg.norm(transform[:3, 3] - target))
                    tool_z = transform[:3, 2]
                    normal_error = float(
                        np.rad2deg(np.arccos(np.clip(np.dot(tool_z, solve_normals[index]), -1.0, 1.0)))
                    )
                    if position_error > max_position_error or normal_error > max_normal_error_deg:
                        candidate_failures.append(
                            f"roll={np.rad2deg(roll):.1f}deg,seed={seed_index}:"
                            f"position_error={position_error * 1000.0:.3f}mm,"
                            f"normal_error={normal_error:.3f}deg"
                        )
                        continue
                    positions = np.asarray(state.get_joint_group_positions(group_name), dtype=float)
                    if previous_positions is not None:
                        positions = nearest_equivalent_joint_positions(positions, previous_positions)
                        joint_delta = np.abs(positions - previous_positions)
                        max_delta = float(np.max(joint_delta))
                    else:
                        max_delta = 0.0
                    state.set_joint_group_positions(group_name, positions)
                    state.update()
                    with psm.read_only() as scene:
                        colliding, collision_summary = _state_collision_summary(scene, state, group_name)
                    if max_delta > max_joint_step:
                        joint_index = int(np.argmax(np.abs(positions - previous_positions)))
                        candidate_failures.append(
                            f"roll={np.rad2deg(roll):.1f}deg,seed={seed_index}:joint={joint_names[joint_index]},"
                            f"joint_step={np.rad2deg(max_delta):.2f}deg"
                        )
                        continue
                    valid_candidates.append((colliding, max_delta, abs(float(roll)), collision_summary, positions.copy()))
            valid_candidates.sort(key=lambda item: (item[0], item[1], item[2]))
            for colliding, max_delta, roll_cost, collision_summary, positions in valid_candidates[:candidates_per_beam]:
                candidate_records.append(
                    {
                        "waypoint": index,
                        "candidate": candidate_rank,
                        "valid": int(not colliding),
                        "colliding": int(colliding),
                        "max_joint_step_deg": float(np.rad2deg(max_delta)),
                        "roll_deg": float(np.rad2deg(roll_cost)),
                        "collision_summary": collision_summary,
                        **{f"q{i}": float(value) for i, value in enumerate(positions, start=1)},
                    }
                )
                candidate_rank += 1
                if colliding:
                    colliding_candidates.append((max_delta, roll_cost, collision_summary, positions))
                    continue
                next_bottleneck_cost = max(beam_bottleneck_cost, max_delta)
                next_step_sum_cost = beam_step_sum_cost + max_delta
                next_roll_cost = beam_roll_cost + roll_cost
                next_beams.append(
                    (next_bottleneck_cost, next_step_sum_cost, next_roll_cost, beam_path + [positions.copy()])
                )
        next_beams.sort(key=lambda item: (item[0], item[1], item[2]))
        beams = _select_diverse_ik_beams(next_beams, beam_width, beam_diversity_joint)
        if beams:
            diagnostic_records.append(
                {
                    "index": index,
                    "elapsed_s": time.monotonic() - solve_start,
                    "input_beam_count": input_beam_count,
                    "candidate_failure_count": len(candidate_failures),
                    "colliding_candidate_count": len(colliding_candidates),
                    "next_beam_count": len(beams),
                    "best_bottleneck_step_deg": float(np.rad2deg(beams[0][0])),
                    "best_step_sum_deg": float(np.rad2deg(beams[0][1])),
                    "best_roll_sum_deg": float(np.rad2deg(beams[0][2])),
                    "status": "ok",
                    "detail": "",
                }
            )
        if not beams:
            normal = solve_normals[index]
            if colliding_candidates:
                colliding_candidates.sort(key=lambda item: (item[0], item[1]))
                _, _, collision_summary, positions = colliding_candidates[0]
                if not verbose_collision_emitted:
                    state.set_joint_group_positions(group_name, positions)
                    state.update()
                    with psm.read_only() as scene:
                        scene.is_state_colliding(state, group_name, True)
                    verbose_collision_emitted = True
                detail = f"all IK candidates are colliding, best={collision_summary}"
                suffix = ""
            else:
                detail = "; ".join(candidate_failures[:4])
                suffix = "" if len(candidate_failures) <= 4 else f"; ... {len(candidate_failures) - 4} more"
            failures.append(
                f"index={index}, tcp=({pose.position.x:.6f},{pose.position.y:.6f},{pose.position.z:.6f}), "
                f"normal=({normal[0]:.6f},{normal[1]:.6f},{normal[2]:.6f}), {detail}{suffix}"
            )
            diagnostic_records.append(
                {
                    "index": index,
                    "elapsed_s": time.monotonic() - solve_start,
                    "input_beam_count": input_beam_count,
                    "candidate_failure_count": len(candidate_failures),
                    "colliding_candidate_count": len(colliding_candidates),
                    "next_beam_count": 0,
                    "best_bottleneck_step_deg": "",
                    "best_step_sum_deg": "",
                    "best_roll_sum_deg": "",
                    "status": "fail",
                    "detail": detail,
                }
            )
            break
    if not failures:
        best_path = beams[0][3]
        for positions in best_path:
            point = JointTrajectoryPoint()
            point.positions = [float(value) for value in positions]
            point.time_from_start = _duration_from_seconds(0.0)
            result.joint_trajectory.points.append(point)
    if loop_size < len(poses) and not failures:
        first_loop_positions = [
            np.asarray(point.positions, dtype=float)
            for point in result.joint_trajectory.points
        ]
        next_start = nearest_equivalent_joint_positions(first_loop_positions[0], first_loop_positions[-1])
        closure_delta = np.abs(next_start - first_loop_positions[-1])
        max_closure_delta = float(np.max(closure_delta))
        if max_closure_delta > max_joint_step:
            joint_index = int(np.argmax(closure_delta))
            failures.append(
                f"closed_loop_reuse, joint={joint_names[joint_index]}, "
                f"joint_step={np.rad2deg(max_closure_delta):.2f} deg"
            )
        else:
            tiled_points = []
            repeat_count = len(poses) // loop_size
            loop_offset = np.zeros(len(joint_names), dtype=float)
            previous_last = first_loop_positions[-1]
            for loop_index in range(repeat_count):
                if loop_index > 0:
                    loop_start = nearest_equivalent_joint_positions(first_loop_positions[0], previous_last)
                    loop_offset = loop_start - first_loop_positions[0]
                for source_index, point in enumerate(result.joint_trajectory.points):
                    tiled = deepcopy(point)
                    tiled.positions = [float(value) for value in first_loop_positions[source_index] + loop_offset]
                    tiled.time_from_start = _duration_from_seconds(0.0)
                    tiled_points.append(tiled)
                previous_last = first_loop_positions[-1] + loop_offset
            result.joint_trajectory.points = tiled_points
    if failures:
        _write_ik_search_diagnostics(ik_search_diagnostics_csv, diagnostic_records)
        preview = "; ".join(failures[:5])
        extra = "" if len(failures) <= 5 else f"; ... {len(failures) - 5} more"
        raise RuntimeError(f"MoveIt2 IK failed the closed contour gate: {preview}{extra}")
    if len(result.joint_trajectory.points) < 2:
        _write_ik_search_diagnostics(ik_search_diagnostics_csv, diagnostic_records)
        raise RuntimeError("MoveIt2 IK produced fewer than two contour waypoints.")
    _write_ik_search_diagnostics(ik_search_diagnostics_csv, diagnostic_records)
    _write_ik_candidate_bank(ik_candidate_bank_csv, candidate_records)
    return result


def collect_ik_candidate_bank(
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
    ik_roll_sample_count: int,
    ik_candidates_per_waypoint: int,
    ik_candidate_bank_csv: Path,
    ik_candidate_collection_report_json: Path | None,
    joint_seeds: np.ndarray,
) -> dict[str, object]:
    """Collect safe IK nodes first, then validate the candidate transition graph.

    This is intentionally separate from the production single-path solver. A
    failed branch must not abort collection of later candidates; only after all
    waypoints are sampled do we apply the adjacent joint-step and interpolated
    collision gates and mark candidates that belong to a complete safe path.
    """

    if len(joint_seeds) != len(poses):
        raise ValueError("Candidate collection requires one verified seed joint row per TCP waypoint.")
    state = RobotState(moveit.get_robot_model())
    psm = moveit.get_planning_scene_monitor()
    candidate_limit = max(1, int(ik_candidates_per_waypoint))
    candidate_sets: list[list[dict[str, object]]] = []
    failure_counts: list[int] = []
    baseline_failures: list[dict[str, object]] = []
    solve_start = time.monotonic()

    def evaluate(positions: np.ndarray, target_pose: Pose, normal: np.ndarray) -> dict[str, object]:
        q = np.asarray(positions, dtype=float).copy()
        state.set_joint_group_positions(group_name, q)
        state.update()
        transform = _transform_matrix(state.get_global_link_transform(ee_link))
        target = np.array([target_pose.position.x, target_pose.position.y, target_pose.position.z], dtype=float)
        position_error = float(np.linalg.norm(transform[:3, 3] - target))
        tool_z = transform[:3, 2]
        normal_error = float(np.rad2deg(np.arccos(np.clip(np.dot(tool_z, normal), -1.0, 1.0))))
        with psm.read_only() as scene:
            colliding, collision_summary = _state_collision_summary(scene, state, group_name)
        node_valid = (
            np.all(np.isfinite(q))
            and position_error <= max_position_error
            and normal_error <= max_normal_error_deg
            and not colliding
        )
        return {
            "positions": q,
            "position_error_mm": position_error * 1000.0,
            "normal_error_deg": normal_error,
            "colliding": int(colliding),
            "collision_summary": collision_summary,
            "node_valid": int(node_valid),
        }

    for index, pose in enumerate(poses):
        baseline = np.asarray(joint_seeds[index], dtype=float)
        candidates: list[dict[str, object]] = []
        failures = 0

        # The seed trajectory has already passed the complete MoveIt collision
        # gate, so retain it as the reference candidate and never replace it
        # with an IK solver's arbitrary equivalent branch.
        baseline_eval = evaluate(baseline, pose, normals[index])
        if int(baseline_eval["node_valid"]):
            baseline_eval["source"] = "verified_seed"
            candidates.append(baseline_eval)
        else:
            failures += 1
            if len(baseline_failures) < 20:
                baseline_failures.append(
                    {
                        "waypoint": index,
                        "position_error_mm": float(baseline_eval["position_error_mm"]),
                        "normal_error_deg": float(baseline_eval["normal_error_deg"]),
                        "colliding": int(baseline_eval["colliding"]),
                        "collision_summary": str(baseline_eval["collision_summary"]),
                    }
                )

        # Reuse the previous waypoint candidates first.  The verified seed
        # trajectory is a collision-safe reference, but it is not guaranteed
        # to be TCP-exact at every waypoint; placing it first can make KDL
        # repeatedly return a locally valid yet globally disconnected branch.
        # Previous candidates therefore have priority for continuity, while
        # the baseline and its equivalent seeds remain fallbacks.
        previous_candidates: list[dict[str, object]] = []
        if candidate_sets:
            previous_candidates = list(candidate_sets[-1][: min(candidate_limit, 16)])
            if len(candidate_sets) >= 2:
                reference_candidates = candidate_sets[-2]

                def continuity_score(item: dict[str, object]) -> float:
                    positions = np.asarray(item["positions"], dtype=float)
                    return min(
                        float(np.max(np.abs(circular_joint_delta(positions - np.asarray(reference["positions"])))))
                        for reference in reference_candidates
                    )

                previous_candidates.sort(key=continuity_score)
        seed_requests: list[np.ndarray] = []
        if previous_candidates:
            for previous_candidate in previous_candidates:
                previous_seed = np.asarray(previous_candidate["positions"], dtype=float)
                if not any(np.allclose(previous_seed, existing, atol=1e-9, rtol=0.0) for existing in seed_requests):
                    seed_requests.append(previous_seed)
        for seed in _candidate_collection_seed_candidates(baseline):
            if not any(np.allclose(seed, existing, atol=1e-9, rtol=0.0) for existing in seed_requests):
                seed_requests.append(seed)

        roll_requests: list[tuple[float, Pose]] = []
        prioritized_rolls_by_seed: list[tuple[np.ndarray, list[float]]] = []
        prioritized_tilt_by_seed: list[tuple[np.ndarray, list[float]]] = []

        def add_roll_request(roll_value: float) -> None:
            normalized = float((roll_value + np.pi) % (2.0 * np.pi) - np.pi)
            if any(abs(float((normalized - existing[0] + np.pi) % (2.0 * np.pi) - np.pi)) < 1e-6 for existing in roll_requests):
                return
            base_matrix = _pose_to_matrix(pose)
            c, s = float(np.cos(normalized)), float(np.sin(normalized))
            rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
            rolled = base_matrix.copy()
            rolled[:3, :3] = base_matrix[:3, :3] @ rz
            roll_requests.append((normalized, _pose_from_matrix(rolled, pose)))

        # Try the unmodified wall-normal pose first with the previous waypoint
        # as the IK seed.  This is the primary continuity path: changing the
        # tool roll is optional, but a wrist branch flip at a singular-looking
        # section is never acceptable merely because a rolled pose solved.
        add_roll_request(0.0)

        # First follow the roll implied by the previous candidate's tool X
        # axis, then keep a small local neighbourhood around it. This is the
        # continuous-wrist branch that a fixed 0/20/40 degree grid can miss.
        if candidate_sets:
            current_matrix = _pose_to_matrix(pose)
            current_x = current_matrix[:3, 0]
            current_y = current_matrix[:3, 1]
            for previous_candidate in previous_candidates:
                previous_seed = np.asarray(previous_candidate["positions"], dtype=float).copy()
                state.set_joint_group_positions(group_name, previous_candidate["positions"])
                state.update()
                previous_matrix = _transform_matrix(state.get_global_link_transform(ee_link))
                previous_x = previous_matrix[:3, 0]
                center = float(np.arctan2(np.dot(previous_x, current_y), np.dot(previous_x, current_x)))
                local_rolls: list[float] = []
                for offset_deg in (
                    0.0,
                    -5.0,
                    5.0,
                    -10.0,
                    10.0,
                    -15.0,
                    15.0,
                    -20.0,
                    20.0,
                    *range(-180, 181, 10),
                ):
                    requested = center + np.deg2rad(offset_deg)
                    add_roll_request(requested)
                    local_rolls.append(float((requested + np.pi) % (2.0 * np.pi) - np.pi))
                prioritized_rolls_by_seed.append((previous_seed, local_rolls))
                previous_q5_deg = abs(float(np.rad2deg(previous_seed[4])))
                if index >= 140 and abs(previous_q5_deg - 90.0) <= 6.0:
                    prioritized_tilt_by_seed.append((previous_seed, local_rolls[:9]))
        for roll, candidate_pose in _tool_axis_roll_pose_candidates(pose, ik_roll_sample_count):
            add_roll_request(roll)

        def pose_for_roll(roll_value: float) -> tuple[float, Pose] | None:
            for existing_roll, existing_pose in roll_requests:
                if abs(float((roll_value - existing_roll + np.pi) % (2.0 * np.pi) - np.pi)) < 1e-6:
                    return existing_roll, existing_pose
            return None

        def pose_with_tilt(roll_value: float, tilt_x_deg: float, tilt_y_deg: float) -> Pose:
            base_matrix = _pose_to_matrix(pose)
            ax = np.deg2rad(float(tilt_x_deg))
            ay = np.deg2rad(float(tilt_y_deg))
            ar = float(roll_value)
            cx, sx = float(np.cos(ax)), float(np.sin(ax))
            cy, sy = float(np.cos(ay)), float(np.sin(ay))
            cr, sr = float(np.cos(ar)), float(np.sin(ar))
            rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=float)
            ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=float)
            rz = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]], dtype=float)
            tilted = base_matrix.copy()
            tilted[:3, :3] = base_matrix[:3, :3] @ rx @ ry @ rz
            return _pose_from_matrix(tilted, pose)

        # Solve each previous branch through its own local roll neighbourhood
        # before falling back to the Cartesian roll grid. This prevents the
        # first branch from consuming the candidate quota and hiding a second
        # branch that is needed to cross the upper-arch singular section.
        ik_requests: list[tuple[np.ndarray, float, Pose, float, float]] = []
        request_keys: set[tuple[tuple[float, ...], int, int, int]] = set()

        def append_request(seed: np.ndarray, roll_value: float, tilt_x_deg: float = 0.0, tilt_y_deg: float = 0.0) -> None:
            resolved = pose_for_roll(roll_value)
            if resolved is None and abs(tilt_x_deg) < 1e-9 and abs(tilt_y_deg) < 1e-9:
                return
            actual_roll = float((roll_value + np.pi) % (2.0 * np.pi) - np.pi) if resolved is None else resolved[0]
            candidate_pose = pose_with_tilt(actual_roll, tilt_x_deg, tilt_y_deg) if abs(tilt_x_deg) >= 1e-9 or abs(tilt_y_deg) >= 1e-9 else resolved[1]
            key = (
                tuple(np.round(np.asarray(seed, dtype=float), 10)),
                int(round(actual_roll * 1e6)),
                int(round(tilt_x_deg * 1000.0)),
                int(round(tilt_y_deg * 1000.0)),
            )
            if key in request_keys:
                return
            request_keys.add(key)
            ik_requests.append((np.asarray(seed, dtype=float).copy(), actual_roll, candidate_pose, tilt_x_deg, tilt_y_deg))

        # Near j5=+/-90 degrees, a tiny tangent-plane tilt can move the IK
        # solver away from the wrist singularity without changing TCP
        # position. The normal-error gate below remains active, so this is a
        # bounded pose adjustment rather than permission to point away from
        # the tunnel wall.
        tilt_pairs = ((-2.0, 0.0), (2.0, 0.0), (0.0, -2.0), (0.0, 2.0), (-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0))
        for seed, local_rolls in prioritized_tilt_by_seed[:2]:
            for local_roll in local_rolls:
                for tilt_x_deg, tilt_y_deg in tilt_pairs:
                    append_request(seed, local_roll, tilt_x_deg, tilt_y_deg)

        for seed, local_rolls in prioritized_rolls_by_seed:
            for local_roll in local_rolls:
                append_request(seed, local_roll)
        for seed in seed_requests:
            for roll, _candidate_pose in roll_requests:
                append_request(seed, roll)

        for seed, roll, candidate_pose, tilt_x_deg, tilt_y_deg in ik_requests:
                state.set_joint_group_positions(group_name, seed)
                state.update()
                if not state.set_from_ik(group_name, candidate_pose, ee_link, ik_timeout):
                    failures += 1
                    continue
                state.update()
                positions = nearest_equivalent_joint_positions(
                    np.asarray(state.get_joint_group_positions(group_name), dtype=float), baseline
                )
                evaluated = evaluate(positions, pose, normals[index])
                if not int(evaluated["node_valid"]):
                    failures += 1
                    continue
                if any(
                    np.max(np.abs(circular_joint_delta(positions - np.asarray(existing["positions"])))) < 1e-5
                    for existing in candidates
                ):
                    continue
                evaluated["source"] = f"ik_roll_{np.rad2deg(roll):.1f}deg"
                evaluated["roll_deg"] = float(np.rad2deg(roll))
                evaluated["tilt_x_deg"] = float(tilt_x_deg)
                evaluated["tilt_y_deg"] = float(tilt_y_deg)
                candidates.append(evaluated)
                if len(candidates) >= candidate_limit:
                    break

        candidate_sets.append(candidates)
        failure_counts.append(failures)

    def check_edge(previous: np.ndarray, current: np.ndarray) -> tuple[bool, float, str]:
        aligned = nearest_equivalent_joint_positions(current, previous)
        delta = aligned - previous
        max_delta = float(np.max(np.abs(delta)))
        if max_delta > max_joint_step:
            return False, max_delta, "joint_step_gate"
        sample_count = max(2, int(np.ceil(max_delta / np.deg2rad(1.0))) + 1)
        for fraction in np.linspace(0.0, 1.0, sample_count):
            q = previous + fraction * delta
            state.set_joint_group_positions(group_name, q)
            state.update()
            with psm.read_only() as scene:
                colliding, _ = _state_collision_summary(scene, state, group_name)
            if colliding:
                return False, max_delta, "interpolated_collision"
        return True, max_delta, "safe"

    edge_info: list[list[list[tuple[int, float]]]] = [[]]
    reachable: list[list[bool]] = []
    predecessor: list[list[int | None]] = []
    path_cost: list[list[float]] = []
    transition_stats: list[dict[str, object]] = []
    if candidate_sets:
        first_count = len(candidate_sets[0])
        reachable.append([True] * first_count)
        predecessor.append([None] * first_count)
        path_cost.append([0.0] * first_count)
    for index in range(1, len(candidate_sets)):
        current_edges: list[list[tuple[int, float]]] = []
        current_reachable: list[bool] = []
        current_predecessor: list[int | None] = []
        current_cost: list[float] = []
        safe_edge_count = 0
        step_gate_count = 0
        interpolated_collision_count = 0
        min_pair_step = float("inf")
        min_pair_step_collision = float("inf")
        for current in candidate_sets[index]:
            edges: list[tuple[int, float]] = []
            best_prev: int | None = None
            best_cost = float("inf")
            for previous_index, previous in enumerate(candidate_sets[index - 1]):
                safe, max_delta, reason = check_edge(
                    np.asarray(previous["positions"], dtype=float),
                    np.asarray(current["positions"], dtype=float),
                )
                min_pair_step = min(min_pair_step, max_delta)
                if reason == "joint_step_gate":
                    step_gate_count += 1
                elif reason == "interpolated_collision":
                    interpolated_collision_count += 1
                    min_pair_step_collision = min(min_pair_step_collision, max_delta)
                if safe:
                    safe_edge_count += 1
                    edges.append((previous_index, max_delta))
                    if reachable[index - 1][previous_index] and path_cost[index - 1][previous_index] + max_delta < best_cost:
                        best_prev = previous_index
                        best_cost = path_cost[index - 1][previous_index] + max_delta
            current_edges.append(edges)
            current_reachable.append(best_prev is not None)
            current_predecessor.append(best_prev)
            current_cost.append(best_cost)
        edge_info.append(current_edges)
        reachable.append(current_reachable)
        predecessor.append(current_predecessor)
        path_cost.append(current_cost)
        transition_stats.append(
            {
                "from_waypoint": index - 1,
                "to_waypoint": index,
                "possible_pairs": len(candidate_sets[index - 1]) * len(candidate_sets[index]),
                "safe_edges": safe_edge_count,
                "joint_step_gate_rejections": step_gate_count,
                "interpolated_collision_rejections": interpolated_collision_count,
                "min_pair_step_deg": float(np.rad2deg(min_pair_step)) if np.isfinite(min_pair_step) else None,
                "min_collision_rejection_step_deg": (
                    float(np.rad2deg(min_pair_step_collision)) if np.isfinite(min_pair_step_collision) else None
                ),
                "reachable_current_count": sum(current_reachable),
            }
        )

    can_reach_end: list[list[bool]] = [[False] * len(items) for items in candidate_sets]
    if candidate_sets:
        can_reach_end[-1] = list(reachable[-1])
    for index in range(len(candidate_sets) - 2, -1, -1):
        for current_index in range(len(candidate_sets[index])):
            can_reach_end[index][current_index] = any(
                any(previous_index == current_index for previous_index, _ in edge_info[index + 1][next_index])
                and can_reach_end[index + 1][next_index]
                for next_index in range(len(candidate_sets[index + 1]))
            )

    selected_path: set[tuple[int, int]] = set()
    if candidate_sets and any(reachable[-1]):
        current_index = int(np.argmin([cost if ok else float("inf") for cost, ok in zip(path_cost[-1], reachable[-1])]))
        for index in range(len(candidate_sets) - 1, -1, -1):
            selected_path.add((index, current_index))
            previous_index = predecessor[index][current_index]
            if previous_index is None:
                break
            current_index = previous_index

    records: list[dict[str, float | int | str]] = []
    for index, candidates in enumerate(candidate_sets):
        for candidate_index, candidate in enumerate(candidates):
            edges = edge_info[index][candidate_index] if index > 0 else []
            best_edge = min((max_delta for _, max_delta in edges), default=0.0)
            edge_valid = int(bool(edges)) if index > 0 else 1
            complete_path = int(can_reach_end[index][candidate_index] and reachable[index][candidate_index])
            positions = np.asarray(candidate["positions"], dtype=float)
            records.append(
                {
                    "waypoint": index,
                    "candidate": candidate_index,
                    "valid": complete_path,
                    "colliding": int(candidate["colliding"]),
                    "max_joint_step_deg": float(np.rad2deg(best_edge)),
                    "roll_deg": float(candidate.get("roll_deg", 0.0)),
                    "collision_summary": str(candidate["collision_summary"]),
                    "source": str(candidate["source"]),
                    "position_error_mm": float(candidate["position_error_mm"]),
                    "normal_error_deg": float(candidate["normal_error_deg"]),
                    "edge_prev_valid": edge_valid,
                    "edge_prev_collision_free": edge_valid,
                    "edge_prev_max_joint_step_deg": float(np.rad2deg(best_edge)),
                    "reachable_in": int(reachable[index][candidate_index]),
                    "can_reach_end": int(can_reach_end[index][candidate_index]),
                    "path_selected": int((index, candidate_index) in selected_path),
                    **{f"q{i}": float(value) for i, value in enumerate(positions, start=1)},
                }
            )
    _write_ik_candidate_bank(ik_candidate_bank_csv, records)
    candidate_counts = [len(items) for items in candidate_sets]
    valid_counts = [sum(int(row["valid"]) for row in records if int(row["waypoint"]) == index) for index in range(len(poses))]
    first_disconnected_transition = next(
        (item for item in transition_stats if int(item["safe_edges"]) == 0),
        None,
    )
    first_reachability_break = next(
        (item for item in transition_stats if int(item["reachable_current_count"]) == 0),
        None,
    )
    report: dict[str, object] = {
        "status": "pass" if candidate_sets and any(reachable[-1]) else "fail:no_complete_safe_path",
        "waypoint_count": len(poses),
        "candidate_limit": candidate_limit,
        "candidate_count_min": min(candidate_counts, default=0),
        "candidate_count_max": max(candidate_counts, default=0),
        "candidate_count_mean": float(np.mean(candidate_counts)) if candidate_counts else 0.0,
        "complete_path_candidate_count_min": min(valid_counts, default=0),
        "complete_path_candidate_count_max": max(valid_counts, default=0),
        "waypoints_with_multiple_complete_candidates": sum(count >= 2 for count in valid_counts),
        "waypoints_with_zero_complete_candidates": sum(count == 0 for count in valid_counts),
        "first_disconnected_transition": first_disconnected_transition,
        "first_reachability_break": first_reachability_break,
        "transition_stats_first_20": transition_stats[:20],
        "max_joint_step_deg_gate": float(np.rad2deg(max_joint_step)),
        "elapsed_s": time.monotonic() - solve_start,
        "candidate_failure_counts_first_10": failure_counts[:10],
        "baseline_failures_first_20": baseline_failures,
        "candidate_bank_csv": str(ik_candidate_bank_csv),
        "interpretation": "valid=1 means the node belongs to at least one complete collision-free adjacent path; path_selected=1 marks the lowest-cost baseline-compatible path.",
    }
    _write_ik_candidate_collection_report(ik_candidate_collection_report_json, report)
    return report


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


def build_p2b1_process_task_dls_trajectory(
    moveit,
    group_name: str,
    ee_link: str,
    target_poses: list[Pose],
    target_normals: np.ndarray,
    joint_seeds: np.ndarray,
    joint_names: list[str],
    *,
    variant: str,
    warm_start_rows: int = 16,
    iterations_per_waypoint: int = 80,
    damping: float = 1.0e-2,
    position_weight: float = 100.0,
    normal_weight: float = 0.1,
    max_joint_step_rad: float = 0.16,
    position_tolerance_m: float = 2.0e-5,
    normal_tolerance_rad: float = 2.0e-4,
    secondary_objective_name: str = "joint_centering",
) -> RobotTrajectoryMsg:
    """P2-B1's exact five-dimensional position-plus-direction task solver."""
    variant = str(variant).upper()
    if variant not in {"B0", "B1"}:
        raise ValueError("P2B1_process_solver_variant_must_be_B0_or_B1")
    p2b2_variant = os.environ.get("P2B2_VARIANT", "").strip().upper()
    if p2b2_variant:
        expected = {
            "R0": ("B0", 16, "none"),
            "R1": ("B1", 16, "joint_centering"),
            "R2": ("B1", 1, "joint_centering"),
            "R3": ("B1", 1, "joint_limit_barrier"),
        }
        if p2b2_variant not in expected:
            raise ValueError("unsupported_P2B2_VARIANT")
        expected_solver, expected_warm_rows, expected_objective = expected[p2b2_variant]
        configured_warm_rows = int(os.environ.get("P2B2_WARM_START_ROWS", str(expected_warm_rows)))
        configured_objective = os.environ.get("P2B2_SECONDARY_OBJECTIVE", expected_objective).strip()
        if variant != expected_solver:
            raise ValueError("P2B2_variant_solver_mismatch")
        if configured_warm_rows != expected_warm_rows or configured_objective != expected_objective:
            raise ValueError("P2B2_frozen_design_policy_mismatch")
        warm_start_rows = configured_warm_rows
        secondary_objective_name = configured_objective
    if secondary_objective_name not in {"joint_centering", "joint_limit_barrier", "none"}:
        raise ValueError("unsupported_secondary_objective")
    if len(target_poses) != len(target_normals) or len(target_poses) != len(joint_seeds):
        raise ValueError("p2b1_target_seed_count_mismatch")
    if len(target_poses) < 2 or len(joint_names) != 6:
        raise ValueError("p2b1_invalid_trajectory_shape")
    if warm_start_rows < 1 or warm_start_rows >= len(target_poses):
        raise ValueError("p2b1_invalid_warm_start_rows")
    pressure_path = os.environ.get("P2B1_SOLVER_PRESSURE_CSV", "").strip()
    if not pressure_path:
        raise RuntimeError("P2B1_SOLVER_PRESSURE_CSV_required_for_candidate_solver")
    if variant in {"B0", "B1"}:
        frozen_q_path = os.environ.get("P2B1_FROZEN_D41_Q_CSV", "").strip()
        fd_json_path = os.environ.get("P2B1_FD_CONSISTENCY_JSON", "").strip()
        if not frozen_q_path or not fd_json_path:
            raise RuntimeError("P2B1_process_solver_requires_frozen_D41_FD_inputs_and_output")
        validate_p2b1_moveit_fd_consistency(
            moveit, group_name, ee_link, target_poses, target_normals, frozen_q_path, fd_json_path,
        )
    case_id = os.environ.get("D41_DLS_CASE_ID", "unspecified").strip() or "unspecified"
    gain = float(os.environ.get("P2B1_SECONDARY_GAIN", "0")) if variant == "B1" else 0.0
    if variant == "B1" and (not np.isfinite(gain) or gain <= 0.0):
        raise RuntimeError("P2B1_B1_requires_positive_global_secondary_gain")
    try:
        lower = np.asarray([float(value) for value in os.environ["D41_DLS_LOWER_LIMITS"].split()], dtype=float)
        upper = np.asarray([float(value) for value in os.environ["D41_DLS_UPPER_LIMITS"].split()], dtype=float)
        buffer_rad = float(os.environ.get("D41_DLS_LIMIT_BUFFER_RAD", "1.0e-4"))
    except (KeyError, ValueError) as exc:
        raise RuntimeError("P2B1_solver_requires_authoritative_joint_limits_and_buffer") from exc
    if lower.shape != (6,) or upper.shape != (6,) or not np.isfinite(np.r_[lower, upper, buffer_rad]).all():
        raise RuntimeError("P2B1_invalid_joint_limit_inputs")
    if np.any(upper - lower <= 2.0 * buffer_rad):
        raise RuntimeError("P2B1_buffer_collapses_joint_interval")

    state = RobotState(moveit.get_robot_model())
    output = RobotTrajectoryMsg()
    output.joint_trajectory.joint_names = joint_names
    seeds = np.asarray(joint_seeds, dtype=np.float64)
    if seeds.shape != (len(target_poses), 6) or not np.isfinite(seeds).all():
        raise ValueError("p2b1_seed_matrix_shape_or_finiteness")

    def target_position(index: int) -> np.ndarray:
        pose = target_poses[index]
        return np.asarray([pose.position.x, pose.position.y, pose.position.z], dtype=np.float64)

    def linearize(q: np.ndarray, index: int):
        state.set_joint_group_positions(group_name, q)
        state.update()
        transform = _transform_matrix(state.get_global_link_transform(ee_link))
        jacobian = np.asarray(state.get_jacobian(group_name, ee_link, np.zeros(3, dtype=float), False), dtype=np.float64)
        if jacobian.shape != (6, 6) or not np.isfinite(jacobian).all():
            raise RuntimeError(f"p2b1_invalid_moveit_jacobian:index={index}:shape={jacobian.shape}")
        normal, basis, angular_basis = _p2b1_tangent_bases(target_normals[index])
        tool_z = transform[:3, 2] / max(float(np.linalg.norm(transform[:3, 2])), 1e-12)
        residual = _p2b1_process_residual(transform[:3, 3], target_position(index), tool_z, normal, basis)
        task_jacobian = _p2b1_process_jacobian(jacobian[:3], jacobian[3:], tool_z, normal, basis)
        weighted_error, weighted_jacobian = _p2b1_weighted_task(residual, task_jacobian, position_weight, normal_weight)
        position_norm = float(np.linalg.norm(residual[:3]))
        normal_angle = float(np.arccos(np.clip(float(np.dot(tool_z, normal)), -1.0, 1.0)))
        return residual, task_jacobian, weighted_error, weighted_jacobian, position_norm, normal_angle

    def objective_value(q: np.ndarray) -> float:
        if p2b2_variant and secondary_objective_name in {"joint_centering", "joint_limit_barrier"}:
            value, _gradient = _p2b2_secondary_objective(q, lower, upper, secondary_objective_name)
            return float(value)
        value, _gradient = _p2b1_joint_centering_objective(q, lower, upper)
        return float(value)

    def pressure_row(
        *, index: int, iteration: int, current: np.ndarray, raw_primary: np.ndarray,
        secondary: np.ndarray, step: np.ndarray, proposal: np.ndarray, projected: np.ndarray,
        accepted: np.ndarray, alpha: float, clipped_joints: list[int], residual: np.ndarray,
        weighted_error: np.ndarray, metrics, objective_before: float, objective_after: float,
        accepted_step: bool,
    ) -> None:
        proposed_j6 = float(proposal[5])
        projected_j6 = float(projected[5])
        upper_excess = max(0.0, proposed_j6 - float(upper[5]))
        buffer_boundary_excess = max(0.0, proposed_j6 - float(upper[5] - buffer_rad))
        _append_p2b1_pressure_record(pressure_path, {
            "case_id": case_id, "variant": variant, "waypoint": index, "iteration": iteration,
            "current_q": current.tolist(), "raw_dls_delta": raw_primary.tolist(),
            "primary_task_contribution": raw_primary.tolist(), "secondary_objective_contribution": secondary.tolist(),
            "trust_region_delta": step.tolist(), "proposed_q_before_projection": proposal.tolist(),
            "projected_q": projected.tolist(), "accepted_q": accepted.tolist(), "accepted_alpha": alpha,
            "projection_triggered": bool(clipped_joints), "projected_joints": clipped_joints,
            "j6_pre_projection": proposed_j6, "j6_post_projection": projected_j6,
            "j6_upper_slack_pre": float(upper[5] - proposed_j6),
            "j6_lower_slack_pre": float(proposed_j6 - lower[5]),
            "j6_upper_slack_post": float(upper[5] - projected_j6),
            "j6_lower_slack_post": float(projected_j6 - lower[5]),
            "max_j6_upper_bound_excess": upper_excess,
            "max_j6_buffer_boundary_excess": buffer_boundary_excess,
            "process_residual_norm": float(np.linalg.norm(residual)),
            "weighted_process_residual_norm": float(np.linalg.norm(weighted_error)),
            "position_residual_m": residual[:3].tolist(), "normal_residual": residual[3:].tolist(),
            "primary_task_contribution_norm": float(np.linalg.norm(raw_primary)),
            "secondary_objective_contribution_norm": float(np.linalg.norm(secondary)),
            "trust_region_delta_norm": float(np.linalg.norm(step)),
            "process_jacobian_rank": metrics.rank,
            "singular_values": metrics.singular_values.tolist(), "nullspace_dimension": metrics.nullity,
            "j6_nullspace_projection": float(metrics.projector[5, 5]),
            "secondary_objective_before": objective_before, "secondary_objective_after": objective_after,
            "line_search_accepted": accepted_step,
        })

    for row_index, row in enumerate(seeds[:warm_start_rows]):
        point = JointTrajectoryPoint()
        point.positions = [float(value) for value in row]
        point.time_from_start = _duration_from_seconds(0.0)
        output.joint_trajectory.points.append(point)
        residual, _task_j, weighted_error, weighted_j, _position_norm, _normal_angle = linearize(row, row_index)
        metrics = _p2b1_nullspace_metrics(weighted_j)
        objective = objective_value(row)
        pressure_row(
            index=row_index, iteration=-1, current=row, raw_primary=np.zeros(6), secondary=np.zeros(6),
            step=np.zeros(6), proposal=row, projected=row, accepted=row, alpha=0.0,
            clipped_joints=[], residual=residual, weighted_error=weighted_error, metrics=metrics,
            objective_before=objective, objective_after=objective, accepted_step=True,
        )

    current, _initial_clips = _p2b1_project_joint_limits(seeds[warm_start_rows - 1], lower, upper, buffer_rad)
    for index in range(warm_start_rows, len(target_poses)):
        accepted = current.copy()
        for iteration in range(int(iterations_per_waypoint)):
            iteration_q = accepted.copy()
            residual, task_jacobian, weighted_error, weighted_jacobian, position_norm, normal_angle = linearize(accepted, index)
            metrics = _p2b1_nullspace_metrics(weighted_jacobian)
            objective_before = objective_value(accepted)
            primary_converged = position_norm <= float(position_tolerance_m) and normal_angle <= float(normal_tolerance_rad)
            if primary_converged and variant == "B0":
                pressure_row(
                    index=index, iteration=iteration, current=accepted, raw_primary=np.zeros(6), secondary=np.zeros(6),
                    step=np.zeros(6), proposal=accepted, projected=accepted, accepted=accepted, alpha=0.0,
                    clipped_joints=[], residual=residual, weighted_error=weighted_error, metrics=metrics,
                    objective_before=objective_before, objective_after=objective_before, accepted_step=True,
                )
                break
            primary = _p2b1_dls_primary_command(weighted_jacobian, weighted_error, damping)
            if variant == "B1" and metrics.rank == 5 and secondary_objective_name != "none":
                if p2b2_variant:
                    secondary, _secondary_norm, _task_residual = _p2b2_rank_aware_secondary_command(
                        accepted, lower, upper, metrics.projector, gain,
                        secondary_objective_name, metrics.rank, task_jacobian=weighted_jacobian,
                    )
                else:
                    secondary, _secondary_norm = _p2b1_secondary_command(accepted, lower, upper, metrics.projector, gain)
            else:
                secondary = np.zeros(6, dtype=np.float64)
            raw_step = primary + secondary
            step = np.clip(raw_step, -float(max_joint_step_rad), float(max_joint_step_rad))
            base_cost = float(np.linalg.norm(weighted_error))
            improved = False
            accepted_alpha = 0.0
            proposal_before_projection = accepted.copy()
            projected_proposal = accepted.copy()
            clipped_joints: list[int] = []
            objective_after = objective_before
            for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625):
                proposal_before_projection = accepted + float(alpha) * step
                projected_proposal, clipped_joints = _p2b1_project_joint_limits(
                    proposal_before_projection, lower, upper, buffer_rad
                )
                candidate_residual, _candidate_j, candidate_weighted_error, _candidate_wj, candidate_position_norm, candidate_normal_angle = linearize(projected_proposal, index)
                candidate_cost = float(np.linalg.norm(candidate_weighted_error))
                objective_after = objective_value(projected_proposal)
                candidate_converged = candidate_position_norm <= float(position_tolerance_m) and candidate_normal_angle <= float(normal_tolerance_rad)
                task_improved = candidate_cost < base_cost - 1e-12
                secondary_improved_within_task_tolerance = (
                    variant == "B1" and primary_converged and candidate_converged
                    and objective_after < objective_before - 1e-12
                )
                if task_improved or secondary_improved_within_task_tolerance:
                    accepted = projected_proposal.copy()
                    improved = True
                    accepted_alpha = float(alpha)
                    break
            pressure_row(
                index=index, iteration=iteration, current=iteration_q,
                raw_primary=primary, secondary=secondary, step=step,
                proposal=proposal_before_projection, projected=projected_proposal, accepted=accepted,
                alpha=accepted_alpha, clipped_joints=clipped_joints, residual=residual,
                weighted_error=weighted_error, metrics=metrics, objective_before=objective_before,
                objective_after=objective_after, accepted_step=improved,
            )
            if not improved:
                break
            if variant == "B1" and iteration + 1 >= int(iterations_per_waypoint):
                break
        final_residual, _final_j, _final_error, _final_wj, position_norm, normal_angle = linearize(accepted, index)
        if position_norm > 4.0e-3 or normal_angle > np.deg2rad(5.0):
            raise RuntimeError(
                f"p2b1_process_geometry_residual:index={index}:position_mm={position_norm * 1000.0:.4f}:"
                f"normal_deg={np.rad2deg(normal_angle):.4f}"
            )
        current = accepted.copy()
        point = JointTrajectoryPoint()
        point.positions = [float(value) for value in current]
        point.time_from_start = _duration_from_seconds(0.0)
        output.joint_trajectory.points.append(point)
    if len(output.joint_trajectory.points) != len(target_poses):
        raise RuntimeError("p2b1_output_count_mismatch")
    return output


def build_normal_constrained_dls_trajectory(
    moveit,
    group_name: str,
    ee_link: str,
    target_poses: list[Pose],
    target_normals: np.ndarray,
    joint_seeds: np.ndarray,
    joint_names: list[str],
    warm_start_rows: int = 16,
    iterations_per_waypoint: int = 80,
    damping: float = 1.0e-2,
    position_weight: float = 100.0,
    normal_weight: float = 0.1,
    max_joint_step_rad: float = 0.16,
    position_tolerance_m: float = 2.0e-5,
    normal_tolerance_rad: float = 2.0e-4,
) -> RobotTrajectoryMsg:
    """Project a causal joint prior onto TCP position plus wall-normal geometry.

    The full-pose KDL route is intentionally not used here.  The authoritative
    process gate constrains TCP position, stand-off, and tool-Z wall normal;
    roll about that normal is a free redundancy.  This Route-D/Route-A hybrid
    uses MoveIt2's native FK and Jacobian to solve exactly those constrained
    quantities while retaining the D39 16-row observed warm start and using
    only the previous accepted state thereafter.
    """

    p2b1_variant = os.environ.get("P2B1_DLS_VARIANT", "").strip().upper()
    p2b2_variant = os.environ.get("P2B2_VARIANT", "").strip().upper()
    if p2b2_variant and p2b1_variant not in {"B0", "B1"}:
        raise ValueError("P2B2_requires_process_task_DLS_route")
    if p2b1_variant in {"B0", "B1"}:
        return build_p2b1_process_task_dls_trajectory(
            moveit, group_name, ee_link, target_poses, target_normals, joint_seeds, joint_names,
            variant=p2b1_variant, warm_start_rows=warm_start_rows,
            iterations_per_waypoint=iterations_per_waypoint, damping=damping,
            position_weight=position_weight, normal_weight=normal_weight,
            max_joint_step_rad=max_joint_step_rad, position_tolerance_m=position_tolerance_m,
            normal_tolerance_rad=normal_tolerance_rad,
            secondary_objective_name=os.environ.get("P2B2_SECONDARY_OBJECTIVE", "joint_centering").strip(),
        )
    if p2b1_variant not in {"", "A0", "A1"}:
        raise ValueError(f"unsupported_P2B1_DLS_VARIANT:{p2b1_variant}")

    if len(target_poses) != len(target_normals) or len(target_poses) != len(joint_seeds):
        raise ValueError("normal_dls_target_seed_count_mismatch")
    if len(target_poses) < 2 or len(joint_names) != 6:
        raise ValueError("normal_dls_invalid_trajectory_shape")
    if warm_start_rows < 1 or warm_start_rows >= len(target_poses):
        raise ValueError("normal_dls_invalid_warm_start_rows")
    state = RobotState(moveit.get_robot_model())
    output = RobotTrajectoryMsg()
    output.joint_trajectory.joint_names = joint_names
    diagnostics_path = os.environ.get("D41_DLS_DIAGNOSTICS_CSV", "").strip()
    p2b1_pressure_path = os.environ.get("P2B1_SOLVER_PRESSURE_CSV", "").strip()
    diagnostics_case = os.environ.get("D41_DLS_CASE_ID", "unspecified").strip() or "unspecified"
    d41_limit_aware = os.environ.get("D41_DLS_LIMIT_AWARE", "false").strip().lower() == "true"
    d41_lower_limits = None
    d41_upper_limits = None
    d41_limit_buffer = 0.0
    if d41_limit_aware:
        try:
            d41_lower_limits = np.asarray([float(value) for value in os.environ["D41_DLS_LOWER_LIMITS"].split()], dtype=float)
            d41_upper_limits = np.asarray([float(value) for value in os.environ["D41_DLS_UPPER_LIMITS"].split()], dtype=float)
        except (KeyError, ValueError) as exc:
            raise RuntimeError("D41_DLS_LIMIT_AWARE requires finite lower/upper position limits") from exc
        if d41_lower_limits.shape != (len(joint_names),) or d41_upper_limits.shape != (len(joint_names),):
            raise RuntimeError("D41_DLS_LIMIT_AWARE position-limit dimension mismatch")
        d41_limit_buffer = float(os.environ.get("D41_DLS_LIMIT_BUFFER_RAD", "1.0e-4"))
        if not np.isfinite(d41_limit_buffer) or d41_limit_buffer < 0.0:
            raise RuntimeError("D41_DLS_LIMIT_AWARE position-limit buffer must be finite and non-negative")

    def enforce_d41_position_limits(values: np.ndarray) -> np.ndarray:
        if not d41_limit_aware:
            return values
        # The position+normal task leaves tool roll redundant.  Projecting
        # proposals into the native URDF bounds is therefore a valid
        # joint-limit safety layer; the ordinary D40 route remains unchanged
        # unless this explicit D41 shadow/retention switch is enabled.
        return np.clip(values, d41_lower_limits + d41_limit_buffer, d41_upper_limits - d41_limit_buffer)

    diagnostics_file = None
    if diagnostics_path:
        diagnostics_file = Path(diagnostics_path)
        diagnostics_file.parent.mkdir(parents=True, exist_ok=True)
        if not diagnostics_file.exists() or diagnostics_file.stat().st_size == 0:
            diagnostics_file.write_text(
                "case_id,waypoint,iteration,pre_position_mm,pre_normal_deg,post_position_mm,post_normal_deg,"
                "damping,damping_increased,accepted_alpha,unclipped_delta_norm_rad,accepted_delta_norm_rad,"
                "trust_region_radius_rad,trust_region_clipped,sigma_min,sigma_max,condition_number,effective_rank,"
                "manipulability,converged,accepted\n",
                encoding="utf-8",
            )

    def write_diagnostic(row: dict[str, object]) -> None:
        if diagnostics_file is None:
            return
        with diagnostics_file.open("a", encoding="utf-8", newline="") as handle:
            csv.DictWriter(
                handle,
                fieldnames=[
                    "case_id", "waypoint", "iteration", "pre_position_mm", "pre_normal_deg",
                    "post_position_mm", "post_normal_deg", "damping", "damping_increased",
                    "accepted_alpha", "unclipped_delta_norm_rad", "accepted_delta_norm_rad",
                    "trust_region_radius_rad", "trust_region_clipped", "sigma_min", "sigma_max",
                    "condition_number", "effective_rank", "manipulability", "converged", "accepted",
                ],
            ).writerow(row)
    seed_values = np.asarray(joint_seeds, dtype=float)
    for row in seed_values[:warm_start_rows]:
        point = JointTrajectoryPoint()
        point.positions = [float(value) for value in row]
        point.time_from_start = _duration_from_seconds(0.0)
        output.joint_trajectory.points.append(point)

    def target_position(index: int) -> np.ndarray:
        return np.asarray(
            [target_poses[index].position.x, target_poses[index].position.y, target_poses[index].position.z],
            dtype=float,
        )

    def normalized_normal(index: int) -> np.ndarray:
        normal = np.asarray(target_normals[index], dtype=float)
        return normal / max(float(np.linalg.norm(normal)), 1.0e-12)

    def residual(q: np.ndarray, index: int) -> tuple[np.ndarray, float, float]:
        state.set_joint_group_positions(group_name, q)
        state.update()
        transform = _transform_matrix(state.get_global_link_transform(ee_link))
        position_error = target_position(index) - transform[:3, 3]
        tool_z = transform[:3, 2] / max(float(np.linalg.norm(transform[:3, 2])), 1.0e-12)
        normal_error = np.cross(tool_z, normalized_normal(index))
        return (
            np.r_[position_error * float(position_weight), normal_error * float(normal_weight)],
            float(np.linalg.norm(position_error)),
            float(np.linalg.norm(normal_error)),
        )

    def write_original_pressure(
        index: int, iteration: int, current_q: np.ndarray, raw_delta: np.ndarray,
        step_delta: np.ndarray, proposal_preprojection: np.ndarray, projected_q: np.ndarray,
        accepted_q: np.ndarray, alpha: float, clipped_joints: list[int],
        error: np.ndarray, position_norm: float, normal_norm: float, accepted_step: bool,
    ) -> None:
        if not p2b1_pressure_path:
            return
        state.set_joint_group_positions(group_name, current_q)
        state.update()
        transform = _transform_matrix(state.get_global_link_transform(ee_link))
        jacobian = np.asarray(state.get_jacobian(group_name, ee_link, np.zeros(3, dtype=float), False), dtype=float)
        if jacobian.shape != (6, 6) or not np.isfinite(jacobian).all():
            raise RuntimeError(f"p2b1_original_pressure_invalid_jacobian:index={index}")
        normal, basis, _angular_basis = _p2b1_tangent_bases(normalized_normal(index))
        tool_z = transform[:3, 2] / max(float(np.linalg.norm(transform[:3, 2])), 1e-12)
        task_jacobian = _p2b1_process_jacobian(jacobian[:3], jacobian[3:], tool_z, normal, basis)
        process_residual = _p2b1_process_residual(
            transform[:3, 3], target_position(index), tool_z, normal, basis,
        )
        weighted_error, weighted_task_jacobian = _p2b1_weighted_task(
            process_residual, task_jacobian, position_weight, normal_weight,
        )
        metrics = _p2b1_nullspace_metrics(weighted_task_jacobian)
        proposed_j6 = float(proposal_preprojection[5])
        projected_j6 = float(projected_q[5])
        objective_before = objective_after = None
        if d41_lower_limits is not None and d41_upper_limits is not None:
            objective_before, _ = _p2b1_joint_centering_objective(current_q, d41_lower_limits, d41_upper_limits)
            objective_after, _ = _p2b1_joint_centering_objective(accepted_q, d41_lower_limits, d41_upper_limits)
        upper_bound = float(d41_upper_limits[5] - d41_limit_buffer) if d41_upper_limits is not None else None
        lower_bound = float(d41_lower_limits[5] + d41_limit_buffer) if d41_lower_limits is not None else None
        _append_p2b1_pressure_record(p2b1_pressure_path, {
            "case_id": diagnostics_case, "variant": p2b1_variant or "A0", "waypoint": index,
            "iteration": iteration, "current_q": current_q.tolist(), "raw_dls_delta": raw_delta.tolist(),
            "primary_task_contribution": raw_delta.tolist(), "secondary_objective_contribution": np.zeros(6).tolist(),
            "trust_region_delta": step_delta.tolist(), "proposed_q_before_projection": proposal_preprojection.tolist(),
            "projected_q": projected_q.tolist(), "accepted_q": accepted_q.tolist(), "accepted_alpha": alpha,
            "projection_triggered": bool(clipped_joints), "projected_joints": clipped_joints,
            "j6_pre_projection": proposed_j6, "j6_post_projection": projected_j6,
            "j6_upper_slack_pre": float(d41_upper_limits[5] - proposed_j6) if d41_upper_limits is not None else None,
            "j6_lower_slack_pre": float(proposed_j6 - d41_lower_limits[5]) if d41_lower_limits is not None else None,
            "j6_upper_slack_post": float(d41_upper_limits[5] - projected_j6) if d41_upper_limits is not None else None,
            "j6_lower_slack_post": float(projected_j6 - d41_lower_limits[5]) if d41_lower_limits is not None else None,
            "max_j6_upper_bound_excess": max(0.0, proposed_j6 - float(d41_upper_limits[5])) if d41_upper_limits is not None else None,
            "max_j6_buffer_boundary_excess": max(0.0, proposed_j6 - upper_bound) if upper_bound is not None else None,
            "process_residual_norm": float(np.linalg.norm(process_residual)),
            "weighted_process_residual_norm": float(np.linalg.norm(weighted_error)),
            "position_residual_m": (error[:3] / float(position_weight)).tolist(),
            "normal_residual": (error[3:] / float(normal_weight)).tolist(),
            "primary_task_contribution_norm": float(np.linalg.norm(raw_delta)),
            "secondary_objective_contribution_norm": 0.0,
            "trust_region_delta_norm": float(np.linalg.norm(step_delta)),
            "process_jacobian_rank": metrics.rank, "singular_values": metrics.singular_values.tolist(),
            "nullspace_dimension": metrics.nullity, "j6_nullspace_projection": float(metrics.projector[5, 5]),
            "secondary_objective_before": objective_before, "secondary_objective_after": objective_after,
            "line_search_accepted": accepted_step,
        })

    if p2b1_pressure_path:
        for row_index, row in enumerate(seed_values[:warm_start_rows]):
            seed_error, seed_position_norm, seed_normal_norm = residual(np.asarray(row, dtype=float), row_index)
            write_original_pressure(
                row_index, -1, np.asarray(row, dtype=float), np.zeros(6), np.zeros(6),
                np.asarray(row, dtype=float), np.asarray(row, dtype=float), np.asarray(row, dtype=float),
                0.0, [], seed_error, seed_position_norm, seed_normal_norm, True,
            )

    current = enforce_d41_position_limits(np.asarray(seed_values[warm_start_rows - 1], dtype=float).copy())
    for index in range(warm_start_rows, len(target_poses)):
        accepted = current.copy()
        for _iteration in range(int(iterations_per_waypoint)):
            iteration_q = accepted.copy()
            error, position_norm, normal_norm = residual(accepted, index)
            if position_norm <= float(position_tolerance_m) and normal_norm <= float(normal_tolerance_rad):
                write_original_pressure(
                    index, _iteration, accepted, np.zeros(6), np.zeros(6), accepted, accepted, accepted,
                    0.0, [], error, position_norm, normal_norm, True,
                )
                write_diagnostic({
                    "case_id": diagnostics_case, "waypoint": index, "iteration": _iteration,
                    "pre_position_mm": position_norm * 1000.0, "pre_normal_deg": np.rad2deg(normal_norm),
                    "post_position_mm": position_norm * 1000.0, "post_normal_deg": np.rad2deg(normal_norm),
                    "damping": float(damping), "damping_increased": False, "accepted_alpha": 0.0,
                    "unclipped_delta_norm_rad": 0.0, "accepted_delta_norm_rad": 0.0,
                    "trust_region_radius_rad": float(max_joint_step_rad), "trust_region_clipped": False,
                    "sigma_min": "", "sigma_max": "", "condition_number": "", "effective_rank": "",
                    "manipulability": "", "converged": True, "accepted": True,
                })
                break
            jacobian = np.asarray(
                state.get_jacobian(group_name, ee_link, np.zeros(3, dtype=float), False),
                dtype=float,
            )
            if jacobian.shape != (6, 6) or not np.isfinite(jacobian).all():
                raise RuntimeError(f"normal_dls_invalid_jacobian:index={index}:shape={jacobian.shape}")
            weighted_jacobian = jacobian.copy()
            weighted_jacobian[:3] *= float(position_weight)
            weighted_jacobian[3:] *= float(normal_weight)
            singular_values = np.linalg.svd(jacobian, compute_uv=False)
            sigma_max = float(singular_values[0]) if singular_values.size else float("nan")
            sigma_min = float(singular_values[-1]) if singular_values.size else float("nan")
            svd_tol = sigma_max * max(jacobian.shape) * np.finfo(float).eps * 100.0
            effective_rank = int(np.sum(singular_values > svd_tol)) if singular_values.size else 0
            condition_number = sigma_max / sigma_min if sigma_min > 0.0 else float("inf")
            manipulability = float(np.prod(singular_values)) if singular_values.size else float("nan")
            system = weighted_jacobian @ weighted_jacobian.T + float(damping) ** 2 * np.eye(6)
            unclipped_delta = weighted_jacobian.T @ np.linalg.solve(system, error)
            if not np.isfinite(unclipped_delta).all():
                raise RuntimeError(f"normal_dls_nonfinite_delta:index={index}")
            delta = np.clip(unclipped_delta, -float(max_joint_step_rad), float(max_joint_step_rad))
            trust_region_clipped = bool(np.any(np.abs(unclipped_delta) > float(max_joint_step_rad) + 1.0e-12))
            base_cost = float(np.linalg.norm(error))
            improved = False
            accepted_alpha = 0.0
            post_position_norm = position_norm
            post_normal_norm = normal_norm
            proposal_before_projection = accepted.copy()
            projected_proposal = accepted.copy()
            clipped_joints: list[int] = []
            for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625):
                proposal_before_projection = accepted + float(alpha) * delta
                proposal = enforce_d41_position_limits(proposal_before_projection)
                projected_proposal = proposal
                clipped_joints = np.flatnonzero(np.abs(proposal - proposal_before_projection) > 1e-12).astype(int).tolist()
                proposal_error, _proposal_position, _proposal_normal = residual(proposal, index)
                if float(np.linalg.norm(proposal_error)) < base_cost:
                    accepted = proposal
                    improved = True
                    accepted_alpha = float(alpha)
                    post_position_norm = float(_proposal_position)
                    post_normal_norm = float(_proposal_normal)
                    break
            write_original_pressure(
                index, _iteration, iteration_q,
                unclipped_delta, delta, proposal_before_projection, projected_proposal,
                accepted, accepted_alpha, clipped_joints, error, position_norm, normal_norm, improved,
            )
            write_diagnostic({
                "case_id": diagnostics_case, "waypoint": index, "iteration": _iteration,
                "pre_position_mm": position_norm * 1000.0, "pre_normal_deg": np.rad2deg(normal_norm),
                "post_position_mm": post_position_norm * 1000.0, "post_normal_deg": np.rad2deg(post_normal_norm),
                "damping": float(damping), "damping_increased": False, "accepted_alpha": accepted_alpha,
                "unclipped_delta_norm_rad": float(np.linalg.norm(unclipped_delta)),
                "accepted_delta_norm_rad": float(accepted_alpha * np.linalg.norm(delta)),
                "trust_region_radius_rad": float(max_joint_step_rad), "trust_region_clipped": trust_region_clipped,
                "sigma_min": sigma_min, "sigma_max": sigma_max, "condition_number": condition_number,
                "effective_rank": effective_rank, "manipulability": manipulability,
                "converged": False, "accepted": improved,
            })
            if not improved:
                break
        _error, position_norm, normal_norm = residual(accepted, index)
        # Keep the generator fail-closed, but do not impose a tighter
        # surrogate threshold than the authoritative native path gate.  The
        # strict MoveIt quality report remains the promotion authority.
        if position_norm > 4.0e-3 or normal_norm > np.deg2rad(5.0):
            raise RuntimeError(
                f"normal_dls_geometry_residual:index={index}:position_mm={position_norm * 1000.0:.4f}:"
                f"normal_deg={np.rad2deg(normal_norm):.4f}"
            )
        current = accepted
        point = JointTrajectoryPoint()
        point.positions = [float(value) for value in current]
        point.time_from_start = _duration_from_seconds(0.0)
        output.joint_trajectory.points.append(point)
    if len(output.joint_trajectory.points) != len(target_poses):
        raise RuntimeError("normal_dls_output_count_mismatch")
    return output


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


def _quaternion_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion, dtype=float)
    norm = max(float(np.linalg.norm([x, y, z, w])), 1e-12)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _pose_to_matrix(pose: Pose) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
    matrix[:3, :3] = _quaternion_to_rotation_matrix(
        np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w], dtype=float)
    )
    return matrix


def _pose_from_matrix(matrix: np.ndarray, source_pose: Pose) -> Pose:
    pose = deepcopy(source_pose)
    pose.position.x = float(matrix[0, 3])
    pose.position.y = float(matrix[1, 3])
    pose.position.z = float(matrix[2, 3])
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = _rotation_matrix_to_quaternion(
        matrix[:3, :3]
    )
    return pose


def _symmetric_roll_offsets(sample_count: int) -> list[float]:
    count = max(1, int(sample_count))
    if count == 1:
        return [0.0]
    step = 2.0 * np.pi / count
    offsets = [0.0]
    for rank in range(1, count):
        magnitude = step * ((rank + 1) // 2)
        offsets.append(magnitude if rank % 2 else -magnitude)
    return offsets[:count]


def _tool_axis_roll_pose_candidates(pose: Pose, sample_count: int) -> list[tuple[float, Pose]]:
    """Return pose variants that preserve TCP position and tool-Z direction."""

    base = _pose_to_matrix(pose)
    candidates: list[tuple[float, Pose]] = []
    for roll in _symmetric_roll_offsets(sample_count):
        c, s = float(np.cos(roll)), float(np.sin(roll))
        rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
        matrix = base.copy()
        matrix[:3, :3] = base[:3, :3] @ rz
        candidates.append((roll, _pose_from_matrix(matrix, pose)))
    return candidates


def _rotation_about_local_axis(axis: int, angle_rad: float) -> np.ndarray:
    """Return a right-handed local-frame rotation matrix."""

    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    if axis == 0:
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)
    if axis == 1:
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)
    if axis == 2:
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    raise ValueError(f"axis must be 0, 1, or 2, got {axis}")


def _continuous_tcp_roll_curve(poses: list[Pose]) -> np.ndarray:
    """Transport the TCP x-axis along the wall-normal curve.

    The input poses can contain an arbitrary per-waypoint x/y frame because
    only tool-Z is constrained by the spray process.  Parallel transport keeps
    that free roll continuous, then unwraps it so the IK solver never receives
    an artificial +/-180 degree frame jump.
    """

    if not poses:
        return np.zeros(0, dtype=float)
    base_frames = [_pose_to_matrix(pose)[:3, :3] for pose in poses]
    transported_x = np.zeros((len(poses), 3), dtype=float)
    transported_y = np.zeros((len(poses), 3), dtype=float)
    first = base_frames[0]
    z0 = first[:, 2] / max(float(np.linalg.norm(first[:, 2])), 1e-12)
    x0 = first[:, 0] - np.dot(first[:, 0], z0) * z0
    if np.linalg.norm(x0) < 1e-9:
        x0 = first[:, 1] - np.dot(first[:, 1], z0) * z0
    x0 /= max(float(np.linalg.norm(x0)), 1e-12)
    y0 = np.cross(z0, x0)
    y0 /= max(float(np.linalg.norm(y0)), 1e-12)
    transported_x[0], transported_y[0] = x0, y0
    for index in range(1, len(poses)):
        current = base_frames[index]
        z = current[:, 2] / max(float(np.linalg.norm(current[:, 2])), 1e-12)
        x = transported_x[index - 1] - np.dot(transported_x[index - 1], z) * z
        if np.linalg.norm(x) < 1e-9:
            x = transported_y[index - 1] - np.dot(transported_y[index - 1], z) * z
        x /= max(float(np.linalg.norm(x)), 1e-12)
        y = np.cross(z, x)
        y /= max(float(np.linalg.norm(y)), 1e-12)
        if np.dot(x, transported_x[index - 1]) < 0.0:
            x, y = -x, -y
        transported_x[index], transported_y[index] = x, y

    rolls = []
    for index, frame in enumerate(base_frames):
        base_x = frame[:, 0]
        base_y = frame[:, 1]
        rolls.append(np.arctan2(np.dot(transported_x[index], base_y), np.dot(transported_x[index], base_x)))
    return np.unwrap(np.asarray(rolls, dtype=float))


def _continuous_curve_step_limit(values: np.ndarray, max_step_rad: float) -> np.ndarray:
    """Unwrap a scalar curve and cap only numerical frame discontinuities."""

    curve = np.unwrap(np.asarray(values, dtype=float).copy())
    if len(curve) < 2 or max_step_rad <= 0.0:
        return curve
    for index in range(1, len(curve)):
        delta = curve[index] - curve[index - 1]
        if abs(delta) > max_step_rad:
            curve[index] = curve[index - 1] + np.sign(delta) * max_step_rad
    return curve


def _global_roll_reparameterized_pose(
    pose: Pose,
    base_roll_rad: float,
    roll_bias_rad: float,
    tilt_x_deg: float,
    tilt_y_deg: float,
) -> Pose:
    """Apply a continuous roll curve and bounded tangent-plane TCP adjustment."""

    base = _pose_to_matrix(pose)
    rx = _rotation_about_local_axis(0, np.deg2rad(float(tilt_x_deg)))
    ry = _rotation_about_local_axis(1, np.deg2rad(float(tilt_y_deg)))
    rz = _rotation_about_local_axis(2, float(base_roll_rad + roll_bias_rad))
    matrix = base.copy()
    matrix[:3, :3] = base[:3, :3] @ rx @ ry @ rz
    return _pose_from_matrix(matrix, pose)


def _contact_pair_label(pair: object) -> str:
    if isinstance(pair, (tuple, list)) and len(pair) >= 2:
        return f"{pair[0]}<->{pair[1]}"
    return str(pair)


def _collision_result_summary(result: CollisionResult, max_pairs: int = 3) -> str:
    count = int(getattr(result, "contact_count", 0))
    try:
        contacts = getattr(result, "contacts", {})
    except TypeError:
        # Jazzy MoveItPy exposes contacts, but the Contact element type is not
        # always registered for Python conversion. Keep diagnostics non-fatal.
        return f"contacts={count},pairs=unavailable"
    if not contacts or not hasattr(contacts, "items"):
        return f"contacts={count}"
    pairs: list[str] = []
    for pair, contact_list in list(contacts.items())[:max_pairs]:
        try:
            pair_count = len(contact_list)
        except TypeError:
            pair_count = 1
        pairs.append(f"{_contact_pair_label(pair)}:{pair_count}")
    if not pairs:
        return f"contacts={count}"
    return f"contacts={count},pairs={'|'.join(pairs)}"


def _state_collision_summary(scene, state, group_name: str) -> tuple[bool, str]:
    request = CollisionRequest()
    request.joint_model_group_name = group_name
    request.contacts = True
    request.max_contacts = 12
    request.max_contacts_per_pair = 2
    result = CollisionResult()
    scene.check_collision(request, result, state)
    return bool(result.collision), _collision_result_summary(result)


def _closed_loop_size_from_points(points: np.ndarray) -> int:
    distances = np.linalg.norm(points[1:] - points[0], axis=1)
    candidates = np.flatnonzero(distances < 1e-7) + 1
    candidates = candidates[candidates >= 16]
    return int(candidates[0]) if len(candidates) else len(points)


def _repeated_closed_loop_size(points: np.ndarray, normals: np.ndarray, tolerance: float = 1e-6) -> int:
    loop_size = _closed_loop_size_from_points(points)
    if loop_size >= len(points) or len(points) % loop_size != 0:
        return len(points)
    reference_points = points[:loop_size]
    reference_normals = normals[:loop_size]
    for start in range(loop_size, len(points), loop_size):
        if np.max(np.linalg.norm(points[start : start + loop_size] - reference_points, axis=1)) > tolerance:
            return len(points)
        normal_error = 1.0 - np.sum(normals[start : start + loop_size] * reference_normals, axis=1)
        if float(np.max(np.abs(normal_error))) > tolerance:
            return len(points)
    return loop_size


def build_wall_collision_objects(
    target_poses: list[Pose],
    target_normals: np.ndarray,
    stand_off: float,
    frame_id: str,
    wall_thickness: float,
    y_thickness: float,
    segment_stride: int,
    include_bottom_closure: bool = True,
    open_path: bool = False,
    tcp_points_to_wall: bool = False,
    include_tunnel_floor: bool = False,
    tunnel_floor_z: float = -0.20,
) -> list[CollisionObject]:
    """Approximate the contour wall as thin oriented boxes."""

    target_tcp = np.array([[pose.position.x, pose.position.y, pose.position.z] for pose in target_poses], dtype=float)
    wall_sign = 1.0 if tcp_points_to_wall else -1.0
    wall_points = target_tcp + wall_sign * stand_off * target_normals
    loop_size = _closed_loop_size_from_points(wall_points)
    wall_loop = wall_points[:loop_size]
    bottom_z = float(np.min(wall_loop[:, 2]))
    objects: list[CollisionObject] = []
    stride = max(1, int(segment_stride))
    y_axis = np.array([0.0, 1.0, 0.0], dtype=float)
    final_start = loop_size - 1 if open_path else loop_size
    for object_index, start in enumerate(range(0, final_start, stride)):
        end = min(start + stride, loop_size - 1) if open_path else (start + stride) % loop_size
        p0 = wall_loop[start]
        p1 = wall_loop[end]
        normal = target_normals[start]
        normal = normal / max(float(np.linalg.norm(normal)), 1e-12)
        segment = p1 - p0
        length = float(np.linalg.norm(segment))
        if length <= 1e-6:
            continue
        if not include_bottom_closure:
            center_candidate = (p0 + p1) * 0.5
            near_bottom = abs(float(center_candidate[2]) - bottom_z) <= max(wall_thickness * 2.0, 1e-3)
            upward_surface = float(normal[2]) > 0.5
            if near_bottom and upward_surface:
                continue
        x_axis = segment / length
        z_axis = np.cross(x_axis, y_axis)
        z_axis /= max(float(np.linalg.norm(z_axis)), 1e-12)
        if float(np.dot(z_axis, normal)) < 0.0:
            z_axis = -z_axis
        # The local box frame is x along the contour, y along tunnel station,
        # and z through wall thickness. The measured wall surface is the inner
        # coating face, so the solid wall volume is offset away from free space.
        rotation = np.column_stack([x_axis, y_axis, z_axis])
        qx, qy, qz, qw = _rotation_matrix_to_quaternion(rotation)
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = [length + wall_thickness, y_thickness, wall_thickness]
        pose = Pose()
        center = (p0 + p1) * 0.5 + wall_sign * normal * (wall_thickness * 0.5)
        pose.position.x = float(center[0])
        pose.position.y = float(center[1])
        pose.position.z = float(center[2])
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
    if include_tunnel_floor:
        floor = SolidPrimitive()
        floor.type = SolidPrimitive.BOX
        floor.dimensions = [
            float(np.max(wall_loop[:, 0]) - np.min(wall_loop[:, 0]) + wall_thickness),
            float(y_thickness),
            float(wall_thickness),
        ]
        floor_pose = Pose()
        floor_pose.position.x = float((np.max(wall_loop[:, 0]) + np.min(wall_loop[:, 0])) * 0.5)
        floor_pose.position.y = float(np.mean(wall_loop[:, 1]))
        floor_pose.position.z = float(tunnel_floor_z - wall_thickness * 0.5)
        floor_pose.orientation.w = 1.0
        floor_obj = CollisionObject()
        floor_obj.header.frame_id = frame_id
        floor_obj.id = "tunnel_floor"
        floor_obj.primitives.append(floor)
        floor_obj.primitive_poses.append(floor_pose)
        floor_obj.operation = CollisionObject.ADD
        objects.append(floor_obj)
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
    include_bottom_closure: bool = True,
    open_path: bool = False,
    tcp_points_to_wall: bool = False,
    include_tunnel_floor: bool = False,
    tunnel_floor_z: float = -0.20,
) -> int:
    objects = build_wall_collision_objects(
        target_poses,
        target_normals,
        stand_off,
        frame_id,
        wall_thickness,
        y_thickness,
        segment_stride,
        include_bottom_closure=include_bottom_closure,
        open_path=open_path,
        tcp_points_to_wall=tcp_points_to_wall,
        include_tunnel_floor=include_tunnel_floor,
        tunnel_floor_z=tunnel_floor_z,
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


def write_joint_waypoint_csv(trajectory_msg, output_path: Path) -> None:
    """Export untimed joint waypoints before MoveIt timing/Ruckig post-processing."""

    trajectory = as_robot_trajectory_message(trajectory_msg).joint_trajectory
    names = list(trajectory.joint_names)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["waypoint"] + [f"{name}_q" for name in names])
        for index, point in enumerate(trajectory.points):
            writer.writerow([index] + [float(value) for value in point.positions])


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


def build_segmented_execution_trajectory_msg(
    robot_trajectory,
    production_joint_step_limit_deg: float = 20.0,
    transition_step_cap_deg: float = 5.0,
    transition_max_speed_deg_s: float = 12.0,
    transition_min_duration_s: float = 2.0,
    stop_hold_s: float = 0.20,
) -> tuple[RobotTrajectoryMsg, list[tuple[int, int]]]:
    """Build the actual controller trajectory with spray-off reorientation stops."""

    source = deepcopy(as_robot_trajectory_message(robot_trajectory))
    points = list(source.joint_trajectory.points)
    if len(points) < 2:
        raise RuntimeError("Segmented execution requires at least two joint points.")
    q = np.asarray([point.positions for point in points], dtype=float)
    raw_time = np.asarray([_duration_seconds(point.time_from_start) for point in points], dtype=float)
    if np.any(np.diff(raw_time) <= 0.0):
        raise RuntimeError("Raw trajectory timestamps must be strictly increasing before segmentation.")

    transition_pairs: list[tuple[int, int]] = []
    for index in range(len(points) - 1):
        delta = circular_joint_delta(q[index + 1] - q[index])
        if float(np.max(np.abs(np.rad2deg(delta)))) > float(production_joint_step_limit_deg):
            transition_pairs.append((index, index + 1))
    if not transition_pairs:
        return source, []

    result = deepcopy(source)
    result.joint_trajectory.points = []
    output_time = 0.0
    last_raw_index = 0

    def append_point(point: JointTrajectoryPoint, time_s: float) -> None:
        point.time_from_start = _duration_from_seconds(max(time_s, output_time))
        result.joint_trajectory.points.append(point)

    def zero_boundary(point: JointTrajectoryPoint) -> JointTrajectoryPoint:
        point.velocities = [0.0] * len(point.positions)
        point.accelerations = [0.0] * len(point.positions)
        return point

    def append_process_point(index: int, force_zero: bool = False) -> None:
        nonlocal output_time, last_raw_index
        point = deepcopy(points[index])
        if force_zero:
            point = zero_boundary(point)
        if result.joint_trajectory.points:
            output_time += max(raw_time[index] - raw_time[last_raw_index], 1e-3)
        append_point(point, output_time)
        last_raw_index = index

    cursor = 0
    for from_index, to_index in transition_pairs:
        for index in range(cursor, from_index + 1):
            append_process_point(index, force_zero=index == from_index)

        q0 = q[from_index]
        delta = q[to_index] - q[from_index]
        max_delta_deg = float(np.max(np.abs(np.rad2deg(delta))))
        interval_count = max(
            8,
            int(np.ceil(max_delta_deg * 1.875 / max(float(transition_step_cap_deg), 1e-6))),
        )
        duration_s = max(float(transition_min_duration_s), max_delta_deg / max(float(transition_max_speed_deg_s), 1e-6))

        if stop_hold_s > 0.0:
            output_time += float(stop_hold_s)
            append_point(zero_boundary(deepcopy(points[from_index])), output_time)

        for sample_index in range(1, interval_count + 1):
            u = sample_index / interval_count
            blend = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
            blend_d1 = 30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4
            blend_d2 = 60.0 * u - 180.0 * u**2 + 120.0 * u**3
            sample = JointTrajectoryPoint()
            sample.positions = (q0 + delta * blend).tolist()
            sample.velocities = (delta * blend_d1 / duration_s).tolist()
            sample.accelerations = (delta * blend_d2 / duration_s**2).tolist()
            output_time += duration_s / interval_count
            append_point(sample, output_time)

        if stop_hold_s > 0.0:
            output_time += float(stop_hold_s)
            append_point(zero_boundary(deepcopy(points[to_index])), output_time)
        last_raw_index = to_index
        cursor = to_index + 1

    for index in range(cursor, len(points)):
        append_process_point(index)
    return result, transition_pairs


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


def _nearest_polyline_distance(points: np.ndarray, polyline: np.ndarray, closed: bool = True) -> np.ndarray:
    """Distance from each point to the closest segment of a 3D polyline."""

    if len(polyline) < 2:
        raise ValueError("A trajectory validation polyline needs at least two target points.")
    a = polyline if closed else polyline[:-1]
    b = np.roll(polyline, -1, axis=0) if closed else polyline[1:]
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
    closed: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project FK points to target segments and interpolate normals."""

    if len(polyline) < 2:
        raise ValueError("A trajectory validation polyline needs at least two target points.")
    if len(normals) != len(polyline):
        raise ValueError("Target normals and target polyline length differ.")
    a = polyline if closed else polyline[:-1]
    b = np.roll(polyline, -1, axis=0) if closed else polyline[1:]
    na = normals if closed else normals[:-1]
    nb = np.roll(normals, -1, axis=0) if closed else normals[1:]
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
    include_bottom_closure_collision: bool = True,
    open_path: bool = False,
    interpolation_step_deg: float = 1.0,
    include_tunnel_floor: bool = False,
    tunnel_floor_z: float = -0.20,
) -> dict[str, float | str]:
    """Check self/world collision at waypoints and interpolated segment states."""

    trajectory = as_robot_trajectory_message(robot_trajectory).joint_trajectory
    state = RobotState(moveit.get_robot_model())
    sample_indices = list(range(0, len(trajectory.points), max(1, stride)))
    if sample_indices[-1] != len(trajectory.points) - 1:
        sample_indices.append(len(trajectory.points) - 1)
    collision_indices: list[int] = []
    collision_segments: list[int] = []
    first_collision_contact_count = 0
    first_collision_summary = ""
    max_step_rad = np.deg2rad(max(float(interpolation_step_deg), 1e-3))
    checks: list[tuple[int, np.ndarray]] = []
    for pair_index, (start_index, end_index) in enumerate(zip(sample_indices[:-1], sample_indices[1:])):
        start_positions = np.asarray(trajectory.points[start_index].positions, dtype=float)
        end_positions = np.asarray(trajectory.points[end_index].positions, dtype=float)
        segment_steps = max(1, int(np.ceil(np.max(np.abs(end_positions - start_positions)) / max_step_rad)))
        for local_index in range(segment_steps):
            alpha = float(local_index) / float(segment_steps)
            checks.append((start_index, start_positions + alpha * (end_positions - start_positions)))
    checks.append((sample_indices[-1], np.asarray(trajectory.points[sample_indices[-1]].positions, dtype=float)))
    psm = moveit.get_planning_scene_monitor()
    with psm.read_only() as scene:
        for check_index, (segment_index, positions) in enumerate(checks):
            state.set_joint_group_positions(group_name, positions)
            state.update()
            request = CollisionRequest()
            request.joint_model_group_name = group_name
            request.contacts = True
            request.max_contacts = 20
            request.max_contacts_per_pair = 1
            result = CollisionResult()
            scene.check_collision(request, result, state)
            if result.collision:
                collision_indices.append(check_index)
                collision_segments.append(segment_index)
                if first_collision_contact_count == 0:
                    first_collision_contact_count = int(result.contact_count)
                    first_collision_summary = _collision_result_summary(result, max_pairs=8)
                    scene.is_state_colliding(state, group_name, True)
    metrics: dict[str, float | str] = {
        "collision_environment_object_count": float(environment_object_count),
        "trajectory_waypoint_count": float(len(trajectory.points)),
        "collision_checked_state_count": float(len(checks)),
        "collision_interpolation_step_deg": float(interpolation_step_deg),
        "collision_count": float(len(collision_indices)),
        "first_collision_index": float(collision_indices[0]) if collision_indices else -1.0,
        "first_collision_trajectory_segment": float(collision_segments[0]) if collision_segments else -1.0,
        "first_collision_contact_count": float(first_collision_contact_count),
        "first_collision_summary": first_collision_summary or "none",
        "include_bottom_closure_collision": "not_applicable_open_path" if open_path else ("true" if include_bottom_closure_collision else "false"),
        "include_tunnel_floor_collision": "true" if include_tunnel_floor else "false",
        "tunnel_floor_z": float(tunnel_floor_z),
        "open_path": "true" if open_path else "false",
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
            + "; trajectory segments: "
            + ", ".join(str(i) for i in collision_segments[:10])
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
    tool_tcp_xyz: str = "",
    tool_tcp_rpy: str = "",
    tool_tcp_source: str = "",
    tool_tcp_measured_by: str = "",
    tool_tcp_measured_date: str = "",
    tool_tcp_calibration_method: str = "",
    max_ik_joint_step_deg: float = 20.0,
    production_joint_step_limit_deg: float = 20.0,
    open_path: bool = False,
    tcp_points_to_wall: bool = False,
    check_joint_continuity: bool = True,
) -> dict[str, float | str]:
    """Reject a post-Ruckig trajectory unless FK satisfies coating constraints."""

    trajectory = as_robot_trajectory_message(robot_trajectory).joint_trajectory
    state = RobotState(moveit.get_robot_model())
    target_tcp = np.array(
        [[pose.position.x, pose.position.y, pose.position.z] for pose in target_poses],
        dtype=float,
    )
    wall_sign = 1.0 if tcp_points_to_wall else -1.0
    wall = target_tcp + wall_sign * stand_off * target_normals
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
    path_deviation = _nearest_polyline_distance(actual, target_tcp, closed=not open_path)
    wall_nearest, normals, _ = _project_to_polyline_with_normals(
        actual, wall, target_normals, closed=not open_path
    )
    normal_error = np.rad2deg(
        np.arccos(np.clip(np.sum(tool_z * normals, axis=1), -1.0, 1.0))
    )
    if tcp_points_to_wall:
        standoff_error = np.sum((wall_nearest - actual) * normals, axis=1) - stand_off
    else:
        standoff_error = np.sum((actual - wall_nearest) * normals, axis=1) - stand_off
    speed = _tcp_speed_from_positions(actual, time)
    speed_mean = max(float(np.mean(speed)), 1e-12)
    speed_p05 = float(np.percentile(speed, 5))
    speed_p95 = float(np.percentile(speed, 95))
    robust_fluctuation = (speed_p95 - speed_p05) / speed_mean
    q = np.asarray([point.positions for point in trajectory.points], dtype=float)
    raw_joint_step = np.abs(np.diff(q, axis=0)) if len(q) > 1 else np.zeros((0, q.shape[1] if q.ndim == 2 else 0))
    joint_step = np.abs(circular_joint_delta(np.diff(q, axis=0))) if len(q) > 1 else raw_joint_step
    if joint_step.size:
        max_step_flat_index = int(np.argmax(joint_step))
        max_step_from_index, max_step_joint_index = np.unravel_index(max_step_flat_index, joint_step.shape)
        max_joint_step_from_index = int(max_step_from_index)
        max_joint_step_deg = float(np.rad2deg(joint_step[max_step_from_index, max_step_joint_index]))
        max_joint_step_to_index = max_joint_step_from_index + 1
        max_joint_step_joint = f"j{max_step_joint_index + 1}"
        max_joint_step_from_deg = float(np.rad2deg(q[max_step_from_index, max_step_joint_index]))
        max_joint_step_to_deg = float(np.rad2deg(q[max_joint_step_to_index, max_step_joint_index]))
        raw_max_step_flat_index = int(np.argmax(raw_joint_step))
        raw_max_step_from_index, raw_max_step_joint_index = np.unravel_index(
            raw_max_step_flat_index,
            raw_joint_step.shape,
        )
        max_joint_step_raw_deg = float(np.rad2deg(raw_joint_step[raw_max_step_from_index, raw_max_step_joint_index]))
        max_joint_step_raw_joint = f"j{raw_max_step_joint_index + 1}"
        max_joint_step_raw_from_index = int(raw_max_step_from_index)
        max_joint_step_raw_to_index = max_joint_step_raw_from_index + 1
    else:
        max_joint_step_deg = 0.0
        max_joint_step_from_index = -1
        max_joint_step_to_index = -1
        max_joint_step_joint = ""
        max_joint_step_from_deg = 0.0
        max_joint_step_to_deg = 0.0
        max_joint_step_raw_deg = 0.0
        max_joint_step_raw_joint = ""
        max_joint_step_raw_from_index = -1
        max_joint_step_raw_to_index = -1
    production_joint_step_limit_deg = float(production_joint_step_limit_deg)
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
        "stand_off_m": stand_off,
        "tcp_points_to_wall": "true" if tcp_points_to_wall else "false",
        "max_joint_step_deg": max_joint_step_deg,
        "max_joint_step_from_index": max_joint_step_from_index,
        "max_joint_step_to_index": max_joint_step_to_index,
        "max_joint_step_joint": max_joint_step_joint,
        "max_joint_step_from_deg": max_joint_step_from_deg,
        "max_joint_step_to_deg": max_joint_step_to_deg,
        "max_joint_step_raw_deg": max_joint_step_raw_deg,
        "max_joint_step_raw_joint": max_joint_step_raw_joint,
        "max_joint_step_raw_from_index": max_joint_step_raw_from_index,
        "max_joint_step_raw_to_index": max_joint_step_raw_to_index,
        "max_ik_joint_step_deg": float(max_ik_joint_step_deg),
        "production_joint_step_limit_deg": production_joint_step_limit_deg,
        "joint_continuity_status": "pass" if max_joint_step_deg <= production_joint_step_limit_deg else "fail",
        "tool_tcp_xyz": tool_tcp_xyz,
        "tool_tcp_rpy": tool_tcp_rpy,
        "tool_tcp_source": tool_tcp_source,
        "tool_tcp_measured_by": tool_tcp_measured_by,
        "tool_tcp_measured_date": tool_tcp_measured_date,
        "tool_tcp_calibration_method": tool_tcp_calibration_method,
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
    report_failures = failures.copy()
    if check_joint_continuity and metrics["joint_continuity_status"] != "pass":
        report_failures.append("joint continuity")
    metrics["status"] = "pass" if not report_failures else "fail:" + ",".join(report_failures)
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
    tool_tcp_xyz = str(node.declare_parameter("tool_tcp_xyz", "").value)
    tool_tcp_rpy = str(node.declare_parameter("tool_tcp_rpy", "").value)
    tool_tcp_source = str(node.declare_parameter("tool_tcp_source", "").value)
    tool_tcp_measured_by = str(node.declare_parameter("tool_tcp_measured_by", "").value)
    tool_tcp_measured_date = str(node.declare_parameter("tool_tcp_measured_date", "").value)
    tool_tcp_calibration_method = str(node.declare_parameter("tool_tcp_calibration_method", "").value)
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
    ik_roll_sample_count = int(node.declare_parameter("ik_roll_sample_count", 1).value)
    ik_beam_width = int(node.declare_parameter("ik_beam_width", 1).value)
    ik_candidates_per_beam = int(node.declare_parameter("ik_candidates_per_beam", 8).value)
    ik_beam_diversity_joint_deg = float(node.declare_parameter("ik_beam_diversity_joint_deg", 0.0).value)
    global_roll_step_deg = float(node.declare_parameter("global_roll_step_deg", 5.0).value)
    global_roll_bias_limit_deg = float(node.declare_parameter("global_roll_bias_limit_deg", 45.0).value)
    global_tilt_step_deg = float(node.declare_parameter("global_tilt_step_deg", 0.5).value)
    global_tilt_limit_deg = float(node.declare_parameter("global_tilt_limit_deg", 2.0).value)
    global_max_backtracks = int(node.declare_parameter("global_max_backtracks", 1200).value)
    global_reparameterization_window = int(node.declare_parameter("global_reparameterization_window", 5).value)
    ik_search_diagnostics_csv = Path(
        node.declare_parameter("ik_search_diagnostics_csv", "outputs/moveit_ik_search_diagnostics.csv").value
    )
    ik_candidate_bank_csv = Path(
        node.declare_parameter("ik_candidate_bank_csv", "outputs/ik_candidate_bank.csv").value
    )
    ik_candidate_collection_report_json = Path(
        node.declare_parameter(
            "ik_candidate_collection_report_json",
            "outputs/ik_candidate_bank_report.json",
        ).value
    )
    ik_max_solve_seconds = float(
        node.declare_parameter(
            "ik_max_solve_seconds",
            0.0,
            ParameterDescriptor(dynamic_typing=True),
        ).value
    )
    seed_joint_csv = str(node.declare_parameter("seed_joint_csv", "").value).strip()
    stand_off = float(node.declare_parameter("stand_off", 0.18).value)
    validate_fk = _parameter_bool(node.declare_parameter("validate_post_ruckig_fk", True).value)
    validate_dynamics = _parameter_bool(node.declare_parameter("validate_joint_dynamics", True).value)
    validation_stride = max(1, int(node.declare_parameter("validation_stride", 1).value))
    max_path_deviation = float(node.declare_parameter("max_path_deviation", 0.005).value)
    max_normal_error_deg = float(node.declare_parameter("max_normal_error_deg", 10.0).value)
    max_standoff_fraction = float(node.declare_parameter("max_standoff_fraction", 0.05).value)
    max_speed_fluctuation = float(node.declare_parameter("max_speed_fluctuation", 0.05).value)
    production_joint_step_limit_deg = float(node.declare_parameter("production_joint_step_limit_deg", 20.0).value)
    quality_report = Path(node.declare_parameter("quality_report_csv", "outputs/moveit_quality_report.csv").value)
    fk_trace_csv = Path(node.declare_parameter("fk_trace_csv", "outputs/moveit_fk_tcp_trace.csv").value)
    trajectory_csv = Path(node.declare_parameter("trajectory_csv", "outputs/moveit_smoothed_joint_trajectory.csv").value)
    segmented_execution = _parameter_bool(node.declare_parameter("segmented_execution", True).value)
    segmented_trajectory_csv = Path(
        node.declare_parameter(
            "segmented_trajectory_csv",
            "outputs/moveit_executed_segmented_joint_trajectory.csv",
        ).value
    )
    transition_step_cap_deg = float(node.declare_parameter("transition_step_cap_deg", 5.0).value)
    transition_max_speed_deg_s = float(node.declare_parameter("transition_max_speed_deg_s", 12.0).value)
    transition_min_duration_s = float(node.declare_parameter("transition_min_duration_s", 2.0).value)
    transition_stop_hold_s = float(node.declare_parameter("transition_stop_hold_s", 0.20).value)
    waypoint_trajectory_csv = Path(
        node.declare_parameter(
            "waypoint_trajectory_csv",
            "outputs/moveit_waypoint_joint_trajectory.csv",
        ).value
    )
    dynamics_report = Path(node.declare_parameter("dynamics_report_csv", "outputs/moveit_joint_dynamics_report.csv").value)
    collision_report = Path(node.declare_parameter("collision_report_csv", "outputs/moveit_collision_report.csv").value)
    validate_collision = _parameter_bool(node.declare_parameter("validate_collision", True).value)
    collision_check_stride = max(1, int(node.declare_parameter("collision_check_stride", validation_stride).value))
    collision_segment_stride = max(1, int(node.declare_parameter("collision_segment_stride", 4).value))
    collision_interpolation_step_deg = max(
        1e-3, float(node.declare_parameter("collision_interpolation_step_deg", 1.0).value)
    )
    tunnel_wall_thickness = float(node.declare_parameter("tunnel_wall_thickness", 0.025).value)
    tunnel_y_thickness = float(node.declare_parameter("tunnel_y_thickness", 0.08).value)
    include_bottom_closure_collision = _parameter_bool(
        node.declare_parameter("include_bottom_closure_collision", True).value
    )
    include_tunnel_floor_collision = _parameter_bool(
        node.declare_parameter("include_tunnel_floor_collision", False).value
    )
    tunnel_floor_z = float(node.declare_parameter("tunnel_floor_z", -0.20).value)
    open_path = _parameter_bool(node.declare_parameter("open_path", False).value)
    tcp_points_to_wall = _parameter_bool(node.declare_parameter("tcp_points_to_wall", False).value)
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
            include_bottom_closure_collision,
            open_path,
            tcp_points_to_wall,
            include_tunnel_floor_collision,
            tunnel_floor_z,
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
    elif planning_mode == "normal_dls_waypoints":
        if joint_seeds is None:
            raise ValueError("planning_mode=normal_dls_waypoints requires seed_joint_csv.")
        node.get_logger().info(
            "Using native MoveIt2 FK/Jacobian normal-constrained DLS projection from the D39 causal warm start."
        )
        trajectory_msg = build_normal_constrained_dls_trajectory(
            moveit,
            group_name,
            ee_link,
            waypoints,
            waypoint_normals,
            joint_seeds,
            joint_names,
        )
    elif planning_mode in {"ik_candidate_bank", "ik_global_roll_backtracking"}:
        if joint_seeds is None:
            raise ValueError("planning_mode=ik_candidate_bank requires seed_joint_csv as a verified reference path.")
        node.get_logger().info(
            "Optimizing one globally continuous TCP roll curve with bounded pose reparameterization and backtracking."
        )
        report = collect_ik_candidate_bank_global_roll_backtracking(
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
            ik_candidate_bank_csv,
            ik_candidate_collection_report_json,
            joint_seeds,
            global_roll_step_deg,
            global_roll_bias_limit_deg,
            global_tilt_step_deg,
            global_tilt_limit_deg,
            global_max_backtracks,
            global_reparameterization_window,
        )
        node.get_logger().info(f"IK candidate bank report: {report}")
        sys.stdout.flush()
        sys.stderr.flush()
        time.sleep(0.1)
        os._exit(0 if report.get("status") == "pass" else 5)
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
            ik_roll_sample_count,
            ik_beam_width,
            ik_candidates_per_beam,
            ik_beam_diversity_joint_deg,
            ik_search_diagnostics_csv,
            ik_candidate_bank_csv,
            ik_max_solve_seconds,
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
            "Use seed_joint_waypoints, normal_dls_waypoints, ik_candidate_bank, "
            "ik_global_roll_backtracking, ik_waypoints, or ompl_segments."
        )
    write_joint_waypoint_csv(trajectory_msg, waypoint_trajectory_csv)
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
    raw_trajectory = trajectory
    executed_trajectory = trajectory
    execution_transition_pairs: list[tuple[int, int]] = []
    if segmented_execution:
        segmented_msg, execution_transition_pairs = build_segmented_execution_trajectory_msg(
            raw_trajectory,
            production_joint_step_limit_deg=production_joint_step_limit_deg,
            transition_step_cap_deg=transition_step_cap_deg,
            transition_max_speed_deg_s=transition_max_speed_deg_s,
            transition_min_duration_s=transition_min_duration_s,
            stop_hold_s=transition_stop_hold_s,
        )
        executed_trajectory = RobotTrajectory(moveit.get_robot_model())
        executed_trajectory.joint_model_group_name = group_name
        executed_trajectory.set_robot_trajectory_msg(execution_start_state, segmented_msg)
        node.get_logger().info(
            "Using executable segmented trajectory with "
            f"{len(execution_transition_pairs)} spray-off reorientation transitions."
        )
    validation_trajectory = executed_trajectory if segmented_execution else raw_trajectory
    quality_metrics: dict[str, float | str] = {}
    if validate_fk:
        quality_metrics = validate_smoothed_trajectory_fk(
            moveit,
            raw_trajectory if segmented_execution else validation_trajectory,
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
            tool_tcp_xyz=tool_tcp_xyz,
            tool_tcp_rpy=tool_tcp_rpy,
            tool_tcp_source=tool_tcp_source,
            tool_tcp_measured_by=tool_tcp_measured_by,
            tool_tcp_measured_date=tool_tcp_measured_date,
            tool_tcp_calibration_method=tool_tcp_calibration_method,
            max_ik_joint_step_deg=float(np.rad2deg(max_ik_joint_step)),
            production_joint_step_limit_deg=production_joint_step_limit_deg,
            open_path=open_path,
            tcp_points_to_wall=tcp_points_to_wall,
            check_joint_continuity=not segmented_execution,
        )
    if validate_dynamics:
        validate_joint_dynamics(moveit, validation_trajectory, group_name, dynamics_report)
    if validate_collision:
        validate_trajectory_collision(
            moveit,
            validation_trajectory,
            group_name,
            collision_report,
            stride=collision_check_stride,
            environment_object_count=collision_object_count,
            include_bottom_closure_collision=include_bottom_closure_collision,
            open_path=open_path,
            interpolation_step_deg=collision_interpolation_step_deg,
            include_tunnel_floor=include_tunnel_floor_collision,
            tunnel_floor_z=tunnel_floor_z,
        )
    write_joint_trajectory_csv(raw_trajectory, trajectory_csv)
    if segmented_execution:
        write_joint_trajectory_csv(executed_trajectory, segmented_trajectory_csv)
    if execute_trajectory:
        if quality_metrics.get("status") != "pass":
            raise RuntimeError(
                "Refusing to execute post-Ruckig trajectory; FK quality report status is "
                f"{quality_metrics.get('status', 'missing')!r}."
            )
        moveit.execute(executed_trajectory, controllers=[])
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
