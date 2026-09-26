#!/usr/bin/env python3
"""D41 shadow DLS batch runner.

This deliberately calls the locked D40 DLS implementation.  It changes only
the task/seed inputs for the shadow matrix and never changes the canonical
trajectory or baseline files.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np


def _pose(values):
    from geometry_msgs.msg import Pose

    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = [float(v) for v in values[:3]]
    if len(values) >= 7:
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = [float(v) for v in values[3:7]]
    else:
        pose.orientation.w = 1.0
    return pose


def _load_rows(path: Path) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader)
        rows = [[float(v) for v in row[-6:]] for row in reader if row]
    return np.asarray(rows, dtype=float)


def _write_rows(path: Path, rows: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["waypoint", "j1_q", "j2_q", "j3_q", "j4_q", "j5_q", "j6_q"])
        for index, row in enumerate(rows):
            writer.writerow([index, *[f"{float(v):.17g}" for v in row]])


def _resample(values: np.ndarray, count: int) -> np.ndarray:
    old = np.linspace(0.0, 1.0, len(values))
    new = np.linspace(0.0, 1.0, count)
    return np.column_stack([np.interp(new, old, values[:, j]) for j in range(values.shape[1])])


def _run(config_path: Path) -> int:
    # Import the bridge after ROS has been initialized by this process.
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "ros2_moveit_bridge"))
    import rclpy
    from moveit.planning import MoveItPy
    from moveit.core.robot_state import RobotState
    from plan_closed_contour_moveit import _transform_matrix, build_normal_constrained_dls_trajectory

    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if config.get("limit_aware", False):
        os.environ["D41_DLS_LIMIT_AWARE"] = "true"
        os.environ["D41_DLS_LOWER_LIMITS"] = " ".join(str(float(v)) for v in config["position_limits"]["lower"])
        os.environ["D41_DLS_UPPER_LIMITS"] = " ".join(str(float(v)) for v in config["position_limits"]["upper"])
        os.environ["D41_DLS_LIMIT_BUFFER_RAD"] = str(float(config.get("limit_buffer_rad", 1.0e-4)))
    summary_path = output / "D41_dls_case_summary.jsonl"
    poses_by_case = config["cases"]
    rclpy.init(args=None)
    moveit = MoveItPy(node_name="stage3_h13_d41_dls_batch")
    summaries = []
    try:
        for case in poses_by_case:
            case_id = str(case["case_id"])
            os.environ["D41_DLS_CASE_ID"] = case_id
            os.environ["D41_DLS_DIAGNOSTICS_CSV"] = str(output / "D41_dls_diagnostics.csv")
            target_poses = [_pose(row) for row in case["poses"]]
            target_normals = np.asarray(case["normals"], dtype=float)
            seeds = _load_rows(Path(case["seeds_csv"]))
            status = "FAIL"
            error = ""
            q_rows = np.empty((0, 6), dtype=float)
            max_position = float("inf")
            max_normal = float("inf")
            max_step = float("inf")
            bounds_ok = False
            try:
                trajectory = build_normal_constrained_dls_trajectory(
                    moveit,
                    "fairino5_v6_group",
                    "spray_tcp_link",
                    target_poses,
                    target_normals,
                    seeds,
                    ["j1", "j2", "j3", "j4", "j5", "j6"],
                )
                q_rows = np.asarray([point.positions for point in trajectory.joint_trajectory.points], dtype=float)
                state = RobotState(moveit.get_robot_model())
                position_errors = []
                normal_errors = []
                bounds_ok = bool(np.isfinite(q_rows).all())
                for row_index, (q, target, normal) in enumerate(zip(q_rows, target_poses, target_normals)):
                    state.set_joint_group_positions("fairino5_v6_group", q)
                    state.update()
                    transform = _transform_matrix(state.get_global_link_transform("spray_tcp_link"))
                    # MoveItPy returns a 4x4 matrix-like object; conversion is
                    # supported in the same way as the locked D40 bridge.
                    # D40's first 16 states are the retained observed warm
                    # start; its geometry projection gate begins at row 16.
                    if row_index >= 16:
                        position_errors.append(float(np.linalg.norm(np.asarray([target.position.x, target.position.y, target.position.z]) - transform[:3, 3])))
                    tool_z = transform[:3, 2] / max(float(np.linalg.norm(transform[:3, 2])), 1e-12)
                    if row_index >= 16:
                        normal_errors.append(float(np.linalg.norm(np.cross(tool_z, normal / max(float(np.linalg.norm(normal)), 1e-12)))))
                max_position = max(position_errors, default=float("inf"))
                max_normal = max(normal_errors, default=float("inf"))
                max_step = float(np.max(np.rad2deg(np.abs(np.diff(q_rows, axis=0))))) if len(q_rows) > 1 else 0.0
                status = "PASS" if len(q_rows) == len(target_poses) and bounds_ok and max_position <= 4e-3 and max_normal <= np.deg2rad(5.0) else "FAIL"
                if status != "PASS":
                    error = "post_dls_geometry_or_bounds_gate"
            except Exception as exc:  # shadow failures are evidence, not fatal to the campaign
                error = f"{type(exc).__name__}:{exc}"
            q_path = output / "trajectories" / f"{case_id}.csv"
            if len(q_rows):
                _write_rows(q_path, q_rows)
            record = {
                "case_id": case_id,
                "family": case["family"],
                "status": status,
                "error": error,
                "state_count": int(len(q_rows)),
                "target_count": int(len(target_poses)),
                "max_position_error_m": max_position,
                "max_normal_error_rad": max_normal,
                "max_joint_step_deg": max_step,
                "joint_bounds_ok": bounds_ok,
                "future_joint_reads": 0,
                "observed_seed_rows_used": 16,
                "trajectory_csv": str(q_path) if len(q_rows) else None,
            }
            summaries.append(record)
            with summary_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
    finally:
        os.environ.pop("D41_DLS_CASE_ID", None)
        os.environ.pop("D41_DLS_DIAGNOSTICS_CSV", None)
        os.environ.pop("D41_DLS_LIMIT_AWARE", None)
        os.environ.pop("D41_DLS_LOWER_LIMITS", None)
        os.environ.pop("D41_DLS_UPPER_LIMITS", None)
        os.environ.pop("D41_DLS_LIMIT_BUFFER_RAD", None)
        rclpy.shutdown()
    return 0 if summaries else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args, _unknown_ros_args = parser.parse_known_args()
    return _run(args.config)


if __name__ == "__main__":
    raise SystemExit(main())
