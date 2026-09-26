#!/usr/bin/env python3
"""Replay frozen Ruckig samples through the real MoveIt PlanningScene/FK."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "ros2_moveit_bridge") not in sys.path:
    sys.path.insert(0, str(ROOT / "ros2_moveit_bridge"))

from src.deterministic_ik_candidates import COLLISION_METHOD, UNAVAILABLE
from src.frozen_replay_diagnostics import OUTPUT_ROOT, load_yaml, raw_hash, read_parquet, write_json, write_parquet
from src.planning_scene_snapshot import build_snapshot, write_snapshot


def run_ros(config_path: Path) -> dict[str, object]:  # pragma: no cover - ROS2 only
    import rclpy
    from moveit.core.robot_state import RobotState
    from moveit.planning import MoveItPy
    import plan_closed_contour_moveit as bridge

    raw = load_yaml(config_path)
    samples = read_parquet(OUTPUT_ROOT / "frozen_ruckig_samples.parquet")
    nodes = {str(row["stable_node_id"]): row for row in read_parquet(OUTPUT_ROOT / "frozen_nodes.parquet")}
    path = sorted(read_parquet(OUTPUT_ROOT / "frozen_selected_joint_path.parquet"), key=lambda row: int(row["waypoint_id"]))
    if not samples or len(path) != 181:
        raise RuntimeError("frozen Ruckig samples and selected 181-point path are required")
    stage17 = load_yaml(ROOT / raw["inputs"]["stage17_config"])
    from src.task_pose_repair import TaskPoseRepairConfig
    stage17_config = TaskPoseRepairConfig.from_mapping(stage17)
    poses_path = ROOT / raw["inputs"]["tcp_pose_csv"]
    rclpy.init()
    moveit = MoveItPy(node_name="stage191_post_validation_replay")
    group, ee_link = str(raw["robot"]["group_name"]), str(raw["robot"]["ee_link"])
    bridge.preflight_moveit_runtime(moveit, group, ee_link, True)
    poses = bridge.load_tcp_poses(poses_path)
    normals = bridge.load_tcp_normals(poses_path)
    objects = bridge.build_wall_collision_objects(poses, normals, stage17_config.nominal_standoff_m, "base_link", 0.040, 1.10, 1, include_bottom_closure=True, open_path=True, tcp_points_to_wall=True, include_tunnel_floor=False)
    bridge.apply_collision_environment(moveit, poses, normals, stage17_config.nominal_standoff_m, "base_link", 0.040, 1.10, 1, True, True, True, False, -0.20)
    psm = moveit.get_planning_scene_monitor()
    snapshot = json.loads((OUTPUT_ROOT / "planning_scene_snapshot.json").read_text(encoding="utf-8"))
    scene_hash = snapshot["planning_scene_semantic_hash"]
    model = moveit.get_robot_model()
    state = RobotState(model)
    records = []
    for iteration in range(int(raw["post_validation_replay"]["repeat_count"])):
        collision_count = 0
        first_collision = None
        max_fk = 0.0
        max_vel = 0.0
        max_acc = 0.0
        max_jerk = 0.0
        previous_acc = None
        dt = 0.01
        for index, sample in enumerate(samples):
            q = np.asarray([float(sample[f"q_j{joint}"]) for joint in range(1, 7)], dtype=float)
            state.set_joint_group_positions(group, q)
            state.update()
            with psm.read_only() as current_scene:
                colliding, _ = bridge._state_collision_summary(current_scene, state, group)
            if colliding:
                collision_count += 1
                if first_collision is None:
                    first_collision = index
            target_index = min(max(int(round(index * 180 / max(len(samples) - 1, 1))), 0), 180)
            target = path[target_index]
            node = nodes[str(target["stable_node_id"])]
            transform = bridge._transform_matrix(state.get_global_link_transform(ee_link))
            target_xyz = np.asarray([float(node["target_x"]), float(node["target_y"]), float(node["target_z"])], dtype=float)
            max_fk = max(max_fk, float(np.linalg.norm(np.asarray(transform[:3, 3], dtype=float) - target_xyz)))
            velocity = np.asarray([float(sample.get(f"v_j{joint}") or 0.0) for joint in range(1, 7)], dtype=float)
            acceleration = np.asarray([float(sample.get(f"a_j{joint}") or 0.0) for joint in range(1, 7)], dtype=float)
            max_vel = max(max_vel, float(np.max(np.abs(velocity))))
            max_acc = max(max_acc, float(np.max(np.abs(acceleration))))
            if previous_acc is not None:
                max_jerk = max(max_jerk, float(np.max(np.abs((acceleration - previous_acc) / dt))))
            previous_acc = acceleration
        summary = {"planning_scene_semantic_hash": scene_hash, "ruckig_trajectory_semantic_hash": raw_hash(samples), "post_validation_semantic_hash": raw_hash({"collision_count": collision_count, "first_collision_sample": first_collision, "max_fk_position_error": max_fk, "max_velocity": max_vel, "max_acceleration": max_acc, "max_jerk": max_jerk}), "fk_pass": max_fk <= float(raw["numeric"]["replay_fk_position_tolerance_m"]), "max_fk_position_error": max_fk, "max_fk_orientation_error": None, "post_collision_count": collision_count, "first_collision_sample": first_collision, "max_velocity_ratio": max_vel, "max_acceleration_ratio": max_acc, "max_jerk_ratio": max_jerk, "collision_method": COLLISION_METHOD, "ccd_status": UNAVAILABLE, "clearance_status": UNAVAILABLE}
        summary.update({"replay_id": f"post:{iteration:03d}", "iteration": iteration})
        records.append(summary)
    write_parquet(OUTPUT_ROOT / "post_validation_replay" / "replay_records.parquet", records)
    comparison = {"repeat_count": len(records), "planning_scene_hashes": sorted(set(row["planning_scene_semantic_hash"] for row in records)), "trajectory_hashes": sorted(set(row["ruckig_trajectory_semantic_hash"] for row in records)), "post_validation_hashes": sorted(set(row["post_validation_semantic_hash"] for row in records)), "all_results_identical": len({row["post_validation_semantic_hash"] for row in records}) == 1, "collision_method": COLLISION_METHOD, "ccd_status": UNAVAILABLE, "clearance_status": UNAVAILABLE}
    write_json(OUTPUT_ROOT / "post_validation_replay" / "comparison.json", comparison)
    rclpy.shutdown()
    return comparison


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ros-node", action="store_true")
    parser.add_argument("--config", type=Path, default=ROOT / "config/stage191_determinism_diagnostics.yaml")
    args, _ = parser.parse_known_args()
    if not args.ros_node:
        return 2
    print(run_ros(args.config.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
