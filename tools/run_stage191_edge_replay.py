#!/usr/bin/env python3
"""Recheck frozen nodes/edges three times in one real PlanningScene process."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "ros2_moveit_bridge") not in sys.path:
    sys.path.insert(0, str(ROOT / "ros2_moveit_bridge"))

from src.deterministic_ik_candidates import COLLISION_METHOD, UNAVAILABLE, transition_sort_key
from src.frozen_replay_diagnostics import OUTPUT_ROOT, load_yaml, raw_hash, read_parquet, write_json, write_parquet
from src.task_pose_repair import TaskPoseRepairConfig, evaluate_transition


def run_ros(config_path: Path) -> dict[str, object]:  # pragma: no cover - ROS2 only
    import rclpy
    from moveit.core.robot_state import RobotState
    from moveit.planning import MoveItPy
    import plan_closed_contour_moveit as bridge

    raw = load_yaml(config_path)
    nodes = read_parquet(OUTPUT_ROOT / "frozen_nodes.parquet")
    if not nodes:
        raise RuntimeError("frozen_nodes.parquet is required")
    stage17 = load_yaml(ROOT / raw["inputs"]["stage17_config"])
    stage17_config = TaskPoseRepairConfig.from_mapping(stage17)
    poses_path = ROOT / raw["inputs"]["tcp_pose_csv"]
    rclpy.init()
    moveit = MoveItPy(node_name="stage191_edge_replay")
    group, ee_link = str(raw["robot"]["group_name"]), str(raw["robot"]["ee_link"])
    bridge.preflight_moveit_runtime(moveit, group, ee_link, True)
    poses = bridge.load_tcp_poses(poses_path)
    normals = bridge.load_tcp_normals(poses_path)
    bridge.apply_collision_environment(moveit, poses, normals, stage17_config.nominal_standoff_m, "base_link", 0.040, 1.10, 1, True, True, True, False, -0.20)
    psm = moveit.get_planning_scene_monitor()
    state = RobotState(moveit.get_robot_model())
    layers: list[list[dict[str, object]]] = [[] for _ in range(181)]
    for node in nodes:
        layers[int(node["waypoint_id"])].append(node)
    all_results = []
    for iteration in range(3):
        edges = []
        def collision_checker(q: np.ndarray) -> tuple[bool, str]:
            state.set_joint_group_positions(group, q)
            state.update()
            with psm.read_only() as scene:
                return bridge._state_collision_summary(scene, state, group)
        for waypoint in range(180):
            for source in layers[waypoint]:
                for target in layers[waypoint + 1]:
                    edge = evaluate_transition(source, target, max_joint_step_deg=float(stage17_config.max_joint_step_deg), interpolation_step_deg=float(raw["post_validation_replay"]["interpolation_step_deg"]), collision_checker=collision_checker)
                    edges.append({"source_stable_node_id": source["stable_node_id"], "target_stable_node_id": target["stable_node_id"], "valid": bool(edge.get("valid")), "max_joint_step_deg": edge.get("max_joint_step_deg"), "total_joint_motion_rad": edge.get("total_joint_motion_rad", 0.0), "cost": float(edge.get("total_joint_motion_rad", 0.0)) + float(edge.get("max_joint_step_deg", 0.0)) / 20.0, "reject_reason": edge.get("reject_reason"), "collision_status": edge.get("collision_status"), "ccd_status": UNAVAILABLE, "clearance_status": UNAVAILABLE})
        edges.sort(key=transition_sort_key)
        grouped = {waypoint: [] for waypoint in range(180)}
        node_layer_by_id = {str(node["stable_node_id"]): int(node["waypoint_id"]) for node in nodes}
        for edge in edges:
            grouped[node_layer_by_id[str(edge["source_stable_node_id"])]].append(edge)
        all_results.append({"replay_id": f"edge:{iteration:03d}", "iteration": iteration, "node_input_semantic_hash": raw_hash(nodes), "planning_scene_semantic_hash": __import__("json").loads((OUTPUT_ROOT / "planning_scene_snapshot.json").read_text(encoding="utf-8"))["planning_scene_semantic_hash"], "edge_count": len(edges), "valid_edge_count": sum(bool(edge["valid"]) for edge in edges), "joint_step_rejected_count": sum(edge.get("reject_reason") == "joint_step_gate" for edge in edges), "interpolation_collision_count": sum(edge.get("reject_reason") == "interpolated_collision" for edge in edges), "edge_semantic_hash": raw_hash(edges), "edge_hash_by_layer": raw_hash([{ "waypoint": waypoint, "hash": raw_hash(grouped[waypoint]) } for waypoint in range(180)]), "collision_method": COLLISION_METHOD, "ccd_status": UNAVAILABLE, "clearance_status": UNAVAILABLE})
    write_parquet(OUTPUT_ROOT / "edge_replay" / "replay_records.parquet", all_results)
    comparison = {"repeat_count": 3, "node_input_hashes": sorted(set(item["node_input_semantic_hash"] for item in all_results)), "planning_scene_hashes": sorted(set(item["planning_scene_semantic_hash"] for item in all_results)), "edge_hashes": sorted(set(item["edge_semantic_hash"] for item in all_results)), "all_results_identical": len({item["edge_semantic_hash"] for item in all_results}) == 1, "collision_method": COLLISION_METHOD, "ccd_status": UNAVAILABLE, "clearance_status": UNAVAILABLE}
    write_json(OUTPUT_ROOT / "edge_replay" / "comparison.json", comparison)
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
