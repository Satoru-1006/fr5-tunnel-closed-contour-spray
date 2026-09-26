#!/usr/bin/env python3
"""Repeat DP over frozen nodes/edges without ROS2, MoveIt2, or IK."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.frozen_replay_diagnostics import OUTPUT_ROOT, load_yaml, raw_hash, read_parquet, write_json, write_parquet


def _key(state: tuple[float, float, float, int, float, float, tuple[str, ...], tuple[str, ...]]) -> tuple[Any, ...]:
    return state


def solve(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> dict[str, Any]:
    layers: dict[int, list[dict[str, Any]]] = {}
    node_by_id: dict[str, dict[str, Any]] = {}
    for node in nodes:
        node_by_id[str(node["stable_node_id"])] = node
        layers.setdefault(int(node["waypoint_id"]), []).append(node)
    edge_by_source: dict[str, list[dict[str, Any]]] = {}
    for edge in edges:
        edge_by_source.setdefault(str(edge["source_stable_node_id"]), []).append(edge)
    if not layers or len(layers) != 181:
        return {"found": False, "first_unreachable_waypoint": next((i for i in range(181) if i not in layers), 0), "candidate_ids": []}
    states: dict[str, tuple[tuple[float, float, float, int, float, float, tuple[str, ...], tuple[str, ...]], list[str]]] = {}
    for node in sorted(layers[0], key=lambda n: str(n["stable_node_id"])):
        if bool(node.get("valid", True)):
            node_id = str(node["stable_node_id"])
            states[node_id] = ((float(node.get("node_cost", 0.0)), 0.0, 0.0, int(not bool(node.get("is_nominal", True))), float(node.get("actual_position_offset_mm", 0.0)), abs(float(node.get("roll_offset_deg", 0.0))) + abs(float(node.get("standoff_offset_mm", 0.0))), (str(node.get("task_pose_stable_id", "")),), (node_id,)), [node_id])
    first_unreachable = None
    for waypoint in range(180):
        next_states: dict[str, tuple[tuple[Any, ...], list[str]]] = {}
        for source_id, (source_key, path) in states.items():
            for edge in sorted(edge_by_source.get(source_id, []), key=lambda e: str(e["target_stable_node_id"])):
                if not bool(edge.get("valid", False)):
                    continue
                target_id = str(edge["target_stable_node_id"])
                target = node_by_id.get(target_id)
                if target is None or int(target["waypoint_id"]) != waypoint + 1 or not bool(target.get("valid", True)):
                    continue
                total = float(source_key[0]) + float(edge.get("cost", 0.0)) + float(target.get("node_cost", 0.0))
                candidate_key = (total, max(float(source_key[1]), float(edge.get("max_joint_step_deg", 0.0))), float(source_key[2]) + float(edge.get("total_joint_motion_rad", edge.get("total_joint_change_rad", 0.0))), int(source_key[3]) + int(not bool(target.get("is_nominal", True))), float(source_key[4]) + float(target.get("actual_position_offset_mm", 0.0)), float(source_key[5]) + abs(float(target.get("roll_offset_deg", 0.0))) + abs(float(target.get("standoff_offset_mm", 0.0))), source_key[6] + (str(target.get("task_pose_stable_id", "")),), source_key[7] + (target_id,))
                previous = next_states.get(target_id)
                if previous is None or _key(candidate_key) < _key(previous[0]):
                    next_states[target_id] = (candidate_key, path + [target_id])
        states = next_states
        if not states:
            first_unreachable = waypoint + 1
            break
    if first_unreachable is not None:
        return {"found": False, "first_unreachable_waypoint": first_unreachable, "candidate_ids": []}
    best_key, best_path = min((state for state in states.values()), key=lambda item: _key(item[0]))
    selected = [node_by_id[node_id] for node_id in best_path]
    return {"found": True, "first_unreachable_waypoint": None, "candidate_ids": best_path, "selected_nodes": selected, "total_cost": best_key[0], "max_joint_step_deg": best_key[1], "total_joint_motion_rad": best_key[2], "modified_waypoint_count": best_key[3], "total_tcp_offset": best_key[4]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config/stage191_determinism_diagnostics.yaml")
    args = parser.parse_args()
    raw = load_yaml(args.config)
    nodes = read_parquet(OUTPUT_ROOT / "frozen_nodes.parquet")
    edges = read_parquet(OUTPUT_ROOT / "frozen_edges.parquet")
    repeat_count = int(raw["dp_replay"]["repeat_count"])
    records = []
    solutions = []
    for iteration in range(repeat_count):
        result = solve(nodes, edges)
        solutions.append(result)
        selected = result.get("selected_nodes", [])
        records.append({"replay_id": f"dp:{iteration:03d}", "iteration": iteration, "selected_task_pose_sequence_hash": raw_hash([node.get("task_pose_stable_id") for node in selected]), "selected_ik_sequence_hash": raw_hash([node.get("stable_node_id") for node in selected]), "selected_joint_path_hash": raw_hash([node.get("q_unwrapped_rad") for node in selected]), "repair_waypoint_set_hash": raw_hash([int(node["waypoint_id"]) for node in selected if not bool(node.get("is_nominal", True))]), "total_cost": result.get("total_cost"), "max_joint_step_deg": result.get("max_joint_step_deg"), "total_joint_motion_rad": result.get("total_joint_motion_rad"), "modified_waypoint_count": result.get("modified_waypoint_count"), "total_tcp_offset": result.get("total_tcp_offset"), "found": bool(result.get("found"))})
    first = solutions[0]
    selected_nodes = first.get("selected_nodes", [])
    write_parquet(OUTPUT_ROOT / "dp_replay" / "replay_records.parquet", records)
    write_parquet(OUTPUT_ROOT / "frozen_selected_task_pose.parquet", [{"waypoint_id": int(node["waypoint_id"]), "task_pose_stable_id": node.get("task_pose_stable_id")} for node in selected_nodes])
    write_parquet(OUTPUT_ROOT / "frozen_selected_joint_path.parquet", [{"waypoint_id": int(node["waypoint_id"]), "stable_node_id": node.get("stable_node_id"), "ik_candidate_id": node.get("ik_candidate_id"), "q_rad": node.get("q_rad"), "q_unwrapped_rad": node.get("q_unwrapped_rad")} for node in selected_nodes])
    comparison = {"repeat_count": repeat_count, "found_count": sum(bool(item.get("found")) for item in solutions), "selected_task_pose_sequence_hashes": sorted(set(item["selected_task_pose_sequence_hash"] for item in records)), "selected_ik_sequence_hashes": sorted(set(item["selected_ik_sequence_hash"] for item in records)), "selected_joint_path_hashes": sorted(set(item["selected_joint_path_hash"] for item in records)), "all_identical": len({item["selected_joint_path_hash"] for item in records}) == 1, "ros2_disabled": True, "moveit_disabled": True, "collision_recheck_disabled": True}
    write_json(OUTPUT_ROOT / "dp_replay" / "comparison.json", comparison)
    print(comparison)
    return 0 if comparison["all_identical"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
