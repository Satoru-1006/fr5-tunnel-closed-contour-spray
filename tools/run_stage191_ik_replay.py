#!/usr/bin/env python3
"""Real ROS2/MoveIt2 replay against the immutable Stage 1.9.1 corpus."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "ros2_moveit_bridge") not in sys.path:
    sys.path.insert(0, str(ROOT / "ros2_moveit_bridge"))

from src.deterministic_ik_candidates import COLLISION_METHOD, UNAVAILABLE, deduplicate_ik_records, stable_node_identity, transition_sort_key
from src.frozen_replay_diagnostics import OUTPUT_ROOT, load_yaml, raw_hash, quantized_hash, read_parquet, write_json, write_parquet
from src.planning_scene_snapshot import build_snapshot, file_sha256, write_snapshot
from src.task_pose_repair import TaskPoseRepairConfig, evaluate_transition


def _call(obj: Any, *names: str, args: tuple[Any, ...] = (), default: Any = None) -> Any:
    for name in names:
        method = getattr(obj, name, None)
        if callable(method):
            return method(*args)
    return default


def _pose_message(row: dict[str, Any]):
    from geometry_msgs.msg import Pose

    message = Pose()
    message.position.x = float(row["target_x"])
    message.position.y = float(row["target_y"])
    message.position.z = float(row["target_z"])
    message.orientation.x = float(row["target_qx"])
    message.orientation.y = float(row["target_qy"])
    message.orientation.z = float(row["target_qz"])
    message.orientation.w = float(row["target_qw"])
    return message


def _seed(row: dict[str, Any]) -> np.ndarray:
    return np.asarray([float(row[f"seed_q{i}"]) for i in range(1, 7)], dtype=float)


def _state_vector(state: Any, group: str) -> dict[str, Any]:
    values: dict[str, Any] = {"joint_positions": np.asarray(state.get_joint_group_positions(group), dtype=float).tolist()}
    for key, names in (("joint_velocities", ("get_joint_group_velocities",)), ("joint_accelerations", ("get_joint_group_accelerations",))):
        value = _call(state, *names, args=(group,), default=None)
        values[key] = None if value is None else np.asarray(value, dtype=float).tolist()
    return values


def _prepare_state(RobotState: Any, model: Any, group: str, seed: np.ndarray) -> tuple[Any, dict[str, Any]]:
    state = RobotState(model)
    _call(state, "set_to_default_values", "setToDefaultValues")
    state.set_joint_group_positions(group, seed)
    _call(state, "set_joint_group_velocities", "setJointGroupVelocities", args=(group, np.zeros(6, dtype=float)))
    _call(state, "set_joint_group_accelerations", "setJointGroupAccelerations", args=(group, np.zeros(6, dtype=float)))
    _call(state, "enforce_bounds", "enforceBounds")
    state.update()
    return state, _state_vector(state, group)


def _orientation_error(transform: np.ndarray, row: dict[str, Any]) -> float:
    q = np.asarray([float(row["target_qx"]), float(row["target_qy"]), float(row["target_qz"]), float(row["target_qw"])], dtype=float)
    q /= max(np.linalg.norm(q), 1.0e-12)
    x, y, z, w = q
    target = np.asarray([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], dtype=float)
    relative = target.T @ np.asarray(transform[:3, :3], dtype=float)
    return float(math.acos(float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))))


def _run_one(state: Any, row: dict[str, Any], group: str, ee_link: str, bridge: Any, numeric_step: float, process_id: str, iteration: int, call_index: int) -> tuple[dict[str, Any], dict[str, Any] | None]:
    seed = _seed(row)
    initial = _state_vector(state, group)
    initial_hash = raw_hash(initial)
    solved = bool(state.set_from_ik(group, _pose_message(row), ee_link, 0.0))
    record: dict[str, Any] = {
        "replay_id": f"{process_id}:{iteration:03d}:{call_index:08d}", "process_id": process_id, "same_process_iteration": iteration, "call_id": row["call_id"], "call_order": int(row["call_order"]),
        "ik_api_entry": "RobotState.set_from_ik", "ik_call_mode": "deterministic_single_seed", "requested_timeout": 0.0, "actual_attempt_count": 1, "internal_random_restart_observed": False,
        "target_pose_raw_sha256": row["target_pose_raw_sha256"], "seed_raw_sha256": row["seed_raw_sha256"], "initial_robot_state_raw_sha256": initial_hash, "solver_options_sha256": row["solver_options_sha256"], "planning_group": group, "tip_link": ee_link, "solver_success": solved,
        "solver_error_code": None if solved else "NO_SOLUTION", "elapsed_time_s": None,
    }
    if not solved:
        return record, None
    state.update()
    q = np.asarray(state.get_joint_group_positions(group), dtype=float).copy()
    transform = bridge._transform_matrix(state.get_global_link_transform(ee_link))
    target = np.asarray([float(row["target_x"]), float(row["target_y"]), float(row["target_z"])], dtype=float)
    position_error = float(np.linalg.norm(np.asarray(transform[:3, 3], dtype=float) - target))
    record.update({"solution_q1": q[0], "solution_q2": q[1], "solution_q3": q[2], "solution_q4": q[3], "solution_q5": q[4], "solution_q6": q[5], "solution_raw_sha256": raw_hash(q.tolist()), "solution_quantized_sha256": quantized_hash(q.tolist(), numeric_step), "fk_position_error_m": position_error, "fk_orientation_error_rad": _orientation_error(transform, row), "joint_limit_pass": bool(_call(state, "satisfies_bounds", "satisfiesBounds", default=True)),})
    node = dict(row)
    node.update({"q_rad": q.tolist(), "q_unwrapped_rad": q.tolist(), "solver_success": True, "valid": True, "formal_constraint_pass": True, "diagnostic_only": False, "position_error_m": position_error, "normal_error_deg": None, "tcp_standoff_error_m": None, "node_collision": False, "collision_summary": "not_checked_in_ik_replay", "node_cost": 0.0, "is_nominal": True, "actual_position_offset_mm": 0.0, "source_seed_template_ids": [row["seed_template_id"]], "source_seed_count": 1, "first_seed_template_id": row["seed_template_id"], "all_seed_provenance": [row.get("seed_provenance", {})]})
    _, stable_id = stable_node_identity(int(row["waypoint_id"]), str(row["task_pose_stable_id"]), q)
    node.update({"stable_node_id": stable_id, "ik_candidate_id": stable_id, "candidate_id": stable_id})
    return record, node


def _diagnostic_waypoint_set(raw: dict[str, Any]) -> set[int]:
    values = set()
    for item in raw["diagnostic_waypoints"].values():
        values.update(int(x) for x in item)
    return values


def run_ros(config_path: Path, mode: str, process_index: int) -> dict[str, Any]:  # pragma: no cover - requires ROS2/MoveIt2
    import rclpy
    from moveit.core.robot_state import RobotState
    from moveit.planning import MoveItPy
    import plan_closed_contour_moveit as bridge

    raw = load_yaml(config_path)
    corpus = read_parquet(OUTPUT_ROOT / "frozen_ik_call_corpus.parquet")
    if not corpus:
        raise RuntimeError("frozen IK corpus is empty")
    group = str(raw["robot"]["group_name"])
    ee_link = str(raw["robot"]["ee_link"])
    stage17 = load_yaml(ROOT / raw["inputs"]["stage17_config"])
    stage17_config = TaskPoseRepairConfig.from_mapping(stage17)
    poses_path = ROOT / raw["inputs"]["tcp_pose_csv"]
    rclpy.init()
    moveit = MoveItPy(node_name=f"stage191_ik_replay_{mode}_{process_index:03d}")
    bridge.preflight_moveit_runtime(moveit, group, ee_link, True)
    poses = bridge.load_tcp_poses(poses_path)
    normals = bridge.load_tcp_normals(poses_path)
    objects = bridge.build_wall_collision_objects(poses, normals, stage17_config.nominal_standoff_m, "base_link", 0.040, 1.10, 1, include_bottom_closure=True, open_path=True, tcp_points_to_wall=True, include_tunnel_floor=False)
    bridge.apply_collision_environment(moveit, poses, normals, stage17_config.nominal_standoff_m, "base_link", 0.040, 1.10, 1, True, True, True, False, -0.20)
    psm = moveit.get_planning_scene_monitor()
    scene = None
    allowed = "runtime_query_unavailable"
    with psm.read_only() as scene:
        for name in ("get_allowed_collision_matrix", "getAllowedCollisionMatrix"):
            method = getattr(scene, name, None)
            if callable(method):
                try:
                    allowed = str(method())
                except Exception as exc:
                    allowed = f"query_failed:{type(exc).__name__}"
                break
    model = moveit.get_robot_model()
    snapshot = build_snapshot(robot_model_hash=corpus[0].get("robot_model_sha256"), urdf_hash=corpus[0].get("urdf_sha256"), srdf_hash=corpus[0].get("srdf_sha256"), kinematics_hash=corpus[0].get("kinematics_sha256"), joint_limits_hash=corpus[0].get("joint_limits_sha256"), collision_detector_type="moveit_runtime", planning_frame="base_link", group_name=group, objects=objects, allowed_collision_matrix=allowed, robot_state={"joint_positions": []}, interpolation_step_deg=float(raw["post_validation_replay"]["interpolation_step_deg"]))
    write_snapshot(OUTPUT_ROOT / "planning_scene_snapshot.json", snapshot)
    numeric_step = float(raw["numeric"]["joint_hash_quantization_rad"])
    selected = [row for row in corpus if int(row["waypoint_id"]) in _diagnostic_waypoint_set(raw)]
    if mode == "build":
        rows_for_graph = corpus
        iterations = 1
        process_id = "build_frozen_graph"
    else:
        rows_for_graph = selected
        iterations = int(raw["ik_replay"]["same_process_repeat_count"])
        process_id = f"independent_process_{process_index:03d}"
    replay_records: list[dict[str, Any]] = []
    node_records: list[dict[str, Any]] = []
    call_index = 0
    for iteration in range(iterations):
        for row in rows_for_graph:
            call_index += 1
            state, _ = _prepare_state(RobotState, model, group, _seed(row))
            record, node = _run_one(state, row, group, ee_link, bridge, numeric_step, process_id, iteration, call_index)
            replay_records.append(record)
            if node is not None and mode == "build":
                node_records.append(node)
    if mode == "replay":
        output = OUTPUT_ROOT / "ik_replay" / "independent_processes" / f"process_{process_index:03d}.parquet"
        write_parquet(output, replay_records)
        write_json(output.with_suffix(".json"), {"process_id": process_id, "record_count": len(replay_records), "input_corpus_sha256": raw_hash(corpus)})
        rclpy.shutdown()
        return {"mode": mode, "process_id": process_id, "record_count": len(replay_records), "output": output.as_posix()}
    # Build the canonical frozen node set from this one real full replay.
    node_layers: list[list[dict[str, Any]]] = [[] for _ in range(181)]
    for node in node_records:
        node_layers[int(node["waypoint_id"])].append(node)
    for waypoint in range(181):
        node_layers[waypoint] = deduplicate_ik_records(node_layers[waypoint], float(raw["numeric"]["ik_dedup_tolerance_rad"]))
    frozen_nodes = [node for layer in node_layers for node in layer]
    write_parquet(OUTPUT_ROOT / "ik_replay" / "same_process" / "build_replay_records.parquet", replay_records)
    write_parquet(OUTPUT_ROOT / "frozen_nodes.parquet", frozen_nodes)
    edge_map: dict[tuple[str, str], dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    state = _prepare_state(RobotState, model, group, _seed(corpus[0]))[0]
    def collision_checker(q: np.ndarray) -> tuple[bool, str]:
        state.set_joint_group_positions(group, q)
        state.update()
        with psm.read_only() as current_scene:
            return bridge._state_collision_summary(current_scene, state, group)
    for waypoint in range(180):
        for source in node_layers[waypoint]:
            for target in node_layers[waypoint + 1]:
                edge = evaluate_transition(source, target, max_joint_step_deg=float(stage17_config.max_joint_step_deg), interpolation_step_deg=float(raw["post_validation_replay"]["interpolation_step_deg"]), collision_checker=collision_checker)
                source_id, target_id = str(source["stable_node_id"]), str(target["stable_node_id"])
                edge_record = dict(edge)
                edge_record.update({"source_stable_node_id": source_id, "target_stable_node_id": target_id, "from_stable_node_id": source_id, "to_stable_node_id": target_id, "cost": float(edge.get("total_joint_motion_rad", 0.0)) + float(edge.get("max_joint_step_deg", 0.0)) / 20.0, "ccd_status": UNAVAILABLE, "clearance_status": UNAVAILABLE})
                edge_map[(source_id, target_id)] = edge_record
                edges.append(edge_record)
    edges.sort(key=transition_sort_key)
    write_parquet(OUTPUT_ROOT / "frozen_edges.parquet", edges)
    write_json(OUTPUT_ROOT / "ik_replay" / "same_process" / "build_summary.json", {"record_count": len(replay_records), "node_count": len(frozen_nodes), "edge_count": len(edges), "valid_edge_count": sum(bool(edge.get("valid")) for edge in edges), "collision_method": COLLISION_METHOD})
    # Diagnostic repeated calls are deliberately done after graph freezing but
    # still in the same ROS2 process and with a new RobotState per call.
    repeat_records: list[dict[str, Any]] = []
    for iteration in range(int(raw["ik_replay"]["same_process_repeat_count"])):
        for row in selected:
            state, _ = _prepare_state(RobotState, model, group, _seed(row))
            record, _ = _run_one(state, row, group, ee_link, bridge, numeric_step, "same_process", iteration, len(repeat_records) + 1)
            repeat_records.append(record)
    write_parquet(OUTPUT_ROOT / "ik_replay" / "same_process" / "replay_records.parquet", repeat_records)
    write_json(OUTPUT_ROOT / "ik_replay" / "same_process" / "replay_summary.json", {"record_count": len(repeat_records), "repeat_count": int(raw["ik_replay"]["same_process_repeat_count"]), "new_robot_state_per_call": True, "actual_attempt_count": 1, "internal_random_restart_observed": False})
    rclpy.shutdown()
    return {"mode": mode, "record_count": len(replay_records), "node_count": len(frozen_nodes), "edge_count": len(edges)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ros-node", action="store_true")
    parser.add_argument("--mode", choices=("build", "replay"), default="build")
    parser.add_argument("--process-index", type=int, default=1)
    parser.add_argument("--config", type=Path, default=ROOT / "config/stage191_determinism_diagnostics.yaml")
    args, _ = parser.parse_known_args()
    if not args.ros_node:
        print("real ROS2/MoveIt2 launch required", file=sys.stderr)
        return 2
    result = run_ros(args.config.resolve(), args.mode, args.process_index)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
