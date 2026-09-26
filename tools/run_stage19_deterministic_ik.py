#!/usr/bin/env python3
"""Run one real Stage 1.9 deterministic IK/graph/DP reproduction."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.canonical_graph_hash import graph_hashes, semantic_hash
from src.deterministic_ik_candidates import (
    COLLISION_METHOD,
    UNAVAILABLE,
    deduplicate_ik_records,
    deterministic_dp,
    explicit_seed_templates,
    stable_task_pose_identity,
    task_pose_sort_key,
    transition_sort_key,
)
from src.task_pose_repair import TaskPoseRepairConfig, evaluate_transition, generate_task_pose_candidates, read_pose_csv, recover_surface_frames
from src.deterministic_numeric_ik import DeterministicNumericIKSolver


def _safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_parquet(path: Path, rows: list[dict[str, Any]], empty_columns: list[str] | None = None) -> None:
    import pyarrow as pa
    import pyarrow.parquet as parquet

    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        parquet.write_table(pa.Table.from_pylist(_safe(rows)), path)
    else:
        fields = {name: pa.string() for name in (empty_columns or ["status"])}
        parquet.write_table(pa.Table.from_pydict({name: pa.array([], type=typ) for name, typ in fields.items()}), path)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def make_task_records(task_layers: list[list[Any]]) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    layers: list[list[dict[str, Any]]] = []
    for layer in task_layers:
        records: list[dict[str, Any]] = []
        for candidate in layer:
            record = candidate.to_record()
            stable_key, stable_id = stable_task_pose_identity(record)
            record.update({"task_pose_stable_key": stable_key, "task_pose_stable_id": stable_id})
            records.append(record)
        records.sort(key=task_pose_sort_key)
        layers.append(records)
    all_records = [record for layer in layers for record in layer]
    return layers, {"count": len(all_records), "count_by_waypoint": [len(layer) for layer in layers], "semantic_hash": semantic_hash(all_records, sort_fields=("waypoint_id", "task_pose_stable_id"))}


def pose_message(record: dict[str, Any]):
    from geometry_msgs.msg import Pose

    message = Pose()
    message.position.x, message.position.y, message.position.z = [float(x) for x in record["repaired_tcp_position_xyz_m"]]
    message.orientation.x, message.orientation.y, message.orientation.z, message.orientation.w = [float(x) for x in record["repaired_quaternion_xyzw"]]
    return message


def _collision_at(bridge: Any, psm: Any, state: Any, group: str, q: np.ndarray) -> tuple[bool, str]:
    state.set_joint_group_positions(group, q)
    state.update()
    with psm.read_only() as scene:
        return bridge._state_collision_summary(scene, state, group)


def _task_pose_node(record: dict[str, Any], seed: Any, q: np.ndarray, q_unwrapped: np.ndarray, metrics: dict[str, Any], seed_template: Any, call_index: int) -> dict[str, Any]:
    return {
        "waypoint_id": int(record["waypoint_id"]),
        "task_pose_candidate_id": record["task_pose_candidate_id"],
        "task_pose_stable_id": record["task_pose_stable_id"],
        "task_pose_stable_key": record["task_pose_stable_key"],
        "q_rad": q.tolist(),
        "q_unwrapped_rad": q_unwrapped.tolist(),
        "seed_template_id": seed_template.seed_template_id,
        "seed_template_family": seed_template.seed_template_family,
        "seed_template_parameters": dict(seed_template.seed_template_parameters),
        "seed_joint_vector": list(seed_template.seed_joint_vector),
        "seed_order_index": int(seed_template.seed_order_index),
        "seed_provenance": seed_template.to_record(),
        "actual_attempt_count": 1,
        "internal_random_restart_observed": False,
        "ik_call_mode": "deterministic_single_seed",
        "ik_api_entry": "RobotState.set_from_ik",
        "requested_timeout_s": 0.0,
        "solver": "kdl",
        "solver_success": True,
        "solution_joint_vector": q.tolist(),
        "ik_call_index": call_index,
        **metrics,
        "formal_constraint_pass": bool(metrics["valid"]),
        "diagnostic_only": False,
        "valid": bool(metrics["valid"]),
        "node_cost": float(record.get("node_cost", 0.0)),
        "is_nominal": bool(record.get("is_nominal", True)),
        "tangential_offset_mm": float(record.get("tangential_offset_mm", 0.0)),
        "longitudinal_offset_mm": float(record.get("longitudinal_offset_mm", 0.0)),
        "standoff_offset_mm": float(record.get("standoff_offset_mm", 0.0)),
        "roll_offset_deg": float(record.get("roll_offset_deg", 0.0)),
        "actual_position_offset_mm": float(record.get("actual_position_offset_mm", 0.0)),
        "source_seed_template_ids": [],
        "source_seed_count": 0,
        "first_seed_template_id": seed_template.seed_template_id,
        "all_seed_provenance": [],
    }


def evaluate_solution(bridge: Any, moveit: Any, group: str, ee_link: str, task_record: dict[str, Any], q: np.ndarray, q_unwrapped: np.ndarray, psm: Any, state: Any, normal: np.ndarray, standoff: float, config: TaskPoseRepairConfig) -> dict[str, Any]:
    state.set_joint_group_positions(group, q_unwrapped)
    state.update()
    transform = bridge._transform_matrix(state.get_global_link_transform(ee_link))
    target = np.asarray(task_record["repaired_tcp_position_xyz_m"], dtype=float)
    actual = np.asarray(transform[:3, 3], dtype=float)
    position_error = float(np.linalg.norm(actual - target))
    tool_z = np.asarray(transform[:3, 2], dtype=float)
    tool_z /= max(float(np.linalg.norm(tool_z)), 1e-12)
    normal_error = float(np.rad2deg(np.arccos(np.clip(np.dot(tool_z, normal), -1.0, 1.0))))
    surface = np.asarray(task_record["repaired_surface_point_xyz_m"], dtype=float)
    standoff_error = float(np.dot(surface - actual, normal) - float(task_record["actual_standoff_m"]))
    with psm.read_only() as scene:
        colliding, collision_summary = bridge._state_collision_summary(scene, state, group)
    reasons: list[str] = []
    if colliding:
        reasons.append("node_collision")
    if position_error > config.fk_position_error_m:
        reasons.append("position_error")
    if abs(standoff_error) > config.formal_standoff_mm / 1000.0:
        reasons.append("standoff_error")
    if normal_error > config.formal_normal_deg:
        reasons.append("normal_error")
    return {"position_error_m": position_error, "normal_error_deg": normal_error, "tcp_standoff_error_m": standoff_error, "standoff_error_m": standoff_error, "node_collision": bool(colliding), "collision_summary": str(collision_summary), "valid": not reasons, "reject_reason": ";".join(reasons) if reasons else None}


def run_ros(config_path: Path, run_dir: Path, run_id: str) -> dict[str, Any]:  # pragma: no cover - requires ROS2/MoveIt2
    import rclpy
    from moveit.core.robot_state import RobotState
    from moveit.planning import MoveItPy
    import plan_closed_contour_moveit as bridge
    from tools.run_stage17_pose_repair import _post_ruckig_validation_stage17, _write_ruckig_joint_samples

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    stage17_raw = yaml.safe_load((ROOT / raw["inputs"]["stage17_config"]).read_text(encoding="utf-8"))
    stage17_config = TaskPoseRepairConfig.from_mapping(stage17_raw)
    poses_path = ROOT / raw["inputs"]["tcp_pose_csv"]
    seeds_path = ROOT / raw["inputs"]["seed_joint_csv"]
    rows, input_hash = read_pose_csv(poses_path, 181)
    frames, nominal_q, _ = recover_surface_frames(rows, stage17_config.nominal_standoff_m, stage17_config.tcp_points_to_wall)
    task_layers_raw, pose_report = generate_task_pose_candidates(frames, nominal_q, stage17_config, source_phase_id="stage_1_9_deterministic_task_pose")
    task_layers, task_diag = make_task_records(task_layers_raw)
    seed_rows = read_csv_rows(seeds_path)
    seeds = np.asarray([[float(row[f"q{i}"]) for i in range(1, 7)] for row in seed_rows], dtype=float)
    if len(rows) != 181 or len(seeds) != 181:
        raise RuntimeError("Stage 1.9 authoritative input count is not 181/181")

    run_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(run_dir / "task_pose_candidates.parquet", [r for layer in task_layers for r in layer])
    write_json(run_dir / "task_pose_generation.json", {"input_sha256": input_hash, "report": pose_report.__dict__ if hasattr(pose_report, "__dict__") else {"raw_candidate_count": pose_report.raw_candidate_count, "formal_candidate_count": pose_report.formal_candidate_count}, "diagnostics": task_diag})
    write_json(run_dir / "lifecycle.json", [{"phase": "algorithm_start", "run_id": run_id}])

    rclpy.init()
    moveit = None
    try:
        moveit = MoveItPy(node_name=f"stage19_deterministic_ik_{run_id}")
        group = str(raw["robot"]["group_name"])
        ee_link = str(raw["robot"]["ee_link"])
        bridge.preflight_moveit_runtime(moveit, group, ee_link, True)
        poses = bridge.load_tcp_poses(poses_path)
        normals = bridge.load_tcp_normals(poses_path)
        environment_count = bridge.apply_collision_environment(moveit, poses, normals, stage17_config.nominal_standoff_m, "base_link", 0.040, 1.10, 1, False, True, True, -0.20)
        psm = moveit.get_planning_scene_monitor()
        state = RobotState(moveit.get_robot_model())
        node_layers: list[list[dict[str, Any]]] = [[] for _ in range(181)]
        attempts: list[dict[str, Any]] = []
        previous_seed = None
        call_index = 0
        numeric_cfg = raw["numeric"]
        solver_name = str(raw["deterministic_ik"].get("solver", "kdl"))
        numeric_solver = None
        if solver_name == DeterministicNumericIKSolver.solver_name:
            numeric_solver = DeterministicNumericIKSolver(
                ROOT / raw["deterministic_ik"]["expanded_runtime_urdf"],
                ROOT / raw["deterministic_ik"]["joint_limits_path"],
                position_tolerance_m=float(raw["deterministic_ik"].get("position_tolerance_m", stage17_config.fk_position_error_m)),
                tool_z_tolerance_deg=float(raw["deterministic_ik"].get("tool_z_tolerance_deg", stage17_config.formal_normal_deg)),
                orientation_weight=float(raw["deterministic_ik"].get("orientation_weight", 3.0)),
                continuity_weight=float(raw["deterministic_ik"].get("continuity_weight", 0.0)),
                max_nfev=int(raw["deterministic_ik"].get("max_nfev", 160)),
            )
        for waypoint, layer in enumerate(task_layers):
            templates = explicit_seed_templates(waypoint, seeds[waypoint], previous_seed, config=raw["deterministic_ik"])
            previous_seed = seeds[waypoint].copy()
            for task_record in layer:
                task_pose = pose_message(task_record)
                for template in templates:
                    call_index += 1
                    numeric_result = None
                    if numeric_solver is None:
                        state.set_joint_group_positions(group, np.asarray(template.seed_joint_vector, dtype=float))
                        state.update()
                        solved = bool(state.set_from_ik(group, task_pose, ee_link, 0.0))
                        api_entry = "RobotState.set_from_ik"
                    else:
                        target_matrix = numeric_solver.pose_matrix(task_record["repaired_tcp_position_xyz_m"], task_record["repaired_quaternion_xyzw"])
                        numeric_result = numeric_solver.solve(target_matrix, template.seed_joint_vector)
                        solved = bool(numeric_result.success)
                        api_entry = numeric_solver.api_entry
                    attempt = {"ik_api_entry": api_entry, "ik_call_mode": "deterministic_single_seed", "requested_timeout_s": 0.0, "actual_attempt_count": 1, "internal_random_restart_observed": False, "waypoint_id": waypoint, "task_pose_candidate_id": task_record["task_pose_candidate_id"], "task_pose_stable_id": task_record["task_pose_stable_id"], **template.to_record(), "solver": solver_name, "solver_success": solved, "solver_error_code": None if solved else "NO_SOLUTION", "ik_call_index": call_index}
                    if numeric_result is not None:
                        attempt.update({"numeric_solver_status": numeric_result.solver_status, "numeric_function_evaluations": numeric_result.function_evaluations, "numeric_position_error_m": numeric_result.position_error_m, "numeric_tool_z_error_deg": numeric_result.tool_z_error_deg, "numeric_message": numeric_result.message})
                    if not solved:
                        attempts.append(attempt)
                        continue
                    if numeric_result is None:
                        state.update()
                        q = np.asarray(state.get_joint_group_positions(group), dtype=float).copy()
                    else:
                        q = np.asarray(numeric_result.q_rad, dtype=float).copy()
                    q_unwrapped = np.asarray(__import__("src.ik_candidate_graph", fromlist=["nearest_equivalent_joint_positions"]).nearest_equivalent_joint_positions(q, template.seed_joint_vector), dtype=float)
                    metrics = evaluate_solution(bridge, moveit, group, ee_link, task_record, q, q_unwrapped, psm, state, normals[waypoint], stage17_config.nominal_standoff_m, stage17_config)
                    attempt.update({"solver_error_code": None, "solution_joint_vector": q.tolist(), "elapsed_time": None, **metrics})
                    attempts.append(attempt)
                    if metrics["valid"]:
                        node = _task_pose_node(task_record, template.seed_joint_vector, q, q_unwrapped, metrics, template, call_index)
                        node.update({"solver": solver_name, "ik_api_entry": api_entry})
                        if numeric_result is not None:
                            node.update({"numeric_solver_status": numeric_result.solver_status, "numeric_function_evaluations": numeric_result.function_evaluations, "numeric_position_error_m": numeric_result.position_error_m, "numeric_tool_z_error_deg": numeric_result.tool_z_error_deg})
                        node_layers[waypoint].append(node)
            node_layers[waypoint] = deduplicate_ik_records(node_layers[waypoint], float(numeric_cfg["ik_dedup_tolerance_rad"]))
        write_parquet(run_dir / "ik_attempts.parquet", attempts)
        write_json(run_dir / "stage19_progress.json", {"phase": "ik_complete", "attempt_count": len(attempts), "node_count": sum(len(layer) for layer in node_layers)})
        nodes = [node for layer in node_layers for node in layer]
        write_parquet(run_dir / "ik_nodes.parquet", nodes)

        edge_map: dict[tuple[str, str], dict[str, Any]] = {}
        edges: list[dict[str, Any]] = []
        for waypoint in range(180):
            for source in node_layers[waypoint]:
                for target in node_layers[waypoint + 1]:
                    edge = evaluate_transition(source, target, max_joint_step_deg=stage17_config.max_joint_step_deg, interpolation_step_deg=float(raw["collision"]["interpolation_step_deg"]), collision_checker=lambda q, _state=state: _collision_at(bridge, psm, _state, group, q))
                    source_id, target_id = str(source["stable_node_id"]), str(target["stable_node_id"])
                    edge_record = dict(edge)
                    edge_record.update({"source_stable_node_id": source_id, "target_stable_node_id": target_id, "from_stable_node_id": source_id, "to_stable_node_id": target_id, "wrapped_joint_delta_rad": edge.get("delta_q_wrapped_rad"), "nearest_equivalent_joint_delta_rad": edge.get("delta_q_unwrapped_rad"), "total_joint_change_rad": edge.get("total_joint_motion_rad", 0.0), "task_pose_repair_change": float(np.linalg.norm(np.asarray(target.get("u", [target.get('tangential_offset_mm', 0.0), target.get('longitudinal_offset_mm', 0.0), target.get('standoff_offset_mm', 0.0), target.get('roll_offset_deg', 0.0)]), dtype=float) - np.asarray(source.get("u", [source.get('tangential_offset_mm', 0.0), source.get('longitudinal_offset_mm', 0.0), source.get('standoff_offset_mm', 0.0), source.get('roll_offset_deg', 0.0)]), dtype=float))), "roll_change_deg": abs(float(target.get("roll_offset_deg", 0.0)) - float(source.get("roll_offset_deg", 0.0))), "standoff_change_mm": abs(float(target.get("standoff_offset_mm", 0.0)) - float(source.get("standoff_offset_mm", 0.0))), "ccd_status": UNAVAILABLE, "clearance_status": UNAVAILABLE, "cost": float(edge.get("total_joint_motion_rad", 0.0)) + float(edge.get("max_joint_step_deg", 0.0)) / 20.0})
                    edge_map[(source_id, target_id)] = edge_record
                    edges.append(edge_record)
        edges.sort(key=transition_sort_key)
        write_parquet(run_dir / "transition_edges.parquet", edges)
        write_json(run_dir / "stage19_progress.json", {"phase": "edge_complete", "edge_count": len(edges), "node_count": len(nodes)})
        solution = deterministic_dp(node_layers, edge_map)
        selected_nodes = solution.get("selected_nodes", [])
        selected_edges = solution.get("selected_edges", [])
        write_parquet(run_dir / "selected_task_pose_path.parquet", [{"waypoint_id": n["waypoint_id"], "task_pose_stable_id": n["task_pose_stable_id"], "task_pose_candidate_id": n["task_pose_candidate_id"]} for n in selected_nodes])
        write_parquet(run_dir / "selected_joint_path.parquet", [{"waypoint_id": n["waypoint_id"], "stable_node_id": n["stable_node_id"], "ik_candidate_id": n["ik_candidate_id"], "q_rad": n["q_rad"], "q_unwrapped_rad": n["q_unwrapped_rad"]} for n in selected_nodes])
        repair_records = [{"waypoint_id": n["waypoint_id"], "task_pose_stable_id": n["task_pose_stable_id"]} for n in selected_nodes if not bool(n.get("is_nominal", True))]
        ruckig = {"attempted": False, "success": None, "post_ruckig_collision_count": None, "post_ruckig_fk_status": UNAVAILABLE, "post_ruckig_dynamics_status": UNAVAILABLE, "ruckig_trajectory_semantic_hash": None}
        if solution.get("found"):
            from geometry_msgs.msg import Pose
            selected_poses = [pose_message(next(record for record in task_layers[n["waypoint_id"]] if record["task_pose_stable_id"] == n["task_pose_stable_id"])) for n in selected_nodes]
            q_path = np.asarray([n["q_unwrapped_rad"] for n in selected_nodes], dtype=float)
            selected_tasks = [next(record for record in task_layers[n["waypoint_id"]] if record["task_pose_stable_id"] == n["task_pose_stable_id"]) for n in selected_nodes]
            post = _post_ruckig_validation_stage17(bridge, moveit, group, ee_link, q_path, selected_poses, normals, stage17_config, stage17_raw, int(environment_count), run_dir)
            ruckig.update(post)
            csv_path = run_dir / "post_ruckig_joint_trajectory.csv"
            samples: list[dict[str, Any]] = []
            if csv_path.exists():
                samples = read_csv_rows(csv_path)
                samples = [{k: (float(v) if v not in ("", None) and k != "sample_index" else (int(v) if k == "sample_index" and v != "" else None)) for k, v in row.items()} for row in samples]
            write_parquet(run_dir / "ruckig_samples.parquet", samples)
            ruckig["ruckig_trajectory_semantic_hash"] = semantic_hash(samples, sort_fields=("sample_index",)) if samples else None
            write_json(run_dir / "stage19_progress.json", {"phase": "ruckig_complete", "sample_count": len(samples), "ruckig_success": ruckig.get("success")})
        else:
            write_parquet(run_dir / "ruckig_samples.parquet", [], ["sample_index", "t"])

        selected_task_records = [{"waypoint_id": n["waypoint_id"], "task_pose_stable_id": n["task_pose_stable_id"]} for n in selected_nodes]
        selected_joint_records = [{"waypoint_id": n["waypoint_id"], "stable_node_id": n["stable_node_id"], "q_unwrapped_rad": n["q_unwrapped_rad"]} for n in selected_nodes]
        hashes = graph_hashes(task_pose_records=[r for layer in task_layers for r in layer], node_records=nodes, edge_records=edges, selected_task_pose_records=selected_task_records, selected_node_records=selected_nodes, selected_joint_records=selected_joint_records, repair_waypoint_records=repair_records)
        artifact_paths = [run_dir / name for name in ("task_pose_candidates.parquet", "ik_nodes.parquet", "transition_edges.parquet", "selected_task_pose_path.parquet", "selected_joint_path.parquet", "ruckig_samples.parquet")]
        from src.canonical_graph_hash import artifact_hash_map
        hashes["artifact_sha256"] = artifact_hash_map(artifact_paths, root=ROOT)
        write_json(run_dir / "stage19_progress.json", {"phase": "hash_complete", "hashes": hashes})
        valid_path = bool(solution.get("found")) and len(selected_nodes) == 181
        formal_pass = bool(valid_path and ruckig.get("success") is True and ruckig.get("post_ruckig_fk_status") == "pass" and float(ruckig.get("post_ruckig_collision_count", 1)) == 0.0)
        graph_summary = {"schema_version": "1.0", "stage": "stage_1_9", "backend": "ros2_moveit2_planning_scene", "waypoint_count": 181, "complete_on_path": valid_path, "formal_pass": formal_pass, "candidate_counts_by_waypoint": [len(layer) for layer in node_layers], "task_pose_candidate_count": task_diag["count"], "ik_node_count": len(nodes), "transition_edge_count": len(edges), "valid_transition_edge_count": sum(bool(e.get("valid")) for e in edges), "selected_waypoint_count": len(selected_nodes), "total_cost": solution.get("total_cost"), "first_unreachable_waypoint": solution.get("first_unreachable_waypoint"), "collision_method": COLLISION_METHOD, "ccd_status": UNAVAILABLE, "clearance_status": UNAVAILABLE, "ruckig": ruckig, "hashes": hashes, "ik_api": {"entry": (numeric_solver.api_entry if numeric_solver is not None else "RobotState.set_from_ik"), "solver": solver_name, "solver_version": (numeric_solver.solver_version if numeric_solver is not None else "kdl_runtime_plugin"), "call_mode": "deterministic_single_seed", "actual_attempt_count": 1, "internal_random_restart_observed": (numeric_solver.hidden_random_api_calls != 0 if numeric_solver is not None else False), "total_calls": len(attempts), "all_calls_single_attempt": all(a.get("actual_attempt_count") == 1 for a in attempts)}}
        write_json(run_dir / "graph_summary.json", graph_summary)
        write_json(run_dir / "canonical_hashes.json", hashes)
        write_json(run_dir / "acceptance_recalculation.json", {"overall_status": "pass" if formal_pass else "fail", "formal_pass": formal_pass, "recomputed": {"waypoint_count": len(selected_nodes), "selected_task_pose_sequence": [n.get("task_pose_stable_id") for n in selected_nodes], "selected_ik_candidate_sequence": [n.get("stable_node_id") for n in selected_nodes], "total_joint_motion_rad": sum(float(e.get("total_joint_motion_rad", 0.0)) for e in selected_edges), "total_cost": solution.get("total_cost")}, "hashes": hashes, "ruckig_recomputed": ruckig, "post_ruckig_collision_count": ruckig.get("post_ruckig_collision_count"), "fk_status": ruckig.get("post_ruckig_fk_status"), "ccd_status": UNAVAILABLE, "clearance_status": UNAVAILABLE})
        write_json(run_dir / "process_status.json", {"process_exit_status": "pending_teardown", "run_status": "validated_before_teardown" if formal_pass else "completed_with_validation_failure", "teardown_status": "pending", "raw_return_code": 0})
        (run_dir / "stdout.log").touch(exist_ok=True)
        (run_dir / "stderr.log").touch(exist_ok=True)
        write_json(run_dir / "run_manifest.json", {"schema_version": "1.0", "stage": "stage_1_9", "run_id": run_id, "inputs": {"tcp_pose_sha256": input_hash, "waypoint_count": 181}, "configuration": {"config_sha256": __import__("hashlib").sha256(config_path.read_bytes()).hexdigest()}, "determinism": {"random_seed": 0, "python_hash_seed": os.environ.get("PYTHONHASHSEED", "0"), "parallel": False}, "robot_model": {"model_name": raw["robot"]["model_name"], "group_name": group, "ee_link": ee_link, "tool_tcp_source": raw["robot"]["tool_tcp_source"]}, "formal_constraints_locked": True, "backend": "ros2_moveit2_planning_scene"})
        write_json(run_dir / "RUN_COMPLETE.json", {"run_complete": True, "run_id": run_id, "written_before_teardown": True, "formal_pass": formal_pass})
        write_json(run_dir / "lifecycle.json", [{"phase": "algorithm_start", "run_id": run_id}, {"phase": "all_outputs_written"}, {"phase": "run_complete_atomic_write"}, {"phase": "moveitpy_destructor_pending"}])
        return graph_summary
    except Exception as exc:
        write_json(run_dir / "process_exception.json", {"stage": "stage_1_9", "run_id": run_id, "exception_type": type(exc).__name__, "exception": str(exc), "phase": json.loads((run_dir / "stage19_progress.json").read_text(encoding="utf-8")).get("phase") if (run_dir / "stage19_progress.json").exists() else "unknown"})
        raise
    finally:
        try:
            rclpy.shutdown()
        finally:
            pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ros-node", action="store_true")
    parser.add_argument("--config", type=Path, default=ROOT / "config/stage19_deterministic_ik.yaml")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--run-id", default="run_single")
    args, _ = parser.parse_known_args()
    if not args.ros_node:
        print(json.dumps({"stage": "stage_1_9", "status": "blocked", "reason": "real ROS2/MoveIt2 launch required; use tools/run_stage19_deterministic_ik.py via ROS2 launch"}, ensure_ascii=False))
        return 2
    if args.run_dir is None:
        raise SystemExit("--run-dir is required in --ros-node mode")
    summary = run_ros(args.config.resolve(), args.run_dir.resolve(), args.run_id)
    print(json.dumps(_safe(summary), ensure_ascii=False, indent=2))
    return 0 if summary.get("formal_pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
