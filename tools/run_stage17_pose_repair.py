#!/usr/bin/env python3
"""Run Stage 1.7 task-pose repair and MoveIt2-backed joint graph search.

The default Windows invocation is a preflight/audit run.  It creates task-pose
evidence but refuses to fabricate IK, PlanningScene, dynamics, or Ruckig pass
records when ROS2/MoveIt2 is unavailable.  The formal backend is launched with
the repository's complete ROS environment and ``--ros-node``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "ros2_moveit_bridge") not in sys.path:
    sys.path.insert(0, str(ROOT / "ros2_moveit_bridge"))

from src.ik_candidate_graph import nearest_equivalent_joint_positions
from src.task_pose_repair import (
    COLLISION_METHOD,
    UNAVAILABLE,
    TaskPoseCandidate,
    TaskPoseRepairConfig,
    evaluate_transition,
    generate_task_pose_candidates,
    read_pose_csv,
    recover_surface_frames,
    solve_second_order_dp,
)


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


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_safe(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _stage18_lifecycle(path: Path, phase: str, status: str = "complete", **details: Any) -> None:
    """Append a fsync'd lifecycle marker without changing Stage 1.7 planning."""
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "status": status,
        **_safe(details),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_ruckig_joint_samples(path: Path, trajectory: Any) -> int:
    """Persist native MoveIt/Ruckig samples for Stage 1.8 independent auditing."""
    message = trajectory
    getter = getattr(trajectory, "get_robot_trajectory_msg", None)
    if callable(getter):
        message = getter()
    points = list(getattr(getattr(message, "joint_trajectory", None), "points", []) or [])
    if not points:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["sample_index", "t"] + [f"q_j{i}" for i in range(1, 7)] + [f"v_j{i}" for i in range(1, 7)] + [f"a_j{i}" for i in range(1, 7)]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, point in enumerate(points):
            duration = getattr(point, "time_from_start", None)
            seconds = float(getattr(duration, "sec", 0)) + float(getattr(duration, "nanosec", 0)) * 1.0e-9
            positions = list(getattr(point, "positions", []) or [])
            velocities = list(getattr(point, "velocities", []) or [])
            accelerations = list(getattr(point, "accelerations", []) or [])
            row: dict[str, Any] = {"sample_index": index, "t": seconds}
            for joint in range(6):
                row[f"q_j{joint + 1}"] = positions[joint] if joint < len(positions) else ""
                row[f"v_j{joint + 1}"] = velocities[joint] if joint < len(velocities) else ""
                row[f"a_j{joint + 1}"] = accelerations[joint] if joint < len(accelerations) else ""
            writer.writerow(row)
    return len(points)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fieldnames or sorted({key for row in rows for key in row}) or ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows([{key: json.dumps(_safe(value), ensure_ascii=False) if isinstance(value, (dict, list, tuple)) else value for key, value in row.items()} for row in rows])


def _write_parquet_or_json(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Write real Parquet only when pyarrow is installed; never fake it."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as parquet
    except Exception as exc:
        fallback = path.with_suffix(".json")
        _write_json(fallback, {"status": "parquet_not_written", "reason": f"{type(exc).__name__}: {exc}", "records": rows})
        return {"path": str(fallback), "format": "json_fallback", "parquet_available": False, "reason": str(exc)}
    if not rows:
        return {"path": None, "format": "parquet_omitted_empty", "parquet_available": True}
    path.parent.mkdir(parents=True, exist_ok=True)
    parquet.write_table(pa.Table.from_pylist(_safe(rows)), path)
    return {"path": str(path), "format": "parquet", "parquet_available": True, "version": getattr(pa, "__version__", "unknown")}


def _load_config(path: Path) -> tuple[dict[str, Any], TaskPoseRepairConfig]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return raw, TaskPoseRepairConfig.from_mapping(raw)


def _runtime(repo_root: Path, backend: str, input_hash: str, formal_locked: bool) -> dict[str, Any]:
    protected = {}
    for relative in (
        "outputs/ik_graph/graph_summary.json",
        "outputs/ik_graph/nodes.parquet",
        "outputs/ik_graph/edges.parquet",
        "outputs/ik_feasibility_ablation_stage1_6/stage1_6_report.md",
        "outputs/ik_feasibility_ablation_stage1_6/ablation_matrix.csv",
    ):
        path = repo_root / relative
        if path.exists():
            protected[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "stage": "stage_1_7",
        "backend": backend,
        "formal_constraints_locked_before_run": formal_locked,
        "input_tcp_sha256": input_hash,
        "python": sys.version,
        "platform": platform.platform(),
        "cwd": str(repo_root),
        "ros2_moveit_required_for_formal_graph": True,
        "collision_validation_method": COLLISION_METHOD,
        "ccd_status": UNAVAILABLE,
        "clearance_status": UNAVAILABLE,
        "protected_stage_outputs_sha256": protected,
    }


def _empty_graph_summary(config: TaskPoseRepairConfig, *, input_hash: str, pose_report: Any, backend: str, reason: str, waypoint_count: int) -> dict[str, Any]:
    return {
        "schema_version": "1.0", "scene_id": "open_arch_181_stage17", "stage": "stage_1_7",
        "formal_constraints_locked_before_run": True, "input_tcp_sha256": input_hash, "waypoint_count": waypoint_count,
        "repair_windows": {str(w["window_id"]): {"start_waypoint": int(w["start_waypoint"]), "end_waypoint": int(w["end_waypoint"])} for w in config.windows},
        "task_pose_statistics": {"raw_candidate_count": pose_report.raw_candidate_count, "formal_candidate_count": pose_report.formal_candidate_count, "diagnostic_candidate_count": pose_report.diagnostic_candidate_count, "rejected_candidate_count": pose_report.rejected_candidate_count, "generation_truncated_waypoint_count": pose_report.generation_truncated_waypoint_count},
        "ik_statistics": {"raw_candidate_count": 0, "deduplicated_candidate_count": 0, "valid_candidate_count": 0, "zero_candidate_waypoint_count": waypoint_count, "backend_status": "not_run"},
        "edge_statistics": {"raw_transition_count": 0, "valid_transition_count": 0, "joint_step_rejected_count": 0, "interpolation_collision_count": 0, "ccd_collision_count": None},
        "solution": {"complete_path_exists": False, "selected_candidate_count": 0, "first_unreachable_waypoint": 0, "max_joint_step_deg": None, "total_joint_motion_rad": None, "modified_waypoint_count": None, "max_position_offset_mm": None, "mean_position_offset_mm": None, "max_standoff_offset_mm": None, "max_roll_offset_deg": None, "max_normal_error_deg": None, "repair_total_variation": None, "repair_second_difference_max": None, "reorientation_count": 0},
        "collision_validation": {"method": COLLISION_METHOD, "ccd_status": UNAVAILABLE, "clearance_status": UNAVAILABLE},
        "ruckig_validation": {"attempted": False, "success": None, "post_ruckig_collision_count": None, "max_velocity_ratio": None, "max_acceleration_ratio": None, "max_jerk_ratio": None},
        "formal_pass": False, "status": "fail:no_complete_safe_path", "backend_failure_reason": reason, "backend": backend,
    }


def _write_report(out_dir: Path, summary: dict[str, Any], config: TaskPoseRepairConfig, *, backend: str) -> None:
    solution = summary["solution"]
    ruckig = summary.get("ruckig_validation", {})
    lines = ["# Stage 1.7 工艺容差内 TCP 路径/姿态局部修复与连续 IK 联合优化", "", f"- backend: `{backend}`", f"- formal_pass: `{summary['formal_pass']}`", f"- complete_path_exists: `{solution['complete_path_exists']}`", f"- first_unreachable_waypoint: `{solution['first_unreachable_waypoint']}`", f"- selected_waypoint_count: `{solution.get('selected_candidate_count')}`", f"- max_joint_step_deg: `{solution.get('max_joint_step_deg')}`", f"- modified_waypoint_count: `{solution.get('modified_waypoint_count')}`", f"- max_position_offset_mm: `{solution.get('max_position_offset_mm')}`", f"- mean_position_offset_mm: `{solution.get('mean_position_offset_mm')}`", f"- max_standoff_offset_mm: `{solution.get('max_standoff_offset_mm')}`", f"- max_roll_offset_deg: `{solution.get('max_roll_offset_deg')}`", f"- repair_total_variation_normalized: `{solution.get('repair_total_variation')}`", f"- repair_second_difference_max_normalized: `{solution.get('repair_second_difference_max')}`", "", "## 约束与证据", "", f"- 正式位置平面偏移上限: `{config.formal_position_mm:g} mm`", f"- 正式喷距偏移上限: `{config.formal_standoff_mm:g} mm`", f"- 正式 roll 上限: `{config.formal_roll_deg:g} deg`", f"- 正式关节步长上限: `{config.max_joint_step_deg:g} deg`", f"- 碰撞方法: `{COLLISION_METHOD}`", f"- Bullet CCD: `{UNAVAILABLE}`", f"- clearance: `{UNAVAILABLE}`", f"- Ruckig attempted/success: `{ruckig.get('attempted')}` / `{ruckig.get('success')}`", f"- post-Ruckig collision count: `{ruckig.get('post_ruckig_collision_count')}`", f"- post-Ruckig FK/dynamics: `{ruckig.get('post_ruckig_fk_status')}` / `{ruckig.get('post_ruckig_dynamics_status')}`", "", "## 运行结论", "", str(summary.get("backend_failure_reason") or "程序已完成正式 Stage 1.7 证据链；详见 graph_summary.json。"), ""]
    (out_dir / "stage17_report.md").write_text("\n".join(lines), encoding="utf-8")


def _preflight(repo_root: Path, config_path: Path) -> dict[str, Any]:
    raw, config = _load_config(config_path)
    input_path = _resolve(repo_root, raw["inputs"]["tcp_pose_csv"])
    rows, input_hash = read_pose_csv(input_path, config.waypoint_count_expected)
    frames, nominal_q, _ = recover_surface_frames(rows, config.nominal_standoff_m, config.tcp_points_to_wall)
    layers, pose_report = generate_task_pose_candidates(frames, nominal_q, config)
    out_dir = _resolve(repo_root, config.output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    pose_records = [candidate.to_record() for layer in layers for candidate in layer]
    pose_file = _write_parquet_or_json(out_dir / "task_pose_candidates.parquet", pose_records)
    _write_json(out_dir / "rejected_task_pose_candidates.json", {"schema_version": "1.0", "formal_constraints_locked_before_run": True, "rejected": [candidate.to_record() for layer in layers for candidate in layer if not candidate.formal_constraint_pass], "truncation": pose_report.truncation})
    _write_json(out_dir / "rejected_ik_candidates.json", {"schema_version": "1.0", "status": "not_run_backend_unavailable", "diagnostic_only": True, "records": []})
    _write_json(out_dir / "minimum_required_relaxation.json", {"status": "not_run_backend_unavailable", "formal_pass": False, "diagnostic_only": True, "reason": "A minimum relaxation cannot be established without re-solving IK and PlanningScene in the MoveIt2 backend.", "position_plane_offset_mm": None, "standoff_offset_mm": None, "roll_offset_deg": None, "normal_error_deg": None, "max_joint_step_deg": None, "collision_status": "not_run"})
    _write_csv(out_dir / "diagnostic_tolerance_sweep.csv", [{"axis": axis, "status": "not_run_backend_unavailable", "diagnostic_only": True, "formal_pass": False, "first_complete_path": None} for axis in ("position_plane_offset", "standoff_offset", "roll_offset", "normal_error")])
    _write_csv(out_dir / "layer_reachability.csv", [{"waypoint_id": i, "task_pose_candidate_count": len(layer), "formal_task_pose_candidate_count": sum(c.formal_constraint_pass and not c.diagnostic_only for c in layer), "ik_candidate_count": 0, "reachable": False, "reason": "moveit2_backend_not_run"} for i, layer in enumerate(layers)])
    _write_json(out_dir / "repair_window_summary.json", {str(w["window_id"]): {"start_waypoint": int(w["start_waypoint"]), "end_waypoint": int(w["end_waypoint"]), "task_pose_candidate_count": sum(len(layers[i]) for i in range(int(w["start_waypoint"]), int(w["end_waypoint"]) + 1))} for w in config.windows})
    summary = _empty_graph_summary(config, input_hash=input_hash, pose_report=pose_report, backend="preflight_no_ros2_moveit", reason="MoveIt2 PlanningScene, FK, dynamics and post-Ruckig were not run; formal IK graph is intentionally not fabricated.", waypoint_count=len(rows))
    summary["task_pose_file"] = pose_file
    _write_json(out_dir / "runtime_environment.json", _runtime(repo_root, "preflight_no_ros2_moveit", input_hash, True))
    _write_json(out_dir / "graph_summary.json", summary)
    _write_report(out_dir, summary, config, backend="preflight_no_ros2_moveit")
    return summary


def _run_ros(repo_root: Path, config_path: Path, *, stage18_run_dir: Path | None = None, run_id: str | None = None) -> dict[str, Any]:  # pragma: no cover - requires ROS2/MoveIt2
    import rclpy
    from moveit.planning import MoveItPy
    from moveit.core.robot_state import RobotState
    from geometry_msgs.msg import Pose
    import plan_closed_contour_moveit as bridge

    raw, config = _load_config(config_path)
    poses_path = _resolve(repo_root, raw["inputs"]["tcp_pose_csv"])
    seeds_path = _resolve(repo_root, raw["inputs"]["seed_joint_csv"])
    rows, input_hash = read_pose_csv(poses_path, config.waypoint_count_expected)
    frames, nominal_q, _ = recover_surface_frames(rows, config.nominal_standoff_m, config.tcp_points_to_wall)
    task_layers, pose_report = generate_task_pose_candidates(frames, nominal_q, config)
    seeds_rows = list(csv.DictReader(seeds_path.open(newline="", encoding="utf-8")))
    seeds = np.asarray([[float(row[f"q{i}"]) for i in range(1, 7)] for row in seeds_rows], dtype=float)
    out_dir = _resolve(repo_root, config.output_directory)
    lifecycle_path = out_dir / "lifecycle.log"
    if stage18_run_dir is not None:
        _stage18_lifecycle(lifecycle_path, "algorithm_start", run_id=run_id, output_directory=str(out_dir))
    rclpy.init()
    try:
        node = rclpy.create_node("stage17_pose_repair")
        moveit = MoveItPy(node_name="stage17_pose_repair")
        group = str(raw["robot"]["group_name"])
        ee_link = str(raw["robot"]["ee_link"])
        bridge.preflight_moveit_runtime(moveit, group, ee_link, True)
        poses = bridge.load_tcp_poses(poses_path)
        normals = bridge.load_tcp_normals(poses_path)
        environment_count = bridge.apply_collision_environment(moveit, poses, normals, config.nominal_standoff_m, "base_link", 0.040, 1.10, 1, False, True, True, -0.20)
        psm = moveit.get_planning_scene_monitor()
        ik_layers: list[list[dict[str, Any]]] = []
        rejected: list[dict[str, Any]] = []
        state = RobotState(moveit.get_robot_model())
        for waypoint, task_layer in enumerate(task_layers):
            layer: list[dict[str, Any]] = []
            seed_requests = [seeds[waypoint]]
            if waypoint > 0:
                seed_requests.append(seeds[waypoint - 1])
            for task in task_layer:
                if task.diagnostic_only or not task.formal_constraint_pass:
                    continue
                pose = Pose()
                pose.position.x, pose.position.y, pose.position.z = map(float, task.repaired_tcp_position_m)
                pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = map(float, task.repaired_quaternion_xyzw)
                unique_q: list[np.ndarray] = []
                for seed_index, seed in enumerate(seed_requests):
                    state.set_joint_group_positions(group, seed)
                    state.update()
                    solved = bool(state.set_from_ik(group, pose, ee_link, 0.20))
                    if not solved:
                        rejected.append({"waypoint_id": waypoint, "task_pose_candidate_id": task.task_pose_candidate_id, "reason": "ik_solver_failure", "seed_index": seed_index})
                        continue
                    state.update()
                    q = np.asarray(state.get_joint_group_positions(group), dtype=float)
                    q = nearest_equivalent_joint_positions(q, seed)
                    if any(np.max(np.abs(((q - old + np.pi) % (2 * np.pi)) - np.pi)) < 1.0e-5 for old in unique_q):
                        continue
                    transform = bridge._transform_matrix(state.get_global_link_transform(ee_link))
                    actual = transform[:3, 3]
                    position_error = float(np.linalg.norm(actual - task.repaired_tcp_position_m))
                    tool_axis = transform[:3, 2] / max(np.linalg.norm(transform[:3, 2]), 1.0e-12)
                    normal_error = float(np.rad2deg(np.arccos(np.clip(np.dot(tool_axis, task.target_normal), -1.0, 1.0))))
                    actual_standoff = float(np.dot(task.repaired_surface_point_m - actual, task.target_normal))
                    standoff_error = actual_standoff - task.actual_standoff_m
                    with psm.read_only() as scene:
                        colliding, collision_summary = bridge._state_collision_summary(scene, state, group)
                    valid = not colliding and position_error <= config.fk_position_error_m and abs(standoff_error) <= config.formal_standoff_mm / 1000.0 and normal_error <= config.formal_normal_deg
                    ik_candidate_id = f"wp{waypoint:03d}:{task.task_pose_candidate_id.split(':')[-1]}:ik{len(layer):02d}"
                    record = {"waypoint_id": waypoint, "candidate_id": ik_candidate_id, "task_pose_candidate_id": task.task_pose_candidate_id, "ik_candidate_id": ik_candidate_id, "q_rad": q.tolist(), "q_unwrapped_rad": q.tolist(), "position_error_m": position_error, "normal_error_deg": normal_error, "standoff_error_m": standoff_error, "tcp_standoff_error_m": standoff_error, "node_collision": bool(colliding), "collision_summary": str(collision_summary), "formal_constraint_pass": bool(valid), "diagnostic_only": False, "valid": bool(valid), "u": [task.tangential_offset_mm, task.longitudinal_offset_mm, task.standoff_offset_mm, task.roll_offset_deg], "node_cost": task.node_cost, "tangential_offset_mm": task.tangential_offset_mm, "longitudinal_offset_mm": task.longitudinal_offset_mm, "standoff_offset_mm": task.standoff_offset_mm, "roll_offset_deg": task.roll_offset_deg, "repaired_tcp_position_m": task.repaired_tcp_position_m.tolist(), "nominal_tcp_position_m": task.nominal_tcp_position_m.tolist(), "actual_position_offset_mm": task.actual_position_offset_mm}
                    if valid:
                        unique_q.append(q)
                        layer.append(record)
                    else:
                        rejected.append({**record, "reason": ";".join([name for name, fail in (("node_collision", colliding), ("position_error", position_error > config.fk_position_error_m), ("standoff_error", abs(standoff_error) > config.formal_standoff_mm / 1000.0), ("normal_error", normal_error > config.formal_normal_deg)) if fail])})
                if len(layer) >= int(raw["candidate_generation"]["max_ik_candidates_per_task_pose"]):
                    break
            ik_layers.append(layer)
        edge_cache: dict[tuple[str, str], dict[str, Any]] = {}
        all_edges: list[dict[str, Any]] = []
        def edge_lookup(source: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
            key = (str(source["ik_candidate_id"]), str(target["ik_candidate_id"]))
            if key not in edge_cache:
                edge_cache[key] = evaluate_transition(source, target, max_joint_step_deg=config.max_joint_step_deg, interpolation_step_deg=config.collision_interpolation_step_deg, collision_checker=lambda q: _collision_at(bridge, psm, state, group, q))
                edge_cache[key].update({"repair_first_difference": float(np.linalg.norm(np.asarray(target["u"]) - np.asarray(source["u"]))), "roll_first_difference": abs(float(target["roll_offset_deg"]) - float(source["roll_offset_deg"])) / max(config.formal_roll_deg, 1e-12), "standoff_first_difference": abs(float(target["standoff_offset_mm"]) - float(source["standoff_offset_mm"])) / max(config.formal_standoff_mm, 1e-12), "adjacent_tcp_repaired_change_m": float(np.linalg.norm(np.asarray(target.get("repaired_tcp_position_m", target.get("q_rad")), dtype=float) - np.asarray(source.get("repaired_tcp_position_m", source.get("q_rad")), dtype=float))) if "repaired_tcp_position_m" in source and "repaired_tcp_position_m" in target else None, "adjacent_standoff_change_mm": abs(float(target["standoff_offset_mm"]) - float(source["standoff_offset_mm"])), "adjacent_roll_change_deg": abs(float(target["roll_offset_deg"]) - float(source["roll_offset_deg"]))})
                all_edges.append(edge_cache[key])
            return edge_cache[key]
        solution = solve_second_order_dp(ik_layers, edge_lookup, cost_weights=config.cost_weights)
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_parquet_or_json(out_dir / "task_pose_candidates.parquet", [c.to_record() for layer in task_layers for c in layer])
        node_records = [node for layer in ik_layers for node in layer]
        node_file = _write_parquet_or_json(out_dir / "ik_nodes.parquet", node_records)
        edge_file = _write_parquet_or_json(out_dir / "transition_edges.parquet", all_edges)
        _write_json(out_dir / "rejected_task_pose_candidates.json", {"truncation": pose_report.truncation})
        _write_json(out_dir / "rejected_ik_candidates.json", {"records": rejected})
        diagnostic_status = "not_applicable_formal_path_exists" if solution["found"] else "not_run_backend_unavailable"
        _write_json(out_dir / "minimum_required_relaxation.json", {"status": diagnostic_status, "formal_pass": False, "diagnostic_only": True, "reason": "Formal path exists; independent relaxation is not required." if solution["found"] else "A minimum relaxation cannot be established without a complete backend run.", "position_plane_offset_mm": None, "standoff_offset_mm": None, "roll_offset_deg": None, "normal_error_deg": None, "max_joint_step_deg": None, "collision_status": "not_applicable" if solution["found"] else "not_run"})
        _write_csv(out_dir / "diagnostic_tolerance_sweep.csv", [{"axis": axis, "status": diagnostic_status, "diagnostic_only": True, "formal_pass": False, "first_complete_path": None} for axis in ("position_plane_offset", "standoff_offset", "roll_offset", "normal_error")])
        _write_csv(out_dir / "layer_reachability.csv", [{"waypoint_id": i, "task_pose_candidate_count": len(task_layers[i]), "ik_candidate_count": len(ik_layers[i]), "reachable": bool(ik_layers[i])} for i in range(len(ik_layers))])
        post_ruckig = {"attempted": False, "success": None, "post_ruckig_collision_count": None, "max_velocity_ratio": None, "max_acceleration_ratio": None, "max_jerk_ratio": None}
        if solution["found"]:
            selected = [next(node for node in ik_layers[i] if node["ik_candidate_id"] == solution["candidate_ids"][i]) for i in range(len(solution["candidate_ids"]))]
            _write_parquet_or_json(out_dir / "selected_joint_path.parquet", [{"waypoint_id": n["waypoint_id"], "ik_candidate_id": n["ik_candidate_id"], "q_rad": n["q_rad"]} for n in selected])
            _write_parquet_or_json(out_dir / "selected_task_pose_path.parquet", [{"waypoint_id": n["waypoint_id"], "task_pose_candidate_id": n["task_pose_candidate_id"]} for n in selected])
            task_by_id = {candidate.task_pose_candidate_id: candidate for layer in task_layers for candidate in layer}
            selected_tasks = [task_by_id[n["task_pose_candidate_id"]] for n in selected]
            selected_poses = []
            for task in selected_tasks:
                pose = Pose()
                pose.position.x, pose.position.y, pose.position.z = map(float, task.repaired_tcp_position_m)
                pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = map(float, task.repaired_quaternion_xyzw)
                selected_poses.append(pose)
            q_path = np.asarray([n["q_rad"] for n in selected], dtype=float)
            for index in range(1, len(q_path)):
                q_path[index] = nearest_equivalent_joint_positions(q_path[index], q_path[index - 1])
            post_ruckig = _post_ruckig_validation_stage17(bridge, moveit, group, ee_link, q_path, selected_poses, np.asarray([task.target_normal for task in selected_tasks], dtype=float), config, raw, environment_count, out_dir)
        summary = _empty_graph_summary(config, input_hash=input_hash, pose_report=pose_report, backend="ros2_moveit2_planning_scene", reason="", waypoint_count=len(rows))
        selected_metrics: dict[str, Any] = {}
        if solution["found"]:
            selected_tasks = [task_by_id[n["task_pose_candidate_id"]] for n in selected]
            selected_q = np.asarray([n["q_rad"] for n in selected], dtype=float)
            for index in range(1, len(selected_q)):
                selected_q[index] = nearest_equivalent_joint_positions(selected_q[index], selected_q[index - 1])
            steps = np.rad2deg(np.max(np.abs(np.diff(selected_q, axis=0)), axis=1))
            u = np.asarray([[t.tangential_offset_mm / config.formal_position_mm, t.longitudinal_offset_mm / config.formal_position_mm, t.standoff_offset_mm / config.formal_standoff_mm, t.roll_offset_deg / config.formal_roll_deg] for t in selected_tasks], dtype=float)
            second = np.diff(u, n=2, axis=0) if len(u) >= 3 else np.zeros((0, 4))
            selected_metrics = {"max_joint_step_deg": float(np.max(steps)) if len(steps) else 0.0, "total_joint_motion_rad": float(np.sum(np.abs(np.diff(selected_q, axis=0)))), "modified_waypoint_count": int(sum(not task.is_nominal for task in selected_tasks)), "max_position_offset_mm": float(max(task.actual_position_offset_mm for task in selected_tasks)), "mean_position_offset_mm": float(np.mean([task.actual_position_offset_mm for task in selected_tasks])), "max_standoff_offset_mm": float(max(abs(task.standoff_offset_mm) for task in selected_tasks)), "max_roll_offset_deg": float(max(abs(task.roll_offset_deg) for task in selected_tasks)), "max_normal_error_deg": float(max(task.normal_error_deg for task in selected_tasks)), "repair_total_variation": float(np.sum(np.linalg.norm(np.diff(u, axis=0), axis=1))) if len(u) >= 2 else 0.0, "repair_second_difference_max": float(np.max(np.linalg.norm(second, axis=1))) if len(second) else 0.0}
        summary.update({"status": "pass" if solution["found"] and post_ruckig.get("success") else ("fail:post_ruckig_validation" if solution["found"] else "fail:no_complete_safe_path"), "ik_statistics": {"raw_candidate_count": len(node_records), "deduplicated_candidate_count": len(node_records), "valid_candidate_count": sum(bool(n["valid"]) for n in node_records), "zero_candidate_waypoint_count": sum(not layer for layer in ik_layers)}, "edge_statistics": {"raw_transition_count": len(all_edges), "valid_transition_count": sum(bool(e.get("valid")) for e in all_edges), "joint_step_rejected_count": sum(e.get("reject_reason") == "joint_step_gate" for e in all_edges), "interpolation_collision_count": sum(e.get("reject_reason") == "interpolated_collision" for e in all_edges), "ccd_collision_count": None}, "solution": {**summary["solution"], **{"complete_path_exists": bool(solution["found"]), "selected_candidate_count": len(solution["candidate_ids"]), "first_unreachable_waypoint": solution["first_unreachable_waypoint"], "reorientation_count": 0, **selected_metrics}}, "ruckig_validation": post_ruckig, "formal_pass": bool(solution["found"] and post_ruckig.get("success")), "discrete_solution": solution, "node_file": node_file, "edge_file": edge_file, "environment_object_count": environment_count})
        _write_json(out_dir / "runtime_environment.json", _runtime(repo_root, "ros2_moveit2_planning_scene", input_hash, True))
        _write_json(out_dir / "repair_window_summary.json", {str(w["window_id"]): {"start_waypoint": int(w["start_waypoint"]), "end_waypoint": int(w["end_waypoint"])} for w in config.windows})
        _write_json(out_dir / "graph_summary.json", summary)
        _write_report(out_dir, summary, config, backend="ros2_moveit2_planning_scene")
        if stage18_run_dir is not None:
            _stage18_lifecycle(lifecycle_path, "algorithm_complete", formal_pass=bool(summary.get("formal_pass")))
            _stage18_lifecycle(lifecycle_path, "all_outputs_written")
            from tools.audit_stage17_reproducibility import finalize_stage18_run

            finalization = finalize_stage18_run(repo_root=repo_root, run_dir=out_dir, run_id=run_id or "unknown")
            if not finalization.get("run_complete", False):
                _stage18_lifecycle(lifecycle_path, "run_complete_atomic_write", status="failed", reason=finalization.get("reason"))
                raise RuntimeError(f"Stage 1.8 finalization failed: {finalization.get('reason')}")
            _stage18_lifecycle(lifecycle_path, "all_outputs_reread")
            _stage18_lifecycle(lifecycle_path, "run_complete_atomic_write")
        return summary
    finally:
        if stage18_run_dir is not None:
            _stage18_lifecycle(lifecycle_path, "ros2_node_destroy_begin", status="not_explicitly_owned_by_stage17")
            _stage18_lifecycle(lifecycle_path, "executor_stop", status="not_explicitly_owned_by_stage17")
            _stage18_lifecycle(lifecycle_path, "context_shutdown_begin")
        try:
            rclpy.shutdown()
        finally:
            if stage18_run_dir is not None:
                _stage18_lifecycle(lifecycle_path, "context_shutdown_complete")
                _stage18_lifecycle(lifecycle_path, "moveitpy_destructor_pending", status="awaiting_python_scope_release")


def _collision_at(bridge: Any, psm: Any, state: Any, group: str, q: np.ndarray) -> tuple[bool, str]:
    state.set_joint_group_positions(group, q)
    state.update()
    with psm.read_only() as scene:
        return bridge._state_collision_summary(scene, state, group)


def _post_ruckig_validation_stage17(bridge: Any, moveit: Any, group: str, ee_link: str, q_path: np.ndarray, target_poses: list[Any], target_normals: np.ndarray, config: TaskPoseRepairConfig, raw_config: dict[str, Any], environment_count: int, out_dir: Path) -> dict[str, Any]:  # pragma: no cover - requires ROS2/MoveIt2
    """Run native Ruckig followed by the repository's three post-checks."""
    from moveit.core.robot_state import RobotState

    result: dict[str, Any] = {"attempted": True, "success": False, "sample_count": None, "post_ruckig_collision_count": None, "post_ruckig_fk_status": "not_available", "post_ruckig_dynamics_status": "not_available", "max_velocity_ratio": None, "max_acceleration_ratio": None, "max_jerk_ratio": None}
    try:
        message = bridge.build_seed_joint_trajectory(q_path, [f"j{i}" for i in range(1, 7)])
        start_state = RobotState(moveit.get_robot_model())
        trajectory = bridge.build_and_smooth_moveit_trajectory(moveit, group, ee_link, start_state, message, 0.15, 0.15, True, time_parameterization="tcp_arclength", target_tcp_speed=0.003, zero_boundary_state=True)
        result["ruckig_success"] = True
        result["sample_count"] = _write_ruckig_joint_samples(out_dir / "post_ruckig_joint_trajectory.csv", trajectory)
    except Exception as exc:
        result["ruckig_error"] = f"{type(exc).__name__}: {exc}"
        result["ruckig_success"] = False
        return result
    dynamics_ok = collision_ok = fk_ok = False
    try:
        dynamics = bridge.validate_joint_dynamics(moveit, trajectory, group, out_dir / "post_ruckig_dynamics.csv")
        result.update({"post_ruckig_dynamics_status": dynamics.get("status", "not_available"), "max_velocity_ratio": dynamics.get("max_velocity_ratio"), "max_acceleration_ratio": dynamics.get("max_acceleration_ratio"), "max_jerk_ratio": dynamics.get("max_jerk_ratio")})
        dynamics_ok = result["post_ruckig_dynamics_status"] == "pass" and all(float(result[key]) <= 1.0 for key in ("max_velocity_ratio", "max_acceleration_ratio", "max_jerk_ratio"))
    except Exception as exc:
        result["post_ruckig_dynamics_error"] = f"{type(exc).__name__}: {exc}"
    try:
        collision = bridge.validate_trajectory_collision(moveit, trajectory, group, out_dir / "post_ruckig_collision.csv", stride=1, environment_object_count=environment_count, include_bottom_closure_collision=False, open_path=True, interpolation_step_deg=config.collision_interpolation_step_deg, include_tunnel_floor=True, tunnel_floor_z=-0.20)
        result.update({"post_ruckig_collision_count": collision.get("collision_count"), "post_ruckig_collision_status": collision.get("status", "not_available"), "post_ruckig_collision_check_type": COLLISION_METHOD})
        collision_ok = result["post_ruckig_collision_status"] == "pass" and float(result["post_ruckig_collision_count"]) == 0.0
    except Exception as exc:
        result["post_ruckig_collision_error"] = f"{type(exc).__name__}: {exc}"
    try:
        quality = bridge.validate_smoothed_trajectory_fk(moveit, trajectory, group, ee_link, target_poses, target_normals, config.nominal_standoff_m, out_dir / "post_ruckig_fk_quality.csv", max_normal_error_deg=config.formal_normal_deg, max_standoff_fraction=config.formal_standoff_mm / 1000.0 / config.nominal_standoff_m, max_speed_fluctuation=0.05, max_path_deviation=config.fk_position_error_m, stride=1, ruckig_smoothing_used=True, time_parameterization="tcp_arclength", fk_trace_path=out_dir / "post_ruckig_fk_trace.csv", tool_tcp_xyz="0 0 0.150", tool_tcp_rpy="0 0 0", tool_tcp_source="assumed_150mm_placeholder", tool_tcp_measured_by="simulation", tool_tcp_measured_date="not_applicable_virtual_design", tool_tcp_calibration_method="virtual_design_parameter", max_ik_joint_step_deg=config.max_joint_step_deg, production_joint_step_limit_deg=config.max_joint_step_deg, open_path=True, tcp_points_to_wall=config.tcp_points_to_wall, check_joint_continuity=True)
        result.update({"post_ruckig_fk_status": quality.get("status", "not_available"), "post_ruckig_max_position_offset_mm": quality.get("fk_path_deviation_max_mm"), "post_ruckig_max_standoff_error_mm": quality.get("fk_standoff_error_max_abs_mm"), "post_ruckig_max_normal_error_deg": quality.get("fk_normal_error_max_deg"), "post_ruckig_max_velocity_ratio": result.get("max_velocity_ratio"), "post_ruckig_max_acceleration_ratio": result.get("max_acceleration_ratio"), "post_ruckig_max_jerk_ratio": result.get("max_jerk_ratio")})
        fk_ok = result["post_ruckig_fk_status"] == "pass"
    except Exception as exc:
        result["post_ruckig_fk_error"] = f"{type(exc).__name__}: {exc}"
    result["success"] = bool(result.get("ruckig_success") and dynamics_ok and collision_ok and fk_ok)
    return result


def generate_launch_description():  # pragma: no cover - executed by ROS2 launch
    """Supply the same robot model, SRDF, kinematics and limits as the bridge."""
    from launch import LaunchDescription
    from launch_ros.actions import Node
    from ament_index_python.packages import get_package_share_directory
    from moveit_configs_utils import MoveItConfigsBuilder

    bridge_share = get_package_share_directory("fr5_tunnel_moveit_bridge")
    moveit_config = (
        MoveItConfigsBuilder("fairino5_v6_robot", package_name="fairino5_v6_moveit2_config")
        .robot_description(file_path=f"{bridge_share}/config/fairino5_v6_spray_tcp.urdf.xacro", mappings={"tool_tcp_xyz": "0 0 0.150", "tool_tcp_rpy": "0 0 0"})
        .robot_description_semantic(file_path=f"{bridge_share}/config/fairino5_v6_spray_tcp.srdf")
        .joint_limits(file_path=f"{bridge_share}/config/joint_limits_with_jerk.yaml")
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )
    params = moveit_config.to_dict()
    params["planning_pipelines"] = {"pipeline_names": ["ompl"], "ompl": params.get("ompl", {})}
    params["plan_request_params"] = {"planning_attempts": 1, "planning_pipeline": "ompl", "max_velocity_scaling_factor": 0.15, "max_acceleration_scaling_factor": 0.15}
    return LaunchDescription([
        Node(executable=sys.executable, name="stage17_pose_repair", output="screen", emulate_tty=True, arguments=[str(Path(__file__).resolve()), "--ros-node"], parameters=[params, {"group_name": "fairino5_v6_group", "ee_link": "spray_tcp_link"}])
    ])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "stage17_pose_repair.yaml")
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--ros-node", action="store_true", help="run only inside the complete ROS2/MoveIt2 launch environment")
    parser.add_argument("--stage18-run-dir", type=Path, default=None, help="enable Stage 1.8 completion and teardown evidence")
    parser.add_argument("--run-id", default=None)
    args, _ = parser.parse_known_args()
    try:
        summary = _run_ros(args.repo_root.resolve(), args.config.resolve(), stage18_run_dir=args.stage18_run_dir.resolve() if args.stage18_run_dir else None, run_id=args.run_id) if args.ros_node else _preflight(args.repo_root.resolve(), args.config.resolve())
    except Exception as exc:
        if args.ros_node:
            raise
        # Preflight errors are emitted as evidence and retain a non-pass exit.
        out_dir = args.repo_root.resolve() / "outputs" / "ik_graph_stage17"
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_json(out_dir / "runtime_environment.json", {"stage": "stage_1_7", "backend": "preflight_failed", "error": f"{type(exc).__name__}: {exc}", "formal_constraints_locked_before_run": True, "collision_validation_method": COLLISION_METHOD, "ccd_status": UNAVAILABLE, "clearance_status": UNAVAILABLE})
        raise
    print(json.dumps(_safe(summary), ensure_ascii=False, indent=2))
    return 0 if summary.get("formal_pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
