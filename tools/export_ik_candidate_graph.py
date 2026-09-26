#!/usr/bin/env python3
"""Export and validate the Stage 0/1 181-point MoveIt2 IK candidate graph.

The file is both a normal Python entry point and a ROS2 launch file.  The
launch mode supplies robot_description, SRDF, kinematics and PlanningScene
parameters before starting the child process.  This avoids the bare-MoveItPy
failure mode where robot_description is absent.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
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

from src.graph_search_solver import solve_layered_graph
from src.ik_candidate_graph import (
    IKCandidate,
    LayeredIKGraph,
    TransitionEdge,
    circular_joint_delta,
    nearest_equivalent_joint_positions,
)


def _load_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Graph config must be a mapping: {path}")
    return data


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _read_seed_csv(path: Path) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No seeds in {path}")
    columns = [f"q{i}" for i in range(1, 7)]
    missing = [column for column in columns if column not in rows[0]]
    if missing:
        raise ValueError(f"{path} is missing seed columns: {missing}")
    return np.asarray([[float(row[column]) for column in columns] for row in rows], dtype=float)


def _require_181(poses_path: Path, seeds_path: Path, expected: int) -> None:
    with poses_path.open(newline="", encoding="utf-8") as handle:
        pose_count = sum(1 for _ in csv.DictReader(handle))
    with seeds_path.open(newline="", encoding="utf-8") as handle:
        seed_count = sum(1 for _ in csv.DictReader(handle))
    if pose_count != expected or seed_count != expected:
        raise ValueError(
            "Stage 0/1 requires exactly the authoritative open-arch 181-point inputs: "
            f"poses={pose_count}, seeds={seed_count}, expected={expected}."
        )


def _write_parquet(path: Path, records: list[dict[str, Any]]) -> str:
    if not records:
        raise RuntimeError(f"Refusing to write an empty Parquet file: {path}")
    try:
        import pyarrow as pa
        import pyarrow.parquet as parquet
    except Exception as exc:  # pragma: no cover - exercised in deployment
        raise RuntimeError(
            "pyarrow is required for real Parquet output; refusing to rename CSV or write an empty file: "
            f"{exc}"
        ) from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    parquet.write_table(pa.Table.from_pylist(_json_safe(records)), path)
    return str(getattr(pa, "__version__", "unknown"))


def _add_unique_seed(seeds: list[np.ndarray], value: np.ndarray) -> None:
    array = np.asarray(value, dtype=float)
    if not any(np.allclose(array, existing, atol=1e-9, rtol=0.0) for existing in seeds):
        seeds.append(array.copy())


def _make_candidate_layers(
    bridge: Any,
    moveit: Any,
    group_name: str,
    ee_link: str,
    poses: list[Any],
    normals: np.ndarray,
    seeds: np.ndarray,
    psm: Any,
    ik_timeout: float,
    max_position_error: float,
    max_standoff_error: float,
    max_normal_error_deg: float,
    stand_off: float,
    roll_sample_count: int,
    candidate_limit: int,
    seed_limit: int,
    dedup_tolerance: float,
) -> tuple[LayeredIKGraph, dict[str, Any]]:
    from moveit.core.robot_state import RobotState

    state = RobotState(moveit.get_robot_model())
    graph = LayeredIKGraph(
        waypoint_count=len(poses),
        metadata={
            "candidate_generation": "MoveIt2_set_from_ik_with_existing_seed_and_roll_helpers",
            "collision_check_type": "adaptive_discrete_interpolation",
            "ccd_status": "not_available",
            "clearance_status": "not_available",
        },
    )
    layer_diagnostics: list[dict[str, Any]] = []
    total_raw_ik = 0
    rejected_counter: Counter[str] = Counter()
    previous_valid: list[IKCandidate] = []

    def evaluate(q: np.ndarray, pose: Any, normal: np.ndarray) -> dict[str, Any]:
        state.set_joint_group_positions(group_name, q)
        state.update()
        transform = bridge._transform_matrix(state.get_global_link_transform(ee_link))
        target = np.asarray([pose.position.x, pose.position.y, pose.position.z], dtype=float)
        actual = np.asarray(transform[:3, 3], dtype=float)
        position_error = float(np.linalg.norm(actual - target))
        tool_z = np.asarray(transform[:3, 2], dtype=float)
        tool_z /= max(float(np.linalg.norm(tool_z)), 1e-12)
        normal_error = float(np.rad2deg(np.arccos(np.clip(np.dot(tool_z, normal), -1.0, 1.0))))
        wall = target + stand_off * normal
        # The authoritative path uses tcp_points_to_wall=True: normal points
        # from the virtual TCP toward the wall, matching the bridge FK gate.
        standoff_error = float(np.dot(wall - actual, normal) - stand_off)
        with psm.read_only() as scene:
            colliding, collision_summary = bridge._state_collision_summary(scene, state, group_name)
        reasons: list[str] = []
        if colliding:
            reasons.append("node_collision")
        if position_error > max_position_error:
            reasons.append("position_error")
        if abs(standoff_error) > max_standoff_error:
            reasons.append("standoff_error")
        if normal_error > max_normal_error_deg:
            reasons.append("normal_error")
        return {
            "position_error_m": position_error,
            "normal_error_deg": normal_error,
            "tcp_standoff_error_m": standoff_error,
            "node_collision": bool(colliding),
            "collision_summary": str(collision_summary),
            "valid": not reasons,
            "reject_reason": ";".join(reasons) if reasons else None,
        }

    for index, pose in enumerate(poses):
        seed_requests: list[np.ndarray] = []
        if previous_valid:
            for previous in previous_valid[:seed_limit]:
                for branch_seed in bridge._ik_seed_candidates(previous.q_rad, previous.q_rad):
                    _add_unique_seed(seed_requests, branch_seed)
        for branch_seed in bridge._candidate_collection_seed_candidates(seeds[index]):
            _add_unique_seed(seed_requests, branch_seed)
        seed_requests = seed_requests[: max(1, seed_limit * 12)]
        roll_requests = bridge._tool_axis_roll_pose_candidates(pose, max(1, roll_sample_count))
        unique: list[IKCandidate] = []
        raw_success_count = 0
        ik_failure_count = 0
        duplicate_count = 0
        for seed_index, seed in enumerate(seed_requests):
            for roll_deg, candidate_pose in roll_requests:
                state.set_joint_group_positions(group_name, seed)
                state.update()
                if not state.set_from_ik(group_name, candidate_pose, ee_link, ik_timeout):
                    ik_failure_count += 1
                    continue
                state.update()
                raw_success_count += 1
                q = np.asarray(state.get_joint_group_positions(group_name), dtype=float)
                q_unwrapped = nearest_equivalent_joint_positions(q, seed)
                metrics = evaluate(q_unwrapped, pose, normals[index])
                candidate = IKCandidate(
                    waypoint=index,
                    candidate_id=f"{index}:{len(unique)}",
                    q_rad=q,
                    q_unwrapped_rad=q_unwrapped,
                    position_error_m=metrics["position_error_m"],
                    normal_error_deg=metrics["normal_error_deg"],
                    tcp_standoff_error_m=metrics["tcp_standoff_error_m"],
                    roll_deg=float(roll_deg),
                    seed_source=f"seed_{seed_index}",
                    node_collision=metrics["node_collision"],
                    collision_summary=metrics["collision_summary"],
                    valid=metrics["valid"],
                    reject_reason=metrics["reject_reason"],
                )
                duplicate = next(
                    (
                        existing
                        for existing in unique
                        if float(np.max(np.abs(circular_joint_delta(candidate.q_rad - existing.q_rad))))
                        < dedup_tolerance
                    ),
                    None,
                )
                if duplicate is not None:
                    duplicate_count += 1
                    if candidate.valid and not duplicate.valid:
                        unique[unique.index(duplicate)] = candidate
                    continue
                if len(unique) < candidate_limit:
                    unique.append(candidate)
                else:
                    duplicate_count += 1
                    break
            if len(unique) >= candidate_limit:
                break
        for candidate in unique:
            graph.add_candidate(candidate)
            if not candidate.valid:
                rejected_counter[candidate.reject_reason or "invalid_node"] += 1
        valid_candidates = [candidate for candidate in unique if candidate.valid]
        previous_valid = valid_candidates
        layer_diagnostics.append(
            {
                "waypoint": index,
                "seed_request_count": len(seed_requests),
                "roll_request_count": len(roll_requests),
                "raw_ik_success_count": raw_success_count,
                "ik_failure_count": ik_failure_count,
                "deduplicated_candidate_count": len(unique),
                "valid_candidate_count": len(valid_candidates),
                "duplicate_or_quota_count": duplicate_count,
                "candidate_shortage": len(unique) < candidate_limit,
            }
        )
        total_raw_ik += raw_success_count

    return graph, {
        "layer_diagnostics": layer_diagnostics,
        "raw_ik_candidate_count": total_raw_ik,
        "node_rejection_counts": dict(rejected_counter),
    }


def _build_edges(
    bridge: Any,
    moveit: Any,
    graph: LayeredIKGraph,
    group_name: str,
    psm: Any,
    max_joint_step_deg: float,
    interpolation_step_deg: float,
) -> dict[str, Any]:
    edge_rejections: Counter[str] = Counter()
    collision_edge_details: list[dict[str, Any]] = []
    raw_count = 0
    valid_count = 0
    max_step_rad = np.deg2rad(max(interpolation_step_deg, 1e-6))

    from moveit.core.robot_state import RobotState

    state = RobotState(moveit.get_robot_model())
    for waypoint in range(graph.waypoint_count - 1):
        previous_layer = graph.layer(waypoint)
        current_layer = graph.layer(waypoint + 1)
        for previous in previous_layer:
            for current in current_layer:
                raw_count += 1
                aligned_current = nearest_equivalent_joint_positions(current.q_rad, previous.q_unwrapped_rad)
                delta = aligned_current - previous.q_unwrapped_rad
                wrapped = circular_joint_delta(delta)
                max_step = float(np.max(np.abs(delta)))
                reasons: list[str] = []
                collision_status = "not_checked"
                collision_summary = "not_checked"
                interpolation_count: int | None = None
                if not previous.valid or not current.valid:
                    reasons.append("node_invalid")
                elif max_step > np.deg2rad(max_joint_step_deg):
                    reasons.append("joint_step_gate")
                else:
                    interpolation_count = max(2, int(math.ceil(max_step / max_step_rad)) + 1)
                    for sample_index, alpha in enumerate(np.linspace(0.0, 1.0, interpolation_count)):
                        q = previous.q_unwrapped_rad + alpha * delta
                        state.set_joint_group_positions(group_name, q)
                        state.update()
                        with psm.read_only() as scene:
                            colliding, summary = bridge._state_collision_summary(scene, state, group_name)
                        if colliding:
                            reasons.append("interpolated_collision")
                            collision_status = "fail"
                            collision_summary = str(summary)
                            collision_edge_details.append(
                                {
                                    "from_waypoint": waypoint,
                                    "to_waypoint": waypoint + 1,
                                    "from_candidate_id": previous.candidate_id,
                                    "to_candidate_id": current.candidate_id,
                                    "alpha": float(alpha),
                                    "sample_index": sample_index,
                                    "summary": str(summary),
                                }
                            )
                            break
                    else:
                        collision_status = "pass"
                        collision_summary = "contacts=0"
                if reasons:
                    for reason in reasons:
                        edge_rejections[reason] += 1
                else:
                    valid_count += 1
                edge = TransitionEdge(
                    from_waypoint=waypoint,
                    to_waypoint=waypoint + 1,
                    from_candidate_id=previous.candidate_id,
                    to_candidate_id=current.candidate_id,
                    delta_q_wrapped_rad=wrapped,
                    delta_q_unwrapped_rad=delta,
                    max_joint_step_deg=float(np.rad2deg(np.max(np.abs(delta)))),
                    interpolation_count=interpolation_count,
                    collision_status=collision_status,
                    collision_summary=collision_summary,
                    ccd_status="not_available",
                    clearance_status="not_available",
                    min_clearance_m=None,
                    min_clearance_alpha=None,
                    tcp_distance_error_m=max(
                        abs(float(previous.tcp_standoff_error_m or 0.0)),
                        abs(float(current.tcp_standoff_error_m or 0.0)),
                    ),
                    normal_error_deg=max(
                        float(previous.normal_error_deg or 0.0),
                        float(current.normal_error_deg or 0.0),
                    ),
                    cost=float(np.rad2deg(np.max(np.abs(delta)))),
                    valid=not reasons,
                    reject_reason=";".join(reasons) if reasons else None,
                )
                graph.add_edge(edge)
    return {
        "raw_transition_edge_count": raw_count,
        "valid_transition_edge_count": valid_count,
        "edge_rejection_counts": dict(edge_rejections),
        "interpolated_collision_details": collision_edge_details[:200],
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _post_ruckig_validation(
    bridge: Any,
    moveit: Any,
    group_name: str,
    ee_link: str,
    poses: list[Any],
    normals: np.ndarray,
    q_path: np.ndarray,
    stand_off: float,
    environment_object_count: int,
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    from moveit.core.robot_state import RobotState

    validation: dict[str, Any] = {
        "ruckig_success": False,
        "post_ruckig_collision_count": None,
        "post_ruckig_fk_status": "not_available",
        "post_ruckig_dynamics_status": "not_available",
    }
    joint_names = [f"j{i}" for i in range(1, 7)]
    velocity_scaling = float(config.get("ruckig", {}).get("velocity_scaling", 0.15))
    acceleration_scaling = float(config.get("ruckig", {}).get("acceleration_scaling", 0.15))
    target_speed = float(config.get("ruckig", {}).get("target_tcp_speed_m_s", 0.003))
    try:
        trajectory_msg = bridge.build_seed_joint_trajectory(q_path, joint_names)
        start_state = RobotState(moveit.get_robot_model())
        trajectory = bridge.build_and_smooth_moveit_trajectory(
            moveit,
            group_name,
            ee_link,
            start_state,
            trajectory_msg,
            velocity_scaling,
            acceleration_scaling,
            True,
            time_parameterization="tcp_arclength",
            target_tcp_speed=target_speed,
            zero_boundary_state=True,
        )
        validation["ruckig_success"] = True
    except Exception as exc:
        validation["ruckig_error"] = f"{type(exc).__name__}: {exc}"
        return validation

    with tempfile.TemporaryDirectory(prefix="ik_graph_validation_") as temp_dir:
        temp = Path(temp_dir)
        try:
            dynamics = bridge.validate_joint_dynamics(
                moveit, trajectory, group_name, temp / "dynamics.csv"
            )
            validation["post_ruckig_dynamics_status"] = dynamics.get("status", "not_available")
            validation["max_velocity_utilization"] = dynamics.get("max_velocity_ratio")
            validation["max_acceleration_utilization"] = dynamics.get("max_acceleration_ratio")
            validation["max_jerk_utilization"] = dynamics.get("max_jerk_ratio")
        except Exception as exc:
            validation["post_ruckig_dynamics_status"] = "fail"
            validation["post_ruckig_dynamics_error"] = f"{type(exc).__name__}: {exc}"
        try:
            collision = bridge.validate_trajectory_collision(
                moveit,
                trajectory,
                group_name,
                temp / "collision.csv",
                stride=1,
                environment_object_count=environment_object_count,
                include_bottom_closure_collision=False,
                open_path=True,
                interpolation_step_deg=float(config["collision"]["max_interpolation_joint_step_deg"]),
                include_tunnel_floor=True,
                tunnel_floor_z=float(config["moveit"]["tunnel_floor_z_m"]),
            )
            validation["post_ruckig_collision_count"] = collision.get("collision_count")
            validation["post_ruckig_collision_status"] = collision.get("status", "not_available")
            validation["post_ruckig_collision_check_type"] = "adaptive_discrete_interpolation"
        except Exception as exc:
            validation["post_ruckig_collision_status"] = "fail"
            validation["post_ruckig_collision_error"] = f"{type(exc).__name__}: {exc}"
        try:
            quality = bridge.validate_smoothed_trajectory_fk(
                moveit,
                trajectory,
                group_name,
                ee_link,
                poses,
                normals,
                stand_off,
                temp / "quality.csv",
                max_normal_error_deg=float(config["ik"]["max_normal_error_deg"]),
                max_standoff_fraction=0.05,
                max_speed_fluctuation=0.05,
                max_path_deviation=0.006,
                stride=1,
                ruckig_smoothing_used=True,
                time_parameterization="tcp_arclength",
                fk_trace_path=temp / "fk_trace.csv",
                tool_tcp_xyz=config["moveit"]["tool_tcp_xyz"],
                tool_tcp_rpy=config["moveit"]["tool_tcp_rpy"],
                tool_tcp_source="assumed_150mm_placeholder",
                tool_tcp_measured_by="simulation",
                tool_tcp_measured_date="not_applicable_virtual_design",
                tool_tcp_calibration_method="virtual_design_parameter",
                max_ik_joint_step_deg=float(config["ik"]["max_joint_step_deg"]),
                production_joint_step_limit_deg=float(config["ik"]["max_joint_step_deg"]),
                open_path=True,
                tcp_points_to_wall=True,
                check_joint_continuity=True,
            )
            validation["post_ruckig_fk_status"] = quality.get("status", "not_available")
            for key in (
                "fk_normal_error_max_deg",
                "fk_normal_error_p95_deg",
                "fk_standoff_error_max_abs_mm",
                "fk_standoff_error_p95_abs_mm",
                "fk_tcp_speed_mean_m_s",
                "fk_tcp_speed_p05_p95_fluctuation",
            ):
                if key in quality:
                    validation[key] = quality[key]
            if "fk_standoff_error_max_abs_mm" in quality:
                validation["tcp_standoff_error_max_m"] = float(quality["fk_standoff_error_max_abs_mm"]) / 1000.0
            if "fk_standoff_error_p95_abs_mm" in quality:
                validation["tcp_standoff_error_p95_m"] = float(quality["fk_standoff_error_p95_abs_mm"]) / 1000.0
            if "fk_normal_error_max_deg" in quality:
                validation["tcp_normal_error_max_deg"] = float(quality["fk_normal_error_max_deg"])
            if "fk_normal_error_p95_deg" in quality:
                validation["tcp_normal_error_p95_deg"] = float(quality["fk_normal_error_p95_deg"])
        except Exception as exc:
            validation["post_ruckig_fk_status"] = "fail"
            validation["post_ruckig_fk_error"] = f"{type(exc).__name__}: {exc}"
    return validation


def _run_export(config_path: Path, root: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    path_cfg = config["path"]
    moveit_cfg = config["moveit"]
    ik_cfg = config["ik"]
    collision_cfg = config["collision"]
    output_cfg = config["output"]
    poses_path = _resolve(root, path_cfg["tcp_pose_csv"])
    seeds_path = _resolve(root, path_cfg["seed_joint_csv"])
    expected = int(path_cfg["expected_waypoint_count"])
    _require_181(poses_path, seeds_path, expected)
    seeds = _read_seed_csv(seeds_path)

    import rclpy
    from moveit.planning import MoveItPy
    from moveit.core.robot_state import RobotState
    import plan_closed_contour_moveit as bridge

    rclpy.init()
    try:
        node = rclpy.create_node("ik_graph_exporter")
        moveit = MoveItPy(node_name="ik_graph_exporter")
        group_name = str(moveit_cfg["group_name"])
        ee_link = str(moveit_cfg["ee_link"])
        bridge.preflight_moveit_runtime(moveit, group_name, ee_link, True)
        poses = bridge.load_tcp_poses(poses_path)
        normals = bridge.load_tcp_normals(poses_path)
        if len(poses) != expected or len(seeds) != expected:
            raise ValueError("The authoritative 181-point pose and seed counts do not match.")
        environment_count = bridge.apply_collision_environment(
            moveit,
            poses,
            normals,
            float(moveit_cfg["stand_off_m"]),
            str(moveit_cfg["base_frame"]),
            float(moveit_cfg["wall_thickness_m"]),
            float(moveit_cfg["tunnel_y_thickness_m"]),
            1,
            False,
            True,
            True,
            float(moveit_cfg["tunnel_floor_z_m"]),
        )
        psm = moveit.get_planning_scene_monitor()
        graph, candidate_diag = _make_candidate_layers(
            bridge,
            moveit,
            group_name,
            ee_link,
            poses,
            normals,
            seeds,
            psm,
            float(ik_cfg["timeout_s"]),
            float(ik_cfg["max_position_error_m"]),
            float(moveit_cfg["stand_off_m"])
            * float(ik_cfg.get("max_standoff_error_fraction", 0.05)),
            float(ik_cfg["max_normal_error_deg"]),
            float(moveit_cfg["stand_off_m"]),
            int(ik_cfg["roll_sample_count"]),
            int(ik_cfg["max_candidates_per_waypoint"]),
            int(ik_cfg["candidate_seed_limit"]),
            float(ik_cfg["dedup_max_circular_delta_rad"]),
        )
        edge_diag = _build_edges(
            bridge,
            moveit,
            graph,
            group_name,
            psm,
            float(ik_cfg["max_joint_step_deg"]),
            float(collision_cfg["max_interpolation_joint_step_deg"]),
        )
        solution = solve_layered_graph(graph)
        output_dir = _resolve(root, output_cfg["directory"])
        output_dir.mkdir(parents=True, exist_ok=True)
        nodes_path = output_dir / str(output_cfg["nodes_parquet"])
        edges_path = output_dir / str(output_cfg["edges_parquet"])
        summary_path = output_dir / str(output_cfg["summary_json"])
        rejected_path = output_dir / str(output_cfg["rejected_json"])
        pyarrow_version = _write_parquet(nodes_path, graph.node_records())
        pyarrow_version_edges = _write_parquet(edges_path, graph.edge_records())

        ruckig = {
            "ruckig_success": "not_run",
            "post_ruckig_collision_count": None,
            "post_ruckig_fk_status": "not_available",
            "post_ruckig_dynamics_status": "not_available",
        }
        if solution.found and solution.q_path_rad is not None:
            q_path: list[np.ndarray] = []
            for index, candidate_id in enumerate(solution.candidate_ids):
                candidate = next(item for item in graph.layer(index) if item.candidate_id == candidate_id)
                q = candidate.q_rad.copy() if not q_path else nearest_equivalent_joint_positions(candidate.q_rad, q_path[-1])
                q_path.append(q)
            ruckig = _post_ruckig_validation(
                bridge,
                moveit,
                group_name,
                ee_link,
                poses,
                normals,
                np.asarray(q_path, dtype=float),
                float(moveit_cfg["stand_off_m"]),
                int(environment_count),
                config,
                output_dir,
            )

        layer_counts = graph.layer_candidate_counts()
        rejected_candidates = {
            "schema_version": 1,
            "status": "complete_path" if solution.found else "no_complete_safe_path",
            "first_unreachable_waypoint": solution.first_unreachable_waypoint,
            "layer_candidate_counts": layer_counts,
            "previous_reachable_candidate_counts": solution.diagnostics.get(
                "previous_reachable_candidate_counts", []
            ),
            "candidate_generation": candidate_diag,
            "edge_generation": edge_diag,
            "rejected_nodes": [record for record in graph.node_records() if not record["valid"]],
            "rejected_edges": [record for record in graph.edge_records() if not record["valid"]],
        }
        _write_json(rejected_path, rejected_candidates)
        summary = {
            "schema_version": 1,
            "stage": "stage_0_1",
            "status": "pass" if solution.found else "fail:no_complete_safe_path",
            "validation_scope": "virtual_tcp_and_simplified_tunnel_environment_only",
            "input_tcp_pose_csv": str(poses_path),
            "input_seed_joint_csv": str(seeds_path),
            "legacy_720_point_inputs_used": False,
            "legacy_reorientation_segments_used": False,
            "waypoint_total_count": expected,
            "valid_waypoint_count": sum(bool(graph.layer(i)) and any(item.valid for item in graph.layer(i)) for i in range(expected)),
            "zero_ik_candidate_waypoint_count": sum(not graph.layer(i) for i in range(expected)),
            "raw_ik_candidate_count": candidate_diag["raw_ik_candidate_count"],
            "deduplicated_candidate_count": graph.node_count(False),
            "valid_candidate_count": graph.node_count(True),
            "node_collision_count": candidate_diag["node_rejection_counts"].get("node_collision", 0),
            "raw_transition_edge_count": edge_diag["raw_transition_edge_count"],
            "valid_transition_edge_count": edge_diag["valid_transition_edge_count"],
            "joint_jump_rejected_edge_count": edge_diag["edge_rejection_counts"].get("joint_step_gate", 0),
            "node_invalid_edge_count": edge_diag["edge_rejection_counts"].get("node_invalid", 0),
            "interpolated_collision_edge_count": edge_diag["edge_rejection_counts"].get("interpolated_collision", 0),
            "pose_error_rejected_candidate_count": candidate_diag["node_rejection_counts"].get("position_error", 0),
            "standoff_error_rejected_candidate_count": candidate_diag["node_rejection_counts"].get("standoff_error", 0),
            "normal_error_rejected_candidate_count": candidate_diag["node_rejection_counts"].get("normal_error", 0),
            "candidate_shortage_waypoint_count": sum(
                bool(item["candidate_shortage"]) for item in candidate_diag["layer_diagnostics"]
            ),
            "search_truncated": False,
            "first_unreachable_waypoint": solution.first_unreachable_waypoint,
            "complete_path_exists": solution.found,
            "total_path_cost": solution.total_cost,
            "collision_check_type": "adaptive_discrete_interpolation",
            "interpolation_joint_step_deg": collision_cfg["max_interpolation_joint_step_deg"],
            "ccd_status": "not_available",
            "ccd_collision_count": None,
            "clearance_status": "not_available",
            "min_clearance_m": None,
            "min_clearance_alpha": None,
            "max_joint_step_deg": max(
                (float(record["max_joint_step_deg"]) for record in graph.edge_records() if record["valid"]),
                default=None,
            ),
            "tcp_standoff_error_max_m": ruckig.get("tcp_standoff_error_max_m"),
            "tcp_standoff_error_p95_m": ruckig.get("tcp_standoff_error_p95_m"),
            "tcp_normal_error_max_deg": ruckig.get("tcp_normal_error_max_deg"),
            "tcp_normal_error_p95_deg": ruckig.get("tcp_normal_error_p95_deg"),
            "ruckig_success": ruckig.get("ruckig_success", "not_run"),
            "max_velocity_utilization": ruckig.get("max_velocity_utilization"),
            "max_acceleration_utilization": ruckig.get("max_acceleration_utilization"),
            "max_jerk_utilization": ruckig.get("max_jerk_utilization"),
            "post_ruckig_collision_count": ruckig.get("post_ruckig_collision_count"),
            "post_ruckig_collision_status": ruckig.get("post_ruckig_collision_status", "not_available"),
            "post_ruckig_fk_status": ruckig.get("post_ruckig_fk_status", "not_available"),
            "post_ruckig_dynamics_status": ruckig.get("post_ruckig_dynamics_status", "not_available"),
            "reorientation_count": 0,
            "parquet_engine": "pyarrow",
            "pyarrow_version_nodes": pyarrow_version,
            "pyarrow_version_edges": pyarrow_version_edges,
            "nodes_parquet": str(nodes_path),
            "edges_parquet": str(edges_path),
            "rejected_candidates_json": str(rejected_path),
            "graph_solution": solution.to_record(),
        }
        _write_json(summary_path, summary)
        return summary
    finally:
        try:
            rclpy.shutdown()
        except Exception:
            pass


def _run_child() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "ik_graph.yaml")
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    args, _unknown_ros_args = parser.parse_known_args()
    summary = _run_export(args.config, args.repo_root)
    print(json.dumps(_json_safe(summary), ensure_ascii=False, indent=2))
    raise SystemExit(0 if summary["status"] in {"pass", "fail:no_complete_safe_path"} else 1)


def generate_launch_description():  # pragma: no cover - executed by ROS2 launch
    from launch import LaunchDescription
    from launch_ros.actions import Node
    from ament_index_python.packages import get_package_share_directory
    from moveit_configs_utils import MoveItConfigsBuilder

    bridge_share = get_package_share_directory("fr5_tunnel_moveit_bridge")
    robot_xacro = f"{bridge_share}/config/fairino5_v6_spray_tcp.urdf.xacro"
    robot_srdf = f"{bridge_share}/config/fairino5_v6_spray_tcp.srdf"
    joint_limits = f"{bridge_share}/config/joint_limits_with_jerk.yaml"
    moveit_config = (
        MoveItConfigsBuilder("fairino5_v6_robot", package_name="fairino5_v6_moveit2_config")
        .robot_description(
            file_path=robot_xacro,
            mappings={"tool_tcp_xyz": "0 0 0.150", "tool_tcp_rpy": "0 0 0"},
        )
        .robot_description_semantic(file_path=robot_srdf)
        .joint_limits(file_path=joint_limits)
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )
    moveit_dict = moveit_config.to_dict()
    moveit_pipeline_params = {
        "planning_pipelines": {
            "pipeline_names": ["ompl"],
            "ompl": moveit_dict.get("ompl", {}),
        },
        "plan_request_params": {
            "planning_attempts": 1,
            "planning_pipeline": "ompl",
            "max_velocity_scaling_factor": 0.15,
            "max_acceleration_scaling_factor": 0.15,
        },
    }
    return LaunchDescription(
        [
            Node(
                executable=sys.executable,
                name="ik_graph_exporter",
                output="screen",
                emulate_tty=True,
                arguments=[str(Path(__file__).resolve()), "--ros-node", "--config", str(ROOT / "config" / "ik_graph.yaml"), "--repo-root", str(ROOT)],
                parameters=[
                    moveit_dict,
                    moveit_pipeline_params,
                    {
                        "group_name": "fairino5_v6_group",
                        "ee_link": "spray_tcp_link",
                    },
                ],
            )
        ]
    )


def main() -> None:
    if "--ros-node" in sys.argv:
        sys.argv.remove("--ros-node")
        _run_child()
    else:
        raise SystemExit(
            "Run this exporter with ROS2 launch so robot_description, SRDF, kinematics and PlanningScene are loaded: "
            "ros2 launch /mnt/c/Users/86198/Desktop/robotfucker/tools/export_ik_candidate_graph.py"
        )


if __name__ == "__main__":
    main()
