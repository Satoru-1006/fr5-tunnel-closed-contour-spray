#!/usr/bin/env python3
"""ROS/MoveIt-backed Stage 1.5 local candidate enrichment.

The enrichment is opt-in and writes a separate graph under
``outputs/ik_graph_diagnostics/local_candidate_enrichment``.  The formal
``outputs/ik_graph`` directory is never written by this command.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ik_candidate_graph import IKCandidate, LayeredIKGraph, circular_joint_delta, nearest_equivalent_joint_positions
from src.phase15_diagnostics import summarize_failure_attempts, threshold_sweep


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _add_seed(seeds: list[np.ndarray], value: np.ndarray) -> None:
    value = np.asarray(value, dtype=float).reshape(-1)
    if not any(np.allclose(value, old, atol=1e-9, rtol=0.0) for old in seeds):
        seeds.append(value.copy())


def _read_seeds(path: Path) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as handle:
        return np.asarray(
            [[float(row[f"q{i}"]) for i in range(1, 7)] for row in csv.DictReader(handle)], dtype=float
        )


def _read_base_nodes(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as parquet

    return parquet.read_table(path).to_pylist()


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as parquet

    path.parent.mkdir(parents=True, exist_ok=True)
    parquet.write_table(pa.Table.from_pylist(_json_safe(rows)), path)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _structured_seed_perturbations(base: np.ndarray) -> list[np.ndarray]:
    """Shoulder/elbow/wrist perturbations, kept local and deterministic."""

    result: list[np.ndarray] = []
    single_groups = {
        "shoulder": (0, 1),
        "elbow": (2,),
        "wrist": (3, 4, 5),
    }
    offsets = np.deg2rad((-30.0, -15.0, -7.5, 7.5, 15.0, 30.0))
    for indices in single_groups.values():
        for index in indices:
            for offset in offsets:
                candidate = base.copy()
                candidate[index] += offset
                result.append(candidate)
    for index_a, index_b in ((0, 2), (1, 2), (2, 3), (2, 4), (0, 5), (3, 5)):
        for offset in np.deg2rad((-15.0, 15.0)):
            candidate = base.copy()
            candidate[index_a] += offset
            candidate[index_b] -= offset
            result.append(candidate)
    return result


def _joint_limit_metrics(state: Any, group_name: str, q: np.ndarray) -> dict[str, Any]:
    try:
        within = bool(state.satisfies_bounds(group_name))
    except Exception:
        within = True
    if within:
        return {"joint_limit": False, "joint_limit_joint": None, "joint_limit_margin_rad": None}
    return {
        "joint_limit": True,
        "joint_limit_joint": "unknown_moveit_bound",
        "joint_limit_margin_rad": 0.0,
    }


def _enrich_window(
    bridge: Any,
    moveit: Any,
    group_name: str,
    ee_link: str,
    poses: list[Any],
    normals: np.ndarray,
    seeds: np.ndarray,
    psm: Any,
    start: int,
    end: int,
    timeout_s: float,
    roll_count: int,
    candidate_limit: int,
    seed_limit: int,
    stand_off: float,
    max_position_error: float,
    max_standoff_error: float,
    max_normal_error_deg: float,
    dedup_tolerance: float,
    max_waypoint_search_s: float,
) -> tuple[dict[int, list[IKCandidate]], dict[int, dict[str, Any]]]:
    from moveit.core.robot_state import RobotState

    state = RobotState(moveit.get_robot_model())
    layers: dict[int, list[IKCandidate]] = {}
    diagnostics: dict[int, dict[str, Any]] = {}
    previous_valid: list[IKCandidate] = []
    base_nodes = _read_base_nodes(ROOT / "outputs/ik_graph/nodes.parquet")
    base_by_wp: dict[int, list[dict[str, Any]]] = {}
    for row in base_nodes:
        base_by_wp.setdefault(int(row["waypoint"]), []).append(row)

    def evaluate(q: np.ndarray, pose: Any, normal: np.ndarray) -> dict[str, Any]:
        state.set_joint_group_positions(group_name, q)
        state.update()
        transform = bridge._transform_matrix(state.get_global_link_transform(ee_link))
        target = np.asarray([pose.position.x, pose.position.y, pose.position.z], dtype=float)
        actual = np.asarray(transform[:3, 3], dtype=float)
        position_error = float(np.linalg.norm(actual - target))
        tool_z = np.asarray(transform[:3, 2], dtype=float)
        tool_z /= max(float(np.linalg.norm(tool_z)), 1e-12)
        normal_error = float(np.rad2deg(np.arccos(np.clip(tool_z @ normal, -1.0, 1.0))))
        wall = target + stand_off * normal
        standoff_error = float((wall - actual) @ normal - stand_off)
        with psm.read_only() as scene:
            colliding, collision_summary = bridge._state_collision_summary(scene, state, group_name)
        limit = _joint_limit_metrics(state, group_name, q)
        reasons: list[str] = []
        if limit["joint_limit"]:
            reasons.append("joint_limit")
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
            "standoff_error_m": standoff_error,
            "tcp_standoff_error_m": standoff_error,
            "node_collision": bool(colliding),
            "collision_summary": str(collision_summary),
            "valid": not reasons,
            "reject_reasons": reasons,
            **limit,
        }

    for waypoint in range(start, end + 1):
        waypoint_started = time.monotonic()
        seed_requests: list[np.ndarray] = []
        # Preserve the formal seed route as a subset of the diagnostic search.
        _add_seed(seed_requests, seeds[waypoint])
        for neighbour in (waypoint - 1, waypoint + 1):
            if 0 <= neighbour < len(seeds):
                _add_seed(seed_requests, seeds[neighbour])
        for row in base_by_wp.get(waypoint, []):
            _add_seed(seed_requests, np.asarray(row["q_rad"], dtype=float))
        for candidate in previous_valid:
            _add_seed(seed_requests, candidate.q_unwrapped_rad)
            for branch_seed in bridge._ik_seed_candidates(candidate.q_unwrapped_rad, candidate.q_unwrapped_rad):
                _add_seed(seed_requests, branch_seed)
        expanded = list(seed_requests)
        for base in list(seed_requests):
            expanded.extend(_structured_seed_perturbations(base))
        seed_requests = []
        for seed in expanded:
            _add_seed(seed_requests, seed)
        seed_requests = seed_requests[: max(1, seed_limit)]
        roll_requests = bridge._tool_axis_roll_pose_candidates(poses[waypoint], max(1, roll_count))
        unique: list[IKCandidate] = []
        attempts: list[dict[str, Any]] = []
        truncation = False
        for seed_index, seed in enumerate(seed_requests):
            for roll_index, (roll_rad, candidate_pose) in enumerate(roll_requests):
                if max_waypoint_search_s > 0.0 and time.monotonic() - waypoint_started >= max_waypoint_search_s:
                    truncation = True
                    break
                started = time.monotonic()
                state.set_joint_group_positions(group_name, seed)
                state.update()
                ik_success = bool(state.set_from_ik(group_name, candidate_pose, ee_link, timeout_s))
                elapsed = time.monotonic() - started
                attempt: dict[str, Any] = {
                    "waypoint": waypoint,
                    "seed_id": f"seed_{seed_index}",
                    "seed_index": seed_index,
                    "roll_index": roll_index,
                    "roll_reparameterization_rad": float(roll_rad),
                    "ik_success": ik_success,
                    "elapsed_s": elapsed,
                    "timeout_s": timeout_s,
                }
                if not ik_success:
                    attempt["failure_class"] = "timeout" if elapsed >= 0.95 * timeout_s else "ik_solver_failure"
                    attempts.append(attempt)
                    continue
                state.update()
                q = np.asarray(state.get_joint_group_positions(group_name), dtype=float)
                q_unwrapped = nearest_equivalent_joint_positions(q, seed)
                metrics = evaluate(q_unwrapped, poses[waypoint], normals[waypoint])
                attempt.update(metrics)
                candidate = IKCandidate(
                    waypoint=waypoint,
                    candidate_id=f"{waypoint}:phase15:{len(unique)}",
                    q_rad=q,
                    q_unwrapped_rad=q_unwrapped,
                    position_error_m=metrics["position_error_m"],
                    normal_error_deg=metrics["normal_error_deg"],
                    tcp_standoff_error_m=metrics["tcp_standoff_error_m"],
                    roll_deg=float(roll_rad),
                    seed_source=f"phase15_seed_{seed_index}",
                    node_collision=metrics["node_collision"],
                    collision_summary=metrics["collision_summary"],
                    valid=metrics["valid"],
                    reject_reason=";".join(metrics["reject_reasons"]) or None,
                )
                duplicate = next(
                    (
                        old
                        for old in unique
                        if float(np.max(np.abs(circular_joint_delta(candidate.q_rad - old.q_rad)))) < dedup_tolerance
                    ),
                    None,
                )
                if duplicate is not None:
                    attempt["duplicate"] = True
                    attempt["unique"] = False
                elif len(unique) >= candidate_limit:
                    truncation = True
                    attempt["candidate_limit_reached"] = True
                    attempt["unique"] = False
                else:
                    unique.append(candidate)
                    attempt["unique"] = True
                    attempt["valid"] = candidate.valid
                attempts.append(attempt)
            if truncation:
                break
        layers[waypoint] = unique
        previous_valid = [item for item in unique if item.valid]
        diagnostics[waypoint] = summarize_failure_attempts(
            attempts,
            waypoint,
            timeout_s,
            len(roll_requests),
            source_phase_id="stage_1_5_local_enrichment",
        )
        diagnostics[waypoint]["seed_request_count"] = len(seed_requests)
        diagnostics[waypoint]["candidate_limit"] = candidate_limit
        diagnostics[waypoint]["candidate_limit_or_search_truncated"] = truncation
        diagnostics[waypoint]["search_truncated"] = bool(
            truncation and len(unique) < candidate_limit
        )
        diagnostics[waypoint]["max_waypoint_search_s"] = max_waypoint_search_s
        diagnostics[waypoint]["raw_success_count"] = sum(bool(row.get("ik_success")) for row in attempts)
        diagnostics[waypoint]["deduplicated_total_count"] = len(unique)
    return layers, diagnostics


def _run_ros(repo_root: Path, out_dir: Path) -> dict[str, Any]:
    import rclpy
    from moveit.planning import MoveItPy
    import plan_closed_contour_moveit as bridge
    from moveit.core.robot_state import RobotState

    poses_path = repo_root / "outputs/internal_wiper_moveit_inputs/open_arch_tcp_poses_base_link.csv"
    seeds_path = repo_root / "outputs/internal_wiper_moveit_inputs/open_arch_seed_joints.csv"
    nodes_path = repo_root / "outputs/ik_graph/nodes.parquet"
    config = {
        # The default remains the recommended 60-100 local window.  The two
        # focused windows used in this run are selected only through env vars
        # so the enrichment is still a configuration switch and formal Stage
        # 1 defaults remain untouched.
        "start": int(os.environ.get("PHASE15_START", "60")),
        "end": int(os.environ.get("PHASE15_END", "100")),
        "timeout_s": 0.10,
        "roll_sample_count": int(os.environ.get("PHASE15_ROLL_COUNT", "72")),
        "candidate_limit": int(os.environ.get("PHASE15_CANDIDATE_LIMIT", "128")),
        "seed_limit": int(os.environ.get("PHASE15_SEED_LIMIT", "128")),
        "max_position_error_m": 0.006,
        "max_normal_error_deg": 10.0,
        "stand_off_m": 0.260,
        "max_standoff_error_fraction": 0.05,
        "dedup_tolerance_rad": 1e-5,
        "max_joint_step_deg": float(os.environ.get("PHASE15_EDGE_MAX_STEP_DEG", "20.0")),
        "interpolation_step_deg": 0.5,
        "max_waypoint_search_s": float(os.environ.get("PHASE15_MAX_WAYPOINT_SEARCH_S", "30.0")),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    try:
        node = rclpy.create_node("phase15_local_candidate_enrichment")
        moveit = MoveItPy(node_name="phase15_local_candidate_enrichment")
        group_name = "fairino5_v6_group"
        ee_link = "spray_tcp_link"
        bridge.preflight_moveit_runtime(moveit, group_name, ee_link, True)
        poses = bridge.load_tcp_poses(poses_path)
        normals = bridge.load_tcp_normals(poses_path)
        seeds = _read_seeds(seeds_path)
        environment_count = bridge.apply_collision_environment(
            moveit,
            poses,
            normals,
            config["stand_off_m"],
            "base_link",
            0.040,
            1.10,
            1,
            False,
            True,
            True,
            -0.20,
        )
        psm = moveit.get_planning_scene_monitor()
        layers, diagnostics = _enrich_window(
            bridge,
            moveit,
            group_name,
            ee_link,
            poses,
            normals,
            seeds,
            psm,
            config["start"],
            config["end"],
            config["timeout_s"],
            config["roll_sample_count"],
            config["candidate_limit"],
            config["seed_limit"],
            config["stand_off_m"],
            config["max_position_error_m"],
            config["stand_off_m"] * config["max_standoff_error_fraction"],
            config["max_normal_error_deg"],
            config["dedup_tolerance_rad"],
            config["max_waypoint_search_s"],
        )
        local_graph = LayeredIKGraph(waypoint_count=181, metadata={"stage": "stage_1_5", "diagnostic_only": True})
        for waypoint in range(config["start"], config["end"] + 1):
            for candidate in layers[waypoint]:
                local_graph.add_candidate(candidate)
        from tools.export_ik_candidate_graph import _build_edges

        if os.environ.get("PHASE15_SKIP_EDGE_BUILD", "false").lower() == "true":
            edge_diag = {
                "status": "not_run_due_to_expanded_edge_budget",
                "raw_transition_edge_count": 0,
                "valid_transition_edge_count": 0,
                "edge_rejection_counts": {},
            }
        else:
            edge_diag = _build_edges(
                bridge,
                moveit,
                local_graph,
                group_name,
                psm,
                config["max_joint_step_deg"],
                config["interpolation_step_deg"],
            )
        local_nodes = local_graph.node_records()
        local_edges = local_graph.edge_records()
        _write_parquet(out_dir / "local_nodes.parquet", local_nodes)
        _write_parquet(out_dir / "local_edges.parquet", local_edges)
        _write_json(out_dir / "waypoint_87_94_candidate_failures.json", {
            "source_phase_id": "stage_1_5_local_enrichment",
            "window": [config["start"], config["end"]],
            "waypoints": [diagnostics[i] for i in range(max(87, config["start"]), min(94, config["end"]) + 1)]
            if config["end"] >= 87 and config["start"] <= 94 else [],
            "collision_check_type": "adaptive_discrete_interpolation",
            "ccd_status": "not_available",
            "clearance_status": "not_available",
        })
        counts = []
        for waypoint in range(config["start"], config["end"] + 1):
            counts.append({
                "source_phase_id": "stage_1_5_local_enrichment",
                "local_waypoint_id": waypoint,
                "csv_original_row_number": waypoint + 2,
                "candidate_count": len(layers[waypoint]),
                "valid_candidate_count": sum(c.valid for c in layers[waypoint]),
                "formal_candidate_count_before_enrichment": sum(int(row["waypoint"]) == waypoint for row in _read_base_nodes(nodes_path)),
            })
        with (out_dir / "candidate_count_by_waypoint.csv").open("w", newline="", encoding="utf-8") as handle:
            import csv as csv_module

            writer = csv_module.DictWriter(handle, fieldnames=list(counts[0]))
            writer.writeheader()
            writer.writerows(counts)
        _write_json(out_dir / "local_candidate_enrichment_summary.json", {
            "diagnostic_only": True,
            "formal_stage1_output_untouched": True,
            "source_phase_id": "stage_1_5_local_enrichment",
            "window": [config["start"], config["end"]],
            "configuration": config,
            "environment_object_count": environment_count,
            "layer_candidate_counts": {str(i): len(layers[i]) for i in layers},
            "failure_diagnostics": {str(i): diagnostics[i] for i in diagnostics},
            "edge_generation": edge_diag,
            "raw_transition_edge_count": local_graph.edge_count(False),
            "valid_transition_edge_count": local_graph.edge_count(True),
            "node_collision_count": sum(bool(c.node_collision) for layer in layers.values() for c in layer),
            "interpolated_collision_edge_count": edge_diag["edge_rejection_counts"].get("interpolated_collision", 0),
            "threshold_sweep": threshold_sweep(local_nodes, local_edges, 181),
            "collision_check_type": "adaptive_discrete_interpolation",
            "ccd_status": "not_available",
            "clearance_status": "not_available",
            "ruckig_status": "not_run",
        })
        return {"out_dir": str(out_dir), "status": "completed", "node_count": len(local_nodes), "edge_count": len(local_edges)}
    finally:
        try:
            rclpy.shutdown()
        except Exception:
            pass


def generate_launch_description():  # pragma: no cover - ROS launch entrypoint
    from launch import LaunchDescription
    from launch_ros.actions import Node
    from ament_index_python.packages import get_package_share_directory
    from moveit_configs_utils import MoveItConfigsBuilder

    bridge_share = get_package_share_directory("fr5_tunnel_moveit_bridge")
    moveit_config = (
        MoveItConfigsBuilder("fairino5_v6_robot", package_name="fairino5_v6_moveit2_config")
        .robot_description(
            file_path=f"{bridge_share}/config/fairino5_v6_spray_tcp.urdf.xacro",
            mappings={"tool_tcp_xyz": "0 0 0.150", "tool_tcp_rpy": "0 0 0"},
        )
        .robot_description_semantic(file_path=f"{bridge_share}/config/fairino5_v6_spray_tcp.srdf")
        .joint_limits(file_path=f"{bridge_share}/config/joint_limits_with_jerk.yaml")
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )
    params = moveit_config.to_dict()
    params["planning_pipelines"] = {"pipeline_names": ["ompl"], "ompl": params.get("ompl", {})}
    params["plan_request_params"] = {
        "planning_attempts": 1,
        "planning_pipeline": "ompl",
        "max_velocity_scaling_factor": 0.15,
        "max_acceleration_scaling_factor": 0.15,
    }
    return LaunchDescription([
        Node(
            executable=sys.executable,
            name="phase15_local_candidate_enrichment",
            output="screen",
            emulate_tty=True,
            arguments=[str(Path(__file__).resolve()), "--ros-node"],
            parameters=[params, {"group_name": "fairino5_v6_group", "ee_link": "spray_tcp_link"}],
        )
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ros-node", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--out-dir", type=Path, default=None)
    args, _ = parser.parse_known_args()
    if not args.ros_node:
        raise SystemExit("Run with: ros2 launch /mnt/c/Users/86198/Desktop/robotfucker/tools/phase15_local_candidate_enrichment.py")
    env_out = os.environ.get("PHASE15_OUT_DIR")
    result = _run_ros(
        args.repo_root.resolve(),
        (args.out_dir or Path(env_out) if env_out else args.repo_root / "outputs/ik_graph_diagnostics/local_candidate_enrichment").resolve(),
    )
    print(json.dumps(_json_safe(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
