#!/usr/bin/env python3
"""Stage 1.6 MoveIt2 IK-feasibility ablation for the authoritative open arch.

This is diagnostic-only.  It deliberately reads the 181-point open-arch pair
under ``outputs/internal_wiper_moveit_inputs`` and writes a new evidence
directory.  It never modifies ``outputs/ik_graph`` and never introduces OFF
states, reorientation edges, or a GNN.

The direct-IK conditions use MoveIt2 RobotState.set_from_ik and then evaluate
the returned state with MoveIt FK and PlanningScene.  The position-only and
relaxed-orientation conditions use a real MoveIt2 PlanningComponent goal made
from PositionConstraint/OrientationConstraint messages.  These two methods
are kept explicit in the output so a planning result is not mistaken for a
strict six-DOF IK result.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


WAYPOINTS_87_94 = tuple(range(87, 95))
WAYPOINTS_67_68 = (67, 68)
GROUP_NAME = "fairino5_v6_group"
EE_LINK = "spray_tcp_link"
BASE_FRAME = "base_link"
STAND_OFF = 0.260
POSITION_TOLERANCE_M = 0.003
STANDOFF_TOLERANCE_M = 0.013
NORMAL_TOLERANCE_DEG = 10.0
MAX_JOINT_STEP_DEG = 20.0


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(v) for v in value]
    return value


def _read_pose_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 181:
        raise ValueError(f"Stage 1.6 requires exactly 181 open-arch rows, got {len(rows)}")
    return rows


def _read_seed_rows(path: Path) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    values = np.asarray([[float(row[f"q{i}"]) for i in range(1, 7)] for row in rows], dtype=float)
    if values.shape != (181, 6):
        raise ValueError(f"Stage 1.6 requires an 181x6 seed matrix, got {values.shape}")
    return values


def _pose_from_row(row: dict[str, str]):
    from geometry_msgs.msg import Pose

    pose = Pose()
    pose.position.x = float(row["x"])
    pose.position.y = float(row["y"])
    pose.position.z = float(row["z"])
    pose.orientation.x = float(row["qx"])
    pose.orientation.y = float(row["qy"])
    pose.orientation.z = float(row["qz"])
    pose.orientation.w = float(row["qw"])
    norm = math.sqrt(sum(float(v) ** 2 for v in (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)))
    if norm < 1e-12:
        raise ValueError("zero TCP quaternion")
    pose.orientation.x /= norm
    pose.orientation.y /= norm
    pose.orientation.z /= norm
    pose.orientation.w /= norm
    return pose


def _normal_from_row(row: dict[str, str]) -> np.ndarray:
    normal = np.asarray([float(row["nx"]), float(row["ny"]), float(row["nz"])], dtype=float)
    return normal / max(float(np.linalg.norm(normal)), 1e-12)


def _pose_matrix(bridge: Any, pose: Any) -> np.ndarray:
    return bridge._pose_to_matrix(pose)


def _shift_pose(bridge: Any, pose: Any, delta: np.ndarray):
    shifted = deepcopy(pose)
    shifted.position.x += float(delta[0])
    shifted.position.y += float(delta[1])
    shifted.position.z += float(delta[2])
    return shifted


def _roll_pose(bridge: Any, pose: Any, roll_deg: float):
    base = _pose_matrix(bridge, pose)
    angle = math.radians(float(roll_deg))
    c, s = math.cos(angle), math.sin(angle)
    rz = np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    matrix = base.copy()
    matrix[:3, :3] = base[:3, :3] @ rz
    return bridge._pose_from_matrix(matrix, pose)


def _nearest_equivalent(bridge: Any, q: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return bridge.nearest_equivalent_joint_positions(np.asarray(q, dtype=float), np.asarray(reference, dtype=float))


def _seed_set(bridge: Any, seeds: np.ndarray, waypoint: int, expanded: bool) -> list[np.ndarray]:
    requests: list[np.ndarray] = []

    def add(value: np.ndarray) -> None:
        value = np.asarray(value, dtype=float).reshape(6)
        if not any(np.allclose(value, old, atol=1e-9, rtol=0.0) for old in requests):
            requests.append(value.copy())

    add(seeds[waypoint])
    for neighbour in (waypoint - 1, waypoint + 1):
        if 0 <= neighbour < len(seeds):
            add(seeds[neighbour])
    if expanded:
        for base in list(requests):
            for branch in bridge._ik_seed_candidates(base, base):
                add(branch)
    limit = max(1, int(os.environ.get("STAGE16_SEED_LIMIT", "3")))
    return requests[:limit]


def _pose_targets(bridge: Any, pose: Any, normal: np.ndarray, kind: str) -> list[tuple[str, Any, np.ndarray, float]]:
    """Return (variant, pose, position_delta, roll_deg) requests."""

    if kind == "full_constraints":
        return [("base", pose, np.zeros(3), 0.0)]
    if kind == "tcp_roll_scan":
        roll_count = max(3, int(os.environ.get("STAGE16_ROLL_COUNT", "25")))
        values = np.linspace(-180.0, 180.0, roll_count, endpoint=True)
        return [(f"roll_{value:+.1f}deg", _roll_pose(bridge, pose, float(value)), np.zeros(3), float(value)) for value in values]
    if kind == "standoff_scan":
        # The TCP target itself is shifted along the supplied wall normal.  The
        # resulting signed error is still measured against the unmodified base
        # target, so this condition isolates stand-off sensitivity.
        values = (-0.020, -0.010, 0.0, 0.010, 0.020)
        return [(f"standoff_{value * 1000.0:+.0f}mm", _shift_pose(bridge, pose, value * normal), value * normal, 0.0) for value in values]
    if kind == "tcp_position_perturbation":
        values = [np.zeros(3)]
        for axis in range(3):
            for sign in (-1.0, 1.0):
                delta = np.zeros(3)
                delta[axis] = sign * 0.005
                values.append(delta)
        return [(f"xyz_{delta[0] * 1000:+.0f}_{delta[1] * 1000:+.0f}_{delta[2] * 1000:+.0f}mm", _shift_pose(bridge, pose, delta), delta, 0.0) for delta in values]
    if kind in {"collision_off", "increased_single_point_budget"}:
        return [("base", pose, np.zeros(3), 0.0)]
    raise ValueError(f"unexpected direct-IK condition {kind}")


def _evaluate_state(
    bridge: Any,
    state: Any,
    psm: Any,
    q: np.ndarray,
    target_pose: Any,
    base_pose: Any,
    normal: np.ndarray,
    check_collision: bool,
    enforce_normal: bool = True,
    max_normal_error_deg: float = NORMAL_TOLERANCE_DEG,
    enforce_standoff: bool = True,
) -> dict[str, Any]:
    state.set_joint_group_positions(GROUP_NAME, q)
    state.update()
    transform = bridge._transform_matrix(state.get_global_link_transform(EE_LINK))
    actual = np.asarray(transform[:3, 3], dtype=float)
    base_target = np.asarray([base_pose.position.x, base_pose.position.y, base_pose.position.z], dtype=float)
    target = np.asarray([target_pose.position.x, target_pose.position.y, target_pose.position.z], dtype=float)
    position_error = float(np.linalg.norm(actual - target))
    target_rotation = bridge._pose_to_matrix(target_pose)[:3, :3]
    relative_rotation = target_rotation.T @ np.asarray(transform[:3, :3], dtype=float)
    orientation_error = float(
        np.degrees(
            np.arccos(
                np.clip((float(np.trace(relative_rotation)) - 1.0) * 0.5, -1.0, 1.0)
            )
        )
    )
    tool_z = np.asarray(transform[:3, 2], dtype=float)
    tool_z /= max(float(np.linalg.norm(tool_z)), 1e-12)
    normal_error = float(np.degrees(np.arccos(np.clip(float(tool_z @ normal), -1.0, 1.0))))
    # Compare against the active diagnostic target.  For the stand-off sweep
    # that target is intentionally shifted along the wall normal; using the
    # unperturbed base target here would reject every deliberate stand-off
    # variant by construction.
    signed_standoff_error = float((target - actual) @ normal)
    colliding = False
    collision_summary = "not_checked"
    with psm.read_only() as scene:
        colliding, collision_summary = bridge._state_collision_summary(scene, state, GROUP_NAME)
    reasons: list[str] = []
    if position_error > POSITION_TOLERANCE_M:
        reasons.append("position_gate")
    if enforce_normal and normal_error > max_normal_error_deg:
        reasons.append("normal_gate")
    if enforce_standoff and abs(signed_standoff_error) > STANDOFF_TOLERANCE_M:
        reasons.append("standoff_gate")
    if check_collision and colliding:
        reasons.append("node_collision")
    return {
        "position_error_m": position_error,
        "orientation_error_deg": orientation_error,
        "normal_error_deg": normal_error,
        "spray_distance_error_m": signed_standoff_error,
        "collision": bool(colliding),
        "collision_summary": str(collision_summary),
        "accepted_by_condition": not reasons,
        "reject_reasons": reasons,
        "q_rad": np.asarray(q, dtype=float).tolist(),
    }


def _position_constraints(target_pose: Any, orientation_pose: Any | None, position_tolerance_m: float, orientation_tolerance_deg: float | None):
    from moveit_msgs.msg import Constraints, OrientationConstraint, PositionConstraint
    from shape_msgs.msg import SolidPrimitive

    constraints = Constraints()
    position = PositionConstraint()
    position.header.frame_id = BASE_FRAME
    position.link_name = EE_LINK
    primitive = SolidPrimitive()
    primitive.type = SolidPrimitive.SPHERE
    primitive.dimensions = [float(position_tolerance_m)]
    center = deepcopy(target_pose)
    center.orientation.x = 0.0
    center.orientation.y = 0.0
    center.orientation.z = 0.0
    center.orientation.w = 1.0
    position.constraint_region.primitives = [primitive]
    position.constraint_region.primitive_poses = [center]
    position.weight = 1.0
    constraints.position_constraints = [position]
    if orientation_pose is not None and orientation_tolerance_deg is not None:
        orientation = OrientationConstraint()
        orientation.header.frame_id = BASE_FRAME
        orientation.link_name = EE_LINK
        orientation.orientation = orientation_pose.orientation
        tolerance = math.radians(float(orientation_tolerance_deg))
        orientation.absolute_x_axis_tolerance = tolerance
        orientation.absolute_y_axis_tolerance = tolerance
        orientation.absolute_z_axis_tolerance = tolerance
        orientation.weight = 1.0
        constraints.orientation_constraints = [orientation]
    return constraints


def _planning_candidate(bridge: Any, moveit: Any, planning_component: Any, plan_parameters: Any, psm: Any, seed: np.ndarray, base_pose: Any, target_pose: Any, normal: np.ndarray, orientation_tolerance_deg: float | None, planning_timeout_s: float) -> dict[str, Any]:
    from moveit.core.robot_state import RobotState

    start = RobotState(moveit.get_robot_model())
    start.set_joint_group_positions(GROUP_NAME, seed)
    start.update()
    planning_component.set_start_state(robot_state=start)
    planning_component.set_goal_state(
        motion_plan_constraints=[_position_constraints(target_pose, target_pose if orientation_tolerance_deg is not None else None, POSITION_TOLERANCE_M, orientation_tolerance_deg)]
    )
    started = time.monotonic()
    try:
        # MoveItPy's PlanRequestParameters constructor requires the internal
        # MoveItCpp handle and is not exposed portably across Jazzy builds.
        # The configured PlanningComponent.plan() path is the same path used
        # by the repository's strict bridge, so keep the runtime-default
        # planner parameters here and record wall-clock timeout evidence.
        result = planning_component.plan()
    except Exception as exc:
        elapsed = time.monotonic() - started
        return {"planning_success": False, "search_timed_out": bool(elapsed >= 0.95 * planning_timeout_s), "failure_stage": "planning_exception", "detail": repr(exc), "elapsed_s": elapsed}
    elapsed = time.monotonic() - started
    if not result or not getattr(result, "trajectory", None):
        return {"planning_success": False, "search_timed_out": bool(elapsed >= 0.95 * planning_timeout_s), "failure_stage": "planning_timeout" if elapsed >= 0.95 * planning_timeout_s else "planning_no_solution", "detail": "MoveIt2 PlanningComponent returned no trajectory", "elapsed_s": elapsed}
    trajectory = bridge.as_robot_trajectory_message(result.trajectory)
    points = list(trajectory.joint_trajectory.points)
    if not points:
        return {"planning_success": False, "search_timed_out": False, "failure_stage": "planning_empty_trajectory", "detail": "MoveIt2 returned an empty trajectory", "elapsed_s": elapsed}
    q = np.asarray(points[-1].positions, dtype=float)
    metrics = _evaluate_state(
        bridge,
        RobotState(moveit.get_robot_model()),
        psm,
        q,
        target_pose,
        base_pose,
        normal,
        True,
        enforce_normal=orientation_tolerance_deg is not None,
        max_normal_error_deg=float(orientation_tolerance_deg or NORMAL_TOLERANCE_DEG),
        enforce_standoff=orientation_tolerance_deg is not None,
    )
    metrics.update({"planning_success": True, "search_timed_out": False, "failure_stage": "accepted" if metrics["accepted_by_condition"] else metrics["reject_reasons"][0], "detail": "position/orientation constraint plan endpoint", "elapsed_s": elapsed, "planning_endpoint_q_rad": q.tolist()})
    return metrics


def _direct_attempts(bridge: Any, moveit: Any, psm: Any, poses: list[Any], normals: np.ndarray, seeds: np.ndarray, waypoint: int, condition: str, ik_timeout_s: float, expanded_seeds: bool, check_collision: bool) -> list[dict[str, Any]]:
    from moveit.core.robot_state import RobotState

    base_pose = poses[waypoint]
    requests = _pose_targets(bridge, base_pose, normals[waypoint], condition)
    seed_requests = _seed_set(bridge, seeds, waypoint, expanded_seeds)
    attempts: list[dict[str, Any]] = []
    for variant, target_pose, position_delta, roll_deg in requests:
        for seed_index, seed in enumerate(seed_requests):
            state = RobotState(moveit.get_robot_model())
            state.set_joint_group_positions(GROUP_NAME, seed)
            state.update()
            started = time.monotonic()
            solved = bool(state.set_from_ik(GROUP_NAME, target_pose, EE_LINK, float(ik_timeout_s)))
            elapsed = time.monotonic() - started
            row: dict[str, Any] = {
                "waypoint": waypoint,
                "condition": condition,
                "method": "moveit_robot_state_set_from_ik",
                "variant": variant,
                "seed_index": seed_index,
                "roll_deg": roll_deg,
                "position_delta_x_m": float(position_delta[0]),
                "position_delta_y_m": float(position_delta[1]),
                "position_delta_z_m": float(position_delta[2]),
                "ik_timeout_s": float(ik_timeout_s),
                "elapsed_s": elapsed,
                "ik_solver_success": solved,
                "search_timed_out": bool(not solved and elapsed >= 0.95 * ik_timeout_s),
                "collision_check_enabled": bool(check_collision),
            }
            if not solved:
                row.update({"candidate_counted": False, "accepted_by_condition": False, "failure_stage": "ik_timeout" if row["search_timed_out"] else "ik_solver_failure", "collision": None, "collision_summary": "not_reached"})
            else:
                state.update()
                q = _nearest_equivalent(bridge, np.asarray(state.get_joint_group_positions(GROUP_NAME), dtype=float), seed)
                metrics = _evaluate_state(bridge, state, psm, q, target_pose, base_pose, normals[waypoint], check_collision)
                if not check_collision:
                    metrics["accepted_by_condition"] = not any(reason != "node_collision" for reason in metrics["reject_reasons"])
                    metrics["reject_reasons"] = [reason for reason in metrics["reject_reasons"] if reason != "node_collision"]
                row.update(metrics)
                row["candidate_counted"] = True
                row["failure_stage"] = "accepted" if row["accepted_by_condition"] else row["reject_reasons"][0]
            attempts.append(_json_safe(row))
    return attempts


def _dedup_successes(attempts: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    for row in attempts:
        if not row.get("ik_solver_success") and not row.get("planning_success"):
            continue
        q = np.asarray(row.get("q_rad") or row.get("planning_endpoint_q_rad"), dtype=float)
        if any(float(np.max(np.abs(((q - np.asarray(old["q_rad"]) + np.pi) % (2.0 * np.pi)) - np.pi))) < 1e-5 for old in unique):
            continue
        item = dict(row)
        item["q_rad"] = q.tolist()
        unique.append(item)
    return unique


def _aggregate(waypoint: int, condition: str, attempts: list[dict[str, Any]], parameter: dict[str, Any]) -> dict[str, Any]:
    successes = [row for row in attempts if row.get("ik_solver_success") or row.get("planning_success")]
    unique = _dedup_successes(attempts)
    accepted = [row for row in unique if row.get("accepted_by_condition")]
    best = min(successes, key=lambda row: float(row.get("position_error_m", float("inf"))), default=None)
    stages: dict[str, int] = {}
    for row in attempts:
        stage = str(row.get("failure_stage", "unknown"))
        stages[stage] = stages.get(stage, 0) + 1
    return {
        "waypoint": waypoint,
        "condition": condition,
        "method": parameter.get("method"),
        "parameter": parameter,
        "attempt_count": len(attempts),
        "ik_found": bool(successes),
        "candidate_count": len(unique),
        "accepted_candidate_count": len(accepted),
        "failure_stage": "accepted" if accepted else (min(stages, key=stages.get) if stages else "no_attempts"),
        "failure_stage_counts": stages,
        "best_position_error_m": None if best is None else best.get("position_error_m"),
        "best_orientation_error_deg": None if best is None else best.get("orientation_error_deg"),
        "best_normal_error_deg": None if best is None else best.get("normal_error_deg"),
        "best_spray_distance_error_m": None if best is None else best.get("spray_distance_error_m"),
        "any_collision": bool(any(row.get("collision") is True for row in successes)),
        "search_timed_out": bool(any(row.get("search_timed_out") for row in attempts)),
        "best_candidate": best,
        "collision_check_type": "adaptive_discrete_interpolation",
        "ccd_status": "not_available",
        "clearance_status": "not_available",
        "validation_scope": "single_point_diagnostic_only",
        "dynamics_status": "not_run",
        "post_ruckig_status": "not_run",
    }


def _run_ros(repo_root: Path, out_dir: Path) -> dict[str, Any]:
    import rclpy
    from moveit.planning import MoveItPy
    import plan_closed_contour_moveit as bridge

    poses_path = repo_root / "outputs/internal_wiper_moveit_inputs/open_arch_tcp_poses_base_link.csv"
    seeds_path = repo_root / "outputs/internal_wiper_moveit_inputs/open_arch_seed_joints.csv"
    rows = _read_pose_rows(poses_path)
    poses = [_pose_from_row(row) for row in rows]
    normals = np.asarray([_normal_from_row(row) for row in rows], dtype=float)
    seeds = _read_seed_rows(seeds_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    try:
        node = rclpy.create_node("stage16_ik_feasibility_ablation")
        moveit = MoveItPy(node_name="stage16_ik_feasibility_ablation")
        bridge.preflight_moveit_runtime(moveit, GROUP_NAME, EE_LINK, True)
        environment_count = bridge.apply_collision_environment(
            moveit, poses, normals, STAND_OFF, BASE_FRAME, 0.040, 1.10, 1, False, True, True, -0.20
        )
        psm = moveit.get_planning_scene_monitor()
        all_conditions = [
            ("full_constraints", {"method": "moveit_robot_state_set_from_ik", "ik_timeout_s": 0.20, "seed_mode": "local_3_seeds", "collision_gate": True}),
            ("collision_off", {"method": "moveit_robot_state_set_from_ik", "ik_timeout_s": 0.20, "seed_mode": "local_3_seeds", "collision_gate": False}),
            ("position_only", {"method": "moveit_planning_component_position_constraint", "planning_timeout_s": 1.0, "position_tolerance_m": POSITION_TOLERANCE_M, "orientation_constraint": False}),
            ("relaxed_orientation", {"method": "moveit_planning_component_position_orientation_constraint", "planning_timeout_s": 1.0, "position_tolerance_m": POSITION_TOLERANCE_M, "orientation_tolerance_deg": 30.0}),
            ("tcp_roll_scan", {"method": "moveit_robot_state_set_from_ik", "ik_timeout_s": 0.20, "seed_mode": "local_3_seeds", "roll_samples": 25, "collision_gate": True}),
            ("standoff_scan", {"method": "moveit_robot_state_set_from_ik", "ik_timeout_s": 0.20, "seed_mode": "local_3_seeds", "standoff_offsets_m": [-0.02, -0.01, 0.0, 0.01, 0.02], "collision_gate": True}),
            ("tcp_position_perturbation", {"method": "moveit_robot_state_set_from_ik", "ik_timeout_s": 0.20, "seed_mode": "local_3_seeds", "position_offsets_m": 0.005, "collision_gate": True}),
            ("increased_single_point_budget", {"method": "moveit_robot_state_set_from_ik", "ik_timeout_s": 2.0, "seed_mode": "expanded_branch_seeds", "collision_gate": True}),
        ]
        requested_conditions = {
            item.strip() for item in os.environ.get("STAGE16_CONDITIONS", ",".join(row[0] for row in all_conditions)).split(",") if item.strip()
        }
        conditions = [row for row in all_conditions if row[0] in requested_conditions]
        if not conditions:
            raise ValueError(f"STAGE16_CONDITIONS selected no known conditions: {sorted(requested_conditions)}")
        attempts: list[dict[str, Any]] = []
        matrix: list[dict[str, Any]] = []
        planning_component = moveit.get_planning_component(GROUP_NAME)
        plan_parameters = None
        run_87_94 = os.environ.get("STAGE16_SKIP_87_94", "false").lower() != "true"
        for condition, parameter in conditions:
            for waypoint in WAYPOINTS_87_94 if run_87_94 else ():
                if condition in {"position_only", "relaxed_orientation"}:
                    condition_attempts: list[dict[str, Any]] = []
                    for seed_index, seed in enumerate(_seed_set(bridge, seeds, waypoint, False)):
                        row = _planning_candidate(bridge, moveit, planning_component, plan_parameters, psm, seed, poses[waypoint], poses[waypoint], normals[waypoint], 30.0 if condition == "relaxed_orientation" else None, 1.0)
                        row.update({"waypoint": waypoint, "condition": condition, "method": parameter["method"], "seed_index": seed_index, "ik_solver_success": False, "candidate_counted": bool(row.get("planning_success")), "collision_check_enabled": True})
                        condition_attempts.append(_json_safe(row))
                else:
                    condition_attempts = _direct_attempts(bridge, moveit, psm, poses, normals, seeds, waypoint, condition, float(parameter["ik_timeout_s"]), condition == "increased_single_point_budget", condition != "collision_off")
                attempts.extend(condition_attempts)
                matrix.append(_aggregate(waypoint, condition, condition_attempts, parameter))

        # A compact 67->68 pair scan.  This does not relax the production 20
        # degree gate; it only tests whether nearby pose variants expose a
        # continuous pair in the diagnostic candidate set.
        pair_rows: list[dict[str, Any]] = []
        pair_conditions = ["base", "tcp_roll_scan", "standoff_scan", "tcp_position_perturbation", "early_branch_seed"]
        run_pair = os.environ.get("STAGE16_SKIP_PAIR", "false").lower() != "true"
        formal_source_all: list[dict[str, Any]] = []
        formal_source_reachable: list[dict[str, Any]] = []
        formal_target_all: list[dict[str, Any]] = []
        if run_pair:
            import pyarrow.parquet as parquet

            formal_nodes = parquet.read_table(repo_root / "outputs/ik_graph/nodes.parquet").to_pylist()
            formal_source_all = [dict(row, ik_solver_success=True) for row in formal_nodes if int(row["waypoint"]) == 67 and bool(row.get("valid", True))]
            formal_target_all = [dict(row, ik_solver_success=True) for row in formal_nodes if int(row["waypoint"]) == 68 and bool(row.get("valid", True))]
            bottleneck_path = repo_root / "outputs/ik_graph_diagnostics/waypoint_67_68_bottleneck.json"
            if bottleneck_path.exists():
                reachable_ids = set(json.loads(bottleneck_path.read_text(encoding="utf-8")).get("reachable_source_candidate_ids", []))
                formal_source_reachable = [row for row in formal_source_all if str(row.get("candidate_id")) in reachable_ids]
            if not formal_source_reachable:
                formal_source_reachable = list(formal_source_all)
        for pair_condition in pair_conditions if run_pair else ():
            if pair_condition == "early_branch_seed":
                source_candidates = list(formal_source_all)
                target_seeds = _seed_set(bridge, seeds, 68, True)
                target_requests = [("base", poses[68], np.zeros(3), 0.0)]
                source_scope = "formal_waypoint_67_all_candidates_proxy_for_early_branch_switch"
                # Use the already collected formal target layer as the
                # downstream branch set.  The diagnostic question is whether
                # changing the source branch before 67 can reconnect to the
                # existing 68 layer; re-solving only three target seeds would
                # confound that question with a second search-budget limit.
                target_from_formal = True
            elif pair_condition == "base":
                source_candidates = list(formal_source_reachable)
                target_seeds = _seed_set(bridge, seeds, 68, False)
                target_requests = []
                source_scope = "formal_dp_reachable_waypoint_67_candidates"
                target_from_formal = True
            else:
                target_seeds = _seed_set(bridge, seeds, 68, False)
                target_kind = "full_constraints" if pair_condition == "base" else pair_condition
                target_requests = _pose_targets(bridge, poses[68], normals[68], target_kind)
                source_candidates = list(formal_source_reachable)
                source_scope = "formal_dp_reachable_waypoint_67_candidates"
                target_from_formal = False
            target_candidates: list[dict[str, Any]] = []
            if target_from_formal:
                target_candidates = list(formal_target_all)
            else:
                for variant, target_pose, delta, roll_deg in target_requests:
                    for seed_index, seed in enumerate(target_seeds):
                        from moveit.core.robot_state import RobotState

                        state = RobotState(moveit.get_robot_model())
                        state.set_joint_group_positions(GROUP_NAME, seed)
                        state.update()
                        started = time.monotonic()
                        solved = bool(state.set_from_ik(GROUP_NAME, target_pose, EE_LINK, 2.0))
                        elapsed = time.monotonic() - started
                        if not solved:
                            continue
                        q = _nearest_equivalent(bridge, np.asarray(state.get_joint_group_positions(GROUP_NAME), dtype=float), seed)
                        metrics = _evaluate_state(bridge, state, psm, q, target_pose, poses[68], normals[68], True)
                        metrics.update({"waypoint": 68, "variant": variant, "roll_deg": roll_deg, "position_delta_m": delta.tolist(), "elapsed_s": elapsed, "ik_solver_success": True})
                        target_candidates.append(metrics)
            pairs: list[dict[str, Any]] = []
            for source in source_candidates:
                for target in _dedup_successes(target_candidates):
                    aligned = _nearest_equivalent(bridge, np.asarray(target["q_rad"]), np.asarray(source["q_rad"]))
                    delta_deg = np.degrees(np.abs(aligned - np.asarray(source["q_rad"])))
                    pairs.append({"max_joint_step_deg": float(np.max(delta_deg)), "joint_step_deg": delta_deg.tolist(), "within_20deg_gate": bool(np.max(delta_deg) <= MAX_JOINT_STEP_DEG), "source_q_rad": source["q_rad"], "target_q_rad": target["q_rad"], "target_variant": target.get("variant")})
            best = min(pairs, key=lambda row: row["max_joint_step_deg"], default=None)
            pair_rows.append({"pair_condition": pair_condition, "source_candidate_scope": source_scope, "source_candidate_count": len(source_candidates), "target_candidate_count": len(_dedup_successes(target_candidates)), "pair_count": len(pairs), "best_max_joint_step_deg": None if best is None else best["max_joint_step_deg"], "within_20deg_gate": None if best is None else best["within_20deg_gate"], "best_pair": best, "production_gate_unchanged": True, "collision_check_type": "adaptive_discrete_interpolation", "ccd_status": "not_available", "clearance_status": "not_available"})

        fieldnames = sorted({key for row in attempts for key in row})
        with (out_dir / "attempts.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(attempts)
        matrix_fields = sorted({key for row in matrix for key in row if key != "best_candidate"})
        with (out_dir / "ablation_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=matrix_fields, extrasaction="ignore")
            writer.writeheader()
            for row in matrix:
                writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value for key, value in row.items()})
        with (out_dir / "67_68_branch_continuity.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = sorted({key for row in pair_rows for key in row if key != "best_pair"})
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in pair_rows:
                writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value for key, value in row.items()})
        summary = {
            "stage": "stage_1_6",
            "diagnostic_only": True,
            "authoritative_inputs": [str(poses_path), str(seeds_path)],
            "waypoints_87_94": list(WAYPOINTS_87_94),
            "waypoints_67_68": list(WAYPOINTS_67_68),
            "environment_object_count": environment_count,
            "production_joint_step_limit_deg": MAX_JOINT_STEP_DEG,
            "collision_check_type": "adaptive_discrete_interpolation",
            "ccd_status": "not_available",
            "clearance_status": "not_available",
            "dynamics_status": "not_run",
            "post_ruckig_status": "not_run",
            "formal_stage_0_1_graph_modified": False,
            "matrix": matrix,
            "branch_continuity": pair_rows,
            "interpretation_rules": {
                "collision_off_success": "collision_infeasible_or_collision_gate_caused_rejection; clearance remains unavailable",
                "relaxed_orientation_success": "original_orientation_or_normal_constraint_infeasible",
                "roll_scan_success": "TCP_roll_freedom_or_frame_branch mattered",
                "standoff_scan_success": "path_or_spray_distance needs joint optimization",
                "position_perturbation_success": "TCP path point needs local adjustment",
                "budget_only_success": "search insufficiency, not proof of physical infeasibility",
                "all_conditions_fail": "closer to true IK reachability failure, subject to timeout and backend limits",
            },
        }
        (out_dir / "stage1_6_summary.json").write_text(json.dumps(_json_safe(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return {"status": "completed", "out_dir": str(out_dir), "attempt_count": len(attempts), "matrix_rows": len(matrix), "pair_rows": len(pair_rows)}
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
        Node(executable=sys.executable, name="stage16_ik_feasibility_ablation", output="screen", emulate_tty=True, arguments=[str(Path(__file__).resolve()), "--ros-node"], parameters=[params, {"group_name": GROUP_NAME, "ee_link": EE_LINK}])
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ros-node", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--out-dir", type=Path, default=None)
    args, _ = parser.parse_known_args()
    if not args.ros_node:
        raise SystemExit("Run with: ros2 launch /mnt/c/Users/86198/Desktop/robotfucker/tools/phase16_ik_feasibility_ablation.py")
    out_dir = args.out_dir or Path(os.environ.get("STAGE16_OUT_DIR", str(args.repo_root / "outputs/ik_feasibility_ablation_stage1_6")))
    result = _run_ros(args.repo_root.resolve(), out_dir.resolve())
    print(json.dumps(_json_safe(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
