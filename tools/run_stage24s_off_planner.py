#!/usr/bin/env python3
"""Native MoveIt2 planner for Stage 2.4S OFF transitions.

This file is intentionally independent from the Stage 2.4A-I graph builders.
It consumes a bounded JSON request made by the Windows orchestration runner,
plans every requested retract/reposition/approach transition in a real
MoveItPy PlanningScene, and persists enough state-level evidence for the
existing native FCL/Bullet edge validator to replay the returned trajectory.

The planner does not create or mutate any Spray-ON graph candidate.  Cartesian
retreat and approach states are OFF-only states.  Collision evidence remains
adaptive_discrete_interpolation; this is not continuous collision detection.
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
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "ros2_moveit_bridge") not in sys.path:
    sys.path.insert(0, str(ROOT / "ros2_moveit_bridge"))


COLLISION_METHOD = "adaptive_discrete_interpolation"
CCD_STATUS = "not_available"
CLEARANCE_STATUS = "not_available"
DEFAULT_GROUP = "fairino5_v6_group"
DEFAULT_EE = "spray_tcp_link"
BASE_FRAME = "base_link"
INTERPOLATION_STEP_DEG = 1.0
IK_TIMEOUT_S = 1.0
PLANNER_ATTEMPTS = 1


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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _stl_mesh(path: Path):
    from shape_msgs.msg import Mesh, MeshTriangle
    from geometry_msgs.msg import Point

    vertices: list[tuple[float, float, float]] = []
    vertex_map: dict[tuple[float, float, float], int] = {}
    triangles: list[MeshTriangle] = []
    facet_vertices: list[int] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.strip().split()
        if len(parts) != 4 or parts[0].lower() != "vertex":
            continue
        key = (float(parts[1]), float(parts[2]), float(parts[3]))
        index = vertex_map.get(key)
        if index is None:
            index = len(vertices)
            vertices.append(key)
            vertex_map[key] = index
        facet_vertices.append(index)
        if len(facet_vertices) == 3:
            triangle = MeshTriangle()
            triangle.vertex_indices = facet_vertices
            triangles.append(triangle)
            facet_vertices = []
    mesh = Mesh()
    mesh.vertices = [Point(x=x, y=y, z=z) for x, y, z in vertices]
    mesh.triangles = triangles
    return mesh


def apply_exact_collision_environment(moveit: Any, parts_dir: Path) -> dict[str, Any]:
    """Install the same compound ASCII-STL world object used by stage24 native."""

    from geometry_msgs.msg import Pose
    from moveit_msgs.msg import CollisionObject

    paths = sorted(parts_dir.glob("*.stl"))
    if not paths:
        raise FileNotFoundError(f"no ASCII STL collision parts under {parts_dir}")
    meshes = [_stl_mesh(path) for path in paths]
    obj = CollisionObject()
    obj.header.frame_id = BASE_FRAME
    obj.id = "horseshoe_collision_compound"
    obj.meshes = meshes
    poses = []
    for _ in meshes:
        pose = Pose()
        pose.position.x = 0.32
        pose.position.y = -0.05
        pose.position.z = -0.38
        pose.orientation.w = 1.0
        poses.append(pose)
    obj.mesh_poses = poses
    obj.operation = CollisionObject.ADD
    psm = moveit.get_planning_scene_monitor()
    with psm.read_write() as scene:
        scene.remove_all_collision_objects()
        scene.apply_collision_object(obj)
    return {
        "object_id": obj.id,
        "frame_id": BASE_FRAME,
        "part_count": len(paths),
        "parts": [str(path.resolve()) for path in paths],
        "mesh_pose_xyz": [0.32, -0.05, -0.38],
        "source": "Stage_2_3B tunnel_collision_parts compound world object",
    }


def pose_from_transform(transform: np.ndarray):
    from geometry_msgs.msg import Pose

    pose = Pose()
    pose.position.x = float(transform[0, 3])
    pose.position.y = float(transform[1, 3])
    pose.position.z = float(transform[2, 3])
    rotation = np.asarray(transform[:3, :3], dtype=float)
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rotation[2, 1] - rotation[1, 2]) / s
        qy = (rotation[0, 2] - rotation[2, 0]) / s
        qz = (rotation[1, 0] - rotation[0, 1]) / s
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        qw = (rotation[2, 1] - rotation[1, 2]) / s
        qx = 0.25 * s
        qy = (rotation[0, 1] + rotation[1, 0]) / s
        qz = (rotation[0, 2] + rotation[2, 0]) / s
    elif rotation[1, 1] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        qw = (rotation[0, 2] - rotation[2, 0]) / s
        qx = (rotation[0, 1] + rotation[1, 0]) / s
        qy = 0.25 * s
        qz = (rotation[1, 2] + rotation[2, 1]) / s
    else:
        s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        qw = (rotation[1, 0] - rotation[0, 1]) / s
        qx = (rotation[0, 2] + rotation[2, 0]) / s
        qy = (rotation[1, 2] + rotation[2, 1]) / s
        qz = 0.25 * s
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    pose.orientation.x = float(qx / norm)
    pose.orientation.y = float(qy / norm)
    pose.orientation.z = float(qz / norm)
    pose.orientation.w = float(qw / norm)
    return pose


def set_state(moveit: Any, group: str, q: list[float]):
    from moveit.core.robot_state import RobotState

    state = RobotState(moveit.get_robot_model())
    state.set_joint_group_positions(group, np.asarray(q, dtype=float))
    state.update()
    return state


def collision_check(psm: Any, state: Any, group: str) -> tuple[bool, str]:
    import plan_closed_contour_moveit as bridge

    with psm.read_only() as scene:
        return bridge._state_collision_summary(scene, state, group)


def state_record(moveit: Any, psm: Any, group: str, q: list[float], joint_bounds: list[list[float]] | None = None) -> dict[str, Any]:
    state = set_state(moveit, group, q)
    if joint_bounds is None:
        bounds = False
        bounds_error = "formal_joint_bounds_not_supplied"
    else:
        values = np.asarray(q, dtype=float)
        bounds = bool(len(values) == len(joint_bounds) and all(float(pair[0]) - 1.0e-10 <= float(value) <= float(pair[1]) + 1.0e-10 for value, pair in zip(values, joint_bounds)))
        bounds_error = None if bounds else "formal_bounded_revolute_limit_violation"
    colliding, summary = collision_check(psm, state, group)
    return {
        "q": [float(x) for x in q],
        "joint_limits_valid": bounds,
        "joint_limits_error": bounds_error,
        "collision_free": not colliding,
        "collision_summary": summary,
        "valid": bool(bounds and not colliding),
    }


def fk_transform(moveit: Any, group: str, ee_link: str, q: list[float]) -> np.ndarray:
    import plan_closed_contour_moveit as bridge

    state = set_state(moveit, group, q)
    return np.asarray(bridge._transform_matrix(state.get_global_link_transform(ee_link)), dtype=float)


def ik_shift(moveit: Any, group: str, ee_link: str, q_seed: list[float], transform: np.ndarray, direction: list[float], distance_m: float) -> tuple[list[float] | None, dict[str, Any]]:
    state = set_state(moveit, group, q_seed)
    pose = pose_from_transform(transform)
    direction_array = np.asarray(direction, dtype=float)
    direction_array /= max(float(np.linalg.norm(direction_array)), 1.0e-12)
    pose.position.x += float(distance_m * direction_array[0])
    pose.position.y += float(distance_m * direction_array[1])
    pose.position.z += float(distance_m * direction_array[2])
    started = time.monotonic()
    try:
        solved = bool(state.set_from_ik(group, pose, ee_link, IK_TIMEOUT_S))
    except Exception as exc:  # pragma: no cover - MoveIt runtime path
        return None, {"ik_solved": False, "failure_reason": "ik_exception", "detail": repr(exc), "elapsed_s": time.monotonic() - started}
    elapsed = time.monotonic() - started
    if not solved:
        return None, {"ik_solved": False, "failure_reason": "ik_no_solution", "elapsed_s": elapsed}
    state.update()
    q = np.asarray(state.get_joint_group_positions(group), dtype=float).tolist()
    return [float(x) for x in q], {"ik_solved": True, "elapsed_s": elapsed, "target_distance_m": distance_m}


def interpolate_states(q0: list[float], q1: list[float], step_deg: float = INTERPOLATION_STEP_DEG) -> list[list[float]]:
    delta = np.asarray(q1, dtype=float) - np.asarray(q0, dtype=float)
    steps = max(1, int(math.ceil(float(np.max(np.abs(delta))) / math.radians(step_deg))))
    return [(np.asarray(q0) + (np.asarray(q1) - np.asarray(q0)) * (i / steps)).tolist() for i in range(steps + 1)]


def validate_state_path(moveit: Any, psm: Any, group: str, states: list[list[float]], joint_bounds: list[list[float]]) -> dict[str, Any]:
    if not states:
        return {"valid": False, "failure_reason": "empty_state_path", "states_checked": 0}
    checked = 0
    for left, right in zip(states, states[1:]):
        for q in interpolate_states(left, right):
            record = state_record(moveit, psm, group, q, joint_bounds)
            checked += 1
            if not record["valid"]:
                return {"valid": False, "failure_reason": "state_path_invalid", "failure_state": record, "states_checked": checked}
    return {"valid": True, "states_checked": checked}


def plan_reposition(moveit: Any, group: str, q_start: list[float], q_goal: list[float]) -> tuple[list[list[float]] | None, dict[str, Any]]:
    planning_component = moveit.get_planning_component(group)
    start = set_state(moveit, group, q_start)
    goal = set_state(moveit, group, q_goal)
    planning_component.set_start_state(robot_state=start)
    planning_component.set_goal_state(robot_state=goal)
    started = time.monotonic()
    try:
        result = planning_component.plan()
    except Exception as exc:  # pragma: no cover - MoveIt runtime path
        return None, {"planning_success": False, "failure_reason": "planning_exception", "detail": repr(exc), "elapsed_s": time.monotonic() - started}
    elapsed = time.monotonic() - started
    if not result or not getattr(result, "trajectory", None):
        return None, {"planning_success": False, "failure_reason": "planning_no_solution", "elapsed_s": elapsed}
    import plan_closed_contour_moveit as bridge

    trajectory = bridge.as_robot_trajectory_message(result.trajectory)
    points = list(trajectory.joint_trajectory.points)
    if not points:
        return None, {"planning_success": False, "failure_reason": "planning_empty_trajectory", "elapsed_s": elapsed}
    states = [[float(x) for x in point.positions] for point in points]
    return states, {
        "planning_success": True,
        "elapsed_s": elapsed,
        "planner_api": "PlanningComponent.set_start_state(robot_state); set_goal_state(robot_state); plan()",
        "planning_pipeline_requested": "ompl",
        "planner_id_requested": "RRTConnectkConfigDefault",
        "planner_id_effective": "not_exposed_by_local_MoveItPy_plan_api",
        "planning_attempts_requested": PLANNER_ATTEMPTS,
        "planning_time_budget_s": "not_exposed_by_local_MoveItPy_plan_api",
        "random_seed_requested": os.environ.get("OMPL_RNG_SEED", "20260802"),
        "random_seed_effective": "not_exposed_by_local_MoveItPy_plan_api",
    }


def plan_transition(moveit: Any, psm: Any, group: str, ee_link: str, request: dict[str, Any], joint_bounds: list[list[float]]) -> dict[str, Any]:
    transition_id = int(request["transition_id"])
    pairs = list(request.get("endpoint_pairs", []))
    distances = [float(x) for x in request.get("retract_distances_mm", [])]
    direction = [float(x) for x in request["direction"]]
    attempts: list[dict[str, Any]] = []
    for pair in pairs:
        q_a = [float(x) for x in pair["q_a"]]
        q_b = [float(x) for x in pair["q_b"]]
        start_record = state_record(moveit, psm, group, q_a, joint_bounds)
        goal_record = state_record(moveit, psm, group, q_b, joint_bounds)
        if not start_record["valid"] or not goal_record["valid"]:
            attempts.append({"transition_id": transition_id, "endpoint_pair_rank": pair.get("rank"), "requested_distance_mm": None, "path_valid": False, "failure_reason": "endpoint_state_invalid", "start_state": start_record, "goal_state": goal_record})
            continue
        q_a_tf = fk_transform(moveit, group, ee_link, q_a)
        q_b_tf = fk_transform(moveit, group, ee_link, q_b)
        for distance_mm in distances:
            distance_m = distance_mm / 1000.0
            q_a_mid, ik_a_mid = ik_shift(moveit, group, ee_link, q_a, q_a_tf, direction, distance_m * 0.5)
            q_a_ret, ik_a_ret = ik_shift(moveit, group, ee_link, q_a, q_a_tf, direction, distance_m)
            q_b_mid, ik_b_mid = ik_shift(moveit, group, ee_link, q_b, q_b_tf, direction, distance_m * 0.5)
            q_b_ret, ik_b_ret = ik_shift(moveit, group, ee_link, q_b, q_b_tf, direction, distance_m)
            attempt = {
                "transition_id": transition_id,
                "endpoint_pair_rank": pair.get("rank"),
                "endpoint_candidate_a": pair.get("candidate_a"),
                "endpoint_candidate_b": pair.get("candidate_b"),
                "direction": direction,
                "requested_distance_mm": distance_mm,
                "achieved_distance_mm": None,
                "ik": {"a_mid": ik_a_mid, "a_retracted": ik_a_ret, "b_mid": ik_b_mid, "b_retracted": ik_b_ret},
                "path_valid": False,
                "failure_reason": None,
            }
            if not all((q_a_mid, q_a_ret, q_b_mid, q_b_ret)):
                attempt["failure_reason"] = "retract_or_approach_ik_failed"
                attempts.append(attempt)
                continue
            retract_states = [q_a, q_a_mid, q_a_ret]
            approach_states = [q_b_ret, q_b_mid, q_b]
            retract_check = validate_state_path(moveit, psm, group, retract_states, joint_bounds)
            approach_check = validate_state_path(moveit, psm, group, approach_states, joint_bounds)
            attempt["retract"] = retract_check
            attempt["approach"] = approach_check
            if not retract_check["valid"]:
                attempt["failure_reason"] = "retract_path_invalid"
                attempts.append(attempt)
                continue
            if not approach_check["valid"]:
                attempt["failure_reason"] = "approach_path_invalid"
                attempts.append(attempt)
                continue
            reposition_states, planning = plan_reposition(moveit, group, q_a_ret, q_b_ret)
            attempt["reposition_planning"] = planning
            if reposition_states is None:
                attempt["failure_reason"] = planning.get("failure_reason", "reposition_failed")
                attempts.append(attempt)
                continue
            reposition_check = validate_state_path(moveit, psm, group, reposition_states, joint_bounds)
            attempt["reposition"] = reposition_check
            if not reposition_check["valid"]:
                attempt["failure_reason"] = "reposition_path_invalid"
                attempts.append(attempt)
                continue
            full_states = retract_states[:-1] + reposition_states + approach_states[1:]
            full_check = validate_state_path(moveit, psm, group, full_states, joint_bounds)
            attempt["full_path"] = full_check
            if not full_check["valid"]:
                attempt["failure_reason"] = "full_path_invalid"
                attempts.append(attempt)
                continue
            attempt.update({
                "path_valid": True,
                "failure_reason": None,
                "achieved_distance_mm": distance_mm,
                "trajectory_states": full_states,
                "retract_states": retract_states,
                "reposition_states": reposition_states,
                "approach_states": approach_states,
                "start_state_valid": True,
                "goal_state_valid": True,
                "joint_limits_valid": True,
                "PlanningScene_path_valid": True,
                "self_collision_free": True,
                "robot_world_collision_free": True,
            })
            attempts.append(attempt)
            return {"transition_id": transition_id, "status": "passed_moveit", "selected_attempt": attempt, "attempts": attempts, "all_attempts_bounded": True}
    return {"transition_id": transition_id, "status": "failed_moveit", "selected_attempt": None, "attempts": attempts, "all_attempts_bounded": True}


def run_ros(request_path: Path) -> dict[str, Any]:  # pragma: no cover - ROS2 runtime
    import rclpy
    from moveit.planning import MoveItPy

    request = read_json(request_path)
    group = str(request.get("group_name", DEFAULT_GROUP))
    ee_link = str(request.get("ee_link", DEFAULT_EE))
    rclpy.init()
    try:
        moveit = MoveItPy(node_name=f"stage24s_off_planner_{request.get('rebuild_index', 1)}")
        scene_info = apply_exact_collision_environment(moveit, Path(request["parts_dir"]))
        psm = moveit.get_planning_scene_monitor()
        joint_bounds = [[float(pair[0]), float(pair[1])] for pair in request.get("joint_bounds", [])]
        results = []
        for transition in request.get("transitions", []):
            results.append(plan_transition(moveit, psm, group, ee_link, transition, joint_bounds))
        result = {
            "status": "passed_moveit" if all(row["status"] == "passed_moveit" for row in results) else "failed_moveit",
            "rebuild_index": request.get("rebuild_index"),
            "group_name": group,
            "ee_link": ee_link,
            "planning_pipeline": "ompl",
            "collision_method": COLLISION_METHOD,
            "CCD": CCD_STATUS,
            "clearance": CLEARANCE_STATUS,
            "scene": scene_info,
            "transitions": results,
        }
        write_json(Path(request["result_path"]), result)
        print(json.dumps(_json_safe(result), ensure_ascii=False))
        return result
    finally:
        try:
            rclpy.shutdown()
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ros-node", action="store_true")
    parser.add_argument("--request", type=Path, required=True)
    args, _ = parser.parse_known_args()
    if not args.ros_node:
        raise SystemExit("This runner must be launched through stage24s_off_planner_launch.py")
    run_ros(args.request.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
