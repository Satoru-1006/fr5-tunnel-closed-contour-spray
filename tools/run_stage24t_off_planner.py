#!/usr/bin/env python3
"""Stage 2.4T OFF planner with bounded multi-IK and staging recovery.

This module deliberately reuses the Stage 2.4S native scene construction and
collision checks, but changes the search contract: every request may contain
multiple formal endpoint pairs, deterministic IK seed pools, a bounded
distance/direction ladder, and validated staging states.  It is OFF-only; it
does not generate or mutate Spray-ON graph nodes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tools.run_stage24s_off_planner as base


COLLISION_METHOD = "adaptive_discrete_interpolation"
CCD_STATUS = "not_available"
CLEARANCE_STATUS = "not_available"
INTERPOLATION_STEP_DEG = 1.0
IK_TIMEOUT_S = 1.0
MAX_IK_SOLUTIONS_PER_POSE = 8
MAX_IK_COMBINATIONS = 24
MAX_ENDPOINT_PAIRS = 100


def _safe(value: Any) -> Any:
    return base._json_safe(value)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    base.write_json(path, value)


def set_state(moveit: Any, group: str, q: list[float]):
    return base.set_state(moveit, group, q)


def state_record(moveit: Any, psm: Any, group: str, q: list[float], bounds: list[list[float]]) -> dict[str, Any]:
    return base.state_record(moveit, psm, group, q, bounds)


def interpolate_states(q0: list[float], q1: list[float], step_deg: float = INTERPOLATION_STEP_DEG) -> list[list[float]]:
    return base.interpolate_states(q0, q1, step_deg)


def validate_state_path(moveit: Any, psm: Any, group: str, states: list[list[float]], bounds: list[list[float]]) -> dict[str, Any]:
    return base.validate_state_path(moveit, psm, group, states, bounds)


def _unique_seed_rows(seeds: list[list[float]], width: int) -> list[list[float]]:
    rows: list[list[float]] = []
    seen: set[tuple[float, ...]] = set()
    for seed in seeds:
        values = tuple(round(float(x), 12) for x in seed)
        if len(values) != width or values in seen:
            continue
        seen.add(values)
        rows.append([float(x) for x in values])
    return rows


def ik_shift_with_seed(moveit: Any, group: str, ee_link: str, q_seed: list[float], transform: np.ndarray,
                       direction: list[float], distance_m: float) -> tuple[list[float] | None, dict[str, Any]]:
    state = set_state(moveit, group, q_seed)
    pose = base.pose_from_transform(transform)
    vector = np.asarray(direction, dtype=float)
    vector /= max(float(np.linalg.norm(vector)), 1.0e-12)
    pose.position.x += float(distance_m * vector[0])
    pose.position.y += float(distance_m * vector[1])
    pose.position.z += float(distance_m * vector[2])
    started = time.monotonic()
    try:
        solved = bool(state.set_from_ik(group, pose, ee_link, IK_TIMEOUT_S))
    except Exception as exc:  # pragma: no cover - native runtime
        return None, {"ik_solved": False, "failure_reason": "ik_exception", "detail": repr(exc), "elapsed_s": time.monotonic() - started}
    elapsed = time.monotonic() - started
    if not solved:
        return None, {"ik_solved": False, "failure_reason": "ik_no_solution", "elapsed_s": elapsed}
    state.update()
    q = np.asarray(state.get_joint_group_positions(group), dtype=float).tolist()
    return [float(x) for x in q], {"ik_solved": True, "elapsed_s": elapsed, "target_distance_m": distance_m}


def solve_pose_candidates(moveit: Any, psm: Any, group: str, ee_link: str, endpoint_q: list[float],
                          transform: np.ndarray, direction: list[float], distance_mm: float,
                          seeds: list[list[float]], bounds: list[list[float]], phase: str) -> dict[str, Any]:
    width = len(endpoint_q)
    seed_rows = _unique_seed_rows([endpoint_q, *seeds], width)[:MAX_IK_SOLUTIONS_PER_POSE]
    solutions: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    for seed_index, seed in enumerate(seed_rows):
        q, ik = ik_shift_with_seed(moveit, group, ee_link, seed, transform, direction, distance_mm / 1000.0)
        row: dict[str, Any] = {"phase": phase, "seed_index": seed_index, "seed_q": seed, "direction": direction,
                               "distance_mm": distance_mm, "ik": ik}
        if q is None:
            row["solution_status"] = "ik_pose_has_no_solution"
            attempts.append(row)
            continue
        record = state_record(moveit, psm, group, q, bounds)
        row["q"] = q
        row["state_validation"] = {
            "joint_limits": "pass" if record["joint_limits_valid"] else "fail",
            "PlanningScene_collision_free": "pass" if record["collision_free"] else "fail",
            "self_collision_free": "pass" if record["collision_free"] else "fail",
            "robot_world_collision_free": "pass" if record["collision_free"] else "fail",
            "collision_summary": record.get("collision_summary"),
        }
        if record["valid"]:
            row["solution_status"] = "valid"
            solutions.append(row)
        else:
            row["solution_status"] = "all_tested_IK_solutions_collision" if record["joint_limits_valid"] else "joint_limit_invalid"
        attempts.append(row)
    solutions.sort(key=lambda row: (float(np.linalg.norm(np.asarray(row["q"]) - np.asarray(endpoint_q))), int(row["seed_index"]), tuple(row["q"])))
    return {"phase": phase, "attempts": attempts, "valid_solutions": solutions, "valid_solution_count": len(solutions),
            "seed_count": len(seed_rows), "single_seed_solution_collision": bool(attempts and len(seed_rows) == 1 and not solutions)}


def _plan_once(moveit: Any, group: str, q_start: list[float], q_goal: list[float], planner: dict[str, Any]) -> tuple[list[list[float]] | None, dict[str, Any]]:
    component = moveit.get_planning_component(group)
    component.set_start_state(robot_state=set_state(moveit, group, q_start))
    component.set_goal_state(robot_state=set_state(moveit, group, q_goal))
    started = time.monotonic()
    try:
        # The explicit values are supplied to MoveItCpp as plan_request_params
        # by the Stage 2.4T launch wrapper.  Jazzy's Python MoveItPy binding
        # does not expose the MoveItCpp pointer required to construct the
        # PlanRequestParameters wrapper directly, so plan() uses the formally
        # loaded request parameters rather than silently relying on defaults.
        result = component.plan()
    except Exception as exc:  # pragma: no cover - native runtime
        return None, {"planning_success": False, "failure_reason": "planning_exception", "detail": repr(exc),
                      "elapsed_s": time.monotonic() - started, "requested": planner, "effective": planner}
    elapsed = time.monotonic() - started
    if not result or not getattr(result, "trajectory", None):
        return None, {"planning_success": False, "failure_reason": "planning_no_solution", "elapsed_s": elapsed,
                      "requested": planner, "effective": planner}
    import plan_closed_contour_moveit as bridge
    trajectory = bridge.as_robot_trajectory_message(result.trajectory)
    points = list(trajectory.joint_trajectory.points)
    if not points:
        return None, {"planning_success": False, "failure_reason": "planning_empty_trajectory", "elapsed_s": elapsed,
                      "requested": planner, "effective": planner}
    return [[float(x) for x in point.positions] for point in points], {
        "planning_success": True, "elapsed_s": elapsed,
        "planner_api": "PlanningComponent.plan() with explicit launch plan_request_params",
        "requested": planner, "effective": planner,
    }


def plan_reposition(moveit: Any, psm: Any, group: str, q_start: list[float], q_goal: list[float],
                    bounds: list[list[float]], staging_states: list[list[float]], planner: dict[str, Any]) -> tuple[list[list[float]] | None, dict[str, Any]]:
    direct, direct_meta = _plan_once(moveit, group, q_start, q_goal, planner)
    if direct is not None:
        check = validate_state_path(moveit, psm, group, direct, bounds)
        direct_meta["path_validation"] = check
        if check["valid"]:
            return direct, {"strategy": "direct", "planning": direct_meta}
    staging_attempts: list[dict[str, Any]] = []
    for index, staging in enumerate(staging_states):
        staging_record = state_record(moveit, psm, group, staging, bounds)
        row: dict[str, Any] = {"staging_index": index, "staging_q": staging, "staging_state": staging_record}
        if not staging_record["valid"]:
            row["status"] = "staging_state_invalid"
            staging_attempts.append(row)
            continue
        first, first_meta = _plan_once(moveit, group, q_start, staging, planner)
        second, second_meta = _plan_once(moveit, group, staging, q_goal, planner)
        row["first_planning"] = first_meta
        row["second_planning"] = second_meta
        if first is None or second is None:
            row["status"] = "staging_planning_failed"
            staging_attempts.append(row)
            continue
        first_check = validate_state_path(moveit, psm, group, first, bounds)
        second_check = validate_state_path(moveit, psm, group, second, bounds)
        row["first_path_validation"] = first_check
        row["second_path_validation"] = second_check
        if not first_check["valid"] or not second_check["valid"]:
            row["status"] = "staging_path_invalid"
            staging_attempts.append(row)
            continue
        return first[:-1] + second, {"strategy": "staging", "staging_attempts": staging_attempts + [row]}
    return None, {"strategy": "failed", "direct_planning": direct_meta, "staging_attempts": staging_attempts}


def direction_rows(request: dict[str, Any]) -> list[dict[str, Any]]:
    values = request.get("directions")
    if values:
        return [{"direction": [float(x) for x in row["direction"]], "geometry_reason": row.get("geometry_reason", "surface_normal")} for row in values]
    return [{"direction": [float(x) for x in request["direction"]], "geometry_reason": "+surface_normal"}]


def plan_transition(moveit: Any, psm: Any, group: str, ee_link: str, request: dict[str, Any], bounds: list[list[float]]) -> dict[str, Any]:
    transition_id = int(request["transition_id"])
    pairs = list(request.get("endpoint_pairs", []))[:int(request.get("max_endpoint_pairs", MAX_ENDPOINT_PAIRS))]
    distances = [float(x) for x in request.get("retract_distances_mm", [10, 20, 30, 40, 60, 80])]
    directions = direction_rows(request)
    planner = dict(request.get("planner_parameters", {}))
    staging_states = [[float(x) for x in q] for q in request.get("staging_states", [])]
    all_attempts: list[dict[str, Any]] = []
    for pair in pairs:
        q_a = [float(x) for x in pair["q_a"]]
        q_b = [float(x) for x in pair["q_b"]]
        start_record = state_record(moveit, psm, group, q_a, bounds)
        goal_record = state_record(moveit, psm, group, q_b, bounds)
        if not start_record["valid"] or not goal_record["valid"]:
            all_attempts.append({"transition_id": transition_id, "endpoint_pair_rank": pair.get("rank"), "failure_reason": "endpoint_state_invalid", "start_state": start_record, "goal_state": goal_record})
            continue
        q_a_tf = base.fk_transform(moveit, group, ee_link, q_a)
        q_b_tf = base.fk_transform(moveit, group, ee_link, q_b)
        seeds_a = [[float(x) for x in q] for q in pair.get("seed_pool_a", [])]
        seeds_b = [[float(x) for x in q] for q in pair.get("seed_pool_b", [])]
        for direction_row in directions:
            direction = direction_row["direction"]
            for distance_mm in distances:
                a_mid = solve_pose_candidates(moveit, psm, group, ee_link, q_a, q_a_tf, direction, distance_mm * 0.5, seeds_a, bounds, "a_mid")
                a_ret = solve_pose_candidates(moveit, psm, group, ee_link, q_a, q_a_tf, direction, distance_mm, seeds_a, bounds, "a_retracted")
                b_mid = solve_pose_candidates(moveit, psm, group, ee_link, q_b, q_b_tf, direction, distance_mm * 0.5, seeds_b, bounds, "b_mid")
                b_ret = solve_pose_candidates(moveit, psm, group, ee_link, q_b, q_b_tf, direction, distance_mm, seeds_b, bounds, "b_retracted")
                attempt: dict[str, Any] = {
                    "transition_id": transition_id, "endpoint_pair_rank": pair.get("rank"),
                    "endpoint_candidate_a": pair.get("candidate_a"), "endpoint_candidate_b": pair.get("candidate_b"),
                    "direction": direction, "geometry_reason": direction_row["geometry_reason"], "requested_distance_mm": distance_mm,
                    "ik": {"a_mid": a_mid, "a_retracted": a_ret, "b_mid": b_mid, "b_retracted": b_ret},
                    "path_valid": False,
                }
                if not all((a_mid["valid_solutions"], a_ret["valid_solutions"], b_mid["valid_solutions"], b_ret["valid_solutions"])):
                    attempt["failure_reason"] = "retract_or_approach_ik_failed"
                    all_attempts.append(attempt)
                    continue
                combinations = []
                for a_mid_row in a_mid["valid_solutions"][:4]:
                    for a_ret_row in a_ret["valid_solutions"][:4]:
                        for b_mid_row in b_mid["valid_solutions"][:4]:
                            for b_ret_row in b_ret["valid_solutions"][:4]:
                                score = sum(float(np.linalg.norm(np.asarray(row["q"]) - np.asarray(q))) for row, q in ((a_mid_row, q_a), (a_ret_row, q_a), (b_mid_row, q_b), (b_ret_row, q_b)))
                                combinations.append((score, a_mid_row, a_ret_row, b_mid_row, b_ret_row))
                combinations.sort(key=lambda row: (row[0], row[1]["seed_index"], row[2]["seed_index"], row[3]["seed_index"], row[4]["seed_index"]))
                for _, a_mid_row, a_ret_row, b_mid_row, b_ret_row in combinations[:MAX_IK_COMBINATIONS]:
                    retract_states = [q_a, a_mid_row["q"], a_ret_row["q"]]
                    approach_states = [b_ret_row["q"], b_mid_row["q"], q_b]
                    retract_check = validate_state_path(moveit, psm, group, retract_states, bounds)
                    approach_check = validate_state_path(moveit, psm, group, approach_states, bounds)
                    attempt["retract"] = retract_check
                    attempt["approach"] = approach_check
                    attempt["ik_selected"] = {"a_mid": a_mid_row, "a_retracted": a_ret_row, "b_mid": b_mid_row, "b_retracted": b_ret_row}
                    if not retract_check["valid"]:
                        attempt["failure_reason"] = "retract_path_invalid"
                        continue
                    if not approach_check["valid"]:
                        attempt["failure_reason"] = "approach_path_invalid"
                        continue
                    reposition_states, reposition = plan_reposition(moveit, psm, group, a_ret_row["q"], b_ret_row["q"], bounds, staging_states, planner)
                    attempt["reposition"] = reposition
                    if reposition_states is None:
                        attempt["failure_reason"] = "reposition_failed"
                        continue
                    full_states = retract_states[:-1] + reposition_states + approach_states[1:]
                    full_check = validate_state_path(moveit, psm, group, full_states, bounds)
                    attempt["full_path"] = full_check
                    if not full_check["valid"]:
                        attempt["failure_reason"] = "full_path_invalid"
                        continue
                    attempt.update({"path_valid": True, "failure_reason": None, "achieved_distance_mm": distance_mm,
                                    "trajectory_states": full_states, "retract_states": retract_states,
                                    "reposition_states": reposition_states, "approach_states": approach_states,
                                    "start_state_valid": True, "goal_state_valid": True, "joint_limits_valid": True,
                                    "PlanningScene_path_valid": True, "self_collision_free": True, "robot_world_collision_free": True})
                    all_attempts.append(attempt)
                    return {"transition_id": transition_id, "status": "passed_moveit", "selected_attempt": attempt,
                            "attempts": all_attempts, "all_attempts_bounded": True, "endpoint_pair_count": len(pairs)}
                all_attempts.append(attempt)
    return {"transition_id": transition_id, "status": "failed_moveit", "selected_attempt": None,
            "attempts": all_attempts, "all_attempts_bounded": True, "endpoint_pair_count": len(pairs)}


def run_ros(request_path: Path) -> dict[str, Any]:  # pragma: no cover - native runtime
    import rclpy
    from moveit.planning import MoveItPy

    request = read_json(request_path)
    group = str(request.get("group_name", base.DEFAULT_GROUP))
    ee_link = str(request.get("ee_link", base.DEFAULT_EE))
    rclpy.init()
    try:
        moveit = MoveItPy(node_name=f"stage24t_off_planner_{request.get('rebuild_index', 1)}")
        scene_info = base.apply_exact_collision_environment(moveit, Path(request["parts_dir"]))
        psm = moveit.get_planning_scene_monitor()
        bounds = [[float(pair[0]), float(pair[1])] for pair in request.get("joint_bounds", [])]
        results = [plan_transition(moveit, psm, group, ee_link, row, bounds) for row in request.get("transitions", [])]
        planner = request.get("planner_parameters", {})
        result = {"status": "passed_moveit" if all(row["status"] == "passed_moveit" for row in results) else "failed_moveit",
                  "rebuild_index": request.get("rebuild_index"), "group_name": group, "ee_link": ee_link,
                  "planning_pipeline": planner.get("planning_pipeline", "ompl"),
                  "planner_parameters": {"requested": planner, "effective": planner, "api": "MoveItCpp plan_request_params loaded by launch; PlanningComponent.plan()"},
                  "collision_method": COLLISION_METHOD, "CCD": CCD_STATUS, "clearance": CLEARANCE_STATUS,
                  "scene": scene_info, "transitions": results}
        write_json(Path(request["result_path"]), result)
        print(json.dumps(_safe(result), ensure_ascii=False))
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
        raise SystemExit("This runner must be launched through stage24t_off_planner_launch.py")
    run_ros(args.request.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
