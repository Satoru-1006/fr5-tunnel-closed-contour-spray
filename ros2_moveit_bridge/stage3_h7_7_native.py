#!/usr/bin/env python3
"""Offline MoveIt worker for Stage 3 H7.7 Tier-B remediation.

The H7.6 C2 timing remediation is reproduced first. Tier B then changes only
the waypoint velocity and acceleration boundary states using positive,
primitive-specific scale factors. Positions, times, primitive structure, IK
branches, process targets, semantic breaks, and SPRAY state remain frozen.
"""

from __future__ import annotations

import gc
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from moveit.core.robot_state import RobotState
from moveit.core.robot_trajectory import RobotTrajectory
from moveit.planning import MoveItPy

import stage3_h7_2_native as base
import stage3_h7_3_native as h73
import stage3_h7_6_native as h76


def load_plan(path: Path) -> dict[str, float]:
    raw = base.load_json(path)
    values = raw.get("alpha_plan", raw)
    plan = {str(key): float(value) for key, value in values.items()}
    missing = [str(value) for value in h73.PRIMITIVE_ORDER if str(value) not in plan]
    if missing:
        raise RuntimeError(f"Tier-B plan missing primitives: {missing}")
    if any(not math.isfinite(value) or not 0.0 < value <= 1.0 for value in plan.values()):
        raise RuntimeError("Tier-B alpha values must be finite and in (0, 1]")
    return plan


def apply_tier_b(
    moveit: MoveItPy,
    trajectory: RobotTrajectory,
    primitive_id: Any,
    plan: Mapping[str, float],
) -> tuple[RobotTrajectory, dict[str, Any]]:
    msg = trajectory.get_robot_trajectory_msg()
    points = msg.joint_trajectory.points
    before_t, before_q, before_v, before_a = base.message_arrays(msg)
    alpha = float(plan[str(primitive_id)])
    after_v = before_v * alpha
    after_a = before_a * alpha * alpha
    for index, point in enumerate(points):
        point.velocities = after_v[index].tolist()
        point.accelerations = after_a[index].tolist()
    start_state = RobotState(moveit.get_robot_model())
    candidate = RobotTrajectory(moveit.get_robot_model())
    candidate.joint_model_group_name = base.GROUP_NAME
    candidate.set_robot_trajectory_msg(start_state, msg)
    after_t, after_q, check_v, check_a = base.message_arrays(candidate.get_robot_trajectory_msg())
    position_delta = float(np.max(np.abs(after_q - before_q)))
    time_delta = float(np.max(np.abs(after_t - before_t)))
    if position_delta != 0.0 or time_delta > 2.0e-9:
        raise RuntimeError(f"H7.7 frozen state mutation: q={position_delta}, t={time_delta}")
    zero_before = np.all(before_v == 0.0, axis=1)
    zero_after = np.all(check_v == 0.0, axis=1)
    new_zero = int(np.count_nonzero(zero_after & ~zero_before))
    if new_zero:
        raise RuntimeError(f"H7.7 unauthorized zero-velocity breaks: {new_zero}")
    return candidate, {
        "schema_version": "stage3-h7-7-local-remediation-v1",
        "primitive_id": primitive_id,
        "tier": "B",
        "method": "positive_uniform_primitive_velocity_scaling_with_alpha_squared_acceleration",
        "velocity_scaling_alpha": alpha,
        "acceleration_scaling_alpha_squared": alpha * alpha,
        "affected_waypoint_count": int(len(points)) if alpha != 1.0 else 0,
        "affected_interval_count": int(max(0, len(points) - 1)) if alpha != 1.0 else 0,
        "maximum_velocity_change_rad_s": float(np.max(np.abs(check_v - before_v))),
        "maximum_acceleration_change_rad_s2": float(np.max(np.abs(check_a - before_a))),
        "position_modification_count": 0,
        "position_max_abs_delta_rad": position_delta,
        "timing_increase_s": 0.0,
        "time_max_abs_delta_s": time_delta,
        "new_zero_velocity_breaks": new_zero,
        "finite": bool(np.all(np.isfinite(check_v)) and np.all(np.isfinite(check_a))),
    }


def run_primitive(
    moveit: MoveItPy,
    primitive: Mapping[str, Any],
    limits: Mapping[str, Any],
    totg: Mapping[str, Any],
    contract: Mapping[str, Any],
    plan: Mapping[str, float],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    result, totg_rows, _unused, validation_rows = base.run_segment(moveit, primitive, limits, totg, False, contract)
    primitive_id = primitive["primitive_id"]
    result["primitive_id"] = primitive_id
    result["h6_4_segment_id"] = int(primitive["segment_id"])
    for row in totg_rows:
        row.update(primitive_id=primitive_id, schema_version="stage3-h7-7-baseline-totg-trajectory-v1")
    for row in validation_rows:
        row.update(primitive_id=primitive_id, schema_version="stage3-h7-7-validation-row-v1")
    if result.get("post_totg_status") != "PASSED":
        return result, totg_rows, [], validation_rows

    raw_q = np.asarray([row["joint_values"] for row in primitive["rows"]], dtype=float)
    start_state = RobotState(moveit.get_robot_model())
    trajectory = RobotTrajectory(moveit.get_robot_model())
    trajectory.joint_model_group_name = base.GROUP_NAME
    trajectory.set_robot_trajectory_msg(start_state, base.make_trajectory_message(primitive["rows"]))
    trajectory.unwind()
    if not trajectory.apply_totg_time_parameterization(
        float(totg["velocity_scaling_factor"]), float(totg["acceleration_scaling_factor"]),
        path_tolerance=float(totg["path_tolerance"]), resample_dt=float(totg["resample_dt"]),
        min_angle_change=float(totg["min_angle_change"]),
    ):
        result.update(status="BLOCKED", first_blocker="native_totg_nondeterministic_repeat_failure")
        return result, totg_rows, [], validation_rows

    trajectory, tier_a = h76.apply_smooth_local_time_warp(moveit, trajectory, primitive_id)
    tier_a_t, tier_a_q, _tier_a_v, _tier_a_a = base.message_arrays(trajectory.get_robot_trajectory_msg())
    trajectory, tier_b = apply_tier_b(moveit, trajectory, primitive_id, plan)
    input_t, input_q, input_v, input_a = base.message_arrays(trajectory.get_robot_trajectory_msg())
    if not np.array_equal(input_q, tier_a_q) or float(np.max(np.abs(input_t - tier_a_t))) > 2.0e-9:
        raise RuntimeError("H7.7 Tier-B changed H7.6 q/t")
    input_rows = base.trajectory_rows(primitive, "POST_TIER_B", input_t, input_q, input_v, input_a)
    for row in input_rows:
        row.update(primitive_id=primitive_id, schema_version="stage3-h7-7-remediated-input-trajectory-v1")

    wrapper_boolean = trajectory.apply_ruckig_smoothing(
        float(totg["velocity_scaling_factor"]), float(totg["acceleration_scaling_factor"]),
        mitigate_overshoot=True, overshoot_threshold=0.01,
    )
    wrapper = h76.latest_wrapper_summary()
    native_result = (wrapper or {}).get("final_ruckig_result") or {}
    strict_complete = bool(
        wrapper_boolean and wrapper and wrapper.get("smoothing_complete") is True
        and wrapper.get("duration_ceiling_hit") is False
        and native_result.get("numeric_result") in (0, 1)
    )
    result.update(
        tier_a_remediation=tier_a, local_remediation=tier_b, remediated_input_rows=input_rows,
        ruckig_run=True, ruckig_wrapper_returned_bool=bool(wrapper_boolean), ruckig_wrapper_summary=wrapper,
        ruckig_smoothing_complete=bool(wrapper and wrapper.get("smoothing_complete") is True),
        ruckig_duration_ceiling_reached=bool(wrapper and wrapper.get("duration_ceiling_hit") is True),
        native_ruckig_result=native_result, ruckig_status="PASSED" if strict_complete else "BLOCKED",
        strict_success=strict_complete, mitigate_overshoot=True,
    )

    final_t, final_q, final_v, final_a = base.message_arrays(trajectory.get_robot_trajectory_msg())
    dynamic = base.dynamics_validation(final_q, final_v, final_a, final_t, limits, "post_ruckig")
    process_rows, process = base.process_validation(moveit, primitive, final_q, final_t, contract)
    collision_rows, collision = base.collision_validate(moveit, final_t, final_q)
    collision["self_collision_failure_count"] = sum(bool(row.get("self_collision")) for row in collision_rows)
    collision["environment_collision_failure_count"] = sum(bool(row.get("environment_collision")) for row in collision_rows)
    endpoints = bool(np.allclose(final_q[0], raw_q[0], atol=1e-10) and np.allclose(final_q[-1], raw_q[-1], atol=1e-10))
    post_pass = bool(dynamic["status"] == process["status"] == collision["status"] == "PASSED" and endpoints)
    result.update(
        post_ruckig_duration_s=float(final_t[-1]), post_ruckig_output_waypoint_count=len(final_t),
        post_ruckig_dynamics=dynamic, post_ruckig_process=process, post_ruckig_collision=collision,
        start_end_configuration_preserved=endpoints,
        status="PASSED" if strict_complete and post_pass else "BLOCKED",
    )
    if not strict_complete:
        result["first_blocker"] = "native_ruckig_strict_completion_failed"
    elif not post_pass:
        result["first_blocker"] = "post_ruckig_validation_failed"
    final_rows = base.trajectory_rows(primitive, "POST_RUCKIG", final_t, final_q, final_v, final_a)
    for row in final_rows:
        row.update(primitive_id=primitive_id, schema_version="stage3-h7-7-ruckig-trajectory-v1")
    validation_rows.extend({"phase": "POST_RUCKIG", "kind": "process", "primitive_id": primitive_id, **row} for row in process_rows)
    validation_rows.extend({"phase": "POST_RUCKIG", "kind": "collision", "primitive_id": primitive_id, **row} for row in collision_rows)
    return result, totg_rows, final_rows, validation_rows


def formal_run(node: Any, moveit: MoveItPy) -> dict[str, Any]:
    output = Path(str(base.param(node, "output_dir"))).resolve()
    validation_path = Path(str(base.param(node, "h6_4_final_validation"))).resolve()
    segments_path = Path(str(base.param(node, "h6_4_segments"))).resolve()
    contract_path = Path(str(base.param(node, "process_contract"))).resolve()
    mesh_path = Path(str(base.param(node, "fixture_mesh"))).resolve()
    totg_path = Path(str(base.param(node, "totg_parameters"))).resolve()
    plan_path = Path(str(base.param(node, "tier_b_plan"))).resolve()
    plan = load_plan(plan_path)
    output.mkdir(parents=True, exist_ok=True)
    configured = Path(get_package_share_directory("fr5_tunnel_moveit_bridge")) / "config/joint_limits_with_jerk.yaml"
    limit_audit = base.runtime_limits(moveit, configured)
    base.dump_json(output / "runtime_limit_audit.json", limit_audit)
    if limit_audit["status"] != "PASSED":
        return {"status": "BLOCKED", "first_blocker": "runtime_joint_limit_audit_failed", "primitive_results": []}
    base.apply_fixture(moveit, mesh_path)
    primitives = h73.derive_primitives(base.load_jsonl(validation_path), base.load_json(segments_path))
    contract = base.load_json(contract_path)
    totg = base.load_json(totg_path)
    filtered = h73.pre_totg_audit(moveit, primitives, float(totg["min_angle_change"]))
    base.dump_json(output / "native_pre_totg_filtered_path_audit.json", filtered)
    pre_results = [base.run_segment(moveit, primitive, limit_audit["runtime_bounds"], totg, False, contract)[0] for primitive in primitives]
    precondition = bool(filtered["status"] == "PASSED" and len(pre_results) == 10 and all(row.get("post_totg_status") == "PASSED" for row in pre_results))
    results, all_totg, all_input, all_final, all_validation = [], [], [], [], []
    if precondition:
        for primitive in primitives:
            item, totg_rows, final_rows, validation_rows = run_primitive(moveit, primitive, limit_audit["runtime_bounds"], totg, contract, plan)
            results.append(item)
            all_totg.extend(totg_rows)
            all_input.extend(item.pop("remediated_input_rows", []))
            all_final.extend(final_rows)
            all_validation.extend(validation_rows)
    base.dump_jsonl(output / "totg_trajectories.jsonl", all_totg)
    base.dump_jsonl(output / "remediated_input_trajectories.jsonl", all_input)
    base.dump_jsonl(output / "ruckig_trajectories.jsonl", all_final)
    base.dump_jsonl(output / "validation_rows.jsonl", all_validation)
    passed = len(results) == 10 and all(row.get("status") == "PASSED" for row in results)
    summary = {
        "schema_version": "stage3-h7-7-native-result-v1",
        "status": "PASSED" if passed else "BLOCKED",
        "first_blocker": next((row.get("first_blocker") for row in results if row.get("status") != "PASSED"), None) or (None if precondition else "totg_global_precondition_failed"),
        "authoritative_primitive_order": h73.PRIMITIVE_ORDER, "primitive_results": results,
        "tier_b_alpha_plan": plan, "totg_global_precondition_passed_before_ruckig": precondition,
        "ruckig_wrapper_true_primitives": sum(row.get("ruckig_wrapper_returned_bool") is True for row in results),
        "ruckig_smoothing_complete_primitives": sum(row.get("ruckig_smoothing_complete") is True for row in results),
        "duration_ceiling_hit_primitives": sum(row.get("ruckig_duration_ceiling_reached") is True for row in results),
        "semantic_hash": base.semantic_hash({"primitives": results, "input": all_input, "ruckig": all_final}),
        "NEW_FJT_GOALS_SENT": 0, "ROBOT_MOTION_STARTED": "NO", "FORMAL_LEDGER_MUTATED": "NO", "STAGE_3_H8_STARTED": "NO",
    }
    base.dump_json(output / "native_result.json", summary)
    return summary


def main() -> int:
    rclpy.init()
    node = rclpy.create_node("stage3_h7_7_native")
    output = Path(str(base.param(node, "output_dir"))).resolve()
    moveit: MoveItPy | None = None
    return_code = 2
    h76.lifecycle(output, "rclpy_initialized")
    try:
        moveit = MoveItPy(node_name="stage3_h7_7_native")
        h76.lifecycle(output, "moveitpy_constructed")
        result = formal_run(node, moveit)
        return_code = 0 if result["status"] == "PASSED" else 2
        h76.lifecycle(output, "trajectory_work_complete", return_code=return_code)
        h76.lifecycle(output, "formal_artifacts_flushed")
    except Exception as exc:
        base.dump_json(output / "native_result.json", {
            "schema_version": "stage3-h7-7-native-result-v1", "status": "BLOCKED",
            "first_blocker": f"native_exception:{type(exc).__name__}:{exc}", "primitive_results": [],
            "NEW_FJT_GOALS_SENT": 0, "ROBOT_MOTION_STARTED": "NO", "FORMAL_LEDGER_MUTATED": "NO", "STAGE_3_H8_STARTED": "NO",
        })
        h76.lifecycle(output, "formal_run_exception", exception=repr(exc))
    finally:
        if moveit is not None:
            h76.lifecycle(output, "moveit.shutdown_called")
            moveit.shutdown()
            h76.lifecycle(output, "moveit.shutdown_returned")
            del moveit
            gc.collect()
            h76.lifecycle(output, "gc_collect_complete")
        node.destroy_node()
        h76.lifecycle(output, "node_destroyed", ros_ok=bool(rclpy.ok()))
        if rclpy.ok():
            rclpy.shutdown()
        h76.lifecycle(output, "ros_shutdown_complete", ros_ok=bool(rclpy.ok()))
    h76.lifecycle(output, "main_return", return_code=return_code)
    return return_code


if __name__ == "__main__":
    sys.exit(main())
