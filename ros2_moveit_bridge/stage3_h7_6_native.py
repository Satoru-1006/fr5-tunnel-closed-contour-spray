#!/usr/bin/env python3
"""Offline native worker for Stage 3 H7.6 timing-only remediation.

The frozen waypoint positions and semantics are reconstructed exactly as in
H7.4/H7.5. A C2 smooth local time reparameterization is applied only to two
bounded maximum-velocity plateau neighborhoods before native Ruckig. This
module has no action client or robot-execution surface.
"""

from __future__ import annotations

import gc
import json
import math
import os
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


REMEDIATION_WINDOWS: dict[str, dict[str, float | int]] = {
    "1001": {"start_state_index": 443, "end_state_index": 790, "duration_increase_ratio": 0.09},
    "1003": {"start_state_index": 450, "end_state_index": 661, "duration_increase_ratio": 0.03},
}


def lifecycle(output: Path, event: str, **details: Any) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "lifecycle_events.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps({"event": event, **details}, sort_keys=True) + "\n")


def latest_wrapper_summary() -> dict[str, Any] | None:
    raw = os.environ.get("STAGE25R_NATIVE_DIR")
    if not raw:
        return None
    path = Path(raw) / "stage25r_native_run_summaries.jsonl"
    if not path.is_file():
        return None
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[-1] if rows else None


def assign_duration(message: Any, seconds: float) -> None:
    sec = math.floor(seconds)
    nanosec = int(round((seconds - sec) * 1_000_000_000.0))
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000
    message.sec = int(sec)
    message.nanosec = nanosec


def apply_smooth_local_time_warp(
    moveit: MoveItPy,
    trajectory: RobotTrajectory,
    primitive_id: Any,
) -> tuple[RobotTrajectory, dict[str, Any]]:
    msg = trajectory.get_robot_trajectory_msg()
    points = msg.joint_trajectory.points
    q = np.asarray([list(point.positions) for point in points], dtype=float)
    v = np.asarray([list(point.velocities) for point in points], dtype=float)
    a = np.asarray([list(point.accelerations) for point in points], dtype=float)
    t = np.asarray([base.duration_seconds(point.time_from_start) for point in points], dtype=float)
    before_t, before_v, before_a = t.copy(), v.copy(), a.copy()
    config = REMEDIATION_WINDOWS.get(str(primitive_id))
    if config is None:
        return trajectory, {
            "primitive_id": primitive_id, "modified": False,
            "position_path_changed": False, "modified_interval_count": 0,
            "duration_increase_s": 0.0,
        }

    start = int(config["start_state_index"])
    end = int(config["end_state_index"])
    ratio = float(config["duration_increase_ratio"])
    if not (0 < start < end < len(t) - 1):
        raise RuntimeError(f"invalid H7.6 window for primitive {primitive_id}: {start}:{end}/{len(t)}")
    old_local_duration = float(t[end] - t[start])
    x = (t[start : end + 1] - t[start]) / old_local_duration
    smoothstep = 6.0 * x**5 - 15.0 * x**4 + 10.0 * x**3
    smoothstep_d1 = 30.0 * x**2 * (x - 1.0) ** 2
    smoothstep_d2 = 60.0 * x * (2.0 * x**2 - 3.0 * x + 1.0)
    duration_increase = ratio * old_local_duration
    time_scale_d1 = 1.0 + ratio * smoothstep_d1
    time_scale_d2 = ratio / old_local_duration * smoothstep_d2

    t[start : end + 1] = t[start : end + 1] + duration_increase * smoothstep
    t[end + 1 :] = t[end + 1 :] + duration_increase
    v[start : end + 1] = v[start : end + 1] / time_scale_d1[:, None]
    a[start : end + 1] = (
        a[start : end + 1] / time_scale_d1[:, None] ** 2
        - before_v[start : end + 1] * time_scale_d2[:, None] / time_scale_d1[:, None] ** 3
    )

    for index, point in enumerate(points):
        assign_duration(point.time_from_start, float(t[index]))
        point.velocities = v[index].tolist()
        point.accelerations = a[index].tolist()

    start_state = RobotState(moveit.get_robot_model())
    retimed = RobotTrajectory(moveit.get_robot_model())
    retimed.joint_model_group_name = base.GROUP_NAME
    retimed.set_robot_trajectory_msg(start_state, msg)
    check_t, check_q, check_v, check_a = base.message_arrays(retimed.get_robot_trajectory_msg())
    position_delta = float(np.max(np.abs(check_q - q)))
    if position_delta != 0.0:
        raise RuntimeError(f"H7.6 position mutation detected: {position_delta}")
    return retimed, {
        "primitive_id": primitive_id,
        "modified": True,
        "method": "C2_quintic_smoothstep_local_time_reparameterization",
        "tier": "A",
        "start_state_index": start,
        "end_state_index": end,
        "modified_interval_count": end - start,
        "baseline_local_duration_s": old_local_duration,
        "duration_increase_ratio": ratio,
        "duration_increase_s": duration_increase,
        "maximum_local_dt_change_s": float(np.max(np.abs(np.diff(check_t)[start:end] - np.diff(before_t)[start:end]))),
        "maximum_velocity_change_rad_s": float(np.max(np.abs(check_v - before_v))),
        "maximum_acceleration_change_rad_s2": float(np.max(np.abs(check_a - before_a))),
        "position_path_changed": False,
        "position_max_abs_delta_rad": position_delta,
        "left_boundary_velocity_delta_rad_s": float(np.max(np.abs(check_v[start] - before_v[start]))),
        "right_boundary_velocity_delta_rad_s": float(np.max(np.abs(check_v[end] - before_v[end]))),
        "left_boundary_acceleration_delta_rad_s2": float(np.max(np.abs(check_a[start] - before_a[start]))),
        "right_boundary_acceleration_delta_rad_s2": float(np.max(np.abs(check_a[end] - before_a[end]))),
        "time_monotonic": bool(np.all(np.diff(check_t) > 0.0)),
        "finite": bool(np.all(np.isfinite(check_t)) and np.all(np.isfinite(check_q)) and np.all(np.isfinite(check_v)) and np.all(np.isfinite(check_a))),
    }


def run_primitive(
    moveit: MoveItPy,
    primitive: Mapping[str, Any],
    limits: Mapping[str, Any],
    totg: Mapping[str, Any],
    contract: Mapping[str, Any],
    mitigate_overshoot: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    result, totg_rows, _unused, validation_rows = base.run_segment(moveit, primitive, limits, totg, False, contract)
    primitive_id = primitive["primitive_id"]
    result["primitive_id"] = primitive_id
    result["h6_4_segment_id"] = int(primitive["segment_id"])
    for row in totg_rows:
        row["primitive_id"] = primitive_id
        row["schema_version"] = "stage3-h7-6-baseline-totg-trajectory-v1"
    for row in validation_rows:
        row["primitive_id"] = primitive_id
        row["schema_version"] = "stage3-h7-6-validation-row-v1"
    if result.get("post_totg_status") != "PASSED":
        return result, totg_rows, [], validation_rows

    raw_q = np.asarray([row["joint_values"] for row in primitive["rows"]], dtype=float)
    start_state = RobotState(moveit.get_robot_model())
    trajectory = RobotTrajectory(moveit.get_robot_model())
    trajectory.joint_model_group_name = base.GROUP_NAME
    trajectory.set_robot_trajectory_msg(start_state, base.make_trajectory_message(primitive["rows"]))
    trajectory.unwind()
    totg_ok = trajectory.apply_totg_time_parameterization(
        float(totg["velocity_scaling_factor"]), float(totg["acceleration_scaling_factor"]),
        path_tolerance=float(totg["path_tolerance"]), resample_dt=float(totg["resample_dt"]),
        min_angle_change=float(totg["min_angle_change"]),
    )
    if not totg_ok:
        result.update(status="BLOCKED", first_blocker="native_totg_nondeterministic_repeat_failure")
        return result, totg_rows, [], validation_rows

    trajectory, remediation = apply_smooth_local_time_warp(moveit, trajectory, primitive_id)
    retimed_t, retimed_q, retimed_dq, retimed_ddq = base.message_arrays(trajectory.get_robot_trajectory_msg())
    retimed_rows = base.trajectory_rows(primitive, "POST_LOCAL_RETIMING", retimed_t, retimed_q, retimed_dq, retimed_ddq)
    for row in retimed_rows:
        row["primitive_id"] = primitive_id
        row["schema_version"] = "stage3-h7-6-retimed-input-trajectory-v1"

    wrapper_bool = trajectory.apply_ruckig_smoothing(
        float(totg["velocity_scaling_factor"]), float(totg["acceleration_scaling_factor"]),
        mitigate_overshoot=mitigate_overshoot, overshoot_threshold=0.01,
    )
    wrapper = latest_wrapper_summary()
    strict_complete = bool(
        wrapper_bool and wrapper and wrapper.get("smoothing_complete") is True
        and wrapper.get("duration_ceiling_hit") is False
        and (wrapper.get("final_ruckig_result") or {}).get("numeric_result") in (0, 1)
    )
    result.update(
        local_remediation=remediation, retimed_input_rows=retimed_rows, ruckig_run=True,
        ruckig_wrapper_returned_bool=bool(wrapper_bool), ruckig_wrapper_summary=wrapper,
        ruckig_smoothing_complete=bool(wrapper and wrapper.get("smoothing_complete") is True),
        ruckig_status="PASSED" if strict_complete else "BLOCKED", mitigate_overshoot=mitigate_overshoot,
    )

    final_t, final_q, final_dq, final_ddq = base.message_arrays(trajectory.get_robot_trajectory_msg())
    final_dynamic = base.dynamics_validation(final_q, final_dq, final_ddq, final_t, limits, "post_ruckig")
    final_process_rows, final_process = base.process_validation(moveit, primitive, final_q, final_t, contract)
    final_collision_rows, final_collision = base.collision_validate(moveit, final_t, final_q)
    final_collision["self_collision_failure_count"] = sum(bool(row.get("self_collision")) for row in final_collision_rows)
    final_collision["environment_collision_failure_count"] = sum(bool(row.get("environment_collision")) for row in final_collision_rows)
    endpoints_preserved = bool(np.allclose(final_q[0], raw_q[0], atol=1e-10) and np.allclose(final_q[-1], raw_q[-1], atol=1e-10))
    post_pass = bool(
        final_dynamic["status"] == final_process["status"] == final_collision["status"] == "PASSED"
        and final_dynamic["zero_velocity_start"] and final_dynamic["zero_velocity_end"] and endpoints_preserved
    )
    result.update(
        post_ruckig_duration_s=float(final_t[-1]), post_ruckig_output_waypoint_count=len(final_t),
        post_ruckig_dynamics=final_dynamic, post_ruckig_process=final_process,
        post_ruckig_collision=final_collision, start_end_configuration_preserved=endpoints_preserved,
        native_jerk_certification="PASSED" if strict_complete else "BLOCKED",
        status="PASSED" if strict_complete and post_pass else "BLOCKED",
    )
    if not strict_complete:
        result["first_blocker"] = "native_ruckig_strict_completion_failed"
    elif not post_pass:
        result["first_blocker"] = "post_ruckig_validation_failed"
    final_rows = base.trajectory_rows(primitive, "POST_RUCKIG", final_t, final_q, final_dq, final_ddq)
    for row in final_rows:
        row["primitive_id"] = primitive_id
        row["schema_version"] = "stage3-h7-6-ruckig-trajectory-v1"
    validation_rows.extend([{"phase": "POST_RUCKIG", "kind": "process", "primitive_id": primitive_id, **row} for row in final_process_rows])
    validation_rows.extend([{"phase": "POST_RUCKIG", "kind": "collision", "primitive_id": primitive_id, **row} for row in final_collision_rows])
    return result, totg_rows, final_rows, validation_rows


def formal_run(node: Any, moveit: MoveItPy) -> dict[str, Any]:
    output = Path(str(base.param(node, "output_dir"))).resolve()
    validation_path = Path(str(base.param(node, "h6_4_final_validation"))).resolve()
    segments_path = Path(str(base.param(node, "h6_4_segments"))).resolve()
    contract_path = Path(str(base.param(node, "process_contract"))).resolve()
    mesh_path = Path(str(base.param(node, "fixture_mesh"))).resolve()
    totg_path = Path(str(base.param(node, "totg_parameters"))).resolve()
    mitigate = bool(base.param(node, "mitigate_overshoot", True))
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
    pre_results = [base.run_segment(moveit, p, limit_audit["runtime_bounds"], totg, False, contract)[0] for p in primitives]
    precondition = bool(filtered["status"] == "PASSED" and len(pre_results) == 10 and all(item.get("post_totg_status") == "PASSED" for item in pre_results))
    results: list[dict[str, Any]] = []
    all_totg: list[dict[str, Any]] = []
    all_retimed: list[dict[str, Any]] = []
    all_final: list[dict[str, Any]] = []
    all_validation: list[dict[str, Any]] = []
    if precondition:
        for primitive in primitives:
            item, totg_rows, final_rows, validation_rows = run_primitive(moveit, primitive, limit_audit["runtime_bounds"], totg, contract, mitigate)
            results.append(item)
            all_totg.extend(totg_rows)
            all_retimed.extend(item.pop("retimed_input_rows", []))
            all_final.extend(final_rows)
            all_validation.extend(validation_rows)
    base.dump_jsonl(output / "totg_trajectories.jsonl", all_totg)
    base.dump_jsonl(output / "retimed_input_trajectories.jsonl", all_retimed)
    base.dump_jsonl(output / "ruckig_trajectories.jsonl", all_final)
    base.dump_jsonl(output / "validation_rows.jsonl", all_validation)
    passed = len(results) == 10 and all(item.get("status") == "PASSED" for item in results)
    summary = {
        "schema_version": "stage3-h7-6-native-result-v1",
        "status": "PASSED" if passed else "BLOCKED",
        "first_blocker": next((item.get("first_blocker") for item in results if item.get("status") != "PASSED"), None) or (None if precondition else "totg_global_precondition_failed"),
        "authoritative_primitive_order": h73.PRIMITIVE_ORDER, "primitive_results": results,
        "totg_global_precondition_passed_before_ruckig": precondition,
        "totg_primitives_passed": sum(item.get("post_totg_status") == "PASSED" for item in results),
        "ruckig_wrapper_true_primitives": sum(item.get("ruckig_wrapper_returned_bool") is True for item in results),
        "ruckig_smoothing_complete_primitives": sum(item.get("ruckig_smoothing_complete") is True for item in results),
        "mitigate_overshoot": mitigate, "remediation_windows": REMEDIATION_WINDOWS,
        "semantic_hash": base.semantic_hash({"primitives": results, "retimed": all_retimed, "ruckig": all_final}),
        "NEW_FJT_GOALS_SENT": 0, "ROBOT_MOTION_STARTED": "NO", "FORMAL_LEDGER_MUTATED": "NO",
    }
    base.dump_json(output / "native_result.json", summary)
    return summary


def main() -> int:
    rclpy.init()
    node = rclpy.create_node("stage3_h7_6_native")
    output = Path(str(base.param(node, "output_dir"))).resolve()
    moveit: MoveItPy | None = None
    return_code = 2
    lifecycle(output, "rclpy_initialized")
    try:
        moveit = MoveItPy(node_name="stage3_h7_6_native")
        lifecycle(output, "moveitpy_constructed")
        result = formal_run(node, moveit)
        return_code = 0 if result["status"] == "PASSED" else 2
        lifecycle(output, "trajectory_work_complete", return_code=return_code)
        lifecycle(output, "formal_artifacts_flushed")
    except Exception as exc:
        base.dump_json(output / "native_result.json", {
            "schema_version": "stage3-h7-6-native-result-v1", "status": "BLOCKED",
            "first_blocker": f"native_exception:{type(exc).__name__}:{exc}", "primitive_results": [],
            "NEW_FJT_GOALS_SENT": 0, "ROBOT_MOTION_STARTED": "NO", "FORMAL_LEDGER_MUTATED": "NO",
        })
        lifecycle(output, "formal_run_exception", exception=repr(exc))
    finally:
        if moveit is not None:
            lifecycle(output, "moveit.shutdown_called")
            moveit.shutdown()
            lifecycle(output, "moveit.shutdown_returned")
            del moveit
            lifecycle(output, "del_moveit_attempted")
            gc.collect()
            lifecycle(output, "gc_collect_complete")
        node.destroy_node()
        lifecycle(output, "node_destroyed", ros_ok=bool(rclpy.ok()))
        if rclpy.ok():
            rclpy.shutdown()
        lifecycle(output, "ros_shutdown_complete", ros_ok=bool(rclpy.ok()))
    lifecycle(output, "main_return", return_code=return_code)
    return return_code


if __name__ == "__main__":
    sys.exit(main())
