#!/usr/bin/env python3
"""Stage 3 H7.4 native MoveIt/Ruckig diagnostic worker.

This is an offline planning-scene certification worker.  It never creates an
action client, sends a trajectory, or starts robot motion.  The project native
interposer records every direct ``ruckig.calculate`` boundary and the complete
MoveIt retry loop; this worker independently rejects the historical wrapper
``true`` result unless the recorded loop reached ``smoothing_complete``.
"""

from __future__ import annotations

import gc
import json
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


def _lifecycle(output: Path, event: str, **details: Any) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "lifecycle_events.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps({"event": event, **details}, sort_keys=True) + "\n")


def _latest_wrapper_summary() -> dict[str, Any] | None:
    raw = os.environ.get("STAGE25R_NATIVE_DIR")
    if not raw:
        return None
    path = Path(raw) / "stage25r_native_run_summaries.jsonl"
    if not path.is_file():
        return None
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[-1] if rows else None


def run_primitive(
    moveit: MoveItPy,
    primitive: Mapping[str, Any],
    limits: Mapping[str, Any],
    totg: Mapping[str, Any],
    contract: Mapping[str, Any],
    mitigate_overshoot: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    # Reuse the exact H7.2 native TOTG/FK/PlanningScene gates, with Ruckig
    # explicitly disabled until every primitive passes the global precondition.
    result, totg_rows, _unused, validation_rows = base.run_segment(
        moveit, primitive, limits, totg, False, contract
    )
    primitive_id = primitive["primitive_id"]
    result["primitive_id"] = primitive_id
    result["h6_4_segment_id"] = int(primitive["segment_id"])
    for row in totg_rows:
        row["primitive_id"] = primitive_id
        row["schema_version"] = "stage3-h7-4-time-parameterized-trajectory-v1"
    for row in validation_rows:
        row["primitive_id"] = primitive_id
        row["schema_version"] = "stage3-h7-4-validation-row-v1"
    if result.get("post_totg_status") != "PASSED":
        return result, totg_rows, [], validation_rows

    # Reconstruct the same RobotTrajectory and repeat frozen TOTG so the exact
    # native post-TOTG states flow directly into the instrumented wrapper.
    raw_q = np.asarray([row["joint_values"] for row in primitive["rows"]], dtype=float)
    start_state = RobotState(moveit.get_robot_model())
    trajectory = RobotTrajectory(moveit.get_robot_model())
    trajectory.joint_model_group_name = base.GROUP_NAME
    trajectory.set_robot_trajectory_msg(start_state, base.make_trajectory_message(primitive["rows"]))
    trajectory.unwind()
    totg_ok = trajectory.apply_totg_time_parameterization(
        float(totg["velocity_scaling_factor"]),
        float(totg["acceleration_scaling_factor"]),
        path_tolerance=float(totg["path_tolerance"]),
        resample_dt=float(totg["resample_dt"]),
        min_angle_change=float(totg["min_angle_change"]),
    )
    if not totg_ok:
        result.update(status="BLOCKED", first_blocker="native_totg_nondeterministic_repeat_failure")
        return result, totg_rows, [], validation_rows

    wrapper_bool = trajectory.apply_ruckig_smoothing(
        float(totg["velocity_scaling_factor"]),
        float(totg["acceleration_scaling_factor"]),
        mitigate_overshoot=mitigate_overshoot,
        overshoot_threshold=0.01,
    )
    wrapper = _latest_wrapper_summary()
    strict_complete = bool(
        wrapper_bool
        and wrapper
        and wrapper.get("smoothing_complete") is True
        and wrapper.get("duration_ceiling_hit") is False
        and (wrapper.get("final_ruckig_result") or {}).get("numeric_result") in (0, 1)
    )
    result.update(
        ruckig_run=True,
        ruckig_wrapper_returned_bool=bool(wrapper_bool),
        ruckig_wrapper_summary=wrapper,
        ruckig_smoothing_complete=bool(wrapper and wrapper.get("smoothing_complete") is True),
        ruckig_status="PASSED" if strict_complete else "BLOCKED",
        mitigate_overshoot=mitigate_overshoot,
    )

    final_t, final_q, final_dq, final_ddq = base.message_arrays(trajectory.get_robot_trajectory_msg())
    final_dynamic = base.dynamics_validation(final_q, final_dq, final_ddq, final_t, limits, "post_ruckig")
    final_process_rows, final_process = base.process_validation(moveit, primitive, final_q, final_t, contract)
    final_collision_rows, final_collision = base.collision_validate(moveit, final_t, final_q)
    final_collision["self_collision_failure_count"] = sum(bool(row.get("self_collision")) for row in final_collision_rows)
    final_collision["environment_collision_failure_count"] = sum(
        bool(row.get("environment_collision")) for row in final_collision_rows
    )
    endpoints_preserved = bool(
        np.allclose(final_q[0], raw_q[0], atol=1e-10) and np.allclose(final_q[-1], raw_q[-1], atol=1e-10)
    )
    post_pass = bool(
        final_dynamic["status"] == final_process["status"] == final_collision["status"] == "PASSED"
        and final_dynamic["zero_velocity_start"]
        and final_dynamic["zero_velocity_end"]
        and endpoints_preserved
    )
    result.update(
        post_ruckig_duration_s=float(final_t[-1]),
        post_ruckig_output_waypoint_count=len(final_t),
        post_ruckig_dynamics=final_dynamic,
        post_ruckig_process=final_process,
        post_ruckig_collision=final_collision,
        start_end_configuration_preserved=endpoints_preserved,
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
        row["schema_version"] = "stage3-h7-4-ruckig-trajectory-v1"
    validation_rows.extend(
        [{"phase": "POST_RUCKIG", "kind": "process", "primitive_id": primitive_id, **row} for row in final_process_rows]
    )
    validation_rows.extend(
        [{"phase": "POST_RUCKIG", "kind": "collision", "primitive_id": primitive_id, **row} for row in final_collision_rows]
    )
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

    # Mandatory all-primitive TOTG gate before the first Ruckig call.
    pre_results = [base.run_segment(moveit, p, limit_audit["runtime_bounds"], totg, False, contract)[0] for p in primitives]
    totg_precondition = bool(
        filtered["status"] == "PASSED"
        and len(pre_results) == 10
        and all(item.get("post_totg_status") == "PASSED" for item in pre_results)
    )
    results: list[dict[str, Any]] = []
    all_totg: list[dict[str, Any]] = []
    all_final: list[dict[str, Any]] = []
    all_validation: list[dict[str, Any]] = []
    if totg_precondition:
        for primitive in primitives:
            item, totg_rows, final_rows, validation_rows = run_primitive(
                moveit, primitive, limit_audit["runtime_bounds"], totg, contract, mitigate
            )
            results.append(item)
            all_totg.extend(totg_rows)
            all_final.extend(final_rows)
            all_validation.extend(validation_rows)

    base.dump_jsonl(output / "totg_trajectories.jsonl", all_totg)
    base.dump_jsonl(output / "ruckig_trajectories.jsonl", all_final)
    base.dump_jsonl(output / "validation_rows.jsonl", all_validation)
    passed = len(results) == 10 and all(item.get("status") == "PASSED" for item in results)
    summary = {
        "schema_version": "stage3-h7-4-native-result-v1",
        "status": "PASSED" if passed else "BLOCKED",
        "first_blocker": next((item.get("first_blocker") for item in results if item.get("status") != "PASSED"), None)
        or (None if totg_precondition else "totg_global_precondition_failed"),
        "authoritative_primitive_order": h73.PRIMITIVE_ORDER,
        "primitive_results": results,
        "totg_global_precondition_passed_before_ruckig": totg_precondition,
        "totg_primitives_passed": sum(item.get("post_totg_status") == "PASSED" for item in results),
        "ruckig_wrapper_true_primitives": sum(item.get("ruckig_wrapper_returned_bool") is True for item in results),
        "ruckig_smoothing_complete_primitives": sum(item.get("ruckig_smoothing_complete") is True for item in results),
        "mitigate_overshoot": mitigate,
        "semantic_hash": base.semantic_hash({"primitives": results, "totg": all_totg, "ruckig": all_final}),
        "NEW_FJT_GOALS_SENT": 0,
        "ROBOT_MOTION_STARTED": "NO",
        "FORMAL_LEDGER_MUTATED": "NO",
    }
    base.dump_json(output / "native_result.json", summary)
    return summary


def main() -> int:
    rclpy.init()
    node = rclpy.create_node("stage3_h7_4_native")
    output = Path(str(base.param(node, "output_dir"))).resolve()
    moveit: MoveItPy | None = None
    return_code = 2
    _lifecycle(output, "rclpy_initialized")
    try:
        moveit = MoveItPy(node_name="stage3_h7_4_native")
        _lifecycle(output, "moveitpy_constructed")
        result = formal_run(node, moveit)
        return_code = 0 if result["status"] == "PASSED" else 2
        _lifecycle(output, "formal_run_complete", return_code=return_code)
    except Exception as exc:
        result = {
            "schema_version": "stage3-h7-4-native-result-v1",
            "status": "BLOCKED",
            "first_blocker": f"native_exception:{type(exc).__name__}:{exc}",
            "primitive_results": [],
            "NEW_FJT_GOALS_SENT": 0,
            "ROBOT_MOTION_STARTED": "NO",
            "FORMAL_LEDGER_MUTATED": "NO",
        }
        base.dump_json(output / "native_result.json", result)
        _lifecycle(output, "formal_run_exception", exception=repr(exc))
    finally:
        if moveit is not None:
            _lifecycle(output, "before_moveitpy_shutdown")
            moveit.shutdown()
            _lifecycle(output, "after_moveitpy_shutdown")
            del moveit
            _lifecycle(output, "after_moveitpy_delete")
            gc.collect()
            _lifecycle(output, "after_moveitpy_delete_gc")
        node.destroy_node()
        _lifecycle(output, "after_node_destroy")
        if rclpy.ok():
            rclpy.shutdown()
        _lifecycle(output, "after_rclpy_shutdown")
    _lifecycle(output, "main_return", return_code=return_code)
    return return_code


if __name__ == "__main__":
    sys.exit(main())
