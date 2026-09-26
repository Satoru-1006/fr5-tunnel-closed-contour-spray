#!/usr/bin/env python3
"""MoveIt FK/FCL audit for Stage 3 H7.5 overshoot events.

This worker is read-only and offline.  It creates no action client and never
dispatches a trajectory.  Every sampled overshoot state is checked with the
same RobotModel, FK, PlanningScene and FCL geometry used by Stage 3 H7.4.
"""

from __future__ import annotations

import gc
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from moveit.core.robot_state import RobotState
from moveit.planning import MoveItPy

import stage3_h7_2_native as base
import stage3_h7_3_native as h73
from plan_closed_contour_moveit import _rotation_matrix_to_quaternion, _transform_matrix
from stage3_h7_native import collision_state


def lifecycle(output: Path, event: str, **details: Any) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "lifecycle_events.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps({"event": event, **details}, sort_keys=True) + "\n")


def pose(state: RobotState, q: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state.set_joint_group_positions(base.GROUP_NAME, list(q))
    state.update()
    transform = _transform_matrix(state.get_global_link_transform(base.EE_LINK))
    position = transform[:3, 3].astype(float)
    rotation = transform[:3, :3].astype(float)
    quaternion = np.asarray(_rotation_matrix_to_quaternion(rotation), dtype=float)
    return position, rotation, quaternion


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    left = np.asarray(a, dtype=float)
    right = np.asarray(b, dtype=float)
    left /= np.linalg.norm(left)
    right /= np.linalg.norm(right)
    return math.degrees(math.acos(float(np.clip(np.dot(left, right), -1.0, 1.0))))


def process_metrics(position: np.ndarray, rotation: np.ndarray, quaternion: np.ndarray,
                    reference: Mapping[str, np.ndarray], contract: Mapping[str, Any]) -> dict[str, Any]:
    standoff = float(np.linalg.norm(position - reference["surface_point"]))
    direction = rotation[:, 2].astype(float)
    direction /= np.linalg.norm(direction)
    result = {
        "standoff_m": standoff,
        "standoff_error_m": abs(standoff - float(contract["geometry"]["nominal_standoff_m"])),
        "surface_normal_deviation_deg": angle_deg(direction, reference["spray_direction"]),
        "tcp_position_error_m": float(np.linalg.norm(position - reference["position"])),
        "tcp_orientation_error_deg": base.quaternion_error_deg(quaternion, reference["orientation"]),
    }
    result["process_tolerance_pass"] = bool(
        result["standoff_error_m"] <= float(contract["geometry"]["standoff_abs_tolerance_m"])
        and result["surface_normal_deviation_deg"] <= float(contract["geometry"]["normal_angle_tolerance_deg"])
        and result["tcp_position_error_m"] <= float(contract["tcp_reproduction"]["tcp_position_tolerance_m"])
        and result["tcp_orientation_error_deg"] <= float(contract["tcp_reproduction"]["tcp_orientation_tolerance_deg"])
    )
    return result


def formal_run(node: Any, moveit: MoveItPy) -> dict[str, Any]:
    output = Path(str(base.param(node, "output_dir"))).resolve()
    request_path = Path(str(base.param(node, "requests"))).resolve()
    validation_path = Path(str(base.param(node, "h6_4_final_validation"))).resolve()
    segments_path = Path(str(base.param(node, "h6_4_segments"))).resolve()
    contract_path = Path(str(base.param(node, "process_contract"))).resolve()
    mesh_path = Path(str(base.param(node, "fixture_mesh"))).resolve()

    requests = base.load_json(request_path)
    contract = base.load_json(contract_path)
    primitives = h73.derive_primitives(base.load_jsonl(validation_path), base.load_json(segments_path))
    by_id = {str(item["primitive_id"]): item for item in primitives}
    configured = Path(get_package_share_directory("fr5_tunnel_moveit_bridge")) / "config/joint_limits_with_jerk.yaml"
    limit_audit = base.runtime_limits(moveit, configured)
    base.apply_fixture(moveit, mesh_path)
    bounds = limit_audit.get("runtime_bounds", {})
    lower = np.asarray([bounds[name]["position_lower_rad"] for name in base.JOINT_NAMES], dtype=float)
    upper = np.asarray([bounds[name]["position_upper_rad"] for name in base.JOINT_NAMES], dtype=float)

    records: list[dict[str, Any]] = []
    state = RobotState(moveit.get_robot_model())
    monitor = moveit.get_planning_scene_monitor()
    with monitor.read_only() as scene:
        for event in requests["events"]:
            primitive = by_id[str(event["primitive_id"])]
            raw_rows = primitive["rows"]
            raw_q = np.asarray([row["joint_values"] for row in raw_rows], dtype=float)
            for point_name in ("first_overshoot", "worst_overshoot_in_call"):
                point = event.get(point_name)
                if not point:
                    continue
                sampled_q = np.asarray(point["q"], dtype=float)
                target_q = np.asarray(point["target_position_vector"], dtype=float)
                edge, alpha, target_path_error = base.nearest_polyline_projection(target_q, raw_q)
                target_p, target_r, target_quat = pose(state, target_q)
                sampled_p, sampled_r, sampled_quat = pose(state, sampled_q)
                collision = collision_state(scene, state, sampled_q)
                target_process = None
                sampled_process = None
                if primitive["spray_state"] == "SPRAY_ON":
                    reference = base.interpolated_reference(raw_rows[edge], raw_rows[edge + 1], alpha)
                    target_process = process_metrics(target_p, target_r, target_quat, reference, contract)
                    sampled_process = process_metrics(sampled_p, sampled_r, sampled_quat, reference, contract)
                joint_violation = bool(np.any(sampled_q < lower - 1e-10) or np.any(sampled_q > upper + 1e-10))
                records.append({
                    "primitive_id": event["primitive_id"],
                    "event_index": int(event["event_index"]),
                    "point_kind": point_name,
                    "waypoint_index": int(event["waypoint_index"]),
                    "attempt_index": int(event["attempt_index"]),
                    "duration_extension_factor": float(event["duration_extension_factor"]),
                    "joint_index": int(point["joint_index"]),
                    "joint_space_excursion_rad": float(point["abs_overshoot_rad"]),
                    "tcp_translation_excursion_m": float(np.linalg.norm(sampled_p - target_p)),
                    "tcp_orientation_excursion_deg": base.quaternion_error_deg(sampled_quat, target_quat),
                    "standoff_deviation_from_target_m": None if sampled_process is None else abs(float(sampled_process["standoff_m"]) - float(target_process["standoff_m"])),
                    "surface_normal_deviation_from_target_deg": None if sampled_process is None else abs(float(sampled_process["surface_normal_deviation_deg"]) - float(target_process["surface_normal_deviation_deg"])),
                    "sampled_process_metrics": sampled_process,
                    "target_process_metrics": target_process,
                    "reference_edge_local": [edge, edge + 1],
                    "reference_alpha": alpha,
                    "target_joint_path_projection_error_rad": target_path_error,
                    "moveit_heuristic_overshoot": True,
                    "joint_limit_violation": joint_violation,
                    "tcp_process_violation": None if sampled_process is None else not bool(sampled_process["process_tolerance_pass"]),
                    "collision_violation": not bool(collision["collision_free"]),
                    "collision": collision,
                })

    summary = {
        "schema_version": "stage3-h7-5-overshoot-physical-effect-v1",
        "status": "PASSED" if limit_audit.get("status") == "PASSED" and len(records) == 2 * len(requests["events"]) else "BLOCKED",
        "runtime_limit_audit": limit_audit,
        "overshoot_event_count": len(requests["events"]),
        "physical_point_count": len(records),
        "records": records,
        "maximum_joint_space_excursion_rad": max((row["joint_space_excursion_rad"] for row in records), default=None),
        "maximum_tcp_translation_excursion_m": max((row["tcp_translation_excursion_m"] for row in records), default=None),
        "maximum_tcp_orientation_excursion_deg": max((row["tcp_orientation_excursion_deg"] for row in records), default=None),
        "joint_limit_violation_count": sum(bool(row["joint_limit_violation"]) for row in records),
        "tcp_process_violation_count": sum(row["tcp_process_violation"] is True for row in records),
        "tcp_process_violating_event_count": len({int(row["event_index"]) for row in records if row["tcp_process_violation"] is True}),
        "collision_violation_count": sum(bool(row["collision_violation"]) for row in records),
        "collision_violating_event_count": len({int(row["event_index"]) for row in records if row["collision_violation"]}),
        "collision_method": "adaptive_discrete_interpolation",
        "CCD": "NOT_AVAILABLE",
        "CLEARANCE": None,
        "NEW_FJT_GOALS_SENT": 0,
        "ROBOT_MOTION_STARTED": "NO",
        "FORMAL_LEDGER_MUTATED": "NO",
    }
    base.dump_json(output / "stage3_h7_5_overshoot_physical_effect.json", summary)
    return summary


def main() -> int:
    rclpy.init()
    node = rclpy.create_node("stage3_h7_5_physical_native")
    output = Path(str(base.param(node, "output_dir"))).resolve()
    moveit: MoveItPy | None = None
    return_code = 2
    lifecycle(output, "rclpy_initialized")
    try:
        moveit = MoveItPy(node_name="stage3_h7_5_physical_native")
        lifecycle(output, "moveitpy_constructed")
        result = formal_run(node, moveit)
        return_code = 0 if result["status"] == "PASSED" else 2
        lifecycle(output, "formal_run_complete", return_code=return_code)
    except Exception as exc:
        base.dump_json(output / "stage3_h7_5_overshoot_physical_effect.json", {
            "schema_version": "stage3-h7-5-overshoot-physical-effect-v1",
            "status": "BLOCKED",
            "first_blocker": f"native_exception:{type(exc).__name__}:{exc}",
            "NEW_FJT_GOALS_SENT": 0,
            "ROBOT_MOTION_STARTED": "NO",
            "FORMAL_LEDGER_MUTATED": "NO",
        })
        lifecycle(output, "formal_run_exception", exception=repr(exc))
    finally:
        if moveit is not None:
            lifecycle(output, "before_moveitpy_shutdown")
            moveit.shutdown()
            lifecycle(output, "after_moveitpy_shutdown")
            del moveit
            lifecycle(output, "after_moveitpy_delete")
            gc.collect()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return return_code


if __name__ == "__main__":
    sys.exit(main())
