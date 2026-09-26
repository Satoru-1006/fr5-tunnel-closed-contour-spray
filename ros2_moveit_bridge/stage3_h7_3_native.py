#!/usr/bin/env python3
"""Native MoveIt worker for Stage 3 H7.3.

The frozen H6.4 segment 3 is split only for time parameterization at the
193|194 boundary.  All ten resulting motion primitives are audited with the
same waypoint filter used by MoveIt TOTG before native TOTG/Ruckig calls.
No planning, IK, controller, action, or robot execution API is used.
"""

from __future__ import annotations

import gc
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from moveit.core.robot_state import RobotState
from moveit.core.robot_trajectory import RobotTrajectory
from moveit.planning import MoveItPy

import stage3_h7_2_native as base


PRIMITIVE_ORDER: list[int | str] = [0, 1000, 1, 1001, 2, 1002, "3A", "3B", 1003, 4]
ANGLE_TOLERANCE = 1.0e-5


def derive_primitives(
    validation_rows: Sequence[Mapping[str, Any]], segment_manifest: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Build the authorized 10 primitives without changing any source row."""
    original = base.group_input_rows(validation_rows, segment_manifest)
    by_id = {int(item["segment_id"]): item for item in original}
    result: list[dict[str, Any]] = []
    for primitive_id in PRIMITIVE_ORDER:
        if primitive_id == "3A":
            source = by_id[3]
            rows = [dict(row) for row in source["rows"][:194]]
        elif primitive_id == "3B":
            source = by_id[3]
            rows = [dict(row) for row in source["rows"][194:]]
        else:
            source = by_id[int(primitive_id)]
            rows = [dict(row) for row in source["rows"]]
        result.append(
            {
                "primitive_id": primitive_id,
                "segment_id": int(source["segment_id"]),
                "spray_state": source["spray_state"],
                "metadata": dict(source["metadata"]),
                "rows": rows,
                "derived_from_h6_4_without_joint_change": True,
            }
        )
    if [len(item["rows"]) for item in result] != [221, 12, 40, 426, 4, 241, 194, 104, 361, 25]:
        raise RuntimeError("derived primitive waypoint counts do not match the authorized H7.3 split")
    return result


def unwound_positions(moveit: MoveItPy, primitive: Mapping[str, Any]) -> np.ndarray:
    start_state = RobotState(moveit.get_robot_model())
    trajectory = RobotTrajectory(moveit.get_robot_model())
    trajectory.joint_model_group_name = base.GROUP_NAME
    trajectory.set_robot_trajectory_msg(start_state, base.make_trajectory_message(primitive["rows"]))
    trajectory.unwind()
    return np.asarray(
        [list(point.positions) for point in trajectory.get_robot_trajectory_msg().joint_trajectory.points], dtype=float
    )


def simulate_moveit_filter(
    positions: np.ndarray, rows: Sequence[Mapping[str, Any]], min_angle_change: float
) -> dict[str, Any]:
    """Mirror Jazzy TOTG doTimeParameterizationCalculations point filtering."""
    points: list[np.ndarray] = []
    source_indices: list[int] = []
    filtered_indices: list[int] = []
    last_index = len(positions) - 1
    for index, point in enumerate(positions):
        diverse = index == 0 or bool(np.any(np.abs(point - points[-1]) > min_angle_change))
        if diverse:
            points.append(point.copy())
            source_indices.append(index)
        elif index == last_index:
            points[-1] = point.copy()
            source_indices[-1] = index
            filtered_indices.append(index)
        else:
            filtered_indices.append(index)
    turns: list[dict[str, Any]] = []
    for index in range(1, len(points) - 1):
        incoming = points[index] - points[index - 1]
        outgoing = points[index + 1] - points[index]
        incoming_norm = float(np.linalg.norm(incoming))
        outgoing_norm = float(np.linalg.norm(outgoing))
        if incoming_norm <= np.finfo(float).eps or outgoing_norm <= np.finfo(float).eps:
            continue
        dot = float(np.dot(incoming, outgoing))
        cosine = float(np.clip(dot / (incoming_norm * outgoing_norm), -1.0, 1.0))
        if cosine <= -1.0 + ANGLE_TOLERANCE:
            source = [source_indices[index - 1], source_indices[index], source_indices[index + 1]]
            turns.append(
                {
                    "filtered_sequence_indices": [index - 1, index, index + 1],
                    "primitive_local_source_indices": source,
                    "h6_4_storage_waypoint_indices": [int(rows[i]["h6_4_storage_waypoint_index"]) for i in source],
                    "h6_4_process_waypoint_indices": [int(rows[i]["h6_4_process_waypoint_index"]) for i in source],
                    "joint_vectors_rad": [points[index - 1].tolist(), points[index].tolist(), points[index + 1].tolist()],
                    "incoming_tangent_rad": incoming.tolist(),
                    "outgoing_tangent_rad": outgoing.tolist(),
                    "incoming_norm_rad": incoming_norm,
                    "outgoing_norm_rad": outgoing_norm,
                    "dot_product_rad2": dot,
                    "cosine": cosine,
                    "turn_angle_deg": math.degrees(math.acos(cosine)),
                }
            )
    return {
        "input_waypoint_count": len(positions),
        "filtered_waypoint_count": len(points),
        "removed_or_replaced_primitive_local_indices": filtered_indices,
        "kept_primitive_local_source_indices": source_indices,
        "unsupported_180_turn_count": len(turns),
        "unsupported_180_turns": turns,
        "algorithm": "MoveIt Jazzy TOTG: keep first; keep a point iff any active joint differs from the last kept point by > min_angle_change; replace the last kept point with a non-diverse final endpoint",
    }


def pre_totg_audit(moveit: MoveItPy, primitives: Sequence[Mapping[str, Any]], min_angle_change: float) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for primitive in primitives:
        audit = simulate_moveit_filter(unwound_positions(moveit, primitive), primitive["rows"], min_angle_change)
        records.append({"primitive_id": primitive["primitive_id"], **audit})
    count = sum(int(item["unsupported_180_turn_count"]) for item in records)
    return {
        "schema_version": "stage3-h7-3-native-pre-totg-filtered-path-audit-v1",
        "status": "PASSED" if count == 0 else "BLOCKED",
        "min_angle_change_rad": min_angle_change,
        "moveit_filtered_180_turn_count": count,
        "primitive_audits": records,
        "authoritative_primitive_order": PRIMITIVE_ORDER,
    }


def run_primitive(
    moveit: MoveItPy,
    primitive: Mapping[str, Any],
    limits: Mapping[str, Any],
    totg: Mapping[str, Any],
    ruckig_authorized: bool,
    contract: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    result, totg_rows, final_rows, validation_rows = base.run_segment(
        moveit, primitive, limits, totg, ruckig_authorized, contract
    )
    primitive_id = primitive["primitive_id"]
    result["primitive_id"] = primitive_id
    result["h6_4_segment_id"] = int(primitive["segment_id"])
    for collection in (totg_rows, final_rows, validation_rows):
        for row in collection:
            row["primitive_id"] = primitive_id
            row["schema_version"] = str(row.get("schema_version", "")).replace("h7-2", "h7-3")
    return result, totg_rows, final_rows, validation_rows


def formal_run(node: Any, moveit: MoveItPy) -> dict[str, Any]:
    output = Path(str(base.param(node, "output_dir"))).resolve()
    validation_path = Path(str(base.param(node, "h6_4_final_validation"))).resolve()
    segments_path = Path(str(base.param(node, "h6_4_segments"))).resolve()
    contract_path = Path(str(base.param(node, "process_contract"))).resolve()
    mesh_path = Path(str(base.param(node, "fixture_mesh"))).resolve()
    totg_path = Path(str(base.param(node, "totg_parameters"))).resolve()
    ruckig_authorized = bool(base.param(node, "ruckig_authorized", False))
    output.mkdir(parents=True, exist_ok=True)

    configured_limits = Path(get_package_share_directory("fr5_tunnel_moveit_bridge")) / "config/joint_limits_with_jerk.yaml"
    limit_audit = base.runtime_limits(moveit, configured_limits)
    base.dump_json(output / "runtime_limit_audit.json", limit_audit)
    if limit_audit["status"] != "PASSED":
        result = {
            "schema_version": "stage3-h7-3-native-result-v1",
            "status": "BLOCKED",
            "first_blocker": "runtime_joint_limit_audit_failed",
            "primitive_results": [],
            "NEW_FJT_GOALS_SENT": 0,
            "ROBOT_MOTION_STARTED": "NO",
            "FORMAL_LEDGER_MUTATED": "NO",
        }
        base.dump_json(output / "native_result.json", result)
        return result

    base.apply_fixture(moveit, mesh_path)
    primitives = derive_primitives(base.load_jsonl(validation_path), base.load_json(segments_path))
    contract = base.load_json(contract_path)
    totg = base.load_json(totg_path)
    filtered_audit = pre_totg_audit(moveit, primitives, float(totg["min_angle_change"]))
    base.dump_json(output / "native_pre_totg_filtered_path_audit.json", filtered_audit)
    if filtered_audit["status"] != "PASSED":
        result = {
            "schema_version": "stage3-h7-3-native-result-v1",
            "status": "BLOCKED",
            "first_blocker": "moveit_filtered_180_turn_remains",
            "authoritative_primitive_order": PRIMITIVE_ORDER,
            "primitive_results": [],
            "totg_primitives_passed": 0,
            "ruckig_primitives_passed": 0,
            "NEW_FJT_GOALS_SENT": 0,
            "ROBOT_MOTION_STARTED": "NO",
            "FORMAL_LEDGER_MUTATED": "NO",
        }
        base.dump_json(output / "native_result.json", result)
        return result

    results: list[dict[str, Any]] = []
    all_totg_rows: list[dict[str, Any]] = []
    all_final_rows: list[dict[str, Any]] = []
    all_validation_rows: list[dict[str, Any]] = []

    # All ten primitives must pass native TOTG and post-TOTG validation before
    # the first native Ruckig call is authorized.
    for primitive in primitives:
        result, totg_rows, _final_rows, validation_rows = run_primitive(
            moveit, primitive, limit_audit["runtime_bounds"], totg, False, contract
        )
        results.append(result)
        all_totg_rows.extend(totg_rows)
        all_validation_rows.extend(validation_rows)
        if result["status"] != "PASSED":
            break
    totg_precondition_passed = len(results) == 10 and all(
        row.get("post_totg_status") == "PASSED" for row in results
    )

    if totg_precondition_passed and ruckig_authorized:
        results = []
        all_totg_rows = []
        all_validation_rows = []
        for primitive in primitives:
            result, totg_rows, final_rows, validation_rows = run_primitive(
                moveit, primitive, limit_audit["runtime_bounds"], totg, True, contract
            )
            results.append(result)
            all_totg_rows.extend(totg_rows)
            all_final_rows.extend(final_rows)
            all_validation_rows.extend(validation_rows)
            if result["status"] != "PASSED":
                break

    base.dump_jsonl(output / "totg_trajectories.jsonl", all_totg_rows)
    base.dump_jsonl(output / "ruckig_trajectories.jsonl", all_final_rows)
    base.dump_jsonl(output / "validation_rows.jsonl", all_validation_rows)
    passed = len(results) == 10 and all(row["status"] == "PASSED" for row in results)
    summary = {
        "schema_version": "stage3-h7-3-native-result-v1",
        "status": "PASSED" if passed else "BLOCKED",
        "first_blocker": next(
            (
                f"primitive_{row.get('primitive_id')}:{row.get('first_blocker', 'primitive_validation_failed')}"
                for row in results
                if row["status"] != "PASSED"
            ),
            None,
        ),
        "authoritative_primitive_order": PRIMITIVE_ORDER,
        "primitive_results": results,
        "moveit_filtered_180_turn_count": filtered_audit["moveit_filtered_180_turn_count"],
        "totg_primitives_passed": sum(row.get("post_totg_status") == "PASSED" for row in results),
        "totg_global_precondition_passed_before_ruckig": totg_precondition_passed,
        "ruckig_authorized": ruckig_authorized,
        "ruckig_primitives_passed": sum(row.get("ruckig_status") == "PASSED" for row in results),
        "semantic_hash": base.semantic_hash(
            {"primitives": results, "totg_trajectories": all_totg_rows, "ruckig_trajectories": all_final_rows}
        ),
        "NEW_FJT_GOALS_SENT": 0,
        "ROBOT_MOTION_STARTED": "NO",
        "FORMAL_LEDGER_MUTATED": "NO",
    }
    base.dump_json(output / "native_result.json", summary)
    return summary


def main() -> int:
    rclpy.init()
    node = rclpy.create_node("stage3_h7_3_native")
    output = Path(str(base.param(node, "output_dir"))).resolve()
    output.mkdir(parents=True, exist_ok=True)
    moveit: MoveItPy | None = None
    return_code = 2
    try:
        moveit = MoveItPy(node_name="stage3_h7_3_native")
        result = formal_run(node, moveit)
        return_code = 0 if result["status"] == "PASSED" else 2
    except Exception as exc:
        result = {
            "schema_version": "stage3-h7-3-native-result-v1",
            "status": "BLOCKED",
            "first_blocker": f"native_exception:{type(exc).__name__}:{exc}",
            "primitive_results": [],
            "NEW_FJT_GOALS_SENT": 0,
            "ROBOT_MOTION_STARTED": "NO",
            "FORMAL_LEDGER_MUTATED": "NO",
        }
        base.dump_json(output / "native_result.json", result)
    finally:
        # MoveItPy owns background resources.  Destroy it while rclpy is still
        # alive, then tear down the node/context, to avoid the H7.2 exit -11.
        if moveit is not None:
            moveit.shutdown()
            del moveit
            gc.collect()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return return_code


if __name__ == "__main__":
    sys.exit(main())
