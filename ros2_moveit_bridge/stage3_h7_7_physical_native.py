#!/usr/bin/env python3
"""Streaming MoveIt FK/FCL certification of every H7.7 native sample."""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from moveit.core.robot_state import RobotState
from moveit.planning import MoveItPy

import stage3_h7_2_native as base
import stage3_h7_3_native as h73
import stage3_h7_5_physical_native as physical
from stage3_h7_native import collision_state


def lifecycle(output: Path, event: str, **details: Any) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "lifecycle_events.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps({"event": event, **details}, sort_keys=True) + "\n")


def formal_run(node: Any, moveit: MoveItPy) -> dict[str, Any]:
    output = Path(str(base.param(node, "output_dir"))).resolve()
    samples_path = Path(str(base.param(node, "native_samples"))).resolve()
    validation_path = Path(str(base.param(node, "h6_4_final_validation"))).resolve()
    segments_path = Path(str(base.param(node, "h6_4_segments"))).resolve()
    contract_path = Path(str(base.param(node, "process_contract"))).resolve()
    mesh_path = Path(str(base.param(node, "fixture_mesh"))).resolve()
    shard_index = int(base.param(node, "shard_index", 0))
    shard_count = int(base.param(node, "shard_count", 1))
    if not 0 <= shard_index < shard_count:
        raise RuntimeError("invalid physical certification shard")

    contract = base.load_json(contract_path)
    primitives = h73.derive_primitives(base.load_jsonl(validation_path), base.load_json(segments_path))
    order = h73.PRIMITIVE_ORDER
    by_id = {str(item["primitive_id"]): item for item in primitives}
    raw_q = {key: np.asarray([row["joint_values"] for row in value["rows"]], dtype=float) for key, value in by_id.items()}
    configured = Path(get_package_share_directory("fr5_tunnel_moveit_bridge")) / "config/joint_limits_with_jerk.yaml"
    limit_audit = base.runtime_limits(moveit, configured)
    base.apply_fixture(moveit, mesh_path)
    bounds = limit_audit.get("runtime_bounds", {})
    lower = np.asarray([bounds[name]["position_lower_rad"] for name in base.JOINT_NAMES], dtype=float)
    upper = np.asarray([bounds[name]["position_upper_rad"] for name in base.JOINT_NAMES], dtype=float)

    counts = {
        "native_sample_count": 0, "processed_state_count": 0, "spray_on_state_count": 0,
        "position_limit_violations": 0, "process_violations": 0,
        "standoff_violations": 0, "normal_violations": 0,
        "tcp_position_violations": 0, "tcp_orientation_violations": 0,
        "self_collision_violations": 0, "environment_collision_violations": 0,
    }
    maxima = {"standoff_error_m": 0.0, "surface_normal_deviation_deg": 0.0, "tcp_position_error_m": 0.0, "tcp_orientation_error_deg": 0.0}
    worst: dict[str, Any] = {key: None for key in maxima}
    first_process = None
    first_collision = None
    boundary_sides: dict[str, int] = {}
    invocation = 0
    previous_call = -1
    state = RobotState(moveit.get_robot_model())
    monitor = moveit.get_planning_scene_monitor()
    with monitor.read_only() as scene, samples_path.open("r", encoding="utf-8") as stream:
        for ordinal, line in enumerate(stream):
            if not line.strip():
                continue
            sample = json.loads(line)
            call_index = int(sample["global_calculate_call_index"])
            if previous_call >= 0 and call_index < previous_call:
                invocation += 1
            previous_call = call_index
            counts["native_sample_count"] += 1
            if ordinal % shard_count != shard_index:
                continue
            primitive_id = order[invocation]
            primitive = by_id[str(primitive_id)]
            q = np.asarray(sample["q"], dtype=float)
            counts["processed_state_count"] += 1
            side = str(sample.get("boundary_side", "unspecified"))
            boundary_sides[side] = boundary_sides.get(side, 0) + 1
            if np.any(q < lower - 1e-10) or np.any(q > upper + 1e-10):
                counts["position_limit_violations"] += 1

            position, rotation, quaternion = physical.pose(state, q)
            collision = collision_state(scene, state, q)
            if collision.get("self_collision"):
                counts["self_collision_violations"] += 1
            if collision.get("environment_collision"):
                counts["environment_collision_violations"] += 1
            if not collision.get("collision_free") and first_collision is None:
                first_collision = {"primitive_id": primitive_id, "call_index": call_index, "local_time_s": sample["local_time"], "q": sample["q"], "collision": collision}

            if primitive["spray_state"] == "SPRAY_ON":
                counts["spray_on_state_count"] += 1
                edge, alpha, _error = base.nearest_polyline_projection(q, raw_q[str(primitive_id)])
                reference = base.interpolated_reference(primitive["rows"][edge], primitive["rows"][edge + 1], alpha)
                metrics = physical.process_metrics(position, rotation, quaternion, reference, contract)
                if not metrics["process_tolerance_pass"]:
                    counts["process_violations"] += 1
                    if first_process is None:
                        first_process = {"primitive_id": primitive_id, "call_index": call_index, "local_time_s": sample["local_time"], "q": sample["q"], "metrics": metrics}
                tests = {
                    "standoff_error_m": ("standoff_violations", float(contract["geometry"]["standoff_abs_tolerance_m"])),
                    "surface_normal_deviation_deg": ("normal_violations", float(contract["geometry"]["normal_angle_tolerance_deg"])),
                    "tcp_position_error_m": ("tcp_position_violations", float(contract["tcp_reproduction"]["tcp_position_tolerance_m"])),
                    "tcp_orientation_error_deg": ("tcp_orientation_violations", float(contract["tcp_reproduction"]["tcp_orientation_tolerance_deg"])),
                }
                for key, (count_key, threshold) in tests.items():
                    value = float(metrics[key])
                    if value > threshold:
                        counts[count_key] += 1
                    if value > maxima[key]:
                        maxima[key] = value
                        worst[key] = {"primitive_id": primitive_id, "call_index": call_index, "local_time_s": sample["local_time"], "q": sample["q"], "value": value}
            if counts["processed_state_count"] % 10000 == 0:
                print(json.dumps({"progress": counts["processed_state_count"], "shard_index": shard_index, "process_violations": counts["process_violations"], "collision_violations": counts["self_collision_violations"] + counts["environment_collision_violations"]}), flush=True)

    collision_violations = counts["self_collision_violations"] + counts["environment_collision_violations"]
    summary = {
        "schema_version": "stage3-h7-7-native-physical-certification-v1",
        "status": "PASSED" if limit_audit.get("status") == "PASSED" and not counts["position_limit_violations"] and not counts["process_violations"] and not collision_violations else "BLOCKED",
        "sampling_basis": "every state emitted by the H7.4 native interposer union of endpoints, every per-DOF phase boundary, native extrema, and 0.01 s adaptive/dense grid",
        "native_samples": str(samples_path), "shard_index": shard_index, "shard_count": shard_count,
        "counts": counts, "boundary_side_counts": boundary_sides, "maxima": maxima, "worst": worst,
        "first_process_violation": first_process, "first_collision": first_collision,
        "collision_method": "adaptive_discrete_interpolation", "CCD": "NOT_AVAILABLE", "CLEARANCE": None,
        "NEW_FJT_GOALS_SENT": 0, "ROBOT_MOTION_STARTED": "NO", "FORMAL_LEDGER_MUTATED": "NO", "STAGE_3_H8_STARTED": "NO",
    }
    base.dump_json(output / "stage3_h7_7_native_physical_certification.json", summary)
    return summary


def main() -> int:
    rclpy.init()
    node = rclpy.create_node("stage3_h7_7_physical_native")
    output = Path(str(base.param(node, "output_dir"))).resolve()
    moveit: MoveItPy | None = None
    return_code = 2
    lifecycle(output, "rclpy_initialized")
    try:
        moveit = MoveItPy(node_name="stage3_h7_7_physical_native")
        lifecycle(output, "moveitpy_constructed")
        result = formal_run(node, moveit)
        return_code = 0 if result["status"] == "PASSED" else 2
        lifecycle(output, "formal_artifacts_flushed", return_code=return_code)
    except Exception as exc:
        base.dump_json(output / "stage3_h7_7_native_physical_certification.json", {"schema_version": "stage3-h7-7-native-physical-certification-v1", "status": "BLOCKED", "first_blocker": f"native_exception:{type(exc).__name__}:{exc}"})
        lifecycle(output, "formal_run_exception", exception=repr(exc))
    finally:
        if moveit is not None:
            moveit.shutdown()
            del moveit
            gc.collect()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    lifecycle(output, "main_return", return_code=return_code)
    return return_code


if __name__ == "__main__":
    sys.exit(main())
