#!/usr/bin/env python3
"""MoveIt2-only software recertification of a small H11 prediction subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rclpy
from moveit.planning import MoveItPy

import stage3_h10_native as h10
import stage3_h7_2_native as base
import stage3_h7_3_native as h73


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def run(args, moveit: MoveItPy) -> dict:
    rows = load_jsonl(Path(args.input_jsonl))
    validation = load_jsonl(Path(args.h6_validation))
    segments = load_json(Path(args.h6_segments))
    contract = load_json(Path(args.process_contract))
    primitives = h73.derive_primitives(validation, segments)
    primitive_map = {str(item["primitive_id"]): item for item in primitives}
    grouped = {}
    for row in rows:
        grouped.setdefault((str(row["split_role"]), str(row["trajectory_family_id"]), int(row["window_id"])), []).append(row)
    try:
        base.apply_fixture(moveit, Path(args.fixture_mesh))
        limits = base.runtime_limits(moveit, Path(args.runtime_limits))
        if limits.get("status") != "PASSED":
            return {"schema_version": "stage3_h11_native_prediction_validation_v1", "status": "BLOCKED", "first_blocker": "native_prediction_validation_unavailable", "reason": "runtime_joint_limit_audit_failed", "runtime_limit_audit": limits, "RAW_MODEL_COLLISION_VIOLATIONS": None, "RAW_MODEL_PROCESS_TOLERANCE_VIOLATIONS": None, "CCD_AVAILABLE": "NO", "CLEARANCE_AVAILABLE": None, "COLLISION_METHOD": "adaptive_discrete_interpolation", "native_backend_executed": False, "planning_scene_executed": False, "fk_executed": False, "dynamics_executed": False, "PHYSICAL_ROBOT_CONNECTED": "NO", "PHYSICAL_DRIVER_LOADED": "NO", "PHYSICAL_FJT_GOALS_SENT": 0, "ROBOT_MOTION_STARTED": "NO"}
        summaries = []
        collision_violations = 0
        process_violations = 0
        for key, group in sorted(grouped.items()):
            group.sort(key=lambda item: int(item["trajectory_index"]))
            primitive = primitive_map.get(str(group[0]["primitive_id"]))
            if primitive is None:
                return {"schema_version": "stage3_h11_native_prediction_validation_v1", "status": "BLOCKED", "first_blocker": "native_prediction_validation_unavailable", "reason": f"primitive_missing:{group[0]['primitive_id']}", "RAW_MODEL_COLLISION_VIOLATIONS": None, "RAW_MODEL_PROCESS_TOLERANCE_VIOLATIONS": None, "CCD_AVAILABLE": "NO", "CLEARANCE_AVAILABLE": None, "COLLISION_METHOD": "adaptive_discrete_interpolation"}
            native_rows = [{"trajectory_family_id": group[0]["trajectory_family_id"], "segment_order": int(row["segment_order"]), "segment_id": int(row["segment_id"]), "primitive_id": row["primitive_id"], "spray_state": row["spray_state"], "trajectory_index": int(row["trajectory_index"]), "time_from_start_s": float(row["time_from_start_s"]), "joint_names": list(base.JOINT_NAMES), "positions_rad": list(row["positions_rad"]), "velocities_rad_s": list(row["velocities_rad_s"]), "accelerations_rad_s2": list(row["accelerations_rad_s2"])} for row in group]
            check, _ = h10.native_recheck(moveit, primitive, native_rows, limits["runtime_bounds"], contract)
            collision = check.get("collision", {})
            process = check.get("process", {})
            collision_count = int(collision.get("collision_failure_count", 0) or 0)
            process_count = int(process.get("failed_spray_on_sample_count", 0) or 0)
            collision_violations += collision_count
            process_violations += process_count
            summaries.append({"split_role": key[0], "trajectory_family_id": key[1], "window_id": key[2], "primitive_id": group[0]["primitive_id"], "status": check.get("status"), "check": check, "raw_model_output": True, "post_processing": "none"})
        return {"schema_version": "stage3_h11_native_prediction_validation_v1", "status": "PASSED", "first_blocker": None, "scope": "one deterministic raw rollout window per TEST and GENERALIZATION family", "evaluated_window_count": len(summaries), "summaries": summaries, "RAW_MODEL_COLLISION_VIOLATIONS": collision_violations, "RAW_MODEL_PROCESS_TOLERANCE_VIOLATIONS": process_violations, "CCD_AVAILABLE": "NO", "CLEARANCE_AVAILABLE": None, "COLLISION_METHOD": "adaptive_discrete_interpolation", "native_backend_executed": True, "planning_scene_executed": True, "fk_executed": True, "dynamics_executed": True, "post_ruckig_executed": True, "PHYSICAL_ROBOT_CONNECTED": "NO", "PHYSICAL_DRIVER_LOADED": "NO", "PHYSICAL_FJT_GOALS_SENT": 0, "ROBOT_MOTION_STARTED": "NO"}
    except Exception as exc:
        return {"schema_version": "stage3_h11_native_prediction_validation_v1", "status": "BLOCKED", "first_blocker": "native_prediction_validation_unavailable", "reason": f"native_exception:{type(exc).__name__}:{exc}", "RAW_MODEL_COLLISION_VIOLATIONS": None, "RAW_MODEL_PROCESS_TOLERANCE_VIOLATIONS": None, "CCD_AVAILABLE": "NO", "CLEARANCE_AVAILABLE": None, "COLLISION_METHOD": "adaptive_discrete_interpolation", "native_backend_executed": False, "planning_scene_executed": False, "fk_executed": False, "dynamics_executed": False, "PHYSICAL_ROBOT_CONNECTED": "NO", "PHYSICAL_DRIVER_LOADED": "NO", "PHYSICAL_FJT_GOALS_SENT": 0, "ROBOT_MOTION_STARTED": "NO"}
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--h6-validation", required=True)
    parser.add_argument("--h6-segments", required=True)
    parser.add_argument("--process-contract", required=True)
    parser.add_argument("--fixture-mesh", required=True)
    parser.add_argument("--runtime-limits", required=True)
    args, _ros_args = parser.parse_known_args()
    rclpy.init()
    try:
        moveit = MoveItPy(node_name="stage3_h11_native")
        result = run(args, moveit)
    except Exception as exc:
        result = {
            "schema_version": "stage3_h11_native_prediction_validation_v1",
            "status": "BLOCKED",
            "first_blocker": "native_prediction_validation_unavailable",
            "reason": f"native_exception:{type(exc).__name__}:{exc}",
            "RAW_MODEL_COLLISION_VIOLATIONS": None,
            "RAW_MODEL_PROCESS_TOLERANCE_VIOLATIONS": None,
            "CCD_AVAILABLE": "NO",
            "CLEARANCE_AVAILABLE": None,
            "COLLISION_METHOD": "adaptive_discrete_interpolation",
            "native_backend_executed": False,
            "planning_scene_executed": False,
            "fk_executed": False,
            "dynamics_executed": False,
            "PHYSICAL_ROBOT_CONNECTED": "NO",
            "PHYSICAL_DRIVER_LOADED": "NO",
            "PHYSICAL_FJT_GOALS_SENT": 0,
            "ROBOT_MOTION_STARTED": "NO",
        }
    dump(Path(args.output_json), result)
    rclpy.shutdown()
    return 0 if result.get("status") == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
