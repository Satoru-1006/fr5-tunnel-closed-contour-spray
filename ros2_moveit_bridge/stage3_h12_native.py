#!/usr/bin/env python3
"""Full-scope native MoveIt2 certification worker for H12 repaired windows.

This worker is software-only.  It consumes a frozen NumPy input bundle and
reuses the H10 native FK, process, dynamics, and PlanningScene/FCL checks.  A
nonzero failure or an incomplete window universe is returned as BLOCKED.
"""

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


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def run(args, moveit: MoveItPy) -> dict:
    bundle = np.load(Path(args.input_npz), allow_pickle=False)
    positions = np.asarray(bundle["positions_rad"], dtype=np.float64)
    velocities = np.asarray(bundle["velocities_rad_s"], dtype=np.float64)
    accelerations = np.asarray(bundle["accelerations_rad_s2"], dtype=np.float64)
    target_times = np.asarray(bundle["target_times_s"], dtype=np.float64)
    windows = load_json(Path(args.window_manifest))["windows"]
    if not (positions.shape == velocities.shape == accelerations.shape == (len(windows), 8, 6) and target_times.shape == (len(windows), 8)):
        return blocked("native_input_shape_mismatch", native_backend_executed=False)
    validation = [json.loads(line) for line in Path(args.h6_validation).read_text(encoding="utf-8").splitlines() if line.strip()]
    segments = load_json(Path(args.h6_segments))
    contract = load_json(Path(args.process_contract))
    primitives = h73.derive_primitives(validation, segments)
    primitive_map = {str(item["primitive_id"]): item for item in primitives}
    try:
        base.apply_fixture(moveit, Path(args.fixture_mesh))
        limits = base.runtime_limits(moveit, Path(args.runtime_limits))
        if limits.get("status") != "PASSED":
            return blocked("runtime_joint_limit_audit_failed", native_backend_executed=True, runtime_limit_audit=limits)
        failures_path = Path(args.output_json).with_name("stage3_h12_native_failures.jsonl")
        summary_path = Path(args.output_json).with_name("stage3_h12_native_progress.json")
        checked = 0
        collision = 0
        self_collision = 0
        environment_collision = 0
        process = 0
        position = 0
        velocity = 0
        acceleration = 0
        jerk = 0
        first_failure = None
        with failures_path.open("w", encoding="utf-8", newline="\n") as failures_handle:
            for index, item in enumerate(windows):
                primitive = primitive_map.get(str(item["primitive_id"]))
                if primitive is None:
                    return blocked(f"primitive_missing:{item['primitive_id']}", native_backend_executed=True)
                t = target_times[index] - target_times[index, 0]
                native_rows = [{
                    "trajectory_family_id": item["trajectory_family_id"],
                    "segment_order": int(item["segment_order"]),
                    "segment_id": int(item["segment_id"]),
                    "primitive_id": item["primitive_id"],
                    "spray_state": item["spray_state"],
                    "trajectory_index": step,
                    "time_from_start_s": float(t[step]),
                    "joint_names": list(base.JOINT_NAMES),
                    "positions_rad": positions[index, step].tolist(),
                    "velocities_rad_s": velocities[index, step].tolist(),
                    "accelerations_rad_s2": accelerations[index, step].tolist(),
                } for step in range(8)]
                check, process_rows = h10.native_recheck(moveit, primitive, native_rows, limits["runtime_bounds"], contract)
                execution = check.get("execution_limits", {})
                collision_check = check.get("collision", {})
                process_check = check.get("process", {})
                collision_count = int(collision_check.get("collision_failure_count", 0) or 0)
                self_count = int(collision_check.get("self_collision_failure_count", 0) or 0)
                environment_count = int(collision_check.get("environment_collision_failure_count", 0) or 0)
                process_count = int(process_check.get("failed_spray_on_sample_count", 0) or 0)
                collision += collision_count
                self_collision += self_count
                environment_collision += environment_count
                process += process_count
                position += int(execution.get("position_violation_count", 0) or 0)
                velocity += int(execution.get("velocity_violation_count", 0) or 0)
                acceleration += int(execution.get("acceleration_violation_count", 0) or 0)
                jerk += int(execution.get("jerk_violation_count", 0) or 0)
                checked += 1
                if check.get("status") != "PASSED":
                    failure = {"window_index": index, "window_id": item["window_id"], "family": item["trajectory_family_id"], "segment": item["segment_order"], "check": check}
                    failures_handle.write(json.dumps(failure, ensure_ascii=True, sort_keys=True, allow_nan=False) + "\n")
                    if first_failure is None:
                        first_failure = failure
                if checked % 100 == 0:
                    dump(summary_path, {"status": "RUNNING", "evaluated_window_count": checked, "required_window_count": len(windows)})
        complete = checked == len(windows)
        status = "PASSED" if complete and not any((collision, process, position, velocity, acceleration, jerk)) else "BLOCKED"
        return {
            "schema_version": "stage3_h12_native_repaired_validation_v1",
            "status": status,
            "first_blocker": None if status == "PASSED" else ("post_repair_process_tolerance_violation" if process else "post_repair_native_constraint_violation"),
            "scope": "all frozen H11 windows; no sampling",
            "evaluated_window_count": checked,
            "required_window_count": len(windows),
            "complete_native_scope": complete,
            "POST_REPAIR_POSITION_LIMIT_VIOLATIONS": position,
            "POST_REPAIR_VELOCITY_LIMIT_VIOLATIONS": velocity,
            "POST_REPAIR_ACCELERATION_LIMIT_VIOLATIONS": acceleration,
            "POST_REPAIR_JERK_LIMIT_VIOLATIONS": jerk,
            "POST_REPAIR_SELF_COLLISION_VIOLATIONS": self_collision,
            "POST_REPAIR_ENVIRONMENT_COLLISION_VIOLATIONS": environment_collision,
            "POST_REPAIR_COLLISION_VIOLATIONS": collision,
            "POST_REPAIR_PROCESS_TOLERANCE_VIOLATIONS": process,
            "POST_REPAIR_HARD_CONSTRAINT_VIOLATIONS": int(position + velocity + acceleration + jerk),
            "first_failure": first_failure,
            "failure_evidence": str(failures_path.resolve()),
            "native_backend_executed": True,
            "planning_scene_executed": True,
            "fk_executed": True,
            "dynamics_executed": True,
            "post_ruckig_executed": False,
            "time_parameterization": "causal_discrete_jerk_limited_projection; native Ruckig not invoked by this worker",
            "COLLISION_METHOD": base.COLLISION_METHOD,
            "CCD_AVAILABLE": "NO",
            "CLEARANCE_AVAILABLE": None,
            "PHYSICAL_ROBOT_CONNECTED": "NO",
            "PHYSICAL_DRIVER_LOADED": "NO",
            "PHYSICAL_FJT_GOALS_SENT": 0,
            "ROBOT_MOTION_STARTED": "NO",
        }
    except Exception as exc:
        return blocked(f"native_exception:{type(exc).__name__}:{exc}", native_backend_executed=False)


def blocked(reason: str, native_backend_executed: bool, **extra) -> dict:
    return {
        "schema_version": "stage3_h12_native_repaired_validation_v1",
        "status": "BLOCKED",
        "first_blocker": reason,
        "scope": "all frozen H11 windows; no sampling",
        "evaluated_window_count": 0,
        "required_window_count": None,
        "complete_native_scope": False,
        "POST_REPAIR_POSITION_LIMIT_VIOLATIONS": None,
        "POST_REPAIR_VELOCITY_LIMIT_VIOLATIONS": None,
        "POST_REPAIR_ACCELERATION_LIMIT_VIOLATIONS": None,
        "POST_REPAIR_JERK_LIMIT_VIOLATIONS": None,
        "POST_REPAIR_SELF_COLLISION_VIOLATIONS": None,
        "POST_REPAIR_ENVIRONMENT_COLLISION_VIOLATIONS": None,
        "POST_REPAIR_COLLISION_VIOLATIONS": None,
        "POST_REPAIR_PROCESS_TOLERANCE_VIOLATIONS": None,
        "POST_REPAIR_HARD_CONSTRAINT_VIOLATIONS": None,
        "native_backend_executed": native_backend_executed,
        "planning_scene_executed": False,
        "fk_executed": False,
        "dynamics_executed": False,
        "post_ruckig_executed": False,
        "COLLISION_METHOD": base.COLLISION_METHOD,
        "CCD_AVAILABLE": "NO",
        "CLEARANCE_AVAILABLE": None,
        "PHYSICAL_ROBOT_CONNECTED": "NO",
        "PHYSICAL_DRIVER_LOADED": "NO",
        "PHYSICAL_FJT_GOALS_SENT": 0,
        "ROBOT_MOTION_STARTED": "NO",
        **extra,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-npz", required=True)
    parser.add_argument("--window-manifest", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--h6-validation", required=True)
    parser.add_argument("--h6-segments", required=True)
    parser.add_argument("--process-contract", required=True)
    parser.add_argument("--fixture-mesh", required=True)
    parser.add_argument("--runtime-limits", required=True)
    args, _ros_args = parser.parse_known_args()
    output = Path(args.output_json)
    rclpy.init()
    moveit = None
    try:
        moveit = MoveItPy(node_name="stage3_h12_native")
        result = run(args, moveit)
    except Exception as exc:
        result = blocked(f"native_exception:{type(exc).__name__}:{exc}", native_backend_executed=False)
    finally:
        if moveit is not None:
            moveit.shutdown()
        rclpy.shutdown()
    dump(output, result)
    return 0 if result.get("status") == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())

