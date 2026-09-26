#!/usr/bin/env python3
"""Resumable native H12-R shard validator.

Each invocation owns one fixed canonical window range.  It reuses the frozen
H10 MoveIt2/FK/process/FCL implementation and additionally runs native TOTG
and MoveIt Ruckig on the repaired window before the final native checks.  The
worker is software-only and never imports a controller or hardware driver.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import rclpy
from moveit.core.robot_state import RobotState
from moveit.core.robot_trajectory import RobotTrajectory
from moveit.planning import MoveItPy

import stage3_h10_native as h10
import stage3_h7_2_native as base
import stage3_h7_3_native as h73


SCHEMA_VERSION = "stage3_h12_r_native_shard_v1"
NATIVE_VALIDATOR_VERSION = "stage3_h12_r_native_moveit_fcl_ruckig_v1"
GROUP_NAME = base.GROUP_NAME


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_write(path, json.dumps(json_safe(value), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")


def atomic_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    atomic_write(path, "".join(json.dumps(json_safe(row), ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n" for row in rows))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def semantic_sha(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def json_safe(value: Any) -> Any:
    """Convert numpy scalars and non-finite floats to JSON-safe evidence."""
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def trajectory_rows(item: Mapping[str, Any], positions: np.ndarray, velocities: np.ndarray, accelerations: np.ndarray, times: np.ndarray) -> list[dict[str, Any]]:
    return [{
        "trajectory_family_id": item["trajectory_family_id"],
        "segment_order": int(item["segment_order"]),
        "segment_id": int(item["segment_id"]),
        "primitive_id": item["primitive_id"],
        "spray_state": item["spray_state"],
        "trajectory_index": int(step),
        "time_from_start_s": float(times[step]),
        "joint_names": list(base.JOINT_NAMES),
        "positions_rad": np.asarray(positions[step], dtype=float).tolist(),
        "velocities_rad_s": np.asarray(velocities[step], dtype=float).tolist(),
        "accelerations_rad_s2": np.asarray(accelerations[step], dtype=float).tolist(),
    } for step in range(len(positions))]


def native_ruckig_recheck(moveit: MoveItPy, primitive: Mapping[str, Any], item: Mapping[str, Any], positions: np.ndarray, limits: Mapping[str, Any], contract: Mapping[str, Any], totg: Mapping[str, Any]) -> dict[str, Any]:
    """Run native TOTG/Ruckig and then the frozen H10 native gates."""

    source_rows = [{"joint_values": np.asarray(q, dtype=float).tolist()} for q in positions]
    start_state = RobotState(moveit.get_robot_model())
    trajectory = RobotTrajectory(moveit.get_robot_model())
    trajectory.joint_model_group_name = GROUP_NAME
    trajectory.set_robot_trajectory_msg(start_state, base.make_trajectory_message(source_rows))
    before = np.asarray(positions, dtype=float)
    trajectory.unwind()
    try:
        totg_ok = bool(trajectory.apply_totg_time_parameterization(
            float(totg["velocity_scaling_factor"]), float(totg["acceleration_scaling_factor"]),
            path_tolerance=float(totg["path_tolerance"]), resample_dt=float(totg["resample_dt"]), min_angle_change=float(totg["min_angle_change"]),
        ))
    except Exception as exc:
        return {"status": "BLOCKED", "first_blocker": f"native_totg_exception:{type(exc).__name__}:{exc}", "totg_run": False, "ruckig_run": False, "post_ruckig_executed": False}
    if not totg_ok:
        return {"status": "BLOCKED", "first_blocker": "native_totg_returned_false", "totg_run": True, "ruckig_run": False, "post_ruckig_executed": False}
    try:
        ruckig_ok = bool(trajectory.apply_ruckig_smoothing(
            float(totg["velocity_scaling_factor"]), float(totg["acceleration_scaling_factor"]), mitigate_overshoot=True, overshoot_threshold=0.01,
        ))
    except Exception as exc:
        return {"status": "BLOCKED", "first_blocker": f"native_ruckig_exception:{type(exc).__name__}:{exc}", "totg_run": True, "ruckig_run": False, "post_ruckig_executed": True}
    if not ruckig_ok:
        return {"status": "BLOCKED", "first_blocker": "native_ruckig_returned_false", "totg_run": True, "ruckig_run": True, "post_ruckig_executed": True}
    try:
        final_t, final_q, final_dq, final_ddq = base.message_arrays(trajectory.get_robot_trajectory_msg())
    except Exception as exc:
        return {"status": "BLOCKED", "first_blocker": f"native_post_ruckig_output_invalid:{type(exc).__name__}:{exc}", "totg_run": True, "ruckig_run": True, "post_ruckig_executed": True}
    endpoint_delta = float(max(np.max(np.abs(final_q[0] - before[0])), np.max(np.abs(final_q[-1] - before[-1]))))
    rows = trajectory_rows(item, final_q, final_dq, final_ddq, final_t)
    check, process_rows = h10.native_recheck(moveit, primitive, rows, limits, contract)
    check.update({
        "totg_run": True,
        "ruckig_run": True,
        "post_ruckig_executed": True,
        "post_ruckig_output_waypoint_count": int(len(final_t)),
        "post_ruckig_endpoint_max_abs_delta_rad": endpoint_delta,
        "post_ruckig_endpoint_preserved": bool(endpoint_delta <= 1.0e-10),
        "post_ruckig_dynamics_phase": check.get("dynamic", {}).get("phase"),
        "process_rows_checked": len(process_rows),
        "ruckig_certification_semantics": "native MoveIt RuckigSmoothing; finite-difference jerk remains diagnostic only",
    })
    if endpoint_delta > 1.0e-10:
        check["status"] = "BLOCKED"
        check["first_blocker"] = "native_post_ruckig_endpoint_changed"
    return check


def failure_category(check: Mapping[str, Any]) -> list[str]:
    categories: list[str] = []
    if check.get("first_blocker", "").startswith("native_") and "exception" in str(check.get("first_blocker")):
        categories.append("NATIVE_RUNTIME_ERROR")
    execution = check.get("execution_limits", {})
    if int(execution.get("position_violation_count", 0) or 0): categories.append("JOINT_POSITION")
    if int(execution.get("velocity_violation_count", 0) or 0): categories.append("JOINT_VELOCITY")
    if int(execution.get("acceleration_violation_count", 0) or 0): categories.append("JOINT_ACCELERATION")
    if int(execution.get("jerk_violation_count", 0) or 0): categories.append("JOINT_JERK")
    collision = check.get("collision", {})
    if int(collision.get("self_collision_failure_count", 0) or 0): categories.append("SELF_COLLISION")
    if int(collision.get("environment_collision_failure_count", 0) or 0): categories.append("ENVIRONMENT_COLLISION")
    process = check.get("process", {})
    if int(process.get("failed_spray_on_sample_count", 0) or 0):
        first = process.get("first_failure") or {}
        if float(first.get("standoff_error_m", 0.0) or 0.0) > float(first.get("standoff_limit_m", float("inf"))): categories.append("STANDOFF")
        if float(first.get("normal_deviation_deg", 0.0) or 0.0) > float(first.get("normal_limit_deg", float("inf"))): categories.append("SURFACE_NORMAL")
        if float(first.get("tcp_position_error_m", 0.0) or 0.0) > float(first.get("tcp_position_limit_m", float("inf"))): categories.append("TCP_POSITION")
        if float(first.get("tcp_orientation_error_deg", 0.0) or 0.0) > float(first.get("tcp_orientation_limit_deg", float("inf"))): categories.append("TCP_ORIENTATION")
        categories.append("SPRAY_STATE_SEMANTICS")
    if not categories and check.get("status") != "PASSED": categories.append("OTHER_EXPLICITLY_DESCRIBED")
    return categories


def run(args: argparse.Namespace, moveit: MoveItPy) -> dict[str, Any]:
    bundle = np.load(Path(args.input_npz), allow_pickle=False)
    positions = np.asarray(bundle["positions_rad"], dtype=np.float64)
    velocities = np.asarray(bundle["velocities_rad_s"], dtype=np.float64)
    accelerations = np.asarray(bundle["accelerations_rad_s2"], dtype=np.float64)
    target_times = np.asarray(bundle["target_times_s"], dtype=np.float64)
    windows = load_json(Path(args.window_manifest))["windows"]
    total = len(windows)
    start, end = int(args.start_index), int(args.end_index)
    if start < 0 or end <= start or end > total or positions.shape != (total, 8, 6) or velocities.shape != positions.shape or accelerations.shape != positions.shape or target_times.shape != (total, 8):
        return {"schema_version": SCHEMA_VERSION, "status": "BLOCKED", "completed": False, "first_blocker": "native_input_shape_or_range_mismatch", "canonical_start_index": start, "canonical_end_index": end, "expected_window_count": max(0, end - start), "actual_window_count": 0, "NATIVE_RUNTIME_ERRORS": 1}
    validation = [json.loads(line) for line in Path(args.h6_validation).read_text(encoding="utf-8").splitlines() if line.strip()]
    segments = load_json(Path(args.h6_segments))
    contract = load_json(Path(args.process_contract))
    totg = load_json(Path(args.totg_parameters))
    post_ruckig_mode = str(args.post_ruckig_mode)
    primitives = h73.derive_primitives(validation, segments)
    primitive_map = {str(item["primitive_id"]): item for item in primitives}
    failures: list[dict[str, Any]] = []
    counts = {key: 0 for key in ("JOINT_POSITION", "JOINT_VELOCITY", "JOINT_ACCELERATION", "JOINT_JERK", "SELF_COLLISION", "ENVIRONMENT_COLLISION", "SPRAY_PROCESS_TOLERANCE", "NATIVE_RUNTIME_ERRORS")}
    checked_indices: list[int] = []
    start_time = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
    try:
        base.apply_fixture(moveit, Path(args.fixture_mesh))
        limits = base.runtime_limits(moveit, Path(args.runtime_limits))
        if limits.get("status") != "PASSED":
            return {"schema_version": SCHEMA_VERSION, "status": "BLOCKED", "completed": False, "first_blocker": "runtime_joint_limit_audit_failed", "canonical_start_index": start, "canonical_end_index": end, "expected_window_count": end - start, "actual_window_count": 0, "NATIVE_RUNTIME_ERRORS": 1, "runtime_limit_audit": limits}
        for index in range(start, end):
            item = windows[index]
            primitive = primitive_map.get(str(item["primitive_id"]))
            checked_indices.append(index)
            if primitive is None:
                check = {"status": "BLOCKED", "first_blocker": f"primitive_missing:{item['primitive_id']}", "native_runtime_error": True}
                categories = ["NATIVE_RUNTIME_ERROR"]
            else:
                times = target_times[index] - target_times[index, 0]
                pre_rows = trajectory_rows(item, positions[index], velocities[index], accelerations[index], times)
                precheck, _ = h10.native_recheck(moveit, primitive, pre_rows, limits["runtime_bounds"], contract)
                if post_ruckig_mode == "per_window":
                    check = native_ruckig_recheck(moveit, primitive, item, positions[index], limits["runtime_bounds"], contract, totg)
                else:
                    # Full-window mode still executes the complete native
                    # H10 FK/dynamics/process/FCL gates.  Ruckig is recorded
                    # as unvalidated for this window; the coordinator must
                    # keep H12-R BLOCKED until a safe per-window Ruckig
                    # backend exists.  This is deliberately not a PASS.
                    check = dict(precheck)
                    check.update({"post_ruckig_executed": False, "post_ruckig_status": "NOT_RUN_IN_SHARD_MODE", "post_ruckig_required": True})
                check["pre_repair_native_check"] = precheck
                categories = failure_category(check) if check.get("status") != "PASSED" else []
            if categories:
                for category in categories:
                    if category == "SPRAY_STATE_SEMANTICS": counts["SPRAY_PROCESS_TOLERANCE"] += 1
                    elif category in counts: counts[category] += 1
                failures.append({"schema_version": "stage3_h12_r_native_failure_v1", "canonical_window_index": index, "window_id": item.get("window_id"), "trajectory_family_id": item.get("trajectory_family_id"), "segment_id": item.get("segment_id"), "segment_order": item.get("segment_order"), "primitive_id": item.get("primitive_id"), "spray_state": item.get("spray_state"), "failure_categories": categories, "exact_failing_gate": check.get("first_blocker"), "check": check})
    except Exception as exc:
        failure = {"schema_version": "stage3_h12_r_native_failure_v1", "canonical_window_index": checked_indices[-1] if checked_indices else start, "failure_categories": ["NATIVE_RUNTIME_ERROR"], "exact_failing_gate": f"native_exception:{type(exc).__name__}:{exc}"}
        failures.append(failure)
        counts["NATIVE_RUNTIME_ERRORS"] += 1
        end_time = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
        atomic_jsonl(Path(args.failures_output), failures)
        return {"schema_version": SCHEMA_VERSION, "status": "BLOCKED", "completed": False, "first_blocker": failure["exact_failing_gate"], "canonical_start_index": start, "canonical_end_index": end, "expected_window_count": end - start, "actual_window_count": len(checked_indices), "checked_indices": checked_indices, "failure_count": len(failures), "failure_categories": counts, "NATIVE_RUNTIME_ERRORS": counts["NATIVE_RUNTIME_ERRORS"], "start_timestamp": start_time, "end_timestamp": end_time}
    end_time = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
    atomic_jsonl(Path(args.failures_output), failures)
    result = {"schema_version": SCHEMA_VERSION, "native_validator_version": NATIVE_VALIDATOR_VERSION, "status": "PASSED" if not failures else "BLOCKED", "completed": True, "first_blocker": None if not failures else "native_repair_constraint_violation:" + failures[0]["failure_categories"][0], "canonical_start_index": start, "canonical_end_index": end, "expected_window_count": end - start, "actual_window_count": len(checked_indices), "checked_indices": checked_indices, "failure_count": len(failures), "failure_categories": counts, "NATIVE_RUNTIME_ERRORS": counts["NATIVE_RUNTIME_ERRORS"], "native_backend_executed": True, "planning_scene_executed": True, "fk_executed": True, "dynamics_executed": True, "post_ruckig_executed": post_ruckig_mode == "per_window", "post_ruckig_mode": post_ruckig_mode, "POST_RUCKIG_UNVALIDATED_WINDOWS": 0 if post_ruckig_mode == "per_window" else len(checked_indices), "collision_method": base.COLLISION_METHOD, "CCD_AVAILABLE": "NO", "CLEARANCE_AVAILABLE": None, "PHYSICAL_ROBOT_CONNECTED": "NO", "PHYSICAL_FJT_GOALS_SENT": 0, "PHYSICAL_MOTION": 0, "start_timestamp": start_time, "end_timestamp": end_time}
    result["output_semantic_sha256"] = semantic_sha({key: value for key, value in result.items() if key != "output_semantic_sha256"})
    atomic_json(Path(args.output_json), result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("input_npz", "window_manifest", "output_json", "failures_output", "h6_validation", "h6_segments", "process_contract", "fixture_mesh", "runtime_limits", "totg_parameters", "post_ruckig_mode"):
        parser.add_argument(f"--{name.replace('_', '-')}", required=True)
    parser.add_argument("--start-index", required=True, type=int)
    parser.add_argument("--end-index", required=True, type=int)
    parser.add_argument("--shard-id", required=True)
    args, _ = parser.parse_known_args()
    rclpy.init()
    moveit = None
    try:
        moveit = MoveItPy(node_name=f"stage3_h12_r_native_{args.shard_id}")
        result = run(args, moveit)
    except Exception as exc:
        result = {"schema_version": SCHEMA_VERSION, "status": "BLOCKED", "completed": False, "first_blocker": f"native_exception:{type(exc).__name__}:{exc}", "canonical_start_index": args.start_index, "canonical_end_index": args.end_index, "expected_window_count": args.end_index - args.start_index, "actual_window_count": 0, "NATIVE_RUNTIME_ERRORS": 1}
        atomic_json(Path(args.output_json), result)
    finally:
        if moveit is not None:
            moveit.shutdown()
        rclpy.shutdown()
    if not Path(args.output_json).is_file():
        atomic_json(Path(args.output_json), result)
    return 0 if result.get("completed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
