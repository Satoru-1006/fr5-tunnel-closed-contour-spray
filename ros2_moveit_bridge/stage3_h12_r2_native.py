#!/usr/bin/env python3
"""Native primitive-level H12-R2 certification worker.

The unit of execution is an authoritative trajectory primitive/segment, not an
isolated prediction window.  Every completed unit runs MoveIt TOTG, native
Ruckig, FK/process checks against the frozen H6.4 primitive, and PlanningScene
collision checks on the returned post-Ruckig trajectory.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import rclpy
from moveit.core.robot_state import RobotState
from moveit.core.robot_trajectory import RobotTrajectory
from moveit.planning import MoveItPy

import stage3_h10_native as h10
import stage3_h7_2_native as base
import stage3_h7_3_native as h73
import stage3_h7_6_native as h76
import stage3_h12_r_native as h12r
from src.stage3_h12_r3 import classify_ruckig_result


SCHEMA_VERSION = "stage3_h12_r2_native_primitive_v1"
EXECUTION_UNIT = "trajectory_primitive"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def json_safe(value: Any) -> Any:
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


def _native_result(wrapper: Mapping[str, Any] | None) -> dict[str, Any]:
    if not wrapper:
        return {"status": "NOT_REPORTED", "RUCKIG_RESULT": None, "RUCKIG_ERROR_INVALID_INPUT": None, "RUCKIG_OTHER_ERRORS": None}
    result = wrapper.get("final_ruckig_result") or {}
    numeric = result.get("numeric_result")
    result_name = result.get("result_name")
    invalid = bool(numeric == -100 or result_name == "ErrorInvalidInput")
    other = bool(numeric not in (0, 1, -100) if numeric is not None else False)
    return {
        "status": "REPORTED",
        "RUCKIG_RESULT": result,
        "classification": classify_ruckig_result(result),
        "RUCKIG_ERROR_INVALID_INPUT": invalid,
        "RUCKIG_OTHER_ERRORS": other,
        "wrapper_returned_bool": bool(wrapper.get("wrapper_returned_bool")),
        "smoothing_complete": bool(wrapper.get("smoothing_complete")),
        "duration_ceiling_hit": bool(wrapper.get("duration_ceiling_hit")),
    }


def _pre_rows(item: Mapping[str, Any], positions: np.ndarray, times: np.ndarray) -> list[dict[str, Any]]:
    velocity = np.gradient(positions, times, axis=0, edge_order=1)
    acceleration = np.gradient(velocity, times, axis=0, edge_order=1)
    return h12r.trajectory_rows(item, positions, velocity, acceleration, times)


def execute_unit(
    moveit: MoveItPy,
    primitive: Mapping[str, Any],
    item: Mapping[str, Any],
    positions: np.ndarray,
    times: np.ndarray,
    limits: Mapping[str, Any],
    contract: Mapping[str, Any],
    totg: Mapping[str, Any],
    output_dir: Path,
) -> tuple[dict[str, Any], np.ndarray | None, np.ndarray | None]:
    """Run one complete unit and return its post-Ruckig arrays when available."""

    if positions.ndim != 2 or positions.shape[1] != 6 or len(positions) < 2:
        raise ValueError("invalid_primitive_position_shape")
    if times.shape != (len(positions),) or not np.all(np.isfinite(positions)) or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
        raise ValueError("invalid_primitive_time_or_nonfinite_input")
    precheck, _ = h10.native_recheck(moveit, primitive, _pre_rows(item, positions, times), limits, contract)
    start_state = RobotState(moveit.get_robot_model())
    trajectory = RobotTrajectory(moveit.get_robot_model())
    trajectory.joint_model_group_name = base.GROUP_NAME
    trajectory.set_robot_trajectory_msg(start_state, base.make_trajectory_message([{"joint_values": row.tolist()} for row in positions]))
    before = positions.copy()
    trajectory.unwind()
    totg_ok = bool(trajectory.apply_totg_time_parameterization(
        float(totg["velocity_scaling_factor"]),
        float(totg["acceleration_scaling_factor"]),
        path_tolerance=float(totg["path_tolerance"]),
        resample_dt=float(totg["resample_dt"]),
        min_angle_change=float(totg["min_angle_change"]),
    ))
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "execution_unit": EXECUTION_UNIT,
        "unit_id": item["unit_id"],
        "trajectory_family_id": item["trajectory_family_id"],
        "segment_key": item["segment_key"],
        "segment_id": int(item["segment_id"]),
        "segment_order": int(item["segment_order"]),
        "primitive_id": item["primitive_id"],
        "spray_state": item["spray_state"],
        "window_ids": item["window_ids"],
        "window_count": int(item["window_count"]),
        "pre_ruckig_executed": True,
        "pre_ruckig_status": precheck.get("status"),
        "pre_ruckig_check": precheck,
        "post_ruckig_executed": False,
        "post_ruckig_status": "NOT_RUN",
        "native_runtime_error": False,
        "collision_method": base.COLLISION_METHOD,
        "CCD": "not_available",
        "clearance": None,
    }
    if not totg_ok:
        result.update(status="BLOCKED", first_blocker="native_totg_returned_false", post_ruckig_status="NOT_RUN")
        return result, None, None
    try:
        ruckig_bool = bool(trajectory.apply_ruckig_smoothing(
            float(totg["velocity_scaling_factor"]),
            float(totg["acceleration_scaling_factor"]),
            mitigate_overshoot=True,
            overshoot_threshold=0.01,
        ))
    except Exception as exc:
        result.update(status="BLOCKED", first_blocker=f"native_ruckig_exception:{type(exc).__name__}:{exc}", post_ruckig_status="ERROR", post_ruckig_executed=False, native_runtime_error=True)
        return result, None, None
    wrapper = h76.latest_wrapper_summary()
    ruckig_native = _native_result(wrapper)
    result.update(ruckig_returned_bool=ruckig_bool, ruckig_native=ruckig_native, post_ruckig_executed=True, post_ruckig_status="EXECUTED")
    if not ruckig_bool:
        result.update(status="BLOCKED", first_blocker="native_ruckig_returned_false")
        return result, None, None
    if ruckig_native["status"] != "REPORTED":
        result.update(status="BLOCKED", first_blocker="ruckig_result_not_reported", post_ruckig_status="BLOCKED")
        return result, None, None
    # MoveIt's wrapper boolean is not a completion certificate.  In
    # particular, native Ruckig returns Working (numeric 0) while extending
    # the trajectory duration.  Only Finished is allowed to enter formal
    # post-Ruckig certification; Working remains explicitly unvalidated.
    if ruckig_native["classification"] != "Finished":
        result.update(
            status="BLOCKED",
            first_blocker=("ruckig_working_incomplete" if ruckig_native["classification"] == "Working_Incomplete" else "ruckig_native_error"),
            post_ruckig_status="UNVALIDATED",
            post_ruckig_executed=True,
            post_ruckig_certification_allowed=False,
        )
        return result, None, None
    result["post_ruckig_certification_allowed"] = True
    try:
        final_t, final_q, final_dq, final_ddq = base.message_arrays(trajectory.get_robot_trajectory_msg())
    except Exception as exc:
        result.update(status="BLOCKED", first_blocker=f"post_ruckig_output_invalid:{type(exc).__name__}:{exc}")
        return result, None, None
    endpoint_delta = float(max(np.max(np.abs(final_q[0] - before[0])), np.max(np.abs(final_q[-1] - before[-1]))))
    post_rows = h12r.trajectory_rows(item, final_q, final_dq, final_ddq, final_t)
    post_check, _ = h10.native_recheck(moveit, primitive, post_rows, limits, contract)
    post_check.update({"post_ruckig_executed": True, "ruckig_native": ruckig_native})
    result.update({
        "post_ruckig_status": "PASSED" if post_check.get("status") == "PASSED" and endpoint_delta <= 1.0e-10 else "FAILED",
        "post_ruckig_check": post_check,
        "post_ruckig_output_waypoint_count": int(len(final_t)),
        "post_ruckig_duration_s": float(final_t[-1]),
        "post_ruckig_endpoint_max_abs_delta_rad": endpoint_delta,
        "post_ruckig_endpoint_preserved": bool(endpoint_delta <= 1.0e-10),
        "status": "PASSED" if post_check.get("status") == "PASSED" and endpoint_delta <= 1.0e-10 else "BLOCKED",
        "first_blocker": None if post_check.get("status") == "PASSED" and endpoint_delta <= 1.0e-10 else ("native_post_ruckig_endpoint_changed" if endpoint_delta > 1.0e-10 else post_check.get("first_blocker") or "post_ruckig_validation_failed"),
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / f"{int(item['unit_index']):05d}.npz", time_s=final_t, positions_rad=final_q, velocities_rad_s=final_dq, accelerations_rad_s2=final_ddq)
    result["post_ruckig_arrays_file"] = f"post_ruckig_units/{int(item['unit_index']):05d}.npz"
    return result, final_t, final_q


def run(args: argparse.Namespace, moveit: MoveItPy) -> int:
    bundle = np.load(Path(args.input_npz), allow_pickle=False)
    positions = np.asarray(bundle["positions_rad"], dtype=np.float64)
    times = np.asarray(bundle["times_s"], dtype=np.float64)
    lengths = np.asarray(bundle["lengths"], dtype=np.int64)
    units = load_jsonl(Path(args.unit_manifest))
    validation = load_jsonl(Path(args.h6_validation))
    segments = load_json(Path(args.h6_segments))
    contract = load_json(Path(args.process_contract))
    totg = load_json(Path(args.totg_parameters))
    primitive_map = {str(item["primitive_id"]): item for item in h73.derive_primitives(validation, segments)}
    base.apply_fixture(moveit, Path(args.fixture_mesh))
    limit_audit = base.runtime_limits(moveit, Path(args.runtime_limits))
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    output_jsonl = Path(args.output_jsonl)
    failure_jsonl = Path(args.failure_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.write_text("", encoding="utf-8", newline="\n")
    failure_jsonl.write_text("", encoding="utf-8", newline="\n")
    result_stream = output_jsonl.open("a", encoding="utf-8", newline="\n")
    failure_stream = failure_jsonl.open("a", encoding="utf-8", newline="\n")
    if limit_audit.get("status") != "PASSED":
        raise RuntimeError("runtime_joint_limit_audit_failed")
    for unit in units:
        idx = int(unit["unit_index"])
        item = dict(unit)
        item["unit_id"] = str(unit["unit_id"])
        primitive = primitive_map.get(str(unit["primitive_id"]))
        if primitive is None:
            result = {**unit, "schema_version": SCHEMA_VERSION, "status": "BLOCKED", "first_blocker": "primitive_missing", "post_ruckig_executed": False, "native_runtime_error": True}
        else:
            try:
                result, _, _ = execute_unit(moveit, primitive, item, positions[idx, : int(lengths[idx])], times[idx, : int(lengths[idx])], limit_audit["runtime_bounds"], contract, totg, Path(args.output_dir) / "post_ruckig_units")
            except Exception as exc:
                result = {**unit, "schema_version": SCHEMA_VERSION, "status": "BLOCKED", "first_blocker": f"native_exception:{type(exc).__name__}:{exc}", "post_ruckig_executed": False, "native_runtime_error": True}
        result = json_safe(result)
        results.append(result)
        result_stream.write(json.dumps(result, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        result_stream.flush()
        if result.get("status") != "PASSED":
            check = result.get("post_ruckig_check") or result.get("pre_ruckig_check") or {}
            categories = h12r.failure_category(check)
            if result.get("native_runtime_error"):
                categories = ["NATIVE_RUNTIME_ERROR"]
            failures.append({"schema_version": "stage3_h12_r2_native_failure_v1", "unit_id": unit.get("unit_id"), "execution_unit": EXECUTION_UNIT, "window_ids": unit.get("window_ids", []), "trajectory_family_id": unit.get("trajectory_family_id"), "segment_id": unit.get("segment_id"), "primitive_id": unit.get("primitive_id"), "spray_state": unit.get("spray_state"), "raw_failure_categories": categories, "exact_failing_gate": result.get("first_blocker"), "post_ruckig_executed": result.get("post_ruckig_executed"), "check": check})
            failure_stream.write(json.dumps(failures[-1], ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
            failure_stream.flush()
    result_stream.close()
    failure_stream.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("input_npz", "unit_manifest", "output_jsonl", "failure_jsonl", "output_dir", "h6_validation", "h6_segments", "process_contract", "fixture_mesh", "runtime_limits", "totg_parameters"):
        parser.add_argument(f"--{name.replace('_', '-')}", required=True)
    args, _ = parser.parse_known_args()
    rclpy.init()
    moveit = None
    try:
        moveit = MoveItPy(node_name="stage3_h12_r2_native")
        return run(args, moveit)
    except Exception as exc:
        error_result = json.dumps({"schema_version": SCHEMA_VERSION, "status": "BLOCKED", "first_blocker": f"native_exception:{type(exc).__name__}:{exc}", "post_ruckig_executed": False, "native_runtime_error": True}, sort_keys=True) + "\n"
        error_failure = json.dumps({"schema_version": "stage3_h12_r2_native_failure_v1", "execution_unit": EXECUTION_UNIT, "raw_failure_categories": ["NATIVE_RUNTIME_ERROR"], "exact_failing_gate": f"native_exception:{type(exc).__name__}:{exc}"}, sort_keys=True) + "\n"
        result_path = Path(args.output_jsonl)
        failure_path = Path(args.failure_jsonl)
        if not result_path.is_file() or not result_path.read_text(encoding="utf-8").strip():
            result_path.write_text(error_result, encoding="utf-8", newline="\n")
        else:
            with result_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(error_result)
        with failure_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(error_failure)
        return 2
    finally:
        if moveit is not None:
            moveit.shutdown()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
