"""Pure H12-R4 diagnostics and fail-closed acceptance rules.

These helpers intentionally keep geometric path validity, diagnostics under
the historical timestamps, native TOTG, and post-Ruckig certification as
separate concepts.  They have no ROS dependency so the semantic contract can
be regression-tested on Windows as well as in the native WSL environment.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

MACHINE_EPSILON = np.finfo(np.float64).eps
NUMERIC_TOLERANCE_MULTIPLIER = 64.0
MOVEIT_TURN_COSINE_TOLERANCE = 1.0e-5
TOTAL_PRIMITIVES = 480
TOTAL_WINDOWS = 189216


def numeric_tolerance(raw_value: float, limit: float, multiplier: float = NUMERIC_TOLERANCE_MULTIPLIER) -> float:
    """Bounded diagnostic-only tolerance; never changes a native limit."""

    scale = max(1.0, abs(float(raw_value)), abs(float(limit)))
    return max(1.0e-12, float(multiplier) * MACHINE_EPSILON * scale)


def diagnostic_limit_comparison(raw_value: float, limit: float) -> dict[str, Any]:
    raw = float(raw_value)
    bound = abs(float(limit))
    excess = abs(raw) - bound
    tolerance = numeric_tolerance(raw, bound)
    return {
        "raw_value": raw,
        "limit": bound,
        "raw_excess": excess,
        "numeric_tolerance": tolerance,
        "meaningful_excess": max(0.0, excess - tolerance),
        "raw_comparison_exceeds": bool(excess > 0.0),
        "meaningful_violation": bool(excess > tolerance),
        "diagnostic_only": True,
    }


def finite_difference_diagnostics_only(times: Sequence[float], positions: Sequence[Sequence[float]], limits: Mapping[str, Sequence[float]] | None = None) -> dict[str, Any]:
    t = np.asarray(times, dtype=np.float64)
    q = np.asarray(positions, dtype=np.float64)
    result: dict[str, Any] = {
        "authoritative_for_totg_admission": False,
        "purpose": "provenance_and_pre_retiming_diagnostics_only",
        "timestamp_count": int(t.size),
        "position_shape": list(q.shape),
        "finite": bool(np.all(np.isfinite(t)) and np.all(np.isfinite(q))),
        "strictly_increasing": bool(t.ndim == 1 and t.size == len(q) and np.all(np.diff(t) > 0.0)),
    }
    if q.ndim != 2 or t.shape != (len(q),) or len(q) < 2 or not result["finite"] or not result["strictly_increasing"]:
        result.update({"status": "INVALID_DIAGNOSTIC_INPUT", "max_abs_velocity": None, "max_abs_acceleration": None, "comparisons": []})
        return result
    edge_order = 2 if len(q) >= 3 else 1
    velocity = np.gradient(q, t, axis=0, edge_order=edge_order)
    acceleration = np.gradient(velocity, t, axis=0, edge_order=edge_order)
    result.update({
        "status": "RECORDED",
        "minimum_dt_s": float(np.min(np.diff(t))),
        "max_abs_velocity": np.max(np.abs(velocity), axis=0).tolist(),
        "max_abs_acceleration": np.max(np.abs(acceleration), axis=0).tolist(),
        "comparisons": [],
    })
    if limits:
        for quantity, values in (("velocity", result["max_abs_velocity"]), ("acceleration", result["max_abs_acceleration"])):
            bounds = limits.get(quantity)
            if bounds is None:
                continue
            for joint_index, (value, bound) in enumerate(zip(values, bounds)):
                result["comparisons"].append({"quantity": quantity, "joint_index": joint_index, **diagnostic_limit_comparison(value, bound)})
    return result


def analyze_geometric_path(positions: Sequence[Sequence[float]], *, min_angle_change: float = 5.0e-4, near_duplicate_tolerance: float = 1.0e-6) -> dict[str, Any]:
    q = np.asarray(positions, dtype=np.float64)
    base: dict[str, Any] = {
        "input_position_finite": bool(np.all(np.isfinite(q))),
        "waypoint_count": int(len(q)) if q.ndim else 0,
        "input_duplicate_waypoint_count": 0,
        "input_near_duplicate_waypoint_count": 0,
        "max_joint_delta": None,
        "min_joint_delta": None,
        "near_180_turn_count": 0,
        "maximum_turn_angle_deg": None,
        "first_suspect_turn_waypoint": None,
        "maximum_turn": None,
        "moveit_turn_cosine_threshold": -1.0 + MOVEIT_TURN_COSINE_TOLERANCE,
        "moveit_min_angle_change": float(min_angle_change),
    }
    if q.ndim != 2 or q.shape[1:] != (6,) or len(q) < 2 or not base["input_position_finite"]:
        base["geometric_shape_valid"] = False
        return base
    delta = np.diff(q, axis=0)
    norms = np.linalg.norm(delta, axis=1)
    exact = norms == 0.0
    near = (norms > 0.0) & (norms <= near_duplicate_tolerance)
    base.update({
        "geometric_shape_valid": True,
        "input_duplicate_waypoint_count": int(np.count_nonzero(exact)),
        "input_near_duplicate_waypoint_count": int(np.count_nonzero(near)),
        "max_joint_delta": float(np.max(norms)),
        "min_joint_delta": float(np.min(norms)),
        "moveit_filtered_segment_count": int(np.count_nonzero(np.max(np.abs(delta), axis=1) <= min_angle_change)),
    })
    maximum: dict[str, Any] | None = None
    suspects: list[dict[str, Any]] = []
    for index in range(len(delta) - 1):
        left, right = delta[index], delta[index + 1]
        left_norm, right_norm = float(norms[index]), float(norms[index + 1])
        if left_norm <= MACHINE_EPSILON or right_norm <= MACHINE_EPSILON:
            continue
        cosine = float(np.clip(np.dot(left, right) / (left_norm * right_norm), -1.0, 1.0))
        angle = float(math.degrees(math.acos(cosine)))
        record = {
            "turn_cosine": cosine,
            "turn_angle_deg": angle,
            "waypoint_triplet": [index, index + 1, index + 2],
            "joint_space_vectors": [left.tolist(), right.tolist()],
        }
        if maximum is None or angle > maximum["turn_angle_deg"]:
            maximum = record
        if cosine <= -1.0 + MOVEIT_TURN_COSINE_TOLERANCE:
            suspects.append(record)
    base.update({
        "near_180_turn_count": len(suspects),
        "maximum_turn_angle_deg": None if maximum is None else maximum["turn_angle_deg"],
        "first_suspect_turn_waypoint": None if not suspects else suspects[0]["waypoint_triplet"][1],
        "maximum_turn": maximum,
        "near_180_turns": suspects,
    })
    return base


def joint_position_gate(positions: Sequence[Sequence[float]], lower: Sequence[float], upper: Sequence[float]) -> dict[str, Any]:
    q = np.asarray(positions, dtype=np.float64)
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    invalid = (~np.isfinite(q)) | (q < lo) | (q > hi)
    locations = np.argwhere(invalid)
    return {
        "input_joint_limit_valid": bool(locations.size == 0),
        "position_violation_count": int(len(locations)),
        "first_position_violation": None if not len(locations) else {"waypoint_index": int(locations[0, 0]), "joint_index": int(locations[0, 1]), "value": float(q[tuple(locations[0])]), "lower": float(lo[locations[0, 1]]), "upper": float(hi[locations[0, 1]])},
    }


def classify_totg_false(geometry: Mapping[str, Any], *, group_valid: bool, velocity_limits_valid: bool, acceleration_limits_valid: bool, path_tolerance: float, exception: str | None = None) -> tuple[str, str]:
    """Classify only exact MoveIt 2.12.4 false-return decision branches."""

    if exception:
        return "native_exception", exception
    if not geometry.get("input_position_finite", False):
        return "nonfinite_input", "input joint positions contain NaN or infinity"
    if int(geometry.get("waypoint_count", 0)) < 2:
        return "degenerate_waypoint_sequence", "MoveIt Path::create requires at least two waypoints"
    if not group_valid:
        return "invalid_robot_model_or_group", "RobotTrajectory joint model group is unavailable"
    if not velocity_limits_valid:
        return "invalid_velocity_limit", "native robot model velocity limits are missing or invalid"
    if not acceleration_limits_valid:
        return "invalid_acceleration_limit", "native robot model acceleration limits are missing or invalid"
    if path_tolerance <= 0.0:
        return "path_tolerance_failure", "TOTG path_tolerance must be greater than zero"
    if int(geometry.get("near_180_turn_count", 0)):
        return "unsupported_near_180_degree_turn", "path satisfies MoveIt 2.12.4 cosine <= -1 + 1e-5 rejection condition"
    if geometry.get("geometric_shape_valid"):
        return "trajectory_creation_failure", "valid Path::create preconditions and no unsupported 180-degree turn; remaining MoveIt 2.12.4 false branch is Trajectory::create"
    return "unknown_native_totg_false", "native binding returned false without sufficient evidence for a narrower class"


def ruckig_completion_gate(record: Mapping[str, Any]) -> dict[str, Any]:
    raw = record.get("ruckig_result") or record.get("final_ruckig_result") or {}
    numeric = raw.get("numeric_result")
    name = raw.get("result_name")
    raw_acceptable = bool(numeric in (0, 1) or name in {"Working", "Finished"})
    error = bool((numeric is not None and numeric < 0) or (isinstance(name, str) and name.startswith("Error")))
    gates = {
        "native_smoothing_returned_success": record.get("native_smoothing_returned_success") is True,
        "raw_result_acceptable": raw_acceptable,
        "last_segment_completed": record.get("last_segment_completed") is True,
        "smoothing_complete": record.get("smoothing_complete") is True,
        "overshoot_gate_passed": record.get("overshoot_gate_passed") is True,
        "duration_ceiling_clear": record.get("duration_ceiling_hit") is False,
        "iteration_limit_clear": record.get("iteration_limit_hit") is False,
        "wall_clock_timeout_clear": record.get("wall_clock_timeout") is False,
        "input_validation_passed": record.get("input_validation_passed") is True,
        "no_native_error": not error and record.get("native_error") in (None, False, ""),
        "output_finite": record.get("output_finite") is True,
    }
    accepted = all(gates.values())
    failed = [key for key, passed in gates.items() if not passed]
    return {"accepted": accepted, "failed_gates": failed, "gates": gates, "raw_result": raw, "raw_working": bool(numeric == 0 or name == "Working"), "raw_finished": bool(numeric == 1 or name == "Finished")}


def process_semantics_gate(spray_mode: str, process_check: Mapping[str, Any], common_checks_passed: bool = True) -> bool:
    if not common_checks_passed:
        return False
    if spray_mode == "SPRAY_ON":
        return process_check.get("status") == "PASSED" and int(process_check.get("failed_spray_on_sample_count", 0)) == 0
    return True


def coverage_audit(units: Iterable[Mapping[str, Any]], total_windows: int = TOTAL_WINDOWS, expected_primitives: int = TOTAL_PRIMITIVES) -> dict[str, Any]:
    rows = list(units)
    unit_ids = [str(row.get("unit_id") or row.get("segment_key")) for row in rows]
    duplicate_primitives = sum(value - 1 for value in Counter(unit_ids).values() if value > 1)
    window_ids: list[int] = []
    for row in rows:
        window_ids.extend(
            int(value)
            for value in (
                row.get("window_ids")
                or row.get("ordered_window_ids")
                or row.get("source_window_ids")
                or []
            )
        )
    counts = Counter(window_ids)
    missing = [index for index in range(total_windows) if counts[index] == 0]
    duplicate_windows = sum(value - 1 for key, value in counts.items() if 0 <= key < total_windows and value > 1)
    out_of_range = sum(value for key, value in counts.items() if key < 0 or key >= total_windows)
    return {
        "PRIMITIVES_EXPECTED": expected_primitives,
        "PRIMITIVES_OBSERVED": len(rows),
        "MISSING_PRIMITIVES": max(0, expected_primitives - len(set(unit_ids))),
        "DUPLICATE_PRIMITIVES": duplicate_primitives,
        "TOTAL_WINDOWS": total_windows,
        "MISSING_WINDOWS": len(missing),
        "DUPLICATE_WINDOWS": duplicate_windows,
        "OUT_OF_RANGE_WINDOWS": out_of_range,
        "coverage_complete": len(rows) == expected_primitives and not duplicate_primitives and not missing and not duplicate_windows and not out_of_range,
    }


def semantic_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def replay_hashes_equal(hashes: Sequence[str], required: int = 3) -> bool:
    return len(hashes) == required and all(hashes) and len(set(hashes)) == 1
