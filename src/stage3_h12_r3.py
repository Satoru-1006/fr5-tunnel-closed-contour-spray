"""Stage 3 H12-R3 primitive reconstruction and fail-closed gates.

This module is intentionally backend-neutral.  It contains the deterministic
parts of H12-R3 which can be audited without a ROS process: overlap
reconciliation, causal finite-difference diagnostics, native-result
classification, Ruckig input validation, and network/ceiling classification.
The native MoveIt2 worker remains the only authority for TOTG, FK, PlanningScene
and post-Ruckig certification.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


JOINTS = 6
HORIZON = 8
HISTORY = 16
POSITION_LOWER = np.asarray([-3.0543, -4.6251, -2.8274, -4.6251, -3.0543, -3.0543], dtype=np.float64)
POSITION_UPPER = np.asarray([3.0543, 1.4835, 2.8274, 1.4835, 3.0543, 3.0543], dtype=np.float64)
VELOCITY_LIMIT = np.asarray([0.4725, 0.4725, 0.4725, 0.48, 0.48, 0.48], dtype=np.float64)
ACCELERATION_LIMIT = np.asarray([0.105] * JOINTS, dtype=np.float64)
JERK_LIMIT = np.asarray([8.0] * JOINTS, dtype=np.float64)

RUCKIG_RESULT_CLASSES = {
    0: "Working_Incomplete",
    1: "Finished",
    -100: "ErrorInvalidInput",
    -101: "ErrorTrajectoryDuration",
    -102: "ErrorPositionalLimits",
    -110: "ErrorExecutionTimeCalculation",
    -111: "ErrorSynchronizationCalculation",
}


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _as_float_array(value: Sequence[float] | np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"invalid_{name}")
    return array


def classify_ruckig_result(result: Mapping[str, Any] | None) -> str:
    """Classify the native result without conflating Working and Finished."""

    result = result or {}
    raw_name = str(result.get("result_name") or result.get("name") or "")
    numeric = result.get("numeric_result", result.get("numeric_code"))
    try:
        numeric_int = int(numeric) if numeric is not None else None
    except (TypeError, ValueError):
        numeric_int = None
    if raw_name == "Finished" or numeric_int == 1:
        return "Finished"
    if raw_name == "Working" or numeric_int == 0:
        return "Working_Incomplete"
    if numeric_int in RUCKIG_RESULT_CLASSES and numeric_int < 0:
        return RUCKIG_RESULT_CLASSES[numeric_int]
    if raw_name in {value for key, value in RUCKIG_RESULT_CLASSES.items() if key < 0}:
        return raw_name
    return "OtherNativeError"


def ruckig_completion_gate(result: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return the formal gate used by H12-R3 post-certification."""

    result = dict(result or {})
    classification = classify_ruckig_result(result)
    complete = classification == "Finished"
    return {
        "classification": classification,
        "finished_required_for_pass": True,
        "smoothing_complete": bool(result.get("smoothing_complete")),
        "duration_ceiling_hit": bool(result.get("duration_ceiling_hit")),
        "post_certification_allowed": complete,
        "pass_candidate": bool(complete and result.get("smoothing_complete") is True and not result.get("duration_ceiling_hit", False)),
    }


def classify_ceiling(evidence: Mapping[str, Any]) -> str:
    """Classify a stopping condition; wall-clock and simulated duration differ."""

    if bool(evidence.get("wall_clock_timeout")) or evidence.get("timeout_kind") == "wall_clock":
        return "WALL_CLOCK_TIMEOUT"
    if bool(evidence.get("max_update_iterations_hit")):
        return "MAX_UPDATE_ITERATIONS"
    if bool(evidence.get("max_simulated_time_hit")):
        return "MAX_SIMULATED_TIME"
    if bool(evidence.get("max_calculation_duration_hit")):
        return "MAX_CALCULATION_DURATION"
    if bool(evidence.get("custom_h12_r2_ceiling_hit")):
        return "CUSTOM_H12_R2_CEILING"
    if bool(evidence.get("duration_ceiling_hit")) or bool(evidence.get("maximum_duration_no_solution")):
        return "TRAJECTORY_DURATION_LIMIT"
    return "NOT_REPORTED"


def _validate_state(value: Any, name: str) -> tuple[np.ndarray | None, str | None]:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None, "INVALID_NUMERICAL_RANGE"
    if array.shape != (JOINTS,) or not np.all(np.isfinite(array)):
        return None, "INVALID_NUMERICAL_RANGE"
    return array, None


def validate_ruckig_input(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one native Ruckig input before execution.

    The result is deliberately conservative: missing state fields are invalid
    rather than silently converted to zeros.  It mirrors the native
    ``validate_input`` categories required by H12-R3.
    """

    fields = ("current_position", "current_velocity", "current_acceleration", "target_position", "target_velocity", "target_acceleration", "max_velocity", "max_acceleration", "max_jerk")
    arrays: dict[str, np.ndarray] = {}
    for field in fields:
        array, error = _validate_state(record.get(field), field)
        if error:
            return {"status": error, "first_invalid_field": field, "native_validate_input": False}
        assert array is not None
        arrays[field] = array

    for field in ("min_position", "max_position"):
        if field in record and record[field] is not None:
            array, error = _validate_state(record[field], field)
            if error:
                return {"status": error, "first_invalid_field": field, "native_validate_input": False}
            assert array is not None
            arrays[field] = array

    if np.any(arrays["max_velocity"] <= 0) or np.any(arrays["max_acceleration"] <= 0) or np.any(arrays["max_jerk"] <= 0):
        return {"status": "INVALID_LIMIT_RELATION", "first_invalid_field": "limits", "native_validate_input": False}
    if "min_position" in arrays and "max_position" in arrays and np.any(arrays["min_position"] > arrays["max_position"]):
        return {"status": "INVALID_LIMIT_RELATION", "first_invalid_field": "position_limits", "native_validate_input": False}
    if "min_position" in arrays and np.any(arrays["current_position"] < arrays["min_position"]):
        return {"status": "INVALID_CURRENT_STATE", "first_invalid_field": "current_position", "native_validate_input": False}
    if "max_position" in arrays and np.any(arrays["current_position"] > arrays["max_position"]):
        return {"status": "INVALID_CURRENT_STATE", "first_invalid_field": "current_position", "native_validate_input": False}
    if "min_position" in arrays and np.any(arrays["target_position"] < arrays["min_position"]):
        return {"status": "INVALID_TARGET_STATE", "first_invalid_field": "target_position", "native_validate_input": False}
    if "max_position" in arrays and np.any(arrays["target_position"] > arrays["max_position"]):
        return {"status": "INVALID_TARGET_STATE", "first_invalid_field": "target_position", "native_validate_input": False}
    if np.any(np.abs(arrays["current_velocity"]) > arrays["max_velocity"] + 1e-12) or np.any(np.abs(arrays["current_acceleration"]) > arrays["max_acceleration"] + 1e-12):
        return {"status": "INVALID_CURRENT_STATE", "first_invalid_field": "current_velocity_or_acceleration", "native_validate_input": False}
    if np.any(np.abs(arrays["target_velocity"]) > arrays["max_velocity"] + 1e-12) or np.any(np.abs(arrays["target_acceleration"]) > arrays["max_acceleration"] + 1e-12):
        return {"status": "INVALID_TARGET_STATE", "first_invalid_field": "target_velocity_or_acceleration", "native_validate_input": False}
    return {"status": "VALID_INPUT", "first_invalid_field": None, "native_validate_input": True}


def reconcile_overlaps(
    predictions: Sequence[Sequence[Sequence[float]]] | np.ndarray,
    starts: Sequence[int],
    *,
    method: str = "center_weighted_consensus",
    confidences: Sequence[Sequence[float]] | np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reconcile overlapping forecasting windows deterministically.

    ``starts`` are local primitive offsets in the forecast domain, where a
    window's first prediction is at ``HISTORY + start``.  No target labels are
    accepted by this function.  The returned array contains only the forecast
    domain; callers add their causal history explicitly.
    """

    values = np.asarray(predictions, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (HORIZON, JOINTS) or not np.all(np.isfinite(values)):
        raise ValueError("invalid_prediction_shape")
    if len(starts) != len(values) or any(int(start) < 0 for start in starts):
        raise ValueError("invalid_window_starts")
    if method not in {"center_weighted_consensus", "confidence_weighted_average", "causal_priority", "overlap_median"}:
        raise ValueError("unknown_reconciliation_method")
    confidence_array = None
    if confidences is not None:
        confidence_array = np.asarray(confidences, dtype=np.float64)
        if confidence_array.shape != (len(values), HORIZON) or not np.all(np.isfinite(confidence_array)):
            raise ValueError("invalid_confidence_shape")
        confidence_array = np.maximum(confidence_array, 0.0)
    last_target = max(int(start) + HORIZON for start in starts)
    output = np.empty((last_target, JOINTS), dtype=np.float64)
    overlap_counts = np.zeros(last_target, dtype=np.int64)
    # Build the sparse overlap index once.  Scanning every window for every
    # target is quadratic for the 1,395-sample primitives and needlessly turns
    # a deterministic audit into a multi-minute Python loop.
    contributors_by_target: list[list[tuple[int, int]]] = [[] for _ in range(last_target)]
    for window_index, start in enumerate(starts):
        for horizon in range(HORIZON):
            target = int(start) + horizon
            if target < last_target:
                contributors_by_target[target].append((window_index, horizon))
    for target in range(last_target):
        contributors = contributors_by_target[target]
        if not contributors:
            raise RuntimeError(f"uncovered_reconstruction_target:{target}")
        overlap_counts[target] = len(contributors)
        if method == "causal_priority":
            window_index, horizon = min(contributors, key=lambda item: (int(starts[item[0]]), item[0], item[1]))
            output[target] = values[window_index, horizon]
            continue
        if method == "overlap_median":
            output[target] = np.median(np.asarray([values[i, h] for i, h in contributors]), axis=0)
            continue
        weights = []
        for window_index, horizon in contributors:
            if method == "confidence_weighted_average":
                weight = float(confidence_array[window_index, horizon]) if confidence_array is not None else 1.0
            else:
                center = (HORIZON - 1) / 2.0
                weight = 1.0 / (1.0 + abs(float(horizon) - center))
            weights.append(max(weight, 1.0e-12))
        weight_array = np.asarray(weights, dtype=np.float64)
        sample_array = np.asarray([values[i, h] for i, h in contributors], dtype=np.float64)
        output[target] = np.sum(sample_array * weight_array[:, None], axis=0) / np.sum(weight_array)
    return output, {
        "method": method,
        "window_count": int(len(values)),
        "forecast_sample_count": int(last_target),
        "overlap_min": int(overlap_counts.min()),
        "overlap_max": int(overlap_counts.max()),
        "overlap_mean": float(overlap_counts.mean()),
        "overlap_counts": overlap_counts.tolist(),
    }


def reconstruct_primitive(
    segment: Any,
    window_ids: Sequence[int],
    predictions_by_id: Mapping[int, np.ndarray],
    *,
    method: str = "center_weighted_consensus",
    confidences_by_id: Mapping[int, Sequence[float]] | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    """Build one complete primitive from causal history plus reconciled windows."""

    ordered_ids = [int(item) for item in window_ids]
    if len(set(ordered_ids)) != len(ordered_ids):
        raise ValueError("duplicate_window_ids")
    predictions = np.asarray([predictions_by_id[item] for item in ordered_ids], dtype=np.float64)
    starts = list(range(len(ordered_ids)))
    confidence = None
    if confidences_by_id is not None:
        confidence = np.asarray([confidences_by_id[item] for item in ordered_ids], dtype=np.float64)
    forecast, overlap = reconcile_overlaps(predictions, starts, method=method, confidences=confidence)
    times = np.asarray(segment.times - segment.times[0], dtype=np.float64)
    positions = np.asarray(segment.positions, dtype=np.float64).copy()
    expected_forecast_count = len(positions) - HISTORY
    if len(forecast) != expected_forecast_count:
        raise ValueError(f"forecast_length_mismatch:{len(forecast)}:{expected_forecast_count}")
    positions[HISTORY:] = forecast
    if not np.all(np.diff(times) > 0.0) or not np.all(np.isfinite(positions)):
        raise ValueError("invalid_reconstructed_primitive")
    record = {
        "primitive_id": str(segment.primitive_id),
        "segment_id": int(segment.segment_id),
        "segment_order": int(segment.segment_order),
        "segment_key": str(segment.key),
        "spray_mode": str(segment.spray_state),
        "ordered_window_ids": ordered_ids,
        "source_sample_ids": [int(item) for item in segment.sample_indices],
        "timestamps_s": times.tolist(),
        "reconciliation": overlap,
        "controlled_stop_count": int(getattr(segment, "controlled_stop_count", 0) or 0),
        "causal_history_sample_count": HISTORY,
        "joint_positions_rad": positions.tolist(),
    }
    return record, positions


def _first_violation(mask: np.ndarray, values: np.ndarray, limits: np.ndarray, quantity: str) -> dict[str, Any] | None:
    indices = np.argwhere(mask)
    if len(indices) == 0:
        return None
    index = tuple(int(item) for item in indices[0])
    sample, joint = index[0], index[1]
    return {
        "quantity": quantity,
        "sample_index": sample,
        "edge_index": max(sample - 1, 0),
        "joint_index": joint,
        "value": float(values[index]),
        "absolute_value": float(abs(values[index])),
        "limit": float(limits[joint]),
        "boundary_vs_interior": "boundary" if sample in {0, len(values) - 1} else "interior",
    }


def finite_difference_diagnostics(times: Sequence[float], positions: Sequence[Sequence[float]], *, position_lower: np.ndarray = POSITION_LOWER, position_upper: np.ndarray = POSITION_UPPER, velocity_limit: np.ndarray = VELOCITY_LIMIT, acceleration_limit: np.ndarray = ACCELERATION_LIMIT, jerk_limit: np.ndarray = JERK_LIMIT, persist_derivatives: bool = True) -> dict[str, Any]:
    """Compute explicit q, dq/dt, d²q/dt², d³q/dt³ diagnostics."""

    t = _as_float_array(times, (len(times),), "times")
    q = _as_float_array(positions, (len(times), JOINTS), "positions")
    if len(t) < 4 or np.any(np.diff(t) <= 0.0):
        raise ValueError("invalid_dynamics_timestamps")
    velocity = np.gradient(q, t, axis=0, edge_order=1)
    acceleration = np.gradient(velocity, t, axis=0, edge_order=1)
    jerk = np.gradient(acceleration, t, axis=0, edge_order=1)
    masks = {
        "position": (q < position_lower[None, :]) | (q > position_upper[None, :]),
        "velocity": np.abs(velocity) > velocity_limit[None, :],
        "acceleration": np.abs(acceleration) > acceleration_limit[None, :],
        "jerk": np.abs(jerk) > jerk_limit[None, :],
    }
    values = {"position": q, "velocity": velocity, "acceleration": acceleration, "jerk": jerk}
    limits = {"position": np.maximum(np.abs(position_lower), np.abs(position_upper)), "velocity": velocity_limit, "acceleration": acceleration_limit, "jerk": jerk_limit}
    first = {key: _first_violation(mask, values[key], limits[key], key) for key, mask in masks.items()}
    counts_by_joint = {key: np.count_nonzero(mask, axis=0).astype(int).tolist() for key, mask in masks.items()}
    result = {
        "finite": bool(np.all(np.isfinite(q)) and np.all(np.isfinite(velocity)) and np.all(np.isfinite(acceleration)) and np.all(np.isfinite(jerk))),
        "minimum_positive_dt_s": float(np.min(np.diff(t))),
        "position_violation_count": int(np.count_nonzero(masks["position"])),
        "velocity_violation_count": int(np.count_nonzero(masks["velocity"])),
        "acceleration_violation_count": int(np.count_nonzero(masks["acceleration"])),
        "jerk_violation_count": int(np.count_nonzero(masks["jerk"])),
        "counts_by_joint": counts_by_joint,
        "first_failure_by_constraint": first,
        "boundary_vs_interior": {key: {"boundary": int(sum(item is not None and item["boundary_vs_interior"] == "boundary" for item in [_first_violation(mask, values[key], limits[key], key)])), "interior": int(np.count_nonzero(mask))} for key, mask in masks.items()},
    }
    if persist_derivatives:
        result["derivative_reconstruction"] = {
            "q": q.tolist(),
            "dq_dt": velocity.tolist(),
            "d2q_dt2": acceleration.tolist(),
            "d3q_dt3": jerk.tolist(),
        }
    return result


def causal_primitive_repair(times: Sequence[float], positions: Sequence[Sequence[float]], *, history_count: int = HISTORY, velocity_limit: np.ndarray = VELOCITY_LIMIT, acceleration_limit: np.ndarray = ACCELERATION_LIMIT, jerk_limit: np.ndarray = JERK_LIMIT) -> tuple[np.ndarray, dict[str, Any]]:
    """Sequential dynamic projection using only preceding repaired states.

    This is a deterministic safety repair, not a claim of process-manifold
    certification.  Process and collision gates must still run natively.
    """

    t = _as_float_array(times, (len(times),), "times")
    raw = _as_float_array(positions, (len(times), JOINTS), "positions")
    repaired = raw.copy()
    modified = 0
    for index in range(max(1, int(history_count)), len(repaired)):
        dt = float(t[index] - t[index - 1])
        previous_v = (repaired[index - 1] - repaired[index - 2]) / float(t[index - 1] - t[index - 2]) if index >= 2 else np.zeros(JOINTS)
        previous_a = ((repaired[index - 1] - repaired[index - 2]) / float(t[index - 1] - t[index - 2]) - (repaired[index - 2] - repaired[index - 3]) / float(t[index - 2] - t[index - 3])) / float(t[index - 1] - t[index - 2]) if index >= 3 else np.zeros(JOINTS)
        desired_v = (raw[index] - repaired[index - 1]) / dt
        bounded_v = np.clip(desired_v, -velocity_limit, velocity_limit)
        bounded_a = np.clip((bounded_v - previous_v) / dt, -acceleration_limit, acceleration_limit)
        bounded_a = np.clip(bounded_a, previous_a - jerk_limit * dt, previous_a + jerk_limit * dt)
        candidate = repaired[index - 1] + (previous_v + bounded_a * dt) * dt
        if not np.allclose(candidate, raw[index], atol=0.0, rtol=0.0):
            modified += 1
        repaired[index] = candidate
    return repaired, {"method": "causal_sequential_velocity_acceleration_jerk_projection_v1", "history_count": int(history_count), "modified_sample_count": int(modified), "position_path_changed": bool(modified)}


def audit_network_dependency(source_paths: Iterable[Path], runtime_evidence: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Audit only the Ruckig execution path for remote/network dependencies."""

    markers = re.compile(r"https?://|\\b(?:socket|requests|urllib|curl|libcurl|grpc|cloud_api)\\b", re.IGNORECASE)
    findings: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    intermediate_positions = False
    for path in sorted({Path(item).resolve() for item in source_paths if Path(item).is_file()}):
        text = path.read_text(encoding="utf-8", errors="replace")
        intermediate_positions = intermediate_positions or "intermediate_positions" in text
        matched = [{"line": index, "text": line.strip()[:240]} for index, line in enumerate(text.splitlines(), 1) if markers.search(line)]
        files.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "network_marker_count": len(matched), "network_markers": matched})
        findings.extend([{**item, "path": str(path)} for item in matched])
    runtime = dict(runtime_evidence or {})
    intermediate_positions_nonempty = runtime.get("intermediate_positions_nonempty")
    if intermediate_positions_nonempty is None:
        intermediate_positions_nonempty = intermediate_positions
    direct_network = bool(findings) or bool(runtime.get("network_request_observed"))
    return {
        "schema_version": "stage3_h12_r3_network_dependency_audit_v1",
        "NETWORK_DEPENDENCY_FOUND": "YES" if direct_network else "NO",
        "OFFLINE_RUCKIG_PATH": "YES" if not direct_network and not intermediate_positions_nonempty else "NO",
        "ruckig_distribution": runtime.get("ruckig_distribution", "not_recorded"),
        "ruckig_version": runtime.get("ruckig_version", "not_recorded_in_runtime_evidence"),
        "binding": "MoveIt2 RobotTrajectory native C++ API" if runtime.get("moveit_ruckig_api") else "not_recorded",
        "moveit_ruckig_smoothing_api": bool(runtime.get("moveit_ruckig_api")),
        "direct_update_call": bool(runtime.get("direct_update_call")),
        "direct_calculate_call": bool(runtime.get("direct_calculate_call")),
        "intermediate_positions_field_present": bool(intermediate_positions),
        "intermediate_positions_nonempty": bool(intermediate_positions_nonempty),
        "intermediate_positions": bool(intermediate_positions_nonempty),
        "http_https_request": bool(findings),
        "dns_socket_evidence": runtime.get("dns_socket_evidence", "not_observed"),
        "cloud_api_evidence": "not_observed" if not intermediate_positions else "requires_runtime_proof",
        "remote_service_dependency": "NO" if not direct_network and not intermediate_positions_nonempty else "YES",
        "findings": findings,
        "files": files,
        "runtime_evidence": runtime,
    }


def first_failure_from_diagnostics(metrics: Mapping[str, Any]) -> dict[str, Any] | None:
    for key in ("position", "velocity", "acceleration", "jerk"):
        item = (metrics.get("first_failure_by_constraint") or {}).get(key)
        if item is not None:
            return item
    return None


__all__ = [
    "ACCELERATION_LIMIT",
    "HISTORY",
    "JERK_LIMIT",
    "POSITION_LOWER",
    "POSITION_UPPER",
    "RUCKIG_RESULT_CLASSES",
    "VELOCITY_LIMIT",
    "audit_network_dependency",
    "canonical_sha256",
    "causal_primitive_repair",
    "classify_ceiling",
    "classify_ruckig_result",
    "finite_difference_diagnostics",
    "first_failure_from_diagnostics",
    "reconcile_overlaps",
    "reconstruct_primitive",
    "ruckig_completion_gate",
    "validate_ruckig_input",
]
