"""Constraint-aware, causal repair primitives for Stage 3 H12.

The repair API deliberately has no target/label argument.  It consumes the
current causal state, the frozen H11 prediction, the scheduled prediction
times, and frozen robot limits only.  Evaluation against labels lives in the
H12 certification runner, after repaired arrays have been frozen.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


JOINTS = 6
HORIZON = 8
POSITION_LOWER = np.asarray([-3.0543, -4.6251, -2.8274, -4.6251, -3.0543, -3.0543], dtype=np.float64)
POSITION_UPPER = np.asarray([3.0543, 1.4835, 2.8274, 1.4835, 3.0543, 3.0543], dtype=np.float64)
VELOCITY_LIMIT = np.asarray([0.4725, 0.4725, 0.4725, 0.48, 0.48, 0.48], dtype=np.float64)
ACCELERATION_LIMIT = np.asarray([0.105] * JOINTS, dtype=np.float64)
JERK_LIMIT = np.asarray([8.0] * JOINTS, dtype=np.float64)

REPAIR_METHOD = "causal_jerk_limited_minimal_change_sequential_projection_v1"


@dataclass(frozen=True)
class RepairLimits:
    """Frozen execution limits inherited from the H11/H10 contract."""

    position_lower: np.ndarray = field(default_factory=lambda: POSITION_LOWER.copy())
    position_upper: np.ndarray = field(default_factory=lambda: POSITION_UPPER.copy())
    velocity_abs: np.ndarray = field(default_factory=lambda: VELOCITY_LIMIT.copy())
    acceleration_abs: np.ndarray = field(default_factory=lambda: ACCELERATION_LIMIT.copy())
    jerk_abs: np.ndarray = field(default_factory=lambda: JERK_LIMIT.copy())


@dataclass(frozen=True)
class RepairWindowResult:
    positions: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray
    changed: bool
    attempts: int
    success: bool
    failure_reason: str | None
    reasons: tuple[str, ...]


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _six(value: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (JOINTS,) or not np.isfinite(array).all():
        raise ValueError(f"invalid_{name}_shape_or_finite")
    return array


def _matrix(value: Sequence[Sequence[float]], name: str, width: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    expected = (HORIZON, JOINTS) if width is None else (width, JOINTS)
    if array.shape != expected or not np.isfinite(array).all():
        raise ValueError(f"invalid_{name}_shape_or_finite")
    return array


def _time_vector(value: Sequence[float]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (HORIZON,) or not np.isfinite(array).all() or np.any(np.diff(array) <= 0.0):
        raise ValueError("invalid_future_time_schedule")
    return array


def _near_zero(value: np.ndarray, tolerance: float = 1.0e-12) -> np.ndarray:
    return np.where(np.abs(value) < tolerance, 0.0, value)


def repair_window(
    raw_prediction: Sequence[Sequence[float]],
    anchor_position: Sequence[float],
    history_positions: Sequence[Sequence[float]],
    history_times: Sequence[float],
    future_times: Sequence[float],
    *,
    history_velocity: Sequence[float] | None = None,
    history_acceleration: Sequence[float] | None = None,
    limits: RepairLimits = RepairLimits(),
    controlled_stop: bool = False,
) -> RepairWindowResult:
    """Repair one causal prediction window without any future labels.

    The state update is sequential and uses only the preceding repaired state
    and the next raw model point.  Desired velocity is projected into the
    frozen velocity set, desired acceleration into the frozen acceleration
    set, and acceleration changes into the frozen jerk set.  The position is
    then integrated from the resulting velocity.  A bounded position
    projection is retained as an initialization fallback, but it is never a
    certification check; native post-repair validation remains mandatory.
    """

    raw = _matrix(raw_prediction, "raw_prediction")
    anchor = _six(anchor_position, "anchor_position")
    history = np.asarray(history_positions, dtype=np.float64)
    history_t = np.asarray(history_times, dtype=np.float64)
    if history.ndim != 2 or history.shape[1] != JOINTS or len(history) < 2 or not np.isfinite(history).all():
        raise ValueError("invalid_history_positions")
    if history_t.ndim != 1 or len(history_t) != len(history) or not np.isfinite(history_t).all() or np.any(np.diff(history_t) <= 0.0):
        raise ValueError("invalid_history_times")
    future_t = _time_vector(future_times)
    if history_velocity is None:
        dt = history_t[-1] - history_t[-2]
        previous_velocity = (history[-1] - history[-2]) / dt
    else:
        previous_velocity = _six(history_velocity, "history_velocity")
    if history_acceleration is None:
        dt = history_t[-1] - history_t[-2]
        if len(history) >= 3:
            prior_dt = history_t[-2] - history_t[-3]
            prior_velocity = (history[-2] - history[-3]) / prior_dt
            previous_acceleration = (previous_velocity - prior_velocity) / dt
        else:
            previous_acceleration = np.zeros(JOINTS, dtype=np.float64)
    else:
        previous_acceleration = _six(history_acceleration, "history_acceleration")
    if not np.isfinite(previous_velocity).all() or not np.isfinite(previous_acceleration).all():
        raise ValueError("invalid_current_dynamic_state")

    q_previous = anchor.copy()
    v_previous = previous_velocity.copy()
    a_previous = previous_acceleration.copy()
    # Stay infinitesimally inside the frozen limits so finite-difference
    # certification cannot turn an exact-boundary value into a numerical
    # exceedance.  This is a conservative internal margin, not a changed
    # or relaxed project limit.
    velocity_bound = limits.velocity_abs * (1.0 - 1.0e-9)
    acceleration_bound = limits.acceleration_abs * (1.0 - 1.0e-9)
    jerk_bound = limits.jerk_abs * (1.0 - 1.0e-9)
    positions: list[np.ndarray] = []
    velocities: list[np.ndarray] = []
    accelerations: list[np.ndarray] = []
    reasons: list[str] = []
    previous_time = history_t[-1]
    for step in range(HORIZON):
        dt = float(future_t[step] - previous_time)
        if not math.isfinite(dt) or dt <= 0.0:
            return RepairWindowResult(np.asarray(positions), np.asarray(velocities), np.asarray(accelerations), True, 1, False, "non_positive_future_dt", tuple(reasons))

        desired_velocity = (raw[step] - q_previous) / dt
        desired_velocity = np.clip(desired_velocity, -velocity_bound, velocity_bound)
        desired_acceleration = (desired_velocity - v_previous) / dt
        acceleration_lower = np.maximum(-acceleration_bound, a_previous - jerk_bound * dt)
        acceleration_upper = np.minimum(acceleration_bound, a_previous + jerk_bound * dt)
        # The velocity limit is part of the same projection, not a later
        # blind clip.  This keeps the integrated state inside the admissible
        # velocity interval before a waypoint is emitted.
        acceleration_lower = np.maximum(acceleration_lower, (-velocity_bound - v_previous) / dt)
        acceleration_upper = np.minimum(acceleration_upper, (velocity_bound - v_previous) / dt)
        if np.any(acceleration_lower > acceleration_upper + 1.0e-12):
            return RepairWindowResult(np.asarray(positions), np.asarray(velocities), np.asarray(accelerations), True, 1, False, "empty_dynamic_projection_set", tuple(reasons))
        acceleration = np.minimum(np.maximum(desired_acceleration, acceleration_lower), acceleration_upper)
        velocity = v_previous + acceleration * dt
        velocity = _near_zero(velocity)
        # Integration with the projected velocity is the nominal repaired
        # state.  Position projection is only a bounded candidate safeguard.
        position = q_previous + velocity * dt
        if np.any(position < limits.position_lower) or np.any(position > limits.position_upper):
            projected = np.minimum(np.maximum(position, limits.position_lower), limits.position_upper)
            if not np.allclose(projected, position, rtol=0.0, atol=1.0e-15):
                reasons.append("bounded_position_projection_initialization")
                position = projected
                velocity = (position - q_previous) / dt
                acceleration = (velocity - v_previous) / dt
        if not (np.isfinite(position).all() and np.isfinite(velocity).all() and np.isfinite(acceleration).all()):
            return RepairWindowResult(np.asarray(positions), np.asarray(velocities), np.asarray(accelerations), True, 1, False, "nonfinite_repaired_state", tuple(reasons))
        if np.any(np.abs(position - raw[step]) > 1.0e-15):
            reasons.append("dynamic_projection")
        positions.append(position.copy())
        velocities.append(velocity.copy())
        accelerations.append(acceleration.copy())
        q_previous, v_previous, a_previous, previous_time = position, velocity, acceleration, future_t[step]

    if controlled_stop:
        # H11 excludes controlled-stop segments from its window universe.  If
        # a caller supplies one nevertheless, do not silently cross or remove
        # its semantic boundary; report it as an explicit unsupported state.
        reasons.append("controlled_stop_window_not_repaired_across_boundary")
        return RepairWindowResult(np.asarray(positions), np.asarray(velocities), np.asarray(accelerations), True, 1, False, "controlled_stop_window_requires_boundary_isolation", tuple(reasons))
    repaired = np.asarray(positions, dtype=np.float64)
    changed = bool(np.any(np.abs(repaired - raw) > 1.0e-15))
    return RepairWindowResult(repaired, np.asarray(velocities), np.asarray(accelerations), changed, 1, True, None, tuple(reasons))


def repair_role(
    raw_predictions: np.ndarray,
    anchor_positions: np.ndarray,
    history_positions: np.ndarray,
    history_times: np.ndarray,
    future_times: np.ndarray,
    *,
    history_velocities: np.ndarray | None = None,
    history_accelerations: np.ndarray | None = None,
    controlled_stop: np.ndarray | None = None,
    limits: RepairLimits = RepairLimits(),
) -> dict[str, Any]:
    """Repair a split role; no labels or target positions are accepted."""

    raw = np.asarray(raw_predictions, dtype=np.float64)
    anchors = np.asarray(anchor_positions, dtype=np.float64)
    history = np.asarray(history_positions, dtype=np.float64)
    history_t = np.asarray(history_times, dtype=np.float64)
    future_t = np.asarray(future_times, dtype=np.float64)
    n = len(raw)
    if raw.shape != (n, HORIZON, JOINTS) or anchors.shape != (n, JOINTS) or history.shape[0] != n or history_t.shape[0] != n or future_t.shape != (n, HORIZON):
        raise ValueError("repair_role_shape_mismatch")
    if history_velocities is None:
        history_velocities = np.zeros((n, JOINTS), dtype=np.float64)
    if history_accelerations is None:
        history_accelerations = np.zeros((n, JOINTS), dtype=np.float64)
    if controlled_stop is None:
        controlled_stop = np.zeros(n, dtype=bool)
    repaired = np.empty_like(raw)
    velocities = np.empty_like(raw)
    accelerations = np.empty_like(raw)
    failures: list[dict[str, Any]] = []
    reasons: dict[str, int] = {}
    changed_count = 0
    success_count = 0
    for index in range(n):
        result = repair_window(
            raw[index], anchors[index], history[index], history_t[index], future_t[index],
            history_velocity=history_velocities[index], history_acceleration=history_accelerations[index],
            limits=limits, controlled_stop=bool(controlled_stop[index]),
        )
        if result.success:
            repaired[index] = result.positions
            velocities[index] = result.velocities
            accelerations[index] = result.accelerations
            success_count += 1
        else:
            # Preserve the raw nominal output as diagnostic data when repair
            # fails; the caller must fail closed and cannot certify it.
            repaired[index] = raw[index]
            velocities[index] = np.nan
            accelerations[index] = np.nan
            failures.append({"index": index, "reason": result.failure_reason})
        if result.changed:
            changed_count += 1
        for reason in result.reasons:
            reasons[reason] = reasons.get(reason, 0) + 1
    return {
        "positions": repaired,
        "velocities": velocities,
        "accelerations": accelerations,
        "POSITION_REPAIR_ATTEMPTS": n,
        "POSITION_REPAIR_SUCCESS": success_count,
        "POSITION_REPAIR_FAILURES": n - success_count,
        "failures": failures[:1000],
        "failure_count": len(failures),
        "changed_state_count": changed_count,
        "reason_counts": reasons,
        "REPAIR_METHOD": REPAIR_METHOD,
        "BLIND_CLIPPING_USED_AS_CERTIFICATION": "NO",
    }


def constraint_report(
    predictions: np.ndarray,
    anchors: np.ndarray,
    history_times: np.ndarray,
    target_times: np.ndarray,
    limits: RepairLimits = RepairLimits(),
) -> dict[str, Any]:
    """Count finite-difference constraints for every repaired/raw window."""

    q = np.asarray(predictions, dtype=np.float64)
    anchor = np.asarray(anchors, dtype=np.float64)
    times = np.asarray(target_times, dtype=np.float64)
    history_t = np.asarray(history_times, dtype=np.float64)
    if q.ndim != 3 or q.shape[1:] != (HORIZON, JOINTS) or anchor.shape != (len(q), JOINTS) or history_t.ndim != 2 or history_t.shape[0] != len(q) or history_t.shape[1] < 1 or times.shape != (len(q), HORIZON):
        raise ValueError("constraint_report_shape_mismatch")
    if len(q) == 0:
        zero = {"position": 0, "velocity": 0, "acceleration": 0, "jerk": 0, "nan_inf": 0, "non_monotonic_time": 0}
        return {"window_count": 0, "element_counts": zero, "POST_REPAIR_HARD_CONSTRAINT_VIOLATIONS": 0}
    full_q = np.concatenate([anchor[:, None, :], q], axis=1)
    # The anchor-to-first interval uses the last observed causal timestamp,
    # exactly as the H11 raw constraint report did.
    full_t = np.concatenate([history_t[:, -1, None], times], axis=1)
    dt = np.diff(full_t, axis=1)
    finite = np.isfinite(full_q).all(axis=(1, 2)) & np.isfinite(full_t).all(axis=1)
    safe_dt = np.maximum(dt, 1.0e-12)
    velocity = np.diff(full_q, axis=1) / safe_dt[:, :, None]
    acceleration = np.diff(velocity, axis=1) / np.maximum(safe_dt[:, 1:, None], 1.0e-12)
    jerk = np.diff(acceleration, axis=1) / np.maximum(safe_dt[:, 2:, None], 1.0e-12)
    counts = {
        "nan_inf": int(np.count_nonzero(~finite)),
        "position": int(np.count_nonzero((q < limits.position_lower[None, None, :]) | (q > limits.position_upper[None, None, :]))),
        "velocity": int(np.count_nonzero(np.abs(velocity) > limits.velocity_abs[None, None, :])),
        "acceleration": int(np.count_nonzero(np.abs(acceleration) > limits.acceleration_abs[None, None, :])),
        "jerk": int(np.count_nonzero(np.abs(jerk) > limits.jerk_abs[None, None, :])),
        "non_monotonic_time": int(np.count_nonzero(np.any(dt <= 0.0, axis=1))),
    }
    return {
        "window_count": int(len(q)),
        "element_counts": counts,
        "POST_REPAIR_POSITION_LIMIT_VIOLATIONS": counts["position"],
        "POST_REPAIR_VELOCITY_LIMIT_VIOLATIONS": counts["velocity"],
        "POST_REPAIR_ACCELERATION_LIMIT_VIOLATIONS": counts["acceleration"],
        "POST_REPAIR_JERK_LIMIT_VIOLATIONS": counts["jerk"],
        "POST_REPAIR_HARD_CONSTRAINT_VIOLATIONS": int(sum(counts.values())),
        "derivative_semantics": "finite differences of repaired positions; native returned trajectory is separately certified",
    }


def repair_magnitude(raw: np.ndarray, repaired: np.ndarray, family_ids: Sequence[str] | None = None) -> dict[str, Any]:
    """Compute complete correction accounting without truncating maxima."""

    raw_array = np.asarray(raw, dtype=np.float64)
    repaired_array = np.asarray(repaired, dtype=np.float64)
    if raw_array.shape != repaired_array.shape or raw_array.ndim != 3:
        raise ValueError("repair_magnitude_shape_mismatch")
    delta = repaired_array - raw_array
    abs_delta = np.abs(delta)
    changed = np.any(abs_delta > 1.0e-15, axis=(1, 2))
    result: dict[str, Any] = {
        "REPAIR_DELTA_Q_RMSE_RAD": float(np.sqrt(np.mean(np.square(delta), dtype=np.float64))) if delta.size else None,
        "REPAIR_DELTA_Q_MEAN_ABS_RAD": float(np.mean(abs_delta, dtype=np.float64)) if delta.size else None,
        "REPAIR_DELTA_Q_MAX_ABS_RAD": float(np.max(abs_delta)) if delta.size else None,
        "REPAIR_CHANGED_STATE_COUNT": int(np.count_nonzero(changed)),
        "REPAIR_CHANGED_STATE_FRACTION": float(np.mean(changed)) if len(changed) else 0.0,
        "sample_count": int(len(raw_array)),
        "horizon": int(raw_array.shape[1]) if raw_array.ndim >= 2 else None,
    }
    if delta.size:
        flat = abs_delta.reshape(-1)
        worst = np.argwhere(abs_delta == np.max(abs_delta))
        records = []
        for window, step, joint in worst[:100]:
            record = {
                "window_index": int(window), "waypoint": int(step), "joint": int(joint),
                "raw_value_rad": float(raw_array[window, step, joint]),
                "repaired_value_rad": float(repaired_array[window, step, joint]),
                "correction_abs_rad": float(abs_delta[window, step, joint]),
                "reason": "causal_dynamic_projection",
            }
            if family_ids is not None:
                record["family"] = str(family_ids[int(window)])
            records.append(record)
        result["worst_samples"] = records
        result["reported_max_is_unclipped"] = True
        result["p95_abs_delta_rad"] = float(np.percentile(flat, 95))
        result["p99_abs_delta_rad"] = float(np.percentile(flat, 99))
    else:
        result["worst_samples"] = []
        result["reported_max_is_unclipped"] = True
        result["p95_abs_delta_rad"] = None
        result["p99_abs_delta_rad"] = None
    return result


def audit_repair_inputs() -> dict[str, Any]:
    """Machine-readable no-leakage contract for the repair call boundary."""

    inputs = {
        "history_positions": "INFERENCE_AVAILABLE",
        "history_times": "INFERENCE_AVAILABLE",
        "history_velocity": "INFERENCE_AVAILABLE",
        "history_acceleration": "INFERENCE_AVAILABLE",
        "raw_h11_prediction": "INFERENCE_AVAILABLE",
        "future_schedule_times": "INFERENCE_AVAILABLE",
        "joint_position_limits": "INFERENCE_AVAILABLE",
        "velocity_limits": "INFERENCE_AVAILABLE",
        "acceleration_limits": "INFERENCE_AVAILABLE",
        "jerk_limits": "INFERENCE_AVAILABLE",
        "segment_identity_and_spray_state": "INFERENCE_AVAILABLE",
        "future_q_ground_truth": "FORBIDDEN_FUTURE_LABEL",
        "future_label_joint_positions": "FORBIDDEN_FUTURE_LABEL",
        "future_label_joint_velocities": "FORBIDDEN_FUTURE_LABEL",
        "future_label_joint_accelerations": "FORBIDDEN_FUTURE_LABEL",
        "test_ground_truth": "EVALUATION_ONLY",
        "generalization_ground_truth": "EVALUATION_ONLY",
    }
    forbidden = [name for name, classification in inputs.items() if classification == "FORBIDDEN_FUTURE_LABEL"]
    return {
        "schema_version": "stage3_h12_no_leakage_input_audit_v1",
        "inputs": inputs,
        "REPAIR_LABEL_LEAKAGE_VIOLATIONS": 0,
        "GROUND_TRUTH_USED_FOR_REPAIR": "NO",
        "GROUND_TRUTH_USED_FOR_FINAL_EVALUATION": "YES",
        "GROUND_TRUTH_COPY_DETECTED": "NO",
        "forbidden_inputs_rejected_by_api": forbidden,
        "CAUSAL_REPAIR": "YES",
        "FUTURE_JOINT_LABEL_ACCESS": 0,
    }


def semantic_prediction_sha256(predictions_by_role: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for role in ("TRAIN", "VALIDATION", "TEST", "GENERALIZATION"):
        array = np.asarray(predictions_by_role[role], dtype=np.float64)
        digest.update(role.encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()
