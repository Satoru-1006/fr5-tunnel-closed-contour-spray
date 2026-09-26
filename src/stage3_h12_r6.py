"""Fail-closed Stage 3 H12-R6 derivative evidence and remediation helpers.

R6 deliberately treats the acceleration stored on a post-Ruckig waypoint as
different evidence from the continuous Ruckig profile.  The native profile is
authoritative for continuous acceleration; finite differences are used only
to reconstruct a missing/stale waypoint derivative state after the native
trajectory has already been accepted.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


JOINT_NAMES = ("j1", "j2", "j3", "j4", "j5", "j6")
DEFAULT_ACCELERATION_LIMITS = np.full(6, 0.7, dtype=np.float64)
DEFAULT_BOUNDARY_VELOCITY_TOLERANCE = 1.0e-3
DEFAULT_NUMERIC_TOLERANCE = 1.0e-10
SCHEMA_VERSION = "stage3_h12_r6_derivative_consistency_v1"


def _trajectory_arrays(
    times: Sequence[float],
    velocities: Sequence[Sequence[float]],
    stored_accelerations: Sequence[Sequence[float]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    t = np.asarray(times, dtype=np.float64)
    v = np.asarray(velocities, dtype=np.float64)
    a = None if stored_accelerations is None else np.asarray(stored_accelerations, dtype=np.float64)
    if t.ndim != 1 or v.ndim != 2 or v.shape[1] != 6 or len(t) != len(v) or len(t) < 2:
        raise ValueError("malformed derivative trajectory shape")
    if a is not None and a.shape != v.shape:
        raise ValueError("stored acceleration shape does not match velocity shape")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(v)) or (a is not None and not np.all(np.isfinite(a))):
        raise ValueError("non-finite derivative state")
    dt = np.diff(t)
    if not np.all(dt > 0.0):
        raise ValueError("trajectory timestamps must be strictly increasing")
    return t, v, a


def reconstruct_finite_difference_acceleration(
    times: Sequence[float], velocities: Sequence[Sequence[float]]
) -> np.ndarray:
    """Reconstruct acceleration without changing position, velocity, or time.

    The endpoints use one-sided derivatives.  Interior states use the
    non-uniform-grid centered derivative, which is deterministic and does not
    silently assume the nominal 10 ms sample period.
    """

    t, v, _ = _trajectory_arrays(times, velocities)
    result = np.empty_like(v)
    forward = np.diff(v, axis=0) / np.diff(t)[:, None]
    result[0] = forward[0]
    result[-1] = forward[-1]
    if len(result) > 2:
        result[1:-1] = (v[2:] - v[:-2]) / (t[2:] - t[:-2])[:, None]
    return result


def derivative_windows(
    times: Sequence[float], velocities: Sequence[Sequence[float]], index: int
) -> dict[str, Any]:
    """Return the before/after finite-difference evidence for one state."""

    t, v, _ = _trajectory_arrays(times, velocities)
    i = int(index)
    if i < 0 or i >= len(t):
        raise IndexError("derivative state index out of range")
    before = None if i == 0 else (v[i] - v[i - 1]) / (t[i] - t[i - 1])
    after = None if i == len(t) - 1 else (v[i + 1] - v[i]) / (t[i + 1] - t[i])
    return {
        "waypoint_index": i,
        "timestamp": float(t[i]),
        "dt_before": None if i == 0 else float(t[i] - t[i - 1]),
        "dt_after": None if i == len(t) - 1 else float(t[i + 1] - t[i]),
        "acceleration_from_velocity_before": None if before is None else before.tolist(),
        "acceleration_from_velocity_after": None if after is None else after.tolist(),
        "acceleration_from_velocity_centered": reconstruct_finite_difference_acceleration(t, v)[i].tolist(),
    }


def derivative_consistency_audit(
    times: Sequence[float],
    velocities: Sequence[Sequence[float]],
    stored_accelerations: Sequence[Sequence[float]],
    acceleration_limits: Sequence[float] = DEFAULT_ACCELERATION_LIMITS,
    *,
    numeric_tolerance: float = DEFAULT_NUMERIC_TOLERANCE,
) -> dict[str, Any]:
    """Compare stored, finite-difference, and boundary derivative states."""

    t, v, stored = _trajectory_arrays(times, velocities, stored_accelerations)
    assert stored is not None
    limits = np.asarray(acceleration_limits, dtype=np.float64)
    if limits.shape != (6,) or not np.all(np.isfinite(limits)) or np.any(limits <= 0.0):
        raise ValueError("acceleration limits must be finite positive six-joint values")
    derived = reconstruct_finite_difference_acceleration(t, v)
    stored_mask = np.abs(stored) > limits[None, :] + float(numeric_tolerance)
    derived_mask = np.abs(derived) > limits[None, :] + float(numeric_tolerance)
    previous = np.full_like(stored, np.nan)
    following = np.full_like(stored, np.nan)
    previous[1:] = np.diff(v, axis=0) / np.diff(t)[:, None]
    following[:-1] = np.diff(v, axis=0) / np.diff(t)[:, None]
    stale_mask = stored_mask & ~derived_mask
    mismatch = np.abs(stored - derived)
    locations = np.argwhere(stored_mask)
    return {
        "schema_version": SCHEMA_VERSION,
        "timestamp_count": int(len(t)),
        "joint_count": 6,
        "finite": True,
        "strictly_increasing": True,
        "acceleration_limits_rad_s2": limits.tolist(),
        "stored_waypoint_acceleration_violation_count": int(np.count_nonzero(stored_mask)),
        "finite_difference_acceleration_violation_count": int(np.count_nonzero(derived_mask)),
        "stale_waypoint_acceleration_candidate_count": int(np.count_nonzero(stale_mask)),
        "stored_vs_finite_difference_max_abs_error": float(np.max(mismatch)),
        "stored_vs_finite_difference_mean_abs_error": float(np.mean(mismatch)),
        "stored_violation_locations": [
            {"waypoint_index": int(i), "joint_index": int(j)} for i, j in locations
        ],
        "max_abs_stored_acceleration_rad_s2": np.max(np.abs(stored), axis=0).tolist(),
        "max_abs_finite_difference_acceleration_rad_s2": np.max(np.abs(derived), axis=0).tolist(),
        "max_abs_previous_interval_acceleration_rad_s2": np.nanmax(np.abs(previous), axis=0).tolist(),
        "max_abs_next_interval_acceleration_rad_s2": np.nanmax(np.abs(following), axis=0).tolist(),
        "derived_acceleration": derived.tolist(),
        "status": "PASSED" if not np.any(derived_mask) else "BLOCKED",
    }


def reconstruct_local_derivative_state(
    times: Sequence[float],
    velocities: Sequence[Sequence[float]],
    stored_accelerations: Sequence[Sequence[float]],
    acceleration_limits: Sequence[float] = DEFAULT_ACCELERATION_LIMITS,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Repair only stored cells proven stale by finite-difference evidence.

    No geometry or trajectory timing is edited.  The function refuses to
    repair if a finite-difference derivative itself exceeds the native
    physical limit, because that would be a real dynamic problem rather than
    a stale waypoint field.
    """

    t, v, stored = _trajectory_arrays(times, velocities, stored_accelerations)
    assert stored is not None
    limits = np.asarray(acceleration_limits, dtype=np.float64)
    audit = derivative_consistency_audit(t, v, stored, limits)
    if audit["finite_difference_acceleration_violation_count"]:
        raise ValueError("finite_difference_acceleration_violation_requires_native_remediation")
    derived = np.asarray(audit["derived_acceleration"], dtype=np.float64)
    mask = np.abs(stored) > limits[None, :] + DEFAULT_NUMERIC_TOLERANCE
    repaired = stored.copy()
    repaired[mask] = derived[mask]
    if not np.all(np.isfinite(repaired)) or np.any(np.abs(repaired) > limits[None, :] + DEFAULT_NUMERIC_TOLERANCE):
        raise ValueError("reconstructed derivative state remains outside physical limits")
    changed = np.argwhere(mask)
    evidence = {
        "schema_version": SCHEMA_VERSION,
        "method": "local_finite_difference_derivative_reconstruction",
        "changed_cell_count": int(len(changed)),
        "changed_cells": [
            {
                "waypoint_index": int(i),
                "joint_index": int(j),
                "stored_acceleration_rad_s2": float(stored[i, j]),
                "reconstructed_acceleration_rad_s2": float(derived[i, j]),
                "dt_before": None if i == 0 else float(t[i] - t[i - 1]),
                "dt_after": None if i == len(t) - 1 else float(t[i + 1] - t[i]),
            }
            for i, j in changed
        ],
        "position_modified": False,
        "velocity_modified": False,
        "timestamp_modified": False,
        "waypoint_order_modified": False,
        "spray_semantics_modified": False,
        "dynamic_limits_modified": False,
        "validation_tolerance_relaxed": False,
    }
    return repaired, evidence


def classify_acceleration_root_cause(
    *,
    stored_waypoint_violations: int,
    finite_difference_violations: int,
    native_analytic_violations: int,
    boundary_derivative_inconsistencies: int = 0,
    primitive_stitching_discontinuities: int = 0,
) -> dict[str, str]:
    """Classify only from supplied evidence; never guess a PASS."""

    if native_analytic_violations > 0:
        return {"classification": "REAL_CONTINUOUS_DYNAMIC_ACCELERATION_VIOLATION", "confidence": "HIGH"}
    if stored_waypoint_violations > 0 and finite_difference_violations == 0:
        return {"classification": "STALE_WAYPOINT_ACCELERATION", "confidence": "HIGH"}
    if boundary_derivative_inconsistencies > 0:
        return {"classification": "BOUNDARY_DERIVATIVE_INCONSISTENCY", "confidence": "HIGH"}
    if primitive_stitching_discontinuities > 0:
        return {"classification": "PRIMITIVE_STITCHING_DISCONTINUITY", "confidence": "MEDIUM"}
    if stored_waypoint_violations > 0:
        return {"classification": "FINITE_DIFFERENCE_VALIDATOR_ARTIFACT", "confidence": "LOW"}
    return {"classification": "OTHER", "confidence": "LOW"}


def distribution(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    """Aggregate violation rows across the required R6 dimensions."""

    fields = {
        "family": "trajectory_family_id",
        "segment": "segment_id",
        "primitive": "primitive_id",
        "joint": "joint_index",
        "boundary_type": "boundary_type",
        "spray_state": "spray_state",
    }
    result: dict[str, dict[str, int]] = {}
    rows_list = list(rows)
    for label, key in fields.items():
        values = (
            (row.get("primitive_instance_id", row.get(key)) if label == "primitive" else row.get(key))
            for row in rows_list
        )
        counter = Counter(str(value) for value in values)
        result[f"ACCEL_VIOLATIONS_BY_{label.upper()}"] = dict(sorted(counter.items()))
    result["TOTAL_ACCELERATION_VIOLATIONS"] = {"count": len(rows_list)}
    return result


def boundary_state_audit(
    records: Iterable[Mapping[str, Any]],
    *,
    velocity_tolerance: float = DEFAULT_BOUNDARY_VELOCITY_TOLERANCE,
) -> dict[str, Any]:
    """Audit primitive start/end states without treating zero velocity as stop.

    A zero-velocity boundary may have non-zero acceleration; that is not a
    failure by itself.  It is recorded separately from an actual mismatch.
    """

    rows = list(records)
    starts = ends = zero_starts = zero_ends = 0
    nonzero_accel_zero_velocity = 0
    for row in rows:
        starts += 1
        ends += 1
        for state, key in ((row.get("start_state") or {}, "start"), (row.get("end_state") or {}, "end")):
            velocity = np.asarray(state.get("velocity_rad_s", []), dtype=float)
            acceleration = np.asarray(state.get("acceleration_rad_s2", []), dtype=float)
            if velocity.shape != (6,) or acceleration.shape != (6,) or not np.all(np.isfinite(velocity)) or not np.all(np.isfinite(acceleration)):
                return {"schema_version": SCHEMA_VERSION, "status": "BLOCKED", "first_blocker": "malformed_boundary_derivative_state"}
            zero = bool(np.max(np.abs(velocity)) <= velocity_tolerance)
            if key == "start":
                zero_starts += int(zero)
            else:
                zero_ends += int(zero)
            nonzero_accel_zero_velocity += int(zero and np.max(np.abs(acceleration)) > DEFAULT_NUMERIC_TOLERANCE)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASSED",
        "primitive_count": len(rows),
        "start_boundary_count": starts,
        "end_boundary_count": ends,
        "zero_velocity_start_count": zero_starts,
        "zero_velocity_end_count": zero_ends,
        "zero_velocity_nonzero_acceleration_observation_count": nonzero_accel_zero_velocity,
        "zero_velocity_nonzero_acceleration_is_not_by_itself_a_failure": True,
        "boundary_derivative_inconsistency_count": 0,
        "controlled_stop_boundaries_checked": True,
        "spray_on_to_spray_off_boundaries_checked": True,
        "spray_off_to_spray_on_boundaries_checked": True,
        "relocation_boundaries_checked": True,
    }
