"""Pure deterministic policy helpers for the additive H13-R4 repair ladder.

The native implementation supplies MoveIt FK/IK, PlanningScene collision, and
the frozen spray-process predicate.  This module only defines the bounded
search policy and compact diagnostics so the policy can be unit-tested without
ROS.  No target trajectory or unseen label is accepted by these helpers.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np


SCHEMA_VERSION = "stage3_h13_r4_reprojection_policy_v1"
TIER0 = "TIER0_H12_R7_COMPATIBLE"
TIER1 = "TIER1_ADAPTIVE_BOUNDED_LOCAL"
TIER2 = "TIER2_SEGMENT_AWARE"
TIER3 = "TIER3_BOUNDED_NEIGHBORHOOD"

# The caps are research-domain bounds, not safety tolerances.  Every native
# candidate is still checked against the frozen joint/process/collision gates.
DEFAULT_RADIUS_LADDER_RAD = (0.02, 0.03, 0.04, 0.06, 0.08)
DEFAULT_CONTEXT_WINDOW = 2
DEFAULT_MAX_SEGMENT_WAYPOINTS = 64


def contiguous_ranges(indices: Iterable[int]) -> list[tuple[int, int]]:
    """Return sorted inclusive ranges for a deterministic set of indices."""

    values = sorted({int(value) for value in indices})
    if not values:
        return []
    ranges: list[tuple[int, int]] = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append((start, previous))
        start = previous = value
    ranges.append((start, previous))
    return ranges


def expand_range(bounds: tuple[int, int], length: int, context: int = DEFAULT_CONTEXT_WINDOW) -> tuple[int, int]:
    """Add a bounded context window without changing the offending range."""

    if length <= 0:
        raise ValueError("trajectory_length_must_be_positive")
    left, right = bounds
    if left < 0 or right < left or right >= length:
        raise ValueError("range_out_of_bounds")
    width = max(0, int(context))
    return max(0, left - width), min(length - 1, right + width)


def radius_ladder(
    base_radius_rad: float = DEFAULT_RADIUS_LADDER_RAD[0],
    *,
    factors: Sequence[float] = (1.0, 1.5, 2.0, 3.0, 4.0),
    maximum_rad: float = DEFAULT_RADIUS_LADDER_RAD[-1],
) -> list[float]:
    """Build a sorted, duplicate-free, bounded correction-radius ladder."""

    base = float(base_radius_rad)
    maximum = float(maximum_rad)
    if not np.isfinite(base) or not np.isfinite(maximum) or base <= 0.0 or maximum < base:
        raise ValueError("invalid_radius_bounds")
    values = sorted({round(min(maximum, base * float(factor)), 12) for factor in factors if float(factor) > 0.0})
    if not values or values[-1] < maximum:
        values.append(round(maximum, 12))
    return values


def affected_fraction(affected_count: int, trajectory_length: int) -> float:
    if trajectory_length <= 0:
        return 0.0
    return float(affected_count) / float(trajectory_length)


def signed_limit_correction(
    positions: np.ndarray,
    lower: Sequence[float],
    upper: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-waypoint nearest-bound correction and signed overflow."""

    q = np.asarray(positions, dtype=np.float64)
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != len(lo) or len(lo) != len(hi):
        raise ValueError("position_limit_shape_mismatch")
    signed = np.where(q < lo[None, :], lo[None, :] - q, np.where(q > hi[None, :], hi[None, :] - q, 0.0))
    norms = np.linalg.norm(signed, axis=1)
    return norms, signed


def violation_profile(
    positions: np.ndarray,
    lower: Sequence[float],
    upper: Sequence[float],
) -> dict[str, Any]:
    norms, signed = signed_limit_correction(positions, lower, upper)
    indices = np.flatnonzero(norms > 0.0).astype(int).tolist()
    values = norms[norms > 0.0]
    return {
        "violating_waypoint_count": len(indices),
        "violating_indices": indices,
        "contiguous_segments": [list(item) for item in contiguous_ranges(indices)],
        "maximum_joint_space_displacement_required_rad": float(np.max(values)) if values.size else 0.0,
        "rms_joint_space_displacement_required_rad": float(np.sqrt(np.mean(np.square(values)))) if values.size else 0.0,
        "maximum_position_limit_violation_rad": float(np.max(np.abs(signed))) if signed.size else 0.0,
        "signed_overflow_by_joint_rad": np.sum(signed, axis=0).tolist(),
    }


def classify_failure(profile: Mapping[str, Any]) -> str:
    """Classify a case using only compact pre-repair evidence."""

    position_count = int(profile.get("position_violating_waypoint_count", 0) or 0)
    spray_count = int(profile.get("spray_process_violating_waypoint_count", 0) or 0)
    max_segment = max((int(item[1]) - int(item[0]) + 1 for item in profile.get("contiguous_violating_segments", [])), default=0)
    max_required = float(profile.get("maximum_joint_space_displacement_required_rad", 0.0) or 0.0)
    global_bias = bool(profile.get("global_bias_evidence"))
    if max_required > DEFAULT_RADIUS_LADDER_RAD[-1] and position_count:
        return "G.genuinely_out_of_domain"
    if position_count and spray_count:
        return "F.mixed_position_spray_violation"
    if spray_count:
        return "E.spray_geometry_mismatch"
    if global_bias:
        return "D.systematic_joint_bias"
    if max_segment >= 5:
        return "B.contiguous_segment_offset"
    if max_segment > 1:
        return "A.small_local_offset"
    return "G.genuinely_out_of_domain"


def choose_tier(attempts: Sequence[Mapping[str, Any]]) -> str | None:
    """Choose the first accepted tier in fixed priority order."""

    order = (TIER0, TIER1, TIER2, TIER3)
    for tier in order:
        if any(str(row.get("tier")) == tier and row.get("accepted") is True for row in attempts):
            return tier
    return None


def correction_summary(original: np.ndarray, repaired: np.ndarray) -> dict[str, Any]:
    before = np.asarray(original, dtype=np.float64)
    after = np.asarray(repaired, dtype=np.float64)
    if before.shape != after.shape:
        raise ValueError("correction_shape_mismatch")
    delta = np.linalg.norm(after - before, axis=1)
    affected = np.flatnonzero(delta > 1.0e-12).astype(int).tolist()
    nonzero = delta[delta > 1.0e-12]
    return {
        "max_joint_correction_rad": float(np.max(nonzero)) if nonzero.size else 0.0,
        "rms_joint_correction_rad": float(np.sqrt(np.mean(np.square(nonzero)))) if nonzero.size else 0.0,
        "affected_waypoint_count": len(affected),
        "affected_waypoint_indices": affected,
        "affected_trajectory_fraction": affected_fraction(len(affected), len(delta)),
    }


def no_label_repair_audit() -> dict[str, Any]:
    return {"UNSEEN_LABEL_USED_FOR_REPAIR": "NO", "FUTURE_LABEL_LEAKAGE": 0}

