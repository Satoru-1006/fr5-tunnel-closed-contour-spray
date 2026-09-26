"""Small, backend-independent safety-bound primitives used by D54 regressions.

These functions deliberately implement fail-closed tri-state semantics.  They
are not a replacement for MoveIt2/FK/FCL execution; the C++ certifier supplies
the model-backed distances and geometry coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


class IntervalStatus(str, Enum):
    CERTIFIED_CLEAR = "CERTIFIED_CLEAR"
    COLLISION_FOUND = "COLLISION_FOUND"
    UNRESOLVED = "UNRESOLVED"


@dataclass(frozen=True)
class IntervalResult:
    status: IntervalStatus
    lower_bound: float | None
    motion_bound: float | None


def endpoint_motion_bound(velocity: float, acceleration: float, jerk_bound: float, duration: float) -> float:
    values = (velocity, acceleration, jerk_bound, duration)
    if not all(math.isfinite(value) for value in values) or duration < 0 or jerk_bound < 0:
        return math.nan
    return abs(velocity) * duration + 0.5 * abs(acceleration) * duration**2 + jerk_bound * duration**3 / 6.0


def conservative_lower_bound(start_distance: float, end_distance: float, motion_bound: float) -> float | None:
    if not all(math.isfinite(value) for value in (start_distance, end_distance, motion_bound)):
        return None
    return min(start_distance, end_distance) - max(0.0, motion_bound)


def classify_interval(
    start_distance: float,
    end_distance: float,
    motion_bound: float,
    *,
    threshold: float = 0.0,
    tolerance: float = 1.0e-9,
) -> IntervalResult:
    """Classify one interval without ever turning uncertainty into safety."""

    lower = conservative_lower_bound(start_distance, end_distance, motion_bound)
    if lower is None or not math.isfinite(threshold):
        return IntervalResult(IntervalStatus.UNRESOLVED, None, None)
    if start_distance <= threshold + tolerance or end_distance <= threshold + tolerance:
        return IntervalResult(IntervalStatus.COLLISION_FOUND, lower, motion_bound)
    if lower > threshold + tolerance:
        return IntervalResult(IntervalStatus.CERTIFIED_CLEAR, lower, motion_bound)
    return IntervalResult(IntervalStatus.UNRESOLVED, lower, motion_bound)
