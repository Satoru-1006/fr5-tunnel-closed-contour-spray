"""Small, dependency-free invariants used by the D51 shadow tests.

The production certificate lives in the native MoveIt2/FCL executable.  These
helpers intentionally model only the policy and fail-closed math invariants so
they can be exercised without ROS.
"""

from __future__ import annotations

from typing import Iterable


def conservative_motion_bound(q0: float, q1: float, v0: float, v1: float,
                              a0: float, a1: float, duration: float,
                              jerk_bound: float) -> float:
    """A closed-form scalar upper bound on displacement over one interval."""
    if duration <= 0.0 or jerk_bound < 0.0:
        raise ValueError("invalid_interval")
    endpoint = abs(q1 - q0)
    forward = abs(v0) * duration + 0.5 * abs(a0) * duration ** 2 + jerk_bound * duration ** 3 / 6.0
    backward = abs(v1) * duration + 0.5 * abs(a1) * duration ** 2 + jerk_bound * duration ** 3 / 6.0
    return max(endpoint, forward, backward)


def conservative_relative_motion_bound(left: Iterable[float], right: Iterable[float]) -> float:
    """Bound relative motion from independent scalar body bounds."""
    left_bound = max(abs(float(value)) for value in left)
    right_bound = max(abs(float(value)) for value in right)
    return left_bound + right_bound


def certified_lower_bound(static_distance: float, motion_bound: float, threshold: float = 0.0) -> float:
    """Return the conservative lower bound; non-finite inputs fail closed."""
    import math
    if not all(math.isfinite(float(value)) for value in (static_distance, motion_bound, threshold)):
        return float("-inf")
    return float(static_distance - motion_bound)


def passes_certificate(static_distance: float, motion_bound: float, threshold: float = 0.0) -> bool:
    """A tangent/penetrating/nonpositive lower bound is never certified."""
    return certified_lower_bound(static_distance, motion_bound, threshold) > float(threshold)


def candidate_promotable(*, all_hard_gates: bool, candidate_clearance: float,
                         protected_clearance: float, clearance_band: float,
                         candidate_torque_slew: float, protected_torque_slew: float,
                         torque_band: float) -> bool:
    """Policy-only promotion veto used by D51 tests and report generation."""
    if not all_hard_gates:
        return False
    if candidate_clearance < protected_clearance - clearance_band:
        return False
    if candidate_torque_slew > protected_torque_slew * (1.0 + torque_band):
        return False
    return True
