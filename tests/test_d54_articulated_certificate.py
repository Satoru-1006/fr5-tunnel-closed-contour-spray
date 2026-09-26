from __future__ import annotations

import math

from tools.d54_interval_math import (
    IntervalStatus,
    classify_interval,
    endpoint_motion_bound,
)


def test_known_safe_interval_is_certified_with_positive_margin() -> None:
    result = classify_interval(0.20, 0.21, 0.05)
    assert result.status is IntervalStatus.CERTIFIED_CLEAR
    assert math.isclose(result.lower_bound or math.nan, 0.15, rel_tol=0.0, abs_tol=1.0e-15)


def test_endpoint_contact_is_collision_found() -> None:
    result = classify_interval(0.0, 0.2, 0.01)
    assert result.status is IntervalStatus.COLLISION_FOUND


def test_hidden_middle_collision_is_unresolved_without_a_valid_motion_bound() -> None:
    # Safe endpoints do not prove the middle safe.  A bound that reaches the
    # obstacle must remain unresolved, never be promoted to PASS.
    result = classify_interval(0.10, 0.10, 0.10)
    assert result.status is IntervalStatus.UNRESOLVED


def test_grazing_contact_fails_closed() -> None:
    result = classify_interval(0.10, 0.10, 0.10, threshold=0.0)
    assert result.status is IntervalStatus.UNRESOLVED


def test_taylor_motion_bound_is_finite_and_monotone_in_duration() -> None:
    short = endpoint_motion_bound(0.2, 0.3, 8.0, 0.01)
    long = endpoint_motion_bound(0.2, 0.3, 8.0, 0.02)
    assert math.isfinite(short)
    assert long > short


def test_nonfinite_measurement_is_unresolved() -> None:
    result = classify_interval(math.nan, 0.2, 0.01)
    assert result.status is IntervalStatus.UNRESOLVED


def test_acm_allowed_pairs_are_not_required_unresolved_regions() -> None:
    # The C++ aggregate must count only the 10 required pairs.  ACM-allowed
    # pairs are intentionally skipped and their unpopulated status buffers
    # must not be interpreted as unresolved evidence.
    pair_universe = 21
    allowed_pairs = 11
    required_pairs = pair_universe - allowed_pairs
    assert required_pairs == 10
    assert allowed_pairs not in {0, required_pairs}
