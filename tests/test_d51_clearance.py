from __future__ import annotations

import math

from tools.d51_clearance_math import (
    candidate_promotable,
    certified_lower_bound,
    conservative_motion_bound,
    conservative_relative_motion_bound,
    passes_certificate,
)


def test_known_positive_static_fixture():
    assert certified_lower_bound(0.25, 0.0) == 0.25
    assert passes_certificate(0.25, 0.0)


def test_known_positive_moving_fixture():
    motion = conservative_motion_bound(0.0, 0.1, 0.1, 0.1, 0.0, 0.0, 1.0, 0.0)
    assert motion >= 0.1
    assert passes_certificate(0.25, motion)


def test_near_grazing_is_conservative_and_tangent_fails_closed():
    assert certified_lower_bound(0.100001, 0.1) > 0.0
    assert passes_certificate(0.100001, 0.1)
    assert certified_lower_bound(0.1, 0.1) == 0.0
    assert not passes_certificate(0.1, 0.1)


def test_hidden_between_safe_endpoints_is_not_a_certificate():
    # Both endpoint samples are 0.1 m from an obstacle, but the constructed
    # interior excursion reaches 0.2 m of motion toward it.
    lower = certified_lower_bound(0.1, 0.2)
    assert lower <= 0.0
    assert not passes_certificate(0.1, 0.2)


def test_two_body_relative_motion_adds_bounds():
    assert conservative_relative_motion_bound((0.02, 0.03), (0.01, 0.04)) == 0.07


def test_nonfinite_backend_result_fails_closed():
    assert math.isinf(certified_lower_bound(float("nan"), 0.0))
    assert not passes_certificate(float("nan"), 0.0)


def test_promotion_vetoes_hard_failure_clearance_and_torque_regression():
    kwargs = dict(
        all_hard_gates=True,
        candidate_clearance=1.0,
        protected_clearance=1.0,
        clearance_band=1.0e-9,
        candidate_torque_slew=10.0,
        protected_torque_slew=10.0,
        torque_band=0.01,
    )
    assert candidate_promotable(**kwargs)
    assert not candidate_promotable(**{**kwargs, "all_hard_gates": False})
    assert not candidate_promotable(**{**kwargs, "candidate_clearance": 0.9})
    assert not candidate_promotable(**{**kwargs, "candidate_torque_slew": 10.2})
