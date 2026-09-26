"""Semantic tests for adaptive self-sweep sampling."""

import numpy as np

from tools.stage4c_self_sweep import adaptive_interpolation_fractions, first_swept_contact


def test_endpoint_safe_mid_interval_contact_is_sampled() -> None:
    left = np.zeros(6)
    right = np.ones(6)

    result = first_swept_contact(left, right, max_step_rad=0.6, collides=lambda q: np.isclose(q[0], 0.5))

    assert result["collision"] is True
    assert result["fraction"] == 0.5
    assert result["sample_count"] == 3


def test_adaptive_fractions_are_endpoint_inclusive_and_deterministic() -> None:
    left = np.zeros(6)
    right = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    first = adaptive_interpolation_fractions(left, right, max_step_rad=0.25)
    second = adaptive_interpolation_fractions(left, right, max_step_rad=0.25)

    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first[[0, -1]], [0.0, 1.0])
    assert len(first) == 5


def test_safe_sweep_does_not_invent_contact() -> None:
    result = first_swept_contact(np.zeros(6), np.ones(6), max_step_rad=0.5, collides=lambda _: False)

    assert result["collision"] is False
    assert result["fraction"] is None
    assert result["state"] is None
