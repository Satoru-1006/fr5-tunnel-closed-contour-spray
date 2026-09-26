from __future__ import annotations

import math

import numpy as np

from tools.stage4a_system_benchmark import finite_derivatives, stats, wave


def test_stats_known_answer_and_tail_fields() -> None:
    result = stats([1.0, 2.0, 3.0, 4.0], "m")
    assert result["count"] == 4
    assert result["median"] == 2.5
    assert result["max"] == 4.0
    assert "statistical_note" in result


def test_wave_zero_endpoints_and_finite() -> None:
    values = wave(181, 3.0, 0.2)
    assert values.shape == (181,)
    assert math.isclose(float(values[0]), 0.0, abs_tol=1e-15)
    assert math.isclose(float(values[-1]), 0.0, abs_tol=1e-15)
    assert np.isfinite(values).all()


def test_derivatives_known_quadratic() -> None:
    t = np.linspace(0.0, 1.0, 101)
    q = (t * t)[:, None] * np.ones((1, 6))
    velocity, acceleration, jerk = finite_derivatives(q, t)
    assert np.max(np.abs(velocity[2:-2, 0] - 2.0 * t[2:-2])) < 1e-10
    assert np.max(np.abs(acceleration[2:-2, 0] - 2.0)) < 1e-10
    assert np.isfinite(jerk).all()
