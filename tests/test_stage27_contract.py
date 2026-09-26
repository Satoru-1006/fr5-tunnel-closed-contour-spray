from __future__ import annotations

import numpy as np

from src.controller_interpolation import (
    audit_intervals,
    detect_hidden_spline_overshoot,
    evaluate_coefficients,
    extrema_times,
    interpolation_order_from_fields,
    segment_coefficients,
)


def test_jtc_field_to_interpolation_order_mapping() -> None:
    assert interpolation_order_from_fields(True, False, False) == "linear"
    assert interpolation_order_from_fields(True, True, False) == "cubic"
    assert interpolation_order_from_fields(True, True, True) == "quintic"


def test_quintic_matches_all_endpoint_derivatives() -> None:
    coeff = segment_coefficients([0.2], [1.1], [0.3], [-0.4], [0.5], [-0.7], 0.8, "quintic")
    start = evaluate_coefficients(coeff, 0.0)
    end = evaluate_coefficients(coeff, 0.8)
    np.testing.assert_allclose(start["position"], [0.2], atol=1e-12)
    np.testing.assert_allclose(end["position"], [1.1], atol=1e-12)
    np.testing.assert_allclose(start["velocity"], [0.3], atol=1e-12)
    np.testing.assert_allclose(end["velocity"], [-0.4], atol=1e-12)
    np.testing.assert_allclose(start["acceleration"], [0.5], atol=1e-12)
    np.testing.assert_allclose(end["acceleration"], [-0.7], atol=1e-12)


def test_analytic_extrema_are_used_for_position_limit_gate() -> None:
    times = np.array([0.0, 1.0])
    q = np.array([[0.0], [0.0]])
    dq = np.array([[4.0], [4.0]])
    ddq = np.zeros((2, 1))
    limits = [{
        "position_lower_rad": -0.1,
        "position_upper_rad": 0.1,
        "max_velocity_rad_s": 100.0,
        "max_acceleration_rad_s2": 100.0,
        "max_jerk_rad_s3": 1000.0,
    }]
    audit = audit_intervals(times, q, dq, ddq, "quintic", limits, ["j1"])
    assert audit["status"]["position"] == "blocked"
    assert audit["maximums"]["position"]["ratio"] > 1.0
    assert audit["maximums"]["position"]["worst_time"] not in (0.0, 1.0)


def test_interpolation_positive_control_detects_hidden_overshoot() -> None:
    result = detect_hidden_spline_overshoot()
    assert result["endpoints_valid"] is True
    assert result["hidden_spline_overshoot_detected"] is True
    assert result["passed"] is True
