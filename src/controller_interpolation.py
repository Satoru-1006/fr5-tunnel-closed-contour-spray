"""Exact joint-trajectory-controller spline primitives used by Stage 2.7.

The ros2_control JointTrajectoryController spline representation uses the
lowest common waypoint specification for each interval: position-only is
linear, position+velocity is cubic Hermite, and
position+velocity+acceleration is quintic.  This module deliberately keeps
the implementation independent of scipy and evaluates extrema analytically
so a dense sample grid is not mistaken for a continuous certificate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


METHOD_ORDER = {"linear": 1, "cubic": 3, "quintic": 5}


def interpolation_order_from_fields(
    position: bool, velocity: bool, acceleration: bool
) -> str:
    """Return the JTC spline order implied by complete endpoint fields."""

    if not position:
        raise ValueError("JointTrajectoryController requires positions")
    if acceleration:
        if not velocity:
            raise ValueError("Acceleration without velocity is not a quintic endpoint")
        return "quintic"
    if velocity:
        return "cubic"
    return "linear"


def _as_vector(value: Iterable[float], dof: int | None = None) -> np.ndarray:
    array = np.asarray(list(value), dtype=float)
    if array.ndim != 1:
        raise ValueError(f"Expected a one-dimensional joint vector, got {array.shape}")
    if dof is not None and len(array) != dof:
        raise ValueError(f"Expected {dof} joints, got {len(array)}")
    if not np.all(np.isfinite(array)):
        raise ValueError("Trajectory endpoint contains a non-finite value")
    return array


def segment_coefficients(
    q0: Iterable[float],
    q1: Iterable[float],
    v0: Iterable[float] | None,
    v1: Iterable[float] | None,
    a0: Iterable[float] | None,
    a1: Iterable[float] | None,
    duration_s: float,
    method: str,
) -> np.ndarray:
    """Return physical-time polynomial coefficients ``c[k, joint]``.

    The polynomial is ``q(t) = c0 + c1*t + ... + c5*t**5`` for
    ``0 <= t <= duration_s``.  Coefficients above the requested order are
    exactly zero.
    """

    if method not in METHOD_ORDER:
        raise ValueError(f"Unknown interpolation method: {method}")
    h = float(duration_s)
    if not np.isfinite(h) or h <= 0.0:
        raise ValueError(f"Duration must be positive and finite, got {duration_s!r}")
    q_start = _as_vector(q0)
    q_end = _as_vector(q1, len(q_start))
    dof = len(q_start)
    coefficients = np.zeros((6, dof), dtype=float)
    coefficients[0] = q_start

    if method == "linear":
        coefficients[1] = (q_end - q_start) / h
        return coefficients

    if v0 is None or v1 is None:
        raise ValueError(f"{method} interpolation requires endpoint velocities")
    velocity_start = _as_vector(v0, dof)
    velocity_end = _as_vector(v1, dof)
    delta = q_end - q_start
    if method == "cubic":
        coefficients[1] = velocity_start
        coefficients[2] = 3.0 * delta / h**2 - (2.0 * velocity_start + velocity_end) / h
        coefficients[3] = -2.0 * delta / h**3 + (velocity_start + velocity_end) / h**2
        return coefficients

    if a0 is None or a1 is None:
        raise ValueError("quintic interpolation requires endpoint accelerations")
    acceleration_start = _as_vector(a0, dof)
    acceleration_end = _as_vector(a1, dof)
    coefficients[1] = velocity_start
    coefficients[2] = 0.5 * acceleration_start
    residual_position = q_end - (coefficients[0] + coefficients[1] * h + coefficients[2] * h**2)
    residual_velocity = velocity_end - (coefficients[1] + 2.0 * coefficients[2] * h)
    residual_acceleration = acceleration_end - 2.0 * coefficients[2]
    coefficients[3] = (
        10.0 * residual_position / h**3
        - 4.0 * residual_velocity / h**2
        + 0.5 * residual_acceleration / h
    )
    coefficients[4] = (
        -15.0 * residual_position / h**4
        + 7.0 * residual_velocity / h**3
        - residual_acceleration / h**2
    )
    coefficients[5] = (
        6.0 * residual_position / h**5
        - 3.0 * residual_velocity / h**4
        + 0.5 * residual_acceleration / h**3
    )
    return coefficients


def evaluate_coefficients(coefficients: np.ndarray, local_time_s: float) -> dict[str, np.ndarray]:
    """Evaluate position, velocity, acceleration, jerk and snap."""

    coeff = np.asarray(coefficients, dtype=float)
    if coeff.ndim != 2 or coeff.shape[0] != 6:
        raise ValueError(f"Expected coefficients with shape (6, dof), got {coeff.shape}")
    t = float(local_time_s)
    powers = np.array([1.0, t, t**2, t**3, t**4, t**5])[:, None]
    position = np.sum(coeff * powers, axis=0)
    velocity = np.sum(coeff[1:] * np.arange(1, 6)[:, None] * powers[:-1], axis=0)
    acceleration = np.sum(
        coeff[2:] * (np.arange(2, 6) * np.arange(1, 5))[:, None] * powers[:-2], axis=0
    )
    jerk = np.sum(
        coeff[3:] * (np.arange(3, 6) * np.arange(2, 5) * np.arange(1, 4))[:, None] * powers[:-3], axis=0
    )
    snap = np.sum(
        coeff[4:] * (np.arange(4, 6) * np.arange(3, 5) * np.arange(2, 4) * np.arange(1, 3))[:, None] * powers[:-4], axis=0
    )
    return {
        "position": position,
        "velocity": velocity,
        "acceleration": acceleration,
        "jerk": jerk,
        "snap": snap,
    }


def _real_roots_in_interval(polynomial: np.ndarray, duration_s: float) -> list[float]:
    """Return finite real roots of an increasing-order polynomial in [0, h]."""

    coefficients = np.asarray(polynomial, dtype=float)
    while len(coefficients) > 1 and np.max(np.abs(coefficients[-1])) <= 1e-14:
        coefficients = coefficients[:-1]
    if len(coefficients) <= 1:
        return []
    roots = np.polynomial.polynomial.polyroots(coefficients)
    result: list[float] = []
    h = float(duration_s)
    tolerance = max(1e-12, h * 1e-10)
    for root in roots:
        if abs(float(np.imag(root))) <= tolerance:
            value = float(np.real(root))
            if -tolerance <= value <= h + tolerance:
                result.append(min(h, max(0.0, value)))
    return sorted(set(round(value, 14) for value in result))


def extrema_times(coefficients: np.ndarray, duration_s: float, quantity: str) -> list[float]:
    """Return analytic candidate extrema times for one polynomial quantity."""

    coeff = np.asarray(coefficients, dtype=float)
    if quantity == "position":
        derivative = coeff[1:] * np.arange(1, 6)[:, None]
    elif quantity == "velocity":
        derivative = coeff[2:] * np.arange(2, 6)[:, None]
    elif quantity == "acceleration":
        derivative = coeff[3:] * np.arange(3, 6)[:, None]
    elif quantity == "jerk":
        derivative = coeff[4:] * np.arange(4, 6)[:, None]
    else:
        raise ValueError(f"Unknown quantity: {quantity}")
    # Each joint can have different roots; use the union for vector auditing.
    times = {0.0, float(duration_s)}
    for joint in range(coeff.shape[1]):
        for value in _real_roots_in_interval(derivative[:, joint], duration_s):
            times.add(value)
    return sorted(times)


def max_abs_at_extrema(coefficients: np.ndarray, duration_s: float, quantity: str) -> tuple[float, int, float]:
    """Return max absolute value, joint index and local time."""

    times = extrema_times(coefficients, duration_s, quantity)
    best_value = -1.0
    best_joint = 0
    best_time = 0.0
    for time_s in times:
        values = evaluate_coefficients(coefficients, time_s)[quantity]
        joint = int(np.argmax(np.abs(values)))
        magnitude = float(abs(values[joint]))
        if magnitude > best_value:
            best_value = magnitude
            best_joint = joint
            best_time = float(time_s)
    return best_value, best_joint, best_time


def position_ratio(value: float, lower: float, upper: float) -> float:
    """Normalize a bounded revolute position against its active bound."""

    if value >= 0.0:
        return value / upper if upper > 0.0 else float("inf")
    return value / lower if lower < 0.0 else float("inf")


@dataclass(frozen=True)
class IntervalAudit:
    interval_index: int
    time_start_s: float
    time_end_s: float
    coefficients: np.ndarray
    extrema: dict[str, dict[str, Any]]


def audit_intervals(
    times_s: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray | None,
    accelerations: np.ndarray | None,
    method: str,
    limits: list[dict[str, float]],
    joint_names: list[str],
) -> dict[str, Any]:
    """Audit all intervals and return continuous analytic maxima.

    ``limits`` entries use ``position_lower_rad``, ``position_upper_rad``,
    ``max_velocity_rad_s``, ``max_acceleration_rad_s2`` and
    ``max_jerk_rad_s3``.  The function does not alter any source waypoint.
    """

    t = np.asarray(times_s, dtype=float)
    q = np.asarray(positions, dtype=float)
    if t.ndim != 1 or q.ndim != 2 or len(t) != len(q) or q.shape[1] != len(joint_names):
        raise ValueError("Trajectory arrays have inconsistent dimensions")
    if len(t) < 2 or not np.all(np.diff(t) > 0.0):
        raise ValueError("Trajectory timestamps must be strictly increasing")
    if velocities is None:
        velocity_array = None
    else:
        velocity_array = np.asarray(velocities, dtype=float)
    if accelerations is None:
        acceleration_array = None
    else:
        acceleration_array = np.asarray(accelerations, dtype=float)
    if velocity_array is not None and velocity_array.shape != q.shape:
        raise ValueError("Velocity array shape mismatch")
    if acceleration_array is not None and acceleration_array.shape != q.shape:
        raise ValueError("Acceleration array shape mismatch")

    quantity_names = ("position", "velocity", "acceleration", "jerk")
    maximums = {
        quantity: {
            "ratio": -1.0,
            "value": 0.0,
            "worst_joint": None,
            "worst_interval": None,
            "worst_time": None,
        }
        for quantity in quantity_names
    }
    interval_audits: list[IntervalAudit] = []

    for index in range(len(t) - 1):
        h = float(t[index + 1] - t[index])
        coeff = segment_coefficients(
            q[index],
            q[index + 1],
            None if velocity_array is None else velocity_array[index],
            None if velocity_array is None else velocity_array[index + 1],
            None if acceleration_array is None else acceleration_array[index],
            None if acceleration_array is None else acceleration_array[index + 1],
            h,
            method,
        )
        extrema_record: dict[str, dict[str, Any]] = {}
        for quantity in quantity_names:
            value, joint, local_time = max_abs_at_extrema(coeff, h, quantity)
            if quantity == "position":
                all_times = extrema_times(coeff, h, quantity)
                ratio = -1.0
                ratio_joint = 0
                ratio_time = 0.0
                ratio_value = 0.0
                for candidate_time in all_times:
                    values = evaluate_coefficients(coeff, candidate_time)[quantity]
                    candidate_joint = int(
                        np.argmax(
                            [
                                position_ratio(values[j], limits[j]["position_lower_rad"], limits[j]["position_upper_rad"])
                                for j in range(len(joint_names))
                            ]
                        )
                    )
                    candidate_ratio = position_ratio(
                        float(values[candidate_joint]),
                        limits[candidate_joint]["position_lower_rad"],
                        limits[candidate_joint]["position_upper_rad"],
                    )
                    if candidate_ratio > ratio:
                        ratio = float(candidate_ratio)
                        ratio_joint = candidate_joint
                        ratio_time = float(candidate_time)
                        ratio_value = float(values[candidate_joint])
            else:
                limit_key = {
                    "velocity": "max_velocity_rad_s",
                    "acceleration": "max_acceleration_rad_s2",
                    "jerk": "max_jerk_rad_s3",
                }[quantity]
                ratios = []
                all_times = extrema_times(coeff, h, quantity)
                for candidate_time in all_times:
                    values = evaluate_coefficients(coeff, candidate_time)[quantity]
                    ratios.append(np.abs(values) / np.asarray([limits[j][limit_key] for j in range(len(joint_names))]))
                ratio_matrix = np.asarray(ratios)
                flat = int(np.argmax(ratio_matrix))
                time_index, ratio_joint = np.unravel_index(flat, ratio_matrix.shape)
                ratio = float(ratio_matrix[time_index, ratio_joint])
                ratio_time = float(all_times[time_index])
                ratio_value = float(evaluate_coefficients(coeff, ratio_time)[quantity][ratio_joint])

            extrema_record[quantity] = {
                "max_abs_value": float(value),
                "ratio": float(ratio),
                "joint": joint_names[int(ratio_joint if quantity == "position" else ratio_joint)],
                "joint_index": int(ratio_joint if quantity == "position" else ratio_joint),
                "local_time_s": float(ratio_time if quantity == "position" else ratio_time),
                "global_time_s": float(t[index] + (ratio_time if quantity == "position" else ratio_time)),
                "interval_index": int(index),
            }
            if ratio > maximums[quantity]["ratio"]:
                maximums[quantity] = {
                    "ratio": float(ratio),
                    "value": float(ratio_value),
                    "worst_joint": joint_names[int(ratio_joint)],
                    "worst_interval": int(index),
                    "worst_time": float(t[index] + ratio_time),
                }
        interval_audits.append(
            IntervalAudit(index, float(t[index]), float(t[index + 1]), coeff, extrema_record)
        )

    status = {quantity: "passed" if maximums[quantity]["ratio"] <= 1.0 + 1e-12 else "blocked" for quantity in quantity_names}
    return {
        "method": method,
        "trajectory_points": int(len(t)),
        "trajectory_intervals": int(len(t) - 1),
        "all_intervals_reconstructed": len(interval_audits) == len(t) - 1,
        "maximums": maximums,
        "status": status,
        "interval_audits": interval_audits,
    }


def compare_methods(
    times_s: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    accelerations: np.ndarray,
    controller_method: str,
    samples_per_interval: int = 21,
) -> dict[str, Any]:
    """Compare controller splines with a frozen q/dq cubic reference.

    Stage 2.5 persists Ruckig samples rather than Ruckig's internal segment
    object.  The result therefore names this reference honestly as
    ``frozen_stage25_piecewise_cubic_q_dq``; it is diagnostic evidence, not a
    claim that Ruckig's hidden continuous segment representation was restored.
    """

    t = np.asarray(times_s, dtype=float)
    q = np.asarray(positions, dtype=float)
    v = np.asarray(velocities, dtype=float)
    a = np.asarray(accelerations, dtype=float)
    if samples_per_interval < 2:
        raise ValueError("At least two comparison samples per interval are required")
    maxima = {
        "position": {"value": -1.0, "worst_joint": None, "worst_interval": None, "worst_time": None},
        "velocity": {"value": -1.0, "worst_joint": None, "worst_interval": None, "worst_time": None},
        "acceleration": {"value": -1.0, "worst_joint": None, "worst_interval": None, "worst_time": None},
    }
    max_controller_vs_linear = 0.0
    for index in range(len(t) - 1):
        h = float(t[index + 1] - t[index])
        controller = segment_coefficients(q[index], q[index + 1], v[index], v[index + 1], a[index], a[index + 1], h, controller_method)
        reference = segment_coefficients(q[index], q[index + 1], v[index], v[index + 1], None, None, h, "cubic")
        for tau in np.linspace(0.0, 1.0, samples_per_interval):
            local_time = float(tau * h)
            controller_values = evaluate_coefficients(controller, local_time)
            reference_values = evaluate_coefficients(reference, local_time)
            for quantity in maxima:
                difference = np.abs(controller_values[quantity] - reference_values[quantity])
                joint = int(np.argmax(difference))
                value = float(difference[joint])
                if value > maxima[quantity]["value"]:
                    maxima[quantity] = {
                        "value": value,
                        "worst_joint": joint,
                        "worst_interval": int(index),
                        "worst_time": float(t[index] + local_time),
                    }
            linear = q[index] + tau * (q[index + 1] - q[index])
            max_controller_vs_linear = max(max_controller_vs_linear, float(np.max(np.abs(controller_values["position"] - linear))))
    return {
        "reference_method": "frozen_stage25_piecewise_cubic_q_dq",
        "reference_exact_ruckig_internal_semantics_available": False,
        "samples_per_source_interval": int(samples_per_interval),
        "max_position_deviation_rad": maxima["position"]["value"],
        "max_velocity_deviation_rad_s": maxima["velocity"]["value"],
        "max_acceleration_deviation_rad_s2": maxima["acceleration"]["value"],
        "worst_joint": maxima["position"]["worst_joint"],
        "worst_interval": maxima["position"]["worst_interval"],
        "worst_time": maxima["position"]["worst_time"],
        "quantity_worst_cases": maxima,
        "controller_vs_stage26_linear_path_max_position_deviation_rad": max_controller_vs_linear,
        "status": "diagnostic_not_formal_ruckig_equivalence",
    }


def detect_hidden_spline_overshoot() -> dict[str, Any]:
    """Run the required endpoint-valid, interior-invalid positive control."""

    q0 = np.array([0.0])
    q1 = np.array([0.0])
    v0 = np.array([4.0])
    v1 = np.array([4.0])
    a0 = np.array([0.0])
    a1 = np.array([0.0])
    coeff = segment_coefficients(q0, q1, v0, v1, a0, a1, 1.0, "quintic")
    position_times = extrema_times(coeff, 1.0, "position")
    values = [float(evaluate_coefficients(coeff, time_s)["position"][0]) for time_s in position_times]
    endpoint_valid = abs(values[0]) <= 0.1 and abs(values[-1]) <= 0.1
    interior_max = max(values)
    interior_min = min(values)
    detected = endpoint_valid and (interior_max > 0.1 or interior_min < -0.1)
    return {
        "method": "quintic",
        "endpoint_positions": [0.0, 0.0],
        "endpoint_velocity": [4.0, 4.0],
        "endpoint_acceleration": [0.0, 0.0],
        "endpoint_position_bounds_rad": [-0.1, 0.1],
        "endpoints_valid": bool(endpoint_valid),
        "analytic_position_extrema_times_s": position_times,
        "analytic_position_extrema_rad": values,
        "interior_max_rad": interior_max,
        "interior_min_rad": interior_min,
        "hidden_spline_overshoot_detected": bool(detected),
        "passed": bool(detected),
    }
