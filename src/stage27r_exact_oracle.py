"""Certification-only copy of the JTC 4.40.1 quintic arithmetic.

The expressions and evaluation order below mirror trajectory.cpp lines
338-380.  It is deliberately independent of the live controller object and
does not import or mutate any ROS trajectory instance.
"""

from __future__ import annotations

import numpy as np


def segment_coefficients_source(q0, q1, v0, v1, a0, a1, duration_s: float) -> np.ndarray:
    h = float(duration_s)
    t1 = h
    t2 = t1 * t1
    t3 = t2 * t1
    t4 = t3 * t1
    t5 = t4 * t1
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    v1 = np.asarray(v1, dtype=float)
    a0 = np.asarray(a0, dtype=float)
    a1 = np.asarray(a1, dtype=float)
    coefficients = np.zeros((6, len(q0)), dtype=float)
    for i in range(len(q0)):
        start_pos = float(q0[i])
        start_vel = float(v0[i])
        start_acc = float(a0[i])
        end_pos = float(q1[i])
        end_vel = float(v1[i])
        end_acc = float(a1[i])
        coefficients[0, i] = start_pos
        coefficients[1, i] = start_vel
        coefficients[2, i] = 0.5 * start_acc
        coefficients[3, i] = (-20.0 * start_pos + 20.0 * end_pos - 3.0 * start_acc * t2 + end_acc * t2 - 12.0 * start_vel * t1 - 8.0 * end_vel * t1) / (2.0 * t3)
        coefficients[4, i] = (30.0 * start_pos - 30.0 * end_pos + 3.0 * start_acc * t2 - 2.0 * end_acc * t2 + 16.0 * start_vel * t1 + 14.0 * end_vel * t1) / (2.0 * t4)
        coefficients[5, i] = (-12.0 * start_pos + 12.0 * end_pos - start_acc * t2 + end_acc * t2 - 6.0 * start_vel * t1 - 6.0 * end_vel * t1) / (2.0 * t5)
    return coefficients


def evaluate_source(coefficients: np.ndarray, local_time_s: float) -> dict[str, np.ndarray]:
    coeff = np.asarray(coefficients, dtype=float)
    x = float(local_time_s)
    powers = [1.0]
    for _ in range(5):
        powers.append(powers[-1] * x)
    p = np.zeros(coeff.shape[1], dtype=float)
    v = np.zeros(coeff.shape[1], dtype=float)
    a = np.zeros(coeff.shape[1], dtype=float)
    for i in range(coeff.shape[1]):
        p[i] = powers[0] * coeff[0, i] + powers[1] * coeff[1, i] + powers[2] * coeff[2, i] + powers[3] * coeff[3, i] + powers[4] * coeff[4, i] + powers[5] * coeff[5, i]
        v[i] = powers[0] * coeff[1, i] + powers[1] * 2.0 * coeff[2, i] + powers[2] * 3.0 * coeff[3, i] + powers[3] * 4.0 * coeff[4, i] + powers[4] * 5.0 * coeff[5, i]
        a[i] = powers[0] * 2.0 * coeff[2, i] + powers[1] * 6.0 * coeff[3, i] + powers[2] * 12.0 * coeff[4, i] + powers[3] * 20.0 * coeff[5, i]
    j = powers[0] * 6.0 * coeff[3] + powers[1] * 24.0 * coeff[4] + powers[2] * 60.0 * coeff[5]
    return {"position": p, "velocity": v, "acceleration": a, "jerk": j}
