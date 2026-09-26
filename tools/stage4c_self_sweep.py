"""Backend-independent helpers for deterministic adaptive self sweeps."""

from __future__ import annotations

import math
from typing import Callable

import numpy as np


def adaptive_interpolation_fractions(q_left: np.ndarray, q_right: np.ndarray, max_step_rad: float) -> np.ndarray:
    """Return endpoint-inclusive deterministic samples for one joint segment."""
    left = np.asarray(q_left, dtype=float)
    right = np.asarray(q_right, dtype=float)
    if left.shape != right.shape or left.ndim != 1 or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("invalid_segment_state")
    if not math.isfinite(max_step_rad) or max_step_rad <= 0.0:
        raise ValueError("invalid_max_step")
    steps = max(1, int(math.ceil(float(np.max(np.abs(right - left))) / max_step_rad)))
    return np.linspace(0.0, 1.0, steps + 1, dtype=float)


def first_swept_contact(q_left: np.ndarray, q_right: np.ndarray, max_step_rad: float, collides: Callable[[np.ndarray], bool]) -> dict[str, object]:
    """Classify a swept segment using an injected collision predicate.

    The helper is intentionally named adaptive/discrete.  It does not claim a
    continuous swept-volume proof; the native worker records that limitation.
    """
    fractions = adaptive_interpolation_fractions(q_left, q_right, max_step_rad)
    left = np.asarray(q_left, dtype=float)
    right = np.asarray(q_right, dtype=float)
    for fraction in fractions:
        state = left + float(fraction) * (right - left)
        if bool(collides(state)):
            return {"collision": True, "fraction": float(fraction), "sample_count": int(len(fractions)), "state": state.tolist()}
    return {"collision": False, "fraction": None, "sample_count": int(len(fractions)), "state": None}
