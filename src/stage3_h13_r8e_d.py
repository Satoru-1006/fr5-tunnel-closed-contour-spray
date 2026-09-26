"""Pure semantics and reductions for the Stage 3 H13 R8E-D audit.

This module deliberately contains no model construction, optimiser, data
loading, or checkpoint-writing code.  It makes the counterfactual intervention
and its fail-closed checks independently testable.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


SEEDS = (13086, 1727279519, 218309645, 258275761, 1638377564)
FEEDBACK_SLICE = slice(0, 18)
NON_FEEDBACK_INDICES = (18, 19)
HORIZONS = (1, 2, 3, 4)


def feature_decomposition() -> list[dict[str, Any]]:
    """Return the frozen 20-channel feature semantics used by R6/R8E."""

    names = (
        *(f"planned_joint_position_{i}" for i in range(1, 7)),
        *(f"planned_joint_velocity_{i}" for i in range(1, 7)),
        *(f"planned_joint_acceleration_{i}" for i in range(1, 7)),
        "spray_on",
        "normalized_local_trajectory_time",
    )
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        feedback = index < 18
        rows.append(
            {
                "FEATURE_NAME": name,
                "FEATURE_INDEX_OR_SLICE": index,
                "SOURCE": "candidate_or_R6_generated_feedback" if feedback else "validation_exogenous_or_temporal_context",
                "FEEDBACK_DEPENDENT": "YES" if feedback else "NO",
                "SWAPPED": "YES" if feedback else "NO",
            }
        )
    return rows


def _as_array(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim < 1 or not np.all(np.isfinite(array)):
        raise ValueError("nonfinite_or_empty_feature_array")
    return array


def verify_feature_swap(
    native_features: Any,
    counterfactual_features: Any,
    candidate_feedback: Any,
    r6_feedback: Any,
    *,
    atol: float = 0.0,
) -> dict[str, Any]:
    """Prove that a paired feature tensor changes only the intended channels."""

    native = _as_array(native_features)
    counterfactual = _as_array(counterfactual_features)
    candidate = _as_array(candidate_feedback)
    r6 = _as_array(r6_feedback)
    if native.shape != counterfactual.shape:
        raise ValueError("paired_feature_shape_mismatch")
    if candidate.shape != r6.shape or candidate.shape != native[..., :18].shape:
        raise ValueError("feedback_feature_shape_mismatch")
    changed = np.abs(counterfactual - native) > float(atol)
    non_feedback = np.asarray(NON_FEEDBACK_INDICES, dtype=np.int64)
    feedback = np.arange(18, dtype=np.int64)
    non_feedback_identical = bool(np.array_equal(native[..., non_feedback], counterfactual[..., non_feedback]))
    candidate_is_native = bool(np.allclose(native[..., feedback], candidate, rtol=0.0, atol=atol))
    r6_is_counterfactual = bool(np.allclose(counterfactual[..., feedback], r6, rtol=0.0, atol=atol))
    only_feedback_changed = bool(not np.any(changed[..., non_feedback]))
    feedback_differs = bool(np.any(np.abs(candidate - r6) > float(atol)))
    return {
        "NON_FEEDBACK_FEATURES_IDENTICAL": "YES" if non_feedback_identical else "NO",
        "FEEDBACK_FEATURES_DIFFER_AS_INTENDED": "YES" if feedback_differs and candidate_is_native and r6_is_counterfactual else "NO",
        "ONLY_FEEDBACK_FEATURES_CHANGED": "YES" if only_feedback_changed and candidate_is_native and r6_is_counterfactual else "NO",
        "changed_feedback_element_count": int(np.count_nonzero(changed[..., feedback])),
        "changed_non_feedback_element_count": int(np.count_nonzero(changed[..., non_feedback])),
        "feedback_max_absolute_difference": float(np.max(np.abs(candidate - r6))) if candidate.size else 0.0,
    }


def relative_l2(reference: np.ndarray, value: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """R8E-C teacher/free relative L2: aggregate and one value per layer."""

    ref = np.asarray(reference, dtype=np.float64)
    actual = np.asarray(value, dtype=np.float64)
    if ref.shape != actual.shape or ref.ndim != 3 or ref.shape[0] != 2:
        raise ValueError("invalid_hidden_tensor_shape")
    delta = actual - ref
    by_case = np.transpose(delta, (1, 0, 2)).reshape(delta.shape[1], -1)
    ref_by_case = np.transpose(ref, (1, 0, 2)).reshape(ref.shape[1], -1)
    aggregate = np.linalg.norm(by_case, axis=1) / np.maximum(np.linalg.norm(ref_by_case, axis=1), 1.0e-12)
    layers = [
        np.linalg.norm(delta[layer], axis=1) / np.maximum(np.linalg.norm(ref[layer], axis=1), 1.0e-12)
        for layer in range(2)
    ]
    return aggregate, layers


def rmse_from_case_cumulative(case_cumulative_rmse: np.ndarray) -> float:
    values = _as_array(case_cumulative_rmse).astype(np.float64, copy=False)
    return float(np.sqrt(np.mean(np.square(values))))


def excess_rescue_fraction(native: float, counterfactual: float, r6: float) -> float | None:
    denominator = float(native) - float(r6)
    if not all(math.isfinite(float(item)) for item in (native, counterfactual, r6)) or denominator <= 1.0e-12:
        return None
    return (float(native) - float(counterfactual)) / denominator


def paired_effect(native: float, counterfactual: float) -> dict[str, float]:
    absolute = float(counterfactual) - float(native)
    percent = None if native == 0.0 else 100.0 * absolute / float(native)
    return {"absolute_change": absolute, "percent_change": float(percent) if percent is not None else math.nan}


def classify_counterfactual(rows: Sequence[Mapping[str, Any]], semantics_proven: bool) -> dict[str, Any]:
    """Apply a predeclared conservative rule without inventing a magnitude gate.

    No R8E-D-specific materiality threshold is frozen in this repository.  A
    directionally uniform rescue is therefore reported as INCONCLUSIVE rather
    than promoted to a causal class.  A directionally uniform worsening/no
    rescue can reject feature-shift dominance, but does not prove recurrent
    sensitivity.
    """

    if not semantics_proven or len(rows) != 5:
        return {"name": "INCONCLUSIVE", "confidence": "LOW", "basis": "semantic_or_seed_integrity_not_proven"}
    deltas = [float(row["h4_hidden_counterfactual"]) - float(row["h4_hidden_native"]) for row in rows]
    if all(delta >= 0.0 for delta in deltas):
        return {"name": "FEEDBACK_FEATURE_SHIFT_NOT_SUFFICIENT", "confidence": "LOW", "basis": "all_five_non_rescue_or_worsening_without_recurrent_sensitivity_claim"}
    return {"name": "INCONCLUSIVE", "confidence": "LOW", "basis": "no_formally_applicable_materiality_rule"}
