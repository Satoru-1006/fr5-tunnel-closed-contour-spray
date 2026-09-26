"""Pure diagnostic helpers for the Stage 3 H13 R8E-C audit.

The module contains no training or model-selection code.  It keeps the exact
seed/guardrail contract and deterministic reduction/classification semantics
small enough to exercise in focused tests and fresh-process replays.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


SEEDS = (13086, 218309645, 258275761, 1638377564, 1727279519)
PASS_SEEDS = (13086, 1727279519)
FAIL_SEEDS = (218309645, 258275761, 1638377564)
H1_GUARDRAIL = 0.00008597914317990215


def seed_group(seed: int) -> str:
    value = int(seed)
    if value in PASS_SEEDS:
        return "PASS"
    if value in FAIL_SEEDS:
        return "FAIL"
    raise ValueError(f"unexpected_seed:{value}")


def guardrail_pass(h1_rmse: float) -> bool:
    """The frozen guardrail is inclusive at its exact boundary."""

    return math.isfinite(float(h1_rmse)) and float(h1_rmse) <= H1_GUARDRAIL


def distribution(values: Sequence[float] | np.ndarray) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if not array.size:
        return {key: None for key in ("mean", "median", "p25", "p75", "p90", "p95", "max")}
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def concentration_shares(squared_error: Sequence[float] | np.ndarray) -> dict[str, float]:
    values = np.asarray(squared_error, dtype=np.float64).reshape(-1)
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("invalid_squared_error")
    total = float(np.sum(values))
    ordered = np.sort(values)[::-1]
    result: dict[str, float] = {}
    for percent in (1, 5, 10):
        count = max(1, int(math.ceil(len(ordered) * percent / 100.0))) if len(ordered) else 0
        result[f"top_{percent}_percent_share"] = float(np.sum(ordered[:count]) / total) if total > 0.0 else 0.0
    return result


def aggregate_h1(errors: np.ndarray) -> dict[str, Any]:
    """Reduce a [case, output-dimension] H1 error matrix exactly."""

    array = np.asarray(errors, dtype=np.float64)
    if array.ndim != 2 or not array.size or np.any(~np.isfinite(array)):
        raise ValueError("invalid_h1_error_matrix")
    squared = np.square(array)
    total = float(np.sum(squared))
    per_case_ss = np.sum(squared, axis=1)
    per_dim_ss = np.sum(squared, axis=0)
    return {
        "rmse": float(np.sqrt(np.mean(squared))),
        "mean_absolute_error": float(np.mean(np.abs(array))),
        "signed_mean_error": float(np.mean(array)),
        "per_case_squared_error": per_case_ss,
        "per_case_relative_contribution": per_case_ss / total if total > 0.0 else np.zeros_like(per_case_ss),
        "per_dimension_rmse": np.sqrt(np.mean(squared, axis=0)),
        "per_dimension_signed_bias": np.mean(array, axis=0),
        "per_dimension_mean_absolute_error": np.mean(np.abs(array), axis=0),
        "per_dimension_squared_error_share": per_dim_ss / total if total > 0.0 else np.zeros_like(per_dim_ss),
        "concentration": concentration_shares(per_case_ss),
    }


def pearson_spearman(x: Sequence[float], y: Sequence[float]) -> dict[str, float | int]:
    left = np.asarray(x, dtype=np.float64)
    right = np.asarray(y, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1 or len(left) < 2:
        raise ValueError("invalid_correlation_inputs")
    pearson = float(np.corrcoef(left, right)[0, 1])
    left_rank = np.argsort(np.argsort(left, kind="stable"), kind="stable").astype(np.float64)
    right_rank = np.argsort(np.argsort(right, kind="stable"), kind="stable").astype(np.float64)
    spearman = float(np.corrcoef(left_rank, right_rank)[0, 1])
    return {"n": int(len(left)), "pearson_r": pearson, "spearman_rho": spearman}


def classify_root_cause(evidence: Mapping[str, Any]) -> dict[str, str]:
    """Conservative, predeclared mechanism classification.

    Correlations never select a class.  A class requires a deterministic
    cross-seed pattern plus an effect-size/materiality gate; otherwise the
    audit fails open scientifically by returning G/INCONCLUSIVE.
    """

    if bool(evidence.get("case_concentrated")):
        return {"code": "E", "name": "CASE_CONCENTRATED_FAILURE"}
    if bool(evidence.get("feature_fail_consistently_higher")) and float(evidence.get("feature_effect", 0.0)) >= 0.05:
        return {"code": "A", "name": "H1_FEATURE_OOD_DRIVEN"}
    if bool(evidence.get("early_hidden_fail_consistently_higher")) and float(evidence.get("hidden_effect", 0.0)) >= 0.05:
        return {"code": "B", "name": "HIDDEN_STATE_EARLY_SHIFT"}
    if bool(evidence.get("input_hidden_similar")) and bool(evidence.get("output_local_difference_strong")):
        return {"code": "C", "name": "OUTPUT_HEAD_LOCAL_TRADEOFF"}
    if bool(evidence.get("common_h1_h4_mechanism")) and bool(evidence.get("temporal_chain_supported")):
        return {"code": "D", "name": "H1_H4_COUPLED_TRADEOFF"}
    if int(evidence.get("supported_mechanism_count", 0)) > 1:
        return {"code": "F", "name": "MIXED_ROOT_CAUSE"}
    return {"code": "G", "name": "INCONCLUSIVE"}


def assert_validation_only_paths(paths: Sequence[str]) -> None:
    normalized = [str(path).replace("\\", "/").lower() for path in paths]
    prohibited = ("frozen20", "frozen_20", "stage3_h13_r3_frozen")
    if any(token in path for path in normalized for token in prohibited):
        raise ValueError("sealed_evaluation_path_forbidden")

