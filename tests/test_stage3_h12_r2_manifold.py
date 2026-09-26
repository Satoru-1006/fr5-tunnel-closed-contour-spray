from __future__ import annotations

import numpy as np
import pytest

from src.stage3_h12_r2_manifold import ALPHA_GRID, canonical_failure_category, nearest_feasible_projection, repair_window_process_manifold
from src.stage3_h12_trajectory_repair import RepairLimits


def metadata(**overrides):
    value = {
        "trajectory_family_id": "family_a",
        "trajectory_id": "trajectory_a",
        "segment_key": "family_a|trajectory_a|0|0",
        "segment_id": 0,
        "segment_order": 0,
        "primitive_id": "0",
        "spray_state": "SPRAY_ON",
        "cross_family": False,
        "cross_trajectory": False,
        "cross_segment": False,
        "cross_controlled_stop": False,
        "discontinuous": False,
    }
    value.update(overrides)
    return value


def causal_case():
    q_ref = np.zeros((8, 6), dtype=float)
    q_cv = q_ref.copy()
    raw = q_ref.copy()
    history_times = np.arange(16, dtype=float)
    target_times = np.arange(16, 24, dtype=float)
    return raw, q_cv, q_ref, history_times, target_times


def test_same_segment_enforcement_and_boundary_protection():
    raw, cv, ref, ht, tt = causal_case()
    with pytest.raises(ValueError, match="cross_segment"):
        repair_window_process_manifold(raw, cv, ref, metadata(cross_segment=True), anchor=np.zeros(6), history_times=ht, target_times=tt)
    with pytest.raises(ValueError, match="cross_controlled_stop"):
        repair_window_process_manifold(raw, cv, ref, metadata(cross_controlled_stop=True), anchor=np.zeros(6), history_times=ht, target_times=tt)


def test_raw_neural_is_first_tier_when_hard_gates_pass():
    raw, cv, ref, ht, tt = causal_case()
    raw[:, 0] = np.linspace(0.0, 0.01, 8)
    result = repair_window_process_manifold(raw, cv, ref, metadata(), anchor=np.zeros(6), history_times=ht, target_times=tt)
    assert result["repair_type"] == "RAW_NEURAL_ACCEPTED"
    assert np.array_equal(result["positions"], raw)


def test_projection_stays_on_same_ordered_reference_bank():
    raw, cv, ref, ht, tt = causal_case()
    bank = np.vstack([np.zeros((1, 6)), np.ones((1, 6)) * 0.01, np.ones((1, 6)) * 0.02])
    candidate, count, counts = nearest_feasible_projection(raw + 1.0, ref, reference_bank=bank, reference_indices=[0, 1, 2], limits=RepairLimits(), anchor=np.zeros(6), history_times=ht, target_times=tt)
    assert count == 3
    assert np.all(np.isfinite(candidate))
    assert sum(counts.values()) == 0
    assert np.all(candidate[1:] >= candidate[:-1])


def test_alpha_search_is_descending_and_cv_fallback_is_final_tier():
    assert ALPHA_GRID[0] == 1.0
    assert ALPHA_GRID[-1] == 0.0
    assert all(left >= right for left, right in zip(ALPHA_GRID, ALPHA_GRID[1:]))
    raw, cv, ref, ht, tt = causal_case()
    raw[:, 0] = 1.0e8
    result = repair_window_process_manifold(raw, cv, ref, metadata(), anchor=np.zeros(6), history_times=ht, target_times=tt)
    assert result["repair_type"] in {"MANIFOLD_PROJECTION", "RESIDUAL_SCALING", "CV_FALLBACK"}
    assert result["search_iterations"] == len(result["search_trace"])


def test_nan_inf_fails_closed():
    raw, cv, ref, ht, tt = causal_case()
    raw[0, 0] = np.nan
    result = repair_window_process_manifold(raw, cv, ref, metadata(), anchor=np.zeros(6), history_times=ht, target_times=tt)
    assert result["finite_output"]
    assert np.all(np.isfinite(result["positions"]))


def test_taxonomy_keeps_spray_process_and_state_semantics_separate():
    assert canonical_failure_category("SPRAY_PROCESS_TOLERANCE") == "SPRAY_PROCESS_TOLERANCE"
    assert canonical_failure_category("SPRAY_STATE_SEMANTICS") == "SPRAY_STATE_SEMANTICS"

