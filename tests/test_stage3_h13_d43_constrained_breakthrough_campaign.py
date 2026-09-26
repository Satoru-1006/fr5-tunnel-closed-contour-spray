from pathlib import Path

import torch

from tools.stage3_h13_d43_constrained_breakthrough_campaign import (
    ALL_NAMES,
    H32_LIMIT,
    cosine_from_flat,
    mask_map,
    retention_class,
)


def test_cosine_from_flat_handles_orthogonal_and_zero_vectors() -> None:
    assert cosine_from_flat([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert cosine_from_flat([0.0, 0.0], [1.0, 0.0]) is None


def test_mask_map_preserves_parameter_order_and_zeroes_unselected_blocks() -> None:
    values = {name: torch.ones(2) for name in ALL_NAMES}
    selected = ALL_NAMES[:2]
    masked = mask_map(values, selected)
    assert list(masked) == list(ALL_NAMES)
    assert all(torch.equal(masked[name], values[name]) for name in selected)
    assert all(torch.count_nonzero(masked[name]) == 0 for name in ALL_NAMES[2:])


def test_retention_class_separates_h32_safety_from_horizon_tradeoff() -> None:
    base = {
        "candidate_H1": 0.9,
        "parent_H1": 1.0,
        "candidate_H32": H32_LIMIT - 1.0e-6,
        "finite_state_pass": True,
        "deterministic_replay_consistency_pass": True,
        "parent_horizons": {str(h): 1.0 for h in range(1, 33)},
    }
    assert retention_class(base, {str(h): 0.99 for h in range(1, 33)}) == "RETENTION_NEUTRAL"
    assert retention_class(base, {"2": 1.01, "32": base["candidate_H32"]}) == "TRADEOFF"


def test_retention_class_marks_h32_over_limit_unsafe() -> None:
    record = {
        "candidate_H1": 0.9,
        "parent_H1": 1.0,
        "candidate_H32": H32_LIMIT + 1.0e-6,
        "finite_state_pass": True,
        "deterministic_replay_consistency_pass": True,
    }
    assert retention_class(record) == "UNSAFE"

