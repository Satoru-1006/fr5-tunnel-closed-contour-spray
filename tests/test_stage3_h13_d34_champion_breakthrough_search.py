from __future__ import annotations

import torch

from tools import stage3_h13_d34_champion_breakthrough_search as d34


def maps(left: tuple[float, ...], right: tuple[float, ...]):
    values = {name: torch.zeros(2) for name in d34.d32.d24.d16.ALL_NAMES}
    values["gru.weight_ih_l0"] = torch.tensor(left)
    values["head.0.weight"] = torch.tensor(right)
    return values


def test_blockwise_pcgrad_removes_each_conflict() -> None:
    objective = maps((1.0, 0.0), (0.0, 1.0))
    constraint = maps((-1.0, 0.0), (0.0, -1.0))
    projected, metrics = d34.blockwise_pcgrad(objective, constraint)
    assert metrics["block_conflicts"] == 2
    assert d34.d32.dot(projected, constraint) >= -1.0e-7


def test_blockwise_tangent_is_constraint_orthogonal() -> None:
    objective = maps((1.0, 2.0), (3.0, 4.0))
    constraint = maps((2.0, -1.0), (4.0, -3.0))
    projected, _ = d34.blockwise_tangent(objective, constraint)
    for block in d34._parameter_blocks(objective):
        left = d34._masked(projected, block)
        right = d34._masked(constraint, block)
        assert abs(d34.d32.dot(left, right)) < 1.0e-6


def test_mgda_respects_h1_priority_floor() -> None:
    objective = maps((1.0, 0.0), (0.0, 0.0))
    constraint = maps((0.0, 1.0), (0.0, 0.0))
    _, metrics = d34.mgda_direction(objective, constraint, 0.9)
    assert metrics["mgda_h1_weight"] >= 0.9


def test_first_wave_has_distinct_method_families() -> None:
    families = {spec["family"] for spec in d34.first_wave_specs()}
    assert "D34_BLOCKWISE_PCGRAD" in families
    assert "D34_BLOCKWISE_TANGENT" in families
    assert "D34_MGDA_H1_PRIORITY" in families
    assert "D34_MOMENT_TARGET_H1" in families
    assert len(families) >= 8
