from __future__ import annotations

import numpy as np
import torch

from tools.stage3_h13_d36_system_evaluation import bootstrap_delta, matched_indices, perturb


def test_matched_indices_are_deterministic_unique_and_family_balanced() -> None:
    families = np.asarray(["a"] * 20 + ["b"] * 20 + ["c"] * 20)
    first = matched_indices(families, 30, 7)
    second = matched_indices(families, 30, 7)
    assert np.array_equal(first, second)
    assert len(np.unique(first)) == 30
    assert {family: int(np.sum(families[first] == family)) for family in "abc"} == {"a": 10, "b": 10, "c": 10}


def test_zero_perturbation_is_identity_and_nonzero_is_seeded() -> None:
    batch = {"inputs": torch.zeros(2, 3, 20), "history_positions": torch.zeros(2, 3, 6)}
    channels = {f"planned_joint_position_{i}": {"scale": 2.0} for i in range(1, 7)}
    zero = perturb(batch, channels, 0.0, 4)
    assert torch.equal(zero["inputs"], batch["inputs"])
    assert torch.equal(zero["history_positions"], batch["history_positions"])
    first = perturb(batch, channels, 0.01, 4)
    second = perturb(batch, channels, 0.01, 4)
    assert torch.equal(first["inputs"], second["inputs"])
    assert torch.equal(first["history_positions"], second["history_positions"])
    assert torch.allclose(first["inputs"][:, :, :6] * 2.0, first["history_positions"])


def test_paired_bootstrap_sign() -> None:
    left = np.asarray([1.0, 2.0, 3.0])
    right = np.asarray([2.0, 3.0, 4.0])
    result = bootstrap_delta(left, right, 9, draws=100)
    assert result["mean_delta"] == -1.0
    assert result["ci95_high"] < 0.0
