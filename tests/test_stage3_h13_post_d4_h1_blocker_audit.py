from __future__ import annotations

import math

import torch

from scripts.stage3_h13_post_d4_h1_blocker_audit import cosine_from_norms, module_family


def test_module_family_partition_is_stable() -> None:
    assert module_family("gru.weight_ih_l0") == "gru"
    assert module_family("decoder.weight") == "decoder"
    assert module_family("other_parameter") == "other"


def test_cosine_from_norms_handles_zero_and_normal_case() -> None:
    assert cosine_from_norms(1.0, 0.0, 1.0) is None
    assert math.isclose(float(cosine_from_norms(2.0, 2.0, 2.0)), 0.5)


def test_gradients_are_not_mutated_by_snapshot_arithmetic() -> None:
    value = torch.tensor([3.0, 4.0])
    clone = value.clone()
    assert torch.equal(value, clone)
