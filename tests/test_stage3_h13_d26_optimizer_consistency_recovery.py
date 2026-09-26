"""Focused D26 AdamW consistency regression tests."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import stage3_h13_d26_optimizer_consistency as consistency


def _run(alpha: float, corrupt: bool = False) -> dict:
    torch.manual_seed(20260822)
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, betas=consistency.BETAS, eps=consistency.EPS, weight_decay=consistency.WEIGHT_DECAY)
    consistency.bind_parameter_names(model, optimizer)
    before_parameters = consistency.snapshot_parameters(model)
    before_optimizer = consistency.snapshot_optimizer_state(optimizer)
    gradients = {}
    for index, (name, parameter) in enumerate(model.named_parameters(), start=1):
        gradient = torch.full_like(parameter, 0.01 * index)
        parameter.grad = gradient
        gradients[name] = gradient.detach().clone()
    optimizer.param_groups[0]["lr"] = 0.01 * alpha
    optimizer.step()
    if corrupt:
        with torch.no_grad():
            next(model.parameters()).add_(0.001)
    after_parameters = consistency.snapshot_parameters(model)
    after_optimizer = consistency.snapshot_optimizer_state(optimizer)
    summary, _ = consistency.validate_adamw_transition(
        model,
        optimizer,
        before_parameters,
        before_optimizer,
        gradients,
        after_parameters,
        after_optimizer,
        assumed_effective_lrs=[0.01 * alpha],
    )
    return summary


def test_alpha_one_validates_with_actual_effective_lr() -> None:
    result = _run(1.0)
    assert result["optimizer_consistency_pass"] is True
    assert result["failure_subconditions"] == []


def test_alpha_one_thirty_second_validates_with_actual_effective_lr() -> None:
    result = _run(1.0 / 32.0)
    assert result["optimizer_consistency_pass"] is True
    assert result["actual_candidate_effective_lr"] == [0.01 / 32.0]
    assert result["adamw_reconstruction_assumed_lr"] == [0.01 / 32.0]


def test_intentionally_corrupted_transition_fails_with_specific_subcondition() -> None:
    result = _run(1.0 / 32.0, corrupt=True)
    assert result["optimizer_consistency_pass"] is False
    assert "parameter_delta_identity" in result["failure_subconditions"]
    assert "total_reconstructed_update_identity" in result["failure_subconditions"]

