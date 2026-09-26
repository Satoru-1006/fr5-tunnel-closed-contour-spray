from __future__ import annotations

import numpy as np
import pytest

from src.stage3_h13_r8e_a import (
    distribution_summary,
    first_sustained_threshold,
    instrumented_forward,
    normalized_hidden_metrics,
)
from src.stage3_h11_r2_model import ResidualCausalGRUTrajectoryPredictor


def test_instrumentation_does_not_change_outputs_or_parameters() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(41)
    model = ResidualCausalGRUTrajectoryPredictor(input_size=20, hidden_size=16, num_layers=2, horizon=8).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    inputs = torch.randn(7, 8, 20)
    expected = model(inputs)
    observed = instrumented_forward(model, inputs)
    assert torch.allclose(observed["output"], expected, rtol=1.0e-6, atol=1.0e-7)
    assert observed["hidden"].shape == (2, 7, 16)
    assert all(torch.equal(model.state_dict()[name], value) for name, value in before.items())
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in model.parameters())


def test_hidden_metrics_and_deterministic_aggregation() -> None:
    teacher = np.asarray([[[1.0, 0.0]], [[0.0, 2.0]]])
    free = np.asarray([[[1.0, 0.0]], [[0.0, 3.0]]])
    values = normalized_hidden_metrics(teacher, free)
    assert values["l2"].tolist() == [[0.0], [1.0]]
    assert values["relative_l2"].tolist() == [[0.0], [0.5]]
    assert distribution_summary(values["l2"])["median"] == 0.5
    assert distribution_summary(values["l2"]) == distribution_summary(values["l2"])


def test_sustained_threshold_is_fail_closed() -> None:
    profile = {"1": {"median": 0.0}, "2": {"median": 0.02}, "3": {"median": 0.03}}
    assert first_sustained_threshold(profile, "median", 0.01) == 2
    assert first_sustained_threshold(profile, "median", 0.04) is None
