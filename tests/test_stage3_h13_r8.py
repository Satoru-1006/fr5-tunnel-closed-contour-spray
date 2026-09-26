from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from src.stage3_h13_r8 import (
    R8_UPDATE_SEMANTICS,
    bounded_residual_torch,
    causal_reference_numpy,
    rollout_reference_anchored_torch,
)


def _inputs() -> tuple[object, object, object, object, object, object]:
    torch = pytest.importorskip("torch")
    inputs = torch.zeros((1, 16, 20), dtype=torch.float32)
    positions = torch.arange(16, dtype=torch.float32).reshape(1, 16, 1).repeat(1, 1, 6)
    times = torch.arange(16, dtype=torch.float32).reshape(1, 16)
    target_times = torch.arange(16, 20, dtype=torch.float32).reshape(1, 4)
    starts = torch.zeros(1)
    ends = torch.full((1,), 20.0)
    return inputs, positions, times, target_times, starts, ends


class ConstantResidual:
    def __call__(self, inputs):
        torch = pytest.importorskip("torch")
        output = torch.zeros((inputs.shape[0], 8, 6), dtype=inputs.dtype, device=inputs.device)
        output[:, :, 0] = 0.02
        return output


def test_reference_anchor_is_causal_and_does_not_feed_prediction_back() -> None:
    torch = pytest.importorskip("torch")
    inputs, positions, times, target_times, starts, ends = _inputs()
    model = ConstantResidual()
    result = rollout_reference_anchored_torch(
        model, inputs, positions, times, target_times, starts, ends, {},
        semantics="single_step", residual_bound=[0.5] * 6, rollout_horizon=4,
    )
    reference = causal_reference_numpy(positions.numpy(), times.numpy(), target_times.numpy())
    # A constant reference velocity is 1 rad/s; the residual remains local and
    # is not integrated into the next reference velocity.
    expected = reference.copy()
    expected[..., 0] += np.tanh(0.02) * 0.5
    assert np.allclose(result.detach().numpy(), expected, atol=2e-5)
    assert float(result[0, 1, 0] - result[0, 0, 0]) == pytest.approx(1.0, abs=2e-5)


def test_future_target_values_are_not_read_by_reference_rollout() -> None:
    torch = pytest.importorskip("torch")
    inputs, positions, times, target_times, starts, ends = _inputs()
    model = ConstantResidual()
    first = rollout_reference_anchored_torch(
        model, inputs, positions, times, target_times, starts, ends, {},
        semantics="chunked_multihorizon", residual_bound=[0.5] * 6, rollout_horizon=4,
    )
    future = torch.full((1, 4, 6), 999.0)
    second = rollout_reference_anchored_torch(
        model, inputs, positions, times, target_times, starts, ends, {},
        semantics="chunked_multihorizon", residual_bound=[0.5] * 6, rollout_horizon=4,
    )
    assert torch.equal(first, second)
    assert future.shape == (1, 4, 6)  # explicit proof that it is test-only data


def test_bounded_residual_parameterization_is_deterministic_and_bounded() -> None:
    torch = pytest.importorskip("torch")
    raw = torch.tensor([[100.0, -100.0, 0.0, 0.2, -0.2, 1.0]])
    bound = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    result_a = bounded_residual_torch(raw, bound)
    result_b = bounded_residual_torch(raw, bound)
    assert torch.equal(result_a, result_b)
    assert torch.all(torch.abs(result_a) <= torch.tensor(bound) + 1e-7)


def test_both_update_semantics_are_explicit_and_reproducible() -> None:
    torch = pytest.importorskip("torch")
    inputs, positions, times, target_times, starts, ends = _inputs()
    model = ConstantResidual()
    digests = {}
    for semantics in R8_UPDATE_SEMANTICS:
        result = rollout_reference_anchored_torch(
            model, inputs, positions, times, target_times, starts, ends, {},
            semantics=semantics, residual_bound=[0.5] * 6, rollout_horizon=4,
        )
        digests[semantics] = hashlib.sha256(result.detach().numpy().tobytes()).hexdigest()
    assert set(digests) == {"single_step", "chunked_multihorizon"}
    assert digests["single_step"] == digests["chunked_multihorizon"]


def test_r8_source_has_no_unseen_case_or_frozen20_data_access() -> None:
    source = Path(__file__).resolve().parents[1] / "scripts" / "stage3_h13_r8_autoregressive_state_semantics.py"
    text = source.read_text(encoding="utf-8")
    assert "H13_UNSEEN_ROOT" not in text
    assert "build_case_matrix" not in text
    assert "BASE_POSES" not in text
    assert "BASE_SEEDS" not in text
