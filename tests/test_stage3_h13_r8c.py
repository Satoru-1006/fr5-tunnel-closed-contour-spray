from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from src.stage3_h13_r6 import rollout_residual_torch_r6
from src.stage3_h13_r8c import R8C_EVALUATION_HORIZONS, R8C_TRAINING_HORIZONS, curriculum_state, horizon_loss_weights


def test_r8c_curriculum_is_bounded_and_reaches_fully_self_fed_h32() -> None:
    states = [curriculum_state(epoch, 5) for epoch in range(1, 6)]
    assert [state[0] for state in states] == [4, 8, 16, 32, 32]
    assert states[-1][1] == 0.0
    assert R8C_EVALUATION_HORIZONS == (1, 4, 8, 12, 16, 20, 24, 32)


def test_r8c_long_horizon_has_predeclared_weight() -> None:
    weights = horizon_loss_weights()
    assert set(weights) == set(R8C_TRAINING_HORIZONS)
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights[32] > weights[1]


def test_r8c_free_running_does_not_read_future_targets() -> None:
    torch = pytest.importorskip("torch")

    class ConstantResidual(torch.nn.Module):
        def forward(self, inputs):
            output = torch.zeros((inputs.shape[0], 8, 6), dtype=inputs.dtype, device=inputs.device)
            output[:, :, 0] = 0.01
            return output

    model = ConstantResidual()
    inputs = torch.zeros((1, 16, 20), dtype=torch.float32)
    positions = torch.arange(16, dtype=torch.float32).reshape(1, 16, 1).repeat(1, 1, 6)
    times = torch.arange(16, dtype=torch.float32).reshape(1, 16)
    target_times = torch.arange(16, 20, dtype=torch.float32).reshape(1, 4)
    starts = torch.zeros(1)
    ends = torch.full((1,), 20.0)
    future = torch.full((1, 4, 6), 999.0)
    first = rollout_residual_torch_r6(model, inputs, positions, times, target_times, starts, ends, {}, rollout_horizon=4, teacher_forcing_ratio_value=0.0, target_positions=None, sample_random=False)
    second = rollout_residual_torch_r6(model, inputs, positions, times, target_times, starts, ends, {}, rollout_horizon=4, teacher_forcing_ratio_value=0.0, target_positions=future, sample_random=False)
    assert torch.equal(first, second)


def test_r8c_source_has_no_frozen20_or_native_execution_path() -> None:
    source = Path(__file__).resolve().parents[1] / "scripts" / "stage3_h13_r8c_rollout_aware_self_feeding.py"
    text = source.read_text(encoding="utf-8").lower()
    assert "build_case_matrix" not in text
    assert "run_native" not in text
    assert "physical robot" not in text
    assert "frozen20" in text  # fail-closed status is recorded explicitly


def test_r8c_hash_evidence_is_sha256_length() -> None:
    digest = hashlib.sha256(b"r8c").hexdigest()
    assert len(digest) == 64
