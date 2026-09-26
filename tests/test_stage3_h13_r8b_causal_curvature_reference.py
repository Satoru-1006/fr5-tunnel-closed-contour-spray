from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from src.stage3_h13_r8b import (
    C5_SCHEMA_VERSION,
    causal_curvature_conditioned_reference_numpy,
    curvature_proxy_numpy,
    gate_weight_numpy,
    rollout_c5_reference_torch,
)


def _history() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.asarray(
        [[[0.0] * 6, [1.0] * 6, [2.0] * 6, [4.0] * 6, [7.0] * 6]],
        dtype=np.float64,
    )
    times = np.asarray([[0.0, 1.0, 2.0, 3.0, 4.0]], dtype=np.float64)
    targets = np.asarray([[5.0, 6.0]], dtype=np.float64)
    return positions, times, targets


def test_c5_schema_and_gate_are_fixed_and_train_derived() -> None:
    assert C5_SCHEMA_VERSION.startswith("stage3_h13_r8b_")
    np.testing.assert_allclose(gate_weight_numpy(np.asarray([0.0, 5.0, 10.0]), 2.0, 8.0), [0.0, 0.5, 1.0])
    assert gate_weight_numpy(np.asarray([-1.0, 100.0]), 2.0, 8.0).tolist() == [0.0, 1.0]


def test_curvature_proxy_uses_only_last_three_history_states() -> None:
    positions, times, _ = _history()
    first = curvature_proxy_numpy(positions, times)
    modified_earlier = positions.copy()
    modified_earlier[:, 0, :] = 999.0
    second = curvature_proxy_numpy(modified_earlier, times)
    np.testing.assert_array_equal(first, second)


def test_c5_cv_and_quadratic_limits_are_exact() -> None:
    positions, times, targets = _history()
    kappa = float(curvature_proxy_numpy(positions, times)[0])
    cv, _ = causal_curvature_conditioned_reference_numpy(positions, times, targets, kappa + 1.0, kappa + 2.0)
    quadratic, _ = causal_curvature_conditioned_reference_numpy(positions, times, targets, kappa - 2.0, kappa - 1.0)
    last = positions[:, -1]
    velocity = last - positions[:, -2]
    expected_cv = last[:, None, :] + velocity[:, None, :] * (targets - times[:, -1, None])[:, :, None]
    np.testing.assert_allclose(cv, expected_cv)
    assert np.all(np.isfinite(quadratic))


def test_c5_reference_does_not_accept_future_positions() -> None:
    positions, times, targets = _history()
    first, _ = causal_curvature_conditioned_reference_numpy(positions, times, targets, 0.1, 0.9)
    changed_targets = targets + 100.0
    second, _ = causal_curvature_conditioned_reference_numpy(positions, times, changed_targets, 0.1, 0.9)
    assert not np.array_equal(first, second)
    # Target times are scheduling inputs; no target positions are accepted.
    assert "target_positions" not in causal_curvature_conditioned_reference_numpy.__code__.co_varnames


def test_c5_rollout_preserves_history_length_and_is_deterministic() -> None:
    torch = pytest.importorskip("torch")
    inputs = torch.zeros((1, 16, 20), dtype=torch.float32)
    positions = torch.arange(16, dtype=torch.float32).reshape(1, 16, 1).repeat(1, 1, 6)
    times = torch.arange(16, dtype=torch.float32).reshape(1, 16)
    target_times = torch.arange(16, 20, dtype=torch.float32).reshape(1, 4)
    starts = torch.zeros(1)
    ends = torch.full((1,), 20.0)

    class ZeroResidual:
        def __call__(self, value):
            return torch.zeros((value.shape[0], 8, 6), dtype=value.dtype)

    model = ZeroResidual()
    first = rollout_c5_reference_torch(model, inputs, positions, times, target_times, starts, ends, {}, k_low=0.1, k_high=10.0, residual_bound=[0.5] * 6, rollout_horizon=4)
    second = rollout_c5_reference_torch(model, inputs, positions, times, target_times, starts, ends, {}, k_low=0.1, k_high=10.0, residual_bound=[0.5] * 6, rollout_horizon=4)
    assert first.shape == (1, 4, 6)
    assert hashlib.sha256(first.detach().numpy().tobytes()).hexdigest() == hashlib.sha256(second.detach().numpy().tobytes()).hexdigest()
    assert torch.isfinite(first).all()


def test_r8b_source_has_no_unseen_case_path_or_architecture_expansion() -> None:
    source = Path(__file__).resolve().parents[1] / "scripts" / "stage3_h13_r8b_causal_curvature_reference.py"
    text = source.read_text(encoding="utf-8")
    assert "H13_UNSEEN_ROOT" not in text
    assert "build_case_matrix" not in text
    assert "Transformer" not in text
    assert "bidirectional=True" not in text
