"""Focused causal and leakage tests for Stage 3 H11-R2."""

from __future__ import annotations

import numpy as np
import pytest

from src.stage3_h11_model import WindowArrays
from src.stage3_h11_r2_model import (
    ResidualCausalGRUTrajectoryPredictor,
    TORCH_AVAILABLE,
    causal_constant_velocity_baseline,
    compose_prediction,
    direct_metrics,
    rollout_residual_numpy,
    rollout_residual_torch,
    zero_residual_prediction,
)


def _arrays(sample_count: int = 2, horizon: int = 8) -> WindowArrays:
    times = np.arange(16 + horizon, dtype=np.float64) * 0.1
    history_times = np.repeat(times[:16][None, :], sample_count, axis=0)
    target_times = np.repeat(times[16:][None, :], sample_count, axis=0)
    base = np.stack([np.linspace(0.0, 0.2, 16 + horizon) for _ in range(6)], axis=1)
    history = np.repeat(base[:16][None, :, :], sample_count, axis=0)
    target = np.repeat(base[16:][None, :, :], sample_count, axis=0)
    inputs = np.zeros((sample_count, 16, 20), dtype=np.float32)
    inputs[:, :, :6] = history
    inputs[:, :, 6:12] = 0.125
    inputs[:, :, 18] = 1.0
    inputs[:, :, 19] = np.linspace(0.0, 1.0, 16)
    return WindowArrays(
        inputs=inputs,
        target_deltas=target - history[:, -1:, :],
        target_positions=target,
        anchor_positions=history[:, -1, :],
        target_times=target_times,
        history_positions=history,
        history_times=history_times,
        window_ids=np.arange(sample_count, dtype=np.int64),
        family_ids=np.asarray(["family_a"] * sample_count, dtype=object),
    )


def test_zero_residual_reproduces_causal_baseline_and_future_targets_are_irrelevant() -> None:
    arrays = _arrays()
    baseline = causal_constant_velocity_baseline(arrays)
    assert np.array_equal(zero_residual_prediction(arrays), baseline)
    changed = _arrays()
    changed.target_positions = changed.target_positions + 100.0
    assert np.array_equal(causal_constant_velocity_baseline(changed), baseline)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch unavailable")
def test_zero_initialized_residual_model_is_baseline_at_initialization() -> None:
    import torch

    arrays = _arrays()
    model = ResidualCausalGRUTrajectoryPredictor()
    model.eval()
    with torch.no_grad():
        residual = model(torch.from_numpy(arrays.inputs)).numpy()
    assert np.array_equal(residual, np.zeros_like(residual))
    predicted, _ = direct_metrics(model, arrays)
    assert np.array_equal(predicted, causal_constant_velocity_baseline(arrays))


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch unavailable")
def test_free_running_rollout_does_not_read_future_labels() -> None:
    arrays_a = _arrays()
    arrays_b = _arrays()
    arrays_b.target_positions = arrays_b.target_positions + 50.0
    model = ResidualCausalGRUTrajectoryPredictor()
    channels = {name: {"mean": 0.0, "scale": 1.0} for name in ("planned_joint_position_1", "planned_joint_position_2", "planned_joint_position_3", "planned_joint_position_4", "planned_joint_position_5", "planned_joint_position_6", "planned_joint_velocity_1", "planned_joint_velocity_2", "planned_joint_velocity_3", "planned_joint_velocity_4", "planned_joint_velocity_5", "planned_joint_velocity_6", "planned_joint_acceleration_1", "planned_joint_acceleration_2", "planned_joint_acceleration_3", "planned_joint_acceleration_4", "planned_joint_acceleration_5", "planned_joint_acceleration_6", "normalized_local_trajectory_time")}
    feature_names = tuple([f"planned_joint_position_{i}" for i in range(1, 7)] + [f"planned_joint_velocity_{i}" for i in range(1, 7)] + [f"planned_joint_acceleration_{i}" for i in range(1, 7)] + ["spray_on", "normalized_local_trajectory_time"])
    prediction_a = rollout_residual_numpy(model, arrays_a, channels, feature_names)
    prediction_b = rollout_residual_numpy(model, arrays_b, channels, feature_names)
    assert np.array_equal(prediction_a, prediction_b)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch unavailable")
def test_rollout_shapes_and_teacher_forcing_are_bounded() -> None:
    import torch

    arrays = _arrays()
    model = ResidualCausalGRUTrajectoryPredictor()
    channels = {name: {"mean": 0.0, "scale": 1.0} for name in ("planned_joint_position_1", "planned_joint_position_2", "planned_joint_position_3", "planned_joint_position_4", "planned_joint_position_5", "planned_joint_position_6", "planned_joint_velocity_1", "planned_joint_velocity_2", "planned_joint_velocity_3", "planned_joint_velocity_4", "planned_joint_velocity_5", "planned_joint_velocity_6", "planned_joint_acceleration_1", "planned_joint_acceleration_2", "planned_joint_acceleration_3", "planned_joint_acceleration_4", "planned_joint_acceleration_5", "planned_joint_acceleration_6", "normalized_local_trajectory_time")}
    feature_names = tuple([f"planned_joint_position_{i}" for i in range(1, 7)] + [f"planned_joint_velocity_{i}" for i in range(1, 7)] + [f"planned_joint_acceleration_{i}" for i in range(1, 7)] + ["spray_on", "normalized_local_trajectory_time"])
    with torch.no_grad():
        rolled = rollout_residual_torch(model, torch.from_numpy(arrays.inputs), torch.from_numpy(arrays.history_positions).float(), torch.from_numpy(arrays.history_times).float(), torch.from_numpy(arrays.target_times).float(), channels, feature_names, 4, 0.5, torch.from_numpy(arrays.target_positions).float(), False)
    assert tuple(rolled.shape) == (2, 4, 6)


def test_checkpoint_selection_contract_is_validation_only() -> None:
    text = open("scripts/stage3_h11_r2_low_storage_unseen_generalization.py", encoding="utf-8").read()
    assert '"selection_split": "VALIDATION"' in text
    assert '"test_or_generalization_used_for_selection": False' in text
    assert '"h13_unseen_used_for_selection": False' in text

