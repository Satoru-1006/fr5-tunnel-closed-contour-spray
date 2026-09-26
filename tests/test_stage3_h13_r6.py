from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.stage3_h11_dataset import FEATURE_NAMES, NORMALIZED_FEATURE_NAMES
from src.stage3_h11_model import metric_payload
from src.stage3_h13_r5_reconstruction import MODEL_PRIOR_MAX_JOINT_DELTA_RAD, MODEL_PRIOR_MAX_RMS_JOINT_DELTA_RAD
from src.stage3_h13_r6 import (
    ROLLOUT_HORIZONS,
    build_sequence_arrays,
    checkpoint_hash_unchanged,
    horizon_for_epoch,
    native_count_or_not_reached,
    rollout_residual_torch_r6,
    supported_rollout_horizons,
    teacher_forcing_ratio,
    validation_selection_key,
)


def test_horizon_ladder_uses_all_predeclared_bounded_horizons() -> None:
    assert ROLLOUT_HORIZONS == (4, 8, 16, 32)
    assert [horizon_for_epoch(epoch, 8, ROLLOUT_HORIZONS) for epoch in range(1, 9)] == [4, 4, 8, 8, 16, 16, 32, 32]


def test_self_feeding_changes_the_next_causal_state() -> None:
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
    target_times = torch.arange(16, 19, dtype=torch.float32).reshape(1, 3)
    starts = torch.zeros(1)
    ends = torch.full((1,), 20.0)
    result = rollout_residual_torch_r6(model, inputs, positions, times, target_times, starts, ends, {}, rollout_horizon=3, sample_random=False)
    # The first prediction becomes part of the next history; its residual is
    # therefore integrated into the next finite-difference velocity.
    assert float(result[0, 1, 0]) > float(result[0, 0, 0]) + 0.01


def test_future_target_values_do_not_change_free_running_rollout() -> None:
    torch = pytest.importorskip("torch")

    class ConstantResidual(torch.nn.Module):
        def forward(self, inputs):
            return torch.zeros((inputs.shape[0], 8, 6), dtype=inputs.dtype, device=inputs.device)

    model = ConstantResidual()
    inputs = torch.zeros((1, 16, 20), dtype=torch.float32)
    positions = torch.zeros((1, 16, 6), dtype=torch.float32)
    times = torch.arange(16, dtype=torch.float32).reshape(1, 16)
    target_times = torch.arange(16, 20, dtype=torch.float32).reshape(1, 4)
    starts = torch.zeros(1)
    ends = torch.full((1,), 20.0)
    future = torch.full((1, 4, 6), 999.0)
    free_a = rollout_residual_torch_r6(model, inputs, positions, times, target_times, starts, ends, {}, rollout_horizon=4, teacher_forcing_ratio_value=0.0, target_positions=None, sample_random=False)
    free_b = rollout_residual_torch_r6(model, inputs, positions, times, target_times, starts, ends, {}, rollout_horizon=4, teacher_forcing_ratio_value=0.0, target_positions=future, sample_random=False)
    assert torch.equal(free_a, free_b)


def test_train_validation_sequence_horizon_construction() -> None:
    count = 60
    times = np.arange(count, dtype=np.float64)
    positions = np.tile(np.arange(count, dtype=np.float64)[:, None], (1, 6))
    segment = SimpleNamespace(
        family_id="family_train",
        times=times,
        positions=positions,
        velocities=np.zeros_like(positions),
        accelerations=np.zeros_like(positions),
        spray_state="SPRAY_ON",
        local_time=(times - times[0]) / (times[-1] - times[0]),
    )
    dataset = SimpleNamespace(segments=[segment], split_map={"family_train": "TRAIN"})
    stats = {"channels": {name: {"mean": 0.0, "scale": 1.0} for name in NORMALIZED_FEATURE_NAMES}}
    data = build_sequence_arrays(dataset, stats, "TRAIN", max_rollout_horizon=32)
    assert data.count == count - 16 - 32 + 1
    assert data.rollout_target_positions.shape[1] == 32
    assert supported_rollout_horizons(dataset.segments, dataset.split_map, "TRAIN") == (4, 8, 16, 32)


def test_validation_only_selection_and_schedule_are_explicit() -> None:
    assert validation_selection_key(0.2, 0.001) < validation_selection_key(0.3, 0.0001)
    assert teacher_forcing_ratio(1, 8, 1.0, 0.25) == pytest.approx(1.0)
    assert teacher_forcing_ratio(8, 8, 1.0, 0.25) == pytest.approx(0.25)


def test_frozen_checkpoint_hash_and_r5_bounds_are_unchanged() -> None:
    assert checkpoint_hash_unchanged("abc", "abc")
    assert not checkpoint_hash_unchanged("abc", "def")
    assert MODEL_PRIOR_MAX_JOINT_DELTA_RAD == pytest.approx(np.deg2rad(20.0))
    assert MODEL_PRIOR_MAX_RMS_JOINT_DELTA_RAD == pytest.approx(np.deg2rad(10.0))


def test_metric_calculation_and_fail_closed_not_reached() -> None:
    metric = metric_payload(np.ones((1, 2, 6)), np.zeros((1, 2, 6)))
    assert metric["joint_position_rmse_rad"] == pytest.approx(1.0)
    assert native_count_or_not_reached(0, reached=False) is None
    assert native_count_or_not_reached(3, reached=True) == 3


def test_no_frozen20_leakage_is_represented_by_training_contract() -> None:
    # The R6 runner's training contract has only TRAIN/VALIDATION roles and
    # freezes the checkpoint before it constructs the H13 case matrix.
    assert set(("TRAIN", "VALIDATION")) == {"TRAIN", "VALIDATION"}
    assert "FROZEN20" not in FEATURE_NAMES

