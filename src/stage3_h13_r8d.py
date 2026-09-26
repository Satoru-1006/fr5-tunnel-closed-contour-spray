"""Stage 3 H13-R8D: frozen-R6 residual rollout correction helpers.

R8D deliberately keeps the authoritative R6 predictor intact.  The only
trainable object in this module is a small bounded position corrector that
is evaluated from the current causal rollout state and the frozen R6 raw
next-state prediction.  Future positions are accepted only by the loss-side
helpers used by the training caller; the free-running rollout itself has no
target argument.
"""

from __future__ import annotations

import io
import math
from typing import Any, Mapping, Sequence

import numpy as np

from src.stage3_h11_dataset import FEATURE_NAMES
from src.stage3_h11_r2_model import SPRAY_FEATURE_INDEX
from src.stage3_h13_r6 import R6SequenceArrays, _next_features, as_window_arrays
from src.stage3_h11_r2_model import causal_constant_velocity_baseline

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - contract tests remain importable
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    TensorDataset = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


R8D_SCHEMA_VERSION = "stage3_h13_r8d_frozen_r6_residual_corrector_v1"
R8D_EVALUATION_HORIZONS = (1, 4, 8, 12, 16, 20, 24, 32)
R8D_TRAINING_HORIZONS = R8D_EVALUATION_HORIZONS
JOINTS = 6
CORRECTOR_INPUT_SIZE = 20 + 18


def parameter_count(model: Any) -> int:
    return int(sum(int(parameter.numel()) for parameter in model.parameters()))


def derive_train_correction_bounds(train_data: R6SequenceArrays, percentile: float = 99.0) -> np.ndarray:
    """Derive physical position bounds from TRAIN labels only.

    The reference is the causal constant-velocity baseline used by the
    existing R6 semantics.  A bound is per-joint and uses only TRAIN target
    positions; validation is never consulted to choose it.
    """

    if train_data.count == 0:
        raise ValueError("r8d_train_data_missing_for_bounds")
    baseline = causal_constant_velocity_baseline(as_window_arrays(train_data))
    residual = np.abs(train_data.direct_target_positions.astype(np.float64) - baseline)
    bounds = np.percentile(residual.reshape(-1, JOINTS), float(percentile), axis=0)
    bounds = np.maximum(bounds, 1.0e-6)
    if not np.all(np.isfinite(bounds)):
        raise ValueError("r8d_nonfinite_train_correction_bounds")
    return bounds.astype(np.float32)


if TORCH_AVAILABLE:

    class BoundedPositionResidualCorrector(nn.Module):
        """Small causal MLP emitting only a bounded physical Δposition."""

        def __init__(self, correction_bounds: Sequence[float], hidden_size: int = 32) -> None:
            super().__init__()
            bounds = torch.as_tensor(list(correction_bounds), dtype=torch.float32)
            if tuple(bounds.shape) != (JOINTS,) or bool(torch.any(bounds <= 0.0)):
                raise ValueError("r8d_invalid_correction_bounds")
            self.input_size = CORRECTOR_INPUT_SIZE
            self.hidden_size = int(hidden_size)
            self.output_size = JOINTS
            self.register_buffer("correction_bounds", bounds)
            self.net = nn.Sequential(
                nn.Linear(CORRECTOR_INPUT_SIZE, self.hidden_size),
                nn.Tanh(),
                nn.Linear(self.hidden_size, JOINTS),
            )
            final = self.net[-1]
            assert isinstance(final, nn.Linear)
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

        def forward(self, features: "torch.Tensor") -> "torch.Tensor":
            return torch.tanh(self.net(features)) * self.correction_bounds

else:

    class BoundedPositionResidualCorrector:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("pytorch_unavailable")


def _normalize(value: "torch.Tensor", index: int, channels: Mapping[str, Mapping[str, Any]]) -> "torch.Tensor":
    name = FEATURE_NAMES[index]
    if name == "spray_on" or name not in channels:
        return value
    item = channels[name]
    scale = float(item.get("scale", item.get("std", 1.0)))
    if not math.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    return (value - float(item.get("mean", 0.0))) / scale


def corrector_features(
    current_inputs: "torch.Tensor",
    raw_next_position: "torch.Tensor",
    last_position: "torch.Tensor",
    prior_position: "torch.Tensor",
    next_time: "torch.Tensor",
    last_time: "torch.Tensor",
    prior_time: "torch.Tensor",
    channels: Mapping[str, Mapping[str, Any]],
) -> "torch.Tensor":
    """Build 38 causal features: current R6 row + raw next kinematics."""

    dt = torch.clamp(next_time - last_time, min=1.0e-9)
    prior_dt = torch.clamp(last_time - prior_time, min=1.0e-9)
    raw_velocity = (raw_next_position - last_position) / dt[:, None]
    previous_velocity = (last_position - prior_position) / prior_dt[:, None]
    raw_acceleration = (raw_velocity - previous_velocity) / dt[:, None]
    raw_kinematics = tuple(raw_next_position.T) + tuple(raw_velocity.T) + tuple(raw_acceleration.T)
    normalized = [_normalize(value, index, channels) for index, value in enumerate(raw_kinematics)]
    return torch.cat((current_inputs[:, -1, :], torch.stack(normalized, dim=1)), dim=1)


def rollout_r8d(
    r6_model: Any,
    corrector: Any,
    inputs: "torch.Tensor",
    history_positions: "torch.Tensor",
    history_times: "torch.Tensor",
    target_times: "torch.Tensor",
    trajectory_start_times: "torch.Tensor",
    trajectory_end_times: "torch.Tensor",
    channels: Mapping[str, Mapping[str, Any]],
    *,
    rollout_horizon: int = 32,
    detach_state: bool = False,
    return_trace: bool = False,
) -> Any:
    """Fully free-running causal rollout; no future target argument exists."""

    if not TORCH_AVAILABLE:
        raise RuntimeError("pytorch_unavailable")
    if rollout_horizon < 1 or rollout_horizon > int(target_times.shape[1]):
        raise ValueError("r8d_invalid_rollout_horizon")
    current_inputs = inputs
    positions = history_positions
    times = history_times
    predictions: list[torch.Tensor] = []
    raw_predictions: list[torch.Tensor] = []
    corrections: list[torch.Tensor] = []
    generated_features: list[torch.Tensor] = []
    for step in range(int(rollout_horizon)):
        raw_residual = r6_model(current_inputs)[:, 0, :]
        last = positions[:, -1, :]
        prior = positions[:, -2, :]
        last_time = times[:, -1]
        prior_time = times[:, -2]
        dt = torch.clamp(target_times[:, step] - last_time, min=1.0e-9)
        raw_next = last + (last - prior) / torch.clamp(last_time - prior_time, min=1.0e-9)[:, None] * dt[:, None] + raw_residual
        features = corrector_features(current_inputs, raw_next, last, prior, target_times[:, step], last_time, prior_time, channels)
        correction = corrector(features)
        next_position = raw_next + correction
        predictions.append(next_position)
        raw_predictions.append(raw_next)
        corrections.append(correction)
        if step + 1 == int(rollout_horizon):
            continue
        next_features = _next_features(
            next_position,
            last,
            prior,
            target_times[:, step],
            last_time,
            prior_time,
            trajectory_start_times,
            trajectory_end_times,
            current_inputs[:, -1, SPRAY_FEATURE_INDEX],
            channels,
            FEATURE_NAMES,
        ).to(dtype=current_inputs.dtype)
        generated_features.append(next_features)
        state_position = next_position.detach() if detach_state else next_position
        state_features = next_features.detach() if detach_state else next_features
        current_inputs = torch.cat((current_inputs[:, 1:, :], state_features[:, None, :]), dim=1)
        positions = torch.cat((positions[:, 1:, :], state_position[:, None, :]), dim=1)
        times = torch.cat((times[:, 1:], target_times[:, step, None]), dim=1)
    prediction_tensor = torch.stack(predictions, dim=1)
    if not return_trace:
        return prediction_tensor
    return {
        "predictions": prediction_tensor,
        "raw_predictions": torch.stack(raw_predictions, dim=1),
        "corrections": torch.stack(corrections, dim=1),
        "generated_features": torch.stack(generated_features, dim=1) if generated_features else None,
    }


def rollout_r8d_teacher_forced_diagnostic(
    r6_model: Any,
    corrector: Any,
    inputs: "torch.Tensor",
    history_positions: "torch.Tensor",
    history_times: "torch.Tensor",
    target_times: "torch.Tensor",
    trajectory_start_times: "torch.Tensor",
    trajectory_end_times: "torch.Tensor",
    target_positions: "torch.Tensor",
    channels: Mapping[str, Mapping[str, Any]],
    *,
    rollout_horizon: int = 32,
) -> "torch.Tensor":
    """Teacher-forced diagnostic only; never used for deployment selection."""

    if not TORCH_AVAILABLE:
        raise RuntimeError("pytorch_unavailable")
    current_inputs = inputs
    positions = history_positions
    times = history_times
    predictions: list[torch.Tensor] = []
    for step in range(int(rollout_horizon)):
        raw_residual = r6_model(current_inputs)[:, 0, :]
        last = positions[:, -1, :]
        prior = positions[:, -2, :]
        last_time = times[:, -1]
        prior_time = times[:, -2]
        dt = torch.clamp(target_times[:, step] - last_time, min=1.0e-9)
        raw_next = last + (last - prior) / torch.clamp(last_time - prior_time, min=1.0e-9)[:, None] * dt[:, None] + raw_residual
        features = corrector_features(current_inputs, raw_next, last, prior, target_times[:, step], last_time, prior_time, channels)
        corrected_next = raw_next + corrector(features)
        predictions.append(corrected_next)
        if step + 1 == int(rollout_horizon):
            continue
        teacher_next = target_positions[:, step, :]
        next_features = _next_features(
            teacher_next,
            last,
            prior,
            target_times[:, step],
            last_time,
            prior_time,
            trajectory_start_times,
            trajectory_end_times,
            current_inputs[:, -1, SPRAY_FEATURE_INDEX],
            channels,
            FEATURE_NAMES,
        ).to(dtype=current_inputs.dtype)
        current_inputs = torch.cat((current_inputs[:, 1:, :], next_features[:, None, :]), dim=1)
        positions = torch.cat((positions[:, 1:, :], teacher_next[:, None, :]), dim=1)
        times = torch.cat((times[:, 1:], target_times[:, step, None]), dim=1)
    return torch.stack(predictions, dim=1)


def _rollout_kinematics(positions: "torch.Tensor", history_positions: "torch.Tensor", history_times: "torch.Tensor", target_times: "torch.Tensor") -> tuple[torch.Tensor, torch.Tensor]:
    combined_positions = torch.cat((history_positions[:, -1:, :], positions), dim=1)
    combined_times = torch.cat((history_times[:, -1:], target_times), dim=1)
    dt = torch.clamp(combined_times[:, 1:] - combined_times[:, :-1], min=1.0e-9)
    velocity = (combined_positions[:, 1:, :] - combined_positions[:, :-1, :]) / dt[:, :, None]
    velocity_dt = torch.clamp(combined_times[:, 2:] - combined_times[:, 1:-1], min=1.0e-9)
    acceleration = (velocity[:, 1:, :] - velocity[:, :-1, :]) / velocity_dt[:, :, None]
    return velocity, acceleration


def multi_horizon_position_loss(predicted: "torch.Tensor", target: "torch.Tensor", weights: Mapping[int, float]) -> "torch.Tensor":
    assert F is not None
    terms = [float(weight) * F.mse_loss(predicted[:, :int(horizon), :], target[:, :int(horizon), :]) for horizon, weight in weights.items()]
    return torch.stack(terms).sum()


def train_r8d_corrector(
    r6_model: Any,
    corrector: Any,
    train_data: R6SequenceArrays,
    channels: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    device: str = "cpu",
) -> tuple[Any, list[dict[str, Any]]]:
    """Train only the corrector on corrected free-running TRAIN rollouts."""

    if not TORCH_AVAILABLE:
        raise RuntimeError("pytorch_unavailable")
    assert torch is not None and F is not None and DataLoader is not None and TensorDataset is not None
    if train_data.count == 0:
        raise RuntimeError("r8d_train_windows_missing")
    for parameter in r6_model.parameters():
        parameter.requires_grad_(False)
    r6_model.eval()
    corrector.to(device)
    optimizer = torch.optim.AdamW(corrector.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config.get("weight_decay", 0.0)))
    stride = max(1, int(config.get("train_window_stride", 1)))
    sampled = np.arange(0, train_data.count, stride, dtype=np.int64)
    loader = DataLoader(TensorDataset(torch.from_numpy(train_data.inputs[sampled]), torch.from_numpy(sampled)), batch_size=int(config["batch_size"]), shuffle=False, num_workers=0, drop_last=False)
    horizon_weights = {int(key): float(value) for key, value in config["horizon_loss_weights"].items()}
    bounds = corrector.correction_bounds.detach().to(device=device, dtype=torch.float32)
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(config["max_epochs"]) + 1):
        corrector.train()
        loss_sum = 0.0
        pos_sum = 0.0
        vel_sum = 0.0
        delta_sum = 0.0
        sample_count = 0
        for batch_inputs, batch_indices in loader:
            batch_inputs = batch_inputs.to(device=device, dtype=torch.float32)
            indices = batch_indices.detach().cpu().numpy()
            hp = torch.from_numpy(train_data.history_positions[indices]).to(device=device, dtype=torch.float32)
            ht = torch.from_numpy(train_data.history_times[indices]).to(device=device, dtype=torch.float32)
            tt = torch.from_numpy(train_data.rollout_target_times[indices]).to(device=device, dtype=torch.float32)
            tp = torch.from_numpy(train_data.rollout_target_positions[indices]).to(device=device, dtype=torch.float32)
            starts = torch.from_numpy(train_data.trajectory_start_times[indices]).to(device=device, dtype=torch.float32)
            ends = torch.from_numpy(train_data.trajectory_end_times[indices]).to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            trace = rollout_r8d(r6_model, corrector, batch_inputs, hp, ht, tt, starts, ends, channels, rollout_horizon=32, detach_state=bool(config.get("detach_state", True)), return_trace=True)
            predicted = trace["predictions"]
            position_loss = multi_horizon_position_loss(predicted, tp, horizon_weights)
            pred_velocity, pred_acceleration = _rollout_kinematics(predicted, hp, ht, tt)
            true_velocity, true_acceleration = _rollout_kinematics(tp, hp, ht, tt)
            velocity_loss = F.mse_loss(pred_velocity, true_velocity)
            acceleration_loss = F.mse_loss(pred_acceleration, true_acceleration)
            local_horizon = min(int(config.get("local_guardrail_horizon", 4)), 32)
            local_loss = F.mse_loss(predicted[:, :local_horizon, :], tp[:, :local_horizon, :])
            delta_loss = torch.mean(torch.square(trace["corrections"] / bounds[None, None, :]))
            total_loss = (
                float(config["w_position"]) * position_loss
                + float(config["w_velocity"]) * velocity_loss
                + float(config["w_acceleration"]) * acceleration_loss
                + float(config["w_local"]) * local_loss
                + float(config["w_delta"]) * delta_loss
            )
            if not torch.isfinite(total_loss):
                raise RuntimeError("r8d_training_nonfinite")
            total_loss.backward()
            if any(parameter.grad is not None for parameter in r6_model.parameters()):
                raise RuntimeError("r8d_r6_gradient_detected")
            torch.nn.utils.clip_grad_norm_(corrector.parameters(), float(config.get("gradient_clip_norm", 1.0)))
            optimizer.step()
            batch_size = len(batch_inputs)
            loss_sum += float(total_loss.detach().cpu()) * batch_size
            pos_sum += float(position_loss.detach().cpu()) * batch_size
            vel_sum += float(velocity_loss.detach().cpu()) * batch_size
            delta_sum += float(delta_loss.detach().cpu()) * batch_size
            sample_count += batch_size
        history.append({
            "epoch": epoch,
            "train_window_count": int(len(sampled)),
            "train_window_stride": stride,
            "train_total_loss": loss_sum / max(sample_count, 1),
            "train_position_loss": pos_sum / max(sample_count, 1),
            "train_velocity_loss": vel_sum / max(sample_count, 1),
            "train_normalized_correction_regularization": delta_sum / max(sample_count, 1),
            "teacher_forcing_ratio": 0.0,
            "free_running_training": True,
        })
    if any(parameter.requires_grad for parameter in r6_model.parameters()):
        raise RuntimeError("r8d_r6_unfrozen")
    return corrector, history


def checkpoint_bytes(corrector: Any, config: Mapping[str, Any], selected_epoch: int) -> bytes:
    if not TORCH_AVAILABLE:
        raise RuntimeError("pytorch_unavailable")
    buffer = io.BytesIO()
    torch.save({
        "model_state_dict": {key: value.detach().cpu().clone() for key, value in corrector.state_dict().items()},
        "model_type": "bounded_position_residual_corrector_mlp",
        "schema_version": R8D_SCHEMA_VERSION,
        "input_size": CORRECTOR_INPUT_SIZE,
        "hidden_size": int(corrector.hidden_size),
        "correction_bounds": [float(value) for value in corrector.correction_bounds.detach().cpu().numpy()],
        "config": dict(config),
        "selected_epoch": int(selected_epoch),
    }, buffer)
    return buffer.getvalue()


def load_corrector(path: Any, *, map_location: str = "cpu") -> Any:
    if not TORCH_AVAILABLE:
        raise RuntimeError("pytorch_unavailable")
    payload = torch.load(path, map_location=map_location, weights_only=False)
    corrector = BoundedPositionResidualCorrector(payload["correction_bounds"], hidden_size=int(payload["hidden_size"]))
    corrector.load_state_dict(payload["model_state_dict"], strict=True)
    corrector.eval()
    return corrector
