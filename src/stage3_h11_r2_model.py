"""Causal residual-over-baseline model utilities for Stage 3 H11-R2.

The module is deliberately additive to :mod:`stage3_h11_model`.  It keeps the
H11 window schema and GRU family, but makes the causal constant-velocity
prediction an explicit, immutable part of both direct prediction and
free-running rollout.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    TensorDataset = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

from src.stage3_h11_model import WindowArrays, metric_payload, parameter_count


JOINTS = 6
FEATURE_COUNT = 20
SPRAY_FEATURE_INDEX = 18
TIME_FEATURE_INDEX = 19


if TORCH_AVAILABLE:

    class ResidualCausalGRUTrajectoryPredictor(nn.Module):
        """Unidirectional GRU whose output is a correction in radians.

        The final linear layer is initialized to zero.  Consequently a fresh
        model emits exactly zero residual and therefore exactly the causal
        constant-velocity baseline until it learns a correction.
        """

        def __init__(self, input_size: int = FEATURE_COUNT, hidden_size: int = 128, num_layers: int = 2, horizon: int = 8) -> None:
            super().__init__()
            self.input_size = int(input_size)
            self.hidden_size = int(hidden_size)
            self.num_layers = int(num_layers)
            self.horizon = int(horizon)
            self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=False)
            self.head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, horizon * JOINTS))
            final = self.head[-1]
            assert isinstance(final, nn.Linear)
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

        def forward(self, inputs: "torch.Tensor") -> "torch.Tensor":
            sequence, _ = self.gru(inputs)
            last = sequence[:, -1, :]
            return self.head(last).reshape(inputs.shape[0], self.horizon, JOINTS)

else:

    class ResidualCausalGRUTrajectoryPredictor:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("pytorch_unavailable")


def _safe_dt(times: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(times, dtype=np.float64), 1.0e-9)


def causal_constant_velocity_baseline(arrays: WindowArrays) -> np.ndarray:
    """Return the independent causal baseline using history only.

    No target positions are read.  This is intentionally equivalent to the
    repository H11 constant-velocity semantics, but is kept local so R2 can
    audit and test its independence directly.
    """
    if len(arrays.history_positions) == 0:
        return np.empty((0, arrays.target_positions.shape[1], JOINTS), dtype=np.float64)
    history_dt = _safe_dt(np.diff(arrays.history_times, axis=1))
    velocity = np.diff(arrays.history_positions, axis=1) / history_dt[:, :, None]
    anchor_velocity = velocity[:, -1, :]
    elapsed = arrays.target_times - arrays.history_times[:, -1, None]
    return arrays.anchor_positions[:, None, :] + anchor_velocity[:, None, :] * elapsed[:, :, None]


def residual_targets(arrays: WindowArrays) -> tuple[np.ndarray, np.ndarray]:
    """Return target residuals and the causal baseline positions."""
    baseline = causal_constant_velocity_baseline(arrays)
    return arrays.target_positions - baseline, baseline


def compose_prediction(arrays: WindowArrays, residual: np.ndarray) -> np.ndarray:
    """Compose final positions from causal baseline plus learned residual."""
    baseline = causal_constant_velocity_baseline(arrays)
    residual = np.asarray(residual, dtype=np.float64)
    if residual.shape != baseline.shape:
        raise ValueError(f"residual_shape_mismatch:{residual.shape}!={baseline.shape}")
    return baseline + residual


def zero_residual_prediction(arrays: WindowArrays) -> np.ndarray:
    return compose_prediction(arrays, np.zeros_like(arrays.target_positions, dtype=np.float64))


def _normalize_feature_tensor(values: "torch.Tensor", index: int, channels: Mapping[str, Mapping[str, Any]], names: Sequence[str]) -> "torch.Tensor":
    name = names[index]
    if name not in channels:
        return values
    item = channels[name]
    mean = float(item["mean"])
    scale = float(item.get("scale", item.get("std", 1.0)))
    if not math.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    return (values - mean) / scale


def _next_features_torch(
    next_position: "torch.Tensor",
    previous_position: "torch.Tensor",
    prior_position: "torch.Tensor",
    next_time: "torch.Tensor",
    last_time: "torch.Tensor",
    prior_time: "torch.Tensor",
    start_time: "torch.Tensor",
    fixed_time_scale: "torch.Tensor",
    spray: "torch.Tensor",
    channels: Mapping[str, Mapping[str, Any]],
    feature_names: Sequence[str],
) -> "torch.Tensor":
    dt = torch.clamp(next_time - last_time, min=1.0e-9)
    prior_dt = torch.clamp(last_time - prior_time, min=1.0e-9)
    velocity = (next_position - previous_position) / dt[:, None]
    prior_velocity = (previous_position - prior_position) / prior_dt[:, None]
    acceleration = (velocity - prior_velocity) / dt[:, None]
    local_time = (next_time - start_time) / torch.clamp(fixed_time_scale, min=1.0e-9)
    columns = []
    for index, value in enumerate(tuple(next_position.T) + tuple(velocity.T) + tuple(acceleration.T) + (spray, local_time)):
        columns.append(_normalize_feature_tensor(value, index, channels, feature_names))
    return torch.stack(columns, dim=1)


def rollout_residual_torch(
    model: Any,
    inputs: "torch.Tensor",
    history_positions: "torch.Tensor",
    history_times: "torch.Tensor",
    target_times: "torch.Tensor",
    channels: Mapping[str, Mapping[str, Any]],
    feature_names: Sequence[str],
    rollout_horizon: int,
    teacher_forcing_ratio: float,
    target_positions: "torch.Tensor | None" = None,
    sample_random: bool = True,
) -> "torch.Tensor":
    """Differentiable bounded free-running rollout with optional sampling.

    Only the current history is used to make the baseline.  Ground-truth
    target positions are used solely for the loss and, when selected by the
    bounded teacher-forcing schedule, for the *next history*; they never enter
    the current baseline before the model prediction is made.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("pytorch_unavailable")
    if rollout_horizon < 1 or rollout_horizon > target_times.shape[1]:
        raise ValueError("invalid_rollout_horizon")
    current_inputs = inputs
    positions = history_positions
    times = history_times
    start_time = history_times[:, 0]
    fixed_time_scale = history_times[:, -1] - history_times[:, 0]
    predictions = []
    for step in range(rollout_horizon):
        residual = model(current_inputs)[:, 0, :]
        last = positions[:, -1, :]
        prior = positions[:, -2, :]
        last_time = times[:, -1]
        prior_time = times[:, -2]
        dt = torch.clamp(target_times[:, step] - last_time, min=1.0e-9)
        velocity = (last - prior) / torch.clamp(last_time - prior_time, min=1.0e-9)[:, None]
        baseline_next = last + velocity * dt[:, None]
        next_position = baseline_next + residual
        predictions.append(next_position)
        if step + 1 == rollout_horizon:
            continue
        if target_positions is not None and teacher_forcing_ratio > 0.0:
            if sample_random:
                use_truth = torch.rand((len(next_position), 1), device=next_position.device) < float(teacher_forcing_ratio)
            else:
                use_truth = torch.full((len(next_position), 1), bool(teacher_forcing_ratio >= 0.5), device=next_position.device, dtype=torch.bool)
            history_position = torch.where(use_truth, target_positions[:, step, :], next_position)
        else:
            history_position = next_position
        next_features = _next_features_torch(
            history_position, last, prior, target_times[:, step], last_time, prior_time,
            start_time, fixed_time_scale, current_inputs[:, -1, SPRAY_FEATURE_INDEX], channels, feature_names,
        ).to(dtype=current_inputs.dtype)
        current_inputs = torch.cat((current_inputs[:, 1:, :], next_features[:, None, :]), dim=1)
        positions = torch.cat((positions[:, 1:, :], history_position[:, None, :]), dim=1)
        times = torch.cat((times[:, 1:], target_times[:, step, None]), dim=1)
    return torch.stack(predictions, dim=1)


def rollout_residual_numpy(model: Any, arrays: WindowArrays, channels: Mapping[str, Mapping[str, Any]], feature_names: Sequence[str], device: str = "cpu", batch_size: int = 512) -> np.ndarray:
    """Fully free-running residual rollout for evaluation, without targets."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("pytorch_unavailable")
    assert torch is not None
    model.eval()
    output = np.empty((len(arrays.inputs), arrays.target_positions.shape[1], JOINTS), dtype=np.float64)
    for start in range(0, len(arrays.inputs), batch_size):
        end = min(len(arrays.inputs), start + batch_size)
        x = torch.from_numpy(arrays.inputs[start:end]).to(device=device, dtype=torch.float32)
        positions = torch.from_numpy(arrays.history_positions[start:end]).to(device=device, dtype=torch.float64)
        times = torch.from_numpy(arrays.history_times[start:end]).to(device=device, dtype=torch.float64)
        target_times = torch.from_numpy(arrays.target_times[start:end]).to(device=device, dtype=torch.float64)
        with torch.no_grad():
            pred = rollout_residual_torch(model, x, positions, times, target_times, channels, feature_names, target_times.shape[1], 0.0, None, False)
        output[start:end] = pred.detach().cpu().numpy()
    return output


def direct_metrics(model: Any, arrays: WindowArrays, device: str = "cpu", batch_size: int = 2048) -> tuple[np.ndarray, dict[str, Any]]:
    if not TORCH_AVAILABLE:
        raise RuntimeError("pytorch_unavailable")
    assert torch is not None
    model.eval()
    residuals = []
    with torch.no_grad():
        for start in range(0, len(arrays.inputs), batch_size):
            batch = torch.from_numpy(arrays.inputs[start:start + batch_size]).to(device=device, dtype=torch.float32)
            residuals.append(model(batch).detach().cpu().numpy())
    residual = np.concatenate(residuals, axis=0) if residuals else np.empty_like(arrays.target_positions)
    predicted = compose_prediction(arrays, residual)
    return predicted, metric_payload(predicted, arrays.target_positions)


def _initialise_from_h11_backbone(model: Any, checkpoint_bytes: bytes) -> None:
    """Copy only compatible GRU/hidden-head weights; retain zero residual head."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("pytorch_unavailable")
    assert torch is not None
    payload = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=False)
    source = payload.get("model_state_dict", payload)
    current = model.state_dict()
    for key, value in source.items():
        if key.startswith("gru.") or key.startswith("head.0."):
            if key in current and tuple(current[key].shape) == tuple(value.shape):
                current[key].copy_(value)
    model.load_state_dict(current)
    final = model.head[-1]
    with torch.no_grad():
        final.weight.zero_(); final.bias.zero_()


def train_residual_model(
    model: Any,
    train_arrays: WindowArrays,
    validation_arrays: WindowArrays,
    channels: Mapping[str, Mapping[str, Any]],
    feature_names: Sequence[str],
    config: Mapping[str, Any],
    device: str = "cpu",
) -> tuple[Any, list[dict[str, Any]], int, bytes]:
    if not TORCH_AVAILABLE:
        raise RuntimeError("pytorch_unavailable")
    assert torch is not None and nn is not None and F is not None and DataLoader is not None and TensorDataset is not None
    train_residual, _ = residual_targets(train_arrays)
    train_inputs = torch.from_numpy(train_arrays.inputs.astype(np.float32, copy=False))
    train_target = torch.from_numpy(train_residual.astype(np.float32, copy=False))
    loader = DataLoader(TensorDataset(train_inputs, train_target), batch_size=int(config["batch_size"]), shuffle=False, num_workers=0, drop_last=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config.get("weight_decay", 0.0)))
    model.to(device)
    best_state: dict[str, Any] | None = None
    best_score = math.inf
    best_epoch = 0
    no_improvement = 0
    history: list[dict[str, Any]] = []
    beta = float(config.get("huber_beta", 1.0e-3))
    w_direct = float(config.get("w_direct", 1.0))
    w_rollout = float(config.get("w_rollout", 0.0))
    rollout_horizon = int(config.get("rollout_horizon", 4))
    initial_tf = float(config.get("teacher_forcing_initial_ratio", 1.0))
    final_tf = float(config.get("teacher_forcing_final_ratio", initial_tf))
    for epoch in range(1, int(config["max_epochs"]) + 1):
        model.train()
        tf_ratio = initial_tf + (final_tf - initial_tf) * ((epoch - 1) / max(int(config["max_epochs"]) - 1, 1))
        direct_sum = 0.0; rollout_sum = 0.0; sample_count = 0
        for batch_index, (batch_inputs, batch_target_residual) in enumerate(loader):
            batch_start = batch_index * int(config["batch_size"])
            batch_end = batch_start + len(batch_inputs)
            batch_inputs = batch_inputs.to(device=device, dtype=torch.float32)
            batch_target_residual = batch_target_residual.to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            predicted_residual = model(batch_inputs)
            if config.get("loss") == "huber":
                direct_loss = F.smooth_l1_loss(predicted_residual, batch_target_residual, beta=beta)
            else:
                direct_loss = F.mse_loss(predicted_residual, batch_target_residual)
            rollout_loss = torch.zeros((), device=device)
            if w_rollout > 0.0:
                hp = torch.from_numpy(train_arrays.history_positions[batch_start:batch_end]).to(device=device, dtype=torch.float32)
                ht = torch.from_numpy(train_arrays.history_times[batch_start:batch_end]).to(device=device, dtype=torch.float32)
                tt = torch.from_numpy(train_arrays.target_times[batch_start:batch_end]).to(device=device, dtype=torch.float32)
                tp = torch.from_numpy(train_arrays.target_positions[batch_start:batch_end]).to(device=device, dtype=torch.float32)
                rolled = rollout_residual_torch(model, batch_inputs, hp, ht, tt, channels, feature_names, rollout_horizon, tf_ratio, tp, True)
                target = tp[:, :rollout_horizon, :]
                rollout_loss = F.smooth_l1_loss(rolled, target, beta=beta) if config.get("loss") == "huber" else F.mse_loss(rolled, target)
            loss = w_direct * direct_loss + w_rollout * rollout_loss
            if not torch.isfinite(loss):
                raise RuntimeError("training_nonfinite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("gradient_clip_norm", 1.0)))
            optimizer.step()
            direct_sum += float(direct_loss.detach().cpu()) * len(batch_inputs)
            rollout_sum += float(rollout_loss.detach().cpu()) * len(batch_inputs)
            sample_count += len(batch_inputs)
        _, val_direct = direct_metrics(model, validation_arrays, device=device, batch_size=int(config.get("eval_batch_size", 2048)))
        val_rollout = rollout_residual_numpy(model, validation_arrays, channels, feature_names, device=device, batch_size=int(config.get("eval_batch_size", 2048)))
        val_rollout_metric = metric_payload(val_rollout, validation_arrays.target_positions)
        direct_rmse = float(val_direct["joint_position_rmse_rad"])
        rollout_rmse = float(val_rollout_metric["joint_position_rmse_rad"])
        score = direct_rmse + float(config.get("selection_rollout_weight", 0.25)) * rollout_rmse
        history.append({"epoch": epoch, "teacher_forcing_ratio": tf_ratio, "train_direct_loss": direct_sum / max(sample_count, 1), "train_rollout_loss": rollout_sum / max(sample_count, 1), "validation_direct_rmse_rad": direct_rmse, "validation_rollout_rmse_rad": rollout_rmse, "validation_selection_score": score})
        if not math.isfinite(score):
            raise RuntimeError("validation_nonfinite")
        if score < best_score - float(config.get("min_delta", 0.0)):
            best_score = score; best_epoch = epoch; no_improvement = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            no_improvement += 1
        if no_improvement >= int(config["patience"]):
            break
    if best_state is None:
        raise RuntimeError("validation_checkpoint_not_selected")
    model.load_state_dict(best_state)
    buffer = io.BytesIO()
    torch.save({"model_state_dict": best_state, "model_type": "residual_causal_gru", "input_history": 16, "prediction_horizon": 8}, buffer)
    return model, history, best_epoch, buffer.getvalue()


def model_summary(model: Any) -> dict[str, Any]:
    return {"model_type": "residual_causal_gru", "input_features": int(model.input_size), "input_history": 16, "prediction_horizon": int(model.horizon), "hidden_size": int(model.hidden_size), "num_layers": int(model.num_layers), "bidirectional": False, "trainable_parameters": parameter_count(model), "residual_output_initialization": "exact_zero", "baseline_semantics": "causal_constant_velocity_from_history_only"}
