"""Causally anchored rollout semantics for Stage 3 H13-R8.

R6 fed each model prediction back into the position/velocity/acceleration
history.  This module keeps the same causal feature schema but advances the
rollout with a reference-only state.  The model predicts a bounded local
residual relative to that reference; a model prediction is never used to
construct the next model input.

The two supported update semantics are intentionally small ablations:

``single_step``
    Recompute one bounded residual at every causal-reference step.

``chunked_multihorizon``
    Use the model's existing eight-step head against the reference-only
    context, then advance the context with the reference states only.

Both modes are causal.  Target positions are accepted by the training caller
only as loss targets and are deliberately absent from this rollout function.
"""

from __future__ import annotations

import io
import math
from typing import Any, Mapping, Sequence

import numpy as np

from src.stage3_h11_dataset import FEATURE_NAMES, INPUT_HISTORY, PREDICTION_HORIZON
from src.stage3_h11_model import metric_payload
from src.stage3_h11_r2_model import JOINTS, SPRAY_FEATURE_INDEX, causal_constant_velocity_baseline
from src.stage3_h13_r6 import R6SequenceArrays, as_window_arrays, direct_residual_targets, _next_features

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - import-only contract tests remain usable
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    TensorDataset = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


R8_SCHEMA_VERSION = "stage3_h13_r8_autoregressive_state_semantics_v1"
R8_UPDATE_SEMANTICS = ("single_step", "chunked_multihorizon")


def bounded_residual_torch(raw: "torch.Tensor", residual_bound: Sequence[float] | float | None) -> "torch.Tensor":
    """Map the unconstrained head to a local train-derived residual range."""

    if residual_bound is None:
        return raw
    bound = torch.as_tensor(residual_bound, dtype=raw.dtype, device=raw.device)
    if bound.ndim == 0:
        bound = bound.reshape(1)
    if bound.shape[-1] != raw.shape[-1]:
        raise ValueError("r8_residual_bound_joint_shape_mismatch")
    if torch.any(~torch.isfinite(bound)) or torch.any(bound <= 0.0):
        raise ValueError("r8_residual_bound_invalid")
    return torch.tanh(raw) * bound.reshape((1,) * (raw.ndim - 1) + (raw.shape[-1],))


def causal_reference_numpy(
    history_positions: np.ndarray,
    history_times: np.ndarray,
    target_times: np.ndarray,
) -> np.ndarray:
    """Build a constant-velocity reference using only the input history."""

    positions = np.asarray(history_positions, dtype=np.float64)
    times = np.asarray(history_times, dtype=np.float64)
    targets = np.asarray(target_times, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[-1] != JOINTS or times.shape != positions.shape[:2]:
        raise ValueError("r8_reference_history_shape_invalid")
    if targets.ndim != 2 or targets.shape[0] != positions.shape[0]:
        raise ValueError("r8_reference_target_time_shape_invalid")
    dt = np.maximum(times[:, -1] - times[:, -2], 1.0e-9)
    velocity = (positions[:, -1] - positions[:, -2]) / dt[:, None]
    elapsed = targets - times[:, -1, None]
    return positions[:, -1, None, :] + velocity[:, None, :] * elapsed[:, :, None]


def _reference_next_features(
    next_position: "torch.Tensor",
    previous_position: "torch.Tensor",
    prior_position: "torch.Tensor",
    next_time: "torch.Tensor",
    last_time: "torch.Tensor",
    prior_time: "torch.Tensor",
    trajectory_start: "torch.Tensor",
    trajectory_end: "torch.Tensor",
    spray: "torch.Tensor",
    channels: Mapping[str, Mapping[str, Any]],
    feature_names: Sequence[str],
) -> "torch.Tensor":
    # R6's feature construction is reused exactly; only the state source is
    # changed from model output to causal reference output.
    return _next_features(
        next_position,
        previous_position,
        prior_position,
        next_time,
        last_time,
        prior_time,
        trajectory_start,
        trajectory_end,
        spray,
        channels,
        feature_names,
    )


def rollout_reference_anchored_torch(
    model: Any,
    inputs: "torch.Tensor",
    history_positions: "torch.Tensor",
    history_times: "torch.Tensor",
    target_times: "torch.Tensor",
    trajectory_start_times: "torch.Tensor",
    trajectory_end_times: "torch.Tensor",
    channels: Mapping[str, Mapping[str, Any]],
    *,
    semantics: str,
    residual_bound: Sequence[float] | float | None,
    feature_names: Sequence[str] = FEATURE_NAMES,
    rollout_horizon: int,
) -> "torch.Tensor":
    """Run a reference-only causal rollout with bounded local residuals.

    ``target_times`` are causal schedule/progress values already available to
    the window.  No target position is accepted or read here.  The only state
    appended to ``current_inputs`` is the analytically generated causal
    reference state, so prediction error cannot become the next velocity or
    acceleration feature.
    """

    if not TORCH_AVAILABLE:
        raise RuntimeError("pytorch_unavailable")
    if semantics not in R8_UPDATE_SEMANTICS:
        raise ValueError(f"r8_unknown_update_semantics:{semantics}")
    if rollout_horizon < 1 or rollout_horizon > target_times.shape[1]:
        raise ValueError("r8_invalid_rollout_horizon")
    current_inputs = inputs
    reference_positions = history_positions
    reference_times = history_times
    predictions: list[torch.Tensor] = []
    step = 0
    while step < int(rollout_horizon):
        raw_head = model(current_inputs)
        chunk = 1 if semantics == "single_step" else min(PREDICTION_HORIZON, int(rollout_horizon) - step)
        raw_residual = raw_head[:, 0:1, :] if chunk == 1 else raw_head[:, :chunk, :]
        residual = bounded_residual_torch(raw_residual, residual_bound)
        for local in range(chunk):
            last = reference_positions[:, -1, :]
            prior = reference_positions[:, -2, :]
            last_time = reference_times[:, -1]
            prior_time = reference_times[:, -2]
            target_time = target_times[:, step]
            dt = torch.clamp(target_time - last_time, min=1.0e-9)
            velocity = (last - prior) / torch.clamp(last_time - prior_time, min=1.0e-9)[:, None]
            reference_next = last + velocity * dt[:, None]
            predictions.append(reference_next + residual[:, local, :])

            # This is the critical semantic boundary: the next input receives
            # reference_next, never the predicted state above.
            next_features = _reference_next_features(
                reference_next,
                last,
                prior,
                target_time,
                last_time,
                prior_time,
                trajectory_start_times,
                trajectory_end_times,
                current_inputs[:, -1, SPRAY_FEATURE_INDEX],
                channels,
                feature_names,
            ).to(dtype=current_inputs.dtype)
            current_inputs = torch.cat((current_inputs[:, 1:, :], next_features[:, None, :]), dim=1)
            reference_positions = torch.cat((reference_positions[:, 1:, :], reference_next[:, None, :]), dim=1)
            reference_times = torch.cat((reference_times[:, 1:], target_time[:, None]), dim=1)
            step += 1
    return torch.stack(predictions, dim=1)


def bounded_direct_prediction(
    model: Any,
    data: R6SequenceArrays,
    residual_bound: Sequence[float] | float | None,
    *,
    device: str = "cpu",
    batch_size: int = 4096,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compose direct predictions using the same causal CV reference."""

    if not TORCH_AVAILABLE:
        raise RuntimeError("pytorch_unavailable")
    model.eval()
    residuals: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, data.count, int(batch_size)):
            end = min(data.count, start + int(batch_size))
            batch = torch.from_numpy(data.inputs[start:end]).to(device=device, dtype=torch.float32)
            residuals.append(bounded_residual_torch(model(batch), residual_bound).detach().cpu().numpy().astype(np.float64))
    residual = np.concatenate(residuals, axis=0) if residuals else np.empty_like(data.direct_target_positions)
    baseline = causal_constant_velocity_baseline(as_window_arrays(data))
    prediction = baseline + residual
    return prediction, metric_payload(prediction, data.direct_target_positions)


def rollout_rmse(
    model: Any,
    data: R6SequenceArrays,
    channels: Mapping[str, Mapping[str, Any]],
    *,
    semantics: str,
    residual_bound: Sequence[float] | float | None,
    horizons: Sequence[int],
    device: str = "cpu",
    batch_size: int = 4096,
) -> dict[str, float | None]:
    """Compute streaming horizon RMSEs for validation checkpoint selection."""

    if not TORCH_AVAILABLE:
        raise RuntimeError("pytorch_unavailable")
    model.eval()
    max_horizon = max(int(item) for item in horizons)
    sums = {int(item): 0.0 for item in horizons}
    counts = {int(item): 0 for item in horizons}
    with torch.no_grad():
        for start in range(0, data.count, int(batch_size)):
            end = min(data.count, start + int(batch_size))
            rolled = rollout_reference_anchored_torch(
                model,
                torch.from_numpy(data.inputs[start:end]).to(device=device, dtype=torch.float32),
                torch.from_numpy(data.history_positions[start:end]).to(device=device, dtype=torch.float32),
                torch.from_numpy(data.history_times[start:end]).to(device=device, dtype=torch.float32),
                torch.from_numpy(data.rollout_target_times[start:end]).to(device=device, dtype=torch.float32),
                torch.from_numpy(data.trajectory_start_times[start:end]).to(device=device, dtype=torch.float32),
                torch.from_numpy(data.trajectory_end_times[start:end]).to(device=device, dtype=torch.float32),
                channels,
                semantics=semantics,
                residual_bound=residual_bound,
                rollout_horizon=max_horizon,
            )
            target = torch.from_numpy(data.rollout_target_positions[start:end]).to(device=device, dtype=rolled.dtype)
            for horizon in horizons:
                h = int(horizon)
                error = rolled[:, :h, :] - target[:, :h, :]
                sums[h] += float(torch.sum(error * error).detach().cpu())
                counts[h] += int(error.numel())
    return {str(h): (math.sqrt(sums[int(h)] / counts[int(h)]) if counts[int(h)] else None) for h in horizons}


def train_reference_anchored_model(
    model: Any,
    train_data: R6SequenceArrays,
    validation_data: R6SequenceArrays,
    channels: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    semantics: str,
    residual_bound: Sequence[float] | float | None,
    device: str = "cpu",
) -> tuple[Any, list[dict[str, Any]], int, bytes]:
    """Train with inference-matched reference-only multi-step semantics."""

    if not TORCH_AVAILABLE:
        raise RuntimeError("pytorch_unavailable")
    if semantics not in R8_UPDATE_SEMANTICS:
        raise ValueError("r8_training_semantics_invalid")
    if train_data.count == 0 or validation_data.count == 0:
        raise RuntimeError("r8_train_or_validation_windows_missing")
    assert torch is not None and nn is not None and F is not None and DataLoader is not None and TensorDataset is not None
    direct_targets = torch.from_numpy(direct_residual_targets(train_data))
    indices = torch.arange(train_data.count, dtype=torch.long)
    loader = DataLoader(TensorDataset(torch.from_numpy(train_data.inputs), direct_targets, indices), batch_size=int(config["batch_size"]), shuffle=False, num_workers=0, drop_last=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config.get("weight_decay", 0.0)))
    model.to(device)
    beta = float(config.get("huber_beta", 1.0e-3))
    best_state: dict[str, Any] | None = None
    best_key: tuple[float, float] | None = None
    best_epoch = 0
    no_improvement = 0
    history: list[dict[str, Any]] = []
    training_horizon = int(config.get("training_rollout_horizon", config["rollout_horizon"]))
    if training_horizon < 1 or training_horizon > int(config["rollout_horizon"]):
        raise ValueError("r8_training_rollout_horizon_invalid")
    for epoch in range(1, int(config["max_epochs"]) + 1):
        model.train()
        direct_sum = 0.0
        rollout_sum = 0.0
        sample_count = 0
        for batch_inputs, batch_target_residual, batch_indices in loader:
            batch_inputs = batch_inputs.to(device=device, dtype=torch.float32)
            batch_target_residual = batch_target_residual.to(device=device, dtype=torch.float32)
            numpy_indices = batch_indices.detach().cpu().numpy()
            optimizer.zero_grad(set_to_none=True)
            direct_output = bounded_residual_torch(model(batch_inputs), residual_bound)
            direct_loss = F.smooth_l1_loss(direct_output, batch_target_residual, beta=beta)
            hp = torch.from_numpy(train_data.history_positions[numpy_indices]).to(device=device, dtype=torch.float32)
            ht = torch.from_numpy(train_data.history_times[numpy_indices]).to(device=device, dtype=torch.float32)
            tt = torch.from_numpy(train_data.rollout_target_times[numpy_indices]).to(device=device, dtype=torch.float32)
            tp = torch.from_numpy(train_data.rollout_target_positions[numpy_indices]).to(device=device, dtype=torch.float32)
            starts = torch.from_numpy(train_data.trajectory_start_times[numpy_indices]).to(device=device, dtype=torch.float32)
            ends = torch.from_numpy(train_data.trajectory_end_times[numpy_indices]).to(device=device, dtype=torch.float32)
            rolled = rollout_reference_anchored_torch(
                model,
                batch_inputs,
                hp,
                ht,
                tt,
                starts,
                ends,
                channels,
                semantics=semantics,
                residual_bound=residual_bound,
                rollout_horizon=training_horizon,
            )
            rollout_loss = F.smooth_l1_loss(rolled, tp[:, :training_horizon, :], beta=beta)
            loss = float(config.get("w_direct", 1.0)) * direct_loss + float(config.get("w_rollout", 1.0)) * rollout_loss
            if not torch.isfinite(loss):
                raise RuntimeError("r8_training_nonfinite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("gradient_clip_norm", 1.0)))
            optimizer.step()
            direct_sum += float(direct_loss.detach().cpu()) * len(batch_inputs)
            rollout_sum += float(rollout_loss.detach().cpu()) * len(batch_inputs)
            sample_count += len(batch_inputs)
        _, validation_direct = bounded_direct_prediction(model, validation_data, residual_bound, device=device, batch_size=int(config.get("eval_batch_size", 4096)))
        validation_rollout = rollout_rmse(model, validation_data, channels, semantics=semantics, residual_bound=residual_bound, horizons=(int(config["rollout_horizon"]),), device=device, batch_size=int(config.get("eval_batch_size", 4096)))
        validation_rollout_rmse = float(validation_rollout[str(int(config["rollout_horizon"]))])
        validation_direct_rmse = float(validation_direct["joint_position_rmse_rad"])
        key = (validation_rollout_rmse, validation_direct_rmse)
        history.append({
            "epoch": epoch,
            "semantics": semantics,
            "training_rollout_horizon": training_horizon,
            "train_direct_loss": direct_sum / max(sample_count, 1),
            "train_rollout_loss": rollout_sum / max(sample_count, 1),
            "validation_direct_rmse_rad": validation_direct_rmse,
            "validation_autoregressive_rmse_rad": validation_rollout_rmse,
            "validation_selection_primary": "autoregressive_rmse",
        })
        if not (math.isfinite(validation_rollout_rmse) and math.isfinite(validation_direct_rmse)):
            raise RuntimeError("r8_validation_nonfinite")
        if best_key is None or key < best_key:
            best_key = key
            best_epoch = epoch
            no_improvement = 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            no_improvement += 1
        if no_improvement >= int(config["patience"]):
            break
    if best_state is None:
        raise RuntimeError("r8_validation_checkpoint_not_selected")
    model.load_state_dict(best_state, strict=True)
    buffer = io.BytesIO()
    torch.save({
        "model_state_dict": best_state,
        "model_type": "residual_causal_gru",
        "input_history": INPUT_HISTORY,
        "prediction_horizon": PREDICTION_HORIZON,
        "r8_schema_version": R8_SCHEMA_VERSION,
        "r8_update_semantics": semantics,
        "r8_training_config": dict(config),
        "r8_residual_bound_rad": [float(value) for value in residual_bound] if residual_bound is not None else None,
        "best_validation_epoch": best_epoch,
    }, buffer)
    return model, history, best_epoch, buffer.getvalue()
