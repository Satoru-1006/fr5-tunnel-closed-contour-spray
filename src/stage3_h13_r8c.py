"""Rollout-aware self-feeding training primitives for Stage 3 H13-R8C.

R8C keeps the authoritative R6 residual-Causal-GRU architecture and changes
only the training objective/schedule.  The helpers in this module deliberately
operate on already materialized TRAIN or VALIDATION windows.  They never read
future labels while producing a current prediction; labels are used only for
the rollout loss and for the explicitly bounded scheduled-sampling update of
the *next* causal history during training.
"""

from __future__ import annotations

import io
import math
from typing import Any, Mapping, Sequence

import numpy as np

from src.stage3_h11_model import metric_payload
from src.stage3_h13_r6 import (
    R6SequenceArrays,
    as_window_arrays,
    direct_residual_targets,
    rollout_residual_torch_r6,
)
from src.stage3_h11_r2_model import causal_constant_velocity_baseline

try:
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - contract tests remain importable without torch
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    TensorDataset = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


R8C_SCHEMA_VERSION = "stage3_h13_r8c_rollout_aware_self_feeding_v1"
R8C_EVALUATION_HORIZONS = (1, 4, 8, 12, 16, 20, 24, 32)
R8C_TRAINING_HORIZONS = (1, 4, 8, 16, 32)


def curriculum_state(epoch: int, max_epochs: int) -> tuple[int, float, str]:
    """Return a deterministic bounded curriculum state.

    R6 is already a strong teacher-forced model, so R8C starts with a short
    stabilisation phase and reaches fully self-fed H32 training early.  The
    schedule is predeclared and contains no validation-driven schedule search.
    """

    if int(epoch) < 1 or int(max_epochs) < 1 or int(epoch) > int(max_epochs):
        raise ValueError("r8c_curriculum_epoch_invalid")
    fraction = (int(epoch) - 1) / max(int(max_epochs) - 1, 1)
    if fraction < 0.20:
        return 4, 0.75, "A_short_rollout_teacher_forcing_dominant"
    if fraction < 0.40:
        return 8, 0.50, "B_h4_h8_mixed_self_feeding"
    if fraction < 0.60:
        return 16, 0.25, "C_h16_self_feeding_dominant"
    return 32, 0.0, "D_h32_fully_self_fed"


def horizon_loss_weights(horizons: Sequence[int] = R8C_TRAINING_HORIZONS) -> dict[int, float]:
    """Predeclared weights that materially preserve the long rollout."""

    requested = tuple(int(value) for value in horizons)
    reference = {1: 0.05, 4: 0.10, 8: 0.15, 16: 0.25, 32: 0.45}
    if not requested or any(value not in reference for value in requested):
        raise ValueError("r8c_unknown_training_horizon")
    total = sum(reference[value] for value in requested)
    return {value: reference[value] / total for value in requested}


def rollout_profile(
    model: Any,
    data: R6SequenceArrays,
    channels: Mapping[str, Mapping[str, Any]],
    *,
    teacher_forcing_ratio: float = 0.0,
    horizons: Sequence[int] = R8C_EVALUATION_HORIZONS,
    device: str = "cpu",
    batch_size: int = 4096,
) -> dict[str, Any]:
    """Evaluate causal rollouts in bounded batches.

    ``teacher_forcing_ratio=0`` is the authoritative free-running mode.  A
    ratio of 1 is a diagnostic only and is never used for R8C model selection.
    """

    if not TORCH_AVAILABLE:
        raise RuntimeError("pytorch_unavailable")
    assert torch is not None
    requested = tuple(int(value) for value in horizons)
    if not requested or min(requested) < 1 or max(requested) > 32:
        raise ValueError("r8c_invalid_evaluation_horizons")
    if data.count == 0:
        return {str(h): {"joint_position_rmse_rad": None, "window_count": 0, "horizon": h} for h in requested}
    max_horizon = max(requested)
    sums = {h: 0.0 for h in requested}
    counts = {h: 0 for h in requested}
    model.eval()
    with torch.no_grad():
        for start in range(0, data.count, int(batch_size)):
            end = min(data.count, start + int(batch_size))
            target_positions = torch.from_numpy(data.rollout_target_positions[start:end]).to(device=device, dtype=torch.float32)
            rolled = rollout_residual_torch_r6(
                model,
                torch.from_numpy(data.inputs[start:end]).to(device=device, dtype=torch.float32),
                torch.from_numpy(data.history_positions[start:end]).to(device=device, dtype=torch.float32),
                torch.from_numpy(data.history_times[start:end]).to(device=device, dtype=torch.float32),
                torch.from_numpy(data.rollout_target_times[start:end]).to(device=device, dtype=torch.float32),
                torch.from_numpy(data.trajectory_start_times[start:end]).to(device=device, dtype=torch.float32),
                torch.from_numpy(data.trajectory_end_times[start:end]).to(device=device, dtype=torch.float32),
                channels,
                rollout_horizon=max_horizon,
                teacher_forcing_ratio_value=float(teacher_forcing_ratio),
                target_positions=target_positions if float(teacher_forcing_ratio) > 0.0 else None,
                sample_random=False,
                detach_state=False,
            )
            error = rolled.detach().cpu().numpy().astype(np.float64) - data.rollout_target_positions[start:end, :max_horizon, :]
            for horizon in requested:
                partial = error[:, :horizon, :]
                sums[horizon] += float(np.sum(np.square(partial), dtype=np.float64))
                counts[horizon] += int(partial.size)
    result: dict[str, Any] = {}
    mode = "free_running_causal_self_feeding_no_future_targets" if float(teacher_forcing_ratio) == 0.0 else "teacher_forced_diagnostic_ground_truth_next_history"
    for horizon in requested:
        result[str(horizon)] = {
            "joint_position_rmse_rad": float(math.sqrt(sums[horizon] / counts[horizon])) if counts[horizon] else None,
            "window_count": data.count,
            "horizon": horizon,
            "method": mode,
        }
    result["full_rollout"] = result[str(max_horizon)]
    return result


def _multi_horizon_loss(
    rolled: "torch.Tensor",
    target_positions: "torch.Tensor",
    active_horizon: int,
    beta: float,
    weights: Mapping[int, float],
    loss_kind: str,
) -> "torch.Tensor":
    assert F is not None
    terms = []
    for horizon, weight in weights.items():
        if int(horizon) > int(active_horizon):
            continue
        prediction = rolled[:, :int(horizon), :]
        target = target_positions[:, :int(horizon), :]
        if loss_kind == "mse":
            component = F.mse_loss(prediction, target)
        elif loss_kind == "huber":
            component = F.smooth_l1_loss(prediction, target, beta=float(beta))
        else:
            raise ValueError("r8c_unknown_rollout_loss_kind")
        terms.append(float(weight) * component)
    if not terms:
        raise RuntimeError("r8c_no_active_rollout_loss_terms")
    return torch.stack(terms).sum()


def train_r8c_candidate(
    initial_state: Mapping[str, Any],
    model_factory: Any,
    train_data: R6SequenceArrays,
    validation_data: R6SequenceArrays,
    channels: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    device: str = "cpu",
) -> tuple[Any, list[dict[str, Any]], int, bytes]:
    """Train one bounded candidate and select only on validation free H32."""

    if not TORCH_AVAILABLE:
        raise RuntimeError("pytorch_unavailable")
    assert torch is not None and F is not None and DataLoader is not None and TensorDataset is not None
    model = model_factory()
    model.load_state_dict(initial_state, strict=True)
    model.to(device)
    model.train()
    train_stride = max(1, int(config.get("train_window_stride", 1)))
    sampled_indices = np.arange(0, train_data.count, train_stride, dtype=np.int64)
    train_targets = torch.from_numpy(direct_residual_targets(train_data)[sampled_indices])
    indices = torch.from_numpy(sampled_indices)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_data.inputs[sampled_indices]), train_targets, indices),
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )
    weights = horizon_loss_weights(config.get("training_horizons", R8C_TRAINING_HORIZONS))
    beta = float(config.get("huber_beta", 1.0e-3))
    best_state: dict[str, Any] | None = None
    best_key: tuple[float, float] | None = None
    best_epoch = 0
    no_improvement = 0
    history: list[dict[str, Any]] = []
    max_epochs = int(config["max_epochs"])
    for epoch in range(1, max_epochs + 1):
        active_horizon, tf_ratio, phase = curriculum_state(epoch, max_epochs)
        model.train()
        direct_sum = 0.0
        rollout_sum = 0.0
        total_sum = 0.0
        sample_count = 0
        for batch_inputs, batch_target_residual, batch_indices in loader:
            batch_inputs = batch_inputs.to(device=device, dtype=torch.float32)
            batch_target_residual = batch_target_residual.to(device=device, dtype=torch.float32)
            numpy_indices = batch_indices.detach().cpu().numpy()
            optimizer.zero_grad(set_to_none=True)
            direct_prediction = model(batch_inputs)
            direct_loss = F.smooth_l1_loss(direct_prediction, batch_target_residual, beta=beta)
            hp = torch.from_numpy(train_data.history_positions[numpy_indices]).to(device=device, dtype=torch.float32)
            ht = torch.from_numpy(train_data.history_times[numpy_indices]).to(device=device, dtype=torch.float32)
            tt = torch.from_numpy(train_data.rollout_target_times[numpy_indices]).to(device=device, dtype=torch.float32)
            tp = torch.from_numpy(train_data.rollout_target_positions[numpy_indices]).to(device=device, dtype=torch.float32)
            starts = torch.from_numpy(train_data.trajectory_start_times[numpy_indices]).to(device=device, dtype=torch.float32)
            ends = torch.from_numpy(train_data.trajectory_end_times[numpy_indices]).to(device=device, dtype=torch.float32)
            rolled = rollout_residual_torch_r6(
                model,
                batch_inputs,
                hp,
                ht,
                tt,
                starts,
                ends,
                channels,
                rollout_horizon=active_horizon,
                teacher_forcing_ratio_value=tf_ratio,
                target_positions=tp,
                sample_random=True,
                detach_state=bool(config.get("truncated_bptt", True)),
            )
            rollout_loss = _multi_horizon_loss(rolled, tp, active_horizon, beta, weights, str(config.get("rollout_loss", "mse")))
            total_loss = float(config["w_direct"]) * direct_loss + float(config["w_rollout"]) * rollout_loss
            if not torch.isfinite(total_loss):
                raise RuntimeError("r8c_training_nonfinite")
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("gradient_clip_norm", 0.5)))
            optimizer.step()
            batch_size = len(batch_inputs)
            direct_sum += float(direct_loss.detach().cpu()) * batch_size
            rollout_sum += float(rollout_loss.detach().cpu()) * batch_size
            total_sum += float(total_loss.detach().cpu()) * batch_size
            sample_count += batch_size
        validation_free = rollout_profile(model, validation_data, channels, teacher_forcing_ratio=0.0, horizons=R8C_EVALUATION_HORIZONS, device=device, batch_size=int(config["eval_batch_size"]))
        h32 = float(validation_free["32"]["joint_position_rmse_rad"])
        h1 = float(validation_free["1"]["joint_position_rmse_rad"])
        direct_state = model.eval()
        with torch.no_grad():
            direct_outputs = []
            for start in range(0, validation_data.count, int(config["eval_batch_size"])):
                end = min(validation_data.count, start + int(config["eval_batch_size"]))
                direct_outputs.append(model(torch.from_numpy(validation_data.inputs[start:end]).to(device=device, dtype=torch.float32)).detach().cpu().numpy())
        predicted_direct = np.concatenate(direct_outputs, axis=0) if direct_outputs else np.empty_like(validation_data.direct_target_positions)
        baseline = causal_constant_velocity_baseline(as_window_arrays(validation_data))
        direct_metric = metric_payload(baseline + predicted_direct, validation_data.direct_target_positions)
        history.append({
            "epoch": epoch,
            "active_rollout_horizon": active_horizon,
            "teacher_forcing_ratio": tf_ratio,
            "curriculum_phase": phase,
            "train_window_count": int(len(sampled_indices)),
            "train_window_stride": train_stride,
            "train_direct_loss": direct_sum / max(sample_count, 1),
            "train_rollout_loss": rollout_sum / max(sample_count, 1),
            "train_total_loss": total_sum / max(sample_count, 1),
            "validation_free_running_h1_rmse_rad": h1,
            "validation_free_running_h4_rmse_rad": validation_free["4"]["joint_position_rmse_rad"],
            "validation_free_running_h8_rmse_rad": validation_free["8"]["joint_position_rmse_rad"],
            "validation_free_running_h16_rmse_rad": validation_free["16"]["joint_position_rmse_rad"],
            "validation_free_running_h32_rmse_rad": h32,
            "validation_teacher_forced_h32_rmse_rad": None,
            "validation_direct_rmse_rad": direct_metric["joint_position_rmse_rad"],
            "validation_selection_primary": "free_running_h32_rmse",
        })
        key = (h32, float(direct_metric["joint_position_rmse_rad"]))
        if best_key is None or key < best_key:
            best_key = key
            best_epoch = epoch
            no_improvement = 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            no_improvement += 1
        if no_improvement >= int(config["patience"]):
            break
        del direct_state
    if best_state is None or best_key is None:
        raise RuntimeError("r8c_validation_checkpoint_not_selected")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    buffer = io.BytesIO()
    torch.save({
        "model_state_dict": best_state,
        "model_type": "residual_causal_gru",
        "stage": "H13-R8C",
        "input_history": 16,
        "prediction_horizon": 8,
        "selection_metric": "validation_free_running_h32_rmse",
        "selected_epoch": best_epoch,
    }, buffer)
    return model, history, best_epoch, buffer.getvalue()


def profile_to_rmse_map(profile: Mapping[str, Any]) -> dict[int, float | None]:
    return {int(key): (None if value.get("joint_position_rmse_rad") is None else float(value["joint_position_rmse_rad"])) for key, value in profile.items() if str(key).isdigit()}


def percent_improvement(baseline: float, candidate: float) -> float:
    if not math.isfinite(float(baseline)) or not math.isfinite(float(candidate)):
        raise ValueError("r8c_nonfinite_improvement")
    return (float(baseline) - float(candidate)) / float(baseline) * 100.0
