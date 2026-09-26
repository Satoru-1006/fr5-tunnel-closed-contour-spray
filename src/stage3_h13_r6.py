"""Pure rollout-aware training helpers for Stage 3 H13-R6.

The R6 runner owns orchestration and frozen-20 evaluation.  This module keeps
the new training semantics small and testable: train/validation windows are
constructed from the frozen H10 trajectory segments, the model is exposed to
bounded self-fed histories, and the validation selector is rollout-primary.
No H13 case or native certification code is imported here.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from src.stage3_h11_dataset import FEATURE_NAMES, INPUT_HISTORY, PREDICTION_HORIZON, _feature_matrix
from src.stage3_h11_model import WindowArrays, metric_payload
from src.stage3_h11_r2_model import JOINTS, SPRAY_FEATURE_INDEX, causal_constant_velocity_baseline, direct_metrics

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - contract tests remain importable without torch
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    TensorDataset = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


ROLLOUT_HORIZONS = (4, 8, 16, 32)
R6_SCHEMA_VERSION = "stage3_h13_r6_rollout_aware_training_v1"


@dataclass
class R6SequenceArrays:
    """Windows with an 8-step direct target and a bounded 32-step target."""

    inputs: np.ndarray
    direct_target_positions: np.ndarray
    direct_target_times: np.ndarray
    rollout_target_positions: np.ndarray
    rollout_target_times: np.ndarray
    history_positions: np.ndarray
    history_times: np.ndarray
    trajectory_start_times: np.ndarray
    trajectory_end_times: np.ndarray
    window_ids: np.ndarray
    family_ids: np.ndarray

    @property
    def count(self) -> int:
        return int(len(self.inputs))


def supported_rollout_horizons(
    segments: Sequence[Any],
    role_by_family: Mapping[str, str],
    role: str,
    requested: Sequence[int] = ROLLOUT_HORIZONS,
) -> tuple[int, ...]:
    """Return horizons supported by at least one native train/validation segment."""

    result = []
    for horizon in requested:
        if any(
            role_by_family.get(str(segment.family_id)) == role
            and len(segment.positions) >= INPUT_HISTORY + int(horizon)
            for segment in segments
        ):
            result.append(int(horizon))
    return tuple(result)


def build_sequence_arrays(
    dataset: Any,
    stats: Mapping[str, Any],
    role: str,
    *,
    max_rollout_horizon: int = max(ROLLOUT_HORIZONS),
) -> R6SequenceArrays:
    """Materialize only train or validation windows; never writes a dataset copy."""

    if int(max_rollout_horizon) < PREDICTION_HORIZON:
        raise ValueError("r6_rollout_horizon_must_cover_direct_horizon")
    inputs: list[np.ndarray] = []
    direct_targets: list[np.ndarray] = []
    direct_times: list[np.ndarray] = []
    rollout_targets: list[np.ndarray] = []
    rollout_times: list[np.ndarray] = []
    histories: list[np.ndarray] = []
    history_times: list[np.ndarray] = []
    starts: list[float] = []
    ends: list[float] = []
    window_ids: list[int] = []
    families: list[str] = []
    next_window_id = 0
    for segment in dataset.segments:
        if dataset.split_map.get(segment.family_id) != role:
            continue
        max_start = len(segment.positions) - INPUT_HISTORY - int(max_rollout_horizon) + 1
        if max_start <= 0:
            continue
        features = _feature_matrix(segment, stats["channels"])
        for start in range(max_start):
            direct_start = start + INPUT_HISTORY
            rollout_end = direct_start + int(max_rollout_horizon)
            inputs.append(features[start:direct_start])
            direct_targets.append(segment.positions[direct_start:direct_start + PREDICTION_HORIZON])
            direct_times.append(segment.times[direct_start:direct_start + PREDICTION_HORIZON])
            rollout_targets.append(segment.positions[direct_start:rollout_end])
            rollout_times.append(segment.times[direct_start:rollout_end])
            histories.append(segment.positions[start:direct_start])
            history_times.append(segment.times[start:direct_start])
            starts.append(float(segment.times[0]))
            ends.append(float(segment.times[-1]))
            window_ids.append(next_window_id)
            families.append(str(segment.family_id))
            next_window_id += 1
    empty = np.empty
    return R6SequenceArrays(
        inputs=np.asarray(inputs, dtype=np.float32) if inputs else empty((0, INPUT_HISTORY, len(FEATURE_NAMES)), dtype=np.float32),
        direct_target_positions=np.asarray(direct_targets, dtype=np.float64) if direct_targets else empty((0, PREDICTION_HORIZON, JOINTS), dtype=np.float64),
        direct_target_times=np.asarray(direct_times, dtype=np.float64) if direct_times else empty((0, PREDICTION_HORIZON), dtype=np.float64),
        rollout_target_positions=np.asarray(rollout_targets, dtype=np.float64) if rollout_targets else empty((0, max_rollout_horizon, JOINTS), dtype=np.float64),
        rollout_target_times=np.asarray(rollout_times, dtype=np.float64) if rollout_times else empty((0, max_rollout_horizon), dtype=np.float64),
        history_positions=np.asarray(histories, dtype=np.float64) if histories else empty((0, INPUT_HISTORY, JOINTS), dtype=np.float64),
        history_times=np.asarray(history_times, dtype=np.float64) if history_times else empty((0, INPUT_HISTORY), dtype=np.float64),
        trajectory_start_times=np.asarray(starts, dtype=np.float64),
        trajectory_end_times=np.asarray(ends, dtype=np.float64),
        window_ids=np.asarray(window_ids, dtype=np.int64),
        family_ids=np.asarray(families, dtype=object),
    )


def as_window_arrays(data: R6SequenceArrays) -> WindowArrays:
    """Adapt the direct 8-step view to the existing H11-R2 metric helpers."""

    anchors = data.history_positions[:, -1, :] if data.count else np.empty((0, JOINTS), dtype=np.float64)
    return WindowArrays(
        inputs=data.inputs,
        target_deltas=(data.direct_target_positions - anchors[:, None, :]).astype(np.float32),
        target_positions=data.direct_target_positions,
        anchor_positions=anchors,
        target_times=data.direct_target_times,
        history_positions=data.history_positions,
        history_times=data.history_times,
        window_ids=data.window_ids,
        family_ids=data.family_ids,
    )


def direct_residual_targets(data: R6SequenceArrays) -> np.ndarray:
    """Compute direct residual labels from history-only causal baselines."""

    baseline = causal_constant_velocity_baseline(as_window_arrays(data))
    return (data.direct_target_positions - baseline).astype(np.float32)


def horizon_for_epoch(epoch: int, max_epochs: int, horizons: Sequence[int]) -> int:
    """Predeclared bounded curriculum: 4 -> 8 -> 16 -> 32."""

    ordered = tuple(int(item) for item in horizons)
    if not ordered:
        raise ValueError("r6_horizon_ladder_empty")
    if epoch < 1 or max_epochs < 1:
        raise ValueError("r6_epoch_invalid")
    bucket = min(len(ordered) - 1, ((int(epoch) - 1) * len(ordered)) // int(max_epochs))
    return ordered[bucket]


def teacher_forcing_ratio(epoch: int, max_epochs: int, initial: float, final: float) -> float:
    """Linear, predeclared exposure schedule from teacher forcing to self feeding."""

    if max_epochs < 1 or epoch < 1:
        raise ValueError("r6_schedule_epoch_invalid")
    fraction = (int(epoch) - 1) / max(int(max_epochs) - 1, 1)
    return float(initial) + (float(final) - float(initial)) * fraction


def validation_selection_key(rollout_rmse: float, direct_rmse: float) -> tuple[float, float]:
    """Rollout is primary; direct/one-step quality is the secondary tie-break."""

    return (float(rollout_rmse), float(direct_rmse))


def checkpoint_hash_unchanged(before: str, after: str) -> bool:
    return bool(before) and before == after


def native_count_or_not_reached(value: Any, *, reached: bool) -> Any:
    """Never turn an unexecuted native stage into a zero count."""

    if not reached:
        return None
    return int(value or 0)


def _normalize(value: "torch.Tensor", index: int, channels: Mapping[str, Mapping[str, Any]], names: Sequence[str]) -> "torch.Tensor":
    name = names[index]
    if name == "spray_on" or name not in channels:
        return value
    item = channels[name]
    scale = float(item.get("scale", item.get("std", 1.0)))
    if not math.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    return (value - float(item.get("mean", 0.0))) / scale


def _next_features(
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
    dt = torch.clamp(next_time - last_time, min=1.0e-9)
    prior_dt = torch.clamp(last_time - prior_time, min=1.0e-9)
    velocity = (next_position - previous_position) / dt[:, None]
    prior_velocity = (previous_position - prior_position) / prior_dt[:, None]
    acceleration = (velocity - prior_velocity) / dt[:, None]
    local_time = (next_time - trajectory_start) / torch.clamp(trajectory_end - trajectory_start, min=1.0e-9)
    raw = tuple(next_position.T) + tuple(velocity.T) + tuple(acceleration.T) + (spray, local_time)
    return torch.stack([_normalize(value, index, channels, feature_names) for index, value in enumerate(raw)], dim=1)


def rollout_residual_torch_r6(
    model: Any,
    inputs: "torch.Tensor",
    history_positions: "torch.Tensor",
    history_times: "torch.Tensor",
    target_times: "torch.Tensor",
    trajectory_start_times: "torch.Tensor",
    trajectory_end_times: "torch.Tensor",
    channels: Mapping[str, Mapping[str, Any]],
    feature_names: Sequence[str] = FEATURE_NAMES,
    rollout_horizon: int = 32,
    teacher_forcing_ratio_value: float = 0.0,
    target_positions: "torch.Tensor | None" = None,
    sample_random: bool = True,
    detach_state: bool = False,
) -> "torch.Tensor":
    """Run a bounded causal rollout; targets can only affect the next history."""

    if not TORCH_AVAILABLE:
        raise RuntimeError("pytorch_unavailable")
    if rollout_horizon < 1 or rollout_horizon > target_times.shape[1]:
        raise ValueError("r6_invalid_rollout_horizon")
    current_inputs = inputs
    positions = history_positions
    times = history_times
    predictions = []
    for step in range(int(rollout_horizon)):
        residual = model(current_inputs)[:, 0, :]
        last = positions[:, -1, :]
        prior = positions[:, -2, :]
        last_time = times[:, -1]
        prior_time = times[:, -2]
        dt = torch.clamp(target_times[:, step] - last_time, min=1.0e-9)
        velocity = (last - prior) / torch.clamp(last_time - prior_time, min=1.0e-9)[:, None]
        next_position = last + velocity * dt[:, None] + residual
        predictions.append(next_position)
        if step + 1 == int(rollout_horizon):
            continue
        if target_positions is not None and float(teacher_forcing_ratio_value) > 0.0:
            if sample_random:
                use_truth = torch.rand((len(next_position), 1), device=next_position.device) < float(teacher_forcing_ratio_value)
            else:
                use_truth = torch.full((len(next_position), 1), bool(teacher_forcing_ratio_value >= 0.5), device=next_position.device, dtype=torch.bool)
            next_history_position = torch.where(use_truth, target_positions[:, step, :], next_position)
        else:
            next_history_position = next_position
        if detach_state:
            # Truncated BPTT: the next causal state is still model-generated,
            # but the graph is not retained across the full 32-step horizon.
            next_history_position = next_history_position.detach()
        next_features = _next_features(
            next_history_position,
            last,
            prior,
            target_times[:, step],
            last_time,
            prior_time,
            trajectory_start_times,
            trajectory_end_times,
            current_inputs[:, -1, SPRAY_FEATURE_INDEX],
            channels,
            feature_names,
        ).to(dtype=current_inputs.dtype)
        current_inputs = torch.cat((current_inputs[:, 1:, :], next_features[:, None, :]), dim=1)
        positions = torch.cat((positions[:, 1:, :], next_history_position[:, None, :]), dim=1)
        times = torch.cat((times[:, 1:], target_times[:, step, None]), dim=1)
    return torch.stack(predictions, dim=1)


def evaluate_rollout_metrics(
    model: Any,
    data: R6SequenceArrays,
    channels: Mapping[str, Mapping[str, Any]],
    *,
    horizons: Sequence[int] = ROLLOUT_HORIZONS,
    device: str = "cpu",
    batch_size: int = 2048,
) -> dict[str, Any]:
    """Stream RMSEs without retaining per-window predictions."""

    if not TORCH_AVAILABLE:
        raise RuntimeError("pytorch_unavailable")
    assert torch is not None
    if data.count == 0:
        return {str(h): {"joint_position_rmse_rad": None, "window_count": 0} for h in horizons}
    model.eval()
    max_horizon = max(int(h) for h in horizons)
    sums = {int(h): 0.0 for h in horizons}
    counts = {int(h): 0 for h in horizons}
    with torch.no_grad():
        for start in range(0, data.count, int(batch_size)):
            end = min(data.count, start + int(batch_size))
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
                teacher_forcing_ratio_value=0.0,
                target_positions=None,
                sample_random=False,
                detach_state=False,
            )
            predicted = rolled.detach().cpu().numpy().astype(np.float64)
            target = data.rollout_target_positions[start:end]
            for horizon in horizons:
                h = int(horizon)
                error = predicted[:, :h, :] - target[:, :h, :]
                sums[h] += float(np.sum(np.square(error), dtype=np.float64))
                counts[h] += int(error.size)
    result = {}
    for horizon in horizons:
        h = int(horizon)
        result[str(h)] = {
            "joint_position_rmse_rad": float(math.sqrt(sums[h] / counts[h])) if counts[h] else None,
            "window_count": data.count,
            "horizon": h,
            "method": "free_running_causal_self_feeding_no_future_targets",
        }
    result["full_rollout"] = result[str(max_horizon)]
    return result


def train_rollout_aware_model(
    model: Any,
    train_data: R6SequenceArrays,
    validation_data: R6SequenceArrays,
    channels: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    device: str = "cpu",
) -> tuple[Any, list[dict[str, Any]], int, bytes]:
    """Train with direct teacher-forced loss plus bounded self-fed loss."""

    if not TORCH_AVAILABLE:
        raise RuntimeError("pytorch_unavailable")
    assert torch is not None and nn is not None and F is not None and DataLoader is not None and TensorDataset is not None
    if train_data.count == 0 or validation_data.count == 0:
        raise RuntimeError("r6_train_or_validation_windows_missing")
    horizons = tuple(int(h) for h in config["rollout_horizons"])
    direct_targets = torch.from_numpy(direct_residual_targets(train_data))
    indices = torch.arange(train_data.count, dtype=torch.long)
    loader = DataLoader(TensorDataset(torch.from_numpy(train_data.inputs), direct_targets, indices), batch_size=int(config["batch_size"]), shuffle=False, num_workers=0, drop_last=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config.get("weight_decay", 0.0)))
    model.to(device)
    best_state: dict[str, Any] | None = None
    best_key: tuple[float, float] | None = None
    best_epoch = 0
    no_improvement = 0
    history: list[dict[str, Any]] = []
    beta = float(config.get("huber_beta", 1.0e-3))
    for epoch in range(1, int(config["max_epochs"]) + 1):
        model.train()
        active_horizon = horizon_for_epoch(epoch, int(config["max_epochs"]), horizons)
        tf_ratio = teacher_forcing_ratio(epoch, int(config["max_epochs"]), float(config["teacher_forcing_initial_ratio"]), float(config["teacher_forcing_final_ratio"]))
        direct_sum = 0.0
        rollout_sum = 0.0
        sample_count = 0
        for batch_inputs, batch_target_residual, batch_indices in loader:
            batch_inputs = batch_inputs.to(device=device, dtype=torch.float32)
            batch_target_residual = batch_target_residual.to(device=device, dtype=torch.float32)
            numpy_indices = batch_indices.detach().cpu().numpy()
            optimizer.zero_grad(set_to_none=True)
            predicted_residual = model(batch_inputs)
            direct_loss = F.smooth_l1_loss(predicted_residual, batch_target_residual, beta=beta)
            hp = torch.from_numpy(train_data.history_positions[numpy_indices]).to(device=device, dtype=torch.float32)
            ht = torch.from_numpy(train_data.history_times[numpy_indices]).to(device=device, dtype=torch.float32)
            tt = torch.from_numpy(train_data.rollout_target_times[numpy_indices]).to(device=device, dtype=torch.float32)
            tp = torch.from_numpy(train_data.rollout_target_positions[numpy_indices]).to(device=device, dtype=torch.float32)
            starts = torch.from_numpy(train_data.trajectory_start_times[numpy_indices]).to(device=device, dtype=torch.float32)
            ends = torch.from_numpy(train_data.trajectory_end_times[numpy_indices]).to(device=device, dtype=torch.float32)
            rolled = rollout_residual_torch_r6(model, batch_inputs, hp, ht, tt, starts, ends, channels, rollout_horizon=active_horizon, teacher_forcing_ratio_value=tf_ratio, target_positions=tp, sample_random=True, detach_state=bool(config.get("truncated_bptt", False)))
            rollout_loss = F.smooth_l1_loss(rolled, tp[:, :active_horizon, :], beta=beta)
            loss = float(config["w_direct"]) * direct_loss + float(config["w_rollout"]) * rollout_loss
            if not torch.isfinite(loss):
                raise RuntimeError("r6_training_nonfinite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("gradient_clip_norm", 1.0)))
            optimizer.step()
            direct_sum += float(direct_loss.detach().cpu()) * len(batch_inputs)
            rollout_sum += float(rollout_loss.detach().cpu()) * len(batch_inputs)
            sample_count += len(batch_inputs)
        _, validation_direct = direct_metrics(model, as_window_arrays(validation_data), device=device, batch_size=int(config.get("eval_batch_size", 2048)))
        validation_rollout = evaluate_rollout_metrics(model, validation_data, channels, horizons=(max(horizons),), device=device, batch_size=int(config.get("eval_batch_size", 2048)))
        validation_rollout_rmse = float(validation_rollout["full_rollout"]["joint_position_rmse_rad"])
        validation_direct_rmse = float(validation_direct["joint_position_rmse_rad"])
        key = validation_selection_key(validation_rollout_rmse, validation_direct_rmse)
        history.append({
            "epoch": epoch,
            "active_rollout_horizon": active_horizon,
            "teacher_forcing_ratio": tf_ratio,
            "train_direct_loss": direct_sum / max(sample_count, 1),
            "train_rollout_loss": rollout_sum / max(sample_count, 1),
            "validation_direct_rmse_rad": validation_direct_rmse,
            "validation_autoregressive_rmse_rad": validation_rollout_rmse,
            "validation_selection_score": validation_rollout_rmse + float(config.get("direct_tie_weight", 0.1)) * validation_direct_rmse,
            "validation_selection_primary": "autoregressive_rmse",
        })
        if not (math.isfinite(validation_rollout_rmse) and math.isfinite(validation_direct_rmse)):
            raise RuntimeError("r6_validation_nonfinite")
        if best_key is None or key < best_key:
            best_key = key
            best_epoch = epoch
            no_improvement = 0
            best_state = {key_name: value.detach().cpu().clone() for key_name, value in model.state_dict().items()}
        else:
            no_improvement += 1
        if no_improvement >= int(config["patience"]):
            break
    if best_state is None:
        raise RuntimeError("r6_validation_checkpoint_not_selected")
    model.load_state_dict(best_state, strict=True)
    buffer = io.BytesIO()
    torch.save({
        "model_state_dict": best_state,
        "model_type": "residual_causal_gru",
        "input_history": INPUT_HISTORY,
        "prediction_horizon": PREDICTION_HORIZON,
        "r6_schema_version": R6_SCHEMA_VERSION,
        "r6_training_config": dict(config),
        "best_validation_epoch": best_epoch,
    }, buffer)
    return model, history, best_epoch, buffer.getvalue()
