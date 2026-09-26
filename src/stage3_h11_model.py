"""Deterministic causal-GRU model and auditable metrics for H11."""

from __future__ import annotations

import csv
import io
import math
import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

try:  # Keep contract/audit tests importable when PyTorch is absent.
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - exercised in dependency-blocked environments
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    TensorDataset = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


MODEL_TYPE = "causal_gru"
HISTORY = 16
HORIZON = 8
JOINTS = 6
INPUT_FEATURES = 20


def set_deterministic(seed: int) -> dict[str, Any]:
    """Set all requested RNGs and PyTorch deterministic switches."""
    random.seed(seed)
    np.random.seed(seed)
    record: dict[str, Any] = {
        "python_seed": seed, "numpy_seed": seed, "pytorch_cpu_seed": seed, "pytorch_cuda_seed": seed,
        "deterministic_algorithms": True, "cudnn_deterministic": True, "cudnn_benchmark": False,
    }
    if not TORCH_AVAILABLE:
        record.update({"torch_available": False, "cuda_available": False})
        return record
    assert torch is not None
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    record.update({"torch_available": True, "cuda_available": bool(torch.cuda.is_available()), "device": "cuda" if torch.cuda.is_available() else "cpu"})
    return record


if TORCH_AVAILABLE:

    class CausalGRUTrajectoryPredictor(nn.Module):
        """Unidirectional GRU: history only, future never enters the input."""

        def __init__(self, input_size: int = INPUT_FEATURES, hidden_size: int = 128, num_layers: int = 2, horizon: int = HORIZON) -> None:
            super().__init__()
            self.input_size = int(input_size)
            self.hidden_size = int(hidden_size)
            self.num_layers = int(num_layers)
            self.horizon = int(horizon)
            self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=False)
            self.head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, horizon * JOINTS))

        def forward(self, inputs: "torch.Tensor") -> "torch.Tensor":
            sequence, _ = self.gru(inputs)
            last = sequence[:, -1, :]
            return self.head(last).reshape(inputs.shape[0], self.horizon, JOINTS)

else:

    class CausalGRUTrajectoryPredictor:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("pytorch_unavailable")


@dataclass
class WindowArrays:
    inputs: np.ndarray
    target_deltas: np.ndarray
    target_positions: np.ndarray
    anchor_positions: np.ndarray
    target_times: np.ndarray
    history_positions: np.ndarray
    history_times: np.ndarray
    window_ids: np.ndarray
    family_ids: np.ndarray


def parameter_count(model: Any) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _safe_rmse(error: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(error), dtype=np.float64))) if error.size else float("nan")


def metric_payload(predicted_positions: np.ndarray, target_positions: np.ndarray) -> dict[str, Any]:
    error = np.asarray(predicted_positions, dtype=np.float64) - np.asarray(target_positions, dtype=np.float64)
    absolute = np.abs(error)
    flat = absolute.reshape(-1)
    return {
        "window_count": int(error.shape[0]), "horizon": int(error.shape[1]) if error.ndim >= 2 else None,
        "joint_position_mae_rad": float(np.mean(absolute, dtype=np.float64)) if flat.size else None,
        "joint_position_rmse_rad": _safe_rmse(error) if flat.size else None,
        "mae_per_joint_rad": [float(value) for value in np.mean(absolute, axis=(0, 1))] if flat.size else [None] * JOINTS,
        "rmse_per_joint_rad": [float(np.sqrt(np.mean(np.square(error[:, :, joint]), dtype=np.float64))) for joint in range(JOINTS)] if flat.size else [None] * JOINTS,
        "maximum_absolute_error_rad": float(np.max(absolute)) if flat.size else None,
        "p50_absolute_error_rad": float(np.percentile(flat, 50)) if flat.size else None,
        "p95_absolute_error_rad": float(np.percentile(flat, 95)) if flat.size else None,
        "p99_absolute_error_rad": float(np.percentile(flat, 99)) if flat.size else None,
    }


def evaluate_model(model: Any, arrays: WindowArrays, device: str = "cpu", batch_size: int = 1024) -> tuple[np.ndarray, dict[str, Any]]:
    if not TORCH_AVAILABLE:
        raise RuntimeError("pytorch_unavailable")
    assert torch is not None
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(arrays.inputs), batch_size):
            batch = torch.from_numpy(arrays.inputs[start:start + batch_size]).to(device=device, dtype=torch.float32)
            outputs.append(model(batch).detach().cpu().numpy())
    deltas = np.concatenate(outputs, axis=0) if outputs else np.empty((0, HORIZON, JOINTS), dtype=np.float32)
    predicted = arrays.anchor_positions[:, None, :] + deltas
    return predicted, metric_payload(predicted, arrays.target_positions)


def hold_last_prediction(arrays: WindowArrays) -> np.ndarray:
    return np.repeat(arrays.anchor_positions[:, None, :], HORIZON, axis=1)


def constant_velocity_prediction(arrays: WindowArrays) -> np.ndarray:
    if len(arrays.history_positions) == 0:
        return np.empty((0, HORIZON, JOINTS), dtype=np.float64)
    dt = np.diff(arrays.history_times, axis=1)
    valid = np.where(dt > 0.0, dt, np.nan)
    velocity = np.diff(arrays.history_positions, axis=1) / valid[:, :, None]
    velocity = np.nan_to_num(velocity, nan=0.0, posinf=0.0, neginf=0.0)
    anchor_velocity = velocity[:, -1, :]
    elapsed = arrays.target_times - arrays.history_times[:, -1, None]
    return arrays.anchor_positions[:, None, :] + anchor_velocity[:, None, :] * elapsed[:, :, None]


def rollout_one_step(model: Any, arrays: WindowArrays, feature_stats: Mapping[str, Mapping[str, Any]], device: str = "cpu", batch_size: int = 512) -> np.ndarray:
    """Open-loop recursive rollout using only predicted positions after step 1.

    Velocity and acceleration channels for the next input are finite
    differences of the reconstructed prediction.  They are diagnostics, not
    native controller feedback.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("pytorch_unavailable")
    assert torch is not None
    model.eval()
    result = np.empty((len(arrays.inputs), HORIZON, JOINTS), dtype=np.float64)
    for batch_start in range(0, len(arrays.inputs), batch_size):
        batch_end = min(len(arrays.inputs), batch_start + batch_size)
        x = arrays.inputs[batch_start:batch_end].astype(np.float64, copy=True)
        positions = arrays.history_positions[batch_start:batch_end].astype(np.float64, copy=True)
        times = arrays.history_times[batch_start:batch_end].astype(np.float64, copy=True)
        last_time = times[:, -1].copy()
        for step in range(HORIZON):
            with torch.no_grad():
                pred_delta = model(torch.from_numpy(x.astype(np.float32)).to(device=device)).detach().cpu().numpy()[:, 0, :]
            next_time = arrays.target_times[batch_start:batch_end, step]
            next_position = positions[:, -1, :] + pred_delta
            result[batch_start:batch_end, step, :] = next_position
            dt = np.maximum(next_time - last_time, 1.0e-9)
            next_velocity = (next_position - positions[:, -1, :]) / dt[:, None]
            if positions.shape[1] >= 2:
                prior_dt = np.maximum(last_time - times[:, -2], 1.0e-9)
                prior_velocity = (positions[:, -1, :] - positions[:, -2, :]) / prior_dt[:, None]
            else:
                prior_velocity = next_velocity
            next_acceleration = (next_velocity - prior_velocity) / dt[:, None]
            next_features = np.column_stack([next_position, next_velocity, next_acceleration, x[:, -1, 18], (np.asarray(next_time) - arrays.history_times[batch_start:batch_end, 0]) / np.maximum(arrays.history_times[batch_start:batch_end, -1] - arrays.history_times[batch_start:batch_end, 0], 1.0e-9)])
            for index, name in enumerate((
                [f"planned_joint_position_{i + 1}" for i in range(JOINTS)]
                + [f"planned_joint_velocity_{i + 1}" for i in range(JOINTS)]
                + [f"planned_joint_acceleration_{i + 1}" for i in range(JOINTS)]
                + ["spray_on", "normalized_local_trajectory_time"]
            )):
                if name not in feature_stats:
                    continue
                scale = float(feature_stats[name].get("scale", feature_stats[name].get("std", 1.0))) or 1.0
                next_features[:, index] = (next_features[:, index] - float(feature_stats[name]["mean"])) / scale
            x = np.concatenate([x[:, 1:, :], next_features[:, None, :]], axis=1)
            positions = np.concatenate([positions[:, 1:, :], next_position[:, None, :]], axis=1)
            times = np.concatenate([times[:, 1:], next_time[:, None]], axis=1)
            last_time = next_time
    return result


def finite_difference_diagnostics(predicted_positions: np.ndarray, target_positions: np.ndarray, times: np.ndarray) -> dict[str, Any]:
    """Derive velocity/acceleration/jerk errors from predicted positions."""
    def derive(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dt = np.diff(times, axis=1)
        dt = np.maximum(dt, 1.0e-9)
        velocity = np.diff(positions, axis=1) / dt[:, :, None]
        acceleration = np.diff(velocity, axis=1) / np.maximum(dt[:, 1:, None], 1.0e-9) if positions.shape[1] >= 3 else np.empty((len(positions), 0, JOINTS))
        jerk = np.diff(acceleration, axis=1) / np.maximum(dt[:, 2:, None], 1.0e-9) if acceleration.shape[1] >= 2 else np.empty((len(positions), 0, JOINTS))
        return velocity, acceleration, jerk
    pred_v, pred_a, pred_j = derive(predicted_positions)
    true_v, true_a, true_j = derive(target_positions)
    def diag(pred: np.ndarray, true: np.ndarray) -> dict[str, Any]:
        if pred.size == 0:
            return {"mae": None, "rmse": None, "max_abs": None, "sample_count": 0}
        err = np.asarray(pred - true, dtype=np.float64)
        return {"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(np.square(err)))), "max_abs": float(np.max(np.abs(err))), "sample_count": int(err.shape[0] * err.shape[1])}
    return {"velocity_error_rad_s": diag(pred_v, true_v), "acceleration_error_rad_s2": diag(pred_a, true_a), "jerk_diagnostic_rad_s3": diag(pred_j, true_j), "method": "finite_difference_of_reconstructed_positions; not_native_controller_feedback"}


def train_model(model: Any, train_arrays: WindowArrays, validation_arrays: WindowArrays, config: Mapping[str, Any], device: str = "cpu") -> tuple[Any, list[dict[str, Any]], int, bytes]:
    if not TORCH_AVAILABLE:
        raise RuntimeError("pytorch_unavailable")
    assert torch is not None and DataLoader is not None and TensorDataset is not None
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config.get("weight_decay", 0.0)))
    loss_fn = nn.MSELoss()
    train_ds = TensorDataset(torch.from_numpy(train_arrays.inputs), torch.from_numpy(train_arrays.target_deltas))
    train_loader = DataLoader(train_ds, batch_size=int(config["batch_size"]), shuffle=False, num_workers=0, drop_last=False)
    model.to(device)
    best_state: dict[str, Any] | None = None
    best_val = math.inf
    best_epoch = 0
    patience = int(config["patience"])
    no_improvement = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(config["max_epochs"]) + 1):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device=device, dtype=torch.float32)
            targets = targets.to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs)
            loss = loss_fn(prediction, targets)
            if not torch.isfinite(loss):
                raise RuntimeError("training_nonfinite")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(inputs)
            sample_count += len(inputs)
        train_loss = loss_sum / max(sample_count, 1)
        _, validation_metrics = evaluate_model(model, validation_arrays, device=device, batch_size=int(config.get("eval_batch_size", 1024)))
        val_rmse = float(validation_metrics["joint_position_rmse_rad"])
        history.append({"epoch": epoch, "train_loss_mse_rad2": train_loss, "validation_joint_position_rmse_rad": val_rmse, "validation_window_count": len(validation_arrays.inputs)})
        if not math.isfinite(val_rmse):
            raise RuntimeError("training_nonfinite")
        if val_rmse < best_val - float(config.get("min_delta", 0.0)):
            best_val = val_rmse
            best_epoch = epoch
            no_improvement = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            no_improvement += 1
        if no_improvement >= patience:
            break
    if best_state is None:
        raise RuntimeError("validation_checkpoint_not_selected")
    model.load_state_dict(best_state)
    buffer = io.BytesIO()
    torch.save({"model_state_dict": best_state, "model_type": MODEL_TYPE, "input_history": HISTORY, "prediction_horizon": HORIZON}, buffer)
    return model, history, best_epoch, buffer.getvalue()


def write_history_csv(path: Any, history: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = ["epoch", "train_loss_mse_rad2", "validation_joint_position_rmse_rad", "validation_window_count"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: item.get(key) for key in fieldnames} for item in history)

