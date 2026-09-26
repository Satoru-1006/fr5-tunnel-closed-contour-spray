"""H12-R residual prediction and causal repair primitives.

The frozen H11 model predicts a position delta from the last observed joint
state.  H12-R evaluates a different, explicitly bounded hypothesis: the
constant-velocity (CV) trajectory is the base prediction and the network only
predicts a residual correction.  This module contains the model/training
helpers and the inference-time repair contract.  No target or future label is
accepted by the repair functions.
"""

from __future__ import annotations

import hashlib
import io
import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from src.stage3_h11_model import (
    HORIZON,
    JOINTS,
    TORCH_AVAILABLE,
    CausalGRUTrajectoryPredictor,
    WindowArrays,
    constant_velocity_prediction,
    evaluate_model,
    metric_payload,
    parameter_count,
)
from src.stage3_h12_trajectory_repair import RepairLimits

if TORCH_AVAILABLE:  # pragma: no cover - import guard is exercised in CI
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
else:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    TensorDataset = None  # type: ignore[assignment]


RESIDUAL_MODEL_TYPE = "causal_gru_residual"
RESIDUAL_REPAIR_METHOD = "causal_cv_residual_backtracking_v1"
BACKTRACK_FACTORS = tuple(0.5 ** index for index in range(0, 17)) + (0.0,)


if TORCH_AVAILABLE:

    class ResidualGRUTrajectoryPredictor(CausalGRUTrajectoryPredictor):
        """Same bounded H11-sized architecture, with a residual output head."""

        model_type = RESIDUAL_MODEL_TYPE

else:

    class ResidualGRUTrajectoryPredictor(CausalGRUTrajectoryPredictor):  # type: ignore[no-redef]
        pass


def residual_targets(arrays: WindowArrays) -> tuple[np.ndarray, np.ndarray]:
    """Return frozen CV positions and supervised residual labels.

    This function is used only for TRAIN/VALIDATION model selection and for
    TEST evaluation after the model is frozen.  It is never called from the
    causal repair boundary.
    """

    cv = constant_velocity_prediction(arrays)
    return cv, np.asarray(arrays.target_positions, dtype=np.float64) - cv


def predict_residual_model(model: Any, arrays: WindowArrays, cv_positions: np.ndarray, *, device: str = "cpu", batch_size: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    if not TORCH_AVAILABLE:
        raise RuntimeError("pytorch_unavailable")
    assert torch is not None
    model.eval()
    residuals: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(arrays.inputs), batch_size):
            batch = torch.from_numpy(arrays.inputs[start:start + batch_size]).to(device=device, dtype=torch.float32)
            residuals.append(model(batch).detach().cpu().numpy().astype(np.float64, copy=False))
    residual = np.concatenate(residuals, axis=0) if residuals else np.empty((0, HORIZON, JOINTS), dtype=np.float64)
    return np.asarray(cv_positions, dtype=np.float64) + residual, residual


def _hinge_penalty(q: "torch.Tensor", anchor: "torch.Tensor", history_end: "torch.Tensor", target_times: "torch.Tensor", limits: RepairLimits) -> tuple["torch.Tensor", dict[str, float]]:
    """Differentiable training surrogate; native certification remains separate."""

    assert torch is not None
    full_q = torch.cat([anchor[:, None, :], q], dim=1)
    full_t = torch.cat([history_end[:, None], target_times], dim=1)
    dt = torch.clamp(full_t[:, 1:] - full_t[:, :-1], min=1.0e-9)
    velocity = (full_q[:, 1:, :] - full_q[:, :-1, :]) / dt[:, :, None]
    acceleration = (velocity[:, 1:, :] - velocity[:, :-1, :]) / torch.clamp(dt[:, 1:, None], min=1.0e-9)
    jerk = (acceleration[:, 1:, :] - acceleration[:, :-1, :]) / torch.clamp(dt[:, 2:, None], min=1.0e-9)

    def excess(value: "torch.Tensor", bound: np.ndarray) -> "torch.Tensor":
        bound_tensor = torch.as_tensor(bound, dtype=value.dtype, device=value.device)
        return torch.relu(torch.abs(value) / bound_tensor - 1.0).square().mean()

    position_lower = torch.as_tensor(limits.position_lower, dtype=q.dtype, device=q.device)
    position_upper = torch.as_tensor(limits.position_upper, dtype=q.dtype, device=q.device)
    position_excess = (torch.relu(position_lower[None, None, :] - q).square() + torch.relu(q - position_upper[None, None, :]).square()).mean()
    values = {
        "position": position_excess,
        "velocity": excess(velocity, limits.velocity_abs),
        "acceleration": excess(acceleration, limits.acceleration_abs),
        "jerk": excess(jerk, limits.jerk_abs),
    }
    return sum(values.values()), {key: float(value.detach().cpu()) for key, value in values.items()}


def train_residual_model(
    model: Any,
    train_arrays: WindowArrays,
    validation_arrays: WindowArrays,
    train_cv: np.ndarray,
    validation_cv: np.ndarray,
    config: Mapping[str, Any],
    limits: RepairLimits,
    *,
    device: str = "cpu",
) -> tuple[Any, list[dict[str, Any]], int, bytes, dict[str, Any]]:
    """Train on TRAIN and choose the checkpoint using VALIDATION only."""

    if not TORCH_AVAILABLE:
        raise RuntimeError("pytorch_unavailable")
    assert torch is not None and nn is not None and DataLoader is not None and TensorDataset is not None
    train_target = np.asarray(train_arrays.target_positions, dtype=np.float64) - train_cv
    validation_target = np.asarray(validation_arrays.target_positions, dtype=np.float64) - validation_cv
    train_ds = TensorDataset(
        torch.from_numpy(train_arrays.inputs),
        torch.from_numpy(train_target.astype(np.float32)),
        torch.from_numpy(train_cv.astype(np.float32)),
        torch.from_numpy(train_arrays.anchor_positions.astype(np.float32)),
        torch.from_numpy(train_arrays.history_times[:, -1].astype(np.float32)),
        torch.from_numpy(train_arrays.target_times.astype(np.float32)),
    )
    loader = DataLoader(train_ds, batch_size=int(config["batch_size"]), shuffle=False, num_workers=0, drop_last=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config.get("weight_decay", 0.0)))
    loss_fn = nn.MSELoss()
    model.to(device)
    best_state: dict[str, Any] | None = None
    best_val = math.inf
    best_epoch = 0
    no_improvement = 0
    history: list[dict[str, Any]] = []
    penalty_weight = float(config.get("constraint_penalty_weight", 0.0))
    for epoch in range(1, int(config["max_epochs"]) + 1):
        model.train()
        loss_sum = 0.0
        prediction_sum = 0.0
        penalty_sum = 0.0
        sample_count = 0
        for inputs, residual_target, cv, anchor, history_end, target_times in loader:
            inputs = inputs.to(device=device, dtype=torch.float32)
            residual_target = residual_target.to(device=device, dtype=torch.float32)
            cv = cv.to(device=device, dtype=torch.float32)
            anchor = anchor.to(device=device, dtype=torch.float32)
            history_end = history_end.to(device=device, dtype=torch.float32)
            target_times = target_times.to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            predicted_residual = model(inputs)
            prediction_loss = loss_fn(predicted_residual, residual_target)
            q_pred = cv + predicted_residual
            penalty, penalty_parts = _hinge_penalty(q_pred, anchor, history_end, target_times, limits)
            loss = prediction_loss + penalty_weight * penalty
            if not torch.isfinite(loss):
                raise RuntimeError("residual_training_nonfinite")
            loss.backward()
            optimizer.step()
            batch_size = len(inputs)
            loss_sum += float(loss.detach().cpu()) * batch_size
            prediction_sum += float(prediction_loss.detach().cpu()) * batch_size
            penalty_sum += float(penalty.detach().cpu()) * batch_size
            sample_count += batch_size
        predicted_validation, _ = predict_residual_model(model, validation_arrays, validation_cv, device=device, batch_size=int(config.get("eval_batch_size", 2048)))
        val_metrics = metric_payload(predicted_validation, validation_arrays.target_positions)
        val_rmse = float(val_metrics["joint_position_rmse_rad"])
        history.append({
            "epoch": epoch,
            "train_loss_total": loss_sum / max(sample_count, 1),
            "train_loss_prediction_mse_rad2": prediction_sum / max(sample_count, 1),
            "train_constraint_penalty": penalty_sum / max(sample_count, 1),
            "validation_joint_position_rmse_rad": val_rmse,
            "validation_window_count": len(validation_arrays.inputs),
            "constraint_penalty_weight": penalty_weight,
        })
        if val_rmse < best_val - float(config.get("min_delta", 0.0)):
            best_val = val_rmse
            best_epoch = epoch
            no_improvement = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            no_improvement += 1
        if no_improvement >= int(config["patience"]):
            break
    if best_state is None:
        raise RuntimeError("residual_validation_checkpoint_not_selected")
    model.load_state_dict(best_state)
    buffer = io.BytesIO()
    torch.save({"model_state_dict": best_state, "model_type": RESIDUAL_MODEL_TYPE, "input_history": 16, "prediction_horizon": HORIZON}, buffer)
    summary = {"parameter_count": parameter_count(model), "best_validation_rmse_rad": best_val, "best_epoch": best_epoch, "constraint_penalty_weight": penalty_weight, "device": device}
    return model, history, best_epoch, buffer.getvalue(), summary


def local_constraint_counts(q: np.ndarray, anchor: np.ndarray, history_times: np.ndarray, target_times: np.ndarray, limits: RepairLimits, history_velocities: np.ndarray | None = None, history_accelerations: np.ndarray | None = None) -> dict[str, int]:
    """Compute the same local finite-difference gates used by H12 diagnostics."""

    full_q = np.concatenate([anchor[:, None, :], q], axis=1)
    full_t = np.concatenate([history_times[:, -1, None], target_times], axis=1)
    dt = np.diff(full_t, axis=1)
    safe_dt = np.maximum(dt, 1.0e-12)
    velocity = np.diff(full_q, axis=1) / safe_dt[:, :, None]
    if history_velocities is None:
        acceleration = np.diff(velocity, axis=1) / safe_dt[:, 1:, None]
    else:
        acceleration = np.empty((len(q), HORIZON, JOINTS), dtype=np.float64)
        acceleration[:, 0, :] = (velocity[:, 0, :] - np.asarray(history_velocities, dtype=np.float64)) / safe_dt[:, 0, None]
        acceleration[:, 1:, :] = np.diff(velocity, axis=1) / safe_dt[:, 1:, None]
    if history_accelerations is None:
        jerk = np.diff(acceleration, axis=1) / safe_dt[:, 2:, None]
    else:
        jerk = np.empty_like(acceleration)
        jerk[:, 0, :] = (acceleration[:, 0, :] - np.asarray(history_accelerations, dtype=np.float64)) / safe_dt[:, 0, None]
        jerk[:, 1:, :] = np.diff(acceleration, axis=1) / safe_dt[:, 1:, None]
    return {
        "position": int(np.count_nonzero((q < limits.position_lower[None, None, :]) | (q > limits.position_upper[None, None, :]))),
        "velocity": int(np.count_nonzero(np.abs(velocity) > limits.velocity_abs[None, None, :])),
        "acceleration": int(np.count_nonzero(np.abs(acceleration) > limits.acceleration_abs[None, None, :])),
        "jerk": int(np.count_nonzero(np.abs(jerk) > limits.jerk_abs[None, None, :])),
        "nan_inf": int(np.count_nonzero(~np.isfinite(full_q)) + np.count_nonzero(~np.isfinite(full_t))),
        "non_monotonic_time": int(np.count_nonzero(np.any(dt <= 0.0, axis=1))),
    }


@dataclass(frozen=True)
class ResidualRepairWindowResult:
    raw_positions: np.ndarray
    repaired_positions: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray
    alpha: float
    success: bool
    fallback_to_cv: bool
    attempts: int
    failure_reason: str | None


def repair_residual_window(
    residual_prediction: np.ndarray,
    cv_positions: np.ndarray,
    anchor_position: np.ndarray,
    history_times: np.ndarray,
    target_times: np.ndarray,
    *,
    history_velocity: np.ndarray | None = None,
    history_acceleration: np.ndarray | None = None,
    limits: RepairLimits = RepairLimits(),
) -> ResidualRepairWindowResult:
    """Backtrack a causal residual proposal toward exact CV until feasible."""

    residual = np.asarray(residual_prediction, dtype=np.float64)
    cv = np.asarray(cv_positions, dtype=np.float64)
    anchor = np.asarray(anchor_position, dtype=np.float64)
    history_t = np.asarray(history_times, dtype=np.float64)
    target_t = np.asarray(target_times, dtype=np.float64)
    if residual.shape != (HORIZON, JOINTS) or cv.shape != residual.shape or anchor.shape != (JOINTS,):
        raise ValueError("residual_repair_shape_mismatch")
    if history_t.ndim != 1 or len(history_t) < 1 or target_t.shape != (HORIZON,) or not np.all(np.isfinite(residual)) or not np.all(np.isfinite(cv)):
        raise ValueError("residual_repair_nonfinite_or_time_shape")
    raw = cv + residual
    if np.any(np.abs(residual) <= 0.0):
        # The all-zero residual case must remain byte-for-byte numerically
        # equivalent to CV; do not pass it through a projection or margin.
        if np.all(residual == 0.0):
            candidate = cv.copy()
            counts = local_constraint_counts(candidate[None, ...], anchor[None, ...], history_t[None, ...], target_t[None, ...], limits, None if history_velocity is None else np.asarray(history_velocity)[None, ...], None if history_acceleration is None else np.asarray(history_acceleration)[None, ...])
            if sum(counts.values()) != 0:
                return ResidualRepairWindowResult(raw, candidate, np.full_like(candidate, np.nan), np.full_like(candidate, np.nan), 0.0, False, True, 1, "cv_local_constraint_violation")
            return _result(raw, candidate, 0.0, 1, False, None, anchor, history_t[-1], target_t, history_velocity, history_acceleration)
    for attempt, alpha in enumerate(BACKTRACK_FACTORS, start=1):
        candidate = cv + float(alpha) * residual
        counts = local_constraint_counts(candidate[None, ...], anchor[None, ...], history_t[None, ...], target_t[None, ...], limits, None if history_velocity is None else np.asarray(history_velocity)[None, ...], None if history_acceleration is None else np.asarray(history_acceleration)[None, ...])
        if sum(counts.values()) == 0:
            return _result(raw, candidate, float(alpha), attempt, float(alpha) == 0.0, None, anchor, history_t[-1], target_t, history_velocity, history_acceleration)
    return ResidualRepairWindowResult(raw, cv.copy(), 0.0, np.full_like(cv, np.nan), 0.0, False, True, len(BACKTRACK_FACTORS), "no_causal_feasible_backtrack_candidate")


def _result(raw: np.ndarray, candidate: np.ndarray, alpha: float, attempts: int, fallback: bool, reason: str | None, anchor: np.ndarray, history_end: float, target_times: np.ndarray, history_velocity: np.ndarray | None, history_acceleration: np.ndarray | None) -> ResidualRepairWindowResult:
    full_q = np.concatenate([anchor[None, :], candidate], axis=0)
    full_t = np.concatenate([[history_end], target_times])
    dt = np.diff(full_t)
    velocity = np.diff(full_q, axis=0) / dt[:, None]
    acceleration = np.empty_like(candidate)
    if history_velocity is None:
        acceleration[0] = 0.0
    else:
        previous_velocity = np.asarray(history_velocity, dtype=np.float64)
        if previous_velocity.shape != (JOINTS,) or not np.all(np.isfinite(previous_velocity)):
            raise ValueError("invalid_history_velocity")
        acceleration[0] = (velocity[0] - previous_velocity) / max(float(dt[0]), 1.0e-9)
    if len(candidate) > 1:
        acceleration[1:] = np.diff(velocity, axis=0) / np.maximum(dt[1:, None], 1.0e-9)
    if history_acceleration is not None:
        previous_acceleration = np.asarray(history_acceleration, dtype=np.float64)
        if previous_acceleration.shape != (JOINTS,) or not np.all(np.isfinite(previous_acceleration)):
            raise ValueError("invalid_history_acceleration")
    return ResidualRepairWindowResult(raw, candidate, velocity, acceleration, alpha, True, fallback, attempts, reason)


def repair_residual_role(residual_predictions: np.ndarray, cv_positions: np.ndarray, anchor_positions: np.ndarray, history_times: np.ndarray, target_times: np.ndarray, *, history_velocities: np.ndarray | None = None, history_accelerations: np.ndarray | None = None, limits: RepairLimits = RepairLimits()) -> dict[str, Any]:
    """Repair a role using only causal arrays and the frozen CV proposal."""

    residual = np.asarray(residual_predictions, dtype=np.float64)
    cv = np.asarray(cv_positions, dtype=np.float64)
    anchors = np.asarray(anchor_positions, dtype=np.float64)
    history_t = np.asarray(history_times, dtype=np.float64)
    target_t = np.asarray(target_times, dtype=np.float64)
    raw = cv + residual
    if history_velocities is None:
        history_v = np.zeros((len(raw), JOINTS), dtype=np.float64)
    else:
        history_v = np.asarray(history_velocities, dtype=np.float64)
        if history_v.shape != (len(raw), JOINTS):
            raise ValueError("residual_repair_history_velocity_shape_mismatch")
    if history_accelerations is None:
        history_a = np.zeros((len(raw), JOINTS), dtype=np.float64)
    else:
        history_a = np.asarray(history_accelerations, dtype=np.float64)
        if history_a.shape != (len(raw), JOINTS):
            raise ValueError("residual_repair_history_acceleration_shape_mismatch")
    if residual.ndim != 3 or residual.shape[1:] != (HORIZON, JOINTS) or cv.shape != residual.shape or anchors.shape != (len(residual), JOINTS) or history_t.shape[0] != len(residual) or target_t.shape != (len(residual), HORIZON):
        raise ValueError("residual_repair_role_shape_mismatch")
    repaired = np.empty_like(raw)
    velocities = np.empty_like(raw)
    accelerations = np.empty_like(raw)
    alphas = np.empty(len(raw), dtype=np.float64)
    fallback = np.zeros(len(raw), dtype=bool)
    failures: list[dict[str, Any]] = []
    for index in range(len(raw)):
        result = repair_residual_window(residual[index], cv[index], anchors[index], history_t[index], target_t[index], history_velocity=history_v[index], history_acceleration=history_a[index], limits=limits)
        repaired[index] = result.repaired_positions
        velocities[index] = result.velocities if result.success else np.nan
        accelerations[index] = result.accelerations if result.success else np.nan
        alphas[index] = result.alpha
        fallback[index] = result.fallback_to_cv
        if not result.success:
            failures.append({"index": index, "reason": result.failure_reason, "attempts": result.attempts})
    return {
        "raw_positions": raw,
        "residual_predictions": residual,
        "cv_positions": cv,
        "positions": repaired,
        "velocities": velocities,
        "accelerations": accelerations,
        "alpha": alphas,
        "fallback_to_cv": fallback,
        "POSITION_REPAIR_ATTEMPTS": int(len(raw)),
        "POSITION_REPAIR_SUCCESS": int(len(raw) - len(failures)),
        "POSITION_REPAIR_FAILURES": int(len(failures)),
        "fallback_count": int(np.count_nonzero(fallback)),
        "fallback_rate": float(np.mean(fallback)) if len(fallback) else 0.0,
        "repair_rate": float(np.mean(np.abs(alphas - 1.0) > 0.0)) if len(alphas) else 0.0,
        "neural_contribution_rate": float(np.mean(alphas)) if len(alphas) else 0.0,
        "failures": failures[:1000],
        "failure_count": len(failures),
        "REPAIR_METHOD": RESIDUAL_REPAIR_METHOD,
        "future_label_access": 0,
    }


def residual_semantic_sha256(values_by_role: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for role in ("TRAIN", "VALIDATION", "TEST", "GENERALIZATION"):
        value = np.asarray(values_by_role[role], dtype=np.float64)
        digest.update(role.encode("ascii"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()
