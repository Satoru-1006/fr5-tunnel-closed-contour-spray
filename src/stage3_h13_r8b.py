"""Causal curvature-conditioned reference semantics for Stage 3 H13-R8B.

The module deliberately keeps the R8/R8A model and state semantics small.  It
only changes the analytical reference used before the learned bounded
residual:

    C5 = (1 - w) * constant_velocity + w * local_quadratic

The gate is a fixed, train-derived mapping of the causal curvature proxy.  No
target positions or future labels are accepted by the reference functions.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from src.stage3_h11_dataset import FEATURE_NAMES, INPUT_HISTORY, PREDICTION_HORIZON
from src.stage3_h11_r2_model import JOINTS, SPRAY_FEATURE_INDEX
from src.stage3_h13_r6 import _next_features
from src.stage3_h13_r8 import bounded_residual_torch

try:
    import torch

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - contract tests can import without torch
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


C5_SCHEMA_VERSION = "stage3_h13_r8b_causal_curvature_reference_v1"
CURVATURE_EPSILON = 1.0e-8
LOCAL_QUADRATIC_FIT_POINTS = 8


def _validate_gate(k_low: float, k_high: float) -> None:
    if not math.isfinite(float(k_low)) or not math.isfinite(float(k_high)):
        raise ValueError("c5_curvature_gate_nonfinite")
    if float(k_high) <= float(k_low):
        raise ValueError("c5_curvature_gate_degenerate")


def gate_weight_numpy(curvature: np.ndarray, k_low: float, k_high: float) -> np.ndarray:
    """Return the fixed clipped C5 gate weight."""

    _validate_gate(k_low, k_high)
    values = np.asarray(curvature, dtype=np.float64)
    return np.clip((values - float(k_low)) / (float(k_high) - float(k_low)), 0.0, 1.0)


def curvature_proxy_numpy(positions: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Compute the R8A causal curvature proxy from the last three states.

    The proxy is ``||a|| / max(||v||^2, epsilon)``.  It uses only the supplied
    history and is therefore valid both for an observed initial history and
    for a history extended with analytically generated causal references.
    """

    p = np.asarray(positions, dtype=np.float64)
    t = np.asarray(times, dtype=np.float64)
    if p.ndim != 3 or p.shape[-1] != JOINTS or p.shape[1] < 3:
        raise ValueError("c5_curvature_position_history_invalid")
    if t.shape != p.shape[:2]:
        raise ValueError("c5_curvature_time_history_invalid")
    dt_last = np.maximum(t[:, -1] - t[:, -2], 1.0e-9)
    dt_prior = np.maximum(t[:, -2] - t[:, -3], 1.0e-9)
    velocity = (p[:, -1] - p[:, -2]) / dt_last[:, None]
    prior_velocity = (p[:, -2] - p[:, -3]) / dt_prior[:, None]
    acceleration = (velocity - prior_velocity) / dt_last[:, None]
    speed_squared = np.sum(np.square(velocity), axis=1)
    return np.linalg.norm(acceleration, axis=1) / np.maximum(speed_squared, CURVATURE_EPSILON)


def _quadratic_reference_numpy(positions: np.ndarray, times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    p = np.asarray(positions, dtype=np.float64)
    t = np.asarray(times, dtype=np.float64)
    tt = np.asarray(target_times, dtype=np.float64)
    fit_n = min(LOCAL_QUADRATIC_FIT_POINTS, p.shape[1])
    last_t = t[:, -1]
    x = t[:, -fit_n:] - last_t[:, None]
    design = np.stack((np.ones_like(x), x, np.square(x)), axis=2)
    # Use the same local least-squares semantics as R8A, without introducing
    # a phase/progress signal or any future trajectory value.
    coeff = np.einsum("bkn,bnj->bkj", np.linalg.pinv(design), p[:, -fit_n:, :])
    elapsed = tt - last_t[:, None]
    basis = np.stack((np.ones_like(elapsed), elapsed, np.square(elapsed)), axis=2)
    return np.einsum("bhi,bik->bhk", basis, coeff)


def causal_curvature_conditioned_reference_numpy(
    positions: np.ndarray,
    times: np.ndarray,
    target_times: np.ndarray,
    k_low: float,
    k_high: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the C5 reference and causal gate weight for each history."""

    _validate_gate(k_low, k_high)
    p = np.asarray(positions, dtype=np.float64)
    t = np.asarray(times, dtype=np.float64)
    tt = np.asarray(target_times, dtype=np.float64)
    if p.ndim != 3 or p.shape[-1] != JOINTS or t.shape != p.shape[:2]:
        raise ValueError("c5_reference_history_shape_invalid")
    if tt.ndim != 2 or tt.shape[0] != p.shape[0]:
        raise ValueError("c5_reference_target_time_shape_invalid")
    last = p[:, -1]
    dt = np.maximum(t[:, -1] - t[:, -2], 1.0e-9)
    velocity = (last - p[:, -2]) / dt[:, None]
    elapsed = tt - t[:, -1, None]
    cv = last[:, None, :] + velocity[:, None, :] * elapsed[:, :, None]
    quadratic = _quadratic_reference_numpy(p, t, tt)
    weights = gate_weight_numpy(curvature_proxy_numpy(p, t), k_low, k_high)
    return cv + weights[:, None, None] * (quadratic - cv), weights


def curvature_proxy_torch(positions: "torch.Tensor", times: "torch.Tensor") -> "torch.Tensor":
    if not TORCH_AVAILABLE:
        raise RuntimeError("pytorch_unavailable")
    if positions.ndim != 3 or positions.shape[-1] != JOINTS or positions.shape[1] < 3:
        raise ValueError("c5_curvature_position_history_invalid")
    if times.shape != positions.shape[:2]:
        raise ValueError("c5_curvature_time_history_invalid")
    dt_last = torch.clamp(times[:, -1] - times[:, -2], min=1.0e-9)
    dt_prior = torch.clamp(times[:, -2] - times[:, -3], min=1.0e-9)
    velocity = (positions[:, -1] - positions[:, -2]) / dt_last[:, None]
    prior_velocity = (positions[:, -2] - positions[:, -3]) / dt_prior[:, None]
    acceleration = (velocity - prior_velocity) / dt_last[:, None]
    speed_squared = torch.sum(torch.square(velocity), dim=1)
    return torch.linalg.vector_norm(acceleration, dim=1) / torch.clamp(speed_squared, min=CURVATURE_EPSILON)


def gate_weight_torch(curvature: "torch.Tensor", k_low: float, k_high: float) -> "torch.Tensor":
    _validate_gate(k_low, k_high)
    return torch.clamp((curvature - float(k_low)) / (float(k_high) - float(k_low)), 0.0, 1.0)


def _quadratic_reference_torch(positions: "torch.Tensor", times: "torch.Tensor", target_time: "torch.Tensor") -> "torch.Tensor":
    fit_n = min(LOCAL_QUADRATIC_FIT_POINTS, positions.shape[1])
    last_t = times[:, -1]
    x = times[:, -fit_n:] - last_t[:, None]
    design = torch.stack((torch.ones_like(x), x, torch.square(x)), dim=2)
    coeff = torch.linalg.lstsq(design, positions[:, -fit_n:, :]).solution
    target_matrix = target_time[:, None] if target_time.ndim == 1 else target_time
    elapsed = torch.clamp(target_matrix - last_t[:, None], min=1.0e-9)
    basis = torch.stack((torch.ones_like(elapsed), elapsed, torch.square(elapsed)), dim=2)
    return torch.bmm(basis, coeff)


def causal_curvature_conditioned_reference_torch(
    positions: "torch.Tensor",
    times: "torch.Tensor",
    target_time: "torch.Tensor",
    k_low: float,
    k_high: float,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Torch C5 reference for direct and streaming rollout evaluation."""

    _validate_gate(k_low, k_high)
    last = positions[:, -1, :]
    prior = positions[:, -2, :]
    last_t = times[:, -1]
    dt = torch.clamp(last_t - times[:, -2], min=1.0e-9)
    velocity = (last - prior) / dt[:, None]
    target_matrix = target_time[:, None] if target_time.ndim == 1 else target_time
    elapsed = torch.clamp(target_matrix - last_t[:, None], min=1.0e-9)
    cv = last[:, None, :] + velocity[:, None, :] * elapsed[:, :, None]
    quadratic = _quadratic_reference_torch(positions, times, target_time)
    weights = gate_weight_torch(curvature_proxy_torch(positions, times), k_low, k_high)
    return cv + weights[:, None, None] * (quadratic - cv), weights


def rollout_c5_reference_torch(
    model: Any,
    inputs: "torch.Tensor",
    history_positions: "torch.Tensor",
    history_times: "torch.Tensor",
    target_times: "torch.Tensor",
    trajectory_start_times: "torch.Tensor",
    trajectory_end_times: "torch.Tensor",
    channels: Mapping[str, Mapping[str, Any]],
    *,
    k_low: float,
    k_high: float,
    residual_bound: Sequence[float] | float | None,
    rollout_horizon: int,
    semantics: str = "chunked_multihorizon",
    feature_names: Sequence[str] = FEATURE_NAMES,
) -> "torch.Tensor":
    """Run the R8A chunked reference-only rollout with C5 references."""

    if not TORCH_AVAILABLE:
        raise RuntimeError("pytorch_unavailable")
    if semantics not in {"single_step", "chunked_multihorizon"}:
        raise ValueError("c5_unknown_rollout_semantics")
    if rollout_horizon < 1 or rollout_horizon > target_times.shape[1]:
        raise ValueError("c5_invalid_rollout_horizon")
    current_inputs = inputs
    positions = history_positions
    times = history_times
    predictions: list[torch.Tensor] = []
    step = 0
    while step < int(rollout_horizon):
        raw_head = model(current_inputs)
        chunk = 1 if semantics == "single_step" else min(PREDICTION_HORIZON, int(rollout_horizon) - step)
        raw_residual = raw_head[:, :chunk, :]
        residual = bounded_residual_torch(raw_residual, residual_bound)
        for local in range(chunk):
            target_time = target_times[:, step]
            reference_block, _ = causal_curvature_conditioned_reference_torch(
                positions, times, target_time, k_low, k_high
            )
            reference_next = reference_block[:, 0, :]
            predictions.append(reference_next + residual[:, local, :])
            next_features = _next_features(
                reference_next,
                positions[:, -1, :],
                positions[:, -2, :],
                target_time,
                times[:, -1],
                times[:, -2],
                trajectory_start_times,
                trajectory_end_times,
                current_inputs[:, -1, SPRAY_FEATURE_INDEX],
                channels,
                feature_names,
            ).to(dtype=current_inputs.dtype)
            current_inputs = torch.cat((current_inputs[:, 1:, :], next_features[:, None, :]), dim=1)
            # The residual prediction is deliberately not appended.
            positions = torch.cat((positions[:, 1:, :], reference_next[:, None, :]), dim=1)
            times = torch.cat((times[:, 1:], target_time[:, None]), dim=1)
            step += 1
    return torch.stack(predictions, dim=1)
