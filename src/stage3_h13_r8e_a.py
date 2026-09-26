"""Read-only helpers for the R6 hidden-state/transition stability audit.

The authoritative R6 module uses ``nn.GRU`` as a window encoder.  These
helpers reproduce one GRU cell transition without changing the model and
expose compact, deterministic aggregation primitives for diagnostic use.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

try:
    import torch
    from torch.nn import functional as F

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


AUDIT_HORIZONS = (1, 2, 3, 4, 5, 6, 8, 12, 16, 20, 24, 32)
MATERIAL_RELATIVE_THRESHOLDS = (0.005, 0.01, 0.02)


def gru_cell_transition(model: Any, layer: int, x: "torch.Tensor", h: "torch.Tensor") -> "torch.Tensor":
    """Reproduce PyTorch's GRU transition for one layer and one time step."""

    if not TORCH_AVAILABLE:
        raise RuntimeError("pytorch_unavailable")
    weight_ih = getattr(model.gru, f"weight_ih_l{int(layer)}")
    weight_hh = getattr(model.gru, f"weight_hh_l{int(layer)}")
    bias_ih = getattr(model.gru, f"bias_ih_l{int(layer)}")
    bias_hh = getattr(model.gru, f"bias_hh_l{int(layer)}")
    gi = F.linear(x, weight_ih, bias_ih)
    gh = F.linear(h, weight_hh, bias_hh)
    i_r, i_z, i_n = gi.chunk(3, dim=1)
    h_r, h_z, h_n = gh.chunk(3, dim=1)
    reset = torch.sigmoid(i_r + h_r)
    update = torch.sigmoid(i_z + h_z)
    candidate = torch.tanh(i_n + reset * h_n)
    return candidate + update * (h - candidate)


def instrumented_forward(model: Any, inputs: "torch.Tensor") -> Mapping[str, Any]:
    """Manual two-layer GRU forward exposing final/pre-final layer states."""

    if not TORCH_AVAILABLE:
        raise RuntimeError("pytorch_unavailable")
    if model.gru.bidirectional or model.gru.num_layers != 2:
        raise ValueError("r8e_a_requires_two_layer_unidirectional_gru")
    batch = int(inputs.shape[0])
    h = [torch.zeros((batch, model.hidden_size), dtype=inputs.dtype, device=inputs.device) for _ in range(2)]
    pre_final: list[torch.Tensor | None] = [None, None]
    final_inputs: list[torch.Tensor | None] = [None, None]
    for step in range(int(inputs.shape[1])):
        layer_input = inputs[:, step, :]
        for layer in range(2):
            if step + 1 == int(inputs.shape[1]):
                pre_final[layer] = h[layer]
                final_inputs[layer] = layer_input
            h[layer] = gru_cell_transition(model, layer, layer_input, h[layer])
            layer_input = h[layer]
    output = model.head(h[1]).reshape(batch, model.horizon, 6)
    return {
        "output": output,
        "hidden": torch.stack(h, dim=0),
        "pre_final": torch.stack([value for value in pre_final if value is not None], dim=0),
        "final_inputs": tuple(value for value in final_inputs if value is not None),
    }


def distribution_summary(values: Sequence[float] | np.ndarray) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return {key: None for key in ("mean", "median", "p90", "p95", "max")}
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def first_sustained_threshold(profile: Mapping[str, Mapping[str, Any]], field: str, threshold: float) -> int | None:
    """First audit horizon at/after which all reported values exceed threshold."""

    horizons = [h for h in AUDIT_HORIZONS if str(h) in profile]
    for index, horizon in enumerate(horizons):
        values = [profile[str(item)].get(field) for item in horizons[index:]]
        if values and all(value is not None and float(value) >= float(threshold) for value in values):
            return int(horizon)
    return None


def normalized_hidden_metrics(teacher: np.ndarray, free: np.ndarray) -> dict[str, np.ndarray]:
    delta = np.asarray(free, dtype=np.float64) - np.asarray(teacher, dtype=np.float64)
    distance = np.linalg.norm(delta, axis=-1)
    teacher_norm = np.linalg.norm(teacher, axis=-1)
    free_norm = np.linalg.norm(free, axis=-1)
    denominator = np.maximum(teacher_norm, 1.0e-12)
    cosine_denominator = np.maximum(teacher_norm * free_norm, 1.0e-12)
    return {
        "l2": distance,
        "relative_l2": distance / denominator,
        "cosine_similarity": np.sum(teacher * free, axis=-1) / cosine_denominator,
        "teacher_norm": teacher_norm,
        "free_norm": free_norm,
        "norm_ratio": free_norm / denominator,
    }


def finite_difference_discrepancy(actual: "torch.Tensor", predicted: "torch.Tensor") -> np.ndarray:
    numerator = torch.linalg.vector_norm(actual - predicted, dim=1)
    denominator = torch.clamp(torch.linalg.vector_norm(predicted, dim=1), min=1.0e-12)
    return (numerator / denominator).detach().cpu().numpy().astype(np.float64)


def amplification_ratio(jv: "torch.Tensor", direction: "torch.Tensor") -> np.ndarray:
    numerator = torch.linalg.vector_norm(jv, dim=1)
    denominator = torch.clamp(torch.linalg.vector_norm(direction, dim=1), min=1.0e-12)
    return (numerator / denominator).detach().cpu().numpy().astype(np.float64)


def safe_rmse(sum_squares: float, count: int) -> float | None:
    return float(math.sqrt(float(sum_squares) / int(count))) if count else None
