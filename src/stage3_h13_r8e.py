"""Causal free-running and hidden-consistency helpers for Stage 3 H13-R8E.

This module is intentionally limited to the R8E training-objective intervention.
It preserves the R6 model and causal feature semantics while exposing pure,
testable loss, curriculum, guardrail, and selection primitives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch.nn import functional as F

from src.stage3_h11_dataset import FEATURE_NAMES
from src.stage3_h11_r2_model import SPRAY_FEATURE_INDEX
from src.stage3_h13_r6 import _next_features


PHASE_HORIZONS = (4, 8, 16, 32)
EVALUATION_HORIZONS = (1, 2, 4, 8, 12, 16, 20, 24, 32)
GRADUATION_THRESHOLD = 0.01856902565856056


@dataclass(frozen=True)
class H1Guardrail:
    reference: float
    fresh_replay: float
    threshold: float
    derivation: str
    frozen_before_candidate_selection: bool = True


def derive_h1_guardrail(
    authoritative_h1: float,
    fresh_h1: float,
    *,
    material_allowance_fraction: float = 0.05,
    absolute_replay_tolerance: float = 5.0e-9,
) -> H1Guardrail:
    """Freeze an auditable local-quality gate before any candidate is compared."""

    if authoritative_h1 <= 0.0 or fresh_h1 <= 0.0:
        raise ValueError("h1_reference_must_be_positive")
    reference = max(float(authoritative_h1), float(fresh_h1))
    threshold = reference * (1.0 + float(material_allowance_fraction)) + float(absolute_replay_tolerance)
    return H1Guardrail(
        reference=float(authoritative_h1),
        fresh_replay=float(fresh_h1),
        threshold=threshold,
        derivation=(
            "max(authoritative_R6_H1,fresh_R6_H1) * 1.05 + 5e-9; "
            "5% is the predeclared material-local-degradation allowance and 5e-9 "
            "is the existing absolute authoritative replay tolerance"
        ),
    )


def hidden_alignment_weights(horizon: int) -> torch.Tensor:
    """H1 zero; H2-H4 strongest; H5-H8 medium; H9-H16 low; later zero."""

    values = []
    for step in range(1, int(horizon) + 1):
        if step == 1 or step > 16:
            values.append(0.0)
        elif step <= 4:
            values.append(1.0)
        elif step <= 8:
            values.append(0.5)
        else:
            values.append(0.15)
    return torch.tensor(values, dtype=torch.float32)


def rollout_horizon_weights(horizon: int) -> torch.Tensor:
    """Mild long-horizon emphasis without suppressing local prediction."""

    if int(horizon) < 1:
        raise ValueError("rollout_horizon_invalid")
    if int(horizon) == 1:
        return torch.ones(1, dtype=torch.float32)
    weights = torch.linspace(1.0, 1.5, int(horizon), dtype=torch.float32)
    return weights / torch.sum(weights)


def phase_for_update(update: int, updates_per_phase: int) -> int:
    if int(update) < 0 or int(updates_per_phase) < 1:
        raise ValueError("curriculum_update_invalid")
    index = min(len(PHASE_HORIZONS) - 1, int(update) // int(updates_per_phase))
    return PHASE_HORIZONS[index]


def model_forward_with_hidden(model: Any, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Native GRU forward with both layer states exposed; parameters are shared."""

    _, hidden = model.gru(inputs)
    output = model.head(hidden[-1]).reshape(inputs.shape[0], model.horizon, 6)
    return output, hidden


def normalized_hidden_smooth_l1(
    free_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    *,
    beta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Layer-wise normalized Smooth-L1 with a stopped teacher reference."""

    teacher_reference = teacher_hidden.detach()
    dimension = int(teacher_reference.shape[-1])
    scale = torch.linalg.vector_norm(teacher_reference, dim=-1, keepdim=True) / math.sqrt(float(dimension))
    scale = torch.clamp(scale, min=1.0e-3)
    element = F.smooth_l1_loss(free_hidden / scale, teacher_reference / scale, beta=float(beta), reduction="none")
    per_layer = element.mean(dim=(1, 2))
    return per_layer.mean(), per_layer


def _advance(
    current_inputs: torch.Tensor,
    positions: torch.Tensor,
    times: torch.Tensor,
    next_position: torch.Tensor,
    next_time: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    channels: Mapping[str, Mapping[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    last = positions[:, -1, :]
    prior = positions[:, -2, :]
    last_time = times[:, -1]
    prior_time = times[:, -2]
    features = _next_features(
        next_position,
        last,
        prior,
        next_time,
        last_time,
        prior_time,
        starts,
        ends,
        current_inputs[:, -1, SPRAY_FEATURE_INDEX],
        channels,
        FEATURE_NAMES,
    ).to(dtype=current_inputs.dtype)
    return (
        torch.cat((current_inputs[:, 1:, :], features[:, None, :]), dim=1),
        torch.cat((positions[:, 1:, :], next_position[:, None, :]), dim=1),
        torch.cat((times[:, 1:], next_time[:, None]), dim=1),
    )


def causal_paired_rollout(
    model: Any,
    inputs: torch.Tensor,
    history_positions: torch.Tensor,
    history_times: torch.Tensor,
    target_times: torch.Tensor,
    trajectory_start_times: torch.Tensor,
    trajectory_end_times: torch.Tensor,
    channels: Mapping[str, Mapping[str, Any]],
    *,
    horizon: int,
    target_positions_for_teacher: torch.Tensor | None,
    hidden_consistency_enabled: bool,
) -> dict[str, Any]:
    """Run differentiable inference-faithful free rollout and optional teacher branch.

    The free branch never reads ``target_positions_for_teacher``.  Targets are
    visible only to the separately maintained teacher branch.
    """

    if hidden_consistency_enabled and target_positions_for_teacher is None:
        raise ValueError("teacher_targets_required_for_hidden_consistency")
    if int(horizon) < 1 or int(horizon) > int(target_times.shape[1]):
        raise ValueError("r8e_invalid_rollout_horizon")
    free_inputs = inputs
    free_positions = history_positions
    free_times = history_times
    teacher_inputs = inputs
    teacher_positions = history_positions
    teacher_times = history_times
    predictions: list[torch.Tensor] = []
    hidden_weight_vector = hidden_alignment_weights(int(horizon)).to(device=inputs.device, dtype=inputs.dtype)
    hidden_terms: list[torch.Tensor] = []
    layer_numerators = torch.zeros(2, device=inputs.device, dtype=inputs.dtype)
    hidden_weight_sum = torch.zeros((), device=inputs.device, dtype=inputs.dtype)
    teacher_reference_stop_gradient = False

    for step in range(int(horizon)):
        free_output, free_hidden = model_forward_with_hidden(model, free_inputs)
        last = free_positions[:, -1, :]
        prior = free_positions[:, -2, :]
        last_time = free_times[:, -1]
        prior_time = free_times[:, -2]
        dt = torch.clamp(target_times[:, step] - last_time, min=1.0e-9)
        velocity = (last - prior) / torch.clamp(last_time - prior_time, min=1.0e-9)[:, None]
        free_next = last + velocity * dt[:, None] + free_output[:, 0, :]
        predictions.append(free_next)

        hidden_weight = hidden_weight_vector[step]
        if hidden_consistency_enabled and float(hidden_weight) > 0.0:
            _, teacher_hidden = model_forward_with_hidden(model, teacher_inputs)
            hidden_value, per_layer = normalized_hidden_smooth_l1(free_hidden, teacher_hidden)
            hidden_terms.append(hidden_weight * hidden_value)
            layer_numerators = layer_numerators + hidden_weight * per_layer
            hidden_weight_sum = hidden_weight_sum + hidden_weight
            teacher_reference_stop_gradient = not teacher_hidden.detach().requires_grad

        if step + 1 == int(horizon):
            continue
        free_inputs, free_positions, free_times = _advance(
            free_inputs,
            free_positions,
            free_times,
            free_next,
            target_times[:, step],
            trajectory_start_times,
            trajectory_end_times,
            channels,
        )
        if hidden_consistency_enabled:
            assert target_positions_for_teacher is not None
            teacher_inputs, teacher_positions, teacher_times = _advance(
                teacher_inputs,
                teacher_positions,
                teacher_times,
                target_positions_for_teacher[:, step, :],
                target_times[:, step],
                trajectory_start_times,
                trajectory_end_times,
                channels,
            )

    zero = torch.zeros((), device=inputs.device, dtype=inputs.dtype)
    if hidden_terms:
        hidden_loss = torch.stack(hidden_terms).sum() / torch.clamp(hidden_weight_sum, min=1.0e-12)
        layer_losses = layer_numerators / torch.clamp(hidden_weight_sum, min=1.0e-12)
    else:
        hidden_loss = zero
        layer_losses = torch.zeros(2, device=inputs.device, dtype=inputs.dtype)
    return {
        "predictions": torch.stack(predictions, dim=1),
        "hidden_loss": hidden_loss,
        "hidden_loss_per_layer": layer_losses,
        "teacher_reference_stop_gradient": teacher_reference_stop_gradient if hidden_consistency_enabled else None,
        "teacher_branch_executed": bool(hidden_terms),
        "free_branch_reads_target_positions": False,
    }


def training_losses(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    hidden_loss: torch.Tensor,
    *,
    lambda_hidden: float,
    lambda_h1: float,
    beta: float,
) -> dict[str, torch.Tensor]:
    horizon = int(predictions.shape[1])
    step_losses = F.smooth_l1_loss(predictions, targets[:, :horizon, :], beta=float(beta), reduction="none").mean(dim=(0, 2))
    weights = rollout_horizon_weights(horizon).to(device=predictions.device, dtype=predictions.dtype)
    rollout_loss = torch.sum(step_losses * weights)
    h1_loss = step_losses[0]
    total = rollout_loss + float(lambda_hidden) * hidden_loss + float(lambda_h1) * h1_loss
    return {"total": total, "rollout": rollout_loss, "hidden": hidden_loss, "h1": h1_loss}


def gradients_are_finite(model: Any) -> bool:
    return all(parameter.grad is None or bool(torch.all(torch.isfinite(parameter.grad))) for parameter in model.parameters())


def select_candidate(
    candidates: Sequence[Mapping[str, Any]],
    guardrail: H1Guardrail,
    *,
    tie_tolerance: float = 5.0e-9,
) -> Mapping[str, Any] | None:
    """Pure validation-only selection; there is no path or external-data access."""

    if not guardrail.frozen_before_candidate_selection:
        raise RuntimeError("h1_guardrail_not_frozen_before_candidate_selection")
    eligible = [
        item for item in candidates
        if bool(item.get("integrity_pass"))
        and int(item.get("future_label_leakage", 1)) == 0
        and float(item["h1_rmse"]) <= guardrail.threshold
    ]
    if not eligible:
        return None
    best_h32 = min(float(item["h32_rmse"]) for item in eligible)
    tied = [item for item in eligible if abs(float(item["h32_rmse"]) - best_h32) <= float(tie_tolerance)]
    return min(
        tied,
        key=lambda item: (
            float(item.get("early_hidden_divergence", math.inf)),
            int(item.get("complexity_rank", 1)),
            int(item.get("persistent_bytes", 0)),
        ),
    )


def graduation_passes(candidate: Mapping[str, Any] | None, guardrail: H1Guardrail) -> bool:
    return bool(
        candidate is not None
        and float(candidate["h32_rmse"]) <= GRADUATION_THRESHOLD
        and float(candidate["h1_rmse"]) <= guardrail.threshold
        and bool(candidate.get("integrity_pass"))
        and int(candidate.get("future_label_leakage", 1)) == 0
    )


def early_hidden_drift_reduced(r6: Mapping[str, float], candidate: Mapping[str, float]) -> bool:
    """Require a lower H2/H4/H8 mean and improvement at a majority of horizons."""

    horizons = ("2", "4", "8")
    return (
        sum(float(candidate[h]) for h in horizons) < sum(float(r6[h]) for h in horizons)
        and sum(float(candidate[h]) < float(r6[h]) for h in horizons) >= 2
    )


def h4_amplification_reduced_or_delayed(
    r6_h4: float,
    candidate_h4: float,
    r6_onset: int | None,
    candidate_onset: int | None,
) -> bool:
    return bool(
        float(candidate_h4) < float(r6_h4)
        or (
            r6_onset is not None
            and candidate_onset is not None
            and int(candidate_onset) > int(r6_onset)
        )
    )
