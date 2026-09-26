"""Focused AdamW transition identity checks for the D26 recovery sprint.

The checker is deliberately independent of the scientific objective.  It
receives the gradients and before/after snapshots from a real optimizer step
and verifies the frozen AdamW equations using the candidate's effective LR.
"""

from __future__ import annotations

import copy
import hashlib
import math
from typing import Any, Mapping, Sequence

import torch


BETAS = (0.9, 0.999)
EPS = 1.0e-8
WEIGHT_DECAY = 1.0e-5
ATOL = 1.0e-7
RTOL = 2.0e-5


def clone_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous().clone()
    if isinstance(value, Mapping):
        return {key: clone_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(clone_value(item) for item in value)
    return copy.deepcopy(value)


def tensor_digest(values: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in values.items():
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def snapshot_parameters(model: Any) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().cpu().contiguous().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}


def snapshot_optimizer_state(optimizer: torch.optim.Optimizer) -> dict[str, dict[str, Any]]:
    names = dict(getattr(optimizer, "_codex_parameter_names", {}))
    result: dict[str, dict[str, Any]] = {}
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            name = names.get(id(parameter))
            if name is None:
                # Synthetic regression models do not carry the repository
                # name mapping; fall back to the live named-parameter order.
                name = next((candidate for candidate, live in optimizer._d26_named_parameters if live is parameter), None) if hasattr(optimizer, "_d26_named_parameters") else None
            if name is None:
                raise ValueError("optimizer_parameter_name_mapping_missing")
            result[name] = {str(key): clone_value(value) for key, value in optimizer.state.get(parameter, {}).items()}
    return result


def bind_parameter_names(model: Any, optimizer: torch.optim.Optimizer) -> None:
    """Bind a stable fallback mapping for synthetic and repository branches."""
    optimizer._d26_named_parameters = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]


def _scalar_step(value: Any) -> int:
    return int(value.item()) if isinstance(value, torch.Tensor) else int(value)


def _state_tensor(state: Mapping[str, Any], key: str, like: torch.Tensor) -> torch.Tensor:
    value = state.get(key)
    if value is None:
        return torch.zeros_like(like)
    return value.detach().cpu().contiguous().clone() if isinstance(value, torch.Tensor) else torch.as_tensor(value, dtype=like.dtype)


def _state_step(state: Mapping[str, Any]) -> int:
    return _scalar_step(state["step"]) if "step" in state else 0


def _allclose(left: torch.Tensor, right: torch.Tensor) -> bool:
    return bool(torch.allclose(left.detach().cpu(), right.detach().cpu(), rtol=RTOL, atol=ATOL, equal_nan=False))


def _effective_lrs(optimizer: torch.optim.Optimizer, assumed: Sequence[float] | None) -> tuple[list[float], list[float], dict[str, float]]:
    actual = [float(group["lr"]) for group in optimizer.param_groups]
    expected = [float(value) for value in (assumed if assumed is not None else actual)]
    if len(actual) != len(expected):
        raise ValueError("optimizer_lr_group_count_mismatch")
    names = dict(getattr(optimizer, "_codex_parameter_names", {}))
    fallback = dict(getattr(optimizer, "_d26_named_parameters", []))
    by_name: dict[str, float] = {}
    for group_index, group in enumerate(optimizer.param_groups):
        for parameter in group["params"]:
            name = names.get(id(parameter))
            if name is None:
                name = next((candidate for candidate, live in fallback.items() if live is parameter), None)
            if name is None:
                raise ValueError("optimizer_parameter_name_mapping_missing")
            by_name[name] = expected[group_index]
    return actual, expected, by_name


def validate_adamw_transition(
    model: Any,
    optimizer: torch.optim.Optimizer,
    before_parameters: Mapping[str, torch.Tensor],
    before_optimizer: Mapping[str, Mapping[str, Any]],
    gradients: Mapping[str, torch.Tensor],
    after_parameters: Mapping[str, torch.Tensor],
    after_optimizer: Mapping[str, Mapping[str, Any]],
    assumed_effective_lrs: Sequence[float] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a summary and tensor evidence for one realized AdamW step."""
    actual_lrs, assumed_lrs, lr_by_name = _effective_lrs(optimizer, assumed_effective_lrs)
    names = list(before_parameters)
    expected_m: dict[str, torch.Tensor] = {}
    realized_m: dict[str, torch.Tensor] = {}
    expected_v: dict[str, torch.Tensor] = {}
    realized_v: dict[str, torch.Tensor] = {}
    expected_adaptive: dict[str, torch.Tensor] = {}
    realized_adaptive: dict[str, torch.Tensor] = {}
    expected_decay: dict[str, torch.Tensor] = {}
    realized_decay: dict[str, torch.Tensor] = {}
    expected_total: dict[str, torch.Tensor] = {}
    realized_delta: dict[str, torch.Tensor] = {}
    residual: dict[str, torch.Tensor] = {}
    step_before: dict[str, int] = {}
    step_after: dict[str, int] = {}
    failure_subconditions: list[str] = []
    checks = {
        "candidate_effective_lr_identity": actual_lrs == assumed_lrs,
        "optimizer_step_identity": True,
        "exp_avg_transition_identity": True,
        "exp_avg_sq_transition_identity": True,
        "weight_decay_identity": True,
        "adaptive_adam_identity": True,
        "parameter_delta_identity": True,
        "total_reconstructed_update_identity": True,
    }
    for name in names:
        before = before_optimizer.get(name, {})
        after = after_optimizer.get(name, {})
        g = gradients[name].detach().cpu().contiguous().clone()
        theta_before = before_parameters[name].detach().cpu().contiguous().clone()
        theta_after = after_parameters[name].detach().cpu().contiguous().clone()
        m_before = _state_tensor(before, "exp_avg", g)
        v_before = _state_tensor(before, "exp_avg_sq", g)
        m_expected = BETAS[0] * m_before + (1.0 - BETAS[0]) * g
        v_expected = BETAS[1] * v_before + (1.0 - BETAS[1]) * g.square()
        m_realized = _state_tensor(after, "exp_avg", g)
        v_realized = _state_tensor(after, "exp_avg_sq", g)
        before_step = _state_step(before)
        after_step = _state_step(after)
        lr = lr_by_name[name]
        adaptive = -lr * (m_expected / (1.0 - BETAS[0] ** after_step)) / (torch.sqrt(v_expected / (1.0 - BETAS[1] ** after_step)) + EPS)
        adaptive_realized = -lr * (m_realized / (1.0 - BETAS[0] ** after_step)) / (torch.sqrt(v_realized / (1.0 - BETAS[1] ** after_step)) + EPS)
        decay = -lr * WEIGHT_DECAY * theta_before
        total = adaptive + decay
        delta = theta_after - theta_before
        remainder = delta - total
        decay_realized = delta - adaptive_realized
        expected_m[name], realized_m[name] = m_expected, m_realized
        expected_v[name], realized_v[name] = v_expected, v_realized
        expected_adaptive[name], realized_adaptive[name] = adaptive, adaptive_realized
        expected_decay[name], realized_decay[name] = decay, decay_realized
        expected_total[name], realized_delta[name], residual[name] = total, delta, remainder
        step_before[name], step_after[name] = before_step, after_step
        checks["optimizer_step_identity"] &= after_step == before_step + 1
        checks["exp_avg_transition_identity"] &= _allclose(m_expected, m_realized)
        checks["exp_avg_sq_transition_identity"] &= _allclose(v_expected, v_realized)
        checks["weight_decay_identity"] &= _allclose(decay, decay_realized)
        checks["adaptive_adam_identity"] &= _allclose(adaptive, adaptive_realized)
        checks["parameter_delta_identity"] &= _allclose(delta, total)
        checks["total_reconstructed_update_identity"] &= bool(torch.linalg.vector_norm(remainder) <= ATOL + RTOL * max(float(torch.linalg.vector_norm(delta)), 1.0))
    for key, passed in checks.items():
        if not passed:
            failure_subconditions.append(key)
    summary = {
        "actual_candidate_effective_lr": actual_lrs,
        "adamw_reconstruction_assumed_lr": assumed_lrs,
        "lr_assumption_identity_pass": bool(checks["candidate_effective_lr_identity"]),
        "optimizer_step_before": step_before,
        "optimizer_step_after": step_after,
        "optimizer_step_identity_pass": bool(checks["optimizer_step_identity"]),
        "exp_avg_transition_pass": bool(checks["exp_avg_transition_identity"]),
        "exp_avg_sq_transition_pass": bool(checks["exp_avg_sq_transition_identity"]),
        "weight_decay_contribution_pass": bool(checks["weight_decay_identity"]),
        "adaptive_adam_contribution_pass": bool(checks["adaptive_adam_identity"]),
        "parameter_delta_pass": bool(checks["parameter_delta_identity"]),
        "total_reconstructed_update_pass": bool(checks["total_reconstructed_update_identity"]),
        "adamw_decomposition_status": "PASS" if not failure_subconditions else "FAIL",
        "optimizer_consistency_pass": not failure_subconditions,
        "failure_subconditions": failure_subconditions,
        "expected_exp_avg_sha256": tensor_digest(expected_m),
        "realized_exp_avg_sha256": tensor_digest(realized_m),
        "expected_exp_avg_sq_sha256": tensor_digest(expected_v),
        "realized_exp_avg_sq_sha256": tensor_digest(realized_v),
        "expected_parameter_delta_sha256": tensor_digest(expected_total),
        "realized_parameter_delta_sha256": tensor_digest(realized_delta),
        "weight_decay_contribution_sha256": tensor_digest(expected_decay),
        "realized_weight_decay_contribution_sha256": tensor_digest(realized_decay),
        "adaptive_adam_contribution_sha256": tensor_digest(expected_adaptive),
        "realized_adaptive_adam_contribution_sha256": tensor_digest(realized_adaptive),
        "total_reconstructed_update_sha256": tensor_digest(expected_total),
        "residual_sha256": tensor_digest(residual),
    }
    tensors = {
        "expected_exp_avg": expected_m,
        "realized_exp_avg": realized_m,
        "expected_exp_avg_sq": expected_v,
        "realized_exp_avg_sq": realized_v,
        "expected_adaptive_adam_update": expected_adaptive,
        "realized_adaptive_adam_update": realized_adaptive,
        "expected_weight_decay_update": expected_decay,
        "realized_weight_decay_update": realized_decay,
        "expected_total_update": expected_total,
        "realized_parameter_delta": realized_delta,
        "reconstruction_residual": residual,
    }
    return summary, tensors
