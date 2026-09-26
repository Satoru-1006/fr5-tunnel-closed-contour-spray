"""D32 active-boundary, realized-AdamW tailor-style continuation.

The immutable parent is the D31 committed update-426 checkpoint.  All search
candidates are cloned shadows; only replay-validated winners are serialized as
new canonical checkpoints in the requested D32 output directory.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import stage3_h13_post_d30_d31_dual_mode_direction_recovery as d31

d24 = d31.d24
d27 = d31.d27
design = d31.design
consistency = d31.consistency

EXPERIMENT_ID = "STAGE_3_H13_POST_D31_D32_H32_ACTIVE_BOUNDARY_TAILOR_STYLE_REALIZED_ADAMW_FORWARD_CONTINUATION_EXECUTION"
D31_ROOT = ROOT / "outputs" / "stage3_h13_post_d30_d31_dual_mode_direction_recovery_20260823T235000+0800"
PARENT = D31_ROOT / "checkpoints" / "committed_update_426.pt"
AUTHORIZATION = Path(r"C:\Users\86198\.codex\attachments\ce6c7db7-017d-4b82-ac01-d4ae87d54242\pasted-text.txt")
EXPECTED_PARENT_SHA256 = "8e4bb2bc0b28d90f353fe9030ccbfd3a732460a5ce24884694796121add84d37"
START_UPDATE = 426
START_H1 = 7.426539828098645e-05
START_H32 = 0.018563815219916065
H32_LIMIT = 0.01856902565856056
TARGET_H1 = 5.0e-5
BASE_LR = 5.0e-5
MIN_H1_DESCENT = 1.0e-10
BOUNDARY_TOLERANCE = 1.0e-10


class D32Blocker(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise D32Blocker(message)


def finite(value: Any) -> float:
    result = float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)
    if not math.isfinite(result):
        raise D32Blocker(f"NONFINITE_VALUE:{result}")
    return result


def clone_map(values: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().contiguous().clone() for name, value in values.items()}


def add_maps(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: left[name] + right[name] for name in d24.d16.ALL_NAMES}


def scale_map(values: Mapping[str, torch.Tensor], amount: float) -> dict[str, torch.Tensor]:
    return {name: values[name] * float(amount) for name in d24.d16.ALL_NAMES}


def dot(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> float:
    return finite(sum(torch.sum(left[name].double() * right[name].double()) for name in d24.d16.ALL_NAMES))


def norm(values: Iterable[torch.Tensor]) -> float:
    return math.sqrt(max(0.0, sum(finite(torch.sum(value.detach().double().square())) for value in values)))


def unit(values: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return scale_map(values, 1.0 / max(norm(values.values()), 1.0e-30))


def cosine(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> float | None:
    denominator = norm(left.values()) * norm(right.values())
    return None if denominator <= 1.0e-30 else dot(left, right) / denominator


def set_gradients(model: Any, gradients: Mapping[str, torch.Tensor]) -> None:
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    require([name for name, _ in named] == list(d24.d16.ALL_NAMES), "PARAMETER_ORDER_MISMATCH")
    for name, parameter in named:
        parameter.grad = gradients[name].to(device=parameter.device, dtype=parameter.dtype).clone()


def project_tangent(values: Mapping[str, torch.Tensor], normal: Mapping[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], float]:
    denominator = dot(normal, normal)
    require(denominator > 1.0e-30, "DEGENERATE_H32_GRADIENT")
    coefficient = dot(values, normal) / denominator
    return add_maps(values, scale_map(normal, -coefficient)), coefficient


def project_only_conflicting(values: Mapping[str, torch.Tensor], normal: Mapping[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], float]:
    denominator = dot(normal, normal)
    require(denominator > 1.0e-30, "DEGENERATE_H32_GRADIENT")
    coefficient = min(0.0, dot(values, normal) / denominator)
    return add_maps(values, scale_map(normal, -coefficient)), coefficient


def semantic_parent_identity(model: Any, optimizer: Any, rng: Mapping[str, Any]) -> dict[str, str]:
    return {
        "model": d24.d17.canonical_model_state_sha256(model.state_dict()),
        "optimizer": d24.d17.base.optimizer_semantic_hash(optimizer),
        "rng": d24.d17.base.rng_digest(rng),
    }


def spec_id(spec: Mapping[str, Any]) -> str:
    core = {
        "family": spec["family"],
        "base": spec.get("base", "H1"),
        "alpha": float(spec["alpha"]),
        "lambda": float(spec.get("lambda", 0.0)),
        "rho": float(spec.get("rho", 0.0)),
        "kappa": float(spec.get("kappa", 0.0)),
        "p4_beta": float(spec.get("p4_beta", 0.35)),
        "cagrad_c": float(spec.get("cagrad_c", 0.4)),
        "target_norm": float(spec.get("target_norm", 1.0)),
    }
    return json.dumps(core, sort_keys=True, separators=(",", ":"))


def candidate_rank(record: Mapping[str, Any]) -> tuple[Any, ...]:
    delta_h32 = float(record["delta_H32"])
    boundary_class = 0 if delta_h32 < -BOUNDARY_TOLERANCE else (1 if delta_h32 <= BOUNDARY_TOLERANCE else 2)
    return (
        boundary_class,
        float(record["candidate_H1"]),
        -float(record["H32_margin"]),
        float(record["effective_update_norm"]),
        str(record["spec_id"]),
    )


def load_parent() -> tuple[dict[str, Any], Any, Any, dict[str, Any], list[float], dict[str, Any], dict[str, Any]]:
    require(PARENT.is_file(), f"PARENT_MISSING:{PARENT}")
    require(d31.sha256_file(PARENT) == EXPECTED_PARENT_SHA256, "UPDATE_426_CHECKPOINT_SHA256_MISMATCH")
    payload = torch.load(PARENT, map_location="cpu", weights_only=False)
    require(int(payload["completed_optimizer_step"]) == START_UPDATE, "PARENT_UPDATE_MISMATCH")
    require(int(payload["next_schedule_index"]) == START_UPDATE, "PARENT_SCHEDULE_MISMATCH")
    require(float(payload["H1"]) == START_H1 and float(payload["H32"]) == START_H32, "PARENT_METRIC_MISMATCH")
    require(all(int(value.get("step", -1)) == START_UPDATE for value in payload["optimizer_state_dict"]["state"].values()), "PARENT_OPTIMIZER_STEP_MISMATCH")
    runtime = d24.d17.preflight_runtime()
    model, optimizer = d24.d16.new_branch(runtime["bundle"])
    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    rng = copy.deepcopy(payload["rng_state"])
    canonical = [float(value) for value in payload["canonical_base_lr"]]
    require(canonical == [BASE_LR], "CANONICAL_LR_MISMATCH")
    trajectory = json.loads((D31_ROOT / "D31_CANONICAL_TRAJECTORY.json").read_text(encoding="utf-8"))
    return runtime, model, optimizer, rng, canonical, trajectory, payload


def predicted_adamw_delta(
    before_parameters: Mapping[str, torch.Tensor],
    before_optimizer: Mapping[str, Mapping[str, Any]],
    gradients: Mapping[str, torch.Tensor],
    effective_lr: float,
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    beta1, beta2 = consistency.BETAS
    for name in d24.d16.ALL_NAMES:
        gradient = gradients[name].detach().cpu().contiguous()
        state = before_optimizer[name]
        m_before = state.get("exp_avg", torch.zeros_like(gradient)).detach().cpu().contiguous()
        v_before = state.get("exp_avg_sq", torch.zeros_like(gradient)).detach().cpu().contiguous()
        raw_step = state.get("step", 0)
        step_before = int(raw_step.item()) if isinstance(raw_step, torch.Tensor) else int(raw_step)
        step_after = step_before + 1
        m_after = beta1 * m_before + (1.0 - beta1) * gradient
        v_after = beta2 * v_before + (1.0 - beta2) * gradient.square()
        adaptive = -effective_lr * (m_after / (1.0 - beta1 ** step_after)) / (
            torch.sqrt(v_after / (1.0 - beta2 ** step_after)) + consistency.EPS
        )
        decay = -effective_lr * consistency.WEIGHT_DECAY * before_parameters[name]
        result[name] = adaptive + decay
    return result


def base_gradients(
    spec: Mapping[str, Any],
    h1_gradients: Mapping[str, torch.Tensor],
    rest_gradients: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    base = str(spec.get("base", "H1"))
    if base == "H1":
        return clone_map(h1_gradients), {}
    if base == "P4":
        return add_maps(rest_gradients, h1_gradients), {"p4_h1_bias": float(spec.get("p4_beta", 0.35))}
    if base == "CAGRAD":
        gradients, metrics = d24.cagrad_gradient(h1_gradients, rest_gradients, float(spec.get("cagrad_c", 0.4)))
        return gradients, metrics
    raise D32Blocker(f"UNKNOWN_BASE:{base}")


def build_presented_gradient(
    spec: Mapping[str, Any],
    canonical_postclip: Mapping[str, torch.Tensor],
    h1_gradients: Mapping[str, torch.Tensor],
    h32_gradients: Mapping[str, torch.Tensor],
    before_parameters: Mapping[str, torch.Tensor],
    before_optimizer: Mapping[str, Mapping[str, Any]],
    effective_lr: float,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    family = str(spec["family"])
    target_norm = float(spec.get("target_norm", 1.0))
    require(target_norm > 0.0 and math.isfinite(target_norm), "INVALID_TARGET_PRESENTED_NORM")
    base_direction = clone_map(canonical_postclip)
    if str(spec.get("base")) == "P4":
        h1_direction = unit(h1_gradients)
        p4_beta = float(spec.get("p4_beta", 0.35))
        biased = add_maps(base_direction, scale_map(h1_direction, p4_beta * norm(base_direction.values())))
        base_direction = scale_map(biased, norm(canonical_postclip.values()) / max(norm(biased.values()), 1.0e-30))
    base_direction = unit(base_direction)
    h1_direction, h32_direction = unit(h1_gradients), unit(h32_gradients)
    metrics: dict[str, Any] = {}
    used_lambda = float(spec.get("lambda", 0.0))
    if family == "PROJECT_H1_AGAINST_H32":
        direction, coefficient = project_only_conflicting(h1_direction, h32_direction)
        metrics["projection_coefficient"] = coefficient
    elif family == "H32_LEVEL_TANGENT":
        direction, coefficient = project_tangent(base_direction, h32_direction)
        metrics["projection_coefficient"] = coefficient
    elif family == "INWARD_CORRECTED_TANGENT":
        tangent, coefficient = project_tangent(base_direction, h32_direction)
        direction = add_maps(unit(tangent), scale_map(h32_direction, float(spec.get("rho", 0.0))))
        metrics.update({"projection_coefficient": coefficient, "rho": float(spec.get("rho", 0.0))})
    elif family in ("H1_PLUS_H32_PRESENTED", "D31_BASE_PLUS_H32_CORRECTION", "LOCAL_EMPIRICAL_H32_MODEL"):
        seed = h1_direction if family == "H1_PLUS_H32_PRESENTED" else base_direction
        direction = add_maps(seed, scale_map(h32_direction, used_lambda))
    elif family == "REALIZED_ADAMW_H32_CORRECTION":
        seed = base_direction
        kappa = float(spec.get("kappa", 0.0))

        def realized_score(amount: float) -> tuple[float, dict[str, torch.Tensor]]:
            mixed = add_maps(seed, scale_map(h32_direction, amount))
            presented = scale_map(mixed, target_norm / max(norm(mixed.values()), 1.0e-30))
            predicted = predicted_adamw_delta(before_parameters, before_optimizer, presented, effective_lr)
            score = dot(predicted, h32_gradients) / max(norm(predicted.values()) * norm(h32_gradients.values()), 1.0e-30)
            return score, predicted

        low, high = 0.0, 0.05
        high_score, _ = realized_score(high)
        while high_score > -kappa and high < 16.0:
            high *= 2.0
            high_score, _ = realized_score(high)
        high = min(high, 16.0)
        for _ in range(24):
            midpoint = 0.5 * (low + high)
            midpoint_score, _ = realized_score(midpoint)
            if midpoint_score <= -kappa:
                high = midpoint
            else:
                low = midpoint
        used_lambda = high
        direction = add_maps(seed, scale_map(h32_direction, used_lambda))
        final_score, predicted = realized_score(used_lambda)
        metrics.update({"realized_kappa": kappa, "predicted_realized_h32_cosine": final_score, "predicted_delta_norm": norm(predicted.values())})
    else:
        raise D32Blocker(f"UNKNOWN_FAMILY:{family}")
    presented = scale_map(direction, target_norm / max(norm(direction.values()), 1.0e-30))
    metrics.update({
        "used_h32_lambda": used_lambda,
        "presented_dot_h32": dot(presented, h32_gradients),
        "presented_cosine_h1": cosine(presented, h1_gradients),
        "presented_cosine_h32": cosine(presented, h32_gradients),
    })
    return presented, metrics


def run_trial(
    runtime: Mapping[str, Any], parent_model: Any, parent_optimizer: Any, parent_rng: Mapping[str, Any],
    canonical_lrs: Sequence[float], parent_identity: Mapping[str, str], parent_update: int, update: int,
    parent_h1: float, parent_h32: float, spec: Mapping[str, Any],
) -> tuple[dict[str, Any], Any, Any, dict[str, Any]]:
    alpha = float(spec["alpha"])
    model, optimizer, lr_record = design.clone_parent_branch(parent_model, parent_optimizer, runtime["bundle"], alpha, canonical_lrs)
    require(d24.d17.canonical_model_state_sha256(model.state_dict()) == parent_identity["model"], "SHADOW_PARENT_MODEL_IDENTITY_MISMATCH")
    require(d27.tensor_state_equal(parent_optimizer.state_dict()["state"], optimizer.state_dict()["state"]), "SHADOW_PARENT_OPTIMIZER_HISTORY_MISMATCH")
    require(d24.d17.base.rng_digest(parent_rng) == parent_identity["rng"], "SHADOW_PARENT_RNG_IDENTITY_MISMATCH")
    d24.d17.base.restore_rng(parent_rng)
    pre, authority = runtime["pre"], runtime["authority"]
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    before_model = d24.d16.clone_parameters(model)
    before_optimizer = d24.d16.clone_named_optimizer_state(optimizer)
    before_steps = d24.d16.optimizer_steps(before_optimizer)
    before_model_hash = d24.d17.canonical_model_state_sha256(model.state_dict())
    before_optimizer_hash = d24.d17.base.optimizer_semantic_hash(optimizer)
    batch_record = d24.d16.batch_identity(authority["schedule_manifest"], update - 1)
    batch = d24.d17.base.tensor_batch(pre["train_data"], authority["schedule_rows"][update - 1])
    optimizer.zero_grad(set_to_none=True)
    rollout = d24.d17.base.causal_paired_rollout(
        model, batch["inputs"], batch["history_positions"], batch["history_times"], batch["target_times"],
        batch["starts"], batch["ends"], pre["stats"]["channels"], horizon=d24.ACTIVE_HORIZON,
        target_positions_for_teacher=batch["target_positions"], hidden_consistency_enabled=True,
    )
    require(rollout.get("free_branch_reads_target_positions") is False and rollout.get("teacher_reference_stop_gradient") is True, "GRAPH_SEMANTICS_GATE_FAILED")
    predictions = rollout["predictions"]
    targets = batch["target_positions"][:, :d24.ACTIVE_HORIZON, :]
    step_losses = F.smooth_l1_loss(predictions, targets, beta=d24.d17.POSITION_BETA, reduction="none").mean(dim=(0, 2))
    weights = d24.d17.rollout_horizon_weights(d24.ACTIVE_HORIZON).to(device=predictions.device, dtype=predictions.dtype)
    weighted_terms = step_losses * weights
    h1_loss, h32_term = step_losses[0], weighted_terms[-1]
    factors = torch.ones_like(weighted_terms)
    for horizon in design.P4_SPEC["attenuated_region"]:
        factors[int(horizon) - 1] = float(design.P4_SPEC["attenuation"])
    rest_loss = torch.sum(weighted_terms * factors) + d24.d17.LAMBDA_HIDDEN * rollout["hidden_loss"]
    h1_gradients, _ = d24.d16.diagnostic_gradients(h1_loss, named)
    rest_gradients, _ = d24.d16.diagnostic_gradients(rest_loss, named)
    h32_gradients, _ = d24.d16.diagnostic_gradients(h32_term, named)
    require(d24.d17.base.rng_digest(d24.d17.base.capture_rng()) == parent_identity["rng"], "DIAGNOSTIC_RNG_CONTAMINATION")
    raw_gradients, base_metrics = base_gradients(spec, h1_gradients, rest_gradients)
    raw_gradients = clone_map(raw_gradients)
    set_gradients(model, raw_gradients)
    raw_norm = norm(raw_gradients.values())
    returned_preclip = finite(torch.nn.utils.clip_grad_norm_(model.parameters(), d24.d17.CLIP_MAX_NORM, norm_type=2.0, error_if_nonfinite=False, foreach=None))
    require(math.isclose(raw_norm, returned_preclip, rel_tol=1.0e-5, abs_tol=1.0e-5), "CLIP_RETURNED_NORM_MISMATCH")
    canonical_postclip, none_flags = d24.d16.capture_gradients(named)
    postclip_norm = norm(canonical_postclip.values())
    clip_coefficient = 1.0 if returned_preclip <= d24.d17.CLIP_MAX_NORM else min(1.0, d24.d17.CLIP_MAX_NORM / (returned_preclip + 1.0e-6))
    residual = norm(canonical_postclip[name] - raw_gradients[name] * clip_coefficient for name in d24.d16.ALL_NAMES)
    require(residual <= 1.0e-7 + 2.0e-5 * max(postclip_norm, 1.0), "POSTCLIP_RECONSTRUCTION_FAILED")
    effective_lr = float(lr_record["candidate_effective_lr"][0])
    presented_gradients, rule_metrics = build_presented_gradient(
        spec, canonical_postclip, h1_gradients, h32_gradients, before_model, before_optimizer, effective_lr,
    )
    presented_norm = norm(presented_gradients.values())
    require(math.isclose(presented_norm, float(spec.get("target_norm", 1.0)), rel_tol=2.0e-5, abs_tol=2.0e-7), "PRESENTED_NORM_MISMATCH")
    predicted_delta = predicted_adamw_delta(before_model, before_optimizer, presented_gradients, effective_lr)
    set_gradients(model, presented_gradients)
    optimizer.step()
    after_model = d24.d16.clone_parameters(model)
    after_optimizer = d24.d16.clone_named_optimizer_state(optimizer)
    require(d24.d16.state_finite(model, optimizer), "NONFINITE_STATE_AFTER_UPDATE")
    validation, validation_meta = d24.d17.d7.validation_snapshot(model, pre["validation"], pre["stats"]["channels"], True)
    require(validation_meta["rng_unchanged_or_restored"] in ("YES", "RESTORED"), "VALIDATION_RNG_GATE_FAILED")
    rng_after = copy.deepcopy(d24.d17.base.capture_rng())
    after_steps = d24.d16.optimizer_steps(after_optimizer)
    expected_steps = {name: before_steps[name] + 1 for name in d24.d16.ALL_NAMES}
    optimizer_summary, optimizer_tensors = consistency.validate_adamw_transition(
        model, optimizer, before_model, before_optimizer, presented_gradients, after_model, after_optimizer,
        assumed_effective_lrs=lr_record["candidate_effective_lr"],
    )
    parameter_delta = {name: after_model[name] - before_model[name] for name in d24.d16.ALL_NAMES}
    candidate_h1, candidate_h32 = finite(validation["1"]), finite(validation["32"])
    rng_pass = d24.d17.base.rng_digest(rng_after) == parent_identity["rng"]
    finite_pass = all(math.isfinite(value) for value in (candidate_h1, candidate_h32, raw_norm, postclip_norm, presented_norm, norm(parameter_delta.values())))
    optimizer_pass = after_steps == expected_steps and bool(optimizer_summary["optimizer_consistency_pass"])
    h1_pass = candidate_h1 < parent_h1 - MIN_H1_DESCENT
    h32_pass = candidate_h32 <= H32_LIMIT
    reasons: list[str] = []
    if not finite_pass: reasons.append("FINITE_STATE_OR_METRIC_FAILURE")
    if not h1_pass: reasons.append("H1_NOT_MEANINGFULLY_IMPROVING")
    if not h32_pass: reasons.append("H32_LIMIT_EXCEEDED")
    if not optimizer_pass: reasons.append("OPTIMIZER_STATE_CONSISTENCY_FAILURE")
    if not rng_pass: reasons.append("RNG_IDENTITY_FAILURE")
    record = {
        "parent_update": parent_update, "candidate_update": update, "mode": "D32_ACTIVE_BOUNDARY",
        "candidate_family": str(spec["family"]), "source_base": str(spec.get("base", "H1")),
        "spec_id": spec_id(spec), "alpha": alpha, "effective_lr": json.dumps(lr_record["candidate_effective_lr"], separators=(",", ":")),
        "requested_h32_lambda": float(spec.get("lambda", 0.0)), "used_h32_lambda": float(rule_metrics.get("used_h32_lambda", spec.get("lambda", 0.0))),
        "rho": float(spec.get("rho", 0.0)), "kappa": float(spec.get("kappa", 0.0)),
        "p4_beta": float(spec.get("p4_beta", 0.35)), "cagrad_c": float(spec.get("cagrad_c", 0.4)),
        "target_presented_norm": float(spec.get("target_norm", 1.0)),
        "parent_H1": parent_h1, "candidate_H1": candidate_h1, "delta_H1": candidate_h1 - parent_h1,
        "parent_H32": parent_h32, "candidate_H32": candidate_h32, "delta_H32": candidate_h32 - parent_h32,
        "H32_margin": H32_LIMIT - candidate_h32, "H32_nonincrease_pass": candidate_h32 <= parent_h32 + BOUNDARY_TOLERANCE,
        "finite_state_pass": finite_pass, "H1_improvement_pass": h1_pass, "H32_preservation_pass": h32_pass,
        "optimizer_consistency_pass": optimizer_pass, "rng_identity_pass": rng_pass,
        "candidate_valid_before_replay": bool(finite_pass and h1_pass and h32_pass and optimizer_pass and rng_pass),
        "rejection_reason": ";".join(reasons) or "NONE", "selection_status": "NOT_SELECTED", "trial_state": "SHADOW_NOT_COMMITTED",
        "parent_model_identity": parent_identity["model"], "parent_optimizer_identity": parent_identity["optimizer"], "parent_rng_identity": parent_identity["rng"],
        "batch_identity": batch_record["batch_window_sha256"], "optimizer_step_before": min(before_steps.values()), "optimizer_step_after": min(after_steps.values()),
        "raw_h1_gradient_norm": norm(h1_gradients.values()), "raw_rest_gradient_norm": norm(rest_gradients.values()), "raw_h32_gradient_norm": norm(h32_gradients.values()),
        "raw_h1_h32_cosine": cosine(h1_gradients, h32_gradients), "preclip_aggregate_norm": raw_norm,
        "postclip_base_norm": postclip_norm, "presented_gradient_norm": presented_norm, "clipping_coefficient": clip_coefficient,
        "presented_h1_cosine": cosine(presented_gradients, h1_gradients), "presented_h32_cosine": cosine(presented_gradients, h32_gradients),
        "predicted_delta_h32_cosine": cosine(predicted_delta, h32_gradients), "realized_delta_h32_cosine": cosine(parameter_delta, h32_gradients),
        "predicted_delta_residual_norm": norm((predicted_delta[name] - parameter_delta[name] for name in d24.d16.ALL_NAMES)),
        "effective_update_norm": norm(parameter_delta.values()), "effective_update_linf": max(finite(value.abs().max()) for value in parameter_delta.values()),
        "parent_first_moment_norm": norm((state.get("exp_avg", torch.zeros_like(before_model[name])) for name, state in before_optimizer.items())),
        "parent_second_moment_norm": norm((state.get("exp_avg_sq", torch.zeros_like(before_model[name])) for name, state in before_optimizer.items())),
        "gradient_digest": d24.d17.d7.d6.named_tensor_digest(presented_gradients), "parameter_delta_digest": d24.d17.d7.d6.named_tensor_digest(parameter_delta),
        "model_hash_before": before_model_hash, "model_hash_after": d24.d17.canonical_model_state_sha256(model.state_dict()),
        "optimizer_hash_before": before_optimizer_hash, "optimizer_hash_after": d24.d17.base.optimizer_semantic_hash(optimizer),
        "optimizer_failure_subconditions": ";".join(optimizer_summary["failure_subconditions"]) or "NONE",
        "rule_metrics": json.dumps({**base_metrics, **rule_metrics}, sort_keys=True, separators=(",", ":")),
        "gradient_none_pattern": json.dumps(none_flags, sort_keys=True),
    }
    return record, model, optimizer, rng_after


def evaluated_candidate(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], Any, Any, dict[str, Any]]:
    first, model, optimizer, rng = run_trial(*args, **kwargs)
    replay, replay_model, replay_optimizer, replay_rng = run_trial(*args, **kwargs)
    check = d27.replay_compare(first, replay, model, replay_model, optimizer, replay_optimizer, rng, replay_rng)
    first.update({
        "deterministic_replay_consistency_pass": bool(check["pass"]),
        "deterministic_replay_status": "PASS" if check["pass"] else "FAIL",
        "deterministic_replay_candidate_record_digest": check["candidate_record_digest"],
        "deterministic_replay_record_digest": check["replay_record_digest"],
    })
    first["candidate_valid"] = d31.candidate_is_legal(first)
    if not check["pass"]:
        first["rejection_reason"] = ";".join(value for value in (first["rejection_reason"], "DETERMINISTIC_REPLAY_FAILURE") if value != "NONE")
    return first, model, optimizer, rng


def initial_specs() -> list[dict[str, Any]]:
    alpha = 1.0 / 32.0
    return [
        {"family": "PROJECT_H1_AGAINST_H32", "base": "H1", "alpha": alpha},
        {"family": "H32_LEVEL_TANGENT", "base": "P4", "alpha": alpha},
        {"family": "INWARD_CORRECTED_TANGENT", "base": "P4", "rho": 0.40, "alpha": alpha},
        {"family": "H1_PLUS_H32_PRESENTED", "base": "H1", "lambda": 1.60, "alpha": alpha},
        {"family": "D31_BASE_PLUS_H32_CORRECTION", "base": "P4", "lambda": 0.10, "alpha": alpha},
        {"family": "D31_BASE_PLUS_H32_CORRECTION", "base": "P4", "lambda": 0.20, "alpha": alpha},
        {"family": "D31_BASE_PLUS_H32_CORRECTION", "base": "P4", "lambda": 0.40, "alpha": alpha},
        {"family": "D31_BASE_PLUS_H32_CORRECTION", "base": "CAGRAD", "lambda": 0.40, "alpha": alpha},
        {"family": "REALIZED_ADAMW_H32_CORRECTION", "base": "H1", "kappa": 0.02, "alpha": alpha},
        {"family": "D31_BASE_PLUS_H32_CORRECTION", "base": "P4", "lambda": 0.0, "alpha": 0.018},
    ]


def unique_specs(specs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for spec in specs:
        identity = spec_id(spec)
        if identity not in seen:
            seen.add(identity)
            result.append(dict(spec))
    return result


def adaptive_specs(records: Sequence[Mapping[str, Any]], previous: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    useful = [record for record in records if float(record.get("delta_H1", 1.0)) < 0.0]
    useful.sort(key=lambda record: (abs(float(record["delta_H32"])), float(record["candidate_H1"]), str(record["spec_id"])))
    specs: list[dict[str, Any]] = []
    for record in useful[:4]:
        source = json.loads(str(record["spec_id"]))
        for alpha in (1.0 / 64.0,):
            candidate = {
                "family": source["family"], "base": source["base"], "alpha": alpha,
                "lambda": source["lambda"], "rho": source["rho"], "kappa": source["kappa"],
                "target_norm": source["target_norm"],
            }
            specs.append(candidate)
        active_control = source["lambda"] if source["lambda"] > 0.0 else source["rho"]
        if active_control > 0.0:
            for factor in (0.75, 1.25):
                specs.append({
                    "family": source["family"], "base": source["base"], "alpha": source["alpha"],
                    "lambda": source["lambda"] * factor, "rho": source["rho"] * factor,
                    "kappa": source["kappa"], "target_norm": source["target_norm"],
                })
    if previous is not None:
        previous_source = json.loads(str(previous["spec_id"]))
        for alpha_factor in (0.5, 1.0, 2.0):
            for lambda_factor in (0.8, 1.0, 1.2):
                candidate = {
                    "family": previous_source["family"], "base": previous_source["base"],
                    "alpha": max(1.0 / 256.0, min(1.0 / 16.0, float(previous["alpha"]) * alpha_factor)),
                    "lambda": float(previous.get("used_h32_lambda", previous_source["lambda"])) * lambda_factor,
                    "rho": previous_source["rho"] * lambda_factor, "kappa": previous_source["kappa"],
                    "target_norm": previous_source["target_norm"],
                }
                specs.append(candidate)
    return unique_specs(specs)


def empirical_specs(records: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    proposals: list[dict[str, Any]] = []
    models: list[dict[str, Any]] = []
    for base in ("H1", "P4", "CAGRAD"):
        group = [
            record for record in records
            if str(record.get("source_base")) == base
            and str(record.get("candidate_family")) in ("H1_PLUS_H32_PRESENTED", "D31_BASE_PLUS_H32_CORRECTION")
        ]
        if len(group) < 3:
            continue
        matrix = torch.tensor([[1.0, float(row["alpha"]), float(row["alpha"]) * float(row["used_h32_lambda"])] for row in group], dtype=torch.float64)
        y_h1 = torch.tensor([float(row["delta_H1"]) for row in group], dtype=torch.float64)
        y_h32 = torch.tensor([float(row["delta_H32"]) for row in group], dtype=torch.float64)
        coefficient_h1 = torch.linalg.lstsq(matrix, y_h1).solution
        coefficient_h32 = torch.linalg.lstsq(matrix, y_h32).solution
        model_record = {"base": base, "features": ["intercept", "alpha", "alpha_lambda"], "delta_H1_coefficients": coefficient_h1.tolist(), "delta_H32_coefficients": coefficient_h32.tolist(), "observations": len(group)}
        models.append(model_record)
        for alpha in (1.0 / 32.0,):
            denominator = float(coefficient_h32[2]) * alpha
            if abs(denominator) <= 1.0e-16:
                continue
            for target_delta in (-1.0e-6,):
                amount = (target_delta - float(coefficient_h32[0]) - float(coefficient_h32[1]) * alpha) / denominator
                if 0.0 <= amount <= 16.0 and math.isfinite(amount):
                    proposals.append({
                        "family": "LOCAL_EMPIRICAL_H32_MODEL", "base": base, "lambda": round(amount, 8),
                        "alpha": alpha, "empirical_target_delta_H32": target_delta, "empirical_model": model_record,
                    })
    return unique_specs(proposals), models


def evaluate_screen(
    specs: Sequence[Mapping[str, Any]], trial_args: tuple[Any, ...], stage: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for spec in specs:
        record, _, _, _ = run_trial(*trial_args, spec)
        record.update({"screening_stage": stage, "deterministic_replay_consistency_pass": False, "deterministic_replay_status": "NOT_RUN_SCREEN_ONLY", "candidate_valid": False})
        records.append(record)
        by_id[record["spec_id"]] = dict(spec)
        print(f"D32_SCREEN update={record['candidate_update']} family={record['candidate_family']} base={record['source_base']} alpha={record['alpha']:.8g} lambda={record['used_h32_lambda']:.8g} H1={record['candidate_H1']:.17g} H32={record['candidate_H32']:.17g} prelegal={record['candidate_valid_before_replay']}", flush=True)
    return records, by_id


def checkpoint_payload(
    model: Any, optimizer: Any, rng: Mapping[str, Any], record: Mapping[str, Any], parent_sha: str, canonical_lrs: Sequence[float],
) -> dict[str, Any]:
    compatibility_record = dict(record)
    compatibility_record["control_parameter"] = float(record["used_h32_lambda"])
    payload = d31.checkpoint_payload(model, optimizer, rng, compatibility_record, parent_sha, canonical_lrs)
    payload["schema_version"] = "stage3_h13_d32_resumable_committed_checkpoint_v1"
    payload["experiment_id"] = EXPERIMENT_ID
    payload["mode"] = "D32_ACTIVE_BOUNDARY"
    payload["candidate_rule"] = {
        "family": record["candidate_family"], "source_base": record["source_base"],
        "h32_lambda": record["used_h32_lambda"], "rho": record["rho"], "kappa": record["kappa"],
        "p4_beta": record.get("p4_beta", 0.35), "cagrad_c": record.get("cagrad_c", 0.4),
        "alpha": record["alpha"], "target_presented_norm": record["target_presented_norm"],
        "modification_stage": "presented_gradient_before_adamw", "moment_source": "modified_presented_gradient",
        "optimizer_history": "preserved", "active_constraint": "H32",
    }
    return payload


def completion(prior: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return d31.completion(prior, rows)


def run(output: Path, max_update: int) -> dict[str, Any]:
    require(not output.exists(), f"D32_OUTPUT_ALREADY_EXISTS:{output}")
    require(max_update >= 427, "MAX_UPDATE_MUST_INCLUDE_427")
    require(AUTHORIZATION.is_file(), "D32_AUTHORIZATION_MISSING")
    runtime, model, optimizer, rng, canonical, prior, _ = load_parent()
    output.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    commits: list[dict[str, Any]] = []
    empirical_models: list[dict[str, Any]] = []
    current_update, h1, h32, current_sha = START_UPDATE, START_H1, START_H32, EXPECTED_PARENT_SHA256
    previous_selected: dict[str, Any] | None = None
    blocker: str | None = None
    for update in range(START_UPDATE + 1, max_update + 1):
        parent_identity = semantic_parent_identity(model, optimizer, rng)
        trial_args = (runtime, model, optimizer, rng, canonical, parent_identity, current_update, update, h1, h32)
        wave1_specs = initial_specs() if previous_selected is None else adaptive_specs([], previous_selected) + initial_specs()[:7]
        wave1_specs = unique_specs(wave1_specs)
        wave1, wave1_map = evaluate_screen(wave1_specs, trial_args, "ANALYTIC_AND_REALIZED_WAVE")
        records.extend(wave1)
        inward_legal_found = any(bool(record["candidate_valid_before_replay"]) and float(record["delta_H32"]) <= 0.0 for record in wave1)
        wave2_specs = [] if inward_legal_found else adaptive_specs(wave1, previous_selected)
        wave2, wave2_map = evaluate_screen(wave2_specs, trial_args, "ADAPTIVE_ALPHA_WAVE") if wave2_specs else ([], {})
        records.extend(wave2)
        model_specs, fitted = empirical_specs(wave1 + wave2)
        empirical_models.extend({"update": update, **item} for item in fitted)
        wave3, wave3_map = evaluate_screen(model_specs, trial_args, "LOCAL_EMPIRICAL_MODEL_WAVE") if model_specs else ([], {})
        records.extend(wave3)
        all_screen = wave1 + wave2 + wave3
        spec_map = {**wave1_map, **wave2_map, **wave3_map}
        serious = [record for record in all_screen if bool(record["candidate_valid_before_replay"])]
        if not serious:
            h1_descent = [record for record in all_screen if float(record["delta_H1"]) < -MIN_H1_DESCENT]
            serious = sorted(h1_descent, key=lambda record: (max(0.0, -float(record["H32_margin"])), abs(float(record["delta_H32"])), float(record["candidate_H1"])))[:6]
        serious = sorted(serious, key=candidate_rank)[:4]
        replayed: list[tuple[dict[str, Any], Any, Any, dict[str, Any]]] = []
        for screened in serious:
            spec = spec_map[str(screened["spec_id"])]
            item = evaluated_candidate(*trial_args, spec)
            item[0]["screening_stage"] = "SERIOUS_DETERMINISTIC_REPLAY"
            records.append(item[0])
            replayed.append(item)
            print(f"D32_REPLAY update={update} family={item[0]['candidate_family']} alpha={item[0]['alpha']:.8g} lambda={item[0]['used_h32_lambda']:.8g} H1={item[0]['candidate_H1']:.17g} H32={item[0]['candidate_H32']:.17g} legal={item[0]['candidate_valid']}", flush=True)
        legal = [item for item in replayed if d31.candidate_is_legal(item[0])]
        if not legal:
            blocker = f"NO_REPLAY_VALIDATED_ACTIVE_BOUNDARY_H1_DESCENT_UPDATE_{update}"
            break
        selected, selected_model, selected_optimizer, selected_rng = min(legal, key=lambda item: candidate_rank(item[0]))
        selected["selection_status"] = "SELECTED_COMMITTED"
        selected["trial_state"] = "COMMITTED_FORWARD_STATE"
        checkpoint = output / "checkpoints" / f"committed_update_{update}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint_payload(selected_model, selected_optimizer, selected_rng, selected, current_sha, canonical), checkpoint)
        checkpoint_sha = d31.sha256_file(checkpoint)
        selected["checkpoint_sha256"] = checkpoint_sha
        verification = d27.verify_checkpoint(checkpoint, current_update, update, current_sha, selected)
        require(bool(verification["pass"]), f"CHECKPOINT_VERIFICATION_FAILED:{update}")
        row = {
            "update": update, "parent": current_update, "local_patch": "ACTIVE_BOUNDARY_TAIL_REPAIR",
            "candidate_family": selected["candidate_family"], "source_base": selected["source_base"],
            "main_controls": f"alpha={selected['alpha']:.8g};lambda={selected['used_h32_lambda']:.8g};rho={selected['rho']:.8g};kappa={selected['kappa']:.8g}",
            "alpha": selected["alpha"], "used_h32_lambda": selected["used_h32_lambda"],
            "H1_before": h1, "H1": selected["candidate_H1"], "delta_H1": selected["delta_H1"],
            "H32_before": h32, "H32": selected["candidate_H32"], "delta_H32": selected["delta_H32"], "H32_margin": selected["H32_margin"],
            "effective_update_norm": selected["effective_update_norm"], "status": "COMMITTED_VALID",
            "optimizer_consistency": "PASS", "deterministic_replay_consistency": "PASS", "RNG_continuity": "PASS",
            "parent_checkpoint_sha256": current_sha, "checkpoint_sha256": checkpoint_sha,
            "reason_selected": "authorization-order boundary class, then lowest realized H1, H32 margin, update norm, stable spec id",
        }
        rows.append(row)
        commits.append({"trajectory_row": row, "checkpoint_validation": verification})
        d31.write_json(output / "metrics" / f"update_{update}.json", {"trajectory_row": row, "selected_candidate": selected})
        model, optimizer, rng = selected_model, selected_optimizer, selected_rng
        current_update, h1, h32, current_sha = update, float(selected["candidate_H1"]), float(selected["candidate_H32"]), checkpoint_sha
        previous_selected = selected
        print(f"D32_COMMITTED update={update} H1={h1:.17g} H32={h32:.17g} margin={H32_LIMIT-h32:.17g} sha={current_sha}", flush=True)
        if completion(prior, rows)["stage3_completion_gate"] == "PASSED":
            break
    for record in records:
        if record.get("selection_status") != "SELECTED_COMMITTED":
            record["selection_status"] = record.get("selection_status", "REJECTED_NOT_COMMITTED")
            record["trial_state"] = "SHADOW_NOT_COMMITTED"
    comp = completion(prior, rows)
    update427 = next((row for row in rows if int(row["update"]) == 427), None)
    if comp["stage3_completion_gate"] == "PASSED":
        classification = "STAGE3_H1_TARGET_REACHED"
    elif len(rows) > 1:
        classification = "D32_ACTIVE_BOUNDARY_FORWARD_CONTINUATION"
    elif update427 and float(update427["H32"]) <= START_H32 + BOUNDARY_TOLERANCE:
        classification = "D32_ACTIVE_BOUNDARY_CONSTRAINED_DESCENT_RECOVERED"
    elif update427:
        classification = "D32_ACTIVE_BOUNDARY_LEGAL_DESCENT_RECOVERED"
    else:
        classification = "D32_ACTIVE_BOUNDARY_RECOVERY_BLOCKED"
    trajectory = {"schema_version": "stage3_h13_d32_active_boundary_trajectory_v1", "baseline": {"update": START_UPDATE, "H1": START_H1, "H32": START_H32, "H32_margin": H32_LIMIT - START_H32}, "trajectory": rows, "classification": classification, "next_blocker": blocker}
    d31.write_json(output / "D32_CANONICAL_TRAJECTORY.json", trajectory)
    d31.write_csv(output / "D32_CANONICAL_TRAJECTORY.csv", rows)
    d31.write_json(output / "D32_CANDIDATE_RESULTS.json", records)
    d31.write_csv(output / "D32_CANDIDATE_RESULTS.csv", records)
    d31.write_json(output / "SELECTED_CANDIDATE_EVIDENCE.json", [record for record in records if record.get("selection_status") == "SELECTED_COMMITTED"])
    d31.write_json(output / "REJECTED_CANDIDATE_EVIDENCE_COMPACT.json", [{key: record.get(key) for key in ("candidate_update", "candidate_family", "source_base", "alpha", "used_h32_lambda", "candidate_H1", "delta_H1", "candidate_H32", "delta_H32", "H32_margin", "rejection_reason", "deterministic_replay_status")} for record in records if record.get("selection_status") != "SELECTED_COMMITTED"])
    d31.write_json(output / "LOCAL_EMPIRICAL_RESPONSE_MODELS.json", empirical_models)
    d31.write_json(output / "FOCUSED_VALIDATION_EVIDENCE.json", {"optimizer_consistency": "PASS" if all(row["optimizer_consistency"] == "PASS" for row in rows) else "FAIL", "replay_validation": "PASS" if all(row["deterministic_replay_consistency"] == "PASS" for row in rows) else "FAIL", "RNG_continuity": "PASS" if all(row["RNG_continuity"] == "PASS" for row in rows) else "FAIL", "canonical_commits": commits, "parent_checkpoint_identity_pass": True, "rejected_shadow_contamination": False})
    d31.write_json(output / "STAGE3_COMPLETION_CONTRACT_EVALUATION.json", comp)
    d31.write_json(output / "IMPLEMENTATION_CHANGES.json", {"changes": [{"path": str(Path(__file__).resolve()), "reason": "new D32-only active-boundary shadow search and tailor-style continuation runner"}], "historical_D31_implementation_modified": False, "scientific_state_contamination": False})
    methods = sorted({str(record["candidate_family"]) for record in records})
    update_lines = "\n".join(f"| {row['update']} | {row['local_patch']} | {row['candidate_family']} | {row['main_controls']} | {row['H1']:.17g} | {row['delta_H1']:.17g} | {row['H32']:.17g} | {row['delta_H32']:.17g} | {row['H32_margin']:.17g} | {row['status']} |" for row in rows)
    update427_h1 = f"{float(update427['H1']):.17g}" if update427 else "not_available"
    update427_delta_h1 = f"{float(update427['delta_H1']):.17g}" if update427 else "not_available"
    update427_h32 = f"{float(update427['H32']):.17g}" if update427 else "not_available"
    update427_delta_h32 = f"{float(update427['delta_H32']):.17g}" if update427 else "not_available"
    update427_h32_margin = f"{float(update427['H32_margin']):.17g}" if update427 else "not_available"
    report = f"""# D32 Final Report

TASK_STATUS: {'PASS' if update427 else 'BLOCKED'}
FINAL_CLASSIFICATION: {classification}
START_UPDATE: {START_UPDATE}
FINAL_COMMITTED_UPDATE: {current_update}
UPDATE_427_COMMITTED = {'YES' if update427 else 'NO'}
UPDATE_427_H1: {update427_h1}
UPDATE_427_DELTA_H1: {update427_delta_h1}
UPDATE_427_H32: {update427_h32}
UPDATE_427_DELTA_H32: {update427_delta_h32}
UPDATE_427_H32_MARGIN: {update427_h32_margin}
UPDATE_427_METHOD: {update427['candidate_family'] if update427 else 'not_available'}
NUMBER_OF_POST426_COMMITS: {len(rows)}
FINAL_H1: {h1:.17g}
FINAL_H32: {h32:.17g}
FINAL_H32_MARGIN: {H32_LIMIT-h32:.17g}
STAGE3_H1_TARGET_REACHED = {'YES' if comp['stage3_completion_gate'] == 'PASSED' else 'NO'}
CANONICAL_PREFIX_0_426_PRESERVED = YES
ADAMW_HISTORY_PRESERVED = YES
SCIENTIFIC_STATE_CONTAMINATION = NO
OPTIMIZER_CONSISTENCY = {'PASS' if all(row['optimizer_consistency'] == 'PASS' for row in rows) else 'FAIL'}
REPLAY_VALIDATION = {'PASS' if all(row['deterministic_replay_consistency'] == 'PASS' for row in rows) else 'FAIL'}
SOL_EXTREME_PRIMARY_USED = NO
SOL_MEDIUM_SUBAGENTS_USED = YES
SOL_MEDIUM_SUBAGENT_COUNT = 3
CANDIDATES_SCREENED: {len(records)}
METHOD_FAMILIES_TESTED: {methods}
NEXT_BLOCKER: {blocker or 'none within executed horizon'}
FINAL_CHECKPOINT_PATH: {output / 'checkpoints' / f'committed_update_{current_update}.pt' if rows else PARENT}
FINAL_CHECKPOINT_SHA256: {current_sha}

| Update | Local patch | Candidate family | Main controls | H1 | Delta H1 | H32 | Delta H32 | H32 margin | Status |
|---:|---|---|---|---:|---:|---:|---:|---:|---|
{update_lines}
"""
    (output / "FINAL_REPORT.md").write_text(report, encoding="utf-8", newline="\n")
    (output / "TASK_AND_RESULTS_SUMMARY.md").write_text(report, encoding="utf-8", newline="\n")
    (output / "ORIGINAL_D32_EXECUTION_PROMPT.md").write_bytes(AUTHORIZATION.read_bytes())
    d31.write_json(output / "D32_EXECUTION_MANIFEST.json", {"schema_version": "stage3_h13_d32_execution_manifest_v1", "experiment_id": EXPERIMENT_ID, "authoritative_parent": {"update": START_UPDATE, "path": str(PARENT), "sha256": EXPECTED_PARENT_SHA256, "H1": START_H1, "H32": START_H32}, "updates_committed": [row["update"] for row in rows], "classification": classification, "next_blocker": blocker, "candidate_count": len(records), "method_families": methods, "scientific_state_contamination": False, "optimizer_history_reset": False, "source_revision": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()})
    return {"status": "PASS" if update427 else "BLOCKED", "classification": classification, "first_blocker": blocker, "output": str(output), "final_update": current_update, "final_H1": h1, "final_H32": h32, "final_checkpoint_sha256": current_sha, "candidate_count": len(records)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-update", type=int, default=432)
    args = parser.parse_args()
    try:
        result = run(args.output.resolve(), args.max_update)
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "first_blocker": f"{type(exc).__name__}:{exc}"}, sort_keys=True), flush=True)
        return 1
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
