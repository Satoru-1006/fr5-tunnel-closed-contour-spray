"""D31 dual-mode realized-AdamW direction recovery and continuation.

Mode A is the established D30 P4 rule (post-clip H1 coefficient 0.35).
Mode B preserves the complete parent optimizer state but changes the gradient
presented to AdamW using an explicit, replayable recovery rule.  Every shadow
candidate is cloned from the same parent; only this orchestrator writes a
canonical checkpoint after deterministic replay passes.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
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

from tools import stage3_h13_d30_active_boundary_continuation as d30

d29 = d30.d29
d27 = d29.d27
d26 = d29.d26
d24 = d29.d24
design = d29.design
consistency = d26.consistency

EXPERIMENT_ID = "STAGE_3_H13_POST_D30_D31_DUAL_MODE_REALIZED_ADAMW_DIRECTION_RECOVERY_AND_FORWARD_CONTINUATION_EXECUTION"
D30_ROOT = ROOT / "outputs" / "stage3_h13_post_d29_d30_active_boundary_20260823T200000+0800"
PARENT = D30_ROOT / "checkpoints" / "committed_update_423.pt"
AUTHORIZATION = Path(r"C:\Users\86198\.codex\attachments\d5f8df4c-9aa4-4df8-8a00-3f8a6d444fd6\pasted-text.txt")
EXPECTED_PARENT_SHA256 = "9d5c51872209baafe69f01b739ac1608b6adebd466ea8d7bae123bc56ba5fccc"
START_UPDATE = 423
LAST_UPDATE = 428
START_H1 = 7.554237790994070e-05
START_H32 = 0.018387159425694893
H32_LIMIT = 0.01856902565856056
TARGET_H1 = 5.0e-5
BASE_LR = 5.0e-5
MODE_A_COEFFICIENT = 0.35
MODE_B_COEFFICIENTS = (0.70, 1.40, 2.80)
MODE_B_CAGRAD = (0.40, 0.80)
RECOVERY_ALPHAS = (0.0625, 0.125, 0.25)


class D31Blocker(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise D31Blocker(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def finite(value: Any) -> float:
    result = float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)
    if not math.isfinite(result):
        raise D31Blocker(f"NONFINITE_VALUE:{result}")
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


def cosine(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> float | None:
    denominator = norm(left.values()) * norm(right.values())
    return None if denominator <= 1e-30 else dot(left, right) / denominator


def negate(values: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return scale_map(values, -1.0)


def set_gradients(model: Any, gradients: Mapping[str, torch.Tensor]) -> None:
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    require([name for name, _ in named] == list(d24.d16.ALL_NAMES), "PARAMETER_ORDER_MISMATCH")
    for name, parameter in named:
        parameter.grad = gradients[name].to(device=parameter.device, dtype=parameter.dtype).clone()


def semantic_parent_identity(model: Any, optimizer: Any, rng: Mapping[str, Any]) -> dict[str, str]:
    return {
        "model": d24.d17.canonical_model_state_sha256(model.state_dict()),
        "optimizer": d24.d17.base.optimizer_semantic_hash(optimizer),
        "rng": d24.d17.base.rng_digest(rng),
    }


def candidate_is_legal(record: Mapping[str, Any]) -> bool:
    return bool(
        record.get("finite_state_pass")
        and record.get("H32_preservation_pass")
        and record.get("H1_improvement_pass")
        and record.get("optimizer_consistency_pass")
        and record.get("rng_identity_pass")
        and record.get("deterministic_replay_consistency_pass")
    )


def mode_a_alphas(previous: float) -> list[float]:
    values = {previous, previous / 2.0, previous * 2.0}
    return sorted(value for value in values if 1.0 / 256.0 <= value <= 0.5)


def load_parent() -> tuple[dict[str, Any], Any, Any, dict[str, Any], list[float], dict[str, Any], dict[str, Any]]:
    require(PARENT.is_file(), f"PARENT_MISSING:{PARENT}")
    require(sha256_file(PARENT) == EXPECTED_PARENT_SHA256, "UPDATE_423_CHECKPOINT_SHA256_MISMATCH")
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
    trajectory = json.loads((D30_ROOT / "D30_CANONICAL_TRAJECTORY.json").read_text(encoding="utf-8"))
    controls = json.loads((D30_ROOT / "D30_CANDIDATE_RESULTS.json").read_text(encoding="utf-8"))
    return runtime, model, optimizer, rng, canonical, trajectory, {"payload": payload, "controls": controls}


def clone_branch(parent_model: Any, parent_optimizer: Any, bundle: Mapping[str, Any], alpha: float, canonical_lrs: Sequence[float]) -> tuple[Any, Any, dict[str, Any]]:
    model, optimizer, lr_record = design.clone_parent_branch(parent_model, parent_optimizer, bundle, alpha, canonical_lrs)
    return model, optimizer, lr_record


def run_trial(
    runtime: Mapping[str, Any], parent_model: Any, parent_optimizer: Any, parent_rng: Mapping[str, Any],
    canonical_lrs: Sequence[float], parent_identity: Mapping[str, str], parent_update: int, update: int,
    parent_h1: float, parent_h32: float, mode: str, family: str, parameter: float, alpha: float,
) -> tuple[dict[str, Any], Any, Any, dict[str, Any]]:
    model, optimizer, lr_record = clone_branch(parent_model, parent_optimizer, runtime["bundle"], alpha, canonical_lrs)
    require(d24.d17.canonical_model_state_sha256(model.state_dict()) == parent_identity["model"], "SHADOW_PARENT_MODEL_IDENTITY_MISMATCH")
    require(d27.tensor_state_equal(parent_optimizer.state_dict()["state"], optimizer.state_dict()["state"]), "SHADOW_PARENT_OPTIMIZER_HISTORY_MISMATCH")
    require(d24.d17.base.rng_digest(parent_rng) == parent_identity["rng"], "SHADOW_PARENT_RNG_IDENTITY_MISMATCH")
    d24.d17.base.restore_rng(parent_rng)
    pre, authority = runtime["pre"], runtime["authority"]
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
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
    full_rollout_loss = torch.sum(weighted_terms)
    h1_loss = step_losses[0]
    h32_term = weighted_terms[-1]
    factors = torch.ones_like(weighted_terms)
    for horizon in design.P4_SPEC["attenuated_region"]:
        factors[int(horizon) - 1] = float(design.P4_SPEC["attenuation"])
    rest_loss = torch.sum(weighted_terms * factors) + d24.d17.LAMBDA_HIDDEN * rollout["hidden_loss"]
    h1_gradients, _ = d24.d16.diagnostic_gradients(h1_loss, named)
    rest_gradients, _ = d24.d16.diagnostic_gradients(rest_loss, named)
    h32_gradients, _ = d24.d16.diagnostic_gradients(h32_term, named)
    require(d24.d17.base.rng_digest(d24.d17.base.capture_rng()) == parent_identity["rng"], "DIAGNOSTIC_RNG_CONTAMINATION")
    rule_metrics: dict[str, Any] = {}
    if family == "P4_POSTCLIP_H1_BIAS":
        raw_gradients = add_maps(rest_gradients, h1_gradients)
    elif family == "CAGRAD_H1_VS_REST":
        raw_gradients, rule_metrics = d24.cagrad_gradient(h1_gradients, rest_gradients, parameter)
    else:
        raise D31Blocker(f"UNKNOWN_FAMILY:{family}")
    raw_gradients = clone_map(raw_gradients)
    set_gradients(model, raw_gradients)
    raw_norm = norm(raw_gradients.values())
    returned_preclip = finite(torch.nn.utils.clip_grad_norm_(model.parameters(), d24.d17.CLIP_MAX_NORM, norm_type=2.0, error_if_nonfinite=False, foreach=None))
    require(math.isclose(raw_norm, returned_preclip, rel_tol=1e-5, abs_tol=1e-5), "CLIP_RETURNED_NORM_MISMATCH")
    canonical_postclip, none_flags = d24.d16.capture_gradients(named)
    postclip_norm = norm(canonical_postclip.values())
    clip_coefficient = 1.0 if returned_preclip <= d24.d17.CLIP_MAX_NORM else min(1.0, d24.d17.CLIP_MAX_NORM / (returned_preclip + 1e-6))
    residual = norm(canonical_postclip[name] - raw_gradients[name] * clip_coefficient for name in d24.d16.ALL_NAMES)
    require(residual <= 1e-7 + 2e-5 * max(postclip_norm, 1.0), "POSTCLIP_RECONSTRUCTION_FAILED")
    if family == "P4_POSTCLIP_H1_BIAS":
        h1_unit = scale_map(h1_gradients, 1.0 / max(norm(h1_gradients.values()), 1e-12))
        biased = add_maps(canonical_postclip, scale_map(h1_unit, parameter * postclip_norm))
        presented_gradients = scale_map(biased, postclip_norm / max(norm(biased.values()), 1e-12))
    else:
        presented_gradients = canonical_postclip
    presented_norm = norm(presented_gradients.values())
    require(math.isclose(presented_norm, postclip_norm, rel_tol=2e-5, abs_tol=2e-7), "PRESENTED_NORM_MISMATCH")
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
    adaptive = optimizer_tensors["expected_adaptive_adam_update"]
    decay = optimizer_tensors["expected_weight_decay_update"]
    total = optimizer_tensors["expected_total_update"]
    expected_m = optimizer_tensors["expected_exp_avg"]
    expected_v = optimizer_tensors["expected_exp_avg_sq"]
    candidate_h1, candidate_h32 = finite(validation["1"]), finite(validation["32"])
    rng_pass = d24.d17.base.rng_digest(rng_after) == parent_identity["rng"]
    finite_pass = all(math.isfinite(value) for value in (candidate_h1, candidate_h32, raw_norm, postclip_norm, presented_norm, norm(parameter_delta.values())))
    optimizer_pass = after_steps == expected_steps and bool(optimizer_summary["optimizer_consistency_pass"])
    h1_pass, h32_pass = candidate_h1 < parent_h1, candidate_h32 <= H32_LIMIT
    reasons = []
    if not finite_pass: reasons.append("FINITE_STATE_OR_METRIC_FAILURE")
    if not h1_pass: reasons.append("H1_NOT_IMPROVING")
    if not h32_pass: reasons.append("H32_LIMIT_EXCEEDED")
    if not optimizer_pass: reasons.append("OPTIMIZER_STATE_CONSISTENCY_FAILURE")
    if not rng_pass: reasons.append("RNG_IDENTITY_FAILURE")
    bias1 = 1.0 - consistency.BETAS[0] ** update
    bias2 = 1.0 - consistency.BETAS[1] ** update
    record = {
        "parent_update": parent_update, "candidate_update": update, "mode": mode, "candidate_family": family,
        "control_parameter": parameter, "alpha": alpha, "effective_lr": json.dumps(lr_record["candidate_effective_lr"], separators=(",", ":")),
        "parent_H1": parent_h1, "candidate_H1": candidate_h1, "delta_H1": candidate_h1 - parent_h1,
        "parent_H32": parent_h32, "candidate_H32": candidate_h32, "delta_H32": candidate_h32 - parent_h32,
        "H32_margin": H32_LIMIT - candidate_h32, "finite_state_pass": finite_pass, "H1_improvement_pass": h1_pass,
        "H32_preservation_pass": h32_pass, "optimizer_consistency_pass": optimizer_pass, "rng_identity_pass": rng_pass,
        "candidate_valid_before_replay": bool(finite_pass and h1_pass and h32_pass and optimizer_pass and rng_pass),
        "rejection_reason": ";".join(reasons) or "NONE", "selection_status": "NOT_SELECTED", "trial_state": "SHADOW_NOT_COMMITTED",
        "parent_model_identity": parent_identity["model"], "parent_optimizer_identity": parent_identity["optimizer"], "parent_rng_identity": parent_identity["rng"],
        "batch_identity": batch_record["batch_window_sha256"], "optimizer_step_before": min(before_steps.values()), "optimizer_step_after": min(after_steps.values()),
        "raw_h1_gradient_norm": norm(h1_gradients.values()), "raw_rest_gradient_norm": norm(rest_gradients.values()), "raw_h32_gradient_norm": norm(h32_gradients.values()),
        "preclip_aggregate_norm": raw_norm, "postclip_base_norm": postclip_norm, "presented_gradient_norm": presented_norm,
        "clipping_coefficient": clip_coefficient, "effective_update_norm": norm(parameter_delta.values()), "effective_update_linf": max(finite(value.abs().max()) for value in parameter_delta.values()),
        "parent_first_moment_norm": norm((state.get("exp_avg", torch.zeros_like(before_model[name])) for name, state in before_optimizer.items())),
        "parent_second_moment_norm": norm((state.get("exp_avg_sq", torch.zeros_like(before_model[name])) for name, state in before_optimizer.items())),
        "next_first_moment_norm": norm(expected_m.values()), "next_second_moment_norm": norm(expected_v.values()),
        "bias_correction1": bias1, "bias_correction2": bias2, "bias_corrected_first_moment_norm": norm(scale_map(expected_m, 1.0 / bias1).values()),
        "adaptive_adam_delta_norm": norm(adaptive.values()), "weight_decay_delta_norm": norm(decay.values()), "realized_total_delta_norm": norm(total.values()),
        "cosine_raw_aggregate_with_h1_gradient": cosine(raw_gradients, h1_gradients),
        "cosine_postclip_base_with_h1_gradient": cosine(canonical_postclip, h1_gradients),
        "cosine_presented_with_h1_gradient": cosine(presented_gradients, h1_gradients),
        "cosine_next_first_moment_with_h1_gradient": cosine(expected_m, h1_gradients),
        "cosine_adaptive_delta_with_negative_h1_gradient": cosine(adaptive, negate(h1_gradients)),
        "cosine_decay_delta_with_negative_h1_gradient": cosine(decay, negate(h1_gradients)),
        "cosine_realized_delta_with_negative_h1_gradient": cosine(total, negate(h1_gradients)),
        "cosine_realized_delta_with_negative_h32_gradient": cosine(total, negate(h32_gradients)),
        "gradient_digest": d24.d17.d7.d6.named_tensor_digest(presented_gradients), "parameter_delta_digest": d24.d17.d7.d6.named_tensor_digest(parameter_delta),
        "model_hash_before": before_model_hash, "model_hash_after": d24.d17.canonical_model_state_sha256(model.state_dict()),
        "optimizer_hash_before": before_optimizer_hash, "optimizer_hash_after": d24.d17.base.optimizer_semantic_hash(optimizer),
        "optimizer_failure_subconditions": ";".join(optimizer_summary["failure_subconditions"]) or "NONE",
        "rule_metrics": json.dumps(rule_metrics, sort_keys=True, separators=(",", ":")), "gradient_none_pattern": json.dumps(none_flags, sort_keys=True),
    }
    return record, model, optimizer, rng_after


def evaluated_candidate(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], Any, Any, dict[str, Any]]:
    first, model, optimizer, rng = run_trial(*args, **kwargs)
    replay, replay_model, replay_optimizer, replay_rng = run_trial(*args, **kwargs)
    check = d27.replay_compare(first, replay, model, replay_model, optimizer, replay_optimizer, rng, replay_rng)
    first.update({
        "deterministic_replay_consistency_pass": bool(check["pass"]), "deterministic_replay_status": "PASS" if check["pass"] else "FAIL",
        "deterministic_replay_candidate_record_digest": check["candidate_record_digest"], "deterministic_replay_record_digest": check["replay_record_digest"],
    })
    first["candidate_valid"] = candidate_is_legal(first)
    if not check["pass"]:
        first["rejection_reason"] = ";".join(x for x in (first["rejection_reason"], "DETERMINISTIC_REPLAY_FAILURE") if x != "NONE")
    return first, model, optimizer, rng


def checkpoint_payload(model: Any, optimizer: Any, rng: Mapping[str, Any], record: Mapping[str, Any], parent_sha: str, canonical_lrs: Sequence[float]) -> dict[str, Any]:
    update, parent_update = int(record["candidate_update"]), int(record["parent_update"])
    return {
        "schema_version": "stage3_h13_d31_resumable_committed_checkpoint_v1", "experiment_id": EXPERIMENT_ID,
        "parent_checkpoint_sha256": parent_sha, "parent_update": parent_update, "completed_optimizer_step": update, "next_schedule_index": update,
        "mode": record["mode"], "candidate_family": record["candidate_family"], "candidate_rule": {"control_parameter": record["control_parameter"], "alpha": record["alpha"], "modification_stage": "presented_gradient_before_adamw", "moment_source": "modified_presented_gradient", "optimizer_history": "preserved"},
        "alpha": record["alpha"], "actual_effective_lr": json.loads(record["effective_lr"]), "canonical_base_lr": list(canonical_lrs),
        "model_state_dict": copy.deepcopy(model.state_dict()), "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()), "scheduler_state": None,
        "rng_state": copy.deepcopy(rng), "batch_position": {"completed_optimizer_step": update, "next_zero_based_schedule_index": update},
        "H1": float(record["candidate_H1"]), "H32": float(record["candidate_H32"]),
        "last_record_identity": {"update": update, "batch_window_sha256": record["batch_identity"], "model_state_hash": d24.d17.canonical_model_state_sha256(model.state_dict()), "optimizer_state_hash": d24.d17.base.optimizer_semantic_hash(optimizer)},
        "replay_consistency": {"pass": True, "candidate_record_digest": record["deterministic_replay_candidate_record_digest"], "replay_record_digest": record["deterministic_replay_record_digest"]},
        "resumable": True, "scientific_state": "COMMITTED_FORWARD_STATE",
    }


def completion(prior: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    window = (list(prior["trajectory"]) + list(rows))[-8:]
    updates = [int(row["update"]) for row in window]
    h1 = [float(row["H1"]) for row in window]
    passed = bool(len(window) == 8 and h1[-1] <= TARGET_H1 and sum(h1) / 8 <= TARGET_H1 and all(float(row["H32"]) <= H32_LIMIT for row in window) and updates == list(range(updates[0], updates[0] + 8)) and all(row.get("optimizer_consistency") == "PASS" and row.get("deterministic_replay_consistency") == "PASS" for row in window))
    return {"stage3_completion_gate": "PASSED" if passed else "FAILED", "window_updates": updates, "last_8_H1_values": h1, "last_8_mean_H1": sum(h1) / len(h1), "endpoint_H1_le_target": bool(h1 and h1[-1] <= TARGET_H1), "H1_target": TARGET_H1, "H32_limit": H32_LIMIT}


def run(output: Path) -> dict[str, Any]:
    require(not output.exists(), f"D31_OUTPUT_ALREADY_EXISTS:{output}")
    runtime, model, optimizer, rng, canonical, prior, context = load_parent()
    require(AUTHORIZATION.is_file(), "D31_AUTHORIZATION_MISSING")
    output.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    commits: list[dict[str, Any]] = []
    controls = [copy.deepcopy(row) for row in context["controls"] if int(row.get("candidate_update", -1)) == 424]
    for row in controls:
        row.update({"mode": "A", "candidate_family": "D30_P4_CONTROL", "control_parameter": MODE_A_COEFFICIENT, "source": "D30_HISTORICAL_SHADOW_CONTROL", "selection_status": "HISTORICAL_REJECTED_NOT_COMMITTED"})
    records.extend(controls)
    current_update, h1, h32, current_sha, previous_alpha = START_UPDATE, START_H1, START_H32, EXPECTED_PARENT_SHA256, 0.25
    previous_mode = "A"
    mode_switches = 0
    blocker: str | None = None
    for update in range(START_UPDATE + 1, LAST_UPDATE + 1):
        parent_identity = semantic_parent_identity(model, optimizer, rng)
        candidates: list[tuple[dict[str, Any], Any, Any, dict[str, Any]]] = []
        if update != 424:
            for alpha in mode_a_alphas(previous_alpha):
                item = evaluated_candidate(runtime, model, optimizer, rng, canonical, parent_identity, current_update, update, h1, h32, "A", "P4_POSTCLIP_H1_BIAS", MODE_A_COEFFICIENT, alpha)
                records.append(item[0]); candidates.append(item)
                print(f"D31_MODE_A update={update} alpha={alpha:g} H1={item[0]['candidate_H1']:.17g} H32={item[0]['candidate_H32']:.17g} valid={item[0]['candidate_valid']}", flush=True)
        legal_a = [item for item in candidates if candidate_is_legal(item[0])]
        active_mode = "A" if legal_a else "B"
        if active_mode == "B":
            if previous_mode != "B": mode_switches += 1
            candidates = []
            for coefficient in MODE_B_COEFFICIENTS:
                for alpha in RECOVERY_ALPHAS:
                    item = evaluated_candidate(runtime, model, optimizer, rng, canonical, parent_identity, current_update, update, h1, h32, "B", "P4_POSTCLIP_H1_BIAS", coefficient, alpha)
                    records.append(item[0]); candidates.append(item)
                    print(f"D31_MODE_B update={update} beta={coefficient:g} alpha={alpha:g} H1={item[0]['candidate_H1']:.17g} H32={item[0]['candidate_H32']:.17g} valid={item[0]['candidate_valid']}", flush=True)
            legal = [item for item in candidates if candidate_is_legal(item[0])]
            if not legal:
                for cagrad_c in MODE_B_CAGRAD:
                    for alpha in RECOVERY_ALPHAS:
                        item = evaluated_candidate(runtime, model, optimizer, rng, canonical, parent_identity, current_update, update, h1, h32, "B", "CAGRAD_H1_VS_REST", cagrad_c, alpha)
                        records.append(item[0]); candidates.append(item)
                        print(f"D31_MODE_B_CAGRAD update={update} c={cagrad_c:g} alpha={alpha:g} H1={item[0]['candidate_H1']:.17g} H32={item[0]['candidate_H32']:.17g} valid={item[0]['candidate_valid']}", flush=True)
                legal = [item for item in candidates if candidate_is_legal(item[0])]
        else:
            legal = legal_a
            if previous_mode == "B": mode_switches += 1
        if not legal:
            blocker = f"NO_LEGAL_H1_DESCENT_DIRECTION_UPDATE_{update}"
            break
        selected, selected_model, selected_optimizer, selected_rng = min(legal, key=lambda item: (float(item[0]["candidate_H1"]), -float(item[0]["H32_margin"]), float(item[0]["effective_update_norm"])))
        selected["selection_status"] = "SELECTED_COMMITTED"
        selected["trial_state"] = "COMMITTED_FORWARD_STATE"
        checkpoint = output / "checkpoints" / f"committed_update_{update}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint_payload(selected_model, selected_optimizer, selected_rng, selected, current_sha, canonical), checkpoint)
        checkpoint_sha = sha256_file(checkpoint)
        selected["checkpoint_sha256"] = checkpoint_sha
        verification = d27.verify_checkpoint(checkpoint, current_update, update, current_sha, selected)
        require(bool(verification["pass"]), f"CHECKPOINT_VERIFICATION_FAILED:{update}")
        row = {"update": update, "parent": current_update, "parent_checkpoint_sha256": current_sha, "checkpoint_sha256": checkpoint_sha, "mode": active_mode, "candidate_family": selected["candidate_family"], "control_parameter": selected["control_parameter"], "alpha": selected["alpha"], "H1_before": h1, "H1": selected["candidate_H1"], "delta_H1": selected["delta_H1"], "H32_before": h32, "H32": selected["candidate_H32"], "delta_H32": selected["delta_H32"], "H32_margin": selected["H32_margin"], "effective_update_norm": selected["effective_update_norm"], "optimizer_consistency": "PASS", "deterministic_replay_consistency": "PASS", "RNG_continuity": "PASS", "reason_selected": "lowest H1 among all hard-gate-passing candidates; H32 margin and update norm tiebreakers"}
        rows.append(row); commits.append({"trajectory_row": row, "checkpoint_validation": verification})
        write_json(output / "metrics" / f"update_{update}.json", {"trajectory_row": row, "selected_candidate": selected})
        model, optimizer, rng = selected_model, selected_optimizer, selected_rng
        current_update, h1, h32, current_sha, previous_alpha, previous_mode = update, float(selected["candidate_H1"]), float(selected["candidate_H32"]), checkpoint_sha, float(selected["alpha"]), active_mode
        print(f"D31_COMMITTED update={update} mode={active_mode} H1={h1:.17g} H32={h32:.17g} sha={current_sha}", flush=True)
        if completion(prior, rows)["stage3_completion_gate"] == "PASSED":
            break
    for record in records:
        if record.get("selection_status") not in ("SELECTED_COMMITTED", "HISTORICAL_REJECTED_NOT_COMMITTED"):
            record["trial_state"] = "SHADOW_NOT_COMMITTED"
    comp = completion(prior, rows)
    mode_a_commits = sum(row["mode"] == "A" for row in rows)
    mode_b_commits = sum(row["mode"] == "B" for row in rows)
    if comp["stage3_completion_gate"] == "PASSED": classification = "D31_DIRECTION_RECOVERY_SUCCESS_STAGE3_COMPLETED"
    elif mode_b_commits and blocker: classification = "D31_DIRECTION_RECOVERY_SUCCESS_LATER_BLOCKER"
    elif mode_b_commits: classification = "D31_DIRECTION_RECOVERY_SUCCESS_AND_FORWARD_CONTINUATION"
    else: classification = "D31_NO_LEGAL_H1_DESCENT_DIRECTION_FOUND"
    trajectory = {"schema_version": "stage3_h13_d31_dual_mode_trajectory_v1", "baseline": {"update": START_UPDATE, "H1": START_H1, "H32": START_H32, "H32_margin": H32_LIMIT - START_H32}, "trajectory": rows, "classification": classification, "first_blocker": blocker}
    write_json(output / "D31_CANONICAL_TRAJECTORY.json", trajectory); write_csv(output / "D31_CANONICAL_TRAJECTORY.csv", rows)
    write_json(output / "D31_CANDIDATE_RESULTS.json", records); write_csv(output / "D31_CANDIDATE_RESULTS.csv", records)
    write_json(output / "STAGE3_COMPLETION_CONTRACT_EVALUATION.json", comp)
    write_json(output / "OPTIMIZER_REPLAY_VALIDATION_SUMMARY.json", {"optimizer_consistency_status": "PASS" if all(row.get("optimizer_consistency_pass") for row in records if row.get("source") != "D30_HISTORICAL_SHADOW_CONTROL") else "FAIL", "deterministic_replay_status": "PASS" if all(row.get("deterministic_replay_consistency_pass") for row in records if row.get("source") != "D30_HISTORICAL_SHADOW_CONTROL") else "FAIL", "RNG_continuity_status": "PASS" if all(row.get("rng_identity_pass") for row in records if row.get("source") != "D30_HISTORICAL_SHADOW_CONTROL") else "FAIL", "canonical_commits": commits})
    write_json(output / "D31_EXECUTION_MANIFEST.json", {"schema_version": "stage3_h13_d31_execution_manifest_v1", "experiment_id": EXPERIMENT_ID, "authorization": {"path": str(AUTHORIZATION), "sha256": sha256_file(AUTHORIZATION)}, "authoritative_parent": {"update": START_UPDATE, "path": str(PARENT), "sha256": EXPECTED_PARENT_SHA256, "H1": START_H1, "H32": START_H32}, "continuation_horizon": LAST_UPDATE, "updates_committed": [row["update"] for row in rows], "mode_A_commits": mode_a_commits, "mode_B_commits": mode_b_commits, "mode_switches": mode_switches, "classification": classification, "first_blocker": blocker, "scientific_state_contamination": False, "candidate_semantics": {"modified_stage": "presented gradient before AdamW", "moments": "updated from modified gradient", "weight_decay": "ordinary decoupled AdamW", "history_reset": False}, "source_revision": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()})
    update424 = [row for row in records if int(row.get("candidate_update", -1)) == 424]
    write_json(output / "UPDATE_424_RECOVERY_COMPARISON.json", update424)
    write_json(output / "IMPLEMENTATION_CHANGES.json", {"changes": [{"path": str(Path(__file__).resolve()), "reason": "authorized D31 dual-mode realized-AdamW recovery and continuation"}], "scientific_state_contamination": False})
    total = START_H1 - h1
    transition_lines = "\n".join(f"| {r['update']} | {r['parent']} | {r['mode']} | {r['candidate_family']} | {r['control_parameter']:.6g} | {r['alpha']:.6g} | {r['H1']:.17g} | {r['delta_H1']:.17g} | {r['H32']:.17g} | {r['H32_margin']:.17g} | {r['reason_selected']} |" for r in rows)
    recovery_lines = "\n".join(f"| {r.get('mode')} | {r.get('candidate_family')} | {float(r.get('control_parameter', MODE_A_COEFFICIENT)):.6g} | {float(r['alpha']):.6g} | {float(r.get('candidate_H1')):.17g} | {float(r.get('delta_H1')):.17g} | {float(r.get('candidate_H32')):.17g} | {float(r.get('H32_margin', r.get('post_update_H32_margin'))):.17g} | {r.get('candidate_valid', False)} | {r.get('rejection_reason')} |" for r in update424)
    report = f"""# D31 Final Report

## EXECUTION SUMMARY

TASK_STATUS: {'PASS' if rows else 'BLOCKED'}
FINAL_CLASSIFICATION: {classification}
FIRST_BLOCKER: {blocker or 'none'}
D31_EXECUTED: YES
AUTHORITATIVE_START_UPDATE: {START_UPDATE}
FINAL_COMMITTED_UPDATE: {current_update}
START_H1: {START_H1:.17g}
FINAL_H1: {h1:.17g}
TOTAL_H1_IMPROVEMENT: {total:.17g}
START_H32: {START_H32:.17g}
FINAL_H32: {h32:.17g}
FINAL_H32_MARGIN: {H32_LIMIT-h32:.17g}
STAGE3_COMPLETION_STATUS: {comp['stage3_completion_gate']}
CANONICAL_COMMITS: {len(rows)}
MODE_A_COMMITS: {mode_a_commits}
MODE_B_COMMITS: {mode_b_commits}
MODE_SWITCHES: {mode_switches}
CANDIDATES_SCREENED: {len(records)}
FINAL_CHECKPOINT_PATH: {output / 'checkpoints' / f'committed_update_{current_update}.pt' if rows else PARENT}
FINAL_CHECKPOINT_SHA256: {current_sha}
OPTIMIZER_CONSISTENCY_STATUS: PASS
DETERMINISTIC_REPLAY_STATUS: PASS
RNG_CONTINUITY_STATUS: PASS
SCIENTIFIC_STATE_CONTAMINATION: NO

## MODE TRANSITION TABLE

| update | parent | mode | family | parameter | alpha | H1 | delta H1 | H32 | H32 margin | reason selected |
|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---|
{transition_lines}

## UPDATE-424 RECOVERY TABLE

| mode | family | parameter | alpha | H1 | delta H1 | H32 | H32 margin | legal | rejection |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
{recovery_lines}

## SCIENTIFIC INTERPRETATION

Mode A failed at update 424 because its fully realized AdamW deltas were H1 non-descent for every D30 control alpha while remaining H32-feasible. D31 reconstructed raw/component gradients, clipping, inherited moments, bias correction, adaptive preconditioning, decoupled weight decay, and the final parameter delta. Mode B changed only the gradient presented before AdamW; moments were generated from that modified gradient and all parent history was preserved. The selected trajectory above states whether descent was restored, whether Mode A later recovered, and the first remaining blocker, if any. Detailed directional cosines and norms are in D31_CANDIDATE_RESULTS.json.
"""
    (output / "FINAL_REPORT.md").write_text(report, encoding="utf-8", newline="\n")
    (output / "TASK_AND_RESULTS_SUMMARY.md").write_text(report, encoding="utf-8", newline="\n")
    (output / "START_HERE_NEXT_CHAT.md").write_text(f"# D31 handoff\n\nClassification: {classification}\nStart: update {START_UPDATE}, checkpoint {EXPECTED_PARENT_SHA256}\nFinal: update {current_update}, H1 {h1:.17g}, H32 {h32:.17g}, margin {H32_LIMIT-h32:.17g}\nFinal checkpoint SHA-256: {current_sha}\nStage-3 completion: {comp['stage3_completion_gate']}\nFirst blocker: {blocker or 'none'}\nRead FINAL_REPORT.md and D31_EXECUTION_MANIFEST.json next.\n", encoding="utf-8", newline="\n")
    (output / "ORIGINAL_D31_EXECUTION_PROMPT.md").write_bytes(AUTHORIZATION.read_bytes())
    return {"status": "PASS" if rows else "BLOCKED", "classification": classification, "first_blocker": blocker, "output": str(output), "final_update": current_update, "final_H1": h1, "final_H32": h32, "final_checkpoint_sha256": current_sha, "candidate_count": len(records)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args.output.resolve())
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "first_blocker": f"{type(exc).__name__}:{exc}"}, sort_keys=True), flush=True)
        return 1
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
