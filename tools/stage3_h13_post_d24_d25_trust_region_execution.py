"""Separately-authorized scientific D25 update-395 execution.

This module is intentionally separate from the design-review command. It
cannot run without an authorization file that names the exact frozen D25
design-manifest hash. Every alpha trial is a fresh child of the update-394
parent; the winning model and optimizer are saved together only after ranking.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from tools import stage3_h13_post_d24_d25_trust_region_design_review as design
from tools import stage3_h13_post_d23_d24_adamw_aware_postclip_multi_proposal_forward_optimization_and_completion_execution as d24
from tools import stage3_h13_d26_optimizer_consistency as d26_consistency


ROOT = design.ROOT
EXPERIMENT_ID = design.EXECUTION_EXPERIMENT_ID
TARGET_UPDATE = design.TARGET_UPDATE
P4_SPEC = design.P4_SPEC
ALPHA_LADDER = design.ALPHA_LADDER
H32_LIMIT = design.H32_LIMIT
PARENT_H1 = design.START_H1
PARENT_H32 = design.START_H32
CANDIDATE_FIELDS = design.CANDIDATE_FIELDS


def finite(value: Any) -> float:
    return design.finite(value)


def require(condition: bool, message: str) -> None:
    design.require(condition, message)


def write_json(path: Path, value: Any) -> None:
    design.write_json(path, value)


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fieldnames), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def add_maps(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: left[name] + right[name] for name in d24.d16.ALL_NAMES}


def scale_map(values: Mapping[str, torch.Tensor], scale: float) -> dict[str, torch.Tensor]:
    return {name: values[name] * float(scale) for name in d24.d16.ALL_NAMES}


def clone_map(values: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().contiguous().clone() for name, value in values.items()}


def dot(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> float:
    return finite(sum(torch.sum(left[name].double() * right[name].double()) for name in d24.d16.ALL_NAMES))


def norm(values: Sequence[torch.Tensor]) -> float:
    total = 0.0
    for value in values:
        total += finite(torch.sum(value.detach().double().square()))
    return finite(math.sqrt(max(total, 0.0)))


def set_gradients(model: Any, gradients: Mapping[str, torch.Tensor]) -> None:
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    require([name for name, _ in named] == list(d24.d16.ALL_NAMES), "PARAMETER_ORDER_MISMATCH")
    for name, parameter in named:
        parameter.grad = gradients[name].to(device=parameter.device, dtype=parameter.dtype).clone()


def run_p4_trial(model: Any, optimizer: torch.optim.Optimizer, runtime: Mapping[str, Any], alpha: float, lr_record: Mapping[str, Any], parent_h1: float, parent_h32: float, rng_before: Mapping[str, Any], diagnostic_path: Path | None = None, target_update: int = TARGET_UPDATE, parent_update: int | None = None) -> dict[str, Any]:
    """Run one exact D24 P4 forward/gradient/clip/AdamW/validation pipeline."""
    zero_index = target_update - 1
    if parent_update is None:
        parent_update = target_update - 1
    d24.d17.base.restore_rng(copy.deepcopy(rng_before))
    pre, authority = runtime["pre"], runtime["authority"]
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    require([name for name, _ in named] == list(d24.d16.ALL_NAMES), "PARAMETER_ORDER_MISMATCH")
    batch_record = d24.d16.batch_identity(authority["schedule_manifest"], zero_index)
    batch = d24.d17.base.tensor_batch(pre["train_data"], authority["schedule_rows"][zero_index])
    before_model = d24.d16.clone_parameters(model)
    before_optimizer = d24.d16.clone_named_optimizer_state(optimizer)
    before_optimizer_hash = d24.d17.base.optimizer_semantic_hash(optimizer)
    before_steps = d24.d16.optimizer_steps(before_optimizer)
    optimizer.zero_grad(set_to_none=True)
    rollout = d24.d17.base.causal_paired_rollout(model, batch["inputs"], batch["history_positions"], batch["history_times"], batch["target_times"], batch["starts"], batch["ends"], pre["stats"]["channels"], horizon=d24.ACTIVE_HORIZON, target_positions_for_teacher=batch["target_positions"], hidden_consistency_enabled=True)
    require(rollout.get("free_branch_reads_target_positions") is False and rollout.get("teacher_reference_stop_gradient") is True, "GRAPH_SEMANTICS_GATE_FAILED")
    predictions = rollout["predictions"]
    targets = batch["target_positions"][:, :d24.ACTIVE_HORIZON, :]
    step_losses = F.smooth_l1_loss(predictions, targets, beta=d24.d17.POSITION_BETA, reduction="none").mean(dim=(0, 2))
    weights = d24.d17.rollout_horizon_weights(d24.ACTIVE_HORIZON).to(device=predictions.device, dtype=predictions.dtype)
    weighted_terms = step_losses * weights
    full_rollout_loss = torch.sum(weighted_terms)
    h1_loss = step_losses[0]
    h32_term = weighted_terms[-1]
    hidden_loss = rollout["hidden_loss"]
    factors = torch.ones_like(weighted_terms)
    for horizon in P4_SPEC["attenuated_region"]:
        factors[int(horizon) - 1] = float(P4_SPEC["attenuation"])
    rollout_loss = torch.sum(weighted_terms * factors)
    rest_loss = rollout_loss + d24.d17.LAMBDA_HIDDEN * hidden_loss
    canonical_total_objective = full_rollout_loss + d24.d17.LAMBDA_HIDDEN * hidden_loss + d24.d17.LAMBDA_H1 * h1_loss
    require(all(bool(torch.isfinite(value)) for value in (full_rollout_loss, rollout_loss, h1_loss, hidden_loss, rest_loss, canonical_total_objective)), "NONFINITE_OBJECTIVE")
    h1_gradients, _ = d24.d16.diagnostic_gradients(h1_loss, named)
    rest_gradients, _ = d24.d16.diagnostic_gradients(rest_loss, named)
    h32_gradients, _ = d24.d16.diagnostic_gradients(h32_term, named)
    require(d24.d17.base.rng_digest(d24.d17.base.capture_rng()) == d24.d17.base.rng_digest(rng_before), "DIAGNOSTIC_RNG_CONTAMINATION")
    raw_gradients = add_maps(rest_gradients, h1_gradients)
    set_gradients(model, raw_gradients)
    require(d24.d17.base.gradients_are_finite(model), "NONFINITE_RAW_GRADIENT")
    raw_norm = norm(list(raw_gradients.values()))
    returned_preclip = finite(torch.nn.utils.clip_grad_norm_(model.parameters(), d24.d17.CLIP_MAX_NORM, norm_type=2.0, error_if_nonfinite=False, foreach=None))
    require(math.isclose(raw_norm, returned_preclip, rel_tol=1e-5, abs_tol=1e-5), "CLIP_RETURNED_NORM_MISMATCH")
    canonical_postclip, none_flags = d24.d16.capture_gradients(named)
    postclip_norm = norm(list(canonical_postclip.values()))
    clip_coefficient = 1.0 if returned_preclip <= d24.d17.CLIP_MAX_NORM else min(1.0, d24.d17.CLIP_MAX_NORM / (returned_preclip + 1e-6))
    residual = norm([canonical_postclip[name] - raw_gradients[name] * clip_coefficient for name in d24.d16.ALL_NAMES])
    require(residual <= 1e-7 + 2e-5 * max(postclip_norm, 1.0), "POSTCLIP_RECONSTRUCTION_FAILED")
    h1_norm = norm(list(h1_gradients.values()))
    d_h1 = scale_map(h1_gradients, 1.0 / max(h1_norm, design.finite(1e-12)))
    trial_raw = add_maps(canonical_postclip, scale_map(d_h1, 0.35 * postclip_norm))
    directional_raw_norm = norm(list(trial_raw.values()))
    presented_gradients = scale_map(trial_raw, postclip_norm / max(directional_raw_norm, design.finite(1e-12)))
    directional_final_norm = norm(list(presented_gradients.values()))
    require(math.isclose(directional_final_norm, postclip_norm, rel_tol=2e-5, abs_tol=2e-7), "POSTCLIP_NORM_PRESERVATION_FAILED")
    set_gradients(model, presented_gradients)
    require(d24.d17.base.gradients_are_finite(model), "NONFINITE_PRESENTED_GRADIENT")
    rng_before_optimizer = d24.d17.base.capture_rng()
    optimizer.step()
    after_model = d24.d16.clone_parameters(model)
    after_optimizer = d24.d16.clone_named_optimizer_state(optimizer)
    state_finite = d24.d16.state_finite(model, optimizer)
    require(state_finite, "NONFINITE_STATE_AFTER_UPDATE")
    rng_after_training = d24.d17.base.capture_rng()
    require(d24.d17.base.rng_digest(rng_before_optimizer) == d24.d17.base.rng_digest(rng_after_training), "OPTIMIZER_RNG_CONSUMPTION_UNMATCHED")
    adamw_metrics, _ = d24.d17.d7.adamw_decomposition(model, optimizer, before_model, presented_gradients, after_model, runtime["table"], effective_lrs=lr_record["candidate_effective_lr"])
    validation, validation_meta = d24.d17.d7.validation_snapshot(model, pre["validation"], pre["stats"]["channels"], True)
    require(validation_meta["rng_unchanged_or_restored"] in ("YES", "RESTORED"), "VALIDATION_RNG_GATE_FAILED")
    rng_after_validation = d24.d17.base.capture_rng()
    after_steps = d24.d16.optimizer_steps(after_optimizer)
    expected_after_steps = {name: before_steps[name] + 1 for name in d24.d16.ALL_NAMES}
    optimizer_step_valid = after_steps == expected_after_steps
    optimizer_consistency, optimizer_consistency_tensors = d26_consistency.validate_adamw_transition(
        model,
        optimizer,
        before_model,
        before_optimizer,
        presented_gradients,
        after_model,
        after_optimizer,
        assumed_effective_lrs=lr_record["candidate_effective_lr"],
    )
    if diagnostic_path is not None:
        diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"summary": optimizer_consistency, "tensors": optimizer_consistency_tensors}, diagnostic_path)
    parameter_delta = {name: after_model[name] - before_model[name] for name in d24.d16.ALL_NAMES}
    adamw_whole = adamw_metrics["aggregate"]["whole_model"]
    parameter_delta_l2 = d24.d16.norm(parameter_delta.values())
    parameter_delta_linf = max(float(value.detach().abs().max()) for value in parameter_delta.values())
    adamw_update_l2 = float(adamw_whole["adaptive_update_norm"])
    candidate_h1 = finite(validation["1"])
    candidate_h32 = finite(validation["32"])
    finite_state_pass = state_finite and all(math.isfinite(value) for value in (raw_norm, returned_preclip, postclip_norm, parameter_delta_l2, parameter_delta_linf, adamw_update_l2, candidate_h1, candidate_h32))
    h32_pass = candidate_h32 <= H32_LIMIT
    h1_pass = candidate_h1 < parent_h1
    optimizer_pass = optimizer_step_valid and adamw_whole["decomposition_pass_fail"] == "PASS" and optimizer_consistency["optimizer_consistency_pass"]
    rng_pass = d24.d17.base.rng_digest(rng_after_validation) == d24.d17.base.rng_digest(rng_before)
    candidate_valid = finite_state_pass and h32_pass and h1_pass and optimizer_pass and rng_pass
    rejection_reasons = []
    if not finite_state_pass:
        rejection_reasons.append("FINITE_STATE_OR_METRIC_FAILURE")
    if not h32_pass:
        rejection_reasons.append("H32_LIMIT_EXCEEDED")
    if not h1_pass:
        rejection_reasons.append("H1_NOT_IMPROVING")
    if not optimizer_pass:
        rejection_reasons.append("OPTIMIZER_STATE_CONSISTENCY_FAILURE")
    if not rng_pass:
        rejection_reasons.append("RNG_IDENTITY_FAILURE")
    return {
        "parent_update": parent_update,
        "candidate_update": target_update,
        "proposal": "P4",
        "alpha": float(alpha),
        "canonical_lr": json.dumps(lr_record["canonical_base_lr"], separators=(",", ":")),
        "effective_lr": json.dumps(lr_record["candidate_effective_lr"], separators=(",", ":")),
        "parent_H1": parent_h1,
        "candidate_H1": candidate_h1,
        "delta_H1": candidate_h1 - parent_h1,
        "parent_H32": parent_h32,
        "candidate_H32": candidate_h32,
        "delta_H32": candidate_h32 - parent_h32,
        "H32_limit": H32_LIMIT,
        "H32_margin_before": H32_LIMIT - parent_h32,
        "H32_margin_after": H32_LIMIT - candidate_h32,
        "finite_state_pass": bool(finite_state_pass),
        "H32_preservation_pass": bool(h32_pass),
        "H1_improvement_pass": bool(h1_pass),
        "candidate_valid": bool(candidate_valid),
        "rejection_reason": "NONE" if not rejection_reasons else ";".join(rejection_reasons),
        "selected": False,
        "model_checkpoint_sha256": d24.d17.canonical_model_state_sha256(model.state_dict()),
        "optimizer_state_sha256_or_equivalent": d24.d17.base.optimizer_semantic_hash(optimizer),
        "batch_identity": batch_record["batch_window_sha256"],
        "source_revision": design.git_head(),
        "parameter_delta_l2": parameter_delta_l2,
        "parameter_delta_linf": parameter_delta_linf,
        "adamw_update_l2": adamw_update_l2,
        "gradient_l2": raw_norm,
        "postclip_gradient_l2": directional_final_norm,
        "raw_gradient_digest": d24.d17.d7.d6.named_tensor_digest(raw_gradients),
        "postclip_gradient_digest": d24.d17.d7.d6.named_tensor_digest(presented_gradients),
        "parameter_delta_digest": d24.d17.d7.d6.named_tensor_digest(parameter_delta),
        "optimizer_step_before": min(before_steps.values()),
        "optimizer_step_after": min(after_steps.values()),
        "optimizer_step_identity_pass": optimizer_step_valid,
        "adamw_decomposition_status": adamw_whole["decomposition_pass_fail"],
        "optimizer_consistency_pass": optimizer_consistency["optimizer_consistency_pass"],
        "optimizer_consistency_failure_subconditions": ";".join(optimizer_consistency["failure_subconditions"]) or "NONE",
        "optimizer_actual_effective_lr": json.dumps(optimizer_consistency["actual_candidate_effective_lr"], separators=(",", ":")),
        "optimizer_reconstruction_assumed_lr": json.dumps(optimizer_consistency["adamw_reconstruction_assumed_lr"], separators=(",", ":")),
        "optimizer_exp_avg_expected_sha256": optimizer_consistency["expected_exp_avg_sha256"],
        "optimizer_exp_avg_realized_sha256": optimizer_consistency["realized_exp_avg_sha256"],
        "optimizer_exp_avg_sq_expected_sha256": optimizer_consistency["expected_exp_avg_sq_sha256"],
        "optimizer_exp_avg_sq_realized_sha256": optimizer_consistency["realized_exp_avg_sq_sha256"],
        "optimizer_parameter_delta_expected_sha256": optimizer_consistency["expected_parameter_delta_sha256"],
        "optimizer_parameter_delta_realized_sha256": optimizer_consistency["realized_parameter_delta_sha256"],
        "optimizer_weight_decay_sha256": optimizer_consistency["weight_decay_contribution_sha256"],
        "optimizer_adaptive_adam_sha256": optimizer_consistency["adaptive_adam_contribution_sha256"],
        "optimizer_total_reconstructed_update_sha256": optimizer_consistency["total_reconstructed_update_sha256"],
        "optimizer_residual_sha256": optimizer_consistency["residual_sha256"],
        "optimizer_step_before_by_parameter": json.dumps(optimizer_consistency["optimizer_step_before"], sort_keys=True, separators=(",", ":")),
        "optimizer_step_after_by_parameter": json.dumps(optimizer_consistency["optimizer_step_after"], sort_keys=True, separators=(",", ":")),
        "rng_identity_pass": rng_pass,
        "gradient_none_pattern": json.dumps(none_flags, sort_keys=True),
        "trial_state": "TRIAL_NOT_COMMITTED",
    }, model, optimizer, copy.deepcopy(rng_after_validation)


def execution_payload(model: Any, optimizer: torch.optim.Optimizer, rng: Mapping[str, Any], selected: Mapping[str, Any], lr_record: Mapping[str, Any], design_manifest_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": "stage3_h13_d25_resumable_committed_checkpoint_v1",
        "experiment_id": EXPERIMENT_ID,
        "design_manifest_sha256": design_manifest_sha256,
        "proposal": "P4",
        "proposal_spec": copy.deepcopy(P4_SPEC),
        "alpha": float(selected["alpha"]),
        "parent_update": design.EXPECTED_START_UPDATE,
        "completed_optimizer_step": TARGET_UPDATE,
        "next_schedule_index": TARGET_UPDATE,
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
        "scheduler_state": None,
        "rng_state": copy.deepcopy(rng),
        "batch_position": {"completed_optimizer_step": TARGET_UPDATE, "next_zero_based_schedule_index": TARGET_UPDATE},
        "canonical_base_lr": copy.deepcopy(lr_record["canonical_base_lr"]),
        "candidate_effective_lr": copy.deepcopy(lr_record["candidate_effective_lr"]),
        "last_record_identity": {"update": TARGET_UPDATE, "batch_window_sha256": selected["batch_identity"], "model_state_hash": d24.d17.canonical_model_state_sha256(model.state_dict()), "optimizer_state_hash": d24.d17.base.optimizer_semantic_hash(optimizer)},
        "resumable": True,
        "scientific_state": "COMMITTED_FORWARD_STATE",
    }


def sha256_manifest(output: Path) -> dict[str, Any]:
    return design.write_sha256_manifest(output)


def run_execution(authorization: Path, design_manifest_path: Path, output: Path) -> dict[str, Any]:
    design_manifest_sha256 = design.sha256_file(design_manifest_path)
    require(design.execution_authorized(authorization, design_manifest_sha256), "D25_EXECUTION_AUTHORIZATION_MISSING_OR_MANIFEST_HASH_MISMATCH")
    manifest = json.loads(design_manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("experiment_id") == design.EXPERIMENT_ID and manifest.get("status") == "FROZEN_FOR_SEPARATE_EXECUTION_AUTHORIZATION", "D25_DESIGN_MANIFEST_NOT_FROZEN")
    require(manifest.get("target_authorization_boundary", {}).get("first_execution_target_update") == TARGET_UPDATE, "D25_TARGET_UPDATE_MISMATCH")
    require(not output.exists(), f"D25_EXECUTION_OUTPUT_ALREADY_EXISTS:{output}")
    output.mkdir(parents=True, exist_ok=False)
    input_identity, context = design.authenticate_inputs()
    runtime = context["runtime"]
    parent_model, parent_optimizer, parent_rng, parent_identity = context["model"], context["optimizer"], context["rng"], context["parent_identity"]
    canonical_lrs = design.canonical_group_lrs(parent_optimizer)
    trial_rows: list[dict[str, Any]] = []
    trial_branches: dict[float, tuple[Any, torch.optim.Optimizer, dict[str, Any]]] = {}
    first_valid_index: int | None = None
    for index, alpha in enumerate(ALPHA_LADDER):
        trial_model, trial_optimizer, lr_record = design.clone_parent_branch(parent_model, parent_optimizer, runtime["bundle"], alpha, canonical_lrs)
        try:
            record, trial_model, trial_optimizer, trial_rng = run_p4_trial(trial_model, trial_optimizer, runtime, alpha, lr_record, PARENT_H1, PARENT_H32, parent_rng)
        except Exception as exc:
            record = {field: None for field in CANDIDATE_FIELDS}
            record.update({"parent_update": design.EXPECTED_START_UPDATE, "candidate_update": TARGET_UPDATE, "proposal": "P4", "alpha": float(alpha), "canonical_lr": json.dumps(canonical_lrs), "effective_lr": json.dumps(design.effective_group_lrs(canonical_lrs, alpha)), "parent_H1": PARENT_H1, "parent_H32": PARENT_H32, "H32_limit": H32_LIMIT, "H32_margin_before": H32_LIMIT - PARENT_H32, "finite_state_pass": False, "H32_preservation_pass": False, "H1_improvement_pass": False, "candidate_valid": False, "rejection_reason": f"{type(exc).__name__}:{exc}", "selected": False, "batch_identity": None, "source_revision": design.git_head(), "trial_state": "TRIAL_FAILED_NOT_COMMITTED"})
            trial_rows.append(record)
            if first_valid_index is not None:
                break
            continue
        trial_rows.append(record)
        if first_valid_index is not None:
            if record["candidate_valid"]:
                trial_branches[float(alpha)] = (trial_model, trial_optimizer, trial_rng)
            else:
                del trial_model, trial_optimizer, trial_rng
            # This is the one predeclared adjacent-smaller candidate.
            break
        if record["candidate_valid"]:
            first_valid_index = index
            trial_branches[float(alpha)] = (trial_model, trial_optimizer, trial_rng)
            if index == len(ALPHA_LADDER) - 1:
                break
            continue
        # A feasible but H1-worsening candidate is not acceptable; continue
        # downward until a valid forward candidate or the ladder floor.
        del trial_model, trial_optimizer, trial_rng
    winner = design.rank_candidates(trial_rows)
    if winner is None:
        h32_feasible = any(bool(row.get("finite_state_pass")) and bool(row.get("H32_preservation_pass")) for row in trial_rows)
        classification = "H32_FEASIBLE_BUT_NO_H1_IMPROVING_ALPHA" if h32_feasible else "NO_H32_FEASIBLE_ALPHA"
        raise design.DesignBlocker(classification)
    winner["selected"] = True
    winner_alpha = float(winner["alpha"])
    selected_model, selected_optimizer, selected_rng = trial_branches[winner_alpha]
    checkpoint_path = output / "D25_UPDATE_395_CHECKPOINT.pt"
    payload = execution_payload(selected_model, selected_optimizer, selected_rng, winner, {"canonical_base_lr": canonical_lrs, "candidate_effective_lr": json.loads(winner["effective_lr"])}, design_manifest_sha256)
    torch.save(payload, checkpoint_path)
    winner["model_checkpoint_sha256"] = design.sha256_file(checkpoint_path)
    winner["trial_state"] = "COMMITTED_FORWARD_STATE"
    trajectory = {"update": TARGET_UPDATE, "selected_proposal": "P4", "alpha": winner_alpha, "H1_before": PARENT_H1, "H1": winner["candidate_H1"], "H32_before": PARENT_H32, "H32": winner["candidate_H32"], "H32_preservation": "YES", "finite_status": "FINITE", "training_stability": "PASS", "model_checkpoint_sha256": winner["model_checkpoint_sha256"], "optimizer_state_sha256_or_equivalent": winner["optimizer_state_sha256_or_equivalent"], "batch_identity": winner["batch_identity"]}
    write_csv(output / "D25_CANDIDATE_RESULTS.csv", list(CANDIDATE_FIELDS), trial_rows)
    write_csv(output / "D25_COMMITTED_FORWARD_TRAJECTORY.csv", list(trajectory), [trajectory])
    execution_manifest = {"schema_version": "stage3_h13_d25_execution_manifest_v1", "experiment_id": EXPERIMENT_ID, "design_manifest": str(design_manifest_path.resolve()), "design_manifest_sha256": design_manifest_sha256, "authorization": {"path": str(authorization.resolve()), "sha256": design.sha256_file(authorization)}, "parent": parent_identity, "target_update": TARGET_UPDATE, "alpha_ladder": list(ALPHA_LADDER), "evaluated_alpha": [row["alpha"] for row in trial_rows], "selection_rule": manifest["candidate_ranking"], "candidate_results_schema": list(CANDIDATE_FIELDS), "source_revision": design.git_head(), "runtime": runtime["environment"], "input_identity": input_identity, "scientific_execution": {"update_395_executed": True, "update_396_plus_executed": False, "training_executed": True, "backward_pass_executed": True, "optimizer_step_count": len(trial_rows)}, "outcome": "D25_UPDATE_395_FEASIBLE_FORWARD_CONTINUATION"}
    write_json(output / "D25_EXECUTION_MANIFEST.json", execution_manifest)
    write_json(output / "D25_EXECUTION_SUMMARY.json", {"status": "PASSED", "classification": "D25_UPDATE_395_FEASIBLE_FORWARD_CONTINUATION", "selected_alpha": winner_alpha, "candidate_count": len(trial_rows), "selected_candidate": winner, "checkpoint": str(checkpoint_path.resolve()), "checkpoint_sha256": winner["model_checkpoint_sha256"]})
    sha256_manifest(output)
    return {"status": "PASSED", "classification": "D25_UPDATE_395_FEASIBLE_FORWARD_CONTINUATION", "output": str(output.resolve()), "selected_alpha": winner_alpha, "checkpoint_sha256": winner["model_checkpoint_sha256"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=EXPERIMENT_ID)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--design-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.execute:
        parser.error("scientific execution requires --execute and a separately authorized file")
    try:
        result = run_execution(args.authorization.resolve(), args.design_manifest.resolve(), args.output.resolve())
    except Exception as exc:
        print(json.dumps({"status": "FAILED", "error": f"{type(exc).__name__}:{exc}"}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
