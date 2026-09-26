"""Execute the authorized Stage-3 H13 D24 forward optimization.

Unlike D23, this runner never ranks a proposal from a pre-step effect or from
materiality.  At every update it starts all fixed proposals from one incumbent
model/AdamW/RNG snapshot, executes the complete realized optimizer step, and
commits only the best valid H32-preserving result.
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import stage3_h13_post_d15_d16_adamw_full_pipeline_h32_component_deletion_causal_intervention_replay as d16  # noqa: E402
from tools import stage3_h13_post_d16_d17_full_pipeline_multi_horizon_rollout_causal_screen_replay as d17  # noqa: E402


EXPERIMENT_ID = "STAGE_3_H13_POST_D23_D24_ADAMW_AWARE_POSTCLIP_MULTI_PROPOSAL_FORWARD_OPTIMIZATION_AND_COMPLETION_EXECUTION"
D23_DIR = ROOT / "outputs" / "stage3_h13_post_d22_d23_progress_first_optimization_sprint_20260821T190000+0800"
START_CHECKPOINT = D23_DIR / "checkpoints" / "C5_update_392.pt"
EXPECTED_START_CHECKPOINT_SHA256 = "226cd512c2f16e415c67b3c0e4045a89841a008f7ec4a09e9249e3c2309efdc1"
EXPECTED_SOURCE_REVISION = "a6dff2b6887cc3f7598ab95549c2dd21c0efe763"
EXPECTED_START_UPDATE = 392
TARGET_H1 = 5.0e-05
MATERIALITY_THRESHOLD = 4.099006815405639e-06
STRONG_THRESHOLD = 2.0 * MATERIALITY_THRESHOLD
H32_THRESHOLD = 0.01856902565856056
CONFIRMATION_WINDOW = 8
MAX_UPDATE = 440
ACTIVE_HORIZON = 32
EPS = 1.0e-12
PROPOSAL_NAMES = ("P0", "P1", "P2", "P3", "P4")

CONTRACT = {
    "version": "1.0",
    "status": "FROZEN",
    "primary_metric": "H1",
    "stage3_completion_target_H1": TARGET_H1,
    "frozen_material_effect_threshold": MATERIALITY_THRESHOLD,
    "confirmation_window": CONFIRMATION_WINDOW,
    "H32_requirement": "existing frozen D22/D23 preservation criterion",
    "stability_requirement": "PASS",
    "real_forward_requirement": ">392",
    "stop_on_completion_gate": True,
    "continue_stage3_training_after_gate": False,
    "materiality_is_forward_progression_gate": False,
}


class StageBlocker(RuntimeError):
    """Fail closed when no scientifically valid forward commit is possible."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fieldnames), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def finite(value: Any) -> float:
    result = float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)
    if not math.isfinite(result):
        raise StageBlocker(f"NONFINITE_MEASUREMENT:{result}")
    return result


def require(condition: bool, message: str) -> None:
    if not condition:
        raise StageBlocker(message)


def dot(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> float:
    return finite(sum(torch.sum(left[name].double() * right[name].double()) for name in d16.ALL_NAMES))


def add_maps(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: left[name] + right[name] for name in d16.ALL_NAMES}


def scale_map(values: Mapping[str, torch.Tensor], scale: float) -> dict[str, torch.Tensor]:
    return {name: values[name] * float(scale) for name in d16.ALL_NAMES}


def clone_map(values: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().contiguous().clone() for name, value in values.items()}


def norm(values: Iterable[torch.Tensor]) -> float:
    total = 0.0
    for value in values:
        total += finite(torch.sum(value.detach().double().square()))
    return finite(math.sqrt(max(total, 0.0)))


def proposal_spec(name: str, *, q0_source: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if name == "P0":
        return {"name": name, "family": "INCUMBENT_C5", "rule": "structure_attenuation", "attenuation": 0.5, "attenuated_region": list(range(18, 32))}
    if name == "P1":
        return {"name": name, "family": "STRONGER_STRUCTURE_ATTENUATION", "rule": "structure_attenuation", "attenuation": 0.25, "attenuated_region": list(range(18, 32))}
    if name == "P2":
        return {"name": name, "family": "POSTCLIP_H1_DIRECTIONAL_BIAS", "rule": "postclip_h1_bias", "postclip_h1_coefficient": 0.35, "attenuation": None}
    if name == "P3":
        return {"name": name, "family": "CAGRAD_H1_VS_REST", "rule": "cagrad", "cagrad_c": 0.4, "attenuation": None}
    if name == "P4":
        return {"name": name, "family": "C5_PLUS_POSTCLIP_H1_BIAS", "rule": "structure_plus_postclip_h1_bias", "attenuation": 0.5, "postclip_h1_coefficient": 0.35, "attenuated_region": list(range(18, 32))}
    if name == "Q0":
        require(q0_source is not None, "Q0_SOURCE_MISSING")
        result = copy.deepcopy(dict(q0_source))
        result["name"] = "Q0"
        result["family"] = f"WAVE_A_WINNER_{q0_source['name']}_{q0_source['family']}"
        return result
    if name == "Q1":
        return {"name": name, "family": "STRONGER_STRUCTURE_ATTENUATION_0P10", "rule": "structure_attenuation", "attenuation": 0.10, "attenuated_region": list(range(18, 32))}
    if name == "Q2":
        return {"name": name, "family": "POSTCLIP_H1_DIRECTIONAL_BIAS_0P70", "rule": "postclip_h1_bias", "postclip_h1_coefficient": 0.70, "attenuation": None}
    if name == "Q3":
        return {"name": name, "family": "CAGRAD_H1_VS_REST_C_0P8", "rule": "cagrad", "cagrad_c": 0.8, "attenuation": None}
    if name == "Q4":
        return {"name": name, "family": "STRUCTURE_0P25_PLUS_POSTCLIP_H1_BIAS", "rule": "structure_plus_postclip_h1_bias", "attenuation": 0.25, "postclip_h1_coefficient": 0.35, "attenuated_region": list(range(18, 32))}
    raise StageBlocker(f"UNKNOWN_PROPOSAL:{name}")


def verify_authorization(authorization: Path) -> dict[str, Any]:
    require(authorization.is_file(), f"AUTHORIZATION_MISSING:{authorization}")
    text = authorization.read_text(encoding="utf-8")
    markers = ("AUTHORIZED: YES", "C5_update_392.pt", "UPDATE 393", "P0", "POSTCLIP_H1_DIRECTIONAL_BIAS", "STOP_ON_COMPLETION_GATE")
    require(all(marker in text for marker in markers), "D24_AUTHORIZATION_SCOPE_MISMATCH")
    return {"path": str(authorization.resolve()), "sha256": sha256_file(authorization)}


def load_checkpoint(path: Path, state: Mapping[str, Any]) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    require(path.is_file(), f"START_CHECKPOINT_MISSING:{path}")
    require(sha256_file(path) == EXPECTED_START_CHECKPOINT_SHA256, "START_CHECKPOINT_SHA256_MISMATCH")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    require(checkpoint.get("schema_version") == "stage3_h13_d23_resumable_checkpoint_v1", "START_CHECKPOINT_SCHEMA_MISMATCH")
    require(checkpoint.get("candidate") == "C5", "START_CHECKPOINT_CANDIDATE_MISMATCH")
    require(int(checkpoint.get("completed_optimizer_step", -1)) == EXPECTED_START_UPDATE, "START_CHECKPOINT_UPDATE_MISMATCH")
    require(int(checkpoint.get("next_schedule_index", -1)) == EXPECTED_START_UPDATE, "START_CHECKPOINT_SCHEDULE_POSITION_MISMATCH")
    require(checkpoint.get("batch_position", {}).get("completed_optimizer_step") == EXPECTED_START_UPDATE, "START_CHECKPOINT_BATCH_STEP_MISMATCH")
    require(checkpoint.get("batch_position", {}).get("next_zero_based_schedule_index") == EXPECTED_START_UPDATE, "START_CHECKPOINT_BATCH_INDEX_MISMATCH")
    require(checkpoint.get("resumable") is True, "START_CHECKPOINT_NOT_RESUMABLE")
    intervention = checkpoint.get("intervention", {})
    require(intervention.get("family") == "D22_STRUCTURE_SOFT_ATTENUATION" and float(intervention.get("attenuation")) == 0.5, "START_CHECKPOINT_INTERVENTION_MISMATCH")
    model, optimizer = d16.new_branch(state["bundle"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    model_hash = d17.canonical_model_state_sha256(model.state_dict())
    optimizer_hash = d17.base.optimizer_semantic_hash(optimizer)
    rng = copy.deepcopy(checkpoint["rng_state"])
    d17.base.restore_rng(rng)
    rng_hash = d17.base.rng_digest(d17.base.capture_rng())
    require(model_hash == checkpoint["last_record_identity"]["model_state_hash"], "START_CHECKPOINT_MODEL_HASH_MISMATCH")
    require(optimizer_hash == checkpoint["last_record_identity"]["optimizer_state_hash"], "START_CHECKPOINT_OPTIMIZER_HASH_MISMATCH")
    require(rng_hash == d17.base.rng_digest(rng), "START_CHECKPOINT_RNG_HASH_MISMATCH")
    steps = d16.optimizer_steps(d16.clone_named_optimizer_state(optimizer))
    require(set(steps.values()) == {EXPECTED_START_UPDATE}, "START_CHECKPOINT_OPTIMIZER_STEP_STATE_MISMATCH")
    batch = d16.batch_identity(state["authority"]["schedule_manifest"], EXPECTED_START_UPDATE - 1)
    require(batch["batch_window_sha256"] == checkpoint["last_record_identity"]["batch_window_sha256"], "START_CHECKPOINT_BATCH_IDENTITY_MISMATCH")
    require(d16.state_finite(model, optimizer), "START_CHECKPOINT_NONFINITE_STATE")
    identity = {"path": str(path.resolve()), "sha256": EXPECTED_START_CHECKPOINT_SHA256, "completed_optimizer_step": EXPECTED_START_UPDATE, "model_state_hash": model_hash, "optimizer_state_hash": optimizer_hash, "rng_state_hash": rng_hash, "batch_identity": batch, "intervention": intervention, "resumable": True}
    return model, optimizer, rng, identity


def preflight(authorization: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    auth = verify_authorization(authorization)
    git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    require(git_head == EXPECTED_SOURCE_REVISION, f"SOURCE_REVISION_MISMATCH:{git_head}")
    runtime = d17.preflight_runtime()
    model, optimizer, rng, start_identity = load_checkpoint(START_CHECKPOINT, runtime)
    validation, validation_meta = d17.d7.validation_snapshot(model, runtime["pre"]["validation"], runtime["pre"]["stats"]["channels"], True)
    require(validation_meta["rng_unchanged_or_restored"] in ("YES", "RESTORED"), "START_VALIDATION_RNG_GATE_FAILED")
    start = {"authorization": auth, "git_head": git_head, "checkpoint": start_identity, "start_H1": validation["1"], "start_H32": validation["32"], "start_validation": validation, "runtime": {"environment": runtime["environment"], "model_hash_at_384": runtime["loaded"]["model_semantic_hash"], "optimizer_hash_at_384": runtime["loaded"]["optimizer_semantic_hash"], "rng_hash_at_384": d17.base.rng_digest(runtime["bundle"]["rng_state"])}}
    runtime["incumbent_model"] = model
    runtime["incumbent_optimizer"] = optimizer
    runtime["incumbent_rng"] = rng
    return start, runtime


def set_gradients(model: Any, gradients: Mapping[str, torch.Tensor]) -> None:
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    require([name for name, _ in named] == list(d16.ALL_NAMES), "PARAMETER_ORDER_MISMATCH")
    for name, parameter in named:
        parameter.grad = gradients[name].to(device=parameter.device, dtype=parameter.dtype).clone()


def flatten_map(values: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([values[name].detach().cpu().reshape(-1).double() for name in d16.ALL_NAMES])


def unflatten_map(flat: torch.Tensor, template: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    offset = 0
    for name in d16.ALL_NAMES:
        shape = template[name].shape
        count = template[name].numel()
        result[name] = flat[offset:offset + count].reshape(shape).to(dtype=template[name].dtype).contiguous()
        offset += count
    require(offset == flat.numel(), "GRADIENT_UNFLATTEN_SIZE_MISMATCH")
    return result


def cagrad_gradient(h1: Mapping[str, torch.Tensor], rest: Mapping[str, torch.Tensor], c: float) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Two-objective CAGrad using the paper's simplex dual formulation."""
    require(0.0 <= c < 1.0, f"CAGRAD_C_OUT_OF_DOMAIN:{c}")
    g1, g2 = flatten_map(h1), flatten_map(rest)
    g0 = 0.5 * (g1 + g2)
    delta = g1 - g2
    g0_norm = float(torch.linalg.vector_norm(g0))

    def at(weight: float) -> tuple[float, torch.Tensor]:
        gw = g2 + float(weight) * delta
        value = float(torch.dot(gw, g0) + float(c) * g0_norm * torch.linalg.vector_norm(gw))
        return value, gw

    # The two-objective simplex is one-dimensional. Golden-section search is
    # deterministic, bounded, and solves the convex dual without a sweep.
    left, right = 0.0, 1.0
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    x1 = right - (right - left) / phi
    x2 = left + (right - left) / phi
    f1, _ = at(x1)
    f2, _ = at(x2)
    for _ in range(80):
        if f1 <= f2:
            right, x2, f2 = x2, x1, f1
            x1 = right - (right - left) / phi
            f1, _ = at(x1)
        else:
            left, x1, f1 = x1, x2, f2
            x2 = left + (right - left) / phi
            f2, _ = at(x2)
    weight = (left + right) / 2.0
    _, gw = at(weight)
    gw_norm = float(torch.linalg.vector_norm(gw))
    direction = g0 if gw_norm <= EPS else g0 + float(c) * g0_norm * gw / gw_norm
    result = unflatten_map(direction, h1)
    return result, {"cagrad_c": float(c), "cagrad_weight_h1": float(weight), "cagrad_weight_rest": float(1.0 - weight), "cagrad_g0_norm": g0_norm, "cagrad_gw_norm": gw_norm, "cagrad_dual_objective": min(f1, f2)}


def clone_trial(model: Any, optimizer: Any, bundle: Mapping[str, Any]) -> tuple[Any, Any]:
    trial_model, trial_optimizer = d16.new_branch(bundle)
    trial_model.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
    trial_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    trial_model.train()
    return trial_model, trial_optimizer


def run_trial(name: str, spec: Mapping[str, Any], model: Any, optimizer: Any, state: Mapping[str, Any], zero_index: int, rng_before: Mapping[str, Any]) -> tuple[dict[str, Any], Any, Any, dict[str, Any]]:
    update = zero_index + 1
    require(update > EXPECTED_START_UPDATE and update <= MAX_UPDATE, f"UNAUTHORIZED_UPDATE:{update}")
    d17.base.restore_rng(rng_before)
    pre, authority = state["pre"], state["authority"]
    named = [(param_name, parameter) for param_name, parameter in model.named_parameters() if parameter.requires_grad]
    require([param_name for param_name, _ in named] == list(d16.ALL_NAMES), f"PARAMETER_ORDER_MISMATCH:{name}:{update}")
    batch_record = d16.batch_identity(authority["schedule_manifest"], zero_index)
    batch = d17.base.tensor_batch(pre["train_data"], authority["schedule_rows"][zero_index])
    before_model = d16.clone_parameters(model)
    before_optimizer = d16.clone_named_optimizer_state(optimizer)
    before_model_hash = d17.canonical_model_state_sha256(model.state_dict())
    before_optimizer_hash = d17.base.optimizer_semantic_hash(optimizer)
    before_steps = d16.optimizer_steps(before_optimizer)
    optimizer.zero_grad(set_to_none=True)
    rollout = d17.base.causal_paired_rollout(model, batch["inputs"], batch["history_positions"], batch["history_times"], batch["target_times"], batch["starts"], batch["ends"], pre["stats"]["channels"], horizon=ACTIVE_HORIZON, target_positions_for_teacher=batch["target_positions"], hidden_consistency_enabled=True)
    require(rollout.get("free_branch_reads_target_positions") is False and rollout.get("teacher_reference_stop_gradient") is True, f"GRAPH_SEMANTICS_GATE_FAILED:{name}:{update}")
    predictions = rollout["predictions"]
    targets = batch["target_positions"][:, :ACTIVE_HORIZON, :]
    step_losses = torch.nn.functional.smooth_l1_loss(predictions, targets, beta=d17.POSITION_BETA, reduction="none").mean(dim=(0, 2))
    weights = d17.rollout_horizon_weights(ACTIVE_HORIZON).to(device=predictions.device, dtype=predictions.dtype)
    weighted_terms = step_losses * weights
    full_rollout_loss = torch.sum(weighted_terms)
    h1_loss = step_losses[0]
    h32_term = weighted_terms[-1]
    hidden_loss = rollout["hidden_loss"]
    attenuation = spec.get("attenuation")
    if attenuation is not None:
        factors = torch.ones_like(weighted_terms)
        for horizon in spec.get("attenuated_region", range(18, 32)):
            factors[int(horizon) - 1] = float(attenuation)
        rollout_loss = torch.sum(weighted_terms * factors)
        structure_identity = f"H18_H31_ATTENUATED_{float(attenuation):g}_BEFORE_BACKWARD"
    else:
        rollout_loss = full_rollout_loss
        structure_identity = "NONE"
    rest_loss = rollout_loss + d17.LAMBDA_HIDDEN * hidden_loss
    canonical_total_objective = full_rollout_loss + d17.LAMBDA_HIDDEN * hidden_loss + d17.LAMBDA_H1 * h1_loss
    require(all(bool(torch.isfinite(value)) for value in (full_rollout_loss, rollout_loss, h1_loss, hidden_loss, rest_loss, canonical_total_objective)), f"NONFINITE_OBJECTIVE:{name}:{update}")
    h1_gradients, _ = d16.diagnostic_gradients(h1_loss, named)
    rest_gradients, _ = d16.diagnostic_gradients(rest_loss, named)
    h32_gradients, _ = d16.diagnostic_gradients(h32_term, named)
    rng_after_diagnostics = d17.base.capture_rng()
    require(d17.base.rng_digest(rng_after_diagnostics) == d17.base.rng_digest(rng_before), f"DIAGNOSTIC_RNG_CONTAMINATION:{name}:{update}")
    rule = str(spec["rule"])
    if rule == "cagrad":
        combined_gradients, rule_metrics = cagrad_gradient(h1_gradients, rest_gradients, float(spec["cagrad_c"]))
    else:
        combined_gradients = add_maps(rest_gradients, h1_gradients)
        rule_metrics = {}
    raw_gradients = clone_map(combined_gradients)
    set_gradients(model, raw_gradients)
    require(d17.base.gradients_are_finite(model), f"NONFINITE_RAW_GRADIENT:{name}:{update}")
    raw_norm = norm(raw_gradients.values())
    returned_preclip = finite(torch.nn.utils.clip_grad_norm_(model.parameters(), d17.CLIP_MAX_NORM, norm_type=2.0, error_if_nonfinite=False, foreach=None))
    require(math.isclose(raw_norm, returned_preclip, rel_tol=1.0e-5, abs_tol=1.0e-5), f"CLIP_RETURNED_NORM_MISMATCH:{name}:{update}")
    canonical_postclip, none_flags = d16.capture_gradients(named)
    postclip_norm = norm(canonical_postclip.values())
    clip_coefficient = 1.0 if returned_preclip <= d17.CLIP_MAX_NORM else min(1.0, d17.CLIP_MAX_NORM / (returned_preclip + 1.0e-6))
    residual = norm(canonical_postclip[name_] - raw_gradients[name_] * clip_coefficient for name_ in d16.ALL_NAMES)
    require(residual <= 1.0e-7 + 2.0e-5 * max(postclip_norm, 1.0), f"POSTCLIP_RECONSTRUCTION_FAILED:{name}:{update}")
    presented_gradients = canonical_postclip
    directional_bias_applied = "NO"
    directional_raw_norm = None
    directional_final_norm = None
    if "postclip_h1_coefficient" in spec:
        coefficient = float(spec["postclip_h1_coefficient"])
        h1_norm = norm(h1_gradients.values())
        d_h1 = scale_map(h1_gradients, 1.0 / max(h1_norm, EPS))
        trial_raw = add_maps(canonical_postclip, scale_map(d_h1, coefficient * postclip_norm))
        directional_raw_norm = norm(trial_raw.values())
        presented_gradients = scale_map(trial_raw, postclip_norm / max(directional_raw_norm, EPS))
        directional_final_norm = norm(presented_gradients.values())
        require(math.isclose(directional_final_norm, postclip_norm, rel_tol=2.0e-5, abs_tol=2.0e-7), f"POSTCLIP_NORM_PRESERVATION_FAILED:{name}:{update}")
        directional_bias_applied = "YES"
    set_gradients(model, presented_gradients)
    require(d17.base.gradients_are_finite(model), f"NONFINITE_PRESENTED_GRADIENT:{name}:{update}")
    rng_before_optimizer = d17.base.capture_rng()
    optimizer.step()
    after_model = d16.clone_parameters(model)
    after_optimizer = d16.clone_named_optimizer_state(optimizer)
    after_model_hash = d17.canonical_model_state_sha256(model.state_dict())
    after_optimizer_hash = d17.base.optimizer_semantic_hash(optimizer)
    state_finite = d16.state_finite(model, optimizer)
    require(state_finite, f"NONFINITE_STATE_AFTER_UPDATE:{name}:{update}")
    rng_after_training = d17.base.capture_rng()
    require(d17.base.rng_digest(rng_before_optimizer) == d17.base.rng_digest(rng_after_training), f"OPTIMIZER_RNG_CONSUMPTION_UNMATCHED:{name}:{update}")
    adamw_metrics, _ = d17.d7.adamw_decomposition(model, optimizer, before_model, presented_gradients, after_model, state["table"])
    validation, validation_meta = d17.d7.validation_snapshot(model, pre["validation"], pre["stats"]["channels"], True)
    require(validation_meta["rng_unchanged_or_restored"] in ("YES", "RESTORED"), f"VALIDATION_RNG_GATE_FAILED:{name}:{update}")
    rng_after_validation = d17.base.capture_rng()
    after_steps = d16.optimizer_steps(after_optimizer)
    expected_after_steps = {param_name: before_steps[param_name] + 1 for param_name in d16.ALL_NAMES}
    optimizer_state_identity_valid = after_steps == expected_after_steps
    parameter_delta = {param_name: after_model[param_name] - before_model[param_name] for param_name in d16.ALL_NAMES}
    adamw_whole = adamw_metrics["aggregate"]["whole_model"]
    parameter_delta_norm = d16.norm(parameter_delta.values())
    adamw_update_norm = float(adamw_whole["adaptive_update_norm"])
    h32_after = float(validation["32"])
    h32_preserved = h32_after <= H32_THRESHOLD
    pathological = (not math.isfinite(parameter_delta_norm) or not math.isfinite(adamw_update_norm) or parameter_delta_norm > 1.0e3 or adamw_update_norm > 1.0e3)
    finite_status = "FINITE" if state_finite and all(math.isfinite(float(value)) for value in (raw_norm, returned_preclip, postclip_norm, parameter_delta_norm, adamw_update_norm, validation["1"], validation["32"])) else "NONFINITE"
    optimizer_corrupted = not optimizer_state_identity_valid or adamw_whole["decomposition_pass_fail"] != "PASS"
    valid = finite_status == "FINITE" and not optimizer_corrupted and not pathological and h32_preserved and d17.base.rng_digest(rng_after_validation) == d17.base.rng_digest(rng_before)
    record = {
        "update": update, "proposal": name, "family": spec["family"], "selected": "NO", "trial_state": "TRIAL_NOT_COMMITTED", "valid": "YES" if valid else "NO",
        "H1_before": None, "H1_after": float(validation["1"]), "H32_before": None, "H32_after": h32_after,
        "canonical_total_objective": finite(canonical_total_objective), "rollout_loss_full": finite(full_rollout_loss), "rollout_loss_used_for_backward": finite(rollout_loss), "weighted_h32_term": finite(h32_term), "explicit_h1_loss": finite(h1_loss), "hidden_loss_unweighted": finite(hidden_loss),
        "gradient_rule": rule, "structure_identity": structure_identity, "raw_gradient_norm": raw_norm, "preclip_norm": returned_preclip, "postclip_norm": postclip_norm, "postclip_norm_final": directional_final_norm if directional_final_norm is not None else postclip_norm, "clipping_activated": "YES" if returned_preclip > d17.CLIP_MAX_NORM else "NO", "clip_coefficient": clip_coefficient, "postclip_reclip": "NO" if directional_bias_applied == "YES" else "NOT_APPLICABLE", "postclip_h1_bias_applied": directional_bias_applied, "postclip_h1_directional_coefficient": spec.get("postclip_h1_coefficient"), "postclip_directional_raw_norm": directional_raw_norm,
        "raw_h1_gradient_norm": d16.norm(h1_gradients.values()), "raw_rest_gradient_norm": d16.norm(rest_gradients.values()), "h1_rest_dot_product": dot(h1_gradients, rest_gradients), "h1_gradient_norm": d16.norm(h1_gradients.values()), "h32_gradient_norm": d16.norm(h32_gradients.values()),
        "adamw_update_norm": adamw_update_norm, "parameter_delta_norm": parameter_delta_norm, "adamw_decomposition_status": adamw_whole["decomposition_pass_fail"], "adamw_decomposition_residual_norm": adamw_whole["decomposition_residual_norm"], "optimizer_state_identity_valid": "YES" if optimizer_state_identity_valid else "NO", "pathological_update_behavior": "YES" if pathological else "NO",
        "finite_status": finite_status, "H32_preservation": "YES" if h32_preserved else "NO", "training_stability": "PASS" if valid else "FAIL", "batch_or_RNG_identity_valid": "YES" if d17.base.rng_digest(rng_after_validation) == d17.base.rng_digest(rng_before) else "NO", "optimizer_corrupted": "YES" if optimizer_corrupted else "NO",
        "batch_identity": batch_record["batch_window_sha256"], "batch_window_sha256": batch_record["batch_window_sha256"], "rng_before_identity": d17.base.rng_digest(rng_before), "rng_after_identity": d17.base.rng_digest(rng_after_validation), "model_hash_before": before_model_hash, "model_hash_after": after_model_hash, "optimizer_hash_before": before_optimizer_hash, "optimizer_hash_after": after_optimizer_hash, "optimizer_step_before": min(before_steps.values()), "optimizer_step_after": min(after_steps.values()),
        "gradient_digest": d17.d7.d6.named_tensor_digest(presented_gradients), "raw_gradient_digest": d17.d7.d6.named_tensor_digest(raw_gradients), "postclip_gradient_digest": d17.d7.d6.named_tensor_digest(canonical_postclip), "parameter_delta_digest": d17.d7.d6.named_tensor_digest(parameter_delta), "gradient_none_pattern": json.dumps(none_flags, sort_keys=True), "rule_metrics": json.dumps(rule_metrics, sort_keys=True),
    }
    capture = {"model": model, "optimizer": optimizer, "rng": copy.deepcopy(rng_after_validation), "record": record}
    return record, model, optimizer, capture


def completion_gate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) < CONFIRMATION_WINDOW:
        return {"stage3_completion_gate": "NOT_READY", "window_updates": [], "endpoint_H1": None, "last_8_mean_H1": None, "H1_target_pass": False, "H32_preservation_pass": False, "stability_pass": False, "real_forward_pass": False}
    window = list(rows[-CONFIRMATION_WINDOW:])
    updates = [int(row["update"]) for row in window]
    h1 = [float(row["H1"]) for row in window]
    h32 = [float(row["H32"]) for row in window]
    consecutive = updates == list(range(updates[0], updates[0] + CONFIRMATION_WINDOW))
    h1_pass = h1[-1] <= TARGET_H1 and sum(h1) / CONFIRMATION_WINDOW <= TARGET_H1
    h32_pass = all(value <= H32_THRESHOLD for value in h32)
    stable = all(row["finite_status"] == "FINITE" and row["training_stability"] == "PASS" and row["adamw_decomposition_status"] == "PASS" and row["selected_valid"] == "YES" for row in window)
    real_forward = consecutive and all(update > EXPECTED_START_UPDATE for update in updates)
    return {"stage3_completion_gate": "PASSED" if h1_pass and h32_pass and stable and real_forward else "FAILED", "window_updates": updates, "endpoint_H1": h1[-1], "last_8_mean_H1": sum(h1) / CONFIRMATION_WINDOW, "last_8_H1_values": h1, "last_8_H32_values": h32, "H1_target_pass": h1_pass, "H32_preservation_pass": h32_pass, "stability_pass": stable, "real_forward_pass": real_forward}


def checkpoint_payload(model: Any, optimizer: Any, rng: Mapping[str, Any], update: int, spec: Mapping[str, Any], record: Mapping[str, Any]) -> dict[str, Any]:
    return {"schema_version": "stage3_h13_d24_resumable_committed_checkpoint_v1", "experiment_id": EXPERIMENT_ID, "proposal": spec["name"], "proposal_spec": dict(spec), "completed_optimizer_step": int(update), "next_schedule_index": int(update), "model_state_dict": copy.deepcopy(model.state_dict()), "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()), "scheduler_state": None, "rng_state": copy.deepcopy(rng), "batch_position": {"completed_optimizer_step": int(update), "next_zero_based_schedule_index": int(update)}, "last_record_identity": {"update": int(update), "batch_window_sha256": record["batch_window_sha256"], "model_state_hash": record["model_hash_after"], "optimizer_state_hash": record["optimizer_hash_after"]}, "resumable": True, "scientific_state": "COMMITTED_FORWARD_STATE"}


def save_committed_checkpoint(path: Path, model: Any, optimizer: Any, rng: Mapping[str, Any], update: int, spec: Mapping[str, Any], record: Mapping[str, Any]) -> None:
    torch.save(checkpoint_payload(model, optimizer, rng, update, spec, record), path)


def selection_key(record: Mapping[str, Any]) -> tuple[float, float, float, float]:
    h32_margin = H32_THRESHOLD - float(record["H32_after"])
    return (float(record["H1_after"]), -h32_margin, float(record["canonical_total_objective"]), float(record["parameter_delta_norm"]))


def choose_active_proposals(records: Sequence[Mapping[str, Any]], specs: Mapping[str, Mapping[str, Any]]) -> list[str]:
    valid = [row for row in records if row.get("valid") == "YES" and row.get("H32_preservation") == "YES"]
    names = sorted(set(str(row["proposal"]) for row in valid), key=lambda name: min(selection_key(row) for row in valid if row["proposal"] == name))
    if len(names) >= 2:
        return names[:2]
    return list(specs)[:2]


def run_block(label: str, proposal_specs: Mapping[str, Mapping[str, Any]], start_update: int, end_update: int, model: Any, optimizer: Any, rng: Mapping[str, Any], state: Mapping[str, Any], output: Path, committed: list[dict[str, Any]], proposal_records: list[dict[str, Any]], selection_records: list[dict[str, Any]], checkpoints: dict[int, Path]) -> tuple[Any, Any, dict[str, Any], dict[str, Any], str | None]:
    stop_reason = None
    for update in range(start_update, end_update + 1):
        before_h1 = float(committed[-1]["H1"]) if committed else float(state["start_H1"])
        before_h32 = float(committed[-1]["H32"]) if committed else float(state["start_H32"])
        trials: list[dict[str, Any]] = []
        best_trial_record: dict[str, Any] | None = None
        best_trial_model: Any | None = None
        best_trial_optimizer: Any | None = None
        best_trial_rng: dict[str, Any] | None = None
        for name, spec in proposal_specs.items():
            trial_model, trial_optimizer = clone_trial(model, optimizer, state["bundle"])
            try:
                record, trial_model, trial_optimizer, capture = run_trial(name, spec, trial_model, trial_optimizer, state, update - 1, rng)
                record["H1_before"] = before_h1
                record["H32_before"] = before_h32
                record["block"] = label
                trials.append(record)
                if record.get("valid") == "YES" and record.get("H32_preservation") == "YES" and (best_trial_record is None or selection_key(record) < selection_key(best_trial_record)):
                    if best_trial_model is not None:
                        del best_trial_model, best_trial_optimizer, best_trial_rng
                    best_trial_record = record
                    best_trial_model = trial_model
                    best_trial_optimizer = trial_optimizer
                    best_trial_rng = capture["rng"]
                else:
                    del trial_model, trial_optimizer, capture
            except Exception as exc:
                record = {"update": update, "proposal": name, "family": spec["family"], "selected": "NO", "trial_state": "TRIAL_FAILED_NOT_COMMITTED", "valid": "NO", "H1_before": before_h1, "H1_after": None, "H32_before": before_h32, "H32_after": None, "gradient_rule": spec["rule"], "finite_status": "NONFINITE", "H32_preservation": "NO", "training_stability": "FAIL", "batch_or_RNG_identity_valid": "NO", "optimizer_corrupted": "NO", "failure_reason": f"{type(exc).__name__}:{exc}", "block": label}
                trials.append(record)
        proposal_records.extend(trials)
        valid = [record for record in trials if record.get("valid") == "YES" and record.get("H32_preservation") == "YES"]
        require(valid and best_trial_record is not None and best_trial_model is not None and best_trial_optimizer is not None and best_trial_rng is not None, f"NO_VALID_H32_PRESERVING_PROPOSAL:{update}")
        selected = best_trial_record
        selected_name = str(selected["proposal"])
        selected["selected"] = "YES"
        selected["trial_state"] = "COMMITTED_FORWARD_STATE"
        selected["selected_valid"] = "YES"
        for record in trials:
            if record is not selected:
                record["selected_valid"] = "NO"
        model, optimizer, rng = best_trial_model, best_trial_optimizer, best_trial_rng
        committed_row = {"update": update, "selected_proposal": selected_name, "selected_family": selected["family"], "H1": float(selected["H1_after"]), "H32": float(selected["H32_after"]), "H1_before": before_h1, "H32_before": before_h32, "gradient_rule": selected["gradient_rule"], "raw_gradient_norm": selected.get("raw_gradient_norm"), "postclip_norm": selected.get("postclip_norm_final", selected.get("postclip_norm")), "adamw_update_norm": selected.get("adamw_update_norm"), "parameter_delta_norm": selected.get("parameter_delta_norm"), "finite_status": selected["finite_status"], "training_stability": selected["training_stability"], "adamw_decomposition_status": selected["adamw_decomposition_status"], "H32_preservation": selected["H32_preservation"], "batch_identity": selected.get("batch_identity"), "rng_identity": selected.get("rng_after_identity"), "model_hash": selected.get("model_hash_after"), "optimizer_hash": selected.get("optimizer_hash_after"), "selected_valid": "YES", "block": label}
        committed.append(committed_row)
        checkpoint_path = output / "checkpoints" / f"committed_update_{update:03d}.pt"
        save_committed_checkpoint(checkpoint_path, model, optimizer, rng, update, proposal_specs[selected_name], selected)
        checkpoints[update] = checkpoint_path
        print(f"D24_COMMITTED update={update} proposal={selected_name} H1={float(selected['H1_after']):.17g} H32={float(selected['H32_after']):.17g}", flush=True)
        selection_records.append({"update": update, "block": label, "selected_proposal": selected_name, "selected_family": selected["family"], "candidate_set": ",".join(proposal_specs), "valid_candidate_set": ",".join(record["proposal"] for record in valid), "H1_before": before_h1, "H1_after": selected["H1_after"], "H32_before": before_h32, "H32_after": selected["H32_after"], "selection_key": json.dumps(list(selection_key(selected))), "selection_rule": "minimum realized post_step_H1; H32 margin; canonical total objective; parameter delta norm", "checkpoint_path": str(checkpoint_path.resolve()), "completion_window_pass": "YES" if completion_gate(committed)["stage3_completion_gate"] == "PASSED" else "NO"})
        current_gate = completion_gate(committed)
        if current_gate["stage3_completion_gate"] == "PASSED":
            stop_reason = "COMPLETION_GATE"
            return model, optimizer, rng, current_gate, stop_reason
        if len(committed) >= CONFIRMATION_WINDOW:
            window = committed[-CONFIRMATION_WINDOW:]
            if float(window[-1]["H1"]) >= float(window[0]["H1"]) and all(row["H32_preservation"] == "YES" for row in window):
                stop_reason = "PLATEAU_FULL_8_UPDATE_WINDOW"
                return model, optimizer, rng, current_gate, stop_reason
    return model, optimizer, rng, completion_gate(committed), stop_reason


def manifest_for(output: Path) -> dict[str, Any]:
    excluded = {"FINAL_REPORT.md", "SHA256_MANIFEST.json"}
    files = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name not in excluded:
            files.append({"relative_path": str(path.relative_to(output)).replace("\\", "/"), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    return {"schema_version": "stage3_h13_d24_sha256_manifest_v1", "hash_algorithm": "SHA-256", "self_excluded": True, "excluded_from_manifest": sorted(excluded), "file_count": len(files), "files": files}


def classify(state: Mapping[str, Any], committed: Sequence[Mapping[str, Any]], completion: Mapping[str, Any], blocker: str | None, stop_reason: str | None) -> str:
    if completion["stage3_completion_gate"] == "PASSED":
        return "STAGE3_COMPLETED_H1_TARGET_SUSTAINED"
    if blocker:
        return "D24_EXECUTION_BLOCKED"
    if stop_reason == "PLATEAU_FULL_8_UPDATE_WINDOW":
        return "D24_POSTCLIP_AND_CAGRAD_FAMILY_PLATEAUED"
    if not committed:
        return "D24_EXECUTION_BLOCKED"
    improvement = float(state["start_H1"]) - float(committed[-1]["H1"])
    if improvement >= STRONG_THRESHOLD:
        return "D24_LARGE_FORWARD_H1_IMPROVEMENT"
    if improvement >= MATERIALITY_THRESHOLD:
        return "D24_MATERIAL_FORWARD_H1_IMPROVEMENT"
    if improvement > 0.0:
        return "D24_POSITIVE_FORWARD_H1_IMPROVEMENT"
    return "D24_POSTCLIP_AND_CAGRAD_FAMILY_PLATEAUED"


def final_report(summary: Mapping[str, Any], committed: Sequence[Mapping[str, Any]]) -> str:
    gate = summary["completion_gate"]
    lines = [
        f"{EXPERIMENT_ID}:", summary["status"], "", "FIRST_BLOCKER:", summary.get("first_blocker") or "none", "", "FINAL_CLASSIFICATION:", summary["classification"], "", "STAGE_3_STATUS:", summary["stage3_status"], "", "ONE_SENTENCE_VERDICT:", summary["verdict"], "",
        "START_CHECKPOINT:", summary["start_checkpoint"], f"START_UPDATE: {summary['start_update']}", f"UPDATE_393_EXECUTED: {summary['update_393_executed']}", f"MAX_VALID_UPDATE_REACHED: {summary['max_valid_update']}", f"COMMITTED_FORWARD_UPDATES: {summary['committed_forward_updates']}", "",
        f"START_H1: {summary['start_H1']:.17g}", f"ENDPOINT_H1: {summary['endpoint_H1']}", f"ABSOLUTE_H1_IMPROVEMENT: {summary['absolute_H1_improvement']}", f"RELATIVE_H1_IMPROVEMENT: {summary['relative_H1_improvement']}", f"MATERIAL_THRESHOLD_MULTIPLE: {summary['material_threshold_multiple']}", f"DISTANCE_TO_5E-05: {summary['distance_to_target']}", "",
        f"LAST_8_MEAN_H1: {gate['last_8_mean_H1']}", f"H32_PRESERVATION: {gate['H32_preservation_pass']}", f"TRAINING_STABILITY: {gate['stability_pass']}", f"STAGE_3_COMPLETION_GATE: {gate['stage3_completion_gate']}", "",
        f"MOST_FREQUENT_WINNING_PROPOSAL: {summary['most_frequent_winning_proposal']}", f"BEST_FORWARD_CHECKPOINT: {summary['best_forward_checkpoint']}", f"BEST_FORWARD_CHECKPOINT_SHA256: {summary['best_forward_checkpoint_sha256']}", "",
        "| update | selected_proposal | H1 | H32 | completion_window_pass |", "|---:|---|---:|---:|---|",
    ]
    for row in committed:
        lines.append(f"| {row['update']} | {row['selected_proposal']} | {float(row['H1']):.17g} | {float(row['H32']):.17g} | {'YES' if any(int(x) == int(row['update']) for x in gate.get('window_updates', [])) and gate['stage3_completion_gate'] == 'PASSED' else 'NO'} |")
    lines += ["", "NEXT_ACTION:", "STOP_STAGE_3_AND_PRESERVE_FINAL_STATE" if gate["stage3_completion_gate"] == "PASSED" else "CONTINUE_FROM_BEST_FORWARD_STATE" if summary["classification"] != "D24_POSTCLIP_AND_CAGRAD_FAMILY_PLATEAUED" else "NEXT_OPTIMIZATION_FAMILY: TASK_AWARE_ADAMW_STATE_SEPARATION", ""]
    return "\n".join(lines)


def execute(authorization: Path, output: Path) -> dict[str, Any]:
    require(not output.exists(), f"OUTPUT_ALREADY_EXISTS:{output}")
    output.mkdir(parents=True)
    (output / "checkpoints").mkdir()
    preflight_state, runtime = preflight(authorization)
    write_json(output / "STAGE3_COMPLETION_CONTRACT.json", CONTRACT)
    model, optimizer, rng = runtime["incumbent_model"], runtime["incumbent_optimizer"], runtime["incumbent_rng"]
    committed: list[dict[str, Any]] = []
    proposal_records: list[dict[str, Any]] = []
    selection_records: list[dict[str, Any]] = []
    checkpoints: dict[int, Path] = {}
    blocker = None
    stop_reason = None
    wave_labels: list[str] = []
    try:
        wave_a_specs = {name: proposal_spec(name) for name in PROPOSAL_NAMES}
        current_specs: dict[str, Mapping[str, Any]] = wave_a_specs
        wave_labels.append("WAVE_A")
        model, optimizer, rng, completion, stop_reason = run_block("WAVE_A", wave_a_specs, 393, 400, model, optimizer, rng, {**runtime, "start_H1": preflight_state["start_H1"], "start_H32": preflight_state["start_H32"]}, output, committed, proposal_records, selection_records, checkpoints)
        if completion["stage3_completion_gate"] != "PASSED" and stop_reason != "PLATEAU_FULL_8_UPDATE_WINDOW":
            block_improvement = preflight_state["start_H1"] - float(committed[-1]["H1"])
            valid_a = [record for record in proposal_records if record.get("block") == "WAVE_A" and record.get("valid") == "YES"]
            winner_name = min(valid_a, key=selection_key)["proposal"] if valid_a else "P0"
            if block_improvement < MATERIALITY_THRESHOLD:
                wave_b_specs = {"Q0": proposal_spec("Q0", q0_source=wave_a_specs.get(winner_name, wave_a_specs["P0"])), "Q1": proposal_spec("Q1"), "Q2": proposal_spec("Q2"), "Q3": proposal_spec("Q3"), "Q4": proposal_spec("Q4")}
                wave_label = "WAVE_B"
            else:
                wave_b_specs = {name: copy.deepcopy(wave_a_specs[name]) for name in PROPOSAL_NAMES}
                wave_label = "WAVE_A_CONTINUATION"
            current_specs = wave_b_specs
            wave_labels.append(wave_label)
            model, optimizer, rng, completion, stop_reason = run_block(wave_label, wave_b_specs, 401, 408, model, optimizer, rng, {**runtime, "start_H1": preflight_state["start_H1"], "start_H32": preflight_state["start_H32"]}, output, committed, proposal_records, selection_records, checkpoints)
        if completion["stage3_completion_gate"] != "PASSED" and stop_reason != "PLATEAU_FULL_8_UPDATE_WINDOW" and committed and int(committed[-1]["update"]) >= 408:
            recent_records = [record for record in proposal_records if record.get("block") in ("WAVE_B", "WAVE_A_CONTINUATION")]
            active_names = choose_active_proposals(recent_records or proposal_records, current_specs)
            compact_specs: dict[str, Mapping[str, Any]] = {name: copy.deepcopy(current_specs[name]) for name in active_names}
            wave_labels.append("COMPACT_FORWARD")
            model, optimizer, rng, completion, stop_reason = run_block("COMPACT_FORWARD", compact_specs, int(committed[-1]["update"]) + 1, MAX_UPDATE, model, optimizer, rng, {**runtime, "start_H1": preflight_state["start_H1"], "start_H32": preflight_state["start_H32"]}, output, committed, proposal_records, selection_records, checkpoints)
    except Exception as exc:
        blocker = f"{type(exc).__name__}:{exc}"
        completion = completion_gate(committed)
    if not committed:
        endpoint_h1 = None
        endpoint_h32 = None
        absolute_improvement = 0.0
    else:
        endpoint_h1 = float(committed[-1]["H1"])
        endpoint_h32 = float(committed[-1]["H32"])
        absolute_improvement = float(preflight_state["start_H1"]) - endpoint_h1
    completion = completion_gate(committed)
    classification = classify(preflight_state, committed, completion, blocker, stop_reason)
    update_numbers = [int(row["update"]) for row in committed]
    winning_counts: dict[str, int] = {}
    for row in committed:
        winning_counts[row["selected_proposal"]] = winning_counts.get(row["selected_proposal"], 0) + 1
    most_frequent = sorted(winning_counts.items(), key=lambda item: (-item[1], item[0]))[0][0] if winning_counts else None
    best_row = min(committed, key=lambda row: float(row["H1"])) if committed else None
    best_checkpoint = checkpoints.get(int(best_row["update"])) if best_row else None
    best_sha = sha256_file(best_checkpoint) if best_checkpoint and best_checkpoint.is_file() else None
    best_metadata = {"proposal": best_row["selected_proposal"] if best_row else None, "update": int(best_row["update"]) if best_row else None, "H1": float(best_row["H1"]) if best_row else None, "H32": float(best_row["H32"]) if best_row else None, "path": str(best_checkpoint.resolve()) if best_checkpoint else None, "sha256": best_sha, "completed_optimizer_step": int(best_row["update"]) if best_row else None, "resumable": bool(best_checkpoint and best_checkpoint.is_file())}
    write_json(output / "BEST_FORWARD_CHECKPOINT_METADATA.json", best_metadata)
    trajectory_fields = ["update", "selected_proposal", "selected_family", "H1_before", "H1", "H32_before", "H32", "gradient_rule", "raw_gradient_norm", "postclip_norm", "adamw_update_norm", "parameter_delta_norm", "finite_status", "H32_preservation", "training_stability", "adamw_decomposition_status", "batch_identity", "rng_identity", "model_hash", "optimizer_hash", "block"]
    write_csv(output / "D24_COMMITTED_FORWARD_TRAJECTORY.csv", trajectory_fields, committed)
    proposal_fields = ["update", "proposal", "family", "selected", "trial_state", "valid", "H1_before", "H1_after", "H32_before", "H32_after", "canonical_total_objective", "gradient_rule", "raw_gradient_norm", "preclip_norm", "postclip_norm", "postclip_norm_final", "clipping_activated", "clip_coefficient", "postclip_reclip", "postclip_h1_bias_applied", "postclip_h1_directional_coefficient", "adamw_update_norm", "parameter_delta_norm", "finite_status", "H32_preservation", "training_stability", "optimizer_corrupted", "optimizer_state_identity_valid", "batch_or_RNG_identity_valid", "batch_identity", "rng_before_identity", "rng_after_identity", "model_hash_before", "model_hash_after", "optimizer_hash_before", "optimizer_hash_after", "optimizer_step_before", "optimizer_step_after", "failure_reason", "block"]
    write_csv(output / "D24_PROPOSAL_RESULTS.csv", proposal_fields, proposal_records)
    write_csv(output / "D24_PROPOSAL_SELECTION_LOG.csv", ["update", "block", "selected_proposal", "selected_family", "candidate_set", "valid_candidate_set", "H1_before", "H1_after", "H32_before", "H32_after", "selection_key", "selection_rule", "checkpoint_path", "completion_window_pass"], selection_records)
    write_json(output / "STAGE3_COMPLETION_GATE_RESULT.json", completion)
    summary = {"status": "PASSED" if blocker is None else "FAILED", "first_blocker": blocker, "classification": classification, "stage3_status": "COMPLETE" if completion["stage3_completion_gate"] == "PASSED" else "BLOCKED" if blocker else "IN_PROGRESS", "verdict": "The frozen Stage-3 completion gate passed on the committed realized D24 trajectory." if completion["stage3_completion_gate"] == "PASSED" else "D24 committed valid realized AdamW forward updates from the authenticated C5 state without yet passing the frozen completion gate." if blocker is None else "D24 could not commit a valid forward update after rejecting all available trial proposals.", "start_checkpoint": str(START_CHECKPOINT.resolve()), "start_checkpoint_sha256": EXPECTED_START_CHECKPOINT_SHA256, "start_update": EXPECTED_START_UPDATE, "update_393_executed": "YES" if 393 in update_numbers else "NO", "max_valid_update": max(update_numbers) if update_numbers else EXPECTED_START_UPDATE, "committed_forward_updates": update_numbers, "start_H1": float(preflight_state["start_H1"]), "start_H32": float(preflight_state["start_H32"]), "endpoint_H1": endpoint_h1, "endpoint_H32": endpoint_h32, "absolute_H1_improvement": absolute_improvement, "relative_H1_improvement": absolute_improvement / float(preflight_state["start_H1"]) if committed else 0.0, "material_threshold_multiple": absolute_improvement / MATERIALITY_THRESHOLD if committed else 0.0, "distance_to_target": max(0.0, endpoint_h1 - TARGET_H1) if endpoint_h1 is not None else None, "completion_gate": completion, "H32_preservation": completion["H32_preservation_pass"], "training_stability": completion["stability_pass"], "most_frequent_winning_proposal": most_frequent, "winning_counts": winning_counts, "best_forward_checkpoint": str(best_checkpoint.resolve()) if best_checkpoint else None, "best_forward_checkpoint_sha256": best_sha, "wave_labels": wave_labels, "stop_reason": stop_reason, "trial_count": len(proposal_records), "selection_count": len(selection_records), "frozen20_used": "NO", "r9_executed": "NO", "causal_localization_continued": "NO", "next_optimization_family": "TASK_AWARE_ADAMW_STATE_SEPARATION" if classification == "D24_POSTCLIP_AND_CAGRAD_FAMILY_PLATEAUED" else None, "authorization": preflight_state["authorization"], "checkpoint_identity": preflight_state["checkpoint"], "runtime": preflight_state["runtime"]}
    write_json(output / "D24_RUN_MANIFEST.json", {"schema_version": "stage3_h13_d24_run_manifest_v1", "experiment_id": EXPERIMENT_ID, "captured_at_utc": utc_now(), "runner_path": str(Path(__file__).resolve()), "runner_sha256": sha256_file(Path(__file__).resolve()), "authorization": preflight_state["authorization"], "source_revision": EXPECTED_SOURCE_REVISION, "start_checkpoint": preflight_state["checkpoint"], "proposal_sets": {"wave_a": list(PROPOSAL_NAMES), "wave_b": ["Q0", "Q1", "Q2", "Q3", "Q4"], "compact_forward": "two best valid realized proposal families"}, "updates_authorized": {"wave_a": [393, 394, 395, 396, 397, 398, 399, 400], "wave_b": [401, 402, 403, 404, 405, 406, 407, 408], "compact_ceiling": [409, 440]}, "materiality_is_forward_gate": False, "frozen20_used": "NO", "r9_executed": "NO", "causal_localization_continued": "NO", "summary": summary})
    write_json(output / "D24_RUN_SUMMARY.json", summary)
    (output / "FINAL_REPORT.md").write_text(final_report(summary, committed), encoding="utf-8", newline="\n")
    manifest = manifest_for(output)
    write_json(output / "SHA256_MANIFEST.json", manifest)
    return {"status": summary["status"], "output": str(output.resolve()), "classification": classification, "stage3_status": summary["stage3_status"], "max_valid_update": summary["max_valid_update"], "first_blocker": blocker, "completion_gate": completion}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = execute(args.authorization.resolve(), args.output.resolve())
        print(json.dumps(result, sort_keys=True, ensure_ascii=True))
        return 0 if result["status"] == "PASSED" else 2
    except Exception as exc:
        print(json.dumps({"status": "FAILED", "first_blocker": f"{type(exc).__name__}:{exc}", "scientific_updates_executed": False}, sort_keys=True, ensure_ascii=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
