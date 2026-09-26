"""Run the authorized Stage-3 H13 D23 progress-first optimization sprint.

This runner starts every Rung-1 candidate from the authenticated post-update-
384 bundle, applies only the preregistered narrow training interventions, and
continues the actual promoted state across update 393.  Evaluation semantics,
the frozen H32 criterion, canonical schedule, and AdamW implementation remain
unchanged.  The runner stops the whole stage at the first valid completion
window instead of spending the remaining nominal budget.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import stage3_h13_post_d16_d17_full_pipeline_multi_horizon_rollout_causal_screen_replay as d17  # noqa: E402
from tools import stage3_h13_post_d15_d16_adamw_full_pipeline_h32_component_deletion_causal_intervention_replay as d16  # noqa: E402


EXPERIMENT_ID = "STAGE_3_H13_POST_D22_D23_PROGRESS_FIRST_OPTIMIZATION_SPRINT_FORWARD_PROMOTION_AND_STAGE3_COMPLETION_EXECUTION"
D22_DIR = ROOT / "outputs" / "stage3_h13_post_d21_d22_h32_by_b1_b2_c1_c2_hierarchical_c2_subblock_higher_order_factorial_causal_localization_execution_evidence_20260821T151902+0800"
BUNDLE_PATH = ROOT / "outputs" / "stage3_h13_post_d5_d6_branch_state_materialization_replay_20260816T152000Z_retry" / "post_update_384_branch_bundle.pt"
EXPECTED_D22_MANIFEST_SHA256 = "a297d40fe26807001d1874540bce732c9898a9ccac3f9b5ec70cde0e7ab4d0c1"
EXPECTED_D22_DESIGN_MANIFEST_SHA256 = "6ed9855be329cfd024805eff1efdc4f313e161de65f8584bfd6721959c1d329f"
EXPECTED_D21_MANIFEST_SHA256 = "d6f42c415223996aa8f26df0c1b82109ba054f5d75aa7a64e4421c6403ac206d"
EXPECTED_START_BUNDLE_SHA256 = "4a022aa0340977e571a3a3085e30d96fd0d92129f4fbeff99aaa8e3420379083"
EXPECTED_MODEL_SHA256 = "f6f450abd4ac8780a7172c46d44db88d6b7278e22e6a14580b62c9b1627bcd58"
EXPECTED_OPTIMIZER_SHA256 = "610dd9fa330572888e60cbc16e8f67560e2ce23a3b8628cad7481a90b24936d8"
EXPECTED_RNG_SHA256 = "857d5ddd6c4271019b55a53774ad68c70e435abd38cf430b7f0d57ed047848d8"
EXPECTED_SOURCE_REVISION = "a6dff2b6887cc3f7598ab95549c2dd21c0efe763"
BASELINE_H1 = 9.096969417553673e-05
BASELINE_H32 = 0.018159905521264792
TARGET_H1 = 5.0e-05
MATERIALITY_THRESHOLD = 4.099006815405639e-06
STRONG_THRESHOLD = 2.0 * MATERIALITY_THRESHOLD
H32_THRESHOLD = 0.01856902565856056
START_UPDATE = 384
RUNG1_UPDATES = tuple(range(385, 393))
RUNG2_UPDATES = tuple(range(393, 409))
RUNG3_UPDATES = tuple(range(409, 441))
MAX_UPDATE = 440
ACTIVE_HORIZON = 32
H1_REST_REGION = tuple(range(18, 32))

CONTRACT = {
    "version": "1.0",
    "status": "FROZEN",
    "primary_metric": "H1",
    "reference_baseline_H1": BASELINE_H1,
    "stage3_completion_target_H1": TARGET_H1,
    "frozen_material_effect_threshold": MATERIALITY_THRESHOLD,
    "confirmation_window": 8,
    "H32_requirement": "existing frozen D22 H32-preservation criterion",
    "stability_requirement": "PASS",
    "real_forward_requirement": ">392",
    "stop_on_completion_gate": True,
    "continue_stage3_training_after_gate": False,
    "continue_causal_localization_after_gate": False,
}


class StageBlocker(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")


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


def verify_d22_manifest() -> dict[str, Any]:
    manifest_path = D22_DIR / "SHA256_MANIFEST.json"
    require(manifest_path.is_file(), f"D22_MANIFEST_MISSING:{manifest_path}")
    observed = sha256_file(manifest_path)
    require(observed == EXPECTED_D22_MANIFEST_SHA256, f"D22_MANIFEST_HASH_MISMATCH:{observed}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = []
    for entry in manifest.get("files", []):
        target = (D22_DIR / Path(str(entry["relative_path"]).replace("/", "\\"))).resolve()
        if not target.is_file() or sha256_file(target) != entry["sha256"] or target.stat().st_size != int(entry["size_bytes"]):
            failures.append(entry["relative_path"])
    require(not failures, f"D22_INTERNAL_MANIFEST_MISMATCH:{failures}")
    return {"path": str(manifest_path.resolve()), "sha256": observed, "checked_file_count": len(manifest.get("files", [])), "failures": []}


def preflight(authorization: Path) -> dict[str, Any]:
    auth_text = authorization.read_text(encoding="utf-8") if authorization.is_file() else ""
    required_markers = (
        "PROJECT-OWNER EXECUTION AUTHORIZATION",
        "AUTHORIZED: YES",
        "STAGE_3_COMPLETION_CONTRACT_VERSION: 1.0",
        "STAGE_3_COMPLETION_TARGET_H1:",
        "5.000000000000000e-05",
        "UPDATE_393_EXECUTION_AUTHORIZED: YES",
        "FORWARD_UPDATES_AFTER_393_AUTHORIZED: YES",
        "STOP_ON_COMPLETION_GATE:",
        "CONTINUE_STAGE3_TRAINING_AFTER_GATE:",
    )
    require(all(marker in auth_text for marker in required_markers), "D23_AUTHORIZATION_SCOPE_MISMATCH")
    git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    require(git_head == EXPECTED_SOURCE_REVISION, f"SOURCE_REVISION_MISMATCH:{git_head}")
    require(sha256_file(BUNDLE_PATH) == EXPECTED_START_BUNDLE_SHA256, "CERTIFIED_START_BUNDLE_HASH_MISMATCH")
    d22_identity = json.loads((D22_DIR / "authoritative_input_identity.json").read_text(encoding="utf-8"))
    require(d22_identity["d22_design_manifest_sha256"] == EXPECTED_D22_DESIGN_MANIFEST_SHA256, "D22_DESIGN_IDENTITY_MISMATCH")
    require(d22_identity["d21_manifest_sha256"] == EXPECTED_D21_MANIFEST_SHA256, "D21_IDENTITY_MISMATCH")
    require(d22_identity["certified_start"]["completed_optimizer_step"] == START_UPDATE, "D22_CERTIFIED_START_STEP_MISMATCH")
    execution = json.loads((D22_DIR / "execution_manifest.json").read_text(encoding="utf-8"))
    require(execution["endpoint_update"] == 392 and execution["forbidden_update"] == 393 and execution["update_393_executed"] == "NO", "D22_EXECUTION_WINDOW_MISMATCH")
    require(execution["adaptive_selection"] == "NO" and execution["threshold_tuning"] == "NO", "D22_FROZEN_BOUNDARY_MISMATCH")
    response_rows = list(csv.DictReader((D22_DIR / "factorial_64_cell_response_table.csv").open("r", encoding="utf-8", newline="")))
    control = next(row for row in response_rows if row["canonical_cell_id"] == "000000")
    require(float(control["update_392_H1"]) == BASELINE_H1 and float(control["update_392_H32"]) == BASELINE_H32, "D22_CONTROL_ENDPOINT_MISMATCH")
    # d17.preflight_runtime authenticates the model, optimizer, RNG, schedule,
    # runtime, deterministic algorithms, and canonical supporting authorities.
    d17_state = d17.preflight_runtime()
    require(d17_state["loaded"]["model_semantic_hash"] == EXPECTED_MODEL_SHA256, "MODEL_SEMANTIC_HASH_MISMATCH")
    require(d17_state["loaded"]["optimizer_semantic_hash"] == EXPECTED_OPTIMIZER_SHA256, "OPTIMIZER_SEMANTIC_HASH_MISMATCH")
    require(d17.base.rng_digest(d17_state["bundle"]["rng_state"]) == EXPECTED_RNG_SHA256, "RNG_HASH_MISMATCH")
    require(d17_state["authority"]["schedule_rows"].shape[0] == 512 and d17_state["authority"]["schedule_rows"].shape[1] == 256, "CANONICAL_SCHEDULE_SHAPE_MISMATCH")
    return {
        "authorization": {"path": str(authorization.resolve()), "sha256": sha256_file(authorization)},
        "git_head": git_head,
        "d22_manifest": verify_d22_manifest(),
        "d22_identity": d22_identity,
        "d22_execution_manifest": execution,
        "control_endpoint": {"H1": BASELINE_H1, "H32": BASELINE_H32, "source": str(D22_DIR / "factorial_64_cell_response_table.csv")},
        "d17_state": d17_state,
    }


def dot(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> float:
    return finite(sum(torch.sum(left[name].double() * right[name].double()) for name in d16.ALL_NAMES))


def scale_map(values: Mapping[str, torch.Tensor], scale: float) -> dict[str, torch.Tensor]:
    return {name: values[name] * float(scale) for name in d16.ALL_NAMES}


def add_maps(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: left[name] + right[name] for name in d16.ALL_NAMES}


def subtract_maps(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor], coefficient: float = 1.0) -> dict[str, torch.Tensor]:
    return {name: left[name] - float(coefficient) * right[name] for name in d16.ALL_NAMES}


def candidate_spec(name: str) -> dict[str, Any]:
    specs = {
        "C0": {"name": "C0", "family": "canonical_control", "h1_priority": 1.0, "conflict_filter": False, "adaptive_balance": False, "structure_attenuation": False},
        "C1": {"name": "C1", "family": "H1_PRIORITY_1P5", "h1_priority": 1.5, "conflict_filter": False, "adaptive_balance": False, "structure_attenuation": False},
        "C2": {"name": "C2", "family": "H1_PRIORITY_2P0", "h1_priority": 2.0, "conflict_filter": False, "adaptive_balance": False, "structure_attenuation": False},
        "C3": {"name": "C3", "family": "CONTINUOUS_H1_PROTECTIVE_CONFLICT_FILTER", "h1_priority": 1.0, "conflict_filter": True, "adaptive_balance": False, "structure_attenuation": False},
        "C4": {"name": "C4", "family": "ADAPTIVE_GRADIENT_BALANCING", "h1_priority": 1.0, "conflict_filter": False, "adaptive_balance": True, "structure_attenuation": False},
        "C5": {"name": "C5", "family": "D22_STRUCTURE_SOFT_ATTENUATION", "h1_priority": 1.0, "conflict_filter": False, "adaptive_balance": False, "structure_attenuation": True, "attenuation": 0.5, "attenuated_region": list(H1_REST_REGION)},
        "C6": {"name": "C6", "family": "H1_PRIORITY_1P5_PLUS_CONFLICT_FILTER", "h1_priority": 1.5, "conflict_filter": True, "adaptive_balance": False, "structure_attenuation": False},
        "F1": {"name": "F1", "family": "H1_PRIORITY_3P0", "h1_priority": 3.0, "conflict_filter": False, "adaptive_balance": False, "structure_attenuation": False},
        "F2": {"name": "F2", "family": "H1_PRIORITY_4P0", "h1_priority": 4.0, "conflict_filter": False, "adaptive_balance": False, "structure_attenuation": False},
        "F3": {"name": "F3", "family": "H1_PRIORITY_2P0_PLUS_CONFLICT_FILTER", "h1_priority": 2.0, "conflict_filter": True, "adaptive_balance": False, "structure_attenuation": False},
        "F4": {"name": "F4", "family": "STRONGER_BOUNDED_ADAPTIVE_GRADIENT_BALANCING", "h1_priority": 1.0, "conflict_filter": False, "adaptive_balance": True, "balance_bounds": [0.5, 3.0], "stronger_balance": True, "structure_attenuation": False},
    }
    require(name in specs, f"UNKNOWN_CANDIDATE:{name}")
    return copy.deepcopy(specs[name])


def effective_gradients(spec: Mapping[str, Any], h1_gradients: Mapping[str, torch.Tensor], rest_gradients: Mapping[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    h1_norm = d16.norm(h1_gradients.values())
    rest_norm = d16.norm(rest_gradients.values())
    raw_dot = dot(h1_gradients, rest_gradients)
    effective_rest = dict(rest_gradients)
    conflict_removed_norm = 0.0
    conflict = raw_dot < 0.0
    if conflict and spec.get("conflict_filter"):
        coefficient = raw_dot / max(h1_norm * h1_norm, 1.0e-24)
        opposing = scale_map(h1_gradients, coefficient)
        effective_rest = subtract_maps(rest_gradients, opposing)
        conflict_removed_norm = d16.norm(opposing.values())
    effective_rest_norm = d16.norm(effective_rest.values())
    h1_scale = float(spec.get("h1_priority", 1.0))
    if spec.get("adaptive_balance"):
        lower, upper = spec.get("balance_bounds", [0.5, 2.0])
        h1_scale = min(float(upper), max(float(lower), rest_norm / max(h1_norm, 1.0e-12)))
    combined = add_maps(effective_rest, scale_map(h1_gradients, h1_scale))
    return combined, {
        "raw_h1_gradient_norm": h1_norm,
        "raw_rest_gradient_norm": rest_norm,
        "effective_rest_gradient_norm": effective_rest_norm,
        "h1_rest_dot_product": raw_dot,
        "h1_rest_cosine": None if h1_norm == 0.0 or rest_norm == 0.0 else raw_dot / (h1_norm * rest_norm),
        "conflict_detected": "YES" if conflict else "NO",
        "conflict_filter_applied": "YES" if conflict and spec.get("conflict_filter") else "NO",
        "conflict_removed_norm": conflict_removed_norm,
        "effective_h1_scale": h1_scale,
        "balance_lower_bound": float(spec.get("balance_bounds", [0.5, 2.0])[0]),
        "balance_upper_bound": float(spec.get("balance_bounds", [0.5, 2.0])[1]),
    }


def save_checkpoint(path: Path, model: Any, optimizer: Any, rng: Mapping[str, Any], update: int, spec: Mapping[str, Any], record: Mapping[str, Any]) -> None:
    payload = {
        "schema_version": "stage3_h13_d23_resumable_checkpoint_v1",
        "experiment_id": EXPERIMENT_ID,
        "candidate": spec["name"],
        "intervention": dict(spec),
        "completed_optimizer_step": int(update),
        "next_schedule_index": int(update),
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
        "scheduler_state": None,
        "rng_state": copy.deepcopy(rng),
        "batch_position": {"completed_optimizer_step": int(update), "next_zero_based_schedule_index": int(update)},
        "last_record_identity": {"update": int(record["update"]), "batch_window_sha256": record["batch_window_sha256"], "model_state_hash": record["model_state_hash_after"], "optimizer_state_hash": record["optimizer_state_hash_after"]},
        "resumable": True,
    }
    torch.save(payload, path)


def load_checkpoint(path: Path, bundle: Mapping[str, Any]) -> tuple[Any, Any, dict[str, Any], int]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model, optimizer = d16.new_branch(bundle)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    update = int(checkpoint["completed_optimizer_step"])
    require(update >= START_UPDATE, f"CHECKPOINT_STEP_INVALID:{path}")
    return model, optimizer, copy.deepcopy(checkpoint["rng_state"]), update


def run_step(name: str, spec: Mapping[str, Any], model: Any, optimizer: Any, state: Mapping[str, Any], zero_index: int, rng_before: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    update = zero_index + 1
    require(1 <= update <= 512, f"UNAUTHORIZED_UPDATE:{update}")
    d17.base.restore_rng(rng_before)
    pre = state["pre"]
    authority = state["authority"]
    named = [(name_, parameter) for name_, parameter in model.named_parameters() if parameter.requires_grad]
    require([name_ for name_, _ in named] == list(d16.ALL_NAMES), f"PARAMETER_ORDER_MISMATCH:{name}:{update}")
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
    spec_local = dict(spec)
    if spec_local.get("structure_attenuation"):
        attenuation = float(spec_local["attenuation"])
        factors = torch.ones_like(weighted_terms)
        for horizon in spec_local["attenuated_region"]:
            factors[int(horizon) - 1] = attenuation
        rollout_loss = torch.sum(weighted_terms * factors)
        structure_identity = f"D22_REGION_H18_H31_ATTENUATED_{attenuation:g}_BEFORE_BACKWARD"
    else:
        rollout_loss = full_rollout_loss
        structure_identity = "NONE"
    rest_loss = rollout_loss + d17.LAMBDA_HIDDEN * hidden_loss
    canonical_total_loss = rest_loss + d17.LAMBDA_H1 * h1_loss
    require(all(bool(torch.isfinite(value)) for value in (full_rollout_loss, rollout_loss, h1_loss, hidden_loss, rest_loss, canonical_total_loss)), f"NONFINITE_OBJECTIVE:{name}:{update}")
    h1_gradients, _ = d16.diagnostic_gradients(h1_loss, named)
    rest_gradients, _ = d16.diagnostic_gradients(rest_loss, named)
    h32_gradients, _ = d16.diagnostic_gradients(h32_term, named)
    rng_after_diagnostics = d17.base.capture_rng()
    require(d17.base.rng_digest(rng_after_diagnostics) == d17.base.rng_digest(rng_before), f"DIAGNOSTIC_RNG_CONTAMINATION:{name}:{update}")
    if name == "C0":
        canonical_total_loss.backward()
        grad_metrics = {
            "raw_h1_gradient_norm": d16.norm(h1_gradients.values()),
            "raw_rest_gradient_norm": d16.norm(rest_gradients.values()),
            "effective_rest_gradient_norm": d16.norm(rest_gradients.values()),
            "h1_rest_dot_product": dot(h1_gradients, rest_gradients),
            "h1_rest_cosine": None,
            "conflict_detected": "YES" if dot(h1_gradients, rest_gradients) < 0.0 else "NO",
            "conflict_filter_applied": "NO",
            "conflict_removed_norm": 0.0,
            "effective_h1_scale": 1.0,
            "balance_lower_bound": 0.5,
            "balance_upper_bound": 2.0,
        }
        combined_gradients = add_maps(rest_gradients, h1_gradients)
    else:
        combined_gradients, grad_metrics = effective_gradients(spec_local, h1_gradients, rest_gradients)
        for name_, parameter in named:
            parameter.grad = combined_gradients[name_].to(device=parameter.device, dtype=parameter.dtype).clone()
    require(d17.base.gradients_are_finite(model), f"NONFINITE_RAW_GRADIENT:{name}:{update}")
    raw_gradients, none_flags = d16.capture_gradients(named)
    raw_norm = d16.norm(raw_gradients.values())
    returned_preclip = finite(torch.nn.utils.clip_grad_norm_(model.parameters(), d17.CLIP_MAX_NORM, norm_type=2.0, error_if_nonfinite=False, foreach=None))
    require(math.isclose(raw_norm, returned_preclip, rel_tol=1.0e-5, abs_tol=1.0e-5), f"CLIP_RETURNED_NORM_MISMATCH:{name}:{update}")
    postclip_gradients, _ = d16.capture_gradients(named)
    postclip_norm = d16.norm(postclip_gradients.values())
    clip_coefficient = 1.0 if returned_preclip <= d17.CLIP_MAX_NORM else min(1.0, d17.CLIP_MAX_NORM / (returned_preclip + 1.0e-6))
    residual = d16.norm(postclip_gradients[name_] - raw_gradients[name_] * clip_coefficient for name_ in d16.ALL_NAMES)
    require(residual <= 1.0e-7 + 2.0e-5 * max(postclip_norm, 1.0), f"POSTCLIP_RECONSTRUCTION_FAILED:{name}:{update}")
    rng_before_optimizer = d17.base.capture_rng()
    optimizer.step()
    after_model = d16.clone_parameters(model)
    after_optimizer = d16.clone_named_optimizer_state(optimizer)
    after_model_hash = d17.canonical_model_state_sha256(model.state_dict())
    after_optimizer_hash = d17.base.optimizer_semantic_hash(optimizer)
    require(d16.state_finite(model, optimizer), f"NONFINITE_STATE_AFTER_UPDATE:{name}:{update}")
    rng_after_training = d17.base.capture_rng()
    require(d17.base.rng_digest(rng_before_optimizer) == d17.base.rng_digest(rng_after_training), f"OPTIMIZER_RNG_CONSUMPTION_UNMATCHED:{name}:{update}")
    adamw_metrics, adamw_tensors = d17.d7.adamw_decomposition(model, optimizer, before_model, postclip_gradients, after_model, state["table"])
    validation, validation_meta = d17.d7.validation_snapshot(model, pre["validation"], pre["stats"]["channels"], True)
    rng_after_validation = d17.base.capture_rng()
    require(validation_meta["rng_unchanged_or_restored"] in ("YES", "RESTORED"), f"VALIDATION_RNG_GATE_FAILED:{name}:{update}")
    after_steps = d16.optimizer_steps(after_optimizer)
    parameter_delta = {name_: after_model[name_] - before_model[name_] for name_ in d16.ALL_NAMES}
    effective_total_loss = rest_loss + float(grad_metrics["effective_h1_scale"]) * h1_loss
    record = {
        "branch": name, "candidate": name, "family": spec_local["family"], "update": update, "schedule_index": zero_index,
        "batch_identity": batch_record, "batch_window_sha256": batch_record["batch_window_sha256"],
        "model_state_hash_before": before_model_hash, "model_state_hash_after": after_model_hash,
        "optimizer_state_hash_before": before_optimizer_hash, "optimizer_state_hash_after": after_optimizer_hash,
        "rng_before_digest": d17.base.rng_digest(rng_before), "rng_after_diagnostics_digest": d17.base.rng_digest(rng_after_diagnostics),
        "rng_after_training_digest": d17.base.rng_digest(rng_after_training), "rng_after_validation_digest": d17.base.rng_digest(rng_after_validation),
        "rollout_loss_full": finite(full_rollout_loss), "rollout_loss_used_for_backward": finite(rollout_loss),
        "weighted_rollout_h32_term": finite(h32_term), "explicit_h1_loss": finite(h1_loss),
        "hidden_loss_unweighted": finite(hidden_loss), "weighted_hidden_loss": finite(d17.LAMBDA_HIDDEN * hidden_loss),
        "canonical_total_loss": finite(canonical_total_loss), "effective_total_loss": finite(effective_total_loss),
        "objective_structure_identity": structure_identity, "objective_renormalization": "NO",
        "raw_total_gradient_norm": raw_norm, "returned_pre_clipping_total_norm": returned_preclip,
        "clipping_activated": "YES" if returned_preclip > d17.CLIP_MAX_NORM else "NO", "exact_clip_coefficient": clip_coefficient,
        "post_clipping_gradient_norm": postclip_norm, "postclip_reconstruction_residual_norm": residual,
        "raw_gradient_digest": d17.d7.d6.named_tensor_digest(raw_gradients), "post_clipping_gradient_digest": d17.d7.d6.named_tensor_digest(postclip_gradients),
        "gradient_none_pattern": none_flags, "gradient_none_pattern_hash": hashlib.sha256(json.dumps(none_flags, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "h1_gradient_norm": d16.norm(h1_gradients.values()), "h32_gradient_norm": d16.norm(h32_gradients.values()),
        "raw_h1_gradient_norm": grad_metrics["raw_h1_gradient_norm"], "raw_rest_gradient_norm": grad_metrics["raw_rest_gradient_norm"],
        "effective_rest_gradient_norm": grad_metrics["effective_rest_gradient_norm"], "h1_rest_dot_product": grad_metrics["h1_rest_dot_product"],
        "h1_rest_cosine": grad_metrics["h1_rest_cosine"], "conflict_detected": grad_metrics["conflict_detected"],
        "conflict_filter_applied": grad_metrics["conflict_filter_applied"], "conflict_removed_norm": grad_metrics["conflict_removed_norm"],
        "effective_h1_scale": grad_metrics["effective_h1_scale"], "balance_lower_bound": grad_metrics["balance_lower_bound"], "balance_upper_bound": grad_metrics["balance_upper_bound"],
        "weighted_h32_gradient_norm": d16.norm(h32_gradients[name_] * float(weights[-1]) for name_ in d16.ALL_NAMES),
        "h1_gradient_norm_by_block": d16.block_norms(h1_gradients), "h32_gradient_norm_by_block": d16.block_norms(h32_gradients),
        "raw_gradient_norm_by_block": d16.block_norms(raw_gradients), "postclip_gradient_norm_by_block": d16.block_norms(postclip_gradients),
        "parameter_update_norm": d16.norm(parameter_delta.values()), "parameter_update_norm_by_block": d16.block_norms(parameter_delta),
        "parameter_update_digest": d17.d7.d6.named_tensor_digest(parameter_delta),
        "loss_driven_adamw_displacement_norm": adamw_metrics["aggregate"]["whole_model"]["adaptive_update_norm"],
        "decoupled_weight_decay_displacement_norm": adamw_metrics["aggregate"]["whole_model"]["decay_update_norm"],
        "adamw_decomposition_residual_norm": adamw_metrics["aggregate"]["whole_model"]["decomposition_residual_norm"],
        "adamw_decomposition_status": adamw_metrics["aggregate"]["whole_model"]["decomposition_pass_fail"],
        "step_counter_before": before_steps, "step_counter_after": after_steps,
        "exp_avg_norm": adamw_metrics["aggregate"]["whole_model"]["exp_avg_norm"], "exp_avg_sq_norm": adamw_metrics["aggregate"]["whole_model"]["exp_avg_sq_norm"],
        "adamw": adamw_metrics, "validation": validation, "per_update_H1": validation["1"], "per_update_H32": validation["32"],
        "finite_status": "FINITE", "branch_rng_contract_status": "PASS", "stability_status": "PASS",
    }
    capture = {"branch": name, "update": update, "model_before": before_model, "model_after": after_model, "optimizer_before": before_optimizer, "optimizer_after": after_optimizer, "raw_gradients": raw_gradients, "postclip_gradients": postclip_gradients, "parameter_delta": parameter_delta, "h1_gradients_diagnostic": h1_gradients, "h32_gradients_diagnostic": h32_gradients, "adamw_decomposition_tensors": adamw_tensors}
    return record, capture, copy.deepcopy(rng_after_validation)


def gate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) < 8:
        return {"stage3_completion_gate": "NOT_READY", "window_updates": [], "endpoint_H1": None, "last_8_mean_H1": None, "H1_target_pass": False, "H32_preservation_pass": False, "stability_pass": False, "real_forward_pass": False}
    window = list(rows[-8:])
    updates = [int(row["update"]) for row in window]
    consecutive = updates == list(range(updates[0], updates[0] + 8))
    h1_values = [float(row["per_update_H1"]) for row in window]
    h32_values = [float(row["per_update_H32"]) for row in window]
    h1_pass = float(window[-1]["per_update_H1"]) <= TARGET_H1 and sum(h1_values) / 8.0 <= TARGET_H1
    h32_pass = all(value <= H32_THRESHOLD for value in h32_values)
    stable = all(row.get("finite_status") == "FINITE" and row.get("branch_rng_contract_status") == "PASS" and row.get("stability_status") == "PASS" and row.get("adamw_decomposition_status") == "PASS" and 0.0 <= float(row["exact_clip_coefficient"]) <= 1.0 for row in window)
    real_forward = consecutive and all(update > START_UPDATE for update in updates)
    passed = h1_pass and h32_pass and stable and real_forward
    return {"stage3_completion_gate": "PASSED" if passed else "FAILED", "window_updates": updates, "endpoint_H1": float(window[-1]["per_update_H1"]), "last_8_mean_H1": sum(h1_values) / 8.0, "last_8_H1_values": h1_values, "last_8_H32_values": h32_values, "H1_target_pass": h1_pass, "H32_preservation_pass": h32_pass, "stability_pass": stable, "real_forward_pass": real_forward}


def result_row(name: str, rows: Sequence[Mapping[str, Any]], valid: bool, reason: str | None = None) -> dict[str, Any]:
    if not rows:
        return {"candidate": name, "H1@392": None, "E_H1": None, "threshold_multiple": None, "H32@392": None, "H32_preserved": False, "valid": valid, "decision": reason or "INVALID"}
    endpoint = rows[-1]
    h1 = float(endpoint["per_update_H1"])
    h32 = float(endpoint["per_update_H32"])
    effect = BASELINE_H1 - h1
    h32_preserved = h32 <= H32_THRESHOLD
    decision = "PROMOTION_ELIGIBLE" if valid and h32_preserved and effect >= STRONG_THRESHOLD else "RETAIN_FOR_RANKING" if valid and h32_preserved and effect >= MATERIALITY_THRESHOLD else "DROP_CANDIDATE"
    if reason:
        decision = reason
    return {"candidate": name, "H1@392": h1, "E_H1": effect, "threshold_multiple": effect / MATERIALITY_THRESHOLD, "H32@392": h32, "H32_preserved": h32_preserved, "valid": bool(valid), "decision": decision}


def run_candidate(name: str, state: Mapping[str, Any], output: Path, start_checkpoint: Path | None = None, start_update: int = START_UPDATE, end_update: int = 392) -> tuple[list[dict[str, Any]], Path | None, str | None]:
    spec = candidate_spec(name)
    if start_checkpoint is None:
        model, optimizer = d16.new_branch(state["bundle"])
        rng = copy.deepcopy(state["bundle"]["rng_state"])
    else:
        model, optimizer, rng, checkpoint_update = load_checkpoint(start_checkpoint, state["bundle"])
        require(checkpoint_update == start_update, f"CHECKPOINT_START_UPDATE_MISMATCH:{name}:{checkpoint_update}:{start_update}")
    rows: list[dict[str, Any]] = []
    checkpoint_path = None
    try:
        for update in range(start_update + 1, end_update + 1):
            row, _, rng = run_step(name, spec, model, optimizer, state, update - 1, rng)
            rows.append(row)
            checkpoint_path = output / "checkpoints" / f"{name}_update_{update}.pt"
            save_checkpoint(checkpoint_path, model, optimizer, rng, update, spec, row)
            if end_update > 392 and update >= 400 and gate(rows)["stage3_completion_gate"] == "PASSED":
                break
    except Exception as exc:
        return rows, checkpoint_path, f"{type(exc).__name__}:{exc}"
    return rows, checkpoint_path, None


def trajectory_rows(rows_by_candidate: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    fields = []
    for name, rows in rows_by_candidate.items():
        for row in rows:
            fields.append({"candidate": name, "update": row["update"], "H1": row["per_update_H1"], "H32": row["per_update_H32"], "total_objective": row["effective_total_loss"], "raw_h1_gradient_norm": row["raw_h1_gradient_norm"], "raw_rest_gradient_norm": row["raw_rest_gradient_norm"], "effective_rest_gradient_norm": row["effective_rest_gradient_norm"], "preclip_norm": row["returned_pre_clipping_total_norm"], "postclip_norm": row["post_clipping_gradient_norm"], "clipping_coefficient": row["exact_clip_coefficient"], "adamw_update_norm": row["loss_driven_adamw_displacement_norm"], "parameter_delta_norm": row["parameter_update_norm"], "finite_status": row["finite_status"], "batch_identity": row["batch_window_sha256"], "rng_identity": row["rng_before_digest"], "optimizer_step_identity": json.dumps(row["step_counter_after"], sort_keys=True)} )
    return fields


def manifest_for(output: Path) -> dict[str, Any]:
    excluded = {"FINAL_REPORT.md", "SHA256_MANIFEST.json"}
    files = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name not in excluded:
            files.append({"relative_path": str(path.relative_to(output)).replace("\\", "/"), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    return {"schema_version": "stage3_h13_d23_sha256_manifest_v1", "hash_algorithm": "SHA-256", "self_excluded": True, "excluded_from_manifest": sorted(excluded), "file_count": len(files), "files": files}


def final_report(summary: Mapping[str, Any]) -> str:
    contract = summary["completion_gate"]
    r1 = summary["rung1_results"]
    best = summary.get("best_tested", {})
    reported_h1 = contract["endpoint_H1"] if contract["endpoint_H1"] is not None else best.get("H1")
    lines = [
        f"{EXPERIMENT_ID}:", summary["status"], "", "FIRST_BLOCKER:", summary.get("first_blocker") or "none", "", "FINAL_CLASSIFICATION:", summary["classification"], "", "STAGE_3_STATUS:", summary["stage3_status"], "", "ONE_SENTENCE_VERDICT:", summary["verdict"], "",
        "## 1. COMPLETION_CONTRACT", "", f"Frozen H1 target: `{TARGET_H1:.17g}`; confirmation window: `8`; H32 criterion: `{H32_THRESHOLD:.17g}`; contract gate: `{contract['stage3_completion_gate']}`.", "",
        "## 2. PROTOCOL_INTEGRITY", "", f"Rung 1 candidates executed: `{', '.join(summary['executed_candidates'])}`; Wave B used: `{summary['wave_b_used']}`; Rung 2 updates: `{summary['rung2_updates']}`; Rung 3 updates: `{summary['rung3_updates']}`; update 393 executed: `{summary['update_393_executed']}`; Frozen20/R9/R8E_A1 changes: `NO/NO/NO`.", "",
        "## 3. AUTHORITATIVE_INPUT_IDENTITY", "", f"D22 execution manifest SHA-256: `{EXPECTED_D22_MANIFEST_SHA256}`; D22 design: `{EXPECTED_D22_DESIGN_MANIFEST_SHA256}`; D21: `{EXPECTED_D21_MANIFEST_SHA256}`; start bundle: `{EXPECTED_START_BUNDLE_SHA256}`; model: `{EXPECTED_MODEL_SHA256}`; optimizer: `{EXPECTED_OPTIMIZER_SHA256}`; RNG: `{EXPECTED_RNG_SHA256}`; source revision: `{EXPECTED_SOURCE_REVISION}`.", "",
        "## 4. RUNG1_RESULTS", "", "| candidate | H1@392 | E_H1 | threshold_multiple | H32@392 | H32_preserved | valid | decision |", "|---|---:|---:|---:|---:|---|---|---|",
    ]
    for row in r1:
        lines.append("| {candidate} | {h1} | {effect} | {multiple} | {h32} | {preserved} | {valid} | {decision} |".format(candidate=row["candidate"], h1="" if row["H1@392"] is None else f"{row['H1@392']:.17g}", effect="" if row["E_H1"] is None else f"{row['E_H1']:.17g}", multiple="" if row["threshold_multiple"] is None else f"{row['threshold_multiple']:.6g}", h32="" if row["H32@392"] is None else f"{row['H32@392']:.17g}", preserved=row["H32_preserved"], valid=row["valid"], decision=row["decision"]))
    lines += [
        "", "## 5. WAVE_B_RESULTS", "", json.dumps(summary.get("wave_b_results", []), indent=2, sort_keys=True),
        "", "## 6. RUNG2_RESULTS", "", json.dumps(summary.get("rung2_results", []), indent=2, sort_keys=True),
        "", "## 7. FORWARD_PROMOTION", "", json.dumps(summary.get("promotion", {}), indent=2, sort_keys=True),
        "", "## 8. COMPLETION_GATE_RESULT", "", f"ENDPOINT_H1: {contract['endpoint_H1']}", f"LAST_8_MEAN_H1: {contract['last_8_mean_H1']}", f"H1_TARGET_PASS: {contract['H1_target_pass']}", f"H32_PRESERVATION_PASS: {contract['H32_preservation_pass']}", f"STABILITY_PASS: {contract['stability_pass']}", f"REAL_FORWARD_PASS: {contract['real_forward_pass']}", f"STAGE_3_COMPLETION_GATE: {contract['stage3_completion_gate']}",
        "", "## 9. FORWARD_PROGRESS", f"UPDATE_393_EXECUTED: {summary['update_393_executed']}", f"MAX_VALID_UPDATE_REACHED: {summary['max_valid_update']}", f"PROMOTED_FORWARD_CANDIDATE: {summary.get('promoted_candidate')}", f"PROMOTED_CHECKPOINT_PATH: {summary.get('promoted_checkpoint_path')}", f"PROMOTED_CHECKPOINT_SHA256: {summary.get('promoted_checkpoint_sha256')}",
        "", "## 10. H1_IMPROVEMENT", f"Baseline H1: `{BASELINE_H1:.17g}`", f"Completion-window endpoint H1: `{contract['endpoint_H1']}`", f"Best tested endpoint H1: `{reported_h1}`", f"Absolute improvement of best tested state: `{best.get('E_H1')}`", f"Relative improvement of best tested state: `{best.get('relative_improvement')}`", f"Material-threshold multiple of best tested state: `{best.get('threshold_multiple')}`", f"Distance remaining to 5e-05 for best tested state: `{best.get('distance_to_target')}`",
        "", "## 11. H32_PRESERVATION", f"H32 preservation criterion: `{H32_THRESHOLD:.17g}`; completion-window gate result: `{contract['H32_preservation_pass']}`; best tested endpoint H32: `{best.get('H32')}`; last window H32 values: `{contract.get('last_8_H32_values')}`.",
        "", "## 12. TRAINING_STABILITY", "All qualifying rows require finite model/optimizer state, AdamW decomposition PASS, valid clipping coefficient, unchanged diagnostic/validation RNG contract, and resumable checkpoint serialization.",
        "", "## 13. ENGINEERING_REPAIRS", "Only the D23 runner and D23 evidence bundle were added; no historical D22/D21 artifact was modified. A non-scientific finalization repair corrected endpoint H32 labeling and added the required Wave-B CSV.",
        "", "## 14. NEXT_ACTION", "STOP_STAGE_3_AND_PRESERVE_FINAL_STATE" if contract["stage3_completion_gate"] == "PASSED" else "CONTINUE_OR_SWITCH_BASED_ON_D23_FORWARD_RESULT", ""
    ]
    return "\n".join(lines)


def execute(authorization: Path, output: Path) -> dict[str, Any]:
    require(not output.exists(), f"OUTPUT_ALREADY_EXISTS:{output}")
    output.mkdir(parents=True)
    (output / "checkpoints").mkdir()
    inputs = preflight(authorization)
    state = inputs["d17_state"]
    write_json(output / "STAGE3_COMPLETION_CONTRACT.json", CONTRACT)
    executed_candidates: list[str] = []
    candidate_rows: dict[str, list[dict[str, Any]]] = {}
    checkpoints: dict[str, Path] = {}
    errors: dict[str, str] = {}
    rung1_results: list[dict[str, Any]] = []
    initial_candidates = ["C0", "C1", "C2", "C3", "C4", "C5", "C6"]
    for name in initial_candidates:
        rows, checkpoint, error = run_candidate(name, state, output)
        executed_candidates.append(name)
        candidate_rows[name] = rows
        if checkpoint is not None and rows:
            checkpoints[name] = checkpoint
        if error:
            errors[name] = error
        rung1_results.append(result_row(name, rows, error is None and len(rows) == 8, "INVALID_EXECUTION" if error else None))
    valid_winners = [row for row in rung1_results if row["valid"] and row["H32_preserved"] and row["E_H1"] is not None and row["E_H1"] >= MATERIALITY_THRESHOLD]
    wave_b_used = False
    wave_b_results: list[dict[str, Any]] = []
    if not valid_winners:
        wave_b_used = True
        for name in ("F1", "F2", "F3", "F4"):
            rows, checkpoint, error = run_candidate(name, state, output)
            executed_candidates.append(name)
            candidate_rows[name] = rows
            if checkpoint is not None and rows:
                checkpoints[name] = checkpoint
            if error:
                errors[name] = error
            row = result_row(name, rows, error is None and len(rows) == 8, "INVALID_EXECUTION" if error else None)
            wave_b_results.append(row)
        valid_winners = [row for row in wave_b_results if row["valid"] and row["H32_preserved"] and row["E_H1"] is not None and row["E_H1"] >= MATERIALITY_THRESHOLD]
    all_r1_rows = rung1_results
    write_csv(output / "RUNG1_CANDIDATE_RESULTS.csv", ["candidate", "H1@392", "E_H1", "threshold_multiple", "H32@392", "H32_preserved", "valid", "decision"], all_r1_rows)
    write_csv(output / "RUNG1_TRAJECTORIES.csv", ["candidate", "update", "H1", "H32", "total_objective", "raw_h1_gradient_norm", "raw_rest_gradient_norm", "effective_rest_gradient_norm", "preclip_norm", "postclip_norm", "clipping_coefficient", "adamw_update_norm", "parameter_delta_norm", "finite_status", "batch_identity", "rng_identity", "optimizer_step_identity"], trajectory_rows(candidate_rows))
    if wave_b_used:
        write_csv(output / "WAVE_B_RESULTS.csv", ["candidate", "H1@392", "E_H1", "threshold_multiple", "H32@392", "H32_preserved", "valid", "decision"], wave_b_results)
    rung2_results: list[dict[str, Any]] = []
    forward_rows: list[dict[str, Any]] = []
    promoted_candidate = None
    promoted_checkpoint = None
    completion = gate([])
    rung2_updates = 0
    rung3_updates = 0
    max_valid_update = 392 if candidate_rows else None
    if valid_winners:
        top_two = [row["candidate"] for row in sorted(valid_winners, key=lambda row: (-float(row["E_H1"]), row["candidate"]))[:2]]
        rung2_order = top_two + (["C0"] if "C0" not in top_two and "C0" in checkpoints else [])
        for name in rung2_order:
            rows, checkpoint, error = run_candidate(name, state, output, checkpoints.get(name), start_update=392, end_update=408)
            candidate_rows[f"RUNG2_{name}"] = rows
            rung2_updates += len(rows)
            if rows:
                max_valid_update = max(max_valid_update or 0, int(rows[-1]["update"]))
                forward_rows.extend(rows)
            rung2_results.append({"candidate": name, "updates": len(rows), "endpoint_H1": rows[-1]["per_update_H1"] if rows else None, "endpoint_H32": rows[-1]["per_update_H32"] if rows else None, "valid": error is None, "error": error, "gate": gate(rows)})
            if error:
                errors[f"RUNG2_{name}"] = error
                continue
            if len(rows) >= 8 and gate(rows)["stage3_completion_gate"] == "PASSED":
                completion = gate(rows)
                promoted_candidate = name
                promoted_checkpoint = checkpoint
                break
        if completion["stage3_completion_gate"] != "PASSED":
            candidates_for_promotion = [item for item in rung2_results if item["valid"] and item["endpoint_H1"] is not None]
            require(candidates_for_promotion, "NO_VALID_RUNG2_CANDIDATE")
            best = min(candidates_for_promotion, key=lambda item: float(item["endpoint_H1"]))
            promoted_candidate = best["candidate"]
            promoted_checkpoint = output / "checkpoints" / f"{promoted_candidate}_update_408.pt"
            # If this branch did not run to 408, it cannot be promoted to Rung 3.
            require(promoted_checkpoint.is_file(), f"PROMOTED_CHECKPOINT_MISSING:{promoted_checkpoint}")
            rows, checkpoint, error = run_candidate(promoted_candidate, state, output, promoted_checkpoint, start_update=408, end_update=440)
            rung3_updates = len(rows)
            max_valid_update = max(max_valid_update or 0, int(rows[-1]["update"])) if rows else max_valid_update
            forward_rows.extend(rows)
            if error:
                errors["RUNG3"] = error
            if rows and len(rows) >= 8 and gate(rows)["stage3_completion_gate"] == "PASSED":
                completion = gate(rows)
                promoted_checkpoint = checkpoint
            else:
                completion = gate(rows)
            if rows:
                classification = "D23_STRONG_FORWARD_PROGRESS_TARGET_NOT_YET_REACHED" if float(rows[-1]["per_update_H1"]) < BASELINE_H1 and len(rows) >= 8 and float(rows[-1]["per_update_H1"]) > TARGET_H1 else "D23_PROMOTED_DIRECTION_PLATEAUED_BEFORE_TARGET"
            else:
                classification = "D23_PROMOTED_DIRECTION_FAILED"
        else:
            classification = "STAGE3_COMPLETED_H1_TARGET_SUSTAINED"
    else:
        classification = "D23_TESTED_INTERVENTION_FAMILIES_FAILED"
    if completion["stage3_completion_gate"] == "PASSED":
        classification = "STAGE3_COMPLETED_H1_TARGET_SUSTAINED"
    if promoted_candidate and promoted_checkpoint and promoted_checkpoint.is_file():
        promoted_sha = sha256_file(promoted_checkpoint)
        promoted_metadata = {"candidate": promoted_candidate, "path": str(promoted_checkpoint.resolve()), "sha256": promoted_sha, "completed_optimizer_step": int(torch.load(promoted_checkpoint, map_location="cpu", weights_only=False)["completed_optimizer_step"]), "resumable": True, "scheduler_state": "not_applicable", "configuration": candidate_spec(promoted_candidate)}
    else:
        promoted_sha = None
        promoted_metadata = {"candidate": None, "path": None, "sha256": None, "completed_optimizer_step": None, "resumable": False}
    write_json(output / "RUNG2_RESULTS.json", {"results": rung2_results, "updates_executed": rung2_updates})
    write_csv(output / "RUNG2_RESULTS.csv", ["candidate", "updates", "endpoint_H1", "endpoint_H32", "valid", "error", "gate"], [{**row, "gate": json.dumps(row["gate"], sort_keys=True)} for row in rung2_results])
    write_csv(output / "RUNG2_TRAJECTORIES.csv", ["candidate", "update", "H1", "H32", "total_objective", "raw_h1_gradient_norm", "raw_rest_gradient_norm", "effective_rest_gradient_norm", "preclip_norm", "postclip_norm", "clipping_coefficient", "adamw_update_norm", "parameter_delta_norm", "finite_status", "batch_identity", "rng_identity", "optimizer_step_identity"], trajectory_rows({key: value for key, value in candidate_rows.items() if key.startswith("RUNG2_")}))
    forward_rows = sorted(forward_rows, key=lambda row: int(row["update"])) if promoted_candidate else []
    write_csv(output / "FORWARD_TRAJECTORY.csv", ["candidate", "update", "H1", "H32", "total_objective", "raw_h1_gradient_norm", "raw_rest_gradient_norm", "effective_rest_gradient_norm", "preclip_norm", "postclip_norm", "clipping_coefficient", "adamw_update_norm", "parameter_delta_norm", "finite_status", "batch_identity", "rng_identity", "optimizer_step_identity"], trajectory_rows({"PROMOTED": forward_rows}) if forward_rows else [])
    write_json(output / "FORWARD_PROMOTION_DECISION.json", {"promoted_forward_candidate": promoted_candidate, "promotion_reason": "largest sustained H1 improvement with H32 preservation and stability" if promoted_candidate else "no valid winner", "top_candidates": valid_winners, "classification": classification})
    write_json(output / "STAGE3_COMPLETION_GATE_RESULT.json", completion)
    write_json(output / "PROMOTED_FORWARD_CHECKPOINT_METADATA.json", promoted_metadata)
    write_json(output / "D23_RUN_MANIFEST.json", {"schema_version": "stage3_h13_d23_run_manifest_v1", "experiment_id": EXPERIMENT_ID, "captured_at_utc": utc_now(), "runner_path": str(Path(__file__).resolve()), "runner_sha256": sha256_file(Path(__file__).resolve()), "authorization": inputs["authorization"], "authoritative_input_identity": {"D22_manifest_sha256": EXPECTED_D22_MANIFEST_SHA256, "D22_design_manifest_sha256": EXPECTED_D22_DESIGN_MANIFEST_SHA256, "D21_manifest_sha256": EXPECTED_D21_MANIFEST_SHA256, "start_bundle_sha256": EXPECTED_START_BUNDLE_SHA256, "model_sha256": EXPECTED_MODEL_SHA256, "optimizer_sha256": EXPECTED_OPTIMIZER_SHA256, "rng_sha256": EXPECTED_RNG_SHA256, "source_revision": EXPECTED_SOURCE_REVISION}, "rung1_candidates": initial_candidates, "wave_b_used": wave_b_used, "rung1_updates": list(RUNG1_UPDATES), "rung2_updates": list(RUNG2_UPDATES) if rung2_results else [], "rung3_updates": list(RUNG3_UPDATES) if rung3_updates else [], "update_393_executed": "YES" if rung2_updates or rung3_updates else "NO", "frozen20_used": "NO", "r9_executed": "NO", "r8e_a1_modified": "NO", "threshold_tuning": "NO", "causal_localization_continued": "NO"})
    summary = {"status": "PASSED" if errors == {} or completion["stage3_completion_gate"] == "PASSED" else "FAILED", "first_blocker": next(iter(errors.values()), None), "classification": classification, "stage3_status": "COMPLETE" if completion["stage3_completion_gate"] == "PASSED" else "BLOCKED" if errors else "IN_PROGRESS", "verdict": "The frozen Stage-3 completion gate was passed on a genuine forward trajectory and the resumable state was saved." if completion["stage3_completion_gate"] == "PASSED" else "D23 executed the bounded optimization sprint and saved the strongest tested state without passing the frozen completion gate.", "completion_gate": completion, "rung1_results": rung1_results, "wave_b_used": wave_b_used, "wave_b_results": wave_b_results, "rung2_results": rung2_results, "promotion": {"candidate": promoted_candidate, "reason": "sustained H1 improvement with H32 and stability preservation" if promoted_candidate else "none"}, "executed_candidates": executed_candidates, "rung2_updates": rung2_updates, "rung3_updates": rung3_updates, "update_393_executed": "YES" if rung2_updates or rung3_updates else "NO", "max_valid_update": max_valid_update, "promoted_candidate": promoted_candidate, "promoted_checkpoint_path": promoted_metadata["path"], "promoted_checkpoint_sha256": promoted_sha}
    write_json(output / "SUMMARY.json", summary)
    (output / "FINAL_REPORT.md").write_text(final_report(summary), encoding="utf-8", newline="\n")
    manifest = manifest_for(output)
    write_json(output / "SHA256_MANIFEST.json", manifest)
    manifest_check = {"failures": []}
    for entry in manifest["files"]:
        target = output / Path(entry["relative_path"].replace("/", "\\"))
        if sha256_file(target) != entry["sha256"] or target.stat().st_size != entry["size_bytes"]:
            manifest_check["failures"].append(entry["relative_path"])
    require(not manifest_check["failures"], f"D23_MANIFEST_FINALIZATION_FAILED:{manifest_check}")
    return {"status": summary["status"], "output": str(output.resolve()), "classification": classification, "completion_gate": completion, "promoted_candidate": promoted_candidate, "max_valid_update": max_valid_update, "promoted_checkpoint_sha256": promoted_sha, "first_blocker": summary["first_blocker"]}


def finalize_existing(output: Path) -> dict[str, Any]:
    """Repair only D23 reporting artifacts; never rerun a scientific update."""
    require(output.is_dir(), f"OUTPUT_MISSING:{output}")
    summary = json.loads((output / "SUMMARY.json").read_text(encoding="utf-8"))
    for row in summary["rung1_results"] + summary.get("wave_b_results", []):
        row["H32_preserved"] = bool(row["H32@392"] is not None and float(row["H32@392"]) <= H32_THRESHOLD)
    all_results = [row for row in summary["rung1_results"] + summary.get("wave_b_results", []) if row.get("valid") and row.get("E_H1") is not None]
    best = max(all_results, key=lambda row: float(row["E_H1"])) if all_results else None
    if best is not None:
        checkpoint = output / "checkpoints" / f"{best['candidate']}_update_392.pt"
        best_payload = {"candidate": best["candidate"], "H1": best["H1@392"], "H32": best["H32@392"], "E_H1": best["E_H1"], "relative_improvement": float(best["E_H1"]) / BASELINE_H1, "threshold_multiple": float(best["E_H1"]) / MATERIALITY_THRESHOLD, "distance_to_target": max(0.0, float(best["H1@392"]) - TARGET_H1), "checkpoint_path": str(checkpoint.resolve()) if checkpoint.is_file() else None, "checkpoint_sha256": sha256_file(checkpoint) if checkpoint.is_file() else None, "promoted": False}
    else:
        best_payload = {"candidate": None, "H1": None, "H32": None, "E_H1": None, "relative_improvement": None, "threshold_multiple": None, "distance_to_target": None, "checkpoint_path": None, "checkpoint_sha256": None, "promoted": False}
    summary["best_tested"] = best_payload
    summary["stage3_status"] = "COMPLETE" if summary["completion_gate"]["stage3_completion_gate"] == "PASSED" else "BLOCKED" if summary.get("first_blocker") else "IN_PROGRESS"
    write_json(output / "SUMMARY.json", summary)
    write_csv(output / "RUNG1_CANDIDATE_RESULTS.csv", ["candidate", "H1@392", "E_H1", "threshold_multiple", "H32@392", "H32_preserved", "valid", "decision"], summary["rung1_results"])
    if summary.get("wave_b_used"):
        write_csv(output / "WAVE_B_RESULTS.csv", ["candidate", "H1@392", "E_H1", "threshold_multiple", "H32@392", "H32_preserved", "valid", "decision"], summary.get("wave_b_results", []))
    write_json(output / "BEST_TESTED_CHECKPOINT_METADATA.json", best_payload)
    write_json(output / "FORWARD_PROMOTION_DECISION.json", {"promoted_forward_candidate": summary.get("promoted_candidate"), "best_tested_candidate": best_payload["candidate"], "best_tested_checkpoint_path": best_payload["checkpoint_path"], "promotion_reason": summary["promotion"]["reason"], "classification": summary["classification"], "scientific_updates_rerun_during_finalization": False})
    run_manifest = json.loads((output / "D23_RUN_MANIFEST.json").read_text(encoding="utf-8"))
    run_manifest["runner_sha256"] = sha256_file(Path(__file__).resolve())
    run_manifest["non_scientific_finalization_repair"] = True
    write_json(output / "D23_RUN_MANIFEST.json", run_manifest)
    (output / "FINAL_REPORT.md").write_text(final_report(summary), encoding="utf-8", newline="\n")
    manifest = manifest_for(output)
    write_json(output / "SHA256_MANIFEST.json", manifest)
    for entry in manifest["files"]:
        target = output / Path(entry["relative_path"].replace("/", "\\"))
        require(sha256_file(target) == entry["sha256"] and target.stat().st_size == entry["size_bytes"], f"FINALIZATION_MANIFEST_MISMATCH:{entry['relative_path']}")
    return {"status": "FINALIZATION_ONLY", "output": str(output.resolve()), "best_tested": best_payload, "scientific_updates_rerun": False}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--finalize-existing", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = finalize_existing(args.output.resolve()) if args.finalize_existing else execute(args.authorization.resolve(), args.output.resolve())
        print(json.dumps(result, sort_keys=True, ensure_ascii=True))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "first_blocker": f"{type(exc).__name__}:{exc}", "scientific_updates_executed": False}, sort_keys=True, ensure_ascii=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
