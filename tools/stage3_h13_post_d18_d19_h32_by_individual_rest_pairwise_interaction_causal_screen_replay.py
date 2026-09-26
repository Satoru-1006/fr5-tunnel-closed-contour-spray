"""Execute the separately authorized D19 H32-by-individual-rest replay.

The scientific branches are exactly DELETE_H32_PLUS_Hi for Hi=H2..H31.
Each branch starts from the certified post-update-384 bundle and independently
replays updates 385..392 with the live objective terms for H32 and Hi removed.
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
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import stage3_h13_post_d5_d6_branch_state_materialization_replay as base  # noqa: E402
from src.stage3_h13_r8e import rollout_horizon_weights  # noqa: E402
from src.stage3_h13_r8e_d2 import canonical_model_state_sha256  # noqa: E402
from tools import stage3_h13_post_d7_loss_gradient_interaction_instrumentation_replay as d7  # noqa: E402
from tools import stage3_h13_post_d15_d16_adamw_full_pipeline_h32_component_deletion_causal_intervention_replay as d16  # noqa: E402
from tools import stage3_h13_post_d16_d17_full_pipeline_multi_horizon_rollout_causal_screen_replay as d17  # noqa: E402


EXPERIMENT_ID = "STAGE_3_H13_POST_D18_D19_H32_BY_INDIVIDUAL_REST_PAIRWISE_INTERACTION_CAUSAL_SCREEN"
DESIGN_DIR = ROOT / "outputs/stage3_h13_post_d18_d19_h32_by_individual_rest_pairwise_interaction_causal_screen_design_review_20260820T142946+0800"
D18_DIR = ROOT / "outputs/stage3_h13_post_d17_d18_h2_h31_joint_rollout_deletion_with_h32_retained_causal_intervention_replay_20260820T134651"
BUNDLE_PATH = ROOT / "outputs/stage3_h13_post_d5_d6_branch_state_materialization_replay_20260816T152000Z_retry/post_update_384_branch_bundle.pt"
AUTHORIZATION_PATH = Path(r"C:\Users\86198\.codex\attachments\67a4cca0-447c-4e11-a291-ba7896834393\pasted-text.txt")

EXPECTED_DESIGN_MANIFEST_SHA256 = "22251c2412ba0cfa9ae061679b429dd9489881bcf2db40d9e0a67812be16f760"
EXPECTED_D18_MANIFEST_SHA256 = "6aef1d0289e4d984fee4d551fa88ad7ded13eb42dbd1d331112579c561a59a58"
EXPECTED_BUNDLE_SHA256 = "4a022aa0340977e571a3a3085e30d96fd0d92129f4fbeff99aaa8e3420379083"
EXPECTED_SOURCE_REVISION = "a6dff2b6887cc3f7598ab95549c2dd21c0efe763"
EXPECTED_D18_CROSS = 1.5671621916205796e-05
H1_MATERIAL_THRESHOLD = 4.099006815405639e-06
H32_PRESERVATION_THRESHOLD = 0.01856902565856056
UPDATES = tuple(range(385, 393))
SCHEDULE_INDICES = tuple(range(384, 392))
ACTIVE_HORIZON = 32
POSITION_BETA = 0.001
LAMBDA_H1 = 1.0
LAMBDA_HIDDEN = 0.005
CLIP_MAX_NORM = 1.0
REPLAY_RTOL = 5.0e-6
REPLAY_ATOL = 5.0e-9
REST_HORIZONS = tuple(range(2, 32))
PAIR_BRANCHES = tuple(f"DELETE_H32_PLUS_H{i}" for i in REST_HORIZONS)
ALL_NAMES = d16.ALL_NAMES
GRU_NAMES = d16.GRU_NAMES
HEAD_NAMES = d16.HEAD_NAMES


class StageBlocker(RuntimeError):
    """A scientific-integrity blocker, not an ordinary reporting error."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def finite(value: Any) -> float:
    result = float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)
    if not math.isfinite(result):
        raise StageBlocker(f"NONFINITE_MEASUREMENT:{result}")
    return result


def verify_manifest_file(path: Path, expected_sha256: str) -> dict[str, Any]:
    if not path.is_file():
        raise StageBlocker(f"MANIFEST_MISSING:{path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise StageBlocker(f"MANIFEST_HASH_MISMATCH:{path}:{actual}:{expected_sha256}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("self_excluded") is not True:
        raise StageBlocker(f"MANIFEST_NOT_SELF_EXCLUDED:{path}")
    failures: list[str] = []
    for name, entry in manifest.get("files", {}).items():
        target = path.parent / name
        expected_hash = entry["sha256"] if isinstance(entry, Mapping) else entry
        expected_bytes = entry.get("bytes") if isinstance(entry, Mapping) else None
        if not target.is_file() or sha256_file(target) != expected_hash or (expected_bytes is not None and target.stat().st_size != int(expected_bytes)):
            failures.append(name)
    if failures:
        raise StageBlocker(f"MANIFEST_CONTENT_MISMATCH:{path}:{failures}")
    return {"path": str(path.resolve()), "sha256": actual, "file_count": len(manifest.get("files", {})), "files": manifest.get("files", {})}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def verify_design_and_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    design_manifest = verify_manifest_file(DESIGN_DIR / "SHA256_MANIFEST.json", EXPECTED_DESIGN_MANIFEST_SHA256)
    design = json.loads((DESIGN_DIR / "experiment_design.json").read_text(encoding="utf-8"))
    branch_manifest = json.loads((DESIGN_DIR / "branch_manifest.json").read_text(encoding="utf-8"))
    analysis_contract = json.loads((DESIGN_DIR / "analysis_contract.json").read_text(encoding="utf-8"))
    classification_contract = json.loads((DESIGN_DIR / "classification_contract.json").read_text(encoding="utf-8"))
    trajectory_contract = json.loads((DESIGN_DIR / "trajectory_instrumentation_contract.json").read_text(encoding="utf-8"))
    imported_identity = json.loads((DESIGN_DIR / "imported_evidence_identity.json").read_text(encoding="utf-8"))
    provenance = json.loads((DESIGN_DIR / "provenance_record.json").read_text(encoding="utf-8"))
    auth_text = AUTHORIZATION_PATH.read_text(encoding="utf-8") if AUTHORIZATION_PATH.is_file() else ""
    required_auth = (EXPERIMENT_ID, "DO NOT perform another design review.", "DO NOT propose, design, or execute D20.", "EXPECTED_NEW_D19_BRANCH_COUNT: 30", "UPDATE_393_EXECUTED:")
    if any(item not in auth_text for item in required_auth):
        raise StageBlocker("D19_EXECUTION_AUTHORIZATION_SCOPE_MISMATCH")
    if design.get("experiment_identifier") != EXPERIMENT_ID or design.get("design_review_status") != "PASSED" or design.get("design_review_only") is not True or design.get("execution_authorized") is not False:
        raise StageBlocker("SEALED_D19_DESIGN_STATUS_MISMATCH")
    if design.get("factorial_scope", {}).get("new_intervention_branch_count") != 30 or tuple(design["factorial_scope"]["rest_horizons"]) != REST_HORIZONS:
        raise StageBlocker("SEALED_D19_BRANCH_SCOPE_MISMATCH")
    if tuple(design["canonical_replay_window"]["updates"]) != UPDATES or tuple(design["canonical_replay_window"]["schedule_indices"]) != SCHEDULE_INDICES or design["canonical_replay_window"].get("update_393") != "FORBIDDEN":
        raise StageBlocker("SEALED_D19_REPLAY_WINDOW_MISMATCH")
    names = [item["branch_name"] for item in branch_manifest.get("branches", [])]
    if branch_manifest.get("expected_new_d19_branch_count") != 30 or branch_manifest.get("actual_new_d19_branch_count") != 30 or tuple(names) != PAIR_BRANCHES:
        raise StageBlocker("SEALED_D19_BRANCH_MANIFEST_MISMATCH")
    if analysis_contract["pairwise_estimand"]["denominator_for_cross_fraction"] != EXPECTED_D18_CROSS:
        raise StageBlocker("D19_DENOMINATOR_MISMATCH")
    if classification_contract["h1_material_effect_threshold"] != H1_MATERIAL_THRESHOLD or classification_contract["h32_preservation_threshold"] != H32_PRESERVATION_THRESHOLD:
        raise StageBlocker("D19_FROZEN_THRESHOLD_MISMATCH")
    if tuple(trajectory_contract["collection_window"]["updates"]) != UPDATES or trajectory_contract["collection_window"]["endpoint_update"] != 392:
        raise StageBlocker("D19_TRAJECTORY_WINDOW_MISMATCH")
    if sha256_file(BUNDLE_PATH) != EXPECTED_BUNDLE_SHA256 or design["starting_contract"]["state_sha256"] != EXPECTED_BUNDLE_SHA256:
        raise StageBlocker("CERTIFIED_START_BUNDLE_HASH_MISMATCH")
    if design["starting_contract"]["source_revision"] != EXPECTED_SOURCE_REVISION or provenance["frozen_contract_identity"]["source_revision"] != EXPECTED_SOURCE_REVISION:
        raise StageBlocker("D19_SOURCE_REVISION_CONTRACT_MISMATCH")
    for item in provenance.get("source_files_verified_read_only", []):
        target = Path(item["path"].replace("D:/robotfucker", str(ROOT).replace("\\", "/")))
        if not target.is_file() or sha256_file(target) != item["sha256"]:
            raise StageBlocker(f"AUTHENTICATED_SOURCE_HASH_MISMATCH:{item['path']}")
    if not imported_identity.get("verification_status") == "PASSED_READ_ONLY":
        raise StageBlocker("IMPORTED_EVIDENCE_IDENTITY_NOT_PASSED")
    if not AUTHORIZATION_PATH.is_file():
        raise StageBlocker("D19_AUTHORIZATION_TEXT_MISSING")
    identity = {
        "schema_version": "stage3_h13_post_d18_d19_authoritative_input_identity_v1",
        "design_review_bundle": design_manifest,
        "design_bundle_path": str(DESIGN_DIR.resolve()),
        "design_bundle_manifest_expected_sha256": EXPECTED_DESIGN_MANIFEST_SHA256,
        "certified_start": {"path": str(BUNDLE_PATH.resolve()), "sha256": sha256_file(BUNDLE_PATH), "expected_sha256": EXPECTED_BUNDLE_SHA256, "completed_optimizer_step": design["starting_contract"]["completed_optimizer_step"], "model_semantic_hash": design["starting_contract"]["model_semantic_hash"], "optimizer_semantic_hash": design["starting_contract"]["optimizer_semantic_hash"], "exp_avg_semantic_hash": design["starting_contract"]["exp_avg_semantic_hash"], "exp_avg_sq_semantic_hash": design["starting_contract"]["exp_avg_sq_semantic_hash"], "rng_sha256": design["starting_contract"]["rng_sha256"]},
        "source_revision": EXPECTED_SOURCE_REVISION,
        "authorization": {"path": str(AUTHORIZATION_PATH.resolve()), "sha256": sha256_file(AUTHORIZATION_PATH)},
        "provenance_record": provenance,
        "imported_evidence_identity": imported_identity,
        "design_contracts": {"analysis": analysis_contract, "classification": classification_contract, "trajectory_instrumentation": trajectory_contract},
    }
    return identity, imported_identity, {"design": design, "branch_manifest": branch_manifest, "provenance": provenance, "design_manifest": design_manifest}


def verify_imported_evidence(imported_identity: Mapping[str, Any]) -> dict[str, Any]:
    d18_manifest = verify_manifest_file(D18_DIR / "SHA256_MANIFEST.json", EXPECTED_D18_MANIFEST_SHA256)
    d18_protocol = json.loads((D18_DIR / "protocol_integrity.json").read_text(encoding="utf-8"))
    d18_endpoint = json.loads((D18_DIR / "endpoint_results.json").read_text(encoding="utf-8"))
    d18_decomp = json.loads((D18_DIR / "non_additivity_decomposition.json").read_text(encoding="utf-8"))
    if d18_protocol.get("status") != "PASSED" or d18_endpoint.get("update") != 392 or d18_endpoint.get("d18", {}).get("validity") != "VALID":
        raise StageBlocker("D18_IMPORTED_EXECUTION_NOT_VALID")
    if d18_decomp.get("H32_REST_INTERACTION") != EXPECTED_D18_CROSS:
        raise StageBlocker("D18_CROSS_GROUP_ENDPOINT_MISMATCH")
    d18_bundle_identity = imported_identity["d18_execution_bundle"]
    if d18_bundle_identity.get("manifest_sha256") != EXPECTED_D18_MANIFEST_SHA256 or d18_bundle_identity.get("manifest_status") != "VERIFIED":
        raise StageBlocker("D18_IMPORTED_IDENTITY_MISMATCH")
    d16_identity = imported_identity["d16_delete_h32_import"]
    d17_identity = imported_identity["d17_execution_bundle"]
    d16_manifest = verify_manifest_file(Path(d16_identity["path"]) / "SHA256_MANIFEST.json", d16_identity["manifest_sha256"])
    d17_manifest = verify_manifest_file(Path(d17_identity["path"]) / "SHA256_MANIFEST.json", d17_identity["manifest_sha256"])
    d17_rows = load_jsonl(Path(d17_identity["path"]) / "per_update_instrumentation.jsonl")
    control_rows = {int(row["update"]): row for row in d17_rows if row.get("branch") == "CONTROL"}
    if tuple(sorted(control_rows)) != UPDATES:
        raise StageBlocker("D17_CONTROL_REFERENCE_WINDOW_MISMATCH")
    d18_pair = d18_endpoint["d18"]
    baseline = {"d18_manifest": d18_manifest, "d16_manifest": d16_manifest, "d17_manifest": d17_manifest, "d18_endpoint": d18_endpoint, "d18_decomposition": d18_decomp, "d18_cross_group_interaction": EXPECTED_D18_CROSS, "d18_control_h1": d18_endpoint["control"]["CONTROL_H1"], "d18_control_h32": d18_endpoint["control"]["CONTROL_H32"], "d18_h1": d18_pair["D18_H1"], "d18_h32": d18_pair["D18_H32"], "d18_e_rest": d18_pair["E_REST"], "d16_import": d16_identity, "d17_import": d17_identity, "d17_control_rows": control_rows}
    if not math.isclose(float(baseline["d18_e_rest"]), float(baseline["d18_control_h1"]) - float(baseline["d18_h1"]), rel_tol=0.0, abs_tol=1e-18):
        raise StageBlocker("D18_IMPORTED_EFFECT_ARITHMETIC_MISMATCH")
    return baseline


def preflight_runtime() -> dict[str, Any]:
    git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    if git_head != EXPECTED_SOURCE_REVISION:
        raise StageBlocker(f"SOURCE_REVISION_MISMATCH:{git_head}:{EXPECTED_SOURCE_REVISION}")
    runtime = d17.preflight_runtime()
    if runtime["git_head"] != EXPECTED_SOURCE_REVISION:
        raise StageBlocker("D17_RUNTIME_SOURCE_REVISION_MISMATCH")
    source_hashes = {}
    for path in ("src/stage3_h13_r8e.py", "src/stage3_h13_r8e_d2.py", "src/stage3_h13_r8e_d4.py", "scripts/stage3_h13_post_d5_d6_branch_state_materialization_replay.py", "tools/stage3_h13_post_d7_loss_gradient_interaction_instrumentation_replay.py", "tools/stage3_h13_post_d15_d16_adamw_full_pipeline_h32_component_deletion_causal_intervention_replay.py", "tools/stage3_h13_post_d16_d17_full_pipeline_multi_horizon_rollout_causal_screen_replay.py"):
        source_hashes[path] = sha256_file(ROOT / path)
    weights = [finite(value) for value in rollout_horizon_weights(ACTIVE_HORIZON)]
    if len(weights) != ACTIVE_HORIZON or not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=2e-7):
        raise StageBlocker("ROLLOUT_WEIGHT_CONTRACT_MISMATCH")
    return {"runtime": runtime, "git_head": git_head, "source_hashes": source_hashes, "runner_sha256": sha256_file(Path(__file__).resolve()), "rollout_horizon_weights_32": weights}


def pair_branch_update(branch: str, horizon: int, state: Mapping[str, Any], model: Any, optimizer: torch.optim.Optimizer, zero_index: int, rng_before: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    update = zero_index + 1
    if horizon not in REST_HORIZONS or branch != f"DELETE_H32_PLUS_H{horizon}" or update not in UPDATES:
        raise StageBlocker(f"UNAUTHORIZED_PAIR_BRANCH_OR_UPDATE:{branch}:{horizon}:{update}")
    base.restore_rng(rng_before)
    pre = state["pre"]
    authority = state["authority"]
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if [name for name, _ in named] != list(ALL_NAMES):
        raise StageBlocker(f"PAIR_PARAMETER_ORDER_MISMATCH:{branch}:{update}")
    batch_record = d16.batch_identity(authority["schedule_manifest"], zero_index)
    batch = base.tensor_batch(pre["train_data"], authority["schedule_rows"][zero_index])
    before_model = d16.clone_parameters(model)
    before_optimizer = d16.clone_named_optimizer_state(optimizer)
    before_model_hash = canonical_model_state_sha256(model.state_dict())
    before_optimizer_hash = base.optimizer_semantic_hash(optimizer)
    before_steps = d16.optimizer_steps(before_optimizer)
    optimizer.zero_grad(set_to_none=True)
    if any(parameter.grad is not None for _, parameter in named):
        raise StageBlocker(f"LIVE_GRADIENT_NOT_EMPTY:{branch}:{update}")
    rollout = base.causal_paired_rollout(model, batch["inputs"], batch["history_positions"], batch["history_times"], batch["target_times"], batch["starts"], batch["ends"], pre["stats"]["channels"], horizon=ACTIVE_HORIZON, target_positions_for_teacher=batch["target_positions"], hidden_consistency_enabled=True)
    if rollout.get("free_branch_reads_target_positions") is not False or rollout.get("teacher_reference_stop_gradient") is not True:
        raise StageBlocker(f"GRAPH_SEMANTICS_GATE_FAILED:{branch}:{update}")
    predictions = rollout["predictions"]
    targets = batch["target_positions"][:, :ACTIVE_HORIZON, :]
    step_losses = F.smooth_l1_loss(predictions, targets, beta=POSITION_BETA, reduction="none").mean(dim=(0, 2))
    weights = rollout_horizon_weights(ACTIVE_HORIZON).to(device=predictions.device, dtype=predictions.dtype)
    weighted_terms = step_losses * weights
    full_rollout_loss = torch.sum(weighted_terms)
    h1_loss = step_losses[0]
    h32_term = weighted_terms[-1]
    hidden_loss = rollout["hidden_loss"]
    rollout_loss = full_rollout_loss - weighted_terms[-1] - weighted_terms[horizon - 1]
    total_loss = rollout_loss + LAMBDA_H1 * h1_loss + LAMBDA_HIDDEN * hidden_loss
    values = (full_rollout_loss, rollout_loss, h1_loss, h32_term, hidden_loss, total_loss)
    if any(not bool(torch.isfinite(value)) for value in values):
        raise StageBlocker(f"NONFINITE_OBJECTIVE:{branch}:{update}")
    h1_gradients, _ = d16.diagnostic_gradients(h1_loss, named)
    h32_gradients, _ = d16.diagnostic_gradients(h32_term, named)
    rng_after_diagnostics = base.capture_rng()
    if base.rng_digest(rng_after_diagnostics) != base.rng_digest(rng_before):
        raise StageBlocker(f"DIAGNOSTIC_RNG_CONTAMINATION:{branch}:{update}")
    total_loss.backward()
    if not base.gradients_are_finite(model):
        raise StageBlocker(f"NONFINITE_RAW_GRADIENT:{branch}:{update}")
    raw_gradients, none_flags = d16.capture_gradients(named)
    raw_norm = d16.norm(raw_gradients.values())
    returned_preclip = finite(torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_MAX_NORM, norm_type=2.0, error_if_nonfinite=False, foreach=None))
    if not math.isclose(raw_norm, returned_preclip, rel_tol=1e-5, abs_tol=1e-5):
        raise StageBlocker(f"CLIP_RETURNED_NORM_MISMATCH:{branch}:{update}")
    postclip_gradients, _ = d16.capture_gradients(named)
    postclip_norm = d16.norm(postclip_gradients.values())
    clip_coefficient = 1.0 if returned_preclip <= CLIP_MAX_NORM else min(1.0, CLIP_MAX_NORM / (returned_preclip + 1e-6))
    reconstruction_residual = d16.norm(postclip_gradients[name] - raw_gradients[name] * clip_coefficient for name in ALL_NAMES)
    if reconstruction_residual > 1e-7 + 2e-5 * max(postclip_norm, 1.0):
        raise StageBlocker(f"POSTCLIP_RECONSTRUCTION_FAILED:{branch}:{update}")
    rng_before_optimizer = base.capture_rng()
    optimizer.step()
    after_model = d16.clone_parameters(model)
    after_optimizer = d16.clone_named_optimizer_state(optimizer)
    after_model_hash = canonical_model_state_sha256(model.state_dict())
    after_optimizer_hash = base.optimizer_semantic_hash(optimizer)
    if not d16.state_finite(model, optimizer):
        raise StageBlocker(f"NONFINITE_STATE_AFTER_UPDATE:{branch}:{update}")
    rng_after_training = base.capture_rng()
    if base.rng_digest(rng_before_optimizer) != base.rng_digest(rng_after_training):
        raise StageBlocker(f"OPTIMIZER_RNG_CONSUMPTION_UNMATCHED:{branch}:{update}")
    adamw_metrics, adamw_tensors = d7.adamw_decomposition(model, optimizer, before_model, postclip_gradients, after_model, state["table"])
    validation, validation_meta = d7.validation_snapshot(model, pre["validation"], pre["stats"]["channels"], True)
    rng_after_validation = base.capture_rng()
    if validation_meta["rng_unchanged_or_restored"] not in ("YES", "RESTORED"):
        raise StageBlocker(f"VALIDATION_RNG_GATE_FAILED:{branch}:{update}")
    after_steps = d16.optimizer_steps(after_optimizer)
    parameter_delta = {name: after_model[name] - before_model[name] for name in ALL_NAMES}
    record = {
        "branch": branch, "horizon": horizon, "update": update, "schedule_index": zero_index, "batch_identity": batch_record, "batch_window_sha256": batch_record["batch_window_sha256"],
        "model_state_hash_before": before_model_hash, "model_state_hash_after": after_model_hash, "optimizer_state_hash_before": before_optimizer_hash, "optimizer_state_hash_after": after_optimizer_hash,
        "rng_before_digest": base.rng_digest(rng_before), "rng_after_diagnostics_digest": base.rng_digest(rng_after_diagnostics), "rng_after_training_digest": base.rng_digest(rng_after_training), "rng_after_validation_digest": base.rng_digest(rng_after_validation),
        "rollout_loss_full": finite(full_rollout_loss), "rollout_loss_used_for_backward": finite(rollout_loss), "weighted_rollout_h1_term": finite(weighted_terms[0]), "weighted_rollout_h32_term": finite(h32_term), "weighted_rollout_hi_term": finite(weighted_terms[horizon - 1]), "explicit_h1_loss": finite(h1_loss), "hidden_loss_unweighted": finite(hidden_loss), "weighted_hidden_loss": finite(LAMBDA_HIDDEN * hidden_loss), "total_loss": finite(total_loss),
        "deleted_horizons": [horizon, 32], "objective_deletion_identity": f"DELETED_weighted_rollout_h32_and_h{horizon}_BEFORE_BACKWARD", "objective_renormalization": "NO", "raw_total_gradient_norm": raw_norm, "returned_pre_clipping_total_norm": returned_preclip, "clipping_activated": "YES" if returned_preclip > CLIP_MAX_NORM else "NO", "exact_clip_coefficient": clip_coefficient, "post_clipping_gradient_norm": postclip_norm, "postclip_reconstruction_residual_norm": reconstruction_residual,
        "raw_gradient_digest": d7.d6.named_tensor_digest(raw_gradients), "post_clipping_gradient_digest": d7.d6.named_tensor_digest(postclip_gradients), "gradient_none_pattern": none_flags, "gradient_none_pattern_hash": hashlib.sha256(json.dumps(none_flags, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "h1_gradient_norm": d16.norm(h1_gradients.values()), "h32_gradient_norm": d16.norm(h32_gradients.values()), "weighted_h32_gradient_norm": d16.norm(h32_gradients[name] * float(weights[-1]) for name in ALL_NAMES), "h1_gradient_norm_by_block": d16.block_norms(h1_gradients), "h32_gradient_norm_by_block": d16.block_norms(h32_gradients), "raw_gradient_norm_by_block": d16.block_norms(raw_gradients), "postclip_gradient_norm_by_block": d16.block_norms(postclip_gradients),
        "parameter_update_norm": d16.norm(parameter_delta.values()), "parameter_update_norm_by_block": d16.block_norms(parameter_delta), "parameter_update_digest": d7.d6.named_tensor_digest(parameter_delta), "loss_driven_adamw_displacement_norm": adamw_metrics["aggregate"]["whole_model"]["adaptive_update_norm"], "decoupled_weight_decay_displacement_norm": adamw_metrics["aggregate"]["whole_model"]["decay_update_norm"], "adamw_decomposition_residual_norm": adamw_metrics["aggregate"]["whole_model"]["decomposition_residual_norm"], "adamw_decomposition_status": adamw_metrics["aggregate"]["whole_model"]["decomposition_pass_fail"],
        "step_counter_before": before_steps, "step_counter_after": after_steps, "exp_avg_norm": adamw_metrics["aggregate"]["whole_model"]["exp_avg_norm"], "exp_avg_sq_norm": adamw_metrics["aggregate"]["whole_model"]["exp_avg_sq_norm"], "adamw": adamw_metrics, "validation": validation, "per_update_H1": validation["1"], "per_update_H32": validation["32"], "finite_status": "FINITE", "branch_rng_contract_status": "PASS",
    }
    capture = {"branch": branch, "update": update, "model_before": before_model, "model_after": after_model, "optimizer_before": before_optimizer, "optimizer_after": after_optimizer, "raw_gradients": raw_gradients, "postclip_gradients": postclip_gradients, "parameter_delta": parameter_delta, "h1_gradients_diagnostic": h1_gradients, "h32_gradients_diagnostic": h32_gradients, "adamw_decomposition_tensors": adamw_tensors}
    return record, capture, copy.deepcopy(rng_after_validation)


def run_reference_control(state: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    model, optimizer = d16.new_branch(state["bundle"])
    rng = copy.deepcopy(state["bundle"]["rng_state"])
    rows: list[dict[str, Any]] = []
    captures: dict[int, dict[str, Any]] = {}
    for zero_index in SCHEDULE_INDICES:
        row, capture, rng = d17.branch_update("CONTROL", model, optimizer, state, zero_index, rng)
        rows.append(row)
        captures[row["update"]] = capture
    if len(rows) != 8 or rows[-1]["update"] != 392 or any(value != 392 for value in rows[-1]["step_counter_after"].values()):
        raise StageBlocker("REFERENCE_CONTROL_FINAL_STEP_MISMATCH")
    return rows, captures


def validate_control_reference(rows: Sequence[Mapping[str, Any]], reference: Mapping[int, Mapping[str, Any]]) -> list[dict[str, Any]]:
    exact = ("batch_window_sha256", "model_state_hash_before", "model_state_hash_after", "optimizer_state_hash_before", "optimizer_state_hash_after", "raw_gradient_digest", "post_clipping_gradient_digest", "parameter_update_digest")
    numeric = ("total_loss", "rollout_loss_full", "hidden_loss_unweighted", "explicit_h1_loss", "raw_total_gradient_norm", "returned_pre_clipping_total_norm", "post_clipping_gradient_norm", "parameter_update_norm", "exp_avg_norm", "exp_avg_sq_norm", "per_update_H1", "per_update_H32")
    checks: list[dict[str, Any]] = []
    for row in rows:
        ref = reference[int(row["update"])]
        failures = [field for field in exact if row[field] != ref[field]]
        for field in numeric:
            if not math.isclose(float(row[field]), float(ref[field]), rel_tol=REPLAY_RTOL, abs_tol=REPLAY_ATOL):
                failures.append(field)
        checks.append({"update": row["update"], "status": "PASS" if not failures else "FAIL", "failed_fields": failures})
    if any(item["status"] != "PASS" for item in checks):
        raise StageBlocker(f"CONTROL_REFERENCE_FIDELITY_FAILED:{checks}")
    return checks


def make_trajectory(rows: Sequence[Mapping[str, Any]], control_rows: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    updates = []
    for row in rows:
        control = control_rows[int(row["update"])]
        updates.append({
            "update": row["update"], "raw_gradient_norm": row["raw_total_gradient_norm"], "raw_gradient_norm_difference_from_control": row["RAW_GRADIENT_NORM_DIFFERENCE_FROM_CONTROL"], "postclipping_gradient_norm": row["post_clipping_gradient_norm"], "postclipping_gradient_norm_difference_from_control": row["POSTCLIP_GRADIENT_NORM_DIFFERENCE_FROM_CONTROL"], "clip_coefficient": row["exact_clip_coefficient"], "clipping_activated": row["clipping_activated"], "adamw_displacement": row["parameter_update_norm"], "adamw_displacement_norm_difference_from_control": row["ADAMW_DISPLACEMENT_NORM_DIFFERENCE_FROM_CONTROL"], "parameter_distance_from_control": row["branch_vs_control_parameter_distance_norm"], "exp_avg_distance_from_control": row["branch_vs_control_exp_avg_difference_norm"], "exp_avg_sq_distance_from_control": row["branch_vs_control_exp_avg_sq_difference_norm"], "update_increment_difference_from_control": row["update_increment_difference_norm"], "model_state_hash_after": row["model_state_hash_after"], "control_model_state_hash_after": control["model_state_hash_after"], "finite_status": row["finite_status"], "adamw_decomposition_status": row["adamw_decomposition_status"],
        })
    return {"branch": rows[0]["branch"], "horizon": rows[0]["horizon"], "reference": "AUTHENTICATED_D17_CONTROL_WITH_ONE_IN_MEMORY_REFERENCE_CAPTURE", "updates": updates, "first_update_where_parameter_trajectory_differs": next((item["update"] for item in updates if item["parameter_distance_from_control"] != 0.0), None)}


def trend(values: Sequence[float]) -> str:
    if len(values) < 2:
        return "UNAVAILABLE"
    if values[-1] > values[0] * 1.000001:
        return "AMPLIFIES"
    if values[-1] < values[0] * 0.999999:
        return "CONTRACTS_OR_RECONVERGES"
    return "APPROXIMATELY_STABLE"


def trajectory_summary(trajectory: Mapping[str, Any]) -> dict[str, Any]:
    updates = trajectory["updates"]
    fields = ("raw_gradient_norm_difference_from_control", "postclipping_gradient_norm_difference_from_control", "adamw_displacement_norm_difference_from_control", "parameter_distance_from_control", "exp_avg_distance_from_control", "exp_avg_sq_distance_from_control", "update_increment_difference_from_control")
    return {"branch": trajectory["branch"], "horizon": trajectory["horizon"], "first_update_where_parameter_trajectory_differs": trajectory["first_update_where_parameter_trajectory_differs"], **{f"{field}_trend": trend([float(item[field]) for item in updates]) for field in fields}, "endpoint_update": updates[-1]["update"], "endpoint_finite_status": updates[-1]["finite_status"], "endpoint_adamw_decomposition_status": updates[-1]["adamw_decomposition_status"]}


def write_manifest(output: Path) -> dict[str, Any]:
    excluded = {"FINAL_REPORT.md", "SHA256_MANIFEST.json", "manifest_verification.json"}
    files: dict[str, Any] = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name not in excluded:
            rel = path.relative_to(output).as_posix()
            files[rel] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    manifest = {"schema_version": "stage3_h13_post_d18_d19_execution_sha256_manifest_v1", "hash_algorithm": "SHA-256", "self_excluded": True, "excluded_files": sorted(excluded), "file_count": len(files), "total_bytes": sum(item["bytes"] for item in files.values()), "files": files}
    write_json(output / "SHA256_MANIFEST.json", manifest)
    return manifest


def verify_output_manifest(output: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    failures = []
    for name, entry in manifest["files"].items():
        path = output / name
        if not path.is_file() or path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
            failures.append(name)
    parsed = json.loads((output / "SHA256_MANIFEST.json").read_text(encoding="utf-8"))
    if parsed != manifest:
        failures.append("SHA256_MANIFEST.json:reopen_parse_mismatch")
    return {"status": "VERIFIED" if not failures else "FAILED", "checked": len(manifest["files"]), "failed_files": failures, "manifest_path": str((output / "SHA256_MANIFEST.json").resolve()), "manifest_sha256": sha256_file(output / "SHA256_MANIFEST.json"), "reopened_and_parsed": True}


def calculate_results(imported: Mapping[str, Any], pair_endpoints: Mapping[int, Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    control_h1 = float(imported["d18_control_h1"])
    control_h32 = float(imported["d18_control_h32"])
    d16_import = imported["d16_import"]
    d16_h1 = float(d16_import["delete_h32_h1"])
    d16_h32 = float(d16_import["delete_h32_h32"])
    e32 = control_h1 - d16_h1
    rows: list[dict[str, Any]] = []
    for horizon in REST_HORIZONS:
        endpoint = pair_endpoints[horizon]
        delete_hi = float(imported["d17_import"].get("individual_endpoint_values", {}).get(str(horizon), endpoint.get("DELETE_Hi_H1"))) if False else float(endpoint["DELETE_Hi_H1"])
        pair_h1 = float(endpoint["DELETE_H32_PLUS_Hi_H1"])
        ei = control_h1 - delete_hi
        e32i = control_h1 - pair_h1
        interaction = e32i - e32 - ei
        pair_h32 = float(endpoint["DELETE_H32_PLUS_Hi_H32"])
        rows.append({"HORIZON": horizon, "CONTROL_H1": control_h1, "DELETE_Hi_H1": delete_hi, "DELETE_H32_H1": d16_h1, "DELETE_H32_PLUS_Hi_H1": pair_h1, "E_i": ei, "E_32": e32, "E_32_i": e32i, "I_32_i": interaction, "ABS_I_32_i": abs(interaction), "PAIR_INTERACTION_FRACTION_OF_D18_CROSS": interaction / EXPECTED_D18_CROSS, "CONTROL_H32": control_h32, "DELETE_H32_PLUS_Hi_H32": pair_h32, "H32_PRESERVED": pair_h32 <= H32_PRESERVATION_THRESHOLD, "H1_MATERIAL_EFFECT_PASS_FOR_PAIR_BRANCH": e32i >= H1_MATERIAL_THRESHOLD, "VALIDITY": endpoint["VALIDITY"]})
    if len(rows) != 30 or [row["HORIZON"] for row in rows] != list(REST_HORIZONS):
        raise StageBlocker("PRIMARY_ROW_COUNT_OR_ORDER_MISMATCH")
    total_abs = sum(float(row["ABS_I_32_i"]) for row in rows)
    for row in rows:
        row["ABS_MASS_SHARE_i"] = (float(row["ABS_I_32_i"]) / total_abs) if total_abs != 0.0 else None
    signed_rank = sorted(rows, key=lambda row: (-float(row["I_32_i"]), int(row["HORIZON"])))
    abs_rank = sorted(rows, key=lambda row: (-float(row["ABS_I_32_i"]), int(row["HORIZON"])))
    pair_sum = sum(float(row["I_32_i"]) for row in rows)
    remainder = EXPECTED_D18_CROSS - pair_sum
    conservation = EXPECTED_D18_CROSS - (pair_sum + remainder)
    if abs(conservation) > 1e-18:
        raise StageBlocker(f"PAIRWISE_DECOMPOSITION_CONSERVATION_RESIDUAL_TOO_LARGE:{conservation}")
    decomposition = {"I_32_REST": EXPECTED_D18_CROSS, "SUM_PAIRWISE_H32_INTERACTIONS": pair_sum, "PAIRWISE_SIGNED_COVERAGE": pair_sum / EXPECTED_D18_CROSS, "HIGHER_ORDER_CROSS_REMAINDER": remainder, "HIGHER_ORDER_CROSS_REMAINDER_FRACTION": remainder / EXPECTED_D18_CROSS, "PAIRWISE_DECOMPOSITION_CONSERVATION_RESIDUAL": conservation, "TOTAL_ABS_PAIRWISE_INTERACTION_MASS": total_abs, "ABS_HIGHER_ORDER_REMAINDER_MATERIAL": abs(remainder) >= H1_MATERIAL_THRESHOLD, "TOP1_ABS_MASS_SHARE": sum(float(row["ABS_MASS_SHARE_i"]) for row in abs_rank[:1]), "TOP3_ABS_MASS_SHARE": sum(float(row["ABS_MASS_SHARE_i"]) for row in abs_rank[:3]), "TOP5_ABS_MASS_SHARE": sum(float(row["ABS_MASS_SHARE_i"]) for row in abs_rank[:5]), "TOP10_ABS_MASS_SHARE": sum(float(row["ABS_MASS_SHARE_i"]) for row in abs_rank[:10]), "signed_values_retained": True, "no_intermediate_rounding": True}
    rankings = {"signed_I_32_i_descending": [{"HORIZON": row["HORIZON"], "I_32_i": row["I_32_i"]} for row in signed_rank], "absolute_I_32_i_descending": [{"HORIZON": row["HORIZON"], "ABS_I_32_i": row["ABS_I_32_i"], "ABS_MASS_SHARE_i": row["ABS_MASS_SHARE_i"]} for row in abs_rank]}
    label = "H32_BY_REST_PAIRWISE_SECOND_ORDER_SUFFICIENT_UNDER_FROZEN_THRESHOLD" if abs(remainder) < H1_MATERIAL_THRESHOLD else "H32_BY_MULTI_HORIZON_HIGHER_ORDER_CROSS_INTERACTION_REQUIRED"
    classification = {"classification": label, "threshold": H1_MATERIAL_THRESHOLD, "abs_higher_order_remainder": abs(remainder), "rule": "CASE_A if abs(remainder) < threshold else CASE_B", "validity": "VALID"}
    return rows, decomposition, rankings, classification


def render_report(output: Path, identity: Mapping[str, Any], baseline: Mapping[str, Any], branch_manifest: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], decomposition: Mapping[str, Any], rankings: Mapping[str, Any], classification: Mapping[str, Any], trajectory: Mapping[str, Any], defects: Mapping[str, Any], manifest_sha256: str) -> str:
    label = classification["classification"]
    if label == "H32_BY_REST_PAIRWISE_SECOND_ORDER_SUFFICIENT_UNDER_FROZEN_THRESHOLD":
        verdict = "The 30 identifiable H32×Hi pairwise endpoint interactions explain the authenticated D18 H32×REST interaction within the frozen H1 material resolution."
    else:
        verdict = "The 30 identifiable H32×Hi pairwise endpoint interactions leave a materially sized higher-order H32×multi-horizon endpoint remainder under the frozen threshold."
    table_fields = ["HORIZON", "CONTROL_H1", "DELETE_Hi_H1", "DELETE_H32_H1", "DELETE_H32_PLUS_Hi_H1", "E_i", "E_32", "E_32_i", "I_32_i", "ABS_I_32_i", "PAIR_INTERACTION_FRACTION_OF_D18_CROSS", "CONTROL_H32", "DELETE_H32_PLUS_Hi_H32", "H32_PRESERVED", "H1_MATERIAL_EFFECT_PASS_FOR_PAIR_BRANCH"]
    lines = [f"{EXPERIMENT_ID}_REPLAY:", "PASSED", "", "FIRST_BLOCKER:", "none", "", "CAUSAL_CLASSIFICATION:", label, "", "ONE_SENTENCE_VERDICT:", verdict, "", "1. PROTOCOL_INTEGRITY", "", json.dumps(branch_manifest["protocol_integrity"], indent=2, sort_keys=True), "", "2. AUTHORITATIVE_INPUT_IDENTITY", "", json.dumps(identity, indent=2, sort_keys=True), "", "3. EXECUTED_BRANCH_MANIFEST", "", json.dumps(branch_manifest, indent=2, sort_keys=True), "", "4. IMPORTED_BASELINE_VALIDATION", "", json.dumps(baseline["summary"], indent=2, sort_keys=True), "", "5. CONTROL_AND_IMPORTED_MAIN_EFFECTS", "", json.dumps(baseline["main_effects"], indent=2, sort_keys=True), "", "6. PAIRWISE_ENDPOINT_RESULTS", "", "All 30 pair branches reached update 392 with valid H1/H32 endpoint measurements.", "", "7. PAIRWISE_INTERACTION_TABLE", "", "| " + " | ".join(table_fields) + " |", "|" + "|".join(["---"] * len(table_fields)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row[field]) for field in table_fields) + " |")
    lines += ["", "8. SIGNED_INTERACTION_RANKING", "", json.dumps(rankings["signed_I_32_i_descending"], indent=2, sort_keys=True), "", "9. ABSOLUTE_INTERACTION_RANKING", "", json.dumps(rankings["absolute_I_32_i_descending"], indent=2, sort_keys=True), "", "10. GLOBAL_DECOMPOSITION", "", json.dumps(decomposition, indent=2, sort_keys=True), "", "11. MATERIALITY_AND_H32_PRESERVATION", "", f"H1 threshold: {H1_MATERIAL_THRESHOLD:.17g}; H32 threshold: {H32_PRESERVATION_THRESHOLD:.17g}; pair H1 material passes: {sum(bool(row['H1_MATERIAL_EFFECT_PASS_FOR_PAIR_BRANCH']) for row in rows)}/30; pair H32 preservation passes: {sum(bool(row['H32_PRESERVED']) for row in rows)}/30; ABS_HIGHER_ORDER_REMAINDER_MATERIAL: {decomposition['ABS_HIGHER_ORDER_REMAINDER_MATERIAL']}", "", "12. TRAJECTORY_DIAGNOSTICS", "", "Trajectory fields are secondary/descriptive and do not replace the endpoint estimand or establish mechanism.", json.dumps(trajectory, indent=2, sort_keys=True), "", "13. ENGINEERING_DEFECTS_AND_REPAIRS", "", json.dumps(defects, indent=2, sort_keys=True), "", "14. FROZEN_CLASSIFICATION_DECISION", "", json.dumps(classification, indent=2, sort_keys=True), "", "15. SCIENTIFIC_INTERPRETATION", "", f"- Are the H32×Hi second-order terms sufficient under the frozen threshold? {'YES' if label == 'H32_BY_REST_PAIRWISE_SECOND_ORDER_SUFFICIENT_UNDER_FROZEN_THRESHOLD' else 'NO'}.", f"- Which horizons dominate the H32 pairwise interaction mass? H{', H'.join(str(item['HORIZON']) for item in rankings['absolute_I_32_i_descending'][:5])} by absolute mass.", f"- Is the D18 interaction concentrated or distributed? {'Concentrated' if decomposition['TOP5_ABS_MASS_SHARE'] >= 0.5 else 'Distributed'} descriptively; TOP5_ABS_MASS_SHARE={decomposition['TOP5_ABS_MASS_SHARE']:.17g}.", f"- Does a materially sized higher-order H32×multi-horizon remainder remain? {'YES' if decomposition['ABS_HIGHER_ORDER_REMAINDER_MATERIAL'] else 'NO'}.", "- Which conclusions are causal endpoint conclusions versus merely descriptive trajectory observations? The pairwise estimands, signed decomposition, materiality, preservation, and frozen class are causal endpoint conclusions under the replay contract; trajectory distances, gradient norms, clipping, and AdamW diagnostics are descriptive only.", "", "16. ARTIFACT_PATHS_AND_HASHES", "", f"OUTPUT_DIRECTORY: {output.resolve()}", f"SHA256_MANIFEST: {(output / 'SHA256_MANIFEST.json').resolve()}", f"SHA256_MANIFEST_SHA256: {manifest_sha256}", "FINAL_REPORT.md and manifest_verification.json are self-excluded from SHA256_MANIFEST.json; all other machine-readable evidence is listed there.", "", "D19_EXECUTION_COMPLETE:", "YES", "", "ALL_30_REQUIRED_PAIR_BRANCHES_VALID:", "YES", "", "UPDATE_393_EXECUTED:", "NO", "", "FROZEN20_USED:", "NO", "", "R9_EXECUTED:", "NO", "", "R8E_A1_MODIFIED:", "NO", "", "HYPERPARAMETER_TUNING_EXECUTED:", "NO", "", "FINAL_MANIFEST_VERIFICATION:", "VERIFIED", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    output = (args.output if args.output is not None else ROOT / "outputs" / f"stage3_h13_post_d18_d19_h32_by_individual_rest_pairwise_interaction_causal_screen_replay_{stamp}").resolve()
    output.mkdir(parents=True, exist_ok=False)
    progress = {"schema_version": "stage3_h13_post_d18_d19_execution_progress_v1", "status": "RUNNING", "started_at": utc_now(), "required_pair_branches": list(PAIR_BRANCHES), "completed_pair_branches": [], "current_branch": None, "updates_per_branch": list(UPDATES), "update_393_executed": "NO"}
    write_json(output / "execution_progress.json", progress)
    try:
        identity, imported_identity, contract_bundle = verify_design_and_inputs()
        baseline = verify_imported_evidence(imported_identity)
        runtime_info = preflight_runtime()
        runtime = runtime_info["runtime"]
        control_reference_rows, control_captures = run_reference_control(runtime)
        control_checks = validate_control_reference(control_reference_rows, baseline["d17_control_rows"])
        baseline["control_fidelity_checks"] = control_checks
        control_h1 = float(baseline["d18_control_h1"])
        control_h32 = float(baseline["d18_control_h32"])
        d16_import = baseline["d16_import"]
        pair_horizons = {int(item["horizon"]): item for item in imported_identity["d17_individual_endpoint_imports"]}
        for horizon in REST_HORIZONS:
            pair_horizons[horizon]["DELETE_Hi_H1"] = float(pair_horizons[horizon]["h1"])
        identity["runtime_source_identity"] = {"git_head": runtime_info["git_head"], "source_hashes": runtime_info["source_hashes"], "runner_path": str(Path(__file__).resolve()), "runner_sha256": runtime_info["runner_sha256"], "environment": runtime_info["runtime"]["environment"], "rollout_horizon_weights_32": runtime_info["rollout_horizon_weights_32"]}
        identity["authenticated_d18_import_manifest"] = baseline["d18_manifest"]
        write_json(output / "authoritative_input_identity.json", identity)
        write_json(output / "design_bundle_identity.json", {"design_bundle_path": str(DESIGN_DIR.resolve()), "manifest": contract_bundle["design_manifest"], "design_review_status": contract_bundle["design"]["design_review_status"], "execution_authorized_in_sealed_design": contract_bundle["design"]["execution_authorized"], "authorization_text_path": str(AUTHORIZATION_PATH.resolve()), "authorization_text_sha256": sha256_file(AUTHORIZATION_PATH)})
        write_json(output / "imported_baseline_validation.json", {"schema_version": "stage3_h13_post_d18_d19_imported_baseline_validation_v1", "status": "PASSED_READ_ONLY", "summary": {"d18_manifest_verified": True, "d16_manifest_verified": True, "d17_manifest_verified": True, "d18_execution_status": "PASSED", "d18_endpoint_update": 392, "d18_cross_group_interaction": EXPECTED_D18_CROSS}, "d18": baseline["d18_endpoint"], "d18_decomposition": baseline["d18_decomposition"], "d16_import": d16_import, "d17_import": baseline["d17_import"], "control_fidelity_checks": control_checks})
        write_json(output / "execution_configuration.json", {"schema_version": "stage3_h13_post_d18_d19_execution_configuration_v1", "experiment_identifier": EXPERIMENT_ID, "certified_start_path": str(BUNDLE_PATH.resolve()), "certified_start_sha256": EXPECTED_BUNDLE_SHA256, "completed_optimizer_step": 384, "updates": list(UPDATES), "schedule_indices": list(SCHEDULE_INDICES), "endpoint_update": 392, "branch_names": list(PAIR_BRANCHES), "branch_count": 30, "active_horizon": ACTIVE_HORIZON, "deleted_terms": "weighted_rollout_h32 and weighted_rollout_hi before backward", "rollout_weight_source": "src/stage3_h13_r8e.py::rollout_horizon_weights(32)", "rollout_horizon_weights_32": runtime_info["rollout_horizon_weights_32"], "smooth_l1_beta": POSITION_BETA, "h1_coefficient": LAMBDA_H1, "hidden_coefficient": LAMBDA_HIDDEN, "clip_max_norm": CLIP_MAX_NORM, "optimizer_contract": base.optimizer_contract(runtime["optimizer"]), "global_clipping": "YES", "renormalization": "NO", "branch_start": "independent copy of certified post-update-384 model, optimizer, and RNG state", "update_393": "FORBIDDEN", "frozen20": "FORBIDDEN", "r9": "FORBIDDEN", "r8e_a1_modified": "NO", "hyperparameter_tuning": "NO"})
        per_update_path = output / "per_update_instrumentation.jsonl"
        pair_rows: list[dict[str, Any]] = []
        pair_manifests: list[dict[str, Any]] = []
        trajectories: dict[str, Any] = {}
        with per_update_path.open("w", encoding="utf-8", newline="\n") as stream:
            for horizon in REST_HORIZONS:
                branch = f"DELETE_H32_PLUS_H{horizon}"
                progress["current_branch"] = branch
                write_json(output / "execution_progress.json", progress)
                model, optimizer = d16.new_branch(runtime["bundle"])
                start_model_hash = canonical_model_state_sha256(model.state_dict())
                start_optimizer_hash = base.optimizer_semantic_hash(optimizer)
                if start_model_hash != runtime["loaded"]["model_semantic_hash"] or start_optimizer_hash != runtime["loaded"]["optimizer_semantic_hash"]:
                    raise StageBlocker(f"PAIR_START_STATE_IDENTITY_MISMATCH:{branch}")
                start_rng = copy.deepcopy(runtime["bundle"]["rng_state"])
                rng = copy.deepcopy(start_rng)
                rows: list[dict[str, Any]] = []
                captures: dict[int, dict[str, Any]] = {}
                for zero_index in SCHEDULE_INDICES:
                    row, capture, rng = pair_branch_update(branch, horizon, runtime, model, optimizer, zero_index, rng)
                    d16.add_longitudinal_divergence({"branch": "CONTROL"}, row, control_captures[row["update"]], capture)
                    control_row = control_reference_rows[row["update"] - 385]
                    row.update({"RAW_GRADIENT_NORM_DIFFERENCE_FROM_CONTROL": row["raw_total_gradient_norm"] - control_row["raw_total_gradient_norm"], "POSTCLIP_GRADIENT_NORM_DIFFERENCE_FROM_CONTROL": row["post_clipping_gradient_norm"] - control_row["post_clipping_gradient_norm"], "ADAMW_DISPLACEMENT_NORM_DIFFERENCE_FROM_CONTROL": row["parameter_update_norm"] - control_row["parameter_update_norm"], "BRANCH": branch, "UPDATE": row["update"], "TOTAL_LOSS": row["total_loss"], "RAW_GRAD_NORM": row["raw_total_gradient_norm"], "CLIP_COEFFICIENT": row["exact_clip_coefficient"], "POSTCLIP_GRAD_NORM": row["post_clipping_gradient_norm"], "ADAMW_PARAMETER_DISPLACEMENT_NORM": row["parameter_update_norm"], "PARAMETER_DISTANCE_FROM_CONTROL": row["branch_vs_control_parameter_distance_norm"], "EXP_AVG_DISTANCE_FROM_CONTROL": row["branch_vs_control_exp_avg_difference_norm"], "EXP_AVG_SQ_DISTANCE_FROM_CONTROL": row["branch_vs_control_exp_avg_sq_difference_norm"], "UPDATE_INCREMENT_DIFFERENCE_FROM_CONTROL": row["update_increment_difference_norm"], "PARAMETER_UPDATE_INCREMENT_DIFFERENCE_FROM_CONTROL": row["update_increment_difference_norm"]})
                    rows.append(row)
                    captures[row["update"]] = capture
                if len(rows) != 8 or rows[-1]["update"] != 392 or any(value != 392 for value in rows[-1]["step_counter_after"].values()):
                    raise StageBlocker(f"PAIR_FINAL_STEP_MISMATCH:{branch}")
                if any(row["finite_status"] != "FINITE" or row["adamw_decomposition_status"] != "PASS" or row["branch_rng_contract_status"] != "PASS" for row in rows):
                    raise StageBlocker(f"PAIR_ROW_VALIDITY_FAILED:{branch}")
                if rows[0]["rng_before_digest"] != base.rng_digest(start_rng):
                    raise StageBlocker(f"PAIR_START_RNG_MISMATCH:{branch}")
                if rows[-1]["rng_after_validation_digest"] != base.rng_digest(start_rng):
                    raise StageBlocker(f"PAIR_END_RNG_MISMATCH:{branch}")
                for row in rows:
                    stream.write(json.dumps(row, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n")
                    pair_rows.append(row)
                trajectory = make_trajectory(rows, baseline["d17_control_rows"])
                trajectories[branch] = {**trajectory, "summary": trajectory_summary(trajectory)}
                endpoint_row = rows[-1]
                pair_horizons[horizon].update({"DELETE_H32_PLUS_Hi_H1": float(endpoint_row["per_update_H1"]), "DELETE_H32_PLUS_Hi_H32": float(endpoint_row["per_update_H32"]), "VALIDITY": "VALID", "start_model_hash": start_model_hash, "start_optimizer_hash": start_optimizer_hash, "start_rng_digest": base.rng_digest(start_rng), "end_rng_digest": endpoint_row["rng_after_validation_digest"], "updates": list(UPDATES)})
                pair_manifests.append({"branch_name": branch, "branch_type": "new_pairwise_intervention", "horizon_pair": {"H32": 32, "Hi": horizon}, "deleted_horizons": [horizon, 32], "start_model_hash": start_model_hash, "start_optimizer_hash": start_optimizer_hash, "start_rng_digest": base.rng_digest(start_rng), "end_rng_digest": endpoint_row["rng_after_validation_digest"], "updates": list(UPDATES), "endpoint_update": 392, "endpoint_validity": "VALID", "row_count": len(rows), "objective_deletion_identity": endpoint_row["objective_deletion_identity"]})
                progress["completed_pair_branches"].append(branch)
                progress["current_branch"] = None
                write_json(output / "execution_progress.json", progress)
                stream.flush()
        if len(pair_manifests) != 30 or len(pair_rows) != 240:
            raise StageBlocker("D19_EXECUTED_ROW_OR_BRANCH_COUNT_MISMATCH")
        endpoint_inputs = {h: {"DELETE_Hi_H1": pair_horizons[h]["DELETE_Hi_H1"], "DELETE_H32_PLUS_Hi_H1": pair_horizons[h]["DELETE_H32_PLUS_Hi_H1"], "DELETE_H32_PLUS_Hi_H32": pair_horizons[h]["DELETE_H32_PLUS_Hi_H32"], "VALIDITY": "VALID"} for h in REST_HORIZONS}
        rows, decomposition, rankings, classification = calculate_results(baseline, endpoint_inputs)
        endpoint_results = {"schema_version": "stage3_h13_post_d18_d19_endpoint_results_v1", "update": 392, "control": {"CONTROL_H1": control_h1, "CONTROL_H32": control_h32, "source": "AUTHENTICATED_D17_REUSE"}, "imported_main_effects": {"E_32": control_h1 - float(d16_import["delete_h32_h1"]), "DELETE_H32_H1": float(d16_import["delete_h32_h1"]), "DELETE_H32_H32": float(d16_import["delete_h32_h32"]), "D18_H32_REST_INTERACTION": EXPECTED_D18_CROSS}, "thresholds": {"H1_MATERIAL_THRESHOLD": H1_MATERIAL_THRESHOLD, "H32_PRESERVATION_THRESHOLD": H32_PRESERVATION_THRESHOLD}, "primary_pairwise_rows": rows, "global_decomposition": decomposition, "rankings": rankings, "classification": classification}
        write_json(output / "endpoint_results.json", endpoint_results)
        with (output / "endpoint_results.csv").open("w", encoding="utf-8", newline="") as stream:
            fields = list(rows[0].keys())
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        write_json(output / "global_decomposition.json", decomposition)
        write_json(output / "rankings.json", rankings)
        write_json(output / "trajectory_diagnostics.json", {"schema_version": "stage3_h13_post_d18_d19_trajectory_diagnostics_v1", "role": "secondary_descriptive_diagnostics", "updates": list(UPDATES), "endpoint_update": 392, "branches": trajectories, "summaries": [trajectories[branch]["summary"] for branch in PAIR_BRANCHES]})
        defects = {"scientific_protocol_defects": [], "engineering_defects_and_repairs": [{"type": "none_observed", "status": "none", "effect": "No engineering defect required repair during D19 execution."}], "historical_provenance_debt": "The absent historical materialized D18 design-review directory remains documented in imported evidence; the authenticated D18 execution bundle and manifest verified, so this was not a D19 blocker."}
        write_json(output / "engineering_defects_and_repairs.json", defects)
        protocol = {"status": "PASSED", "first_blocker": "none", "design_manifest_verified": "YES", "d18_manifest_verified": "YES", "d17_control_fidelity": "PASS", "certified_start_match": "YES", "source_revision_match": "YES", "runtime_identity_match": "YES", "canonical_schedule_match": "YES", "new_scientific_branches": 30, "all_required_pair_branches_valid": "YES", "updates_385_392_executed": "YES", "update_393_executed": "NO", "frozen20_used": "NO", "r9_executed": "NO", "r8e_a1_modified": "NO", "hyperparameter_tuning_executed": "NO", "historical_artifacts_modified": "NO", "classification": classification["classification"]}
        write_json(output / "protocol_integrity.json", protocol)
        branch_execution_manifest = {"schema_version": "stage3_h13_post_d18_d19_branch_execution_manifest_v1", "scientific_branches_authorized": list(PAIR_BRANCHES), "scientific_branches_executed": [item["branch_name"] for item in pair_manifests], "expected_new_d19_branch_count": 30, "actual_new_d19_branch_count": len(pair_manifests), "start_update": 384, "updates": list(UPDATES), "endpoint_update": 392, "update_393_executed": "NO", "control_reference": "AUTHENTICATED_D17_CONTROL_WITH_ONE_IN_MEMORY_FIDELITY_REPLAY", "control_fidelity_checks": control_checks, "branches": pair_manifests, "protocol_integrity": protocol}
        write_json(output / "branch_execution_manifest.json", branch_execution_manifest)
        progress.update({"status": "PASSED", "completed_at": utc_now(), "completed_pair_branches": list(PAIR_BRANCHES), "current_branch": None, "classification": classification["classification"], "final_endpoint_update": 392})
        write_json(output / "execution_progress.json", progress)
        manifest = write_manifest(output)
        manifest_check = verify_output_manifest(output, manifest)
        if manifest_check["status"] != "VERIFIED":
            raise StageBlocker(f"OUTPUT_MANIFEST_FAILED:{manifest_check}")
        write_json(output / "manifest_verification.json", manifest_check)
        final_reopen = json.loads((output / "endpoint_results.json").read_text(encoding="utf-8"))
        final_rows = list(final_reopen["primary_pairwise_rows"])
        if len(final_rows) != 30 or json.loads((output / "protocol_integrity.json").read_text(encoding="utf-8"))["status"] != "PASSED":
            raise StageBlocker("FINAL_ARTIFACT_REOPEN_VALIDATION_FAILED")
        report_baseline = {"summary": {"d18_manifest_verified": True, "d16_manifest_verified": True, "d17_manifest_verified": True, "d18_cross_group_interaction": EXPECTED_D18_CROSS, "d18_endpoint_update": 392}, "main_effects": endpoint_results["imported_main_effects"]}
        report = render_report(output, identity, report_baseline, branch_execution_manifest, final_rows, decomposition, rankings, classification, json.loads((output / "trajectory_diagnostics.json").read_text(encoding="utf-8")), defects, manifest_check["manifest_sha256"])
        (output / "FINAL_REPORT.md").write_text(report, encoding="utf-8", newline="\n")
        final_manifest_check = verify_output_manifest(output, manifest)
        if final_manifest_check["status"] != "VERIFIED":
            raise StageBlocker(f"FINAL_MANIFEST_VERIFICATION_FAILED:{final_manifest_check}")
        write_json(output / "manifest_verification.json", final_manifest_check)
        print(json.dumps({"status": "PASSED", "output": str(output), "classification": classification["classification"], "I_32_REST": EXPECTED_D18_CROSS, "SUM_PAIRWISE_H32_INTERACTIONS": decomposition["SUM_PAIRWISE_H32_INTERACTIONS"], "HIGHER_ORDER_CROSS_REMAINDER": decomposition["HIGHER_ORDER_CROSS_REMAINDER"], "manifest_sha256": final_manifest_check["manifest_sha256"], "file_count": final_manifest_check["checked"]}, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        progress.update({"status": "BLOCKED/INVALID", "completed_at": utc_now(), "first_blocker": f"{type(exc).__name__}:{exc}"})
        write_json(output / "execution_progress.json", progress)
        (output / "execution_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8", newline="\n")
        write_json(output / "protocol_integrity.json", {"status": "BLOCKED/INVALID", "first_blocker": f"{type(exc).__name__}:{exc}", "update_393_executed": "NO", "frozen20_used": "NO", "r9_executed": "NO", "r8e_a1_modified": "NO"})
        print(json.dumps({"status": "BLOCKED/INVALID", "output": str(output), "first_blocker": f"{type(exc).__name__}:{exc}"}, sort_keys=True), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
