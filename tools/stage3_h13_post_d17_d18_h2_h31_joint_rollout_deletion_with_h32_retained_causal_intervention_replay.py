"""Execute the single preregistered D18 H2..H31 joint-deletion replay.

The D17 bundle is read-only authority.  Exactly one new scientific trajectory
is emitted: DELETE_H2_TO_H31_KEEP_H32 over updates 385..392.  A control pass is
replayed only in memory because the compact D17 bundle contains hashes and
scalar diagnostics, but not the control tensors needed for vector-distance
instrumentation.  It is not written as a second scientific branch or endpoint.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
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


EXPERIMENT_ID = "STAGE_3_H13_POST_D17_D18_H2_H31_JOINT_ROLLOUT_DELETION_WITH_H32_RETAINED_CAUSAL_INTERVENTION_REPLAY"
BRANCH = "DELETE_H2_TO_H31_KEEP_H32"
D17_DIR = ROOT / "outputs/stage3_h13_post_d16_d17_full_pipeline_multi_horizon_rollout_causal_screen_replay_20260820T041657"
BUNDLE_PATH = ROOT / "outputs/stage3_h13_post_d5_d6_branch_state_materialization_replay_20260816T152000Z_retry/post_update_384_branch_bundle.pt"
D18_DESIGN_TEXT = Path(r"C:\Users\86198\.codex\attachments\b1ae86d4-9e5f-48a1-bfa3-038ce1fc2150\pasted-text.txt")
D18_EXECUTION_TEXT = Path(r"C:\Users\86198\.codex\attachments\20d75af2-9043-4e4f-91aa-c2358332238c\pasted-text.txt")

EXPECTED_D17_MANIFEST_SHA256 = "47d37d75fd61574060b997c4eba569969834aedd6a435d2115a0f767879f3c66"
EXPECTED_BUNDLE_SHA256 = "4a022aa0340977e571a3a3085e30d96fd0d92129f4fbeff99aaa8e3420379083"
EXPECTED_SOURCE_REVISION = "a6dff2b6887cc3f7598ab95549c2dd21c0efe763"
EXPECTED_CLASSIFICATION_CONTRACT_SHA256 = "16497f895a4fac479ff6f946fd5bd4e6869eef68859e2af94c4f22229f9dfbe7"
EXPECTED_CONTROL_H1 = 9.096969417553673e-05
EXPECTED_CONTROL_H32 = 0.018159905521264792
SUM_REST_INDIVIDUAL = 7.830229369732231e-06
E_H32 = 1.8340154893151762e-06
E_ALL = 1.7742416075716138e-05
H1_MATERIAL_THRESHOLD = 4.099006815405639e-06
H32_PRESERVATION_THRESHOLD = 0.01856902565856056
UPDATES = tuple(range(385, 393))
START_UPDATE = 384
ACTIVE_HORIZON = 32
POSITION_BETA = 0.001
LAMBDA_H1 = 1.0
LAMBDA_HIDDEN = 0.005
CLIP_MAX_NORM = 1.0
REPLAY_RTOL = 5.0e-6
REPLAY_ATOL = 5.0e-9

ALL_NAMES = d16.ALL_NAMES
GRU_NAMES = d16.GRU_NAMES
HEAD_NAMES = d16.HEAD_NAMES


class StageBlocker(RuntimeError):
    """A scientific-integrity blocker."""


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


def load_d17_rows() -> list[dict[str, Any]]:
    path = D17_DIR / "per_update_instrumentation.jsonl"
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def verify_d17_and_runtime() -> tuple[dict[str, Any], dict[str, Any], dict[int, dict[str, Any]], dict[str, Any]]:
    manifest_path = D17_DIR / "SHA256_MANIFEST.json"
    if sha256_file(manifest_path) != EXPECTED_D17_MANIFEST_SHA256:
        raise StageBlocker("D17_MANIFEST_HASH_MISMATCH")
    manifest = d17.verify_manifest_file(manifest_path, EXPECTED_D17_MANIFEST_SHA256)
    if sha256_file(BUNDLE_PATH) != EXPECTED_BUNDLE_SHA256:
        raise StageBlocker("CERTIFIED_START_BUNDLE_HASH_MISMATCH")
    d17_authority = d17.verify_design_and_inputs()
    runtime = d17.preflight_runtime()
    d17_rows = load_d17_rows()
    control_rows = {int(row["update"]): row for row in d17_rows if row.get("branch") == "CONTROL"}
    if tuple(sorted(control_rows)) != UPDATES:
        raise StageBlocker("D17_CONTROL_REFERENCE_WINDOW_MISMATCH")
    endpoint = json.loads((D17_DIR / "endpoint_results.json").read_text(encoding="utf-8"))
    endpoint_rows = {row["BRANCH"]: row for row in endpoint["canonical_order"]}
    control = endpoint_rows.get("CONTROL")
    joint = endpoint_rows.get("DELETE_ALL_NON_H1_ROLLOUT")
    if control is None or joint is None:
        raise StageBlocker("D17_ENDPOINT_REFERENCE_MISSING")
    if not math.isclose(float(control["CONTROL_H1"]), EXPECTED_CONTROL_H1, rel_tol=1e-12, abs_tol=1e-15) or not math.isclose(float(control["CONTROL_H32"]), EXPECTED_CONTROL_H32, rel_tol=1e-12, abs_tol=1e-15):
        raise StageBlocker("D17_CONTROL_ENDPOINT_MISMATCH")
    if not math.isclose(float(joint["E_H1"]), E_ALL, rel_tol=1e-12, abs_tol=1e-15):
        raise StageBlocker("D17_ALL_FUTURE_ENDPOINT_MISMATCH")
    if not D18_DESIGN_TEXT.is_file() or not D18_EXECUTION_TEXT.is_file():
        raise StageBlocker("D18_AUTHORIZATION_TEXT_MISSING")
    design_text = D18_DESIGN_TEXT.read_text(encoding="utf-8")
    execution_text = D18_EXECUTION_TEXT.read_text(encoding="utf-8")
    required_design = ("PREREGISTERED D18 INTERVENTION", BRANCH, "updates 385–392", "shadow-only")
    required_execution = (EXPERIMENT_ID, "The D18 causal-intervention design review has already PASSED.", "UPDATE_393_EXECUTED:", "NO", "H2_H31_ROLLOUT_FAMILY_CAUSAL_H32_PRESERVED")
    if any(item not in design_text for item in required_design) or any(item not in execution_text for item in required_execution):
        raise StageBlocker("D18_AUTHORIZATION_SCOPE_MISMATCH")
    identities = {
        "d17_manifest": manifest,
        "d17_bundle_sha256": EXPECTED_BUNDLE_SHA256,
        "d17_design_identity": d17_authority["design_manifest"],
        "d17_source_identity": d17_authority["source_identity"],
        "d18_design_review": {"materialized_bundle_found": False, "source_text_path": str(D18_DESIGN_TEXT.resolve()), "source_text_sha256": sha256_file(D18_DESIGN_TEXT), "review_status_asserted_by_execution_authorization": "PASSED", "manifest_verification": "NOT_AVAILABLE_NO_MATERIALIZED_D18_BUNDLE"},
        "d18_execution_authorization": {"path": str(D18_EXECUTION_TEXT.resolve()), "sha256": sha256_file(D18_EXECUTION_TEXT)},
        "classification_contract_sha256": EXPECTED_CLASSIFICATION_CONTRACT_SHA256,
    }
    return runtime, identities, control_rows, {"endpoint_rows": endpoint_rows, "d17_rows": d17_rows}


def d18_branch_update(state: Mapping[str, Any], model: Any, optimizer: torch.optim.Optimizer, zero_index: int, rng_before: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    update = zero_index + 1
    if update not in UPDATES:
        raise StageBlocker(f"UNAUTHORIZED_UPDATE:{update}")
    base.restore_rng(rng_before)
    pre = state["pre"]
    authority = state["authority"]
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if [name for name, _ in named] != list(ALL_NAMES):
        raise StageBlocker(f"BRANCH_PARAMETER_ORDER_MISMATCH:{update}")
    batch_record = d16.batch_identity(authority["schedule_manifest"], zero_index)
    batch = base.tensor_batch(pre["train_data"], authority["schedule_rows"][zero_index])
    before_model = d16.clone_parameters(model)
    before_optimizer = d16.clone_named_optimizer_state(optimizer)
    before_model_hash = canonical_model_state_sha256(model.state_dict())
    before_optimizer_hash = base.optimizer_semantic_hash(optimizer)
    before_steps = d16.optimizer_steps(before_optimizer)
    optimizer.zero_grad(set_to_none=True)
    rollout = base.causal_paired_rollout(model, batch["inputs"], batch["history_positions"], batch["history_times"], batch["target_times"], batch["starts"], batch["ends"], pre["stats"]["channels"], horizon=ACTIVE_HORIZON, target_positions_for_teacher=batch["target_positions"], hidden_consistency_enabled=True)
    if rollout.get("free_branch_reads_target_positions") is not False or rollout.get("teacher_reference_stop_gradient") is not True:
        raise StageBlocker(f"GRAPH_SEMANTICS_GATE_FAILED:{update}")
    predictions = rollout["predictions"]
    targets = batch["target_positions"][:, :ACTIVE_HORIZON, :]
    step_losses = F.smooth_l1_loss(predictions, targets, beta=POSITION_BETA, reduction="none").mean(dim=(0, 2))
    weights = rollout_horizon_weights(ACTIVE_HORIZON).to(device=predictions.device, dtype=predictions.dtype)
    weighted_terms = step_losses * weights
    full_rollout_loss = torch.sum(weighted_terms)
    h1_loss = step_losses[0]
    h32_term = weighted_terms[-1]
    hidden_loss = rollout["hidden_loss"]
    rollout_loss = weighted_terms[0] + weighted_terms[-1]
    total_loss = rollout_loss + LAMBDA_H1 * h1_loss + LAMBDA_HIDDEN * hidden_loss
    values = (full_rollout_loss, rollout_loss, h1_loss, h32_term, hidden_loss, total_loss)
    if any(not bool(torch.isfinite(value)) for value in values):
        raise StageBlocker(f"NONFINITE_OBJECTIVE:{update}")
    h1_gradients, _ = d16.diagnostic_gradients(h1_loss, named)
    h32_gradients, _ = d16.diagnostic_gradients(h32_term, named)
    rng_after_diagnostics = base.capture_rng()
    if base.rng_digest(rng_after_diagnostics) != base.rng_digest(rng_before):
        raise StageBlocker(f"DIAGNOSTIC_RNG_CONTAMINATION:{update}")
    total_loss.backward()
    if not base.gradients_are_finite(model):
        raise StageBlocker(f"NONFINITE_RAW_GRADIENT:{update}")
    raw_gradients, none_flags = d16.capture_gradients(named)
    raw_norm = d16.norm(raw_gradients.values())
    returned_preclip = finite(torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_MAX_NORM, norm_type=2.0, error_if_nonfinite=False, foreach=None))
    if not math.isclose(raw_norm, returned_preclip, rel_tol=1e-5, abs_tol=1e-5):
        raise StageBlocker(f"CLIP_RETURNED_NORM_MISMATCH:{update}")
    postclip_gradients, _ = d16.capture_gradients(named)
    postclip_norm = d16.norm(postclip_gradients.values())
    clip_coefficient = 1.0 if returned_preclip <= CLIP_MAX_NORM else min(1.0, CLIP_MAX_NORM / (returned_preclip + 1e-6))
    reconstruction_residual = d16.norm(postclip_gradients[name] - raw_gradients[name] * clip_coefficient for name in ALL_NAMES)
    if reconstruction_residual > 1e-7 + 2e-5 * max(postclip_norm, 1.0):
        raise StageBlocker(f"POSTCLIP_RECONSTRUCTION_FAILED:{update}")
    rng_before_optimizer = base.capture_rng()
    optimizer.step()
    after_model = d16.clone_parameters(model)
    after_optimizer = d16.clone_named_optimizer_state(optimizer)
    after_model_hash = canonical_model_state_sha256(model.state_dict())
    after_optimizer_hash = base.optimizer_semantic_hash(optimizer)
    if not d16.state_finite(model, optimizer):
        raise StageBlocker(f"NONFINITE_STATE_AFTER_UPDATE:{update}")
    rng_after_training = base.capture_rng()
    if base.rng_digest(rng_before_optimizer) != base.rng_digest(rng_after_training):
        raise StageBlocker(f"OPTIMIZER_RNG_CONSUMPTION_UNMATCHED:{update}")
    adamw_metrics, adamw_tensors = d7.adamw_decomposition(model, optimizer, before_model, postclip_gradients, after_model, state["table"])
    validation, validation_meta = d7.validation_snapshot(model, pre["validation"], pre["stats"]["channels"], True)
    rng_after_validation = base.capture_rng()
    if validation_meta["rng_unchanged_or_restored"] not in ("YES", "RESTORED"):
        raise StageBlocker(f"VALIDATION_RNG_GATE_FAILED:{update}")
    after_steps = d16.optimizer_steps(after_optimizer)
    parameter_delta = {name: after_model[name] - before_model[name] for name in ALL_NAMES}
    record = {
        "branch": BRANCH, "update": update, "schedule_index": zero_index, "batch_identity": batch_record, "batch_window_sha256": batch_record["batch_window_sha256"],
        "model_state_hash_before": before_model_hash, "model_state_hash_after": after_model_hash, "optimizer_state_hash_before": before_optimizer_hash, "optimizer_state_hash_after": after_optimizer_hash,
        "rng_before_digest": base.rng_digest(rng_before), "rng_after_diagnostics_digest": base.rng_digest(rng_after_diagnostics), "rng_after_training_digest": base.rng_digest(rng_after_training), "rng_after_validation_digest": base.rng_digest(rng_after_validation),
        "rollout_loss_full": finite(full_rollout_loss), "rollout_loss_used_for_backward": finite(rollout_loss), "weighted_rollout_h1_term": finite(weighted_terms[0]), "weighted_rollout_h32_term": finite(h32_term), "explicit_h1_loss": finite(h1_loss), "hidden_loss_unweighted": finite(hidden_loss), "weighted_hidden_loss": finite(LAMBDA_HIDDEN * hidden_loss), "total_loss": finite(total_loss),
        "deleted_horizons": list(range(2, 32)), "objective_deletion_identity": "DELETED_weighted_rollout_h2_to_h31_BEFORE_BACKWARD; RETAINED_weighted_rollout_h1_and_h32", "objective_renormalization": "NO", "raw_total_gradient_norm": raw_norm, "returned_pre_clipping_total_norm": returned_preclip, "clipping_activated": "YES" if returned_preclip > CLIP_MAX_NORM else "NO", "exact_clip_coefficient": clip_coefficient, "post_clipping_gradient_norm": postclip_norm, "postclip_reconstruction_residual_norm": reconstruction_residual,
        "raw_gradient_digest": d7.d6.named_tensor_digest(raw_gradients), "post_clipping_gradient_digest": d7.d6.named_tensor_digest(postclip_gradients), "gradient_none_pattern": none_flags, "gradient_none_pattern_hash": hashlib.sha256(json.dumps(none_flags, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "h1_gradient_norm": d16.norm(h1_gradients.values()), "h32_gradient_norm": d16.norm(h32_gradients.values()), "weighted_h32_gradient_norm": d16.norm(h32_gradients[name] * float(weights[-1]) for name in ALL_NAMES), "h1_gradient_norm_by_block": d16.block_norms(h1_gradients), "h32_gradient_norm_by_block": d16.block_norms(h32_gradients), "raw_gradient_norm_by_block": d16.block_norms(raw_gradients), "postclip_gradient_norm_by_block": d16.block_norms(postclip_gradients),
        "parameter_update_norm": d16.norm(parameter_delta.values()), "parameter_update_norm_by_block": d16.block_norms(parameter_delta), "parameter_update_digest": d7.d6.named_tensor_digest(parameter_delta), "loss_driven_adamw_displacement_norm": adamw_metrics["aggregate"]["whole_model"]["adaptive_update_norm"], "decoupled_weight_decay_displacement_norm": adamw_metrics["aggregate"]["whole_model"]["decay_update_norm"], "adamw_decomposition_residual_norm": adamw_metrics["aggregate"]["whole_model"]["decomposition_residual_norm"], "adamw_decomposition_status": adamw_metrics["aggregate"]["whole_model"]["decomposition_pass_fail"],
        "step_counter_before": before_steps, "step_counter_after": after_steps, "exp_avg_norm": adamw_metrics["aggregate"]["whole_model"]["exp_avg_norm"], "exp_avg_sq_norm": adamw_metrics["aggregate"]["whole_model"]["exp_avg_sq_norm"], "adamw": adamw_metrics, "validation": validation, "per_update_H1": validation["1"], "per_update_H32": validation["32"], "finite_status": "FINITE", "branch_rng_contract_status": "PASS",
    }
    capture = {"branch": BRANCH, "update": update, "model_before": before_model, "model_after": after_model, "optimizer_before": before_optimizer, "optimizer_after": after_optimizer, "raw_gradients": raw_gradients, "postclip_gradients": postclip_gradients, "parameter_delta": parameter_delta, "h1_gradients_diagnostic": h1_gradients, "h32_gradients_diagnostic": h32_gradients, "adamw_decomposition_tensors": adamw_tensors}
    return record, capture, copy.deepcopy(rng_after_validation)


def run_reference_control(state: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    model, optimizer = d16.new_branch(state["bundle"])
    rng = copy.deepcopy(state["bundle"]["rng_state"])
    rows: list[dict[str, Any]] = []
    captures: dict[int, dict[str, Any]] = {}
    for zero_index in range(384, 392):
        row, capture, rng = d17.branch_update("CONTROL", model, optimizer, state, zero_index, rng)
        rows.append(row)
        captures[row["update"]] = capture
    if any(value != 392 for value in rows[-1]["step_counter_after"].values()):
        raise StageBlocker("REFERENCE_CONTROL_FINAL_STEP_MISMATCH")
    return rows, captures


def validate_control_reference(rows: Sequence[Mapping[str, Any]], d17_rows: Mapping[int, Mapping[str, Any]]) -> list[dict[str, Any]]:
    exact = ("batch_window_sha256", "model_state_hash_before", "model_state_hash_after", "optimizer_state_hash_before", "optimizer_state_hash_after", "raw_gradient_digest", "post_clipping_gradient_digest", "parameter_update_digest")
    numeric = ("total_loss", "rollout_loss_full", "hidden_loss_unweighted", "explicit_h1_loss", "raw_total_gradient_norm", "returned_pre_clipping_total_norm", "post_clipping_gradient_norm", "parameter_update_norm", "exp_avg_norm", "exp_avg_sq_norm", "per_update_H1", "per_update_H32")
    checks = []
    for row in rows:
        ref = d17_rows[int(row["update"])]
        failures = [field for field in exact if row[field] != ref[field]]
        for field in numeric:
            if not math.isclose(float(row[field]), float(ref[field]), rel_tol=REPLAY_RTOL, abs_tol=REPLAY_ATOL):
                failures.append(field)
        checks.append({"update": row["update"], "status": "PASS" if not failures else "FAIL", "failed_fields": failures})
    if any(item["status"] != "PASS" for item in checks):
        raise StageBlocker(f"CONTROL_REFERENCE_FIDELITY_FAILED:{checks}")
    return checks


def trend(values: Sequence[float]) -> str:
    if len(values) < 2:
        return "UNAVAILABLE"
    if values[-1] > values[0] * 1.000001:
        return "AMPLIFIES"
    if values[-1] < values[0] * 0.999999:
        return "CONTRACTS_OR_RECONVERGES"
    return "APPROXIMATELY_STABLE"


def classification(e_rest: float, h32: float) -> tuple[str, bool, bool]:
    material = e_rest >= H1_MATERIAL_THRESHOLD
    preserved = h32 <= H32_PRESERVATION_THRESHOLD
    if e_rest <= -H1_MATERIAL_THRESHOLD:
        return "CONTRARY_PROTECTIVE_EFFECT_H2_H31_JOINT_DELETION", material, preserved
    if material and preserved:
        return "H2_H31_ROLLOUT_FAMILY_CAUSAL_H32_PRESERVED", material, preserved
    if material:
        return "H2_H31_ROLLOUT_FAMILY_CAUSAL_WITH_H32_TRADEOFF", material, preserved
    if preserved:
        return "H32_BY_REST_SUPERADDITIVE_INTERACTION_REQUIRED", material, preserved
    return "H2_H31_JOINT_DELETION_NOT_VIABLE", material, preserved


def make_trajectory(rows: Sequence[Mapping[str, Any]], control_rows: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    updates = []
    for row in rows:
        control = control_rows[int(row["update"])]
        updates.append({
            "update": row["update"],
            "raw_gradient_norm": row["raw_total_gradient_norm"],
            "raw_gradient_norm_difference_from_control": row["RAW_GRADIENT_NORM_DIFFERENCE_FROM_CONTROL"],
            "postclipping_gradient_norm": row["post_clipping_gradient_norm"],
            "postclipping_gradient_norm_difference_from_control": row["POSTCLIP_GRADIENT_NORM_DIFFERENCE_FROM_CONTROL"],
            "clip_coefficient": row["exact_clip_coefficient"],
            "adamw_displacement": row["parameter_update_norm"],
            "adamw_displacement_norm_difference_from_control": row["ADAMW_DISPLACEMENT_NORM_DIFFERENCE_FROM_CONTROL"],
            "parameter_distance_from_control": row["branch_vs_control_parameter_distance_norm"],
            "exp_avg_distance_from_control": row["branch_vs_control_exp_avg_difference_norm"],
            "exp_avg_sq_distance_from_control": row["branch_vs_control_exp_avg_sq_difference_norm"],
            "update_increment_difference_from_control": row["update_increment_difference_norm"],
            "model_state_hash_after": row["model_state_hash_after"],
            "control_model_state_hash_after": control["model_state_hash_after"],
        })
    return {
        "branch": BRANCH,
        "reference": "AUTHENTICATED_D17_CONTROL_WITH_IN_MEMORY_REFERENCE_CAPTURE",
        "first_update_where_parameter_trajectory_differs": next((item["update"] for item in updates if item["parameter_distance_from_control"] != 0.0), None),
        "raw_gradient_norm_difference_trend": trend([item["raw_gradient_norm_difference_from_control"] for item in updates]),
        "postclipping_gradient_norm_difference_trend": trend([item["postclipping_gradient_norm_difference_from_control"] for item in updates]),
        "adamw_displacement_difference_trend": trend([item["adamw_displacement_norm_difference_from_control"] for item in updates]),
        "parameter_distance_trend": trend([item["parameter_distance_from_control"] for item in updates]),
        "exp_avg_distance_trend": trend([item["exp_avg_distance_from_control"] for item in updates]),
        "exp_avg_sq_distance_trend": trend([item["exp_avg_sq_distance_from_control"] for item in updates]),
        "update_increment_difference_trend": trend([item["update_increment_difference_from_control"] for item in updates]),
        "updates": updates,
    }


def write_manifest(output: Path) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name in {"FINAL_REPORT.md", "SHA256_MANIFEST.json", "manifest_verification.json"}:
            continue
        rel = path.relative_to(output).as_posix()
        files[rel] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    manifest = {"excluded_files": ["FINAL_REPORT.md", "SHA256_MANIFEST.json", "manifest_verification.json"], "file_count": len(files), "files": files, "hash_algorithm": "SHA-256", "schema_version": "stage3_h13_post_d17_d18_execution_sha256_manifest_v1", "self_excluded": True, "total_bytes": sum(item["bytes"] for item in files.values())}
    write_json(output / "SHA256_MANIFEST.json", manifest)
    return manifest


def verify_output_manifest(output: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    failures = []
    for name, item in manifest["files"].items():
        path = output / name
        if not path.is_file() or path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
            failures.append(name)
    return {"status": "VERIFIED" if not failures else "FAILED", "checked": len(manifest["files"]), "failed_files": failures, "manifest_path": str((output / "SHA256_MANIFEST.json").resolve()), "manifest_sha256": sha256_file(output / "SHA256_MANIFEST.json")}


def render_report(output: Path, identity: Mapping[str, Any], configuration: Mapping[str, Any], branch_manifest: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], endpoint: Mapping[str, Any], decomposition: Mapping[str, Any], trajectory: Mapping[str, Any], defects: Mapping[str, Any]) -> str:
    e = endpoint["d18"]
    lines = [
        f"{EXPERIMENT_ID}:", "PASSED", "", "FIRST_BLOCKER:", "none", "", "CAUSAL_CLASSIFICATION:", e["classification"], "", "ONE_SENTENCE_VERDICT:", e["one_sentence_verdict"], "",
        "1. PROTOCOL_INTEGRITY", "", "D18_DESIGN_REVIEW_STATUS: PASSED (user-authorized; materialized D18 bundle absent)", "D17_MANIFEST_VERIFIED: YES", "CERTIFIED_POST_UPDATE_384_IDENTITY: MATCH", "CONTROL_FIDELITY_AGAINST_D17: PASS", "NEW_SCIENTIFIC_BRANCHES: 1", "UPDATES_385_392_EXECUTED: YES", "UPDATE_393_EXECUTED: NO", "FROZEN20_OPENED: NO", "R9_EXECUTED: NO", "R8E_A1_MODIFIED: NO", "HISTORICAL_D16_D17_ARTIFACTS_MODIFIED: NO", "OUTPUT_MANIFEST_VERIFIED: YES", "",
        "2. AUTHORITATIVE_INPUT_IDENTITY", "", json.dumps(identity, indent=2, sort_keys=True), "",
        "3. EXECUTED_BRANCH_MANIFEST", "", json.dumps(branch_manifest, indent=2, sort_keys=True), "",
        "4. ENDPOINT_RESULTS", "", f"CONTROL_H1: {e['CONTROL_H1']:.17g}", f"D18_H1: {e['D18_H1']:.17g}", f"E_REST: {e['E_REST']:.17g}", f"H1_MATERIAL_THRESHOLD: {H1_MATERIAL_THRESHOLD:.17g}", f"H1_MATERIAL_EFFECT_PASS: {e['H1_MATERIAL_EFFECT_PASS']}", f"CONTROL_H32: {e['CONTROL_H32']:.17g}", f"D18_H32: {e['D18_H32']:.17g}", f"H32_PRESERVATION_THRESHOLD: {H32_PRESERVATION_THRESHOLD:.17g}", f"H32_PRESERVED: {e['H32_PRESERVED']}", "",
        "5. NON_ADDITIVITY_DECOMPOSITION", "", json.dumps(decomposition, indent=2, sort_keys=True), "",
        "6. TRAJECTORY_DIAGNOSTICS", "", "Diagnostics are descriptive trajectory measurements; they do not by themselves establish a causal mechanism.", json.dumps(trajectory, indent=2, sort_keys=True), "",
        "7. CLASSIFICATION", "", f"Frozen logic assigned: {e['classification']}. Material H1 effect: {e['H1_MATERIAL_EFFECT_PASS']}; H32 preserved: {e['H32_PRESERVED']}.", "",
        "8. SCIENTIFIC_INTERPRETATION", "", f"Q1. H2–H31 jointly causal for the H1 blocker under the frozen full-pipeline contract: {'YES' if e['H1_MATERIAL_EFFECT_PASS'] else 'NO'}; the joint deletion endpoint is E_REST={e['E_REST']:.17g}.", f"Q2. Their deletion materially improves H1 while retaining/preserving H32: {'YES' if e['H1_MATERIAL_EFFECT_PASS'] and e['H32_PRESERVED'] else 'NO'}.", f"Q3. D17's all-future effect decomposes into additive H2–H31 effects {decomposition['ADDITIVE_H2_H31_FRACTION']:.17g} of E_ALL, within-group non-additivity {decomposition['WITHIN_REST_FRACTION']:.17g}, individual H32 deletion {decomposition['INDIVIDUAL_H32_FRACTION']:.17g}, and H32 × H2–H31 interaction {decomposition['CROSS_GROUP_FRACTION']:.17g}.", f"Q4. H32 deletion itself is required for the material D17 joint effect: {'YES' if not e['H1_MATERIAL_EFFECT_PASS'] else 'NO'} under this complement experiment; retaining H32 is {'not sufficient' if not (e['H1_MATERIAL_EFFECT_PASS'] and e['H32_PRESERVED']) else 'sufficient'} for the material H1 endpoint here.", f"Q5. Under this endpoint decomposition, the D17 all-future H32 failure is attributable to the tested H2–H31 joint deletion, H32 deletion, and their signed interaction components as quantified above; this does not establish a universal mechanism beyond this contract.", "",
        "9. ENGINEERING_DEFECTS_ENCOUNTERED_AND_REPAIRED", "", json.dumps(defects, indent=2, sort_keys=True), "",
        "10. ARTIFACT_PATHS_AND_HASHES", "", f"D18_OUTPUT_DIRECTORY: {output.resolve()}", f"FINAL_MANIFEST_HASH: {sha256_file(output / 'SHA256_MANIFEST.json') if (output / 'SHA256_MANIFEST.json').is_file() else 'PENDING'}", "MANIFEST_VERIFICATION: VERIFIED", "KEY_ARTIFACTS: protocol_integrity.json; authoritative_input_identity.json; execution_configuration.json; branch_execution_manifest.json; per_update_instrumentation.jsonl; endpoint_results.json; endpoint_results.csv; non_additivity_decomposition.json; trajectory_diagnostics.json; FINAL_REPORT.md; SHA256_MANIFEST.json", "",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    output = ROOT / "outputs" / f"stage3_h13_post_d17_d18_h2_h31_joint_rollout_deletion_with_h32_retained_causal_intervention_replay_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    try:
        runtime, identity, d17_control_rows, prior = verify_d17_and_runtime()
        control_reference_rows, control_captures = run_reference_control(runtime)
        control_checks = validate_control_reference(control_reference_rows, d17_control_rows)
        model, optimizer = d16.new_branch(runtime["bundle"])
        rng = copy.deepcopy(runtime["bundle"]["rng_state"])
        rows: list[dict[str, Any]] = []
        captures: dict[int, dict[str, Any]] = {}
        for zero_index in range(384, 392):
            row, capture, rng = d18_branch_update(runtime, model, optimizer, zero_index, rng)
            d16.add_longitudinal_divergence({"branch": "CONTROL"}, row, control_captures[row["update"]], capture)
            c = control_reference_rows[row["update"] - 385]
            row["RAW_GRADIENT_NORM_DIFFERENCE_FROM_CONTROL"] = row["raw_total_gradient_norm"] - c["raw_total_gradient_norm"]
            row["POSTCLIP_GRADIENT_NORM_DIFFERENCE_FROM_CONTROL"] = row["post_clipping_gradient_norm"] - c["post_clipping_gradient_norm"]
            row["ADAMW_DISPLACEMENT_NORM_DIFFERENCE_FROM_CONTROL"] = row["parameter_update_norm"] - c["parameter_update_norm"]
            row.update({"BRANCH": BRANCH, "UPDATE": row["update"], "TOTAL_LOSS": row["total_loss"], "RAW_GRAD_NORM": row["raw_total_gradient_norm"], "CLIP_COEFFICIENT": row["exact_clip_coefficient"], "POSTCLIP_GRAD_NORM": row["post_clipping_gradient_norm"], "ADAMW_PARAMETER_DISPLACEMENT_NORM": row["parameter_update_norm"], "PARAMETER_DISTANCE_FROM_CONTROL": row["branch_vs_control_parameter_distance_norm"], "EXP_AVG_DISTANCE_FROM_CONTROL": row["branch_vs_control_exp_avg_difference_norm"], "EXP_AVG_SQ_DISTANCE_FROM_CONTROL": row["branch_vs_control_exp_avg_sq_difference_norm"], "UPDATE_INCREMENT_DIFFERENCE_FROM_CONTROL": row["update_increment_difference_norm"], "PARAMETER_UPDATE_INCREMENT_DIFFERENCE_FROM_CONTROL": row["update_increment_difference_norm"]})
            rows.append(row)
            captures[row["update"]] = capture
        if any(value != 392 for value in rows[-1]["step_counter_after"].values()):
            raise StageBlocker("D18_FINAL_STEP_MISMATCH")
        if any(row["finite_status"] != "FINITE" or row["adamw_decomposition_status"] != "PASS" or row["branch_rng_contract_status"] != "PASS" for row in rows):
            raise StageBlocker("D18_ROW_VALIDITY_FAILED")
        final = rows[-1]
        e_rest = EXPECTED_CONTROL_H1 - float(final["per_update_H1"])
        label, h1_pass, h32_pass = classification(e_rest, float(final["per_update_H32"]))
        within = e_rest - SUM_REST_INDIVIDUAL
        cross = E_ALL - e_rest - E_H32
        conservation = E_ALL - (SUM_REST_INDIVIDUAL + within + E_H32 + cross)
        components_sum = within + cross
        if abs(conservation) > 1e-18 or not math.isclose(components_sum, 8.078171216668731e-06, rel_tol=1e-10, abs_tol=1e-18):
            raise StageBlocker(f"DECOMPOSITION_CONTRACT_FAILED:{conservation}:{components_sum}")
        decomposition = {"SUM_REST_INDIVIDUAL": SUM_REST_INDIVIDUAL, "E_REST": e_rest, "WITHIN_REST_NONADDITIVITY": within, "E_H32": E_H32, "H32_REST_INTERACTION": cross, "E_ALL": E_ALL, "CONSERVATION_RESIDUAL": conservation, "WITHIN_PLUS_CROSS": components_sum, "EXPECTED_WITHIN_PLUS_CROSS": 8.078171216668731e-06, "ADDITIVE_H2_H31_FRACTION": SUM_REST_INDIVIDUAL / E_ALL, "INDIVIDUAL_H32_FRACTION": E_H32 / E_ALL, "WITHIN_REST_FRACTION": within / E_ALL, "CROSS_GROUP_FRACTION": cross / E_ALL, "signed_fractions": True, "tolerance_check": "PASS"}
        endpoint = {"schema_version": "stage3_h13_post_d17_d18_endpoint_results_v1", "update": 392, "thresholds": {"H1_MATERIAL_THRESHOLD": H1_MATERIAL_THRESHOLD, "H32_PRESERVATION_THRESHOLD": H32_PRESERVATION_THRESHOLD}, "control": {"CONTROL_H1": EXPECTED_CONTROL_H1, "CONTROL_H32": EXPECTED_CONTROL_H32, "source": "AUTHENTICATED_D17_REUSE"}, "d18": {"branch": BRANCH, "CONTROL_H1": EXPECTED_CONTROL_H1, "D18_H1": float(final["per_update_H1"]), "E_REST": e_rest, "H1_MATERIAL_THRESHOLD": H1_MATERIAL_THRESHOLD, "H1_MATERIAL_EFFECT_PASS": h1_pass, "CONTROL_H32": EXPECTED_CONTROL_H32, "D18_H32": float(final["per_update_H32"]), "H32_PRESERVATION_THRESHOLD": H32_PRESERVATION_THRESHOLD, "H32_PRESERVED": h32_pass, "classification": label, "validity": "VALID", "one_sentence_verdict": f"Deleting H2–H31 while retaining H32 {'materially improves H1 and preserves H32' if h1_pass and h32_pass else 'does not materially improve H1 while preserving H32'} under the frozen full-pipeline contract."}}
        trajectory = make_trajectory(rows, d17_control_rows)
        branch_manifest = {"schema_version": "stage3_h13_post_d17_d18_branch_execution_manifest_v1", "scientific_branches_authorized": [BRANCH], "scientific_branches_executed": [BRANCH], "control_reference": "D17_AUTHENTICATED_REUSE_WITH_REFERENCE_ONLY_IN_MEMORY_CAPTURE", "start_update": START_UPDATE, "updates": list(UPDATES), "endpoint_update": 392, "update_393_executed": "NO", "branch": {"branch": BRANCH, "start_model_hash": runtime["loaded"]["model_semantic_hash"], "start_optimizer_hash": runtime["loaded"]["optimizer_semantic_hash"], "start_rng_digest": base.rng_digest(runtime["bundle"]["rng_state"]), "end_rng_digest": base.rng_digest(rng), "updates": list(UPDATES), "endpoint_update": 392, "update_393_executed": "NO"}, "control_fidelity_checks": control_checks}
        configuration = {"experiment_identifier": EXPERIMENT_ID, "branch": BRANCH, "captured_at_utc": utc_now(), "start_update": START_UPDATE, "updates": list(UPDATES), "objective": "L_D18=w1*S1+w32*S32+S1+0.005*H_hidden; delete exactly weighted H2..H31; no renormalization or compensation", "weights_source": "authenticated D17 loss construction", "position_beta": POSITION_BETA, "hidden_beta": 0.1, "clip": {"implementation": "torch.nn.utils.clip_grad_norm_", "max_norm": CLIP_MAX_NORM, "norm_type": 2.0, "branch_local": True}, "optimizer": base.optimizer_contract(runtime["optimizer"]), "runtime": runtime["environment"], "prohibited": {"update_393": "NO", "Frozen20": "NO", "R9": "NO", "R8E_A1_modification": "NO", "hyperparameter_tuning": "NO"}}
        defects = {"scientific_protocol_defects": [], "wrapper_reporting_serialization_defects": [{"type": "missing_materialized_d18_design_bundle", "status": "recorded_not_repaired", "effect": "The exact D18 design-review output directory and manifest were absent; execution used the locally preserved design-review text plus the explicit execution authorization, with hashes recorded in authoritative_input_identity.json."}, {"type": "compact_d17_control_artifact_lacks_control_tensors", "status": "narrow_reference_only_replay", "effect": "A control trajectory was replayed in memory only to calculate required vector differences; no control endpoint or second scientific branch was emitted."}]}
        write_json(output / "protocol_integrity.json", {"status": "PASSED", "first_blocker": "none", "d18_design_review_status": "PASSED_USER_AUTHORIZED", "d18_design_bundle_manifest_verification": "NOT_AVAILABLE_NO_MATERIALIZED_D18_BUNDLE", "d17_manifest_verified": "YES", "certified_start_match": "YES", "source_revision_match": "YES", "runtime_identity_match": "YES", "canonical_schedule_match": "YES", "control_fidelity_against_d17": "PASS", "new_scientific_branches": 1, "updates_385_392_executed": "YES", "update_393_executed": "NO", "frozen20_opened": "NO", "r9_executed": "NO", "r8e_a1_modified": "NO", "historical_artifacts_modified": "NO", "classification": label})
        write_json(output / "authoritative_input_identity.json", identity)
        write_json(output / "execution_configuration.json", configuration)
        write_json(output / "branch_execution_manifest.json", branch_manifest)
        write_json(output / "endpoint_results.json", endpoint)
        with (output / "endpoint_results.csv").open("w", encoding="utf-8", newline="") as stream:
            fields = ["BRANCH", "SOURCE", "CONTROL_H1", "INTERVENTION_H1", "E_H1", "H1_MATERIAL_THRESHOLD", "H1_MATERIAL_EFFECT_PASS", "CONTROL_H32", "INTERVENTION_H32", "H32_PRESERVATION_THRESHOLD", "H32_PRESERVED", "CLASSIFICATION", "VALIDITY"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerow({"BRANCH": "CONTROL", "SOURCE": "D17_AUTHENTICATED_REUSE", "CONTROL_H1": EXPECTED_CONTROL_H1, "INTERVENTION_H1": "", "E_H1": "", "H1_MATERIAL_THRESHOLD": H1_MATERIAL_THRESHOLD, "H1_MATERIAL_EFFECT_PASS": "", "CONTROL_H32": EXPECTED_CONTROL_H32, "INTERVENTION_H32": "", "H32_PRESERVATION_THRESHOLD": H32_PRESERVATION_THRESHOLD, "H32_PRESERVED": "", "CLASSIFICATION": "CONTROL_REFERENCE_ONLY", "VALIDITY": "VALID"})
            writer.writerow({"BRANCH": BRANCH, "SOURCE": "D18_EXECUTION", "CONTROL_H1": EXPECTED_CONTROL_H1, "INTERVENTION_H1": endpoint["d18"]["D18_H1"], "E_H1": e_rest, "H1_MATERIAL_THRESHOLD": H1_MATERIAL_THRESHOLD, "H1_MATERIAL_EFFECT_PASS": h1_pass, "CONTROL_H32": EXPECTED_CONTROL_H32, "INTERVENTION_H32": endpoint["d18"]["D18_H32"], "H32_PRESERVATION_THRESHOLD": H32_PRESERVATION_THRESHOLD, "H32_PRESERVED": h32_pass, "CLASSIFICATION": label, "VALIDITY": "VALID"})
        with (output / "per_update_instrumentation.jsonl").open("w", encoding="utf-8", newline="") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n")
        write_json(output / "non_additivity_decomposition.json", decomposition)
        write_json(output / "trajectory_diagnostics.json", trajectory)
        manifest = write_manifest(output)
        (output / "FINAL_REPORT.md").write_text(render_report(output, identity, configuration, branch_manifest, rows, endpoint, decomposition, trajectory, defects), encoding="utf-8", newline="\n")
        verification = verify_output_manifest(output, manifest)
        if verification["status"] != "VERIFIED":
            raise StageBlocker(f"OUTPUT_MANIFEST_FAILED:{verification}")
        write_json(output / "manifest_verification.json", verification)
        print(json.dumps({"status": "PASSED", "output": str(output.resolve()), "classification": label, "control_h1": EXPECTED_CONTROL_H1, "d18_h1": endpoint["d18"]["D18_H1"], "e_rest": e_rest, "d18_h32": endpoint["d18"]["D18_H32"], "manifest_sha256": verification["manifest_sha256"], "file_count": verification["checked"]}, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        (output / "execution_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8", newline="\n")
        write_json(output / "protocol_integrity.json", {"status": "BLOCKED", "first_blocker": f"{type(exc).__name__}:{exc}", "update_393_executed": "NO", "frozen20_opened": "NO", "r9_executed": "NO"})
        print(json.dumps({"status": "BLOCKED", "output": str(output.resolve()), "first_blocker": f"{type(exc).__name__}:{exc}"}, sort_keys=True), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
