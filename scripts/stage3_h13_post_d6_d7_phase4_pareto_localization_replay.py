"""Execute the single precommitted D7 phase-4 Pareto localization replay.

This runner starts from the certified post-update-384 bundle and executes only
the untouched canonical H32 continuation through update 512.  Validation is
measured at the frozen 17-point grid; no intermediate checkpoint search,
intervention, or model-selection action is performed.
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

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import stage3_h13_post_d5_d6_adamw_moment_reset_causal_branch as d6  # noqa: E402
from scripts import stage3_h13_post_d5_d6_branch_state_materialization_replay as base  # noqa: E402


EXPERIMENT_ID = "STAGE_3_H13_POST_D6_D7_PHASE4_PARETO_LOCALIZATION_REPLAY"
AUTHORIZATION_PATH = Path(r"C:\Users\86198\.codex\attachments\6073ffbf-5787-465c-a985-177dee0f245d\pasted-text.txt")
BUNDLE_PATH = ROOT / "outputs/stage3_h13_post_d5_d6_branch_state_materialization_replay_20260816T152000Z_retry/post_update_384_branch_bundle.pt"
BUNDLE_DIR = BUNDLE_PATH.parent
D7_DESIGN_DIR = ROOT / "outputs/stage3_h13_post_d6_d7_phase4_h1_h32_pareto_localization_design_review_20260817"
D7_FINAL_DIR = ROOT / "outputs/stage3_h13_post_d6_d7_final_scientific_interpretation_20260817"
EXPECTED_BUNDLE_SHA256 = "4a022aa0340977e571a3a3085e30d96fd0d92129f4fbeff99aaa8e3420379083"
EXPECTED_MODEL_HASH = "f6f450abd4ac8780a7172c46d44db88d6b7278e22e6a14580b62c9b1627bcd58"
EXPECTED_OPTIMIZER_HASH = "610dd9fa330572888e60cbc16e8f67560e2ce23a3b8628cad7481a90b24936d8"
EXPECTED_TERMINAL_MODEL_HASH = "281008502db053be3bf5945bd95a44391ef2f17c63385221cce61c178abb376e"
EXPECTED_BRANCH_UPDATE = 384
EXPECTED_TOTAL_UPDATES = 512
EXPECTED_STEP = 384
EXPECTED_BATCH_SIZE = 256
EXPECTED_CLIP = 1.0
EXPECTED_LR = 5.0e-5
EXPECTED_WEIGHT_DECAY = 1.0e-5
EXPECTED_BETAS = (0.9, 0.999)
EXPECTED_EPS = 1.0e-8
H1_THRESHOLD = 8.59791431235184e-05
R6_H32_REFERENCE = 0.0232112820732007
H32_THRESHOLD = 0.01856902565856056
GRID = (385, 392, 400, 408, 416, 424, 432, 440, 448, 456, 464, 472, 480, 488, 496, 504, 512)
EXPECTED_385 = {"H1": 7.771333127019206e-05, "H32": 0.018908132312382267}
EXPECTED_512 = {"H1": 9.019693982113058e-05, "H32": 0.016745967327457343}
REPLAY_RTOL = 5.0e-6
REPLAY_ATOL = 5.0e-9


class StageBlocker(RuntimeError):
    """Fail-closed D7 blocker."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def resolve_authority_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def verify_authority_manifest(path: Path, record_key: str) -> dict[str, Any]:
    if not path.is_file():
        raise StageBlocker(f"d7_authority_manifest_missing:{path}")
    manifest = base.load_json(path)
    records = manifest.get(record_key)
    if not isinstance(records, list):
        raise StageBlocker(f"d7_authority_manifest_records_missing:{path}:{record_key}")
    verified = []
    for record in records:
        source = resolve_authority_path(str(record["path"]))
        if not source.is_file():
            raise StageBlocker(f"d7_authority_input_missing:{source}")
        observed = sha256_file(source)
        if observed != record.get("sha256") or source.stat().st_size != int(record.get("bytes", record.get("size_bytes", source.stat().st_size))):
            raise StageBlocker(f"d7_authority_input_hash_mismatch:{source}")
        verified.append({"path": str(source.resolve()), "sha256": observed, "size_bytes": source.stat().st_size})
    return {"manifest_path": str(path.resolve()), "manifest_sha256": sha256_file(path), "record_key": record_key, "verified_count": len(verified), "records": verified}


def validate_current_authorization() -> dict[str, Any]:
    if not AUTHORIZATION_PATH.is_file():
        raise StageBlocker("current_d7_authorization_source_missing")
    text = AUTHORIZATION_PATH.read_text(encoding="utf-8")
    required = [
        EXPERIMENT_ID,
        "Execute D7 now.",
        "Do not open Frozen-20.",
        "Do not execute R9.",
        "Do not replace R8E_A1.",
        "385, 392, 400, 408, 416, 424, 432, 440, 448, 456, 464, 472, 480, 488, 496, 504, 512",
    ]
    missing = [item for item in required if item not in text]
    if missing:
        raise StageBlocker(f"authorization_scope_not_proven:{missing}")
    return {"path": str(AUTHORIZATION_PATH.resolve()), "sha256": sha256_file(AUTHORIZATION_PATH), "scope": EXPERIMENT_ID, "explicit_scope_match": "YES"}


def validate_d7_authority() -> dict[str, Any]:
    design = verify_authority_manifest(D7_DESIGN_DIR / "authority_manifest.json", "authority_inputs_hashed")
    final = verify_authority_manifest(D7_FINAL_DIR / "authority_manifest.json", "inputs")
    design_spec = base.load_json(D7_DESIGN_DIR / "prospective_d7_experiment_design.json")
    contract = base.load_json(D7_DESIGN_DIR / "prospective_d7_graduation_contract.json")
    if design_spec.get("status") != "PROSPECTIVE_DESIGN_ONLY_NOT_EXECUTED":
        raise StageBlocker("d7_design_not_prospective_preexecution_record")
    if design_spec.get("experiment_identifier") != EXPERIMENT_ID:
        raise StageBlocker("d7_design_experiment_identifier_mismatch")
    if tuple(int(x) for x in design_spec["predeclared_evaluation_grid"]["updates"]) != GRID:
        raise StageBlocker("d7_predeclared_grid_mismatch")
    if tuple(int(x) for x in contract["pointwise_rules"].get("_unused", [])):
        raise StageBlocker("unexpected_contract_mutation")
    thresholds = contract["numerical_thresholds"]
    if float(thresholds["h1_threshold"]) != H1_THRESHOLD or float(thresholds["h32_reference"]) != R6_H32_REFERENCE or float(thresholds["h32_threshold"]) != H32_THRESHOLD:
        raise StageBlocker("d7_threshold_contract_mismatch")
    return {"design_authority": design, "final_interpretation_authority": final, "design": design_spec, "contract": contract}


def validate_bundle(bundle: Mapping[str, Any], authority: Mapping[str, Any]) -> dict[str, Any]:
    if sha256_file(BUNDLE_PATH) != EXPECTED_BUNDLE_SHA256:
        raise StageBlocker("certified_branch_bundle_hash_mismatch")
    position = bundle.get("training_position", {})
    if position.get("completed_optimizer_steps") != EXPECTED_BRANCH_UPDATE or position.get("next_completed_optimizer_step") != 385 or position.get("next_zero_based_schedule_index") != EXPECTED_BRANCH_UPDATE or position.get("remaining_updates") != 128:
        raise StageBlocker("certified_update_384_position_mismatch")
    if position.get("remaining_schedule_sha256") != authority["schedule_slice_hashes"]["suffix_128"]:
        raise StageBlocker("certified_remaining_schedule_hash_mismatch")
    if position.get("next_batch_window_sha256") != base.batch_digest(authority["schedule_manifest"], EXPECTED_BRANCH_UPDATE):
        raise StageBlocker("certified_next_batch_identity_mismatch")
    expected_config = base.load_json(BUNDLE_DIR / "training_config.json")
    if bundle.get("training_config") != expected_config:
        raise StageBlocker("certified_bundle_training_config_mismatch")
    expected_evaluator = {"split": "H10 VALIDATION", "horizons": [1, 2, 4, 8, 16, 32], "free_running": True, "future_label_leakage": 0}
    if bundle.get("evaluator_config") != expected_evaluator:
        raise StageBlocker("certified_evaluator_identity_mismatch")
    provenance = bundle.get("provenance", {})
    if provenance.get("source_revision") != base.EXPECTED_SOURCE_COMMIT:
        raise StageBlocker("certified_source_revision_mismatch")
    if provenance.get("canonical_schedule_sha256") != base.EXPECTED_CANONICAL_SCHEDULE or provenance.get("canonical_schedule_suffix_128_sha256") != authority["schedule_slice_hashes"]["suffix_128"]:
        raise StageBlocker("certified_schedule_provenance_mismatch")
    if bundle.get("optimizer_contract", {}).get("class") != "AdamW" or bundle.get("optimizer_contract", {}).get("amsgrad") is not False:
        raise StageBlocker("certified_optimizer_contract_mismatch")
    config = bundle["training_config"]
    expected_recipe = {
        "architecture": "2-layer unidirectional residual GRU, hidden 128, decoder horizon 8",
        "batch_size": EXPECTED_BATCH_SIZE,
        "learning_rate": EXPECTED_LR,
        "betas": list(EXPECTED_BETAS),
        "eps": EXPECTED_EPS,
        "weight_decay": EXPECTED_WEIGHT_DECAY,
        "gradient_clipping": {"enabled": True, "operation": "torch.nn.utils.clip_grad_norm_", "max_norm": EXPECTED_CLIP},
        "lambda_h1": 1.0,
        "lambda_hidden": 0.005,
        "huber_beta": 0.001,
        "scheduler": "ABSENT_BY_DESIGN",
        "future_label_leakage": 0,
    }
    for key, expected in expected_recipe.items():
        if config.get(key) != expected:
            raise StageBlocker(f"canonical_recipe_mismatch:{key}")
    return {
        "raw_sha256": sha256_file(BUNDLE_PATH),
        "state_hashes": bundle.get("state_hashes"),
        "optimizer_parameter_mapping": bundle.get("optimizer_parameter_mapping"),
        "optimizer_contract": bundle.get("optimizer_contract"),
        "training_config": config,
        "evaluator_config": expected_evaluator,
        "training_position": position,
        "provenance": provenance,
    }


def state_is_finite(model: Any, optimizer: torch.optim.Optimizer) -> bool:
    if any(not bool(torch.all(torch.isfinite(parameter.detach()))) for parameter in model.parameters()):
        return False
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor) and not bool(torch.all(torch.isfinite(value))):
                return False
    return True


def grid_classification(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: int(row["UPDATE"]))
    double = [int(row["UPDATE"]) for row in ordered if row["DOUBLE_PASS"] == "PASS"]
    h32 = [int(row["UPDATE"]) for row in ordered if row["H32_PASS"] == "PASS"]
    h1_fail = [int(row["UPDATE"]) for row in ordered if row["H1_PASS"] == "FAIL"]
    indices = {int(update): index for index, update in enumerate(GRID)}
    contiguous = False
    region = "NONE"
    if len(double) >= 2:
        contiguous = all(indices[double[i + 1]] == indices[double[i]] + 1 for i in range(len(double) - 1))
        region = f"{double[0]}-{double[-1]}" if contiguous else "NONCONTIGUOUS"
    isolated = len(double) == 1 and not any(abs(indices[double[0]] - indices[other]) == 1 for other in double if other != double[0])
    return {
        "FIRST_H32_GRADUATING_UPDATE": min(h32) if h32 else "NONE",
        "FIRST_H1_FAILING_UPDATE": min(h1_fail) if h1_fail else "NONE",
        "FIRST_DOUBLE_PASS_UPDATE": min(double) if double else "NONE",
        "LAST_DOUBLE_PASS_UPDATE": max(double) if double else "NONE",
        "DOUBLE_PASS_COUNT": len(double),
        "DOUBLE_PASS_UPDATES": double,
        "CONTIGUOUS_DOUBLE_PASS_GRID_REGION": region,
        "CONTIGUOUS_DOUBLE_PASS_GRID_STATES": contiguous,
        "ISOLATED_DOUBLE_PASS_ONLY": isolated,
        "PARETO_WINDOW_FOUND": bool(double),
        "H32_GRADUATES_WITH_H1_PASS_BUT_NOT_ROBUSTLY_ESTABLISHED": "YES" if isolated else "NO",
    }


def classify(rows: Sequence[Mapping[str, Any]], protocol_valid: bool) -> dict[str, Any]:
    summary = grid_classification(rows)
    if not protocol_valid:
        primary = "INCONCLUSIVE"
    elif summary["DOUBLE_PASS_COUNT"] > 0:
        primary = "PARETO_WINDOW_FOUND"
    else:
        h32_rows = [row for row in rows if row["H32_PASS"] == "PASS"]
        if h32_rows and all(row["H1_PASS"] == "FAIL" for row in h32_rows):
            primary = "H1_FAILS_BEFORE_H32_GRADUATES"
        else:
            update_384_h1 = 0.00007383130315538692
            sustained_h1_regression = any(
                all(float(rows[i + j]["H1"]) > update_384_h1 for j in range(2))
                for i in range(max(0, len(rows) - 1))
            ) if len(rows) >= 2 else False
            primary = "NO_PHASE4_TRADEOFF_EVIDENCE" if not h32_rows and not sustained_h1_regression else "INCONCLUSIVE"
            summary["sustained_h1_regression_relative_to_update_384"] = sustained_h1_regression
    summary["D7_PRIMARY_CLASSIFICATION"] = primary
    return summary


def anchor_record(grid_rows: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for update, expected in ((385, EXPECTED_385), (512, EXPECTED_512)):
        observed = grid_rows[update]
        differences = {key: float(observed[key]) - value for key, value in expected.items()}
        checks = {key: math.isclose(float(observed[key]), value, rel_tol=REPLAY_RTOL, abs_tol=REPLAY_ATOL) for key, value in expected.items()}
        result[str(update)] = {"expected": expected, "observed": {key: float(observed[key]) for key in expected}, "differences": differences, "within_authoritative_tolerance": all(checks.values()), "per_metric": checks}
    result["update_512_terminal_model_semantic_hash_expected"] = EXPECTED_TERMINAL_MODEL_HASH
    result["update_512_terminal_model_semantic_hash_observed"] = grid_rows[512]["MODEL_STATE_DIGEST"]
    result["update_512_model_identity_match"] = grid_rows[512]["MODEL_STATE_DIGEST"] == EXPECTED_TERMINAL_MODEL_HASH
    result["anchors_reproduced"] = bool(result["385"]["within_authoritative_tolerance"] and result["512"]["within_authoritative_tolerance"] and result["update_512_model_identity_match"])
    return result


def run_d7_continuation(model: Any, optimizer: torch.optim.Optimizer, pre: Mapping[str, Any], authority: Mapping[str, Any]) -> dict[str, Any]:
    schedule_rows = authority["schedule_rows"]
    schedule_manifest = authority["schedule_manifest"]
    train_data = pre["train_data"]
    validation = pre["validation"]
    channels = pre["stats"]["channels"]
    parameter_rows = base.parameter_rows(model)
    rows: list[dict[str, Any]] = []
    module_rows: list[dict[str, Any]] = []
    evaluations: dict[int, dict[str, Any]] = {}
    digest_manifest: dict[str, Any] = {}
    leakage_events = 0
    model.train()
    for zero_based_index in range(EXPECTED_BRANCH_UPDATE, EXPECTED_TOTAL_UPDATES):
        step = zero_based_index + 1
        horizon = int(base.PHASE_HORIZONS[zero_based_index // 128])
        batch_hash = base.batch_digest(schedule_manifest, zero_based_index)
        batch = base.tensor_batch(train_data, schedule_rows[zero_based_index])
        optimizer.zero_grad(set_to_none=True)
        rollout = base.causal_paired_rollout(model, batch["inputs"], batch["history_positions"], batch["history_times"], batch["target_times"], batch["starts"], batch["ends"], channels, horizon=horizon, target_positions_for_teacher=batch["target_positions"], hidden_consistency_enabled=True)
        if rollout.get("free_branch_reads_target_positions") is not False:
            leakage_events += 1
            raise StageBlocker(f"future_label_leakage_detected_update_{step}")
        losses = base.training_losses(rollout["predictions"], batch["target_positions"], rollout["hidden_loss"], lambda_hidden=0.005, lambda_h1=1.0, beta=0.001)
        if not all(bool(torch.isfinite(value)) for value in losses.values()):
            raise StageBlocker(f"nonfinite_training_loss_update_{step}")
        losses["total"].backward()
        if not base.gradients_are_finite(model):
            raise StageBlocker(f"nonfinite_gradient_update_{step}")
        before = {name: parameter.detach().cpu().clone() for name, parameter in parameter_rows}
        raw_grads = {name: parameter.grad.detach().cpu().clone() for name, parameter in parameter_rows if parameter.grad is not None}
        raw_norm = base.aggregate_norm(raw_grads.values())
        raw_digest = d6.named_tensor_digest(raw_grads)
        pre_clip_norm = base.finite_float(torch.nn.utils.clip_grad_norm_(model.parameters(), EXPECTED_CLIP))
        if not math.isclose(raw_norm, pre_clip_norm, rel_tol=1.0e-5, abs_tol=1.0e-5):
            raise StageBlocker(f"gradient_norm_instrumentation_mismatch_update_{step}")
        clipped_grads = {name: parameter.grad.detach().cpu().clone() for name, parameter in parameter_rows if parameter.grad is not None}
        clipped_digest = d6.named_tensor_digest(clipped_grads)
        clipped_norm = base.aggregate_norm(clipped_grads.values())
        clipping_activated = pre_clip_norm > EXPECTED_CLIP
        clipping_coefficient = min(1.0, EXPECTED_CLIP / pre_clip_norm) if pre_clip_norm > 0.0 else 1.0
        optimizer.step()
        after = {name: parameter.detach().cpu().clone() for name, parameter in parameter_rows}
        update_tensors = {name: after[name] - before[name] for name, _ in parameter_rows}
        if not state_is_finite(model, optimizer):
            raise StageBlocker(f"nonfinite_model_or_optimizer_state_update_{step}")
        update_norm = base.aggregate_norm(update_tensors.values())
        parameter_norm_before = base.aggregate_norm(before.values())
        max_tensor_update_norm = max(base.aggregate_norm([value]) for value in update_tensors.values())
        max_abs_update = max(float(torch.max(torch.abs(value)).item()) for value in update_tensors.values())
        model_hash = base.canonical_model_state_sha256(model.state_dict())
        optimizer_diag = base.aggregate_optimizer_state(model, optimizer, before, EXPECTED_LR, EXPECTED_WEIGHT_DECAY, EXPECTED_EPS, EXPECTED_BETAS)
        record: dict[str, Any] = {
            "branch": "CONTROL",
            "update": step,
            "zero_based_schedule_index": zero_based_index,
            "active_training_phase": "phase_4_H32",
            "active_horizon": horizon,
            "learning_rate": EXPECTED_LR,
            "scheduler": "ABSENT_BY_DESIGN",
            "batch_window_sha256": batch_hash,
            "training_loss_total": base.finite_float(losses["total"]),
            "training_loss_rollout": base.finite_float(losses["rollout"]),
            "training_loss_hidden": base.finite_float(losses["hidden"]),
            "training_loss_h1": base.finite_float(losses["h1"]),
            "raw_gradient_norm_before_clipping": raw_norm,
            "clipped_gradient_norm_after_clipping": clipped_norm,
            "clipping_activated": "YES" if clipping_activated else "NO",
            "clipping_coefficient": clipping_coefficient,
            "gradient_nonfinite": "NO",
            "parameter_update_norm": update_norm,
            "relative_parameter_update_norm": update_norm / max(parameter_norm_before, 1.0e-30),
            "max_parameter_tensor_update_norm": max_tensor_update_norm,
            "max_parameter_update_abs": max_abs_update,
            "parameter_update_nonfinite": "NO",
            "parameter_update_digest": d6.named_tensor_digest(update_tensors),
            "raw_gradient_digest": raw_digest,
            "clipped_gradient_digest": clipped_digest,
            "model_state_digest": model_hash,
            "nonfinite_events": 0,
            "future_label_leakage": 0,
            **optimizer_diag,
        }
        module_rows.extend(dict(item, branch="CONTROL") for item in d6.module_update_rows(model, optimizer, raw_grads, before, step))
        if step in GRID:
            metrics = d6.eval_metrics(model, validation, channels)
            snapshot = d6.optimizer_snapshot(optimizer, model, base.rng_digest(base.capture_rng()))
            if not all(math.isfinite(float(metrics[key])) for key in ("1", "32")):
                raise StageBlocker(f"nonfinite_validation_measurement_update_{step}")
            row = {
                "UPDATE": step,
                "H1": float(metrics["1"]),
                "H1_THRESHOLD": H1_THRESHOLD,
                "H1_MARGIN": H1_THRESHOLD - float(metrics["1"]),
                "H1_PASS": "PASS" if math.isfinite(float(metrics["1"])) and float(metrics["1"]) <= H1_THRESHOLD else "FAIL",
                "H32": float(metrics["32"]),
                "H32_THRESHOLD": H32_THRESHOLD,
                "H32_MARGIN": H32_THRESHOLD - float(metrics["32"]),
                "H32_PASS": "PASS" if math.isfinite(float(metrics["32"])) and float(metrics["32"]) <= H32_THRESHOLD else "FAIL",
                "DOUBLE_PASS": "PASS" if float(metrics["1"]) <= H1_THRESHOLD and float(metrics["32"]) <= H32_THRESHOLD else "FAIL",
                "ACTIVE_TRAINING_PHASE": "phase_4_H32",
                "LEARNING_RATE": EXPECTED_LR,
                "OPTIMIZER_STEP": int(d6.state_step_range(optimizer)["min"]),
                "GRADIENT_NORM_BEFORE_CLIPPING": raw_norm,
                "GRADIENT_NORM_AFTER_CLIPPING": clipped_norm,
                "PARAMETER_UPDATE_NORM": update_norm,
                "MODEL_STATE_DIGEST": model_hash,
                "OPTIMIZER_STATE_DIGEST": snapshot["optimizer_semantic_hash"],
                "EXP_AVG_DIGEST": snapshot["exp_avg_semantic_hash"],
                "EXP_AVG_SQ_DIGEST": snapshot["exp_avg_sq_semantic_hash"],
                "FINITE_STATUS": "FINITE",
                "FUTURE_LABEL_LEAKAGE": 0,
            }
            evaluations[step] = {"update": step, "metrics": {f"H{key}": float(value) for key, value in metrics.items()}, "trajectory_row": row}
            digest_manifest[str(step)] = {
                "update": step,
                "model_state_digest": model_hash,
                "optimizer_state_digest": snapshot["optimizer_semantic_hash"],
                "optimizer_step_digest": snapshot["optimizer_step_semantic_hash"],
                "exp_avg_digest": snapshot["exp_avg_semantic_hash"],
                "exp_avg_sq_digest": snapshot["exp_avg_sq_semantic_hash"],
                "parameter_groups_digest": snapshot["parameter_groups_semantic_hash"],
                "rng_state_digest": snapshot["rng_state_sha256"],
                "scheduler_state": "NOT_APPLICABLE",
                "state_step_range": snapshot["step_range"],
                "finite_status": "FINITE",
            }
            record.update({f"eval_H{key}": value for key, value in metrics.items()})
        rows.append(record)
    if sorted(evaluations) != list(GRID):
        raise StageBlocker("predeclared_grid_measurement_missing")
    rng_after = base.rng_digest(base.capture_rng())
    terminal = {
        "metrics": evaluations[512]["trajectory_row"],
        "model_state": copy.deepcopy(model.state_dict()),
        "optimizer_state": copy.deepcopy(optimizer.state_dict()),
        "model_semantic_hash": base.canonical_model_state_sha256(model.state_dict()),
        "optimizer_semantic_hash": base.optimizer_semantic_hash(optimizer),
        "optimizer_step_range": d6.state_step_range(optimizer),
        "rng_after_digest": rng_after,
    }
    digest_manifest["terminal"] = {
        "completed_optimizer_steps": EXPECTED_TOTAL_UPDATES,
        "model_state_digest": terminal["model_semantic_hash"],
        "optimizer_state_digest": terminal["optimizer_semantic_hash"],
        "optimizer_step_range": terminal["optimizer_step_range"],
        "rng_after_digest": rng_after,
        "finite_status": "FINITE",
    }
    return {"rows": rows, "module_rows": module_rows, "evaluations": evaluations, "digest_manifest": digest_manifest, "terminal": terminal, "future_label_leakage": leakage_events, "rng_after_digest": rng_after}


def protocol_report(pre: Mapping[str, Any], authority: Mapping[str, Any], d7_authority: Mapping[str, Any], bundle_identity: Mapping[str, Any], loaded_identity: Mapping[str, Any], authorization: Mapping[str, Any], a1_hash_before: str) -> dict[str, Any]:
    current_env = authority["current_environment"]
    config = bundle_identity["training_config"]
    return {
        "schema_version": "stage3_h13_d7_protocol_integrity_identity_v1",
        "experiment_identifier": EXPERIMENT_ID,
        "preflight_completed_before_training": True,
        "certified_update_384_model_state_identity": loaded_identity["model_semantic_hash"],
        "certified_update_384_optimizer_state_identity": loaded_identity["optimizer_semantic_hash"],
        "certified_exp_avg_digest": bundle_identity["state_hashes"]["exp_avg_semantic_hash"],
        "certified_exp_avg_sq_digest": bundle_identity["state_hashes"]["exp_avg_sq_semantic_hash"],
        "certified_optimizer_step_counters": loaded_identity["optimizer_step_range"],
        "scheduler_lr_state": {"scheduler": "ABSENT_BY_DESIGN", "scheduler_state": "NOT_APPLICABLE", "learning_rate": config["learning_rate"]},
        "canonical_batch_schedule_identity": {"full_sha256": authority["schedule_hashes"]["logical_sha256"], "storage_sha256": authority["schedule_hashes"]["storage_sha256"], "suffix_128_sha256": authority["schedule_slice_hashes"]["suffix_128"], "next_batch_sha256": bundle_identity["training_position"]["next_batch_window_sha256"]},
        "source_revision": {"expected": base.EXPECTED_SOURCE_COMMIT, "observed": current_env["source_revision"], "match": current_env["source_revision"] == base.EXPECTED_SOURCE_COMMIT},
        "model_architecture": {"declared": config["architecture"], "actual_class": "ResidualCausalGRUTrajectoryPredictor", "parameter_mapping_digest": sha256_bytes(canonical_json(loaded_identity["optimizer_step_range"]["by_parameter"]))},
        "loss_configuration": {"loss": config["loss"], "lambda_h1": config["lambda_h1"], "lambda_hidden": config["lambda_hidden"], "huber_beta": config["huber_beta"], "phase_horizons": config["phase_horizons"]},
        "gradient_clipping": config["gradient_clipping"],
        "weight_decay": config["weight_decay"],
        "learning_rate": config["learning_rate"],
        "optimizer_type": loaded_identity["optimizer_contract"],
        "evaluator_semantics": bundle_identity["evaluator_config"],
        "validation_dataset_identity": {"split": "H10 VALIDATION", "window_count": int(pre["validation"].count), "source_authority_status": "PASSED", "normalization_source": "H11 TRAIN-only normalization"},
        "rng_state_required_for_exact_continuation": {"bundle_digest": bundle_identity["state_hashes"]["rng_state_sha256"], "bundle_state_digest_recomputed": "RECORDED_AFTER_LOAD"},
        "frozen20": {"opened": "NO", "used": "NO", "status": "SEALED"},
        "future_label_leakage": 0,
        "r8e_a1": {"current_validation_champion": "R8E_A1", "raw_sha256_before": a1_hash_before, "expected_raw_sha256": base.EXPECTED_A1_HASH, "immutable_before_execution": a1_hash_before == base.EXPECTED_A1_HASH},
        "d7_authority": {"design_manifest_sha256": d7_authority["design_authority"]["manifest_sha256"], "final_interpretation_manifest_sha256": d7_authority["final_interpretation_authority"]["manifest_sha256"], "design_records_verified": True},
        "canonical_recipe_unchanged": True,
        "single_allowed_axis": "checkpoint time along untouched canonical H32 continuation",
        "predeclared_grid": list(GRID),
        "adaptive_narrowing": "FORBIDDEN",
        "retrospective_search": "FORBIDDEN",
        "authorization": authorization,
    }


def execution_configuration(bundle_identity: Mapping[str, Any], authority: Mapping[str, Any], authorization: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "stage3_h13_d7_execution_configuration_v1",
        "experiment_identifier": EXPERIMENT_ID,
        "execution_status": "EXECUTED_ONCE",
        "start_state": {"path": str(BUNDLE_PATH.resolve()), "sha256": bundle_identity["raw_sha256"], "completed_optimizer_steps": EXPECTED_BRANCH_UPDATE},
        "continuation": {"updates": 128, "zero_based_schedule_indices": [EXPECTED_BRANCH_UPDATE, EXPECTED_TOTAL_UPDATES - 1], "evaluation_grid": list(GRID), "active_phase": "phase_4_H32"},
        "recipe": bundle_identity["training_config"],
        "evaluator": bundle_identity["evaluator_config"],
        "canonical_schedule": {"logical_sha256": authority["schedule_hashes"]["logical_sha256"], "storage_sha256": authority["schedule_hashes"]["storage_sha256"], "suffix_sha256": authority["schedule_slice_hashes"]["suffix_128"]},
        "diagnostic_axis_only": "checkpoint time",
        "authorization": authorization,
        "device_semantics": {"device": "cpu", "dtype": "float32", "deterministic_algorithms": True, "dataloader_workers": 0},
    }


def render_report(certificate: Mapping[str, Any], trajectory: Sequence[Mapping[str, Any]], artifacts: Mapping[str, str]) -> str:
    lines = [
        f"STAGE_3_H13_POST_D6_D7_PHASE4_PARETO_LOCALIZATION_REPLAY: {certificate['status']}",
        f"FIRST_BLOCKER: {certificate['first_blocker']}",
        f"D7_PRIMARY_CLASSIFICATION: {certificate['classification']['D7_PRIMARY_CLASSIFICATION']}",
        f"ONE_SENTENCE_VERDICT: {certificate['one_sentence_verdict']}",
        "",
        "==================================================", "", "1. PROTOCOL INTEGRITY", "",
    ]
    for key, value in certificate["protocol_gates"].items():
        lines.append(f"{key}: {value}")
    lines += ["", "==================================================", "", "2. REPLAY ANCHOR VERIFICATION", ""]
    for update in (385, 512):
        anchor = certificate["anchor_verification"][str(update)]
        lines.append(f"UPDATE_{update}: observed H1={anchor['observed']['H1']}; authoritative H1={anchor['expected']['H1']}; difference={anchor['differences']['H1']}; observed H32={anchor['observed']['H32']}; authoritative H32={anchor['expected']['H32']}; difference={anchor['differences']['H32']}; within tolerance={anchor['within_authoritative_tolerance']}")
    lines.append(f"UPDATE_512_MODEL_IDENTITY_MATCH: {certificate['anchor_verification']['update_512_model_identity_match']}")
    lines += ["", "==================================================", "", "3. PREDECLARED 17-POINT TRAJECTORY", "", "| UPDATE | H1 | H1_THRESHOLD | H1_MARGIN | H1_PASS | H32 | H32_THRESHOLD | H32_MARGIN | H32_PASS | DOUBLE_PASS |", "|---:|---:|---:|---:|:---:|---:|---:|---:|:---:|:---:|"]
    for row in trajectory:
        lines.append(f"| {row['UPDATE']} | {row['H1']:.17g} | {row['H1_THRESHOLD']:.17g} | {row['H1_MARGIN']:.17g} | {row['H1_PASS']} | {row['H32']:.17g} | {row['H32_THRESHOLD']:.17g} | {row['H32_MARGIN']:.17g} | {row['H32_PASS']} | {row['DOUBLE_PASS']} |")
    lines += ["", "==================================================", "", "4. PARETO RESULT", ""]
    for key in ("FIRST_H32_GRADUATING_UPDATE", "FIRST_H1_FAILING_UPDATE", "FIRST_DOUBLE_PASS_UPDATE", "LAST_DOUBLE_PASS_UPDATE", "DOUBLE_PASS_COUNT", "DOUBLE_PASS_UPDATES", "CONTIGUOUS_DOUBLE_PASS_GRID_REGION", "ISOLATED_DOUBLE_PASS_ONLY", "PARETO_WINDOW_FOUND"):
        lines.append(f"{key}: {certificate['classification'][key]}")
    lines += ["", "==================================================", "", "5. SCIENTIFIC INTERPRETATION", "", "Does the measured canonical H32 continuation contain a predeclared intermediate state satisfying both H1 and H32 requirements?", "", f"Answer: {'YES' if certificate['classification']['PARETO_WINDOW_FOUND'] else 'NO on the predeclared grid'}.", "", "Direct observations: all 17 predeclared grid measurements are finite and the anchors are checked against the authoritative update-385/update-512 values.", "Temporal localization: the result is localized only to the measured grid states along the untouched phase-4 continuation.", "Causal evidence: this is checkpoint-timing evidence under a fixed continuation; it does not isolate the H32 objective as the unique cause of H1 regression.", "Unresolved: cross-seed robustness, independent-holdout generalization, production superiority, hidden-state interaction, and other phase-confounded mechanisms remain unresolved."]
    lines += ["", "==================================================", "", "6. MODEL-SELECTION FIREWALL", "", "CURRENT_VALIDATION_CHAMPION: R8E_A1", "CHAMPION_CHANGED: NO", "D7_DIAGNOSTIC_ONLY: YES", "QUALIFYING_CHECKPOINT_AUTOMATIC_CHAMPION: NO", "FROZEN20_OPENED: NO", "FROZEN20_USED: NO", "R9_EXECUTED: NO", "AUTHORIZED_FOR_R9: NO"]
    lines += ["", "==================================================", "", "7. ARTIFACTS / HASHES", ""]
    for name, digest in artifacts.items():
        lines.append(f"{name}: {digest}")
    lines += ["", "Self-hashes are excluded where required to avoid circular mutation; SHA256_MANIFEST.json is the complete machine-readable file ledger excluding itself and this report.", "", "==================================================", "", "8. NEXT SCIENTIFIC DECISION", "", f"{certificate['one_next_action']}", "", "FINAL_SCIENTIFIC_CLASSIFICATION:", certificate["classification"]["D7_PRIMARY_CLASSIFICATION"], "", "ONE_NEXT_ACTION:", certificate["one_next_action"]]
    return "\n".join(lines) + "\n"


def write_sha256_manifest(output: Path) -> dict[str, str]:
    rows = {}
    for path in sorted(item for item in output.rglob("*") if item.is_file() and item.name != "SHA256_MANIFEST.json"):
        rows[str(path.relative_to(output)).replace("\\", "/")] = sha256_file(path)
    write_json(output / "SHA256_MANIFEST.json", {"schema_version": "stage3_h13_d7_sha256_manifest_v1", "hash_algorithm": "SHA-256", "self_excluded": True, "report_excluded_from_ledger": True, "files": rows})
    return rows


def fail_closed(output: Path, blocker: str, authorization: Mapping[str, Any] | None = None) -> int:
    output.mkdir(parents=True, exist_ok=True)
    certificate = {
        "schema_version": "stage3_h13_d7_terminal_certificate_v1",
        "status": "BLOCKED",
        "first_blocker": blocker,
        "classification": {"D7_PRIMARY_CLASSIFICATION": "INCONCLUSIVE"},
        "protocol_gates": {"FAIL_CLOSED": "YES", "FROZEN20_OPENED": "NO", "FROZEN20_USED": "NO", "R9_EXECUTED": "NO", "R8E_A1_REMAINS_CURRENT_VALIDATION_CHAMPION": "YES", "CHAMPION_CHANGED": "NO"},
        "authorization": authorization,
        "training_started": False,
    }
    write_json(output / "terminal_certificate.json", certificate)
    (output / "FINAL_REPORT.md").write_text("STAGE_3_H13_POST_D6_D7_PHASE4_PARETO_LOCALIZATION_REPLAY: BLOCKED\nFIRST_BLOCKER: " + blocker + "\nD7_PRIMARY_CLASSIFICATION: INCONCLUSIVE\nONE_SENTENCE_VERDICT: No scientific conclusion is permitted because the preflight failed closed.\n", encoding="utf-8", newline="\n")
    write_sha256_manifest(output)
    print(json.dumps({"status": "BLOCKED", "output": str(output.resolve()), "blocker": blocker}, sort_keys=True), flush=True)
    return 2


def execute(output: Path) -> int:
    if output.exists():
        return fail_closed(output, f"output_directory_already_exists_no_resume:{output}")
    output.mkdir(parents=True, exist_ok=False)
    authorization = None
    try:
        authorization = validate_current_authorization()
        d7_authority = validate_d7_authority()
        pre = base.preflight(True)
        authority = base.verify_authority(pre)
        current_env = authority["current_environment"]
        if current_env.get("source_revision") != base.EXPECTED_SOURCE_COMMIT or current_env.get("device") != "cpu" or current_env.get("dtype") != "float32" or current_env.get("deterministic_algorithms") is not True:
            raise StageBlocker("execution_environment_or_source_identity_mismatch")
        a1_hash_before = sha256_file(base.A1_CHECKPOINT)
        if a1_hash_before != base.EXPECTED_A1_HASH:
            raise StageBlocker("r8e_a1_immutability_gate_failed_before_execution")
        bundle = torch.load(BUNDLE_PATH, map_location="cpu", weights_only=False)
        bundle_identity = validate_bundle(bundle, authority)
        if bundle_identity["training_position"]["completed_optimizer_steps"] != EXPECTED_BRANCH_UPDATE:
            raise StageBlocker("certified_start_state_not_update_384")
        base.set_deterministic(base.SEED)
        model, optimizer = base.load_control_from_bundle(copy.deepcopy(bundle))
        loaded_identity = d6.validate_loaded_bundle(bundle, model, optimizer)
        bundle_rng_digest = base.rng_digest(bundle["rng_state"])
        if bundle_rng_digest != bundle["state_hashes"]["rng_state_sha256"]:
            raise StageBlocker("certified_rng_state_digest_mismatch")
        base.restore_rng(bundle["rng_state"])
        rng_before = base.rng_digest(base.capture_rng())
        if rng_before != bundle_rng_digest:
            raise StageBlocker("certified_rng_state_restore_mismatch")
        protocol = protocol_report(pre, authority, d7_authority, bundle_identity, loaded_identity, authorization, a1_hash_before)
        protocol["rng_state_required_for_exact_continuation"]["restored_digest"] = rng_before
        protocol["rng_state_required_for_exact_continuation"]["exact_restore_match"] = True
        write_json(output / "protocol_integrity_identity.json", protocol)
        write_json(output / "execution_configuration_snapshot.json", execution_configuration(bundle_identity, authority, authorization))
        write_json(output / "source_revision_record.json", {"source_revision": current_env["source_revision"], "expected_source_revision": base.EXPECTED_SOURCE_COMMIT, "match": True, "environment": current_env, "captured_at_utc": utc_now()})
        write_json(output / "preflight_identity.json", {"authorization": authorization, "d7_authority": d7_authority, "canonical_authority": {"schedule_hashes": authority["schedule_hashes"], "schedule_slice_hashes": authority["schedule_slice_hashes"], "snapshot_count": len(authority["authority_snapshot"])}, "bundle": bundle_identity, "loaded_bundle": loaded_identity, "rng_before": rng_before, "frozen20": {"opened": "NO", "used": "NO", "status": "SEALED"}, "future_label_leakage": 0, "current_validation_champion": "R8E_A1"})
        result = run_d7_continuation(model, optimizer, pre, authority)
        if result["future_label_leakage"] != 0 or result["rng_after_digest"] != rng_before:
            raise StageBlocker("rng_or_future_label_leakage_gate_failed_during_continuation")
        trajectory = [result["evaluations"][update]["trajectory_row"] for update in GRID]
        grid_rows = {int(row["UPDATE"]): row for row in trajectory}
        classification = classify(trajectory, protocol_valid=True)
        anchors = anchor_record(grid_rows)
        if not anchors["anchors_reproduced"]:
            raise StageBlocker("replay_anchor_identity_failure")
        terminal = result["terminal"]
        if terminal["model_semantic_hash"] != EXPECTED_TERMINAL_MODEL_HASH:
            raise StageBlocker("update_512_terminal_model_identity_failure")
        write_json(output / "trajectory_metrics.json", {"schema_version": "stage3_h13_d7_trajectory_metrics_v1", "experiment_identifier": EXPERIMENT_ID, "predeclared_grid": list(GRID), "thresholds": {"H1": H1_THRESHOLD, "R6_H32_REFERENCE": R6_H32_REFERENCE, "H32": H32_THRESHOLD}, "trajectory": trajectory, "all_training_updates": result["rows"], "evaluations": result["evaluations"]})
        write_json(output / "pareto_classification.json", {"schema_version": "stage3_h13_d7_pareto_classification_v1", "classification": classification, "anchor_verification": anchors, "no_interpolation": True, "no_retrospective_search": True})
        write_csv(output / "training_diagnostics.csv", result["rows"])
        write_csv(output / "module_diagnostics.csv", result["module_rows"])
        write_json(output / "model_optimizer_state_digest_manifest.json", {"schema_version": "stage3_h13_d7_model_optimizer_state_digest_manifest_v1", "start": {"bundle_sha256": bundle_identity["raw_sha256"], "model_state_digest": loaded_identity["model_semantic_hash"], "optimizer_state_digest": loaded_identity["optimizer_semantic_hash"], "exp_avg_digest": bundle_identity["state_hashes"]["exp_avg_semantic_hash"], "exp_avg_sq_digest": bundle_identity["state_hashes"]["exp_avg_sq_semantic_hash"], "optimizer_step_range": loaded_identity["optimizer_step_range"], "rng_state_digest": rng_before}, "grid": result["digest_manifest"], "terminal_state_retained": True})
        terminal_payload = {"schema_version": "stage3_h13_d7_terminal_state_v1", "completed_optimizer_steps": EXPECTED_TOTAL_UPDATES, "model_state_dict": terminal["model_state"], "optimizer_state_dict": terminal["optimizer_state"], "model_semantic_hash": terminal["model_semantic_hash"], "optimizer_semantic_hash": terminal["optimizer_semantic_hash"], "final_metrics": result["evaluations"][512]["metrics"], "source_branch_bundle_sha256": bundle_identity["raw_sha256"], "intervention": "NONE", "diagnostic_axis": "checkpoint time"}
        terminal_path = output / "terminal_state.pt"
        torch.save(terminal_payload, terminal_path)
        base.mark_readonly(terminal_path)
        protocol_gates = {
            "CERTIFIED_UPDATE_384_MODEL_STATE_IDENTITY": "PASS",
            "CERTIFIED_UPDATE_384_OPTIMIZER_STATE_IDENTITY": "PASS",
            "ADAMW_STEP_COUNTERS_AT_384": "PASS",
            "EXP_AVG_PRESERVED": "PASS",
            "EXP_AVG_SQ_PRESERVED": "PASS",
            "SCHEDULER_LR_STATE": "ABSENT_BY_DESIGN / PASS",
            "CANONICAL_BATCH_ORDER_IDENTITY": "PASS",
            "SOURCE_REVISION": "PASS",
            "MODEL_ARCHITECTURE": "PASS",
            "LOSS_CONFIGURATION": "PASS",
            "GRADIENT_CLIPPING_CONFIGURATION": "PASS",
            "WEIGHT_DECAY": "PASS",
            "LEARNING_RATE": "PASS",
            "EVALUATOR_SEMANTICS": "PASS",
            "VALIDATION_DATASET_IDENTITY": "PASS",
            "RNG_EXACT_CONTINUATION_IDENTITY": "PASS",
            "FROZEN20_SEALED": "YES",
            "FUTURE_LABEL_LEAKAGE": 0,
            "R8E_A1_REMAINS_IMMUTABLE": "YES",
            "UNPLANNED_INTERVENTION": "NO",
            "ALL_17_GRID_POINTS_MEASURED": "YES",
            "ALL_GRID_VALUES_FINITE": "YES",
        }
        one_sentence = (f"The untouched canonical H32 continuation contains {classification['DOUBLE_PASS_COUNT']} predeclared measured DOUBLE_PASS state(s), so a phase-4 Pareto-compatible state is observed on this grid; R8E_A1 remains the champion and no causal uniqueness claim is made." if classification["PARETO_WINDOW_FOUND"] else "No DOUBLE_PASS was observed on the predeclared grid; the diagnostic remains fail-closed with R8E_A1 unchanged.")
        one_next_action = ("Keep R8E_A1 as the current validation champion and require a separately authorized independent non-reused holdout/robustness protocol before considering any checkpoint selection." if classification["PARETO_WINDOW_FOUND"] else "Keep R8E_A1 as the current validation champion and reassess the next mechanism only through a separately authorized prospective protocol.")
        certificate = {"schema_version": "stage3_h13_d7_terminal_certificate_v1", "status": "PASSED", "first_blocker": "none", "experiment_identifier": EXPERIMENT_ID, "training_executed": True, "training_updates": list(range(385, 513)), "evaluation_grid": list(GRID), "classification": classification, "anchor_verification": anchors, "protocol_gates": protocol_gates, "one_sentence_verdict": one_sentence, "one_next_action": one_next_action, "current_validation_champion": "R8E_A1", "champion_changed": "NO", "d7_diagnostic_only": "YES", "qualifying_checkpoint_automatic_champion": "NO", "frozen20_opened": "NO", "frozen20_used": "NO", "r9_executed": "NO", "authorized_for_r9": "NO", "terminal_model_state_digest": terminal["model_semantic_hash"], "terminal_optimizer_state_digest": terminal["optimizer_semantic_hash"], "terminal_state_path": str(terminal_path.resolve()), "terminal_state_sha256": sha256_file(terminal_path), "artifact_sha256": {}}
        stable_names = ["protocol_integrity_identity.json", "execution_configuration_snapshot.json", "source_revision_record.json", "preflight_identity.json", "trajectory_metrics.json", "pareto_classification.json", "training_diagnostics.csv", "module_diagnostics.csv", "model_optimizer_state_digest_manifest.json", "terminal_state.pt"]
        certificate["artifact_sha256"] = {name: sha256_file(output / name) for name in stable_names}
        write_json(output / "terminal_certificate.json", certificate)
        artifacts = {**certificate["artifact_sha256"], "terminal_certificate.json": sha256_file(output / "terminal_certificate.json")}
        (output / "FINAL_REPORT.md").write_text(render_report(certificate, trajectory, artifacts), encoding="utf-8", newline="\n")
        ledger = write_sha256_manifest(output)
        artifacts["SHA256_MANIFEST.json"] = sha256_file(output / "SHA256_MANIFEST.json")
        certificate["artifact_sha256"] = artifacts
        write_json(output / "terminal_certificate.json", certificate)
        (output / "FINAL_REPORT.md").write_text(render_report(certificate, trajectory, artifacts), encoding="utf-8", newline="\n")
        # The ledger intentionally excludes the report and itself; all source inputs are rechecked after execution.
        base.verify_snapshot(authority["authority_snapshot"])
        if sha256_file(base.A1_CHECKPOINT) != a1_hash_before:
            raise StageBlocker("r8e_a1_mutated_during_execution")
        print(json.dumps({"status": "PASSED", "output": str(output.resolve()), "classification": classification["D7_PRIMARY_CLASSIFICATION"], "double_pass_updates": classification["DOUBLE_PASS_UPDATES"], "first_h32": classification["FIRST_H32_GRADUATING_UPDATE"]}, sort_keys=True), flush=True)
        return 0
    except StageBlocker as exc:
        return fail_closed(output, str(exc), authorization)
    except Exception as exc:  # pragma: no cover
        return fail_closed(output, f"unexpected_execution_failure:{type(exc).__name__}:{exc}", authorization)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return execute(args.output.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
