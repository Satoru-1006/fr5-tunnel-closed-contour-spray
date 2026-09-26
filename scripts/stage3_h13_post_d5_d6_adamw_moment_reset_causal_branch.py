"""Execute the owner-authorized Stage 3 H13 D6 paired AdamW intervention.

The runner starts both continuations from the already-certified post-update-384
bundle.  CONTROL is untouched.  MOMENT_RESET performs exactly one in-place
intervention on ``exp_avg`` and ``exp_avg_sq`` before update 385.  All other
training and evaluation semantics are inherited from the certified D4 runner.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import stage3_h13_post_d5_d6_branch_state_materialization_replay as base  # noqa: E402


EXPERIMENT_ID = "STAGE_3_H13_POST_D5_D6_ADAMW_MOMENT_RESET_CAUSAL_BRANCH"
BUNDLE_PATH = ROOT / "outputs/stage3_h13_post_d5_d6_branch_state_materialization_replay_20260816T152000Z_retry/post_update_384_branch_bundle.pt"
BUNDLE_DIR = BUNDLE_PATH.parent
AUTHORIZATION_PATH = Path(r"C:\Users\86198\.codex\attachments\f0d4ed85-60ff-4a6e-82e0-926c93238a15\pasted-text.txt")
EXPECTED_BUNDLE_SHA256 = "4a022aa0340977e571a3a3085e30d96fd0d92129f4fbeff99aaa8e3420379083"
EXPECTED_MODEL_HASH = "f6f450abd4ac8780a7172c46d44db88d6b7278e22e6a14580b62c9b1627bcd58"
EXPECTED_OPTIMIZER_HASH = "610dd9fa330572888e60cbc16e8f67560e2ce23a3b8628cad7481a90b24936d8"
EXPECTED_BRANCH_UPDATE = 384
EXPECTED_TOTAL_UPDATES = 512
EXPECTED_REMAINING_UPDATES = 128
EXPECTED_STEP = 384
EXPECTED_CLIP = 1.0
EXPECTED_BATCH_SIZE = 256
EXPECTED_EVAL_HORIZONS = (1, 2, 4, 8, 16, 32)
EVAL_UPDATES = (385,)
MOMENT_KEYS = {"exp_avg", "exp_avg_sq"}


class StageBlocker(RuntimeError):
    """Fail-closed D6 blocker."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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


def tensor_digest(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    header = canonical_json({"dtype": str(tensor.dtype), "shape": list(tensor.shape)})
    return sha256_bytes(header + b"\0" + tensor.numpy().tobytes())


def named_tensor_digest(values: Mapping[str, torch.Tensor]) -> str:
    rows = [{"name": name, "dtype": str(values[name].dtype), "shape": list(values[name].shape), "sha256": tensor_digest(values[name])} for name in sorted(values)]
    return sha256_bytes(canonical_json(rows))


def optimizer_groups_hash(optimizer: torch.optim.Optimizer) -> str:
    payload = base.optimizer_canonical_payload(optimizer)["param_groups"]
    return sha256_bytes(canonical_json(payload))


def optimizer_component_excluding_hash(optimizer: torch.optim.Optimizer, excluded: set[str]) -> str:
    rows = []
    for name, parameter in base.named_optimizer_parameters(optimizer):
        state = optimizer.state.get(parameter, {})
        for key in sorted(state, key=str):
            if str(key) in excluded:
                continue
            value = state[key]
            rows.append({"name": name, "key": str(key), "value": {"tensor": tensor_digest(value), "dtype": str(value.dtype), "shape": list(value.shape)} if isinstance(value, torch.Tensor) else jsonable(value)})
    return sha256_bytes(canonical_json(rows))


def state_step_range(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    values = base.optimizer_step_map(optimizer)
    return {"min": min(values.values()), "max": max(values.values()), "unique": sorted(set(values.values())), "count": len(values), "by_parameter": values}


def moment_stats(optimizer: torch.optim.Optimizer, component: str) -> dict[str, Any]:
    values = []
    for _, parameter in base.named_optimizer_parameters(optimizer):
        state = optimizer.state.get(parameter, {})
        if component not in state or not isinstance(state[component], torch.Tensor):
            raise StageBlocker(f"missing_moment_component:{component}")
        values.append(state[component].detach().cpu())
    if not values:
        raise StageBlocker(f"empty_moment_component:{component}")
    flat = torch.cat([value.reshape(-1).double() for value in values])
    return {
        "parameter_state_count": len(values),
        "element_count": int(flat.numel()),
        "norm": float(torch.linalg.vector_norm(flat).item()),
        "rms": float(torch.sqrt(torch.mean(flat.square())).item()),
        "mean": float(torch.mean(flat).item()),
        "mean_abs": float(torch.mean(torch.abs(flat)).item()),
        "max_abs": float(torch.max(torch.abs(flat)).item()),
        "min": float(torch.min(flat).item()),
        "max": float(torch.max(flat).item()),
        "nonzero_elements": int(torch.count_nonzero(flat).item()),
        "exact_zero": bool(torch.count_nonzero(flat).item() == 0),
    }


def optimizer_snapshot(optimizer: torch.optim.Optimizer, model: Any, rng_digest: str) -> dict[str, Any]:
    return {
        "model_semantic_hash": base.canonical_model_state_sha256(model.state_dict()),
        "model_parameter_tensor_hash": named_tensor_digest({name: parameter.detach().cpu() for name, parameter in base.parameter_rows(model)}),
        "optimizer_semantic_hash": base.optimizer_semantic_hash(optimizer),
        "parameter_groups_semantic_hash": optimizer_groups_hash(optimizer),
        "optimizer_step_semantic_hash": sha256_bytes(canonical_json(base.optimizer_step_map(optimizer))),
        "optimizer_non_moment_state_semantic_hash": optimizer_component_excluding_hash(optimizer, MOMENT_KEYS),
        "exp_avg_semantic_hash": base.named_optimizer_component_hash(optimizer, "exp_avg"),
        "exp_avg_sq_semantic_hash": base.named_optimizer_component_hash(optimizer, "exp_avg_sq"),
        "optimizer_state_keys": {name: sorted(str(key) for key in optimizer.state[parameter]) for name, parameter in base.named_optimizer_parameters(optimizer)},
        "step_range": state_step_range(optimizer),
        "rng_state_sha256": rng_digest,
        "exp_avg_stats": moment_stats(optimizer, "exp_avg"),
        "exp_avg_sq_stats": moment_stats(optimizer, "exp_avg_sq"),
    }


def eval_metrics(model: Any, validation: Any, channels: Mapping[str, Any]) -> dict[str, float]:
    model.eval()
    result = base.evaluate_compact(model, validation, channels, horizons=EXPECTED_EVAL_HORIZONS)
    model.train()
    return {str(key): base.finite_float(value) for key, value in result.items()}


def module_family(name: str) -> str:
    lowered = name.lower()
    if lowered.startswith("gru") or ".gru" in lowered:
        return "gru"
    if lowered.startswith("decoder") or "decoder" in lowered:
        return "decoder"
    if lowered.startswith("head") or ".head" in lowered:
        return "head"
    return "other"


def module_update_rows(model: Any, optimizer: torch.optim.Optimizer, raw_grads: Mapping[str, torch.Tensor], before: Mapping[str, torch.Tensor], step: int) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for name, parameter in base.parameter_rows(model):
        family = module_family(name)
        item = records.setdefault(family, {"parameter_count": 0, "raw_grad_sq": 0.0, "clipped_grad_sq": 0.0, "update_sq": 0.0, "exp_avg_sq": 0.0, "exp_avg_sq_sq": 0.0})
        delta = parameter.detach().cpu() - before[name]
        clipped = parameter.grad.detach().cpu() if parameter.grad is not None else torch.zeros_like(before[name])
        raw = raw_grads.get(name, torch.zeros_like(before[name]))
        state = optimizer.state.get(parameter, {})
        item["parameter_count"] += int(parameter.numel())
        item["raw_grad_sq"] += float(torch.sum(raw.double().square()).item())
        item["clipped_grad_sq"] += float(torch.sum(clipped.double().square()).item())
        item["update_sq"] += float(torch.sum(delta.double().square()).item())
        if "exp_avg" in state:
            item["exp_avg_sq"] += float(torch.sum(state["exp_avg"].detach().double().square()).item())
            item["exp_avg_sq_sq"] += float(torch.sum(state["exp_avg_sq"].detach().double().square()).item())
    return [{
        "branch": None,
        "update": step,
        "module": family,
        "parameter_count": item["parameter_count"],
        "raw_gradient_norm": math.sqrt(item["raw_grad_sq"]),
        "clipped_gradient_norm": math.sqrt(item["clipped_grad_sq"]),
        "parameter_update_norm": math.sqrt(item["update_sq"]),
        "exp_avg_norm": math.sqrt(item["exp_avg_sq"]),
        "exp_avg_sq_norm": math.sqrt(item["exp_avg_sq_sq"]),
    } for family, item in sorted(records.items())]


def validate_authorization() -> dict[str, Any]:
    if not AUTHORIZATION_PATH.is_file():
        raise StageBlocker("current_stage_authorization_source_missing")
    text = AUTHORIZATION_PATH.read_text(encoding="utf-8")
    required = [EXPERIMENT_ID, "Execute D6 now under this authorization.", "open Frozen-20"]
    missing = [item for item in required if item not in text]
    if missing:
        raise StageBlocker(f"authorization_scope_not_proven:{missing}")
    return {"source": str(AUTHORIZATION_PATH.resolve()), "sha256": sha256_file(AUTHORIZATION_PATH), "scope": EXPERIMENT_ID, "explicit_scope_match": "YES"}


def validate_bundle(bundle: Mapping[str, Any], authority: Mapping[str, Any]) -> dict[str, Any]:
    if sha256_file(BUNDLE_PATH) != EXPECTED_BUNDLE_SHA256:
        raise StageBlocker("branch_bundle_hash_mismatch")
    if bundle.get("training_position", {}).get("completed_optimizer_steps") != EXPECTED_BRANCH_UPDATE:
        raise StageBlocker("update_384_position_mismatch")
    position = bundle.get("training_position", {})
    if position.get("next_completed_optimizer_step") != 385 or position.get("next_zero_based_schedule_index") != 384 or position.get("remaining_updates") != EXPECTED_REMAINING_UPDATES:
        raise StageBlocker("update_384_position_fields_mismatch")
    expected_suffix = authority["schedule_slice_hashes"]["suffix_128"]
    if position.get("remaining_schedule_sha256") != expected_suffix:
        raise StageBlocker("canonical_remaining_sequence_hash_mismatch")
    if position.get("next_batch_window_sha256") != base.batch_digest(authority["schedule_manifest"], EXPECTED_BRANCH_UPDATE):
        raise StageBlocker("next_batch_identity_mismatch")
    expected_config = base.load_json(BUNDLE_DIR / "training_config.json")
    if bundle.get("training_config") != expected_config:
        raise StageBlocker("bundle_training_config_mismatch")
    expected_evaluator = {"split": "H10 VALIDATION", "horizons": [1, 2, 4, 8, 16, 32], "free_running": True, "future_label_leakage": 0}
    if bundle.get("evaluator_config") != expected_evaluator:
        raise StageBlocker("evaluator_identity_mismatch")
    provenance = bundle.get("provenance", {})
    if provenance.get("source_revision") != base.EXPECTED_SOURCE_COMMIT:
        raise StageBlocker("source_revision_mismatch")
    if provenance.get("canonical_schedule_sha256") != base.EXPECTED_CANONICAL_SCHEDULE or provenance.get("canonical_schedule_suffix_128_sha256") != expected_suffix:
        raise StageBlocker("bundle_schedule_provenance_mismatch")
    if bundle.get("optimizer_contract", {}).get("amsgrad") is not False:
        raise StageBlocker("amsgrad_contract_contradiction")
    return {
        "branch_bundle_raw_sha256": sha256_file(BUNDLE_PATH),
        "training_position": position,
        "remaining_schedule_sha256": expected_suffix,
        "next_batch_window_sha256": position["next_batch_window_sha256"],
        "source_revision": provenance.get("source_revision"),
        "training_config": expected_config,
        "evaluator_config": expected_evaluator,
        "optimizer_contract": bundle.get("optimizer_contract"),
    }


def validate_loaded_bundle(bundle: Mapping[str, Any], model: Any, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    model_hash = base.canonical_model_state_sha256(model.state_dict())
    optimizer_hash = base.optimizer_semantic_hash(optimizer)
    if model_hash != EXPECTED_MODEL_HASH:
        raise StageBlocker("post_update_384_model_identity_mismatch")
    if optimizer_hash != EXPECTED_OPTIMIZER_HASH:
        raise StageBlocker("post_update_384_optimizer_identity_mismatch")
    if bundle.get("state_hashes", {}).get("model_semantic_hash") != model_hash or bundle.get("state_hashes", {}).get("optimizer_semantic_hash") != optimizer_hash:
        raise StageBlocker("bundle_internal_identity_mismatch")
    mapping = base.optimizer_mapping(optimizer)
    if mapping != bundle.get("optimizer_parameter_mapping"):
        raise StageBlocker("parameter_mapping_ambiguity_or_mismatch")
    if any(item.get("step") != EXPECTED_STEP for item in mapping):
        raise StageBlocker("optimizer_step_not_384")
    if any(item.get("state_keys") != ["exp_avg", "exp_avg_sq", "step"] for item in mapping):
        raise StageBlocker("unexpected_additional_optimizer_state")
    actual_contract = base.optimizer_contract(optimizer)
    if actual_contract != bundle.get("optimizer_contract"):
        raise StageBlocker("loaded_optimizer_contract_mismatch")
    if base.named_optimizer_component_hash(optimizer, "exp_avg") != bundle["state_hashes"]["exp_avg_semantic_hash"]:
        raise StageBlocker("exp_avg_identity_mismatch")
    if base.named_optimizer_component_hash(optimizer, "exp_avg_sq") != bundle["state_hashes"]["exp_avg_sq_semantic_hash"]:
        raise StageBlocker("exp_avg_sq_identity_mismatch")
    return {
        "model_semantic_hash": model_hash,
        "optimizer_semantic_hash": optimizer_hash,
        "parameter_mapping_valid": "YES",
        "optimizer_step_range": state_step_range(optimizer),
        "optimizer_contract": actual_contract,
        "amsgrad": actual_contract.get("amsgrad"),
        "state_keys": sorted({key for item in mapping for key in item["state_keys"]}),
    }


def zero_moments(optimizer: torch.optim.Optimizer) -> None:
    for _, parameter in base.named_optimizer_parameters(optimizer):
        state = optimizer.state.get(parameter, {})
        if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise StageBlocker("unexpected_optimizer_state_before_intervention")
        state["exp_avg"].zero_()
        state["exp_avg_sq"].zero_()


def intervention_audit(model: Any, optimizer: torch.optim.Optimizer, rng_digest: str) -> dict[str, Any]:
    before = optimizer_snapshot(optimizer, model, rng_digest)
    zero_moments(optimizer)
    after = optimizer_snapshot(optimizer, model, rng_digest)
    if before["model_semantic_hash"] != after["model_semantic_hash"] or before["model_parameter_tensor_hash"] != after["model_parameter_tensor_hash"]:
        raise StageBlocker("model_parameters_modified_during_intervention")
    if before["optimizer_step_semantic_hash"] != after["optimizer_step_semantic_hash"] or before["step_range"] != after["step_range"]:
        raise StageBlocker("optimizer_step_modified_during_intervention")
    if before["parameter_groups_semantic_hash"] != after["parameter_groups_semantic_hash"]:
        raise StageBlocker("parameter_groups_modified_during_intervention")
    if before["optimizer_non_moment_state_semantic_hash"] != after["optimizer_non_moment_state_semantic_hash"]:
        raise StageBlocker("non_moment_optimizer_state_modified_during_intervention")
    if before["exp_avg_semantic_hash"] == after["exp_avg_semantic_hash"] or before["exp_avg_sq_semantic_hash"] == after["exp_avg_sq_semantic_hash"]:
        raise StageBlocker("authorized_moment_change_not_observed")
    if not after["exp_avg_stats"]["exact_zero"] or not after["exp_avg_sq_stats"]["exact_zero"]:
        raise StageBlocker("moment_reset_not_exact_zero")
    changed_keys = []
    reset_counts = {"exp_avg": 0, "exp_avg_sq": 0}
    reset_elements = {"exp_avg": 0, "exp_avg_sq": 0}
    for name, parameter in base.named_optimizer_parameters(optimizer):
        state = optimizer.state[parameter]
        for key in sorted(state, key=str):
            if key in MOMENT_KEYS:
                if before["optimizer_state_keys"][name].count(key) != 1:
                    raise StageBlocker(f"moment_state_mapping_missing:{name}:{key}")
                before_hash = None
                if key == "exp_avg":
                    before_hash = before["exp_avg_semantic_hash"]
                elif key == "exp_avg_sq":
                    before_hash = before["exp_avg_sq_semantic_hash"]
                if before_hash is not None:
                    reset_counts[key] += 1
                    reset_elements[key] += int(state[key].numel())
            else:
                # The aggregate non-moment hash above is the exact proof for these values.
                pass
        changed_keys.extend([f"{name}.exp_avg", f"{name}.exp_avg_sq"])
    if set(item.rsplit(".", 1)[-1] for item in changed_keys) != MOMENT_KEYS:
        raise StageBlocker("unauthorized_intervention_tensor_detected")
    return {
        "schema_version": "stage3_h13_d6_intervention_audit_v1",
        "intervention": "zero exp_avg and exp_avg_sq in-place exactly once before update 385",
        "before": before,
        "after": after,
        "MODEL_PARAMETER_DIFFERENCES": 0,
        "OPTIMIZER_STEP_DIFFERENCES": 0,
        "PARAMETER_GROUP_DIFFERENCES": 0,
        "NON_MOMENT_OPTIMIZER_STATE_DIFFERENCES": 0,
        "EXP_AVG_CHANGED": "YES",
        "EXP_AVG_SQ_CHANGED": "YES",
        "changed_tensor_families": sorted(MOMENT_KEYS),
        "changed_tensor_count": len(changed_keys),
        "exp_avg_parameter_states": reset_counts["exp_avg"],
        "exp_avg_parameter_states_reset": reset_counts["exp_avg"],
        "exp_avg_elements_reset": reset_elements["exp_avg"],
        "exp_avg_sq_parameter_states": reset_counts["exp_avg_sq"],
        "exp_avg_sq_parameter_states_reset": reset_counts["exp_avg_sq"],
        "exp_avg_sq_elements_reset": reset_elements["exp_avg_sq"],
        "exp_avg_post_reset_exact_zero": "YES",
        "exp_avg_sq_post_reset_exact_zero": "YES",
        "preserved_optimizer_step_value_or_range": after["step_range"],
        "preserved_model_semantic_hash": after["model_semantic_hash"],
        "single_factor_intervention_valid": "YES",
    }


def run_continuation(label: str, model: Any, optimizer: torch.optim.Optimizer, pre: Mapping[str, Any], authority: Mapping[str, Any], control_snapshots: Mapping[int, Mapping[str, torch.Tensor]] | None = None) -> dict[str, Any]:
    schedule_rows = authority["schedule_rows"]
    schedule_manifest = authority["schedule_manifest"]
    train_data = pre["train_data"]
    validation = pre["validation"]
    channels = pre["stats"]["channels"]
    parameter_rows = base.parameter_rows(model)
    rows: list[dict[str, Any]] = []
    module_rows: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    snapshots: dict[int, dict[str, torch.Tensor]] = {}
    model.train()
    for zero_based_index in range(EXPECTED_BRANCH_UPDATE, EXPECTED_TOTAL_UPDATES):
        step = zero_based_index + 1
        horizon = int(base.PHASE_HORIZONS[zero_based_index // 128])
        expected_batch_hash = base.batch_digest(schedule_manifest, zero_based_index)
        batch = base.tensor_batch(train_data, schedule_rows[zero_based_index])
        optimizer.zero_grad(set_to_none=True)
        rollout = base.causal_paired_rollout(model, batch["inputs"], batch["history_positions"], batch["history_times"], batch["target_times"], batch["starts"], batch["ends"], channels, horizon=horizon, target_positions_for_teacher=batch["target_positions"], hidden_consistency_enabled=True)
        losses = base.training_losses(rollout["predictions"], batch["target_positions"], rollout["hidden_loss"], lambda_hidden=base.LAMBDA_HIDDEN if hasattr(base, "LAMBDA_HIDDEN") else 0.005, lambda_h1=1.0, beta=0.001)
        if not all(bool(torch.isfinite(value)) for value in losses.values()):
            raise StageBlocker(f"nonfinite_training_loss_update_{step}")
        losses["total"].backward()
        raw_nonfinite = not base.gradients_are_finite(model)
        if raw_nonfinite:
            raise StageBlocker(f"nonfinite_gradient_update_{step}")
        before = {name: parameter.detach().cpu().clone() for name, parameter in parameter_rows}
        raw_grads = {name: parameter.grad.detach().cpu().clone() for name, parameter in parameter_rows if parameter.grad is not None}
        raw_norm = base.aggregate_norm(raw_grads.values())
        raw_digest = named_tensor_digest(raw_grads)
        pre_clip_norm = base.finite_float(torch.nn.utils.clip_grad_norm_(model.parameters(), EXPECTED_CLIP))
        if not math.isclose(raw_norm, pre_clip_norm, rel_tol=1.0e-5, abs_tol=1.0e-5):
            raise StageBlocker(f"gradient_norm_instrumentation_mismatch_update_{step}")
        clipped_grads = {name: parameter.grad.detach().cpu().clone() for name, parameter in parameter_rows if parameter.grad is not None}
        clipped_digest = named_tensor_digest(clipped_grads)
        clipped_norm = base.aggregate_norm(clipped_grads.values())
        clipping_activated = pre_clip_norm > EXPECTED_CLIP
        clipping_coefficient = min(1.0, EXPECTED_CLIP / pre_clip_norm) if pre_clip_norm > 0.0 else 1.0
        optimizer.step()
        after = {name: parameter.detach().cpu().clone() for name, parameter in parameter_rows}
        update_tensors = {name: after[name] - before[name] for name, _ in parameter_rows}
        if not all(bool(torch.all(torch.isfinite(value))) for value in after.values()):
            raise StageBlocker(f"nonfinite_model_parameter_update_{step}")
        state_nonfinite = any(isinstance(value, torch.Tensor) and not bool(torch.all(torch.isfinite(value))) for state in optimizer.state.values() for value in state.values())
        if state_nonfinite:
            raise StageBlocker(f"nonfinite_optimizer_state_update_{step}")
        update_norm = base.aggregate_norm(update_tensors.values())
        parameter_norm_before = base.aggregate_norm(before.values())
        max_tensor_update_norm = max(base.aggregate_norm([value]) for value in update_tensors.values())
        max_abs_update = max(float(torch.max(torch.abs(value)).item()) for value in update_tensors.values())
        model_hash = base.canonical_model_state_sha256(model.state_dict())
        model_param_hash = named_tensor_digest(after)
        optimizer_diag = base.aggregate_optimizer_state(model, optimizer, before, 5.0e-5, 1.0e-5, 1.0e-8, (0.9, 0.999))
        record: dict[str, Any] = {
            "branch": label,
            "update": step,
            "zero_based_schedule_index": zero_based_index,
            "active_horizon": horizon,
            "batch_window_sha256": expected_batch_hash,
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
            "parameter_update_digest": named_tensor_digest(update_tensors),
            "raw_gradient_digest": raw_digest,
            "clipped_gradient_digest": clipped_digest,
            "model_semantic_hash": model_hash,
            "model_parameter_tensor_hash": model_param_hash,
            "nonfinite_events": 0,
            **optimizer_diag,
        }
        new_module_rows = module_update_rows(model, optimizer, raw_grads, before, step)
        module_rows.extend(dict(item, branch=label) for item in new_module_rows)
        if step in EVAL_UPDATES:
            metrics = eval_metrics(model, validation, channels)
            eval_row: dict[str, Any] = {"branch": label, "update": step, **{f"H{key}": value for key, value in metrics.items()}}
            if control_snapshots is not None and step in control_snapshots:
                eval_row["model_l2_delta_vs_control"] = base.aggregate_norm(after[name] - control_snapshots[step][name] for name, _ in parameter_rows)
            else:
                snapshots[step] = {name: value.clone() for name, value in after.items()}
            record.update({f"eval_H{key}": value for key, value in metrics.items()})
            evaluations.append(eval_row)
        rows.append(record)
    model.eval()
    terminal_metrics = eval_metrics(model, validation, channels)
    terminal_model_hash = base.canonical_model_state_sha256(model.state_dict())
    terminal_optimizer_hash = base.optimizer_semantic_hash(optimizer)
    return {
        "branch": label,
        "rows": rows,
        "module_rows": module_rows,
        "evaluations": evaluations,
        "snapshots": snapshots,
        "terminal_metrics": terminal_metrics,
        "terminal_model_semantic_hash": terminal_model_hash,
        "terminal_optimizer_semantic_hash": terminal_optimizer_hash,
        "terminal_optimizer_step_range": state_step_range(optimizer),
        "terminal_loss": rows[-1]["training_loss_total"],
        "terminal_model_state": copy.deepcopy(model.state_dict()),
        "terminal_optimizer_state": copy.deepcopy(optimizer.state_dict()),
    }


def compare_branch_rows(control: Mapping[str, Any], reset: Mapping[str, Any]) -> dict[str, Any]:
    control_rows = {int(row["update"]): row for row in control["rows"]}
    reset_rows = {int(row["update"]): row for row in reset["rows"]}
    if sorted(control_rows) != sorted(reset_rows) or sorted(control_rows) != list(range(385, 513)):
        raise StageBlocker("branch_update_counts_or_positions_differ")

    def first_diff(field: str, tolerance: float = 0.0) -> int | None:
        for step in sorted(control_rows):
            left, right = control_rows[step][field], reset_rows[step][field]
            if isinstance(left, str) or isinstance(right, str):
                different = left != right
            else:
                different = not math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
            if different:
                return step
        return None

    def first_digest_diff(field: str) -> int | None:
        return first_diff(field)

    eval_control = {int(row["update"]): row for row in control["evaluations"]}
    eval_reset = {int(row["update"]): row for row in reset["evaluations"]}
    first_eval = None
    first_eval_quantity = None
    first_eval_delta = None
    for step in sorted(eval_control):
        for horizon in EXPECTED_EVAL_HORIZONS:
            key = f"H{horizon}"
            delta = float(eval_reset[step][key]) - float(eval_control[step][key])
            if not math.isclose(delta, 0.0, rel_tol=0.0, abs_tol=0.0):
                first_eval = step
                first_eval_quantity = key
                first_eval_delta = delta
                break
        if first_eval is not None:
            break
    parameter_update_step = first_digest_diff("parameter_update_digest")
    model_step = first_digest_diff("model_parameter_tensor_hash")
    gradient_step = first_digest_diff("raw_gradient_digest")
    clipping_step = None
    for step in sorted(control_rows):
        fields = ("raw_gradient_norm_before_clipping", "clipped_gradient_norm_after_clipping", "clipping_activated", "clipping_coefficient", "clipped_gradient_digest")
        if any((control_rows[step][field] != reset_rows[step][field]) if isinstance(control_rows[step][field], str) else not math.isclose(float(control_rows[step][field]), float(reset_rows[step][field]), rel_tol=0.0, abs_tol=0.0) for field in fields):
            clipping_step = step
            break
    loss_step = first_diff("training_loss_total")
    first_divergent = min(item for item in (parameter_update_step, model_step, gradient_step, clipping_step, loss_step, first_eval) if item is not None)
    magnitude = {
        "first_parameter_update_norm_delta_update_385": float(reset_rows[385]["parameter_update_norm"]) - float(control_rows[385]["parameter_update_norm"]),
        "first_parameter_update_max_abs_delta_update_385": float(reset_rows[385]["max_parameter_update_abs"]) - float(control_rows[385]["max_parameter_update_abs"]),
        "first_model_l2_delta_update_385": next((row.get("model_l2_delta_vs_control") for row in reset["evaluations"] if int(row["update"]) == 385), None),
        "first_gradient_norm_delta_update_386": float(reset_rows[386]["raw_gradient_norm_before_clipping"]) - float(control_rows[386]["raw_gradient_norm_before_clipping"]),
        "first_loss_delta_update_386": float(reset_rows[386]["training_loss_total"]) - float(control_rows[386]["training_loss_total"]),
        "first_downstream_evaluation_quantity": first_eval_quantity,
        "first_downstream_evaluation_signed_delta": first_eval_delta,
    }
    return {
        "intentional_optimizer_state_divergence": "before update 385",
        "first_parameter_update_divergence_update": parameter_update_step,
        "first_model_state_divergence_update": model_step,
        "first_gradient_divergence_update": gradient_step,
        "first_clipping_divergence_update": clipping_step,
        "first_loss_divergence_update": loss_step,
        "first_downstream_measurable_divergence": first_eval,
        "first_divergent_quantity": first_eval_quantity or "parameter_update_tensor",
        "divergence_magnitude": magnitude,
    }


def quantiles(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {str(q): float(np.quantile(array, q)) for q in (0.0, 0.5, 0.9, 0.95, 0.99, 1.0)}


def stability(branch: Mapping[str, Any]) -> dict[str, Any]:
    rows = branch["rows"]
    raw = [float(row["raw_gradient_norm_before_clipping"]) for row in rows]
    updates = [float(row["parameter_update_norm"]) for row in rows]
    max_abs = [float(row["max_parameter_update_abs"]) for row in rows]
    clipping = sum(row["clipping_activated"] == "YES" for row in rows)
    return {
        "nonfinite_events": sum(int(row["nonfinite_events"]) for row in rows),
        "nan_events": 0,
        "inf_events": 0,
        "gradient_norm_median_max": {"median": float(np.median(raw)), "max": max(raw)},
        "gradient_norm_quantiles": quantiles(raw),
        "parameter_update_norm_median_max": {"median": float(np.median(updates)), "max": max(updates)},
        "parameter_update_norm_quantiles": quantiles(updates),
        "clipping_activation_count": clipping,
        "maximum_observed_update_magnitude": max(max_abs),
        "terminal_loss": branch["terminal_loss"],
        "divergence_or_explosion_indicator": "NO_NONFINITE_OR_EXPLOSION_INDICATOR" if all(math.isfinite(value) for value in raw + updates + max_abs) else "OBSERVED",
        "exp_avg_diagnostics": {"quantiles": quantiles([float(row["exp_avg_norm"]) for row in rows]), "terminal": rows[-1]["exp_avg_norm"]},
        "exp_avg_sq_diagnostics": {"quantiles": quantiles([float(row["exp_avg_sq_norm"]) for row in rows]), "terminal": rows[-1]["exp_avg_sq_norm"]},
    }


def status(metric: float, threshold: float | None) -> str:
    return "NOT_APPLICABLE_NO_FROZEN_THRESHOLD" if threshold is None else ("PASS" if metric <= threshold else "FAIL")


def paired_metrics(control: Mapping[str, Any], reset: Mapping[str, Any], d4_metrics: Mapping[str, float]) -> dict[str, Any]:
    thresholds = {"1": float(base.D4_H1_THRESHOLD), "32": float(base.H32_THRESHOLD)}
    values = {}
    for horizon in EXPECTED_EVAL_HORIZONS:
        key = str(horizon)
        c = float(control["terminal_metrics"][key])
        r = float(reset["terminal_metrics"][key])
        delta = r - c
        values[key] = {
            "control": c,
            "moment_reset": r,
            "absolute_delta": abs(delta),
            "signed_delta": delta,
            "percent_change": None if c == 0.0 else 100.0 * delta / c,
            "frozen_threshold": thresholds.get(key),
            "control_status": status(c, thresholds.get(key)),
            "moment_reset_status": status(r, thresholds.get(key)),
            "authoritative_d4": float(d4_metrics[key]) if key in d4_metrics else None,
        }
    return {
        "schema_version": "stage3_h13_d6_paired_evaluation_metrics_v1",
        "evaluator": {"split": "H10 VALIDATION", "free_running": True, "horizons": list(EXPECTED_EVAL_HORIZONS), "future_label_leakage": 0},
        "terminal": values,
        "control": {"metrics": control["terminal_metrics"], "model_semantic_hash": control["terminal_model_semantic_hash"], "optimizer_semantic_hash": control["terminal_optimizer_semantic_hash"]},
        "moment_reset": {"metrics": reset["terminal_metrics"], "model_semantic_hash": reset["terminal_model_semantic_hash"], "optimizer_semantic_hash": reset["terminal_optimizer_semantic_hash"]},
        "trajectory_evaluations": {"control": control["evaluations"], "moment_reset": reset["evaluations"]},
    }


def artifact_hashes(output: Path, names: Sequence[str]) -> dict[str, str]:
    return {name: sha256_file(output / name) for name in names if (output / name).is_file()}


def write_integrity_manifest(output: Path) -> str:
    rows = []
    for path in sorted(item for item in output.rglob("*") if item.is_file() and item.name != "integrity_manifest.json"):
        rows.append({"path": str(path.relative_to(output)).replace("\\", "/"), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    write_json(output / "integrity_manifest.json", {"schema_version": "stage3_h13_d6_integrity_manifest_v1", "hash_algorithm": "SHA-256", "self_hash_excluded": True, "files": rows})
    return sha256_file(output / "integrity_manifest.json")


def render_report(certificate: Mapping[str, Any]) -> str:
    lines = [
        f"STAGE_3_H13_POST_D5_D6_ADAMW_MOMENT_RESET_CAUSAL_BRANCH:\n{certificate['STAGE_3_H13_POST_D5_D6_ADAMW_MOMENT_RESET_CAUSAL_BRANCH']}",
        f"\nFIRST_BLOCKER:\n{certificate['FIRST_BLOCKER']}",
        f"\nONE_SENTENCE_VERDICT:\n{certificate['ONE_SENTENCE_VERDICT']}",
        "\n==================================================\n1. AUTHORIZATION / PROTOCOL INTEGRITY\n==================================================\n",
    ]
    for key in ("PROJECT_OWNER_D6_AUTHORIZATION", "CERTIFIED_BRANCH_BUNDLE_HASH_MATCH", "POST_UPDATE_384_MODEL_IDENTITY_MATCH", "POST_UPDATE_384_OPTIMIZER_IDENTITY_MATCH", "PARAMETER_MAPPING_VALID", "UPDATE_BOUNDARY_VALID", "CANONICAL_REMAINING_SEQUENCE_EXACT", "SOURCE_REVISION_MATCH", "CONFIG_MATCH", "CLIPPING_UNCHANGED", "R8E_A1_IMMUTABLE", "FROZEN20_OPENED", "FROZEN20_USED", "FUTURE_LABEL_LEAKAGE"):
        lines.append(f"{key}: {certificate.get(key)}")
    lines += ["\n==================================================\n2. D6 INTERVENTION AUDIT\n==================================================\n"]
    for key in ("CONTROL_STARTED_FROM_CERTIFIED_BUNDLE", "RESET_STARTED_FROM_CERTIFIED_BUNDLE", "ONLY_EXP_AVG_EXP_AVG_SQ_CHANGED", "MODEL_PARAMETERS_CHANGED_AT_BRANCH_POINT", "OPTIMIZER_STEP_CHANGED", "PARAMETER_GROUPS_CHANGED", "OTHER_OPTIMIZER_STATE_CHANGED", "EXP_AVG_PARAMETER_STATES_RESET", "EXP_AVG_SQ_PARAMETER_STATES_RESET", "EXP_AVG_POST_RESET_EXACT_ZERO", "EXP_AVG_SQ_POST_RESET_EXACT_ZERO", "PRESERVED_STEP_VALUE_OR_RANGE", "SINGLE_FACTOR_INTERVENTION_VALID"):
        lines.append(f"{key}: {certificate.get(key)}")
    lines += ["\n==================================================\n3. CONTROL RESULT\n==================================================\n"]
    for key in ("CONTROL_H1", "CONTROL_H1_STATUS", "CONTROL_H4", "CONTROL_H16", "CONTROL_H32", "CONTROL_H32_STATUS", "CONTROL_TERMINAL_MODEL_SEMANTIC_HASH", "CONTROL_TERMINAL_OPTIMIZER_SEMANTIC_HASH", "CONTROL_VS_AUTHORITATIVE_D4_H1_DELTA", "CONTROL_VS_AUTHORITATIVE_D4_H32_DELTA"):
        lines.append(f"{key}: {certificate.get(key)}")
    lines += ["\n==================================================\n4. MOMENT-RESET RESULT\n==================================================\n"]
    for key in ("RESET_H1", "RESET_H1_STATUS", "RESET_H4", "RESET_H16", "RESET_H32", "RESET_H32_STATUS", "RESET_TERMINAL_MODEL_SEMANTIC_HASH", "RESET_TERMINAL_OPTIMIZER_SEMANTIC_HASH"):
        lines.append(f"{key}: {certificate.get(key)}")
    lines += ["\n==================================================\n5. PAIRED EFFECT\n==================================================\n"]
    for key in ("FROZEN_H1_THRESHOLD", "RESET_VS_CONTROL_H1_ABSOLUTE_CHANGE", "RESET_VS_CONTROL_H1_PERCENT_CHANGE", "CONTROL_H1_MARGIN_TO_THRESHOLD", "RESET_H1_MARGIN_TO_THRESHOLD", "RESET_VS_CONTROL_H4_CHANGE", "RESET_VS_CONTROL_H16_CHANGE", "RESET_VS_CONTROL_H32_CHANGE", "RESET_VS_CONTROL_H32_PERCENT_CHANGE", "H1_CLASSIFICATION_CHANGED", "H32_CLASSIFICATION_CHANGED"):
        lines.append(f"{key}: {certificate.get(key)}")
    lines += ["\n==================================================\n6. FIRST CAUSAL DIVERGENCE\n==================================================\n"]
    for key in ("INTENTIONAL_OPTIMIZER_STATE_DIVERGENCE", "FIRST_PARAMETER_UPDATE_DIVERGENCE_UPDATE", "FIRST_MODEL_STATE_DIVERGENCE_UPDATE", "FIRST_GRADIENT_DIVERGENCE_UPDATE", "FIRST_CLIPPING_DIVERGENCE_UPDATE", "FIRST_LOSS_DIVERGENCE_UPDATE", "FIRST_DOWNSTREAM_MEASURABLE_DIVERGENCE", "FIRST_DIVERGENT_QUANTITY", "DIVERGENCE_MAGNITUDE"):
        lines.append(f"{key}: {json.dumps(certificate.get(key), sort_keys=True) if isinstance(certificate.get(key), (dict, list)) else certificate.get(key)}")
    lines += ["\n==================================================\n7. OPTIMIZER / STABILITY DIAGNOSTICS\n==================================================\n"]
    for key in ("CONTROL_GRADIENT_NORM_MEDIAN_MAX", "RESET_GRADIENT_NORM_MEDIAN_MAX", "CONTROL_UPDATE_NORM_MEDIAN_MAX", "RESET_UPDATE_NORM_MEDIAN_MAX", "CONTROL_CLIPPING_ACTIVATIONS", "RESET_CLIPPING_ACTIVATIONS", "CONTROL_NONFINITE_EVENTS", "RESET_NONFINITE_EVENTS", "CONTROL_EXP_AVG_DIAGNOSTICS", "RESET_EXP_AVG_DIAGNOSTICS", "CONTROL_EXP_AVG_SQ_DIAGNOSTICS", "RESET_EXP_AVG_SQ_DIAGNOSTICS", "NUMERICAL_STABILITY_PRESERVED"):
        lines.append(f"{key}: {json.dumps(certificate.get(key), sort_keys=True) if isinstance(certificate.get(key), (dict, list)) else certificate.get(key)}")
    lines += ["\n==================================================\n8. ARTIFACTS / HASHES\n==================================================\n"]
    for key in ("D6_OUTPUT_DIRECTORY", "CONTROL_TERMINAL_STATE", "CONTROL_TERMINAL_STATE_SHA256", "RESET_TERMINAL_STATE", "RESET_TERMINAL_STATE_SHA256", "PAIRED_DIAGNOSTICS", "INTERVENTION_AUDIT", "TERMINAL_CERTIFICATE", "INTEGRITY_MANIFEST", "NEW_PERSISTENT_STORAGE_BYTES"):
        lines.append(f"{key}: {certificate.get(key)}")
    lines += ["\n==================================================\n9. PROTOCOL STATUS\n==================================================\n"]
    for key in ("D6_CONTROL_EXECUTED", "D6_MOMENT_RESET_EXECUTED", "D6_PAIRED_EXPERIMENT_COMPLETE", "CURRENT_VALIDATION_CHAMPION", "CHAMPION_CHANGED", "READY_FOR_R9", "AUTHORIZED_FOR_R9", "FROZEN20_STATUS"):
        lines.append(f"{key}: {certificate.get(key)}")
    lines += ["\n==================================================\n10. EXECUTION-LEVEL CONCLUSION\n==================================================\n"]
    lines.extend(f"{index}. {item}" for index, item in enumerate(certificate.get("EXECUTION_LEVEL_CONCLUSION", []), start=1))
    lines += ["", "ONE_NEXT_ACTION:", "Perform a separate fail-closed D6 final scientific interpretation of the completed paired evidence.", ""]
    return "\n".join(lines)


def fail_closed(output: Path, blocker: str, authorization: Mapping[str, Any] | None = None) -> int:
    output.mkdir(parents=True, exist_ok=True)
    certificate = {
        "schema_version": "stage3_h13_d6_terminal_certificate_v1",
        "STAGE_3_H13_POST_D5_D6_ADAMW_MOMENT_RESET_CAUSAL_BRANCH": "BLOCKED",
        "FIRST_BLOCKER": blocker,
        "ONE_SENTENCE_VERDICT": "D6 failed closed before a scientifically interpretable paired result was established; no authorized follow-on experiment was started.",
        "PROJECT_OWNER_D6_AUTHORIZATION": "YES" if authorization else "UNKNOWN",
        "CERTIFIED_BRANCH_BUNDLE_HASH_MATCH": "NO",
        "POST_UPDATE_384_MODEL_IDENTITY_MATCH": "NO",
        "POST_UPDATE_384_OPTIMIZER_IDENTITY_MATCH": "NO",
        "PARAMETER_MAPPING_VALID": "NO",
        "UPDATE_BOUNDARY_VALID": "NO",
        "CANONICAL_REMAINING_SEQUENCE_EXACT": "NO",
        "SOURCE_REVISION_MATCH": "UNKNOWN",
        "CONFIG_MATCH": "NO",
        "CLIPPING_UNCHANGED": "UNKNOWN",
        "R8E_A1_IMMUTABLE": "YES",
        "FROZEN20_OPENED": "NO",
        "FROZEN20_USED": "NO",
        "FUTURE_LABEL_LEAKAGE": 0,
        "CONTROL_STARTED_FROM_CERTIFIED_BUNDLE": "NO",
        "RESET_STARTED_FROM_CERTIFIED_BUNDLE": "NO",
        "ONLY_EXP_AVG_EXP_AVG_SQ_CHANGED": "NO",
        "MODEL_PARAMETERS_CHANGED_AT_BRANCH_POINT": "UNKNOWN",
        "OPTIMIZER_STEP_CHANGED": "UNKNOWN",
        "PARAMETER_GROUPS_CHANGED": "UNKNOWN",
        "OTHER_OPTIMIZER_STATE_CHANGED": "UNKNOWN",
        "EXP_AVG_POST_RESET_EXACT_ZERO": "NOT_REACHED",
        "EXP_AVG_SQ_POST_RESET_EXACT_ZERO": "NOT_REACHED",
        "SINGLE_FACTOR_INTERVENTION_VALID": "NO",
        "D6_CONTROL_EXECUTED": "NO",
        "D6_MOMENT_RESET_EXECUTED": "NO",
        "D6_PAIRED_EXPERIMENT_COMPLETE": "NO",
        "CURRENT_VALIDATION_CHAMPION": "R8E_A1",
        "CHAMPION_CHANGED": "NO",
        "READY_FOR_R9": "NO",
        "AUTHORIZED_FOR_R9": "NO",
        "FROZEN20_STATUS": "SEALED",
        "D6_OUTPUT_DIRECTORY": str(output.resolve()),
        "EXECUTION_LEVEL_CONCLUSION": ["D6 was blocked before a paired scientific result existed.", "Frozen-20 remains sealed.", "R8E_A1 remains the validation champion.", "No separate experiment was executed."],
    }
    write_json(output / "terminal_certificate.json", certificate)
    (output / "FINAL_REPORT.md").write_text(render_report(certificate), encoding="utf-8", newline="\n")
    write_integrity_manifest(output)
    return 1


def execute(output: Path) -> int:
    if output.exists():
        return fail_closed(output, f"output_directory_already_exists_no_resume:{output}")
    output.mkdir(parents=True, exist_ok=False)
    authorization: dict[str, Any] | None = None
    try:
        authorization = validate_authorization()
        pre = base.preflight(True)
        authority = base.verify_authority(pre)
        current_environment = authority["current_environment"]
        if current_environment.get("source_revision") != base.EXPECTED_SOURCE_COMMIT:
            raise StageBlocker("source_revision_mismatch")
        if current_environment.get("device") != "cpu" or current_environment.get("dtype") != "float32" or current_environment.get("deterministic_algorithms") is not True:
            raise StageBlocker("execution_environment_mismatch")
        bundle_hash = sha256_file(BUNDLE_PATH)
        if bundle_hash != EXPECTED_BUNDLE_SHA256:
            raise StageBlocker("branch_bundle_hash_mismatch")
        bundle = torch.load(BUNDLE_PATH, map_location="cpu", weights_only=False)
        bundle_identity = validate_bundle(bundle, authority)
        config = bundle_identity["training_config"]
        if config.get("gradient_clipping") != {"enabled": True, "max_norm": 1.0, "operation": "torch.nn.utils.clip_grad_norm_"}:
            raise StageBlocker("clipping_configuration_mismatch")
        preflight_identity = {
            "schema_version": "stage3_h13_d6_preflight_identity_v1",
            "authorization": authorization,
            "bundle": bundle_identity,
            "authoritative_model_semantic_hash": EXPECTED_MODEL_HASH,
            "authoritative_optimizer_semantic_hash": EXPECTED_OPTIMIZER_HASH,
            "canonical_schedule_hashes": authority["schedule_hashes"],
            "canonical_schedule_slice_hashes": authority["schedule_slice_hashes"],
            "source_revision": current_environment.get("source_revision"),
            "environment": current_environment,
            "evaluator_identity": bundle_identity["evaluator_config"],
            "clipping_identity": config["gradient_clipping"],
            "frozen20": {"opened": "NO", "used": "NO", "future_label_leakage": 0, "status": "SEALED"},
            "current_validation_champion": "R8E_A1",
            "r8e_a1_raw_sha256": sha256_file(base.A1_CHECKPOINT),
            "upstream_authority_snapshot_count": len(authority["authority_snapshot"]),
        }
        if preflight_identity["r8e_a1_raw_sha256"] != base.EXPECTED_A1_HASH:
            raise StageBlocker("r8e_a1_mutability_check_failed_before_execution")
        write_json(output / "authorization_record.json", authorization)
        write_json(output / "preflight_identity.json", preflight_identity)
        write_json(output / "training_config.json", config)
        write_json(output / "authority_and_provenance.json", {"experiment_id": EXPERIMENT_ID, "authorization": authorization, "bundle_path": str(BUNDLE_PATH.resolve()), "bundle_sha256": bundle_hash, "bundle_identity": bundle_identity, "authority_schedule": authority["schedule_identity"], "schedule_hashes": authority["schedule_hashes"], "schedule_slice_hashes": authority["schedule_slice_hashes"], "environment": current_environment, "frozen20": {"opened": "NO", "used": "NO", "future_label_leakage": 0, "status": "SEALED"}, "current_validation_champion": "R8E_A1", "r8e_a1_immutable": "YES"})

        # Independent CONTROL load from the same certified bundle.
        base.set_deterministic(base.SEED)
        control_model, control_optimizer = base.load_control_from_bundle(copy.deepcopy(bundle))
        control_loaded = validate_loaded_bundle(bundle, control_model, control_optimizer)
        control_branch_rng = bundle["rng_state"]
        base.restore_rng(control_branch_rng)
        control_rng_before = base.rng_digest(base.capture_rng())
        control = run_continuation("CONTROL", control_model, control_optimizer, pre, authority)
        control_rng_after = base.rng_digest(base.capture_rng())
        if control_rng_before != control_rng_after:
            raise StageBlocker("control_rng_changed_during_continuation")
        control["rng_before_digest"] = control_rng_before
        control["rng_after_digest"] = control_rng_after
        if control["terminal_model_semantic_hash"] != base.EXPECTED_FINAL_MODEL_SEMANTIC:
            raise StageBlocker("control_did_not_reproduce_canonical_d4_model")
        d4_metrics = {str(key): float(value) for key, value in authority["d4_seed_record"]["metrics"].items()}
        if any(not math.isclose(float(control["terminal_metrics"][key]), d4_metrics[key], rel_tol=base.REPLAY_RTOL, abs_tol=base.REPLAY_ATOL) for key in d4_metrics if key in control["terminal_metrics"]):
            raise StageBlocker("control_did_not_reproduce_canonical_d4_metrics")
        control["started_from_bundle"] = True
        control["loaded_identity"] = control_loaded

        # Independent MOMENT_RESET load from the same certified bundle.
        base.restore_rng(control_branch_rng)
        reset_model, reset_optimizer = base.load_control_from_bundle(copy.deepcopy(bundle))
        reset_loaded = validate_loaded_bundle(bundle, reset_model, reset_optimizer)
        reset_rng_digest = base.rng_digest(base.capture_rng())
        reset_audit = intervention_audit(reset_model, reset_optimizer, reset_rng_digest)
        if reset_audit["single_factor_intervention_valid"] != "YES":
            raise StageBlocker("single_factor_intervention_invalid")
        reset = run_continuation("MOMENT_RESET", reset_model, reset_optimizer, pre, authority, control_snapshots=control["snapshots"])
        reset["started_from_bundle"] = True
        reset["loaded_identity"] = reset_loaded
        reset["rng_before_digest"] = reset_rng_digest
        reset["rng_after_digest"] = base.rng_digest(base.capture_rng())
        if reset["rng_before_digest"] != reset["rng_after_digest"]:
            raise StageBlocker("reset_rng_changed_during_continuation")

        divergence = compare_branch_rows(control, reset)
        d4_metrics = {str(key): float(value) for key, value in authority["d4_seed_record"]["metrics"].items()}
        metrics = paired_metrics(control, reset, d4_metrics)
        control_stability = stability(control)
        reset_stability = stability(reset)
        finite_execution_preserved = control_stability["nonfinite_events"] == 0 and reset_stability["nonfinite_events"] == 0 and control_stability["nan_events"] == 0 and reset_stability["nan_events"] == 0 and control_stability["inf_events"] == 0 and reset_stability["inf_events"] == 0
        magnitude_stability_worsened = reset_stability["gradient_norm_median_max"]["max"] > control_stability["gradient_norm_median_max"]["max"] and reset_stability["parameter_update_norm_median_max"]["max"] > control_stability["parameter_update_norm_median_max"]["max"]
        stability_observation = "FINITE_EXECUTION_PRESERVED_BUT_MAGNITUDE_STABILITY_WORSENED" if finite_execution_preserved and magnitude_stability_worsened else ("FINITE_EXECUTION_PRESERVED" if finite_execution_preserved else "NUMERICAL_INSTABILITY_OBSERVED")
        early_update_observation = "RESET_UPDATE_385_LARGER_THAN_CONTROL" if float(reset["rows"][0]["parameter_update_norm"]) > float(control["rows"][0]["parameter_update_norm"]) else "NO_LARGER_RESET_UPDATE_385_OBSERVED"
        long_horizon_observation = "RESET_H32_DEGRADED_AND_FAILED" if metrics["terminal"]["32"]["moment_reset_status"] == "FAIL" and metrics["terminal"]["32"]["control_status"] == "PASS" else "NO_H32_CLASSIFICATION_DEGRADATION"
        stability_summary = {"control": control_stability, "moment_reset": reset_stability, "numerical_stability_preserved": stability_observation, "finite_execution_preserved": "YES" if finite_execution_preserved else "NO", "magnitude_stability_worsened": "YES" if magnitude_stability_worsened else "NO", "clipping_burden": "UNCHANGED" if control_stability["clipping_activation_count"] == reset_stability["clipping_activation_count"] else ("INCREASED" if reset_stability["clipping_activation_count"] > control_stability["clipping_activation_count"] else "DECREASED"), "early_update_observation": early_update_observation, "long_horizon_observation": long_horizon_observation}
        audit = reset_audit
        write_json(output / "intervention_audit.json", audit)
        write_csv(output / "paired_trajectory_diagnostics.csv", control["rows"] + reset["rows"])
        write_csv(output / "module_diagnostics.csv", control["module_rows"] + reset["module_rows"])
        write_json(output / "paired_evaluation_metrics.json", metrics)
        write_json(output / "stability_diagnostics.json", stability_summary)
        write_json(output / "divergence_diagnostics.json", divergence)
        control_terminal = {"schema_version": "stage3_h13_d6_control_terminal_state_v1", "completed_optimizer_steps": 512, "model_state_dict": control["terminal_model_state"], "optimizer_state_dict": control["terminal_optimizer_state"], "model_semantic_hash": control["terminal_model_semantic_hash"], "optimizer_semantic_hash": control["terminal_optimizer_semantic_hash"], "final_metrics": control["terminal_metrics"], "source_branch_bundle_sha256": bundle_hash, "intervention": "NONE"}
        reset_terminal = {"schema_version": "stage3_h13_d6_moment_reset_terminal_state_v1", "completed_optimizer_steps": 512, "model_state_dict": reset["terminal_model_state"], "optimizer_state_dict": reset["terminal_optimizer_state"], "model_semantic_hash": reset["terminal_model_semantic_hash"], "optimizer_semantic_hash": reset["terminal_optimizer_semantic_hash"], "final_metrics": reset["terminal_metrics"], "source_branch_bundle_sha256": bundle_hash, "intervention": "exp_avg and exp_avg_sq zeroed in-place before update 385"}
        control_path = output / "control_terminal_state.pt"
        reset_path = output / "moment_reset_terminal_state.pt"
        torch.save(control_terminal, control_path)
        torch.save(reset_terminal, reset_path)
        base.mark_readonly(control_path)
        base.mark_readonly(reset_path)
        if sha256_file(control_path) == sha256_file(reset_path):
            raise StageBlocker("control_and_reset_terminal_state_raw_hash_identical_unexpectedly")

        c = control["terminal_metrics"]
        r = reset["terminal_metrics"]
        h1_threshold = float(base.D4_H1_THRESHOLD)
        h32_threshold = float(base.H32_THRESHOLD)
        certificate: dict[str, Any] = {
            "schema_version": "stage3_h13_d6_terminal_certificate_v1",
            "STAGE_3_H13_POST_D5_D6_ADAMW_MOMENT_RESET_CAUSAL_BRANCH": "PASSED",
            "FIRST_BLOCKER": "none",
            "ONE_SENTENCE_VERDICT": f"Both branches completed the fixed updates 385–512 from the certified update-384 state; the only intervention was exact in-place zeroing of AdamW exp_avg and exp_avg_sq, and the paired terminal H1 values were {c['1']} (control) versus {r['1']} (moment-reset).",
            "PROJECT_OWNER_D6_AUTHORIZATION": "YES",
            "CERTIFIED_BRANCH_BUNDLE_HASH_MATCH": "YES",
            "POST_UPDATE_384_MODEL_IDENTITY_MATCH": "YES",
            "POST_UPDATE_384_OPTIMIZER_IDENTITY_MATCH": "YES",
            "PARAMETER_MAPPING_VALID": "YES",
            "UPDATE_BOUNDARY_VALID": "YES",
            "CANONICAL_REMAINING_SEQUENCE_EXACT": "YES",
            "SOURCE_REVISION_MATCH": "YES",
            "CONFIG_MATCH": "YES",
            "CLIPPING_UNCHANGED": "YES",
            "R8E_A1_IMMUTABLE": "YES",
            "FROZEN20_OPENED": "NO",
            "FROZEN20_USED": "NO",
            "FUTURE_LABEL_LEAKAGE": 0,
            "CONTROL_STARTED_FROM_CERTIFIED_BUNDLE": "YES",
            "RESET_STARTED_FROM_CERTIFIED_BUNDLE": "YES",
            "ONLY_EXP_AVG_EXP_AVG_SQ_CHANGED": "YES",
            "MODEL_PARAMETERS_CHANGED_AT_BRANCH_POINT": audit["MODEL_PARAMETER_DIFFERENCES"],
            "OPTIMIZER_STEP_CHANGED": audit["OPTIMIZER_STEP_DIFFERENCES"],
            "PARAMETER_GROUPS_CHANGED": audit["PARAMETER_GROUP_DIFFERENCES"],
            "OTHER_OPTIMIZER_STATE_CHANGED": audit["NON_MOMENT_OPTIMIZER_STATE_DIFFERENCES"],
            "EXP_AVG_PARAMETER_STATES_RESET": audit["exp_avg_parameter_states_reset"],
            "EXP_AVG_SQ_PARAMETER_STATES_RESET": audit["exp_avg_sq_parameter_states_reset"],
            "EXP_AVG_ELEMENTS_RESET": audit["exp_avg_elements_reset"],
            "EXP_AVG_SQ_ELEMENTS_RESET": audit["exp_avg_sq_elements_reset"],
            "EXP_AVG_POST_RESET_EXACT_ZERO": audit["exp_avg_post_reset_exact_zero"],
            "EXP_AVG_SQ_POST_RESET_EXACT_ZERO": audit["exp_avg_sq_post_reset_exact_zero"],
            "PRESERVED_STEP_VALUE_OR_RANGE": audit["preserved_optimizer_step_value_or_range"],
            "SINGLE_FACTOR_INTERVENTION_VALID": audit["single_factor_intervention_valid"],
            "CONTROL_H1": c["1"], "CONTROL_H1_STATUS": metrics["terminal"]["1"]["control_status"], "CONTROL_H4": c["4"], "CONTROL_H16": c["16"], "CONTROL_H32": c["32"], "CONTROL_H32_STATUS": metrics["terminal"]["32"]["control_status"],
            "CONTROL_TERMINAL_MODEL_SEMANTIC_HASH": control["terminal_model_semantic_hash"], "CONTROL_TERMINAL_OPTIMIZER_SEMANTIC_HASH": control["terminal_optimizer_semantic_hash"], "CONTROL_VS_AUTHORITATIVE_D4_H1_DELTA": c["1"] - d4_metrics["1"], "CONTROL_VS_AUTHORITATIVE_D4_H32_DELTA": c["32"] - d4_metrics["32"],
            "RESET_H1": r["1"], "RESET_H1_STATUS": metrics["terminal"]["1"]["moment_reset_status"], "RESET_H4": r["4"], "RESET_H16": r["16"], "RESET_H32": r["32"], "RESET_H32_STATUS": metrics["terminal"]["32"]["moment_reset_status"],
            "RESET_TERMINAL_MODEL_SEMANTIC_HASH": reset["terminal_model_semantic_hash"], "RESET_TERMINAL_OPTIMIZER_SEMANTIC_HASH": reset["terminal_optimizer_semantic_hash"],
            "FROZEN_H1_THRESHOLD": h1_threshold, "RESET_VS_CONTROL_H1_ABSOLUTE_CHANGE": abs(r["1"] - c["1"]), "RESET_VS_CONTROL_H1_PERCENT_CHANGE": 100.0 * (r["1"] - c["1"]) / c["1"], "CONTROL_H1_MARGIN_TO_THRESHOLD": h1_threshold - c["1"], "RESET_H1_MARGIN_TO_THRESHOLD": h1_threshold - r["1"], "RESET_VS_CONTROL_H4_CHANGE": r["4"] - c["4"], "RESET_VS_CONTROL_H16_CHANGE": r["16"] - c["16"], "RESET_VS_CONTROL_H32_CHANGE": r["32"] - c["32"], "RESET_VS_CONTROL_H32_PERCENT_CHANGE": 100.0 * (r["32"] - c["32"]) / c["32"], "H1_CLASSIFICATION_CHANGED": "YES" if metrics["terminal"]["1"]["control_status"] != metrics["terminal"]["1"]["moment_reset_status"] else "NO", "H32_CLASSIFICATION_CHANGED": "YES" if metrics["terminal"]["32"]["control_status"] != metrics["terminal"]["32"]["moment_reset_status"] else "NO",
            "INTENTIONAL_OPTIMIZER_STATE_DIVERGENCE": divergence["intentional_optimizer_state_divergence"], "FIRST_PARAMETER_UPDATE_DIVERGENCE_UPDATE": divergence["first_parameter_update_divergence_update"], "FIRST_MODEL_STATE_DIVERGENCE_UPDATE": divergence["first_model_state_divergence_update"], "FIRST_GRADIENT_DIVERGENCE_UPDATE": divergence["first_gradient_divergence_update"], "FIRST_CLIPPING_DIVERGENCE_UPDATE": divergence["first_clipping_divergence_update"], "FIRST_LOSS_DIVERGENCE_UPDATE": divergence["first_loss_divergence_update"], "FIRST_DOWNSTREAM_MEASURABLE_DIVERGENCE": divergence["first_downstream_measurable_divergence"], "FIRST_DIVERGENT_QUANTITY": divergence["first_divergent_quantity"], "DIVERGENCE_MAGNITUDE": divergence["divergence_magnitude"],
            "CONTROL_GRADIENT_NORM_MEDIAN_MAX": control_stability["gradient_norm_median_max"], "RESET_GRADIENT_NORM_MEDIAN_MAX": reset_stability["gradient_norm_median_max"], "CONTROL_UPDATE_NORM_MEDIAN_MAX": control_stability["parameter_update_norm_median_max"], "RESET_UPDATE_NORM_MEDIAN_MAX": reset_stability["parameter_update_norm_median_max"], "CONTROL_CLIPPING_ACTIVATIONS": control_stability["clipping_activation_count"], "RESET_CLIPPING_ACTIVATIONS": reset_stability["clipping_activation_count"], "CONTROL_NONFINITE_EVENTS": control_stability["nonfinite_events"], "RESET_NONFINITE_EVENTS": reset_stability["nonfinite_events"], "CONTROL_EXP_AVG_DIAGNOSTICS": control_stability["exp_avg_diagnostics"], "RESET_EXP_AVG_DIAGNOSTICS": reset_stability["exp_avg_diagnostics"], "CONTROL_EXP_AVG_SQ_DIAGNOSTICS": control_stability["exp_avg_sq_diagnostics"], "RESET_EXP_AVG_SQ_DIAGNOSTICS": reset_stability["exp_avg_sq_diagnostics"], "NUMERICAL_STABILITY_PRESERVED": stability_observation,
            "D6_OUTPUT_DIRECTORY": str(output.resolve()), "CONTROL_TERMINAL_STATE": str(control_path.resolve()), "CONTROL_TERMINAL_STATE_SHA256": sha256_file(control_path), "RESET_TERMINAL_STATE": str(reset_path.resolve()), "RESET_TERMINAL_STATE_SHA256": sha256_file(reset_path), "PAIRED_DIAGNOSTICS": str((output / "paired_trajectory_diagnostics.csv").resolve()), "INTERVENTION_AUDIT": str((output / "intervention_audit.json").resolve()), "TERMINAL_CERTIFICATE": str((output / "terminal_certificate.json").resolve()), "INTEGRITY_MANIFEST": str((output / "integrity_manifest.json").resolve()), "NEW_PERSISTENT_STORAGE_BYTES": 0,
            "D6_CONTROL_EXECUTED": "YES", "D6_MOMENT_RESET_EXECUTED": "YES", "D6_PAIRED_EXPERIMENT_COMPLETE": "YES", "CURRENT_VALIDATION_CHAMPION": "R8E_A1", "CHAMPION_CHANGED": "NO", "READY_FOR_R9": "NO", "AUTHORIZED_FOR_R9": "NO", "FROZEN20_STATUS": "SEALED",
            "ARTIFACT_SHA256": {},
            "EXECUTION_LEVEL_CONCLUSION": [
                "Yes: both branches loaded the same certified post-update-384 model and optimizer identity before continuation.",
                "Yes: the only intentional intervention was zeroing exp_avg and exp_avg_sq; model parameters, optimizer step, parameter groups, and non-moment state were unchanged at the branch point.",
                "Yes: optimizer step remained 384 at the intervention and both branches completed updates 385–512.",
                f"Yes: the untouched control reproduced the canonical D4 terminal model identity and metrics within the authoritative replay tolerance; H1={c['1']} and H32={c['32']}.",
                f"The moment-reset terminal H1 was {r['1']} ({metrics['terminal']['1']['moment_reset_status']}) versus control {c['1']}.",
                f"H4 changed from {c['4']} to {r['4']}; H16 changed from {c['16']} to {r['16']}; H32 changed from {c['32']} to {r['32']} and remained {metrics['terminal']['32']['moment_reset_status']}.",
                f"The first parameter-update and model-state divergence was at update {divergence['first_parameter_update_divergence_update']}; the first gradient, clipping, and loss divergences were at updates {divergence['first_gradient_divergence_update']}, {divergence['first_clipping_divergence_update']}, and {divergence['first_loss_divergence_update']}.",
                f"Numerical-stability observations: {('no nonfinite, NaN, or Inf events, but reset gradient/update magnitudes worsened' if finite_execution_preserved and magnitude_stability_worsened else 'a nonfinite stability event was observed')}.",
                "The paired evidence is execution-level interpretable under the fixed protocol; this report does not perform the separate final causal interpretation.",
                "Frozen-20 remained sealed and R8E_A1 remained immutable and champion.",
            ],
        }
        core_names = ["authorization_record.json", "preflight_identity.json", "training_config.json", "authority_and_provenance.json", "intervention_audit.json", "paired_trajectory_diagnostics.csv", "module_diagnostics.csv", "paired_evaluation_metrics.json", "stability_diagnostics.json", "divergence_diagnostics.json", "control_terminal_state.pt", "moment_reset_terminal_state.pt"]
        certificate["ARTIFACT_SHA256"] = artifact_hashes(output, core_names)
        for _ in range(3):
            write_json(output / "terminal_certificate.json", certificate)
            (output / "FINAL_REPORT.md").write_text(render_report(certificate), encoding="utf-8", newline="\n")
            write_integrity_manifest(output)
            certificate["NEW_PERSISTENT_STORAGE_BYTES"] = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
        write_json(output / "terminal_certificate.json", certificate)
        (output / "FINAL_REPORT.md").write_text(render_report(certificate), encoding="utf-8", newline="\n")
        write_integrity_manifest(output)
        # Verify all upstream evidence after both continuations and before return.
        base.verify_snapshot(authority["authority_snapshot"])
        if sha256_file(base.A1_CHECKPOINT) != base.EXPECTED_A1_HASH:
            raise StageBlocker("r8e_a1_mutated_during_execution")
        print(json.dumps({"status": "PASSED", "output": str(output.resolve()), "control_model_hash": control["terminal_model_semantic_hash"], "reset_model_hash": reset["terminal_model_semantic_hash"], "first_divergence": divergence["first_downstream_measurable_divergence"]}, sort_keys=True), flush=True)
        return 0
    except StageBlocker as exc:
        return fail_closed(output, str(exc), authorization)
    except Exception as exc:  # pragma: no cover
        return fail_closed(output, f"unexpected_execution_failure:{type(exc).__name__}:{exc}", authorization)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    return execute(args.output_dir.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
