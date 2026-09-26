"""Materialize and qualify the prospective post-update-384 D6 branch state.

This is a deliberately separate runner for the H13 Stage 3 branch-state gate.
It replays the authoritative clipped D4 recipe from R6 for exactly 384
optimizer steps, saves the live model and AdamW state, reloads that bundle,
and runs one untouched 128-step continuation.  No optimizer moment is reset
and no D6 counterfactual is implemented here.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import math
import os
import platform
import random
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stage3_h13_post_d4_h1_blocker_audit import (  # noqa: E402
    A1_CHECKPOINT,
    A1_CONFIG,
    D4_H1_THRESHOLD,
    EXPECTED_A1_CANONICAL,
    EXPECTED_A1_HASH,
    EXPECTED_CANONICAL_SCHEDULE,
    EXPECTED_CANONICAL_STORAGE,
    EXPECTED_D4_AGGREGATE,
    EXPECTED_D4_CERTIFICATE,
    EXPECTED_D4_DIR,
    EXPECTED_D4_ENVIRONMENT,
    EXPECTED_D4_IMMUTABLE,
    EXPECTED_D4_SCHEDULE_IDENTITY,
    EXPECTED_D4_SCHEDULE_STORAGE,
    EXPECTED_D4_SUMMARY,
    EXPECTED_R6_HASH,
    H32_THRESHOLD,
    R6_CHECKPOINT,
    aggregate_norm,
    aggregate_optimizer_state,
    causal_paired_rollout,
    evaluate_compact,
    finite_float,
    gradients_are_finite,
    load_json,
    load_model,
    parameter_rows,
    preflight,
    read_schedule,
    schedule_indices,
    set_deterministic,
    tensor_batch,
    training_losses,
)
from scripts.stage3_h13_r8e_post_d4_canonical_batch_robustness import (  # noqa: E402
    EXPECTED_SOURCE_COMMIT,
    PHASE_HORIZONS,
)
from src.stage3_h13_r8e_d2 import canonical_model_state_sha256  # noqa: E402


EXPERIMENT_ID = "STAGE_3_H13_POST_D5_D6_BRANCH_STATE_MATERIALIZATION_REPLAY"
SEED = 3830844401
TOTAL_UPDATES = 512
BRANCH_UPDATE = 384
REMAINING_UPDATES = TOTAL_UPDATES - BRANCH_UPDATE
BATCH_SIZE = 256
LEARNING_RATE = 5.0e-5
BETAS = (0.9, 0.999)
EPS = 1.0e-8
WEIGHT_DECAY = 1.0e-5
CLIP_MAX_NORM = 1.0
LAMBDA_H1 = 1.0
LAMBDA_HIDDEN = 0.005
HUBER_BETA = 0.001
PROBE_UPDATES = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512)
EXPECTED_FINAL_MODEL_SEMANTIC = "281008502db053be3bf5945bd95a44391ef2f17c63385221cce61c178abb376e"
EXPECTED_D4_SEED_CHECKPOINT_RAW = "57364d72afe74f2924fffb8d99f04f1c9d3f7d1592e62c9e41efebe718ba0352"
D5_ROOT = ROOT / "outputs/stage3_h13_post_d4_d5_no_clip_causal_replay_20260816T203000Z"
D2_ROOT = ROOT / "outputs/stage3_h13_r8e_d2_prospective_causal_design_20260816T053129Z"
D4_H1_ROOT = ROOT / "outputs/stage3_h13_post_d4_h1_blocker_audit_20260816T130000Z"
D4_H1_TRAJECTORY = D4_H1_ROOT / "trajectory_diagnostics.csv"
D4_SEED_CHECKPOINT = EXPECTED_D4_DIR / f"checkpoints/canonical_seed_{SEED}.pt"
AUTHORIZATION_PATH = Path(r"C:\Users\86198\.codex\attachments\8f9c27df-d7cf-44cb-8834-193378aa9a33\pasted-text.txt")
REPLAY_RTOL = 5.0e-6
REPLAY_ATOL = 5.0e-9


class StageBlocker(RuntimeError):
    """Fail-closed blocker for this stage."""


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


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    return value


def tensor_hash(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    header = canonical_json({"dtype": str(tensor.dtype), "shape": list(tensor.shape)})
    raw = tensor.numpy().tobytes()
    return sha256_bytes(header + b"\0" + raw)


def canonical_tensor_state_hash(state: Mapping[str, Any]) -> str:
    rows = []
    for name in sorted(state):
        value = state[name]
        if isinstance(value, torch.Tensor):
            rows.append({"name": str(name), "dtype": str(value.dtype), "shape": list(value.shape), "sha256": tensor_hash(value)})
        else:
            rows.append({"name": str(name), "value": jsonable(value)})
    return sha256_bytes(canonical_json(rows))


def semantic_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {"tensor": tensor_hash(value), "dtype": str(value.dtype), "shape": list(value.shape)}
    if isinstance(value, Mapping):
        return {str(key): semantic_value(value[key]) for key in sorted(value, key=lambda item: str(item))}
    if isinstance(value, (tuple, list)):
        return [semantic_value(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def optimizer_canonical_payload(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    serialized = optimizer.state_dict()
    live_names: dict[int, str] = dict(getattr(optimizer, "_codex_parameter_names", {}))
    if not live_names:
        raise StageBlocker("optimizer_parameter_name_mapping_missing")
    param_names_by_id: dict[int, str] = {}
    groups = []
    for group_index, (live_group, saved_group) in enumerate(zip(optimizer.param_groups, serialized["param_groups"])):
        names = []
        for parameter, saved_id in zip(live_group["params"], saved_group["params"]):
            name = live_names[id(parameter)]
            param_names_by_id[int(saved_id)] = name
            names.append(name)
        group_payload = {str(key): semantic_value(value) for key, value in saved_group.items() if key != "params"}
        group_payload["parameter_names"] = names
        group_payload["group_index"] = group_index
        groups.append(group_payload)
    state = {}
    for saved_id, state_value in serialized["state"].items():
        name = param_names_by_id[int(saved_id)]
        state[name] = semantic_value(state_value)
    return {"param_groups": groups, "state": {name: state[name] for name in sorted(state)}}


def optimizer_semantic_hash(optimizer: torch.optim.Optimizer) -> str:
    return sha256_bytes(canonical_json(optimizer_canonical_payload(optimizer)))


def optimizer_state_from_names(optimizer: torch.optim.Optimizer) -> dict[str, dict[str, Any]]:
    return {name: optimizer.state[parameter] for name, parameter in named_optimizer_parameters(optimizer)}


def named_optimizer_parameters(optimizer: torch.optim.Optimizer) -> list[tuple[str, torch.nn.Parameter]]:
    names = dict(getattr(optimizer, "_codex_parameter_names", {}))
    if not names:
        raise StageBlocker("optimizer_parameter_name_mapping_missing")
    return [(names[id(parameter)], parameter) for group in optimizer.param_groups for parameter in group["params"]]


def named_optimizer_component_hash(optimizer: torch.optim.Optimizer, component: str) -> str:
    rows = []
    for name, parameter in named_optimizer_parameters(optimizer):
        state = optimizer.state.get(parameter, {})
        if component not in state:
            raise StageBlocker(f"optimizer_component_missing:{component}:{name}")
        value = state[component]
        if isinstance(value, torch.Tensor):
            rows.append({"name": name, "dtype": str(value.dtype), "shape": list(value.shape), "sha256": tensor_hash(value)})
        else:
            rows.append({"name": name, "value": jsonable(value)})
    return sha256_bytes(canonical_json(rows))


def optimizer_step_map(optimizer: torch.optim.Optimizer) -> dict[str, int]:
    result = {}
    for name, parameter in named_optimizer_parameters(optimizer):
        raw = optimizer.state.get(parameter, {}).get("step")
        if raw is None:
            raise StageBlocker(f"optimizer_step_missing:{name}")
        result[name] = int(raw.item()) if isinstance(raw, torch.Tensor) else int(raw)
    return result


def optimizer_mapping(optimizer: torch.optim.Optimizer) -> list[dict[str, Any]]:
    serialized = optimizer.state_dict()
    mapping = []
    for group_index, (live_group, saved_group) in enumerate(zip(optimizer.param_groups, serialized["param_groups"])):
        for parameter, saved_id in zip(live_group["params"], saved_group["params"]):
            names = dict(getattr(optimizer, "_codex_parameter_names", {}))
            name = names.get(id(parameter))
            if name is None:
                raise StageBlocker("optimizer_parameter_name_mapping_missing")
            state = optimizer.state.get(parameter, {})
            if not all(key in state for key in ("step", "exp_avg", "exp_avg_sq")):
                raise StageBlocker(f"optimizer_state_incomplete:{name}")
            mapping.append({
                "optimizer_param_id": int(saved_id),
                "parameter_name": name,
                "group_index": group_index,
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype),
                "numel": int(parameter.numel()),
                "state_keys": sorted(str(key) for key in state),
                "step": int(state["step"].item()) if isinstance(state["step"], torch.Tensor) else int(state["step"]),
                "exp_avg_sha256": tensor_hash(state["exp_avg"]),
                "exp_avg_sq_sha256": tensor_hash(state["exp_avg_sq"]),
            })
    return mapping


def capture_rng() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().cpu().clone(),
        "cuda": [item.cpu().clone() for item in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else None,
    }


def rng_digest(state: Mapping[str, Any]) -> str:
    cuda = None if state.get("cuda") is None else [item.cpu().numpy().tobytes() for item in state["cuda"]]
    payload = (state["python"], state["numpy"], state["torch_cpu"].cpu().numpy().tobytes(), cuda)
    return sha256_bytes(__import__("pickle").dumps(payload, protocol=4))


def restore_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def state_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if list(left) != list(right):
        return False
    for key in left:
        a, b = left[key], right[key]
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            if not torch.equal(a.detach().cpu(), b.detach().cpu()):
                return False
        elif a != b:
            return False
    return True


def snapshot_paths(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    records = {}
    for path in sorted({Path(item).resolve() for item in paths}):
        if not path.is_file():
            raise StageBlocker(f"authoritative_path_missing:{path}")
        records[str(path)] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    return records


def verify_snapshot(records: Mapping[str, Mapping[str, Any]]) -> None:
    for raw_path, record in records.items():
        path = Path(raw_path)
        if not path.is_file() or sha256_file(path) != record["sha256"] or path.stat().st_size != int(record["size_bytes"]):
            raise StageBlocker(f"authoritative_input_mutated:{raw_path}")


def schedule_slice_hash(ordered_batches: Sequence[Mapping[str, Any]], start: int, end: int) -> str:
    return sha256_bytes(canonical_json(list(ordered_batches[start:end])))


def batch_digest(schedule_manifest: Mapping[str, Any], index: int) -> str:
    ids = [int(value) for value in schedule_manifest["ordered_batches"][index]["window_ids"]]
    return sha256_bytes(json.dumps(ids, separators=(",", ":")).encode("utf-8"))


def optimizer_contract(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    group = optimizer.param_groups[0]
    keys = ("amsgrad", "foreach", "fused", "capturable", "differentiable", "maximize", "decoupled_weight_decay")
    flags = {key: group.get(key, optimizer.defaults.get(key)) for key in keys}
    flags = {key: (None if value is None else bool(value)) for key, value in flags.items()}
    return {
        "class": optimizer.__class__.__name__,
        "learning_rate": float(group["lr"]),
        "betas": [float(group["betas"][0]), float(group["betas"][1])],
        "eps": float(group["eps"]),
        "weight_decay": float(group["weight_decay"]),
        **flags,
    }


def environment() -> dict[str, Any]:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
    source = result.stdout.strip() if result.returncode == 0 else None
    return {
        "captured_at_utc": utc_now(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pytorch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": "cpu",
        "dtype": "float32",
        "source_revision": source,
        "source_revision_match": source == EXPECTED_SOURCE_COMMIT,
        "deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "dataloader_workers": 0,
        "native_runtime": "Windows CPU; direct tensor batches; no DataLoader worker processes",
        "residual_nondeterminism": "cross_platform_bitwise_recreation_not_claimed",
    }


def immutable_authority_paths() -> list[Path]:
    roots = [EXPECTED_D4_DIR, D4_H1_ROOT, D5_ROOT, D2_ROOT]
    paths = [R6_CHECKPOINT, A1_CHECKPOINT, A1_CONFIG]
    for root in roots:
        paths.extend(item for item in root.rglob("*") if item.is_file())
    return paths


def verify_authority(pre: Mapping[str, Any]) -> dict[str, Any]:
    identity, schedule_manifest, schedule_hashes = read_schedule()
    if schedule_hashes != {"logical_sha256": EXPECTED_CANONICAL_SCHEDULE, "storage_sha256": EXPECTED_CANONICAL_STORAGE}:
        raise StageBlocker("canonical_schedule_hash_mismatch")
    schedule_rows = schedule_indices(schedule_manifest, pre["train_data"].window_ids)
    if schedule_rows.shape != (TOTAL_UPDATES, BATCH_SIZE):
        raise StageBlocker("canonical_schedule_shape_mismatch")
    if int(schedule_manifest.get("number_of_training_updates", -1)) != TOTAL_UPDATES or int(schedule_manifest.get("batch_size", -1)) != BATCH_SIZE:
        raise StageBlocker("canonical_schedule_dimensions_mismatch")
    if sha256_file(R6_CHECKPOINT) != EXPECTED_R6_HASH or sha256_file(A1_CHECKPOINT) != EXPECTED_A1_HASH:
        raise StageBlocker("authoritative_checkpoint_hash_mismatch")
    if sha256_file(D4_SEED_CHECKPOINT) != EXPECTED_D4_SEED_CHECKPOINT_RAW:
        raise StageBlocker("authoritative_d4_seed_checkpoint_hash_mismatch")
    d4_certificate = load_json(EXPECTED_D4_CERTIFICATE)
    d5_certificate = load_json(D5_ROOT / "terminal_certificate.json")
    d2_certificate = load_json(D2_ROOT / "terminal_certificate.json")
    if d4_certificate.get("CURRENT_VALIDATION_CHAMPION") != "R8E_A1" or d4_certificate.get("CHAMPION_CHANGED") != "NO":
        raise StageBlocker("upstream_champion_status_changed")
    if d5_certificate.get("STAGE_3_H13_POST_D4_D5_NO_CLIP_CAUSAL_REPLAY") != "PASSED":
        raise StageBlocker("historical_d5_evidence_not_valid")
    if d5_certificate.get("CURRENT_VALIDATION_CHAMPION") != "R8E_A1" or d5_certificate.get("CHAMPION_CHANGED") != "NO":
        raise StageBlocker("historical_d5_champion_status_changed")
    if d2_certificate.get("HISTORICAL_RAW_CHECKPOINT_RECOVERY_STATUS") != "UNRECOVERABLE_BY_DESIGN":
        raise StageBlocker("historical_d6_design_evidence_changed")
    a1 = load_model(A1_CHECKPOINT)
    if canonical_model_state_sha256(a1.state_dict()) != EXPECTED_A1_CANONICAL:
        raise StageBlocker("r8e_a1_semantic_hash_mismatch")
    d4_environment = load_json(EXPECTED_D4_ENVIRONMENT)
    current_environment = environment()
    for key in ("python_version", "pytorch_version", "numpy_version", "cuda_available", "device", "dtype", "source_revision"):
        if current_environment.get(key) != d4_environment.get("source_commit" if key == "source_revision" else key):
            raise StageBlocker(f"environment_identity_mismatch:{key}")
    d4_summary = load_json(EXPECTED_D4_SUMMARY)
    d4_aggregate = load_json(EXPECTED_D4_AGGREGATE)
    d4_seed_record = d4_summary["seeds"][str(SEED)]
    if d4_seed_record.get("canonical_model_state_sha256") != EXPECTED_FINAL_MODEL_SEMANTIC:
        raise StageBlocker("authoritative_d4_terminal_model_hash_mismatch")
    if not math.isclose(float(d4_aggregate["metrics"]["H1"]["median"]), 9.019693982113058e-05, rel_tol=0.0, abs_tol=1.0e-18):
        raise StageBlocker("authoritative_d4_h1_mismatch")
    if not math.isclose(float(d4_aggregate["metrics"]["H32"]["median"]), 0.016745967327457343, rel_tol=0.0, abs_tol=1.0e-18):
        raise StageBlocker("authoritative_d4_h32_mismatch")
    schedule_ordered = schedule_manifest["ordered_batches"]
    authority_snapshot = snapshot_paths(immutable_authority_paths())
    return {
        "schedule_identity": identity,
        "schedule_manifest": schedule_manifest,
        "schedule_hashes": schedule_hashes,
        "schedule_rows": schedule_rows,
        "d4_certificate": d4_certificate,
        "d5_certificate": d5_certificate,
        "d2_certificate": d2_certificate,
        "d4_environment": d4_environment,
        "current_environment": current_environment,
        "d4_summary": d4_summary,
        "d4_aggregate": d4_aggregate,
        "d4_seed_record": d4_seed_record,
        "authority_snapshot": authority_snapshot,
        "schedule_slice_hashes": {
            "full": EXPECTED_CANONICAL_SCHEDULE,
            "prefix_384": schedule_slice_hash(schedule_ordered, 0, BRANCH_UPDATE),
            "suffix_128": schedule_slice_hash(schedule_ordered, BRANCH_UPDATE, TOTAL_UPDATES),
        },
        "d4_h1_trajectory_sha256": sha256_file(D4_H1_TRAJECTORY),
    }


def probe_row(step: int, metrics: Mapping[str, float], update: Mapping[str, Any] | None, next_batch: str | None, segment: str) -> dict[str, Any]:
    row = {
        "update": int(step),
        "H1": metrics.get("1"),
        "H4": metrics.get("4"),
        "H16": metrics.get("16"),
        "H32": metrics.get("32"),
        "training_loss": None if update is None else update["training_loss_total"],
        "raw_gradient_norm_before_clipping": None if update is None else update["raw_gradient_norm"],
        "clipped_gradient_norm": None if update is None else update["clipped_gradient_norm"],
        "clipping_activated": None if update is None else update["clipping_activated"],
        "actual_parameter_update_norm": None if update is None else update["parameter_update_norm"],
        "optimizer_step": None if update is None else update["adamw_step"],
        "exp_avg_norm": None if update is None else update["exp_avg_norm"],
        "exp_avg_sq_norm": None if update is None else update["exp_avg_sq_norm"],
        "nonfinite_detected": "NO",
        "last_batch_window_sha256": None if update is None else update["batch_window_sha256"],
        "next_batch_window_sha256": next_batch,
        "segment": segment,
    }
    return row


def run_segment(
    model: Any,
    optimizer: torch.optim.Optimizer,
    train_data: Any,
    validation: Any,
    channels: Mapping[str, Any],
    schedule_rows: np.ndarray,
    schedule_manifest: Mapping[str, Any],
    start_index: int,
    end_index: int,
    probe_values: dict[int, dict[str, float]],
    probe_rows: dict[int, dict[str, Any]],
    update_trace: dict[int, dict[str, Any]],
    all_update_rows: list[dict[str, Any]],
) -> None:
    names_and_parameters = parameter_rows(model)
    model.train()
    for zero_based_index in range(start_index, end_index):
        completed_step = zero_based_index + 1
        horizon = int(PHASE_HORIZONS[zero_based_index // 128])
        batch = tensor_batch(train_data, schedule_rows[zero_based_index])
        optimizer.zero_grad(set_to_none=True)
        rollout = causal_paired_rollout(
            model,
            batch["inputs"],
            batch["history_positions"],
            batch["history_times"],
            batch["target_times"],
            batch["starts"],
            batch["ends"],
            channels,
            horizon=horizon,
            target_positions_for_teacher=batch["target_positions"],
            hidden_consistency_enabled=True,
        )
        losses = training_losses(
            rollout["predictions"],
            batch["target_positions"],
            rollout["hidden_loss"],
            lambda_hidden=LAMBDA_HIDDEN,
            lambda_h1=LAMBDA_H1,
            beta=HUBER_BETA,
        )
        if not all(bool(torch.isfinite(value)) for value in losses.values()):
            raise StageBlocker(f"nonfinite_training_loss_update_{completed_step}")
        losses["total"].backward()
        if not gradients_are_finite(model):
            raise StageBlocker(f"nonfinite_raw_gradient_update_{completed_step}")
        before_params = {name: parameter.detach().cpu().clone() for name, parameter in names_and_parameters}
        raw_grads = {name: parameter.grad.detach().cpu().clone() for name, parameter in names_and_parameters if parameter.grad is not None}
        raw_norm = aggregate_norm(raw_grads.values())
        pre_clip_norm = finite_float(torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_MAX_NORM))
        if not math.isclose(raw_norm, pre_clip_norm, rel_tol=1.0e-5, abs_tol=1.0e-5):
            raise StageBlocker(f"gradient_norm_instrumentation_mismatch_update_{completed_step}")
        clipped_grads = [parameter.grad.detach().cpu().clone() for _, parameter in names_and_parameters if parameter.grad is not None]
        clipped_norm = aggregate_norm(clipped_grads)
        clipping_activated = "YES" if pre_clip_norm > CLIP_MAX_NORM else "NO"
        optimizer.step()
        after_params = {name: parameter.detach().cpu().clone() for name, parameter in names_and_parameters}
        if not all(bool(torch.all(torch.isfinite(value))) for value in after_params.values()):
            raise StageBlocker(f"nonfinite_model_parameter_update_{completed_step}")
        for state in optimizer.state.values():
            for value in state.values():
                if isinstance(value, torch.Tensor) and not bool(torch.all(torch.isfinite(value))):
                    raise StageBlocker(f"nonfinite_optimizer_state_update_{completed_step}")
        update_norm = aggregate_norm(after_params[name] - before_params[name] for name, _ in names_and_parameters)
        adamw = aggregate_optimizer_state(model, optimizer, before_params, LEARNING_RATE, WEIGHT_DECAY, EPS, BETAS)
        batch_hash = batch_digest(schedule_manifest, zero_based_index)
        record = {
            "state_update_index": completed_step,
            "optimizer_update_index": zero_based_index,
            "active_horizon": horizon,
            "batch_window_sha256": batch_hash,
            "training_loss_total": finite_float(losses["total"]),
            "training_loss_rollout": finite_float(losses["rollout"]),
            "training_loss_hidden": finite_float(losses["hidden"]),
            "training_loss_h1": finite_float(losses["h1"]),
            "raw_gradient_norm": raw_norm,
            "clipped_gradient_norm": clipped_norm,
            "clipping_activated": clipping_activated,
            "parameter_update_norm": update_norm,
            **adamw,
        }
        all_update_rows.append(record)
        update_trace[completed_step] = record
        if completed_step in PROBE_UPDATES:
            model.eval()
            metrics = {str(key): finite_float(value) for key, value in evaluate_compact(model, validation, channels, horizons=(1, 2, 4, 8, 16, 32)).items()}
            model.train()
            probe_values[completed_step] = metrics
            next_batch = batch_digest(schedule_manifest, zero_based_index + 1) if zero_based_index + 1 < TOTAL_UPDATES else None
            probe_rows[completed_step] = probe_row(completed_step, metrics, record, next_batch, "reconstruction" if completed_step <= BRANCH_UPDATE else "control_continuation")


def load_control_from_bundle(bundle: Mapping[str, Any]) -> tuple[Any, torch.optim.Optimizer]:
    model = load_model(R6_CHECKPOINT, trainable=True)
    model.load_state_dict(bundle["model_state_dict"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    optimizer._codex_parameter_names = {id(parameter): name for name, parameter in parameter_rows(model)}
    optimizer.load_state_dict(bundle["optimizer_state_dict"])
    return model, optimizer


def compare_reloaded_bundle(bundle: Mapping[str, Any], branch_model: Mapping[str, Any], branch_optimizer: torch.optim.Optimizer, branch_position: Mapping[str, Any]) -> dict[str, Any]:
    model, optimizer = load_control_from_bundle(bundle)
    model_hash = canonical_model_state_sha256(model.state_dict())
    optimizer_hash = optimizer_semantic_hash(optimizer)
    branch_optimizer_hash = optimizer_semantic_hash(branch_optimizer)
    mapping_match = optimizer_mapping(optimizer) == bundle["optimizer_parameter_mapping"]
    position_match = bundle["training_position"] == branch_position
    direct_model_equal = state_equal(model.state_dict(), branch_model)
    direct_optimizer_equal = optimizer_hash == branch_optimizer_hash and optimizer_step_map(optimizer) == optimizer_step_map(branch_optimizer)
    buffer = io.BytesIO()
    torch.save(bundle, buffer)
    buffer.seek(0)
    roundtrip = torch.load(buffer, map_location="cpu", weights_only=False)
    roundtrip_model, roundtrip_optimizer = load_control_from_bundle(roundtrip)
    semantic_save_load_save = (
        canonical_model_state_sha256(roundtrip_model.state_dict()) == model_hash
        and optimizer_semantic_hash(roundtrip_optimizer) == optimizer_hash
    )
    return {
        "model_state_equal": direct_model_equal,
        "optimizer_state_equal": direct_optimizer_equal,
        "model_semantic_hash": model_hash,
        "optimizer_semantic_hash": optimizer_hash,
        "optimizer_parameter_mapping_equal": mapping_match,
        "training_position_equal": position_match,
        "save_load_save_semantic_replay": semantic_save_load_save,
        "BRANCH_BUNDLE_RELOAD_VALID": "YES" if all((direct_model_equal, direct_optimizer_equal, mapping_match, position_match, semantic_save_load_save)) else "NO",
    }


def compare_authoritative_trajectory(trace: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    if not D4_H1_TRAJECTORY.is_file():
        raise StageBlocker("authoritative_d4_trajectory_missing")
    with D4_H1_TRAJECTORY.open("r", encoding="utf-8", newline="") as stream:
        rows = {int(row["state_update_index"]): row for row in csv.DictReader(stream)}
    checks = (("training_loss_total", "training_loss_total"), ("raw_gradient_norm", "pre_clip_global_grad_norm"), ("clipped_gradient_norm", "post_clip_global_grad_norm"), ("parameter_update_norm", "parameter_update_norm"), ("adamw_step", "adamw_step"), ("exp_avg_norm", "exp_avg_norm"), ("exp_avg_sq_norm", "exp_avg_sq_norm"))
    first = None
    comparisons = 0
    for step in sorted(trace):
        if step not in rows:
            first = {"update": step, "reason": "authoritative_update_missing"}
            break
        for observed_key, authority_key in checks:
            observed = float(trace[step][observed_key])
            expected = float(rows[step][authority_key])
            comparisons += 1
            if not math.isclose(observed, expected, rel_tol=REPLAY_RTOL, abs_tol=REPLAY_ATOL):
                first = {"update": step, "field": observed_key, "observed": observed, "authoritative": expected}
                break
        if first is not None:
            break
    return {"compared_fields": comparisons, "first_measurable_replay_divergence": first, "all_available_update_diagnostics_match": "YES" if first is None else "NO", "authority_trajectory_sha256": sha256_file(D4_H1_TRAJECTORY)}


def quantile_summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {str(q): None for q in (0.0, 0.5, 0.9, 0.95, 0.99, 1.0)}  # type: ignore[return-value]
    return {str(q): finite_float(np.quantile(np.asarray(values, dtype=np.float64), q)) for q in (0.0, 0.5, 0.9, 0.95, 0.99, 1.0)}


def mark_readonly(path: Path) -> None:
    if os.name == "nt":
        subprocess.run(["attrib", "+R", str(path)], cwd=ROOT, capture_output=True, check=False)
    else:
        path.chmod(path.stat().st_mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "update", "H1", "H4", "H16", "H32", "training_loss", "raw_gradient_norm_before_clipping",
        "clipped_gradient_norm", "clipping_activated", "actual_parameter_update_norm", "optimizer_step",
        "exp_avg_norm", "exp_avg_sq_norm", "nonfinite_detected", "last_batch_window_sha256",
        "next_batch_window_sha256", "segment",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_certificate(context: Mapping[str, Any], artifact_hashes: Mapping[str, str], integrity_hash: str | None) -> dict[str, Any]:
    control = context["control"]
    reload_result = context["reload"]
    status = context["status"]
    return {
        "schema_version": "stage3_h13_post_d5_d6_branch_state_materialization_replay_certificate_v1",
        "STAGE_3_H13_POST_D5_D6_BRANCH_STATE_MATERIALIZATION_REPLAY": status,
        "FIRST_BLOCKER": context["first_blocker"],
        "ONE_SENTENCE_VERDICT": context["verdict"],
        "PROJECT_OWNER_AUTHORIZATION": "YES",
        "R8E_A1_IMMUTABLE": "YES",
        "FROZEN20_OPENED": "NO",
        "FROZEN20_USED": "NO",
        "FUTURE_LABEL_LEAKAGE": 0,
        "SOURCE_REVISION_MATCH": "YES" if context["authority"]["current_environment"]["source_revision_match"] else "NO",
        "CANONICAL_SCHEDULE_HASH_MATCH": "YES",
        "D4_RECIPE_EXACTLY_RECONSTRUCTED": context["recipe"]["exact_reconstruction"],
        "BRANCH_UPDATE": BRANCH_UPDATE,
        "UPDATE_BOUNDARY_UNAMBIGUOUS": "YES",
        "COMPLETED_OPTIMIZER_STEPS": BRANCH_UPDATE,
        "MODEL_STATE_CAPTURED": "YES" if context.get("branch") else "NO",
        "OPTIMIZER_STATE_CAPTURED": "YES" if context.get("branch") else "NO",
        "STEP_CAPTURED": "YES" if context.get("branch") else "NO",
        "EXP_AVG_CAPTURED": "YES" if context.get("branch") else "NO",
        "EXP_AVG_SQ_CAPTURED": "YES" if context.get("branch") else "NO",
        "PARAMETER_GROUPS_CAPTURED": "YES" if context.get("branch") else "NO",
        "SCHEDULER_STATE": "ABSENT_BY_DESIGN",
        "RNG_STATE_OR_NO_RNG_PROOF_VALID": context["rng"]["valid"],
        "RNG_REQUIRED_FOR_REMAINING_CONTINUATION": context["rng"]["required"],
        "BRANCH_BUNDLE_SHA256": artifact_hashes.get("post_update_384_branch_bundle.pt", "NONE"),
        "BRANCH_BUNDLE_RELOAD_VALID": reload_result.get("BRANCH_BUNDLE_RELOAD_VALID", "NO"),
        "MODEL_STATE_SEMANTIC_HASH_MATCH": "YES" if reload_result.get("model_semantic_hash") == context.get("branch", {}).get("model_semantic_hash") else "NO",
        "OPTIMIZER_STATE_SEMANTIC_HASH_MATCH": "YES" if reload_result.get("optimizer_semantic_hash") == context.get("branch", {}).get("optimizer_semantic_hash") else "NO",
        "PARAMETER_MAPPING_VALID": "YES" if reload_result.get("optimizer_parameter_mapping_equal") else "NO",
        "REMAINING_128_BATCH_SEQUENCE_EXACT": "YES",
        "CONTROL_CONTINUATION_EXECUTED": "YES" if control.get("executed") else "NO",
        "CONTROL_STARTED_FROM_CERTIFIED_BRANCH_BUNDLE": "YES" if control.get("started_from_bundle") else "NO",
        "CONTROL_OPTIMIZER_STATE_UNMODIFIED": "YES" if control.get("optimizer_intervention") == "NONE" else "NO",
        "CONTROL_CLIPPING_MAX_NORM": CLIP_MAX_NORM,
        "CONTROL_FINAL_H1": control.get("final_metrics", {}).get("1"),
        "CONTROL_FINAL_H4": control.get("final_metrics", {}).get("4"),
        "CONTROL_FINAL_H16": control.get("final_metrics", {}).get("16"),
        "CONTROL_FINAL_H32": control.get("final_metrics", {}).get("32"),
        "CONTROL_H1_STATUS": control.get("h1_status", "NOT_REACHED"),
        "CONTROL_H32_STATUS": control.get("h32_status", "NOT_REACHED"),
        "AUTHORITATIVE_D4_TERMINAL_H1": context["d4_terminal_metrics"].get("1"),
        "AUTHORITATIVE_D4_TERMINAL_H32": context["d4_terminal_metrics"].get("32"),
        "CONTROL_VS_D4_H1_ABSOLUTE_DELTA": control.get("vs_d4", {}).get("1"),
        "CONTROL_VS_D4_H32_ABSOLUTE_DELTA": control.get("vs_d4", {}).get("32"),
        "CONTROL_VS_D4_CLASSIFICATION_MATCH": control.get("classification_match", "NO"),
        "STRONGEST_REPLAY_IDENTITY_LEVEL": context["replay_identity"],
        "CONTROL_REPLAY_QUALIFIED": control.get("qualified", "NO"),
        "DIAGNOSTICS": context["diagnostics"],
        "NEW_PERSISTENT_STORAGE_BYTES": context.get("storage_bytes", 0),
        "FULL_CHECKPOINT_SERIES_CREATED": "NO",
        "ARTIFACT_SHA256": dict(artifact_hashes),
        "FINAL_MANIFEST_SHA256": integrity_hash,
        "ADAMW_MOMENT_RESET_EXECUTED": "NO",
        "D6_COUNTERFACTUAL_EXECUTED": "NO",
        "D6_EXECUTED": "NO",
        "D6_BRANCH_STATE_STATUS": context["d6_status"],
        "READY_FOR_D6_OWNER_AUTHORIZATION": "YES" if context["d6_status"] == "D6_BRANCH_STATE_QUALIFIED" else "NO",
        "CURRENT_VALIDATION_CHAMPION": "R8E_A1",
        "CHAMPION_CHANGED": "NO",
        "READY_FOR_R9": "NO",
        "AUTHORIZED_FOR_R9": "NO",
        "FROZEN20_STATUS": "SEALED",
        "SCIENTIFIC_CONCLUSIONS": context["conclusions"],
    }


def render_report(certificate: Mapping[str, Any]) -> str:
    lines = [
        f"STAGE_3_H13_POST_D5_D6_BRANCH_STATE_MATERIALIZATION_REPLAY:\n{certificate['STAGE_3_H13_POST_D5_D6_BRANCH_STATE_MATERIALIZATION_REPLAY']}",
        f"\nFIRST_BLOCKER:\n{certificate['FIRST_BLOCKER']}",
        f"\nONE_SENTENCE_VERDICT:\n{certificate['ONE_SENTENCE_VERDICT']}",
        "\n## 1. PROTOCOL INTEGRITY",
        "",
        *[f"{key}: {certificate[key]}" for key in ("PROJECT_OWNER_AUTHORIZATION", "R8E_A1_IMMUTABLE", "FROZEN20_OPENED", "FROZEN20_USED", "FUTURE_LABEL_LEAKAGE", "SOURCE_REVISION_MATCH", "CANONICAL_SCHEDULE_HASH_MATCH", "D4_RECIPE_EXACTLY_RECONSTRUCTED")],
        "\n## 2. UPDATE-384 MATERIALIZATION",
        "",
        *[f"{key}: {certificate[key]}" for key in ("BRANCH_UPDATE", "UPDATE_BOUNDARY_UNAMBIGUOUS", "MODEL_STATE_CAPTURED", "OPTIMIZER_STATE_CAPTURED", "STEP_CAPTURED", "EXP_AVG_CAPTURED", "EXP_AVG_SQ_CAPTURED", "PARAMETER_GROUPS_CAPTURED", "SCHEDULER_STATE", "RNG_STATE_OR_NO_RNG_PROOF_VALID", "BRANCH_BUNDLE_SHA256")],
        "\n## 3. BRANCH-BUNDLE IDENTITY",
        "",
        *[f"{key}: {certificate[key]}" for key in ("BRANCH_BUNDLE_RELOAD_VALID", "MODEL_STATE_SEMANTIC_HASH_MATCH", "OPTIMIZER_STATE_SEMANTIC_HASH_MATCH", "PARAMETER_MAPPING_VALID", "REMAINING_128_BATCH_SEQUENCE_EXACT")],
        "\n## 4. UNTOUCHED CONTROL REPLAY",
        "",
        *[f"{key}: {certificate[key]}" for key in ("CONTROL_CONTINUATION_EXECUTED", "CONTROL_STARTED_FROM_CERTIFIED_BRANCH_BUNDLE", "CONTROL_OPTIMIZER_STATE_UNMODIFIED", "CONTROL_CLIPPING_MAX_NORM", "CONTROL_FINAL_H1", "CONTROL_FINAL_H4", "CONTROL_FINAL_H16", "CONTROL_FINAL_H32", "CONTROL_H1_STATUS", "CONTROL_H32_STATUS")],
        "\n## 5. D4 REPLAY COMPARABILITY",
        "",
        *[f"{key}: {certificate[key]}" for key in ("AUTHORITATIVE_D4_TERMINAL_H1", "AUTHORITATIVE_D4_TERMINAL_H32", "CONTROL_VS_D4_H1_ABSOLUTE_DELTA", "CONTROL_VS_D4_H32_ABSOLUTE_DELTA", "CONTROL_VS_D4_CLASSIFICATION_MATCH", "STRONGEST_REPLAY_IDENTITY_LEVEL", "CONTROL_REPLAY_QUALIFIED")],
        "\n## 6. DIAGNOSTICS",
        "",
        json.dumps(certificate["DIAGNOSTICS"], indent=2, sort_keys=True),
        "\n## 7. STORAGE / ARTIFACTS",
        "",
        f"NEW_PERSISTENT_STORAGE_BYTES: {certificate['NEW_PERSISTENT_STORAGE_BYTES']}",
        "FULL_CHECKPOINT_SERIES_CREATED: NO",
        json.dumps(certificate["ARTIFACT_SHA256"], indent=2, sort_keys=True),
        "\n## 8. D6 AUTHORIZATION STATUS",
        "",
        *[f"{key}: {certificate[key]}" for key in ("ADAMW_MOMENT_RESET_EXECUTED", "D6_COUNTERFACTUAL_EXECUTED", "D6_EXECUTED", "D6_BRANCH_STATE_STATUS", "READY_FOR_D6_OWNER_AUTHORIZATION")],
        "\n## 9. CHAMPION / R9 / FROZEN20",
        "",
        *[f"{key}: {certificate[key]}" for key in ("CURRENT_VALIDATION_CHAMPION", "CHAMPION_CHANGED", "READY_FOR_R9", "AUTHORIZED_FOR_R9", "FROZEN20_STATUS")],
        "\n## 10. FINAL SCIENTIFIC CONCLUSION",
        "",
        *[f"{index}. {item}" for index, item in enumerate(certificate["SCIENTIFIC_CONCLUSIONS"], start=1)],
        "",
    ]
    return "\n".join(lines)


def write_integrity_manifest(output: Path) -> str:
    rows = []
    for path in sorted(item for item in output.rglob("*") if item.is_file() and item.name != "integrity_manifest.json"):
        rows.append({"path": str(path.relative_to(output)).replace("\\", "/"), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    write_json(output / "integrity_manifest.json", {"schema_version": "stage3_h13_post_d5_d6_integrity_manifest_v1", "hash_algorithm": "SHA-256", "self_hash_excluded": True, "files": rows})
    return sha256_file(output / "integrity_manifest.json")


def artifact_hashes(output: Path, names: Sequence[str]) -> dict[str, str]:
    return {name: sha256_file(output / name) for name in names if (output / name).is_file()}


def fail_closed(output: Path, blocker: str) -> int:
    output.mkdir(parents=True, exist_ok=True)
    certificate = {
        "schema_version": "stage3_h13_post_d5_d6_branch_state_materialization_replay_certificate_v1",
        "STAGE_3_H13_POST_D5_D6_BRANCH_STATE_MATERIALIZATION_REPLAY": "BLOCKED",
        "FIRST_BLOCKER": blocker,
        "ONE_SENTENCE_VERDICT": "The Stage 3 branch-state qualification failed closed before a valid D6 authorization gate was established.",
        "PROJECT_OWNER_AUTHORIZATION": "YES",
        "R8E_A1_IMMUTABLE": "YES",
        "FROZEN20_OPENED": "NO",
        "FROZEN20_USED": "NO",
        "FUTURE_LABEL_LEAKAGE": 0,
        "SOURCE_REVISION_MATCH": "UNKNOWN",
        "CANONICAL_SCHEDULE_HASH_MATCH": "UNKNOWN",
        "D4_RECIPE_EXACTLY_RECONSTRUCTED": "NO",
        "BRANCH_UPDATE": 384,
        "UPDATE_BOUNDARY_UNAMBIGUOUS": "NO",
        "MODEL_STATE_CAPTURED": "NO",
        "OPTIMIZER_STATE_CAPTURED": "NO",
        "STEP_CAPTURED": "NO",
        "EXP_AVG_CAPTURED": "NO",
        "EXP_AVG_SQ_CAPTURED": "NO",
        "PARAMETER_GROUPS_CAPTURED": "NO",
        "SCHEDULER_STATE": "INVALID",
        "RNG_STATE_OR_NO_RNG_PROOF_VALID": "NO",
        "BRANCH_BUNDLE_SHA256": "NONE",
        "BRANCH_BUNDLE_RELOAD_VALID": "NO",
        "MODEL_STATE_SEMANTIC_HASH_MATCH": "NO",
        "OPTIMIZER_STATE_SEMANTIC_HASH_MATCH": "NO",
        "PARAMETER_MAPPING_VALID": "NO",
        "REMAINING_128_BATCH_SEQUENCE_EXACT": "NO",
        "CONTROL_CONTINUATION_EXECUTED": "NO",
        "CONTROL_STARTED_FROM_CERTIFIED_BRANCH_BUNDLE": "NO",
        "CONTROL_OPTIMIZER_STATE_UNMODIFIED": "YES",
        "CONTROL_CLIPPING_MAX_NORM": 1.0,
        "CONTROL_FINAL_H1": None,
        "CONTROL_FINAL_H4": None,
        "CONTROL_FINAL_H16": None,
        "CONTROL_FINAL_H32": None,
        "CONTROL_H1_STATUS": "NOT_REACHED",
        "CONTROL_H32_STATUS": "NOT_REACHED",
        "AUTHORITATIVE_D4_TERMINAL_H1": None,
        "AUTHORITATIVE_D4_TERMINAL_H32": None,
        "CONTROL_VS_D4_H1_ABSOLUTE_DELTA": None,
        "CONTROL_VS_D4_H32_ABSOLUTE_DELTA": None,
        "CONTROL_VS_D4_CLASSIFICATION_MATCH": "NO",
        "STRONGEST_REPLAY_IDENTITY_LEVEL": "NONE",
        "CONTROL_REPLAY_QUALIFIED": "NO",
        "DIAGNOSTICS": {"first_measurable_replay_divergence": blocker},
        "NEW_PERSISTENT_STORAGE_BYTES": 0,
        "FULL_CHECKPOINT_SERIES_CREATED": "NO",
        "ARTIFACT_SHA256": {},
        "FINAL_MANIFEST_SHA256": None,
        "ADAMW_MOMENT_RESET_EXECUTED": "NO",
        "D6_COUNTERFACTUAL_EXECUTED": "NO",
        "D6_EXECUTED": "NO",
        "D6_BRANCH_STATE_STATUS": "D6_BRANCH_STATE_RECONSTRUCTION_INCONCLUSIVE",
        "READY_FOR_D6_OWNER_AUTHORIZATION": "NO",
        "CURRENT_VALIDATION_CHAMPION": "R8E_A1",
        "CHAMPION_CHANGED": "NO",
        "READY_FOR_R9": "NO",
        "AUTHORIZED_FOR_R9": "NO",
        "FROZEN20_STATUS": "SEALED",
        "SCIENTIFIC_CONCLUSIONS": [
            "The exact update-384 branch state was not certified.",
            "The model-plus-complete-AdamW bundle is not certified as recoverable.",
            "The branch bundle reload/identity gate was not passed.",
            "The untouched control continuation was not authorized to run.",
            "No replay identity level was established.",
            "The state is not scientifically valid for a future D6 branch.",
            "No moment reset was executed.",
            "R8E_A1 remained the champion.",
            "Frozen-20 remains sealed and may not be opened.",
            "Single next action: resolve the exact blocker and rerun this stage only with unchanged authority inputs.",
        ],
    }
    write_json(output / "terminal_certificate.json", certificate)
    (output / "FINAL_REPORT.md").write_text(render_report(certificate), encoding="utf-8", newline="\n")
    write_integrity_manifest(output)
    return 1


def execute(output: Path) -> int:
    if output.exists():
        return fail_closed(output, f"output_directory_already_exists_no_resume:{output}")
    output.mkdir(parents=True, exist_ok=False)
    context: dict[str, Any] = {"status": "BLOCKED", "first_blocker": "unknown", "branch": {}, "control": {}, "reload": {}, "rng": {"valid": "NO", "required": "UNKNOWN"}}
    try:
        pre = preflight(True)
        authority = verify_authority(pre)
        context["authority"] = authority
        if not AUTHORIZATION_PATH.is_file():
            raise StageBlocker("current_stage_authorization_source_missing")
        context["authorization"] = {"source": str(AUTHORIZATION_PATH.resolve()), "sha256": sha256_file(AUTHORIZATION_PATH), "scope": EXPERIMENT_ID}
        config = {
            "experiment_id": EXPERIMENT_ID,
            "seed": SEED,
            "initialization": "R6 authoritative model state",
            "architecture": "2-layer unidirectional residual GRU, hidden 128, decoder horizon 8",
            "batch_size": BATCH_SIZE,
            "total_updates": TOTAL_UPDATES,
            "branch_update": BRANCH_UPDATE,
            "remaining_updates": REMAINING_UPDATES,
            "phase_horizons": list(PHASE_HORIZONS),
            "updates_per_phase": 128,
            "learning_rate": LEARNING_RATE,
            "betas": list(BETAS),
            "eps": EPS,
            "weight_decay": WEIGHT_DECAY,
            "amsgrad": False,
            "foreach": None,
            "fused": None,
            "capturable": False,
            "differentiable": False,
            "gradient_clipping": {"enabled": True, "operation": "torch.nn.utils.clip_grad_norm_", "max_norm": CLIP_MAX_NORM},
            "loss": "causal_paired_rollout + weighted Smooth-L1 position loss + hidden consistency loss",
            "lambda_h1": LAMBDA_H1,
            "lambda_hidden": LAMBDA_HIDDEN,
            "huber_beta": HUBER_BETA,
            "scheduler": "ABSENT_BY_DESIGN",
            "data_preprocessing": "H11 TRAIN-only normalization; H10 TRAIN windows; direct tensor batches",
            "evaluator": "H10 VALIDATION free-running causal self-feeding rollout; horizons 1,2,4,8,16,32",
            "device": "cpu",
            "dtype": "float32",
            "future_label_leakage": 0,
            "source_revision": EXPECTED_SOURCE_COMMIT,
            "canonical_schedule_sha256": EXPECTED_CANONICAL_SCHEDULE,
            "canonical_schedule_storage_sha256": EXPECTED_CANONICAL_STORAGE,
        }
        set_deterministic(SEED)
        model = load_model(R6_CHECKPOINT, trainable=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        optimizer._codex_parameter_names = {id(parameter): name for name, parameter in parameter_rows(model)}
        actual_optimizer = optimizer_contract(optimizer)
        expected_optimizer = {"class": "AdamW", "learning_rate": LEARNING_RATE, "betas": list(BETAS), "eps": EPS, "weight_decay": WEIGHT_DECAY, "amsgrad": False, "foreach": None, "fused": None, "capturable": False, "differentiable": False, "maximize": False, "decoupled_weight_decay": True}
        if actual_optimizer != expected_optimizer:
            raise StageBlocker(f"optimizer_contract_mismatch:{actual_optimizer}")
        context["recipe"] = {"config": config, "optimizer_contract_observed": actual_optimizer, "exact_reconstruction": "YES"}
        starting_model_hash = canonical_model_state_sha256(model.state_dict())
        if starting_model_hash != canonical_model_state_sha256(load_model(R6_CHECKPOINT).state_dict()):
            raise StageBlocker("starting_model_state_load_mismatch")
        if sha256_file(R6_CHECKPOINT) != EXPECTED_R6_HASH:
            raise StageBlocker("starting_checkpoint_hash_changed")
        schedule_rows = authority["schedule_rows"]
        schedule_manifest = authority["schedule_manifest"]
        branch_metrics: dict[int, dict[str, float]] = {}
        branch_rows: dict[int, dict[str, Any]] = {}
        update_trace: dict[int, dict[str, Any]] = {}
        all_update_rows: list[dict[str, Any]] = []
        model.eval()
        initial_metrics = {str(key): finite_float(value) for key, value in evaluate_compact(model, pre["validation"], pre["stats"]["channels"], horizons=(1, 2, 4, 8, 16, 32)).items()}
        model.train()
        branch_metrics[0] = initial_metrics
        branch_rows[0] = probe_row(0, initial_metrics, None, batch_digest(schedule_manifest, 0), "initial")
        run_segment(model, optimizer, pre["train_data"], pre["validation"], pre["stats"]["channels"], schedule_rows, schedule_manifest, 0, BRANCH_UPDATE, branch_metrics, branch_rows, update_trace, all_update_rows)
        if len(all_update_rows) != BRANCH_UPDATE or max((row["state_update_index"] for row in all_update_rows), default=0) != BRANCH_UPDATE:
            raise StageBlocker("update_384_boundary_not_reached_exactly")
        if any(step != BRANCH_UPDATE for step in optimizer_step_map(optimizer).values()):
            raise StageBlocker("optimizer_step_not_384_at_branch")
        branch_rng = capture_rng()
        branch_model_state = copy.deepcopy(model.state_dict())
        branch_optimizer_state = copy.deepcopy(optimizer.state_dict())
        branch_mapping = optimizer_mapping(optimizer)
        branch_position = {"completed_optimizer_steps": BRANCH_UPDATE, "next_completed_optimizer_step": BRANCH_UPDATE + 1, "next_zero_based_schedule_index": BRANCH_UPDATE, "next_batch_window_sha256": batch_digest(schedule_manifest, BRANCH_UPDATE), "remaining_updates": REMAINING_UPDATES, "remaining_schedule_sha256": authority["schedule_slice_hashes"]["suffix_128"]}
        branch_hashes = {
            "model_semantic_hash": canonical_model_state_sha256(branch_model_state),
            "optimizer_semantic_hash": optimizer_semantic_hash(optimizer),
            "parameter_groups_semantic_hash": sha256_bytes(canonical_json(optimizer_canonical_payload(optimizer)["param_groups"])),
            "optimizer_step_semantic_hash": sha256_bytes(canonical_json(optimizer_step_map(optimizer))),
            "exp_avg_semantic_hash": named_optimizer_component_hash(optimizer, "exp_avg"),
            "exp_avg_sq_semantic_hash": named_optimizer_component_hash(optimizer, "exp_avg_sq"),
            "rng_state_sha256": rng_digest(branch_rng),
        }
        bundle = {
            "schema_version": "stage3_h13_post_d5_d6_post_update_384_branch_bundle_v1",
            "experiment_id": EXPERIMENT_ID,
            "model_state_dict": branch_model_state,
            "optimizer_state_dict": branch_optimizer_state,
            "optimizer_parameter_mapping": branch_mapping,
            "optimizer_contract": actual_optimizer,
            "state_hashes": branch_hashes,
            "training_config": config,
            "evaluator_config": {"split": "H10 VALIDATION", "horizons": [1, 2, 4, 8, 16, 32], "free_running": True, "future_label_leakage": 0},
            "training_position": branch_position,
            "rng_state": branch_rng,
            "provenance": {"source_revision": EXPECTED_SOURCE_COMMIT, "starting_checkpoint_sha256": EXPECTED_R6_HASH, "canonical_schedule_sha256": EXPECTED_CANONICAL_SCHEDULE, "canonical_schedule_prefix_384_sha256": authority["schedule_slice_hashes"]["prefix_384"], "canonical_schedule_suffix_128_sha256": authority["schedule_slice_hashes"]["suffix_128"], "captured_at_utc": utc_now(), "environment": authority["current_environment"]},
        }
        bundle_path = output / "post_update_384_branch_bundle.pt"
        torch.save(bundle, bundle_path)
        mark_readonly(bundle_path)
        context["branch"] = {**branch_hashes, "path": str(bundle_path.resolve()), "mapping": branch_mapping, "position": branch_position}
        write_json(output / "training_config.json", config)
        write_json(output / "authority_and_provenance.json", {"experiment_id": EXPERIMENT_ID, "authorization": context["authorization"], "authoritative_paths": authority["authority_snapshot"], "d4_terminal_reference": authority["d4_seed_record"]["metrics"], "d5_certificate": authority["d5_certificate"], "d6_design_certificate": authority["d2_certificate"], "environment": authority["current_environment"], "starting_model_semantic_hash": starting_model_hash, "starting_checkpoint_raw_sha256": EXPECTED_R6_HASH, "canonical_schedule": authority["schedule_identity"], "schedule_slice_hashes": authority["schedule_slice_hashes"], "frozen20": {"opened": "NO", "used": "NO", "status": "SEALED"}})
        write_json(output / "branch_state_hashes.json", {**branch_hashes, "branch_bundle_raw_sha256": sha256_file(bundle_path), "parameter_mapping": branch_mapping, "optimizer_contract": actual_optimizer, "training_position": branch_position})
        reload_result = compare_reloaded_bundle(bundle, branch_model_state, optimizer, branch_position)
        context["reload"] = reload_result
        if reload_result["BRANCH_BUNDLE_RELOAD_VALID"] != "YES":
            raise StageBlocker("branch_bundle_reload_identity_failure")
        write_json(output / "branch_bundle_reload_validation.json", reload_result)
        restore_rng(branch_rng)
        control_model, control_optimizer = load_control_from_bundle(bundle)
        control_start_hash = optimizer_semantic_hash(control_optimizer)
        if control_start_hash != branch_hashes["optimizer_semantic_hash"]:
            raise StageBlocker("control_did_not_start_from_certified_optimizer_state")
        control_rng_before = capture_rng()
        run_segment(control_model, control_optimizer, pre["train_data"], pre["validation"], pre["stats"]["channels"], schedule_rows, schedule_manifest, BRANCH_UPDATE, TOTAL_UPDATES, branch_metrics, branch_rows, update_trace, all_update_rows)
        control_rng_after = capture_rng()
        if len(all_update_rows) != TOTAL_UPDATES or any(step != TOTAL_UPDATES for step in optimizer_step_map(control_optimizer).values()):
            raise StageBlocker("control_completed_update_count_or_step_mismatch")
        control_model.eval()
        final_metrics = {str(key): finite_float(value) for key, value in evaluate_compact(control_model, pre["validation"], pre["stats"]["channels"], horizons=(1, 2, 4, 8, 16, 32)).items()}
        final_model_hash = canonical_model_state_sha256(control_model.state_dict())
        trajectory_match = compare_authoritative_trajectory(update_trace)
        d4_metrics = {str(key): float(value) for key, value in authority["d4_seed_record"]["metrics"].items()}
        classifications = {"H1": "PASS" if final_metrics["1"] <= D4_H1_THRESHOLD else "FAIL", "H32": "PASS" if final_metrics["32"] <= H32_THRESHOLD else "FAIL"}
        d4_classifications = {"H1": "PASS" if d4_metrics["1"] <= D4_H1_THRESHOLD else "FAIL", "H32": "PASS" if d4_metrics["32"] <= H32_THRESHOLD else "FAIL"}
        control = {
            "executed": True,
            "started_from_bundle": True,
            "optimizer_intervention": "NONE",
            "optimizer_start_semantic_hash": control_start_hash,
            "optimizer_final_semantic_hash": optimizer_semantic_hash(control_optimizer),
            "final_model_semantic_hash": final_model_hash,
            "final_metrics": final_metrics,
            "vs_d4": {key: abs(final_metrics[key] - d4_metrics[key]) for key in ("1", "4", "16", "32")},
            "h1_status": classifications["H1"],
            "h32_status": classifications["H32"],
            "classification_match": "YES" if classifications == d4_classifications else "NO",
            "qualified": "YES" if final_model_hash == EXPECTED_FINAL_MODEL_SEMANTIC and trajectory_match["all_available_update_diagnostics_match"] == "YES" and classifications == d4_classifications else "NO",
            "rng_before_digest": rng_digest(control_rng_before),
            "rng_after_digest": rng_digest(control_rng_after),
        }
        context["control"] = control
        context["trajectory_match"] = trajectory_match
        if control["qualified"] != "YES":
            raise StageBlocker("control_replay_did_not_reproduce_authoritative_d4")
        context["rng"] = {"valid": "YES" if control["rng_before_digest"] == control["rng_after_digest"] else "NO", "required": "NO" if control["rng_before_digest"] == control["rng_after_digest"] else "UNKNOWN", "branch_digest": branch_hashes["rng_state_sha256"], "control_before_digest": control["rng_before_digest"], "control_after_digest": control["rng_after_digest"]}
        if context["rng"]["valid"] != "YES":
            raise StageBlocker("unexpected_rng_consumer_during_remaining_continuation")
        terminal_payload = {"schema_version": "stage3_h13_post_d5_d6_control_terminal_state_v1", "completed_optimizer_steps": TOTAL_UPDATES, "model_state_dict": copy.deepcopy(control_model.state_dict()), "model_semantic_hash": final_model_hash, "final_metrics": final_metrics, "optimizer_state_unmodified_intervention": "NONE", "source_branch_bundle_sha256": sha256_file(bundle_path)}
        terminal_path = output / "untouched_control_terminal_state.pt"
        torch.save(terminal_payload, terminal_path)
        mark_readonly(terminal_path)
        write_csv(output / "trajectory_diagnostics.csv", [branch_rows[step] for step in PROBE_UPDATES])
        write_json(output / "replay_qualification.json", {"control": control, "authoritative_d4": {"metrics": d4_metrics, "model_semantic_hash": EXPECTED_FINAL_MODEL_SEMANTIC, "seed_checkpoint_raw_sha256": EXPECTED_D4_SEED_CHECKPOINT_RAW}, "trajectory_comparison": trajectory_match, "strongest_identity_level": "MODEL_STATE_SEMANTIC", "raw_file_identity": "NOT_ESTABLISHED"})
        all_raw = [row["raw_gradient_norm"] for row in all_update_rows]
        all_clipped = [row["clipped_gradient_norm"] for row in all_update_rows]
        all_updates = [row["parameter_update_norm"] for row in all_update_rows]
        all_exp = [row["exp_avg_norm"] for row in all_update_rows]
        all_exp_sq = [row["exp_avg_sq_norm"] for row in all_update_rows]
        context["diagnostics"] = {"gradient_norm_before_clipping": {"quantiles": quantile_summary(all_raw), "clipping_activation_count": sum(row["clipping_activated"] == "YES" for row in all_update_rows), "clipping_activation_fraction": sum(row["clipping_activated"] == "YES" for row in all_update_rows) / TOTAL_UPDATES}, "clipped_gradient_norm": {"quantiles": quantile_summary(all_clipped)}, "actual_parameter_update_norm": {"quantiles": quantile_summary(all_updates)}, "optimizer_step": {"branch": 384, "control_final": 512, "all_parameter_steps_equal_expected": "YES"}, "exp_avg_norm": {"quantiles": quantile_summary(all_exp)}, "exp_avg_sq_norm": {"quantiles": quantile_summary(all_exp_sq)}, "nonfinite_events": 0, "first_measurable_replay_divergence": trajectory_match["first_measurable_replay_divergence"]}
        context["d4_terminal_metrics"] = d4_metrics
        context["status"] = "PASSED"
        context["first_blocker"] = "none"
        context["d6_status"] = "D6_BRANCH_STATE_QUALIFIED"
        context["replay_identity"] = "MODEL_STATE_SEMANTIC"
        context["verdict"] = "The canonical clipped D4 trajectory was replayed through update 384, its complete live AdamW state was reloaded successfully, and an untouched 128-update continuation reproduced the authoritative D4 terminal model state and classification; this earns owner authorization to consider D6 but does not run it."
        context["conclusions"] = [
            "Yes: the exact post-update-384 branch boundary was materialized after completed optimizer step 384 and before update 385.",
            "Yes: the model state and complete live AdamW optimizer state are recoverable from the immutable branch bundle.",
            "Yes: bundle reload, semantic model/optimizer identity, parameter mapping, and save-load-save replay all passed.",
            "Yes: the untouched control reproduced canonical D4 to the predeclared diagnostic and terminal model-state semantic contract.",
            "The strongest established replay identity is MODEL_STATE_SEMANTIC; raw serialized-file identity was not claimed.",
            "Yes: the branch state is scientifically valid for a future single-factor D6 moment-reset branch, subject to later owner authorization.",
            "No AdamW moment reset was executed.",
            "No: R8E_A1 remained the validation champion.",
            "No: Frozen-20 may not be opened in this stage; it remains sealed.",
            "Single next action: obtain explicit owner authorization and run the separately controlled D6 moment-reset experiment from this certified bundle.",
        ]
        core_names = ["post_update_384_branch_bundle.pt", "untouched_control_terminal_state.pt", "trajectory_diagnostics.csv", "training_config.json", "authority_and_provenance.json", "branch_state_hashes.json", "branch_bundle_reload_validation.json", "replay_qualification.json"]
        core_hashes = artifact_hashes(output, core_names)
        context["storage_bytes"] = 0
        certificate = build_certificate(context, core_hashes, None)
        for _ in range(3):
            write_json(output / "terminal_certificate.json", certificate)
            (output / "FINAL_REPORT.md").write_text(render_report(certificate), encoding="utf-8", newline="\n")
            write_integrity_manifest(output)
            measured = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
            if measured == certificate["NEW_PERSISTENT_STORAGE_BYTES"]:
                break
            certificate["NEW_PERSISTENT_STORAGE_BYTES"] = measured
        # The integrity manifest intentionally excludes itself; its final raw
        # SHA-256 is reported from the completed output directory.
        write_json(output / "terminal_certificate.json", certificate)
        (output / "FINAL_REPORT.md").write_text(render_report(certificate), encoding="utf-8", newline="\n")
        write_integrity_manifest(output)
        print(json.dumps({"status": certificate["STAGE_3_H13_POST_D5_D6_BRANCH_STATE_MATERIALIZATION_REPLAY"], "branch_bundle": str(bundle_path), "control_model_hash": final_model_hash, "output": str(output.resolve())}, sort_keys=True), flush=True)
        return 0
    except StageBlocker as exc:
        context["first_blocker"] = str(exc)
        return fail_closed(output, str(exc))
    except Exception as exc:  # pragma: no cover - final fail-closed guard
        context["first_blocker"] = f"unexpected_execution_failure:{type(exc).__name__}:{exc}"
        return fail_closed(output, context["first_blocker"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    return execute(args.output_dir.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
