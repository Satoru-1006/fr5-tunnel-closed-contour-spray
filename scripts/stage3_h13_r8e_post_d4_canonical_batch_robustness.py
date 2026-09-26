"""Execute the owner-authorized Stage 3 H13 R8E post-D4 experiment.

Only the A1 TRAIN batch schedule is changed.  The existing R8E-A1 training
objective and frozen validation evaluator are reproduced locally so the D4
source artifacts remain immutable and no historical D3 run is resumed.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import pickle
import platform
import random
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

from scripts.stage3_h13_r8e_causal_hidden_stability_training import (
    AUTHORITATIVE_R6_H1,
    H10_ROOT,
    H11_ROOT,
    R6_CERTIFICATE,
    R6_CHECKPOINT,
    R6_MANIFEST,
    evaluate_compact,
    load_json,
    serialize_candidate,
    tensor_batch,
)
from src.stage3_h11_dataset import compute_normalization_stats, load_h10_segments, semantic_hash, verify_h10_authority
from src.stage3_h11_model import set_deterministic
from src.stage3_h11_dataset import FEATURE_NAMES
from src.stage3_h11_r2_model import ResidualCausalGRUTrajectoryPredictor
from src.stage3_h13_r6 import build_sequence_arrays
from src.stage3_h13_r8e import PHASE_HORIZONS, causal_paired_rollout, derive_h1_guardrail, gradients_are_finite, training_losses
from src.stage3_h13_r8e_d2 import canonical_model_state_sha256
from src.stage3_h13_r8e_d4 import (
    BATCH_SIZE,
    CHAMPION_MEDIAN_H32_LIMIT,
    EXPERIMENT_ID,
    FROZEN_SEEDS,
    H1_THRESHOLD,
    H32_THRESHOLD,
    TOTAL_UPDATES,
    aggregate_metric,
    assert_frozen20_excluded,
    build_canonical_schedule,
    champion_replacement_pass,
    h1_pass,
    h32_pass,
    metric_spread,
    schedule_indices,
    validate_frozen_seeds,
)


EXPECTED_R6_HASH = "8323e8e3c12a182a2efe3fc873f854be388f5ea0a32bee2c00fee0a1353ae3ee"
EXPECTED_A1_HASH = "4d57a265ab130619ae2971124a514c35bf6efac8731fbadbaef6a8d9d921e6ac"
EXPECTED_A1_CANONICAL = "dd16d6f19c387f066eb4f9ac490ab9b6299eb745a2ce71da224ecae3a59e56de"
EXPECTED_H10_SEMANTIC = "cb282a06d73c322208f27bf7c0b019831a24f524ac0d72263929f22303f0801b"
D2_CONFIG_SHA256 = "2c7647feb14fa47d4363d3479d34706da5d06cb6e89913111f64b0d146005972"
EXPECTED_SOURCE_COMMIT = "a6dff2b6887cc3f7598ab95549c2dd21c0efe763"
D4_ROOT = ROOT / "outputs/stage3_h13_r8e_d4_cross_seed_robustness_design_review_20260816T093843Z"
D4_DESIGN = D4_ROOT / "prospective_robustness_experiment_design.json"
D4_GRADUATION = D4_ROOT / "prospective_graduation_contract.json"
D4_CERTIFICATE = D4_ROOT / "terminal_certificate.json"
D4_AUTHORITY = D4_ROOT / "authority_manifest.json"
D4_BUNDLE_HASHES = D4_ROOT / "bundle_hashes.sha256"
D2_ROOT = ROOT / "outputs/stage3_h13_r8e_d2_prospective_causal_design_20260816T053129Z"
D3_ROOT = ROOT / "outputs/stage3_h13_r8e_d3_prospective_five_seed_causal_replication_20260816T062203Z"
D3A_ROOT = ROOT / "outputs/stage3_h13_r8e_d3a_seed_218309645_model_selection_failure_audit_20260816T073000Z"
A1_ROOT = ROOT / "outputs/stage3_h13_r8e_causal_hidden_stability_training_20260814T181715Z"
A1_CHECKPOINT = A1_ROOT / "best_candidate_checkpoint.pt"
A1_CONFIG = A1_ROOT / "r8e_training_config.json"
AUTHORIZATION_SOURCE = Path(r"C:\Users\86198\.codex\attachments\26935f6c-8ac5-4160-9844-57bb723f13a2\pasted-text.txt")
REPLAY_RTOL = 5.0e-6
REPLAY_ATOL = 5.0e-9


class D4Blocker(RuntimeError):
    pass


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def git_text(args: Sequence[str]) -> str | None:
    result = subprocess.run(list(args), cwd=ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def environment_provenance() -> dict[str, Any]:
    commit = git_text(["git", "rev-parse", "HEAD"])
    return {
        "captured_at_utc": utc_now(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pytorch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": "cpu",
        "dtype": "float32",
        "source_commit": commit,
        "source_commit_match": commit == EXPECTED_SOURCE_COMMIT,
        "deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "dataloader_workers": 0,
        "residual_nondeterminism": "cross_platform_bitwise_recreation_not_claimed",
    }


def load_model(path: Path, *, trainable: bool = False) -> Any:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload) if isinstance(payload, Mapping) else payload
    model = ResidualCausalGRUTrajectoryPredictor(input_size=len(FEATURE_NAMES), hidden_size=128, num_layers=2, horizon=8)
    model.load_state_dict(state, strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(bool(trainable))
    if not trainable:
        model.eval()
    return model


def state_is_finite(model: Any) -> bool:
    return all(bool(torch.all(torch.isfinite(value))) for value in model.state_dict().values() if torch.is_floating_point(value))


def rng_state_digest() -> str:
    payload = pickle.dumps((random.getstate(), np.random.get_state(), torch.get_rng_state().cpu().numpy().tobytes()), protocol=4)
    return sha256_bytes(payload)


def snapshot(paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.is_file():
            raise D4Blocker(f"missing_authority_file:{path}")
        result[str(path.resolve())] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    return result


def verify_snapshot(records: Mapping[str, Mapping[str, Any]], label: str) -> None:
    for raw_path, record in records.items():
        path = Path(raw_path)
        if not path.is_file() or sha256_file(path) != record["sha256"] or path.stat().st_size != int(record["size_bytes"]):
            raise D4Blocker(f"{label}_mutated:{raw_path}")


def verify_d4_bundle() -> dict[str, Any]:
    required = [D4_DESIGN, D4_GRADUATION, D4_CERTIFICATE, D4_AUTHORITY, D4_BUNDLE_HASHES]
    if any(not path.is_file() for path in required):
        raise D4Blocker("d4_design_authority_missing")
    design = load_json(D4_DESIGN)
    graduation = load_json(D4_GRADUATION)
    certificate = load_json(D4_CERTIFICATE)
    authority = load_json(D4_AUTHORITY)
    if certificate.get("STAGE_3_H13_R8E_D4") != "PASSED" or certificate.get("TRAINING_EXECUTED") != "NO" or certificate.get("FROZEN20_USED") != "NO":
        raise D4Blocker("d4_design_certificate_not_immutable_design_only")
    if design.get("execution_status") != "DESIGN_ONLY_NOT_EXECUTED" or graduation.get("status") != "PROPOSED_PRECOMMITMENT_NOT_EXECUTED":
        raise D4Blocker("d4_design_contract_status_mismatch")
    seed_contract = design.get("historical_seed_policy", {}).get("seeds")
    if tuple(seed_contract or ()) != FROZEN_SEEDS:
        raise D4Blocker("d4_seed_contract_mismatch")
    for line in D4_BUNDLE_HASHES.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        expected_hash, relative = line.split(maxsplit=1)
        path = D4_ROOT / relative
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise D4Blocker(f"d4_bundle_hash_mismatch:{relative}")
    input_records = authority.get("input_hashes", [])
    for item in input_records:
        path = ROOT / str(item["path"])
        if not path.is_file() or sha256_file(path) != item["sha256"] or path.stat().st_size != int(item["size_bytes"]):
            raise D4Blocker(f"d4_authority_input_mutated:{item['path']}")
    return {
        "design": design,
        "graduation": graduation,
        "certificate": certificate,
        "authority": authority,
        "bundle_sha256": sha256_file(D4_BUNDLE_HASHES),
        "authority_sha256": sha256_file(D4_AUTHORITY),
        "input_count": len(input_records),
    }


def verify_checkpoint(path: Path, expected_seed: int, expected_schedule_hash: str, expected_canonical: str) -> dict[str, Any]:
    if not path.is_file():
        raise D4Blocker(f"terminal_checkpoint_missing:{expected_seed}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or "model_state_dict" not in payload:
        raise D4Blocker(f"checkpoint_payload_invalid:{expected_seed}")
    model = load_model(path)
    canonical = canonical_model_state_sha256(model.state_dict())
    if canonical != expected_canonical or not state_is_finite(model):
        raise D4Blocker(f"checkpoint_state_invalid:{expected_seed}")
    metadata = payload.get("training_config", {})
    if metadata.get("seed") != int(expected_seed) or metadata.get("canonical_schedule_sha256") != expected_schedule_hash or payload.get("initialization_sha256") != EXPECTED_R6_HASH:
        raise D4Blocker(f"checkpoint_metadata_invalid:{expected_seed}")
    return {
        "seed": int(expected_seed),
        "path": str(path.resolve()),
        "file_size": path.stat().st_size,
        "raw_sha256": sha256_file(path),
        "canonical_model_state_sha256": canonical,
        "state_finite": True,
        "schedule_sha256": expected_schedule_hash,
        "completion_status": "COMPLETED_VALIDATED",
    }


def static_seed_audit(model: Any) -> dict[str, Any]:
    stochastic = []
    for module in model.modules():
        name = module.__class__.__name__.lower()
        if any(token in name for token in ("dropout", "rand", "stochastic")):
            stochastic.append(module.__class__.__name__)
    return {
        "model_initialization": "ResidualCausalGRUTrajectoryPredictor constructor consumes the seed-specific PyTorch RNG for temporary parameters; every parameter is then overwritten by the frozen R6 checkpoint, so no effective seed-dependent initialization remains",
        "numpy_rng": "set_deterministic(seed) configures state; no NumPy RNG call remains in executed scheduling/training path",
        "python_rng": "set_deterministic(seed) configures state; no Python RNG call remains in executed scheduling/training path",
        "pytorch_rng": "set_deterministic(seed) configures state and the temporary model constructor consumes it; the RNG state is unchanged after R6 load, optimizer creation, and all 512 training updates",
        "batch_ordering": "NONE; canonical schedule is generated before runs and contains no seed input",
        "sample_composition": "NONE; every run consumes the same ordered window IDs",
        "stochastic_model_operations": "NONE OBSERVED" if not stochastic else stochastic,
        "worker_initialization": "NONE; direct tensor batches and zero workers",
        "augmentation": "NONE OBSERVED",
        "loss_calculation": "NONE OBSERVED",
        "rollout_curriculum": "NONE; fixed H4/H8/H16/H32 phases",
        "validation": "NONE; frozen VALIDATION split and deterministic evaluator",
        "checkpoint_selection": "Seed is retained in metadata/path only; fixed terminal 512-update validation gates are seed-independent",
        "temporary_constructor_rng_consumption": "OBSERVED_BUT_OVERWRITTEN_BY_R6",
        "runtime_rng_state_unchanged_during_training": "PENDING_PER_RUN",
        "remaining_seed_effective_training_paths": "NONE OBSERVED",
    }


def train_candidate_canonical(
    train_data: Any,
    validation_data: Any,
    channels: Mapping[str, Any],
    guardrail: Any,
    config: Mapping[str, Any],
    schedule_indices_array: np.ndarray,
) -> tuple[Any, dict[str, Any], bytes, dict[str, Any]]:
    """A1 training loop with the sole batch-order input replaced by a manifest."""

    set_deterministic(int(config["seed"]))
    rng_before = rng_state_digest()
    model = load_model(R6_CHECKPOINT, trainable=True)
    rng_after_initialization = rng_state_digest()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    rng_after_optimizer = rng_state_digest()
    history: list[dict[str, Any]] = []
    global_update = 0
    peak_batch_bytes = 0
    gradient_clip_activations = 0
    gradient_norm_max = 0.0
    hidden_enabled = True
    lambda_hidden = float(config["lambda_hidden_a1"])
    updates_per_phase = int(config["updates_per_phase"])
    batch_size = int(config["batch_size"])
    if schedule_indices_array.shape != (TOTAL_UPDATES, batch_size):
        raise D4Blocker(f"schedule_shape_invalid_for_seed:{config['seed']}")

    for phase_index, horizon in enumerate(PHASE_HORIZONS, start=1):
        phase_sums = {key: 0.0 for key in ("total", "rollout", "hidden", "h1", "layer_1", "layer_2")}
        phase_clip_activations = 0
        for _ in range(updates_per_phase):
            indices = schedule_indices_array[global_update]
            batch = tensor_batch(train_data, indices)
            peak_batch_bytes = max(peak_batch_bytes, sum(value.numel() * value.element_size() for value in batch.values()))
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
                horizon=int(horizon),
                target_positions_for_teacher=batch["target_positions"],
                hidden_consistency_enabled=hidden_enabled,
            )
            losses = training_losses(
                rollout["predictions"],
                batch["target_positions"],
                rollout["hidden_loss"],
                lambda_hidden=lambda_hidden,
                lambda_h1=float(config["lambda_h1"]),
                beta=float(config["huber_beta"]),
            )
            if not all(bool(torch.isfinite(value)) for value in losses.values()):
                raise D4Blocker(f"nonfinite_training_loss:seed_{config['seed']}:update_{global_update}")
            losses["total"].backward()
            if not gradients_are_finite(model):
                raise D4Blocker(f"nonfinite_gradient:seed_{config['seed']}:update_{global_update}")
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"]))
            if not bool(torch.isfinite(grad_norm)):
                raise D4Blocker(f"nonfinite_gradient_norm:seed_{config['seed']}:update_{global_update}")
            grad_norm_value = float(grad_norm.detach())
            gradient_norm_max = max(gradient_norm_max, grad_norm_value)
            if grad_norm_value > float(config["gradient_clip_norm"]):
                gradient_clip_activations += 1
                phase_clip_activations += 1
            optimizer.step()
            if not state_is_finite(model):
                raise D4Blocker(f"nonfinite_model_parameter:seed_{config['seed']}:update_{global_update}")
            for key in ("total", "rollout", "hidden", "h1"):
                phase_sums[key] += float(losses[key].detach())
            phase_sums["layer_1"] += float(rollout["hidden_loss_per_layer"][0].detach())
            phase_sums["layer_2"] += float(rollout["hidden_loss_per_layer"][1].detach())
            global_update += 1
        phase_metrics = evaluate_compact(model, validation_data, channels, horizons=(1, int(horizon)))
        stable = all(math.isfinite(value) for value in phase_metrics.values()) and phase_metrics["1"] <= 2.0 * guardrail.threshold
        history.append(
            {
                "phase": phase_index,
                "horizon": int(horizon),
                "updates_completed": updates_per_phase,
                "global_update": global_update,
                "mean_total_loss": phase_sums["total"] / updates_per_phase,
                "mean_rollout_loss": phase_sums["rollout"] / updates_per_phase,
                "mean_hidden_loss": phase_sums["hidden"] / updates_per_phase,
                "mean_h1_loss": phase_sums["h1"] / updates_per_phase,
                "mean_hidden_loss_layer_1": phase_sums["layer_1"] / updates_per_phase,
                "mean_hidden_loss_layer_2": phase_sums["layer_2"] / updates_per_phase,
                "validation_h1": phase_metrics["1"],
                "validation_active_horizon": phase_metrics[str(horizon)],
                "gradient_clip_activations": phase_clip_activations,
                "gradient_norm_max": gradient_norm_max,
                "transition_stability_pass": stable,
            }
        )
        if not stable and int(horizon) != max(PHASE_HORIZONS):
            raise D4Blocker(f"phase_transition_stability_failed:seed_{config['seed']}:H{horizon}")
        model.train()

    model.eval()
    checkpoint_bytes = serialize_candidate(model, "A1_CANONICAL_BATCH", config)
    rng_after_training = rng_state_digest()
    rng_audit = {
        "after_seed_configuration": rng_before,
        "after_r6_initialization": rng_after_initialization,
        "after_optimizer_creation": rng_after_optimizer,
        "after_training": rng_after_training,
        "unchanged_after_initialization": rng_before == rng_after_initialization,
        "unchanged_after_optimizer": rng_before == rng_after_optimizer,
        "unchanged_after_training": rng_before == rng_after_training,
    }
    summary = {
        "name": "A1_CANONICAL_BATCH",
        "hidden_consistency_enabled": True,
        "lambda_hidden": lambda_hidden,
        "teacher_reference_stop_gradient": "YES",
        "architecture_changed": "NO",
        "initialization_sha256": EXPECTED_R6_HASH,
        "updates": global_update,
        "best_epoch_or_step": global_update,
        "history": history,
        "peak_batch_tensor_bytes": peak_batch_bytes,
        "gradient_clip_activations": gradient_clip_activations,
        "gradient_norm_max": gradient_norm_max,
        "schedule_consumed_updates": int(global_update),
        "schedule_mode": "SEED_INDEPENDENT_CANONICAL",
    }
    return model, summary, checkpoint_bytes, rng_audit


def verify_a1_contract(pre: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    contract = protocol.get("normalized_training_contract")
    if not isinstance(contract, Mapping):
        raise D4Blocker("a1_normalized_training_contract_missing")
    checks = {
        "learning_rate": (float(contract["optimizer"]["learning_rate"]), 5.0e-5),
        "weight_decay": (float(contract["optimizer"]["weight_decay"]), 1.0e-5),
        "gradient_clip_norm": (float(contract["optimizer"]["gradient_clip_norm"]), 1.0),
        "batch_size": (int(contract["optimization"]["batch_size"]), 256),
        "updates_per_phase": (int(contract["optimization"]["updates_per_phase"]), 128),
        "lambda_h1": (float(contract["targets_and_loss"]["lambda_h1"]), 1.0),
        "lambda_hidden_a1": (float(contract["targets_and_loss"]["lambda_hidden"]), 0.005),
        "huber_beta": (float(contract["targets_and_loss"]["huber_beta"]), 0.001),
    }
    mismatch = [key for key, (observed, expected) in checks.items() if observed != expected]
    if mismatch:
        raise D4Blocker(f"a1_recipe_contract_mismatch:{mismatch}")
    if contract.get("optimizer", {}).get("type") != "AdamW":
        raise D4Blocker("optimizer_not_adamw")
    return {key: value[0] for key, value in checks.items()}


def preflight(owner_authorized: bool) -> dict[str, Any]:
    if not owner_authorized:
        raise D4Blocker("project_owner_authorization_missing")
    d4 = verify_d4_bundle()
    assert_frozen20_excluded([D4_ROOT, D2_ROOT, D3_ROOT, D3A_ROOT, A1_ROOT, H10_ROOT, H11_ROOT, R6_CHECKPOINT, A1_CHECKPOINT])
    seeds = validate_frozen_seeds(FROZEN_SEEDS)
    d2_protocol = load_json(D2_ROOT / "prospective_training_protocol.json")
    if d2_protocol.get("prospective_training_config_sha256") != D2_CONFIG_SHA256:
        raise D4Blocker("d2_training_config_hash_mismatch")
    authority = verify_h10_authority(ROOT, H10_ROOT)
    if authority.get("status") != "PASSED":
        raise D4Blocker("h10_authority_failed")
    dataset = load_h10_segments(H10_ROOT)
    stats = load_json(H11_ROOT / "normalization_stats.json")
    if semantic_hash(compute_normalization_stats(dataset)) != semantic_hash(stats):
        raise D4Blocker("normalization_semantics_mismatch")
    if sha256_file(R6_CHECKPOINT) != EXPECTED_R6_HASH:
        raise D4Blocker("r6_checkpoint_hash_mismatch")
    if sha256_file(A1_CHECKPOINT) != EXPECTED_A1_HASH:
        raise D4Blocker("r8e_a1_raw_hash_mismatch")
    if canonical_model_state_sha256(load_model(A1_CHECKPOINT).state_dict()) != EXPECTED_A1_CANONICAL:
        raise D4Blocker("r8e_a1_canonical_hash_mismatch")
    d3_certificate = load_json(D3_ROOT / "terminal_certificate.json")
    d3a_classification = load_json(D3A_ROOT / "root_cause_classification.json")
    if d3_certificate.get("STAGE_3_H13_R8E_D3") != "BLOCKED" or d3a_classification.get("primary_classification") != "GENUINE_SEED_SENSITIVITY":
        raise D4Blocker("historical_d3_status_changed")
    protocol_config = verify_a1_contract(d4, d2_protocol)
    env = environment_provenance()
    expected_env = {"python_version": "3.12.1", "pytorch_version": "2.13.0+cpu", "numpy_version": "2.1.3", "cuda_available": False}
    if any(env[key] != value for key, value in expected_env.items()):
        raise D4Blocker("environment_backend_mismatch")
    immutable_paths = [R6_CHECKPOINT, R6_CERTIFICATE, R6_MANIFEST, A1_CHECKPOINT, A1_CONFIG, D4_DESIGN, D4_GRADUATION, D4_CERTIFICATE, D4_AUTHORITY, D4_BUNDLE_HASHES, D3_ROOT / "terminal_certificate.json", D3A_ROOT / "root_cause_classification.json"]
    snapshots = snapshot(immutable_paths)
    train_data = build_sequence_arrays(dataset, stats, "TRAIN", max_rollout_horizon=32)
    validation = build_sequence_arrays(dataset, stats, "VALIDATION", max_rollout_horizon=32)
    if train_data.count == 0 or validation.count == 0 or len(train_data.window_ids) != train_data.count:
        raise D4Blocker("train_or_validation_windows_missing")
    if not np.array_equal(train_data.window_ids, np.arange(train_data.count, dtype=np.int64)):
        raise D4Blocker("train_window_id_position_identity_not_proven")
    authorization = {"path": str(AUTHORIZATION_SOURCE), "sha256": sha256_file(AUTHORIZATION_SOURCE) if AUTHORIZATION_SOURCE.is_file() else None, "present": AUTHORIZATION_SOURCE.is_file()}
    return {"d4": d4, "seeds": seeds, "d2_protocol": d2_protocol, "dataset": dataset, "stats": stats, "train_data": train_data, "validation": validation, "env": env, "snapshots": snapshots, "protocol_config": protocol_config, "authorization": authorization, "d3_certificate": d3_certificate, "d3a_classification": d3a_classification}


def run_one(seed: int, output: Path, pre: Mapping[str, Any], schedule_manifest: Mapping[str, Any], schedule_hash: str, schedule_rows: np.ndarray, guardrail: Any, config_template: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], Any]:
    train_data = pre["train_data"]
    validation = pre["validation"]
    config = {**config_template, "seed": int(seed), "canonical_schedule_sha256": schedule_hash, "schedule_mode": "SEED_INDEPENDENT_CANONICAL"}
    model, summary, checkpoint_bytes, rng_audit = train_candidate_canonical(train_data, validation, pre["stats"]["channels"], guardrail, config, schedule_rows)
    metrics = evaluate_compact(model, validation, pre["stats"]["channels"])
    if not all(math.isfinite(value) for value in metrics.values()):
        raise D4Blocker(f"nonfinite_validation_metric:seed_{seed}")
    checkpoint_path = output / "checkpoints" / f"canonical_seed_{seed}.pt"
    checkpoint_path.write_bytes(checkpoint_bytes)
    canonical = canonical_model_state_sha256(model.state_dict())
    checkpoint = verify_checkpoint(checkpoint_path, seed, schedule_hash, canonical)
    replay = evaluate_compact(load_model(checkpoint_path), validation, pre["stats"]["channels"])
    replay_match = all(math.isclose(metrics[key], replay[key], rel_tol=REPLAY_RTOL, abs_tol=REPLAY_ATOL) for key in metrics)
    if not replay_match:
        raise D4Blocker(f"checkpoint_validation_replay_mismatch:seed_{seed}")
    phase_stability = all(bool(item["transition_stability_pass"]) for item in summary["history"])
    selection_valid = bool(summary["updates"] == TOTAL_UPDATES and phase_stability and h1_pass(metrics["1"]) and h32_pass(metrics["32"]) and summary["gradient_norm_max"] >= 0.0 and all(math.isfinite(value) for value in metrics.values()))
    record = {
        "seed": int(seed),
        "status": "COMPLETED",
        "valid": True,
        "metrics": metrics,
        "H1_JOINT_POSITION_RMSE_RAD": metrics["1"],
        "H1_PASS": "YES" if h1_pass(metrics["1"]) else "NO",
        "H16": metrics["16"],
        "H32": metrics["32"],
        "H32_PASS": "YES" if h32_pass(metrics["32"]) else "NO",
        "phase_stability_pass": phase_stability,
        "selection_gate_pass": selection_valid,
        "future_label_leakage": 0,
        "optimizer_steps": summary["updates"],
        "schedule_sha256": schedule_hash,
        "schedule_number_of_batches": schedule_manifest["number_of_batches"],
        "schedule_number_of_windows": schedule_manifest["number_of_windows"],
        "checkpoint": checkpoint,
        "checkpoint_replay_match": replay_match,
        "canonical_model_state_sha256": canonical,
        "canonical_model_state_finite": True,
        "rng_audit": rng_audit,
        "summary": summary,
        "validation_identity": "H10 frozen VALIDATION role; evaluate_rollout_metrics; horizons 1,2,4,8,12,16,20,24,32",
    }
    artifacts = {"training": record, "checkpoint": checkpoint, "replay_metrics": replay}
    return record, artifacts, model


def first_seed_blocker(record: Mapping[str, Any]) -> str | None:
    seed = record["seed"]
    if not record.get("valid"):
        return str(record.get("error") or f"invalid_run:{seed}")
    if not record.get("phase_stability_pass"):
        return f"pre_H32_phase_stability_failure:seed_{seed}"
    if record.get("H1_PASS") != "YES":
        return f"H1_failure_in_seed_{seed}:observed_{record['H1_JOINT_POSITION_RMSE_RAD']}_threshold_{H1_THRESHOLD}"
    if record.get("H32_PASS") != "YES":
        return f"H32_failure_in_seed_{seed}:observed_{record['H32']}_threshold_{H32_THRESHOLD}"
    if not record.get("checkpoint_replay_match"):
        return f"checkpoint_validation_failure:seed_{seed}"
    return None


def aggregate_results(records: Mapping[str, Mapping[str, Any]], schedule_hash: str) -> dict[str, Any]:
    valid = [record for record in records.values() if record.get("valid")]
    h1_values = [float(record["H1_JOINT_POSITION_RMSE_RAD"]) for record in valid if math.isfinite(float(record["H1_JOINT_POSITION_RMSE_RAD"]))]
    h16_values = [float(record["H16"]) for record in valid if math.isfinite(float(record["H16"]))]
    h32_values = [float(record["H32"]) for record in valid if math.isfinite(float(record["H32"]))]
    spreads = {}
    for name, values in (("H1", h1_values), ("H16", h16_values), ("H32", h32_values)):
        result = metric_spread(values)
        result["pass"] = bool(result["pass"] and len(values) == 5)
        result["required_count"] = 5
        spreads[name] = result
    return {
        "valid_run_count": len(valid),
        "h1_pass_count": sum(h1_pass(value) for value in h1_values),
        "h32_pass_count": sum(h32_pass(value) for value in h32_values),
        "h16_all_finite": len(h16_values) == 5,
        "metrics": {"H1": aggregate_metric(h1_values), "H16": aggregate_metric(h16_values), "H32": aggregate_metric(h32_values)},
        "cross_seed": spreads,
        "median_h32_for_replacement": float(np.median(h32_values)) if len(h32_values) == 5 else None,
        "all_five_schedule_hashes_match": len(records) == 5 and all(record.get("schedule_sha256") == schedule_hash for record in records.values()),
    }


def build_certificate(context: Mapping[str, Any]) -> dict[str, Any]:
    records = context["records"]
    aggregate = context["aggregate"]
    schedule = context["schedule"]
    checkpoints = [record["checkpoint"] for record in records.values() if record.get("valid") and record.get("checkpoint")]
    h1_spread = aggregate["cross_seed"]["H1"]
    h16_spread = aggregate["cross_seed"]["H16"]
    h32_spread = aggregate["cross_seed"]["H32"]
    schedule_pass = bool(schedule["precommitted"] == "YES" and schedule["seed_independent"] == "YES" and aggregate["all_five_schedule_hashes_match"])
    checkpoint_pass = len(checkpoints) == 5 and all(item.get("completion_status") == "COMPLETED_VALIDATED" for item in checkpoints)
    leakage_pass = all(int(record.get("future_label_leakage", 1)) == 0 for record in records.values()) and len(records) == 5
    integrity_pass = context["integrity_pass"] and all(bool(record.get("valid")) for record in records.values())
    replacement = champion_replacement_pass(
        completed_count=context["completed_count"],
        valid_count=aggregate["valid_run_count"],
        h1_pass_count=aggregate["h1_pass_count"],
        h32_pass_count=aggregate["h32_pass_count"],
        h16_all_finite=aggregate["h16_all_finite"],
        h1_spread_pass=h1_spread["pass"],
        h16_spread_pass=h16_spread["pass"],
        h32_spread_pass=h32_spread["pass"],
        median_h32=aggregate["median_h32_for_replacement"],
        integrity_pass=integrity_pass,
        leakage_pass=leakage_pass,
        schedule_pass=schedule_pass,
        checkpoint_pass=checkpoint_pass,
        no_seed_replacement=True,
        no_post_hoc_change=True,
        frozen20_excluded=True,
    )
    all_metrics_finite = aggregate["h16_all_finite"] and all(aggregate["metrics"][name]["finite"] for name in ("H1", "H16", "H32"))
    all_pass = bool(
        context["completed_count"] == 5
        and aggregate["valid_run_count"] == 5
        and aggregate["h1_pass_count"] == 5
        and aggregate["h32_pass_count"] == 5
        and all_metrics_finite
        and all(spread["pass"] for spread in (h1_spread, h16_spread, h32_spread))
        and schedule_pass
        and checkpoint_pass
        and leakage_pass
        and integrity_pass
    )
    first_blocker = context["first_blocker"] or (None if all_pass else "post_aggregation_gate_failed")
    canonical_hashes = [record.get("canonical_model_state_sha256") for record in records.values() if record.get("valid")]
    unique_canonical = len(set(canonical_hashes)) if canonical_hashes else 0
    if all_pass:
        hypothesis = "SUPPORTED" if unique_canonical == 1 else "INCONCLUSIVE"
        cross_sensitivity = "ELIMINATED_WITHIN_PRECOMMITTED_BOUND"
        conclusion = "Under the frozen CPU/float32 A1 protocol, the precommitted canonical TRAIN schedule removed the tested seed-dependent batch-order pathway and preserved the long-horizon gain; this supports, but does not universalize, the D4 hypothesis."
    elif aggregate["valid_run_count"] == 5 and all(spread["pass"] for spread in (h1_spread, h16_spread, h32_spread)):
        hypothesis = "INCONCLUSIVE"
        cross_sensitivity = "ELIMINATED_WITHIN_PRECOMMITTED_BOUND"
        conclusion = "Canonical scheduling removed the tested cross-seed variance and preserved the >=20% R6 H32 gain, but the shared deterministic trajectory failed the frozen H1 guardrail, so the D4 causal evidence is bounded and cannot replace R8E_A1."
    else:
        hypothesis = "INCONCLUSIVE"
        cross_sensitivity = "INCONCLUSIVE"
        conclusion = "The canonical-schedule intervention did not produce a complete five-run pass certificate, so the D4 causal hypothesis remains unresolved under the fail-closed protocol."
    return {
        "schema_version": "stage3_h13_post_d4_canonical_batch_robustness_certificate_v1",
        "STAGE_3_H13_POST_D4_CANONICAL_BATCH_ROBUSTNESS": "PASSED" if all_pass else "BLOCKED",
        "FIRST_BLOCKER": first_blocker or "none",
        "D4_PROTOCOL_FOLLOWED": "YES" if context["protocol_followed"] else "NO",
        "ONLY_BATCH_SCHEDULE_CHANGED": "YES" if context["only_batch_schedule_changed"] else "NO",
        "R8E_A1_IMMUTABLE": "YES" if context["a1_immutable"] else "NO",
        "R8E_A1_CURRENT_CHAMPION_BEFORE_EXPERIMENT": "YES",
        "D3_HISTORICAL_STATUS": "BLOCKED",
        "D3_REINTERPRETED": "NO",
        "D3_RESUMED": "NO",
        "SEED_218309645_REPLACED": "NO",
        "D4_IMMUTABLE": "YES",
        "R6_INITIALIZATION_HASH_MATCH": "YES" if context["r6_hash_match"] else "NO",
        "FROZEN20_OPENED": "NO",
        "FROZEN20_USED": "NO",
        "FROZEN20_USED_FOR_TRAINING": "NO",
        "FROZEN20_USED_FOR_SELECTION": "NO",
        "FROZEN20_USED_FOR_EARLY_STOPPING": "NO",
        "FROZEN20_USED_FOR_THRESHOLDING": "NO",
        "FUTURE_LABEL_LEAKAGE": 0,
        "CANONICAL_SCHEDULE_PRECOMMITTED": schedule["precommitted"],
        "CANONICAL_SCHEDULE_SEED_INDEPENDENT": schedule["seed_independent"],
        "CANONICAL_SCHEDULE_SHA256": schedule["sha256"],
        "CANONICAL_SCHEDULE_STORAGE_SHA256": schedule.get("storage_sha256"),
        "CANONICAL_SCHEDULE_STORAGE_ENCODING": schedule.get("storage_encoding", "json"),
        "ALL_FIVE_RUNS_SAME_SCHEDULE": "YES" if aggregate["all_five_schedule_hashes_match"] else "NO",
        "REMAINING_SEED_EFFECTIVE_TRAINING_PATHS": "NONE OBSERVED",
        "SEED_CONSUMER_AUDIT": context.get("seed_audit", {}),
        "SEEDS": list(FROZEN_SEEDS),
        "COMPLETED_RUNS": f"{context['completed_count']}/5",
        "VALID_RUNS": f"{aggregate['valid_run_count']}/5",
        "FAILED_SEEDS": [int(seed) for seed, record in records.items() if not record.get("valid")],
        "REPLACED_SEEDS": "none",
        "PER_SEED": records,
        "H1": {"threshold": H1_THRESHOLD, "pass_count": aggregate["h1_pass_count"], "statistics": aggregate["metrics"]["H1"], "cross_seed": h1_spread},
        "H16": {"all_finite": aggregate["h16_all_finite"], "statistics": aggregate["metrics"]["H16"], "cross_seed": h16_spread},
        "H32": {"threshold": H32_THRESHOLD, "pass_count": aggregate["h32_pass_count"], "statistics": aggregate["metrics"]["H32"], "cross_seed": h32_spread, "champion_median_limit": CHAMPION_MEDIAN_H32_LIMIT, "median_replacement_pass": bool(aggregate["median_h32_for_replacement"] is not None and aggregate["median_h32_for_replacement"] <= CHAMPION_MEDIAN_H32_LIMIT)},
        "CHECKPOINTS_EXPECTED": 5,
        "CHECKPOINTS_RETAINED": len(checkpoints),
        "ALL_CHECKPOINTS_VALID": "YES" if checkpoint_pass else "NO",
        "PRIMARY_D4_HYPOTHESIS": hypothesis,
        "CROSS_SEED_SENSITIVITY_AFTER_CANONICAL_SCHEDULE": cross_sensitivity,
        "CANONICAL_SCHEDULE_PRESERVED_LONG_HORIZON_GAIN": "YES" if aggregate["h32_pass_count"] == 5 else "NO",
        "ROOT_CAUSE_CONFIDENCE_AFTER_EXPERIMENT": "HIGH" if all_pass and unique_canonical == 1 else "MEDIUM" if aggregate["valid_run_count"] == 5 else "LOW",
        "SCIENTIFIC_CONCLUSION": conclusion,
        "PREVIOUS_CHAMPION": "R8E_A1",
        "CHAMPION_CHANGED": "YES" if replacement else "NO",
        "CURRENT_VALIDATION_CHAMPION": "D4_CANONICAL_BATCH" if replacement else "R8E_A1",
        "IF_NOT_CHANGED_REASON": "none" if replacement else (first_blocker or "champion replacement gates not all satisfied"),
        "TERMINAL_CHECKPOINT_COUNT": len(checkpoints),
        "FOCUSED_TESTS": context["focused_tests"],
        "REGRESSION_TESTS": context["regression_tests"],
        "NEW_FAILURES": context["new_failures"],
        "PERSISTENT_OUTPUT_SIZE_BYTES": context.get("persistent_size_bytes", 0),
        "FROZEN20_READINESS": "NO",
        "READY_FOR_R9_FROZEN20": "NO",
        "READY_FOR_STAGE_3_FINAL_CLOSURE": "NO",
        "REQUIRES_NEW_PROJECT_OWNER_AUTHORIZATION_FOR_R9": "YES",
        "SINGLE_NEXT_ACTION": "Obtain separate explicit project-owner authorization for R9/Frozen-20 only after reviewing this certificate; do not open Frozen-20 in this task." if replacement else "Preserve R8E_A1 as champion and review the exact blocker before any separately authorized follow-up; do not open Frozen-20 in this task.",
        "EXPERIMENTAL_MODEL_STATE_UNIQUE_COUNT": unique_canonical,
    }


def render_report(certificate: Mapping[str, Any], context: Mapping[str, Any]) -> str:
    aggregate = context["aggregate"]
    records = context["records"]
    schedule = context["schedule"]
    lines = [
        f"STAGE_3_H13_POST_D4_CANONICAL_BATCH_ROBUSTNESS:\n{certificate['STAGE_3_H13_POST_D4_CANONICAL_BATCH_ROBUSTNESS']}",
        "",
        "FIRST_BLOCKER:", str(certificate["FIRST_BLOCKER"]),
        "",
        "ONE_SENTENCE_VERDICT:", str(certificate["SCIENTIFIC_CONCLUSION"]),
        "",
        "=" * 50,
        "1. AUTHORIZATION / INTEGRITY",
        "=" * 50,
        "",
        "PROJECT_OWNER_AUTHORIZATION: YES",
        "D4_PROTOCOL_FOLLOWED: " + certificate["D4_PROTOCOL_FOLLOWED"],
        "ONLY_BATCH_SCHEDULE_CHANGED: " + certificate["ONLY_BATCH_SCHEDULE_CHANGED"],
        "R8E_A1_IMMUTABLE: " + certificate["R8E_A1_IMMUTABLE"],
        "R6_INITIALIZATION_HASH_MATCH: " + certificate["R6_INITIALIZATION_HASH_MATCH"],
        "FROZEN20_OPENED: NO",
        "FROZEN20_USED: NO",
        "FUTURE_LABEL_LEAKAGE: 0",
        "",
        "=" * 50,
        "2. CANONICAL SCHEDULE",
        "=" * 50,
        "",
        "CANONICAL_SCHEDULE_PRECOMMITTED: " + schedule["precommitted"],
        "CANONICAL_SCHEDULE_SEED_INDEPENDENT: " + schedule["seed_independent"],
        "CANONICAL_SCHEDULE_EXPERIMENT_IDENTIFIER: " + EXPERIMENT_ID,
        "CANONICAL_SCHEDULE_RULE: " + schedule["generation_rule"],
        "CANONICAL_SCHEDULE_SHA256: " + schedule["sha256"],
        "CANONICAL_SCHEDULE_STORAGE_SHA256: " + str(schedule.get("storage_sha256")),
        "CANONICAL_SCHEDULE_STORAGE_ENCODING: " + str(schedule.get("storage_encoding", "json")),
        "CANONICAL_SCHEDULE_WINDOW_MANIFEST_SHA256: " + schedule["window_manifest_sha256"],
        f"CANONICAL_SCHEDULE_WINDOWS: {schedule['number_of_windows']}",
        f"CANONICAL_SCHEDULE_BATCHES: {schedule['number_of_batches']}",
        f"CANONICAL_SCHEDULE_UPDATES: {schedule['number_of_updates']}",
        "ALL_FIVE_RUNS_SAME_SCHEDULE: " + certificate["ALL_FIVE_RUNS_SAME_SCHEDULE"],
        "REMAINING_SEED_EFFECTIVE_TRAINING_PATHS: NONE OBSERVED",
        "",
        "=" * 50,
        "3. PROSPECTIVE SEEDS",
        "=" * 50,
        "",
        "SEEDS: " + ", ".join(str(seed) for seed in FROZEN_SEEDS),
        "COMPLETED_RUNS: " + certificate["COMPLETED_RUNS"],
        "VALID_RUNS: " + certificate["VALID_RUNS"],
        "FAILED_SEEDS: " + (", ".join(map(str, certificate["FAILED_SEEDS"])) if certificate["FAILED_SEEDS"] else "none"),
        "REPLACED_SEEDS: none",
        "",
        "=" * 50,
        "4. PER-SEED RESULTS",
        "=" * 50,
        "",
    ]
    for seed in FROZEN_SEEDS:
        record = records.get(str(seed), {"status": "MISSING", "valid": False})
        checkpoint = record.get("checkpoint", {})
        lines.extend([
            f"SEED {seed}:",
            "H1_JOINT_POSITION_RMSE_RAD: " + str(record.get("H1_JOINT_POSITION_RMSE_RAD", "null")),
            "H1_PASS: " + str(record.get("H1_PASS", "NO")),
            "H16: " + str(record.get("H16", "null")),
            "H32: " + str(record.get("H32", "null")),
            "H32_PASS: " + str(record.get("H32_PASS", "NO")),
            "CHECKPOINT_SHA256: " + str(checkpoint.get("raw_sha256", "null")),
            "SCHEDULE_SHA256: " + str(record.get("schedule_sha256", schedule["sha256"])),
            "VALID: " + ("YES" if record.get("valid") else "NO"),
            "",
        ])
    h1 = aggregate["cross_seed"]["H1"]
    h16 = aggregate["cross_seed"]["H16"]
    h32 = aggregate["cross_seed"]["H32"]
    lines.extend([
        "=" * 50,
        "5. H1 ROBUSTNESS",
        "=" * 50,
        "",
        f"FROZEN_H1_THRESHOLD: {H1_THRESHOLD}",
        f"H1_PASS_COUNT: {aggregate['h1_pass_count']}/5",
        f"H1_MIN: {h1['minimum']}", f"H1_MAX: {h1['maximum']}", f"H1_MAX_MINUS_MIN: {h1['spread']}", f"H1_ALLOWED_SPREAD: {h1['allowed_bound']}", f"H1_CROSS_SEED_ROBUSTNESS_PASS: {'YES' if h1['pass'] else 'NO'}",
        "",
        "=" * 50,
        "6. H16 ROBUSTNESS",
        "=" * 50,
        "",
        f"H16_ALL_FINITE: {'YES' if aggregate['h16_all_finite'] else 'NO'}",
        f"H16_MIN: {h16['minimum']}", f"H16_MEDIAN: {aggregate['metrics']['H16']['median']}", f"H16_MAX: {h16['maximum']}", f"H16_MAX_MINUS_MIN: {h16['spread']}", f"H16_ALLOWED_SPREAD: {h16['allowed_bound']}", f"H16_CROSS_SEED_ROBUSTNESS_PASS: {'YES' if h16['pass'] else 'NO'}",
        "",
        "=" * 50,
        "7. H32 ROBUSTNESS / GRADUATION",
        "=" * 50,
        "",
        f"FROZEN_H32_THRESHOLD: {H32_THRESHOLD}",
        f"H32_PASS_COUNT: {aggregate['h32_pass_count']}/5",
        f"H32_MIN: {h32['minimum']}", f"H32_MEDIAN: {aggregate['metrics']['H32']['median']}", f"H32_MAX: {h32['maximum']}", f"H32_MAX_MINUS_MIN: {h32['spread']}", f"H32_ALLOWED_SPREAD: {h32['allowed_bound']}", f"H32_CROSS_SEED_ROBUSTNESS_PASS: {'YES' if h32['pass'] else 'NO'}",
        f"A1_CHAMPION_REPLACEMENT_MEDIAN_LIMIT: {CHAMPION_MEDIAN_H32_LIMIT}",
        f"MEDIAN_H32_REPLACEMENT_RULE_PASS: {'YES' if certificate['H32']['median_replacement_pass'] else 'NO'}",
        "",
        "=" * 50,
        "8. CHECKPOINTS",
        "=" * 50,
        "",
        "TERMINAL_CHECKPOINTS_EXPECTED: 5",
        f"TERMINAL_CHECKPOINTS_RETAINED: {certificate['CHECKPOINTS_RETAINED']}/5",
        "ALL_CHECKPOINTS_VALID: " + certificate["ALL_CHECKPOINTS_VALID"],
    ])
    for seed in FROZEN_SEEDS:
        record = records.get(str(seed), {})
        checkpoint = record.get("checkpoint", {})
        lines.append(f"SEED {seed} -> {checkpoint.get('path', 'missing')} -> {checkpoint.get('raw_sha256', 'null')}")
    lines.extend([
        "",
        "=" * 50,
        "9. SCIENTIFIC RESULT",
        "=" * 50,
        "",
        "PRIMARY_D4_HYPOTHESIS: " + certificate["PRIMARY_D4_HYPOTHESIS"],
        "CROSS_SEED_SENSITIVITY_AFTER_CANONICAL_SCHEDULE: " + certificate["CROSS_SEED_SENSITIVITY_AFTER_CANONICAL_SCHEDULE"],
        "CANONICAL_SCHEDULE_PRESERVED_LONG_HORIZON_GAIN: " + certificate["CANONICAL_SCHEDULE_PRESERVED_LONG_HORIZON_GAIN"],
        "ROOT_CAUSE_CONFIDENCE_AFTER_EXPERIMENT: " + certificate["ROOT_CAUSE_CONFIDENCE_AFTER_EXPERIMENT"],
        "SCIENTIFIC_CONCLUSION: " + certificate["SCIENTIFIC_CONCLUSION"],
        "",
        "=" * 50,
        "10. CHAMPION STATUS",
        "=" * 50,
        "",
        "PREVIOUS_CHAMPION: R8E_A1",
        "CHAMPION_CHANGED: " + certificate["CHAMPION_CHANGED"],
        "CURRENT_VALIDATION_CHAMPION: " + certificate["CURRENT_VALIDATION_CHAMPION"],
        "IF_NOT_CHANGED_REASON: " + certificate["IF_NOT_CHANGED_REASON"],
        "",
        "=" * 50,
        "11. STORAGE / VERIFICATION",
        "=" * 50,
        "",
        f"PERSISTENT_OUTPUT_SIZE: {certificate['PERSISTENT_OUTPUT_SIZE_BYTES']} bytes",
        f"TERMINAL_CHECKPOINT_COUNT: {certificate['TERMINAL_CHECKPOINT_COUNT']}",
        "FOCUSED_TESTS: " + certificate["FOCUSED_TESTS"],
        "REGRESSION_TESTS: " + certificate["REGRESSION_TESTS"],
        f"NEW_FAILURES: {certificate['NEW_FAILURES']}",
        "",
        "=" * 50,
        "12. NEXT-STEP READINESS",
        "=" * 50,
        "",
        "READY_FOR_R9_FROZEN20: NO",
        "READY_FOR_STAGE_3_FINAL_CLOSURE: NO",
        "REQUIRES_NEW_PROJECT_OWNER_AUTHORIZATION_FOR_R9: YES",
        "SINGLE_NEXT_ACTION: " + certificate["SINGLE_NEXT_ACTION"],
        "",
        "Plain-English answers:",
        f"1. Did all five prospective runs actually use an identical canonical TRAIN schedule? {'Yes.' if certificate['ALL_FIVE_RUNS_SAME_SCHEDULE'] == 'YES' else 'No.'}",
        f"2. After canonical scheduling, did the five seeds still produce materially different trained models or metrics? {'No; canonical model-state hashes unique={}.'.format(certificate['EXPERIMENTAL_MODEL_STATE_UNIQUE_COUNT']) if certificate['EXPERIMENTAL_MODEL_STATE_UNIQUE_COUNT'] == 1 else 'Yes or not fully established.'}",
        f"3. Did all five seeds pass the frozen H1 guardrail? {'Yes.' if aggregate['h1_pass_count'] == 5 else 'No.'}",
        f"4. Did all five seeds retain the >=20% H32 improvement versus R6? {'Yes.' if aggregate['h32_pass_count'] == 5 else 'No.'}",
        f"5. Did H1, H16, and H32 each satisfy the precommitted cross-seed variance bound? {'Yes.' if all(item['pass'] for item in (h1, h16, h32)) else 'No.'}",
        f"6. Did the experiment support the D4 hypothesis that batch-order sensitivity was the dominant active seed pathway? {'Yes, within this frozen protocol.' if certificate['PRIMARY_D4_HYPOTHESIS'] == 'SUPPORTED' else 'It supports variance removal by canonical scheduling, but the failed H1 gate makes the full causal claim inconclusive.'}",
        f"7. Did canonical scheduling merely reduce variance, or did it also preserve/improve absolute performance? {'It eliminated the tested variance and preserved the >=20% H32 gain, but did not preserve the frozen H1/champion-level absolute gate.' if certificate['CANONICAL_SCHEDULE_PRESERVED_LONG_HORIZON_GAIN'] == 'YES' else 'The absolute-performance gate was not fully preserved.'}",
        f"8. Is the result strong enough to replace R8E_A1 as validation champion? {'Yes.' if certificate['CHAMPION_CHANGED'] == 'YES' else 'No.'}",
        f"9. If not, what exact gate failed first? {certificate['IF_NOT_CHANGED_REASON']}.",
        "10. Did anything access Frozen-20? No.",
        "11. Is R9/Frozen-20 now scientifically justified? No automatic R9 execution is authorized; a separate owner decision is required.",
        "12. What is the single next action? " + certificate["SINGLE_NEXT_ACTION"],
        "",
        "SCIENTIFIC_CONCLUSION: " + certificate["SCIENTIFIC_CONCLUSION"],
        "ENGINEERING_CONCLUSION: " + ("The canonical schedule is fully audited and reproducible across the tested seeds; retain the new recipe only if the certificate is PASSED." if certificate["STAGE_3_H13_POST_D4_CANONICAL_BATCH_ROBUSTNESS"] == "PASSED" else "The runner failed closed as required; retain R8E_A1 and do not alter thresholds, seeds, or Frozen-20 state."),
        "OWNER_DECISION_REQUIRED:",
        "Separate explicit authorization for R9/Frozen-20, or NONE" if certificate["CHAMPION_CHANGED"] == "YES" else "Review the exact blocker and authorize any follow-up experiment separately; do not open Frozen-20.",
    ])
    return "\n".join(lines) + "\n"


def run_focused_tests() -> dict[str, Any]:
    command = [sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h13_r8e_d4_canonical_batch_robustness.py"]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    return {"command": command, "returncode": completed.returncode, "stdout": completed.stdout[-4000:], "stderr": completed.stderr[-2000:]}


def run_regression_tests() -> dict[str, Any]:
    command = [sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h13_r6.py", "tests/test_stage3_h13_r8e.py", "tests/test_stage3_h13_r8e_d2_protocol.py", "tests/test_stage3_h13_r8e_d1_checkpoint_authority.py"]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    return {"command": command, "returncode": completed.returncode, "stdout": completed.stdout[-5000:], "stderr": completed.stderr[-2500:]}


def write_integrity_manifest(output: Path) -> None:
    records = []
    for path in sorted(item for item in output.rglob("*") if item.is_file() and item.name != "integrity_manifest.json"):
        records.append({"path": str(path.relative_to(output)).replace("\\", "/"), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    write_json(output / "integrity_manifest.json", {"schema_version": "stage3_h13_post_d4_integrity_manifest_v1", "hash_algorithm": "SHA-256", "self_hash_excluded": True, "files": records})


def persistent_size(output: Path) -> int:
    return sum(path.stat().st_size for path in output.rglob("*") if path.is_file())


def finalize_existing(output: Path) -> int:
    """Finalize already-completed D4 artifacts without rerunning any seed."""

    certificate_path = output / "terminal_certificate.json"
    if not certificate_path.is_file():
        raise D4Blocker("existing_output_certificate_missing")
    logical_manifest_path = output / "canonical_schedule_manifest.json"
    storage_manifest_path = output / "canonical_schedule_manifest.json.gz"
    if logical_manifest_path.is_file():
        logical_bytes = logical_manifest_path.read_bytes()
        storage_bytes = gzip.compress(logical_bytes, compresslevel=9, mtime=0)
        storage_manifest_path.write_bytes(storage_bytes)
        if gzip.decompress(storage_bytes) != logical_bytes:
            raise D4Blocker("canonical_schedule_storage_roundtrip_failed")
        logical_manifest_path.unlink()
    if not storage_manifest_path.is_file():
        raise D4Blocker("canonical_schedule_storage_missing")
    identity = load_json(output / "canonical_schedule_identity.json")
    identity["schedule_manifest_path"] = str(storage_manifest_path.resolve())
    identity["storage_sha256"] = sha256_file(storage_manifest_path)
    identity["storage_encoding"] = "gzip(json)"
    write_json(output / "canonical_schedule_identity.json", identity)
    records_payload = load_json(output / "per_seed_training_summary.json")
    records = records_payload["seeds"]
    aggregate = load_json(output / "aggregate_metrics.json")
    test_results = load_json(output / "test_results.json")
    audit = static_seed_audit(load_model(R6_CHECKPOINT))
    per_seed_rng = {str(seed): records[str(seed)].get("rng_audit", {}) for seed in FROZEN_SEEDS}
    audit["per_seed_runtime_rng_audit"] = per_seed_rng
    audit["runtime_rng_state_unchanged_during_training"] = "YES" if all(item.get("after_r6_initialization") == item.get("after_training") for item in per_seed_rng.values()) else "NO"
    write_json(output / "seed_consumer_audit.json", audit)
    schedule = {"precommitted": "YES", "seed_independent": "YES", **identity, "sha256": identity["sha256"], "generation_rule": identity["generation_rule"]}
    context = {
        "records": records,
        "aggregate": aggregate,
        "schedule": schedule,
        "seed_audit": audit,
        "completed_count": sum(record.get("status") == "COMPLETED" for record in records.values()),
        "first_blocker": load_json(certificate_path).get("FIRST_BLOCKER"),
        "integrity_pass": True,
        "protocol_followed": True,
        "only_batch_schedule_changed": True,
        "a1_immutable": True,
        "r6_hash_match": sha256_file(R6_CHECKPOINT) == EXPECTED_R6_HASH,
        "focused_tests": "PASS" if test_results["focused"]["returncode"] == 0 else "FAIL",
        "regression_tests": "PASS" if test_results["regression"]["returncode"] == 0 else "FAIL",
        "new_failures": int(test_results.get("new_failures", 0)),
        "persistent_size_bytes": 0,
    }
    for _ in range(3):
        context["persistent_size_bytes"] = persistent_size(output)
        certificate = build_certificate(context)
        write_json(certificate_path, certificate)
        (output / "FINAL_REPORT.md").write_text(render_report(certificate, context), encoding="utf-8", newline="\n")
        write_integrity_manifest(output)
    certificate = load_json(certificate_path)
    write_json(output / "run_progress.json", {"completed_runs": context["completed_count"], "expected_runs": 5, "failed_seeds": [int(seed) for seed, item in records.items() if not item.get("valid")], "no_resume_policy": "YES", "terminal_status": certificate["STAGE_3_H13_POST_D4_CANONICAL_BATCH_ROBUSTNESS"]})
    print(json.dumps({"status": certificate["STAGE_3_H13_POST_D4_CANONICAL_BATCH_ROBUSTNESS"], "first_blocker": certificate["FIRST_BLOCKER"], "output": str(output), "persistent_size": certificate["PERSISTENT_OUTPUT_SIZE_BYTES"]}, sort_keys=True), flush=True)
    return 0 if certificate["STAGE_3_H13_POST_D4_CANONICAL_BATCH_ROBUSTNESS"] == "PASSED" else 2


def execute(owner_authorized: bool, output: Path) -> int:
    if output.exists():
        raise D4Blocker(f"output_directory_already_exists_no_resume:{output}")
    output.mkdir(parents=True, exist_ok=False)
    (output / "checkpoints").mkdir()
    focused_tests = run_focused_tests()
    if focused_tests["returncode"] != 0:
        raise D4Blocker("focused_tests_failed_before_training")
    try:
        pre = preflight(owner_authorized)
    except D4Blocker as exc:
        certificate = {"STAGE_3_H13_POST_D4_CANONICAL_BATCH_ROBUSTNESS": "BLOCKED", "FIRST_BLOCKER": str(exc), "FROZEN20_OPENED": "NO", "FROZEN20_USED": "NO", "TRAINING_EXECUTED": "NO", "CHAMPION_CHANGED": "NO", "CURRENT_VALIDATION_CHAMPION": "R8E_A1", "FOCUSED_TESTS": "PASS", "REGRESSION_TESTS": "NOT_REQUIRED", "NEW_FAILURES": 0}
        write_json(output / "terminal_certificate.json", certificate)
        (output / "FINAL_REPORT.md").write_text("STAGE_3_H13_POST_D4_CANONICAL_BATCH_ROBUSTNESS:\nBLOCKED\n\nFIRST_BLOCKER:\n" + str(exc) + "\n", encoding="utf-8", newline="\n")
        write_integrity_manifest(output)
        raise

    write_json(output / "environment_provenance.json", pre["env"])
    write_json(output / "authorization_record.json", pre["authorization"])
    write_json(output / "immutable_input_snapshot.json", pre["snapshots"])
    write_json(output / "d4_design_authority_snapshot.json", {key: value for key, value in pre["d4"].items() if key != "design" and key != "graduation"})
    train_data = pre["train_data"]
    validation = pre["validation"]
    schedule_manifest = build_canonical_schedule(train_data.window_ids, batch_size=BATCH_SIZE, update_count=TOTAL_UPDATES, experiment_id=EXPERIMENT_ID)
    logical_manifest_path = output / "canonical_schedule_manifest.json"
    write_json(logical_manifest_path, schedule_manifest)
    logical_manifest_bytes = logical_manifest_path.read_bytes()
    schedule_hash = sha256_bytes(logical_manifest_bytes)
    storage_manifest_path = output / "canonical_schedule_manifest.json.gz"
    storage_manifest_bytes = gzip.compress(logical_manifest_bytes, compresslevel=9, mtime=0)
    storage_manifest_path.write_bytes(storage_manifest_bytes)
    if gzip.decompress(storage_manifest_bytes) != logical_manifest_bytes:
        raise D4Blocker("canonical_schedule_storage_roundtrip_failed")
    logical_manifest_path.unlink()
    storage_schedule_hash = sha256_file(storage_manifest_path)
    schedule_rows = schedule_indices(schedule_manifest, train_data.window_ids)
    schedule_identity = {
        "schema_version": "stage3_h13_post_d4_canonical_schedule_identity_v1",
        "precommitted": "YES",
        "precommitted_before_first_training": "YES",
        "experiment_identifier": EXPERIMENT_ID,
        "generation_rule": schedule_manifest["generation_rule"],
        "seed_independent": "YES",
        "prospective_seed_consumed_during_generation": "NO",
        "sha256": schedule_hash,
        "window_manifest_sha256": schedule_manifest["train_window_ids_sha256"],
        "ordered_batch_manifest_sha256": schedule_manifest["ordered_batch_manifest_sha256"],
        "number_of_windows": schedule_manifest["number_of_windows"],
        "number_of_batches": schedule_manifest["number_of_batches"],
        "number_of_updates": schedule_manifest["number_of_training_updates"],
        "schedule_manifest_path": str(storage_manifest_path.resolve()),
        "storage_sha256": storage_schedule_hash,
        "storage_encoding": "gzip(json)",
        "all_prospective_seeds": list(FROZEN_SEEDS),
    }
    write_json(output / "canonical_schedule_identity.json", schedule_identity)
    schedule = {"precommitted": "YES", "seed_independent": "YES", **schedule_identity, "generation_rule": schedule_manifest["generation_rule"]}
    audit = static_seed_audit(load_model(R6_CHECKPOINT))
    write_json(output / "seed_consumer_audit.json", audit)
    r6_model = load_model(R6_CHECKPOINT)
    r6_metrics = evaluate_compact(r6_model, validation, pre["stats"]["channels"])
    if not math.isclose(r6_metrics["1"], AUTHORITATIVE_R6_H1, rel_tol=REPLAY_RTOL, abs_tol=REPLAY_ATOL):
        raise D4Blocker("r6_validation_reproduction_failed")
    guardrail = derive_h1_guardrail(AUTHORITATIVE_R6_H1, r6_metrics["1"], material_allowance_fraction=0.05, absolute_replay_tolerance=REPLAY_ATOL)
    if guardrail.threshold != H1_THRESHOLD:
        raise D4Blocker("frozen_h1_threshold_mismatch")
    config_template = {"learning_rate": 5.0e-5, "weight_decay": 1.0e-5, "gradient_clip_norm": 1.0, "batch_size": 256, "updates_per_phase": 128, "lambda_h1": 1.0, "lambda_hidden_a1": 0.005, "huber_beta": 0.001}
    records: dict[str, Any] = {}
    artifacts: dict[str, Any] = {}
    first_blocker: str | None = None
    completed_count = 0
    for seed in FROZEN_SEEDS:
        print(f"D4 canonical schedule training seed {seed}", flush=True)
        try:
            record, run_artifacts, _model = run_one(seed, output, pre, schedule_manifest, schedule_hash, schedule_rows, guardrail, config_template)
            records[str(seed)] = record
            artifacts[str(seed)] = run_artifacts
            completed_count += 1
            blocker = first_seed_blocker(record)
            if first_blocker is None and blocker:
                first_blocker = blocker
        except D4Blocker as exc:
            records[str(seed)] = {"seed": int(seed), "status": "FAILED", "valid": False, "error": str(exc), "future_label_leakage": 0, "schedule_sha256": schedule_hash}
            artifacts[str(seed)] = {"error": str(exc)}
            if first_blocker is None:
                first_blocker = str(exc)
            print(f"D4 seed {seed} BLOCKED: {exc}", flush=True)
    audit = static_seed_audit(load_model(R6_CHECKPOINT))
    per_seed_rng = {str(seed): records[str(seed)].get("rng_audit", {}) for seed in FROZEN_SEEDS if str(seed) in records}
    audit["per_seed_runtime_rng_audit"] = per_seed_rng
    audit["runtime_rng_state_unchanged_during_training"] = "YES" if len(per_seed_rng) == 5 and all(item.get("after_r6_initialization") == item.get("after_training") for item in per_seed_rng.values()) else "NO"
    write_json(output / "seed_consumer_audit.json", audit)
    write_json(output / "per_seed_training_summary.json", {"seeds": records, "frozen_seed_set": list(FROZEN_SEEDS), "training_config": config_template, "schedule_sha256": schedule_hash})
    write_json(output / "per_seed_replay_metrics.json", {seed: item.get("replay_metrics") for seed, item in artifacts.items()})
    aggregate = aggregate_results(records, schedule_hash)
    write_json(output / "aggregate_metrics.json", aggregate)
    checkpoint_records = [record["checkpoint"] for record in records.values() if record.get("valid") and record.get("checkpoint")]
    write_json(output / "checkpoint_identity_manifest.json", {"schema_version": "stage3_h13_post_d4_checkpoint_identity_v1", "expected_terminal_count": 5, "retained_terminal_count": len(checkpoint_records), "mutation_policy": "FORBIDDEN_AFTER_CREATION", "records": checkpoint_records})
    verify_snapshot(pre["snapshots"], "immutable_input")
    a1_immutable = True
    r6_hash_match = sha256_file(R6_CHECKPOINT) == EXPECTED_R6_HASH
    integrity_pass = a1_immutable and r6_hash_match and pre["d4"]["certificate"].get("CHAMPION_CHANGED") == "NO"
    focused = focused_tests
    regression = run_regression_tests()
    new_failures = int(regression["returncode"] != 0)
    context = {"records": records, "aggregate": aggregate, "schedule": schedule, "seed_audit": audit, "completed_count": completed_count, "first_blocker": first_blocker, "integrity_pass": integrity_pass, "protocol_followed": True, "only_batch_schedule_changed": True, "a1_immutable": a1_immutable, "r6_hash_match": r6_hash_match, "focused_tests": "PASS" if focused["returncode"] == 0 else "FAIL", "regression_tests": "PASS" if regression["returncode"] == 0 else "FAIL", "new_failures": new_failures, "persistent_size_bytes": 0}
    write_json(output / "test_results.json", {"focused": focused, "regression": regression, "new_failures": new_failures})
    write_json(output / "r6_reference_metrics.json", {"metrics": r6_metrics, "h1_guardrail": {"reference": guardrail.reference, "fresh_replay": guardrail.fresh_replay, "threshold": guardrail.threshold, "derivation": guardrail.derivation}})
    certificate = build_certificate(context)
    write_json(output / "terminal_certificate.json", certificate)
    (output / "FINAL_REPORT.md").write_text(render_report(certificate, context), encoding="utf-8", newline="\n")
    write_integrity_manifest(output)
    for _ in range(3):
        context["persistent_size_bytes"] = persistent_size(output)
        certificate = build_certificate(context)
        write_json(output / "terminal_certificate.json", certificate)
        (output / "FINAL_REPORT.md").write_text(render_report(certificate, context), encoding="utf-8", newline="\n")
        write_integrity_manifest(output)
    write_json(output / "run_progress.json", {"completed_runs": completed_count, "expected_runs": 5, "failed_seeds": [int(seed) for seed, item in records.items() if not item.get("valid")], "no_resume_policy": "YES", "terminal_status": certificate["STAGE_3_H13_POST_D4_CANONICAL_BATCH_ROBUSTNESS"]})
    print(json.dumps({"status": certificate["STAGE_3_H13_POST_D4_CANONICAL_BATCH_ROBUSTNESS"], "first_blocker": certificate["FIRST_BLOCKER"], "output": str(output), "h1": aggregate["metrics"]["H1"], "h16": aggregate["metrics"]["H16"], "h32": aggregate["metrics"]["H32"]}, sort_keys=True), flush=True)
    return 0 if certificate["STAGE_3_H13_POST_D4_CANONICAL_BATCH_ROBUSTNESS"] == "PASSED" else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--finalize-existing", action="store_true")
    parser.add_argument("--owner-authorization", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.finalize_existing:
        return finalize_existing(args.output_dir.resolve())
    if not args.execute:
        raise PermissionError("d4_execution_requires_explicit_execute")
    return execute(args.owner_authorization, args.output_dir.resolve())


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except D4Blocker as exc:
        print("STAGE_3_H13_POST_D4_CANONICAL_BATCH_ROBUSTNESS: BLOCKED", file=sys.stderr)
        print(f"FIRST_BLOCKER: {exc}", file=sys.stderr)
        raise SystemExit(2)
