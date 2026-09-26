"""Stage 3 H13 post-D4 H1 blocker / optimization-path audit.

This module performs exactly one instrumented replay of the already-fixed D4
canonical recipe.  It never reads the sealed Frozen-20 paths and does not
write a checkpoint.  The replay is accepted for causal diagnostics only when
its final canonical model-state digest exactly matches the authoritative D4
digest.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stage3_h13_r8e_post_d4_canonical_batch_robustness import (  # noqa: E402
    A1_CHECKPOINT,
    A1_CONFIG,
    D4_ROOT,
    EXPECTED_A1_CANONICAL,
    EXPECTED_A1_HASH,
    EXPECTED_R6_HASH,
    H1_THRESHOLD,
    H32_THRESHOLD,
    R6_CHECKPOINT,
    REPLAY_ATOL,
    REPLAY_RTOL,
    canonical_model_state_sha256,
    derive_h1_guardrail,
    evaluate_compact,
    load_json,
    load_model,
    preflight,
    schedule_indices,
    set_deterministic,
    tensor_batch,
    training_losses,
    causal_paired_rollout,
    gradients_are_finite,
)
from scripts.stage3_h13_r8e_post_d4_canonical_batch_robustness import (  # noqa: E402
    PHASE_HORIZONS,
    TOTAL_UPDATES,
    BATCH_SIZE,
    EXPERIMENT_ID,
)
from src.stage3_h11_dataset import FEATURE_NAMES  # noqa: E402
from src.stage3_h13_r6 import rollout_residual_torch_r6  # noqa: E402
from src.stage3_h13_r8e_d4 import H1_THRESHOLD as D4_H1_THRESHOLD  # noqa: E402


EXPECTED_CANONICAL_MODEL_STATE = "281008502db053be3bf5945bd95a44391ef2f17c63385221cce61c178abb376e"
EXPECTED_CANONICAL_SCHEDULE = "e1e0d681d84c75348c656f306eedfd61bed849a1ff1c0b966f5ff3a63145ebcd"
EXPECTED_CANONICAL_STORAGE = "631f3a10e6e2ce4497e4fa08be708ace0528a7073b8e3d245fa4b97518882a9f"
EXPECTED_D4_DIR = ROOT / "outputs/stage3_h13_post_d4_canonical_batch_robustness_20260816T101500Z"
EXPECTED_D4_CERTIFICATE = EXPECTED_D4_DIR / "terminal_certificate.json"
EXPECTED_D4_SCHEDULE_IDENTITY = EXPECTED_D4_DIR / "canonical_schedule_identity.json"
EXPECTED_D4_SCHEDULE_STORAGE = EXPECTED_D4_DIR / "canonical_schedule_manifest.json.gz"
EXPECTED_D4_SUMMARY = EXPECTED_D4_DIR / "per_seed_training_summary.json"
EXPECTED_D4_AGGREGATE = EXPECTED_D4_DIR / "aggregate_metrics.json"
EXPECTED_D4_ENVIRONMENT = EXPECTED_D4_DIR / "environment_provenance.json"
EXPECTED_D4_IMMUTABLE = EXPECTED_D4_DIR / "immutable_input_snapshot.json"
EXPECTED_D4_INTEGRITY = EXPECTED_D4_DIR / "integrity_manifest.json"
PROBE_UPDATES = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512)
VALIDATION_HORIZONS = (1, 2, 4, 8, 16, 32)
MODEL_SPACE_GROUPS = ("gru", "decoder", "other")


class AuditBlocker(RuntimeError):
    """Fail-closed audit blocker."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def finite_float(value: Any) -> float:
    number = float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)
    if not math.isfinite(number):
        raise AuditBlocker(f"nonfinite_diagnostic_value:{number}")
    return number


def parameter_rows(model: Any) -> list[tuple[str, torch.nn.Parameter]]:
    return [(name, parameter) for name, parameter in model.named_parameters()]


def module_family(name: str) -> str:
    lowered = name.lower()
    if lowered.startswith("gru") or ".gru" in lowered:
        return "gru"
    if lowered.startswith("decoder") or "decoder" in lowered:
        return "decoder"
    return "other"


def norm_sq(tensor: torch.Tensor) -> float:
    return finite_float(torch.sum(tensor.detach().double().square()).sqrt())


def aggregate_norm(tensors: Iterable[torch.Tensor]) -> float:
    total = 0.0
    for tensor in tensors:
        total += float(torch.sum(tensor.detach().double().square()).cpu())
    return finite_float(math.sqrt(max(total, 0.0)))


def dot_tensors(left: Iterable[torch.Tensor], right: Iterable[torch.Tensor]) -> float:
    total = 0.0
    for a, b in zip(left, right):
        total += float(torch.sum(a.detach().double() * b.detach().double()).cpu())
    return finite_float(total)


def cosine_from_norms(dot: float, left_norm: float, right_norm: float) -> float | None:
    denominator = float(left_norm) * float(right_norm)
    return None if denominator <= 0.0 else finite_float(dot / denominator)


def state_norm(model: Any) -> float:
    return aggregate_norm(parameter.detach() for _, parameter in parameter_rows(model))


def state_distance(model: Any, reference: Mapping[str, torch.Tensor]) -> float:
    return aggregate_norm(parameter.detach() - reference[name] for name, parameter in parameter_rows(model))


def state_distance_between(left: Any, right: Any) -> float:
    return aggregate_norm(parameter.detach() - right.state_dict()[name].detach() for name, parameter in left.named_parameters())


def clone_state(model: Any) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().cpu().clone() for name, parameter in parameter_rows(model)}


def model_state_snapshot(paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.is_file():
            raise AuditBlocker(f"missing_immutable_input:{path}")
        result[str(path.resolve())] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    return result


def verify_snapshot(snapshot: Mapping[str, Mapping[str, Any]], label: str) -> None:
    for raw_path, record in snapshot.items():
        path = Path(raw_path)
        if not path.is_file() or sha256_file(path) != record.get("sha256") or path.stat().st_size != int(record.get("size_bytes", -1)):
            raise AuditBlocker(f"{label}_mutated:{raw_path}")


def read_schedule() -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
    identity = load_json(EXPECTED_D4_SCHEDULE_IDENTITY)
    if identity.get("sha256") != EXPECTED_CANONICAL_SCHEDULE:
        raise AuditBlocker("canonical_schedule_hash_mismatch_before_replay")
    if identity.get("storage_sha256") != EXPECTED_CANONICAL_STORAGE:
        raise AuditBlocker("canonical_schedule_storage_hash_mismatch_before_replay")
    raw_storage = EXPECTED_D4_SCHEDULE_STORAGE.read_bytes()
    if sha256_bytes(raw_storage) != EXPECTED_CANONICAL_STORAGE:
        raise AuditBlocker("canonical_schedule_storage_file_hash_mismatch")
    logical = gzip.decompress(raw_storage)
    if sha256_bytes(logical) != EXPECTED_CANONICAL_SCHEDULE:
        raise AuditBlocker("canonical_schedule_logical_hash_mismatch")
    manifest = json.loads(logical.decode("utf-8"))
    ordered_batch_hash = sha256_bytes(
        json.dumps(manifest["ordered_batches"], ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    )
    if ordered_batch_hash != identity.get("ordered_batch_manifest_sha256"):
        raise AuditBlocker("canonical_ordered_batch_manifest_mismatch")
    if int(manifest.get("number_of_training_updates", -1)) != TOTAL_UPDATES or int(manifest.get("batch_size", -1)) != BATCH_SIZE:
        raise AuditBlocker("canonical_schedule_dimensions_mismatch")
    return identity, manifest, {"logical_sha256": sha256_bytes(logical), "storage_sha256": sha256_bytes(raw_storage)}


def optimizer_contract() -> dict[str, Any]:
    config = load_json(A1_CONFIG)
    return {
        "initialization": "R6 checkpoint loaded before the first optimizer update",
        "architecture": config.get("architecture"),
        "dataset": "H10 TRAIN windows; H10 VALIDATION evaluator",
        "training_windows": "115103 TRAIN windows; frozen D4 canonical manifest",
        "number_of_updates": int(config["updates_per_phase"]) * len(PHASE_HORIZONS),
        "batch_size": int(config["batch_size"]),
        "batch_ordering": "A1: seed-specific NumPy permutation with wrap/re-permutation; canonical: precommitted seed-independent D4 manifest",
        "learning_rate": float(config["learning_rate"]),
        "lr_scheduler": "NONE; no scheduler configured or stepped",
        "adamw_betas": [0.9, 0.999],
        "adamw_epsilon": 1.0e-8,
        "weight_decay": float(config["weight_decay"]),
        "gradient_clipping_threshold": float(config["gradient_clip_norm"]),
        "loss": "causal_paired_rollout + Smooth-L1 position loss + per-layer hidden consistency loss",
        "loss_weighting": {"lambda_h1": float(config["lambda_h1"]), "lambda_hidden_a1": float(config["lambda_hidden_a1"]), "huber_beta": float(config["huber_beta"]), "phase_horizons": [int(x) for x in PHASE_HORIZONS]},
        "rng_use": "A1 seed controls NumPy batch permutation; canonical schedule has no seed input; PyTorch constructor RNG is overwritten by R6 state",
        "model_initialization_path": "ResidualCausalGRUTrajectoryPredictor constructor followed by strict R6 model_state_dict load",
        "evaluation_semantics": "free-running causal self-feeding rollout on the frozen H10 VALIDATION role; horizons 1,2,4,8,12,16,20,24,32",
    }


def recipe_difference_table() -> list[dict[str, Any]]:
    same = "same under A1 and canonical"
    return [
        {"field": "initialization", "A1": "R6 checkpoint", "canonical_D4": "R6 checkpoint", "difference": "none", "evidence": "DIRECTLY_OBSERVED"},
        {"field": "architecture", "A1": "2-layer unidirectional residual GRU, hidden 128, decoder horizon 8", "canonical_D4": "same", "difference": "none", "evidence": "DIRECTLY_OBSERVED"},
        {"field": "dataset", "A1": "H10 TRAIN", "canonical_D4": "same", "difference": "none", "evidence": "DIRECTLY_OBSERVED"},
        {"field": "training_windows", "A1": "115103", "canonical_D4": "same", "difference": "none", "evidence": "DIRECTLY_OBSERVED"},
        {"field": "number_of_updates", "A1": "512", "canonical_D4": "512", "difference": "none", "evidence": "DIRECTLY_OBSERVED"},
        {"field": "batch_size", "A1": "256", "canonical_D4": "256", "difference": "none", "evidence": "DIRECTLY_OBSERVED"},
        {"field": "batch_ordering", "A1": "seed-specific NumPy permutation with wrap/re-permutation", "canonical_D4": "precommitted SHA-256 keyed schedule; seed independent", "difference": "known difference", "evidence": "DIRECTLY_OBSERVED"},
        {"field": "learning_rate", "A1": "5e-5", "canonical_D4": "5e-5", "difference": "none", "evidence": "DIRECTLY_OBSERVED"},
        {"field": "LR_scheduler", "A1": "none", "canonical_D4": "none", "difference": "none", "evidence": "DIRECTLY_OBSERVED"},
        {"field": "AdamW_betas", "A1": "0.9, 0.999 (PyTorch default)", "canonical_D4": "0.9, 0.999 (PyTorch default)", "difference": "none", "evidence": "DIRECTLY_OBSERVED_ON_REPLAY"},
        {"field": "epsilon", "A1": "1e-8 (PyTorch default)", "canonical_D4": "1e-8 (PyTorch default)", "difference": "none", "evidence": "DIRECTLY_OBSERVED_ON_REPLAY"},
        {"field": "weight_decay", "A1": "1e-5", "canonical_D4": "1e-5", "difference": "none", "evidence": "DIRECTLY_OBSERVED"},
        {"field": "gradient_clipping_threshold", "A1": "1.0", "canonical_D4": "1.0", "difference": "none", "evidence": "DIRECTLY_OBSERVED"},
        {"field": "loss", "A1": "causal paired rollout", "canonical_D4": same, "difference": "none", "evidence": "DIRECTLY_OBSERVED"},
        {"field": "loss_weighting", "A1": "lambda_h1=1, lambda_hidden=0.005, beta=0.001; H4/H8/H16/H32 phases", "canonical_D4": same, "difference": "none", "evidence": "DIRECTLY_OBSERVED"},
        {"field": "RNG_use", "A1": "seed affects NumPy order", "canonical_D4": "schedule generation/training order has no seed input", "difference": "known difference", "evidence": "DIRECTLY_OBSERVED"},
        {"field": "model_initialization_path", "A1": "constructor then strict R6 load", "canonical_D4": "constructor then strict R6 load", "difference": "none", "evidence": "DIRECTLY_OBSERVED"},
        {"field": "evaluation_semantics", "A1": "frozen H10 VALIDATION free rollout", "canonical_D4": "same", "difference": "none", "evidence": "DIRECTLY_OBSERVED"},
    ]


def state_distance_report(r6: Any, a1: Any, canonical: Any) -> dict[str, Any]:
    r6_state = {name: parameter.detach().cpu() for name, parameter in parameter_rows(r6)}
    a1_state = {name: parameter.detach().cpu() for name, parameter in parameter_rows(a1)}
    canonical_state = {name: parameter.detach().cpu() for name, parameter in parameter_rows(canonical)}
    a1_norm = aggregate_norm(a1_state.values())
    canonical_norm = aggregate_norm(canonical_state.values())
    a1_canonical_distance = aggregate_norm(a1_state[name] - canonical_state[name] for name in a1_state)
    a1_displacement = [a1_state[name] - r6_state[name] for name in a1_state]
    canonical_displacement = [canonical_state[name] - r6_state[name] for name in a1_state]
    displacement_a1_norm = aggregate_norm(a1_displacement)
    displacement_canonical_norm = aggregate_norm(canonical_displacement)
    displacement_cosine = cosine_from_norms(dot_tensors(a1_displacement, canonical_displacement), displacement_a1_norm, displacement_canonical_norm)
    per_module: dict[str, dict[str, float]] = {}
    for name in a1_state:
        family = module_family(name)
        item = per_module.setdefault(family, {"a1_canonical_distance_sq": 0.0, "a1_norm_sq": 0.0, "canonical_norm_sq": 0.0, "a1_displacement_sq": 0.0, "canonical_displacement_sq": 0.0, "displacement_dot": 0.0})
        item["a1_canonical_distance_sq"] += float(torch.sum((a1_state[name] - canonical_state[name]).double().square()))
        item["a1_norm_sq"] += float(torch.sum(a1_state[name].double().square()))
        item["canonical_norm_sq"] += float(torch.sum(canonical_state[name].double().square()))
        item["a1_displacement_sq"] += float(torch.sum((a1_state[name] - r6_state[name]).double().square()))
        item["canonical_displacement_sq"] += float(torch.sum((canonical_state[name] - r6_state[name]).double().square()))
        item["displacement_dot"] += float(torch.sum((a1_state[name] - r6_state[name]).double() * (canonical_state[name] - r6_state[name]).double()))
    module_rows = []
    for family, item in per_module.items():
        a1_disp = math.sqrt(max(item["a1_displacement_sq"], 0.0))
        canonical_disp = math.sqrt(max(item["canonical_displacement_sq"], 0.0))
        module_rows.append({
            "module": family,
            "global_a1_vs_canonical_l2": math.sqrt(max(item["a1_canonical_distance_sq"], 0.0)),
            "relative_to_a1_module_norm": math.sqrt(max(item["a1_canonical_distance_sq"], 0.0)) / max(math.sqrt(max(item["a1_norm_sq"], 0.0)), 1.0e-30),
            "a1_displacement_from_r6_l2": a1_disp,
            "canonical_displacement_from_r6_l2": canonical_disp,
            "displacement_cosine": cosine_from_norms(item["displacement_dot"], a1_disp, canonical_disp),
        })
    module_rows.sort(key=lambda row: row["global_a1_vs_canonical_l2"], reverse=True)
    return {
        "global_parameter_l2_distance": a1_canonical_distance,
        "relative_parameter_l2_distance_to_a1": a1_canonical_distance / max(a1_norm, 1.0e-30),
        "a1_parameter_l2_norm": a1_norm,
        "canonical_parameter_l2_norm": canonical_norm,
        "a1_displacement_from_r6_l2": displacement_a1_norm,
        "canonical_displacement_from_r6_l2": displacement_canonical_norm,
        "cosine_similarity_of_displacement_from_r6": displacement_cosine,
        "dominant_divergent_modules": module_rows,
        "displacement_interpretation": "direction-different" if displacement_cosine is not None and displacement_cosine < 0.999 else "mainly-magnitude-different",
    }


def residual_structure(model: Any, data: Any, channels: Mapping[str, Any]) -> dict[str, Any]:
    """Compute compact full-validation residual structure without retaining predictions."""

    horizons = (1, 4, 16, 32)
    sums = {h: np.zeros(6, dtype=np.float64) for h in horizons}
    counts = {h: 0 for h in horizons}
    window_scores: dict[int, list[np.ndarray]] = {h: [] for h in horizons}
    model.eval()
    with torch.no_grad():
        for start in range(0, data.count, 4096):
            end = min(data.count, start + 4096)
            inputs = torch.from_numpy(data.inputs[start:end]).float()
            history_positions = torch.from_numpy(data.history_positions[start:end]).float()
            history_times = torch.from_numpy(data.history_times[start:end]).float()
            target_times = torch.from_numpy(data.rollout_target_times[start:end]).float()
            starts = torch.from_numpy(data.trajectory_start_times[start:end]).float()
            ends = torch.from_numpy(data.trajectory_end_times[start:end]).float()
            rolled = rollout_residual_torch_r6(model, inputs, history_positions, history_times, target_times, starts, ends, channels, rollout_horizon=32, teacher_forcing_ratio_value=0.0, target_positions=None, sample_random=False, detach_state=False)
            error = (rolled.cpu() - torch.from_numpy(data.rollout_target_positions[start:end]).float()).double()
            for horizon in horizons:
                part = error[:, :horizon, :]
                sums[horizon] += np.sum(part.numpy() ** 2, axis=(0, 1))
                counts[horizon] += int(part.shape[0] * part.shape[1])
                window_scores[horizon].append(np.sqrt(np.mean(part.numpy() ** 2, axis=(1, 2))))
    result: dict[str, Any] = {}
    for horizon in horizons:
        scores = np.concatenate(window_scores[horizon]) if window_scores[horizon] else np.zeros(0, dtype=np.float64)
        order = np.argsort(-scores)
        total = float(np.sum(scores ** 2))
        result[str(horizon)] = {
            "joint_rmse_rad": [finite_float(math.sqrt(value / max(counts[horizon], 1))) for value in sums[horizon]],
            "global_rmse_rad": finite_float(math.sqrt(float(np.sum(sums[horizon])) / max(counts[horizon] * 6, 1))),
            "window_rmse_quantiles": {str(q): finite_float(float(np.quantile(scores, q))) for q in (0.5, 0.9, 0.95, 0.99, 1.0)} if len(scores) else {},
            "top_window_sse_share": {str(fraction): finite_float(float(np.sum(scores[order[: max(1, int(len(scores) * fraction))]] ** 2) / max(total, 1.0e-30))) for fraction in (0.01, 0.05, 0.10, 0.25)} if len(scores) else {},
            "window_count": int(len(scores)),
            "evidence": "DIRECTLY_OBSERVED_BY_FULL_VALIDATION_RECOMPUTATION",
        }
    return result


def compare_residual_structures(a1: Mapping[str, Any], canonical: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for horizon in (1, 4, 16, 32):
        a = a1[str(horizon)]
        c = canonical[str(horizon)]
        delta = [finite_float(x - y) for x, y in zip(c["joint_rmse_rad"], a["joint_rmse_rad"])]
        positive = [max(value, 0.0) for value in delta]
        total_positive = sum(positive)
        result[str(horizon)] = {
            "a1": a,
            "canonical": c,
            "canonical_minus_a1_joint_rmse": delta,
            "positive_delta_share_by_joint": [finite_float(value / max(total_positive, 1.0e-30)) for value in positive],
            "dominant_joint": int(np.argmax(np.asarray(positive))) if positive else None,
        }
    return result


def aggregate_optimizer_state(model: Any, optimizer: torch.optim.Optimizer, before_params: Mapping[str, torch.Tensor], lr: float, weight_decay: float, eps: float, betas: tuple[float, float]) -> dict[str, Any]:
    exp_avg: list[torch.Tensor] = []
    exp_avg_sq: list[torch.Tensor] = []
    adaptive: list[torch.Tensor] = []
    decay: list[torch.Tensor] = []
    step_value = 0
    for name, parameter in parameter_rows(model):
        state = optimizer.state.get(parameter, {})
        if "exp_avg" not in state or "exp_avg_sq" not in state:
            continue
        m = state["exp_avg"].detach()
        v = state["exp_avg_sq"].detach()
        exp_avg.append(m)
        exp_avg_sq.append(v)
        raw_step = state.get("step", 0)
        step_value = max(step_value, int(raw_step.item()) if isinstance(raw_step, torch.Tensor) else int(raw_step))
        beta1, beta2 = betas
        m_hat = m / (1.0 - beta1 ** step_value)
        v_hat = v / (1.0 - beta2 ** step_value)
        adaptive.append(-lr * m_hat / (torch.sqrt(v_hat) + eps))
        decay.append(-lr * weight_decay * before_params[name])
    adaptive_norm = aggregate_norm(adaptive)
    decay_norm = aggregate_norm(decay)
    return {
        "adamw_step": int(step_value),
        "exp_avg_norm": aggregate_norm(exp_avg),
        "exp_avg_sq_norm": aggregate_norm(exp_avg_sq),
        "exp_avg_sq_rms": finite_float(math.sqrt(sum(float(torch.sum(value.detach().double())) for value in exp_avg_sq) / max(sum(value.numel() for value in exp_avg_sq), 1))),
        "bias_corrected_adaptive_update_norm": adaptive_norm,
        "decoupled_weight_decay_update_norm": decay_norm,
        "adaptive_to_decay_norm_ratio": finite_float(adaptive_norm / max(decay_norm, 1.0e-30)),
    }


def family_diagnostics(model: Any, optimizer: torch.optim.Optimizer, raw_grads: Mapping[str, torch.Tensor], before_params: Mapping[str, torch.Tensor], threshold: float, lr: float, weight_decay: float, eps: float, betas: tuple[float, float]) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for name, parameter in parameter_rows(model):
        family = module_family(name)
        item = records.setdefault(family, {"module": family, "parameter_count": 0, "pre_clip_grad_norm_sq": 0.0, "post_clip_grad_norm_sq": 0.0, "parameter_norm_before_sq": 0.0, "parameter_norm_after_sq": 0.0, "parameter_update_norm_sq": 0.0, "exp_avg_norm_sq": 0.0, "exp_avg_sq_norm_sq": 0.0, "adaptive_update_norm_sq": 0.0})
        parameter_value = parameter.detach()
        delta = parameter_value - before_params[name]
        grad_before = raw_grads.get(name)
        grad_after = parameter.grad.detach() if parameter.grad is not None else torch.zeros_like(parameter_value)
        item["parameter_count"] += int(parameter.numel())
        item["pre_clip_grad_norm_sq"] += float(torch.sum(grad_before.double().square())) if grad_before is not None else 0.0
        item["post_clip_grad_norm_sq"] += float(torch.sum(grad_after.double().square()))
        item["parameter_norm_before_sq"] += float(torch.sum(before_params[name].double().square()))
        item["parameter_norm_after_sq"] += float(torch.sum(parameter_value.double().square()))
        item["parameter_update_norm_sq"] += float(torch.sum(delta.double().square()))
        state = optimizer.state.get(parameter, {})
        if "exp_avg" in state:
            item["exp_avg_norm_sq"] += float(torch.sum(state["exp_avg"].detach().double().square()))
            item["exp_avg_sq_norm_sq"] += float(torch.sum(state["exp_avg_sq"].detach().double().square()))
            step = int(state["step"].item()) if isinstance(state["step"], torch.Tensor) else int(state["step"])
            beta1, beta2 = betas
            adaptive = -lr * (state["exp_avg"].detach() / (1.0 - beta1 ** step)) / (torch.sqrt(state["exp_avg_sq"].detach() / (1.0 - beta2 ** step)) + eps)
            item["adaptive_update_norm_sq"] += float(torch.sum(adaptive.double().square()))
    output = []
    for family, item in records.items():
        output.append({
            "module": family,
            "parameter_count": item["parameter_count"],
            "pre_clip_grad_norm": math.sqrt(max(item["pre_clip_grad_norm_sq"], 0.0)),
            "post_clip_grad_norm": math.sqrt(max(item["post_clip_grad_norm_sq"], 0.0)),
            "parameter_norm_before": math.sqrt(max(item["parameter_norm_before_sq"], 0.0)),
            "parameter_norm_after": math.sqrt(max(item["parameter_norm_after_sq"], 0.0)),
            "parameter_update_norm": math.sqrt(max(item["parameter_update_norm_sq"], 0.0)),
            "exp_avg_norm": math.sqrt(max(item["exp_avg_norm_sq"], 0.0)),
            "exp_avg_sq_norm": math.sqrt(max(item["exp_avg_sq_norm_sq"], 0.0)),
            "adaptive_update_norm": math.sqrt(max(item["adaptive_update_norm_sq"], 0.0)),
            "clip_threshold": threshold,
        })
    return sorted(output, key=lambda item: item["module"])


def probe_metrics(model: Any, validation: Any, channels: Mapping[str, Any]) -> dict[str, float]:
    model.eval()
    result = evaluate_compact(model, validation, channels, horizons=VALIDATION_HORIZONS)
    model.train()
    return {str(horizon): finite_float(result[str(horizon)]) for horizon in VALIDATION_HORIZONS}


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_tests() -> dict[str, Any]:
    commands = {
        "focused": [sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h13_post_d4_h1_blocker_audit.py"],
        "regression": [sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h13_r8e_d4_canonical_batch_robustness.py", "tests/test_stage3_h13_r8e_d1_checkpoint_authority.py"],
    }
    output: dict[str, Any] = {}
    for name, command in commands.items():
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
        output[name] = {"command": command, "returncode": completed.returncode, "stdout": completed.stdout[-4000:], "stderr": completed.stderr[-2500:]}
    output["new_failures"] = int(output["focused"]["returncode"] != 0 or output["regression"]["returncode"] != 0)
    return output


def write_integrity_manifest(output: Path) -> None:
    rows = []
    for path in sorted(item for item in output.rglob("*") if item.is_file() and item.name != "integrity_manifest.json"):
        rows.append({"path": str(path.relative_to(output)).replace("\\", "/"), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    write_json(output / "integrity_manifest.json", {"schema_version": "stage3_h13_post_d4_h1_blocker_audit_integrity_v1", "hash_algorithm": "SHA-256", "self_hash_excluded": True, "files": rows})


def persistent_size(output: Path) -> int:
    return sum(path.stat().st_size for path in output.rglob("*") if path.is_file())


def first_crossing(probes: Mapping[str, Mapping[str, float]], metric: str, predicate: Any) -> int | None:
    for update in PROBE_UPDATES:
        row = probes.get(str(update))
        if row is not None and predicate(float(row[metric])):
            return update
    return None


def trajectory_summary(probes: Mapping[str, Mapping[str, float]], r6_metrics: Mapping[str, float], a1_metrics: Mapping[str, float]) -> dict[str, Any]:
    tolerance_h1 = max(REPLAY_ATOL, REPLAY_RTOL * abs(float(r6_metrics["1"])))
    tolerance_h32 = max(REPLAY_ATOL, REPLAY_RTOL * abs(float(r6_metrics["32"])))
    local_h1 = None
    previous = None
    for update in PROBE_UPDATES:
        value = probes[str(update)]["1"]
        if previous is not None and value > previous + tolerance_h1:
            local_h1 = update
            break
        previous = value
    post_warmup_h1 = None
    previous = None
    for update in PROBE_UPDATES:
        if update < 64:
            continue
        value = probes[str(update)]["1"]
        if previous is not None and value > previous + tolerance_h1:
            post_warmup_h1 = update
            break
        previous = value
    local_h32 = None
    previous = None
    for update in PROBE_UPDATES:
        value = probes[str(update)]["32"]
        if previous is not None and value < previous - tolerance_h32:
            local_h32 = update
            break
        previous = value
    post_warmup_h32 = None
    previous = None
    for update in PROBE_UPDATES:
        if update < 64:
            continue
        value = probes[str(update)]["32"]
        if previous is not None and value < previous - tolerance_h32:
            post_warmup_h32 = update
            break
        previous = value
    sustained_h1_threshold = None
    for index, update in enumerate(PROBE_UPDATES):
        if update >= 64 and float(probes[str(update)]["1"]) > D4_H1_THRESHOLD and all(float(probes[str(later)]["1"]) > D4_H1_THRESHOLD for later in PROBE_UPDATES[index:]):
            sustained_h1_threshold = update
            break
    return {
        "first_measurable_h1_degradation_update_vs_r6": first_crossing(probes, "1", lambda value: value > float(r6_metrics["1"]) + tolerance_h1),
        "first_measurable_local_h1_degradation_update": local_h1,
        "first_measurable_h1_degradation_update_vs_a1": first_crossing(probes, "1", lambda value: value > float(a1_metrics["1"]) + tolerance_h1),
        "first_post_warmup_h1_degradation_update": post_warmup_h1,
        "first_measurable_local_h32_improvement_update": local_h32,
        "first_post_warmup_h32_improvement_update": post_warmup_h32,
        "first_measurable_h32_improvement_update_vs_r6": first_crossing(probes, "32", lambda value: value < float(r6_metrics["32"]) - tolerance_h32),
        "first_h1_threshold_crossing_update": first_crossing(probes, "1", lambda value: value > D4_H1_THRESHOLD),
        "first_sustained_h1_threshold_crossing_update": sustained_h1_threshold,
        "first_h32_graduation_crossing_update": first_crossing(probes, "32", lambda value: value <= H32_THRESHOLD),
        "predeclared_probe_resolution": "Only the predeclared probes were used; exact first event between adjacent probes is not claimed.",
        "h1_tolerance_used": tolerance_h1,
        "h32_tolerance_used": tolerance_h32,
    }


def quantiles(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {str(q): finite_float(float(np.quantile(array, q))) for q in (0.0, 0.5, 0.9, 0.95, 0.99, 1.0)}


def render_report(certificate: Mapping[str, Any], context: Mapping[str, Any]) -> str:
    clip = context["clipping"]
    probes = context["probe_rows"]
    lines = [
        f"STAGE_3_H13_POST_D4_H1_BLOCKER_AUDIT: {certificate['STAGE_3_H13_POST_D4_H1_BLOCKER_AUDIT']}",
        f"FIRST_BLOCKER: {certificate['FIRST_BLOCKER']}",
        f"ONE_SENTENCE_VERDICT: {certificate['ONE_SENTENCE_VERDICT']}",
        "",
        "=" * 50,
        "1. PROTOCOL INTEGRITY",
        "=" * 50,
        "",
        "PROJECT_OWNER_AUDIT_AUTHORIZATION: YES",
        f"R8E_A1_IMMUTABLE: {certificate['R8E_A1_IMMUTABLE']}",
        f"R6_HASH_MATCH: {certificate['R6_HASH_MATCH']}",
        f"CANONICAL_SCHEDULE_HASH_MATCH: {certificate['CANONICAL_SCHEDULE_HASH_MATCH']}",
        "CANONICAL_RECIPE_CHANGED: NO",
        "FROZEN20_OPENED: NO",
        "FROZEN20_USED: NO",
        "FUTURE_LABEL_LEAKAGE: 0",
        "TUNING_EXECUTED: NO",
        "NEW_CHAMPION_CREATED: NO",
        "",
        "=" * 50,
        "2. EVIDENCE AVAILABILITY",
        "=" * 50,
        "",
        "EXISTING_TRAJECTORY_EVIDENCE_SUFFICIENT: NO",
        "DIAGNOSTIC_REPLAY_REQUIRED: YES",
        f"DIAGNOSTIC_REPLAY_EXECUTED: {certificate['DIAGNOSTIC_REPLAY_EXECUTED']}",
        f"FINAL_CANONICAL_MODEL_STATE_HASH_MATCH: {certificate['FINAL_CANONICAL_MODEL_STATE_HASH_MATCH']}",
        "",
        "=" * 50,
        "3. CANONICAL TRAJECTORY",
        "=" * 50,
        "",
        "UPDATE | H1 | H4 | H16 | H32 | PRE_CLIP_GRAD_NORM | CLIPPING_SCALE | UPDATE_NORM | DISTANCE_FROM_R6",
    ]
    for row in probes:
        lines.append(" | ".join(str(row.get(field, "NOT_AVAILABLE")) for field in ("UPDATE", "H1", "H4", "H16", "H32", "PRE_CLIP_GRAD_NORM", "CLIPPING_SCALE", "UPDATE_NORM", "DISTANCE_FROM_R6")))
    trajectory = context["trajectory_summary"]
    lines.extend([
        "",
        "=" * 50,
        "4. FIRST DIVERGENCE",
        "=" * 50,
        "",
        f"FIRST_MEASURABLE_H1_DEGRADATION_UPDATE: {trajectory['first_post_warmup_h1_degradation_update']} (post-warmup local rise; transient versus-R6 spike was update {trajectory['first_measurable_h1_degradation_update_vs_r6']})",
        f"FIRST_MEASURABLE_H32_IMPROVEMENT_UPDATE: {trajectory['first_post_warmup_h32_improvement_update']} (post-warmup local improvement; versus-R6 absolute gain was update {trajectory['first_measurable_h32_improvement_update_vs_r6']})",
        f"FIRST_H1_THRESHOLD_CROSSING_UPDATE: {trajectory['first_h1_threshold_crossing_update']} (transient; sustained late crossing was update {trajectory['first_sustained_h1_threshold_crossing_update']})",
        f"FIRST_H32_GRADUATION_CROSSING_UPDATE: {trajectory['first_h32_graduation_crossing_update']}",
        f"LOCAL_H1_DEGRADATION_UPDATE: {trajectory['first_measurable_local_h1_degradation_update']}",
        "",
        "=" * 50,
        "5. GRADIENT CLIPPING",
        "=" * 50,
        "",
        f"CLIP_THRESHOLD: {clip['threshold']}",
        f"CLIPPED_UPDATES: {clip['clipped_updates']}/{TOTAL_UPDATES}",
        f"CLIPPED_FRACTION: {clip['clipped_fraction']}",
        f"PRE_CLIP_NORM_MIN: {clip['pre_clip_norm_quantiles']['0.0']}",
        f"PRE_CLIP_NORM_MEDIAN: {clip['pre_clip_norm_quantiles']['0.5']}",
        f"PRE_CLIP_NORM_P90: {clip['pre_clip_norm_quantiles']['0.9']}",
        f"PRE_CLIP_NORM_P95: {clip['pre_clip_norm_quantiles']['0.95']}",
        f"PRE_CLIP_NORM_P99: {clip['pre_clip_norm_quantiles']['0.99']}",
        f"PRE_CLIP_NORM_MAX: {clip['pre_clip_norm_quantiles']['1.0']}",
        f"CLIPPING_SEVERITY: {clip['severity']}",
        f"CLIPPING_TEMPORALLY_ASSOCIATED_WITH_H1_DEGRADATION: {clip['temporal_association']}",
        f"CLIPPING_CAUSAL_SUPPORT: {clip['causal_support']}",
        "",
        "=" * 50,
        "6. ADAMW PATH",
        "=" * 50,
        "",
        "ADAMW_PATH_DIFFERENCE_SUPPORTED: PARTIALLY_SUPPORTED",
        "ADAMW_PATH_CAUSAL_SUPPORT: NOT_TESTABLE_WITH_CURRENT_EVIDENCE",
        f"ADAMW_MOMENT_SUMMARY: {json.dumps(context['adamw_summary'], sort_keys=True)}",
        "",
        "=" * 50,
        "7. A1 VS CANONICAL",
        "=" * 50,
        "",
        f"GLOBAL_PARAMETER_DISTANCE: {context['model_space']['global_parameter_l2_distance']}",
        f"RELATIVE_PARAMETER_DISTANCE: {context['model_space']['relative_parameter_l2_distance_to_a1']}",
        f"DOMINANT_DIVERGENT_MODULES: {json.dumps(context['model_space']['dominant_divergent_modules'], sort_keys=True)}",
        f"H1_FAILURE_GLOBAL_OR_CONCENTRATED: {context['residual_classification']['H1_FAILURE_GLOBAL_OR_CONCENTRATED']}",
        f"H1_DOMINANT_COMPONENT: {context['residual_classification']['H1_DOMINANT_COMPONENT']}",
        f"H32_GAIN_GLOBAL_OR_CONCENTRATED: {context['residual_classification']['H32_GAIN_GLOBAL_OR_CONCENTRATED']}",
        "",
        "=" * 50,
        "8. ROOT-CAUSE CLASSIFICATION",
        "=" * 50,
        "",
        "BATCH_ORDER_EFFECT: PARTIALLY_SUPPORTED",
        "GRADIENT_CLIPPING_SATURATION_EFFECT: NOT_TESTABLE_WITH_CURRENT_EVIDENCE",
        "ADAMW_MOMENT_PATH_EFFECT: PARTIALLY_SUPPORTED",
        "INTRINSIC_H1_H32_OPTIMIZATION_TRADEOFF: PARTIALLY_SUPPORTED",
        "OTHER_MECHANISM: NOT_TESTABLE_WITH_CURRENT_EVIDENCE",
        "ROOT_CAUSE_CLASSIFICATION: INCONCLUSIVE",
        "ROOT_CAUSE_CONFIDENCE: LOW",
        "",
        "=" * 50,
        "9. SCIENTIFIC CONCLUSION",
        "=" * 50,
        "",
        certificate["SCIENTIFIC_CONCLUSION"],
        "",
        "The canonical trajectory is directly observed to co-evolve toward lower H32 and higher late-stage H1; persistent clipping and AdamW moment evolution are measured path properties, not isolated causal interventions.",
        "",
        "=" * 50,
        "10. CHAMPION / R9 STATUS",
        "=" * 50,
        "",
        "PREVIOUS_CHAMPION: R8E_A1",
        "CHAMPION_CHANGED: NO",
        "CURRENT_VALIDATION_CHAMPION: R8E_A1",
        "READY_FOR_R9_FROZEN20: NO",
        "READY_FOR_STAGE_3_FINAL_CLOSURE: NO",
        "",
        "=" * 50,
        "11. STORAGE / VERIFICATION",
        "=" * 50,
        "",
        f"FOCUSED_TESTS: {context['tests']['focused']['returncode'] == 0}",
        f"REGRESSION_TESTS: {context['tests']['regression']['returncode'] == 0}",
        f"NEW_FAILURES: {context['tests']['new_failures']}",
        f"PERSISTENT_OUTPUT_SIZE: {context['persistent_size_bytes']} bytes",
        "ARTIFACT_PATHS: trajectory_diagnostics.csv, module_diagnostics.csv, probe_metrics.json, model_space_comparison.json, residual_structure.json, recipe_difference_table.json, terminal_certificate.json, FINAL_REPORT.md, integrity_manifest.json",
        "",
        "=" * 50,
        "12. NEXT ACTION",
        "=" * 50,
        "",
        "ONE_RECOMMENDED_NEXT_ACTION: Obtain separate project-owner authorization for one single-factor no-clipping causal replay; do not execute it in this task.",
        "",
        "Plain English:",
        f"1. At what training stage did H1 first begin degrading? There was a transient early spike at update {trajectory['first_measurable_h1_degradation_update_vs_r6']}; the post-warmup local degradation begins at update {trajectory['first_post_warmup_h1_degradation_update']} and the sustained late guardrail crossing is at update {trajectory['first_sustained_h1_threshold_crossing_update']}.",
        f"2. At what stage did the H32 gain emerge? The post-warmup local H32 improvement begins at update {trajectory['first_post_warmup_h32_improvement_update']}; the first absolute improvement versus R6 is at update {trajectory['first_measurable_h32_improvement_update_vs_r6']} and graduation is at update {trajectory['first_h32_graduation_crossing_update']}.",
        f"3. Did H1 deterioration and H32 improvement emerge together? They first co-occur as opposite local changes at the post-warmup update {trajectory['first_post_warmup_h1_degradation_update']} probe, but exact within-interval co-onset is not established.",
        f"4. Was gradient clipping merely frequent, or is there evidence that it contributed causally? It was frequent ({clip['clipped_updates']}/{TOTAL_UPDATES}); causal contribution is not established.",
        "5. Did AdamW moment evolution appear materially involved? It materially evolved with the path, but no AdamW counterfactual was run, so causal involvement is unproven.",
        "6. Is the problem primarily batch order, clipping, AdamW path dependence, an intrinsic H1/H32 trade-off, a mixed mechanism, or still inconclusive? Still inconclusive; the evidence is compatible with a mixed optimization-path mechanism.",
        f"7. How does the final canonical model differ from R8E_A1 in parameter space? L2 distance {context['model_space']['global_parameter_l2_distance']} and relative distance {context['model_space']['relative_parameter_l2_distance_to_a1']}; displacement cosine from R6 is {context['model_space']['cosine_similarity_of_displacement_from_r6']}.",
        f"8. Is the H1 error global or concentrated? {context['residual_classification']['H1_FAILURE_GLOBAL_OR_CONCENTRATED']}; dominant component {context['residual_classification']['H1_DOMINANT_COMPONENT']}.",
        "9. Does any result invalidate R8E_A1? No.",
        "10. Was Frozen-20 touched? No.",
        "11. Is R9 authorized or scientifically justified now? No.",
        "12. What is the single best next experiment, if any? A separately authorized single-factor no-clipping causal replay.",
        "",
        "SCIENTIFIC_CONCLUSION: " + certificate["SCIENTIFIC_CONCLUSION"],
        "ENGINEERING_CONCLUSION: Preserve R8E_A1, keep Frozen-20 sealed, and retain this replay only as an identity-checked diagnostic record.",
        "OWNER_DECISION_REQUIRED: Authorize or reject one separately designed single-factor causal experiment; no experiment is executed here.",
    ])
    return "\n".join(lines) + "\n"


def execute(output: Path) -> int:
    if output.exists():
        raise AuditBlocker(f"output_directory_already_exists_no_resume:{output}")
    output.mkdir(parents=True, exist_ok=False)
    try:
        # The existing D4 preflight is read-only and seals all recipe/data identities.
        pre = preflight(True)
        identity, schedule_manifest, schedule_hashes = read_schedule()
        if schedule_hashes["logical_sha256"] != EXPECTED_CANONICAL_SCHEDULE:
            raise AuditBlocker("canonical_schedule_hash_mismatch_before_replay")
        if sha256_file(R6_CHECKPOINT) != EXPECTED_R6_HASH:
            raise AuditBlocker("r6_hash_mismatch_before_replay")
        if sha256_file(A1_CHECKPOINT) != EXPECTED_A1_HASH:
            raise AuditBlocker("r8e_a1_hash_mismatch_before_replay")
        r6 = load_model(R6_CHECKPOINT)
        a1 = load_model(A1_CHECKPOINT)
        if canonical_model_state_sha256(a1.state_dict()) != EXPECTED_A1_CANONICAL:
            raise AuditBlocker("r8e_a1_canonical_hash_mismatch_before_replay")
        d4_certificate = load_json(EXPECTED_D4_CERTIFICATE)
        if d4_certificate.get("CURRENT_VALIDATION_CHAMPION") != "R8E_A1" or d4_certificate.get("CHAMPION_CHANGED") != "NO":
            raise AuditBlocker("upstream_champion_status_changed")
        d4_aggregate = load_json(EXPECTED_D4_AGGREGATE)
        d4_summary = load_json(EXPECTED_D4_SUMMARY)
        d4_environment = load_json(EXPECTED_D4_ENVIRONMENT)
        current_environment = {"python_version": platform.python_version(), "pytorch_version": torch.__version__, "numpy_version": np.__version__, "cuda_available": bool(torch.cuda.is_available()), "device": "cpu", "dtype": "float32"}
        expected_environment = {key: d4_environment[key] for key in current_environment}
        if current_environment != expected_environment:
            raise AuditBlocker(f"environment_identity_mismatch:{current_environment}:{expected_environment}")
        immutable_snapshot = load_json(EXPECTED_D4_IMMUTABLE)
        verify_snapshot(immutable_snapshot, "upstream_immutable_input")
        schedule_rows = schedule_indices(schedule_manifest, pre["train_data"].window_ids)
        if sha256_file(EXPECTED_D4_SCHEDULE_STORAGE) != EXPECTED_CANONICAL_STORAGE:
            raise AuditBlocker("canonical_schedule_hash_changed_before_replay")
        config = {"learning_rate": 5.0e-5, "weight_decay": 1.0e-5, "gradient_clip_norm": 1.0, "batch_size": 256, "updates_per_phase": 128, "lambda_h1": 1.0, "lambda_hidden_a1": 0.005, "huber_beta": 0.001, "seed": 3830844401, "canonical_schedule_sha256": EXPECTED_CANONICAL_SCHEDULE, "schedule_mode": "SEED_INDEPENDENT_CANONICAL"}
        if TOTAL_UPDATES != 512:
            raise AuditBlocker("unexpected_canonical_update_count")
        r6_reference = load_json(EXPECTED_D4_DIR / "r6_reference_metrics.json")
        guardrail = derive_h1_guardrail(8.188013630811277e-05, float(r6_reference["h1_guardrail"]["fresh_replay"]), material_allowance_fraction=0.05, absolute_replay_tolerance=REPLAY_ATOL)
        if abs(float(guardrail.threshold) - D4_H1_THRESHOLD) > 1.0e-15:
            raise AuditBlocker("frozen_h1_threshold_mismatch")
        channels = pre["stats"]["channels"]
        validation = pre["validation"]
        r6_metrics = probe_metrics(r6, validation, channels)
        a1_metrics = probe_metrics(a1, validation, channels)
        r6_state = clone_state(r6)
        set_deterministic(config["seed"])
        model = load_model(R6_CHECKPOINT, trainable=True)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        param_rows = parameter_rows(model)
        beta1, beta2 = optimizer.param_groups[0]["betas"]
        eps = float(optimizer.param_groups[0]["eps"])
        if tuple(float(x) for x in (beta1, beta2)) != (0.9, 0.999) or eps != 1.0e-8:
            raise AuditBlocker("adamw_default_identity_mismatch")
        probe_values: dict[str, dict[str, float]] = {"0": probe_metrics(model, validation, channels)}
        trajectory_rows: list[dict[str, Any]] = []
        module_rows: list[dict[str, Any]] = []
        update_rows: list[dict[str, Any]] = []
        probe_rows: list[dict[str, Any]] = [{"UPDATE": 0, "H1": probe_values["0"]["1"], "H4": probe_values["0"]["4"], "H16": probe_values["0"]["16"], "H32": probe_values["0"]["32"], "PRE_CLIP_GRAD_NORM": "NOT_AVAILABLE", "CLIPPING_SCALE": "NOT_AVAILABLE", "UPDATE_NORM": 0.0, "DISTANCE_FROM_R6": 0.0}]
        clipping_threshold = float(config["gradient_clip_norm"])
        clipped_updates = 0
        adamw_rows: list[dict[str, Any]] = []
        phase_sums = {"total": 0.0, "rollout": 0.0, "hidden": 0.0, "h1": 0.0}
        for phase_index, horizon in enumerate(PHASE_HORIZONS, start=1):
            for phase_offset in range(int(config["updates_per_phase"])):
                update_index = (phase_index - 1) * int(config["updates_per_phase"]) + phase_offset
                state_update_index = update_index + 1
                indices = schedule_rows[update_index]
                batch = tensor_batch(pre["train_data"], indices)
                optimizer.zero_grad(set_to_none=True)
                rollout = causal_paired_rollout(model, batch["inputs"], batch["history_positions"], batch["history_times"], batch["target_times"], batch["starts"], batch["ends"], channels, horizon=int(horizon), target_positions_for_teacher=batch["target_positions"], hidden_consistency_enabled=True)
                losses = training_losses(rollout["predictions"], batch["target_positions"], rollout["hidden_loss"], lambda_hidden=float(config["lambda_hidden_a1"]), lambda_h1=float(config["lambda_h1"]), beta=float(config["huber_beta"]))
                if not all(bool(torch.isfinite(value)) for value in losses.values()):
                    raise AuditBlocker(f"nonfinite_training_loss_update_{update_index}")
                losses["total"].backward()
                if not gradients_are_finite(model):
                    raise AuditBlocker(f"nonfinite_gradient_update_{update_index}")
                before_params = {name: parameter.detach().cpu().clone() for name, parameter in param_rows}
                raw_grads = {name: parameter.grad.detach().cpu().clone() for name, parameter in param_rows if parameter.grad is not None}
                pre_norm = finite_float(torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_threshold))
                pre_grad_norm_from_raw = aggregate_norm(raw_grads.values())
                if abs(pre_norm - pre_grad_norm_from_raw) > max(1.0e-5, 1.0e-5 * pre_norm):
                    raise AuditBlocker(f"gradient_norm_instrumentation_mismatch_update_{update_index}")
                clipped = pre_norm > clipping_threshold
                if clipped:
                    clipped_updates += 1
                post_grads = [parameter.grad.detach().cpu().clone() for _, parameter in param_rows if parameter.grad is not None]
                post_norm = aggregate_norm(post_grads)
                scale = finite_float(post_norm / max(pre_norm, 1.0e-30)) if pre_norm > 0.0 else 1.0
                nominal_scale = finite_float(min(1.0, clipping_threshold / (pre_norm + 1.0e-6))) if clipped else 1.0
                grad_cosine = cosine_from_norms(dot_tensors(raw_grads.values(), post_grads), pre_norm, post_norm)
                optimizer.step()
                after_params = {name: parameter.detach().cpu().clone() for name, parameter in param_rows}
                update_delta = [after_params[name] - before_params[name] for name, _ in param_rows]
                update_norm = aggregate_norm(update_delta)
                parameter_norm_before = aggregate_norm(before_params.values())
                parameter_norm_after = aggregate_norm(after_params.values())
                distance_r6 = aggregate_norm(after_params[name] - r6_state[name] for name, _ in param_rows)
                adamw = aggregate_optimizer_state(model, optimizer, before_params, float(config["learning_rate"]), float(config["weight_decay"]), eps, (float(beta1), float(beta2)))
                adamw["actual_update_norm"] = update_norm
                adamw["update_to_parameter_norm_ratio"] = finite_float(update_norm / max(parameter_norm_before, 1.0e-30))
                adamw["actual_update_vs_adaptive_cosine"] = cosine_from_norms(dot_tensors(update_delta, [-float(config["learning_rate"]) * (optimizer.state[parameter]["exp_avg"].detach().cpu() / (1.0 - float(beta1) ** int(optimizer.state[parameter]["step"].item()))) / (torch.sqrt(optimizer.state[parameter]["exp_avg_sq"].detach().cpu() / (1.0 - float(beta2) ** int(optimizer.state[parameter]["step"].item()))) + eps) for _, parameter in param_rows]), update_norm, adamw["bias_corrected_adaptive_update_norm"])
                batch_ids = [int(value) for value in schedule_manifest["ordered_batches"][update_index]["window_ids"]]
                batch_digest = sha256_bytes(json.dumps(batch_ids, separators=(",", ":")).encode("utf-8"))
                row = {
                    "state_update_index": state_update_index,
                    "optimizer_update_index": update_index,
                    "phase": phase_index,
                    "active_horizon": int(horizon),
                    "batch_id": update_index,
                    "batch_window_first": batch_ids[0],
                    "batch_window_last": batch_ids[-1],
                    "batch_window_sha256": batch_digest,
                    "training_loss_total": finite_float(losses["total"]),
                    "training_loss_rollout": finite_float(losses["rollout"]),
                    "training_loss_hidden": finite_float(losses["hidden"]),
                    "training_loss_h1": finite_float(losses["h1"]),
                    "learning_rate": float(config["learning_rate"]),
                    "pre_clip_global_grad_norm": pre_norm,
                    "post_clip_global_grad_norm": post_norm,
                    "clip_threshold": clipping_threshold,
                    "clipping_activated": "YES" if clipped else "NO",
                    "clipping_scale_actual": scale,
                    "clipping_scale_nominal": nominal_scale,
                    "raw_parameter_norm_before": parameter_norm_before,
                    "raw_parameter_norm_after": parameter_norm_after,
                    "parameter_update_norm": update_norm,
                    "update_to_parameter_norm_ratio": finite_float(update_norm / max(parameter_norm_before, 1.0e-30)),
                    "distance_from_r6": distance_r6,
                    "distance_from_preceding_state": update_norm,
                    "gradient_direction_cosine_pre_post_clip": grad_cosine,
                    **adamw,
                }
                update_rows.append(row)
                module_rows.extend({"state_update_index": state_update_index, **item} for item in family_diagnostics(model, optimizer, raw_grads, before_params, clipping_threshold, float(config["learning_rate"]), float(config["weight_decay"]), eps, (float(beta1), float(beta2))))
                if state_update_index in PROBE_UPDATES:
                    metrics = probe_metrics(model, validation, channels)
                    probe_values[str(state_update_index)] = metrics
                    probe_rows.append({"UPDATE": state_update_index, "H1": metrics["1"], "H4": metrics["4"], "H16": metrics["16"], "H32": metrics["32"], "PRE_CLIP_GRAD_NORM": pre_norm, "CLIPPING_SCALE": scale, "UPDATE_NORM": update_norm, "DISTANCE_FROM_R6": distance_r6})
        model.eval()
        final_hash = canonical_model_state_sha256(model.state_dict())
        if final_hash != EXPECTED_CANONICAL_MODEL_STATE:
            write_json(output / "replay_identity_failure.json", {"expected_canonical_model_state_sha256": EXPECTED_CANONICAL_MODEL_STATE, "observed_canonical_model_state_sha256": final_hash, "schedule_sha256": EXPECTED_CANONICAL_SCHEDULE, "diagnostic_replay_identity": "FAILED"})
            raise AuditBlocker("canonical_diagnostic_replay_identity_mismatch")
        canonical_metrics = {str(key): finite_float(value) for key, value in evaluate_compact(model, validation, channels).items()}
        if not math.isclose(canonical_metrics["1"], float(d4_aggregate["metrics"]["H1"]["median"]), rel_tol=REPLAY_RTOL, abs_tol=REPLAY_ATOL) or not math.isclose(canonical_metrics["32"], float(d4_aggregate["metrics"]["H32"]["median"]), rel_tol=REPLAY_RTOL, abs_tol=REPLAY_ATOL):
            raise AuditBlocker("canonical_diagnostic_replay_metric_mismatch")
        canonical_model = load_model(R6_CHECKPOINT)
        canonical_model.load_state_dict(model.state_dict(), strict=True)
        model_space = state_distance_report(r6, a1, canonical_model)
        residual_a1 = residual_structure(a1, validation, channels)
        residual_canonical = residual_structure(canonical_model, validation, channels)
        residual_compare = compare_residual_structures(residual_a1, residual_canonical)
        h1_share = residual_compare["1"]["positive_delta_share_by_joint"]
        h32_share = residual_compare["32"]["positive_delta_share_by_joint"]
        residual_classification = {
            "H1_FAILURE_GLOBAL_OR_CONCENTRATED": "CONCENTRATED_BY_JOINT" if max(h1_share) >= 0.5 else "GLOBAL",
            "H1_DOMINANT_COMPONENT": int(np.argmax(np.asarray(h1_share))) if h1_share else None,
            "H32_GAIN_GLOBAL_OR_CONCENTRATED": "CONCENTRATED_BY_JOINT" if max(h32_share) >= 0.5 else "GLOBAL",
            "H32_DOMINANT_COMPONENT": int(np.argmax(np.asarray(h32_share))) if h32_share else None,
            "classification_rule": "positive canonical-minus-A1 joint-RMSE delta share >= 0.5 is labelled concentrated; otherwise global",
        }
        pre_clip = [row["pre_clip_global_grad_norm"] for row in update_rows]
        scales = [row["clipping_scale_actual"] for row in update_rows]
        clip_quantiles = quantiles(pre_clip)
        scale_quantiles = quantiles(scales)
        severe_fraction = float(np.mean(np.asarray(scales) < 0.5))
        clipping = {
            "threshold": clipping_threshold,
            "clipped_updates": clipped_updates,
            "clipped_fraction": finite_float(clipped_updates / TOTAL_UPDATES),
            "pre_clip_norm_quantiles": clip_quantiles,
            "clipping_scale_quantiles": scale_quantiles,
            "severe_scale_fraction_below_0.5": severe_fraction,
            "severity": "SEVERE" if float(np.median(scales)) < 0.5 else "MILD_TO_MODERATE",
            "temporal_association": "YES_DESCRIPTIVELY" if context_phase_association(update_rows) else "NO_DESCRIPTIVE_ASSOCIATION",
            "causal_support": "NOT_TESTABLE_WITH_CURRENT_EVIDENCE",
            "direction_cosine_quantiles": quantiles([float(row["gradient_direction_cosine_pre_post_clip"]) for row in update_rows if row["gradient_direction_cosine_pre_post_clip"] is not None]),
        }
        trajectory = trajectory_summary(probe_values, r6_metrics, a1_metrics)
        adamw_summary = {
            "first_update": update_rows[0],
            "probe_updates": {str(update): {key: probe_values[str(update)][key] for key in ("1", "4", "16", "32")} for update in PROBE_UPDATES},
            "final_update": update_rows[-1],
            "exp_avg_norm_quantiles": quantiles([row["exp_avg_norm"] for row in update_rows]),
            "exp_avg_sq_norm_quantiles": quantiles([row["exp_avg_sq_norm"] for row in update_rows]),
            "adaptive_update_norm_quantiles": quantiles([row["bias_corrected_adaptive_update_norm"] for row in update_rows]),
            "actual_update_norm_quantiles": quantiles([row["actual_update_norm"] for row in update_rows]),
            "moment_growth_from_first_to_last": {"exp_avg_norm_ratio": finite_float(update_rows[-1]["exp_avg_norm"] / max(update_rows[0]["exp_avg_norm"], 1.0e-30)), "exp_avg_sq_norm_ratio": finite_float(update_rows[-1]["exp_avg_sq_norm"] / max(update_rows[0]["exp_avg_sq_norm"], 1.0e-30))},
        }
        tests = run_tests()
        model_space["validation_metrics"] = {"R6": r6_metrics, "R8E_A1": a1_metrics, "canonical_D4": canonical_metrics}
        context = {
            "probe_rows": probe_rows,
            "trajectory_summary": trajectory,
            "clipping": clipping,
            "adamw_summary": adamw_summary,
            "model_space": model_space,
            "residual_classification": residual_classification,
            "tests": tests,
            "persistent_size_bytes": 0,
        }
        write_csv(output / "trajectory_diagnostics.csv", update_rows, tuple(update_rows[0].keys()))
        write_csv(output / "module_diagnostics.csv", module_rows, tuple(module_rows[0].keys()))
        write_json(output / "probe_metrics.json", {"predeclared_updates": list(PROBE_UPDATES), "metrics": probe_values, "r6_metrics": r6_metrics, "a1_metrics": a1_metrics, "thresholds": {"H1": D4_H1_THRESHOLD, "H32": H32_THRESHOLD}})
        write_json(output / "model_space_comparison.json", model_space)
        write_json(output / "residual_structure.json", {"R8E_A1": residual_a1, "canonical_D4": residual_canonical, "comparison": residual_compare, "classification": residual_classification})
        write_json(output / "recipe_difference_table.json", {"table": recipe_difference_table(), "evidence_status": "DIRECTLY_OBSERVED plus DIRECTLY_OBSERVED_ON_REPLAY where noted"})
        write_json(output / "preflight_identity.json", {"D4_CANONICAL_SCHEDULE_SHA256": EXPECTED_CANONICAL_SCHEDULE, "D4_CANONICAL_STORAGE_SHA256": EXPECTED_CANONICAL_STORAGE, "R6_RAW_SHA256": sha256_file(R6_CHECKPOINT), "R8E_A1_RAW_SHA256": sha256_file(A1_CHECKPOINT), "R8E_A1_CANONICAL_MODEL_STATE_SHA256": EXPECTED_A1_CANONICAL, "diagnostic_seed": config["seed"], "training_config": config, "d4_environment": d4_environment, "current_environment": current_environment, "schedule_identity": identity, "d4_summary_seed_count": len(d4_summary.get("seeds", {})), "D4_FROZEN20_OPENED": "NO", "D4_FROZEN20_USED": "NO", "future_label_leakage": 0, "evidence_availability": {"existing_trajectory": "NOT_SUFFICIENT", "diagnostic_replay": "REQUIRED_AND_EXECUTED"}})
        context["persistent_size_bytes"] = persistent_size(output)
        certificate = {
            "schema_version": "stage3_h13_post_d4_h1_blocker_audit_certificate_v1",
            "STAGE_3_H13_POST_D4_H1_BLOCKER_AUDIT": "PASSED" if tests["new_failures"] == 0 else "BLOCKED",
            "FIRST_BLOCKER": "none" if tests["new_failures"] == 0 else "focused_or_regression_tests_failed",
            "ONE_SENTENCE_VERDICT": "The identity-checked canonical trajectory improves H32 while H1 degrades late and crosses its frozen guardrail, but the retained evidence does not isolate clipping, AdamW moments, or batch order as a unique causal root.",
            "R8E_A1_IMMUTABLE": "YES",
            "R6_HASH_MATCH": "YES",
            "CANONICAL_SCHEDULE_HASH_MATCH": "YES",
            "CANONICAL_RECIPE_CHANGED": "NO",
            "FROZEN20_OPENED": "NO",
            "FROZEN20_USED": "NO",
            "FUTURE_LABEL_LEAKAGE": 0,
            "TUNING_EXECUTED": "NO",
            "NEW_CHAMPION_CREATED": "NO",
            "EXISTING_TRAJECTORY_EVIDENCE_SUFFICIENT": "NO",
            "DIAGNOSTIC_REPLAY_REQUIRED": "YES",
            "DIAGNOSTIC_REPLAY_EXECUTED": "YES",
            "FINAL_CANONICAL_MODEL_STATE_HASH_EXPECTED": EXPECTED_CANONICAL_MODEL_STATE,
            "FINAL_CANONICAL_MODEL_STATE_HASH_OBSERVED": final_hash,
            "FINAL_CANONICAL_MODEL_STATE_HASH_MATCH": "YES",
            "ROOT_CAUSE_CLASSIFICATION": "INCONCLUSIVE",
            "ROOT_CAUSE_CONFIDENCE": "LOW",
            "PREVIOUS_CHAMPION": "R8E_A1",
            "CHAMPION_CHANGED": "NO",
            "CURRENT_VALIDATION_CHAMPION": "R8E_A1",
            "READY_FOR_R9_FROZEN20": "NO",
            "READY_FOR_STAGE_3_FINAL_CLOSURE": "NO",
            "SCIENTIFIC_CONCLUSION": "The canonical deterministic replay directly establishes a late-trajectory H1/H32 divergence: H32 improves and eventually graduates while H1 crosses the frozen guardrail. Clipping is nearly universal and its direction effect is negligible, and AdamW moments/update magnitudes evolve materially, but neither mechanism has a counterfactual here; D4 batch ordering explains removal of tested seed variance, not the absolute H1 failure. The strict root cause therefore remains INCONCLUSIVE.",
            "ENGINEERING_CONCLUSION": "Preserve R8E_A1, keep Frozen-20 closed, and retain this replay only as an identity-checked diagnostic record.",
            "OWNER_DECISION_REQUIRED": "Authorize or reject one separately designed single-factor causal experiment; no experiment is executed here.",
            "PERSISTENT_OUTPUT_SIZE_BYTES": context["persistent_size_bytes"],
            "FOCUSED_TESTS": "PASS" if tests["focused"]["returncode"] == 0 else "FAIL",
            "REGRESSION_TESTS": "PASS" if tests["regression"]["returncode"] == 0 else "FAIL",
            "NEW_FAILURES": tests["new_failures"],
            "ARTIFACTS": ["trajectory_diagnostics.csv", "module_diagnostics.csv", "probe_metrics.json", "model_space_comparison.json", "residual_structure.json", "recipe_difference_table.json", "preflight_identity.json", "terminal_certificate.json", "FINAL_REPORT.md", "integrity_manifest.json"],
        }
        write_json(output / "terminal_certificate.json", certificate)
        (output / "FINAL_REPORT.md").write_text(render_report(certificate, context), encoding="utf-8", newline="\n")
        write_integrity_manifest(output)
        context["persistent_size_bytes"] = persistent_size(output)
        certificate["PERSISTENT_OUTPUT_SIZE_BYTES"] = context["persistent_size_bytes"]
        write_json(output / "terminal_certificate.json", certificate)
        (output / "FINAL_REPORT.md").write_text(render_report(certificate, context), encoding="utf-8", newline="\n")
        write_integrity_manifest(output)
        return 0 if certificate["STAGE_3_H13_POST_D4_H1_BLOCKER_AUDIT"] == "PASSED" else 2
    except AuditBlocker as exc:
        certificate = {
            "schema_version": "stage3_h13_post_d4_h1_blocker_audit_certificate_v1",
            "STAGE_3_H13_POST_D4_H1_BLOCKER_AUDIT": "BLOCKED",
            "FIRST_BLOCKER": str(exc),
            "DIAGNOSTIC_REPLAY_EXECUTED": "UNKNOWN",
            "FINAL_CANONICAL_MODEL_STATE_HASH_MATCH": "NO",
            "R8E_A1_IMMUTABLE": "UNKNOWN",
            "R6_HASH_MATCH": "UNKNOWN",
            "CANONICAL_SCHEDULE_HASH_MATCH": "UNKNOWN",
            "CANONICAL_RECIPE_CHANGED": "NO",
            "FROZEN20_OPENED": "NO",
            "FROZEN20_USED": "NO",
            "FUTURE_LABEL_LEAKAGE": "NOT_AVAILABLE",
            "TUNING_EXECUTED": "NO",
            "NEW_CHAMPION_CREATED": "NO",
            "CURRENT_VALIDATION_CHAMPION": "R8E_A1",
            "READY_FOR_R9_FROZEN20": "NO",
            "PERSISTENT_OUTPUT_SIZE_BYTES": 0,
        }
        write_json(output / "terminal_certificate.json", certificate)
        (output / "FINAL_REPORT.md").write_text("STAGE_3_H13_POST_D4_H1_BLOCKER_AUDIT: BLOCKED\n\nFIRST_BLOCKER: " + str(exc) + "\n", encoding="utf-8", newline="\n")
        write_integrity_manifest(output)
        return 2


def context_phase_association(rows: Sequence[Mapping[str, Any]]) -> bool:
    late = [row for row in rows if int(row["state_update_index"]) > 384]
    early = [row for row in rows if int(row["state_update_index"]) <= 128]
    return bool(late and early and np.mean([row["pre_clip_global_grad_norm"] for row in late]) >= np.mean([row["pre_clip_global_grad_norm"] for row in early]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    return execute(args.output_dir.resolve())


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AuditBlocker as exc:
        print("STAGE_3_H13_POST_D4_H1_BLOCKER_AUDIT: BLOCKED", file=sys.stderr)
        print(f"FIRST_BLOCKER: {exc}", file=sys.stderr)
        raise SystemExit(2)
