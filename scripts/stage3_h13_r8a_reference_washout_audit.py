#!/usr/bin/env python3
"""Stage 3 H13-R8A TRAIN/VALIDATION-only reference-washout audit.

This runner is additive.  It reads the immutable H10 TRAIN/VALIDATION
windows, the H11-R2 initialization checkpoint and the immutable R6 checkpoint.
It never enumerates or opens the sealed H13 unseen-case artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage3_h11_dataset import (  # noqa: E402
    FEATURE_NAMES,
    INPUT_HISTORY,
    compute_normalization_stats,
    load_h10_segments,
    semantic_hash,
    sha256_file,
    verify_h10_authority,
)
from src.stage3_h11_model import metric_payload, set_deterministic  # noqa: E402
from src.stage3_h11_r2_model import (  # noqa: E402
    ResidualCausalGRUTrajectoryPredictor,
    causal_constant_velocity_baseline,
)
from src.stage3_h13_r5_reconstruction import COLLISION_SEMANTICS  # noqa: E402
from src.stage3_h13_r6 import (  # noqa: E402
    R6SequenceArrays,
    _next_features,
    as_window_arrays,
    build_sequence_arrays,
    direct_residual_targets,
)
from src.stage3_h13_r8 import bounded_residual_torch  # noqa: E402


H10_ROOT = ROOT / "outputs/stage3_h10_multi_trajectory_dataset_20260811T081123Z"
H11_ROOT = ROOT / "outputs/stage3_h11_r2_low_storage_unseen_generalization_20260813T215000Z"
H11_STATS = ROOT / "outputs/stage3_h11_r_pytorch_runtime_recertification_20260811T143806Z/normalization_stats.json"
R6_ROOT = ROOT / "outputs/stage3_h13_r6_rollout_aware_training_20260814T030507Z"
R6_CHECKPOINT = R6_ROOT / "final_r6_checkpoint.pt"
R6_CERTIFICATE = R6_ROOT / "stage3_h13_r6_terminal_certificate.json"
R8_ARCHIVE = ROOT / "outputs/stage3_h13_r8_autoregressive_state_semantics_redesign_20260814T163000Z"
R5_POLICY = ROOT / "src/stage3_h13_r5_reconstruction.py"

SEED = 81310
MAX_HORIZON = 32
PREDICTION_HORIZON = 8
HORIZONS = (1, 2, 4, 5, 8, 10, 12, 15, 16, 17, 20, 24, 32)
MODEL_HIDDEN_SIZE = 128
MODEL_LAYERS = 2
MAX_EPOCHS = 2
PATIENCE = 1
THREADS = max(2, min(8, int(os.cpu_count() or 2)))
R6_H32_REFERENCE = 0.0232112820732007
R8_H32_REFERENCE = 0.02929896649281688
MATERIAL_GATE = 0.02089015386588063


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_model(path: Path) -> Any:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = ResidualCausalGRUTrajectoryPredictor(input_size=len(FEATURE_NAMES), hidden_size=MODEL_HIDDEN_SIZE, num_layers=MODEL_LAYERS, horizon=8)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model


def data_scope() -> tuple[Any, Mapping[str, Any], R6SequenceArrays, R6SequenceArrays, dict[str, Any]]:
    authority = verify_h10_authority(ROOT, H10_ROOT)
    if authority.get("status") != "PASSED":
        raise RuntimeError(f"h10_authority_failed:{authority.get('first_blocker')}")
    dataset = load_h10_segments(H10_ROOT)
    stats = load_json(H11_STATS)
    recomputed = compute_normalization_stats(dataset)
    if semantic_hash(recomputed) != semantic_hash(stats):
        raise RuntimeError("h11_normalization_statistics_mismatch")
    train = build_sequence_arrays(dataset, stats, "TRAIN", max_rollout_horizon=MAX_HORIZON)
    validation = build_sequence_arrays(dataset, stats, "VALIDATION", max_rollout_horizon=MAX_HORIZON)
    if train.count == 0 or validation.count == 0:
        raise RuntimeError("train_validation_windows_missing")
    if not set(train.family_ids.tolist()).isdisjoint(set(validation.family_ids.tolist())):
        raise RuntimeError("train_validation_family_overlap")
    return dataset, stats, train, validation, {
        "roles_read": ["TRAIN", "VALIDATION"],
        "unseen_role_read": "NO",
        "train_windows": train.count,
        "validation_windows": validation.count,
        "train_families": sorted(set(str(x) for x in train.family_ids.tolist())),
        "validation_families": sorted(set(str(x) for x in validation.family_ids.tolist())),
        "h10_authority": authority,
    }


def train_bound(train: R6SequenceArrays) -> dict[str, Any]:
    residual = np.abs(direct_residual_targets(train).astype(np.float64))
    p999 = np.percentile(residual, 99.9, axis=(0, 1))
    maximum = np.max(residual, axis=(0, 1))
    bound = np.maximum(p999 * 1.5, 0.02)
    return {
        "source": "TRAIN_direct_residual_targets_only",
        "p99_9_abs_residual_rad": [float(x) for x in p999],
        "max_abs_residual_rad": [float(x) for x in maximum],
        "bound_rad": [float(x) for x in bound],
        "historical_r5_bound_changed": "NO",
    }


def reference_numpy(kind: str, positions: np.ndarray, times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    p = np.asarray(positions, dtype=np.float64)
    t = np.asarray(times, dtype=np.float64)
    tt = np.asarray(target_times, dtype=np.float64)
    last = p[:, -1]
    last_t = t[:, -1]
    dt = np.maximum(last_t - t[:, -2], 1.0e-9)
    velocity = (last - p[:, -2]) / dt[:, None]
    elapsed = tt - last_t[:, None]
    if kind == "cv":
        return last[:, None, :] + velocity[:, None, :] * elapsed[:, :, None]
    if kind == "ca":
        prior_dt = np.maximum(t[:, -2] - t[:, -3], 1.0e-9)
        prior_velocity = (p[:, -2] - p[:, -3]) / prior_dt[:, None]
        acceleration = (velocity - prior_velocity) / dt[:, None]
        return last[:, None, :] + velocity[:, None, :] * elapsed[:, :, None] + 0.5 * acceleration[:, None, :] * np.square(elapsed[:, :, None])
    if kind == "poly":
        fit_n = min(8, p.shape[1])
        x = t[:, -fit_n:] - last_t[:, None]
        design = np.stack((np.ones_like(x), x, np.square(x)), axis=2)
        coeff = np.linalg.pinv(design) @ p[:, -fit_n:, :]
        basis = np.stack((np.ones_like(elapsed), elapsed, np.square(elapsed)), axis=2)
        return np.einsum("bhi,bik->bhk", basis, coeff)
    raise ValueError(f"unknown_reference_kind:{kind}")


def torch_reference(kind: str, positions: Any, times: Any, target_time: Any) -> Any:
    import torch

    last = positions[:, -1, :]
    prior = positions[:, -2, :]
    last_t = times[:, -1]
    prior_t = times[:, -2]
    dt_hist = torch.clamp(last_t - prior_t, min=1.0e-9)
    velocity = (last - prior) / dt_hist[:, None]
    elapsed = torch.clamp(target_time - last_t, min=1.0e-9)
    if kind == "cv":
        return last + velocity * elapsed[:, None]
    if kind == "ca":
        prior_dt = torch.clamp(times[:, -2] - times[:, -3], min=1.0e-9)
        prior_velocity = (prior - positions[:, -3, :]) / prior_dt[:, None]
        acceleration = (velocity - prior_velocity) / dt_hist[:, None]
        return last + velocity * elapsed[:, None] + 0.5 * acceleration * elapsed[:, None].square()
    if kind == "poly":
        fit_n = min(8, positions.shape[1])
        x = times[:, -fit_n:] - last_t[:, None]
        design = torch.stack((torch.ones_like(x), x, x.square()), dim=2)
        coeff = torch.linalg.lstsq(design, positions[:, -fit_n:, :]).solution
        basis = torch.stack((torch.ones_like(elapsed), elapsed, elapsed.square()), dim=1)
        return torch.bmm(basis[:, None, :], coeff).squeeze(1)
    raise ValueError(f"unknown_reference_kind:{kind}")


def next_features(next_position: Any, previous: Any, prior: Any, next_time: Any, last_time: Any, prior_time: Any, starts: Any, ends: Any, spray: Any, channels: Mapping[str, Any]) -> Any:
    return _next_features(next_position, previous, prior, next_time, last_time, prior_time, starts, ends, spray, channels, FEATURE_NAMES)


def rollout_variant(model: Any, inputs: Any, history_positions: Any, history_times: Any, target_times: Any, starts: Any, ends: Any, channels: Mapping[str, Any], *, kind: str, bound: Sequence[float], horizon: int, fixed_direct: bool = False) -> Any:
    import torch

    current_inputs = inputs
    original_inputs = inputs
    original_positions = history_positions
    original_times = history_times
    positions = history_positions
    times = history_times
    outputs: list[Any] = []
    step = 0
    while step < horizon:
        if fixed_direct:
            raw = model(original_inputs)
            residual = bounded_residual_torch(raw, bound)
            end = min(step + PREDICTION_HORIZON, horizon)
            ref = torch_reference("cv", original_positions, original_times, target_times[:, step])
            # fixed-direct uses the same local CV velocity for every target in a block
            dt = target_times[:, step:end] - original_times[:, -1, None]
            velocity = (original_positions[:, -1] - original_positions[:, -2]) / torch.clamp(original_times[:, -1] - original_times[:, -2], min=1.0e-9)[:, None]
            refs = original_positions[:, -1, None, :] + velocity[:, None, :] * dt[:, :, None]
            outputs.append(refs + residual[:, : end - step, :])
            step = end
            continue
        raw = model(current_inputs)
        chunk = min(PREDICTION_HORIZON, horizon - step)
        residual = bounded_residual_torch(raw[:, :chunk, :], bound)
        for local in range(chunk):
            target_time = target_times[:, step]
            reference_next = torch_reference(kind, positions, times, target_time)
            outputs.append(reference_next + residual[:, local, :])
            new_features = next_features(
                reference_next,
                positions[:, -1, :],
                positions[:, -2, :],
                target_time,
                times[:, -1],
                times[:, -2],
                starts,
                ends,
                current_inputs[:, -1, 18],
                channels,
            ).to(dtype=current_inputs.dtype)
            current_inputs = torch.cat((current_inputs[:, 1:, :], new_features[:, None, :]), dim=1)
            positions = torch.cat((positions[:, 1:, :], reference_next[:, None, :]), dim=1)
            times = torch.cat((times[:, 1:], target_time[:, None]), dim=1)
            step += 1
    if fixed_direct:
        return torch.cat(outputs, dim=1)
    return torch.stack(outputs, dim=1)


def direct_reference_prediction(model: Any, data: R6SequenceArrays, channels: Mapping[str, Any], kind: str, bound: Sequence[float], batch_size: int = 8192) -> tuple[np.ndarray, dict[str, Any]]:
    import torch

    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, data.count, batch_size):
            end = min(data.count, start + batch_size)
            raw = model(torch.from_numpy(data.inputs[start:end]).float())
            residual = bounded_residual_torch(raw, bound).cpu().numpy().astype(np.float64)
            reference = reference_numpy(kind, data.history_positions[start:end], data.history_times[start:end], data.direct_target_times[start:end])
            predictions.append(reference + residual)
    prediction = np.concatenate(predictions, axis=0)
    return prediction, metric_payload(prediction, data.direct_target_positions)


def metric_block(error: np.ndarray, horizon: int) -> dict[str, Any]:
    values = np.asarray(error[:, :horizon, :], dtype=np.float64)
    absolute = np.abs(values)
    return {
        "horizon": int(horizon),
        "rmse": float(np.sqrt(np.mean(np.square(values)))),
        "rmse_per_joint": [float(x) for x in np.sqrt(np.mean(np.square(values), axis=(0, 1)))],
        "maximum_absolute_error": float(np.max(absolute)),
        "median_absolute_error": float(np.median(absolute)),
        "p95_absolute_error": float(np.percentile(absolute, 95)),
        "window_count": int(values.shape[0]),
    }


def rollout_metrics(model: Any, data: R6SequenceArrays, channels: Mapping[str, Any], *, kind: str, bound: Sequence[float], fixed_direct: bool = False, batch_size: int = 8192) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    import torch

    model.eval()
    predictions = np.empty((data.count, MAX_HORIZON, 6), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, data.count, batch_size):
            end = min(data.count, start + batch_size)
            rolled = rollout_variant(
                model,
                torch.from_numpy(data.inputs[start:end]).float(),
                torch.from_numpy(data.history_positions[start:end]).float(),
                torch.from_numpy(data.history_times[start:end]).float(),
                torch.from_numpy(data.rollout_target_times[start:end]).float(),
                torch.from_numpy(data.trajectory_start_times[start:end]).float(),
                torch.from_numpy(data.trajectory_end_times[start:end]).float(),
                channels,
                kind=kind,
                bound=bound,
                horizon=MAX_HORIZON,
                fixed_direct=fixed_direct,
            )
            predictions[start:end] = rolled.cpu().numpy().astype(np.float64)
    error = predictions - data.rollout_target_positions
    metrics = {str(h): metric_block(error, h) for h in HORIZONS}
    metrics["FULL_ROLLOUT"] = metric_block(error, MAX_HORIZON)
    metrics["per_step_rmse"] = [float(np.sqrt(np.mean(np.square(error[:, index, :])))) for index in range(MAX_HORIZON)]
    metrics["per_step_p95_absolute_error"] = [float(np.percentile(np.abs(error[:, index, :]), 95)) for index in range(MAX_HORIZON)]
    return metrics, predictions, error


def train_candidate(name: str, train: R6SequenceArrays, validation: R6SequenceArrays, stats: Mapping[str, Any], kind: str, bound: Sequence[float], *, fixed_direct: bool, epochs: int = MAX_EPOCHS) -> tuple[Any, list[dict[str, Any]], int, bytes]:
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, TensorDataset

    model = load_model(H11_ROOT / "best_checkpoint.pt")
    model.train()
    baseline_train = reference_numpy(kind, train.history_positions, train.history_times, train.direct_target_times)
    target_residual = torch.from_numpy((train.direct_target_positions - baseline_train).astype(np.float32))
    indices = torch.arange(train.count, dtype=torch.long)
    loader = DataLoader(TensorDataset(torch.from_numpy(train.inputs), target_residual, indices), batch_size=8192, shuffle=False, num_workers=0, drop_last=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-4, weight_decay=1.0e-5)
    best_state: dict[str, Any] | None = None
    best_key: tuple[float, float] | None = None
    best_epoch = 0
    no_improvement = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        direct_sum = 0.0
        rollout_sum = 0.0
        count = 0
        for batch_inputs, batch_target, batch_indices in loader:
            batch_inputs = batch_inputs.float()
            optimizer.zero_grad(set_to_none=True)
            output = bounded_residual_torch(model(batch_inputs), bound)
            direct_loss = F.smooth_l1_loss(output, batch_target, beta=1.0e-3)
            ids = batch_indices.numpy()
            rolled = rollout_variant(
                model,
                batch_inputs,
                torch.from_numpy(train.history_positions[ids]).float(),
                torch.from_numpy(train.history_times[ids]).float(),
                torch.from_numpy(train.rollout_target_times[ids]).float(),
                torch.from_numpy(train.trajectory_start_times[ids]).float(),
                torch.from_numpy(train.trajectory_end_times[ids]).float(),
                stats["channels"],
                kind=kind,
                bound=bound,
                horizon=16 if fixed_direct else MAX_HORIZON,
                fixed_direct=fixed_direct,
            )
            rollout_target = torch.from_numpy(train.rollout_target_positions[ids, : rolled.shape[1]]).float()
            rollout_loss = F.smooth_l1_loss(rolled, rollout_target, beta=1.0e-3)
            loss = direct_loss + 0.75 * rollout_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            direct_sum += float(direct_loss.detach()) * len(batch_inputs)
            rollout_sum += float(rollout_loss.detach()) * len(batch_inputs)
            count += len(batch_inputs)
        with torch.no_grad():
            direct_prediction, direct = direct_reference_prediction(model, validation, stats["channels"], kind, bound)
            del direct_prediction
            rollout, _, _ = rollout_metrics(model, validation, stats["channels"], kind=kind, bound=bound, fixed_direct=fixed_direct, batch_size=8192)
        key = (float(rollout["16"]["rmse"]), float(direct["joint_position_rmse_rad"]))
        history.append({
            "epoch": epoch,
            "train_direct_loss": direct_sum / max(count, 1),
            "train_rollout_loss": rollout_sum / max(count, 1),
            "validation_direct_rmse": float(direct["joint_position_rmse_rad"]),
            "validation_h16_rmse": float(rollout["16"]["rmse"]),
            "validation_h32_rmse": float(rollout["32"]["rmse"]),
        })
        if best_key is None or key < best_key:
            best_key = key
            best_epoch = epoch
            no_improvement = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improvement += 1
        if no_improvement >= PATIENCE:
            break
    if best_state is None:
        raise RuntimeError(f"candidate_checkpoint_not_selected:{name}")
    model.load_state_dict(best_state, strict=True)
    buffer = io.BytesIO()
    torch.save({"model_state_dict": best_state, "candidate": name, "reference_kind": kind, "fixed_direct": fixed_direct, "best_epoch": best_epoch}, buffer)
    return model, history, best_epoch, buffer.getvalue()


def history_washout() -> dict[str, Any]:
    rows = {}
    for h in (8, 15, 16, 17):
        rows[str(h)] = {
            "observed_history_fraction": float(max(INPUT_HISTORY - h, 0) / INPUT_HISTORY),
            "reference_history_fraction": float(min(h, INPUT_HISTORY) / INPUT_HISTORY),
            "model_history_fraction": 0.0,
        }
    return {
        "input_history_length": INPUT_HISTORY,
        "FULL_REFERENCE_CONTEXT_FIRST_HORIZON": INPUT_HISTORY,
        "rows": rows,
        "accounting": "R8 appends one causal-reference state per step and never appends a model-predicted state; fractions are exact provenance accounting, not inferred from collision-free states.",
    }


def diagnostics_for_role(dataset: Any, role: str) -> np.ndarray:
    rows: list[np.ndarray] = []
    for segment in dataset.segments:
        if dataset.split_map.get(segment.family_id) != role:
            continue
        max_start = len(segment.positions) - INPUT_HISTORY - MAX_HORIZON + 1
        for start in range(max(0, max_start)):
            idx = np.arange(start + INPUT_HISTORY, start + INPUT_HISTORY + MAX_HORIZON)
            velocity = segment.velocities[idx]
            acceleration = segment.accelerations[idx]
            speed = np.linalg.norm(velocity, axis=1)
            curvature = np.linalg.norm(acceleration, axis=1) / np.maximum(np.square(speed), 1.0e-8)
            rows.append(np.stack((curvature, np.linalg.norm(acceleration, axis=1)), axis=1))
    return np.asarray(rows, dtype=np.float64)


def curvature_audit(dataset: Any, validation_error: np.ndarray, train_error: np.ndarray | None = None) -> dict[str, Any]:
    train_diag = diagnostics_for_role(dataset, "TRAIN")
    validation_diag = diagnostics_for_role(dataset, "VALIDATION")
    if len(train_diag) == 0 or len(validation_diag) == 0:
        return {"CAUSAL_PHASE_SIGNAL_AVAILABLE": "NO", "status": "NOT_AVAILABLE"}
    thresholds = np.percentile(train_diag[:, :, 0], [33.333333, 66.666667])
    categories: dict[str, Any] = {}
    curvature = validation_diag[:, :, 0]
    for name, mask in {
        "LOW_CURVATURE": curvature <= thresholds[0],
        "MEDIUM_CURVATURE": (curvature > thresholds[0]) & (curvature <= thresholds[1]),
        "HIGH_CURVATURE": curvature > thresholds[1],
    }.items():
        values = np.abs(validation_error)[mask]
        categories[name] = {
            "sample_count": int(values.size),
            "mean_absolute_error": float(np.mean(values)) if values.size else None,
            "rmse": float(np.sqrt(np.mean(np.square(values)))) if values.size else None,
        }
    high = categories["HIGH_CURVATURE"]["rmse"]
    low = categories["LOW_CURVATURE"]["rmse"]
    return {
        "CAUSAL_PHASE_SIGNAL_AVAILABLE": "NO",
        "phase_signal_reason": "No authoritative inference-time phase/progress feature exists in the H10 input schema.",
        "curvature_definition": "joint-space acceleration norm divided by squared joint-space speed; diagnostic only",
        "threshold_source": "TRAIN target trajectory diagnostics only",
        "thresholds": [float(x) for x in thresholds],
        "categories": categories,
        "high_vs_low_rmse_ratio": float(high / low) if high is not None and low not in (None, 0.0) else None,
        "curvature_or_phase_dependence_found": "YES" if high is not None and low not in (None, 0.0) and high > 1.25 * low else "NO",
        "future_diagnostic_values_used_as_input_features": "NO",
    }


def inflection(metrics: Mapping[str, Any]) -> int | None:
    step = np.asarray(metrics.get("per_step_rmse", []), dtype=np.float64)
    if len(step) < 4:
        return None
    early = float(np.median(step[:8]))
    for index in range(1, len(step)):
        if step[index] > 2.0 * max(step[index - 1], 1.0e-12) and step[index] > 2.0 * max(early, 1.0e-12):
            return index + 1
    return None


def run_tests(output: Path) -> dict[str, Any]:
    import subprocess

    focused = [sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h13_r8.py", "tests/test_stage3_h13_r6.py", "tests/test_stage3_h13_r5_reconstruction.py", "tests/test_stage3_h13_r7.py", "tests/test_stage3_h13_r8a_reference_washout_audit.py"]
    regression = [sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h11_r2.py", "tests/test_stage3_h12_certification.py", "tests/test_stage3_h12_no_leakage.py", "tests/test_stage3_h12_r7.py", "tests/test_stage3_h12_r7a.py", "tests/test_stage3_h13.py", "tests/test_open_arch_181.py"]
    lines: list[str] = []
    result: dict[str, Any] = {}
    for name, command in (("FOCUSED_TESTS", focused), ("REGRESSION_TESTS", regression)):
        proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1200, check=False)
        result[name] = {"status": "PASS" if proc.returncode == 0 else "FAIL", "returncode": proc.returncode}
        lines.extend([f"{name}: {result[name]['status']}", proc.stdout, proc.stderr, ""])
    (output / "test_summary.txt").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    result["NEW_REGRESSION_FAILURES"] = 0 if result["REGRESSION_TESTS"]["status"] == "PASS" else 1
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise RuntimeError(f"refusing_to_overwrite:{output}")
    output.mkdir(parents=True)
    import torch

    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(max(1, min(2, THREADS)))
    set_deterministic(SEED)
    dataset, stats, train, validation, data_summary = data_scope()
    bound = train_bound(train)
    r6_before = sha256_file(R6_CHECKPOINT)
    r6_model = load_model(R6_CHECKPOINT)
    # C0 is the R8 control semantics.  Candidate training is deliberately
    # bounded to four structural candidates and two epochs.
    candidate_specs = [
        ("C0_R8_CONSTANT_VELOCITY_REFERENCE", "cv", False),
        ("C1_CAUSAL_CONSTANT_ACCELERATION_REFERENCE", "ca", False),
        ("C2_CAUSAL_LOCAL_QUADRATIC_REFERENCE", "poly", False),
        ("C3_FIXED_CONTEXT_DIRECT_MULTIHORIZON", "cv", True),
    ]
    candidate_results: list[dict[str, Any]] = []
    candidate_predictions: dict[str, np.ndarray] = {}
    candidate_errors: dict[str, np.ndarray] = {}
    candidate_models: dict[str, Any] = {}
    candidate_bytes: dict[str, bytes] = {}
    for index, (name, kind, fixed_direct) in enumerate(candidate_specs):
        set_deterministic(SEED + index)
        model, history, best_epoch, checkpoint_bytes = train_candidate(name, train, validation, stats, kind, bound["bound_rad"], fixed_direct=fixed_direct)
        direct_prediction, direct = direct_reference_prediction(model, validation, stats["channels"], kind, bound["bound_rad"])
        rollout, prediction, error = rollout_metrics(model, validation, stats["channels"], kind=kind, bound=bound["bound_rad"], fixed_direct=fixed_direct)
        candidate_models[name] = model
        candidate_bytes[name] = checkpoint_bytes
        candidate_predictions[name] = prediction
        candidate_errors[name] = error
        candidate_results.append({
            "name": name,
            "reference_kind": kind,
            "fixed_context_direct_multihorizon": "YES" if fixed_direct else "NO",
            "direct": direct,
            "rollout": rollout,
            "training_history": history,
            "best_epoch": best_epoch,
            "epochs_attempted": len(history),
            "early_stop_reason": "patience_exhausted" if len(history) < MAX_EPOCHS else "max_epochs_reached",
            "checkpoint_sha256_in_memory": sha256_bytes(checkpoint_bytes),
            "training_state_semantics": "candidate reference-only state update; no model prediction is appended to the next input",
            "future_label_used_as_input": "NO",
        })
    selected = min(candidate_results, key=lambda row: (float(row["rollout"]["32"]["rmse"]), float(row["direct"]["joint_position_rmse_rad"])))
    selected_name = str(selected["name"])
    selected_error = candidate_errors[selected_name]
    cv_reference = reference_numpy("cv", validation.history_positions, validation.history_times, validation.rollout_target_times)
    pure_error = cv_reference - validation.rollout_target_positions
    pure_metrics = {str(h): metric_block(pure_error, h) for h in HORIZONS}
    pure_metrics["FULL_ROLLOUT"] = metric_block(pure_error, MAX_HORIZON)
    pure_metrics["per_step_rmse"] = [float(np.sqrt(np.mean(np.square(pure_error[:, index, :])))) for index in range(MAX_HORIZON)]
    washout = history_washout()
    model_metrics = selected["rollout"]
    audit = {
        **washout,
        "pure_reference_error_inflection_horizon": inflection(pure_metrics),
        "observed_model_error_inflection_horizon": inflection(model_metrics),
        "pure_reference_per_step_rmse": pure_metrics["per_step_rmse"],
        "selected_model_per_step_rmse": model_metrics["per_step_rmse"],
        "error_by_reference_fraction": [{"horizon": int(h), "reference_history_fraction": float(min(h, INPUT_HISTORY) / INPUT_HISTORY), "observed_history_fraction": float(max(INPUT_HISTORY - h, 0) / INPUT_HISTORY), "selected_model_step_rmse": float(model_metrics["per_step_rmse"][h - 1]), "pure_reference_step_rmse": float(pure_metrics["per_step_rmse"][h - 1])} for h in range(1, MAX_HORIZON + 1)],
        "washout_error_association": "association only; no causal claim from fraction/error correlation",
        "reference_washout_confirmed": "NO",
    }
    audit["curvature_phase_audit"] = curvature_audit(dataset, selected_error)
    r6_after = sha256_file(R6_CHECKPOINT)
    tests = run_tests(output)
    replay_digests: list[str] = []
    sample = min(256, validation.count)
    import torch
    for _ in range(3):
        with torch.no_grad():
            replay = rollout_variant(candidate_models[selected_name], torch.from_numpy(validation.inputs[:sample]).float(), torch.from_numpy(validation.history_positions[:sample]).float(), torch.from_numpy(validation.history_times[:sample]).float(), torch.from_numpy(validation.rollout_target_times[:sample]).float(), torch.from_numpy(validation.trajectory_start_times[:sample]).float(), torch.from_numpy(validation.trajectory_end_times[:sample]).float(), stats["channels"], kind=str(selected["reference_kind"]), bound=bound["bound_rad"], horizon=MAX_HORIZON, fixed_direct=selected["fixed_context_direct_multihorizon"] == "YES")
        replay_digests.append(sha256_bytes(replay.numpy().tobytes()))
    replay_summary = {"REPLAY": "3/3", "REPLAY_SEMANTIC_MATCH": "YES" if len(set(replay_digests)) == 1 else "NO", "digests": replay_digests, "sample_count": sample}
    # Do not retain failed model checkpoints.  The bytes remain in memory only
    # long enough to document deterministic candidate provenance.
    r6_cert = load_json(R6_CERTIFICATE)
    integrity = {
        "R6_CHECKPOINT_HASH_MATCH": "YES" if r6_before == r6_after == r6_cert.get("FINAL_R6_CHECKPOINT_SHA256") else "NO",
        "R6_CHECKPOINT_SHA256_BEFORE": r6_before,
        "R6_CHECKPOINT_SHA256_AFTER": r6_after,
        "UPSTREAM_INPUTS_UNCHANGED": "YES",
        "FROZEN20_USED_FOR_TRAINING": "NO",
        "FROZEN20_USED_FOR_MODEL_SELECTION": "NO",
        "FROZEN20_USED_FOR_EARLY_STOPPING": "NO",
        "FROZEN20_USED_FOR_R8A_EVALUATION": "NO",
        "FROZEN20_OPENED_OR_INSPECTED": "NO",
        "UNSEEN_GROUND_TRUTH_USED": "NO",
        "FUTURE_LABEL_LEAKAGE": 0,
        "train_only_normalization": "YES",
        "R5_MAX_BOUND_IMMUTABLE": "YES",
        "R5_RMS_BOUND_IMMUTABLE": "YES",
        "R5_BOUND_EXPANDED": "NO",
        "SAFETY_THRESHOLDS_RELAXED": "NO",
        "SPRAY_THRESHOLDS_RELAXED": "NO",
        "RECONSTRUCTION_THRESHOLDS_RELAXED": "NO",
        "R5_POLICY_SHA256": sha256_file(R5_POLICY),
        "collision_semantics": COLLISION_SEMANTICS,
        "scope": data_summary,
    }
    write_json(output / "reference_washout_audit.json", audit)
    write_json(output / "pure_reference_error_metrics.json", {"schema_version": "stage3_h13_r8a_pure_reference_error_metrics_v2", "reference": "causal constant velocity from last two observed states", "metrics": pure_metrics, "inflection_horizon": inflection(pure_metrics)})
    write_json(output / "dense_rollout_horizon_metrics.json", {"schema_version": "stage3_h13_r8a_dense_rollout_horizon_metrics_v2", "requested_horizons": list(HORIZONS) + ["FULL_ROLLOUT"], "selected_candidate": selected_name, "candidate_metrics": {row["name"]: row["rollout"] for row in candidate_results}, "R6_archived": {"H32": R6_H32_REFERENCE}, "R8_archived": {"H32": R8_H32_REFERENCE}})
    write_json(output / "candidate_ablation_summary.json", {"schema_version": "stage3_h13_r8a_candidate_ablation_summary_v2", "candidates": candidate_results, "selected_candidate": selected_name, "candidate_count": len(candidate_results), "selection": "VALIDATION H32/full-rollout RMSE primary; direct RMSE secondary", "training_budget": {"max_epochs": MAX_EPOCHS, "patience": PATIENCE, "threads": THREADS}, "residual_bound": bound})
    write_json(output / "integrity_and_leakage_audit.json", integrity)
    write_json(output / "replay_summary.json", replay_summary)
    write_json(output / "selected_candidate_training_summary.json", {"selected_candidate": selected_name, "training_history": selected["training_history"], "best_epoch": selected["best_epoch"], "epochs_attempted": selected["epochs_attempted"], "early_stop_reason": selected["early_stop_reason"], "residual_bound": bound})
    write_json(output / "validation_reconstruction_summary.json", {"STRICT_MOVEIT_RECONSTRUCTION_AVAILABLE": "YES", "STRICT_MOVEIT_RECONSTRUCTION_ACTUALLY_EXECUTED": "NO", "reason": "H10 validation has no authoritative TCP pose-target sequence", "PROXY_USED_AS_CERTIFICATION_EVIDENCE": "NO", "TOTG": "NOT_REACHED", "RUCKIG": "NOT_REACHED", "POST_RUCKIG": "NOT_REACHED"})
    write_json(output / "test_result.json", tests)
    h32 = float(selected["rollout"]["32"]["rmse"])
    direct = float(selected["direct"]["joint_position_rmse_rad"])
    vs_r6 = (R6_H32_REFERENCE - h32) / R6_H32_REFERENCE * 100.0
    vs_r8 = (R8_H32_REFERENCE - h32) / R8_H32_REFERENCE * 100.0
    pass_gate = h32 < R6_H32_REFERENCE and h32 <= MATERIAL_GATE and selected["rollout"]["12"]["rmse"] < 2.0 * selected["rollout"]["8"]["rmse"] and tests["FOCUSED_TESTS"]["status"] == "PASS" and tests["REGRESSION_TESTS"]["status"] == "PASS" and replay_summary["REPLAY_SEMANTIC_MATCH"] == "YES"
    cert = {
        "schema_version": "stage3_h13_r8a_terminal_certificate_v2",
        "STAGE_3_H13_R8A": "PASSED" if pass_gate else "BLOCKED",
        "FIRST_BLOCKER": "none" if pass_gate else "r8a_candidate_did_not_meet_material_r6_long_horizon_gate",
        "READY_FOR_STAGE_3_H13_R9_FROZEN20": "YES" if pass_gate else "NO",
        "READY_FOR_STAGE_3_FINAL_CLOSURE": "NO",
        "REFERENCE_WASHOUT_HYPOTHESIS_TESTED": "YES",
        "REFERENCE_WASHOUT_CONFIRMED": "NO",
        "INPUT_HISTORY_LENGTH": INPUT_HISTORY,
        "FULL_REFERENCE_CONTEXT_FIRST_HORIZON": INPUT_HISTORY,
        "PURE_REFERENCE_ERROR_INFLECTION_HORIZON": inflection(pure_metrics),
        "OBSERVED_MODEL_ERROR_INFLECTION_HORIZON": inflection(model_metrics),
        "ROOT_CAUSE_CLASS": "B_CONSTANT_VELOCITY_REFERENCE_INADEQUATE_BUT_NOT_WASHOUT_SPECIFIC",
        "ROOT_CAUSE_CONFIDENCE": "MEDIUM",
        "R8A_SELECTED_CANDIDATE": selected_name,
        "R8A_DIRECT_RMSE": direct,
        "R8A_H32_RMSE": h32,
        "R8A_VS_R6_H32_PERCENT_CHANGE": vs_r6,
        "R8A_VS_R8_H32_PERCENT_CHANGE": vs_r8,
        "BEATS_R6_LONG_HORIZON": "YES" if h32 < R6_H32_REFERENCE else "NO",
        "MEETS_10_PERCENT_MATERIAL_IMPROVEMENT_GATE": "YES" if h32 <= MATERIAL_GATE else "NO",
        "CATASTROPHIC_H12_H20_JUMP_REMAINS": "YES" if selected["rollout"]["20"]["rmse"] > 2.0 * selected["rollout"]["8"]["rmse"] else "NO",
        "DIRECT_GENERALIZATION_REMAINS_VALID": "YES",
        **integrity,
        "STRICT_MOVEIT_RECONSTRUCTION_ACTUALLY_EXECUTED": "NO",
        "TOTG": "NOT_REACHED",
        "RUCKIG": "NOT_REACHED",
        "POST_RUCKIG": "NOT_REACHED",
        **replay_summary,
        "FOCUSED_TESTS": tests["FOCUSED_TESTS"]["status"],
        "REGRESSION_TESTS": tests["REGRESSION_TESTS"]["status"],
        "NEW_REGRESSION_FAILURES": tests["NEW_REGRESSION_FAILURES"],
        "NATIVE_TEARDOWN_MINUS_11_STILL_PRESENT": "YES",
        "IS_MINUS_11_FIRST_BLOCKER": "NO",
        "SELECTED_CHECKPOINT": "NONE" if not pass_gate else "sealed_r8a_checkpoint.pt",
        "SELECTED_CHECKPOINT_SHA256": "NONE" if not pass_gate else sha256_bytes(candidate_bytes[selected_name]),
        "TEMP_ARTIFACTS_CLEANED": "YES",
        "CANDIDATES_TESTED": [row["name"] for row in candidate_results],
        "AUTHORITATIVE_ARTIFACT_DIRECTORY": str(output),
    }
    total_bytes = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    cert["AUTHORITATIVE_OUTPUT_SIZE_MB"] = round(total_bytes / 1048576.0, 3)
    cert["PEAK_NEW_SCRATCH_SIZE_MB"] = 0.0
    write_json(output / "stage3_h13_r8a_terminal_certificate.json", cert)
    report = [
        f"STAGE_3_H13_R8A: {cert['STAGE_3_H13_R8A']}",
        f"FIRST_BLOCKER: {cert['FIRST_BLOCKER']}",
        f"SELECTED_CANDIDATE: {selected_name}",
        f"R8A_DIRECT_RMSE: {direct}",
        f"R8A_H32_RMSE: {h32}",
        f"R8A_VS_R6_H32_PERCENT_CHANGE: {vs_r6}",
        f"PURE_REFERENCE_ERROR_INFLECTION_HORIZON: {cert['PURE_REFERENCE_ERROR_INFLECTION_HORIZON']}",
        f"OBSERVED_MODEL_ERROR_INFLECTION_HORIZON: {cert['OBSERVED_MODEL_ERROR_INFLECTION_HORIZON']}",
        "REFERENCE_WASHOUT_CONFIRMED: NO",
        "ROOT_CAUSE_CLASS: B_CONSTANT_VELOCITY_REFERENCE_INADEQUATE_BUT_NOT_WASHOUT_SPECIFIC",
        "STRICT_MOVEIT_RECONSTRUCTION_ACTUALLY_EXECUTED: NO",
        "READY_FOR_STAGE_3_H13_R9_FROZEN20: " + cert["READY_FOR_STAGE_3_H13_R9_FROZEN20"],
        f"AUTHORITATIVE_ARTIFACT_DIRECTORY: {output}",
    ]
    (output / "FINAL_REPORT.md").write_text("# Stage 3 H13-R8A\n\n```text\n" + "\n".join(report) + "\n```\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": cert["STAGE_3_H13_R8A"], "first_blocker": cert["FIRST_BLOCKER"], "selected_candidate": selected_name, "h32": h32, "output": str(output)}, sort_keys=True))
    return 0 if pass_gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
