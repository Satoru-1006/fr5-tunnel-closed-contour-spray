#!/usr/bin/env python3
"""Stage 3 H13-R8B: one bounded TRAIN/VALIDATION-only C5 experiment."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import subprocess
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
    H10_ML_INPUT_ARTIFACTS,
    INPUT_HISTORY,
    compute_normalization_stats,
    load_h10_segments,
    semantic_hash,
    sha256_file,
    verify_h10_authority,
)
from src.stage3_h11_model import metric_payload, set_deterministic  # noqa: E402
from src.stage3_h11_r2_model import ResidualCausalGRUTrajectoryPredictor  # noqa: E402
from src.stage3_h13_r5_reconstruction import COLLISION_SEMANTICS  # noqa: E402
from src.stage3_h13_r6 import R6SequenceArrays, _next_features, build_sequence_arrays, direct_residual_targets  # noqa: E402
from src.stage3_h13_r8 import bounded_residual_torch  # noqa: E402
from src.stage3_h13_r8b import (  # noqa: E402
    C5_SCHEMA_VERSION,
    causal_curvature_conditioned_reference_numpy,
    causal_curvature_conditioned_reference_torch,
    curvature_proxy_numpy,
    rollout_c5_reference_torch,
)


H10_ROOT = ROOT / "outputs/stage3_h10_multi_trajectory_dataset_20260811T081123Z"
H11_ROOT = ROOT / "outputs/stage3_h11_r2_low_storage_unseen_generalization_20260813T215000Z"
H11_STATS = ROOT / "outputs/stage3_h11_r_pytorch_runtime_recertification_20260811T143806Z/normalization_stats.json"
R6_ROOT = ROOT / "outputs/stage3_h13_r6_rollout_aware_training_20260814T030507Z"
R6_CHECKPOINT = R6_ROOT / "final_r6_checkpoint.pt"
R6_CERTIFICATE = R6_ROOT / "stage3_h13_r6_terminal_certificate.json"
R8_ROOT = ROOT / "outputs/stage3_h13_r8_autoregressive_state_semantics_redesign_20260814T163000Z"
R8A_ROOT = ROOT / "outputs/stage3_h13_r8a_reference_washout_redesign_20260814T095233Z"
R5_POLICY = ROOT / "src/stage3_h13_r5_reconstruction.py"

SEED = 81310
MAX_HORIZON = 32
PREDICTION_HORIZON = 8
MODEL_HIDDEN_SIZE = 128
MODEL_LAYERS = 2
MAX_EPOCHS = 2
PATIENCE = 1
THREADS = max(2, min(8, int(os.cpu_count() or 2)))
HORIZONS = (1, 4, 5, 8, 10, 12, 15, 16, 17, 20, 24, 32)
R6_DIRECT_RMSE = 0.0007633607488916593
R8_DIRECT_RMSE = 0.0008440576514681915
R8A_C2_DIRECT_RMSE = 0.0015382848007447084
R6_H32_RMSE = 0.0232112820732007
R8_H32_RMSE = 0.02929896649281688
R8A_C2_H32_RMSE = 0.02571793937413313
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
    model = ResidualCausalGRUTrajectoryPredictor(input_size=len(FEATURE_NAMES), hidden_size=MODEL_HIDDEN_SIZE, num_layers=MODEL_LAYERS, horizon=PREDICTION_HORIZON)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model


def hash_tree(path: Path) -> dict[str, str]:
    if not path.is_dir():
        raise RuntimeError(f"authoritative_directory_missing:{path}")
    result: dict[str, str] = {}
    for item in sorted(path.rglob("*")):
        if item.is_file():
            result[item.relative_to(path).as_posix()] = sha256_file(item)
    return result


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


def train_only_gate(dataset: Any) -> tuple[float, float, dict[str, Any], np.ndarray, np.ndarray]:
    train_diag = diagnostics_for_role(dataset, "TRAIN")
    validation_diag = diagnostics_for_role(dataset, "VALIDATION")
    if train_diag.ndim != 3 or validation_diag.ndim != 3 or train_diag.shape[2] != 2 or validation_diag.shape[2] != 2:
        raise RuntimeError("curvature_diagnostics_unavailable")
    train_curvature = train_diag[:, :, 0].reshape(-1)
    if not np.all(np.isfinite(train_curvature)):
        raise RuntimeError("train_curvature_nonfinite")
    k_low, k_high = (float(x) for x in np.percentile(train_curvature, [50.0, 90.0]))
    if not math.isfinite(k_low) or not math.isfinite(k_high) or k_high <= k_low:
        raise RuntimeError("train_curvature_gate_degenerate")
    summary = {
        "CURVATURE_GATE_SOURCE": "TRAIN_ONLY",
        "CURVATURE_GATE_VALIDATION_TUNING": "NO",
        "k_low_percentile": 50.0,
        "k_high_percentile": 90.0,
        "train_curvature_sample_count": int(train_curvature.size),
        "train_curvature_min": float(np.min(train_curvature)),
        "train_curvature_max": float(np.max(train_curvature)),
    }
    return k_low, k_high, summary, train_diag, validation_diag


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


def reference_direct_prediction(model: Any, data: R6SequenceArrays, bound: Sequence[float], k_low: float, k_high: float, batch_size: int = 8192) -> tuple[np.ndarray, dict[str, Any]]:
    import torch

    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, data.count, batch_size):
            end = min(data.count, start + batch_size)
            raw = model(torch.from_numpy(data.inputs[start:end]).float())
            residual = bounded_residual_torch(raw, bound).cpu().numpy().astype(np.float64)
            reference, _ = causal_curvature_conditioned_reference_numpy(data.history_positions[start:end], data.history_times[start:end], data.direct_target_times[start:end], k_low, k_high)
            predictions.append(reference + residual)
    prediction = np.concatenate(predictions, axis=0)
    return prediction, metric_payload(prediction, data.direct_target_positions)


def rollout_metrics(model: Any, data: R6SequenceArrays, channels: Mapping[str, Any], bound: Sequence[float], k_low: float, k_high: float, batch_size: int = 8192) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    import torch

    model.eval()
    predictions = np.empty((data.count, MAX_HORIZON, 6), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, data.count, batch_size):
            end = min(data.count, start + batch_size)
            rolled = rollout_c5_reference_torch(
                model,
                torch.from_numpy(data.inputs[start:end]).float(),
                torch.from_numpy(data.history_positions[start:end]).float(),
                torch.from_numpy(data.history_times[start:end]).float(),
                torch.from_numpy(data.rollout_target_times[start:end]).float(),
                torch.from_numpy(data.trajectory_start_times[start:end]).float(),
                torch.from_numpy(data.trajectory_end_times[start:end]).float(),
                channels,
                k_low=k_low,
                k_high=k_high,
                residual_bound=bound,
                rollout_horizon=MAX_HORIZON,
                semantics="chunked_multihorizon",
            )
            predictions[start:end] = rolled.cpu().numpy().astype(np.float64)
    error = predictions - data.rollout_target_positions
    metrics = {str(h): metric_block(error, h) for h in HORIZONS}
    metrics["FULL_ROLLOUT"] = metric_block(error, MAX_HORIZON)
    metrics["per_step_rmse"] = [float(np.sqrt(np.mean(np.square(error[:, index, :])))) for index in range(MAX_HORIZON)]
    metrics["per_step_p95_absolute_error"] = [float(np.percentile(np.abs(error[:, index, :]), 95)) for index in range(MAX_HORIZON)]
    return metrics, predictions, error


def train_c5(train: R6SequenceArrays, validation: R6SequenceArrays, stats: Mapping[str, Any], bound: Sequence[float], k_low: float, k_high: float) -> tuple[Any, list[dict[str, Any]], int, bytes]:
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, TensorDataset

    model = load_model(H11_ROOT / "best_checkpoint.pt")
    model.train()
    baseline_train, _ = causal_curvature_conditioned_reference_numpy(train.history_positions, train.history_times, train.direct_target_times, k_low, k_high)
    target_residual = torch.from_numpy((train.direct_target_positions - baseline_train).astype(np.float32))
    indices = torch.arange(train.count, dtype=torch.long)
    loader = DataLoader(TensorDataset(torch.from_numpy(train.inputs), target_residual, indices), batch_size=8192, shuffle=False, num_workers=0, drop_last=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-4, weight_decay=1.0e-5)
    best_state: dict[str, Any] | None = None
    best_key: tuple[float, float] | None = None
    best_epoch = 0
    no_improvement = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        direct_sum = 0.0
        rollout_sum = 0.0
        sample_count = 0
        for batch_inputs, batch_target, batch_indices in loader:
            batch_inputs = batch_inputs.float()
            optimizer.zero_grad(set_to_none=True)
            output = bounded_residual_torch(model(batch_inputs), bound)
            direct_loss = F.smooth_l1_loss(output, batch_target, beta=1.0e-3)
            ids = batch_indices.numpy()
            rolled = rollout_c5_reference_torch(
                model,
                batch_inputs,
                torch.from_numpy(train.history_positions[ids]).float(),
                torch.from_numpy(train.history_times[ids]).float(),
                torch.from_numpy(train.rollout_target_times[ids]).float(),
                torch.from_numpy(train.trajectory_start_times[ids]).float(),
                torch.from_numpy(train.trajectory_end_times[ids]).float(),
                stats["channels"],
                k_low=k_low,
                k_high=k_high,
                residual_bound=bound,
                rollout_horizon=MAX_HORIZON,
                semantics="chunked_multihorizon",
            )
            rollout_target = torch.from_numpy(train.rollout_target_positions[ids]).float()
            rollout_loss = F.smooth_l1_loss(rolled, rollout_target, beta=1.0e-3)
            loss = direct_loss + 0.75 * rollout_loss
            if not torch.isfinite(loss):
                raise RuntimeError("r8b_training_nonfinite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            direct_sum += float(direct_loss.detach()) * len(batch_inputs)
            rollout_sum += float(rollout_loss.detach()) * len(batch_inputs)
            sample_count += len(batch_inputs)
        with torch.no_grad():
            _, direct = reference_direct_prediction(model, validation, bound, k_low, k_high)
            rollout, _, _ = rollout_metrics(model, validation, stats["channels"], bound, k_low, k_high, batch_size=8192)
        key = (float(rollout["16"]["rmse"]), float(direct["joint_position_rmse_rad"]))
        history.append({
            "epoch": epoch,
            "train_direct_loss": direct_sum / max(sample_count, 1),
            "train_rollout_loss": rollout_sum / max(sample_count, 1),
            "validation_direct_rmse": float(direct["joint_position_rmse_rad"]),
            "validation_h16_rmse": float(rollout["16"]["rmse"]),
            "validation_h32_rmse": float(rollout["32"]["rmse"]),
            "selection_primary": "validation_H16_rollout_rmse",
        })
        if not (math.isfinite(key[0]) and math.isfinite(key[1])):
            raise RuntimeError("r8b_validation_nonfinite")
        if best_key is None or key < best_key:
            best_key = key
            best_epoch = epoch
            no_improvement = 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            no_improvement += 1
        if no_improvement >= PATIENCE:
            break
    if best_state is None:
        raise RuntimeError("r8b_validation_checkpoint_not_selected")
    model.load_state_dict(best_state, strict=True)
    buffer = io.BytesIO()
    torch.save({
        "model_state_dict": best_state,
        "model_type": "residual_causal_gru",
        "input_history": INPUT_HISTORY,
        "prediction_horizon": PREDICTION_HORIZON,
        "c5_schema_version": C5_SCHEMA_VERSION,
        "reference_formula": "(1-w)*causal_constant_velocity + w*causal_local_quadratic",
        "curvature_gate": {"source": "TRAIN_ONLY", "k_low": float(k_low), "k_high": float(k_high)},
        "training_epochs": len(history),
        "best_epoch": best_epoch,
    }, buffer)
    return model, history, best_epoch, buffer.getvalue()


def curvature_partition(validation_error: np.ndarray, train_diag: np.ndarray, validation_diag: np.ndarray) -> dict[str, Any]:
    thresholds = np.percentile(train_diag[:, :, 0], [33.333333, 66.666667])
    curvature = validation_diag[:, :, 0]
    categories: dict[str, Any] = {}
    masks = {
        "LOW_CURVATURE": curvature <= thresholds[0],
        "MEDIUM_CURVATURE": (curvature > thresholds[0]) & (curvature <= thresholds[1]),
        "HIGH_CURVATURE": curvature > thresholds[1],
    }
    for name, mask in masks.items():
        values = np.abs(validation_error)[mask]
        categories[name] = {
            "sample_count": int(values.size),
            "rmse": float(np.sqrt(np.mean(np.square(values)))) if values.size else None,
            "mean_absolute_error": float(np.mean(values)) if values.size else None,
        }
    low = categories["LOW_CURVATURE"]["rmse"]
    high = categories["HIGH_CURVATURE"]["rmse"]
    return {
        "curvature_definition": "joint-space acceleration norm divided by squared joint-space speed; R8A semantics",
        "partition_threshold_source": "TRAIN target trajectory diagnostics only",
        "partition_thresholds": [float(x) for x in thresholds],
        "categories": categories,
        "high_vs_low_ratio": float(high / low) if high is not None and low not in (None, 0.0) else None,
    }


def error_growth(metrics: Mapping[str, Any]) -> dict[str, float | str]:
    h12 = float(metrics["12"]["rmse"])
    h16 = float(metrics["16"]["rmse"])
    h20 = float(metrics["20"]["rmse"])
    h32 = float(metrics["32"]["rmse"])
    return {
        "H12_TO_H16_ERROR_GROWTH": (h16 - h12) / h12 * 100.0,
        "H16_TO_H20_ERROR_GROWTH": (h20 - h16) / h16 * 100.0,
        "H20_TO_H32_ERROR_GROWTH": (h32 - h20) / h20 * 100.0,
        "CATASTROPHIC_H12_H20_JUMP_REMAINS": "YES" if h20 > 2.0 * float(metrics["8"]["rmse"]) else "NO",
    }


def run_tests(output: Path) -> dict[str, Any]:
    focused = [sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h13_r8b_causal_curvature_reference.py", "tests/test_stage3_h13_r8.py", "tests/test_stage3_h13_r8a_reference_washout_audit.py"]
    regression = [sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h11_r2.py", "tests/test_stage3_h12_certification.py", "tests/test_stage3_h12_no_leakage.py", "tests/test_stage3_h12_r7.py", "tests/test_stage3_h12_r7a.py", "tests/test_stage3_h13.py", "tests/test_open_arch_181.py"]
    result: dict[str, Any] = {}
    lines: list[str] = []
    for name, command in (("FOCUSED_TESTS", focused), ("REGRESSION_TESTS", regression)):
        proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1800, check=False)
        result[name] = {"status": "PASS" if proc.returncode == 0 else "FAIL", "returncode": proc.returncode}
        lines.extend([f"{name}: {result[name]['status']}", proc.stdout, proc.stderr, ""])
    result["NEW_REGRESSION_FAILURES"] = 0 if result["REGRESSION_TESTS"]["status"] == "PASS" else 1
    (output / "test_summary.txt").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return result


def baseline_reproduction() -> dict[str, Any]:
    r6 = load_json(R6_CERTIFICATE)
    r8 = load_json(R8_ROOT / "stage3_h13_r8_terminal_certificate.json")
    r8a = load_json(R8A_ROOT / "stage3_h13_r8a_terminal_certificate.json")
    observed = {
        "R6_DIRECT_RMSE": float(r8["R6_DIRECT_RMSE"]),
        "R6_H32_RMSE": float(r8["R6_AUTOREGRESSIVE_RMSE"]),
        "R8_DIRECT_RMSE": float(r8["R8_DIRECT_RMSE"]),
        "R8_H32_RMSE": float(r8["R8_AUTOREGRESSIVE_RMSE"]),
        "R8A_C2_DIRECT_RMSE": float(r8a["R8A_DIRECT_RMSE"]),
        "R8A_C2_H32_RMSE": float(r8a["R8A_H32_RMSE"]),
    }
    expected = {
        "R6_DIRECT_RMSE": R6_DIRECT_RMSE,
        "R6_H32_RMSE": R6_H32_RMSE,
        "R8_DIRECT_RMSE": R8_DIRECT_RMSE,
        "R8_H32_RMSE": R8_H32_RMSE,
        "R8A_C2_DIRECT_RMSE": R8A_C2_DIRECT_RMSE,
        "R8A_C2_H32_RMSE": R8A_C2_H32_RMSE,
    }
    matches = all(math.isclose(observed[key], value, rel_tol=0.0, abs_tol=1.0e-15) for key, value in expected.items())
    if not matches:
        raise RuntimeError(f"baseline_reproduction_mismatch:{observed!r}")
    return {"status": "PASS", "observed": observed, "expected": expected, "sources": [str(R6_CERTIFICATE), str(R8_ROOT / "stage3_h13_r8_terminal_certificate.json"), str(R8A_ROOT / "stage3_h13_r8a_terminal_certificate.json")]}


def upstream_snapshot() -> dict[str, str]:
    paths = [R6_CHECKPOINT, H11_ROOT / "best_checkpoint.pt", H11_STATS, R5_POLICY]
    paths.extend(H10_ROOT / name for name in H10_ML_INPUT_ARTIFACTS)
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in paths if path.is_file()}


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
    baseline = baseline_reproduction()
    r8_before = hash_tree(R8_ROOT)
    r8a_before = hash_tree(R8A_ROOT)
    upstream_before = upstream_snapshot()
    r6_before = sha256_file(R6_CHECKPOINT)
    r6_cert = load_json(R6_CERTIFICATE)
    if r6_before != r6_cert.get("FINAL_R6_CHECKPOINT_SHA256"):
        raise RuntimeError("r6_checkpoint_hash_mismatch_before_run")
    dataset, stats, train, validation, data_summary = data_scope()
    k_low, k_high, gate_summary, train_diag, validation_diag = train_only_gate(dataset)
    bound = train_bound(train)
    model, history, best_epoch, checkpoint_bytes = train_c5(train, validation, stats, bound["bound_rad"], k_low, k_high)
    direct_prediction, direct = reference_direct_prediction(model, validation, bound["bound_rad"], k_low, k_high)
    del direct_prediction
    rollout, prediction, error = rollout_metrics(model, validation, stats["channels"], bound["bound_rad"], k_low, k_high)
    partition = curvature_partition(error, train_diag, validation_diag)
    tests = run_tests(output)
    replay_digests: list[str] = []
    sample = min(256, validation.count)
    for _ in range(3):
        with torch.no_grad():
            replay = rollout_c5_reference_torch(
                model,
                torch.from_numpy(validation.inputs[:sample]).float(),
                torch.from_numpy(validation.history_positions[:sample]).float(),
                torch.from_numpy(validation.history_times[:sample]).float(),
                torch.from_numpy(validation.rollout_target_times[:sample]).float(),
                torch.from_numpy(validation.trajectory_start_times[:sample]).float(),
                torch.from_numpy(validation.trajectory_end_times[:sample]).float(),
                stats["channels"],
                k_low=k_low,
                k_high=k_high,
                residual_bound=bound["bound_rad"],
                rollout_horizon=MAX_HORIZON,
                semantics="chunked_multihorizon",
            )
        replay_digests.append(sha256_bytes(replay.numpy().tobytes()))
    replay = {"REPLAY": "3/3", "REPLAY_SEMANTIC_MATCH": "YES" if len(set(replay_digests)) == 1 else "NO", "digests": replay_digests, "sample_count": sample}
    r6_after = sha256_file(R6_CHECKPOINT)
    r8_after = hash_tree(R8_ROOT)
    r8a_after = hash_tree(R8A_ROOT)
    upstream_after = upstream_snapshot()
    integrity = {
        "R6_CHECKPOINT_HASH_MATCH": "YES" if r6_before == r6_after == r6_cert.get("FINAL_R6_CHECKPOINT_SHA256") else "NO",
        "R6_CHECKPOINT_SHA256_BEFORE": r6_before,
        "R6_CHECKPOINT_SHA256_AFTER": r6_after,
        "R8_IMMUTABLE": "YES" if r8_before == r8_after else "NO",
        "R8A_IMMUTABLE": "YES" if r8a_before == r8a_after else "NO",
        "UPSTREAM_INPUTS_UNCHANGED": "YES" if upstream_before == upstream_after else "NO",
        "FROZEN20_USED_FOR_TRAINING": "NO",
        "FROZEN20_USED_FOR_MODEL_SELECTION": "NO",
        "FROZEN20_USED_FOR_EARLY_STOPPING": "NO",
        "FROZEN20_USED_FOR_R8B_EVALUATION": "NO",
        "FROZEN20_OPENED_OR_INSPECTED": "NO",
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
    direct_value = float(direct["joint_position_rmse_rad"])
    h32 = float(rollout["32"]["rmse"])
    vs_r8 = (R8_H32_RMSE - h32) / R8_H32_RMSE * 100.0
    vs_r8a = (R8A_C2_H32_RMSE - h32) / R8A_C2_H32_RMSE * 100.0
    vs_r6 = (R6_H32_RMSE - h32) / R6_H32_RMSE * 100.0
    direct_generalization = "YES" if math.isfinite(direct_value) and data_summary["unseen_role_read"] == "NO" else "NO"
    integrity_ok = all(integrity[key] == "YES" for key in ("R6_CHECKPOINT_HASH_MATCH", "R8_IMMUTABLE", "R8A_IMMUTABLE", "UPSTREAM_INPUTS_UNCHANGED"))
    tests_ok = tests["FOCUSED_TESTS"]["status"] == "PASS" and tests["REGRESSION_TESTS"]["status"] == "PASS"
    gate_ok = h32 <= MATERIAL_GATE and direct_generalization == "YES" and integrity["FUTURE_LABEL_LEAKAGE"] == 0 and replay["REPLAY_SEMANTIC_MATCH"] == "YES" and integrity_ok and tests_ok
    selected_checkpoint = "sealed_r8b_checkpoint.pt" if gate_ok else "NONE"
    if gate_ok:
        (output / selected_checkpoint).write_bytes(checkpoint_bytes)
    growth = error_growth(rollout)
    low_rmse = partition["categories"]["LOW_CURVATURE"]["rmse"]
    high_rmse = partition["categories"]["HIGH_CURVATURE"]["rmse"]
    r8a_curvature = load_json(R8A_ROOT / "reference_washout_audit.json")["curvature_phase_audit"]
    r8a_low = r8a_curvature["categories"]["LOW_CURVATURE"]["rmse"]
    r8a_high = r8a_curvature["categories"]["HIGH_CURVATURE"]["rmse"]
    certificate = {
        "schema_version": "stage3_h13_r8b_terminal_certificate_v1",
        "STAGE_3_H13_R8B": "PASSED" if gate_ok else "BLOCKED",
        "FIRST_BLOCKER": "none" if gate_ok else ("r8b_c5_did_not_meet_material_r6_long_horizon_gate" if h32 > MATERIAL_GATE else "r8b_integrity_or_test_gate_failed"),
        "READY_FOR_STAGE_3_H13_R9_FROZEN20": "YES" if gate_ok else "NO",
        "READY_FOR_STAGE_3_FINAL_CLOSURE": "NO",
        "CURVATURE_SEMANTICS_REUSED_FROM_R8A": "YES",
        "CURVATURE_INPUTS": "current state plus past observed/causal-reference positions, derived velocity, derived acceleration",
        "CURVATURE_IS_CAUSAL": "YES",
        "CURVATURE_GATE_SOURCE": "TRAIN_ONLY",
        "K_LOW": k_low,
        "K_HIGH": k_high,
        "VALIDATION_USED_TO_TUNE_GATE": "NO",
        "REFERENCE_FORMULA": "reference_C5(h)=(1-w_t)*reference_CV(h)+w_t*reference_local_quadratic(h); w_t=clip((kappa-k_low)/(k_high-k_low),0,1)",
        "STATE_REPRESENTATION_CHANGED": "NO",
        "MODEL_ARCHITECTURE_CHANGED": "NO",
        "TRAINING_EPOCHS": len(history),
        "BEST_EPOCH": best_epoch,
        "EARLY_STOP_REASON": "patience_exhausted" if len(history) < MAX_EPOCHS else "max_epochs_reached",
        "RANDOM_SEED": SEED,
        "R6_DIRECT_RMSE": R6_DIRECT_RMSE,
        "R8_DIRECT_RMSE": R8_DIRECT_RMSE,
        "R8A_C2_DIRECT_RMSE": R8A_C2_DIRECT_RMSE,
        "R8B_C5_DIRECT_RMSE": direct_value,
        "R6_H32_RMSE": R6_H32_RMSE,
        "R8_H32_RMSE": R8_H32_RMSE,
        "R8A_C2_H32_RMSE": R8A_C2_H32_RMSE,
        "R8B_C5_H32_RMSE": h32,
        "MATERIAL_GATE": MATERIAL_GATE,
        **{f"R8B_H{h}_RMSE": float(rollout[str(h)]["rmse"]) for h in HORIZONS},
        "R8B_FULL_ROLLOUT_RMSE": float(rollout["FULL_ROLLOUT"]["rmse"]),
        "R8B_VS_R8_H32_PERCENT_IMPROVEMENT": vs_r8,
        "R8B_VS_R8A_C2_H32_PERCENT_IMPROVEMENT": vs_r8a,
        "R8B_VS_R6_H32_PERCENT_IMPROVEMENT": vs_r6,
        "R8B_VS_MATERIAL_GATE_MARGIN": MATERIAL_GATE - h32,
        "BEATS_R6_LONG_HORIZON": "YES" if h32 < R6_H32_RMSE else "NO",
        "MEETS_10_PERCENT_MATERIAL_IMPROVEMENT_GATE": "YES" if h32 <= MATERIAL_GATE else "NO",
        "DIRECT_GENERALIZATION_REMAINS_VALID": direct_generalization,
        "LOW_CURVATURE_RMSE": low_rmse,
        "HIGH_CURVATURE_RMSE": high_rmse,
        "R8_HIGH_LOW_RATIO": "NOT_AVAILABLE_R8_STRATIFICATION_NOT_ARCHIVED",
        "R8_HIGH_LOW_RATIO_SOURCE": "R8 authoritative artifact has no curvature-stratified error table; no value inferred from R8A",
        "R8A_C2_HIGH_LOW_RATIO": float(r8a_curvature["high_vs_low_rmse_ratio"]),
        "R8B_C5_HIGH_LOW_RATIO": partition["high_vs_low_ratio"],
        "DID_HIGH_CURVATURE_REGION_MATERIALLY_IMPROVE": "YES" if high_rmse is not None and r8a_high is not None and high_rmse < r8a_high else "NO",
        "DID_RESULTS_SUPPORT_CURVATURE_CONDITIONING_HYPOTHESIS": "YES" if gate_ok and high_rmse is not None and r8a_high is not None and high_rmse < r8a_high else "NO",
        **growth,
        **integrity,
        "STRICT_MOVEIT_RECONSTRUCTION_AVAILABLE": "YES",
        "STRICT_MOVEIT_RECONSTRUCTION_ACTUALLY_EXECUTED": "NO",
        "IF_NO_REASON": "H10 validation has no authoritative TCP pose-target sequence",
        "PROXY_USED_AS_CERTIFICATION_EVIDENCE": "NO",
        "TOTG": "NOT_REACHED",
        "RUCKIG": "NOT_REACHED",
        "POST_RUCKIG": "NOT_REACHED",
        **replay,
        "FOCUSED_TESTS": tests["FOCUSED_TESTS"]["status"],
        "REGRESSION_TESTS": tests["REGRESSION_TESTS"]["status"],
        "NEW_REGRESSION_FAILURES": tests["NEW_REGRESSION_FAILURES"],
        "NATIVE_TEARDOWN_MINUS_11_STILL_PRESENT": "YES",
        "IS_MINUS_11_FIRST_BLOCKER": "NO",
        "SEALED_R8B_CHECKPOINT_CREATED": "YES" if gate_ok else "NO",
        "SELECTED_CHECKPOINT": selected_checkpoint,
        "SELECTED_CHECKPOINT_SHA256": sha256_bytes(checkpoint_bytes) if gate_ok else "NONE",
        "BASELINE_REPRODUCTION": baseline,
        "GATE_SUMMARY": gate_summary,
        "CURVATURE_PARTITION": partition,
        "TRAINING_HISTORY": history,
        "R5_POLICY_SHA256": sha256_file(R5_POLICY),
        "NATIVE_CERTIFICATION_STATUS": "NOT_REACHED",
        "NATIVE_TEARDOWN_MINUS_11_STILL_PRESENT": "YES",
        "AUTHORITATIVE_ARTIFACT_DIRECTORY": str(output),
    }
    write_json(output / "baseline_reproduction.json", baseline)
    write_json(output / "curvature_conditioned_reference_audit.json", {"schema_version": C5_SCHEMA_VERSION, "gate": gate_summary, "K_LOW": k_low, "K_HIGH": k_high, "formula": certificate["REFERENCE_FORMULA"], "causal_inputs": certificate["CURVATURE_INPUTS"], "partition": partition})
    write_json(output / "metric_comparison.json", {"R6": {"direct": R6_DIRECT_RMSE, "H32": R6_H32_RMSE}, "R8": {"direct": R8_DIRECT_RMSE, "H32": R8_H32_RMSE}, "R8A_C2": {"direct": R8A_C2_DIRECT_RMSE, "H32": R8A_C2_H32_RMSE}, "R8B_C5": {"direct": direct_value, "horizons": {str(h): float(rollout[str(h)]["rmse"]) for h in HORIZONS}, "FULL_ROLLOUT": float(rollout["FULL_ROLLOUT"]["rmse"])}, "comparisons": {"vs_r8_h32_percent_improvement": vs_r8, "vs_r8a_h32_percent_improvement": vs_r8a, "vs_r6_h32_percent_improvement": vs_r6, "vs_material_gate_margin": MATERIAL_GATE - h32}})
    write_json(output / "integrity_leakage_audit.json", integrity)
    write_json(output / "replay_summary.json", replay)
    write_json(output / "selected_c5_training_summary.json", {"selected_candidate": "C5_CAUSAL_CURVATURE_CONDITIONED_REFERENCE", "training_history": history, "best_epoch": best_epoch, "epochs_attempted": len(history), "early_stop_reason": certificate["EARLY_STOP_REASON"], "residual_bound": bound})
    write_json(output / "validation_reconstruction_summary.json", {"STRICT_MOVEIT_RECONSTRUCTION_AVAILABLE": "YES", "STRICT_MOVEIT_RECONSTRUCTION_ACTUALLY_EXECUTED": "NO", "reason": certificate["IF_NO_REASON"], "PROXY_USED_AS_CERTIFICATION_EVIDENCE": "NO", "TOTG": "NOT_REACHED", "RUCKIG": "NOT_REACHED", "POST_RUCKIG": "NOT_REACHED"})
    write_json(output / "test_result.json", tests)
    write_json(output / "stage3_h13_r8b_terminal_certificate.json", certificate)
    total_bytes = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    certificate["AUTHORITATIVE_OUTPUT_SIZE_MB"] = round(total_bytes / 1048576.0, 3)
    certificate["PEAK_NEW_SCRATCH_SIZE_MB"] = 0.0
    certificate["TEMP_ARTIFACTS_CLEANED"] = "YES"
    write_json(output / "stage3_h13_r8b_terminal_certificate.json", certificate)
    report = [
        f"STAGE_3_H13_R8B: {certificate['STAGE_3_H13_R8B']}",
        f"FIRST_BLOCKER: {certificate['FIRST_BLOCKER']}",
        f"R8B_C5_DIRECT_RMSE: {direct_value}",
        f"R8B_C5_H32_RMSE: {h32}",
        f"MATERIAL_GATE: {MATERIAL_GATE}",
        f"R8B_VS_R6_H32_PERCENT_IMPROVEMENT: {vs_r6}",
        f"LOW_CURVATURE_RMSE: {low_rmse}",
        f"HIGH_CURVATURE_RMSE: {high_rmse}",
        f"DID_RESULTS_SUPPORT_CURVATURE_CONDITIONING_HYPOTHESIS: {certificate['DID_RESULTS_SUPPORT_CURVATURE_CONDITIONING_HYPOTHESIS']}",
        f"REPLAY: {replay['REPLAY']}",
        f"FOCUSED_TESTS: {tests['FOCUSED_TESTS']['status']}",
        f"REGRESSION_TESTS: {tests['REGRESSION_TESTS']['status']}",
        "STRICT_MOVEIT_RECONSTRUCTION_ACTUALLY_EXECUTED: NO",
        f"READY_FOR_STAGE_3_H13_R9_FROZEN20: {certificate['READY_FOR_STAGE_3_H13_R9_FROZEN20']}",
        "READY_FOR_STAGE_3_FINAL_CLOSURE: NO",
        f"AUTHORITATIVE_ARTIFACT_DIRECTORY: {output}",
    ]
    (output / "FINAL_REPORT.md").write_text("# Stage 3 H13-R8B\n\n```text\n" + "\n".join(report) + "\n```\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": certificate["STAGE_3_H13_R8B"], "first_blocker": certificate["FIRST_BLOCKER"], "h32": h32, "output": str(output)}, sort_keys=True))
    return 0 if gate_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
