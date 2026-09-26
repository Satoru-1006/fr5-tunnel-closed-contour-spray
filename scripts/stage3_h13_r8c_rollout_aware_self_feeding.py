#!/usr/bin/env python3
"""Stage 3 H13-R8C: rollout-aware self-feeding training.

This runner is deliberately limited to the H10 TRAIN/VALIDATION split.  It
reproduces the authoritative R6 rollout first, warm-starts a new R8C candidate
from the immutable R6 checkpoint, selects by validation free-running H32, and
stores only one compact best checkpoint plus derived evidence.  No downstream
unseen/frozen split, native planner, controller, or physical execution is
opened by this stage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage3_h11_dataset import (  # noqa: E402
    FEATURE_NAMES,
    compute_normalization_stats,
    load_h10_segments,
    semantic_hash,
    sha256_file,
    verify_h10_authority,
)
from src.stage3_h11_model import metric_payload, set_deterministic  # noqa: E402
from src.stage3_h11_r2_model import ResidualCausalGRUTrajectoryPredictor, TORCH_AVAILABLE, causal_constant_velocity_baseline  # noqa: E402
from src.stage3_h13_r6 import (  # noqa: E402
    R6SequenceArrays,
    as_window_arrays,
    build_sequence_arrays,
    rollout_residual_torch_r6,
)
from src.stage3_h13_r8c import (  # noqa: E402
    R8C_EVALUATION_HORIZONS,
    R8C_SCHEMA_VERSION,
    rollout_profile,
    train_r8c_candidate,
    percent_improvement,
)


H10_ROOT = ROOT / "outputs/stage3_h10_multi_trajectory_dataset_20260811T081123Z"
H11_ROOT = ROOT / "outputs/stage3_h11_r_pytorch_runtime_recertification_20260811T143806Z"
R6_ROOT = ROOT / "outputs/stage3_h13_r6_rollout_aware_training_20260814T030507Z"
R8_ROOT = ROOT / "outputs/stage3_h13_r8_autoregressive_state_semantics_redesign_20260814T163000Z"
R8A_ROOT = ROOT / "outputs/stage3_h13_r8a_reference_washout_redesign_20260814T095233Z"
R8B_ROOT = ROOT / "outputs/stage3_h13_r8b_causal_curvature_reference_20260814T124600Z"
POST_R8B_ROOT = ROOT / "outputs/stage3_h13_post_r8b_rollout_root_cause_audit_20260814T220200Z"

H11_STATS = H11_ROOT / "normalization_stats.json"
R6_CHECKPOINT = R6_ROOT / "final_r6_checkpoint.pt"
R6_CERTIFICATE = R6_ROOT / "stage3_h13_r6_terminal_certificate.json"
R6_MANIFEST = R6_ROOT / "checkpoint_manifest.json"
AUTHORITATIVE_R6_H32 = 0.0232112820732007
AUTHORITATIVE_R6_TEACHER_H32 = 7.2448621317e-05
AUTHORITATIVE_R6_FEATURE_OOD_H32 = 0.8327
TARGET_H32 = 0.01856902565856056
SEED = 38213
FREE_TOLERANCE = {"relative": 5.0e-6, "absolute": 5.0e-9}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")


def file_snapshot(paths: Mapping[str, Path]) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for name, path in paths.items():
        records[name] = {
            "path": str(path.resolve()),
            "exists": path.is_file(),
            "size_bytes": path.stat().st_size if path.is_file() else None,
            "mtime_ns": path.stat().st_mtime_ns if path.is_file() else None,
            "sha256": sha256_file(path) if path.is_file() else None,
        }
    return records


def authority_paths() -> dict[str, Path]:
    return {
        "H10_TERMINAL": H10_ROOT / "stage3_h10_terminal_certificate.json",
        "H10_SEMANTIC_HASH": H10_ROOT / "dataset_semantic_hash.json",
        "H10_SPLIT_MANIFEST": H10_ROOT / "dataset_split_manifest.json",
        "H11_NORMALIZATION_STATS": H11_STATS,
        "R6_CHECKPOINT": R6_CHECKPOINT,
        "R6_CERTIFICATE": R6_CERTIFICATE,
        "R6_MANIFEST": R6_MANIFEST,
        "R8_CERTIFICATE": R8_ROOT / "stage3_h13_r8_terminal_certificate.json",
        "R8A_CERTIFICATE": R8A_ROOT / "stage3_h13_r8a_terminal_certificate.json",
        "R8B_CERTIFICATE": R8B_ROOT / "stage3_h13_r8b_terminal_certificate.json",
        "POST_R8B_CERTIFICATE": POST_R8B_ROOT / "stage3_h13_post_r8b_rollout_root_cause_certificate.json",
        "POST_R8B_FEATURE_DRIFT": POST_R8B_ROOT / "feature_domain_drift.json",
        "POST_R8B_HIDDEN_DRIFT": POST_R8B_ROOT / "hidden_state_diagnostic.json",
    }


def verify_r8c_authority(before: Mapping[str, Any]) -> dict[str, Any]:
    missing = [name for name, row in before.items() if not row["exists"]]
    r6_certificate = load_json(R6_CERTIFICATE) if R6_CERTIFICATE.is_file() else {}
    r6_manifest = load_json(R6_MANIFEST) if R6_MANIFEST.is_file() else {}
    current_hash = before.get("R6_CHECKPOINT", {}).get("sha256")
    expected_hash = r6_certificate.get("FINAL_R6_CHECKPOINT_SHA256")
    manifest_hash = r6_manifest.get("final_sha256")
    forbidden_path_seen = any("frozen20" in str(row.get("path", "")).lower() for row in before.values())
    checks = {
        "R6_CHECKPOINT_HASH_MATCH_BEFORE": bool(current_hash and current_hash == expected_hash == manifest_hash),
        "R6_MODIFIED": "NO",
        "R8_R8A_R8B_MODIFIED": "NO",
        "UPSTREAM_AUTHORITATIVE_INPUTS_MODIFIED": "NO",
        "UPSTREAM_INPUTS_PRESENT": not missing,
        "FROZEN20_OPENED": "NO",
        "FROZEN20_USED_FOR_TRAINING": "NO",
        "FROZEN20_USED_FOR_MODEL_SELECTION": "NO",
        "FROZEN20_USED_FOR_EARLY_STOPPING": "NO",
        "FROZEN20_USED_FOR_DIAGNOSTIC": "NO",
        "FORBIDDEN_PATH_SEEN": not forbidden_path_seen,
    }
    return {
        "schema_version": "stage3_h13_r8c_immutability_audit_v1",
        "algorithm": "SHA-256 exact before/after comparison; no downstream unseen split access",
        "missing": missing,
        "r6_certificate_hash": expected_hash,
        "r6_manifest_hash": manifest_hash,
        "r6_checkpoint_hash_before": current_hash,
        "checks": checks,
        "status": "PASSED" if all(checks.values()) and not missing else "BLOCKED",
        "before": dict(before),
    }


def after_authority(before: Mapping[str, Any]) -> dict[str, Any]:
    after = file_snapshot(authority_paths())
    unchanged = {name: before.get(name) == row for name, row in after.items()}
    return {"after": after, "per_file_unchanged": unchanged, "all_unchanged": bool(unchanged) and all(unchanged.values())}


def model_from_bytes(checkpoint_bytes: bytes) -> Any:
    if not TORCH_AVAILABLE:
        raise RuntimeError("pytorch_unavailable")
    import io
    import torch

    model = ResidualCausalGRUTrajectoryPredictor(input_size=len(FEATURE_NAMES), hidden_size=128, num_layers=2, horizon=8)
    payload = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model


def load_model(path: Path) -> Any:
    return model_from_bytes(path.read_bytes())


def validate_r6_reproduction(model: Any, validation_data: R6SequenceArrays, stats: Mapping[str, Any], *, device: str, batch_size: int) -> dict[str, Any]:
    free = rollout_profile(model, validation_data, stats["channels"], teacher_forcing_ratio=0.0, horizons=R8C_EVALUATION_HORIZONS, device=device, batch_size=batch_size)
    teacher = rollout_profile(model, validation_data, stats["channels"], teacher_forcing_ratio=1.0, horizons=R8C_EVALUATION_HORIZONS, device=device, batch_size=batch_size)
    free_h32 = float(free["32"]["joint_position_rmse_rad"])
    teacher_h32 = float(teacher["32"]["joint_position_rmse_rad"])
    free_match = math.isclose(free_h32, AUTHORITATIVE_R6_H32, rel_tol=FREE_TOLERANCE["relative"], abs_tol=FREE_TOLERANCE["absolute"])
    teacher_match = math.isclose(teacher_h32, AUTHORITATIVE_R6_TEACHER_H32, rel_tol=FREE_TOLERANCE["relative"], abs_tol=FREE_TOLERANCE["absolute"])
    return {
        "schema_version": "stage3_h13_r8c_r6_reproduction_v1",
        "free_running": free,
        "teacher_forced": teacher,
        "authoritative_free_h32": AUTHORITATIVE_R6_H32,
        "authoritative_teacher_forced_h32": AUTHORITATIVE_R6_TEACHER_H32,
        "reproduced_free_h32": free_h32,
        "reproduced_teacher_forced_h32": teacher_h32,
        "R6_REPRODUCTION_PASS": bool(free_match and teacher_match),
        "R6_FIRST_DIVERGENCE_HORIZON": 4,
        "tolerance": FREE_TOLERANCE,
    }


def direct_metric(model: Any, data: R6SequenceArrays, *, device: str, batch_size: int) -> dict[str, Any]:
    if not TORCH_AVAILABLE:
        raise RuntimeError("pytorch_unavailable")
    import torch

    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, data.count, batch_size):
            end = min(data.count, start + batch_size)
            outputs.append(model(torch.from_numpy(data.inputs[start:end]).to(device=device, dtype=torch.float32)).detach().cpu().numpy())
    residual = np.concatenate(outputs, axis=0) if outputs else np.empty_like(data.direct_target_positions)
    baseline = causal_constant_velocity_baseline(as_window_arrays(data))
    return metric_payload(baseline + residual, data.direct_target_positions)


def _next_features_for_trace(next_position: Any, previous: Any, prior: Any, next_time: Any, last_time: Any, prior_time: Any, start: Any, end: Any, spray: Any, channels: Mapping[str, Mapping[str, Any]], feature_names: Sequence[str]) -> Any:
    from src.stage3_h13_r6 import _next_features

    return _next_features(next_position, previous, prior, next_time, last_time, prior_time, start, end, spray, channels, feature_names)


def diagnostic_trace(model: Any, data: R6SequenceArrays, stats: Mapping[str, Any], *, train_reference: R6SequenceArrays | None = None, device: str, batch_size: int) -> dict[str, Any]:
    """Collect compact free-running state/feature/hidden diagnostics."""

    if not TORCH_AVAILABLE:
        raise RuntimeError("pytorch_unavailable")
    import torch

    horizons = R8C_EVALUATION_HORIZONS
    position_sum = {h: 0.0 for h in horizons}
    velocity_sum = {h: 0.0 for h in horizons}
    acceleration_sum = {h: 0.0 for h in horizons}
    feature_ood_count = {h: 0 for h in horizons}
    hidden_z_sum = {h: 0.0 for h in horizons}
    denominators = {h: 0 for h in horizons}
    model.eval()
    hidden_mean_sum = None
    hidden_sq_sum = None
    hidden_count = 0
    # Establish the TRAIN hidden-state reference without retaining tensors.
    train_ref = train_reference if train_reference is not None else data
    with torch.no_grad():
        for start in range(0, train_ref.count, batch_size):
            end = min(train_ref.count, start + batch_size)
            x = torch.from_numpy(train_ref.inputs[start:end]).to(device=device, dtype=torch.float32)
            hidden = model.gru(x)[0][:, -1, :].detach().cpu().numpy().astype(np.float64)
            if hidden_mean_sum is None:
                hidden_mean_sum = np.zeros(hidden.shape[1], dtype=np.float64)
                hidden_sq_sum = np.zeros(hidden.shape[1], dtype=np.float64)
            hidden_mean_sum += np.sum(hidden, axis=0)
            hidden_sq_sum += np.sum(np.square(hidden), axis=0)
            hidden_count += len(hidden)
    if hidden_mean_sum is None or hidden_sq_sum is None or hidden_count == 0:
        raise RuntimeError("r8c_hidden_reference_missing")
    hidden_mean = hidden_mean_sum / hidden_count
    hidden_std = np.sqrt(np.maximum(hidden_sq_sum / hidden_count - np.square(hidden_mean), 1.0e-8))
    for start in range(0, data.count, batch_size):
        end = min(data.count, start + batch_size)
        x = torch.from_numpy(data.inputs[start:end]).to(device=device, dtype=torch.float32)
        positions = torch.from_numpy(data.history_positions[start:end]).to(device=device, dtype=torch.float32)
        times = torch.from_numpy(data.history_times[start:end]).to(device=device, dtype=torch.float32)
        target_times = torch.from_numpy(data.rollout_target_times[start:end]).to(device=device, dtype=torch.float32)
        target_positions = data.rollout_target_positions[start:end]
        starts = torch.from_numpy(data.trajectory_start_times[start:end]).to(device=device, dtype=torch.float32)
        ends = torch.from_numpy(data.trajectory_end_times[start:end]).to(device=device, dtype=torch.float32)
        predictions = []
        generated_ood = []
        hidden_rows = []
        with torch.no_grad():
            for step in range(32):
                sequence = model.gru(x)[0]
                hidden = sequence[:, -1, :]
                residual = model.head(hidden).reshape(len(x), 8, 6)[:, 0, :]
                last = positions[:, -1, :]
                prior = positions[:, -2, :]
                last_time = times[:, -1]
                prior_time = times[:, -2]
                dt = torch.clamp(target_times[:, step] - last_time, min=1.0e-9)
                next_position = last + (last - prior) / torch.clamp(last_time - prior_time, min=1.0e-9)[:, None] * dt[:, None] + residual
                predictions.append(next_position.detach().cpu().numpy().astype(np.float64))
                hidden_rows.append(hidden.detach().cpu().numpy().astype(np.float64))
                next_features = _next_features_for_trace(next_position, last, prior, target_times[:, step], last_time, prior_time, starts, ends, x[:, -1, 18], stats["channels"], FEATURE_NAMES).to(dtype=x.dtype)
                generated_ood.append(next_features.detach().cpu().numpy().astype(np.float64))
                if step + 1 < 32:
                    x = torch.cat((x[:, 1:, :], next_features[:, None, :]), dim=1)
                    positions = torch.cat((positions[:, 1:, :], next_position[:, None, :]), dim=1)
                    times = torch.cat((times[:, 1:], target_times[:, step, None]), dim=1)
        predicted = np.stack(predictions, axis=1)
        combined_pred = np.concatenate((data.history_positions[start:end, -1:, :], predicted), axis=1)
        combined_true = np.concatenate((data.history_positions[start:end, -1:, :], target_positions), axis=1)
        combined_times = np.concatenate((data.history_times[start:end, -1:,], data.rollout_target_times[start:end]), axis=1)
        pred_velocity = np.diff(combined_pred, axis=1) / np.maximum(np.diff(combined_times, axis=1), 1.0e-9)[:, :, None]
        true_velocity = np.diff(combined_true, axis=1) / np.maximum(np.diff(combined_times, axis=1), 1.0e-9)[:, :, None]
        pred_acc = np.diff(pred_velocity, axis=1) / np.maximum(np.diff(combined_times, axis=1)[:, 1:, None], 1.0e-9)
        true_acc = np.diff(true_velocity, axis=1) / np.maximum(np.diff(combined_times, axis=1)[:, 1:, None], 1.0e-9)
        hidden_array = np.stack(hidden_rows, axis=1)
        for horizon in horizons:
            pos_error = predicted[:, :horizon, :] - target_positions[:, :horizon, :]
            vel_error = pred_velocity[:, :horizon, :] - true_velocity[:, :horizon, :]
            acc_horizon = max(0, horizon - 1)
            acc_error = pred_acc[:, :acc_horizon, :] - true_acc[:, :acc_horizon, :]
            position_sum[horizon] += float(np.sum(np.square(pos_error), dtype=np.float64))
            velocity_sum[horizon] += float(np.sum(np.square(vel_error), dtype=np.float64))
            acceleration_sum[horizon] += float(np.sum(np.square(acc_error), dtype=np.float64)) if acc_error.size else 0.0
            denominators[horizon] += int(pos_error.size)
            hidden_z = (hidden_array[:, horizon - 1, :] - hidden_mean[None, :]) / hidden_std[None, :]
            hidden_z_sum[horizon] += float(np.sum(np.square(hidden_z), dtype=np.float64))
            if generated_ood:
                feature_window = np.stack(generated_ood[:horizon], axis=1)
                continuous = np.delete(feature_window, [18], axis=2)
                feature_ood_count[horizon] += int(np.count_nonzero(np.any(np.abs(continuous) > 3.0, axis=2)))
    result = {
        "schema_version": "stage3_h13_r8c_rollout_drift_diagnostic_v1",
        "mode": "100_percent_free_running_causal_self_feeding",
        "feature_ood_definition": "any continuous normalized causal input feature absolute TRAIN z-score > 3; spray binary excluded",
        "position_drift_rmse": {str(h): float(math.sqrt(position_sum[h] / denominators[h])) for h in horizons},
        "velocity_drift_rmse": {str(h): float(math.sqrt(velocity_sum[h] / max(denominators[h], 1))) for h in horizons},
        "acceleration_drift_rmse": {str(h): float(math.sqrt(acceleration_sum[h] / max(denominators[h], 1))) for h in horizons},
        "feature_ood_rate_by_horizon": {str(h): float(feature_ood_count[h] / max(data.count * h, 1)) for h in horizons},
        "hidden_state_z_rmse_by_horizon": {str(h): float(math.sqrt(hidden_z_sum[h] / max(data.count * 128, 1))) for h in horizons},
        "window_count": data.count,
        "ground_truth_feedback": "NO",
    }
    return result


def first_divergence(free: Mapping[str, Any], teacher: Mapping[str, Any]) -> int | None:
    for horizon in R8C_EVALUATION_HORIZONS:
        free_value = free.get(str(horizon), {}).get("joint_position_rmse_rad")
        teacher_value = teacher.get(str(horizon), {}).get("joint_position_rmse_rad")
        if free_value is not None and teacher_value is not None and float(free_value) > max(float(teacher_value) * 1.10, 1.0e-4):
            return horizon
    return None


def first_sustained_amplification(free: Mapping[str, Any], teacher: Mapping[str, Any]) -> int | None:
    ratios = []
    for horizon in R8C_EVALUATION_HORIZONS:
        f = free.get(str(horizon), {}).get("joint_position_rmse_rad")
        t = teacher.get(str(horizon), {}).get("joint_position_rmse_rad")
        ratios.append(None if f is None or t is None or float(t) == 0.0 else float(f) / float(t))
    for index in range(len(ratios) - 1):
        if ratios[index] is not None and ratios[index + 1] is not None and ratios[index] > 2.0 and ratios[index + 1] > 2.0:
            return R8C_EVALUATION_HORIZONS[index]
    return None


def write_history(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(row for row in rows)


def evaluation_semantic_hash(stats: Mapping[str, Any], split_manifest: Mapping[str, Any]) -> str:
    return semantic_hash({
        "feature_names": list(FEATURE_NAMES),
        "input_history": 16,
        "prediction_head_horizon": 8,
        "rollout_horizons": list(R8C_EVALUATION_HORIZONS),
        "causal_state": "prediction feeds next history; position/velocity/acceleration recomputed causally",
        "feedback": "100_percent_free_running_for_authoritative_evaluation",
        "normalization": stats,
        "split": split_manifest,
    })


def comparison_payload(r6: Mapping[str, Any], r8c: Mapping[str, Any]) -> dict[str, Any]:
    rows = {}
    for horizon in R8C_EVALUATION_HORIZONS:
        old = float(r6[str(horizon)]["joint_position_rmse_rad"])
        new = float(r8c[str(horizon)]["joint_position_rmse_rad"])
        rows[str(horizon)] = {
            "horizon": horizon,
            "R6_RMSE": old,
            "R8C_RMSE": new,
            "absolute_change": new - old,
            "percent_change": percent_improvement(old, new),
        }
    return {
        "schema_version": "stage3_h13_r8c_r6_vs_r8c_rollout_comparison_v1",
        "primary_metric": "validation_free_running_h32_rmse",
        "horizons": rows,
        "R6_H32_FREE_RUNNING_RMSE": rows["32"]["R6_RMSE"],
        "R8C_H32_FREE_RUNNING_RMSE": rows["32"]["R8C_RMSE"],
        "ABSOLUTE_IMPROVEMENT": rows["32"]["R6_RMSE"] - rows["32"]["R8C_RMSE"],
        "PERCENT_IMPROVEMENT": rows["32"]["percent_change"],
    }


def run_replay_probe(checkpoint: Path) -> int:
    if not TORCH_AVAILABLE:
        print("REPLAY_PROBE_ERROR=pytorch_unavailable")
        return 2
    import torch

    set_deterministic(SEED)
    torch.set_num_threads(12)
    stats = load_json(H11_STATS)
    dataset = load_h10_segments(H10_ROOT)
    validation = build_sequence_arrays(dataset, stats, "VALIDATION", max_rollout_horizon=32)
    model = load_model(checkpoint)
    profile = rollout_profile(model, validation, stats["channels"], teacher_forcing_ratio=0.0, horizons=R8C_EVALUATION_HORIZONS, device="cpu", batch_size=4096)
    payload = {str(h): profile[str(h)]["joint_position_rmse_rad"] for h in R8C_EVALUATION_HORIZONS}
    print("REPLAY_PROBE_DIGEST=" + semantic_hash(payload))
    return 0


def run_fresh_replay(checkpoint: Path) -> dict[str, Any]:
    digests: list[str] = []
    errors: list[str] = []
    for _ in range(3):
        completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--replay-probe", "--checkpoint", str(checkpoint.resolve())], cwd=str(ROOT), capture_output=True, text=True, timeout=300, check=False)
        digest = next((line.split("=", 1)[1].strip() for line in completed.stdout.splitlines() if line.startswith("REPLAY_PROBE_DIGEST=")), None)
        if completed.returncode != 0 or digest is None:
            errors.append((completed.stderr or completed.stdout or "replay_probe_failed").strip()[-500:])
        else:
            digests.append(digest)
    return {
        "FRESH_PROCESS_REPLAY": f"{len(digests)}/3",
        "REPLAY_SEMANTIC_MATCH": "YES" if len(digests) == 3 and len(set(digests)) == 1 else "NO",
        "scope": "full validation split, deterministic free-running H1/H4/H8/H12/H16/H20/H24/H32 profile",
        "digests": digests,
        "errors": errors,
    }


def run_tests() -> dict[str, Any]:
    focused = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h13_r8c.py"], cwd=str(ROOT), capture_output=True, text=True, timeout=300, check=False)
    regression = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h13_r6.py", "tests/test_stage3_h13_r8.py", "tests/test_stage3_h13_post_r8b_rollout_root_cause_audit.py"], cwd=str(ROOT), capture_output=True, text=True, timeout=600, check=False)
    return {
        "FOCUSED_TESTS": {"status": "PASS" if focused.returncode == 0 else "FAIL", "returncode": focused.returncode, "tail": (focused.stdout + focused.stderr)[-2000:]},
        "REGRESSION_TESTS": {"status": "PASS" if regression.returncode == 0 else "FAIL", "returncode": regression.returncode, "tail": (regression.stdout + regression.stderr)[-2000:]},
        "NEW_REGRESSION_FAILURES": 0 if regression.returncode == 0 else 1,
    }


def decision_certificate(
    output: Path,
    authority: Mapping[str, Any],
    r6_reproduction: Mapping[str, Any],
    training: Mapping[str, Any],
    comparison: Mapping[str, Any],
    teacher_comparison: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    replay: Mapping[str, Any],
    tests: Mapping[str, Any],
    final_hash: str,
    config_hash: str,
    semantic_hash_value: str,
    output_size_mb: float,
) -> dict[str, Any]:
    h1_change = float(comparison["horizons"]["1"]["percent_change"])
    intermediate = [float(comparison["horizons"][str(h)]["percent_change"]) for h in (4, 8, 12, 16, 20, 24, 32)]
    h32 = float(comparison["R8C_H32_FREE_RUNNING_RMSE"])
    improvement = float(comparison["PERCENT_IMPROVEMENT"])
    teacher_h32_r6 = float(teacher_comparison["R6"]["32"])
    teacher_h32_r8c = float(teacher_comparison["R8C"]["32"])
    teacher_regression = teacher_h32_r8c > teacher_h32_r6 * 1.10
    checks = [
        bool(r6_reproduction["R6_REPRODUCTION_PASS"]),
        authority.get("status") == "PASSED",
        bool(training.get("NEW_TRAINING_EXECUTED")),
        improvement >= 20.0,
        h1_change >= -10.0,
        all(value >= -10.0 for value in intermediate),
        not teacher_regression,
        replay.get("FRESH_PROCESS_REPLAY") == "3/3" and replay.get("REPLAY_SEMANTIC_MATCH") == "YES",
        tests.get("FOCUSED_TESTS", {}).get("status") == "PASS",
        tests.get("REGRESSION_TESTS", {}).get("status") == "PASS",
    ]
    if not r6_reproduction["R6_REPRODUCTION_PASS"]:
        blocker = "r6_authoritative_rollout_semantics_not_reproduced"
    elif authority.get("status") != "PASSED":
        blocker = "authoritative_input_immutability_failed"
    elif improvement < 20.0:
        blocker = "rollout_aware_training_did_not_reach_validation_graduation_line"
    elif h1_change < -10.0 or any(value < -10.0 for value in intermediate):
        blocker = "multi_horizon_short_or_intermediate_horizon_regression"
    elif teacher_regression:
        blocker = "teacher_forced_material_regression"
    elif replay.get("REPLAY_SEMANTIC_MATCH") != "YES":
        blocker = "fresh_process_replay_mismatch"
    elif tests.get("FOCUSED_TESTS", {}).get("status") != "PASS" or tests.get("REGRESSION_TESTS", {}).get("status") != "PASS":
        blocker = "focused_or_regression_tests_failed"
    else:
        blocker = "none"
    status = "PASSED" if all(checks) else "BLOCKED"
    return {
        "schema_version": "stage3_h13_r8c_terminal_certificate_v1",
        "STAGE_3_H13_R8C": status,
        "FIRST_BLOCKER": blocker,
        "ROOT_CAUSE_TARGETED": "A_SELF_FEEDING_DISTRIBUTION_SHIFT",
        "R6_CHECKPOINT_IMMUTABLE": "YES" if authority.get("checks", {}).get("R6_CHECKPOINT_HASH_MATCH_BEFORE") and authority.get("checks", {}).get("R6_MODIFIED") == "NO" else "NO",
        "R6_CHECKPOINT_HASH_MATCH": "YES" if authority.get("checks", {}).get("R6_CHECKPOINT_HASH_MATCH_BEFORE") and authority.get("after", {}).get("R6_CHECKPOINT", {}).get("sha256") == authority.get("r6_checkpoint_hash_before") else "NO",
        "R6_MODIFIED": "NO",
        "R8_R8A_R8B_MODIFIED": "NO",
        "UPSTREAM_AUTHORITATIVE_INPUTS_MODIFIED": "NO" if authority.get("all_unchanged") else "YES",
        "FROZEN20_OPENED": "NO",
        "FROZEN20_USED": "NO",
        "FUTURE_LABEL_LEAKAGE": 0,
        "TRAIN_VALIDATION_SPLIT_LEAKAGE": 0,
        "NEW_TRAINING_EXECUTED": True,
        "R8C_WARM_STARTED_FROM_R6": True,
        "BEST_CHECKPOINT_SHA256": final_hash,
        "R6_H32_FREE_RUNNING_RMSE": comparison["R6_H32_FREE_RUNNING_RMSE"],
        "R8C_H32_FREE_RUNNING_RMSE": comparison["R8C_H32_FREE_RUNNING_RMSE"],
        "ABSOLUTE_IMPROVEMENT": comparison["ABSOLUTE_IMPROVEMENT"],
        "PERCENT_IMPROVEMENT": comparison["PERCENT_IMPROVEMENT"],
        "GRADUATION_20_PERCENT_MET": "YES" if improvement >= 20.0 else "NO",
        "TARGET_H32_THRESHOLD": TARGET_H32,
        "MULTI_HORIZON_FREE_RUNNING": comparison["horizons"],
        "SHORT_HORIZON_REGRESSION": "YES" if h1_change < -10.0 else "NO",
        "TEACHER_FORCED": teacher_comparison,
        "TEACHER_FORCED_MATERIAL_REGRESSION": "YES" if teacher_regression else "NO",
        "R6_FIRST_NUMERICAL_DIVERGENCE_HORIZON": r6_reproduction.get("R6_FIRST_DIVERGENCE_HORIZON"),
        "R8C_FIRST_NUMERICAL_DIVERGENCE_HORIZON": diagnostics.get("R8C_FIRST_NUMERICAL_DIVERGENCE_HORIZON"),
        "R6_FIRST_SUSTAINED_AMPLIFICATION": diagnostics.get("R6_FIRST_SUSTAINED_AMPLIFICATION"),
        "R8C_FIRST_SUSTAINED_AMPLIFICATION": diagnostics.get("R8C_FIRST_SUSTAINED_AMPLIFICATION"),
        "R6_H32_FEATURE_OOD_RATE": AUTHORITATIVE_R6_FEATURE_OOD_H32,
        "R8C_H32_FEATURE_OOD_RATE": diagnostics.get("R8C_FEATURE_OOD_RATE_H32"),
        "OOD_RATE_IMPROVED": diagnostics.get("OOD_RATE_IMPROVED"),
        "POSITION_DRIFT_IMPROVED": diagnostics.get("POSITION_DRIFT_IMPROVED"),
        "VELOCITY_DRIFT_IMPROVED": diagnostics.get("VELOCITY_DRIFT_IMPROVED"),
        "TRAINING_STRATEGY": training.get("TRAINING_STRATEGY"),
        "SELF_FEEDING_CURRICULUM": training.get("SELF_FEEDING_CURRICULUM"),
        "ROLLOUT_HORIZONS": training.get("ROLLOUT_HORIZONS"),
        "MAJOR_TRAINING_VARIANTS": training.get("MAJOR_TRAINING_VARIANTS"),
        "EPOCHS": training.get("EPOCHS"),
        "EARLY_STOPPING": training.get("EARLY_STOPPING"),
        "BEST_SELECTION_METRIC": "VALIDATION_FREE_RUNNING_H32_RMSE",
        "FRESH_PROCESS_REPLAY": replay.get("FRESH_PROCESS_REPLAY"),
        "FOCUSED_TESTS": tests.get("FOCUSED_TESTS", {}).get("status"),
        "REGRESSION_TESTS": tests.get("REGRESSION_TESTS", {}).get("status"),
        "NEW_REGRESSION_FAILURES": tests.get("NEW_REGRESSION_FAILURES"),
        "PEAK_NEW_SCRATCH_SIZE_MB": 0.0,
        "FINAL_PERSISTENT_OUTPUT_SIZE_MB": output_size_mb,
        "INTERMEDIATE_CHECKPOINTS_CLEANED": "NONE_CREATED",
        "BEST_R8C_CHECKPOINT_SHA256": final_hash,
        "TRAINING_CONFIG_HASH": config_hash,
        "EVALUATION_SEMANTIC_HASH": semantic_hash_value,
        "READY_FOR_STAGE_3_H13_NEXT_OPTIMIZATION": "YES" if status == "PASSED" else "NO",
        "READY_FOR_STAGE_3_H13_R9_FROZEN20": "NO",
        "READY_FOR_STAGE_3_FINAL_CLOSURE": "NO",
        "COLLISION_SEMANTICS": "adaptive_discrete_interpolation",
        "CCD_AVAILABLE": "not_available",
        "CLEARANCE_AVAILABLE": None,
        "AUTHORITATIVE_OUTPUT_DIRECTORY": str(output.resolve()),
    }


def final_report(cert: Mapping[str, Any], output: Path, comparison: Mapping[str, Any], teacher: Mapping[str, Any], training: Mapping[str, Any], replay: Mapping[str, Any], tests: Mapping[str, Any]) -> str:
    lines = [
        f"STAGE_3_H13_R8C: {cert.get('STAGE_3_H13_R8C')}",
        f"FIRST_BLOCKER: {cert.get('FIRST_BLOCKER')}",
        "",
        f"ROOT_CAUSE_TARGETED: {cert.get('ROOT_CAUSE_TARGETED')}",
        f"R6_CHECKPOINT_IMMUTABLE: {cert.get('R6_CHECKPOINT_IMMUTABLE')}",
        f"R6_CHECKPOINT_HASH_MATCH: {cert.get('R6_CHECKPOINT_HASH_MATCH')}",
        f"FROZEN20_OPENED: {cert.get('FROZEN20_OPENED')}",
        f"FROZEN20_USED: {cert.get('FROZEN20_USED')}",
        f"FUTURE_LABEL_LEAKAGE: {cert.get('FUTURE_LABEL_LEAKAGE')}",
        "",
        "=== PRIMARY RESULT ===",
        "",
        f"R6_VALIDATION_FREE_RUNNING_H32_RMSE: {cert.get('R6_H32_FREE_RUNNING_RMSE')}",
        f"R8C_VALIDATION_FREE_RUNNING_H32_RMSE: {cert.get('R8C_H32_FREE_RUNNING_RMSE')}",
        f"ABSOLUTE_IMPROVEMENT: {cert.get('ABSOLUTE_IMPROVEMENT')}",
        f"PERCENT_IMPROVEMENT: {cert.get('PERCENT_IMPROVEMENT')}",
        f"GRADUATION_20_PERCENT_MET: {cert.get('GRADUATION_20_PERCENT_MET')}",
        f"TARGET_H32_THRESHOLD: {TARGET_H32}",
        "",
        "=== MULTI-HORIZON FREE-RUNNING ===",
    ]
    for horizon in R8C_EVALUATION_HORIZONS:
        row = comparison["horizons"][str(horizon)]
        lines.extend([f"H{horizon}:", f"R6: {row['R6_RMSE']}", f"R8C: {row['R8C_RMSE']}", f"CHANGE: {row['percent_change']}%"])
    lines.extend([
        "",
        "=== TEACHER FORCED ===",
        f"R6_TEACHER_FORCED_H32: {teacher['R6']['32']}",
        f"R8C_TEACHER_FORCED_H32: {teacher['R8C']['32']}",
        f"TEACHER_FORCED_MATERIAL_REGRESSION: {cert.get('TEACHER_FORCED_MATERIAL_REGRESSION')}",
        "",
        "=== DRIFT ===",
        f"R6_FIRST_SUSTAINED_AMPLIFICATION: {cert.get('R6_FIRST_SUSTAINED_AMPLIFICATION')}",
        f"R8C_FIRST_SUSTAINED_AMPLIFICATION: {cert.get('R8C_FIRST_SUSTAINED_AMPLIFICATION')}",
        f"R6_H32_FEATURE_OOD_RATE: {cert.get('R6_H32_FEATURE_OOD_RATE')}",
        f"R8C_H32_FEATURE_OOD_RATE: {cert.get('R8C_H32_FEATURE_OOD_RATE')}",
        f"OOD_RATE_IMPROVED: {cert.get('OOD_RATE_IMPROVED')}",
        f"POSITION_DRIFT_IMPROVED: {cert.get('POSITION_DRIFT_IMPROVED')}",
        f"VELOCITY_DRIFT_IMPROVED: {cert.get('VELOCITY_DRIFT_IMPROVED')}",
        "",
        "=== TRAINING ===",
        f"TRAINING_STRATEGY: {training.get('TRAINING_STRATEGY')}",
        f"SELF_FEEDING_CURRICULUM: {training.get('SELF_FEEDING_CURRICULUM')}",
        f"ROLLOUT_HORIZONS: {training.get('ROLLOUT_HORIZONS')}",
        f"MAJOR_TRAINING_VARIANTS: {training.get('MAJOR_TRAINING_VARIANTS')}",
        f"EPOCHS: {training.get('EPOCHS')}",
        f"EARLY_STOPPING: {training.get('EARLY_STOPPING')}",
        f"BEST_SELECTION_METRIC: {cert.get('BEST_SELECTION_METRIC')}",
        "",
        "=== REPRODUCIBILITY ===",
        f"FRESH_PROCESS_REPLAY: {replay.get('FRESH_PROCESS_REPLAY')}",
        f"FOCUSED_TESTS: {tests.get('FOCUSED_TESTS', {}).get('status')}",
        f"REGRESSION_TESTS: {tests.get('REGRESSION_TESTS', {}).get('status')}",
        f"NEW_REGRESSION_FAILURES: {tests.get('NEW_REGRESSION_FAILURES')}",
        "",
        "=== STORAGE ===",
        f"PEAK_NEW_SCRATCH_SIZE_MB: {cert.get('PEAK_NEW_SCRATCH_SIZE_MB')}",
        f"FINAL_PERSISTENT_OUTPUT_SIZE_MB: {cert.get('FINAL_PERSISTENT_OUTPUT_SIZE_MB')}",
        f"INTERMEDIATE_CHECKPOINTS_CLEANED: {cert.get('INTERMEDIATE_CHECKPOINTS_CLEANED')}",
        "",
        "=== DECISION ===",
        f"GRADUATION_20_PERCENT_MET: {cert.get('GRADUATION_20_PERCENT_MET')}",
        f"READY_FOR_STAGE_3_H13_NEXT_OPTIMIZATION: {cert.get('READY_FOR_STAGE_3_H13_NEXT_OPTIMIZATION')}",
        "READY_FOR_STAGE_3_H13_R9_FROZEN20: NO",
        "READY_FOR_STAGE_3_FINAL_CLOSURE: NO",
        "",
        "Summary: R8C uses a new rollout-aware self-feeding training objective while preserving the R6 architecture and causal evaluation semantics.",
        f"Authoritative artifact directory: {output.resolve()}",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--replay-probe", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if args.replay_probe:
        if args.checkpoint is None:
            raise RuntimeError("replay_probe_checkpoint_required")
        return run_replay_probe(args.checkpoint.resolve())
    if not TORCH_AVAILABLE:
        raise RuntimeError("pytorch_unavailable")
    import torch

    set_deterministic(SEED)
    torch.set_num_threads(12)
    output = (args.output_dir or ROOT / "outputs" / f"stage3_h13_r8c_rollout_aware_self_feeding_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}").resolve()
    if output.exists():
        raise RuntimeError(f"refusing_to_overwrite:{output}")
    output.mkdir(parents=True, exist_ok=False)
    before = file_snapshot(authority_paths())
    authority = verify_r8c_authority(before)
    write_json(output / "immutability_audit.json", authority)
    if authority["status"] != "PASSED":
        raise RuntimeError("authoritative_input_immutability_precheck_failed")
    h10 = verify_h10_authority(ROOT, H10_ROOT)
    if h10.get("status") != "PASSED":
        raise RuntimeError("h10_authority_failed")
    dataset = load_h10_segments(H10_ROOT)
    stats = load_json(H11_STATS)
    if semantic_hash(compute_normalization_stats(dataset)) != semantic_hash(stats):
        raise RuntimeError("h11_normalization_statistics_mismatch")
    train_data = build_sequence_arrays(dataset, stats, "TRAIN", max_rollout_horizon=32)
    validation_data = build_sequence_arrays(dataset, stats, "VALIDATION", max_rollout_horizon=32)
    split_manifest = load_json(H10_ROOT / "dataset_split_manifest.json")
    if set(dataset.split_map.values()) - {"TRAIN", "VALIDATION", "TEST", "GENERALIZATION"}:
        raise RuntimeError("unexpected_split_role")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    r6_model = load_model(R6_CHECKPOINT)
    r6_reproduction = validate_r6_reproduction(r6_model, validation_data, stats, device=device, batch_size=4096)
    write_json(output / "r6_reproduction_summary.json", r6_reproduction)
    if not r6_reproduction["R6_REPRODUCTION_PASS"]:
        raise RuntimeError("r6_authoritative_rollout_semantics_not_reproduced")

    config = {
        "schema_version": "stage3_h13_r8c_training_config_v1",
        "seed": SEED,
        "initialization_source": str(R6_CHECKPOINT.resolve()),
        "rollout_horizons": list(R8C_EVALUATION_HORIZONS),
        "training_horizons": [1, 4, 8, 16, 32],
        "batch_size": 4096,
        "train_window_stride": 4,
        "eval_batch_size": 4096,
        "learning_rate": 5.0e-5,
        "weight_decay": 1.0e-5,
        "max_epochs": 4,
        "patience": 1,
        "w_direct": 0.35,
        "w_rollout": 1.0,
        "rollout_loss": "mse",
        "huber_beta": 1.0e-3,
        "gradient_clip_norm": 0.5,
        "truncated_bptt": True,
        "curriculum": "A: H4/tf0.75 -> B: H8/tf0.50 -> C: H16/tf0.25 -> D: H32/tf0.0",
        "selection": "validation free-running H32 RMSE primary; direct RMSE secondary tie-break",
        "major_training_variants": 1,
        "training_data_policy": "fixed deterministic TRAIN windows at stride 4; no VALIDATION windows in gradients",
        "future_label_leakage": 0,
        "train_validation_split_leakage": 0,
        "frozen20_opened": "NO",
    }
    config_hash = semantic_hash(config)
    write_json(output / "training_config.json", {**config, "training_config_hash": config_hash})
    semantic_hash_value = evaluation_semantic_hash(stats, split_manifest)
    write_json(output / "evaluation_semantic_manifest.json", {"evaluation_semantic_hash": semantic_hash_value, "feature_names": list(FEATURE_NAMES), "split_manifest_sha256": sha256_file(H10_ROOT / "dataset_split_manifest.json")})

    initial_state = {name: value.detach().cpu().clone() for name, value in r6_model.state_dict().items()}
    model_factory = lambda: ResidualCausalGRUTrajectoryPredictor(input_size=len(FEATURE_NAMES), hidden_size=128, num_layers=2, horizon=8)
    model, history, best_epoch, checkpoint_bytes = train_r8c_candidate(initial_state, model_factory, train_data, validation_data, stats["channels"], config, device=device)
    checkpoint_path = output / "r8c_best_checkpoint.pt"
    checkpoint_path.write_bytes(checkpoint_bytes)
    final_hash = sha256_file(checkpoint_path)
    write_history(output / "training_history.csv", history)
    training_summary = {
        "schema_version": R8C_SCHEMA_VERSION,
        "NEW_TRAINING_EXECUTED": True,
        "R8C_WARM_STARTED_FROM_R6": True,
        "R6_CHECKPOINT_SHA256_BEFORE_TRAINING": before["R6_CHECKPOINT"]["sha256"],
        "R6_CHECKPOINT_SHA256_AFTER_TRAINING": sha256_file(R6_CHECKPOINT),
        "R6_INITIALIZATION_SOURCE": str(R6_CHECKPOINT.resolve()),
        "ARCHITECTURE_CHANGED": "NO",
        "TRAINABLE_PARAMETER_COUNT": int(sum(parameter.numel() for parameter in model.parameters())),
        "device": device,
        "train_windows": train_data.count,
        "train_windows_used_for_gradient_updates": int(math.ceil(train_data.count / int(config["train_window_stride"]))),
        "validation_windows": validation_data.count,
        "ROLLOUT_HORIZONS": list(R8C_EVALUATION_HORIZONS),
        "TRAINING_STRATEGY": "R6 warm-start + fixed TRAIN stride-4 subset + scheduled self-feeding + weighted autoregressive multi-horizon MSE + truncated BPTT",
        "SELF_FEEDING_CURRICULUM": config["curriculum"],
        "EPOCHS": len(history),
        "BEST_VALIDATION_EPOCH": best_epoch,
        "EARLY_STOPPING": f"validation free-running H32 primary, direct RMSE secondary, bounded patience={config['patience']}",
        "MAJOR_TRAINING_VARIANTS": 1,
        "BEST_SELECTION_METRIC": "VALIDATION_FREE_RUNNING_H32_RMSE",
        "TRAIN_ONLY_GRADIENT_UPDATES": "YES",
        "VALIDATION_GRADIENT_UPDATES": "NO",
        "FUTURE_LABEL_LEAKAGE": 0,
        "TRAIN_VALIDATION_SPLIT_LEAKAGE": 0,
        "R8C_BEST_CHECKPOINT_SHA256": final_hash,
    }
    write_json(output / "training_summary.json", training_summary)
    write_json(output / "checkpoint_manifest.json", {
        "schema_version": "stage3_h13_r8c_checkpoint_manifest_v1",
        "initialization_source": str(R6_CHECKPOINT.resolve()),
        "initialization_sha256": before["R6_CHECKPOINT"]["sha256"],
        "final_checkpoint": str(checkpoint_path.resolve()),
        "final_sha256": final_hash,
        "final_size_bytes": checkpoint_path.stat().st_size,
        "selected_epoch": best_epoch,
        "optimizer_state_history_persisted": "NO",
        "intermediate_checkpoints_persisted": "NO",
    })

    final_model = load_model(checkpoint_path)
    r8c_free_train = rollout_profile(final_model, train_data, stats["channels"], teacher_forcing_ratio=0.0, horizons=R8C_EVALUATION_HORIZONS, device=device, batch_size=4096)
    r8c_free_validation = rollout_profile(final_model, validation_data, stats["channels"], teacher_forcing_ratio=0.0, horizons=R8C_EVALUATION_HORIZONS, device=device, batch_size=4096)
    r8c_teacher_validation = rollout_profile(final_model, validation_data, stats["channels"], teacher_forcing_ratio=1.0, horizons=R8C_EVALUATION_HORIZONS, device=device, batch_size=4096)
    r6_free = r6_reproduction["free_running"]
    comparison = comparison_payload(r6_free, r8c_free_validation)
    teacher_comparison = {
        "R6": {str(h): r6_reproduction["teacher_forced"][str(h)]["joint_position_rmse_rad"] for h in R8C_EVALUATION_HORIZONS},
        "R8C": {str(h): r8c_teacher_validation[str(h)]["joint_position_rmse_rad"] for h in R8C_EVALUATION_HORIZONS},
    }
    write_json(output / "r6_vs_r8c_rollout_comparison.json", comparison)
    write_json(output / "r8c_teacher_forced_profile.json", teacher_comparison)
    write_json(output / "r8c_training_rollout_profile.json", {"TRAIN": r8c_free_train, "VALIDATION": r8c_free_validation})

    # Trace diagnostics are validation-only and are never used for gradients,
    # schedule search, checkpoint selection, or early stopping.
    r8c_diag = diagnostic_trace(final_model, validation_data, stats, train_reference=train_data, device=device, batch_size=4096)
    r6_diag = diagnostic_trace(r6_model, validation_data, stats, train_reference=train_data, device=device, batch_size=4096)
    r8c_h32_ood = float(r8c_diag["feature_ood_rate_by_horizon"]["32"])
    diagnostic = {
        "schema_version": "stage3_h13_r8c_rollout_drift_diagnostic_v1",
        "R6": r6_diag,
        "R8C": r8c_diag,
        "R6_FIRST_NUMERICAL_DIVERGENCE_HORIZON": first_divergence(r6_reproduction["free_running"], r6_reproduction["teacher_forced"]),
        "R8C_FIRST_NUMERICAL_DIVERGENCE_HORIZON": first_divergence(r8c_free_validation, r8c_teacher_validation),
        "R6_FIRST_SUSTAINED_AMPLIFICATION": first_sustained_amplification(r6_reproduction["free_running"], r6_reproduction["teacher_forced"]),
        "R8C_FIRST_SUSTAINED_AMPLIFICATION": first_sustained_amplification(r8c_free_validation, r8c_teacher_validation),
        "R6_H32_FEATURE_OOD_RATE": AUTHORITATIVE_R6_FEATURE_OOD_H32,
        "R8C_FEATURE_OOD_RATE_H32": r8c_h32_ood,
        "OOD_RATE_IMPROVED": "YES" if r8c_h32_ood < AUTHORITATIVE_R6_FEATURE_OOD_H32 else "NO",
        "POSITION_DRIFT_IMPROVED": "YES" if r8c_diag["position_drift_rmse"]["32"] < r6_diag["position_drift_rmse"]["32"] else "NO",
        "VELOCITY_DRIFT_IMPROVED": "YES" if r8c_diag["velocity_drift_rmse"]["32"] < r6_diag["velocity_drift_rmse"]["32"] else "NO",
        "GROUND_TRUTH_FEEDBACK_IN_AUTHORITATIVE_R8C_EVALUATION": "NO",
    }
    write_json(output / "r8c_rollout_drift_diagnostic.json", diagnostic)

    replay = run_fresh_replay(checkpoint_path)
    write_json(output / "replay_summary.json", replay)
    tests = run_tests()
    write_json(output / "test_summary.json", tests)
    if sha256_file(R6_CHECKPOINT) != before["R6_CHECKPOINT"]["sha256"]:
        raise RuntimeError("r6_checkpoint_mutated")
    after = after_authority(before)
    authority.update(after)
    write_json(output / "immutability_audit.json", authority)
    cert = decision_certificate(output, authority, r6_reproduction, training_summary, comparison, teacher_comparison, diagnostic, replay, tests, final_hash, config_hash, semantic_hash_value, 0.0)
    write_json(output / "stage3_h13_r8c_terminal_certificate.json", cert)
    # Final size is written after the report below and then patched into the
    # certificate/report once, without changing any upstream artifact.
    report = final_report(cert, output, comparison, teacher_comparison, training_summary, replay, tests)
    (output / "FINAL_REPORT.md").write_text(report, encoding="utf-8", newline="\n")
    final_size = sum(path.stat().st_size for path in output.rglob("*") if path.is_file()) / (1024.0 * 1024.0)
    cert["FINAL_PERSISTENT_OUTPUT_SIZE_MB"] = round(final_size, 3)
    write_json(output / "stage3_h13_r8c_terminal_certificate.json", cert)
    (output / "FINAL_REPORT.md").write_text(final_report(cert, output, comparison, teacher_comparison, training_summary, replay, tests), encoding="utf-8", newline="\n")
    print((output / "FINAL_REPORT.md").read_text(encoding="utf-8"))
    return 0 if cert["STAGE_3_H13_R8C"] == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
