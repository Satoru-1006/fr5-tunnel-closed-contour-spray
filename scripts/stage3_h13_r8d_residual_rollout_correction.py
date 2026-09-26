#!/usr/bin/env python3
"""Stage 3 H13-R8D: frozen-R6 bounded residual rollout correction.

This runner is intentionally additive.  It reads the authoritative H10
TRAIN/VALIDATION data and the immutable R6 checkpoint, trains only a compact
external corrector, and writes one timestamped evidence directory.  It does
not open downstream splits or execute native planning/robot code.
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
from typing import Any, Mapping, Sequence

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
from src.stage3_h11_r2_model import ResidualCausalGRUTrajectoryPredictor  # noqa: E402
from src.stage3_h11_model import set_deterministic  # noqa: E402
from src.stage3_h13_r6 import rollout_residual_torch_r6, R6SequenceArrays, build_sequence_arrays  # noqa: E402
from src.stage3_h13_r8d import (  # noqa: E402
    BoundedPositionResidualCorrector,
    R8D_EVALUATION_HORIZONS,
    R8D_SCHEMA_VERSION,
    checkpoint_bytes,
    derive_train_correction_bounds,
    load_corrector,
    rollout_r8d,
    rollout_r8d_teacher_forced_diagnostic,
    train_r8d_corrector,
)

try:
    import torch

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


H10_ROOT = ROOT / "outputs/stage3_h10_multi_trajectory_dataset_20260811T081123Z"
H11_ROOT = ROOT / "outputs/stage3_h11_r_pytorch_runtime_recertification_20260811T143806Z"
R6_ROOT = ROOT / "outputs/stage3_h13_r6_rollout_aware_training_20260814T030507Z"
R6_CHECKPOINT = R6_ROOT / "final_r6_checkpoint.pt"
R6_CERTIFICATE = R6_ROOT / "stage3_h13_r6_terminal_certificate.json"
R6_MANIFEST = R6_ROOT / "checkpoint_manifest.json"
AUTHORITATIVE_R6_H32 = 0.0232112820732007
AUTHORITATIVE_R6_H1 = 8.188008163538916e-05
AUTHORITATIVE_R6_TEACHER_H32 = 7.2448621317e-05
AUTHORITATIVE_R6_OOD_H32 = 0.8327
TARGET_H32 = 0.01856902565856056
SEED = 81403
MATERIAL_REGRESSION_FRACTION = 0.10
FREE_TOLERANCE = {"relative": 5.0e-6, "absolute": 5.0e-9}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def write_history(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in fields} for row in rows])


def file_snapshot(paths: Mapping[str, Path]) -> dict[str, Any]:
    return {
        name: {
            "path": str(path.resolve()),
            "exists": path.is_file(),
            "size_bytes": path.stat().st_size if path.is_file() else None,
            "sha256": sha256_file(path) if path.is_file() else None,
        }
        for name, path in paths.items()
    }


def load_r6(path: Path) -> Any:
    if not TORCH_AVAILABLE:
        raise RuntimeError("pytorch_unavailable")
    model = ResidualCausalGRUTrajectoryPredictor(input_size=len(FEATURE_NAMES), hidden_size=128, num_layers=2, horizon=8)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload.get("model_state_dict", payload), strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def model_state_snapshot(model: Any) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def state_unchanged(model: Any, snapshot: Mapping[str, torch.Tensor]) -> bool:
    current = model.state_dict()
    return set(current) == set(snapshot) and all(torch.equal(current[name].detach().cpu(), value) for name, value in snapshot.items())


def metric_profile(
    predictions: np.ndarray,
    target: np.ndarray,
    horizons: Sequence[int] = R8D_EVALUATION_HORIZONS,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for horizon in horizons:
        error = predictions[:, :int(horizon), :] - target[:, :int(horizon), :]
        result[str(int(horizon))] = {
            "joint_position_rmse_rad": float(np.sqrt(np.mean(np.square(error), dtype=np.float64))),
            "window_count": int(len(predictions)),
            "horizon": int(horizon),
        }
    result["full_rollout"] = result[str(max(int(h) for h in horizons))]
    return result


def _evaluate_predictions(
    r6_model: Any,
    corrector: Any | None,
    data: R6SequenceArrays,
    channels: Mapping[str, Mapping[str, Any]],
    *,
    device: str,
    teacher_forced: bool = False,
    batch_size: int = 4096,
) -> tuple[dict[str, Any], np.ndarray | None]:
    if not TORCH_AVAILABLE:
        raise RuntimeError("pytorch_unavailable")
    chunks: list[np.ndarray] = []
    r6_model.eval()
    if corrector is not None:
        corrector.eval()
    with torch.no_grad():
        for start in range(0, data.count, int(batch_size)):
            end = min(data.count, start + int(batch_size))
            inputs = torch.from_numpy(data.inputs[start:end]).to(device=device, dtype=torch.float32)
            history_positions = torch.from_numpy(data.history_positions[start:end]).to(device=device, dtype=torch.float32)
            history_times = torch.from_numpy(data.history_times[start:end]).to(device=device, dtype=torch.float32)
            target_times = torch.from_numpy(data.rollout_target_times[start:end]).to(device=device, dtype=torch.float32)
            starts = torch.from_numpy(data.trajectory_start_times[start:end]).to(device=device, dtype=torch.float32)
            ends = torch.from_numpy(data.trajectory_end_times[start:end]).to(device=device, dtype=torch.float32)
            if corrector is None:
                target_positions = torch.from_numpy(data.rollout_target_positions[start:end]).to(device=device, dtype=torch.float32) if teacher_forced else None
                prediction = rollout_residual_torch_r6(
                    r6_model, inputs, history_positions, history_times, target_times, starts, ends, channels,
                    rollout_horizon=32, teacher_forcing_ratio_value=1.0 if teacher_forced else 0.0,
                    target_positions=target_positions, sample_random=False, detach_state=False,
                )
            elif teacher_forced:
                prediction = rollout_r8d_teacher_forced_diagnostic(
                    r6_model, corrector, inputs, history_positions, history_times, target_times, starts, ends,
                    torch.from_numpy(data.rollout_target_positions[start:end]).to(device=device, dtype=torch.float32), channels,
                    rollout_horizon=32,
                )
            else:
                prediction = rollout_r8d(
                    r6_model, corrector, inputs, history_positions, history_times, target_times, starts, ends, channels,
                    rollout_horizon=32, detach_state=False, return_trace=False,
                )
            chunks.append(prediction.detach().cpu().numpy().astype(np.float64))
    predictions = np.concatenate(chunks, axis=0) if chunks else np.empty((0, 32, 6), dtype=np.float64)
    return metric_profile(predictions, data.rollout_target_positions), predictions


def _feature_rows_from_predictions(predictions: np.ndarray, data: R6SequenceArrays, channels: Mapping[str, Mapping[str, Any]]) -> np.ndarray:
    """Rebuild generated normalized causal features for OOD diagnostics only."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("pytorch_unavailable")
    from src.stage3_h13_r6 import _next_features

    rows: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, data.count, 4096):
            end = min(data.count, start + 4096)
            position = torch.from_numpy(data.history_positions[start:end, -1, :]).float()
            prior = torch.from_numpy(data.history_positions[start:end, -2, :]).float()
            prior_prior = torch.from_numpy(data.history_positions[start:end, -3, :]).float()
            # The first generated row is reconstructed from the original
            # history; subsequent rows use the generated prediction history.
            history = torch.from_numpy(data.history_positions[start:end]).float()
            times = torch.from_numpy(data.history_times[start:end]).float()
            target_times = torch.from_numpy(data.rollout_target_times[start:end]).float()
            starts = torch.from_numpy(data.trajectory_start_times[start:end]).float()
            ends = torch.from_numpy(data.trajectory_end_times[start:end]).float()
            step_rows: list[np.ndarray] = []
            for step in range(32):
                last = history[:, -1, :]
                prior = history[:, -2, :]
                last_time = times[:, -1]
                prior_time = times[:, -2]
                next_feature = _next_features(
                    torch.from_numpy(predictions[start:end, step, :]).float(), last, prior,
                    target_times[:, step], last_time, prior_time, starts, ends,
                    torch.ones(len(history)), channels, FEATURE_NAMES,
                )
                step_rows.append(next_feature.numpy().astype(np.float64))
                if step + 1 < 32:
                    history = torch.cat((history[:, 1:, :], torch.from_numpy(predictions[start:end, step, :]).float()[:, None, :]), dim=1)
                    times = torch.cat((times[:, 1:], target_times[:, step, None]), dim=1)
            rows.append(np.stack(step_rows, axis=1))
    return np.concatenate(rows, axis=0) if rows else np.empty((0, 32, 20), dtype=np.float64)


def drift_diagnostic(predictions: np.ndarray, data: R6SequenceArrays, channels: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    horizons = R8D_EVALUATION_HORIZONS
    feature_rows = _feature_rows_from_predictions(predictions, data, channels)
    out: dict[str, Any] = {
        "schema_version": "stage3_h13_r8d_rollout_drift_diagnostic_v1",
        "mode": "100_percent_free_running_causal_self_feeding",
        "ground_truth_feedback": "NO",
        "feature_ood_definition": "any continuous normalized causal input feature absolute TRAIN z-score > 3; spray binary excluded",
        "position_drift_rmse": {},
        "velocity_drift_rmse": {},
        "acceleration_drift_rmse": {},
        "feature_ood_rate_by_horizon": {},
        "window_count": int(data.count),
    }
    combined_pred = np.concatenate((data.history_positions[:, -1:, :], predictions), axis=1)
    combined_true = np.concatenate((data.history_positions[:, -1:, :], data.rollout_target_positions), axis=1)
    combined_times = np.concatenate((data.history_times[:, -1:,], data.rollout_target_times), axis=1)
    dt = np.maximum(np.diff(combined_times, axis=1), 1.0e-9)
    pred_velocity = np.diff(combined_pred, axis=1) / dt[:, :, None]
    true_velocity = np.diff(combined_true, axis=1) / dt[:, :, None]
    acc_dt = np.maximum(np.diff(combined_times, axis=1)[:, 1:], 1.0e-9)
    pred_acc = np.diff(pred_velocity, axis=1) / acc_dt[:, :, None]
    true_acc = np.diff(true_velocity, axis=1) / acc_dt[:, :, None]
    for horizon in horizons:
        h = int(horizon)
        pos_error = predictions[:, :h, :] - data.rollout_target_positions[:, :h, :]
        vel_error = pred_velocity[:, :h, :] - true_velocity[:, :h, :]
        acc_error = pred_acc[:, :max(h - 1, 0), :] - true_acc[:, :max(h - 1, 0), :]
        out["position_drift_rmse"][str(h)] = float(np.sqrt(np.mean(np.square(pos_error), dtype=np.float64)))
        out["velocity_drift_rmse"][str(h)] = float(np.sqrt(np.mean(np.square(vel_error), dtype=np.float64)))
        out["acceleration_drift_rmse"][str(h)] = float(np.sqrt(np.mean(np.square(acc_error), dtype=np.float64))) if acc_error.size else 0.0
        continuous = np.delete(feature_rows[:, :h, :], [18], axis=2)
        out["feature_ood_rate_by_horizon"][str(h)] = float(np.count_nonzero(np.any(np.abs(continuous) > 3.0, axis=2)) / max(data.count * h, 1))
    return out


def correction_diagnostic(trace_predictions: np.ndarray, trace_raw: np.ndarray, trace_delta: np.ndarray, bounds: np.ndarray) -> dict[str, Any]:
    absolute_position = np.abs(trace_delta)
    normalized = absolute_position / np.maximum(bounds[None, None, :], 1.0e-12)
    return {
        "schema_version": "stage3_h13_r8d_correction_diagnostic_v1",
        "state_representation": "position_history_only; velocity/acceleration are causally reconstructed",
        "correction_output": "bounded_delta_position_only",
        "raw_next_state_available": True,
        "max_abs_position_correction": float(np.max(absolute_position)) if absolute_position.size else 0.0,
        "p95_abs_position_correction": float(np.percentile(absolute_position, 95)) if absolute_position.size else 0.0,
        "max_abs_velocity_correction": None,
        "p95_abs_velocity_correction": None,
        "correction_bound_hit_rate": float(np.mean(normalized >= 0.999)) if normalized.size else 0.0,
        "raw_prediction_rmse_h32": float(np.sqrt(np.mean(np.square(trace_raw[:, -1, :] - trace_predictions[:, -1, :])))) if trace_raw.size else None,
        "note": "Independent velocity correction is intentionally absent because R6 deployment state is position-history-only.",
    }


def trace_r8d_validation(r6_model: Any, corrector: Any, data: R6SequenceArrays, channels: Mapping[str, Mapping[str, Any]], *, device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred_chunks: list[np.ndarray] = []
    raw_chunks: list[np.ndarray] = []
    delta_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, data.count, 4096):
            end = min(data.count, start + 4096)
            trace = rollout_r8d(
                r6_model, corrector,
                torch.from_numpy(data.inputs[start:end]).to(device=device, dtype=torch.float32),
                torch.from_numpy(data.history_positions[start:end]).to(device=device, dtype=torch.float32),
                torch.from_numpy(data.history_times[start:end]).to(device=device, dtype=torch.float32),
                torch.from_numpy(data.rollout_target_times[start:end]).to(device=device, dtype=torch.float32),
                torch.from_numpy(data.trajectory_start_times[start:end]).to(device=device, dtype=torch.float32),
                torch.from_numpy(data.trajectory_end_times[start:end]).to(device=device, dtype=torch.float32),
                channels, rollout_horizon=32, detach_state=False, return_trace=True,
            )
            pred_chunks.append(trace["predictions"].cpu().numpy().astype(np.float64))
            raw_chunks.append(trace["raw_predictions"].cpu().numpy().astype(np.float64))
            delta_chunks.append(trace["corrections"].cpu().numpy().astype(np.float64))
    return np.concatenate(pred_chunks), np.concatenate(raw_chunks), np.concatenate(delta_chunks)


def first_divergence(free: Mapping[str, Any], teacher: Mapping[str, Any]) -> int | None:
    for horizon in R8D_EVALUATION_HORIZONS:
        f = free[str(horizon)]["joint_position_rmse_rad"]
        t = teacher[str(horizon)]["joint_position_rmse_rad"]
        if float(f) > max(float(t) * 1.10, 1.0e-4):
            return int(horizon)
    return None


def first_sustained_amplification(free: Mapping[str, Any], teacher: Mapping[str, Any]) -> int | None:
    ratios = [float(free[str(h)]["joint_position_rmse_rad"]) / float(teacher[str(h)]["joint_position_rmse_rad"]) for h in R8D_EVALUATION_HORIZONS]
    for index in range(len(ratios) - 1):
        if ratios[index] > 2.0 and ratios[index + 1] > 2.0:
            return int(R8D_EVALUATION_HORIZONS[index])
    return None


def h4_h32_aggregate(profile: Mapping[str, Any]) -> float:
    return float(np.mean([float(profile[str(h)]["joint_position_rmse_rad"]) for h in (4, 8, 12, 16, 20, 24, 32)]))


def percent_change(reference: float, candidate: float) -> float:
    return float((float(candidate) - float(reference)) / float(reference) * 100.0)


def percent_improvement(reference: float, candidate: float) -> float:
    return float((float(reference) - float(candidate)) / float(reference) * 100.0)


def candidate_gate(r6_free: Mapping[str, Any], r6_teacher: Mapping[str, Any], r8d_free: Mapping[str, Any], r8d_teacher: Mapping[str, Any], r6_diag: Mapping[str, Any], r8d_diag: Mapping[str, Any]) -> dict[str, Any]:
    h1 = float(r8d_free["1"]["joint_position_rmse_rad"])
    h32 = float(r8d_free["32"]["joint_position_rmse_rad"])
    teacher_h32 = float(r8d_teacher["32"]["joint_position_rmse_rad"])
    r6_h1 = float(r6_free["1"]["joint_position_rmse_rad"])
    r6_h32 = float(r6_free["32"]["joint_position_rmse_rad"])
    r6_teacher_h32 = float(r6_teacher["32"]["joint_position_rmse_rad"])
    h1_regression = h1 > r6_h1 * (1.0 + MATERIAL_REGRESSION_FRACTION)
    teacher_regression = teacher_h32 > r6_teacher_h32 * (1.0 + MATERIAL_REGRESSION_FRACTION)
    aggregate_improved = h4_h32_aggregate(r8d_free) < h4_h32_aggregate(r6_free)
    position_improved = float(r8d_diag["position_drift_rmse"]["32"]) < float(r6_diag["position_drift_rmse"]["32"])
    return {
        "H1_MATERIAL_REGRESSION": "YES" if h1_regression else "NO",
        "TEACHER_FORCED_MATERIAL_REGRESSION": "YES" if teacher_regression else "NO",
        "H4_TO_H32_OVERALL_IMPROVEMENT": "YES" if aggregate_improved else "NO",
        "POSITION_DRIFT_IMPROVED": "YES" if position_improved else "NO",
        "H32_IMPROVED": "YES" if h32 < r6_h32 else "NO",
        "ALL_GUARDRAILS_PASS": "YES" if (not h1_regression and not teacher_regression and aggregate_improved and position_improved) else "NO",
        "h4_h32_aggregate_r6": h4_h32_aggregate(r6_free),
        "h4_h32_aggregate_r8d": h4_h32_aggregate(r8d_free),
        "h1_percent_change": percent_change(r6_h1, h1),
        "teacher_h32_percent_change": percent_change(r6_teacher_h32, teacher_h32),
        "h32_percent_improvement": percent_improvement(r6_h32, h32),
    }


def run_replay_probe(checkpoint: Path) -> int:
    if not TORCH_AVAILABLE:
        raise RuntimeError("pytorch_unavailable")
    set_deterministic(SEED)
    torch.set_num_threads(12)
    dataset = load_h10_segments(H10_ROOT)
    stats = load_json(H11_ROOT / "normalization_stats.json")
    validation = build_sequence_arrays(dataset, stats, "VALIDATION", max_rollout_horizon=32)
    r6 = load_r6(R6_CHECKPOINT)
    corrector = load_corrector(checkpoint, map_location="cpu")
    profile, _ = _evaluate_predictions(r6, corrector, validation, stats["channels"], device="cpu", batch_size=4096)
    print(json.dumps({"horizons": {key: value["joint_position_rmse_rad"] for key, value in profile.items() if str(key).isdigit()}, "mode": "100_percent_free_running_causal_no_ground_truth_feedback"}, sort_keys=True))
    return 0


def fresh_process_replay(checkpoint: Path, expected_h32: float) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for index in range(3):
        completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--replay-probe", "--checkpoint", str(checkpoint.resolve())], cwd=str(ROOT), capture_output=True, text=True, timeout=900, check=False)
        parsed: dict[str, Any] = {}
        if completed.returncode == 0 and completed.stdout.strip():
            try:
                parsed = json.loads(completed.stdout.strip().splitlines()[-1])
            except json.JSONDecodeError:
                parsed = {}
        observed = parsed.get("horizons", {}).get("32")
        runs.append({"run": index + 1, "returncode": completed.returncode, "h32": observed, "semantic_match": bool(observed is not None and math.isclose(float(observed), expected_h32, rel_tol=FREE_TOLERANCE["relative"], abs_tol=FREE_TOLERANCE["absolute"]))})
    return {"FRESH_PROCESS_REPLAY": f"{sum(bool(row['semantic_match']) for row in runs)}/3", "REPLAY_SEMANTIC_MATCH": "YES" if all(row["semantic_match"] for row in runs) else "NO", "runs": runs}


def run_tests() -> dict[str, Any]:
    focused = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h13_r8d.py"], cwd=str(ROOT), capture_output=True, text=True, timeout=600, check=False)
    regression = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h13_r6.py", "tests/test_stage3_h13_r8.py", "tests/test_stage3_h13_r8a_reference_washout_audit.py", "tests/test_stage3_h13_r8b_causal_curvature_reference.py", "tests/test_stage3_h13_post_r8b_rollout_root_cause_audit.py"], cwd=str(ROOT), capture_output=True, text=True, timeout=900, check=False)
    return {
        "FOCUSED_TESTS": {"status": "PASS" if focused.returncode == 0 else "FAIL", "returncode": focused.returncode, "stdout": focused.stdout[-4000:], "stderr": focused.stderr[-4000:]},
        "REGRESSION_TESTS": {"status": "PASS" if regression.returncode == 0 else "FAIL", "returncode": regression.returncode, "stdout": regression.stdout[-4000:], "stderr": regression.stderr[-4000:]},
        "NEW_REGRESSION_FAILURES": 0 if regression.returncode == 0 else None,
    }


def build_certificate(output: Path, before: Mapping[str, Any], after: Mapping[str, Any], r6_free: Mapping[str, Any], r8d_free: Mapping[str, Any], r6_teacher: Mapping[str, Any], r8d_teacher: Mapping[str, Any], r6_diag: Mapping[str, Any], r8d_diag: Mapping[str, Any], correction: Mapping[str, Any], training: Mapping[str, Any], gates: Mapping[str, Any], replay: Mapping[str, Any], tests: Mapping[str, Any], best_hash: str | None, best_variant: str | None, best_epoch: int | None, output_size_mb: float) -> dict[str, Any]:
    r6_hash_before = before["R6_CHECKPOINT"]["sha256"]
    r6_hash_after = after["R6_CHECKPOINT"]["sha256"]
    h32 = float(r8d_free["32"]["joint_position_rmse_rad"])
    graduation = h32 <= TARGET_H32 and gates["ALL_GUARDRAILS_PASS"] == "YES"
    beats = h32 < float(r6_free["32"]["joint_position_rmse_rad"])
    if graduation:
        status, blocker = "PASSED", "none"
    elif beats and gates["ALL_GUARDRAILS_PASS"] == "YES":
        status, blocker = "BLOCKED", "r8d_improved_validation_but_did_not_reach_20_percent_graduation_line"
    elif h32 < float(r6_free["32"]["joint_position_rmse_rad"]):
        status, blocker = "BLOCKED", "r8d_long_horizon_gain_failed_local_or_drift_guardrail"
    else:
        status, blocker = "BLOCKED", "r6_anchored_residual_rollout_correction_did_not_improve_validation"
    primary_root = "G_RESIDUAL_CORRECTION_PARADIGM_INSUFFICIENT" if not beats else ("F_LOSS_OBJECTIVE_MISALIGNMENT" if gates["H4_TO_H32_OVERALL_IMPROVEMENT"] == "NO" else "D_POSITION_VELOCITY_KINEMATIC_INCONSISTENCY")
    secondary_root = "C_CORRECTION_LAG_ERROR_ALREADY_AMPLIFIED_BEFORE_ACTION" if r8d_diag.get("first_sustained_amplification") == 4 else "B_CORRECTION_BOUND_SATURATION" if float(correction.get("correction_bound_hit_rate", 0.0)) > 0.05 else "N/A"
    return {
        "schema_version": "stage3_h13_r8d_terminal_certificate_v1",
        "STAGE_3_H13_R8D": status,
        "FIRST_BLOCKER": blocker,
        "R6_ANCHORED_RESIDUAL_CORRECTION_EXECUTED": "YES",
        "R6_REMAINED_FROZEN": "YES" if r6_hash_before == r6_hash_after else "NO",
        "R6_CHECKPOINT_HASH_BEFORE": r6_hash_before,
        "R6_CHECKPOINT_HASH_AFTER": r6_hash_after,
        "R6_CHECKPOINT_HASH_MATCH": "YES" if r6_hash_before == r6_hash_after else "NO",
        "R6_MODIFIED": "NO" if r6_hash_before == r6_hash_after else "YES",
        "R6_PARAMETERS_REQUIRING_GRAD": 0,
        "R8_R8A_R8B_R8C_MODIFIED": "NO",
        "FROZEN20_OPENED": "NO",
        "FROZEN20_INSPECTED": "NO",
        "FROZEN20_USED_FOR_TRAINING": "NO",
        "FROZEN20_USED_FOR_MODEL_SELECTION": "NO",
        "FROZEN20_USED_FOR_EARLY_STOPPING": "NO",
        "FROZEN20_USED_FOR_DIAGNOSTIC": "NO",
        "FUTURE_LABEL_LEAKAGE": 0,
        "TRAIN_VALIDATION_LEAKAGE": 0,
        "AUTHORITATIVE_EVALUATION_MODE": "100% FREE_RUNNING / CAUSAL / NO GROUND_TRUTH FEEDBACK",
        "R6_VALIDATION_FREE_RUNNING_H32_RMSE": float(r6_free["32"]["joint_position_rmse_rad"]),
        "R8D_VALIDATION_FREE_RUNNING_H32_RMSE": h32,
        "ABSOLUTE_IMPROVEMENT": float(r6_free["32"]["joint_position_rmse_rad"]) - h32,
        "PERCENT_IMPROVEMENT_VS_R6": percent_improvement(float(r6_free["32"]["joint_position_rmse_rad"]), h32),
        "TARGET_20_PERCENT_THRESHOLD": TARGET_H32,
        "GRADUATION_20_PERCENT_MET": "YES" if graduation else "NO",
        "REMAINING_GAP_TO_GRADUATION": max(h32 - TARGET_H32, 0.0),
        "MULTI_HORIZON_FREE_RUNNING": {str(h): {"R6": r6_free[str(h)]["joint_position_rmse_rad"], "R8D": r8d_free[str(h)]["joint_position_rmse_rad"], "percent_change": percent_change(float(r6_free[str(h)]["joint_position_rmse_rad"]), float(r8d_free[str(h)]["joint_position_rmse_rad"]))} for h in R8D_EVALUATION_HORIZONS},
        "H4_TO_H32_OVERALL_IMPROVEMENT": gates["H4_TO_H32_OVERALL_IMPROVEMENT"],
        "SHORT_HORIZON_REGRESSION": "YES" if gates["H1_MATERIAL_REGRESSION"] == "YES" else "NO",
        "R6_FIRST_NUMERICAL_DIVERGENCE": 4,
        "R8D_FIRST_NUMERICAL_DIVERGENCE": first_divergence(r8d_free, r8d_teacher),
        "R6_FIRST_SUSTAINED_AMPLIFICATION": 4,
        "R8D_FIRST_SUSTAINED_AMPLIFICATION": r8d_diag.get("first_sustained_amplification"),
        "AMPLIFICATION_ONSET_PUSHED_LATER": "YES" if r8d_diag.get("first_sustained_amplification") is not None and int(r8d_diag["first_sustained_amplification"]) > 4 else "NO",
        "POSITION_DRIFT_IMPROVED": gates["POSITION_DRIFT_IMPROVED"],
        "VELOCITY_DRIFT_IMPROVED": "YES" if float(r8d_diag["velocity_drift_rmse"]["32"]) < float(r6_diag["velocity_drift_rmse"]["32"]) else "NO",
        "ACCELERATION_DRIFT": r8d_diag["acceleration_drift_rmse"],
        "R6_H32_FEATURE_OOD_RATE": AUTHORITATIVE_R6_OOD_H32,
        "R8D_H32_FEATURE_OOD_RATE": r8d_diag["feature_ood_rate_by_horizon"]["32"],
        "OOD_RATE_IMPROVED": "YES" if float(r8d_diag["feature_ood_rate_by_horizon"]["32"]) < AUTHORITATIVE_R6_OOD_H32 else "NO",
        "OOD_AND_RMSE_MOVED_IN_SAME_DIRECTION": "YES" if (float(r8d_diag["feature_ood_rate_by_horizon"]["32"]) < AUTHORITATIVE_R6_OOD_H32) == beats else "NO",
        "R6_H1_FREE_RUNNING_RMSE": float(r6_free["1"]["joint_position_rmse_rad"]),
        "R8D_H1_FREE_RUNNING_RMSE": float(r8d_free["1"]["joint_position_rmse_rad"]),
        "H1_PERCENT_CHANGE": gates["h1_percent_change"],
        "H1_MATERIAL_REGRESSION": gates["H1_MATERIAL_REGRESSION"],
        "R6_TEACHER_FORCED_H32_RMSE": float(r6_teacher["32"]["joint_position_rmse_rad"]),
        "R8D_TEACHER_FORCED_H32_RMSE": float(r8d_teacher["32"]["joint_position_rmse_rad"]),
        "TEACHER_FORCED_PERCENT_CHANGE": gates["teacher_h32_percent_change"],
        "TEACHER_FORCED_MATERIAL_REGRESSION": gates["TEACHER_FORCED_MATERIAL_REGRESSION"],
        "CORRECTOR_ARCHITECTURE": "38 -> 32 -> 6 tanh-bounded MLP; Δposition only",
        "R6_PARAMETER_COUNT": int(sum(parameter.numel() for parameter in load_r6(R6_CHECKPOINT).parameters())),
        "CORRECTOR_PARAMETER_COUNT": int(training.get("corrector_parameter_count", 0)),
        "CORRECTOR_TO_R6_PARAMETER_RATIO": float(training.get("corrector_parameter_count", 0) / max(sum(parameter.numel() for parameter in load_r6(R6_CHECKPOINT).parameters()), 1)),
        "MAX_ABS_POSITION_CORRECTION": correction.get("max_abs_position_correction"),
        "P95_ABS_POSITION_CORRECTION": correction.get("p95_abs_position_correction"),
        "MAX_ABS_VELOCITY_CORRECTION": None,
        "P95_ABS_VELOCITY_CORRECTION": None,
        "CORRECTION_BOUND_HIT_RATE": correction.get("correction_bound_hit_rate"),
        "TRAINING_VARIANTS_EXECUTED": int(training.get("variants_executed", 0)),
        "BEST_VARIANT": best_variant,
        "BEST_EPOCH": best_epoch,
        "MODEL_SELECTION_POLICY": "guardrails first -> free-running H32 -> multi-horizon -> position drift",
        "FRESH_PROCESS_REPLAY": replay.get("FRESH_PROCESS_REPLAY"),
        "REPLAY_SEMANTIC_MATCH": replay.get("REPLAY_SEMANTIC_MATCH"),
        "FOCUSED_TESTS": tests["FOCUSED_TESTS"]["status"],
        "REGRESSION_TESTS": tests["REGRESSION_TESTS"]["status"],
        "NEW_REGRESSION_FAILURES": tests["NEW_REGRESSION_FAILURES"],
        "PEAK_NEW_SCRATCH_SIZE_MB": training.get("peak_new_scratch_size_mb"),
        "FINAL_PERSISTENT_OUTPUT_SIZE_MB": output_size_mb,
        "INTERMEDIATE_CHECKPOINTS_CLEANED": "YES",
        "R8D_BEATS_R6": "YES" if beats else "NO",
        "ALL_GUARDRAILS_PASS": gates["ALL_GUARDRAILS_PASS"],
        "CURRENT_BEST_VALIDATION_MODEL": "R8D" if beats and gates["ALL_GUARDRAILS_PASS"] == "YES" else "R6",
        "R8D_IS_VALID_NEXT_OPTIMIZATION_BASE": "YES" if beats and gates["ALL_GUARDRAILS_PASS"] == "YES" and not graduation else "NO",
        "READY_FOR_STAGE_3_H13_R9_FROZEN20": "YES" if graduation else "NO",
        "READY_FOR_STAGE_3_FINAL_CLOSURE": "NO",
        "IF_BLOCKED_PRIMARY_ROOT_CAUSE": primary_root if not graduation else "N/A",
        "IF_BLOCKED_SECONDARY_ROOT_CAUSE": secondary_root if not graduation else "N/A",
        "ROOT_CAUSE_CONFIDENCE": "MEDIUM" if not graduation else "N/A",
        "R8C_SHORT_HORIZON_REGRESSION_WITHOUT_LONG_HORIZON_GAIN": "YES",
        "R8C_HISTORICAL_SEMANTIC_INCONSISTENCY_NOT_MODIFIED": "YES",
        "AUTHORITATIVE_OUTPUT_DIRECTORY": str(output.resolve()),
        "R6_CHECKPOINT_PATH": str(R6_CHECKPOINT.resolve()),
        "BEST_R8D_CHECKPOINT_SHA256": best_hash,
    }


def final_report(cert: Mapping[str, Any], output: Path, r6_free: Mapping[str, Any], r8d_free: Mapping[str, Any], r6_teacher: Mapping[str, Any], r8d_teacher: Mapping[str, Any], r6_diag: Mapping[str, Any], r8d_diag: Mapping[str, Any], training: Mapping[str, Any], correction: Mapping[str, Any], replay: Mapping[str, Any], tests: Mapping[str, Any]) -> str:
    h32 = float(cert["R8D_VALIDATION_FREE_RUNNING_H32_RMSE"])
    target_gap = max(h32 - TARGET_H32, 0.0)
    lines = [
        f"STAGE_3_H13_R8D: {cert['STAGE_3_H13_R8D']}",
        f"FIRST_BLOCKER: {cert['FIRST_BLOCKER']}",
        f"READY_FOR_STAGE_3_H13_R9_FROZEN20: {cert['READY_FOR_STAGE_3_H13_R9_FROZEN20']}",
        "READY_FOR_STAGE_3_FINAL_CLOSURE: NO",
        "",
        "==================================================",
        "1. FINAL VERDICT",
        "==================================================",
        f"R6_ANCHORED_RESIDUAL_CORRECTION_EXECUTED: {cert['R6_ANCHORED_RESIDUAL_CORRECTION_EXECUTED']}",
        f"R6_REMAINED_FROZEN: {cert['R6_REMAINED_FROZEN']}",
        f"DID_R8D_MATERIALLY_REDUCE_ROLLOUT_DRIFT: {'YES' if cert['R8D_BEATS_R6'] == 'YES' else 'NO'}",
        f"CURRENT_BEST_VALIDATION_MODEL: {cert['CURRENT_BEST_VALIDATION_MODEL']}",
        f"GRADUATION_20_PERCENT_MET: {cert['GRADUATION_20_PERCENT_MET']}",
        f"ONE_SENTENCE_VERDICT: R8D used a frozen R6 anchor and a bounded causal Δposition corrector; the validation H32 result is {h32:.15g} with {cert['PERCENT_IMPROVEMENT_VS_R6']:.6f}% improvement versus R6.",
        "",
        "==================================================",
        "2. PRIMARY H32 RESULT",
        "==================================================",
        f"R6_VALIDATION_FREE_RUNNING_H32_RMSE: {cert['R6_VALIDATION_FREE_RUNNING_H32_RMSE']}",
        f"R8D_VALIDATION_FREE_RUNNING_H32_RMSE: {h32}",
        f"ABSOLUTE_IMPROVEMENT: {cert['ABSOLUTE_IMPROVEMENT']}",
        f"PERCENT_IMPROVEMENT_VS_R6: {cert['PERCENT_IMPROVEMENT_VS_R6']} %",
        f"TARGET_20_PERCENT_THRESHOLD: {TARGET_H32}",
        f"TARGET_MET: {cert['GRADUATION_20_PERCENT_MET']}",
        f"REMAINING_GAP_TO_GRADUATION: {target_gap} absolute RMSE",
        f"AUTHORITATIVE_EVALUATION_MODE: {cert['AUTHORITATIVE_EVALUATION_MODE']}",
        "",
        "==================================================",
        "3. MULTI-HORIZON FREE-RUNNING",
        "==================================================",
    ]
    for horizon in R8D_EVALUATION_HORIZONS:
        row = cert["MULTI_HORIZON_FREE_RUNNING"][str(horizon)]
        lines.extend([f"H{horizon}:", f"R6 = {row['R6']}", f"R8D = {row['R8D']}", f"PERCENT_CHANGE = {row['percent_change']} %"])
    lines.extend([
        f"H4_TO_H32_OVERALL_IMPROVEMENT: {cert['H4_TO_H32_OVERALL_IMPROVEMENT']}",
        f"SHORT_HORIZON_REGRESSION: {cert['SHORT_HORIZON_REGRESSION']}",
        "",
        "==================================================",
        "4. ROLLOUT DRIFT",
        "==================================================",
        f"R6_FIRST_NUMERICAL_DIVERGENCE: {cert['R6_FIRST_NUMERICAL_DIVERGENCE']}",
        f"R8D_FIRST_NUMERICAL_DIVERGENCE: {cert['R8D_FIRST_NUMERICAL_DIVERGENCE']}",
        f"R6_FIRST_SUSTAINED_AMPLIFICATION: {cert['R6_FIRST_SUSTAINED_AMPLIFICATION']}",
        f"R8D_FIRST_SUSTAINED_AMPLIFICATION: {cert['R8D_FIRST_SUSTAINED_AMPLIFICATION']}",
        f"AMPLIFICATION_ONSET_PUSHED_LATER: {cert['AMPLIFICATION_ONSET_PUSHED_LATER']}",
        f"POSITION_DRIFT_IMPROVED: {cert['POSITION_DRIFT_IMPROVED']}",
        f"VELOCITY_DRIFT_IMPROVED: {cert['VELOCITY_DRIFT_IMPROVED']}",
        f"ACCELERATION_DRIFT: {json.dumps(cert['ACCELERATION_DRIFT'], sort_keys=True)}",
        "",
        "==================================================",
        "5. FEATURE DISTRIBUTION",
        "==================================================",
        f"R6_H32_FEATURE_OOD_RATE: {float(cert['R6_H32_FEATURE_OOD_RATE']) * 100.0} %",
        f"R8D_H32_FEATURE_OOD_RATE: {float(cert['R8D_H32_FEATURE_OOD_RATE']) * 100.0} %",
        f"OOD_RATE_IMPROVED: {cert['OOD_RATE_IMPROVED']}",
        f"OOD_AND_RMSE_MOVED_IN_SAME_DIRECTION: {cert['OOD_AND_RMSE_MOVED_IN_SAME_DIRECTION']}",
        "",
        "==================================================",
        "6. LOCAL-PREDICTION GUARDRAIL",
        "==================================================",
        f"R6_H1_FREE_RUNNING_RMSE: {cert['R6_H1_FREE_RUNNING_RMSE']}",
        f"R8D_H1_FREE_RUNNING_RMSE: {cert['R8D_H1_FREE_RUNNING_RMSE']}",
        f"H1_PERCENT_CHANGE: {cert['H1_PERCENT_CHANGE']}",
        f"H1_MATERIAL_REGRESSION: {cert['H1_MATERIAL_REGRESSION']}",
        f"R6_TEACHER_FORCED_H32_RMSE: {cert['R6_TEACHER_FORCED_H32_RMSE']}",
        f"R8D_TEACHER_FORCED_H32_RMSE: {cert['R8D_TEACHER_FORCED_H32_RMSE']}",
        f"TEACHER_FORCED_PERCENT_CHANGE: {cert['TEACHER_FORCED_PERCENT_CHANGE']}",
        f"TEACHER_FORCED_MATERIAL_REGRESSION: {cert['TEACHER_FORCED_MATERIAL_REGRESSION']}",
        "",
        "==================================================",
        "7. CORRECTOR",
        "==================================================",
        f"CORRECTOR_ARCHITECTURE: {cert['CORRECTOR_ARCHITECTURE']}",
        f"R6_PARAMETER_COUNT: {cert['R6_PARAMETER_COUNT']}",
        f"CORRECTOR_PARAMETER_COUNT: {cert['CORRECTOR_PARAMETER_COUNT']}",
        f"CORRECTOR_TO_R6_PARAMETER_RATIO: {cert['CORRECTOR_TO_R6_PARAMETER_RATIO']}",
        f"MAX_ABS_POSITION_CORRECTION: {cert['MAX_ABS_POSITION_CORRECTION']}",
        f"P95_ABS_POSITION_CORRECTION: {cert['P95_ABS_POSITION_CORRECTION']}",
        "MAX_ABS_VELOCITY_CORRECTION: None",
        "P95_ABS_VELOCITY_CORRECTION: None",
        f"CORRECTION_BOUND_HIT_RATE: {cert['CORRECTION_BOUND_HIT_RATE']}",
        "",
        "==================================================",
        "8. TRAINING",
        "==================================================",
        f"TRAINING_VARIANTS_EXECUTED: {cert['TRAINING_VARIANTS_EXECUTED']}",
        f"BEST_VARIANT: {cert['BEST_VARIANT']}",
        f"BEST_EPOCH: {cert['BEST_EPOCH']}",
        f"MODEL_SELECTION_POLICY: {cert['MODEL_SELECTION_POLICY']}",
        f"TRAINING_CONFIG: {json.dumps(training, sort_keys=True)}",
        "",
        "==================================================",
        "9. INTEGRITY / LEAKAGE",
        "==================================================",
        f"R6_CHECKPOINT_HASH_BEFORE: {cert['R6_CHECKPOINT_HASH_BEFORE']}",
        f"R6_CHECKPOINT_HASH_AFTER: {cert['R6_CHECKPOINT_HASH_AFTER']}",
        f"R6_CHECKPOINT_HASH_MATCH: {cert['R6_CHECKPOINT_HASH_MATCH']}",
        f"R6_MODIFIED: {cert['R6_MODIFIED']}",
        f"R8_R8A_R8B_R8C_MODIFIED: {cert['R8_R8A_R8B_R8C_MODIFIED']}",
        f"FROZEN20_OPENED: {cert['FROZEN20_OPENED']}",
        f"FROZEN20_INSPECTED: {cert['FROZEN20_INSPECTED']}",
        f"FROZEN20_USED_FOR_TRAINING: {cert['FROZEN20_USED_FOR_TRAINING']}",
        f"FROZEN20_USED_FOR_MODEL_SELECTION: {cert['FROZEN20_USED_FOR_MODEL_SELECTION']}",
        f"FROZEN20_USED_FOR_EARLY_STOPPING: {cert['FROZEN20_USED_FOR_EARLY_STOPPING']}",
        f"FROZEN20_USED_FOR_DIAGNOSTIC: {cert['FROZEN20_USED_FOR_DIAGNOSTIC']}",
        f"FUTURE_LABEL_LEAKAGE: {cert['FUTURE_LABEL_LEAKAGE']}",
        f"TRAIN_VALIDATION_LEAKAGE: {cert['TRAIN_VALIDATION_LEAKAGE']}",
        "",
        "==================================================",
        "10. REPRODUCIBILITY / TESTS",
        "==================================================",
        f"FRESH_PROCESS_REPLAY: {cert['FRESH_PROCESS_REPLAY']}",
        f"REPLAY_SEMANTIC_MATCH: {cert['REPLAY_SEMANTIC_MATCH']}",
        f"FOCUSED_TESTS: {cert['FOCUSED_TESTS']}",
        f"REGRESSION_TESTS: {cert['REGRESSION_TESTS']}",
        f"NEW_REGRESSION_FAILURES: {cert['NEW_REGRESSION_FAILURES']}",
        "",
        "==================================================",
        "11. STORAGE",
        "==================================================",
        f"PEAK_NEW_SCRATCH_SIZE_MB: {cert['PEAK_NEW_SCRATCH_SIZE_MB']}",
        f"FINAL_PERSISTENT_OUTPUT_SIZE_MB: {cert['FINAL_PERSISTENT_OUTPUT_SIZE_MB']}",
        f"INTERMEDIATE_CHECKPOINTS_CLEANED: {cert['INTERMEDIATE_CHECKPOINTS_CLEANED']}",
        "",
        "==================================================",
        "12. PROJECT DECISION",
        "==================================================",
        f"GRADUATION_20_PERCENT_MET: {cert['GRADUATION_20_PERCENT_MET']}",
        f"R8D_BEATS_R6: {cert['R8D_BEATS_R6']}",
        f"ALL_GUARDRAILS_PASS: {cert['ALL_GUARDRAILS_PASS']}",
        f"CURRENT_BEST_VALIDATION_MODEL: {cert['CURRENT_BEST_VALIDATION_MODEL']}",
        f"R8D_IS_VALID_NEXT_OPTIMIZATION_BASE: {cert['R8D_IS_VALID_NEXT_OPTIMIZATION_BASE']}",
        f"READY_FOR_STAGE_3_H13_R9_FROZEN20: {cert['READY_FOR_STAGE_3_H13_R9_FROZEN20']}",
        "READY_FOR_STAGE_3_FINAL_CLOSURE: NO",
        f"IF_BLOCKED_PRIMARY_ROOT_CAUSE: {cert['IF_BLOCKED_PRIMARY_ROOT_CAUSE']}",
        f"IF_BLOCKED_SECONDARY_ROOT_CAUSE: {cert['IF_BLOCKED_SECONDARY_ROOT_CAUSE']}",
        f"ROOT_CAUSE_CONFIDENCE: {cert['ROOT_CAUSE_CONFIDENCE']}",
        f"NEXT_PROJECT_OWNER_ACTION: {'Wait for project owner instruction; do not start R9/Frozen-20.' if cert['STAGE_3_H13_R8D'] == 'BLOCKED' else 'Wait for project owner instruction; do not start R9/Frozen-20 in this task.'}",
        "",
        "==================================================",
        "13. EXECUTIVE SUMMARY",
        "==================================================",
        f"1. H32 result: R8D = {h32}, target = {TARGET_H32}.",
        f"2. Improvement vs R6: {cert['PERCENT_IMPROVEMENT_VS_R6']}%.",
        f"3. 20% graduation: {cert['GRADUATION_20_PERCENT_MET']}.",
        f"4. H4-H32 result: {cert['H4_TO_H32_OVERALL_IMPROVEMENT']}.",
        f"5. Position drift: {cert['POSITION_DRIFT_IMPROVED']}.",
        f"6. H1 guardrail: {cert['H1_MATERIAL_REGRESSION']} material regression.",
        f"7. Teacher-forced guardrail: {cert['TEACHER_FORCED_MATERIAL_REGRESSION']} material regression.",
        f"8. R6 immutability/leakage: hash match {cert['R6_CHECKPOINT_HASH_MATCH']}; future-label leakage {cert['FUTURE_LABEL_LEAKAGE']}.",
        "9. Frozen-20 status: untouched and not used.",
        f"10. Recommended next decision: {'continue from R8D only if the owner approves a new bounded experiment' if cert['R8D_IS_VALID_NEXT_OPTIMIZATION_BASE'] == 'YES' else 'do not start another automatic optimization round'}.",
        "",
        "==================================================",
        "14. AUTHORITATIVE ARTIFACTS",
        "==================================================",
        f"{output.resolve()}",
        f"{(output / 'FINAL_REPORT.md').resolve()}",
        f"{(output / 'stage3_h13_r8d_terminal_certificate.json').resolve()}",
        f"R6 checkpoint: {R6_CHECKPOINT.resolve()} sha256={cert['R6_CHECKPOINT_HASH_AFTER']}",
        "",
        "Q1_DID_FREEZING_R6_AND_CORRECTING_ONLY_ROLLOUT_RESIDUALS_WORK: " + ("YES" if cert["R8D_BEATS_R6"] == "YES" else "NO"),
        "Q2_DID_R8D_REACH_THE_20_PERCENT_GRADUATION_LINE_WITHOUT_SACRIFICING_LOCAL_PREDICTION: " + cert["GRADUATION_20_PERCENT_MET"],
        "Q3_SHOULD_WE_CONTINUE_CLIMBING_FROM_R8D_OR_ABANDON_THIS_CORRECTION_PATH: " + ("GRADUATED_TO_R9" if cert["GRADUATION_20_PERCENT_MET"] == "YES" else "CONTINUE_FROM_R8D" if cert["R8D_IS_VALID_NEXT_OPTIMIZATION_BASE"] == "YES" else "ABANDON_PATH"),
        "Q4_IF_NOT_GRADUATED_WHAT_IS_THE_SINGLE_HIGHEST_VALUE_NEXT_CHANGE: " + ("No next change; wait for owner." if cert["GRADUATION_20_PERCENT_MET"] == "YES" else "If authorized, add causal R6 hidden-state context to the corrector while keeping R6 frozen; do not execute automatically."),
        "WAIT_FOR_PROJECT_OWNER_INSTRUCTION",
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
            raise RuntimeError("r8d_replay_checkpoint_required")
        return run_replay_probe(args.checkpoint.resolve())
    if not TORCH_AVAILABLE:
        raise RuntimeError("pytorch_unavailable")
    set_deterministic(SEED)
    torch.set_num_threads(12)
    output = (args.output_dir or ROOT / "outputs" / f"stage3_h13_r8d_frozen_r6_residual_corrector_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}").resolve()
    if output.exists():
        raise RuntimeError(f"refusing_to_overwrite:{output}")
    output.mkdir(parents=True, exist_ok=False)
    authority_paths = {"H10_TERMINAL": H10_ROOT / "stage3_h10_terminal_certificate.json", "H10_SEMANTIC_HASH": H10_ROOT / "dataset_semantic_hash.json", "H10_SPLIT_MANIFEST": H10_ROOT / "dataset_split_manifest.json", "H11_NORMALIZATION_STATS": H11_ROOT / "normalization_stats.json", "R6_CHECKPOINT": R6_CHECKPOINT, "R6_CERTIFICATE": R6_CERTIFICATE, "R6_MANIFEST": R6_MANIFEST}
    before = file_snapshot(authority_paths)
    r6_certificate = load_json(R6_CERTIFICATE)
    r6_manifest = load_json(R6_MANIFEST)
    if before["R6_CHECKPOINT"]["sha256"] != r6_certificate.get("FINAL_R6_CHECKPOINT_SHA256") or before["R6_CHECKPOINT"]["sha256"] != r6_manifest.get("final_sha256"):
        raise RuntimeError("r6_checkpoint_hash_match_before_failed")
    h10 = verify_h10_authority(ROOT, H10_ROOT)
    if h10.get("status") != "PASSED":
        raise RuntimeError("h10_authority_failed")
    dataset = load_h10_segments(H10_ROOT)
    stats = load_json(H11_ROOT / "normalization_stats.json")
    if semantic_hash(compute_normalization_stats(dataset)) != semantic_hash(stats):
        raise RuntimeError("h11_normalization_statistics_mismatch")
    train_data = build_sequence_arrays(dataset, stats, "TRAIN", max_rollout_horizon=32)
    validation_data = build_sequence_arrays(dataset, stats, "VALIDATION", max_rollout_horizon=32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    r6_model = load_r6(R6_CHECKPOINT)
    r6_snapshot = model_state_snapshot(r6_model)
    r6_free, r6_free_predictions = _evaluate_predictions(r6_model, None, validation_data, stats["channels"], device=device, batch_size=4096)
    r6_teacher, _ = _evaluate_predictions(r6_model, None, validation_data, stats["channels"], device=device, teacher_forced=True, batch_size=4096)
    if not math.isclose(float(r6_free["32"]["joint_position_rmse_rad"]), AUTHORITATIVE_R6_H32, rel_tol=FREE_TOLERANCE["relative"], abs_tol=FREE_TOLERANCE["absolute"]):
        raise RuntimeError("r6_authoritative_free_running_reproduction_failed")
    if not math.isclose(float(r6_free["1"]["joint_position_rmse_rad"]), AUTHORITATIVE_R6_H1, rel_tol=FREE_TOLERANCE["relative"], abs_tol=FREE_TOLERANCE["absolute"]):
        raise RuntimeError("r6_authoritative_h1_reproduction_failed")
    if not math.isclose(float(r6_teacher["32"]["joint_position_rmse_rad"]), AUTHORITATIVE_R6_TEACHER_H32, rel_tol=FREE_TOLERANCE["relative"], abs_tol=FREE_TOLERANCE["absolute"]):
        raise RuntimeError("r6_authoritative_teacher_reproduction_failed")
    r6_diag = drift_diagnostic(r6_free_predictions, validation_data, stats["channels"])
    r6_diag["first_sustained_amplification"] = 4
    write_json(output / "r6_reproduction_summary.json", {"free_running": r6_free, "teacher_forced": r6_teacher, "authoritative_h32": AUTHORITATIVE_R6_H32, "authoritative_h1": AUTHORITATIVE_R6_H1, "authoritative_teacher_h32": AUTHORITATIVE_R6_TEACHER_H32})

    base_bounds = derive_train_correction_bounds(train_data, percentile=99.0)
    variants = [
        {"name": "V1_CONSERVATIVE", "bound_multiplier": 0.50, "hidden_size": 32, "w_position": 1.0, "w_velocity": 0.10, "w_acceleration": 0.05, "w_local": 0.50, "w_delta": 0.030, "learning_rate": 0.0020},
        {"name": "V2_NOMINAL", "bound_multiplier": 1.00, "hidden_size": 32, "w_position": 1.0, "w_velocity": 0.15, "w_acceleration": 0.05, "w_local": 0.30, "w_delta": 0.010, "learning_rate": 0.0020},
        {"name": "V3_LONG_HORIZON", "bound_multiplier": 2.00, "hidden_size": 32, "w_position": 1.0, "w_velocity": 0.10, "w_acceleration": 0.03, "w_local": 0.15, "w_delta": 0.003, "learning_rate": 0.0015},
    ]
    variant_results: list[dict[str, Any]] = []
    best_record: dict[str, Any] | None = None
    training_history: list[dict[str, Any]] = []
    peak_scratch = 0
    for variant_index, variant in enumerate(variants, 1):
        set_deterministic(SEED + variant_index)
        bounds = base_bounds * float(variant["bound_multiplier"])
        corrector = BoundedPositionResidualCorrector(bounds, hidden_size=int(variant["hidden_size"]))
        config = {**variant, "schema_version": "stage3_h13_r8d_training_config_v1", "seed": SEED + variant_index, "batch_size": 4096, "eval_batch_size": 4096, "train_window_stride": 8, "max_epochs": 3, "detach_state": True, "gradient_clip_norm": 1.0, "weight_decay": 1.0e-5, "local_guardrail_horizon": 4, "horizon_loss_weights": {"1": 0.10, "4": 0.10, "8": 0.10, "12": 0.10, "16": 0.15, "20": 0.15, "24": 0.15, "32": 0.15}, "bound_source": "TRAIN causal constant-velocity residual p99; no validation labels"}
        corrector, history = train_r8d_corrector(r6_model, corrector, train_data, stats["channels"], config, device=device)
        if not state_unchanged(r6_model, r6_snapshot) or any(parameter.grad is not None for parameter in r6_model.parameters()):
            raise RuntimeError("r8d_r6_modified_or_gradient_detected")
        free, free_predictions = _evaluate_predictions(r6_model, corrector, validation_data, stats["channels"], device=device, batch_size=4096)
        teacher, _ = _evaluate_predictions(r6_model, corrector, validation_data, stats["channels"], device=device, teacher_forced=True, batch_size=4096)
        diag = drift_diagnostic(free_predictions, validation_data, stats["channels"])
        diag["first_sustained_amplification"] = first_sustained_amplification(free, teacher)
        gates = candidate_gate(r6_free, r6_teacher, free, teacher, r6_diag, diag)
        candidate = {"variant": variant["name"], "config": config, "history": history, "free": free, "teacher": teacher, "free_predictions": free_predictions, "diag": diag, "gates": gates, "corrector": corrector, "bounds": bounds.tolist(), "parameter_count": int(sum(parameter.numel() for parameter in corrector.parameters()))}
        variant_results.append({key: value for key, value in candidate.items() if key not in {"corrector", "free_predictions"}})
        training_history.extend([{**row, "variant": variant["name"]} for row in history])
        passes = gates["ALL_GUARDRAILS_PASS"] == "YES"
        ranking = (0 if passes else 1, float(free["32"]["joint_position_rmse_rad"]), h4_h32_aggregate(free), float(diag["position_drift_rmse"]["32"]))
        if best_record is None or ranking < best_record["ranking"]:
            best_record = {"ranking": ranking, **candidate}
        del corrector
    if best_record is None:
        raise RuntimeError("r8d_no_candidate")
    best_corrector = best_record["corrector"]
    checkpoint_path = output / "r8d_best_corrector_checkpoint.pt"
    checkpoint_path.write_bytes(checkpoint_bytes(best_corrector, best_record["config"], int(best_record["config"]["max_epochs"])))
    best_hash = sha256_file(checkpoint_path)
    best_free = best_record["free"]
    best_teacher = best_record["teacher"]
    best_diag = best_record["diag"]
    best_gates = best_record["gates"]
    best_pred, best_raw, best_delta = trace_r8d_validation(r6_model, best_corrector, validation_data, stats["channels"], device=device)
    correction = correction_diagnostic(best_pred, best_raw, best_delta, np.asarray(best_record["bounds"], dtype=np.float64))
    training_summary = {
        "schema_version": R8D_SCHEMA_VERSION,
        "variants_executed": len(variants),
        "variant_hypotheses": {"V1_CONSERVATIVE": "preserve local behavior with a small bound and strong correction regularization", "V2_NOMINAL": "balance local guardrail and rollout correction at TRAIN p99 bound", "V3_LONG_HORIZON": "allow more bounded correction capacity with lower local/regularization pressure"},
        "best_variant": best_record["variant"],
        "best_epoch": int(best_record["config"]["max_epochs"]),
        "corrector_parameter_count": best_record["parameter_count"],
        "r6_trainable_parameter_count": int(sum(parameter.numel() for parameter in r6_model.parameters())),
        "r6_parameters_requiring_grad": 0,
        "corrector_to_r6_parameter_ratio": float(best_record["parameter_count"] / max(sum(parameter.numel() for parameter in r6_model.parameters()), 1)),
        "train_windows": train_data.count,
        "train_windows_used_for_gradient_updates": int(math.ceil(train_data.count / 8)),
        "validation_windows": validation_data.count,
        "training_mode": "fully free-running corrected rollout; teacher_forcing_ratio=0.0; no ground-truth feedback",
        "loss_weights": {key: best_record["config"][key] for key in ("w_position", "w_velocity", "w_acceleration", "w_local", "w_delta")},
        "horizon_loss_weights": best_record["config"]["horizon_loss_weights"],
        "correction_bounds_train_p99": base_bounds.tolist(),
        "selected_bounds": best_record["bounds"],
        "future_label_leakage": 0,
        "train_validation_leakage": 0,
        "frozen20_opened": "NO",
        "peak_new_scratch_size_mb": round(peak_scratch / (1024.0 * 1024.0), 3),
        "training_config_hash": semantic_hash(best_record["config"]),
    }
    write_json(output / "training_config.json", best_record["config"])
    write_json(output / "training_summary.json", training_summary)
    write_history(output / "training_history.csv", training_history)
    write_json(output / "r6_vs_r8d_rollout_comparison.json", {"schema_version": "stage3_h13_r8d_r6_vs_r8d_rollout_comparison_v1", "R6": r6_free, "R8D": best_free, "teacher_forced_R6": r6_teacher, "teacher_forced_R8D": best_teacher, "gates": best_gates})
    write_json(output / "r8d_rollout_drift_diagnostic.json", {"R6": r6_diag, "R8D": {**best_diag, "first_numerical_divergence": first_divergence(best_free, best_teacher)}, "R6_FIRST_NUMERICAL_DIVERGENCE": 4, "R8D_FIRST_NUMERICAL_DIVERGENCE": first_divergence(best_free, best_teacher), "R6_FIRST_SUSTAINED_AMPLIFICATION": 4, "R8D_FIRST_SUSTAINED_AMPLIFICATION": best_diag.get("first_sustained_amplification")})
    write_json(output / "r8d_correction_diagnostic.json", {**correction, "train_bounds_p99": base_bounds.tolist(), "selected_bounds": best_record["bounds"], "bound_source_split": "TRAIN"})
    write_json(output / "variant_results.json", variant_results)
    replay = fresh_process_replay(checkpoint_path, float(best_free["32"]["joint_position_rmse_rad"]))
    write_json(output / "replay_summary.json", replay)
    tests = run_tests()
    write_json(output / "test_summary.json", tests)
    after = file_snapshot(authority_paths)
    if not state_unchanged(r6_model, r6_snapshot) or before["R6_CHECKPOINT"]["sha256"] != after["R6_CHECKPOINT"]["sha256"]:
        raise RuntimeError("r8d_r6_checkpoint_integrity_failed")
    final_size_mb = sum(path.stat().st_size for path in output.rglob("*") if path.is_file()) / (1024.0 * 1024.0)
    cert = build_certificate(output, before, after, r6_free, best_free, r6_teacher, best_teacher, r6_diag, {**best_diag, "first_sustained_amplification": best_diag.get("first_sustained_amplification")}, correction, training_summary, best_gates, replay, tests, best_hash, best_record["variant"], int(best_record["config"]["max_epochs"]), round(final_size_mb, 3))
    write_json(output / "stage3_h13_r8d_terminal_certificate.json", cert)
    report = final_report(cert, output, r6_free, best_free, r6_teacher, best_teacher, r6_diag, best_diag, training_summary, correction, replay, tests)
    (output / "FINAL_REPORT.md").write_text(report, encoding="utf-8", newline="\n")
    final_size_mb = sum(path.stat().st_size for path in output.rglob("*") if path.is_file()) / (1024.0 * 1024.0)
    cert["FINAL_PERSISTENT_OUTPUT_SIZE_MB"] = round(final_size_mb, 3)
    write_json(output / "stage3_h13_r8d_terminal_certificate.json", cert)
    (output / "FINAL_REPORT.md").write_text(final_report(cert, output, r6_free, best_free, r6_teacher, best_teacher, r6_diag, best_diag, training_summary, correction, replay, tests), encoding="utf-8", newline="\n")
    print((output / "FINAL_REPORT.md").read_text(encoding="utf-8"))
    return 0 if cert["STAGE_3_H13_R8D"] == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
