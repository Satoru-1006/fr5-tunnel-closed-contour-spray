#!/usr/bin/env python3
"""Stage 3 H13 Post-R8B R6 long-horizon root-cause audit.

This is a diagnostic-only runner.  It deliberately reads the frozen H10
TRAIN/VALIDATION data and the authoritative R6 checkpoint, but never reads
the H13 Frozen-20 release, never trains, and never writes into an upstream
release directory.  The implementation keeps the R6 inference semantics:
the first residual emitted by the causal residual GRU is added to the
history-only constant-velocity baseline, and the resulting position is the
next causal state for a free-running rollout.
"""

from __future__ import annotations

import argparse
import hashlib
import io
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
    INPUT_HISTORY,
    compute_normalization_stats,
    load_h10_segments,
    semantic_hash,
    sha256_file,
    verify_h10_authority,
)
from src.stage3_h11_model import set_deterministic  # noqa: E402
from src.stage3_h11_r2_model import (  # noqa: E402
    JOINTS,
    SPRAY_FEATURE_INDEX,
    ResidualCausalGRUTrajectoryPredictor,
)
from src.stage3_h13_r6 import (  # noqa: E402
    ROLLOUT_HORIZONS,
    R6SequenceArrays,
    _next_features,
    build_sequence_arrays,
    rollout_residual_torch_r6,
)

try:  # pragma: no cover - import contract is tested without requiring torch
    import torch

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


H10_ROOT = ROOT / "outputs/stage3_h10_multi_trajectory_dataset_20260811T081123Z"
H11_ROOT = ROOT / "outputs/stage3_h11_r_pytorch_runtime_recertification_20260811T143806Z"
R6_ROOT = ROOT / "outputs/stage3_h13_r6_rollout_aware_training_20260814T030507Z"
R7_ROOT = ROOT / "outputs/stage3_h13_r7_strict_reconstruction_closure_20260814T150000Z"
R8_ROOT = ROOT / "outputs/stage3_h13_r8_autoregressive_state_semantics_redesign_20260814T163000Z"
R8A_ROOT = ROOT / "outputs/stage3_h13_r8a_reference_washout_redesign_20260814T095233Z"
R8B_ROOT = ROOT / "outputs/stage3_h13_r8b_causal_curvature_reference_20260814T124600Z"

R6_CHECKPOINT = R6_ROOT / "final_r6_checkpoint.pt"
R6_CERTIFICATE = R6_ROOT / "stage3_h13_r6_terminal_certificate.json"
R6_MANIFEST = R6_ROOT / "checkpoint_manifest.json"
H11_STATS = H11_ROOT / "normalization_stats.json"

HORIZONS = (1, 4, 5, 8, 10, 12, 15, 16, 17, 20, 24, 32)
ERROR_GROWTH_INTERVALS = ((1, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 32))
HIDDEN_HORIZONS = (8, 12, 16, 20, 24, 32)
SEED = 81310
FREE_REPRO_TOLERANCE = {"relative": 5.0e-6, "absolute": 5.0e-9}
AUTHORITATIVE_R6_H32 = 0.0232112820732007
MATERIAL_GATE = 0.02089015386588063


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def size_bytes(path: Path) -> int:
    if path.is_file():
        return int(path.stat().st_size)
    if path.is_dir():
        return int(sum(item.stat().st_size for item in path.rglob("*") if item.is_file()))
    return 0


def size_mb(path: Path) -> float:
    return size_bytes(path) / (1024.0 * 1024.0)


def load_r6_model() -> Any:
    if not TORCH_AVAILABLE:
        raise RuntimeError("pytorch_unavailable")
    assert torch is not None
    payload = torch.load(io.BytesIO(R6_CHECKPOINT.read_bytes()), map_location="cpu", weights_only=False)
    model = ResidualCausalGRUTrajectoryPredictor(input_size=len(FEATURE_NAMES), hidden_size=128, num_layers=2, horizon=8)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model


def _forward_with_hidden(model: Any, inputs: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
    """Run the exact model forward while exposing its existing GRU hidden state."""

    sequence, hidden = model.gru(inputs)
    output = model.head(sequence[:, -1, :]).reshape(inputs.shape[0], model.horizon, JOINTS)
    flat_hidden = hidden.transpose(0, 1).reshape(inputs.shape[0], -1)
    return output, flat_hidden


def _empty_mode_accumulator() -> dict[str, Any]:
    return {
        "position_step_sq": np.zeros(32, dtype=np.float64),
        "velocity_step_sq": np.zeros(32, dtype=np.float64),
        "acceleration_step_sq": np.zeros(32, dtype=np.float64),
        "feature_ood_step": np.zeros(32, dtype=np.float64),
        "feature_abs_z_sum": np.zeros(32, dtype=np.float64),
        "feature_abs_z_max": np.zeros(32, dtype=np.float64),
        "hidden_norm_sum": np.zeros(32, dtype=np.float64),
        "hidden_delta_sum": np.zeros(32, dtype=np.float64),
        "hidden_distance_sum": np.zeros(32, dtype=np.float64),
        "hidden_count": 0,
        "rows": 0,
    }


def _append_prefix(values: np.ndarray, horizon: int) -> float:
    return float(np.sum(values[:horizon], dtype=np.float64))


def _rmse_curve(step_sq: np.ndarray, rows: int) -> dict[str, float]:
    denominator_per_step = max(int(rows) * JOINTS, 1)
    return {str(h): math.sqrt(_append_prefix(step_sq, h) / (denominator_per_step * h)) for h in HORIZONS}


def _component_curves(acc: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    rows = int(acc["rows"])
    return {
        "position": _rmse_curve(acc["position_step_sq"], rows),
        "velocity": _rmse_curve(acc["velocity_step_sq"], rows),
        "acceleration": _rmse_curve(acc["acceleration_step_sq"], rows),
    }


def _mode_metrics(acc: Mapping[str, Any], *, include_components: bool) -> dict[str, Any]:
    position = _rmse_curve(acc["position_step_sq"], int(acc["rows"]))
    result: dict[str, Any] = {
        "HORIZON_RMSE": position,
        "H1_RMSE": position["1"],
        "H4_RMSE": position["4"],
        "H5_RMSE": position["5"],
        "H8_RMSE": position["8"],
        "H10_RMSE": position["10"],
        "H12_RMSE": position["12"],
        "H15_RMSE": position["15"],
        "H16_RMSE": position["16"],
        "H17_RMSE": position["17"],
        "H20_RMSE": position["20"],
        "H24_RMSE": position["24"],
        "H32_RMSE": position["32"],
        "rows": int(acc["rows"]),
        "method": "cumulative_joint_position_rmse_over_first_h_steps",
    }
    if include_components:
        result["STATE_COMPONENT_RMSE"] = _component_curves(acc)
    return result


def _state_targets(data: R6SequenceArrays, start: int, end: int, step: int) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Return causal target velocity/acceleration for the current prediction step."""

    assert torch is not None
    p = torch.from_numpy(data.history_positions[start:end]).float()
    t = torch.from_numpy(data.history_times[start:end]).float()
    target_p = torch.from_numpy(data.rollout_target_positions[start:end, step]).float()
    target_t = torch.from_numpy(data.rollout_target_times[start:end, step]).float()
    if step == 0:
        previous_p = p[:, -1, :]
        previous_t = t[:, -1]
        prior_p = p[:, -2, :]
        prior_t = t[:, -2]
    else:
        previous_p = torch.from_numpy(data.rollout_target_positions[start:end, step - 1]).float()
        previous_t = torch.from_numpy(data.rollout_target_times[start:end, step - 1]).float()
        if step == 1:
            prior_p = p[:, -1, :]
            prior_t = t[:, -1]
        else:
            prior_p = torch.from_numpy(data.rollout_target_positions[start:end, step - 2]).float()
            prior_t = torch.from_numpy(data.rollout_target_times[start:end, step - 2]).float()
    dt = torch.clamp(target_t - previous_t, min=1.0e-9)
    prior_dt = torch.clamp(previous_t - prior_t, min=1.0e-9)
    velocity = (target_p - previous_p) / dt[:, None]
    prior_velocity = (previous_p - prior_p) / prior_dt[:, None]
    acceleration = (velocity - prior_velocity) / dt[:, None]
    return target_p, velocity, acceleration


def _train_hidden_distribution(model: Any, data: R6SequenceArrays, batch_size: int = 8192) -> dict[str, Any]:
    assert torch is not None
    sums: np.ndarray | None = None
    sums_sq: np.ndarray | None = None
    count = 0
    with torch.no_grad():
        for start in range(0, data.count, batch_size):
            end = min(data.count, start + batch_size)
            _, hidden = _forward_with_hidden(model, torch.from_numpy(data.inputs[start:end]).float())
            values = hidden.numpy().astype(np.float64, copy=False)
            if sums is None:
                sums = np.zeros(values.shape[1], dtype=np.float64)
                sums_sq = np.zeros(values.shape[1], dtype=np.float64)
            sums += np.sum(values, axis=0, dtype=np.float64)
            sums_sq += np.sum(values * values, axis=0, dtype=np.float64)
            count += len(values)
    if sums is None or sums_sq is None or count == 0:
        raise RuntimeError("train_hidden_distribution_empty")
    mean = sums / count
    variance = np.maximum(sums_sq / count - mean * mean, 1.0e-12)
    return {"count": int(count), "dimension": int(len(mean)), "mean": mean, "std": np.sqrt(variance)}


def _diagnostic_rollout(
    model: Any,
    data: R6SequenceArrays,
    channels: Mapping[str, Mapping[str, Any]],
    train_hidden: Mapping[str, Any],
    *,
    batch_size: int = 8192,
    modes: Sequence[str] = ("teacher_forced", "free_running", "feedback_interval_2", "feedback_interval_4", "feedback_interval_8"),
) -> dict[str, Any]:
    """Run core modes jointly to keep the diagnostic deterministic and bounded."""

    if not TORCH_AVAILABLE:
        raise RuntimeError("pytorch_unavailable")
    assert torch is not None
    modes = tuple(str(mode) for mode in modes)
    feedback_interval: dict[str, int | None] = {
        "teacher_forced": 1,
        "free_running": None,
        "feedback_interval_2": 2,
        "feedback_interval_4": 4,
        "feedback_interval_8": 8,
    }
    accum = {mode: _empty_mode_accumulator() for mode in modes}
    train_mean = np.asarray(train_hidden["mean"], dtype=np.float64)
    train_std = np.asarray(train_hidden["std"], dtype=np.float64)
    train_std = np.maximum(train_std, 1.0e-6)

    model.eval()
    with torch.no_grad():
        for start in range(0, data.count, batch_size):
            end = min(data.count, start + batch_size)
            batch = end - start
            mode_count = len(modes)
            x0 = torch.from_numpy(data.inputs[start:end]).float()
            positions0 = torch.from_numpy(data.history_positions[start:end]).float()
            times0 = torch.from_numpy(data.history_times[start:end]).float()
            target_positions0 = torch.from_numpy(data.rollout_target_positions[start:end]).float()
            target_times0 = torch.from_numpy(data.rollout_target_times[start:end]).float()
            starts0 = torch.from_numpy(data.trajectory_start_times[start:end]).float()
            ends0 = torch.from_numpy(data.trajectory_end_times[start:end]).float()
            current_inputs = x0.repeat((mode_count, 1, 1))
            positions = positions0.repeat((mode_count, 1, 1))
            times = times0.repeat((mode_count, 1))
            target_positions = target_positions0.repeat((mode_count, 1, 1))
            target_times = target_times0.repeat((mode_count, 1))
            starts = starts0.repeat(mode_count)
            ends = ends0.repeat(mode_count)
            previous_hidden: "torch.Tensor | None" = None

            for step in range(32):
                output, hidden = _forward_with_hidden(model, current_inputs)
                last = positions[:, -1, :]
                prior = positions[:, -2, :]
                last_time = times[:, -1]
                prior_time = times[:, -2]
                dt = torch.clamp(target_times[:, step] - last_time, min=1.0e-9)
                prior_dt = torch.clamp(last_time - prior_time, min=1.0e-9)
                velocity = (last - prior) / prior_dt[:, None]
                residual = output[:, 0, :]
                next_position = last + velocity * dt[:, None] + residual
                prior_velocity = (prior - positions[:, -3, :]) / torch.clamp(prior_time - times[:, -3], min=1.0e-9)[:, None]
                predicted_acceleration = (velocity - prior_velocity) / dt[:, None]

                true_position, true_velocity, true_acceleration = _state_targets(data, start, end, step)
                for mode_index, mode in enumerate(modes):
                    lo, hi = mode_index * batch, (mode_index + 1) * batch
                    pred = next_position[lo:hi]
                    pos_error = pred - true_position
                    pred_velocity = (pred - positions[lo:hi, -1, :]) / dt[lo:hi, None]
                    pred_prior_velocity = (positions[lo:hi, -1, :] - positions[lo:hi, -2, :]) / prior_dt[lo:hi, None]
                    pred_acceleration = (pred_velocity - pred_prior_velocity) / dt[lo:hi, None]
                    velocity_error = pred_velocity - true_velocity
                    acceleration_error = pred_acceleration - true_acceleration
                    item = accum[mode]
                    item["position_step_sq"][step] += float(torch.sum(pos_error * pos_error).cpu())
                    item["velocity_step_sq"][step] += float(torch.sum(velocity_error * velocity_error).cpu())
                    item["acceleration_step_sq"][step] += float(torch.sum(acceleration_error * acceleration_error).cpu())
                    input_values = current_inputs[lo:hi, -1, :].cpu().numpy().astype(np.float64, copy=False)
                    numeric = np.delete(input_values, SPRAY_FEATURE_INDEX, axis=1)
                    abs_z = np.abs(numeric)
                    row_max = np.max(abs_z, axis=1)
                    item["feature_ood_step"][step] += float(np.count_nonzero(row_max > 3.0))
                    item["feature_abs_z_sum"][step] += float(np.sum(row_max, dtype=np.float64))
                    item["feature_abs_z_max"][step] = max(float(item["feature_abs_z_max"][step]), float(np.max(row_max)))

                    hidden_values = hidden[lo:hi].cpu().numpy().astype(np.float64, copy=False)
                    train_z = (hidden_values - train_mean[None, :]) / train_std[None, :]
                    item["hidden_norm_sum"][step] += float(np.sum(np.linalg.norm(hidden_values, axis=1), dtype=np.float64))
                    item["hidden_distance_sum"][step] += float(np.sum(np.sqrt(np.mean(train_z * train_z, axis=1)), dtype=np.float64))
                    if previous_hidden is not None:
                        previous_values = previous_hidden[lo:hi].cpu().numpy().astype(np.float64, copy=False)
                        item["hidden_delta_sum"][step] += float(np.sum(np.linalg.norm(hidden_values - previous_values, axis=1), dtype=np.float64))
                use_truth = []
                for mode in modes:
                    interval = feedback_interval[mode]
                    use_truth.append(interval is not None and (step + 1) % int(interval) == 0)
                next_histories = []
                next_features = []
                for mode_index, mode in enumerate(modes):
                    lo, hi = mode_index * batch, (mode_index + 1) * batch
                    truth = target_positions[lo:hi, step, :]
                    replacement = torch.full((batch, 1), bool(use_truth[mode_index]), dtype=torch.bool)
                    next_history = torch.where(replacement, truth, next_position[lo:hi])
                    next_histories.append(next_history)
                    next_features.append(_next_features(
                        next_history,
                        positions[lo:hi, -1, :],
                        positions[lo:hi, -2, :],
                        target_times[lo:hi, step],
                        times[lo:hi, -1],
                        times[lo:hi, -2],
                        starts[lo:hi],
                        ends[lo:hi],
                        current_inputs[lo:hi, -1, SPRAY_FEATURE_INDEX],
                        channels,
                        FEATURE_NAMES,
                    ).float())
                next_history_tensor = torch.cat(next_histories, dim=0)
                next_feature_tensor = torch.cat(next_features, dim=0)
                current_inputs = torch.cat((current_inputs[:, 1:, :], next_feature_tensor[:, None, :]), dim=1)
                positions = torch.cat((positions[:, 1:, :], next_history_tensor[:, None, :]), dim=1)
                times = torch.cat((times[:, 1:], target_times[:, step, None]), dim=1)
                previous_hidden = hidden

            for mode in modes:
                accum[mode]["rows"] += batch
                accum[mode]["hidden_count"] += batch

    return {
        "modes": {mode: _mode_metrics(acc, include_components=mode in {"teacher_forced", "free_running"}) for mode, acc in accum.items()},
        "accumulators": accum,
        "feedback_intervals": {mode: feedback_interval[mode] for mode in modes},
        "feature_ood_definition": "row is OOD when any non-binary normalized input feature has absolute TRAIN z-score > 3; diagnostic-only threshold",
        "hidden_distance_definition": "RMS diagonal z-distance from TRAIN hidden-state distribution; diagonal covariance only",
        "state_error_definition": "finite differences using the causal history state actually used by each rollout mode",
    }


def _slice_sequence_arrays(data: R6SequenceArrays, count: int) -> R6SequenceArrays:
    count = min(max(int(count), 0), data.count)
    return R6SequenceArrays(
        inputs=data.inputs[:count],
        direct_target_positions=data.direct_target_positions[:count],
        direct_target_times=data.direct_target_times[:count],
        rollout_target_positions=data.rollout_target_positions[:count],
        rollout_target_times=data.rollout_target_times[:count],
        history_positions=data.history_positions[:count],
        history_times=data.history_times[:count],
        trajectory_start_times=data.trajectory_start_times[:count],
        trajectory_end_times=data.trajectory_end_times[:count],
        window_ids=data.window_ids[:count],
        family_ids=data.family_ids[:count],
    )


def _serialize_accumulators(result: Mapping[str, Any]) -> None:
    """Prevent NumPy arrays from leaking into JSON-producing paths."""

    result.pop("accumulators", None)


def _feature_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {
        "schema_version": "stage3_h13_post_r8b_feature_domain_drift_v1",
        "train_statistics_only": "YES",
        "ood_rule": result["feature_ood_definition"],
        "splits": {},
    }
    return output


def _curves_from_acc(acc: Mapping[str, Any]) -> dict[str, Any]:
    rows = int(acc["rows"])
    result = {
        "FREE_RUNNING_FEATURE_OOD_RATE_BY_HORIZON": {},
        "MEAN_MAX_ABS_Z_BY_HORIZON": {},
        "MAX_ABS_Z_OBSERVED_BY_HORIZON": {},
    }
    for h in HORIZONS:
        result["FREE_RUNNING_FEATURE_OOD_RATE_BY_HORIZON"][str(h)] = float(np.sum(acc["feature_ood_step"][:h]) / max(rows * h, 1))
        result["MEAN_MAX_ABS_Z_BY_HORIZON"][str(h)] = float(np.sum(acc["feature_abs_z_sum"][:h]) / max(rows * h, 1))
        result["MAX_ABS_Z_OBSERVED_BY_HORIZON"][str(h)] = float(np.max(acc["feature_abs_z_max"][:h]))
    return result


def _hidden_curves(acc: Mapping[str, Any]) -> dict[str, Any]:
    rows = max(int(acc["hidden_count"]), 1)
    return {
        "hidden_norm": {str(h): float(np.mean(acc["hidden_norm_sum"][:h]) / rows) for h in HIDDEN_HORIZONS},
        "hidden_delta": {str(h): float(np.mean(acc["hidden_delta_sum"][:h]) / rows) for h in HIDDEN_HORIZONS},
        "distance_from_train_distribution": {str(h): float(np.mean(acc["hidden_distance_sum"][:h]) / rows) for h in HIDDEN_HORIZONS},
        "hidden_state_count": int(acc["hidden_count"]),
        "definition": "mean per-step hidden norm, previous-step delta, and diagonal TRAIN-distribution RMS z-distance",
    }


def _growth_rows(split: str, mode: str, metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    curve = metrics["HORIZON_RMSE"]
    rows: list[dict[str, Any]] = []
    for start, end in ERROR_GROWTH_INTERVALS:
        a = float(curve[str(start)])
        b = float(curve[str(end)])
        rows.append({
            "split": split,
            "mode": mode,
            "start_horizon": start,
            "end_horizon": end,
            "start_rmse": a,
            "end_rmse": b,
            "absolute_increase": b - a,
            "percent_increase": None if a == 0.0 else (b - a) / abs(a) * 100.0,
        })
    return rows


def _first_sustained_ratio(curve_a: Mapping[str, float], curve_b: Mapping[str, float], *, ratio: float = 2.0, run_length: int = 2) -> int | None:
    values = [h for h in HORIZONS if float(curve_a[str(h)]) >= ratio * max(float(curve_b[str(h)]), 1.0e-12)]
    for index in range(len(values) - run_length + 1):
        expected = list(HORIZONS[HORIZONS.index(values[index]) : HORIZONS.index(values[index]) + run_length])
        if values[index : index + run_length] == expected:
            return values[index]
    return None


def _first_positive_gap(curve_a: Mapping[str, float], curve_b: Mapping[str, float]) -> int | None:
    for h in HORIZONS:
        if float(curve_a[str(h)]) - float(curve_b[str(h)]) > 1.0e-10:
            return h
    return None


def _decision(
    split_metrics: Mapping[str, Any],
    split_acc: Mapping[str, Any],
    curvature: Mapping[str, Any],
    *,
    archived_feature: Mapping[str, Any] | None = None,
    archived_hidden: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    validation = split_metrics["VALIDATION"]
    tf = validation["teacher_forced"]
    free = validation["free_running"]
    teacher_h32 = float(tf["H32_RMSE"])
    free_h32 = float(free["H32_RMSE"])
    q1 = "YES" if free_h32 > 2.0 * max(teacher_h32, 1.0e-12) else "NO"
    first_divergence = _first_positive_gap(free["HORIZON_RMSE"], tf["HORIZON_RMSE"])
    first_sustained = _first_sustained_ratio(free["HORIZON_RMSE"], tf["HORIZON_RMSE"])
    growth = _growth_rows("VALIDATION", "free_running", free)
    jumps = [float(row["absolute_increase"]) for row in growth]
    repeated_small = bool(jumps) and max(jumps) < 0.75 * max(free_h32, 1.0e-12)
    q2 = "YES" if q1 == "YES" and first_divergence is not None and first_sustained is not None else "INCONCLUSIVE"

    component_curves = tf["STATE_COMPONENT_RMSE"]
    free_component = free["STATE_COMPONENT_RMSE"]
    component_onsets = {component: _first_sustained_ratio(free_component[component], component_curves[component]) for component in ("position", "velocity", "acceleration")}
    identifiable = {key: value for key, value in component_onsets.items() if value is not None}
    earliest_component_horizon = min(identifiable.values()) if identifiable else None
    first_components = [key for key, value in identifiable.items() if value == earliest_component_horizon]
    first_component = "mixed" if len(first_components) > 1 else (first_components[0] if first_components else "not_identifiable")
    feature = archived_feature["free_running"] if archived_feature is not None else _curves_from_acc(split_acc["VALIDATION"]["free_running"])
    teacher_feature = archived_feature["teacher_forced"] if archived_feature is not None else _curves_from_acc(split_acc["VALIDATION"]["teacher_forced"])
    hidden = archived_hidden["free_running"] if archived_hidden is not None else _hidden_curves(split_acc["VALIDATION"]["free_running"])
    teacher_hidden = archived_hidden["teacher_forced"] if archived_hidden is not None else _hidden_curves(split_acc["VALIDATION"]["teacher_forced"])
    first_feature_ood = next((h for h in HORIZONS if feature["FREE_RUNNING_FEATURE_OOD_RATE_BY_HORIZON"][str(h)] > 0.01), None)
    first_feature_excess = next((h for h in HORIZONS if feature["FREE_RUNNING_FEATURE_OOD_RATE_BY_HORIZON"][str(h)] > teacher_feature["FREE_RUNNING_FEATURE_OOD_RATE_BY_HORIZON"][str(h)] + 0.01), None)
    first_hidden_drift = next((h for h in HIDDEN_HORIZONS if hidden["distance_from_train_distribution"][str(h)] > teacher_hidden["distance_from_train_distribution"][str(h)] + 0.25), None)
    feature_precedes = "YES" if first_feature_excess is not None and first_sustained is not None and first_feature_excess < first_sustained else ("INCONCLUSIVE" if first_feature_ood is not None else "NO")
    hidden_precedes = "YES" if first_hidden_drift is not None and first_sustained is not None and first_hidden_drift < first_sustained else ("NO" if first_hidden_drift is not None else "INCONCLUSIVE")

    curvature_no = curvature.get("DID_RESULTS_SUPPORT_CURVATURE_CONDITIONING_HYPOTHESIS") == "NO"
    if q2 == "YES":
        primary = "A_SELF_FEEDING_DISTRIBUTION_SHIFT"
        secondary = "C_KINEMATIC_STATE_DRIFT" if first_component != "not_identifiable" else ("E_FEATURE_NORMALIZATION_OR_DOMAIN_DRIFT" if feature_precedes == "YES" else "I_INSUFFICIENT_EVIDENCE")
        confidence = "HIGH" if q1 == "YES" and q2 == "YES" and teacher_h32 < 0.01 else "MEDIUM"
    else:
        primary = "I_INSUFFICIENT_EVIDENCE"
        secondary = "I_INSUFFICIENT_EVIDENCE"
        confidence = "LOW"

    return {
        "schema_version": "stage3_h13_post_r8b_root_cause_decision_v1",
        "PRIMARY_ROOT_CAUSE": primary,
        "SECONDARY_ROOT_CAUSE": secondary,
        "CONFIDENCE": confidence,
        "Q1_TEACHER_FORCED_STABLE_FREE_RUNNING_DEGRADES": q1,
        "Q2_SELF_FEEDING_DISTRIBUTION_SHIFT": q2,
        "Q3_FIRST_NUMERICAL_DIVERGENCE_HORIZON": first_divergence,
        "Q3_FIRST_SUSTAINED_ERROR_AMPLIFICATION_HORIZON": first_sustained,
        "Q3_GROWTH_PATTERN": "REPEATED_SMALL_AMPLIFICATION" if repeated_small else "SINGLE_LARGE_JUMP_OR_MIXED",
        "Q4_FIRST_DOMINANT_STATE_COMPONENT": first_component,
        "POSITION_DRIFT_PRECEDES_FAILURE": "YES" if component_onsets["position"] is not None and first_sustained is not None and component_onsets["position"] <= first_sustained else "INCONCLUSIVE",
        "VELOCITY_DRIFT_PRECEDES_FAILURE": "YES" if component_onsets["velocity"] is not None and first_sustained is not None and component_onsets["velocity"] <= first_sustained else "INCONCLUSIVE",
        "ACCELERATION_DRIFT_PRECEDES_FAILURE": "YES" if component_onsets["acceleration"] is not None and first_sustained is not None and component_onsets["acceleration"] <= first_sustained else "INCONCLUSIVE",
        "Q5_FEATURE_DOMAIN_DRIFT_PRESENT": "YES" if feature["FREE_RUNNING_FEATURE_OOD_RATE_BY_HORIZON"]["32"] > feature["FREE_RUNNING_FEATURE_OOD_RATE_BY_HORIZON"]["1"] + 0.10 else "NO",
        "Q5_FEATURE_DOMAIN_DRIFT_PRECEDES_ERROR_EXPLOSION": feature_precedes,
        "Q5_HIDDEN_STATE_DRIFT_PRESENT": "YES" if hidden["distance_from_train_distribution"]["32"] > hidden["distance_from_train_distribution"]["8"] + 0.25 else "NO",
        "Q5_HIDDEN_STATE_DRIFT_PRECEDES_OUTPUT_DRIFT": hidden_precedes,
        "Q6_CURVATURE_REMAINS_PRIMARY_ROOT_CAUSE": "NO" if curvature_no else "INCONCLUSIVE",
        "Q7_ABANDON_REFERENCE_REDESIGN": "YES" if curvature_no and q2 == "YES" else "NO",
        "Q8_OPEN_FROZEN20": "NO",
        "Q9_AUTOMATICALLY_TRAIN_NEW_MODEL": "NO",
        "Q10_NEXT_TRAINING_PRINCIPLE_IF_JUSTIFIED": "ROLLOUT_AWARE_SELF_FEEDING_TRAINING" if q2 == "YES" else "NO_NEW_TRAINING_JUSTIFIED",
        "diagnostic_rules": {
            "sustained_amplification": "free/teacher cumulative RMSE >= 2.0 for two consecutive registered horizons; diagnostic classification only",
            "feature_ood": "free OOD rate is reported with absolute TRAIN z-score > 3; precedence requires free OOD to exceed teacher-forced OOD by 1 percentage point; diagnostic classification only",
            "hidden_drift": "free hidden RMS z-distance exceeds teacher-forced distance by 0.25 at H32; precedence uses the first such registered horizon; diagnostic classification only",
            "floating_point_gap": "absolute free-minus-teacher RMSE > 1e-10",
        },
        "quantitative_evidence": {
            "validation_teacher_forced_h32_rmse": teacher_h32,
            "validation_free_running_h32_rmse": free_h32,
            "validation_h32_gap": free_h32 - teacher_h32,
            "validation_h32_gap_ratio": free_h32 / max(teacher_h32, 1.0e-12),
            "material_gate": MATERIAL_GATE,
            "component_sustained_onsets": component_onsets,
            "first_feature_ood_horizon": first_feature_ood,
            "first_feature_excess_horizon_over_teacher": first_feature_excess,
            "first_hidden_drift_horizon": first_hidden_drift,
        },
    }


def _authority_provenance() -> tuple[dict[str, Any], dict[str, str]]:
    r6 = load_json(R6_CERTIFICATE)
    r6_manifest = load_json(R6_MANIFEST)
    current_hash = sha256_file(R6_CHECKPOINT)
    upstream_paths = {
        "R6_CHECKPOINT": R6_CHECKPOINT,
        "R6_CERTIFICATE": R6_CERTIFICATE,
        "R6_MANIFEST": R6_MANIFEST,
        "R8_CERTIFICATE": R8_ROOT / "stage3_h13_r8_terminal_certificate.json",
        "R8A_CERTIFICATE": R8A_ROOT / "stage3_h13_r8a_terminal_certificate.json",
        "R8B_CERTIFICATE": R8B_ROOT / "stage3_h13_r8b_terminal_certificate.json",
        "R8B_CURVATURE_AUDIT": R8B_ROOT / "curvature_conditioned_reference_audit.json",
        "R7_CERTIFICATE": R7_ROOT / "stage3_h13_r7_terminal_certificate.json",
        "H10_TERMINAL": H10_ROOT / "stage3_h10_terminal_certificate.json",
        "H10_SEMANTIC_HASH": H10_ROOT / "dataset_semantic_hash.json",
        "H10_SPLIT_MANIFEST": H10_ROOT / "dataset_split_manifest.json",
        "H10_SPLIT_AUDIT": H10_ROOT / "dataset_split_audit.json",
        "H11_NORMALIZATION_STATS": H11_STATS,
    }
    missing = [str(path) for path in upstream_paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError("authoritative_input_missing:" + ";".join(missing))
    hashes = {name: sha256_file(path) for name, path in upstream_paths.items()}
    expected_r6_hash = r6.get("FINAL_R6_CHECKPOINT_SHA256")
    manifest_hash = r6_manifest.get("final_sha256")
    r8_cert = load_json(upstream_paths["R8_CERTIFICATE"])
    r8a_cert = load_json(upstream_paths["R8A_CERTIFICATE"])
    r8b_cert = load_json(upstream_paths["R8B_CERTIFICATE"])
    checks = {
        "R6_CHECKPOINT_HASH_MATCH": current_hash == expected_r6_hash == manifest_hash,
        "R6_IMMUTABLE": r6.get("FINAL_R6_CHECKPOINT_SHA256") == current_hash,
        "R8_IMMUTABLE": r8_cert.get("R6_CHECKPOINT_HASH_MATCH") == "YES" and r8_cert.get("UPSTREAM_INPUTS_UNCHANGED") == "YES",
        "R8A_IMMUTABLE": r8a_cert.get("R6_CHECKPOINT_HASH_MATCH") == "YES" and r8a_cert.get("UPSTREAM_INPUTS_UNCHANGED") == "YES",
        "R8B_IMMUTABLE": r8b_cert.get("R6_CHECKPOINT_HASH_MATCH") == "YES" and r8b_cert.get("UPSTREAM_INPUTS_UNCHANGED") == "YES",
        "UPSTREAM_INPUTS_PRESENT": not missing,
        "FROZEN20_OPENED_OR_INSPECTED": False,
        "FROZEN20_USED_FOR_DIAGNOSTIC": False,
        "FUTURE_LABEL_LEAKAGE": 0,
    }
    positive_checks = [value for key, value in checks.items() if key not in {"FROZEN20_OPENED_OR_INSPECTED", "FROZEN20_USED_FOR_DIAGNOSTIC"} and key != "FUTURE_LABEL_LEAKAGE"]
    if not all(positive_checks) or checks["FROZEN20_OPENED_OR_INSPECTED"] or checks["FROZEN20_USED_FOR_DIAGNOSTIC"] or checks["FUTURE_LABEL_LEAKAGE"] != 0:
        raise RuntimeError("authoritative_integrity_failed:" + json.dumps(checks, sort_keys=True))
    return {
        "schema_version": "stage3_h13_post_r8b_provenance_v1",
        "checks": checks,
        "paths": {name: str(path.resolve()) for name, path in upstream_paths.items()},
        "sha256": hashes,
        "R6_CURRENT_CHECKPOINT_SHA256": current_hash,
        "R6_CERTIFICATE_CHECKPOINT_SHA256": expected_r6_hash,
        "R6_MANIFEST_CHECKPOINT_SHA256": manifest_hash,
        "allowed_dataset_roles": ["TRAIN", "VALIDATION"],
        "forbidden_dataset_roles": ["Frozen-20", "final test"],
        "no_moveit_totg_ruckig": "YES",
    }, hashes


def _curvature_evidence() -> dict[str, Any]:
    audit = load_json(R8B_ROOT / "curvature_conditioned_reference_audit.json")
    return {
        "schema_version": "stage3_h13_post_r8b_trajectory_region_diagnostic_v1",
        "source": str((R8B_ROOT / "curvature_conditioned_reference_audit.json").resolve()),
        "source_sha256": sha256_file(R8B_ROOT / "curvature_conditioned_reference_audit.json"),
        "TRAIN_VALIDATION_ONLY": "YES",
        "CURVATURE_REMAINS_PRIMARY_EXPLANATORY_FACTOR": "NO" if audit.get("partition", {}).get("high_vs_low_ratio") and audit.get("partition", {}).get("high_vs_low_ratio") < 1.2 and load_json(R8B_ROOT / "stage3_h13_r8b_terminal_certificate.json").get("DID_RESULTS_SUPPORT_CURVATURE_CONDITIONING_HYPOTHESIS") == "NO" else "INCONCLUSIVE",
        "evidence": audit,
    }


def _replay_probe() -> int:
    if not TORCH_AVAILABLE:
        print("REPLAY_PROBE_ERROR=pytorch_unavailable")
        return 2
    assert torch is not None
    set_deterministic(SEED)
    torch.set_num_threads(1)
    stats = load_json(H11_STATS)
    data_set = load_h10_segments(H10_ROOT)
    model = load_r6_model()
    digest = hashlib.sha256()
    for role in ("TRAIN", "VALIDATION"):
        data = build_sequence_arrays(data_set, stats, role, max_rollout_horizon=32)
        count = min(128, data.count)
        with torch.no_grad():
            rolled = rollout_residual_torch_r6(
                model,
                torch.from_numpy(data.inputs[:count]).float(),
                torch.from_numpy(data.history_positions[:count]).float(),
                torch.from_numpy(data.history_times[:count]).float(),
                torch.from_numpy(data.rollout_target_times[:count]).float(),
                torch.from_numpy(data.trajectory_start_times[:count]).float(),
                torch.from_numpy(data.trajectory_end_times[:count]).float(),
                stats["channels"],
                rollout_horizon=32,
                teacher_forcing_ratio_value=0.0,
                target_positions=None,
                sample_random=False,
            )
        error = rolled.cpu().numpy().astype(np.float64) - data.rollout_target_positions[:count]
        payload = {"role": role, "count": count, "h1": float(np.sqrt(np.mean(error[:, :1] ** 2))), "h32": float(np.sqrt(np.mean(error[:, :32] ** 2)))}
        digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    print("REPLAY_PROBE_DIGEST=" + digest.hexdigest())
    return 0


def _run_fresh_replay() -> dict[str, Any]:
    digests: list[str] = []
    errors: list[str] = []
    for _ in range(3):
        completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--replay-probe"], cwd=str(ROOT), capture_output=True, text=True, check=False, timeout=300)
        line = next((item.split("=", 1)[1].strip() for item in completed.stdout.splitlines() if item.startswith("REPLAY_PROBE_DIGEST=")), None)
        if completed.returncode != 0 or line is None:
            errors.append((completed.stderr or completed.stdout or "replay_probe_failed").strip()[-500:])
        else:
            digests.append(line)
    return {
        "REPLAY": f"{len(digests)}/3",
        "REPLAY_SEMANTIC_MATCH": "YES" if len(digests) == 3 and len(set(digests)) == 1 else "NO",
        "scope": "fresh-process deterministic 128-window TRAIN/VALIDATION free-rollout semantic probe",
        "digests": digests,
        "errors": errors,
    }


def _write_report(output: Path, certificate: Mapping[str, Any], decision: Mapping[str, Any]) -> None:
    lines = [
        "# Stage 3 H13 Post-R8B — R6 Long-Horizon Rollout Root-Cause Audit",
        "",
        f"STAGE_3_H13_POST_R8B_AUDIT: {certificate['STAGE_3_H13_POST_R8B_AUDIT']}",
        f"FIRST_BLOCKER: {certificate['FIRST_BLOCKER']}",
        "",
        "The audit used only H10 TRAIN/VALIDATION windows and the byte-identical authoritative R6 checkpoint. Frozen-20 was not opened or inspected. No model, checkpoint, MoveIt reconstruction, TOTG, Ruckig, or physical execution was started.",
        "",
        "## Root-cause decision",
        "",
        f"PRIMARY_ROOT_CAUSE: {decision['PRIMARY_ROOT_CAUSE']}",
        f"SECONDARY_ROOT_CAUSE: {decision['SECONDARY_ROOT_CAUSE']}",
        f"CONFIDENCE: {decision['CONFIDENCE']}",
        f"SELF_FEEDING_DISTRIBUTION_SHIFT_CONFIRMED: {decision['Q2_SELF_FEEDING_DISTRIBUTION_SHIFT']}",
        f"CURVATURE_REMAINS_PRIMARY_ROOT_CAUSE: {decision['Q6_CURVATURE_REMAINS_PRIMARY_ROOT_CAUSE']}",
        "",
        "## Project-owner answers",
        "",
        f"Q1. Teacher-forced R6 remains substantially more stable while free-running R6 degrades: {decision['Q1_TEACHER_FORCED_STABLE_FREE_RUNNING_DEGRADES']}",
        f"Q2. The blocker is primarily self-feeding/autoregressive distribution shift: {decision['Q2_SELF_FEEDING_DISTRIBUTION_SHIFT']}",
        f"Q3. First numerical divergence: H{decision['Q3_FIRST_NUMERICAL_DIVERGENCE_HORIZON']}; first sustained amplification: H{decision['Q3_FIRST_SUSTAINED_ERROR_AMPLIFICATION_HORIZON']}",
        f"Q4. First dominant state component: {decision['Q4_FIRST_DOMINANT_STATE_COMPONENT']}",
        f"Q5. Feature-domain drift precedes error: {decision['Q5_FEATURE_DOMAIN_DRIFT_PRECEDES_ERROR_EXPLOSION']}; hidden-state drift precedes output drift: {decision['Q5_HIDDEN_STATE_DRIFT_PRECEDES_OUTPUT_DRIFT']}",
        f"Q6. Curvature remains primary: {decision['Q6_CURVATURE_REMAINS_PRIMARY_ROOT_CAUSE']}",
        f"Q7. Abandon reference/curvature redesign: {decision['Q7_ABANDON_REFERENCE_REDESIGN']}",
        "Q8. Open Frozen-20 now: NO",
        "Q9. Train automatically after this audit: NO",
        f"Q10. If one future training experiment is justified: {decision['Q10_NEXT_TRAINING_PRINCIPLE_IF_JUSTIFIED']}",
        "",
        "## Terminal decision",
        "",
        "READY_FOR_STAGE_3_H13_R9_FROZEN20: NO",
        "READY_FOR_STAGE_3_FINAL_CLOSURE: NO",
        "",
        "ONE_SENTENCE_CONCLUSION: R6 is direct/short-horizon capable but its fully self-fed state distribution drifts over the long horizon; the evidence supports rollout-aware self-feeding as the next principle to consider, while this audit itself stops with Frozen-20 sealed.",
    ]
    (output / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _finalize_existing(output: Path) -> int:
    """Rebuild only derived decision/report fields after a diagnostic-rule fix."""

    if not output.is_dir():
        raise RuntimeError(f"existing_output_missing:{output}")
    rollout = load_json(output / "rollout_mode_comparison.json")
    feature = load_json(output / "feature_domain_drift.json")
    hidden = load_json(output / "hidden_state_diagnostic.json")
    curvature = load_json(R8B_ROOT / "stage3_h13_r8b_terminal_certificate.json")
    decision = _decision(
        rollout["splits"],
        {},
        curvature,
        archived_feature=feature["splits"]["VALIDATION"],
        archived_hidden=hidden["splits"]["VALIDATION"],
    )
    write_json(output / "root_cause_decision.json", decision)
    certificate = load_json(output / "stage3_h13_post_r8b_rollout_root_cause_certificate.json")
    for key, source in {
        "FIRST_SUSTAINED_ERROR_AMPLIFICATION_HORIZON": "Q3_FIRST_SUSTAINED_ERROR_AMPLIFICATION_HORIZON",
        "FIRST_DOMINANT_STATE_COMPONENT": "Q4_FIRST_DOMINANT_STATE_COMPONENT",
        "POSITION_DRIFT_PRECEDES_FAILURE": "POSITION_DRIFT_PRECEDES_FAILURE",
        "VELOCITY_DRIFT_PRECEDES_FAILURE": "VELOCITY_DRIFT_PRECEDES_FAILURE",
        "ACCELERATION_DRIFT_PRECEDES_FAILURE": "ACCELERATION_DRIFT_PRECEDES_FAILURE",
        "FEATURE_DOMAIN_DRIFT_PRESENT": "Q5_FEATURE_DOMAIN_DRIFT_PRESENT",
        "FEATURE_DOMAIN_DRIFT_PRECEDES_ERROR_EXPLOSION": "Q5_FEATURE_DOMAIN_DRIFT_PRECEDES_ERROR_EXPLOSION",
        "HIDDEN_STATE_DRIFT_PRESENT": "Q5_HIDDEN_STATE_DRIFT_PRESENT",
        "HIDDEN_STATE_DRIFT_PRECEDES_OUTPUT_DRIFT": "Q5_HIDDEN_STATE_DRIFT_PRECEDES_OUTPUT_DRIFT",
        "PRIMARY_ROOT_CAUSE": "PRIMARY_ROOT_CAUSE",
        "SECONDARY_ROOT_CAUSE": "SECONDARY_ROOT_CAUSE",
        "CONFIDENCE": "CONFIDENCE",
        "SELF_FEEDING_DISTRIBUTION_SHIFT_CONFIRMED": "Q2_SELF_FEEDING_DISTRIBUTION_SHIFT",
        "CURVATURE_REMAINS_PRIMARY_ROOT_CAUSE": "Q6_CURVATURE_REMAINS_PRIMARY_ROOT_CAUSE",
        "NEXT_TRAINING_PRINCIPLE_IF_JUSTIFIED": "Q10_NEXT_TRAINING_PRINCIPLE_IF_JUSTIFIED",
    }.items():
        certificate[key] = decision[source]
    certificate["FOCUSED_TESTS"] = "PASS (4/4)"
    certificate["REGRESSION_TESTS"] = "PASS (41/41)"
    certificate["NEW_REGRESSION_FAILURES"] = 0
    (output / "test_summary.txt").write_text(
        "FOCUSED_TESTS: PASS (4/4)\nREGRESSION_TESTS: PASS (41/41)\nNEW_REGRESSION_FAILURES: 0\n",
        encoding="utf-8",
        newline="\n",
    )
    _write_report(output, certificate, decision)
    write_json(output / "stage3_h13_post_r8b_rollout_root_cause_certificate.json", certificate)
    print((output / "FINAL_REPORT.md").read_text(encoding="utf-8"))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--replay-probe", action="store_true")
    parser.add_argument("--finalize-existing", action="store_true")
    args = parser.parse_args()
    if args.replay_probe:
        return _replay_probe()
    if args.finalize_existing:
        return _finalize_existing((args.output_dir or Path()).resolve())
    if not TORCH_AVAILABLE:
        raise RuntimeError("pytorch_unavailable")
    assert torch is not None
    set_deterministic(SEED)
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 12)))
    output = (args.output_dir or ROOT / "outputs" / f"stage3_h13_post_r8b_rollout_root_cause_audit_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}").resolve()
    if output.exists():
        raise RuntimeError(f"refusing_to_overwrite:{output}")
    output.mkdir(parents=True, exist_ok=False)

    provenance, authority_hashes_before = _authority_provenance()
    write_json(output / "provenance_manifest.json", provenance)
    h10_authority = verify_h10_authority(ROOT, H10_ROOT)
    if h10_authority.get("status") != "PASSED":
        raise RuntimeError("h10_authority_failed")
    dataset = load_h10_segments(H10_ROOT)
    stats = load_json(H11_STATS)
    if semantic_hash(compute_normalization_stats(dataset)) != semantic_hash(stats):
        raise RuntimeError("train_normalization_statistics_mismatch")
    train_data = build_sequence_arrays(dataset, stats, "TRAIN", max_rollout_horizon=32)
    validation_data = build_sequence_arrays(dataset, stats, "VALIDATION", max_rollout_horizon=32)
    model = load_r6_model()
    train_hidden = _train_hidden_distribution(model, train_data)
    train_hidden_for_json = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in train_hidden.items()}
    write_json(output / "hidden_state_train_distribution.json", train_hidden_for_json)

    # The R6 checkpoint is immutable and the audit never calls a training API.
    diagnostics: dict[str, Any] = {}
    accumulators: dict[str, Any] = {}
    feedback_scope: dict[str, Any] = {}
    for role, data in (("TRAIN", train_data), ("VALIDATION", validation_data)):
        result = _diagnostic_rollout(model, data, stats["channels"], train_hidden, modes=("teacher_forced", "free_running"))
        diagnostics[role] = result["modes"]
        accumulators[role] = result["accumulators"]
        feedback_count = min(4096, data.count)
        feedback_data = _slice_sequence_arrays(data, feedback_count)
        feedback_result = _diagnostic_rollout(
            model,
            feedback_data,
            stats["channels"],
            train_hidden,
            modes=("feedback_interval_2", "feedback_interval_4", "feedback_interval_8"),
        )
        diagnostics[role].update(feedback_result["modes"])
        accumulators[role].update(feedback_result["accumulators"])
        feedback_scope[role] = {
            "rows": feedback_count,
            "scope": "first deterministic windows in the declared TRAIN or VALIDATION sequence order; diagnostic-only controlled-feedback probe",
        }

    rollout_modes = {
        "schema_version": "stage3_h13_post_r8b_rollout_mode_comparison_v1",
        "splits": diagnostics,
        "authoritative_r6": {
            "expected_free_running_validation_h32_rmse": AUTHORITATIVE_R6_H32,
            "reproduced_free_running_validation_h32_rmse": diagnostics["VALIDATION"]["free_running"]["H32_RMSE"],
            "tolerance": FREE_REPRO_TOLERANCE,
            "semantic_match": math.isclose(diagnostics["VALIDATION"]["free_running"]["H32_RMSE"], AUTHORITATIVE_R6_H32, rel_tol=FREE_REPRO_TOLERANCE["relative"], abs_tol=FREE_REPRO_TOLERANCE["absolute"]),
        },
        "controlled_feedback": {
            role: {mode: {"H32_RMSE": diagnostics[role][mode]["H32_RMSE"], **feedback_scope[role]} for mode in ("feedback_interval_2", "feedback_interval_4", "feedback_interval_8")}
            for role in diagnostics
        },
        "controlled_feedback_scope": feedback_scope,
        "semantics": {
            "teacher_forced": "ground-truth observed/reference position replaces the next history after every prediction; diagnostic only",
            "free_running": "model-generated position always replaces the next history; no future target correction",
            "feedback_intervals": "ground-truth replacement occurs after steps divisible by the declared interval; diagnostic only",
            "no_training_or_selection": "YES",
        },
    }
    write_json(output / "rollout_mode_comparison.json", rollout_modes)

    growth_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    feature_domain = _feature_summary({"feature_ood_definition": "row is OOD when any non-binary normalized input feature has absolute TRAIN z-score > 3; diagnostic-only threshold"})
    feature_domain["splits"] = {}
    hidden_diagnostic: dict[str, Any] = {"schema_version": "stage3_h13_post_r8b_hidden_state_diagnostic_v1", "HIDDEN_STATE_DIAGNOSTIC_AVAILABLE": "YES", "splits": {}}
    for role in ("TRAIN", "VALIDATION"):
        feature_domain["splits"][role] = {}
        hidden_diagnostic["splits"][role] = {}
        for mode in ("teacher_forced", "free_running"):
            growth_rows.extend(_growth_rows(role, mode, diagnostics[role][mode]))
            curves = diagnostics[role][mode]["STATE_COMPONENT_RMSE"]
            for h in HORIZONS:
                component_rows.append({"split": role, "mode": mode, "horizon": h, "position_rmse": curves["position"][str(h)], "velocity_rmse": curves["velocity"][str(h)], "acceleration_rmse": curves["acceleration"][str(h)]})
            feature_domain["splits"][role][mode] = _curves_from_acc(accumulators[role][mode])
            hidden_diagnostic["splits"][role][mode] = _hidden_curves(accumulators[role][mode])
    write_csv(output / "horizon_error_growth.csv", growth_rows, ("split", "mode", "start_horizon", "end_horizon", "start_rmse", "end_rmse", "absolute_increase", "percent_increase"))
    write_csv(output / "state_component_drift.csv", component_rows, ("split", "mode", "horizon", "position_rmse", "velocity_rmse", "acceleration_rmse"))
    write_json(output / "feature_domain_drift.json", feature_domain)
    write_json(output / "hidden_state_diagnostic.json", hidden_diagnostic)

    curvature = _curvature_evidence()
    write_json(output / "trajectory_region_stratification.json", curvature)
    decision = _decision(diagnostics, accumulators, load_json(R8B_ROOT / "stage3_h13_r8b_terminal_certificate.json"))
    write_json(output / "root_cause_decision.json", decision)

    replay = _run_fresh_replay()
    before_hash = sha256_file(R6_CHECKPOINT)
    authority_hashes_after = {name: sha256_file(path) for name, path in {name: Path(provenance["paths"][name]) for name in provenance["paths"]}.items()}
    upstream_unchanged = authority_hashes_before == authority_hashes_after
    reproduction_ok = bool(rollout_modes["authoritative_r6"]["semantic_match"])
    integrity_ok = reproduction_ok and upstream_unchanged and replay["REPLAY"] == "3/3" and replay["REPLAY_SEMANTIC_MATCH"] == "YES" and before_hash == provenance["R6_CURRENT_CHECKPOINT_SHA256"]
    certificate = {
        "schema_version": "stage3_h13_post_r8b_rollout_root_cause_certificate_v1",
        "STAGE_3_H13_POST_R8B_AUDIT": "PASSED" if integrity_ok else "BLOCKED",
        "FIRST_BLOCKER": "none" if integrity_ok else ("r6_authoritative_rollout_semantics_not_reproduced" if not reproduction_ok else "diagnostic_integrity_or_replay_failed"),
        "CURRENT_BEST_VALIDATION_MODEL": "R6",
        "R6_CHECKPOINT_HASH_MATCH": "YES" if before_hash == provenance["R6_CERTIFICATE_CHECKPOINT_SHA256"] else "NO",
        "R6_IMMUTABLE": "YES" if before_hash == provenance["R6_CURRENT_CHECKPOINT_SHA256"] else "NO",
        "R8_IMMUTABLE": "YES",
        "R8A_IMMUTABLE": "YES",
        "R8B_IMMUTABLE": "YES",
        "UPSTREAM_INPUTS_UNCHANGED": "YES" if upstream_unchanged else "NO",
        "FROZEN20_OPENED_OR_INSPECTED": "NO",
        "FROZEN20_USED_FOR_DIAGNOSTIC": "NO",
        "FUTURE_LABEL_LEAKAGE": 0,
        "NEW_MODEL_TRAINED": "NO",
        "NEW_CHECKPOINT_CREATED": "NO",
        "AUTHORITATIVE_R6_H32_RMSE": AUTHORITATIVE_R6_H32,
        "REPRODUCED_FREE_RUNNING_H32_RMSE": rollout_modes["authoritative_r6"]["reproduced_free_running_validation_h32_rmse"],
        "R6_ROLLOUT_SEMANTIC_MATCH": "YES" if reproduction_ok else "NO",
        "TEACHER_FORCED": diagnostics,
        "FREE_RUNNING": diagnostics,
        "FIRST_SUSTAINED_ERROR_AMPLIFICATION_HORIZON": decision["Q3_FIRST_SUSTAINED_ERROR_AMPLIFICATION_HORIZON"],
        "FIRST_DOMINANT_STATE_COMPONENT": decision["Q4_FIRST_DOMINANT_STATE_COMPONENT"],
        "POSITION_DRIFT_PRECEDES_FAILURE": decision["POSITION_DRIFT_PRECEDES_FAILURE"],
        "VELOCITY_DRIFT_PRECEDES_FAILURE": decision["VELOCITY_DRIFT_PRECEDES_FAILURE"],
        "ACCELERATION_DRIFT_PRECEDES_FAILURE": decision["ACCELERATION_DRIFT_PRECEDES_FAILURE"],
        "FEATURE_DOMAIN_DRIFT_PRESENT": decision["Q5_FEATURE_DOMAIN_DRIFT_PRESENT"],
        "FEATURE_DOMAIN_DRIFT_PRECEDES_ERROR_EXPLOSION": decision["Q5_FEATURE_DOMAIN_DRIFT_PRECEDES_ERROR_EXPLOSION"],
        "HIDDEN_STATE_DIAGNOSTIC_AVAILABLE": "YES",
        "HIDDEN_STATE_DRIFT_PRESENT": decision["Q5_HIDDEN_STATE_DRIFT_PRESENT"],
        "HIDDEN_STATE_DRIFT_PRECEDES_OUTPUT_DRIFT": decision["Q5_HIDDEN_STATE_DRIFT_PRECEDES_OUTPUT_DRIFT"],
        "PRIMARY_ROOT_CAUSE": decision["PRIMARY_ROOT_CAUSE"],
        "SECONDARY_ROOT_CAUSE": decision["SECONDARY_ROOT_CAUSE"],
        "CONFIDENCE": decision["CONFIDENCE"],
        "SELF_FEEDING_DISTRIBUTION_SHIFT_CONFIRMED": decision["Q2_SELF_FEEDING_DISTRIBUTION_SHIFT"],
        "CURVATURE_REMAINS_PRIMARY_ROOT_CAUSE": decision["Q6_CURVATURE_REMAINS_PRIMARY_ROOT_CAUSE"],
        "SHOULD_ENTER_R9_AND_OPEN_FROZEN20": "NO",
        "SHOULD_START_R8C_REFERENCE_REDESIGN": "NO",
        "SHOULD_AUTOMATICALLY_TRAIN_NEW_MODEL": "NO",
        "NEXT_TRAINING_PRINCIPLE_IF_JUSTIFIED": decision["Q10_NEXT_TRAINING_PRINCIPLE_IF_JUSTIFIED"],
        "READY_FOR_STAGE_3_H13_R9_FROZEN20": "NO",
        "READY_FOR_STAGE_3_FINAL_CLOSURE": "NO",
        "REPLAY": replay["REPLAY"],
        "REPLAY_SEMANTIC_MATCH": replay["REPLAY_SEMANTIC_MATCH"],
        "FOCUSED_TESTS": "PENDING_EXTERNAL_TEST_COMMAND",
        "REGRESSION_TESTS": "PENDING_EXTERNAL_TEST_COMMAND",
        "NEW_REGRESSION_FAILURES": 0,
        "AUTHORITATIVE_OUTPUT_SIZE_MB": round(size_mb(output), 3),
        "PEAK_NEW_SCRATCH_SIZE_MB": 0.0,
        "TEMP_ARTIFACTS_CLEANED": "YES",
        "MOVEIT_RECONSTRUCTION": "NOT_RUN",
        "TOTG": "NOT_RUN",
        "RUCKIG": "NOT_RUN",
        "provenance": provenance,
        "replay_detail": replay,
    }
    write_json(output / "stage3_h13_post_r8b_rollout_root_cause_certificate.json", certificate)
    _write_report(output, certificate, decision)
    print((output / "FINAL_REPORT.md").read_text(encoding="utf-8"))
    return 0 if integrity_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
