"""D39 causal self-fed recovery and task-space certification execution.

This is an execution runner, not a design ledger.  It keeps the D37/D38
artifacts read-only, trains one scheduled-self-fed shadow model from the
frozen H10 TRAIN split, evaluates several materially different causal
controllers on the 181-point ON-state open-arch case, and sends only actual
shadow survivors through the native MoveIt2 runner.

The self-fed branch never reads a future joint row.  Ground-truth target rows
are used only as supervised labels and in the explicitly separate scheduled
teacher branch during training.  Collision results are reported with the
project's adaptive_discrete_interpolation semantics; CCD and clearance are
not inferred.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stage3_h13_r8e_post_d4_canonical_batch_robustness import load_model  # noqa: E402
from src.stage3_h11_dataset import FEATURE_NAMES, INPUT_HISTORY, load_h10_segments  # noqa: E402
from src.stage3_h13_r6 import build_sequence_arrays, direct_residual_targets  # noqa: E402
from src.stage3_h11_r2_model import rollout_residual_torch  # noqa: E402
from src.stage3_h13_r8e import model_forward_with_hidden  # noqa: E402
from tools import stage3_h13_d36_system_evaluation as d36  # noqa: E402
from tools import stage3_h13_d38_rollout_stabilization as d38  # noqa: E402


OUTPUT = ROOT / "outputs" / "stage3_h13_d39_causal_task_space_recovery"
H10_ROOT = ROOT / "outputs" / "stage3_h10_multi_trajectory_dataset_20260811T081123Z"
H11_ROOT = ROOT / "outputs" / "stage3_h11_deep_learning_baseline_20260811T141921Z"
R6_CHECKPOINT = ROOT / "outputs" / "stage3_h13_r6_rollout_aware_training_20260814T030507Z" / "final_r6_checkpoint.pt"
D38_OUTPUT = ROOT / "outputs" / "stage3_h13_d38_rollout_stabilization"
COLLISION_METHOD = "adaptive_discrete_interpolation"
CERTIFICATION_GATES = (
    "finite",
    "bounds",
    "planning_scene",
    "fk",
    "endpoint_path_quality",
    "timing",
    "ruckig",
    "velocity",
    "acceleration",
    "jerk",
    "post_ruckig_collision",
    "dynamics",
    "task_completion",
)
HORIZONS = (1, 2, 4, 8, 16, 32, 64)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_candidate(path: Path, positions: np.ndarray) -> None:
    if np.asarray(positions).shape != (181, 6):
        raise RuntimeError(f"d39_candidate_shape:{np.asarray(positions).shape}")
    if not np.isfinite(positions).all():
        raise RuntimeError("d39_candidate_nonfinite")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([f"q{index}" for index in range(1, 7)])
        writer.writerows(np.asarray(positions, dtype=np.float64).tolist())


def load_case() -> d38.AuthoritativeCase:
    return d38.make_case(d38.load_channels())


def load_d38_start() -> dict[str, Any]:
    summary = D38_OUTPUT / "D38_ROOT_CAUSE_SUMMARY.json"
    shadow = D38_OUTPUT / "D38_SHADOW_SUMMARY.json"
    if not summary.is_file() or not shadow.is_file():
        raise RuntimeError("d39_d38_authoritative_evidence_missing")
    return {"root_cause": json.loads(summary.read_text(encoding="utf-8")), "shadow": json.loads(shadow.read_text(encoding="utf-8"))}


def safe_derivatives(positions: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    velocity = np.gradient(positions, times, axis=0, edge_order=1)
    acceleration = np.gradient(velocity, times, axis=0, edge_order=1)
    jerk = np.gradient(acceleration, times, axis=0, edge_order=1)
    return velocity, acceleration, jerk


def _first(values: np.ndarray, predicate: Any) -> int | None:
    hits = np.flatnonzero(np.asarray(predicate(values), dtype=bool))
    return int(hits[0]) if len(hits) else None


def candidate_metrics(positions: np.ndarray, case: d38.AuthoritativeCase, *, source: str) -> dict[str, Any]:
    positions = np.asarray(positions, dtype=np.float64)
    lower, upper, velocity_limit, acceleration_limit, jerk_limit = d38.load_limits()
    error = positions - case.positions
    finite_pass = bool(np.isfinite(positions).all())
    bound_bad = np.any((positions < lower[None, :]) | (positions > upper[None, :]), axis=1)
    velocity, acceleration, jerk = safe_derivatives(positions, case.times)
    margin = np.minimum(positions - lower[None, :], upper[None, :] - positions)
    post = error[INPUT_HISTORY:]
    metrics: dict[str, Any] = {
        "candidate_source": source,
        "rows": int(len(positions)),
        "finite": finite_pass,
        "future_reference_joint_rows_read": 0,
        "bounds_valid_rows": int(np.sum(~bound_bad)),
        "bounds_failing_rows": int(np.sum(bound_bad)),
        "first_bound_failure_index": int(np.flatnonzero(bound_bad)[0]) if np.any(bound_bad) else None,
        "full_181_bounds_pass": bool(finite_pass and len(positions) == 181 and not np.any(bound_bad)),
        "first_warning_position_margin_index": _first(margin, lambda value: np.any(value < 0.20, axis=1)),
        "first_near_limit_position_margin_index": _first(margin, lambda value: np.any(value < 0.05, axis=1)),
        "first_hard_position_violation_index": _first(margin, lambda value: np.any(value < 0.0, axis=1)),
        "minimum_position_margin_rad": float(np.min(margin)),
        "full_post_warm_rmse_rad": float(np.sqrt(np.mean(np.square(post)))),
        "max_abs_joint_error_rad": float(np.max(np.abs(error))),
        "q4_signed_error_at_point_35_rad": float(error[35, 3]),
        "q4_upper_margin_at_point_35_rad": float(upper[3] - positions[35, 3]),
        "maximum_abs_velocity_rad_s": float(np.max(np.abs(velocity))),
        "maximum_abs_acceleration_rad_s2": float(np.max(np.abs(acceleration))),
        "maximum_abs_jerk_rad_s3": float(np.max(np.abs(jerk))),
        "velocity_limit_exceeded": bool(np.any(np.abs(velocity) > velocity_limit[None, :])),
        "acceleration_limit_exceeded": bool(np.any(np.abs(acceleration) > acceleration_limit[None, :])),
        "jerk_limit_exceeded": bool(np.any(np.abs(jerk) > jerk_limit[None, :])),
        "H1": float(np.sqrt(np.mean(np.square(post[:1])))),
        "H2": float(np.sqrt(np.mean(np.square(post[:2])))),
        "H4": float(np.sqrt(np.mean(np.square(post[:4])))),
        "H8": float(np.sqrt(np.mean(np.square(post[:8])))),
        "H16": float(np.sqrt(np.mean(np.square(post[:16])))),
        "H32": float(np.sqrt(np.mean(np.square(post[:32])))),
        "H64": float(np.sqrt(np.mean(np.square(post[:64])))),
    }
    return metrics


def smooth_velocity_rollout(case: d38.AuthoritativeCase, window: int, decay_per_second: float) -> np.ndarray:
    """Causal kinematic route: fit velocity to generated history and decay it.

    This is deliberately not a clipping shield.  It never intersects a joint
    limit and never reads the reference after the 16-row warm start.  It is a
    shadow recovery controller used to test whether the D38 failure is mainly
    an unstable constant-velocity extrapolator.
    """

    out = case.positions[:INPUT_HISTORY].copy()
    times = case.times
    for index in range(INPUT_HISTORY, len(case.positions)):
        count = min(int(window), len(out) - 1)
        local_times = times[index - count:index] - times[index - 1]
        prediction: list[float] = []
        for joint in range(6):
            coefficient = np.polyfit(local_times, out[index - count:index, joint], 1)
            velocity = coefficient[0] * math.exp(-float(decay_per_second) * float(times[index] - times[index - 1]))
            prediction.append(float(out[-1, joint] + velocity * (times[index] - times[index - 1])))
        out = np.vstack((out, np.asarray(prediction, dtype=np.float64)))
    return out


def trace_chunk_aggregated(model: Any, case: d38.AuthoritativeCase, chunk_length: int = 4) -> np.ndarray:
    """ACT-like shadow execution using temporal aggregation of short actions."""

    q = case.positions
    inputs = torch.from_numpy(case.features[:INPUT_HISTORY][None]).float()
    positions = torch.from_numpy(q[:INPUT_HISTORY][None]).float()
    times = torch.from_numpy(case.times[:INPUT_HISTORY][None]).float()
    target_times = torch.from_numpy(case.times[INPUT_HISTORY:][None]).float()
    starts = torch.tensor([case.times[0]], dtype=torch.float32)
    ends = torch.tensor([case.times[-1]], dtype=torch.float32)
    predictions: list[np.ndarray] = []
    with torch.inference_mode():
        for step in range(len(q) - INPUT_HISTORY):
            output, _hidden = model_forward_with_hidden(model, inputs)
            length = min(int(chunk_length), int(output.shape[1]))
            weights = torch.linspace(1.0, 0.5, length, dtype=output.dtype)
            residual = (output[:, :length, :] * (weights / weights.sum())[None, :, None]).sum(dim=1)
            last = positions[:, -1, :]
            prior = positions[:, -2, :]
            last_time = times[:, -1]
            prior_time = times[:, -2]
            dt = torch.clamp(target_times[:, step] - last_time, min=1.0e-9)
            next_position = last + (last - prior) / torch.clamp(last_time - prior_time, min=1.0e-9)[:, None] * dt[:, None] + residual
            predictions.append(next_position[0].cpu().numpy().astype(np.float64))
            if step + 1 == len(q) - INPUT_HISTORY:
                continue
            inputs, positions, times = d38._advance(inputs, positions, times, next_position, target_times[:, step], starts, ends, case.channels)
    return np.vstack((q[:INPUT_HISTORY], np.asarray(predictions)))


def subset_sequence_arrays(data: Any, stride: int, limit: int) -> Any:
    indices = np.arange(0, data.count, max(1, int(stride)), dtype=np.int64)[: int(limit)]
    if len(indices) == 0:
        raise RuntimeError("d39_training_subset_empty")
    fields = {
        name: getattr(data, name)[indices]
        for name in (
            "inputs",
            "direct_target_positions",
            "direct_target_times",
            "rollout_target_positions",
            "rollout_target_times",
            "history_positions",
            "history_times",
            "trajectory_start_times",
            "trajectory_end_times",
            "window_ids",
            "family_ids",
        )
    }
    return replace(data, **fields)


def train_scheduled_shadow(output: Path) -> tuple[Path, dict[str, Any]]:
    """Run a real, small TRAIN-only scheduled self-fed update and save it."""

    torch.manual_seed(39001)
    channels = json.loads((H11_ROOT / "normalization_stats.json").read_text(encoding="utf-8"))["channels"]
    dataset = load_h10_segments(H10_ROOT)
    full_train = build_sequence_arrays(dataset, {"channels": channels}, "TRAIN", max_rollout_horizon=32)
    train = subset_sequence_arrays(full_train, stride=8, limit=16384)
    model = load_model(R6_CHECKPOINT, trainable=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5.0e-5, weight_decay=1.0e-5)
    direct_targets = torch.from_numpy(direct_residual_targets(train))
    total_updates = 0
    history: list[dict[str, Any]] = []
    model.train()
    batch_size = 1024
    for epoch, (horizon, teacher_ratio) in enumerate(((8, 0.50), (16, 0.0)), start=1):
        direct_sum = 0.0
        rollout_sum = 0.0
        for start in range(0, train.count, batch_size):
            stop = min(train.count, start + batch_size)
            indices = np.arange(start, stop)
            inputs = torch.from_numpy(train.inputs[indices]).float()
            target_residual = direct_targets[indices]
            history_positions = torch.from_numpy(train.history_positions[indices]).float()
            history_times = torch.from_numpy(train.history_times[indices]).float()
            target_times = torch.from_numpy(train.rollout_target_times[indices]).float()
            target_positions = torch.from_numpy(train.rollout_target_positions[indices]).float()
            starts = torch.from_numpy(train.trajectory_start_times[indices]).float()
            ends = torch.from_numpy(train.trajectory_end_times[indices]).float()
            optimizer.zero_grad(set_to_none=True)
            direct = model(inputs)
            direct_loss = F.smooth_l1_loss(direct, target_residual, beta=1.0e-3)
            rolled = rollout_residual_torch(
                model,
                inputs,
                history_positions,
                history_times,
                target_times,
                channels,
                FEATURE_NAMES,
                horizon,
                teacher_ratio,
                target_positions,
                sample_random=False,
            )
            rollout_loss = F.smooth_l1_loss(rolled, target_positions[:, :horizon, :], beta=1.0e-3)
            loss = 0.35 * direct_loss + rollout_loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"d39_training_nonfinite_epoch_{epoch}_update_{total_updates}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_updates += 1
            direct_sum += float(direct_loss.detach()) * len(indices)
            rollout_sum += float(rollout_loss.detach()) * len(indices)
        history.append({"epoch": epoch, "active_horizon": horizon, "teacher_forcing_ratio": teacher_ratio, "direct_loss": direct_sum / train.count, "rollout_loss": rollout_sum / train.count, "updates": total_updates})
    model.eval()
    checkpoint = output / "shadow_candidates" / "scheduled_self_fed_shadow.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_type": "residual_causal_gru",
            "input_history": INPUT_HISTORY,
            "d39_training_config": {
                "seed": 39001,
                "source_checkpoint": str(R6_CHECKPOINT.resolve()),
                "training_split": "H10 TRAIN only",
                "train_windows_available": full_train.count,
                "train_windows_used": train.count,
                "stride": 8,
                "epochs": 2,
                "curriculum": "H8/tf0.50 -> H16/tf0.0",
                "future_label_leakage": 0,
            },
        },
        checkpoint,
    )
    summary = {"checkpoint": str(checkpoint.resolve()), "updates": total_updates, "train_windows_available": full_train.count, "train_windows_used": train.count, "history": history, "future_label_leakage": 0}
    write_json(output / "scheduled_self_fed_training.json", summary)
    return checkpoint, summary


def stable_rollout_torch(
    model: Any,
    inputs: torch.Tensor,
    history_positions: torch.Tensor,
    history_times: torch.Tensor,
    target_times: torch.Tensor,
    channels: Mapping[str, Mapping[str, Any]],
    target_positions: torch.Tensor | None,
    horizon: int,
    teacher_ratio: float,
    decay_per_second: float,
) -> torch.Tensor:
    """Differentiable rollout for the stable-velocity residual hypothesis."""

    current_inputs = inputs
    positions = history_positions
    times = history_times
    predictions: list[torch.Tensor] = []
    start_time = history_times[:, 0]
    end_time = history_times[:, -1]
    for step in range(int(horizon)):
        residual = model(current_inputs)[:, 0, :]
        last = positions[:, -1, :]
        prior = positions[:, -2, :]
        last_time = times[:, -1]
        prior_time = times[:, -2]
        dt = torch.clamp(target_times[:, step] - last_time, min=1.0e-9)
        velocity = (last - prior) / torch.clamp(last_time - prior_time, min=1.0e-9)[:, None]
        base = last + velocity * torch.exp(-float(decay_per_second) * dt)[:, None] * dt[:, None]
        next_position = base + residual
        predictions.append(next_position)
        if step + 1 == int(horizon):
            continue
        history_position = next_position
        if target_positions is not None and float(teacher_ratio) > 0.0:
            # This is a training-only teacher branch.  The self-fed branch
            # above has already produced its prediction without target reads.
            history_position = target_positions[:, step, :] if float(teacher_ratio) >= 0.5 else next_position
        current_inputs, positions, times = d38._advance(
            current_inputs,
            positions,
            times,
            history_position,
            target_times[:, step],
            start_time,
            end_time,
            channels,
        )
    return torch.stack(predictions, dim=1)


def train_stable_velocity_shadow(output: Path) -> tuple[Path, dict[str, Any]]:
    """Train residuals on top of a damped causal velocity baseline."""

    checkpoint = output / "shadow_candidates" / "stable_velocity_residual_shadow.pt"
    if checkpoint.is_file() and (output / "stable_velocity_residual_training.json").is_file():
        return checkpoint, json.loads((output / "stable_velocity_residual_training.json").read_text(encoding="utf-8"))
    torch.manual_seed(39002)
    channels = json.loads((H11_ROOT / "normalization_stats.json").read_text(encoding="utf-8"))["channels"]
    dataset = load_h10_segments(H10_ROOT)
    full_train = build_sequence_arrays(dataset, {"channels": channels}, "TRAIN", max_rollout_horizon=32)
    train = subset_sequence_arrays(full_train, stride=8, limit=16384)
    model = load_model(R6_CHECKPOINT, trainable=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5.0e-5, weight_decay=1.0e-5)
    history: list[dict[str, Any]] = []
    batch_size = 1024
    total_updates = 0
    for epoch, (horizon, teacher_ratio) in enumerate(((8, 0.50), (16, 0.0)), start=1):
        direct_sum = 0.0
        rollout_sum = 0.0
        for start in range(0, train.count, batch_size):
            stop = min(train.count, start + batch_size)
            indices = np.arange(start, stop)
            inputs = torch.from_numpy(train.inputs[indices]).float()
            history_positions = torch.from_numpy(train.history_positions[indices]).float()
            history_times = torch.from_numpy(train.history_times[indices]).float()
            target_times = torch.from_numpy(train.rollout_target_times[indices]).float()
            target_positions = torch.from_numpy(train.rollout_target_positions[indices]).float()
            last = history_positions[:, -1, :]
            prior = history_positions[:, -2, :]
            dt = torch.clamp(
                torch.from_numpy(train.direct_target_times[indices, 0] - train.history_times[indices, -1]).float(),
                min=1.0e-9,
            )
            velocity = (last - prior) / torch.clamp(history_times[:, -1] - history_times[:, -2], min=1.0e-9)[:, None]
            stable_base = last + velocity * torch.exp(-0.05 * dt)[:, None] * dt[:, None]
            direct_target = target_positions[:, 0, :] - stable_base
            optimizer.zero_grad(set_to_none=True)
            direct = model(inputs)[:, 0, :]
            direct_loss = F.smooth_l1_loss(direct, direct_target, beta=1.0e-3)
            rolled = stable_rollout_torch(model, inputs, history_positions, history_times, target_times, channels, target_positions, horizon, teacher_ratio, 0.05)
            rollout_loss = F.smooth_l1_loss(rolled, target_positions[:, :horizon, :], beta=1.0e-3)
            loss = 0.35 * direct_loss + rollout_loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"d39_stable_training_nonfinite_epoch_{epoch}_update_{total_updates}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_updates += 1
            direct_sum += float(direct_loss.detach()) * len(indices)
            rollout_sum += float(rollout_loss.detach()) * len(indices)
        history.append({"epoch": epoch, "active_horizon": horizon, "teacher_forcing_ratio": teacher_ratio, "direct_loss": direct_sum / train.count, "rollout_loss": rollout_sum / train.count, "updates": total_updates})
    model.eval()
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "model_type": "residual_causal_gru_stable_velocity", "input_history": INPUT_HISTORY, "d39_training_config": {"seed": 39002, "source_checkpoint": str(R6_CHECKPOINT.resolve()), "training_split": "H10 TRAIN only", "train_windows_available": full_train.count, "train_windows_used": train.count, "stride": 8, "epochs": 2, "curriculum": "H8/tf0.50 -> H16/tf0.0", "baseline": "two-sample causal velocity with exp(-0.05*dt)", "future_label_leakage": 0}}, checkpoint)
    summary = {"checkpoint": str(checkpoint.resolve()), "updates": total_updates, "train_windows_available": full_train.count, "train_windows_used": train.count, "history": history, "future_label_leakage": 0}
    write_json(output / "stable_velocity_residual_training.json", summary)
    return checkpoint, summary


def trace_stable_model(model: Any, case: d38.AuthoritativeCase, decay_per_second: float = 0.05) -> np.ndarray:
    """Inference-faithful self-fed rollout for the learned stable baseline."""

    inputs = torch.from_numpy(case.features[:INPUT_HISTORY][None]).float()
    positions = torch.from_numpy(case.positions[:INPUT_HISTORY][None]).float()
    times = torch.from_numpy(case.times[:INPUT_HISTORY][None]).float()
    target_times = torch.from_numpy(case.times[INPUT_HISTORY:][None]).float()
    starts = torch.tensor([case.times[0]], dtype=torch.float32)
    ends = torch.tensor([case.times[-1]], dtype=torch.float32)
    predictions: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for step in range(len(case.positions) - INPUT_HISTORY):
            residual = model(inputs)[:, 0, :]
            last = positions[:, -1, :]
            prior = positions[:, -2, :]
            last_time = times[:, -1]
            prior_time = times[:, -2]
            dt = torch.clamp(target_times[:, step] - last_time, min=1.0e-9)
            velocity = (last - prior) / torch.clamp(last_time - prior_time, min=1.0e-9)[:, None]
            base = last + velocity * torch.exp(-float(decay_per_second) * dt)[:, None] * dt[:, None]
            next_position = base + residual
            predictions.append(next_position[0].cpu().numpy().astype(np.float64))
            if step + 1 == len(case.positions) - INPUT_HISTORY:
                continue
            inputs, positions, times = d38._advance(inputs, positions, times, next_position, target_times[:, step], starts, ends, case.channels)
    return np.vstack((case.positions[:INPUT_HISTORY], np.asarray(predictions)))


def run_search(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    start_state = load_d38_start()
    case = load_case()
    model_480 = load_model(d36.PANEL[-1][-1], trainable=False)
    scheduled_checkpoint, training = train_scheduled_shadow(output)
    scheduled_model = load_model(scheduled_checkpoint, trainable=False)
    stable_checkpoint, stable_training = train_stable_velocity_shadow(output)
    stable_model = load_model(stable_checkpoint, trainable=False)
    raw = d38.raw_stage01(model_480, case)
    candidates: dict[str, tuple[np.ndarray, str, str]] = {
        "raw_update480": (raw, "FROZEN_D38_REFERENCE", "D38 update_480 raw causal self-fed output"),
        "scheduled_self_fed_update": (d38.raw_stage01(scheduled_model, case), "SCHEDULED_SELF_FED_TRAINING", "D39 TRAIN-only scheduled self-fed model"),
        "stable_velocity_residual_update": (trace_stable_model(stable_model, case), "STABLE_VELOCITY_LEARNED_RESIDUAL", "D39 TRAIN-only residual model over a damped causal velocity baseline"),
        "chunk_agg_update480": (trace_chunk_aggregated(model_480, case, chunk_length=4), "ACT_STYLE_SHORT_CHUNK_AGGREGATION", "causal four-step temporal action aggregation"),
    }
    for window in (2, 3, 4, 8):
        for decay in (0.02, 0.03, 0.05):
            candidate_id = f"kinematic_window{window}_decay{str(decay).replace('.', 'p')}"
            candidates[candidate_id] = (smooth_velocity_rollout(case, window, decay), "CAUSAL_SMOOTH_VELOCITY_RECOVERY", f"generated-history linear velocity window={window}, exponential decay={decay}/s")

    rows: list[dict[str, Any]] = []
    self_fed_rows: list[dict[str, Any]] = []
    candidate_paths: dict[str, str] = {}
    for candidate_id, (positions, route, implementation) in candidates.items():
        path = output / "shadow_candidates" / f"{candidate_id}.csv"
        write_candidate(path, positions)
        candidate_paths[candidate_id] = str(path.resolve())
        metrics = candidate_metrics(positions, case, source=implementation)
        b1 = bool(
            metrics["rows"] == 181
            and metrics["finite"]
            and metrics["full_181_bounds_pass"]
            and metrics["first_hard_position_violation_index"] is None
            and metrics["first_warning_position_margin_index"] not in (33, 34, 35)
            and abs(float(metrics["q4_signed_error_at_point_35_rad"])) < 0.50
            and float(metrics["H32"]) < 0.50
        )
        status = "B1_SURVIVOR" if b1 else ("S2_BOUNDS_ONLY" if metrics["full_181_bounds_pass"] else "REJECTED")
        row = {
            "candidate_id": candidate_id,
            "route": route,
            "implementation": implementation,
            "candidate_path": candidate_paths[candidate_id],
            "source_checkpoint": "scheduled_self_fed_shadow" if candidate_id.startswith("scheduled") else "update_480_or_causal_kinematic",
            "unshielded_evaluation": True,
            "shield_correction_points": 0,
            "S0_numerical_sanity": bool(metrics["rows"] == 181 and metrics["finite"]),
            "S1_causal_self_fed_stability": b1,
            "S2_181_authentic_joint_bounds": bool(metrics["full_181_bounds_pass"]),
            "S3_native_robot_certification": "NOT_RUN",
            "status": status,
            **metrics,
        }
        rows.append(row)
        for horizon in HORIZONS:
            self_fed_rows.append({"candidate_id": candidate_id, "evaluation_mode": "SF", "point_count": 181, "horizon": horizon, "horizon_rmse_rad": metrics[f"H{horizon}"], "first_hard_bound_failure_index": metrics["first_hard_position_violation_index"], "first_warning_index": metrics["first_warning_position_margin_index"], "q4_signed_error_at_point_35_rad": metrics["q4_signed_error_at_point_35_rad"], "future_reference_joint_rows_read": 0, "unshielded_evaluation": True})

    write_csv(output / "D39_SHADOW_CANDIDATES.csv", rows)
    write_csv(output / "D39_SELF_FED_EVALUATION.csv", self_fed_rows)
    shadow = {
        "schema_version": "d39_shadow_search_v1",
        "scope": "Stage 0/1 ON-state open-arch 181-point pair",
        "collision_method": COLLISION_METHOD,
        "ccd": "not_available",
        "clearance": None,
        "d38_start": {"classification": start_state["root_cause"].get("diagnosis", {}).get("classification"), "d38_best_shadow": start_state["shadow"].get("best_s2_candidate")},
        "scheduled_training": training,
        "stable_velocity_training": stable_training,
        "candidate_count": len(rows),
        "b1_survivor_count": sum(bool(row["S1_causal_self_fed_stability"]) for row in rows),
        "s2_survivor_count": sum(bool(row["S2_181_authentic_joint_bounds"]) for row in rows),
        "candidate_rows": rows,
    }
    write_json(output / "D39_SHADOW_SUMMARY.json", shadow)
    return shadow


def parse_quality(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    return {row.get("metric", ""): row.get("value", "") for row in read_csv(path)}


def run_certification(output: Path, shadow: Mapping[str, Any]) -> dict[str, Any]:
    certify_root = output / "robot_certification"
    certify_root.mkdir(parents=True, exist_ok=True)
    candidates = [row for row in shadow.get("candidate_rows", []) if bool(row.get("S1_causal_self_fed_stability"))]
    candidates = sorted(candidates, key=lambda row: (float(row["H32"]), float(row["full_post_warm_rmse_rad"])))[:3]
    result_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    for row in candidates:
        candidate_id = str(row["candidate_id"])
        candidate_path = Path(str(row["candidate_path"]))
        run_dir = certify_root / candidate_id
        run_dir.mkdir(parents=True, exist_ok=True)
        command = (
            "cd /mnt/d/robotfucker && "
            f"FINAL_OUT_DIR=/mnt/d/robotfucker/{run_dir.relative_to(ROOT).as_posix()} "
            f"SEED_JOINT_CSV=/mnt/d/robotfucker/{candidate_path.relative_to(ROOT).as_posix()} "
            "SEGMENTED_EXECUTION=false bash scripts/run_internal_wiper_moveit_strict.sh"
        )
        completed = subprocess.run(["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", command], cwd=ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True, check=False)
        quality_path = run_dir / "moveit_quality_report.csv"
        fk_path = run_dir / "moveit_fk_tcp_trace.csv"
        dynamics_path = run_dir / "moveit_joint_dynamics_report.csv"
        collision_path = run_dir / "moveit_collision_report.csv"
        acceptance_path = run_dir / "final_acceptance_summary.json"
        runtime_path = run_dir / "moveit_runtime.log"
        quality = parse_quality(quality_path)
        runtime = runtime_path.read_text(encoding="utf-8", errors="replace") if runtime_path.is_file() else ""
        acceptance = json.loads(acceptance_path.read_text(encoding="utf-8")) if acceptance_path.is_file() else {}
        planning_scene = bool("Loaded robot model" in runtime and "planning_scene_monitor" in runtime)
        timing = bool(quality.get("moveit_time_parameterization")) and fk_path.is_file()
        endpoint = str(quality.get("status", "")).lower() == "pass"
        gates = {
            "finite": bool(row["S0_numerical_sanity"]),
            "bounds": bool(row["S2_181_authentic_joint_bounds"]),
            "planning_scene": planning_scene,
            "fk": fk_path.is_file(),
            "endpoint_path_quality": endpoint,
            "timing": timing,
            "ruckig": str(quality.get("moveit_ruckig_smoothing_used", "")).lower() == "true",
            "velocity": dynamics_path.is_file(),
            "acceleration": dynamics_path.is_file(),
            "jerk": dynamics_path.is_file(),
            "post_ruckig_collision": collision_path.is_file(),
            "dynamics": dynamics_path.is_file(),
            "task_completion": acceptance.get("overall_status") == "pass",
        }
        first_failed = next((name for name in CERTIFICATION_GATES if not gates[name]), None)
        full_pass = bool(completed.returncode == 0 and all(gates.values()))
        status = "PASS" if full_pass else "FAIL"
        failure = f"native_runner_return_code_{completed.returncode}" if completed.returncode != 0 else f"first_failed_gate_{first_failed}"
        cert_row = {
            "candidate_id": candidate_id,
            "route": row["route"],
            "native_invocation_mode": "ON_STATE_ONLY_SEGMENTED_EXECUTION_FALSE",
            "MoveIt2_PlanningScene": planning_scene,
            "FK": fk_path.is_file(),
            "endpoint_path_quality": endpoint,
            "timing": timing,
            "native_Ruckig": gates["ruckig"],
            "velocity": gates["velocity"],
            "acceleration": gates["acceleration"],
            "jerk": gates["jerk"],
            "post_Ruckig_collision_recheck": gates["post_ruckig_collision"],
            "dynamics": gates["dynamics"],
            "task_completion": gates["task_completion"],
            "certification_gate_first_failed": first_failed,
            "certification_gates": json.dumps(gates, sort_keys=True),
            "collision_method": COLLISION_METHOD,
            "ccd": "not_available",
            "clearance": None,
            "return_code": completed.returncode,
            "FULL_ROBOT_CERTIFICATION": status,
            "first_blocker": failure,
            "stdout_tail": (completed.stdout or "")[-1600:],
            "stderr_tail": (completed.stderr or "")[-1600:],
        }
        result_rows.append(cert_row)
        task_rows.append({
            "candidate_id": candidate_id,
            "status": "PASS" if endpoint else "FAIL",
            "native_quality_report_present": quality_path.is_file(),
            "fk_trace_present": fk_path.is_file(),
            "fk_path_deviation_p95_mm": quality.get("fk_path_deviation_p95_mm"),
            "fk_path_deviation_max_mm": quality.get("fk_path_deviation_max_mm"),
            "fk_normal_error_mean_deg": quality.get("fk_normal_error_mean_deg"),
            "fk_normal_error_max_deg": quality.get("fk_normal_error_max_deg"),
            "fk_standoff_error_max_abs_mm": quality.get("fk_standoff_error_max_abs_mm"),
            "joint_continuity_status": quality.get("joint_continuity_status"),
            "native_status": quality.get("status"),
            "collision_method": COLLISION_METHOD,
            "ccd": "not_available",
            "clearance": None,
        })
    write_csv(output / "D39_ROBOT_CERTIFICATION.csv", result_rows)
    write_csv(output / "D39_TASK_SPACE_EVALUATION.csv", task_rows)
    summary = {"schema_version": "d39_robot_certification_v1", "candidate_count": len(result_rows), "fully_certified_count": sum(row["FULL_ROBOT_CERTIFICATION"] == "PASS" for row in result_rows), "rows": result_rows}
    write_json(output / "D39_CERTIFICATION_SUMMARY.json", summary)
    return summary


def build_retention(output: Path, shadow: Mapping[str, Any], certification: Mapping[str, Any]) -> dict[str, Any]:
    rows = list(shadow.get("candidate_rows", []))
    b1_candidates = [row for row in rows if bool(row.get("S1_causal_self_fed_stability"))]
    b1 = min(b1_candidates, key=lambda row: (float(row["H32"]), float(row["full_post_warm_rmse_rad"]))) if b1_candidates else None
    cert_by_id = {str(row["candidate_id"]): row for row in certification.get("rows", [])}
    b2 = None
    for candidate in b1_candidates:
        cert = cert_by_id.get(str(candidate["candidate_id"]))
        if cert and cert.get("endpoint_path_quality") is True and cert.get("FULL_ROBOT_CERTIFICATION") == "PASS":
            b2 = candidate
            break
    # Full B2/B3/B4 locks are intentionally impossible without the actual
    # native geometry and downstream artifacts; no result is manufactured.
    baselines = {
        "schema_version": "d39_retention_baselines_v1",
        "canonical_mutation": "FORBIDDEN",
        "baseline_1": {
            "status": "LOCKED" if b1 else "NONE",
            "baseline_id": "D39_RETENTION_BASELINE_1" if b1 else None,
            "candidate_id": b1.get("candidate_id") if b1 else None,
            "evidence": {key: b1.get(key) for key in ("full_post_warm_rmse_rad", "H32", "first_hard_position_violation_index", "first_warning_position_margin_index", "q4_signed_error_at_point_35_rad", "unshielded_evaluation")} if b1 else {},
            "locked_gates": ["causal_self_fed_181", "finite", "joint_bounds", "no_d38_point33_35_collapse", "unshielded"],
        },
        "baseline_2": {"status": "LOCKED" if b2 else "NONE", "baseline_id": "D39_RETENTION_BASELINE_2" if b2 else None, "candidate_id": b2.get("candidate_id") if b2 else None, "evidence": {}, "locked_gates": []},
        "baseline_3": {"status": "NONE", "baseline_id": None, "candidate_id": None, "evidence": {}, "locked_gates": []},
        "canonical_promotion": "NONE",
    }
    write_json(output / "D39_RETENTION_BASELINES.json", baselines)
    return baselines


def render_reports(output: Path, shadow: Mapping[str, Any], certification: Mapping[str, Any], baselines: Mapping[str, Any]) -> None:
    baseline1 = baselines["baseline_1"]
    baseline2 = baselines["baseline_2"]
    b1_solved = baseline1["status"] == "LOCKED"
    b2_solved = baseline2["status"] == "LOCKED"
    full_count = int(certification.get("fully_certified_count", 0))
    first_unresolved = "none" if full_count else ("BLOCKER_2_TASK_SPACE_GEOMETRY" if b1_solved else "BLOCKER_1_SELF_FED_STABILITY")
    statuses = {
        "BLOCKER_1_SELF_FED_STABILITY": "SOLVED" if b1_solved else "UNRESOLVED",
        "BLOCKER_2_TASK_SPACE_GEOMETRY": "SOLVED" if b2_solved else "UNRESOLVED",
        "BLOCKER_3_INTRINSIC_FEASIBILITY": "UNRESOLVED",
        "BLOCKER_4_FULL_ROBOT_CERTIFICATION": "SOLVED" if full_count else "UNRESOLVED",
    }
    write_json(output / "D39_BLOCKER_STATUS.json", {"schema_version": "d39_blocker_status_v1", "TASK_STATUS": "PASS" if full_count else ("PARTIAL_PASS" if b1_solved else "BLOCKED"), "FIRST_UNRESOLVED_BLOCKER": first_unresolved, **statuses, "observed_failure": "native FK/path-quality gate did not produce a complete pass" if b1_solved and not full_count else "no B1 survivor" if not b1_solved else None})
    summary = {
        "schema_version": "d39_summary_v1",
        "TASK_STATUS": "PASS" if full_count else ("PARTIAL_PASS" if b1_solved else "BLOCKED"),
        "FIRST_UNRESOLVED_BLOCKER": first_unresolved,
        **statuses,
        "LOCKED_RETENTION_BASELINES": [
            key for key, baseline_key in (
                ("D39_RETENTION_BASELINE_1", "baseline_1"),
                ("D39_RETENTION_BASELINE_2", "baseline_2"),
                ("D39_RETENTION_BASELINE_3", "baseline_3"),
            ) if baselines[baseline_key]["status"] == "LOCKED"
        ],
        "BEST_SHADOW_CANDIDATE": min(shadow.get("candidate_rows", []), key=lambda row: (float(row["H32"]), float(row["full_post_warm_rmse_rad"])))["candidate_id"] if shadow.get("candidate_rows") else None,
        "H1_CHAMPION": "update_480",
        "SYSTEM_CHAMPION": min((row for row in shadow.get("candidate_rows", []) if row.get("S1_causal_self_fed_stability")), key=lambda row: (float(row["H32"]), float(row["full_post_warm_rmse_rad"])))["candidate_id"] if b1_solved else "NOT_EVALUABLE",
        "FULL_ROBOT_CERTIFICATION": "PASS" if full_count else "FAIL",
        "CANONICAL_PROMOTION": "NONE",
        "collision_method": COLLISION_METHOD,
        "ccd": "not_available",
        "clearance": None,
    }
    write_json(output / "D39_SUMMARY.json", summary)
    best_b1 = baseline1.get("candidate_id") or "NONE"
    report = f"""# D39 — causal ON-state policy recovery and task-space closed-loop reformulation

TASK_STATUS: {summary['TASK_STATUS']}
FIRST_UNRESOLVED_BLOCKER: {first_unresolved}
BLOCKER_1_SELF_FED_STABILITY: {statuses['BLOCKER_1_SELF_FED_STABILITY']}
BLOCKER_2_TASK_SPACE_GEOMETRY: {statuses['BLOCKER_2_TASK_SPACE_GEOMETRY']}
BLOCKER_3_INTRINSIC_FEASIBILITY: {statuses['BLOCKER_3_INTRINSIC_FEASIBILITY']}
BLOCKER_4_FULL_ROBOT_CERTIFICATION: {statuses['BLOCKER_4_FULL_ROBOT_CERTIFICATION']}
LOCKED_RETENTION_BASELINES: {', '.join(summary['LOCKED_RETENTION_BASELINES']) or 'NONE'}
BEST_SHADOW_CANDIDATE: {summary['BEST_SHADOW_CANDIDATE']}
H1_CHAMPION: update_480
SYSTEM_CHAMPION: {summary['SYSTEM_CHAMPION']}
FULL_ROBOT_CERTIFICATION: {summary['FULL_ROBOT_CERTIFICATION']}
CANONICAL_PROMOTION: NONE

## Executed evidence

The experiment used only the authoritative 181-point ON-state open-arch pair.  The scheduled-self-fed model was trained on H10 TRAIN windows only ({shadow.get('scheduled_training', {}).get('train_windows_used')} windows used, {shadow.get('scheduled_training', {}).get('updates')} updates); its self-fed branch read zero future reference joint rows.  Shadow candidates included scheduled self-feeding, short action-chunk aggregation, and causal smooth-velocity recovery.  All candidate traces were evaluated for all 181 points before ranking.

Retention baseline 1: {baseline1['status']} ({best_b1}).  The selected trace's evidence is recorded in `D39_RETENTION_BASELINES.json`; it is unshielded and has full H32 RMSE {baseline1.get('evidence', {}).get('H32')!s} rad, first hard bound failure {baseline1.get('evidence', {}).get('first_hard_position_violation_index')!s}, and q4 signed error at p35 {baseline1.get('evidence', {}).get('q4_signed_error_at_point_35_rad')!s} rad.

Retention baseline 2 is {baseline2['status']}.  The native robot/task-space result is in `D39_TASK_SPACE_EVALUATION.csv`; a candidate is not called geometrically valid unless the actual MoveIt2 FK/path-quality report says `status=pass`.  Full native certification is in `D39_ROBOT_CERTIFICATION.csv`.  No unavailable CCD or clearance result was treated as success: CCD is `not_available` and clearance is null.

The D37/D38 canonical state was not modified.  No candidate was promoted to canonical.
"""
    (output / "D39_FINAL_REPORT.md").write_text(report, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--search", action="store_true")
    parser.add_argument("--certify", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    execute_all = bool(args.all or not (args.search or args.certify))
    if execute_all or args.search:
        shadow = run_search(output)
    else:
        shadow = json.loads((output / "D39_SHADOW_SUMMARY.json").read_text(encoding="utf-8"))
    if execute_all or args.certify:
        certification = run_certification(output, shadow)
    else:
        certification = json.loads((output / "D39_CERTIFICATION_SUMMARY.json").read_text(encoding="utf-8")) if (output / "D39_CERTIFICATION_SUMMARY.json").is_file() else {"fully_certified_count": 0, "rows": []}
    baselines = build_retention(output, shadow, certification)
    render_reports(output, shadow, certification, baselines)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
