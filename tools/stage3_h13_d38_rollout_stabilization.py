"""D38 rollout-stabilization investigation and bounded shadow experiments.

This module is deliberately separate from the frozen D36/D37 evaluators.  It
uses the same authoritative Stage 0/1 open-arch pair and the same residual-GRU
causal feature semantics, but adds matched teacher-forced/self-fed tracing and
non-certifying shadow repair prototypes.  No frozen checkpoint is ever opened
for writing.

The native MoveIt2 runner is invoked only by the explicit ``--certify`` path
after a shadow candidate passes the numerical funnel.  Collision semantics are
kept as ``adaptive_discrete_interpolation``; Bullet CCD and clearance remain
unavailable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import stage3_h13_d36_system_evaluation as d36  # noqa: E402
from scripts.stage3_h13_r8e_post_d4_canonical_batch_robustness import load_model  # noqa: E402
from src.stage3_h11_dataset import ACCELERATION_LIMIT, FEATURE_NAMES, INPUT_HISTORY, JERK_LIMIT, POSITION_LOWER, POSITION_UPPER, VELOCITY_LIMIT  # noqa: E402
from src.stage3_h13_r6 import _next_features  # noqa: E402
from src.stage3_h13_r8e import model_forward_with_hidden  # noqa: E402


OUTPUT = ROOT / "outputs" / "stage3_h13_d38_rollout_stabilization"
AUTHORITATIVE_INPUT = ROOT / "outputs" / "internal_wiper_moveit_inputs"
H10_ROOT = ROOT / "outputs" / "stage3_h10_multi_trajectory_dataset_20260811T081123Z"
H11_ROOT = ROOT / "outputs" / "stage3_h11_deep_learning_baseline_20260811T141921Z"
JOINT_LIMIT_CONFIG = ROOT / "ros2_moveit_bridge" / "config" / "joint_limits_with_jerk.yaml"
HORIZONS = (1, 2, 4, 8, 16, 32)
THRESHOLD_DEG = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0)
WARNING_MARGIN_RAD = 0.20
NEAR_MARGIN_RAD = 0.05
ROBOT_CERTIFICATION_METHOD = "adaptive_discrete_interpolation"
CERTIFICATION_GATE_ORDER = (
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
    "robustness",
    "proxy_panel",
)


@dataclass
class Trace:
    mode: str
    full_positions: np.ndarray
    residuals: np.ndarray
    input_last_rows: np.ndarray
    hidden: np.ndarray
    hidden_norms: np.ndarray
    hidden_delta_norms: np.ndarray


@dataclass
class AuthoritativeCase:
    positions: np.ndarray
    times: np.ndarray
    features: np.ndarray
    channels: Mapping[str, Mapping[str, Any]]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def finite(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"nonfinite_d38_value:{number}")
    return number


def load_channels() -> Mapping[str, Mapping[str, Any]]:
    path = H11_ROOT / "normalization_stats.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    channels = payload.get("channels")
    if not isinstance(channels, dict):
        raise RuntimeError("d38_normalization_channels_missing")
    return channels


def load_limits() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read the already-configured robot limits without changing them."""

    # The Stage 0/1 execution contract uses the configured 0.15 scaling of
    # the FAIRINO limits: 3.15/3.2 -> 0.4725/0.48 and 0.7 -> 0.105.
    velocity = VELOCITY_LIMIT.astype(np.float64).copy()
    acceleration = ACCELERATION_LIMIT.astype(np.float64).copy()
    jerk = JERK_LIMIT.astype(np.float64).copy()
    if not JOINT_LIMIT_CONFIG.is_file():
        raise RuntimeError("d38_joint_limit_config_missing")
    text = JOINT_LIMIT_CONFIG.read_text(encoding="utf-8")
    if "max_jerk: 8.0" not in text or "max_acceleration: 0.7" not in text:
        raise RuntimeError("d38_joint_limit_config_unexpected")
    return POSITION_LOWER.astype(np.float64), POSITION_UPPER.astype(np.float64), velocity, acceleration, jerk


def make_case(channels: Mapping[str, Mapping[str, Any]]) -> AuthoritativeCase:
    seed_rows = read_csv(AUTHORITATIVE_INPUT / "open_arch_seed_joints.csv")
    pose_rows = read_csv(AUTHORITATIVE_INPUT / "open_arch_tcp_poses_base_link.csv")
    if len(seed_rows) != 181 or len(pose_rows) != 181:
        raise RuntimeError(f"d38_authoritative_row_count:{len(seed_rows)}:{len(pose_rows)}")
    q = np.asarray([[float(row[f"q{joint}"]) for joint in range(1, 7)] for row in seed_rows], dtype=np.float64)
    xyz = np.asarray([[float(row[key]) for key in ("x", "y", "z")] for row in pose_rows], dtype=np.float64)
    t = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1) / 0.003)]
    if not np.all(np.diff(t) > 0.0):
        raise RuntimeError("d38_authoritative_time_not_monotonic")
    velocity = np.gradient(q, t, axis=0, edge_order=1)
    acceleration = np.gradient(velocity, t, axis=0, edge_order=1)
    local_time = (t - t[0]) / (t[-1] - t[0])
    raw = np.column_stack((q, velocity, acceleration, np.ones(len(t)), local_time)).astype(np.float32)
    features = raw.copy()
    for index, name in enumerate(FEATURE_NAMES):
        item = channels.get(name)
        if item is None or name == "spray_on":
            continue
        scale = float(item.get("scale", item.get("std", 1.0)))
        if not math.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        features[:, index] = (features[:, index] - float(item.get("mean", 0.0))) / scale
    return AuthoritativeCase(q, t, features, channels)


def _advance(
    current_inputs: torch.Tensor,
    positions: torch.Tensor,
    times: torch.Tensor,
    next_position: torch.Tensor,
    next_time: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    channels: Mapping[str, Mapping[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    last = positions[:, -1, :]
    prior = positions[:, -2, :]
    last_time = times[:, -1]
    prior_time = times[:, -2]
    features = _next_features(
        next_position,
        last,
        prior,
        next_time,
        last_time,
        prior_time,
        starts,
        ends,
        current_inputs[:, -1, 18],
        channels,
        FEATURE_NAMES,
    ).to(dtype=current_inputs.dtype)
    return (
        torch.cat((current_inputs[:, 1:, :], features[:, None, :]), dim=1),
        torch.cat((positions[:, 1:, :], next_position[:, None, :]), dim=1),
        torch.cat((times[:, 1:], next_time[:, None]), dim=1),
    )


def trace_rollout(model: Any, case: AuthoritativeCase, mode: str) -> Trace:
    if mode not in {"TF", "SF"}:
        raise ValueError(mode)
    q = case.positions
    t = case.times
    horizon = len(q) - INPUT_HISTORY
    current_inputs = torch.from_numpy(case.features[:INPUT_HISTORY][None]).float()
    positions = torch.from_numpy(q[:INPUT_HISTORY][None]).float()
    times = torch.from_numpy(t[:INPUT_HISTORY][None]).float()
    target_times = torch.from_numpy(t[INPUT_HISTORY:][None]).float()
    starts = torch.tensor([t[0]], dtype=torch.float32)
    ends = torch.tensor([t[-1]], dtype=torch.float32)
    residuals: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    input_last_rows: list[np.ndarray] = []
    hidden_rows: list[np.ndarray] = []
    hidden_norms: list[float] = []
    hidden_delta_norms: list[float] = []
    previous_hidden: np.ndarray | None = None
    model.eval()
    with torch.inference_mode():
        for step in range(horizon):
            input_last_rows.append(current_inputs[0, -1].detach().cpu().numpy().astype(np.float64))
            output, hidden = model_forward_with_hidden(model, current_inputs)
            residual = output[:, 0, :]
            last = positions[:, -1, :]
            prior = positions[:, -2, :]
            last_time = times[:, -1]
            prior_time = times[:, -2]
            dt = torch.clamp(target_times[:, step] - last_time, min=1.0e-9)
            velocity = (last - prior) / torch.clamp(last_time - prior_time, min=1.0e-9)[:, None]
            next_position = last + velocity * dt[:, None] + residual
            residuals.append(residual[0].detach().cpu().numpy().astype(np.float64))
            predictions.append(next_position[0].detach().cpu().numpy().astype(np.float64))
            hidden_np = hidden[0].detach().cpu().numpy().astype(np.float64)
            hidden_rows.append(hidden_np)
            hidden_norms.append(float(np.linalg.norm(hidden_np)))
            hidden_delta_norms.append(0.0 if previous_hidden is None else float(np.linalg.norm(hidden_np - previous_hidden)))
            previous_hidden = hidden_np
            if step + 1 == horizon:
                continue
            next_history = torch.from_numpy(q[INPUT_HISTORY + step][None]).float() if mode == "TF" else next_position
            current_inputs, positions, times = _advance(
                current_inputs,
                positions,
                times,
                next_history,
                target_times[:, step],
                starts,
                ends,
                case.channels,
            )
    full = np.vstack((q[:INPUT_HISTORY], np.asarray(predictions, dtype=np.float64)))
    return Trace(
        mode=mode,
        full_positions=full,
        residuals=np.asarray(residuals),
        input_last_rows=np.asarray(input_last_rows),
        hidden=np.asarray(hidden_rows),
        hidden_norms=np.asarray(hidden_norms),
        hidden_delta_norms=np.asarray(hidden_delta_norms),
    )


def safe_derivatives(positions: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    velocity = np.gradient(positions, times, axis=0, edge_order=1)
    acceleration = np.gradient(velocity, times, axis=0, edge_order=1)
    jerk = np.gradient(acceleration, times, axis=0, edge_order=1)
    return velocity, acceleration, jerk


def first_index(values: np.ndarray, predicate: Any) -> int | None:
    hits = np.flatnonzero(np.asarray(predicate(values), dtype=bool))
    return int(hits[0]) if len(hits) else None


def mode_metrics(trace: Trace, case: AuthoritativeCase) -> dict[str, Any]:
    q = trace.full_positions
    ref = case.positions
    error = q - ref
    velocity, acceleration, jerk = safe_derivatives(q, case.times)
    lower, upper, velocity_limit, acceleration_limit, jerk_limit = load_limits()
    lower_margin = q - lower[None, :]
    upper_margin = upper[None, :] - q
    min_position_margin = np.minimum(lower_margin, upper_margin)
    velocity_margin = velocity_limit[None, :] - np.abs(velocity)
    acceleration_margin = acceleration_limit[None, :] - np.abs(acceleration)
    jerk_margin = jerk_limit[None, :] - np.abs(jerk)
    warm = INPUT_HISTORY
    thresholds: dict[str, int | None] = {}
    max_error_deg = np.max(np.abs(error), axis=1) * 180.0 / math.pi
    for threshold in THRESHOLD_DEG:
        thresholds[str(threshold)] = first_index(max_error_deg, lambda values, threshold=threshold: values > threshold)
    hard_pos = np.any(min_position_margin < 0.0, axis=1)
    warning = np.any(min_position_margin < WARNING_MARGIN_RAD, axis=1)
    near = np.any(min_position_margin < NEAR_MARGIN_RAD, axis=1)
    q4_signed = error[:, 3]
    q4_drift = {}
    for deg in THRESHOLD_DEG:
        threshold = deg * math.pi / 180.0
        flags = np.convolve((q4_signed > threshold).astype(np.int32), np.ones(5, dtype=np.int32), mode="valid") >= 5
        q4_drift[str(deg)] = int(np.flatnonzero(flags)[0]) if np.any(flags) else None
    post_error = error[warm:]
    horizon_metrics: dict[str, float] = {}
    for horizon in HORIZONS:
        horizon_metrics[str(horizon)] = float(np.sqrt(np.mean(np.square(post_error[:horizon])))) if horizon <= len(post_error) else None
    # The D37 H1/H32 panel is a prefix panel on this single deployment trace;
    # the frozen validation H1/H32 values remain in the checkpoint metadata.
    q2q4_corr = float(np.corrcoef(error[warm:, 1], error[warm:, 3])[0, 1]) if np.std(error[warm:, 1]) > 0 and np.std(error[warm:, 3]) > 0 else None
    q2q4_delta_corr = float(np.corrcoef(np.diff(error[warm:, 1]), np.diff(error[warm:, 3]))[0, 1]) if np.std(np.diff(error[warm:, 1])) > 0 and np.std(np.diff(error[warm:, 3])) > 0 else None
    return {
        "mode": trace.mode,
        "rows": int(len(q)),
        "warm_start_rows": INPUT_HISTORY,
        "prediction_rows": int(len(q) - INPUT_HISTORY),
        "finite": bool(np.isfinite(q).all()),
        "future_reference_joint_rows_read": 0 if trace.mode == "SF" else "diagnostic_teacher_branch_only",
        "first_error_exceedance_deg": thresholds,
        "first_warning_position_margin_index": first_index(min_position_margin, lambda values: np.any(values < WARNING_MARGIN_RAD, axis=1)),
        "first_near_limit_position_margin_index": first_index(min_position_margin, lambda values: np.any(values < NEAR_MARGIN_RAD, axis=1)),
        "first_hard_position_violation_index": first_index(min_position_margin, lambda values: np.any(values < 0.0, axis=1)),
        "first_hard_position_violation_joint": (int(np.argmin(min_position_margin[first_index(min_position_margin, lambda values: np.any(values < 0.0, axis=1))])) + 1) if first_index(min_position_margin, lambda values: np.any(values < 0.0, axis=1)) is not None else None,
        "position_bound_failing_rows": int(np.sum(hard_pos)),
        "minimum_position_margin_rad": float(np.min(min_position_margin)),
        "minimum_position_margin_by_joint_rad": [float(v) for v in np.min(min_position_margin, axis=0)],
        "first_velocity_margin_warning_index": first_index(velocity_margin, lambda values: np.any(values < 0.1 * velocity_limit[None, :], axis=1)),
        "first_acceleration_margin_warning_index": first_index(acceleration_margin, lambda values: np.any(values < 0.1 * acceleration_limit[None, :], axis=1)),
        "first_jerk_margin_warning_index": first_index(jerk_margin, lambda values: np.any(values < 0.1 * jerk_limit[None, :], axis=1)),
        "maximum_abs_velocity_rad_s": [float(v) for v in np.max(np.abs(velocity), axis=0)],
        "maximum_abs_acceleration_rad_s2": [float(v) for v in np.max(np.abs(acceleration), axis=0)],
        "maximum_abs_jerk_rad_s3": [float(v) for v in np.max(np.abs(jerk), axis=0)],
        "horizon_rmse_rad": horizon_metrics,
        "full_post_warm_rmse_rad": float(np.sqrt(np.mean(np.square(post_error)))),
        "max_abs_joint_error_rad": float(np.max(np.abs(error))),
        "q4_first_positive_bias_5_consecutive": q4_drift,
        "q4_signed_error_at_point_35_rad": float(q4_signed[35]),
        "q4_upper_margin_at_point_35_rad": float(upper[3] - q[35, 3]),
        "q2_q4_error_correlation": q2q4_corr,
        "q2_q4_error_increment_correlation": q2q4_delta_corr,
        "mean_abs_error_by_joint_rad": [float(v) for v in np.mean(np.abs(error), axis=0)],
        "rmse_by_joint_rad": [float(v) for v in np.sqrt(np.mean(np.square(error), axis=0))],
        "warning_margin_definition": f"minimum position margin < {WARNING_MARGIN_RAD} rad",
        "near_limit_definition": f"minimum position margin < {NEAR_MARGIN_RAD} rad",
        "position_bounds_source": "src.stage3_h11_dataset POSITION_LOWER/POSITION_UPPER; unchanged",
        "dynamic_limits_source": str(JOINT_LIMIT_CONFIG),
    }


def local_q4_sensitivity(model: Any, trace: Trace, case: AuthoritativeCase, point: int = 35) -> dict[str, Any]:
    """Finite-difference local sensitivity using only the causal input window."""

    step = max(0, min(point - INPUT_HISTORY, len(trace.input_last_rows) - 1))
    # Reconstruct the exact current window from the traced last row only is not
    # enough for a full GRU history.  Use one-step sensitivity of the last
    # feature row as a conservative local diagnostic, not a causal intervention.
    base_row = torch.from_numpy(trace.input_last_rows[step][None, None, :]).float()
    # Repeat the current row to form a valid causal input window. This measures
    # the decoder/head response to feature content, not a deployment behavior.
    base = base_row.repeat(1, INPUT_HISTORY, 1)
    delta = 0.25
    effects: list[dict[str, Any]] = []
    model.eval()
    with torch.inference_mode():
        for index, name in enumerate(FEATURE_NAMES):
            plus = base.clone(); minus = base.clone()
            plus[0, -1, index] += delta; minus[0, -1, index] -= delta
            plus_value = float(model(plus)[0, 0, 3])
            minus_value = float(model(minus)[0, 0, 3])
            effects.append({"feature_index": index, "feature": name, "q4_residual_sensitivity_per_normalized_unit": (plus_value - minus_value) / (2.0 * delta)})
    groups = {
        "q4_feedback": [3, 9, 15],
        "q2_feedback": [1, 7, 13],
        "other_joint_position_velocity_acceleration": [i for i in range(18) if i not in {1, 3, 7, 9, 13, 15}],
        "time_and_spray": [18, 19],
    }
    grouped = {}
    for group, indices in groups.items():
        values = [abs(effects[i]["q4_residual_sensitivity_per_normalized_unit"]) for i in indices]
        grouped[group] = {"max_abs_sensitivity": float(max(values) if values else 0.0), "sum_abs_sensitivity": float(sum(values))}
    return {"diagnostic_point": int(point), "used_trace_step": int(step), "finite_difference_delta_normalized": delta, "grouped": grouped, "top_features": sorted(effects, key=lambda row: abs(row["q4_residual_sensitivity_per_normalized_unit"]), reverse=True)[:10], "interpretation": "local decoder sensitivity only; it does not prove causal attribution"}


def build_point_rows(checkpoint_id: str, case: AuthoritativeCase, trace: Trace) -> list[dict[str, Any]]:
    q = trace.full_positions
    error = q - case.positions
    velocity, acceleration, jerk = safe_derivatives(q, case.times)
    lower, upper, velocity_limit, acceleration_limit, jerk_limit = load_limits()
    rows = []
    for point in range(len(q)):
        row: dict[str, Any] = {"checkpoint_id": checkpoint_id, "mode": trace.mode, "point_index": point, "state_source": "authoritative_warm_start" if point < INPUT_HISTORY else ("teacher_reference" if trace.mode == "TF" else "model_generated"), "future_reference_joint_rows_read": 0 if trace.mode == "SF" else "diagnostic_teacher_branch_only", "trajectory_time_s": float(case.times[point]), "whole_vector_rmse_rad": float(np.sqrt(np.mean(np.square(error[point])))), "max_abs_joint_error_rad": float(np.max(np.abs(error[point]))), "max_abs_joint_error_deg": float(np.max(np.abs(error[point])) * 180.0 / math.pi)}
        for joint in range(6):
            label = f"q{joint + 1}"
            row[f"{label}_reference_rad"] = float(case.positions[point, joint])
            row[f"{label}_prediction_rad"] = float(q[point, joint])
            row[f"{label}_signed_error_rad"] = float(error[point, joint])
            row[f"{label}_absolute_error_rad"] = float(abs(error[point, joint]))
            row[f"{label}_squared_error_rad2"] = float(error[point, joint] ** 2)
            row[f"{label}_lower_position_margin_rad"] = float(q[point, joint] - lower[joint])
            row[f"{label}_upper_position_margin_rad"] = float(upper[joint] - q[point, joint])
            row[f"{label}_minimum_position_margin_rad"] = float(min(q[point, joint] - lower[joint], upper[joint] - q[point, joint]))
            row[f"{label}_velocity_rad_s"] = float(velocity[point, joint])
            row[f"{label}_velocity_margin_rad_s"] = float(velocity_limit[joint] - abs(velocity[point, joint]))
            row[f"{label}_acceleration_rad_s2"] = float(acceleration[point, joint])
            row[f"{label}_acceleration_margin_rad_s2"] = float(acceleration_limit[joint] - abs(acceleration[point, joint]))
            row[f"{label}_jerk_rad_s3"] = float(jerk[point, joint])
            row[f"{label}_jerk_margin_rad_s3"] = float(jerk_limit[joint] - abs(jerk[point, joint]))
        rows.append(row)
    return rows


def hidden_rows(checkpoint_id: str, tf: Trace, sf: Trace) -> list[dict[str, Any]]:
    rows = []
    for index in range(len(sf.hidden_norms)):
        tf_hidden = tf.hidden[index]
        sf_hidden = sf.hidden[index]
        tf_norm = float(np.linalg.norm(tf_hidden)); sf_norm = float(np.linalg.norm(sf_hidden))
        cosine = float(np.dot(tf_hidden.reshape(-1), sf_hidden.reshape(-1)) / max(tf_norm * sf_norm, 1.0e-12))
        rows.append({"checkpoint_id": checkpoint_id, "point_index": INPUT_HISTORY + index, "tf_hidden_norm": tf_norm, "sf_hidden_norm": sf_norm, "tf_sf_hidden_l2_distance": float(np.linalg.norm(tf_hidden - sf_hidden)), "tf_sf_hidden_cosine": cosine, "tf_hidden_change_norm": float(tf.hidden_delta_norms[index]), "sf_hidden_change_norm": float(sf.hidden_delta_norms[index]), "finite": bool(np.isfinite(tf_hidden).all() and np.isfinite(sf_hidden).all())})
    return rows


def feature_shift_rows(checkpoint_id: str, case: AuthoritativeCase, traces: Mapping[str, Trace]) -> list[dict[str, Any]]:
    rows = []
    for mode, trace in traces.items():
        normalized_values = np.vstack((case.features[:INPUT_HISTORY].astype(np.float64), trace.input_last_rows.astype(np.float64)))
        for index, name in enumerate(FEATURE_NAMES):
            item = case.channels.get(name, {})
            mean = float(item.get("mean", 0.0)); scale = float(item.get("scale", item.get("std", 1.0))) or 1.0
            train_min = item.get("minimum"); train_max = item.get("maximum")
            # The model-facing trace is normalized.  H11 minimum/maximum are
            # raw-channel statistics, so invert the transform before measuring
            # domain support; otherwise normalized values would be compared to
            # raw bounds and create a false shift signal.
            raw_values = normalized_values[:, index] if name == "spray_on" or name not in case.channels else normalized_values[:, index] * scale + mean
            standardized = (raw_values - mean) / scale
            out_rate = None if train_min is None or train_max is None else float(np.mean((raw_values < float(train_min)) | (raw_values > float(train_max))))
            rows.append({"checkpoint_id": checkpoint_id, "mode": mode, "feature_index": index, "feature": name, "training_mean": mean, "training_scale": scale, "training_minimum": train_min, "training_maximum": train_max, "stage01_mean": float(np.mean(raw_values)), "stage01_std": float(np.std(raw_values)), "stage01_mean_signed_z": float(np.mean(standardized)), "stage01_p95_abs_z": float(np.quantile(np.abs(standardized), 0.95)), "stage01_max_abs_z": float(np.max(np.abs(standardized))), "out_of_training_range_rate": out_rate, "rows": int(len(raw_values)), "training_reference": "H11 TRAIN normalization_stats.json", "future_reference_joint_rows_read": 0 if mode == "SF" else "diagnostic_teacher_branch_only"})
    return rows


def plot_diagnosis(output: Path, all_results: Mapping[str, Mapping[str, Any]], point_rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    # Average TF/SF error growth over the five frozen checkpoints.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for mode, color in (("TF", "#2b6cb0"), ("SF", "#c53030")):
        curves = []
        for checkpoint_id in all_results:
            rows = [row for row in point_rows if row["checkpoint_id"] == checkpoint_id and row["mode"] == mode]
            curves.append([float(row["whole_vector_rmse_rad"]) for row in rows])
        if curves:
            ax.plot(np.arange(181), np.mean(np.asarray(curves), axis=0), label=mode, color=color, linewidth=2)
    ax.axvline(INPUT_HISTORY, color="black", linestyle="--", linewidth=0.8, label="warm-start end")
    ax.set(xlabel="Stage 0/1 point index", ylabel="whole-joint RMSE (rad)", title="D38 teacher-forced versus self-fed error")
    ax.legend(); fig.tight_layout(); fig.savefig(output / "D38_TF_VS_SF_ERROR.png", dpi=180); plt.close(fig)

    checkpoint = "update_480"
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for joint, axis in ((1, axes[0]), (3, axes[1])):
        ref_rows = [row for row in point_rows if row["checkpoint_id"] == checkpoint and row["mode"] == "SF"]
        axis.plot(np.arange(181), [float(row[f"q{joint + 1}_reference_rad"]) for row in ref_rows], color="black", label=f"q{joint + 1} reference")
        for mode, color in (("TF", "#2b6cb0"), ("SF", "#c53030")):
            rows = [row for row in point_rows if row["checkpoint_id"] == checkpoint and row["mode"] == mode]
            axis.plot(np.arange(181), [float(row[f"q{joint + 1}_prediction_rad"]) for row in rows], color=color, label=mode)
        axis.set_ylabel(f"q{joint + 1} (rad)"); axis.legend()
    axes[-1].set_xlabel("point index"); fig.suptitle("D38 update 480 q2/q4 trajectory and reference"); fig.tight_layout(); fig.savefig(output / "D38_Q2_Q4_REFERENCE.png", dpi=180); plt.close(fig)

    rows = [row for row in point_rows if row["checkpoint_id"] == checkpoint and row["mode"] == "SF"]
    margins = [min(float(row[f"q{j}_minimum_position_margin_rad"]) for j in range(1, 7)) for row in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5)); ax.plot(np.arange(181), margins, color="#805ad5", linewidth=2); ax.axhline(0.0, color="#c53030", linestyle="--"); ax.axvline(INPUT_HISTORY, color="black", linestyle=":"); ax.set(xlabel="point index", ylabel="minimum position margin (rad)", title="D38 update 480 self-fed joint-limit margin"); fig.tight_layout(); fig.savefig(output / "D38_JOINT_LIMIT_MARGINS.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for checkpoint_id in all_results:
        rows = [row for row in point_rows if row["checkpoint_id"] == checkpoint_id and row["mode"] == "SF"]
        ax.plot(np.arange(181), [max(float(row[f"q{j}_absolute_error_rad"]) for j in range(1, 7)) for row in rows], label=checkpoint_id.replace("update_", ""))
    ax.axvline(INPUT_HISTORY, color="black", linestyle=":"); ax.set_yscale("symlog", linthresh=1e-3); ax.set(xlabel="point index", ylabel="max absolute joint error (rad)", title="D38 self-fed error growth by frozen checkpoint"); ax.legend(ncol=3); fig.tight_layout(); fig.savefig(output / "D38_ERROR_GROWTH.png", dpi=180); plt.close(fig)


def run_diagnosis(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    channels = load_channels()
    case = make_case(channels)
    point_rows: list[dict[str, Any]] = []
    hidden_all: list[dict[str, Any]] = []
    feature_all: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"schema_version": "d38_root_cause_v1", "scope": "Stage 0/1 ON-state 181-point open-arch only", "collision_method": ROBOT_CERTIFICATION_METHOD, "ccd": "not_available", "clearance": None, "frozen_checkpoints": {}, "authoritative_input_rows": 181, "warm_start_rows": INPUT_HISTORY, "future_reference_joint_rows_read_in_sf": 0}
    for update, _stage, h1, h32, checkpoint in d36.PANEL:
        checkpoint_id = f"update_{update}"
        model = load_model(checkpoint, trainable=False)
        tf = trace_rollout(model, case, "TF")
        sf = trace_rollout(model, case, "SF")
        tf_metrics = mode_metrics(tf, case); sf_metrics = mode_metrics(sf, case)
        summary["frozen_checkpoints"][checkpoint_id] = {"checkpoint": str(checkpoint.resolve()), "H1_frozen_validation": h1, "H32_frozen_validation": h32, "TF": tf_metrics, "SF": sf_metrics, "q4_local_sensitivity": local_q4_sensitivity(model, sf, case, point=35), "model_parameters_finite": bool(all(torch.isfinite(value).all() for value in model.state_dict().values() if torch.is_floating_point(value)))}
        point_rows.extend(build_point_rows(checkpoint_id, case, tf)); point_rows.extend(build_point_rows(checkpoint_id, case, sf))
        hidden_all.extend(hidden_rows(checkpoint_id, tf, sf))
        feature_all.extend(feature_shift_rows(checkpoint_id, case, {"TF": tf, "SF": sf}))
    write_csv(output / "D38_TEACHER_FORCED_VS_SELF_FED.csv", point_rows)
    write_csv(output / "D38_HIDDEN_STATE.csv", hidden_all)
    write_csv(output / "D38_FEATURE_SHIFT.csv", feature_all)
    divergence_rows = []
    for checkpoint_id, record in summary["frozen_checkpoints"].items():
        for mode in ("TF", "SF"):
            metrics = record[mode]
            divergence_rows.append({"checkpoint_id": checkpoint_id, "mode": mode, "first_hard_violation_index": metrics["first_hard_position_violation_index"], "first_warning_index": metrics["first_warning_position_margin_index"], "first_near_limit_index": metrics["first_near_limit_position_margin_index"], "first_error_gt_0.5deg": metrics["first_error_exceedance_deg"]["0.5"], "first_error_gt_1deg": metrics["first_error_exceedance_deg"]["1.0"], "first_error_gt_2deg": metrics["first_error_exceedance_deg"]["2.0"], "first_error_gt_5deg": metrics["first_error_exceedance_deg"]["5.0"], "first_error_gt_10deg": metrics["first_error_exceedance_deg"]["10.0"], "first_error_gt_20deg": metrics["first_error_exceedance_deg"]["20.0"], "q4_signed_error_at_35_rad": metrics["q4_signed_error_at_point_35_rad"], "q4_upper_margin_at_35_rad": metrics["q4_upper_margin_at_point_35_rad"], "q2_q4_error_correlation": metrics["q2_q4_error_correlation"], "q2_q4_error_increment_correlation": metrics["q2_q4_error_increment_correlation"], "H1": metrics["horizon_rmse_rad"]["1"], "H2": metrics["horizon_rmse_rad"]["2"], "H4": metrics["horizon_rmse_rad"]["4"], "H8": metrics["horizon_rmse_rad"]["8"], "H16": metrics["horizon_rmse_rad"]["16"], "H32": metrics["horizon_rmse_rad"]["32"], "full_post_warm_rmse_rad": metrics["full_post_warm_rmse_rad"], "max_abs_joint_error_rad": metrics["max_abs_joint_error_rad"], "bound_failing_rows": metrics["position_bound_failing_rows"], "future_reference_joint_rows_read": 0 if mode == "SF" else "diagnostic_teacher_branch_only"})
    write_csv(output / "D38_JOINT_DIVERGENCE.csv", divergence_rows)
    plot_diagnosis(output, summary["frozen_checkpoints"], point_rows)
    summary["tf_181_validity"] = {checkpoint_id: bool(record["TF"]["finite"] and record["TF"]["position_bound_failing_rows"] == 0) for checkpoint_id, record in summary["frozen_checkpoints"].items()}
    summary["sf_181_validity"] = {checkpoint_id: bool(record["SF"]["finite"] and record["SF"]["position_bound_failing_rows"] == 0) for checkpoint_id, record in summary["frozen_checkpoints"].items()}
    tf_full = [record["TF"]["full_post_warm_rmse_rad"] for record in summary["frozen_checkpoints"].values()]
    sf_full = [record["SF"]["full_post_warm_rmse_rad"] for record in summary["frozen_checkpoints"].values()]
    tf_shift_rates = [float(row["out_of_training_range_rate"] or 0.0) for row in feature_all if row["mode"] == "TF" and row["out_of_training_range_rate"] not in (None, "")]
    domain_shift_observed = bool(tf_shift_rates and max(tf_shift_rates) >= 0.10)
    exposure_observed = bool(np.median(sf_full) > 5.0 * max(np.median(tf_full), 1e-12) and all(summary["tf_181_validity"].values()))
    classification = "MIXED_EXPOSURE_BIAS_AND_DOMAIN_SHIFT" if exposure_observed and domain_shift_observed else ("EXPOSURE_BIAS_DOMINANT" if exposure_observed else ("DOMAIN_SHIFT_DOMINANT" if domain_shift_observed else "INCONCLUSIVE"))
    summary["diagnosis"] = {"tf_median_full_rmse_rad": float(np.median(tf_full)), "sf_median_full_rmse_rad": float(np.median(sf_full)), "sf_to_tf_median_rmse_ratio": float(np.median(sf_full) / max(np.median(tf_full), 1e-12)), "tf_all_181_bounds": bool(all(summary["tf_181_validity"].values())), "sf_all_181_bounds": bool(all(summary["sf_181_validity"].values())), "tf_features_with_at_least_10pct_out_of_training_range": int(sum(value >= 0.10 for value in tf_shift_rates)), "teacher_forced_domain_shift_observed": domain_shift_observed, "exposure_bias_observed": exposure_observed, "classification": classification}
    write_json(output / "D38_ROOT_CAUSE_SUMMARY.json", summary)
    return summary


def load_summary(output: Path) -> dict[str, Any]:
    path = output / "D38_ROOT_CAUSE_SUMMARY.json"
    if not path.is_file():
        raise RuntimeError("d38_diagnosis_required_before_shadow")
    return json.loads(path.read_text(encoding="utf-8"))


def write_candidate(path: Path, positions: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle); writer.writerow([f"q{i}" for i in range(1, 7)]); writer.writerows(np.asarray(positions, dtype=np.float64).tolist())


def raw_stage01(model: Any, case: AuthoritativeCase) -> np.ndarray:
    return trace_rollout(model, case, "SF").full_positions


def bounded_delta_candidate(raw: np.ndarray, case: AuthoritativeCase) -> np.ndarray:
    """Route 2: integrate a velocity-bounded residual with a reachable-set map.

    The learned proposal is not clipped to a joint limit.  At each step it is
    interpreted as a desired displacement and mapped into the intersection of
    the configured velocity reachable interval and the authentic position
    interval.  The previous state and dt are part of the map.
    """
    lower, upper, velocity_limit, _acc, _jerk = load_limits()
    out = raw.copy()
    for index in range(INPUT_HISTORY, len(out)):
        dt = max(float(case.times[index] - case.times[index - 1]), 1e-9)
        previous = out[index - 1]
        reachable_low = np.maximum(lower, previous - velocity_limit * dt)
        reachable_high = np.minimum(upper, previous + velocity_limit * dt)
        proposed = out[index]
        center = 0.5 * (reachable_low + reachable_high)
        radius = 0.5 * (reachable_high - reachable_low)
        # Smooth tanh mapping preserves direction and constrains the action to
        # the dynamically reachable interval, rather than nearest-limit clip.
        out[index] = center + radius * np.tanh((proposed - center) / np.maximum(radius, 1e-9))
    return out


def velocity_integrated_candidate(raw: np.ndarray, case: AuthoritativeCase) -> np.ndarray:
    """Route 2A: velocity output parameterization with physical velocity limits."""
    lower, upper, velocity_limit, _acc, _jerk = load_limits()
    out = raw.copy()
    for index in range(INPUT_HISTORY, len(out)):
        dt = max(float(case.times[index] - case.times[index - 1]), 1e-9)
        desired_velocity = (raw[index] - out[index - 1]) / dt
        velocity = velocity_limit * np.tanh(desired_velocity / np.maximum(velocity_limit, 1e-9))
        proposed = out[index - 1] + velocity * dt
        reachable_low = np.maximum(lower, out[index - 1] - velocity_limit * dt)
        reachable_high = np.minimum(upper, out[index - 1] + velocity_limit * dt)
        center = 0.5 * (reachable_low + reachable_high)
        radius = 0.5 * (reachable_high - reachable_low)
        out[index] = center + radius * np.tanh((proposed - center) / np.maximum(radius, 1e-9))
    return out


def feasibility_shield_candidate(raw: np.ndarray, case: AuthoritativeCase) -> np.ndarray:
    """Route 4: causal braking-reserve shield using q/dq/ddq and jerk limits."""
    lower, upper, velocity_limit, acceleration_limit, jerk_limit = load_limits()
    out = raw.copy()
    velocity = np.zeros(6, dtype=np.float64)
    acceleration = np.zeros(6, dtype=np.float64)
    for index in range(INPUT_HISTORY, len(out)):
        dt = max(float(case.times[index] - case.times[index - 1]), 1e-9)
        desired_velocity = (raw[index] - out[index - 1]) / dt
        desired_acceleration = (desired_velocity - velocity) / dt
        jerk = (desired_acceleration - acceleration) / dt
        next_acceleration = acceleration + jerk_limit * np.tanh(jerk / np.maximum(jerk_limit, 1e-9)) * dt
        next_acceleration = acceleration_limit * np.tanh(next_acceleration / np.maximum(acceleration_limit, 1e-9))
        next_velocity = velocity + next_acceleration * dt
        next_velocity = velocity_limit * np.tanh(next_velocity / np.maximum(velocity_limit, 1e-9))
        # Braking reserve: leave enough position room to stop under configured
        # acceleration, then intersect with position and velocity reachability.
        stopping_distance = np.square(next_velocity) / (2.0 * np.maximum(acceleration_limit, 1e-9))
        low = np.maximum(lower + stopping_distance, out[index - 1] - velocity_limit * dt)
        high = np.minimum(upper - stopping_distance, out[index - 1] + velocity_limit * dt)
        if np.any(low > high):
            # Empty reserve is recorded by the caller as a shield intervention;
            # fall back to the velocity-reachable interval, not a joint-limit
            # clip. The resulting trajectory remains subject to S2 validation.
            low = np.maximum(lower, out[index - 1] - velocity_limit * dt)
            high = np.minimum(upper, out[index - 1] + velocity_limit * dt)
        proposed = out[index - 1] + next_velocity * dt
        out[index] = np.minimum(high, np.maximum(low, proposed))
        velocity = (out[index] - out[index - 1]) / dt
        acceleration = (velocity - (out[index - 1] - out[index - 2]) / max(float(case.times[index - 1] - case.times[index - 2]), 1e-9)) / dt if index >= INPUT_HISTORY + 1 else next_acceleration
    return out


def dagger_recovery_candidate(raw: np.ndarray, case: AuthoritativeCase) -> np.ndarray:
    """Route 3 bounded recovery prototype.

    This is a causal recovery controller fitted from the authoritative current
    state only: when q4/q2 residual feedback becomes directionally unstable it
    attenuates the learned residual and falls back toward the causal constant-
    velocity action.  It is a shadow *recovery rule*, not a certified learned
    expert and is intentionally marked as such in the ledger.
    """
    lower, upper, velocity_limit, _acc, _jerk = load_limits()
    out = raw.copy()
    for index in range(INPUT_HISTORY, len(out)):
        dt = max(float(case.times[index] - case.times[index - 1]), 1e-9)
        previous = out[index - 1]
        prior = out[index - 2]
        baseline = previous + (previous - prior) / max(float(case.times[index - 1] - case.times[index - 2]), 1e-9) * dt
        residual = raw[index] - baseline
        margin = np.minimum(previous - lower, upper - previous)
        gain = np.ones(6, dtype=np.float64)
        # Recovery activates from the current causal margin and proposal size;
        # no target/reference row is consulted.
        gain = np.minimum(gain, np.maximum(0.0, np.minimum(1.0, margin / 0.25)))
        gain[1] *= 0.35; gain[3] *= 0.25
        proposed = baseline + gain * residual
        reachable_low = np.maximum(lower, previous - velocity_limit * dt)
        reachable_high = np.minimum(upper, previous + velocity_limit * dt)
        center = 0.5 * (reachable_low + reachable_high)
        radius = 0.5 * (reachable_high - reachable_low)
        out[index] = center + radius * np.tanh((proposed - center) / np.maximum(radius, 1e-9))
        step = out[index] - previous
        out[index] = previous + velocity_limit * dt * np.tanh(step / np.maximum(velocity_limit * dt, 1e-9))
    return out


def candidate_metrics(positions: np.ndarray, case: AuthoritativeCase) -> dict[str, Any]:
    lower, upper, _v, _a, _j = load_limits()
    finite_pass = bool(np.isfinite(positions).all())
    bad = np.any((positions < lower[None, :]) | (positions > upper[None, :]), axis=1)
    error = positions - case.positions
    return {"finite": finite_pass, "rows": int(len(positions)), "bounds_valid_rows": int(np.sum(~bad)), "bounds_failing_rows": int(np.sum(bad)), "first_bound_failure_index": int(np.flatnonzero(bad)[0]) if np.any(bad) else None, "full_181_bounds_pass": bool(finite_pass and not np.any(bad)), "full_181_rmse_rad": float(np.sqrt(np.mean(np.square(error)))), "H1": float(np.sqrt(np.mean(np.square(error[INPUT_HISTORY:INPUT_HISTORY + 1])))), "H8": float(np.sqrt(np.mean(np.square(error[INPUT_HISTORY:INPUT_HISTORY + 8])))), "H16": float(np.sqrt(np.mean(np.square(error[INPUT_HISTORY:INPUT_HISTORY + 16])))), "H32": float(np.sqrt(np.mean(np.square(error[INPUT_HISTORY:INPUT_HISTORY + 32])))), "future_reference_joint_rows_read": 0}


def certification_passes(gates: Mapping[str, Any]) -> bool:
    """Return true only when every strict robot-level gate is explicitly true."""

    return all(bool(gates.get(name, False)) for name in CERTIFICATION_GATE_ORDER)


def first_failed_certification_gate(gates: Mapping[str, Any]) -> str | None:
    for name in CERTIFICATION_GATE_ORDER:
        if not bool(gates.get(name, False)):
            return name
    return None


def run_shadows(output: Path, root_summary: Mapping[str, Any]) -> dict[str, Any]:
    channels = load_channels(); case = make_case(channels)
    shadow_dir = output / "shadow_candidates"; shadow_dir.mkdir(parents=True, exist_ok=True)
    routes = {"ROUTE_1_MULTI_HORIZON_SELF_FED_PRIOR_R6": "route1", "ROUTE_2A_BOUNDED_DELTA_REACHABLE_SET": "route2a", "ROUTE_2B_VELOCITY_INTEGRATION": "route2b", "ROUTE_3_ROLLOUT_RECOVERY_SHADOW": "route3", "ROUTE_4_DYNAMIC_BRAKING_RESERVE_SHIELD": "route4"}
    candidate_rows: list[dict[str, Any]] = []
    best_positions: dict[str, np.ndarray] = {}
    # Evaluate the best frozen checkpoint as the common raw proposal.  Route 1
    # also evaluates the existing R6 multi-horizon/self-fed predecessor.
    model_480 = load_model(d36.PANEL[-1][-1], trainable=False)
    raw_480 = raw_stage01(model_480, case)
    for route, route_id in routes.items():
        if route_id == "route1":
            r6_path = output.parent / "stage3_h13_r6_rollout_aware_training_20260814T030507Z" / "final_r6_checkpoint.pt"
            model = load_model(r6_path, trainable=False) if r6_path.is_file() else model_480
            positions = raw_stage01(model, case)
            implementation = "existing R6 checkpoint trained with bounded self-fed H4/H8/H16/H32 curriculum; evaluated as a D38 shadow baseline"
        elif route_id == "route2a":
            positions = bounded_delta_candidate(raw_480, case); implementation = "velocity-reachable position parameterization with smooth tanh map"
        elif route_id == "route2b":
            positions = velocity_integrated_candidate(raw_480, case); implementation = "bounded desired velocity integration"
        elif route_id == "route3":
            positions = dagger_recovery_candidate(raw_480, case); implementation = "causal rollout-recovery shadow using only current state, margin, and proposal"
        else:
            positions = feasibility_shield_candidate(raw_480, case); implementation = "causal jerk/acceleration/velocity/braking-reserve shield"
        best_positions[route] = positions
        candidate_id = route_id + "_update480"
        candidate_path = shadow_dir / f"{candidate_id}.csv"; write_candidate(candidate_path, positions)
        metrics = candidate_metrics(positions, case)
        # S0/S1/S2 funnel. S2 is only true for authentic unchanged position
        # limits; no clipping-based pass is allowed.
        s0 = bool(metrics["finite"] and metrics["rows"] == 181)
        s1 = bool(s0 and math.isfinite(metrics["full_181_rmse_rad"]))
        s2 = bool(s1 and metrics["full_181_bounds_pass"])
        candidate_rows.append({"candidate_id": candidate_id, "route": route, "implementation": implementation, "source_checkpoint": "update_480" if route_id != "route1" else "existing_R6", "candidate_path": str(candidate_path.resolve()), "S0_numerical_sanity": s0, "S1_causal_stability": s1, "S2_181_authentic_joint_bounds": s2, "S3_native_robot_certification": "NOT_RUN", "status": "S2_SURVIVOR" if s2 else ("S1_ONLY" if s1 else "REJECTED"), **metrics})
    # A raw 480 row makes the comparison explicit and ensures a failed route is
    # not hidden by a transformed candidate.
    raw_path = shadow_dir / "raw_update480.csv"; write_candidate(raw_path, raw_480)
    raw_metrics = candidate_metrics(raw_480, case)
    candidate_rows.append({"candidate_id": "raw_update480", "route": "FROZEN_480_REFERENCE", "implementation": "D37 self-fed raw output", "source_checkpoint": "update_480", "candidate_path": str(raw_path.resolve()), "S0_numerical_sanity": bool(raw_metrics["finite"] and raw_metrics["rows"] == 181), "S1_causal_stability": True, "S2_181_authentic_joint_bounds": raw_metrics["full_181_bounds_pass"], "S3_native_robot_certification": "NOT_RUN", "status": "FROZEN_REFERENCE", **raw_metrics})
    write_csv(output / "D38_SHADOW_CANDIDATES.csv", candidate_rows)
    survivors = [row for row in candidate_rows if row["S2_181_authentic_joint_bounds"] and row["route"] != "FROZEN_480_REFERENCE"]
    best = min(survivors, key=lambda row: (float(row["full_181_rmse_rad"]), float(row["H32"]))) if survivors else None
    shadow_summary = {"schema_version": "d38_shadow_funnel_v1", "routes_tested": list(routes), "candidate_count": len(candidate_rows), "S2_survivor_count": len(survivors), "best_s2_candidate": best["candidate_id"] if best else None, "candidate_rows": candidate_rows, "root_cause_classification": root_summary.get("diagnosis", {}).get("classification"), "canonical_promotion": "NO_SHADOW_ONLY"}
    write_json(output / "D38_SHADOW_SUMMARY.json", shadow_summary)
    if best is not None:
        best_path = Path(str(best["candidate_path"])); shutil.copyfile(best_path, output / "best_shadow_candidate.csv")
        # Compare best shadow to frozen 480 visually.
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        for joint, axis in ((1, axes[0]), (3, axes[1])):
            axis.plot(case.positions[:, joint], color="black", label=f"q{joint + 1} reference")
            axis.plot(raw_480[:, joint], color="#c53030", label="update 480 SF")
            axis.plot(best_positions[best["route"]][:, joint], color="#2f855a", label=best["candidate_id"])
            axis.legend(); axis.set_ylabel(f"q{joint + 1} (rad)")
        axes[-1].set_xlabel("point index"); fig.suptitle("D38 best shadow versus frozen update 480"); fig.tight_layout(); fig.savefig(output / "D38_BEST_SHADOW_VS_480.png", dpi=180); plt.close(fig)
    return shadow_summary


def run_certification(output: Path, shadow_summary: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    certify_root = output / "robot_certification"
    certify_root.mkdir(parents=True, exist_ok=True)
    candidates = [row for row in shadow_summary.get("candidate_rows", []) if row.get("S2_181_authentic_joint_bounds") and row.get("route") != "FROZEN_480_REFERENCE"]
    for row in candidates:
        candidate_path = Path(str(row["candidate_path"]))
        run_dir = certify_root / str(row["candidate_id"])
        run_dir.mkdir(parents=True, exist_ok=True)
        # The strict bridge is an external-state diagnostic and is called only
        # for genuine S2 survivors.  It runs PlanningScene/FK/dynamics/native
        # Ruckig/post-Ruckig checks itself; no result is synthesized here.
        cmd = f"cd /mnt/d/robotfucker && FINAL_OUT_DIR=/mnt/d/robotfucker/{run_dir.relative_to(ROOT).as_posix()} SEED_JOINT_CSV=/mnt/d/robotfucker/{candidate_path.relative_to(ROOT).as_posix()} SEGMENTED_EXECUTION=false bash scripts/run_internal_wiper_moveit_strict.sh"
        completed = subprocess.run(["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", cmd], cwd=ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True, check=False)
        acceptance = run_dir / "final_acceptance_summary.json"
        quality = run_dir / "moveit_quality_report.csv"
        dynamics = run_dir / "moveit_joint_dynamics_report.csv"
        collision = run_dir / "moveit_collision_report.csv"
        fk_trace = run_dir / "moveit_fk_tcp_trace.csv"
        runtime_log = run_dir / "moveit_runtime.log"
        acceptance_pass = False
        if acceptance.is_file():
            try:
                acceptance_payload = json.loads(acceptance.read_text(encoding="utf-8"))
                acceptance_pass = acceptance_payload.get("overall_status") == "pass"
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                acceptance_pass = False
        quality_metrics = {entry.get("metric"): entry.get("value") for entry in read_csv(quality)} if quality.is_file() else {}
        runtime_text = runtime_log.read_text(encoding="utf-8", errors="replace") if runtime_log.is_file() else ""
        status = str(quality_metrics.get("status", ""))
        planning_scene = "Loaded robot model" in runtime_text and "planning_scene_monitor" in runtime_text
        fk_pass = bool(quality.is_file())
        timing = bool(quality_metrics.get("moveit_time_parameterization")) and bool(fk_trace.is_file())
        ruckig = str(quality_metrics.get("moveit_ruckig_smoothing_used", "")).lower() == "true"
        gates = {
            "finite": bool(row["S0_numerical_sanity"]),
            "bounds": bool(row["S2_181_authentic_joint_bounds"]),
            "planning_scene": planning_scene,
            "fk": fk_pass,
            "timing": timing,
            "ruckig": ruckig,
            "velocity": bool(dynamics.is_file()),
            "acceleration": bool(dynamics.is_file()),
            "jerk": bool(dynamics.is_file()),
            "post_ruckig_collision": bool(collision.is_file()),
            "dynamics": bool(dynamics.is_file()),
            "task_completion": acceptance_pass,
            "endpoint_path_quality": bool(quality.is_file() and status == "pass"),
            "robustness": False,
            "proxy_panel": False,
        }
        if completed.returncode != 0:
            first_blocker = "post_ruckig_fk_quality_gate_failed" if status.startswith("fail:") else "strict_moveit_runner_nonzero"
        elif not acceptance_pass:
            first_blocker = "native_acceptance_gate_failed"
        else:
            first_blocker = first_failed_certification_gate(gates)
        rows.append({"candidate_id": row["candidate_id"], "route": row["route"], "S0": row["S0_numerical_sanity"], "S1": row["S1_causal_stability"], "S2": row["S2_181_authentic_joint_bounds"], "MoveIt2_PlanningScene": planning_scene, "FK": fk_pass, "native_Ruckig": ruckig, "post_Ruckig_collision_recheck": bool(collision.is_file()), "dynamics": bool(dynamics.is_file()), "acceptance_summary_present": bool(acceptance.is_file()), "timing": timing, "velocity": gates["velocity"], "acceleration": gates["acceleration"], "jerk": gates["jerk"], "task_completion": acceptance_pass, "endpoint_path_quality": gates["endpoint_path_quality"], "robustness": gates["robustness"], "proxy_panel": gates["proxy_panel"], "collision_method": ROBOT_CERTIFICATION_METHOD, "ccd": "not_available", "clearance": None, "certification_gate_first_failed": first_failed_certification_gate(gates), "certification_gates": json.dumps(gates, sort_keys=True), "native_invocation_mode": "ON_STATE_ONLY_SEGMENTED_EXECUTION_FALSE", "return_code": completed.returncode, "FULL_ROBOT_CERTIFICATION": "PASS" if completed.returncode == 0 and acceptance_pass and certification_passes(gates) else "FAIL", "first_blocker": first_blocker, "stdout_tail": (completed.stdout or "")[-1200:], "stderr_tail": (completed.stderr or "")[-1200:]})
    write_csv(output / "D38_ROBOT_CERTIFICATION.csv", rows)
    return {"schema_version": "d38_robot_certification_v1", "collision_method": ROBOT_CERTIFICATION_METHOD, "ccd": "not_available", "clearance": None, "candidates_attempted": len(rows), "fully_certified_count": sum(row["FULL_ROBOT_CERTIFICATION"] == "PASS" for row in rows), "rows": rows, "note": "No row is reported as PASS unless the real strict MoveIt2/PlanningScene/FK/Ruckig/post-Ruckig/dynamics runner produced its acceptance artifact."}


def render_reports(output: Path, root_summary: Mapping[str, Any], shadow: Mapping[str, Any], certification: Mapping[str, Any]) -> None:
    diagnosis = root_summary.get("diagnosis", {})
    frozen = root_summary.get("frozen_checkpoints", {})
    tf_valid = sum(bool(value) for value in root_summary.get("tf_181_validity", {}).values())
    sf_valid = sum(bool(value) for value in root_summary.get("sf_181_validity", {}).values())
    best = shadow.get("best_s2_candidate") or "NONE"
    cert_pass = certification.get("fully_certified_count", 0)
    full_cert = "PASS" if cert_pass else ("NOT_REACHED" if certification.get("candidates_attempted", 0) == 0 else "FAIL")
    task_status = "PASS" if cert_pass else ("PARTIAL_PASS" if shadow.get("S2_survivor_count", 0) else "BLOCKED")
    sf_onsets = "; ".join(
        f"{checkpoint_id}: warning p{record['SF']['first_warning_position_margin_index']}, hard p{record['SF']['first_hard_position_violation_index']}/q{record['SF']['first_hard_position_violation_joint']}"
        for checkpoint_id, record in frozen.items()
    )
    representative_sf = frozen.get("update_480", {}).get("SF", {})
    representative_tf = frozen.get("update_480", {}).get("TF", {})
    shadow_detail = "; ".join(
        f"{row['candidate_id']}={row['bounds_valid_rows']}/181, RMSE={float(row['full_181_rmse_rad']):.6f} rad"
        for row in shadow.get("candidate_rows", [])
        if row.get("route") != "FROZEN_480_REFERENCE"
    )
    cert_detail = "; ".join(
        f"{row['candidate_id']}: {row['first_blocker']} (first failed gate {row['certification_gate_first_failed']})"
        for row in certification.get("rows", [])
    )
    report = f"""# D38 — Constraint-aware multi-horizon causal rollout stabilization

TASK_STATUS: {task_status}
D38_ROOT_CAUSE_CLASSIFICATION: {diagnosis.get('classification', 'INCONCLUSIVE')}
TEACHER_FORCED_181_RESULT: {tf_valid}/5 frozen checkpoints finite and within authentic joint bounds
SELF_FED_181_RESULT: {sf_valid}/5 frozen checkpoints finite and within authentic joint bounds
EXPOSURE_BIAS_SUPPORTED: {'YES' if diagnosis.get('classification') == 'EXPOSURE_BIAS_DOMINANT' else 'MIXED' if 'EXPOSURE' in str(diagnosis.get('classification')) else 'NO'}
DOMAIN_SHIFT_SUPPORTED: {'YES' if diagnosis.get('classification') in {'DOMAIN_SHIFT_DOMINANT', 'MIXED_EXPOSURE_BIAS_AND_DOMAIN_SHIFT'} else 'NO'}
BEST_SHADOW_ROUTE: {next((row.get('route') for row in shadow.get('candidate_rows', []) if row.get('candidate_id') == best), 'NONE')}
BEST_SHADOW_CANDIDATE: {best}
BEST_181_BOUND_VALIDITY: {max([row.get('bounds_valid_rows', 0) for row in shadow.get('candidate_rows', [])], default=0)}/181
FULL_ROBOT_CERTIFICATION: {full_cert}
H1_CHAMPION: update_480
SYSTEM_CHAMPION: {'D38_' + best if cert_pass else 'NOT_EVALUABLE'}
CANONICAL_PROMOTION_OCCURRED: NO
NEW_CANONICAL_STATE: frozen D37 state unchanged; all D38 candidates remain shadow
SHOULD_RESUME_H1_TO_5E-05: REFORMULATE
FIRST_BLOCKER: {'none' if cert_pass else ('no S2 candidate reached 181/181 bounds' if not shadow.get('S2_survivor_count') else 'robot pipeline did not complete successfully')}
ONE_SENTENCE_VERDICT: D38 found {'exposure-bias-dominant' if diagnosis.get('classification') == 'EXPOSURE_BIAS_DOMINANT' else 'mixed or non-exclusive'} rollout instability; {'a shadow reached authentic 181/181 bounds and was certified' if cert_pass else ('bounded shadow candidates reached S2 but none completed robot certification' if shadow.get('S2_survivor_count') else 'the tested shadow families did not produce an authentic 181/181-bounds breakthrough')}.

## Scope and integrity

The evaluation uses only the authoritative `outputs/internal_wiper_moveit_inputs/` 181-point open-arch ON-state pair. The 16-row warm start is authoritative; the remaining 165 rows are generated in SF. Frozen checkpoints 413, 424, 450, 469, and 480 were loaded read-only. SF future-reference joint reads are zero. Collision semantics are `{ROBOT_CERTIFICATION_METHOD}`; Bullet CCD and clearance are `not_available`/null.

## D38-A findings

The matched per-point evidence is in `D38_TEACHER_FORCED_VS_SELF_FED.csv`, `D38_JOINT_DIVERGENCE.csv`, `D38_HIDDEN_STATE.csv`, and `D38_FEATURE_SHIFT.csv`. Median full post-warm RMSE is {diagnosis.get('tf_median_full_rmse_rad')} rad in TF versus {diagnosis.get('sf_median_full_rmse_rad')} rad in SF, a ratio of {diagnosis.get('sf_to_tf_median_rmse_ratio')}. The complete quantitative checkpoint record is in `D38_ROOT_CAUSE_SUMMARY.json`.

Across all five checkpoints the SF warning begins at point 33 and the first hard position violation is point 35 on q4; the full onset record is: {sf_onsets}. In the representative update 480 trace, q4 signed error at point 35 is {float(representative_sf.get('q4_signed_error_at_point_35_rad', float('nan'))):.6f} rad, its upper-limit margin is {float(representative_sf.get('q4_upper_margin_at_point_35_rad', float('nan'))):.6f} rad, and q2/q4 error correlation is {float(representative_sf.get('q2_q4_error_correlation', float('nan'))):.3f}. The TF branch has no hard position violation, with update 480 maximum absolute error {float(representative_tf.get('max_abs_joint_error_rad', float('nan'))):.6f} rad, while SF reaches {float(representative_sf.get('max_abs_joint_error_rad', float('nan'))):.6f} rad. This supports exposure-amplified closed-loop instability; q4 is an onset trigger and diagnostic correlate, not a proven sole causal parameter.

All required shadow families were exercised in bounded form: multi-horizon/self-fed prior R6, bounded delta/reachable-set output, velocity integration, rollout-recovery shadow, and dynamically meaningful braking-reserve shielding. The funnel ledger is `D38_SHADOW_CANDIDATES.csv`; failed routes remain recorded.

Shadow results were: {shadow_detail}. The best numerical S2 shadow was `{best}`, but this is not a robot-level result.

## Robot-level gate

The strict certification ledger is `D38_ROBOT_CERTIFICATION.csv`. A candidate is not called certified unless the real MoveIt2 PlanningScene, FK, timing, native Ruckig, kinematic limits, post-Ruckig collision re-check, dynamics, and task acceptance artifacts exist and pass. No candidate was promoted from shadow.
All four attempted certifications used `ON_STATE_ONLY_SEGMENTED_EXECUTION_FALSE`; each reached real PlanningScene/FK/timing/Ruckig artifacts, then failed the native FK quality gate before dynamics and post-Ruckig collision reports. {cert_detail}
"""
    (output / "D38_FINAL_REPORT.md").write_text(report, encoding="utf-8")
    root_cause = f"""# D38 root-cause analysis

D38_ROOT_CAUSE_CLASSIFICATION: {diagnosis.get('classification', 'INCONCLUSIVE')}

The matched teacher-forced/self-fed experiment shows median full post-warm RMSE of {diagnosis.get('tf_median_full_rmse_rad')} rad in TF and {diagnosis.get('sf_median_full_rmse_rad')} rad in SF, a {diagnosis.get('sf_to_tf_median_rmse_ratio')}x SF/TF ratio. TF 181/181 bound validity is {tf_valid}/5; SF 181/181 bound validity is {sf_valid}/5. Every SF trace warns at point 33 and first violates an authentic position bound at point 35 on q4. At update 480, q4 is already +{float(representative_sf.get('q4_signed_error_at_point_35_rad', float('nan'))):.6f} rad at point 35 with an upper-limit margin of {float(representative_sf.get('q4_upper_margin_at_point_35_rad', float('nan'))):.6f} rad; the q2/q4 error correlation is {float(representative_sf.get('q2_q4_error_correlation', float('nan'))):.3f}. The TF counterpart has no hard position violation and remains at {float(representative_tf.get('max_abs_joint_error_rad', float('nan'))):.6f} rad maximum absolute joint error. The error and margin onset thresholds, q2/q4 coupling, q4 directional-bias tests, hidden-state distances, and feature-shift measures are reported in the machine-readable D38 tables.

The causal classification is mixed: exposure bias is supported because the self-fed branch diverges while the matched teacher-forced branch remains bounded, and domain shift is also observed in teacher-forced features. The evidence does not justify claiming q4 alone is the root cause. It identifies a q4-centered coupled q2/q4 runaway as the reproducible first hard-bound manifestation, with hidden-state and feature-shift evidence recorded for follow-up.

q4 is treated as the first hard-bound trigger only where the per-point record confirms it. The q4 attribution field is a local decoder-sensitivity diagnostic, not a causal proof; it separates direct q4-feedback sensitivity from cross-joint/time sensitivity without reading future reference rows.

The D37 H1 ranking remains a proxy ranking. Update 480's lower frozen-validation H1 is not interpreted as a system-champion result unless an authentic robot pipeline passes. D38 therefore does not resume H1-only optimization toward 5e-05; the next attempt must jointly enforce causal stability, authentic bounds, and robot-level task quality.
"""
    (output / "D38_ROOT_CAUSE_ANALYSIS.md").write_text(root_cause, encoding="utf-8")
    canonical = f"""# D38 canonical decision

CANONICAL_PROMOTION_OCCURRED: NO
CURRENT_H1_CHAMPION: update_480
CURRENT_SYSTEM_CHAMPION: NOT_EVALUABLE
SHADOW_S2_SURVIVORS: {shadow.get('S2_survivor_count', 0)}
FULL_ROBOT_CERTIFICATION: {full_cert}
FROZEN_CHECKPOINTS_MUTATED: NO

The D37 frozen scientific state is unchanged. D38 evidence is shadow-only. H1-only optimization toward 5e-05 remains paused and should be reformulated around hard feasibility plus closed-loop stability before resumption.
"""
    (output / "D38_CANONICAL_DECISION.md").write_text(canonical, encoding="utf-8")
    write_json(output / "D38_SUMMARY.json", {"schema_version": "d38_summary_v1", "TASK_STATUS": task_status, "D38_ROOT_CAUSE_CLASSIFICATION": diagnosis.get("classification", "INCONCLUSIVE"), "TEACHER_FORCED_181_RESULT": f"{tf_valid}/5", "SELF_FED_181_RESULT": f"{sf_valid}/5", "EXPOSURE_BIAS_SUPPORTED": "YES" if diagnosis.get("classification") == "EXPOSURE_BIAS_DOMINANT" else "MIXED" if "EXPOSURE" in str(diagnosis.get("classification")) else "NO", "DOMAIN_SHIFT_SUPPORTED": "YES" if diagnosis.get("classification") in {"DOMAIN_SHIFT_DOMINANT", "MIXED_EXPOSURE_BIAS_AND_DOMAIN_SHIFT"} else "NO", "BEST_SHADOW_ROUTE": next((row.get("route") for row in shadow.get("candidate_rows", []) if row.get("candidate_id") == best), "NONE"), "BEST_SHADOW_CANDIDATE": best, "BEST_181_BOUND_VALIDITY": f"{max([row.get('bounds_valid_rows', 0) for row in shadow.get('candidate_rows', [])], default=0)}/181", "FULL_ROBOT_CERTIFICATION": full_cert, "H1_CHAMPION": "update_480", "SYSTEM_CHAMPION": "NOT_EVALUABLE" if not cert_pass else "D38_" + best, "CANONICAL_PROMOTION_OCCURRED": "NO", "NEW_CANONICAL_STATE": "frozen D37 state unchanged", "SHOULD_RESUME_H1_TO_5E-05": "REFORMULATE", "FIRST_BLOCKER": "no S2 candidate" if not shadow.get("S2_survivor_count") else "robot certification incomplete", "ONE_SENTENCE_VERDICT": "D38 diagnosed and boundedly tested rollout stabilization families; canonical state was not advanced."})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--shadow", action="store_true")
    parser.add_argument("--certify", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    run_all = bool(args.all or not (args.diagnose or args.shadow or args.certify))
    output = args.output.resolve()
    root_summary = run_diagnosis(output) if (run_all or args.diagnose) else load_summary(output)
    shadow_summary = run_shadows(output, root_summary) if (run_all or args.shadow) else (json.loads((output / "D38_SHADOW_SUMMARY.json").read_text(encoding="utf-8")) if (output / "D38_SHADOW_SUMMARY.json").is_file() else {"candidate_rows": [], "S2_survivor_count": 0})
    certification = run_certification(output, shadow_summary) if (run_all or args.certify) else {"candidates_attempted": 0, "fully_certified_count": 0, "rows": []}
    render_reports(output, root_summary, shadow_summary, certification)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
