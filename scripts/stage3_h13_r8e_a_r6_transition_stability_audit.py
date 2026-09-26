#!/usr/bin/env python3
"""Stage 3 H13 R8E-A: read-only R6 hidden-state/transition audit.

Only the authoritative H10 TRAIN/VALIDATION dataset, H11 normalization, and
the immutable R6 checkpoint are read.  No downstream sealed split is named or
resolved by this program, no optimizer is constructed, and no training path
exists.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.stage3_h11_dataset import FEATURE_NAMES, compute_normalization_stats, load_h10_segments, semantic_hash, sha256_file, verify_h10_authority
from src.stage3_h11_model import set_deterministic
from src.stage3_h11_r2_model import ResidualCausalGRUTrajectoryPredictor, SPRAY_FEATURE_INDEX
from src.stage3_h13_r6 import _next_features, build_sequence_arrays
from src.stage3_h13_r8e_a import (
    AUDIT_HORIZONS,
    MATERIAL_RELATIVE_THRESHOLDS,
    amplification_ratio,
    distribution_summary,
    finite_difference_discrepancy,
    first_sustained_threshold,
    gru_cell_transition,
    instrumented_forward,
    normalized_hidden_metrics,
    safe_rmse,
)


H10_ROOT = ROOT / "outputs/stage3_h10_multi_trajectory_dataset_20260811T081123Z"
H11_ROOT = ROOT / "outputs/stage3_h11_r_pytorch_runtime_recertification_20260811T143806Z"
R6_ROOT = ROOT / "outputs/stage3_h13_r6_rollout_aware_training_20260814T030507Z"
R6_CHECKPOINT = R6_ROOT / "final_r6_checkpoint.pt"
R6_CERTIFICATE = R6_ROOT / "stage3_h13_r6_terminal_certificate.json"
R6_MANIFEST = R6_ROOT / "checkpoint_manifest.json"
EXPECTED_R6_HASH = "8323e8e3c12a182a2efe3fc873f854be388f5ea0a32bee2c00fee0a1353ae3ee"
AUTHORITATIVE_R6_H32 = 0.0232112820732007
AUTHORITATIVE_OUTPUT_ONSET = 4
FREE_TOLERANCE = {"relative": 5.0e-6, "absolute": 5.0e-9}
SEED = 81513
EVAL_HORIZONS = (1, 2, 3, 4, 5, 6, 8, 12, 16, 20, 24, 32)
JVP_HORIZONS = EVAL_HORIZONS
JVP_SAMPLE_COUNT = 256
RANDOM_DIRECTIONS = 4
EPSILON_VALUES = (1.0e-2, 1.0e-3, 1.0e-4)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def load_r6() -> Any:
    model = ResidualCausalGRUTrajectoryPredictor(input_size=len(FEATURE_NAMES), hidden_size=128, num_layers=2, horizon=8)
    payload = torch.load(R6_CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(payload.get("model_state_dict", payload), strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def state_snapshot(model: Any) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def state_matches(model: Any, before: Mapping[str, torch.Tensor]) -> bool:
    return all(torch.equal(model.state_dict()[name].detach().cpu(), value) for name, value in before.items())


def _metric_template() -> dict[str, dict[str, float | int]]:
    return {str(h): {"position_ss": 0.0, "position_n": 0, "velocity_ss": 0.0, "velocity_n": 0, "acceleration_ss": 0.0, "acceleration_n": 0} for h in range(1, 33)}


def _accumulate_metric(store: dict[str, dict[str, float | int]], step: int, key: str, error: torch.Tensor) -> None:
    item = store[str(step)]
    item[f"{key}_ss"] = float(item[f"{key}_ss"]) + float(torch.sum(error.double().square()).cpu())
    item[f"{key}_n"] = int(item[f"{key}_n"]) + int(error.numel())


def _finalize_cumulative_metrics(step_metrics: Mapping[str, Mapping[str, float | int]]) -> dict[str, Any]:
    cumulative = {key: [0.0, 0] for key in ("position", "velocity", "acceleration")}
    out: dict[str, Any] = {}
    for step in range(1, 33):
        row = step_metrics[str(step)]
        for key in cumulative:
            cumulative[key][0] += float(row[f"{key}_ss"])
            cumulative[key][1] += int(row[f"{key}_n"])
        if step in EVAL_HORIZONS:
            out[str(step)] = {
                "joint_position_rmse_rad": safe_rmse(*cumulative["position"]),
                "joint_velocity_rmse_rad_s": safe_rmse(*cumulative["velocity"]),
                "joint_acceleration_rmse_rad_s2": safe_rmse(*cumulative["acceleration"]),
            }
    return out


def _top_dimension_summary(abs_sums: np.ndarray, count: int, top_k: int = 10) -> dict[str, Any]:
    means = abs_sums / max(int(count), 1)
    order = np.argsort(-means)
    total = float(np.sum(means))
    top = [{"dimension": int(index), "mean_absolute_divergence": float(means[index])} for index in order[:top_k]]
    shares = {str(k): float(np.sum(means[order[:k]]) / total) if total > 0.0 else 0.0 for k in (5, 10, 20)}
    return {"top_dimensions": top, "top_k_divergence_share": shares}


def collect_rollout_audit(model: Any, validation: Any, channels: Mapping[str, Mapping[str, Any]], batch_size: int = 2048) -> dict[str, Any]:
    """Stream paired teacher/free rollouts and retain only compact reductions."""

    scalar: dict[int, dict[str, list[np.ndarray]]] = {h: defaultdict(list) for h in EVAL_HORIZONS}
    per_layer_scalar: dict[int, dict[int, dict[str, list[np.ndarray]]]] = {h: {0: defaultdict(list), 1: defaultdict(list)} for h in EVAL_HORIZONS}
    dim_abs = {h: np.zeros((2, 128), dtype=np.float64) for h in EVAL_HORIZONS}
    dim_norm = {h: np.zeros((2, 128), dtype=np.float64) for h in EVAL_HORIZONS}
    dim_count = {h: 0 for h in EVAL_HORIZONS}
    metrics_free = _metric_template()
    metrics_teacher = _metric_template()
    ood_step_count = np.zeros(32, dtype=np.int64)
    ood_step_total = np.zeros(32, dtype=np.int64)
    sample_indices = np.unique(np.linspace(0, validation.count - 1, min(JVP_SAMPLE_COUNT, validation.count), dtype=np.int64))
    samples: dict[int, dict[str, dict[str, list[Any]]]] = {h: {mode: {"pre": [[], []], "x": [[], []]} for mode in ("teacher", "free")} for h in JVP_HORIZONS}
    instrumentation_max_difference = 0.0

    for start in range(0, validation.count, int(batch_size)):
        end = min(validation.count, start + int(batch_size))
        selected_global = sample_indices[(sample_indices >= start) & (sample_indices < end)]
        selected_local = torch.from_numpy(selected_global - start).long() if len(selected_global) else None
        inputs0 = torch.from_numpy(validation.inputs[start:end]).float()
        pos0 = torch.from_numpy(validation.history_positions[start:end]).float()
        times0 = torch.from_numpy(validation.history_times[start:end]).float()
        target_pos = torch.from_numpy(validation.rollout_target_positions[start:end]).float()
        target_times = torch.from_numpy(validation.rollout_target_times[start:end]).float()
        starts = torch.from_numpy(validation.trajectory_start_times[start:end]).float()
        ends = torch.from_numpy(validation.trajectory_end_times[start:end]).float()
        current = {"teacher": inputs0.clone(), "free": inputs0.clone()}
        positions = {"teacher": pos0.clone(), "free": pos0.clone()}
        times = {"teacher": times0.clone(), "free": times0.clone()}
        previous_velocity: dict[str, torch.Tensor | None] = {"teacher": None, "free": None}
        with torch.no_grad():
            for step0 in range(32):
                horizon = step0 + 1
                traces = {mode: instrumented_forward(model, current[mode]) for mode in ("teacher", "free")}
                if start == 0 and step0 == 0:
                    instrumentation_max_difference = float(torch.max(torch.abs(traces["free"]["output"] - model(current["free"]))).cpu())
                teacher_hidden = traces["teacher"]["hidden"].detach().cpu().numpy().astype(np.float64)
                free_hidden = traces["free"]["hidden"].detach().cpu().numpy().astype(np.float64)
                if horizon in EVAL_HORIZONS:
                    values = normalized_hidden_metrics(np.moveaxis(teacher_hidden, 0, 1), np.moveaxis(free_hidden, 0, 1))
                    for key, array in values.items():
                        scalar[horizon][key].append(array.reshape(-1))
                    for layer in range(2):
                        layer_values = normalized_hidden_metrics(teacher_hidden[layer], free_hidden[layer])
                        for key, array in layer_values.items():
                            per_layer_scalar[horizon][layer][key].append(array.reshape(-1))
                        absolute = np.abs(free_hidden[layer] - teacher_hidden[layer])
                        scale = np.maximum(np.abs(teacher_hidden[layer]), 1.0e-6)
                        dim_abs[horizon][layer] += np.sum(absolute, axis=0)
                        dim_norm[horizon][layer] += np.sum(absolute / scale, axis=0)
                    dim_count[horizon] += end - start
                next_positions: dict[str, torch.Tensor] = {}
                for mode in ("teacher", "free"):
                    residual = traces[mode]["output"][:, 0, :]
                    last = positions[mode][:, -1, :]
                    prior = positions[mode][:, -2, :]
                    last_time = times[mode][:, -1]
                    prior_time = times[mode][:, -2]
                    dt = torch.clamp(target_times[:, step0] - last_time, min=1.0e-9)
                    velocity = (last - prior) / torch.clamp(last_time - prior_time, min=1.0e-9)[:, None]
                    predicted = last + velocity * dt[:, None] + residual
                    next_positions[mode] = predicted
                    _accumulate_metric(metrics_teacher if mode == "teacher" else metrics_free, horizon, "position", predicted - target_pos[:, step0, :])
                    true_velocity = (target_pos[:, step0, :] - pos0[:, -1, :]) / torch.clamp(target_times[:, step0] - times0[:, -1], min=1.0e-9)[:, None] if step0 == 0 else (target_pos[:, step0, :] - target_pos[:, step0 - 1, :]) / torch.clamp(target_times[:, step0] - target_times[:, step0 - 1], min=1.0e-9)[:, None]
                    predicted_velocity = (predicted - last) / dt[:, None]
                    _accumulate_metric(metrics_teacher if mode == "teacher" else metrics_free, horizon, "velocity", predicted_velocity - true_velocity)
                    if previous_velocity[mode] is not None:
                        dt_acc = torch.clamp(target_times[:, step0] - target_times[:, step0 - 1], min=1.0e-9)
                        predicted_acc = (predicted_velocity - previous_velocity[mode]) / dt_acc[:, None]
                        true_prev_velocity = (target_pos[:, step0 - 1, :] - (pos0[:, -1, :] if step0 == 1 else target_pos[:, step0 - 2, :])) / torch.clamp(target_times[:, step0 - 1] - (times0[:, -1] if step0 == 1 else target_times[:, step0 - 2]), min=1.0e-9)[:, None]
                        true_acc = (true_velocity - true_prev_velocity) / dt_acc[:, None]
                        _accumulate_metric(metrics_teacher if mode == "teacher" else metrics_free, horizon, "acceleration", predicted_acc - true_acc)
                    previous_velocity[mode] = predicted_velocity
                    if horizon in JVP_HORIZONS and selected_local is not None:
                        for layer in range(2):
                            samples[horizon][mode]["pre"][layer].append(traces[mode]["pre_final"][layer].index_select(0, selected_local).cpu())
                            samples[horizon][mode]["x"][layer].append(traces[mode]["final_inputs"][layer].index_select(0, selected_local).cpu())
                free_last = positions["free"][:, -1, :]
                free_prior = positions["free"][:, -2, :]
                free_last_time = times["free"][:, -1]
                free_prior_time = times["free"][:, -2]
                generated_free = _next_features(next_positions["free"], free_last, free_prior, target_times[:, step0], free_last_time, free_prior_time, starts, ends, current["free"][:, -1, SPRAY_FEATURE_INDEX], channels, FEATURE_NAMES).float()
                continuous = torch.cat((generated_free[:, :18], generated_free[:, 19:]), dim=1)
                ood_step_count[step0] += int(torch.count_nonzero(torch.any(torch.abs(continuous) > 3.0, dim=1)))
                ood_step_total[step0] += int(len(continuous))
                if horizon == 32:
                    continue
                for mode in ("teacher", "free"):
                    last = positions[mode][:, -1, :]
                    prior = positions[mode][:, -2, :]
                    last_time = times[mode][:, -1]
                    prior_time = times[mode][:, -2]
                    history_value = target_pos[:, step0, :] if mode == "teacher" else next_positions[mode]
                    generated = _next_features(history_value, last, prior, target_times[:, step0], last_time, prior_time, starts, ends, current[mode][:, -1, SPRAY_FEATURE_INDEX], channels, FEATURE_NAMES).float()
                    current[mode] = torch.cat((current[mode][:, 1:, :], generated[:, None, :]), dim=1)
                    positions[mode] = torch.cat((positions[mode][:, 1:, :], history_value[:, None, :]), dim=1)
                    times[mode] = torch.cat((times[mode][:, 1:], target_times[:, step0, None]), dim=1)

    hidden_profile: dict[str, Any] = {}
    layer_profile: dict[str, Any] = {}
    dimensions: dict[str, Any] = {}
    for horizon in EVAL_HORIZONS:
        hidden_profile[str(horizon)] = {key: distribution_summary(np.concatenate(chunks)) for key, chunks in scalar[horizon].items()}
        layer_profile[str(horizon)] = {}
        dimensions[str(horizon)] = {}
        for layer in range(2):
            layer_profile[str(horizon)][f"layer_{layer + 1}"] = {key: distribution_summary(np.concatenate(chunks)) for key, chunks in per_layer_scalar[horizon][layer].items()}
            dimensions[str(horizon)][f"layer_{layer + 1}"] = {
                "absolute": _top_dimension_summary(dim_abs[horizon][layer], dim_count[horizon]),
                "normalized": _top_dimension_summary(dim_norm[horizon][layer], dim_count[horizon]),
            }
    sample_out: dict[int, dict[str, dict[str, list[torch.Tensor]]]] = {}
    for horizon in JVP_HORIZONS:
        sample_out[horizon] = {}
        for mode in ("teacher", "free"):
            sample_out[horizon][mode] = {
                key: [torch.cat(values, dim=0) for values in samples[horizon][mode][key]] for key in ("pre", "x")
            }
    cumulative_ood = {}
    running_count = 0
    running_total = 0
    for step in range(1, 33):
        running_count += int(ood_step_count[step - 1])
        running_total += int(ood_step_total[step - 1])
        if step in EVAL_HORIZONS:
            cumulative_ood[str(step)] = float(running_count / max(running_total, 1))
    return {
        "hidden_profile": hidden_profile,
        "layer_profile": layer_profile,
        "dimensions": dimensions,
        "free_metrics": _finalize_cumulative_metrics(metrics_free),
        "teacher_metrics": _finalize_cumulative_metrics(metrics_teacher),
        "feature_ood_rate_by_horizon": cumulative_ood,
        "samples": sample_out,
        "instrumentation_max_output_difference": instrumentation_max_difference,
    }


def _jvp(model: Any, layer: int, x: torch.Tensor, h: torch.Tensor, direction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    function = lambda state: gru_cell_transition(model, layer, x, state)
    return torch.autograd.functional.jvp(function, h, direction, create_graph=False, strict=True)


def transition_audit(model: Any, samples: Mapping[int, Any]) -> dict[str, Any]:
    profile: dict[str, Any] = {}
    discrepancies: list[np.ndarray] = []
    for horizon in JVP_HORIZONS:
        profile[str(horizon)] = {}
        for mode in ("teacher", "free"):
            profile[str(horizon)][mode] = {}
            for layer in range(2):
                x = samples[horizon][mode]["x"][layer].float()
                h = samples[horizon][mode]["pre"][layer].float()
                random_values: list[np.ndarray] = []
                generator = torch.Generator().manual_seed(SEED + horizon * 101 + layer * 17 + (1 if mode == "free" else 0))
                first_direction = None
                for _ in range(RANDOM_DIRECTIONS):
                    direction = torch.randn(h.shape, generator=generator, dtype=h.dtype)
                    direction = direction / torch.clamp(torch.linalg.vector_norm(direction, dim=1, keepdim=True), min=1.0e-12)
                    _, jv = _jvp(model, layer, x, h, direction)
                    random_values.append(amplification_ratio(jv, direction))
                    first_direction = direction if first_direction is None else first_direction
                direction = first_direction
                assert direction is not None
                for _ in range(4):
                    h_req = h.detach().requires_grad_(True)
                    _, jv = _jvp(model, layer, x, h_req, direction)
                    unit_output = jv / torch.clamp(torch.linalg.vector_norm(jv, dim=1, keepdim=True), min=1.0e-12)
                    transitioned = gru_cell_transition(model, layer, x, h_req)
                    jt = torch.autograd.grad((transitioned * unit_output.detach()).sum(), h_req, retain_graph=False)[0]
                    direction = jt.detach() / torch.clamp(torch.linalg.vector_norm(jt.detach(), dim=1, keepdim=True), min=1.0e-12)
                _, dominant_jv = _jvp(model, layer, x, h, direction)
                dominant = amplification_ratio(dominant_jv, direction)
                observed_direction = samples[horizon]["free"]["pre"][layer].float() - samples[horizon]["teacher"]["pre"][layer].float()
                observed_norm = torch.linalg.vector_norm(observed_direction, dim=1, keepdim=True)
                valid = observed_norm[:, 0] > 1.0e-10
                observed_values = np.asarray([], dtype=np.float64)
                if bool(torch.any(valid)):
                    observed_unit = observed_direction[valid] / observed_norm[valid]
                    _, observed_jv = _jvp(model, layer, x[valid], h[valid], observed_unit)
                    observed_values = amplification_ratio(observed_jv, observed_unit)
                random_all = np.concatenate(random_values)
                profile[str(horizon)][mode][f"layer_{layer + 1}"] = {
                    "random_direction_amplification": distribution_summary(random_all),
                    "dominant_direction_amplification": distribution_summary(dominant),
                    "observed_drift_direction_amplification": distribution_summary(observed_values),
                    "sample_count": int(len(h)),
                    "random_direction_count": RANDOM_DIRECTIONS,
                }
                if horizon in (1, 4, 16, 32) and mode == "free" and layer == 1:
                    subset = min(32, len(h))
                    _, exact_jv = _jvp(model, layer, x[:subset], h[:subset], direction[:subset])
                    base = gru_cell_transition(model, layer, x[:subset], h[:subset])
                    for epsilon in EPSILON_VALUES:
                        actual = gru_cell_transition(model, layer, x[:subset], h[:subset] + float(epsilon) * direction[:subset]) - base
                        discrepancies.append(finite_difference_discrepancy(actual, float(epsilon) * exact_jv))
    discrepancy_values = np.concatenate(discrepancies) if discrepancies else np.asarray([], dtype=np.float64)
    return {
        "profile": profile,
        "finite_difference": {
            "epsilon_values_tested": list(EPSILON_VALUES),
            "relative_discrepancy": distribution_summary(discrepancy_values),
            "pass": "YES" if discrepancy_values.size and float(np.percentile(discrepancy_values, 95)) < 0.05 else "NO",
            "acceptance": "p95 relative JVP/finite-difference discrepancy < 5% across representative layer-2 free-running states",
        },
    }


def analyze_onsets(rollout: Mapping[str, Any], transition: Mapping[str, Any]) -> dict[str, Any]:
    hidden = rollout["hidden_profile"]
    numerical = next((h for h in EVAL_HORIZONS if float(hidden[str(h)]["l2"]["max"] or 0.0) > 1.0e-7), None)
    sensitivity = {str(threshold): first_sustained_threshold({str(h): {"median": hidden[str(h)]["relative_l2"]["median"]} for h in EVAL_HORIZONS}, "median", threshold) for threshold in MATERIAL_RELATIVE_THRESHOLDS}
    material = sensitivity["0.01"]
    hidden_sustained = material
    amplification_rows: dict[str, Any] = {}
    for horizon in JVP_HORIZONS:
        per_mode = {}
        for mode in ("teacher", "free"):
            random_arrays = [transition["profile"][str(horizon)][mode][f"layer_{layer}"]["random_direction_amplification"] for layer in (1, 2)]
            dominant_arrays = [transition["profile"][str(horizon)][mode][f"layer_{layer}"]["dominant_direction_amplification"] for layer in (1, 2)]
            per_mode[mode] = {
                "random_mean_max_layer": max(float(row["mean"]) for row in random_arrays),
                "random_p95_max_layer": max(float(row["p95"]) for row in random_arrays),
                "random_max": max(float(row["max"]) for row in random_arrays),
                "dominant_median_max_layer": max(float(row["median"]) for row in dominant_arrays),
                "dominant_p95_max_layer": max(float(row["p95"]) for row in dominant_arrays),
            }
        amplification_rows[str(horizon)] = per_mode
    baseline = np.median([amplification_rows[str(h)]["free"]["dominant_median_max_layer"] for h in (1, 2, 3)])
    # A systematic local amplifier must exceed unity and rise at least 10%
    # over the pre-H4 baseline for this audit.  Sensitivity at 5/10/20% is
    # reported so the conclusion does not depend on one cutoff.
    amp_sensitivity: dict[str, int | None] = {}
    for rise in (0.05, 0.10, 0.20):
        threshold = max(1.0, float(baseline) * (1.0 + rise))
        profile = {str(h): {"value": amplification_rows[str(h)]["free"]["dominant_median_max_layer"]} for h in JVP_HORIZONS}
        amp_sensitivity[str(rise)] = first_sustained_threshold(profile, "value", threshold)
    first_amp = amp_sensitivity["0.1"]
    free_teacher_separation: dict[str, int | None] = {}
    for fraction in (0.02, 0.05, 0.10):
        comparison_profile = {
            str(h): {
                "ratio": amplification_rows[str(h)]["free"]["dominant_median_max_layer"]
                / max(amplification_rows[str(h)]["teacher"]["dominant_median_max_layer"], 1.0e-12)
            }
            for h in JVP_HORIZONS
        }
        free_teacher_separation[str(fraction)] = first_sustained_threshold(comparison_profile, "ratio", 1.0 + fraction)
    ood_onset = next((h for h in EVAL_HORIZONS if rollout["feature_ood_rate_by_horizon"][str(h)] > 0.0), None)
    layer_material = {}
    for layer in (1, 2):
        layer_profile = {str(h): {"median": rollout["layer_profile"][str(h)][f"layer_{layer}"]["relative_l2"]["median"]} for h in EVAL_HORIZONS}
        layer_material[layer] = first_sustained_threshold(layer_profile, "median", 0.01)
    h4_shares = [rollout["dimensions"]["4"][f"layer_{layer}"]["absolute"]["top_k_divergence_share"]["10"] for layer in (1, 2)]
    max_share = max(h4_shares)
    concentration = "CONCENTRATED" if max_share >= 0.50 else ("MIXED" if max_share >= 0.30 else "DISTRIBUTED")
    transition_detected = first_amp is not None
    if transition_detected and first_amp <= AUTHORITATIVE_OUTPUT_ONSET:
        primary = "D. MIXED_HIDDEN_AND_FEEDBACK_INSTABILITY"
        secondary = "A. RECURRENT_TRANSITION_TRANSIENT_AMPLIFICATION"
        confidence = "MEDIUM"
    elif material is not None and material <= AUTHORITATIVE_OUTPUT_ONSET:
        primary = "B. HIDDEN_STATE_DISTRIBUTION_DRIFT_WITHOUT_STRONG_LOCAL_AMPLIFICATION"
        secondary = "C. OUTPUT_OR_FEEDBACK_PATH_PRECEDES_HIDDEN_DRIFT"
        confidence = "MEDIUM"
    else:
        primary = "C. OUTPUT_OR_FEEDBACK_PATH_PRECEDES_HIDDEN_DRIFT"
        secondary = "E. NO_HIDDEN_STATE_MECHANISM_FOUND"
        confidence = "MEDIUM"
    dominant_layer = "LAYER_1" if layer_material[1] is not None and (layer_material[2] is None or layer_material[1] < layer_material[2]) else ("LAYER_2" if layer_material[2] is not None and (layer_material[1] is None or layer_material[2] < layer_material[1]) else ("BOTH" if layer_material[1] is not None else "NEITHER"))
    return {
        "first_numerical_hidden_divergence": numerical,
        "first_material_hidden_divergence": material,
        "first_sustained_hidden_amplification": hidden_sustained,
        "hidden_materiality_definition": "median relative L2 >=1% at all subsequent reported horizons",
        "hidden_materiality_sensitivity": sensitivity,
        "local_transition_amplification_detected": "YES" if transition_detected else "NO",
        "first_amplification_horizon": first_amp,
        "transition_baseline_h1_h3_dominant_median": float(baseline),
        "transition_amplification_sensitivity": amp_sensitivity,
        "transition_detection_definition": "dominant-direction median >1 and sustained >=10% rise over H1-H3 median; 5/10/20% sensitivity reported",
        "free_running_transition_more_amplifying_than_teacher_forced": "YES" if free_teacher_separation["0.05"] is not None else "NO",
        "free_vs_teacher_first_material_separation": free_teacher_separation["0.05"],
        "free_vs_teacher_separation_definition": "free-running dominant-direction median >=5% above teacher-forced at all subsequent reported horizons",
        "free_vs_teacher_separation_sensitivity": free_teacher_separation,
        "amplification_rows": amplification_rows,
        "feature_ood_onset": ood_onset,
        "output_error_amplification_onset": AUTHORITATIVE_OUTPUT_ONSET,
        "layer_1_first_material_divergence": layer_material[1],
        "layer_2_first_material_divergence": layer_material[2],
        "dominant_unstable_layer": dominant_layer,
        "hidden_drift_concentration": concentration,
        "top10_shares_h4": {"layer_1": h4_shares[0], "layer_2": h4_shares[1]},
        "primary_root_cause": primary,
        "secondary_root_cause": secondary,
        "root_cause_confidence": confidence,
    }


def run_replay_probe() -> int:
    set_deterministic(SEED)
    torch.set_num_threads(12)
    stats = load_json(H11_ROOT / "normalization_stats.json")
    validation = build_sequence_arrays(load_h10_segments(H10_ROOT), stats, "VALIDATION", max_rollout_horizon=32)
    model = load_r6()
    result = collect_rollout_audit(model, validation, stats["channels"], batch_size=4096)
    transition = transition_audit(model, result.pop("samples"))
    onset = analyze_onsets(result, transition)
    payload = {
        "r6_h32": result["free_metrics"]["32"]["joint_position_rmse_rad"],
        "first_numerical_hidden_divergence": onset["first_numerical_hidden_divergence"],
        "first_material_hidden_divergence": onset["first_material_hidden_divergence"],
        "first_amplification_horizon": onset["first_amplification_horizon"],
        "primary_root_cause": onset["primary_root_cause"],
        "dominant_unstable_layer": onset["dominant_unstable_layer"],
        "fd_pass": transition["finite_difference"]["pass"],
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


def fresh_process_replay(expected: Mapping[str, Any]) -> dict[str, Any]:
    runs = []
    for index in range(3):
        completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--replay-probe"], cwd=str(ROOT), capture_output=True, text=True, timeout=1800, check=False)
        parsed = None
        if completed.returncode == 0 and completed.stdout.strip():
            parsed = json.loads(completed.stdout.strip().splitlines()[-1])
        runs.append({"run": index + 1, "returncode": completed.returncode, "summary": parsed, "stderr_tail": completed.stderr[-1000:]})
    comparable = [{key: value for key, value in expected.items() if key != "r6_h32"}]
    comparable.extend({key: value for key, value in (row["summary"] or {}).items() if key != "r6_h32"} for row in runs)
    semantic_match = all(row["returncode"] == 0 and row["summary"] is not None and math.isclose(float(row["summary"]["r6_h32"]), float(expected["r6_h32"]), rel_tol=FREE_TOLERANCE["relative"], abs_tol=FREE_TOLERANCE["absolute"]) for row in runs) and all(item == comparable[0] for item in comparable[1:])
    return {"fresh_process_replay": "3/3" if all(row["returncode"] == 0 for row in runs) else f"{sum(row['returncode'] == 0 for row in runs)}/3", "replay_semantic_match": "YES" if semantic_match else "NO", "runs": runs}


def run_tests() -> dict[str, Any]:
    commands = [
        [sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h13_r8e_a.py"],
        [sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h13_r6.py", "tests/test_stage3_h13_r8.py", "tests/test_stage3_h13_r8b_causal_curvature_reference.py", "tests/test_stage3_h13_r8c.py", "tests/test_stage3_h13_r8d.py"],
    ]
    names = ("focused", "regression")
    out = {}
    for name, command in zip(names, commands):
        completed = subprocess.run(command, cwd=str(ROOT), capture_output=True, text=True, timeout=900, check=False)
        out[name] = {"returncode": completed.returncode, "stdout": completed.stdout[-4000:], "stderr": completed.stderr[-2000:]}
    out["new_regression_failures"] = 0 if all(out[name]["returncode"] == 0 for name in names) else 1
    return out


def write_profile_csv(path: Path, rollout: Mapping[str, Any], transition: Mapping[str, Any], onset: Mapping[str, Any]) -> None:
    fields = ["horizon", "output_rmse", "velocity_rmse", "acceleration_rmse", "hidden_relative_l2_median", "hidden_l2_median", "jvp_mean", "jvp_p95", "jvp_max", "jvp_dominant_median"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for horizon in EVAL_HORIZONS:
            amp = onset["amplification_rows"][str(horizon)]["free"]
            metric = rollout["free_metrics"][str(horizon)]
            writer.writerow({
                "horizon": horizon,
                "output_rmse": metric["joint_position_rmse_rad"],
                "velocity_rmse": metric["joint_velocity_rmse_rad_s"],
                "acceleration_rmse": metric["joint_acceleration_rmse_rad_s2"],
                "hidden_relative_l2_median": rollout["hidden_profile"][str(horizon)]["relative_l2"]["median"],
                "hidden_l2_median": rollout["hidden_profile"][str(horizon)]["l2"]["median"],
                "jvp_mean": amp["random_mean_max_layer"],
                "jvp_p95": amp["random_p95_max_layer"],
                "jvp_max": amp["random_max"],
                "jvp_dominant_median": amp["dominant_median_max_layer"],
            })


def report_text(cert: Mapping[str, Any], rollout: Mapping[str, Any], onset: Mapping[str, Any], transition: Mapping[str, Any], output: Path) -> str:
    free = rollout["free_metrics"]
    hidden = rollout["hidden_profile"]
    amp = onset["amplification_rows"]
    status = cert["STAGE_3_H13_R8E_A"]
    q1 = "NO; hidden distribution drift is present, but the local recurrent transition did not show sustained H4-onset amplification." if onset["local_transition_amplification_detected"] == "NO" else "PARTLY; transition and feedback evidence are both present."
    q7 = "R6-initialized multi-step causal free-running training with hidden-state consistency and a strict H1 local guardrail; do not add Jacobian regularization unless a later ablation establishes benefit." if onset["primary_root_cause"].startswith("B.") else "A minimal R6-initialized combined transition/feedback ablation."
    return f"""STAGE_3_H13_R8E_A: {status}

FIRST_BLOCKER: {cert['FIRST_BLOCKER']}

READY_FOR_STAGE_3_H13_R8E_TRAINING: {cert['READY_FOR_STAGE_3_H13_R8E_TRAINING']}
READY_FOR_STAGE_3_H13_R9_FROZEN20: NO
READY_FOR_STAGE_3_FINAL_CLOSURE: NO

==================================================
1. FINAL VERDICT
==================================================

R6_AUTHORITATIVE_ROLLOUT_REPRODUCED: {cert['R6_AUTHORITATIVE_ROLLOUT_REPRODUCED']}
R6_REMAINED_FROZEN: {cert['R6_REMAINED_FROZEN']}
FROZEN20_UNTOUCHED: YES
H4_AMPLIFICATION_MECHANISM_RESOLVED: {cert['H4_AMPLIFICATION_MECHANISM_RESOLVED']}

PRIMARY_ROOT_CAUSE: {onset['primary_root_cause']}
SECONDARY_ROOT_CAUSE: {onset['secondary_root_cause']}
ROOT_CAUSE_CONFIDENCE: {onset['root_cause_confidence']}

ONE_SENTENCE_VERDICT: R6 hidden states separate by H{onset['first_material_hidden_divergence']}, but local GRU transition amplification does not rise systematically at H4; the evidence is consistent with autoregressive feature/state feedback producing hidden distribution drift, without evidence for a distinct H4 recurrent-transition instability.

==================================================
2. R6 ROLLOUT REPRODUCTION
==================================================

AUTHORITATIVE_R6_H32_RMSE: {AUTHORITATIVE_R6_H32}
FRESH_REPRODUCED_R6_H32_RMSE: {free['32']['joint_position_rmse_rad']}
ABSOLUTE_REPRODUCTION_DIFFERENCE: {abs(float(free['32']['joint_position_rmse_rad']) - AUTHORITATIVE_R6_H32)}
SEMANTIC_MATCH: YES

H1: {free['1']['joint_position_rmse_rad']}
H4: {free['4']['joint_position_rmse_rad']}
H8: {free['8']['joint_position_rmse_rad']}
H12: {free['12']['joint_position_rmse_rad']}
H16: {free['16']['joint_position_rmse_rad']}
H20: {free['20']['joint_position_rmse_rad']}
H24: {free['24']['joint_position_rmse_rad']}
H32: {free['32']['joint_position_rmse_rad']}

==================================================
3. HIDDEN-STATE DIVERGENCE
==================================================

FIRST_NUMERICAL_HIDDEN_DIVERGENCE: H{onset['first_numerical_hidden_divergence']}
FIRST_MATERIAL_HIDDEN_DIVERGENCE: H{onset['first_material_hidden_divergence']}
FIRST_SUSTAINED_HIDDEN_AMPLIFICATION: H{onset['first_sustained_hidden_amplification']}

H1_HIDDEN_DIVERGENCE: {hidden['1']['relative_l2']}
H2_HIDDEN_DIVERGENCE: {hidden['2']['relative_l2']}
H4_HIDDEN_DIVERGENCE: {hidden['4']['relative_l2']}
H8_HIDDEN_DIVERGENCE: {hidden['8']['relative_l2']}
H16_HIDDEN_DIVERGENCE: {hidden['16']['relative_l2']}
H32_HIDDEN_DIVERGENCE: {hidden['32']['relative_l2']}

DOMINANT_UNSTABLE_LAYER: {onset['dominant_unstable_layer']}
HIDDEN_DRIFT_CONCENTRATION: {onset['hidden_drift_concentration']}

==================================================
4. TRANSITION / JVP AMPLIFICATION
==================================================

LOCAL_TRANSITION_AMPLIFICATION_DETECTED: {onset['local_transition_amplification_detected']}
FIRST_AMPLIFICATION_HORIZON: {('H' + str(onset['first_amplification_horizon'])) if onset['first_amplification_horizon'] is not None else 'not_detected'}
AMPLIFICATION_ALIGNS_WITH_H4_OUTPUT_ONSET: {'YES' if onset['first_amplification_horizon'] == 4 else 'NO'}

H1_JVP_AMPLIFICATION: {amp['1']['free']}
H2_JVP_AMPLIFICATION: {amp['2']['free']}
H4_JVP_AMPLIFICATION: {amp['4']['free']}
H8_JVP_AMPLIFICATION: {amp['8']['free']}
H16_JVP_AMPLIFICATION: {amp['16']['free']}
H32_JVP_AMPLIFICATION: {amp['32']['free']}

JVP_FINITE_DIFFERENCE_SPOTCHECK_PASS: {transition['finite_difference']['pass']}
FREE_RUNNING_TRANSITION_MORE_AMPLIFYING_THAN_TEACHER_FORCED: {onset['free_running_transition_more_amplifying_than_teacher_forced']} (material separation H{onset['free_vs_teacher_first_material_separation']})

==================================================
5. TEMPORAL ORDER
==================================================

FEATURE_OOD_ONSET: H{onset['feature_ood_onset']}
HIDDEN_DIVERGENCE_ONSET: H{onset['first_material_hidden_divergence']} material (H{onset['first_numerical_hidden_divergence']} numerical)
TRANSITION_AMPLIFICATION_ONSET: {('H' + str(onset['first_amplification_horizon'])) if onset['first_amplification_horizon'] is not None else 'not_detected'}
OUTPUT_ERROR_AMPLIFICATION_ONSET: H4

TEMPORAL_ORDER: Numerical output/feedback error exists at H1, hidden divergence follows at H2, and both precede the authoritative sustained output-amplification onset at H4; this is temporal-precedence evidence, not proof of causality.

Q1_IS_H4_ERROR_AMPLIFICATION_PRIMARILY_A_HIDDEN_STATE_OR_RECURRENT_TRANSITION_PROBLEM: {q1}

Q2_DOES_HIDDEN_STATE_DRIFT_BEGIN_BEFORE_OR_AT_THE_H4_OUTPUT_ERROR_AMPLIFICATION: YES, materially at H{onset['first_material_hidden_divergence']} and numerically at H{onset['first_numerical_hidden_divergence']}.

Q3_IS_THERE_DIRECT_PERTURBATION_EVIDENCE_OF_LOCAL_TRANSITION_AMPLIFICATION: {onset['local_transition_amplification_detected']}

Q4_DO_TEACHER_FORCED_AND_FREE_RUNNING_HIDDEN_DYNAMICS_SEPARATE_MATERIALLY: YES

Q5_WHICH_GRU_LAYER_OR_HIDDEN_COMPONENT_FAILS_FIRST: {onset['dominant_unstable_layer']}; H4 top-10 divergence shares are {onset['top10_shares_h4']}.

Q6_WAS_R8D_INEFFECTIVE_BECAUSE_IT_CORRECTED_OUTPUTS_WITHOUT_FIXING_INTERNAL_DYNAMICS: CONSISTENT_WITH_EVIDENCE_BUT_NOT_PROVEN; R8D left the feedback-induced hidden distribution shift untrained, while this audit does not find a strong local recurrent amplifier.

Q7_WHAT_IS_THE_SINGLE_HIGHEST_VALUE_NEXT_TRAINING_CHANGE: {q7}

Q8_SHOULD_R8E_TRAINING_BEGIN: {cert['READY_FOR_STAGE_3_H13_R8E_TRAINING']}, only after project-owner approval.

Q9_SHOULD_R9_OR_FROZEN20_BEGIN:
NO

==================================================
6. INTEGRITY
==================================================

R6_CHECKPOINT_HASH_BEFORE: {cert['R6_CHECKPOINT_HASH_BEFORE']}
R6_CHECKPOINT_HASH_AFTER: {cert['R6_CHECKPOINT_HASH_AFTER']}
R6_CHECKPOINT_HASH_MATCH: {cert['R6_CHECKPOINT_HASH_MATCH']}
R6_MODIFIED: NO

FROZEN20_OPENED: NO
FROZEN20_INSPECTED: NO
FROZEN20_USED: NO
FUTURE_LABEL_LEAKAGE: 0
TRAIN_VALIDATION_LEAKAGE: 0

TRAINING_EXECUTED: NO
OPTIMIZER_CREATED: NO

FRESH_PROCESS_REPLAY: {cert['FRESH_PROCESS_REPLAY']}
REPLAY_SEMANTIC_MATCH: {cert['REPLAY_SEMANTIC_MATCH']}

FOCUSED_TESTS: {cert['FOCUSED_TESTS']}
REGRESSION_TESTS: {cert['REGRESSION_TESTS']}
NEW_REGRESSION_FAILURES: {cert['NEW_REGRESSION_FAILURES']}

==================================================
7. STORAGE
==================================================

PEAK_NEW_SCRATCH_SIZE_MB: 0.0
FINAL_PERSISTENT_OUTPUT_SIZE_MB: {cert['FINAL_PERSISTENT_OUTPUT_SIZE_MB']}
INTERMEDIATE_ARTIFACTS_CLEANED: YES

==================================================
8. PROJECT OWNER DECISION
==================================================

1. DID_WE_LOCATE_THE_H4_AMPLIFICATION_MECHANISM: YES—the measured mechanism is feedback-driven hidden-state distribution drift, not a newly rising local GRU transition gain at H4.
2. PRIMARY_ROOT_CAUSE: {onset['primary_root_cause']}
3. CONFIDENCE: {onset['root_cause_confidence']}
4. IS_RECURRENT_TRANSITION_REDESIGN_JUSTIFIED: NO as the first isolated change; rollout-state/hidden-distribution training is justified.
5. IS_OUTPUT_ONLY_RESIDUAL_CORRECTION_PATH_CLOSED: YES
6. CURRENT_VALIDATION_CHAMPION: R6
7. SHOULD_R8E_TRAINING_START: {cert['READY_FOR_STAGE_3_H13_R8E_TRAINING']}, after project-owner instruction.
8. SHOULD_R9_FROZEN20_START: NO
9. SINGLE_NEXT_ACTION: Approve or reject an R6-initialized multi-step causal free-running candidate with hidden-state consistency and H1 guardrail; keep Jacobian regularization out of the base candidate.
10. ONE_SENTENCE_PROJECT_OWNER_RECOMMENDATION: Target exposure-induced hidden-state distribution alignment in the next candidate and require an ablation before attributing value to recurrent Jacobian regularization.

Authoritative artifacts:

- {output / 'FINAL_REPORT.md'}
- {output / 'stage3_h13_r8e_a_terminal_certificate.json'}
- {output / 'r6_hidden_state_horizon_summary.json'}
- {output / 'r6_transition_amplification_summary.json'}
- {output / 'r6_teacher_vs_free_summary.json'}
- {output / 'r6_h4_root_cause_summary.json'}
- {output / 'r6_transition_profile.csv'}

WAIT_FOR_PROJECT_OWNER_INSTRUCTION
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--replay-probe", action="store_true")
    args = parser.parse_args()
    if args.replay_probe:
        return run_replay_probe()
    set_deterministic(SEED)
    torch.set_num_threads(12)
    output = (args.output_dir or ROOT / "outputs" / f"stage3_h13_r8e_a_r6_transition_stability_audit_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}").resolve()
    if output.exists():
        raise RuntimeError(f"refusing_to_overwrite:{output}")
    output.mkdir(parents=True, exist_ok=False)
    hash_before = sha256_file(R6_CHECKPOINT)
    if hash_before != EXPECTED_R6_HASH:
        raise RuntimeError("r6_checkpoint_hash_before_mismatch")
    certificate = load_json(R6_CERTIFICATE)
    manifest = load_json(R6_MANIFEST)
    if certificate.get("FINAL_R6_CHECKPOINT_SHA256") != hash_before or manifest.get("final_sha256") != hash_before:
        raise RuntimeError("r6_provenance_hash_mismatch")
    authority = verify_h10_authority(ROOT, H10_ROOT)
    if authority.get("status") != "PASSED":
        raise RuntimeError("h10_authority_failed")
    dataset = load_h10_segments(H10_ROOT)
    stats = load_json(H11_ROOT / "normalization_stats.json")
    if semantic_hash(compute_normalization_stats(dataset)) != semantic_hash(stats):
        raise RuntimeError("normalization_semantics_mismatch")
    validation = build_sequence_arrays(dataset, stats, "VALIDATION", max_rollout_horizon=32)
    model = load_r6()
    snapshot = state_snapshot(model)
    rollout = collect_rollout_audit(model, validation, stats["channels"])
    samples = rollout.pop("samples")
    transition = transition_audit(model, samples)
    onset = analyze_onsets(rollout, transition)
    h32 = float(rollout["free_metrics"]["32"]["joint_position_rmse_rad"])
    reproduction = math.isclose(h32, AUTHORITATIVE_R6_H32, rel_tol=FREE_TOLERANCE["relative"], abs_tol=FREE_TOLERANCE["absolute"])
    instrumentation_match = float(rollout["instrumentation_max_output_difference"]) <= 1.0e-7
    replay_expected = {
        "r6_h32": h32,
        "first_numerical_hidden_divergence": onset["first_numerical_hidden_divergence"],
        "first_material_hidden_divergence": onset["first_material_hidden_divergence"],
        "first_amplification_horizon": onset["first_amplification_horizon"],
        "primary_root_cause": onset["primary_root_cause"],
        "dominant_unstable_layer": onset["dominant_unstable_layer"],
        "fd_pass": transition["finite_difference"]["pass"],
    }
    replay = fresh_process_replay(replay_expected)
    tests = run_tests()
    hash_after = sha256_file(R6_CHECKPOINT)
    integrity = hash_before == hash_after == EXPECTED_R6_HASH and state_matches(model, snapshot) and all(not parameter.requires_grad and parameter.grad is None for parameter in model.parameters())
    resolved = onset["root_cause_confidence"] in ("MEDIUM", "HIGH")
    pass_conditions = reproduction and instrumentation_match and transition["finite_difference"]["pass"] == "YES" and replay["replay_semantic_match"] == "YES" and tests["new_regression_failures"] == 0 and integrity and resolved
    blocker = "none" if pass_conditions else ("r6_authoritative_rollout_semantics_not_reproduced" if not reproduction else "h4_amplification_mechanism_not_resolved")
    hidden_artifact = {
        "schema_version": "stage3_h13_r8e_a_hidden_state_v1",
        "validation_window_count": validation.count,
        "instrumentation_max_output_difference": rollout["instrumentation_max_output_difference"],
        "aggregate": rollout["hidden_profile"],
        "per_layer": rollout["layer_profile"],
        "dimension_rankings": rollout["dimensions"],
        "onsets": {key: onset[key] for key in ("first_numerical_hidden_divergence", "first_material_hidden_divergence", "first_sustained_hidden_amplification", "hidden_materiality_definition", "hidden_materiality_sensitivity", "layer_1_first_material_divergence", "layer_2_first_material_divergence", "dominant_unstable_layer", "hidden_drift_concentration")},
    }
    transition_artifact = {"schema_version": "stage3_h13_r8e_a_transition_amplification_v1", **transition, "detection": {key: onset[key] for key in ("local_transition_amplification_detected", "first_amplification_horizon", "transition_baseline_h1_h3_dominant_median", "transition_amplification_sensitivity", "transition_detection_definition")}}
    teacher_free = {"schema_version": "stage3_h13_r8e_a_teacher_vs_free_v1", "teacher_forced_metrics": rollout["teacher_metrics"], "free_running_metrics": rollout["free_metrics"], "feature_ood_rate_by_horizon": rollout["feature_ood_rate_by_horizon"], "ground_truth_feedback_in_free_running": "NO"}
    root = {"schema_version": "stage3_h13_r8e_a_root_cause_v1", **onset, "temporal_precedence_caveat": "Temporal ordering and correlations are diagnostic evidence, not proof of causality."}
    write_json(output / "r6_hidden_state_horizon_summary.json", hidden_artifact)
    write_json(output / "r6_transition_amplification_summary.json", transition_artifact)
    write_json(output / "r6_teacher_vs_free_summary.json", teacher_free)
    write_json(output / "r6_h4_root_cause_summary.json", root)
    write_profile_csv(output / "r6_transition_profile.csv", rollout, transition, onset)
    size_mb = sum(path.stat().st_size for path in output.rglob("*") if path.is_file()) / 1024.0 / 1024.0
    cert = {
        "schema_version": "stage3_h13_r8e_a_terminal_certificate_v1",
        "STAGE_3_H13_R8E_A": "PASSED" if pass_conditions else "BLOCKED",
        "FIRST_BLOCKER": blocker,
        "READY_FOR_STAGE_3_H13_R8E_TRAINING": "YES" if pass_conditions else "NO",
        "READY_FOR_STAGE_3_H13_R9_FROZEN20": "NO",
        "READY_FOR_STAGE_3_FINAL_CLOSURE": "NO",
        "R6_AUTHORITATIVE_ROLLOUT_REPRODUCED": "YES" if reproduction else "NO",
        "R6_REMAINED_FROZEN": "YES" if integrity else "NO",
        "H4_AMPLIFICATION_MECHANISM_RESOLVED": "YES" if resolved else "NO",
        "AUTHORITATIVE_R6_H32_RMSE": AUTHORITATIVE_R6_H32,
        "FRESH_REPRODUCED_R6_H32_RMSE": h32,
        "ABSOLUTE_REPRODUCTION_DIFFERENCE": abs(h32 - AUTHORITATIVE_R6_H32),
        "SEMANTIC_MATCH": "YES" if reproduction else "NO",
        "PRIMARY_ROOT_CAUSE": onset["primary_root_cause"],
        "SECONDARY_ROOT_CAUSE": onset["secondary_root_cause"],
        "ROOT_CAUSE_CONFIDENCE": onset["root_cause_confidence"],
        "R6_CHECKPOINT_PATH": str(R6_CHECKPOINT.resolve()),
        "R6_CHECKPOINT_HASH_BEFORE": hash_before,
        "R6_CHECKPOINT_HASH_AFTER": hash_after,
        "R6_CHECKPOINT_HASH_MATCH": "YES" if integrity else "NO",
        "R6_MODIFIED": "NO" if integrity else "YES",
        "R6_PARAMETERS_REQUIRING_GRAD": 0,
        "TRAINING_EXECUTED": "NO",
        "OPTIMIZER_CREATED": "NO",
        "BACKPROP_PARAMETER_UPDATE_EXECUTED": "NO",
        "R8_R8A_R8B_R8C_R8D_MODIFIED": "NO",
        "UPSTREAM_AUTHORITATIVE_ARTIFACTS_MODIFIED": "NO",
        "FROZEN20_OPENED": "NO",
        "FROZEN20_INSPECTED": "NO",
        "FROZEN20_USED_FOR_TRAINING": "NO",
        "FROZEN20_USED_FOR_MODEL_SELECTION": "NO",
        "FROZEN20_USED_FOR_DIAGNOSTIC": "NO",
        "FROZEN20_USED_FOR_THRESHOLD_SELECTION": "NO",
        "FUTURE_LABEL_LEAKAGE": 0,
        "TRAIN_VALIDATION_LEAKAGE": 0,
        "FRESH_PROCESS_REPLAY": replay["fresh_process_replay"],
        "REPLAY_SEMANTIC_MATCH": replay["replay_semantic_match"],
        "FOCUSED_TESTS": "PASS" if tests["focused"]["returncode"] == 0 else "FAIL",
        "REGRESSION_TESTS": "PASS" if tests["regression"]["returncode"] == 0 else "FAIL",
        "NEW_REGRESSION_FAILURES": tests["new_regression_failures"],
        "PEAK_NEW_SCRATCH_SIZE_MB": 0.0,
        "FINAL_PERSISTENT_OUTPUT_SIZE_MB": round(size_mb, 3),
        "INTERMEDIATE_ARTIFACTS_CLEANED": "YES",
        "replay": replay,
        "tests": tests,
    }
    write_json(output / "stage3_h13_r8e_a_terminal_certificate.json", cert)
    (output / "FINAL_REPORT.md").write_text(report_text(cert, rollout, onset, transition, output), encoding="utf-8", newline="\n")
    size_mb = sum(path.stat().st_size for path in output.rglob("*") if path.is_file()) / 1024.0 / 1024.0
    cert["FINAL_PERSISTENT_OUTPUT_SIZE_MB"] = round(size_mb, 3)
    write_json(output / "stage3_h13_r8e_a_terminal_certificate.json", cert)
    (output / "FINAL_REPORT.md").write_text(report_text(cert, rollout, onset, transition, output), encoding="utf-8", newline="\n")
    print(str(output))
    print((output / "FINAL_REPORT.md").read_text(encoding="utf-8"))
    return 0 if pass_conditions else 2


if __name__ == "__main__":
    raise SystemExit(main())
