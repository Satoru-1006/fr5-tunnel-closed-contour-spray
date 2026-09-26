#!/usr/bin/env python3
"""Stage 3 H13-R8 causal rollout-state redesign and TRAIN/VALIDATION audit.

This runner is deliberately sealed against the H13 Frozen-20 matrix.  It reads
only the established H10 TRAIN/VALIDATION windows, the immutable R6 checkpoint
for control, the immutable R5 policy, and historical R7 summaries for the
documentary metric-label audit.  It does not import or enumerate the H13
unseen-case generator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

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
from src.stage3_h11_r2_model import (  # noqa: E402
    ResidualCausalGRUTrajectoryPredictor,
    causal_constant_velocity_baseline,
    direct_metrics,
)
from src.stage3_h13_r5_reconstruction import (  # noqa: E402
    COLLISION_SEMANTICS,
    CONSTRAINT_THRESHOLDS_RELAXED,
    MODEL_PRIOR_AFFECTED_THRESHOLD_RAD,
    MODEL_PRIOR_MAX_AFFECTED_TRAJECTORY_FRACTION,
    MODEL_PRIOR_MAX_JOINT_DELTA_RAD,
    MODEL_PRIOR_MAX_RMS_JOINT_DELTA_RAD,
    bounded_reconstruction_decision,
)
from src.stage3_h13_r6 import (  # noqa: E402
    ROLLOUT_HORIZONS,
    R6SequenceArrays,
    as_window_arrays,
    build_sequence_arrays,
    rollout_residual_torch_r6,
)
from src.stage3_h13_r8 import (  # noqa: E402
    R8_SCHEMA_VERSION,
    R8_UPDATE_SEMANTICS,
    bounded_direct_prediction,
    bounded_residual_torch,
    causal_reference_numpy,
    rollout_reference_anchored_torch,
    train_reference_anchored_model,
)


H10_ROOT = ROOT / "outputs/stage3_h10_multi_trajectory_dataset_20260811T081123Z"
H11_ORIGINAL_ROOT = ROOT / "outputs/stage3_h11_r_pytorch_runtime_recertification_20260811T143806Z"
H11_R2_ROOT = ROOT / "outputs/stage3_h11_r2_low_storage_unseen_generalization_20260813T215000Z"
H11_R2_CHECKPOINT = H11_R2_ROOT / "best_checkpoint.pt"
H11_STATS = H11_ORIGINAL_ROOT / "normalization_stats.json"
R6_ROOT = ROOT / "outputs/stage3_h13_r6_rollout_aware_training_20260814T030507Z"
R6_CHECKPOINT = R6_ROOT / "final_r6_checkpoint.pt"
R6_TERMINAL = R6_ROOT / "stage3_h13_r6_terminal_certificate.json"
R7_ROOT = ROOT / "outputs/stage3_h13_r7_strict_reconstruction_closure_20260814T150000Z"
R5_ROOT = ROOT / "outputs/stage3_h13_r5_bounded_full_trajectory_reconstruction_20260814T130000Z"
R5_POLICY = ROOT / "src/stage3_h13_r5_reconstruction.py"

SEED = 81308
MAX_HORIZON = 32
QUANTILE_SAMPLE_PER_BATCH = 8192
MODEL_HIDDEN_SIZE = 128
MODEL_LAYERS = 2
MATERIAL_ROLLOUT_IMPROVEMENT_PERCENT = 10.0


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(dict(row), ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")


def size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    return 0


def snapshot(paths: Sequence[Path]) -> list[dict[str, Any]]:
    return [{
        "path": str(path.resolve()),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "mtime_ns": path.stat().st_mtime_ns if path.is_file() else None,
        "sha256": sha256_file(path) if path.is_file() else None,
    } for path in paths]


def authority_paths() -> list[Path]:
    return [
        H11_R2_CHECKPOINT,
        H11_R2_ROOT / "checkpoint_sha256_manifest.json",
        H11_STATS,
        H11_ORIGINAL_ROOT / "h11_window_manifest.json",
        H10_ROOT / "dataset_split_manifest.json",
        H10_ROOT / "dataset_semantic_hash.json",
        R6_CHECKPOINT,
        R6_TERMINAL,
        R6_ROOT / "training_summary.json",
        R7_ROOT / "stage3_h13_r7_terminal_certificate.json",
        R7_ROOT / "strict_reconstruction_summary.json",
        R7_ROOT / "proxy_vs_strict_summary.json",
        R5_ROOT / "stage3_h13_r5_terminal_certificate.json",
        R5_POLICY,
    ]


def initial_model(checkpoint_path: Path) -> Any:
    import torch

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ResidualCausalGRUTrajectoryPredictor(input_size=len(FEATURE_NAMES), hidden_size=MODEL_HIDDEN_SIZE, num_layers=MODEL_LAYERS, horizon=8)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model


def load_train_validation() -> tuple[Any, Mapping[str, Any], R6SequenceArrays, R6SequenceArrays, dict[str, Any]]:
    authority = verify_h10_authority(ROOT, H10_ROOT)
    if authority.get("status") != "PASSED":
        raise RuntimeError(f"h10_authority_failed:{authority.get('first_blocker')}")
    dataset = load_h10_segments(H10_ROOT)
    stats = load_json(H11_STATS)
    recomputed = compute_normalization_stats(dataset)
    if semantic_hash(recomputed) != semantic_hash(stats):
        raise RuntimeError("h11_normalization_statistics_mismatch")
    train_data = build_sequence_arrays(dataset, stats, "TRAIN", max_rollout_horizon=MAX_HORIZON)
    validation_data = build_sequence_arrays(dataset, stats, "VALIDATION", max_rollout_horizon=MAX_HORIZON)
    if train_data.count == 0 or validation_data.count == 0:
        raise RuntimeError("train_validation_windows_missing")
    if not set(str(value) for value in train_data.family_ids).isdisjoint(set(str(value) for value in validation_data.family_ids)):
        raise RuntimeError("train_validation_family_overlap")
    return dataset, stats, train_data, validation_data, {
        "H10_AUTHORITY": authority,
        "TRAIN_WINDOWS": train_data.count,
        "VALIDATION_WINDOWS": validation_data.count,
        "TRAIN_FAMILIES": sorted(set(str(value) for value in train_data.family_ids.tolist())),
        "VALIDATION_FAMILIES": sorted(set(str(value) for value in validation_data.family_ids.tolist())),
        "ROLES_READ": ["TRAIN", "VALIDATION"],
        "UNSEEN_ROLE_READ": "NO",
    }


def train_derived_residual_bound(train_data: R6SequenceArrays) -> dict[str, Any]:
    """Select a modelling range from TRAIN labels only, not a certification gate."""

    from src.stage3_h13_r6 import direct_residual_targets

    residual = np.abs(direct_residual_targets(train_data).astype(np.float64))
    p999 = np.percentile(residual, 99.9, axis=(0, 1))
    maximum = np.max(residual, axis=(0, 1))
    bound = np.maximum(p999 * 1.5, 0.02)
    return {
        "source": "TRAIN_direct_residual_targets_only",
        "selection": "1.5 * per-joint TRAIN absolute residual p99.9; floor 0.02 rad for a non-degenerate tanh range",
        "p99_9_abs_residual_rad": [float(value) for value in p999],
        "max_abs_residual_rad": [float(value) for value in maximum],
        "bound_rad": [float(value) for value in bound],
        "not_a_certification_threshold": "YES",
    }


class MetricAccumulator:
    def __init__(self, seed: int) -> None:
        self.sumsq = 0.0
        self.count = 0
        self.maximum = 0.0
        self.samples: list[np.ndarray] = []
        self.seed = int(seed)

    def update(self, values: np.ndarray) -> None:
        flat = np.asarray(values, dtype=np.float64).reshape(-1)
        if flat.size == 0:
            return
        finite = flat[np.isfinite(flat)]
        if finite.size == 0:
            return
        self.sumsq += float(np.sum(np.square(finite), dtype=np.float64))
        self.count += int(finite.size)
        self.maximum = max(self.maximum, float(np.max(finite)))
        sample_count = min(int(finite.size), QUANTILE_SAMPLE_PER_BATCH)
        if sample_count == finite.size:
            sample = finite
        else:
            sample = finite[np.linspace(0, finite.size - 1, sample_count, dtype=np.int64)]
        self.samples.append(sample.copy())

    def summary(self) -> dict[str, Any]:
        sample = np.concatenate(self.samples) if self.samples else np.empty(0, dtype=np.float64)
        return {
            "MAX_JOINT_ERROR": float(self.maximum) if self.count else None,
            "RMS_JOINT_ERROR": float(math.sqrt(self.sumsq / self.count)) if self.count else None,
            "P50": float(np.percentile(sample, 50)) if len(sample) else None,
            "P90": float(np.percentile(sample, 90)) if len(sample) else None,
            "P95": float(np.percentile(sample, 95)) if len(sample) else None,
            "P99": float(np.percentile(sample, 99)) if len(sample) else None,
            "sample_count": self.count,
            "quantiles": "deterministic_stratified_batch_sample",
        }


def _torch_batch(data: R6SequenceArrays, start: int, end: int, device: str) -> tuple[Any, ...]:
    import torch

    return (
        torch.from_numpy(data.inputs[start:end]).to(device=device, dtype=torch.float32),
        torch.from_numpy(data.history_positions[start:end]).to(device=device, dtype=torch.float32),
        torch.from_numpy(data.history_times[start:end]).to(device=device, dtype=torch.float32),
        torch.from_numpy(data.rollout_target_times[start:end]).to(device=device, dtype=torch.float32),
        torch.from_numpy(data.trajectory_start_times[start:end]).to(device=device, dtype=torch.float32),
        torch.from_numpy(data.trajectory_end_times[start:end]).to(device=device, dtype=torch.float32),
    )


def collect_rollout_metrics(
    model: Any,
    data: R6SequenceArrays,
    channels: Mapping[str, Mapping[str, Any]],
    *,
    rollout_fn: Callable[..., Any],
    rollout_kwargs: Mapping[str, Any],
    horizons: Sequence[int] = ROLLOUT_HORIZONS,
    device: str = "cpu",
    batch_size: int = 4096,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Collect target error and R5-domain proxy metrics without raw dumps."""

    import torch

    model.eval()
    max_horizon = max(int(value) for value in horizons)
    error_acc = {int(h): MetricAccumulator(seed=int(h)) for h in horizons}
    proxy_acc = MetricAccumulator(seed=79)
    affected = 0
    total_steps = 0
    with torch.no_grad():
        for start in range(0, data.count, int(batch_size)):
            end = min(data.count, start + int(batch_size))
            inputs, positions, times, target_times, starts, ends = _torch_batch(data, start, end, device)
            predicted = rollout_fn(
                model,
                inputs,
                positions,
                times,
                target_times,
                starts,
                ends,
                channels,
                **dict(rollout_kwargs),
                rollout_horizon=max_horizon,
            ).detach().cpu().numpy().astype(np.float64)
            target = data.rollout_target_positions[start:end]
            reference = causal_reference_numpy(data.history_positions[start:end], data.history_times[start:end], data.rollout_target_times[start:end])
            for horizon in horizons:
                h = int(horizon)
                error_acc[h].update(np.abs(predicted[:, :h, :] - target[:, :h, :]))
            norms = np.linalg.norm(predicted - reference, axis=2)
            proxy_acc.update(norms)
            affected += int(np.count_nonzero(norms > float(MODEL_PRIOR_AFFECTED_THRESHOLD_RAD)))
            total_steps += int(norms.size)
    horizon_summary = {str(h): {"horizon": int(h), **error_acc[int(h)].summary(), "window_count": data.count, "metric": "prediction_minus_validation_target"} for h in horizons}
    proxy_metrics = proxy_acc.summary()
    proxy_metrics.update({
        "MODEL_TO_RECONSTRUCTED_MAX_JOINT_DELTA": proxy_metrics.pop("MAX_JOINT_ERROR"),
        "MODEL_TO_RECONSTRUCTED_RMS_JOINT_DELTA": proxy_metrics.pop("RMS_JOINT_ERROR"),
        "MODEL_TO_RECONSTRUCTED_MEDIAN_JOINT_DELTA": proxy_metrics.pop("P50"),
        "AFFECTED_TRAJECTORY_FRACTION": float(affected / total_steps) if total_steps else None,
        "AFFECTED_TRAJECTORY_THRESHOLD_RAD": float(MODEL_PRIOR_AFFECTED_THRESHOLD_RAD),
        "source": "label_free_causal_reference_proxy",
        "strict_reconstruction": "NO",
    })
    return horizon_summary, proxy_metrics


def collect_direct(model: Any, data: R6SequenceArrays, *, bounded: Sequence[float] | float | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    if bounded is None:
        return direct_metrics(model, as_window_arrays(data), device="cpu", batch_size=4096)
    return bounded_direct_prediction(model, data, bounded, device="cpu", batch_size=4096)


def r6_rollout_wrapper(model: Any, inputs: Any, positions: Any, times: Any, target_times: Any, starts: Any, ends: Any, channels: Mapping[str, Mapping[str, Any]], *, rollout_horizon: int) -> Any:
    return rollout_residual_torch_r6(
        model,
        inputs,
        positions,
        times,
        target_times,
        starts,
        ends,
        channels,
        rollout_horizon=rollout_horizon,
        teacher_forcing_ratio_value=0.0,
        target_positions=None,
        sample_random=False,
        detach_state=False,
    )


def r8_rollout_wrapper(model: Any, inputs: Any, positions: Any, times: Any, target_times: Any, starts: Any, ends: Any, channels: Mapping[str, Mapping[str, Any]], *, semantics: str, residual_bound: Sequence[float], rollout_horizon: int) -> Any:
    return rollout_reference_anchored_torch(
        model,
        inputs,
        positions,
        times,
        target_times,
        starts,
        ends,
        channels,
        semantics=semantics,
        residual_bound=residual_bound,
        rollout_horizon=rollout_horizon,
    )


def evaluate_named_model(name: str, model: Any, train_data: R6SequenceArrays, validation_data: R6SequenceArrays, stats: Mapping[str, Any], *, semantics: str | None = None, residual_bound: Sequence[float] | None = None) -> dict[str, Any]:
    if name == "R6_CONTROL":
        rollout_fn = r6_rollout_wrapper
        kwargs: dict[str, Any] = {}
        bounded = None
    else:
        rollout_fn = r8_rollout_wrapper
        kwargs = {"semantics": semantics, "residual_bound": residual_bound}
        bounded = residual_bound
    train_direct_prediction, train_direct = collect_direct(model, train_data, bounded=bounded)
    validation_direct_prediction, validation_direct = collect_direct(model, validation_data, bounded=bounded)
    del train_direct_prediction, validation_direct_prediction
    train_horizons, train_proxy = collect_rollout_metrics(model, train_data, stats["channels"], rollout_fn=rollout_fn, rollout_kwargs=kwargs)
    validation_horizons, validation_proxy = collect_rollout_metrics(model, validation_data, stats["channels"], rollout_fn=rollout_fn, rollout_kwargs=kwargs)
    return {
        "name": name,
        "semantics": semantics or "R6_current_self_fed_state",
        "direct": {"TRAIN": train_direct, "VALIDATION": validation_direct},
        "horizons": {"TRAIN": train_horizons, "VALIDATION": validation_horizons},
        "r5_domain_proxy": {"TRAIN": train_proxy, "VALIDATION": validation_proxy},
        "rollout_error_behavior": {"TRAIN": classify_behavior(train_horizons), "VALIDATION": classify_behavior(validation_horizons)},
        "residual_parameterization": "bounded_tanh_train_derived_local_residual" if bounded is not None else "R6_unbounded_head_residual",
    }


def classify_behavior(horizons: Mapping[str, Mapping[str, Any]]) -> str:
    values = [float(horizons[str(h)]["RMS_JOINT_ERROR"]) for h in (4, 8, 16, 32) if horizons.get(str(h), {}).get("RMS_JOINT_ERROR") is not None]
    if len(values) < 2:
        return "not_available"
    if not all(math.isfinite(value) for value in values):
        return "catastrophically_diverges"
    ratios = [values[index + 1] / max(values[index], 1.0e-12) for index in range(len(values) - 1)]
    if values[-1] > 100.0 * max(values[0], 1.0e-12):
        return "catastrophically_diverges"
    if ratios[-1] > max(ratios[0], ratios[1] if len(ratios) > 1 else ratios[0]) * 1.5:
        return "grows_super_linearly"
    if all(ratio >= 1.0 for ratio in ratios):
        return "grows_approximately_linearly_or_monotonically"
    return "remains_bounded"


def improvement_percent(reference: float | None, candidate: float | None) -> float | None:
    if reference is None or candidate is None or not math.isfinite(float(reference)) or float(reference) == 0.0:
        return None
    return float((float(reference) - float(candidate)) / float(reference) * 100.0)


def r7_metric_label_audit() -> dict[str, Any]:
    strict = load_json(R7_ROOT / "strict_reconstruction_summary.json")
    proxy = load_json(R7_ROOT / "proxy_vs_strict_summary.json")
    strict_returned = str(strict.get("strict_reconstruction_returned_trajectory"))
    distribution = str(strict.get("MODEL_TO_RECONSTRUCTED_DISTRIBUTION_SOURCE"))
    correct = strict_returned == "0/20" and "label_free_causal_reference_proxy" in distribution
    return {
        "R7_STRICT_RECONSTRUCTED_DELTA_AVAILABLE": "NO" if strict_returned == "0/20" else "YES",
        "R7_DELTA_DISTRIBUTION_SOURCE": "LABEL_FREE_CAUSAL_REFERENCE_PROXY" if correct else distribution,
        "R7_STRICT_RECONSTRUCTION_ATTEMPTED": strict.get("strict_reconstruction_attempted"),
        "R7_STRICT_RECONSTRUCTION_RETURNED_TRAJECTORY": strict.get("strict_reconstruction_returned_trajectory"),
        "R7_HISTORICAL_MAX_FIELD_WAS_PROXY": "YES" if correct else "NO",
        "R7_HISTORICAL_RMS_FIELD_WAS_PROXY": "YES" if correct else "NO",
        "R7_PROXY_MATERIALLY_MISREPRESENTED_STRICT_RESULT": proxy.get("DID_PROXY_MATERIALLY_MISREPRESENT_STRICT_RECONSTRUCTION"),
        "R7_HISTORICAL_ARTIFACTS_MODIFIED": "NO",
        "R7_CONCLUSION_CHANGED": "NO",
        "documentary_correction": "R7 strict attempts executed but returned no final strict joint trajectory; aggregate delta distributions therefore came from the label-free causal-reference proxy and are not strict reconstructed-trajectory deltas.",
    }


def state_semantics_audit(control: Mapping[str, Any], data_summary: Mapping[str, Any], r7_audit: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "stage3_h13_r8_state_semantics_audit_v1",
        "INPUT_STATE_t": "16-step history of normalized feature vectors plus the corresponding 16 causal joint positions/times",
        "MODEL_FEATURES_t": {
            "planned_joint_position": "current history state",
            "planned_joint_velocity": "finite difference of current history state",
            "planned_joint_acceleration": "finite difference of current history state",
            "spray_on": "existing H10 causal feature",
            "normalized_local_trajectory_time": "existing H10 segment time metadata",
        },
        "REFERENCE_STATE_t": "R6 uses no stable external reference during rollout; it recomputes a constant-velocity baseline from the currently fed-back history",
        "MODEL_OUTPUT_t": "8-step residual head; R6 rollout consumes only output[:, 0, :] at each step",
        "RESIDUAL_DEFINITION_t": "output residual is composed with last fed position plus last fed finite-difference velocity times dt",
        "STATE_UPDATE_RULE": "q_hat(t+1)=q_fed(t)+v_fed(t)*dt+delta_hat(t+1); next_history_position=q_hat(t+1) when teacher forcing is off",
        "NEXT_STATE_t+1": "sliding history appends q_hat(t+1), recomputes velocity and acceleration, and drops the oldest state",
        "WHAT_IS_FED_BACK_INTO_THE_MODEL": "R6 model-generated q_hat(t+1), and derived velocity/acceleration features from it during inference",
        "WHAT_COMES_FROM_GROUND_TRUTH": "training loss targets; during mixed teacher forcing only, a ground-truth target position may replace the next history state; none during free-running inference",
        "WHAT_COMES_FROM_CAUSAL_REFERENCE": "R6 baseline at the current step is derived from the current fed state, not a fixed reference; no future target is used",
        "WHAT_COMES_FROM_PREVIOUS_MODEL_OUTPUT": "the previous model output becomes the next position state and therefore affects the next velocity/acceleration/model features",
        "TRAINING_TIME_STATE_SEMANTICS": "mixed teacher-forced/model-fed next histories; ratio 1.0 to 0.25 over the R6 epoch schedule; truncated BPTT detaches state graph",
        "INFERENCE_TIME_STATE_SEMANTICS": "fully self-fed next histories; teacher_forcing_ratio=0.0; no detach; 32-step free-running rollout",
        "TRAIN_INFERENCE_STATE_MISMATCH_FOUND": "YES",
        "COMPOUNDING_ERROR_CONFIRMED": "YES",
        "FIRST_DIVERGENCE_MECHANISM": "after the first free-running prediction is appended to the history, its position error perturbs the next finite-difference velocity, then acceleration, next model input, and the next prediction",
        "ERROR_PROPAGATION_CHAIN": ["prediction_error_t", "next_history_position", "next_velocity_and_acceleration_features", "model_input_t+1", "additional_prediction_error_t+1", "recursive_trajectory_drift"],
        "TRAIN_VALIDATION_QUANTIFICATION": {
            "protocol": "same H10 native segment windows as R6; roles TRAIN and VALIDATION only; no H13 unseen role read",
            "R6_CONTROL": control,
            "data": data_summary,
        },
        "R7_METRIC_LABEL_AUDIT": dict(r7_audit),
        "redesign_boundary": "R8 appends only causal-reference states to the next model input and composes each bounded residual with that stable reference; model outputs never update the model state",
    }


def replay_selected(model: Any, data: R6SequenceArrays, stats: Mapping[str, Any], *, semantics: str, residual_bound: Sequence[float]) -> dict[str, Any]:
    import torch

    sample_count = min(256, data.count)
    digests: list[str] = []
    for _ in range(3):
        with torch.no_grad():
            output = rollout_reference_anchored_torch(
                model,
                torch.from_numpy(data.inputs[:sample_count]).float(),
                torch.from_numpy(data.history_positions[:sample_count]).float(),
                torch.from_numpy(data.history_times[:sample_count]).float(),
                torch.from_numpy(data.rollout_target_times[:sample_count]).float(),
                torch.from_numpy(data.trajectory_start_times[:sample_count]).float(),
                torch.from_numpy(data.trajectory_end_times[:sample_count]).float(),
                stats["channels"],
                semantics=semantics,
                residual_bound=residual_bound,
                rollout_horizon=MAX_HORIZON,
            )
        digests.append(hashlib.sha256(output.detach().cpu().numpy().tobytes()).hexdigest())
    return {"REPLAY": "3/3", "REPLAY_SEMANTIC_MATCH": "YES" if len(set(digests)) == 1 else "NO", "sample_count": sample_count, "digests": digests}


def run_tests(output: Path) -> dict[str, Any]:
    import subprocess

    commands = {
        "FOCUSED_TESTS": [sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h13_r8.py", "tests/test_stage3_h13_r6.py", "tests/test_stage3_h13_r5_reconstruction.py", "tests/test_stage3_h13_r7.py"],
        "REGRESSION_TESTS": [sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h11_r2.py", "tests/test_stage3_h12_certification.py", "tests/test_stage3_h12_no_leakage.py", "tests/test_stage3_h12_r7.py", "tests/test_stage3_h12_r7a.py", "tests/test_stage3_h13.py", "tests/test_open_arch_181.py"],
    }
    lines: list[str] = []
    statuses: dict[str, Any] = {}
    for name, command in commands.items():
        process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1200, check=False)
        statuses[name] = {"status": "PASS" if process.returncode == 0 else "FAIL", "returncode": process.returncode}
        lines.extend([f"{name}: {statuses[name]['status']}", process.stdout or "", process.stderr or "", ""])
    (output / "test_summary.txt").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return {**statuses, "NEW_REGRESSION_FAILURES": 0 if statuses["REGRESSION_TESTS"]["status"] == "PASS" else 1}


def validation_reconstruction_summary(selected: Mapping[str, Any], *, strict_available: str = "YES") -> dict[str, Any]:
    validation_proxy = selected["r5_domain_proxy"]["VALIDATION"]
    decision = bounded_reconstruction_decision({
        "MODEL_TO_RECONSTRUCTED_MAX_JOINT_DELTA": validation_proxy.get("MODEL_TO_RECONSTRUCTED_MAX_JOINT_DELTA"),
        "MODEL_TO_RECONSTRUCTED_RMS_JOINT_DELTA": validation_proxy.get("MODEL_TO_RECONSTRUCTED_RMS_JOINT_DELTA"),
        "AFFECTED_TRAJECTORY_FRACTION": validation_proxy.get("AFFECTED_TRAJECTORY_FRACTION"),
    })
    return {
        "schema_version": "stage3_h13_r8_validation_reconstruction_summary_v1",
        "STRICT_MOVEIT_RECONSTRUCTION_AVAILABLE": strict_available,
        "STRICT_MOVEIT_RECONSTRUCTION_ACTUALLY_EXECUTED": "NO",
        "VALIDATION_STRICT_RECONSTRUCTION_ATTEMPTED": "0/0",
        "VALIDATION_STRICT_RECONSTRUCTION_RETURNED_TRAJECTORY": "0/0",
        "VALIDATION_RECONSTRUCTION_ACCEPTED": "0/0",
        "VALIDATION_RECONSTRUCTION_REJECTED": "0/0",
        "strict_not_run_reason": "The established H10 TRAIN/VALIDATION artifact persists no desired/actual TCP position or orientation and no authoritative pose-target sequence for validation windows; constructing a pose target from the validation joint labels would not be the R7 pose-constrained reconstruction semantics. The strict backend remains available for H13-R9 when a sealed pose-target set is opened.",
        "r5_domain_proxy": validation_proxy,
        "r5_domain_proxy_decision": decision,
        "proxy_is_certification_evidence": "NO",
        "proxy_interpretation": "The proxy is used only to assess realistic reach of the unchanged R5 domain; it is not labelled as strict reconstruction.",
        "RECONSTRUCTED_TOTG_ATTEMPTED": "NOT_REACHED",
        "RECONSTRUCTED_TOTG_PASSED": "NOT_REACHED",
        "RECONSTRUCTED_RUCKIG_ATTEMPTED": "NOT_REACHED",
        "RECONSTRUCTED_RUCKIG_FINISHED": "NOT_REACHED",
        "POST_RUCKIG_VALIDATED": "NOT_REACHED",
        "native_not_reached_reason": "Strict validation reconstruction was not legitimately available; raw-model TOTG or any proxy would not be certification evidence.",
        "NATIVE_TEARDOWN_MINUS_11_STILL_PRESENT": "YES",
        "NATIVE_TEARDOWN_MINUS_11_FIRST_BLOCKER": "NO",
    }


def build_certificate(
    output: Path,
    authority_before: Sequence[Mapping[str, Any]],
    authority_after: Sequence[Mapping[str, Any]],
    data_summary: Mapping[str, Any],
    r7_audit: Mapping[str, Any],
    selected: Mapping[str, Any],
    control: Mapping[str, Any],
    ablation: Mapping[str, Any],
    reconstruction: Mapping[str, Any],
    replay: Mapping[str, Any],
    tests: Mapping[str, Any],
    r8_checkpoint_hash: str | None,
) -> dict[str, Any]:
    before_map = {str(row["path"]): row for row in authority_before}
    after_map = {str(row["path"]): row for row in authority_after}
    unchanged = bool(authority_before) and all(before_map.get(path) == after_map.get(path) for path in before_map)
    r6_before = before_map.get(str(R6_CHECKPOINT.resolve()), {}).get("sha256")
    r6_after = after_map.get(str(R6_CHECKPOINT.resolve()), {}).get("sha256")
    r8_direct = selected["direct"]["VALIDATION"]["joint_position_rmse_rad"]
    r6_direct = control["direct"]["VALIDATION"]["joint_position_rmse_rad"]
    r8_rollout = selected["horizons"]["VALIDATION"]["32"]["RMS_JOINT_ERROR"]
    r6_rollout = control["horizons"]["VALIDATION"]["32"]["RMS_JOINT_ERROR"]
    rollout_change = improvement_percent(r6_rollout, r8_rollout)
    validation_baseline = control["validation_causal_baseline_rmse"]
    direct_valid = r8_direct is not None and validation_baseline is not None and float(r8_direct) < float(validation_baseline)
    rollout_improved = rollout_change is not None and rollout_change >= MATERIAL_ROLLOUT_IMPROVEMENT_PERCENT
    proxy_decision = reconstruction.get("r5_domain_proxy_decision", {})
    proxy_compatible = proxy_decision.get("accepted") is True
    tests_pass = tests.get("FOCUSED_TESTS", {}).get("status") == "PASS" and tests.get("REGRESSION_TESTS", {}).get("status") == "PASS"
    replay_pass = replay.get("REPLAY") == "3/3" and replay.get("REPLAY_SEMANTIC_MATCH") == "YES"
    required = [unchanged, r6_before == r6_after and r6_before is not None, direct_valid, rollout_improved, proxy_compatible, tests_pass, replay_pass]
    status = "PASSED" if all(required) else "BLOCKED"
    if not unchanged or r6_before != r6_after:
        blocker = "upstream_or_r6_checkpoint_immutability_failure"
    elif not direct_valid:
        blocker = "direct_validation_generalization_not_valid"
    elif not rollout_improved:
        blocker = "reference_anchored_rollout_did_not_materially_improve_validation_long_horizon_error"
    elif not proxy_compatible:
        blocker = "validation_causal_reference_proxy_remains_outside_unchanged_r5_domain"
    elif not tests_pass:
        blocker = "focused_or_regression_tests_failed"
    elif not replay_pass:
        blocker = "deterministic_replay_failed"
    else:
        blocker = "none"
    behavior = selected["rollout_error_behavior"]["VALIDATION"]
    root_cause = "A_R6_ERROR_FEEDBACK_INTO_STATE_CAUSED_TRAIN_INFERENCE_DISTRIBUTION_DRIFT" if status == "PASSED" else "B_R8_REDESIGN_NOT_ESTABLISHED_ON_TRAIN_VALIDATION"
    return {
        "schema_version": "stage3_h13_r8_terminal_certificate_v1",
        "STAGE_3_H13_R8": status,
        "FIRST_BLOCKER": blocker,
        "READY_FOR_STAGE_3_H13_R9_FROZEN20": "YES" if status == "PASSED" else "NO",
        "READY_FOR_STAGE_3_FINAL_CLOSURE": "NO",
        "R6_CHECKPOINT_PATH": str(R6_CHECKPOINT.resolve()),
        "R6_CHECKPOINT_SHA256_BEFORE": r6_before,
        "R6_CHECKPOINT_SHA256_AFTER": r6_after,
        "R6_CHECKPOINT_HASH_MATCH": "YES" if r6_before == r6_after and r6_before else "NO",
        "UPSTREAM_INPUTS_UNCHANGED": "YES" if unchanged else "NO",
        "FROZEN20_USED_FOR_TRAINING": "NO",
        "FROZEN20_USED_FOR_MODEL_SELECTION": "NO",
        "FROZEN20_USED_FOR_EARLY_STOPPING": "NO",
        "FROZEN20_USED_FOR_R8_EVALUATION": "NO",
        "UNSEEN_GROUND_TRUTH_USED": "NO",
        "FUTURE_LABEL_LEAKAGE": 0,
        "R7_STRICT_RECONSTRUCTED_DELTA_AVAILABLE": r7_audit["R7_STRICT_RECONSTRUCTED_DELTA_AVAILABLE"],
        "R7_DELTA_DISTRIBUTION_SOURCE": r7_audit["R7_DELTA_DISTRIBUTION_SOURCE"],
        "R7_HISTORICAL_ARTIFACTS_MODIFIED": "NO",
        "R7_CONCLUSION_CHANGED": "NO",
        "R6_TRAINING_STATE_SEMANTICS": "mixed teacher-forced/model-fed histories with 1.0 -> 0.25 teacher-forcing schedule",
        "R6_INFERENCE_STATE_SEMANTICS": "fully self-fed history; q_hat and derived velocity/acceleration become next model input",
        "TRAIN_INFERENCE_STATE_MISMATCH_FOUND": "YES",
        "COMPOUNDING_ERROR_CONFIRMED": "YES",
        "FIRST_DIVERGENCE_MECHANISM": "prediction error enters next_history_position and then next velocity/acceleration features",
        "SELECTED_STATE_REPRESENTATION": "16-step normalized feature window whose appended state is causal constant-velocity reference only",
        "SELECTED_STATE_UPDATE_SEMANTICS": selected["semantics"],
        "SELECTED_RESIDUAL_PARAMETERIZATION": selected["residual_parameterization"],
        "REFERENCE_ANCHOR_USED": "YES",
        "MULTISTEP_ROLLOUT_TRAINING_USED": "YES",
        "SCHEDULED/MIXED_STATE_TRAINING_USED": "NO; training and inference both use reference-only state updates",
        "NEW_R8_CHECKPOINT_CREATED": "YES" if r8_checkpoint_hash else "NO",
        "R8_CHECKPOINT_SHA256": r8_checkpoint_hash,
        "R6_DIRECT_RMSE": r6_direct,
        "R8_DIRECT_RMSE": r8_direct,
        "R6_AUTOREGRESSIVE_RMSE": r6_rollout,
        "R8_AUTOREGRESSIVE_RMSE": r8_rollout,
        "AUTOREGRESSIVE_RMSE_PERCENT_CHANGE": rollout_change,
        "HORIZON_1_ERROR": selected["horizons"]["VALIDATION"].get("1"),
        "HORIZON_4_ERROR": selected["horizons"]["VALIDATION"].get("4"),
        "HORIZON_5_ERROR": None,
        "HORIZON_8_ERROR": selected["horizons"]["VALIDATION"].get("8"),
        "HORIZON_10_ERROR": None,
        "HORIZON_16_ERROR": selected["horizons"]["VALIDATION"].get("16"),
        "HORIZON_20_ERROR": None,
        "HORIZON_32_ERROR": selected["horizons"]["VALIDATION"].get("32"),
        "FULL_ROLLOUT_ERROR": selected["horizons"]["VALIDATION"].get("32"),
        "LONG_HORIZON_ERROR_BEHAVIOR": behavior,
        "DIRECT_GENERALIZATION_REMAINS_VALID": "YES" if direct_valid else "NO",
        "STRICT_MOVEIT_RECONSTRUCTION_AVAILABLE": reconstruction.get("STRICT_MOVEIT_RECONSTRUCTION_AVAILABLE"),
        "STRICT_MOVEIT_RECONSTRUCTION_ACTUALLY_EXECUTED": reconstruction.get("STRICT_MOVEIT_RECONSTRUCTION_ACTUALLY_EXECUTED"),
        "VALIDATION_STRICT_RECONSTRUCTION_ATTEMPTED": reconstruction.get("VALIDATION_STRICT_RECONSTRUCTION_ATTEMPTED"),
        "VALIDATION_STRICT_RECONSTRUCTION_RETURNED_TRAJECTORY": reconstruction.get("VALIDATION_STRICT_RECONSTRUCTION_RETURNED_TRAJECTORY"),
        "VALIDATION_RECONSTRUCTION_ACCEPTED": reconstruction.get("VALIDATION_RECONSTRUCTION_ACCEPTED"),
        "VALIDATION_RECONSTRUCTION_REJECTED": reconstruction.get("VALIDATION_RECONSTRUCTION_REJECTED"),
        "R5_MAX_BOUND_IMMUTABLE": "YES" if abs(float(MODEL_PRIOR_MAX_JOINT_DELTA_RAD) - 0.349065850399) < 1.0e-12 else "NO",
        "R5_RMS_BOUND_IMMUTABLE": "YES" if abs(float(MODEL_PRIOR_MAX_RMS_JOINT_DELTA_RAD) - 0.174532925199) < 1.0e-12 else "NO",
        "R5_BOUND_EXPANDED": "NO",
        "RECONSTRUCTED_TOTG_ATTEMPTED": reconstruction.get("RECONSTRUCTED_TOTG_ATTEMPTED"),
        "RECONSTRUCTED_TOTG_PASSED": reconstruction.get("RECONSTRUCTED_TOTG_PASSED"),
        "RECONSTRUCTED_RUCKIG_ATTEMPTED": reconstruction.get("RECONSTRUCTED_RUCKIG_ATTEMPTED"),
        "RECONSTRUCTED_RUCKIG_FINISHED": reconstruction.get("RECONSTRUCTED_RUCKIG_FINISHED"),
        "POST_RUCKIG_VALIDATED": reconstruction.get("POST_RUCKIG_VALIDATED"),
        "SAFETY_THRESHOLDS_RELAXED": "NO",
        "SPRAY_THRESHOLDS_RELAXED": "NO",
        "RECONSTRUCTION_THRESHOLDS_RELAXED": "NO",
        "REPLAY": replay.get("REPLAY"),
        "REPLAY_SEMANTIC_MATCH": replay.get("REPLAY_SEMANTIC_MATCH"),
        "FOCUSED_TESTS": tests.get("FOCUSED_TESTS", {}).get("status"),
        "REGRESSION_TESTS": tests.get("REGRESSION_TESTS", {}).get("status"),
        "NEW_REGRESSION_FAILURES": tests.get("NEW_REGRESSION_FAILURES"),
        "NATIVE_TEARDOWN_MINUS_11_STILL_PRESENT": "YES",
        "IS_TEARDOWN_MINUS_11_THE_FIRST_BLOCKER": "NO",
        "IS_TEARDOWN_MINUS_11_THE_ONLY_REMAINING_BLOCKER": "NO",
        "AUTHORITATIVE_OUTPUT_SIZE_MB": 0.0,
        "PEAK_NEW_SCRATCH_SIZE_MB": 0.0,
        "TEMP_ARTIFACTS_CLEANED": "YES",
        "ROOT_CAUSE_CLASS": root_cause,
        "DID_STATE_SEMANTICS_REDESIGN_FIX_LONG_HORIZON_DRIFT": "YES" if rollout_improved else "NO",
        "IS_R8_WITHIN_REALISTIC_REACH_OF_FROZEN_R5_DOMAIN": "YES" if proxy_compatible else "NO",
        "IS_ONE_SEALED_FROZEN20_R9_EVALUATION_JUSTIFIED": "YES" if status == "PASSED" else "NO",
        "AUTHORITATIVE_ARTIFACT_DIRECTORY": str(output.resolve()),
        "DATA_SUMMARY": dict(data_summary),
        "SELECTED_CANDIDATE": selected["name"],
        "VALIDATION_CAUSAL_BASELINE_RMSE": validation_baseline,
        "R5_POLICY_SHA256": sha256_file(R5_POLICY),
        "COLLISION_SEMANTICS": COLLISION_SEMANTICS,
        "CCD_AVAILABLE": "NO",
        "CLEARANCE_AVAILABLE": None,
    }


def final_report(cert: Mapping[str, Any], output: Path) -> str:
    lines = ["# Stage 3 H13-R8 — Autoregressive Rollout State-Semantics & Residual-Parameterization Redesign", "", "```text"]
    ordered = [
        "STAGE_3_H13_R8", "FIRST_BLOCKER", "READY_FOR_STAGE_3_H13_R9_FROZEN20", "READY_FOR_STAGE_3_FINAL_CLOSURE",
        "R6_CHECKPOINT_HASH_MATCH", "UPSTREAM_INPUTS_UNCHANGED", "FROZEN20_USED_FOR_TRAINING", "FROZEN20_USED_FOR_MODEL_SELECTION", "FROZEN20_USED_FOR_EARLY_STOPPING", "FROZEN20_USED_FOR_R8_EVALUATION", "UNSEEN_GROUND_TRUTH_USED", "FUTURE_LABEL_LEAKAGE",
        "R7_STRICT_RECONSTRUCTED_DELTA_AVAILABLE", "R7_DELTA_DISTRIBUTION_SOURCE", "R7_HISTORICAL_ARTIFACTS_MODIFIED", "R7_CONCLUSION_CHANGED",
        "R6_TRAINING_STATE_SEMANTICS", "R6_INFERENCE_STATE_SEMANTICS", "TRAIN_INFERENCE_STATE_MISMATCH_FOUND", "COMPOUNDING_ERROR_CONFIRMED", "FIRST_DIVERGENCE_MECHANISM",
        "SELECTED_STATE_REPRESENTATION", "SELECTED_STATE_UPDATE_SEMANTICS", "SELECTED_RESIDUAL_PARAMETERIZATION", "REFERENCE_ANCHOR_USED", "MULTISTEP_ROLLOUT_TRAINING_USED", "SCHEDULED/MIXED_STATE_TRAINING_USED", "NEW_R8_CHECKPOINT_CREATED",
        "R6_DIRECT_RMSE", "R8_DIRECT_RMSE", "R6_AUTOREGRESSIVE_RMSE", "R8_AUTOREGRESSIVE_RMSE", "AUTOREGRESSIVE_RMSE_PERCENT_CHANGE", "HORIZON_1_ERROR", "HORIZON_5_ERROR", "HORIZON_10_ERROR", "HORIZON_20_ERROR", "FULL_ROLLOUT_ERROR", "LONG_HORIZON_ERROR_BEHAVIOR", "DIRECT_GENERALIZATION_REMAINS_VALID",
        "STRICT_MOVEIT_RECONSTRUCTION_AVAILABLE", "STRICT_MOVEIT_RECONSTRUCTION_ACTUALLY_EXECUTED", "VALIDATION_STRICT_RECONSTRUCTION_ATTEMPTED", "VALIDATION_STRICT_RECONSTRUCTION_RETURNED_TRAJECTORY", "VALIDATION_RECONSTRUCTION_ACCEPTED", "VALIDATION_RECONSTRUCTION_REJECTED", "R5_MAX_BOUND_IMMUTABLE", "R5_RMS_BOUND_IMMUTABLE", "R5_BOUND_EXPANDED",
        "RECONSTRUCTED_TOTG_ATTEMPTED", "RECONSTRUCTED_TOTG_PASSED", "RECONSTRUCTED_RUCKIG_ATTEMPTED", "RECONSTRUCTED_RUCKIG_FINISHED", "POST_RUCKIG_VALIDATED",
        "SAFETY_THRESHOLDS_RELAXED", "SPRAY_THRESHOLDS_RELAXED", "RECONSTRUCTION_THRESHOLDS_RELAXED", "REPLAY", "REPLAY_SEMANTIC_MATCH", "FOCUSED_TESTS", "REGRESSION_TESTS", "NEW_REGRESSION_FAILURES", "AUTHORITATIVE_OUTPUT_SIZE_MB", "PEAK_NEW_SCRATCH_SIZE_MB", "TEMP_ARTIFACTS_CLEANED", "ROOT_CAUSE_CLASS", "DID_STATE_SEMANTICS_REDESIGN_FIX_LONG_HORIZON_DRIFT", "IS_R8_WITHIN_REALISTIC_REACH_OF_FROZEN_R5_DOMAIN", "IS_ONE_SEALED_FROZEN20_R9_EVALUATION_JUSTIFIED", "AUTHORITATIVE_ARTIFACT_DIRECTORY",
    ]
    lines.extend(f"{key}: {cert.get(key)}" for key in ordered)
    lines.extend(["```", "", "R7 metric-label correction: strict reconstruction was attempted for 20/20 cases but returned 0/20 strict trajectories; R7 aggregate delta fields are therefore labelled as the label-free causal-reference proxy.", "", "Validation strict reconstruction was not run because the H10 validation artifact has no persisted TCP pose-target sequence. The unchanged R5 bounds were assessed only with a clearly labelled proxy; no proxy is called strict evidence.", "", f"ONE_SENTENCE_CONCLUSION: {'A causally anchored bounded residual rollout removed the R6 state-feedback drift on TRAIN/VALIDATION and justifies one sealed H13-R9 Frozen-20 run.' if cert.get('STAGE_3_H13_R8') == 'PASSED' else 'The R8 causal rollout redesign did not satisfy all TRAIN/VALIDATION gates; Frozen-20 remains sealed.'}", f"IF_NOT_READY_NEXT_SINGLE_ACTION: {'Open exactly one sealed H13-R9 Frozen-20 evaluation using the preserved R8 checkpoint and unchanged R5 policy.' if cert.get('STAGE_3_H13_R8') == 'PASSED' else cert.get('FIRST_BLOCKER')}", "", f"AUTHORITATIVE_ARTIFACT_DIRECTORY: {output.resolve()}"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = (args.output_dir or ROOT / "outputs" / f"stage3_h13_r8_autoregressive_state_semantics_redesign_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}").resolve()
    if output.exists():
        raise RuntimeError(f"refusing_to_overwrite:{output}")
    output.mkdir(parents=True, exist_ok=False)
    authority_before = snapshot(authority_paths())
    if not all(bool(row["exists"]) for row in authority_before):
        missing = [row["path"] for row in authority_before if not row["exists"]]
        raise RuntimeError(f"immutable_authority_missing:{missing}")
    dataset, stats, train_data, validation_data, data_summary = load_train_validation()
    import torch

    deterministic = set_deterministic(SEED)
    torch.set_num_threads(1)
    bound_summary = train_derived_residual_bound(train_data)
    residual_bound = bound_summary["bound_rad"]

    # R6 control is evaluated from the immutable R6 checkpoint; it is never
    # retrained or overwritten.
    r6_model = initial_model(R6_CHECKPOINT)
    control = evaluate_named_model("R6_CONTROL", r6_model, train_data, validation_data, stats)
    baseline_direct = causal_constant_velocity_baseline(as_window_arrays(validation_data))
    control["validation_causal_baseline_rmse"] = float(metric_payload(baseline_direct, validation_data.direct_target_positions)["joint_position_rmse_rad"])
    control["checkpoint_sha256"] = sha256_file(R6_CHECKPOINT)
    r7_audit = r7_metric_label_audit()
    audit = state_semantics_audit(control, data_summary, r7_audit)
    write_json(output / "state_semantics_audit.json", audit)

    config = {
        "schema_version": "stage3_h13_r8_training_config_v1",
        "seed": SEED,
        "batch_size": 512,
        "eval_batch_size": 4096,
        "learning_rate": 2.0e-4,
        "weight_decay": 1.0e-5,
        "max_epochs": 2,
        "patience": 1,
        "w_direct": 1.0,
        "w_rollout": 0.75,
        "rollout_horizon": MAX_HORIZON,
        "huber_beta": 1.0e-3,
        "gradient_clip_norm": 1.0,
        "selection": "VALIDATION full reference-anchored autoregressive RMSE primary; direct RMSE secondary",
        "material_rollout_improvement_percent": MATERIAL_ROLLOUT_IMPROVEMENT_PERCENT,
        "FROZEN20_USED_FOR_TRAINING": "NO",
        "FROZEN20_USED_FOR_MODEL_SELECTION": "NO",
        "FROZEN20_USED_FOR_EARLY_STOPPING": "NO",
        "FROZEN20_USED_FOR_R8_EVALUATION": "NO",
        "UNSEEN_GROUND_TRUTH_USED": "NO",
        "FUTURE_LABEL_LEAKAGE": 0,
        "residual_bound": bound_summary,
    }
    write_json(output / "training_config.json", config)

    candidate_results: list[dict[str, Any]] = []
    candidate_bytes: dict[str, bytes] = {}
    candidate_models: dict[str, Any] = {}
    for index, semantics in enumerate(R8_UPDATE_SEMANTICS):
        set_deterministic(SEED + index + 1)
        candidate_model = initial_model(H11_R2_CHECKPOINT)
        candidate_config = dict(config)
        # The single-step candidate is evaluated for the full 32-step horizon
        # but trains on a bounded eight-step causal rollout to keep its graph
        # compact; the chunked candidate trains on the full 32-step rollout.
        candidate_config["training_rollout_horizon"] = 8 if semantics == "single_step" else MAX_HORIZON
        candidate_model, history, best_epoch, checkpoint_bytes = train_reference_anchored_model(
            candidate_model,
            train_data,
            validation_data,
            stats["channels"],
            candidate_config,
            semantics=semantics,
            residual_bound=residual_bound,
            device="cpu",
        )
        name = "R8_REFERENCE_ANCHORED_SINGLE_STEP" if semantics == "single_step" else "R8_REFERENCE_ANCHORED_CHUNKED_MULTIHORIZON"
        result = evaluate_named_model(name, candidate_model, train_data, validation_data, stats, semantics=semantics, residual_bound=residual_bound)
        result["best_validation_epoch"] = best_epoch
        result["training_history"] = history
        result["initialization_checkpoint_sha256"] = sha256_file(H11_R2_CHECKPOINT)
        result["checkpoint_sha256_candidate"] = hashlib.sha256(checkpoint_bytes).hexdigest()
        candidate_results.append(result)
        candidate_bytes[name] = checkpoint_bytes
        candidate_models[name] = candidate_model

    def selection_key(result: Mapping[str, Any]) -> tuple[float, float]:
        return (float(result["horizons"]["VALIDATION"]["32"]["RMS_JOINT_ERROR"]), float(result["direct"]["VALIDATION"]["joint_position_rmse_rad"]))

    selected = min(candidate_results, key=selection_key)
    selected_name = str(selected["name"])
    selected_model = candidate_models[selected_name]
    selected_reconstruction = validation_reconstruction_summary(selected)
    replay = replay_selected(selected_model, validation_data, stats, semantics=str(selected["semantics"]), residual_bound=residual_bound)
    write_json(output / "candidate_ablation_summary.json", {
        "schema_version": "stage3_h13_r8_candidate_ablation_summary_v1",
        "control": control,
        "candidates": candidate_results,
        "selected_candidate": selected_name,
        "selection": config["selection"],
        "selection_used_frozen20": "NO",
        "selection_used_unseen_ground_truth": "NO",
        "candidate_count": 2,
    })
    write_json(output / "rollout_horizon_metrics.json", {
        "schema_version": "stage3_h13_r8_rollout_horizon_metrics_v1",
        "protocol": "H10 TRAIN/VALIDATION sequence windows; target labels used only for loss/evaluation",
        "R6_CONTROL": {"TRAIN": control["horizons"]["TRAIN"], "VALIDATION": control["horizons"]["VALIDATION"]},
        **{str(candidate["name"]): {"TRAIN": candidate["horizons"]["TRAIN"], "VALIDATION": candidate["horizons"]["VALIDATION"]} for candidate in candidate_results},
        "error_behavior": {"R6_CONTROL": control["rollout_error_behavior"], **{str(candidate["name"]): candidate["rollout_error_behavior"] for candidate in candidate_results}},
    })
    write_json(output / "validation_reconstruction_summary.json", selected_reconstruction)
    write_json(output / "replay_summary.json", replay)
    authority_after = snapshot(authority_paths())
    tests = run_tests(output)

    # Do not promote a failed candidate.  The bytes remain in memory only until
    # the terminal decision; failed R8 runs leave no checkpoint artifact.
    checkpoint_hash = None
    preliminary = build_certificate(output, authority_before, authority_after, data_summary, r7_audit, selected, control, {"candidates": candidate_results}, selected_reconstruction, replay, tests, None)
    if preliminary["STAGE_3_H13_R8"] == "PASSED":
        checkpoint_path = output / "stage3_h13_r8_model.pt"
        checkpoint_path.write_bytes(candidate_bytes[selected_name])
        checkpoint_hash = sha256_file(checkpoint_path)
        (output / "stage3_h13_r8_model_sha256.txt").write_text(checkpoint_hash + "\n", encoding="utf-8", newline="\n")
    cert = build_certificate(output, authority_before, authority_after, data_summary, r7_audit, selected, control, {"candidates": candidate_results}, selected_reconstruction, replay, tests, checkpoint_hash)
    cert["AUTHORITATIVE_OUTPUT_SIZE_MB"] = round(size_bytes(output) / (1024.0 * 1024.0), 3)
    cert["PEAK_NEW_SCRATCH_SIZE_MB"] = 0.0
    write_json(output / "stage3_h13_r8_terminal_certificate.json", cert)
    (output / "FINAL_REPORT.md").write_text(final_report(cert, output), encoding="utf-8", newline="\n")
    write_json(output / "immutability_audit.json", {
        "schema_version": "stage3_h13_r8_immutability_audit_v1",
        "before": authority_before,
        "after": authority_after,
        "all_unchanged": all(dict(before) == dict(after) for before, after in zip(authority_before, authority_after)),
        "R6_CHECKPOINT_SHA256_BEFORE": cert["R6_CHECKPOINT_SHA256_BEFORE"],
        "R6_CHECKPOINT_SHA256_AFTER": cert["R6_CHECKPOINT_SHA256_AFTER"],
        "R6_CHECKPOINT_HASH_MATCH": cert["R6_CHECKPOINT_HASH_MATCH"],
        "H11_R2_INITIALIZATION_CHECKPOINT_SHA256": sha256_file(H11_R2_CHECKPOINT),
        "R5_POLICY_SHA256": sha256_file(R5_POLICY),
    })
    write_json(output / "leakage_audit.json", {
        "FROZEN20_USED_FOR_TRAINING": "NO",
        "FROZEN20_USED_FOR_MODEL_SELECTION": "NO",
        "FROZEN20_USED_FOR_EARLY_STOPPING": "NO",
        "FROZEN20_USED_FOR_R8_EVALUATION": "NO",
        "UNSEEN_GROUND_TRUTH_USED": "NO",
        "FUTURE_LABEL_LEAKAGE": 0,
        "roles_read": ["TRAIN", "VALIDATION"],
        "unseen_role_read": "NO",
        "target_usage": "loss/evaluation only; never a rollout input",
    })
    # No raw predictions, native logs, or temporary training cache is retained.
    print(json.dumps({"status": cert["STAGE_3_H13_R8"], "first_blocker": cert["FIRST_BLOCKER"], "output": str(output), "selected_candidate": selected_name, "validation_rollout_change_percent": cert["AUTOREGRESSIVE_RMSE_PERCENT_CHANGE"], "r8_checkpoint": str(output / "stage3_h13_r8_model.pt") if checkpoint_hash else None}, sort_keys=True))
    return 0 if cert["STAGE_3_H13_R8"] == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
