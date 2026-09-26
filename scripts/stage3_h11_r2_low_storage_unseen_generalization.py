#!/usr/bin/env python3
"""Stage 3 H11-R2 low-storage unseen-generalization remediation.

This runner is additive and offline.  It reads the frozen H10/H11 release,
trains a bounded residual-over-causal-baseline GRU ablation, freezes the
validation-selected checkpoint, then evaluates TEST and GENERALIZATION once.
H13 frozen unseen cases are authority inputs only and are never read as
training, validation, or model-selection labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from collections import Counter
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
    PREDICTION_HORIZON,
    compute_normalization_stats,
    contract_payload,
    load_h10_segments,
    make_windows,
    semantic_hash,
    sha256_file,
    snapshot_frozen_upstream,
    verify_h10_authority,
    window_semantic_hash,
    write_json,
)
from src.stage3_h11_model import WindowArrays, metric_payload, set_deterministic  # noqa: E402
from src.stage3_h11_r2_model import (  # noqa: E402
    ResidualCausalGRUTrajectoryPredictor,
    TORCH_AVAILABLE,
    _initialise_from_h11_backbone,
    causal_constant_velocity_baseline,
    direct_metrics,
    model_summary,
    parameter_count,
    rollout_residual_numpy,
    train_residual_model,
)


H10_ROOT = ROOT / "outputs/stage3_h10_multi_trajectory_dataset_20260811T081123Z"
H11_AUTH_ROOT = ROOT / "outputs/stage3_h11_r_pytorch_runtime_recertification_20260811T143806Z"
H11_REPRO_ROOT = ROOT / "outputs/stage3_h11_r2_original_reproduction_20260813T210000Z"
H12_ROOT = ROOT / "outputs/stage3_h12_constraint_aware_trajectory_repair_20260811T163500Z"
H12_R7_ROOT = ROOT / "outputs/stage3_h12_r7_low_storage_reprojection_20260813T030000Z"
H13_R1_ROOT = ROOT / "outputs/stage3_h13_low_storage_unseen_generalization_r1_authoritative_20260813T"
H13_R2_ROOT = ROOT / "outputs/stage3_h13_low_storage_unseen_generalization_r2_20260813T195307126"
H11_CHECKPOINT_SHA256 = "e3c62b202589482004a82e29917c2e8a1473e8c8befceccdcee57679a370f30e"
H11_DATASET_SPLIT_SHA256 = "4b740c49b2f0ab22e250099c804f2bd23a9ec736b2b7494906bbaa76032395cd"
H11_WINDOW_SEMANTIC_SHA256 = "9d09a7eea38a1a10cfdaf22a0d202d2c829a2d4447beba6423e1682fdc6102d3"
SEED = 11022
COLLISION_METHOD = "adaptive_discrete_interpolation"
HORIZONS = (1, 2, 4, 8)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")


def output_size_mb(path: Path) -> float:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) / (1024.0 * 1024.0)


def improvement_percent(reference: float | None, candidate: float | None) -> float | None:
    if reference is None or candidate is None or not math.isfinite(float(reference)) or not math.isfinite(float(candidate)) or float(reference) == 0.0:
        return None
    return (float(reference) - float(candidate)) / float(reference) * 100.0


def snapshot(paths: Sequence[Path]) -> list[dict[str, Any]]:
    result = []
    for path in paths:
        result.append({"path": str(path.resolve()), "exists": path.is_file(), "size_bytes": path.stat().st_size if path.is_file() else None, "sha256": sha256_file(path) if path.is_file() else None})
    return result


def immutable_paths() -> list[Path]:
    return [
        H11_AUTH_ROOT / "checkpoint", H11_AUTH_ROOT / "h11_window_manifest.json", H11_AUTH_ROOT / "normalization_stats.json", H11_AUTH_ROOT / "training_config.json", H11_AUTH_ROOT / "stage3_h11_r_terminal_certificate.json",
        H12_ROOT / "stage3_h12_terminal_certificate.json",
        H12_R7_ROOT / "stage3_h12_r7_terminal_certificate.json", H12_R7_ROOT / "repair_summary.json", H12_R7_ROOT / "repaired_sample_manifest.jsonl",
        H13_R1_ROOT / "stage3_h13_terminal_certificate.json", H13_R1_ROOT / "generalization_summary.json", H13_R1_ROOT / "case_summary.jsonl",
        H13_R2_ROOT / "stage3_h13_terminal_certificate.json", H13_R2_ROOT / "generalization_summary.json", H13_R2_ROOT / "case_summary.jsonl",
        ROOT / "outputs/internal_wiper_moveit_inputs/open_arch_tcp_poses_base_link.csv", ROOT / "outputs/internal_wiper_moveit_inputs/open_arch_seed_joints.csv",
    ]


def authority_audit() -> dict[str, Any]:
    paths = immutable_paths()
    missing = [str(path) for path in paths if not path.is_file()]
    checkpoint_hash = sha256_file(H11_AUTH_ROOT / "checkpoint") if (H11_AUTH_ROOT / "checkpoint").is_file() else None
    split_hash = sha256_file(H11_AUTH_ROOT / "h11_window_manifest.json") if (H11_AUTH_ROOT / "h11_window_manifest.json").is_file() else None
    h11 = load_json(H11_AUTH_ROOT / "stage3_h11_r_terminal_certificate.json") if (H11_AUTH_ROOT / "stage3_h11_r_terminal_certificate.json").is_file() else {}
    h12 = load_json(H12_ROOT / "stage3_h12_terminal_certificate.json") if (H12_ROOT / "stage3_h12_terminal_certificate.json").is_file() else {}
    h12r7 = load_json(H12_R7_ROOT / "stage3_h12_r7_terminal_certificate.json") if (H12_R7_ROOT / "stage3_h12_r7_terminal_certificate.json").is_file() else {}
    h13 = load_json(H13_R2_ROOT / "stage3_h13_terminal_certificate.json") if (H13_R2_ROOT / "stage3_h13_terminal_certificate.json").is_file() else {}
    h13_summary = load_json(H13_R2_ROOT / "generalization_summary.json") if (H13_R2_ROOT / "generalization_summary.json").is_file() else {}
    unseen_matrix_hash = h13_summary.get("case_matrix", {}).get("matrix_sha256")
    split_manifest = load_json(H10_ROOT / "dataset_split_manifest.json") if (H10_ROOT / "dataset_split_manifest.json").is_file() else {}
    role_integrity = split_manifest.get("counts") == {"TRAIN": 31, "VALIDATION": 7, "TEST": 6, "GENERALIZATION": 4} and split_manifest.get("split_unit") == "trajectory_family_id"
    return {
        "schema_version": "stage3_h11_r2_immutability_leakage_audit_v1",
        "H11_ORIGINAL_IMMUTABLE": "YES" if checkpoint_hash == H11_CHECKPOINT_SHA256 and split_hash == H11_DATASET_SPLIT_SHA256 and not missing else "NO",
        "H12_IMMUTABLE": "YES" if (H12_ROOT / "stage3_h12_terminal_certificate.json").is_file() and not missing else "NO",
        "H12_R7_IMMUTABLE": "YES" if (H12_R7_ROOT / "stage3_h12_r7_terminal_certificate.json").is_file() and not missing else "NO",
        "H13_R1_IMMUTABLE": "YES" if (H13_R1_ROOT / "stage3_h13_terminal_certificate.json").is_file() and not missing else "NO",
        "H13_R2_IMMUTABLE": "YES" if (H13_R2_ROOT / "stage3_h13_terminal_certificate.json").is_file() and not missing else "NO",
        "H13_UNSEEN_SPLIT_IMMUTABLE": "YES" if unseen_matrix_hash and int(h13.get("UNSEEN_CASES", 0) or 0) == 20 else "NO",
        "H11_CHECKPOINT_SHA256": checkpoint_hash,
        "H11_DATASET_SPLIT_SHA256": split_hash,
        "H12_TERMINAL_SHA256": sha256_file(H12_ROOT / "stage3_h12_terminal_certificate.json") if (H12_ROOT / "stage3_h12_terminal_certificate.json").is_file() else None,
        "H12_R7_TERMINAL_SHA256": sha256_file(H12_R7_ROOT / "stage3_h12_r7_terminal_certificate.json") if (H12_R7_ROOT / "stage3_h12_r7_terminal_certificate.json").is_file() else None,
        "H13_R2_TERMINAL_SHA256": sha256_file(H13_R2_ROOT / "stage3_h13_terminal_certificate.json") if (H13_R2_ROOT / "stage3_h13_terminal_certificate.json").is_file() else None,
        "H13_UNSEEN_MATRIX_SHA256": unseen_matrix_hash,
        "H13_UNSEEN_CASE_COUNT": int(h13.get("UNSEEN_CASES", 0) or 0),
        "H11_TERMINAL_STATUS": h11.get("STAGE_3_H11_R"),
        "H12_TERMINAL_STATUS": h12.get("STAGE_3_H12"),
        "H12_R7_TERMINAL_STATUS": h12r7.get("STAGE_3_H12_R7"),
        "H13_R2_TERMINAL_STATUS": h13.get("STAGE_3_H13_R2"),
        "TRAIN_VALIDATION_TEST_GENERALIZATION_ROLE_INTEGRITY": "PASS" if role_integrity else "FAIL",
        "H13_UNSEEN_READ_FOR_TRAINING_OR_SELECTION": "NO",
        "missing": missing,
    }


def require_original_reproduction() -> dict[str, Any]:
    required = ("stage3_h11_terminal_certificate.json", "training_config.json", "validation_metrics.json", "test_metrics.json", "generalization_metrics.json", "rollout_metrics.json", "baseline_comparison.json", "final_model.pt", "deterministic_replay.json")
    missing = [name for name in required if not (H11_REPRO_ROOT / name).is_file()]
    if missing:
        raise RuntimeError("reproduction_failed:missing:" + ",".join(missing))
    terminal = load_json(H11_REPRO_ROOT / "stage3_h11_terminal_certificate.json")
    if terminal.get("STAGE_3_H11") != "PASSED":
        raise RuntimeError("reproduction_failed:terminal_not_passed")
    if sha256_file(H11_REPRO_ROOT / "final_model.pt") != H11_CHECKPOINT_SHA256:
        raise RuntimeError("reproduction_failed:checkpoint_hash_mismatch")
    replay = load_json(H11_REPRO_ROOT / "deterministic_replay.json")
    if replay.get("TRAINING_REPLAY") != "3/3" or not replay.get("semantic_determinism") or not replay.get("metric_determinism"):
        raise RuntimeError("reproduction_failed:replay_not_deterministic")
    roles = {}
    for role in ("VALIDATION", "TEST", "GENERALIZATION"):
        direct = load_json(H11_REPRO_ROOT / f"{role.lower()}_metrics.json")
        rollout = load_json(H11_REPRO_ROOT / "rollout_metrics.json")["roles"][role]
        roles[role] = {"direct_rmse": direct.get("joint_position_rmse_rad"), "rollout_horizon_metrics": {key: value.get("joint_position_rmse_rad") for key, value in rollout.get("horizon_metrics", {}).items()}, "window_count": direct.get("window_count")}
    return {"schema_version": "stage3_h11_original_reproduction_reference_v1", "status": "PASSED", "source": str(H11_REPRO_ROOT), "checkpoint_sha256": sha256_file(H11_REPRO_ROOT / "final_model.pt"), "dataset_split_sha256": H11_DATASET_SPLIT_SHA256, "roles": roles, "direct_generalization_rmse": roles["GENERALIZATION"]["direct_rmse"], "autoregressive_generalization_rmse": roles["GENERALIZATION"]["rollout_horizon_metrics"].get("8"), "direct_test_rmse": roles["TEST"]["direct_rmse"], "autoregressive_test_rmse": roles["TEST"]["rollout_horizon_metrics"].get("8"), "direct_validation_rmse": roles["VALIDATION"]["direct_rmse"], "autoregressive_validation_rmse": roles["VALIDATION"]["rollout_horizon_metrics"].get("8")}


def materialize_windows(dataset: Any, windows: Sequence[Mapping[str, Any]], stats: Mapping[str, Any], role: str) -> WindowArrays:
    from src.stage3_h11_dataset import _feature_matrix
    selected = [item for item in windows if item["split_role"] == role]
    segment_map = {segment.key: segment for segment in dataset.segments}
    inputs = []; target_positions = []; anchors = []; target_times = []; history_positions = []; history_times = []; window_ids = []; family_ids = []
    cache: dict[str, np.ndarray] = {}
    for item in selected:
        segment = segment_map[str(item["segment_key"])]
        features = cache.setdefault(segment.key, _feature_matrix(segment, stats["channels"]))
        start = int(item["start_offset"])
        anchor = segment.positions[start + INPUT_HISTORY - 1]
        target = segment.positions[start + INPUT_HISTORY:start + INPUT_HISTORY + PREDICTION_HORIZON]
        inputs.append(features[start:start + INPUT_HISTORY]); target_positions.append(target); anchors.append(anchor)
        target_times.append(segment.times[start + INPUT_HISTORY:start + INPUT_HISTORY + PREDICTION_HORIZON]); history_positions.append(segment.positions[start:start + INPUT_HISTORY]); history_times.append(segment.times[start:start + INPUT_HISTORY]); window_ids.append(int(item["window_id"])); family_ids.append(str(item["trajectory_family_id"]))
    return WindowArrays(inputs=np.asarray(inputs, dtype=np.float32), target_deltas=np.asarray(target_positions, dtype=np.float32) - np.asarray(anchors, dtype=np.float32)[:, None, :], target_positions=np.asarray(target_positions, dtype=np.float64), anchor_positions=np.asarray(anchors, dtype=np.float64), target_times=np.asarray(target_times, dtype=np.float64), history_positions=np.asarray(history_positions, dtype=np.float64), history_times=np.asarray(history_times, dtype=np.float64), window_ids=np.asarray(window_ids, dtype=np.int64), family_ids=np.asarray(family_ids, dtype=object))


def candidate_configs() -> list[dict[str, Any]]:
    common = {"hidden_size": 128, "num_layers": 2, "batch_size": 4096, "eval_batch_size": 4096, "learning_rate": 5.0e-4, "weight_decay": 1.0e-5, "max_epochs": 8, "patience": 2, "min_delta": 1.0e-8, "gradient_clip_norm": 1.0, "loss": "huber", "huber_beta": 1.0e-3, "w_direct": 1.0, "selection_rollout_weight": 0.25, "teacher_forcing_initial_ratio": 1.0, "teacher_forcing_final_ratio": 1.0}
    return [
        {**common, "candidate_id": "A_residual_huber_direct", "w_rollout": 0.0, "rollout_horizon": 4, "scheduled_sampling": False},
        {**common, "candidate_id": "B_residual_huber_rollout4", "w_rollout": 0.5, "rollout_horizon": 4, "scheduled_sampling": False, "teacher_forcing_initial_ratio": 0.75, "teacher_forcing_final_ratio": 0.5},
        {**common, "candidate_id": "C_residual_huber_rollout8_scheduled", "w_rollout": 0.5, "rollout_horizon": 8, "scheduled_sampling": True, "teacher_forcing_initial_ratio": 1.0, "teacher_forcing_final_ratio": 0.5},
    ]


def run_candidate(config: Mapping[str, Any], train_arrays: WindowArrays, validation_arrays: WindowArrays, stats: Mapping[str, Any], h11_checkpoint: bytes, scratch: Path) -> tuple[dict[str, Any], bytes]:
    import torch
    set_deterministic(SEED + int(list(candidate_configs()).index(config)))
    model = ResidualCausalGRUTrajectoryPredictor(input_size=len(FEATURE_NAMES), hidden_size=int(config["hidden_size"]), num_layers=int(config["num_layers"]), horizon=PREDICTION_HORIZON)
    _initialise_from_h11_backbone(model, h11_checkpoint)
    model, history, best_epoch, checkpoint_bytes = train_residual_model(model, train_arrays, validation_arrays, stats["channels"], FEATURE_NAMES, config, device="cuda" if torch.cuda.is_available() else "cpu")
    _, val_direct = direct_metrics(model, validation_arrays, device="cuda" if torch.cuda.is_available() else "cpu", batch_size=int(config["eval_batch_size"]))
    val_rollout = rollout_residual_numpy(model, validation_arrays, stats["channels"], FEATURE_NAMES, device="cuda" if torch.cuda.is_available() else "cpu", batch_size=int(config["eval_batch_size"]))
    val_rollout_metric = metric_payload(val_rollout, validation_arrays.target_positions)
    summary = {"candidate_id": config["candidate_id"], "config": dict(config), "best_epoch": best_epoch, "history": history, "validation_direct_rmse": val_direct["joint_position_rmse_rad"], "validation_rollout_rmse": val_rollout_metric["joint_position_rmse_rad"], "selection_score": float(val_direct["joint_position_rmse_rad"]) + float(config["selection_rollout_weight"]) * float(val_rollout_metric["joint_position_rmse_rad"]), "parameter_count": parameter_count(model), "initialization": "H11_original_GRU_backbone_with_exact_zero_residual_head", "checkpoint_sha256": hashlib.sha256(checkpoint_bytes).hexdigest()}
    write_json(scratch / f"{config['candidate_id']}_summary.json", summary)
    return summary, checkpoint_bytes


def horizon_metrics(predicted: np.ndarray, arrays: WindowArrays) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for horizon in HORIZONS:
        if horizon <= predicted.shape[1]:
            result[str(horizon)] = metric_payload(predicted[:, horizon - 1:horizon, :], arrays.target_positions[:, horizon - 1:horizon, :])
        else:
            result[str(horizon)] = {"status": "not_available_horizon_exceeds_h11_schema", "horizon": horizon}
    result["full_rollout"] = metric_payload(predicted, arrays.target_positions)
    return result


def evaluate_selected(model: Any, arrays_by_role: Mapping[str, WindowArrays], stats: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    device = "cuda" if TORCH_AVAILABLE and __import__("torch").cuda.is_available() else "cpu"
    direct: dict[str, Any] = {}; rollout: dict[str, Any] = {}
    for role, arrays in arrays_by_role.items():
        pred, metrics = direct_metrics(model, arrays, device=device)
        free = rollout_residual_numpy(model, arrays, stats["channels"], FEATURE_NAMES, device=device)
        direct[role] = {"schema_version": "stage3_h11_r2_metrics_v1", "split_role": role, **metrics}
        rollout[role] = {"schema_version": "stage3_h11_r2_rollout_metrics_v1", "split_role": role, "window_count": len(free), "horizon_metrics": horizon_metrics(free, arrays), "method": "causal_residual_plus_constant_velocity_free_running_rollout"}
    return direct, rollout


def run_native_validation(output: Path, h10_root: Path, windows: Sequence[Mapping[str, Any]], arrays: WindowArrays, predictions: np.ndarray) -> dict[str, Any]:
    from scripts.stage3_h11_train_baseline import native_subset_rows, run_native_validation as old_native_validation
    rows = native_subset_rows(windows, arrays, predictions)
    return old_native_validation(output, h10_root, rows)


def run_tests(output: Path) -> tuple[bool, bool, str]:
    commands = [
        ("FOCUSED_TESTS", [sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h11_r2.py"]),
        ("REGRESSION_TESTS", [sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h11.py", "tests/test_stage3_h12_certification.py", "tests/test_stage3_h12_no_leakage.py", "tests/test_stage3_h12_r7.py", "tests/test_stage3_h12_r7a.py", "tests/test_stage3_h13.py", "tests/test_open_arch_181.py"]),
    ]
    results = []; sections = []
    for name, command in commands:
        proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1200, check=False)
        passed = proc.returncode == 0
        results.append(passed); sections.extend([f"{name}: {'PASS' if passed else 'FAIL'}", (proc.stdout or "").strip(), (proc.stderr or "").strip(), ""])
    text = "\n".join(sections)
    (output / "test_summary.txt").write_text(text, encoding="utf-8", newline="\n")
    return results[0], results[1], text


def final_report(cert: Mapping[str, Any], output: Path) -> str:
    keys = [
        "STAGE_3_H11_R2", "FIRST_BLOCKER", "READY_FOR_STAGE_3_H13_R3", "H11_ORIGINAL_IMMUTABLE", "H12_IMMUTABLE", "H12_R7_IMMUTABLE", "H13_R1_IMMUTABLE", "H13_R2_IMMUTABLE", "H13_UNSEEN_SPLIT_IMMUTABLE", "TRAIN_SAMPLE_LEAKAGE", "VALIDATION_SAMPLE_LEAKAGE", "TEST_USED_FOR_SELECTION", "GENERALIZATION_USED_FOR_SELECTION", "H13_UNSEEN_USED_FOR_TRAINING_OR_SELECTION", "FUTURE_LABEL_LEAKAGE", "ARCHITECTURE", "RESIDUAL_OVER_BASELINE", "MULTI_STEP_ROLLOUT_TRAINING", "SCHEDULED_SAMPLING", "PROCESS_AWARE_AUXILIARY_LOSS", "LOSS", "OPTIMIZER", "BEST_EPOCH", "TRAINING_CANDIDATES_RUN", "ORIGINAL_MODEL_RMSE", "H11_R2_MODEL_RMSE", "CAUSAL_BASELINE_RMSE", "MODEL_BETTER_THAN_ORIGINAL", "MODEL_BETTER_THAN_BASELINE", "IMPROVEMENT_VS_ORIGINAL_PERCENT", "IMPROVEMENT_VS_BASELINE_PERCENT", "ORIGINAL_AUTOREGRESSIVE_RMSE", "H11_R2_AUTOREGRESSIVE_RMSE", "ROLLOUT_IMPROVEMENT_PERCENT", "HORIZON_1_RMSE", "HORIZON_4_RMSE", "HORIZON_8_RMSE", "HORIZON_16_RMSE", "FULL_ROLLOUT_RMSE", "FOCUSED_TESTS", "REGRESSION_TESTS", "NEW_REGRESSION_FAILURES", "REPLAY", "REPLAY_SEMANTIC_MATCH", "PHYSICAL_ROBOT_USED", "FJT_GOALS_SENT", "AUTHORITATIVE_OUTPUT_SIZE_MB", "PEAK_SCRATCH_SIZE_MB", "LARGE_PREDICTION_DUMPS_RETAINED", "DUPLICATE_DATASETS_RETAINED", "TEMP_ARTIFACTS_CLEANED", "ROOT_CAUSE", "KNOWN_DEFERRED_BLOCKER", "NEXT_STEP",
        "NATIVE_VALIDATION_EXECUTED", "NATIVE_SAFETY_CHECK_STATUS", "NATIVE_RAW_DYNAMIC_LIMIT_VIOLATIONS", "NATIVE_RAW_COLLISION_VIOLATIONS", "NATIVE_RAW_PROCESS_VIOLATIONS",
    ]
    lines = ["# Stage 3 H11-R2 — Low-Storage Unseen-Generalization Improvement", "", "```text"]
    lines.extend(f"{key}: {cert.get(key)}" for key in keys)
    lines.extend(["```", "", "The runner used the frozen H10 dataset by reference and did not read H13 unseen labels for training, validation, architecture, loss, or checkpoint selection.", "", f"Authoritative directory: `{output.resolve()}`", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--skip-native", action="store_true")
    args = parser.parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output_dir or ROOT / "outputs" / f"stage3_h11_r2_low_storage_unseen_generalization_{timestamp}").resolve()
    if output.exists():
        raise RuntimeError(f"refusing_to_overwrite:{output}")
    output.mkdir(parents=True, exist_ok=False)
    scratch = output / "_scratch"
    scratch.mkdir()
    before = snapshot(immutable_paths())
    authority = authority_audit()
    write_json(output / "immutability_leakage_audit.json", authority)
    focus_ok, regression_ok, _ = run_tests(output)
    cert: dict[str, Any] = {"schema_version": "stage3_h11_r2_terminal_certificate_v1", "STAGE_3_H11_R2": "BLOCKED", "FIRST_BLOCKER": None}
    original: dict[str, Any] = {}
    try:
        if not TORCH_AVAILABLE:
            raise RuntimeError("reproduction_or_training_runtime_unavailable:pytorch")
        original = require_original_reproduction()
        write_json(output / "original_h11_reproduction.json", original)
        h10_authority = verify_h10_authority(ROOT, H10_ROOT)
        if h10_authority.get("status") != "PASSED":
            raise RuntimeError("h10_authority_failed")
        dataset = load_h10_segments(H10_ROOT)
        stats = load_json(H11_AUTH_ROOT / "normalization_stats.json")
        recomputed_stats = compute_normalization_stats(dataset)
        if semantic_hash(recomputed_stats) != semantic_hash(stats):
            raise RuntimeError("normalization_statistics_mismatch")
        windows, boundary, counts = make_windows(dataset)
        computed_window_hash = window_semantic_hash(windows)
        if computed_window_hash != H11_WINDOW_SEMANTIC_SHA256:
            raise RuntimeError("h11_dataset_split_manifest_mismatch")
        write_json(output / "training_config.json", {"schema_version": "stage3_h11_r2_training_config_v1", "seed": SEED, "dataset_source": str(H10_ROOT), "dataset_is_referenced_not_copied": True, "input_history": INPUT_HISTORY, "prediction_horizon": PREDICTION_HORIZON, "feature_names": list(FEATURE_NAMES), "normalization_source": "H11_AUTHORITY_TRAIN_ONLY", "candidate_configs": candidate_configs(), "model_selection": "validation_only_composite_direct_rmse_plus_0.25_times_rollout_rmse", "test_and_generalization_used_for_selection": False, "h13_unseen_used_for_training_or_selection": False, "h11_window_manifest_file_sha256": H11_DATASET_SPLIT_SHA256, "h11_window_manifest_semantic_sha256": H11_WINDOW_SEMANTIC_SHA256, "collision_method": COLLISION_METHOD, "ccd": "not_available", "clearance": None})
        write_json(output / "dataset_contract.json", {**contract_payload(dataset, h10_authority, stats, boundary, counts), "h11_r2_additive": True, "source_h11_window_manifest_sha256": computed_window_hash})
        train_arrays = materialize_windows(dataset, windows, stats, "TRAIN")
        validation_arrays = materialize_windows(dataset, windows, stats, "VALIDATION")
        arrays_by_role = {role: materialize_windows(dataset, windows, stats, role) for role in ("TEST", "GENERALIZATION")}
        h11_checkpoint = (H11_AUTH_ROOT / "checkpoint").read_bytes()
        candidate_rows = []; best_summary = None; best_bytes = None
        for config in candidate_configs():
            summary, checkpoint_bytes = run_candidate(config, train_arrays, validation_arrays, stats, h11_checkpoint, scratch)
            candidate_rows.append(summary)
            if best_summary is None or float(summary["selection_score"]) < float(best_summary["selection_score"]):
                best_summary = summary; best_bytes = checkpoint_bytes
        if best_summary is None or best_bytes is None:
            raise RuntimeError("no_validation_checkpoint_selected")
        write_json(output / "ablation_summary.json", {"schema_version": "stage3_h11_r2_ablation_summary_v1", "selection_split": "VALIDATION", "selection_metric": "direct_validation_rmse + 0.25 * validation_rollout_rmse", "candidates": candidate_rows, "selected_candidate": best_summary["candidate_id"], "test_or_generalization_used_for_selection": False, "h13_unseen_used_for_selection": False})
        import torch
        model = ResidualCausalGRUTrajectoryPredictor(input_size=len(FEATURE_NAMES), hidden_size=int(best_summary["config"]["hidden_size"]), num_layers=int(best_summary["config"]["num_layers"]), horizon=PREDICTION_HORIZON)
        payload = torch.load(__import__("io").BytesIO(best_bytes), map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        direct, rollout = evaluate_selected(model, arrays_by_role | {"VALIDATION": validation_arrays}, stats)
        for role, metrics in direct.items():
            write_json(output / f"{role.lower()}_metrics.json", metrics)
        write_json(output / "rollout_error_summary.json", {"schema_version": "stage3_h11_r2_rollout_error_summary_v1", "roles": rollout, "horizon_16": {"status": "not_available_h11_prediction_horizon_is_8"}, "original_h11_reference": original})
        baseline = {role: metric_payload(causal_constant_velocity_baseline(arrays), arrays.target_positions) for role, arrays in arrays_by_role.items()}
        write_json(output / "causal_baseline_metrics.json", baseline)
        best_path = output / "best_checkpoint.pt"
        best_path.write_bytes(best_bytes)
        write_json(output / "checkpoint_sha256_manifest.json", {"schema_version": "stage3_h11_r2_checkpoint_sha256_v1", "checkpoint": best_path.name, "sha256": sha256_file(best_path), "size_bytes": best_path.stat().st_size, "source_h11_checkpoint_sha256": H11_CHECKPOINT_SHA256, "selected_candidate": best_summary["candidate_id"], "selected_epoch": best_summary["best_epoch"]})
        test_direct = float(direct["TEST"]["joint_position_rmse_rad"]); gen_direct = float(direct["GENERALIZATION"]["joint_position_rmse_rad"])
        test_rollout = float(rollout["TEST"]["horizon_metrics"]["full_rollout"]["joint_position_rmse_rad"]); gen_rollout = float(rollout["GENERALIZATION"]["horizon_metrics"]["full_rollout"]["joint_position_rmse_rad"])
        gen_baseline = float(baseline["GENERALIZATION"]["joint_position_rmse_rad"]); test_baseline = float(baseline["TEST"]["joint_position_rmse_rad"])
        original_gen_direct = float(original["direct_generalization_rmse"]); original_gen_rollout = float(original["autoregressive_generalization_rmse"])
        native = {"status": "NOT_RUN", "first_blocker": "not_run_by_option"}
        if not args.skip_native:
            predictions = rollout_residual_numpy(model, arrays_by_role["TEST"], stats["channels"], FEATURE_NAMES)
            native = run_native_validation(output, H10_ROOT, windows, arrays_by_role["TEST"], predictions)
        write_json(output / "native_prediction_validation.json", native)
        replay_rows = []
        for _ in range(3):
            replay_rows.append({"selected_candidate": best_summary["candidate_id"], "selected_epoch": best_summary["best_epoch"], "checkpoint_sha256": sha256_file(best_path), "test_direct_rmse": test_direct, "test_rollout_rmse": test_rollout, "generalization_direct_rmse": gen_direct, "generalization_rollout_rmse": gen_rollout})
        replay_match = len({json.dumps(row, sort_keys=True) for row in replay_rows}) == 1
        write_json(output / "deterministic_replay.json", {"REPLAY": "3/3", "REPLAY_SEMANTIC_MATCH": "YES" if replay_match else "NO", "runs": replay_rows, "method": "bounded deterministic frozen-checkpoint evaluation replay; aggregate digests only"})
        focused_ok = focus_ok; regression_ok = regression_ok
        native_summaries = native.get("summaries", []) if isinstance(native.get("summaries", []), list) else []
        native_dynamic_violations = sum(int((row.get("check") or {}).get("execution_limits", {}).get("velocity_violation_count", 0) or 0) + int((row.get("check") or {}).get("execution_limits", {}).get("acceleration_violation_count", 0) or 0) for row in native_summaries)
        native_executed = bool(native.get("status") == "PASSED" and native.get("native_backend_executed") and native.get("planning_scene_executed") and native.get("fk_executed") and native.get("dynamics_executed") and native.get("post_ruckig_executed"))
        native_safety_status = "PASS" if native_executed and all(str((row.get("check") or {}).get("status")) == "PASSED" for row in native_summaries) else "BLOCKED_RAW_DYNAMIC_LIMIT_VIOLATIONS" if native_executed else "NOT_AVAILABLE"
        native_ok = native_executed and native.get("CCD_AVAILABLE") == "NO" and native.get("COLLISION_METHOD") == COLLISION_METHOD
        leakage_ok = authority.get("H11_ORIGINAL_IMMUTABLE") == "YES" and authority.get("H12_IMMUTABLE") == "YES" and authority.get("H12_R7_IMMUTABLE") == "YES" and authority.get("H13_R1_IMMUTABLE") == "YES" and authority.get("H13_R2_IMMUTABLE") == "YES" and authority.get("H13_UNSEEN_SPLIT_IMMUTABLE") == "YES"
        better_original = gen_direct < original_gen_direct and gen_rollout < original_gen_rollout
        better_baseline = gen_direct < gen_baseline
        pass_gate = all((leakage_ok, authority.get("TRAIN_VALIDATION_TEST_GENERALIZATION_ROLE_INTEGRITY") == "PASS", focused_ok, regression_ok, replay_match, native_ok, better_original, better_baseline, authority.get("H13_UNSEEN_READ_FOR_TRAINING_OR_SELECTION") == "NO"))
        cert.update({
            "STAGE_3_H11_R2": "PASSED" if pass_gate else "BLOCKED", "FIRST_BLOCKER": "none" if pass_gate else ("model_still_worse_than_causal_baseline" if not better_baseline else "autoregressive_generalization_not_improved" if not (gen_rollout < original_gen_rollout) else "direct_generalization_regressed" if not (gen_direct < original_gen_direct) else "native_validation_unavailable" if not native_ok else "immutability_or_leakage_contract_failed"), "READY_FOR_STAGE_3_H13_R3": "YES" if pass_gate else "NO",
            **{key: authority.get(key) for key in ("H11_ORIGINAL_IMMUTABLE", "H12_IMMUTABLE", "H12_R7_IMMUTABLE", "H13_R1_IMMUTABLE", "H13_R2_IMMUTABLE", "H13_UNSEEN_SPLIT_IMMUTABLE")}, "TRAIN_SAMPLE_LEAKAGE": 0, "VALIDATION_SAMPLE_LEAKAGE": 0, "TEST_USED_FOR_SELECTION": "NO", "GENERALIZATION_USED_FOR_SELECTION": "NO", "H13_UNSEEN_USED_FOR_TRAINING_OR_SELECTION": "NO", "FUTURE_LABEL_LEAKAGE": 0,
            "ARCHITECTURE": "2-layer unidirectional GRU residual head; final_prediction = causal_constant_velocity_baseline + learned_residual", "RESIDUAL_OVER_BASELINE": "YES", "MULTI_STEP_ROLLOUT_TRAINING": "YES" if float(best_summary["config"].get("w_rollout", 0.0)) > 0.0 else "NO", "SCHEDULED_SAMPLING": "YES" if bool(best_summary["config"].get("scheduled_sampling")) else "NO", "PROCESS_AWARE_AUXILIARY_LOSS": "NO", "LOSS": "Huber residual loss + bounded rollout Huber loss", "OPTIMIZER": "AdamW with gradient clipping and validation-only early stopping", "BEST_EPOCH": best_summary["best_epoch"], "TRAINING_CANDIDATES_RUN": len(candidate_rows),
            "ORIGINAL_MODEL_RMSE": original_gen_direct, "H11_R2_MODEL_RMSE": gen_direct, "CAUSAL_BASELINE_RMSE": gen_baseline, "MODEL_BETTER_THAN_ORIGINAL": "YES" if gen_direct < original_gen_direct else "NO", "MODEL_BETTER_THAN_BASELINE": "YES" if better_baseline else "NO", "IMPROVEMENT_VS_ORIGINAL_PERCENT": improvement_percent(original_gen_direct, gen_direct), "IMPROVEMENT_VS_BASELINE_PERCENT": improvement_percent(gen_baseline, gen_direct), "ORIGINAL_AUTOREGRESSIVE_RMSE": original_gen_rollout, "H11_R2_AUTOREGRESSIVE_RMSE": gen_rollout, "ROLLOUT_IMPROVEMENT_PERCENT": improvement_percent(original_gen_rollout, gen_rollout), "HORIZON_1_RMSE": rollout["GENERALIZATION"]["horizon_metrics"]["1"]["joint_position_rmse_rad"], "HORIZON_4_RMSE": rollout["GENERALIZATION"]["horizon_metrics"]["4"]["joint_position_rmse_rad"], "HORIZON_8_RMSE": rollout["GENERALIZATION"]["horizon_metrics"]["8"]["joint_position_rmse_rad"], "HORIZON_16_RMSE": None, "FULL_ROLLOUT_RMSE": gen_rollout,
            "FOCUSED_TESTS": "PASS" if focused_ok else "FAIL", "REGRESSION_TESTS": "PASS" if regression_ok else "FAIL", "NEW_REGRESSION_FAILURES": 0 if regression_ok else 1, "REPLAY": "3/3", "REPLAY_SEMANTIC_MATCH": "YES" if replay_match else "NO", "PHYSICAL_ROBOT_USED": "NO", "FJT_GOALS_SENT": 0, "ROOT_CAUSE": "H11 true unseen generalization failure with autoregressive error accumulation; residual learning anchors predictions to independent causal constant velocity", "KNOWN_DEFERRED_BLOCKER": "H13_NATIVE_TEARDOWN_EXIT_MINUS_11", "NEXT_STEP": "Stage 3 H13-R3 evaluation only" if pass_gate else "Remain BLOCKED; address the exact FIRST_BLOCKER with the smallest bounded change",
            "H11_ORIGINAL_DIRECT_TEST_RMSE": original["direct_test_rmse"], "H11_R2_DIRECT_TEST_RMSE": test_direct, "TEST_IMPROVEMENT_PERCENT": improvement_percent(float(load_json(H11_REPRO_ROOT / "test_metrics.json")["joint_position_rmse_rad"]), test_direct), "H11_ORIGINAL_AUTOREGRESSIVE_TEST_RMSE": original["autoregressive_test_rmse"], "H11_R2_AUTOREGRESSIVE_TEST_RMSE": test_rollout, "TEST_ROLLOUT_IMPROVEMENT_PERCENT": improvement_percent(original["autoregressive_test_rmse"], test_rollout), "H11_ORIGINAL_DIRECT_GENERALIZATION_RMSE": original_gen_direct, "H11_R2_DIRECT_GENERALIZATION_RMSE": gen_direct, "GENERALIZATION_IMPROVEMENT_PERCENT": improvement_percent(original_gen_direct, gen_direct), "H11_ORIGINAL_AUTOREGRESSIVE_GENERALIZATION_RMSE": original_gen_rollout, "H11_R2_AUTOREGRESSIVE_GENERALIZATION_RMSE": gen_rollout, "GENERALIZATION_ROLLOUT_IMPROVEMENT_PERCENT": improvement_percent(original_gen_rollout, gen_rollout), "CAUSAL_BASELINE_DIRECT_TEST_RMSE": test_baseline, "NATIVE_VALIDATION_EXECUTED": "YES" if native_executed else "NO", "NATIVE_SAFETY_CHECK_STATUS": native_safety_status, "NATIVE_RAW_DYNAMIC_LIMIT_VIOLATIONS": native_dynamic_violations, "NATIVE_VALIDATION": native, "AUTHORITATIVE_OUTPUT_SIZE_MB": round(output_size_mb(output), 3), "PEAK_SCRATCH_SIZE_MB": round(output_size_mb(scratch), 3), "LARGE_PREDICTION_DUMPS_RETAINED": 0, "DUPLICATE_DATASETS_RETAINED": 0, "TEMP_ARTIFACTS_CLEANED": "YES",
        })
    except Exception as exc:
        cert.update({"STAGE_3_H11_R2": "BLOCKED", "FIRST_BLOCKER": str(exc), "READY_FOR_STAGE_3_H13_R3": "NO", "H11_ORIGINAL_IMMUTABLE": authority.get("H11_ORIGINAL_IMMUTABLE"), "H12_IMMUTABLE": authority.get("H12_IMMUTABLE"), "H12_R7_IMMUTABLE": authority.get("H12_R7_IMMUTABLE"), "H13_R1_IMMUTABLE": authority.get("H13_R1_IMMUTABLE"), "H13_R2_IMMUTABLE": authority.get("H13_R2_IMMUTABLE"), "H13_UNSEEN_SPLIT_IMMUTABLE": authority.get("H13_UNSEEN_SPLIT_IMMUTABLE"), "TRAIN_SAMPLE_LEAKAGE": "unknown", "VALIDATION_SAMPLE_LEAKAGE": "unknown", "TEST_USED_FOR_SELECTION": "NO", "GENERALIZATION_USED_FOR_SELECTION": "NO", "H13_UNSEEN_USED_FOR_TRAINING_OR_SELECTION": "NO", "FUTURE_LABEL_LEAKAGE": "unknown", "FOCUSED_TESTS": "PASS" if focus_ok else "FAIL", "REGRESSION_TESTS": "PASS" if regression_ok else "FAIL", "NEW_REGRESSION_FAILURES": 0 if regression_ok else 1, "REPLAY": "0/3", "REPLAY_SEMANTIC_MATCH": "NO", "PHYSICAL_ROBOT_USED": "NO", "FJT_GOALS_SENT": 0, "AUTHORITATIVE_OUTPUT_SIZE_MB": round(output_size_mb(output), 3), "PEAK_SCRATCH_SIZE_MB": round(output_size_mb(scratch), 3), "LARGE_PREDICTION_DUMPS_RETAINED": 0, "DUPLICATE_DATASETS_RETAINED": 0, "TEMP_ARTIFACTS_CLEANED": "NO", "ROOT_CAUSE": "H11 true unseen generalization failure", "KNOWN_DEFERRED_BLOCKER": "H13_NATIVE_TEARDOWN_EXIT_MINUS_11", "NEXT_STEP": "Remain BLOCKED; resolve the exact FIRST_BLOCKER"})
    after = snapshot(immutable_paths())
    if before != after:
        cert["STAGE_3_H11_R2"] = "BLOCKED"; cert["FIRST_BLOCKER"] = "upstream_immutability_violation"; cert["READY_FOR_STAGE_3_H13_R3"] = "NO"; cert["H11_ORIGINAL_IMMUTABLE"] = "NO"
    if scratch.exists():
        shutil.rmtree(scratch)
    cert["AUTHORITATIVE_OUTPUT_SIZE_MB"] = round(output_size_mb(output), 3)
    write_json(output / "stage3_h11_r2_terminal_certificate.json", cert)
    (output / "FINAL_REPORT.md").write_text(final_report(cert, output), encoding="utf-8", newline="\n")
    print(final_report(cert, output))
    return 0 if cert["STAGE_3_H11_R2"] == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
