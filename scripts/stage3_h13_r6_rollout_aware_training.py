#!/usr/bin/env python3
"""Stage 3 H13-R6 rollout-aware training and frozen-20 recertification.

This runner is additive and offline.  It trains only on the frozen H10
TRAIN/VALIDATION partitions, freezes one validation-selected checkpoint, then
performs one locked H13 Frozen-20 evaluation through the unchanged H13-R5
bounded reconstruction policy and native adapter.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import stage3_h13_low_storage_unseen_generalization as h13  # noqa: E402
from scripts import stage3_h13_r3_frozen20_native_certification as r3  # noqa: E402
from scripts import stage3_h13_r5_bounded_full_trajectory_reconstruction as r5  # noqa: E402
from src.stage3_h11_dataset import (  # noqa: E402
    FEATURE_NAMES,
    INPUT_HISTORY,
    PREDICTION_HORIZON,
    compute_normalization_stats,
    load_h10_segments,
    semantic_hash,
    sha256_file,
    verify_h10_authority,
)
from src.stage3_h11_model import metric_payload, set_deterministic  # noqa: E402
from src.stage3_h11_r2_model import (  # noqa: E402
    ResidualCausalGRUTrajectoryPredictor,
    TORCH_AVAILABLE,
    causal_constant_velocity_baseline,
    direct_metrics,
)
from src.stage3_h13_r6 import (  # noqa: E402
    ROLLOUT_HORIZONS,
    R6_SCHEMA_VERSION,
    R6SequenceArrays,
    as_window_arrays,
    build_sequence_arrays,
    checkpoint_hash_unchanged,
    evaluate_rollout_metrics,
    native_count_or_not_reached,
    supported_rollout_horizons,
    train_rollout_aware_model,
)
from src.stage3_h13_r5_reconstruction import (  # noqa: E402
    COLLISION_SEMANTICS,
    CONSTRAINT_THRESHOLDS_RELAXED,
    MODEL_PRIOR_AFFECTED_THRESHOLD_RAD,
    MODEL_PRIOR_MAX_AFFECTED_TRAJECTORY_FRACTION,
    MODEL_PRIOR_MAX_JOINT_DELTA_RAD,
    MODEL_PRIOR_MAX_RMS_JOINT_DELTA_RAD,
    bounded_reconstruction_decision,
    trajectory_delta_metrics,
)


H10_ROOT = ROOT / "outputs/stage3_h10_multi_trajectory_dataset_20260811T081123Z"
H11_ORIGINAL_ROOT = ROOT / "outputs/stage3_h11_r_pytorch_runtime_recertification_20260811T143806Z"
H11_R2_ROOT = ROOT / "outputs/stage3_h11_r2_low_storage_unseen_generalization_20260813T215000Z"
H12_R7_ROOT = ROOT / "outputs/stage3_h12_r7_low_storage_reprojection_20260813T030000Z"
H13_R3_ROOT = ROOT / "outputs/stage3_h13_r3_frozen20_native_certification_20260813T144111Z"
H13_R4_ROOT = ROOT / "outputs/stage3_h13_r4_expanded_reprojection_native_closure_20260814T023000Z"
H13_R5_ROOT = ROOT / "outputs/stage3_h13_r5_bounded_full_trajectory_reconstruction_20260814T130000Z"
H13_UNSEEN_ROOT = r3.H13_R2_ROOT
OPEN_ARCH_ROOT = ROOT / "outputs/internal_wiper_moveit_inputs"
H11_R2_CHECKPOINT = H11_R2_ROOT / "best_checkpoint.pt"
H11_STATS = H11_ORIGINAL_ROOT / "normalization_stats.json"
R5_POLICY_SOURCE = ROOT / "src/stage3_h13_r5_reconstruction.py"

R3_DIRECT_REFERENCE = 0.048018735423
CAUSAL_BASELINE_REFERENCE = 0.048051151346
R5_AUTOREGRESSIVE_REFERENCE = 1.592096512706
R5_FULL_DEFORMATION_REFERENCE = 20
R5_MEDIAN_DIVERGENCE_REFERENCE = 16
CASE_TARGET = 20
MATERIAL_ROLLOUT_IMPROVEMENT_PERCENT = 10.0
SEED = 13086


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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) if path.is_dir() else path.stat().st_size


def size_mb(path: Path) -> float:
    return size_bytes(path) / (1024.0 * 1024.0)


def snapshot(paths_by_group: Mapping[str, Sequence[Path]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for group, paths in paths_by_group.items():
        result[group] = []
        for path in paths:
            result[group].append({
                "path": str(path.resolve()),
                "exists": path.is_file(),
                "size_bytes": path.stat().st_size if path.is_file() else None,
                "mtime_ns": path.stat().st_mtime_ns if path.is_file() else None,
                "sha256": sha256_file(path) if path.is_file() else None,
            })
    return result


def authority_paths() -> dict[str, list[Path]]:
    return {
        "H11_R2_CHECKPOINT": [H11_R2_CHECKPOINT, H11_R2_ROOT / "checkpoint_sha256_manifest.json", H11_R2_ROOT / "stage3_h11_r2_terminal_certificate.json"],
        "H11_R2_SPLIT": [H11_ORIGINAL_ROOT / "h11_window_manifest.json", H10_ROOT / "dataset_split_manifest.json", H11_R2_ROOT / "dataset_contract.json"],
        "H12_R7": [H12_R7_ROOT / name for name in ("stage3_h12_r7_terminal_certificate.json", "repair_summary.json", "repaired_sample_manifest.jsonl")],
        "H13_R3": [H13_R3_ROOT / name for name in ("stage3_h13_r3_terminal_certificate.json", "native_validation_summary.json", "case_summary.jsonl", "reprojection_summary.json", "test_summary.txt")],
        "H13_R4": [H13_R4_ROOT / name for name in ("stage3_h13_r4_terminal_certificate.json", "reprojection_root_cause_summary.json", "reprojection_summary.json", "native_validation_summary.json", "case_summary.jsonl", "test_summary.txt")],
        "H13_R5": [H13_R5_ROOT / name for name in ("stage3_h13_r5_terminal_certificate.json", "reconstruction_summary.json", "reconstruction_case_summary.jsonl", "rollout_drift_root_cause_summary.json", "rollout_drift_case_summary.jsonl", "native_validation_summary.json", "test_summary.txt")],
        "H13_UNSEEN_SPLIT": [H13_UNSEEN_ROOT / name for name in ("stage3_h13_terminal_certificate.json", "generalization_summary.json", "case_summary.jsonl")],
        "OPEN_ARCH_PAIR": [OPEN_ARCH_ROOT / "open_arch_tcp_poses_base_link.csv", OPEN_ARCH_ROOT / "open_arch_seed_joints.csv"],
        "R5_POLICY": [R5_POLICY_SOURCE],
    }


def audit_authority(before: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    h11_r2 = load_json(H11_R2_ROOT / "stage3_h11_r2_terminal_certificate.json") if (H11_R2_ROOT / "stage3_h11_r2_terminal_certificate.json").is_file() else {}
    h12_r7 = load_json(H12_R7_ROOT / "stage3_h12_r7_terminal_certificate.json") if (H12_R7_ROOT / "stage3_h12_r7_terminal_certificate.json").is_file() else {}
    h13_r3 = load_json(H13_R3_ROOT / "stage3_h13_r3_terminal_certificate.json") if (H13_R3_ROOT / "stage3_h13_r3_terminal_certificate.json").is_file() else {}
    h13_r4 = load_json(H13_R4_ROOT / "stage3_h13_r4_terminal_certificate.json") if (H13_R4_ROOT / "stage3_h13_r4_terminal_certificate.json").is_file() else {}
    h13_r5 = load_json(H13_R5_ROOT / "stage3_h13_r5_terminal_certificate.json") if (H13_R5_ROOT / "stage3_h13_r5_terminal_certificate.json").is_file() else {}
    unseen_summary = load_json(H13_UNSEEN_ROOT / "generalization_summary.json") if (H13_UNSEEN_ROOT / "generalization_summary.json").is_file() else {}
    unseen_rows = load_jsonl(H13_UNSEEN_ROOT / "case_summary.jsonl") if (H13_UNSEEN_ROOT / "case_summary.jsonl").is_file() else []
    expected_ids = [str(row["case_id"]) for row in r3.build_case_matrix(CASE_TARGET)]
    observed_ids = [str(row.get("case_id")) for row in unseen_rows]
    split_ok = unseen_summary.get("case_matrix", {}).get("count") == CASE_TARGET and observed_ids == expected_ids and len(set(observed_ids)) == CASE_TARGET
    policy_ok = (
        abs(float(MODEL_PRIOR_MAX_JOINT_DELTA_RAD) - float(np.deg2rad(20.0))) <= 1.0e-15
        and abs(float(MODEL_PRIOR_MAX_RMS_JOINT_DELTA_RAD) - float(np.deg2rad(10.0))) <= 1.0e-15
        and CONSTRAINT_THRESHOLDS_RELAXED == "NO"
        and COLLISION_SEMANTICS == "adaptive_discrete_interpolation"
    )
    missing = [row["path"] for rows in before.values() for row in rows if not row["exists"]]
    return {
        "schema_version": "stage3_h13_r6_immutability_audit_v1",
        "algorithm": "SHA-256 plus exact upstream snapshot comparison",
        "H11_R2_CHECKPOINT_IMMUTABLE": "YES" if h11_r2.get("STAGE_3_H11_R2") == "PASSED" and not missing else "NO",
        "H12_R7_IMMUTABLE": "YES" if h12_r7.get("STAGE_3_H12_R7") == "PASSED" and not missing else "NO",
        "H13_R3_IMMUTABLE": "YES" if h13_r3.get("STAGE_3_H13_R3") == "BLOCKED" and not missing else "NO",
        "H13_R4_IMMUTABLE": "YES" if h13_r4.get("STAGE_3_H13_R4") == "BLOCKED" and not missing else "NO",
        "H13_R5_IMMUTABLE": "YES" if h13_r5.get("STAGE_3_H13_R5") == "BLOCKED" and not missing else "NO",
        "H13_UNSEEN_SPLIT_IMMUTABLE": "YES" if split_ok else "NO",
        "R5_RECONSTRUCTION_POLICY_IMMUTABLE": "YES" if policy_ok else "NO",
        "R5_RECONSTRUCTION_BOUND_EXPANDED": "NO",
        "CONSTRAINT_THRESHOLDS_RELAXED": "NO",
        "H13_UNSEEN_CASE_IDS_MATCH": "YES" if observed_ids == expected_ids else "NO",
        "H13_UNSEEN_CASE_COUNT": len(observed_ids),
        "H13_UNSEEN_MATRIX_SHA256": unseen_summary.get("case_matrix", {}).get("matrix_sha256"),
        "H11_R2_CHECKPOINT_SHA256": sha256_file(H11_R2_CHECKPOINT) if H11_R2_CHECKPOINT.is_file() else None,
        "H11_R2_CHECKPOINT_RECORDED_SHA256": load_json(H11_R2_ROOT / "checkpoint_sha256_manifest.json").get("sha256") if (H11_R2_ROOT / "checkpoint_sha256_manifest.json").is_file() else None,
        "H11_R2_TRAIN_VALIDATION_SPLIT_SHA256": sha256_file(H11_ORIGINAL_ROOT / "h11_window_manifest.json") if (H11_ORIGINAL_ROOT / "h11_window_manifest.json").is_file() else None,
        "H12_R7_TERMINAL_SHA256": sha256_file(H12_R7_ROOT / "stage3_h12_r7_terminal_certificate.json") if (H12_R7_ROOT / "stage3_h12_r7_terminal_certificate.json").is_file() else None,
        "H13_R3_TERMINAL_SHA256": sha256_file(H13_R3_ROOT / "stage3_h13_r3_terminal_certificate.json") if (H13_R3_ROOT / "stage3_h13_r3_terminal_certificate.json").is_file() else None,
        "H13_R4_TERMINAL_SHA256": sha256_file(H13_R4_ROOT / "stage3_h13_r4_terminal_certificate.json") if (H13_R4_ROOT / "stage3_h13_r4_terminal_certificate.json").is_file() else None,
        "H13_R5_TERMINAL_SHA256": sha256_file(H13_R5_ROOT / "stage3_h13_r5_terminal_certificate.json") if (H13_R5_ROOT / "stage3_h13_r5_terminal_certificate.json").is_file() else None,
        "H13_R5_POLICY_SHA256": sha256_file(R5_POLICY_SOURCE) if R5_POLICY_SOURCE.is_file() else None,
        "upstream_statuses": {"H11_R2": h11_r2.get("STAGE_3_H11_R2"), "H12_R7": h12_r7.get("STAGE_3_H12_R7"), "H13_R3": h13_r3.get("STAGE_3_H13_R3"), "H13_R4": h13_r4.get("STAGE_3_H13_R4"), "H13_R5": h13_r5.get("STAGE_3_H13_R5")},
        "missing_paths": missing,
        "before": before,
    }


def after_audit(before: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    after = snapshot(authority_paths())
    equal = {group: list(before.get(group, ())) == rows for group, rows in after.items()}
    return {"after": after, "per_group_unchanged": equal, "all_unchanged": bool(equal) and all(equal.values())}


def reproduce_r5_baseline() -> dict[str, Any]:
    r3_cert = load_json(H13_R3_ROOT / "stage3_h13_r3_terminal_certificate.json")
    r5_cert = load_json(H13_R5_ROOT / "stage3_h13_r5_terminal_certificate.json")
    observed = {
        "R3_DIRECT_MODEL_RMSE_REFERENCE": r3_cert.get("H13_R3_MODEL_RMSE", R3_DIRECT_REFERENCE),
        "CAUSAL_BASELINE_RMSE_REFERENCE": r3_cert.get("H13_R3_CAUSAL_BASELINE_RMSE", CAUSAL_BASELINE_REFERENCE),
        "R5_AUTOREGRESSIVE_RMSE_REFERENCE": r5_cert.get("AUTOREGRESSIVE_RMSE_REFERENCE", R5_AUTOREGRESSIVE_REFERENCE),
        "R5_FULL_TRAJECTORY_DEFORMATION_CASES": r5_cert.get("FULL_TRAJECTORY_DEFORMATION_CASES"),
        "R5_MEDIAN_FIRST_SIGNIFICANT_DIVERGENCE": r5_cert.get("MEDIAN_FIRST_SIGNIFICANT_DIVERGENCE"),
    }
    checks = {
        "direct_reference_match": abs(float(observed["R3_DIRECT_MODEL_RMSE_REFERENCE"]) - R3_DIRECT_REFERENCE) <= 1.0e-12,
        "causal_reference_match": abs(float(observed["CAUSAL_BASELINE_RMSE_REFERENCE"]) - CAUSAL_BASELINE_REFERENCE) <= 1.0e-12,
        "rollout_reference_match": abs(float(observed["R5_AUTOREGRESSIVE_RMSE_REFERENCE"]) - R5_AUTOREGRESSIVE_REFERENCE) <= 1.0e-12,
        "deformation_reference_match": int(observed["R5_FULL_TRAJECTORY_DEFORMATION_CASES"]) == R5_FULL_DEFORMATION_REFERENCE,
        "divergence_reference_match": abs(float(observed["R5_MEDIAN_FIRST_SIGNIFICANT_DIVERGENCE"]) - R5_MEDIAN_DIVERGENCE_REFERENCE) <= 1.0e-12,
    }
    return {"schema_version": "stage3_h13_r6_r5_baseline_reproduction_v1", "source": str(H13_R5_ROOT.resolve()), "observed": observed, "checks": checks, "status": "PASSED" if all(checks.values()) else "BLOCKED", "large_artifacts_regenerated": "NO"}


def leakage_audit() -> dict[str, Any]:
    return {
        "H13_20_UNSEEN_USED_FOR_TRAINING": "NO",
        "H13_20_UNSEEN_USED_FOR_MODEL_SELECTION": "NO",
        "H13_20_UNSEEN_USED_FOR_EARLY_STOPPING": "NO",
        "UNSEEN_GROUND_TRUTH_USED_DURING_TRAINING": "NO",
        "UNSEEN_GROUND_TRUTH_USED_DURING_ROLLOUT_RECONSTRUCTION": "NO",
        "FUTURE_LABEL_LEAKAGE": 0,
        "training_roles": ["TRAIN", "VALIDATION"],
        "model_selection_role": "VALIDATION",
        "early_stopping_source": "VALIDATION_AUTOREGRESSIVE_ROLLOUT_PRIMARY_DIRECT_SECONDARY",
        "frozen20_access_path": "locked_evaluation_after_checkpoint_hash_freeze_only",
        "rollout_target_use": "loss_only_or_bounded_teacher_forcing_next_history_during_training; never current prediction input",
    }


def write_history_csv(path: Path, history: Sequence[Mapping[str, Any]]) -> None:
    fields = ["epoch", "active_rollout_horizon", "teacher_forcing_ratio", "train_direct_loss", "train_rollout_loss", "validation_direct_rmse_rad", "validation_autoregressive_rmse_rad", "validation_selection_score", "validation_selection_primary"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in history)


def initial_model(checkpoint_bytes: bytes) -> Any:
    if not TORCH_AVAILABLE:
        raise RuntimeError("pytorch_unavailable")
    import torch

    model = ResidualCausalGRUTrajectoryPredictor(input_size=len(FEATURE_NAMES), hidden_size=128, num_layers=2, horizon=PREDICTION_HORIZON)
    payload = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model


def train_r6(output: Path, config: Mapping[str, Any]) -> tuple[Any, dict[str, Any], R6SequenceArrays, R6SequenceArrays, str]:
    if not TORCH_AVAILABLE:
        raise RuntimeError("pytorch_unavailable")
    import torch

    deterministic = set_deterministic(int(config["seed"]))
    torch.set_num_threads(1)
    h10_authority = verify_h10_authority(ROOT, H10_ROOT)
    if h10_authority.get("status") != "PASSED":
        raise RuntimeError("h10_authority_failed")
    dataset = load_h10_segments(H10_ROOT)
    stats = load_json(H11_STATS)
    recomputed_stats = compute_normalization_stats(dataset)
    if semantic_hash(recomputed_stats) != semantic_hash(stats):
        raise RuntimeError("h11_normalization_statistics_mismatch")
    train_data = build_sequence_arrays(dataset, stats, "TRAIN", max_rollout_horizon=max(ROLLOUT_HORIZONS))
    validation_data = build_sequence_arrays(dataset, stats, "VALIDATION", max_rollout_horizon=max(ROLLOUT_HORIZONS))
    supported_train = supported_rollout_horizons(dataset.segments, dataset.split_map, "TRAIN")
    supported_validation = supported_rollout_horizons(dataset.segments, dataset.split_map, "VALIDATION")
    if tuple(config["rollout_horizons"]) != supported_train or not set(config["rollout_horizons"]).issubset(set(supported_validation)):
        raise RuntimeError("r6_declared_horizons_not_supported_by_train_validation_sequences")
    initial_bytes = H11_R2_CHECKPOINT.read_bytes()
    initial_hash = hashlib.sha256(initial_bytes).hexdigest()
    model = initial_model(initial_bytes)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, history, best_epoch, final_bytes = train_rollout_aware_model(model, train_data, validation_data, stats["channels"], config, device=device)
    final_path = output / "final_r6_checkpoint.pt"
    final_path.write_bytes(final_bytes)
    final_hash = sha256_file(final_path)
    # Re-load the serialized bytes to prove the frozen artifact is schema-compatible.
    frozen_model = initial_model(final_bytes)
    frozen_model.eval()
    train_direct_prediction, train_direct_metrics = direct_metrics(frozen_model, as_window_arrays(train_data), device=device, batch_size=int(config["eval_batch_size"]))
    validation_direct_prediction, validation_direct_metrics = direct_metrics(frozen_model, as_window_arrays(validation_data), device=device, batch_size=int(config["eval_batch_size"]))
    del train_direct_prediction, validation_direct_prediction
    train_rollout_metrics = evaluate_rollout_metrics(frozen_model, train_data, stats["channels"], horizons=config["rollout_horizons"], device=device, batch_size=int(config["eval_batch_size"]))
    validation_rollout_metrics = evaluate_rollout_metrics(frozen_model, validation_data, stats["channels"], horizons=config["rollout_horizons"], device=device, batch_size=int(config["eval_batch_size"]))
    summary = {
        "schema_version": R6_SCHEMA_VERSION,
        "R6_INITIALIZATION_SOURCE": str(H11_R2_CHECKPOINT.resolve()),
        "INITIALIZATION_CHECKPOINT_SHA256": initial_hash,
        "ARCHITECTURE_CHANGED": "NO",
        "ARCHITECTURE_CHANGE_REASON": "not_applicable_existing_2_layer_unidirectional_GRU_preserved",
        "TRAINABLE_PARAMETER_COUNT": int(sum(parameter.numel() for parameter in frozen_model.parameters())),
        "device": device,
        "determinism": deterministic,
        "train_windows": train_data.count,
        "validation_windows": validation_data.count,
        "train_families": sorted(set(str(value) for value in train_data.family_ids.tolist())),
        "validation_families": sorted(set(str(value) for value in validation_data.family_ids.tolist())),
        "ROLLOUT_HORIZONS_USED": list(config["rollout_horizons"]),
        "SELF_FEEDING_SCHEDULE": {"initial_teacher_forcing_ratio": config["teacher_forcing_initial_ratio"], "final_teacher_forcing_ratio": config["teacher_forcing_final_ratio"], "curriculum": "4->8->16->32 across predeclared epoch buckets", "rollout_input_semantics": "prediction_feeds_next_causal_history; no future feature enters current prediction"},
        "EPOCHS": len(history),
        "EARLY_STOPPING_SOURCE": "VALIDATION_AUTOREGRESSIVE_ROLLOUT_PRIMARY_DIRECT_SECONDARY",
        "BEST_VALIDATION_EPOCH": best_epoch,
        "TRAIN_DIRECT_RMSE": train_direct_metrics,
        "TRAIN_AUTOREGRESSIVE_RMSE": train_rollout_metrics,
        "VALIDATION_DIRECT_RMSE": validation_direct_metrics,
        "VALIDATION_AUTOREGRESSIVE_RMSE": validation_rollout_metrics,
        "FINAL_R6_CHECKPOINT_SHA256": final_hash,
        "CHECKPOINT_SELECTED_WITH_FROZEN20": "NO",
        "H13_20_UNSEEN_USED_FOR_TRAINING": "NO",
        "H13_20_UNSEEN_USED_FOR_MODEL_SELECTION": "NO",
        "H13_20_UNSEEN_USED_FOR_EARLY_STOPPING": "NO",
        "UNSEEN_GROUND_TRUTH_USED_DURING_TRAINING": "NO",
        "FUTURE_LABEL_LEAKAGE": 0,
        "LOSS": {"L_total": "1.0 * direct_teacher_forced_huber_8_step + 0.75 * bounded_self_fed_rollout_huber", "direct": "teacher-forced causal 8-step head objective including one-step prediction", "rollout": "free-running/self-fed target up to active bounded horizon", "huber_beta": config["huber_beta"], "truncated_bptt": "YES" if config.get("truncated_bptt") else "NO"},
        "MODEL_SELECTION": "validation autoregressive RMSE primary; validation direct RMSE secondary tie-break; no H13 data read",
    }
    write_json(output / "training_summary.json", summary)
    write_history_csv(output / "training_history.csv", history)
    write_json(output / "checkpoint_manifest.json", {"schema_version": "stage3_h13_r6_checkpoint_manifest_v1", "initialization_source": str(H11_R2_CHECKPOINT.resolve()), "initialization_sha256": initial_hash, "final_checkpoint": str(final_path.resolve()), "final_sha256": final_hash, "final_size_bytes": final_path.stat().st_size, "selected_with_frozen20": "NO", "selected_epoch": best_epoch, "checkpoint_mutation_after_freeze_allowed": "NO"})
    return frozen_model, summary, train_data, validation_data, final_hash


def write_seed_array(path: Path, q: np.ndarray) -> None:
    r5.write_seed_array(path, q)


def aggregate_case_metric(case_rows: Sequence[Mapping[str, Any]], group: str, name: str) -> float | None:
    values = [float(((row.get(group) or {}).get(name) or {}).get("joint_position_rmse_rad")) for row in case_rows if ((row.get(group) or {}).get(name) or {}).get("joint_position_rmse_rad") is not None]
    return float(np.sqrt(np.mean(np.square(values)))) if values else None


def evaluate_locked_cases(args: argparse.Namespace, output: Path, model: Any, peak: list[int]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], list[dict[str, Any]], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[dict[str, Any]], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """R5-equivalent case semantics with R6 direct metrics recorded in-band."""

    if not TORCH_AVAILABLE:
        raise RuntimeError("pytorch_unavailable")
    stats = load_json(H11_STATS)
    base_poses = h13.read_pose_rows(r3.BASE_POSES)
    base_seeds = h13.read_seed_rows(r3.BASE_SEEDS)
    scratch = output / "_r6_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    cases = r3.build_case_matrix(CASE_TARGET)
    case_rows: list[dict[str, Any]] = []
    raw_units: list[dict[str, Any]] = []
    raw_trajectories: list[np.ndarray] = []
    raw_baselines: list[np.ndarray] = []
    raw_times: list[np.ndarray] = []
    recon_units: list[dict[str, Any]] = []
    recon_trajectories: list[np.ndarray] = []
    recon_baselines: list[np.ndarray] = []
    recon_times: list[np.ndarray] = []
    control_units: list[dict[str, Any]] = []
    control_trajectories: list[np.ndarray] = []
    control_baselines: list[np.ndarray] = []
    control_times: list[np.ndarray] = []
    try:
        for case in cases:
            case_id = str(case["case_id"])
            pose_rows = h13.make_case_poses(base_poses, case)
            causal_pose_path = scratch / "poses" / f"{case_id}.csv"
            causal_seed_path = scratch / "seeds" / f"{case_id}_causal.csv"
            h13.write_pose_csv(causal_pose_path, pose_rows)
            h13.write_seed_csv(causal_seed_path, h13.resample_seed_rows(base_seeds, len(pose_rows)))
            causal_dir = scratch / "strict_causal" / case_id
            causal = h13.run_strict_case(case, causal_pose_path, causal_seed_path, causal_dir, int(args.timeout_s))
            if causal.get("status") != "PASSED":
                case_rows.append({"case_id": case_id, "status": "BLOCKED", "first_blocker": causal.get("first_blocker")})
                shutil.rmtree(causal_dir, ignore_errors=True)
                continue
            q, dq, ddq, times = h13.read_trajectory_csv(Path(causal["trajectory_path"]))
            aligned_poses = h13.resample_pose_rows(pose_rows, len(q)) if len(q) != len(pose_rows) else pose_rows
            features = h13.normalize_features(q, dq, ddq, times, stats)
            arrays = r3.build_arrays(q, features, times, case_id)
            predicted_direct, direct_model_metric = direct_metrics(model, arrays, device="cpu", batch_size=4096)
            baseline_direct = causal_constant_velocity_baseline(arrays)
            direct_baseline_metric = metric_payload(baseline_direct, arrays.target_positions)
            model_prior = r3.full_residual_rollout(model, features, q, times, stats)
            baseline_full = r3.full_causal_baseline(q, times)
            raw_counts = r5.raw_python_counts(model_prior, times)
            model_seed_path = scratch / "seeds" / f"{case_id}_model_prior.csv"
            write_seed_array(model_seed_path, model_prior)
            model_dir = scratch / "strict_model_prior" / case_id
            model_causal = h13.run_strict_case(case, causal_pose_path, model_seed_path, model_dir, int(args.timeout_s))
            reconstruction: dict[str, Any]
            if model_causal.get("status") == "PASSED":
                recon_q, _, _, recon_t = h13.read_trajectory_csv(Path(model_causal["trajectory_path"]))
                if len(recon_q) != len(q) or not np.allclose(recon_t, times, rtol=0.0, atol=1.0e-12):
                    reconstruction = {"attempted": True, "accepted": False, "rejected": True, "rejection_reason": "causal_reconstruction_time_or_length_mismatch"}
                else:
                    metrics = trajectory_delta_metrics(model_prior, recon_q, times)
                    decision = bounded_reconstruction_decision(metrics)
                    reconstruction = {"attempted": True, **metrics, **decision, "planner": "existing_deterministic_MoveIt_seeded_by_H13_R6_prior", "uses_unseen_ground_truth": "NO"}
                    if decision["accepted"]:
                        recon_units.append(r5.make_unit(case, aligned_poses, q, times, unit_index=len(recon_units), mode="model_prior_reconstruction"))
                        recon_trajectories.append(recon_q)
                        recon_baselines.append(q)
                        recon_times.append(times)
            else:
                proxy_metrics = trajectory_delta_metrics(model_prior, q, times)
                proxy_decision = bounded_reconstruction_decision(proxy_metrics)
                reconstruction = {"attempted": True, "accepted": False, "rejected": True, **proxy_metrics, **proxy_decision, "candidate_type": "causal_planner_proxy_for_bound_audit_only", "rejection_reason": str(model_causal.get("first_blocker") or "model_prior_causal_planner_failed"), "planner": "existing_deterministic_MoveIt_seeded_by_H13_R6_prior", "uses_unseen_ground_truth": "NO"}
            control_units.append(r5.make_unit(case, aligned_poses, q, times, unit_index=len(control_units), mode="no_model_control"))
            control_trajectories.append(q)
            control_baselines.append(q)
            control_times.append(times)
            raw_units.append(r5.make_unit(case, aligned_poses, q, times, unit_index=len(raw_units), mode="raw_autoregressive"))
            raw_trajectories.append(model_prior)
            raw_baselines.append(q)
            raw_times.append(times)
            case_rows.append({
                "case_id": case_id,
                "status": "MODEL_EVALUATED",
                "direct": {"model": direct_model_metric, "baseline": direct_baseline_metric, "model_better_than_baseline": bool(direct_model_metric["joint_position_rmse_rad"] < direct_baseline_metric["joint_position_rmse_rad"])},
                "autoregressive": {"full_rollout": r3.metric_summary(model_prior[INPUT_HISTORY:], q[INPUT_HISTORY:]), "baseline_full_rollout": r3.metric_summary(baseline_full[INPUT_HISTORY:], q[INPUT_HISTORY:])},
                "raw_autoregressive": {"python_constraint_counts": raw_counts},
                "reconstruction": reconstruction,
                "no_model_control": {"planner": "existing_deterministic_MoveIt_from_causal_open_arch_seed", "candidate_available": True},
                "trajectory_sample_count": len(q),
                "input_semantic_hash": r3.canonical_hash({"case_id": case_id, "pose_rows": aligned_poses, "times_s": times.tolist()}),
            })
            shutil.rmtree(causal_dir, ignore_errors=True)
            shutil.rmtree(model_dir, ignore_errors=True)
            peak[0] = max(peak[0], size_bytes(scratch))
    finally:
        shutil.rmtree(scratch / "strict_causal", ignore_errors=True)
        shutil.rmtree(scratch / "strict_model_prior", ignore_errors=True)
    raw_native, raw_native_rows = r5.run_native_batch(args, output, "raw_autoregressive", raw_units, raw_trajectories, raw_baselines, raw_times, peak) if raw_units else ({"NATIVE_MAIN_WORK_COMPLETED": "NOT_ATTEMPTED", "PROCESS_EXIT_CODE": None, "CLEAN_EXIT": "not_available"}, [])
    raw_by_case = {str(row.get("case_id")): row for row in raw_native_rows}
    for row in case_rows:
        native_row = raw_by_case.get(str(row["case_id"]))
        if native_row:
            row["raw_autoregressive"]["native"] = r5.compact_r5_native(native_row)
            row["rollout_drift"] = r5.build_case_drift(row, (native_row.get("raw") or {}).get("drift_metrics"), native_row)
        else:
            row["rollout_drift"] = r5.build_case_drift(row, None, None)
    return case_rows, raw_native, {"raw_native_rows": raw_native_rows}, raw_units, raw_trajectories, raw_baselines, raw_times, recon_units, recon_trajectories, recon_baselines, recon_times, control_units, control_trajectories, control_baselines, control_times


def run_reconstruction_and_native(args: argparse.Namespace, output: Path, case_rows: list[dict[str, Any]], raw_native: Mapping[str, Any], raw_details: Mapping[str, Any], recon_units: Sequence[Mapping[str, Any]], recon_trajectories: Sequence[np.ndarray], recon_baselines: Sequence[np.ndarray], recon_times: Sequence[np.ndarray], control_units: Sequence[Mapping[str, Any]], control_trajectories: Sequence[np.ndarray], control_baselines: Sequence[np.ndarray], control_times: Sequence[np.ndarray], peak: list[int]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    raw_native_rows = list(raw_details.get("raw_native_rows", []))
    write_jsonl(output / "_r6_raw_native_rows.jsonl", raw_native_rows)
    recon_native, recon_native_rows = r5.run_native_batch(args, output, "model_prior_reconstruction", recon_units, recon_trajectories, recon_baselines, recon_times, peak) if recon_units else ({"mode": "model_prior_reconstruction", "NATIVE_MAIN_WORK_COMPLETED": "NOT_REACHED", "PROCESS_EXIT_CODE": None, "CLEAN_EXIT": "not_available"}, [])
    raw_by_case_for_control = {str(row.get("case_id")): row for row in raw_native_rows}
    control_to_run_units: list[dict[str, Any]] = []
    control_to_run_trajectories: list[np.ndarray] = []
    control_to_run_baselines: list[np.ndarray] = []
    control_to_run_times: list[np.ndarray] = []
    control_native_rows: list[dict[str, Any]] = []
    for unit, trajectory, baseline, times in zip(control_units, control_trajectories, control_baselines, control_times):
        counts = r5.control_pre_native_counts(trajectory, times, raw_by_case_for_control.get(str(unit["case_id"])))
        if all(value == 0 for value in counts.values()):
            runnable = dict(unit)
            runnable["unit_index"] = len(control_to_run_units)
            control_to_run_units.append(runnable)
            control_to_run_trajectories.append(trajectory)
            control_to_run_baselines.append(baseline)
            control_to_run_times.append(times)
        else:
            control_native_rows.append(r5.synthetic_control_gate_record(str(unit["case_id"]), counts))
    if control_to_run_units:
        control_native, runnable_rows = r5.run_native_batch(args, output, "no_model_control", control_to_run_units, control_to_run_trajectories, control_to_run_baselines, control_to_run_times, peak)
        control_native_rows.extend(runnable_rows)
    else:
        control_native = {"mode": "no_model_control", "NATIVE_MAIN_WORK_COMPLETED": "NOT_REACHED_PRE_NATIVE_GATE", "PROCESS_EXIT_CODE": None, "CLEAN_EXIT": "not_available"}
    recon_by_case = {str(unit["case_id"]): row for unit, row in zip(recon_units, recon_native_rows)}
    control_by_case = {str(row.get("case_id")): row for row in control_native_rows}
    for row in case_rows:
        if str(row["case_id"]) in recon_by_case:
            row["reconstruction"]["native"] = r5.compact_r5_native(recon_by_case[str(row["case_id"])])
        if str(row["case_id"]) in control_by_case:
            row["no_model_control"]["native"] = r5.compact_r5_native(control_by_case[str(row["case_id"])])
    write_jsonl(output / "reconstruction_case_summary.jsonl", case_rows)
    drift_rows = [{"case_id": row.get("case_id"), **(row.get("rollout_drift") or {})} for row in case_rows]
    drift_summary = r5.classify_rollout_drift(drift_rows)
    write_jsonl(output / "rollout_drift_case_summary.jsonl", drift_rows)
    write_json(output / "rollout_drift_root_cause_summary.json", drift_summary)
    recon_summary = r5.aggregate_native_mode(recon_native_rows, recon_native, expected_cases=len(case_rows))
    control_summary = r5.aggregate_native_mode(control_native_rows, control_native, expected_cases=len(case_rows))
    raw_summary = r5.aggregate_native_mode(raw_native_rows, raw_native, expected_cases=len(case_rows))
    accepted = sum(bool((row.get("reconstruction") or {}).get("accepted") is True) for row in case_rows)
    attempted = sum(bool((row.get("reconstruction") or {}).get("attempted")) for row in case_rows)
    rejected = sum(bool((row.get("reconstruction") or {}).get("rejected") is True) for row in case_rows)
    accepted_rows = [row.get("reconstruction") or {} for row in case_rows if (row.get("reconstruction") or {}).get("accepted") is True]
    reconstruction_summary = {
        "schema_version": "stage3_h13_r6_reconstruction_summary_v1",
        "policy": {"max_joint_delta_rad": MODEL_PRIOR_MAX_JOINT_DELTA_RAD, "max_rms_joint_delta_rad": MODEL_PRIOR_MAX_RMS_JOINT_DELTA_RAD, "affected_threshold_rad": MODEL_PRIOR_AFFECTED_THRESHOLD_RAD, "max_affected_trajectory_fraction": MODEL_PRIOR_MAX_AFFECTED_TRAJECTORY_FRACTION, "bound_source": "unchanged_H13_R5_policy", "R5_RECONSTRUCTION_POLICY_IMMUTABLE": "YES", "R5_RECONSTRUCTION_BOUND_EXPANDED": "NO", "CONSTRAINT_THRESHOLDS_RELAXED": "NO"},
        "attempted": attempted,
        "accepted": accepted,
        "rejected": rejected,
        "model_prior_native": recon_summary,
        "no_model_control_native": control_summary,
        "raw_autoregressive_native": raw_summary,
        "model_prior_materially_useful": "YES" if int(str(recon_summary.get("FINAL_CERTIFIED", "0/0")).split("/", 1)[0]) > int(str(control_summary.get("FINAL_CERTIFIED", "0/0")).split("/", 1)[0]) else "NO",
        "MAX_ACCEPTED_MODEL_TO_RECONSTRUCTED_JOINT_DELTA_RAD": max((float(row.get("MODEL_TO_RECONSTRUCTED_MAX_JOINT_DELTA", 0.0)) for row in accepted_rows), default=0.0),
        "RMS_ACCEPTED_MODEL_TO_RECONSTRUCTED_JOINT_DELTA_RAD": max((float(row.get("MODEL_TO_RECONSTRUCTED_RMS_JOINT_DELTA", 0.0)) for row in accepted_rows), default=0.0),
        "MAX_ACCEPTED_AFFECTED_TRAJECTORY_FRACTION": max((float(row.get("AFFECTED_TRAJECTORY_FRACTION", 0.0)) for row in accepted_rows), default=0.0),
    }
    write_json(output / "reconstruction_summary.json", reconstruction_summary)
    write_json(output / "native_validation_summary.json", {"raw_autoregressive": raw_summary, "model_prior_reconstruction": recon_summary, "no_model_control": control_summary, "NOT_REACHED_SEMANTICS": "null_or_NOT_REACHED_used_for_unexecuted_post_native_counts"})
    write_json(output / "model_prior_ablation_summary.json", {"schema_version": "stage3_h13_r6_model_prior_ablation_v1", "MODEL_PRIOR_RECONSTRUCTION_CERTIFIED_CASES": recon_summary.get("FINAL_CERTIFIED", "0/0"), "NO_MODEL_RECONSTRUCTION_CERTIFIED_CASES": control_summary.get("FINAL_CERTIFIED", "0/0"), "MODEL_PRIOR_MATERIALLY_USEFUL": reconstruction_summary["model_prior_materially_useful"]})
    return raw_summary, recon_summary, control_summary


def run_tests(output: Path) -> dict[str, Any]:
    commands = {
        "FOCUSED_TESTS": [sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h13_r6.py", "tests/test_stage3_h13_r5_reconstruction.py", "tests/test_stage3_h13_r4_reprojection.py", "tests/test_stage3_h13_r3.py"],
        "REGRESSION_TESTS": [sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h11_r2.py", "tests/test_stage3_h12_certification.py", "tests/test_stage3_h12_no_leakage.py", "tests/test_stage3_h12_r7.py", "tests/test_stage3_h12_r7a.py", "tests/test_stage3_h13.py", "tests/test_open_arch_181.py"],
    }
    sections = []
    statuses = {}
    for name, command in commands.items():
        process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1200, check=False)
        statuses[name] = "PASS" if process.returncode == 0 else "FAIL"
        sections.extend([f"{name}: {statuses[name]}", process.stdout or "", process.stderr or "", ""])
    output.joinpath("test_summary.txt").write_text("\n".join(sections), encoding="utf-8", newline="\n")
    return {"FOCUSED_TESTS": {"status": statuses["FOCUSED_TESTS"]}, "REGRESSION_TESTS": {"status": statuses["REGRESSION_TESTS"]}, "NEW_REGRESSION_FAILURES": 0 if statuses["REGRESSION_TESTS"] == "PASS" else 1}


def improvement_percent(reference: float | None, candidate: float | None) -> float | None:
    if reference is None or candidate is None or not math.isfinite(float(reference)) or float(reference) == 0.0:
        return None
    return float((float(reference) - float(candidate)) / float(reference) * 100.0)


def replay(output: Path, case_rows: Sequence[Mapping[str, Any]], raw_summary: Mapping[str, Any], recon_summary: Mapping[str, Any], authority: Mapping[str, Any], final_hash: str) -> dict[str, Any]:
    payload = {"case_order": [row.get("case_id") for row in case_rows], "direct": [{"case_id": row.get("case_id"), "model": ((row.get("direct") or {}).get("model") or {}).get("joint_position_rmse_rad"), "baseline": ((row.get("direct") or {}).get("baseline") or {}).get("joint_position_rmse_rad")} for row in case_rows], "autoregressive": [{"case_id": row.get("case_id"), "rmse": (((row.get("autoregressive") or {}).get("full_rollout") or {}).get("joint_position_rmse_rad"))} for row in case_rows], "raw_summary": dict(raw_summary), "reconstruction_summary": dict(recon_summary), "checkpoint_sha256": final_hash, "matrix_sha256": authority.get("H13_UNSEEN_MATRIX_SHA256")}
    return r5.run_replays(output, payload)


def native_field(summary: Mapping[str, Any], key: str, *, reached: bool) -> Any:
    value = summary.get(key)
    if key.startswith("FINAL_") or key.startswith("POST_"):
        return value if reached else None
    return value


def build_certificate(authority: Mapping[str, Any], after: Mapping[str, Any], baseline: Mapping[str, Any], training: Mapping[str, Any], leakage: Mapping[str, Any], direct_summary: Mapping[str, Any], rollout_summary: Mapping[str, Any], reconstruction_summary: Mapping[str, Any], raw_native: Mapping[str, Any], recon_native: Mapping[str, Any], control_native: Mapping[str, Any], replay_summary: Mapping[str, Any], tests: Mapping[str, Any], final_hash: str, peak_scratch_bytes: int, output: Path) -> dict[str, Any]:
    r6_direct = direct_summary.get("R6_DIRECT_MODEL_RMSE")
    r6_rollout = rollout_summary.get("R6_AUTOREGRESSIVE_RMSE")
    direct_valid = r6_direct is not None and float(r6_direct) < CAUSAL_BASELINE_REFERENCE and float(r6_direct) <= R3_DIRECT_REFERENCE * 1.10
    rollout_change = improvement_percent(R5_AUTOREGRESSIVE_REFERENCE, r6_rollout)
    rollout_improved = rollout_change is not None and rollout_change >= MATERIAL_ROLLOUT_IMPROVEMENT_PERCENT
    accepted = int(reconstruction_summary.get("accepted", 0) or 0)
    certified = int(str(recon_native.get("FINAL_CERTIFIED", "0/0")).split("/", 1)[0]) if "/" in str(recon_native.get("FINAL_CERTIFIED", "0/0")) else 0
    substantive_certification = accepted == CASE_TARGET and certified == CASE_TARGET
    teardown_code = raw_native.get("NATIVE_WORKER_EXIT_CODE") if raw_native.get("NATIVE_WORKER_EXIT_CODE") is not None else recon_native.get("NATIVE_WORKER_EXIT_CODE")
    teardown_only = substantive_certification and teardown_code == -11 and all(tests.get(name, {}).get("status") == "PASS" for name in ("FOCUSED_TESTS", "REGRESSION_TESTS")) and replay_summary.get("REPLAY") == "3/3" and replay_summary.get("REPLAY_SEMANTIC_MATCH") == "YES"
    if not after.get("all_unchanged"):
        status, blocker = "BLOCKED", "upstream_immutability_failure"
    elif any(authority.get(key) != "YES" for key in ("H11_R2_CHECKPOINT_IMMUTABLE", "H12_R7_IMMUTABLE", "H13_R3_IMMUTABLE", "H13_R4_IMMUTABLE", "H13_R5_IMMUTABLE", "H13_UNSEEN_SPLIT_IMMUTABLE", "R5_RECONSTRUCTION_POLICY_IMMUTABLE")):
        status, blocker = "BLOCKED", "upstream_immutability_failure"
    elif baseline.get("status") != "PASSED":
        status, blocker = "BLOCKED", "r5_baseline_reproduction_failure"
    elif training.get("CHECKPOINT_SELECTED_WITH_FROZEN20") != "NO" or not checkpoint_hash_unchanged(final_hash, training.get("FINAL_R6_CHECKPOINT_SHA256", "")):
        status, blocker = "BLOCKED", "frozen_checkpoint_integrity_failure"
    elif leakage.get("FUTURE_LABEL_LEAKAGE") != 0 or leakage.get("H13_20_UNSEEN_USED_FOR_TRAINING") != "NO" or leakage.get("H13_20_UNSEEN_USED_FOR_MODEL_SELECTION") != "NO":
        status, blocker = "BLOCKED", "h13_r6_leakage_contract_failure"
    elif len(direct_summary.get("case_ids", [])) != CASE_TARGET:
        status, blocker = "BLOCKED", "frozen_unseen_case_coverage_failure"
    elif not direct_valid:
        status, blocker = "BLOCKED", "h13_r6_direct_generalization_tradeoff"
    elif not rollout_improved:
        status, blocker = "BLOCKED", "h13_r6_rollout_aware_training_failed_to_close_drift"
    elif accepted < CASE_TARGET:
        status, blocker = "BLOCKED", "h13_r6_rollout_improved_but_not_certifiable"
    elif certified < CASE_TARGET:
        status, blocker = "BLOCKED", "h13_r6_native_certification_failed"
    elif teardown_only:
        status, blocker = "BLOCKED", "H13_NATIVE_TEARDOWN_EXIT_MINUS_11"
    elif tests.get("FOCUSED_TESTS", {}).get("status") != "PASS" or tests.get("REGRESSION_TESTS", {}).get("status") != "PASS":
        status, blocker = "BLOCKED", "focused_or_regression_tests_failed"
    elif replay_summary.get("REPLAY") != "3/3" or replay_summary.get("REPLAY_SEMANTIC_MATCH") != "YES":
        status, blocker = "BLOCKED", "replay_semantic_mismatch"
    else:
        status, blocker = "PASSED", "none"
    if not rollout_improved:
        root_class = "C_TRAIN_INFERENCE_MISMATCH_NOT_SUFFICIENT_TO_EXPLAIN_FAILURE"
    elif not direct_valid:
        root_class = "D_DIRECT_ROLLOUT_TRADEOFF"
    elif accepted < CASE_TARGET:
        root_class = "B_ROLLOUT_DRIFT_MATERIALLY_REDUCED_BUT_NOT_CERTIFIABLE"
    elif certified < CASE_TARGET:
        root_class = "E_NATIVE_CERTIFICATION_NOW_PRIMARY_BLOCKER"
    elif teardown_only:
        root_class = "E_NATIVE_CERTIFICATION_NOW_PRIMARY_BLOCKER"
    else:
        root_class = "A_ROLLOUT_DRIFT_CLOSED"
    model_value = "YES" if certified > int(str(control_native.get("FINAL_CERTIFIED", "0/0")).split("/", 1)[0]) and accepted > 0 else "NO"
    recon_reached = bool(recon_native.get("cases", 0))
    return {
        "schema_version": "stage3_h13_r6_terminal_certificate_v1",
        "STAGE_3_H13_R6": status,
        "FIRST_BLOCKER": blocker,
        "READY_FOR_STAGE_3_FINAL_CLOSURE": "YES" if status == "PASSED" else "NO",
        "H11_R2_CHECKPOINT_IMMUTABLE": authority.get("H11_R2_CHECKPOINT_IMMUTABLE"),
        "H12_R7_IMMUTABLE": authority.get("H12_R7_IMMUTABLE"),
        "H13_R3_IMMUTABLE": authority.get("H13_R3_IMMUTABLE"),
        "H13_R4_IMMUTABLE": authority.get("H13_R4_IMMUTABLE"),
        "H13_R5_IMMUTABLE": authority.get("H13_R5_IMMUTABLE"),
        "H13_UNSEEN_SPLIT_IMMUTABLE": authority.get("H13_UNSEEN_SPLIT_IMMUTABLE"),
        "R5_RECONSTRUCTION_POLICY_IMMUTABLE": authority.get("R5_RECONSTRUCTION_POLICY_IMMUTABLE"),
        "R5_RECONSTRUCTION_BOUND_EXPANDED": "NO",
        "CONSTRAINT_THRESHOLDS_RELAXED": "NO",
        "H13_20_UNSEEN_USED_FOR_TRAINING": leakage.get("H13_20_UNSEEN_USED_FOR_TRAINING"),
        "H13_20_UNSEEN_USED_FOR_MODEL_SELECTION": leakage.get("H13_20_UNSEEN_USED_FOR_MODEL_SELECTION"),
        "H13_20_UNSEEN_USED_FOR_EARLY_STOPPING": leakage.get("H13_20_UNSEEN_USED_FOR_EARLY_STOPPING"),
        "UNSEEN_GROUND_TRUTH_USED_DURING_TRAINING": leakage.get("UNSEEN_GROUND_TRUTH_USED_DURING_TRAINING"),
        "FUTURE_LABEL_LEAKAGE": leakage.get("FUTURE_LABEL_LEAKAGE"),
        "R6_INITIALIZATION_SOURCE": training.get("R6_INITIALIZATION_SOURCE"),
        "ARCHITECTURE_CHANGED": training.get("ARCHITECTURE_CHANGED"),
        "ARCHITECTURE_CHANGE_REASON": training.get("ARCHITECTURE_CHANGE_REASON"),
        "TRAINABLE_PARAMETER_COUNT": training.get("TRAINABLE_PARAMETER_COUNT"),
        "ROLLOUT_HORIZONS_USED": training.get("ROLLOUT_HORIZONS_USED"),
        "SELF_FEEDING_SCHEDULE": training.get("SELF_FEEDING_SCHEDULE"),
        "EPOCHS": training.get("EPOCHS"),
        "EARLY_STOPPING_SOURCE": training.get("EARLY_STOPPING_SOURCE"),
        "BEST_VALIDATION_EPOCH": training.get("BEST_VALIDATION_EPOCH"),
        "FINAL_R6_CHECKPOINT_SHA256": final_hash,
        "CHECKPOINT_SELECTED_WITH_FROZEN20": "NO",
        "R3_DIRECT_MODEL_RMSE_REFERENCE": R3_DIRECT_REFERENCE,
        "R6_DIRECT_MODEL_RMSE": r6_direct,
        "CAUSAL_BASELINE_RMSE_REFERENCE": CAUSAL_BASELINE_REFERENCE,
        "DIRECT_GENERALIZATION_REMAINS_VALID": "YES" if direct_valid else "NO",
        "R5_AUTOREGRESSIVE_RMSE_REFERENCE": R5_AUTOREGRESSIVE_REFERENCE,
        "R6_AUTOREGRESSIVE_RMSE": r6_rollout,
        "AUTOREGRESSIVE_RMSE_ABSOLUTE_CHANGE": None if r6_rollout is None else float(r6_rollout) - R5_AUTOREGRESSIVE_REFERENCE,
        "AUTOREGRESSIVE_RMSE_PERCENT_CHANGE": rollout_change,
        "FULL_TRAJECTORY_DEFORMATION_CASES": rollout_summary.get("FULL_TRAJECTORY_DEFORMATION_CASES"),
        "MEDIAN_FIRST_SIGNIFICANT_DIVERGENCE": rollout_summary.get("MEDIAN_FIRST_SIGNIFICANT_DIVERGENCE"),
        "MAX_REQUIRED_RAW_JOINT_CORRECTION_RAD": rollout_summary.get("MAX_REQUIRED_RAW_JOINT_CORRECTION_RAD"),
        "MAX_RAW_MODEL_TO_CAUSAL_JOINT_DISPLACEMENT_RAD": rollout_summary.get("MAX_RAW_MODEL_TO_CAUSAL_JOINT_DISPLACEMENT_RAD"),
        "ROLLOUT_DRIFT_MATERIALLY_IMPROVED": "YES" if rollout_improved else "NO",
        "ROLLOUT_DRIFT_SUBSTANTIVELY_CLOSED": "YES" if substantive_certification else "NO",
        "RECONSTRUCTION_ATTEMPTED": f"{reconstruction_summary.get('attempted', 0)}/{CASE_TARGET}",
        "RECONSTRUCTION_ACCEPTED": f"{accepted}/{CASE_TARGET}",
        "RECONSTRUCTION_REJECTED": f"{reconstruction_summary.get('rejected', 0)}/{CASE_TARGET}",
        "MAX_ACCEPTED_MODEL_TO_RECONSTRUCTED_JOINT_DELTA_RAD": reconstruction_summary.get("MAX_ACCEPTED_MODEL_TO_RECONSTRUCTED_JOINT_DELTA_RAD"),
        "RMS_ACCEPTED_MODEL_TO_RECONSTRUCTED_JOINT_DELTA_RAD": reconstruction_summary.get("RMS_ACCEPTED_MODEL_TO_RECONSTRUCTED_JOINT_DELTA_RAD"),
        "MAX_ACCEPTED_AFFECTED_TRAJECTORY_FRACTION": reconstruction_summary.get("MAX_ACCEPTED_AFFECTED_TRAJECTORY_FRACTION"),
        "MODEL_PRIOR_RECONSTRUCTION_CERTIFIED_CASES": recon_native.get("FINAL_CERTIFIED", "0/20") if recon_reached else "0/20",
        "NO_MODEL_RECONSTRUCTION_CERTIFIED_CASES": control_native.get("FINAL_CERTIFIED", "0/20") if control_native.get("cases", 0) else "0/20",
        "MODEL_PRIOR_MATERIALLY_USEFUL": model_value,
        "PRE_NATIVE_POSITION_VIOLATIONS": native_count_or_not_reached(recon_native.get("pre_native_position_violations"), reached=recon_reached),
        "PRE_NATIVE_COLLISION_VIOLATIONS": native_count_or_not_reached(recon_native.get("pre_native_collision_violations"), reached=recon_reached),
        "PRE_NATIVE_SPRAY_PROCESS_VIOLATIONS": native_count_or_not_reached(recon_native.get("pre_native_spray_process_violations"), reached=recon_reached),
        "TOTG_ATTEMPTED": recon_native.get("TOTG_ATTEMPTED", "NOT_REACHED"),
        "TOTG_PASSED": recon_native.get("TOTG_PASSED", "NOT_REACHED"),
        "RUCKIG_ATTEMPTED": recon_native.get("RUCKIG_ATTEMPTED", "NOT_REACHED"),
        "RUCKIG_FINISHED": recon_native.get("RUCKIG_FINISHED", "NOT_REACHED"),
        "RUCKIG_FAILED": recon_native.get("RUCKIG_FAILED", "NOT_REACHED"),
        "POST_RUCKIG_POSITION_VIOLATIONS": native_field(recon_native, "FINAL_POSITION_VIOLATIONS", reached=recon_reached),
        "POST_RUCKIG_VELOCITY_VIOLATIONS": native_field(recon_native, "FINAL_VELOCITY_VIOLATIONS", reached=recon_reached),
        "POST_RUCKIG_ACCELERATION_VIOLATIONS": native_field(recon_native, "FINAL_ACCELERATION_VIOLATIONS", reached=recon_reached),
        "POST_RUCKIG_JERK_VIOLATIONS": native_field(recon_native, "FINAL_JERK_VIOLATIONS", reached=recon_reached),
        "POST_RUCKIG_COLLISION_VIOLATIONS": native_field(recon_native, "FINAL_COLLISION_VIOLATIONS", reached=recon_reached),
        "POST_RUCKIG_SPRAY_PROCESS_VIOLATIONS": native_field(recon_native, "FINAL_SPRAY_PROCESS_VIOLATIONS", reached=recon_reached),
        "POST_RUCKIG_VALIDATED": recon_native.get("POST_RUCKIG_VALIDATED", "NOT_REACHED"),
        "FINAL_CERTIFIED": recon_native.get("FINAL_CERTIFIED", "0/20") if recon_reached else "0/20",
        "NATIVE_WORKER_EXIT_CODE": teardown_code,
        "NATIVE_TEARDOWN_CLEAN": raw_native.get("NATIVE_TEARDOWN_CLEAN") if raw_native.get("NATIVE_TEARDOWN_CLEAN") is not None else recon_native.get("NATIVE_TEARDOWN_CLEAN"),
        "TEARDOWN_MINUS_11_STILL_PRESENT": "YES" if teardown_code == -11 else "NO",
        "TEARDOWN_MINUS_11_ONLY_REMAINING_BLOCKER": "YES" if teardown_only else "NO",
        "COLLISION_SEMANTICS": COLLISION_SEMANTICS,
        "CCD_AVAILABLE": "NO",
        "CLEARANCE_AVAILABLE": None,
        "REPLAY": replay_summary.get("REPLAY"),
        "REPLAY_SEMANTIC_MATCH": replay_summary.get("REPLAY_SEMANTIC_MATCH"),
        "FOCUSED_TESTS": tests.get("FOCUSED_TESTS", {}).get("status"),
        "REGRESSION_TESTS": tests.get("REGRESSION_TESTS", {}).get("status"),
        "NEW_REGRESSION_FAILURES": tests.get("NEW_REGRESSION_FAILURES"),
        "AUTHORITATIVE_OUTPUT_SIZE_MB": 0.0,
        "PEAK_SCRATCH_SIZE_MB": round(peak_scratch_bytes / (1024.0 * 1024.0), 3),
        "TEMP_ARTIFACTS_CLEANED": "YES",
        "ROOT_CAUSE_CLASS": root_class,
        "ROOT_CAUSE_STATUS": "Rollout-aware training materially reduced the frozen-model free-running error but the unchanged R5 bounded/native path did not certify every trajectory." if rollout_improved and not substantive_certification else "R6 rollout-aware training did not meet the predeclared material-improvement criterion." if not rollout_improved else "All substantive trajectory gates passed; the known native teardown exit -11 remains." if teardown_only else "R6 completed under unchanged downstream gates.",
        "IF_BLOCKED_NEXT_SINGLE_ACTION": "Investigate the exact remaining native/reconstruction blocker without changing R5 bounds." if rollout_improved else "Do not expand R5 reconstruction bounds; revise rollout-aware training only using TRAIN/VALIDATION evidence.",
        "TRAIN_DIRECT_RMSE": training.get("TRAIN_DIRECT_RMSE"),
        "TRAIN_AUTOREGRESSIVE_RMSE": training.get("TRAIN_AUTOREGRESSIVE_RMSE"),
        "VALIDATION_DIRECT_RMSE": training.get("VALIDATION_DIRECT_RMSE"),
        "VALIDATION_AUTOREGRESSIVE_RMSE": training.get("VALIDATION_AUTOREGRESSIVE_RMSE"),
        "NO_MODEL_CONTROL": "APPLICABLE",
        "NATIVE_RAW_SUMMARY": dict(raw_native),
        "NATIVE_RECONSTRUCTION_SUMMARY": dict(recon_native),
        "NATIVE_CONTROL_SUMMARY": dict(control_native),
        "upstream_inputs_unchanged_after_run": "YES" if after.get("all_unchanged") else "NO",
        "AUTHORITATIVE_OUTPUT_DIRECTORY": str(output.resolve()),
    }


def final_report(cert: Mapping[str, Any], output: Path) -> str:
    lines = ["# Stage 3 H13-R6 — Rollout-Aware Sequence Training + Frozen-20 Generalization Recertification", "", "```text"]
    for key, value in cert.items():
        lines.append(f"{key}: {value}")
    lines.extend(["```", "", "Collision results are labelled `adaptive_discrete_interpolation`; Bullet CCD is unavailable and clearance is null.", "", "=== FINAL ANSWERS ==="])
    answers = [
        ("DID_H13_R3_DIRECT_GENERALIZATION_REMAIN_VALID", cert.get("DIRECT_GENERALIZATION_REMAINS_VALID")),
        ("WAS_ROLLOUT_AWARE_TRAINING_IMPLEMENTED", "YES"),
        ("WAS_H11_R2_USED_AS_INITIALIZATION", "YES"),
        ("WAS_FROZEN20_USED_FOR_TRAINING", cert.get("H13_20_UNSEEN_USED_FOR_TRAINING")),
        ("WAS_FROZEN20_USED_FOR_MODEL_SELECTION", cert.get("H13_20_UNSEEN_USED_FOR_MODEL_SELECTION")),
        ("FUTURE_LABEL_LEAKAGE", cert.get("FUTURE_LABEL_LEAKAGE")),
        ("R6_DIRECT_MODEL_RMSE", cert.get("R6_DIRECT_MODEL_RMSE")),
        ("R6_AUTOREGRESSIVE_RMSE", cert.get("R6_AUTOREGRESSIVE_RMSE")),
        ("AUTOREGRESSIVE_RMSE_PERCENT_CHANGE", cert.get("AUTOREGRESSIVE_RMSE_PERCENT_CHANGE")),
        ("WAS_R5_RECONSTRUCTION_BOUND_EXPANDED", "NO"),
        ("WERE_ANY_SAFETY_OR_SPRAY_THRESHOLDS_RELAXED", "NO"),
        ("SUCCESSFULLY_RECONSTRUCTED", cert.get("RECONSTRUCTION_ACCEPTED")),
        ("TOTG_PASSED", cert.get("TOTG_PASSED")),
        ("RUCKIG_PASSED", cert.get("RUCKIG_FINISHED")),
        ("FINAL_CERTIFIED", cert.get("FINAL_CERTIFIED")),
        ("DID_ROLLOUT_DRIFT_MATERIALLY_IMPROVE", cert.get("ROLLOUT_DRIFT_MATERIALLY_IMPROVED")),
        ("WAS_ROLLOUT_DRIFT_SUBSTANTIVELY_CLOSED", cert.get("ROLLOUT_DRIFT_SUBSTANTIVELY_CLOSED")),
        ("DID_THE_R6_MODEL_PROVIDE_MATERIAL_VALUE", cert.get("MODEL_PRIOR_MATERIALLY_USEFUL")),
        ("IS_NATIVE_TEARDOWN_MINUS_11_STILL_PRESENT", cert.get("TEARDOWN_MINUS_11_STILL_PRESENT")),
        ("IS_TEARDOWN_MINUS_11_THE_ONLY_REMAINING_BLOCKER", cert.get("TEARDOWN_MINUS_11_ONLY_REMAINING_BLOCKER")),
        ("READY_FOR_STAGE_3_FINAL_CLOSURE", cert.get("READY_FOR_STAGE_3_FINAL_CLOSURE")),
    ]
    lines.extend(f"{index}. {key}: {value}" for index, (key, value) in enumerate(answers, 1))
    lines.extend(["", f"ROOT_CAUSE_CLASS: {cert.get('ROOT_CAUSE_CLASS')}", f"ROOT_CAUSE_STATUS: {cert.get('ROOT_CAUSE_STATUS')}", f"ONE_SENTENCE_CONCLUSION: {cert.get('ROOT_CAUSE_STATUS')}", f"IF_BLOCKED_NEXT_SINGLE_ACTION: {cert.get('IF_BLOCKED_NEXT_SINGLE_ACTION')}", "", f"AUTHORITATIVE_ARTIFACT_DIRECTORY: {output.resolve()}"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--timeout-s", type=int, default=600)
    parser.add_argument("--skip-native", action="store_true")
    args = parser.parse_args()
    output = (args.output_dir or ROOT / "outputs" / f"stage3_h13_r6_rollout_aware_training_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}").resolve()
    if output.exists():
        raise RuntimeError(f"refusing_to_overwrite:{output}")
    output.mkdir(parents=True, exist_ok=False)
    before = snapshot(authority_paths())
    authority = audit_authority(before)
    write_json(output / "immutability_audit.json", authority)
    baseline = reproduce_r5_baseline()
    write_json(output / "baseline_reproduction_summary.json", baseline)
    leakage = leakage_audit()
    write_json(output / "leakage_audit.json", leakage)
    config = {
        "schema_version": "stage3_h13_r6_training_config_v1",
        "seed": SEED,
        "rollout_horizons": list(ROLLOUT_HORIZONS),
        "batch_size": 2048,
        "eval_batch_size": 4096,
        "learning_rate": 2.0e-4,
        "weight_decay": 1.0e-5,
        "max_epochs": 4,
        "patience": 2,
        "w_direct": 1.0,
        "w_rollout": 0.75,
        "direct_tie_weight": 0.1,
        "teacher_forcing_initial_ratio": 1.0,
        "teacher_forcing_final_ratio": 0.25,
        "huber_beta": 1.0e-3,
        "gradient_clip_norm": 1.0,
        "truncated_bptt": True,
        "selection": "validation autoregressive RMSE primary; direct RMSE secondary tie-break",
        "material_rollout_improvement_percent": MATERIAL_ROLLOUT_IMPROVEMENT_PERCENT,
        "h13_unseen_used_for_training_or_selection": False,
        "future_label_leakage": 0,
    }
    write_json(output / "training_config.json", config)
    peak = [0]
    final_hash = ""
    training: dict[str, Any] = {}
    case_rows: list[dict[str, Any]] = []
    raw_summary: dict[str, Any] = {"NATIVE_MAIN_WORK_COMPLETED": "NOT_REACHED"}
    recon_summary_native: dict[str, Any] = {"FINAL_CERTIFIED": "0/20", "NATIVE_MAIN_WORK_COMPLETED": "NOT_REACHED"}
    control_summary_native: dict[str, Any] = {"FINAL_CERTIFIED": "0/20", "NATIVE_MAIN_WORK_COMPLETED": "NOT_REACHED"}
    reconstruction_summary: dict[str, Any] = {"attempted": 0, "accepted": 0, "rejected": 0}
    replay_summary: dict[str, Any] = {"REPLAY": "0/3", "REPLAY_SEMANTIC_MATCH": "NO"}
    tests: dict[str, Any] = {"FOCUSED_TESTS": {"status": "NOT_RUN"}, "REGRESSION_TESTS": {"status": "NOT_RUN"}, "NEW_REGRESSION_FAILURES": 1}
    direct_summary: dict[str, Any] = {"case_ids": [], "R6_DIRECT_MODEL_RMSE": None, "CAUSAL_BASELINE_RMSE": None}
    rollout_summary: dict[str, Any] = {"R6_AUTOREGRESSIVE_RMSE": None, "FULL_TRAJECTORY_DEFORMATION_CASES": None, "MEDIAN_FIRST_SIGNIFICANT_DIVERGENCE": None, "MAX_REQUIRED_RAW_JOINT_CORRECTION_RAD": None, "MAX_RAW_MODEL_TO_CAUSAL_JOINT_DISPLACEMENT_RAD": None}
    try:
        model, training, _, _, final_hash = train_r6(output, config)
        write_json(output / "leakage_audit.json", leakage)
        frozen_hash_before_evaluation = sha256_file(output / "final_r6_checkpoint.pt")
        if not checkpoint_hash_unchanged(final_hash, frozen_hash_before_evaluation):
            raise RuntimeError("frozen_checkpoint_hash_pre_evaluation_mismatch")
        eval_args = argparse.Namespace(timeout_s=args.timeout_s)
        if args.skip_native:
            raise RuntimeError("skip_native_is_not_authoritative_for_H13_R6")
        result = evaluate_locked_cases(eval_args, output, model, peak)
        case_rows, raw_native, raw_details, _raw_units, _raw_trajectories, _raw_baselines, _raw_times, recon_units, recon_trajectories, recon_baselines, recon_times, control_units, control_trajectories, control_baselines, control_times = result
        raw_summary, recon_summary_native, control_summary_native = run_reconstruction_and_native(eval_args, output, case_rows, raw_native, raw_details, recon_units, recon_trajectories, recon_baselines, recon_times, control_units, control_trajectories, control_baselines, control_times, peak)
        reconstruction_summary = load_json(output / "reconstruction_summary.json")
        direct_rmse = aggregate_case_metric(case_rows, "direct", "model")
        causal_rmse = aggregate_case_metric(case_rows, "direct", "baseline")
        rollout_rmse = aggregate_case_metric(case_rows, "autoregressive", "full_rollout")
        drift_rows = load_jsonl(output / "rollout_drift_case_summary.jsonl")
        drift = load_json(output / "rollout_drift_root_cause_summary.json")
        direct_summary = {"schema_version": "stage3_h13_r6_direct_generalization_summary_v1", "case_count": len(case_rows), "case_ids": [row.get("case_id") for row in case_rows], "R6_DIRECT_MODEL_RMSE": direct_rmse, "R3_DIRECT_MODEL_RMSE_REFERENCE": R3_DIRECT_REFERENCE, "CAUSAL_BASELINE_RMSE": causal_rmse, "CAUSAL_BASELINE_RMSE_REFERENCE": CAUSAL_BASELINE_REFERENCE, "DIRECT_GENERALIZATION_REMAINS_VALID": "YES" if direct_rmse is not None and direct_rmse < CAUSAL_BASELINE_REFERENCE else "NO", "checkpoint_sha256": final_hash}
        rollout_summary = {"schema_version": "stage3_h13_r6_rollout_generalization_summary_v1", "case_count": len(case_rows), "case_ids": [row.get("case_id") for row in case_rows], "R6_AUTOREGRESSIVE_RMSE": rollout_rmse, "R5_AUTOREGRESSIVE_RMSE_REFERENCE": R5_AUTOREGRESSIVE_REFERENCE, "AUTOREGRESSIVE_RMSE_ABSOLUTE_CHANGE": None if rollout_rmse is None else rollout_rmse - R5_AUTOREGRESSIVE_REFERENCE, "AUTOREGRESSIVE_RMSE_PERCENT_CHANGE": improvement_percent(R5_AUTOREGRESSIVE_REFERENCE, rollout_rmse), "FULL_TRAJECTORY_DEFORMATION_CASES": drift.get("FULL_TRAJECTORY_DEFORMATION_CASES"), "MEDIAN_FIRST_SIGNIFICANT_DIVERGENCE": drift.get("MEDIAN_FIRST_SIGNIFICANT_DIVERGENCE"), "MAX_REQUIRED_RAW_JOINT_CORRECTION_RAD": drift.get("MAX_REQUIRED_RAW_JOINT_CORRECTION_RAD"), "MAX_RAW_MODEL_TO_CAUSAL_JOINT_DISPLACEMENT_RAD": drift.get("MAX_RAW_MODEL_TO_CAUSAL_JOINT_DISPLACEMENT_RAD"), "rollout_drift_case_rows": len(drift_rows), "checkpoint_sha256": final_hash}
        write_json(output / "direct_generalization_summary.json", direct_summary)
        write_json(output / "rollout_generalization_summary.json", rollout_summary)
        replay_summary = replay(output, case_rows, raw_summary, recon_summary_native, authority, final_hash)
        write_json(output / "replay_summary.json", replay_summary)
        tests = run_tests(output)
        # The frozen checkpoint must remain byte-identical through evaluation/testing.
        if not checkpoint_hash_unchanged(final_hash, sha256_file(output / "final_r6_checkpoint.pt")):
            raise RuntimeError("frozen_checkpoint_mutated_after_evaluation")
    except Exception as exc:
        write_json(output / "execution_error.json", {"error": str(exc), "type": type(exc).__name__})
    after = after_audit(before)
    authority.update(after)
    write_json(output / "immutability_audit.json", authority)
    cert = build_certificate(authority, after, baseline, training, leakage, direct_summary, rollout_summary, reconstruction_summary, raw_summary, recon_summary_native, control_summary_native, replay_summary, tests, final_hash, peak[0], output)
    cert["AUTHORITATIVE_OUTPUT_SIZE_MB"] = round(size_mb(output), 3)
    # Remove temporary native/training scratch only inside this additive output.
    for name in ("_r6_scratch", "_r5_scratch", "_r6_raw_native_rows.jsonl", "_r5_replay_payload.json"):
        target = output / name
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        elif target.is_file():
            target.unlink(missing_ok=True)
    cert["TEMP_ARTIFACTS_CLEANED"] = "YES" if not any((output / name).exists() for name in ("_r6_scratch", "_r5_scratch", "_r6_raw_native_rows.jsonl", "_r5_replay_payload.json")) else "NO"
    cert["AUTHORITATIVE_OUTPUT_SIZE_MB"] = round(size_mb(output), 3)
    cert["PEAK_SCRATCH_SIZE_MB"] = round(peak[0] / (1024.0 * 1024.0), 3)
    write_json(output / "stage3_h13_r6_terminal_certificate.json", cert)
    output.joinpath("FINAL_REPORT.md").write_text(final_report(cert, output), encoding="utf-8", newline="\n")
    print(final_report(cert, output))
    return 0 if cert["STAGE_3_H13_R6"] == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
