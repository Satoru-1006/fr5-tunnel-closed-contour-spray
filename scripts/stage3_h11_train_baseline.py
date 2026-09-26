#!/usr/bin/env python3
"""Execute and certify the Stage 3 H11 deterministic GRU baseline.

The script is offline and software-only.  It never imports a robot driver,
controller client, or action transport.  Native prediction recertification is
delegated to the separate MoveIt2-only worker after the checkpoint is frozen.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
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
    ACCELERATION_LIMIT,
    EXPECTED_H10_SEMANTIC_SHA256,
    FEATURE_NAMES,
    H10_ML_INPUT_ARTIFACTS,
    INPUT_HISTORY,
    JERK_LIMIT,
    NORMALIZED_FEATURE_NAMES,
    POSITION_LOWER,
    POSITION_UPPER,
    PREDICTION_HORIZON,
    VELOCITY_LIMIT,
    contract_payload,
    environment_metadata,
    load_h10_segments,
    make_windows,
    sha256_file,
    snapshot_frozen_upstream,
    verify_h10_authority,
    window_semantic_hash,
    write_json,
)
from src.stage3_h11_model import (  # noqa: E402
    CausalGRUTrajectoryPredictor,
    WindowArrays,
    TORCH_AVAILABLE,
    constant_velocity_prediction,
    evaluate_model,
    finite_difference_diagnostics,
    hold_last_prediction,
    metric_payload,
    parameter_count,
    rollout_one_step,
    set_deterministic,
    train_model,
    write_history_csv,
)


SEED = 11011
MODEL_CONFIG = {
    "model_type": "causal_gru", "input_history": INPUT_HISTORY, "prediction_horizon": PREDICTION_HORIZON,
    "input_features": list(FEATURE_NAMES), "hidden_size": 128, "num_layers": 2, "bidirectional": False,
    "head": "Linear(128,128)-Tanh-Linear(128,48)", "batch_size": 1024, "eval_batch_size": 2048,
    "learning_rate": 1.0e-3, "weight_decay": 0.0, "max_epochs": 18, "patience": 4, "min_delta": 1.0e-10,
    "loss": "MSE(predicted_joint_delta,target_joint_delta)", "target_units": "rad", "dtype": "float32",
}
REQUIRED_OUTPUTS = (
    "FINAL_REPORT.md", "stage3_h11_terminal_certificate.json", "stage3_h11_gate_report.json", "run_metadata.json",
    "upstream_immutability_report.json", "h11_dataset_contract.json", "h11_window_manifest.json", "window_boundary_audit.json",
    "normalization_stats.json", "training_config.json", "training_history.json", "training_history.csv", "model_summary.txt",
    "checkpoint_sha256_manifest.json", "deterministic_replay.json", "baseline_comparison.json", "train_metrics.json",
    "validation_metrics.json", "test_metrics.json", "generalization_metrics.json", "rollout_metrics.json",
    "raw_prediction_constraint_report.json", "native_prediction_validation.json", "test_results.txt", "regression_results.txt",
    "frozen_artifact_manifest.json",
)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    suffix = str(resolved).split(":", 1)[-1].lstrip("/").replace(chr(92), "/")
    return f"/mnt/{drive}/{suffix}"


def jsonl_write(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n")


def find_frozen_input(h10_root: Path, marker: str) -> Path | None:
    manifest = h10_root / "frozen_artifact_manifest.json"
    if not manifest.is_file():
        return None
    data = json.loads(manifest.read_text(encoding="utf-8"))
    for item in data.get("records", []):
        if marker in str(item.get("status", "")):
            path = Path(str(item.get("absolute_path") or item.get("path")))
            return path if path.is_absolute() else ROOT / path
    return None


def materialize_windows(dataset: Any, windows: Sequence[Mapping[str, Any]], stats: Mapping[str, Any], role: str) -> WindowArrays:
    from src.stage3_h11_dataset import _feature_matrix  # local private helper keeps public contract small

    selected = [item for item in windows if item["split_role"] == role]
    segment_map = {segment.key: segment for segment in dataset.segments}
    feature_cache: dict[str, np.ndarray] = {}
    inputs: list[np.ndarray] = []
    target_deltas: list[np.ndarray] = []
    target_positions: list[np.ndarray] = []
    anchors: list[np.ndarray] = []
    target_times: list[np.ndarray] = []
    history_positions: list[np.ndarray] = []
    history_times: list[np.ndarray] = []
    window_ids: list[int] = []
    family_ids: list[str] = []
    for item in selected:
        segment = segment_map[str(item["segment_key"])]
        features = feature_cache.setdefault(segment.key, _feature_matrix(segment, stats["channels"]))
        start = int(item["start_offset"])
        anchor = segment.positions[start + INPUT_HISTORY - 1]
        target = segment.positions[start + INPUT_HISTORY:start + INPUT_HISTORY + PREDICTION_HORIZON]
        inputs.append(features[start:start + INPUT_HISTORY])
        target_deltas.append(target - anchor[None, :])
        target_positions.append(target)
        anchors.append(anchor)
        target_times.append(segment.times[start + INPUT_HISTORY:start + INPUT_HISTORY + PREDICTION_HORIZON])
        history_positions.append(segment.positions[start:start + INPUT_HISTORY])
        history_times.append(segment.times[start:start + INPUT_HISTORY])
        window_ids.append(int(item["window_id"]))
        family_ids.append(str(item["trajectory_family_id"]))
    return WindowArrays(
        inputs=np.asarray(inputs, dtype=np.float32), target_deltas=np.asarray(target_deltas, dtype=np.float32),
        target_positions=np.asarray(target_positions, dtype=np.float64), anchor_positions=np.asarray(anchors, dtype=np.float64),
        target_times=np.asarray(target_times, dtype=np.float64), history_positions=np.asarray(history_positions, dtype=np.float64),
        history_times=np.asarray(history_times, dtype=np.float64), window_ids=np.asarray(window_ids, dtype=np.int64),
        family_ids=np.asarray(family_ids, dtype=object),
    )


def stats_leakage(stats: Mapping[str, Any], dataset: Any) -> int:
    train_family_count = len({segment.family_id for segment in dataset.segments if dataset.split_map[segment.family_id] == "TRAIN"})
    violations = 0
    if stats.get("source_split") != "TRAIN" or int(stats.get("source_family_count", -1)) != train_family_count:
        violations += 1
    for name in NORMALIZED_FEATURE_NAMES:
        channel = stats.get("channels", {}).get(name, {})
        if channel.get("source_split") != "TRAIN":
            violations += 1
    return violations


def _empty_metric() -> dict[str, Any]:
    return {"window_count": 0, "horizon": None, "joint_position_mae_rad": None, "joint_position_rmse_rad": None, "mae_per_joint_rad": [None] * 6, "rmse_per_joint_rad": [None] * 6, "maximum_absolute_error_rad": None, "p50_absolute_error_rad": None, "p95_absolute_error_rad": None, "p99_absolute_error_rad": None}


def raw_constraint_report(predictions: Mapping[str, np.ndarray], arrays_by_role: Mapping[str, WindowArrays]) -> dict[str, Any]:
    report: dict[str, Any] = {"schema_version": "stage3_h11_raw_prediction_constraint_report_v1", "raw_predictions_are_unclipped": True, "post_processing": "none", "limits": {"position_lower_rad": POSITION_LOWER.tolist(), "position_upper_rad": POSITION_UPPER.tolist(), "velocity_abs_rad_s": VELOCITY_LIMIT.tolist(), "acceleration_abs_rad_s2": ACCELERATION_LIMIT.tolist(), "jerk_abs_rad_s3": JERK_LIMIT.tolist()}, "roles": {}}
    totals = Counter()
    for role, pred in predictions.items():
        arrays = arrays_by_role[role]
        if len(pred) == 0:
            report["roles"][role] = {"window_count": 0, "element_counts": {}, "window_counts": {}, "RAW_MODEL_HARD_CONSTRAINT_VIOLATIONS": 0}
            continue
        full_q = np.concatenate([arrays.anchor_positions[:, None, :], pred], axis=1)
        full_t = np.concatenate([arrays.history_times[:, -1, None], arrays.target_times], axis=1)
        dt = np.diff(full_t, axis=1)
        finite = np.isfinite(full_q).all(axis=(1, 2)) & np.isfinite(full_t).all(axis=1)
        velocity = np.diff(full_q, axis=1) / np.maximum(dt[:, :, None], 1.0e-12)
        acceleration = np.diff(velocity, axis=1) / np.maximum(dt[:, 1:, None], 1.0e-12)
        jerk = np.diff(acceleration, axis=1) / np.maximum(dt[:, 2:, None], 1.0e-12)
        counts = {
            "nan_inf_windows": int(np.count_nonzero(~finite)), "position_limit_elements": int(np.count_nonzero((pred < POSITION_LOWER[None, None, :]) | (pred > POSITION_UPPER[None, None, :]))),
            "velocity_limit_elements": int(np.count_nonzero(np.abs(velocity) > VELOCITY_LIMIT[None, None, :])), "acceleration_limit_elements": int(np.count_nonzero(np.abs(acceleration) > ACCELERATION_LIMIT[None, None, :])),
            "jerk_limit_elements": int(np.count_nonzero(np.abs(jerk) > JERK_LIMIT[None, None, :])), "non_monotonic_time_windows": int(np.count_nonzero(np.any(dt <= 0.0, axis=1))),
            "discontinuity_windows": int(np.count_nonzero(np.any(np.linalg.norm(np.diff(full_q, axis=1), axis=2) > 0.5, axis=1))),
        }
        window_counts = {key: int(value) for key, value in {
            "nan_inf": np.count_nonzero(~finite), "position_limit": np.count_nonzero(np.any((pred < POSITION_LOWER[None, None, :]) | (pred > POSITION_UPPER[None, None, :]), axis=(1, 2))),
            "velocity_limit": np.count_nonzero(np.any(np.abs(velocity) > VELOCITY_LIMIT[None, None, :], axis=(1, 2))), "acceleration_limit": np.count_nonzero(np.any(np.abs(acceleration) > ACCELERATION_LIMIT[None, None, :], axis=(1, 2))),
            "jerk_limit": np.count_nonzero(np.any(np.abs(jerk) > JERK_LIMIT[None, None, :], axis=(1, 2))), "discontinuity": np.count_nonzero(np.any(np.linalg.norm(np.diff(full_q, axis=1), axis=2) > 0.5, axis=1)),
        }.items()}
        hard = int(sum(counts.values()))
        report["roles"][role] = {"window_count": len(pred), "element_counts": counts, "window_counts": window_counts, "RAW_MODEL_HARD_CONSTRAINT_VIOLATIONS": hard, "discontinuity_threshold_rad_per_sample": 0.5, "derivative_semantics": "finite differences of raw reconstructed model positions; not native controller feedback"}
        totals.update(counts)
    report["RAW_MODEL_HARD_CONSTRAINT_VIOLATIONS"] = int(sum(totals.values()))
    report["aggregate_element_counts"] = dict(totals)
    return report


def native_subset_rows(windows: Sequence[Mapping[str, Any]], arrays: WindowArrays, predictions: np.ndarray) -> list[dict[str, Any]]:
    window_by_id = {int(item["window_id"]): item for item in windows}
    role_by_window_id = {window_id: str(item["split_role"]) for window_id, item in window_by_id.items()}
    selected_indices: list[int] = []
    seen: set[tuple[str, str]] = set()
    for index, family in enumerate(arrays.family_ids.tolist()):
        role = role_by_window_id[int(arrays.window_ids[index])]
        if role not in {"TEST", "GENERALIZATION"}:
            continue
        key = (role, str(family))
        if key not in seen:
            selected_indices.append(index); seen.add(key)
    rows: list[dict[str, Any]] = []
    for index in selected_indices:
        item = window_by_id[int(arrays.window_ids[index])]
        q = predictions[index]
        t = arrays.target_times[index] - arrays.target_times[index, 0]
        dt = np.diff(np.concatenate([[0.0], t]))
        velocity = np.diff(np.concatenate([arrays.anchor_positions[index][None, :], q], axis=0), axis=0) / np.maximum(dt[:, None], 1.0e-9)
        acceleration = np.zeros_like(velocity)
        if len(velocity) > 1:
            acceleration[1:] = np.diff(velocity, axis=0) / np.maximum(np.diff(t)[:, None], 1.0e-9)
            acceleration[0] = acceleration[1]
        for local in range(PREDICTION_HORIZON):
            rows.append({"window_id": int(item["window_id"]), "split_role": item["split_role"], "trajectory_family_id": item["trajectory_family_id"], "trajectory_id": item["trajectory_id"], "segment_id": int(item["segment_id"]), "segment_order": int(item["segment_order"]), "primitive_id": item["primitive_id"], "spray_state": item["spray_state"], "trajectory_index": local, "time_from_start_s": float(t[local]), "positions_rad": q[local].tolist(), "velocities_rad_s": velocity[local].tolist(), "accelerations_rad_s2": acceleration[local].tolist()})
    return rows


def run_native_validation(output: Path, h10_root: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    subset_path = output / "native_prediction_subset.jsonl"
    jsonl_write(subset_path, rows)
    input_path = output / "native_prediction_subset.jsonl"
    native_report = output / "native_prediction_validation.json"
    if not rows:
        result = {"schema_version": "stage3_h11_native_prediction_validation_v1", "status": "BLOCKED", "first_blocker": "native_prediction_validation_unavailable", "reason": "no held-out subset rows", "RAW_MODEL_COLLISION_VIOLATIONS": None, "RAW_MODEL_PROCESS_TOLERANCE_VIOLATIONS": None, "CCD_AVAILABLE": "NO", "CLEARANCE_AVAILABLE": "NO", "COLLISION_METHOD": "adaptive_discrete_interpolation"}
        write_json(native_report, result); return result
    markers = {name: find_frozen_input(h10_root, name) for name in ("h6_validation", "h6_segments", "process_contract", "fixture_mesh", "runtime_limits")}
    if any(value is None or not value.is_file() for value in markers.values()):
        result = {"schema_version": "stage3_h11_native_prediction_validation_v1", "status": "BLOCKED", "first_blocker": "native_prediction_validation_unavailable", "reason": "frozen native input missing", "missing_inputs": {key: str(value) for key, value in markers.items()}, "RAW_MODEL_COLLISION_VIOLATIONS": None, "RAW_MODEL_PROCESS_TOLERANCE_VIOLATIONS": None, "CCD_AVAILABLE": "NO", "CLEARANCE_AVAILABLE": "NO", "COLLISION_METHOD": "adaptive_discrete_interpolation"}
        write_json(native_report, result); return result
    install = "/mnt/d/robotfucker/install/stage3_h7_4_native/lib/python3.12/site-packages"
    ament_prefix = "/mnt/d/robotfucker/install/fr5_tunnel_moveit_bridge:/mnt/d/robotfucker/install/fairino5_v6_moveit2_config:/mnt/d/robotfucker/install/fairino_description:/opt/ros/jazzy"
    launch_arguments = " ".join([
        f"input_jsonl:={wsl_path(input_path)}", f"output_json:={wsl_path(native_report)}", f"h6_validation:={wsl_path(markers['h6_validation'])}", f"h6_segments:={wsl_path(markers['h6_segments'])}", f"process_contract:={wsl_path(markers['process_contract'])}", f"fixture_mesh:={wsl_path(markers['fixture_mesh'])}", f"runtime_limits:={wsl_path(markers['runtime_limits'])}",
    ])
    command = " && ".join([
        "source /opt/ros/jazzy/setup.bash", f"export AMENT_PREFIX_PATH={ament_prefix}", f"export PYTHONPATH={install}:/mnt/d/robotfucker/ros2_moveit_bridge:/opt/ros/jazzy/lib/python3.12/site-packages",
        f"ros2 launch /mnt/d/robotfucker/ros2_moveit_bridge/launch/stage3_h11_native.launch.py {launch_arguments}",
    ])
    (output / "native_prediction_command.txt").write_text(command + "\n", encoding="utf-8", newline="\n")
    try:
        completed = subprocess.run(["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", command], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=3600, check=False)
        (output / "native_prediction_stdout.log").write_text(completed.stdout or "", encoding="utf-8", newline="\n")
        (output / "native_prediction_stderr.log").write_text(completed.stderr or "", encoding="utf-8", newline="\n")
    except Exception as exc:
        result = {"schema_version": "stage3_h11_native_prediction_validation_v1", "status": "BLOCKED", "first_blocker": "native_prediction_validation_unavailable", "reason": f"native invocation exception:{type(exc).__name__}:{exc}", "RAW_MODEL_COLLISION_VIOLATIONS": None, "RAW_MODEL_PROCESS_TOLERANCE_VIOLATIONS": None, "CCD_AVAILABLE": "NO", "CLEARANCE_AVAILABLE": "NO", "COLLISION_METHOD": "adaptive_discrete_interpolation"}
        write_json(native_report, result); return result
    if not native_report.is_file():
        result = {"schema_version": "stage3_h11_native_prediction_validation_v1", "status": "BLOCKED", "first_blocker": "native_prediction_validation_unavailable", "reason": f"native worker result missing; returncode={completed.returncode}", "RAW_MODEL_COLLISION_VIOLATIONS": None, "RAW_MODEL_PROCESS_TOLERANCE_VIOLATIONS": None, "CCD_AVAILABLE": "NO", "CLEARANCE_AVAILABLE": None, "COLLISION_METHOD": "adaptive_discrete_interpolation"}
        write_json(native_report, result); return result
    result = json.loads(native_report.read_text(encoding="utf-8"))
    if completed.returncode != 0 and result.get("status") == "PASSED":
        result["status"] = "BLOCKED"; result["first_blocker"] = "native_prediction_validation_unavailable"
    return result


def run_command(command: Sequence[str], output_path: Path) -> int:
    try:
        process = subprocess.run(list(command), cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
        output_path.write_text((process.stdout or "") + (process.stderr or ""), encoding="utf-8", newline="\n")
        return int(process.returncode)
    except Exception as exc:
        output_path.write_text(f"command_exception:{type(exc).__name__}:{exc}\n", encoding="utf-8", newline="\n")
        return 2


def write_frozen_artifact_manifest(output: Path) -> dict[str, Any]:
    records = []
    for path in sorted(output.iterdir()):
        if not path.is_file() or path.name == "frozen_artifact_manifest.json":
            continue
        records.append({"path": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path), "immutable": True})
    manifest = {"schema_version": "stage3_h11_frozen_artifact_manifest_v1", "algorithm": "SHA-256", "records": records}
    write_json(output / "frozen_artifact_manifest.json", manifest)
    return manifest


def write_blocked_artifacts(output: Path, blocker: str, authority: Mapping[str, Any] | None = None, upstream: Mapping[str, Any] | None = None, values: Mapping[str, Any] | None = None) -> None:
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "run_metadata.json", {"schema_version": "stage3_h11_run_metadata_v1", "status": "BLOCKED", "FIRST_BLOCKER": blocker, "software_only": True, "physical_robot_connected": False, "physical_driver_loaded": False, "physical_fjt_goals_sent": 0, "robot_motion_started": False, "authority": authority or {}})
    write_json(output / "upstream_immutability_report.json", upstream or {"status": "BLOCKED", "first_blocker": blocker, "groups": {"H7_EVIDENCE": False, "H8_R_EVIDENCE": False, "HISTORICAL_H8_EVIDENCE": False, "H9_EVIDENCE": False, "H10_EVIDENCE": False}, "records": []})
    for name in ("h11_dataset_contract.json", "h11_window_manifest.json", "window_boundary_audit.json", "normalization_stats.json", "training_config.json", "training_history.json", "deterministic_replay.json", "baseline_comparison.json", "train_metrics.json", "validation_metrics.json", "test_metrics.json", "generalization_metrics.json", "rollout_metrics.json", "raw_prediction_constraint_report.json", "native_prediction_validation.json"):
        if not (output / name).is_file():
            write_json(output / name, {"schema_version": "stage3_h11_blocked_artifact_v1", "status": "BLOCKED", "FIRST_BLOCKER": blocker})
    if not (output / "training_history.csv").is_file():
        (output / "training_history.csv").write_text("status,first_blocker\nBLOCKED," + blocker + "\n", encoding="utf-8", newline="\n")
    if not (output / "model_summary.txt").is_file():
        (output / "model_summary.txt").write_text(f"STATUS: BLOCKED\nFIRST_BLOCKER: {blocker}\n", encoding="utf-8", newline="\n")
    if not (output / "checkpoint_sha256_manifest.json").is_file():
        write_json(output / "checkpoint_sha256_manifest.json", {"status": "BLOCKED", "FIRST_BLOCKER": blocker, "checkpoint_sha256": None})
    if not (output / "test_results.txt").is_file():
        (output / "test_results.txt").write_text("NOT_RUN: H11 blocked before focused tests\n", encoding="utf-8", newline="\n")
    if not (output / "regression_results.txt").is_file():
        (output / "regression_results.txt").write_text("NOT_RUN: H11 blocked before regression tests\n", encoding="utf-8", newline="\n")
    certificate = certificate_payload(blocker, authority or {}, upstream or {}, values or {})
    gate = {"schema_version": "stage3_h11_gate_report_v1", **certificate, "required": {}, "blockers": [blocker]}
    write_json(output / "stage3_h11_terminal_certificate.json", certificate)
    write_json(output / "stage3_h11_gate_report.json", gate)
    (output / "FINAL_REPORT.md").write_text(report_text(certificate, blocker, {}), encoding="utf-8", newline="\n")
    write_frozen_artifact_manifest(output)


def certificate_payload(blocker: str | None, authority: Mapping[str, Any], upstream: Mapping[str, Any], values: Mapping[str, Any]) -> dict[str, Any]:
    terminal = authority.get("terminal", {}) if authority else {}
    native = values.get("native", {})
    metrics = values.get("metrics", {})
    return {
        "schema_version": "stage3_h11_terminal_certificate_v1", "STAGE_3_H11": "PASSED" if blocker is None else "BLOCKED", "FIRST_BLOCKER": blocker or "none",
        "H7_IMMUTABLE": "YES" if upstream.get("groups", {}).get("H7_EVIDENCE") else "NO", "H8_R_IMMUTABLE": "YES" if upstream.get("groups", {}).get("H8_R_EVIDENCE") else "NO", "HISTORICAL_H8_IMMUTABLE": "YES" if upstream.get("groups", {}).get("HISTORICAL_H8_EVIDENCE") else "NO", "H9_IMMUTABLE": "YES" if upstream.get("groups", {}).get("H9_EVIDENCE") else "NO", "H10_IMMUTABLE": "YES" if upstream.get("groups", {}).get("H10_EVIDENCE") else "NO",
        "SOURCE_H10": "PASSED" if authority.get("status") == "PASSED" else "FAILED", "SOURCE_H10_SEMANTIC_SHA256": authority.get("semantic", {}).get("semantic_dataset_sha256", terminal.get("DATASET_SEMANTIC_SHA256")), "H10_SEMANTIC_HASH_MATCH": "YES" if authority.get("semantic", {}).get("semantic_dataset_sha256") == EXPECTED_H10_SEMANTIC_SHA256 else "NO",
        "TRAIN_FAMILIES": 31, "VALIDATION_FAMILIES": 7, "TEST_FAMILIES": 6, "GENERALIZATION_FAMILIES": 4, "GROUP_LEAKAGE_VIOLATIONS": int(terminal.get("GROUP_LEAKAGE_VIOLATIONS", 0) or 0), "NORMALIZATION_LEAKAGE_VIOLATIONS": int(values.get("normalization_leakage", 0) or 0), "CROSS_FAMILY_WINDOWS": int(values.get("window_audit", {}).get("CROSS_FAMILY_WINDOWS", 0) or 0),
        "MODEL_TYPE": "causal_gru", "INPUT_HISTORY": INPUT_HISTORY, "PREDICTION_HORIZON": PREDICTION_HORIZON, "TRAINABLE_PARAMETERS": values.get("parameter_count"), "SELECTED_EPOCH": values.get("selected_epoch"), "CHECKPOINT_SHA256": values.get("checkpoint_sha256", values.get("checkpoint_hash")),
        "TRAIN_RMSE_RAD": metrics.get("TRAIN", {}).get("joint_position_rmse_rad"), "VALIDATION_RMSE_RAD": metrics.get("VALIDATION", {}).get("joint_position_rmse_rad"), "TEST_RMSE_RAD": metrics.get("TEST", {}).get("joint_position_rmse_rad"), "GENERALIZATION_RMSE_RAD": metrics.get("GENERALIZATION", {}).get("joint_position_rmse_rad"),
        "HOLD_LAST_TEST_RMSE_RAD": values.get("baseline", {}).get("hold_last_test_rmse_rad", values.get("baseline", {}).get("hold_last_position", {}).get("joint_position_rmse_rad")), "CONST_VELOCITY_TEST_RMSE_RAD": values.get("baseline", {}).get("constant_velocity_test_rmse_rad", values.get("baseline", {}).get("constant_velocity_extrapolation", {}).get("joint_position_rmse_rad")), "MODEL_BEATS_NAIVE_BASELINES": "YES" if values.get("model_beats_baselines") else "NO",
        "RAW_MODEL_HARD_CONSTRAINT_VIOLATIONS": values.get("raw_constraints", {}).get("RAW_MODEL_HARD_CONSTRAINT_VIOLATIONS"), "RAW_MODEL_COLLISION_VIOLATIONS": native.get("RAW_MODEL_COLLISION_VIOLATIONS"), "RAW_MODEL_PROCESS_TOLERANCE_VIOLATIONS": native.get("RAW_MODEL_PROCESS_TOLERANCE_VIOLATIONS"),
        "TRAINING_REPLAY": values.get("training_replay", "0/3"), "DETERMINISTIC_TRAINING": "YES" if values.get("deterministic_training") else "NO", "CCD_AVAILABLE": "NO", "CLEARANCE_AVAILABLE": "NO", "COLLISION_METHOD": "adaptive_discrete_interpolation",
        "PHYSICAL_ROBOT_CONNECTED": "NO", "PHYSICAL_DRIVER_LOADED": "NO", "PHYSICAL_FJT_GOALS_SENT": 0, "ROBOT_MOTION_STARTED": "NO", "FOCUSED_TESTS": values.get("focused_tests", "BLOCKED"), "REGRESSION": values.get("regression", "BLOCKED"), "READY_FOR_STAGE_3_H12": "YES" if blocker is None else "NO",
    }


def report_text(certificate: Mapping[str, Any], blocker: str | None, values: Mapping[str, Any]) -> str:
    lines = ["# Stage 3 H11 — Deterministic Deep-Learning Trajectory Baseline", "", "```text"]
    keys = ["STAGE_3_H11", "FIRST_BLOCKER", "H7_IMMUTABLE", "H8_R_IMMUTABLE", "HISTORICAL_H8_IMMUTABLE", "H9_IMMUTABLE", "H10_IMMUTABLE", "SOURCE_H10", "H10_SEMANTIC_HASH_MATCH", "TRAIN_FAMILIES", "VALIDATION_FAMILIES", "TEST_FAMILIES", "GENERALIZATION_FAMILIES", "GROUP_LEAKAGE_VIOLATIONS", "NORMALIZATION_LEAKAGE_VIOLATIONS", "CROSS_FAMILY_WINDOWS", "MODEL_TYPE", "INPUT_HISTORY", "PREDICTION_HORIZON", "TRAINABLE_PARAMETERS", "SELECTED_EPOCH", "TEST_RMSE_RAD", "HOLD_LAST_TEST_RMSE_RAD", "CONST_VELOCITY_TEST_RMSE_RAD", "MODEL_BEATS_NAIVE_BASELINES", "GENERALIZATION_RMSE_RAD", "RAW_MODEL_HARD_CONSTRAINT_VIOLATIONS", "RAW_MODEL_COLLISION_VIOLATIONS", "RAW_MODEL_PROCESS_TOLERANCE_VIOLATIONS", "TRAINING_REPLAY", "DETERMINISTIC_TRAINING", "COLLISION_METHOD", "CCD_AVAILABLE", "CLEARANCE_AVAILABLE", "PHYSICAL_ROBOT_CONNECTED", "PHYSICAL_DRIVER_LOADED", "PHYSICAL_FJT_GOALS_SENT", "ROBOT_MOTION_STARTED", "FOCUSED_TESTS", "REGRESSION", "READY_FOR_STAGE_3_H12"]
    lines.extend(f"{key}: {certificate.get(key)}" for key in keys)
    lines.extend(["```", "", "H11 is a software-only statistical baseline trained from the certified H10 offline trajectory distribution. A PASS does not imply physical deployment, collision-free neural output, real-world generalization, cross-robot generalization, or autonomous operation.", "", "GENERALIZATION is reported as within-scenario held-out-family generalization only. actual_tcp_position remains unavailable/null; CCD is not available and clearance is null unless a native backend explicitly reports otherwise.", ""])
    return "\n".join(lines)


def train_once(output: Path, h10_root: Path, seed: int, child: bool = False) -> dict[str, Any]:
    authority = verify_h10_authority(ROOT, h10_root)
    upstream = snapshot_frozen_upstream(ROOT, h10_root)
    if authority.get("status") != "PASSED":
        raise RuntimeError(str(authority.get("first_blocker") or "h10_authoritative_certificate_mismatch"))
    dataset = load_h10_segments(h10_root)
    stats = __import__("src.stage3_h11_dataset", fromlist=["compute_normalization_stats"]).compute_normalization_stats(dataset)
    windows, window_audit, window_counts = make_windows(dataset, output / "h11_window_manifest.json")
    write_json(output / "window_boundary_audit.json", window_audit)
    write_json(output / "normalization_stats.json", stats)
    write_json(output / "h11_dataset_contract.json", contract_payload(dataset, authority, stats, window_audit, window_counts))
    norm_leakage = stats_leakage(stats, dataset)
    window_hash = window_semantic_hash(windows)
    window_manifest_path = output / "h11_window_manifest.json"
    window_manifest = json.loads(window_manifest_path.read_text(encoding="utf-8"))
    window_manifest["semantic_sha256"] = window_hash
    write_json(window_manifest_path, window_manifest)
    if window_audit.get("CROSS_FAMILY_WINDOWS") != 0 or norm_leakage != 0:
        raise RuntimeError("dataset_split_contamination" if window_audit.get("CROSS_FAMILY_WINDOWS") else "normalization_leakage")
    write_json(output / "training_config.json", {"schema_version": "stage3_h11_training_config_v1", **MODEL_CONFIG, "seed": seed, "optimizer": {"type": "Adam", "learning_rate": MODEL_CONFIG["learning_rate"], "weight_decay": MODEL_CONFIG["weight_decay"]}, "device": "cuda" if TORCH_AVAILABLE and __import__("torch").cuda.is_available() else "cpu", "deterministic": set_deterministic(seed), "window_manifest_semantic_sha256": window_hash})
    if not TORCH_AVAILABLE:
        raise RuntimeError("pytorch_unavailable")
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_deterministic(seed)
    train_arrays = materialize_windows(dataset, windows, stats, "TRAIN")
    validation_arrays = materialize_windows(dataset, windows, stats, "VALIDATION")
    model = CausalGRUTrajectoryPredictor(input_size=len(FEATURE_NAMES), hidden_size=MODEL_CONFIG["hidden_size"], num_layers=MODEL_CONFIG["num_layers"], horizon=PREDICTION_HORIZON)
    model, history, selected_epoch, checkpoint_bytes = train_model(model, train_arrays, validation_arrays, MODEL_CONFIG, device=device)
    checkpoint_path = output / "final_model.pt"
    checkpoint_path.write_bytes(checkpoint_bytes)
    checkpoint_hash = hashlib.sha256(checkpoint_bytes).hexdigest()
    write_json(output / "checkpoint_sha256_manifest.json", {"schema_version": "stage3_h11_checkpoint_sha256_v1", "checkpoint": checkpoint_path.name, "sha256": checkpoint_hash, "size_bytes": len(checkpoint_bytes), "selected_epoch": selected_epoch})
    write_json(output / "training_history.json", {"schema_version": "stage3_h11_training_history_v1", "history": history, "selected_epoch": selected_epoch, "selection_metric": "validation_joint_position_rmse_rad", "selection_split": "VALIDATION"})
    write_history_csv(output / "training_history.csv", history)
    (output / "model_summary.txt").write_text(f"model_type: causal_gru\ninput_features: {len(FEATURE_NAMES)}\ninput_history: {INPUT_HISTORY}\nprediction_horizon: {PREDICTION_HORIZON}\nhidden_size: 128\nnum_layers: 2\nbidirectional: false\ntrainable_parameters: {parameter_count(model)}\nselected_epoch: {selected_epoch}\n", encoding="utf-8", newline="\n")
    # TEST/GENERALIZATION arrays are materialized only after the selected
    # validation checkpoint is frozen.
    arrays_by_role: dict[str, WindowArrays] = {"TRAIN": train_arrays, "VALIDATION": validation_arrays}
    metrics: dict[str, dict[str, Any]] = {}
    direct_predictions: dict[str, np.ndarray] = {}
    for role in ("TRAIN", "VALIDATION"):
        direct_predictions[role], metrics[role] = evaluate_model(model, arrays_by_role[role], device=device, batch_size=MODEL_CONFIG["eval_batch_size"])
    for role in ("TEST", "GENERALIZATION"):
        arrays_by_role[role] = materialize_windows(dataset, windows, stats, role)
        direct_predictions[role], metrics[role] = evaluate_model(model, arrays_by_role[role], device=device, batch_size=MODEL_CONFIG["eval_batch_size"])
    for role in ("TRAIN", "VALIDATION", "TEST", "GENERALIZATION"):
        write_json(output / (role.lower() + "_metrics.json"), {"schema_version": "stage3_h11_metrics_v1", "split_role": role, "family_count": len({str(x) for x in arrays_by_role[role].family_ids.tolist()}), **metrics[role]})
    hold = hold_last_prediction(arrays_by_role["TEST"])
    const = constant_velocity_prediction(arrays_by_role["TEST"])
    baseline = {"schema_version": "stage3_h11_baseline_comparison_v1", "primary_metric": "joint_position_rmse_rad over direct 8-step target", "hold_last_position": metric_payload(hold, arrays_by_role["TEST"].target_positions), "constant_velocity_extrapolation": metric_payload(const, arrays_by_role["TEST"].target_positions), "neural_model": metrics["TEST"]}
    baseline["model_beats_hold_last"] = bool(metrics["TEST"]["joint_position_rmse_rad"] < baseline["hold_last_position"]["joint_position_rmse_rad"])
    baseline["model_beats_constant_velocity"] = bool(metrics["TEST"]["joint_position_rmse_rad"] < baseline["constant_velocity_extrapolation"]["joint_position_rmse_rad"])
    baseline["MODEL_BEATS_NAIVE_BASELINES"] = "YES" if baseline["model_beats_hold_last"] and baseline["model_beats_constant_velocity"] else "NO"
    write_json(output / "baseline_comparison.json", baseline)
    rollout_predictions: dict[str, np.ndarray] = {}
    rollout_report: dict[str, Any] = {"schema_version": "stage3_h11_rollout_metrics_v1", "method": "open_loop_recursive_one_step; predicted positions/finite-difference dynamics fed back into the next input", "horizons": [1, 2, 4, 8], "roles": {}}
    for role, arrays in arrays_by_role.items():
        rollout = rollout_one_step(model, arrays, stats["channels"], device=device, batch_size=MODEL_CONFIG["eval_batch_size"])
        rollout_predictions[role] = rollout
        rollout_report["roles"][role] = {"window_count": len(rollout), "family_count": len({str(x) for x in arrays.family_ids.tolist()}), "horizon_metrics": {str(h): metric_payload(rollout[:, h - 1:h, :], arrays.target_positions[:, h - 1:h, :]) for h in (1, 2, 4, 8)}, "velocity_acceleration_diagnostics": finite_difference_diagnostics(rollout, arrays.target_positions, arrays.target_times)}
    write_json(output / "rollout_metrics.json", rollout_report)
    raw_constraints = raw_constraint_report(rollout_predictions, arrays_by_role)
    write_json(output / "raw_prediction_constraint_report.json", raw_constraints)
    subset = native_subset_rows(windows, arrays_by_role["TEST"], rollout_predictions["TEST"]) + native_subset_rows(windows, arrays_by_role["GENERALIZATION"], rollout_predictions["GENERALIZATION"])
    native = run_native_validation(output, h10_root, subset)
    write_json(output / "native_prediction_validation.json", native)
    # Fresh process replay is done only by the parent run.  Child runs return
    # a compact semantic record and do not recursively launch more children.
    summary = {"status": "PASSED", "window_manifest_semantic_sha256": window_hash, "selected_epoch": selected_epoch, "checkpoint_sha256": checkpoint_hash, "evaluation_counts": {role: int(len(arrays.inputs)) for role, arrays in arrays_by_role.items()}, "test_rmse_rad": metrics["TEST"]["joint_position_rmse_rad"], "hold_last_test_rmse_rad": baseline["hold_last_position"]["joint_position_rmse_rad"], "constant_velocity_test_rmse_rad": baseline["constant_velocity_extrapolation"]["joint_position_rmse_rad"], "parameter_count": parameter_count(model)}
    write_json(output / "replay_summary.json", summary)
    return {"authority": authority, "upstream": upstream, "stats": stats, "windows": windows, "window_audit": window_audit, "window_counts": window_counts, "window_hash": window_hash, "metrics": metrics, "baseline": baseline, "raw_constraints": raw_constraints, "native": native, "selected_epoch": selected_epoch, "checkpoint_hash": checkpoint_hash, "parameter_count": parameter_count(model), "summary": summary}


def run_replays(output: Path, h10_root: Path, seed: int) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    script = Path(__file__).resolve()
    for index in range(1, 4):
        replay_dir = output / f"training_replay_{index}"
        command = [sys.executable, str(script), "--replay-child", "--output-dir", str(replay_dir), "--h10-root", str(h10_root), "--seed", str(seed)]
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
        summary_path = replay_dir / "replay_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
        records.append({"replay_index": index, "returncode": completed.returncode, "summary": summary, "stdout_tail": (completed.stdout or "")[-1000:], "stderr_tail": (completed.stderr or "")[-1000:]})
    valid = [record for record in records if record["returncode"] == 0 and record["summary"].get("status") == "PASSED"]
    semantic_ok = len(valid) == 3 and len({record["summary"].get("window_manifest_semantic_sha256") for record in valid}) == 1
    epoch_ok = len(valid) == 3 and len({record["summary"].get("selected_epoch") for record in valid}) == 1
    counts_ok = len(valid) == 3 and len({json.dumps(record["summary"].get("evaluation_counts"), sort_keys=True) for record in valid}) == 1
    metric_values = [record["summary"].get("test_rmse_rad") for record in valid]
    metric_ok = len(metric_values) == 3 and len(set(metric_values)) == 1
    checkpoint_values = [record["summary"].get("checkpoint_sha256") for record in valid]
    byte_ok = len(checkpoint_values) == 3 and len(set(checkpoint_values)) == 1
    payload = {"schema_version": "stage3_h11_deterministic_replay_v1", "TRAINING_REPLAY": f"{len(valid)}/3", "semantic_determinism": semantic_ok and epoch_ok and counts_ok, "metric_determinism": metric_ok, "byte_level_checkpoint_determinism": byte_ok, "metric_values_test_rmse_rad": metric_values, "replays": records}
    write_json(output / "deterministic_replay.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--h10-root", type=Path, default=ROOT / "outputs/stage3_h10_multi_trajectory_dataset_20260811T081123Z")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--replay-child", action="store_true")
    args = parser.parse_args()
    output = args.output_dir
    if output is None:
        output = ROOT / "outputs" / f"stage3_h11_deep_learning_baseline_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing_to_overwrite_nonempty_output:{output}")
    output.mkdir(parents=True, exist_ok=True)
    authority = verify_h10_authority(ROOT, args.h10_root.resolve())
    upstream = snapshot_frozen_upstream(ROOT, args.h10_root.resolve())
    if authority.get("status") != "PASSED":
        blocker = str(authority.get("first_blocker") or "h10_authoritative_certificate_mismatch")
        write_blocked_artifacts(output, blocker, authority, upstream)
        print((output / "FINAL_REPORT.md").read_text(encoding="utf-8"))
        return 2
    try:
        result = train_once(output, args.h10_root.resolve(), args.seed, child=args.replay_child)
        if args.replay_child:
            print(json.dumps(result["summary"], sort_keys=True))
            return 0
        replay = run_replays(output, args.h10_root.resolve(), args.seed)
        focus_code = run_command([sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h11.py"], output / "test_results.txt")
        regression_code = run_command([sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h10.py", "tests/test_stage3_h9.py", "tests/test_stage3_h8_software_only.py"], output / "regression_results.txt")
        deterministic = bool(replay.get("TRAINING_REPLAY") == "3/3" and replay.get("semantic_determinism") and replay.get("metric_determinism"))
        beats = bool(result["baseline"].get("MODEL_BEATS_NAIVE_BASELINES") == "YES")
        native_ok = result["native"].get("status") == "PASSED"
        blockers = []
        if not deterministic: blockers.append("deterministic_replay_failed")
        if not beats: blockers.append("model_does_not_beat_naive_baselines")
        if result["raw_constraints"].get("RAW_MODEL_HARD_CONSTRAINT_VIOLATIONS") is None: blockers.append("required_metrics_missing")
        if not native_ok: blockers.append(str(result["native"].get("first_blocker") or "native_prediction_validation_unavailable"))
        if focus_code != 0: blockers.append("focused_tests_failed")
        if regression_code != 0: blockers.append("regression_failure")
        blocker = blockers[0] if blockers else None
        values = {**result, "training_replay": replay.get("TRAINING_REPLAY"), "deterministic_training": deterministic, "model_beats_baselines": beats, "focused_tests": "PASSED" if focus_code == 0 else "FAILED", "regression": "PASSED" if regression_code == 0 else "FAILED", "normalization_leakage": 0}
        certificate = certificate_payload(blocker, result["authority"], result["upstream"], values)
        write_json(output / "run_metadata.json", {"schema_version": "stage3_h11_run_metadata_v1", "generated_at": now_utc(), "seed": args.seed, "python_version": platform.python_version(), "platform": platform.platform(), "environment": environment_metadata(__import__("torch")), "device": "cuda" if __import__("torch").cuda.is_available() else "cpu", "software_only": {"physical_robot_connected": False, "physical_driver_loaded": False, "physical_fjt_goals_sent": 0, "robot_motion_started": False}, "source_h10": rel(args.h10_root.resolve()), "window_manifest_semantic_sha256": result["window_hash"]})
        gate = {"schema_version": "stage3_h11_gate_report_v1", **certificate, "required": {"authority": result["authority"].get("status") == "PASSED", "upstream_immutable": result["upstream"].get("status") == "PASSED", "normalization_leakage": values["normalization_leakage"] == 0, "cross_family_windows": result["window_audit"].get("CROSS_FAMILY_WINDOWS") == 0, "training_replay": deterministic, "model_beats_naive_baselines": beats, "native_prediction_validation": native_ok, "focused_tests": focus_code == 0, "regression": regression_code == 0}, "blockers": blockers}
        write_json(output / "stage3_h11_terminal_certificate.json", certificate)
        write_json(output / "stage3_h11_gate_report.json", gate)
        (output / "FINAL_REPORT.md").write_text(report_text(certificate, blocker, values), encoding="utf-8", newline="\n")
        write_frozen_artifact_manifest(output)
        print(report_text(certificate, blocker, values).split("```", 2)[1].strip())
        return 0 if blocker is None else 2
    except Exception as exc:
        blocker = str(exc)
        focus_code = run_command([sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h11.py"], output / "test_results.txt")
        regression_code = run_command([sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h10.py", "tests/test_stage3_h9.py", "tests/test_stage3_h8_software_only.py"], output / "regression_results.txt")
        write_blocked_artifacts(output, blocker, authority, upstream, {"focused_tests": "PASSED" if focus_code == 0 else "FAILED", "regression": "PASSED" if regression_code == 0 else "FAILED"})
        print((output / "FINAL_REPORT.md").read_text(encoding="utf-8"))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
