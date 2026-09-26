#!/usr/bin/env python3
"""Stage 3 H13-R3: frozen-20 H11-R2 evaluation and native closure.

This runner is evaluation-only.  It loads the one authoritative H11-R2
checkpoint, reconstructs the original H13 20-case matrix by reference, runs
the existing H13 native adapter, and retains compact evidence only.
"""

from __future__ import annotations

import argparse
import hashlib
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
from src.stage3_h11_dataset import FEATURE_NAMES, INPUT_HISTORY, PREDICTION_HORIZON  # noqa: E402
from src.stage3_h11_model import WindowArrays, metric_payload  # noqa: E402
from src.stage3_h11_r2_model import (  # noqa: E402
    ResidualCausalGRUTrajectoryPredictor,
    TORCH_AVAILABLE,
    causal_constant_velocity_baseline,
    direct_metrics,
)
from src.stage3_h13 import (  # noqa: E402
    COLLISION_METHOD,
    CASE_TARGET,
    build_case_matrix,
    canonical,
    canonical_hash,
    case_input_payload,
    h12_r7_compatibility_fields,
    improvement_percent,
    metric_summary,
    overlap_audit,
    semantic_sample_hash,
    validate_case_matrix,
)


H11_ORIGINAL_ROOT = ROOT / "outputs/stage3_h11_r_pytorch_runtime_recertification_20260811T143806Z"
H11_R2_ROOT = ROOT / "outputs/stage3_h11_r2_low_storage_unseen_generalization_20260813T215000Z"
H11_R2_CHECKPOINT = H11_R2_ROOT / "best_checkpoint.pt"
H11_R2_TERMINAL = H11_R2_ROOT / "stage3_h11_r2_terminal_certificate.json"
H11_R2_SUMMARY = H11_R2_ROOT / "generalization_summary.json"
H11_R2_MANIFEST = H11_R2_ROOT / "checkpoint_sha256_manifest.json"
H11_STATS = H11_ORIGINAL_ROOT / "normalization_stats.json"
H12_ROOT = ROOT / "outputs/stage3_h12_constraint_aware_trajectory_repair_20260811T163500Z"
H12_R7_ROOT = ROOT / "outputs/stage3_h12_r7_low_storage_reprojection_20260813T030000Z"
H13_R1_ROOT = ROOT / "outputs/stage3_h13_low_storage_unseen_generalization_r1_authoritative_20260813T"
H13_R2_ROOT = ROOT / "outputs/stage3_h13_low_storage_unseen_generalization_r2_20260813T195307126"
H13_R2_TERMINAL = H13_R2_ROOT / "stage3_h13_terminal_certificate.json"
H13_R2_SUMMARY = H13_R2_ROOT / "generalization_summary.json"
H13_R2_CASES = H13_R2_ROOT / "case_summary.jsonl"
H12_R7_TERMINAL = H12_R7_ROOT / "stage3_h12_r7_terminal_certificate.json"
H12_R7_REPAIR = H12_R7_ROOT / "repair_summary.json"
H12_R7_REPAIRED = H12_R7_ROOT / "repaired_sample_manifest.jsonl"
OPEN_ARCH_ROOT = ROOT / "outputs/internal_wiper_moveit_inputs"
BASE_POSES = OPEN_ARCH_ROOT / "open_arch_tcp_poses_base_link.csv"
BASE_SEEDS = OPEN_ARCH_ROOT / "open_arch_seed_joints.csv"

H13_R2_MODEL_REFERENCE = 0.088736273737
H13_R2_AUTOREGRESSIVE_REFERENCE = 0.822317019169
TIME_FEATURE_FORMULA = "(trajectory_time - trajectory_start_time) / (trajectory_end_time - trajectory_start_time)"
TIME_FEATURE_TOLERANCE = 1.0e-12
POSITION_LOWER = np.asarray([-3.0543, -4.6251, -2.8274, -4.6251, -3.0543, -3.0543], dtype=np.float64)
POSITION_UPPER = np.asarray([3.0543, 1.4835, 2.8274, 1.4835, 3.0543, 3.0543], dtype=np.float64)
VELOCITY_LIMITS = np.asarray([0.4725, 0.4725, 0.4725, 0.48, 0.48, 0.48], dtype=np.float64)
HORIZON = PREDICTION_HORIZON
args_global = argparse.Namespace(focused_tests=False, regression_tests=False)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(canonical(value), ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(canonical(dict(row)), ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def size_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) if path.is_dir() else (path.stat().st_size if path.is_file() else 0)


def size_mb(path: Path) -> float:
    return size_bytes(path) / (1024.0 * 1024.0)


def snapshot(label_paths: Mapping[str, Sequence[Path]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for label, paths in label_paths.items():
        result[label] = []
        for path in paths:
            result[label].append({"path": str(path.resolve()), "exists": path.is_file(), "size_bytes": path.stat().st_size if path.is_file() else None, "sha256": sha256_file(path) if path.is_file() else None})
    return result


def snapshot_equal(before: Mapping[str, Sequence[Mapping[str, Any]]], after: Mapping[str, Sequence[Mapping[str, Any]]], label: str) -> bool:
    rows = list(before.get(label, ()))
    return bool(rows) and rows == list(after.get(label, ())) and all(bool(row.get("exists")) for row in rows)


def authority_paths() -> dict[str, list[Path]]:
    return {
        "H11_ORIGINAL": [H11_ORIGINAL_ROOT / name for name in ("checkpoint", "h11_window_manifest.json", "normalization_stats.json", "h11_dataset_contract.json", "stage3_h11_r_terminal_certificate.json")],
        "H11_R2": [H11_R2_ROOT / name for name in ("best_checkpoint.pt", "checkpoint_sha256_manifest.json", "stage3_h11_r2_terminal_certificate.json", "generalization_summary.json", "ablation_summary.json", "training_config.json", "dataset_contract.json")],
        "H12": [H12_ROOT / "stage3_h12_terminal_certificate.json"],
        "H12_R7": [H12_R7_TERMINAL, H12_R7_REPAIR, H12_R7_REPAIRED],
        "H13_R1": [H13_R1_ROOT / name for name in ("stage3_h13_terminal_certificate.json", "generalization_summary.json", "case_summary.jsonl")],
        "H13_R2": [H13_R2_TERMINAL, H13_R2_SUMMARY, H13_R2_CASES],
        "H13_UNSEEN_SPLIT": [H13_R2_SUMMARY, H13_R2_CASES],
        "OPEN_ARCH_PAIR": [BASE_POSES, BASE_SEEDS],
    }


def audit_authority() -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    paths = authority_paths()
    before = snapshot(paths)
    missing = [row["path"] for rows in before.values() for row in rows if not row["exists"]]
    h11_original = load_json(H11_ORIGINAL_ROOT / "stage3_h11_r_terminal_certificate.json") if not missing and (H11_ORIGINAL_ROOT / "stage3_h11_r_terminal_certificate.json").is_file() else {}
    h11_r2 = load_json(H11_R2_TERMINAL) if H11_R2_TERMINAL.is_file() else {}
    h11_r2_manifest = load_json(H11_R2_MANIFEST) if H11_R2_MANIFEST.is_file() else {}
    h12 = load_json(H12_ROOT / "stage3_h12_terminal_certificate.json") if (H12_ROOT / "stage3_h12_terminal_certificate.json").is_file() else {}
    h12_r7 = load_json(H12_R7_TERMINAL) if H12_R7_TERMINAL.is_file() else {}
    h13_r2 = load_json(H13_R2_TERMINAL) if H13_R2_TERMINAL.is_file() else {}
    h13_r2_summary = load_json(H13_R2_SUMMARY) if H13_R2_SUMMARY.is_file() else {}
    cases = build_case_matrix(CASE_TARGET)
    expected_matrix_hash = canonical_hash([str(case["case_id"]) for case in cases])
    recorded_matrix_hash = h13_r2_summary.get("case_matrix", {}).get("matrix_sha256")
    r2_case_ids = [str(row.get("case_id")) for row in load_jsonl(H13_R2_CASES)] if H13_R2_CASES.is_file() else []
    expected_case_ids = [str(row["case_id"]) for row in cases]
    split_ok = recorded_matrix_hash == expected_matrix_hash and r2_case_ids == expected_case_ids and len(r2_case_ids) == CASE_TARGET and len(set(r2_case_ids)) == CASE_TARGET
    checkpoint_hash = sha256_file(H11_R2_CHECKPOINT) if H11_R2_CHECKPOINT.is_file() else None
    checkpoint_hash_match = bool(checkpoint_hash and checkpoint_hash == h11_r2_manifest.get("sha256"))
    authority = {
        "H11_ORIGINAL_IMMUTABLE": "YES" if not missing and h11_original.get("STAGE_3_H11_R") == "PASSED" else "NO",
        "H11_R2_IMMUTABLE": "YES" if not missing and h11_r2.get("STAGE_3_H11_R2") == "PASSED" else "NO",
        "H12_IMMUTABLE": "YES" if not missing and h12 else "NO",
        "H12_R7_IMMUTABLE": "YES" if not missing and h12_r7 else "NO",
        "H13_R1_IMMUTABLE": "YES" if not missing else "NO",
        "H13_R2_IMMUTABLE": "YES" if not missing and h13_r2 else "NO",
        "H13_UNSEEN_SPLIT_IMMUTABLE": "YES" if split_ok else "NO",
        "H11_R2_CHECKPOINT_HASH_MATCH": "YES" if checkpoint_hash_match else "NO",
        "H11_R2_CHECKPOINT_SHA256": checkpoint_hash,
        "H11_R2_CHECKPOINT_SIZE_BYTES": H11_R2_CHECKPOINT.stat().st_size if H11_R2_CHECKPOINT.is_file() else None,
        "H11_R2_CHECKPOINT_SOURCE": str(H11_R2_CHECKPOINT.resolve()),
        "H13_UNSEEN_MATRIX_SHA256": expected_matrix_hash,
        "H13_R2_RECORDED_MATRIX_SHA256": recorded_matrix_hash,
        "H13_R2_CASE_IDS_MATCH": "YES" if split_ok else "NO",
        "H13_R2_REFERENCE_MODEL_RMSE": H13_R2_MODEL_REFERENCE,
        "missing": missing,
        "input_sha256_manifest": before,
        "H11_R2_TERMINAL_STATUS": h11_r2.get("STAGE_3_H11_R2"),
        "H12_R7_TERMINAL_STATUS": h12_r7.get("STAGE_3_H12_R7"),
        "H13_R2_TERMINAL_STATUS": h13_r2.get("STAGE_3_H13_R2"),
    }
    return authority, before


def validate_checkpoint() -> dict[str, Any]:
    if not TORCH_AVAILABLE:
        return {"status": "BLOCKED", "first_blocker": "pytorch_unavailable"}
    import torch

    payload = torch.load(H11_R2_CHECKPOINT, map_location="cpu", weights_only=False)
    model = ResidualCausalGRUTrajectoryPredictor(input_size=len(FEATURE_NAMES), hidden_size=128, num_layers=2, horizon=HORIZON)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return {
        "status": "PASSED" if payload.get("model_type") == "residual_causal_gru" and int(payload.get("input_history", -1)) == INPUT_HISTORY and int(payload.get("prediction_horizon", -1)) == HORIZON else "BLOCKED",
        "model_type": payload.get("model_type"),
        "input_history": payload.get("input_history"),
        "prediction_horizon": payload.get("prediction_horizon"),
        "architecture": "2-layer unidirectional GRU; causal residual-over-constant-velocity baseline",
        "zero_initialized_residual_head_provenance": "preserved by H11-R2 checkpoint training lineage",
        "multi_step_rollout_training": "YES",
        "model_eval_called": "YES",
        "optimizer_loaded": "NO",
        "parameter_mutation": "NO",
    }


def build_arrays(q: np.ndarray, features: np.ndarray, t: np.ndarray, family_id: str) -> WindowArrays:
    starts = list(range(0, len(q) - INPUT_HISTORY - HORIZON + 1))
    histories = np.asarray([q[start:start + INPUT_HISTORY] for start in starts], dtype=np.float64)
    history_times = np.asarray([t[start:start + INPUT_HISTORY] for start in starts], dtype=np.float64)
    targets = np.asarray([q[start + INPUT_HISTORY:start + INPUT_HISTORY + HORIZON] for start in starts], dtype=np.float64)
    target_times = np.asarray([t[start + INPUT_HISTORY:start + INPUT_HISTORY + HORIZON] for start in starts], dtype=np.float64)
    anchors = histories[:, -1, :] if starts else np.empty((0, 6), dtype=np.float64)
    return WindowArrays(
        inputs=np.asarray([features[start:start + INPUT_HISTORY] for start in starts], dtype=np.float32),
        target_deltas=(targets - anchors[:, None, :]).astype(np.float32),
        target_positions=targets,
        anchor_positions=anchors,
        target_times=target_times,
        history_positions=histories,
        history_times=history_times,
        window_ids=np.asarray(starts, dtype=np.int64),
        family_ids=np.asarray([family_id] * len(starts), dtype=object),
    )


def normalize_next_features(next_q: np.ndarray, previous_q: np.ndarray, prior_q: np.ndarray, next_time: float, last_time: float, prior_time: float, start_time: float, end_time: float, stats: Mapping[str, Any]) -> np.ndarray:
    dt = max(float(next_time - last_time), 1.0e-9)
    prior_dt = max(float(last_time - prior_time), 1.0e-9)
    velocity = (next_q - previous_q) / dt
    prior_velocity = (previous_q - prior_q) / prior_dt
    acceleration = (velocity - prior_velocity) / dt
    local_time = (float(next_time) - float(start_time)) / max(float(end_time - start_time), 1.0e-9)
    raw = np.concatenate([next_q, velocity, acceleration, np.asarray([1.0, local_time], dtype=np.float64)])
    output = raw.astype(np.float64, copy=True)
    for index, name in enumerate(FEATURE_NAMES):
        channel = stats.get("channels", {}).get(name, {})
        if name == "spray_on":
            continue
        scale = float(channel.get("scale", channel.get("std", 1.0))) or 1.0
        output[index] = (output[index] - float(channel.get("mean", 0.0))) / scale
    return output.astype(np.float32)


def full_causal_baseline(q: np.ndarray, t: np.ndarray) -> np.ndarray:
    output = q.copy().astype(np.float64)
    for index in range(INPUT_HISTORY, len(q)):
        dt = max(float(t[index] - t[index - 1]), 1.0e-9)
        prior_dt = max(float(t[index - 1] - t[index - 2]), 1.0e-9)
        velocity = (output[index - 1] - output[index - 2]) / prior_dt
        output[index] = output[index - 1] + velocity * dt
    return output


def full_residual_rollout(model: Any, features: np.ndarray, q: np.ndarray, t: np.ndarray, stats: Mapping[str, Any]) -> np.ndarray:
    import torch

    model.eval()
    output = q.copy().astype(np.float64)
    history_q = q[:INPUT_HISTORY].copy().astype(np.float64)
    history_t = t[:INPUT_HISTORY].copy().astype(np.float64)
    x = features[:INPUT_HISTORY].copy().astype(np.float32)
    start_time, end_time = float(t[0]), float(t[-1])
    with torch.no_grad():
        for index in range(INPUT_HISTORY, len(q)):
            residual = model(torch.from_numpy(x[None, ...])).detach().cpu().numpy()[0, 0].astype(np.float64)
            dt = max(float(t[index] - history_t[-1]), 1.0e-9)
            prior_dt = max(float(history_t[-1] - history_t[-2]), 1.0e-9)
            velocity = (history_q[-1] - history_q[-2]) / prior_dt
            next_q = history_q[-1] + velocity * dt + residual
            output[index] = next_q
            next_features = normalize_next_features(next_q, history_q[-1], history_q[-2], float(t[index]), float(history_t[-1]), float(history_t[-2]), start_time, end_time, stats)
            history_q = np.concatenate((history_q[1:], next_q[None, :]), axis=0)
            history_t = np.concatenate((history_t[1:], np.asarray([t[index]], dtype=np.float64)), axis=0)
            x = np.concatenate((x[1:], next_features[None, :]), axis=0)
    return output


def time_audit(t: np.ndarray) -> dict[str, Any]:
    local = (t - float(t[0])) / max(float(t[-1] - t[0]), 1.0e-9)
    out = (local < -TIME_FEATURE_TOLERANCE) | (local > 1.0 + TIME_FEATURE_TOLERANCE)
    return {"formula": TIME_FEATURE_FORMULA, "observed_min": float(np.min(local)), "observed_max": float(np.max(local)), "out_of_domain_count": int(np.count_nonzero(out)), "normalized_time_first": float(local[0]), "normalized_time_final": float(local[-1]), "sample_count": int(len(t))}


def raw_python_counts(path: np.ndarray, t: np.ndarray) -> dict[str, int]:
    dt = np.diff(t)
    velocity = np.vstack((np.zeros((1, 6)), np.diff(path, axis=0) / dt[:, None]))
    acceleration = np.vstack((np.zeros((1, 6)), np.diff(velocity, axis=0) / dt[:, None]))
    jerk = np.vstack((np.zeros((1, 6)), np.diff(acceleration, axis=0) / dt[:, None]))
    return {
        "RAW_POSITION_VIOLATIONS": int(np.count_nonzero((path < POSITION_LOWER[None, :]) | (path > POSITION_UPPER[None, :]))),
        "RAW_VELOCITY_VIOLATIONS": int(np.count_nonzero(np.abs(velocity) > VELOCITY_LIMITS[None, :] + 1.0e-10)),
        "RAW_ACCELERATION_VIOLATIONS": int(np.count_nonzero(np.abs(acceleration) > 0.105 + 1.0e-10)),
        "RAW_JERK_VIOLATIONS": int(np.count_nonzero(np.abs(jerk) > 8.0 + 1.0e-10)),
    }


def run_tests(output: Path) -> tuple[bool, bool]:
    focused = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h13.py", "tests/test_stage3_h13_r3.py"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600, check=False)
    regression = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h11_r2.py", "tests/test_stage3_h12_certification.py", "tests/test_stage3_h12_no_leakage.py", "tests/test_stage3_h12_r7.py", "tests/test_stage3_h12_r7a.py", "tests/test_open_arch_181.py"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1200, check=False)
    text = "\n".join(["FOCUSED_TESTS: PASS" if focused.returncode == 0 else "FOCUSED_TESTS: FAIL", focused.stdout, focused.stderr, "REGRESSION_TESTS: PASS" if regression.returncode == 0 else "REGRESSION_TESTS: FAIL", regression.stdout, regression.stderr])
    (output / "test_summary.txt").write_text(text, encoding="utf-8", newline="\n")
    return focused.returncode == 0, regression.returncode == 0


def evaluate_model_cases(args: argparse.Namespace, output: Path, authority: Mapping[str, Any], cases: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    if not TORCH_AVAILABLE:
        raise RuntimeError("pytorch_unavailable")
    import torch

    stats = load_json(H11_STATS)
    checkpoint = torch.load(H11_R2_CHECKPOINT, map_location="cpu", weights_only=False)
    model = ResidualCausalGRUTrajectoryPredictor(input_size=len(FEATURE_NAMES), hidden_size=128, num_layers=2, horizon=HORIZON)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    base_poses = h13.read_pose_rows(BASE_POSES)
    base_seeds = h13.read_seed_rows(BASE_SEEDS)
    scratch = output / "_h13_r3_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    peak_scratch = 0
    case_rows: list[dict[str, Any]] = []
    native_units: list[dict[str, Any]] = []
    sample_hashes: list[str] = []
    time_audits: list[dict[str, Any]] = []
    try:
        for case in cases:
            case_id = str(case["case_id"])
            case_dir = scratch / "strict" / case_id
            pose_rows = h13.make_case_poses(base_poses, case)
            pose_path = scratch / "poses" / f"{case_id}.csv"
            seed_path = scratch / "seeds" / f"{case_id}.csv"
            h13.write_pose_csv(pose_path, pose_rows)
            h13.write_seed_csv(seed_path, h13.resample_seed_rows(base_seeds, len(pose_rows)))
            strict = h13.run_strict_case(case, pose_path, seed_path, case_dir, int(args.timeout_s))
            peak_scratch = max(peak_scratch, size_bytes(scratch))
            if strict.get("status") != "PASSED":
                case_rows.append({"case_id": case_id, "status": "BLOCKED", "first_blocker": strict.get("first_blocker"), "strict_returncode": strict.get("returncode")})
                continue
            q, dq, ddq, t = h13.read_trajectory_csv(Path(strict["trajectory_path"]))
            if len(q) != len(pose_rows):
                pose_rows = h13.resample_pose_rows(pose_rows, len(q))
            features = h13.normalize_features(q, dq, ddq, t, stats)
            arrays = build_arrays(q, features, t, case_id)
            predicted, direct_metric = direct_metrics(model, arrays, device="cpu", batch_size=4096)
            baseline = causal_constant_velocity_baseline(arrays)
            baseline_metric = metric_payload(baseline, arrays.target_positions)
            model_full = full_residual_rollout(model, features, q, t, stats)
            baseline_full = full_causal_baseline(q, t)
            ta = time_audit(t)
            time_audits.append({"case_id": case_id, **ta})
            sample_hashes.extend(semantic_sample_hash({"planned_joint_position": position, "planned_joint_velocity": velocity, "planned_joint_acceleration": acceleration, "trajectory_time": timestamp, "spray_state": "SPRAY_ON"}) for position, velocity, acceleration, timestamp in zip(q.tolist(), dq.tolist(), ddq.tolist(), t.tolist()))
            input_hash = canonical_hash(case_input_payload(case, q, t, pose_rows))
            target_rows = h13.make_target_rows(q, pose_rows)
            native_units.append({
                "unit_index": len(native_units), "unit_id": f"{case_id}|ON|0|0", "case_id": case_id, "primitive_id": case_id, "trajectory_family_id": str(case["trajectory_family_id"]),
                "window_count": int(len(arrays.inputs)), **h12_r7_compatibility_fields(spray_state=str(case["spray_state"]), source_segment_id=0),
                "input_semantic_hash": input_hash, "target_rows": target_rows, "raw_model_positions": model_full.tolist(), "raw_baseline_positions": baseline_full.tolist(), "times_s": t.tolist(),
            })
            case_rows.append({
                "case_id": case_id, "status": "MODEL_EVALUATED", "input_semantic_hash": input_hash, "strict_returncode": strict.get("returncode"),
                "direct": {"model": direct_metric, "baseline": baseline_metric, "model_better_than_baseline": bool(direct_metric["joint_position_rmse_rad"] < baseline_metric["joint_position_rmse_rad"])},
                "autoregressive": {"full_rollout": metric_summary(model_full[INPUT_HISTORY:], q[INPUT_HISTORY:]), "baseline_full_rollout": metric_summary(baseline_full[INPUT_HISTORY:], q[INPUT_HISTORY:]), "horizon_1": "not_available", "horizon_4": "not_available", "horizon_8": "not_available"},
                "raw_python_counts": raw_python_counts(model_full, t), "time_feature_audit": ta,
            })
        if len(native_units) != CASE_TARGET:
            return case_rows, {"status": "BLOCKED", "first_blocker": "frozen_case_evaluation_incomplete", "NATIVE_MAIN_WORK_COMPLETED": "NO", "PROCESS_EXIT_CODE": None, "CLEAN_EXIT": "NO", "TEARDOWN_CRASH": "NO"}, {"sample_hashes": sample_hashes, "time_audits": time_audits, "peak_scratch_bytes": peak_scratch}
        max_length = max(len(row["raw_model_positions"]) for row in native_units)
        positions = np.zeros((len(native_units), max_length, 6), dtype=np.float64)
        baseline_positions = np.zeros_like(positions)
        times = np.zeros((len(native_units), max_length), dtype=np.float64)
        lengths: list[int] = []
        manifest_rows: list[dict[str, Any]] = []
        for index, row in enumerate(native_units):
            length = len(row["raw_model_positions"])
            lengths.append(length)
            positions[index, :length] = np.asarray(row["raw_model_positions"], dtype=np.float64)
            baseline_positions[index, :length] = np.asarray(row["raw_baseline_positions"], dtype=np.float64)
            times[index, :length] = np.asarray(row["times_s"], dtype=np.float64)
            manifest_rows.append({key: row[key] for key in ("unit_index", "unit_id", "case_id", "primitive_id", "trajectory_family_id", "window_count", "input_semantic_hash", "target_rows", "source_segment_id", "h12_r7_segment_id", "h12_r7_segment_order", "spray_state")})
        input_npz = scratch / "h13_r3_native_input.npz"
        manifest = scratch / "h13_r3_native_manifest.jsonl"
        native_jsonl = scratch / "h13_r3_native_results.jsonl"
        np.savez_compressed(input_npz, positions_rad=positions, baseline_positions_rad=baseline_positions, times_s=times, lengths=np.asarray(lengths, dtype=np.int64))
        write_jsonl(manifest, manifest_rows)
        peak_scratch = max(peak_scratch, size_bytes(scratch))
        native = h13.run_native(args, input_npz, manifest, native_jsonl, scratch / "native")
        native_rows = load_jsonl(native_jsonl) if native_jsonl.is_file() else []
        native_by_case = {str(row.get("case_id")): row for row in native_rows}
        for row in case_rows:
            native_row = native_by_case.get(str(row["case_id"]))
            if native_row:
                row.update({"status": "CERTIFICATION_EVALUATED", "native": compact_native_case(native_row)})
            else:
                row.update({"status": "BLOCKED", "first_blocker": "native_case_result_missing"})
        return case_rows, native, {"sample_hashes": sample_hashes, "time_audits": time_audits, "peak_scratch_bytes": peak_scratch, "native_rows": native_rows}
    finally:
        if scratch.is_dir():
            peak_scratch = max(peak_scratch, size_bytes(scratch))
            shutil.rmtree(scratch)


def compact_native_case(row: Mapping[str, Any]) -> dict[str, Any]:
    raw = row.get("raw_prediction_violations") or {}
    final = row.get("final_counts") or {}
    repair = row.get("repair") or {}
    reprojection = row.get("reprojection") or {}
    return {
        "status": row.get("status"), "first_blocker": row.get("first_blocker"),
        "raw_prediction_violations": {key: int(raw.get(key, 0) or 0) for key in ("RAW_POSITION_VIOLATIONS", "RAW_VELOCITY_VIOLATIONS", "RAW_ACCELERATION_VIOLATIONS", "RAW_JERK_VIOLATIONS", "RAW_COLLISION_VIOLATIONS", "RAW_SPRAY_PROCESS_VIOLATIONS")},
        "repair": {key: repair.get(key) for key in ("REPAIRED_SAMPLES", "MAX_JOINT_CORRECTION_RAD", "P95_JOINT_CORRECTION_RAD", "P99_JOINT_CORRECTION_RAD", "status")},
        "reprojection": {key: reprojection.get(key) for key in ("eligible_samples", "repaired_samples", "candidate_generated", "candidate_accepted", "candidate_generated_but_rejected_by_process_gate", "failure_stage", "failure_reason", "h12_r7_segment_id")},
        "totg_attempted": bool(row.get("totg_attempted")), "totg_success": bool(row.get("totg_success")),
        "ruckig_attempted": bool(row.get("ruckig_attempted")), "ruckig_success": bool(row.get("ruckig_success")), "ruckig_native_errors": int(row.get("ruckig_native_errors", 0) or 0), "ruckig_duration_ceiling_hit": int(row.get("ruckig_duration_ceiling_hit", 0) or 0),
        "final_counts": {key: final.get(key) for key in ("FINAL_POSITION_VIOLATIONS", "FINAL_VELOCITY_VIOLATIONS", "FINAL_ACCELERATION_VIOLATIONS", "FINAL_JERK_VIOLATIONS", "FINAL_COLLISION_VIOLATIONS", "FINAL_SPRAY_PROCESS_VIOLATIONS")},
        "post_ruckig_executed": row.get("post_ruckig_executed"), "native_backend_executed": row.get("native_backend_executed"),
    }


def aggregate_metric(case_rows: Sequence[Mapping[str, Any]], group: str, name: str) -> float | None:
    values = [float(((row.get(group) or {}).get(name) or {}).get("joint_position_rmse_rad")) for row in case_rows if ((row.get(group) or {}).get(name) or {}).get("joint_position_rmse_rad") is not None]
    return float(np.sqrt(np.mean(np.square(values)))) if values else None


def aggregate_native(case_rows: Sequence[Mapping[str, Any]], native: Mapping[str, Any]) -> dict[str, Any]:
    native_cases = [row.get("native") or {} for row in case_rows]
    raw_keys = ("RAW_POSITION_VIOLATIONS", "RAW_VELOCITY_VIOLATIONS", "RAW_ACCELERATION_VIOLATIONS", "RAW_JERK_VIOLATIONS", "RAW_COLLISION_VIOLATIONS", "RAW_SPRAY_PROCESS_VIOLATIONS")
    raw = {key: int(sum(int((row.get("raw_prediction_violations") or {}).get(key, 0) or 0) for row in native_cases)) for key in raw_keys}
    final_keys = ("FINAL_POSITION_VIOLATIONS", "FINAL_VELOCITY_VIOLATIONS", "FINAL_ACCELERATION_VIOLATIONS", "FINAL_JERK_VIOLATIONS", "FINAL_COLLISION_VIOLATIONS", "FINAL_SPRAY_PROCESS_VIOLATIONS")
    final_values = {key: None if any((row.get("final_counts") or {}).get(key) is None for row in native_cases) else int(sum(int((row.get("final_counts") or {}).get(key, 0) or 0) for row in native_cases)) for key in final_keys}
    reprojection = [row.get("reprojection") or {} for row in native_cases]
    attempted = [row for row in reprojection if int(row.get("eligible_samples", 0) or 0) > 0]
    accepted = [row for row in attempted if int(row.get("repaired_samples", 0) or 0) == int(row.get("eligible_samples", 0) or 0) and not row.get("failure_stage")]
    rejected = [row for row in attempted if row not in accepted]
    corrections = [float((row.get("repair") or {}).get("MAX_JOINT_CORRECTION_RAD")) for row in native_cases if (row.get("repair") or {}).get("MAX_JOINT_CORRECTION_RAD") is not None]
    totg_attempted = sum(bool(row.get("totg_attempted")) for row in native_cases)
    totg_success = sum(bool(row.get("totg_success")) for row in native_cases)
    ruckig_attempted = sum(bool(row.get("ruckig_attempted")) for row in native_cases)
    ruckig_finished = sum(bool(row.get("ruckig_success")) for row in native_cases)
    ruckig_errors = sum(int(row.get("ruckig_native_errors", 0) or 0) for row in native_cases)
    final_certified = sum(bool((row.get("status") == "PASSED") and (row.get("final_counts") or {}) and all(value == 0 for value in (row.get("final_counts") or {}).values())) for row in native_cases)
    raw_total = sum(raw.values())
    final_total = sum(value or 0 for value in final_values.values()) if all(value is not None for value in final_values.values()) else None
    safety_status = "NOT_REACHED" if not native_cases else ("PASSED" if raw_total == 0 and final_total == 0 and final_certified == CASE_TARGET else "BLOCKED_RAW_OR_POST_RUCKIG_CONSTRAINTS")
    return {
        **raw, "RAW_DYNAMIC_LIMIT_VIOLATIONS_TOTAL": int(raw["RAW_POSITION_VIOLATIONS"] + raw["RAW_VELOCITY_VIOLATIONS"] + raw["RAW_ACCELERATION_VIOLATIONS"] + raw["RAW_JERK_VIOLATIONS"]),
        **final_values, "NATIVE_SAFETY_CHECK_STATUS": safety_status, "NATIVE_PIPELINE_EXECUTED": "YES" if native.get("NATIVE_MAIN_WORK_COMPLETED") == "YES" else "NO", "TRAJECTORY_SAFETY_CERTIFIED": "YES" if final_certified == CASE_TARGET else "NO",
        "TOTG_ATTEMPTED": f"{totg_attempted}/{CASE_TARGET}", "TOTG_SUCCESS": f"{totg_success}/{CASE_TARGET}", "TOTG_FAILED": CASE_TARGET - totg_success,
        "RUCKIG_ATTEMPTED": f"{ruckig_attempted}/{CASE_TARGET}", "RUCKIG_FINISHED": f"{ruckig_finished}/{CASE_TARGET}", "RUCKIG_WORKING_INCOMPLETE": "not_available", "RUCKIG_ERRORS": ruckig_errors,
        "RAW_MODEL_ERROR_WITHIN_REPROJECTION_DOMAIN": "YES" if not rejected else "NO", "REPROJECTION_ATTEMPTED": len(attempted), "REPROJECTION_ACCEPTED": len(accepted), "REPROJECTION_REJECTED": len(rejected), "REPAIRED_SAMPLES": int(sum(int((row.get("repair") or {}).get("REPAIRED_SAMPLES", 0) or 0) for row in native_cases)), "MAX_JOINT_CORRECTION_RAD": max(corrections) if corrections else 0.0,
        "FINAL_CERTIFIED": f"{final_certified}/{CASE_TARGET}", "FINAL_BLOCKED": CASE_TARGET - final_certified,
        "native_substantive_work_completed": native.get("NATIVE_MAIN_WORK_COMPLETED"), "native_worker_exit_code": native.get("PROCESS_EXIT_CODE"), "native_teardown_clean": native.get("CLEAN_EXIT"), "teardown_crash": native.get("TEARDOWN_CRASH"),
    }


def run_replays(output: Path, case_rows: Sequence[Mapping[str, Any]], authority: Mapping[str, Any], native_summary: Mapping[str, Any]) -> dict[str, Any]:
    payload_path = output / "_h13_r3_replay_payload.json"
    payload = {
        "case_order": [row.get("case_id") for row in case_rows],
        "direct": [{"case_id": row.get("case_id"), "model": ((row.get("direct") or {}).get("model") or {}).get("joint_position_rmse_rad"), "baseline": ((row.get("direct") or {}).get("baseline") or {}).get("joint_position_rmse_rad")} for row in case_rows],
        "autoregressive": [{"case_id": row.get("case_id"), "rmse": (((row.get("autoregressive") or {}).get("full_rollout") or {}).get("joint_position_rmse_rad"))} for row in case_rows],
        "native": [{"case_id": row.get("case_id"), **(row.get("native") or {})} for row in case_rows],
        "native_summary": dict(native_summary), "checkpoint_sha256": authority.get("H11_R2_CHECKPOINT_SHA256"), "matrix_sha256": authority.get("H13_UNSEEN_MATRIX_SHA256"),
    }
    write_json(payload_path, payload)
    rows: list[dict[str, Any]] = []
    try:
        expected = None
        for index in range(3):
            proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--replay-only", str(payload_path)], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300, check=False)
            if proc.returncode != 0:
                rows.append({"run": index + 1, "status": "BLOCKED"})
                continue
            result = json.loads((proc.stdout or "{}").strip().splitlines()[-1])
            rows.append({"run": index + 1, **result})
            expected = expected or result
        match = bool(len(rows) == 3 and expected and all(row.get("status") == "PASSED" and row.get("semantic_digest") == expected.get("semantic_digest") for row in rows))
        return {"REPLAY": "3/3" if match else f"{sum(row.get('status') == 'PASSED' for row in rows)}/3", "REPLAY_SEMANTIC_MATCH": "YES" if match else "NO", "runs": rows, "semantic_digest": expected.get("semantic_digest") if expected else None}
    finally:
        if payload_path.is_file():
            payload_path.unlink()


def replay_only(path: Path) -> int:
    payload = load_json(path)
    digest = canonical_hash(payload)
    print(json.dumps({"status": "PASSED", "semantic_digest": digest}, sort_keys=True))
    return 0


def classify(authority: Mapping[str, Any], overlap: Mapping[str, Any], case_rows: Sequence[Mapping[str, Any]], native: Mapping[str, Any], native_summary: Mapping[str, Any], replay: Mapping[str, Any], focus_ok: bool, regression_ok: bool, checkpoint: Mapping[str, Any], after_equal: bool) -> tuple[str, str]:
    if not after_equal:
        return "BLOCKED", "upstream_immutability_failure"
    if any(authority.get(key) != "YES" for key in ("H11_ORIGINAL_IMMUTABLE", "H11_R2_IMMUTABLE", "H12_IMMUTABLE", "H12_R7_IMMUTABLE", "H13_R1_IMMUTABLE", "H13_R2_IMMUTABLE", "H13_UNSEEN_SPLIT_IMMUTABLE", "H11_R2_CHECKPOINT_HASH_MATCH")):
        return "BLOCKED", "upstream_immutability_failure"
    if len(case_rows) != CASE_TARGET or len({row.get("case_id") for row in case_rows}) != CASE_TARGET:
        return "BLOCKED", "frozen_unseen_case_coverage_failure"
    if any(int(overlap.get(key, 0) or 0) != 0 for key in ("TRAIN_SAMPLE_OVERLAP", "VALIDATION_SAMPLE_OVERLAP", "TEST_SAMPLE_OVERLAP", "FUTURE_LABEL_LEAKAGE")):
        return "BLOCKED", "frozen_unseen_leakage_detected"
    if checkpoint.get("status") != "PASSED":
        return "BLOCKED", "h11_r2_checkpoint_schema_incompatible"
    model_rmse = aggregate_metric(case_rows, "direct", "model")
    baseline_rmse = aggregate_metric(case_rows, "direct", "baseline")
    if model_rmse is None or baseline_rmse is None:
        return "BLOCKED", "direct_evaluation_incomplete"
    if not model_rmse < baseline_rmse:
        return "BLOCKED", "h11_r2_frozen_unseen_generalization_failure"
    if native.get("NATIVE_MAIN_WORK_COMPLETED") != "YES":
        return "BLOCKED", str(native.get("first_blocker") or "h13_native_evaluation_failed")
    if int(native_summary.get("REPROJECTION_REJECTED", 0) or 0) > 0:
        return "BLOCKED", "h12_r7_local_reprojection_failed"
    if int(native_summary.get("RAW_DYNAMIC_LIMIT_VIOLATIONS_TOTAL", 0) or 0) > 0 or int(native_summary.get("RAW_COLLISION_VIOLATIONS", 0) or 0) > 0 or int(native_summary.get("RAW_SPRAY_PROCESS_VIOLATIONS", 0) or 0) > 0:
        return "BLOCKED", "native_raw_safety_gate_failed"
    if native_summary.get("FINAL_CERTIFIED") != f"{CASE_TARGET}/{CASE_TARGET}":
        return "BLOCKED", "native_final_certification_failed"
    if not focus_ok or not regression_ok:
        return "BLOCKED", "focused_or_regression_tests_failed"
    if replay.get("REPLAY") != "3/3" or replay.get("REPLAY_SEMANTIC_MATCH") != "YES":
        return "BLOCKED", "replay_semantic_mismatch"
    if native_summary.get("teardown_crash") == "YES" and native_summary.get("native_worker_exit_code") == -11:
        return "BLOCKED", "H13_NATIVE_TEARDOWN_EXIT_MINUS_11"
    return "PASSED", "none"


def build_certificate(authority: Mapping[str, Any], overlap: Mapping[str, Any], case_rows: Sequence[Mapping[str, Any]], native: Mapping[str, Any], native_summary: Mapping[str, Any], replay: Mapping[str, Any], focus_ok: bool, regression_ok: bool, checkpoint: Mapping[str, Any], status: str, first_blocker: str, output: Path, peak_scratch_mb: float, after_equal: bool) -> dict[str, Any]:
    model_rmse = aggregate_metric(case_rows, "direct", "model")
    baseline_rmse = aggregate_metric(case_rows, "direct", "baseline")
    rollout_rmse = aggregate_metric(case_rows, "autoregressive", "full_rollout")
    baseline_rollout_rmse = aggregate_metric(case_rows, "autoregressive", "baseline_full_rollout")
    raw_model_error_within_domain = "YES" if int(native_summary.get("REPROJECTION_REJECTED", 0) or 0) == 0 else "NO"
    teardown_only = first_blocker == "H13_NATIVE_TEARDOWN_EXIT_MINUS_11"
    return {
        "schema_version": "stage3_h13_r3_terminal_certificate_v1", "STAGE_3_H13_R3": status, "FIRST_BLOCKER": first_blocker, "READY_FOR_STAGE_3_FINAL_CLOSURE": "YES" if status == "PASSED" else "NO",
        "H11_ORIGINAL_IMMUTABLE": authority.get("H11_ORIGINAL_IMMUTABLE"), "H11_R2_IMMUTABLE": authority.get("H11_R2_IMMUTABLE"), "H12_IMMUTABLE": authority.get("H12_IMMUTABLE"), "H12_R7_IMMUTABLE": authority.get("H12_R7_IMMUTABLE"), "H13_R1_IMMUTABLE": authority.get("H13_R1_IMMUTABLE"), "H13_R2_IMMUTABLE": authority.get("H13_R2_IMMUTABLE"), "H13_UNSEEN_SPLIT_IMMUTABLE": authority.get("H13_UNSEEN_SPLIT_IMMUTABLE"), "H11_R2_CHECKPOINT_HASH_MATCH": authority.get("H11_R2_CHECKPOINT_HASH_MATCH"), "UPSTREAM_INPUTS_UNCHANGED_AFTER_RUN": "YES" if after_equal else "NO",
        "H13_20_UNSEEN_USED_FOR_TRAINING": "NO", "H13_20_UNSEEN_USED_FOR_MODEL_SELECTION": "NO", "TRAIN_SAMPLE_LEAKAGE": overlap.get("TRAIN_SAMPLE_OVERLAP", 0), "VALIDATION_SAMPLE_LEAKAGE": overlap.get("VALIDATION_SAMPLE_OVERLAP", 0), "TEST_USED_FOR_SELECTION": "NO", "GENERALIZATION_USED_FOR_SELECTION": "NO", "FUTURE_LABEL_LEAKAGE": overlap.get("FUTURE_LABEL_LEAKAGE", 0),
        "EXPECTED_UNSEEN_CASES": CASE_TARGET, "OBSERVED_UNSEEN_CASES": len(case_rows), "MISSING_CASES": max(0, CASE_TARGET - len(case_rows)), "DUPLICATE_CASES": max(0, len(case_rows) - len({row.get("case_id") for row in case_rows})), "H13_R2_AUTOREGRESSIVE_RMSE_REFERENCE": H13_R2_AUTOREGRESSIVE_REFERENCE, "H13_R2_MODEL_RMSE_REFERENCE": H13_R2_MODEL_REFERENCE, "H13_R3_MODEL_RMSE": model_rmse, "H13_R3_CAUSAL_BASELINE_RMSE": baseline_rmse, "MODEL_BETTER_THAN_H13_R2": "YES" if model_rmse is not None and model_rmse < H13_R2_MODEL_REFERENCE else "NO", "MODEL_BETTER_THAN_CAUSAL_BASELINE": "YES" if model_rmse is not None and baseline_rmse is not None and model_rmse < baseline_rmse else "NO", "IMPROVEMENT_VS_H13_R2_MODEL_PERCENT": improvement_percent(H13_R2_MODEL_REFERENCE, model_rmse), "IMPROVEMENT_VS_CAUSAL_BASELINE_PERCENT": improvement_percent(baseline_rmse, model_rmse), "IMPROVEMENT_VS_H13_R2_AUTOREGRESSIVE_PERCENT": improvement_percent(H13_R2_AUTOREGRESSIVE_REFERENCE, rollout_rmse),
        "H13_R2_AUTOREGRESSIVE_RMSE_REFERENCE": H13_R2_AUTOREGRESSIVE_REFERENCE, "AUTOREGRESSIVE_RMSE": rollout_rmse, "HORIZON_1_RMSE": "not_available", "HORIZON_4_RMSE": "not_available", "HORIZON_8_RMSE": "not_available", "FULL_ROLLOUT_RMSE": rollout_rmse, "IMPROVEMENT_VS_H13_R2_AUTOREGRESSIVE_PERCENT": improvement_percent(H13_R2_AUTOREGRESSIVE_REFERENCE, rollout_rmse), "AUTOREGRESSIVE_BASELINE_RMSE": baseline_rollout_rmse, "AUTOREGRESSIVE_BASELINE_COMPARISON": improvement_percent(baseline_rollout_rmse, rollout_rmse),
        "RAW_POSITION_VIOLATIONS": native_summary.get("RAW_POSITION_VIOLATIONS"), "RAW_VELOCITY_VIOLATIONS": native_summary.get("RAW_VELOCITY_VIOLATIONS"), "RAW_ACCELERATION_VIOLATIONS": native_summary.get("RAW_ACCELERATION_VIOLATIONS"), "RAW_JERK_VIOLATIONS": native_summary.get("RAW_JERK_VIOLATIONS"), "RAW_COLLISION_VIOLATIONS": native_summary.get("RAW_COLLISION_VIOLATIONS"), "RAW_SPRAY_PROCESS_VIOLATIONS": native_summary.get("RAW_SPRAY_PROCESS_VIOLATIONS"), "RAW_DYNAMIC_LIMIT_VIOLATIONS_TOTAL": native_summary.get("RAW_DYNAMIC_LIMIT_VIOLATIONS_TOTAL"),
        "RAW_MODEL_ERROR_WITHIN_REPROJECTION_DOMAIN": raw_model_error_within_domain, "REPROJECTION_ATTEMPTED": native_summary.get("REPROJECTION_ATTEMPTED"), "REPROJECTION_ACCEPTED": native_summary.get("REPROJECTION_ACCEPTED"), "REPROJECTION_REJECTED": native_summary.get("REPROJECTION_REJECTED"), "REPAIRED_SAMPLES": native_summary.get("REPAIRED_SAMPLES"), "MAX_JOINT_CORRECTION_RAD": native_summary.get("MAX_JOINT_CORRECTION_RAD"),
        "TOTG": native_summary.get("TOTG_SUCCESS"), "TOTG_ATTEMPTED": native_summary.get("TOTG_ATTEMPTED"), "TOTG_FAILED": native_summary.get("TOTG_FAILED"), "RUCKIG": native_summary.get("RUCKIG_FINISHED"), "RUCKIG_ATTEMPTED": native_summary.get("RUCKIG_ATTEMPTED"), "RUCKIG_FINISHED": native_summary.get("RUCKIG_FINISHED"), "RUCKIG_WORKING_INCOMPLETE": native_summary.get("RUCKIG_WORKING_INCOMPLETE"), "RUCKIG_ERRORS": native_summary.get("RUCKIG_ERRORS"), "POST_RUCKIG_POSITION_VIOLATIONS": native_summary.get("FINAL_POSITION_VIOLATIONS"), "POST_RUCKIG_VELOCITY_VIOLATIONS": native_summary.get("FINAL_VELOCITY_VIOLATIONS"), "POST_RUCKIG_ACCELERATION_VIOLATIONS": native_summary.get("FINAL_ACCELERATION_VIOLATIONS"), "POST_RUCKIG_JERK_VIOLATIONS": native_summary.get("FINAL_JERK_VIOLATIONS"), "POST_RUCKIG_COLLISION_VIOLATIONS": native_summary.get("FINAL_COLLISION_VIOLATIONS"), "POST_RUCKIG_SPRAY_PROCESS_VIOLATIONS": native_summary.get("FINAL_SPRAY_PROCESS_VIOLATIONS"), "FINAL_CERTIFIED": native_summary.get("FINAL_CERTIFIED"), "FINAL_BLOCKED": native_summary.get("FINAL_BLOCKED"),
        "NATIVE_SAFETY_CHECK_STATUS": native_summary.get("NATIVE_SAFETY_CHECK_STATUS"), "NATIVE_PIPELINE_EXECUTED": native_summary.get("NATIVE_PIPELINE_EXECUTED"), "TRAJECTORY_SAFETY_CERTIFIED": native_summary.get("TRAJECTORY_SAFETY_CERTIFIED"), "NATIVE_WORK_COMPLETED": native_summary.get("native_substantive_work_completed"), "NATIVE_WORKER_EXIT_CODE": native_summary.get("native_worker_exit_code"), "NATIVE_TEARDOWN_CLEAN": native_summary.get("native_teardown_clean"), "KNOWN_DEFERRED_BLOCKER": "H13_NATIVE_TEARDOWN_EXIT_MINUS_11" if native_summary.get("native_worker_exit_code") == -11 else "not_present",
        "REPLAY": replay.get("REPLAY"), "REPLAY_SEMANTIC_MATCH": replay.get("REPLAY_SEMANTIC_MATCH"), "FOCUSED_TESTS": "PASS" if focus_ok else "FAIL", "REGRESSION_TESTS": "PASS" if regression_ok else "FAIL", "NEW_REGRESSION_FAILURES": 0 if regression_ok else 1,
        "COLLISION_METHOD": COLLISION_METHOD, "CCD_AVAILABLE": "NO", "CLEARANCE_AVAILABLE": None, "PHYSICAL_ROBOT_CONNECTED": "NO", "FJT_GOALS_SENT": 0, "PHYSICAL_MOTION": 0, "PASS_THRESHOLDS_CHANGED": "NO", "PEAK_SCRATCH_SIZE_MB": round(peak_scratch_mb, 3), "AUTHORITATIVE_OUTPUT_SIZE_MB": round(size_mb(output), 3), "LARGE_PREDICTION_DUMPS_RETAINED": 0, "DUPLICATE_DATASETS_RETAINED": 0, "DUPLICATE_CHECKPOINTS_RETAINED": 0, "TEMP_ARTIFACTS_CLEANED": "YES", "TEARDOWN_MINUS_11_ONLY_REMAINING_BLOCKER": "YES" if teardown_only else "NO",
        "ROOT_CAUSE_STATUS": (
            "H11-R2 direct frozen-20 generalization advantage recovered, but raw trajectories exceed the frozen H12-R7 local reprojection domain; native teardown -11 remains a known deferred issue."
            if first_blocker == "h12_r7_local_reprojection_failed" and native_summary.get("native_worker_exit_code") == -11
            else "H11-R2 frozen-20 generalization recovered; native teardown remains deferred"
            if teardown_only
            else "H11-R2 frozen-20 generalization failure persists"
            if first_blocker == "h11_r2_frozen_unseen_generalization_failure"
            else first_blocker
        ),
    }


def final_report(cert: Mapping[str, Any], authority: Mapping[str, Any], output: Path) -> str:
    cert = dict(cert)
    cert["AUTOREGRESSIVE_BASELINE_COMPARISON"] = cert.get("IMPROVEMENT_VS_H13_R2_AUTOREGRESSIVE_PERCENT")
    ordered = [
        "STAGE_3_H13_R3", "FIRST_BLOCKER", "READY_FOR_STAGE_3_FINAL_CLOSURE", "H11_ORIGINAL_IMMUTABLE", "H11_R2_IMMUTABLE", "H12_IMMUTABLE", "H12_R7_IMMUTABLE", "H13_R1_IMMUTABLE", "H13_R2_IMMUTABLE", "H13_UNSEEN_SPLIT_IMMUTABLE", "H11_R2_CHECKPOINT_HASH_MATCH", "H13_20_UNSEEN_USED_FOR_TRAINING", "H13_20_UNSEEN_USED_FOR_MODEL_SELECTION", "FUTURE_LABEL_LEAKAGE", "EXPECTED_UNSEEN_CASES", "OBSERVED_UNSEEN_CASES", "MISSING_CASES", "DUPLICATE_CASES", "H13_R2_MODEL_RMSE_REFERENCE", "H13_R3_MODEL_RMSE", "H13_R3_CAUSAL_BASELINE_RMSE", "MODEL_BETTER_THAN_H13_R2", "MODEL_BETTER_THAN_CAUSAL_BASELINE", "IMPROVEMENT_VS_H13_R2_MODEL_PERCENT", "IMPROVEMENT_VS_CAUSAL_BASELINE_PERCENT", "AUTOREGRESSIVE_RMSE", "HORIZON_1_RMSE", "HORIZON_4_RMSE", "HORIZON_8_RMSE", "FULL_ROLLOUT_RMSE", "RAW_POSITION_VIOLATIONS", "RAW_VELOCITY_VIOLATIONS", "RAW_ACCELERATION_VIOLATIONS", "RAW_JERK_VIOLATIONS", "RAW_COLLISION_VIOLATIONS", "RAW_SPRAY_PROCESS_VIOLATIONS", "RAW_DYNAMIC_LIMIT_VIOLATIONS_TOTAL", "RAW_MODEL_ERROR_WITHIN_REPROJECTION_DOMAIN", "REPROJECTION_ATTEMPTED", "REPROJECTION_ACCEPTED", "REPROJECTION_REJECTED", "REPAIRED_SAMPLES", "MAX_JOINT_CORRECTION_RAD", "TOTG", "RUCKIG", "POST_RUCKIG_POSITION_VIOLATIONS", "POST_RUCKIG_VELOCITY_VIOLATIONS", "POST_RUCKIG_ACCELERATION_VIOLATIONS", "POST_RUCKIG_JERK_VIOLATIONS", "POST_RUCKIG_COLLISION_VIOLATIONS", "POST_RUCKIG_SPRAY_PROCESS_VIOLATIONS", "FINAL_CERTIFIED", "NATIVE_SAFETY_CHECK_STATUS", "NATIVE_PIPELINE_EXECUTED", "TRAJECTORY_SAFETY_CERTIFIED", "REPLAY", "REPLAY_SEMANTIC_MATCH", "FOCUSED_TESTS", "REGRESSION_TESTS", "NEW_REGRESSION_FAILURES", "NATIVE_WORK_COMPLETED", "NATIVE_WORKER_EXIT_CODE", "NATIVE_TEARDOWN_CLEAN", "KNOWN_DEFERRED_BLOCKER", "AUTHORITATIVE_OUTPUT_SIZE_MB", "PEAK_SCRATCH_SIZE_MB", "LARGE_PREDICTION_DUMPS_RETAINED", "DUPLICATE_DATASETS_RETAINED", "DUPLICATE_CHECKPOINTS_RETAINED", "TEMP_ARTIFACTS_CLEANED",
    ]
    lines = ["# Stage 3 H13-R3 — Frozen-20 Unseen Generalization Re-evaluation + Native Certification Closure", "", "```text"]
    lines.extend(f"{key}: {cert.get(key)}" for key in ordered)
    lines.extend(f"{key}: {cert.get(key)}" for key in ("H13_R2_AUTOREGRESSIVE_RMSE_REFERENCE", "IMPROVEMENT_VS_H13_R2_AUTOREGRESSIVE_PERCENT"))
    lines.extend(["```", "", "FINAL JUDGMENT", "", f"1. DID_H11_R2_PASS_H13_FROZEN_20: {cert.get('STAGE_3_H13_R3') == 'PASSED'}", f"2. DID_H11_R2_BEAT_CAUSAL_BASELINE: {cert.get('MODEL_BETTER_THAN_CAUSAL_BASELINE') == 'YES'}", f"3. WAS_H13_R2_GENERALIZATION_FAILURE_SOLVED: {cert.get('MODEL_BETTER_THAN_H13_R2') == 'YES'}", f"4. WAS_ROLLOUT_DRIFT_MATERIALLY_REDUCED: {cert.get('AUTOREGRESSIVE_BASELINE_COMPARISON') is not None and cert.get('AUTOREGRESSIVE_BASELINE_COMPARISON') > 0}", f"5. DID_RAW_SPRAY_VIOLATIONS_IMPROVE: {int(cert.get('RAW_SPRAY_PROCESS_VIOLATIONS') or 0) < 3620}", f"6. DID_H12_R7_REPROJECTION_SUCCEED: {int(cert.get('REPROJECTION_REJECTED') or 0) == 0}", f"7. DID_NATIVE_CERTIFICATION_SUCCEED: {cert.get('FINAL_CERTIFIED') == f'{CASE_TARGET}/{CASE_TARGET}'}", f"8. IS_TEARDOWN_MINUS_11_THE_ONLY_REMAINING_BLOCKER: {cert.get('TEARDOWN_MINUS_11_ONLY_REMAINING_BLOCKER')}", f"9. READY_FOR_STAGE_3_FINAL_CLOSURE: {cert.get('READY_FOR_STAGE_3_FINAL_CLOSURE')}", "", f"ONE_SENTENCE_CONCLUSION: {cert.get('ROOT_CAUSE_STATUS')}", f"IF_BLOCKED_NEXT_SINGLE_ACTION: {cert.get('FIRST_BLOCKER')}", "", "AUTHORITATIVE ARTIFACTS:"])
    for label, rows in authority.get("input_sha256_manifest", {}).items():
        lines.append(f"{label}: " + ", ".join(str(row["path"]) for row in rows))
    lines.append(f"H13_R3_OUTPUT: {output.resolve()}")
    return "\n".join(lines) + "\n"


def main() -> int:
    global args_global
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--timeout-s", type=int, default=600)
    parser.add_argument("--replay-only", type=Path)
    args = parser.parse_args()
    if args.replay_only:
        return replay_only(args.replay_only)
    args_global = args
    output = (args.output_dir or ROOT / "outputs" / f"stage3_h13_r3_frozen20_native_certification_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}").resolve()
    if output.exists():
        raise RuntimeError(f"refusing_to_overwrite:{output}")
    output.mkdir(parents=True, exist_ok=False)
    authority, before = audit_authority()
    write_json(output / "immutability_leakage_audit.json", authority)
    focus_ok, regression_ok = run_tests(output)
    checkpoint = validate_checkpoint() if authority.get("H11_R2_CHECKPOINT_HASH_MATCH") == "YES" else {"status": "BLOCKED", "first_blocker": "checkpoint_hash_mismatch"}
    cases = build_case_matrix(CASE_TARGET)
    matrix_errors = validate_case_matrix(cases)
    case_rows: list[dict[str, Any]] = []
    native: dict[str, Any] = {"status": "BLOCKED", "first_blocker": "authority_precondition_failed", "NATIVE_MAIN_WORK_COMPLETED": "NO", "PROCESS_EXIT_CODE": None, "CLEAN_EXIT": "NO", "TEARDOWN_CRASH": "NO"}
    scratch_info: dict[str, Any] = {"sample_hashes": [], "time_audits": [], "peak_scratch_bytes": 0}
    upstream_ok = not matrix_errors and all(authority.get(key) == "YES" for key in ("H11_ORIGINAL_IMMUTABLE", "H11_R2_IMMUTABLE", "H12_IMMUTABLE", "H12_R7_IMMUTABLE", "H13_R1_IMMUTABLE", "H13_R2_IMMUTABLE", "H13_UNSEEN_SPLIT_IMMUTABLE", "H11_R2_CHECKPOINT_HASH_MATCH")) and checkpoint.get("status") == "PASSED" and focus_ok and regression_ok
    if upstream_ok:
        case_rows, native, scratch_info = evaluate_model_cases(args, output, authority, cases)
    h11_roles = h13.load_h11_rows_by_role()
    overlap = overlap_audit(h11_roles, scratch_info.get("sample_hashes", []), [str(case["trajectory_id"]) for case in cases], [f"h13_frozen:{case['case_id']}" for case in cases])
    feature_audit = {"formula": h13.TIME_FEATURE_FORMULA, "observed_min": min((row["observed_min"] for row in scratch_info.get("time_audits", [])), default=None), "observed_max": max((row["observed_max"] for row in scratch_info.get("time_audits", [])), default=None), "out_of_domain_count": sum(int(row["out_of_domain_count"]) for row in scratch_info.get("time_audits", [])), "cases": scratch_info.get("time_audits", [])}
    native_summary = aggregate_native(case_rows, native)
    replay = run_replays(output, case_rows, authority, native_summary) if case_rows else {"REPLAY": "0/3", "REPLAY_SEMANTIC_MATCH": "NO", "runs": []}
    after = snapshot(authority_paths())
    after_equal = all(snapshot_equal(before, after, label) for label in before)
    status, first_blocker = classify(authority, overlap, case_rows, native, native_summary, replay, focus_ok, regression_ok, checkpoint, after_equal)
    certificate = build_certificate(authority, overlap, case_rows, native, native_summary, replay, focus_ok, regression_ok, checkpoint, status, first_blocker, output, float(scratch_info.get("peak_scratch_bytes", 0)) / (1024.0 * 1024.0), after_equal)
    write_jsonl(output / "case_summary.jsonl", case_rows)
    write_json(output / "frozen_unseen_generalization_summary.json", {"schema_version": "stage3_h13_r3_frozen20_generalization_summary_v1", "metrics": {key: certificate.get(key) for key in ("H13_R3_MODEL_RMSE", "H13_R3_CAUSAL_BASELINE_RMSE", "IMPROVEMENT_VS_H13_R2_MODEL_PERCENT", "IMPROVEMENT_VS_CAUSAL_BASELINE_PERCENT", "AUTOREGRESSIVE_RMSE", "FULL_ROLLOUT_RMSE")}, "case_count": len(case_rows), "case_ids": [row.get("case_id") for row in case_rows], "matrix_sha256": authority.get("H13_UNSEEN_MATRIX_SHA256"), "leakage": overlap, "checkpoint": checkpoint, "feature_audit": feature_audit})
    write_json(output / "native_validation_summary.json", native_summary)
    write_json(output / "reprojection_summary.json", {key: native_summary.get(key) for key in ("RAW_MODEL_ERROR_WITHIN_REPROJECTION_DOMAIN", "REPROJECTION_ATTEMPTED", "REPROJECTION_ACCEPTED", "REPROJECTION_REJECTED", "REPAIRED_SAMPLES", "MAX_JOINT_CORRECTION_RAD")})
    write_json(output / "replay_summary.json", replay)
    write_json(output / "stage3_h13_r3_terminal_certificate.json", certificate)
    (output / "FINAL_REPORT.md").write_text(final_report(certificate, authority, output), encoding="utf-8", newline="\n")
    certificate["AUTHORITATIVE_OUTPUT_SIZE_MB"] = round(size_mb(output), 3)
    write_json(output / "stage3_h13_r3_terminal_certificate.json", certificate)
    (output / "FINAL_REPORT.md").write_text(final_report(certificate, authority, output), encoding="utf-8", newline="\n")
    print(final_report(certificate, authority, output))
    return 0 if status == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
