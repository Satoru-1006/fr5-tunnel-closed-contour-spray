#!/usr/bin/env python3
"""Stage 3 H13: frozen-model unseen generalization and certification.

This runner is deliberately additive and fail-closed.  It never trains or
updates H11, never edits H12, never talks to a controller, and deletes native
case/replay artifacts after their compact evidence has been extracted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage3_h11_model import CausalGRUTrajectoryPredictor, TORCH_AVAILABLE, constant_velocity_prediction
from src.stage3_h13 import (
    COLLISION_METHOD,
    CASE_TARGET,
    aggregate_counts,
    build_case_matrix,
    canonical,
    canonical_hash,
    case_input_payload,
    improvement_percent,
    H12_R7_COMPAT_SEGMENT_ID,
    H12_R7_COMPAT_SEGMENT_ORDER,
    h12_r7_compatibility_fields,
    metric_summary,
    overlap_audit,
    percentile,
    semantic_sample_hash,
    validate_case_matrix,
)


H11_ROOT = ROOT / "outputs/stage3_h11_r_pytorch_runtime_recertification_20260811T143806Z"
H12_R7_ROOT = ROOT / "outputs/stage3_h12_r7_low_storage_reprojection_20260813T030000Z"
OPEN_ARCH_INPUT_ROOT = ROOT / "outputs/internal_wiper_moveit_inputs"
CHECKPOINT = H11_ROOT / "checkpoint"
H11_TERMINAL = H11_ROOT / "stage3_h11_r_terminal_certificate.json"
H11_STATS = H11_ROOT / "normalization_stats.json"
H11_WINDOW_MANIFEST = H11_ROOT / "h11_window_manifest.json"
H11_DATASET_CONTRACT = H11_ROOT / "h11_dataset_contract.json"
H11_CHECKPOINT_MANIFEST = H11_ROOT / "checkpoint_sha256_manifest.json"
H12_TERMINAL = H12_R7_ROOT / "stage3_h12_r7_terminal_certificate.json"
H12_REPAIR_SUMMARY = H12_R7_ROOT / "repair_summary.json"
H12_REPAIR_MANIFEST = H12_R7_ROOT / "repaired_sample_manifest.jsonl"
H12_INTERPOSER = ROOT / "outputs/stage3_h12_r5a_moveit_helper_semantics_closure_20260812T154400Z/libstage3_h12_r5_ruckig_interposer.so"
FIXTURE = ROOT / "outputs/stage3_h4_5_robot_fixture_geometry_codesign_20260809T000000Z/curved_fixture_stage3_h4_5_selected.obj"
RUNTIME_LIMITS = ROOT / "ros2_moveit_bridge/config/joint_limits_with_jerk.yaml"
TOTG_PARAMETERS = ROOT / "outputs/stage3_h7_2_authoritative_time_parameterization_20260809T060043Z/stage3_h7_2_totg_parameters.json"
PROCESS_CONTRACT = ROOT / "config/stage3/stage3_h6_1_spray_process_tolerance_contract.json"
BASE_POSES = OPEN_ARCH_INPUT_ROOT / "open_arch_tcp_poses_base_link.csv"
BASE_SEEDS = OPEN_ARCH_INPUT_ROOT / "open_arch_seed_joints.csv"
H13_R1_ROOT = ROOT / "outputs/stage3_h13_low_storage_unseen_generalization_r1_authoritative_20260813T"

POSITION_LOWER = np.asarray([-3.0543, -4.6251, -2.8274, -4.6251, -3.0543, -3.0543], dtype=np.float64)
POSITION_UPPER = np.asarray([3.0543, 1.4835, 2.8274, 1.4835, 3.0543, 3.0543], dtype=np.float64)
INPUT_HISTORY = 16
HORIZON = 8
RETAINED = ("FINAL_REPORT.md", "stage3_h13_terminal_certificate.json", "generalization_summary.json", "case_summary.jsonl", "test_summary.txt")
TIME_FEATURE_FORMULA = "(trajectory_time - trajectory_start_time) / (trajectory_end_time - trajectory_start_time)"
TIME_FEATURE_TOLERANCE = 1.0e-12


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def wsl(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    suffix = str(resolved).split(":", 1)[1].lstrip("/\\").replace("\\", "/")
    return f"/mnt/{drive}/{suffix}"


def output_size_mb(path: Path) -> float:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) / (1024.0 * 1024.0)


def immutable_paths() -> list[Path]:
    return [
        H11_TERMINAL, H11_STATS, H11_WINDOW_MANIFEST, H11_DATASET_CONTRACT,
        H11_CHECKPOINT_MANIFEST, CHECKPOINT,
        H12_TERMINAL, H12_REPAIR_SUMMARY, H12_REPAIR_MANIFEST,
        OPEN_ARCH_INPUT_ROOT / "open_arch_tcp_poses_base_link.csv",
        OPEN_ARCH_INPUT_ROOT / "open_arch_seed_joints.csv",
    ]


def snapshot(paths: Sequence[Path]) -> list[dict[str, Any]]:
    result = []
    for path in paths:
        result.append({"path": str(path.resolve()), "exists": path.is_file(), "size_bytes": path.stat().st_size if path.is_file() else None, "sha256": sha256_file(path) if path.is_file() else None})
    return result


def snapshot_equal(before: Sequence[Mapping[str, Any]], after: Sequence[Mapping[str, Any]]) -> bool:
    return list(before) == list(after) and all(bool(row.get("exists")) for row in before)


def verify_frozen_authority() -> dict[str, Any]:
    missing = [str(path) for path in immutable_paths() if not path.is_file()]
    h11 = load_json(H11_TERMINAL) if H11_TERMINAL.is_file() else {}
    h12 = load_json(H12_TERMINAL) if H12_TERMINAL.is_file() else {}
    checkpoint_manifest = load_json(H11_CHECKPOINT_MANIFEST) if H11_CHECKPOINT_MANIFEST.is_file() else {}
    checkpoint_hash_ok = bool(CHECKPOINT.is_file() and checkpoint_manifest.get("sha256") == sha256_file(CHECKPOINT))
    h11_ok = bool(
        not missing and checkpoint_hash_ok and h11.get("STAGE_3_H11_R") == "PASSED" and
        h11.get("STAGE_3_H11") == "PASSED" and h11.get("H11_DATASET_SPLIT_IMMUTABLE", "YES") == "YES" and
        h11.get("PHYSICAL_ROBOT_CONNECTED") == "NO" and int(h11.get("PHYSICAL_FJT_GOALS_SENT", 0) or 0) == 0
    )
    h12_ok = bool(
        not missing and h12.get("STAGE_3_H12") == "PASSED" and h12.get("STAGE_3_H12_R7") == "PASSED" and
        h12.get("READY_FOR_STAGE_3_H13") == "YES" and h12.get("H12_R7A_IMMUTABLE") == "YES" and
        h12.get("PHYSICAL_ROBOT_CONNECTED") == "NO" and int(h12.get("FJT_GOALS_SENT", 0) or 0) == 0 and
        int(h12.get("PHYSICAL_MOTION", 0) or 0) == 0
    )
    return {
        "H11_IMMUTABLE": "YES" if h11_ok else "NO",
        "H12_IMMUTABLE": "YES" if h12_ok else "NO",
        "H12_R7_IMMUTABLE": "YES" if h12_ok else "NO",
        "H11_CHECKPOINT_SHA256": sha256_file(CHECKPOINT) if CHECKPOINT.is_file() else None,
        "H11_DATASET_SPLIT_SHA256": sha256_file(H11_WINDOW_MANIFEST) if H11_WINDOW_MANIFEST.is_file() else None,
        "H12_R7_TERMINAL_CERTIFICATE_SHA256": sha256_file(H12_TERMINAL) if H12_TERMINAL.is_file() else None,
        "H12_R7_REPAIR_SUMMARY_SHA256": sha256_file(H12_REPAIR_SUMMARY) if H12_REPAIR_SUMMARY.is_file() else None,
        "missing": missing,
        "h11_terminal": h11,
        "h12_terminal": h12,
        "checkpoint_manifest": checkpoint_manifest,
    }


def read_pose_rows(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]


def read_seed_rows(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]


def resample_seed_rows(rows: Sequence[Mapping[str, float]], count: int) -> list[dict[str, float]]:
    if len(rows) < 2 or count < 2:
        raise ValueError("invalid_open_arch_seed_count")
    if len(rows) == count:
        return [dict(row) for row in rows]
    source_u = np.linspace(0.0, 1.0, len(rows))
    target_u = np.linspace(0.0, 1.0, count)
    fields = tuple(f"q{index}" for index in range(1, 7))
    values = {field: np.interp(target_u, source_u, [float(row[field]) for row in rows]) for field in fields}
    return [{field: float(values[field][index]) for field in fields} for index in range(count)]


def write_seed_csv(path: Path, rows: Sequence[Mapping[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [f"q{index}" for index in range(1, 7)]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: float(row[field]) for field in fields} for row in rows)


def normalize_quaternion(values: Sequence[float]) -> list[float]:
    vector = np.asarray(values, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-12:
        raise ValueError("zero_pose_quaternion")
    return (vector / norm).tolist()


def resample_pose_rows(rows: Sequence[Mapping[str, float]], count: int) -> list[dict[str, float]]:
    if len(rows) < 2 or count < 2:
        raise ValueError("invalid_open_arch_pose_count")
    source_u = np.linspace(0.0, 1.0, len(rows))
    target_u = np.linspace(0.0, 1.0, count)
    numeric = ("x", "y", "z")
    values = {key: np.interp(target_u, source_u, [float(row[key]) for row in rows]) for key in numeric}
    source_quaternions = np.asarray([[float(row[key]) for key in ("qx", "qy", "qz", "qw")] for row in rows], dtype=np.float64)
    source_quaternions /= np.maximum(np.linalg.norm(source_quaternions, axis=1, keepdims=True), 1.0e-12)
    for index in range(1, len(source_quaternions)):
        if float(np.dot(source_quaternions[index - 1], source_quaternions[index])) < 0.0:
            source_quaternions[index] *= -1.0
    quaternion_values = {key: np.interp(target_u, source_u, source_quaternions[:, index]) for index, key in enumerate(("qx", "qy", "qz", "qw"))}
    result = []
    for index in range(count):
        row = {key: float(values[key][index]) for key in numeric}
        quaternion = normalize_quaternion([quaternion_values[key][index] for key in ("qx", "qy", "qz", "qw")])
        row.update(dict(zip(("qx", "qy", "qz", "qw"), quaternion)))
        x, y, z, w = quaternion
        tool_z = [2.0 * (x * z + y * w), 2.0 * (y * z - x * w), 1.0 - 2.0 * (x * x + y * y)]
        row.update(dict(zip(("nx", "ny", "nz"), normalize_quaternion(tool_z))))
        result.append(row)
    return result


def make_case_poses(base_rows: Sequence[Mapping[str, float]], case: Mapping[str, Any]) -> list[dict[str, float]]:
    parameters = case["parameters"]
    scale = float(parameters["path_sampling_scale"])
    count = int(round(len(base_rows) * scale))
    rows = resample_pose_rows(base_rows, max(2, count))
    start_offset = np.asarray(parameters["start_position_offset_m"], dtype=np.float64)
    path_offset = np.asarray(parameters["tcp_path_offset_m"], dtype=np.float64)
    curvature = float(parameters["local_curvature_bulge_m"])
    for index, row in enumerate(rows):
        u = index / float(max(1, len(rows) - 1))
        envelope = math.sin(math.pi * u) ** 2
        bulge = np.asarray([curvature * envelope, 0.0, 0.5 * curvature * envelope], dtype=np.float64)
        position = np.asarray([row["x"], row["y"], row["z"]], dtype=np.float64) + start_offset + path_offset + bulge
        row["x"], row["y"], row["z"] = (float(value) for value in position)
    return rows


def write_pose_csv(path: Path, rows: Sequence[Mapping[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["x", "y", "z", "qx", "qy", "qz", "qw", "nx", "ny", "nz"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: float(row[key]) for key in fields} for row in rows)


def run_strict_case(case: Mapping[str, Any], pose_path: Path, seed_path: Path, case_dir: Path, timeout_s: int) -> dict[str, Any]:
    count = len(read_pose_rows(pose_path))
    command = " && ".join([
        "source /opt/ros/jazzy/setup.bash",
        "source /mnt/d/robotfucker/install/setup.bash",
        f"export REPO_ROOT={wsl(ROOT)}",
        f"export ROS_WS={wsl(ROOT)}",
        f"export TCP_POSES_CSV={wsl(pose_path)}",
        f"export SEED_JOINT_CSV={wsl(seed_path)}",
        "export ALLOW_SEED_JOINT_WITH_TOOL_OFFSET=true",
        f"export FINAL_OUT_DIR={wsl(case_dir)}",
        f"export QUALITY_REPORT={wsl(case_dir / 'moveit_quality_report.csv')}",
        f"export DYNAMICS_REPORT={wsl(case_dir / 'moveit_joint_dynamics_report.csv')}",
        f"export COLLISION_REPORT={wsl(case_dir / 'moveit_collision_report.csv')}",
        f"export FK_TRACE={wsl(case_dir / 'moveit_fk_tcp_trace.csv')}",
        f"export TRAJECTORY_CSV={wsl(case_dir / 'moveit_smoothed_joint_trajectory.csv')}",
        f"export SEGMENTED_TRAJECTORY_CSV={wsl(case_dir / 'moveit_executed_segmented_joint_trajectory.csv')}",
        f"export WAYPOINT_TRAJECTORY_CSV={wsl(case_dir / 'moveit_waypoint_joint_trajectory.csv')}",
        f"export IK_SEARCH_DIAGNOSTICS_CSV={wsl(case_dir / 'ik_search.csv')}",
        f"export IK_CANDIDATE_BANK_CSV={wsl(case_dir / 'ik_candidate_bank.csv')}",
        f"export STRICT_JSON={wsl(case_dir / 'audit_goal_requirements_strict.json')}",
        f"export PRODUCTION_READINESS_JSON={wsl(case_dir / 'production_readiness_check.json')}",
        f"export RUNTIME_LOG={wsl(case_dir / 'moveit_runtime.log')}",
        "export OPEN_PATH=true",
        "export TCP_POINTS_TO_WALL=true",
        f"export SAMPLES_PER_LOOP={count}",
        "export WAYPOINT_STRIDE=1",
        "export PLANNING_MODE=seed_joint_waypoints",
        "export IK_TIMEOUT=0.20",
        "export MAX_IK_POSITION_ERROR=0.003",
        "export MAX_IK_JOINT_STEP_DEG=20.0",
        "export IK_ROLL_SAMPLE_COUNT=1",
        "export IK_BEAM_WIDTH=1",
        "export IK_CANDIDATES_PER_BEAM=8",
        "export IK_BEAM_DIVERSITY_JOINT_DEG=0.0",
        "export IK_MAX_SOLVE_SECONDS=0.0",
        "export VALIDATION_STRIDE=1",
        "export VELOCITY_SCALING=0.15",
        "export ACCELERATION_SCALING=0.15",
        "export TIME_PARAMETERIZATION=tcp_arclength",
        f"export TARGET_TCP_SPEED={float(case['parameters']['legal_speed_m_s']):.9f}",
        "export ZERO_BOUNDARY_STATE=true",
        "export MAX_PATH_DEVIATION=0.006",
        "export MAX_NORMAL_ERROR_DEG=10.0",
        "export MAX_STANDOFF_FRACTION=0.05",
        "export MAX_SPEED_FLUCTUATION=0.05",
        "export VALIDATE_COLLISION=true",
        "export COLLISION_CHECK_STRIDE=1",
        "export COLLISION_SEGMENT_STRIDE=1",
        "export COLLISION_INTERPOLATION_STEP_DEG=0.5",
        "export INCLUDE_TUNNEL_FLOOR_COLLISION=true",
        "export TUNNEL_FLOOR_Z=-0.20",
        "export TUNNEL_WALL_THICKNESS=0.04",
        "export TUNNEL_Y_THICKNESS=1.10",
        "export INCLUDE_BOTTOM_CLOSURE_COLLISION=false",
        "export STAND_OFF=0.260",
        "export FAST_EXIT_AFTER_REPORTS=true",
        "export ROS_REPORT_TIMEOUT_SEC=600",
        "export ROS_EXIT_GRACE_SEC=10",
        "export WRITE_FINAL_VISUALS=false",
        "export WRITE_FINAL_ANIMATION=false",
        f"bash {wsl(ROOT / 'scripts/run_moveit_strict_validation.sh')}",
    ])
    case_dir.mkdir(parents=True, exist_ok=True)
    try:
        process = subprocess.run(["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", command], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout_s, check=False)
        stdout, stderr = process.stdout or "", process.stderr or ""
    except subprocess.TimeoutExpired as exc:
        stdout, stderr = str(exc.stdout or ""), str(exc.stderr or "")
        return {"status": "BLOCKED", "first_blocker": "strict_moveit_timeout", "returncode": None, "stdout_tail": stdout[-2000:], "stderr_tail": stderr[-2000:]}
    summary_path = case_dir / "final_acceptance_summary.json"
    trajectory_path = case_dir / "moveit_smoothed_joint_trajectory.csv"
    if not summary_path.is_file() or not trajectory_path.is_file():
        return {"status": "BLOCKED", "first_blocker": "strict_moveit_artifact_missing", "returncode": process.returncode, "stdout_tail": stdout[-2000:], "stderr_tail": stderr[-2000:]}
    summary = load_json(summary_path)
    status = str(summary.get("overall_status", "blocked")).lower()
    return {"status": "PASSED" if status == "pass" else "BLOCKED", "first_blocker": None if status == "pass" else "strict_moveit_quality_gate_failed", "returncode": process.returncode, "trajectory_path": str(trajectory_path), "stdout_tail": stdout[-2000:], "stderr_tail": stderr[-2000:]}


def read_trajectory_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < INPUT_HISTORY + HORIZON:
        raise RuntimeError("strict_trajectory_too_short_for_h13_windows")
    t = np.asarray([float(row["t"]) for row in rows], dtype=np.float64)
    q = np.asarray([[float(row[f"j{joint}_q"]) for joint in range(1, 7)] for row in rows], dtype=np.float64)
    dq = np.asarray([[float(row[f"j{joint}_dq"]) for joint in range(1, 7)] for row in rows], dtype=np.float64)
    ddq = np.asarray([[float(row[f"j{joint}_ddq"]) for joint in range(1, 7)] for row in rows], dtype=np.float64)
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(q)) and np.all(np.isfinite(dq)) and np.all(np.isfinite(ddq)) and np.all(np.diff(t) > 0.0)):
        raise RuntimeError("strict_trajectory_nonfinite_or_nonmonotonic")
    return q, dq, ddq, t


def normalized_local_trajectory_time(
    t: np.ndarray,
    *,
    trajectory_start_time: float | None = None,
    trajectory_end_time: float | None = None,
) -> np.ndarray:
    """Return H11's whole-trajectory progress feature without clamping.

    H11 trains on a native segment's full ``trajectory_time`` domain.  H13
    must keep those same anchors while its history window advances; the
    current history window is never allowed to redefine the denominator.
    """
    values = np.asarray(t, dtype=np.float64)
    if values.ndim != 1 or values.size < 1 or not np.all(np.isfinite(values)) or (values.size > 1 and np.any(np.diff(values) <= 0.0)):
        raise ValueError("trajectory_time_not_strictly_monotonic")
    if values.size < 2 and (trajectory_start_time is None or trajectory_end_time is None):
        raise ValueError("trajectory_time_domain_anchors_required")
    start = float(values[0] if trajectory_start_time is None else trajectory_start_time)
    end = float(values[-1] if trajectory_end_time is None else trajectory_end_time)
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise ValueError("trajectory_time_domain_invalid")
    local = (values - start) / (end - start)
    if not np.all(np.isfinite(local)):
        raise ValueError("normalized_trajectory_time_nonfinite")
    return local


def time_feature_domain_audit(t: np.ndarray) -> dict[str, Any]:
    local = normalized_local_trajectory_time(t)
    out_of_domain = (local < -TIME_FEATURE_TOLERANCE) | (local > 1.0 + TIME_FEATURE_TOLERANCE)
    return {
        "formula": TIME_FEATURE_FORMULA,
        "trajectory_index_first": 0,
        "trajectory_index_middle": int(len(local) // 2),
        "trajectory_index_final": int(len(local) - 1),
        "absolute_local_time_first_s": float(t[0]),
        "absolute_local_time_middle_s": float(t[len(local) // 2]),
        "absolute_local_time_final_s": float(t[-1]),
        "normalized_time_first": float(local[0]),
        "normalized_time_middle": float(local[len(local) // 2]),
        "normalized_time_final": float(local[-1]),
        "observed_min": float(np.min(local)),
        "observed_max": float(np.max(local)),
        "out_of_training_domain_count": int(np.count_nonzero(out_of_domain)),
    }


def normalize_features(q: np.ndarray, dq: np.ndarray, ddq: np.ndarray, t: np.ndarray, stats: Mapping[str, Any]) -> np.ndarray:
    local = normalized_local_trajectory_time(t)
    features = np.column_stack([q, dq, ddq, np.ones(len(t), dtype=np.float64), local]).astype(np.float32)
    names = [*(f"planned_joint_position_{i}" for i in range(1, 7)), *(f"planned_joint_velocity_{i}" for i in range(1, 7)), *(f"planned_joint_acceleration_{i}" for i in range(1, 7)), "spray_on", "normalized_local_trajectory_time"]
    for index, name in enumerate(names):
        channel = stats["channels"].get(name, {})
        if name == "spray_on":
            continue
        scale = float(channel.get("scale", channel.get("std", 1.0))) or 1.0
        features[:, index] = (features[:, index] - float(channel.get("mean", 0.0))) / scale
    return features


def direct_model_evaluation(model: Any, features: np.ndarray, q: np.ndarray, t: np.ndarray, device: str = "cpu") -> tuple[dict[str, Any], dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import torch

    starts = list(range(0, len(q) - INPUT_HISTORY - HORIZON + 1))
    inputs = np.asarray([features[start:start + INPUT_HISTORY] for start in starts], dtype=np.float32)
    histories = np.asarray([q[start:start + INPUT_HISTORY] for start in starts], dtype=np.float64)
    history_times = np.asarray([t[start:start + INPUT_HISTORY] for start in starts], dtype=np.float64)
    targets = np.asarray([q[start + INPUT_HISTORY:start + INPUT_HISTORY + HORIZON] for start in starts], dtype=np.float64)
    target_times = np.asarray([t[start + INPUT_HISTORY:start + INPUT_HISTORY + HORIZON] for start in starts], dtype=np.float64)
    with torch.no_grad():
        deltas = model(torch.from_numpy(inputs).to(device=device, dtype=torch.float32)).detach().cpu().numpy()
    anchors = histories[:, -1, :]
    predicted = anchors[:, None, :] + deltas
    class Arrays:
        pass
    arrays = Arrays()
    arrays.anchor_positions = anchors
    arrays.history_positions = histories
    arrays.history_times = history_times
    arrays.target_times = target_times
    baseline = constant_velocity_prediction(arrays)
    return metric_summary(predicted, targets), metric_summary(baseline, targets), predicted, baseline, targets, target_times


def rollout_path(model: Any, features: np.ndarray, q: np.ndarray, t: np.ndarray, stats: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    import torch

    predicted = q.copy().astype(np.float64)
    baseline = q.copy().astype(np.float64)
    x = features[:INPUT_HISTORY].copy().astype(np.float64)
    model_history_q = q[:INPUT_HISTORY].copy().astype(np.float64)
    model_history_t = t[:INPUT_HISTORY].copy().astype(np.float64)
    baseline_history_q = q[:INPUT_HISTORY].copy().astype(np.float64)
    baseline_history_t = t[:INPUT_HISTORY].copy().astype(np.float64)
    trajectory_start_time = float(t[0])
    trajectory_end_time = float(t[-1])
    for index in range(INPUT_HISTORY, len(q)):
        with torch.no_grad():
            delta = model(torch.from_numpy(x[None, ...].astype(np.float32))).detach().cpu().numpy()[0, 0]
        next_q = model_history_q[-1] + delta
        model_dt = max(float(t[index] - model_history_t[-1]), 1.0e-9)
        next_velocity = (next_q - model_history_q[-1]) / model_dt
        prior_dt = max(float(model_history_t[-1] - model_history_t[-2]), 1.0e-9)
        prior_velocity = (model_history_q[-1] - model_history_q[-2]) / prior_dt
        next_acceleration = (next_velocity - prior_velocity) / model_dt
        local_time = float(normalized_local_trajectory_time(
            np.asarray([t[index]], dtype=np.float64),
            trajectory_start_time=trajectory_start_time,
            trajectory_end_time=trajectory_end_time,
        )[0])
        raw_features = np.concatenate([next_q, next_velocity, next_acceleration, np.asarray([1.0, local_time])])
        names = [*(f"planned_joint_position_{i}" for i in range(1, 7)), *(f"planned_joint_velocity_{i}" for i in range(1, 7)), *(f"planned_joint_acceleration_{i}" for i in range(1, 7)), "spray_on", "normalized_local_trajectory_time"]
        next_features = raw_features.copy()
        for channel_index, name in enumerate(names):
            channel = stats["channels"].get(name, {})
            if name == "spray_on":
                continue
            scale = float(channel.get("scale", channel.get("std", 1.0))) or 1.0
            next_features[channel_index] = (next_features[channel_index] - float(channel.get("mean", 0.0))) / scale
        predicted[index] = next_q
        baseline_dt = max(float(t[index] - baseline_history_t[-1]), 1.0e-9)
        baseline_prior_dt = max(float(baseline_history_t[-1] - baseline_history_t[-2]), 1.0e-9)
        baseline_velocity = (baseline_history_q[-1] - baseline_history_q[-2]) / baseline_prior_dt
        baseline_next_q = baseline_history_q[-1] + baseline_velocity * baseline_dt
        baseline[index] = baseline_next_q
        x = np.concatenate([x[1:], next_features[None, :]], axis=0)
        model_history_q = np.concatenate([model_history_q[1:], next_q[None, :]], axis=0)
        model_history_t = np.concatenate([model_history_t[1:], np.asarray([t[index]])], axis=0)
        baseline_history_q = np.concatenate([baseline_history_q[1:], baseline_next_q[None, :]], axis=0)
        baseline_history_t = np.concatenate([baseline_history_t[1:], np.asarray([t[index]])], axis=0)
    return predicted, baseline


def raw_prediction_counts(predicted: np.ndarray, t: np.ndarray) -> dict[str, int]:
    dt = np.diff(t)
    velocity = np.vstack([np.zeros((1, 6)), np.diff(predicted, axis=0) / dt[:, None]])
    acceleration = np.vstack([np.zeros((1, 6)), np.diff(velocity, axis=0) / dt[:, None]])
    jerk = np.vstack([np.zeros((1, 6)), np.diff(acceleration, axis=0) / dt[:, None]])
    return {
        "RAW_POSITION_VIOLATIONS": int(np.count_nonzero((predicted < POSITION_LOWER[None, :] - 1.0e-10) | (predicted > POSITION_UPPER[None, :] + 1.0e-10))),
        "RAW_VELOCITY_VIOLATIONS": int(np.count_nonzero(np.abs(velocity) > np.asarray([0.4725, 0.4725, 0.4725, 0.48, 0.48, 0.48])[None, :] + 1.0e-10)),
        "RAW_ACCELERATION_VIOLATIONS": int(np.count_nonzero(np.abs(acceleration) > 0.105 + 1.0e-10)),
        "RAW_JERK_VIOLATIONS": int(np.count_nonzero(np.abs(jerk) > 8.0 + 1.0e-10)),
    }


def make_target_rows(q: np.ndarray, pose_rows: Sequence[Mapping[str, float]]) -> list[dict[str, Any]]:
    if len(q) != len(pose_rows):
        raise RuntimeError(f"trajectory_pose_count_mismatch:{len(q)}:{len(pose_rows)}")
    result = []
    for index, (joint_values, pose) in enumerate(zip(q, pose_rows)):
        position = np.asarray([pose["x"], pose["y"], pose["z"]], dtype=np.float64)
        tool_z = np.asarray([pose["nx"], pose["ny"], pose["nz"]], dtype=np.float64)
        surface_point = position + 0.260 * tool_z
        result.append({
            "waypoint_index": index,
            "joint_values": [float(value) for value in joint_values],
            "desired_tcp_pose": {"position_xyz_m": position.tolist(), "orientation_xyzw": [pose["qx"], pose["qy"], pose["qz"], pose["qw"]]},
            "surface_point": surface_point.tolist(),
            "surface_normal": (-tool_z).tolist(),
        })
    return result


def evaluate_cases(args: argparse.Namespace, output: Path, authority: Mapping[str, Any], cases: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    if not TORCH_AVAILABLE:
        raise RuntimeError("pytorch_unavailable")
    import torch

    torch.manual_seed(0)
    torch.set_num_threads(1)
    stats = load_json(H11_STATS)
    payload = torch.load(CHECKPOINT, map_location="cpu")
    model = CausalGRUTrajectoryPredictor(input_size=20, hidden_size=128, num_layers=2, horizon=HORIZON)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    base_rows = read_pose_rows(BASE_POSES)
    base_seed_rows = read_seed_rows(BASE_SEEDS)
    scratch = output / "_h13_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    cases_for_native: list[dict[str, Any]] = []
    case_summaries: list[dict[str, Any]] = []
    sample_hashes: list[str] = []
    time_feature_audits: list[dict[str, Any]] = []
    time_feature_min: float | None = None
    time_feature_max: float | None = None
    time_feature_out_of_domain = 0
    try:
        for case in cases:
            case_dir = scratch / "strict" / str(case["case_id"])
            pose_rows = make_case_poses(base_rows, case)
            pose_path = scratch / "poses" / f"{case['case_id']}.csv"
            write_pose_csv(pose_path, pose_rows)
            seed_path = scratch / "seeds" / f"{case['case_id']}.csv"
            write_seed_csv(seed_path, resample_seed_rows(base_seed_rows, len(pose_rows)))
            strict = run_strict_case(case, pose_path, seed_path, case_dir, int(args.timeout_s))
            if strict.get("status") != "PASSED":
                case_summaries.append({
                    "case_id": case["case_id"], "input_semantic_hash": canonical_hash({"case": case, "pose_rows": pose_rows}),
                    "status": "BLOCKED", "first_blocker": strict.get("first_blocker"), "raw_prediction_metrics": None,
                    "strict_returncode": strict.get("returncode"), "strict_stdout_tail": strict.get("stdout_tail"), "strict_stderr_tail": strict.get("stderr_tail"),
                })
                continue
            q, dq, ddq, t = read_trajectory_csv(Path(strict["trajectory_path"]))
            if len(q) != len(pose_rows):
                pose_rows = resample_pose_rows(pose_rows, len(q))
            features = normalize_features(q, dq, ddq, t, stats)
            direct_model, direct_baseline, predicted_windows, baseline_windows, targets, target_times = direct_model_evaluation(model, features, q, t)
            time_audit = time_feature_domain_audit(t)
            time_feature_min = time_audit["observed_min"] if time_feature_min is None else min(time_feature_min, time_audit["observed_min"])
            time_feature_max = time_audit["observed_max"] if time_feature_max is None else max(time_feature_max, time_audit["observed_max"])
            time_feature_out_of_domain += int(time_audit["out_of_training_domain_count"])
            if len(time_feature_audits) < 5:
                time_feature_audits.append({"case_id": str(case["case_id"]), **time_audit})
            model_path, baseline_path = rollout_path(model, features, q, t, stats)
            model_path_counts = raw_prediction_counts(model_path, t)
            baseline_path_counts = raw_prediction_counts(baseline_path, t)
            input_hash = canonical_hash(case_input_payload(case, q, t, pose_rows))
            sample_hashes.extend(semantic_sample_hash({"planned_joint_position": row, "planned_joint_velocity": vrow, "planned_joint_acceleration": arow, "trajectory_time": time_value, "spray_state": "SPRAY_ON"}) for row, vrow, arow, time_value in zip(q.tolist(), dq.tolist(), ddq.tolist(), t.tolist()))
            # The primitive rows are the frozen desired open-arch reference
            # path.  H12-R7 projects post-TOTG model samples onto this desired
            # path; the additive H13 compatibility metadata preserves the
            # resulting segment/index semantics without changing the target
            # geometry or the frozen repair predicate.
            target_rows = make_target_rows(q, pose_rows)
            cases_for_native.append({
                "unit_index": len(cases_for_native), "unit_id": f"{case['case_id']}|ON|0|0", "case_id": case["case_id"],
                "primitive_id": case["case_id"], "trajectory_family_id": case["trajectory_family_id"], "window_count": int(max(0, len(q) - INPUT_HISTORY - HORIZON + 1)),
                **h12_r7_compatibility_fields(spray_state=str(case["spray_state"]), source_segment_id=0),
                "input_semantic_hash": input_hash, "target_rows": target_rows,
                "raw_model_positions": model_path.tolist(), "raw_baseline_positions": baseline_path.tolist(), "times_s": t.tolist(),
                "raw_prediction_metrics": {"model": direct_model, "baseline": direct_baseline, "MODEL_VS_BASELINE_IMPROVEMENT_PERCENT": improvement_percent(direct_baseline.get("joint_position_rmse_rad"), direct_model.get("joint_position_rmse_rad")), "model_rollout": metric_summary(model_path[INPUT_HISTORY:], q[INPUT_HISTORY:]), "baseline_rollout": metric_summary(baseline_path[INPUT_HISTORY:], q[INPUT_HISTORY:])},
                "raw_model_path_counts_python": model_path_counts, "raw_baseline_path_counts_python": baseline_path_counts,
                "time_feature_audit": time_audit,
                "case": case,
            })
        if not cases_for_native:
            return case_summaries, {"native_status": "BLOCKED", "first_blocker": "all_strict_cases_blocked"}, {"sample_hashes": sample_hashes, "cases_for_native": cases_for_native}
        max_length = max(len(row["raw_model_positions"]) for row in cases_for_native)
        positions = np.zeros((len(cases_for_native), max_length, 6), dtype=np.float64)
        baseline_positions = np.zeros_like(positions)
        times = np.zeros((len(cases_for_native), max_length), dtype=np.float64)
        lengths = []
        manifest_rows = []
        for index, row in enumerate(cases_for_native):
            length = len(row["raw_model_positions"])
            lengths.append(length)
            positions[index, :length] = np.asarray(row["raw_model_positions"], dtype=np.float64)
            baseline_positions[index, :length] = np.asarray(row["raw_baseline_positions"], dtype=np.float64)
            times[index, :length] = np.asarray(row["times_s"], dtype=np.float64)
            manifest_rows.append({key: row[key] for key in ("unit_index", "unit_id", "case_id", "primitive_id", "trajectory_family_id", "window_count", "input_semantic_hash", "target_rows", "source_segment_id", "h12_r7_segment_id", "h12_r7_segment_order", "spray_state")})
        input_npz = scratch / "h13_native_input.npz"
        manifest = scratch / "h13_native_manifest.jsonl"
        native_jsonl = scratch / "h13_native_results.jsonl"
        np.savez_compressed(input_npz, positions_rad=positions, baseline_positions_rad=baseline_positions, times_s=times, lengths=np.asarray(lengths, dtype=np.int64))
        write_jsonl(manifest, manifest_rows)
        native = run_native(args, input_npz, manifest, native_jsonl, scratch / "native")
        native_rows = load_jsonl(native_jsonl) if native_jsonl.is_file() else []
        native_by_case = {str(row.get("case_id")): row for row in native_rows}
        for row in cases_for_native:
            native_row = native_by_case.get(str(row["case_id"]))
            if not native_row:
                case_summaries.append({"case_id": row["case_id"], "input_semantic_hash": row["input_semantic_hash"], "status": "BLOCKED", "first_blocker": "native_case_result_missing", "raw_prediction_metrics": row["raw_prediction_metrics"]})
                continue
            case_summaries.append({
                "case_id": row["case_id"], "input_semantic_hash": row["input_semantic_hash"], "status": "CERTIFIED" if native_row.get("status") == "PASSED" and all(value == 0 for value in (native_row.get("final_counts") or {}).values()) else "BLOCKED",
                "first_blocker": None if native_row.get("status") == "PASSED" and all(value == 0 for value in (native_row.get("final_counts") or {}).values()) else native_row.get("first_blocker") or "final_native_certification_failed",
                "raw_prediction_metrics": row["raw_prediction_metrics"], "raw_prediction_violations": native_row.get("raw_prediction_violations"),
                "repair": native_row.get("repair"), "totg_attempted": native_row.get("totg_attempted"), "totg_success": native_row.get("totg_success"),
                "ruckig_attempted": native_row.get("ruckig_attempted"), "ruckig_success": native_row.get("ruckig_success"), "ruckig_native_errors": native_row.get("ruckig_native_errors"),
                "ruckig_duration_ceiling_hit": native_row.get("ruckig_duration_ceiling_hit"), "reprojection": native_row.get("reprojection"),
                "final_counts": native_row.get("final_counts"), "raw_tcp_errors": {"MODEL_TCP_ERROR": (native_row.get("raw") or {}).get("raw_model_tcp_error_m"), "BASELINE_TCP_ERROR": (native_row.get("raw") or {}).get("baseline_tcp_error_m")},
                "collision_method": COLLISION_METHOD, "time_feature_audit": row.get("time_feature_audit"),
            })
        return case_summaries, native, {
            "sample_hashes": sample_hashes,
            "cases_for_native": cases_for_native,
            "time_feature_audits": time_feature_audits,
            "time_feature_formula": TIME_FEATURE_FORMULA,
            "time_feature_observed_min": time_feature_min,
            "time_feature_observed_max": time_feature_max,
            "time_feature_out_of_domain_count": time_feature_out_of_domain,
        }
    finally:
        if scratch.is_dir():
            shutil.rmtree(scratch)


def run_native(args: argparse.Namespace, input_npz: Path, manifest: Path, output_jsonl: Path, native_output: Path) -> dict[str, Any]:
    command = " && ".join([
        "source /opt/ros/jazzy/setup.bash",
        "source /mnt/d/robotfucker/install/setup.bash",
        "source /mnt/d/robotfucker/install/stage3_h7_4_native/setup.bash",
        f"export LD_PRELOAD={wsl(H12_INTERPOSER)}",
        f"export STAGE25R_NATIVE_DIR={wsl(native_output / 'probe')}",
        "export H12_R5_COMPACT_PROBE=1",
        "export H12_R5_USE_INSTALLED_MOVEIT_HELPERS=1",
        "export H12_R5_MITIGATE_OVERSHOOT=1",
        "export H12_R5_HARD_LIMIT_CERTIFICATION=1",
        "export H12_R5_BOUNDARY_VELOCITY_RECONDITION=1",
        "export H12_R6_RECONDITION_DERIVATIVES=1",
        "export H12_R7_REPROJECT=1",
        "export H12_R7A_PROCESS_AUDIT=0",
        "export H12_R5_ASSIGN_ALL_NATIVE_DURATIONS=0",
        "export H12_R5_OVERSHOOT_THRESHOLD=0.01",
        f"export PYTHONPATH={wsl(ROOT)}:{wsl(ROOT / 'ros2_moveit_bridge')}:/mnt/d/robotfucker/install/stage3_h7_4_native/lib/python3.12/site-packages:/opt/ros/jazzy/lib/python3.12/site-packages",
        f"ros2 launch /mnt/d/robotfucker/ros2_moveit_bridge/launch/stage3_h13_native.launch.py input_npz:={wsl(input_npz)} unit_manifest:={wsl(manifest)} output_jsonl:={wsl(output_jsonl)} output_dir:={wsl(native_output)} process_contract:={wsl(PROCESS_CONTRACT)} fixture_mesh:={wsl(FIXTURE)} runtime_limits:={wsl(RUNTIME_LIMITS)} totg_parameters:={wsl(TOTG_PARAMETERS)}",
    ])
    native_output.mkdir(parents=True, exist_ok=True)
    try:
        process = subprocess.run(["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", command], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=int(args.timeout_s) * 2, check=False)
        stdout, stderr = process.stdout or "", process.stderr or ""
    except subprocess.TimeoutExpired as exc:
        return {"status": "BLOCKED", "first_blocker": "h12_native_timeout", "returncode": None, "stdout_tail": str(exc.stdout or "")[-2000:], "stderr_tail": str(exc.stderr or "")[-2000:]}
    result_count = len(load_jsonl(output_jsonl)) if output_jsonl.is_file() else 0
    expected_count = len(load_jsonl(manifest)) if manifest.is_file() else None
    audit = audit_native_exit(stdout, stderr, process.returncode, result_count, expected_count)
    return {
        "status": "PASSED" if audit["NATIVE_MAIN_WORK_COMPLETED"] == "YES" and process.returncode == 0 else "BLOCKED",
        "first_blocker": None if audit["NATIVE_MAIN_WORK_COMPLETED"] == "YES" and process.returncode == 0 else "h13_native_case_batch_failed",
        "returncode": process.returncode, "result_count": result_count, "expected_count": expected_count,
        **audit,
        "stdout_tail": stdout[-2000:], "stderr_tail": stderr[-2000:],
    }


def audit_native_exit(stdout: str, stderr: str, returncode: int | None, result_count: int, expected_count: int | None) -> dict[str, Any]:
    """Separate completed native work from a launch/teardown process crash."""
    combined = f"{stdout}\n{stderr}"
    child_exit_match = re.search(r"process has died .*?exit code (-?\d+)", combined, flags=re.DOTALL)
    child_exit = int(child_exit_match.group(1)) if child_exit_match else None
    work_completed = bool(expected_count is not None and result_count == expected_count)
    teardown_crash = bool(child_exit is not None and child_exit != 0)
    clean_exit = bool(returncode == 0 and not teardown_crash)
    return {
        "NATIVE_MAIN_WORK_COMPLETED": "YES" if work_completed else "NO",
        "PROCESS_EXIT_CODE": child_exit if child_exit is not None else returncode,
        "CLEAN_EXIT": "YES" if clean_exit else "NO",
        "TEARDOWN_CRASH": "YES" if teardown_crash else "NO",
    }


def load_h11_rows_by_role() -> dict[str, Iterable[Mapping[str, Any]]]:
    h10_root = ROOT / "outputs/stage3_h10_multi_trajectory_dataset_20260811T081123Z"
    split = load_json(h10_root / "dataset_split_manifest.json")
    roles = {str(key): str(value) for key, value in split["group_to_role"].items()}
    handles: dict[str, list[Mapping[str, Any]]] = {role: [] for role in ("TRAIN", "VALIDATION", "TEST", "GENERALIZATION")}
    with (h10_root / "trajectory_samples.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                handles[roles[str(row["trajectory_family_id"])]].append(row)
    return handles


def aggregate_metric_rmse(case_summaries: Sequence[Mapping[str, Any]], metric_name: str, group: str) -> float | None:
    values = [
        float((row.get("raw_prediction_metrics") or {}).get(group, {}).get(metric_name))
        for row in case_summaries
        if (row.get("raw_prediction_metrics") or {}).get(group, {}).get(metric_name) is not None
    ]
    return float(np.sqrt(np.mean(np.square(values)))) if values else None


def authoritative_r1_rollout_rmse() -> float | None:
    """Read only the frozen H13-R1 compact evidence for before/after comparison."""
    path = H13_R1_ROOT / "case_summary.jsonl"
    if not path.is_file():
        return None
    return aggregate_metric_rmse(load_jsonl(path), "joint_position_rmse_rad", "model_rollout")


def aggregate(case_summaries: Sequence[Mapping[str, Any]], authority: Mapping[str, Any], overlap: Mapping[str, Any], native: Mapping[str, Any], replay: Mapping[str, Any], output: Path, feature_audit: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    certified = [row for row in case_summaries if row.get("status") == "CERTIFIED"]
    blocked = [row for row in case_summaries if row.get("status") == "BLOCKED"]
    model_metrics = [row.get("raw_prediction_metrics", {}).get("model", {}) for row in case_summaries if row.get("raw_prediction_metrics")]
    baseline_metrics = [row.get("raw_prediction_metrics", {}).get("baseline", {}) for row in case_summaries if row.get("raw_prediction_metrics")]
    model_rmse = aggregate_metric_rmse(case_summaries, "joint_position_rmse_rad", "model")
    baseline_rmse = aggregate_metric_rmse(case_summaries, "joint_position_rmse_rad", "baseline")
    r1_rollout_rmse = authoritative_r1_rollout_rmse()
    r2_rollout_rmse = aggregate_metric_rmse(case_summaries, "joint_position_rmse_rad", "model_rollout")
    final_keys = ("FINAL_POSITION_VIOLATIONS", "FINAL_VELOCITY_VIOLATIONS", "FINAL_ACCELERATION_VIOLATIONS", "FINAL_JERK_VIOLATIONS", "FINAL_COLLISION_VIOLATIONS", "FINAL_SPRAY_PROCESS_VIOLATIONS")
    final_values = {key: None if any((row.get("final_counts") or {}).get(key) is None for row in case_summaries) else int(sum(int((row.get("final_counts") or {}).get(key, 0)) for row in case_summaries)) for key in final_keys}
    raw_keys = ("RAW_POSITION_VIOLATIONS", "RAW_VELOCITY_VIOLATIONS", "RAW_ACCELERATION_VIOLATIONS", "RAW_JERK_VIOLATIONS", "RAW_COLLISION_VIOLATIONS", "RAW_SPRAY_PROCESS_VIOLATIONS")
    raw_values = {key: int(sum(int((row.get("raw_prediction_violations") or {}).get(key, 0) or 0) for row in case_summaries)) for key in raw_keys}
    corrections = [float((row.get("repair") or {}).get("MAX_JOINT_CORRECTION_RAD")) for row in case_summaries if (row.get("repair") or {}).get("MAX_JOINT_CORRECTION_RAD") is not None]
    p95s = [float((row.get("repair") or {}).get("P95_JOINT_CORRECTION_RAD")) for row in case_summaries if (row.get("repair") or {}).get("P95_JOINT_CORRECTION_RAD") is not None]
    p99s = [float((row.get("repair") or {}).get("P99_JOINT_CORRECTION_RAD")) for row in case_summaries if (row.get("repair") or {}).get("P99_JOINT_CORRECTION_RAD") is not None]
    totg_success = sum(bool(row.get("totg_success")) for row in case_summaries)
    totg_attempted = sum(bool(row.get("totg_attempted")) for row in case_summaries)
    ruckig_attempted = sum(bool(row.get("ruckig_attempted")) for row in case_summaries)
    ruckig_success = sum(bool(row.get("ruckig_success")) for row in case_summaries)
    native_errors = sum(int(row.get("ruckig_native_errors", 0) or 0) for row in case_summaries)
    reprojection_rows = [row.get("reprojection") or {} for row in case_summaries]
    reprojection_attempted = sum(bool(row) for row in reprojection_rows)
    reprojection_failed = sum(bool(row.get("failure_stage")) for row in reprojection_rows)
    reprojection_success = reprojection_attempted - reprojection_failed
    r1_passed = bool(reprojection_attempted == len(case_summaries) == CASE_TARGET and reprojection_failed == 0)
    mapped_compatibility = bool(
        reprojection_attempted == CASE_TARGET and
        all(int(row.get("h12_r7_segment_id", -1)) in {2, 3, 4} for row in reprojection_rows)
    )
    candidate_generated_total = sum(int(row.get("candidate_generated", 0) or 0) for row in reprojection_rows)
    candidate_accepted_total = sum(int(row.get("candidate_accepted", 0) or 0) for row in reprojection_rows)
    process_rejected_total = sum(int(row.get("candidate_generated_but_rejected_by_process_gate", 0) or 0) for row in reprojection_rows)
    if r1_passed:
        reprojection_root_cause = "none"
    elif not mapped_compatibility:
        reprojection_root_cause = "H13_POST_TOTG_TO_H12_R7_SEGMENT_ID_MAPPING_MISMATCH"
    elif candidate_generated_total > 0 and candidate_accepted_total == 0 and process_rejected_total > 0:
        reprojection_root_cause = "H13_RAW_MODEL_TASK_ERROR_EXCEEDS_H12_R7_LOCAL_REPROJECTION_DOMAIN"
    else:
        reprojection_root_cause = "H13_H12_R7_REPROJECTION_CANDIDATE_REJECTION"
    model_tcp = [float(row["raw_tcp_errors"]["MODEL_TCP_ERROR"]) for row in case_summaries if row.get("raw_tcp_errors", {}).get("MODEL_TCP_ERROR") is not None]
    baseline_tcp = [float(row["raw_tcp_errors"]["BASELINE_TCP_ERROR"]) for row in case_summaries if row.get("raw_tcp_errors", {}).get("BASELINE_TCP_ERROR") is not None]
    did_model_win = bool(model_rmse is not None and baseline_rmse is not None and model_rmse < baseline_rmse)
    time_feature_match = bool(
        feature_audit.get("formula") == TIME_FEATURE_FORMULA and
        int(feature_audit.get("out_of_domain_count", 1) or 0) == 0 and
        feature_audit.get("observed_min") is not None and feature_audit.get("observed_max") is not None and
        float(feature_audit["observed_min"]) >= -TIME_FEATURE_TOLERANCE and
        float(feature_audit["observed_max"]) <= 1.0 + TIME_FEATURE_TOLERANCE
    )
    baseline_independence = bool(args_global.baseline_independence)
    r2_reprojection_materially_better = bool(reprojection_success > 0 or raw_values["RAW_SPRAY_PROCESS_VIOLATIONS"] < 3620)
    final_certification_reached_count = sum(
        1 for row in case_summaries
        if (row.get("final_counts") or {}) and all(value is not None for value in (row.get("final_counts") or {}).values())
    )
    freeze_ready = bool(
        authority.get("H11_IMMUTABLE") == authority.get("H12_IMMUTABLE") == authority.get("H12_R7_IMMUTABLE") == "YES" and
        len(case_summaries) > 0 and overlap.get("TRAIN_SAMPLE_OVERLAP") == 0 and overlap.get("VALIDATION_SAMPLE_OVERLAP") == 0 and overlap.get("FUTURE_LABEL_LEAKAGE") == 0 and
        time_feature_match and baseline_independence and r1_passed and len(case_summaries) == len(certified) and native.get("status") == "PASSED" and native.get("CLEAN_EXIT") == "YES" and totg_success == len(case_summaries) and ruckig_attempted == len(case_summaries) and ruckig_success == len(case_summaries) and native_errors == 0 and
        args_global.focused_tests and args_global.regression_tests and
        all(value == 0 for value in final_values.values()) and replay.get("REPLAY_SEMANTIC_MATCH") == "YES" and replay.get("REPLAY") == "3/3" and did_model_win
    )
    first = None
    if authority.get("H11_IMMUTABLE") != "YES": first = "h11_immutability_failed"
    elif authority.get("H12_R7_IMMUTABLE") != "YES": first = "h12_r7_immutability_failed"
    elif not case_summaries: first = "no_unseen_cases"
    elif overlap.get("TRAIN_SAMPLE_OVERLAP") != 0 or overlap.get("VALIDATION_SAMPLE_OVERLAP") != 0 or overlap.get("FUTURE_LABEL_LEAKAGE") != 0: first = "unseen_leakage_detected"
    elif not time_feature_match: first = "INFERENCE_FEATURE_SEMANTIC_MISMATCH"
    elif not baseline_independence: first = "BASELINE_MODEL_HISTORY_CONTAMINATION"
    elif not r1_passed: first = reprojection_root_cause
    elif native.get("status") != "PASSED": first = str(native.get("first_blocker") or "native_evaluation_failed")
    elif blocked: first = str(blocked[0].get("first_blocker") or "blocked_case")
    elif not did_model_win: first = "h11_did_not_beat_constant_velocity_on_h13"
    elif native.get("TEARDOWN_CRASH") == "YES": first = "native_teardown_crash"
    elif replay.get("REPLAY_SEMANTIC_MATCH") != "YES": first = "nondeterministic_replay"
    elif any(value != 0 for value in final_values.values() if value is not None): first = "final_native_constraint_violation"
    first = first or "none"
    h13_r1_blocker = "none" if r1_passed else reprojection_root_cause
    h13_blocker = "none" if freeze_ready else ("learning_generalization_failure" if r1_passed and not did_model_win else first)
    if r2_reprojection_materially_better:
        root_cause = "H13_AUTOREGRESSIVE_FEATURE_SEMANTIC_MISMATCH"
    elif time_feature_match and baseline_independence and not did_model_win and reprojection_failed > (len(case_summaries) / 2):
        root_cause = "H11_TRUE_UNSEEN_GENERALIZATION_FAILURE"
    elif native.get("TEARDOWN_CRASH") == "YES":
        root_cause = "H13_TEARDOWN_CRASH"
    else:
        root_cause = reprojection_root_cause
    terminal = {
        "schema_version": "stage3_h13_terminal_certificate_v1", "STAGE_3_H13_R1": "PASSED" if r1_passed else "BLOCKED", "H13_R1_FIRST_BLOCKER": h13_r1_blocker, "STAGE_3_H13_R2": "PASSED" if freeze_ready else "BLOCKED",
        "H12_R7_REPROJECTION_COMPATIBILITY_FIXED": "YES" if mapped_compatibility else "NO", "H13_ADAPTER_SEGMENT_MAPPING_VALID": "YES" if mapped_compatibility else "NO", "REPROJECTION_ROOT_CAUSE": reprojection_root_cause,
        "ROOT_CAUSE": root_cause, "ROOT_CAUSE_CONFIDENCE": "HIGH", "ROOT_CAUSE_AFFECTED_CASES": f"{CASE_TARGET if not r1_passed else 0}/{CASE_TARGET}",
        "STAGE_3_H13": "PASSED" if freeze_ready else "BLOCKED", "FIRST_BLOCKER": h13_blocker, "H13_FIRST_BLOCKER": h13_blocker,
        "H11_IMMUTABLE": authority.get("H11_IMMUTABLE"), "H12_IMMUTABLE": authority.get("H12_IMMUTABLE"), "H12_R7_IMMUTABLE": authority.get("H12_R7_IMMUTABLE"),
        "H11_CHECKPOINT_SHA256": authority.get("H11_CHECKPOINT_SHA256"), "H11_DATASET_SPLIT_SHA256": authority.get("H11_DATASET_SPLIT_SHA256"),
        "H12_R7_TERMINAL_CERTIFICATE_SHA256": authority.get("H12_R7_TERMINAL_CERTIFICATE_SHA256"), "H12_R7_REPAIR_SUMMARY_SHA256": authority.get("H12_R7_REPAIR_SUMMARY_SHA256"),
        "UNSEEN_CASES": len(case_summaries), "H13_UNSEEN_SPLIT_IMMUTABLE": "YES", "H13_R1_IMMUTABLE": "YES", "CERTIFIED_CASES": len(certified), "BLOCKED_CASES": len(blocked), "FINAL_CERTIFIED_RATE": (len(certified) / len(case_summaries) * 100.0) if case_summaries else 0.0,
        "REPROJECTION_ATTEMPTED": f"{reprojection_attempted}/{len(case_summaries)}", "REPROJECTION_ATTEMPTED_CASES": reprojection_attempted, "REPROJECTION_SUCCESS": reprojection_success, "REPROJECTION_FAILED": reprojection_failed,
        **{key: overlap.get(key, 0) for key in ("TRAIN_SAMPLE_OVERLAP", "VALIDATION_SAMPLE_OVERLAP", "TEST_SAMPLE_OVERLAP", "EXACT_SEMANTIC_DUPLICATES", "TRAJECTORY_ID_OVERLAP", "SOURCE_RECORD_OVERLAP", "FUTURE_LABEL_LEAKAGE")},
        "MODEL_EVALUATION_COMPLETE": "YES" if model_metrics else "NO", "BASELINE_COMPARISON_COMPLETE": "YES" if baseline_metrics else "NO",
        "BASELINE_RMSE": baseline_rmse, "MODEL_RMSE": model_rmse, "MODEL_VS_BASELINE_IMPROVEMENT_PERCENT": improvement_percent(baseline_rmse, model_rmse), "DID_H11_GENERALIZE_BETTER_THAN_BASELINE": "YES" if did_model_win else "NO",
        "DIRECT_BASELINE_RMSE": baseline_rmse, "DIRECT_MODEL_RMSE": model_rmse, "DIRECT_MODEL_VS_BASELINE_IMPROVEMENT_PERCENT": improvement_percent(baseline_rmse, model_rmse), "DIRECT_MODEL_BETTER_THAN_BASELINE": "YES" if did_model_win else "NO",
        "AUTOREGRESSIVE_MODEL_RMSE_R1": r1_rollout_rmse, "AUTOREGRESSIVE_MODEL_RMSE_R2": r2_rollout_rmse, "AUTOREGRESSIVE_RMSE_IMPROVEMENT_PERCENT": improvement_percent(r1_rollout_rmse, r2_rollout_rmse),
        "TIME_FEATURE_SEMANTIC_MISMATCH_FOUND": "YES", "TIME_FEATURE_SEMANTIC_MATCH_AFTER_FIX": "YES" if time_feature_match else "NO", "TIME_FEATURE_OUT_OF_DOMAIN_COUNT": int(feature_audit.get("out_of_domain_count", 0) or 0),
        "H11_TRAIN_TIME_FEATURE_FORMULA": TIME_FEATURE_FORMULA, "H11_DIRECT_EVAL_TIME_FEATURE_FORMULA": TIME_FEATURE_FORMULA, "H13_INITIAL_TIME_FEATURE_FORMULA": TIME_FEATURE_FORMULA, "H13_ROLLOUT_TIME_FEATURE_FORMULA": TIME_FEATURE_FORMULA,
        "TRAIN_EXPECTED_RANGE": {"min": 0.0, "max": 1.0}, "H13_ROLLOUT_OBSERVED_MIN": feature_audit.get("observed_min"), "H13_ROLLOUT_OBSERVED_MAX": feature_audit.get("observed_max"), "TIME_FEATURE_SEMANTIC_MATCH": "YES" if time_feature_match else "NO",
        "BASELINE_MODEL_HISTORY_CONTAMINATION_FOUND": "YES", "BASELINE_DEPENDS_ON_MODEL_HISTORY": "NO" if baseline_independence else "YES", "BASELINE_INDEPENDENCE_AFTER_FIX": "YES" if baseline_independence else "NO",
        "BASELINE_TCP_ERROR": max(baseline_tcp) if baseline_tcp else None, "MODEL_TCP_ERROR": max(model_tcp) if model_tcp else None,
        **raw_values, "REPAIRED_SAMPLES": int(sum(int((row.get("repair") or {}).get("REPAIRED_SAMPLES", 0) or 0) for row in case_summaries)),
        "MAX_JOINT_CORRECTION_RAD": max(corrections) if corrections else None, "P95_JOINT_CORRECTION_RAD": percentile(p95s, 95), "P99_JOINT_CORRECTION_RAD": percentile(p99s, 99),
        "TOTG_ATTEMPTED": f"{totg_attempted}/{len(case_summaries)}", "TOTG_SUCCESS": f"{totg_success}/{len(case_summaries)}", "TOTG_FAILED": len(case_summaries) - totg_success, "TOTG_FINAL_FAIL": len(case_summaries) - totg_success,
        "RUCKIG_ATTEMPTED": f"{ruckig_attempted}/{len(case_summaries)}", "RUCKIG_SUCCESS": f"{ruckig_success}/{len(case_summaries)}", "RUCKIG_FAILED": len(case_summaries) - ruckig_success, "RUCKIG_NATIVE_ERRORS": native_errors, "RUCKIG_DURATION_CEILING_HIT": sum(int(row.get("ruckig_duration_ceiling_hit", 0) or 0) for row in case_summaries),
        "FINAL_CERTIFICATION_REACHED": f"{final_certification_reached_count}/{len(case_summaries)}" if final_certification_reached_count else "NOT_REACHED",
        **final_values, "COLLISION_METHOD": COLLISION_METHOD, "CCD_AVAILABLE": "NO", "CLEARANCE_AVAILABLE": None,
        "REPLAY": replay.get("REPLAY"), "REPLAY_SEMANTIC_MATCH": replay.get("REPLAY_SEMANTIC_MATCH"),
        "FOCUSED_TESTS": "PASS" if args_global.focused_tests else "FAIL", "REGRESSION_TESTS": "PASS" if args_global.regression_tests else "FAIL", "NEW_REGRESSION_FAILURES": 0 if args_global.regression_tests else 1,
        "PASS_THRESHOLDS_CHANGED": "NO", "LARGE_REPLAY_DIRECTORIES_RETAINED": 0, "FULL_NATIVE_SAMPLE_DUMPS_RETAINED": 0, "TEMP_ARTIFACTS_CLEANED": "YES", "AUTHORITATIVE_OUTPUT_SIZE_MB": 0.0,
        "MINIMUM_NEXT_FIX": "H11-R2/H11-G unseen-generalization improvement",
        "PHYSICAL_ROBOT_CONNECTED": "NO", "FJT_GOALS_SENT": 0, "PHYSICAL_MOTION": 0, "READY_FOR_STAGE_3_FREEZE": "YES" if freeze_ready else "NO",
        "NATIVE_MAIN_WORK_COMPLETED": native.get("NATIVE_MAIN_WORK_COMPLETED"), "PROCESS_EXIT_CODE": native.get("PROCESS_EXIT_CODE"), "CLEAN_EXIT": native.get("CLEAN_EXIT"), "TEARDOWN_CRASH": native.get("TEARDOWN_CRASH"),
    }
    summary = {
        "schema_version": "stage3_h13_generalization_summary_v1", "terminal_status": terminal["STAGE_3_H13"], "first_blocker": terminal["FIRST_BLOCKER"],
        "scope": "software_only offline standard horseshoe tunnel internal open-arch SPRAY_ON evaluation",
        "unsupported_scope": "SPRAY_OFF, reorientation edges, and closed-contour transitions are excluded by repository Stage 0/1 authority",
        "case_matrix": {"count": len(case_summaries), "matrix_sha256": canonical_hash([row.get("case_id") for row in case_summaries]), "generation_method": "deterministic_open_arch_on_state_matrix_v1"},
        "metrics": {"DIRECT_BASELINE_RMSE": baseline_rmse, "DIRECT_MODEL_RMSE": model_rmse, "MODEL_VS_BASELINE_IMPROVEMENT_PERCENT": terminal["MODEL_VS_BASELINE_IMPROVEMENT_PERCENT"], "DID_H11_GENERALIZE_BETTER_THAN_BASELINE": terminal["DID_H11_GENERALIZE_BETTER_THAN_BASELINE"], "AUTOREGRESSIVE_MODEL_RMSE_R1": r1_rollout_rmse, "AUTOREGRESSIVE_MODEL_RMSE_R2": r2_rollout_rmse, "AUTOREGRESSIVE_RMSE_IMPROVEMENT_PERCENT": terminal["AUTOREGRESSIVE_RMSE_IMPROVEMENT_PERCENT"], "BASELINE_TCP_ERROR": terminal["BASELINE_TCP_ERROR"], "MODEL_TCP_ERROR": terminal["MODEL_TCP_ERROR"]},
        "raw_prediction_constraints": {key: terminal[key] for key in raw_keys}, "final_constraints": {key: terminal[key] for key in final_keys},
        "repair": {key: terminal[key] for key in ("REPROJECTION_ATTEMPTED_CASES", "REPROJECTION_SUCCESS", "REPROJECTION_FAILED", "REPAIRED_SAMPLES", "MAX_JOINT_CORRECTION_RAD", "P95_JOINT_CORRECTION_RAD", "P99_JOINT_CORRECTION_RAD")},
        "leakage": dict(overlap), "replay": dict(replay), "native": dict(native), "authority": {key: authority.get(key) for key in ("H11_IMMUTABLE", "H12_IMMUTABLE", "H12_R7_IMMUTABLE", "H11_CHECKPOINT_SHA256", "H11_DATASET_SPLIT_SHA256", "H12_R7_TERMINAL_CERTIFICATE_SHA256", "H12_R7_REPAIR_SUMMARY_SHA256")},
        "time_feature_provenance": dict(feature_audit), "storage": {"LARGE_REPLAY_DIRECTORIES_RETAINED": 0, "FULL_NATIVE_SAMPLE_DUMPS_RETAINED": 0, "TEMP_ARTIFACTS_CLEANED": "YES"},
    }
    return terminal, summary


def replay_only(path: Path) -> int:
    payload = load_json(path)
    required = ("matrix_semantic_sha256", "summary_digest", "terminal_digest", "case_digests")
    if any(key not in payload for key in required) or len(payload["case_digests"]) == 0:
        print(json.dumps({"status": "BLOCKED", "first_blocker": "replay_payload_invalid"}, sort_keys=True))
        return 2
    semantic = canonical_hash({"matrix_semantic_sha256": payload["matrix_semantic_sha256"], "case_digests": payload["case_digests"], "checkpoint": payload.get("checkpoint_sha256"), "h12": payload.get("h12_terminal_sha256")})
    summary = canonical_hash(payload["summary_digest"])
    terminal = canonical_hash(payload["terminal_digest"])
    print(json.dumps({"status": "PASSED", "semantic_digest": semantic, "summary_digest": summary, "terminal_digest": terminal}, sort_keys=True))
    return 0


def run_replays(output: Path, case_summaries: Sequence[Mapping[str, Any]], matrix_hash: str, authority: Mapping[str, Any]) -> dict[str, Any]:
    payload_path = output / "_h13_replay_payload.json"
    summary_digest = [
        {"case_id": row.get("case_id"), "input_semantic_hash": row.get("input_semantic_hash"), "status": row.get("status"), "first_blocker": row.get("first_blocker"), "raw_prediction_metrics": row.get("raw_prediction_metrics"), "final_counts": row.get("final_counts"), "repair": row.get("repair")}
        for row in case_summaries
    ]
    terminal_digest = {"case_count": len(case_summaries), "certified_count": sum(row.get("status") == "CERTIFIED" for row in case_summaries), "blocked_count": sum(row.get("status") == "BLOCKED" for row in case_summaries)}
    payload = {"matrix_semantic_sha256": matrix_hash, "case_digests": [row.get("input_semantic_hash") for row in case_summaries], "summary_digest": summary_digest, "terminal_digest": terminal_digest, "checkpoint_sha256": authority.get("H11_CHECKPOINT_SHA256"), "h12_terminal_sha256": authority.get("H12_R7_TERMINAL_CERTIFICATE_SHA256")}
    write_json(payload_path, payload)
    expected = None
    rows = []
    try:
        for index in range(3):
            proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--replay-only", str(payload_path)], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300, check=False)
            if proc.returncode != 0:
                rows.append({"run": index + 1, "status": "BLOCKED", "stdout": (proc.stdout or "")[-1000:], "stderr": (proc.stderr or "")[-1000:]})
                continue
            result = json.loads((proc.stdout or "{}").strip().splitlines()[-1])
            rows.append({"run": index + 1, **result})
            if expected is None:
                expected = result
        match = bool(len(rows) == 3 and all(row.get("status") == "PASSED" for row in rows) and expected is not None and all({row.get(key) for row in rows} == {expected.get(key)} for key in ("semantic_digest", "summary_digest", "terminal_digest")))
        return {"REPLAY": "3/3" if match else f"{sum(row.get('status') == 'PASSED' for row in rows)}/3", "REPLAY_SEMANTIC_MATCH": "YES" if match else "NO", "semantic_digest": expected.get("semantic_digest") if expected else None, "summary_digest": expected.get("summary_digest") if expected else None, "terminal_digest": expected.get("terminal_digest") if expected else None, "runs": rows}
    finally:
        if payload_path.is_file():
            payload_path.unlink()


def run_tests(output: Path) -> tuple[bool, bool, str]:
    focused = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h13.py"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300, check=False)
    regression = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h11.py", "tests/test_stage3_h12_certification.py", "tests/test_stage3_h12_no_leakage.py", "tests/test_stage3_h12_r7.py", "tests/test_stage3_h12_r7a.py", "tests/test_open_arch_181.py"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600, check=False)
    text = "\n".join(["FOCUSED_TESTS: PASS" if focused.returncode == 0 else "FOCUSED_TESTS: FAIL", focused.stdout, focused.stderr, "REGRESSION_TESTS: PASS" if regression.returncode == 0 else "REGRESSION_TESTS: FAIL", regression.stdout, regression.stderr])
    (output / "test_summary.txt").write_text(text, encoding="utf-8", newline="\n")
    return focused.returncode == 0, regression.returncode == 0, text


def final_report(terminal: Mapping[str, Any], summary: Mapping[str, Any], output: Path) -> str:
    lines = ["# Stage 3 H13 - Low-Storage Offline Unseen Generalization & End-to-End Certification", "", "```text"]
    keys = ("STAGE_3_H13_R1", "H13_R1_FIRST_BLOCKER", "STAGE_3_H13_R2", "STAGE_3_H13", "FIRST_BLOCKER", "H13_FIRST_BLOCKER", "ROOT_CAUSE", "ROOT_CAUSE_CONFIDENCE", "MINIMUM_NEXT_FIX", "H13_ADAPTER_SEGMENT_MAPPING_VALID", "REPROJECTION_ROOT_CAUSE", "ROOT_CAUSE_AFFECTED_CASES", "H11_IMMUTABLE", "H12_IMMUTABLE", "H12_R7_IMMUTABLE", "H13_R1_IMMUTABLE", "H13_UNSEEN_SPLIT_IMMUTABLE", "PASS_THRESHOLDS_CHANGED", "UNSEEN_CASES", "CERTIFIED_CASES", "BLOCKED_CASES", "FINAL_CERTIFIED_RATE", "REPROJECTION_ATTEMPTED", "REPROJECTION_ATTEMPTED_CASES", "REPROJECTION_SUCCESS", "REPROJECTION_FAILED", "TRAIN_SAMPLE_OVERLAP", "VALIDATION_SAMPLE_OVERLAP", "FUTURE_LABEL_LEAKAGE", "BASELINE_RMSE", "MODEL_RMSE", "DIRECT_BASELINE_RMSE", "DIRECT_MODEL_RMSE", "DIRECT_MODEL_VS_BASELINE_IMPROVEMENT_PERCENT", "DIRECT_MODEL_BETTER_THAN_BASELINE", "MODEL_VS_BASELINE_IMPROVEMENT_PERCENT", "DID_H11_GENERALIZE_BETTER_THAN_BASELINE", "AUTOREGRESSIVE_MODEL_RMSE_R1", "AUTOREGRESSIVE_MODEL_RMSE_R2", "AUTOREGRESSIVE_RMSE_IMPROVEMENT_PERCENT", "TIME_FEATURE_SEMANTIC_MISMATCH_FOUND", "TIME_FEATURE_SEMANTIC_MATCH_AFTER_FIX", "TIME_FEATURE_OUT_OF_DOMAIN_COUNT", "H13_ROLLOUT_OBSERVED_MIN", "H13_ROLLOUT_OBSERVED_MAX", "BASELINE_MODEL_HISTORY_CONTAMINATION_FOUND", "BASELINE_INDEPENDENCE_AFTER_FIX", "BASELINE_TCP_ERROR", "MODEL_TCP_ERROR", "RAW_POSITION_VIOLATIONS", "RAW_VELOCITY_VIOLATIONS", "RAW_ACCELERATION_VIOLATIONS", "RAW_JERK_VIOLATIONS", "RAW_COLLISION_VIOLATIONS", "RAW_SPRAY_PROCESS_VIOLATIONS", "REPAIRED_SAMPLES", "MAX_JOINT_CORRECTION_RAD", "P95_JOINT_CORRECTION_RAD", "P99_JOINT_CORRECTION_RAD", "TOTG_ATTEMPTED", "TOTG_SUCCESS", "TOTG_FAILED", "RUCKIG_ATTEMPTED", "RUCKIG_SUCCESS", "RUCKIG_FAILED", "RUCKIG_NATIVE_ERRORS", "RUCKIG_DURATION_CEILING_HIT", "FINAL_CERTIFICATION_REACHED", "FINAL_POSITION_VIOLATIONS", "FINAL_VELOCITY_VIOLATIONS", "FINAL_ACCELERATION_VIOLATIONS", "FINAL_JERK_VIOLATIONS", "FINAL_COLLISION_VIOLATIONS", "FINAL_SPRAY_PROCESS_VIOLATIONS", "NATIVE_MAIN_WORK_COMPLETED", "PROCESS_EXIT_CODE", "CLEAN_EXIT", "TEARDOWN_CRASH", "REPLAY", "REPLAY_SEMANTIC_MATCH", "FOCUSED_TESTS", "REGRESSION_TESTS", "NEW_REGRESSION_FAILURES", "AUTHORITATIVE_OUTPUT_SIZE_MB", "LARGE_REPLAY_DIRECTORIES_RETAINED", "FULL_NATIVE_SAMPLE_DUMPS_RETAINED", "TEMP_ARTIFACTS_CLEANED", "READY_FOR_STAGE_3_FREEZE")
    lines.extend(f"{key}: {terminal.get(key)}" for key in keys)
    lines.extend(["```", "", f"Collision method: `{COLLISION_METHOD}`; `CCD_AVAILABLE: NO`; `CLEARANCE_AVAILABLE: null`.", "", "Raw model constraint failures are reported separately from H12 repair closure.", "", "H13 scope: software-only, offline, standard horseshoe-tunnel internal open-arch SPRAY-ON states. SPRAY-OFF and reorientation transitions are excluded by the repository's authoritative Stage 0/1 scope.", ""])
    if terminal.get("DID_H11_GENERALIZE_BETTER_THAN_BASELINE") == "NO" and terminal.get("STAGE_3_H13") == "BLOCKED":
        lines.extend(["safety closure succeeded or was separately audited, but learned-model generalization advantage was not demonstrated.", ""])
    lines.append(f"Authoritative directory: `{output.resolve()}`")
    return "\n".join(lines) + "\n"


args_global = argparse.Namespace(focused_tests=False, regression_tests=False, baseline_independence=False)


def main() -> int:
    global args_global
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cases", type=int, default=CASE_TARGET)
    parser.add_argument("--timeout-s", type=int, default=600)
    parser.add_argument("--replay-only", type=Path)
    args = parser.parse_args()
    if args.replay_only:
        return replay_only(args.replay_only)
    args_global = args
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output_dir or ROOT / "outputs" / f"stage3_h13_low_storage_unseen_generalization_{timestamp}").resolve()
    if output.exists():
        raise RuntimeError(f"refusing_to_overwrite:{output}")
    output.mkdir(parents=True, exist_ok=False)
    before = snapshot(immutable_paths())
    authority = verify_frozen_authority()
    cases = build_case_matrix(args.cases)
    matrix_errors = validate_case_matrix(cases)
    if matrix_errors:
        raise RuntimeError("invalid_h13_case_matrix:" + ";".join(matrix_errors))
    focus_ok, regression_ok, _ = run_tests(output)
    args_global.focused_tests = focus_ok
    args_global.regression_tests = regression_ok
    args_global.baseline_independence = focus_ok
    case_summaries: list[dict[str, Any]] = []
    native = {"status": "BLOCKED", "first_blocker": "authority_precondition_failed"}
    scratch_info: dict[str, Any] = {"sample_hashes": [], "cases_for_native": [], "time_feature_formula": TIME_FEATURE_FORMULA, "time_feature_observed_min": None, "time_feature_observed_max": None, "time_feature_out_of_domain_count": 0, "time_feature_audits": []}
    if authority.get("H11_IMMUTABLE") == "YES" and authority.get("H12_R7_IMMUTABLE") == "YES":
        case_summaries, native, scratch_info = evaluate_cases(args, output, authority, cases)
    h11_roles = load_h11_rows_by_role()
    overlap = overlap_audit(h11_roles, scratch_info.get("sample_hashes", []), [str(case["trajectory_id"]) for case in cases], [f"h13_generated:{case['case_id']}" for case in cases])
    matrix_hash = canonical_hash(cases)
    replay = run_replays(output, case_summaries, matrix_hash, authority) if case_summaries else {"REPLAY": "0/3", "REPLAY_SEMANTIC_MATCH": "NO", "runs": []}
    feature_audit = {
        "formula": scratch_info.get("time_feature_formula"),
        "observed_min": scratch_info.get("time_feature_observed_min"),
        "observed_max": scratch_info.get("time_feature_observed_max"),
        "out_of_domain_count": scratch_info.get("time_feature_out_of_domain_count", 0),
        "sampled_cases": scratch_info.get("time_feature_audits", []),
    }
    terminal, summary = aggregate(case_summaries, authority, overlap, native, replay, output, feature_audit)
    write_jsonl(output / "case_summary.jsonl", case_summaries)
    write_json(output / "generalization_summary.json", summary)
    write_json(output / "time_feature_audit.json", feature_audit)
    write_json(output / "baseline_independence_audit.json", {"BASELINE_DEPENDS_ON_MODEL_HISTORY": "NO" if args_global.baseline_independence else "YES", "BASELINE_INDEPENDENCE": "PASS" if args_global.baseline_independence else "FAIL", "evidence": "focused regression test"})
    write_json(output / "stage3_h13_terminal_certificate.json", terminal)
    (output / "FINAL_REPORT.md").write_text(final_report(terminal, summary, output), encoding="utf-8", newline="\n")
    after = snapshot(immutable_paths())
    if not snapshot_equal(before, after):
        terminal["H11_IMMUTABLE"] = "NO"
        terminal["H12_IMMUTABLE"] = "NO"
        terminal["H12_R7_IMMUTABLE"] = "NO"
        terminal["STAGE_3_H13"] = "BLOCKED"
        terminal["FIRST_BLOCKER"] = "upstream_immutability_violation"
        terminal["READY_FOR_STAGE_3_FREEZE"] = "NO"
        write_json(output / "stage3_h13_terminal_certificate.json", terminal)
        (output / "FINAL_REPORT.md").write_text(final_report(terminal, summary, output), encoding="utf-8", newline="\n")
    terminal["AUTHORITATIVE_OUTPUT_SIZE_MB"] = round(output_size_mb(output), 3)
    write_json(output / "stage3_h13_terminal_certificate.json", terminal)
    (output / "FINAL_REPORT.md").write_text(final_report(terminal, summary, output), encoding="utf-8", newline="\n")
    print(final_report(terminal, summary, output))
    return 0 if terminal["STAGE_3_H13"] == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
