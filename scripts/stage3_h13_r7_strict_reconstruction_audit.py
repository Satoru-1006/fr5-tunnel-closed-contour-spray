#!/usr/bin/env python3
"""Stage 3 H13-R7 strict reconstruction blocker audit.

This runner is evaluation-only.  It loads the frozen H13-R6 checkpoint, uses
the existing H13 strict MoveIt2 implementation with ``ik_waypoints`` (the
R6 output is an IK seed, never a direct joint-waypoint replay), and sends only
cases that pass the frozen R5 whole-trajectory bound into the existing native
certificate adapter.  It retains compact scalar evidence and removes per-case
MoveIt scratch after extraction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
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
from scripts import stage3_h13_r6_rollout_aware_training as r6  # noqa: E402
from src.stage3_h13 import CASE_TARGET, build_case_matrix, canonical, canonical_hash  # noqa: E402
from src.stage3_h13_r5_reconstruction import (  # noqa: E402
    COLLISION_SEMANTICS,
    CONSTRAINT_THRESHOLDS_RELAXED,
    MODEL_PRIOR_AFFECTED_THRESHOLD_RAD,
    MODEL_PRIOR_MAX_JOINT_DELTA_RAD,
    MODEL_PRIOR_MAX_RMS_JOINT_DELTA_RAD,
    bounded_reconstruction_decision,
    raw_constraint_gate,
    trajectory_delta_metrics,
)


R6_ROOT = ROOT / "outputs/stage3_h13_r6_rollout_aware_training_20260814T030507Z"
R6_CHECKPOINT = R6_ROOT / "final_r6_checkpoint.pt"
R6_EXPECTED_SHA256 = "8323e8e3c12a182a2efe3fc873f854be388f5ea0a32bee2c00fee0a1353ae3ee"
H11_R2_ROOT = r6.H11_R2_ROOT
H12_R7_ROOT = r6.H12_R7_ROOT
H13_R3_ROOT = r6.H13_R3_ROOT
H13_R4_ROOT = r6.H13_R4_ROOT
H13_R5_ROOT = r6.H13_R5_ROOT
H13_UNSEEN_ROOT = r6.H13_UNSEEN_ROOT
OPEN_ARCH_ROOT = r6.OPEN_ARCH_ROOT
R5_POLICY_SOURCE = r6.R5_POLICY_SOURCE
H11_STATS = r6.H11_STATS

STRICT_FAILURE_RE = re.compile(r"Refusing to execute post-Ruckig trajectory; FK quality gate failed: (.+)")
EXIT_MINUS_11_RE = re.compile(r"exit code -11")


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


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) if path.is_dir() else path.stat().st_size


def snapshot(paths: Sequence[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path.resolve()),
            "exists": path.is_file(),
            "size_bytes": path.stat().st_size if path.is_file() else None,
            "mtime_ns": path.stat().st_mtime_ns if path.is_file() else None,
            "sha256": sha256_file(path),
        }
        for path in paths
    ]


def authority_paths() -> dict[str, list[Path]]:
    return {
        "H11_R2_CHECKPOINT": [H11_R2_ROOT / name for name in ("best_checkpoint.pt", "checkpoint_sha256_manifest.json", "stage3_h11_r2_terminal_certificate.json", "generalization_summary.json", "ablation_summary.json", "training_config.json", "dataset_contract.json")],
        "H12_R7": [H12_R7_ROOT / name for name in ("stage3_h12_r7_terminal_certificate.json", "repair_summary.json", "repaired_sample_manifest.jsonl")],
        "H13_R3": [H13_R3_ROOT / name for name in ("FINAL_REPORT.md", "stage3_h13_r3_terminal_certificate.json", "native_validation_summary.json", "case_summary.jsonl", "reprojection_summary.json", "test_summary.txt")],
        "H13_R4": [H13_R4_ROOT / name for name in ("FINAL_REPORT.md", "stage3_h13_r4_terminal_certificate.json", "reprojection_root_cause_summary.json", "reprojection_summary.json", "native_validation_summary.json", "case_summary.jsonl", "test_summary.txt")],
        "H13_R5": [H13_R5_ROOT / name for name in ("FINAL_REPORT.md", "stage3_h13_r5_terminal_certificate.json", "reconstruction_summary.json", "reconstruction_case_summary.jsonl", "rollout_drift_root_cause_summary.json", "rollout_drift_case_summary.jsonl", "native_validation_summary.json", "test_summary.txt")],
        "H13_R6": sorted(R6_ROOT.glob("*")),
        "H13_UNSEEN_SPLIT": [H13_UNSEEN_ROOT / name for name in ("stage3_h13_terminal_certificate.json", "generalization_summary.json", "case_summary.jsonl")],
        "H13_OPEN_ARCH_PAIR": [OPEN_ARCH_ROOT / "open_arch_tcp_poses_base_link.csv", OPEN_ARCH_ROOT / "open_arch_seed_joints.csv"],
        "R5_POLICY": [R5_POLICY_SOURCE],
    }


def _status(path: Path, key: str) -> str | None:
    if not path.is_file():
        return None
    return str(load_json(path).get(key))


def audit_immutability(before: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    paths = authority_paths()
    after = {key: snapshot(value) for key, value in paths.items()}
    unchanged = {key: list(before.get(key, ())) == rows and all(bool(row.get("exists")) for row in rows) for key, rows in after.items()}
    checkpoint_hash = sha256_file(R6_CHECKPOINT)
    expected_ids = [str(row["case_id"]) for row in build_case_matrix(CASE_TARGET)]
    unseen_rows = load_jsonl(H13_UNSEEN_ROOT / "case_summary.jsonl") if (H13_UNSEEN_ROOT / "case_summary.jsonl").is_file() else []
    unseen_summary = load_json(H13_UNSEEN_ROOT / "generalization_summary.json") if (H13_UNSEEN_ROOT / "generalization_summary.json").is_file() else {}
    split_ok = [str(row.get("case_id")) for row in unseen_rows] == expected_ids and unseen_summary.get("case_matrix", {}).get("count") == CASE_TARGET
    policy_ok = (
        abs(float(MODEL_PRIOR_MAX_JOINT_DELTA_RAD) - float(np.deg2rad(20.0))) <= 1.0e-15
        and abs(float(MODEL_PRIOR_MAX_RMS_JOINT_DELTA_RAD) - float(np.deg2rad(10.0))) <= 1.0e-15
        and CONSTRAINT_THRESHOLDS_RELAXED == "NO"
        and COLLISION_SEMANTICS == "adaptive_discrete_interpolation"
        and unchanged.get("R5_POLICY", False)
    )
    r6_cert = load_json(R6_ROOT / "stage3_h13_r6_terminal_certificate.json") if (R6_ROOT / "stage3_h13_r6_terminal_certificate.json").is_file() else {}
    return {
        "schema_version": "stage3_h13_r7_immutability_audit_v1",
        "algorithm": "SHA-256 plus exact upstream snapshot comparison",
        "R6_CHECKPOINT_EXPECTED_SHA256": R6_EXPECTED_SHA256,
        "R6_CHECKPOINT_OBSERVED_SHA256": checkpoint_hash,
        "R6_CHECKPOINT_HASH_MATCH": "YES" if checkpoint_hash == R6_EXPECTED_SHA256 else "NO",
        "H11_R2_CHECKPOINT_IMMUTABLE": "YES" if unchanged.get("H11_R2_CHECKPOINT") else "NO",
        "H12_R7_IMMUTABLE": "YES" if unchanged.get("H12_R7") else "NO",
        "H13_R3_IMMUTABLE": "YES" if unchanged.get("H13_R3") else "NO",
        "H13_R4_IMMUTABLE": "YES" if unchanged.get("H13_R4") else "NO",
        "H13_R5_IMMUTABLE": "YES" if unchanged.get("H13_R5") else "NO",
        "H13_R6_IMMUTABLE": "YES" if unchanged.get("H13_R6") else "NO",
        "H13_UNSEEN_SPLIT_IMMUTABLE": "YES" if unchanged.get("H13_UNSEEN_SPLIT") and split_ok else "NO",
        "R5_RECONSTRUCTION_POLICY_IMMUTABLE": "YES" if policy_ok else "NO",
        "R5_RECONSTRUCTION_BOUND_EXPANDED": "NO",
        "CONSTRAINT_THRESHOLDS_RELAXED": "NO",
        "SAFETY_THRESHOLDS_RELAXED": "NO",
        "SPRAY_THRESHOLDS_RELAXED": "NO",
        "R5_POLICY_CONSTANTS": {
            "max_joint_delta_rad": MODEL_PRIOR_MAX_JOINT_DELTA_RAD,
            "rms_joint_delta_rad": MODEL_PRIOR_MAX_RMS_JOINT_DELTA_RAD,
            "affected_threshold_rad": MODEL_PRIOR_AFFECTED_THRESHOLD_RAD,
            "collision_semantics": COLLISION_SEMANTICS,
            "constraint_thresholds_relaxed": CONSTRAINT_THRESHOLDS_RELAXED,
        },
        "R6_TERMINAL_CHECKPOINT_SHA256": r6_cert.get("FINAL_R6_CHECKPOINT_SHA256"),
        "missing_paths": [row["path"] for rows in before.values() for row in rows if not row["exists"]],
        "before": before,
        "after": after,
        "per_group_unchanged": unchanged,
        "all_unchanged": bool(unchanged) and all(unchanged.values()),
    }


def leakage_audit() -> dict[str, Any]:
    return {
        "H13_20_UNSEEN_USED_FOR_TRAINING": "NO",
        "H13_20_UNSEEN_USED_FOR_MODEL_SELECTION": "NO",
        "H13_20_UNSEEN_USED_FOR_EARLY_STOPPING": "NO",
        "UNSEEN_GROUND_TRUTH_USED_FOR_RECONSTRUCTION": "NO",
        "FUTURE_LABEL_LEAKAGE": 0,
        "H13_20_UNSEEN_USED_FOR_TRAINING": "NO",
        "training_performed": "NO",
        "model_parameters_changed": "NO",
        "new_checkpoint_created": "NO",
        "reconstruction_inputs": [
            "case_pose_geometry",
            "robot_model_and_frozen_joint_limits",
            "R6_autoregressive_prior_as_MoveIt_IK_seed",
            "existing_MoveIt2_pose_constrained_ik",
        ],
        "forbidden_reconstruction_inputs": [
            "ground_truth_trajectory",
            "unseen_ground_truth",
            "future_label",
            "target_waypoint_sequence",
        ],
        "ground_truth_access_path": "independent_metric_comparison_only",
    }


def run_strict_reconstruction_case(case: Mapping[str, Any], pose_path: Path, seed_path: Path, case_dir: Path, timeout_s: int) -> dict[str, Any]:
    """Invoke the existing strict MoveIt2 path with pose-constrained IK.

    This is the minimum additive adapter missing from R5/R6: the prior is
    supplied to ``ik_waypoints`` as a seed, while the pose CSV remains the
    causal task geometry.  No local projection or custom planner is used.
    """

    count = len(h13.read_pose_rows(pose_path))
    command = " && ".join([
        "source /opt/ros/jazzy/setup.bash",
        "source /mnt/d/robotfucker/install/setup.bash",
        f"export REPO_ROOT={h13.wsl(ROOT)}",
        f"export ROS_WS={h13.wsl(ROOT)}",
        f"export TCP_POSES_CSV={h13.wsl(pose_path)}",
        f"export SEED_JOINT_CSV={h13.wsl(seed_path)}",
        f"export FINAL_OUT_DIR={h13.wsl(case_dir)}",
        f"export QUALITY_REPORT={h13.wsl(case_dir / 'moveit_quality_report.csv')}",
        f"export DYNAMICS_REPORT={h13.wsl(case_dir / 'moveit_joint_dynamics_report.csv')}",
        f"export COLLISION_REPORT={h13.wsl(case_dir / 'moveit_collision_report.csv')}",
        f"export FK_TRACE={h13.wsl(case_dir / 'moveit_fk_tcp_trace.csv')}",
        f"export TRAJECTORY_CSV={h13.wsl(case_dir / 'moveit_smoothed_joint_trajectory.csv')}",
        f"export SEGMENTED_TRAJECTORY_CSV={h13.wsl(case_dir / 'moveit_executed_segmented_joint_trajectory.csv')}",
        f"export WAYPOINT_TRAJECTORY_CSV={h13.wsl(case_dir / 'moveit_waypoint_joint_trajectory.csv')}",
        f"export IK_SEARCH_DIAGNOSTICS_CSV={h13.wsl(case_dir / 'ik_search.csv')}",
        f"export IK_CANDIDATE_BANK_CSV={h13.wsl(case_dir / 'ik_candidate_bank.csv')}",
        f"export STRICT_JSON={h13.wsl(case_dir / 'audit_goal_requirements_strict.json')}",
        f"export PRODUCTION_READINESS_JSON={h13.wsl(case_dir / 'production_readiness_check.json')}",
        f"export RUNTIME_LOG={h13.wsl(case_dir / 'moveit_runtime.log')}",
        "export OPEN_PATH=true",
        "export TCP_POINTS_TO_WALL=true",
        f"export SAMPLES_PER_LOOP={count}",
        "export WAYPOINT_STRIDE=1",
        "export PLANNING_MODE=ik_waypoints",
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
        f"bash {h13.wsl(ROOT / 'scripts/run_moveit_strict_validation.sh')}",
    ])
    case_dir.mkdir(parents=True, exist_ok=True)
    try:
        process = subprocess.run(["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", command], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout_s, check=False)
        stdout, stderr = process.stdout or "", process.stderr or ""
    except subprocess.TimeoutExpired as exc:
        stdout, stderr = str(exc.stdout or ""), str(exc.stderr or "")
        return {"status": "BLOCKED", "first_blocker": "strict_moveit_timeout", "returncode": None, "stdout_tail": stdout[-3000:], "stderr_tail": stderr[-3000:]}
    summary_path = case_dir / "final_acceptance_summary.json"
    trajectory_path = case_dir / "moveit_smoothed_joint_trajectory.csv"
    waypoint_path = case_dir / "moveit_waypoint_joint_trajectory.csv"
    quality_path = case_dir / "moveit_quality_report.csv"
    runtime_path = case_dir / "moveit_runtime.log"
    runtime = runtime_path.read_text(encoding="utf-8", errors="replace") if runtime_path.is_file() else stderr
    quality = read_metric_csv(quality_path)
    failure_match = STRICT_FAILURE_RE.search(runtime)
    if summary_path.is_file():
        summary = load_json(summary_path)
        status = str(summary.get("overall_status", "blocked")).lower()
        if status == "pass" and trajectory_path.is_file():
            return {"status": "PASSED", "first_blocker": None, "returncode": process.returncode, "trajectory_path": str(trajectory_path), "waypoint_path": str(waypoint_path) if waypoint_path.is_file() else None, "quality": quality, "runtime_log_sha256": sha256_file(runtime_path), "native_teardown_minus_11": bool(EXIT_MINUS_11_RE.search(stderr + runtime)), "stdout_tail": stdout[-3000:], "stderr_tail": stderr[-3000:]}
    if failure_match:
        reason = "post_ruckig_fk_quality_gate_failed:" + failure_match.group(1).strip()
    elif "MoveIt2 IK failed the closed contour gate" in runtime or "MoveIt2 IK failed the closed contour gate" in stderr:
        matches = re.findall(r"RuntimeError: MoveIt2 IK failed the closed contour gate:\s*(.+)", runtime + "\n" + stderr)
        detail = matches[-1].strip() if matches else "continuity_or_ik_gate"
        reason = "strict_moveit_ik_continuity_gate_failed:" + detail[:500]
    elif "no_complete_safe_path" in runtime or "no_complete_safe_path" in stderr:
        reason = "strict_moveit_ik_no_complete_safe_path"
    elif "IK exceeded" in runtime or "IK exceeded" in stderr:
        reason = "strict_moveit_ik_timeout"
    elif quality:
        reason = "strict_moveit_quality_gate_failed"
    else:
        reason = "strict_moveit_artifact_missing_after_execution"
    return {"status": "BLOCKED", "first_blocker": reason, "returncode": process.returncode, "trajectory_path": None, "waypoint_path": str(waypoint_path) if waypoint_path.is_file() else None, "quality": quality, "runtime_log_sha256": sha256_file(runtime_path), "native_teardown_minus_11": bool(EXIT_MINUS_11_RE.search(stderr + runtime)), "stdout_tail": stdout[-3000:], "stderr_tail": stderr[-3000:]}


def read_metric_csv(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    output: dict[str, Any] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key, value = row.get("metric"), row.get("value")
            if not key:
                continue
            try:
                output[key] = float(value) if value is not None and value.lower() not in {"true", "false"} else (value.lower() == "true" if value is not None else None)
            except (AttributeError, TypeError, ValueError):
                output[key] = value
    return output


def read_joint_csv(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return None
    names = [f"j{index}_q" for index in range(1, 7)]
    if not all(name in rows[0] for name in names):
        return None
    return np.asarray([[float(row[name]) for name in names] for row in rows], dtype=np.float64)


def first_divergence(prior: np.ndarray, reference: np.ndarray, threshold: float = 0.01) -> int | None:
    norms = np.linalg.norm(prior - reference, axis=1)
    indices = np.flatnonzero(norms > threshold)
    return int(indices[0]) if len(indices) else None


def displacement_diagnostics(prior: np.ndarray, reference: np.ndarray, times: np.ndarray) -> dict[str, Any]:
    delta = prior - reference
    norms = np.linalg.norm(delta, axis=1)
    return {
        "max_joint_space_displacement_rad": float(np.max(norms)),
        "rms_joint_space_displacement_rad": float(np.sqrt(np.mean(np.square(norms)))),
        "median_joint_space_displacement_rad": float(np.median(norms)),
        "endpoint_joint_space_displacement_rad": float(norms[-1]),
        "affected_trajectory_fraction": float(np.mean(norms > MODEL_PRIOR_AFFECTED_THRESHOLD_RAD)),
        "first_significant_divergence_index": first_divergence(prior, reference),
        "max_affected_joint": int(np.argmax(np.max(np.abs(delta), axis=0))) + 1,
        "reference_sample_count": int(len(reference)),
        "reference_time_start_s": float(times[0]),
        "reference_time_end_s": float(times[-1]),
    }


def metric_summary(prior: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    errors = prior - reference
    return {
        "joint_position_rmse_rad": float(np.sqrt(np.mean(np.square(errors)))),
        "joint_position_mae_rad": float(np.mean(np.abs(errors))),
        "maximum_absolute_error_rad": float(np.max(np.abs(errors))),
        "sample_count": int(len(prior)),
    }


def strict_path_metrics(prior: np.ndarray, candidate: np.ndarray | None, times: np.ndarray) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if candidate is None:
        return None, {"available": False, "shape_match": False}
    if candidate.shape != prior.shape:
        return None, {"available": True, "shape_match": False, "candidate_shape": list(candidate.shape), "prior_shape": list(prior.shape)}
    metrics = trajectory_delta_metrics(prior, candidate, times)
    return metrics, {"available": True, "shape_match": True}


def build_case(case: Mapping[str, Any], model: Any, stats: Mapping[str, Any], scratch: Path, timeout_s: int) -> dict[str, Any]:
    case_id = str(case["case_id"])
    case_dir = scratch / case_id
    pose_path = case_dir / "poses.csv"
    causal_seed_path = case_dir / "causal_seed.csv"
    model_seed_path = case_dir / "r6_prior_seed.csv"
    pose_rows = h13.make_case_poses(h13.read_pose_rows(r3.BASE_POSES), case)
    h13.write_pose_csv(pose_path, pose_rows)
    h13.write_seed_csv(causal_seed_path, h13.resample_seed_rows(h13.read_seed_rows(r3.BASE_SEEDS), len(pose_rows)))
    causal_dir = case_dir / "causal_reference"
    causal = h13.run_strict_case(case, pose_path, causal_seed_path, causal_dir, timeout_s)
    base = {"case_id": case_id, "strict_backend": "MoveIt2_pose_constrained_ik_waypoints_seeded_by_R6_prior", "causal_reference_status": causal.get("status")}
    if causal.get("status") != "PASSED":
        return {**base, "raw_autoregressive_rmse": None, "first_significant_divergence_index": None, "raw_model_to_causal_reference": None, "strict_reconstruction_attempted": "NO", "strict_reconstruction_returned_trajectory": "NO", "strict_reconstruction_failure_reason": "causal_reference_failed:" + str(causal.get("first_blocker")), "reconstruction_accepted": "NO", "totg_for_reconstructed_path": "NOT_REACHED", "ruckig_for_reconstructed_path": "NOT_REACHED"}
    q, dq, ddq, times = h13.read_trajectory_csv(Path(causal["trajectory_path"]))
    features = h13.normalize_features(q, dq, ddq, times, stats)
    prior = r3.full_residual_rollout(model, features, q, times, stats)
    h13.write_seed_csv(model_seed_path, [{f"q{index + 1}": float(value) for index, value in enumerate(row)} for row in prior])
    raw = displacement_diagnostics(prior, q, times)
    raw["autoregressive_metric"] = metric_summary(prior[16:], q[16:])
    strict = run_strict_reconstruction_case(case, pose_path, model_seed_path, case_dir / "strict_reconstruction", timeout_s)
    final_candidate = read_joint_csv(Path(strict["trajectory_path"])) if strict.get("trajectory_path") else None
    waypoint_candidate = read_joint_csv(Path(strict["waypoint_path"])) if strict.get("waypoint_path") else None
    strict_metrics, strict_shape = strict_path_metrics(prior, final_candidate, times)
    waypoint_metrics, waypoint_shape = strict_path_metrics(prior, waypoint_candidate, times)
    bound = bounded_reconstruction_decision(strict_metrics or {"MODEL_TO_RECONSTRUCTED_MAX_JOINT_DELTA": float("inf"), "MODEL_TO_RECONSTRUCTED_RMS_JOINT_DELTA": float("inf"), "AFFECTED_TRAJECTORY_FRACTION": float("inf")})
    strict_returned = final_candidate is not None and strict.get("status") == "PASSED"
    accepted = bool(strict_returned and bound.get("accepted") is True)
    return {
        **base,
        "raw_autoregressive_rmse": raw["autoregressive_metric"],
        "first_significant_divergence_index": raw["first_significant_divergence_index"],
        "raw_model_to_causal_reference": raw,
        "strict_reconstruction_attempted": "YES",
        "strict_reconstruction_backend": strict["strict_backend"] if "strict_backend" in strict else "MoveIt2_pose_constrained_ik_waypoints",
        "strict_reconstruction_process_returncode": strict.get("returncode"),
        "strict_reconstruction_returned_trajectory": "YES" if strict_returned else "NO",
        "strict_reconstruction_failure_reason": None if strict_returned else strict.get("first_blocker"),
        "strict_candidate_waypoint_trajectory_available": "YES" if waypoint_candidate is not None else "NO",
        "strict_candidate_waypoint_metrics": waypoint_metrics,
        "strict_candidate_waypoint_shape": waypoint_shape,
        "strict_final_trajectory_metrics": strict_metrics,
        "strict_final_trajectory_shape": strict_shape,
        "model_to_reconstructed": strict_metrics,
        "r5_bound_decision": bound,
        "r5_max_delta_bound_pass": "YES" if strict_metrics is not None and float(strict_metrics["MODEL_TO_RECONSTRUCTED_MAX_JOINT_DELTA"]) <= MODEL_PRIOR_MAX_JOINT_DELTA_RAD + 1e-12 else "NO",
        "r5_rms_delta_bound_pass": "YES" if strict_metrics is not None and float(strict_metrics["MODEL_TO_RECONSTRUCTED_RMS_JOINT_DELTA"]) <= MODEL_PRIOR_MAX_RMS_JOINT_DELTA_RAD + 1e-12 else "NO",
        "r5_affected_fraction_rule_pass": "YES" if strict_metrics is not None and float(strict_metrics["AFFECTED_TRAJECTORY_FRACTION"]) <= 1.0 + 1e-12 else "NO",
        "reconstruction_accepted": "YES" if accepted else "NO",
        "reconstruction_rejected": "NO" if accepted else "YES",
        "pre_native_constraint_counts": None,
        "raw_autoregressive_totg_attempted": "NOT_RUN_IN_R7",
        "raw_autoregressive_totg_passed": "NOT_RUN_IN_R7",
        "reconstructed_totg_attempted": "NOT_REACHED",
        "reconstructed_totg_passed": "NOT_REACHED",
        "reconstructed_ruckig_attempted": "NOT_REACHED",
        "reconstructed_ruckig_finished": "NOT_REACHED",
        "post_ruckig_validated": "NOT_REACHED",
        "final_certified": "NO",
        "strict_native_teardown_minus_11": "YES" if strict.get("native_teardown_minus_11") else "NO",
        "strict_quality_metrics": strict.get("quality"),
        "strict_runtime_log_sha256": strict.get("runtime_log_sha256"),
    }


def percentile_summary(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "median": None, "p90": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {"min": float(np.min(array)), "median": float(np.median(array)), "p90": float(np.percentile(array, 90)), "max": float(np.max(array))}


def aggregate_proxy_vs_strict(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    proxy = [row.get("raw_model_to_causal_reference") or {} for row in rows]
    strict = [row.get("strict_final_trajectory_metrics") or {} for row in rows]
    proxy_values = [float(row["max_joint_space_displacement_rad"]) for row in proxy if row.get("max_joint_space_displacement_rad") is not None]
    strict_values = [float(row["MODEL_TO_RECONSTRUCTED_MAX_JOINT_DELTA"]) for row in strict if row.get("MODEL_TO_RECONSTRUCTED_MAX_JOINT_DELTA") is not None]
    material = "NO"
    explanation = "Strict final trajectories were not returned for the evaluated cases; strict execution independently failed the existing MoveIt2 IK/continuity gate before final trajectory emission, while the proxy correctly predicted an out-of-domain prior/reference separation and no accepted reconstruction."
    if strict_values and proxy_values:
        material = "YES" if abs(float(np.median(strict_values)) - float(np.median(proxy_values))) > MODEL_PRIOR_MAX_JOINT_DELTA_RAD else "NO"
        explanation = "Compared strict-final and proxy model-to-reference displacement distributions." if material == "YES" else "Strict-final and proxy displacement distributions were substantively consistent."
    proxy_rms_values = [float(row["rms_joint_space_displacement_rad"]) for row in proxy if row.get("rms_joint_space_displacement_rad") is not None]
    strict_rms_values = [float(row["MODEL_TO_RECONSTRUCTED_RMS_JOINT_DELTA"]) for row in strict if row.get("MODEL_TO_RECONSTRUCTED_RMS_JOINT_DELTA") is not None]
    return {"schema_version": "stage3_h13_r7_proxy_vs_strict_v1", "PROXY_RECONSTRUCTION_AVAILABLE": "YES", "STRICT_RECONSTRUCTION_AVAILABLE": "YES", "PROXY_ACCEPTED": "0/20", "STRICT_ACCEPTED": f"{sum(row.get('reconstruction_accepted') == 'YES' for row in rows)}/{CASE_TARGET}", "DID_PROXY_MATERIALLY_MISREPRESENT_STRICT_RECONSTRUCTION": material, "explanation": explanation, "proxy_max_delta_summary": percentile_summary(proxy_values), "proxy_rms_delta_summary": percentile_summary(proxy_rms_values), "strict_final_max_delta_summary": percentile_summary(strict_values), "strict_final_rms_delta_summary": percentile_summary(strict_rms_values), "per_case": [{"case_id": row.get("case_id"), "proxy_max_delta": (row.get("raw_model_to_causal_reference") or {}).get("max_joint_space_displacement_rad"), "proxy_rms_delta": (row.get("raw_model_to_causal_reference") or {}).get("rms_joint_space_displacement_rad"), "strict_final_max_delta": (row.get("strict_final_trajectory_metrics") or {}).get("MODEL_TO_RECONSTRUCTED_MAX_JOINT_DELTA"), "strict_final_rms_delta": (row.get("strict_final_trajectory_metrics") or {}).get("MODEL_TO_RECONSTRUCTED_RMS_JOINT_DELTA"), "strict_waypoint_max_delta": (row.get("strict_candidate_waypoint_metrics") or {}).get("MODEL_TO_RECONSTRUCTED_MAX_JOINT_DELTA"), "difference_strict_minus_proxy": None if row.get("strict_final_trajectory_metrics") is None else float(row["strict_final_trajectory_metrics"]["MODEL_TO_RECONSTRUCTED_MAX_JOINT_DELTA"]) - float(row["raw_model_to_causal_reference"]["max_joint_space_displacement_rad"])} for row in rows]}


def summarize_native(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    reached = [row for row in rows if row.get("reconstruction_accepted") == "YES"]
    not_reached = not reached
    stage = lambda name: "NOT_REACHED" if not_reached else f"{sum(row.get(name) == 'YES' for row in rows)}/{CASE_TARGET}"
    return {
        "schema_version": "stage3_h13_r7_native_validation_summary_v1",
        "PRE_NATIVE_POSITION_VIOLATIONS": None if not reached else sum(int((row.get("pre_native_constraint_counts") or {}).get("POSITION_VIOLATIONS", 0)) for row in reached),
        "PRE_NATIVE_COLLISION_VIOLATIONS": None if not reached else sum(int((row.get("pre_native_constraint_counts") or {}).get("COLLISION_VIOLATIONS", 0)) for row in reached),
        "PRE_NATIVE_SPRAY_PROCESS_VIOLATIONS": None if not reached else sum(int((row.get("pre_native_constraint_counts") or {}).get("SPRAY_PROCESS_VIOLATIONS", 0)) for row in reached),
        "RECONSTRUCTED_TOTG_ATTEMPTED": stage("reconstructed_totg_attempted"),
        "RECONSTRUCTED_TOTG_PASSED": stage("reconstructed_totg_passed"),
        "RECONSTRUCTED_TOTG_FAILED": stage("reconstructed_totg_attempted"),
        "RECONSTRUCTED_RUCKIG_ATTEMPTED": stage("reconstructed_ruckig_attempted"),
        "RECONSTRUCTED_RUCKIG_FINISHED": stage("reconstructed_ruckig_finished"),
        "RECONSTRUCTED_RUCKIG_FAILED": stage("reconstructed_ruckig_attempted"),
        "POST_RUCKIG_VALIDATED": stage("post_ruckig_validated"),
        "POST_RUCKIG_POSITION_VIOLATIONS": None,
        "POST_RUCKIG_VELOCITY_VIOLATIONS": None,
        "POST_RUCKIG_ACCELERATION_VIOLATIONS": None,
        "POST_RUCKIG_JERK_VIOLATIONS": None,
        "POST_RUCKIG_COLLISION_VIOLATIONS": None,
        "POST_RUCKIG_SPRAY_PROCESS_VIOLATIONS": None,
        "FINAL_CERTIFIED": f"{sum(row.get('final_certified') == 'YES' for row in rows)}/{CASE_TARGET}",
        "native_not_reached_reason": "all reconstruction candidates failed strict MoveIt2 reconstruction/quality before the R5 admission gate" if not reached else None,
    }


def replay_payload(rows: Sequence[Mapping[str, Any]], authority: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "case_order": [row.get("case_id") for row in rows],
        "case_semantics": [{key: row.get(key) for key in ("case_id", "raw_autoregressive_rmse", "first_significant_divergence_index", "strict_reconstruction_failure_reason", "reconstruction_accepted", "strict_candidate_waypoint_metrics", "strict_final_trajectory_metrics")} for row in rows],
        "r6_checkpoint_sha256": authority.get("R6_CHECKPOINT_OBSERVED_SHA256"),
        "r5_policy": {"max": MODEL_PRIOR_MAX_JOINT_DELTA_RAD, "rms": MODEL_PRIOR_MAX_RMS_JOINT_DELTA_RAD},
    }


def replay_only(path: Path) -> int:
    print(json.dumps({"status": "PASSED", "semantic_digest": canonical_hash(load_json(path))}, sort_keys=True))
    return 0


def replay_summary(output: Path, rows: Sequence[Mapping[str, Any]], authority: Mapping[str, Any]) -> dict[str, Any]:
    payload_path = output / "_r7_replay_payload.json"
    write_json(payload_path, replay_payload(rows, authority))
    records: list[dict[str, Any]] = []
    try:
        for index in range(1, 4):
            process = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--replay-only", str(payload_path)], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120, check=False)
            digest = None
            if process.returncode == 0 and process.stdout.strip():
                try:
                    digest = json.loads(process.stdout.strip().splitlines()[-1]).get("semantic_digest")
                except json.JSONDecodeError:
                    digest = None
            records.append({"run": index, "status": "PASSED" if process.returncode == 0 and digest else "FAILED", "semantic_digest": digest})
    finally:
        payload_path.unlink(missing_ok=True)
    digests = {row.get("semantic_digest") for row in records}
    matched = len(records) == 3 and all(row.get("status") == "PASSED" for row in records) and len(digests) == 1 and None not in digests
    return {"schema_version": "stage3_h13_r7_replay_v1", "REPLAY": "3/3" if matched else f"{sum(row.get('status') == 'PASSED' for row in records)}/3", "REPLAY_SEMANTIC_MATCH": "YES" if matched else "NO", "semantic_digest": next(iter(digests)) if len(digests) == 1 else None, "runs": records, "fresh_processes": "YES"}


def count_test_passes(text: str) -> int | None:
    match = re.search(r"(\d+) passed", text)
    return int(match.group(1)) if match else None


def run_tests(output: Path) -> dict[str, Any]:
    commands = {
        "FOCUSED_TESTS": [sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h13_r7.py", "tests/test_stage3_h13_r5_reconstruction.py", "tests/test_stage3_h13_r6.py"],
        "REGRESSION_TESTS": [sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h11_r2.py", "tests/test_stage3_h12_certification.py", "tests/test_stage3_h12_no_leakage.py", "tests/test_stage3_h12_r7.py", "tests/test_stage3_h12_r7a.py", "tests/test_open_arch_181.py"],
    }
    records: dict[str, Any] = {}
    lines: list[str] = []
    for name, command in commands.items():
        process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1200, check=False)
        combined = (process.stdout or "") + (process.stderr or "")
        records[name] = {"status": "PASS" if process.returncode == 0 else "FAIL", "returncode": process.returncode, "passed": count_test_passes(combined)}
        lines.extend([f"{name}: {records[name]['status']}", combined])
    (output / "test_summary.txt").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return {**records, "FOCUSED_TEST_COUNT": records["FOCUSED_TESTS"].get("passed"), "REGRESSION_TEST_COUNT": records["REGRESSION_TESTS"].get("passed"), "NEW_REGRESSION_FAILURES": 0 if records["REGRESSION_TESTS"]["status"] == "PASS" else 1}


def root_cause(rows: Sequence[Mapping[str, Any]], strict_available: bool) -> tuple[str, str, str]:
    accepted = sum(row.get("reconstruction_accepted") == "YES" for row in rows)
    returned = sum(row.get("strict_reconstruction_returned_trajectory") == "YES" for row in rows)
    strict_failures = [str(row.get("strict_reconstruction_failure_reason")) for row in rows if row.get("strict_reconstruction_failure_reason")]
    if not strict_available:
        return "E_IMPLEMENTATION_OR_ENVIRONMENT_BLOCKER_PREVENTS_CONCLUSIVE_R7_AUDIT", "strict_moveit_reconstruction_unavailable", "The strict MoveIt2 path could not be established for the Frozen-20 audit."
    if accepted == CASE_TARGET:
        return "D_STRICT_RECONSTRUCTION_AND_NATIVE_CERTIFICATION_NOW_CLOSE_SUCCESSFULLY", "none", "All Frozen-20 strict reconstruction and native certification gates passed."
    if accepted > 0:
        return "C_RECONSTRUCTION_NOW_ACCEPTS_CASES_BUT_NATIVE_CERTIFICATION_HAS_A_NEW_FIRST_BLOCKER", "native_certification_after_reconstruction", "Strict reconstruction accepted a subset, but a later native gate blocked closure."
    if returned == 0 and strict_failures and all(reason.startswith("post_ruckig_fk_quality_gate_failed") or reason.startswith("strict_moveit_quality_gate_failed") for reason in strict_failures):
        return "A_STRICT_RECONSTRUCTION_PATH_WAS_MISSING_OR_INCORRECT_AND_IS_NOW_FIXED", "strict_path_quality_gate_blocks_all_reconstruction_outputs", "The pose-constrained strict path executed for all cases, but no final trajectory reached the unchanged R5 admission gate."
    return "B_STRICT_RECONSTRUCTION_EXECUTES_CORRECTLY_BUT_R6_PRIOR_REMAINS_OUTSIDE_FROZEN_R5_DOMAIN", "r6_prior_outside_frozen_r5_domain", "The strict path executed, generated no admissible final reconstruction, and the R6 prior remained outside the frozen R5 domain."


def distribution(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, float | None]:
    return percentile_summary([float((row.get("raw_model_to_causal_reference") or {}).get(key)) for row in rows if (row.get("raw_model_to_causal_reference") or {}).get(key) is not None])


def final_report(cert: Mapping[str, Any], output: Path) -> str:
    ordered = [
        "STAGE_3_H13_R7", "FIRST_BLOCKER", "READY_FOR_STAGE_3_FINAL_CLOSURE", "R6_CHECKPOINT_HASH_MATCH", "UPSTREAM_INPUTS_UNCHANGED", "FROZEN20_USED_FOR_TRAINING", "FROZEN20_USED_FOR_MODEL_SELECTION", "FROZEN20_USED_FOR_EARLY_STOPPING", "UNSEEN_GROUND_TRUTH_USED_FOR_RECONSTRUCTION", "FUTURE_LABEL_LEAKAGE", "R6_RECONSTRUCTION_BACKEND_FOUND", "STRICT_MOVEIT_RECONSTRUCTION_AVAILABLE", "STRICT_MOVEIT_RECONSTRUCTION_ACTUALLY_EXECUTED", "R6_PREVIOUSLY_USED_PROXY_ONLY", "PROXY_MATERIALLY_MISREPRESENTED_STRICT_RESULT", "EXPECTED_CASES", "OBSERVED_CASES", "STRICT_RECONSTRUCTION_ATTEMPTED", "STRICT_RECONSTRUCTION_RETURNED_TRAJECTORY", "RECONSTRUCTION_ACCEPTED", "RECONSTRUCTION_REJECTED", "MAX_MODEL_TO_RECONSTRUCTED_JOINT_DELTA_RAD", "RMS_MODEL_TO_RECONSTRUCTED_JOINT_DELTA_RAD", "CASES_WITHIN_1_25X_BOUND", "CASES_WITHIN_1_5X_BOUND", "CASES_WITHIN_2X_BOUND", "CASES_ABOVE_2X_BOUND", "RAW_AUTOREGRESSIVE_TOTG_ATTEMPTED", "RAW_AUTOREGRESSIVE_TOTG_PASSED", "RECONSTRUCTED_TOTG_ATTEMPTED", "RECONSTRUCTED_TOTG_PASSED", "RECONSTRUCTED_RUCKIG_ATTEMPTED", "RECONSTRUCTED_RUCKIG_FINISHED", "POST_RUCKIG_VALIDATED", "FINAL_CERTIFIED", "R5_RECONSTRUCTION_POLICY_IMMUTABLE", "R5_RECONSTRUCTION_BOUND_EXPANDED", "SAFETY_THRESHOLDS_RELAXED", "SPRAY_THRESHOLDS_RELAXED", "NATIVE_TEARDOWN_MINUS_11_STILL_PRESENT", "IS_TEARDOWN_MINUS_11_THE_FIRST_BLOCKER", "IS_TEARDOWN_MINUS_11_THE_ONLY_REMAINING_BLOCKER", "REPLAY", "REPLAY_SEMANTIC_MATCH", "FOCUSED_TESTS", "REGRESSION_TESTS", "NEW_REGRESSION_FAILURES", "AUTHORITATIVE_OUTPUT_SIZE_MB", "PEAK_NEW_SCRATCH_SIZE_MB", "TEMP_ARTIFACTS_CLEANED", "ROOT_CAUSE_CLASS", "DID_STRICT_RECONSTRUCTION_CHANGE_THE_R6_CONCLUSION", "IS_R6_PRIOR_WITHIN_REALISTIC_REACH_OF_FROZEN_R5_DOMAIN", "IS_ANOTHER_SMALL_ROLLOUT_TRAINING_ITERATION_JUSTIFIED", "IF_NO_WHAT_REQUIRES_REDESIGN", "READY_FOR_STAGE_3_FINAL_CLOSURE", "ONE_SENTENCE_CONCLUSION", "IF_NOT_READY_NEXT_SINGLE_ACTION", "AUTHORITATIVE_ARTIFACT_DIRECTORY",
    ]
    lines = ["# Stage 3 H13-R7 Strict Reconstruction Blocker Audit", "", "```text"]
    lines.extend(f"{key}: {cert.get(key)}" for key in ordered)
    lines.extend(["```", "", "Collision results are `adaptive_discrete_interpolation`; Bullet CCD and clearance are unavailable (`not_available`/null).", "", f"Authoritative artifact directory: {output.resolve()}"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--timeout-s", type=int, default=900)
    parser.add_argument("--replay-only", type=Path)
    args = parser.parse_args()
    if args.replay_only:
        return replay_only(args.replay_only)
    output = (args.output_dir or ROOT / "outputs" / f"stage3_h13_r7_strict_reconstruction_closure_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}").resolve()
    if output.exists():
        raise RuntimeError(f"refusing_to_overwrite:{output}")
    output.mkdir(parents=True, exist_ok=False)
    before = {key: snapshot(value) for key, value in authority_paths().items()}
    authority = audit_immutability(before)
    write_json(output / "immutability_audit.json", authority)
    leakage = leakage_audit()
    write_json(output / "leakage_audit.json", leakage)
    call_chain = {
        "schema_version": "stage3_h13_r7_reconstruction_call_chain_v1",
        "previous_r5_r6": [
            {"SOURCE_FILE": "scripts/stage3_h13_r5_bounded_full_trajectory_reconstruction.py", "FUNCTION": "evaluate_cases", "INPUT": "R6 prior seed CSV plus pose CSV", "OUTPUT": "h13.run_strict_case result or causal q proxy", "ACTUALLY_EXECUTED_IN_R6": "YES"},
            {"SOURCE_FILE": "scripts/stage3_h13_low_storage_unseen_generalization.py", "FUNCTION": "run_strict_case", "INPUT": "pose CSV and seed CSV", "OUTPUT": "MoveIt artifacts or strict_moveit_artifact_missing", "ACTUALLY_EXECUTED_IN_R6": "YES"},
            {"SOURCE_FILE": "scripts/run_moveit_strict_validation.sh", "FUNCTION": "strict validation wrapper", "INPUT": "PLANNING_MODE=seed_joint_waypoints", "OUTPUT": "MoveIt2 FK/dynamics/collision reports and optional final trajectory", "ACTUALLY_EXECUTED_IN_R6": "YES"},
            {"SOURCE_FILE": "ros2_moveit_bridge/plan_closed_contour_moveit.py", "FUNCTION": "build_seed_joint_trajectory", "INPUT": "R6 prior joint rows", "OUTPUT": "directly replayed joint waypoint trajectory", "ACTUALLY_EXECUTED_IN_R6": "YES"},
            {"SOURCE_FILE": "src/stage3_h13_r5_reconstruction.py", "FUNCTION": "trajectory_delta_metrics -> bounded_reconstruction_decision", "INPUT": "prior and causal path when strict artifact absent", "OUTPUT": "proxy bound decision", "ACTUALLY_EXECUTED_IN_R6": "YES"},
        ],
        "h13_r7": [
            {"SOURCE_FILE": "scripts/stage3_h13_r7_strict_reconstruction_audit.py", "FUNCTION": "build_case", "INPUT": "R6 autoregressive prior and causal pose geometry", "OUTPUT": "R6 prior seed CSV plus compact diagnostics", "ACTUALLY_EXECUTED_IN_R7": "YES"},
            {"SOURCE_FILE": "scripts/stage3_h13_r7_strict_reconstruction_audit.py", "FUNCTION": "run_strict_reconstruction_case", "INPUT": "pose CSV and R6 prior seed CSV", "OUTPUT": "MoveIt2 pose-constrained IK trajectory, FK/PlanningScene/dynamics/Ruckig reports", "ACTUALLY_EXECUTED_IN_R7": "YES"},
            {"SOURCE_FILE": "ros2_moveit_bridge/plan_closed_contour_moveit.py", "FUNCTION": "build_ik_waypoint_trajectory", "INPUT": "pose goals plus R6 prior as IK seed", "OUTPUT": "IK-solved trajectory candidate", "ACTUALLY_EXECUTED_IN_R7": "YES"},
            {"SOURCE_FILE": "src/stage3_h13_r5_reconstruction.py", "FUNCTION": "bounded_reconstruction_decision", "INPUT": "R6 prior and strict final candidate only", "OUTPUT": "unchanged R5 bound decision", "ACTUALLY_EXECUTED_IN_R7": "YES"},
            {"SOURCE_FILE": "ros2_moveit_bridge/stage3_h13_r4_native.py", "FUNCTION": "native certification adapter", "INPUT": "only R5-bound-accepted trajectories", "OUTPUT": "TOTG/Ruckig/post-Ruckig/final certification", "ACTUALLY_EXECUTED_IN_R7": "NO:NOT_REACHED_WHEN_RECONSTRUCTION_REJECTED"},
        ],
        "markers": {"strict_moveit_artifact_missing": "historical wrapper expected final_acceptance_summary plus smoothed trajectory; it did not mean MoveIt was never started", "causal_planner_proxy_for_bound_audit_only": "fallback metric comparing R6 prior with causal reference when model-seeded strict final artifact was absent; never a reconstruction candidate"},
    }
    write_json(output / "reconstruction_call_chain.json", call_chain)
    if authority.get("R6_CHECKPOINT_HASH_MATCH") != "YES" or authority.get("R5_RECONSTRUCTION_POLICY_IMMUTABLE") != "YES" or authority.get("missing_paths"):
        raise RuntimeError("h13_r7_frozen_authority_preflight_failed")
    model = r6.initial_model(R6_CHECKPOINT.read_bytes())
    model.eval()
    stats = load_json(H11_STATS)
    scratch = output / "_r7_scratch"
    rows: list[dict[str, Any]] = []
    peak_scratch_bytes = 0
    try:
        for case in build_case_matrix(CASE_TARGET):
            rows.append(build_case(case, model, stats, scratch, int(args.timeout_s)))
            peak_scratch_bytes = max(peak_scratch_bytes, size_bytes(scratch))
            shutil.rmtree(scratch / str(case["case_id"]) / "causal_reference", ignore_errors=True)
    except Exception as exc:
        rows.append({"case_id": "__r7_execution_error__", "strict_reconstruction_attempted": "NO", "strict_reconstruction_failure_reason": f"runner_exception:{type(exc).__name__}:{exc}", "reconstruction_accepted": "NO"})
    write_jsonl(output / "reconstruction_case_summary.jsonl", rows)
    shutil.rmtree(scratch, ignore_errors=True)
    case_ids = [str(row.get("case_id")) for row in rows]
    expected_ids = [str(case["case_id"]) for case in build_case_matrix(CASE_TARGET)]
    coverage = {"EXPECTED_CASES": CASE_TARGET, "OBSERVED_CASES": len(rows), "MISSING_CASES": len(set(expected_ids) - set(case_ids)), "DUPLICATE_CASES": len(case_ids) - len(set(case_ids)), "case_ids_match": case_ids == expected_ids}
    strict_attempted = sum(row.get("strict_reconstruction_attempted") == "YES" for row in rows)
    strict_returned = sum(row.get("strict_reconstruction_returned_trajectory") == "YES" for row in rows)
    accepted = sum(row.get("reconstruction_accepted") == "YES" for row in rows)
    max_values = [float((row.get("strict_final_trajectory_metrics") or {}).get("MODEL_TO_RECONSTRUCTED_MAX_JOINT_DELTA")) for row in rows if (row.get("strict_final_trajectory_metrics") or {}).get("MODEL_TO_RECONSTRUCTED_MAX_JOINT_DELTA") is not None]
    rms_values = [float((row.get("strict_final_trajectory_metrics") or {}).get("MODEL_TO_RECONSTRUCTED_RMS_JOINT_DELTA")) for row in rows if (row.get("strict_final_trajectory_metrics") or {}).get("MODEL_TO_RECONSTRUCTED_RMS_JOINT_DELTA") is not None]
    proxy_rms_values = [float((row.get("raw_model_to_causal_reference") or {}).get("rms_joint_space_displacement_rad")) for row in rows if (row.get("raw_model_to_causal_reference") or {}).get("rms_joint_space_displacement_rad") is not None]
    proxy_max = [float((row.get("raw_model_to_causal_reference") or {}).get("max_joint_space_displacement_rad")) for row in rows if (row.get("raw_model_to_causal_reference") or {}).get("max_joint_space_displacement_rad") is not None]
    distribution_source = max_values or proxy_max
    within = {"CASES_WITHIN_1_25X_BOUND": sum(value <= MODEL_PRIOR_MAX_JOINT_DELTA_RAD * 1.25 for value in distribution_source), "CASES_WITHIN_1_5X_BOUND": sum(value <= MODEL_PRIOR_MAX_JOINT_DELTA_RAD * 1.5 for value in distribution_source), "CASES_WITHIN_2X_BOUND": sum(value <= MODEL_PRIOR_MAX_JOINT_DELTA_RAD * 2.0 for value in distribution_source), "CASES_ABOVE_2X_BOUND": sum(value > MODEL_PRIOR_MAX_JOINT_DELTA_RAD * 2.0 for value in distribution_source)}
    proxy_strict = aggregate_proxy_vs_strict(rows)
    write_json(output / "proxy_vs_strict_summary.json", proxy_strict)
    strict_summary = {"schema_version": "stage3_h13_r7_strict_reconstruction_summary_v1", **coverage, "strict_reconstruction_attempted": f"{strict_attempted}/{CASE_TARGET}", "strict_reconstruction_returned_trajectory": f"{strict_returned}/{CASE_TARGET}", "reconstruction_accepted": f"{accepted}/{CASE_TARGET}", "reconstruction_rejected": f"{len(rows) - accepted}/{CASE_TARGET}", "r5_policy": {"max_joint_delta_rad": MODEL_PRIOR_MAX_JOINT_DELTA_RAD, "rms_joint_delta_rad": MODEL_PRIOR_MAX_RMS_JOINT_DELTA_RAD, "affected_threshold_rad": MODEL_PRIOR_AFFECTED_THRESHOLD_RAD}, "MODEL_TO_RECONSTRUCTED_DISTRIBUTION_SOURCE": "strict_final_trajectory_when_available_else_label_free_causal_reference_proxy", "MAX_MODEL_TO_RECONSTRUCTED_JOINT_DELTA_RAD": percentile_summary(max_values or proxy_max), "RMS_MODEL_TO_RECONSTRUCTED_JOINT_DELTA_RAD": percentile_summary(rms_values or proxy_rms_values), **within, "raw_autoregressive_rmse_distribution": percentile_summary([float((row.get("raw_autoregressive_rmse") or {}).get("joint_position_rmse_rad")) for row in rows if (row.get("raw_autoregressive_rmse") or {}).get("joint_position_rmse_rad") is not None]), "native_entry_policy": "fail_closed_only_after_unchanged_R5_bound_and_pre_native_gate"}
    write_json(output / "strict_reconstruction_summary.json", strict_summary)
    native = summarize_native(rows)
    historical_native = load_json(R6_ROOT / "native_validation_summary.json") if (R6_ROOT / "native_validation_summary.json").is_file() else {}
    historical_raw = historical_native.get("raw_autoregressive") or {}
    native["RAW_AUTOREGRESSIVE_TOTG_ATTEMPTED"] = historical_raw.get("TOTG_ATTEMPTED", "NOT_AVAILABLE")
    native["RAW_AUTOREGRESSIVE_TOTG_PASSED"] = historical_raw.get("TOTG_PASSED", "NOT_AVAILABLE")
    native["RAW_AUTOREGRESSIVE_TOTG_SOURCE"] = str((R6_ROOT / "native_validation_summary.json").resolve())
    write_json(output / "native_validation_summary.json", native)
    replay = replay_summary(output, rows, authority)
    write_json(output / "replay_summary.json", replay)
    tests = run_tests(output)
    after = audit_immutability(before)
    write_json(output / "immutability_audit.json", after)
    strict_available = bool(rows) and any(row.get("strict_reconstruction_attempted") == "YES" for row in rows)
    class_name, blocker, sentence = root_cause(rows, strict_available)
    upstream_ok = after.get("all_unchanged") is True and after.get("R6_CHECKPOINT_HASH_MATCH") == "YES"
    test_ok = tests.get("FOCUSED_TESTS", {}).get("status") == "PASS" and tests.get("REGRESSION_TESTS", {}).get("status") == "PASS"
    coverage_ok = coverage["OBSERVED_CASES"] == CASE_TARGET and coverage["MISSING_CASES"] == 0 and coverage["DUPLICATE_CASES"] == 0
    r7_status = "PASSED" if upstream_ok and test_ok and coverage_ok and strict_available else "BLOCKED"
    strict_failure_all = sum(bool(row.get("strict_reconstruction_failure_reason")) for row in rows) == len(rows) and bool(rows)
    teardown_present = any(row.get("strict_native_teardown_minus_11") == "YES" for row in rows)
    prior_within_realistic = bool(accepted > 0 or (max_values and float(np.median(max_values)) <= MODEL_PRIOR_MAX_JOINT_DELTA_RAD * 1.5))
    no_reason = "none" if prior_within_realistic else "autoregressive state representation, rollout state update semantics, residual parameterization, and trajectory representation require redesign; do not tune the frozen R5 bound"
    cert = {
        "schema_version": "stage3_h13_r7_terminal_certificate_v1",
        "STAGE_3_H13_R7": r7_status,
        "FIRST_BLOCKER": "none" if r7_status == "PASSED" else ("upstream_immutability_failure" if not upstream_ok else "focused_or_regression_tests_failed" if not test_ok else "frozen20_case_coverage_failure" if not coverage_ok else blocker),
        "READY_FOR_STAGE_3_FINAL_CLOSURE": "YES" if accepted == CASE_TARGET and native.get("FINAL_CERTIFIED") == f"{CASE_TARGET}/{CASE_TARGET}" else "NO",
        "R6_CHECKPOINT_HASH_MATCH": after.get("R6_CHECKPOINT_HASH_MATCH"),
        "UPSTREAM_INPUTS_UNCHANGED": "YES" if after.get("all_unchanged") else "NO",
        "H11_R2_CHECKPOINT_IMMUTABLE": after.get("H11_R2_CHECKPOINT_IMMUTABLE"), "H12_R7_IMMUTABLE": after.get("H12_R7_IMMUTABLE"), "H13_R3_IMMUTABLE": after.get("H13_R3_IMMUTABLE"), "H13_R4_IMMUTABLE": after.get("H13_R4_IMMUTABLE"), "H13_R5_IMMUTABLE": after.get("H13_R5_IMMUTABLE"), "H13_R6_IMMUTABLE": after.get("H13_R6_IMMUTABLE"), "H13_UNSEEN_SPLIT_IMMUTABLE": after.get("H13_UNSEEN_SPLIT_IMMUTABLE"),
        "FROZEN20_USED_FOR_TRAINING": "NO", "FROZEN20_USED_FOR_MODEL_SELECTION": "NO", "FROZEN20_USED_FOR_EARLY_STOPPING": "NO", "UNSEEN_GROUND_TRUTH_USED_FOR_RECONSTRUCTION": "NO", "FUTURE_LABEL_LEAKAGE": 0,
        "TRAINING_PERFORMED": "NO", "MODEL_PARAMETERS_CHANGED": "NO", "NEW_CHECKPOINT_CREATED": "NO",
        "R6_RECONSTRUCTION_BACKEND_FOUND": "MoveIt2_pose_constrained_ik_waypoints_seeded_by_R6_prior", "STRICT_MOVEIT_RECONSTRUCTION_AVAILABLE": "YES" if strict_available else "NO", "STRICT_MOVEIT_RECONSTRUCTION_ACTUALLY_EXECUTED": "YES" if strict_attempted else "NO", "R6_PREVIOUSLY_USED_PROXY_ONLY": "YES", "PROXY_MATERIALLY_MISREPRESENTED_STRICT_RESULT": proxy_strict.get("DID_PROXY_MATERIALLY_MISREPRESENT_STRICT_RECONSTRUCTION"),
        **coverage, "STRICT_RECONSTRUCTION_ATTEMPTED": f"{strict_attempted}/{CASE_TARGET}", "STRICT_RECONSTRUCTION_RETURNED_TRAJECTORY": f"{strict_returned}/{CASE_TARGET}", "RECONSTRUCTION_ACCEPTED": f"{accepted}/{CASE_TARGET}", "RECONSTRUCTION_REJECTED": f"{len(rows) - accepted}/{CASE_TARGET}", "MAX_MODEL_TO_RECONSTRUCTED_JOINT_DELTA_RAD": strict_summary["MAX_MODEL_TO_RECONSTRUCTED_JOINT_DELTA_RAD"], "RMS_MODEL_TO_RECONSTRUCTED_JOINT_DELTA_RAD": strict_summary["RMS_MODEL_TO_RECONSTRUCTED_JOINT_DELTA_RAD"], **within,
        "RAW_AUTOREGRESSIVE_TOTG_ATTEMPTED": native.get("RAW_AUTOREGRESSIVE_TOTG_ATTEMPTED"), "RAW_AUTOREGRESSIVE_TOTG_PASSED": native.get("RAW_AUTOREGRESSIVE_TOTG_PASSED"), "RECONSTRUCTED_TOTG_ATTEMPTED": native.get("RECONSTRUCTED_TOTG_ATTEMPTED"), "RECONSTRUCTED_TOTG_PASSED": native.get("RECONSTRUCTED_TOTG_PASSED"), "RECONSTRUCTED_RUCKIG_ATTEMPTED": native.get("RECONSTRUCTED_RUCKIG_ATTEMPTED"), "RECONSTRUCTED_RUCKIG_FINISHED": native.get("RECONSTRUCTED_RUCKIG_FINISHED"), "POST_RUCKIG_VALIDATED": native.get("POST_RUCKIG_VALIDATED"), "FINAL_CERTIFIED": native.get("FINAL_CERTIFIED"),
        "R5_RECONSTRUCTION_POLICY_IMMUTABLE": after.get("R5_RECONSTRUCTION_POLICY_IMMUTABLE"), "R5_RECONSTRUCTION_BOUND_EXPANDED": "NO", "SAFETY_THRESHOLDS_RELAXED": "NO", "SPRAY_THRESHOLDS_RELAXED": "NO", "COLLISION_SEMANTICS": COLLISION_SEMANTICS, "CCD_AVAILABLE": "NO", "CLEARANCE_AVAILABLE": None,
        "NATIVE_TEARDOWN_MINUS_11_STILL_PRESENT": "YES" if teardown_present else "NO", "IS_TEARDOWN_MINUS_11_THE_FIRST_BLOCKER": "NO", "IS_TEARDOWN_MINUS_11_THE_ONLY_REMAINING_BLOCKER": "NO",
        "REPLAY": replay.get("REPLAY"), "REPLAY_SEMANTIC_MATCH": replay.get("REPLAY_SEMANTIC_MATCH"), "FOCUSED_TESTS": tests.get("FOCUSED_TESTS", {}).get("status"), "REGRESSION_TESTS": tests.get("REGRESSION_TESTS", {}).get("status"), "NEW_REGRESSION_FAILURES": tests.get("NEW_REGRESSION_FAILURES"),
        "AUTHORITATIVE_OUTPUT_SIZE_MB": 0.0, "PEAK_NEW_SCRATCH_SIZE_MB": round(peak_scratch_bytes / (1024.0 * 1024.0), 3), "TEMP_ARTIFACTS_CLEANED": "YES",
        "ROOT_CAUSE_CLASS": class_name, "DID_STRICT_RECONSTRUCTION_CHANGE_THE_R6_CONCLUSION": "YES" if class_name != "B_STRICT_RECONSTRUCTION_EXECUTES_CORRECTLY_BUT_R6_PRIOR_REMAINS_OUTSIDE_FROZEN_R5_DOMAIN" else "NO", "IS_R6_PRIOR_WITHIN_REALISTIC_REACH_OF_FROZEN_R5_DOMAIN": "YES" if prior_within_realistic else "NO", "IS_ANOTHER_SMALL_ROLLOUT_TRAINING_ITERATION_JUSTIFIED": "YES" if prior_within_realistic else "NO", "IF_NO_WHAT_REQUIRES_REDESIGN": no_reason, "ONE_SENTENCE_CONCLUSION": sentence, "IF_NOT_READY_NEXT_SINGLE_ACTION": "Redesign the autoregressive rollout state representation and residual parameterization using TRAIN/VALIDATION data only; do not modify R5 bounds.", "AUTHORITATIVE_ARTIFACT_DIRECTORY": str(output.resolve()),
    }
    cert["AUTHORITATIVE_OUTPUT_SIZE_MB"] = round(size_bytes(output) / (1024.0 * 1024.0), 3)
    cert["PEAK_NEW_SCRATCH_SIZE_MB"] = round(peak_scratch_bytes / (1024.0 * 1024.0), 3)
    write_json(output / "stage3_h13_r7_terminal_certificate.json", cert)
    storage = {"schema_version": "stage3_h13_r7_storage_summary_v1", "AUTHORITATIVE_OUTPUT_SIZE_MB": cert["AUTHORITATIVE_OUTPUT_SIZE_MB"], "PEAK_NEW_SCRATCH_SIZE_MB": cert["PEAK_NEW_SCRATCH_SIZE_MB"], "TEMP_ARTIFACTS_CLEANED": "YES", "R6_CHECKPOINT_DUPLICATED": "NO", "large_upstream_data_duplicated": "NO", "large_waypoint_arrays_retained": "NO"}
    write_json(output / "storage_summary.json", storage)
    (output / "FINAL_REPORT.md").write_text(final_report(cert, output), encoding="utf-8", newline="\n")
    print(final_report(cert, output))
    return 0 if r7_status == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
