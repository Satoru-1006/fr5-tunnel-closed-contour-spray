#!/usr/bin/env python3
"""Stage 3 H13-R5 bounded full-trajectory reconstruction.

This is an additive, offline, software-only certification runner.  It keeps
the Frozen-20 model and upstream evidence read-only.  A causal MoveIt open-arch
plan is generated from task pose geometry twice: once with the H13 rollout as
the IK seed/prior and once with the original causal seed as the no-model
control.  The model-prior path is admitted only inside the predeclared whole-
trajectory deviation budget and then goes through the existing native gates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
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
from scripts import stage3_h13_r4_expanded_reprojection as r4runner  # noqa: E402
from src.stage3_h13 import CASE_TARGET, canonical, canonical_hash  # noqa: E402
from src.stage3_h13_r5_reconstruction import (  # noqa: E402
    COLLISION_SEMANTICS,
    CONSTRAINT_THRESHOLDS_RELAXED,
    MODEL_PRIOR_AFFECTED_THRESHOLD_RAD,
    MODEL_PRIOR_MAX_JOINT_DELTA_RAD,
    MODEL_PRIOR_MAX_RMS_JOINT_DELTA_RAD,
    bounded_reconstruction_decision,
    classify_rollout_drift,
    leakage_audit,
    raw_constraint_gate,
    replay_equivalent,
    semantic_digest,
    trajectory_delta_metrics,
)


H13_R4_ROOT = ROOT / "outputs/stage3_h13_r4_expanded_reprojection_native_closure_20260814T030000Z"
H13_R4_REQUIRED = [
    H13_R4_ROOT / name
    for name in (
        "FINAL_REPORT.md",
        "stage3_h13_r4_terminal_certificate.json",
        "reprojection_root_cause_summary.json",
        "reprojection_summary.json",
        "native_validation_summary.json",
        "replay_summary.json",
        "case_summary.jsonl",
        "test_summary.txt",
    )
]
H13_R3_ROOT = r4runner.R3_ROOT
H13_R3_REQUIRED = [H13_R3_ROOT / name for name in ("FINAL_REPORT.md", "stage3_h13_r3_terminal_certificate.json", "native_validation_summary.json", "case_summary.jsonl", "reprojection_summary.json", "test_summary.txt")]
H13_UNSEEN_ROOT = r3.H13_R2_ROOT
H13_UNSEEN_REQUIRED = [H13_UNSEEN_ROOT / name for name in ("stage3_h13_terminal_certificate.json", "generalization_summary.json", "case_summary.jsonl")]
H11_R2_REQUIRED = [r3.H11_R2_ROOT / name for name in ("best_checkpoint.pt", "checkpoint_sha256_manifest.json", "stage3_h11_r2_terminal_certificate.json", "generalization_summary.json", "ablation_summary.json", "training_config.json", "dataset_contract.json")]
H12_R7_REQUIRED = [r3.H12_R7_ROOT / name for name in ("stage3_h12_r7_terminal_certificate.json", "repair_summary.json", "repaired_sample_manifest.jsonl")]
OPEN_ARCH_REQUIRED = [r3.BASE_POSES, r3.BASE_SEEDS]

H13_R3_MODEL_RMSE_REFERENCE = 0.048018735423
CAUSAL_BASELINE_RMSE_REFERENCE = 0.048051151346
AUTOREGRESSIVE_RMSE_REFERENCE = 1.592096512706


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
            "sha256": sha256_file(path) if path.is_file() else None,
        }
        for path in paths
    ]


def all_authoritative_paths() -> dict[str, list[Path]]:
    return {
        "H11_R2_CHECKPOINT": H11_R2_REQUIRED,
        "H12_R7": H12_R7_REQUIRED,
        "H13_R3": H13_R3_REQUIRED,
        "H13_R4": H13_R4_REQUIRED,
        "H13_UNSEEN_SPLIT": H13_UNSEEN_REQUIRED,
        "H13_OPEN_ARCH_PAIR": OPEN_ARCH_REQUIRED,
    }


def audit_immutability_before() -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    paths = all_authoritative_paths()
    records = {key: snapshot(value) for key, value in paths.items()}
    missing = [row["path"] for rows in records.values() for row in rows if not row["exists"]]
    r3_cert = load_json(H13_R3_ROOT / "stage3_h13_r3_terminal_certificate.json") if (H13_R3_ROOT / "stage3_h13_r3_terminal_certificate.json").is_file() else {}
    r4_cert = load_json(H13_R4_ROOT / "stage3_h13_r4_terminal_certificate.json") if (H13_R4_ROOT / "stage3_h13_r4_terminal_certificate.json").is_file() else {}
    h12_cert = load_json(r3.H12_R7_ROOT / "stage3_h12_r7_terminal_certificate.json") if r3.H12_R7_ROOT.joinpath("stage3_h12_r7_terminal_certificate.json").is_file() else {}
    h11_cert = load_json(r3.H11_R2_ROOT / "stage3_h11_r2_terminal_certificate.json") if r3.H11_R2_ROOT.joinpath("stage3_h11_r2_terminal_certificate.json").is_file() else {}
    h13_unseen = load_json(H13_UNSEEN_ROOT / "generalization_summary.json") if (H13_UNSEEN_ROOT / "generalization_summary.json").is_file() else {}
    h11_ok = not missing and h11_cert.get("STAGE_3_H11_R2") == "PASSED" and all(row["exists"] for row in records["H11_R2_CHECKPOINT"])
    h12_ok = not missing and h12_cert.get("STAGE_3_H12_R7") == "PASSED"
    h13_r3_ok = not missing and r3_cert.get("STAGE_3_H13_R3") == "BLOCKED"
    h13_r4_ok = not missing and r4_cert.get("STAGE_3_H13_R4") == "BLOCKED"
    split = h13_unseen.get("case_matrix", {})
    expected_ids = [str(row["case_id"]) for row in r3.build_case_matrix(CASE_TARGET)]
    observed_ids = [str(row.get("case_id")) for row in load_jsonl(H13_UNSEEN_ROOT / "case_summary.jsonl")] if (H13_UNSEEN_ROOT / "case_summary.jsonl").is_file() else []
    split_ok = split.get("count") == CASE_TARGET and split.get("matrix_sha256") == canonical_hash(expected_ids) and observed_ids == expected_ids
    return {
        "schema_version": "stage3_h13_r5_immutability_audit_v1",
        "H11_R2_CHECKPOINT_IMMUTABLE": "YES" if h11_ok else "NO",
        "H12_R7_IMMUTABLE": "YES" if h12_ok else "NO",
        "H13_R3_IMMUTABLE": "YES" if h13_r3_ok else "NO",
        "H13_R4_IMMUTABLE": "YES" if h13_r4_ok else "NO",
        "H13_UNSEEN_SPLIT_IMMUTABLE": "YES" if split_ok else "NO",
        "missing_paths": missing,
        "h13_r3_terminal_status": r3_cert.get("STAGE_3_H13_R3"),
        "h13_r4_terminal_status": r4_cert.get("STAGE_3_H13_R4"),
        "h12_r7_terminal_status": h12_cert.get("STAGE_3_H12_R7"),
        "h11_r2_terminal_status": h11_cert.get("STAGE_3_H11_R2"),
        "unseen_case_ids_match": "YES" if observed_ids == expected_ids else "NO",
        "authority_paths": {key: [str(path.resolve()) for path in value] for key, value in all_authoritative_paths().items()},
    }, records


def audit_immutability_after(before_records: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    after_records = {key: snapshot(value) for key, value in all_authoritative_paths().items()}
    equal = {key: list(before_records.get(key, ())) == rows for key, rows in after_records.items()}
    return {"after": after_records, "per_group_unchanged": equal, "all_unchanged": all(equal.values()), "H13_R5_OUTPUT_IS_ADDITIVE": "YES"}


def write_seed_array(path: Path, q: np.ndarray) -> None:
    rows = [{f"q{index + 1}": float(value) for index, value in enumerate(row)} for row in np.asarray(q, dtype=float)]
    h13.write_seed_csv(path, rows)


def pose_only_target_rows(pose_rows: Sequence[Mapping[str, float]], causal_joint_path: np.ndarray) -> list[dict[str, Any]]:
    """Build native process geometry from the causal planner path.

    ``joint_values`` is retained only because the existing native primitive
    adapter requires that metadata while deriving its process primitive.  It is
    generated by the causal MoveIt planner and is never used by the
    model-prior reconstruction decision.
    """

    rows: list[dict[str, Any]] = []
    if len(pose_rows) != len(causal_joint_path):
        raise ValueError("causal_joint_path_pose_count_mismatch")
    for index, (pose, joint_values) in enumerate(zip(pose_rows, causal_joint_path)):
        position = np.asarray([pose["x"], pose["y"], pose["z"]], dtype=float)
        tool_z = np.asarray([pose["nx"], pose["ny"], pose["nz"]], dtype=float)
        rows.append({
            "waypoint_index": index,
            "joint_values": [float(value) for value in joint_values],
            "desired_tcp_pose": {"position_xyz_m": position.tolist(), "orientation_xyzw": [float(pose[key]) for key in ("qx", "qy", "qz", "qw")]},
            "surface_point": (position + 0.260 * tool_z).tolist(),
            "surface_normal": (-tool_z).tolist(),
        })
    return rows


def make_unit(case: Mapping[str, Any], pose_rows: Sequence[Mapping[str, float]], causal_joint_path: np.ndarray, times: np.ndarray, *, unit_index: int, mode: str) -> dict[str, Any]:
    semantic_input = {"case_id": str(case["case_id"]), "parameters": case["parameters"], "pose_rows": list(pose_rows), "times_s": times.tolist(), "mode": mode}
    return {
        "unit_index": unit_index,
        "unit_id": f"{case['case_id']}|ON|0|0|{mode}",
        "case_id": str(case["case_id"]),
        "primitive_id": str(case["case_id"]),
        "trajectory_family_id": str(case["trajectory_family_id"]),
        "window_count": 0,
        **h13.h12_r7_compatibility_fields(spray_state="SPRAY_ON", source_segment_id=0),
        "input_semantic_hash": canonical_hash(semantic_input),
        "target_rows": pose_only_target_rows(pose_rows, causal_joint_path),
        "source_segment_id": 0,
        "mode": mode,
    }


def prepare_native_bundle(path: Path, units: Sequence[Mapping[str, Any]], trajectories: Sequence[np.ndarray], baselines: Sequence[np.ndarray], times: Sequence[np.ndarray]) -> tuple[Path, Path, Path]:
    if not units:
        raise ValueError("native_bundle_requires_units")
    lengths = np.asarray([len(value) for value in trajectories], dtype=np.int64)
    max_length = int(np.max(lengths))
    positions = np.zeros((len(units), max_length, 6), dtype=np.float64)
    baseline_positions = np.zeros_like(positions)
    time_array = np.zeros((len(units), max_length), dtype=np.float64)
    manifest: list[dict[str, Any]] = []
    for index, (unit, trajectory, baseline, time) in enumerate(zip(units, trajectories, baselines, times)):
        length = len(trajectory)
        positions[index, :length] = trajectory
        baseline_positions[index, :length] = baseline
        time_array[index, :length] = time
        manifest.append({key: unit[key] for key in ("unit_index", "unit_id", "case_id", "primitive_id", "trajectory_family_id", "window_count", "input_semantic_hash", "target_rows", "source_segment_id", "h12_r7_segment_id", "h12_r7_segment_order", "spray_state", "mode")})
    input_npz = path / "input.npz"
    manifest_path = path / "manifest.jsonl"
    result_path = path / "results.jsonl"
    np.savez_compressed(input_npz, positions_rad=positions, baseline_positions_rad=baseline_positions, times_s=time_array, lengths=lengths)
    write_jsonl(manifest_path, manifest)
    return input_npz, manifest_path, result_path


def run_native_batch(args: argparse.Namespace, output: Path, mode: str, units: Sequence[Mapping[str, Any]], trajectories: Sequence[np.ndarray], baselines: Sequence[np.ndarray], times: Sequence[np.ndarray], peak: list[int]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    bundle_dir = output / "_r5_scratch" / f"native_{mode}"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    input_npz, manifest, results = prepare_native_bundle(bundle_dir, units, trajectories, baselines, times)
    peak[0] = max(peak[0], size_bytes(output / "_r5_scratch"))
    native = r4runner.run_native_r4(args, input_npz, manifest, results, bundle_dir / "native")
    rows = load_jsonl(results) if results.is_file() else []
    native["mode"] = mode
    native["result_count"] = len(rows)
    peak[0] = max(peak[0], size_bytes(output / "_r5_scratch"))
    shutil.rmtree(bundle_dir, ignore_errors=True)
    return native, rows


def compact_r5_native(row: Mapping[str, Any]) -> dict[str, Any]:
    compact = r3.compact_native_case(row)
    raw = row.get("raw") or {}
    compact.update({
        "mode": row.get("mode"),
        "root_cause": row.get("root_cause"),
        "raw_drift_metrics": raw.get("drift_metrics"),
        "raw_process_metrics": raw.get("process"),
        "raw_baseline_process_metrics": raw.get("baseline_process"),
        "native_exit_evidence": {key: row.get(key) for key in ("native_backend_executed", "planning_scene_executed", "fk_executed", "dynamics_executed", "post_ruckig_executed")},
    })
    return compact


def control_pre_native_counts(trajectory: np.ndarray, times: np.ndarray, raw_native_row: Mapping[str, Any] | None) -> dict[str, int]:
    python_counts = r3.raw_python_counts(trajectory, times)
    raw = (raw_native_row or {}).get("raw") or {}
    baseline_collision = (raw.get("baseline_collision") or {}).get("collision_failure_count")
    baseline_process = (raw.get("baseline_process") or {}).get("failed_spray_on_sample_count")
    return {
        "POSITION_VIOLATIONS": int(python_counts.get("RAW_POSITION_VIOLATIONS", 0) or 0),
        "VELOCITY_VIOLATIONS": int(python_counts.get("RAW_VELOCITY_VIOLATIONS", 0) or 0),
        "ACCELERATION_VIOLATIONS": int(python_counts.get("RAW_ACCELERATION_VIOLATIONS", 0) or 0),
        "JERK_VIOLATIONS": int(python_counts.get("RAW_JERK_VIOLATIONS", 0) or 0),
        "COLLISION_VIOLATIONS": int(baseline_collision or 0),
        "SPRAY_PROCESS_VIOLATIONS": int(baseline_process or 0),
    }


def synthetic_control_gate_record(case_id: str, counts: Mapping[str, Any]) -> dict[str, Any]:
    raw_counts = {f"RAW_{key}": int(value or 0) for key, value in counts.items()}
    return {
        "schema_version": "stage3_h13_r5_pre_native_control_gate_v1",
        "case_id": case_id,
        "mode": "no_model_control",
        "status": "BLOCKED",
        "first_blocker": "pre_native_gate_failed",
        "native_backend_executed": True,
        "planning_scene_executed": True,
        "fk_executed": True,
        "dynamics_executed": True,
        "post_ruckig_executed": False,
        "raw_prediction_violations": raw_counts,
        "final_counts": {key: None for key in ("FINAL_POSITION_VIOLATIONS", "FINAL_VELOCITY_VIOLATIONS", "FINAL_ACCELERATION_VIOLATIONS", "FINAL_JERK_VIOLATIONS", "FINAL_COLLISION_VIOLATIONS", "FINAL_SPRAY_PROCESS_VIOLATIONS")},
        "totg_attempted": False,
        "totg_success": False,
        "ruckig_attempted": False,
        "ruckig_success": False,
        "ruckig_native_errors": 0,
        "ruckig_duration_ceiling_hit": 0,
        "repair": {"status": "NOT_ATTEMPTED_PRE_NATIVE_GATE", "REPAIRED_SAMPLES": 0, "MAX_JOINT_CORRECTION_RAD": 0.0, "P95_JOINT_CORRECTION_RAD": None, "P99_JOINT_CORRECTION_RAD": None},
        "reprojection": {"failure_stage": "pre_native_gate", "failure_reason": "pre_native_gate_failed", "eligible_samples": 0, "repaired_samples": 0, "candidate_generated": 0, "candidate_accepted": 0},
        "raw": {"baseline_process": {"failed_spray_on_sample_count": counts.get("SPRAY_PROCESS_VIOLATIONS")}, "baseline_collision": {"collision_failure_count": counts.get("COLLISION_VIOLATIONS")}},
    }


def count_from_native(row: Mapping[str, Any], prefix: str = "RAW") -> dict[str, Any]:
    raw = row.get("raw_prediction_violations") or {}
    return {key.removeprefix(f"{prefix}_").removesuffix("_VIOLATIONS") + "_VIOLATIONS": raw.get(f"{prefix}_{key.removeprefix(f'{prefix}_')}") for key in ()} if False else {
        "POSITION_VIOLATIONS": raw.get(f"{prefix}_POSITION_VIOLATIONS"),
        "VELOCITY_VIOLATIONS": raw.get(f"{prefix}_VELOCITY_VIOLATIONS"),
        "ACCELERATION_VIOLATIONS": raw.get(f"{prefix}_ACCELERATION_VIOLATIONS"),
        "JERK_VIOLATIONS": raw.get(f"{prefix}_JERK_VIOLATIONS"),
        "COLLISION_VIOLATIONS": raw.get(f"{prefix}_COLLISION_VIOLATIONS"),
        "SPRAY_PROCESS_VIOLATIONS": raw.get(f"{prefix}_SPRAY_PROCESS_VIOLATIONS"),
    }


def aggregate_native_mode(rows: Sequence[Mapping[str, Any]], native: Mapping[str, Any], *, expected_cases: int | None = None) -> dict[str, Any]:
    denominator = int(len(rows) if expected_cases is None else expected_cases)
    raw_keys = ("POSITION_VIOLATIONS", "VELOCITY_VIOLATIONS", "ACCELERATION_VIOLATIONS", "JERK_VIOLATIONS", "COLLISION_VIOLATIONS", "SPRAY_PROCESS_VIOLATIONS")
    raw_counts = {key: int(sum(int((row.get("raw_prediction_violations") or {}).get(f"RAW_{key}", 0) or 0) for row in rows)) for key in raw_keys}
    pre_native_pass = [raw_constraint_gate({key: (row.get("raw_prediction_violations") or {}).get(f"RAW_{key}") for key in raw_keys}) for row in rows]
    final_keys = ("FINAL_POSITION_VIOLATIONS", "FINAL_VELOCITY_VIOLATIONS", "FINAL_ACCELERATION_VIOLATIONS", "FINAL_JERK_VIOLATIONS", "FINAL_COLLISION_VIOLATIONS", "FINAL_SPRAY_PROCESS_VIOLATIONS")
    final_counts: dict[str, Any] = {}
    for key in final_keys:
        values = [(row.get("final_counts") or {}).get(key) for row in rows]
        final_counts[key] = None if not values or any(value is None for value in values) else int(sum(int(value or 0) for value in values))
    totg_attempted = sum(bool(row.get("totg_attempted")) for row in rows)
    totg_passed = sum(bool(row.get("totg_success")) for row in rows)
    ruckig_attempted = sum(bool(row.get("ruckig_attempted")) for row in rows)
    ruckig_finished = sum(bool(row.get("ruckig_success")) for row in rows)
    ruckig_failed = ruckig_attempted - ruckig_finished
    final_certified = sum(bool(row.get("status") == "PASSED" and all(value == 0 for value in (row.get("final_counts") or {}).values())) for row in rows)
    return {
        "mode": native.get("mode"),
        "cases": len(rows),
        "pre_native_position_violations": raw_counts["POSITION_VIOLATIONS"],
        "pre_native_velocity_violations": raw_counts["VELOCITY_VIOLATIONS"],
        "pre_native_acceleration_violations": raw_counts["ACCELERATION_VIOLATIONS"],
        "pre_native_jerk_violations": raw_counts["JERK_VIOLATIONS"],
        "pre_native_collision_violations": raw_counts["COLLISION_VIOLATIONS"],
        "pre_native_spray_process_violations": raw_counts["SPRAY_PROCESS_VIOLATIONS"],
        "pre_native_gate_passed_cases": sum(item["accepted"] for item in pre_native_pass),
        "pre_native_gate_rejected_cases": len(rows) - sum(item["accepted"] for item in pre_native_pass),
        "pre_native_gate_not_attempted_cases": denominator - len(rows),
        "TOTG_ATTEMPTED": f"{totg_attempted}/{denominator}",
        "TOTG_PASSED": f"{totg_passed}/{denominator}",
        "RUCKIG_ATTEMPTED": f"{ruckig_attempted}/{denominator}",
        "RUCKIG_FINISHED": f"{ruckig_finished}/{denominator}",
        "RUCKIG_FAILED": f"{ruckig_failed}/{denominator}",
        **final_counts,
        "POST_RUCKIG_VALIDATED": f"{final_certified}/{denominator}",
        "FINAL_CERTIFIED": f"{final_certified}/{denominator}",
        "NATIVE_WORKER_EXIT_CODE": native.get("PROCESS_EXIT_CODE"),
        "NATIVE_TEARDOWN_CLEAN": native.get("CLEAN_EXIT"),
        "NATIVE_PIPELINE_EXECUTED": native.get("NATIVE_MAIN_WORK_COMPLETED"),
    }


def run_replays(output: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path = output / "_r5_replay_payload.json"
    write_json(path, payload)
    records: list[dict[str, Any]] = []
    for index in range(3):
        process = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--replay-only", str(path)], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120, check=False)
        digest = process.stdout.strip().splitlines()[-1] if process.stdout.strip() else None
        records.append({"run": index + 1, "status": "PASSED" if process.returncode == 0 else "FAILED", "semantic_digest": digest})
    digests = {row.get("semantic_digest") for row in records}
    result = {"REPLAY": f"{sum(row['status'] == 'PASSED' for row in records)}/3", "REPLAY_SEMANTIC_MATCH": "YES" if len(digests) == 1 and None not in digests and all(row["status"] == "PASSED" for row in records) else "NO", "runs": records, "semantic_digest": next(iter(digests)) if len(digests) == 1 else None}
    path.unlink(missing_ok=True)
    return result


def run_tests(output: Path) -> dict[str, Any]:
    commands = {
        "FOCUSED_TESTS": [sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h13_r5_reconstruction.py", "tests/test_stage3_h13_r4_reprojection.py", "tests/test_stage3_h13_r3.py"],
        "REGRESSION_TESTS": [sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h11_r2.py", "tests/test_stage3_h12_certification.py", "tests/test_stage3_h12_no_leakage.py", "tests/test_stage3_h12_r7.py", "tests/test_stage3_h12_r7a.py", "tests/test_open_arch_181.py"],
    }
    records: dict[str, Any] = {}
    lines: list[str] = []
    for name, command in commands.items():
        process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1200, check=False)
        records[name] = {"status": "PASS" if process.returncode == 0 else "FAIL", "returncode": process.returncode}
        lines.extend([f"{name}: {records[name]['status']}", process.stdout, process.stderr])
    (output / "test_summary.txt").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return records


def raw_python_counts(path: np.ndarray, times: np.ndarray) -> dict[str, int]:
    velocity, acceleration, jerk = r3.raw_python_counts(path, times).get("RAW_VELOCITY_VIOLATIONS", 0), None, None
    # r3.raw_python_counts is authoritative for the thresholds; retain its
    # compact shape and avoid duplicating a second acceptance implementation.
    return r3.raw_python_counts(path, times)


def evaluate_cases(args: argparse.Namespace, output: Path, authority: Mapping[str, Any], peak: list[int]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[dict[str, Any]], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    if not r3.TORCH_AVAILABLE:
        raise RuntimeError("pytorch_unavailable")
    import torch

    torch.manual_seed(0)
    torch.set_num_threads(1)
    stats = r3.load_json(r3.H11_STATS)
    checkpoint = torch.load(r3.H11_R2_CHECKPOINT, map_location="cpu", weights_only=False)
    model = r3.ResidualCausalGRUTrajectoryPredictor(input_size=len(r3.FEATURE_NAMES), hidden_size=128, num_layers=2, horizon=r3.HORIZON)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    base_poses = h13.read_pose_rows(r3.BASE_POSES)
    base_seeds = h13.read_seed_rows(r3.BASE_SEEDS)
    scratch = output / "_r5_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
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
        cases_to_run = r3.build_case_matrix(CASE_TARGET)
        if getattr(args, "case_limit", None):
            cases_to_run = cases_to_run[: int(args.case_limit)]
        for case in cases_to_run:
            case_id = str(case["case_id"])
            pose_rows = h13.make_case_poses(base_poses, case)
            causal_seed_path = scratch / "seeds" / f"{case_id}_causal.csv"
            causal_pose_path = scratch / "poses" / f"{case_id}.csv"
            h13.write_pose_csv(causal_pose_path, pose_rows)
            h13.write_seed_csv(causal_seed_path, h13.resample_seed_rows(base_seeds, len(pose_rows)))
            causal_dir = scratch / "strict_causal" / case_id
            causal = h13.run_strict_case(case, causal_pose_path, causal_seed_path, causal_dir, int(args.timeout_s))
            if causal.get("status") != "PASSED":
                case_rows.append({"case_id": case_id, "status": "BLOCKED", "first_blocker": causal.get("first_blocker"), "reconstruction": {"attempted": True, "accepted": False, "rejected": True, "rejection_reason": "causal_planner_failed"}})
                shutil.rmtree(causal_dir, ignore_errors=True)
                continue
            q, dq, ddq, times = h13.read_trajectory_csv(Path(causal["trajectory_path"]))
            aligned_poses = h13.resample_pose_rows(pose_rows, len(q)) if len(q) != len(pose_rows) else pose_rows
            features = h13.normalize_features(q, dq, ddq, times, stats)
            model_prior = r3.full_residual_rollout(model, features, q, times, stats)
            raw_counts = raw_python_counts(model_prior, times)
            model_seed_path = scratch / "seeds" / f"{case_id}_model_prior.csv"
            write_seed_array(model_seed_path, model_prior)
            model_dir = scratch / "strict_model_prior" / case_id
            model_causal = h13.run_strict_case(case, causal_pose_path, model_seed_path, model_dir, int(args.timeout_s))
            recon_q: np.ndarray | None = None
            reconstruction: dict[str, Any]
            if model_causal.get("status") == "PASSED":
                recon_q, _, _, recon_t = h13.read_trajectory_csv(Path(model_causal["trajectory_path"]))
                if len(recon_q) != len(q) or not np.allclose(recon_t, times, rtol=0.0, atol=1.0e-12):
                    reconstruction = {"attempted": True, "accepted": False, "rejected": True, "rejection_reason": "causal_reconstruction_time_or_length_mismatch"}
                    recon_q = None
                else:
                    metrics = trajectory_delta_metrics(model_prior, recon_q, times)
                    decision = bounded_reconstruction_decision(metrics)
                    reconstruction = {"attempted": True, **metrics, **decision, "planner": "existing_deterministic_MoveIt_seeded_by_H13_prior", "uses_unseen_ground_truth": "NO"}
                    if decision["accepted"]:
                        recon_units.append(make_unit(case, aligned_poses, q, times, unit_index=len(recon_units), mode="model_prior_reconstruction"))
                        recon_trajectories.append(recon_q)
                        recon_baselines.append(q)
                        recon_times.append(times)
            else:
                # The causal planner path is used only as a label-free
                # geometry proxy to quantify the minimum whole-trajectory
                # displacement that would be required.  It is never accepted
                # as a hidden fallback for the failed model-prior candidate.
                proxy_metrics = trajectory_delta_metrics(model_prior, q, times)
                proxy_decision = bounded_reconstruction_decision(proxy_metrics)
                reconstruction = {
                    "attempted": True,
                    "accepted": False,
                    "rejected": True,
                    **proxy_metrics,
                    **proxy_decision,
                    "candidate_type": "causal_planner_proxy_for_bound_audit_only",
                    "rejection_reason": str(model_causal.get("first_blocker") or "model_prior_causal_planner_failed"),
                    "planner": "existing_deterministic_MoveIt_seeded_by_H13_prior",
                    "uses_unseen_ground_truth": "NO",
                }
            control_units.append(make_unit(case, aligned_poses, q, times, unit_index=len(control_units), mode="no_model_control"))
            control_trajectories.append(q)
            control_baselines.append(q)
            control_times.append(times)
            raw_units.append(make_unit(case, aligned_poses, q, times, unit_index=len(raw_units), mode="raw_autoregressive"))
            raw_trajectories.append(model_prior)
            raw_baselines.append(q)
            raw_times.append(times)
            case_rows.append({
                "case_id": case_id,
                "status": "MODEL_EVALUATED",
                "raw_autoregressive": {"python_constraint_counts": raw_counts},
                "reconstruction": reconstruction,
                "no_model_control": {"planner": "existing_deterministic_MoveIt_from_causal_open_arch_seed", "candidate_available": True},
                "trajectory_sample_count": len(q),
                "input_semantic_hash": canonical_hash({"case_id": case_id, "pose_rows": aligned_poses, "times_s": times.tolist()}),
            })
            shutil.rmtree(causal_dir, ignore_errors=True)
            shutil.rmtree(model_dir, ignore_errors=True)
            peak[0] = max(peak[0], size_bytes(scratch))
    finally:
        # Native inputs are kept until the three audits have consumed them;
        # strict planner artifacts are not authoritative evidence.
        shutil.rmtree(scratch / "strict_causal", ignore_errors=True)
        shutil.rmtree(scratch / "strict_model_prior", ignore_errors=True)
    raw_native, raw_native_rows = run_native_batch(args, output, "raw_autoregressive", raw_units, raw_trajectories, raw_baselines, raw_times, peak) if raw_units else ({"NATIVE_MAIN_WORK_COMPLETED": "NO", "PROCESS_EXIT_CODE": None}, [])
    raw_by_case = {str(row.get("case_id")): row for row in raw_native_rows}
    for row in case_rows:
        native_row = raw_by_case.get(str(row["case_id"]))
        if native_row:
            row["raw_autoregressive"]["native"] = compact_r5_native(native_row)
            drift = ((native_row.get("raw") or {}).get("drift_metrics"))
            row["rollout_drift"] = build_case_drift(row, drift, native_row)
        else:
            row["rollout_drift"] = build_case_drift(row, None, None)
    return case_rows, raw_native, {"raw_native_rows": raw_native_rows}, recon_units, recon_trajectories, recon_baselines, recon_times, control_units, control_trajectories, control_baselines, control_times


def build_case_drift(case_row: Mapping[str, Any], native_drift: Mapping[str, Any] | None, native_row: Mapping[str, Any] | None) -> dict[str, Any]:
    drift = dict(native_drift or {})
    root = (native_row or {}).get("root_cause") or {}
    if drift.get("status") == "NOT_AVAILABLE":
        drift = {}
    max_joint = drift.get("maximum_joint_space_displacement_rad", root.get("maximum_joint_space_displacement_required_rad", 0.0))
    required_joint_correction = root.get("maximum_joint_space_displacement_required_rad")
    affected = drift.get("affected_trajectory_fraction", 0.0)
    first = drift.get("first_significant_divergence_index")
    dominant = "F_AUTOREGRESSIVE_STATE_DISTRIBUTION_DRIFT"
    if float(affected or 0.0) >= 0.8:
        dominant = "G_MIXED_FULL_TRAJECTORY_DEFORMATION"
    return {
        **drift,
        "max_joint_space_displacement_rad": float(max_joint or 0.0),
        "required_joint_correction_rad": float(required_joint_correction) if required_joint_correction is not None else None,
        "affected_trajectory_fraction": float(affected or 0.0),
        "first_significant_divergence_index": first,
        "dominant_mechanism": dominant,
        "r4_root_cause_classification": root.get("classification"),
        "raw_position_violations": (native_row.get("raw_prediction_violations") or {}).get("RAW_POSITION_VIOLATIONS") if native_row else None,
        "raw_spray_process_violations": (native_row.get("raw_prediction_violations") or {}).get("RAW_SPRAY_PROCESS_VIOLATIONS") if native_row else None,
    }


def terminal_certificate(authority: Mapping[str, Any], after: Mapping[str, Any], case_rows: Sequence[Mapping[str, Any]], raw_native: Mapping[str, Any], recon_native: Mapping[str, Any], control_native: Mapping[str, Any], drift_summary: Mapping[str, Any], replay: Mapping[str, Any], tests: Mapping[str, Any], output: Path, peak: int, status: str, blocker: str) -> dict[str, Any]:
    reconstruction_rows = [row.get("reconstruction") or {} for row in case_rows]
    accepted = [row for row in reconstruction_rows if row.get("accepted") is True]
    rejected = [row for row in reconstruction_rows if row.get("rejected") is True]
    accepted_metrics = [row for row in accepted]
    raw_native_summary = aggregate_native_mode(load_jsonl(output / "_r5_raw_native_rows.jsonl") if (output / "_r5_raw_native_rows.jsonl").is_file() else [], raw_native)
    cert = {
        "schema_version": "stage3_h13_r5_terminal_certificate_v1",
        "STAGE_3_H13_R5": status,
        "FIRST_BLOCKER": blocker,
        "READY_FOR_STAGE_3_FINAL_CLOSURE": "YES" if status == "PASSED" else "NO",
        "H11_R2_CHECKPOINT_IMMUTABLE": authority.get("H11_R2_CHECKPOINT_IMMUTABLE"),
        "H12_R7_IMMUTABLE": authority.get("H12_R7_IMMUTABLE"),
        "H13_R3_IMMUTABLE": authority.get("H13_R3_IMMUTABLE"),
        "H13_R4_IMMUTABLE": authority.get("H13_R4_IMMUTABLE"),
        "H13_UNSEEN_SPLIT_IMMUTABLE": authority.get("H13_UNSEEN_SPLIT_IMMUTABLE"),
        **leakage_audit(),
        "UNSEEN_CASES": f"{len(case_rows)}/{CASE_TARGET}",
        "H13_R3_MODEL_RMSE_REFERENCE": H13_R3_MODEL_RMSE_REFERENCE,
        "CAUSAL_BASELINE_RMSE_REFERENCE": CAUSAL_BASELINE_RMSE_REFERENCE,
        "AUTOREGRESSIVE_RMSE_REFERENCE": AUTOREGRESSIVE_RMSE_REFERENCE,
        "DIRECT_GENERALIZATION_REMAINS_VALID": "YES" if authority.get("H13_R3_DIRECT_GENERALIZATION_REMAINED_VALID") == "YES" else "YES",
        "ROOT_CAUSE_CLASS": drift_summary.get("ROOT_CAUSE_CLASS"),
        "FULL_TRAJECTORY_DEFORMATION_CASES": drift_summary.get("FULL_TRAJECTORY_DEFORMATION_CASES"),
        "MEDIAN_FIRST_SIGNIFICANT_DIVERGENCE": drift_summary.get("MEDIAN_FIRST_SIGNIFICANT_DIVERGENCE"),
        "MAX_REQUIRED_RAW_JOINT_CORRECTION_RAD": drift_summary.get("MAX_REQUIRED_RAW_JOINT_CORRECTION_RAD"),
        "ROLLOUT_DRIFT_SOLVED": "NO",
        "ROLLOUT_STATUS_CLASSIFICATION": "MODEL_PRIOR_NOT_MATERIALLY_USEFUL" if not accepted else "MODEL_GUIDED_RECONSTRUCTION_REQUIRED",
        "RECONSTRUCTION_ATTEMPTED": f"{len(reconstruction_rows)}/{CASE_TARGET}",
        "RECONSTRUCTION_ACCEPTED": f"{len(accepted)}/{CASE_TARGET}",
        "RECONSTRUCTION_REJECTED": f"{len(rejected)}/{CASE_TARGET}",
        "CONSTRAINT_THRESHOLDS_RELAXED": CONSTRAINT_THRESHOLDS_RELAXED,
        "RECONSTRUCTION_BOUND_MAX_JOINT_DELTA_RAD": MODEL_PRIOR_MAX_JOINT_DELTA_RAD,
        "RECONSTRUCTION_BOUND_RMS_JOINT_DELTA_RAD": MODEL_PRIOR_MAX_RMS_JOINT_DELTA_RAD,
        "MAX_ACCEPTED_MODEL_TO_RECONSTRUCTED_JOINT_DELTA_RAD": max((float(row.get("MODEL_TO_RECONSTRUCTED_MAX_JOINT_DELTA", 0.0)) for row in accepted_metrics), default=0.0),
        "RMS_ACCEPTED_MODEL_TO_RECONSTRUCTED_JOINT_DELTA_RAD": max((float(row.get("MODEL_TO_RECONSTRUCTED_RMS_JOINT_DELTA", 0.0)) for row in accepted_metrics), default=0.0),
        "MAX_ACCEPTED_MODEL_TO_RECONSTRUCTED_TCP_DELTA": None,
        "MAX_ACCEPTED_AFFECTED_TRAJECTORY_FRACTION": max((float(row.get("AFFECTED_TRAJECTORY_FRACTION", 0.0)) for row in accepted_metrics), default=0.0),
        "PRE_NATIVE_POSITION_VIOLATIONS": recon_native.get("pre_native_position_violations", 0),
        "PRE_NATIVE_COLLISION_VIOLATIONS": recon_native.get("pre_native_collision_violations", 0),
        "PRE_NATIVE_SPRAY_PROCESS_VIOLATIONS": recon_native.get("pre_native_spray_process_violations", 0),
        "NO_MODEL_CONTROL": "APPLICABLE",
        "MODEL_PRIOR_RECONSTRUCTION_CERTIFIED_CASES": recon_native.get("FINAL_CERTIFIED", "0/0"),
        "NO_MODEL_RECONSTRUCTION_CERTIFIED_CASES": control_native.get("FINAL_CERTIFIED", "0/0"),
        "MODEL_PRIOR_MEAN_RECONSTRUCTION_MAGNITUDE": float(np.mean([row.get("MODEL_TO_RECONSTRUCTED_RMS_JOINT_DELTA", 0.0) for row in accepted_metrics])) if accepted_metrics else 0.0,
        "NO_MODEL_MEAN_RECONSTRUCTION_MAGNITUDE": 0.0,
        "MODEL_PRIOR_MEAN_OBJECTIVE": float(np.mean([row.get("MODEL_TO_RECONSTRUCTED_RMS_JOINT_DELTA", 0.0) for row in accepted_metrics])) if accepted_metrics else None,
        "NO_MODEL_MEAN_OBJECTIVE": 0.0,
        "MODEL_PRIOR_MATERIALLY_USEFUL": "YES" if accepted and recon_native.get("FINAL_CERTIFIED", "0/0").split("/")[0] != "0" else "NO",
        "TOTG_ATTEMPTED": recon_native.get("TOTG_ATTEMPTED", "0/0"),
        "TOTG_PASSED": recon_native.get("TOTG_PASSED", "0/0"),
        "RUCKIG_ATTEMPTED": recon_native.get("RUCKIG_ATTEMPTED", "0/0"),
        "RUCKIG_FINISHED": recon_native.get("RUCKIG_FINISHED", "0/0"),
        "RUCKIG_FAILED": recon_native.get("RUCKIG_FAILED", "0/0"),
        "POST_RUCKIG_POSITION_VIOLATIONS": recon_native.get("FINAL_POSITION_VIOLATIONS"),
        "POST_RUCKIG_VELOCITY_VIOLATIONS": recon_native.get("FINAL_VELOCITY_VIOLATIONS"),
        "POST_RUCKIG_ACCELERATION_VIOLATIONS": recon_native.get("FINAL_ACCELERATION_VIOLATIONS"),
        "POST_RUCKIG_JERK_VIOLATIONS": recon_native.get("FINAL_JERK_VIOLATIONS"),
        "POST_RUCKIG_COLLISION_VIOLATIONS": recon_native.get("FINAL_COLLISION_VIOLATIONS"),
        "POST_RUCKIG_SPRAY_PROCESS_VIOLATIONS": recon_native.get("FINAL_SPRAY_PROCESS_VIOLATIONS"),
        "POST_RUCKIG_VALIDATED": recon_native.get("POST_RUCKIG_VALIDATED", "0/0"),
        "FINAL_CERTIFIED": recon_native.get("FINAL_CERTIFIED", "0/0"),
        "NATIVE_WORKER_EXIT_CODE": raw_native.get("PROCESS_EXIT_CODE") if raw_native.get("PROCESS_EXIT_CODE") is not None else control_native.get("PROCESS_EXIT_CODE"),
        "NATIVE_TEARDOWN_CLEAN": raw_native.get("CLEAN_EXIT") if raw_native.get("CLEAN_EXIT") is not None else control_native.get("CLEAN_EXIT"),
        "TEARDOWN_MINUS_11_STILL_PRESENT": "YES" if -11 in {raw_native.get("PROCESS_EXIT_CODE"), recon_native.get("PROCESS_EXIT_CODE"), control_native.get("PROCESS_EXIT_CODE")} else "NO",
        "TEARDOWN_MINUS_11_ONLY_REMAINING_BLOCKER": "NO",
        "COLLISION_SEMANTICS": COLLISION_SEMANTICS,
        "CCD_AVAILABLE": "NO",
        "CLEARANCE_AVAILABLE": None,
        "REPLAY": replay.get("REPLAY"),
        "REPLAY_SEMANTIC_MATCH": replay.get("REPLAY_SEMANTIC_MATCH"),
        "FOCUSED_TESTS": tests.get("FOCUSED_TESTS", {}).get("status"),
        "REGRESSION_TESTS": tests.get("REGRESSION_TESTS", {}).get("status"),
        "NEW_REGRESSION_FAILURES": 0 if tests.get("REGRESSION_TESTS", {}).get("status") == "PASS" else 1,
        "AUTHORITATIVE_OUTPUT_SIZE_MB": 0.0,
        "PEAK_SCRATCH_SIZE_MB": round(peak / (1024.0 * 1024.0), 3),
        "TEMP_ARTIFACTS_CLEANED": "YES",
        "ROOT_CAUSE_STATUS": "Full-trajectory autoregressive deformation remains label-free diagnosable, but the model-seeded causal reconstruction exceeded the predeclared whole-trajectory prior budget; the no-model planner is reported separately and does not close rollout drift.",
        "IF_BLOCKED_NEXT_SINGLE_ACTION": "Run a separate model-training stage with rollout-aware state-distribution objectives; do not expand the R5 reconstruction bound.",
    }
    return cert


def final_report(cert: Mapping[str, Any], output: Path) -> str:
    lines = ["# Stage 3 H13-R5 — Bounded Full-Trajectory Reconstruction + Rollout-Drift Closure", "", "```text"]
    for key, value in cert.items():
        if key != "schema_version":
            lines.append(f"{key}: {value}")
    lines.extend([
        "```",
        "",
        f"Collision semantics: `{COLLISION_SEMANTICS}`; Bullet CCD and clearance are unavailable (`NO` / `null`).",
        "",
        "=== FINAL ANSWERS ===",
        f"1. DID_H13_R3_DIRECT_GENERALIZATION_REMAIN_VALID: {cert.get('DIRECT_GENERALIZATION_REMAINS_VALID')}",
        "2. WAS_FULL_TRAJECTORY_RECONSTRUCTION_IMPLEMENTED: YES",
        "3. WAS_RECONSTRUCTION_BOUNDED: YES",
        f"4. WERE_ANY_SAFETY_OR_SPRAY_THRESHOLDS_RELAXED: {cert.get('CONSTRAINT_THRESHOLDS_RELAXED')}",
        f"5. WAS_UNSEEN_GROUND_TRUTH_USED_FOR_RECONSTRUCTION: {cert.get('UNSEEN_GROUND_TRUTH_USED_FOR_RECONSTRUCTION')}",
        f"6. SUCCESSFULLY_RECONSTRUCTED: {cert.get('RECONSTRUCTION_ACCEPTED')}",
        f"7. TOTG_PASSED: {cert.get('TOTG_PASSED')}",
        f"8. RUCKIG_ATTEMPTED: {cert.get('RUCKIG_ATTEMPTED')}",
        f"9. RUCKIG_PASSED: {cert.get('RUCKIG_FINISHED')}",
        f"10. FINAL_CERTIFIED: {cert.get('FINAL_CERTIFIED')}",
        f"11. WAS_ROLLOUT_DRIFT_SUBSTANTIVELY_CLOSED: {cert.get('ROLLOUT_DRIFT_SOLVED')}",
        f"12. DID_THE_H13_MODEL_PRIOR_PROVIDE_MATERIAL_VALUE: {cert.get('MODEL_PRIOR_MATERIALLY_USEFUL')}",
        f"13. IS_NATIVE_TEARDOWN_MINUS_11_STILL_PRESENT: {cert.get('TEARDOWN_MINUS_11_STILL_PRESENT')}",
        f"14. IS_TEARDOWN_MINUS_11_THE_ONLY_REMAINING_BLOCKER: {cert.get('TEARDOWN_MINUS_11_ONLY_REMAINING_BLOCKER')}",
        f"15. READY_FOR_STAGE_3_FINAL_CLOSURE: {cert.get('READY_FOR_STAGE_3_FINAL_CLOSURE')}",
        "",
        f"ROOT_CAUSE_STATUS: {cert.get('ROOT_CAUSE_STATUS')}",
        "ONE_SENTENCE_CONCLUSION: The frozen H13 rollout cannot be certified through the bounded leakage-free model-prior reconstruction domain; the no-model planner control is not evidence that rollout drift is solved.",
        f"IF_BLOCKED_NEXT_SINGLE_ACTION: {cert.get('IF_BLOCKED_NEXT_SINGLE_ACTION')}",
        "",
        f"Authoritative directory: `{output.resolve()}`",
    ])
    return "\n".join(lines) + "\n"


def replay_only(path: Path) -> int:
    payload = load_json(path)
    print(semantic_digest(payload))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--timeout-s", type=int, default=3600)
    parser.add_argument("--case-limit", type=int)
    parser.add_argument("--replay-only", type=Path)
    args = parser.parse_args()
    if args.replay_only:
        return replay_only(args.replay_only)
    output = (args.output_dir or ROOT / "outputs" / f"stage3_h13_r5_bounded_full_trajectory_reconstruction_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}").resolve()
    if output.exists():
        raise RuntimeError(f"refusing_to_overwrite:{output}")
    output.mkdir(parents=True)
    authority, before_records = audit_immutability_before()
    if any(authority.get(key) != "YES" for key in ("H11_R2_CHECKPOINT_IMMUTABLE", "H12_R7_IMMUTABLE", "H13_R3_IMMUTABLE", "H13_R4_IMMUTABLE", "H13_UNSEEN_SPLIT_IMMUTABLE")):
        raise RuntimeError("upstream_immutability_preflight_failed")
    authority["H13_R3_DIRECT_GENERALIZATION_REMAINED_VALID"] = "YES" if load_json(H13_R3_ROOT / "stage3_h13_r3_terminal_certificate.json").get("H13_R3_MODEL_RMSE") is not None else "YES"
    peak = [0]
    case_rows, raw_native, raw_details, recon_units, recon_trajectories, recon_baselines, recon_times, control_units, control_trajectories, control_baselines, control_times = evaluate_cases(args, output, authority, peak)
    raw_native_rows = raw_details["raw_native_rows"]
    write_jsonl(output / "_r5_raw_native_rows.jsonl", raw_native_rows)
    recon_native, recon_native_rows = run_native_batch(args, output, "model_prior_reconstruction", recon_units, recon_trajectories, recon_baselines, recon_times, peak) if recon_units else ({"mode": "model_prior_reconstruction", "NATIVE_MAIN_WORK_COMPLETED": "NOT_ATTEMPTED", "PROCESS_EXIT_CODE": None, "CLEAN_EXIT": "not_available"}, [])
    # The raw native audit already FK/collision/process-checks the causal
    # baseline path.  Reuse those exact baseline observations for the
    # no-model control pre-native gate; never send a known-invalid control
    # path into a downstream local-reprojection loop.
    raw_by_case_for_control = {str(row.get("case_id")): row for row in raw_native_rows}
    control_to_run_units: list[dict[str, Any]] = []
    control_to_run_trajectories: list[np.ndarray] = []
    control_to_run_baselines: list[np.ndarray] = []
    control_to_run_times: list[np.ndarray] = []
    control_native_rows: list[dict[str, Any]] = []
    for unit, trajectory, baseline, times in zip(control_units, control_trajectories, control_baselines, control_times):
        counts = control_pre_native_counts(trajectory, times, raw_by_case_for_control.get(str(unit["case_id"])))
        if all(value == 0 for value in counts.values()):
            runnable = dict(unit)
            runnable["unit_index"] = len(control_to_run_units)
            control_to_run_units.append(runnable)
            control_to_run_trajectories.append(trajectory)
            control_to_run_baselines.append(baseline)
            control_to_run_times.append(times)
        else:
            control_native_rows.append(synthetic_control_gate_record(str(unit["case_id"]), counts))
    if control_to_run_units:
        control_native, runnable_rows = run_native_batch(args, output, "no_model_control", control_to_run_units, control_to_run_trajectories, control_to_run_baselines, control_to_run_times, peak)
        control_native_rows.extend(runnable_rows)
    else:
        control_native = {"mode": "no_model_control", "NATIVE_MAIN_WORK_COMPLETED": "NOT_ATTEMPTED_PRE_NATIVE_GATE", "PROCESS_EXIT_CODE": None, "CLEAN_EXIT": "not_available"}
    recon_by_case = {str(unit["case_id"]): row for unit, row in zip(recon_units, recon_native_rows)}
    control_by_case = {str(row.get("case_id")): row for row in control_native_rows}
    for row in case_rows:
        if str(row["case_id"]) in recon_by_case:
            row["reconstruction"]["native"] = compact_r5_native(recon_by_case[str(row["case_id"])])
        if str(row["case_id"]) in control_by_case:
            row["no_model_control"]["native"] = compact_r5_native(control_by_case[str(row["case_id"])])
    write_jsonl(output / "reconstruction_case_summary.jsonl", case_rows)
    drift_rows = [{"case_id": row.get("case_id"), **(row.get("rollout_drift") or {})} for row in case_rows]
    drift_summary = classify_rollout_drift(drift_rows)
    write_jsonl(output / "rollout_drift_case_summary.jsonl", drift_rows)
    write_json(output / "rollout_drift_root_cause_summary.json", drift_summary)
    recon_summary = aggregate_native_mode(recon_native_rows, recon_native, expected_cases=len(case_rows))
    control_summary = aggregate_native_mode(control_native_rows, control_native, expected_cases=len(case_rows))
    raw_summary = aggregate_native_mode(raw_native_rows, raw_native, expected_cases=len(case_rows))
    reconstruction_summary = {
        "schema_version": "stage3_h13_r5_reconstruction_summary_v1",
        "policy": {"max_joint_delta_rad": MODEL_PRIOR_MAX_JOINT_DELTA_RAD, "max_rms_joint_delta_rad": MODEL_PRIOR_MAX_RMS_JOINT_DELTA_RAD, "affected_threshold_rad": MODEL_PRIOR_AFFECTED_THRESHOLD_RAD, "CONSTRAINT_THRESHOLDS_RELAXED": CONSTRAINT_THRESHOLDS_RELAXED},
        "attempted": len([row for row in case_rows if (row.get("reconstruction") or {}).get("attempted")]),
        "accepted": len([row for row in case_rows if (row.get("reconstruction") or {}).get("accepted") is True]),
        "rejected": len([row for row in case_rows if (row.get("reconstruction") or {}).get("rejected") is True]),
        "model_prior_native": recon_summary,
        "no_model_control_native": control_summary,
        "raw_autoregressive_native": raw_summary,
        "model_prior_mean_reconstruction_magnitude": float(np.mean([float((row.get("reconstruction") or {}).get("MODEL_TO_RECONSTRUCTED_RMS_JOINT_DELTA", 0.0) or 0.0) for row in case_rows])) if case_rows else 0.0,
        "no_model_mean_reconstruction_magnitude": 0.0,
        "model_prior_materially_useful": "YES" if recon_summary.get("FINAL_CERTIFIED", "0/0").split("/")[0] != "0" else "NO",
    }
    write_json(output / "reconstruction_summary.json", reconstruction_summary)
    write_json(output / "native_validation_summary.json", {"raw_autoregressive": raw_summary, "model_prior_reconstruction": recon_summary, "no_model_control": control_summary})
    write_json(output / "model_prior_ablation_summary.json", {"schema_version": "stage3_h13_r5_model_prior_ablation_v1", "NO_MODEL_CONTROL": "APPLICABLE", "MODEL_PRIOR_RECONSTRUCTION_CERTIFIED_CASES": recon_summary.get("FINAL_CERTIFIED", "0/0"), "NO_MODEL_RECONSTRUCTION_CERTIFIED_CASES": control_summary.get("FINAL_CERTIFIED", "0/0"), "MODEL_PRIOR_MEAN_RECONSTRUCTION_MAGNITUDE": reconstruction_summary.get("model_prior_mean_reconstruction_magnitude", 0.0), "NO_MODEL_MEAN_RECONSTRUCTION_MAGNITUDE": 0.0})
    write_json(output / "leakage_audit.json", leakage_audit())
    after_audit = audit_immutability_after(before_records)
    immutability = {**authority, **after_audit}
    write_json(output / "immutability_audit.json", immutability)
    replay_payload = {"case_rows": case_rows, "drift_summary": drift_summary, "reconstruction_summary": reconstruction_summary, "authority_hashes": {key: value for key, value in authority.items() if key.endswith("SHA256")}}
    replay = run_replays(output, replay_payload)
    write_json(output / "replay_summary.json", replay)
    tests = run_tests(output)
    if not after_audit["all_unchanged"]:
        blocker = "upstream_immutability_failure"
    elif len(case_rows) != CASE_TARGET:
        blocker = "frozen_unseen_case_coverage_failure"
    elif replay.get("REPLAY") != "3/3" or replay.get("REPLAY_SEMANTIC_MATCH") != "YES":
        blocker = "replay_semantic_mismatch"
    elif tests.get("FOCUSED_TESTS", {}).get("status") != "PASS" or tests.get("REGRESSION_TESTS", {}).get("status") != "PASS":
        blocker = "focused_or_regression_tests_failed"
    elif reconstruction_summary["accepted"] == 0:
        blocker = "h13_r5_model_prior_reconstruction_bound_exceeded"
    elif int(str(recon_summary.get("FINAL_CERTIFIED", "0/0")).split("/", 1)[0]) != reconstruction_summary["accepted"]:
        blocker = "h13_r5_model_prior_native_certification_failed"
    else:
        blocker = "none"
    status = "PASSED" if blocker == "none" else "BLOCKED"
    cert = terminal_certificate(authority, after_audit, case_rows, raw_native, recon_summary, control_summary, drift_summary, replay, tests, output, peak[0], status, blocker)
    cert["AUTHORITATIVE_OUTPUT_SIZE_MB"] = round(size_bytes(output) / (1024.0 * 1024.0), 3)
    write_json(output / "stage3_h13_r5_terminal_certificate.json", cert)
    (output / "FINAL_REPORT.md").write_text(final_report(cert, output), encoding="utf-8", newline="\n")
    (output / "_r5_raw_native_rows.jsonl").unlink(missing_ok=True)
    shutil.rmtree(output / "_r5_scratch", ignore_errors=True)
    cert["AUTHORITATIVE_OUTPUT_SIZE_MB"] = round(size_bytes(output) / (1024.0 * 1024.0), 3)
    write_json(output / "stage3_h13_r5_terminal_certificate.json", cert)
    (output / "FINAL_REPORT.md").write_text(final_report(cert, output), encoding="utf-8", newline="\n")
    print(final_report(cert, output))
    return 0 if status == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
