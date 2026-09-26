#!/usr/bin/env python3
"""Native, software-only H13 adapter.

The adapter supplies one deterministic SPRAY-ON open-arch primitive per H13
case to the frozen H12-R7 execution function.  It emits compact case records
only; native trajectory arrays and repair event files are deleted immediately
after each case has been summarized.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import rclpy
from moveit.planning import MoveItPy

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import stage3_h7_2_native as base
import stage3_h12_r4_native as r4
from src.stage3_h13 import COLLISION_METHOD
from src.stage3_h12_r7 import ELIGIBLE_SEGMENTS


JOINT_NAMES = ["j1", "j2", "j3", "j4", "j5", "j6"]
SCHEMA_VERSION = "stage3_h13_native_case_v1"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(json_safe(row), ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")


def finite_difference_state(q: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dt = np.diff(t)
    if len(q) < 2 or np.any(dt <= 0.0):
        raise RuntimeError("raw_prediction_time_not_strictly_monotonic")
    velocity = np.vstack([np.zeros((1, q.shape[1])), np.diff(q, axis=0) / dt[:, None]])
    acceleration = np.vstack([np.zeros((1, q.shape[1])), np.diff(velocity, axis=0) / dt[:, None]])
    jerk = np.vstack([np.zeros((1, q.shape[1])), np.diff(acceleration, axis=0) / dt[:, None]])
    return velocity, acceleration, jerk


def violation_counts(dynamic: Mapping[str, Any], jerk: np.ndarray, limits: Mapping[str, Any]) -> dict[str, int]:
    jerk_limits = np.asarray([float(limits[name]["jerk_rad_s3"]) for name in JOINT_NAMES], dtype=float)
    return {
        "position": int(dynamic.get("position_limit_violation_count", 0) or 0),
        "velocity": int(dynamic.get("velocity_limit_violation_count", 0) or 0),
        "acceleration": int(dynamic.get("acceleration_limit_violation_count", 0) or 0),
        "jerk": int(np.count_nonzero(np.abs(jerk) > jerk_limits[None, :] + 1.0e-10)),
    }


def primitive_from_unit(unit: Mapping[str, Any]) -> dict[str, Any]:
    segment_id = int(unit.get("h12_r7_segment_id", -1))
    if str(unit.get("spray_state")) != "SPRAY_ON":
        raise RuntimeError("h13_adapter_non_on_state")
    if segment_id not in ELIGIBLE_SEGMENTS:
        raise RuntimeError(f"h13_adapter_segment_not_h12_r7_eligible:{segment_id}")
    return {
        "segment_id": segment_id,
        "segment_order": int(unit.get("h12_r7_segment_order", 0)),
        "primitive_id": str(unit["primitive_id"]),
        "spray_state": "SPRAY_ON",
        "rows": list(unit["target_rows"]),
    }


def compact_process(summary: Mapping[str, Any] | None) -> dict[str, Any]:
    if not summary:
        return {"status": None, "failed_spray_on_sample_count": None, "max_tcp_position_error_m": None}
    return {
        "status": summary.get("status"),
        "failed_spray_on_sample_count": summary.get("failed_spray_on_sample_count"),
        "max_tcp_position_error_m": summary.get("max_tcp_position_error_m"),
        "max_standoff_error_m": summary.get("max_standoff_error_m"),
        "max_normal_deviation_deg": summary.get("max_normal_deviation_deg"),
        "max_tcp_orientation_error_deg": summary.get("max_tcp_orientation_error_deg"),
    }


def _quaternion_delta_deg(first: Sequence[float], second: Sequence[float]) -> float:
    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    a /= max(float(np.linalg.norm(a)), 1.0e-12)
    b /= max(float(np.linalg.norm(b)), 1.0e-12)
    return float(np.degrees(2.0 * np.arccos(np.clip(abs(float(np.dot(a, b))), -1.0, 1.0))))


def _trajectory_drift_metrics(
    q: np.ndarray,
    baseline_q: np.ndarray,
    model_process_rows: Sequence[Mapping[str, Any]],
    baseline_process_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compact model-vs-causal drift evidence; no target/label is consulted."""

    if q.shape != baseline_q.shape or len(q) < 2:
        return {"status": "NOT_AVAILABLE", "reason": "shape_mismatch"}
    joint_delta = np.linalg.norm(q - baseline_q, axis=1)
    model_tcp = np.asarray([row.get("tcp_position_xyz_m") for row in model_process_rows], dtype=float)
    baseline_tcp = np.asarray([row.get("tcp_position_xyz_m") for row in baseline_process_rows], dtype=float)
    tcp_available = bool(model_tcp.shape == baseline_tcp.shape == (len(q), 3) and np.all(np.isfinite(model_tcp)) and np.all(np.isfinite(baseline_tcp)))
    result: dict[str, Any] = {
        "status": "PASSED" if tcp_available else "PARTIAL",
        "initial_joint_space_displacement_rad": float(joint_delta[0]),
        "median_joint_space_displacement_rad": float(np.median(joint_delta)),
        "maximum_joint_space_displacement_rad": float(np.max(joint_delta)),
        "endpoint_joint_space_displacement_rad": float(joint_delta[-1]),
        "affected_trajectory_fraction": float(np.count_nonzero(joint_delta > 0.01) / len(joint_delta)),
        "first_significant_divergence_index": int(np.flatnonzero(joint_delta > 0.01)[0]) if np.any(joint_delta > 0.01) else None,
        "joint_path_length_model_rad": float(np.sum(np.linalg.norm(np.diff(q, axis=0), axis=1))),
        "joint_path_length_causal_rad": float(np.sum(np.linalg.norm(np.diff(baseline_q, axis=0), axis=1))),
        "orientation_error_max_deg": max((float(row.get("tcp_orientation_error_deg")) for row in model_process_rows if row.get("tcp_orientation_error_deg") is not None), default=None),
        "spray_surface_error_max_m": max((float(row.get("tcp_position_error_m")) for row in model_process_rows if row.get("tcp_position_error_m") is not None), default=None),
    }
    result["trajectory_length_distortion_ratio"] = (
        result["joint_path_length_model_rad"] / result["joint_path_length_causal_rad"]
        if result["joint_path_length_causal_rad"] > 1.0e-12 else None
    )
    if not tcp_available:
        result.update({
            "initial_tcp_space_displacement_m": None,
            "median_tcp_space_displacement_m": None,
            "maximum_tcp_space_displacement_m": None,
            "endpoint_tcp_space_displacement_m": None,
            "cross_track_error_max_m": None,
            "along_track_error_max_m": None,
            "tcp_path_length_model_m": None,
            "tcp_path_length_causal_m": None,
        })
        return result
    tcp_delta_vectors = model_tcp - baseline_tcp
    tcp_delta = np.linalg.norm(tcp_delta_vectors, axis=1)
    tangent = np.diff(baseline_tcp, axis=0)
    tangent = np.vstack((tangent[0], tangent))
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1.0e-12)
    along = np.abs(np.sum(tcp_delta_vectors * tangent, axis=1))
    cross = np.linalg.norm(tcp_delta_vectors - np.sum(tcp_delta_vectors * tangent, axis=1, keepdims=True) * tangent, axis=1)
    result.update({
        "initial_tcp_space_displacement_m": float(tcp_delta[0]),
        "median_tcp_space_displacement_m": float(np.median(tcp_delta)),
        "maximum_tcp_space_displacement_m": float(np.max(tcp_delta)),
        "endpoint_tcp_space_displacement_m": float(tcp_delta[-1]),
        "cross_track_error_max_m": float(np.max(cross)),
        "along_track_error_max_m": float(np.max(along)),
        "tcp_path_length_model_m": float(np.sum(np.linalg.norm(np.diff(model_tcp, axis=0), axis=1))),
        "tcp_path_length_causal_m": float(np.sum(np.linalg.norm(np.diff(baseline_tcp, axis=0), axis=1))),
        "orientation_displacement_max_deg": max((_quaternion_delta_deg(row.get("tcp_orientation_xyzw"), baseline_process_rows[index].get("tcp_orientation_xyzw")) for index, row in enumerate(model_process_rows) if row.get("tcp_orientation_xyzw") is not None and baseline_process_rows[index].get("tcp_orientation_xyzw") is not None), default=None),
    })
    result["tcp_path_length_distortion_ratio"] = (
        result["tcp_path_length_model_m"] / result["tcp_path_length_causal_m"]
        if result["tcp_path_length_causal_m"] > 1.0e-12 else None
    )
    return result


def raw_audit(
    moveit: MoveItPy,
    primitive: Mapping[str, Any],
    q: np.ndarray,
    t: np.ndarray,
    baseline_q: np.ndarray,
    limits: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    velocity, acceleration, jerk = finite_difference_state(q, t)
    dynamic = base.dynamics_validation(q, velocity, acceleration, t, limits, "raw_model_prediction")
    process_rows, process = base.process_validation(moveit, primitive, q, t - t[0], contract)
    collision_rows, collision = base.collision_validate(moveit, t - t[0], q)
    baseline_process_rows, baseline_process = base.process_validation(moveit, primitive, baseline_q, t - t[0], contract)
    baseline_collision_rows, baseline_collision = base.collision_validate(moveit, t - t[0], baseline_q)
    counts = violation_counts(dynamic, jerk, limits)
    return {
        "dynamic": counts,
        "dynamic_detail": dynamic,
        "process": compact_process(process),
        "collision": {
            "status": collision.get("status"),
            "collision_failure_count": int(collision.get("collision_failure_count", 0) or 0),
            "self_collision_failure_count": sum(bool(row.get("self_collision")) for row in collision_rows),
            "environment_collision_failure_count": sum(bool(row.get("environment_collision")) for row in collision_rows),
            "method": COLLISION_METHOD,
        },
        "baseline_process": compact_process(baseline_process),
        "baseline_collision": {
            "status": baseline_collision.get("status"),
            "collision_failure_count": int(baseline_collision.get("collision_failure_count", 0) or 0),
            "self_collision_failure_count": sum(bool(row.get("self_collision")) for row in baseline_collision_rows),
            "environment_collision_failure_count": sum(bool(row.get("environment_collision")) for row in baseline_collision_rows),
            "method": COLLISION_METHOD,
        },
        "raw_model_tcp_error_m": process.get("max_tcp_position_error_m"),
        "baseline_tcp_error_m": baseline_process.get("max_tcp_position_error_m"),
        "drift_metrics": _trajectory_drift_metrics(q, baseline_q, process_rows, baseline_process_rows),
        "raw_model_process_rows_checked": len(process_rows),
        "raw_baseline_process_rows_checked": len(baseline_process_rows),
        "CCD_AVAILABLE": "NO",
        "CLEARANCE_AVAILABLE": None,
    }


def correction_summary(output_dir: Path, unit_index: int) -> dict[str, Any]:
    path = output_dir.parent / "h12_r7_repair_units" / f"{unit_index:05d}.json"
    if not path.is_file():
        return {"values": [], "p95": None, "p99": None}
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = [float(row.get("joint_delta_norm", 0.0)) for row in payload.get("events", [])]
    return {
        "values": values,
        "p95": float(np.percentile(values, 95)) if values else None,
        "p99": float(np.percentile(values, 99)) if values else None,
    }


def reprojection_diagnostics(unit: Mapping[str, Any], repair: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize H12-R7 admission and rejection stages without retaining dumps."""
    eligible = int(repair.get("eligible_count", 0) or 0)
    failures = list(repair.get("repair_failures") or [])
    generated = 0
    rejected_invalid = 0
    rejected_process = 0
    rejected_joint = 0
    rejected_collision = 0
    rejected_tolerance = 0
    not_generated = 0
    for failure in failures:
        attempts = list(failure.get("attempts") or [])
        if not attempts:
            not_generated += 1
        for attempt in attempts:
            if not attempt.get("ik_solved"):
                not_generated += 1
                continue
            generated += 1
            if not attempt.get("joint_valid"):
                rejected_joint += 1
            if not attempt.get("collision_free"):
                rejected_collision += 1
            if attempt.get("tcp_position_error_m") is None or attempt.get("valid") is not True:
                rejected_tolerance += int(attempt.get("tcp_position_error_m") is None or float(attempt.get("tcp_position_error_m", 0.0)) > 0.001)
                rejected_process += int(attempt.get("valid") is not True)
            if attempt.get("valid") is not True:
                rejected_invalid += 1
    repaired = int(repair.get("repaired_count", 0) or 0)
    if eligible == 0:
        stage = "eligibility_gate"
        reason = "segment_id_not_h12_r7_eligible"
    elif repaired == eligible and not failures:
        stage = None
        reason = None
    elif generated == 0:
        stage = "candidate_generation"
        reason = "candidate_not_generated"
    else:
        stage = "candidate_acceptance"
        reason = "candidate_generated_but_rejected"
    return {
        "source_segment_id": int(unit.get("source_segment_id", 0)),
        "h12_r7_segment_id": int(unit.get("h12_r7_segment_id", -1)),
        "h12_r7_segment_order": int(unit.get("h12_r7_segment_order", -1)),
        "spray_state": str(unit.get("spray_state")),
        "failure_stage": stage,
        "failure_reason": reason,
        "eligible_samples": eligible,
        "repaired_samples": repaired,
        "candidate_not_generated": not_generated,
        "candidate_generated": generated,
        "candidate_accepted": repaired,
        "candidate_generated_but_invalid": rejected_invalid,
        "candidate_generated_but_rejected_by_process_gate": rejected_process,
        "candidate_generated_but_joint_limit_violation": rejected_joint,
        "candidate_generated_but_collision_violation": rejected_collision,
        "candidate_generated_but_reprojection_tolerance_failed": rejected_tolerance,
        "candidate_mapping_failed": 0,
    }


def compact_record(
    unit: Mapping[str, Any],
    record: Mapping[str, Any],
    raw: Mapping[str, Any],
    corrections: Mapping[str, Any],
) -> dict[str, Any]:
    ruckig = record.get("ruckig") or {}
    completion = record.get("ruckig_completion_gate") or {}
    dynamic = record.get("post_ruckig_dynamic") or {}
    jerk = record.get("post_ruckig_jerk") or {}
    process = record.get("post_ruckig_process") or {}
    collision = record.get("post_ruckig_collision") or {}
    repair = record.get("h12_r7_reprojection") or {}
    reprojection = reprojection_diagnostics(unit, repair)
    first_blocker = record.get("first_blocker")
    if first_blocker == "h12_r7_local_reprojection_failed" and reprojection.get("failure_reason"):
        first_blocker = reprojection["failure_reason"]
    final_counts = {
        "position": int(dynamic.get("position_limit_violation_count", 0) or 0) if dynamic else None,
        "velocity": int(dynamic.get("velocity_limit_violation_count", 0) or 0) if dynamic else None,
        "acceleration": int(dynamic.get("acceleration_limit_violation_count", 0) or 0) if dynamic else None,
        "jerk": int(jerk.get("violation_count", 0) or 0) if jerk else None,
        "collision": int(collision.get("collision_failure_count", 0) or 0) if collision else None,
        "spray_process": int(process.get("failed_spray_on_sample_count", 0) or 0) if process else None,
    }
    native_error = bool(ruckig.get("native_error")) or int(((ruckig.get("ruckig_result") or {}).get("numeric_result") or 0)) < 0
    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": str(unit["case_id"]),
        "input_semantic_hash": str(unit["input_semantic_hash"]),
        "status": str(record.get("status") or "BLOCKED"),
        "first_blocker": first_blocker,
        "native_backend_executed": True,
        "planning_scene_executed": True,
        "fk_executed": True,
        "dynamics_executed": True,
        "post_ruckig_executed": bool(record.get("post_ruckig_certification_reached")),
        "raw": raw,
        "raw_prediction_violations": {
            "RAW_POSITION_VIOLATIONS": raw["dynamic"]["position"],
            "RAW_VELOCITY_VIOLATIONS": raw["dynamic"]["velocity"],
            "RAW_ACCELERATION_VIOLATIONS": raw["dynamic"]["acceleration"],
            "RAW_JERK_VIOLATIONS": raw["dynamic"]["jerk"],
            "RAW_COLLISION_VIOLATIONS": raw["collision"]["collision_failure_count"],
            "RAW_SPRAY_PROCESS_VIOLATIONS": raw["process"]["failed_spray_on_sample_count"],
        },
        "totg_attempted": bool(record.get("totg_called")),
        "totg_success": bool(record.get("totg_return_value")),
        "ruckig_attempted": bool(record.get("ruckig_attempted")),
        "ruckig_success": bool(completion.get("accepted")),
        "ruckig_native_errors": int(native_error),
        "ruckig_duration_ceiling_hit": int(bool(ruckig.get("duration_ceiling_hit"))),
        "final_counts": {f"FINAL_{key.upper()}_VIOLATIONS": value for key, value in final_counts.items()},
        "repair": {
            "status": repair.get("status"),
            "REPAIRED_SAMPLES": repair.get("repaired_count"),
            "MAX_JOINT_CORRECTION_RAD": repair.get("max_joint_correction_rad"),
            "P95_JOINT_CORRECTION_RAD": corrections.get("p95"),
            "P99_JOINT_CORRECTION_RAD": corrections.get("p99"),
            "repair_failures": repair.get("repair_failure_count"),
            "immutable_samples_unchanged": repair.get("immutable_samples_unchanged"),
        },
        "reprojection": {
            "source_segment_id": int(unit.get("source_segment_id", 0)),
            "h12_r7_segment_id": int(unit.get("h12_r7_segment_id", -1)),
            "h12_r7_segment_order": int(unit.get("h12_r7_segment_order", -1)),
            "spray_state": str(unit.get("spray_state")),
            **reprojection,
            "candidate_generated_yes": bool(reprojection.get("candidate_generated", 0) or reprojection.get("repaired_samples", 0)),
            "candidate_accepted_yes": bool(reprojection.get("candidate_accepted", 0)),
        },
        "collision_method": COLLISION_METHOD,
        "CCD_AVAILABLE": "NO",
        "CLEARANCE_AVAILABLE": None,
    }


def run(args: argparse.Namespace) -> int:
    bundle = np.load(Path(args.input_npz), allow_pickle=False)
    positions = np.asarray(bundle["positions_rad"], dtype=np.float64)
    baseline_positions = np.asarray(bundle["baseline_positions_rad"], dtype=np.float64)
    times = np.asarray(bundle["times_s"], dtype=np.float64)
    lengths = np.asarray(bundle["lengths"], dtype=np.int64)
    units = load_jsonl(Path(args.unit_manifest))
    if len(units) != len(lengths) or positions.shape != baseline_positions.shape:
        raise RuntimeError("h13_input_manifest_array_mismatch")
    contract = load_json(Path(args.process_contract))
    totg = load_json(Path(args.totg_parameters))
    moveit = MoveItPy(node_name="stage3_h13_native")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    limit_audit = base.runtime_limits(moveit, Path(args.runtime_limits))
    if limit_audit.get("status") != "PASSED":
        raise RuntimeError("runtime_joint_limit_audit_failed")
    base.apply_fixture(moveit, Path(args.fixture_mesh))
    compact: list[dict[str, Any]] = []
    try:
        for unit in units:
            index = int(unit["unit_index"])
            length = int(lengths[index])
            q = positions[index, :length]
            baseline_q = baseline_positions[index, :length]
            t = times[index, :length]
            primitive = primitive_from_unit(unit)
            raw = raw_audit(moveit, primitive, q, t, baseline_q, limit_audit["runtime_bounds"], contract)
            unit_dir = output / f"unit_{index:05d}"
            unit_dir.mkdir(parents=True, exist_ok=True)
            item = {
                "unit_index": index,
                "unit_id": unit["unit_id"],
                "primitive_id": unit["primitive_id"],
                "segment_id": int(unit["h12_r7_segment_id"]),
                "segment_order": int(unit["h12_r7_segment_order"]),
                "segment_key": f"{unit['case_id']}|ON|{int(unit['h12_r7_segment_id'])}|{int(unit['h12_r7_segment_order'])}",
                "trajectory_family_id": unit["trajectory_family_id"],
                "window_ids": [],
                "window_count": int(unit.get("window_count", 0)),
                "spray_state": "SPRAY_ON",
            }
            try:
                record = r4.execute_unit(moveit, primitive, item, q, t, limit_audit, contract, totg, unit_dir)
                corrections = correction_summary(unit_dir, index)
                compact.append(compact_record(unit, record, raw, corrections))
            except Exception as exc:
                compact.append({
                    "schema_version": SCHEMA_VERSION,
                    "case_id": str(unit["case_id"]),
                    "input_semantic_hash": str(unit["input_semantic_hash"]),
                    "status": "BLOCKED",
                    "first_blocker": f"native_exception:{type(exc).__name__}:{exc}",
                    "native_backend_executed": True,
                    "planning_scene_executed": True,
                    "fk_executed": True,
                    "dynamics_executed": True,
                    "post_ruckig_executed": False,
                    "raw": raw,
                    "raw_prediction_violations": {
                        "RAW_POSITION_VIOLATIONS": raw["dynamic"]["position"],
                        "RAW_VELOCITY_VIOLATIONS": raw["dynamic"]["velocity"],
                        "RAW_ACCELERATION_VIOLATIONS": raw["dynamic"]["acceleration"],
                        "RAW_JERK_VIOLATIONS": raw["dynamic"]["jerk"],
                        "RAW_COLLISION_VIOLATIONS": raw["collision"]["collision_failure_count"],
                        "RAW_SPRAY_PROCESS_VIOLATIONS": raw["process"]["failed_spray_on_sample_count"],
                    },
                    "totg_attempted": False,
                    "totg_success": False,
                    "ruckig_attempted": False,
                    "ruckig_success": False,
                    "ruckig_native_errors": 0,
                    "ruckig_duration_ceiling_hit": 0,
                    "final_counts": {key: None for key in ("FINAL_POSITION_VIOLATIONS", "FINAL_VELOCITY_VIOLATIONS", "FINAL_ACCELERATION_VIOLATIONS", "FINAL_JERK_VIOLATIONS", "FINAL_COLLISION_VIOLATIONS", "FINAL_SPRAY_PROCESS_VIOLATIONS")},
                    "repair": {"status": None, "REPAIRED_SAMPLES": None, "MAX_JOINT_CORRECTION_RAD": None, "P95_JOINT_CORRECTION_RAD": None, "P99_JOINT_CORRECTION_RAD": None, "repair_failures": None, "immutable_samples_unchanged": None},
                    "reprojection": {**reprojection_diagnostics(unit, {"eligible_count": 0}), "failure_stage": "adapter_input_invalid", "failure_reason": f"native_exception:{type(exc).__name__}:{exc}"},
                    "collision_method": COLLISION_METHOD,
                    "CCD_AVAILABLE": "NO",
                    "CLEARANCE_AVAILABLE": None,
                })
            finally:
                if unit_dir.is_dir():
                    shutil.rmtree(unit_dir)
                repair_dir = output / "h12_r7_repair_units"
                if repair_dir.is_dir():
                    shutil.rmtree(repair_dir)
        write_jsonl(Path(args.output_jsonl), compact)
    finally:
        moveit.shutdown()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("input_npz", "unit_manifest", "output_jsonl", "output_dir", "process_contract", "fixture_mesh", "runtime_limits", "totg_parameters"):
        parser.add_argument(f"--{name.replace('_', '-')}", required=True)
    args, _ = parser.parse_known_args()
    rclpy.init()
    try:
        return run(args)
    except Exception as exc:
        Path(args.output_jsonl).write_text(json.dumps({"schema_version": SCHEMA_VERSION, "status": "BLOCKED", "first_blocker": f"native_exception:{type(exc).__name__}:{exc}"}, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        return 2
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
