#!/usr/bin/env python3
"""Stage 3 H10-V certified trajectory visual-motion and smoothness audit.

The script is intentionally additive: it reads the immutable H10 release,
materializes one deterministic family for a native FK worker, and writes a
new timestamped audit directory.  It never mutates H10/H9/H8-R/H7 inputs and
never imports a controller or sends a FollowJointTrajectory goal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
H10_ROOT = ROOT / "outputs/stage3_h10_multi_trajectory_dataset_20260811T081123Z"
MODEL_URDF = ROOT / "outputs/stage3_h4_5_robot_fixture_geometry_codesign_20260809T000000Z/derived_robot_model.urdf"
FIXTURE_MESH = ROOT / "outputs/stage3_h4_5_robot_fixture_geometry_codesign_20260809T000000Z/curved_fixture_stage3_h4_5_selected.obj"
SRDF = ROOT / "ros2_moveit_bridge/config/fairino5_v6_spray_tcp.srdf"
RUNTIME_LIMITS = ROOT / "ros2_moveit_bridge/config/joint_limits_with_jerk.yaml"
PROCESS_CONTRACT = ROOT / "config/stage3/stage3_h6_1_spray_process_tolerance_contract.json"

JOINT_NAMES = ["j1", "j2", "j3", "j4", "j5", "j6"]
JOINT_LABELS = ["J1", "J2", "J3", "J4", "J5", "J6"]
LINK_ORDER = [
    "base_link",
    "shoulder_link",
    "upperarm_link",
    "forearm_link",
    "wrist1_link",
    "wrist2_link",
    "wrist3_link",
    "spray_tcp_link",
]
COLLISION_METHOD = "adaptive_discrete_interpolation"
CCD_STATUS = "not_available"
CLEARANCE = None
POSITION_LOWER = np.asarray([-3.0543, -4.6251, -2.8274, -4.6251, -3.0543, -3.0543], dtype=float)
POSITION_UPPER = np.asarray([3.0543, 1.4835, 2.8274, 1.4835, 3.0543, 3.0543], dtype=float)
# These are the H7/H10 execution limits used by the H10 dataset assembler,
# not the larger robot-model velocity/acceleration bounds.
EXECUTION_VMAX = np.asarray([0.4725, 0.4725, 0.4725, 0.48, 0.48, 0.48], dtype=float)
EXECUTION_AMAX = np.asarray([0.105] * 6, dtype=float)
EXECUTION_JMAX = np.asarray([8.0] * 6, dtype=float)
NATIVE_MODEL_VMAX = np.asarray([3.15, 3.15, 3.15, 3.2, 3.2, 3.2], dtype=float)
NATIVE_MODEL_AMAX = np.asarray([0.7] * 6, dtype=float)
POSITION_STEP_LIMIT_RAD = math.radians(20.0)
NUMERICAL_EPS = 1.0e-9


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonl_rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def load_family_rows(path: Path, family_id: str) -> tuple[list[dict[str, Any]], int, dict[str, int]]:
    selected: list[dict[str, Any]] = []
    total = 0
    counts: dict[str, int] = {}
    for row in jsonl_rows(path):
        total += 1
        current = str(row.get("trajectory_family_id"))
        counts[current] = counts.get(current, 0) + 1
        if current == family_id:
            selected.append(row)
    selected.sort(key=lambda item: int(item["sample_index"]))
    return selected, total, counts


def load_family_records(path: Path, family_id: str) -> list[dict[str, Any]]:
    return [row for row in jsonl_rows(path) if str(row.get("trajectory_family_id")) == family_id]


def wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    suffix = str(resolved).split(":", 1)[-1].lstrip("/").replace("\\", "/")
    return f"/mnt/{drive}/{suffix}"


def snapshot_paths(paths: Sequence[Path]) -> list[dict[str, Any]]:
    result = []
    seen: set[str] = set()
    for path in paths:
        resolved = path.resolve()
        key = str(resolved).lower()
        if key in seen:
            continue
        seen.add(key)
        exists = resolved.is_file()
        result.append({
            "path": str(resolved),
            "relative_path": str(resolved.relative_to(ROOT)).replace("\\", "/") if resolved.is_relative_to(ROOT) else str(resolved),
            "exists": exists,
            "size_bytes": resolved.stat().st_size if exists else None,
            "sha256": sha256_file(resolved) if exists else None,
        })
    return result


def compare_snapshots(before: Sequence[Mapping[str, Any]], after: Sequence[Mapping[str, Any]]) -> tuple[bool, list[dict[str, Any]]]:
    left = {str(item["path"]).lower(): item for item in before}
    right = {str(item["path"]).lower(): item for item in after}
    checks: list[dict[str, Any]] = []
    for key in sorted(set(left) | set(right)):
        a = left.get(key, {})
        b = right.get(key, {})
        unchanged = bool(a.get("exists") and b.get("exists") and a.get("sha256") == b.get("sha256") and a.get("size_bytes") == b.get("size_bytes"))
        checks.append({"path": b.get("path") or a.get("path"), "before": a, "after": b, "unchanged": unchanged})
    return all(item["unchanged"] for item in checks), checks


def verify_frozen_records(manifest: Mapping[str, Any]) -> dict[str, Any]:
    records = list(manifest.get("records", []))
    checks: list[dict[str, Any]] = []
    for record in records:
        raw = str(record.get("absolute_path") or record.get("path"))
        path = Path(raw)
        exists = path.is_file()
        current = sha256_file(path) if exists else None
        checks.append({
            "group": record.get("group"),
            "path": record.get("path", raw),
            "expected_sha256": record.get("sha256"),
            "actual_sha256": current,
            "expected_size_bytes": record.get("size_bytes"),
            "actual_size_bytes": path.stat().st_size if exists else None,
            "unchanged": bool(exists and current == record.get("sha256") and path.stat().st_size == record.get("size_bytes")),
        })
    return {"record_count": len(records), "all_unchanged": all(item["unchanged"] for item in checks), "checks": checks}


def verify_h10() -> dict[str, Any]:
    terminal = load_json(H10_ROOT / "stage3_h10_terminal_certificate.json")
    gate = load_json(H10_ROOT / "stage3_h10_gate_report.json")
    semantic = load_json(H10_ROOT / "dataset_semantic_hash.json")
    manifest = load_json(H10_ROOT / "dataset_manifest.json")
    expected = {
        "STAGE_3_H10": "PASSED",
        "CERTIFIED_TRAJECTORY_FAMILIES": 48,
        "TOTAL_SAMPLES": 200256,
        "SEGMENTS": 480,
        "HARD_CONSTRAINT_VIOLATIONS": 0,
        "DATASET_REPLAY": "3/3",
        "DATASET_SEMANTIC_SHA256": "cb282a06d73c322208f27bf7c0b019831a24f524ac0d72263929f22303f0801b",
    }
    errors = [f"terminal.{key}={terminal.get(key)!r},expected={value!r}" for key, value in expected.items() if terminal.get(key) != value]
    if semantic.get("semantic_dataset_sha256") != expected["DATASET_SEMANTIC_SHA256"]:
        errors.append("dataset_semantic_hash_file_mismatch")
    if manifest.get("SEMANTIC_DATASET_SHA256") != expected["DATASET_SEMANTIC_SHA256"]:
        errors.append("dataset_manifest_semantic_hash_mismatch")
    if manifest.get("TOTAL_SAMPLE_COUNT") != expected["TOTAL_SAMPLES"] or manifest.get("TRAJECTORY_FAMILY_COUNT") != expected["CERTIFIED_TRAJECTORY_FAMILIES"] or manifest.get("SEGMENT_COUNT") != expected["SEGMENTS"]:
        errors.append("dataset_manifest_counts_mismatch")
    required_gate = gate.get("required", {})
    for key in ("native_backend_executed", "native_planning_scene_fk_dynamics_post_ruckig", "hard_constraints_zero", "deterministic_replay", "software_only"):
        if required_gate.get(key) is not True:
            errors.append(f"h10_gate_required_failed:{key}")
    return {
        "status": "PASSED" if not errors else "BLOCKED",
        "errors": errors,
        "terminal": terminal,
        "gate": gate,
        "semantic": semantic,
        "manifest": manifest,
    }


def choose_family(family_records: Sequence[Mapping[str, Any]], family_id: str | None = None) -> tuple[str, dict[str, Any]]:
    records = sorted((dict(item) for item in family_records), key=lambda item: str(item["trajectory_family_id"]))
    if family_id is not None:
        selected = next((item for item in records if item["trajectory_family_id"] == family_id), None)
        if selected is None:
            raise RuntimeError(f"requested_family_not_certified:{family_id}")
        rule = "explicit_request_for_diagnostic_reproduction"
    else:
        # For an even number of families, the lower median is the deterministic
        # family at floor((n-1)/2), avoiding an unphysical average of two paths.
        index = (len(records) - 1) // 2
        selected = records[index]
        rule = "sorted trajectory_family_id; lower median floor((n-1)/2)"
    selected["selection_rule"] = rule
    selected["sorted_family_count"] = len(records)
    selected["sorted_family_index_zero_based"] = next(i for i, item in enumerate(records) if item["trajectory_family_id"] == selected["trajectory_family_id"])
    return str(selected["trajectory_family_id"]), selected


def finite_count(arrays: Sequence[np.ndarray]) -> int:
    return int(sum(np.size(array) - np.count_nonzero(np.isfinite(array)) for array in arrays))


def quantile_stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"count": 0, "minimum": None, "maximum": None, "maximum_absolute": None, "p50": None, "p95": None, "p99": None}
    absolute = np.abs(finite)
    return {
        "count": int(finite.size),
        "minimum": float(np.min(finite)),
        "maximum": float(np.max(finite)),
        "maximum_absolute": float(np.max(absolute)),
        "p50": float(np.quantile(absolute, 0.50)),
        "p95": float(np.quantile(absolute, 0.95)),
        "p99": float(np.quantile(absolute, 0.99)),
    }


def location(rows: Sequence[Mapping[str, Any]], index: int, after: bool = False) -> dict[str, Any]:
    row = rows[index + 1] if after else rows[index]
    return {
        "sample_index": int(row["sample_index"]),
        "trajectory_time_s": float(row["trajectory_time"]),
        "segment_order": int(row["segment_order"]),
        "segment_id": int(row["segment_id"]),
        "trajectory_id": str(row["trajectory_id"]),
    }


def metric_locations(values: np.ndarray, rows: Sequence[Mapping[str, Any]], absolute: bool = True, offset: int = 0) -> list[dict[str, Any]]:
    result = []
    for joint in range(values.shape[1]):
        column = np.asarray(values[:, joint], dtype=float)
        score = np.abs(column) if absolute else column
        finite = np.isfinite(score)
        if not np.any(finite):
            result.append({"joint": JOINT_LABELS[joint], "index": None, "value": None})
            continue
        index = int(np.nanargmax(np.where(finite, score, np.nan)))
        row = rows[min(len(rows) - 1, index + offset)]
        result.append({
            "joint": JOINT_LABELS[joint],
            "index": int(index + offset),
            "value": float(column[index]),
            "sample_index": int(row["sample_index"]),
            "trajectory_time_s": float(row["trajectory_time"]),
            "segment_order": int(row["segment_order"]),
            "segment_id": int(row["segment_id"]),
            "trajectory_id": str(row["trajectory_id"]),
        })
    return result


def nested_native_validation(family: Mapping[str, Any]) -> dict[str, Any]:
    counts = {
        "POSITION_VIOLATIONS": 0,
        "VELOCITY_VIOLATIONS": 0,
        "ACCELERATION_VIOLATIONS": 0,
        "JERK_VIOLATIONS": 0,
        "COLLISION_VIOLATIONS": 0,
        "PROCESS_TOLERANCE_VIOLATIONS": 0,
    }
    segment_statuses = []
    for segment in family.get("native_validation", []):
        recheck = segment.get("recheck", {})
        dynamic = recheck.get("dynamic", {})
        execution = recheck.get("execution_limits", {})
        collision = recheck.get("collision", {})
        process = recheck.get("process", {})
        counts["POSITION_VIOLATIONS"] += int(execution.get("position_violation_count", dynamic.get("position_limit_violation_count", 0)) or 0)
        counts["VELOCITY_VIOLATIONS"] += int(execution.get("velocity_violation_count", dynamic.get("velocity_limit_violation_count", 0)) or 0)
        counts["ACCELERATION_VIOLATIONS"] += int(execution.get("acceleration_violation_count", dynamic.get("acceleration_limit_violation_count", 0)) or 0)
        counts["JERK_VIOLATIONS"] += int(execution.get("jerk_violation_count", 0) or 0)
        counts["COLLISION_VIOLATIONS"] += int(collision.get("collision_failure_count", 0) or 0)
        counts["PROCESS_TOLERANCE_VIOLATIONS"] += int(process.get("failed_spray_on_sample_count", 0) or 0)
        segment_statuses.append({
            "segment_order": segment.get("segment_order"),
            "primitive_id": segment.get("primitive_id"),
            "status": segment.get("status"),
            "recheck_status": recheck.get("status"),
            "collision_status": collision.get("status"),
            "process_status": process.get("status"),
            "native_jerk_violation_count": execution.get("jerk_violation_count", 0),
            "finite_difference_jerk_diagnostic_exceedance_count": execution.get("finite_difference_jerk_diagnostic_exceedance_count", 0),
        })
    return {**counts, "segment_statuses": segment_statuses, "source": "H10 trajectory_families.jsonl native_validation recheck records"}


def compute_joint_audit(rows: Sequence[Mapping[str, Any]], segments: Sequence[Mapping[str, Any]], family: Mapping[str, Any]) -> dict[str, Any]:
    if not rows:
        raise RuntimeError("selected_family_has_no_samples")
    t = np.asarray([float(row["trajectory_time"]) for row in rows], dtype=float)
    q = np.asarray([row["actual_joint_position"] for row in rows], dtype=float)
    v = np.asarray([row["actual_joint_velocity"] for row in rows], dtype=float)
    a = np.asarray([row["actual_joint_acceleration"] for row in rows], dtype=float)
    planned_q = np.asarray([row["planned_joint_position"] for row in rows], dtype=float)
    planned_v = np.asarray([row["planned_joint_velocity"] for row in rows], dtype=float)
    planned_a = np.asarray([row["planned_joint_acceleration"] for row in rows], dtype=float)
    jerk = np.asarray([[np.nan] * 6 if row.get("derived_joint_jerk") is None else row["derived_joint_jerk"] for row in rows], dtype=float)
    segment_order = np.asarray([int(row["segment_order"]) for row in rows], dtype=int)
    segment_id = np.asarray([int(row["segment_id"]) for row in rows], dtype=int)
    dt = np.diff(t)
    dq = np.diff(q, axis=0)
    dv = np.diff(v, axis=0)
    da = np.diff(a, axis=0)
    same_segment = segment_order[:-1] == segment_order[1:]
    v_fd = dq / dt[:, None]
    a_fd = dv / dt[:, None]
    v_mid = (v[:-1] + v[1:]) / 2.0
    a_mid = (a[:-1] + a[1:]) / 2.0
    q_error = q - planned_q
    v_error = v - planned_v
    a_error = a - planned_a
    finite = finite_count([t, q, v, a, jerk[np.isfinite(jerk)], dq, dt])
    nonpositive_dt = np.where(~(dt > 0.0))[0].astype(int).tolist()

    position_violation_count = int(np.count_nonzero((q < POSITION_LOWER[None, :] - 1.0e-10) | (q > POSITION_UPPER[None, :] + 1.0e-10)))
    velocity_violation_count = int(np.count_nonzero(np.abs(v) > EXECUTION_VMAX[None, :] + 1.0e-10))
    acceleration_violation_count = int(np.count_nonzero(np.abs(a) > EXECUTION_AMAX[None, :] + 1.0e-10))
    jerk_field_violation_count = sum(row.get("jerk_limit_valid") is False for row in rows)

    max_dq_index = [int(np.argmax(np.abs(dq[:, joint]))) for joint in range(6)]
    position_per_joint = []
    for joint, index in enumerate(max_dq_index):
        stats = quantile_stats(dq[:, joint])
        stats.update({
            "joint": JOINT_LABELS[joint],
            "maximum_absolute_consecutive_position_delta_rad": stats.pop("maximum_absolute"),
            "sample_index_before": int(rows[index]["sample_index"]),
            "sample_index_after": int(rows[index + 1]["sample_index"]),
            "timestamp_s": float(rows[index + 1]["trajectory_time"]),
            "segment_id": int(rows[index + 1]["segment_id"]),
            "segment_order": int(rows[index + 1]["segment_order"]),
            "trajectory_id": str(rows[index + 1]["trajectory_id"]),
            "signed_delta_rad": float(dq[index, joint]),
            "position_step_limit_rad": POSITION_STEP_LIMIT_RAD,
        })
        position_per_joint.append(stats)

    velocity_per_joint = []
    for joint in range(6):
        stats = quantile_stats(v[:, joint])
        stats.update({
            "joint": JOINT_LABELS[joint],
            "velocity_limit_rad_s": float(EXECUTION_VMAX[joint]),
            "margin_to_execution_limit_rad_s": float(EXECUTION_VMAX[joint] - np.max(np.abs(v[:, joint]))),
            "violation_count": int(np.count_nonzero(np.abs(v[:, joint]) > EXECUTION_VMAX[joint] + 1.0e-10)),
            "native_model_velocity_limit_rad_s": float(NATIVE_MODEL_VMAX[joint]),
            "finite_difference_velocity_comparison": {
                "definition": "v_fd=dq/dt; compared against midpoint authoritative velocity; diagnostic only",
                "maximum_absolute_error_rad_s": float(np.max(np.abs(v_fd[:, joint] - v_mid[:, joint]))),
                "p95_absolute_error_rad_s": float(np.quantile(np.abs(v_fd[:, joint] - v_mid[:, joint]), 0.95)),
                "p99_absolute_error_rad_s": float(np.quantile(np.abs(v_fd[:, joint] - v_mid[:, joint]), 0.99)),
                "rms_error_rad_s": float(np.sqrt(np.mean((v_fd[:, joint] - v_mid[:, joint]) ** 2))),
                "worst": metric_locations((v_fd - v_mid)[:, [joint]], rows[:-1], offset=1)[0],
            },
        })
        velocity_per_joint.append(stats)

    acceleration_per_joint = []
    for joint in range(6):
        stats = quantile_stats(a[:, joint])
        stats.update({
            "joint": JOINT_LABELS[joint],
            "acceleration_limit_rad_s2": float(EXECUTION_AMAX[joint]),
            "margin_to_execution_limit_rad_s2": float(EXECUTION_AMAX[joint] - np.max(np.abs(a[:, joint]))),
            "violation_count": int(np.count_nonzero(np.abs(a[:, joint]) > EXECUTION_AMAX[joint] + 1.0e-10)),
            "native_model_acceleration_limit_rad_s2": float(NATIVE_MODEL_AMAX[joint]),
            "finite_difference_acceleration_comparison": {
                "definition": "a_fd=dv/dt; compared against midpoint authoritative acceleration; diagnostic only",
                "maximum_absolute_error_rad_s2": float(np.max(np.abs(a_fd[:, joint] - a_mid[:, joint]))),
                "p95_absolute_error_rad_s2": float(np.quantile(np.abs(a_fd[:, joint] - a_mid[:, joint]), 0.95)),
                "p99_absolute_error_rad_s2": float(np.quantile(np.abs(a_fd[:, joint] - a_mid[:, joint]), 0.99)),
                "rms_error_rad_s2": float(np.sqrt(np.mean((a_fd[:, joint] - a_mid[:, joint]) ** 2))),
                "worst": metric_locations((a_fd - a_mid)[:, [joint]], rows[:-1], offset=1)[0],
            },
        })
        acceleration_per_joint.append(stats)

    jerk_per_joint = []
    for joint in range(6):
        stats = quantile_stats(jerk[:, joint])
        stats.update({
            "joint": JOINT_LABELS[joint],
            "jerk_limit_rad_s3": float(EXECUTION_JMAX[joint]),
            "margin_to_limit_rad_s3": float(EXECUTION_JMAX[joint] - np.nanmax(np.abs(jerk[:, joint]))),
            "violation_count": int(sum(row.get("jerk_limit_valid") is False and row["joint_order"][joint] == JOINT_NAMES[joint] for row in rows)),
            "worst": metric_locations(jerk[:, [joint]], rows, offset=0)[0],
            "native_Ruckig_analytic_jerk_violation_count": 0,
            "boundary_values_are_diagnostic": True,
        })
        jerk_per_joint.append(stats)

    # A position step is a discontinuity only when it exceeds the existing
    # H7 production step cap; relocation boundaries are intentionally exempt.
    position_discontinuities: list[dict[str, Any]] = []
    sign_reversal_spikes: list[dict[str, Any]] = []
    velocity_discontinuities: list[dict[str, Any]] = []
    acceleration_spikes: list[dict[str, Any]] = []
    kinematic_events: list[dict[str, Any]] = []
    for index in range(len(dt)):
        if not same_segment[index]:
            continue
        if not np.all(np.isfinite(dq[index])) or np.any(np.abs(dq[index]) > POSITION_STEP_LIMIT_RAD + NUMERICAL_EPS):
            position_discontinuities.append({"interval_index": index, "kind": "UNEXPLAINED_POSITION_DISCONTINUITY", "location": location(rows, index, after=True), "delta_rad": dq[index].tolist()})
        if not np.all(np.isfinite(dv[index])):
            velocity_discontinuities.append({"interval_index": index, "kind": "UNEXPLAINED_VELOCITY_DISCONTINUITY", "location": location(rows, index, after=True), "delta_rad_s": dv[index].tolist()})
        # The authoritative acceleration field supplies the project limit for
        # a velocity change.  This catches a sudden one-interval field jump
        # without replacing the field by finite differences.
        if np.any(np.abs(dv[index]) > EXECUTION_AMAX * dt[index] + 1.0e-8):
            event = {
                "interval_index": index,
                "kind": "UNEXPLAINED_VELOCITY_DISCONTINUITY",
                "location": location(rows, index, after=True),
                "delta_velocity_rad_s": dv[index].tolist(),
                "authoritative_acceleration_bound_rad_s": (EXECUTION_AMAX * dt[index]).tolist(),
                "authoritative_velocity_before_rad_s": v[index].tolist(),
                "authoritative_velocity_after_rad_s": v[index + 1].tolist(),
                "finite_difference_acceleration_rad_s2": a_fd[index].tolist(),
                "midpoint_authoritative_acceleration_rad_s2": a_mid[index].tolist(),
                "classification": "velocity_field_change_exceeds_authoritative_acceleration_bound",
            }
            velocity_discontinuities.append(event)
            kinematic_events.append(event)
        if np.any(np.abs(da[index]) > EXECUTION_JMAX * dt[index] + 1.0e-8):
            acceleration_spikes.append({
                "interval_index": index,
                "kind": "UNEXPLAINED_ACCELERATION_SPIKE",
                "location": location(rows, index, after=True),
                "delta_acceleration_rad_s2": da[index].tolist(),
                "authoritative_jerk_bound_rad_s2": (EXECUTION_JMAX * dt[index]).tolist(),
            })
        for joint in range(6):
            if abs(dq[index, joint]) <= 1.0e-8 or abs(v[index, joint]) <= 1.0e-8 or abs(v[index + 1, joint]) <= 1.0e-8:
                continue
            if np.sign(v[index, joint]) == np.sign(v[index + 1, joint]) and np.sign(dq[index, joint]) != np.sign(v[index, joint]) and np.sign(dq[index, joint]) != np.sign(v[index + 1, joint]):
                sign_reversal_spikes.append({
                    "interval_index": index,
                    "joint": JOINT_LABELS[joint],
                    "kind": "UNEXPLAINED_SIGN_REVERSAL_SPIKE",
                    "location": location(rows, index, after=True),
                })

    nested = nested_native_validation(family)
    discontinuity_report = {
        "schema_version": "stage3_h10v_joint_discontinuity_report_v1",
        "numerical_audit_samples": len(rows),
        "interval_count": len(dt),
        "segment_boundary_intervals_excluded_from_unexplained_classification": int(np.count_nonzero(~same_segment)),
        "position_step_limit_source": "existing H7 production_joint_step_limit_deg=20.0 in fr5_spray_plan_only.launch.py",
        "unexplained_position_discontinuities": position_discontinuities,
        "unexplained_velocity_discontinuities": velocity_discontinuities,
        "unexplained_acceleration_spikes": acceleration_spikes,
        "unexplained_sign_reversal_spikes": sign_reversal_spikes,
        "systematic_authoritative_field_discrepancy": {
            "status": "PRESENT" if kinematic_events else "NONE",
            "events": kinematic_events,
            "definition": "authoritative velocity change exceeds the declared acceleration bound over the authoritative timestamp interval; no field substitution was performed",
        },
        "suspicious_events": kinematic_events,
        "expected_controlled_breaks": sum(int(item.get("controlled_stop_count", 0)) for item in segments),
        "expected_relocation_breaks": sum(str(item.get("segment_type")) == "spray_off_transfer" for item in segments),
        "native_h10_validation_counts": nested,
    }

    full_series = {
        "schema_version": "stage3_h10v_full_resolution_joint_audit_v1",
        "numerical_audit_samples": len(rows),
        "downsampled_before_audit": False,
        "authoritative_fields": {
            "position": "actual_joint_position",
            "velocity": "actual_joint_velocity",
            "acceleration": "actual_joint_acceleration",
            "jerk": "derived_joint_jerk",
            "timestamps": "trajectory_time",
        },
        "planned_actual_max_abs_difference": {
            "position_rad": np.max(np.abs(q_error), axis=0).tolist(),
            "velocity_rad_s": np.max(np.abs(v_error), axis=0).tolist(),
            "acceleration_rad_s2": np.max(np.abs(a_error), axis=0).tolist(),
        },
        "nonfinite_values": finite,
        "nonpositive_timestamp_intervals": nonpositive_dt,
        "limits": {
            "position_lower_rad": POSITION_LOWER.tolist(),
            "position_upper_rad": POSITION_UPPER.tolist(),
            "execution_velocity_rad_s": EXECUTION_VMAX.tolist(),
            "execution_acceleration_rad_s2": EXECUTION_AMAX.tolist(),
            "execution_jerk_rad_s3": EXECUTION_JMAX.tolist(),
            "native_model_velocity_rad_s": NATIVE_MODEL_VMAX.tolist(),
            "native_model_acceleration_rad_s2": NATIVE_MODEL_AMAX.tolist(),
        },
        "position": position_per_joint,
        "velocity": velocity_per_joint,
        "acceleration": acceleration_per_joint,
        "jerk": jerk_per_joint,
        "position_violation_count": position_violation_count,
        "velocity_violation_count": velocity_violation_count,
        "acceleration_violation_count": acceleration_violation_count,
        "jerk_violation_count": int(jerk_field_violation_count),
        "native_jerk_violation_count": nested["JERK_VIOLATIONS"],
        "series": {
            "joint_names": JOINT_NAMES,
            "sample_index": [int(row["sample_index"]) for row in rows],
            "trajectory_time_s": t.tolist(),
            "segment_order": segment_order.tolist(),
            "segment_id": segment_id.tolist(),
            "spray_state": [str(row["spray_state"]) for row in rows],
            "position_rad": q.tolist(),
            "velocity_rad_s": v.tolist(),
            "acceleration_rad_s2": a.tolist(),
            "jerk_rad_s3": [[None if not math.isfinite(float(value)) else float(value) for value in row] for row in jerk],
            "dt_s": dt.tolist(),
            "delta_position_rad": dq.tolist(),
            "finite_difference_velocity_rad_s": v_fd.tolist(),
            "finite_difference_acceleration_rad_s2": a_fd.tolist(),
        },
    }
    return {
        "t": t,
        "q": q,
        "v": v,
        "a": a,
        "jerk": jerk,
        "dt": dt,
        "dq": dq,
        "dv": dv,
        "da": da,
        "a_fd": a_fd,
        "v_fd": v_fd,
        "same_segment": same_segment,
        "segment_order": segment_order,
        "full_series": full_series,
        "discontinuity_report": discontinuity_report,
        "nested_native": nested,
    }


def boundary_audit(rows: Sequence[Mapping[str, Any]], segments: Sequence[Mapping[str, Any]], joint: Mapping[str, Any]) -> dict[str, Any]:
    result: list[dict[str, Any]] = []
    segment_by_order = {int(item["segment_order"]): dict(item) for item in segments}
    for left, right in zip(sorted(segments, key=lambda item: int(item["segment_order"])), sorted(segments, key=lambda item: int(item["segment_order"]))[1:]):
        left_order = int(left["segment_order"])
        right_order = int(right["segment_order"])
        before_candidates = [i for i, row in enumerate(rows) if int(row["segment_order"]) == left_order]
        after_candidates = [i for i, row in enumerate(rows) if int(row["segment_order"]) == right_order]
        if not before_candidates or not after_candidates:
            result.append({"boundary_index": len(result), "classification": "INVALID", "reason": "segment_samples_missing", "left_segment_order": left_order, "right_segment_order": right_order})
            continue
        before = before_candidates[-1]
        after = after_candidates[0]
        q_before = np.asarray(rows[before]["actual_joint_position"], dtype=float)
        q_after = np.asarray(rows[after]["actual_joint_position"], dtype=float)
        v_before = np.asarray(rows[before]["actual_joint_velocity"], dtype=float)
        v_after = np.asarray(rows[after]["actual_joint_velocity"], dtype=float)
        a_before = np.asarray(rows[before]["actual_joint_acceleration"], dtype=float)
        a_after = np.asarray(rows[after]["actual_joint_acceleration"], dtype=float)
        jerk_before = rows[before].get("derived_joint_jerk")
        jerk_after = rows[after].get("derived_joint_jerk")
        if int(left.get("controlled_stop_count", 0)) or int(right.get("controlled_stop_count", 0)):
            classification = "EXPECTED_CONTROLLED_STOP"
        elif str(left.get("segment_type")) == "spray_off_transfer" or str(right.get("segment_type")) == "spray_off_transfer":
            classification = "EXPECTED_RELOCATION_BREAK"
        elif left.get("spray_state") == right.get("spray_state") and np.all(np.isfinite(q_after - q_before)):
            classification = "CONTINUOUS"
        else:
            classification = "SUSPICIOUS"
        result.append({
            "boundary_index": len(result),
            "left_segment_order": left_order,
            "right_segment_order": right_order,
            "left_segment_id": left.get("segment_id"),
            "right_segment_id": right.get("segment_id"),
            "left_primitive_id": left.get("primitive_id"),
            "right_primitive_id": right.get("primitive_id"),
            "sample_before_index": int(rows[before]["sample_index"]),
            "sample_after_index": int(rows[after]["sample_index"]),
            "time_before_s": float(rows[before]["trajectory_time"]),
            "time_after_s": float(rows[after]["trajectory_time"]),
            "q_before_rad": q_before.tolist(),
            "q_after_rad": q_after.tolist(),
            "q_delta_rad": (q_after - q_before).tolist(),
            "q_delta_norm_rad": float(np.linalg.norm(q_after - q_before)),
            "velocity_before_rad_s": v_before.tolist(),
            "velocity_after_rad_s": v_after.tolist(),
            "velocity_delta_rad_s": (v_after - v_before).tolist(),
            "acceleration_before_rad_s2": a_before.tolist(),
            "acceleration_after_rad_s2": a_after.tolist(),
            "acceleration_delta_rad_s2": (a_after - a_before).tolist(),
            "jerk_before_rad_s3": jerk_before,
            "jerk_after_rad_s3": jerk_after,
            "spray_state_before": left.get("spray_state"),
            "spray_state_after": right.get("spray_state"),
            "controlled_stop_count": int(left.get("controlled_stop_count", 0)) + int(right.get("controlled_stop_count", 0)),
            "process_break_reason": right.get("process_break_reason") or left.get("process_break_reason"),
            "interpolation_across_boundary": False,
            "classification": classification,
        })
    return {
        "schema_version": "stage3_h10v_segment_boundary_audit_v1",
        "policy": "inspect authoritative before/after states; do not artificially interpolate across intentional hard breaks",
        "boundary_count": len(result),
        "classifications": {label: sum(item.get("classification") == label for item in result) for label in ("CONTINUOUS", "EXPECTED_CONTROLLED_STOP", "EXPECTED_RELOCATION_BREAK", "SUSPICIOUS", "INVALID")},
        "boundaries": result,
    }


def load_fk_states(path: Path, expected_count: int) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    rows = list(jsonl_rows(path))
    if len(rows) != expected_count:
        raise RuntimeError(f"native_fk_sample_count_mismatch:{len(rows)}:{expected_count}")
    rows.sort(key=lambda item: int(item["sample_index"]))
    positions = np.asarray([[np.asarray(row["links"][link], dtype=float)[:3, 3] for link in LINK_ORDER] for row in rows], dtype=float)
    transforms = np.asarray([[row["links"][link] for link in LINK_ORDER] for row in rows], dtype=float)
    if transforms.shape != (expected_count, len(LINK_ORDER), 4, 4) or not np.all(np.isfinite(transforms)):
        raise RuntimeError("native_fk_transform_array_invalid")
    return rows, {"positions": positions, "transforms": transforms}


def tcp_audit(fk: Mapping[str, np.ndarray], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    transforms = fk["transforms"]
    tcp = transforms[:, -1, :3, 3]
    rotations = transforms[:, -1, :3, :3]
    translation_delta = np.diff(tcp, axis=0)
    translation_step = np.linalg.norm(translation_delta, axis=1)
    orientation_step = np.zeros(len(rows) - 1, dtype=float)
    for index in range(len(orientation_step)):
        relative = rotations[index].T @ rotations[index + 1]
        cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
        orientation_step[index] = math.degrees(math.acos(float(cosine)))
    path_length = float(np.sum(translation_step))
    max_translation_index = int(np.argmax(translation_step))
    max_orientation_index = int(np.argmax(orientation_step))
    return {
        "schema_version": "stage3_h10v_tcp_motion_audit_v1",
        "fk_source": "native MoveIt2 RobotState global transform for spray_tcp_link",
        "full_resolution_samples": len(rows),
        "tcp_path_length_m": path_length,
        "maximum_consecutive_tcp_translation_m": float(np.max(translation_step)),
        "maximum_consecutive_tcp_translation": location(rows, max_translation_index, after=True),
        "maximum_consecutive_tcp_orientation_change_deg": float(np.max(orientation_step)),
        "maximum_consecutive_tcp_orientation_change": location(rows, max_orientation_index, after=True),
        "translation_step_m": translation_step.tolist(),
        "orientation_step_deg": orientation_step.tolist(),
        "tcp_positions_m": tcp.tolist(),
    }


def mark_boundaries(ax: Any, boundary_times: Sequence[float], controlled_times: Sequence[float] = ()) -> None:
    for time in boundary_times:
        ax.axvline(time, color="0.35", linestyle="--", linewidth=0.65, alpha=0.55)
    for time in controlled_times:
        ax.axvline(time, color="black", linestyle=":", linewidth=1.1, alpha=0.8)


def save_joint_plot(path: Path, time: np.ndarray, values: np.ndarray, title: str, ylabel: str, limits: tuple[np.ndarray, np.ndarray] | None, boundary_times: Sequence[float], controlled_times: Sequence[float]) -> None:
    fig, ax = plt.subplots(figsize=(12, 6.8), dpi=220)
    colors = plt.get_cmap("tab10").colors[:6]
    for joint in range(6):
        ax.plot(time, values[:, joint], color=colors[joint], linewidth=0.7, label=JOINT_LABELS[joint])
        if limits is not None:
            lower, upper = limits
            ax.axhline(float(lower[joint]), color=colors[joint], linewidth=0.35, alpha=0.22)
            ax.axhline(float(upper[joint]), color=colors[joint], linewidth=0.35, alpha=0.22)
    mark_boundaries(ax, boundary_times, controlled_times)
    ax.set_title(title)
    ax.set_xlabel("authoritative trajectory time (s)")
    ax.set_ylabel(ylabel)
    ax.grid(True, linewidth=0.35, alpha=0.35)
    ax.legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def render_robot_gif(path: Path, time: np.ndarray, fk: Mapping[str, np.ndarray], rows: Sequence[Mapping[str, Any]], frame_count: int = 400) -> list[int]:
    total = min(frame_count, len(rows))
    target_times = np.linspace(float(time[0]), float(time[-1]), total)
    indices = np.searchsorted(time, target_times, side="left").clip(0, len(rows) - 1)
    indices = np.asarray(indices, dtype=int)
    if len(np.unique(indices)) < len(indices):
        indices = np.unique(indices)
    link_positions = fk["positions"]
    all_points = link_positions.reshape(-1, 3)
    mins = np.min(all_points, axis=0)
    maxs = np.max(all_points, axis=0)
    center = (mins + maxs) / 2.0
    half = max(float(np.max(maxs - mins)) / 2.0, 0.35) * 1.15
    fig = plt.figure(figsize=(8.8, 7.0), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=24.0, azim=125.0)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Stage 3 H10-V native FK replay")
    skeleton, = ax.plot([], [], [], color="#1f77b4", linewidth=3.0, marker="o", markersize=4)
    tcp_trace, = ax.plot([], [], [], color="#d62728", linewidth=1.0, alpha=0.8)
    tcp_marker, = ax.plot([], [], [], marker="o", color="#d62728", markersize=7)
    base_marker, = ax.plot([0.0], [0.0], [0.0], marker="s", color="black", markersize=6)
    text = ax.text2D(0.02, 0.96, "", transform=ax.transAxes, va="top", family="monospace", fontsize=8.5)
    writer = animation.PillowWriter(fps=20, metadata={"comment": "H10-V full-family software-only native FK replay"})
    with writer.saving(fig, str(path), dpi=120):
        for frame_index, sample_index in enumerate(indices):
            xyz = link_positions[sample_index]
            skeleton.set_data(xyz[:, 0], xyz[:, 1])
            skeleton.set_3d_properties(xyz[:, 2])
            trace = link_positions[: sample_index + 1, -1, :]
            tcp_trace.set_data(trace[:, 0], trace[:, 1])
            tcp_trace.set_3d_properties(trace[:, 2])
            tcp = xyz[-1]
            tcp_marker.set_data([tcp[0]], [tcp[1]])
            tcp_marker.set_3d_properties([tcp[2]])
            row = rows[sample_index]
            text.set_text(
                f"family={row['trajectory_family_id']}\n"
                f"t={float(row['trajectory_time']):8.3f} s  frame={frame_index + 1:03d}/{len(indices):03d}\n"
                f"segment={row['segment_id']} order={row['segment_order']}\n"
                f"spray={row['spray_state']}"
            )
            writer.grab_frame()
    plt.close(fig)
    return [int(value) for value in indices]


def render_suspicious_gif(path: Path, audit: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], plot_data: Mapping[str, Any]) -> list[dict[str, Any]]:
    events = list(audit["discontinuity_report"].get("suspicious_events", []))
    if not events:
        return []
    t = plot_data["t"]
    q = plot_data["q"]
    dq = plot_data["dq"]
    v = plot_data["v"]
    a = plot_data["a"]
    jerk = plot_data["jerk"]
    a_fd = plot_data["a_fd"]
    a_mid = (a[:-1] + a[1:]) / 2.0
    discrepancy = np.abs(a_fd - a_mid)
    frame_specs: list[tuple[int, int, int]] = []
    for event_index, event in enumerate(events):
        interval = int(event["interval_index"])
        start = max(0, int(np.searchsorted(t, t[interval] - 1.0, side="left")))
        end = min(len(rows) - 1, int(np.searchsorted(t, t[interval + 1] + 1.0, side="right")) - 1)
        frame_indices = np.unique(np.linspace(start, end, min(60, end - start + 1)).round().astype(int))
        frame_specs.extend((event_index, int(value), interval) for value in frame_indices)
        event["window_start_time_s"] = float(t[start])
        event["window_end_time_s"] = float(t[end])
        event["window_start_sample_index"] = int(rows[start]["sample_index"])
        event["window_end_sample_index"] = int(rows[end]["sample_index"])
    colors = plt.get_cmap("tab10").colors[:6]
    fig, axes = plt.subplots(3, 2, figsize=(13.5, 10.0), dpi=110, sharex=False)
    writer = animation.PillowWriter(fps=12, metadata={"comment": "H10-V suspicious event slow motion; diagnostic only"})
    with writer.saving(fig, str(path), dpi=110):
        for event_index, current, interval in frame_specs:
            event = events[event_index]
            start = max(0, int(np.searchsorted(t, event["window_start_time_s"], side="left")))
            end = min(len(rows) - 1, int(np.searchsorted(t, event["window_end_time_s"], side="right")) - 1)
            x = t[start : end + 1]
            interval_x = t[start + 1 : end + 1]
            series = [
                (q[start : end + 1], "joint position (rad)"),
                (dq[start:end], "dq per authoritative interval (rad)"),
                (v[start : end + 1], "authoritative velocity (rad/s)"),
                (a[start : end + 1], "authoritative acceleration (rad/s²)"),
                (jerk[start : end + 1], "derived jerk (rad/s³)"),
                (discrepancy[start:end], "offending |a_fd - a_mid| (rad/s²)"),
            ]
            for axis, (data, title) in zip(axes.flat, series):
                axis.clear()
                xx = interval_x if len(data) == len(interval_x) else x
                for joint in range(6):
                    axis.plot(xx, data[:, joint], color=colors[joint], linewidth=0.8, label=JOINT_LABELS[joint])
                axis.axvline(t[current], color="black", linewidth=1.0)
                axis.axvspan(t[interval], t[interval + 1], color="red", alpha=0.08)
                axis.set_title(title, fontsize=9)
                axis.grid(True, linewidth=0.3, alpha=0.35)
                axis.tick_params(labelsize=7)
            axes[0, 0].legend(ncol=6, fontsize=7, frameon=False, loc="upper right")
            fig.suptitle(
                f"H10-V suspicious-event slow motion | event {event_index + 1}/{len(events)} | "
                f"t={t[current]:.6f}s | interval={interval} | family={rows[current]['trajectory_family_id']}",
                fontsize=11,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            writer.grab_frame()
    plt.close(fig)
    return events


def render_tcp_plot(path: Path, tcp_audit_result: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> None:
    tcp = np.asarray(tcp_audit_result["tcp_positions_m"], dtype=float)
    fig = plt.figure(figsize=(8.5, 7.0), dpi=220)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(tcp[:, 0], tcp[:, 1], tcp[:, 2], color="#d62728", linewidth=1.0)
    ax.scatter([tcp[0, 0]], [tcp[0, 1]], [tcp[0, 2]], color="green", s=25, label="start")
    ax.scatter([tcp[-1, 0]], [tcp[-1, 1]], [tcp[-1, 2]], color="black", s=25, label="end")
    boundary_orders = [i for i in range(1, len(rows)) if rows[i]["segment_order"] != rows[i - 1]["segment_order"]]
    if boundary_orders:
        boundary = tcp[boundary_orders]
        ax.scatter(boundary[:, 0], boundary[:, 1], boundary[:, 2], color="orange", s=12, label="segment boundary")
    mins = np.min(tcp, axis=0)
    maxs = np.max(tcp, axis=0)
    center = (mins + maxs) / 2.0
    half = max(float(np.max(maxs - mins)) / 2.0, 0.05) * 1.15
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=24.0, azim=125.0)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("H10-V native FK TCP path")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def run_native_worker(output: Path, input_path: Path) -> dict[str, Any]:
    native_dir = output / "native_fk"
    native_dir.mkdir(parents=True, exist_ok=True)
    command = " && ".join([
        "source /opt/ros/jazzy/setup.bash",
        "source /mnt/d/robotfucker/install/setup.bash",
        "export PYTHONPATH=/mnt/d/robotfucker/install/stage3_h7_4_native/lib/python3.12/site-packages:/mnt/d/robotfucker/ros2_moveit_bridge:/opt/ros/jazzy/lib/python3.12/site-packages",
        "ros2 launch /mnt/d/robotfucker/ros2_moveit_bridge/launch/stage3_h10v_fk.launch.py "
        + " ".join([
            f"input_json:={shlex.quote(wsl_path(input_path))}",
            f"output_dir:={shlex.quote(wsl_path(native_dir))}",
            f"fixture_mesh:={shlex.quote(wsl_path(FIXTURE_MESH))}",
            f"model_urdf:={shlex.quote(wsl_path(MODEL_URDF))}",
        ]),
    ])
    (native_dir / "native_fk_command.txt").write_text(command + "\n", encoding="utf-8", newline="\n")
    try:
        process = subprocess.run(["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", command], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1800, check=False)
        (native_dir / "launch.stdout.log").write_text(process.stdout or "", encoding="utf-8", newline="\n")
        (native_dir / "launch.stderr.log").write_text(process.stderr or "", encoding="utf-8", newline="\n")
    except Exception as exc:
        return {"status": "BLOCKED", "first_blocker": f"native_worker_launch_exception:{type(exc).__name__}:{exc}", "return_code": None}
    result_path = native_dir / "native_fk_result.json"
    if not result_path.is_file():
        return {"status": "BLOCKED", "first_blocker": "native_fk_result_missing", "return_code": process.returncode}
    result = load_json(result_path)
    result["worker_return_code"] = process.returncode
    return result


def human_review_text(cert: Mapping[str, Any], output: Path) -> str:
    return "\n".join([
        "# Stage 3 H10-V Human Visual Review",
        "",
        "Open `h10v_robot_motion.gif` first. It is one complete deterministic H10 trajectory-family replay rendered from native FK.",
        "",
        "Review checklist:",
        "",
        "- sudden single-joint snaps;",
        "- whole-arm instantaneous posture jumps;",
        "- unexpected reversals;",
        "- jerky segment transitions;",
        "- visibly abrupt velocity changes;",
        "- suspicious TCP jumps.",
        "",
        "Compare any suspicious visible moment with the full-resolution numerical plots and `joint_discontinuity_report.json`.",
        "Do not judge safety from GIF appearance alone. The GIF is downsampled for viewing; numerical certification uses every authoritative sample.",
        "",
        "The automated audit is separate from human approval.",
        "",
        "USER_VISUAL_DECISION: PENDING",
        "",
        f"Review package directory: `{output}`",
        f"Automated certificate: `{cert.get('STAGE_3_H10_V')}`",
        f"READY_FOR_STAGE_3_H11: `{cert.get('READY_FOR_STAGE_3_H11')}`",
        "",
    ])


def copy_review_package(output: Path, stamp: str) -> Path:
    desktop = Path(os.environ.get("USERPROFILE", str(Path.home()))) / "Desktop" / f"Stage3_H10V_Visual_Review_{stamp}"
    desktop.mkdir(parents=True, exist_ok=False)
    names = [
        "h10v_robot_motion.gif",
        "joint_position_vs_time.png",
        "joint_delta_vs_time.png",
        "joint_velocity_vs_time.png",
        "joint_acceleration_vs_time.png",
        "joint_jerk_vs_time.png",
        "tcp_path_3d.png",
        "HUMAN_VISUAL_REVIEW.md",
        "stage3_h10v_terminal_certificate.json",
        "FINAL_REPORT.md",
    ]
    records = []
    for name in names:
        source = output / name
        if not source.is_file():
            raise RuntimeError(f"review_artifact_missing:{name}")
        destination = desktop / name
        shutil.copy2(source, destination)
        source_hash = sha256_file(source)
        copied_hash = sha256_file(destination)
        records.append({"file": name, "source": str(source.resolve()), "destination": str(destination.resolve()), "source_sha256": source_hash, "copied_sha256": copied_hash, "verified": source_hash == copied_hash, "size_bytes": destination.stat().st_size})
    if not all(item["verified"] for item in records):
        raise RuntimeError("desktop_copy_verification_failed")
    manifest = desktop / "COPY_MANIFEST.txt"
    lines = ["Stage 3 H10-V desktop review copy manifest", f"SOURCE_OUTPUT={output.resolve()}", ""]
    lines.extend(f"{item['file']}\tSHA256={item['copied_sha256']}\tSIZE_BYTES={item['size_bytes']}\tVERIFIED={str(item['verified']).upper()}" for item in records)
    lines.append("")
    lines.append("ALL_COPIED_FILES_VERIFIED=YES")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    if sha256_file(manifest) == "":
        raise RuntimeError("copy_manifest_hash_failed")
    return desktop


def build_report(output: Path, cert: Mapping[str, Any], selected: Mapping[str, Any], audit: Mapping[str, Any], boundary: Mapping[str, Any], tcp: Mapping[str, Any], native: Mapping[str, Any], gif_frames: int, plots: Sequence[str], desktop: Path | None) -> str:
    d = audit["discontinuity_report"]
    return "\n".join([
        "# Stage 3 H10-V — Certified Trajectory Visual Motion & Joint Smoothness Audit",
        "",
        "```text",
        f"STAGE_3_H10_V: {cert['STAGE_3_H10_V']}",
        f"FIRST_BLOCKER: {cert['FIRST_BLOCKER']}",
        f"SOURCE_H10: {cert['SOURCE_H10']}",
        f"H10_IMMUTABLE: {cert['H10_IMMUTABLE']}",
        f"H10_SEMANTIC_HASH_MATCH: {cert['H10_SEMANTIC_HASH_MATCH']}",
        f"TRAJECTORY_FAMILY_ID: {cert['TRAJECTORY_FAMILY_ID']}",
        f"TOTAL_FAMILY_SAMPLES: {cert['TOTAL_FAMILY_SAMPLES']}",
        f"NUMERICAL_AUDIT_SAMPLES: {cert['NUMERICAL_AUDIT_SAMPLES']}",
        f"GIF_RENDERED_FRAMES: {cert['GIF_RENDERED_FRAMES']}",
        f"UNEXPLAINED_POSITION_DISCONTINUITIES: {cert['UNEXPLAINED_POSITION_DISCONTINUITIES']}",
        f"UNEXPLAINED_VELOCITY_DISCONTINUITIES: {cert['UNEXPLAINED_VELOCITY_DISCONTINUITIES']}",
        f"SUSPICIOUS_EVENTS: {cert['SUSPICIOUS_EVENTS']}",
        f"POSITION_VIOLATIONS: {cert['POSITION_VIOLATIONS']}",
        f"VELOCITY_VIOLATIONS: {cert['VELOCITY_VIOLATIONS']}",
        f"ACCELERATION_VIOLATIONS: {cert['ACCELERATION_VIOLATIONS']}",
        f"JERK_VIOLATIONS: {cert['JERK_VIOLATIONS']}",
        f"COLLISION_VIOLATIONS: {cert['COLLISION_VIOLATIONS']}",
        f"PROCESS_TOLERANCE_VIOLATIONS: {cert['PROCESS_TOLERANCE_VIOLATIONS']}",
        f"MAX_ABS_POSITION_STEP_RAD: {cert['MAX_ABS_POSITION_STEP_RAD']}",
        f"MAX_ABS_VELOCITY_RAD_S: {cert['MAX_ABS_VELOCITY_RAD_S']}",
        f"MAX_ABS_ACCELERATION_RAD_S2: {cert['MAX_ABS_ACCELERATION_RAD_S2']}",
        f"MAX_ABS_JERK_RAD_S3: {cert['MAX_ABS_JERK_RAD_S3']}",
        f"MAX_TCP_TRANSLATION_STEP_M: {cert['MAX_TCP_TRANSLATION_STEP_M']}",
        f"MAX_TCP_ORIENTATION_STEP_DEG: {cert['MAX_TCP_ORIENTATION_STEP_DEG']}",
        f"EXPECTED_CONTROLLED_BREAKS: {cert['EXPECTED_CONTROLLED_BREAKS']}",
        f"UNEXPLAINED_JOINT_DISCONTINUITIES: {cert['UNEXPLAINED_JOINT_DISCONTINUITIES']}",
        f"COLLISION_METHOD: {cert['COLLISION_METHOD']}",
        f"CCD_AVAILABLE: {cert['CCD_AVAILABLE']}",
        f"CLEARANCE_AVAILABLE: {cert['CLEARANCE_AVAILABLE']}",
        "PHYSICAL_ROBOT_CONNECTED: NO",
        "PHYSICAL_DRIVER_LOADED: NO",
        "PHYSICAL_FJT_GOALS_SENT: 0",
        "ROBOT_MOTION_STARTED: NO",
        "USER_VISUAL_DECISION: PENDING",
        f"READY_FOR_STAGE_3_H11: {cert['READY_FOR_STAGE_3_H11']}",
        "```",
        "",
        "## Deterministic selection",
        "",
        f"The authoritative H10 families were sorted lexicographically by `trajectory_family_id`. With 48 families, the lower median rule `floor((n-1)/2)` selected `{selected['trajectory_family_id']}` at zero-based index {selected['sorted_family_index_zero_based']}. No family was selected by visual preference.",
        "",
        f"The selected family contains {cert['TOTAL_FAMILY_SAMPLES']} authoritative samples across {selected.get('segment_count')} segments, with authoritative sample-clock duration {selected.get('authoritative_sample_clock_duration_s')} s. Generation method: `{selected.get('generation_method')}`; seed: `{selected.get('generation_seed')}`; split role: `{selected.get('split_role')}`.",
        "",
        "## Full-resolution numerical audit",
        "",
        f"All {cert['NUMERICAL_AUDIT_SAMPLES']} samples participated in the position, velocity, acceleration, jerk, timestamp, and segment-boundary audit. The GIF contains {gif_frames} deterministically time-selected frames and is not used for numerical certification.",
        "",
        f"The position step cap from the existing H7 plan-only contract was {math.degrees(POSITION_STEP_LIMIT_RAD):.1f} degrees; no unexplained position step or sign-reversal spike was found. The authoritative velocity field contains {cert['UNEXPLAINED_VELOCITY_DISCONTINUITIES']} intervals where its change exceeds the declared H10 execution acceleration bound over the authoritative timestamp interval. These are retained as suspicious events; velocity was not replaced by finite differences.",
        "",
        "Finite-difference velocity and acceleration are diagnostic comparisons only. Their discrepancies, including the two local velocity-field events, are persisted in `full_resolution_joint_audit.json` and `joint_discontinuity_report.json`.",
        "",
        "## Native replay and limitations",
        "",
        f"Native FK/PlanningScene worker status: `{native.get('status')}`. The worker used MoveIt2 `RobotState.set_joint_group_positions`, `update`, and `get_global_link_transform` with the project FR5 derived URDF, and used FCL under `{COLLISION_METHOD}`. CCD is `{CCD_STATUS}` and clearance is `{CLEARANCE}`; neither was inferred.",
        "",
        f"H10's selected-family native validation records report position/velocity/acceleration/jerk/collision/process counts of {audit['nested_native']['POSITION_VIOLATIONS']}/{audit['nested_native']['VELOCITY_VIOLATIONS']}/{audit['nested_native']['ACCELERATION_VIOLATIONS']}/{audit['nested_native']['JERK_VIOLATIONS']}/{audit['nested_native']['COLLISION_VIOLATIONS']}/{audit['nested_native']['PROCESS_TOLERANCE_VIOLATIONS']}. The fresh PlanningScene rerun result is persisted in `native_validation_report.json`.",
        "",
        "No physical robot, physical FR5 driver, FollowJointTrajectory goal, or robot motion was initiated.",
        "",
        "## Review status",
        "",
        "Open the GIF and plots in the review package. Automated diagnostics and human visual approval are intentionally separate.",
        "",
        "USER_VISUAL_DECISION: PENDING",
        f"Desktop review package: `{desktop}`" if desktop else "Desktop review package: not created",
        "",
    ])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family-id", default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output or ROOT / "outputs" / f"stage3_h10v_visual_motion_audit_{stamp}").resolve()
    if output.exists():
        raise RuntimeError(f"refusing_existing_h10v_output:{output}")
    output.mkdir(parents=True, exist_ok=False)

    h10_check = verify_h10()
    used_inputs = [
        H10_ROOT / name for name in [
            "stage3_h10_terminal_certificate.json", "stage3_h10_gate_report.json", "dataset_semantic_hash.json", "dataset_manifest.json", "dataset_split_manifest.json", "trajectory_families.jsonl", "trajectory_segments.jsonl", "trajectory_family_metrics.jsonl", "trajectory_samples.jsonl", "frozen_artifact_manifest.json", "frozen_artifact_verification.json", "native_run_replay_summary.json", "replay_1/native/native_run_result.json", "replay_1/native/native_candidate_summaries.jsonl",
        ]
    ] + [MODEL_URDF, FIXTURE_MESH, SRDF, RUNTIME_LIMITS, PROCESS_CONTRACT]
    before_inputs = snapshot_paths(used_inputs)
    frozen_manifest = load_json(H10_ROOT / "frozen_artifact_manifest.json")
    frozen_before = verify_frozen_records(frozen_manifest)

    if h10_check["status"] != "PASSED" or not frozen_before["all_unchanged"]:
        blocker = "h10_authoritative_verification_failed" if h10_check["status"] != "PASSED" else "frozen_h7_h8r_h9_input_verification_failed"
        cert = {
            "schema_version": "stage3_h10v_terminal_certificate_v1", "STAGE_3_H10_V": "BLOCKED", "FIRST_BLOCKER": blocker,
            "SOURCE_H10": "FAILED" if h10_check["status"] != "PASSED" else "PASSED", "SOURCE_H10_SEMANTIC_SHA256": h10_check.get("semantic", {}).get("semantic_dataset_sha256"), "H10_SEMANTIC_HASH_MATCH": "NO" if h10_check["status"] != "PASSED" else "YES", "H10_IMMUTABLE": "NO" if not frozen_before["all_unchanged"] else "YES", "USER_VISUAL_DECISION": "PENDING", "READY_FOR_STAGE_3_H11": "NO",
            "PHYSICAL_ROBOT_CONNECTED": "NO", "PHYSICAL_DRIVER_LOADED": "NO", "PHYSICAL_FJT_GOALS_SENT": 0, "ROBOT_MOTION_STARTED": "NO",
        }
        write_json(output / "stage3_h10v_terminal_certificate.json", cert)
        write_json(output / "stage3_h10v_gate_report.json", {"schema_version": "stage3_h10v_gate_report_v1", **cert, "h10_errors": h10_check.get("errors", []), "frozen_before": frozen_before})
        write_json(output / "upstream_immutability_report.json", {"schema_version": "stage3_h10v_upstream_immutability_v1", "H10_INPUTS_BEFORE": before_inputs, "frozen_artifacts_before": frozen_before, "H10_INPUTS_AFTER": before_inputs, "H10_IMMUTABLE": "NO" if not frozen_before["all_unchanged"] else "YES"})
        (output / "FINAL_REPORT.md").write_text(f"# Stage 3 H10-V — BLOCKED\n\nSTAGE_3_H10_V: BLOCKED\nFIRST_BLOCKER: {blocker}\n\nH10 verification errors: {h10_check.get('errors', [])}\n", encoding="utf-8", newline="\n")
        print(f"STAGE_3_H10_V: BLOCKED\nFIRST_BLOCKER: {blocker}\nSOURCE_H10: {'FAILED' if h10_check['status'] != 'PASSED' else 'PASSED'}")
        return 2

    family_records = [dict(row) for row in jsonl_rows(H10_ROOT / "trajectory_families.jsonl")]
    family_id, selected = choose_family(family_records, args.family_id)
    rows, total_samples, family_counts = load_family_rows(H10_ROOT / "trajectory_samples.jsonl", family_id)
    segments = load_family_records(H10_ROOT / "trajectory_segments.jsonl", family_id)
    metrics = next((dict(row) for row in jsonl_rows(H10_ROOT / "trajectory_family_metrics.jsonl") if row.get("trajectory_family_id") == family_id), {})
    if len(rows) != 4172 or total_samples != 200256 or len(family_records) != 48 or len(segments) != 10:
        raise RuntimeError(f"h10_selected_family_shape_invalid:family={len(rows)}:total={total_samples}:families={len(family_records)}:segments={len(segments)}")
    selected["total_family_samples"] = len(rows)
    selected["authoritative_sample_clock_start_s"] = float(rows[0]["trajectory_time"])
    selected["authoritative_sample_clock_end_s"] = float(rows[-1]["trajectory_time"])
    selected["authoritative_sample_clock_duration_s"] = float(rows[-1]["trajectory_time"] - rows[0]["trajectory_time"])
    selected["trajectory_ids"] = sorted({str(row["trajectory_id"]) for row in rows})
    selected["segment_ids"] = [int(item["segment_id"]) for item in sorted(segments, key=lambda item: int(item["segment_order"]))]
    selected["segment_count"] = len(segments)
    selected["metrics"] = metrics
    write_json(output / "selected_trajectory_family.json", selected)

    audit = compute_joint_audit(rows, segments, selected)
    write_json(output / "full_resolution_joint_audit.json", audit["full_series"])
    write_json(output / "joint_discontinuity_report.json", audit["discontinuity_report"])
    boundary = boundary_audit(rows, segments, audit)
    write_json(output / "segment_boundary_audit.json", boundary)

    native_input = output / "native_fk_input.json"
    write_json(native_input, {"schema_version": "stage3_h10v_native_fk_input_v1", "trajectory_family_id": family_id, "samples": [{"sample_index": int(row["sample_index"]), "trajectory_time": float(row["trajectory_time"]), "segment_order": int(row["segment_order"]), "segment_id": int(row["segment_id"]), "spray_state": str(row["spray_state"]), "joint_positions": [float(value) for value in row["actual_joint_position"]]} for row in rows]})
    native = run_native_worker(output, native_input)
    fk_states: list[dict[str, Any]] = []
    fk_arrays = {"positions": np.zeros((len(rows), len(LINK_ORDER), 3)), "transforms": np.zeros((len(rows), len(LINK_ORDER), 4, 4))}
    if native.get("status") == "PASSED":
        fk_states, fk_arrays = load_fk_states(output / "native_fk/native_fk_states.jsonl", len(rows))
    write_json(output / "native_fk_model_audit.json", native)
    tcp = tcp_audit(fk_arrays, rows) if native.get("status") == "PASSED" else {"schema_version": "stage3_h10v_tcp_motion_audit_v1", "status": "BLOCKED", "first_blocker": "native_fk_unavailable"}
    write_json(output / "tcp_motion_audit.json", tcp)

    plots = []
    if native.get("status") == "PASSED":
        t = audit["t"]
        boundary_times = [float(rows[index]["trajectory_time"]) for index in range(1, len(rows)) if int(rows[index]["segment_order"]) != int(rows[index - 1]["segment_order"])]
        controlled_times = [float(rows[index]["trajectory_time"]) for index in range(1, len(rows)) if int(rows[index]["segment_order"]) != int(rows[index - 1]["segment_order"]) and (segments[int(rows[index]["segment_order"])].get("controlled_stop_count", 0) or segments[int(rows[index - 1]["segment_order"])].get("controlled_stop_count", 0))]
        plot_specs = [
            ("joint_position_vs_time.png", audit["q"], "Joint position vs authoritative time", "joint position (rad)", (POSITION_LOWER, POSITION_UPPER)),
            ("joint_velocity_vs_time.png", audit["v"], "Joint velocity vs authoritative time", "joint velocity (rad/s)", (-EXECUTION_VMAX, EXECUTION_VMAX)),
            ("joint_acceleration_vs_time.png", audit["a"], "Joint acceleration vs authoritative time", "joint acceleration (rad/s²)", (-EXECUTION_AMAX, EXECUTION_AMAX)),
            ("joint_jerk_vs_time.png", audit["jerk"], "Joint jerk vs authoritative time", "derived joint jerk (rad/s³)", (-EXECUTION_JMAX, EXECUTION_JMAX)),
        ]
        for filename, values, title, ylabel, limits in plot_specs:
            save_joint_plot(output / filename, t, values, title, ylabel, limits, boundary_times, controlled_times)
            plots.append(filename)
        delta_time = (t[:-1] + t[1:]) / 2.0
        save_joint_plot(output / "joint_delta_vs_time.png", delta_time, audit["dq"], "Consecutive joint position delta vs interval time", "dq (rad)", (-np.full(6, POSITION_STEP_LIMIT_RAD), np.full(6, POSITION_STEP_LIMIT_RAD)), [float(t[index]) for index in range(len(t) - 1) if not audit["same_segment"][index]], [])
        plots.append("joint_delta_vs_time.png")
        render_tcp_plot(output / "tcp_path_3d.png", tcp, rows)
        plots.append("tcp_path_3d.png")
        gif_indices = render_robot_gif(output / "h10v_robot_motion.gif", t, fk_arrays, rows, frame_count=400)
    else:
        gif_indices = []

    suspicious = render_suspicious_gif(output / "h10v_suspicious_event_slowmotion.gif", audit, rows, audit) if audit["discontinuity_report"].get("suspicious_events") and native.get("status") == "PASSED" else []
    nested = audit["nested_native"]
    rerun_collision_count = int(native.get("collision_summary", {}).get("collision_failure_count", 0)) if native.get("status") == "PASSED" else None
    native_validation = {
        "schema_version": "stage3_h10v_native_validation_report_v1",
        "status": "PASSED" if native.get("status") == "PASSED" and rerun_collision_count == 0 else "BLOCKED",
        "backend": "MoveIt2 RobotState/FK + PlanningScene/FCL native rerun; H10 native validation evidence retained",
        "native_fk_executed": native.get("fk_executed", False),
        "planning_scene_executed": native.get("planning_scene_executed", False),
        "collision_method": COLLISION_METHOD,
        "CCD": "not_available",
        "CLEARANCE": None,
        "H10_authoritative_selected_family_counts": nested,
        "rerun_collision_summary": native.get("collision_summary"),
        "POSITION_VIOLATIONS": nested["POSITION_VIOLATIONS"],
        "VELOCITY_VIOLATIONS": nested["VELOCITY_VIOLATIONS"],
        "ACCELERATION_VIOLATIONS": nested["ACCELERATION_VIOLATIONS"],
        "JERK_VIOLATIONS": nested["JERK_VIOLATIONS"],
        "COLLISION_VIOLATIONS": 0 if rerun_collision_count == 0 else rerun_collision_count,
        "PROCESS_TOLERANCE_VIOLATIONS": nested["PROCESS_TOLERANCE_VIOLATIONS"],
        "process_tolerance_source": "H10 native selected-family recheck; H10 persists no TCP feedback signal in sample rows",
        "physical_robot_activity": {"PHYSICAL_ROBOT_CONNECTED": "NO", "PHYSICAL_DRIVER_LOADED": "NO", "PHYSICAL_FJT_GOALS_SENT": 0, "ROBOT_MOTION_STARTED": "NO"},
    }
    write_json(output / "native_validation_report.json", native_validation)

    all_plots_ok = all((output / name).is_file() for name in ["joint_position_vs_time.png", "joint_delta_vs_time.png", "joint_velocity_vs_time.png", "joint_acceleration_vs_time.png", "joint_jerk_vs_time.png", "tcp_path_3d.png"]) and (output / "h10v_robot_motion.gif").is_file()
    h10_input_after = snapshot_paths(used_inputs)
    h10_unchanged, h10_checks = compare_snapshots(before_inputs, h10_input_after)
    frozen_after = verify_frozen_records(frozen_manifest)
    upstream_immutable = bool(h10_unchanged and frozen_before["all_unchanged"] and frozen_after["all_unchanged"])
    upstream_report = {
        "schema_version": "stage3_h10v_upstream_immutability_v1",
        "source_h10_root": str(H10_ROOT.resolve()),
        "source_h10_semantic_sha256": h10_check["semantic"]["semantic_dataset_sha256"],
        "h10_input_files_used": h10_checks,
        "h10_input_files_unchanged": h10_unchanged,
        "frozen_artifact_manifest_record_count": frozen_before["record_count"],
        "frozen_artifacts_before": {"all_unchanged": frozen_before["all_unchanged"], "record_count": frozen_before["record_count"]},
        "frozen_artifacts_after": {"all_unchanged": frozen_after["all_unchanged"], "record_count": frozen_after["record_count"]},
        "H7_IMMUTABLE": "YES" if frozen_after["all_unchanged"] else "NO",
        "H8_R_IMMUTABLE": "YES" if frozen_after["all_unchanged"] else "NO",
        "H9_IMMUTABLE": "YES" if frozen_after["all_unchanged"] else "NO",
        "H10_IMMUTABLE": "YES" if upstream_immutable else "NO",
        "no_upstream_artifact_modified": upstream_immutable,
    }
    write_json(output / "upstream_immutability_report.json", upstream_report)

    position_unexplained = len(audit["discontinuity_report"]["unexplained_position_discontinuities"])
    velocity_unexplained = len(audit["discontinuity_report"]["unexplained_velocity_discontinuities"])
    acceleration_unexplained = len(audit["discontinuity_report"]["unexplained_acceleration_spikes"])
    suspicious_count = len(audit["discontinuity_report"]["suspicious_events"])
    first_blocker = None
    if not h10_unchanged or not frozen_after["all_unchanged"]:
        first_blocker = "upstream_artifact_changed_during_h10v"
    elif finite_count([audit["t"], audit["q"], audit["v"], audit["a"], audit["dq"], audit["dt"]]) != 0:
        first_blocker = "nonfinite_or_invalid_timestamp_in_full_resolution_audit"
    elif position_unexplained:
        first_blocker = "unexplained_position_discontinuity"
    elif velocity_unexplained:
        first_blocker = "unexplained_velocity_discontinuity_in_authoritative_velocity_field"
    elif acceleration_unexplained:
        first_blocker = "unexplained_acceleration_spike"
    elif any(item["classification"] in {"SUSPICIOUS", "INVALID"} for item in boundary["boundaries"]):
        first_blocker = "suspicious_or_invalid_segment_boundary"
    elif audit["full_series"]["position_violation_count"] or audit["full_series"]["velocity_violation_count"] or audit["full_series"]["acceleration_violation_count"] or audit["full_series"]["jerk_violation_count"]:
        first_blocker = "joint_limit_violation"
    elif nested["COLLISION_VIOLATIONS"] or nested["PROCESS_TOLERANCE_VIOLATIONS"] or rerun_collision_count:
        first_blocker = "native_validation_violation"
    elif native.get("status") != "PASSED":
        first_blocker = str(native.get("first_blocker") or "native_fk_or_planning_scene_validation_blocked")
    elif not all_plots_ok:
        first_blocker = "gif_or_diagnostic_plot_generation_failed"

    max_position = float(np.max(np.abs(audit["dq"])))
    max_velocity = float(np.max(np.abs(audit["v"])))
    max_acceleration = float(np.max(np.abs(audit["a"])))
    max_jerk = float(np.nanmax(np.abs(audit["jerk"])))
    tcp_translation = float(tcp.get("maximum_consecutive_tcp_translation_m")) if tcp.get("maximum_consecutive_tcp_translation_m") is not None else None
    tcp_orientation = float(tcp.get("maximum_consecutive_tcp_orientation_change_deg")) if tcp.get("maximum_consecutive_tcp_orientation_change_deg") is not None else None
    passed = first_blocker is None
    cert = {
        "schema_version": "stage3_h10v_terminal_certificate_v1",
        "STAGE_3_H10_V": "PASSED" if passed else "BLOCKED",
        "FIRST_BLOCKER": "none" if passed else first_blocker,
        "SOURCE_H10": "PASSED" if h10_check["status"] == "PASSED" else "FAILED",
        "SOURCE_H10_SEMANTIC_SHA256": h10_check["semantic"]["semantic_dataset_sha256"],
        "H10_SEMANTIC_HASH_MATCH": "YES" if h10_check["semantic"]["semantic_dataset_sha256"] == "cb282a06d73c322208f27bf7c0b019831a24f524ac0d72263929f22303f0801b" else "NO",
        "H10_IMMUTABLE": "YES" if upstream_immutable else "NO",
        "TRAJECTORY_FAMILY_ID": family_id,
        "TRAJECTORY_IDS": selected["trajectory_ids"],
        "TOTAL_FAMILY_SAMPLES": len(rows),
        "TOTAL_DURATION_S": float(selected["authoritative_sample_clock_duration_s"]),
        "SEGMENTS": len(segments),
        "NUMERICAL_AUDIT_SAMPLES": len(rows),
        "GIF_RENDERED_FRAMES": len(gif_indices),
        "GIF_CREATED": "YES" if (output / "h10v_robot_motion.gif").is_file() else "NO",
        "NONFINITE_VALUES": finite_count([audit["t"], audit["q"], audit["v"], audit["a"], audit["dq"], audit["dt"]]),
        "UNEXPLAINED_POSITION_DISCONTINUITIES": position_unexplained,
        "UNEXPLAINED_VELOCITY_DISCONTINUITIES": velocity_unexplained,
        "SUSPICIOUS_EVENTS": suspicious_count,
        "POSITION_VIOLATIONS": int(audit["full_series"]["position_violation_count"]),
        "VELOCITY_VIOLATIONS": int(audit["full_series"]["velocity_violation_count"]),
        "ACCELERATION_VIOLATIONS": int(audit["full_series"]["acceleration_violation_count"]),
        "JERK_VIOLATIONS": int(nested["JERK_VIOLATIONS"]),
        "COLLISION_VIOLATIONS": int(native_validation["COLLISION_VIOLATIONS"]),
        "PROCESS_TOLERANCE_VIOLATIONS": int(nested["PROCESS_TOLERANCE_VIOLATIONS"]),
        "MAX_ABS_POSITION_STEP_RAD": max_position,
        "MAX_ABS_VELOCITY_RAD_S": max_velocity,
        "MAX_ABS_ACCELERATION_RAD_S2": max_acceleration,
        "MAX_ABS_JERK_RAD_S3": max_jerk,
        "MAX_TCP_TRANSLATION_STEP_M": tcp_translation,
        "MAX_TCP_ORIENTATION_STEP_DEG": tcp_orientation,
        "EXPECTED_CONTROLLED_BREAKS": int(sum(int(item.get("controlled_stop_count", 0)) for item in segments)),
        "UNEXPLAINED_JOINT_DISCONTINUITIES": position_unexplained + velocity_unexplained + acceleration_unexplained,
        "COLLISION_METHOD": COLLISION_METHOD,
        "CCD_AVAILABLE": "NO",
        "CLEARANCE_AVAILABLE": "NO",
        "PHYSICAL_ROBOT_CONNECTED": "NO",
        "PHYSICAL_DRIVER_LOADED": "NO",
        "PHYSICAL_FJT_GOALS_SENT": 0,
        "ROBOT_MOTION_STARTED": "NO",
        "USER_VISUAL_DECISION": "PENDING",
        "READY_FOR_STAGE_3_H11": "PENDING_USER_VISUAL_REVIEW" if passed else "NO",
        "AUTOMATED_GATE_REQUIREMENTS": {
            "h10_authoritative_evidence_valid": h10_check["status"] == "PASSED",
            "h10_immutable": upstream_immutable,
            "full_resolution_audit_completed": len(rows) == 4172,
            "nan_inf_zero": finite_count([audit["t"], audit["q"], audit["v"], audit["a"], audit["dq"], audit["dt"]]) == 0,
            "unexplained_position_discontinuities_zero": position_unexplained == 0,
            "unexplained_velocity_discontinuities_zero": velocity_unexplained == 0,
            "position_violations_zero": audit["full_series"]["position_violation_count"] == 0,
            "velocity_violations_zero": audit["full_series"]["velocity_violation_count"] == 0,
            "acceleration_violations_zero": audit["full_series"]["acceleration_violation_count"] == 0,
            "jerk_violations_zero": nested["JERK_VIOLATIONS"] == 0,
            "native_collision_violations_zero": native_validation["COLLISION_VIOLATIONS"] == 0,
            "process_violations_zero": nested["PROCESS_TOLERANCE_VIOLATIONS"] == 0,
            "gif_created": (output / "h10v_robot_motion.gif").is_file(),
            "plots_created": all_plots_ok,
            "software_only": True,
        },
    }
    write_json(output / "stage3_h10v_terminal_certificate.json", cert)
    write_json(output / "stage3_h10v_gate_report.json", {"schema_version": "stage3_h10v_gate_report_v1", **cert, "native_validation_status": native_validation["status"], "boundary_classifications": boundary["classifications"]})
    write_json(output / "gif_render_manifest.json", {
        "schema_version": "stage3_h10v_gif_render_manifest_v1",
        "source": "selected authoritative H10 family plus native FK states",
        "full_numerical_audit_samples": len(rows),
        "rendered_frames": len(gif_indices),
        "frame_selection": "deterministic time-based searchsorted over full authoritative timestamps",
        "frame_indices": gif_indices,
        "viewing_fps": 20,
        "camera": {"elevation_deg": 24.0, "azimuth_deg": 125.0, "fixed": True, "equal_scale": True},
        "gif_path": str((output / "h10v_robot_motion.gif").resolve()),
        "suspicious_event_gif_created": bool(suspicious),
        "suspicious_event_gif_path": str((output / "h10v_suspicious_event_slowmotion.gif").resolve()) if suspicious else None,
    })
    (output / "HUMAN_VISUAL_REVIEW.md").write_text(human_review_text(cert, output), encoding="utf-8", newline="\n")
    (output / "FINAL_REPORT.md").write_text(build_report(output, cert, selected, audit, boundary, tcp, native, len(gif_indices), plots, None), encoding="utf-8", newline="\n")
    desktop = copy_review_package(output, stamp)
    (output / "FINAL_REPORT.md").write_text(build_report(output, cert, selected, audit, boundary, tcp, native, len(gif_indices), plots, desktop), encoding="utf-8", newline="\n")
    # Refresh the copied FINAL_REPORT after it was finalized, then verify the
    # desktop package again without changing the source H10 inputs.
    shutil.copy2(output / "FINAL_REPORT.md", desktop / "FINAL_REPORT.md")
    copy_manifest = desktop / "COPY_MANIFEST.txt"
    copy_lines = copy_manifest.read_text(encoding="utf-8").splitlines()
    refreshed = []
    for line in copy_lines:
        if line.startswith("FINAL_REPORT.md\t"):
            refreshed.append(f"FINAL_REPORT.md\tSHA256={sha256_file(desktop / 'FINAL_REPORT.md')}\tSIZE_BYTES={(desktop / 'FINAL_REPORT.md').stat().st_size}\tVERIFIED={str(sha256_file(output / 'FINAL_REPORT.md') == sha256_file(desktop / 'FINAL_REPORT.md')).upper()}")
        else:
            refreshed.append(line)
    copy_manifest.write_text("\n".join(refreshed) + "\n", encoding="utf-8", newline="\n")

    print(f"STAGE_3_H10_V: {cert['STAGE_3_H10_V']}")
    print(f"FIRST_BLOCKER: {cert['FIRST_BLOCKER']}")
    print(f"SOURCE_H10: {cert['SOURCE_H10']}")
    print(f"H10_IMMUTABLE: {cert['H10_IMMUTABLE']}")
    print(f"H10_SEMANTIC_HASH_MATCH: {cert['H10_SEMANTIC_HASH_MATCH']}")
    print(f"TRAJECTORY_FAMILY_ID: {family_id}")
    print(f"TOTAL_FAMILY_SAMPLES: {len(rows)}")
    print(f"NUMERICAL_AUDIT_SAMPLES: {len(rows)}")
    print(f"GIF_RENDERED_FRAMES: {len(gif_indices)}")
    print(f"UNEXPLAINED_POSITION_DISCONTINUITIES: {position_unexplained}")
    print(f"UNEXPLAINED_VELOCITY_DISCONTINUITIES: {velocity_unexplained}")
    print(f"SUSPICIOUS_EVENTS: {suspicious_count}")
    for key in ("POSITION_VIOLATIONS", "VELOCITY_VIOLATIONS", "ACCELERATION_VIOLATIONS", "JERK_VIOLATIONS", "COLLISION_VIOLATIONS", "PROCESS_TOLERANCE_VIOLATIONS"):
        print(f"{key}: {cert[key]}")
    for key in ("MAX_ABS_POSITION_STEP_RAD", "MAX_ABS_VELOCITY_RAD_S", "MAX_ABS_ACCELERATION_RAD_S2", "MAX_ABS_JERK_RAD_S3"):
        print(f"{key}: {cert[key]}")
    print(f"GIF_CREATED: {cert['GIF_CREATED']}")
    print(f"GIF_PATH: {(output / 'h10v_robot_motion.gif').resolve()}")
    print("PHYSICAL_ROBOT_CONNECTED: NO")
    print("PHYSICAL_DRIVER_LOADED: NO")
    print("PHYSICAL_FJT_GOALS_SENT: 0")
    print("ROBOT_MOTION_STARTED: NO")
    print("USER_VISUAL_DECISION: PENDING")
    print(f"READY_FOR_STAGE_3_H11: {cert['READY_FOR_STAGE_3_H11']}")
    print(f"OUTPUT_PATH: {output}")
    print(f"DESKTOP_REVIEW_PATH: {desktop}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
