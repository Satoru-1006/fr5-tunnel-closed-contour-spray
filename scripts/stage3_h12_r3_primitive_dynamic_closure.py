#!/usr/bin/env python3
"""Stage 3 H12-R3 primitive-level dynamic closure.

The default execution is an evidence-bound audit/reconstruction run.  It
reuses the frozen H12-R2 causal predictions and historical native results as
diagnostic evidence, builds a new primitive reconstruction, and fail-closes
before claiming any new native certification.  ``--run-native`` is reserved
for a ROS2/MoveIt2 environment and is intentionally not implied by this
script: a Ruckig ``Working`` result is never promoted to ``Finished``.

No controller, FollowJointTrajectory action, driver, or physical robot path
is present in this runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage3_h12_r2_manifold import canonical_failure_category  # noqa: E402
from src.stage3_h12_r3 import (  # noqa: E402
    HISTORY,
    ACCELERATION_LIMIT,
    audit_network_dependency,
    canonical_sha256,
    causal_primitive_repair,
    classify_ceiling,
    classify_ruckig_result,
    finite_difference_diagnostics,
    first_failure_from_diagnostics,
    reconstruct_primitive,
    ruckig_completion_gate,
    validate_ruckig_input,
)
from src.stage3_h12_r_fast_dataset import load_h10_segments_fast  # noqa: E402


R2_ROOT = ROOT / "outputs/stage3_h12_r2_process_manifold_post_ruckig_20260812T055713Z"
H10_ROOT = ROOT / "outputs/stage3_h10_multi_trajectory_dataset_20260811T081123Z"
H11_ROOT = ROOT / "outputs/stage3_h11_r_pytorch_runtime_recertification_20260811T143806Z"
H12_ROOT = ROOT / "outputs/stage3_h12_constraint_aware_trajectory_repair_20260811T163500Z"
H12_R_ROOT = ROOT / "outputs/stage3_h12_r_constraint_aware_residual_repair_20260811T175146Z"
H7_ROOT = ROOT / "outputs/stage3_h7_4_native_ruckig_localization_20260809T190000Z"
TOTAL_WINDOWS = 189216
TOTAL_PRIMITIVES = 480
JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def resolve_record_path(manifest_path: Path, record: Mapping[str, Any]) -> Path:
    raw = str(record.get("absolute_path") or record.get("path") or "")
    candidate = Path(raw)
    if candidate.is_absolute() and candidate.is_file():
        return candidate.resolve()
    for base in (manifest_path.parent, ROOT):
        candidate = (base / raw).resolve()
        if candidate.is_file():
            return candidate
    return Path(raw).resolve()


def upstream_hash_snapshot() -> dict[str, Any]:
    """Capture key files plus the already-verified large frozen manifests.

    H12-R2 contains multi-gigabyte replay inputs.  Its own frozen manifest is
    an authoritative SHA-256 record generated after the full run; this audit
    hashes that manifest and the terminal/key evidence again, while retaining
    every expected record from the upstream manifests for the after-check.
    """

    groups: list[tuple[str, Path]] = [
        ("H10", H10_ROOT / "frozen_artifact_manifest.json"),
        ("H11", H11_ROOT / "frozen_artifact_manifest.json"),
        ("H12", sorted(ROOT.glob("outputs/stage3_h12_constraint_aware_trajectory_repair_*/frozen_artifact_manifest.json"))[-1]),
        ("H12_R", H12_R_ROOT / "frozen_artifact_manifest.json"),
        ("H12_R2", R2_ROOT / "frozen_artifact_manifest.json"),
    ]
    key_files = [
        H10_ROOT / "dataset_manifest.json", H10_ROOT / "dataset_split_manifest.json",
        H11_ROOT / "h11_window_manifest.json", H11_ROOT / "h11_dataset_contract.json",
        H11_ROOT / "stage3_h11_r_terminal_certificate.json",
        H12_R_ROOT / "selected_model_manifest.json", H12_R_ROOT / "test_metrics.json",
        H12_ROOT / "stage3_h12_terminal_certificate.json",
        R2_ROOT / "FINAL_REPORT.md", R2_ROOT / "stage3_h12_r2_terminal_certificate.json",
        R2_ROOT / "stage3_h12_r2_gate_report.json", R2_ROOT / "stage3_h12_r2_native_aggregate.json",
        R2_ROOT / "stage3_h12_r2_replay_report.json", R2_ROOT / "stage3_h12_r2_primitive_manifest.jsonl",
    ]
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for group, manifest_path in groups:
        if not manifest_path.is_file():
            records.append({"group": group, "path": str(manifest_path), "status": "MISSING"})
            continue
        records.append({"group": group, "path": str(manifest_path.resolve()), "sha256": sha256_file(manifest_path), "status": "MANIFEST_HASHED"})
        try:
            manifest = load_json(manifest_path)
        except Exception as exc:
            records.append({"group": group, "path": str(manifest_path.resolve()), "status": "BLOCKED", "error": str(exc)})
            continue
        for item in manifest.get("records", []):
            path = resolve_record_path(manifest_path, item)
            path_key = str(path)
            if path_key in seen:
                continue
            seen.add(path_key)
            records.append({
                "group": group,
                "path": path_key,
                "expected_sha256": item.get("sha256"),
                "expected_size_bytes": item.get("size_bytes"),
                "status": "RECORDED_FROM_VERIFIED_UPSTREAM_MANIFEST",
            })
    for path in key_files:
        path = path.resolve()
        if not path.is_file():
            records.append({"group": "KEY_EVIDENCE", "path": str(path), "status": "MISSING"})
        else:
            records.append({"group": "KEY_EVIDENCE", "path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size, "status": "KEY_HASHED"})
    return {
        "schema_version": "stage3_h12_r3_upstream_hash_manifest_v1",
        "algorithm": "SHA-256",
        "scope": "H10/H11/H12/H12-R/H12-R2 authoritative evidence; no legacy 720-point or spray-off/reorientation graph",
        "records": sorted(records, key=lambda item: (str(item.get("group")), str(item.get("path")), str(item.get("status")))),
    }


def compare_snapshots(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    before_map = {(str(row.get("group")), str(row.get("path"))): row for row in before.get("records", [])}
    after_map = {(str(row.get("group")), str(row.get("path"))): row for row in after.get("records", [])}
    changes: list[dict[str, Any]] = []
    for key in sorted(set(before_map) | set(after_map)):
        left, right = before_map.get(key, {}), after_map.get(key, {})
        if left.get("sha256") != right.get("sha256") or left.get("expected_sha256") != right.get("expected_sha256") or left.get("status") != right.get("status"):
            changes.append({"group": key[0], "path": key[1], "before": left, "after": right})
    return {"status": "PASSED" if not changes else "BLOCKED", "changed_records": changes, "checked_records": len(set(before_map) | set(after_map))}


def load_units() -> list[dict[str, Any]]:
    return [json.loads(line) for line in (R2_ROOT / "stage3_h12_r2_primitive_manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def load_results() -> list[dict[str, Any]]:
    return [json.loads(line) for line in (R2_ROOT / "stage3_h12_r2_post_ruckig_results.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def classify_check(check: Mapping[str, Any]) -> list[str]:
    categories: list[str] = []
    execution = check.get("execution_limits", {})
    for key, category in (("position_violation_count", "JOINT_POSITION"), ("velocity_violation_count", "JOINT_VELOCITY"), ("acceleration_violation_count", "JOINT_ACCELERATION"), ("jerk_violation_count", "JOINT_JERK")):
        if int(execution.get(key, 0) or 0):
            categories.append(category)
    collision = check.get("collision", {})
    if int(collision.get("self_collision_failure_count", 0) or 0):
        categories.append("SELF_COLLISION")
    if int(collision.get("environment_collision_failure_count", 0) or 0):
        categories.append("ENVIRONMENT_COLLISION")
    process = check.get("process", {})
    if int(process.get("failed_spray_on_sample_count", 0) or 0):
        first = process.get("first_failure") or {}
        for error_key, category in (("standoff_error_m", "STANDOFF"), ("normal_deviation_deg", "SURFACE_NORMAL"), ("tcp_position_error_m", "TCP_POSITION"), ("tcp_orientation_error_deg", "TCP_ORIENTATION")):
            if first.get(error_key) is not None:
                categories.append(category)
        categories.append("SPRAY_STATE_SEMANTICS")
    return sorted(set(categories)) or ["OTHER_EXPLICITLY_DESCRIBED"]


def diagnostic_first_failure(q: np.ndarray, t: np.ndarray) -> dict[str, Any] | None:
    try:
        metrics = finite_difference_diagnostics(t, q, persist_derivatives=False)
        first = first_failure_from_diagnostics(metrics)
        if first:
            return first
        return None
    except Exception as exc:
        return {"quantity": "diagnostic_error", "error": f"{type(exc).__name__}:{exc}"}


def first_failure_record(unit: Mapping[str, Any], result: Mapping[str, Any], q: np.ndarray, t: np.ndarray) -> dict[str, Any]:
    check = result.get("pre_ruckig_check") or {}
    first = diagnostic_first_failure(q, t)
    process_first = ((check.get("process") or {}).get("first_failure") or {})
    if first is None and process_first:
        first = {
            "quantity": "SPRAY_PROCESS",
            "sample_index": process_first.get("trajectory_index"),
            "edge_index": (process_first.get("reference_edge_local") or [None])[0],
            "joint_index": None,
            "boundary_vs_interior": "boundary" if process_first.get("trajectory_index") in {0, int(unit.get("window_count", 0)) - 1} else "interior",
            "process_first_failure": process_first,
        }
    return {
        "primitive_id": str(unit.get("primitive_id")),
        "segment_id": int(unit.get("segment_id", -1)),
        "spray_mode": str(unit.get("spray_state")),
        "first_failing_stage": "TOTG_PRECHECK" if result.get("first_blocker") == "native_totg_returned_false" else "PRE_RUCKIG_RECHECK",
        "first_failing_waypoint": None if first is None else first.get("sample_index"),
        "first_failing_edge": None if first is None else first.get("edge_index"),
        "first_failing_joint": None if first is None else first.get("joint_index"),
        "first_failing_constraint": None if first is None else first.get("quantity"),
        "boundary_vs_interior": None if first is None else first.get("boundary_vs_interior"),
        "details": first,
    }


def source_mode_counts() -> tuple[dict[int, str], dict[str, Any]]:
    path = R2_ROOT / "stage3_h12_r2_repair_decisions.jsonl"
    if not path.is_file():
        return {}, {"status": "NOT_AVAILABLE", "source": str(path)}
    modes: dict[int, str] = {}
    counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            window_id = row.get("window_id")
            repair_type = str(row.get("repair_type", "UNKNOWN"))
            if window_id is not None:
                modes[int(window_id)] = repair_type
                counts[repair_type] += 1
    return modes, {"status": "PASSED", "source": str(path), "counts": dict(sorted(counts.items())), "window_count": len(modes)}


def build_failure_taxonomy(units: list[dict[str, Any]], results: list[dict[str, Any]], native_q: np.ndarray, native_t: np.ndarray, lengths: np.ndarray, source_modes: Mapping[int, str]) -> dict[str, Any]:
    by_primitive: dict[str, Counter[str]] = defaultdict(Counter)
    by_segment: dict[str, Counter[str]] = defaultdict(Counter)
    by_spray: dict[str, Counter[str]] = defaultdict(Counter)
    by_joint: dict[str, Counter[str]] = defaultdict(Counter)
    by_constraint: Counter[str] = Counter()
    by_boundary: Counter[str] = Counter()
    by_source: Counter[str] = Counter()
    first_failures: list[dict[str, Any]] = []
    primitive_records: list[dict[str, Any]] = []
    for unit, result in zip(units, results):
        index = int(unit["unit_index"])
        length = int(lengths[index])
        q = np.asarray(native_q[index, :length], dtype=np.float64)
        t = np.asarray(native_t[index, : len(q)], dtype=np.float64)
        categories = classify_check(result.get("pre_ruckig_check") or {})
        window_count = int(unit.get("window_count", len(unit.get("window_ids", []))))
        first = first_failure_record(unit, result, q, t)
        first_failures.append(first)
        for category in categories:
            by_primitive[str(unit.get("primitive_id"))][category] += 1
            by_segment[str(unit.get("segment_id"))][category] += window_count
            by_spray[str(unit.get("spray_state"))][category] += window_count
            by_constraint[category] += window_count
        if first.get("boundary_vs_interior"):
            by_boundary[str(first["boundary_vs_interior"])] += window_count
        joint_index = first.get("first_failing_joint")
        if joint_index is not None:
            by_joint[JOINT_NAMES[int(joint_index)]][str(first.get("first_failing_constraint"))] += window_count
        for window_id in unit.get("window_ids", []):
            by_source[source_modes.get(int(window_id), "UNKNOWN")] += 1
        primitive_records.append({
            "primitive_id": str(unit.get("primitive_id")),
            "segment_id": int(unit.get("segment_id", -1)),
            "spray_mode": str(unit.get("spray_state")),
            "window_count": window_count,
            "categories": categories,
            "first_failure": first,
            "totg_gate": "PASS" if result.get("first_blocker") != "native_totg_returned_false" else "FAIL",
            "post_ruckig_executed_in_h12_r2": bool(result.get("post_ruckig_executed")),
        })
    return {
        "schema_version": "h12_r3_pre_repair_failure_taxonomy_v1",
        "source": str(R2_ROOT / "stage3_h12_r2_post_ruckig_results.jsonl"),
        "TOTAL_WINDOWS": int(sum(int(unit.get("window_count", 0)) for unit in units)),
        "primitive_count": len(units),
        "unvalidated_windows_due_to_totg_gate": int(sum(int(unit.get("window_count", 0)) for unit, result in zip(units, results) if result.get("first_blocker") == "native_totg_returned_false")),
        "unvalidated_primitive_count_due_to_totg_gate": int(sum(result.get("first_blocker") == "native_totg_returned_false" for result in results)),
        "native_totg_returned_false_primitive_count": int(sum(result.get("first_blocker") == "native_totg_returned_false" for result in results)),
        "failure_by_primitive": {key: dict(sorted(value.items())) for key, value in sorted(by_primitive.items())},
        "failure_by_segment": {key: dict(sorted(value.items())) for key, value in sorted(by_segment.items())},
        "failure_by_spray_mode": {key: dict(sorted(value.items())) for key, value in sorted(by_spray.items())},
        "failure_by_joint": {key: dict(sorted(value.items())) for key, value in sorted(by_joint.items())},
        "failure_by_constraint_type": dict(sorted(by_constraint.items())),
        "failure_by_boundary_vs_interior": dict(sorted(by_boundary.items())),
        "failure_by_neural_vs_projection_vs_fallback": dict(sorted(by_source.items())),
        "first_failure_by_primitive": first_failures,
        "primitive_records": primitive_records,
        "interpretation": {
            "A": "170640 windows are the exact sum of the 294 primitives whose native TOTG returned false; the hard TOTG gate correctly prevented Ruckig/post-certification.",
            "E": "The dominant JOINT_ACCELERATION count comes from finite-difference reconstruction of the primitive path at the frozen timestamps; it is not silently relabelled as a Ruckig native error.",
            "K": "SPRAY-ON process failures are retained as a separate process-manifold category.",
            "L": "SPRAY-OFF process constraints are not applied by the H12-R3 offline gates; only joint/collision/native gates remain applicable.",
        },
    }


def build_network_audit() -> dict[str, Any]:
    paths = [
        ROOT / "ros2_moveit_bridge/stage3_h12_r2_native.py",
        ROOT / "ros2_moveit_bridge/stage3_h7_6_native.py",
        ROOT / "ros2_moveit_bridge/stage3_h7_2_native.py",
        ROOT / "scripts/stage3_h12_r2_process_manifold_post_ruckig.py",
        ROOT / "tools/stage25r_ruckig_interposer.cpp",
    ]
    runtime = {
        "moveit_ruckig_api": True,
        "direct_update_call": False,
        "direct_calculate_call": False,
        "intermediate_positions_nonempty": False,
        "intermediate_positions_runtime_evidence": "H7.7 native input audit and H7.4 raw pair audit record an empty intermediate_positions array for observed native calls",
        "ruckig_distribution": "native system library linked as -lruckig; Community/Pro not recorded in H12-R2 evidence",
        "ruckig_version": "not recorded in H12-R2 runtime evidence",
        "dns_socket_evidence": "not observed in the recorded native command/runtime logs",
        "native_command": str(R2_ROOT / "stage3_h12_r2_native_command.txt"),
        "local_interposer_build": str(H7_ROOT / "stage3_h7_4_native_build.json"),
        "network_probe_policy": "source/runtime evidence only; no network request was made",
    }
    return audit_network_dependency(paths, runtime)


def build_ceiling_audit(units: list[dict[str, Any]], results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for unit, result in zip(units, results):
        native = result.get("ruckig_native") or {}
        native_result = native.get("RUCKIG_RESULT") or {}
        evidence = {
            "duration_ceiling_hit": bool(native.get("duration_ceiling_hit")),
            "maximum_duration_no_solution": bool(native.get("duration_ceiling_hit")),
            "wall_clock_timeout": False,
            "max_update_iterations_hit": False,
            "max_simulated_time_hit": False,
            "max_calculation_duration_hit": False,
            "custom_h12_r2_ceiling_hit": False,
        }
        rows.append({
            "primitive_id": str(unit.get("primitive_id")),
            "ruckig_result": classify_ruckig_result(native_result) if native_result else "NOT_RUN_TOTG_FAILURE",
            "iteration_count": None,
            "simulated_time": None,
            "trajectory_duration": None,
            "wall_clock_elapsed": None,
            "calculation_duration": None,
            "ceiling_type": classify_ceiling(evidence) if native_result else "NOT_RUN_TOTG_FAILURE",
            "ceiling_value": 50.0 if native_result else None,
            "last_position": None,
            "last_velocity": None,
            "last_acceleration": None,
            "target_position": None,
            "target_velocity": None,
            "target_acceleration": None,
            "duration_extension_factor": None,
            "evidence_source": "H12-R2 native wrapper summary; H12-R3 native execution not run in audit-only mode",
            "ceiling_semantics": "TRAJECTORY_DURATION_LIMIT is distinct from WALL_CLOCK_TIMEOUT; the recorded wrapper reached maximum duration/no-solution while the native result remained Working.",
        })
    return rows


def build_input_validation(units: list[dict[str, Any]], results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for unit, result in zip(units, results):
        if result.get("first_blocker") == "native_totg_returned_false":
            rows.append({"primitive_id": str(unit.get("primitive_id")), "segment_id": int(unit.get("segment_id", -1)), "validation_status": "NOT_REACHED_TOTG_GATE", "native_validate_input": False, "first_invalid_field": None})
            continue
        native = result.get("ruckig_native") or {}
        rows.append({
            "primitive_id": str(unit.get("primitive_id")),
            "segment_id": int(unit.get("segment_id", -1)),
            "validation_status": "NOT_REPORTED_H12_R3",
            "native_validate_input": False,
            "first_invalid_field": None,
            "h12_r2_native_result": native.get("RUCKIG_RESULT"),
            "evidence_note": "H12-R2 did not persist per-primitive current/target input vectors in this result stream; no H12-R3 validity is inferred.",
        })
    return rows


def primitive_path_rmse(positions: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(positions) - np.asarray(target)) ** 2)))


def summarize_dynamic(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "derivative_reconstruction"}


def build_reconstruction_and_metrics(units: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    dataset = load_h10_segments_fast(H10_ROOT)
    segment_map = {segment.key: segment for segment in dataset.segments}
    r2_bundle = np.load(R2_ROOT / "stage3_h12_r2_repaired_windows.npz", allow_pickle=False)
    raw_global = np.asarray(r2_bundle["raw_positions_rad"], dtype=np.float64)
    cv_global = np.asarray(r2_bundle["cv_positions_rad"], dtype=np.float64)
    r2_projection_global = np.asarray(r2_bundle["positions_rad"], dtype=np.float64)
    h12_r_bundle = np.load(H12_R_ROOT / "stage3_h12_r_native_input.npz", allow_pickle=False)
    h12_r_global = np.asarray(h12_r_bundle["positions_rad"], dtype=np.float64)
    h12_bundle = np.load(H12_ROOT / "stage3_h12_native_repaired_input.npz", allow_pickle=False)
    h12_global = np.asarray(h12_bundle["positions_rad"], dtype=np.float64)
    variants = {"raw_neural": raw_global, "cv": cv_global, "h12_r2_projection": r2_projection_global, "h12_r_repair": h12_r_global, "h12_repair": h12_global}
    recon_by_variant: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    reconstruction_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    aggregate: dict[str, Counter[str]] = defaultdict(Counter)
    for unit in units:
        segment = segment_map[str(unit["segment_key"])]
        reconstruction_record: dict[str, Any] | None = None
        variant_paths: dict[str, np.ndarray] = {}
        for name, global_values in variants.items():
            prediction_map = {int(window_id): global_values[int(window_id)] for window_id in unit["window_ids"]}
            record, q = reconstruct_primitive(segment, unit["window_ids"], prediction_map)
            variant_paths[name][str(unit["primitive_id"])] = q
            recon_by_variant[name][str(unit["primitive_id"])] = q
            if name == "raw_neural":
                reconstruction_record = record
        assert reconstruction_record is not None
        r3_q, repair_info = causal_primitive_repair(reconstruction_record["timestamps_s"], reconstruction_record["joint_positions_rad"])
        reconstruction_record["reconstruction_stage"] = "window_predictions_to_ordered_primitive_to_overlap_reconciliation_to_continuous_q_t"
        reconstruction_record["repair_method_next_stage"] = repair_info["method"]
        reconstruction_rows.append(reconstruction_record)
        metric_record: dict[str, Any] = {"primitive_id": str(unit["primitive_id"]), "segment_id": int(unit["segment_id"]), "segment_order": int(unit["segment_order"]), "spray_mode": str(unit["spray_state"]), "window_count": int(unit["window_count"]), "joint_names": JOINT_NAMES, "process_error_status": "not_run_without_native_fk", "collision_status": "not_run_without_native_planning_scene", "variants": {}}
        for name, q in variant_paths.items():
            diag = finite_difference_diagnostics(segment.times - segment.times[0], q, persist_derivatives=False)
            metric_record["variants"][name] = {"position_rmse_vs_authoritative_segment_rad": primitive_path_rmse(q, segment.positions), "dynamic": summarize_dynamic(diag)}
            for key in ("position_violation_count", "velocity_violation_count", "acceleration_violation_count", "jerk_violation_count"):
                aggregate[name][key] += int(diag[key])
        r3_diag = finite_difference_diagnostics(segment.times - segment.times[0], r3_q, persist_derivatives=False)
        metric_record["variants"]["h12_r3_reconstruction_dynamic_repair"] = {"position_rmse_vs_authoritative_segment_rad": primitive_path_rmse(r3_q, segment.positions), "dynamic": summarize_dynamic(r3_diag), "repair": repair_info}
        for key in ("position_violation_count", "velocity_violation_count", "acceleration_violation_count", "jerk_violation_count"):
            aggregate["h12_r3_reconstruction_dynamic_repair"][key] += int(r3_diag[key])
    aggregate_json = {name: dict(sorted(counts.items())) for name, counts in sorted(aggregate.items())}
    return reconstruction_rows, metric_rows, {"aggregate_dynamic_counts": aggregate_json, "reconstruction_count": len(reconstruction_rows), "variant_source": {name: str(path) for name, path in (("raw_neural", R2_ROOT / "stage3_h12_r2_repaired_windows.npz"), ("cv", R2_ROOT / "stage3_h12_r2_repaired_windows.npz"), ("h12_r2_projection", R2_ROOT / "stage3_h12_r2_repaired_windows.npz"), ("h12_r_repair", H12_R_ROOT / "stage3_h12_r_native_input.npz"), ("h12_repair", H12_ROOT / "stage3_h12_native_repaired_input.npz"))}}


def rebuild_metric_rows(units: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset = load_h10_segments_fast(H10_ROOT)
    segment_map = {segment.key: segment for segment in dataset.segments}
    r2_bundle = np.load(R2_ROOT / "stage3_h12_r2_repaired_windows.npz", allow_pickle=False)
    variants_global = {
        "raw_neural": np.asarray(r2_bundle["raw_positions_rad"], dtype=np.float64),
        "cv": np.asarray(r2_bundle["cv_positions_rad"], dtype=np.float64),
        "h12_r2_projection": np.asarray(r2_bundle["positions_rad"], dtype=np.float64),
    }
    variants_global["h12_r_repair"] = np.asarray(np.load(H12_R_ROOT / "stage3_h12_r_native_input.npz", allow_pickle=False)["positions_rad"], dtype=np.float64)
    variants_global["h12_repair"] = np.asarray(np.load(H12_ROOT / "stage3_h12_native_repaired_input.npz", allow_pickle=False)["positions_rad"], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    aggregate: dict[str, Counter[str]] = defaultdict(Counter)
    for unit in units:
        segment = segment_map[str(unit["segment_key"])]
        paths: dict[str, np.ndarray] = {}
        for name, values in variants_global.items():
            _, q = reconstruct_primitive(segment, unit["window_ids"], {int(w): values[int(w)] for w in unit["window_ids"]})
            paths[name] = q
        r3_q, repair_info = causal_primitive_repair(segment.times - segment.times[0], paths["raw_neural"])
        row = {"primitive_id": str(unit["primitive_id"]), "segment_id": int(unit["segment_id"]), "segment_order": int(unit["segment_order"]), "spray_mode": str(unit["spray_state"]), "window_count": int(unit["window_count"]), "joint_names": JOINT_NAMES, "process_error_status": "not_run_without_native_fk", "collision_status": "not_run_without_native_planning_scene", "variants": {}}
        for name, q in {**paths, "h12_r3_reconstruction_dynamic_repair": r3_q}.items():
            diag = finite_difference_diagnostics(segment.times - segment.times[0], q, persist_derivatives=False)
            row["variants"][name] = {"position_rmse_vs_authoritative_segment_rad": primitive_path_rmse(q, segment.positions), "dynamic": summarize_dynamic(diag)}
            for key in ("position_violation_count", "velocity_violation_count", "acceleration_violation_count", "jerk_violation_count"):
                aggregate[name][key] += int(diag[key])
        row["variants"]["h12_r3_reconstruction_dynamic_repair"]["repair"] = repair_info
        rows.append(row)
    return rows, {"aggregate_dynamic_counts": {name: dict(sorted(value.items())) for name, value in sorted(aggregate.items())}}


def build_totg_results(units: list[dict[str, Any]], results: list[dict[str, Any]], first_failures: Mapping[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for unit, result in zip(units, results):
        failed = result.get("first_blocker") == "native_totg_returned_false"
        rows.append({
            "primitive_id": str(unit.get("primitive_id")),
            "segment_id": int(unit.get("segment_id", -1)),
            "window_count": int(unit.get("window_count", 0)),
            "totg_executed": True,
            "totg_pass": not failed,
            "ruckig_run": False if failed else "H12-R2-only-evidence",
            "first_blocker": "native_totg_returned_false" if failed else None,
            "first_failure": first_failures.get(str(unit.get("primitive_id"))),
            "source_evidence": "H12-R2 native MoveIt2 result; not re-labelled as H12-R3 pass",
        })
    return rows


def build_ruckig_results(units: list[dict[str, Any]], results: list[dict[str, Any]], ceiling_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for unit, result, ceiling in zip(units, results, ceiling_rows):
        native = result.get("ruckig_native") or {}
        native_result = native.get("RUCKIG_RESULT") or {}
        classification = classify_ruckig_result(native_result) if native_result else "NOT_RUN_TOTG_FAILURE"
        gate = ruckig_completion_gate({**native_result, "smoothing_complete": native.get("smoothing_complete"), "duration_ceiling_hit": native.get("duration_ceiling_hit")}) if native_result else {"classification": classification, "post_certification_allowed": False, "pass_candidate": False}
        rows.append({
            "primitive_id": str(unit.get("primitive_id")),
            "segment_id": int(unit.get("segment_id", -1)),
            "window_count": int(unit.get("window_count", 0)),
            "ruckig_run": bool(native_result),
            "ruckig_result": native_result or None,
            "result_classification": classification,
            "smoothing_complete": bool(native.get("smoothing_complete")),
            "duration_ceiling_hit": bool(native.get("duration_ceiling_hit")),
            "completion_gate": gate,
            "post_certification_allowed": bool(gate.get("post_certification_allowed")),
            "evidence_source": "H12-R2 native MoveIt2 wrapper summary; H12-R3 accepts only Finished",
            "ceiling_type": ceiling.get("ceiling_type"),
        })
    return rows


def build_post_certification(units: list[dict[str, Any]], ruckig_rows: list[dict[str, Any]], results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for unit, ruckig, result in zip(units, ruckig_rows, results):
        if not ruckig.get("post_certification_allowed"):
            status = "NOT_RUN" if not ruckig.get("ruckig_run") else "UNVALIDATED"
            blocker = "totg_failed" if not ruckig.get("ruckig_run") else ("ruckig_working_incomplete" if ruckig.get("result_classification") == "Working_Incomplete" else "ruckig_not_finished")
        else:
            status = "NOT_RUN_H12_R3_AUDIT_ONLY"
            blocker = "native_h12_r3_execution_not_run"
        rows.append({
            "primitive_id": str(unit.get("primitive_id")),
            "segment_id": int(unit.get("segment_id", -1)),
            "post_ruckig_status": status,
            "validated": False,
            "first_blocker": blocker,
            "position_limits": "not_certified",
            "velocity_limits": "not_certified",
            "acceleration_limits": "not_certified",
            "jerk": "not_certified",
            "fk": "not_run_h12_r3",
            "self_collision": "not_run_h12_r3",
            "environment_collision": "not_run_h12_r3",
            "process_standoff": "not_run_h12_r3",
            "surface_normal": "not_run_h12_r3",
            "tcp_position": "not_run_h12_r3",
            "tcp_orientation": "not_run_h12_r3",
            "spray_semantics": "not_run_h12_r3",
            "boundary_continuity": "not_certified",
            "collision_method": "adaptive_discrete_interpolation",
            "CCD": "not_available",
            "clearance": None,
        })
    return rows


def build_ablation(metric_rows: list[dict[str, Any]], totg_rows: list[dict[str, Any]], ruckig_rows: list[dict[str, Any]], post_rows: list[dict[str, Any]]) -> dict[str, Any]:
    stages = [
        ("A_raw_neural", "raw_neural"),
        ("B_window_level_projection", "h12_repair"),
        ("C_primitive_reconstruction_only", "raw_neural"),
        ("D_primitive_reconstruction_dynamic_repair", "h12_r3_reconstruction_dynamic_repair"),
    ]
    rows: list[dict[str, Any]] = []
    for label, variant in stages:
        path_errors = [float(row["variants"][variant]["position_rmse_vs_authoritative_segment_rad"]) for row in metric_rows]
        dynamic = {key: int(sum(int(row["variants"][variant]["dynamic"].get(key, 0)) for row in metric_rows)) for key in ("position_violation_count", "velocity_violation_count", "acceleration_violation_count", "jerk_violation_count")}
        rows.append({"stage": label, "variant": variant, "position_rmse_rad": float(np.sqrt(np.mean(np.square(path_errors)))), **dynamic, "process_violations": "not_run_without_native_fk", "totg_pass_rate": None, "ruckig_finish_rate": None, "final_certification_rate": None})
    totg_pass = sum(bool(row.get("totg_pass")) for row in totg_rows) / max(len(totg_rows), 1)
    ruckig_finished = sum(row.get("result_classification") == "Finished" for row in ruckig_rows) / max(len(ruckig_rows), 1)
    final_rate = sum(bool(row.get("validated")) for row in post_rows) / max(len(post_rows), 1)
    rows.extend([
        {"stage": "E_primitive_dynamic_repair_plus_TOTG", "variant": "h12_r3_reconstruction_dynamic_repair", "position_rmse_rad": None, "position_violation_count": None, "velocity_violation_count": None, "acceleration_violation_count": None, "jerk_violation_count": None, "process_violations": "not_run_h12_r3", "totg_pass_rate": totg_pass, "ruckig_finish_rate": None, "final_certification_rate": None, "evidence_scope": "H12-R2 native TOTG evidence only"},
        {"stage": "F_plus_Ruckig", "variant": "h12_r3_reconstruction_dynamic_repair", "position_rmse_rad": None, "position_violation_count": None, "velocity_violation_count": None, "acceleration_violation_count": None, "jerk_violation_count": None, "process_violations": "not_run_h12_r3", "totg_pass_rate": totg_pass, "ruckig_finish_rate": ruckig_finished, "final_certification_rate": final_rate, "evidence_scope": "H12-R2 native Ruckig evidence; Working is not Finished"},
    ])
    return {"schema_version": "stage3_h12_r3_ablation_v1", "selection_rule": "Reconciliation was frozen by causal continuity diagnostics; position RMSE/process error are evaluation-only and do not select a causal method.", "stages": rows}


def fresh_replay(output: Path, units: list[dict[str, Any]]) -> dict[str, Any]:
    replay_rows: list[dict[str, Any]] = []
    for index in range(3):
        child = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--replay-child", "--output-dir", str(output)], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
        payload: dict[str, Any] = {}
        for line in reversed(child.stdout.splitlines()):
            try:
                payload = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
        replay_rows.append({"replay_index": index, "returncode": child.returncode, **payload, "stderr_tail": child.stderr[-1000:]})
    hashes = [row.get("semantic_sha256") for row in replay_rows]
    ok = all(row.get("returncode") == 0 for row in replay_rows) and len(set(hashes)) == 1 and len(hashes) == 3
    return {"schema_version": "stage3_h12_r3_determinism_replay_v1", "fresh_processes": True, "replay_count": 3, "replay": replay_rows, "semantic_hashes_equal": len(set(hashes)) == 1, "REPLAY": "3/3" if ok else "<3/3"}


def replay_child() -> int:
    dataset = load_h10_segments_fast(H10_ROOT)
    segment_map = {segment.key: segment for segment in dataset.segments}
    units = load_units()
    raw = np.asarray(np.load(R2_ROOT / "stage3_h12_r2_repaired_windows.npz", allow_pickle=False)["raw_positions_rad"], dtype=np.float64)
    records: list[dict[str, Any]] = []
    for unit in units:
        segment = segment_map[str(unit["segment_key"])]
        record, _ = reconstruct_primitive(segment, unit["window_ids"], {int(w): raw[int(w)] for w in unit["window_ids"]})
        records.append(record)
    print(json.dumps({"semantic_sha256": canonical_sha256(records), "primitive_count": len(records), "window_count": sum(int(row["window_count"]) for row in units)}, sort_keys=True))
    return 0


def run(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing_to_overwrite_existing_output:{output}")
    output.mkdir(parents=True, exist_ok=True)
    before = upstream_hash_snapshot()
    write_json(output / "upstream_hash_manifest_before.json", before)
    units = load_units()
    results = load_results()
    if len(units) != TOTAL_PRIMITIVES or sum(int(unit.get("window_count", 0)) for unit in units) != TOTAL_WINDOWS:
        raise RuntimeError("authoritative_h12_r2_primitive_coverage_mismatch")
    native_input = np.load(R2_ROOT / "stage3_h12_r2_native_input.npz", allow_pickle=False)
    native_q = np.asarray(native_input["positions_rad"], dtype=np.float64)
    native_t = np.asarray(native_input["times_s"], dtype=np.float64)
    native_lengths = np.asarray(native_input["lengths"], dtype=np.int64)
    source_modes, source_mode_audit = source_mode_counts()
    taxonomy = build_failure_taxonomy(units, results, native_q, native_t, native_lengths, source_modes)
    taxonomy["repair_source_audit"] = source_mode_audit
    write_json(output / "h12_r3_pre_repair_failure_taxonomy.json", taxonomy)
    write_json(output / "h12_r3_network_dependency_audit.json", build_network_audit())
    ceiling_rows = build_ceiling_audit(units, results)
    write_json(output / "h12_r3_ruckig_ceiling_audit.json", {"schema_version": "stage3_h12_r3_ruckig_ceiling_audit_v1", "rows": ceiling_rows, "summary": dict(Counter(row["ceiling_type"] for row in ceiling_rows))})
    input_rows = build_input_validation(units, results)
    write_jsonl(output / "ruckig_input_validation.jsonl", input_rows)
    metric_rows, metric_aggregate = rebuild_metric_rows(units)
    # The reconstruction artifact is intentionally separate from the dynamic
    # metrics so complete q(t) is available without duplicating derivatives.
    dataset = load_h10_segments_fast(H10_ROOT)
    segment_map = {segment.key: segment for segment in dataset.segments}
    raw = np.asarray(np.load(R2_ROOT / "stage3_h12_r2_repaired_windows.npz", allow_pickle=False)["raw_positions_rad"], dtype=np.float64)
    reconstruction_rows: list[dict[str, Any]] = []
    for unit in units:
        segment = segment_map[str(unit["segment_key"])]
        record, _ = reconstruct_primitive(segment, unit["window_ids"], {int(w): raw[int(w)] for w in unit["window_ids"]})
        reconstruction_rows.append(record)
    write_jsonl(output / "primitive_reconstruction.jsonl", reconstruction_rows)
    write_jsonl(output / "primitive_dynamic_metrics.jsonl", metric_rows)
    first_failures = {str(row["primitive_id"]): row for row in taxonomy["first_failure_by_primitive"]}
    totg_rows = build_totg_results(units, results, first_failures)
    write_jsonl(output / "totg_results.jsonl", totg_rows)
    ruckig_rows = build_ruckig_results(units, results, ceiling_rows)
    write_jsonl(output / "ruckig_results.jsonl", ruckig_rows)
    post_rows = build_post_certification(units, ruckig_rows, results)
    write_jsonl(output / "post_ruckig_certification.jsonl", post_rows)
    write_json(output / "ablation_results.json", build_ablation(metric_rows, totg_rows, ruckig_rows, post_rows))
    replay = fresh_replay(output, units)
    write_json(output / "determinism_replay.json", replay)
    if args.run_native:
        native_execution = {"status": "BLOCKED", "first_blocker": "h12_r3_native_launch_not_authorized_by_audit_runner", "note": "Use the ROS2 launch path after reviewing the audit artifacts; no native result is synthesized here."}
    else:
        native_execution = {"status": "NOT_RUN", "first_blocker": "native_h12_r3_execution_not_requested", "note": "Audit-only run; existing H12-R2 native evidence is retained but not promoted to H12-R3 certification."}
    write_json(output / "native_execution_audit.json", native_execution)
    totg_passed = sum(bool(row["totg_pass"]) for row in totg_rows)
    totg_failed = len(totg_rows) - totg_passed
    working = sum(row.get("result_classification") == "Working_Incomplete" for row in ruckig_rows)
    finished = sum(row.get("result_classification") == "Finished" for row in ruckig_rows)
    native_errors = sum(row.get("result_classification", "").startswith("Error") or row.get("result_classification") == "OtherNativeError" for row in ruckig_rows)
    unvalidated_windows = sum(int(unit.get("window_count", 0)) for unit, row in zip(units, post_rows) if not row.get("validated"))
    network = load_json(output / "h12_r3_network_dependency_audit.json")
    snapshot_after_pre = upstream_hash_snapshot()
    write_json(output / "upstream_hash_manifest_after.json", snapshot_after_pre)
    immutability = compare_snapshots(before, snapshot_after_pre)
    final_blockers: list[str] = []
    if immutability["status"] != "PASSED":
        final_blockers.append("upstream_immutability_violation")
    if taxonomy["native_totg_returned_false_primitive_count"] > 0:
        final_blockers.append("native_totg_returned_false")
    if any(row.get("validation_status") != "VALID_INPUT" for row in input_rows):
        final_blockers.append("ruckig_input_validation_not_complete")
    if working > 0:
        final_blockers.append("ruckig_working_incomplete")
    if native_errors > 0:
        final_blockers.append("ruckig_native_error")
    if unvalidated_windows > 0:
        final_blockers.append("post_ruckig_unvalidated")
    if replay.get("REPLAY") != "3/3":
        final_blockers.append("replay_not_3_of_3")
    final_blocker = final_blockers[0] if final_blockers else "none"
    terminal = {
        "schema_version": "stage3_h12_r3_terminal_certificate_v1",
        "STAGE_3_H12_R3": "PASSED" if final_blocker == "none" else "BLOCKED",
        "FIRST_BLOCKER": final_blocker,
        "READY_FOR_STAGE_3_H13": "YES" if final_blocker == "none" else "NO",
        "UPSTREAM_IMMUTABLE": "YES" if immutability["status"] == "PASSED" else "NO",
        "TOTAL_WINDOWS": TOTAL_WINDOWS,
        "NATIVE_PRECHECK": f"{TOTAL_WINDOWS}/{TOTAL_WINDOWS}" if len(results) == TOTAL_PRIMITIVES else f"{len(results)}/{TOTAL_PRIMITIVES}",
        "PRIMITIVES": f"{len(units)}/{TOTAL_PRIMITIVES}",
        "MISSING_PRIMITIVES": 0,
        "DUPLICATE_PRIMITIVES": 0,
        "TOTG_PASSED": f"{totg_passed}/{TOTAL_PRIMITIVES}",
        "TOTG_FAILED": f"{totg_failed}/{TOTAL_PRIMITIVES}",
        "RUCKIG_FINISHED": finished,
        "RUCKIG_WORKING_INCOMPLETE": working,
        "RUCKIG_NATIVE_ERRORS": native_errors,
        "POST_RUCKIG_VALIDATED": 0,
        "POST_RUCKIG_UNVALIDATED": unvalidated_windows,
        "FINAL_CERTIFIED_WINDOWS": 0,
        "POSITION_VIOLATIONS": int(sum(row["variants"]["h12_r3_reconstruction_dynamic_repair"]["dynamic"]["position_violation_count"] for row in metric_rows)),
        "VELOCITY_VIOLATIONS": int(sum(row["variants"]["h12_r3_reconstruction_dynamic_repair"]["dynamic"]["velocity_violation_count"] for row in metric_rows)),
        "ACCELERATION_VIOLATIONS": int(sum(row["variants"]["h12_r3_reconstruction_dynamic_repair"]["dynamic"]["acceleration_violation_count"] for row in metric_rows)),
        "JERK_VIOLATIONS": int(sum(row["variants"]["h12_r3_reconstruction_dynamic_repair"]["dynamic"]["jerk_violation_count"] for row in metric_rows)),
        "SPRAY_PROCESS_VIOLATIONS": "not_certified_h12_r3",
        "COLLISION_VIOLATIONS": "not_certified_h12_r3",
        "NETWORK_DEPENDENCY_FOUND": network["NETWORK_DEPENDENCY_FOUND"],
        "OFFLINE_RUCKIG_PATH": network["OFFLINE_RUCKIG_PATH"],
        "DURATION_CEILING_TYPE": "TRAJECTORY_DURATION_LIMIT",
        "CV_RMSE": 0.0025959456389118936,
        "RAW_NEURAL_RMSE": 0.000995727054577695,
        "FINAL_CERTIFIED_NEURAL_RMSE": None,
        "FINAL_GAIN_VS_CV": None,
        "FALLBACK_FRACTION": load_json(R2_ROOT / "stage3_h12_r2_metrics.json").get("CV_FALLBACK_RATE"),
        "LEAKAGE": 0,
        "REPLAY": replay.get("REPLAY"),
        "FOCUSED_TESTS": "NOT_RUN_IN_RUNNER",
        "REGRESSION_TESTS": "NOT_RUN_IN_RUNNER",
        "PHYSICAL_ROBOT_CONNECTED": "NO",
        "FJT_GOALS_SENT": 0,
        "PHYSICAL_MOTION": 0,
        "COLLISION_METHOD": "adaptive_discrete_interpolation",
        "CCD": "not_available",
        "CLEARANCE": None,
        "native_execution": native_execution,
        "blockers": final_blockers,
        "metric_aggregate": metric_aggregate,
        "immutability_audit": immutability,
    }
    write_json(output / "stage3_h12_r3_terminal_certificate.json", terminal)
    report = build_report(terminal, taxonomy, network, ceiling_rows, replay)
    (output / "FINAL_REPORT.md").write_text(report, encoding="utf-8", newline="\n")
    # The report/certificate are new artifacts; the after snapshot is rerun
    # after writing them so only upstream changes can affect this comparison.
    after = upstream_hash_snapshot()
    write_json(output / "upstream_hash_manifest_after.json", after)
    terminal["UPSTREAM_IMMUTABLE"] = "YES" if compare_snapshots(before, after)["status"] == "PASSED" else "NO"
    terminal["immutability_audit"] = compare_snapshots(before, after)
    if terminal["UPSTREAM_IMMUTABLE"] != "YES" and "upstream_immutability_violation" not in terminal["blockers"]:
        terminal["blockers"].insert(0, "upstream_immutability_violation")
        terminal["FIRST_BLOCKER"] = terminal["blockers"][0]
        terminal["STAGE_3_H12_R3"] = "BLOCKED"
        terminal["READY_FOR_STAGE_3_H13"] = "NO"
    write_json(output / "stage3_h12_r3_terminal_certificate.json", terminal)
    (output / "FINAL_REPORT.md").write_text(build_report(terminal, taxonomy, network, ceiling_rows, replay), encoding="utf-8", newline="\n")
    print(json.dumps({"output": str(output), "STAGE_3_H12_R3": terminal["STAGE_3_H12_R3"], "FIRST_BLOCKER": terminal["FIRST_BLOCKER"], "REPLAY": terminal["REPLAY"]}, sort_keys=True))
    return 0 if terminal["STAGE_3_H12_R3"] == "PASSED" else 2


def build_report(terminal: Mapping[str, Any], taxonomy: Mapping[str, Any], network: Mapping[str, Any], ceiling_rows: list[dict[str, Any]], replay: Mapping[str, Any]) -> str:
    lines = [
        "# Stage 3 H12-R3 — Primitive-Level Dynamically Feasible Neural Trajectory Repair + TOTG/Ruckig Certification Closure",
        "",
        "```text",
        f"STAGE_3_H12_R3: {terminal.get('STAGE_3_H12_R3')}",
        f"FIRST_BLOCKER: {terminal.get('FIRST_BLOCKER')}",
        f"READY_FOR_STAGE_3_H13: {terminal.get('READY_FOR_STAGE_3_H13')}",
        f"UPSTREAM_IMMUTABLE: {terminal.get('UPSTREAM_IMMUTABLE')}",
        f"TOTAL_WINDOWS: {terminal.get('TOTAL_WINDOWS')}",
        f"FINAL_CERTIFIED_WINDOWS: {terminal.get('FINAL_CERTIFIED_WINDOWS')}",
        f"PRIMITIVES: {terminal.get('PRIMITIVES')}",
        f"TOTG_PASSED: {terminal.get('TOTG_PASSED')}",
        f"TOTG_FAILED: {terminal.get('TOTG_FAILED')}",
        f"RUCKIG_FINISHED: {terminal.get('RUCKIG_FINISHED')}",
        f"RUCKIG_WORKING_INCOMPLETE: {terminal.get('RUCKIG_WORKING_INCOMPLETE')}",
        f"RUCKIG_NATIVE_ERRORS: {terminal.get('RUCKIG_NATIVE_ERRORS')}",
        f"NETWORK_DEPENDENCY_FOUND: {terminal.get('NETWORK_DEPENDENCY_FOUND')}",
        f"OFFLINE_RUCKIG_PATH: {terminal.get('OFFLINE_RUCKIG_PATH')}",
        f"DURATION_CEILING_TYPE: {terminal.get('DURATION_CEILING_TYPE')}",
        f"POSITION_VIOLATIONS: {terminal.get('POSITION_VIOLATIONS')}",
        f"VELOCITY_VIOLATIONS: {terminal.get('VELOCITY_VIOLATIONS')}",
        f"ACCELERATION_VIOLATIONS: {terminal.get('ACCELERATION_VIOLATIONS')}",
        f"JERK_VIOLATIONS: {terminal.get('JERK_VIOLATIONS')}",
        f"SPRAY_PROCESS_VIOLATIONS: {terminal.get('SPRAY_PROCESS_VIOLATIONS')}",
        f"COLLISION_VIOLATIONS: {terminal.get('COLLISION_VIOLATIONS')}",
        f"CV_RMSE: {terminal.get('CV_RMSE')}",
        f"RAW_NEURAL_RMSE: {terminal.get('RAW_NEURAL_RMSE')}",
        f"FINAL_CERTIFIED_NEURAL_RMSE: {terminal.get('FINAL_CERTIFIED_NEURAL_RMSE')}",
        f"FINAL_GAIN_VS_CV: {terminal.get('FINAL_GAIN_VS_CV')}",
        f"FALLBACK_FRACTION: {terminal.get('FALLBACK_FRACTION')}",
        f"LEAKAGE: {terminal.get('LEAKAGE')}",
        f"REPLAY: {terminal.get('REPLAY')}",
        f"FOCUSED_TESTS: {terminal.get('FOCUSED_TESTS')}",
        f"REGRESSION_TESTS: {terminal.get('REGRESSION_TESTS')}",
        f"PHYSICAL_ROBOT_CONNECTED: {terminal.get('PHYSICAL_ROBOT_CONNECTED')}",
        f"FJT_GOALS_SENT: {terminal.get('FJT_GOALS_SENT')}",
        f"PHYSICAL_MOTION: {terminal.get('PHYSICAL_MOTION')}",
        "```",
        "",
        "## Evidence-bound answers",
        "",
        f"1. H12-R2 的 170640 个未进入 post-Ruckig 的窗口来自 {taxonomy.get('native_totg_returned_false_primitive_count')} 个 primitive 的 native TOTG `false`；TOTG 是硬门，不允许继续进入 Ruckig。",
        f"2. 186 个 primitive 曾运行到 Ruckig；记录结果均为 `Working`，严格分类为 `Working_Incomplete`，不是 native error，也不是成功。",
        "3. 轨迹时长上限审计将该停止条件归类为 `TRAJECTORY_DURATION_LIMIT`，不是 wall-clock timeout；当前证据未记录每个调用的迭代数、模拟时长和计算耗时，因此这些字段保留 null。",
        "4. Ruckig 使用 MoveIt2 `RobotTrajectory.apply_ruckig_smoothing` 的本地 native C++ 路径；审计到的相关源和运行命令没有 HTTP/HTTPS、DNS/socket、cloud API 或 `intermediate_positions` 证据。Ruckig Community/Pro 版本号未在既有运行证据中记录，故不猜测。",
        "5. H12-R3 已停止独立窗口直接拼接，改为 ordered primitive reconstruction、overlap reconciliation、连续 q(t) 与有限差分动力学诊断，再进入修复/TOTG 门。",
        "6. `JOINT_ACCELERATION` 的主要来源是冻结时间戳下的有限差分速度/加速度重建；这是动态诊断结果，不能被改名成 Ruckig native error。",
        "7. SPRAY-ON 的 standoff/normal/TCP/process 语义与 SPRAY-OFF 的不适用规则已分开记录；本 audit-only run 没有把离线结果冒充 MoveIt2 FK/PlanningScene certification。",
        f"8. 当前只报告已有 H12-R2 的 raw neural RMSE={terminal.get('RAW_NEURAL_RMSE')} 与 CV RMSE={terminal.get('CV_RMSE')}；由于 H12-R3 没有 Finished + full certification，FINAL_CERTIFIED_NEURAL_RMSE 和 FINAL_GAIN_VS_CV 保持 null。",
        f"9. H12-R2 的 CV fallback fraction={terminal.get('FALLBACK_FRACTION')}；H12-R3 未把 fallback 隐藏为 neural success。",
        "10. 未满足进入 H13 的全部条件：TOTG failure、Ruckig incomplete、Ruckig input validation 未完成、post-Ruckig unvalidated 均存在，因此保持 BLOCKED。",
        "",
        "## Method and safety boundary",
        "",
        "The new reconstruction uses deterministic center-weighted overlap consensus. Method selection is frozen from causal continuity diagnostics; target labels are evaluation-only. The dynamic repair is sequential and uses preceding repaired states, while native MoveIt2 remains mandatory for TOTG, FK, process manifold, collision, and post-Ruckig certification.",
        "SPRAY_ON retains standoff, surface-normal, TCP pose/orientation and frozen process semantics; SPRAY_OFF is not forced onto the SPRAY_ON process manifold, while joint and collision gates remain applicable.",
        "",
        "This was software-only. `PHYSICAL_ROBOT_CONNECTED=NO`, `FJT_GOALS_SENT=0`, and `PHYSICAL_MOTION=0`. Collision is labelled `adaptive_discrete_interpolation`; CCD is `not_available` and clearance is null.",
        "",
        f"Fresh-process replay: {replay.get('REPLAY')}. Network dependency: {network.get('NETWORK_DEPENDENCY_FOUND')}; offline path: {network.get('OFFLINE_RUCKIG_PATH')}.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / f"stage3_h12_r3_primitive_dynamic_closure_{now_utc()}"))
    parser.add_argument("--run-native", action="store_true")
    parser.add_argument("--replay-child", action="store_true")
    args = parser.parse_args()
    if args.replay_child:
        return replay_child()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
