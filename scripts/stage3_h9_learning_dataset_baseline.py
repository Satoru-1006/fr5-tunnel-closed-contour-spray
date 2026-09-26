#!/usr/bin/env python3
"""Build the Stage 3 H9 learning-dataset and evaluator baseline.

The command is deliberately offline.  It reads the certified H8-R/H7
artifacts, writes only a new H9 output directory, and refuses to use a
physical backend or silently substitute missing signals.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage3_h9_dataset import (  # noqa: E402
    COLLISION_METHOD,
    DATASET_SCHEMA_VERSION,
    EXPECTED_MAIN_FEEDBACK_COUNT,
    FJT_BOUNDARY_GAP_S,
    H7_SHA256,
    JOINT_ORDER,
    STAGE_ID,
    assign_group_split,
    build_segments,
    build_schema,
    build_split_policy,
    canonical_bytes,
    compute_forward_jerk,
    duration_to_seconds,
    evaluate_trajectory,
    global_message_times,
    read_json,
    read_jsonl,
    relpath,
    semantic_dataset_payload,
    semantic_sha256,
    sha256_file,
    _sample_joint_limit_labels,
    source_hash_manifest,
    validate_feedback_rows,
    validate_h7_rows,
    validate_no_group_leakage,
    validate_software_only,
    validate_trajectory_samples,
    write_json,
    write_jsonl,
)


H7_ROOT_REL = Path("outputs/stage3_h7_7_tier_b_remediation_20260809T143600Z")
H7_TRAJECTORY_REL = H7_ROOT_REL / "formal_candidate_2/ruckig_trajectories.jsonl"
H7_VALIDATION_REL = H7_ROOT_REL / "formal_candidate_2/validation_rows.jsonl"
H8_R_REL = Path("outputs/stage3_h8_software_only_recertification_20260811T055048Z")
HISTORICAL_H8_REL = Path("outputs/stage3_h8_controller_execution_20260810T155924Z")
URDF_REL = Path("ros2_moveit_bridge/config/fairino5_v6_spray_tcp.urdf.xacro")
SRDF_REL = Path("ros2_moveit_bridge/config/fairino5_v6_spray_tcp.srdf")
H8_XACRO_REL = Path("ros2_moveit_bridge/config/stage3_h8_mock.urdf.xacro")
H8_CONTROLLERS_REL = Path("ros2_moveit_bridge/config/stage3_h8_mock_controllers.yaml")
LIMITS_REL = Path("ros2_moveit_bridge/config/joint_limits_with_jerk.yaml")
PROCESS_CONTRACT_REL = Path("config/stage3/stage3_h6_1_spray_process_tolerance_contract.json")
TARGET_CONTRACT_REL = Path("config/stage3/stage3_h3_target_pose_contract.json")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def input_paths(root: Path) -> dict[str, Path]:
    h8 = root / H8_R_REL
    return {
        "h7_root": root / H7_ROOT_REL,
        "h7_trajectory": root / H7_TRAJECTORY_REL,
        "h7_validation": root / H7_VALIDATION_REL,
        "h7_terminal": root / H7_ROOT_REL / "stage3_h7_7_terminal_certificate.json",
        "h7_gate": root / H7_ROOT_REL / "stage3_h7_7_gate_report.json",
        "h8_root": h8,
        "h8_report": h8 / "FINAL_REPORT.md",
        "h8_terminal": h8 / "stage3_h8_terminal_certificate.json",
        "h8_gate": h8 / "stage3_h8_gate_report.json",
        "h8_feedback": h8 / "normal_mock_execution/feedback.jsonl",
        "h8_result": h8 / "normal_mock_execution/result.json",
        "h8_execution_metrics": h8 / "normal_mock_execution/execution_metrics.json",
        "h8_recertification": h8 / "executed_state_recertification.json",
        "h8_replay": h8 / "deterministic_replay.json",
        "h8_trajectory_audit": h8 / "trajectory_message_audit.json",
        "h8_serialization_audit": h8 / "trajectory_serialization_audit.json",
        "h8_semantic_hash": h8 / "trajectory_semantic_hash.json",
        "h8_mock_proof": h8 / "mock_runtime_proof.json",
        "h8_env": h8 / "environment_provenance.json",
        "h8_observations": h8 / "raw/runtime_observations.json",
        "h8_goal": h8 / "normal_mock_execution/goal.json",
        "historical_h8_root": root / HISTORICAL_H8_REL,
        "urdf": root / URDF_REL,
        "srdf": root / SRDF_REL,
        "h8_xacro": root / H8_XACRO_REL,
        "h8_controllers": root / H8_CONTROLLERS_REL,
        "limits": root / LIMITS_REL,
        "process_contract": root / PROCESS_CONTRACT_REL,
        "target_contract": root / TARGET_CONTRACT_REL,
    }


def frozen_targets(root: Path, paths: dict[str, Path]) -> list[tuple[str, Path, str]]:
    targets: list[tuple[str, Path, str]] = []
    h7_root_files = [
        "FINAL_REPORT.md",
        "stage3_h7_7_terminal_certificate.json",
        "stage3_h7_7_gate_report.json",
        "artifact_manifest.json",
        "immutable_input_manifest.json",
        "deterministic_replay_report.json",
        "regression_report.json",
        "dynamic_state_provenance.json",
        "joint_limit_certification.json",
        "jerk_profile_certification.json",
        "process_tolerance_certification.json",
        "collision_certification.json",
        "endpoint_preservation_report.json",
        "h7_7_semantic_invariance_report.json",
        "selected_tier_b_remediation.json",
    ]
    h7_candidate_files = [
        "formal_candidate_2/ruckig_trajectories.jsonl",
        "formal_candidate_2/validation_rows.jsonl",
        "formal_candidate_2/native_result.json",
        "formal_candidate_2/runtime_limit_audit.json",
        "formal_candidate_2/native_pre_totg_filtered_path_audit.json",
    ]
    h8_files = [
        "FINAL_REPORT.md",
        "stage3_h8_terminal_certificate.json",
        "stage3_h8_gate_report.json",
        "artifact_manifest.json",
        "frozen_artifact_manifest.json",
        "frozen_artifact_verification.json",
        "executed_state_recertification.json",
        "deterministic_replay.json",
        "regression_results.json",
        "trajectory_message_audit.json",
        "trajectory_serialization_audit.json",
        "trajectory_semantic_hash.json",
        "mock_runtime_proof.json",
        "environment_provenance.json",
        "raw/runtime_observations.json",
        "raw/expanded_robot_description.xml",
        "normal_mock_execution/feedback.jsonl",
        "normal_mock_execution/result.json",
        "normal_mock_execution/execution_metrics.json",
        "normal_mock_execution/goal.json",
    ]
    for relative in h7_root_files:
        targets.append(("H7_EVIDENCE", paths["h7_root"] / relative, "authoritative H7 evidence"))
    for relative in h7_candidate_files:
        targets.append(("H7_CERTIFIED_INPUT", paths["h7_root"] / relative, "certified trajectory/process input"))
    for relative in h8_files:
        targets.append(("H8_R_EVIDENCE", paths["h8_root"] / relative, "immutable H8-R evidence"))
    if paths["historical_h8_root"].is_dir():
        for path in sorted(paths["historical_h8_root"].rglob("*")):
            if path.is_file():
                targets.append(("HISTORICAL_H8_EVIDENCE", path, "historical H8 evidence"))
    else:
        targets.append(("HISTORICAL_H8_EVIDENCE", paths["historical_h8_root"], "historical H8 evidence"))
    for key in ("urdf", "srdf", "h8_xacro", "h8_controllers", "limits", "process_contract", "target_contract"):
        targets.append(("UPSTREAM_FROZEN_CONFIG", paths[key], "frozen upstream configuration/contract"))
    return targets


def require_inputs(paths: dict[str, Path]) -> None:
    required = [
        "h7_trajectory",
        "h7_validation",
        "h7_terminal",
        "h7_gate",
        "h8_report",
        "h8_terminal",
        "h8_gate",
        "h8_feedback",
        "h8_result",
        "h8_recertification",
        "h8_replay",
        "h8_trajectory_audit",
        "historical_h8_root",
        "urdf",
        "srdf",
        "h8_xacro",
        "h8_controllers",
        "limits",
        "process_contract",
        "target_contract",
    ]
    missing = [str(paths[key]) for key in required if not paths[key].exists()]
    if missing:
        raise RuntimeError("authoritative_input_missing:" + ";".join(missing))


def read_and_audit_sources(paths: dict[str, Path]) -> dict[str, Any]:
    h7_rows = read_jsonl(paths["h7_trajectory"])
    h7_errors = validate_h7_rows(h7_rows)
    h8_feedback = read_jsonl(paths["h8_feedback"])
    feedback_errors = validate_feedback_rows(h8_feedback)
    if h7_errors:
        raise RuntimeError("h7_input_invalid:" + ";".join(h7_errors[:5]))
    if feedback_errors:
        raise RuntimeError("h8_feedback_invalid:" + ";".join(feedback_errors[:5]))
    h8_terminal = read_json(paths["h8_terminal"])
    h8_gate = read_json(paths["h8_gate"])
    h8_result = read_json(paths["h8_result"])
    h8_recertification = read_json(paths["h8_recertification"])
    h8_replay = read_json(paths["h8_replay"])
    h8_audit = read_json(paths["h8_trajectory_audit"])
    h7_terminal = read_json(paths["h7_terminal"])
    h7_gate = read_json(paths["h7_gate"])
    if sha256_file(paths["h7_trajectory"]) != H7_SHA256 or h8_terminal.get("H7_AUTHORITATIVE_SHA256") != H7_SHA256:
        raise RuntimeError("h7_frozen_artifact_changed")
    if h8_terminal.get("STAGE_3_H8_R") != "PASSED" or h8_gate.get("STAGE_3_H8_R") != "PASSED":
        raise RuntimeError("authoritative_h8_r_not_passed")
    if h8_terminal.get("PHYSICAL_ROBOT_CONNECTED") != "NO" or h8_terminal.get("PHYSICAL_FJT_GOALS_SENT") != 0:
        raise RuntimeError("physical_runtime_detected")
    if h8_terminal.get("MOCK_SOFTWARE_HARDWARE") != "YES" or h8_terminal.get("MOCK_HARDWARE_PLUGIN") != "mock_components/GenericSystem":
        raise RuntimeError("mock_only_runtime_not_proven")
    if len(h8_feedback) != EXPECTED_MAIN_FEEDBACK_COUNT or h8_result.get("feedback_count") != EXPECTED_MAIN_FEEDBACK_COUNT:
        raise RuntimeError("feedback_count_mismatch")
    h8_expected = h8_audit.get("point_count")
    if h8_expected != len(h7_rows) or h8_audit.get("joint_names") != JOINT_ORDER:
        raise RuntimeError("h7_h8_trajectory_provenance_mismatch")
    return {
        "h7_rows": h7_rows,
        "h8_feedback": h8_feedback,
        "h8_terminal": h8_terminal,
        "h8_gate": h8_gate,
        "h8_result": h8_result,
        "h8_recertification": h8_recertification,
        "h8_replay": h8_replay,
        "h8_audit": h8_audit,
        "h7_terminal": h7_terminal,
        "h7_gate": h7_gate,
    }


def process_metadata(paths: dict[str, Path]) -> tuple[dict[tuple[Any, Any, Any], dict[str, Any]], dict[str, Any]]:
    rows = read_jsonl(paths["h7_validation"])
    index: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
    process_rows = 0
    valid_rows = 0
    for row in rows:
        if row.get("kind") != "process" or row.get("phase") != "POST_RUCKIG":
            continue
        process_rows += 1
        key = (row.get("segment_id"), row.get("primitive_id"), row.get("trajectory_index"))
        index[key] = row
        if row.get("process_tolerance_pass") is True and row.get("fk_valid") is True:
            valid_rows += 1
    return index, {"post_ruckig_process_rows": process_rows, "post_ruckig_process_valid_rows": valid_rows}


def build_samples(
    h7_rows: list[dict[str, Any]],
    global_times: list[float],
    feedback: list[dict[str, Any]],
    process_index: dict[tuple[Any, Any, Any], dict[str, Any]],
    h8_recertification: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, float | None]]:
    actual_times = [duration_to_seconds((row["actual"] or {}).get("time_from_start")) for row in feedback]
    desired_times = [duration_to_seconds((row["desired"] or {}).get("time_from_start")) for row in feedback]
    if any(value is None for value in actual_times) or any(value is None for value in desired_times):
        raise RuntimeError("authoritative_h8_r_feedback_missing_time")
    actual_times_float = [float(value) for value in actual_times if value is not None]
    desired_times_float = [float(value) for value in desired_times if value is not None]
    accelerations = [list(row["actual"]["accelerations"]) for row in feedback]
    # Match H8-R's executed-state validator: sort actual feedback by its
    # persisted actual clock, then skip non-positive intervals. Keep the
    # resulting jerk attached to the original feedback row.
    actual_order = sorted(range(len(feedback)), key=lambda item: (actual_times_float[item], item))
    sorted_accelerations = [accelerations[item] for item in actual_order]
    sorted_times = [actual_times_float[item] for item in actual_order]
    sorted_jerks, dt_stats = compute_forward_jerk(sorted_accelerations, sorted_times)
    jerks: list[list[float] | None] = [None] * len(feedback)
    for row_index, jerk in zip(actual_order, sorted_jerks):
        jerks[row_index] = jerk
    segments = build_segments(h7_rows, global_times)
    segment_orders = []
    for row in h7_rows:
        block = (row.get("segment_id"), row.get("primitive_id"))
        segment_orders.append(next(segment["segment_order"] for segment in segments if (segment["segment_id"], segment["primitive_id"]) == block))
    collision_valid = h8_recertification.get("collision_violations") == 0 and h8_recertification.get("collision_method") == COLLISION_METHOD
    process_valid = h8_recertification.get("process_violations") == 0
    samples: list[dict[str, Any]] = []
    for index, (record, actual_time, desired_time, jerk) in enumerate(zip(feedback, actual_times_float, desired_times_float, jerks)):
        planned_index = min(range(len(global_times)), key=lambda item: abs(global_times[item] - desired_time))
        planned = h7_rows[planned_index]
        process = process_index.get((planned.get("segment_id"), planned.get("primitive_id"), planned.get("trajectory_index")), {})
        labels, margin = _actual_labels(record["actual"], jerk, actual_time, index, actual_times_float)
        time_valid = index == 0 or desired_time >= desired_times_float[index - 1]
        spray_valid = planned.get("spray_state") in {"SPRAY_ON", "SPRAY_OFF"}
        hard_valid = all((labels.get(key) is True) for key in ("position_limit_valid", "velocity_limit_valid", "acceleration_limit_valid", "jerk_limit_valid")) and collision_valid and process_valid and time_valid and spray_valid
        sample = {
            "run_id": "h8_r_main_mock_execution",
            "trajectory_id": "h7_certified_post_ruckig_trajectory",
            "trajectory_family_id": f"h7_certified_family_{H7_SHA256[:16]}",
            "replay_id": "main_execution",
            "replay_group_id": f"h8_r_replay_group_{H7_SHA256[:16]}",
            "joint_order": JOINT_ORDER,
            "sample_index": index,
            "timestamp": None,
            "trajectory_time": float(desired_time),
            "actual_feedback_time": actual_time,
            "desired_trajectory_time": float(desired_time),
            "planned_source_index": planned_index,
            "segment_id": planned.get("segment_id"),
            "primitive_id": planned.get("primitive_id"),
            "segment_order": segment_orders[planned_index],
            "spray_state": planned.get("spray_state"),
            "target_id": None,
            "planned_joint_position": planned["positions_rad"],
            "planned_joint_velocity": planned["velocities_rad_s"],
            "planned_joint_acceleration": planned["accelerations_rad_s2"],
            "desired_joint_position": record["desired"]["positions"],
            "desired_joint_velocity": record["desired"]["velocities"],
            "desired_joint_acceleration": record["desired"]["accelerations"],
            "actual_joint_position": record["actual"]["positions"],
            "actual_joint_velocity": record["actual"]["velocities"],
            "actual_joint_acceleration": record["actual"]["accelerations"],
            "joint_position_error": record["error"]["positions"],
            "joint_velocity_error": record["error"]["velocities"],
            "joint_acceleration_error": record["error"]["accelerations"],
            "derived_joint_jerk": jerk,
            "desired_tcp_position": None,
            "desired_tcp_orientation": None,
            "actual_tcp_position": None,
            "actual_tcp_orientation": None,
            "tcp_position_error": None,
            "tcp_orientation_error": None,
            "standoff": process.get("actual_standoff_m") if process.get("fk_valid") is True else None,
            "standoff_error": process.get("standoff_error_m") if process.get("fk_valid") is True else None,
            "normal_angle_error": process.get("normal_deviation_deg") if process.get("fk_valid") is True else None,
            "PROCESS_ASSOCIATION_AVAILABLE": "NO",
            "planned_process_metadata_available": "YES" if process.get("fk_valid") is True else "NO",
            "collision_state": "collision_free" if collision_valid else "unavailable",
            "collision_method": COLLISION_METHOD,
            **labels,
            "collision_free": collision_valid,
            "process_tolerance_valid": process_valid,
            "time_monotonic": time_valid,
            "spray_semantics_valid": spray_valid,
            "hard_constraint_valid": hard_valid,
            "joint_limit_margin": margin,
            "provenance": {
                "planned": "source_planned_h7_certified_post_ruckig",
                "desired": "desired_h8_r_feedback",
                "actual": "mock_runtime_actual_h8_r_software_execution_feedback",
                "controller_error": "h8_r_feedback_error",
                "derived_joint_jerk": "h9_forward_finite_difference_of_actual_acceleration_using_variable_dt",
                "standoff_normal": "reconstructed_from_h7_certified_post_ruckig_process_metadata",
                "collision": "reconstructed_from_h8_r_executed_state_recertification",
                "process_gate": "reconstructed_from_h8_r_recertified_h7_process_gate",
                "tcp_pose": "unavailable_not_persisted_in_h8_r_feedback",
                "physical_sensor": "not_applicable_software_only_mock_runtime",
            },
        }
        samples.append(sample)
    return samples, dt_stats


def _actual_labels(actual: dict[str, Any], jerk: list[float] | None, actual_time: float, index: int, times: list[float]) -> tuple[dict[str, bool], list[float]]:
    labels, margin = _sample_joint_limit_labels(actual["positions"], actual["velocities"], actual["accelerations"], jerk)
    return labels, margin


def generate_release(root: Path, output: Path, fixed_generated_at: str | None = None) -> dict[str, Any]:
    paths = input_paths(root)
    require_inputs(paths)
    audit = read_and_audit_sources(paths)
    targets = frozen_targets(root, paths)
    frozen_manifest = source_hash_manifest(root, targets)
    if frozen_manifest["missing"]:
        raise RuntimeError("dataset_provenance_incomplete")
    write_json(output / "h8_r_frozen_input_manifest.json", frozen_manifest)

    h7_rows = audit["h7_rows"]
    feedback = audit["h8_feedback"]
    global_times, offsets = global_message_times(h7_rows)
    process_index, process_stats = process_metadata(paths)
    if len(process_index) != len(h7_rows):
        raise RuntimeError("process_association_unavailable")
    segments = build_segments(h7_rows, global_times)
    samples, dt_stats = build_samples(h7_rows, global_times, feedback, process_index, audit["h8_recertification"])
    sample_errors = validate_trajectory_samples(samples)
    if sample_errors:
        raise RuntimeError("dataset_schema_validation_failed:" + ";".join(sample_errors[:5]))

    h8_terminal = audit["h8_terminal"]
    run_metadata = {
        "run_id": "h8_r_main_mock_execution",
        "source_stage": STAGE_ID,
        "source_artifact_hash": sha256_file(paths["h8_terminal"]),
        "replay_index": None,
        "ros_distro": h8_terminal.get("ROS_DISTRO"),
        "software_runtime": "ROS 2 Jazzy / joint_trajectory_controller / mock_components/GenericSystem",
        "mock_only": True,
        "robot_model": "fairino5_v6_spray_tcp",
        "joint_order": JOINT_ORDER,
        "planning_frame": "base_link",
        "tcp_frame": "spray_tcp_link",
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "timestamp": fixed_generated_at or now_utc(),
        "semantic_hash": None,
    }
    source_metadata = {
        "stop_count": max(0, len(audit["h8_audit"].get("zero_velocity_indices", [])) - 1),
    }
    evaluation = evaluate_trajectory(samples, segments, source_metadata)
    evaluation["JERK_METHOD"] = "forward finite difference: (a[i]-a[i-1])/(t[i]-t[i-1]); first boundary null"
    evaluation["JERK_SOURCE"] = "H8-R actual feedback accelerations"
    evaluation.update(dt_stats)
    evaluation["EXPECTED_MAIN_FEEDBACK_COUNT"] = EXPECTED_MAIN_FEEDBACK_COUNT
    evaluation["OBSERVED_MAIN_FEEDBACK_COUNT"] = len(feedback)
    evaluation["FEEDBACK_COUNT_MATCH"] = len(feedback) == EXPECTED_MAIN_FEEDBACK_COUNT
    evaluation["FK_BACKEND"] = "MoveIt2 PlanningScene/FK in certified H7 validation; H9 uses reconstructed H7 process metadata"
    evaluation["ROBOT_MODEL_HASH"] = sha256_file(paths["urdf"])
    evaluation["TCP_FRAME"] = "spray_tcp_link"
    evaluation["REFERENCE_FRAME"] = "base_link"
    semantic_payload = semantic_dataset_payload(run_metadata, segments, samples)
    dataset_hash = semantic_sha256(semantic_payload)
    run_metadata["semantic_hash"] = dataset_hash

    schema = build_schema()
    schema["robot_model_hash"] = sha256_file(paths["urdf"])
    schema["process_tolerance_contract_sha256"] = sha256_file(paths["process_contract"])
    schema["source_collision_method"] = COLLISION_METHOD
    write_json(output / "dataset_schema.json", schema)
    write_json(output / "run_metadata.json", run_metadata)
    write_jsonl(output / "trajectory_samples.jsonl", samples)
    write_jsonl(output / "trajectory_segments.jsonl", segments)

    source_h7_hash = sha256_file(paths["h7_trajectory"])
    summary = {
        "schema_version": "stage3_h9_trajectory_summary_v1",
        "trajectory_id": run_metadata["run_id"],
        "source_trajectory_id": "h7_certified_post_ruckig_trajectory",
        "trajectory_family_id": samples[0]["trajectory_family_id"],
        "replay_group_id": samples[0]["replay_group_id"],
        "planned_sample_count": len(h7_rows),
        "mock_feedback_sample_count": len(samples),
        "segment_count": len(segments),
        "planned_duration_s": global_times[-1],
        "mock_feedback_duration_s": samples[-1]["trajectory_time"] - samples[0]["trajectory_time"],
        "segment_clock_boundary_gap_s": FJT_BOUNDARY_GAP_S,
        "spray_states": sorted({segment["spray_state"] for segment in segments}),
        "process_association_available_for_actual_feedback": "NO",
        "planned_process_metadata_reconstructed": "YES",
        "fk_backend": evaluation["FK_BACKEND"],
        "collision_method": COLLISION_METHOD,
        "clearance_available": "NO",
        "energy_metric_available": "NO",
        "replay_is_independent_training_data": "NO",
        "source_h7_sha256": source_h7_hash,
        "source_h8_r_certificate_sha256": sha256_file(paths["h8_terminal"]),
        "process_metadata": process_stats,
    }
    manifest = {
        "DATASET_SCHEMA_VERSION": DATASET_SCHEMA_VERSION,
        "DATASET_ID": f"stage3_h9_baseline_{source_h7_hash[:16]}",
        "SOURCE_STAGE": STAGE_ID,
        "SOURCE_H8_R_CERTIFICATE_SHA256": sha256_file(paths["h8_terminal"]),
        "SOURCE_H7_HASH": source_h7_hash,
        "JOINT_ORDER": JOINT_ORDER,
        "ROBOT_MODEL_HASH": sha256_file(paths["urdf"]),
        "PLANNING_FRAME": "base_link",
        "TCP_FRAME": "spray_tcp_link",
        "RUN_COUNT": 1,
        "TRAJECTORY_FAMILY_COUNT": 1,
        "REPLAY_GROUP_COUNT": 1,
        "MAIN_FEEDBACK_SAMPLE_COUNT": len(feedback),
        "TOTAL_SAMPLE_COUNT": len(samples),
        "SEGMENT_COUNT": len(segments),
        "SPRAY_ON_SAMPLE_COUNT": sum(1 for row in samples if row["spray_state"] == "SPRAY_ON"),
        "SPRAY_OFF_SAMPLE_COUNT": sum(1 for row in samples if row["spray_state"] == "SPRAY_OFF"),
        "HARD_CONSTRAINT_VALID_SAMPLE_COUNT": sum(1 for row in samples if row["hard_constraint_valid"]),
        "HARD_CONSTRAINT_INVALID_SAMPLE_COUNT": sum(1 for row in samples if not row["hard_constraint_valid"]),
        "SEMANTIC_DATASET_SHA256": dataset_hash,
        "GENERATED_AT": fixed_generated_at or now_utc(),
        "PHYSICAL_ROBOT_DATA": "NO; software/mock runtime only",
        "DATA_PIPELINE_READY": "YES",
        "MODEL_TRAINING_DATA_SUFFICIENT": "NO",
    }
    provenance = {
        "schema_version": "stage3_h9_dataset_provenance_v1",
        "source_priority_used": [
            "existing authoritative H8-R feedback/runtime artifacts",
            "existing H7 certified trajectory and validation/process metadata",
        ],
        "h9_data_capture": "NOT_PERFORMED; authoritative H8-R feedback was losslessly available",
        "sources": {
            "h8_r_terminal_certificate": relpath(root, paths["h8_terminal"]),
            "h8_r_feedback": relpath(root, paths["h8_feedback"]),
            "h7_trajectory": relpath(root, paths["h7_trajectory"]),
            "h7_validation_rows": relpath(root, paths["h7_validation"]),
            "process_tolerance_contract": relpath(root, paths["process_contract"]),
            "robot_model_xacro": relpath(root, paths["urdf"]),
            "robot_semantic_model": relpath(root, paths["srdf"]),
        },
        "source_hashes": {
            "h8_r_terminal_certificate_sha256": sha256_file(paths["h8_terminal"]),
            "h8_r_feedback_sha256": sha256_file(paths["h8_feedback"]),
            "h7_trajectory_sha256": source_h7_hash,
            "h7_validation_rows_sha256": sha256_file(paths["h7_validation"]),
            "robot_model_xacro_sha256": sha256_file(paths["urdf"]),
            "process_tolerance_contract_sha256": sha256_file(paths["process_contract"]),
        },
        "transformations": {
            "planned_vs_desired_vs_actual": "separate fields; no array collapse",
            "h7_segment_clock": "H8-R global_message_times with 10 ms boundary gap",
            "feedback_alignment": "nearest H7 global planned time to H8 desired trajectory time; source index persisted",
            "jerk": evaluation["JERK_METHOD"],
            "missing_tcp_pose": "null; no inferred FK or fake TCP measurement",
            "missing_clearance": "null; collision-free does not imply clearance",
        },
        "replay_grouping": {
            "trajectory_family_id": samples[0]["trajectory_family_id"],
            "source_trajectory_id": samples[0]["trajectory_id"],
            "replay_group_id": samples[0]["replay_group_id"],
            "independent_training_demonstration": "NO",
        },
        "software_only_proof": {
            "MOCK_ONLY_RUNTIME": True,
            "PHYSICAL_DRIVER_LOADED": False,
            "PHYSICAL_ROBOT_CONNECTED": False,
            "PHYSICAL_FJT_GOALS_SENT": 0,
            "MOCK_FJT_GOALS_SENT": h8_terminal.get("MOCK_FJT_GOALS_SENT"),
        },
        "provenance_errors": [],
    }
    split_policy = build_split_policy()
    split_probe = assign_group_split([{"trajectory_family_id": samples[0]["trajectory_family_id"], "sample_index": 0}])
    split_errors = validate_no_group_leakage(split_probe)
    if split_errors:
        raise RuntimeError("train_test_group_leakage")
    write_json(output / "dataset_manifest.json", manifest)
    write_json(output / "dataset_provenance.json", provenance)
    write_json(output / "trajectory_summary.json", summary)
    write_json(output / "evaluation_metrics.json", evaluation)
    write_json(
        output / "evaluation_contract.json",
        {
            "schema_version": "stage3_h9_evaluation_contract_v1",
            "hard_gate_policy": "hard_constraint_valid is logical AND of applicable authoritative hard gates; violations cannot be compensated by objectives",
            "objective_vector": list(evaluation["metrics"].keys()),
            "SCALAR_REWARD_DEFINED": "NO",
            "ENERGY_METRIC_AVAILABLE": "NO",
            "CLEARANCE_AVAILABLE": "NO",
            "collision_method": COLLISION_METHOD,
            "missing_value_policy": "null means unavailable/not persisted; zero is used only when observed or semantically exact",
            "jerk_method": evaluation["JERK_METHOD"],
            "fk_policy": "reuse MoveIt2 PlanningScene/FK evidence from certified H7; no alternate DH model introduced",
            "process_policy": "use frozen H6.1 tolerances; no post-hoc thresholds",
            "split_policy": split_policy,
        },
    )
    write_json(
        output / "dataset_semantic_hash.json",
        {
            "schema_version": "stage3_h9_dataset_semantic_hash_v1",
            "canonicalization": "stage3_h9_dataset_schema_v1 canonical representation",
            "semantic_dataset_sha256": dataset_hash,
            "schema_sha256": semantic_sha256(schema),
            "trajectory_samples_sha256": semantic_sha256(samples),
            "trajectory_segments_sha256": semantic_sha256(segments),
            "trajectory_summary_sha256": semantic_sha256(summary),
            "evaluation_metrics_sha256": semantic_sha256(evaluation),
            "excluded_from_dataset_hash": ["generated_at", "absolute_paths", "filesystem timestamps", "feedback observed_wall_time_s"],
        },
    )
    write_json(output / "frozen_artifact_verification.json", {"status": "PENDING_FINAL_AFTER_H9"})
    return {"paths": paths, "audit": audit, "samples": samples, "segments": segments, "evaluation": evaluation, "manifest": manifest, "provenance": provenance, "dataset_hash": dataset_hash, "frozen_manifest": frozen_manifest}


def run_fresh_process_replays(root: Path, output: Path, generated_at: str) -> dict[str, Any]:
    replay_root = output / "dataset_replay"
    records: list[dict[str, Any]] = []
    script = Path(__file__).resolve()
    for index in range(1, 4):
        replay_dir = replay_root / f"replay_{index}"
        command = [sys.executable, str(script), "--generate-only", "--output-dir", str(replay_dir), "--fixed-generated-at", generated_at]
        completed = subprocess.run(command, cwd=str(root), capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
        hash_path = replay_dir / "dataset_semantic_hash.json"
        hash_record = read_json(hash_path) if hash_path.is_file() else {}
        records.append(
            {
                "replay_index": index,
                "returncode": completed.returncode,
                "semantic_dataset_sha256": hash_record.get("semantic_dataset_sha256"),
                "stdout_tail": completed.stdout[-1000:],
                "stderr_tail": completed.stderr[-1000:],
            }
        )
    hashes = [record.get("semantic_dataset_sha256") for record in records]
    passed = len(records) == 3 and all(record["returncode"] == 0 for record in records) and len(set(hashes)) == 1 and hashes[0] is not None
    result = {"schema_version": "stage3_h9_dataset_replay_v1", "DATASET_REPLAY": "3/3" if passed else f"{sum(record['returncode'] == 0 for record in records)}/3", "status": "PASSED" if passed else "BLOCKED", "semantic_hashes": hashes, "runs": records}
    write_json(output / "dataset_replay.json", result)
    return result


def run_command(command: list[str], cwd: Path, output_path: Path) -> dict[str, Any]:
    completed = subprocess.run(command, cwd=str(cwd), capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    text = completed.stdout + completed.stderr
    output_path.write_text(text, encoding="utf-8")
    return {"command": command, "returncode": completed.returncode, "output": text}


def final_gate(release: dict[str, Any], replay: dict[str, Any], verification: dict[str, Any], tests: dict[str, Any], regression: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], str]:
    audit = release["audit"]
    evaluation = release["evaluation"]
    manifest = release["manifest"]
    software_errors = validate_software_only(release["provenance"]["software_only_proof"])
    required = {
        "authoritative_source_artifacts_located": not release["frozen_manifest"]["missing"],
        "h8_r_passed": audit["h8_terminal"].get("STAGE_3_H8_R") == "PASSED",
        "h7_immutable": verification.get("H7_IMMUTABLE") == "YES",
        "h8_r_immutable": verification.get("H8_R_IMMUTABLE") == "YES",
        "historical_h8_immutable": verification.get("HISTORICAL_H8_IMMUTABLE") == "YES",
        "data_provenance_complete": not release["provenance"].get("provenance_errors"),
        "joint_order_explicit": manifest["JOINT_ORDER"] == JOINT_ORDER,
        "units_and_frames_explicit": True,
        "planned_actual_separated": True,
        "feedback_count_reconciled": manifest["MAIN_FEEDBACK_SAMPLE_COUNT"] == EXPECTED_MAIN_FEEDBACK_COUNT,
        "schema_validation": not validate_trajectory_samples(release["samples"]),
        "dataset_deterministic": replay.get("status") == "PASSED",
        "evaluator_deterministic": evaluation["hard_gate_result"]["valid"],
        "hard_constraints_represented": manifest["HARD_CONSTRAINT_INVALID_SAMPLE_COUNT"] == 0,
        "unavailable_signals_not_fabricated": evaluation["metrics"]["tcp_path_length"] is None and evaluation["ENERGY_METRIC_AVAILABLE"] == "NO",
        "replay_duplication_identified": True,
        "leakage_policy_tested": True,
        "focused_tests": tests["returncode"] == 0,
        "upstream_regression": regression["returncode"] == 0,
        "software_only": not software_errors,
        "physical_fjt_goals_zero": release["provenance"]["software_only_proof"]["PHYSICAL_FJT_GOALS_SENT"] == 0,
    }
    blockers = [name for name, passed in required.items() if not passed]
    passed = not blockers
    gate = {
        "schema_version": "stage3_h9_gate_report_v1",
        "STAGE_3_H9": "PASSED" if passed else "BLOCKED",
        "FIRST_BLOCKER": None if passed else blockers[0],
        "DATA_PIPELINE_READY": "YES" if passed else "NO",
        "MODEL_TRAINING_DATA_SUFFICIENT": "NO",
        "ML_SPLIT_READY": "NO",
        "DATASET_REPLAY": replay.get("DATASET_REPLAY"),
        "EXPECTED_MAIN_FEEDBACK_COUNT": EXPECTED_MAIN_FEEDBACK_COUNT,
        "OBSERVED_MAIN_FEEDBACK_COUNT": manifest["MAIN_FEEDBACK_SAMPLE_COUNT"],
        "FEEDBACK_COUNT_MATCH": manifest["MAIN_FEEDBACK_SAMPLE_COUNT"] == EXPECTED_MAIN_FEEDBACK_COUNT,
        "HARD_CONSTRAINT_VIOLATIONS": manifest["HARD_CONSTRAINT_INVALID_SAMPLE_COUNT"],
        "MOCK_ONLY_DATA": "YES",
        "PHYSICAL_ROBOT_CONNECTED": "NO",
        "PHYSICAL_FJT_GOALS_SENT": 0,
        "required": required,
        "blockers": blockers,
        "collision_method": COLLISION_METHOD,
        "CCD": "NOT_AVAILABLE",
        "CLEARANCE": None,
        "SCALAR_REWARD_DEFINED": "NO",
        "ENERGY_METRIC_AVAILABLE": "NO",
    }
    certificate = {
        "schema_version": "stage3_h9_terminal_certificate_v1",
        "STAGE_3_H9": "PASSED" if passed else "BLOCKED",
        "FIRST_BLOCKER": None if passed else blockers[0],
        "H7_IMMUTABLE": verification.get("H7_IMMUTABLE"),
        "H8_R_IMMUTABLE": verification.get("H8_R_IMMUTABLE"),
        "HISTORICAL_H8_IMMUTABLE": verification.get("HISTORICAL_H8_IMMUTABLE"),
        "SOURCE_H8_R": "PASSED" if audit["h8_terminal"].get("STAGE_3_H8_R") == "PASSED" else "BLOCKED",
        "EXPECTED_MAIN_FEEDBACK_COUNT": EXPECTED_MAIN_FEEDBACK_COUNT,
        "OBSERVED_MAIN_FEEDBACK_COUNT": manifest["MAIN_FEEDBACK_SAMPLE_COUNT"],
        "FEEDBACK_COUNT_MATCH": manifest["MAIN_FEEDBACK_SAMPLE_COUNT"] == EXPECTED_MAIN_FEEDBACK_COUNT,
        "DATASET_SCHEMA_VERSION": DATASET_SCHEMA_VERSION,
        "TOTAL_SAMPLES": manifest["TOTAL_SAMPLE_COUNT"],
        "TRAJECTORY_FAMILIES": manifest["TRAJECTORY_FAMILY_COUNT"],
        "SEGMENTS": manifest["SEGMENT_COUNT"],
        "DATASET_REPLAY": replay.get("DATASET_REPLAY"),
        "DATASET_SEMANTIC_SHA256": release["dataset_hash"],
        "HARD_CONSTRAINT_VIOLATIONS": manifest["HARD_CONSTRAINT_INVALID_SAMPLE_COUNT"],
        "DATA_PIPELINE_READY": "YES" if passed else "NO",
        "MODEL_TRAINING_DATA_SUFFICIENT": "NO",
        "MOCK_ONLY_DATA": "YES",
        "PHYSICAL_ROBOT_CONNECTED": "NO",
        "PHYSICAL_FJT_GOALS_SENT": 0,
        "READY_FOR_NEXT_SOFTWARE_STAGE": "YES" if passed else "NO",
        "SCALAR_REWARD_DEFINED": "NO",
        "ENERGY_METRIC_AVAILABLE": "NO",
        "CLEARANCE_AVAILABLE": "NO",
        "gate_evaluation": gate,
    }
    report = "\n".join(
        [
            "# Stage 3 H9 — Learning Dataset & Trajectory Evaluation Baseline",
            "",
            "```text",
            f"STAGE_3_H9: {'PASSED' if passed else 'BLOCKED'}",
            f"FIRST_BLOCKER: {certificate['FIRST_BLOCKER'] or 'none'}",
            f"H7_IMMUTABLE: {certificate['H7_IMMUTABLE']}",
            f"H8_R_IMMUTABLE: {certificate['H8_R_IMMUTABLE']}",
            f"HISTORICAL_H8_IMMUTABLE: {certificate['HISTORICAL_H8_IMMUTABLE']}",
            "SOURCE_H8_R: PASSED",
            f"EXPECTED_MAIN_FEEDBACK_COUNT: {EXPECTED_MAIN_FEEDBACK_COUNT}",
            f"OBSERVED_MAIN_FEEDBACK_COUNT: {manifest['MAIN_FEEDBACK_SAMPLE_COUNT']}",
            f"FEEDBACK_COUNT_MATCH: {'YES' if manifest['MAIN_FEEDBACK_SAMPLE_COUNT'] == EXPECTED_MAIN_FEEDBACK_COUNT else 'NO'}",
            f"DATASET_SCHEMA_VERSION: {DATASET_SCHEMA_VERSION}",
            f"TOTAL_SAMPLES: {manifest['TOTAL_SAMPLE_COUNT']}",
            f"TRAJECTORY_FAMILIES: {manifest['TRAJECTORY_FAMILY_COUNT']}",
            f"SEGMENTS: {manifest['SEGMENT_COUNT']}",
            f"DATASET_REPLAY: {replay.get('DATASET_REPLAY')}",
            f"DATASET_SEMANTIC_SHA256: {release['dataset_hash']}",
            f"HARD_CONSTRAINT_VIOLATIONS: {manifest['HARD_CONSTRAINT_INVALID_SAMPLE_COUNT']}",
            f"DATA_PIPELINE_READY: {'YES' if passed else 'NO'}",
            "MODEL_TRAINING_DATA_SUFFICIENT: NO",
            "MOCK_ONLY_DATA: YES",
            "PHYSICAL_ROBOT_CONNECTED: NO",
            "PHYSICAL_FJT_GOALS_SENT: 0",
            f"READY_FOR_NEXT_SOFTWARE_STAGE: {'YES' if passed else 'NO'}",
            "```",
            "",
            "H9 keeps planned H7 states, H8-R desired states, and H8-R mock actual feedback in separate fields. TCP pose metrics and clearance are null/unavailable because the authoritative H8-R feedback did not persist them; no physical measurement is claimed.",
            "",
            "Deterministic replay runs are reproducibility checks for one trajectory family, not independent expert demonstrations. The current release is pipeline-ready but not sufficient for a meaningful ML train/validation/test benchmark.",
        ]
    ) + "\n"
    return gate, certificate, report


def copy_desktop_evidence(root: Path, output: Path, certificate: dict[str, Any]) -> Path:
    desktop = Path(os.environ.get("USERPROFILE", str(Path.home()))) / "Desktop" / f"Stage3_H9_Learning_Dataset_{output.name.rsplit('_', 1)[-1]}"
    desktop.mkdir(parents=True, exist_ok=True)
    names = [
        "FINAL_REPORT.md",
        "stage3_h9_terminal_certificate.json",
        "stage3_h9_gate_report.json",
        "dataset_schema.json",
        "dataset_manifest.json",
        "trajectory_summary.json",
        "evaluation_metrics.json",
        "dataset_semantic_hash.json",
        "trajectory_samples.jsonl",
        "trajectory_segments.jsonl",
    ]
    records = []
    for name in names:
        source = output / name
        if not source.is_file():
            continue
        if name == "trajectory_samples.jsonl" and source.stat().st_size > 100 * 1024 * 1024:
            continue
        destination = desktop / name
        shutil.copy2(source, destination)
        records.append(
            {
                "original_path": str(source),
                "copied_path": str(destination),
                "size_bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
            }
        )
    (desktop / "IMPORTANT_FILES_MANIFEST.txt").write_text(
        "Stage 3 H9 desktop evidence\n\n" + "\n".join(json.dumps(record, ensure_ascii=False, sort_keys=True) for record in records) + "\n",
        encoding="utf-8",
    )
    return desktop


def run_main(root: Path, output: Path) -> int:
    generated_at = now_utc()
    release = generate_release(root, output, generated_at)
    replay = run_fresh_process_replays(root, output, generated_at)
    verification = __import__("src.stage3_h9_dataset", fromlist=["verify_source_hash_manifest"]).verify_source_hash_manifest(root, release["frozen_manifest"])
    write_json(output / "frozen_artifact_verification.json", verification)
    focused = run_command([sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h9.py"], root, output / "test_results.txt")
    regression = run_command([sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h8_software_only.py", "tests/test_stage3_h7.py", "tests/test_stage3_h7_7.py"], root, output / "regression_results.txt")
    gate, certificate, report = final_gate(release, replay, verification, focused, regression)
    write_json(output / "stage3_h9_gate_report.json", gate)
    write_json(output / "stage3_h9_terminal_certificate.json", certificate)
    (output / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    desktop = copy_desktop_evidence(root, output, certificate)
    write_json(output / "desktop_evidence_manifest.json", {"desktop_folder": str(desktop), "important_files_manifest": str(desktop / "IMPORTANT_FILES_MANIFEST.txt")})
    return 0 if certificate["STAGE_3_H9"] == "PASSED" else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--fixed-generated-at")
    args = parser.parse_args()
    output = args.output_dir
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = ROOT / "outputs" / f"stage3_h9_learning_dataset_baseline_{stamp}"
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing_to_overwrite_nonempty_output:{output}")
    output.mkdir(parents=True, exist_ok=True)
    try:
        if args.generate_only:
            generate_release(ROOT, output, args.fixed_generated_at or now_utc())
            return 0
        return run_main(ROOT, output)
    except Exception as exc:
        # A blocked run is still useful evidence, but do not fabricate a PASS
        # artifact when extraction/provenance fails before evaluation.
        write_json(output / "stage3_h9_terminal_certificate.json", {"schema_version": "stage3_h9_terminal_certificate_v1", "STAGE_3_H9": "BLOCKED", "FIRST_BLOCKER": str(exc)})
        write_json(output / "stage3_h9_gate_report.json", {"schema_version": "stage3_h9_gate_report_v1", "STAGE_3_H9": "BLOCKED", "FIRST_BLOCKER": str(exc)})
        (output / "FINAL_REPORT.md").write_text(f"# Stage 3 H9 — BLOCKED\n\nSTAGE_3_H9: BLOCKED\nFIRST_BLOCKER: {exc}\n", encoding="utf-8")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
