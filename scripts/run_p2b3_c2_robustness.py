#!/usr/bin/env python3
"""Measure the corrected C1 nominal over the 24 available P2-B2 axes."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_p2b1_solver_policy_ablation as p2b1  # noqa: E402
import scripts.run_p2b2_robustness_remapping as p2b2  # noqa: E402
import src.p2a_axiswise_robustness as p2a  # noqa: E402
from src.process_aware_stress import ProcessTrajectory, ResultStatus  # noqa: E402


BRANCH = "codex/fr5-p2b3-c2-robustness-transfer-20260928"
C1_RESULT = ROOT / "outputs/p2b3_c1_result.json"
C1_LINEAGE = ROOT / "outputs/p2b3_c1_r0_waypoint_lineage.csv"
C1_NOMINAL = ROOT / "outputs/p2b3_c2_c1_nominal_post_ruckig.csv"
OLD_LANDSCAPE = ROOT / "outputs/p2b2_robustness_landscape.json"
URDF_DEFAULT = ROOT / "outputs/p2b2_inputs/derived_reference_robot_model.urdf"
SRDF_DEFAULT = ROOT / "ros2_moveit_bridge/config/fairino5_v6_spray_tcp.srdf"
JOINT_LIMITS = ROOT / "ros2_moveit_bridge/config/joint_limits_with_jerk.yaml"
EXPECTED_URDF_SHA256 = "b9d20cd51aa7088b2bab5083b4919473f0e0ec94c4b0d87d4111ca5cfa270096"
EXPECTED_NATIVE_SHA256 = "7ccd4514cf7519f9ff652bc2ced785db681e4c47defbc4ab166a6c476c9d7b8a"
EXPECTED_FK_SHA256 = "78488b248c15569970d9690fe7976f172bc90d4c71d6628cc4f5e34c28625015"
MAX_CANDIDATES_PER_NATIVE_BATCH = 96
MAX_REFINEMENT_ITERATIONS = 80
HARD_MAX_NATIVE_BATCHES = 64


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"json_object_required:{path}")
    return result


def declared_input_sha256(input_hashes: Mapping[str, Any], relative_path: str) -> str | None:
    target = relative_path.replace("\\", "/")
    matches = [str(value) for key, value in input_hashes.items()
               if str(key).replace("\\", "/") == target]
    if len(matches) > 1 and len(set(matches)) != 1:
        raise RuntimeError(f"ambiguous_declared_input_identity:{relative_path}")
    return matches[0] if matches else None


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def git(*args: str) -> str:
    completed = subprocess.run(["git", *args], cwd=ROOT, text=True, encoding="utf-8", capture_output=True)
    if completed.returncode:
        raise RuntimeError(f"git_{args[0]}_failed:{completed.stderr.strip()}")
    return completed.stdout.strip()


def git_at(directory: Path, *args: str) -> str:
    completed = subprocess.run(["git", "-C", str(directory), *args], cwd=ROOT,
                               text=True, encoding="utf-8", capture_output=True)
    if completed.returncode:
        raise RuntimeError(f"git_at_{args[0]}_failed:{directory}:{completed.stderr.strip()}")
    return completed.stdout.strip()


def require_file(path: Path, label: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"{label}:{path}")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def verify_lineage(path: Path) -> dict[str, Any]:
    rows = read_rows(path)
    if len(rows) != 181:
        raise RuntimeError(f"C1_lineage_count_mismatch:{len(rows)}")
    required = {"index", "baseline_semantic_classification", "c1_semantic_classification",
                "c1_target_attempted", "c1_target_state_emitted", "c1_tight_converged",
                "c1_final_position_residual_m", "c1_final_normal_residual_rad",
                "c1_iterations_used", "c1_termination_reason", "c1_final_relaxed_geometry_gate"}
    if not required.issubset(rows[0]):
        raise RuntimeError("C1_lineage_schema_missing_required_semantics")
    for index, row in enumerate(rows):
        if int(row["index"]) != index:
            raise RuntimeError("C1_lineage_index_order_mismatch")
        expected_baseline = "OBSERVED_D39_ROW_EMITTED_AS_UNSOLVED_PREFIX" if index < 16 else "DLS_TARGET_STATE"
        expected_c1 = "DLS_TARGET_STATE_SEEDED_FROM_D39_ROW0" if index == 0 else "DLS_TARGET_STATE"
        if row["baseline_semantic_classification"] != expected_baseline or row["c1_semantic_classification"] != expected_c1:
            raise RuntimeError(f"C1_semantic_identity_mismatch:{index}")
        if row["c1_target_attempted"] != "True" or row["c1_target_state_emitted"] != "True":
            raise RuntimeError(f"C1_target_attempt_or_emission_missing:{index}")
        if row["c1_final_relaxed_geometry_gate"] != "PASS_4MM_5DEG":
            raise RuntimeError(f"C1_relaxed_acceptance_gate_missing:{index}")
        for key in ("c1_final_position_residual_m", "c1_final_normal_residual_rad"):
            if not math.isfinite(float(row[key])):
                raise RuntimeError(f"C1_lineage_nonfinite_residual:{index}:{key}")
    tight_count = sum(row["c1_tight_converged"].lower() == "true" for row in rows)
    return {"waypoint_count": 181, "tight_converged_count": tight_count,
            "accepted_target_count": sum(row["c1_target_state_emitted"].lower() == "true" for row in rows)}


def verify_c1_identity(c1: Mapping[str, Any], nominal_path: Path, lineage_path: Path) -> tuple[str, dict[str, Any]]:
    if c1.get("PROJECT") != "FAIRINO_FR5" or c1.get("STAGE") != "P2-B3-C1":
        raise RuntimeError("C1_project_or_stage_identity_mismatch")
    if c1.get("P2B3_C1_STATUS") not in {"COMPLETE", "COMPLETE_WITH_LIMITATIONS"}:
        raise RuntimeError(f"C1_status_not_acceptable_for_transfer:{c1.get('P2B3_C1_STATUS')}")
    if not isinstance(c1.get("C1_FATAL_GATES"), dict) or any(value != "PASS" for value in c1["C1_FATAL_GATES"].values()):
        raise RuntimeError("C1_fatal_gates_not_all_pass")
    expected_hash = c1.get("C1_POST_RUCKIG_NOMINAL", {}).get("sha256")
    actual_hash = sha256(nominal_path)
    if actual_hash != expected_hash:
        raise RuntimeError(f"C1_nominal_sha256_mismatch:expected={expected_hash}:actual={actual_hash}")
    lineage = verify_lineage(lineage_path)
    if c1.get("c1_measurement", {}).get("waypoint_count") != 181:
        raise RuntimeError("C1_result_waypoint_count_mismatch")
    if c1.get("c1_measurement", {}).get("tight_converged_count") != lineage["tight_converged_count"]:
        raise RuntimeError("C1_result_lineage_tight_convergence_count_mismatch")
    return actual_hash, lineage


def frozen_project_root(d46_root: Path) -> Path:
    resolved_d46 = d46_root.resolve()
    project_root = resolved_d46.parents[1]
    expected_d46 = project_root / "outputs" / "D46_STAGE4A_SYSTEM_BASELINE_V1"
    if expected_d46.resolve() != resolved_d46:
        raise RuntimeError(f"D46_root_is_not_inside_expected_project_layout:{resolved_d46}")
    return project_root


def authenticate_frozen_inputs(d46_root: Path) -> dict[str, Any]:
    auth_path = d46_root / "STAGE3_AUTHENTICATION_V1.json"
    manifest_path = d46_root / "STAGE4_SYSTEM_BENCHMARK_V1.json"
    native_provenance_path = d46_root / "native/D41_native_provenance.json"
    case_results_path = d46_root / "STAGE4_CASE_RESULTS.csv"
    scorecard_path = d46_root / "STAGE4_BASELINE_SCORECARD_V1.json"
    for path in (auth_path, manifest_path, native_provenance_path, case_results_path, scorecard_path):
        require_file(path, "frozen_D46_identity_file_missing")
    auth = read_json(auth_path)
    project_root = frozen_project_root(d46_root)
    release = project_root / auth["release"]
    release_manifest_path = release / "release_manifest.json"
    checkpoint_path = project_root / auth["canonical_checkpoint"]
    require_file(release_manifest_path, "frozen_Stage3_release_manifest_missing")
    require_file(checkpoint_path, "frozen_Stage3_checkpoint_missing")
    release_manifest = read_json(release_manifest_path)
    if release_manifest.get("release_status") != "FROZEN_AND_CLOSED":
        raise RuntimeError("Stage3_release_not_frozen_and_closed")
    if sha256(release_manifest_path) != auth.get("release_manifest_sha256"):
        raise RuntimeError("Stage3_release_manifest_hash_mismatch")
    if sha256(checkpoint_path) != auth.get("canonical_checkpoint_sha256"):
        raise RuntimeError("Stage3_checkpoint_hash_mismatch")
    authoritative = auth.get("authoritative_inputs", {})
    for name in ("poses", "seed_joints"):
        record = authoritative.get(name, {})
        path = project_root / record.get("path", "")
        require_file(path, f"Stage3_authoritative_{name}_missing")
        if sha256(path) != record.get("sha256"):
            raise RuntimeError(f"Stage3_authoritative_{name}_hash_mismatch")
    if authoritative.get("point_count") != 181 or authoritative.get("scope") != "Stage 0/1 ON-state open-arch only":
        raise RuntimeError("Stage3_authoritative_scope_mismatch")
    d41_summary_path = project_root / auth["retained_stage3_robot"]["D41_summary"]
    require_file(d41_summary_path, "authenticated_D41_summary_missing")
    d41_summary = read_json(d41_summary_path)
    if d41_summary.get("TASK_STATUS") != "PASS" or d41_summary.get("FULL_OFFLINE_WORKFLOW_SCOPE") != "181-point ON-state open-arch only":
        raise RuntimeError("authenticated_D41_nominal_scope_or_status_mismatch")
    snapshot_paths = (auth_path, manifest_path, native_provenance_path, case_results_path, scorecard_path,
                      release_manifest_path, checkpoint_path)
    return {
        "auth": auth,
        "scientific_project_root": str(project_root),
        "checkpoint_sha256": sha256(checkpoint_path),
        "release_manifest_sha256": sha256(release_manifest_path),
        "frozen_identity_sha256": {str(path): sha256(path) for path in snapshot_paths},
        "benchmark": read_json(manifest_path),
        "native_provenance": read_json(native_provenance_path),
    }


def static_candidate_budget(
    specs: Sequence[p2a.AxisSpec], nominal: ProcessTrajectory, *, max_native_batches: int,
) -> dict[str, Any]:
    total_coarse = sum(spec.coarse_intervals + 1 for spec in specs if spec.capability_status.value == "AVAILABLE")
    valid_candidate_q = 0
    unique_q: set[bytes] = set()
    direct_joint_limit_failures = 0
    for spec in specs:
        if spec.capability_status.value != "AVAILABLE":
            continue
        magnitudes = np.linspace(0.0, spec.search_domain_max, spec.coarse_intervals + 1)
        for magnitude in magnitudes:
            candidate = p2a.apply_axis_perturbation(nominal, spec, float(magnitude))
            checked = p2a.evaluate_joint_limits(candidate.application.trajectory)
            if checked.status == ResultStatus.FAIL:
                direct_joint_limit_failures += 1
                continue
            if checked.status != ResultStatus.PASS:
                raise RuntimeError("static_candidate_joint_limit_state_unknown")
            valid_candidate_q += 1
            unique_q.add(np.ascontiguousarray(candidate.application.trajectory.joint_states, dtype=np.float64).tobytes())
    max_transitions = sum(2 * spec.coarse_intervals for spec in specs if spec.capability_status.value == "AVAILABLE")
    typical_transition_estimate = sum(1 for spec in specs if spec.capability_status.value == "AVAILABLE")
    return {
        "available_axis_count": len(specs),
        "coarse_candidate_count": total_coarse,
        "valid_joint_state_candidate_count": valid_candidate_q,
        "unique_valid_joint_state_count_before_native_cache": len(unique_q),
        "joint_limit_failures_classified_without_native_call": direct_joint_limit_failures,
        "max_refinement_iterations_per_observed_transition": MAX_REFINEMENT_ITERATIONS,
        "max_native_batches": max_native_batches,
        "hard_max_native_batches": HARD_MAX_NATIVE_BATCHES,
        "theoretical_max_observed_transition_count_on_coarse_grid": max_transitions,
        "theoretical_max_refinement_candidate_count": max_transitions * MAX_REFINEMENT_ITERATIONS,
        "nominal_one_transition_per_axis_estimated_total_candidates": total_coarse + typical_transition_estimate * 20,
        "native_batch_candidate_cap": MAX_CANDIDATES_PER_NATIVE_BATCH,
        "estimated_native_batches_if_one_transition_per_axis": math.ceil(len(unique_q) / MAX_CANDIDATES_PER_NATIVE_BATCH)
        + math.ceil(typical_transition_estimate * 20 / MAX_CANDIDATES_PER_NATIVE_BATCH),
        "search_domain_and_grids": "P2A frozen _build_axis_specs policy: joint 64 intervals; TCP translation 48; rotation 48",
    }


def run_campaign(args: argparse.Namespace) -> dict[str, Any]:
    if not 1 <= args.max_native_batches <= HARD_MAX_NATIVE_BATCHES:
        raise ValueError(f"max_native_batches_must_be_between_1_and_{HARD_MAX_NATIVE_BATCHES}")
    if not 1 <= args.native_timeout_seconds <= 7200:
        raise ValueError("native_timeout_seconds_must_be_between_1_and_7200")
    if not 1 <= args.max_campaign_hours <= 12:
        raise ValueError("max_campaign_hours_must_be_between_1_and_12")
    campaign_started = time.monotonic()
    campaign_deadline = campaign_started + args.max_campaign_hours * 3600
    branch, execution_commit = git("branch", "--show-current"), git("rev-parse", "HEAD")
    if branch != BRANCH or git("status", "--porcelain"):
        raise RuntimeError(f"campaign_requires_clean_C2_execution_commit:branch={branch}")
    c1_result_path = args.c1_result.resolve()
    lineage_path = args.c1_lineage.resolve()
    nominal_path = args.nominal.resolve()
    old_landscape_path = args.old_landscape.resolve()
    old_nominal_path = args.old_nominal.resolve()
    urdf_path, srdf_path = args.urdf.resolve(), args.srdf.resolve()
    d46_root, scratch = args.d46_root.resolve(), args.scratch.resolve()
    underlay, overlay = args.underlay_install.resolve(), args.overlay_install.resolve()
    native_binary, fk_binary = args.native_binary.resolve(), args.fk_binary.resolve()
    for path, label in ((c1_result_path, "C1_result"), (lineage_path, "C1_lineage"), (nominal_path, "C1_nominal"),
                        (old_landscape_path, "old_P2B2_landscape"), (urdf_path, "C1_and_P2B2_robot_model"),
                        (old_nominal_path, "old_P2B2_B0_nominal_trajectory"),
                        (srdf_path, "MoveIt_SRDF"), (native_binary, "D41_native_binary"), (fk_binary, "MoveIt_FK_binary"),
                        (underlay / "setup.bash", "FAIRINO_underlay_setup"), (overlay / "setup.bash", "MoveIt_overlay_setup")):
        require_file(path, label)
    if sha256(urdf_path) != EXPECTED_URDF_SHA256:
        raise RuntimeError("robot_model_does_not_match_C1_and_P2B2_frozen_model")
    if sha256(native_binary) != EXPECTED_NATIVE_SHA256 or sha256(fk_binary) != EXPECTED_FK_SHA256:
        raise RuntimeError("D41_or_FK_binary_identity_mismatch")

    c1_result = read_json(c1_result_path)
    nominal_sha256, lineage_summary = verify_c1_identity(c1_result, nominal_path, lineage_path)
    if c1_result.get("C1_DERIVED_REFERENCE_URDF_SHA256") != sha256(urdf_path):
        raise RuntimeError("C1_and_C2_robot_model_identity_mismatch")
    if c1_result.get("FAIRINO_SOURCE_COMMIT") != "60755d44d521a5ad6bee8494cc19522f8801aa20":
        raise RuntimeError("C1_official_FAIRINO_source_commit_mismatch")
    fairino_source_identity = verify_fairino_source(
        args.fairino_source_checkout, str(c1_result.get("FAIRINO_SOURCE_COMMIT")),
    )
    if not old_nominal_path.is_relative_to(ROOT):
        raise RuntimeError("historical_P2B2_B0_nominal_must_be_a_declared_repository_input")
    old_relative = old_nominal_path.relative_to(ROOT).as_posix()
    expected_old_nominal_hash = declared_input_sha256(
        c1_result.get("input_identity_sha256", {}), old_relative,
    )
    if not expected_old_nominal_hash or sha256(old_nominal_path) != expected_old_nominal_hash:
        raise RuntimeError("historical_P2B2_B0_nominal_identity_mismatch_or_undeclared_input")
    old_landscape = read_json(old_landscape_path)
    old_rows = [row for row in old_landscape.get("B0_REFERENCE_AXES", []) if row.get("capability_status") == "AVAILABLE"]
    old_by_axis = {str(row.get("axis_id")): row for row in old_rows}
    if len(old_rows) != 24 or len(old_by_axis) != 24:
        raise RuntimeError("P2B2_B0_reference_does_not_have_24_unique_available_axis_rows")

    frozen = authenticate_frozen_inputs(d46_root)
    identity_before = frozen["frozen_identity_sha256"]
    import tools.stage4a_system_benchmark as d46
    d46.OUT = d46_root
    d46.URDF = urdf_path
    d46.SRDF = srdf_path
    lower, upper, dynamic_limits = d46.load_limits()
    timestamps, q, _velocity, _acceleration, _jerk = d46.load_post_ruckig(nominal_path)
    nominal_rows = read_rows(nominal_path)
    expected_nominal_columns = {"t"} | {
        f"j{joint}_{suffix}" for joint in range(1, 7) for suffix in ("q", "dq", "ddq", "jerk")
    }
    if not nominal_rows or set(nominal_rows[0]) != expected_nominal_columns:
        raise RuntimeError("C1_nominal_joint_column_schema_mismatch")
    if q.shape != (181, 6) or timestamps.shape != (181,) or np.any(np.diff(timestamps) <= 0.0):
        raise RuntimeError("C1_nominal_waypoint_or_timebase_mismatch")
    if not np.isfinite(np.concatenate((q, timestamps[:, None]), axis=1)).all():
        raise RuntimeError("C1_nominal_nonfinite_state")
    nominal_limit_result = p2a.evaluate_joint_limits(ProcessTrajectory(
        joint_states=q, joint_lower_rad=lower, joint_upper_rad=upper,
    ))
    if nominal_limit_result.status != ResultStatus.PASS:
        raise RuntimeError(f"C1_nominal_joint_limit_gate:{nominal_limit_result.status.value}")

    target_positions, target_quaternions, target_normals = d46.read_targets()
    preflight_nominal = ProcessTrajectory(
        tcp_poses=np.column_stack((target_positions, target_quaternions)), joint_states=q,
        timestamps_s=timestamps, surface_normals=target_normals,
        joint_lower_rad=lower, joint_upper_rad=upper,
    )
    old_times, old_q, _ov, _oa, _oj = d46.load_post_ruckig(old_nominal_path)
    if old_q.shape != (181, 6) or np.any(np.diff(old_times) <= 0.0):
        raise RuntimeError("historical_P2B2_B0_nominal_waypoint_or_timebase_mismatch")
    old_specs = p2a._build_axis_specs(old_q, lower, upper)
    old_domain_by_axis = {spec.axis_id: {"search_domain": spec.to_record()["search_domain"],
                                         "coarse_intervals": spec.coarse_intervals,
                                         "refinement_tolerance": spec.refinement_tolerance,
                                         "coarse_step": spec.search_domain_max / spec.coarse_intervals}
                          for spec in old_specs if spec.capability_status.value == "AVAILABLE"}
    all_specs = p2a._build_axis_specs(q, lower, upper)
    capability_gaps = [{"axis_id": spec.axis_id, "status": spec.capability_status.value,
                        "reason": spec.unavailable_reason}
                       for spec in all_specs if spec.capability_status.value != "AVAILABLE"]
    specs = [spec for spec in all_specs if spec.capability_status.value == "AVAILABLE"]
    spec_by_axis = {spec.axis_id: spec for spec in specs}
    if len(specs) != 24 or set(spec_by_axis) != set(old_by_axis):
        raise RuntimeError("C2_available_axis_definition_mismatch_from_P2B2")
    budget = static_candidate_budget(specs, preflight_nominal, max_native_batches=args.max_native_batches)
    preflight = {
        "nominal_sha256": nominal_sha256, "waypoints": int(q.shape[0]),
        "available_axes": len(specs), "unavailable_axes_excluded": len(capability_gaps),
        "campaign_budget": budget, "max_native_batches": args.max_native_batches,
        "max_campaign_hours": args.max_campaign_hours,
        "per_native_batch_timeout_seconds": args.native_timeout_seconds,
        "retry_policy": "No unbounded retry; a failed native/FK chunk becomes UNKNOWN and independent chunks continue.",
    }
    print("C2_PREFLIGHT=" + json.dumps(preflight, sort_keys=True), flush=True)

    scratch.mkdir(parents=True, exist_ok=False)
    runtime_provenance = p2b1.runtime_provenance(underlay, overlay, args.distro, scratch)
    fk_root = scratch / "nominal_fk"
    p2a._run_existing_fk = lambda batch_root, native_root, module: p2b1._run_fresh_fk(
        batch_root, native_root, module, fk_binary, overlay, underlay, urdf_path, args.distro,
        timeout_s=args.native_timeout_seconds,
    )
    d46.run_native = lambda run_dir, cases, native_name="native": p2b1._run_native_with_fresh_build(
        run_dir, cases, native_name, d46, native_binary, overlay, underlay, urdf_path, args.distro,
        timeout_s=args.native_timeout_seconds,
    )
    nominal_case = "P2B3_C2_C1_NOMINAL"
    fk_trace, q_by_case = p2a._run_fk_only_cases(fk_root, [(nominal_case, q)], d46)
    fk_by_case = p2a._read_fk_trace(fk_trace)
    if set(q_by_case) != {nominal_case} or set(fk_by_case) != {nominal_case}:
        raise RuntimeError("fresh_C1_nominal_FK_case_set_mismatch")
    fk = fk_by_case[nominal_case]
    poses = np.column_stack((fk["position"], fk["quaternion"]))
    thresholds = frozen["benchmark"].get("diagnostic_thresholds", {})
    position_limit = float(thresholds["tcp_trajectory_error_m"]["value"])
    terminal_limit = float(thresholds["terminal_position_error_m"]["value"])
    joint_step_limit = float(thresholds["joint_step_rad"]["value"])
    stored_c1_fk_path = args.c1_fk_trace.resolve()
    require_file(stored_c1_fk_path, "C1_stored_MoveIt_FK_trace")
    expected_c1_fk_hash = c1_result.get("candidate_input_identity_sha256", {}).get("candidate_fk_trace")
    if not expected_c1_fk_hash or sha256(stored_c1_fk_path) != expected_c1_fk_hash:
        raise RuntimeError("C1_stored_FK_trace_sha256_mismatch")
    stored_fk = read_rows(stored_c1_fk_path)
    if len(stored_fk) != 181:
        raise RuntimeError("C1_stored_FK_waypoint_count_mismatch")
    if any(float(row["t"]) != float(timestamps[index]) for index, row in enumerate(stored_fk)):
        raise RuntimeError("C1_stored_FK_timestamp_or_row_order_mismatch")
    stored_positions = np.asarray([[float(row[key]) for key in ("actual_tcp_x", "actual_tcp_y", "actual_tcp_z")] for row in stored_fk], dtype=np.float64)
    fresh_position_difference = float(np.max(np.linalg.norm(poses[:, :3] - stored_positions, axis=1)))
    stored_tool_z = np.asarray([[float(row[key]) for key in ("tool_z_x", "tool_z_y", "tool_z_z")] for row in stored_fk], dtype=np.float64)
    stored_normals = np.asarray([[float(row[key]) for key in ("wall_normal_x", "wall_normal_y", "wall_normal_z")] for row in stored_fk], dtype=np.float64)
    aligned_target_normals, normal_reference_mapping = p2b2._project_target_normals_to_fk_samples(
        target_positions, target_normals, poses[:, :3], max_path_deviation_m=position_limit,
    )
    normal_reference_delta = float(np.max(np.linalg.norm(aligned_target_normals - stored_normals, axis=1)))
    fresh_tool_z = np.asarray([p2a._tool_z_direction(row) for row in fk["quaternion"]])
    fresh_tool_difference = float(max(p2a._direction_angle_rad(a, b) for a, b in zip(fresh_tool_z, stored_tool_z)))
    if fresh_position_difference > 1e-6 or fresh_tool_difference > 1e-6 or normal_reference_delta > 1e-9:
        raise RuntimeError(f"C1_nominal_FK_model_or_normal_reference_crosscheck_failed:{fresh_position_difference}:{fresh_tool_difference}:{normal_reference_delta}")

    nominal = ProcessTrajectory(
        tcp_poses=poses, joint_states=q, timestamps_s=timestamps,
        surface_normals=aligned_target_normals, joint_lower_rad=lower, joint_upper_rad=upper,
        metadata={
            "robot": "FAIRINO_FR5", "scope": "181-point ON-state open-arch only",
            "source": "C1 corrected post-Ruckig trajectory with fresh MoveIt2 FK",
            "trajectory_sha256": nominal_sha256,
            "robot_model_sha256": sha256(urdf_path),
        },
    )
    import tools.audit_stage17_reproducibility as d17
    normal_limit = math.radians(float(d17.FORMAL_NORMAL_DEG))
    evaluator = p2a._D41BatchEvaluator(
        nominal, timestamps, lower, upper, dynamic_limits,
        position_limit, terminal_limit, normal_limit, joint_step_limit,
        scratch / "scan", d46,
    )
    (scratch / "scan").mkdir()
    callback_stats: dict[str, Any] = {"fresh_cases_submitted": 0, "reused_candidate_cases": 0,
                                      "native_callback_chunks": 0, "unknown_candidate_cases": 0,
                                      "joint_limit_direct_failures": 0, "batch_failures": []}
    progress_path = scratch / "campaign_progress.jsonl"

    def evaluate_in_chunks(candidates: Sequence[p2a.ScanCandidate]) -> Mapping[str, p2a.EvaluationResult]:
        result: dict[str, p2a.EvaluationResult] = {}
        for start in range(0, len(candidates), MAX_CANDIDATES_PER_NATIVE_BATCH):
            chunk = candidates[start:start + MAX_CANDIDATES_PER_NATIVE_BATCH]
            known = set(evaluator._physical_by_q)
            seen: set[bytes] = set()
            fresh_keys: set[bytes] = set()
            reused = 0
            direct_limits = 0
            for candidate in chunk:
                if candidate.application.status.value != "APPLIED":
                    continue
                limits = p2a.evaluate_joint_limits(candidate.application.trajectory)
                if limits.status == ResultStatus.FAIL:
                    direct_limits += 1
                    continue
                if limits.status != ResultStatus.PASS:
                    continue
                key = np.ascontiguousarray(candidate.application.trajectory.joint_states, dtype=np.float64).tobytes()
                if key in known or key in seen:
                    reused += 1
                else:
                    seen.add(key)
                    fresh_keys.add(key)
            callback_stats["fresh_cases_submitted"] += len(fresh_keys)
            callback_stats["reused_candidate_cases"] += reused
            callback_stats["joint_limit_direct_failures"] += direct_limits
            callback_stats["native_callback_chunks"] += 1
            failure: str | None = None
            if (evaluator._batch_number >= args.max_native_batches or time.monotonic() >= campaign_deadline) and fresh_keys:
                failure = "MAX_NATIVE_BATCH_BUDGET_REACHED" if evaluator._batch_number >= args.max_native_batches else "MAX_CAMPAIGN_WALLTIME_REACHED"
                callback_stats["batch_failures"].append({"candidate_ids": [row.candidate_id for row in chunk], "error": failure})
                for candidate in chunk:
                    if candidate.application.status.value != "APPLIED":
                        result[candidate.candidate_id] = p2a.EvaluationResult(
                            ResultStatus.UNKNOWN, evidence={"reason": "operator_not_applied"},
                        )
                        continue
                    limit_result = p2a.evaluate_joint_limits(candidate.application.trajectory)
                    if limit_result.status != ResultStatus.PASS:
                        result[candidate.candidate_id] = limit_result
                        continue
                    key = np.ascontiguousarray(candidate.application.trajectory.joint_states, dtype=np.float64).tobytes()
                    physical = evaluator._physical_by_q.get(key)
                    if physical is None:
                        result[candidate.candidate_id] = p2a.EvaluationResult(
                            ResultStatus.UNKNOWN,
                            evaluator_capabilities={"native_moveit2": "UNKNOWN"},
                            evidence={"reason": "MAX_NATIVE_BATCH_BUDGET_REACHED"},
                        )
                    else:
                        result[candidate.candidate_id] = evaluator._evaluate_one(candidate, physical)
            else:
                try:
                    chunk_result = evaluator.evaluate_batch(chunk)
                    result.update(chunk_result)
                except Exception as exc:
                    failure = f"{type(exc).__name__}:{exc}"
                    callback_stats["batch_failures"].append({"candidate_ids": [row.candidate_id for row in chunk], "error": failure})
                    for candidate in chunk:
                        result[candidate.candidate_id] = p2a.EvaluationResult(
                            ResultStatus.UNKNOWN,
                            evaluator_capabilities={"native_moveit2": "UNKNOWN"},
                            evidence={"reason": "native_or_fk_batch_failed", "error": failure},
                        )
            callback_stats["unknown_candidate_cases"] += sum(
                result.get(row.candidate_id, p2a.EvaluationResult(ResultStatus.UNKNOWN)).status == ResultStatus.UNKNOWN
                for row in chunk
            )
            with progress_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"chunk": callback_stats["native_callback_chunks"],
                                         "candidates": len(chunk), "fresh_unique_q": len(fresh_keys),
                                         "reused_candidates": reused, "direct_joint_limit_failures": direct_limits,
                                         "batch_error": failure}, sort_keys=True) + "\n")
            print(f"C2 batch {callback_stats['native_callback_chunks']}: candidates={len(chunk)} fresh_q={len(fresh_keys)} reused={reused} direct_joint_limit_failures={direct_limits} error={failure or 'none'}", flush=True)
        return result

    axes = p2a.scan_axes_batched(
        nominal, specs, evaluate_in_chunks,
        max_refinement_iterations=MAX_REFINEMENT_ITERATIONS,
    )
    axes_path = scratch / "axis_results_full.json"
    write_json(axes_path, {"schema": "p2b3-c2-axis-results-v1", "axis_results": axes})
    axis_by_id = {axis["specification"]["axis_id"]: axis for axis in axes}
    if len(axis_by_id) != 24 or set(axis_by_id) != set(spec_by_axis):
        raise RuntimeError("C2_axis_result_set_mismatch")
    if any(sha256(Path(key)) != value for key, value in identity_before.items()):
        raise RuntimeError("protected_Stage3_or_D46_identity_changed_during_C2")

    comparison = build_transfer_comparison(axes, old_by_axis, old_domain_by_axis)
    profile = build_margin_profile(axes)
    findings = build_stage4b_findings(axes)
    complete_axes = sum(axis_is_complete(row) for row in axes)
    unknown_axes = sum(axis_has_unknown(row) for row in axes)
    nonmonotonic = [row["specification"]["axis_id"] for row in axes if row.get("non_monotonic_failure_region_observed")]
    multiple_regions = [row["specification"]["axis_id"] for row in axes if len(row.get("all_failure_intervals", [])) >= 2]
    family_controllers = family_control(axes)
    transfer_counts = {name: sum(name in row["transfer_classifications"] for row in comparison)
                       for name in ("CONSISTENT_WITHIN_TOLERANCE", "MARGIN_SHIFTED", "CRITICAL_LOCATION_CHANGED",
                                    "FAILURE_MODE_CHANGED", "OLD_RESULT_INCOMPLETE", "C1_RESULT_INCOMPLETE", "NOT_COMPARABLE")}
    failure_count = sum(
        observation["status"] == "FAIL"
        for axis in axes for observation in axis.get("coarse_observations", []) + axis.get("refinement_observations", [])
    )
    pipeline_complete = complete_axes == 24 and unknown_axes == 0
    result = {
        "schema": "p2b3-c2-c1-robustness-transfer-v1",
        "PROJECT": "FAIRINO_FR5", "STAGE": "P2-B3-C2",
        "P2B3_C2_STATUS": "P2B3_C2_AXISWISE_RESCAN_COMPLETE_WITH_MEASURED_BASELINE_FINDINGS" if pipeline_complete else "INCOMPLETE_UNKNOWN_OR_UNREFINED_MEASUREMENTS",
        "MEASUREMENT_PIPELINE_STATUS": "PASS" if pipeline_complete else "INCOMPLETE",
        "FROZEN_ROBOT_BASELINE_PERFORMANCE_STATUS": "MEASURED_FAILURES_PRESENT" if failure_count else "NO_FAILURE_OBSERVED_IN_TESTED_DOMAINS",
        "SOURCE_BASE_COMMIT": c1_result.get("C1_ARTIFACT_PUBLISH_COMMIT"),
        "C1_NOMINAL_SOURCE_COMMIT": c1_result.get("C1_EXECUTION_CODE_COMMIT"),
        "C1_EXECUTION_CODE_COMMIT": c1_result.get("C1_EXECUTION_CODE_COMMIT"),
        "C1_ARTIFACT_PUBLISH_COMMIT": c1_result.get("C1_ARTIFACT_PUBLISH_COMMIT"),
        "C2_EXECUTION_CODE_COMMIT": execution_commit,
        "C2_ARTIFACT_PUBLISH_COMMIT": {
            "status": "TWO_PHASE_SELF_REFERENCE_SAFE",
            "phase_1_execution_code_commit": execution_commit,
            "phase_2_artifact_commit": "RESOLVE_AFTER_FIRST_CANONICAL_ARTIFACT_COMMIT; the artifact commit cannot contain its own Git object id",
        },
        "FAIRINO_SOURCE_COMMIT": c1_result.get("FAIRINO_SOURCE_COMMIT"),
        "branch": branch,
        "C1_NOMINAL_IDENTITY": "PASS",
        "C1_WAYPOINT_COUNT": lineage_summary["waypoint_count"],
        "C1_TIGHT_CONVERGED_COUNT": lineage_summary["tight_converged_count"],
        "C1_ACCEPTED_TARGET_COUNT": lineage_summary["accepted_target_count"],
        "TOTAL_AXIS_DIRECTIONS": 24,
        "COMPLETE_AXIS_DIRECTIONS": complete_axes,
        "INCOMPLETE_AXIS_DIRECTIONS": 24 - complete_axes,
        "UNKNOWN_AXIS_DIRECTIONS": unknown_axes,
        "FAMILY_AXIS_COMPLETENESS": {
            family: {"complete": sum(axis_is_complete(row) for row in axes if row["specification"]["perturbation_family"] == family),
                     "total": sum(1 for row in axes if row["specification"]["perturbation_family"] == family)}
            for family in sorted({row["specification"]["perturbation_family"] for row in axes})
        },
        "NON_MONOTONIC_AXES": nonmonotonic,
        "MULTIPLE_FAILURE_REGION_AXES": multiple_regions,
        "OLD_P2B2_TRANSFER": transfer_counts,
        "FAMILY_CONTROLLING_DIRECTIONS": family_controllers,
        "GLOBAL_RAW_RANKING": "GLOBAL_RAW_RANKING_NOT_DEFINED_ACROSS_INCOMMENSURATE_UNITS",
        "CRITICAL_WAYPOINT_DISTRIBUTION": critical_location_distribution(axes),
        "WP0_16_BECAME_CONTROLLING": wp0_16_became_controlling(family_controllers),
        "WP179_180_TIMING_ANOMALY": {
            "C1_speed_m_s": c1_result["c1_measurement"]["local_speed_max_m_s"],
            "configured_target_m_s": 0.003, "configured_band_fraction": 0.05,
            "C2_out_of_band_candidate_count": sum(
                row.get("evidence", {}).get("wp179_to_wp180_local_timing_status") in {"ABOVE_CONFIGURED_BAND", "BELOW_CONFIGURED_BAND"}
                for axis in axes for row in axis.get("coarse_observations", []) + axis.get("refinement_observations", [])
            ),
            "interpretation": "LOCAL_TIMING_ANOMALY; no validated deposition model, so no coating-quality inference",
        },
        "STAGE4_FAILURE_TAXONOMY_V1": findings["taxonomy"],
        "STAGE4_RISK_RANKED_BOTTLENECKS_V1": findings["ranked"],
        "COLLISION_METHOD": "adaptive_discrete_interpolation",
        "DYNAMICS_EVALUATION_SEMANTICS": "Perturbed q states use frozen C1 timestamps and the existing D46 finite-difference audit; candidates are not retimed or smoothed with Ruckig.",
        "STRICT_CONTINUOUS_CCD": "NOT_AVAILABLE",
        "HARDWARE_VALIDATION": "NOT_RUN",
        "COATING_QUALITY_VALIDATED": "NO",
        "AXIS_RESULTS": axes,
        "TRANSFER_COMPARISON": comparison,
        "MARGIN_PROFILE": profile,
        "COVERAGE_GAPS": capability_gaps,
        "UNRESOLVED_THRESHOLDS": ["physical singularity threshold", "clearance acceptance threshold", "vendor-certified dynamic limits", "calibrated TCP uncertainty"],
        "C1_EVIDENCE_CLOSURE": {
            "C1_RESULT_SHA256": sha256(c1_result_path), "C1_LINEAGE_SHA256": sha256(lineage_path),
            "C1_NOMINAL_SHA256": nominal_sha256, "C1_SOLVER_PRESSURE_SHA256": c1_result["C1_SOLVER_PRESSURE_EVIDENCE"].get("sha256"),
            "C1_FATAL_GATES": c1_result["C1_FATAL_GATES"],
            "C1_STATUS": c1_result["P2B3_C1_STATUS"],
            "DLS_COUNTS": c1_result["C1_SOLVER_PRESSURE_EVIDENCE"],
        },
        "IDENTITY": {
            "C2_EXECUTION_CODE_COMMIT": execution_commit,
            "C1_EXECUTION_CODE_COMMIT": c1_result.get("C1_EXECUTION_CODE_COMMIT"),
            "C1_ARTIFACT_PUBLISH_COMMIT": c1_result.get("C1_ARTIFACT_PUBLISH_COMMIT"),
            "FAIRINO_SOURCE_COMMIT": c1_result.get("FAIRINO_SOURCE_COMMIT"),
            "nominal_trajectory_sha256": nominal_sha256,
            "robot_model_sha256": sha256(urdf_path), "srdf_sha256": sha256(srdf_path),
            "joint_limit_and_dynamics_config_sha256": sha256(JOINT_LIMITS),
            "D41_native_binary_sha256": sha256(native_binary), "MoveIt_FK_binary_sha256": sha256(fk_binary),
            "D46_frozen_identity_sha256": identity_before,
            "C1_nominal_FK_crosscheck": {"max_position_delta_m": fresh_position_difference,
                                         "max_tool_z_delta_rad": fresh_tool_difference,
                                         "max_projected_surface_normal_delta": normal_reference_delta,
                                         "surface_normal_alignment": normal_reference_mapping},
            "inputs_sha256": {str(path): sha256(path) for path in (
                c1_result_path, lineage_path, nominal_path, stored_c1_fk_path, old_landscape_path,
                old_nominal_path,
                urdf_path, srdf_path, JOINT_LIMITS, d46_root / "STAGE4_SYSTEM_BENCHMARK_V1.json",
                d46_root / "native/D41_native_provenance.json", native_binary, fk_binary,
            )},
        "source_checkout": fairino_source_identity,
        },
        "SEARCH_BUDGET": budget,
        "CAMPAIGN_ELAPSED_SECONDS": time.monotonic() - campaign_started,
        "NATIVE_RUNTIME_PROVENANCE": {
            "text": runtime_provenance,
            "sha256": sha256(scratch / "runtime_provenance.txt")
            if (scratch / "runtime_provenance.txt").is_file() else None,
        },
        "EXECUTION_COUNTS": {
            "FRESH_CASES": callback_stats["fresh_cases_submitted"],
            "REUSED_CASES": callback_stats["reused_candidate_cases"],
            "REUSE_REASON": "Exact in-campaign q-state cache keyed by complete float64 joint-state bytes under the same C1 nominal SHA, evaluator code, model, config, and capability semantics; duplicated TCP target operators do not trigger duplicate native q evaluation.",
            "native_callback_chunks": callback_stats["native_callback_chunks"],
            "native_evaluator_batches": evaluator._batch_number,
            "joint_limit_direct_failures": callback_stats["joint_limit_direct_failures"],
            "unknown_candidate_cases": sum(
                row["status"] == "UNKNOWN" for axis in axes
                for row in axis.get("coarse_observations", []) + axis.get("refinement_observations", [])
            ),
            "failed_batches": callback_stats["batch_failures"],
            "native_case_ids": len(evaluator._case_for_q),
        },
        "LIMITATIONS": [
            "Stage 3 canonical release/checkpoint, official joint limits, and D46 acceptance thresholds were read-only.",
            "Collision evidence is adaptive_discrete_interpolation; strict continuous CCD is NOT_AVAILABLE.",
            "Clearance values are emitted only where the native FCL backend returned them; acceptance threshold remains unresolved.",
            "Configured finite-difference dynamics limits are not vendor-certified; hardware validation is NOT_RUN.",
            "The C1 WP179-to-WP180 speed deviation remains a local timing diagnostic, not a coating-quality finding.",
            "P2A candidate evaluations retain the frozen C1 timestamps and do not rerun candidate Ruckig; configured dynamics ratios are finite-difference audit measurements.",
        ],
    }
    output_path = args.campaign_output.resolve()
    write_json(output_path, result)
    return {"campaign_output": str(output_path), "C2_STATUS": result["P2B3_C2_STATUS"],
            "complete_axes": complete_axes, "unknown_axes": unknown_axes,
            "fresh_cases": callback_stats["fresh_cases_submitted"],
            "reused_cases": callback_stats["reused_candidate_cases"],
            "native_batches": evaluator._batch_number, "scratch": str(scratch)}


def axis_is_complete(axis: Mapping[str, Any]) -> bool:
    complete = axis.get("search_completeness")
    return axis.get("specification", {}).get("capability_status") == "AVAILABLE" and complete in {
        "ALL_OBSERVED_TRANSITIONS_REFINED", "FULL_COARSE_DOMAIN_SAMPLED",
    }


def axis_has_unknown(axis: Mapping[str, Any]) -> bool:
    observations = axis.get("coarse_observations", []) + axis.get("refinement_observations", [])
    return any(row.get("status") == "UNKNOWN" for row in observations)


def _first_fail(axis: Mapping[str, Any]) -> Mapping[str, Any] | None:
    item = axis.get("first_failing_perturbation")
    return item if isinstance(item, dict) else None


def _old_old_complete(row: Mapping[str, Any]) -> bool:
    return row.get("search_completeness") in {"LOCAL_BRACKET_REFINED", "FULL_COARSE_DOMAIN_SAMPLED"}


def build_transfer_comparison(axes: Sequence[Mapping[str, Any]], old_by_axis: Mapping[str, Mapping[str, Any]],
                             old_domain_by_axis: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for axis in axes:
        spec = axis["specification"]
        axis_id = spec["axis_id"]
        old = old_by_axis.get(axis_id)
        first = _first_fail(axis)
        new_margin = axis.get("first_failure_margin")
        old_margin = None if old is None else old.get("estimated_raw_axis_margin")
        old_value = float(old_margin) if isinstance(old_margin, (float, int)) else None
        new_value = None if not isinstance(new_margin, dict) or new_margin.get("value") is None else float(new_margin["value"])
        new_failure = None if first is None else first.get("dominant_failure_mode")
        old_failure = None if old is None else old.get("dominant_failure_mode")
        new_wp = None if first is None else first.get("critical_waypoint")
        old_wp = None if old is None else old.get("critical_waypoint")
        new_segment = None if first is None else first.get("critical_segment")
        old_segment = None if old is None else old.get("critical_segment")
        new_boundary_width = (axis.get("all_refined_boundaries") or [{}])[0].get("width")
        old_width = None if old is None or old.get("last_pass") is None or old.get("first_fail") is None else abs(float(old["first_fail"]) - float(old["last_pass"]))
        tolerance = max(float(value) for value in (new_boundary_width, old_width, spec["refinement_tolerance"]) if value is not None)
        classes: list[str] = []
        if old is None or not _old_old_complete(old):
            classes.append("OLD_RESULT_INCOMPLETE")
        if not axis_is_complete(axis) or axis_has_unknown(axis):
            classes.append("C1_RESULT_INCOMPLETE")
        if old is not None and (old.get("units") != spec["units"] or axis.get("specification", {}).get("capability_status") != "AVAILABLE"):
            classes.append("NOT_COMPARABLE")
        if not classes:
            if new_value is None and old_value is None:
                classes.append("CONSISTENT_WITHIN_TOLERANCE")
            else:
                if new_value is None or old_value is None:
                    classes.append("NOT_COMPARABLE")
                else:
                    if abs(new_value - old_value) <= tolerance:
                        classes.append("CONSISTENT_WITHIN_TOLERANCE")
                    else:
                        classes.append("MARGIN_SHIFTED")
                    if old_failure != new_failure:
                        classes.append("FAILURE_MODE_CHANGED")
                    if old_wp != new_wp or old_segment != new_segment:
                        classes.append("CRITICAL_LOCATION_CHANGED")
        delta = None if old_value is None or new_value is None else new_value - old_value
        relative = None if old_value in (None, 0.0) or delta is None else delta / abs(old_value)
        output.append({
            "axis_id": axis_id, "family": spec["perturbation_family"], "direction": spec["direction"],
            "units": spec["units"], "old_margin": old_value, "c1_margin": new_value,
            "absolute_delta": delta, "relative_delta": relative,
            "comparison_tolerance_from_refinement_resolution": tolerance,
            "old_failure_mode": old_failure, "c1_failure_mode": new_failure,
            "old_critical_waypoint": old_wp, "c1_critical_waypoint": new_wp,
            "old_critical_segment": old_segment, "c1_critical_segment": new_segment,
            "old_search_completeness": None if old is None else old.get("search_completeness"),
            "c1_search_completeness": axis.get("search_completeness"),
            "transfer_classifications": classes,
            "search_domains": {"old": old_domain_by_axis.get(axis_id),
                                "c1": {"search_domain": spec["search_domain"],
                                       "coarse_intervals": spec["coarse_intervals"],
                                       "refinement_tolerance": spec["refinement_tolerance"],
                                       "coarse_step": spec["coarse_step"]},
                                "definition_comparison": "SAME_POLICY; joint domain endpoint is recomputed from the respective nominal one-sided margin"},
            "nominal_trajectory_sha256": next((row.get("nominal_trajectory_sha256") for row in axis.get("coarse_observations", [])[:1]), None),
        })
    return output


def build_margin_profile(axes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for axis in axes:
        spec = axis["specification"]
        key = (str(spec["perturbation_family"]), str(spec["units"]))
        groups.setdefault(key, []).append(axis)
    output: list[dict[str, Any]] = []
    for (family, units), rows in sorted(groups.items()):
        numeric = sorted((row for row in rows if row.get("first_failure_margin") is not None),
                         key=lambda row: float(row["first_failure_margin"]["value"]))
        ranks = {row["specification"]["axis_id"]: index + 1 for index, row in enumerate(numeric)}
        for axis in rows:
            spec = axis["specification"]
            first = _first_fail(axis)
            primary_boundary = next((item for item in axis.get("all_refined_boundaries", []) if item.get("transition") == "PASS_TO_FAIL"), None)
            margin = axis.get("first_failure_margin")
            output.append({
                "family": family, "unit_family": units, "rank_within_family_and_unit": ranks.get(spec["axis_id"]),
                "axis_id": spec["axis_id"], "direction": spec["direction"],
                "margin": None if margin is None else margin.get("value"), "units": units,
                "last_pass": None if primary_boundary is None else primary_boundary.get("last_pass", {}).get("magnitude"),
                "first_fail": None if primary_boundary is None else primary_boundary.get("first_fail", {}).get("magnitude"),
                "critical_waypoint": None if first is None else first.get("critical_waypoint"),
                "critical_segment": None if first is None else first.get("critical_segment"),
                "failure_mode": None if first is None else first.get("dominant_failure_mode"),
                "search_completeness": axis.get("search_completeness"),
                "non_monotonic_failure_region_observed": axis.get("non_monotonic_failure_region_observed", False),
                "failure_region_count": len(axis.get("all_failure_intervals", [])),
            })
    return output


def family_control(axes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for family in sorted({row["specification"]["perturbation_family"] for row in axes}):
        rows = [row for row in axes if row["specification"]["perturbation_family"] == family
                and row.get("first_failure_margin") is not None]
        rows.sort(key=lambda row: float(row["first_failure_margin"]["value"]))
        controller = rows[0] if rows else None
        first = None if controller is None else _first_fail(controller)
        result[family] = {
            "controlling_axis": None if controller is None else controller["specification"]["axis_id"],
            "margin": None if controller is None else controller["first_failure_margin"]["value"],
            "units": None if controller is None else controller["specification"]["units"],
            "critical_waypoint": None if first is None else first.get("critical_waypoint"),
            "critical_segment": None if first is None else first.get("critical_segment"),
            "failure_mode": None if first is None else first.get("dominant_failure_mode"),
            "status": "MEASURED_FIRST_FAILURE" if controller else "UNRESOLVED_NO_MARGIN_OR_FAILURE",
        }
    return result


def critical_location_distribution(axes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    waypoints: dict[str, int] = {}
    segments: dict[str, int] = {}
    for axis in axes:
        first = _first_fail(axis)
        if first is None or first.get("status") != "FAIL":
            continue
        if first.get("critical_waypoint") is not None:
            key = str(first["critical_waypoint"])
            waypoints[key] = waypoints.get(key, 0) + 1
        if first.get("critical_segment") is not None:
            key = str(first["critical_segment"])
            segments[key] = segments.get(key, 0) + 1
    return {"first_failure_critical_waypoint_counts": waypoints,
            "first_failure_critical_segment_counts": segments,
            "WP0_16_axis_count": sum(count for wp, count in waypoints.items() if 0 <= int(wp) <= 16),
            "WP17_180_axis_count": sum(count for wp, count in waypoints.items() if 17 <= int(wp) <= 180)}


def wp0_16_became_controlling(controllers: Mapping[str, Any]) -> str:
    observed = [row.get("critical_waypoint") for row in controllers.values() if row.get("status") == "MEASURED_FIRST_FAILURE"]
    if not observed:
        return "UNRESOLVED"
    return "YES" if any(value is not None and 0 <= int(value) <= 16 for value in observed) else "NO"


def build_stage4b_findings(axes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    observations: list[dict[str, Any]] = []
    for axis in axes:
        spec = axis["specification"]
        for row in axis.get("coarse_observations", []) + axis.get("refinement_observations", []):
            if row.get("status") != "FAIL":
                continue
            evidence = row.get("evidence", {})
            for mode in row.get("failure_modes", []):
                observations.append({"mode": mode, "axis_id": spec["axis_id"], "family": spec["perturbation_family"],
                                    "candidate_id": row.get("candidate_id"), "magnitude": row.get("magnitude"),
                                    "units": spec["units"], "critical_waypoint": row.get("critical_waypoint"),
                                    "critical_segment": row.get("critical_segment"), "evidence": evidence,
                                    "failure_margin": row.get("failure_margin", {})})
    severity = {
        "ENVIRONMENT_COLLISION": (5, "Potential contact exposure; adaptive-discrete evidence only."),
        "SELF_COLLISION": (5, "Potential self-contact exposure; adaptive-discrete evidence only."),
        "ROBOT_WORLD_TRANSITION_COLLISION": (5, "Endpoint-pair transition query found contact; not a strict global CCD conclusion."),
        "JOINT_LIMIT_FAILURE": (4, "Joint-limit constraint excursion; software model only."),
        "VELOCITY_LIMIT": (4, "Configured velocity limit excursion; configured limits are not vendor-certified."),
        "ACCELERATION_LIMIT": (4, "Configured acceleration limit excursion; configured limits are not vendor-certified."),
        "JERK_LIMIT": (4, "Configured jerk limit excursion; configured limits are not vendor-certified."),
        "JOINT_DISCONTINUITY": (4, "Configured joint-step gate excursion."),
        "TCP_PATH_DEVIATION": (3, "Configured TCP path gate excursion."),
        "TERMINAL_POSITION_ERROR": (3, "Configured terminal position gate excursion."),
        "SPRAY_AXIS_NORMAL_ERROR": (3, "Configured spray-axis normal gate excursion; no deposition model."),
    }
    taxonomy: list[dict[str, Any]] = []
    for mode in sorted({row["mode"] for row in observations}):
        rows = [row for row in observations if row["mode"] == mode]
        axes_by_family: dict[str, set[str]] = {}
        for row in rows:
            axes_by_family.setdefault(row["family"], set()).add(row["axis_id"])
        waypoint_counts: dict[str, int] = {}
        segment_counts: dict[str, int] = {}
        for row in rows:
            if row["critical_waypoint"] is not None:
                key = str(row["critical_waypoint"])
                waypoint_counts[key] = waypoint_counts.get(key, 0) + 1
            if row["critical_segment"] is not None:
                key = str(row["critical_segment"])
                segment_counts[key] = segment_counts.get(key, 0) + 1
        fatal_margin_values = [(key, float(value), row) for row in rows for key, value in row["failure_margin"].items()
                               if isinstance(value, (int, float)) and math.isfinite(float(value))]
        worst = max(fatal_margin_values, key=lambda item: item[1]) if fatal_margin_values else None
        magnitude_by_family: dict[str, Any] = {}
        for family in sorted({row["family"] for row in rows}):
            family_rows = [row for row in rows if row["family"] == family]
            chosen = max(family_rows, key=lambda row: float(row["magnitude"]))
            magnitude_by_family[family] = {"value": chosen["magnitude"], "units": chosen["units"],
                                           "axis_id": chosen["axis_id"], "candidate_id": chosen["candidate_id"]}
        severity_rank, safety_impact = severity.get(mode, (2, "Measured gate failure; safety impact not determined."))
        metric_units = None if worst is None else (
            "m" if worst[0].endswith("_m") else
            "rad" if worst[0].endswith("_rad") else
            "ratio" if "ratio" in worst[0] else
            "count" if "count" in worst[0] else "metric_specific"
        )
        taxonomy.append({
            "problem_category": mode, "affected_benchmark_families": sorted(axes_by_family),
            "affected_axis_ids": sorted({row["axis_id"] for row in rows}),
            "affected_case_ids": sorted({row["candidate_id"] for row in rows}),
            "observed_failure_count": len(rows), "sampled_frequency_denominator": sum(
                len(axis.get("coarse_observations", [])) + len(axis.get("refinement_observations", [])) for axis in axes
            ),
            "critical_waypoint_frequency": waypoint_counts, "critical_segment_frequency": segment_counts,
            "worst_sampled_perturbation_magnitude_by_family": magnitude_by_family,
            "worst_failure_margin_metric": None if worst is None else {"metric": worst[0], "value": worst[1], "units": metric_units,
                                                                       "candidate_id": worst[2]["candidate_id"],
                                                                       "axis_id": worst[2]["axis_id"],
                                                                       "critical_waypoint": worst[2]["critical_waypoint"],
                                                                       "critical_segment": worst[2]["critical_segment"]},
            "severity_rank_for_stage4b_prioritization": severity_rank,
            "reproducibility": "REPEATED_DETERMINISTIC_CAMPAIGN_CASES" if len(rows) > 1 else "ONE_MEASURED_CASE",
            "safety_impact": safety_impact,
            "likely_subsystem": "inference from failure mode; requires Stage4B confirmation",
            "measurement_confidence": "HIGH_WITHIN_AUTHENTICATED_SOFTWARE_EVALUATOR" if mode not in {"ROBOT_WORLD_TRANSITION_COLLISION"} else "BOUNDED_ENDPOINT_PAIR_DIAGNOSTIC",
        })
    taxonomy.sort(key=lambda row: (-row["severity_rank_for_stage4b_prioritization"], -row["observed_failure_count"], row["problem_category"]))
    ranked = [dict(row, priority_rank=index + 1) for index, row in enumerate(taxonomy[:3])]
    return {"taxonomy": taxonomy, "ranked": ranked}


def publish_campaign(args: argparse.Namespace) -> dict[str, Any]:
    campaign = read_json(args.campaign_result.resolve())
    if campaign.get("schema") != "p2b3-c2-c1-robustness-transfer-v1":
        raise RuntimeError("campaign_result_schema_mismatch")
    outputs = args.output_dir.resolve()
    outputs.mkdir(parents=True, exist_ok=True)
    result_path = outputs / "p2b3_c2_result.json"
    profile_path = outputs / "p2b3_c2_axis_margin_profile.csv"
    comparison_path = outputs / "p2b3_c2_transfer_comparison.csv"
    write_json(result_path, campaign)
    write_machine_csv(profile_path, campaign["MARGIN_PROFILE"])
    write_machine_csv(comparison_path, campaign["TRANSFER_COMPARISON"])
    return {"result": str(result_path), "profile": str(profile_path), "comparison": str(comparison_path),
            "status": campaign["P2B3_C2_STATUS"]}


def write_machine_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing_empty_canonical_csv:{path.name}")
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True, allow_nan=False) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def resolve_artifact_commit(args: argparse.Namespace) -> dict[str, Any]:
    result_path = args.result.resolve()
    result = read_json(result_path)
    artifact_commit = args.artifact_publish_commit
    if not artifact_commit or len(artifact_commit) != 40 or any(char not in "0123456789abcdef" for char in artifact_commit):
        raise ValueError("artifact_publish_commit_must_be_an_exact_40_character_git_commit")
    result["C2_ARTIFACT_PUBLISH_COMMIT"] = {
        "status": "RESOLVED_IN_SECOND_PHASE",
        "phase_1_execution_code_commit": result["C2_EXECUTION_CODE_COMMIT"],
        "phase_2_first_canonical_artifact_commit": artifact_commit,
        "phase_2_resolution_commit_is_a_later_commit_to_avoid_self_reference": True,
    }
    write_json(result_path, result)
    return {"result": str(result_path), "artifact_publish_commit": artifact_commit,
            "result_sha256": sha256(result_path)}


def run_tests() -> dict[str, Any]:
    command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
               "tests/test_p2a_axiswise_robustness.py", "tests/test_p2b3_c1_publisher.py",
               "tests/test_p2b3_c1_ordered_validation.py", "tests/test_p2b3_c1_scope.py",
               "tests/test_p2b3_c2_runner.py", "tests/test_process_aware_stress.py",
               "tests/test_p2b2_robustness_remapping.py", "tests/test_p2b1_solver_policy_ablation.py"]
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(command, cwd=ROOT, env=env, text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=1800)
    if completed.returncode:
        raise RuntimeError("C2_targeted_tests_failed:\n" + completed.stdout + "\n" + completed.stderr)
    return {"status": "PASS", "command": " ".join(command), "stdout_tail": completed.stdout.strip().splitlines()[-3:]}


def verify_fairino_source(checkout: Path, expected_commit: str) -> dict[str, Any]:
    root = checkout.resolve()
    actual_commit = git_at(root, "rev-parse", "HEAD")
    remote = git_at(root, "remote", "get-url", "origin")
    dirty = git_at(root, "status", "--porcelain")
    if actual_commit != expected_commit or dirty or remote.rstrip("/") != "https://github.com/FAIR-INNOVATION/frcobot_ros2.git":
        raise RuntimeError(f"official_FAIRINO_source_checkout_identity_mismatch:{actual_commit}:{remote}:{bool(dirty)}")
    return {"remote": remote, "commit": actual_commit, "working_tree_clean": not bool(dirty)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("test", "campaign", "publish", "resolve-provenance"), required=True)
    parser.add_argument("--c1-result", type=Path, default=C1_RESULT)
    parser.add_argument("--c1-lineage", type=Path, default=C1_LINEAGE)
    parser.add_argument("--c1-fk-trace", type=Path, default=ROOT / "outputs/p2b3_c1_r0_fk_trace.csv")
    parser.add_argument("--nominal", type=Path, default=C1_NOMINAL)
    parser.add_argument("--old-landscape", type=Path, default=OLD_LANDSCAPE)
    parser.add_argument("--old-nominal", type=Path, default=ROOT / "outputs/p2b2_inputs/p2b1_b0_reference/strict_replay/moveit_smoothed_joint_trajectory.csv")
    parser.add_argument("--fairino-source-checkout", type=Path, default=Path(r"D:\fr5-p2b3-c2-20260928\fairino_source"))
    parser.add_argument("--d46-root", type=Path, default=Path(r"D:\robotfucker\outputs\D46_STAGE4A_SYSTEM_BASELINE_V1"))
    parser.add_argument("--scratch", type=Path)
    parser.add_argument("--campaign-output", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--campaign-result", type=Path)
    parser.add_argument("--result", type=Path, default=ROOT / "outputs/p2b3_c2_result.json")
    parser.add_argument("--artifact-publish-commit")
    parser.add_argument("--urdf", type=Path, default=URDF_DEFAULT)
    parser.add_argument("--srdf", type=Path, default=SRDF_DEFAULT)
    parser.add_argument("--underlay-install", type=Path)
    parser.add_argument("--overlay-install", type=Path)
    parser.add_argument("--native-binary", type=Path)
    parser.add_argument("--fk-binary", type=Path)
    parser.add_argument("--distro", default="Ubuntu-24.04-D")
    parser.add_argument("--native-timeout-seconds", type=int, default=7200)
    parser.add_argument("--max-native-batches", type=int, default=64)
    parser.add_argument("--max-campaign-hours", type=int, default=12)
    args = parser.parse_args()
    if args.mode == "test":
        result = run_tests()
    elif args.mode == "campaign":
        required_args = (args.scratch, args.campaign_output, args.underlay_install, args.overlay_install, args.native_binary, args.fk_binary)
        if any(value is None for value in required_args):
            parser.error("campaign requires --scratch, --campaign-output, --underlay-install, --overlay-install, --native-binary, and --fk-binary")
        result = run_campaign(args)
    elif args.mode == "publish":
        if args.campaign_result is None:
            parser.error("publish requires --campaign-result")
        result = publish_campaign(args)
    else:
        if not args.artifact_publish_commit:
            parser.error("resolve-provenance requires --artifact-publish-commit")
        result = resolve_artifact_commit(args)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
