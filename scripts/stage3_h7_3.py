#!/usr/bin/env python3
"""Stage 3 H7.3 target-55 controlled-stop TOTG/Ruckig recertification.

This runner is additive and fail-closed.  It treats the passed H6.4 output and
the authoritative blocked H7.2 output as immutable inputs, derives only the
authorized 3A/3B timing boundary, and never invokes robot execution.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import stage3_h6_4 as h64
from scripts import stage3_h7_2 as h72


H72_ROOT = ROOT / "outputs/stage3_h7_2_authoritative_time_parameterization_20260809T060043Z"
H72_TERMINAL = H72_ROOT / "stage3_h7_2_terminal_certificate.json"
PRIMITIVE_ORDER: list[int | str] = [0, 1000, 1, 1001, 2, 1002, "3A", "3B", 1003, 4]
PRIMITIVE_COUNTS = [221, 12, 40, 426, 4, 241, 194, 104, 361, 25]
COLLISION_METHOD = "adaptive_discrete_interpolation"
ANGLE_TOLERANCE = 1.0e-5

REQUIRED_OUTPUTS = [
    "FINAL_REPORT.md",
    "stage3_h7_3_terminal_certificate.json",
    "stage3_h7_3_gate_report.json",
    "stage3_h7_3_input_freeze_manifest.json",
    "stage3_h7_3_authoritative_primitive_order.json",
    "stage3_h7_3_target55_cusp_audit.json",
    "stage3_h7_3_pre_totg_filtered_path_audit.json",
    "stage3_h7_3_totg_parameters.json",
    "stage3_h7_3_totg_primitive_results.json",
    "stage3_h7_3_time_parameterized_trajectories.jsonl",
    "stage3_h7_3_post_totg_validation.json",
    "stage3_h7_3_ruckig_limits_audit.json",
    "stage3_h7_3_ruckig_results.json",
    "stage3_h7_3_ruckig_trajectories.jsonl",
    "stage3_h7_3_post_ruckig_validation.json",
    "stage3_h7_3_replay_determinism.json",
    "stage3_h7_3_regression_report.json",
    "stage3_h7_3_immutable_after_check.json",
]


def source_snapshot() -> dict[str, Any]:
    files = {
        "h6_4_tree": h64.tree_snapshot(h72.H64_ROOT),
        "h7_2_tree": h64.tree_snapshot(H72_ROOT),
        "derived_urdf": h72.file_record(h72.DERIVED_URDF, "frozen derived robot model"),
        "fixture_mesh": h72.file_record(h72.FIXTURE_MESH, "frozen collision fixture"),
        "srdf": h72.file_record(h72.SRDF, "frozen semantic robot model"),
        "joint_limits": h72.file_record(h72.JOINT_LIMITS, "frozen velocity/acceleration/jerk limits"),
        "process_contract": h72.file_record(h72.PROCESS_CONTRACT, "frozen H6.1 process contract"),
        "metric_contract": h72.file_record(h72.METRIC_CONTRACT, "frozen Stage 3 metric contract"),
        "gate_contract": h72.file_record(h72.GATE_CONTRACT, "frozen Stage 3 gate contract"),
    }
    return files


def immutable_compare(before: Mapping[str, Any]) -> dict[str, Any]:
    after = source_snapshot()
    checks = {name: before.get(name) == after.get(name) for name in before}
    return {
        "schema_version": "stage3-h7-3-immutable-after-check-v1",
        "all_after_hashes_match": all(checks.values()),
        "checks": checks,
        "after": after,
    }


def audit_h7_2() -> dict[str, Any]:
    if not H72_TERMINAL.is_file():
        return {"status": "BLOCKED", "errors": ["authoritative_h7_2_terminal_missing"]}
    terminal = h72.load_json(H72_TERMINAL)
    errors: list[str] = []
    expected = {
        "STAGE_3_H7_2": "BLOCKED",
        "FIRST_BLOCKER": "segment_3:native_totg_path_requires_180_degree_turn",
        "TOTG_SEGMENTS_PASSED": "6/9",
        "READY_FOR_STAGE_3_H8": "NO",
    }
    for key, value in expected.items():
        if terminal.get(key) != value:
            errors.append(f"h7_2_terminal_mismatch:{key}:{terminal.get(key)}:{value}")
    return {
        "status": "PASSED" if not errors else "BLOCKED",
        "errors": errors,
        "authoritative_root": h72.rel(H72_ROOT),
        "terminal": terminal,
    }


def _filter_indices(q: Sequence[Sequence[float]], min_angle_change: float) -> tuple[list[int], list[int]]:
    kept: list[list[float]] = []
    kept_indices: list[int] = []
    removed_or_replaced: list[int] = []
    for index, raw in enumerate(q):
        point = [float(value) for value in raw]
        diverse = index == 0 or any(abs(point[j] - kept[-1][j]) > min_angle_change for j in range(len(point)))
        if diverse:
            kept.append(point)
            kept_indices.append(index)
        elif index == len(q) - 1:
            kept[-1] = point
            kept_indices[-1] = index
            removed_or_replaced.append(index)
        else:
            removed_or_replaced.append(index)
    return kept_indices, removed_or_replaced


def _turn_record(q: Sequence[Sequence[float]], indices: Sequence[int]) -> dict[str, Any]:
    vectors = [[float(value) for value in q[index]] for index in indices]
    incoming = [vectors[1][j] - vectors[0][j] for j in range(6)]
    outgoing = [vectors[2][j] - vectors[1][j] for j in range(6)]
    incoming_norm = math.sqrt(sum(value * value for value in incoming))
    outgoing_norm = math.sqrt(sum(value * value for value in outgoing))
    dot = sum(left * right for left, right in zip(incoming, outgoing))
    cosine = max(-1.0, min(1.0, dot / (incoming_norm * outgoing_norm)))
    return {
        "joint_vectors_rad": vectors,
        "incoming_tangent_rad": incoming,
        "outgoing_tangent_rad": outgoing,
        "incoming_norm_rad": incoming_norm,
        "outgoing_norm_rad": outgoing_norm,
        "dot_product_rad2": dot,
        "cosine": cosine,
        "turn_angle_deg": math.degrees(math.acos(cosine)),
    }


def target55_cusp_audit() -> dict[str, Any]:
    manifest = h72.load_json(h72.H64_SEGMENTS)
    segment_meta = next(item for item in manifest["segments"] if int(item["segment_id"]) == 3)
    rows = [row for row in h72.load_jsonl(h72.H64_VALIDATION) if int(row["segment_id"]) == 3]
    q = [[float(value) for value in row["selected_joint_values"]] for row in rows]
    kept, removed = _filter_indices(q, 0.0005)
    filtered_position = kept.index(193)
    exposed = kept[filtered_position - 1 : filtered_position + 2]
    displacement = [q[194][j] - q[193][j] for j in range(6)]
    displacement_norm = math.sqrt(sum(value * value for value in displacement))
    max_joint_change = max(abs(value) for value in displacement)
    turn = _turn_record(q, exposed)
    surface_delta = math.sqrt(
        sum((float(rows[194]["surface_point"][j]) - float(rows[193]["surface_point"][j])) ** 2 for j in range(3))
    )
    storage = [int(segment_meta["waypoint_start"]) + index for index in exposed]
    process = [int(rows[index]["waypoint_index"]) for index in exposed]
    confirmed = bool(
        194 in removed
        and exposed == [192, 193, 195]
        and storage == [457, 458, 460]
        and process == [1136, 1137, 1139]
        and turn["cosine"] <= -1.0 + ANGLE_TOLERANCE
        and int(rows[193]["source_target_id"]) == 47
        and int(rows[193]["destination_target_id"]) == 55
        and int(rows[194]["source_target_id"]) == 55
        and int(rows[194]["destination_target_id"]) == 2
        and surface_delta <= 1.0e-12
    )
    return {
        "schema_version": "stage3-h7-3-target55-cusp-audit-v1",
        "status": "PASSED" if confirmed else "BLOCKED",
        "classification": "TARGET_55_FILTER_EXPOSED_REVERSAL_CUSP" if confirmed else "NOT_CONFIRMED",
        "min_angle_change_rad": 0.0005,
        "local_193_to_194_joint_delta_rad": displacement,
        "local_193_to_194_joint_space_displacement_rad": displacement_norm,
        "local_193_to_194_max_single_joint_change_rad": max_joint_change,
        "local_194_filtered": 194 in removed,
        "filtered_segment_local_triple": exposed,
        "h6_4_storage_triple": storage,
        "process_order_triple": process,
        **turn,
        "local_193_edge": "47→55",
        "local_194_edge": "55→2",
        "local_193_surface_point_m": rows[193]["surface_point"],
        "local_194_surface_point_m": rows[194]["surface_point"],
        "surface_point_difference_m": surface_delta,
        "local_195_returns_along_previous_direction": turn["cosine"] <= -1.0 + ANGLE_TOLERANCE,
        "moveit_filter_semantics": "keep first; compare every later point to the last kept point; keep iff any active joint changes by more than min_angle_change; preserve the final endpoint by replacement",
        "h6_4_joint_values_modified": False,
    }


def authoritative_primitives() -> dict[str, Any]:
    original = h72.authoritative_segments()
    by_id = {int(item["segment_id"]): item for item in original["segments"]}
    primitives: list[dict[str, Any]] = []
    for execution_order, primitive_id in enumerate(PRIMITIVE_ORDER):
        if primitive_id == "3A":
            source, local_start, local_end = by_id[3], 0, 193
        elif primitive_id == "3B":
            source, local_start, local_end = by_id[3], 194, 297
        else:
            source = by_id[int(primitive_id)]
            local_start, local_end = 0, int(source["input_waypoint_count"]) - 1
        primitives.append(
            {
                "execution_order": execution_order,
                "primitive_id": primitive_id,
                "h6_4_segment_id": int(source["segment_id"]),
                "spray_state": source["spray_state"],
                "h6_4_segment_local_start": local_start,
                "h6_4_segment_local_end": local_end,
                "h6_4_storage_waypoint_start": int(source["h6_4_storage_waypoint_start"]) + local_start,
                "h6_4_storage_waypoint_end": int(source["h6_4_storage_waypoint_start"]) + local_end,
                "input_waypoint_count": local_end - local_start + 1,
                "independent_robot_trajectory": True,
                "required_start_velocity_rad_s": 0.0,
                "required_end_velocity_rad_s": 0.0,
                "h6_4_joint_values_preserved": True,
            }
        )
    errors: list[str] = []
    if original["status"] != "PASSED":
        errors.append("h6_4_authoritative_segment_audit_failed")
    if [item["input_waypoint_count"] for item in primitives] != PRIMITIVE_COUNTS:
        errors.append("derived_primitive_count_mismatch")
    if sum(item["input_waypoint_count"] for item in primitives) != 1628:
        errors.append("derived_waypoint_total_changed")
    return {
        "schema_version": "stage3-h7-3-authoritative-primitive-order-v1",
        "status": "PASSED" if not errors else "BLOCKED",
        "h6_4_process_segment_count": 9,
        "h7_3_time_parameterization_primitive_count": 10,
        "spray_on_segments": 5,
        "spray_off_segments": 4,
        "authoritative_primitive_order": PRIMITIVE_ORDER,
        "primitives": primitives,
        "target_55_boundary": {"left": "3A:end_local_193", "right": "3B:start_local_194"},
        "does_not_rewrite_h6_4_segmentation": True,
        "errors": errors,
    }


def pre_totg_filtered_path_audit(order: Mapping[str, Any]) -> dict[str, Any]:
    rows = h72.load_jsonl(h72.H64_VALIDATION)
    grouped: dict[int, list[Mapping[str, Any]]] = {segment_id: [] for segment_id in h72.SEGMENT_ORDER}
    for row in rows:
        grouped[int(row["segment_id"])].append(row)
    audits: list[dict[str, Any]] = []
    total_turns = 0
    for primitive in order["primitives"]:
        source = grouped[int(primitive["h6_4_segment_id"])]
        start = int(primitive["h6_4_segment_local_start"])
        end = int(primitive["h6_4_segment_local_end"])
        selected = source[start : end + 1]
        q = [row["selected_joint_values"] for row in selected]
        kept, removed = _filter_indices(q, 0.0005)
        turns: list[dict[str, Any]] = []
        for index in range(1, len(kept) - 1):
            triple = kept[index - 1 : index + 2]
            record = _turn_record(q, triple)
            if record["cosine"] <= -1.0 + ANGLE_TOLERANCE:
                turns.append({"primitive_local_source_indices": triple, **record})
        total_turns += len(turns)
        audits.append(
            {
                "primitive_id": primitive["primitive_id"],
                "input_waypoint_count": len(q),
                "filtered_waypoint_count": len(kept),
                "removed_or_replaced_primitive_local_indices": removed,
                "unsupported_180_turn_count": len(turns),
                "unsupported_180_turns": turns,
            }
        )
    return {
        "schema_version": "stage3-h7-3-pre-totg-filtered-path-audit-v1",
        "status": "PASSED" if total_turns == 0 else "BLOCKED",
        "min_angle_change_rad": 0.0005,
        "angle_gate_cosine_threshold": -1.0 + ANGLE_TOLERANCE,
        "moveit_filtered_180_turn_count": total_turns,
        "primitive_audits": audits,
        "native_worker_repeats_audit_after_RobotTrajectory_unwind": True,
    }


def totg_parameters() -> dict[str, Any]:
    frozen = h72.load_json(H72_ROOT / "stage3_h7_2_totg_parameters.json")
    required = {
        "path_tolerance": 0.00025,
        "resample_dt": 0.01,
        "min_angle_change": 0.0005,
        "velocity_scaling_factor": 0.15,
        "acceleration_scaling_factor": 0.15,
    }
    mismatches = {key: {"h7_2": frozen.get(key), "required": value} for key, value in required.items() if frozen.get(key) != value}
    return {
        **frozen,
        "schema_version": "stage3-h7-3-totg-parameters-v1",
        "status": "FROZEN_FROM_AUTHORITATIVE_H7_2" if not mismatches else "BLOCKED_H7_2_PARAMETER_MISMATCH",
        "parameterization_scope": "ten independent RobotTrajectory motion primitives; no fitting across 3A→3B or any other primitive boundary",
        "h7_2_parameter_mismatches": mismatches,
        "h7_2_source": h72.file_record(H72_ROOT / "stage3_h7_2_totg_parameters.json", "authoritative frozen H7.2 parameters"),
    }


def build_worker(output: Path) -> dict[str, Any]:
    build_base = ROOT / "build/stage3_h7_3_native"
    install_base = ROOT / "install/stage3_h7_3_native"
    command = " && ".join(
        [
            "source /opt/ros/jazzy/setup.bash",
            f"source {shlex.quote(h72.wsl_path(ROOT / 'install/setup.bash'))}",
            f"colcon build --base-paths {shlex.quote(h72.wsl_path(ROOT / 'ros2_moveit_bridge'))} --build-base {shlex.quote(h72.wsl_path(build_base))} --install-base {shlex.quote(h72.wsl_path(install_base))} --merge-install",
        ]
    )
    code, stdout, stderr = h72.run_wsl(command, 1200)
    h72.dump_text(output / "stage3_h7_3_native_build.log", stdout + "\n--- STDERR ---\n" + stderr)
    result = {
        "status": "PASSED" if code == 0 else "BLOCKED",
        "returncode": code,
        "build_base": h72.rel(build_base),
        "install_base": h72.rel(install_base),
        "entrypoint": "stage3_h7_3_native",
    }
    h72.dump_json(output / "stage3_h7_3_native_build.json", result)
    return result


def run_replay(output: Path, replay_index: int, ruckig_authorized: bool) -> dict[str, Any]:
    replay_dir = output / "fresh_process_replays" / f"replay_{replay_index}"
    replay_dir.mkdir(parents=True, exist_ok=True)
    command = " && ".join(
        [
            "source /opt/ros/jazzy/setup.bash",
            f"source {shlex.quote(h72.wsl_path(ROOT / 'install/setup.bash'))}",
            f"source {shlex.quote(h72.wsl_path(ROOT / 'install/stage3_h7_3_native/setup.bash'))}",
            "ros2 launch fr5_tunnel_moveit_bridge stage3_h7_3_native.launch.py "
            f"output_dir:={shlex.quote(h72.wsl_path(replay_dir))} "
            f"h6_4_final_validation:={shlex.quote(h72.wsl_path(h72.H64_VALIDATION))} "
            f"h6_4_segments:={shlex.quote(h72.wsl_path(h72.H64_SEGMENTS))} "
            f"process_contract:={shlex.quote(h72.wsl_path(h72.PROCESS_CONTRACT))} "
            f"fixture_mesh:={shlex.quote(h72.wsl_path(h72.FIXTURE_MESH))} "
            f"totg_parameters:={shlex.quote(h72.wsl_path(output / 'stage3_h7_3_totg_parameters.json'))} "
            f"ruckig_authorized:={'true' if ruckig_authorized else 'false'}",
        ]
    )
    code, stdout, stderr = h72.run_wsl(command, 3600)
    h72.dump_text(replay_dir / "launch.log", stdout + "\n--- STDERR ---\n" + stderr)
    result_path = replay_dir / "native_result.json"
    result = h72.load_json(result_path) if result_path.is_file() else {
        "status": "BLOCKED",
        "first_blocker": "native_result_missing",
        "primitive_results": [],
    }
    combined_log = stdout + "\n" + stderr
    exit_codes = [int(value) for value in re.findall(r"process has died .*?exit code (-?\d+)", combined_log)]
    shutdown_clean = all(value >= 0 for value in exit_codes)
    ruckig_errors = combined_log.count(
        "Ruckig extended the trajectory duration to its maximum and still did not find a solution"
    )
    result.update(
        {
            "fresh_process_index": replay_index,
            "launcher_returncode": code,
            "native_process_exit_codes_observed": exit_codes,
            "native_process_clean_shutdown": shutdown_clean,
            "moveitpy_shutdown_clean": shutdown_clean,
            "ruckig_native_no_solution_log_count": ruckig_errors,
        }
    )
    if ruckig_errors:
        result["status"] = "BLOCKED"
        result["first_blocker"] = result.get("first_blocker") or "native_ruckig_reported_no_solution"
    if not shutdown_clean:
        result["status"] = "BLOCKED"
        result["first_blocker"] = result.get("first_blocker") or "moveitpy_shutdown_not_clean"
    h72.dump_json(result_path, result)
    return result


def copy_primary_outputs(output: Path) -> None:
    first = output / "fresh_process_replays/replay_1"
    shutil.copyfile(first / "totg_trajectories.jsonl", output / "stage3_h7_3_time_parameterized_trajectories.jsonl")
    shutil.copyfile(first / "ruckig_trajectories.jsonl", output / "stage3_h7_3_ruckig_trajectories.jsonl")
    if (first / "native_pre_totg_filtered_path_audit.json").is_file():
        shutil.copyfile(
            first / "native_pre_totg_filtered_path_audit.json",
            output / "stage3_h7_3_native_pre_totg_filtered_path_audit.json",
        )


def aggregate_native(output: Path, native: Mapping[str, Any], jerk: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    primitives = list(native.get("primitive_results", []))
    complete = len(primitives) == 10
    native_ruckig_no_solution_count = int(native.get("ruckig_native_no_solution_log_count", 0))
    native_ruckig_completion = native_ruckig_no_solution_count == 0
    totg_passed = sum(item.get("post_totg_status") == "PASSED" for item in primitives)
    totg_results = {
        "schema_version": "stage3-h7-3-totg-primitive-results-v1",
        "primitive_count": 10,
        "primitives_passed": totg_passed,
        "primitives": primitives,
    }
    h72.dump_json(output / "stage3_h7_3_totg_primitive_results.json", totg_results)
    post_totg = {
        "schema_version": "stage3-h7-3-post-totg-validation-v1",
        "status": "PASSED" if complete and all(item.get("post_totg_status") == "PASSED" for item in primitives) else "BLOCKED",
        "process_tolerance_status": "PASSED" if complete and all(item.get("post_totg_process", {}).get("status") == "PASSED" for item in primitives) else "BLOCKED",
        "collision_certification_status": "PASSED" if complete and all(item.get("post_totg_collision", {}).get("status") == "PASSED" for item in primitives) else "BLOCKED",
        "collision_method": COLLISION_METHOD,
        "CCD": "NOT_AVAILABLE",
        "CLEARANCE": "NOT_AVAILABLE",
        "interpolated_sample_count": sum(item.get("post_totg_collision", {}).get("checked_state_count", 0) for item in primitives),
        "self_collision_failures": sum(item.get("post_totg_collision", {}).get("self_collision_failure_count", 0) for item in primitives),
        "environment_collision_failures": sum(item.get("post_totg_collision", {}).get("environment_collision_failure_count", 0) for item in primitives),
    }
    h72.dump_json(output / "stage3_h7_3_post_totg_validation.json", post_totg)
    ruckig_results = {
        "schema_version": "stage3-h7-3-ruckig-results-v1",
        "limits_audit": jerk,
        "primitives": primitives,
        "native_api_returned_true_primitives": sum(item.get("ruckig_status") == "PASSED" for item in primitives),
        "native_no_solution_log_count": native_ruckig_no_solution_count,
        "native_smoothing_completion_status": "PASSED" if native_ruckig_completion else "BLOCKED",
        "certification_status": "PASSED"
        if complete and native_ruckig_completion and all(item.get("status") == "PASSED" for item in primitives)
        else "BLOCKED",
        "fail_closed_note": "MoveIt Jazzy may return true when the final low-level Ruckig result is acceptable even though the overall duration-extension loop did not set smoothing_complete; the explicit maximum-duration/no-solution log is authoritative.",
    }
    h72.dump_json(output / "stage3_h7_3_ruckig_results.json", ruckig_results)
    post_ruckig = {
        "schema_version": "stage3-h7-3-post-ruckig-validation-v1",
        "status": "PASSED"
        if complete and native_ruckig_completion and all(item.get("status") == "PASSED" for item in primitives)
        else "BLOCKED",
        "native_api_returned_true_primitives": sum(item.get("ruckig_status") == "PASSED" for item in primitives),
        "native_no_solution_log_count": native_ruckig_no_solution_count,
        "native_smoothing_completion_status": "PASSED" if native_ruckig_completion else "BLOCKED",
        "collision_method": COLLISION_METHOD,
        "CCD": "NOT_AVAILABLE",
        "CLEARANCE": "NOT_AVAILABLE",
        "zero_velocity_primitive_boundaries": complete and all(
            item.get("post_ruckig_dynamics", {}).get("zero_velocity_start")
            and item.get("post_ruckig_dynamics", {}).get("zero_velocity_end")
            for item in primitives
        ),
        "start_end_configuration_preserved": complete and all(item.get("start_end_configuration_preserved") for item in primitives),
        "jerk_certification_semantics": "native MoveIt RuckigSmoothing with the project-authoritative 8 rad/s^3 research bounds; these are not FAIRINO manufacturer jerk specifications",
    }
    h72.dump_json(output / "stage3_h7_3_post_ruckig_validation.json", post_ruckig)
    timing_rows = [
        {
            "primitive_id": item.get("primitive_id"),
            "h6_4_segment_id": item.get("h6_4_segment_id"),
            "spray_state": item.get("spray_state"),
            "duration_s": item.get("post_ruckig_duration_s", item.get("trajectory_duration_s")),
            "duration_source": "POST_RUCKIG" if item.get("post_ruckig_duration_s") is not None else "POST_TOTG",
        }
        for item in primitives
    ]
    h72.dump_json(
        output / "stage3_h7_3_process_timing.json",
        {
            "schema_version": "stage3-h7-3-process-timing-v1",
            "primitives": timing_rows,
            "no_cross_3A_3B_fitting_or_smoothing": True,
        },
    )
    return totg_results, post_totg, ruckig_results, post_ruckig


def replay_report(output: Path, runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    hashes = [run.get("semantic_hash") for run in runs]
    counters = [
        {
            "totg": run.get("totg_primitives_passed"),
            "ruckig": run.get("ruckig_primitives_passed"),
            "ruckig_no_solution_logs": run.get("ruckig_native_no_solution_log_count"),
            "status": run.get("status"),
            "filtered_turns": run.get("moveit_filtered_180_turn_count"),
            "moveitpy_shutdown_clean": run.get("moveitpy_shutdown_clean"),
        }
        for run in runs
    ]
    identical = bool(
        len(runs) == 3
        and len(set(hashes)) == 1
        and all(run.get("authoritative_primitive_order") == PRIMITIVE_ORDER for run in runs)
        and len({h72.semantic_hash(item) for item in counters}) == 1
    )
    clean = len(runs) == 3 and all(run.get("moveitpy_shutdown_clean") is True for run in runs)
    report = {
        "schema_version": "stage3-h7-3-replay-determinism-v1",
        "DETERMINISTIC_REPLAY": "3/3" if identical else "0/3",
        "identical": identical,
        "MOVEITPY_SHUTDOWN_CLEAN": "YES" if clean else "NO",
        "semantic_hashes": hashes,
        "validation_counters": counters,
        "native_process_exit_codes": [run.get("native_process_exit_codes_observed", []) for run in runs],
    }
    h72.dump_json(output / "stage3_h7_3_replay_determinism.json", report)
    return report


def run_regression(output: Path) -> dict[str, Any]:
    tests = [
        "tests/test_stage3_h7_3.py",
        "tests/test_stage3_h7_2.py",
        "tests/test_stage3_h7_1.py",
        "tests/test_stage3_h7.py",
        "tests/test_stage3_h6_4.py",
        "tests/test_stage3_h6_3.py",
        "tests/test_stage3_h6_2.py",
        "tests/test_stage3_h6_1.py",
        "tests/test_stage3_h6.py",
        "tests/test_stage3_h5.py",
        "tests/test_stage3_h4_4.py",
        "tests/test_stage3_h4_reachability.py",
        "tests/test_stage3_h3_task_representation.py",
        "tests/test_stage3_h2_geometry.py",
        "tests/test_stage3_h1_contract.py",
    ]
    process = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *tests],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=1800,
        check=False,
    )
    raw = process.stdout + "\n--- STDERR ---\n" + process.stderr
    failed_match = re.search(r"(\d+) failed", raw)
    passed_match = re.search(r"(\d+) passed", raw)
    failures = int(failed_match.group(1)) if failed_match else (0 if process.returncode == 0 else 1)
    report = {
        "schema_version": "stage3-h7-3-regression-report-v1",
        "returncode": process.returncode,
        "passed": int(passed_match.group(1)) if passed_match else None,
        "new_regression_failures": failures,
        "raw_output_tail": raw[-16000:],
    }
    h72.dump_json(output / "stage3_h7_3_regression_report.json", report)
    return report


def terminal_and_report(
    output: Path,
    h6_audit: Mapping[str, Any],
    h7_2_audit: Mapping[str, Any],
    cusp: Mapping[str, Any],
    order: Mapping[str, Any],
    filtered: Mapping[str, Any],
    native: Mapping[str, Any],
    post_totg: Mapping[str, Any],
    post_ruckig: Mapping[str, Any],
    replay: Mapping[str, Any],
    regression: Mapping[str, Any],
    immutable: Mapping[str, Any],
) -> dict[str, Any]:
    primitives = list(native.get("primitive_results", []))
    totg_passed = sum(item.get("post_totg_status") == "PASSED" for item in primitives)
    ruckig_run = any(item.get("ruckig_run") for item in primitives)
    ruckig_pass = post_ruckig.get("status") == "PASSED"
    dynamics = [item.get("post_ruckig_dynamics", item.get("post_totg_dynamics", {})) for item in primitives]
    by_primitive = {str(item.get("primitive_id")): item for item in primitives}
    primitive_3a_totg_dynamics = by_primitive.get("3A", {}).get("post_totg_dynamics", {})
    primitive_3b_totg_dynamics = by_primitive.get("3B", {}).get("post_totg_dynamics", {})
    primitive_3a_end_velocity = primitive_3a_totg_dynamics.get("end_velocity_rad_s", [])
    primitive_3b_start_velocity = primitive_3b_totg_dynamics.get("start_velocity_rad_s", [])
    break_55 = bool(
        by_primitive.get("3A", {}).get("post_ruckig_dynamics", {}).get("zero_velocity_end")
        and by_primitive.get("3B", {}).get("post_ruckig_dynamics", {}).get("zero_velocity_start")
    )
    all_boundaries = bool(post_ruckig.get("zero_velocity_primitive_boundaries"))
    mandatory = {
        "h6_4_input": h6_audit.get("status") == "PASSED",
        "h7_2_input": h7_2_audit.get("status") == "PASSED",
        "target55_cusp": cusp.get("status") == "PASSED",
        "primitive_order": order.get("status") == "PASSED",
        "pre_totg_filtered_path": filtered.get("moveit_filtered_180_turn_count") == 0,
        "native_pre_totg_filtered_path": native.get("moveit_filtered_180_turn_count") == 0,
        "totg_10_of_10": totg_passed == 10,
        "post_totg_process": post_totg.get("process_tolerance_status") == "PASSED",
        "post_totg_collision": post_totg.get("collision_certification_status") == "PASSED",
        "ruckig": ruckig_pass,
        "zero_velocity_boundaries": all_boundaries and break_55,
        "determinism": replay.get("DETERMINISTIC_REPLAY") == "3/3",
        "moveitpy_shutdown": replay.get("MOVEITPY_SHUTDOWN_CLEAN") == "YES",
        "regression": regression.get("new_regression_failures") == 0,
        "immutability": immutable.get("all_after_hashes_match") is True,
    }
    passed = all(mandatory.values())
    first_blocker = None if passed else native.get("first_blocker") or next(
        (name for name, value in mandatory.items() if not value), "mandatory_gate_failed"
    )
    terminal = {
        "schema_version": "stage3-h7-3-terminal-certificate-v1",
        "STAGE_3_H7_3": "PASSED" if passed else "BLOCKED",
        "FIRST_BLOCKER": first_blocker,
        "H6_4_IMMUTABLE": "YES" if immutable.get("checks", {}).get("h6_4_tree") else "NO",
        "H7_2_IMMUTABLE": "YES" if immutable.get("checks", {}).get("h7_2_tree") else "NO",
        "TARGET_55_FILTER_EXPOSED_REVERSAL_CUSP": "CONFIRMED" if cusp.get("status") == "PASSED" else "NOT_CONFIRMED",
        "H6_4_PROCESS_SEGMENT_COUNT": 9,
        "H7_3_TIME_PARAMETERIZATION_PRIMITIVE_COUNT": 10,
        "H7_3_AUTHORITATIVE_PRIMITIVE_ORDER": "0→1000→1→1001→2→1002→3A→3B→1003→4",
        "SPRAY_ON_SEGMENTS": 5,
        "SPRAY_OFF_SEGMENTS": 4,
        "TARGET_55_ZERO_VELOCITY_BREAK_PRESERVED": "YES" if break_55 else "NO",
        "TARGET_55_3A_END_MAX_ABS_VELOCITY_RAD_S": max((abs(float(value)) for value in primitive_3a_end_velocity), default=None),
        "TARGET_55_3B_START_MAX_ABS_VELOCITY_RAD_S": max((abs(float(value)) for value in primitive_3b_start_velocity), default=None),
        "ZERO_VELOCITY_BOUNDARY_TOLERANCE_RAD_S": primitive_3b_totg_dynamics.get("zero_velocity_boundary_tolerance_rad_s"),
        "WAYPOINT_220_221_BREAK_PRESERVED": "YES" if all_boundaries else "NO",
        "TARGET_53_46_BREAK_PRESERVED": "YES" if all_boundaries else "NO",
        "MOVEIT_FILTERED_180_TURN_COUNT": native.get("moveit_filtered_180_turn_count", filtered.get("moveit_filtered_180_turn_count")),
        "TOTG_PRIMITIVES_PASSED": f"{totg_passed}/10",
        "POST_TOTG_PROCESS_TOLERANCE": post_totg.get("process_tolerance_status", "BLOCKED"),
        "POST_TOTG_COLLISION_CERTIFICATION": post_totg.get("collision_certification_status", "BLOCKED"),
        "RUCKIG_RUN": "YES" if ruckig_run else "NO",
        "RUCKIG_CERTIFICATION": "PASSED" if ruckig_pass else ("BLOCKED" if ruckig_run else "NOT_RUN"),
        "RUCKIG_NATIVE_API_TRUE_PRIMITIVES": f"{sum(item.get('ruckig_status') == 'PASSED' for item in primitives)}/10",
        "RUCKIG_NATIVE_NO_SOLUTION_LOG_COUNT": int(native.get("ruckig_native_no_solution_log_count", 0)),
        "RUCKIG_SMOOTHING_COMPLETION": post_ruckig.get("native_smoothing_completion_status", "BLOCKED"),
        "VELOCITY_LIMIT_VIOLATIONS": sum(item.get("velocity_limit_violation_count", 0) for item in dynamics),
        "ACCELERATION_LIMIT_VIOLATIONS": sum(item.get("acceleration_limit_violation_count", 0) for item in dynamics),
        "JERK_LIMIT_VIOLATIONS": 0 if ruckig_pass else "NOT_CERTIFIED",
        "NON_MONOTONIC_TIMESTAMP_COUNT": sum(item.get("non_monotonic_timestamp_count", 0) for item in dynamics),
        "NaN_INF_COUNT": sum(item.get("nan_inf_count", 0) for item in dynamics),
        "DETERMINISTIC_REPLAY": replay.get("DETERMINISTIC_REPLAY", "0/3"),
        "NEW_REGRESSION_FAILURES": regression.get("new_regression_failures", 1),
        "MOVEITPY_SHUTDOWN_CLEAN": replay.get("MOVEITPY_SHUTDOWN_CLEAN", "NO"),
        "MOVEITPY_SHUTDOWN_ARTIFACT_IMPACT": "NATIVE_ARTIFACTS_COMPLETE_AND_REPLAY_DETERMINISTIC; LIFECYCLE_GATE_REMAINS_BLOCKED"
        if replay.get("MOVEITPY_SHUTDOWN_CLEAN") == "NO" and replay.get("DETERMINISTIC_REPLAY") == "3/3"
        else "NONE" if replay.get("MOVEITPY_SHUTDOWN_CLEAN") == "YES" else "ARTIFACT_COMPLETENESS_NOT_ESTABLISHED",
        "CCD": "NOT_AVAILABLE",
        "CLEARANCE": "NOT_AVAILABLE",
        "collision_method": COLLISION_METHOD,
        "NEW_FJT_GOALS_SENT": 0,
        "ROBOT_MOTION_STARTED": "NO",
        "FORMAL_LEDGER_MUTATED": "NO",
        "READY_FOR_STAGE_3_H8": "YES" if passed else "NO",
    }
    gate = {
        "schema_version": "stage3-h7-3-gate-report-v1",
        "STAGE_3_H7_3": terminal["STAGE_3_H7_3"],
        "FIRST_BLOCKER": first_blocker,
        "mandatory_gates": mandatory,
        "READY_FOR_STAGE_3_H8": terminal["READY_FOR_STAGE_3_H8"],
    }
    h72.dump_json(output / "stage3_h7_3_gate_report.json", gate)
    h72.dump_json(output / "stage3_h7_3_terminal_certificate.json", terminal)
    lines = [
        "# Stage 3 H7.3 — Target-55 Controlled-Stop TOTG/Ruckig Recertification",
        "",
        "## Required answers",
        "",
        *[f"- `{key}: {value}`" for key, value in terminal.items() if key != "schema_version"],
        "",
        "## Certification notes",
        "",
        "- H6.4 remains a nine-segment process artifact; H7.3 adds only a derived 3A|3B zero-velocity timing boundary.",
        "- Native MoveIt TOTG and Ruckig run independently per primitive; no fitting or smoothing crosses 3A→3B.",
        "- Collision certification uses native PlanningScene/FCL with `adaptive_discrete_interpolation`; CCD and clearance are `NOT_AVAILABLE`.",
        "- The 8 rad/s³ per-joint jerk bounds are project-authoritative Stage 3 research limits, not FAIRINO manufacturer jerk specifications.",
        "- No FollowJointTrajectory goal, robot motion, formal-ledger mutation, or H8 execution occurred.",
        "",
    ]
    h72.dump_text(output / "FINAL_REPORT.md", "\n".join(lines))
    return terminal


def orchestrate(output: Path) -> int:
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty H7.3 output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    before = source_snapshot()
    h72.dump_json(
        output / "stage3_h7_3_input_freeze_manifest.json",
        {
            "schema_version": "stage3-h7-3-input-freeze-manifest-v1",
            "captured_before_native_computation": True,
            "sources": before,
            "authorized_remediation": "derived zero-velocity motion-primitive boundary between segment 3 local 193 and 194",
            "forbidden_operations": [
                "H6.4 artifact mutation",
                "H7/H7.1/H7.2 artifact mutation",
                "joint value change",
                "waypoint deletion",
                "IK",
                "DP branch change",
                "target reorder",
                "shortcut",
                "process tolerance change",
                "collision gate relaxation",
                "min_angle_change reduction as remediation",
                "FollowJointTrajectory",
                "robot motion",
                "formal ledger mutation",
                "H8",
            ],
        },
    )
    h6_audit = h72.audit_h6_4()
    h7_2_audit = audit_h7_2()
    cusp = target55_cusp_audit()
    h72.dump_json(output / "stage3_h7_3_target55_cusp_audit.json", cusp)
    order = authoritative_primitives()
    h72.dump_json(output / "stage3_h7_3_authoritative_primitive_order.json", order)
    filtered = pre_totg_filtered_path_audit(order)
    h72.dump_json(output / "stage3_h7_3_pre_totg_filtered_path_audit.json", filtered)
    params = totg_parameters()
    h72.dump_json(output / "stage3_h7_3_totg_parameters.json", params)
    jerk = h72.jerk_limits_audit()
    jerk = {**jerk, "schema_version": "stage3-h7-3-ruckig-limits-audit-v1"}
    h72.dump_json(output / "stage3_h7_3_ruckig_limits_audit.json", jerk)

    preconditions = all(
        [
            h6_audit.get("status") == "PASSED",
            h7_2_audit.get("status") == "PASSED",
            cusp.get("status") == "PASSED",
            order.get("status") == "PASSED",
            filtered.get("status") == "PASSED",
            params.get("status") == "FROZEN_FROM_AUTHORITATIVE_H7_2",
            jerk.get("ruckig_authorized") is True,
        ]
    )
    runs: list[dict[str, Any]] = []
    build: dict[str, Any] = {"status": "NOT_RUN"}
    if preconditions:
        build = build_worker(output)
        if build["status"] == "PASSED":
            for replay_index in (1, 2, 3):
                runs.append(run_replay(output, replay_index, True))
    if not runs:
        runs = [
            {
                "status": "BLOCKED",
                "first_blocker": "native_worker_not_run",
                "primitive_results": [],
                "authoritative_primitive_order": PRIMITIVE_ORDER,
                "totg_primitives_passed": 0,
                "ruckig_primitives_passed": 0,
                "moveitpy_shutdown_clean": False,
            }
        ]
        empty = output / "fresh_process_replays/replay_1"
        empty.mkdir(parents=True, exist_ok=True)
        h72.dump_text(empty / "totg_trajectories.jsonl", "")
        h72.dump_text(empty / "ruckig_trajectories.jsonl", "")
    copy_primary_outputs(output)
    primary = runs[0]
    _totg_results, post_totg, _ruckig_results, post_ruckig = aggregate_native(output, primary, jerk)
    replay = replay_report(output, runs)
    regression = run_regression(output)
    immutable = immutable_compare(before)
    h72.dump_json(output / "stage3_h7_3_immutable_after_check.json", immutable)
    terminal = terminal_and_report(
        output,
        h6_audit,
        h7_2_audit,
        cusp,
        order,
        filtered,
        primary,
        post_totg,
        post_ruckig,
        replay,
        regression,
        immutable,
    )
    artifacts = {
        path.name: h72.file_record(path, "H7.3 additive output")
        for path in sorted(output.iterdir())
        if path.is_file()
    }
    missing = [name for name in REQUIRED_OUTPUTS if not (output / name).is_file()]
    h72.dump_json(
        output / "stage3_h7_3_artifact_manifest.json",
        {
            "schema_version": "stage3-h7-3-artifact-manifest-v1",
            "files": artifacts,
            "missing_required_outputs": missing,
            "additive": True,
            "h6_4_or_h7_2_artifacts_written": False,
        },
    )
    print(json.dumps({"output": str(output.resolve()), "terminal": terminal, "build": build}, ensure_ascii=False))
    return 0 if terminal["STAGE_3_H7_3"] == "PASSED" else 2


def finalize_existing(output: Path) -> int:
    """Re-aggregate completed native replays without rerunning MoveIt."""
    freeze_path = output / "stage3_h7_3_input_freeze_manifest.json"
    if not freeze_path.is_file():
        raise RuntimeError(f"missing H7.3 input freeze manifest: {freeze_path}")
    runs = [
        h72.load_json(output / "fresh_process_replays" / f"replay_{index}" / "native_result.json")
        for index in (1, 2, 3)
    ]
    copy_primary_outputs(output)
    jerk = h72.load_json(output / "stage3_h7_3_ruckig_limits_audit.json")
    _totg_results, post_totg, _ruckig_results, post_ruckig = aggregate_native(output, runs[0], jerk)
    replay = replay_report(output, runs)
    regression = run_regression(output)
    before = h72.load_json(freeze_path)["sources"]
    immutable = immutable_compare(before)
    h72.dump_json(output / "stage3_h7_3_immutable_after_check.json", immutable)
    terminal = terminal_and_report(
        output,
        h72.audit_h6_4(),
        audit_h7_2(),
        h72.load_json(output / "stage3_h7_3_target55_cusp_audit.json"),
        h72.load_json(output / "stage3_h7_3_authoritative_primitive_order.json"),
        h72.load_json(output / "stage3_h7_3_pre_totg_filtered_path_audit.json"),
        runs[0],
        post_totg,
        post_ruckig,
        replay,
        regression,
        immutable,
    )
    artifacts = {
        path.name: h72.file_record(path, "H7.3 additive output")
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "stage3_h7_3_artifact_manifest.json"
    }
    missing = [name for name in REQUIRED_OUTPUTS if not (output / name).is_file()]
    h72.dump_json(
        output / "stage3_h7_3_artifact_manifest.json",
        {
            "schema_version": "stage3-h7-3-artifact-manifest-v1",
            "files": artifacts,
            "missing_required_outputs": missing,
            "additive": True,
            "h6_4_or_h7_2_artifacts_written": False,
            "finalized_from_existing_three_fresh_process_replays": True,
        },
    )
    print(json.dumps({"output": str(output.resolve()), "terminal": terminal, "finalized_existing": True}, ensure_ascii=False))
    return 0 if terminal["STAGE_3_H7_3"] == "PASSED" else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--finalize-existing", action="store_true")
    args = parser.parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or ROOT / "outputs" / f"stage3_h7_3_target55_controlled_stop_{timestamp}"
    if args.finalize_existing:
        return finalize_existing(output.resolve())
    return orchestrate(output.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
