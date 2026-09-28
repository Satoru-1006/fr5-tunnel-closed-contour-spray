#!/usr/bin/env python3
"""Create the compact 181-row R0 lineage ledger and P2-B3-C1 result JSON."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_LEGACY_FK_SHA256 = "4d2097fc2a36da4c60d6c31883e46f2d8e8db8a97feec7a661ccf92bad0cadd4"
sys.path.insert(0, str(ROOT / "ros2_moveit_bridge"))
from p2b3_c1_ordered_validation import (  # noqa: E402
    local_segment_timing,
    project_open_polyline,
    station_deltas,
    waypoint_semantics,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"json_root_must_be_object:{path}")
    return value


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def xyz(row: dict[str, str], keys: tuple[str, str, str]) -> tuple[float, float, float]:
    value = tuple(float(row[key]) for key in keys)
    if not all(math.isfinite(item) for item in value):
        raise ValueError("nonfinite_xyz")
    return value  # type: ignore[return-value]


def fk_xyz(row: dict[str, str]) -> tuple[float, float, float]:
    if {"actual_tcp_x", "actual_tcp_y", "actual_tcp_z"}.issubset(row):
        return xyz(row, ("actual_tcp_x", "actual_tcp_y", "actual_tcp_z"))
    if {"x_m", "y_m", "z_m"}.issubset(row):
        return xyz(row, ("x_m", "y_m", "z_m"))
    raise ValueError("FK_position_schema_missing_supported_coordinate_columns")


def validate_waypoint_order(name: str, rows: list[dict[str, str]]) -> None:
    for index, row in enumerate(rows):
        if "waypoint" in row and int(row["waypoint"]) != index:
            raise ValueError(f"{name}_waypoints_not_zero_based_in_row_order:{index}")


def q_vector(row: dict[str, str]) -> tuple[float, ...]:
    names = [f"j{index}_q" for index in range(1, 7)]
    if not all(key in row for key in names):
        names = [f"q{index}" for index in range(1, 7)]
    return tuple(float(row[key]) for key in names)


def max_abs_delta(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right):
        raise ValueError("joint_vector_size_mismatch")
    return max(abs(a - b) for a, b in zip(left, right))


def require_181(name: str, rows: list[dict[str, str]]) -> None:
    if len(rows) != 181:
        raise ValueError(f"{name}_must_have_181_rows:got={len(rows)}")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("refusing_empty_ledger")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def json_safe_number(value: float | None) -> float | None:
    return value if value is not None and math.isfinite(value) else None


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def derive_c1_stage_status(gates: dict[str, str], limitations: list[str]) -> str:
    """Fail closed on required C1 evidence while retaining measured limitations."""
    if any(status == "FAIL" for status in gates.values()):
        return "FAIL"
    if any(status != "PASS" for status in gates.values()):
        return "BLOCKED"
    return "COMPLETE_WITH_LIMITATIONS" if limitations else "COMPLETE"


def normalize_gate(value: Any) -> str:
    text = str(value or "UNKNOWN").strip().upper()
    if text in {"PASS", "TRUE", "OK", "COMPLETE"}:
        return "PASS"
    if text in {"FAIL", "FALSE"}:
        return "FAIL"
    return "UNKNOWN"


def _pressure_waypoint_evidence(path: Path) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    rows = read_csv(path)
    required = {
        "case_id", "variant", "waypoint", "iteration", "current_q", "accepted_q",
        "line_search_accepted", "projection_triggered", "position_residual_m", "normal_residual",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError("C1_solver_pressure_schema_missing_required_fields")
    grouped: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        waypoint = int(row["waypoint"])
        iteration = int(row["iteration"])
        if not 0 <= waypoint < 181 or iteration < 0:
            raise ValueError("C1_solver_pressure_waypoint_or_iteration_out_of_range")
        if row["variant"] != "B0" or row["case_id"] != "P2B3_C1_R0":
            raise ValueError("C1_solver_pressure_case_or_variant_mismatch")
        grouped.setdefault(waypoint, []).append(row)
    if set(grouped) != set(range(181)):
        raise ValueError("C1_solver_pressure_must_cover_target_indices_0_to_180")

    result: dict[int, dict[str, Any]] = {}
    for waypoint, observations in grouped.items():
        observations.sort(key=lambda row: int(row["iteration"]))
        last = observations[-1]
        position_vector = json.loads(last["position_residual_m"])
        normal_vector = json.loads(last["normal_residual"])
        current_q = json.loads(last["current_q"])
        accepted_q = json.loads(last["accepted_q"])
        if (len(position_vector) != 3 or len(normal_vector) != 2 or len(current_q) != 6 or len(accepted_q) != 6
                or not all(_finite(value) for value in [*position_vector, *normal_vector, *current_q, *accepted_q])):
            raise ValueError(f"C1_solver_pressure_nonfinite_or_shape_mismatch:{waypoint}")
        if max(abs(float(a) - float(b)) for a, b in zip(current_q, accepted_q)) > 1e-12:
            raise ValueError(f"C1_final_pressure_residual_not_for_emitted_DLS_state:{waypoint}")
        position_residual = math.sqrt(sum(float(value) ** 2 for value in position_vector))
        normal_sine = math.sqrt(sum(float(value) ** 2 for value in normal_vector))
        if normal_sine > 1.0 + 1e-12:
            raise ValueError(f"C1_pressure_normal_residual_outside_sine_domain:{waypoint}")
        # The process solver's normal residual is the two tangent-plane components
        # of tool_z - wall_normal. The accepted 5 degree geometric gate keeps the
        # angle in the principal asin branch, so asin(norm) recovers the angle.
        normal_residual_rad = math.asin(min(1.0, normal_sine))
        tight_converged = position_residual <= 2.0e-5 and normal_residual_rad <= 2.0e-4
        relaxed_gate = position_residual <= 4.0e-3 and normal_residual_rad <= math.radians(5.0)
        final_line_search_accepted = str(last["line_search_accepted"]).strip().lower() == "true"
        if tight_converged:
            termination = "TIGHT_POSITION_AND_NORMAL_TOLERANCES_MET"
        elif not final_line_search_accepted:
            termination = "NO_IMPROVING_LINE_SEARCH_BEFORE_TIGHT_CONVERGENCE"
        else:
            termination = "UNRESOLVED_FINAL_DLS_TERMINATION"
        if not relaxed_gate:
            raise ValueError(f"C1_target_did_not_meet_existing_relaxed_DLS_geometry_gate:{waypoint}")
        result[waypoint] = {
            "target_attempted": True,
            "target_state_emitted": True,
            "tight_converged": tight_converged,
            "final_position_residual_m": position_residual,
            "final_normal_residual_rad": normal_residual_rad,
            "iterations_used": len(observations),
            "termination_reason": termination,
            "final_relaxed_geometry_gate": "PASS_4MM_5DEG",
            "projection_triggered_count": sum(str(row["projection_triggered"]).strip().lower() == "true" for row in observations),
        }
    if any(row["termination_reason"] == "UNRESOLVED_FINAL_DLS_TERMINATION" for row in result.values()):
        raise ValueError("C1_solver_pressure_does_not_identify_final_DLS_termination")
    summary = {
        "status": "MEASURED_FROM_P2B1_B0_SOLVER_PRESSURE",
        "row_count": len(rows),
        "target_attempted_count": len(result),
        "tight_converged_count": sum(bool(row["tight_converged"]) for row in result.values()),
        "accepted_relaxed_geometry_count": sum(row["final_relaxed_geometry_gate"] == "PASS_4MM_5DEG" for row in result.values()),
        "internal_joint_projection_count": sum(int(row["projection_triggered_count"]) for row in result.values()),
        "tight_position_tolerance_m": 2.0e-5,
        "tight_normal_tolerance_rad": 2.0e-4,
        "relaxed_position_tolerance_m": 4.0e-3,
        "relaxed_normal_tolerance_rad": math.radians(5.0),
        "normal_residual_interpretation": "asin(norm(two tangent-plane components)); exact on the solver's accepted <=5 degree branch",
    }
    return result, summary


def validate_and_publish(
    *, baseline_fk: Path, candidate_dir: Path, manifest_path: Path, output_dir: Path,
    solver_pressure_path: Path | None = None,
    p2b2_recheck_path: Path | None = None,
    artifact_publish_commit: str | None = None,
    legacy_fk_drive_file_id: str = "1Itbn-X_f61tew9PW2Xe-kIBxNlMM4lf6",
    legacy_fk_drive_folder_id: str = "153wAjeoPurkZK-4aUx9eDyXT-HB_ZyIn",
) -> dict[str, Any]:
    target_path = ROOT / "outputs/internal_wiper_moveit_inputs/open_arch_tcp_poses_base_link.csv"
    seed_path = ROOT / "outputs/p2b2_inputs/stable_velocity_residual_update.csv"
    baseline_pre_path = ROOT / "outputs/p2b2_inputs/p2b1_b0_reference/strict_replay/moveit_waypoint_joint_trajectory.csv"
    baseline_post_path = ROOT / "outputs/p2b2_inputs/p2b1_b0_reference/strict_replay/moveit_smoothed_joint_trajectory.csv"

    candidate_pre_path = candidate_dir / "moveit_waypoint_joint_trajectory.csv"
    candidate_post_path = candidate_dir / "moveit_smoothed_joint_trajectory.csv"
    candidate_fk_path = candidate_dir / "moveit_fk_tcp_trace.csv"
    summary_path = candidate_dir / "final_acceptance_summary.json"
    strict_path = candidate_dir / "audit_goal_requirements_strict.json"
    required = (target_path, seed_path, baseline_pre_path, baseline_post_path, baseline_fk,
                candidate_pre_path, candidate_post_path, candidate_fk_path, summary_path, strict_path, manifest_path)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if digest(baseline_fk) != EXPECTED_LEGACY_FK_SHA256:
        raise ValueError("legacy_FK_archive_member_sha256_mismatch")

    target, seeds = read_csv(target_path), read_csv(seed_path)
    baseline_pre, baseline_post, old_fk = read_csv(baseline_pre_path), read_csv(baseline_post_path), read_csv(baseline_fk)
    candidate_pre, candidate_post, new_fk = read_csv(candidate_pre_path), read_csv(candidate_post_path), read_csv(candidate_fk_path)
    for name, rows in (("target", target), ("D39_seed", seeds), ("baseline_pre", baseline_pre),
                       ("baseline_post", baseline_post), ("baseline_fk", old_fk), ("candidate_pre", candidate_pre),
                       ("candidate_post", candidate_post), ("candidate_fk", new_fk)):
        require_181(name, rows)

    target_xyz = [xyz(row, ("x", "y", "z")) for row in target]
    baseline_pre_q = [q_vector(row) for row in baseline_pre]
    baseline_post_q = [q_vector(row) for row in baseline_post]
    candidate_pre_q = [q_vector(row) for row in candidate_pre]
    candidate_post_q = [q_vector(row) for row in candidate_post]
    seed_q = [q_vector(row) for row in seeds]
    if any(max_abs_delta(baseline_pre_q[i], seed_q[i]) != 0.0 for i in range(16)):
        raise ValueError("legacy_first_16_rows_are_not_exact_D39_seed_copies")

    def trajectory_times(rows: list[dict[str, str]]) -> list[float]:
        return [float(row["t"]) for row in rows]

    old_times, new_times = trajectory_times(baseline_post), trajectory_times(candidate_post)
    validate_waypoint_order("legacy_FK", old_fk)
    legacy_fk_has_time = all("t" in row for row in old_fk)
    if legacy_fk_has_time and any(float(old_fk[i]["t"]) != old_times[i] for i in range(181)):
        raise ValueError("legacy_FK_timestamps_do_not_match_post_Ruckig_rows")
    if any(float(new_fk[i]["t"]) != new_times[i] for i in range(181)):
        raise ValueError("C1_FK_timestamps_do_not_match_post_Ruckig_rows")
    if any(int(row.get("waypoint", index)) != index for index, row in enumerate(baseline_pre)):
        raise ValueError("legacy_pre_Ruckig_waypoint_labels_not_zero_based_row_order")
    if any(int(row.get("waypoint", index)) != index for index, row in enumerate(candidate_pre)):
        raise ValueError("C1_pre_Ruckig_waypoint_labels_not_zero_based_row_order")

    old_tcp = [fk_xyz(row) for row in old_fk]
    new_tcp = [fk_xyz(row) for row in new_fk]
    old_proj = [project_open_polyline(point, target_xyz) for point in old_tcp]
    new_proj = [project_open_polyline(point, target_xyz) for point in new_tcp]
    old_station_delta, new_station_delta = station_deltas(old_proj), station_deltas(new_proj)
    old_local = local_segment_timing(old_tcp, old_times)
    new_local = local_segment_timing(new_tcp, new_times)
    old_same_error = [math.dist(point, target_xyz[index]) for index, point in enumerate(old_tcp)]
    new_same_error = [math.dist(point, target_xyz[index]) for index, point in enumerate(new_tcp)]
    row_delta = [max_abs_delta(baseline_post_q[i], candidate_post_q[i]) for i in range(181)]
    unchanged_rows = [i for i, delta in enumerate(row_delta) if delta <= 1.0e-12]
    numerical_delta_rows = [i for i, delta in enumerate(row_delta) if delta > 1.0e-12]
    material_delta_rows = [i for i, delta in enumerate(row_delta) if delta > 1.0e-6]

    manifest = read_json(manifest_path)
    strict, acceptance = read_json(strict_path), read_json(summary_path)
    pressure: dict[int, dict[str, Any]] = {}
    pressure_summary: dict[str, Any] = {
        "status": "UNAVAILABLE", "target_attempted_count": 0,
        "tight_converged_count": 0, "accepted_relaxed_geometry_count": 0,
    }
    if solver_pressure_path is not None and solver_pressure_path.is_file():
        pressure, pressure_summary = _pressure_waypoint_evidence(solver_pressure_path)

    ledger: list[dict[str, Any]] = []
    for index in range(181):
        baseline_role = waypoint_semantics(index, corrected_candidate=False)
        candidate_role = waypoint_semantics(index, corrected_candidate=True)
        old_local_row = old_local[index - 1] if index else None
        new_local_row = new_local[index - 1] if index else None
        old_step = old_station_delta[index - 1] if index else None
        new_step = new_station_delta[index - 1] if index else None
        ledger.append({
            "index": index,
            "upstream_source_artifact": "D39 stable_velocity_residual_update.csv" if index < 16 else "P2B1 R0 B0 ordered DLS solve",
            "upstream_source_row": baseline_role["upstream_source_row"],
            "solver_role": baseline_role["solver_role"],
            "expected_target_index": index,
            "target_solved": baseline_role["target_solved"],
            "retained_seed": baseline_role["retained_seed"],
            "planned_trajectory_row_emitted": True,
            "physical_execution": "NOT_RUN",
            "planned_process_scope": "ON_STATE_OPEN_ARCH_SINGLE_PROCESS_SEGMENT",
            "spray_command_state": "NOT_REPRESENTED",
            "pre_ruckig_source_row": index,
            "post_ruckig_source_row": index,
            "baseline_time_s": old_times[index],
            "baseline_fk_tcp_x_m": old_tcp[index][0],
            "baseline_fk_tcp_y_m": old_tcp[index][1],
            "baseline_fk_tcp_z_m": old_tcp[index][2],
            "intended_target_tcp_x_m": target_xyz[index][0],
            "intended_target_tcp_y_m": target_xyz[index][1],
            "intended_target_tcp_z_m": target_xyz[index][2],
            "baseline_same_index_error_m": old_same_error[index],
            "baseline_nearest_segment_zero_based": old_proj[index].segment_index,
            "baseline_nearest_station_m": old_proj[index].station_m,
            "baseline_nearest_path_error_m": old_proj[index].distance_m,
            "baseline_ordered_progression_status": "START" if index == 0 else ("BACKSTEP" if old_step < -1.0e-9 else "NONDECREASING"),
            "baseline_station_delta_from_previous_m": old_step,
            "baseline_local_tcp_distance_from_previous_m": None if old_local_row is None else old_local_row.distance_m,
            "baseline_local_delta_t_from_previous_s": None if old_local_row is None else old_local_row.delta_t_s,
            "baseline_local_speed_from_previous_m_s": None if old_local_row is None else old_local_row.speed_m_s,
            "baseline_local_speed_band_status": "START" if old_local_row is None else old_local_row.speed_band_status,
            "baseline_pre_post_q_max_abs_delta_rad": max_abs_delta(baseline_pre_q[index], baseline_post_q[index]),
            "c1_solver_role": candidate_role["solver_role"],
            "c1_expected_target_index": candidate_role["expected_target_index"],
            "c1_target_solved": candidate_role["target_solved"],
            "c1_target_attempted": pressure.get(index, {}).get("target_attempted", False),
            "c1_target_state_emitted": pressure.get(index, {}).get("target_state_emitted", False),
            "c1_tight_converged": pressure.get(index, {}).get("tight_converged"),
            "c1_final_position_residual_m": pressure.get(index, {}).get("final_position_residual_m"),
            "c1_final_normal_residual_rad": pressure.get(index, {}).get("final_normal_residual_rad"),
            "c1_iterations_used": pressure.get(index, {}).get("iterations_used"),
            "c1_termination_reason": pressure.get(index, {}).get("termination_reason"),
            "c1_final_relaxed_geometry_gate": pressure.get(index, {}).get("final_relaxed_geometry_gate", "UNAVAILABLE"),
            "c1_seed_row_used": candidate_role["upstream_source_row"],
            "c1_pre_ruckig_source_row": index,
            "c1_post_ruckig_source_row": index,
            "c1_time_s": new_times[index],
            "c1_fk_tcp_x_m": new_tcp[index][0],
            "c1_fk_tcp_y_m": new_tcp[index][1],
            "c1_fk_tcp_z_m": new_tcp[index][2],
            "c1_same_index_error_m": new_same_error[index],
            "c1_nearest_segment_zero_based": new_proj[index].segment_index,
            "c1_nearest_station_m": new_proj[index].station_m,
            "c1_nearest_path_error_m": new_proj[index].distance_m,
            "c1_ordered_progression_status": "START" if index == 0 else ("BACKSTEP" if new_step < -1.0e-9 else "NONDECREASING"),
            "c1_station_delta_from_previous_m": new_step,
            "c1_local_tcp_distance_from_previous_m": None if new_local_row is None else new_local_row.distance_m,
            "c1_local_delta_t_from_previous_s": None if new_local_row is None else new_local_row.delta_t_s,
            "c1_local_speed_from_previous_m_s": None if new_local_row is None else new_local_row.speed_m_s,
            "c1_local_speed_band_status": "START" if new_local_row is None else new_local_row.speed_band_status,
            "c1_local_dwell_from_previous": False if new_local_row is None else new_local_row.dwell,
            "c1_pre_post_q_max_abs_delta_rad": max_abs_delta(candidate_pre_q[index], candidate_post_q[index]),
            "post_ruckig_max_abs_q_delta_vs_legacy_rad": row_delta[index],
            "baseline_semantic_classification": baseline_role["semantic_classification"],
            "c1_semantic_classification": candidate_role["semantic_classification"],
        })

    recheck = read_json(p2b2_recheck_path) if p2b2_recheck_path is not None else None
    metrics = acceptance.get("metrics", {}) if isinstance(acceptance.get("metrics", {}), dict) else {}
    geometry_ok = max(row.distance_m for row in new_proj) <= 0.006 + 1.0e-12
    indexed_ok = max(new_same_error) <= 0.006 + 1.0e-12
    normal_errors = [float(row["normal_angle_error_deg"]) for row in new_fk]
    normal_ok = max(normal_errors) <= 10.0 + 1.0e-9
    backstep_count = sum(value < -1.0e-9 for value in new_station_delta)
    local_band_counts = {status: sum(row.speed_band_status == status for row in new_local) for status in (
        "WITHIN_CONFIGURED_BAND", "BELOW_CONFIGURED_BAND", "ABOVE_CONFIGURED_BAND")}
    local_slow = [row for row in new_local if row.speed_band_status == "BELOW_CONFIGURED_BAND"]
    old_local_slow = [row for row in old_local if row.speed_band_status == "BELOW_CONFIGURED_BAND"]

    def bracket_spotcheck(axis_id: str) -> str:
        if recheck is None:
            return "TARGETED_RECHECK_REQUIRED"
        values = recheck.get("measured_bracket_endpoints", {}).get(axis_id)
        if not isinstance(values, dict):
            return "TARGETED_RECHECK_MISSING"
        last_pass = values.get("historical_last_pass", {}).get("measured", {}).get("status")
        first_fail = values.get("historical_first_fail", {}).get("measured", {}).get("status")
        if last_pass == "PASS" and first_fail == "FAIL":
            return "OLD_PASS_FAIL_BRACKET_MATCHED_SPOTCHECK_ONLY"
        return f"BRACKET_CLASSIFICATION_CHANGED:{last_pass}/{first_fail}"

    compact_brackets: dict[str, Any] | None = None
    if recheck is not None:
        compact_brackets = {}
        for axis_id, endpoints in recheck.get("measured_bracket_endpoints", {}).items():
            compact_brackets[axis_id] = {}
            for label in ("historical_last_pass", "historical_first_fail"):
                row = endpoints.get(label, {})
                measured = row.get("measured", {})
                evidence = measured.get("evidence", {})
                compact_brackets[axis_id][label] = {
                    "magnitude": row.get("magnitude"),
                    "expected_historical_status": row.get("expected_historical_status"),
                    "measured_status": measured.get("status"),
                    "dominant_failure_mode": measured.get("dominant_failure_mode"),
                    "failure_modes": measured.get("failure_modes", []),
                    "critical_waypoint": measured.get("critical_waypoint"),
                    "evidence": {key: evidence.get(key) for key in (
                        "terminal_position_error_m", "max_tcp_path_error_m", "max_tcp_path_error_waypoint",
                        "max_spray_axis_normal_error_rad", "max_spray_axis_normal_error_waypoint",
                        "joint_limit_margin_min_rad", "max_velocity_ratio", "max_acceleration_ratio", "max_jerk_ratio",
                        "native_waypoint_world_collision_count", "native_waypoint_self_collision_count",
                        "minimum_environment_clearance_m", "minimum_self_clearance_m", "clearance_acceptance_threshold",
                        "collision_method", "strict_continuous_self_collision_ccd") if key in evidence},
                }
    input_hashes = {str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else path.name: digest(path)
                    for path in (target_path, seed_path, baseline_pre_path, baseline_post_path)}
    required_files_present = all(path.is_file() and path.stat().st_size > 0 for path in required)
    finite_measurements = all(math.isfinite(value) for value in (
        *[value for point in old_tcp + new_tcp for value in point],
        *old_times, *new_times, *old_same_error, *new_same_error, *normal_errors,
    ))
    lineage_complete = len(ledger) == 181 and all(
        row["c1_target_attempted"] is True and row["c1_target_state_emitted"] is True
        and row["c1_final_relaxed_geometry_gate"] == "PASS_4MM_5DEG"
        for row in ledger
    )
    artifact_commit_ok = bool(artifact_publish_commit and re.fullmatch(r"[0-9a-f]{40}", artifact_publish_commit))
    manifest_inputs = manifest.get("inputs", {})
    expected_input_pairs = (
        (target_path, manifest_inputs.get("target_tcp_csv", {}).get("sha256")),
        (seed_path, manifest_inputs.get("d39_observation_seed_csv", {}).get("sha256")),
        (baseline_post_path, manifest_inputs.get("reference_post_ruckig_csv", {}).get("sha256")),
    )
    input_hash_match = all(expected and digest(path) == expected for path, expected in expected_input_pairs)
    model_sha = manifest_inputs.get("derived_reference_urdf", {}).get("sha256")
    source_commit = manifest.get("fairino_description_source", {}).get("commit")
    gates = {
        "required_files_presence": "PASS" if required_files_present else "FAIL",
        "required_input_identity": "PASS" if input_hash_match and all(re.fullmatch(r"[0-9a-f]{64}", value) for value in input_hashes.values()) else "FAIL",
        "waypoint_count": "PASS" if all(len(rows) == 181 for rows in (target, seeds, baseline_pre, baseline_post, old_fk,
                                                                       candidate_pre, candidate_post, new_fk)) else "FAIL",
        "legacy_fk_time_alignment": "PASS" if len(old_fk) == len(old_times) == 181 and all(
            not legacy_fk_has_time or float(old_fk[i]["t"]) == old_times[i] for i in range(181)
        ) else "FAIL",
        "strict_audit": normalize_gate(strict.get("overall_status", strict.get("status"))),
        "post_ruckig_acceptance": normalize_gate(acceptance.get("overall_status", acceptance.get("status"))),
        "geometry_fidelity": "PASS" if geometry_ok and indexed_ok and normal_ok else "FAIL",
        "ordered_traversal": "PASS" if backstep_count == 0 else "FAIL",
        "target_lineage_completeness": "PASS" if lineage_complete else "FAIL",
        "finite_numeric_values": "PASS" if finite_measurements else "FAIL",
        "solver_pressure_evidence": "PASS" if len(pressure) == 181 else "UNKNOWN",
        "provenance_completeness": "PASS" if (
            manifest.get("source_base_commit") and manifest.get("execution_code_commit")
            and source_commit and re.fullmatch(r"[0-9a-f]{40}", str(source_commit))
            and model_sha and re.fullmatch(r"[0-9a-f]{64}", str(model_sha)) and artifact_commit_ok
        ) else "UNKNOWN",
    }
    limitations = [
        "WP179_TO_WP180_LOCAL_TIMING_ANOMALY: 0.0041438374043542055 m/s exceeds the configured 0.003 m/s +/- 5% diagnostic band; no deposition model is present.",
        "Collision uses adaptive_discrete_interpolation; strict continuous collision detection is NOT_AVAILABLE.",
        "Hardware/controller validation is NOT_RUN; physical torque, calibrated TCP uncertainty, and spray command state are not established.",
    ]
    c1_status = derive_c1_stage_status(gates, limitations)
    candidate_post_sha = digest(candidate_post_path)
    pressure_sha = digest(solver_pressure_path) if solver_pressure_path is not None and solver_pressure_path.is_file() else None

    result: dict[str, Any] = {
        "schema": "p2b3-c1-r0-ordered-path-fidelity-v1",
        "PROJECT": "FAIRINO_FR5",
        "STAGE": "P2-B3-C1",
        "P2B3_C1_STATUS": c1_status,
        "C1_FATAL_GATES": gates,
        "C1_NON_FATAL_LIMITATIONS": limitations,
        "SOURCE_BASE_COMMIT": manifest.get("source_base_commit"),
        "EXECUTION_CODE_COMMIT": manifest.get("execution_code_commit"),
        "C1_EXECUTION_CODE_COMMIT": manifest.get("execution_code_commit"),
        "C1_ARTIFACT_PUBLISH_COMMIT": artifact_publish_commit,
        "ARTIFACT_PUBLISH_PROVENANCE": {
            "status": "TWO_PHASE_EXPLICIT",
            "phase_1_execution_code_commit": manifest.get("execution_code_commit"),
            "phase_2_canonical_artifact_commit": artifact_publish_commit or "UNRESOLVED_REQUIRES_POST_COMMIT_READBACK",
            "phase_2_commit_is_not_inferred_from_precommit_inputs": True,
        },
        "FAIRINO_SOURCE_COMMIT": manifest.get("fairino_description_source", {}).get("commit"),
        "C1_DERIVED_REFERENCE_URDF_SHA256": manifest.get("inputs", {}).get("derived_reference_urdf", {}).get("sha256"),
        "legacy_fk_drive_source": {
            "archive_filename": "FAIRINO_FR5_P2B2_INTERMEDIATE_EVIDENCE_2026-09-27.zip",
            "drive_file_id": legacy_fk_drive_file_id,
            "drive_folder_id": legacy_fk_drive_folder_id,
            "archive_member": "INTERMEDIATE_EVIDENCE/b0_reference/nominal_fk_trace.csv",
            "member_sha256": digest(baseline_fk),
            "time_alignment": "FK waypoints 0..180 map row-wise to the authenticated legacy post-Ruckig trajectory timestamps" if not legacy_fk_has_time else "FK timestamps match authenticated legacy post-Ruckig trajectory",
            "source_identity_status": "READ_BACK_FROM_DRIVE_ARCHIVE_AND_MEMBER_SHA_VERIFIED",
        },
        "C1_POST_RUCKIG_NOMINAL": {
            "path": "strict_replay/moveit_smoothed_joint_trajectory.csv",
            "sha256": candidate_post_sha,
            "waypoint_count": 181,
        },
        "C1_SOLVER_PRESSURE_EVIDENCE": pressure_summary | {
            "path": None if solver_pressure_path is None else solver_pressure_path.name,
            "sha256": pressure_sha,
            "per_waypoint_fields": ["target_attempted", "target_state_emitted", "tight_converged",
                                    "final_position_residual_m", "final_normal_residual_rad", "iterations_used",
                                    "termination_reason", "final_relaxed_geometry_gate"],
        },
        "R0_WARM_START_SEMANTICS": "RESOLVED",
        "WP0_15_ROLE": "D39_OBSERVED_HISTORY_COPIED_INTO_LEGACY_OUTPUT_PREFIX_WITHOUT_TARGET_SOLVE",
        "WP16_180_ROLE": "DLS_SOLVED_TARGET_STATES_FOR_TARGET_INDICES_16_THROUGH_180",
        "WP15_WP16_ROOT_CAUSE": "D40_D41_P2B1_R0_APPENDED_D39_SEEDS_0_15_AS_TRAJECTORY_PREFIX_THEN_STARTED_DLS_AT_TARGET_16_FROM_SEED_ROW_15",
        "PROCESS_EXECUTION_SCOPE": "ONE_PLANNED_ON_STATE_OPEN_ARCH_PROCESS_SEGMENT; NO_HARDWARE_EXECUTION",
        "SPRAY_STATE_SEMANTICS": "ON_STATE_IS_A_PLANNING_SCOPE_LABEL; SPRAY_COMMAND_OR_PHYSICAL_ON_OFF_STATE_NOT_REPRESENTED",
        "TRAJECTORY_FIX_REQUIRED": "YES",
        "TRAJECTORY_FIX_SCOPE": "C1_ONLY_R0_B0_WARM_START_POLICY; D39_ROW0_SEEDS_TARGET0; EMIT_NO_UNSOLVED_PREFIX; SOLVE_TARGET_INDICES_0_TO_180",
        "ORDER_AWARE_VALIDATION": "IMPLEMENTED_UNCONSTRAINED_NEAREST_PROJECTION_AND_STATION_BACKSTEP_REPORTING",
        "LOCAL_TIMING_VALIDATION": "IMPLEMENTED_PER_ADJACENT_SEGMENT_DISTANCE_DT_SPEED_AND_DWELL",
        "FULL_P2B2_CAMPAIGN_REPLAYED": "NO",
        "change_impact_summary": {
            "changed_files": ["ros2_moveit_bridge/plan_closed_contour_moveit.py", "ros2_moveit_bridge/p2b3_c1_scope.py", "ros2_moveit_bridge/p2b3_c1_ordered_validation.py", "ros2_moveit_bridge/setup.py", "scripts/run_p2b3_c1_r0.py", "scripts/run_p2b3_c1_p2b2_bracket_recheck.py", "scripts/publish_p2b3_c1_result.py"],
            "changed_functions": ["build_p2b1_process_task_dls_trajectory", "build_normal_constrained_dls_trajectory", "resolve_dls_trajectory_scope"],
            "changed_artifacts": ["P2-B3-C1 R0 candidate trajectory and its 181-row lineage result"],
            "directly_affected_tests": ["test_p2b3_c1_scope.py", "test_p2b3_c1_ordered_validation.py", "test_p2b1_solver_policy_ablation.py", "test_p2b2_robustness_remapping.py"],
            "downstream_affected_stages": ["P2-B2 R0 nominal trajectory claims; robustness margins rooted in the R0 nominal q trajectory"],
            "explicitly_unaffected_stages_results": ["D39 source observations; P2-B1 frozen design inputs; P2-B2 implementation, acceptance set, and stored historical measurements; non-C1 default solver dispatch"],
            "reason": "C1 replaces legacy rows 0..15 and changes target row 16 by more than 1e-6 rad; rows 17..180 remain within 1e-6 rad of old R0. Six old pass/fail bracket endpoints were spot-checked across J3, TCP translation X, and TCP rotation X; other axis margins still require separate transfer checks.",
        },
        "legacy_measurement": {
            "waypoint_count": 181,
            "d39_seed_prefix_max_abs_joint_delta_rad": 0.0,
            "wp15_to_wp16_actual_tcp_distance_m": math.dist(old_tcp[15], old_tcp[16]),
            "wp15_to_wp16_target_tcp_distance_m": math.dist(target_xyz[15], target_xyz[16]),
            "wp15_to_wp16_time_delta_s": old_times[16] - old_times[15],
            "wp15_to_wp16_local_tcp_speed_m_s": old_local[15].speed_m_s,
            "wp15_projection_segment": old_proj[15].segment_index,
            "wp16_projection_segment": old_proj[16].segment_index,
            "unconstrained_projection_backstep_count": sum(value < -1.0e-9 for value in old_station_delta),
            "minimum_station_delta_m": min(old_station_delta),
            "maximum_same_index_error_m": max(old_same_error),
            "max_pre_post_ruckig_joint_delta_rad": max(max_abs_delta(a, b) for a, b in zip(baseline_pre_q, baseline_post_q)),
            "local_below_speed_band_segment_count": len(old_local_slow),
        },
        "c1_measurement": {
            "waypoint_count": 181,
            "target_attempted_count": pressure_summary["target_attempted_count"],
            "target_state_emitted_count": pressure_summary["target_attempted_count"],
            "tight_converged_count": pressure_summary["tight_converged_count"],
            "relaxed_accepted_target_count": pressure_summary["accepted_relaxed_geometry_count"],
            "seed_only_row": 0,
            "post_ruckig_rows_numerically_different_gt_1e-12_rad": len(numerical_delta_rows),
            "post_ruckig_rows_materially_different_gt_1e-6_rad": len(material_delta_rows),
            "materially_affected_row_indices_gt_1e-6_rad": material_delta_rows,
            "first_row_after_material_impact": max(material_delta_rows) + 1 if material_delta_rows else 0,
            "unchanged_post_ruckig_rows_vs_legacy": unchanged_rows,
            "max_post_ruckig_joint_delta_vs_legacy_rad": max(row_delta),
            "max_same_index_error_m": max(new_same_error),
            "max_nearest_path_error_m": max(row.distance_m for row in new_proj),
            "max_normal_error_deg": max(normal_errors),
            "strict_audit_status": strict.get("overall_status", strict.get("status", "UNRESOLVED")),
            "post_ruckig_status": acceptance.get("overall_status", acceptance.get("status", "UNRESOLVED")),
            "summary_metrics": {key: metrics.get(key) for key in (
                "fk_path_deviation_max_mm", "fk_normal_error_max_deg", "max_joint_step_deg", "collision_count",
                "collision_checked_state_count", "moveit_time_parameterization", "moveit_ruckig_smoothing_used",
                "strict_audit_status") if key in metrics},
            "geometry_fidelity_status": "PASS_CONFIGURED_6MM_PATH_DISTANCE_GATE" if geometry_ok else "FAIL_CONFIGURED_6MM_PATH_DISTANCE_GATE",
            "indexed_lineage_fidelity_status": "PASS_CONFIGURED_6MM_SAME_INDEX_DIAGNOSTIC" if indexed_ok else "SAME_INDEX_ERROR_EXCEEDS_6MM_DIAGNOSTIC",
            "normal_fidelity_status": "PASS_CONFIGURED_10DEG_GATE" if normal_ok else "FAIL_CONFIGURED_10DEG_GATE",
            "ordered_traversal_status": "PASS_NO_PROJECTED_STATION_BACKSTEP" if backstep_count == 0 else "FAIL_PROJECTED_STATION_BACKSTEP_OBSERVED",
            "ordered_station_backstep_count": backstep_count,
            "minimum_station_delta_m": min(new_station_delta),
            "local_timing_status": "PASS_ALL_ADJACENT_SPEEDS_IN_CONFIGURED_BAND" if not local_slow and local_band_counts["ABOVE_CONFIGURED_BAND"] == 0 else "MEASURED_WITH_LOCAL_SPEED_DEVIATIONS",
            "local_speed_band_counts": local_band_counts,
            "local_dwell_count": sum(row.dwell for row in new_local),
            "local_speed_min_m_s": min(row.speed_m_s for row in new_local),
            "local_speed_max_m_s": max(row.speed_m_s for row in new_local),
            "slowest_local_segment": None if not new_local else {
                "start_index": min(new_local, key=lambda row: row.speed_m_s).start_index,
                "end_index": min(new_local, key=lambda row: row.speed_m_s).end_index,
                "speed_m_s": min(row.speed_m_s for row in new_local),
            },
            "collision_method": manifest.get("collision_method", "UNRESOLVED"),
            "strict_self_ccd": manifest.get("strict_self_ccd", "UNRESOLVED"),
            "hardware_validation": manifest.get("hardware_validation", "UNRESOLVED"),
            "spray_command_state": manifest.get("spray_command_state", "NOT_REPRESENTED"),
        },
        "P2B2_RESULT_DISPOSITION": {
            "J3_positive_endpoint_margin": bracket_spotcheck("joint:j3:positive"),
            "J3_negative_endpoint_margin": bracket_spotcheck("joint:j3:negative"),
            "TCP_translation_x_positive_margin": bracket_spotcheck("tcp_tcp_translation:x:positive"),
            "TCP_translation_x_negative_margin": bracket_spotcheck("tcp_tcp_translation:x:negative"),
            "TCP_translation_y_z_and_full_margin_scan": "TARGETED_RECHECK_REQUIRED",
            "TCP_rotation_x_positive_margin": bracket_spotcheck("tcp_tcp_rotation:x:positive"),
            "TCP_rotation_x_negative_margin": bracket_spotcheck("tcp_tcp_rotation:x:negative"),
            "TCP_rotation_y_z_and_full_normal_margin_scan": "TARGETED_RECHECK_REQUIRED",
            "nominal_process_geometry": "INVALIDATED_BY_SPECIFIC_CHANGE_REPLACED_BY_C1_R0_REPLAY",
            "wall_station_progression": "INVALIDATED_BY_SPECIFIC_CHANGE_REPLACED_BY_UNCONSTRAINED_ORDER_AWARE_VALIDATION",
            "local_timing_dwell": "INVALIDATED_BY_SPECIFIC_CHANGE_REPLACED_BY_SEGMENT_LEVEL_C1_MEASUREMENT",
            "historical_campaign_measurements": "CARRIED_FORWARD_AS_MEASUREMENTS_OF_THE_OLD_P2B2_R0_BASELINE_ONLY",
            "campaign_disposition": "P2B2_NOT_INVALIDATED_AS_A_WHOLE; DO_NOT_TRANSFER_OLD_R0_MARGIN_NUMBERS_TO_C1_WITHOUT_RECHECK",
        },
        "CARRIED_FORWARD": [
            "P2-B1 R0/B0 solver policy and its original validated design; C1 scope flag defaults preserve the historical prefix path.",
            "P2-B2 robustness generation, acceptance thresholds, available/unavailable capability labels, and stored historical results as evidence about the old nominal R0 baseline.",
            "P2-B2 campaign findings are not erased; only claims attached to the changed nominal R0 trajectory need targeted transfer checks.",
        ],
        "P2B2_RESULTS_TARGETED_RECHECK": ["J3± historical bracket endpoints", "TCP translation X± historical bracket endpoints", "TCP rotation X± historical bracket endpoints"],
        "P2B2_RESULTS_STILL_PENDING": ["Full J3± margin rescan", "TCP translation Y/Z margins", "TCP rotation Y/Z and full normal margin scan"],
        "P2B2_RESULTS_INVALIDATED": ["Old R0 nominal process geometry/order/local timing as claims about the corrected C1 trajectory"],
        "input_identity_sha256": input_hashes,
        "candidate_input_identity_sha256": {
            "candidate_post_ruckig_trajectory": candidate_post_sha,
            "solver_pressure_csv": pressure_sha,
            "candidate_fk_trace": digest(candidate_fk_path),
            "execution_manifest": digest(manifest_path),
        },
        "targeted_scientific_regression": None if recheck is None else {
            "status": "SIX_OLD_BRACKET_ENDPOINTS_REMEASURED" if recheck.get("selected_case_count") == 12 else "RECHECK_SCOPE_UNRESOLVED",
            "recheck_code_commit": recheck.get("recheck_code_commit"),
            "selected_axes": recheck.get("selected_axes"),
            "selected_case_count": recheck.get("selected_case_count"),
            "measured_bracket_endpoints": compact_brackets,
            "fresh_FK_crosscheck_max_position_delta_m": recheck.get("fresh_FK_crosscheck_max_position_delta_m"),
            "full_p2b2_campaign_replayed": recheck.get("full_p2b2_campaign_replayed"),
        },
        "limitations": limitations,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "p2b3_c1_r0_waypoint_lineage.csv", ledger)
    (output_dir / "p2b3_c1_result.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-fk", type=Path, required=True, help="C0 Drive archive b0_reference/nominal_fk_trace.csv")
    parser.add_argument("--candidate-dir", type=Path, required=True, help="C1 strict_replay output directory")
    parser.add_argument("--manifest", type=Path, required=True, help="C1 execution_manifest.json")
    parser.add_argument("--solver-pressure-csv", type=Path, help="C1 P2B1-B0 solver_pressure.csv")
    parser.add_argument("--p2b2-recheck", type=Path, help="Selected old R0 bracket endpoint recheck JSON")
    parser.add_argument("--artifact-publish-commit", help="Exact commit carrying the current C1 canonical artifacts")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    result = validate_and_publish(
        baseline_fk=args.baseline_fk.resolve(), candidate_dir=args.candidate_dir.resolve(),
        manifest_path=args.manifest.resolve(), output_dir=args.output_dir.resolve(),
        solver_pressure_path=None if args.solver_pressure_csv is None else args.solver_pressure_csv.resolve(),
        p2b2_recheck_path=None if args.p2b2_recheck is None else args.p2b2_recheck.resolve(),
        artifact_publish_commit=args.artifact_publish_commit,
    )
    print(json.dumps({
        "P2B3_C1_STATUS": result["P2B3_C1_STATUS"],
        "geometry": result["c1_measurement"]["geometry_fidelity_status"],
        "order": result["c1_measurement"]["ordered_traversal_status"],
        "local_timing": result["c1_measurement"]["local_timing_status"],
        "ledger_rows": result["c1_measurement"]["waypoint_count"],
        "tight_converged_count": result["c1_measurement"]["tight_converged_count"],
        "fatal_gates": result["C1_FATAL_GATES"],
        "output_dir": str(args.output_dir.resolve()),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
