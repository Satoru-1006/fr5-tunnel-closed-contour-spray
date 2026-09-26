"""Audit Stage 2.5R native Ruckig evidence without finite-difference jerk."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable


JOINTS = ["j1", "j2", "j3", "j4", "j5", "j6"]
REQUIRED_JERK = 8.0
TOL = 1e-10


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSONL {path}:{line_number}: {exc}") from exc
    return records


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(records: Iterable[Any]) -> str:
    h = hashlib.sha256()
    for record in records:
        h.update(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
        h.update(b"\n")
    return h.hexdigest()


def canonical_file_hash(path: Path) -> str:
    return canonical_hash(read_jsonl(path))


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def vector(row: dict[str, str], prefix: str) -> list[float]:
    return [float(row[f"{joint}_{prefix}"]) for joint in JOINTS]


def finite_vector(values: Iterable[float]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def close(a: float, b: float, tol: float = TOL) -> bool:
    return abs(float(a) - float(b)) <= tol


def find_joint_limits(path: Path) -> dict[str, dict[str, float | bool]]:
    result: dict[str, dict[str, float | bool]] = {}
    current: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^  (j[1-6]):\s*$", line)
        if match:
            current = match.group(1)
            result[current] = {}
            continue
        if current is None:
            continue
        match = re.match(r"^    (has_velocity_limits|has_acceleration_limits|has_jerk_limits):\s*(true|false)\s*$", line)
        if match:
            result[current][match.group(1)] = match.group(2) == "true"
            continue
        match = re.match(r"^    (max_velocity|max_acceleration|max_jerk):\s*([-+0-9.eE]+)\s*$", line)
        if match:
            result[current][match.group(1)] = float(match.group(2))
    return result


def observed_call_summary(calls: list[dict[str, Any]], resolutions: list[dict[str, Any]]) -> dict[str, Any]:
    resolution_by_call = {int(item["global_calculate_call_index"]): item for item in resolutions}
    accepted = [item for item in resolutions if item.get("accepted_segment_call") is True]
    retry = [item for item in resolutions if item.get("retry_call") is True]
    graph = []
    for call in calls:
        resolution = resolution_by_call.get(int(call["global_calculate_call_index"]), {})
        graph.append(
            {
                "global_calculate_call_index": call["global_calculate_call_index"],
                "waypoint_idx": call["waypoint_idx"],
                "attempt_index": call["attempt_index"],
                "duration_extension_factor": call["duration_extension_factor"],
                "numeric_result": call["result"]["numeric_result"],
                "result_name": call["result"]["result_name"],
                "overshoot": resolution.get("overshoot"),
                "accepted_segment_call": resolution.get("accepted_segment_call"),
                "retry_call": resolution.get("retry_call"),
                "native_ruckig_output_duration": resolution.get("native_ruckig_output_duration"),
            }
        )
    return {
        "calls_total": len(calls),
        "accepted_segment_calls": len(accepted),
        "retry_calls": len(retry),
        "accepted_waypoints": [item["waypoint_idx"] for item in accepted],
        "last_called_waypoint_idx": max((int(item["waypoint_idx"]) for item in calls), default=None),
        "call_graph": graph,
        "call_graph_hash": canonical_hash(graph),
    }


def limits_and_profiles(
    calls: list[dict[str, Any]],
    resolutions: list[dict[str, Any]],
    nominal_segments: int,
    position_lower: list[float],
    position_upper: list[float],
) -> dict[str, Any]:
    resolution_by_call = {int(item["global_calculate_call_index"]): item for item in resolutions}
    accepted_calls = [
        call for call in calls if resolution_by_call.get(int(call["global_calculate_call_index"]), {}).get("accepted_segment_call") is True
    ]
    max_jerk_vectors = [tuple(float(v) for v in call["input"]["max_jerk"]) for call in calls]
    model_bounds = calls[0].get("robot_model_bounds", []) if calls else []
    effective = list(max_jerk_vectors[0]) if max_jerk_vectors else []
    limits_consistent = bool(max_jerk_vectors) and all(vector == tuple(effective) for vector in max_jerk_vectors)
    model_effective = [float(item["max_jerk"]) if item.get("jerk_bounded") else None for item in model_bounds]
    model_proven = bool(model_bounds) and all(item.get("jerk_bounded") is True for item in model_bounds)
    model_proven = model_proven and all(close(float(item["max_jerk"]), effective[index]) for index, item in enumerate(model_bounds))

    worst: dict[str, Any] = {"max_abs_jerk": 0.0, "max_jerk_ratio": 0.0, "segment": None, "joint": None, "phase": None}
    position_passed = True
    velocity_passed = True
    acceleration_passed = True
    jerk_passed = True
    profile_count = 0
    for call in accepted_calls:
        output = call.get("native_output", {})
        profiles = output.get("profiles") or []
        extrema = output.get("kinematic_extrema") or []
        if not output.get("profile_available") or len(profiles) != len(JOINTS):
            jerk_passed = False
            continue
        profile_count += 1
        max_velocity = [float(v) for v in call["input"]["max_velocity"]]
        max_acceleration = [float(v) for v in call["input"]["max_acceleration"]]
        for joint_index, profile in enumerate(profiles):
            position = output.get("position_extrema", [])[joint_index]
            position_passed = position_passed and position["min"] >= position_lower[joint_index] - TOL and position["max"] <= position_upper[joint_index] + TOL
            if joint_index < len(extrema):
                velocity_passed = velocity_passed and float(extrema[joint_index]["max_abs_velocity"]) <= max_velocity[joint_index] + TOL
                acceleration_passed = acceleration_passed and float(extrema[joint_index]["max_abs_acceleration"]) <= max_acceleration[joint_index] + TOL
            jerks = [abs(float(value)) for value in profile.get("j", [])]
            max_abs = max(jerks, default=0.0)
            if max_abs > worst["max_abs_jerk"]:
                phase = jerks.index(max_abs) if jerks else None
                worst = {
                    "max_abs_jerk": max_abs,
                    "max_jerk_ratio": max_abs / REQUIRED_JERK if REQUIRED_JERK else None,
                    "segment": call["waypoint_idx"],
                    "joint": JOINTS[joint_index],
                    "phase": phase,
                }
            jerk_passed = jerk_passed and max_abs <= REQUIRED_JERK + TOL and max_abs <= float(call["input"]["max_jerk"][joint_index]) + TOL
    coverage_complete = len({int(item["waypoint_idx"]) for item in resolutions if item.get("accepted_segment_call") is True}) == nominal_segments
    return {
        "formal_effective_ruckig_max_jerk": {joint: effective[index] for index, joint in enumerate(JOINTS)} if len(effective) == 6 else None,
        "effective_native_limit_used_by_ruckig": {joint: effective[index] for index, joint in enumerate(JOINTS)} if len(effective) == 6 else None,
        "formal_project_required_limit": {joint: REQUIRED_JERK for joint in JOINTS},
        "all_joints_effectively_8_rad_s3": len(effective) == 6 and all(close(value, REQUIRED_JERK) for value in effective),
        "robot_model_bounds_effective_jerk": {joint: model_effective[index] for index, joint in enumerate(JOINTS)} if len(model_effective) == 6 else None,
        "robot_model_bounds": model_bounds,
        "effective_limit_all_calls_identical": limits_consistent,
        "formal_effective_ruckig_max_jerk_proven": limits_consistent and model_proven,
        "profiled_accepted_segment_count": profile_count,
        "native_segment_coverage_complete": coverage_complete,
        "native_jerk_certification": {
            "passed": jerk_passed and coverage_complete,
            "observed_segments_passed": jerk_passed,
            "position_limits": {"passed": position_passed},
            "velocity_limits": {"passed": velocity_passed},
            "acceleration_limits": {"passed": acceleration_passed},
            "jerk_limits": {"passed": jerk_passed},
            "max_abs_jerk": worst["max_abs_jerk"],
            "max_jerk_ratio": worst["max_jerk_ratio"],
            "worst_segment": worst["segment"],
            "worst_joint": worst["joint"],
            "worst_phase": worst["phase"],
            "jerk_source": "native Profile.j; no finite difference used",
        },
        "all_native_segments_pass_required_8_rad_s3": jerk_passed and coverage_complete,
    }


def duration_audit(
    calls: list[dict[str, Any]],
    resolutions: list[dict[str, Any]],
    exported: list[dict[str, str]],
    summary: dict[str, Any],
) -> dict[str, Any]:
    accepted_by_segment: dict[int, dict[str, Any]] = {}
    for resolution in resolutions:
        if resolution.get("accepted_segment_call") is True:
            accepted_by_segment[int(resolution["waypoint_idx"])] = resolution
    match = mismatch = unavailable = 0
    max_abs_error = 0.0
    for segment in range(len(exported) - 1):
        resolution = accepted_by_segment.get(segment)
        if resolution is None:
            unavailable += 1
            continue
        exported_dt = float(exported[segment + 1]["t"]) - float(exported[segment]["t"])
        native_duration = float(resolution["native_ruckig_output_duration"])
        error = abs(exported_dt - native_duration)
        max_abs_error = max(max_abs_error, error)
        if exported_dt == native_duration:
            match += 1
        else:
            mismatch += 1
    sources = summary.get("duration_sources", [])
    source_counts = {source: sources.count(source) for source in sorted(set(sources))}
    extensions = summary.get("changed_by_duration_extension", [])
    finals = summary.get("changed_by_final_segment_assignment", [])
    return {
        "segments_total": len(exported) - 1,
        "segments_with_accepted_native_duration": len(accepted_by_segment),
        "segments_whose_exported_dt_equals_native_ruckig_duration": match,
        "segments_whose_exported_dt_differs_from_native_ruckig_duration": mismatch,
        "segments_without_accepted_native_duration": unavailable,
        "exported_csv_dt_native_duration_match_count": match,
        "exported_csv_dt_native_duration_mismatch_count": mismatch,
        "max_abs_exported_dt_native_duration_error_s": max_abs_error,
        "duration_source_counts": source_counts,
        "segments_retaining_TOTG_duration": source_counts.get("original_TOTG", 0),
        "segments_changed_by_duration_extension": sum(bool(value) for value in extensions),
        "segments_changed_by_final_segment_assignment": sum(bool(value) for value in finals),
        "duration_source_arrays_from_native_run_summary": {
            "duration_sources": sources,
            "changed_by_duration_extension": extensions,
            "changed_by_final_segment_assignment": finals,
        },
    }


def sample_audit(native: Path, calls: list[dict[str, Any]]) -> dict[str, Any]:
    samples = read_jsonl(native / "stage25r_native_samples.jsonl")
    calls_by_key = {(int(call["waypoint_idx"]), int(call["attempt_index"])): call for call in calls}
    compared = mismatched = 0
    for sample in samples:
        call = calls_by_key.get((int(sample["segment_index"]), int(sample["attempt_index"])))
        if not call:
            mismatched += 1
            continue
        profiles = call.get("native_output", {}).get("profiles") or []
        if len(profiles) != len(JOINTS):
            mismatched += 1
            continue
        for index, phase in enumerate(sample["phase"]):
            if int(phase) == -1:
                compared += 1
                if not close(float(sample["native_jerk"][index]), 0.0, 1e-12):
                    mismatched += 1
                continue
            if int(phase) <= -2:
                brake_index = -2 - int(phase)
                brake_j = (profiles[index].get("brake") or {}).get("j", [])
                if brake_index >= len(brake_j):
                    mismatched += 1
                    continue
                compared += 1
                if not close(float(sample["native_jerk"][index]), float(brake_j[brake_index]), 1e-12):
                    mismatched += 1
                continue
            j = profiles[index].get("j", [])
            if int(phase) >= len(j):
                mismatched += 1
                continue
            compared += 1
            if not close(float(sample["native_jerk"][index]), float(j[int(phase)]), 1e-12):
                mismatched += 1
    return {
        "sample_records": len(samples),
        "sample_native_jerk_comparisons": compared,
        "sample_native_jerk_profile_mismatches": mismatched,
        "sample_trace_profile_consistent": mismatched == 0,
        "finite_difference_jerk": "derived_diagnostic_only; not used for certification",
    }


def interval_5_audit(
    calls: list[dict[str, Any]],
    resolutions: list[dict[str, Any]],
    exported: list[dict[str, str]],
    semantics_path: Path,
) -> dict[str, Any]:
    segment = 5
    actual_calls = [call for call in calls if int(call["waypoint_idx"]) == segment]
    resolution_by_call = {int(item["global_calculate_call_index"]): item for item in resolutions}
    accepted = next((call for call in actual_calls if resolution_by_call[int(call["global_calculate_call_index"])] .get("accepted_segment_call") is True), None)
    row0 = exported[segment]
    row1 = exported[segment + 1]
    exported_payload = {
        "dt": float(row1["t"]) - float(row0["t"]),
        "dq0": float(row0["j3_dq"]),
        "dq1": float(row1["j3_dq"]),
        "ddq0": float(row0["j3_ddq"]),
        "ddq1": float(row1["j3_ddq"]),
    }
    result: dict[str, Any] = {
        "csv_interval": segment,
        "joint": "j3",
        "waypoint_idx": segment if accepted else None,
        "attempt_index": accepted.get("attempt_index") if accepted else None,
        "native_calculate_call_found": accepted is not None,
        "native_input": None,
        "native_output": None,
        "exported": exported_payload,
        "exported_dt_equals_native_duration": None,
        "exported_endpoint_states_equal_native_input": None,
    }
    if accepted:
        inp = accepted["input"]
        out = accepted["native_output"]
        profile = (out.get("profiles") or [])[2] if len(out.get("profiles") or []) > 2 else None
        result["native_input"] = {
            "q0": inp["current_position"][2],
            "dq0": inp["current_velocity"][2],
            "ddq0": inp["current_acceleration"][2],
            "q1": inp["target_position"][2],
            "dq1": inp["target_velocity"][2],
            "ddq1": inp["target_acceleration"][2],
            "max_jerk": inp["max_jerk"][2],
        }
        result["native_output"] = {
            "duration": out.get("duration"),
            "phase_boundaries": profile.get("cumulative_phase_boundaries") if profile else None,
            "phase_jerks": profile.get("j") if profile else None,
            "max_abs_jerk": max((abs(float(v)) for v in (profile.get("j", []) if profile else [])), default=None),
        }
        result["exported_dt_equals_native_duration"] = exported_payload["dt"] == float(out["duration"])
        result["exported_endpoint_states_equal_native_input"] = all(
            close(exported_payload[key], result["native_input"][key], 1e-12) for key in ("dq0", "dq1", "ddq0", "ddq1")
        )
    else:
        upstream = [call for call in calls if int(call["waypoint_idx"]) == segment - 1]
        last_upstream = upstream[-1] if upstream else None
        result["actual_call_gap"] = {
            "calls_for_waypoint_5": 0,
            "last_calculate_waypoint_idx": calls[-1]["waypoint_idx"] if calls else None,
            "last_calculate_attempt_index": calls[-1]["attempt_index"] if calls else None,
            "last_upstream_call_target_waypoint_5": (
                {
                    "waypoint_idx": last_upstream["waypoint_idx"],
                    "attempt_index": last_upstream["attempt_index"],
                    "global_calculate_call_index": last_upstream["global_calculate_call_index"],
                    "target_position_j3": last_upstream["input"]["target_position"][2],
                    "target_velocity_j3": last_upstream["input"]["target_velocity"][2],
                    "target_acceleration_j3": last_upstream["input"]["target_acceleration"][2],
                    "result": last_upstream["result"],
                    "resolution": resolution_by_call[int(last_upstream["global_calculate_call_index"])],
                }
                if last_upstream
                else None
            ),
        }
    semantics = read_json(semantics_path) if semantics_path.exists() else None
    result["endpoint_feasibility_audit"] = semantics.get("interval_5_j3") if semantics else None
    result["explanation"] = (
        "The exported interval 5 fixes the exported dt and endpoint dq/ddq and asks whether a jerk-bounded "
        "acceleration profile exists at J=8. That test is false for the exported waypoint derivatives. "
        "A native Ruckig calculate result is evaluated on the actual InputParameter for one local segment, "
        "with its own native duration and Profile; it is not the exported finite-difference endpoint-feasibility "
        "test. In this formal run no calculate call for waypoint_idx=5 exists: MoveIt reached the duration-"
        "extension ceiling while retrying waypoint_idx=4. The last call still returned numeric Result::Working, "
        "but MoveIt rejected it because overshoot=true and then returned according to its 2.12.4 result-only final "
        "status. Thus the J=8=false result and a successful Working calculate result can coexist without being "
        "the same interval. The exported row 5 is not evidence of a native interval-5 trajectory."
    )
    return result


def final_trajectory_comparison(recovery: list[dict[str, str]], frozen: list[dict[str, str]]) -> dict[str, Any]:
    q_equal = len(recovery) == len(frozen) and all(
        [float(row[f"{joint}_q"]) for joint in JOINTS] == [float(row[f"{joint}_q"]) for joint in JOINTS]
        for row in []
    )
    if len(recovery) != len(frozen):
        q_equal = t_equal = False
    else:
        q_equal = all(
            all(float(recovery[index][f"{joint}_q"]) == float(frozen[index][f"{joint}_q"]) for joint in JOINTS)
            for index in range(len(recovery))
        )
        t_equal = all(float(recovery[index]["t"]) == float(frozen[index]["t"]) for index in range(len(recovery)))
    return {
        "q_bit_identical": q_equal,
        "t_bit_identical": t_equal,
        "recovery_csv_sha256": sha256(Path(recovery[0]["__path__"])) if recovery and "__path__" in recovery[0] else None,
    }


def audit_rebuild(rebuild: Path, frozen_csv: Path, nominal_segments: int, limits_yaml: Path, semantics_path: Path) -> dict[str, Any]:
    native = rebuild / "native"
    moveit = rebuild / "moveit"
    calls = read_jsonl(native / "stage25r_ruckig_calls.jsonl")
    resolutions = read_jsonl(native / "stage25r_ruckig_call_resolutions.jsonl")
    summary = read_json(native / "stage25r_native_run_summary.json")
    exported = load_csv(moveit / "stage25_ruckig_trajectory.csv")
    frozen = load_csv(frozen_csv)
    ruckig_audit = read_json(moveit / "stage25_ruckig_audit.json")["audit"]
    profile = limits_and_profiles(calls, resolutions, nominal_segments, ruckig_audit["position_lower_rad"], ruckig_audit["position_upper_rad"])
    chain_yaml = find_joint_limits(limits_yaml)
    yaml_jerk = [float(chain_yaml[joint]["max_jerk"]) for joint in JOINTS]
    input_jerk = [float(v) for v in calls[0]["input"]["max_jerk"]]
    model_bounds = calls[0]["robot_model_bounds"]
    chain = {
        "yaml_path": str(limits_yaml),
        "yaml_max_jerk": {joint: yaml_jerk[index] for index, joint in enumerate(JOINTS)},
        "launch_uses_joint_limits_yaml": ".joint_limits(file_path=" in (Path(__file__).resolve().parents[1] / "tools" / "stage25_moveit_launch.py").read_text(encoding="utf-8"),
        "robot_model_variable_bounds": model_bounds,
        "first_runtime_ruckig_input_max_jerk": {joint: input_jerk[index] for index, joint in enumerate(JOINTS)},
        "yaml_to_input_exact": yaml_jerk == input_jerk,
        "yaml_to_robot_model_exact": all(item.get("jerk_bounded") is True and close(float(item["max_jerk"]), yaml_jerk[index]) for index, item in enumerate(model_bounds)),
        "robot_model_to_input_exact": all(close(float(item["max_jerk"]), input_jerk[index]) for index, item in enumerate(model_bounds)),
    }
    recovery_rows = load_csv(moveit / "stage25_ruckig_trajectory.csv")
    return {
        "rebuild_path": str(rebuild),
        "call_summary": {key: value for key, value in observed_call_summary(calls, resolutions).items() if key != "call_graph"},
        "call_graph_hash": observed_call_summary(calls, resolutions)["call_graph_hash"],
        "native_trace_hash": canonical_file_hash(native / "stage25r_ruckig_calls.jsonl"),
        "resolution_trace_hash": canonical_file_hash(native / "stage25r_ruckig_call_resolutions.jsonl"),
        "sample_trace_hash": canonical_file_hash(native / "stage25r_native_samples.jsonl"),
        "input_model_limit_chain": chain,
        "immediate_runtime_input_capture": {
            "all_calls_captured_immediately_before_calculate": all(call.get("captured_immediately_before_calculate") is True for call in calls),
            "all_calls_captured_immediately_after_calculate": all(call.get("captured_immediately_after_calculate") is True for call in calls),
            "calls_with_input_after_calculate": all("input_after_calculate" in call for call in calls),
            "calls_with_complete_required_input_fields": all(
                all(field in call.get("input", {}) for field in (
                    "current_position", "current_velocity", "current_acceleration", "target_position", "target_velocity",
                    "target_acceleration", "max_velocity", "max_acceleration", "max_jerk", "min_velocity", "min_acceleration",
                    "enabled", "control_interface", "synchronization", "duration_discretization", "per_dof_control_interface",
                    "per_dof_synchronization", "minimum_duration", "intermediate_positions"
                )) for call in calls
            ),
            "input_before_after_runtime_objects_identical": all(
                call.get("input") == call.get("input_after_calculate") for call in calls
            ),
        },
        "profile_and_limit_certification": profile,
        "duration_propagation": duration_audit(calls, resolutions, recovery_rows, summary),
        "native_sampling": sample_audit(native, calls),
        "interval_5_j3": interval_5_audit(calls, resolutions, recovery_rows, semantics_path),
        "trajectory_vs_frozen_stage25": {
            "q_bit_identical": len(recovery_rows) == len(frozen) and all(
                all(float(recovery_rows[i][f"{joint}_q"]) == float(frozen[i][f"{joint}_q"]) for joint in JOINTS) for i in range(len(recovery_rows))
            ),
            "t_bit_identical": len(recovery_rows) == len(frozen) and all(float(recovery_rows[i]["t"]) == float(frozen[i]["t"]) for i in range(len(recovery_rows))),
            "recovery_csv_sha256": sha256(moveit / "stage25_ruckig_trajectory.csv"),
            "frozen_csv_sha256": sha256(frozen_csv),
        },
        "nominal_segments": nominal_segments,
        "exported_rows": len(exported),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    repo = Path(__file__).resolve().parents[1]
    frozen_dir = repo / "outputs" / "ik_graph_stage25_timed_certification" / "fr5_scaled_horseshoe_demo_v45_20260803_formal_000002_Stage25"
    frozen_csv = frozen_dir / "stage25_ruckig_trajectory.csv"
    limits_yaml = repo / "ros2_moveit_bridge" / "config" / "joint_limits_with_jerk.yaml"
    semantics_path = repo / "outputs" / "stage25_semantics_audit_20260804_corrected" / "stage25_jerk_endpoint_feasibility.json"
    frozen_rows = load_csv(frozen_csv)
    nominal_segments = len(frozen_rows) - 1
    rebuilds = [audit_rebuild(run_root / f"rebuild_{index:02d}", frozen_csv, nominal_segments, limits_yaml, semantics_path) for index in range(1, 4)]
    first = rebuilds[0]
    trace_fields = ["call_graph_hash", "native_trace_hash", "resolution_trace_hash", "sample_trace_hash"]
    determinism = {
        "rebuilds": 3,
        "native_trace_match": all(item[field] == first[field] for item in rebuilds for field in trace_fields),
        "calculate_call_graph_match": all(item["call_graph_hash"] == first["call_graph_hash"] for item in rebuilds),
        "input_result_profile_duration_trace_match": all(
            item["native_trace_hash"] == first["native_trace_hash"] and item["resolution_trace_hash"] == first["resolution_trace_hash"] for item in rebuilds
        ),
        "core_artifact_mismatch_count": sum(
            1 for item in rebuilds[1:] if item["trajectory_vs_frozen_stage25"]["recovery_csv_sha256"] != first["trajectory_vs_frozen_stage25"]["recovery_csv_sha256"]
        ),
    }
    frozen_hashes_before = read_json(run_root / "frozen_artifact_hashes_before.json")
    frozen_hashes_after = read_json(run_root / "frozen_artifact_hashes_after.json")
    frozen_unchanged = frozen_hashes_before == frozen_hashes_after
    effective_limits = first["profile_and_limit_certification"]["formal_effective_ruckig_max_jerk"]
    effective_proven = first["profile_and_limit_certification"]["formal_effective_ruckig_max_jerk_proven"]
    native_pass = first["profile_and_limit_certification"]["native_jerk_certification"]["observed_segments_passed"]
    coverage = first["profile_and_limit_certification"]["native_segment_coverage_complete"]
    duration = first["duration_propagation"]
    trajectory = first["trajectory_vs_frozen_stage25"]
    if not effective_proven:
        classification = "D"
    elif any(not close(float(value), REQUIRED_JERK) for value in (effective_limits or {}).values()):
        classification = "C"
    elif not native_pass:
        classification = "A"
    else:
        classification = "B"
    report = {
        "schema_version": "stage25r-gate-report-v1",
        "Stage_2_5R": {
            "recovery_unit": "one_ruckig_calculate_call",
            "formal_effective_ruckig_max_jerk": effective_limits,
            "actual_ruckig_max_jerk": effective_limits,
            "effective_native_limit_used_by_ruckig": first["profile_and_limit_certification"]["effective_native_limit_used_by_ruckig"],
            "formal_project_required_limit": first["profile_and_limit_certification"]["formal_project_required_limit"],
            "all_joints_effectively_8_rad_s3": first["profile_and_limit_certification"]["all_joints_effectively_8_rad_s3"],
            "formal_effective_ruckig_max_jerk_proven": effective_proven,
            "calculate_calls_total": first["call_summary"]["calls_total"],
            "accepted_segment_calls": first["call_summary"]["accepted_segment_calls"],
            "retry_calls": first["call_summary"]["retry_calls"],
            "last_called_waypoint_idx": first["call_summary"]["last_called_waypoint_idx"],
            "native_jerk_certification": first["profile_and_limit_certification"]["native_jerk_certification"],
            "all_native_segments_pass_required_8_rad_s3": first["profile_and_limit_certification"]["all_native_segments_pass_required_8_rad_s3"],
            "native_trajectory_violates_required_8_on_observed_segments": not native_pass,
            "exported_csv_dt_native_duration_match_count": duration["exported_csv_dt_native_duration_match_count"],
            "exported_csv_dt_native_duration_mismatch_count": duration["exported_csv_dt_native_duration_mismatch_count"],
            "exported_csv_dt_native_duration_unavailable_count": duration["segments_without_accepted_native_duration"],
            "segments_total": duration["segments_total"],
            "segments_retaining_TOTG_duration": duration["segments_retaining_TOTG_duration"],
            "segments_changed_by_duration_extension": duration["segments_changed_by_duration_extension"],
            "segments_changed_by_final_segment_assignment": duration["segments_changed_by_final_segment_assignment"],
            "interval_5_j3_explained": first["interval_5_j3"],
            "final_classification": classification,
            "classification_note": "B is the observed native-limit result; formal full-segment certification remains incomplete when MoveIt returns after the duration-extension ceiling before waypoint 5.",
        },
        "RobotModel_jerk_chain": first["input_model_limit_chain"],
        "determinism": determinism,
        "rebuilds": rebuilds,
        "frozen_artifacts": {
            "before_after_equal": frozen_unchanged,
            "before": frozen_hashes_before,
            "after": frozen_hashes_after,
        },
        "q_vs_frozen_stage25": trajectory["q_bit_identical"],
        "t_vs_frozen_stage25": trajectory["t_bit_identical"],
        "Stage_2_5": "passed" if classification == "B" and coverage else "requires_recertification",
        "Stage_2_6": {
            "formal_gate_recertification_required": True,
            "geometric_collision_recompute_required": not (trajectory["q_bit_identical"] and trajectory["t_bit_identical"]),
            "started": False,
        },
        "Stage_2_7": {"allowed_next": False, "started": False},
        "single_next_action": "Resolve the MoveIt Ruckig duration-extension early-return/native-coverage semantics and rerun Stage 2.5R full-segment certification; keep Stage 2.7 closed.",
    }
    (run_root / "stage25r_gate_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_root / "stage25r_report.yaml").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(run_root / "stage25r_gate_report.json"), "classification": classification, "frozen_unchanged": frozen_unchanged, "determinism": determinism}, indent=2))
    return 0 if frozen_unchanged and determinism["native_trace_match"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
