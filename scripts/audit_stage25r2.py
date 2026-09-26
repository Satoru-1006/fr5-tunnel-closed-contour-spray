"""Audit Stage 2.5R2 native traces and independent final replay artifacts.

The auditor is deliberately fail-closed.  It never treats MoveIt's boolean
return value as completion, never aggregates accepted calls across discarded
trajectory generations, and never derives formal jerk from finite differences.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


JOINTS = ["j1", "j2", "j3", "j4", "j5", "j6"]
REQUIRED_JERK = 8.0
JERK_TOLERANCE = 1.0e-9
POSITION_LOWER = [-3.0543, -4.6251, -2.8274, -4.6251, -3.0543, -3.0543]
POSITION_UPPER = [3.0543, 1.4835, 2.8274, 1.4835, 3.0543, 3.0543]
VELOCITY_LIMITS = [0.4725, 0.4725, 0.4725, 0.48, 0.48, 0.48]
ACCELERATION_LIMITS = [0.105] * 6
REPO_ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def case_artifact_path(case_root: Path, relative_path: str) -> Path:
    """Resolve a migrated duplicate to its retained canonical physical file."""
    manifest = read_json(case_root / "case_manifest.json", {}) or {}
    for record in manifest.get("canonical_content_migrations", []):
        if record.get("original_relative_path") != relative_path:
            continue
        canonical = Path(record["canonical_content_path"])
        if not canonical.is_absolute():
            canonical = REPO_ROOT / canonical
        if not canonical.is_file():
            raise FileNotFoundError(f"migrated canonical content is missing: {canonical}")
        expected_size = record.get("size_bytes")
        if expected_size is not None and canonical.stat().st_size != int(expected_size):
            raise ValueError(f"migrated canonical content size mismatch: {canonical}")
        return canonical
    return case_root / relative_path


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return []

    def iterator() -> Iterable[dict[str, Any]]:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)

    return iterator()


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(items: Iterable[Any]) -> str:
    digest = hashlib.sha256()
    for item in items:
        digest.update(json.dumps(item, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def numeric_result(call: dict[str, Any]) -> str:
    return str((call.get("result") or {}).get("result_name", "unknown"))


def profile_jerk_values(profile: dict[str, Any]) -> Iterable[tuple[str, int, float]]:
    for phase, value in enumerate(profile.get("j") or []):
        yield "main", phase, abs(float(value))
    brake = profile.get("brake") or {}
    for phase, value in enumerate(brake.get("j") or []):
        yield "brake", phase, abs(float(value))


def profile_stats(calls_path: Path, resolutions: list[dict[str, Any]]) -> dict[str, Any]:
    resolution_by_call = {int(item["global_calculate_call_index"]): item for item in resolutions}
    accepted_profile_count = 0
    accepted_pass = True
    accepted_max = 0.0
    accepted_worst: dict[str, Any] | None = None
    successful_profile_count = 0
    successful_pass = True
    successful_max = 0.0
    successful_worst: dict[str, Any] | None = None
    all_profile_count = 0
    all_calls = 0
    error_calls = 0
    all_profile_pass = True
    all_max = 0.0
    all_worst: dict[str, Any] | None = None
    input_hash = hashlib.sha256()
    profile_hash = hashlib.sha256()
    call_graph_hash = hashlib.sha256()
    call_waypoints: list[int] = []
    waypoint4_calls: list[dict[str, Any]] = []

    def update_worst(current: dict[str, Any] | None, value: float, call: dict[str, Any], phase: str, phase_index: int, joint: int) -> dict[str, Any]:
        if current is not None and value <= float(current["max_abs_jerk"]):
            return current
        return {
            "max_abs_jerk": value,
            "global_calculate_call_index": int(call["global_calculate_call_index"]),
            "waypoint_idx": int(call["waypoint_idx"]),
            "attempt_index": int(call["attempt_index"]),
            "joint": JOINTS[joint],
            "joint_index": joint,
            "phase": phase,
            "phase_index": phase_index,
        }

    with calls_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            call = json.loads(line)
            all_calls += 1
            result_name = numeric_result(call)
            if result_name not in {"Working", "Finished"}:
                error_calls += 1
            waypoint = int(call["waypoint_idx"])
            call_waypoints.append(waypoint)
            graph = {
                "global_calculate_call_index": int(call["global_calculate_call_index"]),
                "waypoint_idx": waypoint,
                "attempt_index": int(call["attempt_index"]),
                "trajectory_generation": int(call.get("trajectory_generation", 0)),
                "duration_extension_factor": call.get("duration_extension_factor"),
                "result": call.get("result"),
                "profile_available": bool((call.get("native_output") or {}).get("profile_available")),
            }
            call_graph_hash.update(json.dumps(graph, sort_keys=True, separators=(",", ":")).encode("utf-8"))
            call_graph_hash.update(b"\n")
            input_hash.update(json.dumps(call.get("input"), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))
            input_hash.update(b"\n")
            native_output = call.get("native_output") or {}
            profiles = native_output.get("profiles")
            available = bool(native_output.get("profile_available") is True and isinstance(profiles, list) and len(profiles) == 6)
            if available:
                all_profile_count += 1
                profile_hash.update(json.dumps(native_output, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))
                profile_hash.update(b"\n")
            successful_call = result_name in {"Working", "Finished"}
            accepted = bool((resolution_by_call.get(int(call["global_calculate_call_index"])) or {}).get("accepted_segment_call") is True)
            if available:
                call_max = 0.0
                call_worst: dict[str, Any] | None = None
                for joint, profile in enumerate(profiles):
                    for phase, phase_index, value in profile_jerk_values(profile):
                        call_max = max(call_max, value)
                        call_worst = update_worst(call_worst, value, call, phase, phase_index, joint)
                all_max = max(all_max, call_max)
                all_worst = update_worst(all_worst, call_max, call, (call_worst or {}).get("phase", "unknown"), int((call_worst or {}).get("phase_index", -1)), int((call_worst or {}).get("joint_index", 0)))
                if call_max > REQUIRED_JERK + JERK_TOLERANCE:
                    all_profile_pass = False
                if accepted:
                    accepted_profile_count += 1
                    accepted_max = max(accepted_max, call_max)
                    accepted_worst = update_worst(accepted_worst, call_max, call, (call_worst or {}).get("phase", "unknown"), int((call_worst or {}).get("phase_index", -1)), int((call_worst or {}).get("joint_index", 0)))
                    if call_max > REQUIRED_JERK + JERK_TOLERANCE:
                        accepted_pass = False
                if successful_call:
                    successful_profile_count += 1
                    successful_max = max(successful_max, call_max)
                    successful_worst = update_worst(successful_worst, call_max, call, (call_worst or {}).get("phase", "unknown"), int((call_worst or {}).get("phase_index", -1)), int((call_worst or {}).get("joint_index", 0)))
                    if call_max > REQUIRED_JERK + JERK_TOLERANCE:
                        successful_pass = False
            if waypoint == 4:
                resolution = resolution_by_call.get(int(call["global_calculate_call_index"]), {})
                worst = resolution.get("worst_overshoot_in_call") or {}
                waypoint4_calls.append(
                    {
                        "attempt": int(call["attempt_index"]),
                        "global_calculate_call_index": int(call["global_calculate_call_index"]),
                        "duration_extension_factor_at_call": call.get("duration_extension_factor"),
                        "duration_extension_factor_after_retry": resolution.get("duration_extension_factor"),
                        "segment_duration": resolution.get("robot_trajectory_segment_duration_after"),
                        "ruckig_result": call.get("result"),
                        "native_duration": native_output.get("duration"),
                        "overshoot": resolution.get("overshoot"),
                        "trigger_joint": (resolution.get("first_overshoot") or {}).get("joint_index"),
                        "trigger_time": (resolution.get("first_overshoot") or {}).get("local_time"),
                        "overshoot_magnitude_rad": (resolution.get("first_overshoot") or {}).get("abs_overshoot_rad"),
                        "worst_overshoot_magnitude_rad": worst.get("abs_overshoot_rad"),
                        "native_max_jerk": max((value for profile in (profiles or []) for _, _, value in profile_jerk_values(profile)), default=None),
                    }
                )
    return {
        "total_calls": all_calls,
        "successful_profile_calls": successful_profile_count,
        "error_result_calls": error_calls,
        "native_profiles_available": all_profile_count,
        "accepted_native_profiles": {
            "profile_count": accepted_profile_count,
            "passed_J8": accepted_pass,
            "max_abs_jerk": accepted_max,
            "worst_call": accepted_worst,
            "worst_waypoint": accepted_worst.get("waypoint_idx") if accepted_worst else None,
            "worst_joint": accepted_worst.get("joint") if accepted_worst else None,
            "worst_phase": accepted_worst.get("phase_index") if accepted_worst else None,
        },
        "all_successful_calculate_profiles": {
            "profile_count": successful_profile_count,
            "passed_J8": successful_pass,
            "max_abs_jerk": successful_max,
            "worst_call": successful_worst,
            "worst_waypoint": successful_worst.get("waypoint_idx") if successful_worst else None,
            "worst_attempt": successful_worst.get("attempt_index") if successful_worst else None,
            "worst_joint": successful_worst.get("joint") if successful_worst else None,
            "worst_phase": successful_worst.get("phase_index") if successful_worst else None,
        },
        "all_calculate_profiles": {
            "total_calls": all_calls,
            "successful_profile_calls": successful_profile_count,
            "error_result_calls": error_calls,
            "native_profiles_available": all_profile_count,
            "passed_J8_for_every_available_profile": all_profile_pass,
        },
        "all_available_native_profiles_pass_J8": all_profile_pass,
        "max_abs_jerk": all_max,
        "worst_call": all_worst,
        "input_trace_hash": input_hash.hexdigest(),
        "native_profile_hash": profile_hash.hexdigest(),
        "calculate_call_graph_hash": call_graph_hash.hexdigest(),
        "waypoints_observed": sorted(set(call_waypoints)),
        "waypoint4_retry_audit": waypoint4_calls,
    }


def generation_audit(native: Path, resolutions: list[dict[str, Any]], nominal_segments: int) -> dict[str, Any]:
    summary = read_json(native / "stage25r_native_run_summary.json", {}) or {}
    events = list(read_jsonl(native / "stage25r2_trajectory_generation_events.jsonl"))
    stale_file = read_json(native / "stage25r2_stale_accepted_calls.json", {}) or {}
    final_generation = int(summary.get("trajectory_generation_final", max((int(item.get("trajectory_generation", 0)) for item in resolutions), default=0)))
    final_accepted = sorted({int(item["waypoint_idx"]) for item in resolutions if item.get("accepted_segment_call") is True and int(item.get("trajectory_generation", -1)) == final_generation})
    all_accepted = sorted({int(item["waypoint_idx"]) for item in resolutions if item.get("accepted_segment_call") is True})
    stale_indices = sorted({int(value) for value in stale_file.get("call_indices", [])})
    if not stale_indices:
        for event in events:
            stale_indices.extend(int(value) for value in event.get("invalidated_accepted_call_indices", []))
        stale_indices = sorted(set(stale_indices))
    required = list(range(nominal_segments))
    return {
        "initial_generation": 0,
        "final_generation": final_generation,
        "reset_count": int(summary.get("trajectory_generation_reset_count", sum(event.get("event") == "trajectory_reset" for event in events))),
        "events": events,
        "unique_waypoints_ever_accepted": all_accepted,
        "unique_waypoints_accepted_in_final_generation": final_accepted,
        "all_nominal_segments_reached": bool(summary.get("moveit_smoothing_complete") is True and final_accepted == required),
        "first_segment": 0 if final_accepted == required else (final_accepted[0] if final_accepted else None),
        "last_segment": nominal_segments - 1 if final_accepted == required else (final_accepted[-1] if final_accepted else None),
        "missing_segments": [index for index in required if index not in final_accepted],
        "stale_accepted_calls": stale_indices,
        "stale_accepted_call_count": len(stale_indices),
        "stale_evidence_used_for_final_certification": False,
    }


def trajectory_audit(csv_path: Path) -> dict[str, Any]:
    if not csv_path.exists():
        return {"available": False, "segments_total": 0, "segments_checked": 0}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    values = {key: [float(row[key]) for row in rows] for key in rows[0]} if rows else {}
    def maximum(prefix: str) -> list[float]:
        return [max(abs(values[f"j{joint}_{prefix}"][index]) for index in range(len(rows))) for joint in range(1, 7)]
    q_pass = bool(rows and all(POSITION_LOWER[i] - 1e-12 <= min(values[f"j{i+1}_q"]) and max(values[f"j{i+1}_q"]) <= POSITION_UPPER[i] + 1e-12 for i in range(6)))
    dq = maximum("dq") if rows else [math.inf] * 6
    ddq = maximum("ddq") if rows else [math.inf] * 6
    jerk = maximum("jerk") if rows else [math.inf] * 6
    return {
        "available": True,
        "states_total": len(rows),
        "segments_total": max(0, len(rows) - 1),
        "q_hash": sha256(csv_path),
        "q_limits_passed": q_pass,
        "velocity_limits_passed": all(dq[i] <= VELOCITY_LIMITS[i] + 1e-9 for i in range(6)),
        "acceleration_limits_passed": all(ddq[i] <= ACCELERATION_LIMITS[i] + 1e-9 for i in range(6)),
        "jerk_limits_passed": all(jerk[i] <= REQUIRED_JERK + JERK_TOLERANCE for i in range(6)),
        "max_abs_velocity": dq,
        "max_abs_acceleration": ddq,
        "max_abs_exported_jerk": jerk,
        "final_time": values["t"][-1] if rows else None,
    }


def final_replay_audit(path: Path, nominal_segments: int, overshoot_check_applicable: bool = True) -> dict[str, Any]:
    records = list(read_jsonl(path))
    successful = [record for record in records if (record.get("ruckig_result") or {}).get("result_name") in {"Working", "Finished"}]
    available = [record for record in records if record.get("native_profiles_available") is True]
    jerk_pass = [record for record in available if float(record.get("max_abs_jerk", math.inf)) <= REQUIRED_JERK + JERK_TOLERANCE]
    raw_overshoot_pass = [record for record in available if (record.get("overshoot_at_selected_semantics") or {}).get("passed") is True]
    failed = [int(record.get("segment_index", -1)) for record in records if (record.get("ruckig_result") or {}).get("result_name") not in {"Working", "Finished"}]
    return {
        "segments_total": nominal_segments,
        "segments_checked": len(records),
        "working_or_finished": f"{len(successful)}/{nominal_segments}",
        "successful_native_results": len(successful),
        "native_profiles_available": f"{len(available)}/{nominal_segments}",
        "jerk_pass_J8": f"{len(jerk_pass)}/{nominal_segments}",
        "overshoot_check_applicable": overshoot_check_applicable,
        "raw_overshoot_pass": f"{len(raw_overshoot_pass)}/{nominal_segments}",
        "overshoot_pass": f"{len(raw_overshoot_pass)}/{nominal_segments}" if overshoot_check_applicable else "not_applicable_mitigate_overshoot_false",
        "max_abs_native_jerk": max((float(record.get("max_abs_jerk", 0.0)) for record in available), default=None),
        "failed_segments": failed,
        "duration_identity": {
            "equal_count": sum(record.get("duration_identity") == "equal" for record in records),
            "mismatch_count": sum(record.get("duration_identity") == "mismatch" for record in records),
            "unavailable_count": sum(record.get("duration_identity") == "unavailable" for record in records),
            "required_for_all_segments": False,
        },
    }


def audit_case(case_root: Path, nominal_segments: int, persist: bool = True) -> dict[str, Any]:
    native = case_root / "native"
    moveit = case_root / "moveit"
    calls_path = case_artifact_path(case_root, "native/stage25r_ruckig_calls.jsonl")
    resolutions_path = case_artifact_path(case_root, "native/stage25r_ruckig_call_resolutions.jsonl")
    resolutions = list(read_jsonl(resolutions_path))
    summary = read_json(native / "stage25r_native_run_summary.json", {}) or {}
    profiles = profile_stats(calls_path, resolutions) if calls_path.exists() else {
        "total_calls": 0,
        "successful_profile_calls": 0,
        "error_result_calls": 0,
        "native_profiles_available": 0,
        "accepted_native_profiles": {"profile_count": 0, "passed_J8": False, "max_abs_jerk": None, "worst_call": None, "worst_waypoint": None, "worst_joint": None, "worst_phase": None},
        "all_successful_calculate_profiles": {"profile_count": 0, "passed_J8": False, "max_abs_jerk": None, "worst_call": None, "worst_waypoint": None, "worst_attempt": None, "worst_joint": None, "worst_phase": None},
        "all_calculate_profiles": {"total_calls": 0, "successful_profile_calls": 0, "error_result_calls": 0, "native_profiles_available": 0, "passed_J8_for_every_available_profile": False},
        "all_available_native_profiles_pass_J8": False,
        "max_abs_jerk": None,
        "worst_call": None,
        "input_trace_hash": None,
        "native_profile_hash": None,
        "calculate_call_graph_hash": None,
        "waypoint4_retry_audit": [],
    }
    generation = generation_audit(native, resolutions, nominal_segments)
    final_csv = moveit / "stage25_ruckig_trajectory.csv"
    trajectory = trajectory_audit(final_csv)
    replay_path = case_artifact_path(case_root, "stage25r2_final_trajectory_native_replay.jsonl")
    manifest = read_json(case_root / "case_manifest.json", {}) or {}
    replay = final_replay_audit(replay_path, nominal_segments, bool(manifest.get("mitigate_overshoot", True))) if replay_path.exists() else None
    moveit_runtime = read_json(moveit / "stage25_native_moveit_runtime.json", {}) or {}
    ruckig_audit = read_json(moveit / "stage25_ruckig_audit.json", {}) or {}
    result = {
        "schema_version": "stage25r2-case-audit-v1",
        "case_root": str(case_root),
        "nominal_segments": nominal_segments,
        "semantics": {
            "mitigate_overshoot": read_json(case_root / "case_manifest.json", {}).get("mitigate_overshoot"),
            "overshoot_threshold": read_json(case_root / "case_manifest.json", {}).get("overshoot_threshold"),
            "overshoot_check_period": 0.01,
        },
        "moveit_runtime": {
            "native_return_value": summary.get("moveit_native_return_value"),
            "smoothing_complete": summary.get("moveit_smoothing_complete"),
            "strict_stage25r2_completion": summary.get("strict_stage25r2_completion"),
            "last_result": summary.get("moveit_last_ruckig_result"),
            "last_called_waypoint_idx": summary.get("last_called_waypoint_idx"),
            "duration_ceiling_hit": summary.get("duration_ceiling_hit"),
            "final_duration": trajectory.get("final_time"),
        },
        "call_summary": {
            "calculate_calls_total": profiles["total_calls"],
            "accepted_calls_total": sum(item.get("accepted_segment_call") is True for item in resolutions),
            "retry_calls_total": sum(item.get("retry_call") is True for item in resolutions),
            "unique_waypoints_ever_accepted": generation["unique_waypoints_ever_accepted"],
        },
        "native_jerk_certification": {
            "accepted_native_profiles": profiles["accepted_native_profiles"],
            "all_successful_calculate_profiles": profiles["all_successful_calculate_profiles"],
            "all_calculate_profiles": profiles["all_calculate_profiles"],
            "all_available_native_profiles_pass_J8": profiles["all_available_native_profiles_pass_J8"],
            "jerk_source": "native Profile.j and Profile.brake.j; no finite-difference jerk used",
        },
        "waypoint4_retry_audit": {
            "attempts": profiles["waypoint4_retry_audit"],
            "first_retry": profiles["waypoint4_retry_audit"][0] if profiles["waypoint4_retry_audit"] else None,
            "last_retry": profiles["waypoint4_retry_audit"][-1] if profiles["waypoint4_retry_audit"] else None,
            "maximum_extension_factor_after_retry": max((item.get("duration_extension_factor_after_retry") or 0.0 for item in profiles["waypoint4_retry_audit"]), default=None),
            "ceiling_crossing_call": next((item for item in profiles["waypoint4_retry_audit"] if float(item.get("duration_extension_factor_after_retry") or 0.0) > 50.0), None),
            "trigger_joints": sorted({item["trigger_joint"] for item in profiles["waypoint4_retry_audit"] if item.get("trigger_joint") is not None}),
            "ruckig_results": sorted({(item.get("ruckig_result") or {}).get("result_name") for item in profiles["waypoint4_retry_audit"]}),
            "overshoot_magnitudes_monotone_nonincreasing": all(
                float(b.get("worst_overshoot_magnitude_rad")) <= float(a.get("worst_overshoot_magnitude_rad")) + 1e-15
                for a, b in zip(profiles["waypoint4_retry_audit"], profiles["waypoint4_retry_audit"][1:])
                if a.get("worst_overshoot_magnitude_rad") is not None and b.get("worst_overshoot_magnitude_rad") is not None
            ),
        },
        "trajectory_generation": generation,
        "trajectory": trajectory,
        "independent_final_trajectory_certification": replay,
        "pose_constraints": (moveit_runtime.get("pose_validation") or {}),
        "original_moveit_ruckig_audit": ruckig_audit.get("audit"),
        "hashes": {
            "calculate_call_graph_hash": profiles["calculate_call_graph_hash"],
            "native_input_hash": profiles["input_trace_hash"],
            "native_profile_hash": profiles["native_profile_hash"],
            "resolution_trace_hash": sha256(resolutions_path),
            "overshoot_trace_hash": sha256(native / "stage25r2_overshoot_events.jsonl"),
            "trajectory_generation_trace_hash": sha256(native / "stage25r2_trajectory_generation_events.jsonl"),
            "final_robot_trajectory_hash": sha256(final_csv),
            "final_replay_hash": sha256(replay_path),
        },
    }
    if persist:
        (case_root / "stage25r2_case_audit.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (case_root / "stage25r2_waypoint4_retry_audit.json").write_text(json.dumps(result["waypoint4_retry_audit"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (case_root / "stage25r2_all_call_native_jerk_audit.json").write_text(json.dumps(result["native_jerk_certification"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--nominal-segments", type=int, default=25531)
    args = parser.parse_args()
    result = audit_case(args.case_root.resolve(), args.nominal_segments)
    print(json.dumps({"case_root": str(args.case_root.resolve()), "calls": result["call_summary"]["calculate_calls_total"], "full_coverage": result["trajectory_generation"]["all_nominal_segments_reached"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
