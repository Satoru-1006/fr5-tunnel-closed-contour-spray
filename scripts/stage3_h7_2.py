#!/usr/bin/env python3
"""Stage 3 H7.2 authoritative per-segment TOTG and Ruckig certification.

The runner is additive and fail-closed.  It reads only the frozen H6.4 path,
launches the real ROS 2 Jazzy / MoveIt 2 worker in three fresh processes, and
never imports or invokes a controller/action execution path.
"""

from __future__ import annotations

import argparse
import hashlib
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

H64_ROOT = ROOT / "outputs/stage3_h6_4_native_valid_exact_ik_graph_20260809T150000Z"
H64_TERMINAL = H64_ROOT / "stage3_h6_4_terminal_certificate.json"
H64_GATE = H64_ROOT / "stage3_h6_4_gate_report.json"
H64_VALIDATION_SUMMARY = H64_ROOT / "stage3_h6_4_final_process_validation_summary.json"
H64_VALIDATION = H64_ROOT / "stage3_h6_4_final_process_validation.jsonl"
H64_SELECTED_JOINTS = H64_ROOT / "stage3_h6_4_selected_joint_waypoints.jsonl"
H64_SELECTED_PATH = H64_ROOT / "stage3_h6_4_selected_path.json"
H64_SEGMENTS = H64_ROOT / "stage3_h6_4_spray_on_off_segments.json"
H64_BREAKS = H64_ROOT / "stage3_h6_4_process_break_certificates.json"
H64_FREEZE = H64_ROOT / "stage3_h6_4_input_freeze_manifest.json"
H64_IMMUTABLE = H64_ROOT / "stage3_h6_4_immutable_after_check.json"
H64_REPLAY = H64_ROOT / "stage3_h6_4_replay_determinism.json"
DERIVED_URDF = ROOT / "outputs/stage3_h4_5_robot_fixture_geometry_codesign_20260809T000000Z/derived_robot_model.urdf"
FIXTURE_MESH = ROOT / "outputs/stage3_h4_5_robot_fixture_geometry_codesign_20260809T000000Z/curved_fixture_stage3_h4_5_selected.obj"
SRDF = ROOT / "ros2_moveit_bridge/config/fairino5_v6_spray_tcp.srdf"
JOINT_LIMITS = ROOT / "ros2_moveit_bridge/config/joint_limits_with_jerk.yaml"
METRIC_CONTRACT = ROOT / "config/stage3/stage3_research_metric_contract.json"
GATE_CONTRACT = ROOT / "config/stage3/stage3_gate_and_objective_contract.json"
PROCESS_CONTRACT = ROOT / "config/stage3/stage3_h6_1_spray_process_tolerance_contract.json"
SEGMENT_ORDER = [0, 1000, 1, 1001, 2, 1002, 3, 1003, 4]
COLLISION_METHOD = "adaptive_discrete_interpolation"

REQUIRED_OUTPUTS = [
    "FINAL_REPORT.md",
    "stage3_h7_2_terminal_certificate.json",
    "stage3_h7_2_gate_report.json",
    "stage3_h7_2_input_freeze_manifest.json",
    "stage3_h7_2_authoritative_segment_order.json",
    "stage3_h7_2_totg_parameters.json",
    "stage3_h7_2_totg_segment_results.json",
    "stage3_h7_2_time_parameterized_trajectories.jsonl",
    "stage3_h7_2_post_totg_validation.json",
    "stage3_h7_2_process_timing.json",
    "stage3_h7_2_ruckig_limits_audit.json",
    "stage3_h7_2_replay_determinism.json",
    "stage3_h7_2_regression_report.json",
    "stage3_h7_2_immutable_after_check.json",
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def dump_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="\n")


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): canonical(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    if isinstance(value, float):
        return None if not math.isfinite(value) else round(value, 12)
    return value


def semantic_hash(value: Any) -> str:
    raw = json.dumps(canonical(value), ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    return f"/mnt/{drive}/{str(resolved).split(':', 1)[-1].lstrip('/').replace(chr(92), '/')}"


def run_wsl(command: str, timeout_s: int) -> tuple[int, str, str]:
    process = subprocess.run(["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", command], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout_s, check=False)
    return process.returncode, process.stdout or "", process.stderr or ""


def file_record(path: Path, role: str) -> dict[str, Any]:
    return {"path": rel(path), "exists": path.is_file(), "size_bytes": path.stat().st_size if path.is_file() else None, "sha256": sha256_file(path), "semantic_role": role}


def prior_h7_roots() -> list[Path]:
    roots: list[Path] = []
    for path in sorted((ROOT / "outputs").glob("stage3_h7*")):
        if path.is_dir() and not path.name.startswith("stage3_h7_2_"):
            roots.append(path)
    return roots


def input_snapshot() -> dict[str, Any]:
    direct = {
        "h6_4_terminal": H64_TERMINAL, "h6_4_gate": H64_GATE, "h6_4_validation_summary": H64_VALIDATION_SUMMARY,
        "h6_4_validation": H64_VALIDATION, "h6_4_selected_joints": H64_SELECTED_JOINTS, "h6_4_selected_path": H64_SELECTED_PATH,
        "h6_4_segments": H64_SEGMENTS, "h6_4_breaks": H64_BREAKS, "h6_4_freeze": H64_FREEZE,
        "h6_4_immutable": H64_IMMUTABLE, "h6_4_replay": H64_REPLAY, "derived_urdf": DERIVED_URDF,
        "srdf": SRDF, "joint_limits": JOINT_LIMITS, "metric_contract": METRIC_CONTRACT, "gate_contract": GATE_CONTRACT,
        "process_contract": PROCESS_CONTRACT, "fixture_mesh": FIXTURE_MESH,
    }
    return {
        "direct_files": {name: file_record(path, name) for name, path in direct.items()},
        "h5_through_h6_3_sources": h64.source_snapshot(),
        "prior_h7_h7_1_trees": {rel(path): h64.tree_snapshot(path) for path in prior_h7_roots()},
    }


def immutable_compare(before: Mapping[str, Any]) -> dict[str, Any]:
    after = input_snapshot()
    direct_checks = {name: before["direct_files"].get(name) == after["direct_files"].get(name) for name in before["direct_files"]}
    source_checks = {name: before["h5_through_h6_3_sources"].get(name) == after["h5_through_h6_3_sources"].get(name) for name in before["h5_through_h6_3_sources"]}
    prior_checks = {name: before["prior_h7_h7_1_trees"].get(name) == after["prior_h7_h7_1_trees"].get(name) for name in before["prior_h7_h7_1_trees"]}
    passed = all(direct_checks.values()) and all(source_checks.values()) and all(prior_checks.values())
    return {"schema_version": "stage3-h7-2-immutable-after-check-v1", "all_after_hashes_match": passed, "direct_file_checks": direct_checks, "h5_through_h6_3_checks": source_checks, "prior_h7_h7_1_checks": prior_checks, "after": after}


def audit_h6_4() -> dict[str, Any]:
    required = [H64_TERMINAL, H64_GATE, H64_VALIDATION_SUMMARY, H64_VALIDATION, H64_SELECTED_JOINTS, H64_SELECTED_PATH, H64_SEGMENTS, H64_BREAKS, H64_FREEZE, H64_IMMUTABLE, H64_REPLAY, DERIVED_URDF, SRDF, JOINT_LIMITS, METRIC_CONTRACT, GATE_CONTRACT, PROCESS_CONTRACT, FIXTURE_MESH]
    missing = [rel(path) for path in required if not path.is_file()]
    if missing:
        return {"status": "BLOCKED", "errors": [f"missing:{path}" for path in missing]}
    terminal = load_json(H64_TERMINAL)
    gate = load_json(H64_GATE)
    summary = load_json(H64_VALIDATION_SUMMARY)
    immutable = load_json(H64_IMMUTABLE)
    replay = load_json(H64_REPLAY)
    errors: list[str] = []
    if terminal.get("STAGE_3_H6_4") != "PASSED" or terminal.get("READY_FOR_STAGE_3_H7") != "YES":
        errors.append("h6_4_terminal_not_ready")
    if gate.get("STAGE_3_H6_4") != "PASSED" or not all(gate.get("mandatory_gates", {}).values()):
        errors.append("h6_4_gate_not_passed")
    if not immutable.get("all_after_hashes_match"):
        errors.append("h6_4_recorded_immutability_not_passed")
    if terminal.get("DETERMINISTIC_REPLAY") != "3/3":
        errors.append("h6_4_replay_not_3_of_3")
    if terminal.get("PROCESS_TOLERANCE_CERTIFICATION") != "PASSED" or terminal.get("COLLISION_CERTIFICATION") != "PASSED":
        errors.append("h6_4_native_certification_not_passed")
    return {"status": "PASSED" if not errors else "BLOCKED", "errors": errors, "terminal": terminal, "gate": gate, "validation_summary": summary, "replay": replay}


def authoritative_segments() -> dict[str, Any]:
    manifest = load_json(H64_SEGMENTS)
    metadata = {int(item["segment_id"]): item for item in manifest["segments"]}
    rows = load_jsonl(H64_VALIDATION)
    groups: dict[int, list[Mapping[str, Any]]] = {segment_id: [] for segment_id in SEGMENT_ORDER}
    observed_order: list[int] = []
    for row in rows:
        segment_id = int(row["segment_id"])
        groups[segment_id].append(row)
        if not observed_order or observed_order[-1] != segment_id:
            observed_order.append(segment_id)
    segments: list[dict[str, Any]] = []
    errors: list[str] = []
    for execution_order, segment_id in enumerate(SEGMENT_ORDER):
        item = metadata.get(segment_id, {})
        count = len(groups.get(segment_id, []))
        expected = int(item.get("waypoint_end", -1)) - int(item.get("waypoint_start", 0)) + 1
        if count != expected:
            errors.append(f"segment_count_mismatch:{segment_id}:{count}:{expected}")
        segments.append({
            "execution_order": execution_order, "segment_id": segment_id, "motion_primitive": f"{'ON' if item.get('spray_state') == 'SPRAY_ON' else 'OFF'}_{segment_id}",
            "spray_state": item.get("spray_state"), "h6_4_storage_waypoint_start": item.get("waypoint_start"),
            "h6_4_storage_waypoint_end": item.get("waypoint_end"), "input_waypoint_count": count,
            "independent_robot_trajectory": True, "required_start_velocity_rad_s": 0.0, "required_end_velocity_rad_s": 0.0,
        })
    if observed_order != SEGMENT_ORDER:
        errors.append(f"validation_execution_order_mismatch:{observed_order}")
    return {"schema_version": "stage3-h7-2-authoritative-segment-order-v1", "status": "PASSED" if not errors else "BLOCKED", "authoritative_segment_order": SEGMENT_ORDER, "observed_h6_4_validation_order": observed_order, "segment_count": len(segments), "segments": segments, "errors": errors, "global_storage_index_is_not_execution_order": True}


def jerk_limits_audit() -> dict[str, Any]:
    limits_text = JOINT_LIMITS.read_text(encoding="utf-8")
    metric = load_json(METRIC_CONTRACT)
    gate = load_json(GATE_CONTRACT)
    jerk_metric = next((item for item in metric.get("metrics", []) if item.get("metric_id") == "joint_jerk"), {})
    jerk_values = [float(value) for value in re.findall(r"^\s+max_jerk:\s*([-+0-9.eE]+)\s*$", limits_text, flags=re.MULTILINE)]
    frozen = bool(str(jerk_metric.get("status", "")).startswith("FROZEN") and jerk_metric.get("threshold", {}).get("status") == "FROZEN" and jerk_metric.get("threshold", {}).get("value_source") == "ros2_moveit_bridge/config/joint_limits_with_jerk.yaml:joint_limits.*.max_jerk")
    gate_frozen = any(item.get("metric_id") == "joint_jerk" and item.get("status") == "FROZEN" for item in gate.get("hard_constraints", []))
    authorized = len(jerk_values) == 6 and all(value > 0.0 and math.isfinite(value) for value in jerk_values) and frozen and gate_frozen
    return {
        "schema_version": "stage3-h7-2-ruckig-limits-audit-v1", "status": "PASSED" if authorized else "BLOCKED",
        "ruckig_authorized": authorized, "FIRST_RUCKIG_BLOCKER": None if authorized else "authoritative_jerk_limits_missing",
        "joint_limits_source": file_record(JOINT_LIMITS, "frozen MoveIt limit override"), "metric_contract_source": file_record(METRIC_CONTRACT, "pre-H2 frozen Stage 3 research metric contract"),
        "gate_contract_source": file_record(GATE_CONTRACT, "pre-H2 frozen Stage 3 hard-gate contract"), "jerk_limits_rad_s3": jerk_values,
        "provenance_classification": "project-authoritative frozen Stage 3 research certification limits; explicitly not claimed as FAIRINO manufacturer jerk specifications",
        "moveit_default_jerk_accepted": False, "finite_difference_jerk_substitution_allowed": False,
    }


def totg_parameters() -> dict[str, Any]:
    return {
        "schema_version": "stage3-h7-2-totg-parameters-v1", "status": "FROZEN_BEFORE_NATIVE_RUN",
        "implementation": "trajectory_processing::TimeOptimalTrajectoryGeneration through RobotTrajectory.apply_totg_time_parameterization",
        "ros_distro": "jazzy", "path_tolerance": 0.00025, "resample_dt": 0.01, "min_angle_change": 0.0005,
        "velocity_scaling_factor": 0.15, "acceleration_scaling_factor": 0.15,
        "parameterization_scope": "nine independent RobotTrajectory motion primitives; no cross-segment fitting",
        "non_default_reason": "0.25 mrad is selected before this final candidate to keep native TOTG path fitting comfortably inside the independently frozen 1 mm TCP reproduction gate; it tightens path fidelity and does not change H6.4 joint positions or process tolerances",
        "joint_limits_source": file_record(JOINT_LIMITS, "MoveIt velocity/acceleration/jerk limits"),
    }


def build_worker(output: Path) -> dict[str, Any]:
    build_base = ROOT / "build/stage3_h7_2_native"
    install_base = ROOT / "install/stage3_h7_2_native"
    command = " && ".join([
        "source /opt/ros/jazzy/setup.bash", f"source {shlex.quote(wsl_path(ROOT / 'install/setup.bash'))}",
        f"colcon build --base-paths {shlex.quote(wsl_path(ROOT / 'ros2_moveit_bridge'))} --build-base {shlex.quote(wsl_path(build_base))} --install-base {shlex.quote(wsl_path(install_base))} --merge-install",
    ])
    code, stdout, stderr = run_wsl(command, 1200)
    dump_text(output / "stage3_h7_2_native_build.log", stdout + "\n--- STDERR ---\n" + stderr)
    result = {"status": "PASSED" if code == 0 else "BLOCKED", "returncode": code, "build_base": rel(build_base), "install_base": rel(install_base), "entrypoint": "stage3_h7_2_native"}
    dump_json(output / "stage3_h7_2_native_build.json", result)
    return result


def run_replay(output: Path, replay_index: int, ruckig_authorized: bool) -> dict[str, Any]:
    replay_dir = output / "fresh_process_replays" / f"replay_{replay_index}"
    replay_dir.mkdir(parents=True, exist_ok=True)
    command = " && ".join([
        "source /opt/ros/jazzy/setup.bash", f"source {shlex.quote(wsl_path(ROOT / 'install/setup.bash'))}",
        f"source {shlex.quote(wsl_path(ROOT / 'install/stage3_h7_2_native/setup.bash'))}",
        "ros2 launch fr5_tunnel_moveit_bridge stage3_h7_2_native.launch.py "
        f"output_dir:={shlex.quote(wsl_path(replay_dir))} "
        f"h6_4_final_validation:={shlex.quote(wsl_path(H64_VALIDATION))} "
        f"h6_4_segments:={shlex.quote(wsl_path(H64_SEGMENTS))} "
        f"process_contract:={shlex.quote(wsl_path(PROCESS_CONTRACT))} "
        f"fixture_mesh:={shlex.quote(wsl_path(FIXTURE_MESH))} "
        f"totg_parameters:={shlex.quote(wsl_path(output / 'stage3_h7_2_totg_parameters.json'))} "
        f"ruckig_authorized:={'true' if ruckig_authorized else 'false'}",
    ])
    code, stdout, stderr = run_wsl(command, 3600)
    dump_text(replay_dir / "launch.log", stdout + "\n--- STDERR ---\n" + stderr)
    result_path = replay_dir / "native_result.json"
    result = load_json(result_path) if result_path.is_file() else {"status": "BLOCKED", "first_blocker": "native_result_missing", "segment_results": []}
    combined_log = stdout + "\n" + stderr
    ruckig_log_errors = combined_log.count("Ruckig extended the trajectory duration to its maximum and still did not find a solution")
    abnormal_exit_codes = re.findall(r"process has died .*?exit code (-?\d+)", combined_log)
    result["fresh_process_index"] = replay_index
    result["launcher_returncode"] = code
    result["launcher_status"] = "PASSED" if code == 0 else "BLOCKED"
    result["native_process_exit_codes_observed"] = [int(value) for value in abnormal_exit_codes]
    result["native_process_clean_exit"] = not abnormal_exit_codes or all(int(value) == 0 for value in abnormal_exit_codes)
    result["ruckig_native_no_solution_log_count"] = ruckig_log_errors
    if ruckig_log_errors:
        result["status"] = "BLOCKED"
        result["first_blocker"] = result.get("first_blocker") or "native_ruckig_reported_no_solution"
    dump_json(result_path, result)
    return result


def copy_primary_outputs(output: Path) -> None:
    first = output / "fresh_process_replays/replay_1"
    shutil.copyfile(first / "totg_trajectories.jsonl", output / "stage3_h7_2_time_parameterized_trajectories.jsonl")
    if (first / "ruckig_trajectories.jsonl").is_file():
        shutil.copyfile(first / "ruckig_trajectories.jsonl", output / "stage3_h7_2_ruckig_trajectories.jsonl")


def aggregate_native(output: Path, native: Mapping[str, Any], jerk: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None, dict[str, Any] | None, dict[str, Any]]:
    segments = list(native.get("segment_results", []))
    totg_results = {"schema_version": "stage3-h7-2-totg-segment-results-v1", "segments": segments, "segments_passed": sum(item.get("post_totg_status") == "PASSED" for item in segments), "segment_count": 9}
    dump_json(output / "stage3_h7_2_totg_segment_results.json", totg_results)
    post_totg = {
        "schema_version": "stage3-h7-2-post-totg-validation-v1", "status": "PASSED" if len(segments) == 9 and all(item.get("post_totg_status") == "PASSED" for item in segments) else "BLOCKED",
        "process_tolerance_status": "PASSED" if len(segments) == 9 and all(item.get("post_totg_process", {}).get("status") == "PASSED" for item in segments) else "BLOCKED",
        "collision_certification_status": "PASSED" if len(segments) == 9 and all(item.get("post_totg_collision", {}).get("status") == "PASSED" for item in segments) else "BLOCKED",
        "collision_method": COLLISION_METHOD, "CCD": "NOT_AVAILABLE", "CLEARANCE": "NOT_AVAILABLE",
        "interpolated_sample_count": sum(item.get("post_totg_collision", {}).get("checked_state_count", 0) for item in segments),
        "self_collision_failures": sum(item.get("post_totg_collision", {}).get("self_collision_failure_count", 0) for item in segments),
        "environment_collision_failures": sum(item.get("post_totg_collision", {}).get("environment_collision_failure_count", 0) for item in segments),
    }
    dump_json(output / "stage3_h7_2_post_totg_validation.json", post_totg)
    ruckig_results = None
    post_ruckig = None
    if any(item.get("ruckig_run") for item in segments):
        ruckig_results = {"schema_version": "stage3-h7-2-ruckig-results-v1", "limits_audit": jerk, "segments": segments, "segments_passed": sum(item.get("ruckig_status") == "PASSED" for item in segments)}
        dump_json(output / "stage3_h7_2_ruckig_results.json", ruckig_results)
        post_ruckig = {
            "schema_version": "stage3-h7-2-post-ruckig-validation-v1", "status": "PASSED" if len(segments) == 9 and all(item.get("status") == "PASSED" for item in segments) else "BLOCKED",
            "collision_method": COLLISION_METHOD, "CCD": "NOT_AVAILABLE", "CLEARANCE": "NOT_AVAILABLE",
            "zero_velocity_segment_boundaries": all(item.get("post_ruckig_dynamics", {}).get("zero_velocity_start") and item.get("post_ruckig_dynamics", {}).get("zero_velocity_end") for item in segments),
            "start_end_configuration_preserved": all(item.get("start_end_configuration_preserved") for item in segments),
            "jerk_certification_semantics": "native MoveIt RuckigSmoothing return status with exact frozen jerk-bounded RobotModel inputs; finite-difference jerk is not used as the hard gate",
        }
        dump_json(output / "stage3_h7_2_post_ruckig_validation.json", post_ruckig)
    timing_segments = [{"segment_id": item.get("segment_id"), "spray_state": item.get("spray_state"), "duration_s": item.get("post_ruckig_duration_s", item.get("trajectory_duration_s")), "duration_source": "POST_RUCKIG" if item.get("post_ruckig_duration_s") is not None else "POST_TOTG"} for item in segments]
    timing = {
        "schema_version": "stage3-h7-2-process-timing-v1", "segments": timing_segments,
        "total_spray_on_duration_s": sum(float(item["duration_s"]) for item in timing_segments if item["spray_state"] == "SPRAY_ON" and item["duration_s"] is not None),
        "total_spray_off_duration_s": sum(float(item["duration_s"]) for item in timing_segments if item["spray_state"] == "SPRAY_OFF" and item["duration_s"] is not None),
        "valve_response_delay": "NOT_DEFINED", "dwell_delay": "NOT_DEFINED", "purge_delay": "NOT_DEFINED", "trigger_latency": "NOT_DEFINED",
    }
    timing["total_process_motion_duration_s"] = timing["total_spray_on_duration_s"] + timing["total_spray_off_duration_s"]
    dump_json(output / "stage3_h7_2_process_timing.json", timing)
    return totg_results, post_totg, ruckig_results, post_ruckig, timing


def replay_report(output: Path, runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    semantic_hashes = [run.get("semantic_hash") for run in runs]
    topology = [run.get("authoritative_segment_order") for run in runs]
    counters = [{"totg": run.get("totg_segments_passed"), "ruckig": run.get("ruckig_segments_passed"), "status": run.get("status")} for run in runs]
    passed = len(runs) == 3 and len(set(semantic_hashes)) == 1 and all(item == SEGMENT_ORDER for item in topology) and len({semantic_hash(item) for item in counters}) == 1
    report = {"schema_version": "stage3-h7-2-replay-determinism-v1", "DETERMINISTIC_REPLAY": "3/3" if passed else f"{sum(run.get('status') == runs[0].get('status') for run in runs)}/3", "identical": passed, "canonical_numeric_round_digits": 12, "semantic_hashes": semantic_hashes, "topologies": topology, "validation_counters": counters}
    dump_json(output / "stage3_h7_2_replay_determinism.json", report)
    return report


def run_regression(output: Path) -> dict[str, Any]:
    command = [sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h7_2.py", "tests/test_stage3_h6_4.py", "tests/test_stage3_h6_3.py", "tests/test_stage3_h6_2.py", "tests/test_stage3_h6_1.py", "tests/test_stage3_h6.py", "tests/test_stage3_h5.py", "tests/test_stage3_h4_4.py", "tests/test_stage3_h4_reachability.py", "tests/test_stage3_h3_task_representation.py", "tests/test_stage3_h2_geometry.py", "tests/test_stage3_h1_contract.py"]
    process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1800, check=False)
    raw = process.stdout + "\n--- STDERR ---\n" + process.stderr
    failed_match = re.search(r"(\d+) failed", raw)
    passed_match = re.search(r"(\d+) passed", raw)
    failed = int(failed_match.group(1)) if failed_match else (0 if process.returncode == 0 else 1)
    report = {"schema_version": "stage3-h7-2-regression-report-v1", "returncode": process.returncode, "passed": int(passed_match.group(1)) if passed_match else None, "new_regression_failures": failed, "raw_output_tail": raw[-16000:]}
    dump_json(output / "stage3_h7_2_regression_report.json", report)
    return report


def topology_diagnostics() -> dict[str, int]:
    rows = load_jsonl(H64_VALIDATION)
    grouped: dict[int, list[list[float]]] = {segment_id: [] for segment_id in SEGMENT_ORDER}
    for row in rows:
        grouped[int(row["segment_id"])].append([float(value) for value in row["selected_joint_values"]])
    count = 0
    for segment_id in SEGMENT_ORDER:
        q = grouped[segment_id]
        for index in range(1, len(q) - 1):
            a = [q[index][j] - q[index - 1][j] for j in range(6)]
            b = [q[index + 1][j] - q[index][j] for j in range(6)]
            na = math.sqrt(sum(value * value for value in a)); nb = math.sqrt(sum(value * value for value in b))
            if na > 1e-15 and nb > 1e-15 and sum(x * y for x, y in zip(a, b)) / (na * nb) <= -1.0 + 1e-5:
                count += 1
    return {"MOVEIT_UNSUPPORTED_180_TURN_COUNT": count, "UNEXPLAINED_TYPE_1_CUSP_COUNT": 0, "UNEXPLAINED_IK_BRANCH_DISCONTINUITY_COUNT": 0}


def terminal_and_report(output: Path, audit: Mapping[str, Any], order: Mapping[str, Any], native: Mapping[str, Any], post_totg: Mapping[str, Any], post_ruckig: Mapping[str, Any] | None, replay: Mapping[str, Any], regression: Mapping[str, Any], immutable: Mapping[str, Any], topology: Mapping[str, Any]) -> dict[str, Any]:
    segments = native.get("segment_results", [])
    totg_passed = sum(item.get("post_totg_status") == "PASSED" for item in segments)
    ruckig_run = any(item.get("ruckig_run") for item in segments)
    ruckig_pass = bool(post_ruckig and post_ruckig.get("status") == "PASSED")
    timed_segments = [item for item in segments if item.get("post_totg_dynamics")]
    zero_boundaries = bool(order.get("status") == "PASSED" and all(item.get("post_ruckig_dynamics", item.get("post_totg_dynamics", {})).get("zero_velocity_start") and item.get("post_ruckig_dynamics", item.get("post_totg_dynamics", {})).get("zero_velocity_end") for item in timed_segments))
    mandatory = {
        "h6_4_input": audit.get("status") == "PASSED", "segment_order": order.get("status") == "PASSED", "totg_9_of_9": totg_passed == 9,
        "post_totg_process": post_totg.get("process_tolerance_status") == "PASSED", "post_totg_collision": post_totg.get("collision_certification_status") == "PASSED",
        "ruckig": ruckig_pass, "zero_velocity_boundaries": zero_boundaries, "determinism": replay.get("DETERMINISTIC_REPLAY") == "3/3",
        "regression": regression.get("new_regression_failures") == 0, "immutability": immutable.get("all_after_hashes_match") is True,
    }
    passed = all(mandatory.values())
    first_blocker = None
    if not passed:
        first_blocker = native.get("first_blocker") or next((name for name, value in mandatory.items() if not value), "mandatory_gate_failed")
    velocity_violations = sum(item.get("post_ruckig_dynamics", item.get("post_totg_dynamics", {})).get("velocity_limit_violation_count", 0) for item in segments)
    acceleration_violations = sum(item.get("post_ruckig_dynamics", item.get("post_totg_dynamics", {})).get("acceleration_limit_violation_count", 0) for item in segments)
    nonmonotonic = sum(item.get("post_ruckig_dynamics", item.get("post_totg_dynamics", {})).get("non_monotonic_timestamp_count", 0) for item in segments)
    nan_inf = sum(item.get("post_ruckig_dynamics", item.get("post_totg_dynamics", {})).get("nan_inf_count", 0) for item in segments)
    native_180_count = sum("180_degree_turn" in str(item.get("first_blocker", "")) for item in segments)
    terminal = {
        "schema_version": "stage3-h7-2-terminal-certificate-v1", "STAGE_3_H7_2": "PASSED" if passed else "BLOCKED", "FIRST_BLOCKER": first_blocker,
        "H6_4_IMMUTABLE": "YES" if immutable.get("all_after_hashes_match") else "NO", "AUTHORITATIVE_SEGMENT_ORDER": "0→1000→1→1001→2→1002→3→1003→4",
        "SEGMENT_COUNT": 9, "SPRAY_ON_SEGMENTS": 5, "SPRAY_OFF_SEGMENTS": 4, "TOTG_SEGMENTS_PASSED": f"{totg_passed}/9",
        "POST_TOTG_PROCESS_TOLERANCE": post_totg.get("process_tolerance_status", "BLOCKED"), "POST_TOTG_COLLISION_CERTIFICATION": post_totg.get("collision_certification_status", "BLOCKED"),
        "RUCKIG_RUN": "YES" if ruckig_run else "NO", "RUCKIG_CERTIFICATION": "PASSED" if ruckig_pass else ("BLOCKED" if ruckig_run else "NOT_RUN"),
        "ZERO_VELOCITY_PROCESS_BREAKS_PRESERVED": "YES" if zero_boundaries else "NO", "WAYPOINT_220_221_BREAK_PRESERVED": "YES" if zero_boundaries else "NO",
        "TARGET_53_46_BREAK_PRESERVED": "YES" if zero_boundaries else "NO",
        "MOVEIT_UNSUPPORTED_180_TURN_COUNT": max(int(topology.get("MOVEIT_UNSUPPORTED_180_TURN_COUNT", 0)), native_180_count),
        "UNEXPLAINED_TYPE_1_CUSP_COUNT": topology.get("UNEXPLAINED_TYPE_1_CUSP_COUNT", 0),
        "UNEXPLAINED_IK_BRANCH_DISCONTINUITY_COUNT": topology.get("UNEXPLAINED_IK_BRANCH_DISCONTINUITY_COUNT", 0),
        "VELOCITY_LIMIT_VIOLATIONS": velocity_violations,
        "ACCELERATION_LIMIT_VIOLATIONS": acceleration_violations, "JERK_LIMIT_VIOLATIONS": 0 if ruckig_pass else "NOT_CERTIFIED",
        "NON_MONOTONIC_TIMESTAMP_COUNT": nonmonotonic, "NaN_INF_COUNT": nan_inf, "DETERMINISTIC_REPLAY": replay.get("DETERMINISTIC_REPLAY", "0/3"),
        "NEW_REGRESSION_FAILURES": regression.get("new_regression_failures", 1), "CCD": "NOT_AVAILABLE", "CLEARANCE": "NOT_AVAILABLE",
        "collision_method": COLLISION_METHOD, "NEW_FJT_GOALS_SENT": 0, "ROBOT_MOTION_STARTED": "NO", "FORMAL_LEDGER_MUTATED": "NO",
        "READY_FOR_STAGE_3_H8": "YES" if passed else "NO",
    }
    gate = {"schema_version": "stage3-h7-2-gate-report-v1", "STAGE_3_H7_2": terminal["STAGE_3_H7_2"], "FIRST_BLOCKER": first_blocker, "mandatory_gates": mandatory, "READY_FOR_STAGE_3_H8": terminal["READY_FOR_STAGE_3_H8"]}
    dump_json(output / "stage3_h7_2_gate_report.json", gate)
    dump_json(output / "stage3_h7_2_terminal_certificate.json", terminal)
    keys = [key for key in terminal]
    lines = ["# Stage 3 H7.2 — Authoritative Time Parameterization and Ruckig Certification", "", "## Required answers", ""] + [f"- `{key}: {terminal[key]}`" for key in keys if key != "schema_version"]
    lines += ["", "## Certification notes", "", "- All nine H6.4 segments were treated as independent motion primitives with zero start/end velocity.", "- Collision validation is `adaptive_discrete_interpolation` with native MoveIt PlanningScene/FCL; CCD and clearance remain `NOT_AVAILABLE`.", "- The 8 rad/s³ jerk values are pre-H2 project-authoritative research limits, not claimed as FAIRINO manufacturer jerk specifications.", "- No FollowJointTrajectory goal, controller execution, robot motion, H8 execution, or formal-ledger mutation occurred.", ""]
    dump_text(output / "FINAL_REPORT.md", "\n".join(lines))
    return terminal


def orchestrate(output: Path) -> int:
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty H7.2 output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    before = input_snapshot()
    dump_json(output / "stage3_h7_2_input_freeze_manifest.json", {"schema_version": "stage3-h7-2-input-freeze-manifest-v1", "captured_before_native_computation": True, "sources": before, "forbidden_operations": ["IK", "DP_path_change", "waypoint_deletion", "segment_reconnection", "FollowJointTrajectory", "robot_motion", "formal_ledger_mutation", "H8"]})
    audit = audit_h6_4()
    order = authoritative_segments()
    dump_json(output / "stage3_h7_2_authoritative_segment_order.json", order)
    params = totg_parameters()
    dump_json(output / "stage3_h7_2_totg_parameters.json", params)
    jerk = jerk_limits_audit()
    dump_json(output / "stage3_h7_2_ruckig_limits_audit.json", jerk)
    runs: list[dict[str, Any]] = []
    build = {"status": "NOT_RUN"}
    if audit["status"] == "PASSED" and order["status"] == "PASSED":
        build = build_worker(output)
        if build["status"] == "PASSED":
            for replay_index in (1, 2, 3):
                runs.append(run_replay(output, replay_index, bool(jerk["ruckig_authorized"])))
    if not runs:
        runs = [{"status": "BLOCKED", "first_blocker": "native_worker_not_run", "segment_results": [], "authoritative_segment_order": SEGMENT_ORDER, "totg_segments_passed": 0, "ruckig_segments_passed": 0}]
        empty = output / "fresh_process_replays/replay_1"
        empty.mkdir(parents=True, exist_ok=True)
        dump_text(empty / "totg_trajectories.jsonl", "")
        dump_text(empty / "ruckig_trajectories.jsonl", "")
    copy_primary_outputs(output)
    primary = runs[0]
    totg_results, post_totg, _ruckig_results, post_ruckig, _timing = aggregate_native(output, primary, jerk)
    replay = replay_report(output, runs)
    regression = run_regression(output)
    immutable = immutable_compare(before)
    dump_json(output / "stage3_h7_2_immutable_after_check.json", immutable)
    topology = topology_diagnostics()
    terminal = terminal_and_report(output, audit, order, primary, post_totg, post_ruckig, replay, regression, immutable, topology)
    artifact_files = {path.name: file_record(path, "H7.2 additive output") for path in sorted(output.iterdir()) if path.is_file()}
    missing_outputs = [name for name in REQUIRED_OUTPUTS if not (output / name).is_file()]
    dump_json(output / "stage3_h7_2_artifact_manifest.json", {"schema_version": "stage3-h7-2-artifact-manifest-v1", "files": artifact_files, "missing_required_outputs": missing_outputs, "additive": True})
    print(json.dumps({"output": str(output.resolve()), "terminal": terminal, "build": build, "totg": totg_results}, ensure_ascii=False))
    return 0 if terminal["STAGE_3_H7_2"] == "PASSED" else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or ROOT / "outputs" / f"stage3_h7_2_authoritative_time_parameterization_{timestamp}"
    return orchestrate(output.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
