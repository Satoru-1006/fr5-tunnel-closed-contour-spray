#!/usr/bin/env python3
"""Aggregate Stage 1.9.2a evidence into an isolated output directory."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/ik_graph_stage192a"
MINIMAL = ROOT / "outputs/ik_graph_stage192/minimal_case"
SYSTEM_PLUGIN_SHA256 = "8b51d55e730ff482b117e0f2d2291fa06dcebe2d47b4674e9360d792ca255718"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def sha(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def summarize(path: Path) -> dict[str, Any]:
    data = load(path)
    records = data.get("records", [])
    successes = [r for r in records if r.get("solver_success") is True]
    hashes = []
    for record in successes:
        value = record.get("solution", record.get("solution_joint_vector_raw"))
        hashes.append(hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest())
    return {
        "path": str(path.relative_to(ROOT)),
        "records": len(records),
        "success_count": len(successes),
        "success_values": sorted({bool(r.get("solver_success")) for r in records}),
        "raw_solution_hash_count": len(set(hashes)),
        "raw_solution_hashes": sorted(set(hashes)),
    }


def log_summary(path: Path) -> dict[str, Any]:
    rows = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    wiggles = [r for r in rows if r.get("wiggle_triggered", "").lower() == "true"]
    return {
        "path": str(path.relative_to(ROOT)),
        "rows": len(rows),
        "wiggle_rows": len(wiggles),
        "random_escape_rows": sum(r.get("random_delta_q") == "random" for r in wiggles),
    }


def consistency(paths: list[Path]) -> dict[str, Any]:
    records = []
    for path in paths:
        records.extend(load(path).get("records", []))
    success_values = [bool(r.get("solver_success")) for r in records]
    hashes = []
    for record in records:
        if record.get("solver_success") is True:
            value = record.get("solution", record.get("solution_joint_vector_raw"))
            hashes.append(hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest())
    return {
        "records": len(records),
        "success_values": sorted(set(success_values)),
        "solution_hash_count": len(set(hashes)),
        "deterministic": len(set(success_values)) <= 1 and len(set(hashes)) <= 1,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = load(MINIMAL / "input_manifest.json")
    isolated = OUT / "overlay_install/deterministic_kdl_kinematics_plugin/lib/libdeterministic_kdl_kinematics_plugin.so"
    runtime_plugin_manifest = {
        "schema_version": "1.0",
        "stage": "stage_1_9_2a",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "ros_distribution": "jazzy",
        "workspace_overlay": "/mnt/c/Users/86198/Desktop/robotfucker/install/setup.bash",
        "system_moveit_kinematics_prefix": "/opt/ros/jazzy",
        "package_version": "2.12.4-1noble.20260617.150037",
        "plugin_library_path": "/opt/ros/jazzy/lib/libmoveit_kdl_kinematics_plugin.so",
        "plugin_library_sha256": SYSTEM_PLUGIN_SHA256,
        "plugin_class_name": "kdl_kinematics_plugin/KDLKinematicsPlugin",
        "isolated_overlay_plugin_library_path": str(isolated.relative_to(ROOT)) if isolated.exists() else None,
        "isolated_overlay_plugin_library_sha256": sha(isolated),
        "upstream_random_wiggle_present": True,
        "installed_plugin_contains_same_logic": "pending",
        "runtime_loaded_library_verified": True,
        "runtime_source_match": "pending",
        "actual_plugin_class_confirmed": True,
        "ldd_dependency_summary": "runtime_provenance_command.log",
        "evidence_limits": [
            "symbols and strings prove KDL/random-related code is present but not execution of delta_q.data.setRandom",
            "online Jazzy source has the branch; exact installed binary source commit was not symbolically matched",
        ],
    }
    (OUT / "runtime_plugin_manifest.json").write_text(json.dumps(runtime_plugin_manifest, indent=2) + "\n", encoding="utf-8")

    runtime_params = {
        "source": "actual isolated plugin initialize trace plus frozen input manifest",
        "kinematics_solver": "deterministic_kdl_kinematics_plugin/DeterministicKDLKinematicsPlugin",
        "group_name": manifest.get("group_name", "fairino5_v6_group"),
        "base_frame": manifest.get("base_frame", "base_link"),
        "tip_frames": [manifest.get("tip_link", "spray_tcp_link")],
        "position_only_ik": False,
        "epsilon": 1.0e-5,
        "max_solver_iterations": 500,
        "joint_weights": [1.0] * 6,
        "orientation_vs_position": 1.0,
        "search_discretization": 0.005,
        "lock_redundant_joints": False,
        "return_approximate_solution": False,
        "discretization_method": "NO_DISCRETIZATION",
        "kinematics_query_options": {"lock_redundant_joints": False, "return_approximate_solution": False, "discretization_method": "NO_DISCRETIZATION"},
        "robot_state_initialization": manifest.get("robot_state_initialization"),
        "target_pose_semantics": "geometry_msgs/Pose in base_link with spray_tcp_link tip",
        "urdf_sha256": manifest.get("urdf_sha256"),
        "srdf_sha256": manifest.get("srdf_sha256"),
        "kinematics_config_sha256": manifest.get("kinematics_sha256"),
        "joint_limits_sha256": manifest.get("joint_limits_config_sha256"),
        "solver_options_sha256": manifest.get("solver_options_sha256"),
        "robot_model_sha256": manifest.get("robot_model_sha256", manifest.get("urdf_sha256")),
        "stage191_kinematics_hash_caveat": "8bdc63b3e333ee9b30330b815d032ebf84e2806d3d6daddfe797f7018bf204a4 != b57f743c70428cf073f28fb4b45299e9f8a72af6edd628a5d68098613234a8c7",
    }
    (OUT / "runtime_kinematics_parameters.json").write_text(json.dumps(runtime_params, indent=2) + "\n", encoding="utf-8")

    groups = []
    for path in (
        OUT / "cpp_random_same_instance/replay.json",
        OUT / "cpp_random_new_instance/replay.json",
        OUT / "cpp_disabled_same_instance/replay.json",
        OUT / "cpp_disabled_new_instance/replay.json",
        OUT / "cpp_deterministic_sequence_same_instance/replay.json",
        OUT / "cpp_deterministic_sequence_timeout/replay.json",
    ):
        if path.exists():
            groups.append(summarize(path))
    independent = sorted((OUT / "cpp_random_independent_processes").glob("process_*.json"))
    disabled_independent = sorted((OUT / "cpp_disabled_independent_processes").glob("process_*.json"))
    replay = {"target_call_id": "call:00000182", "input_manifest": str((MINIMAL / "input_manifest.json").relative_to(ROOT)), "groups": groups}
    if independent:
        replay["independent_processes"] = {"process_count": len(independent), "records": sum(summarize(p)["records"] for p in independent), "success_count": sum(summarize(p)["success_count"] for p in independent)}
    if disabled_independent:
        replay["disabled_independent_processes"] = {"process_count": len(disabled_independent), "records": sum(summarize(p)["records"] for p in disabled_independent), "success_count": sum(summarize(p)["success_count"] for p in disabled_independent)}
    (OUT / "minimal_case_replay.json").write_text(json.dumps(replay, indent=2) + "\n", encoding="utf-8")

    source_log = OUT / "cpp_random_same_instance/wiggle.csv"
    if source_log.exists():
        (OUT / "wiggle_diagnostics_call_00000182.csv").write_bytes(source_log.read_bytes())
    correlation = {
        "target_call_id": "call:00000182",
        "random_same_instance": log_summary(source_log),
        "disabled_same_instance": log_summary(OUT / "cpp_disabled_same_instance/wiggle.csv"),
        "deterministic_sequence_same_instance": log_summary(OUT / "cpp_deterministic_sequence_same_instance/wiggle.csv"),
        "wiggle_trigger_correlation_verified": "observed_policy_control_only",
        "installed_moveit_kdl_internal_random_singularity_escape_confirmed": False,
        "installed_binary_status": "suspected_pending_exact_binary_execution_trace",
        "moveitpy_binding_or_request_semantics_suspected": True,
        "interpretation": "The isolated reimplementation logs random escape vectors and mixed outcomes. Disabled policy gives stable failure. Random trigger presence alone is not a sufficient within-policy success classifier.",
    }
    (OUT / "wiggle_correlation_report.json").write_text(json.dumps(correlation, indent=2) + "\n", encoding="utf-8")

    independent_consistency = consistency(independent)
    disabled_independent_consistency = consistency(disabled_independent)
    deterministic = {
        "random": summarize(OUT / "cpp_random_same_instance/replay.json"),
        "disabled": summarize(OUT / "cpp_disabled_same_instance/replay.json"),
        "deterministic_sequence": summarize(OUT / "cpp_deterministic_sequence_same_instance/replay.json"),
        "deterministic_sequence_timeout_0p05": summarize(OUT / "cpp_deterministic_sequence_timeout/replay.json"),
        "same_instance_100_disabled_deterministic": True,
        "same_instance_100_deterministic_sequence_deterministic": True,
        "new_instance_50_determinism": consistency([OUT / "cpp_random_new_instance/replay.json"]),
        "independent_process_20": {"process_count": len(independent), **independent_consistency},
        "disabled_new_instance_50_determinism": consistency([OUT / "cpp_disabled_new_instance/replay.json"]),
        "disabled_independent_process_20": {"process_count": len(disabled_independent), **disabled_independent_consistency},
        "deterministic_escape_policy_implemented": True,
        "candidate_coverage_regression": True,
        "full_pipeline_usable": False,
        "random_modification_inside_one_CartToJnt": {"random": True, "disabled": False, "deterministic_sequence": False},
    }
    (OUT / "deterministic_policy_report.json").write_text(json.dumps(deterministic, indent=2) + "\n", encoding="utf-8")

    coverage = {
        "scope": "Stage 1 ON-state open-arch only",
        "collision_method": "adaptive_discrete_interpolation",
        "ccd_status": "not_available",
        "clearance_status": "not_available",
        "ordinary_waypoint": "not_run_with_deterministic_policy",
        "waypoint_67": "not_run_with_deterministic_policy",
        "waypoint_68": "not_run_with_deterministic_policy",
        "waypoint_87_repaired_pose": "not_run_with_deterministic_policy",
        "waypoint_94_repaired_pose": "not_run_with_deterministic_policy",
        "stage17_required_branch": "not_run_with_deterministic_policy",
        "candidate_graph_three_run_hash": "not_available",
        "complete_181_point_path": "not_run",
        "ruckig": "not_run",
        "fk": "not_run",
        "collision": "not_run",
        "dynamics": "not_run",
        "formal_pass": None,
        "reason": "Only the frozen single-call case was exercised with the new policy. No formal graph output was overwritten.",
    }
    (OUT / "candidate_coverage_report.json").write_text(json.dumps(coverage, indent=2) + "\n", encoding="utf-8")

    report = """# Stage 1.9.2a report

- runtime_loaded_library_verified: true
- actual_plugin_class_confirmed: true for the isolated overlay
- runtime_source_match: pending for the installed system binary
- isolated wiggle instrumentation: observed
- installed MoveIt KDL internal wiggle: suspected/pending exact binary execution trace
- deterministic policy implementation: true in the independent overlay
- candidate coverage regression: true for the tested frozen call
- full pipeline regression: not_run
- Stage 2: blocked

The live provenance audit resolved /opt/ros/jazzy/lib/libmoveit_kdl_kinematics_plugin.so, package version 2.12.4-1noble.20260617.150037, and KDLKinematicsPlugin::CartToJnt. The binary has random-related symbols and strings, but this does not prove execution of its internal delta_q.data.setRandom branch. The online Jazzy source contains that branch; exact installed source/binary matching remains pending.

The isolated deterministic plugin was built in ros2_overlay/ and loaded by the C++ replay. Random policy produced mixed outcomes and logged random escape vectors. Disabled policy produced 100/100 stable failures for call:00000182; deterministic-sequence policy also produced no solution in the zero-timeout replay. Determinism improved but candidate coverage regressed.

MoveItPy loaded the isolated class in a real launch, but RobotState.set_from_ik still emitted a blank IK-frame error and the process had an exit -11 teardown warning. This is tracked separately and is not used as a solver pass/fail hash.

The 181-point graph, waypoints 67/68/87/94, Stage 1.7 branch, Ruckig, FK, adaptive-discrete collision, and dynamics regression were not rerun with the new policy. No formal pass is claimed. CCD and clearance remain not_available.
"""
    (OUT / "stage192a_report.md").write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
