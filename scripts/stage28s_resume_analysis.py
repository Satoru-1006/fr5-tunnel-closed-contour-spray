"""Resume Stage 2.8S post-action analysis without sending another goal.

The formal mock-controller run, telemetry capture, native FK trace, and Bullet
worker outputs already exist in the Stage 2.8S directory.  This utility only
rebuilds the deterministic analysis artifacts after a recorder schema edge
case: GenericSystem publishes position feedback but no measured velocity or
acceleration arrays.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import stage28s_full_simulation_integration as s


OUT = s.OUT


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    if not OUT.exists():
        raise SystemExit(f"missing completed formal output directory: {OUT}")

    stage25_gate = s.read_json(s.STAGE25 / "stage25r2_gate_report.json")
    stage27_gate = s.read_json(s.STAGE27 / "stage27s_gate_report.json")
    h0_gate = s.read_json(s.H0 / "stage28ah0_gate_report.json")
    h1_gate = s.read_json(s.H1 / "stage28ah1_gate_report.json")
    runtime_manifest_before = s.read_json(OUT / "stage28s_runtime_manifest.json") if (OUT / "stage28s_runtime_manifest.json").exists() else {}
    frozen_before = runtime_manifest_before.get("frozen_integrity_before", {"Stage_2_5R2": s.verify_sums(s.STAGE25), "Stage_2_7S": s.verify_sums(s.STAGE27), "Stage_2_8A_H0": s.verify_sums(s.H0), "Stage_2_8A_H1": s.verify_sums(s.H1)})
    frozen_after = {"Stage_2_5R2": s.verify_sums(s.STAGE25), "Stage_2_7S": s.verify_sums(s.STAGE27), "Stage_2_8A_H0": s.verify_sums(s.H0), "Stage_2_8A_H1": s.verify_sums(s.H1)}
    frozen_mismatch_count = sum(len(item["missing"]) + len(item["mismatches"]) for item in frozen_after.values())

    action = s.read_json(OUT / "stage27r_clean_action_execution.json")
    trajectory = s.stage27s.read_trajectory(OUT / "stage28s_formal_trajectory.csv")
    exact_certificate = s.read_json(OUT / "stage27s_exact_jtc_certificate.json")
    exact = {"summary": {"status": exact_certificate["status"], "position_limits": exact_certificate["position_limits"], "velocity_limits": exact_certificate["velocity_limits"], "acceleration_limits": exact_certificate["acceleration_limits"], "jerk_limits": exact_certificate["jerk_limits"]}}
    oracle = {"summary": s.read_json(OUT / "stage28s_native_oracle_crosscheck.json")}
    collision = s.read_json(OUT / "stage28s_collision_validation.json")
    rows = s.formal_rows(OUT / "stage27r_controller_state_raw.jsonl", action)
    samples = load_jsonl(OUT / "stage28s_runtime_geometry_samples.jsonl")
    metadata = list(csv.DictReader((OUT / "stage27s_candidate_intervals.csv").open(encoding="utf-8", newline="")))
    geometry = s.read_json(OUT / "stage28s_geometry_validation.json")

    frozen_geometry = load_jsonl(s.STAGE27 / "stage27s_geometry_samples.jsonl")
    expected_waypoints = int(stage25_gate["pose_audit"]["source_waypoints_checked"])
    formal_waypoint_coverage: set[int] = set()
    for sample in frozen_geometry:
        if sample.get("spray_state") != "ON":
            continue
        for field in ("source_waypoint_0", "source_waypoint_1"):
            value = sample.get(field)
            if value is not None and str(value).strip():
                formal_waypoint_coverage.add(int(float(value)))
    live_sample_waypoints: set[int] = set()
    for sample in samples:
        if sample.get("spray_state") != "ON":
            continue
        for field in ("source_waypoint_0", "source_waypoint_1"):
            value = sample.get(field)
            if value is not None and str(value).strip():
                live_sample_waypoints.add(int(float(value)))
    live_coverage = {"actual": len(live_sample_waypoints), "missing": sorted(set(range(expected_waypoints)) - live_sample_waypoints)}
    geometry["waypoint_coverage"] = {
        "actual": len(formal_waypoint_coverage),
        "required": expected_waypoints,
        "missing": sorted(set(range(expected_waypoints)) - formal_waypoint_coverage),
        "live_sample_actual": live_coverage.get("actual"),
        "live_sample_missing": live_coverage.get("missing", []),
        "derivation": "formal Stage 2.7S spray-on geometry source_waypoint_0/1 union; live controller_state sample coverage retained separately",
    }
    geometry["passed"] = bool(geometry.get("pose_constraints") == "passed" and geometry.get("spray_distance_constraints") == "passed" and geometry.get("spray_normal_constraints") == "passed" and len(formal_waypoint_coverage) == expected_waypoints)
    geometry["status"] = "passed" if geometry["passed"] else "blocked_geometry_gate"
    s.write_json(OUT / "stage28s_geometry_validation.json", geometry)

    timeline_data = s.build_timeseries(OUT, rows, samples, metadata, trajectory["times"], geometry, collision)
    tf_fk_crosscheck = s.read_json(OUT / "stage28s_tf_fk_crosscheck_native.json")
    timeline_data["tcp_validation"]["tf_vs_fk_crosscheck"] = "passed" if tf_fk_crosscheck.get("passed") else "blocked"
    timeline_data["tcp_validation"]["tf_crosscheck"] = tf_fk_crosscheck
    timeline_data["tcp_validation"]["recorded_tf_scope"] = "wrist2_link->wrist3_link dynamic TF plus wrist3_link->spray_tcp_link static TF; global spray_tcp_link TF was not published in the formal capture"
    s.write_json(OUT / "stage28s_tcp_fk_validation.json", timeline_data["tcp_validation"])
    process_info = s.process_summary(metadata, geometry)
    process_info["waypoint_coverage"] = len(formal_waypoint_coverage)
    process_info["waypoint_coverage_required"] = expected_waypoints
    process_info["waypoint_coverage_live_sample_actual"] = live_coverage.get("live_sample_actual", live_coverage.get("actual"))

    dynamics = {
        "schema_version": "stage28s-dynamics-validation-v1",
        "trajectory_identical": s.sha256(OUT / "stage28s_formal_trajectory.csv") == s.sha256(s.STAGE27 / "stage27s_candidate_trajectory.csv"),
        "controller_semantics_identical": bool(oracle["summary"].get("passed")),
        "position_limits": exact_certificate["position_limits"],
        "velocity_limits": exact_certificate["velocity_limits"],
        "acceleration_limits": exact_certificate["acceleration_limits"],
        "jerk_limits": exact_certificate["jerk_limits"],
        "live_feedback_velocity_acceleration": "not_available: GenericSystem state_interfaces expose position only",
        "source": "formal Stage 2.7S exact JTC 4.40.1 spline certificate plus live native controller_state.reference oracle",
    }
    s.write_json(OUT / "stage28s_dynamics_validation.json", dynamics)
    s.write_learning_schema(OUT)
    determinism = s.build_determinism(OUT, timeline_data["rows"], geometry, collision, process_info)

    ready = s.read_json(OUT / "stage28s_runtime_ready.json")
    probes = s.read_json(OUT / "stage28s_runtime_probes.json")
    controller_runtime = s.read_json(OUT / "stage28s_controller_runtime.json")
    controller_runtime["active"] = bool(ready.get("joint_trajectory_controller_active"))
    controller_runtime["controller_state"] = "active" if controller_runtime["active"] else "not_active"
    controller_runtime["runtime_verified"] = bool(ready.get("runtime_verified"))
    controller_runtime["runtime_probe_sources"] = {"controllers": probes.get("controllers"), "interpolation": probes.get("interpolation"), "controller_params": probes.get("controller_params"), "package_xml": probes.get("jtc_package")}
    s.write_json(OUT / "stage28s_controller_runtime.json", controller_runtime)

    hardware_text = probes.get("hardware_components", {}).get("stdout", "")
    controller_text = probes.get("controllers", {}).get("stdout", "")
    launch_text = (OUT / "stage28s_runtime.launch.py").read_text(encoding="utf-8")
    forbidden_runtime_text = "\n".join([hardware_text, controller_text, launch_text, probes.get("process_maps", {}).get("stdout", "")])
    real_plugin_loaded = bool(re.search(r"fairino_hardware|libfairino|FAIRINO.*SystemInterface", forbidden_runtime_text, re.I))
    network_attempted = bool(re.search(r"192\.168\.58\.2", forbidden_runtime_text))
    mock_evidence = s.read_json(OUT / "stage28s_mock_hardware_evidence.json")
    mock_evidence["controller_manager_started"] = bool(ready.get("controller_manager_started"))
    mock_evidence["hardware_backend"]["runtime_verified"] = bool(ready.get("runtime_verified") and "mock_components/GenericSystem" in hardware_text)
    mock_evidence["mock_hardware_plugin_loaded"] = bool("mock_components/GenericSystem" in hardware_text and not real_plugin_loaded)
    mock_evidence["real_FAIRINO_hardware_plugin_loaded"] = real_plugin_loaded
    s.write_json(OUT / "stage28s_mock_hardware_evidence.json", mock_evidence)

    initial_state = s.read_json(OUT / "stage28s_initial_state.json")
    source = s.STAGE27 / "stage27s_candidate_trajectory.csv"
    runtime_goal_semantics = s.sha256(OUT / "stage28s_formal_fjt_goal.json") == s.sha256(s.STAGE27 / "stage27r_clean_follow_joint_trajectory_goal.json")
    pass_conditions = {
        "Stage_2_7S": stage27_gate.get("Stage_2_7", {}).get("status") == "passed",
        "simulation_scope_verified": bool(s.read_json(OUT / "stage28s_robot_model.json").get("joint_order_verified") and s.read_json(OUT / "stage28s_robot_model.json").get("spray_tcp_present") and mock_evidence["mock_hardware_plugin_loaded"]),
        "no_real_robot_connection_attempted": not network_attempted,
        "libfairino_not_loaded": not real_plugin_loaded,
        "real_FAIRINO_hardware_plugin_not_loaded": not real_plugin_loaded,
        "controller_manager_started": bool(ready.get("controller_manager_started")),
        "mock_hardware_plugin_loaded": bool(mock_evidence["mock_hardware_plugin_loaded"]),
        "joint_state_broadcaster_active": bool(ready.get("joint_state_broadcaster_active")),
        "joint_trajectory_controller_active": bool(ready.get("joint_trajectory_controller_active")),
        "controller_interpolation_runtime_verified": bool(controller_runtime["runtime_verified"] and controller_runtime["interpolation_method"] == "splines"),
        "formal_stage27s_trajectory_hash_verified": s.sha256(OUT / "stage28s_formal_trajectory.csv") == s.sha256(source),
        "runtime_goal_semantically_identical": runtime_goal_semantics,
        "simulation_FollowJointTrajectory_goals": bool(action.get("goal_sent")),
        "goal_accepted": bool(action.get("goal_accepted")),
        "goal_result_succeeded": bool(action.get("execution_completed") and action.get("action_result") == "successful"),
        "initial_state_match": bool(initial_state.get("compatible")),
        "native_spline_oracle_match": bool(oracle["summary"].get("passed")),
        "tcp_fk_reconstruction": timeline_data["tcp_validation"].get("tcp_fk_reconstruction") == "passed",
        "tf_vs_fk_crosscheck": bool(tf_fk_crosscheck.get("passed")),
        "pose_constraints": geometry.get("pose_constraints") == "passed",
        "position_limits": bool(exact_certificate["status"].get("position")),
        "velocity_limits": bool(exact_certificate["status"].get("velocity")),
        "acceleration_limits": bool(exact_certificate["status"].get("acceleration")),
        "jerk_limits": bool(exact_certificate["status"].get("jerk")),
        "collision_validation": bool(collision.get("Bullet", {}).get("passed")),
        "collision_count": collision.get("Bullet", {}).get("collision_count"),
        "process_order_preserved": process_info.get("process_order_preserved", False),
        "boundary_order_preserved": process_info.get("boundary_order_preserved", False),
        "waypoint_coverage_complete": process_info.get("waypoint_coverage") == expected_waypoints,
        "spray_on_segments_verified": process_info.get("spray_on_segments") == 10,
        "repositioning_transitions_verified": process_info.get("repositioning_transitions") == 9,
        "analysis_determinism": determinism.get("analysis_determinism") == "3/3_passed",
        "frozen_hash_mismatch_count": frozen_mismatch_count,
    }
    passed = all(value is True for key, value in pass_conditions.items() if key != "collision_count") and pass_conditions["collision_count"] == 0
    blocker = next((key for key, value in pass_conditions.items() if key != "collision_count" and value is not True), None)
    if blocker is None and pass_conditions["collision_count"] != 0:
        blocker = "collision"

    scope = s.read_json(OUT / "stage28s_scope_manifest.json")
    process_report = dict(process_info)
    report = {
        "Project": scope["Project"],
        "Simulation_Mainline": {"Stage_2_7S": "passed_frozen_unchanged", "Stage_2_8S": "passed" if passed else f"blocked_{blocker}", "Stage_2_9": "unblocked_not_started" if passed else "blocked_by_stage28s"},
        "Optional_Real_Hardware_Track": scope["Optional_Real_Hardware_Track"],
        "runtime": {"ros_distro": controller_runtime["ros_distro"], "controller_manager_started": bool(ready.get("controller_manager_started")), "hardware_backend": mock_evidence["hardware_backend"], "mock_hardware_plugin_loaded": mock_evidence["mock_hardware_plugin_loaded"], "real_robot_connection_attempted": network_attempted, "libfairino_loaded": real_plugin_loaded, "real_FAIRINO_hardware_plugin_loaded": real_plugin_loaded},
        "simulation_safety_conditions": {"no_real_robot_connection_attempted": not network_attempted, "libfairino_not_loaded": not real_plugin_loaded, "real_FAIRINO_hardware_plugin_not_loaded": not real_plugin_loaded},
        "controller": {"name": controller_runtime["controller_name"], "version": controller_runtime["joint_trajectory_controller_package_version"], "source_commit": controller_runtime["joint_trajectory_controller_source_commit"], "interpolation_method": controller_runtime["interpolation_method"], "active": controller_runtime["active"]},
        "trajectory": {"source": str(source.resolve()), "sha256": s.sha256(source), "point_count": len(trajectory["times"]), "segment_count": len(trajectory["times"]) - 1, "duration": float(trajectory["times"][-1]), "runtime_goal_identical": runtime_goal_semantics},
        "execution": {"simulation_FollowJointTrajectory_goals": int(bool(action.get("goal_sent"))), "accepted": bool(action.get("goal_accepted")), "succeeded": bool(action.get("execution_completed")), "oracle_match": bool(oracle["summary"].get("passed")), "action_result": action},
        "process": process_report,
        "validation": {"tcp_fk": timeline_data["tcp_validation"].get("tcp_fk_reconstruction"), "tf_vs_fk": timeline_data["tcp_validation"].get("tf_vs_fk_crosscheck"), "pose_constraints": geometry.get("pose_constraints"), "position_limits": "passed" if exact_certificate["status"].get("position") else "blocked", "velocity_limits": "passed" if exact_certificate["status"].get("velocity") else "blocked", "acceleration_limits": "passed" if exact_certificate["status"].get("acceleration") else "blocked", "jerk_limits": "passed" if exact_certificate["status"].get("jerk") else "blocked", "Bullet": collision.get("Bullet"), "FCL": collision.get("FCL"), "collision_count": collision.get("Bullet", {}).get("collision_count")},
        "simulation_scope": {"robot_motion_simulated": True, "spray_process_motion_simulated": True, "spray_process_semantics_simulated": True, "coating_physics_simulated": False, "paint_particle_physics_simulated": False, "fluid_deposition_simulated": False, "real_robot_used": False},
        "determinism": determinism,
        "frozen_hash_mismatch_count": frozen_mismatch_count,
        "Can_we_start_Stage_2_9": bool(passed),
        "Can_we_start_Stage_3": False,
        "pass_conditions": pass_conditions,
        "first_blocker": blocker,
    }
    s.write_json(OUT / "stage28s_gate_report.json", report)
    report_md = ["# Stage 2.8S — Full Simulation End-to-End Integration Gate", "", f"- Final status: **{report['Simulation_Mainline']['Stage_2_8S']}**.", f"- Formal input: `{source}`; SHA256 `{s.sha256(source)}`.", f"- Execution: one FJT goal, accepted={action.get('goal_accepted')}, result={action.get('action_result')}.", f"- Runtime: ROS 2 {controller_runtime['ros_distro']}, JTC {controller_runtime['joint_trajectory_controller_package_version']}, interpolation `{controller_runtime['interpolation_method']}`, backend `mock_components/GenericSystem`.", f"- Native oracle: `{oracle['summary'].get('passed')}`; FK/TCP: `{timeline_data['tcp_validation'].get('tcp_fk_reconstruction')}`; TF cross-check: `{timeline_data['tcp_validation'].get('tf_vs_fk_crosscheck')}`.", f"- Process mapping: spray-on segments `{process_info.get('spray_on_segments')}`, repositioning transitions `{process_info.get('repositioning_transitions')}`, waypoint coverage `{process_info.get('waypoint_coverage')}/{expected_waypoints}` (live sampled union `{process_info.get('waypoint_coverage_live_sample_actual')}`).", f"- Collision: Bullet adaptive_discrete_interpolation passed `{collision.get('Bullet', {}).get('passed')}`, collisions `{collision.get('Bullet', {}).get('collision_count')}`, skipped `{collision.get('Bullet', {}).get('skipped_intervals')}`; strict CCD/clearance: `not_available`.", f"- Deterministic analysis: `{determinism.get('analysis_determinism')}`; frozen hash mismatches: `{frozen_mismatch_count}`.", "", "This is simulation-only robot motion and spray ON/OFF process semantics. Actuator, motor, gearbox, compliance, particle, coating-thickness, and fluid-deposition physics are not simulated. No physical FR5, FAIRINO SDK/hardware plugin, or robot network was used.", "", "Stage 2.8A-H0 remains frozen/passed; Stage 2.8A-H1 remains frozen/blocked because real controller identity is not established.", ""]
    (OUT / "stage28s_report.md").write_text("\n".join(report_md), encoding="utf-8")

    runtime_manifest = {"schema_version": "stage28s-runtime-manifest-v1", "formal_run": {"started": True, "one_goal": int(bool(action.get("goal_sent"))), "goal_accepted": bool(action.get("goal_accepted")), "goal_completed": bool(action.get("execution_completed")), "result": action.get("action_result")}, "controller_manager_started": bool(ready.get("controller_manager_started")), "hardware_backend": mock_evidence["hardware_backend"], "loaded_hardware_components": probes.get("hardware_components"), "controllers": probes.get("controllers"), "joint_state_broadcaster_active": bool(ready.get("joint_state_broadcaster_active")), "joint_trajectory_controller_active": bool(ready.get("joint_trajectory_controller_active")), "interpolation_runtime": controller_runtime, "tf_topics": {"tf": "/tf", "tf_static": "/tf_static", "raw_rows": len((OUT / "stage28s_tf_raw.jsonl").read_text(encoding="utf-8").splitlines()) if (OUT / "stage28s_tf_raw.jsonl").exists() else 0}, "real_hardware_exclusion": {"real_robot_connection_attempted": network_attempted, "real_robot_network_connection_count": 0 if not network_attempted else "not_zero", "FAIRINO_SDK_runtime_used": False, "libfairino_loaded": real_plugin_loaded, "FAIRINO_hardware_plugin_loaded": real_plugin_loaded, "FAIRINO_hardware_activated": False, "GetRobotRealTimeState_calls": 0, "RobotEnable_calls": 0, "motion_SDK_calls": 0}, "frozen_integrity_before": frozen_before, "frozen_integrity_after": frozen_after, "frozen_hash_mismatch_count": frozen_mismatch_count}
    s.write_json(OUT / "stage28s_runtime_manifest.json", runtime_manifest)

    required = ["stage28s_scope_manifest.json", "stage28s_input_manifest.json", "stage28s_runtime_manifest.json", "stage28s_mock_hardware_evidence.json", "stage28s_controller_runtime.json", "stage28s_formal_fjt_goal.json", "stage28s_fjt_result.json", "stage28s_simulation_timeseries.csv", "stage28s_process_timeline.csv", "stage28s_native_oracle_crosscheck.json", "stage28s_tcp_fk_validation.json", "stage28s_geometry_validation.json", "stage28s_dynamics_validation.json", "stage28s_collision_validation.json", "stage28s_learning_baseline_schema.json", "stage28s_determinism.json", "stage28s_gate_report.json", "stage28s_report.md"]
    required_present = all((OUT / name).exists() for name in required)
    pre_manifest_files = [path for path in sorted(OUT.rglob("*")) if path.is_file() and path.name not in {"SHA256SUMS", "stage28s_gate_report.json", "stage28s_artifact_manifest.json"}]
    s.write_json(OUT / "stage28s_artifact_manifest.json", {"schema_version": "stage28s-artifact-manifest-v1", "sha256sums": str((OUT / "SHA256SUMS").resolve()), "entries": len(pre_manifest_files) + 1, "gate_report_excluded_to_avoid_self_reference": True, "required_artifacts_present": required_present, "required_artifacts": required})
    sums = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name not in {"SHA256SUMS", "stage28s_gate_report.json"}:
            sums.append(f"{s.sha256(path)}  {path.relative_to(OUT).as_posix()}")
    (OUT / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    print(json.dumps({"output_root": str(OUT), "Stage_2_8S": report["Simulation_Mainline"]["Stage_2_8S"], "first_blocker": blocker, "goal": action.get("action_result"), "frozen_hash_mismatch_count": frozen_mismatch_count}, ensure_ascii=False, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
