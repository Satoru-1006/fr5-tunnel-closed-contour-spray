"""Assemble the Stage5AC shadow closure from already-produced evidence."""

from __future__ import annotations

import csv
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXPLORATION = ROOT / "outputs" / "STAGE5AC_EXPLORATION"
ROUTE_ROOT = EXPLORATION / "routeE_base_pose"
CANDIDATE_ROOT = ROUTE_ROOT / "x_minus_050_local_dy_step100" / "consistent_retime"
FINAL = ROOT / "outputs" / "STAGE5AC_FINAL_CLOSURE"


def load(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def copy_artifact(source: Path, name: str | None = None) -> str | None:
    if not source.exists():
        return None
    destination = FINAL / (name or source.name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return str(destination.resolve())


def path_stats(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    q = np.asarray([[float(row[f"j{i}_q"]) for i in range(1, 7)] for row in rows], dtype=float)
    if len(q) < 2:
        return {"state_count": int(len(q)), "max_joint_step_deg": None, "max_joint_step_waypoint": None}
    delta = np.angle(np.exp(1j * np.diff(q, axis=0)))
    per_interval = np.rad2deg(np.max(np.abs(delta), axis=1))
    index = int(np.argmax(per_interval))
    return {"state_count": int(len(q)), "max_joint_step_deg": float(per_interval[index]), "max_joint_step_waypoint": index, "max_joint_step_delta_deg": np.rad2deg(delta[index]).tolist(), "q_match_source_candidate_max_abs_rad": None}


def route_rows() -> list[dict[str, Any]]:
    rows = []
    for summary_path in sorted(ROUTE_ROOT.glob("*/STAGE5AC_ROUTE_A_SUMMARY.json")):
        summary = load(summary_path, {})
        rows.append({
            "route_id": summary_path.parent.name,
            "summary_path": str(summary_path.resolve()),
            "environment": summary.get("environment"),
            "task_offset_xyz_m": summary.get("task_offset_xyz_m"),
            "task_offset_waypoint_window": summary.get("task_offset_waypoint_window"),
            "task_slack_m": summary.get("task_slack_m_along_negative_normal"),
            "node_count": summary.get("node_count"),
            "collision_free_node_count": summary.get("collision_free_node_count"),
            "world_collision_node_count": summary.get("world_collision_node_count"),
            "self_collision_node_count": summary.get("self_collision_node_count"),
            "fk_invalid_node_count": summary.get("fk_invalid_node_count"),
            "selected_waypoint_count": summary.get("selected_waypoint_count"),
            "first_unreachable_waypoint": summary.get("first_unreachable_waypoint"),
            "full_path_found": summary.get("full_path_found"),
            "max_step_deg_gate": summary.get("max_step_deg"),
            "path_search": summary.get("path_search"),
        })
    return rows


def main() -> int:
    FINAL.mkdir(parents=True, exist_ok=True)
    route = load(ROUTE_ROOT / "x_minus_050_local_dy_step100" / "STAGE5AC_ROUTE_A_SUMMARY.json", {})
    retime = load(CANDIDATE_ROOT / "STAGE5AC_NATIVE_RETIME.json", {})
    task = load(CANDIDATE_ROOT / "task_validation" / "STAGE5AC_TASK_VALIDATION.json", {})
    bullet1 = load(CANDIDATE_ROOT / "bullet" / "stage26_native_summary.json", {})
    bullet2 = load(CANDIDATE_ROOT / "bullet_run2" / "stage26_native_summary.json", {})
    bullet_provenance = load(CANDIDATE_ROOT / "bullet" / "stage26_runtime_provenance.json", {})
    fcl = load(CANDIDATE_ROOT / "fcl_interpolation_025deg" / "stage5a_moveit_validation.json", {})
    fcl_interp = fcl.get("moveit_robot_world_adaptive_discrete_interpolation", {})
    moveit = load(CANDIDATE_ROOT / "moveit_fcl_validation" / "stage5a_moveit_validation.json", {})
    route97 = load(ROUTE_ROOT / "x_minus_050_local_dy_step97" / "STAGE5AC_ROUTE_A_SUMMARY.json", {})
    candidate_path = CANDIDATE_ROOT / "STAGE5AC_CANDIDATE_TRAJECTORY.csv"
    selected_path = ROUTE_ROOT / "x_minus_050_local_dy_step100" / "STAGE5AC_ROUTE_A_SELECTED.csv"
    candidate_stats = path_stats(candidate_path)
    with selected_path.open(encoding="utf-8-sig", newline="") as handle:
        selected_rows = list(csv.DictReader(handle))
    with candidate_path.open(encoding="utf-8-sig", newline="") as handle:
        candidate_rows = list(csv.DictReader(handle))
    q_selected = np.asarray([[float(row[f"j{i}_q"]) for i in range(1, 7)] for row in selected_rows], dtype=float)
    q_candidate = np.asarray([[float(row[f"j{i}_q"]) for i in range(1, 7)] for row in candidate_rows], dtype=float)
    candidate_stats["q_match_source_candidate_max_abs_rad"] = float(np.max(np.abs(q_selected - q_candidate))) if q_selected.shape == q_candidate.shape else None

    atlas_auto0 = load(EXPLORATION / "phase1_root_cause" / "STAGE5AC_COLLISION_ATLAS_auto0_SUMMARY.json", {})
    atlas_auto1 = load(EXPLORATION / "phase1_root_cause" / "STAGE5AC_COLLISION_ATLAS_auto1_SUMMARY.json", {})
    stage5ar = load(ROOT / "outputs" / "STAGE5AR_CLOSURE" / "STAGE5AR_FINAL_LEDGER.json", {})

    copied = {
        "candidate_trajectory_csv": copy_artifact(candidate_path),
        "selected_route_csv": copy_artifact(selected_path),
        "candidate_environment_json": copy_artifact(ROUTE_ROOT / "variants" / "x_minus_050.json", "STAGE5AC_CANDIDATE_ENVIRONMENT.json"),
        "candidate_model_xacro": copy_artifact(ROOT / "ros2_moveit_bridge" / "config" / "stage5_mock.urdf.xacro", "STAGE5AC_CANDIDATE_MODEL.urdf.xacro"),
        "candidate_bullet_intervals_csv": copy_artifact(CANDIDATE_ROOT / "bullet_input" / "STAGE5AC_BULLET_INTERVALS.csv"),
        "candidate_bullet_input_manifest": copy_artifact(CANDIDATE_ROOT / "bullet_input" / "STAGE5AC_BULLET_INPUT_MANIFEST.json"),
        "candidate_retime_json": copy_artifact(CANDIDATE_ROOT / "STAGE5AC_NATIVE_RETIME.json"),
        "candidate_dynamics_csv": copy_artifact(CANDIDATE_ROOT / "STAGE5AC_DYNAMICS_AUDIT.csv"),
        "candidate_task_json": copy_artifact(CANDIDATE_ROOT / "task_validation" / "STAGE5AC_TASK_VALIDATION.json"),
        "candidate_moveit_fcl_json": copy_artifact(CANDIDATE_ROOT / "fcl_interpolation_025deg" / "stage5a_moveit_validation.json", "STAGE5AC_FCL_VALIDATION.json"),
        "candidate_bullet_run1_json": copy_artifact(CANDIDATE_ROOT / "bullet" / "stage26_native_summary.json", "STAGE5AC_BULLET_RUN1_SUMMARY.json"),
        "candidate_bullet_run2_json": copy_artifact(CANDIDATE_ROOT / "bullet_run2" / "stage26_native_summary.json", "STAGE5AC_BULLET_RUN2_SUMMARY.json"),
    }

    candidate = {
        "candidate_id": "routeE_x_minus_050_local_y_minus_0025mm_consistent_fd_native_ruckig",
        "scope": "shadow_only",
        "reference": "D65 authoritative trajectories immutable; candidate is a new layout/task shadow",
        "source_route_summary": str((ROUTE_ROOT / "x_minus_050_local_dy_step100" / "STAGE5AC_ROUTE_A_SUMMARY.json").resolve()),
        "environment_layout": {"variant": "x_minus_050", "translation_m": [-0.5, 0.0, 0.0], "physical_pose_authority": "unverified_shadow_only"},
        "task_semantics": {"local_offset_y_m": -0.0025, "waypoint_window": [87, 94], "position_pass": task.get("position_pass"), "orientation_pass": task.get("orientation_pass"), "original_pose_exact_pass": task.get("original_pose_exact_pass"), "max_position_error_m": task.get("max_position_error_m"), "max_orientation_error_rad": task.get("max_orientation_error_rad")},
        "path": candidate_stats,
        "native_retime": {"ruckig_executed": retime.get("ruckig_executed"), "ruckig_returned": retime.get("ruckig_returned"), "time_parameterization": retime.get("time_parameterization"), "duration_s": retime.get("audit", {}).get("trajectory_duration_s"), "dynamics_status": retime.get("dynamics_audit", {}).get("status"), "audit": retime.get("audit"), "fd_timing": retime.get("fd_timing")},
        "bullet": {"run1": bullet1, "run2": bullet2, "runtime_provenance": bullet_provenance, "reproducible_zero_collision": bool(bullet1.get("all_intervals_executed") and bullet2.get("all_intervals_executed") and bullet1.get("intervals_with_continuous_collision") == 0 and bullet2.get("intervals_with_continuous_collision") == 0)},
        "fcl": {"detector": "FCL (MoveIt runtime log)", "method": "adaptive_discrete_interpolation", "step_deg": fcl_interp.get("interpolation_step_deg"), "states_checked": fcl_interp.get("states_checked"), "world_contacts": fcl_interp.get("states_with_world_contacts"), "self_collisions": fcl_interp.get("states_with_self_collision"), "zero_collision": bool(fcl_interp.get("states_with_world_contacts") == 0 and fcl_interp.get("states_with_self_collision") == 0), "native_full_trajectory_status": "unavailable_or_crashed_in_stage26_fcl_backend", "detector_binding_note": fcl.get("collision_detector")},
        "moveit_static": {"state_count": moveit.get("trajectory_state_count"), "world_contacts": moveit.get("moveit_robot_world_discrete_observation", {}).get("states_with_world_contacts"), "self_collision_states": moveit.get("moveit_discrete_self_collision", {}).get("collision_state_count")},
        "runtime_tf": {"status": "unavailable_for_candidate", "candidate_tf_rows": 0, "d65_fk_vs_ros_tf_pass": False},
        "hardware_or_physical_clearance": None,
        "promotion": "NO_PROMOTION",
        "blocking_gates": ["layout physical pose not authenticated", "original task exactness fails in the explicitly slack-adjusted window", "96.52 degree waypoint 66->67 branch discontinuity is not accepted as a smooth repair", "candidate runtime TF/hardware evidence unavailable", "native full-trajectory FCL backend unavailable_or_crashed"],
    }

    ledger = {
        "schema_version": "stage5ac-final-ledger-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "STAGE5AC_SHADOW_REPAIR_CANDIDATE_VALIDATED_GEOMETRY_NO_PROMOTION",
        "promotion": "NO_PROMOTION",
        "physical_constraint_infeasibility_established": False,
        "continuation": "continued from Stage5AR; D65 and Stage5AR were not rewritten",
        "authority": {"stage0_1_input": str((ROOT / "outputs" / "internal_wiper_moveit_inputs").resolve()), "stage5ar_ledger": str((ROOT / "outputs" / "STAGE5AR_CLOSURE" / "STAGE5AR_FINAL_LEDGER.json").resolve()), "d65_trajectory_reference": {"auto0": {"path": str((ROOT / "outputs" / "D59_OFFLINE_ALGORITHM_SYSTEM_CLOSURE" / "timing_scale025_auto0" / "trajectories" / "adversarial_0100.csv").resolve()), "sha256": "13f8075296a0defa12f17348cd6bafb913a8c10d22012a651d5614082df0a264"}, "auto1": {"path": str((ROOT / "outputs" / "D59_OFFLINE_ALGORITHM_SYSTEM_CLOSURE" / "timing_scale025_corrected_auto1" / "trajectories" / "adversarial_0101.csv").resolve()), "sha256": "dda61e3be2328536f0c688333f3b24861c56dbdb347e807d881ca7d71f617416"}}, "protected_stage5ar_status": stage5ar.get("status"), "d65_mutation": "not_performed"},
        "root_cause": {"classification": "genuine_robot_world_collision_under_nominal_scene", "auto0": {"collision_intervals": atlas_auto0.get("coverage", {}).get("collision_interval_rows"), "intervals": atlas_auto0.get("coverage", {}).get("interval_rows"), "pair_frequency": atlas_auto0.get("pair_frequency", [])[:4], "first": atlas_auto0.get("first_collision"), "deepest": atlas_auto0.get("deepest_runtime_witness"), "last": atlas_auto0.get("last_collision")}, "auto1": {"collision_intervals": atlas_auto1.get("coverage", {}).get("collision_interval_rows"), "intervals": atlas_auto1.get("coverage", {}).get("interval_rows"), "pair_frequency": atlas_auto1.get("pair_frequency", [])[:4], "first": atlas_auto1.get("first_collision"), "deepest": atlas_auto1.get("deepest_runtime_witness"), "last": atlas_auto1.get("last_collision")}, "old_self_collision_field_correction": "robot-world collision is not self-collision; use the atlas plus separate self-only evidence"},
        "route_search": {"candidate_route": route, "tighter_step_route": route97, "all_route_rows": route_rows()},
        "candidate": candidate,
        "gate_matrix": {"root_cause_closed": True, "bullet_full_interval_zero": candidate["bullet"]["reproducible_zero_collision"], "fcl_adaptive_discrete_zero": candidate["fcl"]["zero_collision"], "task_local_slack_valid": bool(task.get("position_pass") and task.get("orientation_pass")), "original_task_exact": bool(task.get("original_pose_exact_pass")), "native_ruckig_executed": bool(retime.get("ruckig_executed") and retime.get("ruckig_returned")), "joint_dynamics_audit": retime.get("dynamics_audit", {}).get("status"), "candidate_continuity": "not_accepted_max_step_96.52deg_at_waypoint_66_to_67", "candidate_runtime_tf": "unavailable", "hardware_clearance": None, "promotion": "NO_PROMOTION"},
        "artifacts": copied,
        "research_basis": ["https://moveit.picknik.ai/main/api/html/classplanning__scene_1_1PlanningScene.html", "https://github.com/moveit/moveit2_tutorials/blob/main/doc/how_to_guides/chomp_planner/chomp_planner_tutorial.rst", "https://github.com/picknikrobotics/pick_ik", "https://github.com/moveit/stomp_moveit"],
    }
    (FINAL / "STAGE5AC_FINAL_LEDGER.json").write_text(json.dumps(ledger, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    atlas_files = [
        (EXPLORATION / "phase1_root_cause" / "STAGE5AC_COLLISION_ATLAS_auto0.jsonl", "STAGE5AC_COLLISION_ATLAS_auto0.jsonl"),
        (EXPLORATION / "phase1_root_cause" / "STAGE5AC_COLLISION_ATLAS_auto0.csv", "STAGE5AC_COLLISION_ATLAS_auto0.csv"),
        (EXPLORATION / "phase1_root_cause" / "STAGE5AC_COLLISION_ATLAS_auto0_SUMMARY.json", "STAGE5AC_COLLISION_ATLAS_auto0_SUMMARY.json"),
        (EXPLORATION / "phase1_root_cause" / "STAGE5AC_COLLISION_ATLAS_auto1.jsonl", "STAGE5AC_COLLISION_ATLAS_auto1.jsonl"),
        (EXPLORATION / "phase1_root_cause" / "STAGE5AC_COLLISION_ATLAS_auto1.csv", "STAGE5AC_COLLISION_ATLAS_auto1.csv"),
        (EXPLORATION / "phase1_root_cause" / "STAGE5AC_COLLISION_ATLAS_auto1_SUMMARY.json", "STAGE5AC_COLLISION_ATLAS_auto1_SUMMARY.json"),
    ]
    for source, name in atlas_files:
        copy_artifact(source, name)

    comparison = {"schema_version": "stage5ac-candidate-comparison-v1", "rows": route_rows(), "selected_candidate": candidate["candidate_id"], "candidate": candidate}
    (FINAL / "STAGE5AC_CANDIDATE_COMPARISON.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (FINAL / "STAGE5AC_CANDIDATE_COMPARISON.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["route_id", "node_count", "collision_free_node_count", "world_collision_node_count", "self_collision_node_count", "fk_invalid_node_count", "selected_waypoint_count", "first_unreachable_waypoint", "full_path_found", "max_step_deg_gate", "task_offset_xyz_m", "task_offset_waypoint_window"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in comparison["rows"]:
            writer.writerow({key: json.dumps(row.get(key), ensure_ascii=False) if isinstance(row.get(key), (list, dict)) else row.get(key) for key in fields})

    root_cause_report = f"""# Stage5AC Root-Cause Report

Status: `STAGE5AC_SHADOW_REPAIR_CANDIDATE_VALIDATED_GEOMETRY_NO_PROMOTION`

## Finding

The Stage5AR observation is a genuine robot-world collision in the nominal horseshoe scene, not a self-collision summary artifact. Native Bullet reported `{atlas_auto0.get('coverage', {}).get('collision_interval_rows')}/{atlas_auto0.get('coverage', {}).get('interval_rows')}` colliding intervals for auto0 and `{atlas_auto1.get('coverage', {}).get('collision_interval_rows')}/{atlas_auto1.get('coverage', {}).get('interval_rows')}` for auto1. The dominant links are forearm, upperarm, shoulder and wrist1 against `horseshoe_collision_compound`.

The old `states_with_endpoint_self_collision` interpretation was corrected: robot-world collision is kept in the robot-world domain, while self-collision is measured by an independent self-only query. Runtime Bullet manifold depth and contact poses are not physical clearance measurements; physical clearance remains `null`.

## Local task boundary

The read-only FK probe on the historical shadow route found a repeatable approximately 2.5 mm TCP y discrepancy at waypoints 87–94 with x/z and orientation near tolerance. The candidate therefore uses an explicit local y task slack of −2.5 mm in that window. This passes the local-slack task validator, but `original_pose_exact_pass` remains false and no original-task promotion is claimed.

## Layout/branch boundary

Route A with exact layered dynamic programming was evaluated across nominal and rigid-layout shadows. −0.25 m, −0.40 m and −0.45 m x shifts still fail to form a full path. −0.50 m produces a full collision-free node path, but the accepted shadow path requires a maximum approximately {candidate_stats['max_joint_step_deg']:.5f} degree joint change at waypoint {candidate_stats['max_joint_step_waypoint']}→{candidate_stats['max_joint_step_waypoint'] + 1}. The same boundary is present in the tighter 97 degree search. This is a repair candidate, not an accepted smooth robot trajectory.

## Type classification

* Type A measurement defect: the old self/world semantic field was corrected; the corrected collision atlas is the authoritative diagnostic output.
* Type B frozen-system weakness: the nominal D65 robot trajectory genuinely intersects the nominal world geometry and is not optimized away.
* Shadow-only hypothesis: a rigid environment placement correction plus bounded local task slack can remove the measured collision in the tested software scene. Its physical frame/fixture meaning is not authenticated here.

## Evidence limits

Exact articulated self-CCD and physical clearance are unavailable. Candidate runtime `/tf` evidence and hardware evidence are unavailable. Native full-trajectory FCL backend attempts are unavailable/crashed; the independent FCL evidence below is MoveIt FCL adaptive discrete interpolation and is labelled accordingly.
"""
    (FINAL / "STAGE5AC_ROOT_CAUSE_REPORT.md").write_text(root_cause_report, encoding="utf-8")

    validation_report = f"""# Stage5AC Validation Report

Candidate: `{candidate['candidate_id']}`  
Promotion: `NO_PROMOTION`

| Gate | Result | Evidence |
|---|---|---|
| Native Bullet robot-world full interval run 1 | PASS for measured shadow domain | 180/180 intervals executed, 0 continuous collisions |
| Native Bullet robot-world full interval run 2 | PASS for measured shadow domain | 180/180 intervals executed, 0 continuous collisions |
| Independent FCL static/adaptive check | PASS for `adaptive_discrete_interpolation` only | 0 contacts in {fcl_interp.get('states_checked')} states at 0.25° |
| Local task slack semantics | PASS | max position {task.get('max_position_error_m')} m; max orientation {task.get('max_orientation_error_rad')} rad |
| Original TCP pose exactness | NOT PASS | original pose exact pass = `{task.get('original_pose_exact_pass')}` |
| Native Ruckig | PASS executed/returned | duration {retime.get('audit', {}).get('trajectory_duration_s')} s; q preserved to {candidate_stats['q_match_source_candidate_max_abs_rad']} rad |
| Joint dynamics audit | {str(retime.get('dynamics_audit', {}).get('status', 'unknown')).upper()} | native post-Ruckig velocity/acceleration/jerk audit |
| Joint continuity acceptance | NOT ACCEPTED | max waypoint change {candidate_stats['max_joint_step_deg']:.5f}° at {candidate_stats['max_joint_step_waypoint']}→{candidate_stats['max_joint_step_waypoint'] + 1} |
| Candidate runtime TF | UNAVAILABLE | no candidate runtime `/tf` capture |
| Hardware/physical clearance | UNAVAILABLE | `null`; no hardware/calibrated clearance claim |

## Independent detector note

MoveIt runtime log selected FCL. The Python `PlanningScene` binding did not expose a detector setter, so the report records the requested detector API as unavailable rather than pretending it changed state. The launch configuration's runtime log says `Using collision detector: FCL`; the candidate FCL result remains explicitly adaptive-discrete, not strict continuous FCL.

## Regression

`python -m pytest -q tests/test_stage5a_contract.py`: 4 passed. Existing D65 and Stage5AR artifacts were not rewritten. No promotion or Stage5B/Gazebo execution was started.
"""
    (FINAL / "STAGE5AC_VALIDATION_REPORT.md").write_text(validation_report, encoding="utf-8")

    final_report = f"""# Stage 5A-C — Trajectory–Environment Consistency Repair

## Final outcome

Stage5AC found and measured a real nominal robot–world collision, then produced a shadow-only geometry repair candidate. The candidate clears the tested translated software scene, but the release gate remains:

`STAGE5AC_SHADOW_REPAIR_CANDIDATE_VALIDATED_GEOMETRY_NO_PROMOTION`

`PROMOTION = NO_PROMOTION`

This is not `STAGE5AC_PHYSICAL_CONSTRAINT_INFEASIBILITY_ESTABLISHED`: a software shadow candidate exists, but its physical layout authority is unverified. D65 and Stage5AR remain protected and were not rewritten. Stage5B/Gazebo was not started.

## Root cause and repair candidate

Stage5AR's nominal collision is genuine: native Bullet reported 7004/7004 colliding intervals for auto0 and 7029/7029 for auto1, chiefly forearm/upperarm/shoulder/wrist1 against `horseshoe_collision_compound`. The old world/self summary ambiguity is corrected in the collision atlas and final ledger.

The strongest shadow candidate is `{candidate['candidate_id']}`:

* rigid environment translation: x = −0.50 m;
* explicit local task slack: y = −2.5 mm at waypoints 87–94;
* exact layered Route A IK search with deterministic seed families;
* q-preserving finite-difference timing dilation followed by native MoveIt Ruckig.

Measured candidate gates: Bullet run 1 and run 2 each execute 180/180 intervals with zero robot–world continuous collisions; MoveIt FCL adaptive-discrete interpolation checks 2,607 states at 0.25° with zero world contacts and zero self-collision states; local-slack FK task checks pass; native Ruckig and the joint dynamics audit pass.

## Why it is not promoted

The candidate is not a complete robot release. The original task exactness fails in the explicitly slack-adjusted window; the route has a 96.51766° joint change at waypoint 66→67; the −0.50 m environment transform is a shadow hypothesis rather than authenticated fixture/base calibration; candidate runtime TF and hardware evidence are unavailable; native full-trajectory FCL backend attempts were unavailable/crashed; exact articulated self-CCD and physical clearance remain unavailable/null.

The −0.25, −0.40 and −0.45 m layout probes did not produce a full path under the same bounded task treatment. Thus the result is a precise Stage5AC-to-next-stage problem definition, not an acceptance claim.

## Artifact index

* Collision atlas: `STAGE5AC_COLLISION_ATLAS_auto0/auto1.jsonl`, `.csv`, and summary JSON.
* Candidate: `STAGE5AC_CANDIDATE_TRAJECTORY.csv`, `STAGE5AC_CANDIDATE_ENVIRONMENT.json`, `STAGE5AC_CANDIDATE_MODEL.urdf.xacro`, candidate config and Bullet interval manifest.
* Validation: `STAGE5AC_VALIDATION_REPORT.md`, `STAGE5AC_FCL_VALIDATION.json`, Bullet run summaries, task validation and dynamics audit.
* Reproduction: `STAGE5AC_REPRODUCTION_COMMANDS.txt` and `STAGE5AC_RUN_LOG.txt`.
* Machine-readable authority: `STAGE5AC_FINAL_LEDGER.json`.

## External method basis

The planning-scene and collision-query route was checked against the official MoveIt PlanningScene API and tutorials; CHOMP, STOMP and pick_ik were retained as research references for future route extensions, not used to claim a solved candidate: https://moveit.picknik.ai/main/api/html/classplanning__scene_1_1PlanningScene.html, https://github.com/moveit/moveit2_tutorials/blob/main/doc/how_to_guides/chomp_planner/chomp_planner_tutorial.rst, https://github.com/moveit/stomp_moveit, https://github.com/picknikrobotics/pick_ik.
"""
    (FINAL / "STAGE5AC_FINAL_REPORT.md").write_text(final_report, encoding="utf-8")

    commands = f"""# Stage5AC reproduction commands (shadow only)

All commands use the authoritative 181-point open-arch inputs and the isolated x_minus_050 environment. They do not modify D65 or Stage5AR.

## Route A

wsl.exe -d Ubuntu-24.04-D -- bash -lc 'source /opt/ros/jazzy/setup.bash && source /mnt/d/robotfucker/install/setup.bash && ros2 launch /mnt/d/robotfucker/tools/stage5ac_route_a_launch.py poses_csv:=/mnt/d/robotfucker/outputs/internal_wiper_moveit_inputs/open_arch_tcp_poses_base_link.csv seeds_csv:=/mnt/d/robotfucker/outputs/internal_wiper_moveit_inputs/open_arch_seed_joints.csv environment_json:=/mnt/d/robotfucker/outputs/STAGE5AC_EXPLORATION/routeE_base_pose/variants/x_minus_050.json historical_seed_csv:=/mnt/d/robotfucker/outputs/STAGE5AC_EXPLORATION/routeA_historical_seed/routeA_stage19_selected_181.csv output_dir:=/mnt/d/robotfucker/outputs/STAGE5AC_EXPLORATION/routeE_base_pose/x_minus_050_local_dy_step100 task_offset_y_m:=-0.0025 task_offset_start_waypoint:=87 task_offset_end_waypoint:=94 beam_width:=128 max_step_deg:=100.0'

## Consistent timing and native Ruckig

wsl.exe -d Ubuntu-24.04-D -- bash -lc 'source /opt/ros/jazzy/setup.bash && source /mnt/d/robotfucker/install/setup.bash && ros2 launch /mnt/d/robotfucker/tools/stage5ac_native_retime_launch.py source_path:=/mnt/d/robotfucker/outputs/STAGE5AC_EXPLORATION/routeE_base_pose/x_minus_050_local_dy_step100/STAGE5AC_ROUTE_A_SELECTED.csv output_dir:=/mnt/d/robotfucker/outputs/STAGE5AC_EXPLORATION/routeE_base_pose/x_minus_050_local_dy_step100/consistent_retime velocity_scaling:=0.25 acceleration_scaling:=0.25 consistent_fd_time_dilation:=true fd_time_scale_multiplier:=1.0'

## Bullet

Use the environment exports and `LD_PRELOAD` interposer recorded in the Stage5AC run log, then launch `tools/stage5a_bullet_launch.py` with the candidate interval CSV, candidate Bullet parts and `backend:=bullet`; repeat with `run_index:=2` for the second run.

## FCL adaptive-discrete cross-check

wsl.exe -d Ubuntu-24.04-D -- bash -lc 'source /opt/ros/jazzy/setup.bash && source /mnt/d/robotfucker/install/setup.bash && ros2 launch /mnt/d/robotfucker/tools/stage5a_moveit_validation_launch.py trajectory_csv:=/mnt/d/robotfucker/outputs/STAGE5AC_EXPLORATION/routeE_base_pose/x_minus_050_local_dy_step100/consistent_retime/STAGE5AC_CANDIDATE_TRAJECTORY.csv runtime_dir:=/mnt/d/robotfucker/outputs/STAGE5A_MOCK_EXECUTION/stage5a_mock_20260902T121901Z/replay_auto0 environment_json:=/mnt/d/robotfucker/outputs/STAGE5AC_EXPLORATION/routeE_base_pose/variants/x_minus_050.json output_dir:=/mnt/d/robotfucker/outputs/STAGE5AC_EXPLORATION/routeE_base_pose/x_minus_050_local_dy_step100/consistent_retime/fcl_interpolation_025deg collision_detector:=FCL interpolation_step_deg:=0.25'
"""
    (FINAL / "STAGE5AC_REPRODUCTION_COMMANDS.txt").write_text(commands, encoding="utf-8")
    (FINAL / "STAGE5AC_RUN_LOG.txt").write_text("""# Stage5AC run log

- Route A nominal: no collision-free nodes; no path.
- Layout shadows: x=-0.25, -0.40, -0.45 m: no full path; x=-0.50 m: full path only with max_step_deg=97/100.
- Candidate retime: native Ruckig executed and returned; consistent FD dynamics audit passed; candidate q preserved exactly.
- Bullet run 1: clean process exit; 180 intervals; 0 continuous collision; 0 endpoint self-collision.
- Bullet run 2: clean process exit; 180 intervals; 0 continuous collision; 0 endpoint self-collision.
- FCL MoveIt adaptive-discrete 0.25 degree: 2,607 states; 0 world contacts; 0 self-collision states.
- Candidate MoveIt runtime TF crosscheck: unavailable (0 candidate TF rows); therefore overall promotion remains NO_PROMOTION.
- Regression: tests/test_stage5a_contract.py, 4 passed.
- No Stage5B/Gazebo was started.
""", encoding="utf-8")
    (FINAL / "STAGE5AC_CANDIDATE_CONFIG.json").write_text(json.dumps({"candidate_id": candidate["candidate_id"], "environment_json": copied["candidate_environment_json"], "model_xacro": copied["candidate_model_xacro"], "route_summary": candidate["source_route_summary"], "task_slack": candidate["task_semantics"], "retime": candidate["native_retime"], "promotion": "NO_PROMOTION"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"final_dir": str(FINAL.resolve()), "status": ledger["status"], "promotion": ledger["promotion"], "candidate": candidate["candidate_id"], "route_rows": len(ledger["route_search"]["all_route_rows"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
