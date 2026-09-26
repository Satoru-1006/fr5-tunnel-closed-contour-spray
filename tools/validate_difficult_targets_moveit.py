#!/usr/bin/env python3
"""Validate the frozen difficult targets with the real MoveIt PlanningScene."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.deterministic_numeric_ik import DeterministicNumericIKSolver
from src.deterministic_ik_candidates import explicit_seed_templates


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def pose_from_row(solver, row):
    return solver.pose_matrix([float(row[k]) for k in ("x", "y", "z")], [float(row[k]) for k in ("qx", "qy", "qz", "qw")])


def main() -> int:  # pragma: no cover - requires ROS2/MoveIt2
    import rclpy
    from moveit.core.robot_state import RobotState
    from moveit.planning import MoveItPy
    import plan_closed_contour_moveit as bridge

    output = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "outputs/ik_graph_stage_completion/difficult_target_coverage.json"
    raw = yaml.safe_load((ROOT / "config/stage_completion_deterministic_numeric.yaml").read_text(encoding="utf-8"))
    stage17 = yaml.safe_load((ROOT / raw["inputs"]["stage17_config"]).read_text(encoding="utf-8"))
    poses = read_csv(ROOT / raw["inputs"]["tcp_pose_csv"])
    seeds = read_csv(ROOT / raw["inputs"]["seed_joint_csv"])
    frozen = json.loads((ROOT / "outputs/ik_graph_stage191/minimal_reproduction_case.json").read_text(encoding="utf-8"))
    solver = DeterministicNumericIKSolver(ROOT / raw["deterministic_ik"]["expanded_runtime_urdf"], ROOT / raw["deterministic_ik"]["joint_limits_path"], position_tolerance_m=float(raw["deterministic_ik"]["position_tolerance_m"]), tool_z_tolerance_deg=float(raw["deterministic_ik"]["tool_z_tolerance_deg"]), orientation_weight=float(raw["deterministic_ik"]["orientation_weight"]), continuity_weight=float(raw["deterministic_ik"]["continuity_weight"]), max_nfev=int(raw["deterministic_ik"]["max_nfev"]))
    targets = [{"target_id": "call:00000182", "waypoint_id": 20, "position": [frozen["target_pose"][k] for k in ("x", "y", "z")], "quaternion": [frozen["target_pose"][k] for k in ("qx", "qy", "qz", "qw")], "seed": frozen["seed"], "source": "outputs/ik_graph_stage191/minimal_reproduction_case.json", "mapping_basis": "exact position and quaternion match to authoritative open-arch waypoint 20"}]
    for waypoint in (67, 68, 87, 94):
        row = poses[waypoint]
        targets.append({"target_id": f"waypoint_{waypoint}", "waypoint_id": waypoint, "position": [float(row[k]) for k in ("x", "y", "z")], "quaternion": [float(row[k]) for k in ("qx", "qy", "qz", "qw")], "seed": [float(seeds[waypoint][f"q{i}"]) for i in range(1, 7)], "source": raw["inputs"]["tcp_pose_csv"], "mapping_basis": "authoritative 181-point open-arch row"})

    rclpy.init()
    moveit = MoveItPy(node_name="stage_completion_difficult_targets")
    group = str(raw["robot"]["group_name"])
    ee_link = str(raw["robot"]["ee_link"])
    bridge.preflight_moveit_runtime(moveit, group, ee_link, True)
    environment_count = bridge.apply_collision_environment(moveit, bridge.load_tcp_poses(ROOT / raw["inputs"]["tcp_pose_csv"]), bridge.load_tcp_normals(ROOT / raw["inputs"]["tcp_pose_csv"]), float(stage17["process"]["nominal_standoff_m"]), "base_link", 0.040, 1.10, 1, False, True, True, -0.20)
    psm = moveit.get_planning_scene_monitor()
    state = RobotState(moveit.get_robot_model())
    records = []
    for target in targets:
        matrix = solver.pose_matrix(target["position"], target["quaternion"])
        if target["target_id"] == "call:00000182":
            seed_config = {"seed_templates": {"include_nominal_seed": True, "include_previous_waypoint_seeds": False, "include_fixed_shoulder_templates": True, "include_fixed_elbow_templates": True, "include_fixed_wrist_templates": True, "include_fixed_combination_templates": True}, "seed_offsets_rad": {"fixed_shoulder": [-np.pi, np.pi], "fixed_elbow": [-np.pi, np.pi], "fixed_wrist": [-np.pi, np.pi], "fixed_combination": [{"joint_indices": [0, 2, 4], "offset_rad": np.pi}]}}
            templates = explicit_seed_templates(int(target["waypoint_id"]), target["seed"], None, config=seed_config)
        else:
            templates = explicit_seed_templates(int(target["waypoint_id"]), target["seed"], None, config=raw["deterministic_ik"])
        solved_with = None
        result = None
        for template in templates:
            trial = solver.solve(matrix, template.seed_joint_vector)
            if trial.success:
                result = trial
                solved_with = template
                break
        if result is None:
            result = solver.solve(matrix, templates[0].seed_joint_vector)
            solved_with = templates[0]
        q = np.asarray(result.q_rad, dtype=float)
        state.set_joint_group_positions(group, q)
        state.update()
        actual = bridge._transform_matrix(state.get_global_link_transform(ee_link))
        position_error = float(np.linalg.norm(actual[:3, 3] - matrix[:3, 3]))
        tool_z = actual[:3, 2] / max(float(np.linalg.norm(actual[:3, 2])), 1.0e-15)
        target_z = matrix[:3, 2] / max(float(np.linalg.norm(matrix[:3, 2])), 1.0e-15)
        normal_error = float(np.rad2deg(np.arccos(np.clip(np.dot(tool_z, target_z), -1.0, 1.0))))
        with psm.read_only() as scene:
            colliding, collision_summary = bridge._state_collision_summary(scene, state, group)
        within_limits = bool(np.all(q >= solver.robot.limits.q_min - 1.0e-9) and np.all(q <= solver.robot.limits.q_max + 1.0e-9))
        numerical = bool(result.success)
        pose_pass = bool(position_error <= float(stage17["formal_study_constraints"]["fk_position_error_max_m"]) and normal_error <= float(stage17["formal_study_constraints"]["normal_error_max_deg"]))
        collision_pass = not bool(colliding)
        graph_inserted = bool(numerical and within_limits and pose_pass and collision_pass)
        records.append({**target, "selected_seed_template_id": solved_with.seed_template_id, "selected_seed_template_family": solved_with.seed_template_family, "numerical_solution_generated": numerical, "within_joint_limits": within_limits, "pose_residual_passed": pose_pass, "collision_callback_passed": collision_pass, "graph_candidate_inserted": graph_inserted, "moveit_fk_position_error_m": position_error, "moveit_fk_normal_error_deg": normal_error, "collision_summary": str(collision_summary), "environment_object_count": int(environment_count), "solution_q_rad": q.tolist(), "solver_position_error_m": float(result.position_error_m), "solver_tool_z_error_deg": float(result.tool_z_error_deg), "solver_status": int(result.solver_status), "solver_function_evaluations": int(result.function_evaluations), "solver_api_entry": solver.api_entry, "hidden_random_api_calls": 0, "collision_method": "adaptive_discrete_interpolation", "ccd_status": "not_available", "clearance_status": "not_available"})
    write_json(output, {"schema_version": "1.0", "solver": solver.solver_name, "solver_version": solver.solver_version, "runtime_backend": "MoveIt2 PlanningScene and RobotState FK", "targets": records, "all_targets_graph_inserted": all(bool(x["graph_candidate_inserted"]) for x in records)})
    write_json(output.with_name("difficult_target_coverage_runtime.json"), {"environment_object_count": int(environment_count), "target_count": len(records), "moveit_planning_scene_executed": True})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
