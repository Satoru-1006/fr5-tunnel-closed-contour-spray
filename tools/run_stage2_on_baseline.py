#!/usr/bin/env python3
"""Run the first Stage 2 closed-horseshoe ON-state baseline."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.deterministic_ik_candidates import explicit_seed_templates
from src.task_pose_repair import evaluate_transition
from src.deterministic_numeric_ik import DeterministicNumericIKSolver


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> int:  # pragma: no cover - requires ROS2/MoveIt2
    import rclpy
    from moveit.core.robot_state import RobotState
    from moveit.planning import MoveItPy
    import plan_closed_contour_moveit as bridge

    out = ROOT / "outputs/ik_graph_stage2"
    out.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((ROOT / "config/stage_completion_deterministic_numeric.yaml").read_text(encoding="utf-8"))
    stage2_poses = ROOT / "outputs/tcp_poses_base_link.csv"
    stage2_seeds = ROOT / "outputs/ik_waypoints.csv"
    poses = load_csv(stage2_poses)
    seeds = load_csv(stage2_seeds)
    if len(poses) != 720 or len(seeds) != 720:
        raise RuntimeError("Stage 2 closed-horseshoe baseline requires the 720-point nominal pair")
    solver = DeterministicNumericIKSolver(ROOT / config["deterministic_ik"]["expanded_runtime_urdf"], ROOT / config["deterministic_ik"]["joint_limits_path"], position_tolerance_m=float(config["deterministic_ik"]["position_tolerance_m"]), tool_z_tolerance_deg=float(config["deterministic_ik"]["tool_z_tolerance_deg"]), orientation_weight=float(config["deterministic_ik"]["orientation_weight"]), continuity_weight=float(config["deterministic_ik"]["continuity_weight"]), max_nfev=int(config["deterministic_ik"]["max_nfev"]))

    rclpy.init()
    moveit = MoveItPy(node_name="stage2_closed_horseshoe_on_baseline")
    group = str(config["robot"]["group_name"])
    ee_link = str(config["robot"]["ee_link"])
    bridge.preflight_moveit_runtime(moveit, group, ee_link, True)
    environment_count = bridge.apply_collision_environment(moveit, bridge.load_tcp_poses(stage2_poses), bridge.load_tcp_normals(stage2_poses), 0.260, "base_link", 0.040, 1.10, 1, False, True, True, -0.20)
    psm = moveit.get_planning_scene_monitor()
    state = RobotState(moveit.get_robot_model())
    seed_cfg = config["deterministic_ik"]
    layers = []
    selected = []
    attempts = []
    previous_seed = None
    for waypoint, row in enumerate(poses):
        target = solver.pose_matrix([float(row[k]) for k in ("x", "y", "z")], [float(row[k]) for k in ("qx", "qy", "qz", "qw")])
        nominal_seed = [float(seeds[waypoint][f"q{i}"]) for i in range(1, 7)]
        templates = explicit_seed_templates(waypoint, nominal_seed, previous_seed, config=seed_cfg)
        valid_nodes = []
        for order, template in enumerate(templates):
            result = solver.solve(target, template.seed_joint_vector)
            record = {"waypoint_id": waypoint, "seed_template_id": template.seed_template_id, "seed_template_family": template.seed_template_family, "solver_success": bool(result.success), "solver_position_error_m": float(result.position_error_m), "solver_tool_z_error_deg": float(result.tool_z_error_deg), "solver_status": int(result.solver_status), "solver_function_evaluations": int(result.function_evaluations), "collision_callback_passed": False}
            if result.success:
                q = np.asarray(result.q_rad, dtype=float)
                state.set_joint_group_positions(group, q); state.update()
                actual = bridge._transform_matrix(state.get_global_link_transform(ee_link))
                pos_err = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
                tool_z = actual[:3, 2] / max(float(np.linalg.norm(actual[:3, 2])), 1.0e-15)
                target_z = target[:3, 2] / max(float(np.linalg.norm(target[:3, 2])), 1.0e-15)
                normal_err = float(np.rad2deg(np.arccos(np.clip(np.dot(tool_z, target_z), -1.0, 1.0))))
                with psm.read_only() as scene:
                    colliding, summary = bridge._state_collision_summary(scene, state, group)
                limits_pass = bool(np.all(q >= solver.robot.limits.q_min - 1.0e-9) and np.all(q <= solver.robot.limits.q_max + 1.0e-9))
                pose_pass = bool(pos_err <= 0.006 and normal_err <= 10.0)
                record.update({"moveit_fk_position_error_m": pos_err, "moveit_fk_normal_error_deg": normal_err, "within_joint_limits": limits_pass, "pose_residual_passed": pose_pass, "collision_callback_passed": not bool(colliding), "collision_summary": str(summary), "q_rad": q.tolist()})
                if limits_pass and pose_pass and not colliding:
                    node = {"waypoint_id": waypoint, "candidate_id": f"stage2-node-{waypoint:04d}-{len(valid_nodes):02d}", "ik_candidate_id": f"stage2-node-{waypoint:04d}-{len(valid_nodes):02d}", "q_rad": q.tolist(), "q_unwrapped_rad": q.tolist(), "formal_constraint_pass": True, "valid": True, "diagnostic_only": False, "node_collision": False, "is_nominal": order == 0, "node_cost": float(np.sum(np.abs(q - np.asarray(nominal_seed))))}
                    valid_nodes.append(node)
            attempts.append(record)
        layers.append(valid_nodes)
        if valid_nodes:
            selected.append(valid_nodes[0])
            previous_seed = np.asarray(valid_nodes[0]["q_rad"], dtype=float)
        else:
            selected.append(None)
            previous_seed = np.asarray(nominal_seed, dtype=float)

    def collision_at(q):
        state.set_joint_group_positions(group, q); state.update()
        with psm.read_only() as scene:
            return bridge._state_collision_summary(scene, state, group)

    edges = []
    first_unreachable = next((i for i, layer in enumerate(layers) if not layer), None)
    for i in range(719):
        if selected[i] is None or selected[i + 1] is None:
            edges.append({"from_waypoint": i, "to_waypoint": i + 1, "valid": False, "reject_reason": "candidate_layer_unreachable"})
            continue
        edges.append(evaluate_transition(selected[i], selected[i + 1], max_joint_step_deg=20.0, interpolation_step_deg=0.5, collision_checker=collision_at))
    unresolved = []
    bad = [i for i, edge in enumerate(edges) if not bool(edge.get("valid"))]
    for i in bad:
        if not unresolved or i > unresolved[-1]["end_waypoint"] + 1:
            unresolved.append({"start_waypoint": i, "end_waypoint": i + 1, "reason": edges[i].get("reject_reason")})
        else:
            unresolved[-1]["end_waypoint"] = i + 1
    complete = first_unreachable is None and not bad and len(selected) == 720
    q_path = [x["q_rad"] for x in selected if x is not None]
    candidate_counts = [len(x) for x in layers]
    candidate_summary = {"schema_version": "1.0", "stage": "stage_2_closed_horseshoe_on", "waypoint_count": 720, "nominal_path_source": "outputs/tcp_poses_base_link.csv", "candidate_counts": candidate_counts, "ik_candidate_count": sum(candidate_counts), "attempt_count": len(attempts), "complete_on_path_exists": complete, "first_unreachable_waypoint": first_unreachable, "unresolved_regions": unresolved, "pose_repair_used": False, "max_tcp_offset": 0.0, "max_roll_offset": 0.0, "collision_count": sum(1 for x in attempts if x.get("collision_callback_passed") is False and x.get("solver_success")), "ruckig_status": "not_run_complete_on_path_false" if not complete else "not_run_requires_stage2_retimer_review", "collision_method": "adaptive_discrete_interpolation", "ccd_status": "not_available", "clearance_status": "not_available"}
    graph = {"schema_version": "1.0", "stage": "stage_2_closed_horseshoe_on", "waypoint_count": 720, "candidate_layer_count": sum(bool(x) for x in layers), "edge_count": len(edges), "valid_edge_count": sum(bool(x.get("valid")) for x in edges), "complete_on_path_exists": complete, "first_unreachable_waypoint": first_unreachable, "unresolved_regions": unresolved, "selected_waypoint_count": len(q_path), "environment_object_count": int(environment_count), "collision_method": "adaptive_discrete_interpolation", "ccd_status": "not_available", "clearance_status": "not_available"}
    write_json(out / "horseshoe_model_manifest.json", {"schema_version": "1.0", "nominal_path_source": str(stage2_poses.relative_to(ROOT)).replace("\\", "/"), "waypoint_count": 720, "closed_nominal_path": True, "tcp_frame": "spray_tcp_link", "tcp_transform": {"xyz_m": [0.0, 0.0, 0.15], "rpy_rad": [0.0, 0.0, 0.0]}, "input_sha256": sha256(stage2_poses), "seed_sha256": sha256(stage2_seeds), "runtime_urdf_sha256": sha256(ROOT / config["deterministic_ik"]["expanded_runtime_urdf"]), "moveit_planning_scene_executed": True, "collision_method": "adaptive_discrete_interpolation"})
    write_json(out / "horseshoe_nominal_path.json", {"schema_version": "1.0", "waypoint_count": 720, "source": str(stage2_poses.relative_to(ROOT)).replace("\\", "/"), "selected_joint_path": q_path, "complete_on_path_exists": complete})
    write_json(out / "horseshoe_on_candidate_summary.json", candidate_summary)
    write_json(out / "horseshoe_on_graph_summary.json", {**graph, "edge_records": edges, "attempts": attempts})
    write_json(out / "stage2_status.yaml", {"schema_version": "1.0", "stage2_status": "unblocked", "current_phase": "on_state_closed_horseshoe_baseline", "baseline_status": "completed", "complete_on_path_exists": complete})
    (out / "stage2_status.yaml").write_text("schema_version: '1.0'\nstage2_status: unblocked\ncurrent_phase: on_state_closed_horseshoe_baseline\nbaseline_status: completed\ncomplete_on_path_exists: %s\n" % str(complete).lower(), encoding="utf-8")
    (out / "stage2_report.md").write_text("# Stage 2 closed-horseshoe ON baseline\n\nThe nominal 720-point closed path was evaluated with the deterministic numeric IK backend and the MoveIt2 PlanningScene.\n\n- Complete ON path: `%s`\n- First unreachable waypoint: `%s`\n- Unresolved regions: `%s`\n- Pose repair: `false`\n- Collision method: `adaptive_discrete_interpolation`\n- CCD: `not_available`\n- Clearance: `not_available`\n\nNo OFF, retreat, approach, reorientation, GNN, or reinforcement-learning state was added.\n" % (complete, first_unreachable, json.dumps(unresolved, ensure_ascii=False)), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
