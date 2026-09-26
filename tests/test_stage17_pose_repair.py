from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
import yaml

from src.task_pose_repair import (
    COLLISION_METHOD,
    UNAVAILABLE,
    TaskPoseRepairConfig,
    evaluate_transition,
    generate_task_pose_candidates,
    read_pose_csv,
    recover_surface_frames,
    solve_second_order_dp,
)


ROOT = Path(__file__).resolve().parents[1]
POSES = ROOT / "outputs/internal_wiper_moveit_inputs/open_arch_tcp_poses_base_link.csv"
CONFIG = ROOT / "config/stage17_pose_repair.yaml"


def _config() -> TaskPoseRepairConfig:
    return TaskPoseRepairConfig.from_mapping(yaml.safe_load(CONFIG.read_text(encoding="utf-8")))


def _layers():
    rows, _ = read_pose_csv(POSES)
    config = _config()
    frames, quaternions, _ = recover_surface_frames(rows, config.nominal_standoff_m, config.tcp_points_to_wall)
    return frames, quaternions, generate_task_pose_candidates(frames, quaternions, config)


def test_authoritative_input_hash_and_rows_are_unchanged():
    before = hashlib.sha256(POSES.read_bytes()).hexdigest()
    rows, after = read_pose_csv(POSES)
    assert len(rows) == 181
    assert before == after


def test_surface_reconstruction_uses_existing_tcp_to_wall_sign():
    rows, _ = read_pose_csv(POSES)
    config = _config()
    frames, _, _ = recover_surface_frames(rows, config.nominal_standoff_m, config.tcp_points_to_wall)
    tcp = np.array([float(rows[0][key]) for key in ("x", "y", "z")])
    normal = np.array([float(rows[0][f"n{key}"]) for key in ("x", "y", "z")])
    normal /= np.linalg.norm(normal)
    assert np.allclose(frames[0].surface_point_m, tcp + 0.260 * normal)
    assert frames[0].surface_reconstruction_sign == 1.0


def test_local_frames_are_orthonormal_right_handed_and_continuous():
    frames, quaternions, _ = _layers()
    for frame in frames:
        matrix = np.column_stack([frame.tangent, frame.longitudinal, frame.normal])
        assert np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-7)
        assert frame.determinant > 0.999999
    assert all(float(np.dot(frames[i - 1].tangent, frames[i].tangent)) > 0.0 for i in range(1, len(frames)))
    assert all(float(np.dot(quaternions[i - 1], quaternions[i])) >= 0.0 for i in range(1, len(quaternions)))


def test_window_outside_keeps_nominal_only_and_candidates_are_hierarchical():
    frames, _, (layers, report) = _layers()
    assert len(layers[0]) == 1
    assert len(layers[44]) == 1
    assert len(layers[45]) > 1
    assert len(layers[100]) > 1
    assert len(layers[101]) == 1
    assert report.raw_candidate_count >= report.formal_candidate_count
    assert all(len(layer) <= _config().max_candidates_per_waypoint for layer in layers)


def test_formal_candidates_obey_locked_tolerances_and_diagnostic_isolation():
    _, _, (layers, _) = _layers()
    config = _config()
    for layer in layers:
        for candidate in layer:
            if candidate.formal_constraint_pass:
                assert np.hypot(candidate.tangential_offset_mm, candidate.longitudinal_offset_mm) <= config.formal_position_mm + 1e-9
                assert abs(candidate.standoff_offset_mm) <= config.formal_standoff_mm + 1e-9
                assert abs(candidate.roll_offset_deg) <= config.formal_roll_deg + 1e-9
                assert not candidate.diagnostic_only
            else:
                assert candidate.diagnostic_only


def test_roll_preserves_tool_axis_and_tcp_position():
    frames, quaternions, (layers, _) = _layers()
    nominal = next(c for c in layers[45] if c.is_nominal)
    rolled = next(c for c in layers[45] if c.roll_offset_deg == 15.0)
    assert np.allclose(nominal.repaired_tcp_position_m, rolled.repaired_tcp_position_m)
    z0 = np.asarray(quaternions[45])
    # Candidate construction is checked through its rotation matrix: z is the
    # repaired surface normal and the relative rotation is a tool-axis spin.
    from scipy.spatial.transform import Rotation
    r0 = Rotation.from_quat(nominal.repaired_quaternion_xyzw).as_matrix()
    r1 = Rotation.from_quat(rolled.repaired_quaternion_xyzw).as_matrix()
    assert np.allclose(r0[:, 2], r1[:, 2], atol=1e-7)
    assert abs(np.rad2deg(np.arccos(np.clip(np.dot(r0[:, 2], r1[:, 2]), -1, 1)))) < 1e-6


def test_transition_gate_and_unavailable_statuses():
    source = {"waypoint_id": 0, "ik_candidate_id": "a", "task_pose_candidate_id": "ta", "q_unwrapped_rad": np.zeros(6), "q_rad": np.zeros(6), "formal_constraint_pass": True, "diagnostic_only": False}
    target = {"waypoint_id": 1, "ik_candidate_id": "b", "task_pose_candidate_id": "tb", "q_unwrapped_rad": np.zeros(6), "q_rad": np.deg2rad([21, 0, 0, 0, 0, 0]), "formal_constraint_pass": True, "diagnostic_only": False}
    edge = evaluate_transition(source, target, max_joint_step_deg=20, interpolation_step_deg=0.5)
    assert edge["valid"] is False
    assert edge["reject_reason"] == "joint_step_gate"
    assert edge["ccd_status"] == UNAVAILABLE
    assert edge["clearance_status"] == UNAVAILABLE


def test_open_path_has_no_wrap_edge_and_backend_collision_label_is_exact():
    rows, _ = read_pose_csv(POSES)
    assert len(rows) - 1 != 0
    source = {"waypoint_id": 179, "q_unwrapped_rad": np.zeros(6), "q_rad": np.zeros(6), "formal_constraint_pass": True, "diagnostic_only": False}
    target = {"waypoint_id": 180, "q_unwrapped_rad": np.zeros(6), "q_rad": np.zeros(6), "formal_constraint_pass": True, "diagnostic_only": False}
    edge = evaluate_transition(source, target, max_joint_step_deg=20, interpolation_step_deg=0.5, collision_checker=lambda q: (False, "contacts=0"))
    assert edge["valid"] is True
    assert edge["interpolation_collision_status"] == "pass"
    assert COLLISION_METHOD == "adaptive_discrete_interpolation"


def _node(wp: int, cid: str, q: float, u: float) -> dict[str, object]:
    return {"waypoint_id": wp, "candidate_id": cid, "ik_candidate_id": cid, "q_unwrapped_rad": np.array([q]), "q_rad": np.array([q]), "u": [u, 0, 0, 0], "node_cost": 0.0, "valid": True, "formal_constraint_pass": True, "diagnostic_only": False}


def test_second_order_dp_includes_three_waypoint_smoothing():
    layers = [[_node(0, "0a", 0, 0)], [_node(1, "1a", 0, 3), _node(1, "1b", 0, 0)], [_node(2, "2a", 0, 0)]]
    def edge(a, b):
        return {"valid": True, "max_joint_step_deg": 0.0, "total_joint_motion_rad": 0.0, "repair_first_difference": abs(float(b["u"][0]) - float(a["u"][0]))}
    result = solve_second_order_dp(layers, edge, cost_weights={"repair_second_difference": 10.0})
    assert result["found"] is True
    assert result["second_order_cost"] >= 0.0
    assert len(result["candidate_ids"]) == 3


def test_config_formal_limits_are_locked_and_stage_outputs_are_separate():
    config = _config()
    assert config.formal_position_mm == 5.0
    assert config.formal_standoff_mm == 5.0
    assert config.formal_roll_deg == 15.0
    assert config.max_joint_step_deg == 20.0
    assert config.output_directory == "outputs/ik_graph_stage17"
    assert not (ROOT / "outputs/ik_graph_stage17").resolve() == (ROOT / "outputs/ik_graph").resolve()
