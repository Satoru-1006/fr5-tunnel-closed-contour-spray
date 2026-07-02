from __future__ import annotations

import numpy as np

from src.path_planner import generate_closed_horseshoe_contour_path
from src.metrics import detect_jerk_spikes, polyline_distance, project_points_to_polyline, tcp_speed_from_positions
from src.robot_model import FR5Robot
from src.robot_model import JointLimits
from src.time_parameterization import make_closed_loop_ruckig_time_profile


def test_closed_horseshoe_is_c2_fixed_station_and_exact_offset() -> None:
    path = generate_closed_horseshoe_contour_path(
        R=0.60,
        H=0.50,
        y0=0.30,
        stand_off=0.18,
        n_loops=3,
        samples_per_loop=240,
        fillet_radius=0.20,
    )
    one_wall = path.surface_points[:240]
    one_tcp = path.tcp_points[:240]
    one_normal = path.normals[:240]

    assert np.std(path.surface_points[:, 1]) < 1e-12
    assert np.std(path.tcp_points[:, 1]) < 1e-12
    offset = one_tcp - one_wall
    assert np.allclose(np.linalg.norm(offset, axis=1), 0.18, atol=1e-10)
    assert np.allclose(np.sum(offset * one_normal, axis=1), 0.18, atol=1e-10)
    assert np.allclose(path.tcp_points[:240], path.tcp_points[240:480], atol=1e-12)
    assert np.allclose(path.tcp_points[:240], path.tcp_points[480:720], atol=1e-12)

    adjacent_dot = np.sum(one_normal * np.roll(one_normal, 1, axis=0), axis=1)
    max_normal_step = np.rad2deg(np.max(np.arccos(np.clip(adjacent_dot, -1.0, 1.0))))
    assert max_normal_step < 6.0


def test_periodic_ruckig_profile_is_continuous_and_limit_compliant() -> None:
    samples = 80
    loops = 3
    phase = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
    one_path = np.column_stack([0.4 * np.cos(phase), np.zeros(samples), 0.3 * np.sin(phase)])
    one_q = np.column_stack(
        [
            0.3 * np.sin(phase + shift)
            for shift in np.linspace(0.0, np.pi, 6, endpoint=False)
        ]
    )
    path = np.vstack([one_path] * loops)
    q = np.vstack([one_q] * loops)
    limits = JointLimits(
        q_min=np.full(6, -2.0),
        q_max=np.full(6, 2.0),
        dq_max=np.full(6, 1.0),
        ddq_max=np.full(6, 2.0),
        jerk_max=np.full(6, 8.0),
    )

    profile = make_closed_loop_ruckig_time_profile(
        path,
        q,
        limits,
        target_tcp_speed=0.08,
        dt=0.02,
        joint_smoothness=0.0,
    )

    assert profile.method.startswith("ruckig-periodic-c4:")
    assert np.all(np.diff(profile.t) > 0.0)
    assert np.max(np.abs(profile.dq)) <= np.max(limits.dq_max) + 1e-7
    assert np.max(np.abs(profile.ddq)) <= np.max(limits.ddq_max) + 1e-7
    assert np.max(np.abs(profile.jerk)) <= np.max(limits.jerk_max) + 1e-7
    assert np.min(profile.tcp_speed) > 0.0


def test_jerk_outlier_is_not_hidden_by_a_high_absolute_limit() -> None:
    jerk = np.zeros((200, 6))
    jerk[:, 3] = 0.01
    jerk[100, 3] = 2.0
    indices, details = detect_jerk_spikes(jerk, jerk_limits=np.full(6, 8.0))

    assert 100 in indices
    assert details[0]["joint"] == 4
    assert "max/p95" in str(details[0]["reason"])


def test_polyline_distance_uses_segment_projection_for_closed_contours() -> None:
    square = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )
    points = np.array(
        [
            [0.5, 0.0, 0.2],
            [0.0, 0.0, 0.5],
        ]
    )
    distances = polyline_distance(points, square, closed=True)

    assert np.allclose(distances, [0.2, 0.0])


def test_project_points_to_polyline_interpolates_normals_on_segments() -> None:
    line = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
        ]
    )
    normals = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )
    point = np.array([[0.5, 0.0, 0.2]])
    projected, projected_normals, distance = project_points_to_polyline(point, line, normals, closed=False)

    expected_normal = np.array([1.0, 0.0, 1.0])
    expected_normal /= np.linalg.norm(expected_normal)
    assert np.allclose(projected, [[0.5, 0.0, 0.0]])
    assert projected_normals is not None
    assert np.allclose(projected_normals[0], expected_normal)
    assert np.allclose(distance, [0.2])


def test_tcp_speed_from_fk_positions_has_no_gradient_endpoint_spike() -> None:
    t = np.array([0.0, 0.5, 1.0, 1.5])
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.3, 0.0, 0.0],
        ]
    )
    speed = tcp_speed_from_positions(points, t)

    assert np.allclose(speed, 0.2)


def test_default_ik_includes_tool_axis_orientation_constraint() -> None:
    robot = FR5Robot.placeholder()
    q_target = np.array([0.35, -0.70, 0.90, 0.25, -0.45, 0.10])
    pose = robot.fk(q_target)
    target_normal = pose[:3, 2]
    q, _, ok = robot.ik(pose, np.zeros(6))
    actual_tool_z = robot.fk(q)[:3, 2]
    angle = np.rad2deg(np.arccos(np.clip(np.dot(actual_tool_z, target_normal), -1.0, 1.0)))

    assert ok
    assert angle < 5.0


def test_trajectory_ik_enforces_single_step_limit() -> None:
    robot = FR5Robot.placeholder()
    q_start = np.zeros(6)
    q_far = np.array([1.0, -0.9, 0.8, 0.7, -0.6, 0.5])
    poses = [robot.fk(q_start), robot.fk(q_far)]
    q, _, ok = robot.solve_trajectory_ik(poses, q0=q_start, max_step_deg=5.0)

    assert np.max(np.abs(np.diff(q, axis=0))) <= np.deg2rad(5.0) + 1e-9
    assert not ok[1]
