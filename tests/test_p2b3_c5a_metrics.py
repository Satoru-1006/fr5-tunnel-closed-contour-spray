from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from src.p2b3_c5a_metrics import metric_summary, process_metrics, registration_cases, scene_rows
from src.physical_uncertainty import registration_transform
from scripts.run_p2b3_c5a_registration_metrics import ros2_run_command


def project(points: np.ndarray, line: np.ndarray, normals: np.ndarray, closed: bool = True):
    if closed:
        raise AssertionError("C1 open-arch projection must use closed=False")
    starts, ends = line[:-1], line[1:]
    n_start, n_end = normals[:-1], normals[1:]
    best_d2 = np.full(len(points), np.inf)
    best_projection = np.zeros_like(points)
    best_normal = np.zeros_like(points)
    for a, b, na, nb in zip(starts, ends, n_start, n_end):
        edge = b - a
        denom = max(float(np.dot(edge, edge)), 1e-15)
        alpha = np.clip(((points - a) @ edge) / denom, 0.0, 1.0)
        projection = a + alpha[:, None] * edge
        d2 = np.sum((points - projection) ** 2, axis=1)
        choose = d2 < best_d2
        normal = (1 - alpha[:, None]) * na + alpha[:, None] * nb
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
        best_d2[choose], best_projection[choose], best_normal[choose] = d2[choose], projection[choose], normal[choose]
    return best_projection, best_normal, np.sqrt(best_d2)


def test_frozen_case_set_has_25_zero_small_and_large_controls():
    cases = registration_cases()
    assert len(cases) == 25
    assert cases[0]["case_id"] == "registration_zero"
    assert sum(case["magnitude_label"] == "1mm" for case in cases) == 6
    assert sum(case["magnitude_label"] == "0p1deg" for case in cases) == 6
    assert sum(case["magnitude_label"] == "5mm" for case in cases) == 6
    assert sum(case["magnitude_label"] == "0p5deg" for case in cases) == 6


def test_transform_known_answers_positive_negative_translation_and_rotation():
    poses = np.asarray([[0.25, -0.1, 0.5, 0.0, 0.0, 0.0, 1.0]])
    normals = np.asarray([[0.0, 0.0, 1.0]])
    for sign in (-1.0, 1.0):
        delta = registration_transform([sign * 0.001, 0, 0], [0, 0, 0])
        from src.physical_uncertainty import transform_pose_array_xyzw, transform_surface_normals
        moved = transform_pose_array_xyzw(poses, delta)
        assert np.allclose(moved[0, :3], poses[0, :3] + [sign * 0.001, 0, 0], atol=1e-15)
        assert np.array_equal(transform_surface_normals(normals, delta), normals)
    for sign in (-1.0, 1.0):
        delta = registration_transform([0, 0, 0], [0, sign * math.radians(0.1), 0])
        from src.physical_uncertainty import transform_surface_normals
        moved = transform_surface_normals(normals, delta)
        assert np.allclose(moved[0], [sign * math.sin(math.radians(0.1)), 0, math.cos(math.radians(0.1))], atol=1e-12)


def test_process_position_normal_and_standoff_known_answers():
    target = np.asarray([[0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 1], [2, 0, 0, 0, 0, 0, 1]], dtype=float)
    rotations = np.repeat(np.eye(3)[None, :, :], 3, axis=0)
    normals = np.repeat([[0, 0, 1]], 3, axis=0)
    actual = target[:, :3].copy()
    metrics = process_metrics(
        actual_position_xyz=actual, actual_rotation_matrices=rotations,
        target_pose_xyz_quat_xyzw=target, target_normals=normals,
        delta_transform=np.eye(4), stand_off_m=0.26, project_to_polyline=project,
    )
    assert metrics["tcp_position_error"]["maximum_absolute"] == 0.0
    assert metrics["tcp_positive_z_normal_error"]["maximum_absolute"] == 0.0
    assert metrics["projected_standoff_error"]["maximum_absolute"] < 1e-12
    actual[:, 2] += 0.01
    offset_metrics = process_metrics(
        actual_position_xyz=actual, actual_rotation_matrices=rotations,
        target_pose_xyz_quat_xyzw=target, target_normals=normals,
        delta_transform=np.eye(4), stand_off_m=0.26, project_to_polyline=project,
    )
    assert np.isclose(offset_metrics["tcp_position_error"]["maximum_absolute"], 0.01)
    assert np.isclose(offset_metrics["projected_standoff_error"]["maximum_signed"], -0.01)


def test_metric_summary_rejects_nan_and_scene_transform_keeps_exact_c4_ids():
    try:
        metric_summary([float("nan")], "m")
    except ValueError:
        pass
    else:
        raise AssertionError("nonfinite result was accepted")
    objects = [{
        "frame_id": "base_link", "id": f"horseshoe_wall_{i:03d}",
        "primitives": [{"type": "BOX", "dimensions": [0.1, 1.1, 0.04], "pose": {
            "position_xyz": [float(i), 0, 0], "orientation_xyzw": [0, 0, 0, 1],
        }}],
    } for i in range(181)]
    rows = scene_rows({"collision_objects": {"ordered": objects}}, registration_transform([0.001, 0, 0], [0, 0, 0]))
    assert len(rows) == 181
    assert rows[0]["id"] == "horseshoe_wall_000" and rows[-1]["id"] == "horseshoe_wall_180"
    assert np.isclose(rows[4]["px"], 4.001)


def test_native_ros2_arguments_remain_one_command_argv():
    command = ros2_run_command(
        Path("/tmp/c5a install"), "p2b3_c5a_native",
        ["--cases", "/tmp/case inputs.csv", "--tip", "spray_tcp_link"],
    )
    assert command.startswith("source '/tmp/c5a install/install/setup.bash' && ros2 run ")
    assert " && --cases" not in command
    assert command.endswith("--cases '/tmp/case inputs.csv' --tip spray_tcp_link")
