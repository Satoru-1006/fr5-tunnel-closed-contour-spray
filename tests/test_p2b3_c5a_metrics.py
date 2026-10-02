from __future__ import annotations

import math
from pathlib import Path, PurePosixPath

import numpy as np

from src.p2b3_c5a_metrics import metric_summary, process_metrics, registration_cases, scene_rows
from src.physical_uncertainty import registration_transform
from scripts.run_p2b3_c5a_registration_metrics import (
    c1_polyline_projector, clearance_record, ros2_run_command, write_q_only,
)


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
    assert offset_metrics["position_error_m"] == [0.01, 0.01, 0.01]
    assert offset_metrics["critical_position_waypoint"] == 0
    assert offset_metrics["max_abs_stand_off_error_m"] > 0.0
    assert len(offset_metrics["stand_off_error_m"]) == 3
    assert len(offset_metrics["normal_error_rad"]) == 3


def test_process_normal_uses_c1_projected_surface_normal_and_reports_radians():
    target = np.asarray([
        [0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [2, 0, 0, 0, 0, 0, 1],
    ], dtype=float)
    normals = np.asarray([[0, 0, 1], [0, 1, 0], [0, 0, -1]], dtype=float)
    actual = np.repeat([[0.5, 0.5, 0.5]], 3, axis=0)
    wall_reference = target[:, :3] + 0.26 * normals
    _, expected_projected_normals, _ = project(actual, wall_reference, normals, False)

    rotations = []
    for z_axis in expected_projected_normals:
        reference = np.asarray([1.0, 0.0, 0.0]) if abs(z_axis[0]) < 0.9 else np.asarray([0.0, 1.0, 0.0])
        x_axis = np.cross(reference, z_axis); x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        rotations.append(np.column_stack((x_axis, y_axis, z_axis)))

    metrics = process_metrics(
        actual_position_xyz=actual, actual_rotation_matrices=np.asarray(rotations),
        target_pose_xyz_quat_xyzw=target, target_normals=normals,
        delta_transform=np.eye(4), stand_off_m=0.26, project_to_polyline=project,
    )
    assert np.allclose(metrics["projected_surface_normal_for_normal_error"], expected_projected_normals, atol=1e-15)
    assert metrics["tcp_positive_z_normal_error_rad"]["maximum_absolute"] < 1e-12
    assert metrics["max_normal_error_rad"] < 1e-12
    assert len(metrics["normal_error_rad"]) == 3
    assert metrics["critical_normal_waypoint"] == 0


def test_metric_summary_rejects_nan_and_scene_transform_keeps_exact_c4_ids():
    for invalid in (float("nan"), float("inf"), -float("inf")):
        try:
            metric_summary([invalid], "m")
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
    assert rows[0]["dx"] == 0.1 and rows[0]["dy"] == 1.1 and rows[0]["dz"] == 0.04
    assert np.isclose(rows[4]["px"], 4.001)


def test_clearance_record_separates_world_self_and_unavailable_domains():
    available = {
        "status": "AVAILABLE", "distance_m": 0.12,
        "pair": ["upperarm_link", "horseshoe_wall_026"],
        "nearest_points": [[0, 0, 0], [0.12, 0, 0]],
        "sample_index": 7, "waypoint_index": -1, "segment": "1->2",
        "segment_fraction": 0.5, "time_s": 1.0,
    }
    world = clearance_record({"minimum_robot_world_clearance": available}, "minimum_robot_world_clearance", {"horseshoe_wall_026"})
    assert world["robot_link_name"] == {"status": "AVAILABLE", "name": "upperarm_link"}
    assert world["world_object_name"] == {"status": "AVAILABLE", "name": "horseshoe_wall_026"}
    self_raw = {**available, "pair": ["wrist2_link", "forearm_link"]}
    self_result = clearance_record({"minimum_self_clearance": self_raw}, "minimum_self_clearance", {"horseshoe_wall_026"})
    assert self_result["self_link_pair"]["names"] == ["wrist2_link", "forearm_link"]
    assert "world_object_name" not in self_result
    unavailable = clearance_record(
        {"minimum_robot_world_clearance": {"status": "NOT_AVAILABLE_NO_FINITE_PAIR", "distance_m": None}},
        "minimum_robot_world_clearance", {"horseshoe_wall_026"},
    )
    assert unavailable["signed_distance_m"] is None
    assert unavailable["pair"]["status"] == "NOT_AVAILABLE_NO_FINITE_PAIR"


def test_registration_q_and_timestamps_are_copied_without_modification(tmp_path):
    rows = [{
        "t": f"{index * 0.02:.8f}",
        **{f"j{joint}_q": f"{index * 0.001 + joint * 0.01:.8f}" for joint in range(1, 7)},
        "j1_v": "99" if index == 0 else "-99",
    } for index in range(181)]
    q, times = write_q_only(tmp_path / "q.csv", rows)
    expected_times = np.asarray([float(row["t"]) for row in rows])
    expected_q = np.asarray([[float(row[f"j{i}_q"]) for i in range(1, 7)] for row in rows])
    assert np.array_equal(times, expected_times)
    assert np.array_equal(q, expected_q)


def test_native_ros2_arguments_remain_one_command_argv():
    command = ros2_run_command(
        PurePosixPath("/tmp/c5a install"), "p2b3_c5a_native",
        ["--cases", "/tmp/case inputs.csv", "--tip", "spray_tcp_link"],
    )
    assert command.startswith(
        "source '/tmp/c5a install/fairino_install/setup.bash' && "
        "source '/tmp/c5a install/bridge_install/setup.bash' && "
        "source '/tmp/c5a install/install/setup.bash' && ros2 run "
    )
    assert " && --cases" not in command
    assert command.endswith("--cases '/tmp/case inputs.csv' --tip spray_tcp_link")


def test_extracted_c1_projection_matches_independent_synthetic_projection():
    points = np.asarray([[0.4, 0.1, 0.0], [1.8, -0.2, 0.05]])
    line = np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
    normals = np.repeat([[0, 0, 1]], 3, axis=0)
    actual = c1_polyline_projector()(points, line, normals, False)
    expected = project(points, line, normals, False)
    assert all(np.allclose(a, b, atol=0.0, rtol=0.0) for a, b in zip(actual, expected))
