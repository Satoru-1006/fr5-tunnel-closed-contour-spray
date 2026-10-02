"""Pure registration and process-metric calculations for P2-B3-C5A."""

from __future__ import annotations

import math
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from src.physical_uncertainty import (
    registration_transform,
    transform_pose_array_xyzw,
    transform_surface_normals,
)


def registration_cases() -> list[dict[str, Any]]:
    """Return the frozen zero, small-axis, and controlled larger diagnostic set."""
    cases: list[dict[str, Any]] = [{
        "case_id": "registration_zero", "family": "zero",
        "translation_base_m": [0.0, 0.0, 0.0], "rotation_vector_base_rad": [0.0, 0.0, 0.0],
        "magnitude_label": "identity",
    }]
    axes = "xyz"
    for magnitude, label in ((0.001, "1mm"), (0.005, "5mm")):
        for axis, axis_name in enumerate(axes):
            for sign in (-1.0, 1.0):
                vector = [0.0, 0.0, 0.0]; vector[axis] = sign * magnitude
                cases.append({
                    "case_id": f"translation_{label}_{axis_name}_{'minus' if sign < 0 else 'plus'}",
                    "family": "translation", "translation_base_m": vector,
                    "rotation_vector_base_rad": [0.0, 0.0, 0.0], "magnitude_label": label,
                })
    for magnitude, label in ((math.radians(0.1), "0p1deg"), (math.radians(0.5), "0p5deg")):
        for axis, axis_name in enumerate(axes):
            for sign in (-1.0, 1.0):
                vector = [0.0, 0.0, 0.0]; vector[axis] = sign * magnitude
                cases.append({
                    "case_id": f"rotation_{label}_{axis_name}_{'minus' if sign < 0 else 'plus'}",
                    "family": "rotation", "translation_base_m": [0.0, 0.0, 0.0],
                    "rotation_vector_base_rad": vector, "magnitude_label": label,
                })
    if len(cases) != 25 or len({case["case_id"] for case in cases}) != 25:
        raise RuntimeError("C5A_frozen_case_set_invalid")
    return cases


def metric_summary(values: Sequence[float], unit: str) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not array.size or not np.isfinite(array).all():
        raise ValueError("metric_values_must_be_finite_nonempty_vector")
    abs_array = np.abs(array)
    return {
        "sample_count": int(array.size),
        "maximum_signed": float(np.max(array)),
        "minimum_signed": float(np.min(array)),
        "maximum_absolute": float(np.max(abs_array)),
        "maximum_absolute_index": int(np.argmax(abs_array)),
        "root_mean_square": float(np.sqrt(np.mean(array * array))),
        "p95_absolute": float(np.quantile(abs_array, 0.95)),
        "unit": unit,
    }


def process_metrics(
    *,
    actual_position_xyz: Sequence[Sequence[float]],
    actual_rotation_matrices: Sequence[Sequence[Sequence[float]]],
    target_pose_xyz_quat_xyzw: Sequence[Sequence[float]],
    target_normals: Sequence[Sequence[float]],
    delta_transform: Sequence[Sequence[float]],
    stand_off_m: float,
    project_to_polyline: Callable[[np.ndarray, np.ndarray, np.ndarray, bool], tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    """Evaluate C1 position, TCP +Z normal, and projected stand-off semantics."""
    actual_position = np.asarray(actual_position_xyz, dtype=np.float64)
    actual_rotation = np.asarray(actual_rotation_matrices, dtype=np.float64)
    target = np.asarray(target_pose_xyz_quat_xyzw, dtype=np.float64)
    normals = np.asarray(target_normals, dtype=np.float64)
    if (actual_position.ndim != 2 or actual_position.shape[1] != 3 or
            actual_rotation.shape != (len(actual_position), 3, 3) or
            target.shape != (len(actual_position), 7) or normals.shape != (len(actual_position), 3) or
            not all(np.isfinite(array).all() for array in (actual_position, actual_rotation, target, normals))):
        raise ValueError("C5A_process_metric_shape_or_finite_check_failed")
    if not math.isfinite(float(stand_off_m)) or stand_off_m <= 0.0:
        raise ValueError("C1_stand_off_must_be_positive_and_finite")
    moved_target = transform_pose_array_xyzw(target, delta_transform)
    moved_normals = transform_surface_normals(normals, delta_transform)
    normal_lengths = np.linalg.norm(moved_normals, axis=1)
    if np.any(normal_lengths <= 1.0e-12):
        raise ValueError("C5A_transformed_surface_normal_zero")
    moved_normals = moved_normals / normal_lengths[:, None]

    position_errors = np.linalg.norm(actual_position - moved_target[:, :3], axis=1)
    tool_z = actual_rotation[:, :, 2]
    tool_z_lengths = np.linalg.norm(tool_z, axis=1)
    if np.any(tool_z_lengths <= 1.0e-12):
        raise ValueError("C5A_actual_tcp_positive_z_axis_invalid")
    tool_z = tool_z / tool_z_lengths[:, None]
    normal_dot = np.clip(np.einsum("ij,ij->i", tool_z, moved_normals), -1.0, 1.0)
    normal_error_deg = np.degrees(np.arccos(normal_dot))

    wall_reference = moved_target[:, :3] + float(stand_off_m) * moved_normals
    projected_wall, projected_normals, projection_distance = project_to_polyline(
        actual_position, wall_reference, moved_normals, False,
    )
    standoff_error = np.einsum("ij,ij->i", projected_wall - actual_position, projected_normals) - float(stand_off_m)
    if not all(np.isfinite(array).all() for array in (position_errors, normal_error_deg, standoff_error, projection_distance)):
        raise ValueError("C5A_process_metric_result_nonfinite")
    return {
        "tcp_position_error": metric_summary(position_errors, "m"),
        "tcp_positive_z_normal_error": metric_summary(normal_error_deg, "deg"),
        "projected_standoff_error": metric_summary(standoff_error, "m"),
        "projected_wall_distance": metric_summary(projection_distance, "m"),
        "target_pose_transform_max_position_delta_m": float(np.max(np.linalg.norm(moved_target[:, :3] - target[:, :3], axis=1))),
        "surface_normal_transform_max_delta": float(np.max(np.linalg.norm(moved_normals - normals / np.linalg.norm(normals, axis=1)[:, None], axis=1))),
        "stand_off_m": float(stand_off_m),
        "stand_off_source": "C1 authoritative STAND_OFF=0.260 and _project_to_polyline_with_normals(closed=False)",
        "tcp_axis_convention": "actual FK rotation matrix column +Z compared directly with transformed raw open_arch_tcp_poses_base_link.csv nx,ny,nz",
    }


def scene_rows(manifest: Mapping[str, Any], delta_transform: Sequence[Sequence[float]]) -> list[dict[str, Any]]:
    """Transform the exact C4 manifest boxes by DeltaT_base left composition."""
    objects = manifest.get("collision_objects", {}).get("ordered")
    if not isinstance(objects, list) or len(objects) != 181:
        raise ValueError("C4_scene_manifest_must_have_181_ordered_objects")
    rows: list[dict[str, Any]] = []
    for obj in objects:
        if obj.get("frame_id") != "base_link" or len(obj.get("primitives", [])) != 1:
            raise ValueError("C4_scene_object_frame_or_primitive_count_mismatch")
        primitive = obj["primitives"][0]
        if primitive.get("type") != "BOX" or len(primitive.get("dimensions", [])) != 3:
            raise ValueError("C5A_requires_manifest_box_geometry")
        pose = primitive["pose"]
        pose_array = np.asarray([[
            *pose["position_xyz"], *pose["orientation_xyzw"],
        ]], dtype=np.float64)
        moved_pose = transform_pose_array_xyzw(pose_array, delta_transform)[0]
        rows.append({
            "id": str(obj["id"]), "dx": float(primitive["dimensions"][0]),
            "dy": float(primitive["dimensions"][1]), "dz": float(primitive["dimensions"][2]),
            "px": float(moved_pose[0]), "py": float(moved_pose[1]), "pz": float(moved_pose[2]),
            "qx": float(moved_pose[3]), "qy": float(moved_pose[4]), "qz": float(moved_pose[5]), "qw": float(moved_pose[6]),
        })
    if len({row["id"] for row in rows}) != 181:
        raise ValueError("C4_scene_object_ids_not_unique")
    return rows
