"""P2-B3-C5B0 registration specification and fail-closed semantics.

This module is deliberately independent of the C5A native evaluator.  It
defines the two coordinate conventions that C5B may compare, preserves the
frozen q(t) trajectory, and keeps engineering diagnostics separate from
physical process claims.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "p2b3-c5b0-registration-specification-v1"
BASE_FRAME = "base_link"
BASE_LEFT = "BASE_LEFT"
WORKPIECE_LOCAL = "WORKPIECE_LOCAL"
PRIMARY_PARAMETERIZATION = BASE_LEFT
WORKPIECE_ORIGIN_STATUS = "NOT_AVAILABLE"
WORKPIECE_REFERENCE_STATUS = "ENGINEERING_REFERENCE_ONLY"


def _vector(value: Sequence[float], size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name}_must_be_finite_{size}_vector")
    return result


def skew(value: Sequence[float]) -> np.ndarray:
    x, y, z = _vector(value, 3, "skew_input")
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def so3_exp(rotation_vector_rad: Sequence[float]) -> np.ndarray:
    """Rodrigues exponential for a right-handed rotation vector."""
    phi = _vector(rotation_vector_rad, 3, "rotation_vector_rad")
    theta = float(np.linalg.norm(phi))
    k = skew(phi)
    if theta < 1.0e-8:
        a = 1.0 - theta * theta / 6.0
        b = 0.5 - theta * theta / 24.0
    else:
        a = float(np.sin(theta) / theta)
        b = float((1.0 - np.cos(theta)) / (theta * theta))
    return np.eye(3) + a * k + b * (k @ k)


def se3_exp(twist: Sequence[float]) -> np.ndarray:
    """SE(3) exponential for twist ordering [rho[m], phi[rad]]."""
    xi = _vector(twist, 6, "twist")
    rho, phi = xi[:3], xi[3:]
    theta = float(np.linalg.norm(phi))
    k = skew(phi)
    if theta < 1.0e-8:
        b = 0.5 - theta * theta / 24.0
        c = 1.0 / 6.0 - theta * theta / 120.0
    else:
        b = float((1.0 - np.cos(theta)) / (theta * theta))
        c = float((theta - np.sin(theta)) / (theta * theta * theta))
    rotation = so3_exp(phi)
    v = np.eye(3) + b * k + c * (k @ k)
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = v @ rho
    return result


def se3_adjoint(transform: Sequence[Sequence[float]]) -> np.ndarray:
    transform_array = np.asarray(transform, dtype=np.float64)
    if transform_array.shape != (4, 4) or not np.isfinite(transform_array).all():
        raise ValueError("transform_must_be_finite_4x4")
    rotation = transform_array[:3, :3]
    translation = transform_array[:3, 3]
    result = np.zeros((6, 6), dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3:] = skew(translation) @ rotation
    result[3:, 3:] = rotation
    return result


def se3_inverse(transform: Sequence[Sequence[float]]) -> np.ndarray:
    transform_array = np.asarray(transform, dtype=np.float64)
    if transform_array.shape != (4, 4) or not np.isfinite(transform_array).all():
        raise ValueError("transform_must_be_finite_4x4")
    rotation = transform_array[:3, :3]
    result = np.eye(4)
    result[:3, :3] = rotation.T
    result[:3, 3] = -rotation.T @ transform_array[:3, 3]
    return result


def direct_delta(translation_m: Sequence[float], rotation_vector_rad: Sequence[float]) -> np.ndarray:
    """C5A-compatible direct base-frame translation plus Rodrigues rotation."""
    translation = _vector(translation_m, 3, "translation_m")
    result = np.eye(4)
    result[:3, :3] = so3_exp(rotation_vector_rad)
    result[:3, 3] = translation
    return result


def induced_delta(
    delta: Sequence[Sequence[float]],
    convention: str,
    workpiece_transform_base: Sequence[Sequence[float]] | None = None,
) -> np.ndarray:
    delta_array = np.asarray(delta, dtype=np.float64)
    if delta_array.shape != (4, 4) or not np.isfinite(delta_array).all():
        raise ValueError("delta_must_be_finite_4x4")
    if convention == BASE_LEFT:
        return delta_array
    if convention != WORKPIECE_LOCAL:
        raise ValueError(f"unknown_registration_convention:{convention}")
    if workpiece_transform_base is None:
        raise ValueError("WORKPIECE_LOCAL_requires_workpiece_transform")
    return (
        np.asarray(workpiece_transform_base, dtype=np.float64)
        @ delta_array
        @ se3_inverse(workpiece_transform_base)
    )


def apply_registration(
    entity_transform_base: Sequence[Sequence[float]],
    delta: Sequence[Sequence[float]],
    convention: str,
    workpiece_transform_base: Sequence[Sequence[float]] | None = None,
) -> np.ndarray:
    """Propagate one rigid transform without changing q or timestamps."""
    entity = np.asarray(entity_transform_base, dtype=np.float64)
    if entity.shape != (4, 4) or not np.isfinite(entity).all():
        raise ValueError("entity_transform_must_be_finite_4x4")
    delta_array = np.asarray(delta, dtype=np.float64)
    if delta_array.shape == (4, 4) and np.array_equal(delta_array, np.eye(4)):
        return entity.copy()
    return induced_delta(delta, convention, workpiece_transform_base) @ entity


def transform_points(
    points_base: Sequence[Sequence[float]],
    delta: Sequence[Sequence[float]],
    convention: str,
    workpiece_transform_base: Sequence[Sequence[float]] | None = None,
) -> np.ndarray:
    points = np.asarray(points_base, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points_must_be_finite_Nx3")
    transform = induced_delta(delta, convention, workpiece_transform_base)
    return (transform[:3, :3] @ points.T).T + transform[:3, 3]


def transform_normals(
    normals_base: Sequence[Sequence[float]],
    delta: Sequence[Sequence[float]],
    convention: str,
    workpiece_transform_base: Sequence[Sequence[float]] | None = None,
) -> np.ndarray:
    normals = np.asarray(normals_base, dtype=np.float64)
    if normals.ndim != 2 or normals.shape[1] != 3 or not np.isfinite(normals).all():
        raise ValueError("normals_must_be_finite_Nx3")
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths <= 1.0e-14):
        raise ValueError("normal_must_have_nonzero_length")
    rotation = induced_delta(delta, convention, workpiece_transform_base)[:3, :3]
    result = (rotation @ normals.T).T
    return result / np.linalg.norm(result, axis=1, keepdims=True)


def _quat_xyzw_to_matrix(quaternion: Sequence[float]) -> np.ndarray:
    q = _vector(quaternion, 4, "quaternion_xyzw")
    norm = float(np.linalg.norm(q))
    if norm <= 1.0e-14:
        raise ValueError("quaternion_must_have_nonzero_length")
    x, y, z, w = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _matrix_to_quat_xyzw(rotation: Sequence[Sequence[float]]) -> list[float]:
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        w, x, y, z = 0.25 * s, (matrix[2, 1] - matrix[1, 2]) / s, (matrix[0, 2] - matrix[2, 0]) / s, (matrix[1, 0] - matrix[0, 1]) / s
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        s = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
        w, x, y, z = (matrix[2, 1] - matrix[1, 2]) / s, 0.25 * s, (matrix[0, 1] + matrix[1, 0]) / s, (matrix[0, 2] + matrix[2, 0]) / s
    elif matrix[1, 1] > matrix[2, 2]:
        s = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
        w, x, y, z = (matrix[0, 2] - matrix[2, 0]) / s, (matrix[0, 1] + matrix[1, 0]) / s, 0.25 * s, (matrix[1, 2] + matrix[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
        w, x, y, z = (matrix[1, 0] - matrix[0, 1]) / s, (matrix[0, 2] + matrix[2, 0]) / s, (matrix[1, 2] + matrix[2, 1]) / s, 0.25 * s
    result = np.asarray([x, y, z, w], dtype=np.float64)
    if result[3] < 0.0:
        result = -result
    return result.tolist()


def _pose_to_matrix(pose: Mapping[str, Any]) -> np.ndarray:
    result = np.eye(4)
    result[:3, 3] = _vector(pose["position_xyz"], 3, "position_xyz")
    result[:3, :3] = _quat_xyzw_to_matrix(pose["orientation_xyzw"])
    return result


def _matrix_to_pose(transform: np.ndarray) -> dict[str, Any]:
    return {
        "position_xyz": transform[:3, 3].tolist(),
        "orientation_xyzw": _matrix_to_quat_xyzw(transform[:3, :3]),
    }


def transform_c4_scene_manifest(
    manifest: Mapping[str, Any],
    delta: Sequence[Sequence[float]],
    convention: str,
    workpiece_transform_base: Sequence[Sequence[float]] | None = None,
) -> dict[str, Any]:
    """Transform C4 boxes while preserving ordered IDs and dimensions."""
    result = deepcopy(dict(manifest))
    ordered = result["collision_objects"]["ordered"]
    for object_record in ordered:
        for primitive in object_record.get("primitives", []):
            if "pose" not in primitive:
                raise ValueError(f"scene_primitive_missing_pose:{object_record.get('id')}")
            primitive["pose"] = _matrix_to_pose(
                apply_registration(
                    _pose_to_matrix(primitive["pose"]), delta, convention, workpiece_transform_base
                )
            )
    return result


def engineering_reference_workpiece_transform(manifest: Mapping[str, Any]) -> np.ndarray:
    """Return a reproducible geometry reference, never a claimed fixture origin."""
    centers: list[np.ndarray] = []
    for object_record in manifest["collision_objects"]["ordered"]:
        for primitive in object_record.get("primitives", []):
            centers.append(_vector(primitive["pose"]["position_xyz"], 3, "scene_position"))
    if not centers:
        raise ValueError("scene_has_no_geometry_reference")
    result = np.eye(4)
    result[:3, 3] = np.mean(np.asarray(centers), axis=0)
    return result


def failure_predicate_taxonomy() -> dict[str, dict[str, Any]]:
    return {
        "F1_ROBOT_WORLD_COLLISION": {
            "predicate": "minimum_signed_robot_world_distance <= 0",
            "status": "AVAILABLE_MODEL_BASED",
            "source": "MoveIt2 PlanningScene FCL getCollisionEnvUnpadded distanceRobot",
            "sampling_label": "adaptive_discrete_interpolation",
        },
        "F2_SELF_COLLISION": {
            "predicate": "minimum_signed_self_distance <= 0",
            "status": "AVAILABLE_MODEL_BASED",
            "source": "MoveIt2 PlanningScene FCL getCollisionEnvUnpadded distanceSelf",
            "sampling_label": "adaptive_discrete_interpolation",
        },
        "F3_LOW_CLEARANCE": {
            "predicate": "NOT_AVAILABLE",
            "status": "UNRESOLVED_THRESHOLD",
            "metric": "FCL sampled signed distance may be reported without a failure threshold",
        },
        "F4_PROCESS_POSITION": {
            "predicate": "engineering_position_error > configured 6 mm gate",
            "status": "ENGINEERING_DIAGNOSTIC_ONLY",
            "source": "C1 configured geometry/path fidelity gate; not physical coating evidence",
        },
        "F5_PROCESS_NORMAL": {
            "predicate": "engineering_normal_error > configured 10 deg gate",
            "status": "ENGINEERING_DIAGNOSTIC_ONLY",
            "source": "C1 configured normal fidelity gate; not physical coating evidence",
        },
        "F6_STAND_OFF": {
            "predicate": "NOT_AVAILABLE",
            "status": "NO_AUTHENTICATED_PHYSICAL_THRESHOLD",
        },
        "F7_COATING_QUALITY": {
            "predicate": "NOT_AVAILABLE",
            "status": "NO_DEPOSITION_OR_VALIDATED_PROCESS_MODEL",
        },
    }


def margin_taxonomy() -> dict[str, dict[str, Any]]:
    return {
        "model_collision_boundary": {
            "name": "rho_collision_axis",
            "definition": "first perturbation ray point with d_robot_world <= 0",
            "status": "AVAILABLE_FOR_MODEL_BASED_C5B",
        },
        "engineering_diagnostic_boundary": {
            "name": "rho_position_diag / rho_normal_diag",
            "definition": "first ray point crossing a configured C1 diagnostic gate",
            "status": "AVAILABLE_WITH_ENGINEERING_DIAGNOSTIC_LABEL",
        },
        "physical_robustness_margin": {
            "name": "PHYSICAL_REGISTRATION_MARGIN",
            "definition": "requires measured, authenticated manufacturer, validated calibration, or validated process evidence",
            "status": "NOT_AVAILABLE",
        },
    }


def discrete_convergence_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize sampled-resolution bookkeeping without calling it CCD."""
    if not records:
        raise ValueError("convergence_records_empty")
    distances = [float(record["min_robot_world_signed_distance_m"]) for record in records]
    if not np.isfinite(distances).all():
        raise ValueError("convergence_distance_nonfinite")
    pairs = [tuple(record["pair"]) for record in records]
    segments = [str(record["segment"]) for record in records]
    return {
        "status": "PASS",
        "method": "adaptive_discrete_interpolation",
        "resolutions": [str(record["resolution"]) for record in records],
        "sample_counts": [int(record["sample_count"]) for record in records],
        "min_distance_m": distances,
        "max_abs_change_from_first_m": float(np.max(np.abs(np.asarray(distances) - distances[0]))),
        "nearest_pair_stable": len(set(pairs)) == 1,
        "critical_segment_stable": len(set(segments)) == 1,
        "collision_counts": [int(record["collision_samples"]) for record in records],
        "strict_continuous_collision_detection": "NOT_AVAILABLE",
    }


def specification_summary() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "base_frame": BASE_FRAME,
        "primary_parameterization": PRIMARY_PARAMETERIZATION,
        "conventions": {
            BASE_LEFT: {
                "equation": "T_base_entity_actual = DeltaT_base @ T_base_entity_nominal",
                "translation_frame": BASE_FRAME,
                "rotation_frame": BASE_FRAME,
                "rotation_origin": "base_link origin",
                "twist_units": "[m,m,m,rad,rad,rad]",
                "composition_side": "left",
            },
            WORKPIECE_LOCAL: {
                "equation": "T_base_workpiece_actual = T_base_workpiece_nominal @ Exp(xi_workpiece^)",
                "translation_frame": "workpiece frame",
                "rotation_frame": "workpiece frame",
                "rotation_origin": "declared workpiece origin; current physical origin NOT_AVAILABLE",
                "twist_units": "[m,m,m,rad,rad,rad]",
                "composition_side": "right/body",
            },
        },
        "workpiece_origin_status": WORKPIECE_ORIGIN_STATUS,
        "engineering_reference_status": WORKPIECE_REFERENCE_STATUS,
        "frozen_state_invariants": ["q unchanged", "timestamps unchanged", "C4 IDs and dimensions unchanged"],
        "failure_predicates": failure_predicate_taxonomy(),
        "margin_taxonomy": margin_taxonomy(),
    }
