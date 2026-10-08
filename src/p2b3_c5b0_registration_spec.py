"""P2-B3-C5B0 registration specification and fail-closed semantics.

This module is deliberately independent of the C5A native evaluator.  It
defines the two coordinate conventions that C5B may compare, preserves the
frozen q(t) trajectory, and keeps engineering diagnostics separate from
physical process claims.
"""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "p2b3-c5b0-registration-specification-v1"
BASE_FRAME = "base_link"
BASE_LEFT = "BASE_LEFT"
WORKPIECE_LOCAL = "WORKPIECE_LOCAL"
PRIMARY_PARAMETERIZATION = BASE_LEFT
WORKPIECE_ORIGIN_STATUS = "NOT_AVAILABLE"
WORKPIECE_REFERENCE_STATUS = "ENGINEERING_REFERENCE_ONLY"

# One validation policy is used at every public homogeneous-transform entry
# point.  These are numerical representation tolerances, not physical safety
# margins and must not be used to project an invalid matrix onto SE(3).
SE3_ORTHOGONAL_TOL = 1.0e-9
SE3_DETERMINANT_TOL = 1.0e-9
SE3_HOMOGENEOUS_ROW_TOL = 1.0e-12


def _vector(value: Sequence[float], size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name}_must_be_finite_{size}_vector")
    return result


def skew(value: Sequence[float]) -> np.ndarray:
    x, y, z = _vector(value, 3, "skew_input")
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def _validate_se3(transform: Sequence[Sequence[float]], name: str) -> np.ndarray:
    """Validate, but never repair, a finite proper rigid transform."""
    result = np.asarray(transform, dtype=np.float64)
    if result.shape != (4, 4) or not np.isfinite(result).all():
        raise ValueError(f"{name}_must_be_finite_4x4")
    if not np.allclose(
        result[3, :], np.asarray([0.0, 0.0, 0.0, 1.0]),
        atol=SE3_HOMOGENEOUS_ROW_TOL, rtol=0.0,
    ):
        raise ValueError(f"{name}_invalid_homogeneous_row")
    rotation = result[:3, :3]
    if not np.allclose(
        rotation.T @ rotation, np.eye(3),
        atol=SE3_ORTHOGONAL_TOL, rtol=0.0,
    ):
        raise ValueError(f"{name}_rotation_not_orthogonal")
    determinant = float(np.linalg.det(rotation))
    if not np.isclose(determinant, 1.0, atol=SE3_DETERMINANT_TOL, rtol=0.0):
        raise ValueError(f"{name}_rotation_determinant_not_positive_one")
    return result


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
    transform_array = _validate_se3(transform, "transform")
    rotation = transform_array[:3, :3]
    translation = transform_array[:3, 3]
    result = np.zeros((6, 6), dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3:] = skew(translation) @ rotation
    result[3:, 3:] = rotation
    return result


def se3_inverse(transform: Sequence[Sequence[float]]) -> np.ndarray:
    transform_array = _validate_se3(transform, "transform")
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
    delta_array = _validate_se3(delta, "delta")
    if convention == BASE_LEFT:
        return delta_array
    if convention != WORKPIECE_LOCAL:
        raise ValueError(f"unknown_registration_convention:{convention}")
    if workpiece_transform_base is None:
        raise ValueError("WORKPIECE_LOCAL_requires_workpiece_transform")
    workpiece = _validate_se3(workpiece_transform_base, "workpiece_transform_base")
    return (
        workpiece
        @ delta_array
        @ se3_inverse(workpiece)
    )


def apply_registration(
    entity_transform_base: Sequence[Sequence[float]],
    delta: Sequence[Sequence[float]],
    convention: str,
    workpiece_transform_base: Sequence[Sequence[float]] | None = None,
) -> np.ndarray:
    """Propagate one rigid transform without changing q or timestamps."""
    entity = _validate_se3(entity_transform_base, "entity_transform")
    delta_array = _validate_se3(delta, "delta")
    # Validate convention and the optional frame even for identity.  The
    # identity shortcut is numerical only; it is not an input-validation path.
    induced = induced_delta(delta_array, convention, workpiece_transform_base)
    if np.array_equal(delta_array, np.eye(4)):
        return entity.copy()
    return induced @ entity


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


def _resolution_step_rad(value: Any) -> float:
    """Parse an explicitly unit-labelled angular sampling resolution."""
    if isinstance(value, bool):
        raise ValueError("resolution_must_be_positive")
    if isinstance(value, (int, float, np.integer, np.floating)):
        step = float(value)
    elif isinstance(value, str):
        match = re.fullmatch(
            r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(deg|rad)\s*",
            value,
        )
        if match is None:
            raise ValueError("resolution_must_have_numeric_value_and_unit")
        step = float(match.group(1))
        if match.group(2) == "deg":
            step = float(np.deg2rad(step))
    else:
        raise ValueError("resolution_must_have_numeric_value_and_unit")
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("resolution_must_be_positive")
    return step


def _convergence_observation(record: Mapping[str, Any], index: int) -> dict[str, Any]:
    required = {
        "resolution", "sample_count", "min_robot_world_signed_distance_m",
        "pair", "segment", "collision_samples",
    }
    missing = sorted(required.difference(record))
    if missing:
        raise ValueError(f"record_{index}_missing_fields:{','.join(missing)}")
    sample_count = record["sample_count"]
    collision_samples = record["collision_samples"]
    if isinstance(sample_count, bool) or not isinstance(sample_count, (int, np.integer)) or sample_count <= 0:
        raise ValueError(f"record_{index}_sample_count_must_be_positive_integer")
    if isinstance(collision_samples, bool) or not isinstance(collision_samples, (int, np.integer)):
        raise ValueError(f"record_{index}_collision_samples_must_be_integer")
    if collision_samples < 0 or collision_samples > sample_count:
        raise ValueError(f"record_{index}_collision_samples_out_of_range")
    distance = float(record["min_robot_world_signed_distance_m"])
    if not np.isfinite(distance):
        raise ValueError(f"record_{index}_distance_must_be_finite")
    pair = record["pair"]
    if isinstance(pair, (str, bytes)) or not isinstance(pair, Sequence) or len(pair) != 2 or not all(str(item) for item in pair):
        raise ValueError(f"record_{index}_pair_must_be_two_nonempty_names")
    segment = record["segment"]
    if not isinstance(segment, str) or not segment.strip():
        raise ValueError(f"record_{index}_segment_must_be_nonempty_string")
    return {
        "resolution": str(record["resolution"]),
        "resolution_step_rad": _resolution_step_rad(record["resolution"]),
        "sample_count": int(sample_count),
        "min_robot_world_signed_distance_m": distance,
        "pair": [str(pair[0]), str(pair[1])],
        "segment": segment,
        "collision_samples": int(collision_samples),
    }


def discrete_convergence_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    distance_tolerance_m: float | None = None,
) -> dict[str, Any]:
    """Summarize sampled-resolution evidence without calling it CCD.

    A convergence claim requires three coarse-to-fine records, strictly
    increasing sample counts, and a caller-declared numerical tolerance.  No
    tolerance is inferred from the observed values, because doing so would
    make the acceptance criterion self-fulfilling.  Pair and segment changes
    are retained as independent diagnostics rather than being conflated with
    the scalar distance criterion.
    """
    result: dict[str, Any] = {
        "status": "INSUFFICIENT_EVIDENCE",
        "method": "adaptive_discrete_interpolation",
        "resolutions": [],
        "resolution_steps_rad": [],
        "sample_counts": [],
        "min_distance_m": [],
        "raw_observations": [],
        "distance_changes_m": [],
        "max_abs_change_from_first_m": None,
        "max_abs_change_between_adjacent_m": None,
        "distance_tolerance_m": None,
        "distance_converged": None,
        "nearest_pair_stable": None,
        "critical_segment_stable": None,
        "collision_counts": [],
        "collision_status_stable": None,
        "strict_continuous_collision_detection": "NOT_AVAILABLE",
        "validation_errors": [],
    }
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)) or not records:
        result["validation_errors"].append("convergence_records_empty")
        return result

    observations: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            result["validation_errors"].append(f"record_{index}_must_be_mapping")
            continue
        try:
            observation = _convergence_observation(record, index)
        except (TypeError, ValueError, OverflowError) as error:
            result["validation_errors"].append(str(error))
            continue
        observations.append(observation)
    result["raw_observations"] = observations
    result["resolutions"] = [item["resolution"] for item in observations]
    result["resolution_steps_rad"] = [item["resolution_step_rad"] for item in observations]
    result["sample_counts"] = [item["sample_count"] for item in observations]
    result["min_distance_m"] = [item["min_robot_world_signed_distance_m"] for item in observations]
    result["collision_counts"] = [item["collision_samples"] for item in observations]

    if len(observations) < 3:
        result["validation_errors"].append("at_least_three_refinements_required")
    if distance_tolerance_m is None:
        result["validation_errors"].append("distance_tolerance_not_declared")
    else:
        try:
            tolerance = float(distance_tolerance_m)
        except (TypeError, ValueError, OverflowError):
            tolerance = float("nan")
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            result["validation_errors"].append("distance_tolerance_must_be_positive_and_finite")
        else:
            result["distance_tolerance_m"] = tolerance

    if observations:
        steps = np.asarray(result["resolution_steps_rad"], dtype=np.float64)
        counts = np.asarray(result["sample_counts"], dtype=np.int64)
        if len(observations) > 1 and not np.all(np.diff(steps) < 0.0):
            result["validation_errors"].append("resolutions_must_be_strictly_coarse_to_fine")
        if len(observations) > 1 and not np.all(np.diff(counts) > 0):
            result["validation_errors"].append("sample_counts_must_increase_with_refinement")
        distances = np.asarray(result["min_distance_m"], dtype=np.float64)
        pairs = [tuple(item["pair"]) for item in observations]
        segments = [item["segment"] for item in observations]
        collision_status = [item["collision_samples"] > 0 for item in observations]
        result["nearest_pair_stable"] = len(set(pairs)) == 1
        result["critical_segment_stable"] = len(set(segments)) == 1
        result["collision_status_stable"] = len(set(collision_status)) == 1
        if len(distances) > 1:
            changes = np.abs(np.diff(distances))
            result["distance_changes_m"] = changes.tolist()
            result["max_abs_change_between_adjacent_m"] = float(np.max(changes))
            result["max_abs_change_from_first_m"] = float(np.max(np.abs(distances - distances[0])))
            if result["distance_tolerance_m"] is not None:
                result["distance_converged"] = bool(np.all(changes <= result["distance_tolerance_m"]))

    if not result["validation_errors"]:
        if result["distance_converged"] and result["collision_status_stable"]:
            result["status"] = "CONVERGED"
        else:
            result["status"] = "NOT_CONVERGED"
    return result


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
