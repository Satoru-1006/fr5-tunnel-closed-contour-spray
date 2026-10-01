"""Physical uncertainty semantics and base/workpiece SE(3) propagation.

This module separates software-defined target perturbations from physical
uncertainty sources.  It contains no robot model, collision backend, or
hardware claims; native PlanningScene propagation is performed by the bounded
C3 smoke runner.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


CONTRACT_SCHEMA = "p2b3-c3-uncertainty-semantics-v1"
BASE_FRAME = "base_link"
COMPOSITION_CONVENTION = "T_base_entity_actual = DeltaT_base_left @ T_base_entity_nominal"


class UncertaintySource(str, Enum):
    MEASURED = "MEASURED"
    MANUFACTURER = "MANUFACTURER"
    PROJECT_CONFIG = "PROJECT_CONFIG"
    LITERATURE = "LITERATURE"
    ASSUMED = "ASSUMED"
    UNKNOWN = "UNKNOWN"


class SemanticClass(str, Enum):
    JOINT_STATE_ERROR = "JOINT_STATE_ERROR"
    TARGET_POSE_GEOMETRIC_PERTURBATION = "TARGET_POSE_GEOMETRIC_PERTURBATION"
    TCP_CALIBRATION_ERROR = "TCP_CALIBRATION_ERROR"
    BASE_WORKPIECE_REGISTRATION_ERROR = "BASE_WORKPIECE_REGISTRATION_ERROR"
    SURFACE_RECONSTRUCTION_ERROR = "SURFACE_RECONSTRUCTION_ERROR"
    EXECUTION_TRACKING_ERROR = "EXECUTION_TRACKING_ERROR"


_REQUIRED_FIELDS = frozenset(
    {
        "uncertainty_id",
        "semantic_class",
        "physical_source",
        "perturbed_entity",
        "fixed_entities",
        "coordinate_frame",
        "composition_convention",
        "units",
        "changes_joint_state",
        "changes_target_pose",
        "changes_surface_geometry",
        "changes_surface_normals",
        "changes_collision_world",
        "requires_reik",
        "requires_retiming",
        "requires_hardware_data",
        "physical_interpretation",
        "capability_status",
        "normalization_bound",
        "normalization_bound_source",
        "normalization_status",
    }
)


@dataclass(frozen=True)
class UncertaintySemantics:
    uncertainty_id: str
    semantic_class: SemanticClass
    physical_source: UncertaintySource
    perturbed_entity: str
    fixed_entities: tuple[str, ...]
    coordinate_frame: str
    composition_convention: str
    units: str
    changes_joint_state: bool
    changes_target_pose: bool
    changes_surface_geometry: bool
    changes_surface_normals: bool
    changes_collision_world: bool
    requires_reik: bool
    requires_retiming: bool
    requires_hardware_data: bool
    physical_interpretation: str
    capability_status: str
    normalization_bound: float | tuple[float, ...] | None
    normalization_bound_source: UncertaintySource
    normalization_status: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "semantic_class", SemanticClass(self.semantic_class))
        object.__setattr__(self, "physical_source", UncertaintySource(self.physical_source))
        object.__setattr__(self, "normalization_bound_source", UncertaintySource(self.normalization_bound_source))
        if not self.uncertainty_id.strip() or not self.perturbed_entity.strip():
            raise ValueError("uncertainty identity and perturbed entity are required")
        if not isinstance(self.fixed_entities, tuple):
            object.__setattr__(self, "fixed_entities", tuple(self.fixed_entities))
        bound = self.normalization_bound
        if bound is not None:
            values = np.asarray(bound, dtype=np.float64)
            if values.ndim > 1 or values.size == 0 or not np.isfinite(values).all() or np.any(values <= 0.0):
                raise ValueError("normalization bounds must be positive and finite")
            allowed_bound_sources = {UncertaintySource.MEASURED, UncertaintySource.MANUFACTURER, UncertaintySource.PROJECT_CONFIG}
            if self.normalization_bound_source not in allowed_bound_sources:
                raise ValueError("unsupported source cannot establish an FR5 physical normalization bound")
        if self.normalization_bound is None and self.normalization_status.startswith("AVAILABLE"):
            raise ValueError("available normalization requires an explicit bound")

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["semantic_class"] = self.semantic_class.value
        record["physical_source"] = self.physical_source.value
        record["normalization_bound_source"] = self.normalization_bound_source.value
        record["fixed_entities"] = list(self.fixed_entities)
        if isinstance(self.normalization_bound, tuple):
            record["normalization_bound"] = list(self.normalization_bound)
        return record


def validate_contract_record(record: Mapping[str, Any]) -> None:
    missing = _REQUIRED_FIELDS.difference(record)
    if missing:
        raise ValueError("missing_uncertainty_fields:" + ",".join(sorted(missing)))
    SemanticClass(str(record["semantic_class"]))
    UncertaintySource(str(record["physical_source"]))
    UncertaintySource(str(record["normalization_bound_source"]))
    for field in (
        "changes_joint_state",
        "changes_target_pose",
        "changes_surface_geometry",
        "changes_surface_normals",
        "changes_collision_world",
        "requires_reik",
        "requires_retiming",
        "requires_hardware_data",
    ):
        if not isinstance(record[field], bool):
            raise ValueError(f"{field}_must_be_boolean")
    bound = record["normalization_bound"]
    if bound is None and str(record["normalization_status"]).startswith("AVAILABLE"):
        raise ValueError("available_normalization_requires_a_bound")
    if bound is not None:
        values = np.asarray(bound, dtype=np.float64)
        if values.ndim > 1 or values.size == 0 or not np.isfinite(values).all() or np.any(values <= 0.0):
            raise ValueError("normalization_bound_must_be_positive_finite_or_null")
    if record["physical_source"] in {UncertaintySource.ASSUMED.value, UncertaintySource.LITERATURE.value}:
        if bound is not None or str(record["normalization_status"]).startswith("AVAILABLE"):
            raise ValueError("non_FR5_measurement_source_cannot_support_physical_normalization")
    if bound is not None and record["normalization_bound_source"] not in {
        UncertaintySource.MEASURED.value,
        UncertaintySource.MANUFACTURER.value,
        UncertaintySource.PROJECT_CONFIG.value,
    }:
        raise ValueError("unsupported_bound_source_cannot_support_physical_normalization")


def default_uncertainty_semantics() -> list[dict[str, Any]]:
    """Return the C3 ontology; unavailable physical sources remain explicit."""
    rows = [
        UncertaintySemantics(
            "joint_state_error",
            SemanticClass.JOINT_STATE_ERROR,
            UncertaintySource.ASSUMED,
            "robot joint state q",
            ("target pose specification", "workpiece surface geometry", "collision world"),
            "joint coordinates j1..j6, radians",
            "direct deterministic joint-state offset; no pose composition",
            "rad",
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            True,
            "C2 uniform joint-state offsets are deterministic software stresses, not a measured tracking distribution or hardware probability.",
            "AVAILABLE_DETERMINISTIC_STRESS_OPERATOR",
            None,
            UncertaintySource.UNKNOWN,
            "NOT_AVAILABLE_PHYSICAL_BOUNDS_MISSING",
        ),
        UncertaintySemantics(
            "target_pose_geometric_perturbation",
            SemanticClass.TARGET_POSE_GEOMETRIC_PERTURBATION,
            UncertaintySource.ASSUMED,
            "target TCP pose specification",
            ("robot joint state q", "workpiece collision geometry", "surface reconstruction"),
            "base_link for translation axes; target TCP for local rotation axes",
            "C2 axis operator convention; historical raw axis id and value are preserved",
            "m or rad, according to historical C2 axis",
            False,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            "Changes the geometric target used for comparison; does not represent TCP execution error, calibration error, or workpiece registration.",
            "AVAILABLE_DETERMINISTIC_STRESS_OPERATOR",
            None,
            UncertaintySource.UNKNOWN,
            "NOT_AVAILABLE_PHYSICAL_BOUNDS_MISSING",
        ),
        UncertaintySemantics(
            "tcp_calibration_error",
            SemanticClass.TCP_CALIBRATION_ERROR,
            UncertaintySource.UNKNOWN,
            "flange-to-TCP calibration transform",
            ("robot base", "workpiece registration", "nominal joint trajectory"),
            BASE_FRAME,
            "not implemented; no transform is inferred",
            "m and rad",
            False,
            True,
            False,
            False,
            False,
            True,
            False,
            True,
            "A calibrated TCP error distribution or bound was not found in the FR5 records searched for C3.",
            "NOT_AVAILABLE",
            None,
            UncertaintySource.UNKNOWN,
            "NOT_AVAILABLE_PHYSICAL_BOUNDS_MISSING",
        ),
        UncertaintySemantics(
            "base_workpiece_registration_error",
            SemanticClass.BASE_WORKPIECE_REGISTRATION_ERROR,
            UncertaintySource.UNKNOWN,
            "workpiece-attached geometry, surface normals, and process reference poses",
            ("robot base", "robot model", "nominal q trajectory", "nominal timestamps"),
            BASE_FRAME,
            COMPOSITION_CONVENTION,
            "translation m; axis-angle rotation rad",
            False,
            True,
            True,
            True,
            True,
            False,
            False,
            True,
            "Post-plan relative registration mismatch: the robot executes frozen q while all workpiece-attached quantities are left-composed by one base-frame SE(3) delta.",
            "IMPLEMENTED_ENGINEERING_PROPAGATION; PHYSICAL_BOUND_UNKNOWN",
            None,
            UncertaintySource.UNKNOWN,
            "NOT_AVAILABLE_PHYSICAL_BOUNDS_MISSING",
        ),
        UncertaintySemantics(
            "surface_reconstruction_error",
            SemanticClass.SURFACE_RECONSTRUCTION_ERROR,
            UncertaintySource.UNKNOWN,
            "reconstructed workpiece surface and surface normals",
            ("robot base", "nominal joint trajectory"),
            BASE_FRAME,
            "not implemented; available open-arch targets are not a calibrated reconstruction uncertainty model",
            "m and rad",
            False,
            True,
            True,
            True,
            True,
            False,
            False,
            True,
            "A surface reconstruction error model and physical bound are unavailable; the C3 wall boxes remain the existing project geometry approximation.",
            "NOT_AVAILABLE",
            None,
            UncertaintySource.UNKNOWN,
            "NOT_AVAILABLE_PHYSICAL_BOUNDS_MISSING",
        ),
        UncertaintySemantics(
            "execution_tracking_error",
            SemanticClass.EXECUTION_TRACKING_ERROR,
            UncertaintySource.UNKNOWN,
            "executed robot joint state and timestamps",
            ("target pose specification", "workpiece geometry"),
            "joint coordinates j1..j6 and controller time",
            "not implemented; no controller or hardware trace is available",
            "rad and s",
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            True,
            "No controller or hardware tracking trace was available for C3.",
            "NOT_AVAILABLE",
            None,
            UncertaintySource.UNKNOWN,
            "NOT_AVAILABLE_PHYSICAL_BOUNDS_MISSING",
        ),
    ]
    result = [row.to_record() for row in rows]
    for row in result:
        validate_contract_record(row)
    return result


def classify_legacy_c2_axis(axis_id: str) -> dict[str, str]:
    """Add semantics to a C2 axis without changing its stored identity or value."""
    if axis_id.startswith("tcp_tcp_translation:") or axis_id.startswith("tcp_tcp_rotation:"):
        return {
            "historical_axis_id": axis_id,
            "semantic_class": SemanticClass.TARGET_POSE_GEOMETRIC_PERTURBATION.value,
            "physical_source": UncertaintySource.ASSUMED.value,
            "physical_interpretation": "target pose comparison perturbation; not physical TCP execution or calibration error",
            "raw_historical_value_status": "PRESERVED_UNCHANGED",
        }
    if axis_id.startswith("joint:"):
        return {
            "historical_axis_id": axis_id,
            "semantic_class": SemanticClass.JOINT_STATE_ERROR.value,
            "physical_source": UncertaintySource.ASSUMED.value,
            "physical_interpretation": "deterministic joint-state offset stress; not a measured tracking distribution",
            "raw_historical_value_status": "PRESERVED_UNCHANGED",
        }
    return {
        "historical_axis_id": axis_id,
        "semantic_class": "UNCLASSIFIED",
        "physical_source": UncertaintySource.UNKNOWN.value,
        "physical_interpretation": "historical axis identity retained; no semantic mapping inferred",
        "raw_historical_value_status": "PRESERVED_UNCHANGED",
    }


def _finite_vector(values: Sequence[float], size: int, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (size,) or not np.isfinite(vector).all():
        raise ValueError(f"{name}_must_be_{size}_finite_values")
    return vector


def rotation_matrix_from_vector(rotation_vector_rad: Sequence[float]) -> np.ndarray:
    """Rodrigues Exp map for a base-frame axis-angle rotation vector."""
    vector = _finite_vector(rotation_vector_rad, 3, "rotation_vector_rad")
    theta = float(np.linalg.norm(vector))
    if theta <= 1e-15:
        return np.eye(3, dtype=np.float64)
    axis = vector / theta
    skew = np.asarray(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
        dtype=np.float64,
    )
    result = np.eye(3, dtype=np.float64) + math.sin(theta) * skew + (1.0 - math.cos(theta)) * (skew @ skew)
    if not np.isfinite(result).all() or not np.allclose(result.T @ result, np.eye(3), atol=1e-12) or not math.isclose(float(np.linalg.det(result)), 1.0, abs_tol=1e-12):
        raise ValueError("rotation_matrix_is_not_finite_SO3")
    return result


def registration_transform(
    translation_base_m: Sequence[float] = (0.0, 0.0, 0.0),
    rotation_vector_base_rad: Sequence[float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Build DeltaT_base with direct base-frame translation and Exp(phi) rotation."""
    translation = _finite_vector(translation_base_m, 3, "translation_base_m")
    rotation = rotation_matrix_from_vector(rotation_vector_base_rad)
    delta = np.eye(4, dtype=np.float64)
    delta[:3, :3] = rotation
    delta[:3, 3] = translation
    return delta


def validate_transform(transform: Sequence[Sequence[float]], *, atol: float = 1e-10) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("transform_must_be_finite_4x4")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=atol, rtol=0.0):
        raise ValueError("transform_last_row_invalid")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=atol, rtol=0.0) or not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=atol):
        raise ValueError("transform_rotation_not_SO3")
    return matrix


def inverse_transform(transform: Sequence[Sequence[float]]) -> np.ndarray:
    matrix = validate_transform(transform)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = matrix[:3, :3].T
    result[:3, 3] = -matrix[:3, :3].T @ matrix[:3, 3]
    return result


def quaternion_xyzw_to_matrix(quaternion_xyzw: Sequence[float]) -> np.ndarray:
    quaternion = _finite_vector(quaternion_xyzw, 4, "quaternion_xyzw")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError("quaternion_must_be_nonzero")
    x, y, z, w = quaternion / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quaternion_xyzw_from_matrix(matrix: Sequence[Sequence[float]]) -> np.ndarray:
    rotation = np.asarray(matrix, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("rotation_matrix_must_be_finite_3x3")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-10) or not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-10):
        raise ValueError("rotation_matrix_not_SO3")
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.asarray(
            [(rotation[2, 1] - rotation[1, 2]) / scale, (rotation[0, 2] - rotation[2, 0]) / scale,
             (rotation[1, 0] - rotation[0, 1]) / scale, 0.25 * scale],
            dtype=np.float64,
        )
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            quaternion = np.asarray([0.25 * scale, (rotation[0, 1] + rotation[1, 0]) / scale,
                                     (rotation[0, 2] + rotation[2, 0]) / scale, (rotation[2, 1] - rotation[1, 2]) / scale])
        elif index == 1:
            scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            quaternion = np.asarray([(rotation[0, 1] + rotation[1, 0]) / scale, 0.25 * scale,
                                     (rotation[1, 2] + rotation[2, 1]) / scale, (rotation[0, 2] - rotation[2, 0]) / scale])
        else:
            scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            quaternion = np.asarray([(rotation[0, 2] + rotation[2, 0]) / scale,
                                     (rotation[1, 2] + rotation[2, 1]) / scale, 0.25 * scale,
                                     (rotation[1, 0] - rotation[0, 1]) / scale])
    quaternion /= float(np.linalg.norm(quaternion))
    return quaternion


def transform_pose_array_xyzw(poses: Sequence[Sequence[float]], delta_transform: Sequence[Sequence[float]]) -> np.ndarray:
    values = np.asarray(poses, dtype=np.float64)
    delta = validate_transform(delta_transform)
    if values.ndim != 2 or values.shape[1] != 7 or not np.isfinite(values).all():
        raise ValueError("poses_must_be_finite_N_by_7_xyz_quaternion_xyzw")
    for pose in values:
        quaternion_xyzw_to_matrix(pose[3:7])
    if np.array_equal(delta, np.eye(4, dtype=np.float64)):
        return np.array(values, copy=True)
    output = np.array(values, copy=True)
    for index, pose in enumerate(values):
        output[index, :3] = delta[:3, :3] @ pose[:3] + delta[:3, 3]
        output[index, 3:7] = quaternion_xyzw_from_matrix(delta[:3, :3] @ quaternion_xyzw_to_matrix(pose[3:7]))
    if not np.isfinite(output).all() or not np.allclose(np.linalg.norm(output[:, 3:7], axis=1), 1.0, atol=1e-12):
        raise ValueError("transformed_pose_nonfinite_or_nonunit_quaternion")
    return output


def transform_surface_normals(normals: Sequence[Sequence[float]], delta_transform: Sequence[Sequence[float]]) -> np.ndarray:
    values = np.asarray(normals, dtype=np.float64)
    delta = validate_transform(delta_transform)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("surface_normals_must_be_finite_N_by_3")
    lengths = np.linalg.norm(values, axis=1)
    if np.any(lengths <= 1e-12):
        raise ValueError("surface_normal_must_be_nonzero")
    if np.array_equal(delta[:3, :3], np.eye(3, dtype=np.float64)):
        return np.array(values, copy=True)
    output = (delta[:3, :3] @ values.T).T
    if not np.isfinite(output).all() or not np.allclose(np.linalg.norm(output, axis=1), lengths, atol=1e-12):
        raise ValueError("transformed_surface_normal_invalid")
    return output


def apply_registration_to_trajectory(
    tcp_poses_xyz_quat_xyzw: Sequence[Sequence[float]],
    surface_normals_base: Sequence[Sequence[float]],
    joint_states: Sequence[Sequence[float]],
    timestamps_s: Sequence[float],
    delta_transform: Sequence[Sequence[float]],
) -> dict[str, np.ndarray]:
    """Transform workpiece-attached references; return frozen robot q/time copies."""
    q = np.asarray(joint_states, dtype=np.float64)
    times = np.asarray(timestamps_s, dtype=np.float64)
    if q.ndim != 2 or not np.isfinite(q).all():
        raise ValueError("joint_states_must_be_finite_2d")
    if times.ndim != 1 or not np.isfinite(times).all() or len(times) != len(q):
        raise ValueError("timestamps_must_be_finite_and_match_joint_states")
    poses = transform_pose_array_xyzw(tcp_poses_xyz_quat_xyzw, delta_transform)
    normals = transform_surface_normals(surface_normals_base, delta_transform)
    if len(poses) != len(q) or len(normals) != len(q):
        raise ValueError("registration_reference_row_count_mismatch")
    return {
        "tcp_poses_xyz_quat_xyzw": poses,
        "surface_normals_base": normals,
        "joint_states": np.array(q, copy=True),
        "timestamps_s": np.array(times, copy=True),
    }


def transform_collision_object_message(collision_object: Any, delta_transform: Sequence[Sequence[float]], *, base_frame: str = BASE_FRAME) -> Any:
    """Return a transformed copy of a MoveIt world CollisionObject message.

    Geometry dimensions/vertices stay in object-local coordinates. Every pose
    must be expressed in ``base_frame`` and is left-composed by the same delta.
    """
    delta = validate_transform(delta_transform)
    result = deepcopy(collision_object)
    frame = str(getattr(getattr(result, "header", None), "frame_id", ""))
    if frame != base_frame:
        raise ValueError(f"collision_object_frame_mismatch:{frame}:{base_frame}")
    transformed_pose_count = 0
    for field in ("primitive_poses", "mesh_poses", "plane_poses", "subframe_poses"):
        for pose in getattr(result, field, ()):
            point = np.asarray([pose.position.x, pose.position.y, pose.position.z], dtype=np.float64)
            quaternion = np.asarray([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w], dtype=np.float64)
            if not np.isfinite(point).all():
                raise ValueError("collision_object_pose_translation_nonfinite")
            transformed = transform_pose_array_xyzw(
                np.asarray([[*point.tolist(), *quaternion.tolist()]], dtype=np.float64), delta
            )[0]
            pose.position.x, pose.position.y, pose.position.z = map(float, transformed[:3])
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = map(float, transformed[3:7])
            transformed_pose_count += 1
    if transformed_pose_count == 0:
        raise ValueError("collision_object_has_no_transformable_geometry_pose")
    return result


def normalization_result(
    raw_margin: float | None,
    bound: float | None,
    bound_source: UncertaintySource | str,
) -> dict[str, Any]:
    source = UncertaintySource(bound_source)
    if raw_margin is None or not math.isfinite(float(raw_margin)):
        return {"normalized_margin": None, "status": "NOT_AVAILABLE_RAW_MARGIN_MISSING", "bound_source": source.value}
    if bound is None:
        return {"normalized_margin": None, "status": "NOT_AVAILABLE_PHYSICAL_BOUNDS_MISSING", "bound_source": source.value}
    value = float(bound)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("normalization_bound_must_be_positive_and_finite")
    if source not in {UncertaintySource.MEASURED, UncertaintySource.MANUFACTURER, UncertaintySource.PROJECT_CONFIG}:
        return {"normalized_margin": None, "status": "NOT_AVAILABLE_UNSUPPORTED_BOUND_SOURCE", "bound_source": source.value}
    return {"normalized_margin": float(raw_margin) / value, "status": "AVAILABLE_WITH_DECLARED_BOUND", "bound_source": source.value}


def six_dof_directional_smoke(translation_m: float = 0.001, rotation_rad: float = math.radians(0.1)) -> list[dict[str, Any]]:
    """Return 12 signed engineering-only directions, never physical bounds."""
    if not math.isfinite(translation_m) or translation_m <= 0.0 or not math.isfinite(rotation_rad) or rotation_rad <= 0.0:
        raise ValueError("smoke_amplitudes_must_be_positive_and_finite")
    rows: list[dict[str, Any]] = []
    for component, axis in enumerate("xyz"):
        for sign in (1, -1):
            translation = np.zeros(3, dtype=np.float64)
            translation[component] = sign * translation_m
            rows.append(
                {
                    "case_id": f"registration_translation_{axis}_{'positive' if sign > 0 else 'negative'}",
                    "family": "translation",
                    "axis": axis,
                    "sign": sign,
                    "translation_base_m": translation.tolist(),
                    "rotation_vector_base_rad": [0.0, 0.0, 0.0],
                    "amplitude": translation_m,
                    "units": "m",
                    "interpretation": "engineering propagation smoke only; not an FR5 physical uncertainty bound",
                }
            )
    for component, axis in enumerate("xyz"):
        for sign in (1, -1):
            rotation = np.zeros(3, dtype=np.float64)
            rotation[component] = sign * rotation_rad
            rows.append(
                {
                    "case_id": f"registration_rotation_{axis}_{'positive' if sign > 0 else 'negative'}",
                    "family": "rotation",
                    "axis": axis,
                    "sign": sign,
                    "translation_base_m": [0.0, 0.0, 0.0],
                    "rotation_vector_base_rad": rotation.tolist(),
                    "amplitude": rotation_rad,
                    "units": "rad",
                    "interpretation": "engineering propagation smoke only; not an FR5 physical uncertainty bound",
                }
            )
    return rows


def foreign_project_reference_count(texts: Iterable[str]) -> int:
    """Count forbidden cross-project labels without embedding them in source text."""
    tokens = ("ES" + "TUN", "i" + "ER8", "ier" + "8", "720" + "-MI")
    return sum(text.count(token) for text in texts for token in tokens)
