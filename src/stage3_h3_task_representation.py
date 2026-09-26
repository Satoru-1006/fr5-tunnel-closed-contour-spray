"""Stage 3 H3 geometric coverage-task and surface-target baseline.

H3 consumes only a verified Stage 3 H2 canonical mesh.  It creates geometric
task targets and coverage diagnostics; it deliberately has no IK, planning,
ROS, controller, ML, or robot-execution dependency.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from src.stage3_h2_geometry import (
    GeometryConfig,
    GeometryValidationError,
    Mesh,
    PointCloud,
    build_geometry_manifest,
    canonical_hash,
    canonical_json,
    canonicalize_geometry,
    load_geometry,
    sample_mesh_area_weighted,
    sha256_file,
)


SCHEMA_VERSION = "stage3-h3-task-representation-v1"
POSE_SCHEMA_VERSION = "stage3-h3-target-pose-v1"
FOOTPRINT_SCHEMA_VERSION = "stage3-h3-geometric-footprint-v1"
REPLAY_SCHEMA_VERSION = "stage3-h3-deterministic-replay-v1"
UNIT_TOLERANCE = 1.0e-9
RECONSTRUCTION_TOLERANCE = 1.0e-12
ROLL_DEGENERACY_EPSILON = 1.0e-12


class H3ValidationError(ValueError):
    """Fail-closed H3 input or semantic validation error."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class H3Config:
    """Explicit H3 parameters frozen by the H3 contracts."""

    nominal_standoff_m: float = 0.260
    cone_half_angle_deg: float = 15.0
    maximum_incidence_angle_deg: float = 90.0
    sample_count: int = 64
    random_seed: int = 17
    roll_reference_axes: tuple[tuple[float, float, float], ...] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    roll_degeneracy_epsilon: float = ROLL_DEGENERACY_EPSILON

    def __post_init__(self) -> None:
        if not math.isfinite(self.nominal_standoff_m) or self.nominal_standoff_m <= 0.0:
            raise H3ValidationError("invalid_standoff", "nominal_standoff_m must be finite and positive")
        if not math.isfinite(self.cone_half_angle_deg) or not 0.0 < self.cone_half_angle_deg < 90.0:
            raise H3ValidationError("invalid_footprint_parameter", "cone_half_angle_deg must be in (0, 90)")
        if not math.isfinite(self.maximum_incidence_angle_deg) or not 0.0 <= self.maximum_incidence_angle_deg <= 90.0:
            raise H3ValidationError("invalid_footprint_parameter", "maximum_incidence_angle_deg must be in [0, 90]")
        if self.sample_count <= 0 or self.random_seed < 0:
            raise H3ValidationError("invalid_sampling", "sample_count must be positive and random_seed nonnegative")
        if len(self.roll_reference_axes) == 0:
            raise H3ValidationError("invalid_roll_policy", "at least one roll reference axis is required")
        if self.roll_degeneracy_epsilon <= 0.0 or not math.isfinite(self.roll_degeneracy_epsilon):
            raise H3ValidationError("invalid_roll_policy", "roll_degeneracy_epsilon must be finite and positive")

    @property
    def effective_radius_m(self) -> float:
        return self.nominal_standoff_m * math.tan(math.radians(self.cone_half_angle_deg))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "nominal_standoff_m": self.nominal_standoff_m,
            "cone_half_angle_deg": self.cone_half_angle_deg,
            "maximum_incidence_angle_deg": self.maximum_incidence_angle_deg,
            "sample_count": self.sample_count,
            "random_seed": self.random_seed,
            "effective_radius_m": self.effective_radius_m,
            "roll_reference_axes": [list(axis) for axis in self.roll_reference_axes],
            "roll_degeneracy_epsilon": self.roll_degeneracy_epsilon,
            "stand_off_classification": "DECLARED_BASELINE_PARAMETER",
            "physical_deposition_model": "NOT_AVAILABLE",
        }


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _vector(value: Sequence[Any], *, field_name: str) -> tuple[float, float, float]:
    if len(value) != 3:
        raise H3ValidationError("invalid_vector", f"{field_name} must have exactly three values")
    result = tuple(float(component) for component in value)
    if not all(math.isfinite(component) for component in result):
        raise H3ValidationError("nonfinite_value", f"{field_name} contains NaN or Inf")
    return result


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(float(a[index]) * float(b[index]) for index in range(3))


def _cross(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return (
        float(a[1]) * float(b[2]) - float(a[2]) * float(b[1]),
        float(a[2]) * float(b[0]) - float(a[0]) * float(b[2]),
        float(a[0]) * float(b[1]) - float(a[1]) * float(b[0]),
    )


def _norm(value: Sequence[float]) -> float:
    return math.sqrt(_dot(value, value))


def _unit(value: Sequence[Any], *, field_name: str, tolerance: float = UNIT_TOLERANCE) -> tuple[float, float, float]:
    vector = _vector(value, field_name=field_name)
    length = _norm(vector)
    if not math.isfinite(length) or length <= tolerance:
        raise H3ValidationError("invalid_unit_vector", f"{field_name} has zero or nonfinite norm")
    result = tuple(component / length for component in vector)
    if not all(math.isfinite(component) for component in result):
        raise H3ValidationError("nonfinite_value", f"{field_name} normalizes to a nonfinite vector")
    return result


def _sub(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return tuple(float(a[index]) - float(b[index]) for index in range(3))  # type: ignore[return-value]


def _add(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return tuple(float(a[index]) + float(b[index]) for index in range(3))  # type: ignore[return-value]


def _scale(value: Sequence[float], scalar: float) -> tuple[float, float, float]:
    return tuple(float(component) * scalar for component in value)  # type: ignore[return-value]


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _canonical_quaternion_sign(quaternion: Sequence[float]) -> tuple[float, float, float, float]:
    values = tuple(float(value) for value in quaternion)
    if values[3] < -UNIT_TOLERANCE or (abs(values[3]) <= UNIT_TOLERANCE and next((value for value in values[:3] if abs(value) > UNIT_TOLERANCE), 0.0) < 0.0):
        values = tuple(-value for value in values)  # type: ignore[assignment]
    return values  # type: ignore[return-value]


def quaternion_from_basis(x_axis: Sequence[Any], y_axis: Sequence[Any], z_axis: Sequence[Any]) -> tuple[float, float, float, float]:
    """Return canonical xyzw quaternion for a right-handed local-axis basis."""

    x = _unit(x_axis, field_name="orientation_x_axis")
    y = _unit(y_axis, field_name="orientation_y_axis")
    z = _unit(z_axis, field_name="orientation_z_axis")
    if abs(_dot(x, y)) > UNIT_TOLERANCE or abs(_dot(x, z)) > UNIT_TOLERANCE or abs(_dot(y, z)) > UNIT_TOLERANCE:
        raise H3ValidationError("invalid_orientation_basis", "orientation basis is not orthogonal")
    if _dot(_cross(x, y), z) <= 0.0:
        raise H3ValidationError("invalid_orientation_basis", "orientation basis is not right-handed")
    matrix_trace = x[0] + y[1] + z[2]
    if matrix_trace > 0.0:
        scale = math.sqrt(matrix_trace + 1.0) * 2.0
        w = 0.25 * scale
        qx = (y[2] - z[1]) / scale
        qy = (z[0] - x[2]) / scale
        qz = (x[1] - y[0]) / scale
    elif x[0] > y[1] and x[0] > z[2]:
        scale = math.sqrt(1.0 + x[0] - y[1] - z[2]) * 2.0
        w = (y[2] - z[1]) / scale
        qx = 0.25 * scale
        qy = (y[0] + x[1]) / scale
        qz = (z[0] + x[2]) / scale
    elif y[1] > z[2]:
        scale = math.sqrt(1.0 + y[1] - x[0] - z[2]) * 2.0
        w = (z[0] - x[2]) / scale
        qx = (y[0] + x[1]) / scale
        qy = 0.25 * scale
        qz = (z[1] + y[2]) / scale
    else:
        scale = math.sqrt(1.0 + z[2] - x[0] - y[1]) * 2.0
        w = (x[1] - y[0]) / scale
        qx = (z[0] + x[2]) / scale
        qy = (z[1] + y[2]) / scale
        qz = 0.25 * scale
    quaternion = _canonical_quaternion_sign((qx, qy, qz, w))
    length = math.sqrt(sum(value * value for value in quaternion))
    return tuple(value / length for value in quaternion)  # type: ignore[return-value]


def quaternion_rotate_vector(quaternion: Sequence[Any], vector: Sequence[Any]) -> tuple[float, float, float]:
    """Rotate a vector by an xyzw quaternion for semantic pose assertions."""

    q = tuple(float(value) for value in quaternion)
    if len(q) != 4 or not all(math.isfinite(value) for value in q):
        raise H3ValidationError("invalid_quaternion", "quaternion must contain four finite values")
    length = math.sqrt(sum(value * value for value in q))
    if abs(length - 1.0) > UNIT_TOLERANCE:
        raise H3ValidationError("invalid_quaternion", "quaternion must be unit length")
    x, y, z, w = q
    v = _vector(vector, field_name="rotation_vector")
    qv = (x, y, z)
    t = _scale(_cross(qv, v), 2.0)
    return _add(v, _add(_scale(t, w), _cross(qv, t)))


def _select_roll_tangent(spray_direction: Sequence[Any], config: H3Config) -> tuple[tuple[float, float, float], dict[str, Any]]:
    z = _unit(spray_direction, field_name="spray_direction_unit")
    for index, reference_axis in enumerate(config.roll_reference_axes):
        reference = _unit(reference_axis, field_name=f"roll_reference_axis[{index}]")
        projection = _sub(reference, _scale(z, _dot(reference, z)))
        projection_norm = _norm(projection)
        if projection_norm > config.roll_degeneracy_epsilon:
            x = _unit(projection, field_name="roll_tangent")
            return x, {
                "method": "projected_global_reference_axis",
                "selected_reference_axis_index": index,
                "selected_reference_axis": list(reference),
                "reference_axes_in_priority_order": [list(axis) for axis in config.roll_reference_axes],
                "projection_norm": projection_norm,
                "degeneracy_epsilon": config.roll_degeneracy_epsilon,
                "fallback_used": index > 0,
                "sign_rule": "preserve_selected_reference_projection_sign",
                "filesystem_order_dependency": False,
            }
    raise H3ValidationError("roll_reference_degenerate", "all declared roll reference-axis projections are degenerate")


def construct_target_pose(surface_point_xyz_m: Sequence[Any], surface_normal_unit: Sequence[Any], *, coordinate_frame: str, config: H3Config) -> dict[str, Any]:
    """Construct a deterministic geometric target pose from point, normal, and stand-off."""

    if not coordinate_frame:
        raise H3ValidationError("missing_frame", "coordinate_frame is required for target pose")
    point = _vector(surface_point_xyz_m, field_name="surface_point_xyz_m")
    normal = _unit(surface_normal_unit, field_name="surface_normal_unit")
    spray_direction = _scale(normal, -1.0)
    tangent_x, roll_policy = _select_roll_tangent(spray_direction, config)
    tangent_y = _unit(_cross(spray_direction, tangent_x), field_name="orientation_y_axis")
    quaternion = quaternion_from_basis(tangent_x, tangent_y, spray_direction)
    tcp_position = _sub(point, _scale(spray_direction, config.nominal_standoff_m))
    reconstructed = _add(tcp_position, _scale(spray_direction, config.nominal_standoff_m))
    if _norm(_sub(reconstructed, point)) > RECONSTRUCTION_TOLERANCE:
        raise H3ValidationError("tcp_reconstruction_failed", "TCP stand-off reconstruction failed")
    if _norm(_sub(quaternion_rotate_vector(quaternion, (0.0, 0.0, 1.0)), spray_direction)) > 1.0e-9:
        raise H3ValidationError("tcp_orientation_failed", "TCP local +Z is not the spray direction")
    return {
        "surface_point_xyz_m": list(point),
        "surface_normal_unit": list(normal),
        "spray_direction_unit": list(spray_direction),
        "tcp_target_position_xyz_m": list(tcp_position),
        "tcp_target_orientation_xyzw": list(quaternion),
        "coordinate_frame": coordinate_frame,
        "orientation_convention": {
            "schema_version": POSE_SCHEMA_VERSION,
            "quaternion_order": "xyzw",
            "rotation_convention": "active right-handed rotation mapping TCP-local axes into coordinate_frame",
            "local_axes": {
                "+X": "deterministic projected roll tangent",
                "+Y": "cross(+Z,+X)",
                "+Z": "spray_direction_unit",
            },
        },
        "roll_policy": roll_policy,
        "stand_off_reconstruction_error_m": _norm(_sub(reconstructed, point)),
    }


def footprint_parameters(config: H3Config) -> dict[str, Any]:
    return {
        "schema_version": FOOTPRINT_SCHEMA_VERSION,
        "model_id": "geometric_cone_projection_v1",
        "model_classification": "GEOMETRIC_FOOTPRINT_ONLY",
        "physical_deposition_model": "NOT_AVAILABLE",
        "cone_half_angle_deg": config.cone_half_angle_deg,
        "effective_radius_m": config.effective_radius_m,
        "nominal_standoff_m": config.nominal_standoff_m,
        "nominal_standoff_classification": "DECLARED_BASELINE_PARAMETER",
        "clipping_rule": "finite tangent-plane disk; distance <= effective_radius_m",
        "incidence_angle_semantics": "acos(clamp(dot(surface_normal_unit, -spray_direction_unit), -1, 1))",
        "maximum_incidence_angle_deg": config.maximum_incidence_angle_deg,
        "calibration_status": "NOT_AVAILABLE",
    }


def _h2_sample_weights(samples: Sequence[Mapping[str, Any]], triangle_probability: Sequence[float]) -> list[float]:
    counts = Counter(int(sample["triangle_id"]) for sample in samples)
    weights: list[float] = []
    for sample in samples:
        triangle_id = int(sample["triangle_id"])
        probability = float(triangle_probability[triangle_id])
        count = counts[triangle_id]
        weights.append(probability / count if count else 0.0)
    total = sum(weights)
    if total <= 0.0 or not math.isfinite(total):
        raise H3ValidationError("invalid_sample_weights", "H2 sample weights are not positive and finite")
    return [weight / total for weight in weights]


def build_surface_task_samples(mesh: Mesh, h2_record: Mapping[str, Any], *, h2_manifest_hash: str, config: H3Config, geometry_id: str | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Convert H2 mesh samples into fully provenance-linked H3 task samples."""

    if not isinstance(mesh, Mesh):
        raise H3ValidationError("unsupported_geometry", "H3 surface tasks require an H2 canonical mesh")
    if h2_record.get("purpose") != "TEST_FIXTURE":
        raise H3ValidationError("unapproved_dataset", "H3 baseline only accepts explicitly labelled TEST_FIXTURE geometry")
    if h2_record.get("formal_stage3_dataset") is not False:
        raise H3ValidationError("dataset_authorization", "formal Stage 3 dataset creation is not authorized in H3")
    expected_geometry_id = str(geometry_id or h2_record.get("geometry_id") or "")
    if expected_geometry_id != h2_record.get("geometry_id"):
        raise H3ValidationError("geometry_identity_mismatch", "geometry_id does not match the H2 manifest record")
    geometry_hash = str(h2_record.get("geometry_hash") or "")
    if len(geometry_hash) != 64:
        raise H3ValidationError("missing_geometry_hash", "H2 geometry hash is required")
    seed = config.random_seed
    samples, sample_metadata = sample_mesh_area_weighted(mesh, config.sample_count, seed)
    weights = _h2_sample_weights(samples, sample_metadata["triangle_probability"])
    h3_config_hash = canonical_hash(config.to_dict())
    records: list[dict[str, Any]] = []
    for index, (sample, weight) in enumerate(zip(samples, weights)):
        point = [sample["x"], sample["y"], sample["z"]]
        normal = [sample["normal_x"], sample["normal_y"], sample["normal_z"]]
        pose = construct_target_pose(point, normal, coordinate_frame=str(h2_record["coordinate_frame"]["target_frame"]), config=config)
        record = {
            "schema_version": SCHEMA_VERSION,
            "geometry_id": expected_geometry_id,
            "task_sample_id": f"{expected_geometry_id}:sample_{index:06d}",
            "surface_point_xyz_m": point,
            "surface_normal_unit": pose["surface_normal_unit"],
            "source_triangle_id": int(sample["triangle_id"]),
            "source_barycentric_uvw": [sample["barycentric_u"], sample["barycentric_v"], sample["barycentric_w"]],
            "source_point_id": None,
            "coordinate_frame": pose["coordinate_frame"],
            "nominal_standoff_m": config.nominal_standoff_m,
            "spray_direction_unit": pose["spray_direction_unit"],
            "tcp_target_position_xyz_m": pose["tcp_target_position_xyz_m"],
            "tcp_target_orientation_xyzw": pose["tcp_target_orientation_xyzw"],
            "orientation_convention": pose["orientation_convention"],
            "roll_policy": pose["roll_policy"],
            "spray_state": "OFFLINE_GEOMETRIC_BASELINE",
            "target_semantics": "GEOMETRIC_TASK_TARGET",
            "coverage_footprint_parameters": footprint_parameters(config),
            "source_geometry_hash": geometry_hash,
            "h2_manifest_hash": h2_manifest_hash,
            "configuration_hash": h3_config_hash,
            "random_seed": seed,
            "surface_sample_weight": weight,
            "h2_sampling_provenance": {
                "triangle_id": int(sample["triangle_id"]),
                "barycentric_uvw": [sample["barycentric_u"], sample["barycentric_v"], sample["barycentric_w"]],
                "sampling_algorithm_version": sample["sampling_algorithm_version"],
                "h2_sample_count": sample["sample_count"],
                "h2_random_seed": sample["random_seed"],
                "normal_source": sample["normal_source"],
            },
        }
        validate_surface_task_record(record)
        records.append(record)
    return records, {
        "schema_version": "stage3-h3-sampling-metadata-v1",
        "geometry_id": expected_geometry_id,
        "geometry_hash": geometry_hash,
        "h2_manifest_hash": h2_manifest_hash,
        "sample_count": len(records),
        "random_seed": seed,
        "sample_weight_policy": "H2 triangle probability divided by observed deterministic sample count, normalized within geometry",
        "triangle_probability": sample_metadata["triangle_probability"],
        "degenerate_triangle_ids": sample_metadata["degenerate_triangle_ids"],
    }


def validate_surface_task_record(record: Mapping[str, Any]) -> None:
    required = {
        "geometry_id", "task_sample_id", "surface_point_xyz_m", "surface_normal_unit",
        "source_triangle_id", "source_barycentric_uvw", "coordinate_frame", "nominal_standoff_m",
        "spray_direction_unit", "tcp_target_position_xyz_m", "tcp_target_orientation_xyzw",
        "orientation_convention", "roll_policy", "spray_state", "coverage_footprint_parameters",
        "source_geometry_hash", "h2_manifest_hash", "configuration_hash", "random_seed",
        "surface_sample_weight",
    }
    missing = sorted(required - set(record))
    if missing:
        raise H3ValidationError("task_schema_invalid", f"missing task fields: {','.join(missing)}")
    point = _vector(record["surface_point_xyz_m"], field_name="surface_point_xyz_m")
    normal = _unit(record["surface_normal_unit"], field_name="surface_normal_unit")
    spray = _unit(record["spray_direction_unit"], field_name="spray_direction_unit")
    if _norm(_add(normal, spray)) > 1.0e-9:
        raise H3ValidationError("spray_direction_invalid", "spray_direction_unit must be -surface_normal_unit")
    if not isinstance(record["coordinate_frame"], str) or not record["coordinate_frame"]:
        raise H3ValidationError("missing_frame", "coordinate_frame is required")
    standoff = float(record["nominal_standoff_m"])
    if not math.isfinite(standoff) or standoff <= 0.0:
        raise H3ValidationError("invalid_standoff", "nominal_standoff_m must be finite and positive")
    tcp = _vector(record["tcp_target_position_xyz_m"], field_name="tcp_target_position_xyz_m")
    if _norm(_sub(_add(tcp, _scale(spray, standoff)), point)) > RECONSTRUCTION_TOLERANCE:
        raise H3ValidationError("tcp_reconstruction_failed", "TCP position does not reconstruct surface point")
    quaternion = record["tcp_target_orientation_xyzw"]
    if len(quaternion) != 4 or abs(math.sqrt(sum(float(value) ** 2 for value in quaternion)) - 1.0) > UNIT_TOLERANCE:
        raise H3ValidationError("invalid_quaternion", "TCP orientation must be a unit xyzw quaternion")
    if _norm(_sub(quaternion_rotate_vector(quaternion, (0.0, 0.0, 1.0)), spray)) > 1.0e-9:
        raise H3ValidationError("tcp_orientation_failed", "TCP local +Z does not equal spray direction")
    barycentric = record["source_barycentric_uvw"]
    if barycentric is None or len(barycentric) != 3 or abs(sum(float(value) for value in barycentric) - 1.0) > RECONSTRUCTION_TOLERANCE:
        raise H3ValidationError("sampling_provenance_invalid", "source barycentric coordinates are invalid")
    if not _finite(record["surface_sample_weight"]) or float(record["surface_sample_weight"]) < 0.0:
        raise H3ValidationError("invalid_sample_weights", "surface_sample_weight must be finite and nonnegative")
    for field in ("source_geometry_hash", "h2_manifest_hash", "configuration_hash"):
        value = str(record[field])
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value.lower()):
            raise H3ValidationError("invalid_hash", f"{field} must be a SHA-256 hexadecimal hash")


def geometry_topology_diagnostics(mesh: Mesh) -> dict[str, Any]:
    """Return explicit topology diagnostics without repairing the mesh."""

    edge_to_triangles: dict[tuple[int, int], list[int]] = defaultdict(list)
    for triangle_id, triangle in enumerate(mesh.triangles):
        a, b, c = triangle
        for edge in ((a, b), (b, c), (c, a)):
            edge_to_triangles[tuple(sorted(edge))].append(triangle_id)
    boundary_edges = sorted(edge for edge, triangles in edge_to_triangles.items() if len(triangles) == 1)
    nonmanifold_edges = sorted(edge for edge, triangles in edge_to_triangles.items() if len(triangles) > 2)
    adjacency: dict[int, set[int]] = defaultdict(set)
    for triangles in edge_to_triangles.values():
        for left in triangles:
            for right in triangles:
                if left != right:
                    adjacency[left].add(right)
    components: list[list[int]] = []
    unseen = set(range(len(mesh.triangles)))
    while unseen:
        start = min(unseen)
        queue = deque([start])
        unseen.remove(start)
        component: list[int] = []
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbour in sorted(adjacency[current]):
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    queue.append(neighbour)
        components.append(sorted(component))
    referenced = {index for triangle in mesh.triangles for index in triangle}
    isolated_vertices = sorted(set(range(len(mesh.vertices))) - referenced)
    invalid_normals = []
    if mesh.vertex_normals is None:
        invalid_normals = list(range(len(mesh.vertices)))
    else:
        invalid_normals = [index for index, normal in enumerate(mesh.vertex_normals) if not all(math.isfinite(float(value)) for value in normal) or abs(_norm(normal) - 1.0) > 1.0e-8]
    return {
        "schema_version": "stage3-h3-geometry-topology-diagnostics-v1",
        "triangle_count": len(mesh.triangles),
        "vertex_count": len(mesh.vertices),
        "degenerate_triangle_ids": list(mesh.degenerate_triangle_ids),
        "degenerate_triangle_policy": "REPORT_AND_EXCLUDE_FROM_H2_SAMPLING",
        "open_boundary_edge_count": len(boundary_edges),
        "open_boundary_edges": [list(edge) for edge in boundary_edges],
        "hole_policy": "REPORT_ONLY_NO_REPAIR",
        "nonmanifold_edge_count": len(nonmanifold_edges),
        "nonmanifold_edges": [list(edge) for edge in nonmanifold_edges],
        "disconnected_component_count": len(components),
        "disconnected_components": components,
        "isolated_vertex_ids": isolated_vertices,
        "isolated_region_policy": "REPORT_ONLY_NO_REPAIR",
        "invalid_normal_ids": invalid_normals,
        "invalid_normal_policy": "FAIL_CLOSED",
        "orientation_policy": "H2_CANONICAL_NORMAL_ORIENTATION_REQUIRED",
        "silent_repair_applied": False,
    }


def _incidence_angle_deg(surface_normal: Sequence[float], target_spray_direction: Sequence[float]) -> tuple[float, float]:
    cosine = _clamp(_dot(surface_normal, _scale(target_spray_direction, -1.0)), -1.0, 1.0)
    return math.degrees(math.acos(cosine)), max(0.0, cosine)


def _weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    total = sum(weights)
    return sum(value * weight for value, weight in zip(values, weights)) / total if total > 0.0 else 0.0


def evaluate_coverage(surface_samples: Sequence[Mapping[str, Any]], spray_targets: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Calculate explicit geometric coverage metrics and deterministic arrays."""

    if not surface_samples:
        raise H3ValidationError("empty_surface", "at least one surface evaluation sample is required")
    targets = list(spray_targets if spray_targets is not None else surface_samples)
    geometry_ids = {str(sample["geometry_id"]) for sample in surface_samples}
    if len(geometry_ids) != 1 or any(str(target["geometry_id"]) not in geometry_ids for target in targets):
        raise H3ValidationError("coverage_geometry_mismatch", "coverage evaluation must use one geometry_id")
    for sample in surface_samples:
        validate_surface_task_record(sample)
    for target in targets:
        validate_surface_task_record(target)
    weights = [float(sample["surface_sample_weight"]) for sample in surface_samples]
    weight_total = sum(weights)
    if weight_total <= 0.0 or not math.isfinite(weight_total):
        raise H3ValidationError("invalid_sample_weights", "coverage evaluation weights must be positive and finite")
    normalized_weights = [weight / weight_total for weight in weights]
    counts: list[int] = []
    effective_values: list[float] = []
    incidence_arrays: list[list[float]] = []
    for sample in surface_samples:
        sample_point = _vector(sample["surface_point_xyz_m"], field_name="surface_point_xyz_m")
        sample_normal = _unit(sample["surface_normal_unit"], field_name="surface_normal_unit")
        hits = 0
        effective_sum = 0.0
        incidence_values: list[float] = []
        for target in targets:
            target_point = _vector(target["surface_point_xyz_m"], field_name="target_surface_point_xyz_m")
            footprint = target["coverage_footprint_parameters"]
            radius = float(footprint["effective_radius_m"])
            distance = _norm(_sub(sample_point, target_point))
            angle_deg, cosine = _incidence_angle_deg(sample_normal, target["spray_direction_unit"])
            if distance <= radius + RECONSTRUCTION_TOLERANCE and angle_deg <= float(footprint["maximum_incidence_angle_deg"]) + 1.0e-12:
                hits += 1
                effective_sum += cosine
                incidence_values.append(angle_deg)
        counts.append(hits)
        effective_values.append(min(1.0, max(0.0, effective_sum)))
        incidence_arrays.append(incidence_values)
    covered_flags = [count > 0 for count in counts]
    coverage_ratio = sum(weight for weight, covered in zip(normalized_weights, covered_flags) if covered)
    uncovered_ratio = 1.0 - coverage_ratio
    overlap_redundancy = sum(weight * max(count - 1, 0) for weight, count in zip(normalized_weights, counts))
    effective_coverage = _weighted_mean(effective_values, normalized_weights)
    mean_count = _weighted_mean([float(count) for count in counts], normalized_weights)
    variance = _weighted_mean([(float(count) - mean_count) ** 2 for count in counts], normalized_weights)
    weighted_std = math.sqrt(max(0.0, variance))
    uniformity = 1.0 / (1.0 + weighted_std / mean_count) if mean_count > 0.0 else 0.0
    arrays = {
        "evaluation_sample_ids": [str(sample["task_sample_id"]) for sample in surface_samples],
        "coverage_counts": counts,
        "covered_flags": covered_flags,
        "normalized_weights": normalized_weights,
        "effective_coverage_values": effective_values,
        "incidence_angle_deg_by_sample": incidence_arrays,
    }
    return {
        "schema_version": "stage3-h3-coverage-metrics-v1",
        "geometry_id": next(iter(geometry_ids)),
        "surface_sample_count": len(surface_samples),
        "spray_target_count": len(targets),
        "metrics": {
            "surface_coverage_ratio": coverage_ratio,
            "uncovered_ratio": uncovered_ratio,
            "coverage_overlap_redundancy": overlap_redundancy,
            "effective_spray_coverage": effective_coverage,
            "coverage_uniformity": uniformity,
        },
        "metric_classification": {
            "surface_coverage_ratio": "CALCULATED",
            "uncovered_ratio": "CALCULATED",
            "coverage_overlap_redundancy": "CALCULATED",
            "effective_spray_coverage": "DIAGNOSTIC",
            "coverage_uniformity": "DIAGNOSTIC",
        },
        "thresholds": {
            "surface_coverage_ratio": "UNRESOLVED",
            "uncovered_ratio": "UNRESOLVED",
            "coverage_overlap_redundancy": "UNRESOLVED",
            "effective_spray_coverage": "UNRESOLVED",
            "coverage_uniformity": "UNRESOLVED",
        },
        "model": {
            "model_id": "geometric_cone_projection_v1",
            "classification": "GEOMETRIC_FOOTPRINT_ONLY",
            "physical_deposition_model": "NOT_AVAILABLE",
            "distance_policy": "Euclidean distance against evaluation surface sample",
            "incidence_policy": "explicit target footprint maximum incidence angle",
        },
        "arrays": arrays,
    }


def semantic_output_hash(task_records: Sequence[Mapping[str, Any]], coverage_metrics: Sequence[Mapping[str, Any]]) -> str:
    return canonical_hash({
        "schema_version": SCHEMA_VERSION,
        "task_records": list(task_records),
        "coverage_metrics": list(coverage_metrics),
    })


def h2_geometry_config_from_record(source_path: str, record: Mapping[str, Any]) -> GeometryConfig:
    """Build the exact H2 config used by the frozen synthetic H2 fixtures."""

    source_type = str(record["source_type"])
    orientation_method = str(record["normal_orientation_method"])
    if orientation_method == "winding_preserved":
        normal_policy: Mapping[str, Any] = {
            "method": "winding_preserved",
            "reference_convention": "triangle_winding_or_declared_surface_orientation",
        }
        preprocessing: Mapping[str, Any] = {"deduplicate_tolerance_m": None}
        sampling: Mapping[str, Any] = {
            "algorithm": "area_weighted_triangle",
            "seed": int(record.get("random_seed", 17)),
            "sample_count": 64,
        }
    elif orientation_method == "reference_direction":
        normal_policy = {"method": "reference_direction", "reference_direction": [0.0, 0.0, 1.0], "reference_frame": "base_link"}
        preprocessing = {"downsampling": {"enabled": False, "voxel_size_m": None}}
        sampling = {}
    else:
        raise H3ValidationError("h2_orientation_policy_unknown", f"unsupported frozen H2 orientation method: {orientation_method}")
    frames = record.get("coordinate_frame") or {}
    return GeometryConfig(
        geometry_id=str(record["geometry_id"]),
        dataset_id=str(record["dataset_id"]),
        source_path=source_path,
        source_type=source_type,
        source_unit=str(record["length_unit_original"]),
        source_frame=str(frames["source_frame"]),
        target_frame=str(frames["target_frame"]),
        transform_matrix=record["transform_matrix"],
        random_seed=int(record.get("random_seed", 0)),
        normal_policy=normal_policy,
        sampling_policy=sampling,
        preprocessing_config=preprocessing,
        purpose=str(record.get("purpose", "TEST_FIXTURE")),
    )


def load_and_verify_h2_mesh(repo_root: str, h2_record: Mapping[str, Any]) -> tuple[Mesh, dict[str, Any], dict[str, Any]]:
    """Load an H2 source and require its re-derived canonical hash to match H2."""

    source = str(h2_record["source_path"])
    source_path = source if source.startswith(("/", "\\")) or ":\\" in source else f"{repo_root.rstrip('/\\')}\\{source.replace('/', '\\')}"
    if not source_path:
        raise H3ValidationError("missing_geometry_source", "H2 source path is empty")
    if sha256_file(source_path) != h2_record.get("source_hash"):
        raise H3ValidationError("h2_source_hash_mismatch", f"H2 source hash mismatch for {h2_record.get('geometry_id')}")
    config = h2_geometry_config_from_record(source_path, h2_record)
    geometry = load_geometry(source_path)
    if not isinstance(geometry, Mesh):
        raise H3ValidationError("unsupported_geometry", "selected H2 H3 input is not a mesh")
    canonical_geometry, processing = canonicalize_geometry(config, geometry)
    if not isinstance(canonical_geometry, Mesh):
        raise H3ValidationError("unsupported_geometry", "selected H2 H3 input did not remain a mesh")
    derived_manifest = build_geometry_manifest(config, canonical_geometry, processing)
    if derived_manifest["geometry_hash"] != h2_record.get("geometry_hash"):
        raise H3ValidationError("h2_geometry_hash_mismatch", f"re-derived H2 geometry hash mismatch for {h2_record.get('geometry_id')}")
    return canonical_geometry, processing, derived_manifest


def canonical_jsonl_hash(records: Iterable[Mapping[str, Any]]) -> str:
    return canonical_hash([canonical_json(record) for record in records])


__all__ = [
    "H3Config",
    "H3ValidationError",
    "build_surface_task_samples",
    "canonical_jsonl_hash",
    "construct_target_pose",
    "evaluate_coverage",
    "footprint_parameters",
    "geometry_topology_diagnostics",
    "h2_geometry_config_from_record",
    "load_and_verify_h2_mesh",
    "quaternion_from_basis",
    "quaternion_rotate_vector",
    "semantic_output_hash",
    "validate_surface_task_record",
]
