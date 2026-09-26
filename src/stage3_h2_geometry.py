"""Stage 3 H2 canonical mesh and point-cloud representation baseline.

This module deliberately uses a small, dependency-free implementation for the
formal baseline.  Optional ROS/PCL/Open3D/trimesh capabilities are probed by
the H2 runner, but are not silently selected as a backend.  All public
operations require explicit units, frames, transforms, and the parameters of
any preprocessing step.
"""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
import math
import os
import platform
import random
import struct
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Iterator, Mapping, Sequence


SCHEMA_VERSION = "stage3-h2-canonical-geometry-v1"
PREPROCESSING_VERSION = "stage3-h2-stdlib-geometry-v1"
SAMPLING_ALGORITHM_VERSION = "area_weighted_triangle_sampling_python_mt19937_v1"
VOXEL_ALGORITHM_VERSION = "voxel_first_point_sorted_keys_v1"
NORMAL_ESTIMATION_VERSION = "brute_force_covariance_jacobi_v1"
HASH_SPEC_VERSION = "stage3-h2-canonical-hash-v1"
CANONICAL_UNIT = "m"
EPSILON_DEFAULT_M2 = 1.0e-15


class GeometryValidationError(ValueError):
    """A fail-closed geometry validation error with a machine-readable code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _float(value: Any, *, field_name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise GeometryValidationError("non_numeric", f"{field_name} is not numeric") from exc
    if not math.isfinite(result):
        raise GeometryValidationError("nonfinite_value", f"{field_name} contains NaN or infinity")
    return 0.0 if result == 0.0 else result


def _canonicalize(value: Any) -> Any:
    """Normalize data before deterministic JSON hashing.

    Numeric values remain JSON numbers.  Python's JSON encoder uses the
    shortest round-trip float representation; the hash specification records
    this explicitly and rejects non-finite values before serialization.
    """
    if isinstance(value, Mapping):
        return {str(key): _canonicalize(value[key]) for key in sorted(value, key=lambda item: str(item))}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise GeometryValidationError("nonfinite_value", "non-finite value cannot be hashed")
        return 0.0 if value == 0.0 else value
    return str(value)


def canonical_json(value: Any) -> str:
    return json.dumps(
        _canonicalize(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_hash(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def identity_matrix() -> list[list[float]]:
    return [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


def _validate_matrix(matrix: Sequence[Sequence[Any]]) -> list[list[float]]:
    if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
        raise GeometryValidationError("invalid_transform", "transform_matrix must be 4x4")
    result = [[_float(value, field_name="transform_matrix") for value in row] for row in matrix]
    if any(abs(result[3][idx] - expected) > 1.0e-12 for idx, expected in enumerate((0.0, 0.0, 0.0, 1.0))):
        raise GeometryValidationError("invalid_transform", "transform_matrix must be affine")
    return result


def _apply_transform(point: Sequence[float], matrix: Sequence[Sequence[float]]) -> tuple[float, float, float]:
    x, y, z = point
    values = (
        matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z + matrix[0][3],
        matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z + matrix[1][3],
        matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z + matrix[2][3],
    )
    return tuple(_float(value, field_name="transformed_coordinate") for value in values)


def _transform_normal(normal: Sequence[float], matrix: Sequence[Sequence[float]]) -> tuple[float, float, float]:
    # The baseline accepts rigid transforms.  Refuse scale/shear rather than
    # silently applying an incorrect inverse-transpose normal transform.
    linear = [[matrix[row][col] for col in range(3)] for row in range(3)]
    gram = [[sum(linear[k][row] * linear[k][col] for k in range(3)) for col in range(3)] for row in range(3)]
    if any(abs(gram[row][col] - (1.0 if row == col else 0.0)) > 1.0e-9 for row in range(3) for col in range(3)):
        raise GeometryValidationError("unsupported_transform", "normal transform requires a rigid 4x4 matrix")
    values = tuple(sum(linear[row][col] * normal[col] for col in range(3)) for row in range(3))
    return _unit(values, field_name="transformed_normal")


def _unit(vector: Sequence[float], *, field_name: str = "normal") -> tuple[float, float, float]:
    values = tuple(_float(item, field_name=field_name) for item in vector)
    length = math.sqrt(sum(item * item for item in values))
    if length <= 1.0e-15:
        raise GeometryValidationError("zero_normal", f"{field_name} has zero norm")
    return tuple(item / length for item in values)


def _sub(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return tuple(a[index] - b[index] for index in range(3))  # type: ignore[return-value]


def _add(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return tuple(a[index] + b[index] for index in range(3))  # type: ignore[return-value]


def _scale(a: Sequence[float], scalar: float) -> tuple[float, float, float]:
    return tuple(value * scalar for value in a)  # type: ignore[return-value]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(a[index] * b[index] for index in range(3))


def _cross(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _norm(a: Sequence[float]) -> float:
    return math.sqrt(_dot(a, a))


def triangle_area(v0: Sequence[float], v1: Sequence[float], v2: Sequence[float]) -> float:
    return 0.5 * _norm(_cross(_sub(v1, v0), _sub(v2, v0)))


def _bounds(points: Sequence[Sequence[float]]) -> tuple[list[float], list[float]]:
    if not points:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    return (
        [min(point[index] for point in points) for index in range(3)],
        [max(point[index] for point in points) for index in range(3)],
    )


def _normal_angle_deg(a: Sequence[float], b: Sequence[float]) -> float:
    cosine = max(-1.0, min(1.0, _dot(_unit(a), _unit(b))))
    return math.degrees(math.acos(cosine))


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


@dataclass(frozen=True)
class GeometryConfig:
    geometry_id: str
    dataset_id: str
    source_path: str | Path
    source_type: str
    source_unit: str | None
    source_frame: str | None
    target_frame: str | None
    transform_matrix: Sequence[Sequence[float]] | None = None
    random_seed: int = 0
    normal_policy: Mapping[str, Any] = field(default_factory=dict)
    sampling_policy: Mapping[str, Any] = field(default_factory=dict)
    preprocessing_config: Mapping[str, Any] = field(default_factory=dict)
    parent_geometry_id: str | None = None
    purpose: str = "TEST_FIXTURE"


@dataclass
class Mesh:
    vertices: list[tuple[float, float, float]]
    triangles: list[tuple[int, int, int]]
    vertex_normals: list[tuple[float, float, float]] | None = None
    face_normals: list[tuple[float, float, float]] | None = None
    source_vertex_count: int = 0
    source_triangle_count: int = 0
    source_bounds_min: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    source_bounds_max: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    degenerate_triangle_ids: list[int] = field(default_factory=list)
    duplicate_vertex_provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def triangle_areas(self) -> list[float]:
        return [triangle_area(self.vertices[a], self.vertices[b], self.vertices[c]) for a, b, c in self.triangles]

    @property
    def bounds(self) -> tuple[list[float], list[float]]:
        return _bounds(self.vertices)

    @property
    def centroid(self) -> list[float]:
        if not self.vertices:
            return [0.0, 0.0, 0.0]
        return [sum(point[index] for point in self.vertices) / len(self.vertices) for index in range(3)]


@dataclass
class PointCloud:
    points: list[tuple[float, float, float]]
    colors: list[tuple[int, int, int]] | None = None
    normals: list[tuple[float, float, float]] | None = None
    source_point_count: int = 0
    source_bounds_min: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    source_bounds_max: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    @property
    def bounds(self) -> tuple[list[float], list[float]]:
        return _bounds(self.points)

    @property
    def centroid(self) -> list[float]:
        if not self.points:
            return [0.0, 0.0, 0.0]
        return [sum(point[index] for point in self.points) / len(self.points) for index in range(3)]


def _parse_float_tokens(tokens: Sequence[str], field_name: str) -> tuple[float, ...]:
    return tuple(_float(token, field_name=field_name) for token in tokens)


def _parse_obj(path: Path) -> Mesh:
    vertices: list[tuple[float, float, float]] = []
    normals: list[tuple[float, float, float]] = []
    triangles: list[tuple[int, int, int]] = []
    face_normal_indices: list[tuple[int | None, int | None, int | None]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        tokens = line.split()
        if tokens[0] == "v":
            if len(tokens) < 4:
                raise GeometryValidationError("invalid_mesh", f"OBJ vertex at line {line_number} has fewer than 3 values")
            vertices.append(tuple(_parse_float_tokens(tokens[1:4], "OBJ vertex")))  # type: ignore[arg-type]
        elif tokens[0] == "vn":
            if len(tokens) < 4:
                raise GeometryValidationError("invalid_mesh", f"OBJ normal at line {line_number} has fewer than 3 values")
            normals.append(tuple(_parse_float_tokens(tokens[1:4], "OBJ normal")))  # type: ignore[arg-type]
        elif tokens[0] == "f":
            if len(tokens) != 4:
                raise GeometryValidationError("unsupported_mesh", "OBJ baseline requires triangular faces")
            parsed: list[tuple[int, int | None]] = []
            for token in tokens[1:]:
                parts = token.split("/")
                index = int(parts[0])
                vertex_index = index - 1 if index > 0 else len(vertices) + index
                if vertex_index < 0 or vertex_index >= len(vertices):
                    raise GeometryValidationError("invalid_mesh", f"OBJ face at line {line_number} references invalid vertex")
                normal_index = None
                if len(parts) == 3 and parts[2]:
                    normal_index = int(parts[2]) - 1
                    if normal_index < 0 or normal_index >= len(normals):
                        raise GeometryValidationError("invalid_mesh", f"OBJ face at line {line_number} references invalid normal")
                parsed.append((vertex_index, normal_index))
            triangles.append(tuple(item[0] for item in parsed))  # type: ignore[arg-type]
            face_normal_indices.append(tuple(item[1] for item in parsed))  # type: ignore[arg-type]
    vertex_normals = None
    if normals and all(index is not None for face in face_normal_indices for index in face):
        sums = [[0.0, 0.0, 0.0] for _ in vertices]
        counts = [0] * len(vertices)
        for face, indices in zip(triangles, face_normal_indices):
            for vertex_index, normal_index in zip(face, indices):
                assert normal_index is not None
                sums[vertex_index] = list(_add(sums[vertex_index], normals[normal_index]))
                counts[vertex_index] += 1
        vertex_normals = [_unit(sums[index], field_name="OBJ vertex normal") if counts[index] else (0.0, 0.0, 1.0) for index in range(len(vertices))]
    bounds_min, bounds_max = _bounds(vertices)
    return Mesh(vertices, triangles, vertex_normals=vertex_normals, source_vertex_count=len(vertices), source_triangle_count=len(triangles), source_bounds_min=bounds_min, source_bounds_max=bounds_max)


def _parse_ascii_stl(text: str) -> Mesh:
    vertices: list[tuple[float, float, float]] = []
    triangles: list[tuple[int, int, int]] = []
    for line in text.splitlines():
        tokens = line.strip().split()
        if len(tokens) == 4 and tokens[0].lower() == "vertex":
            vertices.append(tuple(_parse_float_tokens(tokens[1:4], "STL vertex")))  # type: ignore[arg-type]
    if len(vertices) % 3:
        raise GeometryValidationError("invalid_mesh", "ASCII STL vertex count is not divisible by 3")
    # STL has no stable shared-vertex identity in the format.  Retain its
    # declared order; no implicit deduplication is permitted.
    for index in range(0, len(vertices), 3):
        triangles.append((index, index + 1, index + 2))
    bounds_min, bounds_max = _bounds(vertices)
    return Mesh(vertices, triangles, source_vertex_count=len(vertices), source_triangle_count=len(triangles), source_bounds_min=bounds_min, source_bounds_max=bounds_max)


def _parse_binary_stl(data: bytes) -> Mesh:
    if len(data) < 84:
        raise GeometryValidationError("invalid_mesh", "binary STL is shorter than its header")
    count = struct.unpack_from("<I", data, 80)[0]
    expected = 84 + 50 * count
    if expected > len(data):
        raise GeometryValidationError("invalid_mesh", "binary STL triangle count exceeds file length")
    vertices: list[tuple[float, float, float]] = []
    triangles: list[tuple[int, int, int]] = []
    offset = 84
    for triangle_id in range(count):
        _normal = struct.unpack_from("<3f", data, offset)
        offset += 12
        base = len(vertices)
        for _ in range(3):
            vertex = struct.unpack_from("<3f", data, offset)
            vertices.append(tuple(_float(value, field_name="STL vertex") for value in vertex))
            offset += 12
        triangles.append((base, base + 1, base + 2))
        offset += 2
    bounds_min, bounds_max = _bounds(vertices)
    return Mesh(vertices, triangles, source_vertex_count=len(vertices), source_triangle_count=len(triangles), source_bounds_min=bounds_min, source_bounds_max=bounds_max)


def _parse_stl(path: Path) -> Mesh:
    data = path.read_bytes()
    if len(data) >= 84:
        declared = struct.unpack_from("<I", data, 80)[0]
        if 84 + 50 * declared == len(data):
            return _parse_binary_stl(data)
    return _parse_ascii_stl(data.decode("utf-8"))


def _parse_ply(path: Path) -> Mesh | PointCloud:
    data = path.read_bytes()
    marker = data.find(b"end_header\n")
    newline_size = 1
    if marker < 0:
        marker = data.find(b"end_header\r\n")
        newline_size = 2
    if marker < 0:
        raise GeometryValidationError("invalid_ply", "PLY end_header is missing")
    header = data[: marker + len(b"end_header")].decode("ascii", errors="strict").splitlines()
    if not header or header[0].strip() != "ply":
        raise GeometryValidationError("invalid_ply", "PLY magic is missing")
    format_line = next((line for line in header if line.startswith("format ")), "")
    if not format_line.startswith("format ascii "):
        raise GeometryValidationError("unsupported_format", "H2 baseline accepts ASCII PLY only")
    vertex_count = 0
    properties: list[str] = []
    face_count = 0
    current_element = None
    for line in header[1:]:
        tokens = line.split()
        if tokens[:1] == ["element"]:
            current_element = tokens[1]
            if current_element == "vertex":
                vertex_count = int(tokens[2])
            elif current_element == "face":
                face_count = int(tokens[2])
        elif tokens[:1] == ["property"] and current_element == "vertex":
            if len(tokens) == 3:
                properties.append(tokens[2])
            elif len(tokens) >= 5 and tokens[1] != "list":
                properties.append(tokens[-1])
    body = data[marker + len(b"end_header") + newline_size :].decode("utf-8").splitlines()
    if len(body) < vertex_count + face_count:
        raise GeometryValidationError("invalid_ply", "PLY body has fewer rows than declared")
    try:
        x_index, y_index, z_index = (properties.index(name) for name in ("x", "y", "z"))
    except ValueError as exc:
        raise GeometryValidationError("invalid_point_cloud", "PLY must declare x, y, and z vertex properties") from exc
    normal_indices = tuple(properties.index(name) if name in properties else None for name in ("nx", "ny", "nz"))
    color_indices = tuple(properties.index(name) if name in properties else None for name in ("red", "green", "blue"))
    vertices: list[tuple[float, float, float]] = []
    normals: list[tuple[float, float, float]] | None = [] if all(index is not None for index in normal_indices) else None
    colors: list[tuple[int, int, int]] | None = [] if all(index is not None for index in color_indices) else None
    for row in body[:vertex_count]:
        tokens = row.split()
        if len(tokens) < len(properties):
            raise GeometryValidationError("invalid_ply", "PLY vertex row has fewer values than properties")
        vertices.append(tuple(_float(tokens[index], field_name="PLY vertex") for index in (x_index, y_index, z_index)))
        if normals is not None:
            normals.append(_unit(tuple(_float(tokens[index], field_name="PLY normal") for index in normal_indices if index is not None), field_name="PLY normal"))
        if colors is not None:
            colors.append(tuple(int(float(tokens[index])) for index in color_indices if index is not None))  # type: ignore[arg-type]
    if face_count:
        triangles: list[tuple[int, int, int]] = []
        for row in body[vertex_count : vertex_count + face_count]:
            tokens = row.split()
            count = int(tokens[0])
            indices = [int(value) for value in tokens[1 : count + 1]]
            if count != 3:
                raise GeometryValidationError("unsupported_mesh", "H2 baseline requires triangular PLY faces")
            triangles.append(tuple(indices))  # type: ignore[arg-type]
        bounds_min, bounds_max = _bounds(vertices)
        return Mesh(vertices, triangles, vertex_normals=normals, source_vertex_count=len(vertices), source_triangle_count=len(triangles), source_bounds_min=bounds_min, source_bounds_max=bounds_max)
    bounds_min, bounds_max = _bounds(vertices)
    return PointCloud(vertices, colors=colors, normals=normals, source_point_count=len(vertices), source_bounds_min=bounds_min, source_bounds_max=bounds_max)


def _parse_xyz(path: Path) -> PointCloud:
    points: list[tuple[float, float, float]] = []
    colors: list[tuple[int, int, int]] | None = None
    normals: list[tuple[float, float, float]] | None = None
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        tokens = line.replace(",", " ").split()
        if len(tokens) < 3:
            raise GeometryValidationError("invalid_point_cloud", f"XYZ row {line_number} has fewer than 3 values")
        points.append(tuple(_parse_float_tokens(tokens[:3], "XYZ point")))  # type: ignore[arg-type]
        if len(tokens) >= 6:
            if colors is None:
                colors = []
            colors.append(tuple(int(float(value)) for value in tokens[3:6]))  # type: ignore[arg-type]
        elif colors is not None:
            raise GeometryValidationError("invalid_point_cloud", "XYZ color columns must be present for every row")
        if len(tokens) >= 9:
            if normals is None:
                normals = []
            normals.append(_unit(_parse_float_tokens(tokens[6:9], "XYZ normal"), field_name="XYZ normal"))
        elif normals is not None:
            raise GeometryValidationError("invalid_point_cloud", "XYZ normal columns must be present for every row")
    bounds_min, bounds_max = _bounds(points)
    return PointCloud(points, colors=colors, normals=normals, source_point_count=len(points), source_bounds_min=bounds_min, source_bounds_max=bounds_max)


def load_geometry(path: str | Path) -> Mesh | PointCloud:
    source = Path(path)
    if not source.is_file():
        raise GeometryValidationError("missing_geometry", f"geometry source does not exist: {source}")
    suffix = source.suffix.lower()
    if suffix == ".obj":
        return _parse_obj(source)
    if suffix == ".stl":
        return _parse_stl(source)
    if suffix == ".ply":
        return _parse_ply(source)
    if suffix in {".xyz", ".pts"}:
        return _parse_xyz(source)
    raise GeometryValidationError("unsupported_format", f"H2 selected baseline has no loader for {suffix}")


def _validate_explicit_config(config: GeometryConfig) -> list[list[float]]:
    if not config.source_unit:
        raise GeometryValidationError("missing_unit", "source unit is mandatory and cannot be inferred")
    if config.source_unit not in {"m", "cm", "mm", "um", "in", "ft"}:
        raise GeometryValidationError("unsupported_unit", f"unsupported explicit source unit: {config.source_unit}")
    if not config.source_frame:
        raise GeometryValidationError("missing_frame", "source_frame is mandatory")
    if not config.target_frame:
        raise GeometryValidationError("missing_frame", "target_frame is mandatory")
    if config.transform_matrix is None:
        raise GeometryValidationError("missing_transform", "transform_matrix must be explicit, including identity")
    return _validate_matrix(config.transform_matrix)


def _unit_scale(unit: str) -> float:
    return {"m": 1.0, "cm": 1.0e-2, "mm": 1.0e-3, "um": 1.0e-6, "in": 0.0254, "ft": 0.3048}[unit]


def _transform_points(points: Sequence[Sequence[float]], config: GeometryConfig, matrix: Sequence[Sequence[float]]) -> list[tuple[float, float, float]]:
    scale = _unit_scale(config.source_unit or "")
    return [_apply_transform(_scale(point, scale), matrix) for point in points]


def _transform_normals(normals: Sequence[Sequence[float]] | None, matrix: Sequence[Sequence[float]]) -> list[tuple[float, float, float]] | None:
    return None if normals is None else [_transform_normal(normal, matrix) for normal in normals]


def _derive_face_normals(mesh: Mesh, epsilon_m2: float) -> list[tuple[float, float, float]]:
    normals: list[tuple[float, float, float]] = []
    for triangle_id, (a, b, c) in enumerate(mesh.triangles):
        normal = _cross(_sub(mesh.vertices[b], mesh.vertices[a]), _sub(mesh.vertices[c], mesh.vertices[a]))
        if triangle_area(mesh.vertices[a], mesh.vertices[b], mesh.vertices[c]) <= epsilon_m2:
            normals.append((0.0, 0.0, 0.0))
        else:
            normals.append(_unit(normal, field_name=f"face_normal[{triangle_id}]"))
    return normals


def _derive_vertex_normals(mesh: Mesh, face_normals: Sequence[Sequence[float]], epsilon_m2: float) -> list[tuple[float, float, float]]:
    sums = [[0.0, 0.0, 0.0] for _ in mesh.vertices]
    for triangle_id, (a, b, c) in enumerate(mesh.triangles):
        area = triangle_area(mesh.vertices[a], mesh.vertices[b], mesh.vertices[c])
        if area <= epsilon_m2:
            continue
        weighted = _scale(face_normals[triangle_id], area)
        for index in (a, b, c):
            sums[index] = list(_add(sums[index], weighted))
    return [_unit(value, field_name=f"vertex_normal[{index}]") if _norm(value) > 1.0e-15 else (0.0, 0.0, 1.0) for index, value in enumerate(sums)]


def _orient_normals(normals: Sequence[Sequence[float]], policy: Mapping[str, Any], points: Sequence[Sequence[float]]) -> list[tuple[float, float, float]]:
    method = policy.get("method")
    if not method:
        raise GeometryValidationError("missing_normal_orientation", "normal orientation method is mandatory")
    oriented: list[tuple[float, float, float]] = []
    reference = policy.get("reference_direction")
    viewpoint = policy.get("viewpoint")
    origin = policy.get("origin", [0.0, 0.0, 0.0])
    if method == "reference_direction":
        if reference is None:
            raise GeometryValidationError("missing_normal_orientation", "reference_direction is required")
        ref = _unit(reference, field_name="reference_direction")
    elif method == "declared_viewpoint":
        if viewpoint is None:
            raise GeometryValidationError("missing_normal_orientation", "viewpoint is required")
        ref = None
    elif method in {"winding_preserved", "outward_from_origin"}:
        ref = None
    else:
        raise GeometryValidationError("unsupported_normal_orientation", f"unsupported normal orientation method: {method}")
    for index, normal in enumerate(normals):
        current = _unit(normal, field_name=f"normal[{index}]")
        if method == "reference_direction":
            if _dot(current, ref) < 0.0:  # type: ignore[arg-type]
                current = _scale(current, -1.0)
        elif method == "declared_viewpoint":
            direction = _sub(viewpoint, points[index])  # type: ignore[arg-type]
            if _dot(current, direction) < 0.0:
                current = _scale(current, -1.0)
        elif method == "outward_from_origin":
            direction = _sub(points[index], origin)
            if _dot(current, direction) < 0.0:
                current = _scale(current, -1.0)
        oriented.append(current)
    return oriented


def _deduplicate_mesh(mesh: Mesh, tolerance: float | None) -> Mesh:
    if tolerance is None:
        mesh.duplicate_vertex_provenance = {"applied": False, "tolerance_m": None, "before_count": len(mesh.vertices), "after_count": len(mesh.vertices), "mapping": list(range(len(mesh.vertices)))}
        return mesh
    tolerance = _float(tolerance, field_name="deduplicate_tolerance_m")
    if tolerance <= 0.0:
        raise GeometryValidationError("invalid_preprocessing", "deduplicate tolerance must be positive")
    cells: dict[tuple[int, int, int], list[int]] = {}
    mapping: list[int] = []
    unique: list[tuple[float, float, float]] = []
    for vertex in mesh.vertices:
        cell = tuple(math.floor(value / tolerance) for value in vertex)
        selected = None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for candidate in cells.get((cell[0] + dx, cell[1] + dy, cell[2] + dz), []):
                        if _norm(_sub(vertex, unique[candidate])) <= tolerance:
                            selected = candidate
                            break
                    if selected is not None:
                        break
                if selected is not None:
                    break
            if selected is not None:
                break
        if selected is None:
            selected = len(unique)
            unique.append(vertex)
            cells.setdefault(cell, []).append(selected)
        mapping.append(selected)
    triangles = [tuple(mapping[index] for index in triangle) for triangle in mesh.triangles]
    mesh.vertices = unique
    mesh.triangles = triangles
    mesh.vertex_normals = None
    mesh.duplicate_vertex_provenance = {"applied": True, "tolerance_m": tolerance, "before_count": len(mapping), "after_count": len(unique), "mapping": mapping}
    return mesh


def canonicalize_geometry(config: GeometryConfig, geometry: Mesh | PointCloud, *, degenerate_epsilon_m2: float = EPSILON_DEFAULT_M2) -> tuple[Mesh | PointCloud, dict[str, Any]]:
    matrix = _validate_explicit_config(config)
    if isinstance(geometry, Mesh):
        geometry.vertices = _transform_points(geometry.vertices, config, matrix)
        geometry.vertex_normals = _transform_normals(geometry.vertex_normals, matrix)
        geometry = _deduplicate_mesh(geometry, config.preprocessing_config.get("deduplicate_tolerance_m"))
        geometry.source_bounds_min, geometry.source_bounds_max = _bounds(geometry.vertices)
        geometry.degenerate_triangle_ids = [index for index, area in enumerate(geometry.triangle_areas) if area <= degenerate_epsilon_m2]
        face_normals = _derive_face_normals(geometry, degenerate_epsilon_m2)
        if geometry.vertex_normals is not None:
            vertex_normals = [_unit(normal, field_name="imported_vertex_normal") for normal in geometry.vertex_normals]
            normal_source = "file"
        else:
            vertex_normals = _derive_vertex_normals(geometry, face_normals, degenerate_epsilon_m2)
            normal_source = "vertex_derived"
        vertex_normals = _orient_normals(vertex_normals, config.normal_policy, geometry.vertices)
        face_normals = [_orient_normals([normal if _norm(normal) > 0 else (0.0, 0.0, 1.0)], config.normal_policy, [geometry.vertices[geometry.triangles[index][0]]])[0] if _norm(normal) > 0 else (0.0, 0.0, 1.0) for index, normal in enumerate(face_normals)]
        geometry.face_normals = face_normals
        geometry.vertex_normals = vertex_normals
        return geometry, {
            "normal_source": normal_source,
            "normal_orientation_method": config.normal_policy.get("method"),
            "degenerate_triangle_epsilon_m2": degenerate_epsilon_m2,
            "degenerate_triangle_count": len(geometry.degenerate_triangle_ids),
            "degenerate_triangle_ids": geometry.degenerate_triangle_ids,
            "duplicate_vertex_provenance": geometry.duplicate_vertex_provenance,
            "mesh_repair": {"applied": False, "policy": "NOT_APPLIED_TO_FORMAL_BASELINE"},
        }
    geometry.points = _transform_points(geometry.points, config, matrix)
    geometry.normals = _transform_normals(geometry.normals, matrix)
    geometry.source_bounds_min, geometry.source_bounds_max = _bounds(geometry.points)
    if geometry.normals is not None:
        geometry.normals = _orient_normals(geometry.normals, config.normal_policy, geometry.points)
        normal_source = "file"
    else:
        normal_source = "estimated"
    return geometry, {
        "normal_source": normal_source,
        "normal_orientation_method": config.normal_policy.get("method"),
        "voxel_downsampling": {"applied": False, "algorithm": "none", "voxel_size_m": None, "before_point_count": len(geometry.points), "after_point_count": len(geometry.points)},
    }


def _geometry_hash_payload(config: GeometryConfig, geometry: Mesh | PointCloud, source_hash: str, processing: Mapping[str, Any]) -> dict[str, Any]:
    base: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "geometry_id": config.geometry_id,
        "dataset_id": config.dataset_id,
        "source_type": config.source_type,
        "source_hash": source_hash,
        "source_unit": config.source_unit,
        "canonical_unit": CANONICAL_UNIT,
        "source_frame": config.source_frame,
        "target_frame": config.target_frame,
        "transform_matrix": config.transform_matrix,
        "transform_provenance": {"source_frame": config.source_frame, "target_frame": config.target_frame, "matrix_hash": canonical_hash(config.transform_matrix)},
        "preprocessing_version": PREPROCESSING_VERSION,
        "preprocessing_config": config.preprocessing_config,
        "normal_policy": dict(config.normal_policy),
        "sampling_policy": dict(config.sampling_policy),
        "processing": dict(processing),
    }
    if isinstance(geometry, Mesh):
        base["vertices"] = geometry.vertices
        base["triangles"] = geometry.triangles
        base["face_normals"] = geometry.face_normals
        base["vertex_normals"] = geometry.vertex_normals
        base["degenerate_triangle_ids"] = geometry.degenerate_triangle_ids
    else:
        base["points"] = geometry.points
        base["colors"] = geometry.colors
        base["normals"] = geometry.normals
    return base


def build_geometry_manifest(config: GeometryConfig, geometry: Mesh | PointCloud, processing: Mapping[str, Any]) -> dict[str, Any]:
    source_path = Path(config.source_path)
    source_hash = sha256_file(source_path)
    bounds_min, bounds_max = geometry.bounds
    hash_payload = _geometry_hash_payload(config, geometry, source_hash, processing)
    return {
        "schema_version": SCHEMA_VERSION,
        "geometry_id": config.geometry_id,
        "dataset_id": config.dataset_id,
        "source_type": config.source_type,
        "source_path": source_path.as_posix(),
        "source_hash": source_hash,
        "geometry_hash": canonical_hash(hash_payload),
        "coordinate_frame": {"source_frame": config.source_frame, "target_frame": config.target_frame},
        "length_unit_original": config.source_unit,
        "canonical_length_unit": CANONICAL_UNIT,
        "transform_matrix": _validate_matrix(config.transform_matrix or []),
        "transform_provenance": {"source_frame": config.source_frame, "target_frame": config.target_frame, "matrix_hash": canonical_hash(config.transform_matrix)},
        "bounds_min": bounds_min,
        "bounds_max": bounds_max,
        "vertex_count": len(geometry.vertices) if isinstance(geometry, Mesh) else None,
        "triangle_count": len(geometry.triangles) if isinstance(geometry, Mesh) else None,
        "point_count": len(geometry.points) if isinstance(geometry, PointCloud) else None,
        "normal_source": processing.get("normal_source"),
        "normal_orientation_method": processing.get("normal_orientation_method"),
        "preprocessing_version": PREPROCESSING_VERSION,
        "preprocessing_config_hash": canonical_hash(config.preprocessing_config),
        "software_versions": software_identity(),
        "random_seed": config.random_seed,
        "parent_geometry_id": config.parent_geometry_id,
        "created_utc": utc_now(),
        "purpose": config.purpose,
        "dataset_role": None,
        "formal_stage3_dataset": False,
        "processing": dict(processing),
    }


def _cumulative_area(areas: Sequence[float]) -> tuple[list[float], float]:
    cumulative: list[float] = []
    total = 0.0
    for area in areas:
        total += area if area > 0.0 else 0.0
        cumulative.append(total)
    if total <= 0.0:
        raise GeometryValidationError("no_sampleable_triangles", "mesh has no non-degenerate triangle area")
    return cumulative, total


def _choose_triangle(cumulative: Sequence[float], total: float, random_value: float) -> int:
    target = random_value * total
    low, high = 0, len(cumulative)
    while low < high:
        middle = (low + high) // 2
        if target < cumulative[middle]:
            high = middle
        else:
            low = middle + 1
    return min(low, len(cumulative) - 1)


def sample_mesh_area_weighted(mesh: Mesh, sample_count: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if sample_count <= 0:
        raise GeometryValidationError("invalid_sampling", "sample_count must be positive")
    if mesh.face_normals is None or mesh.vertex_normals is None:
        raise GeometryValidationError("missing_normals", "mesh must be canonicalized before sampling")
    areas = mesh.triangle_areas
    cumulative, total_area = _cumulative_area(areas)
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    for sample_id in range(sample_count):
        triangle_id = _choose_triangle(cumulative, total_area, rng.random())
        r1, r2 = rng.random(), rng.random()
        root = math.sqrt(r1)
        u, v, w = 1.0 - root, root * (1.0 - r2), root * r2
        a, b, c = mesh.triangles[triangle_id]
        point = _add(_add(_scale(mesh.vertices[a], u), _scale(mesh.vertices[b], v)), _scale(mesh.vertices[c], w))
        normal = _unit(_add(_add(_scale(mesh.vertex_normals[a], u), _scale(mesh.vertex_normals[b], v)), _scale(mesh.vertex_normals[c], w)), field_name="sample_normal")
        records.append({
            "sample_id": sample_id,
            "triangle_id": triangle_id,
            "barycentric_u": u,
            "barycentric_v": v,
            "barycentric_w": w,
            "x": point[0], "y": point[1], "z": point[2],
            "normal_x": normal[0], "normal_y": normal[1], "normal_z": normal[2],
            "normal_source": "vertex_interpolated",
            "random_seed": seed,
            "sampling_algorithm_version": SAMPLING_ALGORITHM_VERSION,
            "sample_count": sample_count,
        })
    metadata = {
        "random_seed": seed,
        "sampling_algorithm_version": SAMPLING_ALGORITHM_VERSION,
        "sample_count": sample_count,
        "triangle_probability": [area / total_area if area > 0 else 0.0 for area in areas],
        "total_sampleable_area_m2": total_area,
        "degenerate_triangle_ids": list(mesh.degenerate_triangle_ids),
        "barycentric_policy": "sqrt(r1) uniform-area mapping; u+v+w=1 within declared floating policy",
    }
    return records, metadata


def reconstruct_sample(mesh: Mesh, record: Mapping[str, Any]) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    triangle_id = int(record["triangle_id"])
    u, v, w = (float(record[key]) for key in ("barycentric_u", "barycentric_v", "barycentric_w"))
    if abs(u + v + w - 1.0) > 1.0e-12:
        raise GeometryValidationError("invalid_sampling", "barycentric coordinates do not sum to one")
    a, b, c = mesh.triangles[triangle_id]
    point = _add(_add(_scale(mesh.vertices[a], u), _scale(mesh.vertices[b], v)), _scale(mesh.vertices[c], w))
    normal = _unit(_add(_add(_scale(mesh.vertex_normals[a], u), _scale(mesh.vertex_normals[b], v)), _scale(mesh.vertex_normals[c], w)), field_name="reconstructed_normal") if mesh.vertex_normals else mesh.face_normals[triangle_id]  # type: ignore[index]
    return point, normal


def voxel_downsample(cloud: PointCloud, *, voxel_size: Sequence[float] | float | None, enabled: bool) -> tuple[PointCloud, dict[str, Any]]:
    before = len(cloud.points)
    if not enabled:
        if voxel_size is not None:
            raise GeometryValidationError("invalid_preprocessing", "no_downsampling must explicitly use voxel_size=null")
        return cloud, {"applied": False, "algorithm": "none", "voxel_size_m": None, "before_point_count": before, "after_point_count": before, "library": "python_stdlib", "version": PREPROCESSING_VERSION}
    if voxel_size is None:
        raise GeometryValidationError("missing_preprocessing_parameter", "voxel_size is required when downsampling is enabled")
    if isinstance(voxel_size, (int, float)):
        values = (float(voxel_size),) * 3
    else:
        if len(voxel_size) != 3:
            raise GeometryValidationError("invalid_preprocessing", "voxel_size must have x/y/z values")
        values = tuple(float(item) for item in voxel_size)
    if any(not math.isfinite(item) or item <= 0.0 for item in values):
        raise GeometryValidationError("invalid_preprocessing", "voxel_size values must be finite and positive")
    cells: dict[tuple[int, int, int], int] = {}
    for index, point in enumerate(cloud.points):
        key = tuple(math.floor(point[axis] / values[axis]) for axis in range(3))
        cells.setdefault(key, index)
    selected = [cells[key] for key in sorted(cells)]
    result = PointCloud([cloud.points[index] for index in selected], colors=[cloud.colors[index] for index in selected] if cloud.colors is not None else None, normals=[cloud.normals[index] for index in selected] if cloud.normals is not None else None, source_point_count=before, source_bounds_min=cloud.source_bounds_min, source_bounds_max=cloud.source_bounds_max)
    return result, {"applied": True, "algorithm": VOXEL_ALGORITHM_VERSION, "voxel_size_m": list(values), "before_point_count": before, "after_point_count": len(result.points), "library": "python_stdlib", "version": PREPROCESSING_VERSION}


def _jacobi_smallest_eigenvector(matrix: Sequence[Sequence[float]]) -> tuple[float, float, float]:
    a = [[float(matrix[row][col]) for col in range(3)] for row in range(3)]
    v = [[1.0 if row == col else 0.0 for col in range(3)] for row in range(3)]
    for _ in range(32):
        p, q = max(((row, col) for row in range(3) for col in range(row + 1, 3)), key=lambda item: abs(a[item[0]][item[1]]))
        if abs(a[p][q]) <= 1.0e-15:
            break
        theta = 0.5 * math.atan2(2.0 * a[p][q], a[q][q] - a[p][p])
        cosine, sine = math.cos(theta), math.sin(theta)
        for index in range(3):
            aip, aiq = a[index][p], a[index][q]
            a[index][p] = cosine * aip - sine * aiq
            a[index][q] = sine * aip + cosine * aiq
        for index in range(3):
            api, aqi = a[p][index], a[q][index]
            a[p][index] = cosine * api - sine * aqi
            a[q][index] = sine * api + cosine * aqi
        for index in range(3):
            vip, viq = v[index][p], v[index][q]
            v[index][p] = cosine * vip - sine * viq
            v[index][q] = sine * vip + cosine * viq
    eigenvalues = [a[index][index] for index in range(3)]
    index = min(range(3), key=lambda item: (eigenvalues[item], item))
    return _unit([v[row][index] for row in range(3)], field_name="estimated_normal")


def estimate_point_normals(cloud: PointCloud, *, method: str, k: int, radius: float | None, max_nn: int | None, orientation_policy: Mapping[str, Any]) -> tuple[PointCloud, dict[str, Any]]:
    if method != "brute_force_covariance":
        raise GeometryValidationError("unsupported_normal_estimation", "only brute_force_covariance is selected in H2")
    if k <= 0 or k > len(cloud.points):
        raise GeometryValidationError("invalid_normal_parameters", "K must be positive and no greater than point count")
    if radius is None or not math.isfinite(radius) or radius <= 0.0:
        raise GeometryValidationError("missing_normal_parameter", "radius must be explicitly declared and positive")
    if max_nn is None or max_nn <= 0:
        raise GeometryValidationError("missing_normal_parameter", "max_nn must be explicitly declared and positive")
    normals: list[tuple[float, float, float]] = []
    for index, point in enumerate(cloud.points):
        candidates = sorted(((sum((point[axis] - other[axis]) ** 2 for axis in range(3)), other_index) for other_index, other in enumerate(cloud.points) if other_index != index and math.sqrt(sum((point[axis] - other[axis]) ** 2 for axis in range(3))) <= radius), key=lambda item: (item[0], item[1]))
        selected = [other_index for _, other_index in candidates[: min(k, max_nn)]]
        if len(selected) < 2:
            raise GeometryValidationError("insufficient_neighbors", f"point {index} has fewer than 2 declared neighbors")
        centroid = [sum(cloud.points[other][axis] for other in selected) / len(selected) for axis in range(3)]
        covariance = [[0.0] * 3 for _ in range(3)]
        for other in selected:
            delta = _sub(cloud.points[other], centroid)
            for row in range(3):
                for col in range(3):
                    covariance[row][col] += delta[row] * delta[col]
        normals.append(_jacobi_smallest_eigenvector(covariance))
    oriented = _orient_normals(normals, orientation_policy, cloud.points)
    result = PointCloud(cloud.points, colors=cloud.colors, normals=oriented, source_point_count=cloud.source_point_count, source_bounds_min=cloud.source_bounds_min, source_bounds_max=cloud.source_bounds_max)
    return result, {"method": method, "neighbor_search_method": "brute_force_sorted_euclidean_v1", "K": k, "radius_m": radius, "max_nn": max_nn, "software_version": NORMAL_ESTIMATION_VERSION, "normal_source": "estimated", "orientation": dict(orientation_policy)}


def software_identity() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "os": platform.system(),
        "architecture": platform.machine(),
        "module": PREPROCESSING_VERSION,
        "dependencies": {name: _module_version(name) for name in ("numpy", "scipy", "open3d", "trimesh")},
    }


def _module_version(name: str) -> str:
    try:
        module = importlib.import_module(name)
    except Exception:
        return "NOT_AVAILABLE"
    return str(getattr(module, "__version__", "AVAILABLE_VERSION_UNKNOWN"))


def probe_capabilities(repo_root: str | Path) -> dict[str, Any]:
    """Probe installed capabilities without installing or starting ROS nodes."""
    def import_status(name: str, import_name: str | None = None) -> dict[str, Any]:
        try:
            module = importlib.import_module(import_name or name)
            return {"status": "AVAILABLE", "version": str(getattr(module, "__version__", "AVAILABLE_VERSION_UNKNOWN")), "import": import_name or name}
        except Exception as exc:
            return {"status": "NOT_AVAILABLE", "version": None, "import": import_name or name, "error_type": type(exc).__name__}

    capabilities: dict[str, Any] = {
        "python": {"status": "AVAILABLE", "version": platform.python_version()},
        "selected_baseline": {"status": "AVAILABLE", "implementation": PREPROCESSING_VERSION, "library": "python_stdlib", "reason": "No dependency installation; deterministic STL/OBJ/PLY/XYZ implementation is selected for H2 formal baseline."},
        "stl_obj_ply_xyz_loader": {"status": "AVAILABLE", "implementation": "python_stdlib"},
        "gltf_glb_loader": {"status": "AVAILABLE_NOT_SELECTED", "reason": "H2 formal baseline selects STL/OBJ/PLY/XYZ; no GLB dependency is introduced."},
        "numpy": import_status("numpy"),
        "scipy": import_status("scipy"),
        "open3d": import_status("open3d"),
        "trimesh": import_status("trimesh"),
    }
    wsl: dict[str, Any] = {"status": "NOT_AVAILABLE", "distribution": "Ubuntu-24.04-D", "checks": {}}
    command = "source /opt/ros/jazzy/setup.bash >/dev/null 2>&1; test -d /opt/ros/jazzy && echo ros2_jazzy=AVAILABLE || echo ros2_jazzy=NOT_AVAILABLE; ros2 pkg prefix moveit_core >/dev/null 2>&1 && echo moveit_core=AVAILABLE || echo moveit_core=NOT_AVAILABLE; ros2 pkg prefix moveit_ros_planning >/dev/null 2>&1 && echo moveit_ros_planning=AVAILABLE || echo moveit_ros_planning=NOT_AVAILABLE; ros2 pkg prefix geometric_shapes >/dev/null 2>&1 && echo geometric_shapes=AVAILABLE || echo geometric_shapes=NOT_AVAILABLE; ros2 pkg prefix shape_msgs >/dev/null 2>&1 && echo shape_msgs=AVAILABLE || echo shape_msgs=NOT_AVAILABLE; ros2 pkg prefix pcl_ros >/dev/null 2>&1 && echo pcl_ros=AVAILABLE || echo pcl_ros=NOT_AVAILABLE; pkg-config --exists PCL && echo PCL=AVAILABLE || echo PCL=NOT_AVAILABLE; pkg-config --exists Eigen3 && echo Eigen3=AVAILABLE || echo Eigen3=NOT_AVAILABLE; pkg-config --exists assimp && echo assimp=AVAILABLE || echo assimp=NOT_AVAILABLE; pkg-config --exists octomap && echo octomap=AVAILABLE || echo octomap=NOT_AVAILABLE"
    try:
        completed = subprocess.run(["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", command], capture_output=True, text=False, timeout=20, check=False)
        if completed.returncode == 0:
            wsl["status"] = "AVAILABLE"
            stdout = completed.stdout.decode("utf-8", errors="replace")
            for line in stdout.splitlines():
                if "=" in line:
                    name, value = line.split("=", 1)
                    if name:
                        wsl["checks"][name] = "AVAILABLE" if value.strip() == "AVAILABLE" else "NOT_AVAILABLE"
        else:
            wsl["error"] = "wsl_probe_nonzero"
            wsl["stderr_sha256"] = sha256_bytes(completed.stderr)
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        wsl["error"] = type(exc).__name__
    capabilities["wsl_jazzy"] = wsl
    return {
        "schema_version": "stage3-h2-geometry-capability-probe-v1",
        "generated_utc": utc_now(),
        "offline_only": True,
        "dependency_install_attempted": False,
        "capabilities": capabilities,
        "dependency_provenance": {"selected": "python_stdlib", "pip_install": "NOT_RUN", "ros_environment_modified": False, "repo_root": str(Path(repo_root).resolve())},
        "notes": ["AVAILABLE means observed in this probe; AVAILABLE_NOT_SELECTED is not used by the formal baseline; no capability is inferred from collision-free or geometry-free results.", "H2 does not select mesh repair, registration, reconstruction, coverage, planning, ML, RL, ROS execution, or robot motion."],
    }


def json_dump(path: str | Path, value: Any) -> None:
    Path(path).write_text(json.dumps(_canonicalize(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def jsonl_dump(path: str | Path, records: Iterable[Mapping[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(canonical_json(record) + "\n")


def mesh_to_ply(path: str | Path, mesh: Mesh, *, include_normals: bool = True) -> None:
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        properties = ["x", "y", "z"] + (["nx", "ny", "nz"] if include_normals else [])
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(mesh.vertices)}\n")
        for prop in properties:
            handle.write(f"property float {prop}\n")
        handle.write(f"element face {len(mesh.triangles)}\nproperty list uchar int vertex_indices\nend_header\n")
        for index, point in enumerate(mesh.vertices):
            values = list(point)
            if include_normals and mesh.vertex_normals:
                values.extend(mesh.vertex_normals[index])
            handle.write(" ".join(format(float(value), ".17g") for value in values) + "\n")
        for triangle in mesh.triangles:
            handle.write("3 " + " ".join(str(index) for index in triangle) + "\n")


def pointcloud_to_ply(path: str | Path, cloud: PointCloud) -> None:
    has_normals = cloud.normals is not None
    has_colors = cloud.colors is not None
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(cloud.points)}\n")
        for prop in ("x", "y", "z"):
            handle.write(f"property float {prop}\n")
        if has_normals:
            for prop in ("nx", "ny", "nz"):
                handle.write(f"property float {prop}\n")
        if has_colors:
            for prop in ("red", "green", "blue"):
                handle.write(f"property uchar {prop}\n")
        handle.write("end_header\n")
        for index, point in enumerate(cloud.points):
            values: list[str] = [format(float(value), ".17g") for value in point]
            if has_normals:
                values.extend(format(float(value), ".17g") for value in cloud.normals[index])  # type: ignore[index]
            if has_colors:
                values.extend(str(int(value)) for value in cloud.colors[index])  # type: ignore[index]
            handle.write(" ".join(values) + "\n")


def compare_hash_identity(values: Sequence[Any]) -> bool:
    hashes = [canonical_hash(value) for value in values]
    return len(set(hashes)) == 1


def normal_validation(records: Sequence[Mapping[str, Any]], ground_truth: Sequence[Sequence[float]]) -> dict[str, Any]:
    errors = [_normal_angle_deg((record["normal_x"], record["normal_y"], record["normal_z"]), truth) for record, truth in zip(records, ground_truth)]
    finite_unit = all(all(_finite(record[key]) for key in ("normal_x", "normal_y", "normal_z")) and abs(math.sqrt(sum(float(record[key]) ** 2 for key in ("normal_x", "normal_y", "normal_z"))) - 1.0) <= 1.0e-9 for record in records)
    return {"sample_count": len(errors), "finite_unit": finite_unit, "mean_angular_error_deg": sum(errors) / len(errors) if errors else 0.0, "median_angular_error_deg": median(errors) if errors else 0.0, "p95_angular_error_deg": _percentile(errors, 0.95), "max_angular_error_deg": max(errors) if errors else 0.0, "threshold_status": "DIAGNOSTIC_ONLY_NO_H2_HARD_THRESHOLD"}
