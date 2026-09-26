"""D61 explicit-FK(q(t)) conservative articulated collision certificate.

This module is intentionally independent of the inherited D54/D60 binary.
It proves a narrower but explicit statement:

    If every required link pair has disjoint swept axis-aligned bounding boxes
    over every time subinterval, then the collision meshes are disjoint over
    the complete q(t) trajectory under the recorded URDF and jerk-bound model.

The implementation is fail-closed.  An overlapping enclosure is never called
collision-free; it is reported as ``UNRESOLVED`` unless an exact endpoint
primitive witness is available.  Meshes are enclosed by their local AABB, and
the FK transform is propagated with outward interval arithmetic.  This is a
certificate for the stated enclosure model, not a claim of exact triangle
CCD or hardware clearance.

Route B is represented by the Bernstein polynomial helper and is accepted
only when explicit cubic coefficients are supplied.  Native CSV files that
contain sampled q/dq/ddq/jerk states do not silently become polynomial
trajectories.  Route C is the conservative interval-bisection fallback used
by the certificate engine.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import struct
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from decimal import Decimal, localcontext
from pathlib import Path
from typing import ClassVar, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URDF = ROOT / "outputs" / "stage3_h4_5_robot_fixture_geometry_codesign_20260809T000000Z" / "derived_robot_model.urdf"
DEFAULT_SRDF = ROOT / "ros2_moveit_bridge" / "config" / "fairino5_v6_spray_tcp.srdf"


class CertificateInputError(ValueError):
    """Raised when an input contract is not sufficient for certification."""

    status = "BLOCKED_INVALID_INPUT"

    def __init__(self, message: str, *, code: str = "INVALID_INPUT", details: object | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = details


def _down(value: float) -> float:
    """Return a directed-downward floating-point enclosure endpoint.

    Python evaluates each elementary operation in binary64, but the rounded
    result is not itself an interval bound.  Widening with ``nextafter`` is
    the smallest representable outward step and, unlike a decimal epsilon,
    scales correctly at every magnitude (including subnormals and overflow).
    """

    value = float(value)
    if math.isnan(value):
        raise CertificateInputError("interval arithmetic produced NaN", code="NUMERIC_NAN")
    # Both infinities are already exact directed bounds.  In particular,
    # ``nextafter(+inf, -inf)`` would silently turn an overflow/unbounded
    # lower endpoint into the largest finite float and could make an
    # unbounded enclosure look finite to a downstream certificate.
    if not math.isfinite(value):
        return value
    return math.nextafter(value, -math.inf)


def _up(value: float) -> float:
    """Return a directed-upward floating-point enclosure endpoint."""

    value = float(value)
    if math.isnan(value):
        raise CertificateInputError("interval arithmetic produced NaN", code="NUMERIC_NAN")
    # Preserve both infinities for the same reason as ``_down``.  A directed
    # upper bound must not turn ``-inf`` into the most-negative finite float.
    if not math.isfinite(value):
        return value
    return math.nextafter(value, math.inf)


def _outward(lo: float, hi: float) -> "Interval":
    """Construct an outward interval, conservatively handling indeterminacy."""

    if math.isnan(float(lo)) or math.isnan(float(hi)):
        # An indeterminate endpoint (for example inf + -inf) means the
        # operation has lost useful finite information.  The full extended
        # real enclosure is sound and forces the certificate to remain
        # unresolved rather than manufacturing a finite bound.
        return Interval(-math.inf, math.inf)
    if float(lo) > float(hi):
        raise CertificateInputError(f"reversed interval result [{lo}, {hi}]", code="INTERVAL_REVERSED")
    return Interval(_down(float(lo)), _up(float(hi)))


@dataclass(frozen=True)
class Interval:
    lo: float
    hi: float
    _empty: bool = False

    # Populated after the class definition.  A named sentinel makes the
    # disjoint intersection result explicit while retaining ordinary
    # ``Interval(lo, hi)`` construction for non-empty intervals.
    EMPTY: ClassVar["Interval"]

    def __post_init__(self) -> None:
        try:
            lo = float(self.lo)
            hi = float(self.hi)
        except (TypeError, ValueError, OverflowError) as error:
            raise CertificateInputError(f"interval endpoints must be numeric: [{self.lo}, {self.hi}]") from error
        object.__setattr__(self, "lo", lo)
        object.__setattr__(self, "hi", hi)
        if self._empty:
            if lo != math.inf or hi != -math.inf:
                raise CertificateInputError("empty interval must use [+inf, -inf]", code="INTERVAL_EMPTY")
            return
        if math.isnan(lo) or math.isnan(hi) or lo > hi:
            raise CertificateInputError(f"invalid interval [{lo}, {hi}]")

    @staticmethod
    def point(value: float) -> "Interval":
        return Interval(float(value), float(value))

    @staticmethod
    def empty() -> "Interval":
        return Interval(math.inf, -math.inf, True)

    @property
    def is_empty(self) -> bool:
        return self._empty

    @property
    def width(self) -> float:
        return 0.0 if self.is_empty or self.lo == self.hi else self.hi - self.lo

    def contains(self, value: float) -> bool:
        return not self.is_empty and self.lo <= float(value) <= self.hi

    def intersect(self, other: "Interval", tolerance: float | None = None) -> "Interval":
        """Return the exact closed-interval intersection.

        ``tolerance`` is retained as a source-compatible, deprecated
        argument.  It is deliberately ignored: collapsing two disjoint
        intervals to a midpoint turns an enclosure into a value that is not
        known to belong to either interval and can create a false certificate.
        Touching endpoints remain a one-point interval; one-ULP and larger
        separations return the explicit ``EMPTY`` sentinel.
        """

        del tolerance
        if not isinstance(other, Interval):
            raise CertificateInputError("interval intersection requires another Interval", code="INVALID_INTERVAL")
        if self.is_empty or other.is_empty:
            return Interval.EMPTY
        lo = max(self.lo, other.lo)
        hi = min(self.hi, other.hi)
        if lo <= hi:
            return Interval(lo, hi)
        return Interval.EMPTY


Interval.EMPTY = Interval(math.inf, -math.inf, True)


def interval_add(a: Interval, b: Interval) -> Interval:
    if a.is_empty or b.is_empty:
        return Interval.EMPTY
    return _outward(a.lo + b.lo, a.hi + b.hi)


def interval_sub(a: Interval, b: Interval) -> Interval:
    if a.is_empty or b.is_empty:
        return Interval.EMPTY
    return _outward(a.lo - b.hi, a.hi - b.lo)


def interval_scale(a: Interval, scalar: float) -> Interval:
    if a.is_empty:
        return Interval.EMPTY
    scalar = float(scalar)
    if math.isnan(scalar) or not math.isfinite(scalar):
        raise CertificateInputError("interval scale requires a finite scalar", code="NONFINITE_SCALAR")
    if scalar == 0.0:
        return Interval.point(0.0)
    values = (a.lo * scalar, a.hi * scalar)
    return _outward(min(values), max(values))


def interval_mul(a: Interval, b: Interval) -> Interval:
    if a.is_empty or b.is_empty:
        return Interval.EMPTY
    if (a.lo == 0.0 and a.hi == 0.0) or (b.lo == 0.0 and b.hi == 0.0):
        return Interval.point(0.0)

    def product(first: float, second: float) -> float:
        value = first * second
        # IEEE 754 defines 0*inf as NaN.  For interval extension, that
        # endpoint contributes the limiting value zero; any unbounded result
        # from the other endpoint candidates is still retained.
        return 0.0 if math.isnan(value) and (first == 0.0 or second == 0.0) else value

    values = tuple(product(first, second) for first in (a.lo, a.hi) for second in (b.lo, b.hi))
    if any(math.isnan(value) for value in values):
        return Interval(-math.inf, math.inf)
    return _outward(min(values), max(values))


def interval_div(a: Interval, b: Interval) -> Interval:
    """Conservative interval quotient.

    A denominator containing zero has no finite quotient enclosure.  The
    full extended-real result is therefore returned and downstream geometry
    remains unresolved.  This is preferable to silently dropping a pole.
    """

    if a.is_empty or b.is_empty:
        return Interval.EMPTY
    if b.lo <= 0.0 <= b.hi:
        return Interval(-math.inf, math.inf)
    reciprocal = _outward(1.0 / b.hi, 1.0 / b.lo)
    return interval_mul(a, reciprocal)


def interval_pow_nonnegative(a: Interval, power: int) -> Interval:
    if a.is_empty:
        return Interval.EMPTY
    if not isinstance(power, int) or power < 0:
        raise CertificateInputError("interval power requires a nonnegative integer", code="INVALID_POWER")
    if power == 0:
        return Interval.point(1.0)
    if a.lo < 0:
        # Directed widening of an exact zero produces the least negative
        # subnormal.  This is still a sound enclosure, but it should not make
        # a nonnegative time interval fail the power helper's domain check.
        # Do not silently clamp a materially negative interval: that would
        # hide an invalid Taylor domain.
        if a.hi >= 0.0 and a.lo == math.nextafter(0.0, -math.inf):
            a = Interval(0.0, a.hi)
        else:
            raise CertificateInputError("nonnegative interval expected")
    # Reuse the directed multiplication primitive instead of ``**``.  The
    # latter is an elementary libm/power operation whose rounding contract is
    # not part of Python's API; a one-ULP widening around its result is not a
    # proof for every platform.  Repeated interval multiplication is slightly
    # wider but remains sound for finite, degenerate, and unbounded inputs.
    result = Interval.point(1.0)
    for _ in range(power):
        result = interval_mul(result, a)
    return result


def _trigonometric_interval(interval: Interval, function: str) -> Interval:
    """Compute a sound closed interval for sin or cos over ``interval``."""

    if interval.is_empty:
        return Interval.EMPTY
    # Unbounded angle domains contain every phase.  The endpoint values are
    # not evaluated because sin/cos(inf) is undefined in IEEE arithmetic.
    if not math.isfinite(interval.lo) or not math.isfinite(interval.hi):
        return Interval(-1.0, 1.0)
    lo, hi = interval.lo, interval.hi
    if hi - lo >= 2.0 * math.pi:
        return Interval(-1.0, 1.0)

    if function == "sin":
        evaluate = math.sin
        phase = math.pi / 2.0
    elif function == "cos":
        evaluate = math.cos
        phase = 0.0
    else:  # pragma: no cover - private helper contract
        raise CertificateInputError(f"unsupported trigonometric function {function!r}")

    # Widen endpoint evaluations by one representable value.  The critical
    # extrema are inserted analytically as exactly +/-1, so a libm endpoint
    # rounding error cannot lose a maximum/minimum at a boundary.
    values = [_down(evaluate(lo)), _up(evaluate(lo)), _down(evaluate(hi)), _up(evaluate(hi))]
    period = math.pi
    # Expand the search by one floating-point step at each boundary.  This is
    # an indexing guard only; any extra extremum merely widens the enclosure.
    expanded_lo = math.nextafter(lo, -math.inf)
    expanded_hi = math.nextafter(hi, math.inf)
    first = math.ceil((expanded_lo - phase) / period)
    last = math.floor((expanded_hi - phase) / period)
    for index in range(first, last + 1):
        critical = phase + index * period
        if expanded_lo <= critical <= expanded_hi:
            values.append(1.0 if index % 2 == 0 else -1.0)
    lower, upper = min(values), max(values)
    # The global extrema are exact binary64 values; preserve them exactly so
    # callers can distinguish a mathematically tight [-1, 1] result from a
    # merely widened endpoint-only enclosure.
    if lower <= -1.0:
        lower = -1.0
    if upper >= 1.0:
        upper = 1.0
    return Interval(lower, upper)


def sin_interval(interval: Interval) -> Interval:
    return _trigonometric_interval(interval, "sin")


def cos_interval(interval: Interval) -> Interval:
    return _trigonometric_interval(interval, "cos")


def _zero_matrix() -> list[list[Interval]]:
    return [[Interval.point(0.0) for _ in range(4)] for _ in range(4)]


def _identity_interval() -> list[list[Interval]]:
    result = _zero_matrix()
    for index in range(4):
        result[index][index] = Interval.point(1.0)
    return result


def interval_matrix_multiply(a: list[list[Interval]], b: list[list[Interval]]) -> list[list[Interval]]:
    if len(a) != 4 or any(len(row) != 4 for row in a) or len(b) != 4 or any(len(row) != 4 for row in b):
        raise CertificateInputError("interval matrix multiplication requires two 4x4 matrices", code="INVALID_MATRIX")
    if any(item.is_empty for row in a for item in row) or any(item.is_empty for row in b for item in row):
        return [[Interval.EMPTY for _ in range(4)] for _ in range(4)]
    result = _zero_matrix()
    for row in range(4):
        for col in range(4):
            value = Interval.point(0.0)
            for index in range(4):
                value = interval_add(value, interval_mul(a[row][index], b[index][col]))
            result[row][col] = value
    return result


_DECIMAL_PI = Decimal(
    "3.14159265358979323846264338327950288419716939937510582097494459230781640628620899"
    "86280348253421170679"
)


def _decimal_sin_cos(value: float) -> tuple[Decimal, Decimal]:
    """Evaluate sin/cos of a binary64 input in a high-precision context.

    URDF RPY origins are fixed constants, but their ordinary ``math`` matrix
    products can lose several ulps before entering the interval chain.  A
    small Decimal range-reduced Taylor evaluator gives a reproducible,
    high-precision reference without relying on an optional numerical
    package.  The result is converted to binary64 only at the API boundary
    and is widened by ``exact_to_interval`` when used for certification.
    """

    if not math.isfinite(float(value)):
        raise CertificateInputError("RPY values must be finite", code="NONFINITE_RPY")
    with localcontext() as context:
        context.prec = 150
        x = Decimal.from_float(float(value))
        pi = +_DECIMAL_PI
        two_pi = pi * Decimal(2)
        turns = int((x / two_pi).to_integral_value())
        reduced = x - Decimal(turns) * two_pi
        if reduced > pi:
            reduced -= two_pi
        elif reduced < -pi:
            reduced += two_pi
        x_squared = reduced * reduced
        sine_term = reduced
        sine = reduced
        cosine_term = Decimal(1)
        cosine = Decimal(1)
        for index in range(1, 180):
            sine_term = -(sine_term * x_squared) / Decimal((2 * index) * (2 * index + 1))
            cosine_term = -(cosine_term * x_squared) / Decimal((2 * index - 1) * (2 * index))
            sine += sine_term
            cosine += cosine_term
            if abs(sine_term) < Decimal(10) ** -140 and abs(cosine_term) < Decimal(10) ** -140:
                break
        return +sine, +cosine


def exact_rpy_matrix(rpy: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = (float(value) for value in rpy)
    if not all(math.isfinite(value) for value in (roll, pitch, yaw)):
        raise CertificateInputError("RPY values must be finite", code="NONFINITE_RPY")
    sr, cr = _decimal_sin_cos(roll)
    sp, cp = _decimal_sin_cos(pitch)
    sy, cy = _decimal_sin_cos(yaw)
    with localcontext() as context:
        context.prec = 150
        matrix = (
            (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr),
        )
        return np.asarray([[float(value) for value in row] for row in matrix], dtype=float)


def homogeneous(rotation: np.ndarray, translation: Sequence[float]) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=float)
    translation = np.asarray(translation, dtype=float)
    if rotation.shape != (3, 3) or translation.shape != (3,) or not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(translation)):
        raise CertificateInputError("homogeneous transform components must be finite 3x3 and 3-vector values", code="INVALID_TRANSFORM")
    result = np.eye(4, dtype=float)
    result[:3, :3] = rotation
    result[:3, 3] = translation
    return result


def exact_to_interval(matrix: np.ndarray) -> list[list[Interval]]:
    values = np.asarray(matrix, dtype=float)
    if values.shape != (4, 4) or not np.all(np.isfinite(values)):
        raise CertificateInputError("homogeneous transform must be a finite 4x4 matrix", code="INVALID_TRANSFORM")
    # A transform assembled with floating-point sin/cos/matrix products is a
    # rounded representation of the URDF transform.  Treat each stored value
    # as a computed constant and widen it by one ULP before it enters the
    # interval product chain.
    return [[_outward(float(values[row, col]), float(values[row, col])) for col in range(4)] for row in range(4)]


def interval_to_exact(matrix: list[list[Interval]]) -> np.ndarray:
    if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
        raise CertificateInputError("interval matrix conversion requires a 4x4 matrix", code="INVALID_MATRIX")
    if any(item.is_empty or not math.isfinite(item.lo) or not math.isfinite(item.hi) for row in matrix for item in row):
        raise CertificateInputError("cannot convert empty or unbounded interval matrix to an exact transform", code="INVALID_TRANSFORM")
    return np.array([[0.5 * (item.lo + item.hi) for item in row] for row in matrix], dtype=float)


def axis_rotation_interval(axis: Sequence[float], angle: Interval) -> list[list[Interval]]:
    axis_array = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(axis_array))
    if not math.isfinite(norm) or norm <= 0.0:
        raise CertificateInputError("revolute joint axis must be finite and nonzero")
    if angle.is_empty:
        return [[Interval.EMPTY for _ in range(4)] for _ in range(4)]

    # Derive a directed enclosure for the normalized axis from exact binary
    # input components.  Decimal is used only for the fixed normalization
    # constant; no decimal epsilon is introduced into the certificate.
    with localcontext() as context:
        context.prec = 100
        decimal_axis = [Decimal.from_float(float(value)) for value in axis_array]
        decimal_norm = sum((value * value for value in decimal_axis), Decimal(0)).sqrt()
        normalized_axis = tuple(float(value / decimal_norm) for value in decimal_axis)
    # ``exact_fk``/``axis_rotation_exact`` use NumPy's binary norm.  Retain
    # that implementation's rounded value as well as the high-precision
    # mathematical normalization; this makes the interval path enclose both
    # the recorded model quantity and the independent exact reference.
    numpy_normalized = tuple(float(value) for value in (axis_array / norm))
    axis_intervals = tuple(
        _outward(min(decimal_value, numpy_value), max(decimal_value, numpy_value))
        for decimal_value, numpy_value in zip(normalized_axis, numpy_normalized)
    )
    s = sin_interval(angle)
    c = cos_interval(angle)
    one_minus_c = interval_sub(Interval.point(1.0), c)
    skew = (
        (Interval.point(0.0), interval_scale(axis_intervals[2], -1.0), axis_intervals[1]),
        (axis_intervals[2], Interval.point(0.0), interval_scale(axis_intervals[0], -1.0)),
        (interval_scale(axis_intervals[1], -1.0), axis_intervals[0], Interval.point(0.0)),
    )
    result = _identity_interval()
    for row in range(3):
        for col in range(3):
            value = interval_mul(c, Interval.point(1.0 if row == col else 0.0))
            coefficient = interval_mul(axis_intervals[row], axis_intervals[col])
            value = interval_add(value, interval_mul(one_minus_c, coefficient))
            value = interval_add(value, interval_mul(s, skew[row][col]))
            result[row][col] = value
    return result


def _parse_floats(text: str | None, count: int, default: Sequence[float]) -> tuple[float, ...]:
    try:
        if text is None:
            values = tuple(float(value) for value in default)
        else:
            values = tuple(float(value) for value in text.split())
    except (TypeError, ValueError, OverflowError) as error:
        raise CertificateInputError(f"expected {count} numeric values, received {text!r}", code="NONNUMERIC_GEOMETRY") from error
    if len(values) != count or not all(math.isfinite(value) for value in values):
        raise CertificateInputError(f"expected {count} finite values, received {text!r}")
    return values


def _stl_aabb(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = path.read_bytes()
    vertices: list[tuple[float, float, float]] = []
    if len(data) >= 84:
        try:
            triangle_count = struct.unpack_from("<I", data, 80)[0]
        except struct.error:
            triangle_count = -1
        if triangle_count >= 0 and 84 + triangle_count * 50 <= len(data):
            for index in range(triangle_count):
                offset = 84 + index * 50 + 12
                vertices.append(struct.unpack_from("<3f", data, offset + 0))
                vertices.append(struct.unpack_from("<3f", data, offset + 12))
                vertices.append(struct.unpack_from("<3f", data, offset + 24))
    if not vertices:
        text = data.decode("utf-8", errors="ignore")
        for match in re.finditer(r"vertex\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)", text, re.IGNORECASE):
            vertices.append(tuple(float(value) for value in match.groups()))
    if not vertices:
        raise CertificateInputError(f"no STL vertices found in {path}")
    values = np.asarray(vertices, dtype=float)
    if not np.all(np.isfinite(values)):
        raise CertificateInputError(f"non-finite STL vertex found in {path}", code="NONFINITE_GEOMETRY")
    return np.min(values, axis=0), np.max(values, axis=0), values


def _resolve_mesh(filename: str, urdf_path: Path) -> Path:
    if filename.startswith("file://"):
        candidate = Path(filename[7:])
        if candidate.exists():
            return candidate
    if filename.startswith("package://"):
        remainder = filename[len("package://") :]
        package, _, relative = remainder.partition("/")
        candidates = [
            urdf_path.parent / package / relative,
            ROOT / package / relative,
            ROOT / "tmp" / package / relative,
            ROOT / "tmp" / "stage28ah0_fairino_v398" / package / relative,
        ]
        candidates.extend(ROOT.glob(f"tmp/*/{package}/{relative}"))
    else:
        candidates = [urdf_path.parent / filename, Path(filename)]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise CertificateInputError(f"collision mesh not found: {filename}")


@dataclass(frozen=True)
class Shape:
    link: str
    local_min: np.ndarray
    local_max: np.ndarray
    origin: np.ndarray
    kind: str
    half_extents: np.ndarray | None
    support_vertices: np.ndarray | None
    source: str


@dataclass(frozen=True)
class Joint:
    name: str
    joint_type: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray


@dataclass(frozen=True)
class RobotModel:
    links: tuple[str, ...]
    joints: tuple[Joint, ...]
    shapes: tuple[Shape, ...]
    # Optional URDF hard limits.  Keeping this as a defaulted field preserves
    # compatibility with older diagnostic fixtures while allowing production
    # certificate gates to bind q/velocity limits to the recorded model.
    joint_limits: Mapping[str, Mapping[str, float]] = field(default_factory=dict)

    @property
    def shape_links(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(shape.link for shape in self.shapes))


def load_model(urdf_path: Path) -> RobotModel:
    root = ET.parse(urdf_path).getroot()
    link_names = [element.attrib["name"] for element in root.findall("link")]
    links_with_shapes: list[Shape] = []
    for link_element in root.findall("link"):
        link_name = link_element.attrib["name"]
        for collision in link_element.findall("collision"):
            origin_element = collision.find("origin")
            origin = homogeneous(
                exact_rpy_matrix(_parse_floats(origin_element.attrib.get("rpy") if origin_element is not None else None, 3, (0.0, 0.0, 0.0))),
                _parse_floats(origin_element.attrib.get("xyz") if origin_element is not None else None, 3, (0.0, 0.0, 0.0)),
            )
            geometry = collision.find("geometry")
            if geometry is None or len(geometry) == 0:
                raise CertificateInputError(f"collision geometry missing for {link_name}")
            shape_element = next(iter(geometry))
            kind = shape_element.tag
            source = kind
            half_extents: np.ndarray | None = None
            if kind == "mesh":
                mesh_path = _resolve_mesh(shape_element.attrib["filename"], urdf_path)
                local_min, local_max, support_vertices = _stl_aabb(mesh_path)
                source = str(mesh_path)
            elif kind == "box":
                size = np.asarray(_parse_floats(shape_element.attrib.get("size"), 3, (0.0, 0.0, 0.0)), dtype=float)
                if np.any(size < 0.0):
                    raise CertificateInputError(f"box size must be nonnegative for {link_name}", code="INVALID_GEOMETRY")
                half_extents = 0.5 * size
                local_min, local_max = -half_extents, half_extents
                support_vertices = None
            elif kind == "sphere":
                radius = float(shape_element.attrib["radius"])
                if not math.isfinite(radius) or radius < 0.0:
                    raise CertificateInputError(f"sphere radius must be finite and nonnegative for {link_name}", code="INVALID_GEOMETRY")
                half_extents = np.array([radius, radius, radius], dtype=float)
                local_min, local_max = -half_extents, half_extents
                support_vertices = None
            elif kind == "cylinder":
                radius = float(shape_element.attrib["radius"])
                length = float(shape_element.attrib["length"])
                if not math.isfinite(radius) or not math.isfinite(length) or radius < 0.0 or length < 0.0:
                    raise CertificateInputError(f"cylinder radius/length must be finite and nonnegative for {link_name}", code="INVALID_GEOMETRY")
                local_min = np.array([-radius, -radius, -0.5 * length], dtype=float)
                local_max = np.array([radius, radius, 0.5 * length], dtype=float)
                support_vertices = None
            else:
                raise CertificateInputError(f"unsupported collision geometry <{kind}> for {link_name}")
            links_with_shapes.append(Shape(link_name, local_min, local_max, origin, kind, half_extents, support_vertices, source))
    joints: list[Joint] = []
    joint_limits: dict[str, dict[str, float]] = {}
    for element in root.findall("joint"):
        parent = element.find("parent")
        child = element.find("child")
        origin_element = element.find("origin")
        axis_element = element.find("axis")
        if parent is None or child is None:
            raise CertificateInputError(f"joint {element.attrib.get('name')} missing parent/child")
        origin = homogeneous(
            exact_rpy_matrix(_parse_floats(origin_element.attrib.get("rpy") if origin_element is not None else None, 3, (0.0, 0.0, 0.0))),
            _parse_floats(origin_element.attrib.get("xyz") if origin_element is not None else None, 3, (0.0, 0.0, 0.0)),
        )
        axis = np.asarray(_parse_floats(axis_element.attrib.get("xyz") if axis_element is not None else None, 3, (0.0, 0.0, 1.0)), dtype=float)
        joint_name = element.attrib["name"]
        joints.append(Joint(joint_name, element.attrib["type"], parent.attrib["link"], child.attrib["link"], origin, axis))
        limit_element = element.find("limit")
        if limit_element is not None:
            parsed_limits: dict[str, float] = {}
            for key in ("lower", "upper", "velocity"):
                raw_value = limit_element.attrib.get(key)
                if raw_value is not None:
                    try:
                        numeric = float(raw_value)
                    except (TypeError, ValueError, OverflowError) as error:
                        raise CertificateInputError(f"joint {joint_name} has a non-numeric {key} limit", code="INVALID_JOINT_LIMIT") from error
                    if not math.isfinite(numeric):
                        raise CertificateInputError(f"joint {joint_name} has a non-finite {key} limit", code="INVALID_JOINT_LIMIT")
                    parsed_limits[key] = numeric
            if parsed_limits:
                joint_limits[joint_name] = parsed_limits
    if not link_names:
        raise CertificateInputError("URDF has no links")
    child_names = {joint.child for joint in joints}
    roots = [name for name in link_names if name not in child_names]
    if len(roots) != 1:
        raise CertificateInputError(f"expected one URDF root, found {roots}")
    by_parent: dict[str, list[Joint]] = {}
    for joint in joints:
        by_parent.setdefault(joint.parent, []).append(joint)
    ordered_joints: list[Joint] = []
    ordered_links: list[str] = [roots[0]]
    current = roots[0]
    while by_parent.get(current):
        candidates = by_parent[current]
        if len(candidates) != 1:
            raise CertificateInputError(f"branched URDF is not supported by this linear certificate at {current}")
        joint = candidates[0]
        ordered_joints.append(joint)
        ordered_links.append(joint.child)
        current = joint.child
    reachable = set(ordered_links)
    if any(name not in reachable for name in link_names):
        raise CertificateInputError("URDF contains disconnected links")
    return RobotModel(tuple(ordered_links), tuple(ordered_joints), tuple(links_with_shapes), joint_limits)


def load_disabled_pairs(srdf_path: Path | None) -> set[tuple[str, str]]:
    if srdf_path is None:
        return set()
    root = ET.parse(srdf_path).getroot()
    result: set[tuple[str, str]] = set()
    for element in root.findall("disable_collisions"):
        first, second = element.attrib.get("link1"), element.attrib.get("link2")
        if first and second:
            result.add(tuple(sorted((first, second))))
    return result


def exact_fk(model: RobotModel, q: dict[str, float]) -> dict[str, np.ndarray]:
    if any(not math.isfinite(float(value)) for value in q.values()):
        raise CertificateInputError("FK joint positions must be finite", code="NONFINITE_JOINT_STATE")
    transforms: dict[str, np.ndarray] = {}
    current = np.eye(4, dtype=float)
    transforms[model.links[0]] = current.copy()
    joints_by_child = {joint.child: joint for joint in model.joints}
    for link in model.links[1:]:
        joint = joints_by_child[link]
        current = current @ joint.origin
        if joint.joint_type in {"revolute", "continuous"}:
            if joint.name not in q:
                raise CertificateInputError(f"missing q value for joint {joint.name}", code="MISSING_JOINT_STATE")
            current = current @ homogeneous(axis_rotation_exact(joint.axis, q[joint.name]), (0.0, 0.0, 0.0))
        transforms[link] = current.copy()
    return transforms


def axis_rotation_exact(axis: Sequence[float], angle: float) -> np.ndarray:
    axis_array = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(axis_array))
    if axis_array.shape != (3,) or not np.all(np.isfinite(axis_array)) or not math.isfinite(float(angle)) or not math.isfinite(norm) or norm <= 0.0:
        raise CertificateInputError("exact axis rotation requires finite nonzero axis and angle", code="INVALID_ROTATION")
    axis_array = axis_array / norm
    x, y, z = axis_array
    c, s = math.cos(angle), math.sin(angle)
    return np.array(
        [
            [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
            [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
            [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
        ],
        dtype=float,
    )


def interval_fk(model: RobotModel, q_intervals: dict[str, Interval]) -> dict[str, list[list[Interval]]]:
    transforms: dict[str, list[list[Interval]]] = {}
    current = _identity_interval()
    transforms[model.links[0]] = current
    joints_by_child = {joint.child: joint for joint in model.joints}
    for link in model.links[1:]:
        joint = joints_by_child[link]
        current = interval_matrix_multiply(current, exact_to_interval(joint.origin))
        if joint.joint_type in {"revolute", "continuous"}:
            if joint.name not in q_intervals:
                raise CertificateInputError(f"missing q interval for joint {joint.name}", code="MISSING_JOINT_STATE")
            current = interval_matrix_multiply(current, axis_rotation_interval(joint.axis, q_intervals[joint.name]))
        transforms[link] = current
    return transforms


def shape_corners(shape: Shape) -> Iterable[np.ndarray]:
    for mask in range(8):
        yield np.array(
            [shape.local_max[index] if (mask >> index) & 1 else shape.local_min[index] for index in range(3)]
            + [1.0],
            dtype=float,
        )


def _require_nonempty(interval: Interval, context: str) -> Interval:
    if interval.is_empty:
        raise CertificateInputError(f"empty interval enclosure in {context}", code="EMPTY_ENCLOSURE")
    return interval


def _linear_form(coefficients: Sequence[Interval], values: Sequence[float], constant: Interval | None = None) -> Interval:
    """Evaluate a finite linear form with directed interval operations."""

    result = constant if constant is not None else Interval.point(0.0)
    for coefficient, value in zip(coefficients, values):
        result = interval_add(result, interval_scale(coefficient, float(value)))
    return result


def transformed_shape_aabb(transform: list[list[Interval]], shape: Shape) -> tuple[np.ndarray, np.ndarray]:
    shape_transform = interval_matrix_multiply(transform, exact_to_interval(shape.origin))
    lower = np.full(3, math.inf, dtype=float)
    upper = np.full(3, -math.inf, dtype=float)
    if shape.support_vertices is not None:
        vertices = shape.support_vertices
        for row in range(3):
            bounds = [
                _linear_form(shape_transform[row][:3], vertex, shape_transform[row][3])
                for vertex in vertices
            ]
            if any(bound.is_empty for bound in bounds):
                raise CertificateInputError("empty mesh AABB transform enclosure", code="EMPTY_ENCLOSURE")
            lower[row] = min(bound.lo for bound in bounds)
            upper[row] = max(bound.hi for bound in bounds)
        return lower, upper
    for corner in shape_corners(shape):
        for row in range(3):
            value = _linear_form(shape_transform[row], corner)
            _require_nonempty(value, "primitive AABB transform")
            lower[row] = min(lower[row], value.lo)
            upper[row] = max(upper[row], value.hi)
    return lower, upper


def transformed_shape_projection(transform: list[list[Interval]], shape: Shape, axis: Sequence[float]) -> tuple[float, float]:
    """Bound n dot T(q) p for every p in one collision shape.

    For a mesh, extrema over the triangle surface occur at mesh vertices for
    every fixed transform.  Choosing the appropriate endpoint of each matrix
    coefficient for each vertex gives a sound joint-interval support bound.
    """

    direction = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(direction))
    if not math.isfinite(norm) or norm <= 0.0:
        raise CertificateInputError("projection axis must be finite and nonzero")
    direction = direction / norm
    shape_transform = interval_matrix_multiply(transform, exact_to_interval(shape.origin))
    coefficient_intervals: list[Interval] = []
    for col in range(3):
        coefficient_intervals.append(_linear_form([shape_transform[row][col] for row in range(3)], direction))
    translation = _linear_form([shape_transform[row][3] for row in range(3)], direction)
    if shape.support_vertices is not None:
        vertices = shape.support_vertices
        bounds = [_linear_form(coefficient_intervals, vertex, translation) for vertex in vertices]
        if any(bound.is_empty for bound in bounds):
            raise CertificateInputError("empty mesh projection enclosure", code="EMPTY_ENCLOSURE")
        lower = min(bound.lo for bound in bounds)
        upper = max(bound.hi for bound in bounds)
        return lower, upper
    lower, upper = math.inf, -math.inf
    for corner in shape_corners(shape):
        value = _linear_form(coefficient_intervals, corner[:3], translation)
        _require_nonempty(value, "primitive projection enclosure")
        lower, upper = min(lower, value.lo), max(upper, value.hi)
    return lower, upper


def transformed_shape_projections(transform: list[list[Interval]], shape: Shape, axes: Sequence[np.ndarray], *, tight: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Support bounds for all candidate axes of one shape.

    The previous vectorized ``numpy.sum`` implementation could round a lower
    support value upward (or an upper support value downward).  The number of
    candidate axes is small relative to FK evaluation cost, so use the scalar
    interval implementation as the authoritative path.  This keeps mesh and
    primitive semantics identical and makes every add/multiply directed.
    """

    direction = np.asarray(axes, dtype=float)
    if direction.ndim != 2 or direction.shape[1] != 3 or direction.shape[0] == 0 or not np.all(np.isfinite(direction)):
        raise CertificateInputError("projection axes must be finite, nonzero 3-vectors")
    norms = np.linalg.norm(direction, axis=1)
    if np.any(norms <= 0.0) or not np.all(np.isfinite(norms)):
        raise CertificateInputError("projection axes must be finite, nonzero 3-vectors")
    selected_shape = shape if tight else Shape(
        link=shape.link,
        local_min=np.asarray(shape.local_min, dtype=float),
        local_max=np.asarray(shape.local_max, dtype=float),
        origin=shape.origin,
        kind=shape.kind,
        half_extents=shape.half_extents,
        support_vertices=None,
        source=shape.source,
    )
    bounds = [transformed_shape_projection(transform, selected_shape, axis) for axis in direction]
    return np.asarray([bound[0] for bound in bounds], dtype=float), np.asarray([bound[1] for bound in bounds], dtype=float)


def _unit_axis(axis: Sequence[float]) -> np.ndarray | None:
    value = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm <= 0.0:
        return None
    return value / norm


def pair_candidate_axes(model: RobotModel, pair: tuple[str, str], exact_transforms: dict[str, np.ndarray]) -> tuple[np.ndarray, ...]:
    """Create fixed, sound candidate axes for a separating-axis query.

    The axes are only a proof search set.  Omitting a separating axis can make
    a case unresolved, but can never make a collision-free claim unsound.
    """

    axes: list[np.ndarray] = [np.eye(3, dtype=float)[index] for index in range(3)]
    first_axes: list[np.ndarray] = []
    second_axes: list[np.ndarray] = []
    for shape in model.shapes:
        if shape.link == pair[0]:
            rotation = (exact_transforms[shape.link] @ shape.origin)[:3, :3]
            first_axes.extend(rotation[:, index] for index in range(3))
        elif shape.link == pair[1]:
            rotation = (exact_transforms[shape.link] @ shape.origin)[:3, :3]
            second_axes.extend(rotation[:, index] for index in range(3))
    axes.extend(first_axes)
    axes.extend(second_axes)
    axes.extend(np.cross(first, second) for first in first_axes for second in second_axes)
    unique: list[np.ndarray] = []
    for axis in axes:
        normalized = _unit_axis(axis)
        if normalized is None:
            continue
        # Keep only exactly identical/opposite representable axes.  A decimal
        # angular tolerance could discard a genuinely distinct proof axis;
        # retaining it is cheap and can only improve resolution.
        if not any(np.array_equal(normalized, previous) or np.array_equal(normalized, -previous) for previous in unique):
            unique.append(normalized)
    return tuple(unique)


def swept_pair_separation(model: RobotModel, transforms: dict[str, list[list[Interval]]], pair: tuple[str, str], axes: Sequence[np.ndarray]) -> tuple[float, np.ndarray | None]:
    first_shapes = [shape for shape in model.shapes if shape.link == pair[0]]
    second_shapes = [shape for shape in model.shapes if shape.link == pair[1]]
    best_gap = 0.0
    best_axis: np.ndarray | None = None
    axis_array = np.asarray(axes, dtype=float)
    world_axes = np.eye(3, dtype=float)
    first_world = [transformed_shape_projections(transforms[shape.link], shape, world_axes) for shape in first_shapes]
    second_world = [transformed_shape_projections(transforms[shape.link], shape, world_axes) for shape in second_shapes]
    first_world_low = np.min(np.stack([bound[0] for bound in first_world]), axis=0)
    first_world_high = np.max(np.stack([bound[1] for bound in first_world]), axis=0)
    second_world_low = np.min(np.stack([bound[0] for bound in second_world]), axis=0)
    second_world_high = np.max(np.stack([bound[1] for bound in second_world]), axis=0)
    world_gaps = _projection_gaps(first_world_low, first_world_high, second_world_low, second_world_high)
    world_index = int(np.argmax(world_gaps))
    if world_gaps[world_index] > best_gap:
        return float(world_gaps[world_index]), world_axes[world_index]
    # A local-AABB corner pass is a cheap sound separator.  The tighter mesh
    # vertex support pass is reserved for the genuinely difficult cases.
    first_coarse = [transformed_shape_projections(transforms[shape.link], shape, axis_array, tight=False) for shape in first_shapes]
    second_coarse = [transformed_shape_projections(transforms[shape.link], shape, axis_array, tight=False) for shape in second_shapes]
    coarse_first_low = np.min(np.stack([bound[0] for bound in first_coarse]), axis=0)
    coarse_first_high = np.max(np.stack([bound[1] for bound in first_coarse]), axis=0)
    coarse_second_low = np.min(np.stack([bound[0] for bound in second_coarse]), axis=0)
    coarse_second_high = np.max(np.stack([bound[1] for bound in second_coarse]), axis=0)
    coarse_gaps = _projection_gaps(coarse_first_low, coarse_first_high, coarse_second_low, coarse_second_high)
    coarse_index = int(np.argmax(coarse_gaps))
    if coarse_gaps[coarse_index] > best_gap:
        return float(coarse_gaps[coarse_index]), axis_array[coarse_index]
    first_bounds = [transformed_shape_projections(transforms[shape.link], shape, axis_array) for shape in first_shapes]
    second_bounds = [transformed_shape_projections(transforms[shape.link], shape, axis_array) for shape in second_shapes]
    first_low = np.min(np.stack([bound[0] for bound in first_bounds]), axis=0)
    first_high = np.max(np.stack([bound[1] for bound in first_bounds]), axis=0)
    second_low = np.min(np.stack([bound[0] for bound in second_bounds]), axis=0)
    second_high = np.max(np.stack([bound[1] for bound in second_bounds]), axis=0)
    gaps = _projection_gaps(first_low, first_high, second_low, second_high)
    axis_index = int(np.argmax(gaps))
    if gaps[axis_index] > best_gap:
        best_gap = float(gaps[axis_index])
        best_axis = axis_array[axis_index]
    return best_gap, best_axis


def _projection_gaps(
    first_low: np.ndarray,
    first_high: np.ndarray,
    second_low: np.ndarray,
    second_high: np.ndarray,
) -> np.ndarray:
    """Conservative positive gaps for paired projection enclosures.

    A positive gap is only useful as a certificate when subtraction is
    rounded downward.  ``numpy.maximum`` on ordinary float subtraction can
    otherwise promote a mathematically zero/negative gap by one ULP.
    """

    first_low = np.asarray(first_low, dtype=float)
    first_high = np.asarray(first_high, dtype=float)
    second_low = np.asarray(second_low, dtype=float)
    second_high = np.asarray(second_high, dtype=float)
    if not (first_low.shape == first_high.shape == second_low.shape == second_high.shape):
        raise CertificateInputError("projection enclosure shapes do not match", code="INVALID_PROJECTION_BOUNDS")
    gaps = []
    for a_lo, a_hi, b_lo, b_hi in zip(first_low, first_high, second_low, second_high):
        if not all(math.isfinite(float(value)) for value in (a_lo, a_hi, b_lo, b_hi)):
            # Unbounded geometry cannot provide a finite positive separation
            # certificate.  Treat it as no usable gap and let the caller
            # remain unresolved.
            gaps.append(0.0)
            continue
        raw_a_before_b = float(b_lo - a_hi)
        raw_b_before_a = float(a_lo - b_hi)
        # inf - inf is indeterminate; it provides no separation evidence.
        a_before_b = 0.0 if math.isnan(raw_a_before_b) else _down(raw_a_before_b)
        b_before_a = 0.0 if math.isnan(raw_b_before_a) else _down(raw_b_before_a)
        gaps.append(max(0.0, a_before_b, b_before_a))
    return np.asarray(gaps, dtype=float)


def link_aabbs(model: RobotModel, transforms: dict[str, list[list[Interval]]]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for link in model.shape_links:
        shapes = [shape for shape in model.shapes if shape.link == link]
        lows, highs = zip(*(transformed_shape_aabb(transforms[link], shape) for shape in shapes))
        result[link] = (np.min(np.stack(lows), axis=0), np.max(np.stack(highs), axis=0))
    return result


def aabb_gap(first: tuple[np.ndarray, np.ndarray], second: tuple[np.ndarray, np.ndarray]) -> tuple[float, tuple[float, float, float]]:
    first_low, first_high = first
    second_low, second_high = second
    axis_gaps = _projection_gaps(
        np.asarray(first_low, dtype=float),
        np.asarray(first_high, dtype=float),
        np.asarray(second_low, dtype=float),
        np.asarray(second_high, dtype=float),
    )
    norm = math.sqrt(math.fsum(float(value) * float(value) for value in axis_gaps))
    return _down(norm), tuple(float(value) for value in axis_gaps)


def _box_obb(shape: Shape, link_transform: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if shape.kind != "box" or shape.half_extents is None:
        raise CertificateInputError("exact OBB witness requires box primitives")
    transform = link_transform @ shape.origin
    return transform[:3, 3], transform[:3, :3], shape.half_extents


def _obb_overlap(first: tuple[np.ndarray, np.ndarray, np.ndarray], second: tuple[np.ndarray, np.ndarray, np.ndarray]) -> bool:
    """Conservatively test endpoint OBB overlap without a decimal tolerance.

    The witness is allowed to err toward ``overlap`` (a false collision
    witness is visible and fail-closed), but must never err toward
    ``separated``.  Radius sums are rounded upward and separation projections
    downward using the same ULP-directed helpers as interval arithmetic.
    """

    center_a, axes_a, half_a = first
    center_b, axes_b, half_b = second
    rotation = axes_a.T @ axes_b
    translation = axes_a.T @ (center_b - center_a)
    abs_rotation = np.nextafter(np.abs(rotation), math.inf)
    for index in range(3):
        radius_a = half_a[index]
        radius_b = _up(float(math.fsum(float(value) for value in half_b * abs_rotation[index, :])))
        separation = _down(abs(float(translation[index])))
        if separation > _up(float(radius_a + radius_b)):
            return False
    for index in range(3):
        radius_a = _up(float(math.fsum(float(value) for value in half_a * abs_rotation[:, index])))
        radius_b = half_b[index]
        separation = _down(abs(float(np.dot(translation, rotation[:, index]))))
        if separation > _up(float(radius_a + radius_b)):
            return False
    for first_axis in range(3):
        for second_axis in range(3):
            radius_a = _up(float(
                half_a[(first_axis + 1) % 3] * abs_rotation[(first_axis + 2) % 3, second_axis]
                + half_a[(first_axis + 2) % 3] * abs_rotation[(first_axis + 1) % 3, second_axis]
            ))
            radius_b = _up(float(
                half_b[(second_axis + 1) % 3] * abs_rotation[first_axis, (second_axis + 2) % 3]
                + half_b[(second_axis + 2) % 3] * abs_rotation[first_axis, (second_axis + 1) % 3]
            ))
            separation = abs(translation[(first_axis + 2) % 3] * rotation[(first_axis + 1) % 3, second_axis] - translation[(first_axis + 1) % 3] * rotation[(first_axis + 2) % 3, second_axis])
            if _down(float(separation)) > _up(float(radius_a + radius_b)):
                return False
    return True


def exact_box_collision_witness(model: RobotModel, q: dict[str, float], pair: tuple[str, str]) -> dict[str, object] | None:
    shapes_a = [shape for shape in model.shapes if shape.link == pair[0]]
    shapes_b = [shape for shape in model.shapes if shape.link == pair[1]]
    if not shapes_a or not shapes_b or not all(shape.kind == "box" for shape in shapes_a + shapes_b):
        return None
    transforms = exact_fk(model, q)
    for shape_a in shapes_a:
        for shape_b in shapes_b:
            if _obb_overlap(_box_obb(shape_a, transforms[shape_a.link]), _box_obb(shape_b, transforms[shape_b.link])):
                return {"pair": "|".join(pair), "kind": "exact_box_obb_endpoint", "q": q}
    return None


@dataclass(frozen=True)
class Trajectory:
    times: np.ndarray
    q: dict[str, np.ndarray]
    velocity: dict[str, np.ndarray]
    acceleration: dict[str, np.ndarray]
    jerk: dict[str, np.ndarray]
    joint_names: tuple[str, ...]
    field_contract: dict[str, object]


def _trajectory_required_columns(joint_names: Sequence[str]) -> list[str]:
    return [
        "t",
        *[
            f"{joint}_{field}"
            for joint in joint_names
            for field in ("q", "dq", "ddq", "jerk")
        ],
    ]


def _trajectory_float(row: dict[str, str | None], column: str, row_index: int) -> float:
    raw = row.get(column)
    if raw is None or not str(raw).strip():
        raise CertificateInputError(
            f"trajectory row {row_index} has an empty value in {column}",
            code="MISSING_VALUE",
            details={"row": row_index, "column": column},
        )
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError) as error:
        raise CertificateInputError(
            f"trajectory row {row_index} has a non-numeric value in {column}: {raw!r}",
            code="NONNUMERIC_VALUE",
            details={"row": row_index, "column": column},
        ) from error
    if not math.isfinite(value):
        raise CertificateInputError(
            f"trajectory row {row_index} has a non-finite value in {column}",
            code="NONFINITE_VALUE",
            details={"row": row_index, "column": column},
        )
    return value


def read_trajectory(path: Path, joint_names: Sequence[str], *, require_six_dof: bool = False) -> Trajectory:
    names = tuple(str(joint) for joint in joint_names)
    if len(set(names)) != len(names) or not names:
        raise CertificateInputError("trajectory joint_names must be non-empty and unique", code="INVALID_JOINT_ORDER")
    if require_six_dof and len(names) != 6:
        raise CertificateInputError(
            f"trajectory certificate requires exactly 6 joints, received {len(names)}",
            code="INVALID_DOF",
            details={"joint_count": len(names)},
        )
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            header = reader.fieldnames
            if not header:
                raise CertificateInputError(f"trajectory CSV has no header: {path}", code="MISSING_HEADER")
            if any(field is None or not str(field).strip() for field in header):
                raise CertificateInputError("trajectory CSV contains an empty header column", code="INVALID_HEADER")
            if len(set(header)) != len(header):
                raise CertificateInputError("trajectory CSV contains duplicate header columns", code="DUPLICATE_HEADER")
            rows = list(reader)
    except CertificateInputError:
        raise
    except (OSError, csv.Error, UnicodeError) as error:
        raise CertificateInputError(f"unable to read trajectory CSV {path}: {error}", code="CSV_READ_ERROR") from error

    if len(rows) < 2:
        raise CertificateInputError(f"trajectory requires at least two rows: {path}", code="TOO_FEW_ROWS")
    required = _trajectory_required_columns(names)
    missing = [name for name in required if name not in header]
    if missing:
        if any(name.endswith("_dq") for name in missing) and any(name.endswith("_ddq") for name in missing):
            message = f"trajectory missing required dq and ddq and jerk columns: {missing}"
        else:
            message = f"trajectory missing required columns: {missing}"
        raise CertificateInputError(message, code="MISSING_COLUMNS", details={"missing_columns": missing})

    for row_index, row in enumerate(rows, start=2):
        if None in row:
            raise CertificateInputError(
                f"trajectory row {row_index} has extra fields beyond the CSV header",
                code="MALFORMED_ROW",
                details={"row": row_index},
            )

    times = np.asarray([_trajectory_float(row, "t", index) for index, row in enumerate(rows, start=2)], dtype=float)
    if not np.all(np.isfinite(times)):
        # ``_trajectory_float`` already catches this; keep the aggregate gate
        # explicit for callers that construct/modify arrays later.
        raise CertificateInputError("trajectory time must be finite", code="NONFINITE_TIME")
    differences = np.diff(times)
    if not np.all(differences > 0.0):
        bad = [index + 2 for index, delta in enumerate(differences) if not delta > 0.0]
        raise CertificateInputError(
            "trajectory time must be finite and strictly increasing",
            code="NONMONOTONIC_TIME",
            details={"row_boundaries": bad},
        )

    q = {joint: np.asarray([_trajectory_float(row, f"{joint}_q", index) for index, row in enumerate(rows, start=2)], dtype=float) for joint in names}
    velocity = {joint: np.asarray([_trajectory_float(row, f"{joint}_dq", index) for index, row in enumerate(rows, start=2)], dtype=float) for joint in names}
    acceleration = {joint: np.asarray([_trajectory_float(row, f"{joint}_ddq", index) for index, row in enumerate(rows, start=2)], dtype=float) for joint in names}
    jerk = {joint: np.asarray([_trajectory_float(row, f"{joint}_jerk", index) for index, row in enumerate(rows, start=2)], dtype=float) for joint in names}
    matrices = list(q.values()) + list(velocity.values()) + list(acceleration.values()) + list(jerk.values())
    if not all(np.all(np.isfinite(matrix)) for matrix in matrices):
        raise CertificateInputError("trajectory q/dq/ddq/jerk contains non-finite values", code="NONFINITE_STATE")
    return Trajectory(
        times,
        q,
        velocity,
        acceleration,
        jerk,
        names,
        {
            "t": True,
            "q": True,
            "dq": True,
            "ddq": True,
            "jerk": True,
            "complete": True,
            "dimension": len(names),
            "required_columns": required,
        },
    )


def _poly_interval(q0: float, v0: float, a0: float, jerk_bound: float, s: Interval) -> Interval:
    if not all(math.isfinite(float(value)) for value in (q0, v0, a0, jerk_bound)) or jerk_bound < 0.0:
        raise CertificateInputError("Taylor enclosure requires finite state and nonnegative jerk bound", code="INVALID_TAYLOR_INPUT")
    if s.is_empty:
        return Interval.EMPTY
    s2 = interval_pow_nonnegative(s, 2)
    s3 = interval_pow_nonnegative(s, 3)
    result = Interval.point(q0)
    result = interval_add(result, interval_scale(s, v0))
    result = interval_add(result, interval_scale(s2, 0.5 * a0))
    # ``jerk_bound / 6`` is itself a rounded operation.  It is used as an
    # upper magnitude below, so round that coefficient upward before the
    # directed interval product.  Without this step a very small (or a
    # 1-ULP boundary) jerk remainder can be narrowed by the scalar division.
    jerk_coefficient = 0.0 if jerk_bound == 0.0 else math.nextafter(jerk_bound / 6.0, math.inf)
    jerk_term = interval_scale(s3, jerk_coefficient)
    # The unknown jerk remainder has either sign.  Its magnitude is bounded
    # by the largest value over ``s``; retaining ``[-max, +max]`` is required
    # when the subinterval does not start at zero.
    result = interval_add(result, Interval(-jerk_term.hi, jerk_term.hi))
    return result


def q_envelope_for_subinterval(trajectory: Trajectory, index: int, fraction_lo: float, fraction_hi: float, jerk_bound: float) -> dict[str, Interval]:
    if index < 0 or index + 1 >= len(trajectory.times):
        raise CertificateInputError(f"trajectory interval index out of range: {index}", code="INVALID_INTERVAL_INDEX")
    if not all(math.isfinite(float(value)) for value in (fraction_lo, fraction_hi, jerk_bound)) or jerk_bound < 0.0:
        raise CertificateInputError("subinterval fractions and jerk bound must be finite", code="INVALID_SUBINTERVAL")
    if not 0.0 <= fraction_lo <= fraction_hi <= 1.0:
        raise CertificateInputError("subinterval fractions must satisfy 0 <= lo <= hi <= 1", code="INVALID_SUBINTERVAL")
    time0 = float(trajectory.times[index])
    time1 = float(trajectory.times[index + 1])
    raw_dt = time1 - time0
    if not math.isfinite(raw_dt) or raw_dt <= 0.0:
        raise CertificateInputError("trajectory interval duration must be finite and positive", code="INVALID_TIMEBASE")
    # Enclose the exact difference/product of the recorded binary64 times and
    # fractions.  Plain ``dt * fraction`` can round an endpoint inward before
    # the Taylor enclosure ever sees it; interval operations preserve the
    # time-domain contract at subnormal and 1-ULP boundaries.
    dt = interval_sub(Interval.point(time1), Interval.point(time0))
    start_fraction = Interval(fraction_lo, fraction_hi)
    end_fraction = interval_sub(Interval.point(1.0), start_fraction)
    start_s = interval_mul(dt, start_fraction)
    end_s = interval_mul(dt, end_fraction)
    # ``0 * [positive interval]`` is mathematically nonnegative, while the
    # outward product quite correctly widens the representable lower endpoint
    # to a tiny negative subnormal.  The time-domain contract supplies the
    # missing sign information, so restore that known domain boundary before
    # asking the nonnegative-power helper to evaluate s²/s³.
    if start_s.lo < 0.0 <= start_s.hi:
        start_s = Interval(0.0, start_s.hi)
    if end_s.lo < 0.0 <= end_s.hi:
        end_s = Interval(0.0, end_s.hi)
    result: dict[str, Interval] = {}
    for joint in trajectory.joint_names:
        start = _poly_interval(float(trajectory.q[joint][index]), float(trajectory.velocity[joint][index]), float(trajectory.acceleration[joint][index]), jerk_bound, start_s)
        end = _poly_interval(float(trajectory.q[joint][index + 1]), -float(trajectory.velocity[joint][index + 1]), float(trajectory.acceleration[joint][index + 1]), jerk_bound, end_s)
        result[joint] = start.intersect(end)
    return result


def _interval_contains_value(interval: Interval, value: float) -> bool:
    return not interval.is_empty and interval.lo <= float(value) <= interval.hi


def _jerk_moment_bounds(
    q0: float,
    v0: float,
    a0: float,
    q1: float,
    v1: float,
    a1: float,
    dt: float,
    jerk_bound: float,
) -> tuple[bool, dict[str, float]]:
    """Check necessary *joint* moment feasibility for ``|q'''| <= J``.

    Independent q/v/a endpoint cones are insufficient when the acceleration
    change consumes the available jerk budget.  Normalising jerk to
    ``u in [-1, 1]`` gives closed rearrangement bounds for the first two
    weighted moments after the zeroth moment (the acceleration change) is
    fixed.  These inequalities are necessary for every measurable/C3 jerk
    signal and therefore can only reject an impossible input; one-ULP outward
    widening prevents the floating implementation from rejecting a boundary
    that is mathematically feasible.
    """

    if not all(math.isfinite(float(value)) for value in (q0, v0, a0, q1, v1, a1, dt, jerk_bound)) or dt <= 0.0 or jerk_bound < 0.0:
        return False, {"code": "INVALID_MOMENT_INPUT"}
    delta_a = a1 - a0
    if jerk_bound == 0.0:
        expected_v = v0 + a0 * dt
        expected_q = q0 + v0 * dt + 0.5 * a0 * dt * dt
        feasible = v1 == expected_v and q1 == expected_q and a1 == a0
        return feasible, {"alpha": 0.0, "v_lower": expected_v, "v_upper": expected_v, "q_lower": expected_q, "q_upper": expected_q}
    alpha = delta_a / (jerk_bound * dt)
    if not math.isfinite(alpha) or abs(alpha) > 1.0:
        return False, {"alpha": alpha}
    # For a fixed integral of u, the decreasing weights (1-x) and
    # 0.5(1-x)^2 attain extrema by putting +1 jerk first and -1 last (or the
    # reverse).  See the closed forms in the docstring above.
    v_max_moment = (1.0 + 2.0 * alpha - alpha * alpha) / 4.0
    v_min_moment = (-1.0 + 2.0 * alpha + alpha * alpha) / 4.0
    q_max_moment = 1.0 / 6.0 - (1.0 - alpha) ** 3 / 24.0
    q_min_moment = -1.0 / 6.0 + (1.0 + alpha) ** 3 / 24.0
    v_base = v0 + a0 * dt
    q_base = q0 + v0 * dt + 0.5 * a0 * dt * dt
    v_lower = v_base + jerk_bound * dt * dt * v_min_moment
    v_upper = v_base + jerk_bound * dt * dt * v_max_moment
    q_lower = q_base + jerk_bound * dt * dt * dt * q_min_moment
    q_upper = q_base + jerk_bound * dt * dt * dt * q_max_moment
    # Widen only the accepted bounds; the input values themselves are never
    # rounded or clipped.
    v_lower = math.nextafter(v_lower, -math.inf)
    v_upper = math.nextafter(v_upper, math.inf)
    q_lower = math.nextafter(q_lower, -math.inf)
    q_upper = math.nextafter(q_upper, math.inf)
    feasible = v_lower <= v1 <= v_upper and q_lower <= q1 <= q_upper
    return feasible, {"alpha": alpha, "v_lower": v_lower, "v_upper": v_upper, "q_lower": q_lower, "q_upper": q_upper}


def validate_trajectory_contract(
    trajectory: Trajectory,
    jerk_bound: float,
    *,
    require_six_dof: bool = False,
    check_state_consistency: bool | None = None,
    joint_limits: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, object]:
    """Validate all certificate preconditions without emitting a traceback.

    The derivative transition checks are necessary conditions implied by the
    configured absolute jerk bound.  They are enabled by default for every
    trajectory; callers may explicitly disable them only for a diagnostic
    fixture.  Production certificate entry points never disable this gate.
    """

    errors: list[dict[str, object]] = []
    if check_state_consistency is None:
        check_state_consistency = True
    elif not isinstance(check_state_consistency, bool):
        errors.append({"code": "INVALID_VALIDATION_OPTION", "message": "check_state_consistency must be boolean or None"})
        check_state_consistency = bool(check_state_consistency)

    try:
        bound_value = float(jerk_bound)
    except (TypeError, ValueError, OverflowError):
        bound_value = None
    if bound_value is None or not math.isfinite(bound_value) or bound_value < 0.0:
        errors.append({"code": "INVALID_JERK_BOUND", "message": "jerk bound must be finite and nonnegative"})

    try:
        names = tuple(trajectory.joint_names)
    except (AttributeError, TypeError):
        names = tuple()
        errors.append({"code": "INVALID_JOINT_ORDER", "message": "trajectory joint_names must be a non-empty sequence"})
    try:
        unique_names = len(set(names)) == len(names)
    except TypeError:
        unique_names = False
    if not names or any(not isinstance(name, str) or not name.strip() for name in names) or not unique_names:
        errors.append({"code": "INVALID_JOINT_ORDER", "message": "trajectory joint_names must be non-empty, non-blank, unique strings"})
    if require_six_dof and len(names) != 6:
        errors.append({"code": "INVALID_DOF", "message": f"trajectory requires exactly 6 joints, received {len(names)}"})

    try:
        times = np.asarray(trajectory.times, dtype=float)
    except (AttributeError, TypeError, ValueError, OverflowError):
        times = np.asarray([], dtype=float)
        errors.append({"code": "INVALID_TIME_DIMENSION", "message": "trajectory times must be a numeric one-dimensional array"})
    if times.ndim != 1:
        errors.append({"code": "INVALID_TIME_DIMENSION", "message": "trajectory times must be one-dimensional"})
        row_count = 0
    else:
        row_count = int(times.shape[0])
    if row_count < 2:
        errors.append({"code": "TOO_FEW_ROWS", "message": "trajectory requires at least two rows"})
    elif not np.all(np.isfinite(times)):
        errors.append({"code": "NONFINITE_TIME", "message": "trajectory time must be finite"})
    elif not np.all(np.diff(times) > 0.0):
        errors.append({"code": "NONMONOTONIC_TIME", "message": "trajectory time must be strictly increasing"})

    contract = getattr(trajectory, "field_contract", None)
    required_fields = ("t", "q", "dq", "ddq", "jerk")
    if not isinstance(contract, Mapping):
        errors.append({"code": "MISSING_COLUMNS", "message": "trajectory requires t, q, dq, and jerk field metadata"})
    elif contract.get("complete") is not True or any(contract.get(field) is not True for field in required_fields):
        errors.append({"code": "MISSING_COLUMNS", "message": "trajectory requires t, q, dq, ddq, and jerk for every joint"})

    try:
        expected_keys = set(names)
    except TypeError:
        expected_keys = set()
        errors.append({"code": "INVALID_JOINT_ORDER", "message": "trajectory joint_names must contain hashable strings"})
    state_arrays: dict[str, dict[str, np.ndarray]] = {field: {} for field in ("q", "dq", "ddq", "jerk")}
    state_attributes = {"q": "q", "dq": "velocity", "ddq": "acceleration", "jerk": "jerk"}
    structural_error = bool(errors)
    for field, attribute in state_attributes.items():
        mapping = getattr(trajectory, attribute, None)
        if not isinstance(mapping, Mapping):
            errors.append({"code": "MISSING_STATE_FIELDS", "message": f"trajectory {field} values must be a mapping keyed by joint"})
            structural_error = True
            continue
        try:
            mapping_keys = set(mapping.keys())
        except TypeError:
            mapping_keys = set()
        missing_keys = sorted(expected_keys - mapping_keys, key=repr)
        extra_keys = sorted(mapping_keys - expected_keys, key=repr)
        if missing_keys:
            errors.append({"code": "MISSING_STATE_FIELDS", "message": f"trajectory {field} is missing joint values", "joints": missing_keys})
            structural_error = True
        if extra_keys:
            errors.append({"code": "UNEXPECTED_STATE_FIELDS", "message": f"trajectory {field} contains unexpected joint values", "joints": extra_keys})
            structural_error = True
        for joint in names:
            try:
                present = joint in mapping
            except TypeError:
                present = False
            if not present:
                continue
            try:
                array = np.asarray(mapping[joint], dtype=float)
            except (TypeError, ValueError, OverflowError):
                errors.append({"code": "INVALID_STATE_DIMENSION", "message": f"trajectory {field} values for {joint} are not numeric", "joint": joint})
                structural_error = True
                continue
            try:
                state_arrays[field][joint] = array
            except TypeError:
                errors.append({"code": "INVALID_JOINT_ORDER", "message": f"trajectory joint name {joint!r} is not hashable", "joint": repr(joint)})
                structural_error = True
                continue
            if array.ndim != 1 or array.shape[0] != row_count:
                errors.append({"code": "INCONSISTENT_STATE_LENGTH", "message": f"{field} arrays must be one-dimensional with one value per time row", "joint": joint, "field": field})
                structural_error = True
            elif not np.all(np.isfinite(array)):
                errors.append({"code": "NONFINITE_STATE", "message": f"trajectory {field} values must be finite", "joint": joint, "field": field})
                structural_error = True

    if not structural_error and bound_value is not None and math.isfinite(bound_value) and bound_value >= 0.0:
        # An exported jerk sample outside the configured bound is invalid,
        # not an unresolved geometry region.  No decimal tolerance is used.
        for joint in names:
            maximum = float(np.max(np.abs(state_arrays["jerk"][joint])))
            if maximum > bound_value:
                errors.append({
                    "code": "JERK_BOUND_VIOLATION",
                    "message": f"{joint} exported jerk exceeds configured bound",
                    "joint": joint,
                    "max_abs_jerk": maximum,
                    "jerk_bound": bound_value,
                })

        if check_state_consistency:
            for index, (time0, time1) in enumerate(zip(times[:-1], times[1:])):
                raw_dt = float(time1 - time0)
                if not math.isfinite(raw_dt) or raw_dt <= 0.0:
                    # The structural time gate above normally catches this.
                    # Keep the local guard so a manually assembled Trajectory
                    # can never turn a bad timebase into a passing check.
                    errors.append({
                        "code": "INVALID_TIMEBASE",
                        "message": f"trajectory interval {index} has a non-positive or non-finite duration",
                        "interval_index": index,
                    })
                    continue
                dt = interval_sub(Interval.point(float(time1)), Interval.point(float(time0)))
                for joint in names:
                    q0 = float(state_arrays["q"][joint][index])
                    q1 = float(state_arrays["q"][joint][index + 1])
                    v0 = float(state_arrays["dq"][joint][index])
                    v1 = float(state_arrays["dq"][joint][index + 1])
                    a0 = float(state_arrays["ddq"][joint][index])
                    a1 = float(state_arrays["ddq"][joint][index + 1])
                    try:
                        q_bound = _poly_interval(q0, v0, a0, bound_value, dt)
                        v_center = interval_add(Interval.point(v0), interval_scale(dt, a0))
                        velocity_coefficient = 0.0 if bound_value == 0.0 else math.nextafter(bound_value / 2.0, math.inf)
                        v_deviation = interval_scale(interval_pow_nonnegative(dt, 2), velocity_coefficient)
                        v_bound = interval_add(v_center, Interval(-v_deviation.hi, v_deviation.hi))
                        a_deviation = interval_scale(dt, bound_value)
                        a_bound = interval_add(Interval.point(a0), Interval(-a_deviation.hi, a_deviation.hi))
                    except CertificateInputError as error:
                        errors.append({
                            "code": "STATE_CONSISTENCY_UNAVAILABLE",
                            "message": str(error),
                            "joint": joint,
                            "interval_index": index,
                        })
                        continue
                    checks = ((q_bound, q1, "q"), (v_bound, v1, "dq"), (a_bound, a1, "ddq"))
                    for bound, value, field in checks:
                        if not _interval_contains_value(bound, value):
                            errors.append({
                                "code": "STATE_CONSISTENCY_VIOLATION",
                                "message": f"{joint} {field} endpoint is inconsistent with jerk-bounded transition",
                                "joint": joint,
                                "interval_index": index,
                                "field": field,
                                "value": value,
                                "allowed": [bound.lo, bound.hi],
                                "jerk_bound": bound_value,
                            })
                    moment_ok, moment_bounds = _jerk_moment_bounds(q0, v0, a0, q1, v1, a1, raw_dt, bound_value)
                    if not moment_ok:
                        errors.append({
                            "code": "STATE_MOMENT_FEASIBILITY_VIOLATION",
                            "message": f"{joint} q/dq/ddq endpoints are not jointly reachable under the jerk bound",
                            "joint": joint,
                            "interval_index": index,
                            "jerk_bound": bound_value,
                            "bounds": moment_bounds,
                            "values": {"q0": q0, "v0": v0, "a0": a0, "q1": q1, "v1": v1, "a1": a1, "dt": raw_dt},
                        })

        if joint_limits is not None:
            if not isinstance(joint_limits, Mapping):
                errors.append({"code": "INVALID_JOINT_LIMITS", "message": "joint limits must be a mapping"})
            else:
                for joint in names:
                    spec = joint_limits.get(joint)
                    if spec is None:
                        continue
                    if not isinstance(spec, Mapping):
                        errors.append({"code": "INVALID_JOINT_LIMITS", "message": f"joint limits for {joint} must be an object", "joint": joint})
                        continue
                    q_values = state_arrays["q"][joint]
                    velocity_values = state_arrays["dq"][joint]
                    acceleration_values = state_arrays["ddq"][joint]
                    for lower_key, upper_key, label, values in (
                        ("lower", "upper", "q", q_values),
                    ):
                        if lower_key in spec or upper_key in spec:
                            try:
                                lower = float(spec[lower_key]); upper = float(spec[upper_key])
                            except (KeyError, TypeError, ValueError, OverflowError):
                                errors.append({"code": "INVALID_JOINT_LIMITS", "message": f"joint {joint} position limits are incomplete or non-numeric", "joint": joint})
                                continue
                            if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
                                errors.append({"code": "INVALID_JOINT_LIMITS", "message": f"joint {joint} position limits are invalid", "joint": joint})
                            elif np.any(values < lower) or np.any(values > upper):
                                errors.append({"code": "JOINT_POSITION_LIMIT_VIOLATION", "message": f"joint {joint} q exceeds URDF position limits", "joint": joint, "lower": lower, "upper": upper})
                    for key, label, values in (("velocity", "dq", velocity_values), ("max_velocity", "dq", velocity_values), ("acceleration", "ddq", acceleration_values), ("max_acceleration", "ddq", acceleration_values)):
                        if key not in spec:
                            continue
                        try:
                            limit = float(spec[key])
                        except (TypeError, ValueError, OverflowError):
                            errors.append({"code": "INVALID_JOINT_LIMITS", "message": f"joint {joint} {key} limit is non-numeric", "joint": joint})
                            continue
                        if not math.isfinite(limit) or limit < 0.0:
                            errors.append({"code": "INVALID_JOINT_LIMITS", "message": f"joint {joint} {key} limit is invalid", "joint": joint})
                        elif np.any(np.abs(values) > limit):
                            errors.append({"code": "JOINT_{}_LIMIT_VIOLATION".format(label.upper()), "message": f"joint {joint} {label} exceeds configured limit", "joint": joint, "limit": limit})
    return {
        "status": "VALID" if not errors else "BLOCKED_INVALID_INPUT",
        "errors": errors,
        "joint_count": len(names),
        "row_count": row_count,
        "jerk_bound_rad_s3": bound_value,
        "state_consistency_checked": bool(check_state_consistency),
    }


# Short alias for callers that used the noun-first spelling in early D65
# experiments.
validate_trajectory = validate_trajectory_contract


def bernstein_cubic_interval(coefficients: Sequence[float]) -> tuple[float, float]:
    if len(coefficients) != 4 or not all(math.isfinite(float(value)) for value in coefficients):
        raise CertificateInputError("cubic polynomial requires four finite power-basis coefficients")
    # Evaluate the power-to-Bernstein conversion with exact rational 1/3 in
    # a high-precision decimal context, then enclose the conversion back to
    # binary64.  Direct ``c1 / 3.0`` can round a control point inward.
    with localcontext() as context:
        context.prec = 120
        c0, c1, c2, c3 = (Decimal.from_float(float(value)) for value in coefficients)
        controls = (
            c0,
            c0 + c1 / Decimal(3),
            c0 + Decimal(2) * c1 / Decimal(3) + c2 / Decimal(3),
            c0 + c1 + c2 + c3,
        )

    def decimal_interval(value: Decimal) -> Interval:
        converted = float(value)
        if Decimal.from_float(converted) == value:
            return Interval.point(converted)
        return _outward(converted, converted)

    control_intervals = tuple(decimal_interval(value) for value in controls)
    return min(interval.lo for interval in control_intervals), max(interval.hi for interval in control_intervals)


def route_b_probe(path: Path) -> dict[str, object]:
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            row = next(csv.DictReader(stream), None)
    except (OSError, csv.Error, UnicodeError) as error:
        return {"status": "BLOCKED_INVALID_INPUT", "reason": "unable_to_read_trajectory", "error": str(error)}
    if row is None:
        return {"status": "UNAVAILABLE", "reason": "empty_trajectory"}
    coefficient_columns = [f"j{index}_{term}" for index in range(1, 7) for term in ("c0", "c1", "c2", "c3")]
    if not all(column in row for column in coefficient_columns):
        return {"status": "UNAVAILABLE", "reason": "native_csv_has_no_explicit_piecewise_polynomial_coefficients", "reconstruction_is_not_accepted": True}
    bounds = {}
    try:
        for index in range(1, 7):
            coefficients = [float(row[f"j{index}_{term}"]) for term in ("c0", "c1", "c2", "c3")]
            bounds[f"j{index}"] = bernstein_cubic_interval(coefficients)
    except (TypeError, ValueError, OverflowError, CertificateInputError) as error:
        return {"status": "BLOCKED_INVALID_INPUT", "reason": "nonfinite_or_nonnumeric_polynomial_coefficients", "error": str(error)}
    return {"status": "AVAILABLE_FOR_EXPLICIT_COEFFICIENT_INPUT", "semantics": "cubic_power_basis_to_Bernstein_convex_hull", "first_row_bounds": bounds}


def _pair_key(first: str, second: str) -> tuple[str, str]:
    return tuple(sorted((first, second)))


def certify(urdf_path: Path, srdf_path: Path | None, trajectory_path: Path, output_path: Path, trajectory_id: str, jerk_bound: float, threshold: float, max_depth: int, max_nodes: int, max_intervals: int | None = None) -> dict[str, object]:
    if jerk_bound < 0.0 or not math.isfinite(jerk_bound):
        raise CertificateInputError("jerk bound must be finite and nonnegative", code="INVALID_JERK_BOUND")
    if threshold < 0.0 or not math.isfinite(threshold):
        raise CertificateInputError("threshold must be finite and nonnegative", code="INVALID_THRESHOLD")
    if not isinstance(max_depth, int) or max_depth < 0:
        raise CertificateInputError("max_depth must be a nonnegative integer", code="INVALID_MAX_DEPTH")
    if not isinstance(max_nodes, int) or max_nodes <= 0:
        raise CertificateInputError("max_nodes must be a positive integer", code="INVALID_MAX_NODES")
    if max_intervals is not None and (not isinstance(max_intervals, int) or max_intervals < 0):
        raise CertificateInputError("max_intervals must be a nonnegative integer", code="INVALID_MAX_INTERVALS")
    model = load_model(urdf_path)
    joint_names = tuple(joint.name for joint in model.joints if joint.joint_type in {"revolute", "continuous"})
    trajectory = read_trajectory(trajectory_path, joint_names, require_six_dof=len(joint_names) == 6)
    trajectory_validation = validate_trajectory_contract(
        trajectory,
        jerk_bound,
        require_six_dof=len(joint_names) == 6,
        joint_limits=model.joint_limits,
    )
    if trajectory_validation["status"] != "VALID":
        first_error = trajectory_validation["errors"][0]
        assert isinstance(first_error, dict)
        raise CertificateInputError(
            str(first_error.get("message", "trajectory contract validation failed")),
            code=str(first_error.get("code", "INVALID_TRAJECTORY")),
            details=trajectory_validation,
        )
    disabled = load_disabled_pairs(srdf_path)
    pairs = []
    for first_index, first in enumerate(model.shape_links):
        for second in model.shape_links[first_index + 1 :]:
            pair = _pair_key(first, second)
            if pair not in disabled:
                pairs.append(pair)
    if not pairs:
        raise CertificateInputError("no required collision pairs remain after ACM filtering", code="EMPTY_PAIR_UNIVERSE")
    counters = {"refinement_nodes": 0, "certified_regions": 0, "unresolved_regions": 0, "collision_regions": 0, "potential_overlap_regions": 0}
    witnesses: list[dict[str, object]] = []
    minimum_certified_gap = math.inf
    worst_certified: dict[str, object] | None = None
    max_joint_width = 0.0
    resource_limit_reached = False

    def evaluate(index: int, fraction_lo: float, fraction_hi: float, depth: int, pair: tuple[str, str], axes: Sequence[np.ndarray]) -> None:
        nonlocal minimum_certified_gap, worst_certified, max_joint_width, resource_limit_reached
        if counters["refinement_nodes"] >= max_nodes:
            resource_limit_reached = True
            if not witnesses or witnesses[-1].get("reason") != "max_refinement_nodes_reached":
                witnesses.append({"pair": "|".join(pair), "interval_index": index, "fraction": [fraction_lo, fraction_hi], "reason": "max_refinement_nodes_reached", "status": "UNRESOLVED"})
            return
        counters["refinement_nodes"] += 1
        q_intervals = q_envelope_for_subinterval(trajectory, index, fraction_lo, fraction_hi, jerk_bound)
        if any(interval.is_empty for interval in q_intervals.values()):
            raise CertificateInputError(
                f"empty q enclosure at trajectory interval {index} fraction [{fraction_lo}, {fraction_hi}]",
                code="EMPTY_ENCLOSURE",
            )
        max_joint_width = max(max_joint_width, max(interval.width for interval in q_intervals.values()))
        transforms = interval_fk(model, q_intervals)
        gap, separating_axis = swept_pair_separation(model, transforms, pair, axes)
        dt = float(trajectory.times[index + 1] - trajectory.times[index])
        if gap > threshold:
            counters["certified_regions"] += 1
            if gap < minimum_certified_gap:
                minimum_certified_gap = gap
                worst_certified = {"pair": "|".join(pair), "interval_index": index, "fraction": [fraction_lo, fraction_hi], "time_start_s": float(trajectory.times[index] + fraction_lo * dt), "time_end_s": float(trajectory.times[index] + fraction_hi * dt), "separating_axis_gap_m": gap, "separating_axis": separating_axis.tolist() if separating_axis is not None else None, "lower_bound_semantics": "projection gap between disjoint swept FK enclosures"}
            return
        counters["potential_overlap_regions"] += 1
        if depth < max_depth and fraction_hi > fraction_lo:
            midpoint = 0.5 * (fraction_lo + fraction_hi)
            evaluate(index, fraction_lo, midpoint, depth + 1, pair, axes)
            evaluate(index, midpoint, fraction_hi, depth + 1, pair, axes)
            return
        counters["unresolved_regions"] += 1
        witnesses.append({"pair": "|".join(pair), "interval_index": index, "fraction": [fraction_lo, fraction_hi], "time_start_s": float(trajectory.times[index] + fraction_lo * dt), "time_end_s": float(trajectory.times[index] + fraction_hi * dt), "separating_axis_gap_m": gap, "reason": "candidate swept FK projection enclosures overlap_or_touch_at_refinement_limit", "status": "UNRESOLVED"})

    requested_interval_count = len(trajectory.times) - 1
    checked_interval_count = requested_interval_count if max_intervals is None else min(requested_interval_count, max(0, max_intervals))
    for index in range(checked_interval_count):
        q_start = {joint: float(values[index]) for joint, values in trajectory.q.items()}
        start_transforms = exact_fk(model, q_start)
        for pair in pairs:
            witness = exact_box_collision_witness(model, q_start, pair)
            if witness is not None:
                witness.update({"interval_index": index, "time_s": float(trajectory.times[index]), "status": "COLLISION_FOUND"})
                witnesses.append(witness)
                counters["collision_regions"] += 1
                continue
            evaluate(index, 0.0, 1.0, 0, pair, pair_candidate_axes(model, pair, start_transforms))
            if resource_limit_reached:
                break
        if resource_limit_reached:
            break

    if counters["collision_regions"]:
        verification_result = "COLLISION_FOUND"
        certificate_status = "COLLISION"
    elif counters["unresolved_regions"] or resource_limit_reached or checked_interval_count < requested_interval_count:
        verification_result = "UNRESOLVED"
        certificate_status = "UNRESOLVED"
    else:
        verification_result = "PASS"
        certificate_status = "VALID"
    certificate = {
        "schema_version": "d61-explicit-fk-qt-interval-aabb-certificate-v1",
        "trajectory_id": trajectory_id,
        "verification_domain": "articulated_self_collision_only",
        "checked_time_interval": {"start_s": float(trajectory.times[0]), "end_s": float(trajectory.times[checked_interval_count]) if checked_interval_count else float(trajectory.times[0]), "duration_s": float(trajectory.times[checked_interval_count] - trajectory.times[0]) if checked_interval_count else 0.0, "interval_count": int(checked_interval_count), "requested_interval_count": int(requested_interval_count), "coverage_complete": bool(checked_interval_count == requested_interval_count and not resource_limit_reached)},
        "collision_pairs_checked": ["|".join(pair) for pair in pairs],
        "collision_pair_count": len(pairs),
        "minimum_clearance": None if verification_result != "PASS" else float(minimum_certified_gap),
        "minimum_certified_clearance_m": None if verification_result != "PASS" else float(minimum_certified_gap),
        "minimum_certified_gap_observed_m": None if not math.isfinite(minimum_certified_gap) else float(minimum_certified_gap),
        "worst_certified_region": worst_certified,
        "certificate_type": "CONSERVATIVE_FK_INTERVAL_SWEPT_SEPARATING_AXIS_ENCLOSURE",
        "conservativeness_bound": {
            "joint_enclosure": "endpoint Taylor enclosure q0 + dq0*s + 0.5*ddq0*s^2 +/- J*s^3/6 intersected with the reverse endpoint enclosure",
            "jerk_bound_rad_s3": jerk_bound,
            "geometry_enclosure": "each collision mesh is enclosed by interval support bounds of all STL vertices; primitive shapes use their local AABB",
            "fk_semantics": "interval homogeneous-transform propagation of URDF origin and exact axis-angle joint transforms",
            "separation_rule": "PASS only when one fixed candidate axis has a strictly positive projection gap for the full swept FK pair enclosure",
            "maximum_joint_interval_width_rad": max_joint_width,
            "unresolved_is_not_pass": True,
        },
        "failure_witness": witnesses[0] if witnesses else None,
        "failure_witness_count": len(witnesses),
        "verification_result": verification_result,
        # ``verification_result`` retains the historical vocabulary.  The
        # normalized status is explicit so VALID/COLLISION/UNRESOLVED cannot
        # be confused with input blocking.
        "status": certificate_status,
        "certificate_status": certificate_status,
        "legacy_status": verification_result,
        "collision_method": "explicit_interval_FK_q(t)_with_mesh_AABB_enclosure",
        "sampling_semantics": "trajectory rows define endpoint states only; no sampled-only PASS is emitted",
        "backend_status": {"MoveIt2": "not_used_by_this_independent_shadow", "FCL": "not_used_by_this_independent_shadow", "Bullet": "not_used_by_this_independent_shadow"},
        "input_contract": {"urdf": str(urdf_path.resolve()), "srdf": str(srdf_path.resolve()) if srdf_path else None, "trajectory": str(trajectory_path.resolve()), "trajectory_fields": trajectory.field_contract, "trajectory_validation": trajectory_validation, "joint_order": list(trajectory.joint_names), "required_pair_count": len(pairs), "disabled_pair_count": len(disabled)},
        "measurement": counters,
        "coverage": {"checked_interval_count": int(checked_interval_count), "requested_interval_count": int(requested_interval_count), "coverage_complete": bool(checked_interval_count == requested_interval_count and not resource_limit_reached), "resource_limit_reached": resource_limit_reached, "max_intervals": max_intervals, "max_nodes": max_nodes},
        "route_evidence": {"route_a": "IMPLEMENTED_AND_EXECUTED_INTERVAL_FK_SUPPORT_ENCLOSURE", "route_a_plus": "FIXED_CANDIDATE_SEPARATING_AXIS_PROJECTION", "route_b": route_b_probe(trajectory_path), "route_c": "IMPLEMENTED_AS_CERTIFICATE_REFINEMENT_BISECTION", "route_d": "EXTERNAL_BACKENDS_RESEARCHED_BUT_NOT_USED_AS_LOCAL_AUTHORITY"},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return certificate


def certify_fail_closed(*args, **kwargs) -> dict[str, object]:
    """Run :func:`certify` and convert ordinary input failures to a record.

    The strict API continues to raise ``CertificateInputError`` for library
    callers that need to distinguish invalid input programmatically.  CLI and
    orchestration callers can use this wrapper to obtain a structured
    ``BLOCKED_INVALID_INPUT`` result without a traceback.
    """

    try:
        return certify(*args, **kwargs)
    except CertificateInputError as error:
        record = {
            "schema_version": "d61-explicit-fk-qt-interval-aabb-certificate-v1",
            "status": "BLOCKED_INVALID_INPUT",
            "certificate_status": "BLOCKED_INVALID_INPUT",
            "verification_result": "BLOCKED_INVALID_INPUT",
            "error": str(error),
            "error_code": error.code,
            "error_details": error.details,
        }
        output_path = kwargs.get("output_path")
        if output_path is None and len(args) >= 4:
            output_path = args[3]
        if output_path is not None:
            try:
                output = Path(output_path).resolve()
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                record["output"] = str(output)
            except (OSError, TypeError, ValueError):
                pass
        return record
    except (OSError, ET.ParseError, TypeError, ValueError) as error:
        record = {
            "schema_version": "d61-explicit-fk-qt-interval-aabb-certificate-v1",
            "status": "BLOCKED_INVALID_INPUT",
            "certificate_status": "BLOCKED_INVALID_INPUT",
            "verification_result": "BLOCKED_INVALID_INPUT",
            "error": str(error),
            "error_code": "INPUT_OR_FILE_ERROR",
            "error_details": None,
        }
        output_path = kwargs.get("output_path")
        if output_path is None and len(args) >= 4:
            output_path = args[3]
        if output_path is not None:
            try:
                output = Path(output_path).resolve()
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                record["output"] = str(output)
            except (OSError, TypeError, ValueError):
                pass
        return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--srdf", type=Path, default=DEFAULT_SRDF)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectory-id", default="trajectory")
    parser.add_argument("--jerk-bound", type=float, default=8.0)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--max-nodes", type=int, default=20000)
    parser.add_argument("--max-intervals", type=int, default=None, help="bounded shadow window; omitted means all trajectory intervals")
    args = parser.parse_args(argv)
    try:
        certificate = certify(args.urdf.resolve(), args.srdf.resolve() if args.srdf else None, args.trajectory.resolve(), args.output.resolve(), args.trajectory_id, args.jerk_bound, args.threshold, args.max_depth, args.max_nodes, args.max_intervals)
    except (CertificateInputError, OSError, ET.ParseError, KeyError, ValueError) as error:
        payload = {
            "schema_version": "d61-explicit-fk-qt-interval-aabb-certificate-v1",
            "status": "BLOCKED_INVALID_INPUT",
            "certificate_status": "BLOCKED_INVALID_INPUT",
            "verification_result": "BLOCKED_INVALID_INPUT",
            "error": str(error),
            "error_code": getattr(error, "code", "INPUT_OR_FILE_ERROR"),
            "error_details": getattr(error, "details", None),
        }
        # Keep CLI failures machine-readable and optionally persist the same
        # fail-closed record at the requested output path.  No exception or
        # traceback is exposed for ordinary malformed-input failures.
        try:
            output = args.output.resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            payload["output"] = str(output)
        except (OSError, TypeError, ValueError):
            pass
        print(json.dumps(payload, sort_keys=True))
        return 2
    print(json.dumps({"status": certificate["status"], "output": str(args.output.resolve()), "measurement": certificate["measurement"]}, sort_keys=True))
    return 0 if certificate["status"] == "VALID" else 2


if __name__ == "__main__":
    raise SystemExit(main())
