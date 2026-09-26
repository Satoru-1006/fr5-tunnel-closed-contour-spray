"""D62 shadow certificate using a sound mesh-BVH outer-enclosure proof.

This module keeps D61's explicit interval-FK and fail-closed contracts, but
replaces the one-box-per-link mesh enclosure with a recursively partitioned
axis-aligned BVH.  A link pair is certified only when *every* BVH node pair is
separated by a positive gap on one fixed candidate axis.  A node overlap at a
leaf, missing coverage, or a resource limit remains ``UNRESOLVED``.

The BVH is an acceleration and enclosure-refinement layer, not a triangle CCD
claim.  Each node encloses the local mesh vertices assigned to it; the whole
triangle soup is therefore still enclosed conservatively by its node AABBs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

try:
    from tools.d61_fk_interval_certificate import (
        CertificateInputError,
        Shape,
        DEFAULT_SRDF,
        DEFAULT_URDF,
        exact_box_collision_witness,
        exact_fk,
        exact_to_interval,
        homogeneous,
        axis_rotation_interval,
        interval_matrix_multiply,
        Interval,
        interval_fk,
        load_disabled_pairs,
        load_model,
        pair_candidate_axes,
        q_envelope_for_subinterval,
        read_trajectory,
        transformed_shape_projections,
        validate_trajectory_contract,
    )
except ModuleNotFoundError:
    # Direct ``python tools\\d62_bvh_interval_certificate.py`` execution does
    # not put the repository root on sys.path.  Keep the script runnable both
    # as a module and as the documented CLI entry point.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools.d61_fk_interval_certificate import (
        CertificateInputError,
        Shape,
        DEFAULT_SRDF,
        DEFAULT_URDF,
        exact_box_collision_witness,
        exact_fk,
        exact_to_interval,
        homogeneous,
        axis_rotation_interval,
        interval_matrix_multiply,
        Interval,
        interval_fk,
        load_disabled_pairs,
        load_model,
        pair_candidate_axes,
        q_envelope_for_subinterval,
        read_trajectory,
        transformed_shape_projections,
        validate_trajectory_contract,
    )


@dataclass(frozen=True)
class BVHNode:
    local_min: np.ndarray
    local_max: np.ndarray
    rotation: np.ndarray
    left: "BVHNode | None" = None
    right: "BVHNode | None" = None
    point_count: int = 0

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


@dataclass
class _IntervalKinematicsCache:
    """Caches sound interval-FK factors shared by all required link pairs.

    D62 evaluates several pairs with the same common ancestor.  Rebuilding the
    same endpoint Taylor envelope, joint rotation intervals, and ancestor-to-
    descendant prefixes for every pair was pure repeated work.  The cache is
    scoped to one ``certify`` invocation and keyed by the exact time
    subinterval, so it cannot mix q-enclosures from different regions.
    """

    q_envelopes: dict[tuple[int, float, float], dict[str, Interval]]
    relative_paths: dict[tuple[int, float, float, int, int], list[list[Interval]]]
    rotations: dict[tuple[int, float, float, int], list[list[Interval]]]
    joint_origins: tuple[list[list[Interval]], ...]
    identity: list[list[Interval]]
    link_indices: dict[str, int]


def _outward_down(value: float) -> float:
    """One-step directed rounding toward negative infinity."""

    value = float(value)
    if math.isnan(value):
        return value
    if not math.isfinite(value):
        return value
    return float(np.nextafter(value, -np.inf))


def _outward_up(value: float) -> float:
    """One-step directed rounding toward positive infinity."""

    value = float(value)
    if math.isnan(value):
        return value
    if not math.isfinite(value):
        return value
    return float(np.nextafter(value, np.inf))


def _outward_sum(values: np.ndarray, direction: float) -> float:
    """Return a directed enclosure of an N-term floating-point sum.

    ``numpy.sum`` performs a finite-precision reduction whose accumulated
    error can exceed one ULP once more than a few terms are present.  A single
    ``nextafter`` around that reduction is therefore not a sound support
    bound.  Instead each binary64 addition is widened in the requested
    direction.  Inductively, every partial sum encloses the exact real sum of
    the represented terms seen so far.
    """

    try:
        flat = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return math.nan
    if flat.size == 0 or not math.isfinite(float(direction)) or direction == 0.0:
        return math.nan
    if np.any(np.isnan(flat)):
        # An indeterminate term cannot provide a separating support bound.
        return math.nan
    positive_infinity = bool(np.any(np.isposinf(flat)))
    negative_infinity = bool(np.any(np.isneginf(flat)))
    if positive_infinity or negative_infinity:
        # +inf + -inf is indeterminate; a one-sided infinite sum remains
        # infinite and is preserved as such (never narrowed to max-finite).
        if positive_infinity and negative_infinity:
            return math.nan
        return math.inf if positive_infinity else -math.inf
    outward = _outward_down if direction < 0.0 else _outward_up
    total = 0.0
    for value in flat:
        total = total + float(value)
        if math.isinf(total):
            # A directed overflow is conservative only on the matching side.
            # On the opposite side, step back to max-finite so the bound does
            # not jump across the exact finite sum.
            if total > 0.0 and direction < 0.0:
                total = math.nextafter(math.inf, -math.inf)
            elif total < 0.0 and direction > 0.0:
                total = math.nextafter(-math.inf, math.inf)
            else:
                return total
        total = outward(total)
    return total


def _outward_reduce(values: np.ndarray, axis: int, direction: float) -> np.ndarray:
    """Vectorized stepwise directed reduction along one axis.

    This retains NumPy batching across independent axes/columns while applying
    one outward rounding step after every term, so it has the same induction
    argument as :func:`_outward_sum` without falling back to Python loops for
    each projection.
    """

    array = np.asarray(values, dtype=float)
    moved = np.moveaxis(array, axis, 0)
    if moved.shape[0] == 0 or not math.isfinite(float(direction)) or direction == 0.0:
        return np.full(moved.shape[1:], math.nan, dtype=float)
    total = np.zeros(moved.shape[1:], dtype=float)
    target = -np.inf if direction < 0.0 else np.inf
    maximum = np.finfo(float).max
    with np.errstate(over="ignore", invalid="ignore"):
        for term in moved:
            previous = total
            raw = previous + term
            # If two finite operands overflow toward the side opposite the
            # requested enclosure, max-finite is the conservative endpoint;
            # a genuine infinite input remains infinite.
            if direction < 0.0:
                overflow = np.isposinf(raw) & np.isfinite(previous) & np.isfinite(term)
                raw = np.where(overflow, maximum, raw)
            else:
                overflow = np.isneginf(raw) & np.isfinite(previous) & np.isfinite(term)
                raw = np.where(overflow, -maximum, raw)
            total = np.nextafter(raw, target)
    return total


def build_bvh(points: np.ndarray, max_leaf_points: int = 64) -> BVHNode:
    """Build a median-split AABB hierarchy over complete STL triangles.

    STL vertices are stored as three consecutive vertices per facet by the
    D61 loader.  Partitioning triangles rather than individual vertices is
    essential: a triangle whose vertices are split across leaves would not be
    enclosed by the union of the leaf boxes.
    """

    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] == 0 or values.shape[0] % 3:
        raise CertificateInputError("BVH input must contain complete STL triangles as 3D vertices")
    if not np.all(np.isfinite(values)):
        raise CertificateInputError("BVH input contains non-finite points")
    if isinstance(max_leaf_points, bool) or not isinstance(max_leaf_points, int) or max_leaf_points < 1:
        raise CertificateInputError("max_leaf_points must be positive")

    triangles = values.reshape((-1, 3, 3))
    triangle_centers = np.mean(triangles, axis=1)

    def oriented_bounds(local: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return an enclosing OBB from PCA directions and exact extrema."""

        flat = local.reshape((-1, 3))
        if flat.shape[0] < 4:
            rotation = np.eye(3, dtype=float)
        else:
            centered = flat - np.mean(flat, axis=0)
            covariance = centered.T @ centered
            if np.linalg.matrix_rank(covariance, tol=1.0e-14) < 2:
                rotation = np.eye(3, dtype=float)
            else:
                _, eigenvectors = np.linalg.eigh(covariance)
                rotation = eigenvectors[:, ::-1]
                if np.linalg.det(rotation) < 0.0:
                    rotation[:, 2] *= -1.0
        coordinates = flat @ rotation
        # PCA/eigenvector and vectorized dot products are ordinary binary64
        # operations rather than directed interval arithmetic.  Inflate the
        # extrema by a conservative floating-point envelope before storing the
        # node; later projection sums widen again with nextafter.  This keeps
        # a computed OBB from becoming an accidentally inward enclosure.
        scale = np.maximum(1.0, np.max(np.abs(coordinates), axis=0))
        margin = np.nextafter(32.0 * np.finfo(float).eps * scale, np.inf)
        bounds_min = np.nextafter(np.min(coordinates, axis=0) - margin, -np.inf)
        bounds_max = np.nextafter(np.max(coordinates, axis=0) + margin, np.inf)
        return rotation, bounds_min, bounds_max

    def recurse(indices: np.ndarray) -> BVHNode:
        local = triangles[indices]
        rotation, local_min, local_max = oriented_bounds(local)
        if len(indices) <= max_leaf_points:
            return BVHNode(local_min, local_max, rotation, point_count=len(indices))
        split_axis = int(np.argmax(local_max - local_min))
        ordered = indices[np.argsort(triangle_centers[indices, split_axis], kind="mergesort")]
        midpoint = len(ordered) // 2
        if midpoint <= 0 or midpoint >= len(ordered):
            return BVHNode(local_min, local_max, rotation, point_count=len(indices))
        return BVHNode(
            local_min,
            local_max,
            rotation,
            left=recurse(ordered[:midpoint]),
            right=recurse(ordered[midpoint:]),
            point_count=len(indices),
        )

    return recurse(np.arange(triangles.shape[0], dtype=np.int64))


def _node_shape(shape: Shape, node: BVHNode) -> Shape:
    """Turn a BVH node into a conservative local-AABB shape."""

    center = 0.5 * (node.local_min + node.local_max)
    half_extents = np.nextafter(0.5 * (node.local_max - node.local_min), np.inf)
    # Reconcile the rounded center/half pair against the stored directed
    # extrema.  This avoids an inward local box when min/max straddle a large
    # exponent or when the PCA margin is subnormal.
    for index in range(3):
        lower = center[index] - half_extents[index]
        upper = center[index] + half_extents[index]
        if lower > node.local_min[index]:
            half_extents[index] = np.nextafter(half_extents[index] + (lower - node.local_min[index]), np.inf)
        if upper < node.local_max[index]:
            half_extents[index] = np.nextafter(half_extents[index] + (node.local_max[index] - upper), np.inf)
    node_origin = homogeneous(node.rotation, node.rotation @ center)

    return Shape(
        link=shape.link,
        local_min=-half_extents,
        local_max=half_extents,
        origin=shape.origin @ node_origin,
        kind="bvh_node_aabb",
        half_extents=None,
        support_vertices=None,
        source=shape.source,
    )


def _node_extent(node: BVHNode) -> float:
    return float(np.max(node.local_max - node.local_min))


def _validated_q_envelope(trajectory, interval_index: int, fraction_lo: float, fraction_hi: float, jerk_bound: float) -> dict[str, Interval]:
    """Build a non-empty jerk-cone intersection or fail the input contract."""

    q_intervals = q_envelope_for_subinterval(trajectory, interval_index, fraction_lo, fraction_hi, jerk_bound)
    empty = [joint for joint, interval in q_intervals.items() if interval.is_empty]
    if empty:
        raise CertificateInputError(
            f"empty q enclosure at trajectory interval {interval_index} fraction [{fraction_lo}, {fraction_hi}]",
            code="EMPTY_ENCLOSURE",
            details={"joints": empty, "interval_index": interval_index, "fraction": [fraction_lo, fraction_hi]},
        )
    return q_intervals


def _cached_box_projections(
    transform: list[list],
    shape: Shape,
    axes: Sequence[np.ndarray],
    shape_transform_cache: dict[tuple[int, int], list[list]],
) -> tuple[np.ndarray, np.ndarray]:
    """Project an interval-transformed node box without recomputing FK products.

    The projection is a linear form over an interval matrix and a local AABB.
    The previous reference implementation enumerated all eight box corners and
    called the Python outward-sum helper for every ``(axis, corner)`` pair.  A
    box linear form reaches its extrema at the two endpoints of each local
    coordinate, so the same enclosure can be computed with three products per
    axis and vectorized NumPy reductions.  Every product and reduction is still
    widened outward; this is an acceleration of the proof search, not a change
    to its separation rule.
    """

    direction = np.asarray(axes, dtype=float)
    if direction.ndim != 2 or direction.shape[1] != 3 or direction.shape[0] == 0 or not np.all(np.isfinite(direction)):
        raise CertificateInputError("projection axes must be finite, nonzero 3-vectors")
    norms = np.linalg.norm(direction, axis=1)
    if not np.all(np.isfinite(norms)) or np.any(norms <= 1.0e-15):
        raise CertificateInputError("projection axes must be finite, nonzero 3-vectors")
    direction = direction / norms[:, None]
    cache_key = (id(transform), id(shape))
    shape_transform = shape_transform_cache.get(cache_key)
    if shape_transform is None:
        shape_transform = interval_matrix_multiply(transform, exact_to_interval(shape.origin))
        shape_transform_cache[cache_key] = shape_transform
    matrix_lows = np.asarray([[shape_transform[row][col].lo for col in range(3)] for row in range(3)], dtype=float)
    matrix_highs = np.asarray([[shape_transform[row][col].hi for col in range(3)] for row in range(3)], dtype=float)
    translation_lows = np.asarray([shape_transform[row][3].lo for row in range(3)], dtype=float)
    translation_highs = np.asarray([shape_transform[row][3].hi for row in range(3)], dtype=float)
    # ``direction[:, :, None]`` broadcasts axes against matrix rows, yielding
    # one interval endpoint product for each (axis, row, column).  Reducing
    # the three rows reproduces the coefficient interval construction above.
    axis_matrix_lows = direction[:, :, None] * matrix_lows[None, :, :]
    axis_matrix_highs = direction[:, :, None] * matrix_highs[None, :, :]
    coefficient_lower_terms = np.nextafter(np.minimum(axis_matrix_lows, axis_matrix_highs), -np.inf)
    coefficient_upper_terms = np.nextafter(np.maximum(axis_matrix_lows, axis_matrix_highs), np.inf)
    coefficient_lows = _outward_reduce(coefficient_lower_terms, axis=1, direction=-1.0)
    coefficient_highs = _outward_reduce(coefficient_upper_terms, axis=1, direction=1.0)

    translation_products = direction * translation_lows[None, :]
    translation_products_high = direction * translation_highs[None, :]
    translation_low_terms = np.nextafter(np.minimum(translation_products, translation_products_high), -np.inf)
    translation_high_terms = np.nextafter(np.maximum(translation_products, translation_products_high), np.inf)
    translation_low = _outward_reduce(translation_low_terms, axis=1, direction=-1.0)
    translation_high = _outward_reduce(translation_high_terms, axis=1, direction=1.0)

    local_min = np.asarray(shape.local_min, dtype=float)
    local_max = np.asarray(shape.local_max, dtype=float)
    # For each local coordinate, choose the coefficient endpoint that gives a
    # lower/upper product at each AABB endpoint.  Taking min/max of those two
    # endpoint products is the exact linear-form extremum over the box.
    lower_at_min = np.where(local_min[None, :] >= 0.0, local_min[None, :] * coefficient_lows, local_min[None, :] * coefficient_highs)
    lower_at_max = np.where(local_max[None, :] >= 0.0, local_max[None, :] * coefficient_lows, local_max[None, :] * coefficient_highs)
    upper_at_min = np.where(local_min[None, :] >= 0.0, local_min[None, :] * coefficient_highs, local_min[None, :] * coefficient_lows)
    upper_at_max = np.where(local_max[None, :] >= 0.0, local_max[None, :] * coefficient_highs, local_max[None, :] * coefficient_lows)
    lower_terms = np.nextafter(np.minimum(lower_at_min, lower_at_max), -np.inf)
    upper_terms = np.nextafter(np.maximum(upper_at_min, upper_at_max), np.inf)
    lower_values = _outward_reduce(
        np.concatenate((lower_terms, translation_low[:, None]), axis=1), axis=1, direction=-1.0
    )
    upper_values = _outward_reduce(
        np.concatenate((upper_terms, translation_high[:, None]), axis=1), axis=1, direction=1.0
    )
    return lower_values, upper_values


def _box_corners(shape: Shape) -> list[np.ndarray]:
    corners: list[np.ndarray] = []
    for mask in range(8):
        corners.append(np.array(
            [shape.local_max[index] if (mask >> index) & 1 else shape.local_min[index] for index in range(3)],
            dtype=float,
        ))
    return corners


def _node_pair_separation(
    transform_a: list[list],
    transform_b: list[list],
    shape_a: Shape,
    shape_b: Shape,
    node_a: BVHNode,
    node_b: BVHNode,
    axes: Sequence[np.ndarray],
    exact_transform_a: np.ndarray,
    exact_transform_b: np.ndarray,
    threshold: float,
    node_shape_cache: dict[tuple[int, int], Shape],
    node_axis_cache: dict[tuple[int, int], tuple[np.ndarray, ...]],
    shape_transform_cache: dict[tuple[int, int], list[list]],
) -> tuple[bool, float, np.ndarray | None]:
    """Return a sound positive projection separator for one node pair."""

    shape_key_a = (id(shape_a), id(node_a))
    shape_key_b = (id(shape_b), id(node_b))
    node_shape_a = node_shape_cache.get(shape_key_a)
    if node_shape_a is None:
        node_shape_a = _node_shape(shape_a, node_a)
        node_shape_cache[shape_key_a] = node_shape_a
    node_shape_b = node_shape_cache.get(shape_key_b)
    if node_shape_b is None:
        node_shape_b = _node_shape(shape_b, node_b)
        node_shape_cache[shape_key_b] = node_shape_b
    axis_key = (id(node_a), id(node_b))
    valid_axes = node_axis_cache.get(axis_key)
    if valid_axes is None:
        node_axes = list(axes)
        node_rotation_a = (exact_transform_a @ node_shape_a.origin)[:3, :3]
        node_rotation_b = (exact_transform_b @ node_shape_b.origin)[:3, :3]
        node_axes.extend(node_rotation_a[:, index] for index in range(3))
        node_axes.extend(node_rotation_b[:, index] for index in range(3))
        node_axes.extend(np.cross(first, second) for first in node_rotation_a.T for second in node_rotation_b.T)
        candidate_axes: list[np.ndarray] = []
        for axis in node_axes:
            value = np.asarray(axis, dtype=float)
            norm = float(np.linalg.norm(value))
            if math.isfinite(norm) and norm > 1.0e-15:
                candidate_axes.append(value / norm)
        valid_axes = tuple(candidate_axes)
        node_axis_cache[axis_key] = valid_axes
    if not valid_axes:
        raise CertificateInputError("BVH node pair produced no valid projection axes")
    bounds_a = _cached_box_projections(transform_a, node_shape_a, valid_axes, shape_transform_cache)
    bounds_b = _cached_box_projections(transform_b, node_shape_b, valid_axes, shape_transform_cache)
    def directed_gap(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        raw = np.asarray(lower, dtype=float) - np.asarray(upper, dtype=float)
        # A non-finite support subtraction cannot establish a finite positive
        # separator.  In particular, do not let nextafter(+inf, -inf) turn an
        # overflow into a large finite proof margin.
        result = np.zeros(raw.shape, dtype=float)
        finite = np.isfinite(raw)
        result[finite] = np.nextafter(raw[finite], -np.inf)
        negative_infinite = np.isneginf(raw)
        result[negative_infinite] = -math.inf
        return result

    gap_ab = directed_gap(bounds_a[0], bounds_b[1])
    gap_ba = directed_gap(bounds_b[0], bounds_a[1])
    gaps = np.maximum(gap_ab, gap_ba)
    axis_index = int(np.argmax(gaps))
    gap = float(gaps[axis_index])
    return gap > threshold, gap, np.asarray(valid_axes[axis_index], dtype=float) if gap > threshold else None


def _certify_node_pair(
    node_a: BVHNode,
    node_b: BVHNode,
    transform_a: list[list],
    transform_b: list[list],
    shape_a: Shape,
    shape_b: Shape,
    axes: Sequence[np.ndarray],
    exact_transform_a: np.ndarray,
    exact_transform_b: np.ndarray,
    threshold: float,
    counters: dict[str, int],
    max_node_pairs: int,
    node_shape_cache: dict[tuple[int, int], Shape],
    node_axis_cache: dict[tuple[int, int], tuple[np.ndarray, ...]],
    shape_transform_cache: dict[tuple[int, int], list[list]],
) -> tuple[bool, float, dict[str, object] | None, bool]:
    """Recursively prove separation for all node pairs.

    The final boolean is ``resource_limit_reached``.  A false result is never
    interpreted as collision; it is an unresolved enclosure overlap.
    """

    if counters["node_pairs_visited"] >= max_node_pairs:
        counters["resource_limit_hits"] += 1
        return False, 0.0, {"reason": "max_bvh_node_pairs_reached", "status": "UNRESOLVED"}, True
    counters["node_pairs_visited"] += 1
    separated, gap, axis = _node_pair_separation(
        transform_a,
        transform_b,
        shape_a,
        shape_b,
        node_a,
        node_b,
        axes,
        exact_transform_a,
        exact_transform_b,
        threshold,
        node_shape_cache,
        node_axis_cache,
        shape_transform_cache,
    )
    if separated:
        counters["node_pairs_separated"] += 1
        return True, gap, None, False

    if node_a.is_leaf and node_b.is_leaf:
        counters["leaf_overlaps"] += 1
        return False, 0.0, {
            "reason": "BVH_leaf_outer_enclosures_overlap_or_touch",
            "status": "UNRESOLVED",
            "node_a_point_count": node_a.point_count,
            "node_b_point_count": node_b.point_count,
        }, False

    if not node_a.is_leaf and (node_b.is_leaf or _node_extent(node_a) >= _node_extent(node_b)):
        assert node_a.left is not None and node_a.right is not None
        first = _certify_node_pair(node_a.left, node_b, transform_a, transform_b, shape_a, shape_b, axes, exact_transform_a, exact_transform_b, threshold, counters, max_node_pairs, node_shape_cache, node_axis_cache, shape_transform_cache)
        second = _certify_node_pair(node_a.right, node_b, transform_a, transform_b, shape_a, shape_b, axes, exact_transform_a, exact_transform_b, threshold, counters, max_node_pairs, node_shape_cache, node_axis_cache, shape_transform_cache)
    else:
        assert node_b.left is not None and node_b.right is not None
        first = _certify_node_pair(node_a, node_b.left, transform_a, transform_b, shape_a, shape_b, axes, exact_transform_a, exact_transform_b, threshold, counters, max_node_pairs, node_shape_cache, node_axis_cache, shape_transform_cache)
        second = _certify_node_pair(node_a, node_b.right, transform_a, transform_b, shape_a, shape_b, axes, exact_transform_a, exact_transform_b, threshold, counters, max_node_pairs, node_shape_cache, node_axis_cache, shape_transform_cache)
    ok = first[0] and second[0]
    minimum_gap = min(first[1], second[1]) if ok else 0.0
    witness = first[2] or second[2]
    return ok, minimum_gap, witness, first[3] or second[3]


def _certify_pair_subinterval(
    trajectory,
    interval_index: int,
    fraction_lo: float,
    fraction_hi: float,
    depth: int,
    max_depth: int,
    jerk_bound: float,
    transform_model,
    shape_a: Shape,
    shape_b: Shape,
    node_a: BVHNode,
    node_b: BVHNode,
    axes: Sequence[np.ndarray],
    exact_transform_a: np.ndarray,
    exact_transform_b: np.ndarray,
    threshold: float,
    counters: dict[str, int],
    max_node_pairs: int,
    node_shape_cache: dict[tuple[int, int], Shape],
    node_axis_cache: dict[tuple[int, int], tuple[np.ndarray, ...]],
    interval_transform_cache: dict[tuple[int, float, float, str, str], dict[str, list[list]]],
    shape_transform_cache: dict[tuple[int, int], list[list]],
    kinematics_cache: _IntervalKinematicsCache,
) -> tuple[bool, float, dict[str, object] | None, bool]:
    """Certify one pair over a time subinterval, with fail-closed splitting."""

    counters["time_subintervals_visited"] += 1
    cache_key = (interval_index, float(fraction_lo), float(fraction_hi), shape_a.link, shape_b.link)
    transforms = interval_transform_cache.get(cache_key)
    if transforms is None:
        q_cache_key = (int(interval_index), float(fraction_lo), float(fraction_hi))
        q_intervals = kinematics_cache.q_envelopes.get(q_cache_key)
        if q_intervals is None:
            q_intervals = _validated_q_envelope(trajectory, interval_index, fraction_lo, fraction_hi, jerk_bound)
            kinematics_cache.q_envelopes[q_cache_key] = q_intervals
        transforms = _relative_interval_transforms_cached(
            transform_model,
            q_intervals,
            shape_a.link,
            shape_b.link,
            interval_index,
            fraction_lo,
            fraction_hi,
            kinematics_cache,
        )
        interval_transform_cache[cache_key] = transforms
    result = _certify_node_pair(
        node_a,
        node_b,
        transforms[shape_a.link],
        transforms[shape_b.link],
        shape_a,
        shape_b,
        axes,
        exact_transform_a,
        exact_transform_b,
        threshold,
        counters,
        max_node_pairs,
        node_shape_cache,
        node_axis_cache,
        shape_transform_cache,
    )
    if result[0] or result[3] or depth >= max_depth or fraction_hi - fraction_lo <= 1.0e-12:
        if not result[0] and depth >= max_depth:
            counters["time_leaf_unresolved"] += 1
        return result
    midpoint = 0.5 * (fraction_lo + fraction_hi)
    counters["time_subinterval_splits"] += 1
    first = _certify_pair_subinterval(
        trajectory,
        interval_index,
        fraction_lo,
        midpoint,
        depth + 1,
        max_depth,
        jerk_bound,
        transform_model,
        shape_a,
        shape_b,
        node_a,
        node_b,
        axes,
        exact_transform_a,
        exact_transform_b,
        threshold,
        counters,
        max_node_pairs,
        node_shape_cache,
        node_axis_cache,
        interval_transform_cache,
        shape_transform_cache,
        kinematics_cache,
    )
    second = _certify_pair_subinterval(
        trajectory,
        interval_index,
        midpoint,
        fraction_hi,
        depth + 1,
        max_depth,
        jerk_bound,
        transform_model,
        shape_a,
        shape_b,
        node_a,
        node_b,
        axes,
        exact_transform_a,
        exact_transform_b,
        threshold,
        counters,
        max_node_pairs,
        node_shape_cache,
        node_axis_cache,
        interval_transform_cache,
        shape_transform_cache,
        kinematics_cache,
    )
    ok = first[0] and second[0]
    minimum_gap = min(first[1], second[1]) if ok else 0.0
    return ok, minimum_gap, first[2] or second[2], first[3] or second[3]


def _shape_indices(model) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for index, shape in enumerate(model.shapes):
        result.setdefault(shape.link, []).append(index)
    return result


def _relative_interval_transforms(
    model,
    q_intervals: dict[str, Interval],
    first_link: str,
    second_link: str,
) -> dict[str, list[list[Interval]]]:
    """Enclose a serial-chain link pair in their common ancestor frame.

    Computing both link transforms from the world root independently loses the
    exact correlation of every shared upstream joint.  A serial robot permits
    a tighter equivalent representation: the ancestor link is the identity
    and only the joints between the two links are propagated.  No coordinate
    inversion is performed in interval arithmetic.
    """

    link_indices = {link: index for index, link in enumerate(model.links)}
    try:
        first_index = link_indices[first_link]
        second_index = link_indices[second_link]
    except KeyError as error:
        raise CertificateInputError(f"unknown link in relative FK pair: {error.args[0]}") from error
    ancestor_index = min(first_index, second_index)
    descendant_index = max(first_index, second_index)
    ancestor_link = model.links[ancestor_index]
    descendant_link = model.links[descendant_index]
    current = exact_to_interval(np.eye(4, dtype=float))
    result = {ancestor_link: current}
    for joint_index in range(ancestor_index, descendant_index):
        joint = model.joints[joint_index]
        current = interval_matrix_multiply(current, exact_to_interval(joint.origin))
        if joint.joint_type in {"revolute", "continuous"}:
            rotation = axis_rotation_interval(joint.axis, q_intervals.get(joint.name, Interval.point(0.0)))
            current = interval_matrix_multiply(current, rotation)
    result[descendant_link] = current
    return {first_link: result[first_link], second_link: result[second_link]}


def _relative_interval_transforms_cached(
    model,
    q_intervals: dict[str, Interval],
    first_link: str,
    second_link: str,
    interval_index: int,
    fraction_lo: float,
    fraction_hi: float,
    cache: _IntervalKinematicsCache,
) -> dict[str, list[list[Interval]]]:
    """Build a pair transform from cached ancestor-to-link prefixes.

    The uncached implementation is intentionally retained above as a small,
    readable reference.  This production path keeps the same multiplication
    order (prefix -> joint origin -> joint rotation), but reuses each prefix
    for pairs sharing an ancestor.  Since every cached value is an outward
    interval enclosure and no values are narrowed, this changes runtime only;
    it does not change the proof domain or acceptance rule.
    """

    try:
        first_index = cache.link_indices[first_link]
        second_index = cache.link_indices[second_link]
    except KeyError as error:
        raise CertificateInputError(f"unknown link in relative FK pair: {error.args[0]}") from error
    ancestor_index = min(first_index, second_index)
    descendant_index = max(first_index, second_index)
    ancestor_link = model.links[ancestor_index]
    descendant_link = model.links[descendant_index]
    prefix = (int(interval_index), float(fraction_lo), float(fraction_hi), ancestor_index)
    identity_key = (*prefix, ancestor_index)
    current = cache.relative_paths.get(identity_key)
    if current is None:
        current = cache.identity
        cache.relative_paths[identity_key] = current
    for joint_index in range(ancestor_index, descendant_index):
        path_key = (*prefix, joint_index + 1)
        next_transform = cache.relative_paths.get(path_key)
        if next_transform is None:
            joint = model.joints[joint_index]
            next_transform = interval_matrix_multiply(current, cache.joint_origins[joint_index])
            if joint.joint_type in {"revolute", "continuous"}:
                rotation_key = (int(interval_index), float(fraction_lo), float(fraction_hi), joint_index)
                rotation = cache.rotations.get(rotation_key)
                if rotation is None:
                    rotation = axis_rotation_interval(joint.axis, q_intervals.get(joint.name, Interval.point(0.0)))
                    cache.rotations[rotation_key] = rotation
                next_transform = interval_matrix_multiply(next_transform, rotation)
            cache.relative_paths[path_key] = next_transform
        current = next_transform
    return {ancestor_link: cache.relative_paths[(*prefix, ancestor_index)], descendant_link: current}.copy()


def certify(
    urdf_path: Path,
    srdf_path: Path | None,
    trajectory_path: Path,
    output_path: Path,
    trajectory_id: str,
    jerk_bound: float,
    threshold: float,
    max_leaf_points: int,
    max_node_pairs: int,
    max_time_depth: int = 8,
    max_intervals: int | None = None,
    interval_indices: Sequence[int] | None = None,
) -> dict[str, object]:
    started_wall = time.perf_counter()
    try:
        jerk_bound = float(jerk_bound)
    except (TypeError, ValueError, OverflowError) as error:
        raise CertificateInputError("jerk bound must be finite and nonnegative", code="INVALID_JERK_BOUND") from error
    try:
        threshold = float(threshold)
    except (TypeError, ValueError, OverflowError) as error:
        raise CertificateInputError("threshold must be finite and nonnegative", code="INVALID_THRESHOLD") from error
    if jerk_bound < 0.0 or not math.isfinite(jerk_bound):
        raise CertificateInputError("jerk bound must be finite and nonnegative", code="INVALID_JERK_BOUND")
    if threshold < 0.0 or not math.isfinite(threshold):
        raise CertificateInputError("threshold must be finite and nonnegative", code="INVALID_THRESHOLD")
    if isinstance(max_leaf_points, bool) or not isinstance(max_leaf_points, int) or max_leaf_points < 1:
        raise CertificateInputError("max_leaf_points must be a positive integer", code="INVALID_MAX_LEAF_POINTS")
    if isinstance(max_node_pairs, bool) or not isinstance(max_node_pairs, int) or max_node_pairs <= 0:
        raise CertificateInputError("max_node_pairs must be a positive integer", code="INVALID_MAX_NODE_PAIRS")
    if isinstance(max_time_depth, bool) or not isinstance(max_time_depth, int) or max_time_depth < 0:
        raise CertificateInputError("max_time_depth must be a nonnegative integer", code="INVALID_MAX_TIME_DEPTH")
    if max_intervals is not None and (isinstance(max_intervals, bool) or not isinstance(max_intervals, int) or max_intervals < 0):
        raise CertificateInputError("max_intervals must be a nonnegative integer", code="INVALID_MAX_INTERVALS")
    if interval_indices is not None:
        if not isinstance(interval_indices, Sequence) or isinstance(interval_indices, (str, bytes)):
            raise CertificateInputError("interval_indices must be an integer sequence", code="INVALID_INTERVAL_INDICES")
        if any(isinstance(index, bool) or not isinstance(index, int) for index in interval_indices):
            raise CertificateInputError("interval_indices must contain integers only", code="INVALID_INTERVAL_INDICES")
        if list(interval_indices) != sorted(set(interval_indices)):
            raise CertificateInputError("interval_indices must be sorted and duplicate-free", code="INVALID_INTERVAL_INDICES")
    model = load_model(urdf_path)
    joint_order = [joint.name for joint in model.joints if joint.joint_type in {"revolute", "continuous"}]
    trajectory = read_trajectory(trajectory_path, joint_order)
    trajectory_validation = validate_trajectory_contract(
        trajectory,
        jerk_bound,
        require_six_dof=len(joint_order) == 6,
        check_state_consistency=len(joint_order) == 6,
        joint_limits=model.joint_limits,
    )
    if trajectory_validation["status"] != "VALID":
        first_error = trajectory_validation["errors"][0] if trajectory_validation["errors"] else {"message": "invalid trajectory contract"}
        raise CertificateInputError(
            str(first_error.get("message", "invalid trajectory contract")),
            code=str(first_error.get("code", "INVALID_TRAJECTORY_CONTRACT")),
            details=trajectory_validation,
        )
    disabled = load_disabled_pairs(srdf_path)
    shape_indices = _shape_indices(model)
    pairs: list[tuple[str, str]] = []
    for first_index, first in enumerate(model.shape_links):
        for second in model.shape_links[first_index + 1 :]:
            pair = tuple(sorted((first, second)))
            if pair not in disabled:
                pairs.append(pair)
    if not pairs:
        raise CertificateInputError("no required collision pairs remain after ACM filtering")

    bvhs: dict[int, BVHNode] = {}
    for index, shape in enumerate(model.shapes):
        if shape.support_vertices is not None:
            bvhs[index] = build_bvh(shape.support_vertices, max_leaf_points)
        else:
            bvhs[index] = BVHNode(shape.local_min, shape.local_max, np.eye(3, dtype=float), point_count=8)

    counters = {
        "intervals_evaluated": 0,
        "certified_regions": 0,
        "unresolved_regions": 0,
        "collision_regions": 0,
        "node_pairs_visited": 0,
        "node_pairs_separated": 0,
        "leaf_overlaps": 0,
        "resource_limit_hits": 0,
        "time_subintervals_visited": 0,
        "time_subinterval_splits": 0,
        "time_leaf_unresolved": 0,
    }
    witnesses: list[dict[str, object]] = []
    minimum_gap = math.inf
    worst_certified: dict[str, object] | None = None
    resource_limit_reached = False
    node_shape_cache: dict[tuple[int, int], Shape] = {}
    kinematics_cache = _IntervalKinematicsCache(
        q_envelopes={},
        relative_paths={},
        rotations={},
        joint_origins=tuple(exact_to_interval(joint.origin) for joint in model.joints),
        identity=exact_to_interval(np.eye(4, dtype=float)),
        link_indices={link: index for index, link in enumerate(model.links)},
    )

    requested_interval_count = len(trajectory.times) - 1
    if interval_indices is None:
        selected_interval_indices = list(range(
            requested_interval_count if max_intervals is None else min(requested_interval_count, max(0, max_intervals))
        ))
        explicit_interval_selection = False
    else:
        # The explicit selection is part of the evidence identity.  Do not
        # coerce floats/strings or silently drop duplicates: either input
        # shape would make the reported coverage differ from what the caller
        # requested and must block the certificate.
        selected_interval_indices = list(interval_indices)
        if max_intervals is not None:
            selected_interval_indices = selected_interval_indices[: max(0, max_intervals)]
        if any(index < 0 or index >= requested_interval_count for index in selected_interval_indices):
            raise CertificateInputError("interval_indices contains an out-of-range interval")
        explicit_interval_selection = True
    checked_interval_count = len(selected_interval_indices)
    certified_interval_indices: list[int] = []
    unresolved_interval_indices: list[int] = []
    collision_interval_indices: list[int] = []
    for interval_index in selected_interval_indices:
        q_start = {joint: float(values[interval_index]) for joint, values in trajectory.q.items()}
        exact_transforms = exact_fk(model, q_start)
        dt = float(trajectory.times[interval_index + 1] - trajectory.times[interval_index])
        interval_ok = True
        interval_gap = math.inf
        interval_witness: dict[str, object] | None = None
        interval_transform_cache: dict[tuple[int, float, float, str, str], dict[str, list[list]]] = {}
        shape_transform_cache: dict[tuple[int, int], list[list]] = {}
        for pair in pairs:
            endpoint_witness = exact_box_collision_witness(model, q_start, pair)
            if endpoint_witness is not None:
                endpoint_witness.update({"interval_index": interval_index, "time_s": float(trajectory.times[interval_index]), "status": "COLLISION_FOUND"})
                witnesses.append(endpoint_witness)
                counters["collision_regions"] += 1
                interval_ok = False
                interval_witness = endpoint_witness
                continue
            pair_ok = True
            pair_gap = math.inf
            pair_witness: dict[str, object] | None = None
            first_index = model.links.index(pair[0])
            second_index = model.links.index(pair[1])
            ancestor_link = model.links[min(first_index, second_index)]
            ancestor_inverse = np.linalg.inv(exact_transforms[ancestor_link])
            pair_exact_transforms = {
                pair[0]: ancestor_inverse @ exact_transforms[pair[0]],
                pair[1]: ancestor_inverse @ exact_transforms[pair[1]],
            }
            axes = pair_candidate_axes(model, pair, pair_exact_transforms)
            for shape_a_index in shape_indices[pair[0]]:
                for shape_b_index in shape_indices[pair[1]]:
                    shape_a = model.shapes[shape_a_index]
                    shape_b = model.shapes[shape_b_index]
                    node_axis_cache: dict[tuple[int, int], tuple[np.ndarray, ...]] = {}
                    result = _certify_pair_subinterval(
                        trajectory,
                        interval_index,
                        0.0,
                        1.0,
                        0,
                        max_time_depth,
                        jerk_bound,
                        model,
                        shape_a,
                        shape_b,
                        bvhs[shape_a_index],
                        bvhs[shape_b_index],
                        axes,
                        pair_exact_transforms[shape_a.link],
                        pair_exact_transforms[shape_b.link],
                        threshold,
                        counters,
                        max_node_pairs,
                        node_shape_cache,
                        node_axis_cache,
                        interval_transform_cache,
                        shape_transform_cache,
                        kinematics_cache,
                    )
                    pair_ok = pair_ok and result[0]
                    if result[0]:
                        pair_gap = min(pair_gap, result[1])
                    elif pair_witness is None:
                        pair_witness = result[2]
                    resource_limit_reached = resource_limit_reached or result[3]
            if not pair_ok:
                interval_ok = False
                if interval_witness is None:
                    interval_witness = {
                        "pair": "|".join(pair),
                        "interval_index": interval_index,
                        "fraction": [0.0, 1.0],
                        "time_start_s": float(trajectory.times[interval_index]),
                        "time_end_s": float(trajectory.times[interval_index + 1]),
                        "status": "UNRESOLVED",
                        **(pair_witness or {"reason": "BVH_outer_enclosure_overlap"}),
                    }
            else:
                interval_gap = min(interval_gap, pair_gap)
        counters["intervals_evaluated"] += 1
        if interval_ok and not resource_limit_reached:
            counters["certified_regions"] += 1
            certified_interval_indices.append(interval_index)
            if interval_gap < minimum_gap:
                minimum_gap = interval_gap
                worst_certified = {
                    "interval_index": interval_index,
                    "pair_projection_gap_m": float(interval_gap),
                    "time_start_s": float(trajectory.times[interval_index]),
                    "time_end_s": float(trajectory.times[interval_index + 1]),
                    "lower_bound_semantics": "minimum positive projection gap over all separated BVH node pairs",
                }
        elif counters["collision_regions"] and interval_witness and interval_witness.get("status") == "COLLISION_FOUND":
            collision_interval_indices.append(interval_index)
        else:
            counters["unresolved_regions"] += 1
            unresolved_interval_indices.append(interval_index)
            if interval_witness is not None:
                witnesses.append(interval_witness)

    if counters["collision_regions"]:
        status = "COLLISION_FOUND"
    elif counters["unresolved_regions"] or resource_limit_reached or checked_interval_count < requested_interval_count:
        status = "UNRESOLVED"
    else:
        status = "PASS"
    certificate = {
        "schema_version": "d62-bvh-explicit-fk-qt-interval-certificate-v1",
        "trajectory_id": trajectory_id,
        "verification_domain": "articulated_self_collision_only",
        "verification_result": status,
        "status": status,
        "interval_count": int(requested_interval_count),
        "certificate_type": "CONSERVATIVE_FK_INTERVAL_BVH_NODE_SEPARATING_ENCLOSURE",
        "collision_method": "explicit_interval_FK_q(t)_with_recursive_mesh_BVH_AABB_enclosure",
        "sampling_semantics": "trajectory rows define endpoint states; no sampled-only PASS is emitted",
        "checked_time_interval": {
            "start_s": float(trajectory.times[0]),
            "end_s": float(trajectory.times[selected_interval_indices[-1] + 1]) if selected_interval_indices else float(trajectory.times[0]),
            "duration_s": float(trajectory.times[selected_interval_indices[-1] + 1] - trajectory.times[selected_interval_indices[0]]) if selected_interval_indices else 0.0,
            "interval_count": int(checked_interval_count),
            "requested_interval_count": int(requested_interval_count),
            "selected_interval_indices": selected_interval_indices,
            "coverage_complete": bool(len(selected_interval_indices) == requested_interval_count and not resource_limit_reached and selected_interval_indices == list(range(requested_interval_count))),
        },
        "certified_interval_indices": certified_interval_indices,
        "unresolved_interval_indices": unresolved_interval_indices,
        "collision_interval_indices": collision_interval_indices,
        "checked_interval_indices": selected_interval_indices,
        "collision_pairs_checked": ["|".join(pair) for pair in pairs],
        "collision_pair_count": len(pairs),
        # A separating-axis gap is not a Euclidean mesh clearance measurement.
        # Keep physical clearance explicitly unavailable and expose the
        # projection certificate under its own field.
        "minimum_clearance": None,
        "minimum_certified_clearance": None,
        "minimum_certified_projection_gap_m": None if not math.isfinite(minimum_gap) else float(minimum_gap),
        "worst_certified_region": worst_certified,
        "failure_witness": witnesses[0] if witnesses else None,
        "failure_witness_count": len(witnesses),
        "measurement": counters,
        "coverage": {
            "checked_interval_count": int(checked_interval_count),
            "requested_interval_count": int(requested_interval_count),
            "coverage_complete": bool(len(selected_interval_indices) == requested_interval_count and not resource_limit_reached and selected_interval_indices == list(range(requested_interval_count))),
            "resource_limit_reached": bool(resource_limit_reached),
            "max_intervals": max_intervals,
            "explicit_interval_selection": explicit_interval_selection,
            "selected_interval_count": int(len(selected_interval_indices)),
            "max_node_pairs": int(max_node_pairs),
            "max_time_depth": int(max_time_depth),
        },
        "conservativeness_bound": {
            "joint_enclosure": "D61 endpoint Taylor enclosure with configured absolute jerk bound",
            "jerk_bound_rad_s3": jerk_bound,
            "geometry_enclosure": "each mesh BVH node encloses its assigned STL vertices by a local AABB; all node pairs are covered",
            "fk_semantics": "D61 interval homogeneous-transform propagation of URDF origin and exact axis-angle joint transforms",
            "separation_rule": "PASS only when every required BVH node pair has a strictly positive fixed-axis projection gap",
            "unresolved_is_not_pass": True,
        },
        "input_contract": {
            "urdf": str(urdf_path.resolve()),
            "srdf": str(srdf_path.resolve()) if srdf_path else None,
            "trajectory": str(trajectory_path.resolve()),
            "urdf_sha256": _sha256(urdf_path),
            "srdf_sha256": _sha256(srdf_path) if srdf_path else None,
            "trajectory_sha256": _sha256(trajectory_path),
            "trajectory_fields": trajectory.field_contract,
            "trajectory_validation": trajectory_validation,
            "joint_order": list(trajectory.joint_names),
            "required_pair_count": len(pairs),
            "required_pairs": ["|".join(pair) for pair in pairs],
            "disabled_pair_count": len(disabled),
            "max_leaf_points": int(max_leaf_points),
            "interval_indices_explicit": explicit_interval_selection,
        },
        "backend_status": {
            "MoveIt2": "not_used_by_this_independent_shadow",
            "FCL": "not_used_by_this_independent_shadow",
            "Bullet": "not_used_by_this_independent_shadow",
        },
        "route_evidence": {
            "route_c_bvh": "IMPLEMENTED_AND_EXECUTED_SOUND_RECURSIVE_NODE_ENCLOSURE",
            "route_c_interval_fk": "INHERITED_D61_EXPLICIT_INTERVAL_FK",
            "route_b_explicit_polynomial": "UNAVAILABLE_UNLESS_NATIVE_COEFFICIENTS_ARE_EXPORTED",
            "route_d_external_backend": "NOT_USED_AS_LOCAL_AUTHORITY",
        },
        "execution": {
            "wall_time_s": float(time.perf_counter() - started_wall),
            "implementation": "Python reference shadow; no compiled backend claim",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # A certificate containing NaN/Infinity is not machine-verifiable and can
    # accidentally be interpreted as a passing numeric witness by a consumer.
    # Fail closed at the serialization boundary as well as at the arithmetic
    # boundary.
    output_path.write_text(json.dumps(certificate, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return certificate


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--srdf", type=Path, default=DEFAULT_SRDF)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectory-id", default="trajectory")
    parser.add_argument("--jerk-bound", type=float, default=8.0)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--max-leaf-points", type=int, default=64)
    parser.add_argument("--max-node-pairs", type=int, default=250000)
    parser.add_argument("--max-time-depth", type=int, default=8)
    parser.add_argument("--max-intervals", type=int, default=None)
    parser.add_argument("--interval-indices-file", type=Path, default=None, help="JSON array of interval indices to certify")
    args = parser.parse_args(argv)
    try:
        interval_indices = None
        if args.interval_indices_file is not None:
            interval_indices = json.loads(args.interval_indices_file.resolve().read_text(encoding="utf-8"))
            if not isinstance(interval_indices, list):
                raise CertificateInputError("interval-indices-file must contain a JSON array")
        certificate = certify(
            args.urdf.resolve(),
            args.srdf.resolve() if args.srdf else None,
            args.trajectory.resolve(),
            args.output.resolve(),
            args.trajectory_id,
            args.jerk_bound,
            args.threshold,
            args.max_leaf_points,
            args.max_node_pairs,
            args.max_time_depth,
            args.max_intervals,
            interval_indices,
        )
    except (CertificateInputError, OSError, TypeError, ValueError) as error:
        payload = {
            "schema_version": "d62-bvh-explicit-fk-qt-interval-certificate-v1",
            "status": "BLOCKED_INVALID_INPUT",
            "certificate_status": "BLOCKED_INVALID_INPUT",
            "verification_result": "BLOCKED_INVALID_INPUT",
            "error": str(error),
        }
        if isinstance(error, CertificateInputError):
            payload["error_code"] = getattr(error, "code", "INVALID_INPUT")
            payload["details"] = getattr(error, "details", None)
        else:
            payload["error_code"] = "INPUT_OR_FILE_ERROR"
            payload["details"] = None
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
    return 0 if certificate["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
