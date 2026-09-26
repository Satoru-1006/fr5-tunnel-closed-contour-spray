"""D62 BVH enclosure known-answer and fail-closed tests."""

from __future__ import annotations

import csv
import json
import math
import random
from decimal import Decimal
from pathlib import Path

import numpy as np

from tools.d61_fk_interval_certificate import CertificateInputError, Interval, Shape, Trajectory, exact_fk, interval_fk, load_model, q_envelope_for_subinterval, read_trajectory
from tools.d62_bvh_interval_certificate import _cached_box_projections, _outward_reduce, _outward_sum, _validated_q_envelope, build_bvh, certify, main as d62_main


URDF = """<?xml version="1.0"?>
<robot name="d62_bvh_known_answer">
  <link name="base"><collision><geometry><mesh filename="base.stl"/></geometry></collision></link>
  <link name="tip"><collision><geometry><mesh filename="tip.stl"/></geometry></collision></link>
  <joint name="j1" type="revolute"><parent link="base"/><child link="tip"/><origin xyz="0 0 0"/><axis xyz="0 0 1"/></joint>
</robot>
"""


def write_stl(path: Path, points: list[tuple[float, float, float]]) -> None:
    lines = ["solid d62"]
    for x, y, z in points:
        lines.extend([" facet normal 0 0 1", "  outer loop"])
        for _ in range(3):
            lines.append(f"   vertex {x} {y} {z}")
        lines.extend(["  endloop", " endfacet"])
    lines.append("endsolid d62")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_case(tmp_path: Path, base_points: list[tuple[float, float, float]], tip_points: list[tuple[float, float, float]], q0: float = 0.0, q1: float = 0.0) -> tuple[Path, Path]:
    write_stl(tmp_path / "base.stl", base_points)
    write_stl(tmp_path / "tip.stl", tip_points)
    urdf = tmp_path / "model.urdf"
    urdf.write_text(URDF, encoding="utf-8")
    trajectory = tmp_path / "trajectory.csv"
    with trajectory.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("t", "j1_q", "j1_dq", "j1_ddq", "j1_jerk"))
        writer.writeheader()
        writer.writerows([
            {"t": 0.0, "j1_q": q0, "j1_dq": 0.0, "j1_ddq": 0.0, "j1_jerk": 0.0},
            {"t": 1.0, "j1_q": q1, "j1_dq": 0.0, "j1_ddq": 0.0, "j1_jerk": 0.0},
        ])
    return urdf, trajectory


def test_bvh_contains_all_points_and_subdivides() -> None:
    points = np.array([[float(index), 0.0, 0.0] for index in range(9)])
    root = build_bvh(points, max_leaf_points=1)
    assert root.point_count == 3
    assert not root.is_leaf
    assert root.left is not None and root.right is not None
    assert np.all(root.local_min <= np.array([0.0, 0.0, 0.0]))
    assert np.all(root.local_max >= np.array([8.0, 0.0, 0.0]))


def test_bvh_recovers_disconnected_mesh_separation(tmp_path: Path) -> None:
    # The one-box link envelopes overlap, but every disconnected cluster pair
    # is separated.  This is the exact resolution gain D62 is meant to test.
    base_points = [(-1.0, 0.0, 0.0), (-0.9, 0.0, 0.0), (0.9, 0.0, 0.0), (1.0, 0.0, 0.0)]
    tip_points = [(0.0, -1.0, 0.0), (0.0, -0.9, 0.0), (0.0, 0.9, 0.0), (0.0, 1.0, 0.0)]
    urdf, trajectory = write_case(tmp_path, base_points, tip_points)
    result = certify(urdf, None, trajectory, tmp_path / "certificate.json", "bvh_disconnected_safe", 0.0, 0.0, 1, 1000)
    assert result["verification_result"] == "PASS"
    assert result["coverage"]["coverage_complete"] is True
    assert result["measurement"]["node_pairs_separated"] > 0


def test_bvh_leaf_overlap_is_unresolved(tmp_path: Path) -> None:
    urdf, trajectory = write_case(tmp_path, [(0.0, 0.0, 0.0)] * 3, [(0.0, 0.0, 0.0)] * 3)
    result = certify(urdf, None, trajectory, tmp_path / "certificate.json", "bvh_overlap", 0.0, 0.0, 2, 1000)
    assert result["verification_result"] == "UNRESOLVED"
    assert result["measurement"]["leaf_overlaps"] > 0
    assert result["failure_witness"]["status"] == "UNRESOLVED"


def test_bvh_resource_limit_is_fail_closed(tmp_path: Path) -> None:
    urdf, trajectory = write_case(
        tmp_path,
        [(-1.0, 0.0, 0.0), (-0.9, 0.0, 0.0), (0.9, 0.0, 0.0), (1.0, 0.0, 0.0)],
        [(0.0, -1.0, 0.0), (0.0, -0.9, 0.0), (0.0, 0.9, 0.0), (0.0, 1.0, 0.0)],
    )
    result = certify(urdf, None, trajectory, tmp_path / "certificate.json", "bvh_budget", 0.0, 0.0, 1, 1)
    assert result["verification_result"] == "UNRESOLVED"
    assert result["coverage"]["resource_limit_reached"] is True
    assert result["coverage"]["coverage_complete"] is False


def test_d62_rejects_exported_jerk_above_certificate_bound(tmp_path: Path) -> None:
    urdf, trajectory = write_case(tmp_path, [(-1.0, 0.0, 0.0)] * 3, [(1.0, 0.0, 0.0)] * 3)
    rows = trajectory.read_text(encoding="utf-8").splitlines()
    rows[1] = rows[1].rsplit(",", 1)[0] + ",100.0"
    trajectory.write_text("\n".join(rows) + "\n", encoding="utf-8")
    try:
        certify(urdf, None, trajectory, tmp_path / "certificate.json", "jerk_violation", 8.0, 0.0, 2, 1000)
    except CertificateInputError as error:
        assert error.code == "JERK_BOUND_VIOLATION"
    else:
        raise AssertionError("D62 must fail closed when exported jerk exceeds the configured bound")


def test_d62_cli_malformed_csv_is_structured_fail_closed(tmp_path: Path) -> None:
    urdf, trajectory = write_case(tmp_path, [(-1.0, 0.0, 0.0)] * 3, [(1.0, 0.0, 0.0)] * 3)
    rows = trajectory.read_text(encoding="utf-8").splitlines()
    rows[2] = rows[2].replace(",0.0,", ",not-a-number,", 1)
    trajectory.write_text("\n".join(rows) + "\n", encoding="utf-8")
    output = tmp_path / "malformed_certificate.json"
    exit_code = d62_main(["--urdf", str(urdf), "--trajectory", str(trajectory), "--output", str(output), "--trajectory-id", "malformed"])
    assert exit_code == 2
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "BLOCKED_INVALID_INPUT"
    assert payload["error_code"] in {"NONNUMERIC_VALUE", "INPUT_OR_FILE_ERROR"}


def test_d62_projection_sum_is_directed_outward_at_ulp_boundary() -> None:
    values = np.array([5.149712858281723, 1.9369322773033935, 134261373966372.25], dtype=float)
    exact = sum((Decimal.from_float(float(value)) for value in values), Decimal(0))
    lower = Decimal.from_float(_outward_sum(values, -1.0))
    upper = Decimal.from_float(_outward_sum(values, 1.0))
    assert lower <= exact <= upper


def test_d62_projection_sum_encloses_long_cancelling_reductions() -> None:
    rng = np.random.default_rng(20260902)
    for _ in range(200):
        exponents = rng.integers(-900, 900, size=64)
        mantissas = rng.uniform(-1.0, 1.0, size=64)
        values = np.ldexp(mantissas, exponents).astype(float)
        exact = sum((Decimal.from_float(float(value)) for value in values), Decimal(0))
        lower = Decimal.from_float(_outward_sum(values, -1.0))
        upper = Decimal.from_float(_outward_sum(values, 1.0))
        assert lower <= exact <= upper


def test_vectorized_directed_reduce_matches_decimal_oracle() -> None:
    rng = np.random.default_rng(20260903)
    values = np.ldexp(rng.uniform(-1.0, 1.0, size=(17, 64, 5)), rng.integers(-900, 900, size=(17, 64, 5)))
    lower = _outward_reduce(values, axis=1, direction=-1.0)
    upper = _outward_reduce(values, axis=1, direction=1.0)
    for row in range(values.shape[0]):
        for column in range(values.shape[2]):
            exact = sum((Decimal.from_float(float(value)) for value in values[row, :, column]), Decimal(0))
            assert Decimal.from_float(float(lower[row, column])) <= exact
            assert exact <= Decimal.from_float(float(upper[row, column]))


def test_d62_projection_sum_contains_many_term_cancellation_oracle() -> None:
    # This deliberately defeats a naive left-to-right reduction: the large
    # terms cancel, while several small terms remain significant.
    values = np.asarray(
        [1.0e16, 1.0, -1.0e16, 3.0, -2.0, 2.0 ** -52, -2.0 ** -52, 7.0, -4.0, 0.5],
        dtype=float,
    )
    exact = sum((Decimal.from_float(float(value)) for value in values), Decimal(0))
    lower = Decimal.from_float(_outward_sum(values, -1.0))
    upper = Decimal.from_float(_outward_sum(values, 1.0))
    assert lower <= exact <= upper


def test_d62_projection_sum_randomized_decimal_oracle_many_terms() -> None:
    rng = random.Random(65065)
    for _ in range(250):
        values = np.asarray(
            [math.ldexp(rng.choice((-1.0, 1.0)) * rng.random(), rng.randint(-900, 900)) for _ in range(rng.randint(4, 32))],
            dtype=float,
        )
        exact = sum((Decimal.from_float(float(value)) for value in values), Decimal(0))
        lower = Decimal.from_float(_outward_sum(values, -1.0))
        upper = Decimal.from_float(_outward_sum(values, 1.0))
        assert lower <= exact <= upper


def test_d62_projection_sum_mixed_infinity_is_not_a_finite_separator() -> None:
    assert math.isnan(_outward_sum(np.asarray([math.inf, -math.inf, 1.0]), -1.0))
    assert _outward_sum(np.asarray([math.inf, 1.0]), -1.0) == math.inf


def test_vectorized_box_projection_contains_random_interval_samples() -> None:
    rng = np.random.default_rng(65066)
    transform = [[Interval.point(0.0) for _ in range(4)] for _ in range(4)]
    transform[3][3] = Interval.point(1.0)
    for row in range(3):
        for column in range(3):
            center = 1.0 if row == column else 0.0
            radius = 1.0e-8 * (1 + row + column)
            transform[row][column] = Interval(center - radius, center + radius)
        transform[row][3] = Interval(-0.2 - row * 0.1, 0.3 + row * 0.1)
    shape = Shape(
        link="box",
        local_min=np.asarray([-2.0, -0.5, -3.0]),
        local_max=np.asarray([1.5, 4.0, 0.25]),
        origin=np.eye(4),
        kind="box",
        half_extents=None,
        support_vertices=None,
        source="known-answer",
    )
    axes = tuple(rng.normal(size=3) for _ in range(12))
    lower, upper = _cached_box_projections(transform, shape, axes, {})
    normalized = np.asarray(axes) / np.linalg.norm(np.asarray(axes), axis=1)[:, None]
    corners = np.asarray(
        [[shape.local_max[i] if (mask >> i) & 1 else shape.local_min[i] for i in range(3)] for mask in range(8)]
    )
    for _ in range(200):
        matrix = np.asarray([[rng.uniform(transform[r][c].lo, transform[r][c].hi) for c in range(3)] for r in range(3)])
        translation = np.asarray([rng.uniform(transform[r][3].lo, transform[r][3].hi) for r in range(3)])
        world = corners @ matrix.T + translation
        projections = normalized @ world.T
        assert np.all(lower <= np.min(projections, axis=1))
        assert np.all(np.max(projections, axis=1) <= upper)


def test_d62_empty_endpoint_jerk_cone_intersection_blocks_input() -> None:
    trajectory = Trajectory(
        times=np.asarray([0.0, 1.0]),
        q={"j": np.asarray([0.0, 1.0 / 6.0])},
        velocity={"j": np.asarray([0.0, -0.5])},
        acceleration={"j": np.asarray([0.0, -1.0])},
        jerk={"j": np.asarray([0.0, 0.0])},
        joint_names=("j",),
        field_contract={"t": True, "q": True, "dq": True, "ddq": True, "jerk": True, "complete": True},
    )
    with np.testing.assert_raises(CertificateInputError) as caught:
        _validated_q_envelope(trajectory, 0, 0.5, 0.5, 1.0)
    assert caught.exception.code == "EMPTY_ENCLOSURE"
