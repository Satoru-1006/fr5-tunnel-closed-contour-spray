"""Known-answer and fail-closed tests for the D61 shadow certificate."""

from __future__ import annotations

import csv
import math
import random
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest

from tools.d61_fk_interval_certificate import (
    CertificateInputError,
    Interval,
    axis_rotation_exact,
    axis_rotation_interval,
    bernstein_cubic_interval,
    certify,
    cos_interval,
    interval_add,
    interval_div,
    interval_mul,
    interval_sub,
    route_b_probe,
    Trajectory,
    sin_interval,
    validate_trajectory_contract,
)


URDF = """<?xml version="1.0"?>
<robot name="d61_known_answer">
  <link name="base">
    <collision><geometry><box size="0.2 0.2 0.2"/></geometry></collision>
  </link>
  <link name="tip">
    <collision><geometry><box size="0.2 0.2 0.2"/></geometry></collision>
  </link>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="tip"/>
    <origin xyz="0 0 2" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>
</robot>
"""

HIDDEN_MIDDLE_COLLISION_URDF = """<?xml version="1.0"?>
<robot name="d61_hidden_middle_collision">
  <link name="base">
    <collision><origin xyz="0.5 0 0"/><geometry><box size="0.1 0.1 0.1"/></geometry></collision>
  </link>
  <link name="arm">
    <collision><origin xyz="1 0 0"/><geometry><box size="2 0.1 0.1"/></geometry></collision>
  </link>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="arm"/>
    <origin xyz="0 0 0"/><axis xyz="0 0 1"/>
  </joint>
</robot>
"""


def write_case(tmp_path: Path, rows: list[dict[str, float]]) -> tuple[Path, Path]:
    urdf = tmp_path / "known_answer.urdf"
    urdf.write_text(URDF, encoding="utf-8")
    trajectory = tmp_path / "trajectory.csv"
    with trajectory.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("t", "j1_q", "j1_dq", "j1_ddq", "j1_jerk"))
        writer.writeheader()
        writer.writerows(rows)
    return urdf, trajectory


def test_trigonometric_interval_contains_dense_samples() -> None:
    interval = Interval(-0.8, 1.4)
    sine = sin_interval(interval)
    cosine = cos_interval(interval)
    for index in range(1001):
        angle = interval.lo + (interval.hi - interval.lo) * index / 1000.0
        assert sine.lo - 1.0e-12 <= math.sin(angle) <= sine.hi + 1.0e-12
        assert cosine.lo - 1.0e-12 <= math.cos(angle) <= cosine.hi + 1.0e-12


def test_interval_intersection_is_exact_at_touching_and_ulp_boundaries() -> None:
    touching = Interval(0.0, 1.0).intersect(Interval(1.0, 2.0))
    assert not touching.is_empty
    assert touching.lo == 1.0 and touching.hi == 1.0

    separated = Interval(0.0, 1.0)
    for _ in range(4):
        separated = Interval(separated.lo, math.nextafter(separated.hi, math.inf))
        result = Interval(0.0, 1.0).intersect(Interval(separated.hi, 2.0))
        assert result.is_empty
    assert Interval(-math.inf, 0.0).intersect(Interval(0.0, math.inf)) == Interval.point(0.0)
    assert Interval(-math.inf, math.inf).intersect(Interval(2.0, 3.0)) == Interval(2.0, 3.0)
    assert Interval.EMPTY.intersect(Interval.point(0.0)).is_empty
    with pytest.raises(CertificateInputError):
        Interval(math.nan, 0.0)


def test_directed_arithmetic_contains_high_precision_endpoint_references() -> None:
    rng = random.Random(651)
    with localcontext() as context:
        context.prec = 120
        for _ in range(100):
            a0, a1 = sorted((rng.uniform(-100.0, 100.0) for _ in range(2)))
            b0, b1 = sorted((rng.uniform(-100.0, 100.0) for _ in range(2)))
            a, b = Interval(a0, a1), Interval(b0, b1)
            add = interval_add(a, b)
            sub = interval_sub(a, b)
            mul = interval_mul(a, b)
            for result, expected_lo, expected_hi in (
                (add, Decimal.from_float(a0) + Decimal.from_float(b0), Decimal.from_float(a1) + Decimal.from_float(b1)),
                (sub, Decimal.from_float(a0) - Decimal.from_float(b1), Decimal.from_float(a1) - Decimal.from_float(b0)),
            ):
                assert Decimal.from_float(result.lo) <= expected_lo <= Decimal.from_float(result.hi)
                assert Decimal.from_float(result.lo) <= expected_hi <= Decimal.from_float(result.hi)
            products = [Decimal.from_float(x) * Decimal.from_float(y) for x in (a0, a1) for y in (b0, b1)]
            assert Decimal.from_float(mul.lo) <= min(products)
            assert max(products) <= Decimal.from_float(mul.hi)
            denominator = Interval(1.0, 3.0)
            quotient = interval_div(a, denominator)
            quotients = [Decimal.from_float(x) / Decimal.from_float(y) for x in (a0, a1) for y in (1.0, 3.0)]
            assert Decimal.from_float(quotient.lo) <= min(quotients)
            assert max(quotients) <= Decimal.from_float(quotient.hi)


def test_interval_fk_rotation_contains_randomized_exact_reference() -> None:
    rng = random.Random(652)
    for _ in range(80):
        axis = np.asarray([rng.uniform(-2.0, 2.0) for _ in range(3)], dtype=float)
        lo = rng.uniform(-8.0, 8.0)
        hi = lo + rng.uniform(0.0, 2.0)
        enclosure = axis_rotation_interval(axis, Interval(lo, hi))
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            angle = lo if fraction == 0.0 else hi if fraction == 1.0 else lo + (hi - lo) * fraction
            exact = axis_rotation_exact(axis, angle)
            for row in range(3):
                for column in range(3):
                    assert enclosure[row][column].lo <= exact[row, column] <= enclosure[row][column].hi


def test_six_dof_trajectory_contract_rejects_jerk_and_derivative_inconsistency() -> None:
    names = tuple(f"j{index}" for index in range(1, 7))
    valid = Trajectory(
        np.asarray([0.0, 1.0]),
        {name: np.asarray([0.0, 1.0]) for name in names},
        {name: np.asarray([1.0, 1.0]) for name in names},
        {name: np.asarray([0.0, 0.0]) for name in names},
        {name: np.asarray([0.0, 0.0]) for name in names},
        names,
        {"t": True, "q": True, "dq": True, "ddq": True, "jerk": True, "complete": True},
    )
    assert validate_trajectory_contract(valid, 2.0, require_six_dof=True)["status"] == "VALID"

    invalid = Trajectory(
        valid.times,
        {name: np.asarray([0.0, 100.0]) for name in names},
        valid.velocity,
        {name: np.asarray([0.0, 0.0]) for name in names},
        {name: np.asarray([0.0, 3.0]) for name in names},
        names,
        valid.field_contract,
    )
    result = validate_trajectory_contract(invalid, 2.0, require_six_dof=True)
    assert result["status"] == "BLOCKED_INVALID_INPUT"
    assert {error["code"] for error in result["errors"]} >= {"JERK_BOUND_VIOLATION", "STATE_CONSISTENCY_VIOLATION"}


def test_trajectory_contract_rejects_jointly_impossible_jerk_moments() -> None:
    # When Δa saturates |j| <= 1 over one second, jerk must be +1 almost
    # everywhere, forcing q1=+1/6.  Independent endpoint cones alone miss the
    # sign contradiction in q1=-1/6.
    trajectory = Trajectory(
        np.asarray([0.0, 1.0]),
        {"j1": np.asarray([0.0, -1.0 / 6.0])},
        {"j1": np.asarray([0.0, 0.5])},
        {"j1": np.asarray([0.0, 1.0])},
        {"j1": np.asarray([0.0, 0.0])},
        ("j1",),
        {"t": True, "q": True, "dq": True, "ddq": True, "jerk": True, "complete": True},
    )
    result = validate_trajectory_contract(trajectory, 1.0)
    assert result["status"] == "BLOCKED_INVALID_INPUT"
    assert any(error["code"] == "STATE_MOMENT_FEASIBILITY_VIOLATION" for error in result["errors"])


def test_trajectory_contract_enforces_bound_model_limits() -> None:
    trajectory = Trajectory(
        np.asarray([0.0, 1.0]),
        {"j1": np.asarray([0.0, 0.1])},
        {"j1": np.asarray([0.0, 2.0])},
        {"j1": np.asarray([0.0, 0.0])},
        {"j1": np.asarray([0.0, 0.0])},
        ("j1",),
        {"t": True, "q": True, "dq": True, "ddq": True, "jerk": True, "complete": True},
    )
    result = validate_trajectory_contract(trajectory, 8.0, joint_limits={"j1": {"lower": -1.0, "upper": 1.0, "velocity": 1.0}})
    assert result["status"] == "BLOCKED_INVALID_INPUT"
    assert any(error["code"] == "JOINT_DQ_LIMIT_VIOLATION" for error in result["errors"])


def test_explicit_fk_interval_certifies_known_safe_motion(tmp_path: Path) -> None:
    urdf, trajectory = write_case(
        tmp_path,
        [
            {"t": 0.0, "j1_q": 0.0, "j1_dq": 1.0, "j1_ddq": 0.0, "j1_jerk": 0.0},
            {"t": 1.0, "j1_q": 1.0, "j1_dq": 1.0, "j1_ddq": 0.0, "j1_jerk": 0.0},
        ],
    )
    output = tmp_path / "certificate.json"
    result = certify(urdf, None, trajectory, output, "known_safe", 8.0, 0.0, 4, 1000)
    assert result["verification_result"] == "PASS"
    assert result["certificate_type"] == "CONSERVATIVE_FK_INTERVAL_SWEPT_SEPARATING_AXIS_ENCLOSURE"
    assert result["minimum_clearance"] is not None
    assert result["failure_witness"] is None
    assert output.exists()


def test_high_curvature_fk_motion_remains_certifiable_when_geometry_is_separated(tmp_path: Path) -> None:
    urdf, trajectory = write_case(
        tmp_path,
        [
            {"t": 0.0, "j1_q": 0.0, "j1_dq": 0.0, "j1_ddq": 4.0, "j1_jerk": 0.0},
            {"t": 1.0, "j1_q": 2.0, "j1_dq": 4.0, "j1_ddq": 4.0, "j1_jerk": 0.0},
        ],
    )
    result = certify(urdf, None, trajectory, tmp_path / "high_curvature.json", "high_curvature_safe", 0.0, 0.0, 4, 1000)
    assert result["verification_result"] == "PASS"
    assert result["coverage"]["coverage_complete"] is True


def test_exact_box_endpoint_collision_is_not_hidden(tmp_path: Path) -> None:
    urdf, trajectory = write_case(
        tmp_path,
        [
            {"t": 0.0, "j1_q": 0.0, "j1_dq": 0.0, "j1_ddq": 0.0, "j1_jerk": 0.0},
            {"t": 1.0, "j1_q": 0.0, "j1_dq": 0.0, "j1_ddq": 0.0, "j1_jerk": 0.0},
        ],
    )
    text = urdf.read_text(encoding="utf-8").replace('xyz="0 0 2"', 'xyz="0 0 0"')
    urdf.write_text(text, encoding="utf-8")
    result = certify(urdf, None, trajectory, tmp_path / "collision.json", "known_collision", 8.0, 0.0, 4, 1000)
    assert result["verification_result"] == "COLLISION_FOUND"
    assert result["failure_witness"]["kind"] == "exact_box_obb_endpoint"


def test_endpoint_safe_middle_collision_is_not_promoted_to_pass(tmp_path: Path) -> None:
    urdf = tmp_path / "hidden_middle_collision.urdf"
    urdf.write_text(HIDDEN_MIDDLE_COLLISION_URDF, encoding="utf-8")
    trajectory = tmp_path / "trajectory.csv"
    with trajectory.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("t", "j1_q", "j1_dq", "j1_ddq", "j1_jerk"))
        writer.writeheader()
        writer.writerows(
            [
                {"t": 0.0, "j1_q": -math.pi / 2.0, "j1_dq": math.pi, "j1_ddq": 0.0, "j1_jerk": 0.0},
                {"t": 1.0, "j1_q": math.pi / 2.0, "j1_dq": math.pi, "j1_ddq": 0.0, "j1_jerk": 0.0},
            ]
        )
    result = certify(urdf, None, trajectory, tmp_path / "hidden_collision.json", "endpoint_safe_middle_collision", 0.0, 0.0, 8, 1000)
    assert result["verification_result"] == "UNRESOLVED"
    assert result["measurement"]["collision_regions"] == 0
    assert result["measurement"]["unresolved_regions"] > 0


def test_overlapping_mesh_enclosure_is_unresolved_not_pass(tmp_path: Path) -> None:
    urdf, trajectory = write_case(
        tmp_path,
        [
            {"t": 0.0, "j1_q": 0.0, "j1_dq": 0.0, "j1_ddq": 0.0, "j1_jerk": 0.0},
            {"t": 1.0, "j1_q": 0.0, "j1_dq": 0.0, "j1_ddq": 0.0, "j1_jerk": 0.0},
        ],
    )
    text = urdf.read_text(encoding="utf-8").replace('xyz="0 0 2"', 'xyz="0 0 0.15"')
    text = text.replace(
        '<link name="tip">\n    <collision><geometry><box size="0.2 0.2 0.2"/></geometry></collision>',
        '<link name="tip">\n    <collision><geometry><cylinder radius="0.1" length="0.2"/></geometry></collision>',
    )
    urdf.write_text(text, encoding="utf-8")
    result = certify(urdf, None, trajectory, tmp_path / "unresolved.json", "known_overlap", 8.0, 0.0, 2, 1000)
    assert result["verification_result"] == "UNRESOLVED"
    assert result["minimum_clearance"] is None
    assert result["failure_witness"]["status"] == "UNRESOLVED"


def test_route_b_requires_explicit_coefficients(tmp_path: Path) -> None:
    _, trajectory = write_case(
        tmp_path,
        [{"t": 0.0, "j1_q": 0.0, "j1_dq": 0.0, "j1_ddq": 0.0, "j1_jerk": 0.0}, {"t": 1.0, "j1_q": 0.0, "j1_dq": 0.0, "j1_ddq": 0.0, "j1_jerk": 0.0}],
    )
    assert route_b_probe(trajectory)["status"] == "UNAVAILABLE"
    assert bernstein_cubic_interval((0.0, 3.0, -3.0, 1.0)) == (0.0, 1.0)


def test_missing_native_derivatives_blocks_certificate(tmp_path: Path) -> None:
    urdf = tmp_path / "known_answer.urdf"
    urdf.write_text(URDF, encoding="utf-8")
    trajectory = tmp_path / "q_only.csv"
    trajectory.write_text("t,j1_q\n0,0\n1,1\n", encoding="utf-8")
    with pytest.raises(CertificateInputError, match="dq and ddq"):
        certify(urdf, None, trajectory, tmp_path / "blocked.json", "q_only", 8.0, 0.0, 2, 1000)
