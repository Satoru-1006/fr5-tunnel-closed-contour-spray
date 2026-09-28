from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ros2_moveit_bridge"))

from p2b3_c1_scope import parse_initialization_only_flag, resolve_dls_trajectory_scope  # noqa: E402


def test_frozen_r0_default_retains_legacy_prefix_and_target_start() -> None:
    scope = resolve_dls_trajectory_scope(
        initialization_only=False,
        p2b2_variant="R0",
        solver_variant="B0",
        warm_start_rows=16,
        secondary_objective="none",
    )
    assert scope.emit_warm_start_prefix is True
    assert scope.initial_seed_row == 15
    assert scope.first_target_index == 16


def test_c1_r0_uses_only_first_observation_as_seed_and_solves_all_targets() -> None:
    scope = resolve_dls_trajectory_scope(
        initialization_only=True,
        p2b2_variant="R0",
        solver_variant="B0",
        warm_start_rows=16,
        secondary_objective="none",
    )
    assert scope.emit_warm_start_prefix is False
    assert scope.initial_seed_row == 0
    assert scope.first_target_index == 0


@pytest.mark.parametrize("value,expected", [("", False), ("false", False), ("0", False), ("true", True), ("1", True)])
def test_initialization_flag_accepts_explicit_boolean_values(value: str, expected: bool) -> None:
    assert parse_initialization_only_flag(value) is expected


def test_initialization_flag_rejects_ambiguous_values() -> None:
    with pytest.raises(ValueError, match="must_be_boolean"):
        parse_initialization_only_flag("yes")


@pytest.mark.parametrize(
    "field,value",
    [("p2b2_variant", "R1"), ("solver_variant", "B1"), ("warm_start_rows", 15), ("secondary_objective", "joint_centering")],
)
def test_c1_scope_rejects_policy_drift(field: str, value: object) -> None:
    arguments: dict[str, object] = {
        "initialization_only": True,
        "p2b2_variant": "R0",
        "solver_variant": "B0",
        "warm_start_rows": 16,
        "secondary_objective": "none",
    }
    arguments[field] = value
    with pytest.raises(ValueError, match="requires_frozen_R0_B0_policy"):
        resolve_dls_trajectory_scope(**arguments)  # type: ignore[arg-type]
