"""Synthetic tests for the isolated D34 direction-generator shadow module."""

from __future__ import annotations

import inspect

import pytest
import torch

from tools import stage3_h13_d34_direction_generators_shadow as d34


def flat(values: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([values[name].detach().reshape(-1).double() for name in values])


def norm(values: dict[str, torch.Tensor]) -> float:
    return float(torch.linalg.vector_norm(flat(values)))


def dot(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> float:
    return float(torch.dot(flat(left), flat(right)))


def maps(values: list[float]) -> dict[str, torch.Tensor]:
    split = len(values) - 1
    return {
        "block_a.weight": torch.tensor(values[:split], dtype=torch.float32),
        "block_b.bias": torch.tensor(values[split:], dtype=torch.float32),
    }


def test_shadow_contract_is_explicit_and_has_no_canonical_io_symbols():
    assert d34.SHADOW_ONLY
    assert d34.CANONICAL_AUTHORITY is False
    assert d34.CHECKPOINT_WRITES is False
    assert d34.OPTIMIZER_STEPS is False
    assert d34.RNG_CONSUMPTION is False
    source = inspect.getsource(d34)
    assert "torch.save(" not in source
    assert "optimizer.step(" not in source


def test_tangent_null_space_is_orthogonal_and_handles_duplicate_constraints():
    primary = maps([2.0, -1.0, 0.5])
    constraint = maps([1.0, 0.0, 0.0])
    duplicate = {name: value.clone() for name, value in constraint.items()}
    direction, metadata = d34.tangent_null_space(primary, [constraint, duplicate])
    assert norm(direction) == pytest.approx(1.0, abs=1.0e-6)
    assert dot(direction, constraint) == pytest.approx(0.0, abs=1.0e-6)
    assert metadata["constraint_matrix_rank"] == 1
    assert metadata["method"] == d34.METHOD_TANGENT_NULL_SPACE


def test_blockwise_sensitivity_attenuates_conflicting_block_and_repairs_constraint():
    primary = maps([1.0, 0.0, -2.0])
    constraint = maps([1.0, 0.0, 1.0])
    before = {name: value.clone() for name, value in primary.items()}
    direction, metadata = d34.blockwise_sensitivity(
        primary,
        constraint,
        {"block_a.weight": "aligned", "block_b.bias": "conflicting"},
        sensitivity_scale=2.0,
        target_norm=1.0,
    )
    assert norm(direction) == pytest.approx(1.0, abs=1.0e-6)
    assert dot(direction, constraint) >= -1.0e-8
    metrics = metadata["block_metrics"]
    assert metrics["conflicting"]["weight"] < metrics["aligned"]["weight"]
    assert all(torch.equal(primary[name], before[name]) for name in primary)


def test_pcgrad_removes_pairwise_conflict_in_deterministic_order():
    first = maps([1.0, 0.0, 0.0])
    second = maps([-1.0, 1.0, 0.0])
    direction, metadata = d34.pcgrad({"h1": first, "rest": second}, order=[0, 1])
    assert metadata["conflict_count"] >= 1
    assert metadata["post_projection_pair_dots"]["h1::rest"] >= -1.0e-8
    assert norm(direction) == pytest.approx(1.0, abs=1.0e-6)


def test_mgda_returns_simplex_weights_and_zero_solution_when_objectives_cancel():
    first = maps([1.0, 0.0, 0.0])
    second = maps([-1.0, 0.0, 0.0])
    direction, metadata = d34.mgda_min_norm([first, second], target_norm=None)
    assert metadata["weights"][0] == pytest.approx(0.5, abs=1.0e-6)
    assert metadata["weights"][1] == pytest.approx(0.5, abs=1.0e-6)
    assert metadata["simplex_sum"] == pytest.approx(1.0, abs=1.0e-10)
    assert norm(direction) == pytest.approx(0.0, abs=1.0e-8)


def test_constrained_lagrangian_selects_smallest_nonnegative_multiplier():
    primary = maps([-1.0, 1.0, 0.0])
    constraint = maps([1.0, 0.0, 0.0])
    direction, metadata = d34.constrained_lagrangian(
        primary, constraint, min_constraint_cosine=0.0, target_norm=1.0
    )
    assert metadata["lambda_multiplier"] == pytest.approx(1.0, abs=1.0e-6)
    assert metadata["final_constraint_cosine"] >= -1.0e-8
    assert dot(direction, constraint) >= -1.0e-8


def test_low_dimensional_subspace_stays_in_span_and_repairs_inside_span():
    primary = {"block_a.weight": torch.tensor([-1.0, 1.0, 0.0])}
    constraint = {"block_a.weight": torch.tensor([1.0, 0.0, 0.0])}
    basis = [
        {"block_a.weight": torch.tensor([1.0, 0.0, 0.0])},
        {"block_a.weight": torch.tensor([0.0, 1.0, 0.0])},
    ]
    direction, metadata = d34.low_dimensional_subspace(primary, constraint, basis)
    assert norm(direction) == pytest.approx(1.0, abs=1.0e-6)
    assert dot(direction, constraint) >= -1.0e-8
    assert abs(float(direction["block_a.weight"][2])) <= 1.0e-8
    assert metadata["basis_rank"] == 2
    assert metadata["subspace_residual_norm"] <= 1.0e-8


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        (d34.METHOD_TANGENT_NULL_SPACE, {"constraint": maps([1.0, 0.0, 0.0])}),
        (
            d34.METHOD_BLOCKWISE_SENSITIVITY,
            {
                "constraint": maps([1.0, 0.0, 0.0]),
                "blocks": {"block_a.weight": "a", "block_b.bias": "b"},
            },
        ),
        (d34.METHOD_PCGRAD, {"objectives": [maps([1.0, 0.0, 0.0]), maps([0.0, 1.0, 0.0])]}),
        (d34.METHOD_MGDA_MIN_NORM, {"objectives": [maps([1.0, 0.0, 0.0]), maps([0.0, 1.0, 0.0])]}),
        (d34.METHOD_CONSTRAINED_LAGRANGIAN, {"constraint": maps([1.0, 0.0, 0.0])}),
        (
            d34.METHOD_LOW_DIMENSIONAL_SUBSPACE,
            {
                "constraint": maps([1.0, 0.0, 0.0]),
                "basis": [maps([1.0, 0.0, 0.0]), maps([0.0, 1.0, 0.0])],
            },
        ),
    ],
)
def test_dispatcher_marks_every_family_shadow_only(method, kwargs):
    direction, metadata = d34.generate_shadow_direction(method, maps([1.0, 0.5, 0.0]), **kwargs)
    assert metadata["shadow_only"] is True
    assert metadata["canonical_authority"] is False
    assert metadata["checkpoint_write"] is False
    assert norm(direction) == pytest.approx(1.0, abs=1.0e-6)

