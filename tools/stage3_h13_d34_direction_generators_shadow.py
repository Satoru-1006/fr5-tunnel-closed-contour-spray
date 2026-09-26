"""D34 shadow-only direction generators.

This module is deliberately detached from the D33 runner.  It contains pure
operations on named tensor maps so a future D34 screen can compare directions
without loading a checkpoint, consuming the training RNG, running an
optimizer step, or writing canonical evidence.  A caller still has to run the
real D34 candidate through the existing MoveIt/rollout/AdamW/replay gates
before any result could be considered scientific evidence.

The public functions return ``(direction, metadata)``.  ``direction`` is a
fresh map and input maps are never modified in place.  A direction is a
*presented gradient* (the optimizer applies its negative); therefore a
non-negative dot product with a constraint gradient means that the first
order update does not increase that constraint loss.

Implemented families:

* ``tangent_null_space``: orthogonal projection against one or more constraint
  gradients using a pseudoinverse, including rank-deficient constraints.
* ``blockwise_sensitivity``: attenuate parameter blocks according to their
  local primary/constraint conflict before an optional global constraint
  repair.
* ``pcgrad``: deterministic PCGrad over an explicitly ordered objective list.
* ``mgda_min_norm``: deterministic projected-simplex solver for the MGDA
  minimum-norm convex combination.
* ``constrained_lagrangian``: smallest non-negative scalar multiplier that
  reaches a requested constraint-cosine floor.
* ``low_dimensional_subspace``: SVD-orthonormalized subspace projection with
  the same constrained Lagrangian repair performed in coefficient space.

No function in this file has filesystem, checkpoint, optimizer, RNG, or
MoveIt side effects.  ``CANONICAL_AUTHORITY`` is intentionally false.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import torch


CANONICAL_AUTHORITY = False
CHECKPOINT_WRITES = False
OPTIMIZER_STEPS = False
RNG_CONSUMPTION = False
SHADOW_ONLY = True

METHOD_TANGENT_NULL_SPACE = "TANGENT_NULL_SPACE"
METHOD_BLOCKWISE_SENSITIVITY = "BLOCKWISE_SENSITIVITY"
METHOD_PCGRAD = "PCGRAD"
METHOD_MGDA_MIN_NORM = "MGDA_MIN_NORM"
METHOD_CONSTRAINED_LAGRANGIAN = "CONSTRAINED_LAGRANGIAN"
METHOD_LOW_DIMENSIONAL_SUBSPACE = "LOW_DIMENSIONAL_SUBSPACE"

_EPS = 1.0e-12

TensorMap = Mapping[str, torch.Tensor]


class D34InputError(ValueError):
    """Raised when a shadow direction cannot be defined from its inputs."""


@dataclass(frozen=True)
class ShadowDirection:
    """Typed convenience wrapper for callers that prefer an object result."""

    direction: dict[str, torch.Tensor]
    metadata: dict[str, Any]


def _names(*maps: TensorMap) -> list[str]:
    if not maps:
        raise D34InputError("AT_LEAST_ONE_TENSOR_MAP_REQUIRED")
    names = list(maps[0].keys())
    if not names:
        raise D34InputError("EMPTY_TENSOR_MAP")
    expected = set(names)
    for values in maps[1:]:
        if set(values.keys()) != expected:
            raise D34InputError("TENSOR_MAP_KEYS_MISMATCH")
        for name in names:
            if tuple(values[name].shape) != tuple(maps[0][name].shape):
                raise D34InputError(f"TENSOR_MAP_SHAPE_MISMATCH:{name}")
    for values in maps:
        for name in names:
            tensor = values[name]
            if not isinstance(tensor, torch.Tensor):
                raise D34InputError(f"NON_TENSOR_VALUE:{name}")
            # Complex maps have no unambiguous Euclidean presented-gradient
            # interpretation here; reject them rather than silently dropping
            # their imaginary component during float64 flattening.
            if not tensor.is_floating_point() or tensor.is_complex():
                raise D34InputError(f"NON_FLOAT_TENSOR:{name}")
            if not bool(torch.isfinite(tensor.detach()).all()):
                raise D34InputError(f"NONFINITE_TENSOR:{name}")
    return names


def _flatten(values: TensorMap, names: Sequence[str]) -> tuple[torch.Tensor, list[tuple[int, ...]]]:
    pieces: list[torch.Tensor] = []
    shapes: list[tuple[int, ...]] = []
    for name in names:
        tensor = values[name].detach()
        pieces.append(tensor.reshape(-1).to(dtype=torch.float64, device="cpu"))
        shapes.append(tuple(tensor.shape))
    return torch.cat(pieces), shapes


def _unflatten(flat: torch.Tensor, template: TensorMap, names: Sequence[str]) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    offset = 0
    for name in names:
        source = template[name].detach()
        count = source.numel()
        result[name] = flat[offset : offset + count].reshape(source.shape).to(
            dtype=source.dtype, device=source.device
        )
        offset += count
    if offset != flat.numel():
        raise D34InputError("FLAT_TENSOR_LENGTH_MISMATCH")
    return result


def _as_constraint_list(constraints: TensorMap | Sequence[TensorMap]) -> list[TensorMap]:
    # A map is the singular-constraint shorthand; a sequence carries several
    # rows for J in the null-space projector.
    if isinstance(constraints, Mapping):
        return [constraints]
    result = list(constraints)
    if not result:
        raise D34InputError("AT_LEAST_ONE_CONSTRAINT_REQUIRED")
    return result


def _normalize_flat(flat: torch.Tensor, target_norm: float | None, *, allow_zero: bool = False) -> torch.Tensor:
    if target_norm is None:
        return flat
    target = float(target_norm)
    if not math.isfinite(target) or target < 0.0:
        raise D34InputError("INVALID_TARGET_NORM")
    current = float(torch.linalg.vector_norm(flat))
    if current <= _EPS:
        if allow_zero or target == 0.0:
            return torch.zeros_like(flat)
        raise D34InputError("DIRECTION_COLLAPSED_TO_ZERO")
    return flat * (target / current)


def _cosine_flat(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left_norm = float(torch.linalg.vector_norm(left))
    right_norm = float(torch.linalg.vector_norm(right))
    if left_norm <= _EPS or right_norm <= _EPS:
        return None
    return float(torch.dot(left, right) / (left_norm * right_norm))


def _metadata(method: str, **extra: Any) -> dict[str, Any]:
    return {
        "method": method,
        "shadow_only": True,
        "canonical_authority": False,
        "checkpoint_write": False,
        "optimizer_step": False,
        "rng_consumed": False,
        **extra,
    }


def _constraint_cosine_repair(
    direction: torch.Tensor,
    constraint: torch.Tensor,
    min_constraint_cosine: float,
    *,
    max_lambda: float | None = None,
) -> tuple[torch.Tensor, float, bool]:
    """Add the smallest ``lambda * constraint`` satisfying a cosine floor."""
    floor = float(min_constraint_cosine)
    if not math.isfinite(floor) or floor < -1.0 or floor >= 1.0:
        raise D34InputError("CONSTRAINT_COSINE_FLOOR_MUST_BE_IN[-1,1)")
    c_norm = float(torch.linalg.vector_norm(constraint))
    if c_norm <= _EPS:
        if floor <= 0.0:
            return direction, 0.0, True
        return direction, 0.0, False
    initial_cos = _cosine_flat(direction, constraint)
    if initial_cos is not None and initial_cos >= floor - 1.0e-12:
        return direction, 0.0, True

    def candidate(lam: float) -> torch.Tensor:
        return direction + float(lam) * constraint

    # The cosine tends to one as lambda grows.  Expand a finite bracket first,
    # then use bisection for deterministic, reproducible multiplier selection.
    low = 0.0
    high = 1.0
    for _ in range(80):
        high_direction = candidate(high)
        high_cos = _cosine_flat(high_direction, constraint)
        if high_cos is not None and high_cos >= floor:
            break
        high *= 2.0
    else:
        return direction, high, False
    for _ in range(80):
        midpoint = 0.5 * (low + high)
        midpoint_cos = _cosine_flat(candidate(midpoint), constraint)
        if midpoint_cos is not None and midpoint_cos >= floor:
            high = midpoint
        else:
            low = midpoint
    if max_lambda is not None:
        cap = float(max_lambda)
        if not math.isfinite(cap) or cap < 0.0:
            raise D34InputError("INVALID_MAX_LAMBDA")
        if high > cap:
            capped = candidate(cap)
            capped_cos = _cosine_flat(capped, constraint)
            return capped, cap, capped_cos is not None and capped_cos >= floor - 1.0e-12
    repaired = candidate(high)
    repaired_cos = _cosine_flat(repaired, constraint)
    return repaired, high, repaired_cos is not None and repaired_cos >= floor - 1.0e-12


def normalize_direction(values: TensorMap, target_norm: float = 1.0) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Return a detached map with a requested Euclidean norm."""
    names = _names(values)
    flat, _ = _flatten(values, names)
    normalized = _normalize_flat(flat, target_norm)
    return _unflatten(normalized, values, names), _metadata(
        "NORMALIZE", raw_norm=float(torch.linalg.vector_norm(flat)), target_norm=float(target_norm)
    )


def tangent_null_space(
    primary: TensorMap,
    constraints: TensorMap | Sequence[TensorMap],
    *,
    target_norm: float | None = 1.0,
    damping: float = 0.0,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Project ``primary`` into the null space of one or more constraints.

    For rows ``c_i`` of ``C`` the raw direction is
    ``d = p - C^T (C C^T + damping I)^+ C p``.  The pseudoinverse handles
    duplicate or rank-deficient constraints without a random basis choice.
    """
    constraint_list = _as_constraint_list(constraints)
    names = _names(primary, *constraint_list)
    primary_flat, _ = _flatten(primary, names)
    rows = torch.stack([_flatten(item, names)[0] for item in constraint_list], dim=0)
    if damping < 0.0 or not math.isfinite(float(damping)):
        raise D34InputError("INVALID_NULL_SPACE_DAMPING")
    gram = rows @ rows.T
    if damping:
        gram = gram + float(damping) * torch.eye(gram.shape[0], dtype=gram.dtype)
    coefficients = torch.linalg.pinv(gram) @ (rows @ primary_flat)
    raw = primary_flat - rows.T @ coefficients
    projected_dots = [float(torch.dot(raw, row)) for row in rows]
    direction = _normalize_flat(raw, target_norm)
    return _unflatten(direction, primary, names), _metadata(
        METHOD_TANGENT_NULL_SPACE,
        constraint_count=len(constraint_list),
        damping=float(damping),
        raw_norm=float(torch.linalg.vector_norm(raw)),
        projected_constraint_dots=projected_dots,
        constraint_matrix_rank=int(torch.linalg.matrix_rank(rows).item()),
    )


def blockwise_sensitivity(
    primary: TensorMap,
    constraint: TensorMap,
    blocks: Mapping[str, str],
    *,
    sensitivity_scale: float = 1.0,
    weight_power: float = 1.0,
    min_constraint_cosine: float = 0.0,
    target_norm: float | None = 1.0,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Generate a blockwise conflict-sensitive direction.

    Let ``p_b`` and ``c_b`` be primary and constraint vectors in block ``b``.
    The local outward sensitivity is
    ``s_b = max(0, -p_b^T c_b)/(||p_b|| ||c_b|| + eps)`` and the block is
    weighted by ``w_b = (1 + sensitivity_scale*s_b)^(-weight_power)``.
    A single constrained repair is then applied if the weighted direction is
    below ``min_constraint_cosine``.  Block scores and weights are persisted
    in metadata so a later real rollout can audit which parameter blocks drove
    the proposal.
    """
    names = _names(primary, constraint)
    if not blocks or any(name not in blocks for name in names):
        raise D34InputError("BLOCK_ASSIGNMENT_MUST_COVER_ALL_PARAMETERS")
    scale = float(sensitivity_scale)
    power = float(weight_power)
    if not math.isfinite(scale) or scale < 0.0 or not math.isfinite(power) or power < 0.0:
        raise D34InputError("INVALID_BLOCKWISE_SENSITIVITY_HYPERPARAMETER")
    p, _ = _flatten(primary, names)
    c, _ = _flatten(constraint, names)
    block_indices: dict[str, list[int]] = {}
    offset = 0
    for name in names:
        count = primary[name].numel()
        block_indices.setdefault(str(blocks[name]), []).extend(range(offset, offset + count))
        offset += count
    weighted = torch.zeros_like(p)
    block_meta: dict[str, dict[str, float]] = {}
    for block, indices in block_indices.items():
        index = torch.tensor(indices, dtype=torch.long)
        p_block, c_block = p[index], c[index]
        p_norm = float(torch.linalg.vector_norm(p_block))
        c_norm = float(torch.linalg.vector_norm(c_block))
        raw_dot = float(torch.dot(p_block, c_block))
        sensitivity = max(0.0, -raw_dot) / (p_norm * c_norm + _EPS)
        weight = (1.0 + scale * sensitivity) ** (-power)
        weighted[index] = p_block * weight
        block_meta[block] = {
            "primary_norm": p_norm,
            "constraint_norm": c_norm,
            "primary_constraint_dot": raw_dot,
            "outward_sensitivity": sensitivity,
            "weight": weight,
        }
    repaired, lam, constraint_pass = _constraint_cosine_repair(
        weighted, c, min_constraint_cosine
    )
    direction = _normalize_flat(repaired, target_norm)
    return _unflatten(direction, primary, names), _metadata(
        METHOD_BLOCKWISE_SENSITIVITY,
        sensitivity_scale=scale,
        weight_power=power,
        min_constraint_cosine=float(min_constraint_cosine),
        block_metrics=block_meta,
        repair_lambda=lam,
        constraint_cosine=_cosine_flat(direction, c),
        constraint_pass=bool(constraint_pass),
    )


def _objective_list(
    objectives: Sequence[TensorMap] | Mapping[str, TensorMap],
) -> tuple[list[str], list[TensorMap]]:
    if isinstance(objectives, Mapping):
        labels = [str(label) for label in objectives.keys()]
        maps = list(objectives.values())
    else:
        maps = list(objectives)
        labels = [f"objective_{index}" for index in range(len(maps))]
    if not maps:
        raise D34InputError("AT_LEAST_ONE_OBJECTIVE_REQUIRED")
    return labels, maps


def pcgrad(
    objectives: Sequence[TensorMap] | Mapping[str, TensorMap],
    *,
    order: Sequence[int] | None = None,
    target_norm: float | None = 1.0,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Apply deterministic PCGrad and average the projected objectives.

    For each ordered pair ``(i, j)``, if ``g_i^T g_j < 0`` then
    ``g_i <- g_i - (g_i^T g_j)/(||g_j||^2+eps) g_j``.  The explicit order is
    required for reproducibility; absent an order, input order is used and no
    RNG is consumed.
    """
    labels, maps = _objective_list(objectives)
    names = _names(*maps)
    flats = torch.stack([_flatten(item, names)[0] for item in maps], dim=0)
    count = len(maps)
    permutation = list(range(count)) if order is None else [int(item) for item in order]
    if sorted(permutation) != list(range(count)):
        raise D34InputError("PCGRAD_ORDER_MUST_BE_A_PERMUTATION")
    projected = flats.clone()
    conflicts: list[dict[str, Any]] = []
    for i in permutation:
        for j in permutation:
            if i == j:
                continue
            dot_ij = float(torch.dot(projected[i], flats[j]))
            if dot_ij < 0.0:
                denominator = float(torch.dot(flats[j], flats[j])) + _EPS
                coefficient = dot_ij / denominator
                projected[i] = projected[i] - coefficient * flats[j]
                conflicts.append(
                    {"i": labels[i], "j": labels[j], "dot_before": dot_ij, "projection_coefficient": coefficient}
                )
    aggregate = projected.mean(dim=0)
    direction = _normalize_flat(aggregate, target_norm, allow_zero=target_norm in (None, 0.0))
    post_pair_dots = {
        f"{labels[i]}::{labels[j]}": float(torch.dot(projected[i], projected[j]))
        for i in range(count)
        for j in range(i + 1, count)
    }
    return _unflatten(direction, maps[0], names), _metadata(
        METHOD_PCGRAD,
        objective_labels=labels,
        order=permutation,
        conflict_count=len(conflicts),
        conflicts=conflicts,
        post_projection_pair_dots=post_pair_dots,
        raw_aggregate_norm=float(torch.linalg.vector_norm(aggregate)),
    )


def _project_simplex(values: torch.Tensor) -> torch.Tensor:
    """Euclidean projection onto ``{w >= 0, sum(w) = 1}``."""
    if values.ndim != 1 or values.numel() == 0:
        raise D34InputError("SIMPLEX_VECTOR_MUST_BE_NONEMPTY_1D")
    sorted_values, _ = torch.sort(values, descending=True)
    cumulative = torch.cumsum(sorted_values, dim=0)
    indices = torch.arange(1, values.numel() + 1, dtype=values.dtype)
    feasible = sorted_values - (cumulative - 1.0) / indices > 0.0
    if not bool(feasible.any()):
        rho = values.numel() - 1
    else:
        rho = int(torch.nonzero(feasible, as_tuple=False)[-1].item())
    theta = (cumulative[rho] - 1.0) / float(rho + 1)
    return torch.clamp(values - theta, min=0.0)


def mgda_min_norm(
    objectives: Sequence[TensorMap] | Mapping[str, TensorMap],
    *,
    target_norm: float | None = 1.0,
    max_iter: int = 512,
    tolerance: float = 1.0e-11,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Compute a deterministic MGDA minimum-norm convex combination.

    With rows ``g_i`` and Gram matrix ``K_ij = g_i^T g_j``, solve
    ``min_{w in simplex} 1/2 w^T K w`` by projected gradient descent.  The
    direction is ``sum_i w_i g_i``.  ``target_norm=None`` is useful when the
    true MGDA solution is the zero vector due to exact objective cancellation.
    """
    labels, maps = _objective_list(objectives)
    names = _names(*maps)
    gradients = torch.stack([_flatten(item, names)[0] for item in maps], dim=0)
    count = gradients.shape[0]
    if max_iter <= 0 or tolerance <= 0.0:
        raise D34InputError("INVALID_MGDA_SOLVER_SETTINGS")
    gram = gradients @ gradients.T
    eigenvalues = torch.linalg.eigvalsh(gram)
    lipschitz = max(float(eigenvalues[-1]), _EPS)
    step = 1.0 / lipschitz
    weights = torch.full((count,), 1.0 / count, dtype=torch.float64)
    converged = False
    iterations = 0
    for iterations in range(1, int(max_iter) + 1):
        proposal = _project_simplex(weights - step * (gram @ weights))
        if float(torch.max(torch.abs(proposal - weights))) <= float(tolerance):
            weights = proposal
            converged = True
            break
        weights = proposal
    aggregate = weights @ gradients
    direction = _normalize_flat(aggregate, target_norm, allow_zero=target_norm in (None, 0.0))
    return _unflatten(direction, maps[0], names), _metadata(
        METHOD_MGDA_MIN_NORM,
        objective_labels=labels,
        weights=[float(value) for value in weights],
        simplex_sum=float(weights.sum()),
        min_norm_objective=0.5 * float(weights @ gram @ weights),
        raw_aggregate_norm=float(torch.linalg.vector_norm(aggregate)),
        solver_iterations=iterations,
        solver_converged=converged,
    )


def constrained_lagrangian(
    primary: TensorMap,
    constraint: TensorMap,
    *,
    min_constraint_cosine: float = 0.0,
    target_norm: float | None = 1.0,
    max_lambda: float | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Add a non-negative constraint multiplier to a primary gradient.

    The returned presented gradient is ``d = normalize(p + lambda*c)`` where
    ``lambda`` is the smallest non-negative multiplier whose cosine with ``c``
    reaches ``min_constraint_cosine`` (subject to an optional cap).
    """
    names = _names(primary, constraint)
    p, _ = _flatten(primary, names)
    c, _ = _flatten(constraint, names)
    repaired, multiplier, constraint_pass = _constraint_cosine_repair(
        p, c, min_constraint_cosine, max_lambda=max_lambda
    )
    direction = _normalize_flat(repaired, target_norm)
    return _unflatten(direction, primary, names), _metadata(
        METHOD_CONSTRAINED_LAGRANGIAN,
        min_constraint_cosine=float(min_constraint_cosine),
        lambda_multiplier=float(multiplier),
        raw_primary_constraint_cosine=_cosine_flat(p, c),
        final_constraint_cosine=_cosine_flat(direction, c),
        constraint_pass=bool(constraint_pass),
    )


def low_dimensional_subspace(
    primary: TensorMap,
    constraint: TensorMap,
    basis: Sequence[TensorMap],
    *,
    min_constraint_cosine: float = 0.0,
    target_norm: float | None = 1.0,
    rank_tolerance: float = 1.0e-10,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Generate a constrained direction in a low-dimensional vector subspace.

    The columns of ``B`` are the supplied basis maps.  SVD gives an
    orthonormal basis ``Q`` for their span, then ``p_s = QQ^T p`` and
    ``c_s = QQ^T c`` are formed.  Lagrangian repair occurs only in that span,
    so the returned vector cannot acquire a component outside the proposed
    low-dimensional subspace.
    """
    if not basis:
        raise D34InputError("AT_LEAST_ONE_SUBSPACE_BASIS_VECTOR_REQUIRED")
    names = _names(primary, constraint, *basis)
    p, _ = _flatten(primary, names)
    c, _ = _flatten(constraint, names)
    columns = torch.stack([_flatten(item, names)[0] for item in basis], dim=1)
    if rank_tolerance <= 0.0 or not math.isfinite(float(rank_tolerance)):
        raise D34InputError("INVALID_SUBSPACE_RANK_TOLERANCE")
    _, singular_values, vh = torch.linalg.svd(columns, full_matrices=False)
    if singular_values.numel() == 0:
        raise D34InputError("EMPTY_SUBSPACE_BASIS")
    threshold = float(rank_tolerance) * max(float(singular_values[0]), 1.0)
    rank = int((singular_values > threshold).sum().item())
    if rank == 0:
        raise D34InputError("SUBSPACE_BASIS_IS_NUMERICALLY_ZERO")
    # ``U`` is reconstructed from B*V*S^-1 so it remains available without a
    # second decomposition and is numerically orthonormal at float64.
    u, _, _ = torch.linalg.svd(columns, full_matrices=False)
    q = u[:, :rank]
    p_sub = q @ (q.T @ p)
    c_sub = q @ (q.T @ c)
    repaired, multiplier, constraint_pass = _constraint_cosine_repair(
        p_sub, c_sub, min_constraint_cosine
    )
    direction = _normalize_flat(repaired, target_norm)
    residual = direction - q @ (q.T @ direction)
    return _unflatten(direction, primary, names), _metadata(
        METHOD_LOW_DIMENSIONAL_SUBSPACE,
        supplied_basis_count=len(basis),
        basis_rank=rank,
        singular_values=[float(value) for value in singular_values],
        rank_tolerance=float(rank_tolerance),
        projected_primary_norm=float(torch.linalg.vector_norm(p_sub)),
        projected_constraint_norm=float(torch.linalg.vector_norm(c_sub)),
        repair_lambda=float(multiplier),
        constraint_cosine=_cosine_flat(direction, c),
        constraint_pass=bool(constraint_pass),
        subspace_residual_norm=float(torch.linalg.vector_norm(residual)),
    )


def generate_shadow_direction(
    method: str,
    primary: TensorMap,
    constraint: TensorMap | None = None,
    *,
    objectives: Sequence[TensorMap] | Mapping[str, TensorMap] | None = None,
    blocks: Mapping[str, str] | None = None,
    basis: Sequence[TensorMap] | None = None,
    **options: Any,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Dispatch one D34 shadow family without touching D33 state.

    This adapter is intentionally small: it makes future candidate specs easy
    to enumerate while keeping each mathematical generator independently
    testable.  It does not apply AdamW, evaluate H1/H32, or decide promotion.
    """
    family = str(method).upper()
    if family == METHOD_TANGENT_NULL_SPACE:
        if constraint is None:
            raise D34InputError("CONSTRAINT_REQUIRED_FOR_TANGENT_NULL_SPACE")
        return tangent_null_space(primary, constraint, **options)
    if family == METHOD_BLOCKWISE_SENSITIVITY:
        if constraint is None or blocks is None:
            raise D34InputError("CONSTRAINT_AND_BLOCKS_REQUIRED_FOR_BLOCKWISE_SENSITIVITY")
        return blockwise_sensitivity(primary, constraint, blocks, **options)
    if family == METHOD_PCGRAD:
        return pcgrad(objectives if objectives is not None else [primary], **options)
    if family == METHOD_MGDA_MIN_NORM:
        return mgda_min_norm(objectives if objectives is not None else [primary], **options)
    if family == METHOD_CONSTRAINED_LAGRANGIAN:
        if constraint is None:
            raise D34InputError("CONSTRAINT_REQUIRED_FOR_CONSTRAINED_LAGRANGIAN")
        return constrained_lagrangian(primary, constraint, **options)
    if family == METHOD_LOW_DIMENSIONAL_SUBSPACE:
        if constraint is None or basis is None:
            raise D34InputError("CONSTRAINT_AND_BASIS_REQUIRED_FOR_LOW_DIMENSIONAL_SUBSPACE")
        return low_dimensional_subspace(primary, constraint, basis, **options)
    raise D34InputError(f"UNKNOWN_D34_SHADOW_METHOD:{method}")
