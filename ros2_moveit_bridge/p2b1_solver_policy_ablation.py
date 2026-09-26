"""Mathematics and fail-closed helpers for the P2-B1 solver ablation.

The direction residual is expressed in a fixed tangent basis of the desired
normal.  Its exact off-manifold differential is used during iteration; at the
aligned process state it reduces to the usual projected angular Jacobian.
"""
from __future__ import annotations

from dataclasses import dataclass
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


PRESSURE_FIELDS = (
    "case_id", "variant", "waypoint", "iteration", "current_q", "raw_dls_delta",
    "primary_task_contribution", "secondary_objective_contribution", "trust_region_delta",
    "proposed_q_before_projection", "projected_q", "accepted_q", "accepted_alpha",
    "projection_triggered", "projected_joints", "j6_pre_projection", "j6_post_projection",
    "j6_upper_slack_pre", "j6_lower_slack_pre", "j6_upper_slack_post", "j6_lower_slack_post",
    "max_j6_upper_bound_excess", "max_j6_buffer_boundary_excess", "process_residual_norm",
    "weighted_process_residual_norm", "position_residual_m", "normal_residual",
    "primary_task_contribution_norm", "secondary_objective_contribution_norm",
    "trust_region_delta_norm", "process_jacobian_rank", "singular_values", "nullspace_dimension",
    "j6_nullspace_projection", "secondary_objective_before", "secondary_objective_after",
    "line_search_accepted",
)


def append_pressure_record(path: str | Path, row: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    record = {key: row.get(key) for key in PRESSURE_FIELDS}
    for key in ("current_q", "raw_dls_delta", "primary_task_contribution", "secondary_objective_contribution", "trust_region_delta", "proposed_q_before_projection", "projected_q", "accepted_q", "projected_joints", "singular_values"):
        value = record.get(key)
        if isinstance(value, (list, tuple, np.ndarray)):
            record[key] = json.dumps(np.asarray(value).tolist() if isinstance(value, np.ndarray) else list(value), separators=(",", ":"), allow_nan=False)
    new_file = not target.exists() or target.stat().st_size == 0
    with target.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=PRESSURE_FIELDS, lineterminator="\n")
        if new_file:
            writer.writeheader()
        writer.writerow(record)


@dataclass(frozen=True)
class NullspaceMetrics:
    projector: np.ndarray
    rank: int
    nullity: int
    singular_values: np.ndarray
    rank_tolerance: float


def _finite_array(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{label}_shape_or_finiteness")
    return array


def skew(vector: Any) -> np.ndarray:
    x, y, z = _finite_array(vector, (3,), "vector")
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def tangent_bases(normal: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return desired normal n and two orthonormal bases B,U tangent to n.

    B coordinates the normal-direction residual.  U = [n]x B is the angular
    task basis, making the exact residual differential equal U.T J_omega when
    the tool axis is aligned with n.
    """
    n = _finite_array(normal, (3,), "normal")
    norm = float(np.linalg.norm(n))
    if norm <= 1e-12:
        raise ValueError("zero_or_degenerate_normal")
    n = n / norm
    reference = np.eye(3)[int(np.argmin(np.abs(n)))]
    first = reference - n * float(np.dot(reference, n))
    first_norm = float(np.linalg.norm(first))
    if first_norm <= 1e-12:
        raise ValueError("tangent_basis_degenerate")
    first /= first_norm
    second = np.cross(n, first)
    B = np.column_stack((first, second))
    U = skew(n) @ B
    return n, B, U


def process_residual(
    position: Any, target_position: Any, tool_z: Any, desired_normal: Any, residual_basis: Any
) -> np.ndarray:
    p = _finite_array(position, (3,), "position")
    target = _finite_array(target_position, (3,), "target_position")
    z = _finite_array(tool_z, (3,), "tool_z")
    n, _, _ = tangent_bases(desired_normal)
    B = _finite_array(residual_basis, (3, 2), "residual_basis")
    znorm = float(np.linalg.norm(z))
    if znorm <= 1e-12:
        raise ValueError("zero_or_degenerate_tool_axis")
    z = z / znorm
    return np.r_[p - target, B.T @ (z - n)]


def process_jacobian(
    linear_jacobian: Any,
    angular_jacobian: Any,
    tool_z: Any,
    desired_normal: Any,
    residual_basis: Any,
) -> np.ndarray:
    """Exact 5x6 derivative of process_residual for spatial angular Jacobians.

    For spatial angular velocity, dz/dq = -[z]x J_omega.  Thus the lower
    block is -B.T [z]x J_omega.  At z=n this becomes U.T J_omega for the
    desired-normal tangent basis U=[n]xB.
    """
    Jv = _finite_array(linear_jacobian, (3, 6), "linear_jacobian")
    Jw = _finite_array(angular_jacobian, (3, 6), "angular_jacobian")
    z = _finite_array(tool_z, (3,), "tool_z")
    znorm = float(np.linalg.norm(z))
    if znorm <= 1e-12:
        raise ValueError("zero_or_degenerate_tool_axis")
    z = z / znorm
    tangent_bases(desired_normal)
    B = _finite_array(residual_basis, (3, 2), "residual_basis")
    return np.vstack((Jv, -B.T @ skew(z) @ Jw))


def weighted_task(
    residual: Any, jacobian: Any, position_weight: float, normal_weight: float
) -> tuple[np.ndarray, np.ndarray]:
    error = _finite_array(residual, (5,), "process_residual")
    matrix = _finite_array(jacobian, (5, 6), "process_jacobian")
    if not all(math.isfinite(v) and v > 0.0 for v in (position_weight, normal_weight)):
        raise ValueError("process_weights_must_be_finite_and_positive")
    weights = np.asarray([position_weight] * 3 + [normal_weight] * 2, dtype=np.float64)
    return weights * error, weights[:, None] * matrix


def nullspace_metrics(jacobian: Any) -> NullspaceMetrics:
    matrix = np.asarray(jacobian, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != 6 or matrix.shape[0] < 1 or not np.isfinite(matrix).all():
        raise ValueError("process_jacobian_shape_or_finiteness")
    _u, singular_values, vh = np.linalg.svd(matrix, full_matrices=True)
    sigma_max = float(singular_values[0]) if singular_values.size else 0.0
    tolerance = sigma_max * max(matrix.shape) * np.finfo(np.float64).eps * 100.0
    rank = int(np.sum(singular_values > tolerance)) if sigma_max > 0.0 else 0
    null_rows = vh[rank:, :]
    projector = null_rows.T @ null_rows if null_rows.size else np.zeros((6, 6), dtype=np.float64)
    return NullspaceMetrics(projector, rank, 6 - rank, singular_values, tolerance)


def dls_primary_command(jacobian: Any, residual: Any, damping: float) -> np.ndarray:
    matrix = _finite_array(jacobian, (5, 6), "process_jacobian")
    error = _finite_array(residual, (5,), "process_residual")
    if not math.isfinite(damping) or damping <= 0.0:
        raise ValueError("damping_must_be_finite_and_positive")
    system = matrix @ matrix.T + float(damping) ** 2 * np.eye(5)
    command = -matrix.T @ np.linalg.solve(system, error)
    if not np.isfinite(command).all():
        raise ValueError("nonfinite_primary_command")
    return command


def joint_centering_objective(q: Any, lower: Any, upper: Any) -> tuple[float, np.ndarray]:
    values = _finite_array(q, (6,), "joint_state")
    lo = _finite_array(lower, (6,), "lower_limits")
    hi = _finite_array(upper, (6,), "upper_limits")
    half = (hi - lo) / 2.0
    if np.any(half <= 0.0):
        raise ValueError("joint_limit_ranges_must_be_positive")
    normalized = (values - (lo + hi) / 2.0) / half
    objective = 0.5 * float(normalized @ normalized)
    gradient = (values - (lo + hi) / 2.0) / np.square(half)
    return objective, gradient


def secondary_command(
    q: Any, lower: Any, upper: Any, projector: Any, gain: float
) -> tuple[np.ndarray, float]:
    values = _finite_array(q, (6,), "joint_state")
    N = _finite_array(projector, (6, 6), "nullspace_projector")
    if not math.isfinite(gain) or gain < 0.0:
        raise ValueError("secondary_gain_must_be_finite_and_nonnegative")
    _objective, gradient = joint_centering_objective(values, lower, upper)
    command = -float(gain) * (N @ gradient)
    if not np.isfinite(command).all():
        raise ValueError("nonfinite_secondary_command")
    return command, float(np.linalg.norm(command))


def project_joint_limits(
    proposal: Any, lower: Any, upper: Any, buffer_rad: float
) -> tuple[np.ndarray, list[int]]:
    q = _finite_array(proposal, (6,), "joint_proposal")
    lo = _finite_array(lower, (6,), "lower_limits")
    hi = _finite_array(upper, (6,), "upper_limits")
    if not math.isfinite(buffer_rad) or buffer_rad < 0.0:
        raise ValueError("buffer_must_be_finite_and_nonnegative")
    if np.any(hi - lo <= 2.0 * float(buffer_rad)):
        raise ValueError("buffer_collapses_joint_interval")
    projected = np.clip(q, lo + buffer_rad, hi - buffer_rad)
    clipped = np.flatnonzero(np.abs(projected - q) > 1e-12).astype(int).tolist()
    return projected, clipped


def capability_status(moveit_available: bool, reason: str = "") -> dict[str, str]:
    if moveit_available:
        return {"status": "AVAILABLE", "reason": ""}
    return {"status": "UNAVAILABLE", "reason": reason or "MoveIt2 capability unavailable"}


def compare_baseline(
    reference: Any, generated: Any, *, absolute_tolerance: float = 1e-9
) -> dict[str, Any]:
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(generated, dtype=np.float64)
    valid = a.shape == b.shape and a.ndim == 2 and a.shape[1] == 6 and np.isfinite(a).all() and np.isfinite(b).all()
    if not valid:
        return {"status": "FAIL", "shape_match": False, "max_abs_difference_rad": None}
    difference = float(np.max(np.abs(a - b))) if a.size else 0.0
    return {
        "status": "PASS" if difference <= absolute_tolerance else "FAIL",
        "shape_match": True,
        "max_abs_difference_rad": difference,
        "absolute_tolerance_rad": float(absolute_tolerance),
    }


def result_schema_is_valid(result: Any) -> bool:
    if not isinstance(result, dict):
        return False
    required = {
        "schema", "project", "stage", "source_base_commit", "execution_code_commit",
        "variants", "causal_classification", "measurement_pipeline_status",
        "frozen_robot_baseline_performance_status", "limitations",
    }
    if not required.issubset(result):
        return False
    return (
        result.get("schema") == "p2b1-solver-policy-causal-ablation-v1"
        and result.get("project") == "FAIRINO_FR5"
        and result.get("stage") == "P2-B1"
        and isinstance(result.get("variants"), list)
        and result.get("causal_classification") in {
            "BUFFER_POLICY_DOMINATED", "PROCESS_TASK_FORMULATION_DOMINATED",
            "REDUNDANCY_ALLOCATION_DOMINATED", "MULTIFACTOR", "TASK_GEOMETRY_LIMITED", "INCONCLUSIVE",
        }
    )
