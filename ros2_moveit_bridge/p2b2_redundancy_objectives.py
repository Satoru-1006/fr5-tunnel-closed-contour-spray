"""Fail-closed secondary objectives and joint-margin summaries for P2-B2."""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


JOINT_NAMES = tuple(f"j{i}" for i in range(1, 7))
SUPPORTED_SECONDARY_OBJECTIVES = {"joint_centering", "joint_limit_barrier"}
RESULT_REQUIRED_FIELDS = {
    "PROJECT", "STAGE", "SOURCE_BASE_COMMIT", "EXECUTION_CODE_COMMIT",
    "ARTIFACT_PUBLISH_COMMIT", "P2B2_STATUS", "P2B2_CANDIDATE_REFERENCE",
    "P2B2_REFERENCE_REPRODUCTION", "AXIS_COUNT", "JOINT_AXIS_COUNT",
    "TOTAL_AXIS_MEASUREMENT_ROWS", "AVAILABLE_FORMAL_AXIS_COUNT", "UNAVAILABLE_FORMAL_AXIS_COUNT",
    "ROBUSTNESS_MAPPING_STATUS", "CONTROLLING_JOINT_AXIS",
    "CONTROLLING_JOINT_DIRECTION", "CONTROLLING_JOINT_MARGIN_RAD",
    "CONTROLLING_JOINT_WAYPOINT", "CONTROLLING_FAILURE_MODE",
    "MIN_NORMALIZED_JOINT_MARGIN", "MIN_NORMALIZED_JOINT_MARGIN_JOINT",
    "MIN_NORMALIZED_JOINT_MARGIN_WAYPOINT", "P2B1_B1_EFFICACY_CONFOUND",
    "REDUNDANCY_VARIANTS", "BEST_VALIDATED_VARIANT", "BEST_VARIANT_REASON",
    "BOTTLENECK_MIGRATION_STATUS", "OLD_BOTTLENECK", "NEW_BOTTLENECK",
    "COLLISION_METHOD", "STRICT_SELF_CCD", "DYNAMIC_LIMIT_PROVENANCE",
    "HARDWARE_VALIDATION", "HARDWARE_SAFETY_CERTIFIED", "SOURCE_REPLAY",
    "BUILD_REPLAY", "TEST_REPLAY", "SCIENTIFIC_CAMPAIGN_REPLAY", "LIMITATIONS",
}


def _finite_vector(value: Any, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (6,) or not np.isfinite(array).all():
        raise ValueError(f"{label}_must_be_finite_shape_6")
    return array


def _limits(lower: Any, upper: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo = _finite_vector(lower, "lower_limits")
    hi = _finite_vector(upper, "upper_limits")
    widths = hi - lo
    if np.any(widths <= 0.0):
        raise ValueError("joint_limit_widths_must_be_positive")
    return lo, hi, widths


def joint_centering_objective(q: Any, lower: Any, upper: Any) -> tuple[float, np.ndarray]:
    """Dimensionless quadratic centering objective and its rad^-1 gradient."""
    values = _finite_vector(q, "joint_state")
    lo, hi, widths = _limits(lower, upper)
    normalized = (values - (lo + hi) / 2.0) / (widths / 2.0)
    gradient = normalized / (widths / 2.0)
    return 0.5 * float(normalized @ normalized), gradient


def joint_limit_barrier_objective(
    q: Any, lower: Any, upper: Any, *, epsilon_normalized: float = 0.02,
) -> tuple[float, np.ndarray]:
    """Smooth joint-limit barrier and analytic gradient.

    For normalized lower/upper slacks s_l=(q-l)/(u-l), s_u=(u-q)/(u-l),
    H=sum_j[(s_l+eps)^-2 + (s_u+eps)^-2]. The positive normalized epsilon
    keeps the objective finite near a boundary; states at or beyond a hard
    position limit fail closed.
    """
    values = _finite_vector(q, "joint_state")
    lo, hi, widths = _limits(lower, upper)
    if not math.isfinite(epsilon_normalized) or epsilon_normalized <= 0.0:
        raise ValueError("barrier_epsilon_must_be_finite_and_positive")
    if np.any(values <= lo) or np.any(values >= hi):
        raise ValueError("barrier_requires_strictly_in_bounds_state")
    lower_slack = (values - lo) / widths
    upper_slack = (hi - values) / widths
    lower_term = lower_slack + epsilon_normalized
    upper_term = upper_slack + epsilon_normalized
    objective = float(np.sum(np.reciprocal(lower_term**2) + np.reciprocal(upper_term**2)))
    gradient = (-2.0 / lower_term**3 + 2.0 / upper_term**3) / widths
    if not math.isfinite(objective) or not np.isfinite(gradient).all():
        raise ValueError("nonfinite_joint_limit_barrier")
    return objective, gradient


def secondary_objective(
    q: Any, lower: Any, upper: Any, objective_name: str,
) -> tuple[float, np.ndarray]:
    name = str(objective_name)
    if name == "joint_centering":
        return joint_centering_objective(q, lower, upper)
    if name == "joint_limit_barrier":
        return joint_limit_barrier_objective(q, lower, upper)
    raise ValueError("unsupported_secondary_objective")


def projected_secondary_command(
    q: Any, lower: Any, upper: Any, projector: Any, gain: float,
    objective_name: str, *, task_jacobian: Any | None = None,
    invariance_tolerance: float = 1.0e-8,
) -> tuple[np.ndarray, float, float]:
    """Return a bounded null-space command and verify first-order invariance."""
    values = _finite_vector(q, "joint_state")
    matrix = np.asarray(projector, dtype=np.float64)
    if matrix.shape != (6, 6) or not np.isfinite(matrix).all():
        raise ValueError("projector_must_be_finite_shape_6x6")
    if not math.isfinite(gain) or gain < 0.0:
        raise ValueError("gain_must_be_finite_and_nonnegative")
    if not math.isfinite(invariance_tolerance) or invariance_tolerance <= 0.0:
        raise ValueError("invariance_tolerance_must_be_finite_and_positive")
    if not np.allclose(matrix, matrix.T, atol=1e-8, rtol=0.0):
        raise ValueError("projector_not_symmetric")
    if not np.allclose(matrix @ matrix, matrix, atol=1e-8, rtol=0.0):
        raise ValueError("projector_not_idempotent")
    _objective, gradient = secondary_objective(values, lower, upper, objective_name)
    command = -float(gain) * (matrix @ gradient)
    if not np.isfinite(command).all():
        raise ValueError("nonfinite_secondary_command")
    residual_norm = 0.0
    if task_jacobian is not None:
        jacobian = np.asarray(task_jacobian, dtype=np.float64)
        if jacobian.ndim != 2 or jacobian.shape[1] != 6 or not np.isfinite(jacobian).all():
            raise ValueError("task_jacobian_must_be_finite_with_six_columns")
        residual_norm = float(np.linalg.norm(jacobian @ command))
        scale = max(1.0, float(np.linalg.norm(jacobian)) * float(np.linalg.norm(command)))
        if not math.isfinite(residual_norm) or residual_norm > invariance_tolerance * scale:
            raise ValueError("secondary_command_violates_first_order_task_invariance")
    return command, float(np.linalg.norm(command)), residual_norm


def rank_aware_secondary_command(
    q: Any, lower: Any, upper: Any, projector: Any, gain: float,
    objective_name: str, rank: int, *, task_jacobian: Any | None = None,
) -> tuple[np.ndarray, float, float]:
    """Apply secondary motion only for the intended rank-5, one-DOF nullspace.

    Rank-deficient or full-rank process Jacobians produce a bounded zero
    secondary command. This keeps a rank change from silently widening the
    optimizer's authority beyond the frozen ablation design.
    """
    if int(rank) != 5:
        return np.zeros(6, dtype=np.float64), 0.0, 0.0
    return projected_secondary_command(
        q, lower, upper, projector, gain, objective_name,
        task_jacobian=task_jacobian,
    )


def normalized_joint_margin_profile(q: Any, lower: Any, upper: Any) -> dict[str, Any]:
    """Compute raw and range-normalized joint margins for every trajectory row."""
    states = np.asarray(q, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != 6 or states.shape[0] < 1 or not np.isfinite(states).all():
        raise ValueError("trajectory_must_be_finite_shape_n_by_6")
    lo, hi, widths = _limits(lower, upper)
    lower_margins = states - lo[None, :]
    upper_margins = hi[None, :] - states
    raw = np.minimum(lower_margins, upper_margins)
    normalized = raw / widths[None, :]
    waypoint, joint = np.unravel_index(int(np.argmin(normalized)), normalized.shape)
    side = "lower" if lower_margins[waypoint, joint] <= upper_margins[waypoint, joint] else "upper"
    rows = [
        {
            "waypoint": int(i),
            "joint_margins_rad": [float(v) for v in raw[i]],
            "normalized_joint_margins": [float(v) for v in normalized[i]],
        }
        for i in range(states.shape[0])
    ]
    return {
        "waypoint_count": int(states.shape[0]),
        "minimum_joint_margin_rad": float(raw[waypoint, joint]),
        "minimum_normalized_joint_margin": float(normalized[waypoint, joint]),
        "controlling_joint": JOINT_NAMES[joint],
        "controlling_waypoint": int(waypoint),
        "controlling_limit_side": side,
        "raw_joint_margin_matrix_rad": raw.tolist(),
        "normalized_joint_margin_matrix": normalized.tolist(),
        "per_waypoint": rows,
    }


def rank_joint_axis_results(axis_results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Rank only comparable, supported joint-state boundaries in radians."""
    selected: list[dict[str, Any]] = []
    for raw in axis_results:
        spec = raw.get("specification") or {}
        axis_id = str(spec.get("axis_id", raw.get("axis_id", "")))
        family = str(spec.get("perturbation_family", ""))
        if not axis_id.startswith("joint:") or family != "JOINT_STATE":
            continue
        if spec.get("units") != "rad":
            raise ValueError("joint_axis_units_must_be_radians")
        margin = raw.get("estimated_raw_axis_margin")
        value = margin.get("value") if isinstance(margin, Mapping) else None
        if value is None or not math.isfinite(float(value)):
            selected.append({"axis_id": axis_id, "margin_rad": None, "result": dict(raw)})
        else:
            selected.append({"axis_id": axis_id, "margin_rad": float(value), "result": dict(raw)})
    return sorted(selected, key=lambda row: (
        row["margin_rad"] is None,
        math.inf if row["margin_rad"] is None else row["margin_rad"],
        row["axis_id"],
    ))


def rank_axis_results_by_family(axis_results: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Sort margins within each perturbation family and unit; never mix units."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for raw in axis_results:
        spec = raw.get("specification") or {}
        family = str(spec.get("perturbation_family", "UNKNOWN"))
        units = str(spec.get("units", "UNKNOWN"))
        margin = raw.get("estimated_raw_axis_margin")
        value = margin.get("value") if isinstance(margin, Mapping) else None
        if value is not None and not math.isfinite(float(value)):
            raise ValueError("axis_margin_must_be_finite_or_null")
        grouped.setdefault((family, units), []).append({
            "axis_id": str(spec.get("axis_id", raw.get("axis_id", ""))),
            "margin": None if value is None else float(value),
            "units": units,
            "result": dict(raw),
        })
    output: dict[str, list[dict[str, Any]]] = {}
    for (family, units), rows in grouped.items():
        rows.sort(key=lambda row: (
            row["margin"] is None,
            math.inf if row["margin"] is None else row["margin"],
            row["axis_id"],
        ))
        output[f"{family}|{units}"] = rows
    return output


def validate_result_schema(result: Any) -> bool:
    if not isinstance(result, dict) or not RESULT_REQUIRED_FIELDS.issubset(result):
        return False
    if result.get("PROJECT") != "FAIRINO_FR5" or result.get("STAGE") != "P2-B2":
        return False
    if not isinstance(result.get("REDUNDANCY_VARIANTS"), list):
        return False
    if result.get("COLLISION_METHOD") != "adaptive_discrete_interpolation":
        return False
    if result.get("STRICT_SELF_CCD") != "NOT_AVAILABLE":
        return False
    if result.get("HARDWARE_VALIDATION") != "NOT_RUN" or result.get("HARDWARE_SAFETY_CERTIFIED") != "NO":
        return False
    return True


def validate_provenance_schema(provenance: Any) -> bool:
    if not isinstance(provenance, dict):
        return False
    required = {
        "project", "stage", "source_base_commit", "execution_code_commit", "branch",
        "repository", "model_source_commit", "runtime", "inputs", "collision_method",
    }
    if not required.issubset(provenance) or provenance.get("project") != "FAIRINO_FR5" or provenance.get("stage") != "P2-B2":
        return False
    if provenance.get("collision_method") != "adaptive_discrete_interpolation":
        return False
    if not isinstance(provenance.get("inputs"), dict) or not isinstance(provenance.get("runtime"), dict):
        return False
    return True


__all__ = [
    "JOINT_NAMES", "SUPPORTED_SECONDARY_OBJECTIVES", "joint_centering_objective",
    "joint_limit_barrier_objective", "secondary_objective", "projected_secondary_command",
    "rank_aware_secondary_command",
    "normalized_joint_margin_profile", "rank_joint_axis_results",
    "rank_axis_results_by_family", "validate_result_schema", "validate_provenance_schema",
]
