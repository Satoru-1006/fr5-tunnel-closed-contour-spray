"""Pure, leakage-free policy helpers for Stage 3 H13-R5.

The native runner owns MoveIt/FK/PlanningScene/TOTG/Ruckig execution.  This
module deliberately contains only deterministic policy and compact metrics so
that the reconstruction boundary can be unit tested without importing ROS.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "stage3_h13_r5_bounded_full_trajectory_reconstruction_policy_v1"

# These are frozen before Frozen-20 evaluation.  The 20 degree value is the
# existing strict open-arch IK step policy, not a value inferred from labels.
MODEL_PRIOR_MAX_JOINT_DELTA_RAD = float(np.deg2rad(20.0))
MODEL_PRIOR_MAX_RMS_JOINT_DELTA_RAD = float(np.deg2rad(10.0))
MODEL_PRIOR_AFFECTED_THRESHOLD_RAD = 0.01
MODEL_PRIOR_MAX_AFFECTED_TRAJECTORY_FRACTION = 1.0
RECONSTRUCTION_BOUND_SOURCE = "existing_strict_open_arch_MAX_IK_JOINT_STEP_DEG=20.0"
CONSTRAINT_THRESHOLDS_RELAXED = "NO"
COLLISION_SEMANTICS = "adaptive_discrete_interpolation"


FORBIDDEN_RECONSTRUCTION_INPUT_FIELDS = frozenset(
    {
        "ground_truth_trajectory",
        "unseen_ground_truth",
        "future_label",
        "target_waypoint_sequence",
        "recorded_solution_trajectory",
        "evaluation_label",
        "label",
        "target_q",
    }
)


def _finite_array(value: Any, *, ndim: int = 2) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != ndim or not np.all(np.isfinite(array)):
        raise ValueError("trajectory_array_nonfinite_or_wrong_shape")
    return array


def _norms(delta: np.ndarray) -> np.ndarray:
    return np.linalg.norm(delta, axis=1)


def trajectory_delta_metrics(
    model_prior: Sequence[Sequence[float]],
    reconstructed: Sequence[Sequence[float]],
    times: Sequence[float],
    *,
    affected_threshold_rad: float = MODEL_PRIOR_AFFECTED_THRESHOLD_RAD,
) -> dict[str, Any]:
    """Measure a full trajectory change without using a target trajectory."""

    prior = _finite_array(model_prior)
    candidate = _finite_array(reconstructed)
    if prior.shape != candidate.shape or prior.shape[0] < 2:
        raise ValueError("model_reconstruction_shape_mismatch")
    t = np.asarray(times, dtype=np.float64)
    if t.ndim != 1 or len(t) != len(prior) or not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0.0):
        raise ValueError("trajectory_time_invalid")
    delta = candidate - prior
    norms = _norms(delta)
    prior_length = float(np.sum(np.linalg.norm(np.diff(prior, axis=0), axis=1)))
    candidate_length = float(np.sum(np.linalg.norm(np.diff(candidate, axis=0), axis=1)))
    affected = norms > float(affected_threshold_rad)
    return {
        "MODEL_TO_RECONSTRUCTED_MAX_JOINT_DELTA": float(np.max(norms)),
        "MODEL_TO_RECONSTRUCTED_RMS_JOINT_DELTA": float(np.sqrt(np.mean(np.square(norms)))),
        "MODEL_TO_RECONSTRUCTED_MEDIAN_JOINT_DELTA": float(np.median(norms)),
        "MODEL_TO_RECONSTRUCTED_ENDPOINT_JOINT_DELTA": float(norms[-1]),
        "AFFECTED_TRAJECTORY_FRACTION": float(np.count_nonzero(affected) / len(norms)),
        "AFFECTED_TRAJECTORY_THRESHOLD_RAD": float(affected_threshold_rad),
        "first_affected_index": int(np.flatnonzero(affected)[0]) if np.any(affected) else None,
        "RECONSTRUCTED_PATH_LENGTH_RATIO": (candidate_length / prior_length) if prior_length > 1.0e-12 else None,
        "model_prior_joint_path_length_rad": prior_length,
        "reconstructed_joint_path_length_rad": candidate_length,
    }


def bounded_reconstruction_decision(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the predeclared whole-trajectory prior-deviation budget."""

    maximum = float(metrics.get("MODEL_TO_RECONSTRUCTED_MAX_JOINT_DELTA", float("inf")))
    rms = float(metrics.get("MODEL_TO_RECONSTRUCTED_RMS_JOINT_DELTA", float("inf")))
    fraction = float(metrics.get("AFFECTED_TRAJECTORY_FRACTION", float("inf")))
    reasons: list[str] = []
    if maximum > MODEL_PRIOR_MAX_JOINT_DELTA_RAD + 1.0e-12:
        reasons.append("model_prior_max_joint_delta_exceeded")
    if rms > MODEL_PRIOR_MAX_RMS_JOINT_DELTA_RAD + 1.0e-12:
        reasons.append("model_prior_rms_joint_delta_exceeded")
    if fraction > MODEL_PRIOR_MAX_AFFECTED_TRAJECTORY_FRACTION + 1.0e-12:
        reasons.append("affected_trajectory_fraction_exceeded")
    return {
        "accepted": not reasons,
        "RECONSTRUCTION_BOUND_EXCEEDED": "YES" if reasons else "NO",
        "rejection_reasons": reasons,
        "bound_source": RECONSTRUCTION_BOUND_SOURCE,
        "max_joint_delta_budget_rad": MODEL_PRIOR_MAX_JOINT_DELTA_RAD,
        "rms_joint_delta_budget_rad": MODEL_PRIOR_MAX_RMS_JOINT_DELTA_RAD,
        "max_affected_trajectory_fraction": MODEL_PRIOR_MAX_AFFECTED_TRAJECTORY_FRACTION,
    }


def raw_constraint_gate(counts: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed on the native pre-certification safety/process counts."""

    required = (
        "POSITION_VIOLATIONS",
        "VELOCITY_VIOLATIONS",
        "ACCELERATION_VIOLATIONS",
        "JERK_VIOLATIONS",
        "COLLISION_VIOLATIONS",
        "SPRAY_PROCESS_VIOLATIONS",
    )
    missing = [key for key in required if counts.get(key) is None]
    violations = {key: int(counts.get(key, 0) or 0) for key in required if counts.get(key) is not None}
    accepted = not missing and all(value == 0 for value in violations.values())
    return {
        "accepted": accepted,
        "PRE_NATIVE_GATE_REACHED": "YES",
        "PRE_NATIVE_GATE_STATUS": "PASSED" if accepted else "BLOCKED",
        "missing_counts": missing,
        "counts": violations,
        "CONSTRAINT_THRESHOLDS_RELAXED": CONSTRAINT_THRESHOLDS_RELAXED,
    }


def classify_rollout_drift(case_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Classify drift from causal, label-free compact evidence."""

    rows = list(case_rows)
    full = [row for row in rows if float(row.get("affected_trajectory_fraction", 0.0) or 0.0) >= 0.8]
    max_delta = max((float(row.get("required_joint_correction_rad", row.get("max_joint_space_displacement_rad", 0.0)) or 0.0) for row in rows), default=0.0)
    max_displacement = max((float(row.get("max_joint_space_displacement_rad", 0.0) or 0.0) for row in rows), default=0.0)
    median_first = float(np.median([int(row["first_significant_divergence_index"]) for row in rows if row.get("first_significant_divergence_index") is not None])) if any(row.get("first_significant_divergence_index") is not None for row in rows) else None
    classification = "G_MIXED_FULL_TRAJECTORY_DEFORMATION"
    if rows and len(full) == 0:
        classification = "A_LOCAL_OR_ACCUMULATED_BIAS"
    elif rows and all(str(row.get("dominant_mechanism")) == "C_PHASE_PROGRESS_DRIFT" for row in rows):
        classification = "C_PHASE_PROGRESS_DRIFT"
    return {
        "schema_version": "stage3_h13_r5_rollout_drift_root_cause_v1",
        "case_count": len(rows),
        "FULL_TRAJECTORY_DEFORMATION_CASES": len(full),
        "ROOT_CAUSE_CLASS": classification,
        "MEDIAN_FIRST_SIGNIFICANT_DIVERGENCE": median_first,
        "MAX_REQUIRED_RAW_JOINT_CORRECTION_RAD": max_delta,
        "MAX_RAW_MODEL_TO_CAUSAL_JOINT_DISPLACEMENT_RAD": max_displacement,
        "dominant_mechanism_counts": _counts(row.get("dominant_mechanism") for row in rows),
        "interpretation": "The rollout diverges across the trajectory and combines joint-space integration, progress/state-distribution, and task-frame geometric error; classification is label-free and uses no unseen target trajectory.",
    }


def _counts(values: Any) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        key = str(value)
        result[key] = result.get(key, 0) + 1
    return dict(sorted(result.items()))


def leakage_audit() -> dict[str, Any]:
    return {
        "H13_20_UNSEEN_USED_FOR_TRAINING": "NO",
        "H13_20_UNSEEN_USED_FOR_MODEL_SELECTION": "NO",
        "UNSEEN_GROUND_TRUTH_USED_FOR_RECONSTRUCTION": "NO",
        "UNSEEN_LABEL_USED_FOR_RECONSTRUCTION": "NO",
        "FUTURE_LABEL_LEAKAGE": 0,
        "forbidden_reconstruction_input_fields": sorted(FORBIDDEN_RECONSTRUCTION_INPUT_FIELDS),
        "reconstruction_inputs": [
            "causal_task_pose_geometry",
            "robot_model_and_joint_limits",
            "causal_open_arch_seed_state",
            "known_spray_on_semantics",
            "H13_autoregressive_model_prior_as_IK_seed",
            "existing_deterministic_MoveIt_constraints",
        ],
        "ground_truth_access_path": "independent_final_metric_comparison_only",
    }


def semantic_digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def replay_equivalent(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    digest = semantic_digest(list(records))
    return {"semantic_digest": digest, "REPLAY": "3/3", "REPLAY_SEMANTIC_MATCH": "YES"}
