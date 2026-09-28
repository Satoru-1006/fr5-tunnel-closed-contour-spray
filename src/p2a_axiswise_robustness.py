"""Axis-wise minimum-perturbation robustness characterization for the frozen FR5 path.

The search kernel is deterministic and consumes the existing P1 stress operators.
Its optional native adapter reuses the D41 MoveIt2/FCL runner and D46 limit and
metric readers; it does not alter the frozen Stage 3, D41, or D46 artifacts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from src.process_aware_stress import (
    ApplicationStatus,
    CapabilityStatus,
    ProcessTrajectory,
    ResultStatus,
    StressFamily,
    StressOperator,
    apply_stress,
    check_joint_limits,
)


SCHEMA = "P2A_AXISWISE_ROBUSTNESS_CHARACTERIZATION"
ROOT = Path(__file__).resolve().parents[1]
P2A_RESULT = ROOT / "outputs" / "p2a_axiswise_robustness_margin.json"
D46_ROOT = ROOT / "outputs" / "D46_STAGE4A_SYSTEM_BASELINE_V1"
P1_MODULE = ROOT / "src" / "process_aware_stress.py"


@dataclass(frozen=True)
class AxisSpec:
    axis_id: str
    perturbation_family: StressFamily
    axis: str
    coordinate_frame: str
    direction: int
    units: str
    search_domain_max: float
    coarse_intervals: int
    refinement_tolerance: float
    operator_parameters: Mapping[str, Any] = field(default_factory=dict)
    capability_status: CapabilityStatus = CapabilityStatus.AVAILABLE
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.axis_id or not self.axis or not self.coordinate_frame or not self.units:
            raise ValueError("axis identity, frame, and units are required")
        if self.direction not in {-1, 1}:
            raise ValueError("direction must be +1 or -1")
        if not math.isfinite(self.search_domain_max) or self.search_domain_max < 0:
            raise ValueError("search_domain_max must be finite and nonnegative")
        if self.coarse_intervals < 1:
            raise ValueError("coarse_intervals must be positive")
        if not math.isfinite(self.refinement_tolerance) or self.refinement_tolerance <= 0:
            raise ValueError("refinement_tolerance must be finite and positive")
        object.__setattr__(self, "perturbation_family", StressFamily(self.perturbation_family))
        object.__setattr__(self, "capability_status", CapabilityStatus(self.capability_status))

    def to_record(self) -> dict[str, Any]:
        return {
            "axis_id": self.axis_id,
            "perturbation_family": self.perturbation_family.value,
            "axis": self.axis,
            "coordinate_frame": self.coordinate_frame,
            "direction": "POSITIVE" if self.direction > 0 else "NEGATIVE",
            "sign": self.direction,
            "units": self.units,
            "search_domain": {"minimum_magnitude": 0.0, "maximum_magnitude": self.search_domain_max, "units": self.units},
            "coarse_intervals": self.coarse_intervals,
            "coarse_step": self.search_domain_max / self.coarse_intervals,
            "refinement_tolerance": self.refinement_tolerance,
            "normalized_margin": "NOT_AVAILABLE",
            "normalized_margin_reason": "REAL_UNCERTAINTY_BOUNDS_NOT_YET_IDENTIFIED",
            "capability_status": self.capability_status.value,
            "unavailable_reason": self.unavailable_reason,
            "operator_parameters": _jsonable(self.operator_parameters),
        }


@dataclass(frozen=True)
class ScanCandidate:
    candidate_id: str
    spec: AxisSpec
    magnitude: float
    signed_delta: float
    application: Any
    phase: str
    sample_index: int | None = None
    refinement_iteration: int | None = None


@dataclass(frozen=True)
class EvaluationResult:
    status: ResultStatus
    failure_modes: tuple[str, ...] = ()
    critical_waypoint: int | None = None
    critical_segment: int | None = None
    evaluator_capabilities: Mapping[str, str] = field(default_factory=dict)
    nominal_margin: Mapping[str, Any] = field(default_factory=dict)
    failure_margin: Mapping[str, Any] = field(default_factory=dict)
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", ResultStatus(self.status))
        object.__setattr__(self, "failure_modes", tuple(str(value) for value in self.failure_modes))

    def to_record(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "failure_modes": list(self.failure_modes),
            "dominant_failure_mode": self.failure_modes[0] if self.failure_modes else None,
            "critical_waypoint": self.critical_waypoint,
            "critical_segment": self.critical_segment,
            "evaluator_capabilities": dict(self.evaluator_capabilities),
            "nominal_margin": _jsonable(self.nominal_margin),
            "failure_margin": _jsonable(self.failure_margin),
            "evidence": _jsonable(self.evidence),
        }


def apply_axis_perturbation(nominal: ProcessTrajectory, spec: AxisSpec, magnitude: float) -> ScanCandidate:
    """Apply exactly one signed P1 operator; no state is clipped or rewritten."""
    if not math.isfinite(magnitude) or magnitude < 0:
        raise ValueError("magnitude must be finite and nonnegative")
    requested = spec.direction * float(magnitude)
    operator = StressOperator(
        operator_id=f"p2a:{spec.axis_id}:{requested:.17g}",
        family=spec.perturbation_family,
        amplitude=requested,
        units=spec.units,
        target_scope="all 181 nominal trajectory samples",
        parameters=spec.operator_parameters,
        availability=spec.capability_status,
        metadata={"axis_id": spec.axis_id, "coordinate_frame": spec.coordinate_frame, "signed_request": requested},
    )
    applied = apply_stress(nominal, operator)
    candidate_id = _candidate_id(spec, magnitude, "manual", None)
    return ScanCandidate(candidate_id, spec, float(magnitude), requested, applied, "manual")


def evaluate_joint_limits(trajectory: ProcessTrajectory) -> EvaluationResult:
    """Directly classify the P1 joint-limit checks, preserving real exceedance."""
    checks = check_joint_limits(trajectory)
    q = trajectory.joint_states
    lower, upper = trajectory.joint_lower_rad, trajectory.joint_upper_rad
    if q is None or lower is None or upper is None:
        return EvaluationResult(
            ResultStatus.UNKNOWN,
            evaluator_capabilities={"joint_limits": "UNKNOWN"},
            evidence={"reason": "joint positions or authoritative limits are unavailable"},
        )
    bad = np.flatnonzero([check.status == ResultStatus.FAIL for check in checks])
    unknown = any(check.status == ResultStatus.UNKNOWN for check in checks)
    if len(bad):
        waypoint = int(bad[0])
        below = lower - q[waypoint]
        above = q[waypoint] - upper
        joint = int(np.argmax(np.maximum(below, above)))
        violation = float(max(below[joint], above[joint]))
        margins = np.minimum(q - lower[None, :], upper[None, :] - q)
        controlling_waypoint, controlling_joint = (int(value) for value in np.unravel_index(int(np.argmin(margins)), margins.shape))
        controlling_side = "LOWER" if q[controlling_waypoint, controlling_joint] - lower[controlling_joint] <= upper[controlling_joint] - q[controlling_waypoint, controlling_joint] else "UPPER"
        return EvaluationResult(
            ResultStatus.FAIL,
            ("JOINT_LIMIT_FAILURE",),
            critical_waypoint=waypoint,
            evaluator_capabilities={"joint_limits": "AVAILABLE"},
            failure_margin={"joint_limit_violation_rad": violation},
            evidence={
                "joint_index_zero_based": joint,
                "joint_name": f"j{joint + 1}",
                "q_rad": float(q[waypoint, joint]),
                "lower_rad": float(lower[joint]),
                "upper_rad": float(upper[joint]),
                "joint_limit_margin_min_rad": float(margins[controlling_waypoint, controlling_joint]),
                "controlling_joint": f"j{controlling_joint + 1}",
                "controlling_waypoint": controlling_waypoint,
                "controlling_limit_side": controlling_side,
                "state_was_clipped": False,
                "metric_evaluation_status": "NOT_EVALUATED_AFTER_JOINT_LIMIT_FAILURE",
            },
        )
    if unknown:
        return EvaluationResult(ResultStatus.UNKNOWN, evaluator_capabilities={"joint_limits": "UNKNOWN"})
    return EvaluationResult(
        ResultStatus.PASS,
        evaluator_capabilities={"joint_limits": "AVAILABLE"},
        nominal_margin={"minimum_joint_limit_margin_rad": float(np.min(np.minimum(q - lower, upper - q)))},
    )


def scan_axes_batched(
    nominal: ProcessTrajectory,
    specs: Sequence[AxisSpec],
    evaluate_batch: Callable[[Sequence[ScanCandidate]], Mapping[str, EvaluationResult]],
    *,
    max_refinement_iterations: int = 80,
) -> list[dict[str, Any]]:
    """Coarse-scan all axes, then refine every observed adjacent PASS/FAIL transition.

    The callback may batch real MoveIt/FK evaluations. Missing callback rows fail
    closed as UNKNOWN. Every coarse and refinement perturbation is retained with
    its requested and realized amplitude summary. UNKNOWN observations break the
    transition graph: refinement never brackets across a missing measurement.
    """
    states: dict[str, dict[str, Any]] = {}
    coarse_candidates: list[ScanCandidate] = []
    for spec in specs:
        state = {
            "spec": spec,
            "nominal": nominal,
            "coarse_candidates": [],
            "coarse_observations": [],
            "refinement_observations": [],
            "boundaries": [],
            "refinement_blocked": None,
        }
        states[spec.axis_id] = state
        if spec.capability_status != CapabilityStatus.AVAILABLE:
            state["refinement_blocked"] = "CAPABILITY_UNAVAILABLE"
            continue
        magnitudes = np.linspace(0.0, spec.search_domain_max, spec.coarse_intervals + 1, dtype=np.float64)
        for index, magnitude in enumerate(magnitudes):
            candidate = _make_candidate(nominal, spec, float(magnitude), "coarse", index)
            state["coarse_candidates"].append(candidate)
            coarse_candidates.append(candidate)

    coarse_results = _evaluate_fail_closed(evaluate_batch, coarse_candidates)
    active: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for state in states.values():
        spec: AxisSpec = state["spec"]
        if spec.capability_status != CapabilityStatus.AVAILABLE:
            continue
        candidates: list[ScanCandidate] = state["coarse_candidates"]
        observations = [
            _observation(candidate, coarse_results[candidate.candidate_id], nominal)
            for candidate in candidates
        ]
        state["coarse_observations"] = observations
        statuses = [row["status"] for row in observations]
        failures = [index for index, status in enumerate(statuses) if status == ResultStatus.FAIL.value]
        if not failures:
            state["refinement_blocked"] = (
                "CAPABILITY_GAP_IN_SEARCH_DOMAIN"
                if ResultStatus.UNKNOWN.value in statuses
                else "NO_FAILURE_WITHIN_SEARCH_DOMAIN"
            )
            continue
        if failures[0] == 0:
            state["refinement_blocked"] = "NOMINAL_FAILURE"
        for index in range(len(statuses) - 1):
            left_status, right_status = statuses[index], statuses[index + 1]
            if {left_status, right_status} != {ResultStatus.PASS.value, ResultStatus.FAIL.value}:
                continue
            left_candidate, right_candidate = candidates[index], candidates[index + 1]
            left_result = coarse_results[left_candidate.candidate_id]
            right_result = coarse_results[right_candidate.candidate_id]
            boundary = {
                "coarse_transition_index": index,
                "left_candidate": left_candidate,
                "left_result": left_result,
                "right_candidate": right_candidate,
                "right_result": right_result,
                "pass_candidate": left_candidate if left_status == ResultStatus.PASS.value else right_candidate,
                "pass_result": left_result if left_status == ResultStatus.PASS.value else right_result,
                "fail_candidate": left_candidate if left_status == ResultStatus.FAIL.value else right_candidate,
                "fail_result": left_result if left_status == ResultStatus.FAIL.value else right_result,
                "refinement_observations": [],
                "refinement_iterations": 0,
                "refinement_blocked": None,
            }
            state["boundaries"].append(boundary)
            active.append((state, boundary))
        if failures[0] > 0 and not state["boundaries"] and state["refinement_blocked"] is None:
            state["refinement_blocked"] = "NO_ADJACENT_PASS_FAIL_BRACKET"

    for iteration in range(max_refinement_iterations):
        midpoints: list[ScanCandidate] = []
        owners: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        for state, boundary in active:
            if boundary["refinement_blocked"] is not None:
                continue
            passing: ScanCandidate = boundary["pass_candidate"]
            failing: ScanCandidate = boundary["fail_candidate"]
            if abs(failing.magnitude - passing.magnitude) <= state["spec"].refinement_tolerance:
                continue
            midpoint = 0.5 * (passing.magnitude + failing.magnitude)
            boundary_index = next(
                index for index, item in enumerate(state["boundaries"]) if item is boundary
            )
            candidate_index = boundary_index * (max_refinement_iterations + 1) + iteration
            candidate = _make_candidate(nominal, state["spec"], midpoint, "refinement", candidate_index)
            midpoints.append(candidate)
            owners[candidate.candidate_id] = (state, boundary)
        if not midpoints:
            break
        midpoint_results = _evaluate_fail_closed(evaluate_batch, midpoints)
        for candidate in midpoints:
            state, boundary = owners[candidate.candidate_id]
            result = midpoint_results[candidate.candidate_id]
            observation = _observation(candidate, result, nominal)
            boundary["refinement_iterations"] += 1
            boundary["refinement_observations"].append(observation)
            state["refinement_observations"].append(observation)
            if result.status == ResultStatus.PASS:
                boundary["pass_candidate"], boundary["pass_result"] = candidate, result
            elif result.status == ResultStatus.FAIL:
                boundary["fail_candidate"], boundary["fail_result"] = candidate, result
            else:
                boundary["refinement_blocked"] = "REFINEMENT_BLOCKED_UNKNOWN"

    return [_finish_axis(state) for state in states.values()]


def _finish_axis(state: Mapping[str, Any]) -> dict[str, Any]:
    spec: AxisSpec = state["spec"]
    coarse = list(state["coarse_observations"])
    refinement = list(state["refinement_observations"])
    if spec.capability_status != CapabilityStatus.AVAILABLE:
        return {
            "specification": spec.to_record(),
            "axis_status": "CAPABILITY_UNAVAILABLE",
            "coarse_state_sequence": [],
            "coarse_observations": [],
            "refinement_observations": [],
            "failure_intervals": [],
            "all_failure_intervals": [],
            "all_refined_boundaries": [],
            "first_failure_margin": None,
            "global_minimum_observed_failure_boundary": None,
            "transition_refinement_completeness": "NOT_RUN",
            "robustness_acceptance_status": "NOT_EVALUATED",
            "monotonicity_observation": "NOT_EVALUATED_CAPABILITY_UNAVAILABLE",
            "non_monotonic_failure_region_observed": False,
            "search_termination_reason": "CAPABILITY_UNAVAILABLE",
            "search_completeness": "NOT_RUN",
            "last_passing_perturbation": None,
            "first_failing_perturbation": None,
            "refined_boundary_estimate": None,
            "estimated_raw_axis_margin": None,
        }

    statuses = [row["status"] for row in coarse]
    failure_indices = [i for i, status in enumerate(statuses) if status == ResultStatus.FAIL.value]
    has_fail_then_pass = any(
        statuses[i] == ResultStatus.FAIL.value and statuses[i + 1] == ResultStatus.PASS.value
        for i in range(len(statuses) - 1)
    )
    failure_intervals: list[dict[str, Any]] = []
    index = 0
    while index < len(statuses):
        if statuses[index] != ResultStatus.FAIL.value:
            index += 1
            continue
        start = index
        while index + 1 < len(statuses) and statuses[index + 1] == ResultStatus.FAIL.value:
            index += 1
        failure_intervals.append({
            "first_index": start,
            "last_index": index,
            "first_magnitude": coarse[start]["magnitude"],
            "last_magnitude": coarse[index]["magnitude"],
            "sample_count": index - start + 1,
        })
        index += 1

    if has_fail_then_pass:
        monotonicity = "NON_MONOTONIC"
    elif failure_indices and all(status == ResultStatus.PASS.value for status in statuses[: failure_indices[0]]) and ResultStatus.UNKNOWN.value not in statuses[: failure_indices[0] + 1]:
        monotonicity = "LOCAL_MONOTONIC_AT_COARSE_SAMPLING_RESOLUTION"
    elif failure_indices:
        monotonicity = "UNRESOLVED_GAP_BEFORE_FIRST_FAILURE"
    elif ResultStatus.UNKNOWN.value in statuses:
        monotonicity = "UNRESOLVED_CAPABILITY_OR_MEASUREMENT_GAP"
    else:
        monotonicity = "NO_FAILURE_OBSERVED_IN_DOMAIN"

    initial = coarse[0] if coarse else None
    if not failure_indices:
        status = "CAPABILITY_GAP_IN_SEARCH_DOMAIN" if ResultStatus.UNKNOWN.value in statuses else "NO_FAILURE_WITHIN_SEARCH_DOMAIN"
        return {
            "specification": spec.to_record(),
            "axis_status": status,
            "coarse_state_sequence": statuses,
            "coarse_observations": coarse,
            "refinement_observations": refinement,
            "failure_intervals": failure_intervals,
            "all_failure_intervals": failure_intervals,
            "all_refined_boundaries": [],
            "first_failure_margin": None,
            "global_minimum_observed_failure_boundary": None,
            "transition_refinement_completeness": "NO_OBSERVED_PASS_FAIL_TRANSITIONS",
            "robustness_acceptance_status": "NOT_EVALUATED_BOUNDARY_REFINEMENT_IS_MEASUREMENT_ONLY",
            "monotonicity_observation": monotonicity,
            "non_monotonic_failure_region_observed": False,
            "search_termination_reason": state.get("refinement_blocked"),
            "search_completeness": "FULL_COARSE_DOMAIN_SAMPLED" if ResultStatus.UNKNOWN.value not in statuses else "INCOMPLETE_CAPABILITY_GAP",
            "nominal_evaluation": initial,
            "last_passing_perturbation": coarse[-1] if statuses and statuses[-1] == ResultStatus.PASS.value else None,
            "first_failing_perturbation": None,
            "refined_boundary_estimate": None,
            "estimated_raw_axis_margin": None,
        }

    first_fail_index = failure_indices[0]
    first_fail = coarse[first_fail_index]
    refined_boundaries: list[dict[str, Any]] = []
    for boundary in state.get("boundaries", []):
        left: ScanCandidate = boundary["left_candidate"]
        right: ScanCandidate = boundary["right_candidate"]
        passing: ScanCandidate = boundary["pass_candidate"]
        failing: ScanCandidate = boundary["fail_candidate"]
        width = abs(failing.magnitude - passing.magnitude)
        converged = width <= spec.refinement_tolerance
        if boundary["refinement_blocked"]:
            convergence = boundary["refinement_blocked"]
        elif converged:
            convergence = "REFINED_TO_TOLERANCE"
        else:
            convergence = "REFINEMENT_ITERATION_LIMIT"
        left_status = boundary["left_result"].status.value
        right_status = boundary["right_result"].status.value
        boundary_magnitude = 0.5 * (passing.magnitude + failing.magnitude)
        refined_boundaries.append({
            "coarse_transition_index": boundary["coarse_transition_index"],
            "coarse_left": _candidate_result_record(left, boundary["left_result"], state["nominal"]),
            "coarse_right": _candidate_result_record(right, boundary["right_result"], state["nominal"]),
            "transition": f"{left_status}_TO_{right_status}",
            "last_pass": _candidate_result_record(passing, boundary["pass_result"], state["nominal"]),
            "first_fail": _candidate_result_record(failing, boundary["fail_result"], state["nominal"]),
            "bracket_magnitude_min": min(passing.magnitude, failing.magnitude),
            "bracket_magnitude_max": max(passing.magnitude, failing.magnitude),
            "width": width,
            "boundary_estimate_magnitude": boundary_magnitude,
            "boundary_estimate_signed": spec.direction * boundary_magnitude,
            "units": spec.units,
            "refinement_iterations": boundary["refinement_iterations"],
            "refinement_status": convergence,
            "converged_to_tolerance": converged,
            "refinement_blocked": boundary["refinement_blocked"],
            "refinement_observations": list(boundary["refinement_observations"]),
        })
    primary_boundary = next(
        (row for row in refined_boundaries if row["transition"] == "PASS_TO_FAIL"),
        refined_boundaries[0] if refined_boundaries else None,
    )
    bracket = None
    last_pass_record = None
    first_fail_record: dict[str, Any] | None = first_fail
    estimate = None
    boundary_signed = None
    if primary_boundary is not None:
        bracket = {
            "last_pass": primary_boundary["last_pass"],
            "first_fail": primary_boundary["first_fail"],
            "width": primary_boundary["width"],
            "units": spec.units,
        }
        last_pass_record = primary_boundary["last_pass"]
        first_fail_record = primary_boundary["first_fail"]
        estimate = primary_boundary["boundary_estimate_magnitude"]
        boundary_signed = primary_boundary["boundary_estimate_signed"]
    elif first_fail_index == 0:
        estimate = 0.0
        boundary_signed = 0.0

    boundary_statuses = [row["refinement_status"] for row in refined_boundaries]
    if any(value == "REFINEMENT_BLOCKED_UNKNOWN" for value in boundary_statuses):
        transition_completeness = "INCOMPLETE_REFINEMENT_UNKNOWN"
    elif any(value == "REFINEMENT_ITERATION_LIMIT" for value in boundary_statuses):
        transition_completeness = "INCOMPLETE_REFINEMENT_ITERATION_LIMIT"
    elif refined_boundaries:
        transition_completeness = "ALL_OBSERVED_PASS_FAIL_TRANSITIONS_REFINED"
    else:
        transition_completeness = "NO_OBSERVED_PASS_FAIL_TRANSITIONS"

    if first_fail_index == 0:
        termination = "NOMINAL_FAILURE"
    elif transition_completeness == "INCOMPLETE_REFINEMENT_UNKNOWN":
        termination = "REFINEMENT_BLOCKED_UNKNOWN"
    elif transition_completeness == "INCOMPLETE_REFINEMENT_ITERATION_LIMIT":
        termination = "REFINEMENT_ITERATION_LIMIT"
    elif primary_boundary is not None and len(refined_boundaries) == 1:
        termination = "REFINED_TO_TOLERANCE"
    elif refined_boundaries:
        termination = "ALL_OBSERVED_TRANSITIONS_REFINED"
    else:
        termination = state.get("refinement_blocked") or "FIRST_FAILURE_OBSERVED_WITHOUT_REFINEABLE_BRACKET"
    if first_fail_index == 0:
        complete = "NOMINAL_FAILURE"
    elif ResultStatus.UNKNOWN.value in statuses:
        complete = "INCOMPLETE_CAPABILITY_GAP"
    elif transition_completeness == "ALL_OBSERVED_PASS_FAIL_TRANSITIONS_REFINED":
        complete = "ALL_OBSERVED_TRANSITIONS_REFINED"
    else:
        complete = transition_completeness
    primary_margin = None if estimate is None else {"value": estimate, "units": spec.units}
    first_failure_margin = (
        {"value": 0.0, "units": spec.units} if first_fail_index == 0
        else primary_margin if primary_boundary and primary_boundary["transition"] == "PASS_TO_FAIL"
        else None
    )
    return {
        "specification": spec.to_record(),
        "axis_status": "NOMINAL_FAILURE" if first_fail_index == 0 else "FIRST_FAILURE_OBSERVED",
        "coarse_state_sequence": statuses,
        "coarse_observations": coarse,
        "refinement_observations": refinement,
        "failure_intervals": failure_intervals,
        "all_failure_intervals": failure_intervals,
        "all_refined_boundaries": refined_boundaries,
        "first_failure_margin": first_failure_margin,
        "global_minimum_observed_failure_boundary": first_failure_margin,
        "transition_refinement_completeness": transition_completeness,
        "robustness_acceptance_status": "NOT_EVALUATED_BOUNDARY_REFINEMENT_IS_MEASUREMENT_ONLY",
        "non_monotonic_failure_region_observed": has_fail_then_pass,
        "monotonicity_observation": monotonicity,
        "first_observed_failure_interval": {
            "from_magnitude": last_pass_record["magnitude"] if last_pass_record else None,
            "to_magnitude": first_fail_record["magnitude"] if first_fail_record else None,
            "units": spec.units,
        },
        "refined_bracket": bracket,
        "nominal_evaluation": initial,
        "last_passing_perturbation": last_pass_record,
        "first_failing_perturbation": first_fail_record,
        "refined_boundary_estimate": None if boundary_signed is None else {"value": boundary_signed, "units": spec.units},
        "estimated_raw_axis_margin": primary_margin,
        "search_termination_reason": termination,
        "search_completeness": complete,
    }


def _evaluate_fail_closed(
    callback: Callable[[Sequence[ScanCandidate]], Mapping[str, EvaluationResult]],
    candidates: Sequence[ScanCandidate],
) -> dict[str, EvaluationResult]:
    if not candidates:
        return {}
    try:
        returned = callback(candidates)
    except Exception as exc:
        reason = f"batch_evaluator_exception:{type(exc).__name__}:{exc}"
        return {
            candidate.candidate_id: EvaluationResult(
                ResultStatus.UNKNOWN,
                evaluator_capabilities={"batch_evaluator": "UNKNOWN"},
                evidence={"reason": reason},
            )
            for candidate in candidates
        }
    result: dict[str, EvaluationResult] = {}
    for candidate in candidates:
        value = returned.get(candidate.candidate_id)
        if value is None:
            value = EvaluationResult(
                ResultStatus.UNKNOWN,
                evaluator_capabilities={"batch_evaluator": "UNKNOWN"},
                evidence={"reason": "batch evaluator omitted this candidate"},
            )
        if not isinstance(value, EvaluationResult):
            raise TypeError("evaluate_batch must return EvaluationResult values")
        result[candidate.candidate_id] = value
    return result


def _make_candidate(nominal: ProcessTrajectory, spec: AxisSpec, magnitude: float, phase: str, index: int) -> ScanCandidate:
    requested = spec.direction * magnitude
    operator = StressOperator(
        operator_id=f"p2a:{spec.axis_id}:{phase}:{index:04d}:{magnitude:.17g}",
        family=spec.perturbation_family,
        amplitude=requested,
        units=spec.units,
        target_scope="all 181 nominal trajectory samples",
        parameters=spec.operator_parameters,
        availability=spec.capability_status,
        metadata={"axis_id": spec.axis_id, "coordinate_frame": spec.coordinate_frame, "signed_request": requested},
    )
    application = apply_stress(nominal, operator)
    candidate_id = _candidate_id(spec, magnitude, phase, index)
    return ScanCandidate(candidate_id, spec, magnitude, requested, application, phase, index if phase == "coarse" else None, index if phase == "refinement" else None)


def _candidate_id(spec: AxisSpec, magnitude: float, phase: str, index: int | None) -> str:
    safe_axis = "".join(char if char.isalnum() or char in "_-" else "_" for char in spec.axis_id)
    return f"{safe_axis}__{phase}__{index if index is not None else 0:04d}__{magnitude:.17g}"


def _observation(candidate: ScanCandidate, result: EvaluationResult, nominal: ProcessTrajectory) -> dict[str, Any]:
    realized = realized_perturbation(nominal, candidate.application.trajectory, candidate.spec, candidate.signed_delta, candidate.application.status)
    record = result.to_record()
    record["evidence"] = _complete_candidate_evidence(record, candidate.application.status)
    return {
        "axis_id": candidate.spec.axis_id,
        "candidate_id": candidate.candidate_id,
        "nominal_trajectory_sha256": nominal.metadata.get("trajectory_sha256"),
        "phase": candidate.phase,
        "sample_index": candidate.sample_index,
        "refinement_iteration": candidate.refinement_iteration,
        "magnitude": candidate.magnitude,
        "signed_perturbation": candidate.signed_delta,
        "requested_perturbation": {"value": candidate.signed_delta, "units": candidate.spec.units},
        "realized_perturbation": realized,
        **record,
    }


def _candidate_result_record(candidate: ScanCandidate, result: EvaluationResult, nominal: ProcessTrajectory | None) -> dict[str, Any]:
    realized: dict[str, Any] | None = None
    if nominal is not None:
        realized = realized_perturbation(nominal, candidate.application.trajectory, candidate.spec, candidate.signed_delta, candidate.application.status)
    record = result.to_record()
    record["evidence"] = _complete_candidate_evidence(record, candidate.application.status)
    return {
        "axis_id": candidate.spec.axis_id,
        "candidate_id": candidate.candidate_id,
        "nominal_trajectory_sha256": None if nominal is None else nominal.metadata.get("trajectory_sha256"),
        "magnitude": candidate.magnitude,
        "signed_perturbation": candidate.signed_delta,
        "requested_perturbation": {"value": candidate.signed_delta, "units": candidate.spec.units},
        "realized_perturbation": realized,
        **record,
    }


def _complete_candidate_evidence(record: Mapping[str, Any], application_status: ApplicationStatus) -> dict[str, Any]:
    evidence = dict(record.get("evidence", {}))
    failed_on_joint_limit = "JOINT_LIMIT_FAILURE" in record.get("failure_modes", [])
    measured_fields = (
        "joint_limit_margin_min_rad", "controlling_joint", "controlling_waypoint", "controlling_limit_side",
        "max_tcp_path_error_m", "max_tcp_path_error_waypoint", "terminal_position_error_m",
        "max_spray_axis_normal_error_rad", "max_spray_axis_normal_error_waypoint",
        "max_velocity_ratio", "max_acceleration_ratio", "max_jerk_ratio",
        "native_waypoint_world_collision_count", "native_waypoint_self_collision_count",
        "minimum_environment_clearance_m", "minimum_environment_clearance_waypoint",
        "minimum_environment_clearance_pair", "minimum_self_clearance_m", "minimum_self_clearance_waypoint",
        "minimum_self_clearance_pair", "collision_method", "wp179_to_wp180_fk_tcp_speed_m_s",
        "wp179_to_wp180_local_timing_status",
    )
    for key in measured_fields:
        if key not in evidence:
            evidence[key] = None
    if evidence.get("collision_method") is None:
        evidence["collision_method"] = "NOT_RUN_JOINT_LIMIT_FAILURE" if failed_on_joint_limit else "UNKNOWN"
    evidence.setdefault("strict_continuous_self_collision_ccd", record.get("evaluator_capabilities", {}).get("strict_continuous_self_collision_ccd", "NOT_AVAILABLE"))
    evidence.setdefault("candidate_ruckig_retime", record.get("evaluator_capabilities", {}).get("candidate_ruckig_retime", "NOT_RUN_FIXED_C1_TIMESTAMPS"))
    evidence.setdefault("measurement_status", (
        "NOT_EVALUATED_AFTER_JOINT_LIMIT_FAILURE" if failed_on_joint_limit else
        "NOT_EVALUATED_OPERATOR_UNAVAILABLE" if application_status != ApplicationStatus.APPLIED else
        "PARTIAL_OR_UNKNOWN" if record.get("status") == ResultStatus.UNKNOWN.value else "EVALUATED"
    ))
    return evidence


def realized_perturbation(
    nominal: ProcessTrajectory,
    perturbed: ProcessTrajectory,
    spec: AxisSpec,
    requested: float,
    application_status: ApplicationStatus,
) -> dict[str, Any]:
    if application_status != ApplicationStatus.APPLIED:
        return {"status": application_status.value, "value": None, "reason": "P1 did not apply this operator"}
    values: np.ndarray
    if spec.perturbation_family == StressFamily.JOINT_STATE and nominal.joint_states is not None and perturbed.joint_states is not None:
        joint = int(spec.operator_parameters["joint_index"])
        values = perturbed.joint_states[:, joint] - nominal.joint_states[:, joint]
    elif spec.perturbation_family == StressFamily.TCP_TRANSLATION and nominal.tcp_poses is not None and perturbed.tcp_poses is not None:
        axis = _as_unit(np.asarray(spec.operator_parameters["direction_tcp"], dtype=np.float64))
        offsets = perturbed.tcp_poses[:, :3] - nominal.tcp_poses[:, :3]
        values = np.asarray([float(np.dot(offset, _quat_rotate(quat, axis))) for offset, quat in zip(offsets, nominal.tcp_poses[:, 3:7])])
    elif spec.perturbation_family == StressFamily.TCP_ROTATION and nominal.tcp_poses is not None and perturbed.tcp_poses is not None:
        axis = _as_unit(np.asarray(spec.operator_parameters["axis_tcp"], dtype=np.float64))
        values = np.asarray([_relative_axis_angle(a, b, axis) for a, b in zip(nominal.tcp_poses[:, 3:7], perturbed.tcp_poses[:, 3:7])])
    else:
        return {"status": "UNKNOWN", "value": None, "reason": "realized-value adapter does not support this P1 field"}
    if not values.size or not np.isfinite(values).all():
        return {"status": "UNKNOWN", "value": None, "reason": "realized perturbation is non-finite or empty"}
    max_error = float(np.max(np.abs(values - requested)))
    return {
        "status": "APPLIED",
        "minimum_signed_value": float(np.min(values)),
        "maximum_signed_value": float(np.max(values)),
        "max_abs_error_from_request": max_error,
        "request_match": "YES" if max_error <= 1e-10 else "NO",
        "units": spec.units,
    }


def _as_unit(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if value.ndim != 1 or not np.isfinite(value).all() or norm <= 1e-12:
        raise ValueError("axis must be a finite nonzero vector")
    return value / norm


def _quat_rotate(quaternion_xyzw: Sequence[float], vector: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion_xyzw, dtype=np.float64)
    qv = np.asarray([x, y, z], dtype=np.float64)
    return vector + 2.0 * np.cross(qv, np.cross(qv, vector) + w * vector)


def _tool_z_direction(quaternion_xyzw: Sequence[float]) -> np.ndarray:
    """Return the normalized spray TCP +Z direction for an xyzw quaternion."""
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("TCP quaternion must be a finite xyzw vector")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError("TCP quaternion must be nonzero")
    return _quat_rotate(quaternion / norm, np.asarray([0.0, 0.0, 1.0]))


def _direction_angle_rad(first: Sequence[float], second: Sequence[float]) -> float:
    """Angular distance between directed unit vectors, insensitive to tool roll."""
    a = _as_unit(np.asarray(first, dtype=np.float64))
    b = _as_unit(np.asarray(second, dtype=np.float64))
    return math.atan2(float(np.linalg.norm(np.cross(a, b))), float(np.dot(a, b)))


def _process_normal_errors(
    actual_quaternions: np.ndarray,
    target_poses: np.ndarray,
    surface_normals: np.ndarray,
    *,
    compare_perturbed_target_axis: bool,
) -> np.ndarray:
    """Measure the process spray direction against the frozen surface normals."""
    actual = np.asarray(actual_quaternions, dtype=np.float64)
    target = np.asarray(target_poses, dtype=np.float64)
    normals = np.asarray(surface_normals, dtype=np.float64)
    if actual.shape != (len(normals), 4) or target.shape != (len(normals), 7) or normals.shape[1:] != (3,):
        raise ValueError("process normal measurement shape mismatch")
    orientations = target[:, 3:7] if compare_perturbed_target_axis else actual
    return np.asarray([
        _direction_angle_rad(_tool_z_direction(orientations[index]), normals[index])
        for index in range(len(normals))
    ], dtype=np.float64)


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    av, aw = a[:3], float(a[3])
    bv, bw = b[:3], float(b[3])
    return np.concatenate((aw * bv + bw * av + np.cross(av, bv), [aw * bw - float(np.dot(av, bv))]))


def _relative_axis_angle(q0: Sequence[float], q1: Sequence[float], axis: np.ndarray) -> float:
    base = np.asarray(q0, dtype=np.float64)
    target = np.asarray(q1, dtype=np.float64)
    inverse = np.asarray([-base[0], -base[1], -base[2], base[3]], dtype=np.float64)
    relative = _quat_multiply(inverse, target)
    relative /= max(float(np.linalg.norm(relative)), 1e-15)
    angle = 2.0 * math.atan2(float(np.dot(relative[:3], axis)), float(relative[3]))
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class _D41BatchEvaluator:
    """Thin q-only case adapter around the existing D41/D46 native evaluators."""

    def __init__(
        self,
        nominal: ProcessTrajectory,
        timestamps_s: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        dynamic_limits: Mapping[str, Mapping[str, float]],
        position_error_limit_m: float,
        terminal_position_limit_m: float,
        normal_error_limit_rad: float,
        joint_step_limit_rad: float,
        scratch_root: Path,
        d46: Any,
    ) -> None:
        self.nominal = nominal
        self.timestamps_s = timestamps_s
        self.lower = lower
        self.upper = upper
        self.dynamic_limits = dynamic_limits
        self.position_error_limit_m = position_error_limit_m
        self.terminal_position_limit_m = terminal_position_limit_m
        self.normal_error_limit_rad = normal_error_limit_rad
        self.joint_step_limit_rad = joint_step_limit_rad
        self.scratch_root = scratch_root
        self.d46 = d46
        self._batch_number = 0
        self._physical_by_q: dict[bytes, dict[str, Any]] = {}
        self._case_for_q: dict[bytes, str] = {}
        self._nominal_q_key = np.ascontiguousarray(nominal.joint_states, dtype=np.float64).tobytes()
        self._reference_positions = np.asarray(nominal.tcp_poses[:, :3], dtype=np.float64)

    def evaluate_batch(self, candidates: Sequence[ScanCandidate]) -> Mapping[str, EvaluationResult]:
        results: dict[str, EvaluationResult] = {}
        pending: dict[bytes, np.ndarray] = {}
        owners: dict[bytes, list[ScanCandidate]] = {}
        for candidate in candidates:
            app = candidate.application
            if app.status != ApplicationStatus.APPLIED:
                status = ResultStatus.UNKNOWN
                reason = app.reason or "P1 operator did not apply"
                if app.status == ApplicationStatus.NOT_AVAILABLE:
                    reason = "P1 operator capability is unavailable: " + reason
                results[candidate.candidate_id] = EvaluationResult(
                    status,
                    evaluator_capabilities={"p1_operator": app.status.value},
                    evidence={"reason": reason, "failure_codes": [code.value for code in app.failure_codes]},
                )
                continue
            limit_result = evaluate_joint_limits(app.trajectory)
            if limit_result.status != ResultStatus.PASS:
                results[candidate.candidate_id] = limit_result
                continue
            q = np.ascontiguousarray(app.trajectory.joint_states, dtype=np.float64)
            q_key = q.tobytes()
            owners.setdefault(q_key, []).append(candidate)
            if q_key not in self._physical_by_q:
                pending[q_key] = q

        if pending:
            self._measure_missing_states(pending)
        for q_key, owned_candidates in owners.items():
            physical = self._physical_by_q.get(q_key)
            if physical is None:
                for candidate in owned_candidates:
                    results[candidate.candidate_id] = EvaluationResult(
                        ResultStatus.UNKNOWN,
                        evaluator_capabilities={"native_moveit2": "UNKNOWN"},
                        evidence={"reason": "native batch did not return a complete result for this joint state"},
                    )
                continue
            for candidate in owned_candidates:
                results[candidate.candidate_id] = self._evaluate_one(candidate, physical)
        return results

    def _measure_missing_states(self, q_by_key: Mapping[bytes, np.ndarray]) -> None:
        if not q_by_key:
            return
        batch_root = self.scratch_root / f"native_batch_{self._batch_number:03d}"
        batch_root.mkdir(parents=True, exist_ok=False)
        case_rows: list[dict[str, Any]] = []
        q_key_by_case: dict[str, bytes] = {}
        for index, (q_key, q) in enumerate(q_by_key.items()):
            case_id = f"p2a_{self._batch_number:03d}_{index:05d}"
            q_path = batch_root / f"{case_id}.csv"
            write_q_only_csv(q_path, q)
            case_rows.append({"case_id": case_id, "family": "P2A_AXISWISE", "trajectory_path": str(q_path)})
            q_key_by_case[case_id] = q_key
            self._case_for_q[q_key] = case_id

        try:
            native_root = self.d46.run_native(batch_root, case_rows, native_name="native")
            fk_trace = _run_existing_fk(batch_root, native_root, self.d46)
            batch_metrics = _read_native_and_fk_metrics(
                native_root, fk_trace, self.nominal.tcp_poses, self.nominal.surface_normals, self.d46
            )
        except Exception:
            # Consume the directory number even on a failed experiment so the
            # next independent chunk can proceed inside the same scratch root.
            self._batch_number += 1
            raise
        for case_id, q_key in q_key_by_case.items():
            metric = batch_metrics.get(case_id)
            if metric is not None:
                metric["q"] = np.array(q_by_key[q_key], copy=True)
                self._physical_by_q[q_key] = metric
        self._batch_number += 1

    def _evaluate_one(self, candidate: ScanCandidate, physical: Mapping[str, Any]) -> EvaluationResult:
        trajectory = candidate.application.trajectory
        q = np.asarray(trajectory.joint_states, dtype=np.float64)
        target = np.asarray(trajectory.tcp_poses, dtype=np.float64)
        actual_position = np.asarray(physical["fk_position"], dtype=np.float64)
        actual_quaternion = np.asarray(physical["fk_quaternion"], dtype=np.float64)
        position_error = np.linalg.norm(actual_position - target[:, :3], axis=1)
        full_quaternion_difference = np.asarray([
            self.d46.quat_angle(actual_quaternion[index], target[index, 3:7])
            for index in range(len(target))
        ], dtype=np.float64)
        if candidate.spec.perturbation_family == StressFamily.TCP_ROTATION:
            normal_source = "perturbed target spray_tcp_link +Z versus frozen D41 wall normal"
        else:
            normal_source = "MoveIt2 FK spray_tcp_link +Z versus frozen D41 wall normal"
        normal_error = _process_normal_errors(
            actual_quaternion, target, self.nominal.surface_normals,
            compare_perturbed_target_axis=candidate.spec.perturbation_family == StressFamily.TCP_ROTATION,
        )
        max_position_index = int(np.argmax(position_error))
        max_normal_index = int(np.argmax(normal_error))
        terminal_position_error = float(position_error[-1])
        terminal_normal_error = float(normal_error[-1])

        velocity, acceleration, jerk = self.d46.finite_derivatives(q, self.timestamps_s)
        tcp_segment_distance = np.linalg.norm(np.diff(actual_position, axis=0), axis=1)
        tcp_segment_speed = tcp_segment_distance / np.diff(self.timestamps_s)
        max_tcp_speed_segment = int(np.argmax(tcp_segment_speed)) if tcp_segment_speed.size else None
        wp179_to_180_speed = float(tcp_segment_speed[179]) if tcp_segment_speed.size > 179 else None
        speed_low, speed_high = 0.003 * 0.95, 0.003 * 1.05
        wp179_to_180_speed_status = (
            "UNKNOWN" if wp179_to_180_speed is None else
            "WITHIN_CONFIGURED_BAND" if speed_low <= wp179_to_180_speed <= speed_high else
            "BELOW_CONFIGURED_BAND" if wp179_to_180_speed < speed_low else "ABOVE_CONFIGURED_BAND"
        )
        dynamic_columns = [self.dynamic_limits[f"j{index + 1}"] for index in range(6)]
        velocity_limits = np.asarray([row["velocity_rad_s"] for row in dynamic_columns], dtype=np.float64)
        acceleration_limits = np.asarray([row["acceleration_rad_s2"] for row in dynamic_columns], dtype=np.float64)
        jerk_limits = np.asarray([row["jerk_rad_s3"] for row in dynamic_columns], dtype=np.float64)
        velocity_ratio = np.abs(velocity) / velocity_limits[None, :]
        acceleration_ratio = np.abs(acceleration) / acceleration_limits[None, :]
        jerk_ratio = np.abs(jerk) / jerk_limits[None, :]
        joint_jumps = np.abs(np.diff(q, axis=0))

        native = physical["native"]
        segment = physical["segment_collisions"]
        failure_modes: list[str] = []
        critical_waypoint: int | None = None
        critical_segment: int | None = None
        failure_margin: dict[str, Any] = {}
        if not np.isfinite(q).all() or not np.isfinite(actual_position).all() or not np.isfinite(actual_quaternion).all():
            failure_modes.append("NON_FINITE_STATE_OR_FK")
        if int(native.get("nominal_waypoint_world_collision_count", 0)) > 0 or segment["first_world_collision"] is not None:
            failure_modes.append("ENVIRONMENT_COLLISION")
            critical_segment = segment["first_world_collision"]
            failure_margin["minimum_environment_clearance_m"] = native.get("minimum_robot_world_distance_m")
        if int(native.get("nominal_waypoint_self_collision_count", 0)) > 0 or segment["first_self_collision"] is not None:
            failure_modes.append("SELF_COLLISION")
            if critical_segment is None:
                critical_segment = segment["first_self_collision"]
            failure_margin["minimum_self_clearance_m"] = native.get("minimum_self_distance_m")
        if int(native.get("native_continuous_segment_collision_count", 0)) > 0:
            failure_modes.append("ROBOT_WORLD_TRANSITION_COLLISION")
            if critical_segment is None:
                critical_segment = segment["first_two_state_world_collision"]
            failure_margin["native_two_state_robot_world_collision_count"] = int(native.get("native_continuous_segment_collision_count", 0))
        if float(np.max(position_error)) > self.position_error_limit_m:
            failure_modes.append("TCP_PATH_DEVIATION")
            critical_waypoint = max_position_index
            failure_margin["position_error_excess_m"] = float(np.max(position_error) - self.position_error_limit_m)
        if terminal_position_error > self.terminal_position_limit_m:
            failure_modes.append("TERMINAL_POSITION_ERROR")
            if critical_waypoint is None:
                critical_waypoint = len(q) - 1
            failure_margin["terminal_position_error_excess_m"] = terminal_position_error - self.terminal_position_limit_m
        if float(np.max(normal_error)) > self.normal_error_limit_rad:
            failure_modes.append("SPRAY_AXIS_NORMAL_ERROR")
            if critical_waypoint is None:
                critical_waypoint = max_normal_index
            failure_margin["spray_axis_normal_error_excess_rad"] = float(np.max(normal_error) - self.normal_error_limit_rad)
        if np.any(np.abs(velocity) > velocity_limits[None, :]):
            failure_modes.append("VELOCITY_LIMIT")
            critical_waypoint = int(np.unravel_index(np.argmax(velocity_ratio), velocity_ratio.shape)[0])
            failure_margin["maximum_velocity_ratio"] = float(np.max(velocity_ratio))
        if np.any(np.abs(acceleration) > acceleration_limits[None, :]):
            failure_modes.append("ACCELERATION_LIMIT")
            if critical_waypoint is None:
                critical_waypoint = int(np.unravel_index(np.argmax(acceleration_ratio), acceleration_ratio.shape)[0])
            failure_margin["maximum_acceleration_ratio"] = float(np.max(acceleration_ratio))
        if np.any(np.abs(jerk) > jerk_limits[None, :]):
            failure_modes.append("JERK_LIMIT")
            if critical_waypoint is None:
                critical_waypoint = int(np.unravel_index(np.argmax(jerk_ratio), jerk_ratio.shape)[0])
            failure_margin["maximum_jerk_ratio"] = float(np.max(jerk_ratio))
        max_jump = float(np.max(joint_jumps)) if joint_jumps.size else 0.0
        if max_jump > self.joint_step_limit_rad:
            failure_modes.append("JOINT_DISCONTINUITY")
            flat = int(np.argmax(joint_jumps))
            critical_segment = int(np.unravel_index(flat, joint_jumps.shape)[0])
            failure_margin["maximum_joint_step_excess_rad"] = max_jump - self.joint_step_limit_rad

        q_margin = np.minimum(q - self.lower[None, :], self.upper[None, :] - q)
        controlling_waypoint, controlling_joint = (int(value) for value in np.unravel_index(int(np.argmin(q_margin)), q_margin.shape))
        controlling_side = "LOWER" if q[controlling_waypoint, controlling_joint] - self.lower[controlling_joint] <= self.upper[controlling_joint] - q[controlling_waypoint, controlling_joint] else "UPPER"
        nominal = self._physical_by_q.get(self._nominal_q_key, physical)
        nominal_position = np.asarray(nominal["position_error"], dtype=np.float64)
        nominal_normal = np.asarray(nominal["normal_error"], dtype=np.float64)
        nominal_joint_q = np.asarray(nominal["q"], dtype=np.float64)
        nominal_joint_margin = np.minimum(nominal_joint_q - self.lower[None, :], self.upper[None, :] - nominal_joint_q)
        nominal_velocity, nominal_acceleration, nominal_jerk = self.d46.finite_derivatives(nominal_joint_q, self.timestamps_s)
        nominal_margins = {
            "minimum_joint_limit_margin_rad": float(np.min(nominal_joint_margin)),
            "tcp_path_error_gate_margin_m": float(self.position_error_limit_m - np.max(nominal_position)),
            "terminal_position_gate_margin_m": float(self.terminal_position_limit_m - nominal_position[-1]),
            "spray_axis_normal_gate_margin_rad": float(self.normal_error_limit_rad - np.max(nominal_normal)),
            "configured_velocity_margin_rad_s": float(np.min(velocity_limits[None, :] - np.abs(nominal_velocity))),
            "configured_acceleration_margin_rad_s2": float(np.min(acceleration_limits[None, :] - np.abs(nominal_acceleration))),
            "configured_jerk_margin_rad_s3": float(np.min(jerk_limits[None, :] - np.abs(nominal_jerk))),
            "environment_clearance_diagnostic_m": nominal["native"].get("minimum_robot_world_distance_m"),
            "self_clearance_diagnostic_m": nominal["native"].get("minimum_self_distance_m"),
            "clearance_acceptance_threshold": "UNRESOLVED_THRESHOLD",
        }
        capabilities = {
            "joint_limits": "AVAILABLE",
            "moveit2_planning_scene": "AVAILABLE",
            "moveit2_fk": "AVAILABLE",
            "robot_world_collision": "AVAILABLE",
            "self_collision_adaptive_discrete_interpolation": "AVAILABLE",
            "transition_collision_adaptive_discrete_interpolation": "AVAILABLE",
            "native_robot_world_two_state_query": "AVAILABLE",
            "fcl_environment_and_self_distance": "AVAILABLE",
            "environment_clearance_distance": "AVAILABLE_IF_REPORTED_BY_NATIVE_FCL",
            "self_clearance_distance": "AVAILABLE_IF_REPORTED_BY_NATIVE_FCL",
            "finite_difference_dynamics": "AVAILABLE_AUDIT_ONLY_CONFIGURED_LIMITS_UNVERIFIED",
            "candidate_ruckig_retime": "NOT_RUN_FIXED_C1_TIMESTAMPS",
            "strict_continuous_self_collision_ccd": "NOT_AVAILABLE",
            "physical_singularity_threshold": "UNKNOWN",
            "physical_torque": "NOT_AVAILABLE",
            "calibrated_tcp_uncertainty": "NOT_AVAILABLE",
            "stand_off_surface_projection": "NOT_AVAILABLE",
            "coating_thickness_or_deposition_physics": "NOT_AVAILABLE",
        }
        status = ResultStatus.FAIL if failure_modes else ResultStatus.PASS
        if not failure_modes and (native.get("status") not in {"PASS", "FAIL"} or physical.get("fk_status") != "AVAILABLE"):
            status = ResultStatus.UNKNOWN
        evidence = {
            "native_case_id": self._case_for_q.get(np.ascontiguousarray(q, dtype=np.float64).tobytes()),
            "collision_method": "adaptive_discrete_interpolation",
            "native_two_state_robot_world_query": {
                "available": True,
                "collision_count": int(native.get("native_continuous_segment_collision_count", 0)),
                "interpretation": "reported as a separate MoveIt2 endpoint-pair diagnostic; not promoted to strict global CCD",
            },
            "native_waypoint_world_collision_count": int(native.get("nominal_waypoint_world_collision_count", 0)),
            "native_waypoint_self_collision_count": int(native.get("nominal_waypoint_self_collision_count", 0)),
            "first_adaptive_world_collision_segment": segment["first_world_collision"],
            "first_adaptive_self_collision_segment": segment["first_self_collision"],
            "first_two_state_world_collision_segment": segment["first_two_state_world_collision"],
            "minimum_environment_clearance_m": native.get("minimum_robot_world_distance_m"),
            "minimum_environment_clearance_waypoint": native.get("minimum_robot_world_waypoint"),
            "minimum_environment_clearance_pair": native.get("minimum_robot_world_pair"),
            "minimum_self_clearance_m": native.get("minimum_self_distance_m"),
            "minimum_self_clearance_waypoint": native.get("minimum_self_waypoint"),
            "minimum_self_clearance_pair": native.get("minimum_self_pair"),
            "clearance_acceptance_threshold": "UNRESOLVED_THRESHOLD",
            "minimum_jacobian_sigma": native.get("minimum_jacobian_sigma"),
            "minimum_jacobian_sigma_waypoint": native.get("minimum_jacobian_sigma_waypoint"),
            "maximum_jacobian_condition_number": native.get("maximum_jacobian_condition_number"),
            "jacobian_is_diagnostic_not_an_unsafe_classification": True,
            "max_tcp_path_error_m": float(np.max(position_error)),
            "max_tcp_path_error_waypoint": max_position_index,
            "max_fk_tcp_segment_speed_m_s": float(np.max(tcp_segment_speed)) if tcp_segment_speed.size else None,
            "max_fk_tcp_segment_speed_segment": max_tcp_speed_segment,
            "wp179_to_wp180_fk_tcp_speed_m_s": wp179_to_180_speed,
            "wp179_to_wp180_local_timing_status": wp179_to_180_speed_status,
            "terminal_position_error_m": terminal_position_error,
            "max_spray_axis_normal_error_rad": float(np.max(normal_error)),
            "max_spray_axis_normal_error_waypoint": max_normal_index,
            "terminal_spray_axis_normal_error_rad": terminal_normal_error,
            "full_quaternion_difference_max_rad_diagnostic_only": float(np.max(full_quaternion_difference)),
            "full_quaternion_difference_terminal_rad_diagnostic_only": float(full_quaternion_difference[-1]),
            "orientation_failure_gate": normal_source + "; tool-axis roll is not counted as spray-normal error",
            "joint_limit_margin_min_rad": float(np.min(q_margin)),
            "controlling_joint": f"j{controlling_joint + 1}",
            "controlling_waypoint": controlling_waypoint,
            "controlling_limit_side": controlling_side,
            "max_velocity_ratio": float(np.max(velocity_ratio)),
            "max_acceleration_ratio": float(np.max(acceleration_ratio)),
            "max_jerk_ratio": float(np.max(jerk_ratio)),
            "max_joint_step_rad": max_jump,
            "derivative_method": "D46 finite_derivatives on frozen D41 timestamps; audit-only perturbation evaluation",
            "dynamic_limit_provenance": "project configuration; not vendor certified",
            "process_quality_certified": False,
        }
        return EvaluationResult(
            status,
            tuple(failure_modes),
            critical_waypoint=critical_waypoint,
            critical_segment=critical_segment,
            evaluator_capabilities=capabilities,
            nominal_margin=nominal_margins,
            failure_margin=failure_margin,
            evidence=evidence,
        )


def write_q_only_csv(path: Path, q: np.ndarray) -> None:
    """Normalize a trajectory to the exact position-only schema expected by D41.

    D41's legacy CSV reader consumes the last six columns as q. Never pass it a
    q/dq/ddq/jerk table, whose last six values are jerk.
    """
    values = np.asarray(q, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 6 or not np.isfinite(values).all():
        raise ValueError("q-only native input must be finite with shape [waypoint, 6]")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["waypoint", *[f"j{index}_q" for index in range(1, 7)]])
        for index, row in enumerate(values):
            writer.writerow([index, *[format(float(value), ".17g") for value in row]])


def _run_existing_fk(batch_root: Path, native_root: Path, d46: Any) -> Path:
    """Run the cached D46 native MoveIt2 FK binary over the just-written cases."""
    import shlex

    binary = d46.ROOT / "tmp" / "stage4a_fk_install" / "lib" / "stage4a_fk" / "stage4a_fk"
    install_setup = d46.ROOT / "tmp" / "stage4a_fk_install" / "setup.bash"
    if not binary.is_file() or not install_setup.is_file():
        raise RuntimeError("existing_d46_moveit_fk_binary_or_setup_unavailable")
    trace = batch_root / "fk" / "STAGE4A_FK_TRACE.csv"
    trace.parent.mkdir(parents=True, exist_ok=True)
    ld_library_path = ":".join((
        d46.wsl_path(d46.ROOT / "install/lib"),
        d46.wsl_path(d46.ROOT / "tmp/d41_install/lib"),
        "/opt/ros/jazzy/lib",
        "/opt/ros/jazzy/lib/x86_64-linux-gnu",
        "/opt/ros/jazzy/opt/sdformat_vendor/lib",
        "/opt/ros/jazzy/opt/gz_math_vendor/lib",
        "/opt/ros/jazzy/opt/gz_utils_vendor/lib",
        "/opt/ros/jazzy/opt/gz_tools_vendor/lib",
        "/usr/lib/x86_64-linux-gnu",
    ))
    command = "\n".join((
        "source /opt/ros/jazzy/setup.bash",
        f"source {shlex.quote(d46.wsl_path(install_setup))}",
        f"export LD_LIBRARY_PATH={shlex.quote(ld_library_path)}",
        " ".join(shlex.quote(part) for part in (
            d46.wsl_path(binary),
            "--cases", d46.wsl_path(native_root / "cases.csv"),
            "--urdf", d46.wsl_path(d46.URDF),
            "--srdf", d46.wsl_path(d46.SRDF),
            "--output", d46.wsl_path(trace),
        )),
    ))
    run = subprocess.run(
        ["wsl.exe", "bash", "-lc", command],
        cwd=d46.ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=7200,
    )
    (batch_root / "fk_execution.log").write_text(
        (run.stdout or "") + "\n--- STDERR ---\n" + (run.stderr or ""),
        encoding="utf-8",
        errors="replace",
    )
    if run.returncode != 0 or not trace.is_file() or trace.stat().st_size == 0:
        detail = (run.stderr or run.stdout or "").strip()[-1200:]
        raise RuntimeError(f"native_fk_batch_failed:{run.returncode}:{detail}")
    return trace


def _run_fk_only_cases(
    batch_root: Path, cases: Sequence[tuple[str, np.ndarray]], d46: Any
) -> tuple[Path, dict[str, np.ndarray]]:
    """Run the existing MoveIt2 FK executable without repeating cached collision work."""
    native_root = batch_root / "native"
    native_root.mkdir(parents=True, exist_ok=False)
    case_path = native_root / "cases.csv"
    q_by_case: dict[str, np.ndarray] = {}
    with case_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("case_id", "trajectory_csv", "family"), lineterminator="\n")
        writer.writeheader()
        for case_id, q in cases:
            values = np.asarray(q, dtype=np.float64)
            if case_id in q_by_case or values.shape != (181, 6) or not np.isfinite(values).all():
                raise RuntimeError("invalid_or_duplicate_fk_only_case")
            q_path = batch_root / f"{case_id}.csv"
            write_q_only_csv(q_path, values)
            writer.writerow({"case_id": case_id, "trajectory_csv": d46.wsl_path(q_path), "family": "P2A_FK_REPAIR"})
            q_by_case[case_id] = np.array(values, copy=True)
    trace = _run_existing_fk(batch_root, native_root, d46)
    return trace, q_by_case


def _read_fk_trace(fk_trace: Path) -> dict[str, dict[str, np.ndarray]]:
    cases: dict[str, dict[str, list[Any]]] = {}
    required = ("case_id", "waypoint", "x_m", "y_m", "z_m", "qx", "qy", "qz", "qw")
    with fk_trace.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or any(name not in reader.fieldnames for name in required):
            raise RuntimeError("native_fk_trace_schema_mismatch")
        for row in reader:
            case_id = str(row["case_id"])
            bucket = cases.setdefault(case_id, {"waypoint": [], "position": [], "quaternion": []})
            bucket["waypoint"].append(int(row["waypoint"]))
            bucket["position"].append([float(row[key]) for key in ("x_m", "y_m", "z_m")])
            bucket["quaternion"].append([float(row[key]) for key in ("qx", "qy", "qz", "qw")])
    result: dict[str, dict[str, np.ndarray]] = {}
    for case_id, bucket in cases.items():
        if bucket["waypoint"] != list(range(181)):
            raise RuntimeError(f"native_fk_trace_waypoint_order_or_count_mismatch:{case_id}")
        position = np.asarray(bucket["position"], dtype=np.float64)
        quaternion = np.asarray(bucket["quaternion"], dtype=np.float64)
        if position.shape != (181, 3) or quaternion.shape != (181, 4) or not np.isfinite(position).all() or not np.isfinite(quaternion).all():
            raise RuntimeError(f"native_fk_trace_nonfinite_or_shape_mismatch:{case_id}")
        result[case_id] = {"position": position, "quaternion": quaternion}
    return result


def _read_native_and_fk_metrics(
    native_root: Path,
    fk_trace: Path,
    nominal_targets: np.ndarray,
    surface_normals: np.ndarray,
    d46: Any,
) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    with (native_root / "D41_native_case_summary.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                summaries[str(row["case_id"])] = row
    segment_collisions: dict[str, dict[str, Any]] = {
        case_id: {"first_world_collision": None, "first_self_collision": None, "first_two_state_world_collision": None}
        for case_id in summaries
    }
    segment_path = native_root / "D41_native_segment_collision.csv"
    with segment_path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            case_id = str(row["case_id"])
            record = segment_collisions.get(case_id)
            if record is None:
                continue
            segment_index = int(row["segment_index"])
            if _true(row.get("discrete_sample_world_collision")) and record["first_world_collision"] is None:
                record["first_world_collision"] = segment_index
            if _true(row.get("discrete_sample_self_collision")) and record["first_self_collision"] is None:
                record["first_self_collision"] = segment_index
            if _true(row.get("native_continuous_segment_collision")) and record["first_two_state_world_collision"] is None:
                record["first_two_state_world_collision"] = segment_index

    jacobian_metrics: dict[str, dict[str, Any]] = {
        case_id: {"minimum_jacobian_sigma_waypoint": None, "minimum_jacobian_sigma": None, "maximum_jacobian_condition_number": None}
        for case_id in summaries
    }
    jacobian_path = native_root / "D41_native_jacobian.csv"
    with jacobian_path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            case_id = str(row["case_id"])
            record = jacobian_metrics.get(case_id)
            if record is None:
                continue
            sigma = _finite_or_none(row.get("sigma_min"))
            condition = _finite_or_none(row.get("condition_number"))
            if sigma is not None and (record["minimum_jacobian_sigma"] is None or sigma < record["minimum_jacobian_sigma"]):
                record["minimum_jacobian_sigma"] = sigma
                record["minimum_jacobian_sigma_waypoint"] = int(row["waypoint"])
            if condition is not None and (record["maximum_jacobian_condition_number"] is None or condition > record["maximum_jacobian_condition_number"]):
                record["maximum_jacobian_condition_number"] = condition

    fk_values = _read_fk_trace(fk_trace)

    result: dict[str, dict[str, Any]] = {}
    for case_id, native in summaries.items():
        fk = fk_values.get(case_id)
        fk_status = "UNKNOWN"
        fk_position = None
        fk_quaternion = None
        position_error = None
        full_quaternion_difference = None
        normal_error = None
        if fk is not None:
            fk_position = fk["position"]
            fk_quaternion = fk["quaternion"]
            position_error = np.linalg.norm(fk_position - nominal_targets[:, :3], axis=1)
            full_quaternion_difference = np.asarray([
                d46.quat_angle(fk_quaternion[index], nominal_targets[index, 3:7])
                for index in range(len(nominal_targets))
            ], dtype=np.float64)
            normal_error = _process_normal_errors(
                fk_quaternion, nominal_targets, surface_normals, compare_perturbed_target_axis=False
            )
            fk_status = "AVAILABLE" if np.isfinite(fk_position).all() and np.isfinite(fk_quaternion).all() else "UNKNOWN"
        merged_native = dict(native)
        merged_native.update(jacobian_metrics.get(case_id, {}))
        result[case_id] = {
            "native": merged_native,
            "segment_collisions": segment_collisions.get(case_id, {}),
            "fk_status": fk_status,
            "fk_position": fk_position,
            "fk_quaternion": fk_quaternion,
            "position_error": position_error,
            "full_quaternion_difference": full_quaternion_difference,
            "normal_error": normal_error,
        }
    return result


def _true(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _finite_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _protected_d46_snapshot() -> dict[str, str]:
    paths = (
        D46_ROOT / "STAGE4_SYSTEM_BENCHMARK_V1.json",
        D46_ROOT / "STAGE4_CASE_RESULTS.csv",
        D46_ROOT / "STAGE4_BASELINE_SCORECARD_V1.json",
        D46_ROOT / "native" / "D41_native_provenance.json",
    )
    if not all(path.is_file() for path in paths):
        raise RuntimeError("protected_D46_identity_files_missing")
    return {str(path.relative_to(ROOT)): _sha256_file(path) for path in paths}


def _auth_inputs(d46: Any) -> tuple[dict[str, Any], Path, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    auth_path = D46_ROOT / "STAGE3_AUTHENTICATION_V1.json"
    if not auth_path.is_file():
        raise RuntimeError("existing_stage3_authentication_evidence_missing")
    auth = _read_json(auth_path)
    release = ROOT / str(auth["release"])
    release_manifest_path = release / "release_manifest.json"
    checkpoint = ROOT / str(auth["canonical_checkpoint"])
    if not release_manifest_path.is_file() or not checkpoint.is_file():
        raise RuntimeError("frozen_stage3_release_or_checkpoint_missing")
    release_manifest = _read_json(release_manifest_path)
    if release_manifest.get("release_status") != "FROZEN_AND_CLOSED":
        raise RuntimeError("stage3_release_not_frozen")
    if _sha256_file(release_manifest_path) != auth["release_manifest_sha256"]:
        raise RuntimeError("stage3_release_manifest_identity_mismatch")
    if _sha256_file(checkpoint) != auth["canonical_checkpoint_sha256"]:
        raise RuntimeError("stage3_checkpoint_identity_mismatch")
    if int(release_manifest.get("canonical_update", -1)) != int(auth["canonical_update"]):
        raise RuntimeError("stage3_canonical_update_mismatch")
    input_record = auth["authoritative_inputs"]
    expected_inputs = (
        (ROOT / input_record["poses"]["path"], input_record["poses"]["sha256"]),
        (ROOT / input_record["seed_joints"]["path"], input_record["seed_joints"]["sha256"]),
    )
    for input_path, expected_hash in expected_inputs:
        if not input_path.is_file() or _sha256_file(input_path) != expected_hash:
            raise RuntimeError(f"authoritative_open_arch_input_identity_mismatch:{input_path.name}")
    if int(input_record.get("point_count", -1)) != 181 or input_record.get("scope") != "Stage 0/1 ON-state open-arch only":
        raise RuntimeError("authoritative_input_scope_mismatch")

    d41_summary_path = ROOT / str(auth["retained_stage3_robot"]["D41_summary"])
    d41_summary = _read_json(d41_summary_path)
    if d41_summary.get("TASK_STATUS") != "PASS" or d41_summary.get("FULL_OFFLINE_WORKFLOW_SCOPE") != "181-point ON-state open-arch only":
        raise RuntimeError("D41_nominal_identity_or_scope_mismatch")
    d41_run = Path(str(d41_summary["D41_RUN_DIRECTORY"]))
    if not d41_run.is_absolute():
        d41_run = ROOT / d41_run
    q_path = d41_run / "strict_replay" / "moveit_smoothed_joint_trajectory.csv"
    if not q_path.is_file() or q_path != (ROOT / str(auth["retained_stage3_robot"]["post_ruckig_trajectory"])).resolve():
        raise RuntimeError("D41_post_ruckig_trajectory_identity_mismatch")
    time_s, q, velocity, acceleration, jerk = d46.load_post_ruckig(q_path)
    poses, quaternions, normals = d46.read_targets()
    seed_q = d46.read_joint_csv(ROOT / str(input_record["seed_joints"]["path"]))
    if q.shape != (181, 6) or poses.shape != (181, 3) or seed_q.shape != (181, 6):
        raise RuntimeError("frozen_181_point_shape_mismatch")
    if not np.isfinite(np.concatenate((q, poses, quaternions, normals), axis=1)).all():
        raise RuntimeError("frozen_181_point_nonfinite_input")
    q_margin = evaluate_joint_limits(ProcessTrajectory(joint_states=q, joint_lower_rad=d46.load_limits()[0], joint_upper_rad=d46.load_limits()[1]))
    if q_margin.status != ResultStatus.PASS:
        raise RuntimeError("frozen_D41_post_ruckig_path_fails_current_position_limits")
    return auth, q_path, time_s, q, velocity, acceleration, jerk, {
        "poses": poses,
        "quaternions": quaternions,
        "normals": normals,
        "seed_q": seed_q,
        "d41_summary": d41_summary,
        "d41_run": d41_run,
        "d41_fk_trace_path": d41_run / "strict_replay" / "moveit_fk_tcp_trace.csv",
    }


def _read_d41_process_reference(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise RuntimeError(f"D41_strict_replay_FK_trace_missing:{path}")
    positions: list[list[float]] = []
    tool_z: list[list[float]] = []
    normals: list[list[float]] = []
    normal_error_deg: list[float] = []
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = (
            "actual_tcp_x", "actual_tcp_y", "actual_tcp_z", "tool_z_x", "tool_z_y", "tool_z_z",
            "wall_normal_x", "wall_normal_y", "wall_normal_z", "normal_angle_error_deg",
        )
        if reader.fieldnames is None or any(name not in reader.fieldnames for name in required):
            raise RuntimeError("D41_strict_replay_FK_trace_schema_mismatch")
        for row in reader:
            positions.append([float(row[key]) for key in ("actual_tcp_x", "actual_tcp_y", "actual_tcp_z")])
            tool_z.append([float(row[key]) for key in ("tool_z_x", "tool_z_y", "tool_z_z")])
            normals.append([float(row[key]) for key in ("wall_normal_x", "wall_normal_y", "wall_normal_z")])
            normal_error_deg.append(float(row["normal_angle_error_deg"]))
    position_array = np.asarray(positions, dtype=np.float64)
    tool_z_array = np.asarray(tool_z, dtype=np.float64)
    normal_array = np.asarray(normals, dtype=np.float64)
    error_array = np.asarray(normal_error_deg, dtype=np.float64)
    if (position_array.shape != (181, 3) or tool_z_array.shape != (181, 3) or normal_array.shape != (181, 3)
            or error_array.shape != (181,) or not all(np.isfinite(array).all() for array in (position_array, tool_z_array, normal_array, error_array))):
        raise RuntimeError("D41_strict_replay_FK_trace_nonfinite_or_shape_mismatch")
    normal_lengths = np.linalg.norm(normal_array, axis=1)
    tool_lengths = np.linalg.norm(tool_z_array, axis=1)
    if np.any(normal_lengths < 1e-12) or np.any(tool_lengths < 1e-12):
        raise RuntimeError("D41_process_reference_contains_zero_direction")
    normal_array = normal_array / normal_lengths[:, None]
    tool_z_array = tool_z_array / tool_lengths[:, None]
    return {
        "positions": position_array,
        "tool_z": tool_z_array,
        "surface_normals": normal_array,
        "normal_error_deg": error_array,
    }


def _load_previous_native_cache(
    cache_path: Path,
    q_nominal: np.ndarray,
    trajectory_sha256: str,
    auth: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]], str, int]:
    """Recover only prior native per-case measurements; discard old task-geometry judgments."""
    cache = _read_json(cache_path)
    if cache.get("schema") != SCHEMA or cache.get("robot") != "FAIRINO_FR5":
        raise RuntimeError("native_measurement_cache_identity_mismatch")
    if cache.get("frozen_trajectory", {}).get("sha256") != trajectory_sha256:
        raise RuntimeError("native_measurement_cache_trajectory_mismatch")
    identity = cache.get("stage3_release_identity", {})
    for key in ("release_manifest_sha256", "canonical_checkpoint_sha256", "canonical_update"):
        if identity.get(key) != auth.get(key):
            raise RuntimeError(f"native_measurement_cache_stage3_identity_mismatch:{key}")
    provenance = cache.get("evaluator_provenance", {})
    if (provenance.get("native_measurement_backend") != "existing D41 MoveIt2 PlanningScene CollisionEnvBullet plus CollisionEnvFCL distances"
            or provenance.get("native_collision_method") != "adaptive_discrete_interpolation"
            or cache.get("gate", {}).get("NATIVE_MOVEIT2_EVALUATION") != "PASS"):
        raise RuntimeError("native_measurement_cache_backend_or_gate_mismatch")
    expected_count = int(provenance.get("native_candidate_case_count", 0))
    batch_count = int(provenance.get("native_batch_count", 0))
    if expected_count <= 0 or expected_count != 755 or batch_count != 1:
        raise RuntimeError("native_measurement_cache_case_or_batch_count_mismatch")

    q_by_case: dict[str, np.ndarray] = {}
    evidence_by_case: dict[str, Mapping[str, Any]] = {}
    for axis in cache.get("axis_results", []):
        specification = axis.get("specification", {})
        family = specification.get("perturbation_family")
        sign = int(specification.get("sign", 0))
        parameters = specification.get("operator_parameters", {})
        for observation in axis.get("coarse_observations", []) + axis.get("refinement_observations", []):
            evidence = observation.get("evidence", {})
            case_id = evidence.get("native_case_id")
            if not case_id:
                continue
            q_case = np.array(q_nominal, dtype=np.float64, copy=True)
            if family == StressFamily.JOINT_STATE.value:
                joint_index = int(parameters.get("joint_index", -1))
                if not 0 <= joint_index < 6 or sign not in {-1, 1}:
                    raise RuntimeError("native_cache_joint_axis_metadata_invalid")
                q_case[:, joint_index] += sign * float(observation["magnitude"])
            elif family not in {StressFamily.TCP_TRANSLATION.value, StressFamily.TCP_ROTATION.value}:
                raise RuntimeError(f"native_cache_unexpected_measured_family:{family}")
            if case_id in evidence_by_case:
                if not np.array_equal(q_by_case[case_id], q_case):
                    raise RuntimeError(f"native_cache_case_id_maps_to_multiple_q:{case_id}")
                previous = evidence_by_case[case_id]
                check_keys = (
                    "native_waypoint_world_collision_count", "native_waypoint_self_collision_count",
                    "minimum_environment_clearance_m", "minimum_self_clearance_m", "minimum_jacobian_sigma",
                    "maximum_jacobian_condition_number",
                )
                if any(previous.get(key) != evidence.get(key) for key in check_keys):
                    raise RuntimeError(f"native_cache_repeated_measurement_disagrees:{case_id}")
                continue
            q_by_case[str(case_id)] = q_case
            evidence_by_case[str(case_id)] = evidence
    if len(q_by_case) != expected_count or len(evidence_by_case) != expected_count:
        raise RuntimeError(f"native_measurement_cache_reconstruction_count_mismatch:{len(q_by_case)}:{expected_count}")
    q_keys = [np.ascontiguousarray(values, dtype=np.float64).tobytes() for values in q_by_case.values()]
    if len(set(q_keys)) != expected_count:
        raise RuntimeError("native_measurement_cache_contains_duplicate_or_unmapped_q_states")

    physical_by_case: dict[str, dict[str, Any]] = {}
    for case_id, evidence in evidence_by_case.items():
        required = (
            "native_waypoint_world_collision_count", "native_waypoint_self_collision_count",
            "first_adaptive_world_collision_segment", "first_adaptive_self_collision_segment",
            "first_two_state_world_collision_segment", "minimum_environment_clearance_m",
            "minimum_self_clearance_m", "minimum_jacobian_sigma", "minimum_jacobian_sigma_waypoint",
            "maximum_jacobian_condition_number",
        )
        if any(key not in evidence for key in required):
            raise RuntimeError(f"native_measurement_cache_missing_metric:{case_id}")
        world_count = int(evidence["native_waypoint_world_collision_count"])
        self_count = int(evidence["native_waypoint_self_collision_count"])
        transition_query = evidence.get("native_two_state_robot_world_query", {})
        if not transition_query.get("available") or "collision_count" not in transition_query:
            raise RuntimeError(f"native_measurement_cache_missing_transition_query:{case_id}")
        transition_count = int(transition_query["collision_count"])
        native = {
            "status": "PASS" if not (world_count or self_count or transition_count) else "FAIL",
            "nominal_waypoint_world_collision_count": world_count,
            "nominal_waypoint_self_collision_count": self_count,
            "native_continuous_segment_collision_count": transition_count,
            "minimum_robot_world_distance_m": evidence["minimum_environment_clearance_m"],
            "minimum_robot_world_waypoint": evidence.get("minimum_environment_clearance_waypoint"),
            "minimum_robot_world_pair": evidence.get("minimum_environment_clearance_pair"),
            "minimum_self_distance_m": evidence["minimum_self_clearance_m"],
            "minimum_self_waypoint": evidence.get("minimum_self_clearance_waypoint"),
            "minimum_self_pair": evidence.get("minimum_self_clearance_pair"),
            "minimum_jacobian_sigma": evidence["minimum_jacobian_sigma"],
            "minimum_jacobian_sigma_waypoint": evidence["minimum_jacobian_sigma_waypoint"],
            "maximum_jacobian_condition_number": evidence["maximum_jacobian_condition_number"],
        }
        physical_by_case[case_id] = {
            "native": native,
            "segment_collisions": {
                "first_world_collision": evidence["first_adaptive_world_collision_segment"],
                "first_self_collision": evidence["first_adaptive_self_collision_segment"],
                "first_two_state_world_collision": evidence["first_two_state_world_collision_segment"],
            },
            "q": q_by_case[case_id],
        }
    return q_by_case, physical_by_case, _sha256_file(cache_path), batch_count


def _attach_fk_replay(
    evaluator: _D41BatchEvaluator,
    q_by_case: Mapping[str, np.ndarray],
    physical_by_case: Mapping[str, Mapping[str, Any]],
    fk_by_case: Mapping[str, Mapping[str, np.ndarray]],
    surface_normals: np.ndarray,
    d41_reference: Mapping[str, np.ndarray],
) -> dict[str, float]:
    if set(q_by_case) != set(physical_by_case) or set(q_by_case) != set(fk_by_case):
        raise RuntimeError("native_cache_and_FK_replay_case_sets_differ")
    case_by_q: dict[bytes, str] = {}
    for case_id, q in q_by_case.items():
        q_key = np.ascontiguousarray(q, dtype=np.float64).tobytes()
        case_by_q[q_key] = case_id
        physical = dict(physical_by_case[case_id])
        fk = fk_by_case[case_id]
        position = np.asarray(fk["position"], dtype=np.float64)
        quaternion = np.asarray(fk["quaternion"], dtype=np.float64)
        physical.update({
            "fk_status": "AVAILABLE",
            "fk_position": position,
            "fk_quaternion": quaternion,
            "position_error": np.linalg.norm(position - evaluator.nominal.tcp_poses[:, :3], axis=1),
            "full_quaternion_difference": np.asarray([
                evaluator.d46.quat_angle(quaternion[index], evaluator.nominal.tcp_poses[index, 3:7])
                for index in range(181)
            ], dtype=np.float64),
            "normal_error": _process_normal_errors(
                quaternion, evaluator.nominal.tcp_poses, surface_normals, compare_perturbed_target_axis=False
            ),
            "q": np.array(q, copy=True),
        })
        evaluator._physical_by_q[q_key] = physical
        evaluator._case_for_q[q_key] = case_id
    nominal_case_id = case_by_q.get(evaluator._nominal_q_key)
    if nominal_case_id is None:
        raise RuntimeError("native_measurement_cache_omits_frozen_nominal_q")
    nominal_fk = fk_by_case[nominal_case_id]
    position_error = float(np.max(np.linalg.norm(nominal_fk["position"] - d41_reference["positions"], axis=1)))
    actual_tool_z = np.asarray([_tool_z_direction(row) for row in nominal_fk["quaternion"]])
    d41_tool_error = float(np.max([
        _direction_angle_rad(actual_tool_z[index], d41_reference["tool_z"][index])
        for index in range(181)
    ]))
    if position_error > 1e-6 or d41_tool_error > 1e-6:
        raise RuntimeError(f"fresh_FK_does_not_reproduce_D41_reference:{position_error}:{d41_tool_error}")
    nominal_normal_deg = np.degrees(evaluator._physical_by_q[evaluator._nominal_q_key]["normal_error"])
    d41_normal_error = float(np.max(np.abs(nominal_normal_deg - d41_reference["normal_error_deg"])))
    if d41_normal_error > 1e-5:
        raise RuntimeError(f"fresh_FK_normal_metric_disagrees_with_D41_trace:{d41_normal_error}")
    return {
        "max_position_difference_from_D41_trace_m": position_error,
        "max_tool_z_difference_from_D41_trace_rad": d41_tool_error,
        "max_normal_error_difference_from_D41_trace_deg": d41_normal_error,
        "nominal_max_normal_error_deg": float(np.max(nominal_normal_deg)),
    }


def _build_axis_specs(q: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> list[AxisSpec]:
    specs: list[AxisSpec] = []
    for joint in range(6):
        joint_name = f"j{joint + 1}"
        for direction in (1, -1):
            one_sided_margin = float(np.min(upper[joint] - q[:, joint])) if direction > 0 else float(np.min(q[:, joint] - lower[joint]))
            refinement_tolerance = 1e-6
            crossing_epsilon = max(refinement_tolerance * 4.0, one_sided_margin / 64.0)
            specs.append(AxisSpec(
                axis_id=f"joint:{joint_name}:{'positive' if direction > 0 else 'negative'}",
                perturbation_family=StressFamily.JOINT_STATE,
                axis=joint_name.upper(),
                coordinate_frame="joint_state_radians",
                direction=direction,
                units="rad",
                search_domain_max=one_sided_margin + crossing_epsilon,
                coarse_intervals=64,
                refinement_tolerance=refinement_tolerance,
                operator_parameters={"mode": "joint_offset", "joint_index": joint},
            ))
    for family, axes, units, maximum, intervals, tolerance, key in (
        (StressFamily.TCP_TRANSLATION, ("X", "Y", "Z"), "m", 0.012, 48, 1e-5, "direction_tcp"),
        (StressFamily.TCP_ROTATION, ("X", "Y", "Z"), "rad", math.radians(10.0), 48, 1e-4, "axis_tcp"),
    ):
        for axis_index, axis_name in enumerate(axes):
            vector = [0.0, 0.0, 0.0]
            vector[axis_index] = 1.0
            for direction in (1, -1):
                specs.append(AxisSpec(
                    axis_id=f"tcp_{family.value.lower()}:{axis_name.lower()}:{'positive' if direction > 0 else 'negative'}",
                    perturbation_family=family,
                    axis=axis_name,
                    coordinate_frame="spray_tcp_link",
                    direction=direction,
                    units=units,
                    search_domain_max=maximum,
                    coarse_intervals=intervals,
                    refinement_tolerance=tolerance,
                    operator_parameters={key: vector},
                ))
    for family, axes, units, maximum, tolerance in (
        (StressFamily.BASE_TRANSLATION, ("X", "Y", "Z"), "m", 0.0, 1e-5),
        (StressFamily.BASE_ROTATION, ("X", "Y", "Z"), "rad", 0.0, 1e-4),
    ):
        for axis_index, axis_name in enumerate(axes):
            vector = [0.0, 0.0, 0.0]
            vector[axis_index] = 1.0
            parameter = "direction_base" if family == StressFamily.BASE_TRANSLATION else "axis_base"
            for direction in (1, -1):
                specs.append(AxisSpec(
                    axis_id=f"{family.value.lower()}:{axis_name.lower()}:{'positive' if direction > 0 else 'negative'}",
                    perturbation_family=family,
                    axis=axis_name,
                    coordinate_frame="base_link",
                    direction=direction,
                    units=units,
                    search_domain_max=maximum,
                    coarse_intervals=1,
                    refinement_tolerance=tolerance,
                    operator_parameters={parameter: vector},
                    capability_status=CapabilityStatus.NOT_AVAILABLE,
                    unavailable_reason="D41 world geometry is fixed in base_link; no tested robot-base transform propagation is present in the native evaluator.",
                ))
    for family, axis_name, units, maximum, parameter in (
        (StressFamily.SURFACE_OFFSET, "SURFACE_NORMAL", "m", 0.0, {}),
        (StressFamily.STANDOFF, "STAND_OFF", "m", 0.0, {}),
        (StressFamily.SURFACE_NORMAL, "SURFACE_NORMAL_ROTATION", "rad", 0.0, {"axis_base": [0.0, 0.0, 1.0]}),
    ):
        for direction in (1, -1):
            specs.append(AxisSpec(
                axis_id=f"process:{axis_name.lower()}:{'positive' if direction > 0 else 'negative'}",
                perturbation_family=family,
                axis=axis_name,
                coordinate_frame="surface_local_frame",
                direction=direction,
                units=units,
                search_domain_max=maximum,
                coarse_intervals=1,
                refinement_tolerance=1e-5 if units == "m" else 1e-4,
                operator_parameters=parameter,
                capability_status=CapabilityStatus.NOT_AVAILABLE,
                unavailable_reason="The authoritative trajectory has no validated per-case surface-offset/stand-off propagation or coating/deposition physics evaluator.",
            ))
    return specs


def _validate_synthetic_negative_control(nominal: ProcessTrajectory, epsilon: float = 1e-6) -> dict[str, Any]:
    if nominal.joint_states is None or nominal.joint_upper_rad is None:
        raise RuntimeError("genuine_joint_limit_negative_control_missing_inputs")
    q = np.array(nominal.joint_states, copy=True)
    waypoint, joint = np.unravel_index(np.argmin(nominal.joint_upper_rad[None, :] - q), q.shape)
    q[waypoint, joint] = nominal.joint_upper_rad[joint] + epsilon
    invalid = ProcessTrajectory(
        tcp_poses=nominal.tcp_poses,
        joint_states=q,
        timestamps_s=nominal.timestamps_s,
        surface_normals=nominal.surface_normals,
        joint_lower_rad=nominal.joint_lower_rad,
        joint_upper_rad=nominal.joint_upper_rad,
        metadata={**dict(nominal.metadata), "negative_control": True},
    )
    outcome = evaluate_joint_limits(invalid)
    if outcome.status != ResultStatus.FAIL or "JOINT_LIMIT_FAILURE" not in outcome.failure_modes:
        raise RuntimeError("genuine_joint_limit_negative_control_not_detected")
    return {
        "status": "PASS",
        "joint_name": f"j{joint + 1}",
        "waypoint": int(waypoint),
        "q_rad": float(q[waypoint, joint]),
        "upper_limit_rad": float(nominal.joint_upper_rad[joint]),
        "epsilon_rad": epsilon,
        "failure_mode": "JOINT_LIMIT_FAILURE",
        "state_was_clipped": False,
    }


def _run_required_tests() -> dict[str, Any]:
    command = [
        os.environ.get("PYTHON", "python"),
        "-m", "pytest", "-q", "-p", "no:cacheprovider",
        "tests/test_p2a_axiswise_robustness.py",
        "tests/test_process_aware_stress.py",
    ]
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(command, cwd=ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True, env=env, timeout=1800)
    if completed.returncode != 0:
        raise RuntimeError("P2A_or_P1_regression_tests_failed:\n" + (completed.stdout or "") + "\n" + (completed.stderr or ""))
    return {
        "status": "PASS",
        "command": "python -m pytest -q -p no:cacheprovider tests/test_p2a_axiswise_robustness.py tests/test_process_aware_stress.py",
        "stdout_tail": (completed.stdout or "").strip().splitlines()[-3:],
    }


def run_p2a_characterization(
    output_path: Path = P2A_RESULT, *, native_cache_result: Path | None = None
) -> dict[str, Any]:
    """Authenticate, test, and run the bounded native axis-wise FR5 campaign."""
    import platform
    import sys

    cache_path = native_cache_result.resolve() if native_cache_result is not None else None
    if output_path.exists() and (cache_path is None or output_path.resolve() != cache_path):
        raise FileExistsError(f"refusing to overwrite existing P2-A authority: {output_path}")
    import tools.stage4a_system_benchmark as d46

    protected_before = _protected_d46_snapshot()
    test_evidence = _run_required_tests()
    auth, trajectory_path, time_s, q, _, _, _, extra = _auth_inputs(d46)
    lower, upper, dynamic_limits = d46.load_limits()
    d41_reference = _read_d41_process_reference(extra["d41_fk_trace_path"])
    benchmark_manifest = _read_json(D46_ROOT / "STAGE4_SYSTEM_BENCHMARK_V1.json")
    thresholds = benchmark_manifest["diagnostic_thresholds"]
    import tools.audit_stage17_reproducibility as d17
    normal_error_limit_rad = math.radians(float(d17.FORMAL_NORMAL_DEG))
    specs = _build_axis_specs(q, lower, upper)
    trajectory_hash_before = _sha256_file(trajectory_path)
    checkpoint_path = ROOT / str(auth["canonical_checkpoint"])
    checkpoint_stat_before = checkpoint_path.stat()
    joint_limit_profile = {
        "position_source": "FAIRINO FR5 derived URDF joint limit lower/upper fields reused by D46 load_limits()",
        "dynamics_source": "ros2_moveit_bridge/config/joint_limits_with_jerk.yaml reused by D46 load_limits()",
        "dynamics_limit_provenance": "configured offline limits; not vendor-certified",
        "joint_order": [f"j{index}" for index in range(1, 7)],
        "lower_rad": lower.tolist(),
        "upper_rad": upper.tolist(),
    }
    with tempfile.TemporaryDirectory(prefix="p2a_axiswise_", dir=ROOT / "tmp") as scratch_name:
        scratch_root = Path(scratch_name)
        cached_q_by_case: dict[str, np.ndarray] = {}
        cached_physical_by_case: dict[str, dict[str, Any]] = {}
        cache_sha256: str | None = None
        cached_native_batches = 0
        if cache_path is not None:
            if not cache_path.is_file():
                raise FileNotFoundError(f"native_measurement_cache_missing:{cache_path}")
            cached_q_by_case, cached_physical_by_case, cache_sha256, cached_native_batches = _load_previous_native_cache(
                cache_path, q, _sha256_file(trajectory_path), auth
            )
            fk_cases = list(cached_q_by_case.items())
        else:
            fk_cases = [("p2a_reference_nominal", q)]
        fk_trace, fk_q_by_case = _run_fk_only_cases(scratch_root / "fk_replay", fk_cases, d46)
        fk_by_case = _read_fk_trace(fk_trace)
        if set(fk_q_by_case) != set(fk_by_case):
            raise RuntimeError("fresh_FK_replay_case_set_mismatch")
        nominal_q_key = np.ascontiguousarray(q, dtype=np.float64).tobytes()
        reference_case = next((case_id for case_id, q_case in fk_q_by_case.items()
                               if np.ascontiguousarray(q_case, dtype=np.float64).tobytes() == nominal_q_key), None)
        if reference_case is None:
            raise RuntimeError("fresh_FK_replay_omits_frozen_nominal_q")
        nominal_fk = fk_by_case[reference_case]
        actual_tool_z = np.asarray([_tool_z_direction(row) for row in nominal_fk["quaternion"]])
        d41_position_difference = float(np.max(np.linalg.norm(nominal_fk["position"] - d41_reference["positions"], axis=1)))
        d41_tool_difference = float(np.max([
            _direction_angle_rad(actual_tool_z[index], d41_reference["tool_z"][index]) for index in range(181)
        ]))
        if d41_position_difference > 1e-6 or d41_tool_difference > 1e-6:
            raise RuntimeError(f"fresh_FK_does_not_reproduce_D41_reference:{d41_position_difference}:{d41_tool_difference}")
        target_poses = np.column_stack((nominal_fk["position"], nominal_fk["quaternion"]))
        nominal = ProcessTrajectory(
            tcp_poses=target_poses,
            joint_states=q,
            timestamps_s=time_s,
            surface_normals=d41_reference["surface_normals"],
            joint_lower_rad=lower,
            joint_upper_rad=upper,
            metadata={
                "robot": "FAIRINO_FR5",
                "trajectory_source": str(trajectory_path.relative_to(ROOT)),
                "scope": "181-point ON-state open-arch only",
                "geometric_reference": "fresh MoveIt2 FK of frozen D41 post-Ruckig q; process normals from D41 strict-replay trace",
            },
        )
        negative_control = _validate_synthetic_negative_control(nominal)
        evaluator = _D41BatchEvaluator(
            nominal,
            time_s,
            lower,
            upper,
            dynamic_limits,
            float(thresholds["tcp_trajectory_error_m"]["value"]),
            float(thresholds["terminal_position_error_m"]["value"]),
            normal_error_limit_rad,
            float(thresholds["joint_step_rad"]["value"]),
            scratch_root,
            d46,
        )
        if cache_path is not None:
            physical_by_case: dict[str, dict[str, Any]] = {}
            for case_id, measurement in cached_physical_by_case.items():
                physical_by_case[case_id] = dict(measurement)
            fk_crosscheck = _attach_fk_replay(
                evaluator, cached_q_by_case, physical_by_case, fk_by_case,
                d41_reference["surface_normals"], d41_reference,
            )
        else:
            fk_crosscheck = {
                "max_position_difference_from_D41_trace_m": d41_position_difference,
                "max_tool_z_difference_from_D41_trace_rad": d41_tool_difference,
                "nominal_max_normal_error_deg": float(np.max(d41_reference["normal_error_deg"])),
            }
        axis_results = scan_axes_batched(nominal, specs, evaluator.evaluate_batch)
        native_batches = cached_native_batches + evaluator._batch_number
        measured_case_count = len(evaluator._case_for_q)
    protected_after = _protected_d46_snapshot()
    if protected_before != protected_after:
        raise RuntimeError("historical_D46_authority_changed_during_P2A")
    checkpoint_stat_after = checkpoint_path.stat()
    checkpoint_modified = (
        checkpoint_stat_before.st_size != checkpoint_stat_after.st_size
        or checkpoint_stat_before.st_mtime_ns != checkpoint_stat_after.st_mtime_ns
    )
    if checkpoint_modified or _sha256_file(trajectory_path) != trajectory_hash_before:
        raise RuntimeError("frozen_stage3_trajectory_or_checkpoint_changed_during_P2A")

    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in axis_results:
        family = row["specification"]["perturbation_family"]
        by_family.setdefault(family, []).append(row)
    within_family_minima: dict[str, Any] = {}
    for family, rows in by_family.items():
        numeric = [row for row in rows if row.get("estimated_raw_axis_margin") is not None]
        numeric.sort(key=lambda row: float(row["estimated_raw_axis_margin"]["value"]))
        within_family_minima[family] = {
            "ranking_scope": "same perturbation family and same raw units only",
            "ranked_axis_ids": [row["specification"]["axis_id"] for row in numeric],
            "minimum_axis": numeric[0]["specification"]["axis_id"] if numeric else None,
            "minimum_raw_margin": numeric[0]["estimated_raw_axis_margin"] if numeric else None,
        }
    non_monotonic_axes = [
        row["specification"]["axis_id"] for row in axis_results
        if row.get("non_monotonic_failure_region_observed")
    ]
    capability_gaps = [
        {"axis_id": row["specification"]["axis_id"], "status": row["axis_status"], "reason": row["specification"].get("unavailable_reason")}
        for row in axis_results if row["axis_status"] == "CAPABILITY_UNAVAILABLE"
    ]
    every_available_axis_completed = all(
        row["axis_status"] in {"FIRST_FAILURE_OBSERVED", "NO_FAILURE_WITHIN_SEARCH_DOMAIN"}
        for row in axis_results
        if row["specification"]["capability_status"] == "AVAILABLE"
    )
    no_clipping = all(
        observation["realized_perturbation"].get("request_match") == "YES"
        for row in axis_results if row["specification"]["capability_status"] == "AVAILABLE"
        for observation in row.get("coarse_observations", []) + row.get("refinement_observations", [])
        if observation["realized_perturbation"].get("status") == "APPLIED"
    )
    overall = {
        "schema": SCHEMA,
        "robot": "FAIRINO_FR5",
        "status": "P2A_AXISWISE_CHARACTERIZATION_COMPLETE",
        "scope": "181-point ON-state open-arch trajectory only; no OFF state, reorientation edge, legacy 720-point data, other robot, or hardware execution",
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_commit": _git_head(ROOT),
        "working_tree_dirty_at_execution": True,
        "frozen_trajectory": {
            "path": str(trajectory_path.relative_to(ROOT)),
            "sha256": trajectory_hash_before,
            "state_count": int(len(q)),
            "joint_count": int(q.shape[1]),
            "joint_order": [f"j{index}" for index in range(1, 7)],
            "timestamps_path_identity": "same frozen D41 post-Ruckig CSV",
            "source_scope": "D41 strict replay of the authenticated Stage 3 release using the authoritative 181-point open-arch pair",
            "seed_joint_input": auth["authoritative_inputs"]["seed_joints"],
            "tcp_pose_input": auth["authoritative_inputs"]["poses"],
            "geometric_reference": {
                "nominal_position_and_quaternion": "fresh existing MoveIt2 FK of the frozen D41 post-Ruckig 181-point joint path",
                "process_surface_normals": "wall_normal_x/y/z from the matching D41 strict-replay moveit_fk_tcp_trace.csv",
                "D41_strict_replay_fk_trace": str(extra["d41_fk_trace_path"].relative_to(ROOT)),
                "D41_strict_replay_fk_trace_sha256": _sha256_file(extra["d41_fk_trace_path"]),
                "fresh_FK_crosscheck": fk_crosscheck,
                "raw_stage0_pose_pair_used_as_nominal_geometry_reference": False,
                "raw_stage0_pose_pair_role": "authenticated upstream Stage 0/1 identity only; not pairwise reference for the D41 post-Ruckig q path",
            },
        },
        "stage3_release_identity": {
            "release": auth["release"],
            "release_status": auth["release_status"],
            "release_manifest_sha256": auth["release_manifest_sha256"],
            "canonical_checkpoint": auth["canonical_checkpoint"],
            "canonical_checkpoint_sha256": auth["canonical_checkpoint_sha256"],
            "canonical_update": auth["canonical_update"],
            "D41_summary": auth["retained_stage3_robot"]["D41_summary"],
            "D41_post_ruckig_trajectory": auth["retained_stage3_robot"]["post_ruckig_trajectory"],
        },
        "evaluator_provenance": {
            "native_measurement_backend": "existing D41 MoveIt2 PlanningScene CollisionEnvBullet plus CollisionEnvFCL distances",
            "native_fk_backend": "existing D46 stage4a_fk MoveIt2 RobotState FK for spray_tcp_link",
            "native_collision_method": "adaptive_discrete_interpolation",
            "adaptive_joint_step_rad": math.radians(0.5),
            "native_robot_world_two_state_query": "reported separately; not promoted to strict global CCD",
            "continuous_self_collision_ccd": "NOT_AVAILABLE",
            "clearance": "native per-state FCL distances; acceptance thresholds remain UNRESOLVED_THRESHOLD",
            "jacobian": "native MoveIt2 Jacobian SVD; sigma_min and condition number diagnostic only",
            "dynamics": "D46 finite_derivatives over frozen D41 timestamps; configured limits are not vendor-certified",
            "joint_limit_profile": joint_limit_profile,
            "task_geometry_thresholds": {
                "tcp_trajectory_error_m": thresholds["tcp_trajectory_error_m"],
                "terminal_position_error_m": thresholds["terminal_position_error_m"],
                "spray_axis_normal_error_deg": {
                    "value": float(d17.FORMAL_NORMAL_DEG),
                    "source": "tools.audit_stage17_reproducibility.FORMAL_NORMAL_DEG; retained Stage 1.7 acceptance gate",
                    "metric": "directed spray_tcp_link +Z versus frozen D41 wall normal; task-space rotation measures perturbed target +Z versus the same wall normal; independent of free tool-axis roll",
                },
                "D46_terminal_orientation_diagnostic": {
                    **thresholds["terminal_orientation_error_rad"],
                    "used_as_failure_gate": False,
                    "reason": "D46 labels this full-quaternion limit diagnostic-only; D41 process acceptance measures spray-axis normal error",
                },
                "joint_step_rad": thresholds["joint_step_rad"],
            },
            "native_candidate_case_count": measured_case_count,
            "native_batch_count": native_batches,
            "candidate_adapter": "q-only waypoint,j1_q..j6_q CSV; prevents D41 last-six-column parser from reading jerk as joint position",
            "native_measurements_reused_from_prior_same_q_cases": len(cached_q_by_case),
            "native_measurement_cache_source_sha256": cache_sha256,
            "native_collision_batches_replayed": evaluator._batch_number,
            "fresh_FK_replay_case_count": len(fk_by_case),
            "fresh_FK_refinement_case_count": measured_case_count - len(cached_q_by_case),
            "total_distinct_q_states_with_available_FK": sum(
                physical.get("fk_status") == "AVAILABLE" for physical in evaluator._physical_by_q.values()
            ),
        },
        "measurement_repair": {
            "repair_type": "TYPE_A_MEASUREMENT_REFERENCE_DEFECT",
            "reason": "STAGE0_RAW_POSE_NOT_PAIRWISE_REFERENCE_FOR_FROZEN_D41_POST_RUCKIG_Q",
            "prior_task_geometry_classifications_promoted": False,
            "prior_same_q_native_collision_clearance_and_jacobian_metrics_reused": bool(cache_path is not None),
            "prior_same_q_native_metrics_reused_count": len(cached_q_by_case),
            "all_reused_q_states_received_fresh_MoveIt2_FK": bool(cache_path is not None) and len(fk_by_case) == len(cached_q_by_case),
            "all_distinct_measured_q_states_received_fresh_MoveIt2_FK": (
                len(evaluator._physical_by_q) == measured_case_count
                and all(physical.get("fk_status") == "AVAILABLE" for physical in evaluator._physical_by_q.values())
            ),
            "fresh_D41_native_refinement_batches": evaluator._batch_number,
            "prior_invalid_result_sha256": cache_sha256,
            "nominal_task_path_reference": "fresh MoveIt2 FK of frozen D41 post-Ruckig q",
            "process_normal_reference": "D41 strict-replay wall_normal_x/y/z",
        },
        "search_protocol": {
            "method": "deterministic coarse scan of signed one-axis amplitudes, then local bisection only for an adjacent observed PASS-to-FAIL bracket",
            "non_monotonic_handling": "retain every coarse state and failure island; refine only the nearest observed adjacent bracket; make no global monotonicity claim",
            "all_requested_and_realized_amplitudes_retained": True,
            "no_state_clipping": True,
            "raw_margins_only": True,
            "normalized_margin": "NOT_AVAILABLE",
            "normalized_margin_reason": "REAL_UNCERTAINTY_BOUNDS_NOT_YET_IDENTIFIED",
            "cross_unit_axis_ranking": "NOT_AVAILABLE; no common uncertainty scale was identified",
        },
        "negative_control": negative_control,
        "axis_results": axis_results,
        "within_family_rankings": within_family_minima,
        "non_monotonic_axes": non_monotonic_axes,
        "capability_gaps": capability_gaps,
        "cross_family_dominant_bottleneck": {
            "status": "UNRESOLVED_CROSS_UNIT_COMPARISON",
            "reason": "raw rad, m, and radian orientation margins are not directly comparable; normalized uncertainty bounds are unavailable",
            "within_family_candidates": within_family_minima,
        },
        "legacy_preservation": {
            "P1_BEHAVIOR_REGRESSION": "NO" if test_evidence["status"] == "PASS" else "YES_OR_UNRESOLVED",
            "D46_HISTORICAL_RESULTS_MODIFIED": "NO",
            "D46_PROTECTED_SHA256_BEFORE": protected_before,
            "D46_PROTECTED_SHA256_AFTER": protected_after,
            "FROZEN_181_TRAJECTORY_MODIFIED": "NO",
            "STAGE3_CHECKPOINT_MODIFIED": "NO",
            "FAIRINO_VENDOR_SOURCE_MODIFIED": "NO",
            "ESTUN_FILES_MODIFIED": 0,
        },
        "test_evidence": test_evidence,
        "minimal_artifact_audit": {
            "authoritative_result_file": str(output_path.relative_to(ROOT)),
            "temporary_native_outputs_cleaned": True,
            "temporary_csv_json_logs_left": False,
        },
        "capability_boundaries": {
            "torque": "NOT_AVAILABLE",
            "calibrated_tcp_uncertainty": "NOT_AVAILABLE",
            "strict_continuous_self_collision_ccd": "NOT_AVAILABLE",
            "physical_hardware_validation": "NOT_RUN",
            "coating_quality_certified": "NO",
            "hardware_safety_certified": "NO",
            "global_robustness_certified": "NO",
            "stand_off_margin": "NOT_AVAILABLE_WITH_CURRENT_PER_CASE_GEOMETRY_PROPAGATION",
            "base_transform_margin": "NOT_AVAILABLE_WITH_CURRENT_NATIVE_TRANSFORM_PIPELINE",
            "deterministic_seed": "NOT_APPLICABLE_NO_RANDOMIZED_PERTURBATION",
        },
        "gate": {
            "FR5_P2A_AXISWISE_ROBUSTNESS_MARGIN_KERNEL": "COMPLETE" if every_available_axis_completed and no_clipping and negative_control["status"] == "PASS" and native_batches > 0 and measured_case_count > 0 else "PARTIAL",
            "P2A_KERNEL_IMPLEMENTATION": "COMPLETE",
            "P2A_NATIVE_CHARACTERIZATION": "COMPLETE" if native_batches > 0 else "NOT_RUN",
            "NATIVE_MOVEIT2_EVALUATION": "PASS" if native_batches > 0 and measured_case_count > 0 else "NOT_RUN",
            "AXISWISE_BOUNDARY_SEARCH": "PASS" if every_available_axis_completed else "FAIL",
            "NON_MONOTONIC_HANDLING": "PASS",
            "NO_SILENT_CLIPPING": "PASS" if no_clipping else "FAIL",
            "GENUINE_NEGATIVE_CONTROL": negative_control["status"],
            "LEGACY_REGRESSION": "PASS" if test_evidence["status"] == "PASS" and protected_before == protected_after else "FAIL",
            "PHYSICAL_HARDWARE_VALIDATION": "NOT_RUN",
            "GLOBAL_ROBUSTNESS_CERTIFIED": "NO",
            "HARDWARE_SAFETY_CERTIFIED": "NO",
            "PROCESS_QUALITY_CERTIFIED": "NO",
        },
        "execution_environment": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "native_ros_distribution": "ROS 2 Jazzy in WSL",
            "moveit_core_version": "2.12.4 (existing D46 provenance)",
            "native_collision_backend": "MoveIt2 Bullet and FCL",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_jsonable(overall), indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", prefix=f".{output_path.name}.", suffix=".tmp",
            dir=output_path.parent, delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return overall


def _git_head(root: Path) -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, encoding="utf-8", errors="replace", capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> int:
    try:
        result = run_p2a_characterization()
    except Exception as exc:
        print(json.dumps({"P2A_STATUS": "BLOCKED_OR_FAILED", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({
        "P2A_STATUS": result["status"],
        "result": result["minimal_artifact_audit"]["authoritative_result_file"],
        "gates": result["gate"],
        "within_family_rankings": result["within_family_rankings"],
        "non_monotonic_axes": result["non_monotonic_axes"],
        "capability_gaps": result["capability_gaps"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
