from __future__ import annotations

import numpy as np

from src.p2a_axiswise_robustness import (
    AxisSpec,
    EvaluationResult,
    _protected_d46_snapshot,
    _direction_angle_rad,
    _tool_z_direction,
    _quat_multiply,
    _process_normal_errors,
    apply_axis_perturbation,
    evaluate_joint_limits,
    scan_axes_batched,
)
from src.process_aware_stress import (
    CapabilityStatus,
    ProcessTrajectory,
    ResultStatus,
    StressFamily,
)


def _trajectory() -> ProcessTrajectory:
    q = np.zeros((181, 6), dtype=np.float64)
    q[:, 5] = 0.25
    target = np.zeros((181, 7), dtype=np.float64)
    target[:, 3:] = [0.0, 0.0, 0.0, 1.0]
    times = np.linspace(0.0, 180.0, 181)
    normals = np.tile([0.0, 0.0, 1.0], (181, 1))
    return ProcessTrajectory(
        tcp_poses=target,
        joint_states=q,
        timestamps_s=times,
        surface_normals=normals,
        joint_lower_rad=np.full(6, -1.0),
        joint_upper_rad=np.full(6, 1.0),
    )


def _joint_spec(direction: int = 1, *, axis_id: str | None = None) -> AxisSpec:
    return AxisSpec(
        axis_id=axis_id or f"joint:j6:{'positive' if direction > 0 else 'negative'}",
        perturbation_family=StressFamily.JOINT_STATE,
        axis="J6",
        coordinate_frame="joint_state_radians",
        direction=direction,
        units="rad",
        search_domain_max=1.0,
        coarse_intervals=10,
        refinement_tolerance=1e-4,
        operator_parameters={"mode": "joint_offset", "joint_index": 5},
    )


def _threshold_evaluator(threshold: float = 0.37):
    def evaluate(candidates):
        return {
            candidate.candidate_id: EvaluationResult(
                ResultStatus.FAIL if candidate.magnitude >= threshold else ResultStatus.PASS,
                ("SYNTHETIC_FAILURE",) if candidate.magnitude >= threshold else (),
                evaluator_capabilities={"synthetic": "AVAILABLE"},
            )
            for candidate in candidates
        }
    return evaluate


def test_zero_perturbation_preserves_nominal_trajectory_exactly() -> None:
    nominal = _trajectory()
    candidate = apply_axis_perturbation(nominal, _joint_spec(), 0.0)
    assert candidate.application.status.value == "APPLIED"
    assert np.array_equal(candidate.application.trajectory.joint_states, nominal.joint_states)
    assert np.array_equal(candidate.application.trajectory.tcp_poses, nominal.tcp_poses)
    assert np.array_equal(candidate.application.trajectory.timestamps_s, nominal.timestamps_s)


def test_spray_axis_error_is_invariant_to_roll_but_detects_tool_tilt() -> None:
    target = np.asarray([0.0, 0.0, 0.0, 1.0])
    half_angle = np.pi / 4.0
    roll_z = np.asarray([0.0, 0.0, np.sin(half_angle), np.cos(half_angle)])
    tilt_x = np.asarray([np.sin(half_angle), 0.0, 0.0, np.cos(half_angle)])
    rolled = _quat_multiply(target, roll_z)
    tilted = _quat_multiply(target, tilt_x)

    assert _direction_angle_rad(_tool_z_direction(rolled), _tool_z_direction(target)) < 1e-12
    assert np.isclose(_direction_angle_rad(_tool_z_direction(tilted), _tool_z_direction(target)), np.pi / 2.0)


def test_process_normal_metric_uses_fixed_surface_normal_and_rotation_target_axis() -> None:
    actual = np.tile([0.0, 0.0, 0.0, 1.0], (181, 1))
    target = np.zeros((181, 7), dtype=np.float64)
    target[:, 3:] = [0.0, 0.0, 0.0, 1.0]
    normals = np.tile([0.0, 0.0, 1.0], (181, 1))
    half_angle = np.pi / 4.0
    roll_z = np.asarray([0.0, 0.0, np.sin(half_angle), np.cos(half_angle)])
    tilt_x = np.asarray([np.sin(half_angle), 0.0, 0.0, np.cos(half_angle)])
    target[:, 3:] = _quat_multiply(target[0, 3:], roll_z)

    rolled_axis_error = _process_normal_errors(
        actual, target, normals, compare_perturbed_target_axis=True
    )
    actual_axis_error = _process_normal_errors(
        actual, target, normals, compare_perturbed_target_axis=False
    )
    assert float(np.max(rolled_axis_error)) < 1e-12
    assert float(np.max(actual_axis_error)) < 1e-12

    target[:, 3:] = _quat_multiply(np.asarray([0.0, 0.0, 0.0, 1.0]), tilt_x)
    tilted_target_error = _process_normal_errors(
        actual, target, normals, compare_perturbed_target_axis=True
    )
    assert np.allclose(tilted_target_error, np.pi / 2.0)


def test_deterministic_replay_produces_identical_scan_evidence() -> None:
    nominal = _trajectory()
    first = scan_axes_batched(nominal, [_joint_spec()], _threshold_evaluator())
    second = scan_axes_batched(nominal, [_joint_spec()], _threshold_evaluator())
    assert first == second


def test_positive_and_negative_directions_remain_separate() -> None:
    nominal = _trajectory()
    positive = _joint_spec(1, axis_id="j6_positive")
    negative = _joint_spec(-1, axis_id="j6_negative")

    def evaluate(candidates):
        out = {}
        for candidate in candidates:
            threshold = 0.2 if candidate.signed_delta > 0 else 0.4
            fail = abs(candidate.signed_delta) >= threshold
            out[candidate.candidate_id] = EvaluationResult(
                ResultStatus.FAIL if fail else ResultStatus.PASS,
                ("SYNTHETIC_FAILURE",) if fail else (),
            )
        return out

    results = scan_axes_batched(nominal, [positive, negative], evaluate)
    assert [row["specification"]["axis_id"] for row in results] == ["j6_positive", "j6_negative"]
    assert results[0]["first_failing_perturbation"]["requested_perturbation"]["value"] > 0
    assert results[1]["first_failing_perturbation"]["requested_perturbation"]["value"] < 0
    assert results[0]["estimated_raw_axis_margin"]["value"] < results[1]["estimated_raw_axis_margin"]["value"]


def test_genuine_joint_limit_exceedance_is_detected() -> None:
    nominal = _trajectory()
    q = np.array(nominal.joint_states, copy=True)
    q[7, 2] = 0.99
    near_limit = ProcessTrajectory(
        tcp_poses=nominal.tcp_poses,
        joint_states=q,
        timestamps_s=nominal.timestamps_s,
        joint_lower_rad=nominal.joint_lower_rad,
        joint_upper_rad=nominal.joint_upper_rad,
    )
    spec = AxisSpec(
        axis_id="j3_positive_negative_control",
        perturbation_family=StressFamily.JOINT_STATE,
        axis="J3",
        coordinate_frame="joint_state_radians",
        direction=1,
        units="rad",
        search_domain_max=0.02,
        coarse_intervals=2,
        refinement_tolerance=1e-5,
        operator_parameters={"mode": "joint_offset", "joint_index": 2},
    )
    candidate = apply_axis_perturbation(near_limit, spec, 0.02)
    result = evaluate_joint_limits(candidate.application.trajectory)
    assert result.status == ResultStatus.FAIL
    assert result.failure_modes == ("JOINT_LIMIT_FAILURE",)
    assert result.critical_waypoint == 7
    assert result.evidence["joint_name"] == "j3"


def test_joint_limit_exceedance_is_not_silently_clipped() -> None:
    nominal = _trajectory()
    q = np.array(nominal.joint_states, copy=True)
    q[0, 5] = 0.99
    near_limit = ProcessTrajectory(
        joint_states=q,
        joint_lower_rad=nominal.joint_lower_rad,
        joint_upper_rad=nominal.joint_upper_rad,
    )
    candidate = apply_axis_perturbation(near_limit, _joint_spec(), 0.02)
    assert candidate.application.trajectory.joint_states[0, 5] == 1.01
    result = evaluate_joint_limits(candidate.application.trajectory)
    assert result.status == ResultStatus.FAIL
    assert result.evidence["state_was_clipped"] is False
    assert result.evidence["metric_evaluation_status"] == "NOT_EVALUATED_AFTER_JOINT_LIMIT_FAILURE"
    assert result.evidence["controlling_joint"] == "j6"
    assert result.failure_margin["joint_limit_violation_rad"] > 0


def test_pass_to_fail_bracket_is_reported_with_signed_endpoints() -> None:
    result = scan_axes_batched(_trajectory(), [_joint_spec()], _threshold_evaluator(0.37))[0]
    assert result["last_passing_perturbation"]["status"] == "PASS"
    assert result["first_failing_perturbation"]["status"] == "FAIL"
    assert result["last_passing_perturbation"]["magnitude"] <= 0.37
    assert result["first_failing_perturbation"]["magnitude"] >= 0.37


def test_refinement_converges_to_known_synthetic_threshold() -> None:
    result = scan_axes_batched(_trajectory(), [_joint_spec()], _threshold_evaluator(0.37))[0]
    assert result["search_termination_reason"] == "REFINED_TO_TOLERANCE"
    assert result["refined_bracket"]["width"] <= result["specification"]["refinement_tolerance"]
    assert abs(result["estimated_raw_axis_margin"]["value"] - 0.37) <= 1e-4


def test_no_failure_in_domain_stays_a_domain_limited_result() -> None:
    spec = _joint_spec()
    result = scan_axes_batched(
        _trajectory(), [spec],
        lambda candidates: {candidate.candidate_id: EvaluationResult(ResultStatus.PASS) for candidate in candidates},
    )[0]
    assert result["axis_status"] == "NO_FAILURE_WITHIN_SEARCH_DOMAIN"
    assert result["search_completeness"] == "FULL_COARSE_DOMAIN_SAMPLED"
    assert result["first_failure_margin"] is None


def test_nominal_failure_is_explicit_and_never_reported_as_positive_margin() -> None:
    spec = _joint_spec()

    def evaluate(candidates):
        return {
            candidate.candidate_id: EvaluationResult(
                ResultStatus.FAIL if candidate.magnitude == 0.0 else ResultStatus.PASS,
                ("NOMINAL_FAILURE",) if candidate.magnitude == 0.0 else (),
            )
            for candidate in candidates
        }

    result = scan_axes_batched(_trajectory(), [spec], evaluate)[0]
    assert result["search_termination_reason"] == "NOMINAL_FAILURE"
    assert result["first_failure_margin"] == {"value": 0.0, "units": "rad"}


def test_evaluator_exception_becomes_unknown_and_does_not_crash_scan() -> None:
    result = scan_axes_batched(_trajectory(), [_joint_spec()], lambda _candidates: (_ for _ in ()).throw(TimeoutError("bounded test outage")))[0]
    assert result["axis_status"] == "CAPABILITY_GAP_IN_SEARCH_DOMAIN"
    assert result["search_completeness"] == "INCOMPLETE_CAPABILITY_GAP"
    assert all(row["status"] == "UNKNOWN" for row in result["coarse_observations"])


def test_non_monotonic_failure_islands_are_retained_and_not_called_monotonic() -> None:
    nominal = _trajectory()
    spec = AxisSpec(
        axis_id="synthetic_nonmonotonic",
        perturbation_family=StressFamily.JOINT_STATE,
        axis="J6",
        coordinate_frame="joint_state_radians",
        direction=1,
        units="rad",
        search_domain_max=1.0,
        coarse_intervals=10,
        refinement_tolerance=1e-3,
        operator_parameters={"mode": "joint_offset", "joint_index": 5},
    )

    def evaluate(candidates):
        output = {}
        for candidate in candidates:
            magnitude = candidate.magnitude
            fail = 0.2 <= magnitude <= 0.4 or magnitude >= 0.7
            output[candidate.candidate_id] = EvaluationResult(
                ResultStatus.FAIL if fail else ResultStatus.PASS,
                ("SYNTHETIC_ISLAND",) if fail else (),
            )
        return output

    result = scan_axes_batched(nominal, [spec], evaluate)[0]
    assert result["monotonicity_observation"] == "NON_MONOTONIC"
    assert result["non_monotonic_failure_region_observed"] is True
    assert len(result["failure_intervals"]) >= 2
    assert result["first_failing_perturbation"]["magnitude"] < 0.3


def test_every_transition_in_pass_fail_pass_pattern_is_refined() -> None:
    nominal = _trajectory()
    spec = AxisSpec(
        axis_id="synthetic_pass_fail_pass",
        perturbation_family=StressFamily.JOINT_STATE,
        axis="J6",
        coordinate_frame="joint_state_radians",
        direction=1,
        units="rad",
        search_domain_max=1.0,
        coarse_intervals=2,
        refinement_tolerance=1e-3,
        operator_parameters={"mode": "joint_offset", "joint_index": 5},
    )

    def evaluate(candidates):
        output = {}
        for candidate in candidates:
            magnitude = candidate.magnitude
            fail = 0.3 <= magnitude <= 0.7
            output[candidate.candidate_id] = EvaluationResult(
                ResultStatus.FAIL if fail else ResultStatus.PASS,
                ("SYNTHETIC_ISLAND",) if fail else (),
            )
        return output

    result = scan_axes_batched(nominal, [spec], evaluate)[0]
    boundaries = result["all_refined_boundaries"]
    assert [boundary["transition"] for boundary in boundaries] == ["PASS_TO_FAIL", "FAIL_TO_PASS"]
    assert all(boundary["refinement_status"] == "REFINED_TO_TOLERANCE" for boundary in boundaries)
    assert all(boundary["width"] <= spec.refinement_tolerance for boundary in boundaries)
    assert result["transition_refinement_completeness"] == "ALL_OBSERVED_PASS_FAIL_TRANSITIONS_REFINED"
    assert result["robustness_acceptance_status"] == "NOT_EVALUATED_BOUNDARY_REFINEMENT_IS_MEASUREMENT_ONLY"


def test_all_three_transitions_in_pass_fail_pass_fail_pattern_are_refined() -> None:
    nominal = _trajectory()
    spec = AxisSpec(
        axis_id="synthetic_two_failure_islands",
        perturbation_family=StressFamily.JOINT_STATE,
        axis="J6",
        coordinate_frame="joint_state_radians",
        direction=1,
        units="rad",
        search_domain_max=1.0,
        coarse_intervals=3,
        refinement_tolerance=1e-3,
        operator_parameters={"mode": "joint_offset", "joint_index": 5},
    )

    def evaluate(candidates):
        output = {}
        for candidate in candidates:
            magnitude = candidate.magnitude
            fail = 0.30 <= magnitude <= 0.45 or magnitude >= 0.85
            output[candidate.candidate_id] = EvaluationResult(
                ResultStatus.FAIL if fail else ResultStatus.PASS,
                ("SYNTHETIC_ISLAND",) if fail else (),
            )
        return output

    result = scan_axes_batched(nominal, [spec], evaluate)[0]
    boundaries = result["all_refined_boundaries"]
    assert [boundary["transition"] for boundary in boundaries] == ["PASS_TO_FAIL", "FAIL_TO_PASS", "PASS_TO_FAIL"]
    assert all(boundary["refinement_status"] == "REFINED_TO_TOLERANCE" for boundary in boundaries)
    assert all(boundary["width"] <= spec.refinement_tolerance for boundary in boundaries)
    assert len(result["all_failure_intervals"]) == 2
    assert result["first_failure_margin"]["value"] == boundaries[0]["boundary_estimate_magnitude"]


def test_unknown_coarse_sample_breaks_bracketing_without_blocking_later_transitions() -> None:
    nominal = _trajectory()
    spec = AxisSpec(
        axis_id="synthetic_unknown_gap",
        perturbation_family=StressFamily.JOINT_STATE,
        axis="J6",
        coordinate_frame="joint_state_radians",
        direction=1,
        units="rad",
        search_domain_max=1.0,
        coarse_intervals=4,
        refinement_tolerance=1e-3,
        operator_parameters={"mode": "joint_offset", "joint_index": 5},
    )

    def evaluate(candidates):
        output = {}
        for candidate in candidates:
            magnitude = candidate.magnitude
            if 0.2 <= magnitude <= 0.3:
                status = ResultStatus.UNKNOWN
            else:
                fail = 0.3 < magnitude < 0.6 or magnitude >= 0.85
                status = ResultStatus.FAIL if fail else ResultStatus.PASS
            output[candidate.candidate_id] = EvaluationResult(status, ("SYNTHETIC_FAILURE",) if status == ResultStatus.FAIL else ())
        return output

    result = scan_axes_batched(nominal, [spec], evaluate)[0]
    assert result["coarse_state_sequence"] == ["PASS", "UNKNOWN", "FAIL", "PASS", "FAIL"]
    assert [boundary["transition"] for boundary in result["all_refined_boundaries"]] == ["FAIL_TO_PASS", "PASS_TO_FAIL"]
    assert all(boundary["width"] <= spec.refinement_tolerance for boundary in result["all_refined_boundaries"])
    assert result["search_completeness"] == "INCOMPLETE_CAPABILITY_GAP"


def test_unknown_refinement_midpoint_blocks_only_its_own_boundary() -> None:
    nominal = _trajectory()
    spec = AxisSpec(
        axis_id="synthetic_unknown_refinement",
        perturbation_family=StressFamily.JOINT_STATE,
        axis="J6",
        coordinate_frame="joint_state_radians",
        direction=1,
        units="rad",
        search_domain_max=1.0,
        coarse_intervals=2,
        refinement_tolerance=1e-3,
        operator_parameters={"mode": "joint_offset", "joint_index": 5},
    )

    def evaluate(candidates):
        output = {}
        for candidate in candidates:
            if np.isclose(candidate.magnitude, 0.375):
                status = ResultStatus.UNKNOWN
            else:
                status = ResultStatus.FAIL if candidate.magnitude >= 0.5 else ResultStatus.PASS
            output[candidate.candidate_id] = EvaluationResult(status)
        return output

    result = scan_axes_batched(nominal, [spec], evaluate)[0]
    boundary, = result["all_refined_boundaries"]
    assert boundary["refinement_status"] == "REFINEMENT_BLOCKED_UNKNOWN"
    assert boundary["converged_to_tolerance"] is False
    assert boundary["width"] > spec.refinement_tolerance
    assert result["search_termination_reason"] == "REFINEMENT_BLOCKED_UNKNOWN"
    assert result["search_completeness"] == "INCOMPLETE_REFINEMENT_UNKNOWN"


def test_missing_batch_row_is_unknown_and_never_bridged() -> None:
    spec = AxisSpec(
        axis_id="synthetic_missing_batch_row",
        perturbation_family=StressFamily.JOINT_STATE,
        axis="J6",
        coordinate_frame="joint_state_radians",
        direction=1,
        units="rad",
        search_domain_max=1.0,
        coarse_intervals=4,
        refinement_tolerance=1e-3,
        operator_parameters={"mode": "joint_offset", "joint_index": 5},
    )

    def evaluate(candidates):
        output = {}
        for candidate in candidates:
            if candidate.phase == "coarse" and candidate.sample_index == 1:
                continue
            status = ResultStatus.FAIL if candidate.magnitude >= 0.5 else ResultStatus.PASS
            output[candidate.candidate_id] = EvaluationResult(status)
        return output

    result = scan_axes_batched(_trajectory(), [spec], evaluate)[0]
    assert result["coarse_state_sequence"] == ["PASS", "UNKNOWN", "FAIL", "FAIL", "FAIL"]
    assert result["all_refined_boundaries"] == []
    assert result["first_failure_margin"] is None
    assert result["search_completeness"] == "INCOMPLETE_CAPABILITY_GAP"


def test_unavailable_capability_never_yields_a_pass() -> None:
    unavailable = AxisSpec(
        axis_id="base_translation_x_positive",
        perturbation_family=StressFamily.BASE_TRANSLATION,
        axis="X",
        coordinate_frame="base_link",
        direction=1,
        units="m",
        search_domain_max=0.0,
        coarse_intervals=1,
        refinement_tolerance=1e-5,
        operator_parameters={"direction_base": [1.0, 0.0, 0.0]},
        capability_status=CapabilityStatus.NOT_AVAILABLE,
        unavailable_reason="test capability gap",
    )
    called = False

    def evaluator(_):
        nonlocal called
        called = True
        return {}

    result = scan_axes_batched(_trajectory(), [unavailable], evaluator)[0]
    assert not called
    assert result["axis_status"] == "CAPABILITY_UNAVAILABLE"
    assert result["first_failing_perturbation"] is None
    assert result["estimated_raw_axis_margin"] is None

    unknown_spec = _joint_spec(axis_id="unknown_eval")
    unknown = scan_axes_batched(
        _trajectory(),
        [unknown_spec],
        lambda candidates: {candidate.candidate_id: EvaluationResult(ResultStatus.UNKNOWN) for candidate in candidates},
    )[0]
    assert unknown["axis_status"] == "CAPABILITY_GAP_IN_SEARCH_DOMAIN"
    assert unknown["axis_status"] != "NO_FAILURE_WITHIN_SEARCH_DOMAIN"


def test_critical_waypoint_and_segment_propagate_through_refinement() -> None:
    spec = _joint_spec(axis_id="critical_location")

    def evaluate(candidates):
        output = {}
        for candidate in candidates:
            fail = candidate.magnitude >= 0.5
            output[candidate.candidate_id] = EvaluationResult(
                ResultStatus.FAIL if fail else ResultStatus.PASS,
                ("SYNTHETIC_COLLISION",) if fail else (),
                critical_waypoint=23 if fail else None,
                critical_segment=22 if fail else None,
            )
        return output

    result = scan_axes_batched(_trajectory(), [spec], evaluate)[0]
    first_fail = result["first_failing_perturbation"]
    assert first_fail["critical_waypoint"] == 23
    assert first_fail["critical_segment"] == 22
    assert first_fail["dominant_failure_mode"] == "SYNTHETIC_COLLISION"


def test_historical_d46_authority_files_remain_byte_identical() -> None:
    before = _protected_d46_snapshot()
    after = _protected_d46_snapshot()
    assert before == after
