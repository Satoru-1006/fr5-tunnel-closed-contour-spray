from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ros2_moveit_bridge.p2b2_redundancy_objectives import (  # noqa: E402
    RESULT_REQUIRED_FIELDS,
    joint_centering_objective,
    joint_limit_barrier_objective,
    normalized_joint_margin_profile,
    rank_aware_secondary_command,
    rank_axis_results_by_family,
    rank_joint_axis_results,
    validate_provenance_schema,
    validate_result_schema,
)
from scripts.run_p2b2_robustness_remapping import VARIANT_DESIGN, _campaign_status  # noqa: E402
from src.p2a_axiswise_robustness import (  # noqa: E402
    AxisSpec,
    EvaluationResult,
    _build_axis_specs,
    scan_axes_batched,
)
from src.process_aware_stress import (  # noqa: E402
    CapabilityStatus,
    ProcessTrajectory,
    ResultStatus,
    StressFamily,
)


def _limits() -> tuple[np.ndarray, np.ndarray]:
    return np.full(6, -2.0), np.full(6, 2.0)


def test_centering_and_barrier_gradients_match_finite_differences() -> None:
    lower, upper = _limits()
    q = np.asarray([-1.4, -0.7, -0.1, 0.35, 0.9, 1.45], dtype=np.float64)
    epsilon = 1e-6

    for objective in (joint_centering_objective, joint_limit_barrier_objective):
        value, gradient = objective(q, lower, upper)
        observed = np.empty(6)
        for index in range(6):
            delta = np.zeros(6)
            delta[index] = epsilon
            plus = objective(q + delta, lower, upper)[0]
            minus = objective(q - delta, lower, upper)[0]
            observed[index] = (plus - minus) / (2.0 * epsilon)
        assert np.isfinite(value)
        assert np.allclose(gradient, observed, rtol=2e-6, atol=2e-7)


def test_barrier_fails_closed_at_and_beyond_position_limits() -> None:
    lower, upper = _limits()
    for q in (lower.copy(), upper.copy(), lower - 1e-6, upper + 1e-6):
        with pytest.raises(ValueError, match="strictly_in_bounds"):
            joint_limit_barrier_objective(q, lower, upper)
    with pytest.raises(ValueError, match="finite_shape_6"):
        joint_limit_barrier_objective([0.0, 0.0, np.nan, 0.0, 0.0, 0.0], lower, upper)


def test_rank5_secondary_command_is_projected_and_other_ranks_are_bounded_zero() -> None:
    lower, upper = _limits()
    task = np.column_stack((np.eye(5), np.zeros(5)))
    projector = np.zeros((6, 6))
    projector[5, 5] = 1.0
    q = np.asarray([0.2, -0.3, 0.1, 0.5, -0.4, 1.2])

    command, norm, task_residual = rank_aware_secondary_command(
        q, lower, upper, projector, 5e-4, "joint_centering", 5, task_jacobian=task,
    )
    assert norm > 0.0
    assert task_residual < 1e-12
    assert np.linalg.norm(task @ command) < 1e-12

    for rank in (0, 1, 4, 6):
        command, norm, residual = rank_aware_secondary_command(
            q, lower, upper, projector, 5e-4, "joint_centering", rank, task_jacobian=task,
        )
        assert np.array_equal(command, np.zeros(6))
        assert norm == 0.0
        assert residual == 0.0


def test_rank5_secondary_command_rejects_bad_projector_or_task_invariance() -> None:
    lower, upper = _limits()
    q = np.asarray([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    task = np.column_stack((np.eye(5), np.zeros(5)))
    bad_projector = 0.5 * np.eye(6)
    with pytest.raises(ValueError, match="not_idempotent"):
        rank_aware_secondary_command(q, lower, upper, bad_projector, 1e-3, "joint_centering", 5, task_jacobian=task)
    with pytest.raises(ValueError, match="violates_first_order"):
        rank_aware_secondary_command(q, lower, upper, np.eye(6), 1e-3, "joint_centering", 5, task_jacobian=task)


def test_frozen_p2a_formal_axis_contract_has_42_axes_and_18_unavailable() -> None:
    q = np.zeros((181, 6), dtype=np.float64)
    q[:, 5] = 0.2
    lower, upper = _limits()
    specs = _build_axis_specs(q, lower, upper)
    assert len(specs) == 42
    assert sum(spec.perturbation_family == StressFamily.JOINT_STATE for spec in specs) == 12
    assert sum(spec.capability_status == CapabilityStatus.NOT_AVAILABLE for spec in specs) == 18
    assert sum(spec.capability_status == CapabilityStatus.AVAILABLE for spec in specs) == 24


def test_redundancy_ablation_has_frozen_r0_r1_r2_r3_policies_and_three_gains() -> None:
    assert len(VARIANT_DESIGN) == 10
    by_family = {
        family: [row for row in VARIANT_DESIGN if row["objective_family"] == family]
        for family in ("R1", "R2", "R3")
    }
    r0 = VARIANT_DESIGN[0]
    assert (r0["variant_id"], r0["solver"], r0["warm_start_rows"], r0["objective"]) == ("R0", "B0", 16, "none")
    expected_gains = {2.5e-4, 5e-4, 1e-3}
    for family in ("R1", "R2", "R3"):
        assert {row["gain"] for row in by_family[family]} == expected_gains
    assert all(row["warm_start_rows"] == 16 and row["objective"] == "joint_centering" for row in by_family["R1"])
    assert all(row["warm_start_rows"] == 1 and row["objective"] == "joint_centering" for row in by_family["R2"])
    assert all(row["warm_start_rows"] == 1 and row["objective"] == "joint_limit_barrier" for row in by_family["R3"])


def test_missing_capability_is_not_invoked_or_promoted_to_pass() -> None:
    q = np.zeros((181, 6), dtype=np.float64)
    trajectory = ProcessTrajectory(joint_states=q, joint_lower_rad=np.full(6, -1.0), joint_upper_rad=np.full(6, 1.0))
    unavailable = AxisSpec(
        axis_id="base_translation:x:positive", perturbation_family=StressFamily.BASE_TRANSLATION,
        axis="X", coordinate_frame="base_link", direction=1, units="m", search_domain_max=0.0,
        coarse_intervals=1, refinement_tolerance=1e-5, operator_parameters={"direction_base": [1, 0, 0]},
        capability_status=CapabilityStatus.NOT_AVAILABLE, unavailable_reason="MoveIt base transform propagation absent",
    )
    called = False

    def evaluator(_):
        nonlocal called
        called = True
        return {}

    result = scan_axes_batched(trajectory, [unavailable], evaluator)[0]
    assert not called
    assert result["axis_status"] == "CAPABILITY_UNAVAILABLE"
    assert result["search_completeness"] == "NOT_RUN"
    assert result["estimated_raw_axis_margin"] is None


def test_joint_margin_profile_reports_controlling_joint_waypoint_and_side() -> None:
    lower = np.asarray([-2, -2, -2, -2, -2, -1], dtype=np.float64)
    upper = np.asarray([2, 2, 2, 2, 2, 1], dtype=np.float64)
    q = np.zeros((2, 6), dtype=np.float64)
    q[1, 5] = 0.8
    profile = normalized_joint_margin_profile(q, lower, upper)
    assert profile["minimum_joint_margin_rad"] == pytest.approx(0.2)
    assert profile["minimum_normalized_joint_margin"] == pytest.approx(0.1)
    assert profile["controlling_joint"] == "j6"
    assert profile["controlling_waypoint"] == 1
    assert profile["controlling_limit_side"] == "upper"
    with pytest.raises(ValueError, match="finite_shape_n_by_6"):
        normalized_joint_margin_profile(np.full((2, 6), np.inf), lower, upper)


def test_joint_axes_rank_in_radians_and_family_reports_keep_units_separate() -> None:
    axes = [
        {"specification": {"axis_id": "joint:j2:positive", "perturbation_family": "JOINT_STATE", "units": "rad"},
         "estimated_raw_axis_margin": {"value": 0.2}},
        {"specification": {"axis_id": "joint:j1:negative", "perturbation_family": "JOINT_STATE", "units": "rad"},
         "estimated_raw_axis_margin": {"value": 0.1}},
        {"specification": {"axis_id": "joint:j3:negative", "perturbation_family": "JOINT_STATE", "units": "rad"},
         "estimated_raw_axis_margin": {"value": 0.0}},
        {"specification": {"axis_id": "tcp:x:positive", "perturbation_family": "TCP_TRANSLATION", "units": "m"},
         "estimated_raw_axis_margin": {"value": 0.001}},
    ]
    assert [row["axis_id"] for row in rank_joint_axis_results(axes)] == [
        "joint:j3:negative", "joint:j1:negative", "joint:j2:positive",
    ]
    families = rank_axis_results_by_family(axes)
    assert set(families) == {"JOINT_STATE|rad", "TCP_TRANSLATION|m"}
    bad = [dict(axes[0], specification={**axes[0]["specification"], "units": "m"})]
    with pytest.raises(ValueError, match="radians"):
        rank_joint_axis_results(bad)


def test_full_domain_no_failure_is_complete_only_with_p2a_terminal_state() -> None:
    reference = {"status": "PASS"}
    variant = {"variant_id": "R0", "full_validation": {"status": "PASS"}}
    rows = []
    for index in range(42):
        available = index < 24
        is_joint = index < 12
        rows.append({
            "variant_id": "R0",
            "axis_id": f"joint:j{index // 2 + 1}:{'positive' if index % 2 == 0 else 'negative'}" if is_joint else f"axis:{index}",
            "capability_status": "AVAILABLE" if available else "NOT_AVAILABLE",
            "axis_status": "NO_FAILURE_WITHIN_SEARCH_DOMAIN" if available else "CAPABILITY_UNAVAILABLE",
            "search_completeness": "FULL_COARSE_DOMAIN_SAMPLED" if available else "NOT_RUN",
            "estimated_raw_axis_margin": None,
        })
    assert _campaign_status(reference, [variant], rows) == "COMPLETE_WITH_FORMAL_CAPABILITY_GAPS"
    rows[0]["search_completeness"] = "INCOMPLETE_CAPABILITY_GAP"
    assert _campaign_status(reference, [variant], rows) == "INCOMPLETE"


def test_result_and_provenance_contracts_preserve_unavailable_hardware_and_ccd() -> None:
    result = {key: None for key in RESULT_REQUIRED_FIELDS}
    result.update({
        "PROJECT": "FAIRINO_FR5", "STAGE": "P2-B2", "REDUNDANCY_VARIANTS": [],
        "COLLISION_METHOD": "adaptive_discrete_interpolation", "STRICT_SELF_CCD": "NOT_AVAILABLE",
        "HARDWARE_VALIDATION": "NOT_RUN", "HARDWARE_SAFETY_CERTIFIED": "NO",
    })
    assert validate_result_schema(result)
    result["STRICT_SELF_CCD"] = "PASS"
    assert not validate_result_schema(result)

    provenance = {
        "project": "FAIRINO_FR5", "stage": "P2-B2", "source_base_commit": "base",
        "execution_code_commit": "code", "branch": "codex/fr5-p2b2", "repository": "repo",
        "model_source_commit": "model", "runtime": {}, "inputs": {},
        "collision_method": "adaptive_discrete_interpolation",
    }
    assert validate_provenance_schema(provenance)
    provenance["collision_method"] = "continuous_collision_detection"
    assert not validate_provenance_schema(provenance)
