from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
import tempfile

import numpy as np

from src.process_aware_stress import (
    ApplicationStatus,
    CapabilityStatus,
    D46_CAPABILITY_PROFILE,
    FailureCode,
    LegacyMappingStatus,
    ProcessTrajectory,
    ResultStatus,
    StressFamily,
    StressOperator,
    ValidityCheck,
    aggregate_trajectory_validity,
    apply_stress,
    check_joint_limits,
    map_legacy_d46_case,
)
from tools.stage4a_system_benchmark import FAMILY_COUNTS, make_cases


def _trajectory() -> ProcessTrajectory:
    poses = np.asarray([
        [1.0, 2.0, 3.0, 0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)],
        [2.0, 2.0, 3.0, 0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)],
    ])
    q = np.zeros((2, 6))
    return ProcessTrajectory(
        tcp_poses=poses,
        joint_states=q,
        timestamps_s=np.asarray([0.0, 1.0]),
        surface_normals=np.asarray([[0.0, 0.0, 2.0], [0.0, 0.0, 1.0]]),
        surface_offsets_m=np.asarray([0.0, 0.0]),
        stand_off_m=np.asarray([0.26, 0.26]),
        joint_lower_rad=np.full(6, -1.0),
        joint_upper_rad=np.full(6, 1.0),
    )


def test_zero_amplitude_is_identity_for_every_operator_family() -> None:
    trajectory = _trajectory()
    units = {
        StressFamily.BASE_TRANSLATION: "m", StressFamily.BASE_ROTATION: "rad",
        StressFamily.TCP_TRANSLATION: "m", StressFamily.TCP_ROTATION: "rad",
        StressFamily.SURFACE_NORMAL: "rad", StressFamily.SURFACE_OFFSET: "m",
        StressFamily.STANDOFF: "m", StressFamily.JOINT_STATE: "rad",
        StressFamily.TIMING_SCALE: "ratio", StressFamily.REGRESSION_CONTROL: "1",
    }
    for family in StressFamily:
        operator = StressOperator(
            operator_id=f"zero:{family.value}", family=family, amplitude=0.0,
            units=units[family], target_scope="fixture", seed=7,
        )
        result = apply_stress(trajectory, operator)
        assert result.status == ApplicationStatus.APPLIED
        assert result.trajectory is trajectory


def test_deterministic_replay_serializes_identically() -> None:
    trajectory = _trajectory()
    operator = StressOperator(
        operator_id="tcp-local-x", family=StressFamily.TCP_TRANSLATION,
        amplitude=0.01, units="m", target_scope="all target TCP poses",
        parameters={"direction_tcp": [1.0, 0.0, 0.0]},
    )
    first = apply_stress(trajectory, operator)
    second = apply_stress(trajectory, operator)
    assert first.serialize() == second.serialize()

    seeded_noise = StressOperator(
        operator_id="seeded-joint-noise", family=StressFamily.JOINT_STATE,
        amplitude=0.001, units="rad", target_scope="all joint states", seed=981,
        parameters={"mode": "seeded_gaussian"},
    )
    assert apply_stress(trajectory, seeded_noise).serialize() == apply_stress(trajectory, seeded_noise).serialize()


def test_surface_normal_is_renormalized_after_rotation() -> None:
    trajectory = _trajectory()
    operator = StressOperator(
        operator_id="normal-tilt", family=StressFamily.SURFACE_NORMAL,
        amplitude=0.2, units="rad", target_scope="all surface normals",
        parameters={"axis_base": [1.0, 0.0, 0.0]},
    )
    result = apply_stress(trajectory, operator)
    assert result.status == ApplicationStatus.APPLIED
    assert np.allclose(np.linalg.norm(result.trajectory.surface_normals, axis=1), 1.0)


def test_base_rotation_and_tcp_rotation_keep_their_frame_semantics() -> None:
    trajectory = ProcessTrajectory(tcp_poses=np.asarray([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]))
    base = apply_stress(trajectory, StressOperator(
        operator_id="base-z-90", family=StressFamily.BASE_ROTATION,
        amplitude=math.pi / 2.0, units="rad", target_scope="all target TCP poses",
        parameters={"axis_base": [0.0, 0.0, 1.0]},
    ))
    assert np.allclose(base.trajectory.tcp_poses[0, :3], [0.0, 1.0, 0.0], atol=1e-12)
    assert np.allclose(base.trajectory.tcp_poses[0, 3:7], [0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)])

    local = apply_stress(trajectory, StressOperator(
        operator_id="tcp-z-90", family=StressFamily.TCP_ROTATION,
        amplitude=math.pi / 2.0, units="rad", target_scope="tool orientation",
        parameters={"axis_tcp": [0.0, 0.0, 1.0]},
    ))
    assert np.allclose(local.trajectory.tcp_poses[0, :3], [1.0, 0.0, 0.0])
    assert np.allclose(local.trajectory.tcp_poses[0, 3:7], [0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)])


def test_process_offsets_and_timing_scale_are_applied_without_filling_missing_data() -> None:
    trajectory = _trajectory()
    surface = apply_stress(trajectory, StressOperator(
        operator_id="surface-offset", family=StressFamily.SURFACE_OFFSET,
        amplitude=-0.002, units="m", target_scope="all surface offsets",
    ))
    standoff = apply_stress(trajectory, StressOperator(
        operator_id="standoff", family=StressFamily.STANDOFF,
        amplitude=0.003, units="m", target_scope="all stand-off values",
    ))
    timing = apply_stress(trajectory, StressOperator(
        operator_id="shorten-time", family=StressFamily.TIMING_SCALE,
        amplitude=-0.5, units="ratio", target_scope="all trajectory timestamps",
        parameters={"scale_factor": 0.5},
    ))
    assert np.allclose(surface.trajectory.surface_offsets_m, [-0.002, -0.002])
    assert np.allclose(standoff.trajectory.stand_off_m, [0.263, 0.263])
    assert np.allclose(timing.trajectory.timestamps_s, [0.0, 0.5])
    assert apply_stress(ProcessTrajectory(), StressOperator(
        operator_id="missing-standoff", family=StressFamily.STANDOFF,
        amplitude=0.001, units="m", target_scope="stand-off",
    )).status == ApplicationStatus.NOT_AVAILABLE


def test_base_and_tcp_translation_use_declared_frames() -> None:
    trajectory = _trajectory()
    base = apply_stress(trajectory, StressOperator(
        operator_id="base-x", family=StressFamily.BASE_TRANSLATION,
        amplitude=0.01, units="m", target_scope="all target TCP poses",
        parameters={"direction_base": [1.0, 0.0, 0.0]},
    ))
    assert np.allclose(base.trajectory.tcp_poses[:, :3], trajectory.tcp_poses[:, :3] + [0.01, 0.0, 0.0])

    tcp = apply_stress(trajectory, StressOperator(
        operator_id="tcp-x", family=StressFamily.TCP_TRANSLATION,
        amplitude=0.01, units="m", target_scope="all target TCP poses",
        parameters={"direction_tcp": [1.0, 0.0, 0.0]},
    ))
    assert np.allclose(tcp.trajectory.tcp_poses[:, :3], trajectory.tcp_poses[:, :3] + [0.0, 0.01, 0.0])


def test_joint_limit_crossing_is_reported_without_clamping() -> None:
    trajectory = _trajectory()
    q = np.array(trajectory.joint_states, copy=True)
    q[0, 2] = 0.99
    near_limit = ProcessTrajectory(
        joint_states=q, joint_lower_rad=trajectory.joint_lower_rad,
        joint_upper_rad=trajectory.joint_upper_rad,
    )
    result = apply_stress(near_limit, StressOperator(
        operator_id="joint-j3-positive", family=StressFamily.JOINT_STATE,
        amplitude=0.02, units="rad", target_scope="joint state",
        parameters={"direction": [0, 0, 1, 0, 0, 0]},
    ))
    assert result.trajectory.joint_states[0, 2] == 1.01
    checks = check_joint_limits(result.trajectory)
    assert checks[0].status == ResultStatus.FAIL
    assert FailureCode.JOINT_LIMIT in checks[0].failure_codes


def test_unavailable_capability_cannot_aggregate_to_pass() -> None:
    assert D46_CAPABILITY_PROFILE["continuous_robot_world_collision_api"].status == CapabilityStatus.AVAILABLE
    assert D46_CAPABILITY_PROFILE["continuous_self_collision"].status == CapabilityStatus.NOT_AVAILABLE
    assert D46_CAPABILITY_PROFILE["model_based_required_torque"].status == CapabilityStatus.NOT_AVAILABLE
    assert D46_CAPABILITY_PROFILE["clearance_acceptance_threshold"].status == CapabilityStatus.UNKNOWN
    result = aggregate_trajectory_validity(
        [ValidityCheck(ResultStatus.PASS)], [],
        {"continuous_self_collision": CapabilityStatus.NOT_AVAILABLE},
    )
    assert result.trajectory_validity == ResultStatus.UNKNOWN
    assert FailureCode.CAPABILITY_UNAVAILABLE in {event.code for event in result.failures}


def test_valid_points_do_not_hide_invalid_transition() -> None:
    result = aggregate_trajectory_validity(
        [ValidityCheck(ResultStatus.PASS), ValidityCheck(ResultStatus.PASS)],
        [ValidityCheck(ResultStatus.FAIL, (FailureCode.ENV_COLLISION,))],
    )
    assert result.point_validity == ResultStatus.PASS
    assert result.transition_validity == ResultStatus.FAIL
    assert result.trajectory_validity == ResultStatus.FAIL
    assert result.failures[0].index == 0

    incomplete_but_failing = aggregate_trajectory_validity(
        [ValidityCheck(ResultStatus.PASS), ValidityCheck(ResultStatus.PASS), ValidityCheck(ResultStatus.PASS)],
        [ValidityCheck(ResultStatus.FAIL, (FailureCode.ENV_COLLISION,))],
    )
    assert incomplete_but_failing.transition_validity == ResultStatus.FAIL
    assert incomplete_but_failing.trajectory_validity == ResultStatus.FAIL


def test_legacy_d46_mapping_preserves_families_and_case_counts() -> None:
    cases = [
        {"case_id": "normal_0000", "family": "NORMAL", "index": 0, "seed": 100000,
         "generation": "smooth_bounded_nominal_variation", "time_scale": 0.95},
        {"case_id": "boundary_0000", "family": "BOUNDARY", "index": 0, "seed": 110000,
         "generation": "bounded_urdf_position_boundary_push", "time_scale": 0.45},
        {"case_id": "collision_sensitive_0000", "family": "COLLISION_SENSITIVE", "index": 0,
         "seed": 120000, "generation": "smooth_excursion_to_known_self_collision_state", "time_scale": 0.8},
        {"case_id": "adversarial_0000", "family": "ADVERSARIAL", "index": 0, "seed": 130000,
         "generation": "deliberate_j3_position_limit_exceedance", "time_scale": 0.5},
        {"case_id": "adversarial_0020", "family": "ADVERSARIAL", "index": 20, "seed": 130020,
         "generation": "high_curvature_short_horizon_stress", "time_scale": 0.22},
        {"case_id": "perturbation_0000", "family": "PERTURBATION", "index": 0, "seed": 140000,
         "generation": "fixed_seed_joint_and_target_pose_perturbation", "time_scale": 0.8,
         "target_shift_m": [0.001, -0.002, 0.0]},
        {"case_id": "regression_0000", "family": "REGRESSION", "index": 0, "seed": 150000,
         "generation": "actual_frozen_stage3_post_ruckig_replay", "time_scale": 1.0},
    ]
    mapped = [map_legacy_d46_case(case) for case in cases]
    assert [item.family for item in mapped] == [case["family"] for case in cases]
    assert all(item.status != LegacyMappingStatus.LEGACY_UNMAPPED for item in mapped)
    assert mapped[5].status == LegacyMappingStatus.PARTIALLY_MAPPED
    assert map_legacy_d46_case({"case_id": "unknown_0000", "family": "UNKNOWN"}).status == LegacyMappingStatus.LEGACY_UNMAPPED
    assert [item.operators[0].family for item in mapped] == [
        StressFamily.JOINT_STATE, StressFamily.JOINT_STATE, StressFamily.JOINT_STATE,
        StressFamily.JOINT_STATE, StressFamily.JOINT_STATE, StressFamily.JOINT_STATE,
        StressFamily.REGRESSION_CONTROL,
    ]
    assert FAMILY_COUNTS == {
        "NORMAL": 200, "BOUNDARY": 200, "COLLISION_SENSITIVE": 200,
        "ADVERSARIAL": 200, "PERTURBATION": 150, "REGRESSION": 50,
    }
    assert sum(FAMILY_COUNTS.values()) == 1000
    assert any(op.family == StressFamily.TIMING_SCALE for op in mapped[1].operators)
    assert any(op.family == StressFamily.BASE_TRANSLATION for op in mapped[5].operators)
    boundary_operator = mapped[1].operators[0]
    pushed = apply_stress(_trajectory(), boundary_operator)
    assert math.isclose(float(np.max(pushed.trajectory.joint_states[:, 0])), 1.0 - 0.0001, abs_tol=1e-12)
    timing_operator = next(op for op in mapped[1].operators if op.family == StressFamily.TIMING_SCALE)
    assert np.allclose(apply_stress(_trajectory(), timing_operator).trajectory.timestamps_s, [0.0, 0.45])


def test_legacy_d46_generator_is_deterministic_and_preserves_1000_case_counts() -> None:
    base_q = np.zeros((181, 6), dtype=np.float64)
    lower = np.full(6, -math.pi, dtype=np.float64)
    upper = np.full(6, math.pi, dtype=np.float64)
    base_q[:, 2] = upper[2] - 0.005
    pre_q = np.full((181, 6), 0.001, dtype=np.float64)
    stable_fields = (
        "case_id", "family", "index", "seed", "generation", "time_scale",
        "target_shift_m", "actual_post_ruckig", "audit_only_perturbation",
        "hard_invalid_input_by_design", "trajectory_sha256",
    )
    with tempfile.TemporaryDirectory(prefix="fr5_p1_d46_") as scratch:
        first = make_cases(Path(scratch) / "first", base_q, pre_q, lower, upper)
        second = make_cases(Path(scratch) / "second", base_q, pre_q, lower, upper)
    assert len(first) == len(second) == 1000
    assert Counter(case["family"] for case in first) == FAMILY_COUNTS
    assert [[case[field] for field in stable_fields] for case in first] == [
        [case[field] for field in stable_fields] for case in second
    ]
