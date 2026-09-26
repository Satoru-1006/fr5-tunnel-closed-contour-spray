import numpy as np
import pytest

from src.stage3_h12_r6 import (
    boundary_state_audit,
    classify_acceleration_root_cause,
    derivative_consistency_audit,
    distribution,
    reconstruct_finite_difference_acceleration,
    reconstruct_local_derivative_state,
)


def _velocities(count=4):
    values = np.zeros((count, 6), dtype=float)
    values[:, 0] = np.array([0.0, 0.02, 0.05, 0.09][:count])
    values[:, 1] = np.linspace(0.0, 0.03, count)
    return values


def test_nonuniform_finite_difference_reconstruction_is_deterministic():
    times = np.array([0.0, 0.1, 0.3, 0.6])
    velocities = np.zeros((4, 6), dtype=float)
    velocities[:, 0] = [0.0, 0.1, 0.3, 0.6]
    result = reconstruct_finite_difference_acceleration(times, velocities)
    np.testing.assert_allclose(result[:, 0], [1.0, 1.0, 1.0, 1.0])
    np.testing.assert_array_equal(result, reconstruct_finite_difference_acceleration(times, velocities))


def test_malformed_or_nonmonotonic_derivative_state_fails_closed():
    with pytest.raises(ValueError, match="strictly increasing"):
        reconstruct_finite_difference_acceleration([0.0, 0.1, 0.1], _velocities(3))
    with pytest.raises(ValueError, match="malformed"):
        reconstruct_finite_difference_acceleration([0.0, 0.1], np.zeros((2, 5)))


def test_stored_waypoint_acceleration_is_separated_from_finite_difference():
    times = np.array([0.0, 0.1, 0.2, 0.3])
    velocities = _velocities()
    stored = reconstruct_finite_difference_acceleration(times, velocities)
    stored[1, 2] = 1.3
    stored[2, 4] = -1.4
    audit = derivative_consistency_audit(times, velocities, stored)
    assert audit["stored_waypoint_acceleration_violation_count"] == 2
    assert audit["finite_difference_acceleration_violation_count"] == 0
    assert audit["stale_waypoint_acceleration_candidate_count"] == 2
    repaired, evidence = reconstruct_local_derivative_state(times, velocities, stored)
    assert evidence["changed_cell_count"] == 2
    np.testing.assert_array_equal(repaired[:, :2], stored[:, :2])
    np.testing.assert_allclose(repaired, reconstruct_finite_difference_acceleration(times, velocities))


def test_real_finite_difference_violation_is_not_repaired():
    times = np.array([0.0, 0.1, 0.2])
    velocities = np.zeros((3, 6), dtype=float)
    velocities[1, 0] = 0.2
    stored = np.zeros_like(velocities)
    with pytest.raises(ValueError, match="finite_difference"):
        reconstruct_local_derivative_state(times, velocities, stored)


def test_root_cause_and_required_distributions_are_deterministic():
    cause = classify_acceleration_root_cause(stored_waypoint_violations=40, finite_difference_violations=0, native_analytic_violations=0)
    assert cause == {"classification": "STALE_WAYPOINT_ACCELERATION", "confidence": "HIGH"}
    rows = [
        {"trajectory_family_id": "f1", "segment_id": 1, "primitive_id": "p1", "joint_index": 0, "boundary_type": "interior", "spray_state": "SPRAY_OFF"},
        {"trajectory_family_id": "f1", "segment_id": 1, "primitive_id": "p2", "joint_index": 2, "boundary_type": "primitive_start", "spray_state": "SPRAY_OFF"},
    ]
    first = distribution(rows)
    second = distribution(rows)
    assert first == second
    assert first["ACCEL_VIOLATIONS_BY_FAMILY"] == {"f1": 2}
    assert first["ACCEL_VIOLATIONS_BY_JOINT"] == {"0": 1, "2": 1}


def test_zero_velocity_boundary_with_nonzero_acceleration_is_observation_not_failure():
    state = {"position_rad": [0.0] * 6, "velocity_rad_s": [0.0] * 6, "acceleration_rad_s2": [0.2] + [0.0] * 5, "timestamp_s": 0.0}
    result = boundary_state_audit([{"start_state": state, "end_state": state}])
    assert result["status"] == "PASSED"
    assert result["zero_velocity_nonzero_acceleration_observation_count"] == 2
    assert result["boundary_derivative_inconsistency_count"] == 0

