from __future__ import annotations

import numpy as np

from src.stage3_h12_trajectory_repair import (
    RepairLimits,
    REPAIR_METHOD,
    constraint_report,
    repair_magnitude,
    repair_role,
    repair_window,
    semantic_prediction_sha256,
)


def causal_case(count: int = 2):
    history_times = np.arange(16, dtype=float)[None, :] * 0.1
    history_times = np.repeat(history_times, count, axis=0)
    history_positions = np.zeros((count, 16, 6), dtype=float)
    history_velocity = np.zeros((count, 6), dtype=float)
    history_acceleration = np.zeros((count, 6), dtype=float)
    future_times = np.arange(16, 24, dtype=float)[None, :] * 0.1
    future_times = np.repeat(future_times, count, axis=0)
    raw = np.zeros((count, 8, 6), dtype=float)
    raw[:, :, 0] = np.linspace(0.0, 1.0, 8)
    return raw, history_positions, history_times, future_times, history_velocity, history_acceleration


def test_repair_is_deterministic_and_causal():
    raw, history, history_times, future_times, velocity, acceleration = causal_case()
    first = repair_role(raw, history[:, -1], history, history_times, future_times, history_velocities=velocity, history_accelerations=acceleration)
    second = repair_role(raw, history[:, -1], history, history_times, future_times, history_velocities=velocity, history_accelerations=acceleration)
    assert REPAIR_METHOD.startswith("causal_")
    assert np.array_equal(first["positions"], second["positions"])
    assert np.array_equal(first["velocities"], second["velocities"])
    assert first["POSITION_REPAIR_FAILURES"] == 0


def test_dynamic_projection_meets_frozen_discrete_limits():
    raw, history, history_times, future_times, velocity, acceleration = causal_case(1)
    result = repair_role(raw, history[:, -1], history, history_times, future_times, history_velocities=velocity, history_accelerations=acceleration)
    report = constraint_report(result["positions"], history[:, -1], history_times, future_times)
    assert report["POST_REPAIR_POSITION_LIMIT_VIOLATIONS"] == 0
    assert report["POST_REPAIR_VELOCITY_LIMIT_VIOLATIONS"] == 0
    assert report["POST_REPAIR_ACCELERATION_LIMIT_VIOLATIONS"] == 0
    assert report["POST_REPAIR_JERK_LIMIT_VIOLATIONS"] == 0
    assert report["POST_REPAIR_HARD_CONSTRAINT_VIOLATIONS"] == 0


def test_controlled_stop_window_fails_closed_instead_of_crossing_boundary():
    raw, history, history_times, future_times, velocity, acceleration = causal_case(1)
    result = repair_window(raw[0], history[0, -1], history[0], history_times[0], future_times[0], history_velocity=velocity[0], history_acceleration=acceleration[0], controlled_stop=True)
    assert result.success is False
    assert result.failure_reason == "controlled_stop_window_requires_boundary_isolation"


def test_repair_magnitude_reports_unclipped_maximum_and_changed_states():
    raw, history, history_times, future_times, velocity, acceleration = causal_case(1)
    result = repair_role(raw, history[:, -1], history, history_times, future_times, history_velocities=velocity, history_accelerations=acceleration)
    magnitude = repair_magnitude(raw, result["positions"], ["family_a"])
    assert magnitude["REPAIR_DELTA_Q_MAX_ABS_RAD"] >= magnitude["REPAIR_DELTA_Q_MEAN_ABS_RAD"]
    assert magnitude["reported_max_is_unclipped"] is True
    assert magnitude["REPAIR_CHANGED_STATE_COUNT"] > 0


def test_segment_boundary_is_not_repaired_across_families():
    raw, history, history_times, future_times, velocity, acceleration = causal_case(2)
    result = repair_role(raw, history[:, -1], history, history_times, future_times, history_velocities=velocity, history_accelerations=acceleration)
    assert result["POSITION_REPAIR_ATTEMPTS"] == 2
    assert result["REPAIR_METHOD"] == REPAIR_METHOD


def test_frozen_limits_are_not_redefined():
    limits = RepairLimits()
    assert np.array_equal(limits.velocity_abs, np.asarray([0.4725, 0.4725, 0.4725, 0.48, 0.48, 0.48]))
    assert np.array_equal(limits.acceleration_abs, np.asarray([0.105] * 6))
    assert np.array_equal(limits.jerk_abs, np.asarray([8.0] * 6))


def test_prediction_semantic_hash_is_stable():
    raw, history, history_times, future_times, velocity, acceleration = causal_case()
    result = repair_role(raw, history[:, -1], history, history_times, future_times, history_velocities=velocity, history_accelerations=acceleration)
    payload = {"TRAIN": result["positions"], "VALIDATION": result["positions"], "TEST": result["positions"], "GENERALIZATION": result["positions"]}
    assert semantic_prediction_sha256(payload) == semantic_prediction_sha256(payload)

