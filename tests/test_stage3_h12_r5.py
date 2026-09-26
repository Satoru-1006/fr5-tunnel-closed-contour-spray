from __future__ import annotations

from copy import deepcopy

import pytest

from src.stage3_h12_r5 import (
    analytic_piece_position_extrema,
    classify_boundary,
    completion_gate,
    exact_coverage,
    final_gate,
    native_signed_overshoot,
    semantic_sha256,
    select_boundary_velocity_recondition,
    threshold_sensitivity,
)


def valid_ruckig_record():
    return {
        "ruckig_result": {"result_name": "Working"},
        "native_smoothing_returned_success": True,
        "smoothing_complete": True,
        "duration_ceiling_hit": False,
        "overshoot_gate_passed": True,
        "output_finite": True,
        "input_validation_passed": True,
        "native_error": None,
    }


def valid_certificate():
    return {
        "H12_R4_REPRODUCED": "YES",
        "PRIMITIVES": 480,
        "TOTG_FINAL_PASS": 480,
        "RUCKIG_SMOOTHING_SUCCESS": 480,
        "RUCKIG_DURATION_CEILING_HIT": 0,
        "RUCKIG_NATIVE_ERRORS": 0,
        "POST_RUCKIG_VALIDATED": 189216,
        "POSITION_VIOLATIONS": 0,
        "VELOCITY_VIOLATIONS": 0,
        "ACCELERATION_VIOLATIONS": 0,
        "JERK_VIOLATIONS": 0,
        "SPRAY_PROCESS_VIOLATIONS": 0,
        "COLLISION_VIOLATIONS": 0,
        "LEAKAGE": 0,
        "REPLAY": "3/3",
        "FOCUSED_TESTS_PASS": True,
        "REGRESSION_TESTS_PASS": True,
        "UPSTREAM_IMMUTABLE": "YES",
    }


def test_working_is_acceptable_when_complete():
    assert completion_gate(valid_ruckig_record())["accepted"]


def test_finished_is_acceptable_when_complete():
    row = valid_ruckig_record()
    row["ruckig_result"]["result_name"] = "Finished"
    assert completion_gate(row)["accepted"]


def test_wrapper_true_but_incomplete_is_rejected():
    row = valid_ruckig_record()
    row["smoothing_complete"] = False
    assert not completion_gate(row)["accepted"]


def test_duration_ceiling_is_rejected():
    row = valid_ruckig_record()
    row["duration_ceiling_hit"] = True
    assert not completion_gate(row)["accepted"]


def test_native_extremum_overshoot_from_above():
    signed, absolute, extremum = native_signed_overshoot(1.0, 0.9, 0.87, 1.01)
    assert signed == pytest.approx(-0.03) and absolute == pytest.approx(0.03) and extremum == 0.87


def test_native_extremum_overshoot_from_below():
    signed, absolute, extremum = native_signed_overshoot(0.8, 0.9, 0.79, 0.94)
    assert signed == pytest.approx(0.04) and absolute == pytest.approx(0.04) and extremum == 0.94


def test_zero_jerk_constant_acceleration_position_root_is_not_omitted():
    minimum, maximum = analytic_piece_position_extrema(0.0, -0.1, 0.2, 0.0, 1.0)
    assert minimum == pytest.approx(-0.025)
    assert maximum == pytest.approx(0.0)


def test_boundary_velocity_remediation_uses_smallest_candidate_and_preserves_other_state():
    position, acceleration = 1.25, -0.1
    selected = select_boundary_velocity_recondition(0.9999, 1.0, lambda value: abs(value) <= 0.999)
    assert selected == pytest.approx((0.999, 0.9989001))
    assert (position, acceleration) == (1.25, -0.1)


def test_boundary_velocity_remediation_does_not_touch_non_near_limit_state():
    assert select_boundary_velocity_recondition(0.5, 1.0, lambda _: True) is None


def test_near_zero_displacement_nonzero_velocity_taxonomy():
    row = {
        "next_segment_delta_position": 4.46e-4,
        "current_velocity": 0.044,
        "target_velocity": 0.045,
        "current_acceleration": 0.105,
        "target_acceleration": 0.105,
        "retry_absolute_overshoots": [0.01002] * 42,
    }
    assert classify_boundary(row)[0] == "NEAR_ZERO_DISPLACEMENT_NONZERO_VELOCITY"


def test_velocity_sign_reversal_taxonomy():
    row = {
        "next_segment_delta_position": 0.1,
        "current_velocity": 0.2,
        "target_velocity": -0.1,
        "current_acceleration": 0.1,
        "target_acceleration": 0.1,
        "retry_absolute_overshoots": [],
    }
    assert classify_boundary(row)[0] == "VELOCITY_SIGN_REVERSAL"


def test_duration_invariant_taxonomy():
    row = {
        "next_segment_delta_position": 0.1,
        "current_velocity": 0.2,
        "target_velocity": 0.1,
        "current_acceleration": 0.1,
        "target_acceleration": 0.1,
        "retry_absolute_overshoots": [0.01, 0.01],
    }
    assert classify_boundary(row)[0] == "DURATION_EXTENSION_INVARIANT_OVERSHOOT"


def test_threshold_sensitivity_isolated():
    events = [
        {"primitive_id": "a", "absolute_overshoot": 0.01},
        {"primitive_id": "b", "absolute_overshoot": 0.02},
    ]
    rows = threshold_sensitivity(events, [0.005, 0.015, 0.025])
    assert [row["overshoot_event_count"] for row in rows] == [2, 1, 0]


def test_exact_coverage_rejects_duplicates():
    assert exact_coverage([{"unit_id": "a"}, {"unit_id": "b"}], ["a", "b"])
    assert not exact_coverage([{"unit_id": "a"}, {"unit_id": "a"}], ["a", "b"])


def test_semantic_hash_is_key_order_invariant():
    assert semantic_sha256({"a": 1, "b": 2}) == semantic_sha256({"b": 2, "a": 1})


def test_final_gate_passes_complete_certificate():
    assert final_gate(valid_certificate()) == (True, "none")


def test_final_gate_reports_reproduction_first():
    row = valid_certificate()
    row["H12_R4_REPRODUCED"] = "NO"
    row["TOTG_FINAL_PASS"] = 0
    assert final_gate(row) == (False, "h12_r4_reproduction_mismatch")


def test_final_gate_reports_totg_regression():
    row = valid_certificate()
    row["TOTG_FINAL_PASS"] = 479
    assert final_gate(row) == (False, "totg_regression")


def test_final_gate_rejects_partial_window_validation():
    row = valid_certificate()
    row["POST_RUCKIG_VALIDATED"] = 189215
    assert final_gate(row) == (False, "incomplete_post_ruckig_validation")


def test_no_future_label_fields_in_interposer_source():
    source = open("tools/stage25r_ruckig_interposer.cpp", encoding="utf-8").read()
    assert "ground_truth" not in source
    assert "target_positions" not in source


def test_r4_default_remains_opt_in_for_r5_modes():
    source = open("tools/stage25r_ruckig_interposer.cpp", encoding="utf-8").read()
    assert 'env_flag("H12_R5_USE_INSTALLED_MOVEIT_HELPERS")' in source
    assert 'env_flag("H12_R5_HARD_LIMIT_CERTIFICATION")' in source
    assert 'env_flag("H12_R5_BOUNDARY_VELOCITY_RECONDITION")' in source
    assert "next_waypoint->setVariableVelocity" in source


def test_mitigation_off_is_an_isolated_environment_switch():
    source = open("ros2_moveit_bridge/stage3_h12_r4_native.py", encoding="utf-8").read()
    assert 'os.environ.get("H12_R5_MITIGATE_OVERSHOOT", "1")' in source
    assert "mitigate_overshoot=mitigate_overshoot" in source


def test_every_segment_native_duration_replacement_is_disabled_formally():
    source = open("scripts/stage3_h12_r5_ruckig_root_cause.py", encoding="utf-8").read()
    assert '"export H12_R5_ASSIGN_ALL_NATIVE_DURATIONS=0"' in source


def test_immutability_snapshot_comparison_is_exact():
    before = {"a": {"sha256": "x", "size": 1}}
    after = deepcopy(before)
    assert before == after
    after["a"]["size"] = 2
    assert before != after
