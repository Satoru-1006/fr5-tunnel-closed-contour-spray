"""Known-answer tests for the D65 hybrid certificate input gate."""

from __future__ import annotations

from copy import deepcopy

from tools.d65_hybrid_certificate import validate_authorized_binding, validate_profile_j


def valid_profile(states: int = 5) -> dict[str, object]:
    intervals = states - 1
    return {
        "input_state_count": states,
        "segments": intervals,
        "strict_current_target_valid": intervals,
        "successful_native_profiles": intervals,
        "error_invalid_input_count": 0,
        "other_error_count": 0,
        "overshoot_segment_count": 0,
        "extended_segment_count": 0,
        "max_abs_analytic_jerk_rad_s3": 8.0,
        "limits": {"max_jerk_rad_s3": [8.0] * 6},
        "jerk_truth": "NATIVE_RUCKIG_PROFILE_J",
        "passed": True,
    }


def test_profile_j_gate_accepts_complete_native_oracle() -> None:
    result = validate_profile_j(valid_profile(), state_count=5, jerk_bound=8.0)
    assert result["status"] == "VALID"
    assert result["errors"] == []


def test_profile_j_gate_rejects_missing_segment() -> None:
    profile = valid_profile()
    profile["successful_native_profiles"] = 3
    result = validate_profile_j(profile, state_count=5, jerk_bound=8.0)
    assert result["status"] == "BLOCKED_INVALID_INPUT"
    assert any(error["code"] == "PROFILE_COUNT_MISMATCH" for error in result["errors"])


def test_profile_j_gate_rejects_analytic_jerk_excess() -> None:
    profile = deepcopy(valid_profile())
    profile["max_abs_analytic_jerk_rad_s3"] = 8.000000000000002
    result = validate_profile_j(profile, state_count=5, jerk_bound=8.0)
    assert result["status"] == "BLOCKED_INVALID_INPUT"
    assert any(error["code"] == "PROFILE_JERK_BOUND_VIOLATION" for error in result["errors"])


def test_profile_j_gate_rejects_non_authoritative_label() -> None:
    profile = valid_profile()
    profile["jerk_truth"] = "FINITE_DIFFERENCE_DIAGNOSTIC"
    result = validate_profile_j(profile, state_count=5, jerk_bound=8.0)
    assert result["status"] == "BLOCKED_INVALID_INPUT"
    assert any(error["code"] == "PROFILE_NOT_AUTHORITATIVE" for error in result["errors"])


def test_authorized_binding_rejects_cross_trajectory_bytes() -> None:
    result = validate_authorized_binding("adversarial_0100", "0" * 64, "0" * 64, 7005)
    assert result["status"] == "BLOCKED_INVALID_INPUT"
    assert {error["code"] for error in result["errors"]} == {"AUTHORIZED_IDENTITY_MISMATCH"}


def test_authorized_binding_rejects_unknown_case() -> None:
    result = validate_authorized_binding("other", "0" * 64, "0" * 64, 5)
    assert result["status"] == "BLOCKED_INVALID_INPUT"
    assert result["errors"][0]["code"] == "UNKNOWN_TRAJECTORY_BINDING"
