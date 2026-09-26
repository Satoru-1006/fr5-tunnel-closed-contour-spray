"""Pure fail-closed gate semantics for Stage 3 H7.7."""

from __future__ import annotations

from typing import Any, Mapping


def evaluate_h7_7_gate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "wrapper_boolean": candidate.get("wrapper_boolean") is True,
        "smoothing_complete": candidate.get("smoothing_complete") is True,
        "duration_ceiling_not_reached": candidate.get("duration_ceiling_reached") is False,
        "native_result_success": candidate.get("native_result") in {"Working", "Finished"},
        "strict_current_target_valid": candidate.get("strict_current_target_valid") is True,
        "no_native_input_error": int(candidate.get("native_input_errors", -1)) == 0,
        "no_native_other_error": int(candidate.get("native_other_errors", -1)) == 0,
        "positions_immutable": candidate.get("positions_changed") is False,
        "no_unauthorized_stop": candidate.get("unauthorized_stop_added") is False,
        "position_limits": int(candidate.get("position_limit_violations", -1)) == 0,
        "velocity_limits": int(candidate.get("velocity_limit_violations", -1)) == 0,
        "acceleration_limits": int(candidate.get("acceleration_limit_violations", -1)) == 0,
        "jerk_limits": int(candidate.get("jerk_limit_violations", -1)) == 0,
        "process_tolerance": int(candidate.get("process_violations", -1)) == 0,
        "collision": int(candidate.get("collision_violations", -1)) == 0,
        "deterministic_replay": candidate.get("deterministic_replay") == "3/3",
    }
    blocker_order = [
        ("positions_immutable", "joint_position_path_changed"),
        ("no_unauthorized_stop", "unauthorized_stop_added"),
        ("strict_current_target_valid", "strict_ruckig_invalid_input"),
        ("no_native_input_error", "raw_ruckig_error_invalid_input"),
        ("no_native_other_error", "raw_ruckig_other_error"),
        ("wrapper_boolean", "moveit_wrapper_false"),
        ("smoothing_complete", "smoothing_incomplete"),
        ("duration_ceiling_not_reached", "duration_ceiling_reached"),
        ("native_result_success", "native_ruckig_result_error"),
        ("position_limits", "position_limit_violation"),
        ("velocity_limits", "velocity_limit_violation"),
        ("acceleration_limits", "acceleration_limit_violation"),
        ("jerk_limits", "jerk_limit_violation"),
        ("process_tolerance", "process_tolerance_violation"),
        ("collision", "collision_violation"),
        ("deterministic_replay", "nondeterministic_replay"),
    ]
    first_blocker = next((reason for key, reason in blocker_order if not checks[key]), None)
    return {"passed": first_blocker is None, "first_blocker": first_blocker, "checks": checks}


def passing_fixture() -> dict[str, Any]:
    return {
        "wrapper_boolean": True, "smoothing_complete": True, "duration_ceiling_reached": False,
        "native_result": "Working", "strict_current_target_valid": True,
        "native_input_errors": 0, "native_other_errors": 0,
        "positions_changed": False, "unauthorized_stop_added": False,
        "position_limit_violations": 0, "velocity_limit_violations": 0,
        "acceleration_limit_violations": 0, "jerk_limit_violations": 0,
        "process_violations": 0, "collision_violations": 0,
        "deterministic_replay": "3/3",
    }
