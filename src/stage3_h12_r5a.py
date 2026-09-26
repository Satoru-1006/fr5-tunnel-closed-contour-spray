"""Pure fail-closed helpers for Stage 3 H12-R5A semantic closure."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence


def semantic_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def changed_indices(before: Sequence[Any], after: Sequence[Any], tolerance: float = 1.0e-12) -> list[int]:
    if len(before) != len(after):
        raise ValueError("array length changed")

    def differs(left: Any, right: Any) -> bool:
        if isinstance(left, list) and isinstance(right, list):
            return len(left) != len(right) or any(differs(a, b) for a, b in zip(left, right))
        return not math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)

    return [index for index, (left, right) in enumerate(zip(before, after)) if differs(left, right)]


def classify_installed_helper_semantics(probes: Sequence[Mapping[str, Any]]) -> str:
    """Classify only directly observable mutations; ambiguous behavior is UNKNOWN."""
    if len(probes) < 2:
        return "UNKNOWN"
    waypoint_local = True
    global_all_segments = True
    observed_arguments: set[int] = set()
    for probe in probes:
        argument = int(probe["second_argument"])
        waypoint_count = int(probe["num_waypoints"])
        observed_arguments.add(argument)
        durations = list(probe.get("duration_changed_indices") or [])
        positions = list(probe.get("position_changed_indices") or [])
        velocities = list(probe.get("velocity_changed_indices") or [])
        accelerations = list(probe.get("acceleration_changed_indices") or [])
        target = argument + 1
        waypoint_local &= (
            target < waypoint_count
            and durations == [target]
            and not positions
            and all(index == target for index in velocities)
            and all(index == target for index in accelerations)
            and target in velocities
            and target in accelerations
        )
        global_all_segments &= durations == list(range(1, waypoint_count))
    if len(observed_arguments) < 2:
        return "UNKNOWN"
    if waypoint_local:
        return "waypoint_idx"
    if global_all_segments:
        return "num_waypoints/global"
    return "UNKNOWN"


def validate_extension_mutation(record: Mapping[str, Any], tolerance: float = 1.0e-10) -> tuple[bool, list[str]]:
    """Validate the proven 2.12.4 local mutation, including derivative formulae."""
    errors: list[str] = []
    waypoint_idx = int(record["waypoint_idx"])
    target = waypoint_idx + 1
    factor = float(record["duration_extension_factor"])
    before = record["before"]
    after = record["after"]
    if target >= len(before["duration_from_previous"]) or target >= len(after["duration_from_previous"]):
        return False, ["target_waypoint_out_of_range"]
    duration_changed = changed_indices(before["duration_from_previous"], after["duration_from_previous"], tolerance)
    position_changed = changed_indices(before["position"], after["position"], tolerance)
    velocity_changed = changed_indices(before["velocity"], after["velocity"], tolerance)
    acceleration_changed = changed_indices(before["acceleration"], after["acceleration"], tolerance)
    if duration_changed != [target]:
        errors.append("unexpected_duration_mutation")
    if position_changed:
        errors.append("position_mutated")
    if any(index != target for index in velocity_changed):
        errors.append("unrelated_velocity_mutated")
    if any(index != target for index in acceleration_changed):
        errors.append("unrelated_acceleration_mutated")
    expected_duration = factor * float(record["original_duration_from_previous"])
    if not math.isclose(float(after["duration_from_previous"][target]), expected_duration, rel_tol=tolerance, abs_tol=tolerance):
        errors.append("target_duration_formula_mismatch")
    timestep = float(after["duration_from_previous"][target])
    for joint, before_velocity in enumerate(before["velocity"][target]):
        expected_velocity = float(before_velocity) / factor
        actual_velocity = float(after["velocity"][target][joint])
        if not math.isclose(actual_velocity, expected_velocity, rel_tol=tolerance, abs_tol=tolerance):
            errors.append(f"target_velocity_formula_mismatch:{joint}")
        previous_velocity = float(after["velocity"][waypoint_idx][joint])
        expected_acceleration = (actual_velocity - previous_velocity) / timestep
        actual_acceleration = float(after["acceleration"][target][joint])
        if not math.isclose(actual_acceleration, expected_acceleration, rel_tol=tolerance, abs_tol=tolerance):
            errors.append(f"target_acceleration_formula_mismatch:{joint}")
    return not errors, errors


def extension_summary(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    by_waypoint = Counter(str(int(row["waypoint_idx"])) for row in rows)
    primitive_ids = {str(row["unit_id"]) for row in rows}
    failures = sum(row.get("helper_mutation_invariant") != "PASSED" for row in rows)
    return {
        "FORMAL_INSTALLED_EXTENSION_CALLS": len(rows),
        "FORMAL_PRIMITIVES_WITH_EXTENSION_CALLS": len(primitive_ids),
        "FORMAL_EXTENSION_CALLS_BY_WAYPOINT": dict(sorted(by_waypoint.items(), key=lambda item: int(item[0]))),
        "FORMAL_MAX_EXTENSION_FACTOR": max((float(row["duration_extension_factor"]) for row in rows), default=None),
        "FORMAL_EXTENSION_MUTATION_INVARIANT_FAILURES": failures,
    }


def spray_validation_schema(validated_windows: int, violations: int, expected_windows: int = 189216) -> dict[str, Any]:
    complete = validated_windows == expected_windows
    return {
        "COVERAGE_STATUS": "COMPLETE" if complete else "INCOMPLETE",
        "VALIDATION_RESULT": "PASSED" if complete and violations == 0 else "FAILED",
        "VALIDATED_WINDOWS": validated_windows,
        "SPRAY_PROCESS_VIOLATIONS": violations,
    }


def semantic_closure_gate(certificate: Mapping[str, Any]) -> tuple[bool, str]:
    checks = [
        (certificate.get("H12_R5_IMMUTABLE") == "YES", "h12_r5_mutated"),
        (certificate.get("INSTALLED_BINARY_PROVENANCE") == "PASSED", "installed_moveit_provenance_failure"),
        (certificate.get("INSTALLED_BINARY_OBSERVED_SEMANTICS") != "UNKNOWN", "installed_helper_semantics_unproven"),
        (certificate.get("INTERPOSER_SECOND_ARGUMENT_MATCHES_PROVEN_BINARY_SEMANTICS") == "YES", "interposer_argument_mismatch"),
        (certificate.get("HELPER_MUTATION_INVARIANT_FAILURES") == 0, "helper_mutation_invariant_failure"),
        (certificate.get("FORMAL_PRIMITIVES") == 480, "primitive_coverage_failure"),
        (certificate.get("POST_RUCKIG_VALIDATED") == 189216, "incomplete_post_ruckig_validation"),
        (certificate.get("REPLAY") == "3/3", "nondeterministic_replay"),
        (certificate.get("UPSTREAM_IMMUTABLE") == "YES", "upstream_mutation"),
        (certificate.get("FOCUSED_TESTS") == "PASS", "focused_test_failure"),
        (certificate.get("REGRESSION_TESTS") == "PASS", "regression_test_failure"),
        (certificate.get("PHYSICAL_ROBOT_CONNECTED") == "NO", "physical_robot_connected"),
        (certificate.get("FJT_GOALS_SENT") == 0, "fjt_goal_sent"),
        (certificate.get("PHYSICAL_MOTION") == 0, "physical_motion"),
    ]
    for passed, blocker in checks:
        if not passed:
            return False, blocker
    return True, "none"
