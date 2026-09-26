from __future__ import annotations

from copy import deepcopy

from src.stage3_h12_r5a import (
    classify_installed_helper_semantics,
    extension_summary,
    semantic_closure_gate,
    semantic_sha256,
    spray_validation_schema,
    validate_extension_mutation,
)


def probe(argument: int) -> dict:
    target = argument + 1
    return {
        "second_argument": argument,
        "num_waypoints": 6,
        "duration_changed_indices": [target],
        "position_changed_indices": [],
        "velocity_changed_indices": [target],
        "acceleration_changed_indices": [target],
    }


def mutation() -> dict:
    before = {
        "duration_from_previous": [0.0, 0.13, 0.16, 0.19, 0.22, 0.25],
        "position": [[float(i)] for i in range(6)],
        "velocity": [[0.1 * (i + 1)] for i in range(6)],
        "acceleration": [[-0.2 * (i + 1)] for i in range(6)],
    }
    after = deepcopy(before)
    factor = 1.1
    target = 3
    after["duration_from_previous"][target] *= factor
    after["velocity"][target][0] /= factor
    after["acceleration"][target][0] = (
        after["velocity"][target][0] - after["velocity"][target - 1][0]
    ) / after["duration_from_previous"][target]
    return {
        "waypoint_idx": 2,
        "duration_extension_factor": factor,
        "original_duration_from_previous": before["duration_from_previous"][target],
        "before": before,
        "after": after,
    }


def valid_certificate() -> dict:
    return {
        "H12_R5_IMMUTABLE": "YES",
        "INSTALLED_BINARY_PROVENANCE": "PASSED",
        "INSTALLED_BINARY_OBSERVED_SEMANTICS": "waypoint_idx",
        "INTERPOSER_SECOND_ARGUMENT_MATCHES_PROVEN_BINARY_SEMANTICS": "YES",
        "HELPER_MUTATION_INVARIANT_FAILURES": 0,
        "FORMAL_PRIMITIVES": 480,
        "POST_RUCKIG_VALIDATED": 189216,
        "REPLAY": "3/3",
        "UPSTREAM_IMMUTABLE": "YES",
        "FOCUSED_TESTS": "PASS",
        "REGRESSION_TESTS": "PASS",
        "PHYSICAL_ROBOT_CONNECTED": "NO",
        "FJT_GOALS_SENT": 0,
        "PHYSICAL_MOTION": 0,
    }


def test_installed_helper_probe_classifies_waypoint_index():
    assert classify_installed_helper_semantics([probe(2), probe(1)]) == "waypoint_idx"


def test_probe_tracks_argument_dependent_target():
    first, second = probe(2), probe(1)
    assert first["duration_changed_indices"] == [3]
    assert second["duration_changed_indices"] == [2]


def test_ambiguous_probe_fails_closed():
    row = probe(2)
    row["duration_changed_indices"] = [1, 2, 3, 4, 5]
    assert classify_installed_helper_semantics([row, probe(1)]) == "UNKNOWN"


def test_expected_local_mutation_passes():
    assert validate_extension_mutation(mutation()) == (True, [])


def test_unrelated_duration_mutation_fails_closed():
    row = mutation()
    row["after"]["duration_from_previous"][5] *= 1.1
    passed, errors = validate_extension_mutation(row)
    assert not passed and "unexpected_duration_mutation" in errors


def test_out_of_range_target_fails_closed():
    row = mutation()
    row["waypoint_idx"] = 5
    assert validate_extension_mutation(row) == (False, ["target_waypoint_out_of_range"])


def test_target_velocity_recalculation_is_enforced():
    row = mutation()
    row["after"]["velocity"][3][0] += 0.01
    passed, errors = validate_extension_mutation(row)
    assert not passed and any(error.startswith("target_velocity_formula_mismatch") for error in errors)


def test_target_acceleration_recalculation_is_enforced():
    row = mutation()
    row["after"]["acceleration"][3][0] += 0.01
    passed, errors = validate_extension_mutation(row)
    assert not passed and any(error.startswith("target_acceleration_formula_mismatch") for error in errors)


def test_zero_extension_summary():
    assert extension_summary([]) == {
        "FORMAL_INSTALLED_EXTENSION_CALLS": 0,
        "FORMAL_PRIMITIVES_WITH_EXTENSION_CALLS": 0,
        "FORMAL_EXTENSION_CALLS_BY_WAYPOINT": {},
        "FORMAL_MAX_EXTENSION_FACTOR": None,
        "FORMAL_EXTENSION_MUTATION_INVARIANT_FAILURES": 0,
    }


def test_used_extension_summary():
    rows = [
        {"unit_id": "a", "waypoint_idx": 2, "duration_extension_factor": 1.1, "helper_mutation_invariant": "PASSED"},
        {"unit_id": "a", "waypoint_idx": 2, "duration_extension_factor": 1.21, "helper_mutation_invariant": "PASSED"},
    ]
    summary = extension_summary(rows)
    assert summary["FORMAL_INSTALLED_EXTENSION_CALLS"] == 2
    assert summary["FORMAL_PRIMITIVES_WITH_EXTENSION_CALLS"] == 1


def test_spray_schema_separates_coverage_from_result():
    row = spray_validation_schema(189216, 2304)
    assert row["COVERAGE_STATUS"] == "COMPLETE"
    assert row["VALIDATION_RESULT"] == "FAILED"


def test_semantic_serialization_is_deterministic():
    assert semantic_sha256({"b": 2, "a": 1}) == semantic_sha256({"a": 1, "b": 2})


def test_semantic_closure_gate_does_not_require_h12_dynamic_pass():
    assert semantic_closure_gate(valid_certificate()) == (True, "none")


def test_semantic_closure_gate_rejects_unknown_binary_behavior():
    row = valid_certificate()
    row["INSTALLED_BINARY_OBSERVED_SEMANTICS"] = "UNKNOWN"
    assert semantic_closure_gate(row) == (False, "installed_helper_semantics_unproven")


def test_r5_historical_evidence_is_addressed_only_by_hash_snapshot():
    source = open("scripts/stage3_h12_r5a_moveit_helper_semantics.py", encoding="utf-8").read()
    assert "R5_ROOT.rglob" in source
    assert "H12_R5_IMMUTABLE" in source


def test_interposer_installed_call_passes_proven_waypoint_index():
    source = open("tools/stage25r_ruckig_interposer.cpp", encoding="utf-8").read()
    assert "extendTrajectoryDuration(duration_extension_factor, waypoint_idx, num_dof" in source
    assert "second_argument_passed" in source
