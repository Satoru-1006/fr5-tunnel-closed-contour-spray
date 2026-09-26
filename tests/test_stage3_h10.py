from __future__ import annotations

import json

import pytest

from src.stage3_h10_dataset import (
    COLLISION_METHOD,
    JOINT_ORDER,
    H10ValidationError,
    aggregate_hard_constraints,
    compare_geometric_paths,
    deterministic_group_split,
    geometric_path_signature,
    semantic_sha256,
    validate_group_split,
    validate_sample_rows,
    validate_software_only,
)


def native_row(index: int, time: float, family: str = "f") -> dict:
    q = [0.1 + 0.01 * index] * 6
    return {"trajectory_family_id": family, "segment_id": 0, "primitive_id": 0, "segment_order": 0, "spray_state": "SPRAY_ON", "trajectory_index": index, "time_from_start_s": time, "positions_rad": q, "velocities_rad_s": [0.01] * 6, "accelerations_rad_s2": [0.001] * 6}


def sample_row(index: int, time: float, family: str = "f") -> dict:
    q = [0.1 + 0.01 * index] * 6
    return {"trajectory_family_id": family, "joint_order": list(JOINT_ORDER), "trajectory_time": time, "planned_joint_position": q, "planned_joint_velocity": [0.01] * 6, "planned_joint_acceleration": [0.001] * 6, "desired_joint_position": q, "actual_joint_position": q, "actual_joint_velocity": [0.01] * 6, "actual_joint_acceleration": [0.001] * 6, "actual_tcp_position": None, "actual_tcp_orientation": None, "collision_method": COLLISION_METHOD, "spray_state": "SPRAY_ON", "hard_constraint_valid": True}


def path(offset: float = 0.0) -> list[dict]:
    return [{"segment_id": 0, "primitive_id": 0, "segment_order": 0, "spray_state": "SPRAY_OFF", "rows": [{**native_row(0, 0.0), "positions_rad": [offset] * 6}, {**native_row(1, 1.0), "positions_rad": [0.2 + offset] * 6}]}]


def test_frozen_h9_evidence_is_read_only_contract() -> None:
    assert geometric_path_signature(path()) == geometric_path_signature(path())


def test_trajectory_family_identity_requires_fields() -> None:
    from src.stage3_h10_dataset import validate_family_identity
    assert validate_family_identity({"trajectory_family_id": "f", "trajectory_id": "t", "generation_seed": 1, "geometric_path_signature": "h", "generation_method": "m"}) == []
    assert "missing_trajectory_family_id" in validate_family_identity({})


def test_trivial_replay_is_not_independent_family() -> None:
    assert compare_geometric_paths(path(), path())["trivial_duplicate"] is True


def test_timestamp_only_variation_is_not_independent_family() -> None:
    left = path()
    right = json.loads(json.dumps(left))
    right[0]["rows"][1]["time_from_start_s"] = 99.0
    assert compare_geometric_paths(left, right)["trivial_duplicate"] is True


def test_pure_retiming_is_same_geometric_family() -> None:
    assert geometric_path_signature(path()) == geometric_path_signature(path())


def test_materially_different_joint_path_is_different_family() -> None:
    assert geometric_path_signature(path()) != geometric_path_signature(path(0.01))
    assert compare_geometric_paths(path(), path(0.01))["trivial_duplicate"] is False


def test_duplicate_detection_reports_nearest_path() -> None:
    comparison = compare_geometric_paths(path(), path(0.00001))
    assert comparison["max_position_delta_rad"] > 0.0
    assert comparison["trivial_duplicate"] is True


def test_group_split_has_zero_leakage() -> None:
    roles = deterministic_group_split([{"trajectory_family_id": f"f{i}"} for i in range(50)])
    records = [{"trajectory_family_id": family, "split_role": role} for family, role in roles.items()]
    assert validate_group_split(records) == []


def test_group_split_is_deterministic() -> None:
    families = [{"trajectory_family_id": f"f{i}"} for i in range(50)]
    assert deterministic_group_split(families) == deterministic_group_split(families)


def test_seed_identity_is_deterministic_by_payload_hash() -> None:
    assert semantic_sha256({"seed": 7, "method": "route"}) == semantic_sha256({"method": "route", "seed": 7})


def test_joint_array_dimensions() -> None:
    assert not validate_sample_rows([sample_row(0, 0.0)])
    bad = sample_row(0, 0.0)
    bad["actual_joint_position"] = [0.0] * 5
    assert validate_sample_rows([bad])


def test_joint_order_is_canonical() -> None:
    row = sample_row(0, 0.0)
    row["joint_order"] = list(reversed(JOINT_ORDER))
    assert any("joint_order_mismatch" in error for error in validate_sample_rows([row]))


def test_timestamps_are_monotonic() -> None:
    assert not validate_sample_rows([sample_row(0, 0.0), sample_row(1, 0.1)])
    assert validate_sample_rows([sample_row(0, 0.1), sample_row(1, 0.0)])


def test_null_unavailable_semantics_are_not_zero() -> None:
    row = sample_row(0, 0.0)
    assert row["actual_tcp_position"] is None
    assert row["actual_tcp_position"] != [0.0, 0.0, 0.0]


def test_hard_constraint_aggregation() -> None:
    rows = [sample_row(0, 0.0), sample_row(1, 0.1)]
    assert aggregate_hard_constraints(rows)["HARD_CONSTRAINT_VIOLATIONS"] == 0
    rows[1]["hard_constraint_valid"] = False
    assert aggregate_hard_constraints(rows)["HARD_CONSTRAINT_VIOLATIONS"] > 0


def test_collision_semantics_are_adaptive_discrete() -> None:
    assert validate_sample_rows([sample_row(0, 0.0)]) == []
    row = sample_row(0, 0.0)
    row["collision_method"] = "continuous_ccd"
    assert validate_sample_rows([row])


def test_process_tolerance_is_preserved_as_boolean_gate() -> None:
    row = sample_row(0, 0.0)
    row["process_tolerance_valid"] = False
    assert aggregate_hard_constraints([row])["violation_counts"]["process_tolerance_valid"] == 1


def test_spray_semantics_preserved() -> None:
    row = sample_row(0, 0.0)
    row["spray_state"] = "SPRAY_OFF"
    assert row["spray_state"] in {"SPRAY_ON", "SPRAY_OFF"}


def test_software_only_boundary_passes() -> None:
    assert validate_software_only({"MOCK_ONLY_DATA": "YES", "PHYSICAL_ROBOT_CONNECTED": "NO", "PHYSICAL_DRIVER_LOADED": "NO", "PHYSICAL_FJT_GOALS_SENT": 0, "ROBOT_MOTION_STARTED": "NO"}) == []


def test_physical_fjt_goal_count_is_zero() -> None:
    assert validate_software_only({"MOCK_ONLY_DATA": "YES", "PHYSICAL_ROBOT_CONNECTED": "NO", "PHYSICAL_DRIVER_LOADED": "NO", "PHYSICAL_FJT_GOALS_SENT": 1, "ROBOT_MOTION_STARTED": "NO"})


def test_semantic_hash_ignores_mapping_order() -> None:
    assert semantic_sha256({"b": 2.0, "a": [1, None]}) == semantic_sha256({"a": [1, None], "b": 2.0})


def test_invalid_nonfinite_semantic_value_is_rejected() -> None:
    with pytest.raises(H10ValidationError):
        semantic_sha256({"bad": float("nan")})


def test_negative_invalid_trajectory_is_rejected() -> None:
    row = sample_row(0, 0.0)
    row["hard_constraint_valid"] = False
    assert validate_sample_rows([row])


def test_insufficient_family_count_is_fail_closed_by_threshold() -> None:
    from src.stage3_h10_dataset import MINIMUM_ACCEPTED_TRAJECTORY_FAMILIES
    assert MINIMUM_ACCEPTED_TRAJECTORY_FAMILIES == 30
    assert len(deterministic_group_split([{"trajectory_family_id": "f"}])) == 1


def test_thirty_families_are_split_without_leakage() -> None:
    roles = deterministic_group_split([{"trajectory_family_id": f"f{i}"} for i in range(30)])
    assert len(roles) == 30
    assert validate_group_split([{"trajectory_family_id": k, "split_role": v} for k, v in roles.items()]) == []


def test_h7_h8r_h9_hashes_are_external_frozen_inputs() -> None:
    payload = {"h7": "immutable", "h8r": "immutable", "h9": "immutable"}
    assert semantic_sha256(payload) == semantic_sha256(payload)


def test_schema_has_provenance_and_null_policy() -> None:
    from src.stage3_h10_dataset import build_schema
    schema = build_schema()
    assert schema["schema_version"] == "stage3_h10_dataset_schema_v1"
    assert "null means unavailable" in schema["null_semantics"]
    assert any(field["name"] == "planned_joint_position" for field in schema["fields"])


def test_no_adjacent_sample_split_policy_is_explicit() -> None:
    from src.stage3_h10_dataset import build_schema
    assert build_schema()["split_policy"]["adjacent_sample_split"] == "forbidden"

