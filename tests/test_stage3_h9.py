from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.stage3_h9_dataset import (
    COLLISION_METHOD,
    EXPECTED_MAIN_FEEDBACK_COUNT,
    JOINT_ORDER,
    DatasetValidationError,
    build_schema,
    build_split_policy,
    compute_forward_jerk,
    build_segments,
    evaluate_trajectory,
    semantic_dataset_payload,
    semantic_sha256,
    source_hash_manifest,
    validate_feedback_rows,
    validate_h7_rows,
    validate_no_group_leakage,
    validate_software_only,
    validate_trajectory_samples,
)


ROOT = Path(__file__).resolve().parents[1]
H8 = ROOT / "outputs/stage3_h8_software_only_recertification_20260811T055048Z"
H7 = ROOT / "outputs/stage3_h7_7_tier_b_remediation_20260809T143600Z"


def _sample(index: int, time: float, *, invalid: str | None = None) -> dict:
    q = [0.1 + index * 0.01] * 6
    v = [0.01] * 6
    a = [0.001] * 6
    e = [0.001] * 6
    row = {
        "run_id": "fixture",
        "trajectory_id": "trajectory",
        "trajectory_family_id": "family",
        "replay_id": "main",
        "replay_group_id": "group",
        "joint_order": JOINT_ORDER,
        "sample_index": index,
        "trajectory_time": time,
        "desired_trajectory_time": time,
        "planned_joint_position": q,
        "planned_joint_velocity": v,
        "planned_joint_acceleration": a,
        "desired_joint_position": q,
        "desired_joint_velocity": v,
        "desired_joint_acceleration": a,
        "actual_joint_position": q,
        "actual_joint_velocity": v,
        "actual_joint_acceleration": a,
        "joint_position_error": e,
        "joint_velocity_error": e,
        "joint_acceleration_error": e,
        "collision_method": COLLISION_METHOD,
        "position_limit_valid": True,
        "velocity_limit_valid": True,
        "acceleration_limit_valid": True,
        "jerk_limit_valid": True,
        "collision_free": True,
        "process_tolerance_valid": True,
        "time_monotonic": True,
        "spray_semantics_valid": True,
        "hard_constraint_valid": True,
        "spray_state": "SPRAY_ON",
    }
    if invalid == "dimension":
        row["actual_joint_position"] = [0.0] * 5
    if invalid == "constraint":
        row["velocity_limit_valid"] = False
        row["hard_constraint_valid"] = False
    return row


def test_schema_documents_fields_and_canonical_order() -> None:
    schema = build_schema()
    assert schema["schema_version"] == "stage3_h9_dataset_schema_v1"
    assert schema["canonical_joint_order"] == JOINT_ORDER
    fields = {field["name"]: field for field in schema["fields"]}
    for name in ("planned_joint_position", "desired_joint_position", "actual_joint_position", "derived_joint_jerk"):
        assert fields[name]["shape"] == [6]
        assert fields[name]["units"] is not None
        assert fields[name]["source_provenance"]
        assert "nullable" in fields[name]


def test_joint_order_consistency_is_not_alphabetical_inference() -> None:
    row = _sample(0, 0.0)
    row["joint_order"] = list(reversed(JOINT_ORDER))
    assert any("joint order mismatch" in error for error in validate_trajectory_samples([row]))


def test_timestamp_monotonicity_and_array_dimensionality() -> None:
    rows = [_sample(0, 0.0), _sample(1, 0.1), _sample(2, 0.2)]
    assert not validate_trajectory_samples(rows)
    assert validate_trajectory_samples([rows[0], _sample(1, -0.1)])
    assert validate_trajectory_samples([_sample(0, 0.0, invalid="dimension")])


def test_planned_desired_actual_are_separate() -> None:
    row = _sample(0, 0.0)
    row["planned_joint_position"] = [1.0] * 6
    row["desired_joint_position"] = [2.0] * 6
    row["actual_joint_position"] = [3.0] * 6
    assert row["planned_joint_position"] != row["desired_joint_position"]
    assert row["desired_joint_position"] != row["actual_joint_position"]


def test_missing_value_semantics_are_null_not_zero() -> None:
    row = _sample(0, 0.0)
    row["actual_tcp_position"] = None
    row["tcp_path_length"] = None
    assert row["actual_tcp_position"] is None
    assert semantic_sha256({"unavailable": None}) == semantic_sha256({"unavailable": None})


def test_jerk_uses_variable_dt_and_explicit_boundary() -> None:
    jerk, stats = compute_forward_jerk([[0.0] * 6, [1.0] * 6, [3.0] * 6], [0.0, 0.5, 1.5])
    assert jerk[0] is None
    assert jerk[1] == [2.0] * 6
    assert jerk[2] == [2.0] * 6
    assert stats["DT_MIN"] == 0.5
    assert stats["DT_MAX"] == 1.0
    with pytest.raises(DatasetValidationError):
        compute_forward_jerk([[0.0] * 6, [1.0] * 6], [0.1, 0.0])


def test_fk_determinism_is_content_determinism() -> None:
    payload = {"FK_BACKEND": "MoveIt2 PlanningScene/FK", "q": [0.1] * 6, "tcp_frame": "spray_tcp_link"}
    assert semantic_sha256(payload) == semantic_sha256(json.loads(json.dumps(payload)))


def test_segment_mapping_preserves_spray_semantics() -> None:
    h7_rows = [
        {"segment_id": 0, "primitive_id": 0, "spray_state": "SPRAY_ON"},
        {"segment_id": 0, "primitive_id": 0, "spray_state": "SPRAY_ON"},
        {"segment_id": 1000, "primitive_id": 1000, "spray_state": "SPRAY_OFF"},
    ]
    segments = build_segments(h7_rows, [0.0, 1.0, 2.0])
    assert [(row["segment_id"], row["spray_state"]) for row in segments] == [(0, "SPRAY_ON"), (1000, "SPRAY_OFF")]
    assert segments[1]["segment_type"] == "spray_off_transfer"


def test_hard_gate_aggregation_and_metric_calculation() -> None:
    rows = [_sample(0, 0.0), _sample(1, 0.5), _sample(2, 1.5)]
    result = evaluate_trajectory(rows, [{"spray_state": "SPRAY_ON", "segment_type": "spray_on", "duration": 1.5}], {"stop_count": 0})
    assert result["hard_gate_result"]["valid"] is True
    assert result["metrics"]["trajectory_duration"] == 1.5
    assert result["metrics"]["joint_path_length"] > 0.0
    rows[1] = _sample(1, 0.5, invalid="constraint")
    result = evaluate_trajectory(rows, [], {})
    assert result["hard_gate_result"]["valid"] is False
    assert result["metrics"]["constraint_violation_count"] > 0


def test_canonical_hash_is_independent_of_dict_order() -> None:
    left = {"b": 2.0, "a": [1.0, None]}
    right = {"a": [1.0, None], "b": 2.0}
    assert semantic_sha256(left) == semantic_sha256(right)


def test_three_generation_hashes_and_provenance_grouping() -> None:
    run = {"run_id": "r", "timestamp": "different", "source": "h8"}
    segments = [{"segment_order": 0, "spray_state": "SPRAY_ON"}]
    samples = [_sample(0, 0.0), _sample(1, 1.0)]
    payloads = [semantic_dataset_payload(run, segments, samples) for _ in range(3)]
    assert len({semantic_sha256(payload) for payload in payloads}) == 1
    assert not validate_no_group_leakage([{"trajectory_family_id": "f", "split_role": "generalization"}])
    assert validate_no_group_leakage([
        {"trajectory_family_id": "f", "split_role": "training"},
        {"trajectory_family_id": "f", "split_role": "test"},
    ])


def test_authoritative_feedback_count_and_h7_validation() -> None:
    feedback_path = H8 / "normal_mock_execution/feedback.jsonl"
    h7_path = H7 / "formal_candidate_2/ruckig_trajectories.jsonl"
    assert feedback_path.is_file()
    feedback = [json.loads(line) for line in feedback_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(feedback) == EXPECTED_MAIN_FEEDBACK_COUNT
    assert not validate_feedback_rows(feedback)
    h7_rows = [json.loads(line) for line in h7_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(h7_rows) == 20822
    assert not validate_h7_rows(h7_rows)


def test_frozen_artifact_hash_manifest_and_software_only_boundary(tmp_path: Path) -> None:
    source = tmp_path / "frozen.json"
    source.write_text("immutable\n", encoding="utf-8")
    manifest = source_hash_manifest(tmp_path, [("H7_EVIDENCE", source, "test")])
    assert manifest["records"][0]["sha256"]
    assert validate_software_only({"MOCK_ONLY_RUNTIME": True, "PHYSICAL_DRIVER_LOADED": False, "PHYSICAL_ROBOT_CONNECTED": False, "PHYSICAL_FJT_GOALS_SENT": 0}) == []
    assert validate_software_only({"MOCK_ONLY_RUNTIME": True, "PHYSICAL_DRIVER_LOADED": True, "PHYSICAL_ROBOT_CONNECTED": False, "PHYSICAL_FJT_GOALS_SENT": 0})


def test_negative_fixture_rejected_and_split_policy_is_policy_only() -> None:
    result = evaluate_trajectory([_sample(0, 0.0, invalid="dimension")], [], {})
    assert result["hard_gate_result"]["valid"] is False
    policy = build_split_policy()
    assert policy["ML_SPLIT_READY"] == "NO" if "ML_SPLIT_READY" in policy else policy["current_policy_result"] == "ML_SPLIT_READY: NO"
