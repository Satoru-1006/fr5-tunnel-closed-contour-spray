from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from src.stage3_h4_reachability import (
    H4Config,
    H4ValidationError,
    classify_target,
    deduplicate_candidates,
    detect_forbidden_execution_source,
    generate_seed_states,
    replay_hash,
    validate_h3_target,
    validate_joint_vector,
)


ROOT = Path(__file__).resolve().parents[1]
H3_TARGETS = ROOT / "outputs/stage3_h3_coverage_baseline_20260808T034526Z/stage3_h3_surface_targets.jsonl"


def _target() -> dict:
    return json.loads(H3_TARGETS.read_text(encoding="utf-8").splitlines()[0])


def test_h3_target_schema_and_provenance_are_preserved() -> None:
    target = _target()
    validated = validate_h3_target(target, config=H4Config())
    assert validated == target
    assert validated["task_sample_id"].endswith("sample_000000")


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("coordinate_frame", "map", "invalid_frame"),
        ("tcp_target_orientation_xyzw", [0.0, 0.0, 0.0, 0.0], "invalid_quaternion"),
        ("tcp_target_position_xyz_m", [math.nan, 0.0, 0.0], "nonfinite_value"),
        ("source_geometry_hash", "bad", "invalid_provenance_hash"),
    ],
)
def test_malformed_h3_target_fails_closed(field: str, value: object, code: str) -> None:
    target = _target()
    target[field] = value
    with pytest.raises(H4ValidationError) as error:
        validate_h3_target(target, config=H4Config())
    assert error.value.code == code


def test_missing_tcp_and_missing_provenance_fail_closed() -> None:
    target = _target()
    del target["tcp_target_position_xyz_m"]
    with pytest.raises(H4ValidationError, match="missing H3 fields"):
        validate_h3_target(target, config=H4Config())
    target = _target()
    del target["h2_manifest_hash"]
    with pytest.raises(H4ValidationError, match="missing H3 fields"):
        validate_h3_target(target, config=H4Config())


def test_seed_contract_is_exact_and_reproducible() -> None:
    config = H4Config()
    first = generate_seed_states(["j1", "j2"], [-1.0, -2.0], [1.0, 2.0], config=config)
    second = generate_seed_states(["j1", "j2"], [-1.0, -2.0], [1.0, 2.0], config=config)
    assert first == second
    assert len(first) == config.seed_count == 13
    assert first[0]["joint_positions"] == [0.0, 0.0]
    assert [item["seed_index"] for item in first] == list(range(13))


def test_joint_validation_rejects_wrong_dimension_nan_and_limits_without_repair() -> None:
    common = {"expected_joint_names": ["j1", "j2"], "lower": [-1.0, -1.0], "upper": [1.0, 1.0], "tolerance": 1e-9}
    assert validate_joint_vector([0.0], **common)["rejection_reason"] == "invalid_vector"
    assert validate_joint_vector([math.inf, 0.0], **common)["rejection_reason"] == "nonfinite_value"
    result = validate_joint_vector([2.0, 0.0], **common)
    assert result["rejection_reason"] == "joint_limit_violation"
    assert result["silent_repair_applied"] is False


def test_duplicate_candidate_and_canonical_ordering_are_deterministic() -> None:
    candidates = [
        {"seed_id": "seed_002", "seed_index": 2, "joint_positions": [0.5, -0.25]},
        {"seed_id": "seed_001", "seed_index": 1, "joint_positions": [0.5 + 1e-10, -0.25]},
        {"seed_id": "seed_003", "seed_index": 3, "joint_positions": [-0.5, 0.25]},
    ]
    unique = deduplicate_candidates(candidates, tolerance=1e-8)
    assert len(unique) == 2
    assert [row["candidate_id"] for row in unique] == ["candidate_000", "candidate_001"]
    assert unique[0]["joint_positions"] == [-0.5, 0.25]


def test_target_classifications_are_fail_closed_and_reasoned() -> None:
    assert classify_target(input_valid=False, candidate_count_total=0, candidate_count_joint_valid=0, candidate_count_fk_valid=0, candidate_count_collision_free=0) == "INVALID_INPUT"
    assert classify_target(input_valid=True, candidate_count_total=0, candidate_count_joint_valid=0, candidate_count_fk_valid=0, candidate_count_collision_free=0) == "IK_UNREACHABLE"
    assert classify_target(input_valid=True, candidate_count_total=1, candidate_count_joint_valid=1, candidate_count_fk_valid=1, candidate_count_collision_free=0, self_collision_count=1) == "IK_FOUND_SELF_COLLISION"
    assert classify_target(input_valid=True, candidate_count_total=1, candidate_count_joint_valid=1, candidate_count_fk_valid=1, candidate_count_collision_free=0, environment_collision_count=1) == "IK_FOUND_ENV_COLLISION"
    assert classify_target(input_valid=True, candidate_count_total=1, candidate_count_joint_valid=1, candidate_count_fk_valid=1, candidate_count_collision_free=0, collision_backend_disagreement=True) == "COLLISION_BACKEND_DISAGREEMENT"


def test_replay_hash_changes_on_candidate_or_metric_change() -> None:
    records = [{"task_sample_id": "g:sample_000000", "classification": "IK_UNREACHABLE"}]
    metrics = {"target_count": 1, "reachable_target_ratio": 0.0}
    first = replay_hash(records, metrics)
    assert first == replay_hash(copy.deepcopy(records), copy.deepcopy(metrics))
    assert first != replay_hash([{"task_sample_id": "g:sample_000000", "classification": "REACHABLE_COLLISION_FREE"}], metrics)


def test_forbidden_execution_source_is_detectable() -> None:
    assert "send_goal_async" in detect_forbidden_execution_source("node.send_goal_async(goal)")
    assert detect_forbidden_execution_source("RobotState.set_from_ik(); PlanningScene.check_collision()") == []
