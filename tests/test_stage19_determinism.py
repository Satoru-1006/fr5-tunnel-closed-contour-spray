from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from src.canonical_graph_hash import semantic_hash
from src.deterministic_ik_candidates import (
    deduplicate_ik_records,
    deterministic_dp,
    explicit_seed_templates,
    stable_node_identity,
    stable_task_pose_identity,
)


def test_explicit_seed_order_has_no_random_source() -> None:
    config = {"seed_templates": {"include_nominal_seed": True, "include_previous_waypoint_seeds": True, "include_fixed_shoulder_templates": True, "include_fixed_elbow_templates": True, "include_fixed_wrist_templates": True, "include_fixed_combination_templates": True}, "seed_offsets_rad": {"fixed_shoulder": [-np.pi, np.pi], "fixed_elbow": [-np.pi, np.pi], "fixed_wrist": [-np.pi, np.pi], "fixed_combination": [{"joint_indices": [0, 2, 4], "offset_rad": np.pi}]}}
    templates = explicit_seed_templates(4, [0] * 6, [0.1] * 6, config=config)
    assert [item.seed_order_index for item in templates] == list(range(len(templates)))
    assert [item.seed_template_family for item in templates][:2] == ["nominal", "previous_waypoint"]
    assert all(item.seed_template_id.startswith("seed:") for item in templates)


def test_single_seed_record_is_explicit_and_timeout_zero_is_not_the_only_evidence() -> None:
    config = {"seed_templates": {"include_nominal_seed": True}, "seed_offsets_rad": {}}
    template = explicit_seed_templates(0, [0] * 6, None, config=config)[0]
    assert template.seed_order_index == 0
    record = {"waypoint_id": 0, "task_pose_stable_id": "tp:x", "q_unwrapped_rad": [0] * 6, "seed_template_id": template.seed_template_id, "seed_provenance": template.to_record(), "actual_attempt_count": 1, "internal_random_restart_observed": False, "ik_call_mode": "deterministic_single_seed"}
    assert record["actual_attempt_count"] == 1
    assert record["ik_call_mode"] == "deterministic_single_seed"
    assert record["internal_random_restart_observed"] is False


def test_task_pose_and_node_identity_do_not_use_discovery_order() -> None:
    left = {"waypoint_id": 7, "tangential_offset_mm": 0.0, "longitudinal_offset_mm": 2.0, "standoff_offset_mm": 0.0, "roll_offset_deg": 0.0, "generation_strategy": "single_longitudinal", "formal_constraint_pass": True, "diagnostic_only": False}
    right = dict(left)
    assert stable_task_pose_identity(left) == stable_task_pose_identity(right)
    assert stable_node_identity(7, "tp:x", [0.1] * 6) == stable_node_identity(7, "tp:x", [0.1] * 6)


def test_equivalent_solutions_are_deduplicated_with_provenance() -> None:
    base = {"waypoint_id": 1, "task_pose_stable_id": "tp:x", "q_unwrapped_rad": [0.1] * 6, "q_rad": [0.1] * 6, "seed_template_id": "seed:b", "seed_provenance": {"seed_order_index": 1}, "valid": True}
    other = dict(base, seed_template_id="seed:a", seed_provenance={"seed_order_index": 0}, q_unwrapped_rad=[0.100001] * 6)
    result = deduplicate_ik_records([base, other], 1.0e-5)
    assert len(result) == 1
    assert result[0]["source_seed_count"] == 2
    assert result[0]["source_seed_template_ids"] == ["seed:a", "seed:b"]
    assert "seed_template_id" not in result[0]["stable_node_id"]


def test_dp_uses_stable_tie_break_not_first_discovery() -> None:
    def node(wp: int, node_id: str, pose_id: str, nominal: bool = True):
        return {"waypoint_id": wp, "stable_node_id": node_id, "task_pose_stable_id": pose_id, "valid": True, "node_cost": 0.0, "is_nominal": nominal, "actual_position_offset_mm": 0.0, "roll_offset_deg": 0.0, "standoff_offset_mm": 0.0}
    a, b = node(0, "a", "tp:a"), node(0, "b", "tp:b")
    c = node(1, "c", "tp:c")
    edges = {("a", "c"): {"valid": True, "cost": 1.0, "max_joint_step_deg": 1.0, "total_joint_motion_rad": 1.0}, ("b", "c"): {"valid": True, "cost": 1.0, "max_joint_step_deg": 1.0, "total_joint_motion_rad": 1.0}}
    result = deterministic_dp([[b, a], [c]], edges)
    assert result["found"]
    assert result["candidate_ids"] == ["a", "c"]


def test_artifact_and_semantic_hash_are_distinct_and_noise_is_ignored(tmp_path: Path) -> None:
    path = tmp_path / "x.json"
    path.write_text('{"b": 1, "a": 2}\n', encoding="utf-8")
    artifact_one = hashlib.sha256(path.read_bytes()).hexdigest()
    semantic_one = semantic_hash([{"id": "x", "value": 1.0, "timestamp": "a"}], sort_fields=("id",))
    path.write_text('{"a": 2, "b": 1, "timestamp": "different"}\n', encoding="utf-8")
    artifact_two = hashlib.sha256(path.read_bytes()).hexdigest()
    semantic_two = semantic_hash([{"id": "x", "value": 1.0, "timestamp": "b"}], sort_fields=("id",))
    assert artifact_one != artifact_two
    assert semantic_one == semantic_two
