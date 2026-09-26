from __future__ import annotations

import math

from scripts import stage3_h5 as h5


def _rules() -> list[dict[str, object]]:
    return [
        {"name": "j1", "type": "continuous", "continuous": True, "bounded_revolute": False, "lower": None, "upper": None, "range": 2 * math.pi},
        {"name": "j2", "type": "revolute", "continuous": False, "bounded_revolute": True, "lower": -2.0, "upper": 2.0, "range": 4.0},
    ]


def test_topology_aware_delta_wraps_only_continuous_joints() -> None:
    delta = h5.topology_delta([math.pi - 0.01, 0.0], [-math.pi + 0.01, 1.0], _rules())
    assert math.isclose(delta[0], 0.02, abs_tol=1e-12)
    assert math.isclose(delta[1], 1.0, abs_tol=1e-12)


def test_topology_dedup_is_not_raw_vector_equality() -> None:
    raw = [
        {"seed_id": "seed_000", "seed_index": 0, "joint_values": [math.pi - 0.01, 0.5], "solver_plugin": "p", "solver_api": "a"},
        {"seed_id": "seed_001", "seed_index": 1, "joint_values": [-math.pi - 0.01, 0.5 + 1e-10], "solver_plugin": "p", "solver_api": "a"},
    ]
    candidates = h5.deduplicate_topology(raw, 1, _rules())
    assert len(candidates) == 1
    assert candidates[0]["seed_ids"] == ["seed_000", "seed_001"]


def test_interpolation_sample_count_uses_frozen_adaptive_contract() -> None:
    rules = [
        {"name": "j1", "continuous": False, "lower": -2.0, "upper": 2.0, "range": 4.0},
    ]
    source = {"ik_candidate_id": "a", "target_id": 1, "task_sample_id": "a", "joint_values": [0.0], "joint_rules": rules, "h5_joint_fk_valid": True, "h5_endpoint_collision_free": True}
    target = {"ik_candidate_id": "b", "target_id": 2, "task_sample_id": "b", "joint_values": [math.radians(1.1)], "joint_rules": rules, "h5_joint_fk_valid": True, "h5_endpoint_collision_free": True}
    edges, states = h5.prepare_edges([source, target], {"surface_adjacent_edges": [[1, 2]]})
    assert edges[0]["interpolation_method"] == "adaptive_discrete_interpolation"
    assert edges[0]["interpolation_sample_count"] == 4
    assert len(states) == 4


def test_no_stage3_h5_code_starts_forbidden_operations() -> None:
    text = open("scripts/stage3_h5.py", encoding="utf-8").read()
    assert "FJT_GOALS_SENT" in text
    assert "RUCKIG_STARTED" in text
    assert "MoveIt PlanningScene + native FCL" in text
    assert "DETERMINISTIC_MULTI_SEED_NON_EXHAUSTIVE" in text
