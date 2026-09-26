from __future__ import annotations

import json
from pathlib import Path

from src.stage3_h4_4 import (
    COLLISION_METHOD,
    apply_transform,
    continuity_graph,
    immutable_mismatch_count,
    placement_bank,
    placement_matrix,
    ranking_key,
)


def test_h4_4_first_layer_is_frozen_baseline_single_and_two_axis_bank() -> None:
    bank = placement_bank(placement_matrix(-0.65, -0.65, -0.15))
    assert len(bank) == 73
    assert bank[0]["label"] == "baseline"
    assert {item["label"] for item in bank[1:13]} == {
        "dx_minus_050", "dx_plus_050", "dy_minus_050", "dy_plus_050", "dz_minus_050", "dz_plus_050",
        "yaw_minus_15", "yaw_plus_15", "pitch_minus_10", "pitch_plus_10", "roll_minus_10", "roll_plus_10",
    }
    assert all(item["random_search"] is False for item in bank)
    assert all(item["ml"] is False and item["rl"] is False for item in bank)
    assert all(abs(value) <= 0.05 + 1e-12 for item in bank for value in item["delta_translation_m"])
    assert all(abs(item["delta_yaw_pitch_roll_deg"][0]) <= 15.0 + 1e-12 for item in bank)
    assert all(abs(item["delta_yaw_pitch_roll_deg"][1]) <= 10.0 + 1e-12 for item in bank)
    assert all(abs(item["delta_yaw_pitch_roll_deg"][2]) <= 10.0 + 1e-12 for item in bank)


def test_transform_preserves_frozen_standoff_side() -> None:
    matrix = placement_matrix(-0.65, -0.65, -0.15, yaw_deg=15.0, pitch_deg=-10.0, roll_deg=10.0)
    surface = [0.9, 0.2, 0.5]
    normal = [0.9761870601839527, 0.21693045781865616, 0.0]
    tcp = [surface[i] + 0.26 * normal[i] for i in range(3)]
    surface_world = apply_transform(matrix, surface)
    tcp_world = apply_transform(matrix, tcp)
    normal_world = [sum(matrix[row][col] * normal[col] for col in range(3)) for row in range(3)]
    vector = [tcp_world[i] - surface_world[i] for i in range(3)]
    assert abs(sum(vector[i] * normal_world[i] for i in range(3)) - 0.26) < 1e-12
    assert COLLISION_METHOD == "adaptive_discrete_interpolation"


def test_ranking_is_lexicographic_and_deterministic() -> None:
    candidate = {"candidate_id": "h4_4_001", "delta_translation_m": [0.05, 0.0, 0.0], "delta_yaw_pitch_roll_deg": [0.0, 0.0, 0.0]}
    better = {"curved_metrics": {"reachable_collision_free": 2, "environment_collision": 0, "self_collision": 0, "ik_unreachable": 1, "fk_mismatch": 0}, "continuity_summary": {"largest_connected_component_target_count": 2}, "minimum_clearance_m": None}
    worse = {"curved_metrics": {"reachable_collision_free": 1, "environment_collision": 0, "self_collision": 0, "ik_unreachable": 0, "fk_mismatch": 0}, "continuity_summary": {"largest_connected_component_target_count": 10}, "minimum_clearance_m": None}
    assert ranking_key(candidate, better) < ranking_key(candidate, worse)


def test_immutable_mismatch_count_is_fail_closed() -> None:
    before = {"artifacts": {"h3": {"tree_hash": "a"}, "h4": {"tree_hash": "b"}}}
    after = json.loads(json.dumps(before))
    assert immutable_mismatch_count(before, after) == 0
    after["artifacts"]["h4"]["tree_hash"] = "changed"
    assert immutable_mismatch_count(before, after) == 1


def test_continuity_graph_requires_surface_adjacent_free_branches() -> None:
    targets = [
        {"geometry_id": "fixture_curved_cylinder_patch", "task_sample_id": "t0", "surface_point_xyz_m": [0.0, 0.0, 0.0], "tcp_target_position_xyz_m": [0.0, 0.0, 0.26]},
        {"geometry_id": "fixture_curved_cylinder_patch", "task_sample_id": "t1", "surface_point_xyz_m": [0.01, 0.0, 0.0], "tcp_target_position_xyz_m": [0.01, 0.0, 0.26]},
    ]
    target_rows = [
        {"target_index": 0, "geometry_id": "fixture_curved_cylinder_patch", "classification": "REACHABLE_COLLISION_FREE"},
        {"target_index": 1, "geometry_id": "fixture_curved_cylinder_patch", "classification": "REACHABLE_COLLISION_FREE"},
    ]
    ik_rows = [
        {"target_index": 0, "geometry_id": "fixture_curved_cylinder_patch", "task_sample_id": "t0", "deduplicated_candidates": [{"candidate_id": "candidate_000", "seed_id": "seed_000", "seed_index": 0, "joint_positions": [0, 0, 0, 0, 0, 0]}]},
        {"target_index": 1, "geometry_id": "fixture_curved_cylinder_patch", "task_sample_id": "t1", "deduplicated_candidates": [{"candidate_id": "candidate_000", "seed_id": "seed_001", "seed_index": 1, "joint_positions": [0.1, 0, 0, 0, 0, 0]}]},
    ]
    fk_rows = [
        {"target_index": 0, "candidate_id": "candidate_000", "fk_valid": True, "joint_validation": {"valid": True}},
        {"target_index": 1, "candidate_id": "candidate_000", "fk_valid": True, "joint_validation": {"valid": True}},
    ]
    native = {(0, "candidate_000"): {"collision_free": True}, (1, "candidate_000"): {"collision_free": True}}
    graph, summary = continuity_graph(target_rows, ik_rows, fk_rows, native, targets)
    assert len(graph["nodes"]) == 2
    assert summary["adjacent_collision_free_edge_count"] >= 1
    assert summary["largest_connected_component_target_count"] == 2
