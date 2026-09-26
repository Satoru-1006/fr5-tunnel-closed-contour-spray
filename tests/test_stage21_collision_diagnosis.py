from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.run_stage2_1_collision_diagnosis import build_statistics, classify_terminal, digest, make_per_waypoint


def test_terminal_status_is_single_and_ordered() -> None:
    row = {"solver_success": True, "joint_bounds_valid": True, "fk_valid": True, "fk_position_error_m": 0.0, "collision_category": "both"}
    assert classify_terminal(row) == ("rejected_both_collision", "collision")


def test_waypoint_and_candidate_counts_are_separate() -> None:
    records = [
        {"candidate_id": "c0", "waypoint_index": 0, "joint_bounds_valid": True, "fk_valid": True, "self_collision": True, "robot_world_collision": False, "collision_category": "self", "terminal_status": "rejected_self_collision", "collision_pairs": ["wrist3_link<->wall"], "robot_links": ["wrist3_link"], "world_objects": ["wall"]},
        {"candidate_id": "c1", "waypoint_index": 0, "joint_bounds_valid": True, "fk_valid": True, "self_collision": False, "robot_world_collision": True, "collision_category": "robot_world", "terminal_status": "rejected_robot_world_collision", "collision_pairs": ["spray_tcp_link<->wall"], "robot_links": ["spray_tcp_link"], "world_objects": ["wall"]},
        {"candidate_id": "c2", "waypoint_index": 1, "joint_bounds_valid": True, "fk_valid": True, "self_collision": False, "robot_world_collision": False, "collision_category": "none", "terminal_status": "valid_node", "collision_pairs": [], "robot_links": [], "world_objects": []},
    ]
    per = make_per_waypoint(records, 2)
    stats = build_statistics(records, per, 2)
    assert stats["colliding_waypoint_count"] == 1
    assert stats["fully_blocked_waypoint_count"] == 1
    assert stats["colliding_candidate_count"] == 2
    assert stats["self_colliding_candidate_count"] == 1
    assert stats["robot_world_colliding_candidate_count"] == 1
    assert stats["final_valid_node_count"] == 1


def test_hash_is_stable() -> None:
    assert digest({"b": 2, "a": 1}) == digest({"a": 1, "b": 2})
