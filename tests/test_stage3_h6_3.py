from __future__ import annotations

from scripts import stage3_h6_3 as h63


def _candidate(candidate_id: str, waypoint_index: int, value: float, branch: str = "branch") -> dict:
    return {
        "candidate_id": candidate_id,
        "waypoint_index": waypoint_index,
        "joint_values": [value, 0.0, 0.0, 0.0, 0.0, 0.0],
        "branch_node_id": branch,
        "state": {
            "joint_limit_valid": True,
            "fk_computable": True,
            "translation_error_m": 0.0,
            "rotation_error_rad": 0.0,
            "environment_collision": False,
            "self_collision": False,
        },
    }


def test_h6_3_exact_search_is_not_beam_limited() -> None:
    rows = [{"waypoint_index": index} for index in range(3)]
    layers = {
        0: [_candidate("a0", 0, 0.0)],
        1: [_candidate("dead", 1, 0.7), _candidate("live", 1, 0.1)],
        2: [_candidate("end", 2, 0.2)],
    }
    result = h63.exact_segment_reachability(rows, layers)
    assert result["status"] == "PASSED"
    assert [item["candidate_id"] for item in result["selected_path"]] == ["a0", "live", "end"]
    assert result["exact_graph_search_used"] is True


def test_h6_3_reports_first_exact_disconnect_and_branch_frontier() -> None:
    rows = [{"waypoint_index": index} for index in range(2)]
    layers = {
        0: [_candidate("left", 0, 0.0, "target_53_branch")],
        1: [_candidate("right", 1, 2.0, "target_46_branch")],
    }
    result = h63.exact_segment_reachability(rows, layers)
    assert result["status"] == "BLOCKED"
    disconnect = result["first_graph_disconnect"]
    assert disconnect["previous_reachable_candidate_count"] == 1
    assert disconnect["next_layer_candidate_count"] == 1
    assert disconnect["previous_reachable_branch_node_ids"] == ["target_53_branch"]
    assert disconnect["unreachable_branch_node_ids"] == ["target_46_branch"]
    assert disconnect["minimum_reachable_L2_joint_delta_rad"] > h63.MAX_GRAPH_EDGE_L2_RAD


def test_h6_3_split_is_exactly_four_on_and_three_off_shape() -> None:
    original = [
        {"segment_id": 0, "target_ids": [58], "waypoint_start": 0, "waypoint_end": 260},
        {"segment_id": 1, "target_ids": [51, 30], "waypoint_start": 261, "waypoint_end": 264},
        {"segment_id": 2, "target_ids": [56, 41, 22, 62, 34, 10, 29, 36, 52, 18, 47, 55, 2, 53, 46, 43], "waypoint_start": 265, "waypoint_end": 587},
    ]
    split = h63.split_segments(original)
    assert len(split) == 4
    assert [item["segment_id"] for item in split] == [0, 1, 2, 3]
    assert split[2]["target_ids"][-1] == 53
    assert split[3]["target_ids"][0] == 46
    assert split[2]["waypoint_end"] == h63.BREAK_FROM_WAYPOINT
    assert split[3]["waypoint_start"] == h63.BREAK_TO_WAYPOINT


def test_h6_3_native_contract_labels_are_fail_closed() -> None:
    assert h63.COLLISION_METHOD == "adaptive_discrete_interpolation"
    assert h63.CCD == "NOT_AVAILABLE"
    assert h63.CLEARANCE == "NOT_AVAILABLE"
    assert h63.H62_ROOT.name.endswith("retry2")
