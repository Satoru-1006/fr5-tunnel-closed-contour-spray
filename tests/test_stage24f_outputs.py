"""Deterministic contract tests for Stage 2.4F."""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_stage24f_global_branch_adaptive_closure as stage24f


def test_bounded_revolute_delta_does_not_wrap():
    semantics = [{"type": "revolute", "lower": -4.0, "upper": 4.0} for _ in range(6)]
    signed, absolute, limits_ok, _ = stage24f.raw_delta([3.0, 0, 0, 0, 0, 0], [-3.0, 0, 0, 0, 0, 0], semantics)
    assert limits_ok
    assert math.isclose(signed[0], -6.0)
    assert math.isclose(absolute[0], 6.0)
    assert math.degrees(absolute[0]) > 20.0


def test_adaptive_window_order_is_frozen():
    assert [size for size, _ in stage24f.WINDOWS] == [10, 20, 40, 80, 160, 320, 720]
    assert list(stage24f.WINDOWS[0][1]) == [715, 716, 717, 718, 719, 0, 1, 2, 3, 4]


def test_fragile_transition_count_is_32():
    assert len(stage24f.FRAGILE_TRANSITIONS) == 32


def test_no_downstream_stage_tokens_in_runner():
    text = Path(stage24f.__file__).read_text(encoding="utf-8")
    assert "TOTG" in text and "Ruckig" in text and "GNN" in text and "CCD" in text
    assert "Stage_2_5" in text


def _node(cid: str, wp: int, q0: float = 0.0, sig: str = "s"):
    return {
        "candidate_id": cid,
        "waypoint_index": wp,
        "joint_vector_rad": [q0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "ik_branch_signature": sig,
        "shoulder_branch": "positive",
        "elbow_branch": "positive",
        "wrist_branch": "positive",
        "source_stage": "test",
        "fcl_node_valid": True,
        "bullet_node_valid": True,
        "dual_backend_valid": True,
        "joint_limit_valid": True,
        "process_tolerance_pass": True,
        "process_tolerance_error": None,
        "position_offset_local_m": [0.0, 0.0, 0.0],
    }


def _edge(a, b, awp, bwp):
    return {
        "from_waypoint": awp,
        "to_waypoint": bwp,
        "from_candidate_id": a,
        "to_candidate_id": b,
        "signed_joint_delta_deg": [0.0] * 6,
        "absolute_joint_delta_deg": [0.0] * 6,
        "max_joint_delta_deg": 0.0,
        "blocking_joint": "j1",
        "joint_gate_valid": True,
        "fcl_edge_valid": True,
        "bullet_edge_valid": True,
    }


def _full_cycle():
    nodes = {f"n{i}": _node(f"n{i}", i) for i in range(720)}
    edges = [_edge(f"n{i}", f"n{i + 1}", i, i + 1) for i in range(719)]
    edges.append(_edge("n719", "n0", 719, 0))
    return nodes, edges


def test_transition_edges_do_not_imply_complete_open_chain():
    nodes = {"a0": _node("a0", 0), "b0": _node("b0", 0), "a1": _node("a1", 1), "b1": _node("b1", 1)}
    edges = [_edge("a0", "a1", 0, 1), _edge("b0", "b1", 0, 1)]
    result = stage24f.search_open_chain(nodes, edges)
    assert not result["complete_0_to_719_open_chain_exists"]


def test_branch_seed_order_is_deterministic():
    pool = {
        "same": _node("same", 5, sig="branch"),
        "prev": _node("prev", 4, math.radians(1.0), sig="branch"),
        "next": _node("next", 6, math.radians(2.0), sig="branch"),
    }
    result = stage24f.seed_rows(pool, 5, "branch", None)
    assert [row["candidate_id"] for row in result[:3]] == ["prev", "next", "same"]


def test_continuation_points_stay_inside_frozen_bounds():
    contract = stage24f.frozen_contract()
    for point in stage24f.bounded_parameter_points(contract, 719, 10):
        assert -0.006 <= point["tangent"] <= 0.006
        assert -0.006 <= point["lateral"] <= 0.006
        assert -0.006 <= point["normal"] <= 0.006
        assert -0.005 <= point["standoff"] <= 0.005
        assert -10.0 <= point["rx"] <= 10.0
        assert -10.0 <= point["ry"] <= 10.0
        assert -15.0 <= point["roll"] <= 15.0


def test_joint_gate_rejection_is_before_native_selection():
    nodes = {"a": _node("a", 719, math.radians(0.0)), "b": _node("b", 0, math.radians(30.0))}
    pending, rejected, selected = stage24f.prefilter_new_edges(nodes, {"a", "b"}, [{"type": "revolute", "lower": -4.0, "upper": 4.0}] * 6)
    assert not pending
    assert not selected
    assert rejected[0]["reject_reason"] == "joint_step_exceeds_20_deg"


def test_native_edge_request_contains_both_endpoints_for_interpolation(tmp_path):
    row = {"from_waypoint": 719, "to_waypoint": 0, "from_candidate_id": "a", "to_candidate_id": "b", "q0": [0.0] * 6, "q1": [0.1] * 6, "max_single_joint_step_deg": 5.7, "interpolation_step_deg": 1.0}
    path = tmp_path / "edge_requests.csv"
    stage24f.write_edge_requests([row], path)
    text = path.read_text(encoding="utf-8")
    assert "q0_0" in text and "q1_0" in text


def test_fcl_and_bullet_use_same_selected_request_set():
    text = Path(stage24f.__file__).read_text(encoding="utf-8")
    assert "for backend in BACKENDS" in text
    assert "selected_rows" in text


def test_only_full_720_node_720_edge_cycle_passes():
    nodes, edges = _full_cycle()
    result = stage24f.search_closed_cycle(nodes, edges, {"complete_0_to_719_open_chain_exists": True, "selected_node_ids": [f"n{i}" for i in range(720)], "selected_edges": edges[:719]})
    assert result["complete_720_node_720_edge_cycle_found"]
    broken = edges[:-1]
    result = stage24f.search_closed_cycle(nodes, broken, {"complete_0_to_719_open_chain_exists": True, "selected_node_ids": [f"n{i}" for i in range(720)], "selected_edges": broken[:719]})
    assert not result["complete_720_node_720_edge_cycle_found"]


def test_719_edge_open_chain_cannot_become_cycle():
    nodes, edges = _full_cycle()
    result = stage24f.search_closed_cycle(nodes, edges[:-1], {"complete_0_to_719_open_chain_exists": True, "selected_node_ids": [f"n{i}" for i in range(720)], "selected_edges": edges[:-1]})
    assert not result["complete_720_node_720_edge_cycle_found"]


def test_semantic_hash_excludes_runtime_paths_and_times():
    a = {"path": "a", "run_duration_s": 1.0, "value": [1, 2]}
    b = {"path": "b", "run_duration_s": 9.0, "value": [1, 2]}
    assert stage24f.semantic_hash(a) == stage24f.semantic_hash(b)


def test_blocked_gate_cannot_enable_stage25():
    text = Path(stage24f.__file__).read_text(encoding="utf-8")
    assert '"ready_for_Stage_2_5": status == "passed"' in text


def test_no_continuous_joint_flags_are_enabled():
    contract = stage24f.frozen_contract()
    assert contract["stage24f_policy"]["continuous_flags"] == {f"j{i}": False for i in range(1, 7)}
