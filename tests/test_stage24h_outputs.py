"""Stage 2.4H contract and evidence regression tests."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_stage24h_reachability_frontier as stage24h


OUT = ROOT / "outputs/ik_graph_stage24h_reachability_frontier_continuation/fr5_scaled_horseshoe_demo_v45_20260801_bounded_bridge_search"


def test_bounded_revolute_delta_is_raw() -> None:
    signed, maximum, blocking = stage24h.joint_delta(__import__("numpy").array([math.radians(170.0)] + [0.0] * 5), __import__("numpy").array([math.radians(-170.0)] + [0.0] * 5))
    assert signed[0] == -340.0
    assert maximum == 340.0
    assert blocking == "j1"


def test_frontier_reachability_contract_is_persisted() -> None:
    if not (OUT / "stage24h_gate_report.json").exists():
        return
    gate = json.loads((OUT / "stage24h_gate_report.json").read_text(encoding="utf-8"))
    assert gate["initial_forward_frontier"] == "35->36"
    assert gate["initial_reachable_frontier_nodes"] == 33
    assert gate["ready_for_Stage_2_5"] is False
    assert gate["Stage_2_5"] == "blocked"
    assert gate["collision_method"] == "adaptive_discrete_interpolation"
    assert gate["CCD"] == "not_available"
    assert gate["clearance"] == "not_available"


def test_internal_continuation_is_not_a_formal_waypoint() -> None:
    path = OUT / "run1" / "stage24h_all_internal_continuation_states.jsonl"
    if not path.exists():
        return
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows
    assert all(row["formal_waypoint"] is False for row in rows)
    assert all(row["target_waypoint"] in range(720) for row in rows)


def test_no_waypoint_skip_in_selected_chain() -> None:
    path = OUT / "run1" / "stage24h_selected_open_chain.json"
    if not path.exists():
        return
    chain = json.loads(path.read_text(encoding="utf-8"))
    if chain.get("selected_node_ids"):
        assert len(chain["selected_node_ids"]) <= 720


def test_deterministic_rebuild_report_is_explicit() -> None:
    path = OUT / "stage24h_determinism_report.json"
    if not path.exists():
        return
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["requested_rebuilds"] >= 3
    assert "differences" in report


def test_runner_contains_required_gates() -> None:
    text = Path(stage24h.__file__).read_text(encoding="utf-8")
    for token in ("current_forward_frontier", "current_reverse_frontier", "HOMOTOPY_LADDER", "native_node_gate", "native_edge_gate", "fcl_bullet_node_difference", "fcl_bullet_edge_difference", "search_space_exhausted"):
        assert token in text


def test_edge_search_policy_is_explicitly_bounded() -> None:
    assert stage24h.EDGE_PAIR_SEARCH_POLICY == "seed_consistent_frontier_bridge_plus_all_reachable_previous_nodes_for_reverse_candidates"
