"""Regression checks for the persisted Stage 2.4 formal bundle."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_stage24_closed_loop_graph import MAX_STEP_DEG, OUT, shortest_delta


def _json(name: str):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def test_stage23b_counts_and_hash_gate() -> None:
    gate = _json("stage24_gate_report.json")
    assert gate["input_integrity"]["candidate_count"] == 3077
    assert gate["input_integrity"]["formal_valid_nodes"] == 2882
    assert gate["input_integrity"]["waypoint_count"] == 720
    assert gate["input_integrity"]["sha256_verification_failures"] == 0


def test_frozen_nodes_cover_every_waypoint() -> None:
    rows = (OUT / "stage24_nodes.jsonl").read_text(encoding="utf-8").splitlines()
    nodes = [json.loads(line) for line in rows if line.strip()]
    assert len(nodes) == 2882
    assert {node["waypoint_id"] for node in nodes} == set(range(720))


def test_only_adjacent_and_closed_layer_edges() -> None:
    for name in ("stage24_edges_fcl.jsonl", "stage24_edges_bullet.jsonl"):
        for line in (OUT / name).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            assert row["to_waypoint"] == (row["from_waypoint"] + 1) % 720


def test_continuous_shortest_angle_and_bounded_no_wrap() -> None:
    assert abs(shortest_delta(3.1, -3.1, "continuous")) < 0.1
    assert abs(shortest_delta(3.1, -3.1, "revolute")) > 6.0


def test_twenty_degree_gate_and_no_indeterminate_edges() -> None:
    gate = _json("stage24_gate_report.json")
    eq = _json("stage24_edge_equivalence.json")
    assert MAX_STEP_DEG == 20.0
    assert eq["indeterminate_edges"] == 0
    assert gate["joint_constraints"]["closure_edge_joint_step_checked"] is True


def test_backend_edge_equivalence_and_run_determinism() -> None:
    eq = _json("stage24_edge_equivalence.json")
    det = _json("stage24_determinism_report.json")
    gate = _json("stage24_gate_report.json")
    if gate["Stage_2_4"] == "passed":
        assert eq["accepted_edge_symmetric_difference_count"] == 0
    else:
        assert gate["Stage_2_4"] == "blocked_backend_edge_disagreement"
        assert eq["accepted_edge_symmetric_difference_count"] > 0
    assert det["independent_runs"] == 3
    assert det["all_core_hashes_identical"] is True


def test_stage24_does_not_claim_time_parameterization_or_ccd() -> None:
    gate = _json("stage24_gate_report.json")
    assert gate["process_constraints"]["Ruckig_executed"] is False
    assert gate["process_constraints"]["TOTG_executed"] is False
    assert gate["not_certified_here"]["continuous_collision_detection"] == "not_evaluated"
