"""Regression tests for the Stage 2.4D global branch audit bundle."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_stage24d_global_branch_cycle_audit import MAX_STEP_DEG, OUT, pair_metrics


def _json(name: str):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def test_frozen_union_and_waypoint_coverage() -> None:
    gate = _json("stage24d_gate_report.json")
    assert gate["candidate_counts"]["Stage_2_4A"] == 2882
    assert gate["candidate_counts"]["Stage_2_4B"] == 59
    assert gate["candidate_counts"]["Stage_2_4C_dual_backend_valid"] == 3781
    assert gate["candidate_counts"]["union_unique_nodes"] == 6722
    nodes = [json.loads(line) for line in (OUT / "stage24d_nodes.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(nodes) == 6722
    assert {row["waypoint_index"] for row in nodes} == set(range(720))


def test_topology_is_bounded_and_j5_has_no_wrap() -> None:
    topology = _json("stage24d_joint_topology_audit.json")
    assert all(row["urdf_type"] == "revolute" for row in topology["joints"])
    assert all(row["continuous"] is False for row in topology["joints"])
    assert "bounded revolute" in topology["j5_conclusion"]
    assert all(len(row["winding_state"]) == 6 for row in [json.loads(line) for line in (OUT / "stage24d_nodes.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()])


def test_bounded_and_continuous_distance_rules() -> None:
    lower = [-3.0543] * 6
    upper = [3.0543] * 6
    bounded = pair_metrics([3.1, 0, 0, 0, 0, 0], [-3.1, 0, 0, 0, 0, 0], lower, upper, ["revolute"] * 6)
    continuous = pair_metrics([3.1, 0, 0, 0, 0, 0], [-3.1, 0, 0, 0, 0, 0], lower, upper, ["continuous"] + ["revolute"] * 5)
    assert bounded["max_single_joint_step_deg"] > 300.0
    assert continuous["max_single_joint_step_deg"] < 10.0


def test_closure_is_present_in_audit_and_is_blocked() -> None:
    gate = _json("stage24d_gate_report.json")
    cert = _json("stage24d_infeasibility_certificate.json")
    assert gate["blocking_transition"] == "719->0"
    assert gate["closed_cycle_found"] is False
    assert gate["selected_nodes"] == 0
    assert gate["selected_edges"] == 0
    assert cert["minimum_achievable_max_step_deg"] > MAX_STEP_DEG
    assert cert["closure_validated_edge_count"] == 0


def test_full_search_and_determinism() -> None:
    gate = _json("stage24d_gate_report.json")
    det = _json("stage24d_determinism_report.json")
    runs = [json.loads(line) for line in (OUT / "stage24d_search_runs.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(runs) == 1440
    assert {row["direction"] for row in runs} == {"forward", "reverse"}
    assert {row["start_waypoint"] for row in runs} == set(range(720))
    assert all(row["closed_cycle_found"] is False for row in runs)
    assert det["independent_runs"] == 3
    assert det["hash_mismatches"] == 0
    assert det["exit_codes"] == [0, 0, 0]
    assert gate["deterministic_reproduction"] == "passed"


def test_no_backend_difference_and_no_downstream_execution() -> None:
    gate = _json("stage24d_gate_report.json")
    assert gate["fcl_bullet_node_difference"] == 0
    assert gate["fcl_bullet_edge_difference"] == 0
    assert gate["collision_method"] == "adaptive_discrete_interpolation"
    assert gate["CCD"] == "not_available"
    assert gate["Ruckig"] == "not_run"
    assert gate["TOTG"] == "not_run"
    assert gate["Stage_2_5"] == "blocked"

