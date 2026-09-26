"""Integrity tests for the Stage 2.4E evidence bundle."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/ik_graph_stage24e_tolerance_search/fr5_scaled_horseshoe_demo_v45_retry_4_20260731"


def read_json(name: str):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def test_contract_is_source_backed_and_finite_search_is_labelled() -> None:
    contract = read_json("stage24e_frozen_tolerance_contract.yaml.json")
    assert contract["Stage_2_4E"] == "contract_frozen"
    assert contract["authoritative_inputs_modified"] is False
    assert contract["finite_search_design"]["continuous_space_claim"] is False
    assert contract["stage24d_sha256_verification"]["failure_count"] == 0
    assert all(item["sha256"] for item in contract["source_files"] if item["exists"])


def test_candidate_generation_and_native_nodes_are_deterministic() -> None:
    audit = read_json("stage24e_candidate_generation_audit.json")
    node = read_json("stage24e_node_gate_audit.json")
    assert audit["runs"] == 3
    assert audit["deterministic"] is True
    assert node["three_run_fcl_set_equal"] is True
    assert node["three_run_bullet_set_equal"] is True
    assert node["fcl_bullet_node_difference"] == 0


def test_all_720_transition_summary_and_downstream_stop() -> None:
    gate = read_json("stage24e_gate_report.json")
    with (OUT / "stage24e_all_720_transition_summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 720
    assert gate["collision_method"] == "adaptive_discrete_interpolation"
    assert gate["CCD"] == "not_available"
    assert gate["clearance"] == "not_available"
    assert gate["Ruckig"] == "not_run"
    assert gate["TOTG"] == "not_run"
    assert gate["Stage_2_5"] == "blocked"
    assert gate["complete_tolerance_space_covered"] is False


def test_no_overclaim_when_cycle_is_absent() -> None:
    gate = read_json("stage24e_gate_report.json")
    cycle = read_json("stage24e_selected_cycle.json")
    assert gate["global_tolerance_aware_closed_cycle_exists"] == cycle["closed_cycle_found"]
    if not cycle["closed_cycle_found"]:
        assert gate["ready_for_next_stage"] is False
        assert gate["selected_nodes"] == 0
        assert gate["selected_edges"] == 0
