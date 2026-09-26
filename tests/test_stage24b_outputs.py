"""Persisted Stage 2.4B closed-cycle recovery checks."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/ik_graph_stage24b_closed_cycle_recovery/fr5_scaled_horseshoe_demo_v45"


def load(name: str):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def test_stage24a_gate_was_respected() -> None:
    gate = load("stage24b_gate_report.json")
    assert gate["Stage_2_4A"] == "passed"


def test_all_closure_pairs_are_real_parquet_and_bounded_semantics() -> None:
    table = pq.read_table(OUT / "stage24b_all_closure_pairs.parquet")
    assert table.num_rows >= 12
    flags = table.column("continuous_flags").to_pylist()
    assert all(not any(item) for item in flags)


def test_no_open_chain_is_promoted_to_cycle() -> None:
    gate = load("stage24b_gate_report.json")
    selected = load("stage24b_selected_cycle.json")
    if gate["closed_cycle_found"]:
        assert gate["selected_nodes"] == 720
        assert gate["selected_edges"] == 720
    else:
        assert gate["Stage_2_4B"] == "blocked_no_model_aware_closed_cycle"
        assert gate["Stage_2_4"] == "blocked_no_closed_cycle"
        assert selected["complete_cycle"] is False
        assert gate["selected_nodes"] == 0
        assert gate["selected_edges"] == 0


def test_closure_gate_and_known_original_nearest_pair() -> None:
    summary = load("stage24b_cycle_search_summary.json")
    original = summary["strategies"]["A_existing_2882_node_full_cycle_search"]
    assert original["complete_cycle_found"] is False
    assert original["closure_edge_count"] == 0
    nearest = summary["strategies"]["E_seam_local_IK"]["nearest_closure_candidate"]
    assert nearest["max_model_aware_joint_step_deg"] > 20.0


def test_seam_ik_and_search_are_deterministic() -> None:
    det = load("stage24b_determinism.json")
    assert det["IK_independent_runs"] == 3
    assert det["IK_candidate_hash_equal"] is True
    assert det["cycle_search_independent_runs"] == 3
    assert det["cycle_search_hash_equal"] is True
    assert det["node_validation"]["three_run_valid_set_equal"]["fcl"] is True
    assert det["node_validation"]["three_run_valid_set_equal"]["bullet"] is True


def test_new_candidates_use_formal_dual_backend_node_gate() -> None:
    gate = load("stage24b_gate_report.json")
    audit = load("stage24b_new_candidate_node_audit.json")
    assert audit["all_process_exit_codes_zero"] is True
    assert audit["valid_symmetric_difference"] == 0
    assert gate["new_candidates_dual_backend_valid"] == audit["formal_dual_backend_valid"]


def test_seam_edges_preserve_backend_equivalence() -> None:
    gate = load("stage24b_gate_report.json")
    assert gate["seam_symmetric_difference"] == 0


def test_no_duplicate_terminal_waypoint_or_forbidden_stage() -> None:
    gate = load("stage24b_gate_report.json")
    fk = load("stage24b_fk_process_audit.json")
    assert fk["selected_cycle_checks"]["duplicate_terminal_waypoint_added"] is False
    assert gate["Ruckig"] == "not_run"
    assert gate["TOTG"] == "not_run"
    assert gate["CCD"] == "not_run"
    assert gate["clearance"] == "not_available"
    assert gate["Stage_2_5"] == "blocked"
