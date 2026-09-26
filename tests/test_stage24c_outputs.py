"""Stage 2.4C persisted evidence and gate checks."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pyarrow.parquet as pq
import pytest


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/ik_graph_stage24c_seam_window_repair/fr5_scaled_horseshoe_demo_v45"


def load(name: str):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def test_required_parquet_outputs_are_real_and_reloadable() -> None:
    for name in (
        "stage24c_all_window_candidates.parquet",
        "stage24c_valid_window_nodes.parquet",
        "stage24c_branch_audit.parquet",
        "stage24c_all_window_edges.parquet",
        "stage24c_nearest_cycle_candidates.parquet",
    ):
        assert pq.read_table(OUT / name).num_rows > 0


def test_windows_and_strict_environment_are_persisted() -> None:
    gate = load("stage24c_gate_report.json")
    env = load("stage24c_environment_manifest.json")
    assert gate["windows_tested"] == [4, 8, 16, 32, 64]
    assert env["actual_environment"]["FR5_BULLET_SHAPE_MODE"] == "use_shape_type"
    assert env["required_environment"]["FR5_BULLET_SHAPE_MODE"] == "use_shape_type"


def test_missing_or_wrong_shape_mode_fails_closed() -> None:
    from scripts.run_stage24c_seam_window_repair import require_shape_mode

    assert require_shape_mode("use_shape_type") == "use_shape_type"
    with pytest.raises(RuntimeError):
        require_shape_mode(None if "FR5_BULLET_SHAPE_MODE" not in os.environ else "")
    with pytest.raises(RuntimeError):
        require_shape_mode("legacy_whole_hull")


def test_window_summary_contains_entry_exit_and_real_closure_transition() -> None:
    summary = load("stage24c_transition_summary.json")
    for radius in ("4", "8", "16", "32", "64"):
        transitions = summary[radius]["transitions"]
        assert f"{719 - int(radius)}->{720 - int(radius)}" in transitions
        assert f"{int(radius)}->{int(radius) + 1}" in transitions
        assert "719->0" in transitions
        assert summary[radius]["transitions"]["719->0"]["fcl_accepted"] == 0


def test_continuous_wrap_is_model_dependent_and_bounded_wrap_is_rejected() -> None:
    from scripts.run_stage24_closed_loop_graph import shortest_delta

    assert abs(shortest_delta(3.1, -3.1, "continuous")) < 0.1
    assert abs(shortest_delta(3.1, -3.1, "revolute")) > 6.0


def test_node_and_edge_backend_equivalence() -> None:
    gate = load("stage24c_gate_report.json")
    node = load("stage24c_node_gate_audit.json")
    eq = load("stage24c_backend_equivalence.json")
    assert node["all_process_exit_codes_zero"] is True
    assert node["node_symmetric_difference"] == 0
    assert node["fcl_valid"] == node["bullet_valid"]
    assert gate["node_symmetric_difference"] == 0
    assert all(item["edge_symmetric_difference"] == 0 for item in eq.values())


def test_no_cycle_is_promoted_and_no_downstream_stage_started() -> None:
    gate = load("stage24c_gate_report.json")
    cycle = load("stage24c_selected_cycle.json")
    assert gate["Stage_2_4C"] == "blocked_seam_window_repair_failed"
    assert gate["Stage_2_4"] == "blocked_no_closed_cycle"
    assert gate["Stage_2_5"] == "blocked"
    assert gate["closed_cycle_found"] is False
    assert gate["selected_nodes"] == 0
    assert gate["selected_edges"] == 0
    assert gate["closure_edge"] is None
    assert cycle["complete_cycle_found"] is False
    assert gate["Ruckig"] == "not_run"
    assert gate["TOTG"] == "not_run"
    assert gate["CCD"] == "not_run"


def test_bounded_revolute_closure_has_no_legal_wrap() -> None:
    summary = load("stage24c_window_search_summary.json")
    nearest = summary["nearest_cycle"]
    assert nearest["max_step_joint"] == "j5"
    assert nearest["max_model_aware_joint_step_deg"] > 20.0
    assert all(value is False for value in nearest["wrap_legal"])
    assert all(value == "revolute" for value in nearest["joint_types"])


def test_determinism_hash_and_dirty_worktree_gate() -> None:
    gate = load("stage24c_gate_report.json")
    det = load("stage24c_determinism.json")
    sha = load("stage24c_sha256_verification.json")
    dirty = load("dirty_worktree_preservation.json")
    assert det["determinism"] is True
    assert len(set(det["candidate_hashes"])) == 1
    assert sha["failure_count"] == 0
    assert gate["SHA256SUMS"]["failure_count"] == 0
    assert dirty["unchanged"] is True
