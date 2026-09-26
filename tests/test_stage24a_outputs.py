"""Persisted Stage 2.4A gate and dependency regression checks."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/ik_graph_stage24a_edge_equivalence/fr5_scaled_horseshoe_demo_v45"


def load(name: str):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def test_frozen_counts_and_zero_sha_failure() -> None:
    gate = load("stage24a_gate_report.json")
    assert gate["formal_nodes"] == 2882
    assert gate["formal_waypoints"] == 720
    assert gate["candidate_edges_before_joint_gate"] == 12688
    assert gate["joint_gate_accepted_edges"] == 7855
    assert gate["joint_gate_rejected_edges"] == 4833
    assert gate["input_SHA256SUMS_failure_count"] == 0


def test_historical_difference_export_is_complete() -> None:
    all_edges = pq.read_table(OUT / "stage24a_all_disagreement_edges.parquet")
    first_samples = pq.read_table(OUT / "stage24a_first_differing_samples.parquet")
    classification = load("stage24a_edge_classification.json")
    assert all_edges.num_rows == 1697
    assert first_samples.num_rows == 1697
    assert classification["historical_difference_count"] == 1697
    assert classification["all_historical_pairs_limited_to_stage23b_target_links"] is True


def test_first_edge_exact_replay_and_representation_fix() -> None:
    replay = load("stage24a_first_edge_exact_replay.json")
    assert replay["transition"] == "299 -> 300"
    assert replay["edge"] == "0299-02 -> 0300-02"
    assert replay["first_historical_difference"]["historical_bullet_collision"] is True
    assert replay["first_historical_difference"]["post_fix_bullet_collision"] is False
    assert replay["comparison"]["joint_vector_equal"] is True
    assert replay["comparison"]["link_transform_equal"] is True
    assert replay["classification"] == "runtime_geometry_representation_not_activated_in_stage24_edge_path"


def test_backend_equivalence_gate() -> None:
    gate = load("stage24a_gate_report.json")
    eq = load("stage24a_backend_equivalence.json")
    assert gate["Stage_2_4A"] == "passed"
    assert eq["FCL_accepted_edges"] == 7855
    assert eq["Bullet_accepted_edges"] == 7855
    assert eq["accepted_edge_symmetric_difference"] == 0
    assert eq["FCL_only_edges"] == 0
    assert eq["Bullet_only_edges"] == 0
    assert eq["identical_joint_samples"] is True
    assert eq["identical_sample_fk_transforms"] is True


def test_three_run_determinism_and_process_status() -> None:
    gate = load("stage24a_gate_report.json")
    det = load("stage24a_determinism.json")
    assert gate["all_process_exit_codes_zero"] is True
    assert det["independent_process_runs"] == 3
    assert det["all_three_run_hashes_equal"] is True


def test_bounded_joint_semantics_and_known_closure_delta() -> None:
    manifest = load("stage24a_frozen_input_verification.json")
    assert manifest["failure_count"] == 0
    # The robot has six bounded revolute joints; no modulo wrap is permitted.
    from scripts.run_stage24_closed_loop_graph import shortest_delta

    raw = 213.748
    shortest_wrapped = 360.0 - raw
    assert abs(shortest_wrapped - 146.252) < 1e-9
    assert shortest_delta(0.0, raw, "revolute") == raw


def test_no_forbidden_downstream_claims() -> None:
    gate = load("stage24a_gate_report.json")
    assert gate["Ruckig"] == "not_run"
    assert gate["TOTG"] == "not_run"
    assert gate["CCD"] == "not_run"
    assert gate["clearance"] == "not_available"
    assert gate["Stage_2_4B"] == "unblocked_not_started"


def test_parquet_dependency_is_real() -> None:
    env = load("environment_manifest.json")
    assert env["pyarrow_version"]
    assert "site-packages" in env["pyarrow_module"].replace("\\", "/")
    assert pq.read_table(OUT / "stage24a_all_disagreement_edges.parquet").num_rows == 1697
