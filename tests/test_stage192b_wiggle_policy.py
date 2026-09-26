from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/ik_graph_stage192b"


def _json(name: str):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def test_replay_exact_gate_is_evidenced():
    report = _json("exact_replay_report.json")["exact_wiggle_replay"]
    assert report["same_internal_state_reached"] is True
    assert report["injected_delta_identical"] is True
    assert report["success_count"] == 100
    assert report["solution_hash_count"] == 1


def test_capture_distinguishes_condition_and_trigger_and_preserves_vector_chain():
    rows = list(csv.DictReader((OUT / "wiggle_capture_random_same_instance_100.csv").open(newline="", encoding="utf-8")))
    events = [row for row in rows if row["wiggle_triggered"] == "true"]
    assert events
    assert all(row["wiggle_condition_reached"] == "true" for row in events)
    assert all(row["random_vector_raw"] and row["delta_q_after_scale"] and row["delta_q_after_joint_limit_clip"] for row in events)


def test_full_regression_is_not_claimed_before_coverage_gate():
    coverage = _json("candidate_coverage_report.json")
    full = _json("full_pipeline_reproducibility.json")
    assert coverage["candidate_coverage_restored"] is False
    assert full["status"] == "not_run_gate_not_met"
    assert full["formal_pass"] is None


def test_stage192b_constraints_remain_explicit():
    coverage = _json("candidate_coverage_report.json")
    assert coverage["collision_method"] == "adaptive_discrete_interpolation"
    assert coverage["ccd_status"] == "not_available"
    assert coverage["clearance_status"] == "not_available"
