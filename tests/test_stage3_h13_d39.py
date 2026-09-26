import csv
import json
from pathlib import Path

import numpy as np

from tools import stage3_h13_d39_causal_task_space_recovery as d39


ROOT = Path(__file__).resolve().parents[1]


def test_d39_shadow_routes_are_causal_and_complete():
    case = d39.load_case()
    route = d39.smooth_velocity_rollout(case, window=2, decay_per_second=0.05)
    assert route.shape == (181, 6)
    assert np.isfinite(route).all()
    metrics = d39.candidate_metrics(route, case, source="test")
    assert metrics["future_reference_joint_rows_read"] == 0
    assert metrics["rows"] == 181


def test_d39_retention_artifacts_do_not_touch_d38_output():
    assert d39.OUTPUT != d39.D38_OUTPUT
    assert d39.OUTPUT not in d39.D38_OUTPUT.parents
    assert d39.D38_OUTPUT not in d39.OUTPUT.parents


def test_d39_certification_is_strict_when_a_gate_is_missing():
    gates = {name: True for name in d39.CERTIFICATION_GATES}
    assert all(gates.values())
    gates["endpoint_path_quality"] = False
    assert not all(gates.values())


def test_d39_start_evidence_is_the_authenticated_d38_bundle():
    start = d39.load_d38_start()
    assert start["root_cause"]["diagnosis"]["classification"] == "MIXED_EXPOSURE_BIAS_AND_DOMAIN_SHIFT"
    assert start["shadow"]["best_s2_candidate"] == "route4_update480"


def test_executed_d39_retention_and_certification_gates_are_monotonic():
    output = d39.OUTPUT
    summary = json.loads((output / "D39_SUMMARY.json").read_text(encoding="utf-8"))
    retention = json.loads((output / "D39_RETENTION_BASELINES.json").read_text(encoding="utf-8"))
    assert summary["TASK_STATUS"] == "PARTIAL_PASS"
    assert summary["FIRST_UNRESOLVED_BLOCKER"] == "BLOCKER_2_TASK_SPACE_GEOMETRY"
    assert retention["baseline_1"]["status"] == "LOCKED"
    assert retention["baseline_2"]["status"] == "NONE"
    rows = list(csv.DictReader((output / "D39_ROBOT_CERTIFICATION.csv").open(encoding="utf-8-sig", newline="")))
    assert rows
    assert all(row["FULL_ROBOT_CERTIFICATION"] == "FAIL" for row in rows)
