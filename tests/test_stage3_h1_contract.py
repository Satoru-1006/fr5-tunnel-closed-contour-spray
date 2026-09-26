from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_DIR = ROOT / "config/stage3"
REQUIRED_METRIC_FIELDS = {
    "metric_id", "name", "symbol", "unit", "definition", "formula", "input", "reference_frame",
    "aggregation", "threshold", "threshold_type", "hard_gate_or_objective", "source/provenance", "version",
}


def read(name: str) -> dict:
    return json.loads((CONTRACT_DIR / name).read_text(encoding="utf-8"))


def test_metric_contract_is_machine_complete() -> None:
    contract = read("stage3_research_metric_contract.json")
    assert contract["status"] == "FROZEN_FOR_H2"
    assert len(contract["metrics"]) >= 20
    assert all(REQUIRED_METRIC_FIELDS <= set(metric) for metric in contract["metrics"])
    assert any(metric["threshold_type"] == "unresolved" for metric in contract["metrics"])


def test_gate_classification_is_explicit() -> None:
    contract = read("stage3_gate_and_objective_contract.json")
    assert contract["hard_constraints"]
    assert contract["optimization_objectives"]
    assert contract["report_only_metrics"]
    assert all(item["status"] for item in contract["hard_constraints"])


def test_data_split_and_adaptive_contracts_are_frozen_without_fabricated_data() -> None:
    data = read("stage3_data_contract.json")
    split = read("stage3_geometry_dataset_split_contract.json")
    adaptive = read("stage3_adaptive_discrete_collision_contract.json")
    assert data["stage01_baseline_identity"]["row_count"] == 181
    assert split["stage3_h1_state"]["dataset_records_present"] is False
    assert set(split["split_roles"]) == {"training", "validation", "test", "generalization"}
    for implementation in adaptive["implementations"]:
        assert implementation["recursive_subdivision_criterion"] == "none"
        assert implementation["maximum_recursion_depth"] is None
        assert implementation["minimum_segment_length"] is None
        assert implementation["source_sha256"]


def test_latest_h1_certificate_preserves_fail_closed_boundary() -> None:
    outputs = sorted(ROOT.glob("outputs/stage3_h1_research_contract_*/stage3_h1_terminal_certificate.json"))
    if not outputs:
        return
    certificate = json.loads(outputs[-1].read_text(encoding="utf-8"))
    assert certificate["new_fjt_goals_sent"] == 0
    assert certificate["send_goal_async_call_count"] == 0
    assert certificate["robot_motion_started"] == "NO"
    assert certificate["formal_ledger_mutated"] == "NO"
    assert certificate["formal_r2_immutable"] == "YES"
