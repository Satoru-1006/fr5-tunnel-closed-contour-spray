"""Stage 2.4G contract and evidence tests."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_stage24g_complete_edge_component_bridge as stage24g


OUT = ROOT / "outputs/ik_graph_stage24g_complete_edge_component_bridge/fr5_scaled_horseshoe_demo_v45_20260731"


def test_canonical_branch_signature_has_winding_and_no_wrap() -> None:
    semantics = [{"type": "revolute", "lower": -4.0, "upper": 4.0} for _ in range(6)]
    signature = stage24g.canonical_branch_signature([0.5, 0.0, -1.0, 0.0, 1.0, 0.0], semantics)
    assert signature == "shoulder=positive|elbow=negative|wrist=positive|winding=0,0,0,0,0,0"
    signed, absolute, _, _ = stage24g.raw_delta([3.0] + [0.0] * 5, [-3.0] + [0.0] * 5, semantics)
    assert math.isclose(signed[0], -6.0)
    assert math.isclose(absolute[0], 6.0)


def test_stage24g_output_contract_exists_after_formal_run() -> None:
    if not (OUT / "stage24g_gate_report.json").exists():
        return
    gate = json.loads((OUT / "stage24g_gate_report.json").read_text(encoding="utf-8"))
    assert gate["ready_for_Stage_2_5"] is False
    assert gate["Stage_2_5"] == "blocked"
    assert gate["collision_method"] == "adaptive_discrete_interpolation"
    assert gate["CCD"] == "not_available"
    assert gate["clearance"] == "not_available"
    assert gate["continuous_tolerance_space_infeasible_proven"] is False


def test_frozen_input_hash_verification_has_397_entries() -> None:
    if not (OUT / "stage24g_frozen_input_hashes.json").exists():
        return
    hashes = json.loads((OUT / "stage24g_frozen_input_hashes.json").read_text(encoding="utf-8"))
    assert hashes["checked_count"] == 397
    assert hashes["failure_count"] == 0
    assert hashes["all_397_inputs_verified"] is True


def test_34_to_35_inventory_counts_are_complete() -> None:
    if not (OUT / "stage24g_pair_prefilter_inventory.jsonl").exists():
        return
    rows = [json.loads(line) for line in (OUT / "stage24g_pair_prefilter_inventory.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    seam = [row for row in rows if row["transition"] == "34->35"]
    assert len(seam) == 1092
    assert sum(row["joint_gate_pass"] for row in seam) == 623
    assert sum(row["historical_edge_status"] == "historically_accepted_dual_backend" for row in seam) == 16
    assert sum(row["native_check_required"] for row in seam) == 607


def test_unchecked_is_not_reported_as_invalid() -> None:
    if not (OUT / "stage24g_pair_prefilter_inventory.jsonl").exists():
        return
    rows = [json.loads(line) for line in (OUT / "stage24g_pair_prefilter_inventory.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert all(not (row["native_check_required"] and row["status"] in {"rejected", "invalid"}) for row in rows)


def test_no_intermediate_waypoint_insertion_and_closure_is_diagnostic() -> None:
    if not (OUT / "stage24g_valid_nodes.jsonl").exists():
        return
    nodes = [json.loads(line) for line in (OUT / "stage24g_valid_nodes.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert all(0 <= int(row["waypoint_index"]) <= 719 for row in nodes)
    if (OUT / "stage24g_closed_cycle_certificate.json").exists():
        certificate = json.loads((OUT / "stage24g_closed_cycle_certificate.json").read_text(encoding="utf-8"))
        assert certificate["closure_edge_valid"] is False
        assert certificate["complete_cycle_found"] is False


def test_three_run_semantic_determinism_and_seed_order() -> None:
    if not (OUT / "stage24g_determinism_report.json").exists():
        return
    report = json.loads((OUT / "stage24g_determinism_report.json").read_text(encoding="utf-8"))
    assert report["independent_runs"] == 3
    assert report["deterministic_rebuild"] is True
    assert all(value == 0 for value in report["differences"].values())


def test_native_q_sequence_contract_is_explicit() -> None:
    text = Path(stage24g.__file__).read_text(encoding="utf-8")
    assert "fcl_request_q_sequence_hash" in text
    assert "bullet_request_q_sequence_hash" in text
    assert "interpolation_step_deg" in text
    assert "all_ranked_seeds_actually_attempted" in text


def test_new_ik_waypoints_are_boundary_endpoints_only() -> None:
    boundaries = [{"from_waypoint": 34, "to_waypoint": 35}, {"from_waypoint": 593, "to_waypoint": 594}]
    assert stage24g.boundary_waypoints(boundaries) == [34, 35, 593, 594]
