from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/ik_graph_stage23a6_bullet_self_collision_audit/fr5_scaled_horseshoe_demo_v44"
BASE = ROOT / "outputs/ik_graph_stage23a4_backend_equivalent/fr5_scaled_horseshoe_demo_v44"


def rows(path: Path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_baseline_counts_and_three_run_hashes():
    assert [json.loads((BASE / f"{b}_run1_summary.json").read_text())["valid_node_count"] for b in ("fcl", "bullet")] == [2882, 2224]
    assert [json.loads((BASE / f"{b}_run1_summary.json").read_text())["valid_waypoint_count"] for b in ("fcl", "bullet")] == [720, 670]
    for b in ("fcl", "bullet"):
        assert len({sha(BASE / f"{b}_candidate_results_run{i}.jsonl") for i in (1, 2, 3)}) == 1


def test_split_classification_identity_and_exact_disputed_set():
    fcl = rows(OUT / "fcl_classification_run1.jsonl")
    bullet = rows(OUT / "bullet_classification_run1.jsonl")
    assert len(fcl) == len(bullet) == 3077
    disputed = {r["candidate_id"] for r in bullet if r["self_collision"]} - {r["candidate_id"] for r in fcl if r["self_collision"]}
    assert len(disputed) == 658
    for r in fcl + bullet:
        assert r["combined_collision"] == (r["self_collision"] or r["robot_world_collision"])
        assert set(r["combined_collision_pairs"]) == set(r["self_collision_pairs"]) | set(r["robot_world_collision_pairs"])


def test_robot_world_difference_and_lost_waypoints():
    fcl = {r["candidate_id"] for r in rows(BASE / "fcl_candidate_results_run1.jsonl") if r["robot_world_collision"]}
    bullet = {r["candidate_id"] for r in rows(BASE / "bullet_candidate_results_run1.jsonl") if r["robot_world_collision"]}
    assert fcl == bullet
    lost = json.loads((OUT / "candidate_classification/lost_bullet_waypoint_details.json").read_text())
    assert lost["count"] == 50


def test_acm_hash_and_no_modification_gate():
    acm = json.loads((OUT / "acm_audit/original_acm.sha256").read_text())
    assert acm["equal"] is True
    assert json.loads((OUT / "stage23a6_gate_report.json").read_text())["gate"]["unjustified_acm_changes"] == 0


def test_unavailable_fields_are_explicit_and_downstream_blocked():
    row = next(r for r in rows(OUT / "bullet_classification_run1.jsonl") if r["self_collision"])
    contact = row["self_contacts"][0]
    assert contact["shape_index_1"] == "unavailable_moveit_contact_api"
    assert contact["distance"] == "unavailable_collision_contact_has_no_distance_field"
    gate = json.loads((OUT / "stage23a6_gate_report.json").read_text())
    assert gate["Stage_2_3A_6"]["status"] == "blocked_inconclusive_self_collision_disagreement"
    assert gate["Stage_2_3B"]["status"] == "blocked"
