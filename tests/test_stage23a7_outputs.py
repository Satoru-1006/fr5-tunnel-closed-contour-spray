from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/ik_graph_stage23a7_independent_geometry_bullet_runtime_audit/fr5_scaled_horseshoe_demo_v44"
SRC = ROOT / "outputs/ik_graph_stage23a6_bullet_self_collision_audit/fr5_scaled_horseshoe_demo_v44"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(name: str):
    with (OUT / name).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_frozen_input_and_exact_population():
    manifest = json.loads((OUT / "stage23a7_input_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "verified"
    assert all(item["match"] for item in manifest["files"])
    population = json.loads((OUT / "stage23a7_population_check.json").read_text(encoding="utf-8"))
    assert population["disputed_node_count"] == 658
    assert population["bullet_fully_invalid_waypoints"] == 50
    assert population["robot_world_difference_nodes"] == 0


def test_independent_geometry_covers_all_nodes_and_two_pairs():
    geometry = rows("stage23a7_independent_geometry.csv")
    assert len({r["node_id"] for r in geometry}) == 658
    assert {r["pair"] for r in geometry} == {"forearm_link<->wrist2_link", "forearm_link<->wrist3_link"}
    assert all(r["geometry_status"] in {"exact_triangle_intersection", "strictly_positive_separation", "near_zero_separation"} for r in geometry)
    assert all(float(r["minimum_distance_m"]) >= 0 for r in geometry)


def test_geometry_repetitions_are_deterministic():
    assert len({sha(OUT / f"geometry_run_{i}/stage23a7_independent_geometry.csv") for i in (1, 2, 3)}) == 1
    assert len({sha(OUT / f"geometry_run_{i}/stage23a7_fk_transforms.csv") for i in (1, 2, 3)}) == 1


def test_formal_acm_and_downstream_gate_are_preserved():
    acm = json.loads((OUT / "stage23a7_acm_ablation.json").read_text(encoding="utf-8"))
    assert acm["formal_acm_modified"] is False
    gate = json.loads((OUT / "stage23a7_gate_report.json").read_text(encoding="utf-8"))
    assert gate["Stage_2_3A_7"]["status"] == "blocked_bullet_runtime_instrumentation_incomplete"
    assert gate["Stage_2_3B"]["status"] == "blocked"
    assert gate["gate"]["downstream_started"] is False


def test_missing_runtime_fields_are_explicit():
    shapes = json.loads((OUT / "stage23a7_bullet_runtime_shapes.json").read_text(encoding="utf-8"))
    assert shapes["status"] == "instrumentation_not_available"
    for link in ("forearm_link", "wrist2_link", "wrist3_link"):
        assert shapes["fields"][link]["shape_type"] == "not_exposed_by_runtime_api"
    perturb = json.loads((OUT / "stage23a7_perturbation_status.json").read_text(encoding="utf-8"))
    assert perturb["status"] == "not_run"
