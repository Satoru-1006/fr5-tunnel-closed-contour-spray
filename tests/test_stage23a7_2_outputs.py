import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/ik_graph_stage23a7_2_bullet_exact_replay/fr5_scaled_horseshoe_demo_v45"


def test_gate_preserves_formal_configuration():
    gate = json.loads((OUT / "stage23a7_2_gate_report.json").read_text(encoding="utf-8"))
    assert gate["Stage_2_3A_7_2"]["formal_files_modified"] is False
    assert gate["Stage_2_3B"]["status"] == "blocked"


def test_replay_population_and_groups_are_complete():
    with (OUT / "exact_replay_results.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 658
    assert {r["group"] for r in rows} == {"A", "B", "C", "D"}
    assert len({r["candidate_id"] for r in rows}) == 658
    assert all(r["leaf_pair_match"] == "true" for r in rows if r["group"] in {"A", "B"})


def test_coverage_keeps_certified_separate():
    report = json.loads((OUT / "waypoint_coverage_report.json").read_text(encoding="utf-8"))
    assert report["symmetric_difference_nodes"] == 658
    assert report["bullet_valid_subset_of_fcl_valid"] is True
    assert report["certified_waypoints"] == "0/720"


def test_three_run_determinism_manifest_exists():
    d = json.loads((OUT / "determinism_manifest.json").read_text(encoding="utf-8"))
    assert d["independent_runs"] == 3
    assert len(d["capture_shape_hashes"]) == 3
    assert len(d["replay_result_hashes"]) == 3
    assert len(d["double_precision_replay_result_hashes"]) == 3
    assert d["double_precision_replay_files_byte_identical"] is True


def test_runtime_shape_vertex_manifest_is_complete():
    m = json.loads((OUT / "runtime_shape_vertex_manifest.json").read_text(encoding="utf-8"))
    assert m["disputed_nodes"] == 658
    assert m["disputed_pair_rows"] == 992
    assert m["raw_scalar_bytes"] == 4
    assert len(m["runtime_vertex_file_sha256"]) == 64


def test_native_shape_provenance_has_vertices_scaling_and_margin():
    rows = [json.loads(x) for x in (OUT / "runtime_shape_provenance.jsonl").read_text(encoding="utf-8").splitlines() if x]
    target = [r for r in rows if r["body_name"] in {"forearm_link", "wrist2_link", "wrist3_link"} and r["child_index"] == -1]
    assert len(target) == 3
    assert all(r["convex_point_count"] > 0 for r in target)
    assert all(r["vertex_scalar_bytes"] == 4 for r in target)
    assert all(r["child_margin"] == 0 for r in target)
    assert all(r["child_local_scaling"] == [1, 1, 1] for r in target)


def test_exact_replay_group_semantics():
    with (OUT / "exact_replay_results.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert sum(r["group"] == "A" and r["sign_match"] == "true" for r in rows) == 992
    assert all(r["algorithm_match"] == "true" for r in rows if r["group"] in {"A", "B"})
    assert all(r["leaf_pair_match"] == "false" for r in rows if r["group"] == "D")


def test_zero_margin_and_double_precision_are_isolated():
    with (OUT / "group_c_double_precision.csv").open(encoding="utf-8", newline="") as f:
        c = list(csv.DictReader(f))
    assert len(c) == 992
    provenance = json.loads((OUT / "double_precision_library_provenance.json").read_text(encoding="utf-8"))
    assert provenance["sizeof_btScalar"] == 8
    assert provenance["BT_USE_DOUBLE_PRECISION"] is True
    assert all(r["algorithm_match"] == "true" for r in c)
    assert (OUT / "group_b_zero_margin.csv").exists()
    assert (OUT / "group_d_triangle_representation.csv").exists()


def test_perturbation_audit_is_complete_and_diagnostic_only():
    p = json.loads((OUT / "perturbation_results.json").read_text(encoding="utf-8"))
    assert p["status"] == "captured"
    assert p["nodes"] == 658
    assert p["states"] == 94752


def test_formal_hash_audit_is_clean():
    p = json.loads((OUT / "formal_file_hash_audit.json").read_text(encoding="utf-8"))
    assert p["formal_configuration_modified_by_stage23a7_2"] is False
    assert p["records"]
    assert all(x["matches_frozen"] for x in p["records"])
