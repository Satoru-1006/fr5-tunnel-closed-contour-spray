import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/ik_graph_stage23b_representation_remediation/fr5_scaled_horseshoe_demo_v45"
REPLAY = ROOT / "outputs/ik_graph_stage23a7_2_bullet_exact_replay/fr5_scaled_horseshoe_demo_v45"


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_stage23b_gate_is_formally_passed():
    gate = json.loads((OUT / "stage23b_gate_report.json").read_text(encoding="utf-8"))
    assert gate["Stage_2_3B"]["status"] == "passed"
    assert gate["Stage_2_4"]["status"] == "unblocked_not_started"
    assert gate["backend_equivalence"]["valid_node_symmetric_difference"] == 0
    assert gate["backend_equivalence"]["valid_waypoint_symmetric_difference"] == 0


def test_formal_runs_are_complete_and_deterministic():
    gate = json.loads((OUT / "stage23b_gate_report.json").read_text(encoding="utf-8"))
    prior = load_json(REPLAY / "stage23a7_2_gate_report.json")["Stage_2_3A_7_2"]["coverage"]
    assert gate["FCL"]["valid_nodes"] == prior["fcl"]
    assert gate["Bullet"]["valid_nodes"] == prior["fcl"]
    assert gate["FCL"]["valid_waypoints"] == prior["fcl_waypoints"]
    assert gate["Bullet"]["valid_waypoints"] == prior["fcl_waypoints"]
    assert gate["FCL"]["three_run_hash_equal"] is True
    assert gate["Bullet"]["three_run_hash_equal"] is True


def test_shape_manifest_has_three_compound_target_links():
    before = load_json(OUT / "stage23b_shape_manifest_before.json")
    manifest = load_json(OUT / "stage23b_shape_manifest_after.json")
    old_entries = {row["link_name"]: row for row in before["entries"]}
    entries = {row["link_name"]: row for row in manifest["entries"]}
    assert set(entries) == {"forearm_link", "wrist2_link", "wrist3_link"}
    assert all(row["root_bt_shape_type"] == "btConvexHullShape" for row in old_entries.values())
    assert all(row["root_bt_shape_type"] == "btCompoundShape" for row in entries.values())
    assert all(row["child_shape_count"] == row["source_triangle_count"] for row in entries.values())
    assert all(row["root_margin"] == 0 and row["child_margins"] == [0] for row in entries.values())
    assert all(row["child_shape_types"] == ["btTriangleShapeEx"] for row in entries.values())
    assert all(row["aabb_anomaly_count"] == 0 for row in entries.values())


def test_historical_658_and_precision_7_are_complete():
    with (OUT / "stage23b_historical_658_replay.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 658
    assert all(row["new_bullet_valid"] == "True" and row["fcl_valid"] == "True" for row in rows)
    with (OUT / "stage23b_precision_sensitive_7.csv").open(encoding="utf-8", newline="") as f:
        precision = list(csv.DictReader(f))
    assert len(precision) == 7
    assert all(row["auxiliary_only"] == "True" for row in precision)


def test_frozen_input_manifest_and_runtime_provenance():
    inputs = json.loads((OUT / "stage23b_input_manifest.json").read_text(encoding="utf-8"))
    runtime = json.loads((OUT / "stage23b_runtime_manifest.json").read_text(encoding="utf-8"))
    assert inputs["all_frozen_records_unchanged"] is True
    assert inputs["formal_acm_modified"] is False
    assert runtime["moveit_version"] == "2.12.4"
    assert runtime["bullet_version"] == "3.24"
    assert runtime["sizeof_btScalar"] == 4
    assert runtime["clean_process_exit_count"] == 6


def test_each_formal_run_exports_complete_sets_telemetry_and_clean_exit():
    for backend in ("fcl", "bullet"):
        for index in (1, 2, 3):
            run = load_json(OUT / f"stage23b_{backend}_run{index}.json")
            assert run["candidate_count"] == 3077
            assert len(run["valid_node_ids"]) + len(run["invalid_node_ids"]) == 3077
            assert len(run["valid_waypoint_ids"]) == 720
            assert run["process_exit_code"] == 0
            assert run["gnu_time_exit_status"] == 0
            assert run["peak_rss_kbytes"] > 0
            assert run["elapsed_wall_clock"]
            assert run["process_destroyed_cleanly"] is True


def test_each_candidate_has_frozen_hashes_and_no_inferred_clearance():
    required = {
        "shape_representation_manifest_hash",
        "acm_hash",
        "urdf_hash",
        "srdf_hash",
        "waypoint_hash",
        "candidate_set_hash",
    }
    for backend in ("fcl", "bullet"):
        path = OUT / "certified_runs_final3" / f"{backend}_run1" / f"{backend}_certified_results_run1.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 3077
        assert all(required <= row.keys() for row in rows)
        assert all(row["minimum_distance"] is None for row in rows)
        assert all(row["minimum_distance_status"] == "not_available" for row in rows)
        assert all(row["collision_method"] == "adaptive_discrete_interpolation" for row in rows)
        assert all(row["continuous_collision_detection"] == "not_available" for row in rows)


def test_exact_node_waypoint_and_pair_sets_match():
    nodes = load_json(OUT / "stage23b_node_set_comparison.json")
    waypoints = load_json(OUT / "stage23b_waypoint_set_comparison.json")
    assert nodes["fcl_valid_node_set"] == nodes["bullet_valid_node_set"]
    assert nodes["fcl_invalid_node_set"] == nodes["bullet_invalid_node_set"]
    assert nodes["valid_node_fcl_only"] == []
    assert nodes["valid_node_bullet_only"] == []
    assert nodes["self_collision_pair_symmetric_difference"] == 0
    assert nodes["robot_world_pair_symmetric_difference"] == 0
    assert waypoints["fcl_valid_waypoint_set"] == waypoints["bullet_valid_waypoint_set"]


def test_geometry_regression_has_no_known_false_positive_or_false_negative():
    probes = load_json(OUT / "stage23b_geometry_probe_results.json")
    direct = probes["direct_native_triangle_compound_probe"]
    cases = {row["id"]: row for row in probes["cases"]}
    assert cases["concave_cavity_free_space"]["new_bullet"] == "false"
    assert cases["true_triangle_intersection"]["new_bullet"] == "collision"
    assert cases["strict_positive_clearance"]["new_bullet"] == "free"
    assert direct["passed"] is True
    assert all(row["passed"] for row in direct["cases"])
    assert probes["runtime_aabb_anomaly_count"] == 0
    assert all(row["nonfinite_triangle_count"] == 0 for row in probes["target_mesh_triangle_audits"])


def test_precision_sensitive_cases_are_diagnostic_only_under_model_scale_perturbations():
    with (OUT / "stage23b_precision_sensitive_7.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 7
    assert all(row["perturbation_deltas_rad"] for row in rows)
    assert all(row["perturbation_sign_flip_count"] == "0" for row in rows)
    assert all(row["formal_node_set_affected"] == "False" for row in rows)
