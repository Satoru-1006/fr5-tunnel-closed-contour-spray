import csv, json, hashlib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/ik_graph_stage23a7_1_runtime_audit/fr5_scaled_horseshoe_demo_v44"
REQ = [
    "stage23a7_1_report.md", "stage23a7_1_gate_report.json", "frozen_input_manifest.json", "frozen_input_hashes.json",
    "runtime_provenance.json", "loaded_shared_libraries.csv", "loaded_library_hashes.json", "collision_request_manifest.jsonl",
    "collision_request_result_comparison.csv", "robot_state_update_audit.jsonl", "runtime_link_transforms.parquet",
    "moveit_padding_scale_manifest.json", "bullet_runtime_shape_manifest.jsonl", "bullet_runtime_shape_tree.jsonl",
    "bullet_runtime_aabbs.parquet", "bullet_runtime_manifolds.jsonl", "bullet_runtime_contacts.parquet",
    "bullet_moveit_contact_mapping.csv", "acm_case_definitions.json", "acm_case_results.parquet", "acm_pair_coverage.json",
    "acm_non_target_invariance_report.json", "candidate_runtime_classification.csv", "joint_perturbation_results.parquet",
    "perturbation_transition_summary.json", "perturbation_monotonicity_report.json", "full_candidate_backend_results.csv",
    "waypoint_runtime_coverage.csv", "three_run_reproducibility.json", "canonical_artifact_hashes.json",
]

def jlines(name):
    return [json.loads(x) for x in (OUT / name).read_text(encoding="utf-8").splitlines() if x.strip()]

def test_required_artifacts_and_frozen_counts():
    assert all((OUT / x).exists() for x in REQ)
    m = json.loads((OUT / "frozen_input_manifest.json").read_text())
    assert m["counts"] == {"waypoints": 720, "candidates": 3077, "disputed_nodes": 658, "independent_pair_records": 992}
    for rel, digest in json.loads((OUT / "frozen_input_hashes.json").read_text()).items():
        if digest:
            p = ROOT / rel
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            assert h == digest

def test_runtime_detector_request_and_result_clear():
    p = json.loads((OUT / "runtime_provenance.json").read_text())
    assert p["runtime_detector_proven"] is True
    assert p["bullet"]["collision_environment_concrete_class"] == "collision_detection::CollisionEnvBullet"
    assert p["bullet"]["sizeof_btScalar"] == 4
    r = jlines("collision_request_manifest.jsonl")
    assert len(r) == 3077 * 2 * 2
    required = {"invocation_id", "backend", "api_function", "overload_signature", "group_name", "pad_environment_collisions", "pad_self_collisions", "distance", "detailed_distance", "cost", "contacts", "max_contacts", "max_contacts_per_pair", "max_cost_sources", "is_done_is_null", "verbose"}
    assert required <= set(r[0])
    c = list(csv.DictReader((OUT / "collision_request_result_comparison.csv").open(encoding="utf-8")))
    assert len(c) == 3077 * 2 and all(x["collision_equal"] == "True" for x in c) and all(x["result_clear_called"] == "True" for x in c)

def test_state_shape_and_manifold_evidence_is_explicit():
    s = jlines("robot_state_update_audit.jsonl")
    assert len(s) == 3077 * 2
    assert all(x["collision_body_update_called"] and not x["dirty_collision_body_transforms_after"] for x in s)
    tree = jlines("bullet_runtime_shape_tree.jsonl")
    names = {x["body_name"] for x in tree}
    assert {"forearm_link", "wrist2_link", "wrist3_link", "horseshoe_collision_compound"} <= names
    world = [x for x in tree if x["body_name"] == "horseshoe_collision_compound"]
    assert world and world[0]["is_compound"] and world[0]["child_count"] == 131
    assert all(x["moveit_link_padding"] == 0.0 and x["moveit_link_scale"] == 1.0 for x in tree)
    manifold = jlines("bullet_runtime_manifolds.jsonl")
    assert manifold and all(x["status"] == "callback_observed" for x in manifold)
    assert all(isinstance(x["m_distance1"], (int, float)) and x["m_distance1_hex"].startswith("0x") for x in manifold)

def test_acm_perturbation_classification_and_full_replay():
    acm = pd.read_parquet(OUT / "acm_case_results.parquet")
    assert len(acm) == 658 * 6 * 2
    cls = list(csv.DictReader((OUT / "candidate_runtime_classification.csv").open(encoding="utf-8")))
    assert len(cls) == 658 and {x["final_classification"] for x in cls} == {"runtime_geometry_representation_mismatch"}
    full = list(csv.DictReader((OUT / "full_candidate_backend_results.csv").open(encoding="utf-8")))
    assert len(full) == 3077
    gate = json.loads((OUT / "stage23a7_1_gate_report.json").read_text())
    assert gate["Stage_2_3A_7_1"]["backend_validity"]["symmetric_difference_count"] == 658
    pert = pd.read_parquet(OUT / "joint_perturbation_results.parquet")
    assert len(pert) == 658 * 6 * 8 * 2
    assert pert["from_original_frozen_state"].all()

def test_waypoint_aggregation_and_downstream_gate():
    cov = list(csv.DictReader((OUT / "waypoint_runtime_coverage.csv").open(encoding="utf-8")))
    assert len(cov) == 720
    assert all(x["formally_covered"] == "False" and x["formal_valid_nodes"] == "0" for x in cov)
    gate = json.loads((OUT / "stage23a7_1_gate_report.json").read_text())
    assert gate["Stage_2_3A_7_1"]["status"] == "blocked_runtime_geometry_representation_mismatch"
    assert gate["Stage_2_3B"]["status"] == "blocked"
    rep = json.loads((OUT / "three_run_reproducibility.json").read_text())
    assert rep["all_runtime_result_hashes_identical"] is True
    assert rep["three_run_reproducibility"] is True

def test_callback_mapping_and_disputed_node_coverage():
    raw = jlines("bullet_runtime_manifolds.jsonl")
    disputed = {r["node_id"] for r in csv.DictReader((OUT / "candidate_runtime_classification.csv").open(encoding="utf-8"))}
    r1 = [r for r in raw if "_R1" in r.get("invocation_id", "") and r.get("node_id") in disputed]
    assert len({r["node_id"] for r in r1}) == 658
    assert all(r["m_distance1"] <= 0 for r in r1)
    mapping = list(csv.DictReader((OUT / "bullet_moveit_contact_mapping.csv").open(encoding="utf-8")))
    assert mapping and all(x["bullet_m_distance1"] != "not_available" for x in mapping)
    assert all(abs(float(x["depth_minus_m_distance1"])) < 1e-6 for x in mapping)

def test_acm_cases_are_explicit_and_routed():
    acm = pd.read_parquet(OUT / "acm_case_results.parquet")
    assert set(acm["case_id"]) == {"A", "B1", "B2", "B3", "C1", "C2"}
    assert all((acm.groupby(["backend", "case_id"]).size() == 658).to_numpy())
    report = json.loads((OUT / "acm_non_target_invariance_report.json").read_text())
    assert report["formal_acm_hash_unchanged"] is True
    assert report["non_target_pair_invariance"] is True
    assert report["target_routing_observed"] is True

def test_three_run_shape_raw_and_library_evidence():
    rep = json.loads((OUT / "three_run_reproducibility.json").read_text())
    assert rep["shape_manifest_three_run_comparison"] is True
    assert rep["raw_manifold_three_run_comparison"] is True
    prov = json.loads((OUT / "runtime_provenance.json").read_text())
    assert prov["bullet"]["runtime_detector_proven"] is True
    assert prov["bullet"]["loaded_library_paths"]
    libs = json.loads((OUT / "loaded_library_hashes.json").read_text())
    assert libs and all(len(v) == 64 for v in libs.values())

def test_perturbation_raw_callback_evidence_and_backend_sets():
    p = pd.read_parquet(OUT / "joint_perturbation_results.parquet")
    assert len(p) == 658 * 6 * 8 * 2
    assert p["from_original_frozen_state"].all()
    raw = jlines("bullet_runtime_manifolds.jsonl")
    assert json.loads((OUT / "perturbation_transition_summary.json").read_text())["raw_bullet_callback_rows"] == 47616
    gate = json.loads((OUT / "stage23a7_1_gate_report.json").read_text())["Stage_2_3A_7_1"]
    assert gate["gate"]["backend_symmetric_difference_count"] == 658
    assert gate["gate"]["fcl_valid_node_set_sha256_equals_bullet"] is False

def test_required_test_logs_and_mutually_exclusive_classification():
    assert (OUT / "test_results.txt").exists()
    assert (OUT / "test_command_manifest.json").exists()
    rows_ = list(csv.DictReader((OUT / "candidate_runtime_classification.csv").open(encoding="utf-8")))
    allowed = {"backend_agreed_valid", "true_self_collision_confirmed", "runtime_geometry_representation_mismatch", "bullet_narrowphase_false_positive_confirmed", "numeric_boundary_inconclusive", "acm_routing_anomaly", "robot_state_transform_anomaly", "collision_request_execution_anomaly", "instrumentation_incomplete"}
    assert all(r["final_classification"] in allowed and r["formal_valid"] == "False" for r in rows_)
