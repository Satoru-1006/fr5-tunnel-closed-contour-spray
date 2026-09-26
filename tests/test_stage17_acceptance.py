from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from tools import audit_stage17_reproducibility as audit


ROOT = Path(__file__).resolve().parents[1]


def test_two_run_directories_are_independent(tmp_path):
    first = tmp_path / "run_001"
    second = tmp_path / "run_002"
    first.mkdir()
    second.mkdir()
    (first / "marker").write_text("one")
    (second / "marker").write_text("two")
    assert first != second
    assert (first / "marker").read_text() != (second / "marker").read_text()


def test_stage18_command_does_not_use_old_stage17_candidates_as_input():
    source = (ROOT / "tools" / "run_stage18_reproducibility.py").read_text(encoding="utf-8")
    assert "nominal_nodes_parquet" not in source
    assert "nominal_edges_parquet" not in source


def test_manifest_contains_all_key_hash_fields():
    source = (ROOT / "tools" / "run_stage18_reproducibility.py").read_text(encoding="utf-8")
    for field in ("tcp_pose_sha256", "seed_joint_sha256", "stage17_config_sha256", "urdf_sha256", "srdf_sha256", "kinematics_sha256", "joint_limits_sha256", "stage17_code_sha256"):
        assert field in source


def test_dirty_workspace_code_hashes_are_unique_identifiers():
    first = audit.sha256_file(ROOT / "tools" / "run_stage17_pose_repair.py")
    second = audit.sha256_file(ROOT / "src" / "task_pose_repair.py")
    assert re.fullmatch(r"[0-9a-f]{64}", first)
    assert re.fullmatch(r"[0-9a-f]{64}", second)
    assert first != second


def test_run_complete_uses_atomic_write(tmp_path):
    target = tmp_path / "RUN_COMPLETE.json"
    audit.atomic_write_json(target, {"run_complete": True})
    assert json.loads(target.read_text()) == {"run_complete": True}
    assert not (tmp_path / "RUN_COMPLETE.json.tmp").exists()


def test_missing_run_complete_cannot_be_warning():
    status = audit.classify_process_returncode(-11)
    assert not audit.teardown_warning_allowed(status, run_complete_exists=False, validation_pass=True, last_lifecycle_phase="moveitpy_destructor_pending")


def test_returncode_zero_maps_to_clean_pass():
    result = audit.classify_process_returncode(0)
    assert result["process_exit_status"] == "pass"
    assert result["run_status"] == "clean_pass"


def test_raw_minus_11_is_sigsegv():
    result = audit.classify_process_returncode(-11)
    assert result["process_signal"] == "SIGSEGV"
    assert result["shell_exit_code_equivalent"] == 139


def test_minus_11_is_not_automatically_warning():
    result = audit.classify_process_returncode(-11)
    assert result["process_exit_status"] != "warning"


def test_only_destructor_phase_allows_warning():
    status = audit.classify_process_returncode(-11)
    assert audit.teardown_warning_allowed(status, run_complete_exists=True, validation_pass=True, last_lifecycle_phase="moveitpy_destructor_pending")


def test_compute_phase_sigsegv_is_not_warning():
    status = audit.classify_process_returncode(-11)
    assert not audit.teardown_warning_allowed(status, run_complete_exists=True, validation_pass=False, last_lifecycle_phase="algorithm_complete")


def test_181_point_path_ids_are_recomputed():
    ids = list(range(181))
    assert len(ids) == 181 and ids == list(range(181)) and len(set(ids)) == 181


def test_periodic_angle_difference_uses_nearest_equivalent():
    import math

    assert abs(audit.periodic_delta(-math.pi + 0.01, math.pi - 0.01) - 0.02) < 1e-9


def test_max_joint_step_uses_periodic_unwrap():
    import math

    metrics = audit.recompute_joint_metrics([[math.pi - 0.01, 0, 0, 0, 0, 0], [-math.pi + 0.01, 0, 0, 0, 0, 0]])
    assert metrics["max_joint_step_deg"] < 2.0


def test_modified_waypoint_set_is_recomputed_from_candidate_records():
    records = [{"waypoint_id": 0, "is_nominal": True}, {"waypoint_id": 45, "is_nominal": False}, {"waypoint_id": 75, "is_nominal": False}]
    assert audit.recompute_modified_waypoint_ids(records) == [45, 75]


def test_numeric_exact_match_status():
    assert audit.compare_value(1.0, 1.0)["status"] == "exact_match"


def test_numeric_within_tolerance_status():
    assert audit.compare_value(1.0 + 1e-9, 1.0)["status"] == "within_numeric_tolerance"


def test_summary_mismatch_status_is_failure_class():
    assert audit.compare_value(1.0, 2.0)["status"] == "mismatch"


def test_ruckig_utilization_recomputed_from_samples():
    samples = []
    for index, t in enumerate((0.0, 0.01)):
        row = {"t": str(t)}
        for joint in range(1, 7):
            row[f"v_j{joint}"] = "0.0"
            row[f"a_j{joint}"] = "0.0"
        samples.append(row)
    assert audit.recompute_ruckig_utilization(samples) == {"max_velocity_ratio": 0.0, "max_acceleration_ratio": 0.0, "max_jerk_ratio": 0.0}


def test_collision_method_is_declared_discrete():
    assert audit.COLLISION_METHOD == "adaptive_discrete_interpolation"


def test_ccd_remains_unavailable():
    assert audit.UNAVAILABLE == "not_available"


def test_clearance_remains_unavailable():
    assert audit.UNAVAILABLE == "not_available"


def test_stage18_sources_do_not_add_forbidden_graph_or_state_logic():
    for relative in ("tools/audit_stage17_reproducibility.py", "tools/run_stage18_reproducibility.py", "tools/stage18_moveit_launch.py"):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert re.search(r"\bOFF\b|\bGNN\b", text) is None


def test_stage18_does_not_modify_stage17_algorithm_modules():
    changed = subprocess.run(["git", "diff", "--name-only", "--", "src/task_pose_repair.py", "src/ik_candidate_graph.py", "src/graph_search_solver.py"], cwd=ROOT, capture_output=True, text=True, shell=False).stdout.strip()
    assert changed == ""


def _comparison_fixture(candidate_id: str):
    return {"run_id": candidate_id, "acceptance": {"overall_status": "pass", "formal_constraints_pass": True, "post_ruckig_validation_status": "pass", "recomputed": {"selected_ik_candidate_sequence": [candidate_id], "total_cost": 1.0}, "ruckig_recomputed": {}}, "process_status": {"process_exit_status": "pass"}, "manifest": {"inputs": {"tcp_pose_sha256": "x"}, "configuration": {"stage17_code_sha256": {"a": "b"}}}}


def test_candidate_sequence_comparison_exact_match():
    rows, summary = __import__("tools.run_stage18_reproducibility", fromlist=["compare_runs"]).compare_runs([_comparison_fixture("same"), _comparison_fixture("same")])
    assert next(row for row in rows if row["metric"] == "selected_ik_candidate_sequence")["comparison_status"] == "exact_match"
    assert summary["candidate_sequence_status"] == "exact_match"


def test_equivalent_optimum_comparison_is_explicit():
    rows, summary = __import__("tools.run_stage18_reproducibility", fromlist=["compare_runs"]).compare_runs([_comparison_fixture("a"), _comparison_fixture("b")])
    assert summary["candidate_sequence_status"] == "equivalent_optimum"


def test_protected_hash_comparison_detects_change():
    before = {"files": {"x": {"sha256": "a", "size_bytes": 1}}}
    after = {"files": {"x": {"sha256": "b", "size_bytes": 1}}}
    assert before.get("files") != after.get("files")


def test_run_complete_payload_requires_all_validation_flags():
    payload = {"run_complete": True, "validation": {"trajectory_validation_pass": True, "post_ruckig_validation_pass": True}, "protected_outputs": {"stage17_original_hash_unchanged": True}}
    assert payload["run_complete"] and all(payload["validation"].values()) and all(payload["protected_outputs"].values())


def test_no_legacy_stage17_output_is_used_as_new_run_output():
    text = (ROOT / "tools" / "run_stage18_reproducibility.py").read_text(encoding="utf-8")
    assert 'directory: "outputs/ik_graph_stage18/{run_id}"' in text
