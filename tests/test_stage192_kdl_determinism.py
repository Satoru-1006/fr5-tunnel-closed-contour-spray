"""Evidence and boundary tests for Stage 1.9.2.

These tests are intentionally ROS-independent; the ROS evidence is produced by
the replay tools and consumed here as immutable artifacts.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/ik_graph_stage192"
TARGET = "call:00000182"


def _json(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _records(path: str) -> list[dict]:
    return _json(path).get("records", [])


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_call_id_is_frozen_target() -> None:
    case = _json("outputs/ik_graph_stage192/minimal_case/minimal_reproduction_case.json")
    assert case["target_call_id"] == TARGET


def test_minimal_case_source_is_stage191() -> None:
    case = _json("outputs/ik_graph_stage192/minimal_case/minimal_reproduction_case.json")
    assert case["source_minimal_case"] == "outputs/ik_graph_stage191/minimal_reproduction_case.json"


def test_corpus_match_was_real() -> None:
    manifest = _json("outputs/ik_graph_stage192/minimal_case/input_manifest.json")
    assert manifest["corpus_match"]["status"] == "match"
    assert manifest["corpus_match"]["row"]["call_id"] == TARGET


def test_binary_snapshot_dimensions_and_byte_lengths() -> None:
    fields = _json("outputs/ik_graph_stage192/minimal_case/input_manifest.json")["binary_fields"]
    assert fields["target_transform_f64.bin"]["shape"] == [4, 4]
    assert fields["target_transform_f64.bin"]["byte_length"] == 128
    assert fields["seed_joint_vector_f64.bin"]["shape"] == [6]
    assert fields["all_robot_variables_f64.bin"]["shape"] == [18]
    assert fields["joint_limits_f64.bin"]["shape"] == [12]


@pytest.mark.parametrize("name", ["target_transform_f64.bin", "seed_joint_vector_f64.bin", "all_robot_variables_f64.bin", "joint_limits_f64.bin", "solver_options.bin"])
def test_binary_snapshot_hashes(name: str) -> None:
    manifest = _json("outputs/ik_graph_stage192/minimal_case/input_manifest.json")
    assert _sha(OUT / "minimal_case" / name) == manifest["binary_fields"][name]["sha256"]


def test_binary_format_is_little_endian_f64() -> None:
    fields = _json("outputs/ik_graph_stage192/minimal_case/input_manifest.json")["binary_fields"]
    assert fields["target_transform_f64.bin"]["dtype"] == "float64"
    assert fields["target_transform_f64.bin"]["endianness"] == "little"


def test_solver_plugin_and_api_are_kdl() -> None:
    manifest = _json("outputs/ik_graph_stage192/minimal_case/input_manifest.json")
    assert manifest["solver_plugin_name"] == "kdl_kinematics_plugin/KDLKinematicsPlugin"
    assert "set_from_ik" in manifest["solver_api_entry"]


def test_moveitpy_creates_new_robot_state_each_call() -> None:
    rows = _records("outputs/ik_graph_stage192/moveitpy_same_instance/replay.json")
    assert len(rows) == 100
    assert all(row["new_robot_state_per_call"] for row in rows)


def test_same_and_new_solver_modes_are_distinct_in_source() -> None:
    text = (ROOT / "tools/run_stage192_moveitpy_replay.py").read_text(encoding="utf-8")
    assert "mode in (\"same_instance\", \"single\")" in text
    assert "mode == \"new_instance\"" in text


def test_single_threaded_and_unrelated_callbacks_disabled() -> None:
    rows = _records("outputs/ik_graph_stage192/moveitpy_same_instance/replay.json")
    runtime = _json("outputs/ik_graph_stage192/moveitpy_same_instance/replay.json")["runtime"]
    assert runtime["executor"] == "SingleThreadedExecutor"
    assert runtime["unrelated_callbacks_started"] is False
    assert all(row["actual_attempt_count"] == 1 for row in rows)


def test_formal_replay_counts() -> None:
    matrix = (OUT / "stage192_determinism_matrix.csv").read_text(encoding="utf-8")
    for count in ("moveitpy_same_instance,100", "moveitpy_new_instance,50", "moveitpy_independent_processes,20", "cpp_moveit_plugin_same_instance,100", "direct_orocos_kdl_same_instance,100"):
        assert count in matrix


def test_independent_process_count_is_twenty() -> None:
    assert len(_records("outputs/ik_graph_stage192/moveitpy_independent_processes/comparison.json")) == 20
    assert len(_records("outputs/ik_graph_stage192/cpp_moveit_plugin_independent_processes/comparison.json")) == 20
    assert len(_records("outputs/ik_graph_stage192/direct_orocos_kdl/independent_processes/comparison.json")) == 20


@pytest.mark.parametrize("path", ["outputs/ik_graph_stage192/moveitpy_same_instance/replay.json", "outputs/ik_graph_stage192/moveitpy_new_instance/child_processes/comparison.json", "outputs/ik_graph_stage192/moveitpy_independent_processes/comparison.json", "outputs/ik_graph_stage192/cpp_moveit_plugin_same_instance/replay.json", "outputs/ik_graph_stage192/direct_orocos_kdl/same_instance.json"])
def test_replay_input_hashes_are_identical(path: str) -> None:
    rows = _records(path)
    keys = ("target_transform_raw_sha256", "seed_vector_raw_sha256", "all_robot_variables_raw_sha256", "solver_options_raw_sha256")
    assert all(len({row[key] for row in rows}) == 1 for key in keys)


def test_first_difference_is_solution_not_input() -> None:
    report = _json("outputs/ik_graph_stage192/first_difference_report.json")
    assert report["call_id"] == TARGET
    assert report["input_hashes_consistent_across_formal_same_instance"] is True
    assert "solver_success" in report["differences"] or "solution_raw_sha256" in report["differences"]


def test_raw_and_quantized_difference_are_recorded() -> None:
    report = _json("outputs/ik_graph_stage192/first_difference_report.json")
    assert "solution_raw_sha256" in report["differences"]
    assert "solution_quantized_sha256" in report["differences"]


def test_layer_matrix_classification_matches_evidence() -> None:
    report = _json("outputs/ik_graph_stage192/solver_diagnosis_report.json")
    assert report["first_unstable_layer"] == "moveitpy"
    assert report["root_cause_classification"] == "moveitpy_binding_or_lifecycle"


def test_cpp_plugin_is_get_position_ik_minimal() -> None:
    source = (ROOT / "cpp/stage192_moveit_kdl_minimal.cpp").read_text(encoding="utf-8")
    assert "getPositionIK" in source
    assert "RobotState" in source
    assert "planning" not in source.lower().replace("planning_scene", "")


def test_direct_kdl_does_not_call_moveitpy() -> None:
    source = (ROOT / "cpp/stage192_direct_orocos_kdl.cpp").read_text(encoding="utf-8")
    assert "ChainIkSolverPos_NR_JL" in source
    assert "MoveItPy" not in source


def test_sanitizer_gate_not_enabled_without_cpp_instability() -> None:
    report = _json("outputs/ik_graph_stage192/solver_diagnosis_report.json")
    assert report["sanitizer_status"] == "not_enabled_cpp_reproduction_was_stable_failure"


def test_no_ikfast_before_kdl_diagnosis() -> None:
    config = (ROOT / "config/stage192_kdl_determinism.yaml").read_text(encoding="utf-8")
    assert "enabled_only_if_kdl_fix_fails: true" in config
    assert "use_as_determinism_baseline: false" in config


def test_success_is_not_hardcoded() -> None:
    source = (ROOT / "tools/run_stage192_moveitpy_replay.py").read_text(encoding="utf-8")
    assert "0.01106337094396438" not in source
    assert "2.16305071112227" not in source


def test_full_regression_is_gated() -> None:
    report = _json("outputs/ik_graph_stage192/deterministic_fix_report.json")
    assert report["full_stage191_regression"] == "not_run_gate_not_met"


def test_stage2_is_blocked() -> None:
    report = _json("outputs/ik_graph_stage192/solver_diagnosis_report.json")
    assert report["stage2_status"] == "blocked"
    assert report["deterministic_solver_fix_complete"] is False


def test_ccd_and_clearance_remain_unavailable() -> None:
    report = _json("outputs/ik_graph_stage192/solver_diagnosis_report.json")
    assert report["ccd_status"] == "not_available"
    assert report["clearance_status"] == "not_available"


def test_teardown_is_separate_from_algorithm_status() -> None:
    report = _json("outputs/ik_graph_stage192/solver_diagnosis_report.json")
    assert report["teardown_status"] == "tracked_separately"
    assert report["moveitpy_teardown_segmentation_fault_observed"] is True


def test_protected_outputs_unchanged() -> None:
    report = _json("outputs/ik_graph_stage192/solver_diagnosis_report.json")
    assert report["protected_outputs_unchanged"] is True


def test_no_off_or_gnn_stage192_code() -> None:
    for path in [ROOT / "tools/run_stage192_moveitpy_replay.py", ROOT / "tools/run_stage192_solver_comparison.py", ROOT / "cpp/stage192_moveit_kdl_minimal.cpp", ROOT / "cpp/stage192_direct_orocos_kdl.cpp"]:
        text = path.read_text(encoding="utf-8").lower()
        assert "gnn" not in text
        assert "spray_off" not in text


def test_controlled_timeout_is_not_accepted_as_fix() -> None:
    report = _json("outputs/ik_graph_stage192/deterministic_fix_report.json")
    assert report["accepted_fix"] is None
    assert report["candidate_control"] == "MoveItPy timeout=0.005s"


def test_model_metadata_mismatch_is_visible() -> None:
    report = _json("outputs/ik_graph_stage192/solver_diagnosis_report.json")
    warning = report["input_identity_warning"]
    assert warning["status"] == "metadata_mismatch_requires_reproduction_caveat"
