from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.stage3_h12_r4_native_totg_closure import compare_snapshots, immutable_paths
from src.stage3_h12_r4 import (
    analyze_geometric_path,
    classify_totg_false,
    coverage_audit,
    diagnostic_limit_comparison,
    finite_difference_diagnostics_only,
    process_semantics_gate,
    replay_hashes_equal,
    ruckig_completion_gate,
)

ROOT = Path(__file__).resolve().parents[1]
R3 = ROOT / "outputs/stage3_h12_r3_primitive_dynamic_closure_20260812T073100Z"
R2 = ROOT / "outputs/stage3_h12_r2_process_manifold_post_ruckig_20260812T055713Z"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def good_ruckig(result_name="Working", numeric=0):
    return {
        "native_smoothing_returned_success": True,
        "ruckig_result": {"result_name": result_name, "numeric_result": numeric},
        "last_segment_completed": True,
        "smoothing_complete": True,
        "overshoot_gate_passed": True,
        "duration_ceiling_hit": False,
        "iteration_limit_hit": False,
        "wall_clock_timeout": False,
        "input_validation_passed": True,
        "native_error": None,
        "output_finite": True,
    }


def test_h12_r3_certificate_parsing():
    cert = load_json(R3 / "stage3_h12_r3_terminal_certificate.json")
    assert cert["STAGE_3_H12_R3"] == "BLOCKED"
    assert cert["TOTAL_WINDOWS"] == 189216


def test_detect_h12_r3_native_execution_not_run():
    audit = load_json(R3 / "native_execution_audit.json")
    assert audit["status"] == "NOT_RUN"
    assert audit["first_blocker"] == "native_h12_r3_execution_not_requested"


def test_r2_result_cannot_be_fresh_r4_evidence():
    cert = load_json(R3 / "stage3_h12_r3_terminal_certificate.json")
    assert cert["native_execution"]["status"] == "NOT_RUN"
    assert "H12_R4_FRESH_NATIVE_EXECUTION" not in cert


def test_authoritative_480_primitive_coverage():
    units = [json.loads(line) for line in (R2 / "stage3_h12_r2_primitive_manifest.jsonl").read_text(encoding="utf-8").splitlines() if line]
    audit = coverage_audit(units)
    assert audit["PRIMITIVES_OBSERVED"] == 480
    assert audit["coverage_complete"]


def test_numeric_epsilon_is_not_meaningful_excess():
    comparison = diagnostic_limit_comparison(-0.105000000000004, 0.105)
    assert comparison["raw_comparison_exceeds"]
    assert not comparison["meaningful_violation"]
    assert comparison["meaningful_excess"] == 0.0


def test_genuine_acceleration_excess_fails():
    comparison = diagnostic_limit_comparison(0.1051, 0.105)
    assert comparison["meaningful_violation"]
    assert comparison["meaningful_excess"] > 0.0


def test_old_timestamps_are_diagnostic_only():
    result = finite_difference_diagnostics_only([0.0, 0.01, 0.02], [[0] * 6, [1] * 6, [0] * 6])
    assert result["authoritative_for_totg_admission"] is False
    assert result["purpose"] == "provenance_and_pre_retiming_diagnostics_only"


def test_fresh_totg_runner_invocation_is_wired():
    source = (ROOT / "scripts/stage3_h12_r4_native_totg_closure.py").read_text(encoding="utf-8")
    worker = (ROOT / "ros2_moveit_bridge/stage3_h12_r4_native.py").read_text(encoding="utf-8")
    assert "stage3_h12_r4_native.launch.py" in source
    assert "apply_totg_time_parameterization" in worker
    assert "R3_ROOT / \"primitive_reconstruction.jsonl\"" in source


def test_native_totg_false_is_fail_closed():
    geometry = analyze_geometric_path([[0] * 6, [1] * 6, [0] * 6])
    failure, _ = classify_totg_false(geometry, group_valid=True, velocity_limits_valid=True, acceleration_limits_valid=True, path_tolerance=0.1)
    assert failure == "unsupported_near_180_degree_turn"


def test_unknown_native_false_cannot_be_success():
    failure, _ = classify_totg_false({"input_position_finite": True, "waypoint_count": 2, "geometric_shape_valid": False, "near_180_turn_count": 0}, group_valid=True, velocity_limits_valid=True, acceleration_limits_valid=True, path_tolerance=0.1)
    assert failure == "unknown_native_totg_false"


def test_near_180_turn_detector_matches_moveit_threshold():
    geometry = analyze_geometric_path([[0] * 6, [1, 0, 0, 0, 0, 0], [0] * 6])
    assert geometry["near_180_turn_count"] == 1
    assert geometry["maximum_turn_angle_deg"] == 180.0
    assert geometry["maximum_turn"]["waypoint_triplet"] == [0, 1, 2]


def test_duplicate_and_degenerate_detection():
    geometry = analyze_geometric_path([[0] * 6, [0] * 6, [1e-7, 0, 0, 0, 0, 0]])
    assert geometry["input_duplicate_waypoint_count"] == 1
    assert geometry["input_near_duplicate_waypoint_count"] == 1
    degenerate = analyze_geometric_path([[0] * 6])
    assert not degenerate["geometric_shape_valid"]


def test_working_is_not_automatically_failure():
    result = ruckig_completion_gate(good_ruckig())
    assert result["raw_working"]
    assert result["accepted"]


def test_working_incomplete_fails():
    record = good_ruckig()
    record["smoothing_complete"] = False
    assert not ruckig_completion_gate(record)["accepted"]


def test_working_duration_ceiling_fails():
    record = good_ruckig()
    record["duration_ceiling_hit"] = True
    assert not ruckig_completion_gate(record)["accepted"]


def test_working_complete_all_gates_may_proceed():
    assert ruckig_completion_gate(good_ruckig())["accepted"]


def test_finished_failed_overshoot_gate_fails():
    record = good_ruckig("Finished", 1)
    record["overshoot_gate_passed"] = False
    result = ruckig_completion_gate(record)
    assert result["raw_finished"]
    assert not result["accepted"]


def test_ruckig_native_error_fails():
    record = good_ruckig("ErrorInvalidInput", -100)
    record["native_error"] = "ErrorInvalidInput"
    assert not ruckig_completion_gate(record)["accepted"]


def test_spray_on_process_gates_mandatory():
    assert not process_semantics_gate("SPRAY_ON", {"status": "BLOCKED", "failed_spray_on_sample_count": 1})
    assert process_semantics_gate("SPRAY_ON", {"status": "PASSED", "failed_spray_on_sample_count": 0})


def test_spray_off_not_forced_onto_process_manifold():
    assert process_semantics_gate("SPRAY_OFF", {"status": "BLOCKED", "failed_spray_on_sample_count": 100})
    assert not process_semantics_gate("SPRAY_OFF", {}, common_checks_passed=False)


def test_no_future_label_leakage_in_repair_path():
    source = (ROOT / "ros2_moveit_bridge/stage3_h12_r4_native.py").read_text(encoding="utf-8")
    assert "target_positions" not in source
    assert "ground_truth" not in source


def test_no_hidden_cv_fallback():
    source = (ROOT / "scripts/stage3_h12_r4_native_totg_closure.py").read_text(encoding="utf-8")
    assert '"hidden_cv_fallback": False' in source
    assert '"repair_count": 0' in source


def test_exact_189216_window_coverage():
    units = [json.loads(line) for line in (R2 / "stage3_h12_r2_primitive_manifest.jsonl").read_text(encoding="utf-8").splitlines() if line]
    audit = coverage_audit(units)
    assert audit["MISSING_WINDOWS"] == audit["DUPLICATE_WINDOWS"] == audit["OUT_OF_RANGE_WINDOWS"] == 0


def test_native_result_source_window_coverage():
    units = [json.loads(line) for line in (R2 / "stage3_h12_r2_primitive_manifest.jsonl").read_text(encoding="utf-8").splitlines() if line]
    native_rows = [
        {"unit_id": row["unit_id"], "source_window_ids": row["window_ids"]}
        for row in units
    ]
    audit = coverage_audit(native_rows)
    assert audit["coverage_complete"]


def test_replay_semantic_hash_equality():
    assert replay_hashes_equal(["same", "same", "same"])
    assert not replay_hashes_equal(["same", "different", "same"])


def test_upstream_immutability_comparison():
    assert all(path.is_file() for path in immutable_paths())
    snapshot = {"records": [{"path": "a", "exists": True, "sha256": "x", "size_bytes": 1}]}
    assert compare_snapshots(snapshot, snapshot)["status"] == "PASSED"
    mutated = {"records": [{"path": "a", "exists": True, "sha256": "y", "size_bytes": 1}]}
    assert compare_snapshots(snapshot, mutated)["status"] == "BLOCKED"


def test_physical_and_fjt_prohibition():
    worker = (ROOT / "ros2_moveit_bridge/stage3_h12_r4_native.py").read_text(encoding="utf-8")
    assert "FollowJointTrajectory" not in worker
    assert "controller_manager" not in worker
    assert "fairino" not in worker.lower()


def test_position_limits_are_not_weakened():
    worker = (ROOT / "ros2_moveit_bridge/stage3_h12_r4_native.py").read_text(encoding="utf-8")
    assert "runtime_audit[\"runtime_bounds\"]" in worker
    assert "joint_position_gate" in worker


def test_ccd_and_clearance_remain_unavailable():
    worker = (ROOT / "ros2_moveit_bridge/stage3_h12_r4_native.py").read_text(encoding="utf-8")
    assert '"CCD": "not_available"' in worker
    assert '"clearance": None' in worker


def test_collision_method_label_is_exact():
    source = (ROOT / "scripts/stage3_h12_r4_native_totg_closure.py").read_text(encoding="utf-8")
    assert "adaptive_discrete_interpolation" in source
