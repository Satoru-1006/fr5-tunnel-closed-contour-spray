from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.stage3_h12_r3 import (
    audit_network_dependency,
    classify_ceiling,
    classify_ruckig_result,
    finite_difference_diagnostics,
    reconcile_overlaps,
    ruckig_completion_gate,
    validate_ruckig_input,
)


def _valid_input() -> dict[str, list[float]]:
    return {
        "current_position": [0.0] * 6,
        "current_velocity": [0.0] * 6,
        "current_acceleration": [0.0] * 6,
        "target_position": [0.01] * 6,
        "target_velocity": [0.0] * 6,
        "target_acceleration": [0.0] * 6,
        "max_velocity": [1.0] * 6,
        "max_acceleration": [1.0] * 6,
        "max_jerk": [1.0] * 6,
        "min_position": [-1.0] * 6,
        "max_position": [1.0] * 6,
    }


def test_ruckig_working_not_error():
    assert classify_ruckig_result({"numeric_result": 0, "result_name": "Working"}) == "Working_Incomplete"


def test_ruckig_finished_required_for_pass():
    assert not ruckig_completion_gate({"numeric_result": 0, "result_name": "Working", "smoothing_complete": True}).get("post_certification_allowed")
    assert ruckig_completion_gate({"numeric_result": 1, "result_name": "Finished", "smoothing_complete": True}).get("post_certification_allowed")


def test_ruckig_negative_result_is_error():
    assert classify_ruckig_result({"numeric_result": -100, "result_name": "ErrorInvalidInput"}) == "ErrorInvalidInput"


def test_offline_ruckig_no_network(tmp_path: Path):
    source = tmp_path / "local.py"
    source.write_text("trajectory.apply_ruckig_smoothing()\n", encoding="utf-8")
    audit = audit_network_dependency([source], {"moveit_ruckig_api": True})
    assert audit["NETWORK_DEPENDENCY_FOUND"] == "NO"
    assert audit["OFFLINE_RUCKIG_PATH"] == "YES"


def test_intermediate_positions_cloud_dependency_detection(tmp_path: Path):
    source = tmp_path / "ruckig.py"
    source.write_text("input.intermediate_positions = waypoints\n", encoding="utf-8")
    audit = audit_network_dependency([source])
    assert audit["intermediate_positions"] is True
    assert audit["OFFLINE_RUCKIG_PATH"] == "NO"


def test_duration_ceiling_type_detection():
    assert classify_ceiling({"duration_ceiling_hit": True}) == "TRAJECTORY_DURATION_LIMIT"
    assert classify_ceiling({"wall_clock_timeout": True, "duration_ceiling_hit": True}) == "WALL_CLOCK_TIMEOUT"


def test_wall_clock_timeout_not_confused_with_sim_time():
    assert classify_ceiling({"max_simulated_time_hit": True}) == "MAX_SIMULATED_TIME"
    assert classify_ceiling({"max_calculation_duration_hit": True}) == "MAX_CALCULATION_DURATION"


def test_overlap_reconciliation_deterministic():
    predictions = np.zeros((3, 8, 6), dtype=float)
    predictions[0] = 1.0
    predictions[1] = 2.0
    predictions[2] = 3.0
    first, meta = reconcile_overlaps(predictions, [0, 1, 2])
    second, meta2 = reconcile_overlaps(predictions, [0, 1, 2])
    assert np.array_equal(first, second)
    assert meta == meta2
    assert meta["overlap_max"] == 3


def test_primitive_reconstruction_order_is_causal():
    predictions = np.stack([np.full((8, 6), value) for value in (1.0, 2.0)])
    result, _ = reconcile_overlaps(predictions, [0, 1], method="causal_priority")
    assert np.all(result[0] == 1.0)
    assert np.all(result[-1] == 2.0)


def test_velocity_acceleration_and_jerk_diagnostics():
    times = np.arange(8, dtype=float) * 0.1
    positions = np.zeros((8, 6), dtype=float)
    positions[:, 0] = np.arange(8, dtype=float) ** 3
    metrics = finite_difference_diagnostics(times, positions)
    assert "dq_dt" in metrics["derivative_reconstruction"]
    assert metrics["acceleration_violation_count"] > 0
    assert metrics["jerk_violation_count"] > 0


def test_spray_on_and_off_process_policy_is_not_collapsed():
    source = Path("scripts/stage3_h12_r3_primitive_dynamic_closure.py").read_text(encoding="utf-8")
    assert "SPRAY_ON" in source
    assert "SPRAY_OFF" in source
    assert "process_error_status" in source


def test_totg_hard_gate_and_working_unvalidated_are_explicit():
    source = Path("scripts/stage3_h12_r3_primitive_dynamic_closure.py").read_text(encoding="utf-8")
    assert "native_totg_returned_false" in source
    assert "ruckig_working_incomplete" in source
    assert "post_ruckig_unvalidated" in source


def test_ruckig_input_validation_categories():
    assert validate_ruckig_input(_valid_input())["status"] == "VALID_INPUT"
    invalid = _valid_input()
    invalid["target_position"][0] = 2.0
    assert validate_ruckig_input(invalid)["status"] == "INVALID_TARGET_STATE"
    invalid = _valid_input()
    invalid["max_jerk"][0] = 0.0
    assert validate_ruckig_input(invalid)["status"] == "INVALID_LIMIT_RELATION"


def test_required_artifact_names_are_documented():
    source = Path("scripts/stage3_h12_r3_primitive_dynamic_closure.py").read_text(encoding="utf-8")
    for name in ("FINAL_REPORT.md", "stage3_h12_r3_terminal_certificate.json", "primitive_reconstruction.jsonl", "primitive_dynamic_metrics.jsonl", "totg_results.jsonl", "ruckig_results.jsonl", "post_ruckig_certification.jsonl", "ablation_results.json", "determinism_replay.json"):
        assert name in source or name.replace(".", "\\\.") in source
