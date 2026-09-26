from __future__ import annotations

import inspect

from src.stage3_h12_trajectory_repair import audit_repair_inputs, repair_role, repair_window


def test_repair_input_audit_is_zero_leakage():
    audit = audit_repair_inputs()
    assert audit["REPAIR_LABEL_LEAKAGE_VIOLATIONS"] == 0
    assert audit["GROUND_TRUTH_USED_FOR_REPAIR"] == "NO"
    assert audit["GROUND_TRUTH_USED_FOR_FINAL_EVALUATION"] == "YES"
    assert audit["GROUND_TRUTH_COPY_DETECTED"] == "NO"
    assert all(value in {"INFERENCE_AVAILABLE", "EVALUATION_ONLY", "FORBIDDEN_FUTURE_LABEL"} for value in audit["inputs"].values())


def test_repair_api_has_no_target_label_argument():
    assert "target_positions" not in inspect.signature(repair_window).parameters
    assert "target_positions" not in inspect.signature(repair_role).parameters
    source = inspect.getsource(repair_role)
    assert "target_positions" not in source


def test_ground_truth_is_evaluation_only_in_contract():
    audit = audit_repair_inputs()
    assert audit["inputs"]["test_ground_truth"] == "EVALUATION_ONLY"
    assert audit["inputs"]["generalization_ground_truth"] == "EVALUATION_ONLY"
    assert audit["FUTURE_JOINT_LABEL_ACCESS"] == 0

