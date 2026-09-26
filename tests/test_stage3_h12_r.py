import inspect
from pathlib import Path

import numpy as np

from scripts.stage3_h12_r_certification import classify_failure
from src.stage3_h11_model import CausalGRUTrajectoryPredictor
from src.stage3_h12_r_residual import (
    BACKTRACK_FACTORS,
    RESIDUAL_REPAIR_METHOD,
    ResidualGRUTrajectoryPredictor,
    local_constraint_counts,
    repair_residual_role,
    repair_residual_window,
    residual_semantic_sha256,
)
from src.stage3_h12_trajectory_repair import RepairLimits, audit_repair_inputs


def causal_case(n=3):
    history = np.zeros((n, 16, 6), dtype=np.float64)
    history_times = np.tile(np.arange(16, dtype=np.float64), (n, 1))
    future_times = np.tile(np.arange(16, 24, dtype=np.float64), (n, 1))
    anchor = history[:, -1, :]
    cv = np.zeros((n, 8, 6), dtype=np.float64)
    residual = np.zeros_like(cv)
    return residual, cv, anchor, history_times, future_times


def test_zero_residual_is_exact_cv():
    residual, cv, anchor, ht, ft = causal_case()
    result = repair_residual_role(residual, cv, anchor, ht, ft)
    assert np.array_equal(result["positions"], cv)


def test_zero_residual_has_zero_neural_contribution():
    residual, cv, anchor, ht, ft = causal_case()
    result = repair_residual_role(residual, cv, anchor, ht, ft)
    assert result["neural_contribution_rate"] == 0.0


def test_zero_residual_has_no_fallback():
    residual, cv, anchor, ht, ft = causal_case()
    result = repair_residual_role(residual, cv, anchor, ht, ft)
    assert result["fallback_count"] == 0


def test_zero_residual_has_no_failure():
    residual, cv, anchor, ht, ft = causal_case()
    result = repair_residual_role(residual, cv, anchor, ht, ft)
    assert result["POSITION_REPAIR_FAILURES"] == 0


def test_nonzero_residual_is_added_before_repair():
    residual, cv, anchor, ht, ft = causal_case(1)
    residual[0, :, 0] = 0.01
    result = repair_residual_role(residual, cv, anchor, ht, ft)
    assert np.allclose(result["raw_positions"], cv + residual)


def test_backtracking_alpha_is_from_frozen_sequence():
    assert BACKTRACK_FACTORS[-1] == 0.0
    assert all(BACKTRACK_FACTORS[i] >= BACKTRACK_FACTORS[i + 1] for i in range(len(BACKTRACK_FACTORS) - 1))


def test_backtracking_is_deterministic():
    residual, cv, anchor, ht, ft = causal_case(2)
    residual[:, :, 0] = 0.02
    first = repair_residual_role(residual, cv, anchor, ht, ft)
    second = repair_residual_role(residual, cv, anchor, ht, ft)
    assert np.array_equal(first["positions"], second["positions"])
    assert np.array_equal(first["alpha"], second["alpha"])


def test_repair_api_has_no_target_label_argument():
    assert "target_positions" not in inspect.signature(repair_residual_role).parameters
    assert "target_positions" not in inspect.signature(repair_residual_window).parameters


def test_repair_method_is_explicit():
    assert RESIDUAL_REPAIR_METHOD == "causal_cv_residual_backtracking_v1"


def test_repair_uses_frozen_limits():
    residual, cv, anchor, ht, ft = causal_case(1)
    custom = RepairLimits(position_lower=np.full(6, -1.0), position_upper=np.full(6, 1.0), velocity_abs=np.full(6, 0.5), acceleration_abs=np.full(6, 0.2), jerk_abs=np.full(6, 8.0))
    counts = local_constraint_counts(cv, anchor, ht, ft, custom)
    assert counts["position"] == 0


def test_infeasible_residual_falls_back_to_cv():
    residual, cv, anchor, ht, ft = causal_case(1)
    residual[0, :, 0] = 1.0e8
    result = repair_residual_role(residual, cv, anchor, ht, ft)
    assert result["fallback_count"] == 1
    assert np.array_equal(result["positions"], cv)


def test_local_position_gate_detects_violation():
    residual, cv, anchor, ht, ft = causal_case(1)
    q = cv[:1].copy()
    q[0, 0, 0] = 10.0
    limits = RepairLimits()
    counts = local_constraint_counts(q, anchor, ht, ft, limits)
    assert counts["position"] > 0


def test_local_velocity_gate_detects_violation():
    residual, cv, anchor, ht, ft = causal_case(1)
    q = cv[:1].copy()
    q[0, 0, 0] = 1.0
    counts = local_constraint_counts(q, anchor, ht, ft, RepairLimits())
    assert counts["velocity"] > 0


def test_local_acceleration_gate_detects_violation():
    residual, cv, anchor, ht, ft = causal_case(1)
    q = cv[:1].copy()
    q[0, 0, 0] = 0.2
    q[0, 1, 0] = -0.2
    counts = local_constraint_counts(q, anchor, ht, ft, RepairLimits())
    assert counts["acceleration"] > 0


def test_local_jerk_gate_detects_violation():
    residual, cv, anchor, ht, ft = causal_case(1)
    q = cv[:1].copy()
    q[0, 0, 0] = 3.0
    q[0, 1, 0] = 0.0
    q[0, 2, 0] = 3.0
    counts = local_constraint_counts(q, anchor, ht, ft, RepairLimits())
    assert counts["jerk"] > 0


def test_residual_semantic_digest_is_stable():
    residual, cv, anchor, ht, ft = causal_case(2)
    payload = {"TRAIN": residual, "VALIDATION": residual, "TEST": residual, "GENERALIZATION": residual}
    assert residual_semantic_sha256(payload) == residual_semantic_sha256(payload)


def test_residual_model_keeps_h11_parameterization():
    direct = CausalGRUTrajectoryPredictor()
    residual = ResidualGRUTrajectoryPredictor()
    assert sum(x.numel() for x in direct.parameters()) == sum(x.numel() for x in residual.parameters())


def test_residual_model_is_unidirectional():
    model = ResidualGRUTrajectoryPredictor()
    assert model.gru.bidirectional is False


def test_no_leakage_audit_is_zero():
    audit = audit_repair_inputs()
    assert audit["REPAIR_LABEL_LEAKAGE_VIOLATIONS"] == 0


def test_native_validator_contract_is_shard_schema():
    source = Path("ros2_moveit_bridge/stage3_h12_r_native.py").read_text(encoding="utf-8")
    assert 'SCHEMA_VERSION = "stage3_h12_r_native_shard_v1"' in source
    assert 'NATIVE_VALIDATOR_VERSION = "stage3_h12_r_native_moveit_fcl_ruckig_v1"' in source


def test_failure_category_preserves_joint_gate():
    categories = classify_failure({"status": "BLOCKED", "execution_limits": {"velocity_violation_count": 2}})
    assert "JOINT_VELOCITY" in categories


def test_failure_category_preserves_collision_gate():
    categories = classify_failure({"status": "BLOCKED", "collision": {"self_collision_failure_count": 1, "environment_collision_failure_count": 0}})
    assert categories == ["SELF_COLLISION"]


def test_failure_category_preserves_spray_gate():
    categories = classify_failure({"status": "BLOCKED", "process": {"failed_spray_on_sample_count": 1, "first_failure": {"tcp_position_error_m": 0.01}}})
    assert "SPRAY_STATE_SEMANTICS" in categories
    assert "TCP_POSITION" in categories


def test_software_only_contract_is_explicit():
    assert audit_repair_inputs()["GROUND_TRUTH_USED_FOR_REPAIR"] == "NO"


def test_zero_residual_does_not_mutate_cv_array():
    residual, cv, anchor, ht, ft = causal_case(1)
    original = cv.copy()
    repair_residual_role(residual, cv, anchor, ht, ft)
    assert np.array_equal(cv, original)
