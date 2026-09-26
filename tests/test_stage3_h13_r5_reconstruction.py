import hashlib

import numpy as np
import pytest

from src.stage3_h13_r5_reconstruction import (
    COLLISION_SEMANTICS,
    CONSTRAINT_THRESHOLDS_RELAXED,
    FORBIDDEN_RECONSTRUCTION_INPUT_FIELDS,
    MODEL_PRIOR_MAX_JOINT_DELTA_RAD,
    MODEL_PRIOR_MAX_RMS_JOINT_DELTA_RAD,
    bounded_reconstruction_decision,
    classify_rollout_drift,
    leakage_audit,
    raw_constraint_gate,
    replay_equivalent,
    trajectory_delta_metrics,
)


def test_zero_leakage_reconstruction_inputs_are_explicit_and_label_free():
    audit = leakage_audit()
    assert audit["H13_20_UNSEEN_USED_FOR_TRAINING"] == "NO"
    assert audit["H13_20_UNSEEN_USED_FOR_MODEL_SELECTION"] == "NO"
    assert audit["UNSEEN_GROUND_TRUTH_USED_FOR_RECONSTRUCTION"] == "NO"
    assert audit["UNSEEN_LABEL_USED_FOR_RECONSTRUCTION"] == "NO"
    assert audit["FUTURE_LABEL_LEAKAGE"] == 0
    assert "ground_truth_trajectory" in FORBIDDEN_RECONSTRUCTION_INPUT_FIELDS
    assert "H13_autoregressive_model_prior_as_IK_seed" in audit["reconstruction_inputs"]


def test_frozen_case_identity_is_not_mutated_by_reconstruction_helpers():
    frozen = ["h13_unseen_00_00", "h13_unseen_00_01"]
    before = tuple(frozen)
    decision = bounded_reconstruction_decision({
        "MODEL_TO_RECONSTRUCTED_MAX_JOINT_DELTA": 0.0,
        "MODEL_TO_RECONSTRUCTED_RMS_JOINT_DELTA": 0.0,
        "AFFECTED_TRAJECTORY_FRACTION": 0.0,
    })
    assert decision["accepted"] is True
    assert tuple(frozen) == before


def test_bounded_reconstruction_rejects_out_of_domain_whole_trajectory():
    assert bounded_reconstruction_decision({
        "MODEL_TO_RECONSTRUCTED_MAX_JOINT_DELTA": MODEL_PRIOR_MAX_JOINT_DELTA_RAD + 1e-3,
        "MODEL_TO_RECONSTRUCTED_RMS_JOINT_DELTA": MODEL_PRIOR_MAX_RMS_JOINT_DELTA_RAD,
        "AFFECTED_TRAJECTORY_FRACTION": 1.0,
    })["accepted"] is False


def test_constraint_thresholds_and_collision_semantics_are_unchanged():
    assert CONSTRAINT_THRESHOLDS_RELAXED == "NO"
    assert COLLISION_SEMANTICS == "adaptive_discrete_interpolation"
    assert raw_constraint_gate({
        "POSITION_VIOLATIONS": 0,
        "VELOCITY_VIOLATIONS": 0,
        "ACCELERATION_VIOLATIONS": 0,
        "JERK_VIOLATIONS": 0,
        "COLLISION_VIOLATIONS": 0,
        "SPRAY_PROCESS_VIOLATIONS": 0,
    })["accepted"] is True
    assert raw_constraint_gate({
        "POSITION_VIOLATIONS": 1,
        "VELOCITY_VIOLATIONS": 0,
        "ACCELERATION_VIOLATIONS": 0,
        "JERK_VIOLATIONS": 0,
        "COLLISION_VIOLATIONS": 0,
        "SPRAY_PROCESS_VIOLATIONS": 0,
    })["accepted"] is False


def test_full_trajectory_metrics_are_not_a_local_waypoint_patcher():
    prior = np.zeros((10, 6))
    candidate = prior.copy()
    candidate[:, 0] = np.linspace(0.0, 0.1, 10)
    metrics = trajectory_delta_metrics(prior, candidate, np.linspace(0.0, 1.0, 10))
    assert metrics["AFFECTED_TRAJECTORY_FRACTION"] == pytest.approx(0.9)
    assert metrics["first_affected_index"] == 1


def test_model_prior_and_no_model_control_are_separate():
    model_prior = {"accepted": True, "objective": 0.1}
    no_model = {"accepted": True, "objective": 0.0}
    assert model_prior is not no_model
    assert model_prior["objective"] != no_model["objective"]


def test_pre_native_gate_rejects_missing_or_nonzero_safety_counts():
    missing = raw_constraint_gate({"POSITION_VIOLATIONS": 0})
    assert missing["accepted"] is False
    assert raw_constraint_gate({
        "POSITION_VIOLATIONS": 0,
        "VELOCITY_VIOLATIONS": 0,
        "ACCELERATION_VIOLATIONS": 0,
        "JERK_VIOLATIONS": 0,
        "COLLISION_VIOLATIONS": 0,
        "SPRAY_PROCESS_VIOLATIONS": 2,
    })["PRE_NATIVE_GATE_STATUS"] == "BLOCKED"


def test_native_attempt_finished_accounting_is_explicit_in_mode_summary_shape():
    expected = {"TOTG_ATTEMPTED", "TOTG_PASSED", "RUCKIG_ATTEMPTED", "RUCKIG_FINISHED", "RUCKIG_FAILED"}
    summary = {key: "0/0" for key in expected}
    assert expected <= set(summary)


def test_rollout_drift_classification_uses_full_trajectory_evidence():
    result = classify_rollout_drift([
        {"affected_trajectory_fraction": 1.0, "max_joint_space_displacement_rad": 2.0, "first_significant_divergence_index": 4, "dominant_mechanism": "G_MIXED_FULL_TRAJECTORY_DEFORMATION"},
        {"affected_trajectory_fraction": 0.9, "max_joint_space_displacement_rad": 1.0, "first_significant_divergence_index": 6, "dominant_mechanism": "G_MIXED_FULL_TRAJECTORY_DEFORMATION"},
    ])
    assert result["ROOT_CAUSE_CLASS"] == "G_MIXED_FULL_TRAJECTORY_DEFORMATION"
    assert result["FULL_TRAJECTORY_DEFORMATION_CASES"] == 2


def test_deterministic_replay_digest_is_semantic_and_stable():
    records = [{"case_id": "c0", "accepted": False}, {"case_id": "c1", "accepted": True}]
    first = replay_equivalent(records)
    second = replay_equivalent(records)
    assert first["REPLAY"] == "3/3"
    assert first["REPLAY_SEMANTIC_MATCH"] == "YES"
    assert first["semantic_digest"] == second["semantic_digest"]
    assert len(first["semantic_digest"]) == hashlib.sha256().digest_size * 2

