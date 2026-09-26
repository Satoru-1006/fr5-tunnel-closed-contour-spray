from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.stage3_h13_r3_frozen20_native_certification import (
    H13_R2_SUMMARY,
    HORIZON,
    INPUT_HISTORY,
    CASE_TARGET,
    build_arrays,
    full_residual_rollout,
    load_json,
    normalize_next_features,
    validate_checkpoint,
)
from src.stage3_h13 import build_case_matrix, canonical_hash


def test_r3_uses_the_original_h13_frozen_twenty_case_order() -> None:
    summary = load_json(H13_R2_SUMMARY)
    cases = build_case_matrix(CASE_TARGET)
    assert summary["case_matrix"]["matrix_sha256"] == canonical_hash([case["case_id"] for case in cases])
    assert len(cases) == 20
    assert [case["case_id"] for case in cases] == [
        "h13_unseen_00_00", "h13_unseen_00_01", "h13_unseen_00_02", "h13_unseen_00_03",
        "h13_unseen_01_00", "h13_unseen_01_01", "h13_unseen_01_02", "h13_unseen_01_03",
        "h13_unseen_02_00", "h13_unseen_02_01", "h13_unseen_02_02", "h13_unseen_02_03",
        "h13_unseen_03_00", "h13_unseen_03_01", "h13_unseen_03_02", "h13_unseen_03_03",
        "h13_unseen_04_00", "h13_unseen_04_01", "h13_unseen_04_02", "h13_unseen_04_03",
    ]


@pytest.mark.skipif(not (H13_R2_SUMMARY.is_file()), reason="authoritative H13-R2 output unavailable")
def test_r3_checkpoint_schema_is_inference_only() -> None:
    result = validate_checkpoint()
    assert result["status"] == "PASSED"
    assert result["model_type"] == "residual_causal_gru"
    assert result["input_history"] == INPUT_HISTORY == 16
    assert result["prediction_horizon"] == HORIZON == 8
    assert result["model_eval_called"] == "YES"
    assert result["parameter_mutation"] == "NO"


@pytest.mark.skipif(not pytest.importorskip("torch", reason="PyTorch unavailable"), reason="PyTorch unavailable")
def test_r3_rollout_does_not_read_future_labels() -> None:
    import torch
    from src.stage3_h11_r2_model import ResidualCausalGRUTrajectoryPredictor

    count = INPUT_HISTORY + 8
    t = np.linspace(0.0, 1.0, count)
    q = np.column_stack([np.linspace(0.01, 0.02, count) * (joint + 1) for joint in range(6)])
    dq = np.gradient(q, t, axis=0)
    ddq = np.gradient(dq, t, axis=0)
    stats = {"channels": {}}
    features = np.column_stack([q, dq, ddq, np.ones(count), t]).astype(np.float32)
    model = ResidualCausalGRUTrajectoryPredictor()
    model.eval()
    changed_q = q.copy()
    changed_q[INPUT_HISTORY:] += 50.0
    changed_features = features.copy()
    changed_features[INPUT_HISTORY:, :6] += 50.0
    left = full_residual_rollout(model, features, q, t, stats)
    right = full_residual_rollout(model, changed_features, changed_q, t, stats)
    assert np.array_equal(left, right)


def test_r3_next_feature_time_uses_whole_trajectory_anchors() -> None:
    q0 = np.zeros(6)
    features = normalize_next_features(q0, q0, q0, 5.0, 4.0, 3.0, 0.0, 10.0, {"channels": {}})
    assert features[-1] == pytest.approx(0.5)
