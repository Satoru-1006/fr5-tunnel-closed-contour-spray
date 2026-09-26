from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.stage3_h11_dataset import (
    EXPECTED_H10_SEMANTIC_SHA256,
    FEATURE_NAMES,
    INPUT_HISTORY,
    PREDICTION_HORIZON,
    H11Dataset,
    Segment,
    compute_normalization_stats,
    make_windows,
    semantic_hash,
    verify_h10_authority,
)
from src.stage3_h11_model import constant_velocity_prediction, hold_last_prediction, metric_payload


def fake_segment(family: str, order: int = 0, spray: str = "SPRAY_ON", count: int = 30) -> Segment:
    times = np.arange(count, dtype=np.float64) * 0.01
    q = np.column_stack([0.1 * times + 0.01 * joint for joint in range(6)])
    dq = np.full_like(q, 0.1)
    ddq = np.zeros_like(q)
    return Segment(family, f"{family}_trajectory", order, order, spray, str(order), times, q, dq, ddq, np.arange(count), 0)


def fake_dataset() -> H11Dataset:
    segments = [fake_segment("train"), fake_segment("test", 1, "SPRAY_OFF")]
    return H11Dataset(Path("."), {"train": "TRAIN", "test": "TEST"}, segments, 60, 2, 2)


def test_h10_hash_constant_is_frozen() -> None:
    assert EXPECTED_H10_SEMANTIC_SHA256 == "cb282a06d73c322208f27bf7c0b019831a24f524ac0d72263929f22303f0801b"


def test_window_generation_stays_inside_segments() -> None:
    windows, audit, counts = make_windows(fake_dataset())
    assert windows
    assert audit["CROSS_FAMILY_WINDOWS"] == 0
    assert all(not item["cross_family"] and not item["cross_segment"] for item in windows)
    assert counts["TRAIN"]["families"] == 1


def test_window_hash_is_order_sensitive_to_membership_but_not_mapping_order() -> None:
    windows, _, _ = make_windows(fake_dataset())
    assert semantic_hash(windows) == semantic_hash(json.loads(json.dumps(windows)))
    assert semantic_hash(windows) != semantic_hash(windows[:-1])


def test_normalization_uses_train_only() -> None:
    stats = compute_normalization_stats(fake_dataset())
    assert stats["source_split"] == "TRAIN"
    assert stats["source_family_count"] == 1
    assert all(stats["channels"][name]["source_split"] == "TRAIN" for name in stats["channels"])
    assert len(FEATURE_NAMES) == 20


def test_required_history_and_horizon_are_causal() -> None:
    assert INPUT_HISTORY == 16
    assert PREDICTION_HORIZON == 8


def test_naive_baselines_are_deterministic_and_metric_schema_is_auditable() -> None:
    windows, _, _ = make_windows(fake_dataset())
    dataset = fake_dataset()
    from scripts.stage3_h11_train_baseline import materialize_windows

    stats = compute_normalization_stats(dataset)
    arrays = materialize_windows(dataset, windows, stats, "TRAIN")
    hold = hold_last_prediction(arrays)
    constant = constant_velocity_prediction(arrays)
    assert np.array_equal(hold, hold_last_prediction(arrays))
    assert constant.shape == (len(arrays.inputs), PREDICTION_HORIZON, 6)
    metrics = metric_payload(hold, arrays.target_positions)
    assert set(("joint_position_mae_rad", "joint_position_rmse_rad", "mae_per_joint_rad", "rmse_per_joint_rad", "p50_absolute_error_rad", "p95_absolute_error_rad", "p99_absolute_error_rad")) <= set(metrics)


def test_authoritative_h10_directory_passes_when_present() -> None:
    root = Path(__file__).resolve().parents[1]
    h10 = root / "outputs/stage3_h10_multi_trajectory_dataset_20260811T081123Z"
    result = verify_h10_authority(root, h10)
    assert result["status"] == "PASSED"

