from __future__ import annotations

import numpy as np

from scripts.stage3_h13_low_storage_unseen_generalization import (
    audit_native_exit,
    make_case_poses,
    read_pose_rows,
    read_seed_rows,
    resample_seed_rows,
    resample_pose_rows,
    make_target_rows,
    BASE_POSES,
    BASE_SEEDS,
    INPUT_HISTORY,
    normalize_features,
    normalized_local_trajectory_time,
    rollout_path,
)
from src.stage3_h11_dataset import Segment
from src.stage3_h13 import (
    build_case_matrix,
    H12_R7_COMPAT_SEGMENT_ID,
    h12_r7_compatibility_fields,
    improvement_percent,
    metric_summary,
    overlap_audit,
    validate_case_matrix,
)


def test_h13_case_matrix_is_deterministic_and_on_state_only():
    first = build_case_matrix(20)
    second = build_case_matrix(20)
    assert first == second
    assert validate_case_matrix(first) == []
    assert len({row["case_id"] for row in first}) == 20
    assert all(row["spray_state"] == "SPRAY_ON" for row in first)


def test_h13_matrix_covers_multiple_supported_dimensions():
    cases = build_case_matrix(20)
    assert len({tuple(row["parameters"]["start_position_offset_m"]) for row in cases}) > 1
    assert len({row["parameters"]["local_curvature_bulge_m"] for row in cases}) > 1
    assert len({row["parameters"]["legal_speed_m_s"] for row in cases}) > 1
    assert len({row["parameters"]["path_sampling_scale"] for row in cases}) > 1


def test_h13_leakage_audit_distinguishes_semantic_overlap_from_identity():
    h11 = {
        "TRAIN": [{"planned_joint_position": [1, 2], "planned_joint_velocity": [0, 0], "planned_joint_acceleration": [0, 0], "trajectory_time": 0.0, "spray_state": "SPRAY_ON", "trajectory_id": "old"}],
        "VALIDATION": [], "TEST": [], "GENERALIZATION": [],
    }
    same = h11["TRAIN"][0]
    audit = overlap_audit(h11, [__import__("src.stage3_h13", fromlist=["semantic_sample_hash"]).semantic_sample_hash(same)], ["new"], ["new_source"])
    assert audit["TRAIN_SAMPLE_OVERLAP"] == 1
    assert audit["EXACT_SEMANTIC_DUPLICATES"] == 1
    assert audit["TRAJECTORY_ID_OVERLAP"] == 0
    assert audit["SOURCE_RECORD_OVERLAP"] == 0
    assert audit["FUTURE_LABEL_LEAKAGE"] == 0


def test_h13_metric_and_improvement_are_finite_and_auditable():
    target = np.zeros((2, 3, 2), dtype=float)
    predicted = np.ones_like(target)
    metrics = metric_summary(predicted, target)
    assert metrics["joint_position_rmse_rad"] == 1.0
    assert improvement_percent(2.0, 1.0) == 50.0
    assert improvement_percent(0.0, 1.0) is None


def test_h13_rejects_off_state_cases():
    case = build_case_matrix(20)[0].copy()
    case["spray_state"] = "SPRAY_OFF"
    assert any("non_on_state" in item for item in validate_case_matrix([case]))


def test_h13_pose_resampling_preserves_quaternion_sign_continuity_and_normal_alignment():
    rows = make_case_poses(read_pose_rows(BASE_POSES), build_case_matrix(20)[0])
    quaternions = np.asarray([[row[key] for key in ("qx", "qy", "qz", "qw")] for row in rows])
    dots = np.sum(quaternions[1:] * quaternions[:-1], axis=1)
    x, y, z, w = quaternions.T
    tool_z = np.column_stack([2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)])
    normals = np.asarray([[row[key] for key in ("nx", "ny", "nz")] for row in rows])
    assert float(np.min(dots)) > 0.0
    assert float(np.rad2deg(np.arccos(np.clip(np.sum(tool_z * normals, axis=1), -1.0, 1.0))).max()) < 1.0e-4


def test_h13_seed_resampling_matches_pose_count_without_mutating_authority():
    pose_rows = resample_pose_rows(read_pose_rows(BASE_POSES), 177)
    seed_rows = resample_seed_rows(read_seed_rows(BASE_SEEDS), len(pose_rows))
    assert len(pose_rows) == len(seed_rows) == 177
    assert set(seed_rows[0]) == {"q1", "q2", "q3", "q4", "q5", "q6"}


def test_h13_adapter_preserves_frozen_h12_r7_repair_eligibility():
    fields = h12_r7_compatibility_fields(spray_state="SPRAY_ON", source_segment_id=0)
    assert fields["source_segment_id"] == 0
    assert fields["h12_r7_segment_id"] == H12_R7_COMPAT_SEGMENT_ID == 2
    assert fields["h12_r7_segment_order"] == 4
    assert fields["spray_state"] == "SPRAY_ON"


def test_h13_adapter_rejects_process_state_reinterpretation():
    import pytest
    with pytest.raises(ValueError, match="requires_spray_on"):
        h12_r7_compatibility_fields(spray_state="SPRAY_OFF", source_segment_id=0)


def test_h13_native_exit_audit_does_not_hide_teardown_crash():
    audit = audit_native_exit("[ERROR] process has died [pid 3, exit code -11]", "", 0, 20, 20)
    assert audit == {"NATIVE_MAIN_WORK_COMPLETED": "YES", "PROCESS_EXIT_CODE": -11, "CLEAN_EXIT": "NO", "TEARDOWN_CRASH": "YES"}


def test_h13_native_exit_audit_requires_complete_case_batch():
    audit = audit_native_exit("", "", 0, 19, 20)
    assert audit["NATIVE_MAIN_WORK_COMPLETED"] == "NO"
    assert audit["CLEAN_EXIT"] == "YES"


def test_h13_reprojection_reference_rows_preserve_passed_joint_path():
    poses = make_case_poses(read_pose_rows(BASE_POSES), build_case_matrix(20)[0])[:3]
    model_path = np.arange(18, dtype=float).reshape(3, 6)
    rows = make_target_rows(model_path, poses)
    assert [row["joint_values"] for row in rows] == model_path.tolist()


def test_h13_training_and_recursive_time_feature_use_same_whole_trajectory_domain():
    count = INPUT_HISTORY + 12
    times = np.cumsum(np.linspace(0.03, 0.07, count))
    segment = Segment(
        family_id="synthetic", trajectory_id="synthetic", segment_id=0, segment_order=0,
        spray_state="SPRAY_ON", primitive_id="0", times=times,
        positions=np.zeros((count, 6)), velocities=np.zeros((count, 6)),
        accelerations=np.zeros((count, 6)), sample_indices=np.arange(count), controlled_stop_count=0,
    )
    training_time = segment.local_time
    inference_time = normalized_local_trajectory_time(times)
    features = normalize_features(
        segment.positions, segment.velocities, segment.accelerations, times,
        {"channels": {}},
    )
    assert np.allclose(training_time, inference_time, atol=0.0, rtol=0.0)
    assert np.allclose(features[:, -1], training_time, atol=1.0e-7, rtol=0.0)
    assert float(np.min(inference_time)) >= 0.0
    assert float(np.max(inference_time)) <= 1.0


def test_h13_baseline_is_independent_of_model_rollout_history():
    import torch

    class ConstantModel:
        def __init__(self, value: float):
            self.value = value

        def __call__(self, inputs):
            return torch.full((inputs.shape[0], 8, 6), self.value, dtype=inputs.dtype)

    count = INPUT_HISTORY + 10
    times = np.linspace(0.0, 1.0, count)
    q = np.column_stack([np.linspace(0.01, 0.02, count) * (joint + 1) for joint in range(6)])
    dq = np.gradient(q, times, axis=0)
    ddq = np.gradient(dq, times, axis=0)
    stats = {"channels": {}}
    features = normalize_features(q, dq, ddq, times, stats)
    _, baseline_a = rollout_path(ConstantModel(0.0), features, q, times, stats)
    model_b, baseline_b = rollout_path(ConstantModel(0.01), features, q, times, stats)
    assert not np.allclose(model_b[INPUT_HISTORY:], q[INPUT_HISTORY:])
    assert np.allclose(baseline_a, baseline_b, atol=0.0, rtol=0.0)
