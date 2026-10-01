from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from src.physical_uncertainty import (
    BASE_FRAME,
    COMPOSITION_CONVENTION,
    SemanticClass,
    UncertaintySource,
    apply_registration_to_trajectory,
    classify_legacy_c2_axis,
    default_uncertainty_semantics,
    foreign_project_reference_count,
    inverse_transform,
    normalization_result,
    registration_transform,
    rotation_matrix_from_vector,
    six_dof_directional_smoke,
    transform_collision_object_message,
    transform_pose_array_xyzw,
    transform_surface_normals,
    validate_contract_record,
    validate_transform,
)


def test_contract_has_required_machine_readable_ontology() -> None:
    rows = default_uncertainty_semantics()
    assert {row["semantic_class"] for row in rows} == {value.value for value in SemanticClass}
    for row in rows:
        validate_contract_record(row)
        assert row["normalization_bound"] is None
        assert row["normalization_status"] == "NOT_AVAILABLE_PHYSICAL_BOUNDS_MISSING"


def test_contract_rejects_unknown_source_enum() -> None:
    row = default_uncertainty_semantics()[0]
    row["physical_source"] = "UNLISTED_SOURCE"
    with pytest.raises(ValueError):
        validate_contract_record(row)


def test_assumed_and_literature_bounds_cannot_be_promoted() -> None:
    for source, status in (
        (UncertaintySource.ASSUMED, "NOT_AVAILABLE_UNSUPPORTED_BOUND_SOURCE"),
        (UncertaintySource.LITERATURE, "NOT_AVAILABLE_UNSUPPORTED_BOUND_SOURCE"),
        (UncertaintySource.UNKNOWN, "NOT_AVAILABLE_UNSUPPORTED_BOUND_SOURCE"),
    ):
        result = normalization_result(0.5, 0.1, source)
        assert result == {"normalized_margin": None, "status": status, "bound_source": source.value}


def test_measured_bound_normalization_is_explicit() -> None:
    assert normalization_result(0.5, 0.1, UncertaintySource.MEASURED) == {
        "normalized_margin": 5.0,
        "status": "AVAILABLE_WITH_DECLARED_BOUND",
        "bound_source": "MEASURED",
    }


def test_unknown_bound_fails_closed() -> None:
    result = normalization_result(0.5, None, UncertaintySource.UNKNOWN)
    assert result["normalized_margin"] is None
    assert result["status"] == "NOT_AVAILABLE_PHYSICAL_BOUNDS_MISSING"


def test_unknown_bound_source_fails_closed_even_with_numeric_value() -> None:
    result = normalization_result(0.5, 0.1, UncertaintySource.UNKNOWN)
    assert result["normalized_margin"] is None
    assert result["status"] == "NOT_AVAILABLE_UNSUPPORTED_BOUND_SOURCE"


def test_legacy_tcp_axes_are_target_pose_semantics_and_raw_values_stay_separate() -> None:
    historical = {"axis_id": "tcp_tcp_translation:x:positive", "raw_margin": 0.00399609}
    before = deepcopy(historical)
    derived = classify_legacy_c2_axis(historical["axis_id"])
    assert derived["semantic_class"] == SemanticClass.TARGET_POSE_GEOMETRIC_PERTURBATION.value
    assert derived["physical_source"] == "ASSUMED"
    assert derived["raw_historical_value_status"] == "PRESERVED_UNCHANGED"
    assert historical == before


def test_legacy_joint_axis_is_deterministic_state_offset_semantics() -> None:
    derived = classify_legacy_c2_axis("joint:j3:positive")
    assert derived["semantic_class"] == SemanticClass.JOINT_STATE_ERROR.value
    assert "not a measured tracking distribution" in derived["physical_interpretation"]


def test_zero_registration_is_exact_identity_for_transform_and_pose() -> None:
    delta = registration_transform()
    poses = np.asarray([[1.0, -2.0, 0.5, 0.0, 0.0, 0.0, 1.0]])
    assert np.array_equal(delta, np.eye(4))
    assert np.array_equal(transform_pose_array_xyzw(poses, delta), poses)


def test_translation_known_answer_and_left_composition() -> None:
    delta = registration_transform([0.25, -0.5, 1.0], [0.0, 0.0, 0.0])
    point = np.asarray([1.0, 2.0, 3.0, 1.0])
    transformed = delta @ point
    assert np.array_equal(transformed[:3], [1.25, 1.5, 4.0])
    assert COMPOSITION_CONVENTION == "T_base_entity_actual = DeltaT_base_left @ T_base_entity_nominal"


@pytest.mark.parametrize(
    ("axis", "source", "expected"),
    [
        ([np.pi / 2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
        ([0.0, np.pi / 2.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]),
        ([0.0, 0.0, np.pi / 2.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
    ],
)
def test_ninety_degree_rotation_known_answer(
    axis: list[float], source: list[float], expected: list[float]
) -> None:
    delta = registration_transform([0.0, 0.0, 0.0], axis)
    actual = delta[:3, :3] @ np.asarray(source)
    assert np.allclose(actual, expected, atol=1e-12)


@pytest.mark.parametrize("translation", [[0.7, 0.0, 0.0], [0.0, -0.2, 0.0], [0.0, 0.0, 1.1]])
def test_translation_keeps_normals_exactly_unchanged(translation: list[float]) -> None:
    normals = np.asarray([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
    delta = registration_transform(translation, [0.0, 0.0, 0.0])
    assert np.array_equal(transform_surface_normals(normals, delta), normals)


def test_rotation_propagates_surface_normal_direction() -> None:
    normals = np.asarray([[1.0, 0.0, 0.0]])
    delta = registration_transform(rotation_vector_base_rad=[0.0, 0.0, np.pi / 2.0])
    assert np.allclose(transform_surface_normals(normals, delta), [[0.0, 1.0, 0.0]], atol=1e-12)


def test_inverse_transform_recovers_pose_and_normal() -> None:
    delta = registration_transform([0.01, -0.02, 0.03], [0.2, -0.1, 0.3])
    pose = np.asarray([[0.3, 0.2, -0.1, 0.0, 0.0, 0.0, 1.0]])
    normal = np.asarray([[0.0, 0.0, 1.0]])
    moved_pose = transform_pose_array_xyzw(pose, delta)
    moved_normal = transform_surface_normals(normal, delta)
    recovered_pose = transform_pose_array_xyzw(moved_pose, inverse_transform(delta))
    recovered_normal = transform_surface_normals(moved_normal, inverse_transform(delta))
    assert np.allclose(recovered_pose[:, :3], pose[:, :3], atol=1e-12)
    assert np.allclose(recovered_pose[:, 3:7], pose[:, 3:7], atol=1e-12)
    assert np.allclose(recovered_normal, normal, atol=1e-12)


def test_rotation_matrix_is_orthonormal_and_quaternion_is_unit_length() -> None:
    rotation = rotation_matrix_from_vector([0.3, -0.4, 0.2])
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(rotation), 1.0, atol=1e-12)
    delta = registration_transform(rotation_vector_base_rad=[0.3, -0.4, 0.2])
    pose = transform_pose_array_xyzw([[0, 0, 0, 0, 0, 0, 1]], delta)
    assert np.isclose(np.linalg.norm(pose[0, 3:7]), 1.0, atol=1e-12)


def test_registration_keeps_joint_states_and_timestamps_bitwise_unchanged() -> None:
    q = np.arange(18, dtype=np.float64).reshape(3, 6) / 10.0
    times = np.asarray([0.0, 0.2, 0.5])
    poses = np.asarray([[0.1, 0.2, 0.3, 0, 0, 0, 1]] * 3, dtype=np.float64)
    normals = np.asarray([[0.0, 0.0, 1.0]] * 3)
    result = apply_registration_to_trajectory(poses, normals, q, times, registration_transform([0.1, 0, 0]))
    assert np.array_equal(result["joint_states"], q)
    assert np.array_equal(result["timestamps_s"], times)
    assert not np.array_equal(result["tcp_poses_xyz_quat_xyzw"], poses)


def test_collision_object_pose_moves_but_local_geometry_is_preserved() -> None:
    pose = SimpleNamespace(
        position=SimpleNamespace(x=1.0, y=2.0, z=3.0),
        orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    collision_object = SimpleNamespace(
        header=SimpleNamespace(frame_id=BASE_FRAME),
        id="workpiece_wall_0",
        primitive_poses=[pose],
        primitives=[SimpleNamespace(dimensions=[0.1, 0.2, 0.3])],
        mesh_poses=[],
        plane_poses=[],
        subframe_poses=[],
    )
    moved = transform_collision_object_message(collision_object, registration_transform([0.01, 0, 0]))
    assert np.allclose([moved.primitive_poses[0].position.x, moved.primitive_poses[0].position.y, moved.primitive_poses[0].position.z], [1.01, 2.0, 3.0])
    assert moved.primitives[0].dimensions == [0.1, 0.2, 0.3]
    assert collision_object.primitive_poses[0].position.x == 1.0


def test_collision_object_frame_mismatch_fails_closed() -> None:
    pose = SimpleNamespace(position=SimpleNamespace(x=0.0, y=0.0, z=0.0), orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0))
    collision_object = SimpleNamespace(header=SimpleNamespace(frame_id="other_frame"), primitive_poses=[pose])
    with pytest.raises(ValueError, match="collision_object_frame_mismatch"):
        transform_collision_object_message(collision_object, np.eye(4))


@pytest.mark.parametrize("bad", [[np.nan, 0, 0], [0, np.inf, 0], [0, 0, -np.inf]])
def test_nonfinite_registration_values_fail_closed(bad: list[float]) -> None:
    with pytest.raises(ValueError):
        registration_transform(bad, [0.0, 0.0, 0.0])


def test_nonrigid_transform_fails_closed() -> None:
    transform = np.eye(4)
    transform[0, 0] = 2.0
    with pytest.raises(ValueError, match="not_SO3"):
        validate_transform(transform)


def test_unknown_normalization_source_is_rejected() -> None:
    with pytest.raises(ValueError):
        normalization_result(0.2, 0.1, "UNLISTED_SOURCE")


def test_twelve_directional_cases_are_signed_and_unclipped() -> None:
    rows = six_dof_directional_smoke()
    assert len(rows) == 12
    assert len({row["case_id"] for row in rows}) == 12
    for row in rows:
        delta = registration_transform(row["translation_base_m"], row["rotation_vector_base_rad"])
        assert not np.array_equal(delta, np.eye(4))
        assert row["amplitude"] > 0.0
        assert "not an FR5 physical uncertainty bound" in row["interpretation"]


def test_foreign_project_reference_guard() -> None:
    forbidden = ["".join(("ES", "TUN")), "".join(("i", "ER8")), "".join(("ier", "8")), "".join(("720", "-MI"))]
    assert foreign_project_reference_count(["clean FR5 source tree"]) == 0
    assert foreign_project_reference_count(forbidden) == len(forbidden)
