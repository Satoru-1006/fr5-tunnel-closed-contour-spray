from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from src.p2b3_c5b0_registration_spec import (
    BASE_LEFT,
    WORKPIECE_LOCAL,
    apply_registration,
    direct_delta,
    discrete_convergence_summary,
    engineering_reference_workpiece_transform,
    failure_predicate_taxonomy,
    se3_adjoint,
    se3_exp,
    transform_c4_scene_manifest,
    transform_normals,
    transform_points,
)


def test_base_left_identity_and_workpiece_local_identity_are_exact():
    entity = se3_exp([0.12, -0.03, 0.45, 0.2, -0.1, 0.08])
    workpiece = se3_exp([-0.2, 0.4, 0.1, -0.1, 0.03, 0.2])
    identity = np.eye(4)
    assert np.array_equal(apply_registration(entity, identity, BASE_LEFT), entity)
    assert np.array_equal(apply_registration(entity, identity, WORKPIECE_LOCAL, workpiece), entity)


def test_base_left_pure_translation_known_answer():
    entity = np.eye(4)
    delta = direct_delta([0.001, -0.002, 0.003], [0.0, 0.0, 0.0])
    result = apply_registration(entity, delta, BASE_LEFT)
    assert np.array_equal(result[:3, :3], np.eye(3))
    assert np.array_equal(result[:3, 3], np.asarray([0.001, -0.002, 0.003]))


def test_base_left_rotation_is_about_base_origin_and_has_lever_arm():
    point = np.asarray([[0.0, 0.6477447891692404, 0.0]])
    theta = math.radians(0.5)
    moved = transform_points(point, direct_delta([0, 0, 0], [theta, 0, 0]), BASE_LEFT)
    assert np.isclose(np.linalg.norm(moved[0] - point[0]), 0.6477447891692404 * 2 * math.sin(theta / 2), atol=1e-14)


def test_workpiece_local_rotation_is_about_selected_origin():
    workpiece = np.eye(4)
    workpiece[:3, 3] = [0.5, 0.0, 0.0]
    entity = np.eye(4)
    entity[:3, 3] = [0.6, 0.0, 0.0]
    delta = direct_delta([0, 0, 0], [0, 0, math.radians(90)])
    result = apply_registration(entity, delta, WORKPIECE_LOCAL, workpiece)
    assert np.allclose(result[:3, 3], [0.5, 0.1, 0.0], atol=1e-14)


def test_adjoint_consistency_for_left_and_body_coordinates():
    transform = se3_exp([0.2, -0.1, 0.3, 0.1, 0.2, -0.15])
    twist = np.asarray([0.03, 0.04, -0.02, 0.05, -0.01, 0.02])
    lhs = transform @ se3_exp(twist) @ np.linalg.inv(transform)
    rhs = se3_exp(se3_adjoint(transform) @ twist)
    assert np.allclose(lhs, rhs, atol=2.0e-12, rtol=2.0e-12)


def test_normals_remain_unit_and_scene_identity_is_preserved():
    root = Path(__file__).parents[1]
    import json

    manifest = json.loads((root / "outputs/p2b3_c4_authoritative_scene_manifest.json").read_text())
    reference = engineering_reference_workpiece_transform(manifest)
    delta = direct_delta([0.001, 0.0, 0.0], [0.0, 0.0, math.radians(0.1)])
    transformed = transform_c4_scene_manifest(manifest, delta, WORKPIECE_LOCAL, reference)
    before = manifest["collision_objects"]["ordered"]
    after = transformed["collision_objects"]["ordered"]
    assert len(before) == len(after) == 181
    assert [row["id"] for row in before] == [row["id"] for row in after]
    for old, new in zip(before, after):
        assert old["primitives"][0]["dimensions"] == new["primitives"][0]["dimensions"]
    normals = transform_normals([[1.0, 2.0, 3.0]], delta, WORKPIECE_LOCAL, reference)
    assert np.allclose(np.linalg.norm(normals, axis=1), [1.0], atol=1.0e-15)


def test_failure_taxonomy_fails_closed_for_unsupported_physical_thresholds():
    taxonomy = failure_predicate_taxonomy()
    assert taxonomy["F1_ROBOT_WORLD_COLLISION"]["sampling_label"] == "adaptive_discrete_interpolation"
    assert taxonomy["F2_SELF_COLLISION"]["sampling_label"] == "adaptive_discrete_interpolation"
    assert taxonomy["F3_LOW_CLEARANCE"]["predicate"] == "NOT_AVAILABLE"
    assert taxonomy["F6_STAND_OFF"]["predicate"] == "NOT_AVAILABLE"
    assert taxonomy["F7_COATING_QUALITY"]["predicate"] == "NOT_AVAILABLE"
    assert taxonomy["F4_PROCESS_POSITION"]["status"] == "ENGINEERING_DIAGNOSTIC_ONLY"
    assert taxonomy["F5_PROCESS_NORMAL"]["status"] == "ENGINEERING_DIAGNOSTIC_ONLY"


def test_discrete_resolution_convergence_bookkeeping_is_not_ccd():
    summary = discrete_convergence_summary([
        {"resolution": "0.5deg", "sample_count": 858, "min_robot_world_signed_distance_m": 0.08004969620976321, "pair": ["upperarm_link", "horseshoe_wall_026"], "segment": "85->86", "collision_samples": 0},
        {"resolution": "0.25deg", "sample_count": 1653, "min_robot_world_signed_distance_m": 0.08004804691553469, "pair": ["upperarm_link", "horseshoe_wall_026"], "segment": "85->86", "collision_samples": 0},
        {"resolution": "0.1deg", "sample_count": 3983, "min_robot_world_signed_distance_m": 0.08004767537031543, "pair": ["upperarm_link", "horseshoe_wall_026"], "segment": "85->86", "collision_samples": 0},
    ])
    assert summary["status"] == "PASS"
    assert summary["nearest_pair_stable"] and summary["critical_segment_stable"]
    assert summary["strict_continuous_collision_detection"] == "NOT_AVAILABLE"
    assert summary["max_abs_change_from_first_m"] < 3.0e-6
