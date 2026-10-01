from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ros2_moveit_bridge"))

from p2b3_c4_scene_contract import (  # noqa: E402
    C1SceneContract,
    aggregate_reported_distances,
    assess_zero_transform_gate,
    build_scene_manifest,
    canonical_json_bytes,
    canonical_quaternion_xyzw,
    compare_scene_manifests,
    compare_c3_to_c1_parameters,
    extract_c1_scene_contract,
    extract_legacy_c3_scene_contract,
    manifest_sha256,
    validate_phase_d_identity_gate,
    verify_frozen_c1_source_blobs,
)
from p2b3_c3_native_smoke import summarize_collision_response  # noqa: E402


def contract() -> C1SceneContract:
    return C1SceneContract(
        project="FAIRINO_FR5",
        source_stage="P2-B3-C1",
        source_execution_commit="c" * 40,
        nominal_trajectory_sha256="a" * 64,
        waypoint_count=181,
        expected_sample_count=858,
        expected_collision_count=0,
        frame_id="base_link",
        parameters={
            "stand_off_m": 0.260,
            "tcp_points_to_wall": True,
            "wall_thickness_m": 0.04,
            "y_thickness_m": 1.10,
            "segment_stride": 1,
            "open_path": True,
            "include_bottom_closure": False,
            "include_tunnel_floor": True,
            "tunnel_floor_z_m": -0.20,
            "collision_check_stride": 1,
            "collision_method": "adaptive_discrete_interpolation",
        },
        sampling={
            "policy": "adaptive_discrete_interpolation",
            "maximum_joint_interpolation_step_deg": 0.5,
            "collision_check_stride": 1,
            "waypoint_count": 181,
        },
    )


def scene_object(object_id: str = "wall_000", *, frame: str = "base_link", dimensions=None, pose=None):
    return {
        "id": object_id,
        "frame_id": frame,
        "primitives": [{
            "type": "BOX",
            "dimensions": dimensions or [0.2, 1.1, 0.04],
            "pose": pose or {"position_xyz": [0.1, 0.2, 0.3], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]},
        }],
    }


def manifest(objects=None):
    return build_scene_manifest(
        contract(),
        objects or [scene_object()],
        acm={"status": "AVAILABLE", "sha256": "a" * 64},
        padding={"status": "AVAILABLE", "sha256": "b" * 64},
    )


def test_authoritative_scene_contract_extracted_from_frozen_c1_sources():
    runner = '''
exports = {
    "SAMPLES_PER_LOOP": "181", "OPEN_PATH": "true", "TCP_POINTS_TO_WALL": "true",
    "STAND_OFF": "0.260", "VALIDATE_COLLISION": "true", "COLLISION_CHECK_STRIDE": "1",
    "COLLISION_SEGMENT_STRIDE": "1", "COLLISION_INTERPOLATION_STEP_DEG": "0.5",
    "INCLUDE_TUNNEL_FLOOR_COLLISION": "true", "TUNNEL_FLOOR_Z": "-0.20",
    "TUNNEL_WALL_THICKNESS": "0.04", "TUNNEL_Y_THICKNESS": "1.10",
    "INCLUDE_BOTTOM_CLOSURE_COLLISION": "false",
}
'''
    planner = 'node.declare_parameter("base_frame", "base_link")\n'
    result = {
        "PROJECT": "FAIRINO_FR5",
        "STAGE": "P2-B3-C1",
        "C1_EXECUTION_CODE_COMMIT": "c" * 40,
        "C1_POST_RUCKIG_NOMINAL": {"sha256": "a" * 64},
        "c1_measurement": {
            "collision_method": "adaptive_discrete_interpolation",
            "summary_metrics": {"collision_checked_state_count": "858.0", "collision_count": "0.0"},
        },
    }
    extracted = extract_c1_scene_contract(result, runner, planner)
    assert extracted.parameters["stand_off_m"] == 0.260
    assert extracted.parameters["tcp_points_to_wall"] is True
    assert extracted.parameters["wall_thickness_m"] == 0.04
    assert extracted.parameters["y_thickness_m"] == 1.10
    assert extracted.parameters["segment_stride"] == 1
    assert extracted.expected_sample_count == 858
    assert extracted.expected_collision_count == 0


def test_manifest_serialization_and_hash_are_deterministic():
    first = build_scene_manifest(contract(), [scene_object("z"), scene_object("a")])
    second = build_scene_manifest(contract(), [scene_object("a"), scene_object("z")])
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert manifest_sha256(first) == manifest_sha256(second)


def test_identity_scene_is_equivalent():
    scene = manifest()
    assert compare_scene_manifests(scene, copy.deepcopy(scene))["status"] == "PASS"


def test_quaternion_equivalent_signs_canonicalize_identically():
    assert canonical_quaternion_xyzw([0, 0, 0.5, 0.5]) == canonical_quaternion_xyzw([0, 0, -0.5, -0.5])
    expected = manifest([scene_object(pose={"position_xyz": [0, 0, 0], "orientation_xyzw": [0, 0, 0.5, 0.5]})])
    observed = manifest([scene_object(pose={"position_xyz": [0, 0, 0], "orientation_xyzw": [0, 0, -0.5, -0.5]})])
    assert compare_scene_manifests(expected, observed)["status"] == "PASS"


def test_parameter_drift_is_reported():
    expected = manifest()
    observed = copy.deepcopy(expected)
    observed["effective_parameters"]["stand_off_m"] = 0.18
    result = compare_scene_manifests(expected, observed)
    assert result["status"] == "FAIL"
    assert result["differences"][0]["path"] == "effective_parameters.stand_off_m"


def test_missing_collision_object_is_reported():
    expected = manifest([scene_object("wall_000"), scene_object("wall_001")])
    observed = manifest([scene_object("wall_000")])
    assert "collision_objects.wall_001" in [d["path"] for d in compare_scene_manifests(expected, observed)["differences"]]


def test_extra_collision_object_is_reported():
    expected = manifest([scene_object("wall_000")])
    observed = manifest([scene_object("wall_000"), scene_object("wall_001")])
    assert "collision_objects.wall_001" in [d["path"] for d in compare_scene_manifests(expected, observed)["differences"]]


def test_dimensions_mismatch_is_reported():
    expected = manifest()
    observed = manifest([scene_object(dimensions=[0.2, 0.08, 0.025])])
    assert any(d["path"].endswith(".dimensions") for d in compare_scene_manifests(expected, observed)["differences"])


def test_pose_mismatch_is_reported():
    expected = manifest()
    observed = manifest([scene_object(pose={"position_xyz": [0.11, 0.2, 0.3], "orientation_xyzw": [0, 0, 0, 1]})])
    assert any(d["path"].endswith("pose.position_xyz") for d in compare_scene_manifests(expected, observed)["differences"])


def test_frame_mismatch_is_reported():
    expected = manifest()
    observed = manifest([scene_object(frame="world")])
    assert any(d["path"].endswith("frame_id") for d in compare_scene_manifests(expected, observed)["differences"])


def test_acm_and_padding_mismatches_are_reported():
    expected = manifest()
    observed = copy.deepcopy(expected)
    observed["acm"]["sha256"] = "d" * 64
    observed["robot_padding_scale"]["sha256"] = "e" * 64
    result = compare_scene_manifests(expected, observed)
    assert {d["path"] for d in result["differences"]} == {"acm", "robot_padding_scale"}


def test_runtime_api_surface_mismatch_is_reported():
    expected = manifest()
    expected["runtime_api_surface"] = {"planning_scene_type": "moveit.core.planning_scene.PlanningScene"}
    observed = copy.deepcopy(expected)
    observed["runtime_api_surface"]["planning_scene_type"] = "different.binding.PlanningScene"
    result = compare_scene_manifests(expected, observed)
    assert [item["path"] for item in result["differences"]] == ["runtime_api_surface"]


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        ([], "NOT_RUN_OR_INCOMPLETE"),
        ([{"sample_count": 10, "full_scene_collision_sample_count": 0, "self_collision_sample_count": 0}], "NO_COLLISIONS_OBSERVED"),
        ([{"sample_count": 10, "full_scene_collision_sample_count": 10, "self_collision_sample_count": 10}], "SATURATED_ALL_CASES"),
        ([{"sample_count": 10, "full_scene_collision_sample_count": 2, "self_collision_sample_count": 0}], "UNIFORM_PARTIAL_COLLISIONS"),
        ([{"sample_count": 10, "full_scene_collision_sample_count": 1, "self_collision_sample_count": 0}, {"sample_count": 10, "full_scene_collision_sample_count": 2, "self_collision_sample_count": 1}], "COLLISION_COUNTS_VARY_BY_CASE"),
    ],
)
def test_collision_response_summary_classifies_known_counts(rows, expected):
    assert summarize_collision_response(rows)["status"] == expected


def test_zero_registration_reproduction_gate_accepts_c1_baseline():
    c = contract()
    scene = manifest()
    result = assess_zero_transform_gate(
        c,
        trajectory_sha256=c.nominal_trajectory_sha256,
        waypoint_count=181,
        sampled_state_count=858,
        collision_count=0,
        fk_status="PASS",
        q_unchanged=True,
        timestamps_unchanged=True,
        target_pose_max_change=0.0,
        normal_max_change=0.0,
        scene_comparison=compare_scene_manifests(scene, scene),
    )
    assert result["status"] == "PASS"


def test_moveit_minus_one_distance_sentinel_is_rejected():
    classified = aggregate_reported_distances([-1.0, 0.2])
    assert classified["sentinel_count"] == 1
    assert classified["minimum_reported_full_scene_distance_m"] is None
    assert classified["status"] == "PARTIAL_NOT_AVAILABLE"


def test_missing_distance_diagnostic_fails_closed():
    result = aggregate_reported_distances([None, -1.0, float("nan")])
    assert result["status"] == "NOT_AVAILABLE"
    assert result["minimum_robot_world_distance_m"] is None
    assert result["minimum_reported_full_scene_distance_m"] is None


def test_foreign_project_guard_rejects_non_fr5_result():
    foreign = {
        "PROJECT": "ESTUN_IER8",
        "STAGE": "P2-B3-C1",
        "C1_EXECUTION_CODE_COMMIT": "c" * 40,
        "C1_POST_RUCKIG_NOMINAL": {"sha256": "a" * 64},
    }
    with pytest.raises(ValueError, match="foreign_or_non_C1_project_result_rejected"):
        extract_c1_scene_contract(foreign, "exports = {}", 'node.declare_parameter("base_frame", "base_link")')


def test_recorded_c3_execution_source_confirms_scene_parameter_drift():
    c1_result = json.loads((ROOT / "outputs" / "p2b3_c1_result.json").read_text(encoding="utf-8"))
    c3_result = json.loads((ROOT / "outputs" / "p2b3_c3_result.json").read_text(encoding="utf-8"))
    c1 = extract_c1_scene_contract(
        c1_result,
        (ROOT / "scripts" / "run_p2b3_c1_r0.py").read_text(encoding="utf-8"),
        (ROOT / "ros2_moveit_bridge" / "plan_closed_contour_moveit.py").read_text(encoding="utf-8"),
    )
    source_blobs = verify_frozen_c1_source_blobs(ROOT, c1.source_execution_commit)
    legacy = extract_legacy_c3_scene_contract(ROOT, c3_result)
    drift = compare_c3_to_c1_parameters(legacy, c1)
    assert source_blobs["scripts/run_p2b3_c1_r0.py"]
    assert legacy["stand_off"] == 0.18
    assert legacy["tcp_points_to_wall"] is False
    assert legacy["wall_thickness"] == 0.025
    assert legacy["y_thickness"] == 0.08
    assert legacy["segment_stride"] == 4
    assert {item["parameter"] for item in drift["differences"]} == {
        "stand_off_m", "tcp_points_to_wall", "wall_thickness_m", "y_thickness_m", "segment_stride",
    }


def test_phase_d_requires_a_passed_matching_phase_c_gate():
    record = {
        "project": "FAIRINO_FR5",
        "stage": "P2-B3-C4",
        "scene_manifest_sha256": "f" * 64,
        "zero_transform_equivalence": {"status": "PASS"},
        "baseline": {"sample_count": 858},
    }
    assert validate_phase_d_identity_gate(record, "f" * 64)["sample_count"] == 858
    with pytest.raises(ValueError, match="phase_d_rejected_without_zero_transform_pass"):
        validate_phase_d_identity_gate({**record, "zero_transform_equivalence": {"status": "FAIL"}}, "f" * 64)
    with pytest.raises(ValueError, match="phase_d_scene_manifest_differs_from_phase_c"):
        validate_phase_d_identity_gate(record, "e" * 64)
