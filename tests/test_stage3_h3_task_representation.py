from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from src.stage3_h2_geometry import GeometryConfig, GeometryValidationError, canonical_hash, canonicalize_geometry, identity_matrix, load_geometry, sha256_file
from src.stage3_h3_task_representation import (
    H3Config,
    H3ValidationError,
    build_surface_task_samples,
    construct_target_pose,
    evaluate_coverage,
    geometry_topology_diagnostics,
    h2_geometry_config_from_record,
    load_and_verify_h2_mesh,
    quaternion_rotate_vector,
    semantic_output_hash,
)


ROOT = Path(__file__).resolve().parents[1]
H2_ROOT = sorted(ROOT.glob("outputs/stage3_h2_geometry_baseline_*/stage3_h2_geometry_manifest.json"))[-1].parent
H2_MANIFEST_PATH = H2_ROOT / "stage3_h2_geometry_manifest.json"
H2_MANIFEST_HASH = sha256_file(H2_MANIFEST_PATH)
H2_MANIFEST = json.loads(H2_MANIFEST_PATH.read_text(encoding="utf-8"))
H2_RECORDS = {record["geometry_id"]: record for record in H2_MANIFEST["records"]}
CONFIG = H3Config()


def mesh_and_tasks(geometry_id: str = "fixture_planar_patch"):
    mesh, _processing, _derived = load_and_verify_h2_mesh(str(ROOT), H2_RECORDS[geometry_id])
    tasks, metadata = build_surface_task_samples(mesh, H2_RECORDS[geometry_id], h2_manifest_hash=H2_MANIFEST_HASH, config=CONFIG)
    return mesh, tasks, metadata


def test_planar_surface_generates_provenance_linked_tasks() -> None:
    mesh, tasks, metadata = mesh_and_tasks("fixture_planar_patch")
    assert len(mesh.triangles) == 2
    assert len(tasks) == CONFIG.sample_count
    assert metadata["geometry_hash"] == H2_RECORDS["fixture_planar_patch"]["geometry_hash"]
    assert all(task["source_triangle_id"] in (0, 1) for task in tasks)


def test_curved_cylinder_and_tunnel_surfaces_generate_tasks() -> None:
    cylinder_mesh, cylinder_tasks, _ = mesh_and_tasks("fixture_curved_cylinder_patch")
    tunnel_mesh, tunnel_tasks, _ = mesh_and_tasks("fixture_tunnel_like_patch")
    assert len(cylinder_mesh.triangles) > 1 and len(tunnel_mesh.triangles) > 1
    assert len(cylinder_tasks) == len(tunnel_tasks) == CONFIG.sample_count
    assert len({tuple(task["surface_normal_unit"]) for task in cylinder_tasks}) > 1
    assert len({tuple(task["surface_normal_unit"]) for task in tunnel_tasks}) > 1


def test_triangle_ordering_is_not_silently_aliased() -> None:
    mesh, _tasks, _ = mesh_and_tasks()
    reordered = copy.deepcopy(mesh)
    reordered.triangles = list(reversed(reordered.triangles))
    assert canonical_hash(mesh.triangles) != canonical_hash(reordered.triangles)
    assert H2_RECORDS["fixture_planar_patch"]["geometry_hash"] == H2_RECORDS["fixture_planar_patch"]["geometry_hash"]


def test_nan_and_inf_fail_closed() -> None:
    with pytest.raises(H3ValidationError, match="NaN or Inf"):
        construct_target_pose([math.nan, 0.0, 0.0], [0.0, 0.0, 1.0], coordinate_frame="base_link", config=CONFIG)
    with pytest.raises(H3ValidationError, match="NaN or Inf"):
        construct_target_pose([0.0, 0.0, 0.0], [0.0, math.inf, 1.0], coordinate_frame="base_link", config=CONFIG)


def test_missing_frame_fails_closed() -> None:
    with pytest.raises(H3ValidationError, match="coordinate_frame"):
        construct_target_pose([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], coordinate_frame="", config=CONFIG)


def test_missing_units_fail_closed_in_h2_input_contract() -> None:
    record = H2_RECORDS["fixture_planar_patch"]
    source = ROOT / record["source_path"]
    raw = load_geometry(source)
    config = h2_geometry_config_from_record(str(source), record)
    missing_unit = GeometryConfig(config.geometry_id, config.dataset_id, config.source_path, config.source_type, None, config.source_frame, config.target_frame, identity_matrix(), normal_policy=config.normal_policy, sampling_policy=config.sampling_policy, preprocessing_config=config.preprocessing_config)
    with pytest.raises(GeometryValidationError, match="unit"):
        canonicalize_geometry(missing_unit, copy.deepcopy(raw))


def test_degenerate_triangle_is_reported_without_repair() -> None:
    mesh, _processing, _derived = load_and_verify_h2_mesh(str(ROOT), H2_RECORDS["fixture_degenerate_triangle"])
    diagnostics = geometry_topology_diagnostics(mesh)
    assert diagnostics["degenerate_triangle_ids"] == [1]
    assert diagnostics["degenerate_triangle_policy"] == "REPORT_AND_EXCLUDE_FROM_H2_SAMPLING"
    assert diagnostics["silent_repair_applied"] is False


def test_normal_orientation_and_spray_sign_are_consistent() -> None:
    _mesh, tasks, _ = mesh_and_tasks("fixture_tunnel_like_patch")
    assert all(sum(a * b for a, b in zip(task["surface_normal_unit"], task["spray_direction_unit"])) == pytest.approx(-1.0, abs=1e-9) for task in tasks)


def test_roll_reference_degeneracy_fails_closed() -> None:
    degenerate_roll_config = H3Config(roll_reference_axes=((1.0, 0.0, 0.0),))
    with pytest.raises(H3ValidationError, match="degenerate"):
        construct_target_pose([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], coordinate_frame="base_link", config=degenerate_roll_config)


def test_tcp_standoff_reconstruction() -> None:
    pose = construct_target_pose([0.1, -0.2, 0.3], [0.0, 0.0, 1.0], coordinate_frame="base_link", config=CONFIG)
    reconstructed = [pose["tcp_target_position_xyz_m"][axis] + pose["spray_direction_unit"][axis] * CONFIG.nominal_standoff_m for axis in range(3)]
    assert reconstructed == pytest.approx([0.1, -0.2, 0.3], abs=1e-12)


def test_spray_direction_is_unit_and_tcp_local_z() -> None:
    pose = construct_target_pose([0.0, 0.0, 0.0], [1.0, 2.0, 3.0], coordinate_frame="base_link", config=CONFIG)
    assert math.sqrt(sum(value * value for value in pose["spray_direction_unit"])) == pytest.approx(1.0, abs=1e-12)
    assert quaternion_rotate_vector(pose["tcp_target_orientation_xyzw"], [0.0, 0.0, 1.0]) == pytest.approx(pose["spray_direction_unit"], abs=1e-9)


def test_footprint_assignment_is_deterministic() -> None:
    _mesh, first, _ = mesh_and_tasks("fixture_curved_cylinder_patch")
    _mesh, second, _ = mesh_and_tasks("fixture_curved_cylinder_patch")
    assert canonical_hash([task["coverage_footprint_parameters"] for task in first]) == canonical_hash([task["coverage_footprint_parameters"] for task in second])
    assert [task["task_sample_id"] for task in first] == [task["task_sample_id"] for task in second]


def test_coverage_ratio_controlled_fixture() -> None:
    _mesh, tasks, _ = mesh_and_tasks()
    result = evaluate_coverage([tasks[0]], [tasks[0]])
    assert result["metrics"]["surface_coverage_ratio"] == pytest.approx(1.0)
    assert result["metrics"]["uncovered_ratio"] == pytest.approx(0.0)


def test_overlap_controlled_fixture() -> None:
    _mesh, tasks, _ = mesh_and_tasks()
    result = evaluate_coverage([tasks[0]], [tasks[0], tasks[0]])
    assert result["metrics"]["coverage_overlap_redundancy"] > 0.0
    assert result["arrays"]["coverage_counts"] == [2]


def test_uncovered_controlled_fixture() -> None:
    _mesh, tasks, _ = mesh_and_tasks()
    result = evaluate_coverage([tasks[0]], [])
    assert result["metrics"]["surface_coverage_ratio"] == pytest.approx(0.0)
    assert result["metrics"]["uncovered_ratio"] == pytest.approx(1.0)


def test_three_run_deterministic_replay() -> None:
    runs = []
    for _ in range(3):
        _mesh, tasks, _ = mesh_and_tasks("fixture_tunnel_like_patch")
        runs.append((tasks, [evaluate_coverage(tasks, tasks)]))
    hashes = [semantic_output_hash(tasks, metrics) for tasks, metrics in runs]
    assert len(set(hashes)) == 1


def test_immutable_baseline_before_after() -> None:
    manifest = json.loads((ROOT / "outputs/stage3_h0_entry_authorization_20260807T161253Z/stage3_h0_stage2_immutable_baseline_manifest.json").read_text(encoding="utf-8"))
    assert all(Path(entry["absolute_path"]).is_file() and sha256_file(entry["absolute_path"]) == entry["sha256"] for entry in manifest["entries"])


def test_h3_gate_and_terminal_artifacts_are_frozen_and_passed() -> None:
    output_dirs = sorted(ROOT.glob("outputs/stage3_h3_coverage_baseline_*/stage3_h3_terminal_certificate.json"))
    assert output_dirs
    output_root = output_dirs[-1].parent
    gate = json.loads((output_root / "stage3_h3_gate_definition.json").read_text(encoding="utf-8"))
    certificate = json.loads((output_root / "stage3_h3_terminal_certificate.json").read_text(encoding="utf-8"))
    assert gate["status"] == "FROZEN_BEFORE_EXPERIMENT"
    assert certificate["stage3_h3"] == "PASSED"
    assert certificate["first_blocker"] == "none"
    assert certificate["new_fjt_goals_sent"] == 0
    assert certificate["send_goal_async_call_count"] == 0
    assert certificate["robot_motion_started"] == "NO"
    assert certificate["formal_ledger_mutated"] == "NO"

