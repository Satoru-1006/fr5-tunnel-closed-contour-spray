from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from src.stage3_h2_geometry import (
    GeometryConfig,
    GeometryValidationError,
    Mesh,
    PointCloud,
    canonical_hash,
    canonicalize_geometry,
    compare_hash_identity,
    estimate_point_normals,
    identity_matrix,
    load_geometry,
    sample_mesh_area_weighted,
    sha256_file,
    voxel_downsample,
)


ROOT = Path(__file__).resolve().parents[1]
H0_MANIFEST = ROOT / "outputs/stage3_h0_entry_authorization_20260807T161253Z/stage3_h0_stage2_immutable_baseline_manifest.json"


def write_ply(path: Path, vertices: list[tuple[float, float, float]], triangles: list[tuple[int, int, int]] | None = None) -> None:
    triangles = triangles or []
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(vertices)}\nproperty float x\nproperty float y\nproperty float z\n")
        if triangles:
            handle.write(f"element face {len(triangles)}\nproperty list uchar int vertex_indices\n")
        handle.write("end_header\n")
        handle.writelines("%s %s %s\n" % point for point in vertices)
        handle.writelines("3 %s %s %s\n" % triangle for triangle in triangles)


def mesh_config(path: Path, *, unit: str | None = "m", source_frame: str | None = "source", target_frame: str | None = "target", transform: list[list[float]] | None = None) -> GeometryConfig:
    return GeometryConfig("test_mesh", "test_dataset", path, "mesh", unit, source_frame, target_frame, transform or identity_matrix(), normal_policy={"method": "reference_direction", "reference_direction": [0, 0, 1]})


def cloud_config(path: Path) -> GeometryConfig:
    return GeometryConfig("test_cloud", "test_dataset", path, "point_cloud", "m", "source", "target", identity_matrix(), normal_policy={"method": "reference_direction", "reference_direction": [0, 0, 1]})


def simple_mesh(tmp_path: Path) -> tuple[Path, Mesh]:
    path = tmp_path / "mesh.ply"
    vertices = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    write_ply(path, vertices, [(0, 1, 2)])
    geometry, _ = canonicalize_geometry(mesh_config(path), load_geometry(path))
    assert isinstance(geometry, Mesh)
    return path, geometry


def test_missing_unit_and_frame_fail_closed(tmp_path: Path) -> None:
    path, _ = simple_mesh(tmp_path)
    with pytest.raises(GeometryValidationError, match="unit") as unit_error:
        canonicalize_geometry(mesh_config(path, unit=None), load_geometry(path))
    assert unit_error.value.code == "missing_unit"
    with pytest.raises(GeometryValidationError, match="frame") as frame_error:
        canonicalize_geometry(mesh_config(path, source_frame=None), load_geometry(path))
    assert frame_error.value.code == "missing_frame"


def test_nonfinite_mesh_and_point_fail_closed(tmp_path: Path) -> None:
    mesh_path = tmp_path / "nan.ply"
    write_ply(mesh_path, [(math.nan, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)], [(0, 1, 2)])
    with pytest.raises(GeometryValidationError) as mesh_error:
        canonicalize_geometry(mesh_config(mesh_path), load_geometry(mesh_path))
    assert mesh_error.value.code == "nonfinite_value"
    point_path = tmp_path / "nan.xyz"
    point_path.write_text("0 0 inf\n", encoding="utf-8")
    with pytest.raises(GeometryValidationError) as point_error:
        canonicalize_geometry(cloud_config(point_path), load_geometry(point_path))
    assert point_error.value.code == "nonfinite_value"


def test_degenerate_triangle_is_reported_and_not_sampled(tmp_path: Path) -> None:
    path = tmp_path / "degenerate.ply"
    write_ply(path, [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0.5, 0, 0)], [(0, 1, 2), (0, 1, 3)])
    geometry, processing = canonicalize_geometry(mesh_config(path), load_geometry(path))
    assert isinstance(geometry, Mesh)
    assert processing["degenerate_triangle_ids"] == [1]
    samples, metadata = sample_mesh_area_weighted(geometry, 50, 7)
    assert all(sample["triangle_id"] == 0 for sample in samples)
    assert metadata["triangle_probability"][1] == 0.0


def test_area_sampling_provenance_reconstructs_exact_point(tmp_path: Path) -> None:
    _, mesh = simple_mesh(tmp_path)
    samples, _ = sample_mesh_area_weighted(mesh, 10, 3)
    assert all(abs(sample["barycentric_u"] + sample["barycentric_v"] + sample["barycentric_w"] - 1.0) <= 1e-12 for sample in samples)
    for sample in samples:
        triangle = mesh.triangles[sample["triangle_id"]]
        expected = tuple(sum(weight * mesh.vertices[triangle[index]][axis] for index, weight in enumerate((sample["barycentric_u"], sample["barycentric_v"], sample["barycentric_w"]))) for axis in range(3))
        assert expected == pytest.approx((sample["x"], sample["y"], sample["z"]), abs=1e-12)
        assert {"sample_id", "triangle_id", "barycentric_u", "barycentric_v", "barycentric_w", "x", "y", "z", "normal_x", "normal_y", "normal_z"} <= set(sample)


def test_same_seed_replays_and_different_seed_can_differ(tmp_path: Path) -> None:
    _, mesh = simple_mesh(tmp_path)
    runs = [sample_mesh_area_weighted(mesh, 20, 11)[0] for _ in range(3)]
    other = sample_mesh_area_weighted(mesh, 20, 12)[0]
    assert compare_hash_identity(runs)
    assert canonical_hash(runs[0]) != canonical_hash(other)


def test_explicit_transform_and_bounds(tmp_path: Path) -> None:
    path = tmp_path / "transform.ply"
    write_ply(path, [(0, 0, 0), (1, 0, 0), (0, 1, 0)], [(0, 1, 2)])
    transform = identity_matrix()
    transform[0][3] = 2.0
    transform[1][3] = -1.0
    mesh, _ = canonicalize_geometry(mesh_config(path, transform=transform), load_geometry(path))
    assert isinstance(mesh, Mesh)
    assert mesh.bounds == ([2.0, -1.0, 0.0], [3.0, 0.0, 0.0])


def test_xyz_voxel_size_is_explicit_and_normal_parameters_are_explicit(tmp_path: Path) -> None:
    path = tmp_path / "points.xyz"
    path.write_text("\n".join(f"{i / 4} {j / 4} 0" for i in range(5) for j in range(5)) + "\n", encoding="utf-8")
    cloud, _ = canonicalize_geometry(cloud_config(path), load_geometry(path))
    assert isinstance(cloud, PointCloud)
    no_downsampled, no_downsample = voxel_downsample(cloud, voxel_size=None, enabled=False)
    downsampled, downsample = voxel_downsample(cloud, voxel_size=(0.5, 0.5, 0.5), enabled=True)
    assert len(no_downsampled.points) == 25
    assert downsample["voxel_size_m"] == [0.5, 0.5, 0.5]
    assert downsample["before_point_count"] == 25 and downsample["after_point_count"] < 25
    estimated, normal_parameters = estimate_point_normals(cloud, method="brute_force_covariance", k=8, radius=0.6, max_nn=8, orientation_policy={"method": "reference_direction", "reference_direction": [0, 0, 1]})
    assert isinstance(estimated, PointCloud) and estimated.normals is not None
    assert all(math.isfinite(value) and abs(math.sqrt(sum(component * component for component in normal)) - 1.0) < 1e-9 for normal in estimated.normals for value in normal)
    assert {"method", "neighbor_search_method", "K", "radius_m", "max_nn", "software_version"} <= set(normal_parameters)


def test_stl_loader_is_real_and_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "triangle.stl"
    path.write_text("solid tri\n facet normal 0 0 1\n  outer loop\n   vertex 0 0 0\n   vertex 1 0 0\n   vertex 0 1 0\n  endloop\n endfacet\nendsolid tri\n", encoding="utf-8")
    mesh = load_geometry(path)
    assert isinstance(mesh, Mesh)
    assert len(mesh.vertices) == 3 and len(mesh.triangles) == 1


def test_stage2_immutable_hashes_unchanged() -> None:
    manifest = json.loads(H0_MANIFEST.read_text(encoding="utf-8"))
    assert len(manifest["entries"]) == 125
    mismatches = []
    for entry in manifest["entries"]:
        path = Path(entry["absolute_path"])
        if path.is_file() and sha256_file(path) == entry["sha256"] and path.stat().st_size == entry["file_size_bytes"]:
            continue
        mismatches.append(entry["relative_path"])
    assert mismatches == []


def test_latest_h2_artifact_is_passed_without_execution() -> None:
    candidates = sorted(ROOT.glob("outputs/stage3_h2_geometry_baseline_*/stage3_h2_terminal_certificate.json"))
    assert candidates, "run scripts/stage3_h2_geometry_baseline.py before the formal H2 artifact test"
    passed = [path for path in candidates if json.loads(path.read_text(encoding="utf-8"))["stage3_h2"] == "PASSED"]
    assert passed
    certificate = json.loads(passed[-1].read_text(encoding="utf-8"))
    assert certificate["new_fjt_goals_sent"] == 0
    assert certificate["send_goal_async_call_count"] == 0
    assert certificate["robot_motion_started"] == "NO"
    assert certificate["formal_ledger_mutated"] == "NO"
    assert certificate["formal_stage3_dataset_created"] == "NO"
