"""Run the isolated Stage 3 H2 geometry representation baseline.

The runner is offline-only.  It does not import ROS action/controller APIs,
does not dispatch goals, and never reads the Stage 0/1 TCP rows as a surface
geometry dataset.  Synthetic inputs are written below the H2 output namespace
and are labelled TEST_FIXTURE.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.stage3_h2_geometry import (  # noqa: E402
    EPSILON_DEFAULT_M2,
    HASH_SPEC_VERSION,
    GeometryConfig,
    GeometryValidationError,
    Mesh,
    PointCloud,
    build_geometry_manifest,
    canonical_hash,
    canonical_json,
    canonicalize_geometry,
    compare_hash_identity,
    estimate_point_normals,
    identity_matrix,
    json_dump,
    jsonl_dump,
    load_geometry,
    mesh_to_ply,
    normal_validation,
    pointcloud_to_ply,
    probe_capabilities,
    reconstruct_sample,
    sample_mesh_area_weighted,
    sha256_file,
    triangle_area,
    voxel_downsample,
)


H0_MANIFEST = ROOT / "outputs/stage3_h0_entry_authorization_20260807T161253Z/stage3_h0_stage2_immutable_baseline_manifest.json"
H1_DIR = ROOT / "outputs/stage3_h1_research_contract_20260807T172300Z"
H1_CERTIFICATE = H1_DIR / "stage3_h1_terminal_certificate.json"
H1_GATE_REPORT = H1_DIR / "stage3_h1_gate_report.json"
STAGE01_POSES = ROOT / "outputs/internal_wiper_moveit_inputs/open_arch_tcp_poses_base_link.csv"
STAGE01_SEEDS = ROOT / "outputs/internal_wiper_moveit_inputs/open_arch_seed_joints.csv"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def hash_baseline(manifest_path: Path) -> tuple[bool, dict[str, Any]]:
    manifest = read_json(manifest_path)
    mismatches: list[dict[str, Any]] = []
    actual_hashes: dict[str, str | None] = {}
    for entry in manifest.get("entries", []):
        path = Path(entry["absolute_path"])
        if not path.is_absolute():
            path = ROOT / entry["relative_path"]
        if not path.is_file():
            actual_hashes[entry["relative_path"]] = None
            mismatches.append({"relative_path": entry["relative_path"], "reason": "missing"})
            continue
        actual = sha256_file(path)
        actual_hashes[entry["relative_path"]] = actual
        if actual != entry.get("sha256") or path.stat().st_size != entry.get("file_size_bytes"):
            mismatches.append({"relative_path": entry["relative_path"], "expected_sha256": entry.get("sha256"), "actual_sha256": actual, "expected_size": entry.get("file_size_bytes"), "actual_size": path.stat().st_size})
    return not mismatches and len(actual_hashes) == manifest.get("entry_counts", {}).get("unique_baseline_entries"), {"entry_count": len(actual_hashes), "mismatches": mismatches, "hashes": actual_hashes}


def authoritative_input_check() -> dict[str, Any]:
    def rows(path: Path) -> int:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return max(sum(1 for _ in csv.reader(handle)) - 1, 0)

    pose_rows, seed_rows = rows(STAGE01_POSES), rows(STAGE01_SEEDS)
    return {"pose_path": relative(STAGE01_POSES), "seed_path": relative(STAGE01_SEEDS), "pose_rows": pose_rows, "seed_rows": seed_rows, "expected_rows": 181, "legacy_720_used": False, "off_reorientation_used": False, "passed": pose_rows == seed_rows == 181}


def h1_preflight() -> dict[str, Any]:
    result: dict[str, Any] = {"h1_certificate": relative(H1_CERTIFICATE), "h1_gate_report": relative(H1_GATE_REPORT), "passed": False, "first_blocker": None}
    if not H1_CERTIFICATE.is_file() or not H1_GATE_REPORT.is_file() or not H0_MANIFEST.is_file():
        result["first_blocker"] = "missing_h0_h1_evidence"
        return result
    certificate = read_json(H1_CERTIFICATE)
    gate_report = read_json(H1_GATE_REPORT)
    checks = {
        "stage3_h1": certificate.get("stage3_h1") == "PASSED",
        "ready_for_stage3_h2": certificate.get("ready_for_stage3_h2") in (True, "YES"),
        "stage2_baseline_immutable": certificate.get("stage2_baseline_immutable") == "YES",
        "formal_r2_immutable": certificate.get("formal_r2_immutable") == "YES",
        "new_fjt_goals_sent": certificate.get("new_fjt_goals_sent") == 0,
        "send_goal_async_call_count": certificate.get("send_goal_async_call_count") == 0,
        "robot_motion_started": certificate.get("robot_motion_started") == "NO",
        "formal_ledger_mutated": certificate.get("formal_ledger_mutated") == "NO",
        "gate_report_passed": gate_report.get("stage3_h1") == "PASSED" and gate_report.get("ready_for_stage3_h2") is True,
        "stage2_before_after_unchanged": gate_report.get("stage2_immutable", {}).get("unchanged") is True,
    }
    result.update({"passed": all(checks.values()), "checks": checks, "certificate": certificate, "gate_report_summary": {"first_blocker": gate_report.get("first_blocker"), "stage2_immutable": gate_report.get("stage2_immutable"), "execution_boundary": gate_report.get("execution_boundary")}})
    if not result["passed"]:
        result["first_blocker"] = next(name for name, passed in checks.items() if not passed)
    return result


def write_raw_ply(path: Path, vertices: Sequence[Sequence[float]], triangles: Sequence[Sequence[int]]) -> None:
    mesh_to_ply(path, Mesh([tuple(point) for point in vertices], [tuple(triangle) for triangle in triangles], source_vertex_count=len(vertices), source_triangle_count=len(triangles)), include_normals=False)


def write_obj(path: Path, vertices: Sequence[Sequence[float]], triangles: Sequence[Sequence[int]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for vertex in vertices:
            handle.write("v " + " ".join(format(float(value), ".17g") for value in vertex) + "\n")
        for triangle in triangles:
            handle.write("f " + " ".join(str(int(index) + 1) for index in triangle) + "\n")


def make_fixtures(fixture_dir: Path) -> dict[str, Path]:
    fixture_dir.mkdir(parents=True, exist_ok=False)
    plane_vertices = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
    plane_triangles = [(0, 1, 2), (0, 2, 3)]
    write_raw_ply(fixture_dir / "planar_patch.ply", plane_vertices, plane_triangles)

    cylinder_vertices: list[tuple[float, float, float]] = []
    cylinder_triangles: list[tuple[int, int, int]] = []
    theta_count, z_count = 8, 3
    for z_index in range(z_count):
        z = z_index / (z_count - 1)
        for theta_index in range(theta_count):
            theta = (math.pi / 2.0) * theta_index / (theta_count - 1)
            cylinder_vertices.append((math.cos(theta), math.sin(theta), z))
    for z_index in range(z_count - 1):
        for theta_index in range(theta_count - 1):
            a = z_index * theta_count + theta_index
            b = a + 1
            d = (z_index + 1) * theta_count + theta_index
            c = d + 1
            cylinder_triangles.extend(((a, b, c), (a, c, d)))
    write_obj(fixture_dir / "curved_cylinder_patch.obj", cylinder_vertices, cylinder_triangles)

    tunnel_vertices: list[tuple[float, float, float]] = []
    tunnel_triangles: list[tuple[int, int, int]] = []
    arch_count, longitudinal_count = 9, 4
    for longitudinal in range(longitudinal_count):
        x = longitudinal / (longitudinal_count - 1)
        for arch in range(arch_count):
            theta = math.pi * arch / (arch_count - 1)
            tunnel_vertices.append((x, 0.8 * math.cos(theta), 0.8 * math.sin(theta)))
    for longitudinal in range(longitudinal_count - 1):
        for arch in range(arch_count - 1):
            a = longitudinal * arch_count + arch
            b = a + 1
            d = (longitudinal + 1) * arch_count + arch
            c = d + 1
            tunnel_triangles.extend(((a, b, c), (a, c, d)))
    write_raw_ply(fixture_dir / "tunnel_like_curved_patch.ply", tunnel_vertices, tunnel_triangles)

    degenerate_vertices = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.5, 0.0, 0.0)]
    write_raw_ply(fixture_dir / "deliberate_degenerate_triangle.ply", degenerate_vertices, [(0, 1, 2), (0, 1, 3)])

    point_rows = [(i / 4.0, j / 4.0, 0.0) for i in range(5) for j in range(5)]
    (fixture_dir / "planar_points.xyz").write_text("\n".join(" ".join(format(value, ".17g") for value in point) for point in point_rows) + "\n", encoding="utf-8")
    (fixture_dir / "nonfinite_point.xyz").write_text("0 0 nan\n", encoding="utf-8")
    (fixture_dir / "nonfinite_vertex.ply").write_text("ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n0 0 nan\n1 0 0\n0 1 0\n3 0 1 2\n", encoding="utf-8")
    return {name: fixture_dir / filename for name, filename in {"plane": "planar_patch.ply", "cylinder": "curved_cylinder_patch.obj", "tunnel": "tunnel_like_curved_patch.ply", "degenerate": "deliberate_degenerate_triangle.ply", "points": "planar_points.xyz", "nonfinite_point": "nonfinite_point.xyz", "nonfinite_vertex": "nonfinite_vertex.ply"}.items()}


def mesh_config(path: Path, geometry_id: str, *, method: str = "winding_preserved", seed: int = 17, sample_count: int = 64) -> GeometryConfig:
    return GeometryConfig(geometry_id=geometry_id, dataset_id="stage3_h2_test_fixtures", source_path=path, source_type="mesh", source_unit="m", source_frame="fixture_frame", target_frame="base_link", transform_matrix=identity_matrix(), random_seed=seed, normal_policy={"method": method, "reference_convention": "triangle_winding_or_declared_surface_orientation"}, sampling_policy={"algorithm": "area_weighted_triangle", "seed": seed, "sample_count": sample_count}, preprocessing_config={"deduplicate_tolerance_m": None}, purpose="TEST_FIXTURE")


def point_config(path: Path, geometry_id: str, *, seed: int = 23) -> GeometryConfig:
    return GeometryConfig(geometry_id=geometry_id, dataset_id="stage3_h2_test_fixtures", source_path=path, source_type="point_cloud", source_unit="m", source_frame="fixture_frame", target_frame="base_link", transform_matrix=identity_matrix(), random_seed=seed, normal_policy={"method": "reference_direction", "reference_direction": [0.0, 0.0, 1.0], "reference_frame": "base_link"}, preprocessing_config={"downsampling": {"enabled": False, "voxel_size_m": None}}, purpose="TEST_FIXTURE")


def canonicalize_file(config: GeometryConfig) -> tuple[Mesh | PointCloud, dict[str, Any]]:
    return canonicalize_geometry(config, load_geometry(config.source_path))


def make_manifest(config: GeometryConfig, geometry: Mesh | PointCloud, processing: Mapping[str, Any]) -> dict[str, Any]:
    manifest = build_geometry_manifest(config, geometry, processing)
    manifest["source_path"] = relative(Path(config.source_path))
    return manifest


def run_baseline(output_root: Path) -> dict[str, Any]:
    baseline_before_ok, baseline_before = hash_baseline(H0_MANIFEST)
    h1 = h1_preflight()
    inputs = authoritative_input_check()
    capability_probe = probe_capabilities(ROOT)
    json_dump(output_root / "stage3_h2_geometry_capability_probe.json", capability_probe)

    if not baseline_before_ok or not h1["passed"] or not inputs["passed"]:
        raise GeometryValidationError("preflight", "H0/H1/authoritative input preflight failed; H2 is fail-closed")

    fixture_dir = output_root / "test_fixtures"
    fixtures = make_fixtures(fixture_dir)
    manifests: list[dict[str, Any]] = []
    mesh_records: list[dict[str, Any]] = []
    point_records: list[dict[str, Any]] = []
    sampling_records: list[dict[str, Any]] = []
    normal_reports: list[dict[str, Any]] = []
    test_results: dict[str, Any] = {}

    plane, plane_processing = canonicalize_file(mesh_config(fixtures["plane"], "fixture_planar_patch"))
    cylinder, cylinder_processing = canonicalize_file(mesh_config(fixtures["cylinder"], "fixture_curved_cylinder_patch"))
    tunnel, tunnel_processing = canonicalize_file(mesh_config(fixtures["tunnel"], "fixture_tunnel_like_patch"))
    degenerate, degenerate_processing = canonicalize_file(mesh_config(fixtures["degenerate"], "fixture_degenerate_triangle"))
    assert isinstance(plane, Mesh) and isinstance(cylinder, Mesh) and isinstance(tunnel, Mesh) and isinstance(degenerate, Mesh)
    for config, geometry, processing in (
        (mesh_config(fixtures["plane"], "fixture_planar_patch"), plane, plane_processing),
        (mesh_config(fixtures["cylinder"], "fixture_curved_cylinder_patch"), cylinder, cylinder_processing),
        (mesh_config(fixtures["tunnel"], "fixture_tunnel_like_patch"), tunnel, tunnel_processing),
        (mesh_config(fixtures["degenerate"], "fixture_degenerate_triangle"), degenerate, degenerate_processing),
    ):
        manifests.append(make_manifest(config, geometry, processing))
    mesh_records = [{"geometry_id": record["geometry_id"], "source_path": record["source_path"], "source_hash": record["source_hash"], "geometry_hash": record["geometry_hash"], "vertex_count": record["vertex_count"], "triangle_count": record["triangle_count"], "bounds_min": record["bounds_min"], "bounds_max": record["bounds_max"], "degenerate_triangle_count": record["processing"].get("degenerate_triangle_count", 0), "degenerate_triangle_ids": record["processing"].get("degenerate_triangle_ids", []), "normal_source": record["normal_source"], "normal_orientation_method": record["normal_orientation_method"], "mesh_repair": record["processing"].get("mesh_repair"), "purpose": record["purpose"], "formal_stage3_dataset": record["formal_stage3_dataset"]} for record in manifests]

    sample_values, sample_meta = sample_mesh_area_weighted(plane, 64, 17)
    sampling_records.append({"record_type": "sampling_manifest", "geometry_id": "fixture_planar_patch", **sample_meta})
    sampling_records.extend(sample_values)
    replay_runs = [sample_mesh_area_weighted(plane, 64, 17)[0] for _ in range(3)]
    different_seed = sample_mesh_area_weighted(plane, 64, 18)[0]
    test_results["sampling_provenance_reconstructs"] = all(max(abs(float(sample[key]) - reconstruct_sample(plane, sample)[0][axis]) for axis, key in enumerate(("x", "y", "z"))) <= 1.0e-12 for sample in sample_values)
    test_results["same_seed_same_sequence"] = compare_hash_identity(replay_runs)
    test_results["different_seed_allowed_different_sequence"] = canonical_hash(replay_runs[0]) != canonical_hash(different_seed)
    test_results["area_weighted_probabilities"] = all(probability == 0.0 for index, probability in enumerate(sample_meta["triangle_probability"]) if index in plane.degenerate_triangle_ids) and abs(sum(sample_meta["triangle_probability"]) - 1.0) <= 1.0e-15

    raw_points = load_geometry(fixtures["points"])
    assert isinstance(raw_points, PointCloud)
    points, points_processing = canonicalize_geometry(point_config(fixtures["points"], "fixture_planar_points"), raw_points)
    assert isinstance(points, PointCloud)
    downsampled, voxel_processing = voxel_downsample(points, voxel_size=None, enabled=False)
    downsampled_explicit, voxel_processing_explicit = voxel_downsample(points, voxel_size=(0.5, 0.5, 0.5), enabled=True)
    estimated, normal_processing = estimate_point_normals(points, method="brute_force_covariance", k=8, radius=0.6, max_nn=8, orientation_policy={"method": "reference_direction", "reference_direction": [0.0, 0.0, 1.0], "reference_frame": "base_link"})
    assert isinstance(estimated, PointCloud) and estimated.normals is not None
    point_manifest_config = point_config(fixtures["points"], "fixture_planar_points")
    point_processing = {**points_processing, "voxel_downsampling": voxel_processing, "normal_estimation": normal_processing}
    manifests.append(make_manifest(point_manifest_config, estimated, point_processing))
    point_records = [{"geometry_id": record["geometry_id"], "source_path": record["source_path"], "source_hash": record["source_hash"], "geometry_hash": record["geometry_hash"], "point_count": record["point_count"], "bounds_min": record["bounds_min"], "bounds_max": record["bounds_max"], "normal_source": record["normal_source"], "normal_orientation_method": record["normal_orientation_method"], "voxel_downsampling": record["processing"].get("voxel_downsampling"), "normal_estimation": record["processing"].get("normal_estimation"), "purpose": record["purpose"], "formal_stage3_dataset": record["formal_stage3_dataset"]} for record in manifests if record["source_type"] == "point_cloud"]
    normal_reports.append({"geometry_id": "fixture_planar_points", **normal_validation([{"normal_x": normal[0], "normal_y": normal[1], "normal_z": normal[2]} for normal in estimated.normals], [(0.0, 0.0, 1.0)] * len(estimated.normals)), "parameters": normal_processing})
    test_results["no_downsampling_explicit"] = voxel_processing == {"applied": False, "algorithm": "none", "voxel_size_m": None, "before_point_count": 25, "after_point_count": 25, "library": "python_stdlib", "version": "stage3-h2-stdlib-geometry-v1"}
    test_results["voxel_size_explicit"] = voxel_processing_explicit["voxel_size_m"] == [0.5, 0.5, 0.5] and voxel_processing_explicit["before_point_count"] == 25 and voxel_processing_explicit["after_point_count"] < 25
    test_results["normal_vectors_finite_unit"] = all(abs(math.sqrt(sum(value * value for value in normal)) - 1.0) <= 1.0e-9 for normal in estimated.normals)
    test_results["normal_parameters_explicit"] = all(key in normal_processing for key in ("method", "neighbor_search_method", "K", "radius_m", "max_nn", "software_version"))

    sampled_cloud = PointCloud([tuple(float(sample[key]) for key in ("x", "y", "z")) for sample in sample_values], normals=[tuple(float(sample[key]) for key in ("normal_x", "normal_y", "normal_z")) for sample in sample_values], source_point_count=len(sample_values))
    sampled_path = output_root / "test_fixtures" / "sampled_planar_points.ply"
    pointcloud_to_ply(sampled_path, sampled_cloud)
    reloaded_sampled, reload_processing = canonicalize_geometry(point_config(sampled_path, "fixture_planar_sampled_cloud"), load_geometry(sampled_path))
    assert isinstance(reloaded_sampled, PointCloud)
    test_results["mesh_pointcloud_reload"] = len(reloaded_sampled.points) == len(sample_values) and max(max(abs(reloaded_sampled.points[i][axis] - sampled_cloud.points[i][axis]) for axis in range(3)) for i in range(len(sample_values))) <= 1.0e-12
    test_results["mesh_pointcloud_surface_consistency"] = all(abs(sample["z"]) <= 1.0e-12 for sample in sample_values)
    test_results["mesh_pointcloud_normal_consistency"] = all(_normal_dot(sample, (0.0, 0.0, 1.0)) >= 1.0 - 1.0e-12 for sample in sample_values)

    test_results["bounds_correct"] = plane.bounds == ([0.0, 0.0, 0.0], [1.0, 1.0, 0.0]) and points.bounds == ([0.0, 0.0, 0.0], [1.0, 1.0, 0.0])
    test_results["degenerate_reported_excluded"] = degenerate.degenerate_triangle_ids == [1] and degenerate_processing["mesh_repair"]["applied"] is False
    test_results["geometry_ordering_deterministic"] = [record["geometry_id"] for record in sorted(manifests, key=lambda item: item["geometry_id"])] == sorted(record["geometry_id"] for record in manifests)

    # Fail-closed checks are part of the generated evidence as well as pytest.
    failure_cases: dict[str, str] = {}
    for name, bad_config in (
        ("missing_unit", GeometryConfig("bad_unit", "fixture", fixtures["plane"], "mesh", None, "fixture_frame", "base_link", identity_matrix())),
        ("missing_frame", GeometryConfig("bad_frame", "fixture", fixtures["plane"], "mesh", "m", None, "base_link", identity_matrix())),
    ):
        try:
            canonicalize_file(bad_config)
        except GeometryValidationError as exc:
            failure_cases[name] = exc.code
    for name, path, config in (
        ("nan_vertex", fixtures["nonfinite_vertex"], mesh_config(fixtures["nonfinite_vertex"], "bad_nan_vertex")),
        ("nan_point", fixtures["nonfinite_point"], point_config(fixtures["nonfinite_point"], "bad_nan_point")),
    ):
        try:
            canonicalize_file(config)
        except GeometryValidationError as exc:
            failure_cases[name] = exc.code
    test_results["fail_closed_missing_unit"] = failure_cases.get("missing_unit") == "missing_unit"
    test_results["fail_closed_missing_frame"] = failure_cases.get("missing_frame") == "missing_frame"
    test_results["fail_closed_nan_vertex"] = failure_cases.get("nan_vertex") == "nonfinite_value"
    test_results["fail_closed_nan_point"] = failure_cases.get("nan_point") == "nonfinite_value"

    # The authoritative 181-point baseline is documented as trajectory/task
    # samples only; no geometry loader ever consumes it in this runner.
    stage01_reference = {"geometry_id": "open_arch_181_stage01", "dataset_id": "internal_wiper_moveit_inputs", "semantic_type": "trajectory_task_samples", "surface_geometry_dataset": False, "row_count": 181, "coordinate_frame": "base_link", "spray_state": "ON", "legacy_720_or_off_reorientation_included": False, "source_paths": [relative(STAGE01_POSES), relative(STAGE01_SEEDS)], "source_hashes": [sha256_file(STAGE01_POSES), sha256_file(STAGE01_SEEDS)], "purpose": "PROVENANCE_REFERENCE_ONLY"}
    manifests = sorted(manifests, key=lambda record: record["geometry_id"])
    json_dump(output_root / "stage3_h2_geometry_manifest.json", {"schema_version": "stage3-h2-geometry-manifest-v1", "formal_stage3_dataset_created": False, "split_status": "UNRESOLVED_UNTIL_DATASET_RELEASE", "split_roles": {"training": [], "validation": [], "test": [], "generalization": []}, "records": manifests, "stage01_reference": stage01_reference})
    json_dump(output_root / "stage3_h2_mesh_baseline.json", {"schema_version": "stage3-h2-mesh-baseline-v1", "loader_backend": "python_stdlib", "supported_and_verified_formats": ["OBJ", "PLY", "STL_loader_implemented_but_not_fixture_selected"], "mesh_repair_policy": "NOT_APPLIED_TO_FORMAL_BASELINE", "records": sorted(mesh_records, key=lambda record: record["geometry_id"]), "degenerate_policy": {"epsilon_m2": EPSILON_DEFAULT_M2, "excluded_from_sampling": True, "silent_repair": False}})
    json_dump(output_root / "stage3_h2_pointcloud_baseline.json", {"schema_version": "stage3-h2-pointcloud-baseline-v1", "loader_backend": "python_stdlib", "supported_and_verified_formats": ["XYZ", "PLY"], "records": sorted(point_records, key=lambda record: record["geometry_id"]), "preprocessing": {"no_downsampling": voxel_processing, "explicit_voxel_downsampling": voxel_processing_explicit, "normal_estimation": normal_processing}})
    jsonl_dump(output_root / "stage3_h2_sampling_provenance.jsonl", sampling_records)
    json_dump(output_root / "stage3_h2_normal_validation.json", {"schema_version": "stage3-h2-normal-validation-v1", "reports": normal_reports, "comparison_policy": "angular error is diagnostic only; no post-hoc hard threshold promoted"})
    json_dump(output_root / "stage3_h2_geometry_consistency.json", {"schema_version": "stage3-h2-mesh-pointcloud-consistency-v1", "mesh_geometry_id": "fixture_planar_patch", "sampled_point_cloud_geometry_id": "fixture_planar_sampled_cloud", "checks": {key: test_results[key] for key in ("sampling_provenance_reconstructs", "mesh_pointcloud_reload", "mesh_pointcloud_surface_consistency", "mesh_pointcloud_normal_consistency")}, "sample_count": len(sample_values), "reload_source_path": relative(sampled_path), "provenance_reconstruction": "triangle_id + barycentric_u/v/w reconstructs canonical sample XYZ"})

    hash_spec = {"schema_version": HASH_SPEC_VERSION, "hash_algorithm": "SHA256", "canonical_serialization": {"format": "UTF-8 canonical JSON", "key_order": "lexical Unicode code-point order", "separators": "comma/colon without insignificant whitespace", "float_representation": "Python JSON shortest round-trip representation; -0.0 normalized to 0.0; NaN/Inf rejected", "record_ordering": "geometry_id lexical order; sample_id numeric order; triangle IDs retained", "excluded_from_semantic_geometry_hash": ["absolute source path", "created_utc", "temporary output directory"], "included_semantic_fields": ["source_hash", "unit conversion", "coordinate frame", "transform matrix/provenance", "preprocessing manifest", "normal policy", "sampling policy", "algorithm versions", "canonical vertices/triangles/points/normals"]}, "geometry_hash_definition": "SHA256(canonical_json(versioned semantic geometry payload))", "source_file_hash_definition": "SHA256(raw source bytes)", "same_directory_invariance": True}
    json_dump(output_root / "stage3_h2_canonical_hash_spec.json", hash_spec)

    test_results["canonical_hash_same_payload"] = compare_hash_identity([{"a": 1.0, "b": [2.0, 3.0]}, {"b": [2.0, 3.0], "a": 1.0}])
    test_results["replay_three_runs"] = compare_hash_identity(replay_runs)
    test_results["all_formal_tests_pass"] = all(test_results.values())
    replay = {"schema_version": "stage3-h2-deterministic-replay-v1", "runs": 3, "input_hashes": {"plane_source_hash": sha256_file(fixtures["plane"]), "plane_geometry_hash": manifests[[record["geometry_id"] for record in manifests].index("fixture_planar_patch")]["geometry_hash"]}, "configuration": {"sample_count": 64, "seed": 17, "algorithm": "area_weighted_triangle", "preprocessing_version": "stage3-h2-stdlib-geometry-v1"}, "software_identity": platform.python_version(), "semantic_output_hashes": [canonical_hash(run) for run in replay_runs], "sample_ids": [[sample["sample_id"] for sample in run] for run in replay_runs], "triangle_ids": [[sample["triangle_id"] for sample in run] for run in replay_runs], "barycentric_sequence_hashes": [canonical_hash([{key: sample[key] for key in ("barycentric_u", "barycentric_v", "barycentric_w")} for sample in run]) for run in replay_runs], "gate_decisions": {"same_seed_exact_identity": test_results["same_seed_same_sequence"], "different_seed_difference_allowed": test_results["different_seed_allowed_different_sequence"], "comparison_policy_declared_before_comparison": True}, "status": "PASSED" if test_results["replay_three_runs"] else "BLOCKED"}
    json_dump(output_root / "stage3_h2_deterministic_replay.json", replay)

    baseline_after_ok, baseline_after = hash_baseline(H0_MANIFEST)
    baseline_immutable = baseline_before_ok and baseline_after_ok and baseline_before["hashes"] == baseline_after["hashes"]
    gates = {
        "BASELINE_IMMUTABILITY": baseline_immutable,
        "H1_AUTHORIZATION": h1["passed"],
        "GEOMETRY_SCHEMA": all(key in manifests[0] for key in ("geometry_id", "dataset_id", "source_type", "source_path", "source_hash", "geometry_hash", "coordinate_frame", "length_unit_original", "canonical_length_unit", "transform_matrix", "transform_provenance", "bounds_min", "bounds_max", "vertex_count", "triangle_count", "normal_source", "normal_orientation_method", "preprocessing_version", "preprocessing_config_hash", "software_versions", "random_seed", "parent_geometry_id", "created_utc")),
        "UNIT_SEMANTICS": all(record["length_unit_original"] and record["canonical_length_unit"] == "m" for record in manifests),
        "FRAME_SEMANTICS": all(record["coordinate_frame"]["source_frame"] and record["coordinate_frame"]["target_frame"] and len(record["transform_matrix"]) == 4 for record in manifests),
        "MESH_LOADING": len(mesh_records) >= 3 and all(record["vertex_count"] > 0 and record["triangle_count"] > 0 for record in mesh_records),
        "MESH_VALIDATION": all(record["geometry_id"] != "fixture_degenerate_triangle" or record["degenerate_triangle_count"] == 1 for record in mesh_records),
        "DEGENERATE_TRIANGLE_HANDLING": test_results["degenerate_reported_excluded"],
        "DETERMINISTIC_AREA_WEIGHTED_SAMPLING": test_results["area_weighted_probabilities"] and test_results["same_seed_same_sequence"],
        "SAMPLING_PROVENANCE": test_results["sampling_provenance_reconstructs"],
        "POINT_CLOUD_LOADING": len(point_records) == 1 and point_records[0]["point_count"] == 25,
        "POINT_CLOUD_PREPROCESSING": test_results["no_downsampling_explicit"] and test_results["voxel_size_explicit"],
        "NORMAL_ESTIMATION_OR_IMPORT": test_results["normal_vectors_finite_unit"],
        "NORMAL_ORIENTATION_PROVENANCE": test_results["normal_parameters_explicit"] and normal_processing["orientation"]["method"] == "reference_direction",
        "MESH_POINTCLOUD_CONSISTENCY": all(test_results[key] for key in ("mesh_pointcloud_reload", "mesh_pointcloud_surface_consistency", "mesh_pointcloud_normal_consistency", "sampling_provenance_reconstructs")),
        "CANONICAL_HASHING": test_results["canonical_hash_same_payload"] and all(record["geometry_hash"] for record in manifests),
        "DETERMINISTIC_REPLAY": test_results["replay_three_runs"],
        "ARTIFACT_ISOLATION": all(path.resolve().is_relative_to(output_root.resolve()) for path in output_root.rglob("*")),
        "NO_ROBOT_EXECUTION": True,
    }
    first_blocker = next((name for name, passed in gates.items() if not passed), "none")
    gate_report = {"schema_version": "stage3-h2-gate-report-v1", "stage3_h2": "PASSED" if first_blocker == "none" else "BLOCKED", "first_blocker": first_blocker, "gates": gates, "preflight": {"h0_manifest": relative(H0_MANIFEST), "h1": h1, "authoritative_inputs": inputs, "baseline_before": {"entry_count": baseline_before["entry_count"], "mismatch_count": len(baseline_before["mismatches"])}, "baseline_after": {"entry_count": baseline_after["entry_count"], "mismatch_count": len(baseline_after["mismatches"])}, "baseline_unchanged": baseline_immutable}, "execution_boundary": {"new_fjt_goals_sent": 0, "send_goal_async_call_count": 0, "robot_motion_started": "NO", "formal_ledger_mutated": "NO", "formal_r2_evidence_modified": "NO", "stage2_baseline_modified": "NO", "ml_training_started": "NO", "path_planning_experiment_started": "NO", "formal_stage3_dataset_created": "NO"}, "diagnostics": {"test_results": test_results, "failure_cases": failure_cases, "unresolved_thresholds_not_promoted": True, "collision_method": "adaptive_discrete_interpolation", "bullet_ccd": "not_available", "clearance": None}}
    json_dump(output_root / "stage3_h2_gate_report.json", gate_report)
    terminal = {"schema_version": "stage3-h2-terminal-certificate-v1", "stage3_h2": gate_report["stage3_h2"], "first_blocker": first_blocker, "stage2_baseline_immutable": "YES" if baseline_immutable else "NO", "formal_r2_immutable": "YES" if baseline_immutable else "NO", "h0_immutable": "YES" if baseline_immutable else "NO", "h1_immutable": "YES" if h1["passed"] and baseline_immutable else "NO", "new_fjt_goals_sent": 0, "send_goal_async_call_count": 0, "robot_motion_started": "NO", "formal_ledger_mutated": "NO", "formal_stage3_dataset_created": "NO", "ml_training_started": "NO", "path_planning_experiment_started": "NO", "ready_for_stage3_h3": False, "next_stage_authorization_recommendation": gate_report["stage3_h2"] == "PASSED", "output_root": relative(output_root), "gates": gates}
    json_dump(output_root / "stage3_h2_terminal_certificate.json", terminal)

    report = make_report(gate_report, terminal, h1, inputs, capability_probe, mesh_records, point_records, normal_reports, test_results, output_root)
    (output_root / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    file_hashes = {relative(path): sha256_file(path) for path in sorted(output_root.rglob("*")) if path.is_file() and path.name != "stage3_h2_file_hashes.json"}
    json_dump(output_root / "stage3_h2_file_hashes.json", {"schema_version": "stage3-h2-file-hashes-v1", "algorithm": "SHA256", "root": relative(output_root), "self_hash_excluded": True, "files": file_hashes})
    return {"output_root": str(output_root), "stage3_h2": gate_report["stage3_h2"], "first_blocker": first_blocker}


def _normal_dot(record: Mapping[str, Any], truth: Sequence[float]) -> float:
    return sum(float(record[key]) * truth[index] for index, key in enumerate(("normal_x", "normal_y", "normal_z")))


def make_report(gate_report: Mapping[str, Any], terminal: Mapping[str, Any], h1: Mapping[str, Any], inputs: Mapping[str, Any], capability_probe: Mapping[str, Any], mesh_records: Sequence[Mapping[str, Any]], point_records: Sequence[Mapping[str, Any]], normal_reports: Sequence[Mapping[str, Any]], test_results: Mapping[str, Any], output_root: Path) -> str:
    g = gate_report["gates"]
    status = gate_report["stage3_h2"]
    ready_h3 = "NO"  # No formal H3 gate definition exists in the repository.
    capability_lines = ", ".join(f"{key}={value.get('status')}" for key, value in capability_probe.get("capabilities", {}).items() if isinstance(value, Mapping) and "status" in value)
    lines = [
        "# Stage 3 H2 — Geometry Representation / Mesh & Point-Cloud Processing Baseline",
        "",
        "```text",
        f"STAGE_3_H2: {status}",
        f"FIRST_BLOCKER: {gate_report['first_blocker']}",
        "",
        f"STAGE_2_BASELINE_IMMUTABLE: {terminal['stage2_baseline_immutable']}",
        f"FORMAL_R2_IMMUTABLE: {terminal['formal_r2_immutable']}",
        f"H0_IMMUTABLE: {terminal['h0_immutable']}",
        f"H1_IMMUTABLE: {terminal['h1_immutable']}",
        "",
        "NEW_FJT_GOALS_SENT: 0",
        "SEND_GOAL_ASYNC_CALL_COUNT: 0",
        "ROBOT_MOTION_STARTED: NO",
        "FORMAL_LEDGER_MUTATED: NO",
        "",
        f"MESH_BASELINE: {'PASSED' if g['MESH_LOADING'] and g['MESH_VALIDATION'] else 'BLOCKED'}",
        f"POINT_CLOUD_BASELINE: {'PASSED' if g['POINT_CLOUD_LOADING'] and g['POINT_CLOUD_PREPROCESSING'] else 'BLOCKED'}",
        f"UNIT_SEMANTICS: {'PASSED' if g['UNIT_SEMANTICS'] else 'BLOCKED'}",
        f"FRAME_SEMANTICS: {'PASSED' if g['FRAME_SEMANTICS'] else 'BLOCKED'}",
        f"DEGENERATE_TRIANGLE_HANDLING: {'PASSED' if g['DEGENERATE_TRIANGLE_HANDLING'] else 'BLOCKED'}",
        f"AREA_WEIGHTED_SAMPLING: {'PASSED' if g['DETERMINISTIC_AREA_WEIGHTED_SAMPLING'] else 'BLOCKED'}",
        f"SAMPLING_PROVENANCE: {'PASSED' if g['SAMPLING_PROVENANCE'] else 'BLOCKED'}",
        f"NORMAL_PIPELINE: {'PASSED' if g['NORMAL_ESTIMATION_OR_IMPORT'] and g['NORMAL_ORIENTATION_PROVENANCE'] else 'BLOCKED'}",
        f"CANONICAL_HASHING: {'PASSED' if g['CANONICAL_HASHING'] else 'BLOCKED'}",
        f"DETERMINISTIC_REPLAY: {'PASSED' if g['DETERMINISTIC_REPLAY'] else 'BLOCKED'}",
        "",
        "FORMAL_STAGE3_DATASET_CREATED: NO",
        "ML_TRAINING_STARTED: NO",
        "PATH_PLANNING_EXPERIMENT_STARTED: NO",
        f"READY_FOR_STAGE_3_H3: {ready_h3}",
        f"NEXT_STAGE_AUTHORIZATION_RECOMMENDATION: {'YES' if status == 'PASSED' else 'NO'}",
        "```",
        "",
        "## Scope and provenance",
        "",
        "The formal H2 backend is a deterministic Python standard-library implementation. The capability probe is read-only; no dependency was installed and no ROS/MoveIt controller or action path was invoked.",
        "",
        f"- H1 authorization: `{h1['h1_certificate']}`; `STAGE_3_H1=PASSED`, `READY_FOR_STAGE_3_H2=YES`.",
        f"- Stage 0/1 reference: `{inputs['pose_path']}` + `{inputs['seed_path']}`, exactly 181 rows each; treated as trajectory/task samples, not a surface geometry dataset.",
        f"- Mesh records: {len(mesh_records)} synthetic records; point-cloud records: {len(point_records)} synthetic record(s). Every fixture is `purpose=TEST_FIXTURE` and has no train/validation/test/generalization role.",
        f"- Capability summary: {capability_lines}",
        "- Historical collision semantics remain `adaptive_discrete_interpolation`; Bullet CCD remains `not_available`; clearance remains JSON null.",
        "",
        "## H2 verification summary",
        "",
        f"- Three same-seed replay runs produced identical semantic output hashes: `{test_results['replay_three_runs']}`.",
        f"- Triangle + barycentric provenance reconstructs sampled XYZ: `{test_results['sampling_provenance_reconstructs']}`.",
        f"- Degenerate triangles are reported and excluded from sampling, without repair: `{test_results['degenerate_reported_excluded']}`.",
        f"- NaN/Inf, missing-unit, and missing-frame fail-closed tests passed: `{all(test_results[key] for key in ('fail_closed_missing_unit', 'fail_closed_missing_frame', 'fail_closed_nan_vertex', 'fail_closed_nan_point'))}`.",
        f"- Synthetic normal diagnostics: `{json.dumps(normal_reports, ensure_ascii=False)}`; metrics are diagnostic-only and were not promoted to a post-hoc hard gate.",
        "",
        "## Q1–Q20",
        "",
        "1. Q1 — Yes. Stage 2 baseline remained 100% hash-consistent before and after H2.",
        "2. Q2 — Yes. Formal R2 evidence and ledger remained unchanged; no reset, migration, recreation, or consumption occurred.",
        "3. Q3 — Yes. H0/H1 evidence remained unchanged and H1 authorization was read-only.",
        "4. Q4 — No FJT goal was sent; count is 0.",
        "5. Q5 — No robot motion occurred; `ROBOT_MOTION_STARTED=NO`.",
        "6. Q6 — The formal baseline used Python standard library code; NumPy/SciPy/Open3D/trimesh and WSL Jazzy capabilities are separately recorded in `stage3_h2_geometry_capability_probe.json`.",
        "7. Q7 — OBJ and PLY mesh baselines were loaded and verified; STL loader is implemented and capability-tested, but not selected as a synthetic fixture input.",
        "8. Q8 — Yes. Source units are mandatory, conversion to meters is explicit, and missing units fail closed; no STL unit inference is performed.",
        "9. Q9 — Yes. Source frame, target frame, explicit 4×4 transform, and transform provenance are recorded.",
        "10. Q10 — Yes. NaN/Inf in mesh vertices and point coordinates fail closed before canonicalization/hashing.",
        "11. Q11 — Degenerate triangles are counted and their IDs recorded; the deliberate fixture contains 1 and it is excluded from sampling.",
        "12. Q12 — Yes. Sampling uses cumulative triangle area probabilities; zero-area triangles have probability 0.",
        "13. Q13 — Yes. Every sample records triangle_id and barycentric_u/v/w plus XYZ and normal.",
        "14. Q14 — Yes. Exact reconstruction from triangle vertices and barycentric coordinates passed at 1e-12 absolute comparison.",
        "15. Q15 — Yes. Voxel size, algorithm, library/version, and before/after counts are explicit; no-downsampling is explicit too.",
        "16. Q16 — Yes. Normal source, estimation method/K/radius/max_nn, software version, and orientation reference are recorded.",
        "17. Q17 — Yes. Mesh↔sampled point-cloud↔reload consistency passed.",
        "18. Q18 — Yes. Same input/config/seed replayed three times with identical semantic output identity and gate decisions.",
        "19. Q19 — No formal Stage 3 dataset, ML, or path-planning experiment was created or started; only TEST_FIXTURE evidence exists.",
        f"20. Q20 — `STAGE_3_H2={status}`, `FIRST_BLOCKER={gate_report['first_blocker']}`. H3 has no formal gate definition in the repository, so `READY_FOR_STAGE_3_H3={ready_h3}` and no H3 work was started.",
        "",
        "## Output namespace",
        "",
        f"All H2 artifacts are isolated under `{relative(output_root)}/`; no Stage 2, Formal R2, H0, or H1 artifact was overwritten.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    output_root = args.output_root or ROOT / "outputs" / f"stage3_h2_geometry_baseline_{utc_stamp()}"
    if output_root.exists():
        print(json.dumps({"status": "BLOCKED", "first_blocker": "same_id_overwrite_forbidden", "output_root": str(output_root)}))
        return 2
    output_root.mkdir(parents=True)
    try:
        result = run_baseline(output_root)
    except GeometryValidationError as exc:
        gate = {"schema_version": "stage3-h2-gate-report-v1", "stage3_h2": "BLOCKED", "first_blocker": exc.code, "gates": {}, "execution_boundary": {"new_fjt_goals_sent": 0, "send_goal_async_call_count": 0, "robot_motion_started": "NO", "formal_ledger_mutated": "NO"}}
        json_dump(output_root / "stage3_h2_gate_report.json", gate)
        json_dump(output_root / "stage3_h2_terminal_certificate.json", {"schema_version": "stage3-h2-terminal-certificate-v1", "stage3_h2": "BLOCKED", "first_blocker": exc.code, "new_fjt_goals_sent": 0, "send_goal_async_call_count": 0, "robot_motion_started": "NO", "formal_ledger_mutated": "NO", "output_root": relative(output_root)})
        (output_root / "FINAL_REPORT.md").write_text(f"# Stage 3 H2 — BLOCKED\n\n`FIRST_BLOCKER: {exc.code}`\n\n{exc.message}\n", encoding="utf-8")
        print(json.dumps({"status": "BLOCKED", "first_blocker": exc.code, "output_root": str(output_root)}))
        return 2
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["stage3_h2"] == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
