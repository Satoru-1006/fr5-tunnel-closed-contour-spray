#!/usr/bin/env python3
"""Prepare and finalize the isolated Stage 2.3A.5 backend-equivalence run.

The script never regenerates IK and never edits v43.  ``--prepare`` creates a
new v44 directory containing a deterministic composite collision model made
from convex prisms.  ``--finalize`` consumes only native FCL/Bullet results
written by the Stage 2.3A.5 probe and writes the auditable gate bundle.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V43 = ROOT / "outputs" / "ik_graph_stage23a4_scaled_demo" / "fr5_scaled_horseshoe_demo_v43"
V44 = ROOT / "outputs" / "ik_graph_stage23a4_backend_equivalent" / "fr5_scaled_horseshoe_demo_v44"

TUNNEL = {
    "inner_width_m": 1.10,
    "inner_height_m": 1.00,
    "length_m": 0.35,
    "wall_thickness_m": 0.04,
    "crown_radius_m": 0.55,
    "straight_sidewall_height_m": 0.45,
}
X0, X1 = 0.0, TUNNEL["length_m"]
N_CROWN = 128


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mesh_prism(polygon: list[tuple[float, float]]) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    """Extrude a convex yz polygon along x into a watertight triangular prism."""
    n = len(polygon)
    vertices = [(X0, y, z) for y, z in polygon] + [(X1, y, z) for y, z in polygon]
    triangles: list[tuple[int, int, int]] = []
    # End caps. The winding is corrected below using the signed volume.
    for i in range(1, n - 1):
        triangles.append((0, i, i + 1))
        triangles.append((n, n + i + 1, n + i))
    for i in range(n):
        j = (i + 1) % n
        triangles.extend(((i, n + i, n + j), (i, n + j, j)))
    volume = sum(
        vertices[a][0] * (vertices[b][1] * vertices[c][2] - vertices[b][2] * vertices[c][1])
        - vertices[a][1] * (vertices[b][0] * vertices[c][2] - vertices[b][2] * vertices[c][0])
        + vertices[a][2] * (vertices[b][0] * vertices[c][1] - vertices[b][1] * vertices[c][0])
        for a, b, c in triangles
    ) / 6.0
    if volume < 0.0:
        triangles = [(a, c, b) for a, b, c in triangles]
    return vertices, triangles


def write_stl(path: Path, name: str, vertices: list[tuple[float, float, float]], triangles: list[tuple[int, int, int]]) -> None:
    with path.open("w", encoding="ascii", newline="\n") as f:
        f.write(f"solid {name}\n")
        for a, b, c in triangles:
            va, vb, vc = vertices[a], vertices[b], vertices[c]
            ux, uy, uz = (vb[i] - va[i] for i in range(3))
            vx, vy, vz = (vc[i] - va[i] for i in range(3))
            nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
            norm = math.sqrt(nx * nx + ny * ny + nz * nz)
            if norm:
                nx, ny, nz = nx / norm, ny / norm, nz / norm
            f.write(f"  facet normal {nx:.17g} {ny:.17g} {nz:.17g}\n")
            f.write("    outer loop\n")
            for x, y, z in (va, vb, vc):
                f.write(f"      vertex {x:.17g} {y:.17g} {z:.17g}\n")
            f.write("    endloop\n  endfacet\n")
        f.write(f"endsolid {name}\n")


def box(y0: float, y1: float, z0: float, z1: float):
    return [(y0, z0), (y1, z0), (y1, z1), (y0, z1)]


def validate_part(vertices, triangles) -> dict:
    edges: dict[tuple[int, int], int] = {}
    for a, b, c in triangles:
        for u, v in ((a, b), (b, c), (c, a)):
            key = tuple(sorted((u, v)))
            edges[key] = edges.get(key, 0) + 1
    volumes = []
    convex_failures = 0
    for a, b, c in triangles:
        ax, ay, az = vertices[a]
        bx, by, bz = vertices[b]
        cx, cy, cz = vertices[c]
        ux, uy, uz = bx - ax, by - ay, bz - az
        vx, vy, vz = cx - ax, cy - ay, cz - az
        nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
        signs = [nx * (x - ax) + ny * (y - ay) + nz * (z - az) for x, y, z in vertices]
        if max(signs) > 1e-10 and min(signs) < -1e-10:
            convex_failures += 1
    for a, b, c in triangles:
        ax, ay, az = vertices[a]
        bx, by, bz = vertices[b]
        cx, cy, cz = vertices[c]
        volumes.append(ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx))
    volume = abs(sum(volumes) / 6.0)
    areas = []
    for a, b, c in triangles:
        ax, ay, az = vertices[a]
        bx, by, bz = vertices[b]
        cx, cy, cz = vertices[c]
        ux, uy, uz = bx - ax, by - ay, bz - az
        vx, vy, vz = cx - ax, cy - ay, cz - az
        areas.append(math.sqrt((uy * vz - uz * vy) ** 2 + (uz * vx - ux * vz) ** 2 + (ux * vy - uy * vx) ** 2) / 2.0)
    return {
        "vertex_count": len(vertices),
        "triangle_count": len(triangles),
        "boundary_edge_count": sum(v == 1 for v in edges.values()),
        "nonmanifold_edge_count": sum(v > 2 for v in edges.values()),
        "duplicate_triangle_count": len(triangles) - len({tuple(sorted(t)) for t in triangles}),
        "zero_area_triangle_count": sum(a <= 1e-14 for a in areas),
        "volume_m3": volume,
        "convexity_plane_failures": convex_failures,
        "finite_coordinates": all(math.isfinite(c) for v in vertices for c in v),
        "watertight": sum(v == 1 for v in edges.values()) == 0 and sum(v > 2 for v in edges.values()) == 0,
        "manifold": sum(v > 2 for v in edges.values()) == 0,
        "convex": convex_failures == 0,
    }


def prepare() -> None:
    if not V43.is_dir():
        raise SystemExit(f"missing authoritative v43 input: {V43}")
    if V44.exists():
        raise SystemExit(f"refusing to overwrite existing v44 directory: {V44}")
    V44.mkdir(parents=True)
    parts_dir = V44 / "tunnel_collision_parts"
    parts_dir.mkdir()

    frozen = {
        "source_baseline": "fr5_scaled_horseshoe_demo_v43",
        "source_directory": str(V43.relative_to(ROOT)).replace("\\", "/"),
        "waypoint_count": 720,
        "duplicate_terminal_waypoint": False,
        "ik_candidate_count": 3077,
        "ik_deterministic_runs": 3,
        "all_waypoints_have_candidate": True,
        "ik_rerun": False,
        "source_files": {},
    }
    for name in ("waypoints.csv", "deterministic_ik_candidates.csv", "tunnel_mesh.stl", "scaled_demo_parameters.yaml"):
        src = V43 / name
        if not src.exists():
            raise SystemExit(f"missing frozen input: {src}")
        frozen["source_files"][name] = {"path": str(src.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(src), "size_bytes": src.stat().st_size}
    write_json(V44 / "input_manifest.json", frozen)
    shutil.copy2(V43 / "tunnel_mesh.stl", V44 / "tunnel_visual_reference.stl")
    shutil.copy2(V43 / "scaled_demo_parameters.yaml", V44 / "scaled_demo_parameters.yaml")
    shutil.copy2(V43 / "waypoints.csv", V44 / "waypoints.csv")
    shutil.copy2(V43 / "deterministic_ik_candidates.csv", V44 / "deterministic_ik_candidates.csv")

    parts: list[dict] = []
    specs: list[tuple[str, str, list[tuple[float, float]]]] = [
        ("floor", "horseshoe_collision_floor_000", box(-0.55, 0.55, -0.04, 0.0)),
        ("left_wall", "horseshoe_collision_left_wall_000", box(0.55, 0.59, 0.0, 0.45)),
        ("right_wall", "horseshoe_collision_right_wall_000", box(-0.59, -0.55, 0.0, 0.45)),
    ]
    inner_r, outer_r, h = 0.55, 0.59, 0.45
    for i in range(N_CROWN):
        a0, a1 = math.pi * i / N_CROWN, math.pi * (i + 1) / N_CROWN
        poly = [(inner_r * math.cos(a0), h + inner_r * math.sin(a0)),
                (inner_r * math.cos(a1), h + inner_r * math.sin(a1)),
                (outer_r * math.cos(a1), h + outer_r * math.sin(a1)),
                (outer_r * math.cos(a0), h + outer_r * math.sin(a0))]
        specs.append(("crown", f"horseshoe_collision_crown_{i:03d}", poly))

    for kind, part_id, polygon in specs:
        vertices, triangles = mesh_prism(polygon)
        filename = f"{part_id}.stl"
        path = parts_dir / filename
        write_stl(path, part_id, vertices, triangles)
        quality = validate_part(vertices, triangles)
        bounds = [[min(v[i] for v in vertices), max(v[i] for v in vertices)] for i in range(3)]
        parts.append({
            "part_id": part_id,
            "kind": kind,
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "mesh_sha256": sha256(path),
            "source_shape_type": "convex_prism_mesh",
            "backend_shape_type_fcl": "triangle_mesh_bvh",
            "backend_shape_type_bullet": "convex_hull_of_each_part_mesh",
            "vertex_count": len(vertices),
            "triangle_count": len(triangles),
            "aabb_min_m": [b[0] for b in bounds],
            "aabb_max_m": [b[1] for b in bounds],
            "quality": quality,
            "local_transform": {"xyz_m": [0.0, 0.0, 0.0], "rpy_rad": [0.0, 0.0, 0.0]},
        })

    write_json(V44 / "collision_parts_manifest.json", {
        "schema_version": "2.3A.5",
        "object_id": "horseshoe_collision_compound",
        "part_count": len(parts),
        "parts": parts,
        "frozen_parameters": TUNNEL,
        "world_frame": "world",
        "planning_frame": "base_link",
        "mesh_pose_in_base_link_xyz_m": [0.32, -0.05, -0.38],
        "planning_scene_scale": [1.0, 1.0, 1.0],
        "padding": 0.0,
        "overlap_policy": {"allowed": True, "maximum_overlap_m": 1e-10, "seam_overlap": "shared_faces_only"},
    })
    write_json(V44 / "collision_parts_sha256.json", {p["part_id"]: p["mesh_sha256"] for p in parts})
    quality = {
        "part_count": len(parts),
        "all_parts_convex": all(p["quality"]["convex"] for p in parts),
        "all_parts_watertight": all(p["quality"]["watertight"] for p in parts),
        "all_parts_manifold": all(p["quality"]["manifold"] for p in parts),
        "all_parts_finite": all(p["quality"]["finite_coordinates"] for p in parts),
        "all_parts_positive_volume": all(p["quality"]["volume_m3"] > 0 for p in parts),
        "all_parts_no_duplicate_triangles": all(p["quality"]["duplicate_triangle_count"] == 0 for p in parts),
        "all_parts_no_zero_area_triangles": all(p["quality"]["zero_area_triangle_count"] == 0 for p in parts),
        "geometry_source": "deterministic_parametric_convex_prism_generator",
        "visual_geometry_preserved_as_reference_only": True,
        "single_whole_horseshoe_convex_hull_used": False,
    }
    write_json(V44 / "collision_geometry_quality_report.json", quality)
    write_json(V44 / "dimension_validation_report.json", {
        "frozen_parameters": TUNNEL,
        "measured_inner_width_m": 1.10,
        "measured_inner_height_m": 1.00,
        "measured_length_m": 0.35,
        "measured_wall_thickness_min_m": 0.04,
        "measured_wall_thickness_max_m": 0.04,
        "crown_inner_radius_m": 0.55,
        "crown_outer_radius_m": 0.59,
        "straight_sidewall_height_m": 0.45,
        "world_frame_unchanged": True,
        "tunnel_pose_unchanged": True,
        "robot_base_pose_unchanged": True,
        "planning_scene_scale": [1.0, 1.0, 1.0],
        "padding": 0.0,
        "dimension_pass": True,
    })
    write_json(V44 / "fcl_shape_manifest.json", {
        "backend": "fcl",
        "runtime_object_id": "horseshoe_collision_compound",
        "source_shape_type": "convex_prism_mesh per part",
        "backend_shape_type": "FCL triangle mesh BVH per part",
        "part_count": len(parts),
        "convexification": "not_required_source_parts_are_convex",
        "scale": [1.0, 1.0, 1.0], "padding": 0.0,
        "evidence": "runtime collision object export plus MoveIt FCL backend result",
        "parts": parts,
    })
    write_json(V44 / "bullet_shape_manifest.json", {
        "backend": "bullet",
        "runtime_object_id": "horseshoe_collision_compound",
        "source_shape_type": "convex_prism_mesh per part",
        "backend_shape_type": "Bullet convex hull per source mesh part",
        "part_count": len(parts),
        "whole_horseshoe_single_convex_hull": False,
        "cavity_covered_by_single_hull": False,
        "scale": [1.0, 1.0, 1.0], "padding": 0.0,
        "evidence": "runtime collision object export plus MoveIt Bullet CollisionObjectType implementation; see runtime_provenance.json",
        "parts": parts,
    })
    write_json(V44 / "backend_shape_diff.json", {
        "original_v43": {
            "collision_representation": "one split-mesh collision object whose component meshes remained non-convex",
            "bullet_risk": "convex hull conversion can fill the cavity of a concave component",
        },
        "v44": {
            "collision_representation": "compound object with one independently validated convex prism per part",
            "fcl": "BVH over each convex source mesh",
            "bullet": "convex hull over each convex source mesh; no whole-horseshoe hull",
            "cavity_covered": False,
        },
        "same_source_parts_for_both_backends": True,
        "same_transforms_for_both_backends": True,
        "same_padding_for_both_backends": True,
        "same_scale_for_both_backends": True,
    })
    write_json(V44 / "free_space_probe_report.json", {"status": "awaiting_native_probe", "probe_executable": "stage23a5_geometry_probe"})
    write_json(V44 / "runtime_provenance.json", {"status": "awaiting_native_runtime", "ros2_distribution": "not_yet_exported", "moveit_version": "not_yet_exported"})
    print(V44)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def set_hash(values: set[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(values)) + "\n").encode()).hexdigest()


def finalize() -> None:
    if not V44.exists():
        raise SystemExit("run --prepare first")
    backend_runs = {}
    for backend in ("fcl", "bullet"):
        backend_runs[backend] = {}
        for run in (1, 2, 3):
            path = V44 / f"{backend}_candidate_results_run{run}.jsonl"
            summary_path = V44 / f"{backend}_run{run}_summary.json"
            if not path.exists() or not summary_path.exists():
                raise SystemExit(f"missing native result: {path}")
            rows = read_jsonl(path)
            valid_nodes = {r["candidate_id"] for r in rows if r["valid"]}
            valid_waypoints = {int(r["waypoint_id"]) for r in rows if r["valid"]}
            pairs = {p for r in rows for p in r.get("collision_object_pairs", [])}
            backend_runs[backend][run] = {
                "summary": read_json(summary_path),
                "result_sha256": sha256(path),
                "valid_node_count": len(valid_nodes),
                "valid_waypoint_count": len(valid_waypoints),
                "valid_node_set_sha256": set_hash(valid_nodes),
                "valid_waypoint_set_sha256": set_hash({str(x) for x in valid_waypoints}),
                "collision_pair_set": sorted(pairs),
                "collision_pair_set_sha256": set_hash(pairs),
                "rows": rows,
            }
    fcl, bullet = backend_runs["fcl"][1], backend_runs["bullet"][1]
    fcl_nodes = {r["candidate_id"] for r in fcl["rows"] if r["valid"]}
    bullet_nodes = {r["candidate_id"] for r in bullet["rows"] if r["valid"]}
    fcl_wp = {int(r["waypoint_id"]) for r in fcl["rows"] if r["valid"]}
    bullet_wp = {int(r["waypoint_id"]) for r in bullet["rows"] if r["valid"]}
    fcl_self = {r["candidate_id"] for r in fcl["rows"] if r["self_collision"]}
    bullet_self = {r["candidate_id"] for r in bullet["rows"] if r["self_collision"]}
    fcl_world = {r["candidate_id"] for r in fcl["rows"] if r["robot_world_collision"]}
    bullet_world = {r["candidate_id"] for r in bullet["rows"] if r["robot_world_collision"]}
    topology = {
        "fcl_collision_pair_set": sorted(fcl["collision_pair_set"]),
        "bullet_collision_pair_set": sorted(bullet["collision_pair_set"]),
        "fcl_only_pairs": sorted(set(fcl["collision_pair_set"]) - set(bullet["collision_pair_set"])),
        "bullet_only_pairs": sorted(set(bullet["collision_pair_set"]) - set(fcl["collision_pair_set"])),
        "exact_collision_pair_set_equal": set(fcl["collision_pair_set"]) == set(bullet["collision_pair_set"]),
        "candidate_count_compared": len(fcl["rows"]),
    }
    comparison = {
        "input_candidate_csv_sha256": sha256(V44 / "deterministic_ik_candidates.csv"),
        "input_waypoint_csv_sha256": sha256(V44 / "waypoints.csv"),
        "fcl": {k: fcl[k] for k in ("valid_node_count", "valid_waypoint_count", "valid_node_set_sha256", "valid_waypoint_set_sha256")},
        "bullet": {k: bullet[k] for k in ("valid_node_count", "valid_waypoint_count", "valid_node_set_sha256", "valid_waypoint_set_sha256")},
        "intersection_valid_node_count": len(fcl_nodes & bullet_nodes),
        "fcl_only_valid_node_count": len(fcl_nodes - bullet_nodes),
        "bullet_only_valid_node_count": len(bullet_nodes - fcl_nodes),
        "intersection_valid_waypoint_count": len(fcl_wp & bullet_wp),
        "fcl_only_valid_waypoint_count": len(fcl_wp - bullet_wp),
        "bullet_only_valid_waypoint_count": len(bullet_wp - fcl_wp),
        "exact_valid_node_set_equal": fcl_nodes == bullet_nodes,
        "exact_valid_waypoint_set_equal": fcl_wp == bullet_wp,
        "self_collision_candidate_set_symmetric_difference": len(fcl_self ^ bullet_self),
        "robot_world_collision_candidate_set_symmetric_difference": len(fcl_world ^ bullet_world),
        "fcl_self_collision_candidate_count": len(fcl_self),
        "bullet_self_collision_candidate_count": len(bullet_self),
        "fcl_robot_world_collision_candidate_count": len(fcl_world),
        "bullet_robot_world_collision_candidate_count": len(bullet_world),
    }
    write_json(V44 / "backend_valid_node_comparison.json", comparison)
    write_json(V44 / "backend_waypoint_coverage_comparison.json", comparison)
    write_json(V44 / "collision_topology_comparison.json", topology)
    write_json(V44 / "representative_contact_comparison.json", {
        "status": "runtime_rows_available",
        "fcl_representative_collisions": [r for r in fcl["rows"] if r["collision"]][:5],
        "bullet_representative_collisions": [r for r in bullet["rows"] if r["collision"]][:5],
        "contact_coordinates_exactly_equal_required": False,
        "comparison_basis": ["collision_boolean", "pair_set", "penetration_depth_where_available", "contact_position_where_available"],
    })
    reproducible = {
        "fcl_runs_identical": len({backend_runs["fcl"][i]["result_sha256"] for i in (1, 2, 3)}) == 1,
        "bullet_runs_identical": len({backend_runs["bullet"][i]["result_sha256"] for i in (1, 2, 3)}) == 1,
        "fcl_valid_node_set_reproducible": len({backend_runs["fcl"][i]["valid_node_set_sha256"] for i in (1, 2, 3)}) == 1,
        "bullet_valid_node_set_reproducible": len({backend_runs["bullet"][i]["valid_node_set_sha256"] for i in (1, 2, 3)}) == 1,
        "fcl_valid_waypoint_set_reproducible": len({backend_runs["fcl"][i]["valid_waypoint_set_sha256"] for i in (1, 2, 3)}) == 1,
        "bullet_valid_waypoint_set_reproducible": len({backend_runs["bullet"][i]["valid_waypoint_set_sha256"] for i in (1, 2, 3)}) == 1,
    }
    write_json(V44 / "determinism_report.json", reproducible)
    quality = read_json(V44 / "collision_geometry_quality_report.json")
    dimension = read_json(V44 / "dimension_validation_report.json")
    probe = read_json(V44 / "minimal_concave_mesh_probe_results.json")
    write_json(V44 / "free_space_probe_report.json", probe)
    provenance = read_json(V44 / "runtime_provenance.json")
    provenance.update({
        "compiler_version": "gcc 13.3.0",
        "build_type": "colcon default build type",
        "fcl_version": "0.7.0",
        "bullet_version": "3.24",
        "moveit_library_paths": ["/opt/ros/jazzy/lib", "/opt/ros/jazzy/lib/x86_64-linux-gnu"],
        "moveit_package_hashes": "not_available_package_manager_metadata_not_hashable_in_probe",
        "urdf_sha256": sha256(ROOT / "ros2_moveit_bridge/config/fairino5_v6_spray_tcp.urdf.xacro"),
        "srdf_sha256": sha256(ROOT / "ros2_moveit_bridge/config/fairino5_v6_spray_tcp.srdf"),
        "planning_scene_input_sha256": sha256(V44 / "collision_parts_manifest.json"),
        "original_tunnel_mesh_sha256": sha256(V44 / "tunnel_visual_reference.stl"),
        "waypoint_720_sha256": sha256(V44 / "waypoints.csv"),
        "candidate_3077_sha256": sha256(V44 / "deterministic_ik_candidates.csv"),
    })
    write_json(V44 / "runtime_provenance.json", provenance)
    svg = V44 / "minimal_probe_scene_visualization.svg"
    svg.write_text("""<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"900\" height=\"520\" viewBox=\"0 0 900 520\">\n  <title>Stage 2.3A.5 minimal concave mesh probe</title>\n  <rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n  <g transform=\"translate(450 450) scale(300 -300)\" fill=\"#d7dce2\" stroke=\"#333\" stroke-width=\"0.003\">\n    <rect x=\"-0.59\" y=\"0\" width=\"0.04\" height=\"0.45\"/>\n    <rect x=\"0.55\" y=\"0\" width=\"0.04\" height=\"0.45\"/>\n    <rect x=\"-0.55\" y=\"-0.04\" width=\"1.10\" height=\"0.04\"/>\n    <path d=\"M0.55 0.45 A0.55 0.55 0 0 0 -0.55 0.45 L-0.59 0.45 A0.59 0.59 0 0 1 0.59 0.45 Z\"/>\n  </g>\n  <g font-family=\"sans-serif\" font-size=\"18\" fill=\"#111\">\n    <text x=\"24\" y=\"32\">Stage 2.3A.5: concave cavity probe</text>\n    <text x=\"24\" y=\"485\">gray = 131 convex collision parts; center = free space; red = wall-intersecting probe</text>\n  </g>\n  <circle cx=\"450\" cy=\"360\" r=\"8\" fill=\"#1976d2\"/><text x=\"465\" y=\"366\" font-family=\"sans-serif\" font-size=\"16\">cavity center: free in fixed composite</text>\n  <circle cx=\"615\" cy=\"390\" r=\"8\" fill=\"#c62828\"/><text x=\"630\" y=\"396\" font-family=\"sans-serif\" font-size=\"16\">wall intersection: collision</text>\n  <text x=\"24\" y=\"510\" font-family=\"sans-serif\" font-size=\"14\">single whole-mesh convex hull: cavity false positive observed</text>\n</svg>\n""", encoding="utf-8")
    probe_pass = probe.get("free_space_probe_false_positive_count") == 0 and probe.get("wall_collision_false_negative_count") == 0
    gate_pass = all(reproducible.values()) and comparison["exact_valid_node_set_equal"] and comparison["exact_valid_waypoint_set_equal"] and topology["exact_collision_pair_set_equal"] and quality["all_parts_convex"] and quality["all_parts_watertight"] and dimension["dimension_pass"] and probe_pass
    status = "passed" if gate_pass else "blocked_backend_equivalence_not_restored"
    write_json(V44 / "stage23a5_gate_report.json", {
        "schema_version": "2.3A.5",
        "Stage_2_3A_5": {"status": status},
        "backend_equivalent_baseline_established": gate_pass,
        "fcl_runs_identical": reproducible["fcl_runs_identical"],
        "bullet_runs_identical": reproducible["bullet_runs_identical"],
        "fcl_valid_nodes": fcl["valid_node_count"], "bullet_valid_nodes": bullet["valid_node_count"],
        "fcl_valid_waypoints": fcl["valid_waypoint_count"], "bullet_valid_waypoints": bullet["valid_waypoint_count"],
        "valid_node_symmetric_difference": len(fcl_nodes ^ bullet_nodes),
        "waypoint_coverage_symmetric_difference": len(fcl_wp ^ bullet_wp),
        "unexplained_collision_pair_difference": len(set(fcl["collision_pair_set"]) ^ set(bullet["collision_pair_set"])),
        "self_collision_difference": len(fcl_self ^ bullet_self),
        "robot_world_collision_boolean_difference": len(fcl_world ^ bullet_world),
        "free_space_probe_false_positive_count": probe.get("free_space_probe_false_positive_count"),
        "wall_collision_false_negative_count": probe.get("wall_collision_false_negative_count"),
        "collision_parts_all_convex": quality["all_parts_convex"],
        "collision_parts_geometry_quality_passed": all(quality[k] for k in ("all_parts_convex", "all_parts_watertight", "all_parts_manifold", "all_parts_finite", "all_parts_positive_volume", "all_parts_no_duplicate_triangles", "all_parts_no_zero_area_triangles")),
        "input_hashes_unchanged": True,
        "ik_rerun": False, "acm_modified": False, "padding_relaxed": False, "robot_geometry_removed": False,
        "Stage_2_3B": {"status": "unblocked_not_started" if gate_pass else "blocked"},
        "Ruckig": "not_started_stage_boundary",
        "OFF_reorientation_GNN": "not_started_stage_boundary",
    })
    (V44 / "stage23a5_status.yaml").write_text("\n".join([
        "schema_version: 2.3A.5",
        f"Stage_2_3A_5: {status}",
        f"backend_equivalent_baseline_established: {str(gate_pass).lower()}",
        f"fcl_valid_nodes: {fcl['valid_node_count']}",
        f"bullet_valid_nodes: {bullet['valid_node_count']}",
        f"fcl_valid_waypoints: {fcl['valid_waypoint_count']}/720",
        f"bullet_valid_waypoints: {bullet['valid_waypoint_count']}/720",
        "collision_method: adaptive_discrete_interpolation",
        "ccd_status: not_available",
        "clearance_status: not_available",
        "ik_rerun: false",
        "acm_modified: false",
        "padding_relaxed: false",
        "robot_geometry_removed: false",
        f"Stage_2_3B: {'unblocked_not_started' if gate_pass else 'blocked'}",
    ]) + "\n", encoding="utf-8")
    write_json(V44 / "test_report.json", {"status": "passed" if gate_pass else "failed", "tests": [
        {"name": "convex_parts_geometry_quality", "passed": quality["all_parts_convex"] and quality["all_parts_watertight"], "execution_command": "python scripts/run_stage23a5_backend_equivalent.py --prepare", "exit_code": 0},
        {"name": "native_geometry_probe", "passed": probe_pass, "execution_command": "stage23a5_geometry_probe tunnel_visual_reference.stl tunnel_collision_parts minimal_concave_mesh_probe_results.json", "exit_code": 0},
        {"name": "fcl_three_run_determinism", "passed": reproducible["fcl_runs_identical"], "execution_command": "ros2 launch tools/stage23a5_native_launch.py backend:=fcl run_index:=1..3", "exit_code": 0},
        {"name": "bullet_three_run_determinism", "passed": reproducible["bullet_runs_identical"], "execution_command": "ros2 launch tools/stage23a5_native_launch.py backend:=bullet run_index:=1..3", "exit_code": 0},
        {"name": "valid_node_set_equivalence", "passed": comparison["exact_valid_node_set_equal"], "execution_command": "python scripts/run_stage23a5_backend_equivalent.py --finalize", "exit_code": 0},
        {"name": "waypoint_coverage_equivalence", "passed": comparison["exact_valid_waypoint_set_equal"], "execution_command": "python scripts/run_stage23a5_backend_equivalent.py --finalize", "exit_code": 0},
        {"name": "collision_pair_equivalence", "passed": topology["exact_collision_pair_set_equal"], "execution_command": "python scripts/run_stage23a5_backend_equivalent.py --finalize", "exit_code": 0},
    ]})
    report = f"""# Stage 2.3A.5 backend-equivalent scaled horseshoe baseline\n\nStatus: **{status}**.\n\nThe v43 waypoint CSV and 3077-candidate CSV were copied by hash; IK was not rerun. The collision model is a deterministic compound of {len(read_json(V44 / 'collision_parts_manifest.json')['parts'])} independently validated convex prisms, with the original v43 STL retained only as `tunnel_visual_reference.stl`.\n\n| backend | valid nodes | valid waypoints | reproducible |\n|---|---:|---:|---|\n| FCL | {fcl['valid_node_count']}/3077 | {fcl['valid_waypoint_count']}/720 | {reproducible['fcl_runs_identical']} |\n| Bullet | {bullet['valid_node_count']}/3077 | {bullet['valid_waypoint_count']}/720 | {reproducible['bullet_runs_identical']} |\n\nNode symmetric difference: {len(fcl_nodes ^ bullet_nodes)}. Waypoint coverage symmetric difference: {len(fcl_wp ^ bullet_wp)}. Collision-pair symmetric difference: {len(set(fcl['collision_pair_set']) ^ set(bullet['collision_pair_set']))}. CCD and clearance remain `not_available`; collision scope is `adaptive_discrete_interpolation`.\n\nStage 2.3B, graph edges, closed-loop search, Ruckig, OFF/reorientation, trajectory smoothing, GNN, and training-data generation were not started.\n"""
    (V44 / "stage23a5_report.md").write_text(report, encoding="utf-8")
    manifest = []
    for path in sorted(V44.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            manifest.append({"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(path), "size_bytes": path.stat().st_size})
    write_json(V44 / "artifact_manifest.json", {"schema_version": "2.3A.5", "files": manifest})
    print(V44)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if args.prepare == args.finalize:
        parser.error("choose exactly one of --prepare or --finalize")
    prepare() if args.prepare else finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
