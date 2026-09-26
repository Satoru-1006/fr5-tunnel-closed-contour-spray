from __future__ import annotations

import json
from pathlib import Path

from scripts import stage3_h6 as h6


def test_h3_orientation_maps_tcp_z_to_spray_direction() -> None:
    x_axis, y_axis, spray = h6.h3_orientation([0.6, 0.8, 0.0])
    assert spray == [-0.6, -0.8, -0.0]
    assert abs(h6.dot(x_axis, spray)) < 1e-12
    assert abs(h6.dot(y_axis, spray)) < 1e-12
    assert abs(h6.dot(x_axis, y_axis)) < 1e-12


def test_h2_barycentric_reconstruction_matches_h3_targets() -> None:
    targets = h6.load_jsonl(h6.H3_TARGETS)
    vertices, triangles = h6.read_obj(h6.H2_MESH)
    vertex_normals, _ = h6.mesh_normals(vertices, triangles)
    errors = []
    for row in targets[:64]:
        triangle = triangles[int(row["source_triangle_id"])]
        point = h6.bary_point(vertices, triangle, row["source_barycentric_uvw"])
        normal = h6.bary_normal(vertex_normals, triangle, row["source_barycentric_uvw"])
        errors.append(h6.vec_norm(h6.vec_sub(point, row["surface_point_xyz_m"])))
        assert h6.vec_norm(h6.vec_sub(normal, row["surface_normal_unit"])) < 1e-12
    assert max(errors) < 1e-12


def test_surface_triangle_route_stays_on_mesh() -> None:
    targets = h6.load_jsonl(h6.H3_TARGETS)
    vertices, triangles = h6.read_obj(h6.H2_MESH)
    adjacency = h6.mesh_triangle_adjacency(triangles)
    pieces = h6.surface_pieces(targets[1], targets[2], triangles, adjacency)
    assert pieces
    for triangle_id, first, last in pieces:
        assert abs(sum(first) - 1.0) < 1e-12
        assert abs(sum(last) - 1.0) < 1e-12
        assert all(value >= -1e-12 for value in first + last)
        assert h6.vec_norm(h6.vec_sub(h6.bary_point(vertices, triangles[triangle_id], first), h6.bary_point(vertices, triangles[triangle_id], first))) == 0.0


def test_preflight_finds_the_frozen_30_target_three_component_input() -> None:
    pre = h6.preflight(Path("outputs/stage3_h6_test_preflight"))
    assert pre["status"] == "PASSED"
    assert pre["target_count"] == 192
    assert len(pre["formal_target_ids"]) == 30
    assert pre["component_count"] == 3
    assert pre["selected_configuration"] == h6.EXPECTED_CONFIGURATION
    assert pre["tolerance_audit"]["no_threshold_invented"] is True


def test_component_path_covers_all_nodes_without_target_id_sorting() -> None:
    nodes = {
        10: {"surface_point_xyz_m": [0.0, 0.0, 0.0], "surface_normal_unit": [1.0, 0.0, 0.0], "source_triangle_id": 0, "source_barycentric_uvw": [1.0, 0.0, 0.0]},
        20: {"surface_point_xyz_m": [1.0, 0.0, 0.0], "surface_normal_unit": [1.0, 0.0, 0.0], "source_triangle_id": 0, "source_barycentric_uvw": [0.0, 1.0, 0.0]},
        30: {"surface_point_xyz_m": [2.0, 0.0, 0.0], "surface_normal_unit": [1.0, 0.0, 0.0], "source_triangle_id": 0, "source_barycentric_uvw": [0.0, 0.0, 1.0]},
    }
    path = h6.component_path([30, 10, 20], {10: {20}, 20: {10, 30}, 30: {20}}, nodes)
    assert path["hamiltonian_path_exists"] is True
    assert set(path["target_order"]) == {10, 20, 30}
    assert path["revisited_target_ids"] == []
