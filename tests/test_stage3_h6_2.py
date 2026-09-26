from __future__ import annotations

from pathlib import Path

from scripts import stage3_h6_2 as h62


def test_h6_2_cosine_rejects_with_conservative_margin() -> None:
    assert h62.cosine([1.0, 0.0], [-1.0, 0.0]) == -1.0
    assert h62.ROUTE_COS_REJECT > h62.MOVEIT_COS_REJECT
    assert -1.0 <= h62.ROUTE_COS_REJECT < -0.999


def test_h6_2_path_enumeration_is_deterministic_and_complete_on_toy_graph() -> None:
    nodes = {
        1: {"surface_point_xyz_m": [0.0, 0.0, 0.0], "surface_normal_unit": [0.0, 0.0, 1.0], "source_triangle_id": 0, "source_barycentric_uvw": [1.0, 0.0, 0.0]},
        2: {"surface_point_xyz_m": [1.0, 0.0, 0.0], "surface_normal_unit": [0.0, 0.0, 1.0], "source_triangle_id": 0, "source_barycentric_uvw": [0.0, 1.0, 0.0]},
        3: {"surface_point_xyz_m": [2.0, 0.0, 0.0], "surface_normal_unit": [0.0, 0.0, 1.0], "source_triangle_id": 0, "source_barycentric_uvw": [0.0, 0.0, 1.0]},
    }
    adjacency = {1: {2}, 2: {1, 3}, 3: {2}}
    paths = h62.enumerate_hamiltonian_paths([3, 1, 2], adjacency, nodes)
    assert paths == [(1, 2, 3), (3, 2, 1)]


def test_h6_2_global_graph_rejects_multi_radian_edge() -> None:
    rows = [{"waypoint_index": 0}, {"waypoint_index": 1}]
    layers = {
        0: [{"candidate_id": "a", "waypoint_index": 0, "joint_values": [0.0] * 6, "branch_node_id": "branch_a"}],
        1: [{"candidate_id": "b", "waypoint_index": 1, "joint_values": [2.0] + [0.0] * 5, "branch_node_id": "branch_b"}],
    }
    result = h62.optimize_segment(rows, layers, [None] * 6)
    assert result["status"] == "BLOCKED"


def test_h6_2_input_identity_points_at_frozen_h6_and_h6_1() -> None:
    assert h62.H6_ROOT.name == "stage3_h6_surface_coverage_20260808T225000Z"
    assert h62.H61_ROOT.name == "stage3_h6_1_process_tolerance_recertification_20260808T161719Z"
    assert h62.COLLISION_METHOD == "adaptive_discrete_interpolation"
    assert h62.CCD == "NOT_AVAILABLE"
    assert h62.CLEARANCE == "NOT_AVAILABLE"
