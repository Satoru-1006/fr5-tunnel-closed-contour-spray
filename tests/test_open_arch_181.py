from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from src.graph_search_solver import solve_layered_graph
from src.ik_candidate_graph import (
    GraphSolution,
    IKCandidate,
    LayeredIKGraph,
    TransitionEdge,
    circular_joint_delta,
    nearest_equivalent_joint_positions,
)


ROOT = Path(__file__).resolve().parents[1]
POSES = ROOT / "outputs" / "internal_wiper_moveit_inputs" / "open_arch_tcp_poses_base_link.csv"
SEEDS = ROOT / "outputs" / "internal_wiper_moveit_inputs" / "open_arch_seed_joints.csv"


def test_authoritative_open_arch_inputs_are_181_points() -> None:
    with POSES.open(newline="", encoding="utf-8") as handle:
        pose_rows = list(csv.DictReader(handle))
    with SEEDS.open(newline="", encoding="utf-8") as handle:
        seed_rows = list(csv.DictReader(handle))
    assert len(pose_rows) == 181
    assert len(seed_rows) == 181
    assert set(pose_rows[0]) == {"x", "y", "z", "qx", "qy", "qz", "qw", "nx", "ny", "nz"}
    assert set(seed_rows[0]) == {"q1", "q2", "q3", "q4", "q5", "q6"}
    assert np.isclose(float(pose_rows[0]["qw"]) ** 2 + sum(float(pose_rows[0][key]) ** 2 for key in ("qx", "qy", "qz")), 1.0)


def test_open_arch_input_is_not_closed_by_stage_one() -> None:
    with POSES.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    first = np.asarray([float(rows[0][key]) for key in ("x", "y", "z")])
    last = np.asarray([float(rows[-1][key]) for key in ("x", "y", "z")])
    assert np.linalg.norm(first - last) > 1e-3


def test_circular_delta_and_equivalent_unwrap() -> None:
    delta = circular_joint_delta(np.deg2rad([357.0, 145.0, -358.0]))
    assert np.allclose(np.rad2deg(delta), [-3.0, 145.0, 2.0])
    unwrapped = nearest_equivalent_joint_positions(np.deg2rad([-179.0, 10.0]), np.deg2rad([179.0, 10.0]))
    assert np.allclose(np.rad2deg(unwrapped), [181.0, 10.0])


def _candidate(waypoint: int, candidate_id: str, q: list[float], valid: bool = True) -> IKCandidate:
    return IKCandidate(
        waypoint=waypoint,
        candidate_id=candidate_id,
        q_rad=np.asarray(q, dtype=float),
        q_unwrapped_rad=np.asarray(q, dtype=float),
        position_error_m=0.001,
        normal_error_deg=1.0,
        node_collision=False,
        valid=valid,
        reject_reason=None if valid else "node_collision",
    )


def _edge(source: IKCandidate, target: IKCandidate, cost: float, valid: bool = True, reason: str | None = None) -> TransitionEdge:
    delta = target.q_unwrapped_rad - source.q_unwrapped_rad
    return TransitionEdge(
        from_waypoint=source.waypoint,
        to_waypoint=target.waypoint,
        from_candidate_id=source.candidate_id,
        to_candidate_id=target.candidate_id,
        delta_q_wrapped_rad=circular_joint_delta(delta),
        delta_q_unwrapped_rad=delta,
        max_joint_step_deg=float(np.rad2deg(np.max(np.abs(delta)))),
        interpolation_count=3,
        collision_status="pass" if valid else "fail",
        cost=cost,
        valid=valid,
        reject_reason=reason,
    )


def test_layered_graph_dp_returns_lowest_cost_path() -> None:
    graph = LayeredIKGraph(waypoint_count=3)
    layers = [
        [_candidate(0, "0:a", [0, 0, 0, 0, 0, 0])],
        [
            _candidate(1, "1:a", [0.1, 0, 0, 0, 0, 0]),
            _candidate(1, "1:b", [0.2, 0, 0, 0, 0, 0]),
        ],
        [_candidate(2, "2:a", [0.3, 0, 0, 0, 0, 0])],
    ]
    for layer in layers:
        for candidate in layer:
            graph.add_candidate(candidate)
    graph.add_edge(_edge(layers[0][0], layers[1][0], cost=1.0))
    graph.add_edge(_edge(layers[0][0], layers[1][1], cost=5.0))
    graph.add_edge(_edge(layers[1][0], layers[2][0], cost=1.0))
    graph.add_edge(_edge(layers[1][1], layers[2][0], cost=5.0))
    solution = solve_layered_graph(graph)
    assert isinstance(solution, GraphSolution)
    assert solution.found
    assert solution.candidate_ids == ["0:a", "1:a", "2:a"]


def test_layered_graph_reports_first_unreachable_waypoint() -> None:
    graph = LayeredIKGraph(waypoint_count=3)
    first = _candidate(0, "0:a", [0, 0, 0, 0, 0, 0])
    blocked = _candidate(1, "1:a", [0.1, 0, 0, 0, 0, 0])
    final = _candidate(2, "2:a", [0.2, 0, 0, 0, 0, 0])
    for candidate in (first, blocked, final):
        graph.add_candidate(candidate)
    graph.add_edge(_edge(first, blocked, cost=1.0, valid=False, reason="interpolated_collision"))
    solution = solve_layered_graph(graph)
    assert not solution.found
    assert solution.first_unreachable_waypoint == 1
    assert solution.diagnostics["layer_candidate_counts"] == [1, 1, 1]
    assert solution.diagnostics["previous_reachable_candidate_counts"] == [1, 0, 0]


def test_graph_records_do_not_encode_unavailable_clearance_as_zero() -> None:
    candidate = _candidate(0, "0:a", [0, 0, 0, 0, 0, 0])
    edge = _edge(candidate, _candidate(1, "1:a", [0, 0, 0, 0, 0, 0]), cost=0.0)
    assert candidate.to_record()["tcp_standoff_error_m"] is None
    assert edge.to_record()["min_clearance_m"] is None
    assert edge.to_record()["min_clearance_alpha"] is None
    assert edge.to_record()["ccd_status"] == "not_available"
