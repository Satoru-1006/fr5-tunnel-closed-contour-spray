"""Dynamic programming solver for the Stage 0/1 layered IK graph."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.ik_candidate_graph import GraphSolution, LayeredIKGraph


def solve_layered_graph(graph: LayeredIKGraph) -> GraphSolution:
    """Find the minimum-cost valid path through adjacent IK layers.

    The graph is intentionally solved as a DAG. Invalid nodes and invalid
    edges are retained for auditability but never enter the DP frontier.
    """

    layer_counts = graph.layer_candidate_counts()
    previous_reachable_counts: list[int] = []
    reachable: dict[str, float] = {}
    predecessor: dict[str, str | None] = {}
    candidate_by_id = {
        candidate.candidate_id: candidate
        for layer in graph.candidates.values()
        for candidate in layer
    }
    edge_by_source: dict[str, list[Any]] = {}
    for edges in graph.edges.values():
        for edge in edges:
            edge_by_source.setdefault(edge.from_candidate_id, []).append(edge)

    first_unreachable: int | None = None
    if graph.waypoint_count == 0:
        return GraphSolution(False, first_unreachable_waypoint=0, diagnostics={"layer_candidate_counts": []})

    first_layer = graph.layer(0)
    for candidate in first_layer:
        if candidate.valid:
            reachable[candidate.candidate_id] = 0.0
            predecessor[candidate.candidate_id] = None
    previous_reachable_counts.append(len(reachable))
    if not reachable:
        first_unreachable = 0

    for waypoint in range(1, graph.waypoint_count):
        if first_unreachable is not None:
            previous_reachable_counts.append(0)
            continue
        next_costs: dict[str, float] = {}
        next_predecessor: dict[str, str] = {}
        for edge in graph.edge_layer(waypoint - 1):
            if not edge.valid or edge.from_candidate_id not in reachable:
                continue
            target = candidate_by_id.get(edge.to_candidate_id)
            if target is None or not target.valid:
                continue
            edge_cost = float(edge.cost if edge.cost is not None else 0.0)
            proposed = reachable[edge.from_candidate_id] + edge_cost
            if proposed < next_costs.get(edge.to_candidate_id, float("inf")):
                next_costs[edge.to_candidate_id] = proposed
                next_predecessor[edge.to_candidate_id] = edge.from_candidate_id
        reachable = next_costs
        predecessor.update(next_predecessor)
        previous_reachable_counts.append(len(reachable))
        if not reachable:
            first_unreachable = waypoint

    diagnostics: dict[str, Any] = {
        "layer_candidate_counts": layer_counts,
        "previous_reachable_candidate_counts": previous_reachable_counts,
        "raw_transition_edge_count": graph.edge_count(False),
        "valid_transition_edge_count": graph.edge_count(True),
    }
    if first_unreachable is not None:
        diagnostics["search_status"] = "unreachable_layer"
        return GraphSolution(
            found=False,
            first_unreachable_waypoint=first_unreachable,
            diagnostics=diagnostics,
        )

    terminal_id, total_cost = min(reachable.items(), key=lambda item: item[1])
    path_ids: list[str] = []
    current: str | None = terminal_id
    while current is not None:
        path_ids.append(current)
        current = predecessor.get(current)
    path_ids.reverse()
    q_path = np.asarray([candidate_by_id[candidate_id].q_unwrapped_rad for candidate_id in path_ids], dtype=float)
    diagnostics["search_status"] = "complete_path"
    return GraphSolution(
        found=True,
        candidate_ids=path_ids,
        q_path_rad=q_path,
        total_cost=float(total_cost),
        diagnostics=diagnostics,
    )
