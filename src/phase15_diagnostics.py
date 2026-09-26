"""Pure offline helpers for Stage 1.5 IK graph diagnosis.

The functions in this module deliberately operate on records rather than on
ROS or MoveIt objects.  This keeps the evidence calculations deterministic and
also makes it possible to unit-test the distinction between all candidate
combinations and the candidates reachable by the Stage 1 DP frontier.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping

import numpy as np

from src.ik_candidate_graph import circular_joint_delta, nearest_equivalent_joint_positions


JOINT_NAMES = [f"j{i}" for i in range(1, 7)]


def _as_q(value: Iterable[float]) -> np.ndarray:
    return np.asarray(list(value), dtype=float).reshape(-1)


def _record_q(record: Mapping[str, Any], field: str = "q_rad") -> np.ndarray:
    return _as_q(record[field])


def reachable_frontier(
    node_records: Iterable[Mapping[str, Any]],
    edge_records: Iterable[Mapping[str, Any]],
    waypoint_count: int,
) -> list[set[str]]:
    """Return the valid DP frontier at every waypoint without changing input."""

    nodes = list(node_records)
    edges = list(edge_records)
    reachable: dict[str, float] = {
        str(row["candidate_id"]): 0.0
        for row in nodes
        if int(row["waypoint"]) == 0 and bool(row.get("valid", True))
    }
    frontiers: list[set[str]] = [set(reachable)]
    for waypoint in range(1, int(waypoint_count)):
        next_costs: dict[str, float] = {}
        for edge in edges:
            if int(edge["from_waypoint"]) != waypoint - 1:
                continue
            if not bool(edge.get("valid", False)):
                continue
            source = str(edge["from_candidate_id"])
            target = str(edge["to_candidate_id"])
            if source not in reachable:
                continue
            value = reachable[source] + float(edge.get("cost") or 0.0)
            if value < next_costs.get(target, float("inf")):
                next_costs[target] = value
        reachable = next_costs
        frontiers.append(set(reachable))
    return frontiers


def edge_delta_detail(
    source: Mapping[str, Any], target: Mapping[str, Any]
) -> dict[str, Any]:
    """Compute wrapped and nearest-equivalent deltas for one candidate pair."""

    source_q = _record_q(source)
    target_q = _record_q(target)
    wrapped = circular_joint_delta(target_q - source_q)
    nearest_target = nearest_equivalent_joint_positions(target_q, source_q)
    unwrapped = nearest_target - source_q
    abs_deg = np.rad2deg(np.abs(unwrapped))
    dominant = int(np.argmax(abs_deg))
    return {
        "source_candidate_id": str(source["candidate_id"]),
        "target_candidate_id": str(target["candidate_id"]),
        "wrapped_delta_rad": wrapped.tolist(),
        "wrapped_delta_deg": np.rad2deg(wrapped).tolist(),
        "nearest_equivalent_target_rad": nearest_target.tolist(),
        "nearest_equivalent_delta_rad": unwrapped.tolist(),
        "nearest_equivalent_delta_deg": np.rad2deg(unwrapped).tolist(),
        "absolute_joint_delta_deg": abs_deg.tolist(),
        "max_joint_step_deg": float(np.max(abs_deg)),
        "dominant_jump_joint": JOINT_NAMES[dominant],
        "dominant_jump_deg": float(abs_deg[dominant]),
    }


def analyze_bridge(
    node_records: Iterable[Mapping[str, Any]],
    edge_records: Iterable[Mapping[str, Any]],
    source_waypoint: int = 67,
    target_waypoint: int = 68,
    waypoint_count: int = 181,
) -> dict[str, Any]:
    """Diagnose a bridge, including the DP-reachable source frontier."""

    nodes = list(node_records)
    edges = list(edge_records)
    by_id = {str(row["candidate_id"]): row for row in nodes}
    source_layer = [row for row in nodes if int(row["waypoint"]) == source_waypoint]
    target_layer = [row for row in nodes if int(row["waypoint"]) == target_waypoint]
    frontier = reachable_frontier(nodes, edges, waypoint_count)
    reachable_sources = [row for row in source_layer if str(row["candidate_id"]) in frontier[source_waypoint]]
    bridge_edges = [
        row
        for row in edges
        if int(row["from_waypoint"]) == source_waypoint
        and int(row["to_waypoint"]) == target_waypoint
    ]
    combinations: list[dict[str, Any]] = []
    for source in source_layer:
        for target in target_layer:
            combinations.append(edge_delta_detail(source, target))
    reachable_combinations = [
        edge_delta_detail(source, target)
        for source in reachable_sources
        for target in target_layer
    ]
    edge_by_pair = {
        (str(row["from_candidate_id"]), str(row["to_candidate_id"])): row
        for row in bridge_edges
    }
    for row in combinations:
        edge = edge_by_pair.get((row["source_candidate_id"], row["target_candidate_id"]))
        row["formal_edge_valid"] = None if edge is None else bool(edge.get("valid", False))
        row["formal_edge_reject_reason"] = None if edge is None else edge.get("reject_reason")
        row["collision_status"] = None if edge is None else edge.get("collision_status")

    all_best = min(combinations, key=lambda row: row["max_joint_step_deg"]) if combinations else None
    reachable_best = (
        min(reachable_combinations, key=lambda row: row["max_joint_step_deg"])
        if reachable_combinations
        else None
    )
    return {
        "source_waypoint": int(source_waypoint),
        "target_waypoint": int(target_waypoint),
        "source_candidate_count": len(source_layer),
        "target_candidate_count": len(target_layer),
        "raw_candidate_combination_count": len(combinations),
        "reachable_source_candidate_count": len(reachable_sources),
        "reachable_source_candidate_ids": [str(row["candidate_id"]) for row in reachable_sources],
        "all_candidate_min_possible_max_joint_step_deg": None if all_best is None else all_best["max_joint_step_deg"],
        "all_candidate_min_pair": all_best,
        "reachable_frontier_min_possible_max_joint_step_deg": None
        if reachable_best is None
        else reachable_best["max_joint_step_deg"],
        "reachable_frontier_min_pair": reachable_best,
        "all_candidate_combinations": combinations,
        "reachable_frontier_combinations": reachable_combinations,
        "formal_bridge_edge_count": len(bridge_edges),
        "formal_bridge_valid_edge_count": sum(bool(row.get("valid", False)) for row in bridge_edges),
        "frontier_at_source": sorted(frontier[source_waypoint]),
        "frontier_at_target": sorted(frontier[target_waypoint]),
        "interpretation": {
            "input_path_pose_discontinuity": "not_assessed_without_pose_metrics",
            "candidate_coverage_shortage": bool(not source_layer or not target_layer),
            "periodic_angle_handling_error": False,
            "reachable_branch_break": bool(reachable_best and reachable_best["max_joint_step_deg"] > 20.0),
            "note": (
                "The minimum over all layer combinations can be small even when the DP-reachable source "
                "frontier has no <=20 degree edge. These are separate quantities."
            ),
        },
    }


def classify_attempt_failure(attempt: Mapping[str, Any], timeout_s: float) -> str:
    """Map one enriched search attempt to one primary diagnostic reason."""

    if not bool(attempt.get("ik_success", False)):
        elapsed = float(attempt.get("elapsed_s") or 0.0)
        if timeout_s > 0.0 and elapsed >= 0.95 * timeout_s:
            return "timeout"
        return "ik_solver_failure"
    reasons = list(attempt.get("reject_reasons") or [])
    if "joint_limit" in reasons:
        return "joint_limit_failure"
    for reason in (
        "position_error",
        "standoff_error",
        "normal_error",
        "node_collision",
    ):
        if reason in reasons:
            return reason
    if attempt.get("duplicate", False):
        return "deduplication"
    if attempt.get("candidate_limit_reached", False):
        return "candidate_limit_or_search_truncation"
    return "accepted"


def summarize_failure_attempts(
    attempts: Iterable[Mapping[str, Any]],
    waypoint: int,
    timeout_s: float,
    roll_sample_count: int,
    source_phase_id: str = "stage_0_1",
) -> dict[str, Any]:
    """Summarize every attempted seed/roll evaluation for one waypoint."""

    rows = list(attempts)
    reasons = Counter(classify_attempt_failure(row, timeout_s) for row in rows)
    successes = [row for row in rows if bool(row.get("ik_success", False))]
    unique_count = sum(bool(row.get("unique", False)) for row in rows)
    valid_count = sum(bool(row.get("valid", False)) for row in rows)

    def best(metric: str) -> dict[str, Any] | None:
        values = [row for row in successes if row.get(metric) is not None]
        return min(values, key=lambda row: float(row[metric])) if values else None

    best_position = best("position_error_m")
    best_standoff = min(
        (row for row in successes if row.get("standoff_error_m") is not None),
        key=lambda row: abs(float(row["standoff_error_m"])),
        default=None,
    )
    best_normal = best("normal_error_deg")
    closest_limit = min(
        (row for row in successes if row.get("joint_limit_margin_rad") is not None),
        key=lambda row: float(row["joint_limit_margin_rad"]),
        default=None,
    )
    elapsed = [float(row.get("elapsed_s") or 0.0) for row in rows]
    return {
        "source_phase_id": source_phase_id,
        "local_waypoint_id": int(waypoint),
        "csv_original_row_number": int(waypoint) + 2,
        "attempt_seed_count": len({str(row.get("seed_id")) for row in rows}),
        "roll_sample_count": int(roll_sample_count),
        "ik_success_count": sum(bool(row.get("ik_success", False)) for row in rows),
        "deduplicated_candidate_count": int(unique_count),
        "valid_candidate_count": int(valid_count),
        "failure_reason_counts": dict(reasons),
        "best_failed_position_error_m": None if best_position is None else float(best_position["position_error_m"]),
        "best_failed_standoff_error_m": None
        if best_standoff is None
        else float(abs(float(best_standoff["standoff_error_m"]))),
        "best_failed_normal_error_deg": None if best_normal is None else float(best_normal["normal_error_deg"]),
        "closest_joint_limit_joint": None if closest_limit is None else closest_limit.get("joint_limit_joint"),
        "closest_joint_limit_margin_rad": None
        if closest_limit is None
        else float(closest_limit["joint_limit_margin_rad"]),
        "search_elapsed_s": float(sum(elapsed)),
        "search_max_attempt_elapsed_s": float(max(elapsed, default=0.0)),
        "attempt_count": len(rows),
        "attempts": rows,
    }


def threshold_sweep(
    node_records: Iterable[Mapping[str, Any]],
    edge_records: Iterable[Mapping[str, Any]],
    waypoint_count: int,
    thresholds_deg: Iterable[float] = (20.0, 25.0, 30.0, 45.0, 60.0, 90.0, 180.0),
) -> list[dict[str, Any]]:
    """Sweep a joint-step threshold without mutating candidate or edge records.

    An edge whose collision status is ``not_checked`` is not promoted by a
    wider threshold.  The output therefore reports both the conservative
    known-collision result and the joint-step-only diagnostic result.
    """

    nodes = list(node_records)
    edges = list(edge_records)
    valid_nodes = {str(row["candidate_id"]) for row in nodes if bool(row.get("valid", False))}

    def solve(edge_predicate) -> tuple[bool, int | None, list[int], float, float]:
        reachable = {
            str(row["candidate_id"]): 0.0
            for row in nodes
            if int(row["waypoint"]) == 0 and bool(row.get("valid", False))
        }
        path_max_step = {candidate_id: 0.0 for candidate_id in reachable}
        counts = [len(reachable)]
        first_unreachable = None if reachable else 0
        for waypoint in range(1, waypoint_count):
            next_costs: dict[str, float] = {}
            next_max_step: dict[str, float] = {}
            for edge in edges:
                if int(edge["from_waypoint"]) != waypoint - 1 or not edge_predicate(edge):
                    continue
                source = str(edge["from_candidate_id"])
                target = str(edge["to_candidate_id"])
                if source not in reachable or target not in valid_nodes:
                    continue
                step = float(edge.get("max_joint_step_deg") or 0.0)
                cost = reachable[source] + float(edge.get("cost") or step)
                if cost < next_costs.get(target, float("inf")):
                    next_costs[target] = cost
                    next_max_step[target] = max(path_max_step[source], step)
            reachable = next_costs
            path_max_step = next_max_step
            counts.append(len(reachable))
            if not reachable and first_unreachable is None:
                first_unreachable = waypoint
        if not reachable:
            return False, first_unreachable, counts, 0.0, 0.0
        terminal_id = min(reachable, key=reachable.get)
        return True, first_unreachable, counts, float(path_max_step[terminal_id]), float(reachable[terminal_id])

    results: list[dict[str, Any]] = []
    for threshold in thresholds_deg:
        limit = float(threshold)
        known_edge_predicate = lambda edge: bool(edge.get("valid", False)) and float(
            edge.get("max_joint_step_deg") or 0.0
        ) <= limit
        diagnostic_edge_predicate = lambda edge: str(edge.get("reject_reason") or "") in {"", "joint_step_gate"} and float(
            edge.get("max_joint_step_deg") or 0.0
        ) <= limit and edge.get("collision_status") != "fail"
        known = solve(
            known_edge_predicate
        )
        diagnostic = solve(
            lambda edge: diagnostic_edge_predicate(edge)
            and bool(edge.get("from_candidate_id") in valid_nodes)
            and bool(edge.get("to_candidate_id") in valid_nodes)
        )
        known_max_edge_step = max(
            (float(edge.get("max_joint_step_deg") or 0.0) for edge in edges if known_edge_predicate(edge)),
            default=None,
        )
        diagnostic_max_edge_step = max(
            (
                float(edge.get("max_joint_step_deg") or 0.0)
                for edge in edges
                if diagnostic_edge_predicate(edge)
                and str(edge.get("from_candidate_id")) in valid_nodes
                and str(edge.get("to_candidate_id")) in valid_nodes
            ),
            default=None,
        )
        unknown_edge_count = sum(
            float(edge.get("max_joint_step_deg") or 0.0) <= limit
            and edge.get("collision_status") == "not_checked"
            for edge in edges
        )
        results.append(
            {
                "threshold_deg": limit,
                "diagnostic_only": bool(limit > 20.0),
                "known_collision_label_path_exists": known[0],
                "known_collision_label_first_unreachable_waypoint": known[1],
                "known_collision_label_valid_edge_count": sum(
                    bool(edge.get("valid", False))
                    and float(edge.get("max_joint_step_deg") or 0.0) <= limit
                    for edge in edges
                ),
                "known_collision_label_max_joint_step_deg": known_max_edge_step,
                "known_collision_label_total_joint_motion_deg": known[4] if known[0] else None,
                "joint_step_only_diagnostic_path_exists": diagnostic[0],
                "joint_step_only_diagnostic_first_unreachable_waypoint": diagnostic[1],
                "joint_step_only_diagnostic_valid_edge_count": sum(
                    str(edge.get("reject_reason") or "") in {"", "joint_step_gate"}
                    and float(edge.get("max_joint_step_deg") or 0.0) <= limit
                    and edge.get("collision_status") != "fail"
                    for edge in edges
                ),
                "joint_step_only_diagnostic_max_joint_step_deg": diagnostic_max_edge_step,
                "joint_step_only_diagnostic_total_joint_motion_deg": diagnostic[4] if diagnostic[0] else None,
                "edges_with_unavailable_collision_label_below_threshold": int(unknown_edge_count),
                "collision_label_coverage_sufficient_for_formal_claim": bool(unknown_edge_count == 0),
            }
        )
    return results


def candidate_count_rows(
    node_records: Iterable[Mapping[str, Any]],
    waypoint_count: int,
    frontiers: list[set[str]] | None = None,
) -> list[dict[str, Any]]:
    nodes = list(node_records)
    by_wp: defaultdict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in nodes:
        by_wp[int(row["waypoint"])].append(row)
    rows: list[dict[str, Any]] = []
    for waypoint in range(waypoint_count):
        layer = by_wp[waypoint]
        rows.append(
            {
                "source_phase_id": "stage_0_1",
                "local_waypoint_id": waypoint,
                "csv_original_row_number": waypoint + 2,
                "candidate_count": len(layer),
                "valid_candidate_count": sum(bool(row.get("valid", False)) for row in layer),
                "dp_reachable_candidate_count": None if frontiers is None else len(frontiers[waypoint]),
            }
        )
    return rows
