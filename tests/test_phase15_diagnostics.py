from __future__ import annotations

import copy

import numpy as np

from src.ik_candidate_graph import circular_joint_delta
from src.phase15_diagnostics import (
    analyze_bridge,
    classify_attempt_failure,
    summarize_failure_attempts,
    threshold_sweep,
)


def _node(waypoint: int, candidate_id: str, q: list[float], valid: bool = True) -> dict[str, object]:
    return {
        "waypoint": waypoint,
        "candidate_id": candidate_id,
        "q_rad": q,
        "q_unwrapped_rad": q,
        "valid": valid,
        "node_collision": False,
    }


def _edge(source: dict[str, object], target: dict[str, object], valid: bool, reason: str | None = None) -> dict[str, object]:
    delta = np.asarray(target["q_rad"], dtype=float) - np.asarray(source["q_rad"], dtype=float)
    step = float(np.rad2deg(np.max(np.abs(delta))))
    return {
        "from_waypoint": source["waypoint"],
        "to_waypoint": target["waypoint"],
        "from_candidate_id": source["candidate_id"],
        "to_candidate_id": target["candidate_id"],
        "delta_q_wrapped_rad": circular_joint_delta(delta).tolist(),
        "delta_q_unwrapped_rad": delta.tolist(),
        "max_joint_step_deg": step,
        "cost": step,
        "valid": valid,
        "reject_reason": reason,
        "collision_status": "pass" if valid else "not_checked",
    }


def test_67_68_minimum_bottleneck_and_dominant_joint() -> None:
    source_a = _node(67, "67:a", [0, 0, 0, 0, 0, 0])
    source_b = _node(67, "67:b", [0, 0, 0, 0, 0, np.deg2rad(90)])
    target = _node(68, "68:a", [0, 0, 0, 0, 0, np.deg2rad(92)])
    result = analyze_bridge([_node(0, "0:a", [0, 0, 0, 0, 0, 0]), source_a, source_b, target], [], 67, 68, 69)
    assert result["raw_candidate_combination_count"] == 2
    assert np.isclose(result["all_candidate_min_possible_max_joint_step_deg"], 2.0)
    assert result["all_candidate_min_pair"]["dominant_jump_joint"] == "j6"


def test_threshold_sweep_does_not_modify_edges_and_marks_diagnostics() -> None:
    n0 = _node(0, "0:a", [0, 0, 0, 0, 0, 0])
    n1 = _node(1, "1:a", [np.deg2rad(25), 0, 0, 0, 0, 0])
    nodes = [n0, n1]
    edges = [_edge(n0, n1, False, "joint_step_gate")]
    before = copy.deepcopy(edges)
    result = threshold_sweep(nodes, edges, 2, (20, 25))
    assert edges == before
    assert result[0]["diagnostic_only"] is False
    assert result[1]["diagnostic_only"] is True
    assert result[1]["collision_label_coverage_sufficient_for_formal_claim"] is False
    assert result[1]["known_collision_label_path_exists"] is False


def test_failure_reason_classification_and_source_mapping() -> None:
    assert classify_attempt_failure({"ik_success": False, "elapsed_s": 0.01}, 0.2) == "ik_solver_failure"
    assert classify_attempt_failure({"ik_success": False, "elapsed_s": 0.2}, 0.2) == "timeout"
    assert classify_attempt_failure({"ik_success": True, "reject_reasons": ["normal_error"]}, 0.2) == "normal_error"
    summary = summarize_failure_attempts(
        [
            {"seed_id": "s0", "ik_success": False, "elapsed_s": 0.01},
            {"seed_id": "s1", "ik_success": True, "elapsed_s": 0.01, "reject_reasons": ["normal_error"], "normal_error_deg": 12.0, "position_error_m": 0.001, "standoff_error_m": 0.001},
        ],
        87,
        0.2,
        36,
        source_phase_id="stage_1_5_local_enrichment",
    )
    assert summary["source_phase_id"] == "stage_1_5_local_enrichment"
    assert summary["local_waypoint_id"] == 87
    assert summary["csv_original_row_number"] == 89
    assert summary["failure_reason_counts"]["ik_solver_failure"] == 1
    assert summary["failure_reason_counts"]["normal_error"] == 1
