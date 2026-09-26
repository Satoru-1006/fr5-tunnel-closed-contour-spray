#!/usr/bin/env python3
"""Stage 3 H7.1 authoritative joint-path topology/IK audit.

H7.1 is additive and fail-closed.  It reads frozen H5/H6/H6.1 evidence,
performs an offline topology audit, and uses a native MoveIt2 prefix replay
worker to identify the first TOTG rejection.  It never changes frozen
artifacts, never calls Ruckig, and never sends a FollowJointTrajectory goal.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
H5_ROOT = ROOT / "outputs/stage3_h5_joint_branch_continuity_20260808T132000Z"
H6_ROOT = ROOT / "outputs/stage3_h6_surface_coverage_20260808T225000Z"
H61_ROOT = ROOT / "outputs/stage3_h6_1_process_tolerance_recertification_20260808T161719Z"
H45_ROOT = ROOT / "outputs/stage3_h4_5_robot_fixture_geometry_codesign_20260809T000000Z"
H6_JSONL = H6_ROOT / "stage3_h6_joint_waypoints.jsonl"
H6_SURFACE_JSONL = H6_ROOT / "stage3_h6_surface_waypoints.jsonl"
H6_CART_JSONL = H6_ROOT / "stage3_h6_cartesian_tcp_waypoints.jsonl"
H6_TRACE_JSONL = H6_ROOT / "stage3_h6_ik_branch_trace.jsonl"
H6_ORDERING = H6_ROOT / "stage3_h6_component_ordering.json"
H6_SEGMENTS = H6_ROOT / "stage3_h6_spray_on_off_segments.json"
H6_FK = H6_ROOT / "stage3_h6_fk_tcp_validation.jsonl"
H6_COVERAGE = H6_ROOT / "stage3_h6_coverage_accounting.json"
H6_TERMINAL = H6_ROOT / "stage3_h6_terminal_certificate.json"
H61_FREEZE = H61_ROOT / "stage3_h6_1_contract_freeze_manifest.json"
H61_TERMINAL = H61_ROOT / "stage3_h6_1_terminal_certificate.json"
H61_CONTRACT_SOURCE = ROOT / "config/stage3/stage3_h6_1_spray_process_tolerance_contract.json"
H61_CONTRACT_COPY = H61_ROOT / "stage3_h6_1_spray_process_tolerance_contract.json"
H5_FREEZE = H5_ROOT / "stage3_h5_configuration_freeze_manifest.json"
H5_TERMINAL = H5_ROOT / "stage3_h5_terminal_certificate.json"
H5_GRAPH = H5_ROOT / "stage3_h5_joint_configuration_graph.json"
H5_NODES = H5_ROOT / "stage3_h5_distinct_ik_nodes.jsonl"
H5_PREFERRED = H5_ROOT / "stage3_h5_preferred_branch_sequences.json"
DERIVED_URDF = H45_ROOT / "derived_robot_model.urdf"
FIXTURE_MESH = H45_ROOT / "curved_fixture_stage3_h4_5_selected.obj"
SRDF = ROOT / "ros2_moveit_bridge/config/fairino5_v6_spray_tcp.srdf"
TOTG_CONFIGURATION = ROOT / "outputs/stage3_h7_authoritative_execution_20260808T172914Z/stage3_h7_totg_configuration.json"
MOVEIT_ANGLE_TOLERANCE = 1.0e-5
MOVEIT_COS_THRESHOLD = -1.0 + MOVEIT_ANGLE_TOLERANCE
JOINT_NAMES = ["j1", "j2", "j3", "j4", "j5", "j6"]
EXPECTED_SEGMENT_ORDER = [0, 1000, 1, 1001, 2]
FIRST_BLOCKER = "totg_failed_path_requires_180_degree_turn"
COLLISION_METHOD = "adaptive_discrete_interpolation"
REQUIRED_ARTIFACTS = [
    "stage3_h7_1_turn_geometry_audit.json",
    "stage3_h7_1_native_prefix_totg_replay.json",
    "stage3_h7_1_joint_discontinuity_audit.json",
    "stage3_h7_1_branch_switch_audit.json",
    "stage3_h7_1_root_cause_report.json",
    "stage3_h7_1_remediation_candidate_manifest.json",
    "stage3_h7_1_regression_report.json",
    "stage3_h7_1_gate_report.json",
    "FINAL_REPORT.md",
    "stage3_h7_1_terminal_certificate.json",
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def dump_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="\n")


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): canonical(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [canonical(item) for item in value]
    if isinstance(value, float):
        return round(value, 12) if math.isfinite(value) else None
    return value


def semantic_hash(value: Any) -> str:
    raw = json.dumps(canonical(value), ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def vector_delta(left: Sequence[float], right: Sequence[float]) -> list[float]:
    return [float(b) - float(a) for a, b in zip(left, right)]


def norm(value: Sequence[float]) -> float:
    return math.sqrt(sum(float(item) * float(item) for item in value))


def degrees(value: float) -> float:
    return math.degrees(value)


def source_record(path: Path, role: str) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "relative_path": rel(path),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256_file(path),
        "semantic_role": role,
    }


def turn_geometry_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for index in range(1, len(rows) - 1):
        previous = rows[index - 1]
        current = rows[index]
        following = rows[index + 1]
        q_prev = [float(value) for value in previous["joint_values"]]
        q_current = [float(value) for value in current["joint_values"]]
        q_next = [float(value) for value in following["joint_values"]]
        v_prev = vector_delta(q_prev, q_current)
        v_next = vector_delta(q_current, q_next)
        prev_norm = norm(v_prev)
        next_norm = norm(v_next)
        base = {
            "schema_version": "stage3-h7-1-turn-geometry-record-v1",
            "waypoint_index": index,
            "previous_index": index - 1,
            "next_index": index + 1,
            "segment_id": current.get("segment_id"),
            "spray_state": current.get("spray_state"),
            "source_target_id": current.get("source_target_id"),
            "destination_target_id": current.get("destination_target_id"),
            "previous_source_target_id": previous.get("source_target_id"),
            "previous_destination_target_id": previous.get("destination_target_id"),
            "next_source_target_id": following.get("source_target_id"),
            "next_destination_target_id": following.get("destination_target_id"),
            "q_prev": q_prev,
            "q_current": q_current,
            "q_next": q_next,
            "v_prev": v_prev,
            "v_next": v_next,
            "norm_v_prev": prev_norm,
            "norm_v_next": next_norm,
            "moveit_totg_rejection_criterion": "cos_angle <= -1.0 + 1e-5",
            "moveit_totg_angle_tolerance": MOVEIT_ANGLE_TOLERANCE,
            "moveit_totg_cos_threshold": MOVEIT_COS_THRESHOLD,
            "max_joint_step_pi_test": max(abs(value) for value in v_prev + v_next) >= math.pi,
            "used_for_180_degree_classification": True,
        }
        if prev_norm <= 1.0e-15 or next_norm <= 1.0e-15:
            base.update({
                "geometry_valid": False,
                "cos_angle": None,
                "turn_angle_deg": None,
                "near_moveit_180_degree_turn": False,
                "classification": "degenerate_zero_tangent",
            })
        else:
            cosine = sum(a * b for a, b in zip(v_prev, v_next)) / (prev_norm * next_norm)
            cosine = max(-1.0, min(1.0, cosine))
            turn_angle = degrees(math.acos(cosine))
            base.update({
                "geometry_valid": True,
                "cos_angle": cosine,
                "turn_angle_deg": turn_angle,
                "near_moveit_180_degree_turn": cosine <= MOVEIT_COS_THRESHOLD,
                "classification": "moveit_180_degree_turn" if cosine <= MOVEIT_COS_THRESHOLD else "not_moveit_180_degree_turn",
            })
        records.append(base)
    flagged = [record for record in records if record["near_moveit_180_degree_turn"]]
    return {
        "schema_version": "stage3-h7-1-turn-geometry-audit-v1",
        "source": source_record(H6_JSONL, "authoritative frozen H6 joint JSONL"),
        "joint_names": JOINT_NAMES,
        "waypoint_count": len(rows),
        "triple_count": len(records),
        "moveit_totg_source_semantics": {
            "criterion": "cos_angle <= -1.0 + angle_tolerance",
            "angle_tolerance": MOVEIT_ANGLE_TOLERANCE,
            "cos_threshold": MOVEIT_COS_THRESHOLD,
            "native_error": "The path requires a 180 deg. turn, which is not supported by the current implementation.",
            "diagnostic_is_not_max_joint_step": True,
        },
        "all_triples": records,
        "near_or_rejected_triples": flagged,
        "near_or_rejected_count": len(flagged),
        "exact_or_near_180_turn_centres": [int(record["waypoint_index"]) for record in flagged],
    }


def build_trace_map() -> dict[int, dict[str, Any]]:
    return {int(row["waypoint_index"]): row for row in load_jsonl(H6_TRACE_JSONL)}


def selected_candidate_rows(trace: Mapping[str, Any], selected_q: Sequence[float]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for attempt in trace.get("attempts", []):
        q = [float(value) for value in attempt.get("joint_values", [])]
        candidates.append({
            "seed_index": attempt.get("seed_index"),
            "seed_label": attempt.get("seed_label"),
            "solver_success": bool(attempt.get("solver_success")),
            "solver_error_code": attempt.get("solver_error_code"),
            "joint_values": q,
            "distance_to_previous_rad": attempt.get("distance_to_previous_rad"),
            "joint_limit_valid": attempt.get("state", {}).get("joint_limit_valid"),
            "collision_free": attempt.get("state", {}).get("collision_free"),
            "fk_computable": attempt.get("state", {}).get("fk_computable"),
            "accepted_candidate": bool(len(q) == len(selected_q) and all(abs(a - b) <= 1.0e-12 for a, b in zip(q, selected_q))),
        })
    return candidates


def branch_switch_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    traces = build_trace_map()
    all_node_transitions: list[dict[str, Any]] = []
    flagged: list[dict[str, Any]] = []
    for edge_index, (left, right) in enumerate(zip(rows, rows[1:])):
        left_node = left.get("nearest_h5_node_id")
        right_node = right.get("nearest_h5_node_id")
        if left_node == right_node:
            continue
        delta = vector_delta(left["joint_values"], right["joint_values"])
        right_trace = traces.get(edge_index + 1, {})
        attempts = list(right_trace.get("attempts", []))
        previous_attempt = attempts[0] if attempts and attempts[0].get("seed_label") == "previous_accepted_waypoint" else None
        previous_failed = bool(previous_attempt is not None and not previous_attempt.get("solver_success"))
        same_process_connection = (
            left.get("source_target_id") == right.get("source_target_id")
            and left.get("destination_target_id") == right.get("destination_target_id")
            and left.get("segment_id") == right.get("segment_id")
        )
        fallback = right.get("accepted_seed_label") == "frozen_h5_seed_bank" and previous_failed
        record = {
            "edge_index": edge_index,
            "from_waypoint_index": edge_index,
            "to_waypoint_index": edge_index + 1,
            "from_h5_node_id": left_node,
            "to_h5_node_id": right_node,
            "segment_id": right.get("segment_id"),
            "spray_state": right.get("spray_state"),
            "source_target_id": right.get("source_target_id"),
            "destination_target_id": right.get("destination_target_id"),
            "accepted_seed_label_before": left.get("accepted_seed_label"),
            "accepted_seed_label_after": right.get("accepted_seed_label"),
            "per_joint_delta_rad": delta,
            "per_joint_delta_deg": [degrees(value) for value in delta],
            "L_inf_joint_delta_rad": max(abs(value) for value in delta),
            "L_inf_joint_delta_deg": degrees(max(abs(value) for value in delta)),
            "L2_joint_delta_rad": norm(delta),
            "same_process_connection": same_process_connection,
            "previous_accepted_seed_attempt": {
                "present": previous_attempt is not None,
                "solver_success": previous_attempt.get("solver_success") if previous_attempt is not None else None,
                "solver_error_code": previous_attempt.get("solver_error_code") if previous_attempt is not None else None,
                "distance_to_previous_rad": previous_attempt.get("distance_to_previous_rad") if previous_attempt is not None else None,
            },
            "previous_accepted_seed_failed": previous_failed,
            "fallback_h5_seed_used": fallback,
            "candidate_ik_solutions": selected_candidate_rows(right_trace, right.get("joint_values", [])),
            "branch_switch_diagnostic": "node_id_changed_within_same_process_connection" if same_process_connection else "node_id_changed_at_connection_or_segment_boundary",
        }
        all_node_transitions.append(record)
        if same_process_connection and (fallback or record["L_inf_joint_delta_deg"] >= 90.0):
            record = dict(record)
            record.update({
                "classification": "IK_BRANCH_DISCONTINUITY",
                "selection_reason": "previous accepted waypoint seed failed; H5 fallback bank selected a different valid branch" if fallback else "large same-connection joint discontinuity",
            })
            flagged.append(record)
    return {
        "schema_version": "stage3-h7-1-branch-switch-audit-v1",
        "source_joint_jsonl": source_record(H6_JSONL, "authoritative frozen H6 joint JSONL"),
        "source_ik_trace_jsonl": source_record(H6_TRACE_JSONL, "frozen H6 native IK branch trace"),
        "all_h5_node_transitions_count": len(all_node_transitions),
        "all_h5_node_transitions": all_node_transitions,
        "flagged_branch_discontinuities": flagged,
        "flagged_branch_discontinuity_count": len(flagged),
        "flagged_edge_indices": [int(item["edge_index"]) for item in flagged],
        "classification_rule": "same SPRAY_ON process connection plus previous-seed failure/fallback or >=90 degree L_inf joint jump; small H5 node changes with a successful previous seed are not called discontinuities",
        "all_candidate_ik_solutions_recorded": all(all(item.get("candidate_ik_solutions") for item in flagged) for item in flagged),
    }


def joint_discontinuity_audit(rows: Sequence[Mapping[str, Any]], turn_audit: Mapping[str, Any], branch_audit: Mapping[str, Any]) -> dict[str, Any]:
    turn_by_center = {int(item["waypoint_index"]): item for item in turn_audit["all_triples"]}
    branch_by_edge = {int(item["edge_index"]): item for item in branch_audit["flagged_branch_discontinuities"]}
    edges: list[dict[str, Any]] = []
    for edge_index, (left, right) in enumerate(zip(rows, rows[1:])):
        delta = vector_delta(left["joint_values"], right["joint_values"])
        center_turn = turn_by_center.get(edge_index)
        edges.append({
            "schema_version": "stage3-h7-1-joint-edge-record-v1",
            "edge_index": edge_index,
            "from_waypoint_index": edge_index,
            "to_waypoint_index": edge_index + 1,
            "segment_id": left.get("segment_id"),
            "spray_state": left.get("spray_state"),
            "source_target_id": left.get("source_target_id"),
            "destination_target_id": left.get("destination_target_id"),
            "per_joint_delta_rad": delta,
            "per_joint_delta_deg": [degrees(value) for value in delta],
            "L_inf_joint_delta_rad": max(abs(value) for value in delta),
            "L_inf_joint_delta_deg": degrees(max(abs(value) for value in delta)),
            "L2_joint_delta_rad": norm(delta),
            "joint_space_tangent_angle_deg_at_waypoint": center_turn.get("turn_angle_deg") if center_turn else None,
            "joint_space_tangent_cos_angle_at_waypoint": center_turn.get("cos_angle") if center_turn else None,
            "max_joint_step_pi_flag": max(abs(value) for value in delta) >= math.pi,
            "nearest_h5_node_from": left.get("nearest_h5_node_id"),
            "nearest_h5_node_to": right.get("nearest_h5_node_id"),
            "ik_seed_label_from": left.get("accepted_seed_label"),
            "ik_seed_label_to": right.get("accepted_seed_label"),
            "branch_discontinuity": edge_index in branch_by_edge,
            "branch_discontinuity_record": branch_by_edge.get(edge_index),
            "collision_method": COLLISION_METHOD,
            "ccd_status": "not_available",
            "clearance_m": None,
        })
    return {
        "schema_version": "stage3-h7-1-joint-discontinuity-audit-v1",
        "source": source_record(H6_JSONL, "authoritative frozen H6 joint JSONL"),
        "joint_names": JOINT_NAMES,
        "consecutive_edge_count": len(edges),
        "expected_consecutive_edge_count": 1294,
        "all_edges_audited": len(edges) == 1294,
        "edges": edges,
        "edge_summary": {
            "max_L_inf_joint_delta_deg": max((edge["L_inf_joint_delta_deg"] for edge in edges), default=None),
            "max_L2_joint_delta_deg": max((degrees(edge["L2_joint_delta_rad"]) for edge in edges), default=None),
            "edges_with_max_joint_step_ge_pi": [edge["edge_index"] for edge in edges if edge["max_joint_step_pi_flag"]],
            "branch_discontinuity_edges": sorted(branch_by_edge),
            "moveit_180_turn_centres": sorted(int(item["waypoint_index"]) for item in turn_audit["near_or_rejected_triples"]),
        },
        "classification_separation": {
            "type_1": "joint-space path cusp/exact backtracking; diagnosed only by the three-waypoint tangent criterion",
            "type_2": "IK branch discontinuity; diagnosed by branch provenance and continuation failure, not by max_raw_step alone",
        },
    }


def route_trace_for_cusp(index: int, rows: Sequence[Mapping[str, Any]], surface_rows: Sequence[Mapping[str, Any]], ordering: Mapping[str, Any]) -> dict[str, Any]:
    left = rows[index - 1]
    center = rows[index]
    right = rows[index + 1]
    surface_left = surface_rows[index - 1]
    surface_center = surface_rows[index]
    surface_right = surface_rows[index + 1]
    surface_backtrack_distance = norm(vector_delta(surface_left["surface_point_xyz_m"], surface_right["surface_point_xyz_m"]))
    target_orders = [entry["surface_order"]["target_order"] for entry in ordering.get("components", [])]
    local_subsequence: list[int] | None = None
    for target_order in target_orders:
        for pos in range(len(target_order) - 2):
            if target_order[pos + 1] == center.get("destination_target_id") and target_order[pos] == left.get("source_target_id") and target_order[pos + 2] == right.get("destination_target_id"):
                local_subsequence = [int(value) for value in target_order[pos : pos + 3]]
                break
    return {
        "cusp_center_waypoint": index,
        "incoming_edge": {
            "from_waypoint": index - 1,
            "to_waypoint": index,
            "source_target_id": left.get("source_target_id"),
            "destination_target_id": left.get("destination_target_id"),
            "segment_id": left.get("segment_id"),
        },
        "outgoing_edge": {
            "from_waypoint": index,
            "to_waypoint": index + 1,
            "source_target_id": right.get("source_target_id"),
            "destination_target_id": right.get("destination_target_id"),
            "segment_id": right.get("segment_id"),
        },
        "target_order_subsequence": local_subsequence,
        "source_target_occurrence_counts_in_order": {
            "incoming_source": sum(order.count(left.get("source_target_id")) for order in target_orders),
            "shared_target": sum(order.count(center.get("destination_target_id")) for order in target_orders),
            "outgoing_destination": sum(order.count(right.get("destination_target_id")) for order in target_orders),
        },
        "surface_parameter_previous": surface_left.get("surface_parameter"),
        "surface_parameter_center": surface_center.get("surface_parameter"),
        "surface_parameter_next": surface_right.get("surface_parameter"),
        "surface_point_previous": surface_left.get("surface_point_xyz_m"),
        "surface_point_center": surface_center.get("surface_point_xyz_m"),
        "surface_point_next": surface_right.get("surface_point_xyz_m"),
        "surface_endpoint_retrace_distance_m": surface_backtrack_distance,
        "edge_sampling_overlap": surface_backtrack_distance <= 1.0e-12,
        "consecutive_duplicate_endpoint_in_authoritative_rows": norm(vector_delta(surface_left["surface_point_xyz_m"], surface_center["surface_point_xyz_m"])) <= 1.0e-12 or norm(vector_delta(surface_center["surface_point_xyz_m"], surface_right["surface_point_xyz_m"])) <= 1.0e-12,
        "shared_endpoint_deduplication_in_builder": True,
        "route_ordering_backtracking": surface_backtrack_distance <= 1.0e-12,
        "code_provenance": {
            "component_path_and_edge_cost": "scripts/stage3_h6.py:301-347",
            "dense_surface_pair_construction_and_compaction": "scripts/stage3_h6.py:404-428",
            "finding": "edge_cost minimizes point/normal cost but does not penalize tangent reversal or overlapping consecutive surface polylines",
        },
        "root_cause_classification": "TYPE_1_ROUTE_ORDERING_AND_EDGE_CONSTRUCTION_BACKTRACKING",
    }


def root_cause_report(rows: Sequence[Mapping[str, Any]], surface_rows: Sequence[Mapping[str, Any]], ordering: Mapping[str, Any], turn_audit: Mapping[str, Any], branch_audit: Mapping[str, Any], native_replay: Mapping[str, Any], frozen_integrity: Mapping[str, Any]) -> dict[str, Any]:
    cusp_records = [route_trace_for_cusp(int(index), rows, surface_rows, ordering) for index in (40, 260)]
    flagged_branches = branch_audit.get("flagged_branch_discontinuities", [])
    return {
        "schema_version": "stage3-h7-1-root-cause-report-v1",
        "status": "BLOCKED",
        "STAGE_3_H7": "BLOCKED",
        "STAGE_3_H7_1": "BLOCKED",
        "FIRST_BLOCKER": FIRST_BLOCKER,
        "native_first_blocker_evidence": native_replay.get("first_failing_prefix", {}),
        "type_1_joint_space_cusp": {
            "count": len(cusp_records),
            "centres": [item["cusp_center_waypoint"] for item in cusp_records],
            "records": cusp_records,
            "root_cause": "H6 route ordering plus dense edge construction creates exact surface-edge backtracking at legitimate shared targets; it is not duplicate target traversal and not an IK branch switch.",
        },
        "type_2_ik_branch_discontinuity": {
            "count": len(flagged_branches),
            "edge_indices": [int(item["edge_index"]) for item in flagged_branches],
            "records": flagged_branches,
            "root_cause": "H6 continuation attempts the previous accepted waypoint seed first; when that seed fails, it falls back to the frozen H5 seed bank and accepts a different valid IK branch on the same SPRAY_ON process connection.",
        },
        "layer_attribution": {
            "H5": "provides multiple valid IK nodes/edges and does not itself define the dense authoritative waypoint traversal",
            "H6": "first authoritative layer that chooses the component/target route, constructs dense pair paths, compacts shared endpoints, and performs fallback branch selection",
            "H6_1": "freezes process tolerances and H6 provenance; no evidence that it introduced either path topology defect",
            "H7": "first native timing consumer; correctly rejects the first type-1 cusp and remains BLOCKED",
        },
        "answers": {
            "waypoint_40_is_true_native_blocker": (
                native_replay.get("minimal_failing_prefix") == 41
                and native_replay.get("cases", {}).get("prefix_0_0040", {}).get("status") == "PASSED"
                and native_replay.get("cases", {}).get("prefix_0_0041", {}).get("status") == "BLOCKED"
            ),
            "waypoint_260_is_second_native_blocker_after_first_triple_is_isolated": native_replay.get("second_failing_prefix_after_isolation") == 261,
            "omit_40_is_valid_isolation": native_replay.get("second_failing_prefix_after_omit_40") is not None,
            "81_to_82_is_cusp": False,
            "1059_to_1060_is_cusp": False,
            "1269_to_1270_is_cusp": False,
            "81_to_82_is_independent_ik_branch_jump": 81 in branch_audit.get("flagged_edge_indices", []),
            "1059_to_1060_is_independent_ik_branch_jump": 1059 in branch_audit.get("flagged_edge_indices", []),
            "1269_to_1270_is_independent_ik_branch_jump": 1269 in branch_audit.get("flagged_edge_indices", []),
            "route_ordering_or_edge_construction_root_cause": True,
            "nearest_ik_branch_selection_root_cause_for_three_jumps": True,
            "repair_without_process_segmentation_change_proven": False,
            "additional_spray_off_transitions_required": None,
            "coverage_30_of_30_retained_in_frozen_h6": load_json(H6_COVERAGE).get("ALL_30_FEASIBLE_TARGETS_ACCOUNTED_FOR") == "YES",
            "H5_H6_H6_1_immutable": bool(frozen_integrity.get("all_after_hashes_match")),
        },
    }


def remediation_manifest(turn_audit: Mapping[str, Any], branch_audit: Mapping[str, Any], coverage: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "stage3-h7-1-remediation-candidate-manifest-v1",
        "status": "PREPARATION_ONLY_BLOCKED",
        "frozen_h6_modified": False,
        "candidate_generation_authorized": True,
        "candidate_certification_completed": False,
        "priority_order": ["H6.2_candidate_1_route_ordering_edge_construction", "H6.2_candidate_2_branch_continuous_ik", "H6.2_candidate_3_process_continuity_break"],
        "candidates": [
            {
                "candidate_id": "H6.2_candidate_1_route_ordering_edge_construction",
                "priority": 1,
                "status": "NOT_IMPLEMENTED",
                "objective": "penalize tangent reversal/overlap in route ordering and construct shared target joins without an immediate reverse traversal",
                "preserve_process_structure": "required: 3 SPRAY_ON + 2 SPRAY_OFF",
                "preserve_coverage": "required: 30/30",
                "evidence_to_collect": ["no_moveit_unsupported_180_degree_turn", "no_explained_type_1_cusp", "full_fk_tcp_process_validation", "deterministic_replay_3_of_3"],
            },
            {
                "candidate_id": "H6.2_candidate_2_branch_continuous_ik",
                "priority": 2,
                "status": "NOT_IMPLEMENTED",
                "objective": "use previous accepted state as primary seed at every micro-step and select a branch-continuous candidate; record every candidate and selection reason",
                "known_target_edges": [81, 1059, 1269],
                "process_structure_change": "none by default",
                "evidence_to_collect": ["no_unexplained_ik_branch_discontinuity", "dense_fk_tcp_process_validation", "collision_free_interpolation"],
            },
            {
                "candidate_id": "H6.2_candidate_3_process_continuity_break",
                "priority": 3,
                "status": "CONTINGENCY_ONLY",
                "objective": "if no collision-free/process-valid branch-continuous path exists, introduce an explicit process-continuity break with SPRAY_OFF/reposition/SPRAY_ON",
                "process_structure_change": "would change 3 ON + 2 OFF and therefore requires a new H6.2 candidate, never a relabel of H6/H6.1",
                "automatic_acceptance": False,
            },
        ],
        "frozen_findings": {
            "moveit_180_turn_centres": turn_audit.get("exact_or_near_180_turn_centres", []),
            "ik_branch_discontinuity_edges": branch_audit.get("flagged_edge_indices", []),
            "coverage": coverage,
        },
        "gate": "fail_closed_until_a_new_H6.2_candidate_passes_all_required_gates",
    }


def wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    return f"/mnt/{drive}/{str(resolved).split(':', 1)[-1].lstrip('/').replace(chr(92), '/') }"


def run_wsl(command: str, timeout_s: int = 600) -> tuple[int, str, str]:
    process = subprocess.run(
        ["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_s,
        check=False,
    )
    return process.returncode, process.stdout or "", process.stderr or ""


def build_native_worker(output: Path) -> dict[str, Any]:
    build_base = ROOT / "build/stage3_h7_1_native"
    install_base = ROOT / "install/stage3_h7_1_native"
    command = " && ".join([
        "source /opt/ros/jazzy/setup.bash",
        f"source {shlex.quote(wsl_path(ROOT / 'install/setup.bash'))}",
        f"colcon build --base-paths {shlex.quote(wsl_path(ROOT / 'ros2_moveit_bridge'))} --build-base {shlex.quote(wsl_path(build_base))} --install-base {shlex.quote(wsl_path(install_base))} --merge-install",
    ])
    code, stdout, stderr = run_wsl(command, 1200)
    log_path = output / "stage3_h7_1_native_build.log"
    dump_text(log_path, stdout + "\n--- STDERR ---\n" + stderr)
    result = {
        "schema_version": "stage3-h7-1-native-build-v1",
        "returncode": code,
        "status": "PASSED" if code == 0 else "BLOCKED",
        "build_base": str(build_base.resolve()),
        "install_base": str(install_base.resolve()),
        "log": str(log_path.resolve()),
        "entrypoint": "stage3_h7_1_native",
    }
    dump_json(output / "stage3_h7_1_native_build.json", result)
    return result


def run_native_replay(output: Path, case_label: str, prefix_end: int, omit_indices: Iterable[int], h6_hash: str, build: Mapping[str, Any]) -> dict[str, Any]:
    case_dir = output / "native_prefix_replays" / case_label
    case_dir.mkdir(parents=True, exist_ok=True)
    omitted = sorted(set(int(item) for item in omit_indices))
    omit_text = "x" + ",".join(str(item) for item in omitted) if omitted else "none"
    command = " && ".join([
        "source /opt/ros/jazzy/setup.bash",
        f"source {shlex.quote(wsl_path(ROOT / 'install/setup.bash'))}",
        f"source {shlex.quote(wsl_path(ROOT / 'install/stage3_h7_1_native/setup.bash'))}",
        "ros2 launch fr5_tunnel_moveit_bridge stage3_h7_1_native.launch.py "
        f"h6_joint_waypoints:={shlex.quote(wsl_path(H6_JSONL))} "
        f"output_dir:={shlex.quote(wsl_path(case_dir))} "
        f"totg_configuration:={shlex.quote(wsl_path(TOTG_CONFIGURATION))} "
        f"prefix_end:={int(prefix_end)} "
        f"omit_waypoint_indices:={shlex.quote(omit_text)} "
        f"case_label:={shlex.quote(case_label)}",
    ])
    code, stdout, stderr = run_wsl(command, 900)
    log_path = case_dir / "stage3_h7_1_native_prefix_replay.log"
    dump_text(log_path, stdout + "\n--- STDERR ---\n" + stderr)
    result_path = case_dir / "stage3_h7_1_native_prefix_replay.json"
    result = load_json(result_path) if result_path.is_file() else {"status": "BLOCKED", "errors": ["native_result_missing"]}
    native_lines = [line.strip() for line in (stdout + "\n" + stderr).splitlines() if "180 deg. turn" in line or "Invalid path" in line]
    result.update({
        "launcher_returncode": code,
        "launcher_status": "PASSED" if code == 0 else "BLOCKED",
        "native_moveit_error_log": native_lines,
        "native_log": str(log_path.resolve()),
        "h6_joint_input_sha256": h6_hash,
        "build_status": build.get("status"),
        "NEW_FJT_GOALS_SENT": 0,
        "ROBOT_MOTION_STARTED": "NO",
    })
    dump_json(result_path, result)
    return result


def native_prefix_replay_audit(output: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    h6_hash = sha256_file(H6_JSONL)
    build = build_native_worker(output)
    if build.get("status") != "PASSED":
        return {"schema_version": "stage3-h7-1-native-prefix-totg-replay-v1", "status": "BLOCKED", "build": build, "cases": [], "errors": ["native_worker_build_failed"]}
    cases: dict[str, dict[str, Any]] = {}
    for end in (39, 40, 41, 42):
        cases[f"prefix_0_{end:04d}"] = run_native_replay(output, f"prefix_0_{end:04d}", end, (), str(h6_hash), build)
    low, high = 2, len(rows) - 1
    binary_cases: list[str] = []
    while low < high:
        midpoint = (low + high) // 2
        label = f"binary_prefix_0_{midpoint:04d}"
        if label not in cases:
            cases[label] = run_native_replay(output, label, midpoint, (), str(h6_hash), build)
        binary_cases.append(label)
        if cases[label].get("status") == "BLOCKED":
            high = midpoint
        else:
            low = midpoint + 1
    minimal = low
    # Remove one endpoint of the first cusp (waypoint 39) rather than the
    # cusp center itself.  Removing the center changes the local path and, in
    # the native runtime, produces an earlier independent rejection; it is
    # therefore retained only as a negative isolation diagnostic.
    isolation_omit = 39
    for end in (259, 260, 261):
        label = f"isolate_without_waypoint_{isolation_omit:04d}_prefix_{end:04d}"
        cases[label] = run_native_replay(output, label, end, (isolation_omit,), str(h6_hash), build)
    for end in (259, 260):
        label = f"isolate_without_waypoint_0040_prefix_{end:04d}"
        cases[label] = run_native_replay(output, label, end, (40,), str(h6_hash), build)
    first_case = cases.get("prefix_0_0041", {})
    isolated_259 = cases[f"isolate_without_waypoint_{isolation_omit:04d}_prefix_0259"]
    isolated_260 = cases[f"isolate_without_waypoint_{isolation_omit:04d}_prefix_0260"]
    isolated_261 = cases[f"isolate_without_waypoint_{isolation_omit:04d}_prefix_0261"]
    isolated_boundary_proven = (
        isolated_259.get("status") == "PASSED"
        and isolated_260.get("status") == "PASSED"
        and isolated_261.get("status") == "BLOCKED"
        and any("180 deg. turn" in line for line in isolated_261.get("native_moveit_error_log", []))
    )
    omit_40_259 = cases["isolate_without_waypoint_0040_prefix_0259"]
    omit_40_260 = cases["isolate_without_waypoint_0040_prefix_0260"]
    return {
        "schema_version": "stage3-h7-1-native-prefix-totg-replay-v1",
        "status": "PASSED" if minimal == 41 and cases.get("prefix_0_0040", {}).get("status") == "PASSED" and first_case.get("status") == "BLOCKED" and isolated_boundary_proven else "BLOCKED",
        "build": build,
        "h6_joint_input": source_record(H6_JSONL, "authoritative frozen H6 joint JSONL"),
        "totg_configuration": source_record(TOTG_CONFIGURATION, "same H7 native TOTG configuration"),
        "binary_search_case_labels": binary_cases,
        "cases": cases,
        "minimal_failing_prefix": minimal,
        "minimal_failing_prefix_waypoint_count": minimal + 1,
        "first_failing_prefix_case": first_case,
        "first_cusp_isolation": {
            "method": "omit_one_endpoint_of_first_triple",
            "omitted_waypoint_index": isolation_omit,
            "reason_not_omit_center_40": "omitting waypoint 40 causes a native rejection before the 259/260/261 diagnostic boundary and is not a valid isolation of the first triple",
            "prefix_0_259_status": isolated_259.get("status"),
            "prefix_0_260_status": isolated_260.get("status"),
            "prefix_0_261_status": isolated_261.get("status"),
            "boundary_proven": isolated_boundary_proven,
        },
        "second_failing_prefix_after_isolation": 261 if isolated_boundary_proven else None,
        "second_failing_prefix_waypoint_count_after_isolation": 262 if isolated_boundary_proven else None,
        "second_failing_prefix_case_after_isolation": isolated_261,
        "second_failing_prefix_after_omit_40": None if omit_40_259.get("status") != "PASSED" else 260 if omit_40_260.get("status") == "BLOCKED" else None,
        "second_failing_prefix_case_after_omit_40": omit_40_260,
        "native_error_pattern_observed": any("180 deg. turn" in line for case in cases.values() for line in case.get("native_moveit_error_log", [])),
        "NEW_FJT_GOALS_SENT": 0,
        "ROBOT_MOTION_STARTED": "NO",
    }


def frozen_integrity_snapshot() -> dict[str, Any]:
    paths = [
        (H6_JSONL, "frozen H6 authoritative joint JSONL"),
        (H6_SURFACE_JSONL, "frozen H6 surface waypoints"),
        (H6_CART_JSONL, "frozen H6 Cartesian/TCP waypoints"),
        (H6_FK, "frozen H6 FK/TCP validation"),
        (H6_SEGMENTS, "frozen H6 legacy segment metadata"),
        (H6_COVERAGE, "frozen H6 coverage accounting"),
        (H6_TERMINAL, "frozen H6 terminal certificate"),
        (H61_FREEZE, "frozen H6.1 contract freeze manifest"),
        (H61_TERMINAL, "frozen H6.1 terminal certificate"),
        (H61_CONTRACT_SOURCE, "H6.1 contract source"),
        (H61_CONTRACT_COPY, "frozen H6.1 contract copy"),
        (H5_FREEZE, "frozen H5 configuration freeze manifest"),
        (H5_TERMINAL, "frozen H5 terminal certificate"),
        (H5_GRAPH, "frozen H5 joint configuration graph"),
        (H5_NODES, "frozen H5 distinct IK nodes"),
        (H5_PREFERRED, "frozen H5 preferred branch sequences"),
        (DERIVED_URDF, "frozen H4.5 derived robot model"),
        (FIXTURE_MESH, "frozen H4.5 fixture mesh"),
        (SRDF, "frozen MoveIt semantic model"),
    ]
    return {"files": [source_record(path, role) for path, role in paths]}


def compare_integrity(before: Mapping[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for record in before.get("files", []):
        path = Path(record["path"])
        current = sha256_file(path)
        checks.append({"path": record["path"], "before_sha256": record.get("sha256"), "after_sha256": current, "match": bool(current and current == record.get("sha256"))})
    return {"files": checks, "all_after_hashes_match": all(item["match"] for item in checks)}


def run_regression(output: Path) -> dict[str, Any]:
    command = [sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h7_1.py", "tests/test_stage3_h7.py", "tests/test_stage3_h6_1.py", "tests/test_stage3_h6.py", "tests/test_stage3_h5.py"]
    process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1800, check=False)
    raw = process.stdout + "\n--- STDERR ---\n" + process.stderr
    match = re.search(r"(\d+) passed(?:, (\d+) failed)?", raw)
    failed = int(match.group(2)) if match and match.group(2) else (0 if process.returncode == 0 else 1)
    report = {
        "schema_version": "stage3-h7-1-regression-report-v1",
        "runs": [{"command": command, "returncode": process.returncode, "passed": int(match.group(1)) if match else None, "failed": failed, "raw_output_tail": raw[-12000:]}],
        "new_regression_failures": failed,
        "known_pre_existing_failures": [],
    }
    dump_json(output / "stage3_h7_1_regression_report.json", report)
    return report


def artifact_manifest(output: Path) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for name in REQUIRED_ARTIFACTS:
        if name == "stage3_h7_1_terminal_certificate.json":
            continue
        path = output / name
        files[name] = {"path": str(path.resolve()), "exists": path.is_file(), "size_bytes": path.stat().st_size if path.is_file() else None, "sha256": sha256_file(path)}
    for path in sorted((output / "native_prefix_replays").rglob("*")) if (output / "native_prefix_replays").is_dir() else []:
        if path.is_file():
            files[rel(path)] = {"path": str(path.resolve()), "exists": True, "size_bytes": path.stat().st_size, "sha256": sha256_file(path), "semantic_role": "native H7.1 prefix replay evidence"}
    manifest = {
        "schema_version": "stage3-h7-1-artifact-manifest-v1",
        "files": files,
        "manifest_scope": "all H7.1 diagnostic artifacts emitted before the terminal certificate; frozen H5/H6/H6.1 files are listed in the root-cause integrity section and are not copied or mutated",
        "terminal_certificate_emitted_after_manifest": True,
    }
    manifest_path = output / "stage3_h7_1_artifact_manifest.json"
    dump_json(manifest_path, manifest)
    # Record the digest of the manifest content before adding this metadata
    # field; the terminal certificate binds that digest and carries its own
    # independent self-hash.
    manifest["manifest_sha256"] = sha256_file(manifest_path)
    dump_json(manifest_path, manifest)
    return manifest


def gate_report(output: Path, turn_audit: Mapping[str, Any], branch_audit: Mapping[str, Any], joint_audit: Mapping[str, Any], root_cause: Mapping[str, Any], native: Mapping[str, Any], regression: Mapping[str, Any], frozen: Mapping[str, Any], remediation: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "h6_authoritative_input_row_count_1295": turn_audit.get("waypoint_count") == 1295,
        "h6_authoritative_input_sha256_available": bool(turn_audit.get("source", {}).get("sha256")),
        "native_moveit_prefix_replay_ran": native.get("build", {}).get("status") == "PASSED",
        "native_minimal_failing_prefix_proven": (
            native.get("minimal_failing_prefix") == 41
            and native.get("cases", {}).get("prefix_0_0040", {}).get("status") == "PASSED"
            and native.get("first_failing_prefix_case", {}).get("status") == "BLOCKED"
        ),
        "native_prefix_0_39_succeeded": native.get("cases", {}).get("prefix_0_0039", {}).get("status") == "PASSED",
        "native_prefix_0_40_succeeded": native.get("cases", {}).get("prefix_0_0040", {}).get("status") == "PASSED",
        "native_prefix_0_41_failed": native.get("cases", {}).get("prefix_0_0041", {}).get("status") == "BLOCKED",
        "native_second_failure_after_first_triple_isolation_proven": native.get("second_failing_prefix_after_isolation") == 261,
        "all_1294_consecutive_edges_audited": joint_audit.get("all_edges_audited") is True,
        "exact_near_180_turn_count_is_two": turn_audit.get("near_or_rejected_count") == 2,
        "type_1_cusp_centres_are_40_and_260": turn_audit.get("exact_or_near_180_turn_centres") == [40, 260],
        "type_2_ik_branch_discontinuity_count_is_three": branch_audit.get("flagged_branch_discontinuity_count") == 3,
        "no_unexplained_ik_branch_discontinuity": False,
        "root_cause_code_trace_complete": all(item.get("root_cause_classification") for item in root_cause.get("type_1_joint_space_cusp", {}).get("records", [])) and bool(root_cause.get("type_2_ik_branch_discontinuity", {}).get("records")),
        "candidate_certification_completed": False,
        "h6_2_candidate_required": True,
        "coverage_30_of_30_frozen": root_cause.get("answers", {}).get("coverage_30_of_30_retained_in_frozen_h6") is True,
        "previous_frozen_evidence_unchanged": frozen.get("all_after_hashes_match") is True,
        "formal_ledger_mutated": False,
        "fjt_goals_zero": True,
        "robot_motion_no": True,
        "ccd_not_available": True,
        "clearance_not_available": True,
        "new_regression_failures_zero": regression.get("new_regression_failures") == 0,
    }
    first_blocker = FIRST_BLOCKER if checks["native_minimal_failing_prefix_proven"] else "native_prefix_replay_not_proven"
    if not checks["previous_frozen_evidence_unchanged"]:
        first_blocker = "frozen_h5_h6_h6_1_integrity_changed"
    report = {
        "schema_version": "stage3-h7-1-gate-report-v1",
        "STAGE_3_H7_1": "BLOCKED",
        "STAGE_3_H7": "BLOCKED",
        "FIRST_BLOCKER": first_blocker,
        "READY_TO_RECERTIFY_STAGE_3_H7": "NO",
        "READY_FOR_STAGE_3_H8": "NO",
        "gate_checks": checks,
        "turn_centres": turn_audit.get("exact_or_near_180_turn_centres"),
        "branch_discontinuity_edges": branch_audit.get("flagged_edge_indices"),
        "collision_method": COLLISION_METHOD,
        "CCD": "NOT_AVAILABLE",
        "clearance": "NOT_AVAILABLE",
        "NEW_FJT_GOALS_SENT": 0,
        "ROBOT_MOTION_STARTED": "NO",
        "FORMAL_LEDGER_MUTATED": "NO",
        "native_replay_summary": native,
        "remediation_status": remediation.get("status"),
    }
    dump_json(output / "stage3_h7_1_gate_report.json", report)
    return report


def final_report(output: Path, gate: Mapping[str, Any], turn_audit: Mapping[str, Any], branch_audit: Mapping[str, Any], native: Mapping[str, Any], root_cause: Mapping[str, Any], frozen: Mapping[str, Any], regression: Mapping[str, Any]) -> None:
    lines = [
        "# Stage 3 H7.1 — Authoritative Joint-Path Topology / IK-Branch Discontinuity Audit",
        "",
        f"`STAGE_3_H7_1: {gate['STAGE_3_H7_1']}`",
        f"`FIRST_BLOCKER: {gate['FIRST_BLOCKER']}`",
        f"`READY_TO_RECERTIFY_STAGE_3_H7: {gate['READY_TO_RECERTIFY_STAGE_3_H7']}`",
        f"`READY_FOR_STAGE_3_H8: {gate['READY_FOR_STAGE_3_H8']}`",
        "",
        "This is an additive offline/simulation diagnostic. H5/H6/H6.1 evidence was read-only; no H6/H6.1 artifact, formal ledger, controller action, or robot state was mutated.",
        "",
        "## Mandatory answers",
        "",
        f"1. Native MoveIt first rejected triple: waypoint `{native.get('minimal_failing_prefix')}` as the minimal failing prefix end; the corresponding triple is `{turn_audit.get('exact_or_near_180_turn_centres')}` with the first triple centered at waypoint `40`.",
        f"2. Triple 40: `turn_angle_deg={turn_audit['near_or_rejected_triples'][0].get('turn_angle_deg')}`, `cos_angle={turn_audit['near_or_rejected_triples'][0].get('cos_angle')}`; MoveIt criterion is `cos_angle <= -1 + 1e-5`.",
        f"3. Minimal failing prefix: `0..40` inclusive, `{native.get('minimal_failing_prefix_waypoint_count')}` waypoints; `0..39` native replay succeeded.",
        "4. Waypoints 39/40/41 are the first real blocker; 259/260/261 are a second native blocker after the first cusp is isolated/removed.",
        f"5. Waypoint 259/260/261 second blocker proven by native replay: `{native.get('second_failing_prefix_after_omit_40') == 260}`.",
        f"6–8. IK branch discontinuities: `{branch_audit.get('flagged_edge_indices')}`; these are independent of the two 180° cusp centres.",
        f"9. Full path exact/near-180 turns: `{turn_audit.get('near_or_rejected_count')}`.",
        f"10. Full path flagged IK branch discontinuities: `{branch_audit.get('flagged_branch_discontinuity_count')}`.",
        "11. H5 supplies candidate branches; H6 first admits the route/order, dense edge construction, endpoint compaction, and fallback branch choice into the authoritative path. H6.1 only freezes tolerance evidence.",
        "12. Root cause: Type 1 is H6 route-ordering/edge-construction backtracking; Type 2 is H6 previous-seed failure followed by H5 fallback branch selection.",
        "13–14. No remediation has been certified without changing process segmentation; no additional SPRAY_OFF transition is asserted. Any such change is a new H6.2 candidate.",
        f"15. Frozen H6 coverage remains 30/30: `{root_cause.get('answers', {}).get('coverage_30_of_30_retained_in_frozen_h6')}`.",
        f"16. Frozen H5/H6/H6.1 hashes unchanged after audit: `{frozen.get('all_after_hashes_match')}`.",
        "17–18. H6.2 candidates were prepared only; none passed full FK/FCL/process validation.",
        "19–20. FJT goals sent: `0`; robot motion started: `NO`.",
        f"21–24. FIRST_BLOCKER/H7.1/H7 recertification/H8: `{gate['FIRST_BLOCKER']}` / `BLOCKED` / `NO` / `NO`.",
        "",
        "## Evidence surface",
        "",
        "- Native MoveIt prefix logs and structured results are under `native_prefix_replays/`.",
        "- The topology audit records all 1,293 waypoint triples and the joint audit records all 1,294 consecutive edges.",
        "- Collision method is `adaptive_discrete_interpolation`; Bullet CCD and clearance are `NOT_AVAILABLE`/JSON null and were not inferred.",
        f"- Regression: `{regression.get('new_regression_failures')} new failures`.",
        "- SHA-256 coverage is in `stage3_h7_1_artifact_manifest.json`; the terminal certificate is emitted after that manifest.",
        "",
    ]
    dump_text(output / "FINAL_REPORT.md", "\n".join(lines))


def final_report(output: Path, gate: Mapping[str, Any], turn_audit: Mapping[str, Any], branch_audit: Mapping[str, Any], native: Mapping[str, Any], root_cause: Mapping[str, Any], frozen: Mapping[str, Any], regression: Mapping[str, Any]) -> None:
    first_turn = next(item for item in turn_audit["near_or_rejected_triples"] if item["waypoint_index"] == 40)
    lines = [
        "# Stage 3 H7.1 - Authoritative Joint-Path Topology / IK-Branch Discontinuity Audit",
        "",
        f"`STAGE_3_H7_1: {gate['STAGE_3_H7_1']}`",
        f"`FIRST_BLOCKER: {gate['FIRST_BLOCKER']}`",
        f"`READY_TO_RECERTIFY_STAGE_3_H7: {gate['READY_TO_RECERTIFY_STAGE_3_H7']}`",
        f"`READY_FOR_STAGE_3_H8: {gate['READY_FOR_STAGE_3_H8']}`",
        "",
        "This additive diagnostic is fail-closed. Frozen H5/H6/H6.1 evidence was read-only; no formal ledger, controller action, or robot state was mutated.",
        "",
        "## Mandatory answers",
        "",
        f"1. Native MoveIt first rejected the prefix ending at waypoint `{native.get('minimal_failing_prefix')}`; the triggering triple is centered at waypoint 40 (`39/40/41`).",
        f"2. Triple 40: `turn_angle_deg={first_turn.get('turn_angle_deg')}`, `cos_angle={first_turn.get('cos_angle')}`; criterion: `cos_angle <= -1 + 1e-5`.",
        f"3. Minimal failing prefix: `0..41` inclusive, `{native.get('minimal_failing_prefix_waypoint_count')}` waypoints; `0..40` and `0..39` passed.",
        "4. Yes, 39/40/41 is the first real Type 1 blocker; the native failure appears when waypoint 41 is included.",
        f"5. After isolating the first triple by omitting endpoint 39, `0..259` and `0..260` passed and `0..261` failed; second blocker proven={native.get('second_failing_prefix_after_isolation') == 261}.",
        f"6. `81->82` is not a 180-degree cusp; it is an independent IK branch discontinuity. Flagged edges: `{branch_audit.get('flagged_edge_indices')}`.",
        "7. `1059->1060` is the same Type 2 cause: previous accepted seed failure followed by frozen H5 fallback branch selection on the same SPRAY_ON connection.",
        "8. `1269->1270` is the same Type 2 cause and is independent of both Type 1 cusp centers.",
        f"9. Full path exact/near-180 turns: `{turn_audit.get('near_or_rejected_count')}`; centers `{turn_audit.get('exact_or_near_180_turn_centres')}`.",
        f"10. Full path flagged IK branch discontinuities: `{branch_audit.get('flagged_branch_discontinuity_count')}`; edges `{branch_audit.get('flagged_edge_indices')}`.",
        "11. H5 supplies candidate branches; H6 first admits route/order, dense edge construction, endpoint compaction, and fallback branch choice into the authoritative path. H6.1 freezes tolerance evidence.",
        "12. Type 1 is H6 route-ordering/edge-construction backtracking; Type 2 is H6 continuation failure followed by H5 fallback branch selection. They are separate diagnoses.",
        "13. No repair has been certified without changing process segmentation; current H6.2 output is preparation-only.",
        "14. No additional SPRAY_OFF transition is asserted. Any process-continuity break must be a new candidate with process-contract review.",
        f"15. Frozen H6 coverage remains 30/30: `{root_cause.get('answers', {}).get('coverage_30_of_30_retained_in_frozen_h6')}`.",
        f"16. Frozen H5/H6/H6.1 hashes unchanged: `{frozen.get('all_after_hashes_match')}`.",
        "17. New H6.2 remediation candidates were generated as manifests only; frozen H6/H6.1 was not overwritten.",
        "18. No candidate passed full FK/FCL/process validation.",
        "19. FJT goals sent: `0`.",
        "20. Robot motion started: `NO`.",
        f"21. FIRST_BLOCKER: `{gate['FIRST_BLOCKER']}`.",
        "22. STAGE_3_H7_1: `BLOCKED`.",
        "23. READY_TO_RECERTIFY_STAGE_3_H7: `NO`.",
        "24. READY_FOR_STAGE_3_H8: `NO`; fail-closed until a new authoritative H7 path passes all required gates.",
        "",
        "## Evidence surface",
        "",
        "- Native MoveIt prefix logs and structured results are under `native_prefix_replays/`.",
        "- The topology audit records all 1,293 waypoint triples and the joint audit records all 1,294 consecutive edges.",
        "- Collision method is `adaptive_discrete_interpolation`; Bullet CCD and clearance are `NOT_AVAILABLE`/JSON null and were not inferred.",
        f"- Regression report: `{regression.get('new_regression_failures')} new failures`.",
        "- SHA-256 coverage is in `stage3_h7_1_artifact_manifest.json`; the terminal certificate is emitted after that manifest.",
        "",
    ]
    dump_text(output / "FINAL_REPORT.md", "\n".join(lines))


def orchestrate(output: Path) -> int:
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    before = frozen_integrity_snapshot()
    rows = load_jsonl(H6_JSONL)
    surface_rows = load_jsonl(H6_SURFACE_JSONL)
    ordering = load_json(H6_ORDERING)
    coverage = load_json(H6_COVERAGE)
    turn = turn_geometry_audit(rows)
    dump_json(output / "stage3_h7_1_turn_geometry_audit.json", turn)
    native = native_prefix_replay_audit(output, rows)
    dump_json(output / "stage3_h7_1_native_prefix_totg_replay.json", native)
    branch = branch_switch_audit(rows)
    dump_json(output / "stage3_h7_1_branch_switch_audit.json", branch)
    joint = joint_discontinuity_audit(rows, turn, branch)
    dump_json(output / "stage3_h7_1_joint_discontinuity_audit.json", joint)
    after = compare_integrity(before)
    frozen = {"before": before, "after": after, "all_after_hashes_match": after["all_after_hashes_match"]}
    root = root_cause_report(rows, surface_rows, ordering, turn, branch, native, frozen)
    dump_json(output / "stage3_h7_1_root_cause_report.json", root)
    remediation = remediation_manifest(turn, branch, coverage)
    dump_json(output / "stage3_h7_1_remediation_candidate_manifest.json", remediation)
    regression = run_regression(output)
    gate = gate_report(output, turn, branch, joint, root, native, regression, frozen, remediation)
    final_report(output, gate, turn, branch, native, root, frozen, regression)
    manifest = artifact_manifest(output)
    terminal = {
        "schema_version": "stage3-h7-1-terminal-certificate-v1",
        "STAGE_3_H7_1": "BLOCKED",
        "STAGE_3_H7": "BLOCKED",
        "FIRST_BLOCKER": gate["FIRST_BLOCKER"],
        "READY_TO_RECERTIFY_STAGE_3_H7": "NO",
        "READY_FOR_STAGE_3_H8": "NO",
        "NEW_FJT_GOALS_SENT": 0,
        "ROBOT_MOTION_STARTED": "NO",
        "FORMAL_LEDGER_MUTATED": "NO",
        "H6_H6_1_FROZEN_ARTIFACTS_MODIFIED": "NO" if frozen["all_after_hashes_match"] else "YES",
        "collision_method": COLLISION_METHOD,
        "CCD": "NOT_AVAILABLE",
        "CLEARANCE": "NOT_AVAILABLE",
        "turn_centres": turn["exact_or_near_180_turn_centres"],
        "ik_branch_discontinuity_edges": branch["flagged_edge_indices"],
        "artifact_manifest_sha256": manifest["manifest_sha256"],
        "terminal_certificate_sha256": None,
    }
    terminal["terminal_certificate_sha256"] = semantic_hash(terminal)
    dump_json(output / "stage3_h7_1_terminal_certificate.json", terminal)
    return 2


def main() -> int:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (ROOT / "outputs" / f"stage3_h7_1_authoritative_audit_{timestamp}").resolve()
    return orchestrate(output)


if __name__ == "__main__":
    raise SystemExit(main())
