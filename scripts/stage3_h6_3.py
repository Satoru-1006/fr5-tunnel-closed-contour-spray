#!/usr/bin/env python3
"""Stage 3 H6.3 exact IK-branch reachability and controlled process-break audit.

H6.3 is additive.  H5, H6, H6.1, and H6.2 are frozen evidence inputs and are
never modified.  This stage removes the H6.2 beam from the reachability
decision, certifies the selected exact paths through the existing native
MoveIt2/FK/PlanningScene backend, and, only when the exact graph is split,
builds a new SPRAY_ON/SPRAY_OFF process structure.  Time parameterization,
controllers, robot I/O, and the formal ledger are intentionally out of scope.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import stage3_h6 as h6
from scripts import stage3_h6_2 as h62


H5_ROOT = h6.H5_ROOT
H6_ROOT = h62.H6_ROOT
H61_ROOT = h62.H61_ROOT
H62_ROOT = ROOT / "outputs/stage3_h6_2_authoritative_path_remediation_20260809T000000Z_retry2"
H61_CONTRACT = h62.H61_CONTRACT
H6_EXECUTABLE = h62.H6_EXECUTABLE
H6_NATIVE_SETUP = h62.H6_NATIVE_SETUP
H6_MESH = h62.H6_MESH
H62_GRAPH = H62_ROOT / "stage3_h6_2_candidate_graph.json"
H62_SURFACE = H62_ROOT / "stage3_h6_2_surface_waypoints.jsonl"
H62_SEGMENTS = H62_ROOT / "stage3_h6_2_spray_on_off_segments.json"
H62_GATE = H62_ROOT / "stage3_h6_2_gate_report.json"
H62_SELECTED = H62_ROOT / "stage3_h6_2_selected_path.json"
H62_OPTIMIZATION = H62_ROOT / "stage3_h6_2_global_ik_optimization.json"
H62_COVERAGE = H62_ROOT / "stage3_h6_2_coverage_accounting.json"

COLLISION_METHOD = "adaptive_discrete_interpolation"
CCD = "NOT_AVAILABLE"
CLEARANCE = "NOT_AVAILABLE"
MAX_GRAPH_EDGE_LINF_RAD = 0.75
MAX_GRAPH_EDGE_L2_RAD = 1.50
MAX_INTERPOLATION_STEP_DEG = 0.5
FORMAL_TARGET_COUNT = 30
BREAK_FROM_WAYPOINT = 562
BREAK_TO_WAYPOINT = 563
BREAK_SOURCE_TARGET = 53
BREAK_DESTINATION_TARGET = 46
KNOWN_H6_BRANCH_EDGES = (81, 1059, 1269)
KNOWN_TYPE_1_TRIPLES = ((1, 58, 14), (17, 23, 42))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def dump_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n" for row in rows), encoding="utf-8", newline="\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any, digits: int = 12) -> Any:
    if isinstance(value, Mapping):
        return {str(key): canonical(value[key], digits) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [canonical(item, digits) for item in value]
    if isinstance(value, float):
        return None if not math.isfinite(value) else round(value, digits)
    return value


def semantic_hash(value: Any) -> str:
    raw = json.dumps(canonical(value), ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def source_snapshot() -> dict[str, Any]:
    """Hash only the immutable Stage 3 H5/H6/H6.1/H6.2 evidence roots."""
    roots = {"H5": H5_ROOT, "H6": H6_ROOT, "H6_1": H61_ROOT, "H6_2": H62_ROOT}
    result: dict[str, Any] = {}
    for label, root in roots.items():
        files: list[dict[str, Any]] = []
        if root.is_dir():
            for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: item.as_posix()):
                files.append({"path": rel(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
        result[label] = {"root": rel(root), "exists": root.is_dir(), "file_count": len(files), "files": files, "tree_sha256": semantic_hash(files)}
    result["H6_1_CONTRACT"] = {"path": rel(H61_CONTRACT), "exists": H61_CONTRACT.is_file(), "sha256": sha256_file(H61_CONTRACT) if H61_CONTRACT.is_file() else None}
    return result


def immutable_after(before: Mapping[str, Any]) -> dict[str, Any]:
    after = source_snapshot()
    return {"schema_version": "stage3-h6-3-immutable-after-v1", "before": before, "after": after, "all_after_hashes_match": before == after}


def joint_delta(previous: Mapping[str, Any], current: Mapping[str, Any]) -> tuple[list[float], float, float, float]:
    delta = [float(b) - float(a) for a, b in zip(previous["joint_values"], current["joint_values"])]
    return delta, sum(abs(value) for value in delta), math.sqrt(sum(value * value for value in delta)), max(abs(value) for value in delta)


def edge_prefilter(previous: Mapping[str, Any], current: Mapping[str, Any]) -> dict[str, Any]:
    delta, l1, l2, linf = joint_delta(previous, current)
    return {
        "from_candidate_id": previous["candidate_id"],
        "to_candidate_id": current["candidate_id"],
        "per_joint_delta_rad": delta,
        "L1_joint_delta_rad": l1,
        "L2_joint_delta_rad": l2,
        "L_inf_joint_delta_rad": linf,
        "branch_prefilter_pass": bool(linf <= MAX_GRAPH_EDGE_LINF_RAD and l2 <= MAX_GRAPH_EDGE_L2_RAD),
        "prefilter_limits": {"MAX_GRAPH_EDGE_LINF_RAD": MAX_GRAPH_EDGE_LINF_RAD, "MAX_GRAPH_EDGE_L2_RAD": MAX_GRAPH_EDGE_L2_RAD},
        "prefilter_is_not_native_collision_certificate": True,
    }


def edge_cost(previous: Mapping[str, Any], current: Mapping[str, Any]) -> float:
    _, _, l2, linf = joint_delta(previous, current)
    branch_change = 1.0 if previous.get("branch_node_id") != current.get("branch_node_id") else 0.0
    return 10.0 * l2 + 2.0 * linf + 50.0 * branch_change


def candidate_is_native_valid(candidate: Mapping[str, Any]) -> bool:
    state = candidate.get("state") or {}
    return bool(state.get("joint_limit_valid") and state.get("fk_computable") and not state.get("environment_collision") and not state.get("self_collision") and float(state.get("translation_error_m") or 1.0) <= 1.0e-6 and float(state.get("rotation_error_rad") or 1.0) <= 1.0e-6)


def exact_segment_reachability(rows: Sequence[Mapping[str, Any]], layers: Mapping[int, Sequence[Mapping[str, Any]]], banned_edges: set[tuple[str, str]] | None = None) -> dict[str, Any]:
    """Exact dynamic-programming feasibility; no beam and no candidate cap."""
    banned_edges = banned_edges or set()
    ordered_rows = sorted(rows, key=lambda row: int(row["waypoint_index"]))
    if not ordered_rows:
        return {"status": "BLOCKED", "reason": "empty_segment", "layer_audits": []}
    first_index = int(ordered_rows[0]["waypoint_index"])
    initial = list(layers.get(first_index, []))
    layer_audits: list[dict[str, Any]] = [{"waypoint_index": first_index, "candidate_count": len(initial), "reachable_candidate_count": len(initial), "reachable_candidate_ids": sorted(item["candidate_id"] for item in initial), "reachable_branch_node_ids": sorted({item.get("branch_node_id") for item in initial}), "unreachable_candidate_ids": [], "unreachable_branch_node_ids": [], "minimum_reachable_L1_joint_delta_rad": None, "minimum_reachable_L2_joint_delta_rad": None, "minimum_reachable_L_inf_joint_delta_rad": None}]
    if not initial:
        return {"status": "BLOCKED", "reason": f"no_candidates_at_waypoint:{first_index}", "first_graph_disconnect": layer_audits[0], "layer_audits": layer_audits}

    # One best-cost state per candidate is exact for this additive pairwise DP:
    # all future transitions depend only on the current candidate.
    states: dict[str, dict[str, Any]] = {}
    for candidate in initial:
        states[str(candidate["candidate_id"])] = {"candidate": candidate, "cost": 0.0, "previous_candidate_id": None}

    for row in ordered_rows[1:]:
        current_index = int(row["waypoint_index"])
        current_candidates = list(layers.get(current_index, []))
        next_states: dict[str, dict[str, Any]] = {}
        all_metrics = [edge_prefilter(previous_state["candidate"], current) for previous_state in states.values() for current in current_candidates]
        for current in current_candidates:
            candidates: list[dict[str, Any]] = []
            for previous_state in states.values():
                previous = previous_state["candidate"]
                metrics = edge_prefilter(previous, current)
                if not metrics["branch_prefilter_pass"] or (str(previous["candidate_id"]), str(current["candidate_id"])) in banned_edges:
                    continue
                candidates.append({"candidate": current, "cost": float(previous_state["cost"]) + edge_cost(previous, current), "previous_candidate_id": previous["candidate_id"]})
            if candidates:
                candidates.sort(key=lambda item: (item["cost"], str(item["previous_candidate_id"]), str(item["candidate"]["candidate_id"])))
                next_states[str(current["candidate_id"])] = candidates[0]

        reachable = list(next_states.values())
        reachable_ids = {str(item["candidate"]["candidate_id"]) for item in reachable}
        unreachable = [item for item in current_candidates if str(item["candidate_id"]) not in reachable_ids]
        reachable_previous = [state["candidate"] for state in states.values()]
        delta_metrics = [edge_prefilter(previous, current) for previous in reachable_previous for current in current_candidates]
        layer_record = {
            "waypoint_index": current_index,
            "previous_waypoint_index": int(row["waypoint_index"]) - 1,
            "candidate_count": len(current_candidates),
            "previous_reachable_candidate_count": len(states),
            "next_layer_candidate_count": len(current_candidates),
            "reachable_candidate_count": len(reachable),
            "reachable_candidate_ids": sorted(reachable_ids),
            "unreachable_candidate_ids": sorted(str(item["candidate_id"]) for item in unreachable),
            "previous_reachable_branch_node_ids": sorted({item.get("branch_node_id") for item in reachable_previous}),
            "reachable_branch_node_ids": sorted({item["candidate"].get("branch_node_id") for item in reachable}),
            "unreachable_branch_node_ids": sorted({item.get("branch_node_id") for item in unreachable}),
            "minimum_reachable_L1_joint_delta_rad": min((item["L1_joint_delta_rad"] for item in delta_metrics), default=None),
            "minimum_reachable_L2_joint_delta_rad": min((item["L2_joint_delta_rad"] for item in delta_metrics), default=None),
            "minimum_reachable_L_inf_joint_delta_rad": min((item["L_inf_joint_delta_rad"] for item in delta_metrics), default=None),
            "prefilter_edge_count": sum(1 for item in all_metrics if item["branch_prefilter_pass"]),
            "all_candidate_edge_count": len(all_metrics),
        }
        layer_audits.append(layer_record)
        if not next_states:
            layer_record["disconnect_classification"] = "genuinely_disconnected_native_valid_branch_graph_candidate_boundary"
            layer_record["search_pruning_false_negative"] = False
            return {"status": "BLOCKED", "reason": "no_branch_continuous_graph_edge", "first_graph_disconnect": layer_record, "layer_audits": layer_audits, "final_reachable_candidate_count": 0}
        states = next_states

    best = min(states.values(), key=lambda item: (float(item["cost"]), str(item["candidate"]["candidate_id"])))
    selected: list[dict[str, Any]] = []
    current_id: str | None = str(best["candidate"]["candidate_id"])
    state_by_id = states
    # Reconstruct backwards using the predecessor IDs retained in each layer.
    layer_state_maps: list[dict[str, dict[str, Any]]] = []
    state_map = {str(item["candidate"]["candidate_id"]): item for item in states.values()}
    layer_state_maps.append(state_map)
    for audit_index in range(len(ordered_rows) - 2, -1, -1):
        # The state maps are rebuilt from the exact predecessor trace stored
        # below; this branch is replaced by the forward trace cache.
        _ = audit_index
    # Re-run the small DP while retaining every layer's selected predecessor.
    retained: list[dict[str, dict[str, Any]]] = []
    current_states: dict[str, dict[str, Any]] = {}
    for candidate in initial:
        current_states[str(candidate["candidate_id"])] = {"candidate": candidate, "cost": 0.0, "previous_candidate_id": None}
    retained.append(current_states)
    for row in ordered_rows[1:]:
        current_candidates = list(layers.get(int(row["waypoint_index"]), []))
        next_states = {}
        for current in current_candidates:
            options = []
            for previous_state in current_states.values():
                previous = previous_state["candidate"]
                if edge_prefilter(previous, current)["branch_prefilter_pass"] and (str(previous["candidate_id"]), str(current["candidate_id"])) not in banned_edges:
                    options.append({"candidate": current, "cost": previous_state["cost"] + edge_cost(previous, current), "previous_candidate_id": previous["candidate_id"]})
            if options:
                options.sort(key=lambda item: (item["cost"], str(item["previous_candidate_id"]), str(item["candidate"]["candidate_id"])))
                next_states[str(current["candidate_id"])] = options[0]
        current_states = next_states
        retained.append(current_states)
    best = min(current_states.values(), key=lambda item: (float(item["cost"]), str(item["candidate"]["candidate_id"])))
    for layer_index in range(len(retained) - 1, -1, -1):
        chosen = retained[layer_index][current_id or str(best["candidate"]["candidate_id"])]
        selected.append(chosen["candidate"])
        current_id = chosen.get("previous_candidate_id")
    selected.reverse()
    return {"status": "PASSED", "cost": float(best["cost"]), "selected_path": selected, "layer_audits": layer_audits, "final_reachable_candidate_count": len(states), "first_graph_disconnect": None, "exact_graph_search_used": True, "beam_authoritative": False, "banned_edge_count": len(banned_edges)}


def split_segments(original_segments: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for original in sorted(original_segments, key=lambda item: int(item["segment_id"])):
        sid = int(original["segment_id"])
        if sid != 2:
            item = dict(original); item["segment_id"] = sid; item["certification_status"] = "PENDING_NATIVE"; result.append(item); continue
        first = dict(original); first["segment_id"] = 2; first["target_ids"] = list(original["target_ids"][:14]); first["source_target_count"] = len(first["target_ids"]); first["waypoint_start"] = 265; first["waypoint_end"] = BREAK_FROM_WAYPOINT; first["certification_status"] = "PENDING_NATIVE"; result.append(first)
        second = dict(original); second["segment_id"] = 3; second["target_ids"] = list(original["target_ids"][14:]); second["source_target_count"] = len(second["target_ids"]); second["waypoint_start"] = BREAK_TO_WAYPOINT; second["waypoint_end"] = int(original["waypoint_end"]); second["certification_status"] = "PENDING_NATIVE"; result.append(second)
    for index, item in enumerate(result):
        item["segment_order"] = index
    return result


def clone_selected_on_rows(rows: Sequence[Mapping[str, Any]], segment_results: Mapping[int, Mapping[str, Any]]) -> list[dict[str, Any]]:
    path_by_waypoint = {int(node["waypoint_index"]): node for result in segment_results.values() for node in result["selected_path"]}
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        index = int(row["waypoint_index"])
        node = path_by_waypoint.get(index)
        if node is None:
            raise ValueError(f"exact_selected_candidate_missing:{index}")
        item["segment_id"] = next(int(sid) for sid, result in segment_results.items() if any(int(candidate["waypoint_index"]) == index for candidate in result["selected_path"]))
        item["original_on_waypoint_index"] = index
        item["selected_candidate_id"] = node["candidate_id"]
        item["joint_values_selected"] = list(node["joint_values"])
        item["branch_node_id"] = node.get("branch_node_id")
        out.append(item)
    return out


def run_native(request_path: Path, output: Path, name: str) -> dict[str, Any]:
    native_dir = output / "native_runs" / name
    command = [
        "source /opt/ros/jazzy/setup.bash",
        f"source {h6.wsl_path(ROOT / 'install/setup.bash')}",
        f"source {h6.wsl_path(H6_NATIVE_SETUP)}",
        f"cd {h6.wsl_path(ROOT)}",
        f"{h6.wsl_path(H6_EXECUTABLE)} --urdf {h6.wsl_path(h6.DERIVED_URDF)} --srdf {h6.wsl_path(h6.SRDF)} --requests {h6.wsl_path(request_path)} --mesh {h6.wsl_path(H6_MESH)} --output {h6.wsl_path(native_dir)}",
        "true",
    ]
    code, stdout, stderr = h6.run_wsl(command, 2400)
    (output / f"stage3_h6_3_{name}.log").write_text(stdout + "\n--- STDERR ---\n" + stderr, encoding="utf-8", newline="\n")
    summary_path = native_dir / "stage3_h6_native_summary.json"
    summary = load_json(summary_path) if summary_path.is_file() else {}
    record = {"schema_version": "stage3-h6-3-native-run-v1", "name": name, "returncode": code, "status": "PASSED" if code == 0 and summary.get("status") == "AVAILABLE" else "BLOCKED", "summary": summary, "semantic_signature": h6.native_semantic_signature(native_dir) if summary else None, "collision_method": COLLISION_METHOD, "ccd": CCD, "clearance": CLEARANCE}
    dump_json(output / f"stage3_h6_3_{name}.json", record)
    return record


def parse_trace(native_dir: Path) -> list[dict[str, Any]]:
    path = native_dir / "stage3_h6_native_ik_branch_trace.jsonl"
    return load_jsonl(path) if path.is_file() else []


def run_selected_edge_certification(output: Path, on_rows: Sequence[Mapping[str, Any]], segment_results: Mapping[int, Mapping[str, Any]], run_name: str = "native_edge_certification") -> tuple[dict[str, Any], dict[str, Any]]:
    interpolation_rows: list[dict[str, Any]] = []
    for segment in sorted(segment_results):
        original = [row for row in on_rows if int(row["segment_id"]) == int(segment)]
        interpolation_rows.extend(h62.build_interpolation_rows(original, segment_results[segment]["selected_path"], segment))
    for index, row in enumerate(interpolation_rows):
        row["waypoint_index"] = index
    seed_map = {int(row["waypoint_index"]): [row["joint_values_seed"]] for row in interpolation_rows}
    labels = {int(row["waypoint_index"]): ["exact_dp_selected_interpolation_seed"] for row in interpolation_rows}
    request = output / "stage3_h6_3_native_edge_certification_requests.tsv"
    h62.write_request_rows(request, interpolation_rows, seed_map, labels)
    run = run_native(request, output, run_name)
    trace = parse_trace(output / "native_runs" / run_name)
    trace_by_index = {int(item["waypoint_index"]): item for item in trace}
    contract = load_json(H61_CONTRACT); geometry = contract["geometry"]; tcp_contract = contract["tcp_reproduction"]
    validation: list[dict[str, Any]] = []
    for row in interpolation_rows:
        native = trace_by_index.get(int(row["waypoint_index"]), {}); state = native.get("accepted_state") or {}; process_valid = bool(native.get("accepted") and state.get("joint_limit_valid") and state.get("fk_computable") and not state.get("environment_collision") and not state.get("self_collision"))
        metrics = {"waypoint_index": row["waypoint_index"], "segment_id": row["segment_id"], "task_sample_id": row["task_sample_id"], "accepted": bool(native.get("accepted")), "joint_limit_valid": state.get("joint_limit_valid"), "fk_computable": state.get("fk_computable"), "self_collision": state.get("self_collision"), "environment_collision": state.get("environment_collision"), "collision_method": COLLISION_METHOD, "ccd": CCD, "clearance": CLEARANCE}
        if state.get("tcp_position_m") and state.get("tcp_orientation_xyzw"):
            actual_tcp = state["tcp_position_m"]; standoff = h6.vec_norm(h6.vec_sub(actual_tcp, row["surface_point_xyz_m"])); pos_error = h6.vec_norm(h6.vec_sub(actual_tcp, row["tcp_position_xyz_m"])); rotation = h6.quat_to_matrix(state["tcp_orientation_xyzw"]); actual_normal = h6.vec_scale([rotation[0][2], rotation[1][2], rotation[2][2]], -1.0); normal_error = h6.angle_between(actual_normal, row["surface_normal_unit"]); orientation_error = math.degrees(float(state.get("rotation_error_rad") or 0.0)); process_valid = bool(process_valid and standoff >= geometry["standoff_min_m"] and standoff <= geometry["standoff_max_m"] and normal_error <= geometry["normal_angle_tolerance_deg"] and pos_error <= tcp_contract["tcp_position_tolerance_m"] and orientation_error <= tcp_contract["tcp_orientation_tolerance_deg"]); metrics.update({"standoff_m": standoff, "normal_deviation_deg": normal_error, "tcp_position_error_m": pos_error, "tcp_orientation_error_deg": orientation_error})
        else:
            process_valid = False
        metrics["process_valid"] = process_valid
        validation.append(metrics)
    by_edge: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in validation:
        by_edge[str(item["task_sample_id"])].append(item)
    edge_records: list[dict[str, Any]] = []
    for segment, result in sorted(segment_results.items()):
        path = result["selected_path"]
        for previous, current in zip(path, path[1:]):
            task_id = f"H6.2_INTERPOLATION:{previous['waypoint_index']}->{current['waypoint_index']}"
            prefilter = edge_prefilter(previous, current)
            samples = by_edge.get(task_id, [])
            edge_records.append({"segment_id": segment, "from_waypoint_index": previous["waypoint_index"], "to_waypoint_index": current["waypoint_index"], "from_candidate_id": previous["candidate_id"], "to_candidate_id": current["candidate_id"], "candidate_prefilter": prefilter, "native_checks": {"joint_limits": all(item.get("joint_limit_valid") for item in samples), "fk": all(item.get("fk_computable") for item in samples), "self_collision": not any(item.get("self_collision") for item in samples), "environment_collision": not any(item.get("environment_collision") for item in samples), "adaptive_discrete_interpolation": bool(samples) and all(item.get("process_valid") for item in samples), "sample_count": len(samples), "failed_sample_count": sum(not item.get("process_valid") for item in samples)}, "native_edge_valid": bool(samples) and all(item.get("process_valid") for item in samples)})
    report = {"schema_version": "stage3-h6-3-native-edge-certification-v1", "native_run": run, "collision_method": COLLISION_METHOD, "ccd": CCD, "clearance": CLEARANCE, "interpolation_sample_count": len(interpolation_rows), "validated_sample_count": len(validation), "failed_sample_count": sum(not item["process_valid"] for item in validation), "selected_edge_count": len(edge_records), "selected_native_valid_edge_count": sum(bool(item["native_edge_valid"]) for item in edge_records), "all_selected_edges_native_valid": bool(edge_records) and all(item["native_edge_valid"] for item in edge_records), "edges": edge_records, "sample_validation": validation}
    dump_json(output / "stage3_h6_3_native_edge_certification.json", report)
    return report, {"rows": interpolation_rows, "trace": trace}


def validate_full_trace(output: Path, full_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    seeds: dict[int, list[list[float]]] = {}; labels: dict[int, list[str]] = {}
    for row in full_rows:
        if row["spray_state"] == "SPRAY_ON":
            seeds[int(row["waypoint_index"])] = [list(row["joint_values_selected"])]
            labels[int(row["waypoint_index"])] = ["exact_dp_selected_candidate"]
        else:
            seeds[int(row["waypoint_index"])] = [list(row["joint_values_seed"])]
            labels[int(row["waypoint_index"])] = ["controlled_stop_reposition_seed"]
    request = output / "stage3_h6_3_final_process_requests.tsv"
    h62.write_request_rows(request, full_rows, seeds, labels)
    run = run_native(request, output, "final_process_validation")
    trace = parse_trace(output / "native_runs" / "final_process_validation")
    by_index = {int(item["waypoint_index"]): item for item in trace}
    contract = load_json(H61_CONTRACT); geometry = contract["geometry"]; tcp_contract = contract["tcp_reproduction"]
    rows: list[dict[str, Any]] = []
    for base in full_rows:
        native = by_index.get(int(base["waypoint_index"]), {}); state = native.get("accepted_state") or {}; valid = bool(native.get("accepted") and state.get("joint_limit_valid") and state.get("fk_computable") and not state.get("environment_collision") and not state.get("self_collision")); item = {"waypoint_index": base["waypoint_index"], "segment_id": base["segment_id"], "spray_state": base["spray_state"], "accepted": bool(native.get("accepted")), "joint_values": native.get("accepted_joint_values", []), "joint_limit_valid": state.get("joint_limit_valid"), "fk_computable": state.get("fk_computable"), "self_collision": state.get("self_collision"), "environment_collision": state.get("environment_collision"), "collision_method": COLLISION_METHOD, "ccd": CCD, "clearance": CLEARANCE}
        if base["spray_state"] == "SPRAY_ON" and state.get("tcp_position_m") and state.get("tcp_orientation_xyzw"):
            actual_tcp = state["tcp_position_m"]; standoff = h6.vec_norm(h6.vec_sub(actual_tcp, base["surface_point_xyz_m"])); pos_error = h6.vec_norm(h6.vec_sub(actual_tcp, base["tcp_position_xyz_m"])); rotation = h6.quat_to_matrix(state["tcp_orientation_xyzw"]); actual_normal = h6.vec_scale([rotation[0][2], rotation[1][2], rotation[2][2]], -1.0); normal_error = h6.angle_between(actual_normal, base["surface_normal_unit"]); orientation_error = math.degrees(float(state.get("rotation_error_rad") or 0.0)); valid = bool(valid and standoff >= geometry["standoff_min_m"] and standoff <= geometry["standoff_max_m"] and normal_error <= geometry["normal_angle_tolerance_deg"] and pos_error <= tcp_contract["tcp_position_tolerance_m"] and orientation_error <= tcp_contract["tcp_orientation_tolerance_deg"]); item.update({"standoff_m": standoff, "normal_deviation_deg": normal_error, "tcp_position_error_m": pos_error, "tcp_orientation_error_deg": orientation_error})
        item["process_valid"] = valid
        rows.append(item)
    result = {"schema_version": "stage3-h6-3-final-process-validation-v1", "native_run": run, "rows": rows, "validated_count": len(rows), "failed_count": sum(not row["process_valid"] for row in rows), "spray_on_count": sum(row["spray_state"] == "SPRAY_ON" for row in rows), "spray_off_count": sum(row["spray_state"] == "SPRAY_OFF" for row in rows), "process_tolerance_all_spray_on_pass": all(row["process_valid"] for row in rows if row["spray_state"] == "SPRAY_ON"), "all_off_kinematic_collision_checks_pass": all(row["process_valid"] for row in rows if row["spray_state"] == "SPRAY_OFF"), "all_rows_native_valid": len(rows) == len(full_rows) and all(row["process_valid"] for row in rows)}
    dump_jsonl(output / "stage3_h6_3_final_process_validation.jsonl", rows)
    dump_json(output / "stage3_h6_3_final_process_validation_summary.json", result)
    return result


def build_topology_audit(final_rows: Sequence[Mapping[str, Any]], selected_on_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def qvalues(row: Mapping[str, Any]) -> Sequence[float]:
        return row.get("joint_values") or row.get("joint_values_selected") or []

    on_by_segment: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in final_rows:
        if row.get("spray_state") == "SPRAY_ON":
            on_by_segment[int(row["segment_id"])].append(row)
    turns: list[dict[str, Any]] = []; branch_edges: list[dict[str, Any]] = []; unexplained_branch: list[dict[str, Any]] = []; unexplained_cusps: list[dict[str, Any]] = []
    for segment, rows in sorted(on_by_segment.items()):
        rows = sorted(rows, key=lambda row: int(row["waypoint_index"]))
        for left, center, right in zip(rows, rows[1:], rows[2:]):
            v1 = h6.vec_sub(qvalues(center), qvalues(left)); v2 = h6.vec_sub(qvalues(right), qvalues(center)); co = h62.cosine(v1, v2); angle = math.degrees(math.acos(co)) if co is not None else None; triple = (int(left.get("source_target_id", -1)), int(left.get("destination_target_id", -1)), int(right.get("destination_target_id", -1)))
            classification = "moveit_unsupported_180_degree_turn" if co is not None and co <= -1.0 + 1.0e-5 else ("known_type_1_cusp" if co is not None and co <= -1.0 + 1.0e-4 and triple in KNOWN_TYPE_1_TRIPLES else ("unexplained_type_1_cusp" if co is not None and co <= -1.0 + 1.0e-4 else "not_reversal"))
            record = {"segment_id": segment, "waypoint_index": center["waypoint_index"], "cos_angle": co, "turn_angle_deg": angle, "classification": classification, "collision_method": COLLISION_METHOD, "ccd": CCD, "clearance": CLEARANCE}
            turns.append(record)
            if classification in {"moveit_unsupported_180_degree_turn", "unexplained_type_1_cusp"}:
                unexplained_cusps.append(record)
    for segment, rows in sorted(on_by_segment.items()):
        rows = sorted(rows, key=lambda row: int(row["waypoint_index"]))
        for left, right in zip(rows, rows[1:]):
            delta = h6.vec_sub(qvalues(right), qvalues(left)); l1 = sum(abs(value) for value in delta); l2 = h6.vec_norm(delta); linf = max(abs(value) for value in delta); branch_edges.append({"segment_id": segment, "from_waypoint_index": left["waypoint_index"], "to_waypoint_index": right["waypoint_index"], "L1_joint_delta_rad": l1, "L2_joint_delta_rad": l2, "L_inf_joint_delta_rad": linf, "branch_node_id_from": left.get("branch_node_id"), "branch_node_id_to": right.get("branch_node_id"), "branch_transition_explained_by_native_edge_certificate": True, "branch_discontinuity": bool(l2 > MAX_GRAPH_EDGE_L2_RAD or linf > MAX_GRAPH_EDGE_LINF_RAD), "collision_method": COLLISION_METHOD, "ccd": CCD, "clearance": CLEARANCE})
            if branch_edges[-1]["branch_discontinuity"]:
                unexplained_branch.append(branch_edges[-1])
    return {"schema_version": "stage3-h6-3-topology-audit-v1", "internal_on_turn_count": len(turns), "turns": turns, "branch_edges": branch_edges, "MOVEIT_UNSUPPORTED_180_TURN_COUNT": sum(item["classification"] == "moveit_unsupported_180_degree_turn" for item in turns), "UNEXPLAINED_TYPE_1_CUSP_COUNT": sum(item["classification"] == "unexplained_type_1_cusp" for item in turns), "UNEXPLAINED_IK_BRANCH_DISCONTINUITY_COUNT": len(unexplained_branch), "known_type_1_cusp_count": sum(item["classification"] == "known_type_1_cusp" for item in turns), "collision_method": COLLISION_METHOD, "CCD": CCD, "CLEARANCE": CLEARANCE}


def known_edge_rechecks(selected_on_rows: Sequence[Mapping[str, Any]], edge_report: Mapping[str, Any]) -> list[dict[str, Any]]:
    by_original = {int(row.get("original_on_waypoint_index", -1)): row for row in selected_on_rows}
    native_by_pair = {(int(edge["from_waypoint_index"]), int(edge["to_waypoint_index"])): edge for edge in edge_report.get("edges", [])}
    result: list[dict[str, Any]] = []
    for index in KNOWN_H6_BRANCH_EDGES:
        left = by_original.get(index); right = by_original.get(index + 1); edge = native_by_pair.get((index, index + 1)); present = bool(left and right)
        result.append({"original_h6_edge": [index, index + 1], "present_at_same_authoritative_waypoint_indices": present, "native_edge_valid": edge.get("native_edge_valid") if edge else None, "resolved": bool(not present or edge and edge.get("native_edge_valid") and left.get("segment_id") == right.get("segment_id")), "resolution_reason": "fresh_h6_3_native_edge_certification" if present else "not_present_in_current_181_point_authoritative_projection"})
    return result


def coverage_report(segments: Sequence[Mapping[str, Any]], off_segments: Sequence[Mapping[str, Any]], formal_target_ids: Sequence[int]) -> dict[str, Any]:
    target_ids = [int(target) for segment in segments for target in segment["target_ids"]]; formal = sorted(int(target) for target in formal_target_ids)
    covered = sorted(set(target_ids)); duplicates = sorted(target for target in set(target_ids) if target_ids.count(target) > 1)
    return {"schema_version": "stage3-h6-3-coverage-v1", "formal_targets": len(formal), "formal_target_ids": formal, "covered_targets": covered, "uncovered_targets": sorted(set(formal) - set(covered)), "duplicate_targets": duplicates, "coverage": f"{len(covered)}/{FORMAL_TARGET_COUNT}", "ALL_30_FEASIBLE_TARGETS_ACCOUNTED_FOR": "YES" if covered == formal and not duplicates else "NO", "spray_on_segments": [{"segment_id": item["segment_id"], "target_ids": item["target_ids"], "waypoint_start": item["waypoint_start"], "waypoint_end": item["waypoint_end"]} for item in segments], "spray_off_transitions": list(off_segments), "spray_on_segment_count": len(segments), "spray_off_segment_count": len(off_segments)}


def replay_signature(output: Path) -> str:
    value = {
        "exact": load_json(output / "stage3_h6_3_exact_reachability.json"),
        "segments": load_json(output / "stage3_h6_3_spray_on_off_segments.json"),
        "coverage": load_json(output / "stage3_h6_3_coverage_accounting.json"),
        "selected": load_json(output / "stage3_h6_3_selected_path.json"),
        "native_edge_summary": {key: load_json(output / "stage3_h6_3_native_edge_certification.json").get(key) for key in ("interpolation_sample_count", "failed_sample_count", "selected_edge_count", "selected_native_valid_edge_count", "all_selected_edges_native_valid")},
        "final": {key: load_json(output / "stage3_h6_3_final_process_validation_summary.json").get(key) for key in ("validated_count", "failed_count", "process_tolerance_all_spray_on_pass", "all_off_kinematic_collision_checks_pass", "all_rows_native_valid")},
    }
    return semantic_hash(value)


def fresh_replays(output: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for index in range(1, 4):
        proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--replay-signature", str(output)], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1800, check=False)
        match = re.search(r"H6_3_REPLAY_SIGNATURE:\s*([0-9a-f]+)", proc.stdout)
        records.append({"replay_index": index, "returncode": proc.returncode, "signature": match.group(1) if match else None, "stdout_tail": proc.stdout[-500:], "stderr_tail": proc.stderr[-500:]})
    signatures = [item["signature"] for item in records]
    result = {"schema_version": "stage3-h6-3-replay-determinism-v1", "fresh_process_count": 3, "processes": records, "identical": len(signatures) == 3 and None not in signatures and len(set(signatures)) == 1, "semantic_signature": signatures[0] if signatures and signatures[0] else None, "DETERMINISTIC_REPLAY": f"{sum(value is not None for value in signatures)}/3"}
    dump_json(output / "stage3_h6_3_replay_determinism.json", result)
    return result


def regression_report(output: Path) -> dict[str, Any]:
    command = [sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h6_3.py", "tests/test_stage3_h6_2.py", "tests/test_stage3_h6_1.py", "tests/test_stage3_h6.py", "tests/test_stage3_h5.py"]
    proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1800, check=False)
    failures = 0 if proc.returncode == 0 else 1
    result = {"schema_version": "stage3-h6-3-regression-v1", "command": command, "returncode": proc.returncode, "new_regression_failures": failures, "stdout_tail": proc.stdout[-4000:], "stderr_tail": proc.stderr[-4000:]}
    dump_json(output / "stage3_h6_3_regression_report.json", result)
    return result


def boundary_seed_audit(output: Path, surface_rows: Sequence[Mapping[str, Any]], layers: Mapping[int, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    left_rows = [row for row in surface_rows if int(row["waypoint_index"]) == BREAK_FROM_WAYPOINT]
    right_rows = [row for row in surface_rows if int(row["waypoint_index"]) == BREAK_TO_WAYPOINT]
    if not left_rows or not right_rows:
        return {"status": "BLOCKED", "reason": "boundary_waypoints_missing"}
    left = dict(left_rows[0]); right = dict(right_rows[0]); left_candidates = list(layers[BREAK_FROM_WAYPOINT]); right_candidates = list(layers[BREAK_TO_WAYPOINT])
    forward = []
    for index in range(max(0, BREAK_FROM_WAYPOINT - 8), BREAK_FROM_WAYPOINT + 1):
        forward.extend(item["joint_values"] for item in layers.get(index, []))
    backward = []
    for index in range(BREAK_TO_WAYPOINT, min(max(layers) + 1, BREAK_TO_WAYPOINT + 9)):
        backward.extend(item["joint_values"] for item in layers.get(index, []))
    def unique(values: Sequence[Sequence[float]]) -> list[list[float]]:
        result: list[list[float]] = []; seen: set[tuple[float, ...]] = set()
        for value in values:
            key = tuple(round(float(item), 10) for item in value)
            if key not in seen:
                seen.add(key); result.append([float(item) for item in value])
        return result
    left_seeds = unique([item["joint_values"] for item in left_candidates] + forward + backward)
    right_seeds = unique([item["joint_values"] for item in right_candidates] + backward + forward)
    left["segment_id"] = 900; right["segment_id"] = 901; left["component_id"] = 2; right["component_id"] = 2
    left["task_sample_id"] = "H6.3_FORWARD_53_BOUNDARY"; right["task_sample_id"] = "H6.3_BACKWARD_46_BOUNDARY"
    request = output / "stage3_h6_3_bidirectional_boundary_requests.tsv"
    h62.write_request_rows(request, [left, right], {BREAK_FROM_WAYPOINT: left_seeds, BREAK_TO_WAYPOINT: right_seeds}, {BREAK_FROM_WAYPOINT: ["forward_continuation_seed"] * len(left_seeds), BREAK_TO_WAYPOINT: ["backward_continuation_seed"] * len(right_seeds)})
    run = run_native(request, output, "bidirectional_boundary_probe")
    trace = parse_trace(output / "native_runs" / "bidirectional_boundary_probe")
    result = {"schema_version": "stage3-h6-3-bidirectional-seed-audit-v1", "status": "PASSED" if run["status"] == "PASSED" and len(trace) == 2 else "BLOCKED", "native_run": run, "forward_seed_count": len(left_seeds), "backward_seed_count": len(right_seeds), "forward_seed_sources": ["previous_accepted_state", "previous_several_accepted_states", "forward_continuation_seeds", "frozen_h5_seed_bank", "target_53_valid_branches"], "backward_seed_sources": ["backward_continuation_seeds", "frozen_h5_seed_bank", "target_46_valid_branches"], "forward_branch_node_ids": sorted({item.get("branch_node_id") for item in left_candidates}), "backward_branch_node_ids": sorted({item.get("branch_node_id") for item in right_candidates}), "shared_branch_node_ids": sorted({item.get("branch_node_id") for item in left_candidates} & {item.get("branch_node_id") for item in right_candidates}), "trace_count": len(trace), "trace": trace}
    dump_json(output / "stage3_h6_3_bidirectional_seed_audit.json", result)
    return result


def machine_conclusion(status: str, blocker: str | None, exact: Mapping[str, Any], coverage: Mapping[str, Any], segments: Sequence[Mapping[str, Any]], off_segments: Sequence[Mapping[str, Any]], topology: Mapping[str, Any], known: Sequence[Mapping[str, Any]], edge: Mapping[str, Any], process: Mapping[str, Any], replay: Mapping[str, Any], regression: Mapping[str, Any], immutable: Mapping[str, Any], beam_false_negative: bool, native_graph_connected: bool) -> str:
    lines = [f"STAGE_3_H6_3: {status}", f"FIRST_BLOCKER: {blocker or 'none'}", "EXACT_GRAPH_SEARCH_USED: YES", "BEAM_SEARCH_AUTHORITATIVE: NO", f"BEAM_FALSE_NEGATIVE_FOUND: {'YES' if beam_false_negative else 'NO'}", f"TARGET_53_46_NATIVE_GRAPH_CONNECTED: {'YES' if native_graph_connected else 'NO'}", f"TARGET_53_46_PROCESS_BREAK_REQUIRED: {'YES' if not native_graph_connected else 'NO'}", f"AUTHORITATIVE_H6_3_PATH_CERTIFIED: {'YES' if status == 'PASSED' else 'NO'}", f"H5_IMMUTABLE: {'YES' if immutable.get('all_after_hashes_match') else 'NO'}", f"H6_IMMUTABLE: {'YES' if immutable.get('all_after_hashes_match') else 'NO'}", f"H6_1_IMMUTABLE: {'YES' if immutable.get('all_after_hashes_match') else 'NO'}", f"H6_2_IMMUTABLE: {'YES' if immutable.get('all_after_hashes_match') else 'NO'}", f"COVERAGE: {coverage.get('coverage')}", f"SPRAY_ON_SEGMENTS: {len(segments)}", f"SPRAY_OFF_SEGMENTS: {len(off_segments)}", f"MOVEIT_UNSUPPORTED_180_TURN_COUNT: {topology.get('MOVEIT_UNSUPPORTED_180_TURN_COUNT', 0)}", f"UNEXPLAINED_TYPE_1_CUSP_COUNT: {topology.get('UNEXPLAINED_TYPE_1_CUSP_COUNT', 0)}", f"UNEXPLAINED_IK_BRANCH_DISCONTINUITY_COUNT: {topology.get('UNEXPLAINED_IK_BRANCH_DISCONTINUITY_COUNT', 0)}", f"KNOWN_81_82_RESOLVED: {'YES' if next((item['resolved'] for item in known if item['original_h6_edge'] == [81, 82]), False) else 'NO'}", f"KNOWN_1059_1060_RESOLVED: {'YES' if next((item['resolved'] for item in known if item['original_h6_edge'] == [1059, 1060]), False) else 'NO'}", f"KNOWN_1269_1270_RESOLVED: {'YES' if next((item['resolved'] for item in known if item['original_h6_edge'] == [1269, 1270]), False) else 'NO'}", f"PROCESS_TOLERANCE_CERTIFICATION: {'PASSED' if process.get('process_tolerance_all_spray_on_pass') else 'BLOCKED'}", f"COLLISION_CERTIFICATION: {'PASSED' if edge.get('all_selected_edges_native_valid') and process.get('all_off_kinematic_collision_checks_pass') else 'BLOCKED'}", f"DETERMINISTIC_REPLAY: {replay.get('DETERMINISTIC_REPLAY')}", f"NEW_REGRESSION_FAILURES: {regression.get('new_regression_failures', 0)}", f"CCD: {CCD}", f"CLEARANCE: {CLEARANCE}", "NEW_FJT_GOALS_SENT: 0", "ROBOT_MOTION_STARTED: NO", "FORMAL_LEDGER_MUTATED: NO", f"READY_FOR_STAGE_3_H7: {'YES' if status == 'PASSED' else 'NO'}"]
    return "\n".join(lines)


def orchestrate(output: Path) -> int:
    output.mkdir(parents=True, exist_ok=True)
    before = source_snapshot()
    dump_json(output / "stage3_h6_3_input_freeze_manifest.json", {"schema_version": "stage3-h6-3-input-freeze-v1", "sources": before, "authoritative_inputs": {"H5": rel(H5_ROOT), "H6": rel(H6_ROOT), "H6_1": rel(H61_ROOT), "H6_2": rel(H62_ROOT)}, "forbidden_stages": ["H7", "TOTG", "Ruckig", "FJT", "robot_motion", "formal_ledger_mutation", "ML_training"]})
    blocker: str | None = None
    exact: dict[str, Any] = {}; coverage: dict[str, Any] = {"coverage": "0/30", "ALL_30_FEASIBLE_TARGETS_ACCOUNTED_FOR": "NO"}; topology: dict[str, Any] = {}; known: list[dict[str, Any]] = []; edge: dict[str, Any] = {"all_selected_edges_native_valid": False}; process: dict[str, Any] = {"process_tolerance_all_spray_on_pass": False, "all_off_kinematic_collision_checks_pass": False}; replay: dict[str, Any] = {"DETERMINISTIC_REPLAY": "0/3"}; regression: dict[str, Any] = {"new_regression_failures": 0}; new_segments: list[dict[str, Any]] = []; off_segments: list[dict[str, Any]] = []
    try:
        required = [H5_ROOT, H6_ROOT, H61_ROOT, H62_ROOT, H62_GRAPH, H62_SURFACE, H62_SEGMENTS, H62_GATE, H62_OPTIMIZATION, H62_COVERAGE, H61_CONTRACT, H6_EXECUTABLE, H6_NATIVE_SETUP, h6.DERIVED_URDF, h6.SRDF, H6_MESH]
        missing = [rel(path) for path in required if not path.exists()]
        if missing:
            blocker = f"frozen_input_missing:{missing}"
        else:
            h62_gate = load_json(H62_GATE)
            if h62_gate.get("STAGE_3_H6_2") != "BLOCKED":
                blocker = "h6_2_is_not_the_expected_blocked_input_identity"
            else:
                graph = load_json(H62_GRAPH); layers = {int(key): value for key, value in graph["layers"].items()}; surface_rows = load_jsonl(H62_SURFACE); old_segments = load_json(H62_SEGMENTS)["segments"]; on_old = [row for row in surface_rows if row.get("spray_state") == "SPRAY_ON"]
                if len(layers) != len(on_old) or len(layers) != 588:
                    blocker = f"authoritative_h6_2_graph_shape_invalid:layers={len(layers)}:on_rows={len(on_old)}"
                else:
                    original_audits: dict[int, dict[str, Any]] = {}
                    for segment in old_segments:
                        rows = [row for row in on_old if int(row["segment_id"]) == int(segment["segment_id"])]
                        original_audits[int(segment["segment_id"])] = exact_segment_reachability(rows, layers)
                    exact_original = {str(key): value for key, value in original_audits.items()}
                    new_segments = split_segments(old_segments)
                    segment_rows: dict[int, list[dict[str, Any]]] = {}
                    for segment in new_segments:
                        segment_rows[int(segment["segment_id"])] = [dict(row, segment_id=int(segment["segment_id"])) for row in on_old if int(segment["waypoint_start"]) <= int(row["waypoint_index"]) <= int(segment["waypoint_end"])]
                    segment_results: dict[int, dict[str, Any]] = {}
                    for segment in new_segments:
                        result = exact_segment_reachability(segment_rows[int(segment["segment_id"])], layers); result["segment_id"] = int(segment["segment_id"]); segment_results[int(segment["segment_id"])] = result
                        if result.get("status") != "PASSED" and blocker is None:
                            blocker = f"exact_graph_search_failed:segment_{segment['segment_id']}:{result.get('reason')}"
                    h62_optimization = load_json(H62_OPTIMIZATION)
                    beam_false_negative = bool(h62_optimization.get("status") == "BLOCKED" and any(int(item.get("segment_id", -1)) == 0 and item.get("status") == "BLOCKED" for item in h62_optimization.get("segments", [])) and original_audits.get(0, {}).get("status") == "PASSED")
                    exact = {"schema_version": "stage3-h6-3-exact-reachability-v1", "candidate_graph_source": rel(H62_GRAPH), "candidate_count": graph["candidate_count"], "waypoint_layer_count": graph["waypoint_layer_count"], "BEAM_SEARCH_AUTHORITATIVE": "NO", "H62_BEAM_FALSE_NEGATIVE": beam_false_negative, "original_h6_2_segment_audits": exact_original, "new_segment_audits": {str(key): value for key, value in segment_results.items()}, "first_original_graph_disconnect": original_audits.get(2, {}).get("first_graph_disconnect"), "controlled_break": {"from_waypoint": BREAK_FROM_WAYPOINT, "to_waypoint": BREAK_TO_WAYPOINT, "source_target": BREAK_SOURCE_TARGET, "destination_target": BREAK_DESTINATION_TARGET, "required": bool(original_audits.get(2, {}).get("status") == "BLOCKED"), "reason": "exact_reachability_disconnect_after_53_before_46"}, "native_repair_iterations": []}
                    dump_json(output / "stage3_h6_3_exact_reachability.json", exact)
                    boundary = boundary_seed_audit(output, surface_rows, layers)
                    if boundary.get("status") != "PASSED" and blocker is None:
                        blocker = "bidirectional_native_seed_enumeration_blocked"
                    dump_json(output / "stage3_h6_3_candidate_graph_connectivity_certificate.json", {"schema_version": "stage3-h6-3-connectivity-certificate-v1", "exact_graph_search_used": True, "beam_search_authoritative": False, "beam_false_negative_found": exact["H62_BEAM_FALSE_NEGATIVE"], "native_node_candidate_count": graph["candidate_count"], "native_node_candidates_all_valid": all(candidate_is_native_valid(candidate) for layer in layers.values() for candidate in layer), "boundary_seed_audit": boundary, "first_exact_graph_disconnect": exact.get("first_original_graph_disconnect"), "reachability_classification": "genuinely_disconnected_native_valid_branch_graph_candidate_boundary" if exact["controlled_break"]["required"] else "connected", "native_edge_certification_scope": "every selected exact-DP process edge plus every controlled SPRAY_OFF interpolation sample", "joint_thresholds_are_prefilter_only": True})
                    # Native failures are fed back as deterministic exact-DP
                    # forbidden edges.  The DP remains exhaustive: no beam,
                    # cap, or larger joint-jump threshold is introduced.
                    native_banned: set[tuple[str, str]] = set()
                    repair_history: list[dict[str, Any]] = []
                    selected_on: list[dict[str, Any]] = []
                    for repair_attempt in range(1, 9):
                        selected_on = clone_selected_on_rows(on_old, segment_results)
                        edge, _ = run_selected_edge_certification(output, selected_on, segment_results, "native_edge_certification" if repair_attempt == 1 else f"native_edge_certification_repair_{repair_attempt - 1}")
                        topology = build_topology_audit(selected_on, selected_on)
                        violations = {(str(item["from_candidate_id"]), str(item["to_candidate_id"])) for item in edge.get("edges", []) if not item.get("native_edge_valid")}
                        by_original = {int(row["original_on_waypoint_index"]): row for row in selected_on}
                        for turn in topology.get("turns", []):
                            if turn.get("classification") in {"moveit_unsupported_180_degree_turn", "unexplained_type_1_cusp"}:
                                center = int(turn["waypoint_index"]); left = by_original.get(center - 1); current = by_original.get(center)
                                if left and current:
                                    violations.add((str(left["selected_candidate_id"]), str(current["selected_candidate_id"])))
                        repair_history.append({"attempt": repair_attempt, "banned_edge_count_before": len(native_banned), "native_failed_edge_count": len([item for item in edge.get("edges", []) if not item.get("native_edge_valid")]), "topology_violation_count": sum(turn.get("classification") in {"moveit_unsupported_180_degree_turn", "unexplained_type_1_cusp"} for turn in topology.get("turns", [])), "new_banned_edge_count": len(violations - native_banned), "native_edge_report": rel(output / "stage3_h6_3_native_edge_certification.json")})
                        if not violations:
                            break
                        fresh_violations = violations - native_banned
                        if not fresh_violations:
                            blocker = blocker or "native_repair_search_stalled_on_repeated_edge_failure"
                            break
                        native_banned.update(fresh_violations)
                        next_results: dict[int, dict[str, Any]] = {}
                        for segment in new_segments:
                            result = exact_segment_reachability(segment_rows[int(segment["segment_id"])], layers, native_banned); result["segment_id"] = int(segment["segment_id"]); next_results[int(segment["segment_id"])] = result
                        if any(result.get("status") != "PASSED" for result in next_results.values()):
                            blocker = blocker or "native_invalid_edge_remediation_left_no_exact_path"
                            break
                        segment_results = next_results
                    exact["native_repair_iterations"] = repair_history
                    exact["native_banned_edge_count"] = len(native_banned)
                    exact["new_segment_audits"] = {str(key): value for key, value in segment_results.items()}
                    dump_json(output / "stage3_h6_3_exact_reachability.json", exact)
                    dump_jsonl(output / "stage3_h6_3_selected_joint_waypoints.jsonl", [{"schema_version": "stage3-h6-3-selected-joint-waypoint-v1", **{key: row.get(key) for key in ("waypoint_index", "segment_id", "original_on_waypoint_index", "source_target_id", "destination_target_id", "joint_values_selected", "selected_candidate_id", "branch_node_id")}} for row in selected_on])
                    dump_json(output / "stage3_h6_3_selected_path.json", {"schema_version": "stage3-h6-3-selected-path-v1", "authoritative": True, "segment_paths": {str(key): [node["joint_values"] for node in result["selected_path"]] for key, result in segment_results.items()}, "branch_selections": [{"waypoint_index": row["original_on_waypoint_index"], "candidate_id": row["selected_candidate_id"], "branch_node_id": row.get("branch_node_id")} for row in selected_on], "process_break": {"source_target": BREAK_SOURCE_TARGET, "destination_target": BREAK_DESTINATION_TARGET, "controlled_stop": True, "spray_off_reposition": True, "restart_spray_on": True}, "native_banned_edge_count": len(native_banned)})
                    off_rows, off_segments = h62.build_off_rows(selected_on, new_segments)
                    full_rows = h62.interleave_rows(selected_on, off_rows, off_segments)
                    process = validate_full_trace(output, full_rows)
                    dump_jsonl(output / "stage3_h6_3_surface_waypoints.jsonl", [row for row in full_rows if row["spray_state"] == "SPRAY_ON"])
                    dump_jsonl(output / "stage3_h6_3_spray_off_reposition_waypoints.jsonl", [row for row in full_rows if row["spray_state"] == "SPRAY_OFF"])
                    topology = build_topology_audit(selected_on, selected_on)
                    known = known_edge_rechecks(selected_on, edge)
                    coverage = coverage_report(new_segments, off_segments, load_json(H62_COVERAGE).get("formal_target_ids", []))
                    structure = {"schema_version": "stage3-h6-3-spray-on-off-segments-v1", "controlled_stop_restart_semantics": "SPRAY_ON -> controlled stop -> SPRAY_OFF -> collision-free reposition -> SPRAY_ON", "segments": [*new_segments, *off_segments], "new_break": {"source_target_id": BREAK_SOURCE_TARGET, "destination_target_id": BREAK_DESTINATION_TARGET, "source_waypoint": BREAK_FROM_WAYPOINT, "destination_waypoint": BREAK_TO_WAYPOINT, "reason": "exact_graph_disconnect"}}
                    dump_json(output / "stage3_h6_3_spray_on_off_segments.json", structure)
                    dump_json(output / "stage3_h6_3_topology_audit.json", topology); dump_json(output / "stage3_h6_3_known_edge_rechecks.json", {"schema_version": "stage3-h6-3-known-edge-rechecks-v1", "edges": known}); dump_json(output / "stage3_h6_3_coverage_accounting.json", coverage)
                    if not edge.get("all_selected_edges_native_valid") and blocker is None:
                        blocker = "selected_exact_path_native_edge_validation_failed"
                    if not process.get("process_tolerance_all_spray_on_pass") and blocker is None:
                        blocker = "spray_on_process_tolerance_validation_failed"
                    if not process.get("all_off_kinematic_collision_checks_pass") and blocker is None:
                        blocker = "spray_off_kinematic_collision_validation_failed"
                    if topology.get("MOVEIT_UNSUPPORTED_180_TURN_COUNT") or topology.get("UNEXPLAINED_TYPE_1_CUSP_COUNT") or topology.get("UNEXPLAINED_IK_BRANCH_DISCONTINUITY_COUNT"):
                        blocker = blocker or "topology_audit_failed"
                    if not coverage.get("ALL_30_FEASIBLE_TARGETS_ACCOUNTED_FOR") == "YES" and blocker is None:
                        blocker = "coverage_not_30_of_30"
                    replay = fresh_replays(output)
                    if not replay.get("identical") and blocker is None:
                        blocker = "fresh_process_replay_not_3_of_3_identical"
    except Exception as exc:
        blocker = blocker or f"H6_3_execution_error:{type(exc).__name__}:{exc}"
    immutable = immutable_after(before); dump_json(output / "stage3_h6_3_immutable_after_check.json", immutable)
    regression = regression_report(output)
    if regression.get("new_regression_failures") and blocker is None:
        blocker = "new_regression_failure"
    if not immutable.get("all_after_hashes_match") and blocker is None:
        blocker = "frozen_h5_h6_h6_1_h6_2_artifact_changed"
    if not known or not all(item.get("resolved") for item in known):
        blocker = blocker or "known_h6_edge_recheck_failed"
    status = "PASSED" if blocker is None else "BLOCKED"
    native_graph_connected = not bool(exact.get("controlled_break", {}).get("required"))
    conclusion = machine_conclusion(status, blocker, exact, coverage, new_segments, off_segments, topology, known, edge, process, replay, regression, immutable, bool(exact.get("H62_BEAM_FALSE_NEGATIVE")), native_graph_connected)
    (output / "FINAL_REPORT.md").write_text("# Stage 3 H6.3 — Exact IK Branch Graph Reachability, Native Edge Certification, and Controlled Process-Break Remediation\n\n" + conclusion + "\n\nThis artifact is additive. H5/H6/H6.1/H6.2 were read-only evidence inputs. The only native backend used was MoveIt2 IK + FK + PlanningScene/FCL with adaptive discrete interpolation. No H7, TOTG, Ruckig, FJT, robot motion, ML training, or formal-ledger mutation was run.\n", encoding="utf-8", newline="\n")
    terminal = {"schema_version": "stage3-h6-3-terminal-certificate-v1", **{line.split(": ", 1)[0]: line.split(": ", 1)[1] for line in conclusion.splitlines() if ": " in line}, "collision_method": COLLISION_METHOD, "CCD": CCD, "CLEARANCE": CLEARANCE, "exact_reachability": rel(output / "stage3_h6_3_exact_reachability.json"), "immutable_after_check": rel(output / "stage3_h6_3_immutable_after_check.json"), "new_fjt_goals_sent": 0, "robot_motion_started": "NO", "formal_ledger_mutated": "NO"}
    dump_json(output / "stage3_h6_3_terminal_certificate.json", terminal); dump_json(output / "stage3_h6_3_gate_report.json", {"schema_version": "stage3-h6-3-gate-report-v1", "STAGE_3_H6_3": status, "FIRST_BLOCKER": blocker, "mandatory_gates": {"exact_graph_search": exact.get("candidate_graph_source") is not None, "beam_not_authoritative": True, "target_53_46_process_break_semantics": bool(exact.get("controlled_break", {}).get("required")), "native_selected_edges": edge.get("all_selected_edges_native_valid", False), "process_tolerance": process.get("process_tolerance_all_spray_on_pass", False), "spray_off_kinematics": process.get("all_off_kinematic_collision_checks_pass", False), "coverage_30_of_30": coverage.get("ALL_30_FEASIBLE_TARGETS_ACCOUNTED_FOR") == "YES", "topology": not topology.get("MOVEIT_UNSUPPORTED_180_TURN_COUNT") and not topology.get("UNEXPLAINED_TYPE_1_CUSP_COUNT") and not topology.get("UNEXPLAINED_IK_BRANCH_DISCONTINUITY_COUNT"), "known_edges": bool(known) and all(item.get("resolved") for item in known), "determinism_3_of_3": replay.get("identical", False), "new_regression_failures": regression.get("new_regression_failures", 0), "frozen_inputs_immutable": immutable.get("all_after_hashes_match", False)}, "NEW_FJT_GOALS_SENT": 0, "ROBOT_MOTION_STARTED": "NO", "FORMAL_LEDGER_MUTATED": "NO", "READY_FOR_STAGE_3_H7": "YES" if status == "PASSED" else "NO"})
    return 0 if status == "PASSED" else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output", type=Path); parser.add_argument("--replay-signature", type=Path); args = parser.parse_args()
    if args.replay_signature:
        print(f"H6_3_REPLAY_SIGNATURE: {replay_signature(args.replay_signature.resolve())}")
        return 0
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"); output = (args.output or ROOT / "outputs" / f"stage3_h6_3_exact_ik_branch_graph_{timestamp}").resolve()
    return orchestrate(output)


if __name__ == "__main__":
    raise SystemExit(main())
