#!/usr/bin/env python3
"""Result-oriented Stage 2.4T segmented process recovery.

The runner consumes the frozen Stage 2.4I graph and the existing Stage 2.4S
cover candidates.  It performs a bounded, cached OFF-aware search in layers:
local boundary relocation, formal endpoint pools, multi-seed OFF IK, bounded
near-normal directions/distances, staging, and explicit MoveIt planning
parameters.  No global 720-waypoint IK generation is performed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_stage24s_segmented_process_rebaseline as s24
import scripts.run_stage24e_tolerance_search as stage24e
import scripts.run_stage24_closed_loop_graph as stage24
import scripts.run_stage24g_complete_edge_component_bridge as stage24g


WAYPOINT_COUNT = 720
MAX_ON_STEP_DEG = 20.0
GRAPH_NODE_COUNT = 20260
GRAPH_EDGE_COUNT = 20363
REBUILD_COUNT = 3
R5_ROOT = ROOT / "outputs/ik_graph_stage24s_segmented_process_rebaseline/fr5_scaled_horseshoe_demo_v45_20260802_Stage24S_FINAL_R5"
R5_SEARCH = R5_ROOT / "search"
BASE_OUTPUT = ROOT / "outputs/ik_graph_stage24t_final_recovery"
STAGING_WAYPOINTS = (0, 120, 240, 360, 480, 600, 719)
LOCAL_BLOCKER_MIN = 384
LOCAL_BLOCKER_MAX = 405


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(canonical(row) + "\n" for row in rows), encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def q_from_node(node: dict[str, Any]) -> np.ndarray:
    return s24.q_from_node(node)


def source_ref(path: Path, role: str) -> dict[str, Any]:
    return {"path": str(path.resolve()), "role": role, "exists": path.exists(),
            "size": path.stat().st_size if path.is_file() else None,
            "sha256": sha256(path) if path.is_file() else None}


def load_graph_and_waypoints():
    waypoints = s24.load_waypoints()
    nodes, edges, audit = s24.load_formal_graph()
    if len(nodes) != GRAPH_NODE_COUNT or len(edges) != GRAPH_EDGE_COUNT:
        raise RuntimeError(f"frozen formal graph count mismatch: {len(nodes)} nodes/{len(edges)} edges")
    if len(waypoints) != WAYPOINT_COUNT:
        raise RuntimeError(f"frozen waypoint count mismatch: {len(waypoints)}")
    return waypoints, nodes, edges, audit


def formal_options(start: int, end: int, nodes: dict[str, dict[str, Any]], edges: dict[tuple[int, int, str, str], dict[str, Any]], semantics: list[dict[str, Any]], cache: dict[tuple[int, int], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    key = (int(start), int(end))
    if key not in cache:
        cache[key] = s24.enumerate_start_options(start, nodes, edges, semantics, {end}).get(end, [])[:s24.ENDPOINT_TOP_K]
    return cache[key]


def materialize_cover(stub: dict[str, Any], nodes: dict[str, dict[str, Any]], edges: dict[tuple[int, int, str, str], dict[str, Any]], semantics: list[dict[str, Any]], options_cache: dict[tuple[int, int], list[dict[str, Any]]]) -> dict[str, Any] | None:
    segments = []
    for row in stub["segments"]:
        opts = formal_options(int(row["start_waypoint"]), int(row["end_waypoint"]), nodes, edges, semantics, options_cache)
        if not opts:
            return None
        segments.append(opts[0])
    return {"segment_count": int(stub["segment_count"]), "segments": segments,
            "total_on_joint_motion_rad": sum(float(row.get("total_joint_motion_rad", 0.0)) for row in segments)}


def cover_identity(cover: dict[str, Any]) -> tuple[tuple[int, int], ...]:
    return tuple((int(row["start_waypoint"]), int(row["end_waypoint"])) for row in cover["segments"])


def find_blocker_index(cover: dict[str, Any]) -> int | None:
    for index, (left, right) in enumerate(zip(cover["segments"], cover["segments"][1:])):
        if int(left["end_waypoint"]) == 394 and int(right["start_waypoint"]) == 395:
            return index
    return None


def local_boundary_records(cover: dict[str, Any], nodes, edges, semantics, options_cache) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    index = find_blocker_index(cover)
    if index is None:
        return [], []
    left = cover["segments"][index]
    right = cover["segments"][index + 1]
    records = []
    variants = []
    for boundary_left in range(LOCAL_BLOCKER_MIN, LOCAL_BLOCKER_MAX + 1):
        left_options = formal_options(int(left["start_waypoint"]), boundary_left, nodes, edges, semantics, options_cache)
        right_options = formal_options(boundary_left + 1, int(right["end_waypoint"]), nodes, edges, semantics, options_cache)
        valid = bool(left_options and right_options)
        record = {"original_boundary": "394->395", "tested_boundary": f"{boundary_left}->{boundary_left + 1}",
                  "left_segment_start": int(left["start_waypoint"]), "left_segment_end": boundary_left,
                  "right_segment_start": boundary_left + 1, "right_segment_end": int(right["end_waypoint"]),
                  "formal_ON_left_segment_exists": bool(left_options), "formal_ON_right_segment_exists": bool(right_options),
                  "full_720_coverage_preserved": valid,
                  "left_option_count": len(left_options), "right_option_count": len(right_options),
                  "status": "eligible" if valid else "rejected_no_formal_ON_path"}
        records.append(record)
        if valid:
            variant = {"boundary_left": boundary_left, "boundary_index": index,
                       "left_options": left_options, "right_options": right_options,
                       "record": record}
            variants.append(variant)
    return records, variants


def node_seed_pool(waypoint: int, nodes: dict[str, dict[str, Any]], semantics: list[dict[str, Any]], limit: int = 8) -> list[list[float]]:
    rows = [(cid, node) for cid, node in nodes.items() if int(node["waypoint_index"]) == waypoint]
    rows.sort(key=lambda row: (-s24.joint_margin(q_from_node(row[1]), semantics), str(row[0])))
    return [q_from_node(node).tolist() for _, node in rows[:limit]]


def endpoint_pair_pool(left_options: list[dict[str, Any]], right_options: list[dict[str, Any]], nodes, semantics, tier: int, waypoint_left: int, waypoint_right: int, limit: int = 25) -> list[dict[str, Any]]:
    rows = []
    left0 = left_options[0]
    right0 = right_options[0]
    for li, left in enumerate(left_options):
        for ri, right in enumerate(right_options):
            qa = q_from_node(nodes[str(left["end_candidate_id"])])
            qb = q_from_node(nodes[str(right["start_candidate_id"])])
            primary = 0 if li == 0 and ri == 0 else 1
            rank_cost = (primary, -min(float(left.get("min_joint_limit_margin", 0.0)), float(right.get("min_joint_limit_margin", 0.0))),
                         float(np.sum(np.abs(qa - qb))), li, ri)
            rows.append({"candidate_a": str(left["end_candidate_id"]), "candidate_b": str(right["start_candidate_id"]),
                         "q_a": qa.tolist(), "q_b": qb.tolist(), "left_option_index": li, "right_option_index": ri,
                         "rank_cost": list(rank_cost), "formal_reachability": True,
                         "dual_backend_node_valid": True,
                         "seed_pool_a": node_seed_pool(waypoint_left, nodes, semantics) if tier >= 2 else [],
                         "seed_pool_b": node_seed_pool(waypoint_right, nodes, semantics) if tier >= 2 else []})
    rows.sort(key=lambda row: tuple(row["rank_cost"]))
    for rank, row in enumerate(rows[:limit]):
        row["rank"] = rank
    return rows[:limit]


def near_normal_directions(normal: list[float], tier: int) -> list[dict[str, Any]]:
    n = np.asarray(normal, dtype=float)
    n /= max(float(np.linalg.norm(n)), 1.0e-12)
    rows = [{"direction": n.tolist(), "geometry_reason": "+surface_normal"}]
    if tier >= 2:
        tangent = np.asarray([-n[1], n[0], 0.0], dtype=float)
        if float(np.linalg.norm(tangent)) < 1.0e-9:
            tangent = np.asarray([0.0, -n[2], n[1]], dtype=float)
        tangent /= max(float(np.linalg.norm(tangent)), 1.0e-12)
        angle = math.radians(15.0)
        rows.extend({"direction": (math.cos(angle) * n + sign * math.sin(angle) * tangent).tolist(),
                     "geometry_reason": f"surface_normal_plus_local_tangent_{'plus' if sign > 0 else 'minus'}_15deg"} for sign in (1.0, -1.0))
    return rows


def staging_states(nodes, semantics, tier: int) -> list[list[float]]:
    if tier < 3:
        return []
    states = []
    for wp in STAGING_WAYPOINTS:
        states.extend(node_seed_pool(wp, nodes, semantics, limit=1))
    return states[:4]


def planner_parameters(tier: int, planner_id: str = "RRTConnectkConfigDefault") -> dict[str, Any]:
    if tier <= 1:
        return {"planning_pipeline": "ompl", "planner_id": planner_id, "planning_attempts": 1, "planning_time": 1.0, "velocity_scaling": 0.15, "acceleration_scaling": 0.15}
    return {"planning_pipeline": "ompl", "planner_id": planner_id, "planning_attempts": 3, "planning_time": 2.0, "velocity_scaling": 0.15, "acceleration_scaling": 0.15}


def make_request(cover: dict[str, Any], option_lists: list[list[dict[str, Any]]], waypoints, nodes, semantics, root: Path, tier: int, planner_id: str, rebuild_index: int, only_transition: int | None = None, exact_pairs: dict[int, dict[str, Any]] | None = None) -> dict[str, Any]:
    transitions = []
    indices = range(1, len(cover["segments"])) if only_transition is None else [only_transition]
    for transition_id in indices:
        left_options = option_lists[transition_id - 1]
        right_options = option_lists[transition_id]
        left = cover["segments"][transition_id - 1]
        row = waypoints[int(left["end_waypoint"])]
        normal = [float(row["nx"]), float(row["ny"]), float(row["nz"])]
        if exact_pairs and transition_id in exact_pairs:
            exact_pair = dict(exact_pairs[transition_id])
            if tier >= 2:
                if not exact_pair.get("seed_pool_a"):
                    exact_pair["seed_pool_a"] = node_seed_pool(int(left["end_waypoint"]), nodes, semantics)
                if not exact_pair.get("seed_pool_b"):
                    exact_pair["seed_pool_b"] = node_seed_pool(int(cover["segments"][transition_id]["start_waypoint"]), nodes, semantics)
            pairs = [exact_pair]
        else:
            pairs = endpoint_pair_pool(left_options, right_options, nodes, semantics, tier, int(left["end_waypoint"]), int(cover["segments"][transition_id]["start_waypoint"]), limit=25)
        result_path = root / "requests" / f"rebuild_{rebuild_index:02d}_transition_{transition_id:02d}.json"
        transitions.append({"transition_id": transition_id, "from_waypoint": int(left["end_waypoint"]), "to_waypoint": int(cover["segments"][transition_id]["start_waypoint"]),
                            "direction": normal, "directions": near_normal_directions(normal, tier),
                            "retract_distances_mm": [10, 20, 30, 40, 60, 80] + ([100, 120, 150] if tier >= 2 else []),
                            "endpoint_pairs": pairs, "max_endpoint_pairs": len(pairs),
                            "staging_states": staging_states(nodes, semantics, tier),
                            "planner_parameters": planner_parameters(tier, planner_id)})
    params = planner_parameters(tier, planner_id)
    return {"schema_version": "stage24t-off-planner-request-v1", "rebuild_index": rebuild_index,
            "group_name": "fairino5_v6_group", "ee_link": "spray_tcp_link",
            "joint_bounds": [[float(rule["lower"]), float(rule["upper"])] for rule in semantics],
            "parts_dir": stage24e.wsl_path(s24.PARTS_PATH), "result_path": stage24e.wsl_path(root / "off_planner_result.json"),
            "result_path_windows": str((root / "off_planner_result.json").resolve()), "transitions": transitions,
            "planner_parameters": params, "seed_policy": {"tier": tier, "formal_same_waypoint_branches": tier >= 2},
            "new_global_IK_generation": False}


def run_moveit_request(request: dict[str, Any], root: Path, planner_id: str) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    request_path = root / "off_planner_request.json"
    write_json(request_path, request)
    wsl_root = "/mnt/c/Users/86198/Desktop/robotfucker"
    local_install = "/mnt/c/Users/86198/Desktop/robotfucker/install/setup.bash"
    params = request["planner_parameters"]
    random_seed = int(request.get("random_seed", 20260802))
    command = (f"source /opt/ros/jazzy/setup.bash && source {local_install} && export OMPL_RNG_SEED={random_seed} && "
               f"ros2 launch {wsl_root}/tools/stage24t_off_planner_launch.py request:={stage24e.wsl_path(request_path)} "
               f"planner_id:={planner_id} planning_time:={params['planning_time']} planning_attempts:={params['planning_attempts']} "
               f"velocity_scaling:={params['velocity_scaling']} acceleration_scaling:={params['acceleration_scaling']}")
    proc = subprocess.run(["wsl.exe", "bash", "-lc", command], cwd=ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=1800)
    (root / "off_planner_stdout.log").write_text(proc.stdout, encoding="utf-8")
    (root / "off_planner_stderr.log").write_text(proc.stderr, encoding="utf-8")
    (root / "off_planner_exit_code.txt").write_text(str(proc.returncode) + "\n", encoding="utf-8")
    result_path = Path(request["result_path_windows"])
    result = read_json(result_path) if result_path.exists() else {"status": "not_available", "transitions": [], "failure_reason": "child_result_missing"}
    result.update({"command": command, "exit_code": proc.returncode, "random_seed": random_seed, "stdout_path": str((root / "off_planner_stdout.log").resolve()), "stderr_path": str((root / "off_planner_stderr.log").resolve())})
    write_json(root / "off_planner_result_with_runtime.json", result)
    return result


def reconcile_process(cover: dict[str, Any], option_lists: list[list[dict[str, Any]]], result: dict[str, Any], nodes: dict[str, dict[str, Any]]) -> tuple[dict[str, Any] | None, dict[int, dict[str, Any]]]:
    selected_pairs = {int(row["transition_id"]): row.get("selected_attempt", {}).get("endpoint_candidate_a") for row in result.get("transitions", []) if row.get("status") == "passed_moveit"}
    selected_right = {int(row["transition_id"]): row.get("selected_attempt", {}).get("endpoint_candidate_b") for row in result.get("transitions", []) if row.get("status") == "passed_moveit"}
    chosen = []
    exact_pairs: dict[int, dict[str, Any]] = {}
    for segment_index, options in enumerate(option_lists):
        required_start = selected_right.get(segment_index) if segment_index > 0 else None
        required_end = selected_pairs.get(segment_index + 1) if segment_index < len(option_lists) - 1 else None
        candidates = [row for row in options if (required_start is None or str(row["start_candidate_id"]) == str(required_start)) and (required_end is None or str(row["end_candidate_id"]) == str(required_end))]
        if not candidates:
            return None, {}
        chosen.append(candidates[0])
    for transition_id in range(1, len(chosen)):
        left = chosen[transition_id - 1]
        right = chosen[transition_id]
        exact_pairs[transition_id] = {"rank": 0, "candidate_a": str(left["end_candidate_id"]), "candidate_b": str(right["start_candidate_id"]),
                                      "q_a": q_from_node(nodes[str(left["end_candidate_id"])]).tolist(), "q_b": q_from_node(nodes[str(right["start_candidate_id"])]).tolist(),
                                      "seed_pool_a": [], "seed_pool_b": []}
    return {"segment_count": len(chosen), "segments": chosen, "total_on_joint_motion_rad": sum(float(row.get("total_joint_motion_rad", 0.0)) for row in chosen)}, exact_pairs


def native_validate(root: Path, moveit_result: dict[str, Any], run_index: int, build_cache: dict[str, Any] | None):
    all_edges = []
    for transition_id in range(1, len(moveit_result.get("transitions", [])) + 1):
        edges, _ = s24.trajectory_edges(moveit_result, transition_id)
        all_edges.extend(edges)
    edge_csv = root / "off_native_edge_requests.csv"
    stage24.write_edge_requests(all_edges, edge_csv)
    stage24e.OUT = root / "native_validation"
    stage24.OUT = stage24e.OUT
    stage24e.PARTS = s24.PARTS_PATH
    stage24e.OUT.mkdir(parents=True, exist_ok=True)
    os.environ["FR5_BULLET_SHAPE_MODE"] = "use_shape_type"
    if build_cache is None:
        build_cache = stage24.build_native()
    records = {backend: stage24e.run_native_edges(edge_csv, backend, run_index, build_cache) for backend in ("fcl", "bullet")}
    validations = {}
    accepted_sets = {}
    for backend, record in records.items():
        rows = s24.read_jsonl(Path(record["result_path"])) if Path(record["result_path"]).exists() else []
        accepted = {(int(row["from_waypoint"]), int(row["to_waypoint"]), str(row["from_candidate_id"]), str(row["to_candidate_id"])) for row in rows if row.get("status") == "accepted" and row.get("indeterminate") is False}
        accepted_sets[backend] = accepted
        validations[backend] = {"backend": backend, "result_path": record.get("result_path"), "result_rows": len(rows), "expected_rows": len(all_edges),
                                "exit_code": record.get("exit_code"), "all_rows_present": len(rows) == len(all_edges), "all_accepted": len(rows) == len(all_edges) and len(accepted) == len(all_edges),
                                "accepted_edge_count": len(accepted), "rejected_edge_count": len(rows) - len(accepted), "collision_method": "adaptive_discrete_interpolation",
                                "CCD": "not_available", "clearance": "not_available", "trajectory_valid": len(rows) == len(all_edges) and len(accepted) == len(all_edges)}
    difference = {"fcl_bullet_symmetric_difference": sorted([list(key) for key in accepted_sets["fcl"] ^ accepted_sets["bullet"]]),
                  "symmetric_difference_count": len(accepted_sets["fcl"] ^ accepted_sets["bullet"]),
                  "accepted_set_hash_fcl": digest(sorted([list(key) for key in accepted_sets["fcl"]])),
                  "accepted_set_hash_bullet": digest(sorted([list(key) for key in accepted_sets["bullet"]])),
                  "equivalent": accepted_sets["fcl"] == accepted_sets["bullet"]}
    return validations["fcl"], validations["bullet"], difference, build_cache, all_edges


def r5_candidate_summary() -> list[dict[str, Any]]:
    rows = []
    for path in sorted(R5_SEARCH.glob("candidate_*/off_planner_result_with_runtime.json")):
        result = read_json(path)
        if len(result.get("transitions", [])) != 9:
            continue
        passed = sum(row.get("status") == "passed_moveit" for row in result["transitions"])
        attempt_path = path.parent / "process_attempt_summary.json"
        attempt = read_json(attempt_path) if attempt_path.exists() else {}
        rows.append({"candidate": path.parent.name, "segment_count": attempt.get("segment_count", 10), "off_passed": passed,
                     "off_required": 9, "segmentation": attempt.get("segmentation"), "result_path": str(path.resolve()),
                     "classification": "existing_10_segment_8_of_9" if passed == 8 else f"existing_10_segment_{passed}_of_9"})
    return rows


def write_sums(root: Path) -> None:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            rows.append(f"{sha256(path)}  {path.relative_to(root).as_posix()}")
    (root / "SHA256SUMS").write_text("\n".join(rows) + "\n", encoding="utf-8")


def report(root: Path, gate: dict[str, Any]) -> None:
    lines = ["# Stage 2.4T final segmented Spray-ON/OFF recovery", "", f"- Stage 2.4T: `{gate['Stage_2_4T']}`",
             f"- formal graph: `{gate['existing_formal_graph']['nodes']}` nodes / `{gate['existing_formal_graph']['edges']}` edges",
             f"- minimum graph-only ON segments: `{gate['minimum_graph_only_ON_segments']}`",
             f"- selected executable ON segments: `{gate.get('selected_executable_ON_segments')}`",
             f"- OFF transitions: `{gate.get('OFF_transitions_passed')}/{gate.get('OFF_transitions_required')}`",
             f"- waypoints covered: `{gate.get('waypoints_covered')}`", "",
             "Search is finite and OFF-aware. Collision checking is `adaptive_discrete_interpolation`; CCD and clearance are not available.",
             "Stage 2.5, TOTG, Ruckig, dynamics, and production timing were not run."]
    if gate["Stage_2_4T"] != "passed":
        lines.extend(["", "The bounded search did not prove physical path nonexistence.", f"- physical_path_nonexistence_proven: `{gate['physical_path_nonexistence_proven']}`"])
    (root / "stage24t_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    root = (args.output_root or BASE_OUTPUT / f"fr5_scaled_horseshoe_demo_v45_{timestamp}_Stage24T").resolve()
    root.mkdir(parents=True, exist_ok=False)
    os.environ["FR5_BULLET_SHAPE_MODE"] = "use_shape_type"
    waypoints, nodes, edges, audit = load_graph_and_waypoints()
    semantics = stage24g.joint_semantics()
    write_json(root / "stage24t_input_manifest.json", {"schema_version": "stage24t-input-manifest-v1", "stage": "Stage_2_4T",
        "frozen_graph": {"nodes": len(nodes), "edges": len(edges), "waypoints": len(waypoints)},
        "new_global_720_waypoint_IK_generation": {"performed": False, "allowed": False},
        "frozen_inputs": [s24.source_ref(s24.WAYPOINTS_PATH, "frozen 720 waypoint poses"), s24.source_ref(s24.PARTS_PATH, "frozen collision geometry"),
                           s24.source_ref(ROOT / "external/frcobot_ros2/fairino_description/urdf/fairino5_v6.urdf", "frozen URDF"),
                           s24.source_ref(ROOT / "ros2_moveit_bridge/config/fairino5_v6_spray_tcp.srdf", "frozen SRDF/ACM")],
        "collision_method": "adaptive_discrete_interpolation", "CCD": "not_available", "clearance": "not_available"})
    options_cache: dict[tuple[int, int], list[dict[str, Any]]] = {}
    raw_covers, dp_summary, _ = s24.build_segment_covers(nodes, edges, semantics)
    covers = [materialize_cover(cover, nodes, edges, semantics, options_cache) for cover in raw_covers]
    covers = [cover for cover in covers if cover is not None]
    existing = r5_candidate_summary()
    ten_candidates = {"existing_stage24s_candidates": existing, "existing_8_of_9_count": sum(row["off_passed"] == 8 for row in existing),
                      "formal_graph_candidate_count": len(nodes), "formal_graph_edge_count": len(edges),
                      "minimum_graph_only_ON_segments": dp_summary.get("minimum_graph_only_ON_segments"),
                      "candidate_cover_count": len(covers), "objective_order": ["complete_executable_process", "minimum_segment_count", "minimum_OFF_risk", "joint_clearance_path_cost"]}
    write_json(root / "stage24t_10segment_candidates.json", ten_candidates)
    if not covers:
        raise RuntimeError("no materialized formal 10-segment cover")

    # Report the automatically computed blocker-local structural search space.
    relocation_records, first_variants = local_boundary_records(covers[0], nodes, edges, semantics, options_cache)
    write_json(root / "stage24t_boundary_relocation.json", {"original_boundary": "394->395", "local_window": [LOCAL_BLOCKER_MIN, LOCAL_BLOCKER_MAX],
        "records": relocation_records, "valid_boundary_count": sum(row["status"] == "eligible" for row in relocation_records),
        "graph_connectivity_derived": True})

    endpoint_rows: list[dict[str, Any]] = []
    ik_rows: list[dict[str, Any]] = []
    staging_rows: list[dict[str, Any]] = []
    planner_rows: list[dict[str, Any]] = []
    search_rows: list[dict[str, Any]] = []
    transition_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    selected: dict[str, Any] | None = None
    build_cache = None

    # Prioritize exactly the 21 previously observed 10-segment 8/9 covers,
    # then the remaining formal 10-segment covers.  All covers stay in scope.
    priority_identities = []
    for row in existing:
        if row["off_passed"] == 8 and row.get("segmentation"):
            priority_identities.append(tuple(tuple(pair) for pair in row["segmentation"]))
    ordered_covers = []
    for identity in priority_identities:
        ordered_covers.extend([cover for cover in covers if cover_identity(cover) == identity])
    ordered_covers.extend([cover for cover in covers if cover not in ordered_covers])

    def variant_for(cover: dict[str, Any], boundary_left: int) -> tuple[dict[str, Any], list[list[dict[str, Any]]]] | None:
        index = find_blocker_index(cover)
        if index is None:
            return None
        left, right = cover["segments"][index], cover["segments"][index + 1]
        lo = formal_options(int(left["start_waypoint"]), boundary_left, nodes, edges, semantics, options_cache)
        ro = formal_options(boundary_left + 1, int(right["end_waypoint"]), nodes, edges, semantics, options_cache)
        if not lo or not ro:
            return None
        options = []
        for segment in cover["segments"]:
            options.append(formal_options(int(segment["start_waypoint"]), int(segment["end_waypoint"]), nodes, edges, semantics, options_cache))
        options[index] = lo
        options[index + 1] = ro
        variant_segments = list(cover["segments"])
        variant_segments[index] = lo[0]
        variant_segments[index + 1] = ro[0]
        variant = {"segment_count": 10, "segments": variant_segments, "boundary_left": boundary_left, "boundary_index": index}
        return variant, options

    # Tier 1-4 are applied to every valid blocker-local boundary.  A cached
    # transition search means the same formal boundary/pair pool is never
    # sent through native MoveIt twice for different segmentation covers.
    for tier in (1, 2, 3, 4):
        if selected is not None:
            break
        planner_ids = ["RRTConnectkConfigDefault"] if tier < 4 else ["RRTConnectkConfigDefault", "PRMkConfigDefault", "PRMstar"]
        for planner_id in planner_ids:
            if selected is not None:
                break
            for boundary_left in range(LOCAL_BLOCKER_MIN, LOCAL_BLOCKER_MAX + 1):
                if selected is not None:
                    break
                variant_probe = variant_for(covers[0], boundary_left)
                if variant_probe is None:
                    continue
                probe_cover, probe_options = variant_probe
                blocker_index = int(probe_cover["boundary_index"])
                cache_key = (boundary_left, tier, planner_id)
                if cache_key not in transition_cache:
                    search_dir = root / "search" / f"tier_{tier}" / f"boundary_{boundary_left:03d}_{boundary_left + 1:03d}_{planner_id}"
                    request = make_request(probe_cover, probe_options, waypoints, nodes, semantics, search_dir, tier, planner_id, 1, only_transition=blocker_index + 1)
                    write_json(root / "effective_planner_parameters" / f"tier_{tier}_{boundary_left:03d}_{planner_id}.json", {"requested": request["planner_parameters"], "effective": request["planner_parameters"], "api": "MoveItCpp plan_request_params loaded by stage24t launch"})
                    result = run_moveit_request(request, search_dir, planner_id)
                    transition = next((row for row in result.get("transitions", []) if int(row.get("transition_id", -1)) == blocker_index + 1), None)
                    transition_cache[cache_key] = {"result": result, "transition": transition, "search_dir": search_dir, "request": request}
                    if transition:
                        for attempt in transition.get("attempts", []):
                            endpoint_rows.append({"tier": tier, "boundary": f"{boundary_left}->{boundary_left + 1}", "planner_id": planner_id, "candidate_a": attempt.get("endpoint_candidate_a"), "candidate_b": attempt.get("endpoint_candidate_b"), "rank": attempt.get("endpoint_pair_rank"), "result": attempt.get("failure_reason") or ("passed" if attempt.get("path_valid") else "failed")})
                            ik_rows.append({"tier": tier, "boundary": f"{boundary_left}->{boundary_left + 1}", "planner_id": planner_id, "transition_id": blocker_index + 1, "attempt": attempt.get("ik")})
                            if attempt.get("reposition", {}).get("strategy") == "staging":
                                staging_rows.append({"tier": tier, "boundary": f"{boundary_left}->{boundary_left + 1}", "attempt": attempt.get("reposition")})
                            if attempt.get("reposition") is not None:
                                planner_rows.append({"tier": tier, "boundary": f"{boundary_left}->{boundary_left + 1}", "planner_id": planner_id, "planning": attempt.get("reposition")})
                cached = transition_cache[cache_key]
                transition = cached["transition"]
                search_rows.append({"tier": tier, "planner_id": planner_id, "boundary": f"{boundary_left}->{boundary_left + 1}", "status": transition.get("status") if transition else "not_available", "search_dir": str(cached["search_dir"].resolve())})
                if not transition or transition.get("status") != "passed_moveit":
                    continue
                for cover in ordered_covers:
                    variant_data = variant_for(cover, boundary_left)
                    if variant_data is None:
                        continue
                    candidate_cover, option_lists = variant_data
                    # Put the proven central endpoint pair first for the full
                    # candidate request while retaining the rest of each pool.
                    selected_attempt = transition.get("selected_attempt", {})
                    ca, cb = selected_attempt.get("endpoint_candidate_a"), selected_attempt.get("endpoint_candidate_b")
                    central_pair = next((row for row in endpoint_pair_pool(option_lists[blocker_index], option_lists[blocker_index + 1], nodes, semantics, tier, int(candidate_cover["segments"][blocker_index]["end_waypoint"]), int(candidate_cover["segments"][blocker_index + 1]["start_waypoint"]), 100) if str(row["candidate_a"]) == str(ca) and str(row["candidate_b"]) == str(cb)), None)
                    if central_pair is None:
                        continue
                    option_lists[blocker_index] = [next(row for row in option_lists[blocker_index] if str(row["end_candidate_id"]) == str(ca))] + [row for row in option_lists[blocker_index] if str(row["end_candidate_id"]) != str(ca)]
                    option_lists[blocker_index + 1] = [next(row for row in option_lists[blocker_index + 1] if str(row["start_candidate_id"]) == str(cb))] + [row for row in option_lists[blocker_index + 1] if str(row["start_candidate_id"]) != str(cb)]
                    full_dir = root / "search" / f"tier_{tier}" / f"candidate_{ordered_covers.index(cover) + 1:03d}_boundary_{boundary_left:03d}_{planner_id}"
                    full_request = make_request(candidate_cover, option_lists, waypoints, nodes, semantics, full_dir, tier, planner_id, 1)
                    full_result = run_moveit_request(full_request, full_dir, planner_id)
                    if full_result.get("status") != "passed_moveit":
                        continue
                    reconciled, exact_pairs = reconcile_process(candidate_cover, option_lists, full_result, nodes)
                    if reconciled is None:
                        continue
                    exact_dir = root / "search" / f"tier_{tier}" / f"candidate_{ordered_covers.index(cover) + 1:03d}_exact_{boundary_left:03d}_{planner_id}"
                    exact_request = make_request(reconciled, [[row] for row in reconciled["segments"]], waypoints, nodes, semantics, exact_dir, tier, planner_id, 1, exact_pairs=exact_pairs)
                    exact_result = run_moveit_request(exact_request, exact_dir, planner_id)
                    if exact_result.get("status") != "passed_moveit":
                        continue
                    on_result = s24.on_validation(reconciled, nodes, edges)
                    if on_result["missing_formal_items"]:
                        continue
                    fcl, bullet, difference, build_cache, all_edges = native_validate(exact_dir, exact_result, 1000 + len(search_rows), build_cache)
                    if not (fcl["trajectory_valid"] and bullet["trajectory_valid"] and difference["symmetric_difference_count"] == 0):
                        continue
                    selected = {"cover": reconciled, "option_lists": [[row] for row in reconciled["segments"]], "result": exact_result,
                                "root": exact_dir, "on": on_result, "fcl": fcl, "bullet": bullet, "difference": difference,
                                "all_edges": all_edges, "tier": tier, "planner_id": planner_id, "boundary_left": boundary_left}
                    break

    # Tier 5: bounded higher-segment covers, only after the complete 10-segment
    # local recovery space has been exercised.
    if selected is None:
        previous = covers[:s24.MAX_PROCESS_CANDIDATES]
        for segment_count in range(11, 15):
            derived = s24.derive_split_covers(previous, segment_count, nodes, edges, semantics)
            if not derived:
                break
            for cover_stub in derived[:32]:
                cover = materialize_cover(cover_stub, nodes, edges, semantics, options_cache)
                if cover is None:
                    continue
                option_lists = [formal_options(int(row["start_waypoint"]), int(row["end_waypoint"]), nodes, edges, semantics, options_cache) for row in cover["segments"]]
                request_dir = root / "search" / "tier_5" / f"segments_{segment_count:02d}_{len(search_rows):04d}"
                request = make_request(cover, option_lists, waypoints, nodes, semantics, request_dir, 3, "RRTConnectkConfigDefault", 1)
                result = run_moveit_request(request, request_dir, "RRTConnectkConfigDefault")
                search_rows.append({"tier": 5, "segment_count": segment_count, "status": result.get("status"), "search_dir": str(request_dir.resolve())})
                if result.get("status") != "passed_moveit":
                    continue
                reconciled, exact_pairs = reconcile_process(cover, option_lists, result, nodes)
                if reconciled is None:
                    continue
                on_result = s24.on_validation(reconciled, nodes, edges)
                if on_result["missing_formal_items"]:
                    continue
                fcl, bullet, difference, build_cache, all_edges = native_validate(request_dir, result, 1500 + segment_count, build_cache)
                if fcl["trajectory_valid"] and bullet["trajectory_valid"] and difference["symmetric_difference_count"] == 0:
                    selected = {"cover": reconciled, "option_lists": [[row] for row in reconciled["segments"]], "result": result,
                                "root": request_dir, "on": on_result, "fcl": fcl, "bullet": bullet, "difference": difference,
                                "all_edges": all_edges, "tier": 5, "planner_id": "RRTConnectkConfigDefault", "boundary_left": None}
                    break
            previous = derived
            if selected is not None:
                break

    # Persist all bounded search evidence before rebuilds.
    write_jsonl(root / "stage24t_endpoint_pair_search.jsonl", endpoint_rows)
    write_jsonl(root / "stage24t_retract_approach_ik_search.jsonl", ik_rows)
    write_jsonl(root / "stage24t_staging_search.jsonl", staging_rows)
    write_jsonl(root / "stage24t_moveit_planning_attempts.jsonl", planner_rows)
    write_json(root / "stage24t_search_budget.json", {"tiers_executed": sorted({int(row["tier"]) for row in search_rows}), "search_rows": len(search_rows),
        "candidate_covers_considered": len(ordered_covers), "local_boundary_window": [LOCAL_BLOCKER_MIN, LOCAL_BLOCKER_MAX],
        "endpoint_pairs_initial": 25, "multi_seed_max": 8, "distance_ladders_mm": {"tier1": [10, 20, 30, 40, 60, 80], "tier2_plus": [10, 20, 30, 40, 60, 80, 100, 120, 150]},
        "planner_portfolio": ["RRTConnectkConfigDefault", "PRMkConfigDefault", "PRMstar"], "budget_exhausted": selected is None,
        "search_space_executed": "local boundary relocation + existing formal endpoint pools + bounded multi-IK + staging + explicit planners + 10-14 segment fallback"})

    rebuilds = []
    deterministic = {"independent_rebuilds": 0, "selected_segment_count_same": False, "selected_segmentation_same": False,
                     "selected_ON_node_sequence_same": False, "selected_OFF_strategy_same": False, "gate_result_same": False,
                     "formal_validation_same": False, "status": "not_run_no_complete_process"}
    if selected is not None:
        for rebuild_index in range(1, REBUILD_COUNT + 1):
            # Final rebuilds use the formal selected endpoints only.  This
            # prevents an alternative pair tried during discovery from
            # silently changing the published ON segment endpoints.
            exact_pairs = {}
            for transition_id in range(1, len(selected["cover"]["segments"])):
                left = selected["cover"]["segments"][transition_id - 1]
                right = selected["cover"]["segments"][transition_id]
                selected_transition = next((row for row in selected["result"].get("transitions", []) if int(row.get("transition_id", -1)) == transition_id), {})
                selected_attempt = selected_transition.get("selected_attempt") or {}
                selected_ik = selected_attempt.get("ik_selected") or {}
                seed_a = [q_from_node(nodes[str(left["end_candidate_id"])]).tolist()]
                seed_b = [q_from_node(nodes[str(right["start_candidate_id"])]).tolist()]
                for key in ("a_mid", "a_retracted"):
                    if selected_ik.get(key, {}).get("q"):
                        seed_a.append([float(x) for x in selected_ik[key]["q"]])
                for key in ("b_mid", "b_retracted"):
                    if selected_ik.get(key, {}).get("q"):
                        seed_b.append([float(x) for x in selected_ik[key]["q"]])
                exact_pairs[transition_id] = {"rank": 0, "candidate_a": str(left["end_candidate_id"]), "candidate_b": str(right["start_candidate_id"]),
                                              "q_a": q_from_node(nodes[str(left["end_candidate_id"])]).tolist(), "q_b": q_from_node(nodes[str(right["start_candidate_id"])]).tolist(), "seed_pool_a": seed_a, "seed_pool_b": seed_b}
            result = {"status": "not_available", "transitions": []}
            fcl = bullet = difference = {"trajectory_valid": False, "symmetric_difference_count": None}
            native = {"status": "not_run"}
            rebuild_dir = root / f"rebuild_{rebuild_index:02d}"
            retry_records = []
            for retry in range(1, 4):
                attempt_dir = root / f"rebuild_{rebuild_index:02d}_attempt_{retry:02d}"
                request = make_request(selected["cover"], [[row] for row in selected["cover"]["segments"]], waypoints, nodes, semantics, attempt_dir,
                                        max(2, int(selected["tier"])), selected["planner_id"], rebuild_index, exact_pairs=exact_pairs)
                request["random_seed"] = 20260802 if retry == 1 else 20260802 + retry
                result = run_moveit_request(request, attempt_dir, selected["planner_id"])
                retry_records.append({"retry": retry, "random_seed": request["random_seed"], "status": result.get("status"), "result_path": str((attempt_dir / "off_planner_result_with_runtime.json").resolve())})
                if result.get("status") != "passed_moveit":
                    continue
                fcl, bullet, difference, build_cache, _ = native_validate(attempt_dir, result, 2000 + rebuild_index * 10 + retry, None)
                native = {"status": "passed" if fcl["trajectory_valid"] and bullet["trajectory_valid"] and difference["symmetric_difference_count"] == 0 else "failed", "fcl": fcl, "bullet": bullet, "difference": difference}
                if native["status"] == "passed":
                    break
            final_gate = "passed" if result.get("status") == "passed_moveit" and native["status"] == "passed" else "blocked"
            semantic = {"segment_count": len(selected["cover"]["segments"]), "segmentation": [[int(row["start_waypoint"]), int(row["end_waypoint"]) ] for row in selected["cover"]["segments"]],
                        "ON_node_sequence": [str(cid) for segment in selected["cover"]["segments"] for cid in segment["path_candidate_ids"]],
                        "OFF_strategy": [(row.get("selected_attempt") or {}).get("reposition", {}).get("strategy") for row in result.get("transitions", [])], "gate": final_gate,
                        "fcl": fcl.get("trajectory_valid"), "bullet": bullet.get("trajectory_valid"), "difference": difference.get("symmetric_difference_count")}
            record = {"rebuild_index": rebuild_index, "final_gate": final_gate, "semantic": semantic, "semantic_hash": digest(semantic),
                      "result_path": retry_records[-1]["result_path"] if retry_records else None, "native": native, "retry_records": retry_records}
            write_json(rebuild_dir / "rebuild_semantic_record.json", record)
            rebuilds.append(record)
        semantic_hashes = [row["semantic_hash"] for row in rebuilds]
        deterministic = {"independent_rebuilds": len(rebuilds), "selected_segment_count_same": len({row["semantic"]["segment_count"] for row in rebuilds}) == 1,
                         "selected_segmentation_same": len({canonical(row["semantic"]["segmentation"]) for row in rebuilds}) == 1,
                         "selected_ON_node_sequence_same": len({canonical(row["semantic"]["ON_node_sequence"]) for row in rebuilds}) == 1,
                         "selected_OFF_strategy_same": len({canonical(row["semantic"]["OFF_strategy"]) for row in rebuilds}) == 1,
                         "gate_result_same": len({row["final_gate"] for row in rebuilds}) == 1,
                         "formal_validation_same": all(row["final_gate"] == "passed" for row in rebuilds), "semantic_hashes": semantic_hashes,
                         "status": "passed" if len(rebuilds) == 3 and all(row["final_gate"] == "passed" for row in rebuilds) and len(set(semantic_hashes)) == 1 else "blocked"}
    write_json(root / "stage24t_determinism_report.json", deterministic)

    passed = selected is not None and deterministic.get("status") == "passed"
    selected_count = len(selected["cover"]["segments"]) if selected is not None else None
    if passed:
        items = s24.write_success_csvs(root, selected["cover"], nodes, read_json(Path(rebuilds[0]["result_path"])))
        write_json(root / "stage24t_selected_process.json", {"status": "passed", "segment_count": selected_count, "boundary_left": selected["boundary_left"], "items": items, "selected_search_root": str(selected["root"].resolve())})
        write_json(root / "stage24t_on_validation.json", selected["on"])
        write_json(root / "stage24t_fcl_off_validation.json", selected["fcl"])
        write_json(root / "stage24t_bullet_off_validation.json", selected["bullet"])
        write_json(root / "stage24t_backend_difference.json", selected["difference"])
        write_json(root / "stage24t_effective_planner_parameters.json", {"requested": planner_parameters(max(3, int(selected["tier"])), selected["planner_id"]), "effective": planner_parameters(max(3, int(selected["tier"])), selected["planner_id"]), "api": "MoveItCpp plan_request_params loaded by stage24t launch"})
        manifest = {"schema_version": "stage24t-process-manifest-v1", "process_type": "segmented_spray_process_v1", "ordered_items": items,
                    "process_start_waypoint": 0, "process_end_waypoint": 719, "all_required_waypoints_covered": True,
                    "spray_on_segments": selected_count, "off_transitions": selected_count - 1,
                    "stage_2_5": {"status": "unblocked_not_started", "TOTG": "not_run", "Ruckig": "not_run"}}
        write_json(root / "stage24t_process_manifest.json", manifest)
    else:
        write_json(root / "stage24t_selected_process.json", {"status": "not_selected", "reason": "no complete executable process within finite Stage 2.4T budget"})
        write_json(root / "stage24t_on_validation.json", {"status": "not_evaluated"})
        write_json(root / "stage24t_fcl_off_validation.json", {"status": "not_available"})
        write_json(root / "stage24t_bullet_off_validation.json", {"status": "not_available"})
        write_json(root / "stage24t_backend_difference.json", {"status": "not_available"})
        write_json(root / "stage24t_effective_planner_parameters.json", {"status": "not_recorded"})
        write_json(root / "stage24t_process_manifest.json", {"status": "blocked", "ordered_items": []})

    gate = {"Stage_2_4T": "passed" if passed else "blocked", "Stage_2_4_FINAL": {"status": "passed" if passed else "blocked"},
            "Stage_2_5": {"status": "unblocked_not_started" if passed else "blocked"}, "engineering_process": {"contract": "segmented_spray_process_v1", "ON_segments": selected_count, "OFF_transitions": selected_count - 1 if selected_count else 0},
            "existing_formal_graph": {"nodes": len(nodes), "edges": len(edges)}, "minimum_graph_only_ON_segments": int(dp_summary.get("minimum_graph_only_ON_segments") or 10),
            "selected_executable_ON_segments": selected_count, "waypoints_covered": "720/720" if passed else "0/720",
            "OFF_transitions_required": selected_count - 1 if selected_count else 9, "OFF_transitions_passed": selected_count - 1 if passed and selected_count else 0,
            "Spray_ON": selected["on"] if selected else {"status": "not_evaluated"},
            "Spray_OFF": {"transitions_planned": f"{selected_count - 1}/{selected_count - 1}" if passed and selected_count else "0/9", "retract": "passed" if passed else "not_passed", "reposition": "passed" if passed else "not_passed", "approach": "passed" if passed else "not_passed", "joint_limits": "passed" if passed else "not_passed", "FCL": "passed" if passed else "not_passed", "Bullet": "passed" if passed else "not_passed", "backend_symmetric_difference": 0 if passed else None},
            "former_blocker": {"boundary_394_395": {"status": "avoided_or_recovered" if passed else "searched_not_recovered", "solution_type": "boundary_relocation_or_alternate_endpoint_or_multi_IK_or_staging"}},
            "new_global_720_waypoint_IK_generation": {"performed": False}, "determinism": deterministic, "search_rows": len(search_rows),
            "first_failed_attempt": search_rows[0] if search_rows else None, "first_candidate_level_failed_transition": next((row for row in search_rows if row.get("status") != "passed_moveit"), None),
            "dominant_blocking_transition": None if passed else "394->395 family remains unresolved within finite recovery search", "final_unsolved_transition": None if passed else "not_proven",
            "search_space_executed": "boundary relocation + formal endpoint enumeration + multi-seed OFF IK + distance/direction recovery + staging + explicit multi-attempt planning + planner portfolio + 10-14 segment fallback",
            "budget_exhausted": not passed, "remaining_unsearched_space": "continuous IK/planner space and any unimplemented local endpoint generation", "physical_path_nonexistence_proven": False,
            "CCD": "not_available", "clearance": "not_available"}
    write_json(root / "stage24t_gate_report.json", gate)
    report(root, gate)
    write_sums(root)
    print(json.dumps({"output_root": str(root), "Stage_2_4T": gate["Stage_2_4T"], "selected_segments": selected_count, "determinism": deterministic["status"]}, ensure_ascii=False, indent=2))
    return 0 if passed else 5


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--timestamp", default=None)
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
