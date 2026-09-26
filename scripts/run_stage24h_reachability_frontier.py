"""Stage 2.4H reachability-aware bidirectional seeded IK continuation.

This runner consumes the persisted Stage 2.4G canonical graph read-only.  It
recomputes forward/reverse reachability before every frontier search, uses only
the currently reachable forward frontier nodes as primary seeds, performs the
required reverse continuation, and sends every newly accepted node/edge
through the existing native FCL/Bullet validators.

The search is finite and deterministic.  Internal Cartesian continuation poses
are persisted as solver evidence only; they never become formal waypoints.
Collision results retain the repository contract:
``adaptive_discrete_interpolation`` (not continuous collision detection).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TextIO

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_stage24_closed_loop_graph as stage24
import scripts.run_stage24e_tolerance_search as stage24e
import scripts.run_stage24g_complete_edge_component_bridge as stage24g


STAGE24G = ROOT / "outputs/ik_graph_stage24g_complete_edge_component_bridge/fr5_scaled_horseshoe_demo_v45_20260731"
STAGE24G_RUN = STAGE24G / "run1"
STAGE24F_EDGES = stage24g.FROZEN_F / "stage24f_layered_graph_edges.jsonl"
STAGE24G_NODES = STAGE24G / "stage24g_valid_nodes.jsonl"
STAGE24G_EDGE_EVIDENCE = STAGE24G_RUN / "stage24g_native_edge_evidence.jsonl"
STAGE24G_CANDIDATE_EDGE_EVIDENCE = tuple(sorted(STAGE24G_RUN.glob("candidate_edge_materialization_round*/stage24g_native_edge_evidence.jsonl")))
STAGE24G_GATE = STAGE24G / "stage24g_gate_report.json"
STAGE24G_HASHES = STAGE24G / "stage24g_frozen_input_hashes.json"
WAYPOINT_COUNT = 720
MAX_STEP_DEG = 20.0
INTERPOLATION_STEP_DEG = 1.0
BACKENDS = ("fcl", "bullet")
NATIVE_RUN_INDEX = 2
HOMOTOPY_LADDER = (1, 2, 4, 8, 16, 32)
DEFAULT_OUT = ROOT / "outputs/ik_graph_stage24h_reachability_frontier_continuation/fr5_scaled_horseshoe_demo_v45_20260801_bounded_bridge_search"
EDGE_PAIR_SEARCH_POLICY = "seed_consistent_frontier_bridge_plus_all_reachable_previous_nodes_for_reverse_candidates"


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def value_hash(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def semantic_hash(value: Any) -> str:
    if isinstance(value, dict):
        ignored = {"path", "output", "run_duration_s", "captured_at_utc", "started_at_utc", "finished_at_utc", "run_index"}
        return value_hash({k: semantic_hash(v) if isinstance(v, (dict, list)) else v for k, v in value.items() if k not in ignored})
    if isinstance(value, list):
        return value_hash([semantic_hash(v) if isinstance(v, (dict, list)) else v for v in value])
    return value_hash(value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")


def write_jsonl_row(handle: TextIO, row: dict[str, Any]) -> None:
    """Write one evidence row immediately so large failed searches stay bounded."""
    handle.write(canonical(row) + "\n")
    handle.flush()


def concatenate_jsonl(target: Path, sources: list[Path]) -> None:
    """Concatenate already persisted deterministic JSONL batches without loading them."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as output:
        for source in sources:
            if not source.exists():
                continue
            with source.open("r", encoding="utf-8") as input_handle:
                for line in input_handle:
                    if line.strip():
                        output.write(line)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fields or (list(rows[0]) if rows else ["empty"])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_q(value: Any) -> list[float]:
    if isinstance(value, str):
        return [float(x) for x in json.loads(value)]
    return [float(x) for x in value]


def git_status() -> str:
    import subprocess

    proc = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True, check=False)
    return proc.stdout + proc.stderr


def source_ref(path: Path, role: str) -> dict[str, Any]:
    return {"path": str(path.resolve()), "role": role, "exists": path.exists(), "sha256": sha256(path) if path.is_file() else None, "size": path.stat().st_size if path.is_file() else None}


def verify_frozen_inputs() -> dict[str, Any]:
    persisted = read_json(STAGE24G_HASHES) if STAGE24G_HASHES.exists() else {"missing": True}
    recomputed = stage24g.recompute_frozen_hashes()
    return {
        "stage24g_persisted_hash_report": persisted,
        "stage24g_recomputed_hash_report": recomputed,
        "checked_count": recomputed.get("checked_count", 0),
        "failure_count": recomputed.get("failure_count", 1),
        "all_397_inputs_verified": bool(recomputed.get("all_397_inputs_verified")) and recomputed.get("checked_count") == 397,
        "persisted_stage24g_hashes_clean": bool(persisted.get("all_397_inputs_verified")) and persisted.get("checked_count") == 397 and persisted.get("failure_count") == 0,
    }


def load_canonical_graph(semantics: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[tuple[int, int, str, str], dict[str, Any]], dict[str, Any]]:
    if not STAGE24G_NODES.exists() or not STAGE24G_EDGE_EVIDENCE.exists() or not STAGE24F_EDGES.exists():
        raise FileNotFoundError("Stage 2.4G canonical graph artifacts are incomplete")

    nodes: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(STAGE24G_NODES):
        cid = str(row["candidate_id"])
        nodes[cid] = stage24g.normalized_node(row, "Stage_2_4G_canonical", semantics, row.get("provenance"))

    edges: dict[tuple[int, int, str, str], dict[str, Any]] = {}

    def add_edge(row: dict[str, Any], source_stage: str, evidence: list[str]) -> None:
        if not (row.get("accepted") is True or (row.get("status") == "accepted" and row.get("valid") is True) or (row.get("fcl_edge_valid") is True and row.get("bullet_edge_valid") is True)):
            return
        source = str(row.get("from_candidate_id", row.get("source_node")))
        target = str(row.get("to_candidate_id", row.get("target_node")))
        if source not in nodes or target not in nodes:
            return
        from_wp = int(row["from_waypoint"])
        to_wp = int(row["to_waypoint"])
        edge = stage24g.edge_from_q(from_wp, to_wp, source, target, nodes[source]["joint_vector_rad"], nodes[target]["joint_vector_rad"], source_stage, evidence)
        edges[stage24g.edge_key(edge)] = edge

    for row in read_jsonl(STAGE24F_EDGES):
        add_edge(row, "Stage_2_4F_canonical", list(row.get("source_evidence") or ["Stage_2_4F_canonical"]))

    evidence_files = [STAGE24G_EDGE_EVIDENCE, *STAGE24G_CANDIDATE_EDGE_EVIDENCE]
    evidence_count = 0
    for evidence_file in evidence_files:
        if not evidence_file.exists():
            continue
        for item in read_jsonl(evidence_file):
            if item.get("dual_backend_valid") is not True:
                continue
            request = dict(item["request"])
            request["accepted"] = True
            add_edge(request, "Stage_2_4G_native_canonical", ["Stage_2_4G_native_canonical", str(evidence_file.name)])
            evidence_count += 1

    audit = {
        "canonical_nodes_path": str(STAGE24G_NODES.resolve()),
        "canonical_nodes_sha256": sha256(STAGE24G_NODES),
        "stage24f_edges_path": str(STAGE24F_EDGES.resolve()),
        "stage24f_edges_sha256": sha256(STAGE24F_EDGES),
        "stage24g_edge_evidence_files": [source_ref(path, "Stage_2_4G native edge evidence") for path in evidence_files],
        "stage24f_edge_rows_read": sum(1 for _ in read_jsonl(STAGE24F_EDGES)),
        "stage24g_native_dual_edge_rows_read": evidence_count,
        "canonical_node_count": len(nodes),
        "canonical_edge_count": len(edges),
    }
    return nodes, edges, audit


def recompute_reachability(nodes: dict[str, dict[str, Any]], edges: dict[tuple[int, int, str, str], dict[str, Any]]) -> dict[str, Any]:
    forward = stage24g.forward_reachable_sets(nodes, edges)
    reverse = stage24g.reverse_reachable_sets(nodes, edges)
    membership, components = stage24g.weak_components(nodes, edges)
    fsets = {int(row["waypoint"]): set(row["forward_candidate_ids"]) for row in forward["sets"]}
    rsets = {int(row["waypoint"]): set(row["reverse_candidate_ids"]) for row in reverse["sets"]}

    forward_frontier = None
    for i in range(WAYPOINT_COUNT - 1):
        if fsets.get(i) and not fsets.get(i + 1):
            forward_frontier = f"{i}->{i + 1}"
            break
    reverse_frontier = None
    for i in range(WAYPOINT_COUNT - 1):
        if rsets.get(i + 1) and not rsets.get(i):
            reverse_frontier = f"{i}->{i + 1}"
            break

    furthest = max((i for i, ids in fsets.items() if ids), default=None)
    reverse_nearest = min((i for i, ids in rsets.items() if ids), default=None)
    forward_ids = sorted(fsets.get(furthest, set())) if furthest is not None else []
    reverse_ids = sorted(rsets.get(reverse_nearest, set())) if reverse_nearest is not None else []
    return {
        "forward_reachable_sets": forward,
        "reverse_reachable_sets": reverse,
        "current_forward_frontier": forward_frontier,
        "current_reverse_frontier": reverse_frontier,
        "forward_frontier_waypoint": furthest,
        "forward_frontier_candidate_ids": forward_ids,
        "reverse_frontier_waypoint": reverse_nearest,
        "reverse_frontier_candidate_ids": reverse_ids,
        "wp0_reachable_furthest_waypoint": furthest,
        "component_membership": membership,
        "component_summary": components,
    }


def build_target_pose(waypoints: dict[int, dict[str, str]], waypoint: int, parameter: dict[str, float]) -> np.ndarray:
    return stage24g.pose_target(waypoints, waypoint, parameter)


def node_target_pose(waypoints: dict[int, dict[str, str]], node: dict[str, Any]) -> np.ndarray:
    parameters = {name: 0.0 for name in ("tangent", "lateral", "normal", "standoff", "rx", "ry", "roll")}
    offset = node.get("position_offset_local_m")
    if isinstance(offset, list) and len(offset) >= 3:
        parameters.update({"tangent": float(offset[0]), "lateral": float(offset[1]), "normal": float(offset[2])})
    normal_rotation = node.get("normal_rotation_vector_deg")
    if isinstance(normal_rotation, list) and len(normal_rotation) >= 2:
        parameters.update({"rx": float(normal_rotation[0]), "ry": float(normal_rotation[1])})
    if node.get("standoff_offset_m") is not None:
        parameters["standoff"] = float(node["standoff_offset_m"])
    if node.get("roll_offset_deg") is not None:
        parameters["roll"] = float(node["roll_offset_deg"])
    variant = node.get("target_variant")
    if isinstance(variant, dict) and isinstance(variant.get("parameters"), dict):
        parameters.update({key: float(value) for key, value in variant["parameters"].items() if key in parameters})
    return build_target_pose(waypoints, int(node["waypoint_index"]), parameters)


def pose_errors(solver: Any, q: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    fk = solver.fk_tcp(q)
    position_error = float(np.linalg.norm(fk[:3, 3] - target[:3, 3]))
    actual_z = fk[:3, 2] / max(float(np.linalg.norm(fk[:3, 2])), 1.0e-15)
    target_z = target[:3, 2] / max(float(np.linalg.norm(target[:3, 2])), 1.0e-15)
    z_error = float(math.degrees(math.acos(float(np.clip(np.dot(actual_z, target_z), -1.0, 1.0)))))
    return position_error, z_error


def joint_delta(seed_q: np.ndarray, q: np.ndarray) -> tuple[list[float], float, str | None]:
    # All six joints are bounded revolute.  Deliberately use raw differences.
    delta_deg = np.degrees(q - seed_q).tolist()
    maximum = max((abs(value) for value in delta_deg), default=0.0)
    blocking = f"j{int(np.argmax(np.abs(delta_deg))) + 1}" if delta_deg else None
    return delta_deg, float(maximum), blocking


def target_variant_row(parameter: dict[str, Any]) -> dict[str, Any]:
    return {"parameter_index": int(parameter["parameter_index"]), "level": int(parameter["level"]), "description": str(parameter["description"]), "parameters": {str(k): float(v) for k, v in parameter["parameters"].items()}}


def solve_one(
    solver: Any,
    waypoints: dict[int, dict[str, str]],
    target_waypoint: int,
    target_variant: dict[str, Any],
    seed_id: str,
    seed_q: np.ndarray,
    seed_signature: str,
    direction: str,
    attempt_key: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    target = build_target_pose(waypoints, target_waypoint, target_variant["parameters"])
    started = time.perf_counter()
    try:
        result = solver.solve(target, seed_q)
    except Exception as exc:
        return ({
            "attempt_key": attempt_key,
            "direction": direction,
            "seed_candidate_id": seed_id,
            "target_waypoint": target_waypoint,
            "target_variant": target_variant,
            "solver_success": False,
            "solver_status": "exception",
            "solver_message": repr(exc),
            "solution_q": None,
            "joint_delta_from_seed_deg": None,
            "max_joint_delta_deg": None,
            "blocking_joint": None,
            "joint_limits_valid": False,
            "pose_error": None,
            "branch_signature": None,
            "branch_change_from_seed": None,
            "duplicate_of_existing_candidate": None,
            "continuation_method": "direct_seeded",
            "subdivision_depth": None,
            "elapsed_s": time.perf_counter() - started,
        }, None)

    q = np.asarray(result.q_rad, dtype=float)
    # DeterministicNumericIKSolver already computes these values for the exact
    # returned q and target.  Reusing them avoids a second FK per solve without
    # changing the formal pose contract or the recorded evidence.
    position_error = float(result.position_error_m)
    z_error = float(result.tool_z_error_deg)
    delta_deg, max_delta, blocking = joint_delta(seed_q, q)
    limits_valid = bool(np.all(q >= solver.robot.limits.q_min - 1.0e-9) and np.all(q <= solver.robot.limits.q_max + 1.0e-9))
    signature = stage24g.canonical_branch_signature(q.tolist(), stage24g.joint_semantics())
    solver_success = bool(result.success)
    formal = bool(solver_success and limits_valid and position_error <= 0.006 + 1.0e-9 and z_error <= 10.0 + 1.0e-9)
    attempt = {
        "attempt_key": attempt_key,
        "direction": direction,
        "seed_candidate_id": seed_id,
        "target_waypoint": target_waypoint,
        "target_variant": target_variant,
        "solver_success": solver_success,
        "solver_status": "success" if solver_success else "failed",
        "solver_message": str(result.message),
        "solver_status_code": int(result.solver_status),
        "function_evaluations": int(result.function_evaluations),
        "solution_q": q.tolist(),
        "joint_delta_from_seed_deg": delta_deg,
        "max_joint_delta_deg": max_delta,
        "blocking_joint": blocking,
        "joint_limits_valid": limits_valid,
        "pose_error": {"position_m": position_error, "tool_z_deg": z_error},
        "branch_signature": signature,
        "branch_change_from_seed": signature != seed_signature,
        "duplicate_of_existing_candidate": None,
        "formal_candidate": formal,
        "continuation_method": "direct_seeded",
        "subdivision_depth": None,
        "elapsed_s": time.perf_counter() - started,
    }
    candidate = None
    if formal:
        candidate = {
            "waypoint_id": target_waypoint,
            "waypoint_index": target_waypoint,
            "joint_values_rad": q.tolist(),
            "joint_values": q.tolist(),
            "joint_values_deg": np.degrees(q).tolist(),
            "ik_branch_signature": signature,
            "canonical_branch_signature": signature,
            "seed_id": seed_id,
            "direction": direction,
            "target_variant": target_variant,
            "source_stage": "Stage_2_4H",
            "position_offset_local_m": [target_variant["parameters"]["tangent"], target_variant["parameters"]["lateral"], target_variant["parameters"]["normal"]],
            "standoff_offset_m": target_variant["parameters"]["standoff"],
            "normal_rotation_vector_deg": [target_variant["parameters"]["rx"], target_variant["parameters"]["ry"]],
            "roll_offset_deg": target_variant["parameters"]["roll"],
            "FK_position_error_m": position_error,
            "FK_orientation_error_deg": z_error,
            "pose_error": position_error,
            "joint_limit_valid": limits_valid,
            "joint_limit_pass": limits_valid,
            "process_tolerance_pass": True,
            "solver_success": solver_success,
            "candidate_parameter_hash": value_hash(target_variant["parameters"]),
            "formal_node_gate_pending": True,
        }
    return attempt, candidate


def make_homotopy_attempts(
    solver: Any,
    source_pose: np.ndarray,
    target_pose: np.ndarray,
    target_waypoint: int,
    target_variant: dict[str, Any],
    seed_id: str,
    seed_q: np.ndarray,
    seed_signature: str,
    direction: str,
    attempt_key: str,
    state_sink: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[int, dict[str, Any] | None]:
    state_count = 0
    for depth in HOMOTOPY_LADDER:
        q_previous = np.asarray(seed_q, dtype=float)
        depth_states: list[dict[str, Any]] = []
        success = True
        for step in range(1, depth + 1):
            alpha = step / depth
            pose = np.eye(4, dtype=float)
            # Rotation interpolation is deterministic in this frozen solver:
            # use the source-to-target relative rotation's scaled rotvec.
            from scipy.spatial.transform import Rotation

            relative = source_pose[:3, :3].T @ target_pose[:3, :3]
            pose[:3, :3] = source_pose[:3, :3] @ Rotation.from_rotvec(Rotation.from_matrix(relative).as_rotvec() * alpha).as_matrix()
            pose[:3, 3] = source_pose[:3, 3] + alpha * (target_pose[:3, 3] - source_pose[:3, 3])
            state_started = time.perf_counter()
            try:
                result = solver.solve(pose, q_previous)
                q = np.asarray(result.q_rad, dtype=float)
                # Reuse the solver's exact error report; do not perform a
                # duplicate FK for every internal continuation state.
                position_error = float(result.position_error_m)
                z_error = float(result.tool_z_error_deg)
                limits_valid = bool(np.all(q >= solver.robot.limits.q_min - 1.0e-9) and np.all(q <= solver.robot.limits.q_max + 1.0e-9))
                state_success = bool(result.success and limits_valid and position_error <= 0.006 + 1.0e-9 and z_error <= 10.0 + 1.0e-9)
                message = str(result.message)
                status_code = int(result.solver_status)
                nfev = int(result.function_evaluations)
            except Exception as exc:
                q = None
                position_error = None
                z_error = None
                limits_valid = False
                state_success = False
                message = repr(exc)
                status_code = None
                nfev = None
            state = {
                "direction": direction,
                "seed_candidate_id": seed_id,
                "target_waypoint": target_waypoint,
                "target_variant": target_variant,
                "subdivision_depth": depth,
                "internal_step": step,
                "internal_step_count": depth,
                "alpha": alpha,
                "solver_success": state_success,
                "solver_status_code": status_code,
                "solver_message": message,
                "seed_q": q_previous.tolist(),
                "solution_q": q.tolist() if q is not None else None,
                "pose_position": pose[:3, 3].tolist(),
                "pose_rotation": pose[:3, :3].tolist(),
                "pose_error": {"position_m": position_error, "tool_z_deg": z_error} if position_error is not None else None,
                "joint_limits_valid": limits_valid,
                "elapsed_s": time.perf_counter() - state_started,
                "formal_waypoint": False,
            }
            depth_states.append(state)
            state_count += 1
            if state_sink is not None:
                state_sink(state)
            if not state_success or q is None:
                success = False
                break
            q_previous = q
        if success and depth_states:
            final = depth_states[-1]
            q = np.asarray(final["solution_q"], dtype=float)
            delta_deg, max_delta, blocking = joint_delta(seed_q, q)
            signature = stage24g.canonical_branch_signature(q.tolist(), stage24g.joint_semantics())
            attempt = {
                "attempt_key": attempt_key,
                "direction": direction,
                "seed_candidate_id": seed_id,
                "target_waypoint": target_waypoint,
                "target_variant": target_variant,
                "solver_success": True,
                "solver_status": "success",
                "solver_message": final["solver_message"],
                "solution_q": q.tolist(),
                "joint_delta_from_seed_deg": delta_deg,
                "max_joint_delta_deg": max_delta,
                "blocking_joint": blocking,
                "joint_limits_valid": bool(final["joint_limits_valid"]),
                "pose_error": final["pose_error"],
                "branch_signature": signature,
                "branch_change_from_seed": signature != seed_signature,
                "duplicate_of_existing_candidate": None,
                "formal_candidate": True,
                "continuation_method": "solver_internal_cartesian_homotopy",
                "subdivision_depth": depth,
                "internal_state_count": state_count,
                "formal_waypoint": True,
            }
            candidate = {
                "waypoint_id": target_waypoint,
                "waypoint_index": target_waypoint,
                "joint_values_rad": q.tolist(),
                "joint_values": q.tolist(),
                "joint_values_deg": np.degrees(q).tolist(),
                "ik_branch_signature": signature,
                "canonical_branch_signature": signature,
                "seed_id": seed_id,
                "direction": direction,
                "target_variant": target_variant,
                "source_stage": "Stage_2_4H",
                "position_offset_local_m": [target_variant["parameters"]["tangent"], target_variant["parameters"]["lateral"], target_variant["parameters"]["normal"]],
                "standoff_offset_m": target_variant["parameters"]["standoff"],
                "normal_rotation_vector_deg": [target_variant["parameters"]["rx"], target_variant["parameters"]["ry"]],
                "roll_offset_deg": target_variant["parameters"]["roll"],
                "FK_position_error_m": final["pose_error"]["position_m"],
                "FK_orientation_error_deg": final["pose_error"]["tool_z_deg"],
                "pose_error": final["pose_error"]["position_m"],
                "joint_limit_valid": bool(final["joint_limits_valid"]),
                "joint_limit_pass": bool(final["joint_limits_valid"]),
                "process_tolerance_pass": True,
                "solver_success": True,
                "candidate_parameter_hash": value_hash(target_variant["parameters"]),
                "formal_node_gate_pending": True,
                "subdivision_depth": depth,
                "internal_state_count": state_count,
            }
            return state_count, (attempt, candidate)
    return state_count, None


# Process-pool state is deliberately limited to solver and frozen waypoint
# data.  Each worker returns one seed's deterministic evidence in schedule
# order; the parent emits it in the same order, so parallel execution does not
# change candidate identity, reachability, or persisted row ordering.
_SEED_WORKER_SOLVER: Any | None = None
_SEED_WORKER_WAYPOINTS: dict[int, dict[str, str]] | None = None


def _seed_worker_init(contract: dict[str, Any], waypoints: dict[int, dict[str, str]]) -> None:
    global _SEED_WORKER_SOLVER, _SEED_WORKER_WAYPOINTS
    _SEED_WORKER_SOLVER = stage24e.solver_from_contract(contract)
    _SEED_WORKER_WAYPOINTS = waypoints


def _run_seed_payload(payload: dict[str, Any]) -> dict[str, Any]:
    solver = _SEED_WORKER_SOLVER
    waypoints = _SEED_WORKER_WAYPOINTS
    if solver is None or waypoints is None:
        raise RuntimeError("seed_worker_not_initialized")
    direction = str(payload["direction"])
    seed_id = str(payload["seed_id"])
    target_waypoint = int(payload["target_waypoint"])
    seed_row = payload["seed_row"]
    label = str(payload["frontier"])
    seed_q = np.asarray(seed_row["joint_vector_rad"], dtype=float)
    seed_signature = str(seed_row.get("canonical_branch_signature"))
    source_pose = node_target_pose(waypoints, seed_row)
    manifest = {"direction": direction, "seed_candidate_id": seed_id, "seed_waypoint": seed_row["waypoint_index"], "target_waypoint": target_waypoint, "seed_canonical_branch_signature": seed_signature, "seed_joint_values": seed_row["joint_vector_rad"], "frontier": label}
    items: list[dict[str, Any]] = []
    internal_states: list[dict[str, Any]] = []
    for parameter in payload["schedule"]:
        target_variant = target_variant_row(parameter)
        key = canonical({"frontier": label, "direction": direction, "seed": seed_id, "target_waypoint": target_waypoint, "parameter_index": target_variant["parameter_index"]})
        direct_attempt, direct_candidate = solve_one(solver, waypoints, target_waypoint, target_variant, seed_id, seed_q, seed_signature, direction, key)
        attempt_rows = [direct_attempt]
        final_candidate = direct_candidate
        state_count = 0
        if final_candidate is None and not direct_attempt.get("solver_success"):
            target_pose = build_target_pose(waypoints, target_waypoint, target_variant["parameters"])
            state_count, homotopy_result = make_homotopy_attempts(solver, source_pose, target_pose, target_waypoint, target_variant, seed_id, seed_q, seed_signature, direction, key, internal_states.append)
            if homotopy_result is not None:
                homotopy_attempt, final_candidate = homotopy_result
                attempt_rows.append(homotopy_attempt)
        items.append({"attempt_rows": attempt_rows, "candidate": final_candidate, "attempt_key": key, "target_variant": target_variant, "target_waypoint": target_waypoint, "internal_state_count": state_count})
    return {"manifest": manifest, "items": items, "internal_states": internal_states}


def generate_seeded_frontier_candidates(
    nodes: dict[str, dict[str, Any]],
    reachability: dict[str, Any],
    waypoints: dict[int, dict[str, str]],
    solver: Any,
    schedule: list[dict[str, Any]],
    attempted_keys: set[str],
    existing_q: dict[tuple[int, tuple[float, ...]], str],
    attempts_handle: TextIO,
    internal_handle: TextIO,
    seed_handle: TextIO,
    contract: dict[str, Any],
    workers: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    label = reachability.get("current_forward_frontier")
    if not label:
        return [], {"seed_count": 0, "attempt_count": 0, "solver_success_count": 0, "solver_failure_count": 0, "internal_state_count": 0, "candidate_count": 0}
    from_wp, to_wp = (int(value) for value in label.split("->", 1))
    forward_seed_ids = sorted(reachability["forward_reachable_sets"]["sets"][from_wp]["forward_candidate_ids"])
    target_wp_nodes = sorted(cid for cid, row in nodes.items() if int(row["waypoint_index"]) == to_wp and row.get("dual_backend_valid") is True)
    reverse_seed_ids = target_wp_nodes
    seeds = [("forward", cid, to_wp, cid) for cid in forward_seed_ids]
    seeds.extend(("reverse", cid, from_wp, cid) for cid in reverse_seed_ids)
    candidates: list[dict[str, Any]] = []
    stats = {"seed_count": 0, "attempt_count": 0, "solver_success_count": 0, "solver_failure_count": 0, "internal_state_count": 0, "candidate_count": 0}
    payloads = [{"direction": direction, "seed_id": seed_id, "target_waypoint": target_waypoint, "seed_row": nodes[seed_id], "frontier": label, "schedule": schedule} for direction, seed_id, target_waypoint, _ in seeds]

    def consume_seed_result(result: dict[str, Any]) -> None:
        manifest = result["manifest"]
        write_jsonl_row(seed_handle, manifest)
        stats["seed_count"] += 1
        direction = str(manifest["direction"])
        seed_id = str(manifest["seed_candidate_id"])
        for internal_state in result["internal_states"]:
            write_jsonl_row(internal_handle, internal_state)
        for item in result["items"]:
            key = str(item["attempt_key"])
            if key in attempted_keys:
                continue
            attempted_keys.add(key)
            attempt_rows = item["attempt_rows"]
            final_candidate = item["candidate"]
            target_variant = item["target_variant"]
            target_waypoint = int(item["target_waypoint"])
            stats["internal_state_count"] += int(item.get("internal_state_count", 0))
            if final_candidate is not None:
                q_key = (target_waypoint, tuple(round(float(value), 10) for value in final_candidate["joint_values_rad"]))
                duplicate = existing_q.get(q_key)
                if duplicate is not None:
                    attempt_rows[-1]["duplicate_of_existing_candidate"] = duplicate
                else:
                    candidate_id = f"s24h-{direction[:3]}-{target_waypoint:04d}-{value_hash({'frontier': label, 'seed': seed_id, 'parameter': target_variant['parameter_index'], 'q': [round(float(value), 12) for value in final_candidate['joint_values_rad']]})[:16]}"
                    final_candidate["candidate_id"] = candidate_id
                    attempt_rows[-1]["candidate_id"] = candidate_id
                    final_candidate["source_attempt_key"] = key
                    existing_q[q_key] = candidate_id
                    candidates.append(final_candidate)
            for attempt in attempt_rows:
                write_jsonl_row(attempts_handle, attempt)
                stats["attempt_count"] += 1
                if attempt.get("solver_success") is True:
                    stats["solver_success_count"] += 1
                else:
                    stats["solver_failure_count"] += 1

    if int(workers) > 1 and len(payloads) > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=int(workers), initializer=_seed_worker_init, initargs=(contract, waypoints)) as executor:
            for result in executor.map(_run_seed_payload, payloads, chunksize=1):
                consume_seed_result(result)
    else:
        global _SEED_WORKER_SOLVER, _SEED_WORKER_WAYPOINTS
        _SEED_WORKER_SOLVER = solver
        _SEED_WORKER_WAYPOINTS = waypoints
        for payload in payloads:
            consume_seed_result(_run_seed_payload(payload))
    stats["candidate_count"] = len(candidates)
    candidates.sort(key=lambda row: (int(row["waypoint_index"]), str(row["direction"]), str(row["seed_id"]), int(row["target_variant"]["parameter_index"]), str(row["candidate_id"])))
    return candidates, stats


def native_node_gate(out: Path, candidates: list[dict[str, Any]]) -> tuple[set[str], dict[str, Any]]:
    if not candidates:
        return set(), {"candidate_count": 0, "fcl_valid": 0, "bullet_valid": 0, "dual_backend_valid": 0, "fcl_bullet_node_difference": 0, "records": []}
    node_dir = out / "node_validation"
    stage24e.OUT = node_dir
    stage24e.PARTS = stage24g.PARTS
    stage24e.STAGE23B = stage24g.STAGE23B
    candidate_csv = node_dir / "stage24h_native_node_requests.csv"
    node_dir.mkdir(parents=True, exist_ok=True)
    with candidate_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["waypoint_id", "candidate_id", "joint_values"], lineterminator="\n")
        writer.writeheader()
        for row in candidates:
            writer.writerow({"waypoint_id": row["waypoint_id"], "candidate_id": row["candidate_id"], "joint_values": canonical(row["joint_values_rad"])})
    records = [stage24e.run_native_nodes(candidate_csv, backend, NATIVE_RUN_INDEX) for backend in BACKENDS]
    backend_ids: dict[str, set[str]] = {}
    for record in records:
        result_path = Path(record["result_path"])
        rows = read_jsonl(result_path) if result_path.exists() else []
        backend_ids[record["backend"]] = {str(row["candidate_id"]) for row in rows if row.get("valid") is True}
    fcl = backend_ids.get("fcl", set())
    bullet = backend_ids.get("bullet", set())
    audit = {"candidate_count": len(candidates), "fcl_valid": len(fcl), "bullet_valid": len(bullet), "dual_backend_valid": len(fcl & bullet), "fcl_bullet_node_difference": len(fcl ^ bullet), "fcl_valid_ids": sorted(fcl), "bullet_valid_ids": sorted(bullet), "dual_backend_valid_ids": sorted(fcl & bullet), "records": records}
    write_json(out / "stage24h_native_node_gate.json", audit)
    return fcl & bullet, audit


def edge_request(from_node: dict[str, Any], to_node: dict[str, Any]) -> dict[str, Any]:
    q0 = parse_q(from_node["joint_vector_rad"])
    q1 = parse_q(to_node["joint_vector_rad"])
    signed = [b - a for a, b in zip(q0, q1)]
    absolute = [abs(value) for value in signed]
    max_step = math.degrees(max(absolute, default=0.0))
    limits_valid = all(float(rule["lower"]) - 1.0e-12 <= value <= float(rule["upper"]) + 1.0e-12 for value, rule in zip(q0 + q1, stage24g.joint_semantics() + stage24g.joint_semantics()))
    return {
        "transition": f"{int(from_node['waypoint_index'])}->{int(to_node['waypoint_index'])}",
        "from_waypoint": int(from_node["waypoint_index"]),
        "to_waypoint": int(to_node["waypoint_index"]),
        "from_candidate_id": str(from_node["candidate_id"]),
        "to_candidate_id": str(to_node["candidate_id"]),
        "q0": q0,
        "q1": q1,
        "per_joint_delta_deg": np.degrees(signed).tolist(),
        "absolute_joint_delta_deg": np.degrees(absolute).tolist(),
        "max_joint_delta_deg": max_step,
        "blocking_joint": f"j{int(np.argmax(absolute)) + 1}" if absolute else None,
        "joint_limit_valid": limits_valid,
        "joint_gate_pass": bool(limits_valid and max_step <= MAX_STEP_DEG + 1.0e-12),
        "interpolation_step_deg": INTERPOLATION_STEP_DEG,
        "collision_method": "adaptive_discrete_interpolation",
        "native_check_cache_key": value_hash({"from_waypoint": int(from_node["waypoint_index"]), "to_waypoint": int(to_node["waypoint_index"]), "from_q": [round(x, 12) for x in q0], "to_q": [round(x, 12) for x in q1], "interpolation_step_deg": INTERPOLATION_STEP_DEG}),
        "new_node_involved": str(from_node["candidate_id"]).startswith("s24h-") or str(to_node["candidate_id"]).startswith("s24h-"),
    }


def relevant_edge_requests(nodes: dict[str, dict[str, Any]], reachability: dict[str, Any], known_edges: dict[tuple[int, int, str, str], dict[str, Any]], tried_edges: set[tuple[int, int, str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    label = reachability.get("current_forward_frontier")
    if not label:
        return [], []
    from_wp, to_wp = (int(value) for value in label.split("->", 1))
    f_ids = sorted(reachability["forward_reachable_sets"]["sets"][from_wp]["forward_candidate_ids"])
    layer_to = sorted(cid for cid, row in nodes.items() if int(row["waypoint_index"]) == to_wp and row.get("dual_backend_valid") is True)
    layer_from = sorted(cid for cid, row in nodes.items() if int(row["waypoint_index"]) == from_wp and row.get("dual_backend_valid") is True)
    previous_reachable = set()
    if from_wp > 0:
        previous_reachable = set(reachability["forward_reachable_sets"]["sets"][from_wp - 1]["forward_candidate_ids"])
    layer_before = sorted(cid for cid in previous_reachable if cid in nodes and nodes[cid].get("dual_backend_valid") is True)
    requests: dict[tuple[int, int, str, str], dict[str, Any]] = {}
    rejected: list[dict[str, Any]] = []

    def add_pair(source_id: str, target_id: str, reason: str) -> None:
        source = nodes[source_id]
        target = nodes[target_id]
        row = edge_request(source, target)
        row["relevance_reason"] = reason
        key = stage24g.edge_key(row)
        if key in tried_edges or key in known_edges:
            return
        if row["joint_gate_pass"]:
            requests[key] = row
        else:
            rejected.append(row)

    # Forward bridge: only true F_i sources are primary seeds.  A generated
    # target is paired with the seed that generated its branch; cross-seed
    # pairs are retained as a declared finite-search omission, not silently
    # mistaken for a globally exhaustive edge set.
    for source_id in f_ids:
        for target_id in layer_to:
            target = nodes[target_id]
            if target_id.startswith("s24h-") and target.get("direction") == "forward" and str(target.get("seed_id")) == source_id:
                add_pair(source_id, target_id, "reachable_forward_frontier_seed_consistent")
            elif not target_id.startswith("s24h-"):
                add_pair(source_id, target_id, "reachable_forward_frontier_existing_target")
    # Reverse bridge: candidate wp_i must also connect to reachable wp_{i-1},
    # and each reverse candidate is checked toward its explicit wp_{i+1}
    # seed.  This is the deterministic bridge set used for the bounded native
    # run; arbitrary cross-seed combinations are not claimed as exhausted.
    for source_id in layer_before:
        for target_id in layer_from:
            if str(target_id).startswith("s24h-") and nodes[target_id].get("direction") == "reverse":
                add_pair(source_id, target_id, "reverse_to_reachable_previous_frontier")
    for source_id in layer_from:
        source = nodes[source_id]
        if not (str(source_id).startswith("s24h-") and source.get("direction") == "reverse"):
            continue
        seed_target = str(source.get("seed_id"))
        if seed_target in nodes and int(nodes[seed_target]["waypoint_index"]) == to_wp:
            add_pair(source_id, seed_target, "reverse_to_seeded_forward_target")
    ordered = sorted(requests.values(), key=lambda row: (int(row["from_waypoint"]), int(row["to_waypoint"]), float(row["max_joint_delta_deg"]), str(row["from_candidate_id"]), str(row["to_candidate_id"])))
    rejected.sort(key=lambda row: (int(row["from_waypoint"]), int(row["to_waypoint"]), float(row["max_joint_delta_deg"]), str(row["from_candidate_id"]), str(row["to_candidate_id"])))
    return ordered, rejected


def native_edge_gate(out: Path, requests: list[dict[str, Any]], round_index: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not requests:
        return [], {"native_checked": 0, "fcl_valid": 0, "bullet_valid": 0, "dual_backend_valid": 0, "fcl_bullet_edge_difference": 0, "records": []}
    batch_dir = out / f"edge_validation_round{round_index:03d}"
    evidence, meta = stage24g.native_edge_batch(batch_dir, requests)
    write_jsonl(batch_dir / "stage24h_native_edge_evidence.jsonl", evidence)
    return evidence, meta


def accepted_edges_from_evidence(evidence: list[dict[str, Any]], nodes: dict[str, dict[str, Any]]) -> tuple[dict[tuple[int, int, str, str], dict[str, Any]], set[tuple[int, int, str, str]], set[tuple[int, int, str, str]]]:
    accepted: dict[tuple[int, int, str, str], dict[str, Any]] = {}
    fcl_set: set[tuple[int, int, str, str]] = set()
    bullet_set: set[tuple[int, int, str, str]] = set()
    for item in evidence:
        request = item["request"]
        key = stage24g.edge_key(request)
        if item.get("fcl", {}).get("valid") is True and item.get("fcl", {}).get("status") == "accepted":
            fcl_set.add(key)
        if item.get("bullet", {}).get("valid") is True and item.get("bullet", {}).get("status") == "accepted":
            bullet_set.add(key)
        if item.get("dual_backend_valid") is not True:
            continue
        source = str(request["from_candidate_id"])
        target = str(request["to_candidate_id"])
        if source not in nodes or target not in nodes:
            continue
        accepted[key] = stage24g.edge_from_q(int(request["from_waypoint"]), int(request["to_waypoint"]), source, target, nodes[source]["joint_vector_rad"], nodes[target]["joint_vector_rad"], "Stage_2_4H_native", ["Stage_2_4H_native", str(request.get("relevance_reason"))])
    return accepted, fcl_set, bullet_set


def add_candidates(nodes: dict[str, dict[str, Any]], candidates: list[dict[str, Any]], valid_ids: set[str], semantics: list[dict[str, Any]]) -> None:
    for candidate in candidates:
        if candidate["candidate_id"] not in valid_ids:
            continue
        node = stage24g.normalized_node(candidate, "Stage_2_4H", semantics, [{"source_stage": "Stage_2_4H", "candidate_id": candidate["candidate_id"], "seed_id": candidate.get("seed_id"), "direction": candidate.get("direction"), "target_variant": candidate.get("target_variant")}])
        for key in ("seed_id", "direction", "target_variant", "subdivision_depth", "internal_state_count", "source_attempt_key"):
            if key in candidate:
                node[key] = candidate[key]
        nodes[candidate["candidate_id"]] = node


def best_bridge(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid = [row for row in rows if row.get("joint_gate_pass") is True]
    if not valid:
        return None
    row = min(valid, key=lambda item: (float(item["max_joint_delta_deg"]), str(item["from_candidate_id"]), str(item["to_candidate_id"])))
    return {"from_candidate_id": row["from_candidate_id"], "to_candidate_id": row["to_candidate_id"], "transition": row["transition"], "joint_deltas_deg": row["per_joint_delta_deg"], "max_joint_step_deg": row["max_joint_delta_deg"], "blocking_joint": row["blocking_joint"]}


def run_one(run_dir: Path, run_index: int, frozen: dict[str, Any], semantics: list[dict[str, Any]], contract: dict[str, Any], before_status: str, workers: int = 1) -> dict[str, Any]:
    started = time.time()
    run_dir.mkdir(parents=True, exist_ok=False)
    nodes, edges, graph_audit = load_canonical_graph(semantics)
    initial = recompute_reachability(nodes, edges)
    write_json(run_dir / "stage24h_initial_reachability.json", {k: v for k, v in initial.items() if k not in {"component_membership"}})
    write_json(run_dir / "stage24h_canonical_graph_audit.json", graph_audit)
    if initial["current_forward_frontier"] != "35->36" or len(initial["forward_frontier_candidate_ids"]) != 33:
        raise RuntimeError(f"canonical_stage24g_frontier_mismatch:{initial['current_forward_frontier']}:{len(initial['forward_frontier_candidate_ids'])}")

    waypoints = {int(row["waypoint_id"]): row for row in stage24e.read_csv(stage24e.SOURCE / "waypoints.csv")}
    solver = stage24e.solver_from_contract(contract)
    schedule = stage24g.parameter_schedule(contract)
    attempted_keys: set[str] = set()
    existing_q = {(int(row["waypoint_index"]), tuple(round(float(value), 10) for value in row["joint_vector_rad"])): cid for cid, row in nodes.items()}
    all_candidates: list[dict[str, Any]] = []
    all_prefilter_rejected: list[dict[str, Any]] = []
    all_native_evidence: list[dict[str, Any]] = []
    attempted_frontiers: list[dict[str, Any]] = []
    crossed_frontiers: list[str] = []
    node_gate_reports: list[dict[str, Any]] = []
    edge_gate_reports: list[dict[str, Any]] = []
    tried_edges: set[tuple[int, int, str, str]] = set(edges)
    fcl_edge_set: set[tuple[int, int, str, str]] = set()
    bullet_edge_set: set[tuple[int, int, str, str]] = set()
    attempt_streams: list[Path] = []
    internal_streams: list[Path] = []
    seed_streams: list[Path] = []
    all_attempt_count = 0
    all_solver_success_count = 0
    search_space_exhausted = False
    complete = False
    blocked_frontier: dict[str, Any] | None = None
    loop_limit = WAYPOINT_COUNT + 1

    for frontier_round in range(1, loop_limit):
        reachability = recompute_reachability(nodes, edges)
        label = reachability.get("current_forward_frontier")
        if label is None:
            complete = reachability.get("wp0_reachable_furthest_waypoint") == WAYPOINT_COUNT - 1
            if complete:
                break
            blocked_frontier = {"current_forward_frontier": None, "reason": "no_forward_frontier"}
            search_space_exhausted = True
            break
        before_furthest = reachability["wp0_reachable_furthest_waypoint"]
        before_nodes = len(nodes)
        before_edges = len(edges)
        attempt_path = run_dir / f"stage24h_seeded_attempts_frontier{frontier_round:03d}.jsonl"
        internal_path = run_dir / f"stage24h_internal_continuation_frontier{frontier_round:03d}.jsonl"
        seed_path = run_dir / f"stage24h_seed_manifest_frontier{frontier_round:03d}.jsonl"
        attempt_streams.append(attempt_path)
        internal_streams.append(internal_path)
        seed_streams.append(seed_path)
        with attempt_path.open("w", encoding="utf-8", newline="\n") as attempts_handle, internal_path.open("w", encoding="utf-8", newline="\n") as internal_handle, seed_path.open("w", encoding="utf-8", newline="\n") as seed_handle:
            candidates, frontier_stats = generate_seeded_frontier_candidates(nodes, reachability, waypoints, solver, schedule, attempted_keys, existing_q, attempts_handle, internal_handle, seed_handle, contract, workers)
        all_attempt_count += frontier_stats["attempt_count"]
        all_solver_success_count += frontier_stats["solver_success_count"]
        all_candidates.extend(candidates)
        if candidates:
            valid_ids, node_meta = native_node_gate(run_dir / f"frontier{frontier_round:03d}", candidates)
        else:
            valid_ids, node_meta = set(), {"candidate_count": 0, "fcl_valid": 0, "bullet_valid": 0, "dual_backend_valid": 0, "fcl_bullet_node_difference": 0, "records": []}
        node_gate_reports.append({"frontier_round": frontier_round, **node_meta})
        for candidate in candidates:
            candidate["node_gate_fcl_valid"] = candidate["candidate_id"] in set(node_meta.get("fcl_valid_ids", []))
            candidate["node_gate_bullet_valid"] = candidate["candidate_id"] in set(node_meta.get("bullet_valid_ids", []))
            candidate["node_gate_dual_backend_valid"] = candidate["candidate_id"] in valid_ids
        add_candidates(nodes, candidates, valid_ids, semantics)
        frontier_after_nodes = recompute_reachability(nodes, edges)
        requests, rejected = relevant_edge_requests(nodes, frontier_after_nodes, edges, tried_edges)
        all_prefilter_rejected.extend(rejected)
        for row in requests + rejected:
            tried_edges.add(stage24g.edge_key(row))
        if requests:
            evidence, edge_meta = native_edge_gate(run_dir, requests, frontier_round)
            accepted, fcl_batch, bullet_batch = accepted_edges_from_evidence(evidence, nodes)
            edges.update(accepted)
            all_native_evidence.extend(evidence)
            fcl_edge_set.update(fcl_batch)
            bullet_edge_set.update(bullet_batch)
        else:
            evidence, edge_meta = [], {"native_checked": 0, "fcl_valid": 0, "bullet_valid": 0, "dual_backend_valid": 0, "fcl_bullet_edge_difference": 0, "records": []}
        edge_gate_reports.append({"frontier_round": frontier_round, "request_count": len(requests), "joint_gate_rejected_count": len(rejected), **edge_meta})
        after = recompute_reachability(nodes, edges)
        after_furthest = after["wp0_reachable_furthest_waypoint"]
        advanced = after_furthest is not None and before_furthest is not None and after_furthest > before_furthest
        if advanced:
            crossed_frontiers.append(label)
        attempted_frontiers.append({"frontier_round": frontier_round, "current_forward_frontier": label, "current_reverse_frontier": reachability.get("current_reverse_frontier"), "reachable_frontier_nodes": len(reachability["forward_frontier_candidate_ids"]), "reverse_target_nodes": frontier_stats["seed_count"] - len(reachability["forward_frontier_candidate_ids"]), "seeded_ik_attempts": frontier_stats["attempt_count"], "seeded_ik_solutions": frontier_stats["solver_success_count"], "seeded_ik_internal_states": frontier_stats["internal_state_count"], "new_candidates": len(candidates), "new_dual_backend_valid_nodes": len(valid_ids), "new_dual_backend_valid_edges": len(accepted), "edge_requests": len(requests), "joint_gate_rejected_edges": len(rejected), "before_wp0_reachable_furthest_waypoint": before_furthest, "after_wp0_reachable_furthest_waypoint": after_furthest, "frontier_advanced": advanced, "nodes_before": before_nodes, "nodes_after": len(nodes), "edges_before": before_edges, "edges_after": len(edges), "best_reachable_bridge_candidate": best_bridge(requests + rejected)})
        write_json(run_dir / f"stage24h_reachability_after_frontier{frontier_round:03d}.json", {k: v for k, v in after.items() if k not in {"component_membership"}})
        if after["current_forward_frontier"] is None and after_furthest == WAYPOINT_COUNT - 1:
            complete = True
            break
        if not advanced:
            blocked_frontier = {"current_forward_frontier": label, "search_space_exhausted": True, "best_reachable_bridge_candidate": best_bridge(requests + rejected), "solver_failure_count": frontier_stats["solver_failure_count"], "joint_gate_failure_count": len(rejected), "node_collision_failure_count": sum(1 for row in candidates if not row.get("node_gate_dual_backend_valid")), "edge_collision_failure_count": sum(1 for item in evidence if item.get("dual_backend_valid") is not True), "reachable_frontier_nodes": len(reachability["forward_frontier_candidate_ids"])}
            search_space_exhausted = True
            break

    final = recompute_reachability(nodes, edges)
    open_chain = stage24g.search_open_chain(nodes, edges)
    complete = bool(open_chain.get("complete_0_to_719_open_chain_exists"))
    concatenate_jsonl(run_dir / "stage24h_all_seeded_ik_attempts.jsonl", attempt_streams)
    concatenate_jsonl(run_dir / "stage24h_all_internal_continuation_states.jsonl", internal_streams)
    concatenate_jsonl(run_dir / "stage24h_all_seed_manifest.jsonl", seed_streams)
    write_jsonl(run_dir / "stage24h_generated_candidates.jsonl", all_candidates)
    write_jsonl(run_dir / "stage24h_joint_gate_rejected_edges.jsonl", all_prefilter_rejected)
    write_jsonl(run_dir / "stage24h_native_edge_evidence.jsonl", all_native_evidence)
    write_json(run_dir / "stage24h_forward_reachability.json", final["forward_reachable_sets"])
    write_json(run_dir / "stage24h_reverse_reachability.json", final["reverse_reachable_sets"])
    write_json(run_dir / "stage24h_reachability_summary.json", {k: v for k, v in final.items() if k not in {"component_membership"}})
    write_json(run_dir / "stage24h_selected_open_chain.json", open_chain)
    if open_chain.get("selected_node_ids"):
        write_csv(run_dir / "stage24h_selected_open_chain.csv", [nodes[cid] for cid in open_chain["selected_node_ids"]])
    else:
        write_csv(run_dir / "stage24h_selected_open_chain.csv", [])

    fcl_difference = len(fcl_edge_set ^ bullet_edge_set)
    node_difference = sum(int(item.get("fcl_bullet_node_difference", 0)) for item in node_gate_reports) if node_gate_reports else 0
    best_unresolved = best_bridge(all_prefilter_rejected + [item for item in all_native_evidence for item in [item["request"]]])
    status = "passed_complete_0_to_719_open_chain" if complete else "blocked_reachability_frontier_fixed_point" if search_space_exhausted else "blocked_native_collision"
    if node_difference:
        status = "blocked_backend_disagreement"
    if fcl_difference:
        status = "blocked_backend_disagreement"
    gate = {
        "Stage_2_4H": status,
        "initial_forward_frontier": "35->36",
        "initial_reachable_frontier_nodes": 33,
        "frontiers_attempted": [item["current_forward_frontier"] for item in attempted_frontiers],
        "frontiers_crossed": crossed_frontiers,
        "seeded_ik_attempts": all_attempt_count,
        "seeded_ik_solutions": all_solver_success_count,
        "new_candidates": len(all_candidates),
        "new_dual_backend_valid_nodes": sum(int(item.get("dual_backend_valid", 0)) for item in node_gate_reports),
        "new_dual_backend_valid_edges": sum(int(item.get("dual_backend_valid", 0)) for item in edge_gate_reports),
        "initial_wp0_reachable_furthest_waypoint": 35,
        "final_wp0_reachable_furthest_waypoint": final.get("wp0_reachable_furthest_waypoint"),
        "complete_0_to_719_open_chain_exists": complete,
        "first_unresolved_frontier": final.get("current_forward_frontier") if not complete else None,
        "best_unresolved_candidate_ids": [best_unresolved.get("from_candidate_id"), best_unresolved.get("to_candidate_id")] if best_unresolved else [],
        "best_unresolved_joint_deltas_deg": best_unresolved.get("joint_deltas_deg") if best_unresolved else None,
        "best_unresolved_max_joint_step_deg": best_unresolved.get("max_joint_step_deg") if best_unresolved else None,
        "best_unresolved_blocking_joint": best_unresolved.get("blocking_joint") if best_unresolved else None,
        "current_forward_frontier": final.get("current_forward_frontier"),
        "current_reverse_frontier": final.get("current_reverse_frontier"),
        "fcl_bullet_node_difference": node_difference,
        "fcl_bullet_edge_difference": fcl_difference,
        "deterministic_rebuilds": "pending_top_level_comparison",
        "determinism_passed": False,
        "search_space_exhausted": search_space_exhausted,
        "blocked_frontier": blocked_frontier,
        "frozen_input_hashes": frozen,
        "canonical_graph_audit": graph_audit,
        "node_gate_reports": node_gate_reports,
        "edge_gate_reports": edge_gate_reports,
        "edge_pair_search_policy": EDGE_PAIR_SEARCH_POLICY,
        "solver_workers": int(workers),
        "cross_seed_edge_pairs_not_claimed_exhausted": True,
        "frontier_rounds": attempted_frontiers,
        "collision_method": "adaptive_discrete_interpolation",
        "CCD": "not_available",
        "clearance": "not_available",
        "Ruckig": "not_run",
        "TOTG": "not_run",
        "GNN": "not_run",
        "closure_diagnostic": {"source": str(stage24g.STAGE24E / "stage24e_gate_report.json"), "status": "not_recomputed_in_Stage_2_4H", "ready_for_Stage_2_5": False},
        "ready_for_Stage_2_5": False,
        "Stage_2_5": "blocked",
        "dirty_worktree_before": before_status,
        "dirty_worktree_after": git_status(),
    }
    write_json(run_dir / "stage24h_gate_report.json", gate)
    metadata = {
        "run_index": run_index,
        "runtime_duration": time.time() - started,
        "node_set": semantic_hash(sorted((cid, row["waypoint_index"], row["joint_vector_rad"], row["canonical_branch_signature"]) for cid, row in nodes.items())),
        "edge_set": semantic_hash(sorted((key, value.get("source_stage"), value.get("max_joint_delta_deg")) for key, value in edges.items())),
        "generated_candidate_ids": semantic_hash(sorted(row["candidate_id"] for row in all_candidates)),
        "candidate_joint_values": semantic_hash(sorted((row["candidate_id"], row["joint_values_rad"]) for row in all_candidates)),
        "node_validity": semantic_hash(node_gate_reports),
        "accepted_edge_set": semantic_hash(sorted(edges)),
        "forward_reachability": semantic_hash(final["forward_reachable_sets"]),
        "reverse_reachability": semantic_hash(final["reverse_reachable_sets"]),
        "selected_chain": semantic_hash(open_chain),
        "gate_status": semantic_hash({k: v for k, v in gate.items() if k not in {"dirty_worktree_before", "dirty_worktree_after", "determinism_passed", "deterministic_rebuilds"}}),
    }
    write_json(run_dir / "stage24h_run_metadata.json", metadata)
    return {"run_dir": str(run_dir.resolve()), "gate": gate, "metadata": metadata, "error": None}


def run_regression_tests(out: Path) -> dict[str, Any]:
    import subprocess

    tests = ["tests/test_stage24h_outputs.py"]
    results = []
    for test in tests:
        proc = subprocess.run([sys.executable, "-m", "pytest", "-q", test], cwd=ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True, check=False)
        results.append({"test": test, "status": "passed" if proc.returncode == 0 else "failed", "exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr})
    report = {"status": "passed" if all(row["status"] == "passed" for row in results) else "failed", "results": results}
    write_json(out / "stage24h_test_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = args.output_dir.resolve()
    if out.exists():
        raise RuntimeError(f"refusing to overwrite existing Stage 2.4H output directory: {out}")
    out.mkdir(parents=True, exist_ok=False)
    before = git_status()
    frozen = verify_frozen_inputs()
    write_json(out / "stage24h_frozen_input_hashes.json", frozen)
    write_json(out / "stage24h_input_manifest.json", {"schema_version": "stage24h-input-manifest-v1", "stage": "2.4H", "captured_at_utc": datetime.now(timezone.utc).isoformat(), "stage24g_gate": source_ref(STAGE24G_GATE, "Stage_2_4G canonical gate"), "stage24g_nodes": source_ref(STAGE24G_NODES, "Stage_2_4G canonical nodes"), "stage24g_edge_evidence": source_ref(STAGE24G_EDGE_EVIDENCE, "Stage_2_4G native edge evidence"), "frozen_input_hashes": frozen, "frozen_inputs_modified": False})
    if not frozen["all_397_inputs_verified"] or not frozen["persisted_stage24g_hashes_clean"]:
        gate = {"Stage_2_4H": "blocked_input_hash_mismatch", "ready_for_Stage_2_5": False, "Stage_2_5": "blocked", "frozen_input_hashes": frozen, "collision_method": "adaptive_discrete_interpolation", "CCD": "not_available", "clearance": "not_available"}
        write_json(out / "stage24h_gate_report.json", gate)
        write_json(out / "stage24h_determinism_report.json", {"independent_rebuilds": 0, "determinism_passed": False})
        return 2

    semantics = stage24g.joint_semantics()
    contract = stage24g.frozen_tolerance_contract()
    results = []
    errors = []
    for index in range(1, args.runs + 1):
        try:
            results.append(run_one(out / f"run{index}", index, frozen, semantics, contract, before, max(1, int(args.workers))))
        except Exception as exc:
            error = {"run_index": index, "type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
            errors.append(error)
            run_dir = out / f"run{index}"
            run_dir.mkdir(parents=True, exist_ok=True)
            write_json(run_dir / "stage24h_error.json", error)

    fields = ["node_set", "edge_set", "generated_candidate_ids", "candidate_joint_values", "node_validity", "accepted_edge_set", "forward_reachability", "reverse_reachability", "selected_chain", "gate_status"]
    determinism = {"independent_rebuilds": len(results), "requested_rebuilds": args.runs, "errors": errors, "semantic_hashes": {field: [result["metadata"].get(field) for result in results] for field in fields}, "differences": {field: len({result["metadata"].get(field) for result in results}) - 1 for field in fields}, "determinism_passed": not errors and len(results) == args.runs and all(len({result["metadata"].get(field) for result in results}) == 1 for field in fields), "nonsemantic_fields_excluded": ["absolute_output_directory", "run_timestamp", "runtime_duration", "process_id"]}
    write_json(out / "stage24h_determinism_report.json", determinism)
    if results:
        gate = dict(results[0]["gate"])
    else:
        gate = {"Stage_2_4H": "blocked_native_collision", "ready_for_Stage_2_5": False, "Stage_2_5": "blocked", "frozen_input_hashes": frozen}
    gate["deterministic_rebuilds"] = determinism["independent_rebuilds"]
    gate["determinism_passed"] = determinism["determinism_passed"]
    if errors:
        gate["Stage_2_4H"] = "blocked_native_collision"
        gate["errors"] = errors
    elif not determinism["determinism_passed"]:
        gate["Stage_2_4H"] = "blocked_determinism_failure"
    gate["ready_for_Stage_2_5"] = False
    gate["Stage_2_5"] = "blocked"
    write_json(out / "stage24h_gate_report.json", gate)
    tests = run_regression_tests(out)
    if tests["status"] != "passed":
        gate["Stage_2_4H"] = "blocked_regression_tests_failed"
        write_json(out / "stage24h_gate_report.json", gate)
    preservation = {"git_before": before, "git_after": git_status(), "preexisting_status_preserved": all(line in git_status() for line in before.splitlines() if line.strip()), "new_output_root": str(out)}
    write_json(out / "dirty_worktree_preservation.json", preservation)
    report = ["# Stage 2.4H reachability-aware bidirectional seeded IK frontier continuation", "", f"- Stage 2.4H: `{gate.get('Stage_2_4H')}`", f"- Initial forward frontier: `{gate.get('initial_forward_frontier')}` with `{gate.get('initial_reachable_frontier_nodes')}` reachable nodes.", f"- Final reachable furthest waypoint: `{gate.get('final_wp0_reachable_furthest_waypoint')}`; complete open chain: `{gate.get('complete_0_to_719_open_chain_exists')}`.", f"- Frontiers attempted/crossed: `{len(gate.get('frontiers_attempted', []))}/{len(gate.get('frontiers_crossed', []))}`.", f"- New dual-backend nodes/edges: `{gate.get('new_dual_backend_valid_nodes')}/{gate.get('new_dual_backend_valid_edges')}`.", f"- FCL/Bullet node/edge difference: `{gate.get('fcl_bullet_node_difference')}/{gate.get('fcl_bullet_edge_difference')}`.", f"- Deterministic rebuilds: `{determinism.get('independent_rebuilds')}`; passed: `{determinism.get('determinism_passed')}`.", f"- Regression tests: `{tests.get('status')}`.", f"- Native edge search policy: `{gate.get('edge_pair_search_policy')}`; arbitrary cross-seed edge combinations are not claimed exhausted.", "", "Collision evidence is native MoveIt2 PlanningScene evidence labelled `adaptive_discrete_interpolation`; CCD and clearance remain `not_available`. Internal Cartesian continuation poses are solver evidence only and are not formal waypoints. Stage 2.5 remains blocked and was not run.", "", "```yaml", json.dumps(gate, ensure_ascii=False, indent=2, sort_keys=True), "```", ""]
    (out / "stage24h_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"output": str(out), "Stage_2_4H": gate.get("Stage_2_4H"), "runs": len(results), "determinism_passed": determinism.get("determinism_passed"), "tests": tests.get("status"), "errors": errors}, ensure_ascii=False, sort_keys=True))
    return 0 if gate.get("Stage_2_4H") == "passed_complete_0_to_719_open_chain" else 2


if __name__ == "__main__":
    raise SystemExit(main())
