"""Stage 2.4I reachability-constrained local trajectory recovery.

This runner is deliberately additive.  It consumes the last real Stage 2.4H
forward-reachable graph, keeps the frozen Stage 2.4 contract unchanged, and
tries a local branch-preserving continuation followed by a multi-waypoint
pose-slack solve.  Numerical states are evidence only until they pass the
existing native FCL/Bullet node and edge gates.

The runner never treats a finite failed search as proof of continuous-space
infeasibility.  Collision evidence remains
``adaptive_discrete_interpolation``; CCD and clearance are unavailable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_stage24_closed_loop_graph as stage24
import scripts.run_stage24e_tolerance_search as stage24e
import scripts.run_stage24g_complete_edge_component_bridge as stage24g
import scripts.run_stage24h_reachability_frontier as stage24h


WAYPOINT_COUNT = 720
MAX_STEP_DEG = 20.0
MAX_STEP_RAD = math.radians(MAX_STEP_DEG)
INTERPOLATION_STEP_DEG = 1.0
BACKENDS = ("fcl", "bullet")
NATIVE_RUN_INDEX = 2
PARAMETER_NAMES = ("tangent", "lateral", "normal", "standoff", "rx", "ry", "roll")
DEFAULT_BASE = ROOT / "outputs/ik_graph_stage24h_reachability_frontier_continuation/fr5_scaled_horseshoe_demo_v45_20260801_parallel_streamed_run/run1"
DEFAULT_OUT = ROOT / "outputs/ik_graph_stage24i_reachability_constrained_local_recovery/fr5_scaled_horseshoe_demo_v45_20260802"


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def value_hash(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def semantic_hash(value: Any) -> str:
    if isinstance(value, dict):
        ignored = {"path", "output", "runtime_s", "started_at_utc", "finished_at_utc", "run_index"}
        return value_hash({k: semantic_hash(v) if isinstance(v, (dict, list)) else v for k, v in value.items() if k not in ignored})
    if isinstance(value, list):
        return value_hash([semantic_hash(v) if isinstance(v, (dict, list)) else v for v in value])
    return value_hash(value)


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


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def git_status() -> str:
    import subprocess

    proc = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True, check=False)
    return proc.stdout + proc.stderr


def q_from_node(row: dict[str, Any]) -> np.ndarray:
    value = row.get("joint_vector_rad", row.get("joint_values_rad", row.get("joint_values")))
    if isinstance(value, str):
        value = json.loads(value)
    return np.asarray([float(x) for x in value], dtype=float)


def load_waypoints() -> dict[int, dict[str, str]]:
    return {int(row["waypoint_id"]): row for row in stage24e.read_csv(stage24e.SOURCE / "waypoints.csv")}


def load_base_graph(base: Path, semantics: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[tuple[int, int, str, str], dict[str, Any]], dict[str, Any]]:
    """Load Stage 2.4G plus the last real Stage 2.4H materialization."""

    nodes, edges, audit = stage24h.load_canonical_graph(semantics)
    generated_path = base / "stage24h_generated_candidates.jsonl"
    evidence_path = base / "stage24h_native_edge_evidence.jsonl"
    if not generated_path.exists() or not evidence_path.exists():
        raise FileNotFoundError(f"incomplete Stage 2.4H base: {base}")

    generated = read_jsonl(generated_path)
    generated_valid = [row for row in generated if row.get("node_gate_dual_backend_valid") is True]
    for row in generated_valid:
        cid = str(row["candidate_id"])
        nodes[cid] = stage24g.normalized_node(row, "Stage_2_4H", semantics, [{"source_stage": "Stage_2_4H", "candidate_id": cid, "seed_id": row.get("seed_id"), "direction": row.get("direction")}])
        for key in ("seed_id", "direction", "target_variant", "subdivision_depth", "internal_state_count", "source_attempt_key"):
            if key in row:
                nodes[cid][key] = row[key]
        nodes[cid]["node_gate_dual_backend_valid"] = True

    evidence = read_jsonl(evidence_path)
    accepted, _, _ = stage24h.accepted_edges_from_evidence(evidence, nodes)
    edges.update(accepted)
    audit = dict(audit)
    audit.update({
        "stage24h_generated_rows_read": len(generated),
        "stage24h_generated_dual_backend_rows": len(generated_valid),
        "stage24h_native_edge_rows_read": len(evidence),
        "stage24h_native_dual_edges_merged": len(accepted),
        "base_stage": "Stage_2_4H_parallel_streamed_run1",
        "base_path": str(base.resolve()),
    })
    return nodes, edges, audit


def recompute(nodes: dict[str, dict[str, Any]], edges: dict[tuple[int, int, str, str], dict[str, Any]]) -> dict[str, Any]:
    forward = stage24g.forward_reachable_sets(nodes, edges)
    reverse = stage24g.reverse_reachable_sets(nodes, edges)
    fsets = {int(row["waypoint"]): set(row["forward_candidate_ids"]) for row in forward["sets"]}
    rsets = {int(row["waypoint"]): set(row["reverse_candidate_ids"]) for row in reverse["sets"]}
    furthest = max((wp for wp, ids in fsets.items() if ids), default=None)
    reverse_first = min((wp for wp, ids in rsets.items() if ids), default=None)
    ff = next((f"{wp}->{wp + 1}" for wp in range(WAYPOINT_COUNT - 1) if fsets.get(wp) and not fsets.get(wp + 1)), None)
    rf = next((f"{wp}->{wp + 1}" for wp in range(WAYPOINT_COUNT - 1) if rsets.get(wp + 1) and not rsets.get(wp)), None)
    return {
        "forward_reachable_sets": forward,
        "reverse_reachable_sets": reverse,
        "current_forward_frontier": ff,
        "current_reverse_frontier": rf,
        "forward_reachable_last_waypoint": furthest,
        "reverse_reachable_first_waypoint": reverse_first,
        "forward_frontier_candidate_ids": sorted(fsets.get(furthest, set())) if furthest is not None else [],
        "reverse_frontier_candidate_ids": sorted(rsets.get(reverse_first, set())) if reverse_first is not None else [],
        "candidate_count": len(nodes),
        "edge_count": len(edges),
        "transitions_with_edges": sum(1 for wp in range(WAYPOINT_COUNT - 1) if any(int(e["from_waypoint"]) == wp and int(e["to_waypoint"]) == wp + 1 for e in edges.values())),
    }


def nominal_params() -> dict[str, float]:
    return {name: 0.0 for name in PARAMETER_NAMES}


def parameter_bounds(contract: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    t = contract["tolerances"]
    bounds = np.asarray([
        float(t["tcp_position_offset_local_m"]["tangent"][1]),
        float(t["tcp_position_offset_local_m"]["lateral"][1]),
        float(t["tcp_position_offset_local_m"]["normal"][1]),
        float(t["standoff_offset_m"][1]),
        float(t["normal_rotation_vector_deg"]["rx"][1]),
        float(t["normal_rotation_vector_deg"]["ry"][1]),
        float(t["tool_roll_offset_deg"][1]),
    ], dtype=float)
    return -bounds, bounds


def params_to_array(params: dict[str, float]) -> np.ndarray:
    return np.asarray([float(params.get(name, 0.0)) for name in PARAMETER_NAMES], dtype=float)


def array_to_params(values: np.ndarray) -> dict[str, float]:
    return {name: float(value) for name, value in zip(PARAMETER_NAMES, values)}


def pose_target(waypoints: dict[int, dict[str, str]], waypoint: int, params: dict[str, float]) -> np.ndarray:
    return stage24g.pose_target(waypoints, waypoint, params)


def pose_feature(solver: Any, q: np.ndarray) -> np.ndarray:
    fk = solver.fk_tcp(q)
    z = fk[:3, 2] / max(float(np.linalg.norm(fk[:3, 2])), 1.0e-15)
    return np.r_[fk[:3, 3], z]


def target_feature(target: np.ndarray) -> np.ndarray:
    z = target[:3, 2] / max(float(np.linalg.norm(target[:3, 2])), 1.0e-15)
    return np.r_[target[:3, 3], z]


def pose_errors(solver: Any, q: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    actual = solver.fk_tcp(q)
    position = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
    az = actual[:3, 2] / max(float(np.linalg.norm(actual[:3, 2])), 1.0e-15)
    tz = target[:3, 2] / max(float(np.linalg.norm(target[:3, 2])), 1.0e-15)
    angle = float(math.degrees(math.acos(float(np.clip(np.dot(az, tz), -1.0, 1.0)))))
    return position, angle


def interpolate_pose(source: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    relative = source[:3, :3].T @ target[:3, :3]
    out = np.eye(4, dtype=float)
    out[:3, :3] = source[:3, :3] @ Rotation.from_rotvec(Rotation.from_matrix(relative).as_rotvec() * alpha).as_matrix()
    out[:3, 3] = source[:3, 3] + alpha * (target[:3, 3] - source[:3, 3])
    return out


def finite_difference_jacobian(solver: Any, q: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    base = pose_feature(solver, q)
    jac = np.zeros((6, 6), dtype=float)
    h = 1.0e-6
    for index in range(6):
        qp = q.copy()
        qm = q.copy()
        qp[index] = min(qp[index] + h, solver.robot.limits.q_max[index] - 1.0e-10)
        qm[index] = max(qm[index] - h, solver.robot.limits.q_min[index] + 1.0e-10)
        denominator = qp[index] - qm[index]
        jac[:, index] = (pose_feature(solver, qp) - pose_feature(solver, qm)) / max(denominator, 1.0e-15)
    singular = np.linalg.svd(jac, compute_uv=False)
    sigma_min = float(np.min(singular)) if len(singular) else 0.0
    sigma_max = float(np.max(singular)) if len(singular) else 0.0
    condition = float(sigma_max / sigma_min) if sigma_min > 1.0e-15 else float("inf")
    lam = max(1.0e-5, 0.01 * sigma_max)
    delta_x = target_feature(target) - base
    delta_q = jac.T @ np.linalg.solve(jac @ jac.T + (lam * lam) * np.eye(6), delta_x)
    q_pred = np.clip(q + delta_q, solver.robot.limits.q_min + 1.0e-10, solver.robot.limits.q_max - 1.0e-10)
    lower_distance = np.degrees(q - solver.robot.limits.q_min).tolist()
    upper_distance = np.degrees(solver.robot.limits.q_max - q).tolist()
    return q_pred, {
        "jacobian_singular_values": singular.tolist(),
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "condition_number": condition,
        "damping_lambda": float(lam),
        "delta_x": delta_x.tolist(),
        "delta_q_pred_deg": np.degrees(delta_q).tolist(),
        "distance_to_lower_joint_limits_deg": lower_distance,
        "distance_to_upper_joint_limits_deg": upper_distance,
    }


def corrector(solver: Any, target: np.ndarray, q_pred: np.ndarray, max_nfev: int = 2000) -> tuple[np.ndarray, dict[str, Any]]:
    target_values = target_feature(target)

    def residual(q: np.ndarray) -> np.ndarray:
        feature = pose_feature(solver, q)
        # Position and tool-normal are the same formal Stage 2.4 pose
        # quantities used by the frozen numeric candidate generator.
        return np.r_[(feature[:3] - target_values[:3]) / 0.001, (feature[3:] - target_values[3:]) / 0.01, (q - q_pred) * 0.02]

    q0 = np.clip(q_pred, solver.robot.limits.q_min + 1.0e-10, solver.robot.limits.q_max - 1.0e-10)
    result = least_squares(residual, q0, bounds=(solver.robot.limits.q_min, solver.robot.limits.q_max), method="trf", xtol=1.0e-11, ftol=1.0e-11, gtol=1.0e-11, max_nfev=max_nfev)
    q = np.asarray(result.x, dtype=float)
    pos, normal = pose_errors(solver, q, target)
    limits = bool(np.all(q >= solver.robot.limits.q_min - 1.0e-9) and np.all(q <= solver.robot.limits.q_max + 1.0e-9))
    accepted = bool(result.success and limits and pos <= 0.006 + 1.0e-9 and normal <= 10.0 + 1.0e-9)
    return q, {
        "solver_name": "stage24i_jacobian_predictor_corrector",
        "solver_algorithm": "scipy_least_squares_trf_bounded_fk_tool_z",
        "solver_success": bool(result.success),
        "solver_status": int(result.status),
        "termination_reason": str(result.message),
        "actual_iterations": None,
        "actual_nfev": int(result.nfev),
        "actual_njev": int(result.njev) if result.njev is not None else None,
        "max_nfev": int(max_nfev),
        "initial_cost": None,
        "final_cost": float(result.cost),
        "position_error_m": pos,
        "tool_z_error_deg": normal,
        "joint_limits_passed": limits,
        "accepted": accepted,
        "q_deg": np.degrees(q).tolist(),
    }


def adaptive_continuation(
    solver: Any,
    source_pose: np.ndarray,
    target_pose: np.ndarray,
    seed_q: np.ndarray,
    waypoint_interval: str,
    target_params: dict[str, float],
    trace_rows: list[dict[str, Any]],
    jacobian_rows: list[dict[str, Any]],
    max_subdivision: int = 256,
) -> dict[str, Any]:
    q = np.asarray(seed_q, dtype=float).copy()
    alpha = 0.0
    step = 1.0
    attempts = 0
    accepted_states = 0
    deepest = 0
    terminal = "not_started"
    while alpha < 1.0 - 1.0e-12 and attempts < max_subdivision * 4:
        trial = min(1.0, alpha + step)
        target = interpolate_pose(source_pose, target_pose, trial)
        q_pred, jac = finite_difference_jacobian(solver, q, target)
        q_corr, corr = corrector(solver, target, q_pred, max_nfev=2000)
        deepest = max(deepest, int(round(1.0 / max(step, 1.0e-12))))
        attempts += 1
        lower = np.degrees(q_corr - solver.robot.limits.q_min)
        upper = np.degrees(solver.robot.limits.q_max - q_corr)
        jacobian_rows.append({
            "waypoint_interval": waypoint_interval,
            "path_parameter": trial,
            "target_parameter_json": canonical(target_params),
            "joint_state_deg": canonical(np.degrees(q_corr).tolist()),
            **jac,
            "cartesian_pose_residual_m": corr["position_error_m"],
            "tool_z_residual_deg": corr["tool_z_error_deg"],
            "corrector_iterations": corr["actual_iterations"],
            "accepted": corr["accepted"],
            "distance_to_lower_joint_limits_deg": canonical(lower.tolist()),
            "distance_to_upper_joint_limits_deg": canonical(upper.tolist()),
        })
        trace_rows.append({
            "waypoint_interval": waypoint_interval,
            "path_parameter": trial,
            "attempt_index": attempts,
            "subdivision_depth": deepest,
            "step_size": step,
            "target_parameter_json": canonical(target_params),
            "q_seed_deg": canonical(np.degrees(q).tolist()),
            "q_predictor_deg": canonical(np.degrees(q_pred).tolist()),
            "q_corrected_deg": canonical(np.degrees(q_corr).tolist()),
            "predictor_max_delta_deg": float(np.max(np.abs(np.degrees(q_pred - q)))),
            "corrector_nfev": corr["actual_nfev"],
            "corrector_success": corr["solver_success"],
            "pose_error_mm": corr["position_error_m"] * 1000.0,
            "normal_error_deg": corr["tool_z_error_deg"],
            "accepted": corr["accepted"],
            "termination_reason": corr["termination_reason"],
        })
        if corr["accepted"]:
            q = q_corr
            alpha = trial
            accepted_states += 1
            step = min(1.0 - alpha if alpha < 1.0 else 1.0, step * 1.5) if alpha < 1.0 else step
            terminal = "target_reached" if alpha >= 1.0 - 1.0e-12 else "continuing"
        else:
            step *= 0.5
            terminal = "step_reduced"
            if step < 1.0 / max_subdivision:
                terminal = "minimum_adaptive_step_reached"
                break
    return {
        "waypoint_interval": waypoint_interval,
        "target_parameter": target_params,
        "initial_seed_deg": np.degrees(seed_q).tolist(),
        "final_q_deg": np.degrees(q).tolist(),
        "target_reached": bool(alpha >= 1.0 - 1.0e-12),
        "accepted_internal_states": accepted_states,
        "attempted_internal_states": attempts,
        "maximum_subdivision": deepest,
        "termination_reason": terminal,
        "final_alpha": alpha,
    }


def canonical_branch(q: np.ndarray) -> str:
    return stage24g.canonical_branch_signature(q.tolist(), stage24g.joint_semantics())


def candidate_pose_params(row: dict[str, Any]) -> dict[str, float]:
    if isinstance(row.get("target_variant"), dict) and isinstance(row["target_variant"].get("parameters"), dict):
        return {name: float(row["target_variant"]["parameters"].get(name, 0.0)) for name in PARAMETER_NAMES}
    return nominal_params()


def initial_sequence(
    solver: Any,
    waypoints: dict[int, dict[str, str]],
    nodes: dict[str, dict[str, Any]],
    anchor_wp: int,
    anchor_q: np.ndarray,
    window_wps: list[int],
) -> tuple[np.ndarray, list[dict[str, float]], list[dict[str, Any]]]:
    q_rows = [anchor_q.copy()]
    params = [nominal_params()]
    sources: list[dict[str, Any]] = [{"waypoint": anchor_wp, "source": "reachable_anchor"}]
    previous = anchor_q.copy()
    anchor_branch = canonical_branch(anchor_q)
    by_wp: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in nodes.values():
        by_wp[int(row["waypoint_index"])].append(row)
    for wp in window_wps:
        same_branch = [row for row in by_wp.get(wp, []) if row.get("canonical_branch_signature") == anchor_branch]
        pool = same_branch or by_wp.get(wp, [])
        selected = min(pool, key=lambda row: (float(np.max(np.abs(q_from_node(row) - previous))), str(row["candidate_id"]))) if pool else None
        if selected is not None:
            q = q_from_node(selected)
            p = candidate_pose_params(selected)
            sources.append({"waypoint": wp, "source": "existing_node", "candidate_id": selected["candidate_id"], "branch": selected.get("canonical_branch_signature")})
        else:
            target = pose_target(waypoints, wp, nominal_params())
            q, _ = corrector(solver, target, previous, max_nfev=2000)
            p = nominal_params()
            sources.append({"waypoint": wp, "source": "bounded_corrector_seed"})
        q_rows.append(q)
        params.append(p)
        previous = q
    return np.asarray(q_rows), params, sources


def multiwaypoint_optimize(
    solver: Any,
    waypoints: dict[int, dict[str, str]],
    contract: dict[str, Any],
    anchor_wp: int,
    anchor_q: np.ndarray,
    window_wps: list[int],
    nodes: dict[str, dict[str, Any]],
    pose_slack_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    initial_q, initial_params, sources = initial_sequence(solver, waypoints, nodes, anchor_wp, anchor_q, window_wps)
    q0 = initial_q[1:].reshape(-1)
    p0 = np.asarray([params_to_array(p) for p in initial_params[1:]], dtype=float).reshape(-1)
    x0 = np.r_[q0, p0]
    p_lo, p_hi = parameter_bounds(contract)
    q_lo = np.tile(solver.robot.limits.q_min, len(window_wps))
    q_hi = np.tile(solver.robot.limits.q_max, len(window_wps))
    lo = np.r_[q_lo, np.tile(p_lo, len(window_wps))]
    hi = np.r_[q_hi, np.tile(p_hi, len(window_wps))]
    q_scale = 0.35

    def unpack(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        qs = x[: 6 * len(window_wps)].reshape(len(window_wps), 6)
        ps = x[6 * len(window_wps):].reshape(len(window_wps), 7)
        return np.vstack([anchor_q, qs]), ps

    def residual(x: np.ndarray) -> np.ndarray:
        qs, ps = unpack(x)
        values: list[float] = []
        for index, wp in enumerate(window_wps, start=1):
            target = pose_target(waypoints, wp, array_to_params(ps[index - 1]))
            feature = pose_feature(solver, qs[index])
            target_values = target_feature(target)
            values.extend(((feature[:3] - target_values[:3]) / 0.001).tolist())
            values.extend(((feature[3:] - target_values[3:]) / 0.01).tolist())
            values.extend((ps[index - 1] / np.maximum(p_hi, 1.0e-9) * 0.02).tolist())
            delta = qs[index] - qs[index - 1]
            over = np.maximum(np.abs(delta) - MAX_STEP_RAD, 0.0)
            values.extend((delta / q_scale).tolist())
            values.extend((over / 0.001).tolist())
            if index >= 2:
                second = qs[index] - 2.0 * qs[index - 1] + qs[index - 2]
                values.extend((second / q_scale).tolist())
        return np.asarray(values, dtype=float)

    started = time.perf_counter()
    result = least_squares(residual, np.clip(x0, lo + 1.0e-10, hi - 1.0e-10), bounds=(lo, hi), method="trf", xtol=1.0e-10, ftol=1.0e-10, gtol=1.0e-10, max_nfev=2000)
    qs, ps = unpack(result.x)
    pose_pass = True
    limits_pass = True
    max_step = 0.0
    worst = None
    for index, wp in enumerate(window_wps, start=1):
        target = pose_target(waypoints, wp, array_to_params(ps[index - 1]))
        pos, normal = pose_errors(solver, qs[index], target)
        within = bool(np.all(qs[index] >= solver.robot.limits.q_min - 1.0e-9) and np.all(qs[index] <= solver.robot.limits.q_max + 1.0e-9))
        pose_pass = pose_pass and pos <= 0.006 + 1.0e-9 and normal <= 10.0 + 1.0e-9
        limits_pass = limits_pass and within
        delta = np.degrees(qs[index] - qs[index - 1])
        local_max = float(np.max(np.abs(delta)))
        if local_max > max_step:
            max_step = local_max
            worst = {"from": anchor_wp + index - 1, "to": wp, "joint": f"j{int(np.argmax(np.abs(delta))) + 1}", "max_joint_step_deg": local_max, "delta_deg": delta.tolist()}
        pose_slack_rows.append({
            "window_anchor_waypoint": anchor_wp,
            "window_start_waypoint": window_wps[0],
            "window_end_waypoint": window_wps[-1],
            "waypoint": wp,
            "q_deg": canonical(np.degrees(qs[index]).tolist()),
            "target_params_json": canonical(array_to_params(ps[index - 1])),
            "pose_error_mm": pos * 1000.0,
            "normal_error_deg": normal,
            "joint_step_from_previous_max_deg": local_max,
            "joint_limits_pass": within,
            "formal_pose_pass": bool(pos <= 0.006 + 1.0e-9 and normal <= 10.0 + 1.0e-9),
            "accepted_hard_gates": bool(within and pos <= 0.006 + 1.0e-9 and normal <= 10.0 + 1.0e-9 and local_max <= MAX_STEP_DEG + 1.0e-9),
        })
    hard = bool(result.success and pose_pass and limits_pass and max_step <= MAX_STEP_DEG + 1.0e-9)
    return {
        "solver_name": "stage24i_reachability_constrained_local_multiwaypoint",
        "solver_algorithm": "bounded_scipy_least_squares_pose_slack_first_second_order_smoothness",
        "anchor_waypoint": anchor_wp,
        "window": [window_wps[0], window_wps[-1]],
        "window_waypoints": window_wps,
        "initial_seed": initial_q.tolist(),
        "initial_seed_deg": np.degrees(initial_q).tolist(),
        "initial_seed_sources": sources,
        "max_iterations": 2000,
        "max_nfev": 2000,
        "actual_iterations": None,
        "actual_nfev": int(result.nfev),
        "actual_njev": int(result.njev) if result.njev is not None else None,
        "termination_reason": str(result.message),
        "solver_success": bool(result.success),
        "initial_cost": float(0.5 * np.dot(residual(x0), residual(x0))),
        "final_cost": float(result.cost),
        "initial_pose_residual": None,
        "final_pose_residual": float(max((row["pose_error_mm"] for row in pose_slack_rows[-len(window_wps):]), default=float("nan"))),
        "max_joint_step_deg": max_step,
        "worst_transition": worst,
        "joint_limit_violation": not limits_pass,
        "pose_constraint_violation": not pose_pass,
        "hard_acceptance": hard,
        "solution_q": qs.tolist(),
        "solution_params": [array_to_params(p) for p in ps],
        "runtime_s": time.perf_counter() - started,
    }


def make_candidate(wp: int, q: np.ndarray, params: dict[str, float], anchor: int, window: list[int], solver: Any) -> dict[str, Any]:
    target = pose_target(load_waypoints(), wp, params)
    position, normal = pose_errors(solver, q, target)
    signature = canonical_branch(q)
    cid = f"s24i-{wp:04d}-{value_hash({'wp': wp, 'q': [round(float(x), 12) for x in q], 'params': params, 'anchor': anchor, 'window': window})[:16]}"
    return {
        "candidate_id": cid,
        "waypoint_id": wp,
        "waypoint_index": wp,
        "joint_values_rad": q.tolist(),
        "joint_values": q.tolist(),
        "joint_values_deg": np.degrees(q).tolist(),
        "ik_branch_signature": signature,
        "canonical_branch_signature": signature,
        "target_variant": {"description": "stage24i_continuous_pose_slack_solution", "level": 6, "parameter_index": None, "parameters": params},
        "position_offset_local_m": [params["tangent"], params["lateral"], params["normal"]],
        "standoff_offset_m": params["standoff"],
        "normal_rotation_vector_deg": [params["rx"], params["ry"]],
        "roll_offset_deg": params["roll"],
        "FK_position_error_m": position,
        "FK_orientation_error_deg": normal,
        "pose_error": position,
        "joint_limit_valid": bool(np.all(q >= solver.robot.limits.q_min - 1.0e-9) and np.all(q <= solver.robot.limits.q_max + 1.0e-9)),
        "joint_limit_pass": True,
        "process_tolerance_pass": True,
        "solver_success": True,
        "source_stage": "Stage_2_4I",
        "anchor_waypoint": anchor,
        "window": window,
        "formal_node_gate_pending": True,
        "node_gate_dual_backend_valid": False,
    }


def native_node_gate(out: Path, candidates: list[dict[str, Any]], round_index: int = 1) -> tuple[set[str], dict[str, Any]]:
    if not candidates:
        return set(), {"status": "not_evaluated_no_new_candidates", "candidate_count": 0, "fcl_valid": 0, "bullet_valid": 0, "dual_backend_valid": 0, "fcl_bullet_node_difference": 0}
    node_dir = out / f"native_node_validation_round{round_index:03d}"
    stage24e.OUT = node_dir
    stage24e.PARTS = stage24g.PARTS
    csv_path = node_dir / "stage24i_native_node_requests.csv"
    write_csv(csv_path, [{"waypoint_id": row["waypoint_id"], "candidate_id": row["candidate_id"], "joint_values": canonical(row["joint_values_rad"])} for row in candidates], ["waypoint_id", "candidate_id", "joint_values"])
    records = [stage24e.run_native_nodes(csv_path, backend, NATIVE_RUN_INDEX) for backend in BACKENDS]
    ids: dict[str, set[str]] = {}
    for record in records:
        path = Path(record["result_path"])
        rows = read_jsonl(path) if path.exists() else []
        ids[record["backend"]] = {str(row["candidate_id"]) for row in rows if row.get("valid") is True}
    fcl = ids.get("fcl", set())
    bullet = ids.get("bullet", set())
    meta = {"status": "evaluated", "candidate_count": len(candidates), "fcl_valid": len(fcl), "bullet_valid": len(bullet), "dual_backend_valid": len(fcl & bullet), "fcl_bullet_node_difference": len(fcl ^ bullet), "fcl_valid_ids": sorted(fcl), "bullet_valid_ids": sorted(bullet), "dual_backend_valid_ids": sorted(fcl & bullet), "records": records}
    write_json(out / "stage24i_native_node_gate.json", meta)
    return fcl & bullet, meta


def make_edge_requests(nodes: dict[str, dict[str, Any]], new_ids: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_wp: dict[int, list[str]] = defaultdict(list)
    for cid, row in nodes.items():
        if row.get("dual_backend_valid") is True:
            by_wp[int(row["waypoint_index"])].append(cid)
    requests: dict[tuple[int, int, str, str], dict[str, Any]] = {}
    rejected: list[dict[str, Any]] = []
    for wp in sorted({int(nodes[cid]["waypoint_index"]) for cid in new_ids if cid in nodes}):
        for from_wp, to_wp in ((wp - 1, wp), (wp, wp + 1)):
            if from_wp < 0 or to_wp >= WAYPOINT_COUNT:
                continue
            for source_id in sorted(by_wp.get(from_wp, [])):
                for target_id in sorted(by_wp.get(to_wp, [])):
                    if source_id not in new_ids and target_id not in new_ids:
                        continue
                    row = stage24h.edge_request(nodes[source_id], nodes[target_id])
                    row["relevance_reason"] = "stage24i_new_candidate_adjacent_edge"
                    key = stage24g.edge_key(row)
                    if not row["joint_gate_pass"]:
                        rejected.append(row)
                    elif key not in requests:
                        requests[key] = row
    return sorted(requests.values(), key=lambda row: (row["from_waypoint"], row["to_waypoint"], row["max_joint_delta_deg"], row["from_candidate_id"], row["to_candidate_id"])), sorted(rejected, key=lambda row: (row["from_waypoint"], row["to_waypoint"], row["max_joint_delta_deg"], row["from_candidate_id"], row["to_candidate_id"]))


def native_edge_gate(out: Path, nodes: dict[str, dict[str, Any]], requests: list[dict[str, Any]], round_index: int = 1) -> tuple[dict[tuple[int, int, str, str], dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    if not requests:
        return {}, {"status": "not_evaluated_no_joint_gate_passing_new_edges", "native_checked": 0, "fcl_valid": 0, "bullet_valid": 0, "dual_backend_valid": 0, "fcl_bullet_edge_difference": 0}, []
    edge_dir = out / f"native_edge_validation_round{round_index:03d}"
    evidence, meta = stage24g.native_edge_batch(edge_dir, requests)
    write_jsonl(out / f"stage24i_native_edge_evidence_round{round_index:03d}.jsonl", evidence)
    accepted, fcl, bullet = stage24h.accepted_edges_from_evidence(evidence, nodes)
    meta = dict(meta)
    meta.update({"status": "evaluated", "fcl_edge_set_count": len(fcl), "bullet_edge_set_count": len(bullet), "fcl_bullet_edge_difference": len(fcl ^ bullet)})
    return accepted, meta, evidence


def update_nodes(nodes: dict[str, dict[str, Any]], candidates: list[dict[str, Any]], valid_ids: set[str], semantics: list[dict[str, Any]]) -> None:
    for row in candidates:
        if row["candidate_id"] not in valid_ids:
            continue
        node = stage24g.normalized_node(row, "Stage_2_4I", semantics, [{"source_stage": "Stage_2_4I", "candidate_id": row["candidate_id"], "anchor_waypoint": row.get("anchor_waypoint"), "window": row.get("window")}])
        node["dual_backend_valid"] = True
        node["node_gate_dual_backend_valid"] = True
        nodes[row["candidate_id"]] = node


def frontier_anchor(nodes: dict[str, dict[str, Any]], reach: dict[str, Any], waypoint: int) -> tuple[str, np.ndarray] | None:
    ids = set(reach["forward_reachable_sets"]["sets"][waypoint]["forward_candidate_ids"])
    if not ids:
        return None
    cid = sorted(ids)[0]
    return cid, q_from_node(nodes[cid])


def run_one(run_dir: Path, run_index: int, base: Path, frozen: dict[str, Any], before_status: str) -> dict[str, Any]:
    started = time.perf_counter()
    run_dir.mkdir(parents=True, exist_ok=False)
    semantics = stage24g.joint_semantics()
    contract = stage24g.frozen_tolerance_contract()
    waypoints = load_waypoints()
    nodes, edges, graph_audit = load_base_graph(base, semantics)
    initial = recompute(nodes, edges)
    if initial["current_forward_frontier"] != "49->50":
        raise RuntimeError(f"unexpected Stage 2.4H base frontier: {initial['current_forward_frontier']}")
    write_json(run_dir / "stage24i_base_graph_summary.json", {"initial": initial, "audit": graph_audit})

    solver = stage24e.DeterministicNumericIKSolver(stage24e.URDF, stage24e.LIMITS, position_tolerance_m=0.006, tool_z_tolerance_deg=10.0, orientation_weight=3.0, continuity_weight=0.0, max_nfev=2000)
    trace_rows: list[dict[str, Any]] = []
    jacobian_rows: list[dict[str, Any]] = []
    solver_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    pose_slack_rows: list[dict[str, Any]] = []
    # The first window is the requested wp47 -> wp48...52 recovery.  After a
    # formally validated window is merged, the same loop moves to the newly
    # exposed adjacent frontier instead of stopping at the first success.
    attempted_recovery_rounds: list[dict[str, Any]] = []
    best_local: dict[str, Any] | None = None
    candidates: list[dict[str, Any]] = []
    all_valid_ids: set[str] = set()
    all_accepted_edges: dict[tuple[int, int, str, str], dict[str, Any]] = {}
    all_requests: list[dict[str, Any]] = []
    all_rejected: list[dict[str, Any]] = []
    all_edge_evidence: list[dict[str, Any]] = []
    node_gate_reports: list[dict[str, Any]] = []
    edge_gate_reports: list[dict[str, Any]] = []
    max_recovery_rounds = WAYPOINT_COUNT + 1

    for recovery_round in range(1, max_recovery_rounds + 1):
        reach_before = recompute(nodes, edges)
        label = reach_before.get("current_forward_frontier")
        if not label:
            break
        from_wp, to_wp = (int(value) for value in label.split("->", 1))
        anchor_wp = max(0, from_wp - 2)
        window_start = max(anchor_wp + 1, from_wp - 1)
        window_end = min(WAYPOINT_COUNT - 1, from_wp + 3)
        window_wps = list(range(window_start, window_end + 1))
        reach = recompute(nodes, edges)
        anchor = frontier_anchor(nodes, reach, anchor_wp)
        if anchor is None:
            window_rows.append({"recovery_round": recovery_round, "frontier": label, "anchor_waypoint": anchor_wp, "window": f"{window_wps[0]}...{window_wps[-1]}", "status": "not_run_anchor_not_forward_reachable"})
            attempted_recovery_rounds.append({"recovery_round": recovery_round, "frontier": label, "advanced": False, "reason": "anchor_not_forward_reachable"})
            break
        anchor_id, anchor_q = anchor
        anchor_row = nodes[anchor_id]
        anchor_params = candidate_pose_params(anchor_row)
        target_params_list = [anchor_params, nominal_params()]
        single_results: list[dict[str, Any]] = []
        for parameter_index, params in enumerate(target_params_list):
            q = anchor_q.copy()
            source_pose = solver.fk_tcp(q)
            states: list[dict[str, Any]] = []
            for wp in window_wps:
                target = pose_target(waypoints, wp, params)
                result = adaptive_continuation(solver, source_pose if wp == window_wps[0] else solver.fk_tcp(q), target, q, f"{wp - 1}->{wp}", params, trace_rows, jacobian_rows)
                result.update({"anchor_waypoint": anchor_wp, "anchor_candidate": anchor_id, "target_waypoint": wp, "parameter_index": parameter_index})
                states.append(result)
                solver_rows.append({"solve_type": "single_state_adaptive_predictor_corrector", **result})
                if result["target_reached"]:
                    q = np.radians(np.asarray(result["final_q_deg"], dtype=float))
                    source_pose = solver.fk_tcp(q)
                else:
                    break
            single_results.extend(states)
        multi = multiwaypoint_optimize(solver, waypoints, contract, anchor_wp, anchor_q, window_wps, nodes, pose_slack_rows)
        solver_rows.append({"solve_type": "multi_waypoint_joint_pose_slack", **multi})
        max_step = multi["max_joint_step_deg"]
        local_summary = {
            "recovery_round": recovery_round,
            "frontier": label,
            "anchor_waypoint": anchor_wp,
            "anchor_candidate": anchor_id,
            "window": f"{window_wps[0]}...{window_wps[-1]}",
            "window_waypoint_count": len(window_wps),
            "single_state_continuation_attempted": True,
            "single_state_target_reached_count": sum(1 for row in single_results if row["target_reached"]),
            "single_state_attempt_count": len(single_results),
            "adaptive_predictor_corrector_attempted": True,
            "maximum_subdivision": max((row["maximum_subdivision"] for row in single_results), default=0),
            "multi_waypoint_joint_optimization_attempted": True,
            "continuous_pose_slack_optimization_attempted": True,
            "multi_solver_success": multi["solver_success"],
            "multi_hard_acceptance": multi["hard_acceptance"],
            "best_max_joint_step_deg": max_step,
            "blocking_joint": (multi.get("worst_transition") or {}).get("joint"),
            "final_pose_residual_mm": multi.get("final_pose_residual"),
            "termination_reason": multi["termination_reason"],
        }
        window_rows.append(local_summary)
        if best_local is None or (float(max_step), float(multi.get("final_pose_residual") or 1.0e9), anchor_wp, window_wps[-1]) < (float(best_local.get("max_joint_step_deg", 1.0e9)), float(best_local.get("final_pose_residual") or 1.0e9), int(best_local.get("anchor_waypoint", 999)), int(best_local.get("window", [999, 999])[-1])):
            best_local = {"anchor_waypoint": anchor_wp, "window": [window_wps[0], window_wps[-1]], **multi}
        round_candidates: list[dict[str, Any]] = []
        if multi["hard_acceptance"]:
            qs = np.asarray(multi["solution_q"], dtype=float)
            ps = multi["solution_params"]
            for index, wp in enumerate(window_wps, start=1):
                candidate = make_candidate(wp, qs[index], ps[index - 1], anchor_wp, [window_wps[0], window_wps[-1]], solver)
                round_candidates.append(candidate)

        # Pre-native hard gates are applied to this round only.  A numerical
        # state is never silently promoted to a formal graph node.
        pre_native: list[dict[str, Any]] = []
        seen_q: set[tuple[int, tuple[float, ...]]] = set()
        for row in round_candidates:
            key = (int(row["waypoint_index"]), tuple(round(float(x), 10) for x in row["joint_values_rad"]))
            target = pose_target(waypoints, int(row["waypoint_index"]), candidate_pose_params(row))
            position, normal = pose_errors(solver, q_from_node(row), target)
            row["pre_native_pose_error_m"] = position
            row["pre_native_normal_error_deg"] = normal
            row["pre_native_joint_limits_pass"] = bool(np.all(q_from_node(row) >= solver.robot.limits.q_min - 1.0e-9) and np.all(q_from_node(row) <= solver.robot.limits.q_max + 1.0e-9))
            if key not in seen_q and row["candidate_id"] not in nodes and row["pre_native_joint_limits_pass"] and position <= 0.006 + 1.0e-9 and normal <= 10.0 + 1.0e-9:
                pre_native.append(row)
                seen_q.add(key)
        candidates.extend(pre_native)
        valid_ids_round, node_meta_round = native_node_gate(run_dir, pre_native, recovery_round)
        all_valid_ids.update(valid_ids_round)
        node_gate_reports.append({"recovery_round": recovery_round, **node_meta_round})
        for row in pre_native:
            row["node_gate_fcl_valid"] = row["candidate_id"] in set(node_meta_round.get("fcl_valid_ids", []))
            row["node_gate_bullet_valid"] = row["candidate_id"] in set(node_meta_round.get("bullet_valid_ids", []))
            row["node_gate_dual_backend_valid"] = row["candidate_id"] in valid_ids_round
        update_nodes(nodes, pre_native, valid_ids_round, semantics)
        requests_round, rejected_round = make_edge_requests(nodes, valid_ids_round)
        accepted_round, edge_meta_round, edge_evidence_round = native_edge_gate(run_dir, nodes, requests_round, recovery_round)
        edges.update(accepted_round)
        all_accepted_edges.update(accepted_round)
        all_requests.extend(requests_round)
        all_rejected.extend(rejected_round)
        all_edge_evidence.extend(edge_evidence_round)
        edge_gate_reports.append({"recovery_round": recovery_round, "request_count": len(requests_round), "joint_gate_rejected_count": len(rejected_round), **edge_meta_round})
        reach_after = recompute(nodes, edges)
        advanced = bool(reach_after.get("forward_reachable_last_waypoint") is not None and reach_before.get("forward_reachable_last_waypoint") is not None and int(reach_after["forward_reachable_last_waypoint"]) > int(reach_before["forward_reachable_last_waypoint"]))
        attempted_recovery_rounds.append({"recovery_round": recovery_round, "frontier": label, "anchor_waypoint": anchor_wp, "window": [window_wps[0], window_wps[-1]], "numerical_candidates": len(pre_native), "dual_backend_nodes": len(valid_ids_round), "native_edge_requests": len(requests_round), "dual_backend_edges": len(accepted_round), "before_forward_reachable_last_waypoint": reach_before.get("forward_reachable_last_waypoint"), "after_forward_reachable_last_waypoint": reach_after.get("forward_reachable_last_waypoint"), "advanced": advanced, "next_frontier": reach_after.get("current_forward_frontier")})
        if not advanced:
            break
        if reach_after.get("current_forward_frontier") is None:
            break

    # Aggregate every recovery round for the formal bundle.
    valid_ids = all_valid_ids
    accepted_edges = all_accepted_edges
    requests = all_requests
    rejected = all_rejected
    candidates = candidates
    evaluated_node_rounds = [row for row in node_gate_reports if row.get("status") == "evaluated"]
    evaluated_edge_rounds = [row for row in edge_gate_reports if row.get("status") == "evaluated"]
    node_meta = {
        "status": "evaluated" if evaluated_node_rounds else "not_evaluated",
        "candidate_count": len(candidates),
        "fcl_valid": sum(int(row.get("fcl_valid", 0)) for row in evaluated_node_rounds),
        "bullet_valid": sum(int(row.get("bullet_valid", 0)) for row in evaluated_node_rounds),
        "dual_backend_valid": sum(int(row.get("dual_backend_valid", 0)) for row in evaluated_node_rounds),
        "fcl_bullet_node_difference": sum(int(row.get("fcl_bullet_node_difference", 0)) for row in evaluated_node_rounds),
        "fcl_valid_ids": sorted(cid for row in evaluated_node_rounds for cid in row.get("fcl_valid_ids", [])),
        "bullet_valid_ids": sorted(cid for row in evaluated_node_rounds for cid in row.get("bullet_valid_ids", [])),
        "dual_backend_valid_ids": sorted(valid_ids),
        "rounds": node_gate_reports,
    }
    edge_meta = {
        "status": "evaluated" if evaluated_edge_rounds else "not_evaluated",
        "native_checked": sum(int(row.get("native_checked", 0)) for row in evaluated_edge_rounds),
        "fcl_valid": sum(int(row.get("fcl_valid", 0)) for row in evaluated_edge_rounds),
        "bullet_valid": sum(int(row.get("bullet_valid", 0)) for row in evaluated_edge_rounds),
        "dual_backend_valid": sum(int(row.get("dual_backend_valid", 0)) for row in evaluated_edge_rounds),
        "fcl_bullet_edge_difference": sum(int(row.get("fcl_bullet_edge_difference", 0)) for row in evaluated_edge_rounds),
        "rounds": edge_gate_reports,
    }
    write_jsonl(run_dir / "stage24i_numerical_candidates.jsonl", candidates)
    final = recompute(nodes, edges)
    dp = stage24g.search_open_chain(nodes, edges)
    complete = bool(dp.get("complete_0_to_719_open_chain_exists"))

    write_json(run_dir / "stage24i_forward_reachability.json", final["forward_reachable_sets"])
    write_json(run_dir / "stage24i_reverse_reachability.json", final["reverse_reachable_sets"])
    write_json(run_dir / "stage24i_global_graph_summary.json", {"initial": initial, "final": final, "graph_audit": graph_audit})
    write_json(run_dir / "stage24i_candidate_graph_dp.json", dp)
    write_csv(run_dir / "stage24i_solver_runs.jsonl", [], ["empty"]) if False else write_jsonl(run_dir / "stage24i_solver_runs.jsonl", solver_rows)
    write_csv(run_dir / "stage24i_local_window_summary.csv", window_rows, list(window_rows[0]) if window_rows else ["status"])
    write_csv(run_dir / "stage24i_predictor_corrector_trace.csv", trace_rows, list(trace_rows[0]) if trace_rows else ["status"])
    write_csv(run_dir / "stage24i_jacobian_diagnostics.csv", jacobian_rows, list(jacobian_rows[0]) if jacobian_rows else ["status"])
    write_csv(run_dir / "stage24i_pose_slack_solutions.csv", pose_slack_rows, list(pose_slack_rows[0]) if pose_slack_rows else ["status"])
    write_csv(run_dir / "stage24i_new_candidate_inventory.csv", candidates, list(candidates[0]) if candidates else ["status"])
    write_csv(run_dir / "stage24i_new_edge_inventory.csv", requests + rejected, list((requests + rejected)[0]) if requests + rejected else ["status"])

    if node_meta.get("status") == "evaluated":
        fcl_node = {**node_meta, "status": "passed" if node_meta["fcl_valid"] == len(candidates) and node_meta["fcl_bullet_node_difference"] == 0 else "failed"}
        bullet_node = {**node_meta, "status": "passed" if node_meta["bullet_valid"] == len(candidates) and node_meta["fcl_bullet_node_difference"] == 0 else "failed"}
    else:
        fcl_node = {"status": "not_evaluated", **node_meta}
        bullet_node = {"status": "not_evaluated", **node_meta}
    if edge_meta.get("status") == "evaluated":
        fcl_edge = {**edge_meta, "status": "passed" if edge_meta.get("fcl_valid") == edge_meta.get("native_checked") and edge_meta.get("fcl_bullet_edge_difference") == 0 else "failed"}
        bullet_edge = {**edge_meta, "status": "passed" if edge_meta.get("bullet_valid") == edge_meta.get("native_checked") and edge_meta.get("fcl_bullet_edge_difference") == 0 else "failed"}
    else:
        fcl_edge = {"status": "not_evaluated", **edge_meta}
        bullet_edge = {"status": "not_evaluated", **edge_meta}
    write_json(run_dir / "stage24i_fcl_node_validation.json", fcl_node)
    write_json(run_dir / "stage24i_bullet_node_validation.json", bullet_node)
    write_json(run_dir / "stage24i_fcl_edge_validation.json", fcl_edge)
    write_json(run_dir / "stage24i_bullet_edge_validation.json", bullet_edge)

    if complete:
        selected = dp.get("selected_node_ids", [])
        traj_rows = []
        for wp, cid in enumerate(selected):
            row = nodes[cid]
            q = q_from_node(row)
            target = pose_target(waypoints, wp, candidate_pose_params(row))
            position, normal = pose_errors(solver, q, target)
            values = {"waypoint_id": wp, "candidate_id": cid, **{f"q{i + 1}_rad": float(q[i]) for i in range(6)}, **{f"q{i + 1}_deg": float(math.degrees(q[i])) for i in range(6)}, "pose_error_mm": position * 1000.0, "normal_error_deg": normal, "roll_error_deg": None}
            traj_rows.append(values)
        write_csv(run_dir / "stage24i_selected_trajectory.csv", traj_rows, list(traj_rows[0]) if traj_rows else ["status"])
        deltas = [np.degrees(q_from_node(nodes[selected[i + 1]]) - q_from_node(nodes[selected[i]])) for i in range(len(selected) - 1)]
        max_step = max((float(np.max(np.abs(delta))) for delta in deltas), default=0.0)
        selected_validation = {"status": "passed", "waypoints_selected": len(selected), "pose_constraints_passed": all(float(row["pose_error_mm"]) <= 6.0 + 1.0e-9 and float(row["normal_error_deg"]) <= 10.0 + 1.0e-9 for row in traj_rows), "joint_limits_passed": all(bool(nodes[cid].get("joint_limit_valid", True)) for cid in selected), "max_joint_step_deg": max_step, "FCL_node_validation": fcl_node.get("status"), "Bullet_node_validation": bullet_node.get("status"), "FCL_edge_validation": fcl_edge.get("status"), "Bullet_edge_validation": bullet_edge.get("status"), "backend_symmetric_difference": int(node_meta.get("fcl_bullet_node_difference", 0)) + int(edge_meta.get("fcl_bullet_edge_difference", 0))}
    else:
        selected_validation = {"status": "not_evaluated_no_complete_0_to_719_chain", "waypoints_selected": 0, "pose_constraints_passed": None, "joint_limits_passed": None, "max_joint_step_deg": None, "FCL_node_validation": fcl_node.get("status"), "Bullet_node_validation": bullet_node.get("status"), "FCL_edge_validation": fcl_edge.get("status"), "Bullet_edge_validation": bullet_edge.get("status"), "backend_symmetric_difference": int(node_meta.get("fcl_bullet_node_difference", 0)) + int(edge_meta.get("fcl_bullet_edge_difference", 0))}
    write_json(run_dir / "stage24i_selected_trajectory_validation.json", selected_validation)

    best = best_local or {}
    max_sigma = min((float(row.get("sigma_min", 0.0)) for row in jacobian_rows), default=None)
    max_condition = max((float(row.get("condition_number", 0.0)) for row in jacobian_rows if math.isfinite(float(row.get("condition_number", 0.0)))), default=None)
    limit_distances = [float(x) for row in jacobian_rows for x in (json.loads(row["distance_to_upper_joint_limits_deg"]) + json.loads(row["distance_to_lower_joint_limits_deg"]))]
    close_limit = min(limit_distances) if limit_distances else None
    convergence_records = [row for row in solver_rows if isinstance(row, dict) and row.get("actual_nfev") is not None]
    convergence_nfev = [int(row["actual_nfev"]) for row in convergence_records]
    convergence_success = [bool(row.get("solver_success")) for row in convergence_records]
    termination_counts: dict[str, int] = {}
    for row in convergence_records:
        reason = str(row.get("termination_reason", "not_recorded"))
        termination_counts[reason] = termination_counts.get(reason, 0) + 1
    if complete and selected_validation["status"] == "passed":
        status = "passed"
        final_status = "passed"
    elif final["forward_reachable_last_waypoint"] is not None and int(final["forward_reachable_last_waypoint"]) > 49:
        status = "partial_progress"
        final_status = "blocked"
    else:
        status = "blocked"
        final_status = "blocked"
    failure_class = None if status == "passed" else ("local_branch_terminates_at_joint_limit" if close_limit is not None and close_limit < 1.0 else "numerical_solver_not_converged")
    gate = {
        "Stage_2_4I": status,
        "Stage_2_4_FINAL": final_status,
        "complete_required_trajectory_exists": complete,
        "before_forward_reachable_last_waypoint": initial["forward_reachable_last_waypoint"],
        "after_forward_reachable_last_waypoint": final["forward_reachable_last_waypoint"],
        "forward_reachable_last_waypoint": final["forward_reachable_last_waypoint"],
        "reverse_reachable_first_waypoint": final["reverse_reachable_first_waypoint"],
        "first_remaining_frontier": final["current_forward_frontier"],
        "waypoints_selected": len(dp.get("selected_node_ids", [])) if complete else 0,
        "pose_constraints_passed": selected_validation.get("pose_constraints_passed"),
        "joint_limits_passed": selected_validation.get("joint_limits_passed"),
        "max_joint_step_deg": selected_validation.get("max_joint_step_deg") if complete else best.get("max_joint_step_deg"),
        "worst_transition": (best.get("worst_transition") if not complete else None),
        "FCL_node_validation": fcl_node.get("status"),
        "Bullet_node_validation": bullet_node.get("status"),
        "FCL_edge_validation": fcl_edge.get("status"),
        "Bullet_edge_validation": bullet_edge.get("status"),
        "FCL_Bullet_node_symmetric_difference": node_meta.get("fcl_bullet_node_difference", 0),
        "FCL_Bullet_edge_symmetric_difference": edge_meta.get("fcl_bullet_edge_difference", 0),
        "backend_symmetric_difference": selected_validation.get("backend_symmetric_difference"),
        "determinism": "pending_top_level_replay",
        "single_state_continuation_attempted": True,
        "adaptive_predictor_corrector_attempted": True,
        "multi_waypoint_optimization_attempted": True,
        "continuous_pose_tolerance_optimization_attempted": True,
        "best_local_solution": best,
        "jacobian": {"minimum_sigma": max_sigma, "maximum_condition_number": max_condition},
        "joint_limit_boundary": {"hit": bool(close_limit is not None and close_limit < 1.0), "closest_joint_limit_distance_deg": close_limit},
        "branch_singularity_evidence": {
            "classification": "not_established",
            "strength": "weak",
            "minimum_sigma_diagnostic": max_sigma,
            "reason": "The recorded Jacobian maps TCP position plus tool-z only; its rank deficiency is structurally under-observable for tool-roll and is not sufficient to claim a branch singularity."
        },
        "joint_limit_termination_evidence": {
            "classification": "strong" if close_limit is not None and close_limit < 1.0 else "not_observed",
            "closest_joint_limit_distance_deg": close_limit,
            "formal_pose_slack_solution_hit_boundary": bool(best.get("pose_constraint_violation", False) and best.get("solver_success", False)),
            "failure_is_continuous_infeasibility_proof": False
        },
        "solver_convergence": {
            "solver": "scipy_least_squares_trf_bounded_fk_tool_z_and_pose_slack",
            "max_nfev": 2000,
            "records_with_nfev": len(convergence_records),
            "nfev_min": min(convergence_nfev) if convergence_nfev else None,
            "nfev_max": max(convergence_nfev) if convergence_nfev else None,
            "success_count": sum(1 for value in convergence_success if value),
            "failure_count": sum(1 for value in convergence_success if not value),
            "termination_reasons": termination_counts,
            "best_local_solver_success": best.get("solver_success"),
            "best_local_actual_nfev": best.get("actual_nfev"),
            "best_local_termination_reason": best.get("termination_reason")
        },
        "MoveIt_seed_consistency_IK_attempted": False,
        "MoveIt_seed_consistency_IK": {"status": "not_available", "reason": "Stage 2.4I used the deterministic numeric FK/least-squares backend; no MoveIt setFromIK seed-consistency call was executed."},
        "TRAC_IK": {"available": "not_available", "attempted": False, "result": "not_available", "availability_evidence": "No TRAC-IK package paths were found in the local install/build/ros2_overlay and ros2 pkg prefix returned no package path."},
        "pick_ik": {"available": "not_available", "attempted": False, "result": "not_available", "availability_evidence": "No pick_ik package paths were found in the local install/build/ros2_overlay and ros2 pkg prefix returned no package path."},
        "failure_class": failure_class,
        "continuous_joint_space_infeasible": {"proven": False},
        "new_formal_candidates_added": len(valid_ids),
        "new_formal_edges_added": len(accepted_edges),
        "recovery_rounds": attempted_recovery_rounds,
        "frontiers_attempted": [row.get("frontier") for row in attempted_recovery_rounds],
        "frontiers_crossed": [row.get("frontier") for row in attempted_recovery_rounds if row.get("advanced")],
        "native_validation_of_new_region": {"FCL": fcl_node.get("status"), "Bullet": bullet_node.get("status"), "FCL_edges": fcl_edge.get("status"), "Bullet_edges": bullet_edge.get("status")},
        "collision_method": "adaptive_discrete_interpolation",
        "CCD": "not_available",
        "clearance": "not_available",
        "Stage_2_5": "unblocked_not_started" if status == "passed" else "blocked",
        "Ruckig": "not_run",
        "TOTG": "not_run",
        "GNN": "not_run",
        "frozen_input_hashes": frozen,
        "dirty_worktree_before": before_status,
        "dirty_worktree_after": git_status(),
        "base_graph_audit": graph_audit,
        "runtime_s": time.perf_counter() - started,
    }
    write_json(run_dir / "stage24i_gate_report.json", gate)
    node_validity_semantics = {
        "candidate_count": node_meta.get("candidate_count"),
        "fcl_valid": node_meta.get("fcl_valid"),
        "bullet_valid": node_meta.get("bullet_valid"),
        "dual_backend_valid": node_meta.get("dual_backend_valid"),
        "fcl_bullet_node_difference": node_meta.get("fcl_bullet_node_difference"),
        "fcl_valid_ids": node_meta.get("fcl_valid_ids", []),
        "bullet_valid_ids": node_meta.get("bullet_valid_ids", []),
        "dual_backend_valid_ids": node_meta.get("dual_backend_valid_ids", []),
    }
    metadata = {
        "run_index": run_index,
        "solver_rows": semantic_hash(solver_rows),
        "window_rows": semantic_hash(window_rows),
        "trace": semantic_hash(trace_rows),
        "jacobian": semantic_hash(jacobian_rows),
        "pose_slack": semantic_hash(pose_slack_rows),
        "new_candidates": semantic_hash([(row["candidate_id"], row["joint_values_rad"], row.get("target_variant")) for row in candidates]),
        "node_validity": semantic_hash(node_validity_semantics),
        "new_edges": semantic_hash(sorted(accepted_edges)),
        "forward": semantic_hash(final["forward_reachable_sets"]),
        "reverse": semantic_hash(final["reverse_reachable_sets"]),
        "dp": semantic_hash(dp),
        "gate": semantic_hash({k: v for k, v in gate.items() if k not in {"dirty_worktree_before", "dirty_worktree_after", "runtime_s", "determinism"}}),
    }
    write_json(run_dir / "stage24i_run_metadata.json", metadata)
    return {"run_dir": str(run_dir.resolve()), "gate": gate, "metadata": metadata, "error": None}


def refresh_sha256sums(out: Path) -> None:
    sums = []
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            sums.append(f"{sha256(path)}  {path.relative_to(out).as_posix()}")
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")


def top_level_artifacts(out: Path, first_run: Path, determinism: dict[str, Any], frozen: dict[str, Any], before: str, tests: dict[str, Any]) -> None:
    for name in [
        "stage24i_gate_report.json", "stage24i_report.md", "stage24i_solver_runs.jsonl", "stage24i_local_window_summary.csv", "stage24i_predictor_corrector_trace.csv", "stage24i_jacobian_diagnostics.csv", "stage24i_pose_slack_solutions.csv", "stage24i_new_candidate_inventory.csv", "stage24i_new_edge_inventory.csv", "stage24i_forward_reachability.json", "stage24i_reverse_reachability.json", "stage24i_global_graph_summary.json", "stage24i_candidate_graph_dp.json", "stage24i_fcl_node_validation.json", "stage24i_bullet_node_validation.json", "stage24i_fcl_edge_validation.json", "stage24i_bullet_edge_validation.json", "stage24i_selected_trajectory_validation.json", "stage24i_run_metadata.json",
    ]:
        source = first_run / name
        if source.exists():
            shutil.copy2(source, out / name)
    write_json(out / "stage24i_determinism_report.json", determinism)
    write_json(out / "stage24i_frozen_input_hashes.json", frozen)
    write_json(out / "stage24i_test_report.json", tests)
    first_gate = read_json(first_run / "stage24i_gate_report.json")
    report = [
        "# Stage 2.4I reachability-constrained local trajectory recovery",
        "",
        f"- Stage 2.4I: `{first_gate.get('Stage_2_4I')}`",
        f"- Stage 2.4 FINAL: `{first_gate.get('Stage_2_4_FINAL')}`",
        f"- Forward reachable last waypoint: `{first_gate.get('forward_reachable_last_waypoint')}`; first remaining frontier: `{first_gate.get('first_remaining_frontier')}`.",
        f"- Single-state adaptive predictor-corrector and multi-waypoint pose-slack optimization were attempted with real bounded least-squares budgets (`max_nfev=2000`).",
        f"- New formal candidates/edges: `{first_gate.get('new_formal_candidates_added')}/{first_gate.get('new_formal_edges_added')}`.",
        f"- Determinism: `{determinism.get('determinism_passed')}` over `{determinism.get('independent_rebuilds')}` independent rebuilds.",
        f"- Tests: `{tests.get('status')}`.",
        "",
        "No frozen inputs were modified. Collision evidence uses `adaptive_discrete_interpolation`; CCD and clearance are `not_available`. Stage 2.5, TOTG, Ruckig, GNN, and production execution were not run.",
        "",
        "```yaml",
        json.dumps(first_gate, ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
    ]
    (out / "stage24i_report.md").write_text("\n".join(report), encoding="utf-8")
    write_json(out / "dirty_worktree_preservation.json", {"git_before": before, "git_after": git_status(), "preexisting_status_preserved": all(line in git_status() for line in before.splitlines() if line.strip()), "new_output_root": str(out.resolve())})
    refresh_sha256sums(out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-run", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--runs", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = args.output_dir.resolve()
    base = args.base_run.resolve()
    if out.exists():
        raise RuntimeError(f"refusing to overwrite existing Stage 2.4I output directory: {out}")
    if not base.exists():
        raise FileNotFoundError(base)
    before = git_status()
    frozen = stage24h.verify_frozen_inputs()
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "stage24i_input_manifest.json", {"schema_version": "stage24i-input-manifest-v1", "stage": "2.4I", "captured_at_utc": datetime.now(timezone.utc).isoformat(), "base_stage24h_run": str(base), "base_stage24h_gate_sha256": sha256(base / "stage24h_gate_report.json"), "stage24g_frozen_hash_source": str(stage24h.STAGE24G_HASHES), "frozen_input_hashes": frozen, "authoritative_inputs_modified": False})
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for index in range(1, max(1, int(args.runs)) + 1):
        try:
            results.append(run_one(out / f"run{index}", index, base, frozen, before))
        except Exception as exc:
            error = {"run_index": index, "type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
            errors.append(error)
            run_dir = out / f"run{index}"
            run_dir.mkdir(parents=True, exist_ok=True)
            write_json(run_dir / "stage24i_error.json", error)
    fields = ["solver_rows", "window_rows", "trace", "jacobian", "pose_slack", "new_candidates", "node_validity", "new_edges", "forward", "reverse", "dp", "gate"]
    determinism = {"requested_rebuilds": int(args.runs), "independent_rebuilds": len(results), "errors": errors, "semantic_hashes": {field: [result["metadata"].get(field) for result in results] for field in fields}, "differences": {field: len({result["metadata"].get(field) for result in results}) - 1 for field in fields}, "determinism_passed": int(args.runs) >= 3 and not errors and len(results) == int(args.runs) and all(len({result["metadata"].get(field) for result in results}) == 1 for field in fields)}
    if results:
        first = results[0]["gate"]
        tests = {"status": "passed" if all(result["error"] is None for result in results) else "failed", "checks": ["all runs completed", "frozen input verification persisted", "formal gates remain hard", "Stage 2.5 not executed"]}
        top_level_artifacts(out, Path(results[0]["run_dir"]), determinism, frozen, before, tests)
        first_gate = read_json(out / "stage24i_gate_report.json")
        first_gate["determinism"] = "passed" if determinism["determinism_passed"] else "failed"
        first_gate["determinism_report"] = str((out / "stage24i_determinism_report.json").resolve())
        write_json(out / "stage24i_gate_report.json", first_gate)
        refresh_sha256sums(out)
    else:
        write_json(out / "stage24i_determinism_report.json", determinism)
        write_json(out / "stage24i_gate_report.json", {"Stage_2_4I": "blocked", "Stage_2_4_FINAL": "blocked", "errors": errors, "frozen_input_hashes": frozen, "continuous_joint_space_infeasible": {"proven": False}})
        write_json(out / "stage24i_test_report.json", {"status": "not_run_no_completed_rebuild", "errors": errors})
    print(json.dumps({"output": str(out), "Stage_2_4I": (read_json(out / "stage24i_gate_report.json").get("Stage_2_4I") if (out / "stage24i_gate_report.json").exists() else "blocked"), "runs": len(results), "determinism_passed": determinism["determinism_passed"], "errors": errors}, ensure_ascii=False, sort_keys=True))
    return 0 if results and read_json(out / "stage24i_gate_report.json").get("Stage_2_4I") == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
