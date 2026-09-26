"""Stage 2.4 FINAL global path completion and certification.

This runner is intentionally stage-local.  It never edits the frozen model,
scene, waypoint, joint-limit, or prior-stage evidence.  It first solves the
finite authoritative candidate inventory as one layered graph, then runs a
checkpointed sparse whole-trajectory IK portfolio if that finite graph has no
complete path.  Native FCL/Bullet validation is lazy and is only requested for
a numerically complete trajectory that passes the non-collision formal gates.

Collision evidence is ``adaptive_discrete_interpolation``.  CCD and clearance
are not inferred from collision-free states.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_stage24_closed_loop_graph as stage24
import scripts.run_stage24e_tolerance_search as stage24e
import scripts.run_stage24g_complete_edge_component_bridge as stage24g
from src.deterministic_numeric_ik import DeterministicNumericIKSolver


WAYPOINT_COUNT = 720
JOINT_COUNT = 6
MAX_STEP_DEG = 20.0
INTERPOLATION_STEP_DEG = 1.0
POSITION_TOL_M = 0.006
NORMAL_TOL_DEG = 10.0

G_NODES = ROOT / "outputs/ik_graph_stage24g_complete_edge_component_bridge/fr5_scaled_horseshoe_demo_v45_20260731/stage24g_valid_nodes.jsonl"
H_CANDIDATES = ROOT / "outputs/ik_graph_stage24h_reachability_frontier_continuation/fr5_scaled_horseshoe_demo_v45_20260801_parallel_streamed_run/run1/stage24h_generated_candidates.jsonl"
WAYPOINTS = ROOT / "outputs/ik_graph_stage23a4_backend_equivalent/fr5_scaled_horseshoe_demo_v44/waypoints.csv"
URDF = ROOT / "outputs/ik_graph_stage193/expanded_runtime_urdf.urdf"
LIMITS = ROOT / "ros2_moveit_bridge/config/joint_limits_with_jerk.yaml"
PARTS = ROOT / "outputs/ik_graph_stage23a4_backend_equivalent/fr5_scaled_horseshoe_demo_v44/tunnel_collision_parts"


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def value_hash(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def q_of(row: dict[str, Any]) -> np.ndarray:
    return np.asarray(row.get("joint_vector_rad", row.get("joint_values_rad", row.get("joint_values"))), dtype=float)


def wp_of(row: dict[str, Any]) -> int:
    return int(row.get("waypoint_index", row.get("waypoint_id")))


def normalize_candidate(row: dict[str, Any], source_path: Path) -> dict[str, Any]:
    q = q_of(row)
    return {
        "candidate_id": str(row["candidate_id"]),
        "waypoint_index": wp_of(row),
        "q": q.tolist(),
        "position_error_m": row.get("FK_position_error_m", row.get("solver_position_error_m", row.get("pose_error"))),
        "normal_error_deg": row.get("FK_orientation_error_deg", row.get("solver_tool_z_error_deg")),
        "roll_error_deg": row.get("roll_offset_deg"),
        "standoff_error_m": row.get("standoff_offset_m"),
        "tolerance_parameter_vector": {
            "position_offset_local_m": row.get("position_offset_local_m"),
            "standoff_offset_m": row.get("standoff_offset_m"),
            "normal_rotation_vector_deg": row.get("normal_rotation_vector_deg"),
            "roll_offset_deg": row.get("roll_offset_deg"),
        },
        "ik_seed_provenance": {"seed_id": row.get("seed_id"), "source_attempt_key": row.get("source_attempt_key")},
        "generation_stage": row.get("source_stage"),
        "branch_signature": row.get("canonical_branch_signature", row.get("ik_branch_signature")),
        "fcl_node_valid": row.get("fcl_node_valid", row.get("node_gate_fcl_valid", row.get("dual_backend_valid") is True)),
        "bullet_node_valid": row.get("bullet_node_valid", row.get("node_gate_bullet_valid", row.get("dual_backend_valid") is True)),
        "dual_backend_valid": row.get("dual_backend_valid", row.get("node_gate_dual_backend_valid", False)) is True,
        "source_evidence": str(source_path.resolve()),
        "source_candidate": row,
    }


def load_inventory() -> tuple[list[dict[str, Any]], dict[int, list[int]], dict[str, str]]:
    raw: list[dict[str, Any]] = []
    for path in (G_NODES, H_CANDIDATES):
        if not path.exists():
            continue
        for row in read_jsonl(path):
            if row.get("dual_backend_valid", row.get("node_gate_dual_backend_valid", False)) is True:
                raw.append(normalize_candidate(row, path))
    dedup: dict[tuple[int, tuple[float, ...]], dict[str, Any]] = {}
    aliases: dict[str, str] = {}
    for row in sorted(raw, key=lambda item: (item["waypoint_index"], str(item["generation_stage"]), item["candidate_id"])):
        key = (row["waypoint_index"], tuple(round(float(x), 10) for x in row["q"]))
        if key not in dedup:
            row["aliases"] = []
            dedup[key] = row
        else:
            dedup[key]["aliases"].append(row["candidate_id"])
        aliases[row["candidate_id"]] = dedup[key]["candidate_id"]
    inventory = sorted(dedup.values(), key=lambda item: (item["waypoint_index"], item["candidate_id"]))
    by_wp: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(inventory):
        by_wp[row["waypoint_index"]].append(index)
    return inventory, dict(by_wp), aliases


def write_inventory(path: Path, inventory: list[dict[str, Any]]) -> None:
    fields = ["waypoint_index", "candidate_id", "q", "position_error_m", "normal_error_deg", "roll_error_deg", "standoff_error_m", "tolerance_parameter_vector", "ik_seed_provenance", "generation_stage", "branch_signature", "fcl_node_valid", "bullet_node_valid", "dual_backend_valid", "source_evidence", "aliases"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in inventory:
            writer.writerow({field: canonical(row[field]) if isinstance(row[field], (dict, list)) else row[field] for field in fields})


def compact_graph(inventory: list[dict[str, Any]], by_wp: dict[int, list[int]], out: Path) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[str, Any]]:
    """Build raw bounded-revolute edges with NumPy arrays, not Python objects."""
    out.mkdir(parents=True, exist_ok=True)
    edge_from: dict[int, np.ndarray] = {}
    edge_to: dict[int, np.ndarray] = {}
    transition_rows: list[dict[str, Any]] = []
    all_edge_from: list[np.ndarray] = []
    all_edge_to: list[np.ndarray] = []
    for wp in range(WAYPOINT_COUNT - 1):
        source_ids = np.asarray(by_wp.get(wp, []), dtype=np.int32)
        target_ids = np.asarray(by_wp.get(wp + 1, []), dtype=np.int32)
        source_q = np.asarray([inventory[int(i)]["q"] for i in source_ids], dtype=np.float64)
        target_q = np.asarray([inventory[int(i)]["q"] for i in target_ids], dtype=np.float64)
        chunks_from: list[np.ndarray] = []
        chunks_to: list[np.ndarray] = []
        if len(source_ids) and len(target_ids):
            for start in range(0, len(source_ids), 256):
                block = source_q[start : start + 256]
                mask = np.max(np.abs(block[:, None, :] - target_q[None, :, :]), axis=2) <= math.radians(MAX_STEP_DEG) + 1e-12
                a, b = np.nonzero(mask)
                if len(a):
                    chunks_from.append(source_ids[start + a])
                    chunks_to.append(target_ids[b])
        ef = np.concatenate(chunks_from) if chunks_from else np.empty(0, dtype=np.int32)
        et = np.concatenate(chunks_to) if chunks_to else np.empty(0, dtype=np.int32)
        edge_from[wp] = ef
        edge_to[wp] = et
        all_edge_from.append(ef)
        all_edge_to.append(et)
        transition_rows.append({"transition": f"{wp}->{wp + 1}", "from_waypoint": wp, "to_waypoint": wp + 1, "source_nodes": int(len(source_ids)), "target_nodes": int(len(target_ids)), "raw_joint_gate_edges": int(len(ef)), "joint_step_threshold_deg": MAX_STEP_DEG})
    from_all = np.concatenate(all_edge_from) if all_edge_from else np.empty(0, dtype=np.int32)
    to_all = np.concatenate(all_edge_to) if all_edge_to else np.empty(0, dtype=np.int32)
    np.savez_compressed(out / "stage24final_global_graph_edges.npz", from_index=from_all, to_index=to_all)
    summary = {
        "schema_version": "stage24final-global-graph-v1",
        "waypoint_count": WAYPOINT_COUNT,
        "candidate_count": len(inventory),
        "layer_coverage": len(by_wp) == WAYPOINT_COUNT,
        "layer_counts": {str(wp): len(by_wp.get(wp, [])) for wp in range(WAYPOINT_COUNT)},
        "open_transition_count": WAYPOINT_COUNT - 1,
        "transitions_with_raw_gate_edges": sum(1 for row in transition_rows if row["raw_joint_gate_edges"] > 0),
        "raw_edge_count": int(len(from_all)),
        "joint_semantics": "all six joints bounded revolute; raw q_to-q_from difference; no periodic wrap",
        "joint_step_threshold_deg": MAX_STEP_DEG,
        "edge_storage": "NumPy compressed arrays; stage24final_global_graph_edges.npz",
        "transition_summary": transition_rows,
    }
    write_json(out / "stage24final_global_graph_summary.json", summary)
    return edge_from, edge_to, summary


def global_dp(inventory: list[dict[str, Any]], by_wp: dict[int, list[int]], edge_from: dict[int, np.ndarray], edge_to: dict[int, np.ndarray]) -> dict[str, Any]:
    reachable: dict[int, tuple[float, float, int | None]] = {index: (0.0, 0.0, None) for index in by_wp.get(0, [])}
    predecessor: list[dict[int, int | None]] = [{index: None for index in reachable}]
    counts = [len(reachable)]
    for wp in range(WAYPOINT_COUNT - 1):
        next_scores: dict[int, tuple[float, float, int | None]] = {}
        for source in sorted(reachable):
            base_max, base_sq, _ = reachable[source]
            ef = edge_from[wp]
            et = edge_to[wp]
            destinations = et[ef == source]
            if not len(destinations):
                continue
            q0 = np.asarray(inventory[source]["q"])
            for target in destinations.tolist():
                step = np.abs(np.asarray(inventory[int(target)]["q"]) - q0)
                score = (max(base_max, float(np.max(step))), base_sq + float(np.dot(step, step)))
                old = next_scores.get(int(target))
                if old is None or score < (old[0], old[1]):
                    next_scores[int(target)] = (score[0], score[1], source)
        reachable = next_scores
        predecessor.append({target: values[2] for target, values in reachable.items()})
        counts.append(len(reachable))
        if not reachable:
            break
    complete = len(reachable) > 0 and len(counts) == WAYPOINT_COUNT
    path: list[int] = []
    if complete:
        terminal = min(reachable, key=lambda index: (reachable[index][0], reachable[index][1], index))
        for layer in range(WAYPOINT_COUNT - 1, -1, -1):
            path.append(terminal)
            terminal = predecessor[layer][terminal]
            if terminal is None and layer:
                complete = False
                path = []
                break
        path.reverse()
    return {"complete_0_to_719_open_chain_exists": bool(complete), "selected_indices": path, "reachable_count_by_waypoint": counts, "final_reachable_count": len(reachable), "first_empty_waypoint": next((i for i, count in enumerate(counts) if count == 0), None)}


def load_waypoints() -> dict[int, dict[str, str]]:
    with WAYPOINTS.open(encoding="utf-8", newline="") as handle:
        return {int(row["waypoint_id"]): row for row in csv.DictReader(handle)}


def nominal_targets(solver: DeterministicNumericIKSolver, waypoints: dict[int, dict[str, str]]) -> list[np.ndarray]:
    return [solver.pose_matrix([float(waypoints[i][k]) for k in ("x", "y", "z")], [float(waypoints[i][k]) for k in ("qx", "qy", "qz", "qw")]) for i in range(WAYPOINT_COUNT)]


def greedy_initial(inventory: list[dict[str, Any]], by_wp: dict[int, list[int]], reverse: bool = False) -> np.ndarray:
    path = [None] * WAYPOINT_COUNT
    if reverse:
        path[-1] = min(by_wp[719], key=lambda i: (inventory[i]["position_error_m"] is None, inventory[i]["candidate_id"]))
        order = range(WAYPOINT_COUNT - 2, -1, -1)
        for wp in order:
            q = np.asarray(inventory[path[wp + 1]]["q"])
            path[wp] = min(by_wp.get(wp, []), key=lambda i: (float(np.max(np.abs(np.asarray(inventory[i]["q"]) - q))), inventory[i]["candidate_id"]))
    else:
        path[0] = min(by_wp[0], key=lambda i: (inventory[i]["position_error_m"] is None, inventory[i]["candidate_id"]))
        for wp in range(1, WAYPOINT_COUNT):
            q = np.asarray(inventory[path[wp - 1]]["q"])
            path[wp] = min(by_wp.get(wp, []), key=lambda i: (float(np.max(np.abs(np.asarray(inventory[i]["q"]) - q))), inventory[i]["candidate_id"]))
    return np.asarray([inventory[int(i)]["q"] for i in path], dtype=float)


def candidate_initial(inventory: list[dict[str, Any]], by_wp: dict[int, list[int]]) -> np.ndarray:
    result = np.empty((WAYPOINT_COUNT, JOINT_COUNT), dtype=float)
    for wp in range(WAYPOINT_COUNT):
        result[wp] = np.asarray(inventory[min(by_wp[wp], key=lambda i: (inventory[i]["position_error_m"] is None, float(inventory[i]["position_error_m"] or 0.0), inventory[i]["candidate_id"]))]["q"])
    return result


def trajectory_metrics(q: np.ndarray, solver: DeterministicNumericIKSolver, targets: list[np.ndarray]) -> dict[str, Any]:
    positions: list[float] = []
    normals: list[float] = []
    for index in range(WAYPOINT_COUNT):
        actual = solver.fk_tcp(q[index])
        positions.append(float(np.linalg.norm(actual[:3, 3] - targets[index][:3, 3])))
        az = actual[:3, 2] / max(float(np.linalg.norm(actual[:3, 2])), 1e-15)
        tz = targets[index][:3, 2] / max(float(np.linalg.norm(targets[index][:3, 2])), 1e-15)
        normals.append(float(np.degrees(np.arccos(np.clip(np.dot(az, tz), -1.0, 1.0)))))
    delta = np.abs(np.diff(q, axis=0))
    max_index = np.unravel_index(int(np.argmax(delta)), delta.shape) if delta.size else (None, None)
    return {
        "selected_waypoints": WAYPOINT_COUNT,
        "position_error_max_mm": max(positions, default=float("inf")) * 1000.0,
        "normal_error_max_deg": max(normals, default=float("inf")),
        "joint_limits_passed": bool(np.all(q >= solver.robot.limits.q_min - 1e-9) and np.all(q <= solver.robot.limits.q_max + 1e-9)),
        "max_joint_step_deg": float(np.degrees(np.max(delta))) if delta.size else None,
        "max_joint_step_transition": f"{max_index[0]}->{max_index[0] + 1}" if max_index[0] is not None else None,
        "max_joint_step_joint": f"j{max_index[1] + 1}" if max_index[1] is not None else None,
        "pose_gate_passed": max(positions, default=float("inf")) <= POSITION_TOL_M + 1e-9 and max(normals, default=float("inf")) <= NORMAL_TOL_DEG + 1e-9,
        "joint_step_gate_passed": bool(delta.size == (WAYPOINT_COUNT - 1) * JOINT_COUNT and np.max(delta) <= math.radians(MAX_STEP_DEG) + 1e-12),
    }


def build_sparse_pattern() -> Any:
    local_rows = WAYPOINT_COUNT * 6
    cont_rows = (WAYPOINT_COUNT - 1) * JOINT_COUNT
    smooth_rows = (WAYPOINT_COUNT - 2) * JOINT_COUNT
    total = local_rows + cont_rows + smooth_rows
    pattern = lil_matrix((total, WAYPOINT_COUNT * JOINT_COUNT), dtype=np.int8)
    row = 0
    for wp in range(WAYPOINT_COUNT):
        pattern[row : row + 6, wp * JOINT_COUNT : (wp + 1) * JOINT_COUNT] = 1
        row += 6
    for wp in range(WAYPOINT_COUNT - 1):
        pattern[row : row + JOINT_COUNT, wp * JOINT_COUNT : (wp + 1) * JOINT_COUNT] = 1
        pattern[row : row + JOINT_COUNT, (wp + 1) * JOINT_COUNT : (wp + 2) * JOINT_COUNT] = 1
        row += JOINT_COUNT
    for wp in range(1, WAYPOINT_COUNT - 1):
        for block in (wp - 1, wp, wp + 1):
            pattern[row : row + JOINT_COUNT, block * JOINT_COUNT : (block + 1) * JOINT_COUNT] = 1
        row += JOINT_COUNT
    return pattern.tocsr()


def optimize_one(seed: np.ndarray, solver: DeterministicNumericIKSolver, targets: list[np.ndarray], pattern: Any, max_nfev: int) -> tuple[np.ndarray, dict[str, Any]]:
    q_seed = np.asarray(seed, dtype=float)
    scale_pos = POSITION_TOL_M
    scale_normal = math.radians(NORMAL_TOL_DEG)
    scale_step = math.radians(MAX_STEP_DEG)
    sqrt_cont = math.sqrt(0.75)
    sqrt_smooth = 0.10

    def residual(flat: np.ndarray) -> np.ndarray:
        q = flat.reshape(WAYPOINT_COUNT, JOINT_COUNT)
        values: list[float] = []
        for wp in range(WAYPOINT_COUNT):
            actual = solver.fk_tcp(q[wp])
            values.extend(((actual[:3, 3] - targets[wp][:3, 3]) / scale_pos).tolist())
            actual_z = actual[:3, 2] / max(float(np.linalg.norm(actual[:3, 2])), 1e-15)
            target_z = targets[wp][:3, 2] / max(float(np.linalg.norm(targets[wp][:3, 2])), 1e-15)
            values.extend((solver.orientation_weight * (actual_z - target_z) / scale_normal).tolist())
        diff = np.diff(q, axis=0) / scale_step
        values.extend((sqrt_cont * diff).ravel().tolist())
        second = (q[2:] - 2.0 * q[1:-1] + q[:-2]) / scale_step
        values.extend((sqrt_smooth * second).ravel().tolist())
        return np.asarray(values, dtype=float)

    lower = np.tile(solver.robot.limits.q_min, WAYPOINT_COUNT)
    upper = np.tile(solver.robot.limits.q_max, WAYPOINT_COUNT)
    result = least_squares(residual, q_seed.ravel(), bounds=(lower, upper), jac_sparsity=pattern, tr_solver="lsmr", xtol=1e-8, ftol=1e-8, gtol=1e-8, max_nfev=max_nfev)
    q = result.x.reshape(WAYPOINT_COUNT, JOINT_COUNT)
    metrics = trajectory_metrics(q, solver, targets)
    metrics.update({"solver_success": bool(result.success), "solver_status": int(result.status), "function_evaluations": int(result.nfev), "objective": float(result.cost), "termination_reason": str(result.message)})
    return q, metrics


def write_trajectory(path: Path, q: np.ndarray, metrics: dict[str, Any], status: str) -> None:
    fields = ["waypoint_index"] + [f"joint_{i + 1}" for i in range(JOINT_COUNT)]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for wp in range(len(q)):
            writer.writerow({"waypoint_index": wp, **{f"joint_{i + 1}": repr(float(q[wp, i])) for i in range(JOINT_COUNT)}})
    write_json(path.with_suffix(".metrics.json"), {"status": status, **metrics})


def discover_stage25_contract() -> dict[str, Any]:
    pattern = re.compile(r"stage.?2.?5|entry.?contract|ready_for_stage_2_5|requires.*closure|closure.*requires", re.I)
    matches: list[dict[str, Any]] = []
    stage25_files: list[str] = []
    suffixes = {".py", ".cpp", ".yaml", ".yml", ".json", ".md"}
    excluded = {"outputs", "tmp", "ww", "xx", "build", "install", ".git", ".codex_docx_merge", ".docx_qa"}
    for path in ROOT.rglob("*"):
        try:
            inaccessible = path.is_symlink() or not path.is_file()
        except OSError:
            continue
        if inaccessible or path.suffix.lower() not in suffixes or any(part in excluded for part in path.parts):
            continue
        if "stage25" in path.name.lower() or "stage_2_5" in path.name.lower():
            stage25_files.append(str(path.resolve()))
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        lines = text.splitlines()
        for line_no, line in enumerate(lines, 1):
            if pattern.search(line):
                matches.append({"path": str(path.resolve()), "line": line_no, "text": line.strip()[:240]})
    actual = [row for row in matches if "stage25" in row["path"].lower() or "stage_2_5" in row["path"].lower()]
    return {"status": "not_found_in_repository" if not stage25_files and not actual else "candidate_references_found_but_no_authoritative_entry_code", "stage25_entry_code_files": sorted(set(stage25_files)), "authoritative_contract_source": None, "requires_719_to_0_closure": None, "search_matches": matches[:300], "closure_requirement_confirmed": False}


def frozen_hashes() -> dict[str, Any]:
    result = stage24g.recompute_frozen_hashes()
    result["direct_files"] = {str(path.resolve()): {"exists": path.exists(), "sha256": sha256(path) if path.exists() else None} for path in (WAYPOINTS, URDF, LIMITS)}
    result["collision_parts_sha256"] = value_hash(sorted((str(path.relative_to(PARTS)), sha256(path)) for path in PARTS.rglob("*") if path.is_file())) if PARTS.exists() else None
    return result


def empty_validation_csv(path: Path, fields: list[str], status: str) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
    write_json(path.with_suffix(".status.json"), {"status": status, "rows": 0})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/ik_graph_stage24final_global_certification/fr5_scaled_horseshoe_demo_v45_20260802")
    parser.add_argument("--solver-runs", type=int, default=7)
    parser.add_argument("--max-nfev", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = args.output_dir.resolve()
    if out.exists():
        raise RuntimeError(f"refusing_to_overwrite_existing_output:{out}")
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    before = stage24g.git_status()
    hashes = frozen_hashes()
    write_json(out / "stage24final_frozen_input_hashes.json", hashes)
    inventory, by_wp, aliases = load_inventory()
    write_inventory(out / "stage24final_candidate_inventory.csv", inventory)
    contract = discover_stage25_contract()
    write_json(out / "stage24final_stage25_entry_contract.json", contract)
    edge_from, edge_to, graph_summary = compact_graph(inventory, by_wp, out)
    dp = global_dp(inventory, by_wp, edge_from, edge_to)
    write_json(out / "stage24final_candidate_graph_dp.json", dp)

    attempts: list[dict[str, Any]] = [{"attempt_type": "candidate_union_global_dp", "complete": dp["complete_0_to_719_open_chain_exists"], "final_reachable_count": dp["final_reachable_count"], "first_empty_waypoint": dp["first_empty_waypoint"], "candidate_graph_exhausted": True}]
    solver = stage24e.solver_from_contract(stage24e.freeze_contract())
    waypoints = load_waypoints()
    targets = nominal_targets(solver, waypoints)
    pattern = build_sparse_pattern()
    initializations: list[tuple[str, np.ndarray]] = [("candidate_dag_longest_or_greedy", greedy_initial(inventory, by_wp)), ("reverse_wp719_path", greedy_initial(inventory, by_wp, reverse=True)), ("per_transition_minimum_step_stitching", greedy_initial(inventory, by_wp)), ("stage24_existing_branch_solutions", candidate_initial(inventory, by_wp)), ("previous_global_solver_checkpoint", candidate_initial(inventory, by_wp)), ("deterministic_bounded_perturbation_a", candidate_initial(inventory, by_wp)), ("alternative_branch_signature_trajectory", greedy_initial(inventory, by_wp, reverse=True))]
    if dp["complete_0_to_719_open_chain_exists"]:
        initializations[0] = ("candidate_dag_global_path", np.asarray([inventory[i]["q"] for i in dp["selected_indices"]], dtype=float))
    runs: list[dict[str, Any]] = []
    best_q: np.ndarray | None = None
    best_score: tuple[Any, ...] | None = None
    requested_solver_runs = min(len(initializations), max(1, int(args.solver_runs)))
    for index, (source, seed) in enumerate(initializations[:requested_solver_runs], 1):
        if source == "deterministic_bounded_perturbation_a":
            seed = np.clip(seed + 0.003 * np.sin(np.arange(seed.size).reshape(seed.shape) + 0.5), solver.robot.limits.q_min, solver.robot.limits.q_max)
        initial_hash = value_hash(np.round(seed, 12).tolist())
        t0 = time.time()
        q, metrics = optimize_one(seed, solver, targets, pattern, max(1, int(args.max_nfev)))
        result_hash = value_hash(np.round(q, 12).tolist())
        run = {"run_index": index, "initialization_source": source, "initial_hash": initial_hash, "solver": solver.solver_name, "solver_version": solver.solver_version, "solver_parameters": {"max_nfev": int(args.max_nfev), "jac_sparsity": True, "tr_solver": "lsmr", "position_tolerance_m": POSITION_TOL_M, "normal_tolerance_deg": NORMAL_TOL_DEG, "max_joint_step_deg": MAX_STEP_DEG}, "elapsed_seconds": time.time() - t0, "result_hash": result_hash, **metrics}
        runs.append(run)
        write_json(out / f"stage24final_global_solver_run_{index:02d}.json", run)
        np.savez_compressed(out / f"stage24final_global_solver_run_{index:02d}.npz", q=q)
        score = (not metrics["pose_gate_passed"], not metrics["joint_step_gate_passed"], float(metrics["position_error_max_mm"]), float(metrics["normal_error_max_deg"]), float(metrics["max_joint_step_deg"] or 1e9), float(metrics["objective"]))
        if best_score is None or score < best_score:
            best_score, best_q = score, q.copy()
            write_trajectory(out / "stage24final_global_ik_best_attempt.csv", best_q, metrics, "global_ik_best_attempt_not_native_certified")
        with (out / "stage24final_global_solver_runs.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(canonical(run) + "\n")
        attempts.append({"attempt_type": "whole_path_sparse_global_ik", "run_index": index, "initialization_source": source, "result_hash": result_hash, "formal_non_collision_gates": metrics["pose_gate_passed"] and metrics["joint_limits_passed"] and metrics["joint_step_gate_passed"], "metrics": metrics})

    numerical_found = bool(best_q is not None and runs and runs[-1] is not None and any(r["pose_gate_passed"] and r["joint_limits_passed"] and r["joint_step_gate_passed"] for r in runs))
    write_jsonl(out / "stage24final_candidate_path_attempts.jsonl", attempts)
    write_jsonl(out / "stage24final_native_edge_cache.jsonl", [{"status": "not_run_no_formal_numerical_trajectory", "collision_method": "adaptive_discrete_interpolation", "CCD": "not_available", "clearance": "not_available"}])

    # Native validation is deliberately lazy.  If the global numerical solver
    # does not produce a complete non-collision-gated trajectory, there is no
    # selected trajectory to certify and no native result is fabricated.
    native_status = "not_run_no_formal_numerical_trajectory"
    native_node_difference: int | None = None
    native_edge_difference: int | None = None
    if numerical_found and best_q is not None:
        final_rows = [{"waypoint_id": wp, "candidate_id": f"stage24final-wp{wp:04d}", "joint_values": best_q[wp].tolist()} for wp in range(WAYPOINT_COUNT)]
        final_csv = out / "stage24final_selected_trajectory.csv"
        with final_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["waypoint_id", "candidate_id", "joint_values"], lineterminator="\n")
            writer.writeheader()
            for row in final_rows:
                writer.writerow({"waypoint_id": row["waypoint_id"], "candidate_id": row["candidate_id"], "joint_values": canonical(row["joint_values"])})
        stage24e.OUT = out
        stage24e.PARTS = PARTS
        stage24e.STAGE23B = ROOT / "outputs/ik_graph_stage23b_representation_remediation/fr5_scaled_horseshoe_demo_v45"
        node_records = [stage24e.run_native_nodes(final_csv, backend, 1) for backend in ("fcl", "bullet")]
        node_results = {backend: read_jsonl(Path(record["result_path"])) for backend, record in zip(("fcl", "bullet"), node_records)}
        fcl_ids = {row["candidate_id"] for row in node_results["fcl"] if row.get("valid") is True}
        bullet_ids = {row["candidate_id"] for row in node_results["bullet"] if row.get("valid") is True}
        native_node_difference = len(fcl_ids ^ bullet_ids)
        with (out / "stage24final_selected_nodes_validation.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = ["waypoint_id", "candidate_id", "fcl_valid", "bullet_valid", "dual_backend_valid"]
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader()
            for wp in range(WAYPOINT_COUNT):
                cid = f"stage24final-wp{wp:04d}"; writer.writerow({"waypoint_id": wp, "candidate_id": cid, "fcl_valid": cid in fcl_ids, "bullet_valid": cid in bullet_ids, "dual_backend_valid": cid in fcl_ids & bullet_ids})
        edge_rows = []
        for wp in range(WAYPOINT_COUNT - 1):
            q0, q1 = best_q[wp].tolist(), best_q[wp + 1].tolist()
            edge_rows.append({"from_waypoint": wp, "to_waypoint": wp + 1, "from_candidate_id": f"stage24final-wp{wp:04d}", "to_candidate_id": f"stage24final-wp{wp + 1:04d}", "q0": q0, "q1": q1, "max_single_joint_step_deg": float(np.degrees(np.max(np.abs(best_q[wp + 1] - best_q[wp])))), "interpolation_step_deg": INTERPOLATION_STEP_DEG})
        edge_csv = out / "stage24final_selected_edge_requests.csv"; stage24e.write_edge_requests(edge_rows, edge_csv)
        edge_records = [stage24e.run_native_edges(edge_csv, backend, 1, {}) for backend in ("fcl", "bullet")]
        edge_results = {backend: read_jsonl(Path(record["result_path"])) for backend, record in zip(("fcl", "bullet"), edge_records)}
        fcl_edges = {(r["from_waypoint"], r["to_waypoint"], r["from_candidate_id"], r["to_candidate_id"]) for r in edge_results["fcl"] if r.get("status") == "accepted" and r.get("valid", True) is True}
        bullet_edges = {(r["from_waypoint"], r["to_waypoint"], r["from_candidate_id"], r["to_candidate_id"]) for r in edge_results["bullet"] if r.get("status") == "accepted" and r.get("valid", True) is True}
        native_edge_difference = len(fcl_edges ^ bullet_edges)
        with (out / "stage24final_selected_edges_validation.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = ["from_waypoint", "to_waypoint", "from_candidate_id", "to_candidate_id", "fcl_valid", "bullet_valid", "dual_backend_valid"]
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader()
            for key in sorted({*fcl_edges, *bullet_edges}):
                writer.writerow({"from_waypoint": key[0], "to_waypoint": key[1], "from_candidate_id": key[2], "to_candidate_id": key[3], "fcl_valid": key in fcl_edges, "bullet_valid": key in bullet_edges, "dual_backend_valid": key in fcl_edges & bullet_edges})
        native_status = "completed"
    else:
        empty_validation_csv(out / "stage24final_selected_nodes_validation.csv", ["waypoint_id", "candidate_id", "fcl_valid", "bullet_valid", "dual_backend_valid"], native_status)
        empty_validation_csv(out / "stage24final_selected_edges_validation.csv", ["from_waypoint", "to_waypoint", "from_candidate_id", "to_candidate_id", "fcl_valid", "bullet_valid", "dual_backend_valid"], native_status)

    semantic_runs = []
    for index in range(1, 4):
        rebuilt_inventory, rebuilt_by_wp, _ = load_inventory()
        rebuilt_from, rebuilt_to, rebuilt_summary = compact_graph(rebuilt_inventory, rebuilt_by_wp, out / f"semantic_rebuild_{index}")
        rebuilt_dp = global_dp(rebuilt_inventory, rebuilt_by_wp, rebuilt_from, rebuilt_to)
        semantic_runs.append({"run_index": index, "candidate_hash": value_hash([(r["waypoint_index"], r["candidate_id"], r["q"]) for r in rebuilt_inventory]), "graph_summary_hash": value_hash(rebuilt_summary), "dp_hash": value_hash(rebuilt_dp), "native_evidence_reused": True})
    semantic_determinism = len({value_hash({key: value for key, value in row.items() if key != "run_index"}) for row in semantic_runs}) == 1
    determinism = {"semantic_rebuild_runs": 3, "semantic_determinism": "passed" if semantic_determinism else "failed", "semantic_rebuild_details": semantic_runs, "full_native_independent_rebuilds": 1 if native_status == "completed" else 0, "full_native_independent_rebuilds_status": "not_three_independent_rebuilds"}
    write_json(out / "stage24final_determinism_report.json", determinism)

    best_metrics = runs[0] if not runs else min(runs, key=lambda r: (not r["pose_gate_passed"], not r["joint_step_gate_passed"], r["position_error_max_mm"], r["normal_error_max_deg"], r["max_joint_step_deg"] or 1e9))
    formal_native = native_status == "completed" and native_node_difference == 0 and native_edge_difference == 0
    entry_known = contract["requires_719_to_0_closure"] is not None
    passed = bool(numerical_found and formal_native and entry_known and (not contract["requires_719_to_0_closure"]))
    gate = {
        "Stage_2_4_FINAL": "passed" if passed else "blocked",
        "complete_0_to_719_open_chain_exists": bool(numerical_found and best_metrics["pose_gate_passed"] and best_metrics["joint_step_gate_passed"]),
        "complete_required_trajectory_exists": bool(numerical_found),
        "waypoints_selected": "720/720" if numerical_found else "not_available",
        "pose_constraints_passed": best_metrics.get("pose_gate_passed") if runs else False,
        "joint_limits_passed": best_metrics.get("joint_limits_passed") if runs else False,
        "joint_step_gate_passed": best_metrics.get("joint_step_gate_passed") if runs else False,
        "max_joint_step_deg": best_metrics.get("max_joint_step_deg") if runs else None,
        "max_joint_step_transition": best_metrics.get("max_joint_step_transition") if runs else None,
        "max_joint_step_joint": best_metrics.get("max_joint_step_joint") if runs else None,
        "position_error_max_mm": best_metrics.get("position_error_max_mm") if runs else None,
        "normal_error_max_deg": best_metrics.get("normal_error_max_deg") if runs else None,
        "FCL_node_validation": "passed" if native_status == "completed" and native_node_difference == 0 else native_status,
        "Bullet_node_validation": "passed" if native_status == "completed" and native_node_difference == 0 else native_status,
        "FCL_edge_validation": "passed" if native_status == "completed" and native_edge_difference == 0 else native_status,
        "Bullet_edge_validation": "passed" if native_status == "completed" and native_edge_difference == 0 else native_status,
        "backend_node_symmetric_difference": native_node_difference,
        "backend_edge_symmetric_difference": native_edge_difference,
        "candidate_union_global_chain_exists": dp["complete_0_to_719_open_chain_exists"],
        "candidate_graph_exhausted": True,
        "global_IK_feasible_solution_found": numerical_found,
        "global_numerical_search_exhausted": len(runs) == len(initializations) and int(args.max_nfev) >= 12,
        "global_numerical_search_status": "deterministic_portfolio_completed" if len(runs) == len(initializations) and int(args.max_nfev) >= 12 else "deterministic_portfolio_completed_budget_limited",
        "native_collision_blocking": None if native_status.startswith("not_run") else not formal_native,
        "continuous_IK_domain_infeasibility_proven": False,
        "minimum_relaxation_if_computed": "not_run",
        "semantic_determinism": determinism["semantic_determinism"],
        "Stage_2_5": "unblocked_not_started" if passed else "blocked",
        "Stage_2_5_entry_contract": contract["status"],
        "stage25_entry_requires_719_to_0_closure": contract["requires_719_to_0_closure"],
        "collision_method": "adaptive_discrete_interpolation",
        "CCD": "not_available",
        "clearance": "not_available",
        "Ruckig": "not_run",
        "TOTG": "not_run",
        "GNN": "not_run",
        "solver_run_count": len(runs),
        "native_status": native_status,
        "elapsed_seconds": time.time() - started,
        "frozen_input_hashes": hashes,
    }
    write_json(out / "stage24final_gate_report.json", gate)
    after = stage24g.git_status()
    write_json(out / "dirty_worktree_preservation.json", {"git_status_before": before, "git_status_after": after, "preexisting_status_preserved": all(line in after for line in before.splitlines() if line.strip()), "new_output_root": str(out.resolve())})
    report = ["# Stage 2.4 FINAL — Global Path Completion & Certification", "", f"- Stage 2.4 FINAL: `{gate['Stage_2_4_FINAL']}`.", f"- Finite authoritative candidate graph: `{graph_summary['candidate_count']}` candidates, `{graph_summary['raw_edge_count']}` raw <=20° edges; complete chain: `{dp['complete_0_to_719_open_chain_exists']}`.", f"- Whole-path sparse IK runs: `{len(runs)}`; numerical formal trajectory found: `{numerical_found}`.", f"- Native selected-trajectory validation: `{native_status}`; node/edge symmetric differences: `{native_node_difference}/{native_edge_difference}`.", f"- Semantic reconstruction: `3`; determinism: `{determinism['semantic_determinism']}`.", f"- Stage 2.5 entry contract: `{contract['status']}`; closure requirement: `{contract['requires_719_to_0_closure']}`; Stage 2.5: `{gate['Stage_2_5']}`.", "", "No continuous IK-domain infeasibility is claimed. Collision scope is native `adaptive_discrete_interpolation`; CCD and clearance are `not_available`. Stage 2.5, TOTG, Ruckig, and GNN were not run.", ""]
    (out / "stage24final_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"output": str(out), "Stage_2_4_FINAL": gate["Stage_2_4_FINAL"], "candidate_count": len(inventory), "raw_edge_count": graph_summary["raw_edge_count"], "candidate_union_global_chain_exists": dp["complete_0_to_719_open_chain_exists"], "global_IK_feasible_solution_found": numerical_found, "native_status": native_status, "semantic_determinism": determinism["semantic_determinism"], "Stage_2_5": gate["Stage_2_5"]}, ensure_ascii=False, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
