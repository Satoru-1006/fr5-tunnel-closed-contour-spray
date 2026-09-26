"""Stage 2.4F global branch-conditioned adaptive closure search.

This runner is intentionally additive.  It reads the frozen Stage 2.4D and
Stage 2.4E bundles, reconstructs their dual-backend graph, performs a
deterministic branch-conditioned continuation search in the frozen tolerance
box, and validates only newly generated nodes/edges with the native MoveIt2
PlanningScene probes.  It never mutates the authoritative robot, scene, ACM,
waypoints, or prior-stage evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_stage24_closed_loop_graph as stage24
import scripts.run_stage24e_tolerance_search as stage24e
from src.deterministic_numeric_ik import DeterministicNumericIKSolver


EVIDENCE = ROOT / "outputs/ik_graph_stage24e_tolerance_search/fr5_scaled_horseshoe_demo_v45_retry_4_20260731"
STAGE24D = ROOT / "outputs/ik_graph_stage24d_global_branch_cycle_audit/fr5_scaled_horseshoe_demo_v45"
SOURCE = ROOT / "outputs/ik_graph_stage23a4_backend_equivalent/fr5_scaled_horseshoe_demo_v44"
STAGE23B = ROOT / "outputs/ik_graph_stage23b_representation_remediation/fr5_scaled_horseshoe_demo_v45"
URDF = ROOT / "outputs/ik_graph_stage193/expanded_runtime_urdf.urdf"
LIMITS = ROOT / "ros2_moveit_bridge/config/joint_limits_with_jerk.yaml"
PARTS = SOURCE / "tunnel_collision_parts"
WAYPOINT_COUNT = 720
MAX_STEP_DEG = 20.0
INTERPOLATION_STEP_DEG = 1.0
BACKENDS = ("fcl", "bullet")
JOBS = tuple(f"j{i}" for i in range(1, 7))
FRAGILE_TRANSITIONS = tuple(
    [f"{i}->{i + 1}" for i in range(199, 227)]
    + ["301->302", "302->303", "402->403", "527->528"]
)
WINDOWS = (
    (10, tuple(range(715, 720)) + tuple(range(0, 5))),
    (20, tuple(range(710, 720)) + tuple(range(0, 10))),
    (40, tuple(range(700, 720)) + tuple(range(0, 20))),
    (80, tuple(range(680, 720)) + tuple(range(0, 40))),
    (160, tuple(range(640, 720)) + tuple(range(0, 80))),
    (320, tuple(range(560, 720)) + tuple(range(0, 160))),
    (720, tuple(range(0, 720))),
)
MAX_BRANCHES_PER_WAYPOINT = 8
MAX_NATIVE_EDGE_REQUESTS_PER_TRANSITION = 8
MAX_NATIVE_CLOSURE_REQUESTS = 200


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def value_hash(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")


def run_capture(command: list[str], *, timeout: int = 120) -> dict[str, Any]:
    try:
        proc = subprocess.run(command, cwd=ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=timeout)
        return {"command": command, "exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
    except Exception as exc:  # environment evidence must survive failed probes
        return {"command": command, "exit_code": None, "stdout": "", "stderr": repr(exc)}


def git_status() -> str:
    result = run_capture(["git", "status", "--porcelain"])
    return result["stdout"] + result["stderr"]


def git_snapshot() -> dict[str, Any]:
    return {
        "head": run_capture(["git", "rev-parse", "HEAD"]),
        "status_porcelain": git_status(),
        "diff_stat": run_capture(["git", "diff", "--stat"]),
        "diff_name_only": run_capture(["git", "diff", "--name-only"]),
    }


def file_entry(path: Path, role: str) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "role": role,
        "exists": path.exists(),
        "size": path.stat().st_size if path.exists() else None,
        "sha256": sha256(path) if path.is_file() else None,
    }


def directory_entries(path: Path, role: str) -> list[dict[str, Any]]:
    if not path.exists():
        return [{"path": str(path.resolve()), "role": role, "exists": False}]
    return [file_entry(item, f"{role}:{item.relative_to(path).as_posix()}") for item in sorted(path.rglob("*")) if item.is_file()]


def environment_manifest() -> dict[str, Any]:
    import scipy

    wsl = run_capture(["wsl.exe", "bash", "-lc", "source /opt/ros/jazzy/setup.bash; printf 'ROS_DISTRO=%s\\n' \"$ROS_DISTRO\"; grep -m1 '<version>' \"$(ros2 pkg prefix moveit_core)/share/moveit_core/package.xml\"; pkg-config --modversion fcl 2>/dev/null || true; pkg-config --modversion bullet 2>/dev/null || true; g++ --version | head -1"], timeout=120)
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "scipy": scipy.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "ros_moveit_fcl_bullet_compiler_probe": wsl,
        "ROS_DISTRO": wsl.get("stdout", "").splitlines()[0].split("=", 1)[1] if wsl.get("stdout", "").startswith("ROS_DISTRO=") else None,
        "native_runtime_required": True,
        "collision_method": "adaptive_discrete_interpolation",
        "CCD": "not_available",
        "clearance": "not_available",
    }


def authoritative_input_manifest(before: dict[str, Any]) -> dict[str, Any]:
    paths: list[tuple[Path, str]] = [
        (ROOT / "external/frcobot_ros2/fairino_description/urdf/fairino5_v6.urdf", "authoritative_urdf"),
        (URDF, "runtime_urdf_used_by_native_stage24_probes"),
        (ROOT / "ros2_moveit_bridge/config/fairino5_v6_spray_tcp.srdf", "authoritative_srdf_and_acm"),
        (ROOT / "ros2_moveit_bridge/config/fairino5_v6_spray_tcp.urdf.xacro", "tcp_model_config"),
        (LIMITS, "authoritative_joint_limits"),
        (SOURCE / "waypoints.csv", "frozen_720_waypoints"),
        (SOURCE / "scaled_demo_parameters.yaml", "frozen_process_and_scene_parameters"),
        (STAGE24D / "stage24d_nodes.jsonl", "Stage_2_4D_formal_candidates"),
        (STAGE24D / "stage24d_edges.jsonl", "Stage_2_4D_formal_edges"),
        (STAGE24D / "next_stage_decision/stage24d_all_720_transition_decision.csv", "Stage_2_4D_transition_evidence"),
        (EVIDENCE / "stage24e_candidates.jsonl", "Stage_2_4E_new_candidates"),
        (EVIDENCE / "stage24e_nodes_fcl.jsonl", "Stage_2_4E_fcl_node_results"),
        (EVIDENCE / "stage24e_nodes_bullet.jsonl", "Stage_2_4E_bullet_node_results"),
        (EVIDENCE / "stage24e_edges_fcl.jsonl", "Stage_2_4E_fcl_edge_results"),
        (EVIDENCE / "stage24e_edges_bullet.jsonl", "Stage_2_4E_bullet_edge_results"),
        (EVIDENCE / "stage24e_frozen_tolerance_contract.yaml", "Stage_2_4E_frozen_tolerance_contract_yaml"),
        (EVIDENCE / "stage24e_frozen_tolerance_contract.yaml.json", "Stage_2_4E_frozen_tolerance_contract_json"),
        (EVIDENCE / "stage24e_fragile_transition_audit.json", "authoritative_32_fragile_transition_list"),
        (EVIDENCE / "stage24e_gate_report.json", "Stage_2_4E_gate_report"),
        (EVIDENCE / "stage24e_search_manifest.json", "Stage_2_4E_search_manifest"),
        (ROOT / "tools/stage23b_formal_launch.py", "native_planning_scene_launch"),
        (ROOT / "tools/stage24_formal_launch.py", "native_edge_check_launch"),
        (ROOT / "cpp/stage24/stage24_formal_probe.cpp", "native_formal_collision_probe"),
    ]
    entries = [file_entry(path, role) for path, role in paths]
    entries.extend(directory_entries(PARTS, "authoritative_collision_geometry"))
    entries.extend(directory_entries(EVIDENCE, "all_files_read_from_frozen_Stage_2_4E_authority"))
    return {
        "schema_version": "stage24f-input-manifest-v1",
        "stage": "2.4F",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": before,
        "frozen_stage24e_authority": read_json(EVIDENCE / "stage24e_gate_report.json"),
        "frozen_stage24e_facts": {
            "reported_status": "blocked_no_dual_backend_valid_closure_edge",
            "precise_failure_layer": "no_joint_gate_admissible_closure_pair",
            "candidate_count": 1750,
            "dual_backend_valid_nodes": 1656,
            "seam_waypoints": [715, 716, 717, 718, 719, 0, 1, 2, 3, 4],
            "parameter_groups": 24,
            "fcl_bullet_node_difference": 0,
            "fcl_bullet_edge_difference": 0,
            "fragile_transitions_preserved": 32,
            "deterministic_rebuild_hash_difference": 0,
            "closure_pair_count_native_checked": 0,
        },
        "planning_scene_standalone_file": None,
        "planning_scene_authority": "native MoveIt2 launch and collision-parts inputs listed above",
        "acm_standalone_file": None,
        "acm_authority": "SRDF disabled-collision declarations listed above",
        "files": entries,
        "authoritative_inputs_modified": False,
    }


def frozen_contract() -> dict[str, Any]:
    contract = read_json(EVIDENCE / "stage24e_frozen_tolerance_contract.yaml.json")
    # These are explicit Stage 2.4F constraints, not re-derived values.
    contract["stage24f_policy"] = {
        "bounded_revolute_raw_delta_only": True,
        "continuous_flags": {job: False for job in JOBS},
        "max_joint_step_deg": MAX_STEP_DEG,
        "interpolation_step_deg": INTERPOLATION_STEP_DEG,
        "adaptive_windows": [{"window_size": size, "waypoints": list(points)} for size, points in WINDOWS],
        "seed_order": [
            "selected_open_chain_adjacent_waypoint",
            "same_ik_branch_signature_previous_waypoint",
            "same_ik_branch_signature_next_waypoint",
            "Stage_2_4D_frozen_candidate",
            "Stage_2_4E_frozen_candidate",
            "sorted_frozen_seed_bank_by_branch_and_candidate_id",
        ],
        "random_seed": None,
        "random_api_calls": 0,
        "continuous_refinement": {
            "method": "deterministic_bounded_coordinate_refinement",
            "fractions": [0.0, 0.25, 0.5],
            "formal_validation_required": True,
            "continuous_space_covered": False,
        },
    }
    return contract


def parse_q(value: Any) -> list[float]:
    if isinstance(value, str):
        return [float(x) for x in json.loads(value)]
    return [float(x) for x in value]


def branch_signature(row: dict[str, Any]) -> str:
    return str(row.get("ik_branch_signature", ""))


def row_wp(row: dict[str, Any]) -> int:
    return int(row.get("waypoint_index", row.get("waypoint_id")))


def q_of(row: dict[str, Any]) -> list[float]:
    return parse_q(row.get("joint_vector_rad", row.get("joint_values_rad", row.get("joint_values"))))


def normalize_node(row: dict[str, Any], source_stage: str, fcl_valid: bool, bullet_valid: bool) -> dict[str, Any]:
    q = q_of(row)
    sig = branch_signature(row)
    return {
        "waypoint_index": row_wp(row),
        "candidate_id": str(row["candidate_id"]),
        "joint_vector_rad": q,
        "joint_vector_deg": [math.degrees(v) for v in q],
        "shoulder_branch": row.get("shoulder_branch"),
        "elbow_branch": row.get("elbow_branch"),
        "wrist_branch": row.get("wrist_branch"),
        "ik_branch_signature": sig,
        "source_stage": source_stage,
        "fcl_node_valid": bool(fcl_valid),
        "bullet_node_valid": bool(bullet_valid),
        "dual_backend_valid": bool(fcl_valid and bullet_valid),
        "joint_limit_valid": bool(row.get("joint_limit_pass", row.get("joint_limit_valid", True))),
        "pose_error": row.get("FK_position_error_m", row.get("fk_position_error")),
        "process_tolerance_error": None if row.get("process_tolerance_pass", True) else "process_tolerance_failed",
        "process_tolerance_pass": bool(row.get("process_tolerance_pass", row.get("process_tolerance_valid", True))),
        "position_offset_local_m": row.get("position_offset_local_m"),
        "standoff_offset_m": row.get("standoff_offset_m"),
        "normal_rotation_vector_deg": row.get("normal_rotation_vector_deg"),
        "roll_offset_deg": row.get("roll_offset_deg"),
        "candidate_parameter_hash": row.get("candidate_parameter_hash"),
    }


def load_frozen_graph() -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    d_nodes = read_jsonl(STAGE24D / "stage24d_nodes.jsonl")
    d_edges = read_jsonl(STAGE24D / "stage24d_edges.jsonl")
    e_candidates = read_jsonl(EVIDENCE / "stage24e_candidates.jsonl")
    e_fcl = {str(x["candidate_id"]): bool(x.get("valid") is True) for x in read_jsonl(EVIDENCE / "stage24e_nodes_fcl.jsonl")}
    e_bullet = {str(x["candidate_id"]): bool(x.get("valid") is True) for x in read_jsonl(EVIDENCE / "stage24e_nodes_bullet.jsonl")}
    nodes: dict[str, dict[str, Any]] = {}
    for row in d_nodes:
        item = normalize_node(row, "Stage_2_4D", bool(row.get("fcl_node_valid")), bool(row.get("bullet_node_valid")))
        nodes[item["candidate_id"]] = item
    e_by_id = {str(x["candidate_id"]): x for x in e_candidates}
    for cid, row in sorted(e_by_id.items()):
        item = normalize_node(row, "Stage_2_4E", e_fcl.get(cid, False), e_bullet.get(cid, False))
        nodes[cid] = item
    edges: list[dict[str, Any]] = []
    for row in d_edges:
        if not (row.get("accepted") is True and row.get("fcl_valid") is True and row.get("bullet_valid") is True):
            continue
        edges.append(normalize_edge(row, "Stage_2_4D"))
    e_fcl_edges = read_jsonl(EVIDENCE / "stage24e_edges_fcl.jsonl")
    e_bullet_keys = {
        (int(x["from_waypoint"]), int(x["to_waypoint"]), str(x.get("from_candidate_id", x.get("source_node"))), str(x.get("to_candidate_id", x.get("target_node"))))
        for x in read_jsonl(EVIDENCE / "stage24e_edges_bullet.jsonl")
        if x.get("status") == "accepted" and x.get("valid") is True
    }
    for row in e_fcl_edges:
        key = (int(row["from_waypoint"]), int(row["to_waypoint"]), str(row.get("from_candidate_id", row.get("source_node"))), str(row.get("to_candidate_id", row.get("target_node"))))
        if row.get("status") == "accepted" and row.get("valid") is True and key in e_bullet_keys:
            edges.append(normalize_edge(row, "Stage_2_4E"))
    return nodes, dedup_edges(edges)


def normalize_edge(row: dict[str, Any], source_stage: str) -> dict[str, Any]:
    source = str(row.get("source_node", row.get("from_candidate_id")))
    target = str(row.get("target_node", row.get("to_candidate_id")))
    q0 = parse_q(row.get("q0", row.get("joint_vector_from", [])))
    q1 = parse_q(row.get("q1", row.get("joint_vector_to", [])))
    signed = [b - a for a, b in zip(q0, q1)]
    absolute = [abs(x) for x in signed]
    max_step = math.degrees(max(absolute, default=0.0))
    blocking = f"j{int(np.argmax(absolute)) + 1}" if absolute else None
    return {
        "from_waypoint": int(row["from_waypoint"]),
        "to_waypoint": int(row["to_waypoint"]),
        "from_candidate_id": source,
        "to_candidate_id": target,
        "signed_joint_delta_deg": [math.degrees(x) for x in signed],
        "absolute_joint_delta_deg": [math.degrees(x) for x in absolute],
        "max_joint_delta_deg": float(row.get("maximum_joint_step_deg", row.get("max_model_aware_joint_step_deg", max_step))),
        "blocking_joint": row.get("blocking_joint", blocking),
        "joint_gate_valid": bool(max_step <= MAX_STEP_DEG + 1e-12),
        "fcl_edge_valid": True,
        "bullet_edge_valid": True,
        "samples_checked": row.get("samples_checked"),
        "first_invalid_sample": row.get("first_collision_sample"),
        "collision_pairs": row.get("collision_pairs", []),
        "collision_method": "adaptive_discrete_interpolation",
        "interpolation_step_deg": float(row.get("interpolation_step_deg", row.get("sampling_resolution", INTERPOLATION_STEP_DEG))),
        "source_stage": source_stage,
        "source_evidence": row.get("source_evidence", [source_stage]),
        "edge_id": f"{int(row['from_waypoint']):04d}->{int(row['to_waypoint']):04d}:{source}->{target}",
    }


def dedup_edges(edges: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result: dict[tuple[Any, ...], dict[str, Any]] = {}
    for edge in edges:
        key = (edge["from_waypoint"], edge["to_waypoint"], edge["from_candidate_id"], edge["to_candidate_id"])
        result[key] = edge
    return [result[key] for key in sorted(result)]


def joint_semantics() -> list[dict[str, Any]]:
    return stage24.parse_joint_semantics()


def raw_delta(q0: list[float], q1: list[float], semantics: list[dict[str, Any]]) -> tuple[list[float], list[float], bool, str | None]:
    # Every authoritative joint is bounded revolute; no periodic normalization.
    signed = [b - a for a, b in zip(q0, q1)]
    absolute = [abs(x) for x in signed]
    limits_valid = True
    for q, rule in zip(q0 + q1, semantics + semantics):
        if rule.get("lower") is not None and not (float(rule["lower"]) - 1e-12 <= q <= float(rule["upper"]) + 1e-12):
            limits_valid = False
    blocking = f"j{int(np.argmax(absolute)) + 1}" if absolute else None
    return signed, absolute, limits_valid, blocking


def bounded_parameter_points(contract: dict[str, Any], wp: int, window_size: int) -> list[dict[str, float]]:
    t = contract["tolerances"]
    bounds = {
        "tangent": t["tcp_position_offset_local_m"]["tangent"][1],
        "lateral": t["tcp_position_offset_local_m"]["lateral"][1],
        "normal": t["tcp_position_offset_local_m"]["normal"][1],
        "standoff": t["standoff_offset_m"][1],
        "rx": t["normal_rotation_vector_deg"]["rx"][1],
        "ry": t["normal_rotation_vector_deg"]["ry"][1],
        "roll": t["tool_roll_offset_deg"][1],
    }
    names = tuple(bounds)
    sign = 1.0 if (wp + window_size) % 2 == 0 else -1.0
    pattern = (1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
    reverse = tuple(-x for x in pattern)
    points = []
    for fraction, signs in ((0.0, pattern), (0.25, pattern), (0.5, reverse)):
        points.append({name: float(sign * fraction * signs[i] * bounds[name]) for i, name in enumerate(names)})
    return points


def pose_target(waypoints: dict[int, dict[str, str]], wp: int, values: dict[str, float]) -> np.ndarray:
    nominal_position, nominal_rotation, tangent, lateral = stage24e.pose_frame(waypoints, wp)
    nominal_normal = nominal_rotation[:, 2]
    position = nominal_position + tangent * values["tangent"] + lateral * values["lateral"] + nominal_normal * (values["normal"] + values["standoff"])
    tilt = Rotation.from_rotvec(tangent * math.radians(values["rx"]) + lateral * math.radians(values["ry"])).as_matrix()
    target = np.eye(4, dtype=float)
    target[:3, :3] = tilt @ nominal_rotation @ Rotation.from_euler("z", values["roll"], degrees=True).as_matrix()
    target[:3, 3] = position
    return target


def seed_rows(pool: dict[str, dict[str, Any]], wp: int, signature: str, selected_path: list[str] | None) -> list[dict[str, Any]]:
    ordered: list[dict[str, Any]] = []
    # 1. Adjacent solution from a selected open chain, if one exists.
    if selected_path:
        for adjacent in (wp - 1, wp + 1):
            if 0 <= adjacent < WAYPOINT_COUNT and selected_path[adjacent] in pool:
                ordered.append(pool[selected_path[adjacent]])
    by_wp_sig: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in pool.values():
        by_wp_sig[(row_wp(row), branch_signature(row))].append(row)
    # 2/3. Same signature at previous/next waypoint.
    for adjacent in (wp - 1, wp + 1):
        ordered.extend(sorted(by_wp_sig[(adjacent, signature)], key=lambda x: str(x["candidate_id"])))
    # 4/5. Stage 2.4D then Stage 2.4E candidates at the current waypoint.
    ordered.extend(sorted(by_wp_sig[(wp, signature)], key=lambda x: (str(x.get("source_stage", "")), str(x["candidate_id"]))))
    # 6. Frozen bank in branch/candidate order.
    ordered.extend(sorted((row for row in pool.values() if branch_signature(row) == signature), key=lambda x: (branch_signature(x), str(x["candidate_id"]))))
    result: list[dict[str, Any]] = []
    seen: set[tuple[float, ...]] = set()
    for row in ordered:
        q = tuple(round(float(x), 12) for x in q_of(row))
        if q in seen:
            continue
        seen.add(q)
        result.append(row)
    return result


def generate_continuation(
    solver: DeterministicNumericIKSolver,
    waypoints: dict[int, dict[str, str]],
    pool: dict[str, dict[str, Any]],
    window_size: int,
    window_waypoints: tuple[int, ...],
    selected_path: list[str] | None,
    contract: dict[str, Any],
    existing_q: set[tuple[int, tuple[float, ...]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    signatures_by_wp: dict[int, set[str]] = defaultdict(set)
    for row in pool.values():
        signatures_by_wp[row_wp(row)].add(branch_signature(row))
    attempts: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    optimizer_runs: list[dict[str, Any]] = []
    for wp in window_waypoints:
        signatures = sorted(signatures_by_wp[wp])
        if not signatures:
            signatures = sorted({branch_signature(row) for row in pool.values()})
        signatures = signatures[:MAX_BRANCHES_PER_WAYPOINT]
        points = bounded_parameter_points(contract, wp, window_size)
        for signature in signatures:
            seeds = seed_rows(pool, wp, signature, selected_path)
            if not seeds:
                attempts.append({"waypoint_index": wp, "status": "no_seed", "ik_branch_signature": signature, "window_size": window_size})
                continue
            seed = seeds[0]
            seed_q = np.asarray(q_of(seed), dtype=float)
            run = {
                "run_id": f"window{window_size}-wp{wp:04d}-{value_hash({'signature': signature, 'seed': seed['candidate_id']})[:12]}",
                "waypoint_index": wp,
                "window_size": window_size,
                "ik_branch_signature": signature,
                "seed_order_rank": 0,
                "seed_candidate_id": seed["candidate_id"],
                "objective": "bounded continuation candidate generation; formal IK validity is the acceptance criterion",
                "bounds": contract["tolerances"],
                "initial_point": points[0],
                "termination_tolerances": {"position_error_m": 0.006, "tool_z_error_deg": 10.0},
                "max_evaluations": 3,
                "Jacobian_policy": "deterministic_numeric_IK_solver_internal_jacobian",
                "loss_function": "bounded_coordinate_refinement",
                "iteration_history": points,
                "termination_reason": "fixed_deterministic_refinement_points",
            }
            optimizer_runs.append(run)
            for point_index, values in enumerate(points):
                try:
                    result = solver.solve(pose_target(waypoints, wp, values), seed_q)
                except Exception as exc:
                    attempts.append({"waypoint_index": wp, "window_size": window_size, "ik_branch_signature": signature, "seed_id": seed["candidate_id"], "parameter_index": point_index, "status": "solver_exception", "error": repr(exc)})
                    continue
                if not result.success:
                    attempts.append({"waypoint_index": wp, "window_size": window_size, "ik_branch_signature": signature, "seed_id": seed["candidate_id"], "parameter_index": point_index, "status": "ik_failed", "message": str(getattr(result, "message", ""))})
                    continue
                q = np.asarray(result.q_rad, dtype=float)
                actual = solver.fk_tcp(q)
                target = pose_target(waypoints, wp, values)
                pos_error = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
                actual_z = actual[:3, 2] / max(float(np.linalg.norm(actual[:3, 2])), 1e-15)
                target_z = target[:3, 2] / max(float(np.linalg.norm(target[:3, 2])), 1e-15)
                z_error = float(math.degrees(math.acos(float(np.clip(np.dot(actual_z, target_z), -1.0, 1.0)))))
                limits_ok = bool(np.all(q >= solver.robot.limits.q_min - 1e-9) and np.all(q <= solver.robot.limits.q_max + 1e-9))
                actual_branch = {
                    "shoulder_branch": "positive" if q[0] >= 0 else "negative",
                    "elbow_branch": "positive" if q[2] >= 0 else "negative",
                    "wrist_branch": "positive" if q[4] >= 0 else "negative",
                }
                actual_branch["ik_branch_signature"] = "shoulder={shoulder_branch}|elbow={elbow_branch}|wrist={wrist_branch}".format(**actual_branch)
                branch_ok = actual_branch["ik_branch_signature"] == signature
                q_key = (wp, tuple(round(float(v), 10) for v in q))
                formal_ok = limits_ok and pos_error <= 0.006 + 1e-9 and z_error <= 10.0 + 1e-9 and branch_ok
                attempt = {
                    "waypoint_index": wp,
                    "window_size": window_size,
                    "ik_branch_signature": signature,
                    "actual_ik_branch_signature": actual_branch["ik_branch_signature"],
                    "seed_id": seed["candidate_id"],
                    "parameter_index": point_index,
                    "parameters": values,
                    "joint_values_rad": q.tolist(),
                    "position_error_m": pos_error,
                    "tool_z_error_deg": z_error,
                    "joint_limit_valid": limits_ok,
                    "branch_signature_valid": branch_ok,
                    "status": "formal_candidate" if formal_ok else "formal_rejected",
                }
                attempts.append(attempt)
                if not formal_ok or q_key in existing_q:
                    continue
                existing_q.add(q_key)
                stable = value_hash({"wp": wp, "seed": seed["candidate_id"], "parameters": values, "q": [round(float(v), 12) for v in q], "window": window_size})[:16]
                candidates.append({
                    "waypoint_index": wp,
                    "waypoint_id": wp,
                    "candidate_id": f"s24f-{wp:04d}-{stable}",
                    "joint_values_rad": q.tolist(),
                    "joint_values": q.tolist(),
                    "joint_values_deg": [math.degrees(float(v)) for v in q],
                    "shoulder_branch": actual_branch["shoulder_branch"],
                    "elbow_branch": actual_branch["elbow_branch"],
                    "wrist_branch": actual_branch["wrist_branch"],
                    "ik_branch_signature": actual_branch["ik_branch_signature"],
                    "seed_id": seed["candidate_id"],
                    "seed_branch_signature": signature,
                    "source_stage": "Stage_2_4F",
                    "position_offset_local_m": [values["tangent"], values["lateral"], values["normal"]],
                    "standoff_offset_m": values["standoff"],
                    "normal_rotation_vector_deg": [values["rx"], values["ry"]],
                    "roll_offset_deg": values["roll"],
                    "solver_success": True,
                    "FK_position_error_m": pos_error,
                    "FK_orientation_error_deg": z_error,
                    "joint_limit_pass": limits_ok,
                    "process_tolerance_pass": True,
                    "formal_node_gate_pending": True,
                    "fcl_node_validity": None,
                    "bullet_node_validity": None,
                    "search_window_size": window_size,
                    "candidate_parameter_hash": value_hash(values),
                })
    candidates.sort(key=lambda x: (int(x["waypoint_index"]), str(x["candidate_id"])))
    return candidates, attempts, optimizer_runs


def write_candidate_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["waypoint_id", "candidate_id", "joint_values"], lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({"waypoint_id": row["waypoint_id"], "candidate_id": row["candidate_id"], "joint_values": canonical(row["joint_values"])})


def wsl_path(path: Path) -> str:
    drive, rest = str(path.resolve()).replace("\\", "/").split(":/", 1)
    return f"/mnt/{drive.lower()}/{rest}"


def semantic_hash(value: Any) -> str:
    if isinstance(value, dict):
        return value_hash({k: semantic_hash(v) if isinstance(v, (dict, list)) else v for k, v in value.items() if k not in {"path", "output", "run_duration_s", "captured_at_utc", "started_at_utc", "finished_at_utc"}})
    if isinstance(value, list):
        return value_hash([semantic_hash(v) if isinstance(v, (dict, list)) else v for v in value])
    return value_hash(value)


def graph_edges_by_source(edges: Iterable[dict[str, Any]]) -> dict[tuple[int, str], list[dict[str, Any]]]:
    result: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        result[(int(edge["from_waypoint"]), str(edge["from_candidate_id"]))].append(edge)
    for values in result.values():
        values.sort(key=lambda e: (float(e["max_joint_delta_deg"]), str(e["to_candidate_id"])))
    return result


def path_cost(path_ids: tuple[str, ...], edges: tuple[dict[str, Any], ...], nodes: dict[str, dict[str, Any]]) -> tuple[Any, ...]:
    max_step = max((float(e["max_joint_delta_deg"]) for e in edges), default=0.0)
    total = sum(float(e["max_joint_delta_deg"]) for e in edges)
    switches = sum(branch_signature(nodes[path_ids[i]]) != branch_signature(nodes[path_ids[i - 1]]) for i in range(1, len(path_ids)))
    perturb = sum(sum(abs(float(x)) for x in (nodes[cid].get("position_offset_local_m") or [])) for cid in path_ids if nodes[cid].get("source_stage") == "Stage_2_4F")
    return (round(max_step, 12), round(total, 12), int(switches), round(perturb, 12), path_ids)


def search_open_chain(nodes: dict[str, dict[str, Any]], edges: list[dict[str, Any]]) -> dict[str, Any]:
    layers: dict[int, list[str]] = defaultdict(list)
    for cid, row in nodes.items():
        if row["dual_backend_valid"] and row["joint_limit_valid"] and row["process_tolerance_pass"]:
            layers[int(row["waypoint_index"])].append(cid)
    for values in layers.values():
        values.sort()
    by_source = graph_edges_by_source(edges)
    labels: dict[str, tuple[tuple[str, ...], tuple[dict[str, Any], ...], tuple[Any, ...]]] = {
        cid: ((cid,), tuple(), (0.0, 0.0, 0, 0.0, (cid,))) for cid in layers.get(0, [])
    }
    counts = {0: len(labels)}
    first_break = None
    for wp in range(0, 719):
        next_labels: dict[str, tuple[tuple[str, ...], tuple[dict[str, Any], ...], tuple[Any, ...]]] = {}
        for cid, (path_ids, path_edges, _) in sorted(labels.items()):
            for edge in by_source.get((wp, cid), []):
                target = str(edge["to_candidate_id"])
                if target not in layers.get(wp + 1, []):
                    continue
                candidate_path = path_ids + (target,)
                candidate_edges = path_edges + (edge,)
                cost = path_cost(candidate_path, candidate_edges, nodes)
                old = next_labels.get(target)
                if old is None or cost < old[2]:
                    next_labels[target] = (candidate_path, candidate_edges, cost)
        if not next_labels and first_break is None:
            first_break = wp + 1
        labels = next_labels
        counts[wp + 1] = len(labels)
        if not labels:
            break
    if labels:
        best = min(labels.values(), key=lambda x: x[2])
        return {
            "complete_0_to_719_open_chain_exists": len(best[0]) == WAYPOINT_COUNT,
            "selected_node_ids": list(best[0]),
            "selected_edges": list(best[1]),
            "objective_cost": list(best[2][:-1]) + [list(best[2][-1])],
            "reachable_count_by_waypoint": counts,
            "earliest_break_waypoint": first_break,
            "frontier_candidate_ids": sorted(labels),
        }
    frontier = sorted(labels)
    return {
        "complete_0_to_719_open_chain_exists": False,
        "selected_node_ids": [],
        "selected_edges": [],
        "objective_cost": None,
        "reachable_count_by_waypoint": counts,
        "earliest_break_waypoint": first_break,
        "frontier_candidate_ids": frontier,
        "pre_break_reachable_candidate_ids": sorted([cid for cid, count in []]),
    }


def search_closed_cycle(nodes: dict[str, dict[str, Any]], edges: list[dict[str, Any]], open_chain: dict[str, Any]) -> dict[str, Any]:
    if not open_chain.get("complete_0_to_719_open_chain_exists"):
        return {"complete_720_node_720_edge_cycle_found": False, "selected_node_ids": [], "selected_edges": [], "reason": "no_complete_open_chain"}
    edge_map = {(int(e["from_waypoint"]), int(e["to_waypoint"]), str(e["from_candidate_id"]), str(e["to_candidate_id"])): e for e in edges}
    selected = open_chain["selected_node_ids"]
    closure = edge_map.get((719, 0, selected[-1], selected[0]))
    if closure is None:
        return {"complete_720_node_720_edge_cycle_found": False, "selected_node_ids": selected, "selected_edges": open_chain["selected_edges"], "reason": "selected_open_chain_has_no_closure_edge"}
    return {"complete_720_node_720_edge_cycle_found": True, "selected_node_ids": selected, "selected_edges": open_chain["selected_edges"] + [closure], "reason": None}


def weak_components(nodes: dict[str, dict[str, Any]], edges: list[dict[str, Any]]) -> tuple[dict[str, int], list[dict[str, Any]]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    valid_ids = {cid for cid, row in nodes.items() if row["dual_backend_valid"]}
    for edge in edges:
        a, b = str(edge["from_candidate_id"]), str(edge["to_candidate_id"])
        if a in valid_ids and b in valid_ids:
            adjacency[a].add(b)
            adjacency[b].add(a)
    membership: dict[str, int] = {}
    components: list[dict[str, Any]] = []
    component_id = 0
    for start in sorted(valid_ids):
        if start in membership:
            continue
        queue = deque([start])
        membership[start] = component_id
        members: list[str] = []
        while queue:
            current = queue.popleft()
            members.append(current)
            for nxt in sorted(adjacency[current]):
                if nxt not in membership:
                    membership[nxt] = component_id
                    queue.append(nxt)
        waypoints = sorted({int(nodes[cid]["waypoint_index"]) for cid in members})
        components.append({"component_id": component_id, "node_count": len(members), "waypoint_count": len(waypoints), "min_waypoint": waypoints[0] if waypoints else None, "max_waypoint": waypoints[-1] if waypoints else None, "candidate_ids": sorted(members)})
        component_id += 1
    return membership, components


def build_layered_graph(nodes: dict[str, dict[str, Any]], edges: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    graph_nodes = [nodes[cid] for cid in sorted(nodes, key=lambda x: (int(nodes[x]["waypoint_index"]), x))]
    graph_edges = sorted(edges, key=lambda e: (int(e["from_waypoint"]), int(e["to_waypoint"]), str(e["from_candidate_id"]), str(e["to_candidate_id"])))
    return graph_nodes, graph_edges


def prefilter_new_edges(nodes: dict[str, dict[str, Any]], new_ids: set[str], semantics: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_wp: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in nodes.values():
        if row["dual_backend_valid"]:
            by_wp[int(row["waypoint_index"])].append(row)
    for rows in by_wp.values():
        rows.sort(key=lambda x: str(x["candidate_id"]))
    pending: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    for from_wp in range(WAYPOINT_COUNT):
        to_wp = (from_wp + 1) % WAYPOINT_COUNT
        all_rows: list[dict[str, Any]] = []
        for source in by_wp.get(from_wp, []):
            for target in by_wp.get(to_wp, []):
                if str(source["candidate_id"]) not in new_ids and str(target["candidate_id"]) not in new_ids:
                    continue
                signed, absolute, limits_ok, blocking = raw_delta(source["joint_vector_rad"], target["joint_vector_rad"], semantics)
                item = {
                    "from_waypoint": from_wp,
                    "to_waypoint": to_wp,
                    "from_candidate_id": source["candidate_id"],
                    "to_candidate_id": target["candidate_id"],
                    "q0": source["joint_vector_rad"],
                    "q1": target["joint_vector_rad"],
                    "signed_delta_rad": signed,
                    "absolute_delta_rad": absolute,
                    "max_single_joint_step_deg": math.degrees(max(absolute, default=0.0)),
                    "max_model_aware_joint_step_deg": math.degrees(max(absolute, default=0.0)),
                    "max_joint_delta_deg": math.degrees(max(absolute, default=0.0)),
                    "blocking_joint": blocking,
                    "joint_limit_valid": limits_ok,
                    "joint_gate_valid": limits_ok and max(absolute, default=float("inf")) <= math.radians(MAX_STEP_DEG) + 1e-12,
                    "interpolation_step_deg": INTERPOLATION_STEP_DEG,
                    "collision_method": "adaptive_discrete_interpolation",
                    "new_node_involved": True,
                    "closure_transition": from_wp == 719,
                }
                if not limits_ok or not item["joint_gate_valid"]:
                    item.update({"status": "rejected", "valid": False, "reject_reason": "joint_limit_violation" if not limits_ok else "joint_step_exceeds_20_deg", "reject_reasons": ["joint_limit_violation" if not limits_ok else "joint_step_exceeds_20_deg"]})
                    rejected.append(item)
                else:
                    item.update({"status": "pending_collision", "valid": False, "reject_reason": None, "reject_reasons": []})
                    all_rows.append(item)
        if from_wp == 719:
            all_rows.sort(key=lambda x: (float(x["max_joint_delta_deg"]), str(x["from_candidate_id"]), str(x["to_candidate_id"])))
            selected.extend(all_rows[:MAX_NATIVE_CLOSURE_REQUESTS])
        else:
            all_rows.sort(key=lambda x: (branch_signature(nodes[x["from_candidate_id"]]) != branch_signature(nodes[x["to_candidate_id"]]), float(x["max_joint_delta_deg"]), str(x["from_candidate_id"]), str(x["to_candidate_id"])))
            selected.extend(all_rows[:MAX_NATIVE_EDGE_REQUESTS_PER_TRANSITION])
        pending.extend(all_rows)
    return pending, rejected, selected


def write_edge_requests(rows: list[dict[str, Any]], path: Path) -> None:
    fields = ["from_waypoint", "to_waypoint", "from_candidate_id", "to_candidate_id", "max_single_joint_step_deg", "interpolation_step_deg"] + [f"q0_{i}" for i in range(6)] + [f"q1_{i}" for i in range(6)]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for edge in rows:
            item = {key: edge[key] for key in fields if key in edge}
            for i, value in enumerate(edge["q0"]):
                item[f"q0_{i}"] = repr(value)
            for i, value in enumerate(edge["q1"]):
                item[f"q1_{i}"] = repr(value)
            writer.writerow(item)


def node_validation(run_dir: Path, candidate_rows: list[dict[str, Any]]) -> tuple[set[str], dict[str, Any], list[dict[str, Any]]]:
    if not candidate_rows:
        return set(), {"candidate_count": 0, "dual_backend_valid": 0, "fcl_bullet_node_difference": 0, "exit_codes": []}, []
    candidate_csv = run_dir / "stage24f_node_validation_candidates.csv"
    write_candidate_csv(candidate_rows, candidate_csv)
    stage24e.OUT = run_dir
    stage24e.PARTS = PARTS
    stage24e.STAGE23B = STAGE23B
    records: list[dict[str, Any]] = []
    for backend in BACKENDS:
        records.append(stage24e.run_native_nodes(candidate_csv, backend, 1))
    fcl_rows = read_jsonl(Path(records[0]["result_path"])) if Path(records[0]["result_path"]).exists() else []
    bullet_rows = read_jsonl(Path(records[1]["result_path"])) if Path(records[1]["result_path"]).exists() else []
    fcl_ids = {str(row["candidate_id"]) for row in fcl_rows if row.get("valid") is True}
    bullet_ids = {str(row["candidate_id"]) for row in bullet_rows if row.get("valid") is True}
    audit = {"candidate_count": len(candidate_rows), "fcl_valid": len(fcl_ids), "bullet_valid": len(bullet_ids), "dual_backend_valid": len(fcl_ids & bullet_ids), "fcl_bullet_node_difference": len(fcl_ids ^ bullet_ids), "exit_codes": [x["exit_code"] for x in records], "fcl_result_sha256": records[0].get("result_sha256"), "bullet_result_sha256": records[1].get("result_sha256")}
    write_json(run_dir / "stage24f_node_validation_runs.json", records)
    write_json(run_dir / "stage24f_node_gate_audit.json", audit)
    return fcl_ids & bullet_ids, audit, records


def native_edge_validation(run_dir: Path, selected_rows: list[dict[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if not selected_rows:
        return {"fcl": [], "bullet": []}, {"native_checked": 0, "records": [], "build": None}
    edge_csv = run_dir / "stage24f_edge_requests.csv"
    write_edge_requests(selected_rows, edge_csv)
    stage24e.OUT = run_dir
    stage24e.PARTS = PARTS
    stage24e.STAGE23B = STAGE23B
    stage24.OUT = run_dir
    build = stage24e.stage24.build_native()
    write_json(run_dir / "stage24f_edge_build_report.json", build)
    if build.get("exit_code") != 0:
        raise RuntimeError(f"native_stage24f_edge_build_failed:{build}")
    manifests: dict[str, list[dict[str, Any]]] = {}
    records = []
    for backend in BACKENDS:
        record = stage24e.run_native_edges(edge_csv, backend, 1, build)
        records.append(record)
        native = stage24.merge_native_edges(Path(record["result_path"]), backend)
        manifests[backend] = stage24e.edge_manifest(selected_rows, native, backend)
        write_jsonl(run_dir / f"stage24f_edges_{backend}.jsonl", manifests[backend])
    write_json(run_dir / "stage24f_edge_validation_runs.json", records)
    return manifests, {"native_checked": len(selected_rows), "records": records, "build": build}


def merged_edge_evidence(base_edges: list[dict[str, Any]], manifests: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fcl = {(
        int(x["from_waypoint"]), int(x["to_waypoint"]), str(x.get("from_candidate_id", x.get("source_node"))), str(x.get("to_candidate_id", x.get("target_node")))
    ): x for x in manifests.get("fcl", []) if x.get("status") == "accepted" and x.get("valid") is True}
    bullet = {(
        int(x["from_waypoint"]), int(x["to_waypoint"]), str(x.get("from_candidate_id", x.get("source_node"))), str(x.get("to_candidate_id", x.get("target_node")))
    ): x for x in manifests.get("bullet", []) if x.get("status") == "accepted" and x.get("valid") is True}
    combined = list(base_edges)
    for key in sorted(set(fcl) & set(bullet)):
        row = dict(fcl[key])
        row["source_node"] = key[2]
        row["target_node"] = key[3]
        row["accepted"] = True
        row["fcl_edge_valid"] = True
        row["bullet_edge_valid"] = True
        row["joint_gate_valid"] = True
        row["source_stage"] = "Stage_2_4F_native"
        combined.append(normalize_edge(row, "Stage_2_4F_native"))
    comparison = {
        "fcl_accepted": len(fcl),
        "bullet_accepted": len(bullet),
        "dual_backend_accepted": len(set(fcl) & set(bullet)),
        "fcl_bullet_edge_difference": len(set(fcl) ^ set(bullet)),
        "fcl_only": [list(x) for x in sorted(set(fcl) - set(bullet))],
        "bullet_only": [list(x) for x in sorted(set(bullet) - set(fcl))],
        "sample_level_symmetric_difference": None,
        "collision_method": "adaptive_discrete_interpolation",
        "CCD": "not_available",
    }
    return dedup_edges(combined), comparison


def transition_summary(edges: list[dict[str, Any]], nodes: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    edge_groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        edge_groups[(int(edge["from_waypoint"]), int(edge["to_waypoint"]))].append(edge)
    for wp in range(WAYPOINT_COUNT):
        key = (wp, (wp + 1) % WAYPOINT_COUNT)
        values = edge_groups[key]
        best = min(values, key=lambda x: (float(x["max_joint_delta_deg"]), str(x["from_candidate_id"]), str(x["to_candidate_id"]))) if values else None
        transition = f"{wp}->{(wp + 1) % WAYPOINT_COUNT}"
        rows.append({
            "source_waypoint": wp,
            "target_waypoint": (wp + 1) % WAYPOINT_COUNT,
            "transition": transition,
            "transition_type": "closure" if wp == 719 else "ordinary",
            "dual_backend_valid_edge_count": len(values),
            "minimum_max_joint_step_deg": best["max_joint_delta_deg"] if best else None,
            "best_source_candidate_id": best["from_candidate_id"] if best else None,
            "best_target_candidate_id": best["to_candidate_id"] if best else None,
            "fragile_transition": transition in FRAGILE_TRANSITIONS,
            "fragile_transition_preserved": bool(values) if transition in FRAGILE_TRANSITIONS else None,
            "collision_method": "adaptive_discrete_interpolation",
            "clearance_m": None,
            "clearance_status": "not_available",
        })
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def selected_path_records(nodes: dict[str, dict[str, Any]], open_chain: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    node_rows = [nodes[cid] for cid in open_chain.get("selected_node_ids", []) if cid in nodes]
    edge_rows = list(open_chain.get("selected_edges", []))
    return node_rows, edge_rows


def branch_analysis(nodes: dict[str, dict[str, Any]], edges: list[dict[str, Any]], open_chain: dict[str, Any], rejected: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    path = open_chain.get("selected_node_ids", [])
    branch_rows: list[dict[str, Any]] = []
    for index, cid in enumerate(path):
        row = nodes[cid]
        previous = nodes[path[index - 1]]["ik_branch_signature"] if index else None
        branch_rows.append({"sequence_index": index, "waypoint_index": row["waypoint_index"], "candidate_id": cid, "ik_branch_signature": row["ik_branch_signature"], "shoulder_branch": row["shoulder_branch"], "elbow_branch": row["elbow_branch"], "wrist_branch": row["wrist_branch"], "branch_switch": bool(previous is not None and previous != row["ik_branch_signature"]), "source_stage": row["source_stage"]})
    transition_analysis: dict[str, Any] = {
        "branch_switches_on_selected_open_chain": [row for row in branch_rows if row["branch_switch"]],
        "blocked_reasons_from_joint_gate_rejections": defaultdict(int),
        "waypoint_zero_candidates": sorted(cid for cid, row in nodes.items() if int(row["waypoint_index"]) == 0),
        "waypoint_719_candidates": sorted(cid for cid, row in nodes.items() if int(row["waypoint_index"]) == 719),
        "seam_waypoints": [715, 716, 717, 718, 719, 0, 1, 2, 3, 4],
    }
    for row in rejected:
        transition_analysis["blocked_reasons_from_joint_gate_rejections"][str(row.get("reject_reason"))] += 1
    transition_analysis["blocked_reasons_from_joint_gate_rejections"] = dict(transition_analysis["blocked_reasons_from_joint_gate_rejections"])
    gap_rows: list[dict[str, Any]] = []
    for edge in edges:
        if int(edge["from_waypoint"]) <= 719:
            gap_rows.append({"from_waypoint": edge["from_waypoint"], "to_waypoint": edge["to_waypoint"], "transition": f"{edge['from_waypoint']}->{edge['to_waypoint']}", "from_candidate_id": edge["from_candidate_id"], "to_candidate_id": edge["to_candidate_id"], "j3_delta_deg": edge["signed_joint_delta_deg"][2], "j4_delta_deg": edge["signed_joint_delta_deg"][3], "j5_delta_deg": edge["signed_joint_delta_deg"][4], "max_joint_delta_deg": edge["max_joint_delta_deg"]})
    return branch_rows, [{"from_waypoint": row["from_waypoint"], "to_waypoint": row["to_waypoint"], "from_candidate_id": row["from_candidate_id"], "to_candidate_id": row["to_candidate_id"], "reason": row.get("reject_reason")} for row in rejected], transition_analysis, gap_rows


def run_one(run_dir: Path, run_index: int, contract: dict[str, Any], input_manifest: dict[str, Any], environment: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "stage24f_input_manifest.json", {**input_manifest, "run_index": run_index})
    write_json(run_dir / "stage24f_environment_manifest.json", {**environment, "run_index": run_index})
    (run_dir / "stage24f_frozen_search_contract.yaml").write_text(yaml.safe_dump(contract, allow_unicode=True, sort_keys=False), encoding="utf-8")
    frozen_nodes, frozen_edges = load_frozen_graph()
    semantics = joint_semantics()
    waypoints = {int(row["waypoint_id"]): row for row in read_csv(SOURCE / "waypoints.csv")}
    solver = stage24e.solver_from_contract(contract)
    nodes = dict(frozen_nodes)
    existing_q = {(int(row["waypoint_index"]), tuple(round(float(v), 10) for v in row["joint_vector_rad"])) for row in nodes.values()}
    all_attempts: list[dict[str, Any]] = []
    all_optimizer_runs: list[dict[str, Any]] = []
    all_candidates: list[dict[str, Any]] = []
    window_history: list[dict[str, Any]] = []
    adaptive_previous: list[dict[str, Any]] = []
    base_search = search_open_chain(nodes, frozen_edges)
    selected_path = base_search.get("selected_node_ids") if base_search.get("complete_0_to_719_open_chain_exists") else None
    for window_size, window_waypoints in WINDOWS:
        candidates, attempts, optimizer_runs = generate_continuation(solver, waypoints, nodes, window_size, window_waypoints, selected_path, contract, existing_q)
        all_candidates.extend(candidates)
        all_attempts.extend(attempts)
        all_optimizer_runs.extend(optimizer_runs)
        for row in candidates:
            nodes[row["candidate_id"]] = normalize_node(row, "Stage_2_4F", False, False)
        provisional = search_open_chain(nodes, frozen_edges)
        if provisional.get("complete_0_to_719_open_chain_exists"):
            selected_path = provisional["selected_node_ids"]
        valid_signatures = sorted({branch_signature(row) for row in candidates})
        best_closure = None
        endpoint_rows = [row for row in nodes.values() if row["waypoint_index"] in {0, 719}]
        for left in [row for row in endpoint_rows if row["waypoint_index"] == 719]:
            for right in [row for row in endpoint_rows if row["waypoint_index"] == 0]:
                _, absolute, limits_ok, blocking = raw_delta(left["joint_vector_rad"], right["joint_vector_rad"], semantics)
                value = math.degrees(max(absolute, default=0.0))
                key = (value, str(left["candidate_id"]), str(right["candidate_id"]))
                if best_closure is None or key < best_closure[0]:
                    best_closure = (key, blocking)
        window_row = {
            "window_size": window_size,
            "waypoints_modified": list(window_waypoints),
            "new_candidates": len(candidates),
            "valid_candidates": None,
            "branch_signatures": valid_signatures,
            "compatible_open_chain_found": provisional.get("complete_0_to_719_open_chain_exists", False),
            "best_closure_max_joint_delta_deg": best_closure[0][0] if best_closure else None,
            "best_closure_blocking_joint": best_closure[1] if best_closure else None,
            "closure_pairs_passing_joint_gate": None,
            "closure_pairs_native_checked": None,
            "candidate_generation_status": "generated_and_pending_native_node_gate",
        }
        window_history.append(window_row)
        adaptive_previous.append({"window_size": window_size, "candidate_count": len(candidates), "attempt_count": len(attempts)})
        # Native node validation is intentionally delayed until the complete
        # deterministic candidate set is generated, so the graph never mixes
        # formally checked and unchecked nodes.
    write_jsonl(run_dir / "stage24f_candidate_generation.jsonl", all_attempts)
    write_jsonl(run_dir / "stage24f_optimizer_runs.jsonl", all_optimizer_runs)
    write_jsonl(run_dir / "stage24f_candidates.jsonl", all_candidates)
    write_json(run_dir / "stage24f_adaptive_generation_summary.json", {"windows": window_history, "total_new_candidates": len(all_candidates), "candidate_attempts": len(all_attempts), "optimizer_runs": len(all_optimizer_runs), "random_api_calls": 0})
    write_csv(run_dir / "stage24f_adaptive_window_history.csv", window_history)
    dual_new_ids, node_audit, node_records = node_validation(run_dir, all_candidates)
    for row in all_candidates:
        cid = row["candidate_id"]
        if cid in nodes:
            nodes[cid]["fcl_node_valid"] = cid in dual_new_ids
            nodes[cid]["bullet_node_valid"] = cid in dual_new_ids
            nodes[cid]["dual_backend_valid"] = cid in dual_new_ids
    all_node_rows = [nodes[cid] for cid in sorted(nodes, key=lambda x: (int(nodes[x]["waypoint_index"]), x))]
    write_jsonl(run_dir / "stage24f_layered_graph_nodes.jsonl", all_node_rows)
    pending, rejected, selected_native = prefilter_new_edges(nodes, {row["candidate_id"] for row in all_candidates}, semantics)
    write_jsonl(run_dir / "stage24f_closure_joint_gate_rejected.jsonl", [row for row in rejected if int(row["from_waypoint"]) == 719])
    write_jsonl(run_dir / "stage24f_all_joint_gate_rejected.jsonl", rejected)
    write_json(run_dir / "stage24f_edge_prefilter_summary.json", {"prefilter_pair_count": len(pending), "joint_gate_rejected_count": len(rejected), "native_request_count": len(selected_native), "ordinary_edge_cap_per_transition": MAX_NATIVE_EDGE_REQUESTS_PER_TRANSITION, "closure_edge_cap": MAX_NATIVE_CLOSURE_REQUESTS, "joint_gate_before_native": True})
    manifests, native_meta = native_edge_validation(run_dir, selected_native)
    edges, comparison = merged_edge_evidence(frozen_edges, manifests)
    write_json(run_dir / "stage24f_fcl_bullet_comparison.json", {**comparison, "node_difference": node_audit.get("fcl_bullet_node_difference", 0), "native_meta": native_meta})
    write_jsonl(run_dir / "stage24f_closure_native_edge_evidence.jsonl", [{"fcl": f, "bullet": next((b for b in manifests.get("bullet", []) if b.get("from_candidate_id") == f.get("from_candidate_id") and b.get("to_candidate_id") == f.get("to_candidate_id")), None)} for f in manifests.get("fcl", []) if int(f["from_waypoint"]) == 719])
    write_jsonl(run_dir / "stage24f_layered_graph_edges.jsonl", edges)
    open_chain = search_open_chain(nodes, edges)
    closed_cycle = search_closed_cycle(nodes, edges, open_chain)
    graph_nodes, graph_edges = build_layered_graph(nodes, edges)
    write_json(run_dir / "stage24f_graph_reconstruction_report.json", {
        "schema_version": "stage24f-graph-reconstruction-v1",
        "waypoint_count": WAYPOINT_COUNT,
        "node_count": len(graph_nodes),
        "edge_count": len(graph_edges),
        "ordinary_transition_count": sum(1 for edge in graph_edges if int(edge["from_waypoint"]) != 719),
        "closure_transition_count": sum(1 for edge in graph_edges if int(edge["from_waypoint"]) == 719 and int(edge["to_waypoint"]) == 0),
        "complete_0_to_719_open_chain_exists": bool(open_chain.get("complete_0_to_719_open_chain_exists")),
        "complete_720_node_720_edge_cycle_found": bool(closed_cycle.get("complete_720_node_720_edge_cycle_found")),
        "component_count": None,
        "source_stage_counts": {stage: sum(1 for row in graph_nodes if row.get("source_stage") == stage) for stage in sorted({str(row.get("source_stage")) for row in graph_nodes})},
        "fcl_bullet_node_difference": node_audit.get("fcl_bullet_node_difference", 0),
        "fcl_bullet_edge_difference": comparison.get("fcl_bullet_edge_difference", 0),
        "reconstruction_policy": "reuse frozen Stage 2.4D/2.4E accepted dual-backend edges; add only newly generated Stage 2.4F native dual-backend edges",
        "collision_method": "adaptive_discrete_interpolation",
    })
    write_json(run_dir / "stage24f_selected_open_chain.json", {**open_chain, "selected_node_count": len(open_chain.get("selected_node_ids", [])), "selected_edge_count": len(open_chain.get("selected_edges", []))})
    write_json(run_dir / "stage24f_selected_closed_cycle.json", closed_cycle)
    open_node_rows, open_edge_rows = selected_path_records(nodes, open_chain)
    write_csv(run_dir / "stage24f_selected_open_chain.csv", open_node_rows)
    write_jsonl(run_dir / "stage24f_open_chain_transition_evidence.jsonl", open_edge_rows)
    write_csv(run_dir / "stage24f_selected_closed_cycle.csv", [nodes[cid] for cid in closed_cycle.get("selected_node_ids", []) if cid in nodes])
    write_jsonl(run_dir / "stage24f_closed_cycle_edge_evidence.jsonl", closed_cycle.get("selected_edges", []))
    membership, components = weak_components(nodes, edges)
    graph_report = read_json(run_dir / "stage24f_graph_reconstruction_report.json")
    graph_report["component_count"] = len(components)
    write_json(run_dir / "stage24f_graph_reconstruction_report.json", graph_report)
    write_json(run_dir / "stage24f_component_summary.json", {"component_count": len(components), "components": components, "waypoint_0_components": sorted({membership.get(cid) for cid, row in nodes.items() if row["waypoint_index"] == 0 and cid in membership}), "waypoint_719_components": sorted({membership.get(cid) for cid, row in nodes.items() if row["waypoint_index"] == 719 and cid in membership})})
    write_csv(run_dir / "stage24f_candidate_component_membership.csv", [{"candidate_id": cid, "waypoint_index": row["waypoint_index"], "component_id": membership.get(cid), "ik_branch_signature": row["ik_branch_signature"], "dual_backend_valid": row["dual_backend_valid"]} for cid, row in sorted(nodes.items())])
    branch_rows, rejected_rows, branch_analysis_data, gap_rows = branch_analysis(nodes, edges, open_chain, rejected)
    write_csv(run_dir / "stage24f_branch_sequence.csv", branch_rows)
    write_json(run_dir / "stage24f_branch_transition_analysis.json", branch_analysis_data)
    write_csv(run_dir / "stage24f_joint_gap_accumulation.csv", gap_rows)
    summary = transition_summary(edges, nodes)
    write_csv(run_dir / "stage24f_all_720_transition_summary.csv", summary)
    fragile_graph = {transition: next((x for x in summary if x["transition"] == transition), None) for transition in FRAGILE_TRANSITIONS}
    fragile_preserved_graph = all(row is not None and row["dual_backend_valid_edge_count"] > 0 for row in fragile_graph.values())
    selected_fragile = all(any(f"{e['from_waypoint']}->{e['to_waypoint']}" == transition for e in open_chain.get("selected_edges", [])) for transition in FRAGILE_TRANSITIONS) if open_chain.get("complete_0_to_719_open_chain_exists") else False
    endpoint_nodes = [row for row in nodes.values() if row["waypoint_index"] in {0, 719} and row["dual_backend_valid"]]
    closure_total = len([1 for a in endpoint_nodes if a["waypoint_index"] == 719 for b in endpoint_nodes if b["waypoint_index"] == 0])
    closure_gate_rows = [row for row in pending if int(row["from_waypoint"]) == 719]
    closure_gate_rows.sort(key=lambda x: (float(x["max_joint_delta_deg"]), str(x["from_candidate_id"]), str(x["to_candidate_id"])))
    closure_pass = [row for row in closure_gate_rows if row["joint_gate_valid"]]
    native_checked = len([row for row in selected_native if int(row["from_waypoint"]) == 719])
    dual_closure = [e for e in edges if int(e["from_waypoint"]) == 719 and int(e["to_waypoint"]) == 0 and e.get("source_stage") == "Stage_2_4F_native"]
    if not open_chain.get("complete_0_to_719_open_chain_exists"):
        status = "blocked_no_compatible_0_to_719_open_chain"
    elif not closure_pass:
        status = "blocked_no_joint_gate_admissible_closure_pair_after_adaptive_search"
    elif native_checked > 0 and not dual_closure:
        status = "blocked_no_dual_backend_valid_closure_edge"
    elif not closed_cycle.get("complete_720_node_720_edge_cycle_found"):
        status = "blocked_closure_edge_not_part_of_complete_cycle"
    else:
        status = "passed"
    gate = {
        "Stage_2_4F": status,
        "complete_0_to_719_open_chain_exists": bool(open_chain.get("complete_0_to_719_open_chain_exists")),
        "selected_open_chain_max_joint_step_deg": open_chain.get("objective_cost", [None])[0] if open_chain.get("objective_cost") else None,
        "graph_node_count": len(graph_nodes),
        "graph_edge_count": len(graph_edges),
        "graph_component_count": len(components),
        "selected_open_chain_branch_switch_count": sum(1 for row in branch_rows if row.get("branch_switch")),
        "adaptive_windows_attempted": [row["window_size"] for row in window_history],
        "new_candidates_generated": len(all_candidates),
        "dual_backend_valid_new_nodes": len(dual_new_ids),
        "best_closure_candidate_ids": [closure_gate_rows[0]["from_candidate_id"], closure_gate_rows[0]["to_candidate_id"]] if closure_gate_rows else None,
        "best_closure_joint_deltas_deg": [math.degrees(x) for x in (closure_gate_rows[0]["signed_delta_rad"] if closure_gate_rows else [])] if closure_gate_rows else None,
        "best_closure_max_joint_delta_deg": closure_gate_rows[0]["max_joint_delta_deg"] if closure_gate_rows else None,
        "best_closure_blocking_joint": closure_gate_rows[0]["blocking_joint"] if closure_gate_rows else None,
        "closure_pair_count_total": closure_total,
        "closure_pair_count_node_valid": closure_total,
        "closure_pairs_passing_20deg_joint_gate": len(closure_pass),
        "closure_pairs_native_checked": native_checked,
        "dual_backend_valid_closure_edges": len(dual_closure),
        "fcl_bullet_node_difference": node_audit.get("fcl_bullet_node_difference", 0),
        "fcl_bullet_edge_difference": comparison.get("fcl_bullet_edge_difference", 0),
        "complete_720_node_720_edge_cycle_found": bool(closed_cycle.get("complete_720_node_720_edge_cycle_found")),
        "fragile_transitions_preserved": 32 if fragile_graph and fragile_preserved_graph else 0,
        "fragile_transitions_in_selected_open_chain": 32 if selected_fragile else 0,
        "deterministic_rebuild": "pending",
        "all_nodes_dual_backend_valid": all(row["dual_backend_valid"] for row in nodes.values()),
        "all_edges_dual_backend_valid": all(edge["fcl_edge_valid"] and edge["bullet_edge_valid"] for edge in edges),
        "collision_method": "adaptive_discrete_interpolation",
        "CCD": "not_available",
        "clearance": "not_available",
        "Ruckig": "not_run",
        "TOTG": "not_run",
        "GNN": "not_run",
        "Stage_2_5": "blocked",
        "ready_for_Stage_2_5": status == "passed",
        "continuous_tolerance_space_infeasible_proven": False,
        "discrete_stage24e_search_exhausted": True,
        "stage24f_adaptive_search_executed": True,
        "closure_pairs_evaluated": closure_total,
        "complete_continuous_tolerance_space_covered": False,
        "native_node_validation": node_audit,
        "native_edge_validation": native_meta,
    }
    write_json(run_dir / "stage24f_gate_report.json", gate)
    write_json(run_dir / "stage24f_closed_cycle_certificate.json", {"certificate_status": "passed" if status == "passed" else "blocked", "gate": gate, "selected_cycle": closed_cycle, "all_waypoints_covered": len(closed_cycle.get("selected_node_ids", [])) == 720})
    write_json(run_dir / "stage24f_search_coverage_certificate.json", {"discrete_stage24e_search_exhausted": True, "stage24f_adaptive_search_executed": True, "adaptive_windows_attempted": window_history, "continuous_optimizer_runs": len(all_optimizer_runs), "branch_signatures_attempted": sorted({row["ik_branch_signature"] for row in all_attempts if row.get("ik_branch_signature")}), "closure_pairs_evaluated": closure_total, "closure_pairs_passing_joint_gate": len(closure_pass), "closure_pairs_native_checked": native_checked, "complete_continuous_tolerance_space_covered": False, "continuous_tolerance_space_infeasible_proven": False})
    report = [
        "# Stage 2.4F Global Branch-Conditioned Adaptive Closure Search",
        "",
        f"- Stage 2.4F: `{status}`",
        f"- Complete 0→719 open chain: `{gate['complete_0_to_719_open_chain_exists']}`; graph nodes/edges: `{len(graph_nodes)}/{len(graph_edges)}`.",
        f"- Adaptive windows: `{gate['adaptive_windows_attempted']}`; new candidates: `{len(all_candidates)}`; dual-backend-valid new nodes: `{len(dual_new_ids)}`.",
        f"- Closure joint-gate pairs: `{len(closure_pass)}`; native checked: `{native_checked}`; dual-backend-valid closure edges: `{len(dual_closure)}`.",
        f"- Failure layer: `{'open-chain compatibility' if not gate['complete_0_to_719_open_chain_exists'] else 'joint delta gate / native edge / complete-cycle graph compatibility'}`.",
        "",
        "Stage 2.4E frozen facts are preserved. Bounded revolute joints use raw q_to-q_from differences with no periodic winding. Native collision evidence is adaptive_discrete_interpolation; CCD and clearance are unavailable. Stage 2.5, TOTG, Ruckig, GNN, and CCD were not run.",
        "",
        "The adaptive refinement is a finite candidate generator within the frozen tolerance box and is not a proof over the continuous tolerance space.",
        "",
        "```yaml",
        yaml.safe_dump(gate, allow_unicode=True, sort_keys=False),
        "```",
        "",
    ]
    (run_dir / "stage24f_report.md").write_text("\n".join(report), encoding="utf-8")
    (run_dir / "stage24f_commands.log").write_text("python scripts/run_stage24f_global_branch_adaptive_closure.py --output-dir <independent-output>\nNative commands are persisted by stage24e native runner logs.\n", encoding="utf-8")
    (run_dir / "stage24f_stdout.log").write_text(json.dumps({"run_index": run_index, "gate": status}, ensure_ascii=False) + "\n", encoding="utf-8")
    (run_dir / "stage24f_stderr.log").write_text("", encoding="utf-8")
    (run_dir / "stage24f_test_report.txt").write_text("not_run_until_top_level_test_phase\n", encoding="utf-8")
    gate_semantic = {key: value for key, value in gate.items() if key not in {"native_node_validation", "native_edge_validation"}}
    write_json(run_dir / "stage24f_run_metadata.json", {"run_index": run_index, "duration_s": time.time() - started, "candidate_hash": semantic_hash(all_candidates), "node_set_hash": semantic_hash(sorted(dual_new_ids)), "edge_set_hash": semantic_hash(sorted((e["from_waypoint"], e["to_waypoint"], e["from_candidate_id"], e["to_candidate_id"]) for e in edges)), "component_membership_hash": semantic_hash(membership), "selected_open_chain_hash": semantic_hash(open_chain), "selected_closed_cycle_hash": semantic_hash(closed_cycle), "branch_sequence_hash": semantic_hash(branch_rows), "closure_evidence_hash": semantic_hash([e for e in edges if int(e["from_waypoint"]) == 719]), "gate_report_hash": semantic_hash(gate_semantic)})
    return {"gate": gate, "run_metadata": read_json(run_dir / "stage24f_run_metadata.json"), "run_dir": str(run_dir.resolve())}


def copy_run1_outputs(root_out: Path, run1: Path) -> None:
    for item in sorted(run1.iterdir()):
        if item.is_file():
            shutil.copy2(item, root_out / item.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--runs", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = (args.output_dir or ROOT / f"outputs/ik_graph_stage24f_global_branch_adaptive_closure/fr5_scaled_horseshoe_demo_v45_{timestamp}").resolve()
    if out.exists():
        raise RuntimeError(f"refusing to overwrite existing Stage 2.4F output directory: {out}")
    out.mkdir(parents=True, exist_ok=False)
    before = git_snapshot()
    input_manifest = authoritative_input_manifest(before)
    environment = environment_manifest()
    contract = frozen_contract()
    write_json(out / "stage24f_input_manifest.json", input_manifest)
    write_json(out / "stage24f_environment_manifest.json", environment)
    (out / "stage24f_frozen_search_contract.yaml").write_text(yaml.safe_dump(contract, allow_unicode=True, sort_keys=False), encoding="utf-8")
    write_json(out / "stage24e_frozen_facts.json", input_manifest["frozen_stage24e_facts"])
    run_results: list[dict[str, Any]] = []
    error: dict[str, Any] | None = None
    try:
        for index in range(1, args.runs + 1):
            run_results.append(run_one(out / f"run{index}", index, contract, input_manifest, environment))
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        (out / "stage24f_stderr.log").write_text(traceback.format_exc(), encoding="utf-8")
    if run_results:
        copy_run1_outputs(out, Path(run_results[0]["run_dir"]))
    semantic_keys = ["candidate_hash", "node_set_hash", "edge_set_hash", "component_membership_hash", "selected_open_chain_hash", "selected_closed_cycle_hash", "branch_sequence_hash", "closure_evidence_hash", "gate_report_hash"]
    determinism = {"independent_runs": len(run_results), "semantic_hashes": {key: [result["run_metadata"].get(key) for result in run_results] for key in semantic_keys}, "differences": {key: len({result["run_metadata"].get(key) for result in run_results}) - 1 for key in semantic_keys}, "deterministic_rebuild": error is None and all(len({result["run_metadata"].get(key) for result in run_results}) == 1 for key in semantic_keys), "nonsemantic_fields_excluded": ["paths", "timestamps", "durations"]}
    write_json(out / "stage24f_determinism_report.json", determinism)
    if error:
        gate = {"Stage_2_4F": "blocked_environment", "ready_for_Stage_2_5": False, "error": error, "continuous_tolerance_space_infeasible_proven": False, "collision_method": "adaptive_discrete_interpolation", "CCD": "not_available", "clearance": "not_available", "Stage_2_5": "blocked"}
    elif not determinism["deterministic_rebuild"]:
        gate = read_json(out / "stage24f_gate_report.json")
        gate.update({"Stage_2_4F": "blocked_nondeterministic", "ready_for_Stage_2_5": False, "deterministic_rebuild": "failed"})
    else:
        gate = read_json(out / "stage24f_gate_report.json")
        gate["deterministic_rebuild"] = "passed"
    write_json(out / "stage24f_gate_report.json", gate)
    report_path = out / "stage24f_report.md"
    if report_path.exists():
        report_path.write_text(report_path.read_text(encoding="utf-8") + f"\nDeterministic rebuild: `{gate.get('deterministic_rebuild')}`.\n", encoding="utf-8")
    write_json(out / "stage24f_test_report.json", {"status": "pending_top_level_test_phase", "runs": len(run_results), "error": error})
    after = git_status()
    write_json(out / "dirty_worktree_preservation.json", {"git_before": before, "git_after": after, "preexisting_status_sha256": value_hash(before["status_porcelain"]), "preexisting_status_preserved": all(line in after for line in before["status_porcelain"].splitlines() if line.strip()), "new_output_root": str(out.resolve())})
    print(json.dumps({"output": str(out.resolve()), "Stage_2_4F": gate.get("Stage_2_4F"), "runs": len(run_results), "deterministic_rebuild": gate.get("deterministic_rebuild"), "error": error}, ensure_ascii=False, sort_keys=True))
    return 0 if gate.get("Stage_2_4F") == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
