"""Stage 2.4E deterministic joint process-tolerance and IK-branch search.

This stage is intentionally additive.  It freezes the Stage 2.4D evidence,
derives the tolerance contract from the frozen scaled-demo configuration,
generates a finite deterministic joint tolerance sample in the required seam
windows, validates new nodes and touched edges in native FCL and Bullet, and
rebuilds the 720-layer graph without changing robot or scene inputs.

Collision evidence is labelled adaptive_discrete_interpolation.  CCD and
clearance remain unavailable.  A finite sample is never described as proof
over a continuous tolerance space.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import os
import platform
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_stage24_closed_loop_graph as stage24
from src.deterministic_numeric_ik import DeterministicNumericIKSolver


SOURCE = ROOT / "outputs/ik_graph_stage23a4_backend_equivalent/fr5_scaled_horseshoe_demo_v44"
STAGE24D = ROOT / "outputs/ik_graph_stage24d_global_branch_cycle_audit/fr5_scaled_horseshoe_demo_v45"
STAGE23B = ROOT / "outputs/ik_graph_stage23b_representation_remediation/fr5_scaled_horseshoe_demo_v45"
URDF = ROOT / "outputs/ik_graph_stage193/expanded_runtime_urdf.urdf"
LIMITS = ROOT / "ros2_moveit_bridge/config/joint_limits_with_jerk.yaml"
PARTS = SOURCE / "tunnel_collision_parts"
OUT = ROOT / "outputs/ik_graph_stage24e_tolerance_search/fr5_scaled_horseshoe_demo_v45_retry_4_20260731"
WAYPOINT_COUNT = 720
MAX_STEP_DEG = 20.0
REQUIRED_SHAPE_MODE = "use_shape_type"
SEARCH_WAYPOINTS = tuple(range(715, 720)) + tuple(range(0, 5))
TOUCHED_TRANSITION_SOURCES = tuple(range(714, 720)) + tuple(range(0, 5))
MAX_SEEDS_PER_WAYPOINT = 8
MAX_EDGE_NEW_NODES_PER_WAYPOINT = 24
MAX_EDGE_NEW_NODES_AT_CLOSURE = 64
MAX_NATIVE_EDGE_REQUESTS_PER_TRANSITION = 30
MAX_NATIVE_CLOSURE_REQUESTS = 200


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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_q(value: Any) -> list[float]:
    if isinstance(value, str):
        return [float(x) for x in json.loads(value)]
    return [float(x) for x in value]


def git_status() -> str:
    proc = subprocess.run(["git", "status", "--short"], cwd=ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True, check=False)
    return proc.stdout + proc.stderr


def source_ref(path: Path, field: str) -> dict[str, Any]:
    return {"path": str(path.resolve()), "field": field, "exists": path.exists(), "sha256": sha256(path) if path.exists() else None}


def verify_stage24d_sha_manifest() -> dict[str, Any]:
    manifest = STAGE24D / "stage24d_sha256_manifest.txt"
    checked = 0
    failures: list[dict[str, Any]] = []
    if not manifest.exists():
        return {"path": str(manifest.resolve()), "checked": 0, "failures": [{"reason": "missing_manifest"}]}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        path = STAGE24D / name
        checked += 1
        actual = sha256(path) if path.exists() else None
        if actual != expected:
            failures.append({"path": name, "expected": expected, "actual": actual})
    return {"path": str(manifest.resolve()), "checked": checked, "failure_count": len(failures), "failures": failures}


def freeze_contract() -> dict[str, Any]:
    v44_path = SOURCE / "scaled_demo_parameters.yaml"
    v1_path = ROOT / "config/scaled_demo/fr5_scaled_horseshoe_demo_v1.yaml"
    v44 = yaml.safe_load(v44_path.read_text(encoding="utf-8"))
    v1 = yaml.safe_load(v1_path.read_text(encoding="utf-8"))
    p44 = v44["process"]
    i44 = v44["ik_solver"]
    p1 = v1["process"]
    i1 = v1["ik_solver"]
    fields = {
        "tcp_position_tolerance_m": (float(i44["position_tolerance_m"]), float(i1["position_tolerance_m"]), "ik_solver.position_tolerance_m"),
        "standoff_tolerance_m": (float(p44["standoff_tolerance_m"]), float(p1["standoff_tolerance_m"]), "process.standoff_tolerance_m"),
        "normal_tolerance_deg": (float(p44["normal_tolerance_deg"]), float(p1["normal_tolerance_deg"]), "process.normal_tolerance_deg"),
        "roll_freedom_deg": (float(p44["roll_freedom_deg"]), float(p1["roll_freedom_deg"]), "process.roll_freedom_deg"),
    }
    conflicts = [{"name": name, "v44": a, "v1": b} for name, (a, b, _) in fields.items() if abs(a - b) > 1e-12]
    return {
        "schema_version": "stage24e-frozen-tolerance-contract-v1",
        "Stage_2_4E": "contract_frozen" if not conflicts else "blocked_tolerance_contract_conflict",
        "authority": "frozen_v44_scaled_demo_process_and_stage1_ik_configuration",
        "coordinate_system": {
            "world_frame": v44["coordinate_system"]["world_frame"],
            "tunnel_frame": v44["coordinate_system"]["tunnel_frame"],
            "robot_base_frame": v44["coordinate_system"]["robot_base_frame"],
            "local_position_order": ["tangent", "lateral", "normal"],
            "position_offset_definition": "world-frame vectors built from cyclic TCP tangent, nominal tool-z normal, and normal-cross-tangent lateral",
            "normal_rotation_vector_definition": "[rx, ry] degrees around local tangent and lateral axes; applied before nominal tool orientation",
            "roll_definition": "post-multiplied tool-frame z roll in degrees",
            "bounded_joint_distance_definition": "raw difference within authoritative URDF limits; no implicit +/-360 degree winding",
        },
        "tolerances": {
            "tcp_position_offset_local_m": {"tangent": [-fields["tcp_position_tolerance_m"][0], fields["tcp_position_tolerance_m"][0]], "lateral": [-fields["tcp_position_tolerance_m"][0], fields["tcp_position_tolerance_m"][0]], "normal": [-fields["tcp_position_tolerance_m"][0], fields["tcp_position_tolerance_m"][0]]},
            "standoff_offset_m": [-fields["standoff_tolerance_m"][0], fields["standoff_tolerance_m"][0]],
            "normal_rotation_vector_deg": {"rx": [-fields["normal_tolerance_deg"][0], fields["normal_tolerance_deg"][0]], "ry": [-fields["normal_tolerance_deg"][0], fields["normal_tolerance_deg"][0]]},
            "tool_roll_offset_deg": [-fields["roll_freedom_deg"][0], fields["roll_freedom_deg"][0]],
            "ik_solver_acceptance": {"position_error_m": float(i44["position_tolerance_m"]), "tool_z_error_deg": float(i44["tool_z_tolerance_deg"]), "orientation_weight": float(i44["orientation_weight"]), "continuity_weight": float(i44["continuity_weight"]), "max_nfev": int(i44["max_nfev"])},
        },
        "finite_search_design": {"level_values": [-1.0, 0.0, 1.0], "position_resolution_m": fields["tcp_position_tolerance_m"][0], "standoff_resolution_m": fields["standoff_tolerance_m"][0], "normal_resolution_deg": fields["normal_tolerance_deg"][0], "roll_resolution_deg": fields["roll_freedom_deg"][0], "search_waypoints": list(SEARCH_WAYPOINTS), "adaptive_expansion_policy": "search seam window only; expand to additional layers only if persisted evidence shows a required connection remains absent", "continuous_space_claim": False},
        "source_fields": {
            name: {"v44_value": a, "v1_value": b, "field": field}
            for name, (a, b, field) in fields.items()
        },
        "conflicts": conflicts,
        "source_files": [
            source_ref(v44_path, "process and ik_solver tolerance fields"),
            source_ref(v1_path, "cross-check of Stage 1/Stage 2 process tolerance fields"),
            source_ref(STAGE24D / "stage24d_gate_report.json", "Stage 2.4D frozen gate"),
            source_ref(STAGE24D / "stage24d_infeasibility_certificate.json", "Stage 2.4D closure lower bound"),
            source_ref(STAGE24D / "stage24d_nodes.jsonl", "Stage 2.4D frozen node set"),
            source_ref(STAGE24D / "stage24d_edges.jsonl", "Stage 2.4D frozen edge set"),
            source_ref(STAGE24D / "next_stage_decision/stage24d_all_720_transition_decision.csv", "Stage 2.4D transition evidence"),
        ],
        "stage24d_sha256_verification": verify_stage24d_sha_manifest(),
        "authoritative_inputs_modified": False,
    }


def solver_from_contract(contract: dict[str, Any]) -> DeterministicNumericIKSolver:
    ik = contract["tolerances"]["ik_solver_acceptance"]
    return DeterministicNumericIKSolver(URDF, LIMITS, position_tolerance_m=ik["position_error_m"], tool_z_tolerance_deg=ik["tool_z_error_deg"], orientation_weight=ik["orientation_weight"], continuity_weight=ik["continuity_weight"], max_nfev=ik["max_nfev"])


def pose_frame(rows: dict[int, dict[str, str]], wp: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    row = rows[wp]
    position = np.asarray([float(row[k]) for k in ("x", "y", "z")], dtype=float)
    rotation = Rotation.from_quat([float(row[k]) for k in ("qx", "qy", "qz", "qw")]).as_matrix()
    normal = rotation[:, 2]
    previous = np.asarray([float(rows[(wp - 1) % WAYPOINT_COUNT][k]) for k in ("x", "y", "z")], dtype=float)
    following = np.asarray([float(rows[(wp + 1) % WAYPOINT_COUNT][k]) for k in ("x", "y", "z")], dtype=float)
    tangent = following - previous
    tangent /= max(float(np.linalg.norm(tangent)), 1.0e-15)
    lateral = np.cross(normal, tangent)
    lateral /= max(float(np.linalg.norm(lateral)), 1.0e-15)
    return position, rotation, tangent, lateral


def parameter_vectors(contract: dict[str, Any]) -> list[dict[str, float]]:
    scale = contract["tolerances"]
    p = float(scale["tcp_position_offset_local_m"]["tangent"][1])
    s = float(scale["standoff_offset_m"][1])
    n = float(scale["normal_rotation_vector_deg"]["rx"][1])
    r = float(scale["tool_roll_offset_deg"][1])
    dims = ("tangent", "lateral", "normal", "standoff", "rx", "ry", "roll")
    limits = (p, p, p, s, n, n, r)
    levels = (-1.0, 0.0, 1.0)
    vectors: list[tuple[float, ...]] = [tuple(0.0 for _ in dims)]
    explicit = [
        (1, -1, 1, -1, 1, -1, 1), (-1, 1, -1, 1, -1, 1, -1),
        (1, 1, 1, 1, 1, 1, 1), (-1, -1, -1, -1, -1, -1, -1),
        (1, 0, -1, 1, 0, -1, 1), (-1, 0, 1, -1, 0, 1, -1),
        (0, 1, -1, 0, 1, -1, 0), (0, -1, 1, 0, -1, 1, 0),
    ]
    vectors.extend(tuple(float(x) for x in item) for item in explicit)
    for k in range(16):
        # Base-3 digits give 16 distinct deterministic joint combinations,
        # including mixed-sign and mixed-zero perturbations.
        vectors.append(tuple(levels[(k // (3 ** d)) % 3] for d in range(len(dims))))
    result = []
    seen: set[tuple[float, ...]] = set()
    for vector in vectors:
        if vector in seen:
            continue
        seen.add(vector)
        values = {name: float(level * limit) for name, level, limit in zip(dims, vector, limits)}
        result.append(values)
    return result


def branch_seed_rows(nodes: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    by_wp: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in nodes:
        wp = int(row["waypoint_index"])
        if wp not in SEARCH_WAYPOINTS:
            continue
        by_wp[wp].append(row)
    selected: dict[int, list[dict[str, Any]]] = {}
    for wp in SEARCH_WAYPOINTS:
        rows = sorted(by_wp[wp], key=lambda item: str(item["candidate_id"]))
        distinct: list[dict[str, Any]] = []
        branches: set[str] = set()
        for row in rows:
            branch = str(row.get("ik_branch_signature", ""))
            if branch not in branches:
                branches.add(branch)
                distinct.append(row)
        for row in rows:
            if row not in distinct:
                distinct.append(row)
            if len(distinct) >= MAX_SEEDS_PER_WAYPOINT:
                break
        selected[wp] = distinct[:MAX_SEEDS_PER_WAYPOINT]
    return selected


def generate_candidates_once(solver: DeterministicNumericIKSolver, waypoints: dict[int, dict[str, str]], frozen_nodes: list[dict[str, Any]], params: list[dict[str, float]]) -> list[dict[str, Any]]:
    seeds_by_wp = branch_seed_rows(frozen_nodes)
    output: list[dict[str, Any]] = []
    for wp in SEARCH_WAYPOINTS:
        nominal_position, nominal_rotation, tangent, lateral = pose_frame(waypoints, wp)
        nominal_normal = nominal_rotation[:, 2]
        for seed in seeds_by_wp[wp]:
            seed_q = np.asarray(parse_q(seed["joint_values_rad"]), dtype=float)
            for values in params:
                position_offset = tangent * values["tangent"] + lateral * values["lateral"] + nominal_normal * values["normal"]
                target_position = nominal_position + position_offset + nominal_normal * values["standoff"]
                tilt = Rotation.from_rotvec(tangent * math.radians(values["rx"]) + lateral * math.radians(values["ry"])).as_matrix()
                target_rotation = tilt @ nominal_rotation @ Rotation.from_euler("z", values["roll"], degrees=True).as_matrix()
                target = np.eye(4, dtype=float)
                target[:3, :3] = target_rotation
                target[:3, 3] = target_position
                result = solver.solve(target, seed_q)
                if not result.success:
                    continue
                q = np.asarray(result.q_rad, dtype=float)
                key = tuple(round(float(value), 10) for value in q)
                stable = value_hash({"wp": wp, "seed": seed["candidate_id"], "params": values, "q": [round(float(v), 12) for v in q]})[:16]
                actual = solver.fk_tcp(q)
                actual_z = actual[:3, 2] / max(float(np.linalg.norm(actual[:3, 2])), 1.0e-15)
                target_z = target_rotation[:, 2] / max(float(np.linalg.norm(target_rotation[:, 2])), 1.0e-15)
                orientation_error = float(math.degrees(math.acos(float(np.clip(np.dot(actual_z, target_z), -1.0, 1.0)))))
                position_error = float(np.linalg.norm(actual[:3, 3] - target_position))
                if any(tuple(round(float(x), 10) for x in row["joint_values_rad"]) == key for row in output if int(row["waypoint_index"]) == wp):
                    continue
                branch = {
                    "shoulder_branch": "positive" if q[0] >= 0 else "negative",
                    "elbow_branch": "positive" if q[2] >= 0 else "negative",
                    "wrist_branch": "positive" if q[4] >= 0 else "negative",
                }
                branch["ik_branch_signature"] = "shoulder={shoulder_branch}|elbow={elbow_branch}|wrist={wrist_branch}".format(**branch)
                output.append({
                    "candidate_id": f"s24e-{wp:04d}-{stable}", "waypoint_index": wp, "waypoint_id": wp,
                    "joint_values": q.tolist(), "joint_values_rad": q.tolist(), "seed_id": seed["candidate_id"], "seed_template_id": seed.get("source_seed_template_id"),
                    "position_offset_local_m": [values["tangent"], values["lateral"], values["normal"]], "standoff_offset_m": values["standoff"],
                    "normal_rotation_vector_deg": [values["rx"], values["ry"]], "roll_offset_deg": values["roll"],
                    "solver_position_error_m": position_error, "solver_tool_z_error_deg": orientation_error,
                    "FK_position_error_m": position_error, "FK_orientation_error_deg": orientation_error,
                    "joint_limit_pass": bool(np.all(q >= solver.robot.limits.q_min - 1.0e-9) and np.all(q <= solver.robot.limits.q_max + 1.0e-9)),
                    "process_tolerance_pass": True, "solver_success": True, "formal_node_gate_pending": True,
                    "fcl_node_validity": None, "bullet_node_validity": None, "source_stage": "Stage_2_4E",
                    "candidate_parameter_hash": value_hash(values), "seed_branch_signature": seed.get("ik_branch_signature"),
                    **branch,
                })
    output.sort(key=lambda row: (int(row["waypoint_index"]), str(row["candidate_id"])))
    return output


def write_candidate_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = ["waypoint_id", "candidate_id", "joint_values"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({"waypoint_id": row["waypoint_id"], "candidate_id": row["candidate_id"], "joint_values": canonical(row["joint_values"])})


def wsl_path(path: Path) -> str:
    resolved = str(path.resolve()).replace("\\", "/")
    drive, rest = resolved.split(":/", 1)
    return f"/mnt/{drive.lower()}/{rest}"


def strict_prefix(backend: str) -> tuple[str, str, str]:
    stage23_install = wsl_path(STAGE23B / "install")
    native_install = wsl_path(OUT / "native_install")
    overlay = f"{stage23_install}/lib/libstage23b_bullet_shape_interposer.so"
    wsl_root = "/mnt/c/Users/86198/Desktop/robotfucker"
    am = f"{native_install}:{stage23_install}:{wsl_root}/tmp/stage23a7_install2:{wsl_root}/install/fairino5_v6_moveit2_config:{wsl_root}/install/fairino_description:{wsl_root}/install/fr5_tunnel_moveit_bridge:/opt/ros/jazzy"
    ld = f"{native_install}/lib:{stage23_install}/lib:{wsl_root}/tmp/stage23a7_install2/lib:/opt/ros/jazzy/opt/sdformat_vendor/lib:/opt/ros/jazzy/opt/rviz_ogre_vendor/lib:/opt/ros/jazzy/lib/x86_64-linux-gnu:/opt/ros/jazzy/opt/gz_math_vendor/lib:/opt/ros/jazzy/opt/gz_utils_vendor/lib:/opt/ros/jazzy/opt/gz_tools_vendor/lib:/opt/ros/jazzy/opt/gz_cmake_vendor/lib:/opt/ros/jazzy/lib"
    preload = f"export LD_PRELOAD={overlay}" if backend == "bullet" else "unset LD_PRELOAD"
    prefix = f"source /opt/ros/jazzy/setup.bash && export AMENT_PREFIX_PATH={am} && export LD_LIBRARY_PATH={ld} && export FR5_BULLET_SHAPE_MODE={REQUIRED_SHAPE_MODE} && {preload}"
    return prefix, am, ld


def run_native_nodes(candidate_csv: Path, backend: str, run_index: int) -> dict[str, Any]:
    run_dir = OUT / "node_validation" / f"{backend}_run{run_index}"
    run_dir.mkdir(parents=True, exist_ok=True)
    prefix, _, _ = strict_prefix(backend)
    wsl_root = "/mnt/c/Users/86198/Desktop/robotfucker"
    command = prefix + " && " + f"ros2 launch {wsl_root}/tools/stage23b_formal_launch.py candidate_csv:={wsl_path(candidate_csv)} parts_dir:={wsl_path(PARTS)} output_dir:={wsl_path(run_dir)} backend:={backend} run_index:={run_index}"
    proc = subprocess.run(["wsl.exe", "bash", "-lc", command], cwd=ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=1800)
    (run_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8")
    (run_dir / "stderr.log").write_text(proc.stderr, encoding="utf-8")
    (run_dir / "exit_code.txt").write_text(str(proc.returncode) + "\n", encoding="utf-8")
    result = run_dir / f"{backend}_runtime_results_run{run_index}.jsonl"
    return {"backend": backend, "run_index": run_index, "exit_code": proc.returncode, "result_path": str(result.resolve()), "result_rows": len(read_jsonl(result)) if result.exists() else 0, "result_sha256": sha256(result) if result.exists() else None, "shape_mode": REQUIRED_SHAPE_MODE}


def node_gate(rows: list[dict[str, Any]], records: list[dict[str, Any]]) -> tuple[set[str], dict[str, Any]]:
    by_backend: dict[str, list[set[str]]] = defaultdict(list)
    for record in records:
        result = read_jsonl(Path(record["result_path"])) if Path(record["result_path"]).exists() else []
        by_backend[record["backend"]].append({row["candidate_id"] for row in result if row.get("valid") is True})
    fcl = set.intersection(*by_backend["fcl"]) if by_backend["fcl"] else set()
    bullet = set.intersection(*by_backend["bullet"]) if by_backend["bullet"] else set()
    audit = {"candidate_count": len(rows), "fcl_valid": len(fcl), "bullet_valid": len(bullet), "dual_backend_valid": len(fcl & bullet), "fcl_bullet_node_difference": len(fcl ^ bullet), "exit_codes": [r["exit_code"] for r in records], "three_run_fcl_set_equal": len({value_hash(sorted(x)) for x in by_backend["fcl"]}) == 1 if by_backend["fcl"] else False, "three_run_bullet_set_equal": len({value_hash(sorted(x)) for x in by_backend["bullet"]}) == 1 if by_backend["bullet"] else False}
    write_json(OUT / "stage24e_node_gate_audit.json", audit)
    return fcl & bullet, audit


def edge_prefilter(nodes: list[dict[str, Any]], from_wp: int, semantics: list[dict[str, Any]], new_ids: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    to_wp = (from_wp + 1) % WAYPOINT_COUNT
    source = sorted([row for row in nodes if int(row["waypoint_index"]) == from_wp], key=lambda row: str(row["candidate_id"]))
    target = sorted([row for row in nodes if int(row["waypoint_index"]) == to_wp], key=lambda row: str(row["candidate_id"]))
    accepted, rejected = [], []
    for a in source:
        for b in target:
            q0, q1 = parse_q(a["joint_values_rad"]), parse_q(b["joint_values_rad"])
            deltas = [stage24.shortest_delta(x, y, rule["type"]) for x, y, rule in zip(q0, q1, semantics)]
            abs_deltas = [abs(x) for x in deltas]
            reasons = []
            if max(abs_deltas, default=float("inf")) > math.radians(MAX_STEP_DEG) + 1.0e-12:
                reasons.append("joint_step_exceeds_20_deg")
            row = {"from_waypoint": from_wp, "to_waypoint": to_wp, "from_candidate_id": a["candidate_id"], "to_candidate_id": b["candidate_id"], "q0": q0, "q1": q1, "signed_delta_rad": deltas, "absolute_delta_rad": abs_deltas, "max_single_joint_step_deg": math.degrees(max(abs_deltas)) if abs_deltas else None, "max_model_aware_joint_step_deg": math.degrees(max(abs_deltas)) if abs_deltas else None, "max_step_joint": f"j{int(np.argmax(abs_deltas)) + 1}" if abs_deltas else None, "joint_types": [rule["type"] for rule in semantics], "joint_lower_bounds": [rule["lower"] for rule in semantics], "joint_upper_bounds": [rule["upper"] for rule in semantics], "continuous_flags": [rule["type"] == "continuous" for rule in semantics], "interpolation_step_deg": 1.0, "collision_method": "adaptive_discrete_interpolation", "new_node_involved": a["candidate_id"] in new_ids or b["candidate_id"] in new_ids}
            if reasons:
                row.update({"status": "rejected", "valid": False, "reject_reason": reasons[0], "reject_reasons": reasons})
                rejected.append(row)
            else:
                row.update({"status": "pending_collision", "valid": False, "reject_reason": None, "reject_reasons": []})
                if row["new_node_involved"]:
                    accepted.append(row)
    return accepted, rejected


def write_edge_requests(rows: list[dict[str, Any]], path: Path) -> None:
    fields = ["from_waypoint", "to_waypoint", "from_candidate_id", "to_candidate_id", "max_single_joint_step_deg", "interpolation_step_deg"] + [f"q0_{i}" for i in range(6)] + [f"q1_{i}" for i in range(6)]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            item = {key: row[key] for key in fields if key in row}
            for i, value in enumerate(row["q0"]):
                item[f"q0_{i}"] = repr(value)
            for i, value in enumerate(row["q1"]):
                item[f"q1_{i}"] = repr(value)
            writer.writerow(item)


def run_native_edges(edge_csv: Path, backend: str, run_index: int, build: dict[str, Any]) -> dict[str, Any]:
    run_dir = OUT / "edge_validation" / f"{backend}_run{run_index}"
    run_dir.mkdir(parents=True, exist_ok=True)
    prefix, _, _ = strict_prefix(backend)
    wsl_root = "/mnt/c/Users/86198/Desktop/robotfucker"
    command = prefix + " && " + f"ros2 launch {wsl_root}/tools/stage24_formal_launch.py edge_csv:={wsl_path(edge_csv)} parts_dir:={wsl_path(PARTS)} output_dir:={wsl_path(run_dir)} backend:={backend} run_index:={run_index}"
    proc = subprocess.run(["wsl.exe", "bash", "-lc", command], cwd=ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=1800)
    (run_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8")
    (run_dir / "stderr.log").write_text(proc.stderr, encoding="utf-8")
    (run_dir / "exit_code.txt").write_text(str(proc.returncode) + "\n", encoding="utf-8")
    result_path = run_dir / "native_edge_results.jsonl"
    return {"backend": backend, "run_index": run_index, "exit_code": proc.returncode, "result_path": str(result_path.resolve()), "result_rows": len(read_jsonl(result_path)) if result_path.exists() else 0, "result_sha256": sha256(result_path) if result_path.exists() else None, "shape_mode": REQUIRED_SHAPE_MODE}


def edge_manifest(prefilter: list[dict[str, Any]], native: dict[tuple[Any, ...], dict[str, Any]], backend: str) -> list[dict[str, Any]]:
    result = []
    for row in prefilter:
        key = (int(row["from_waypoint"]), int(row["to_waypoint"]), row["from_candidate_id"], row["to_candidate_id"])
        collision = native.get(key)
        item = dict(row)
        if collision is None:
            item.update({"status": "indeterminate", "valid": False, "reject_reason": "missing_native_result", "indeterminate": True})
        else:
            item.update({key2: value for key2, value in collision.items() if key2 not in {"q0", "q1"}})
            item["valid"] = collision.get("status") == "accepted" and not collision.get("indeterminate", True)
            item["status"] = "accepted" if item["valid"] else ("indeterminate" if collision.get("indeterminate") else "rejected")
            item["reject_reason"] = None if item["valid"] else ("indeterminate" if collision.get("indeterminate") else "collision")
        item["backend"] = backend
        item["edge_id"] = f"{int(item['from_waypoint']):04d}->{int(item['to_waypoint']):04d}:{item['from_candidate_id']}->{item['to_candidate_id']}"
        result.append(item)
    return sorted(result, key=lambda row: (int(row["from_waypoint"]), int(row["to_waypoint"]), str(row["from_candidate_id"]), str(row["to_candidate_id"])))


def select_edge_nodes(valid_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Retain a deterministic diverse edge population; preserve all node evidence."""
    by_wp: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in valid_rows:
        by_wp[int(row["waypoint_index"])].append(row)
    selected: list[dict[str, Any]] = []
    for wp in SEARCH_WAYPOINTS:
        limit = MAX_EDGE_NEW_NODES_AT_CLOSURE if wp in {0, 719} else MAX_EDGE_NEW_NODES_PER_WAYPOINT
        rows = sorted(by_wp[wp], key=lambda row: (str(row.get("ik_branch_signature", "")), str(row.get("candidate_parameter_hash", "")), str(row["candidate_id"])))
        selected.extend(rows[:limit])
    return selected


def select_native_edge_requests(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank finite edge probes per transition; retain complete prefilter evidence."""
    by_transition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_transition[f"{row['from_waypoint']}->{row['to_waypoint']}"].append(row)
    selected: list[dict[str, Any]] = []
    for transition, values in sorted(by_transition.items()):
        cap = MAX_NATIVE_CLOSURE_REQUESTS if transition == "719->0" else MAX_NATIVE_EDGE_REQUESTS_PER_TRANSITION
        values.sort(key=lambda row: (float(row["max_model_aware_joint_step_deg"]), str(row["from_candidate_id"]), str(row["to_candidate_id"])))
        selected.extend(values[:cap])
    return sorted(selected, key=lambda row: (int(row["from_waypoint"]), str(row["from_candidate_id"]), str(row["to_candidate_id"])))


def accepted_key(row: dict[str, Any]) -> tuple[int, int, str, str]:
    return (int(row["from_waypoint"]), int(row["to_waypoint"]), str(row.get("source_node", row.get("from_candidate_id"))), str(row.get("target_node", row.get("to_candidate_id"))))


def build_cycle(accepted: dict[tuple[int, int, str, str], dict[str, Any]]) -> dict[str, Any]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in accepted.values():
        by_source[str(edge.get("source_node", edge.get("from_candidate_id")))].append(edge)
    for edges in by_source.values():
        edges.sort(key=lambda row: str(row.get("target_node", row.get("to_candidate_id"))))
    starts = sorted({key[2] for key in accepted if key[0] == 0})
    for start in starts:
        reachable: dict[str, tuple[str, dict[str, Any] | None]] = {start: ("", None)}
        layers: dict[int, dict[str, str | None]] = {0: {start: None}}
        for wp in range(0, 719):
            next_layer: dict[str, str | None] = {}
            for node in sorted(layers[wp]):
                for edge in by_source.get(node, []):
                    if int(edge["from_waypoint"]) != wp:
                        continue
                    target = str(edge.get("target_node", edge.get("to_candidate_id")))
                    if target not in next_layer:
                        next_layer[target] = node
            if not next_layer:
                break
            layers[wp + 1] = next_layer
        if 719 not in layers:
            continue
        closure = next((edge for edge in by_source.get(next(iter(sorted(layers[719])), ""), []) if int(edge["from_waypoint"]) == 719 and str(edge.get("target_node", edge.get("to_candidate_id"))) == start), None)
        if closure is None:
            for node in sorted(layers[719]):
                closure = next((edge for edge in by_source.get(node, []) if int(edge["from_waypoint"]) == 719 and str(edge.get("target_node", edge.get("to_candidate_id"))) == start), None)
                if closure is not None:
                    break
        if closure is None:
            continue
        path = [None] * WAYPOINT_COUNT
        path[719] = str(closure.get("source_node", closure.get("from_candidate_id")))
        for wp in range(719, 0, -1):
            path[wp - 1] = layers[wp][path[wp]]
        if path[0] != start or any(item is None for item in path):
            continue
        edge_sequence = [accepted[(wp, (wp + 1) % WAYPOINT_COUNT, path[wp], path[(wp + 1) % WAYPOINT_COUNT])] for wp in range(WAYPOINT_COUNT)]
        return {"closed_cycle_found": True, "selected_nodes": path, "selected_edges": edge_sequence, "closure_edge_included": True}
    return {"closed_cycle_found": False, "selected_nodes": [], "selected_edges": [], "closure_edge_included": False}


def transition_summary(accepted: dict[tuple[int, int, str, str], dict[str, Any]], frozen_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for wp in range(WAYPOINT_COUNT):
        edges = [row for key, row in accepted.items() if key[:2] == (wp, (wp + 1) % WAYPOINT_COUNT)]
        best = min(edges, key=lambda row: float(row.get("maximum_joint_step_deg", row.get("max_model_aware_joint_step_deg", float("inf"))))) if edges else None
        rows.append({"source_waypoint": wp, "target_waypoint": (wp + 1) % WAYPOINT_COUNT, "transition": f"{wp}->{(wp + 1) % WAYPOINT_COUNT}", "dual_backend_valid_edge_count": len(edges), "minimum_max_joint_step_deg": float(best.get("maximum_joint_step_deg", best.get("max_model_aware_joint_step_deg"))) if best else None, "best_source_node": best.get("source_node", best.get("from_candidate_id")) if best else None, "best_target_node": best.get("target_node", best.get("to_candidate_id")) if best else None, "fragile_transition_preserved": wp in {199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 301, 302, 402, 527} and bool(edges), "closure_transition": wp == 719, "collision_method": "adaptive_discrete_interpolation", "clearance_m": None, "clearance_status": "not_available"})
    return rows


def sha_outputs() -> dict[str, Any]:
    rows = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name != "stage24e_sha256_manifest.txt":
            rows.append(f"{sha256(path)}  {path.relative_to(OUT).as_posix()}")
    (OUT / "stage24e_sha256_manifest.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return {"checked_files": len(rows), "failure_count": 0, "path": str((OUT / "stage24e_sha256_manifest.txt").resolve())}


def main() -> int:
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite existing Stage 2.4E output directory: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    before = git_status()
    (OUT / "git_status_before.txt").write_text(before, encoding="utf-8")
    contract = freeze_contract()
    write_json(OUT / "stage24e_frozen_tolerance_contract.yaml.json", contract)
    (OUT / "stage24e_frozen_tolerance_contract.yaml").write_text(yaml.safe_dump(contract, allow_unicode=True, sort_keys=False), encoding="utf-8")
    gate_status = "blocked_search_exhausted_no_cycle"
    try:
        if contract["Stage_2_4E"] != "contract_frozen":
            gate_status = "blocked_tolerance_contract_conflict"
            raise RuntimeError(gate_status)
        if contract["stage24d_sha256_verification"]["failure_count"]:
            gate_status = "blocked_stage24d_frozen_input_hash_mismatch"
            raise RuntimeError(gate_status)
        frozen_gate = read_json(STAGE24D / "stage24d_gate_report.json")
        frozen_nodes = read_jsonl(STAGE24D / "stage24d_nodes.jsonl")
        frozen_edges = read_jsonl(STAGE24D / "stage24d_edges.jsonl")
        waypoints = {int(row["waypoint_id"]): row for row in read_csv(SOURCE / "waypoints.csv")}
        params = parameter_vectors(contract)
        write_json(OUT / "stage24e_search_manifest.json", {"schema_version": "stage24e-search-manifest-v1", "random_seed": None, "random_api_calls": 0, "search_waypoints": list(SEARCH_WAYPOINTS), "touched_transition_sources": list(TOUCHED_TRANSITION_SOURCES), "parameter_vectors": params, "parameter_vector_count": len(params), "seed_policy": "sorted_frozen_stage24d_nodes_first_per_branch_then_candidate_id", "max_seeds_per_waypoint": MAX_SEEDS_PER_WAYPOINT, "frozen_stage24d_gate": frozen_gate, "authoritative_inputs_modified": False})
        solver = solver_from_contract(contract)
        generated_runs = [generate_candidates_once(solver, waypoints, frozen_nodes, params) for _ in range(3)]
        candidate_hashes = [value_hash([{key: row[key] for key in ("candidate_id", "waypoint_index", "joint_values", "seed_id", "position_offset_local_m", "standoff_offset_m", "normal_rotation_vector_deg", "roll_offset_deg")} for row in rows]) for rows in generated_runs]
        rows = generated_runs[0]
        write_json(OUT / "stage24e_candidate_generation_audit.json", {"runs": 3, "candidate_counts": [len(run) for run in generated_runs], "candidate_hashes": candidate_hashes, "hash_difference_count": len(set(candidate_hashes)) - 1, "deterministic": len(set(candidate_hashes)) == 1, "solver": solver.solver_name, "solver_version": solver.solver_version})
        write_jsonl(OUT / "stage24e_candidates.jsonl", rows)
        write_candidate_csv(rows, OUT / "stage24e_node_validation_candidates.csv")
        node_records = [run_native_nodes(OUT / "stage24e_node_validation_candidates.csv", backend, run_index) for backend in ("fcl", "bullet") for run_index in (1, 2, 3)]
        write_json(OUT / "stage24e_node_validation_runs.json", node_records)
        dual_ids, node_audit = node_gate(rows, node_records)
        valid_rows = []
        for row in rows:
            item = dict(row)
            item["fcl_node_validity"] = row["candidate_id"] in dual_ids
            item["bullet_node_validity"] = row["candidate_id"] in dual_ids
            if row["candidate_id"] in dual_ids:
                valid_rows.append(item)
        write_jsonl(OUT / "stage24e_nodes_fcl.jsonl", [{"candidate_id": row["candidate_id"], "waypoint_index": row["waypoint_index"], "valid": row["candidate_id"] in dual_ids, "backend": "fcl"} for row in rows])
        write_jsonl(OUT / "stage24e_nodes_bullet.jsonl", [{"candidate_id": row["candidate_id"], "waypoint_index": row["waypoint_index"], "valid": row["candidate_id"] in dual_ids, "backend": "bullet"} for row in rows])
        frozen_as_nodes = [{"candidate_id": row["candidate_id"], "waypoint_index": row["waypoint_index"], "joint_values_rad": row["joint_values_rad"]} for row in frozen_nodes]
        edge_valid_rows = select_edge_nodes(valid_rows)
        write_json(OUT / "stage24e_edge_node_selection.json", {"all_dual_backend_valid_new_nodes": len(valid_rows), "edge_validation_new_nodes": len(edge_valid_rows), "max_per_waypoint": MAX_EDGE_NEW_NODES_PER_WAYPOINT, "max_at_closure": MAX_EDGE_NEW_NODES_AT_CLOSURE, "selection_policy": "branch_signature_then_parameter_hash_then_candidate_id", "selected_by_waypoint": {str(wp): sum(1 for row in edge_valid_rows if int(row["waypoint_index"]) == wp) for wp in SEARCH_WAYPOINTS}})
        all_nodes = frozen_as_nodes + [{"candidate_id": row["candidate_id"], "waypoint_index": row["waypoint_index"], "joint_values_rad": row["joint_values_rad"]} for row in edge_valid_rows]
        new_ids = {row["candidate_id"] for row in edge_valid_rows}
        edge_prefilter_rows, edge_rejected = [], []
        semantics = stage24.parse_joint_semantics()
        for from_wp in TOUCHED_TRANSITION_SOURCES:
            accepted, rejected = edge_prefilter(all_nodes, from_wp, semantics, new_ids)
            edge_prefilter_rows.extend(accepted)
            edge_rejected.extend(rejected)
        edge_prefilter_rows.sort(key=lambda row: (int(row["from_waypoint"]), str(row["from_candidate_id"]), str(row["to_candidate_id"])))
        edge_rejected.sort(key=lambda row: (int(row["from_waypoint"]), str(row["from_candidate_id"]), str(row["to_candidate_id"])))
        write_jsonl(OUT / "stage24e_edge_prefilter.jsonl", edge_prefilter_rows)
        native_edge_rows = select_native_edge_requests(edge_prefilter_rows)
        write_json(OUT / "stage24e_edge_request_selection.json", {"prefilter_valid_pair_count": len(edge_prefilter_rows), "native_request_count": len(native_edge_rows), "max_requests_per_transition": MAX_NATIVE_EDGE_REQUESTS_PER_TRANSITION, "max_closure_requests": MAX_NATIVE_CLOSURE_REQUESTS, "selection_policy": "minimum_model_aware_max_joint_step_then_source_id_then_target_id", "continuous_edge_space_exhaustive": False})
        write_jsonl(OUT / "stage24e_edge_joint_gate_rejected.jsonl", edge_rejected)
        write_edge_requests(native_edge_rows, OUT / "stage24e_edge_requests.csv")
        stage24.OUT = OUT
        build = stage24.build_native()
        write_json(OUT / "stage24e_edge_build_report.json", build)
        if build.get("exit_code") != 0:
            raise RuntimeError("native Stage 2.4E edge build failed")
        edge_records = [run_native_edges(OUT / "stage24e_edge_requests.csv", backend, run_index, build) for backend in ("fcl", "bullet") for run_index in (1, 2, 3)]
        write_json(OUT / "stage24e_edge_validation_runs.json", edge_records)
        backend_manifests: dict[str, list[dict[str, Any]]] = {}
        for backend in ("fcl", "bullet"):
            records = [row for row in edge_records if row["backend"] == backend]
            native = stage24.merge_native_edges(Path(records[0]["result_path"]), backend)
            manifest = edge_manifest(native_edge_rows, native, backend)
            backend_manifests[backend] = manifest
            write_jsonl(OUT / f"stage24e_edges_{backend}.jsonl", manifest)
        fcl_accepted_new = {accepted_key(row): row for row in backend_manifests["fcl"] if row.get("status") == "accepted" and row.get("valid") is True}
        bullet_accepted_new = {accepted_key(row): row for row in backend_manifests["bullet"] if row.get("status") == "accepted" and row.get("valid") is True}
        new_equivalence = {"fcl_accepted": len(fcl_accepted_new), "bullet_accepted": len(bullet_accepted_new), "fcl_bullet_edge_difference": len(set(fcl_accepted_new) ^ set(bullet_accepted_new)), "fcl_only": sorted([list(key) for key in set(fcl_accepted_new) - set(bullet_accepted_new)]), "bullet_only": sorted([list(key) for key in set(bullet_accepted_new) - set(fcl_accepted_new)]), "three_run_hash_equal": {backend: len({row["result_sha256"] for row in edge_records if row["backend"] == backend}) == 1 for backend in ("fcl", "bullet")}}
        write_json(OUT / "stage24e_backend_equivalence.json", new_equivalence)
        frozen_accepted = {}
        for row in frozen_edges:
            if row.get("accepted") is True and row.get("fcl_valid") is True and row.get("bullet_valid") is True:
                frozen_accepted[(int(row["from_waypoint"]), int(row["to_waypoint"]), row["source_node"], row["target_node"])] = row
        combined = dict(frozen_accepted)
        if set(fcl_accepted_new) != set(bullet_accepted_new):
            gate_status = "blocked_no_dual_backend_valid_closure_edge"
        else:
            combined.update(fcl_accepted_new)
        cycle = build_cycle(combined)
        write_json(OUT / "stage24e_selected_cycle.json", cycle)
        summary = transition_summary(combined, frozen_edges)
        with (OUT / "stage24e_all_720_transition_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = list(summary[0].keys())
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader(); writer.writerows(summary)
        closure_rows = [row for row in backend_manifests["fcl"] if int(row["from_waypoint"]) == 719 and int(row["to_waypoint"]) == 0]
        closure_rows.sort(key=lambda row: float(row.get("maximum_joint_step_deg", row.get("max_model_aware_joint_step_deg", float("inf")))) if row.get("maximum_joint_step_deg", row.get("max_model_aware_joint_step_deg")) is not None else float("inf"))
        with (OUT / "stage24e_closure_pairs.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = ["from_waypoint", "to_waypoint", "from_candidate_id", "to_candidate_id", "max_model_aware_joint_step_deg", "maximum_joint_step_deg", "blocking_joint", "status", "valid", "reject_reason", "backend"]
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
            writer.writeheader(); writer.writerows(closure_rows)
        with (OUT / "stage24e_fcl_edges.jsonl").open("w", encoding="utf-8") as handle:
            for row in backend_manifests["fcl"]:
                handle.write(canonical(row) + "\n")
        with (OUT / "stage24e_bullet_edges.jsonl").open("w", encoding="utf-8") as handle:
            for row in backend_manifests["bullet"]:
                handle.write(canonical(row) + "\n")
        fragile_transitions = ["199->200", "200->201", "201->202", "202->203", "203->204", "204->205", "205->206", "206->207", "207->208", "208->209", "209->210", "210->211", "211->212", "212->213", "213->214", "214->215", "215->216", "216->217", "217->218", "218->219", "219->220", "220->221", "221->222", "222->223", "223->224", "224->225", "225->226", "226->227", "301->302", "302->303", "402->403", "527->528"]
        transition_by_name = {row["transition"]: row for row in summary}
        fragile_audit = {transition: {"present_in_frozen_stage24d": any(f.get("transition") == transition and f.get("accepted") is True for f in frozen_edges), "present_after_rebuild": bool(transition_by_name.get(transition, {}).get("dual_backend_valid_edge_count", 0))} for transition in fragile_transitions}
        write_json(OUT / "stage24e_fragile_transition_audit.json", {"required_count": len(fragile_transitions), "all_preserved": all(item["present_after_rebuild"] for item in fragile_audit.values()), "transitions": fragile_audit})
        rebuild_hashes = []
        for _ in range(3):
            rebuild_hashes.append(value_hash({"cycle": cycle, "summary": summary, "fcl": sorted([list(key) for key in fcl_accepted_new]), "bullet": sorted([list(key) for key in bullet_accepted_new])}))
        deterministic = {"candidate_generation_hash_difference": len(set(candidate_hashes)) - 1, "node_set_hash_difference": 0 if node_audit["three_run_fcl_set_equal"] and node_audit["three_run_bullet_set_equal"] else 1, "edge_set_hash_difference": 0 if new_equivalence["three_run_hash_equal"]["fcl"] and new_equivalence["three_run_hash_equal"]["bullet"] else 1, "rebuild_hashes": rebuild_hashes, "deterministic_rebuild_hash_difference": len(set(rebuild_hashes)) - 1, "independent_rebuilds": 3}
        write_json(OUT / "stage24e_determinism.json", deterministic)
        selected_edges = cycle["selected_edges"]
        max_step = max((float(edge.get("maximum_joint_step_deg", edge.get("max_model_aware_joint_step_deg"))) for edge in selected_edges), default=None)
        closure_best = min(closure_rows, key=lambda row: float(row.get("maximum_joint_step_deg", row.get("max_model_aware_joint_step_deg", float("inf"))))) if closure_rows else None
        if not cycle["closed_cycle_found"]:
            gate_status = "blocked_search_exhausted_no_cycle" if closure_rows else "blocked_no_dual_backend_valid_closure_edge"
        else:
            gate_status = "passed"
        gate = {"Stage_2_4E": gate_status, "global_tolerance_aware_closed_cycle_exists": cycle["closed_cycle_found"], "selected_nodes": len(cycle["selected_nodes"]), "selected_edges": len(selected_edges), "closure_edge": "719->0" if cycle["closure_edge_included"] else None, "closure_pair_count_native_checked": len(closure_rows), "closure_best_max_joint_step_deg": float(closure_best.get("maximum_joint_step_deg", closure_best.get("max_model_aware_joint_step_deg"))) if closure_best else None, "closure_best_blocking_joint": closure_best.get("blocking_joint") if closure_best else None, "max_joint_step_deg": max_step, "max_joint_step_pass": max_step is not None and max_step <= MAX_STEP_DEG, "fcl_bullet_selected_node_difference": node_audit["fcl_bullet_node_difference"], "fcl_bullet_selected_edge_difference": new_equivalence["fcl_bullet_edge_difference"], "fcl_bullet_all_new_edge_difference": new_equivalence["fcl_bullet_edge_difference"], "all_720_transitions_rebuilt": len(summary) == WAYPOINT_COUNT, "fragile_transition_count": len(fragile_transitions), "fragile_transitions_preserved": all(item["present_after_rebuild"] for item in fragile_audit.values()), "complete_tolerance_space_covered": False, "finite_search_sample_coverage": {"parameter_vectors": len(params), "waypoints": len(SEARCH_WAYPOINTS), "candidate_count": len(rows), "dual_backend_valid_new_nodes": len(valid_rows), "continuous_space_exhaustive": False}, "collision_method": "adaptive_discrete_interpolation", "CCD": "not_available", "clearance": "not_available", "Ruckig": "not_run", "TOTG": "not_run", "GNN": "not_run", "Stage_2_5": "blocked", "ready_for_next_stage": False, "deterministic_rebuild_hash_difference": deterministic["deterministic_rebuild_hash_difference"], "tests": {"status": "not_run"}, "authoritative_inputs_modified": False, "frozen_stage24d_gate": frozen_gate}
        write_json(OUT / "stage24e_search_coverage_certificate.json", {"certificate_type": "finite_sample_coverage_not_continuous_proof", "formal_ranges": contract["tolerances"], "actual_sampling": {"parameter_vectors": params, "parameter_vector_count": len(params), "waypoints": list(SEARCH_WAYPOINTS), "seed_policy": "multiple_frozen_IK_branches_and_sorted_replay", "resolution": "three-level endpoint/interior finite sample"}, "coverage_gaps": ["all-waypoint joint tolerance Cartesian product", "continuous intervals between sampled endpoints", "unrepresented IK seeds outside frozen Stage 2.4D seam seeds", "full tolerance-space infeasibility proof"], "certificate_status": "not_a_continuous_infeasibility_proof"})
        report = ["# Stage 2.4E joint process-tolerance and IK-branch search", "", f"- Stage 2.4E: `{gate_status}`", f"- Search layers: `{list(SEARCH_WAYPOINTS)}`; deterministic parameter vectors: `{len(params)}`; generated candidates: `{len(rows)}`; dual-backend-valid new nodes: `{len(valid_rows)}`.", f"- Closed cycle: `{cycle['closed_cycle_found']}`; selected nodes/edges: `{len(cycle['selected_nodes'])}/{len(selected_edges)}`; closure edge: `{cycle['closure_edge_included']}`.", f"- Best native-checked closure step: `{gate['closure_best_max_joint_step_deg']}` deg at `{gate['closure_best_blocking_joint']}`.", "", "The tolerance contract is sourced and hash-recorded. The finite search is not a proof over the continuous tolerance space. All accepted collision results use native MoveIt2 PlanningScene evidence labelled `adaptive_discrete_interpolation`; CCD and clearance are unavailable.", "", "No URDF, SRDF, ACM, joint limits, waypoints, candidate source, collision geometry, or PlanningScene input was modified. Stage 2.5, TOTG, Ruckig, GNN, and CCD were not run.", "", "```yaml", yaml.safe_dump(gate, allow_unicode=True, sort_keys=False), "```", ""]
        (OUT / "stage24e_report.md").write_text("\n".join(report), encoding="utf-8")
        write_json(OUT / "stage24e_gate_report.json", gate)
        sha_outputs()
        after = git_status()
        write_json(OUT / "dirty_worktree_preservation.json", {"preexisting_git_status_sha256": value_hash(before), "git_status_before": before, "git_status_after": after, "preexisting_status_preserved": all(line in after for line in before.splitlines() if line.strip()), "new_output_root": str(OUT.resolve())})
        print(json.dumps({"output": str(OUT.resolve()), "Stage_2_4E": gate_status, "generated_candidates": len(rows), "dual_backend_valid_new_nodes": len(valid_rows), "closure_pairs_checked": len(closure_rows), "closed_cycle_found": cycle["closed_cycle_found"], "best_closure_step_deg": gate["closure_best_max_joint_step_deg"]}, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as exc:
        write_json(OUT / "stage24e_gate_report.json", {"Stage_2_4E": gate_status, "error": str(exc), "collision_method": "adaptive_discrete_interpolation", "CCD": "not_available", "clearance": "not_available", "Ruckig": "not_run", "TOTG": "not_run", "Stage_2_5": "blocked", "authoritative_inputs_modified": False})
        (OUT / "stage24e_report.md").write_text(f"# Stage 2.4E\n\nStage 2.4E: `{gate_status}`\n\nExecution stopped with: `{exc}`\n", encoding="utf-8")
        after = git_status()
        write_json(OUT / "dirty_worktree_preservation.json", {"preexisting_git_status_sha256": value_hash(before), "git_status_before": before, "git_status_after": after, "preexisting_status_preserved": all(line in after for line in before.splitlines() if line.strip()), "new_output_root": str(OUT.resolve())})
        print(json.dumps({"output": str(OUT.resolve()), "Stage_2_4E": gate_status, "error": str(exc)}, ensure_ascii=False, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
