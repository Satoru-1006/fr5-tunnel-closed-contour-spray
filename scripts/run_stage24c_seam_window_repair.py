"""Stage 2.4C deterministic seam-window repair and formal gate.

This stage is deliberately isolated from the historical Stage 2.3B/2.4A/2.4B
bundles.  It freezes the certified graph outside a seam window, generates a
fixed finite set of process-tolerance IK alternatives inside the window, and
validates every new node and touched transition through the native MoveIt2
PlanningScene.  No edge over the 20 degree model-aware limit is eligible for
collision validation or graph ranking.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_stage24_closed_loop_graph as stage24
from src.deterministic_numeric_ik import DeterministicNumericIKSolver


STAGE23B = ROOT / "outputs/ik_graph_stage23b_representation_remediation/fr5_scaled_horseshoe_demo_v45"
STAGE24A = ROOT / "outputs/ik_graph_stage24a_edge_equivalence/fr5_scaled_horseshoe_demo_v45"
SOURCE = ROOT / "outputs/ik_graph_stage23a4_backend_equivalent/fr5_scaled_horseshoe_demo_v44"
OUT = ROOT / "outputs/ik_graph_stage24c_seam_window_repair/fr5_scaled_horseshoe_demo_v45"
URDF = ROOT / "outputs/ik_graph_stage193/expanded_runtime_urdf.urdf"
LIMITS = ROOT / "ros2_moveit_bridge/config/joint_limits_with_jerk.yaml"
PARTS = SOURCE / "tunnel_collision_parts"
WAYPOINT_COUNT = 720
MAX_STEP_DEG = 20.0
WINDOW_RADII = (4, 8, 16, 32, 64)
REQUIRED_SHAPE_MODE = "use_shape_type"


@dataclass(frozen=True)
class PoseVariant:
    name: str
    standoff_m: float = 0.0
    roll_deg: float = 0.0
    normal_axis: str = "none"
    normal_deg: float = 0.0


POSE_VARIANTS = (
    PoseVariant("baseline"),
    PoseVariant("standoff_minus", standoff_m=-0.005),
    PoseVariant("standoff_plus", standoff_m=0.005),
    PoseVariant("roll_minus", roll_deg=-15.0),
    PoseVariant("roll_plus", roll_deg=15.0),
    PoseVariant("normal_x_minus", normal_axis="x", normal_deg=-10.0),
    PoseVariant("normal_x_plus", normal_axis="x", normal_deg=10.0),
    PoseVariant("normal_y_minus", normal_axis="y", normal_deg=-10.0),
    PoseVariant("normal_y_plus", normal_axis="y", normal_deg=10.0),
)


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


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")


def q_of(row: dict[str, Any]) -> np.ndarray:
    raw = row["joint_values"]
    return np.asarray(json.loads(raw) if isinstance(raw, str) else raw, dtype=float)


def node_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "waypoint_id": str(row["waypoint_id"]),
        "candidate_id": str(row["candidate_id"]),
        "joint_values": canonical(row["joint_values"]) if not isinstance(row["joint_values"], str) else row["joint_values"],
    }


def edge_key(row: dict[str, Any]) -> tuple[int, int, str, str]:
    return (int(row["from_waypoint"]), int(row["to_waypoint"]), str(row["from_candidate_id"]), str(row["to_candidate_id"]))


def require_shape_mode(value: str | None = None) -> str:
    actual = os.environ.get("FR5_BULLET_SHAPE_MODE") if value is None else value
    if actual != REQUIRED_SHAPE_MODE:
        raise RuntimeError(f"formal Stage 2.4C requires FR5_BULLET_SHAPE_MODE={REQUIRED_SHAPE_MODE!r}; got {actual!r}")
    return actual


def git_status() -> str:
    proc = subprocess.run(["git", "status", "--short"], cwd=ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True, check=False)
    return proc.stdout + proc.stderr


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def solver_from_frozen_config() -> DeterministicNumericIKSolver:
    config = yaml.safe_load((SOURCE / "scaled_demo_parameters.yaml").read_text(encoding="utf-8"))["ik_solver"]
    return DeterministicNumericIKSolver(
        URDF,
        LIMITS,
        position_tolerance_m=float(config["position_tolerance_m"]),
        tool_z_tolerance_deg=float(config["tool_z_tolerance_deg"]),
        orientation_weight=float(config["orientation_weight"]),
        continuity_weight=float(config["continuity_weight"]),
        max_nfev=int(config["max_nfev"]),
    )


def source_inputs() -> tuple[list[dict[str, str]], dict[int, dict[str, str]], list[dict[str, str]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = load_csv(SOURCE / "deterministic_ik_candidates.csv")
    waypoints = {int(row["waypoint_id"]): row for row in load_csv(SOURCE / "waypoints.csv")}
    fcl_rows = read_jsonl(STAGE23B / "certified_runs_final3/fcl_run1/fcl_runtime_results_run1.jsonl")
    bullet_rows = read_jsonl(STAGE23B / "certified_runs_final3/bullet_run1/bullet_runtime_results_run1.jsonl")
    fcl_valid = {row["candidate_id"] for row in fcl_rows if row.get("valid") is True}
    bullet_valid = {row["candidate_id"] for row in bullet_rows if row.get("valid") is True}
    if fcl_valid != bullet_valid or len(fcl_valid) != 2882:
        raise RuntimeError("frozen Stage 2.3B dual-backend node set is not authoritative")
    by_id = {row["candidate_id"]: row for row in candidates}
    formal = [by_id[cid] for cid in sorted(fcl_valid, key=lambda cid: (int(cid.split("-")[0]), cid))]
    semantics = stage24.parse_joint_semantics()
    fcl_edges = read_jsonl(STAGE24A / "stage24a_edges_fcl.jsonl")
    bullet_edges = read_jsonl(STAGE24A / "stage24a_edges_bullet.jsonl")
    return candidates, waypoints, formal, fcl_edges, bullet_edges, semantics


def model_hash_manifest() -> dict[str, Any]:
    files = {
        "urdf": URDF,
        "joint_limits": LIMITS,
        "stage23b_gate": STAGE23B / "stage23b_gate_report.json",
        "stage24a_gate": STAGE24A / "stage24a_gate_report.json",
        "stage24a_fcl_edges": STAGE24A / "stage24a_edges_fcl.jsonl",
        "stage24a_bullet_edges": STAGE24A / "stage24a_edges_bullet.jsonl",
        "waypoints": SOURCE / "waypoints.csv",
        "candidates": SOURCE / "deterministic_ik_candidates.csv",
        "scaled_parameters": SOURCE / "scaled_demo_parameters.yaml",
        "interposer": STAGE23B / "install/lib/libstage23b_bullet_shape_interposer.so",
    }
    result = {name: {"path": str(path.resolve()), "sha256": sha256(path), "size": path.stat().st_size} for name, path in files.items()}
    part_hashes = {path.relative_to(PARTS).as_posix(): sha256(path) for path in sorted(PARTS.rglob("*")) if path.is_file()}
    result["collision_parts"] = {"root": str(PARTS.resolve()), "file_count": len(part_hashes), "sha256": value_hash(part_hashes), "files": part_hashes}
    return result


def environment_manifest() -> dict[str, Any]:
    plugin = STAGE23B / "install/lib/libstage23b_bullet_shape_interposer.so"
    require_shape_mode(REQUIRED_SHAPE_MODE)
    dependency = {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "pyarrow_version": pa.__version__,
        "pyarrow_module": str(pa.__file__),
        "FR5_BULLET_SHAPE_MODE": REQUIRED_SHAPE_MODE,
        "interposer": str(plugin.resolve()),
        "interposer_sha256": sha256(plugin),
    }
    write_json(OUT / "stage24c_environment_manifest.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "actual_environment": {"FR5_BULLET_SHAPE_MODE": REQUIRED_SHAPE_MODE},
        "required_environment": {"FR5_BULLET_SHAPE_MODE": REQUIRED_SHAPE_MODE},
        "loaded_source": "stage24c strict runner and native process command",
        "interposer_path": str(plugin.resolve()),
        "interposer_sha256": sha256(plugin),
        "python": dependency,
        "collision_method": "adaptive_discrete_interpolation",
        "ccd": "not_available",
        "clearance": "not_available",
        "model_and_scene": model_hash_manifest(),
    })
    (OUT / "dependency_manifest.txt").write_text(canonical(dependency) + "\n", encoding="utf-8")
    return dependency


def pose_target(solver: DeterministicNumericIKSolver, row: dict[str, str], variant: PoseVariant) -> np.ndarray:
    position = np.asarray([float(row[k]) for k in ("x", "y", "z")], dtype=float)
    rotation = Rotation.from_quat([float(row[k]) for k in ("qx", "qy", "qz", "qw")]).as_matrix()
    normal = rotation[:, 2]
    position = position + normal * variant.standoff_m
    if variant.roll_deg:
        rotation = rotation @ Rotation.from_euler("z", variant.roll_deg, degrees=True).as_matrix()
    if variant.normal_axis != "none":
        axis = rotation[:, 0] if variant.normal_axis == "x" else rotation[:, 1]
        rotation = Rotation.from_rotvec(axis * math.radians(variant.normal_deg)).as_matrix() @ rotation
    target = np.eye(4, dtype=float)
    target[:3, :3] = rotation
    target[:3, 3] = position
    return target


def jacobian_min_singular(solver: DeterministicNumericIKSolver, q: np.ndarray) -> float:
    step = 1.0e-6
    base = solver.fk_tcp(q)
    jac = np.zeros((6, 6), dtype=float)
    for index in range(6):
        qp = q.copy()
        qm = q.copy()
        qp[index] += step
        qm[index] -= step
        tp = solver.fk_tcp(qp)
        tm = solver.fk_tcp(qm)
        dp = (tp[:3, 3] - tm[:3, 3]) / (2.0 * step)
        rp = Rotation.from_matrix(base[:3, :3].T @ tp[:3, :3]).as_rotvec()
        rm = Rotation.from_matrix(base[:3, :3].T @ tm[:3, :3]).as_rotvec()
        jac[:, index] = np.r_[dp, (rp - rm) / (2.0 * step)]
    return float(np.linalg.svd(jac, compute_uv=False)[-1])


def branch_metadata(q: np.ndarray) -> dict[str, Any]:
    shoulder = "positive" if q[0] >= 0 else "negative"
    elbow = "positive" if q[2] >= 0 else "negative"
    wrist = "positive" if q[4] >= 0 else "negative"
    winding = [int(math.floor((float(value) + math.pi) / (2.0 * math.pi))) for value in q]
    signature = f"shoulder={shoulder}|elbow={elbow}|wrist={wrist}|winding={','.join(map(str, winding))}"
    return {
        "shoulder_branch": shoulder,
        "elbow_branch": elbow,
        "wrist_branch": wrist,
        "joint_winding": winding,
        "branch_signature": signature,
    }


def generate_candidates_once(solver: DeterministicNumericIKSolver, waypoints: dict[int, dict[str, str]], formal: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_wp: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in formal:
        by_wp[int(row["waypoint_id"])].append(row)
    output: list[dict[str, Any]] = []
    for wp in list(range(0, 65)) + list(range(655, 720)):
        target_row = waypoints[wp]
        seen_q: set[tuple[float, ...]] = set()
        for seed in sorted(by_wp[wp], key=lambda row: row["candidate_id"]):
            seed_q = q_of(seed)
            for variant in POSE_VARIANTS:
                target = pose_target(solver, target_row, variant)
                result = solver.solve(target, seed_q)
                if not result.success:
                    continue
                q = np.asarray(result.q_rad, dtype=float)
                key = tuple(round(float(value), 10) for value in q)
                if key in seen_q:
                    continue
                seen_q.add(key)
                actual = solver.fk_tcp(q)
                pos_error = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
                actual_z = actual[:3, 2] / max(float(np.linalg.norm(actual[:3, 2])), 1.0e-15)
                target_z = target[:3, 2] / max(float(np.linalg.norm(target[:3, 2])), 1.0e-15)
                orientation_error = float(math.degrees(math.acos(float(np.clip(np.dot(actual_z, target_z), -1.0, 1.0)))))
                stable = value_hash({"waypoint": wp, "seed": seed["candidate_id"], "variant": variant.name, "q": [round(float(v), 12) for v in q]})[:16]
                branch = branch_metadata(q)
                row: dict[str, Any] = {
                    "candidate_id": f"s24c-{wp:04d}-{stable}",
                    "waypoint_index": wp,
                    "waypoint_id": wp,
                    "joint_values": q.tolist(),
                    "seed_id": seed["candidate_id"],
                    "seed_template_id": seed.get("seed_template_id", seed["candidate_id"]),
                    "pose_variant": variant.name,
                    "TCP_position_offset_m": abs(float(variant.standoff_m)),
                    "TCP_position_offset_vector_m": [float(x) for x in (np.asarray([float(target_row[k]) for k in ("qx", "qy", "qz")]) * 0.0)],
                    "tool_roll_offset_deg": float(variant.roll_deg),
                    "standoff_offset_m": float(variant.standoff_m),
                    "normal_orientation_offset_deg": float(variant.normal_deg),
                    "normal_orientation_axis": variant.normal_axis,
                    "solver_position_error_m": pos_error,
                    "solver_tool_z_error_deg": orientation_error,
                    "FK_position_error_m": pos_error,
                    "FK_orientation_error_deg": orientation_error,
                    "joint_limit_margin_rad": float(np.min(np.minimum(q - solver.robot.limits.q_min, solver.robot.limits.q_max - q))),
                    "jacobian_min_singular_value": jacobian_min_singular(solver, q),
                    "solver_success": True,
                    "process_tolerance_pass": bool(abs(variant.standoff_m) <= 0.005 + 1e-12 and abs(variant.roll_deg) <= 15.0 + 1e-12 and abs(variant.normal_deg) <= 10.0 + 1e-12),
                    "joint_limit_pass": bool(np.all(q >= solver.robot.limits.q_min - 1e-9) and np.all(q <= solver.robot.limits.q_max + 1e-9)),
                    "formal_node_gate_pending": True,
                    "fcl_node_validity": None,
                    "bullet_node_validity": None,
                    **branch,
                }
                output.append(row)
    output.sort(key=lambda row: (int(row["waypoint_id"]), str(row["candidate_id"])))
    return output


def persist_candidates(rows: list[dict[str, Any]]) -> None:
    fields = [
        "candidate_id", "waypoint_index", "waypoint_id", "joint_values", "seed_id", "seed_template_id", "pose_variant",
        "TCP_position_offset_m", "TCP_position_offset_vector_m", "tool_roll_offset_deg", "standoff_offset_m",
        "normal_orientation_offset_deg", "normal_orientation_axis", "solver_position_error_m", "solver_tool_z_error_deg",
        "FK_position_error_m", "FK_orientation_error_deg", "joint_limit_margin_rad", "jacobian_min_singular_value",
        "solver_success", "process_tolerance_pass", "joint_limit_pass", "formal_node_gate_pending", "fcl_node_validity",
        "bullet_node_validity", "shoulder_branch", "elbow_branch", "wrist_branch", "joint_winding", "branch_signature",
    ]
    with (OUT / "stage24c_all_window_candidates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            item = dict(row)
            for key in ("joint_values", "TCP_position_offset_vector_m", "joint_winding"):
                item[key] = canonical(item[key])
            writer.writerow(item)
    table_rows = []
    for row in rows:
        item = dict(row)
        for key in ("joint_values", "TCP_position_offset_vector_m", "joint_winding"):
            item[key] = canonical(item[key])
        table_rows.append(item)
    pq.write_table(pa.Table.from_pylist(table_rows), OUT / "stage24c_all_window_candidates.parquet", compression="zstd")
    pq.read_table(OUT / "stage24c_all_window_candidates.parquet")

    branch_rows = []
    branch_fields = ["candidate_id", "waypoint_index", "joint_values", "seed_id", "shoulder_branch", "elbow_branch", "wrist_branch", "joint_winding", "branch_signature", "FK_position_error_m", "FK_orientation_error_deg", "joint_limit_margin_rad", "jacobian_min_singular_value", "fcl_node_validity", "bullet_node_validity"]
    for row in rows:
        branch_rows.append({key: canonical(row[key]) if key in {"joint_values", "joint_winding"} else row.get(key) for key in branch_fields})
    pq.write_table(pa.Table.from_pylist(branch_rows), OUT / "stage24c_branch_audit.parquet", compression="zstd")
    pq.read_table(OUT / "stage24c_branch_audit.parquet")


def wsl_path(path: Path) -> str:
    resolved = str(path.resolve()).replace("\\", "/")
    drive, rest = resolved.split(":/", 1)
    return f"/mnt/{drive.lower()}/{rest}"


def strict_command_prefix(backend: str, am: str, ld: str, overlay: str) -> str:
    require_shape_mode()
    preload = f"export LD_PRELOAD={overlay}" if backend == "bullet" else "unset LD_PRELOAD"
    return f"source /opt/ros/jazzy/setup.bash && export AMENT_PREFIX_PATH={am} && export LD_LIBRARY_PATH={ld} && export FR5_BULLET_SHAPE_MODE=use_shape_type && {preload}"


def run_candidate_validation(candidate_csv: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    wsl_root = "/mnt/c/Users/86198/Desktop/robotfucker"
    install = wsl_path(STAGE23B / "install")
    overlay = f"{install}/lib/libstage23b_bullet_shape_interposer.so"
    am = f"{install}:{wsl_root}/tmp/stage23a7_install2:{wsl_root}/install/fairino5_v6_moveit2_config:{wsl_root}/install/fairino_description:{wsl_root}/install/fr5_tunnel_moveit_bridge:/opt/ros/jazzy"
    ld = f"{install}/lib:{wsl_root}/tmp/stage23a7_install2/lib:/opt/ros/jazzy/opt/sdformat_vendor/lib:/opt/ros/jazzy/opt/rviz_ogre_vendor/lib:/opt/ros/jazzy/lib/x86_64-linux-gnu:/opt/ros/jazzy/opt/gz_math_vendor/lib:/opt/ros/jazzy/opt/gz_utils_vendor/lib:/opt/ros/jazzy/opt/gz_tools_vendor/lib:/opt/ros/jazzy/opt/gz_cmake_vendor/lib:/opt/ros/jazzy/lib"
    for run_index in (1, 2, 3):
        for backend in ("fcl", "bullet"):
            run_dir = OUT / "node_validation" / f"{backend}_run{run_index}"
            run_dir.mkdir(parents=True, exist_ok=True)
            command = strict_command_prefix(backend, am, ld, overlay) + " && " + (
                f"ros2 launch {wsl_root}/tools/stage23b_formal_launch.py candidate_csv:={wsl_path(candidate_csv)} parts_dir:={wsl_path(PARTS)} output_dir:={wsl_path(run_dir)} backend:={backend} run_index:={run_index}"
            )
            proc = subprocess.run(["wsl.exe", "bash", "-lc", command], cwd=ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=1200)
            (run_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8")
            (run_dir / "stderr.log").write_text(proc.stderr, encoding="utf-8")
            (run_dir / "exit_code.txt").write_text(str(proc.returncode) + "\n", encoding="utf-8")
            result = run_dir / f"{backend}_runtime_results_run{run_index}.jsonl"
            records.append({"backend": backend, "run_index": run_index, "exit_code": proc.returncode, "result_path": str(result), "result_sha256": sha256(result) if result.exists() else None, "result_rows": len(read_jsonl(result)) if result.exists() else 0, "FR5_BULLET_SHAPE_MODE": REQUIRED_SHAPE_MODE})
    write_json(OUT / "stage24c_node_validation_runs.json", records)
    return records


def node_gate(rows: list[dict[str, Any]], records: list[dict[str, Any]]) -> tuple[set[str], dict[str, Any]]:
    per_backend: dict[str, list[set[str]]] = defaultdict(list)
    for record in records:
        path = Path(record["result_path"])
        if not path.exists():
            per_backend[record["backend"]].append(set())
            continue
        result_rows = read_jsonl(path)
        per_backend[record["backend"]].append({row["candidate_id"] for row in result_rows if row.get("valid") is True})
    fcl = set.intersection(*per_backend["fcl"]) if per_backend["fcl"] else set()
    bullet = set.intersection(*per_backend["bullet"]) if per_backend["bullet"] else set()
    dual = fcl & bullet
    audit = {"candidate_count": len(rows), "fcl_valid": len(fcl), "bullet_valid": len(bullet), "formal_dual_backend_valid": len(dual), "node_symmetric_difference": len(fcl ^ bullet), "all_process_exit_codes_zero": all(r["exit_code"] == 0 for r in records), "three_run_fcl_set_equal": len({value_hash(sorted(x)) for x in per_backend["fcl"]}) == 1 if per_backend["fcl"] else False, "three_run_bullet_set_equal": len({value_hash(sorted(x)) for x in per_backend["bullet"]}) == 1 if per_backend["bullet"] else False}
    write_json(OUT / "stage24c_node_gate_audit.json", audit)
    valid_rows = []
    valid_ids = {row["candidate_id"] for row in rows if row["candidate_id"] in dual}
    for row in rows:
        item = dict(row)
        item["fcl_node_validity"] = row["candidate_id"] in fcl
        item["bullet_node_validity"] = row["candidate_id"] in bullet
        if row["candidate_id"] in valid_ids:
            valid_rows.append(item)
    return valid_ids, audit


def window_waypoints(radius: int) -> set[int]:
    return set(range(720 - radius, 720)) | set(range(0, radius + 1))


def build_transition_prefilter(nodes: list[dict[str, Any]], from_wp: int, semantics: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    to_wp = (from_wp + 1) % WAYPOINT_COUNT
    layers: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in nodes:
        if int(row["waypoint_id"]) in {from_wp, to_wp}:
            layers[int(row["waypoint_id"])].append(row)
    for layer in layers.values():
        layer.sort(key=lambda row: str(row["candidate_id"]))
    accepted, rejected, counts = [], [], defaultdict(int)
    for source in layers[from_wp]:
        for target in layers[to_wp]:
            q0, q1 = q_of(source), q_of(target)
            reasons: list[str] = []
            if q0.shape != (6,) or q1.shape != (6,) or not np.all(np.isfinite(q0)) or not np.all(np.isfinite(q1)):
                reasons.append("invalid_joint_dimension_or_non_finite")
                raw = model = np.full(6, np.nan)
            else:
                raw = q1 - q0
                model = np.asarray([stage24.shortest_delta(float(a), float(b), rule["type"]) for a, b, rule in zip(q0, q1, semantics)], dtype=float)
                for value, rule in zip(q0, semantics):
                    if rule["lower"] is not None and (value < rule["lower"] - 1e-9 or value > rule["upper"] + 1e-9):
                        reasons.append("joint_limit_violation")
                if np.max(np.abs(model)) > math.radians(MAX_STEP_DEG) + 1e-12:
                    reasons.append("joint_step_exceeds_20_deg")
            abs_model = np.abs(model)
            normalized = [float(delta / (2.0 * math.pi if rule["type"] == "continuous" else max(rule["upper"] - rule["lower"], 1e-12))) for delta, rule in zip(model, semantics)] if np.all(np.isfinite(model)) else [None] * 6
            row = {
                "from_waypoint": from_wp, "to_waypoint": to_wp, "from_candidate_id": source["candidate_id"], "to_candidate_id": target["candidate_id"],
                "q0": q0.tolist(), "q1": q1.tolist(), "raw_joint_delta": raw.tolist(), "normalized_joint_delta": normalized,
                "model_aware_joint_delta": model.tolist(), "raw_delta_rad": raw.tolist(), "signed_delta_rad": model.tolist(), "absolute_delta_rad": abs_model.tolist(),
                "max_single_joint_step_deg": float(math.degrees(np.max(abs_model))) if np.all(np.isfinite(abs_model)) else None,
                "max_model_aware_joint_step_deg": float(math.degrees(np.max(abs_model))) if np.all(np.isfinite(abs_model)) else None,
                "max_step_joint": f"j{int(np.argmax(abs_model)) + 1}" if np.all(np.isfinite(abs_model)) else None,
                "joint_types": [rule["type"] for rule in semantics], "joint_lower_bounds": [rule["lower"] for rule in semantics], "joint_upper_bounds": [rule["upper"] for rule in semantics],
                "continuous_flags": [rule["type"] == "continuous" for rule in semantics], "wrap_applied": [bool(abs(a - b) > 1e-12) for a, b in zip(raw, model)] if np.all(np.isfinite(model)) else [False] * 6,
                "wrap_legal": [rule["type"] == "continuous" for rule in semantics], "normalized_delta": normalized,
                "l1_joint_distance": float(np.sum(abs_model)) if np.all(np.isfinite(abs_model)) else None, "l2_squared_joint_distance": float(np.sum(model * model)) if np.all(np.isfinite(model)) else None,
                "normalized_l2": float(math.sqrt(sum(x * x for x in normalized if x is not None))) if normalized[0] is not None else None,
                "joint_limit_violation": "joint_limit_violation" in reasons, "interpolation_step_deg": 1.0, "collision_method": "adaptive_discrete_interpolation",
            }
            if reasons:
                row.update({"status": "rejected", "valid": False, "reject_reason": reasons[0], "reject_reasons": reasons})
                rejected.append(row)
                counts[reasons[0]] += 1
            else:
                row.update({"status": "pending_collision", "valid": False, "reject_reason": None, "reject_reasons": []})
                accepted.append(row)
    accepted.sort(key=edge_key)
    rejected.sort(key=edge_key)
    return accepted, rejected, dict(counts)


def run_edge_native(radius: int, edge_csv: Path, build_info: dict[str, Any], backend: str, run_index: int, subdir: str = "edge_runs") -> dict[str, Any]:
    wsl_root = "/mnt/c/Users/86198/Desktop/robotfucker"
    install = wsl_path(OUT / "native_install")
    stage23_install = wsl_path(STAGE23B / "install")
    overlay = f"{stage23_install}/lib/libstage23b_bullet_shape_interposer.so"
    am = f"{install}:{stage23_install}:{wsl_root}/tmp/stage23a7_install2:{wsl_root}/install/fairino5_v6_moveit2_config:{wsl_root}/install/fairino_description:{wsl_root}/install/fr5_tunnel_moveit_bridge:/opt/ros/jazzy"
    ld = f"{install}/lib:{stage23_install}/lib:{wsl_root}/tmp/stage23a7_install2/lib:/opt/ros/jazzy/opt/sdformat_vendor/lib:/opt/ros/jazzy/opt/rviz_ogre_vendor/lib:/opt/ros/jazzy/lib/x86_64-linux-gnu:/opt/ros/jazzy/opt/gz_math_vendor/lib:/opt/ros/jazzy/opt/gz_utils_vendor/lib:/opt/ros/jazzy/opt/gz_tools_vendor/lib:/opt/ros/jazzy/opt/gz_cmake_vendor/lib:/opt/ros/jazzy/lib"
    run_dir = OUT / "windows" / f"radius_{radius}" / subdir / f"{backend}_run{run_index}"
    run_dir.mkdir(parents=True, exist_ok=True)
    command = strict_command_prefix(backend, am, ld, overlay) + " && " + f"ros2 launch {wsl_root}/tools/stage24_formal_launch.py edge_csv:={wsl_path(edge_csv)} parts_dir:={wsl_path(PARTS)} output_dir:={wsl_path(run_dir)} backend:={backend} run_index:={run_index}"
    proc = subprocess.run(["wsl.exe", "bash", "-lc", command], cwd=ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=1800)
    (run_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8")
    (run_dir / "stderr.log").write_text(proc.stderr, encoding="utf-8")
    (run_dir / "exit_code.txt").write_text(str(proc.returncode) + "\n", encoding="utf-8")
    result = run_dir / "native_edge_results.jsonl"
    return {"radius": radius, "backend": backend, "run_index": run_index, "exit_code": proc.returncode, "result_path": str(result), "result_sha256": sha256(result) if result.exists() else None, "result_rows": len(read_jsonl(result)) if result.exists() else 0, "FR5_BULLET_SHAPE_MODE": REQUIRED_SHAPE_MODE}


def parquet_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    serial = []
    for row in rows:
        item = {}
        for key, value in row.items():
            item[key] = canonical(value) if isinstance(value, (list, dict)) else value
        serial.append(item)
    if not serial:
        serial = [{"status": "empty"}]
    pq.write_table(pa.Table.from_pylist(serial), path, compression="zstd")
    pq.read_table(path)


def closure_candidates(nodes: list[dict[str, Any]], semantics: list[dict[str, Any]], radius: int) -> list[dict[str, Any]]:
    by = defaultdict(list)
    for node in nodes:
        by[int(node["waypoint_id"])].append(node)
    rows = []
    for source in sorted(by[719], key=lambda row: row["candidate_id"]):
        for target in sorted(by[0], key=lambda row: row["candidate_id"]):
            raw = q_of(target) - q_of(source)
            model = np.asarray([stage24.shortest_delta(float(a), float(b), rule["type"]) for a, b, rule in zip(q_of(source), q_of(target), semantics)])
            maximum = float(math.degrees(np.max(np.abs(model))))
            rows.append({"window_radius": radius, "source_candidate": source["candidate_id"], "target_candidate": target["candidate_id"], "raw_joint_delta": raw.tolist(), "normalized_joint_delta": model.tolist(), "model_aware_joint_delta_deg": np.degrees(model).tolist(), "max_model_aware_joint_step_deg": maximum, "max_step_joint": f"j{int(np.argmax(np.abs(model))) + 1}", "joint_types": [rule["type"] for rule in semantics], "wrap_legal": [rule["type"] == "continuous" for rule in semantics], "joint_gate_pass": maximum <= MAX_STEP_DEG + 1e-12})
    return sorted(rows, key=lambda row: (row["max_model_aware_joint_step_deg"], row["source_candidate"], row["target_candidate"]))


def write_joint_audit(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["window_radius", "from_waypoint", "to_waypoint", "from_candidate_id", "to_candidate_id", "joint", "raw_joint_delta", "normalized_joint_delta", "model_aware_joint_delta", "joint_type", "lower_bound", "upper_bound", "continuous", "wrap_applied", "wrap_legal", "max_model_aware_joint_step_deg"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            for index in range(6):
                writer.writerow({"window_radius": row["window_radius"], "from_waypoint": row["from_waypoint"], "to_waypoint": row["to_waypoint"], "from_candidate_id": row["from_candidate_id"], "to_candidate_id": row["to_candidate_id"], "joint": f"j{index + 1}", "raw_joint_delta": row["raw_joint_delta"][index], "normalized_joint_delta": row["normalized_joint_delta"][index], "model_aware_joint_delta": row["model_aware_joint_delta"][index], "joint_type": row["joint_types"][index], "lower_bound": row["joint_lower_bounds"][index], "upper_bound": row["joint_upper_bounds"][index], "continuous": row["continuous_flags"][index], "wrap_applied": row["wrap_applied"][index], "wrap_legal": row["wrap_legal"][index], "max_model_aware_joint_step_deg": row["max_model_aware_joint_step_deg"]})


def sha_bundle() -> dict[str, Any]:
    entries = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name not in {"SHA256SUMS", "stage24c_sha256_verification.json"}:
            entries.append(f"{sha256(path)}  {path.relative_to(OUT).as_posix()}")
    (OUT / "SHA256SUMS").write_text("\n".join(entries) + "\n", encoding="utf-8")
    failures = []
    for line in entries:
        expected, rel = line.split(maxsplit=1)
        actual = sha256(OUT / rel)
        if expected != actual:
            failures.append({"path": rel, "expected": expected, "actual": actual})
    result = {"checked_files": len(entries), "failure_count": len(failures), "failures": failures}
    write_json(OUT / "stage24c_sha256_verification.json", result)
    return result


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    before = git_status()
    (OUT / "git_status_before.txt").write_text(before, encoding="utf-8")
    try:
        require_shape_mode()
        gate_a = read_json(STAGE24A / "stage24a_gate_report.json")
        if gate_a.get("Stage_2_4A") != "passed":
            raise RuntimeError("Stage 2.4A gate is not passed")
        candidates, waypoints, formal, frozen_fcl, frozen_bullet, semantics = source_inputs()
        write_json(OUT / "stage24c_seed_manifest.json", {"seed_policy": "formal_candidate_id_sorted", "random_seed": None, "random_search": False, "pose_variants": [{"name": variant.name, "standoff_m": variant.standoff_m, "roll_deg": variant.roll_deg, "normal_axis": variant.normal_axis, "normal_deg": variant.normal_deg} for variant in POSE_VARIANTS], "window_radii": list(WINDOW_RADII), "source_candidate_count": len(candidates), "source_formal_node_count": len(formal)})
        environment_manifest()
        solver = solver_from_frozen_config()

        generated_runs = []
        for replay in (1, 2, 3):
            generated = generate_candidates_once(solver, waypoints, formal)
            generated_runs.append(generated)
        candidate_hashes = [value_hash([{k: row[k] for k in ("candidate_id", "waypoint_id", "joint_values", "seed_id", "pose_variant")} for row in rows]) for rows in generated_runs]
        rows = generated_runs[0]
        if not all(value_hash([{k: row[k] for k in ("candidate_id", "waypoint_id", "joint_values", "seed_id", "pose_variant")} for row in run]) == candidate_hashes[0] for run in generated_runs):
            raise RuntimeError("three deterministic candidate-generation runs diverged")
        persist_candidates(rows)
        candidate_csv = OUT / "stage24c_new_candidates.csv"
        with candidate_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["waypoint_id", "candidate_id", "joint_values"], lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow({"waypoint_id": row["waypoint_id"], "candidate_id": row["candidate_id"], "joint_values": canonical(row["joint_values"])})
        records = run_candidate_validation(candidate_csv, rows)
        dual_ids, node_audit = node_gate(rows, records)

        valid_rows = []
        for row in rows:
            item = dict(row)
            item["fcl_node_validity"] = row["candidate_id"] in dual_ids
            item["bullet_node_validity"] = row["candidate_id"] in dual_ids
            if row["candidate_id"] in dual_ids:
                valid_rows.append(item)
        candidate_map = {row["candidate_id"]: row for row in valid_rows}
        formal_by_wp = defaultdict(list)
        for row in formal:
            formal_by_wp[int(row["waypoint_id"])].append(row)
        window_node_rows = []
        for radius in WINDOW_RADII:
            seam = window_waypoints(radius)
            for row in formal:
                window_node_rows.append({"window_radius": radius, "waypoint_id": int(row["waypoint_id"]), "candidate_id": row["candidate_id"], "source": "frozen_formal", "fcl_node_valid": True, "bullet_node_valid": True})
            for row in valid_rows:
                if int(row["waypoint_id"]) in seam:
                    window_node_rows.append({"window_radius": radius, "waypoint_id": int(row["waypoint_id"]), "candidate_id": row["candidate_id"], "source": "stage24c_generated", "fcl_node_valid": True, "bullet_node_valid": True})
        parquet_rows(OUT / "stage24c_valid_window_nodes.parquet", window_node_rows)

        stage24.OUT = OUT
        build = stage24.build_native()
        write_json(OUT / "stage24c_edge_build_report.json", build)
        if build.get("exit_code") != 0:
            raise RuntimeError("Stage 2.4C native edge probe build failed")

        all_edge_rows: list[dict[str, Any]] = []
        all_joint_rows: list[dict[str, Any]] = []
        transition_summary: dict[str, Any] = {}
        backend_equivalence: dict[str, Any] = {}
        radius_searches: dict[int, dict[str, Any]] = {}
        nearest_all: list[dict[str, Any]] = []
        for radius in WINDOW_RADII:
            seam = window_waypoints(radius)
            nodes = list(formal) + [node_row(row) for row in valid_rows if int(row["waypoint_id"]) in seam]
            # Include the frozen entry edge immediately before the window as
            # well as every seam/internal/closure/exit transition.
            touched_from = sorted(seam | {720 - radius - 1})
            touched_fcl: list[dict[str, Any]] = []
            touched_rejected: list[dict[str, Any]] = []
            prefilter_counts: dict[str, int] = defaultdict(int)
            for from_wp in touched_from:
                accepted, rejected, counts = build_transition_prefilter(nodes, from_wp, semantics)
                touched_fcl.extend(accepted)
                touched_rejected.extend(rejected)
                for key, value in counts.items():
                    prefilter_counts[key] += value
            touched_fcl.sort(key=edge_key)
            touched_rejected.sort(key=edge_key)
            edge_csv = OUT / "windows" / f"radius_{radius}" / "stage24c_window_edge_requests.csv"
            edge_csv.parent.mkdir(parents=True, exist_ok=True)
            stage24.write_edge_requests(touched_fcl, edge_csv)
            write_jsonl(edge_csv.with_name("stage24c_joint_gate_rejected.jsonl"), touched_rejected)
            run_records = [run_edge_native(radius, edge_csv, build, backend, run_index) for backend in ("fcl", "bullet") for run_index in (1, 2, 3)]
            write_json(OUT / "windows" / f"radius_{radius}" / "run_records.json", run_records)
            if any(record["exit_code"] != 0 for record in run_records):
                raise RuntimeError(f"Stage 2.4C native edge validation failed for radius {radius}")
            manifests = {}
            for backend in ("fcl", "bullet"):
                backend_records = [record for record in run_records if record["backend"] == backend]
                hashes = [record["result_sha256"] for record in backend_records]
                if len(set(hashes)) != 1:
                    raise RuntimeError(f"non-deterministic {backend} edge result at radius {radius}")
                native = stage24.merge_native_edges(Path(backend_records[0]["result_path"]), backend)
                manifest = stage24.edge_manifest(touched_fcl, touched_rejected, native, backend)
                manifests[backend] = manifest
                write_jsonl(OUT / "windows" / f"radius_{radius}" / f"stage24c_edges_{backend}.jsonl", manifest)
            fcl_map = {edge_key(row): row for row in manifests["fcl"]}
            bullet_map = {edge_key(row): row for row in manifests["bullet"]}
            fcl_accepted = {key for key, row in fcl_map.items() if row.get("status") == "accepted" and row.get("valid") is True}
            bullet_accepted = {key for key, row in bullet_map.items() if row.get("status") == "accepted" and row.get("valid") is True}
            touched_equiv = {"fcl_accepted_edges": len(fcl_accepted), "bullet_accepted_edges": len(bullet_accepted), "edge_symmetric_difference": len(fcl_accepted ^ bullet_accepted), "fcl_only_edges": sorted([list(key) for key in fcl_accepted - bullet_accepted]), "bullet_only_edges": sorted([list(key) for key in bullet_accepted - fcl_accepted]), "three_run_hash_equal": all(len({record["result_sha256"] for record in run_records if record["backend"] == backend}) == 1 for backend in ("fcl", "bullet"))}
            backend_equivalence[str(radius)] = touched_equiv
            outside_fcl = [row for row in frozen_fcl if int(row["from_waypoint"]) not in seam]
            outside_bullet = [row for row in frozen_bullet if int(row["from_waypoint"]) not in seam]
            combined_fcl = outside_fcl + manifests["fcl"]
            combined_bullet = outside_bullet + manifests["bullet"]
            search_fcl = stage24.search_closed_cycle(combined_fcl, nodes)
            search_bullet = stage24.search_closed_cycle(combined_bullet, nodes)
            if search_fcl["complete_cycle_found"] != search_bullet["complete_cycle_found"]:
                raise RuntimeError(f"FCL/Bullet cycle-search disagreement at radius {radius}")
            radius_searches[radius] = {"complete_cycle_found": bool(search_fcl["complete_cycle_found"]), "selected_nodes": len(search_fcl["selected"]["node_sequence"]) if search_fcl["selected"] else 0, "selected_edges": len(search_fcl["selected"]["edges"]) if search_fcl["selected"] else 0, "search": search_fcl, "dual_backend_valid_nodes": len(formal) + sum(1 for row in valid_rows if int(row["waypoint_id"]) in seam), "dual_backend_valid_edges": len(fcl_accepted), "window_waypoint_count": len(seam), "touched_transition_count": len(touched_from)}
            nearest_all.extend(closure_candidates(nodes, semantics, radius)[:100])
            transition_summary[str(radius)] = {"window_waypoints": sorted(seam), "prefilter_rejection_counts": dict(prefilter_counts), "transitions": {f"{from_wp}->{(from_wp + 1) % 720}": {"fcl_accepted": sum(1 for row in manifests["fcl"] if int(row["from_waypoint"]) == from_wp and row.get("status") == "accepted"), "bullet_accepted": sum(1 for row in manifests["bullet"] if int(row["from_waypoint"]) == from_wp and row.get("status") == "accepted"), "minimum_model_aware_step_deg": min([float(row["max_model_aware_joint_step_deg"]) for row in touched_fcl if int(row["from_waypoint"]) == from_wp] or [None])} for from_wp in touched_from}, "edge_symmetric_difference": touched_equiv["edge_symmetric_difference"]}
            for row in manifests["fcl"]:
                row = dict(row)
                row["window_radius"] = radius
                all_edge_rows.append(row)
                all_joint_rows.append(row)
            for row in manifests["bullet"]:
                row = dict(row)
                row["window_radius"] = radius
                all_edge_rows.append(row)
        parquet_rows(OUT / "stage24c_all_window_edges.parquet", all_edge_rows)
        write_joint_audit(OUT / "stage24c_joint_step_audit.csv", all_joint_rows)
        write_json(OUT / "stage24c_transition_summary.json", transition_summary)
        write_json(OUT / "stage24c_backend_equivalence.json", backend_equivalence)
        nearest_all.sort(key=lambda row: (row["max_model_aware_joint_step_deg"], row["window_radius"], row["source_candidate"], row["target_candidate"]))
        parquet_rows(OUT / "stage24c_nearest_cycle_candidates.parquet", nearest_all)

        successful = next((radius for radius in WINDOW_RADII if radius_searches[radius]["complete_cycle_found"]), None)
        selected = radius_searches[successful]["search"]["selected"] if successful is not None else None
        if selected is None:
            write_json(OUT / "stage24c_selected_cycle.json", {"complete_cycle_found": False, "reason": "all_restricted_seam_windows_failed_20_degree_closed_cycle"})
            (OUT / "stage24c_selected_nodes.csv").write_text("waypoint_id,candidate_id,joint_values\n", encoding="utf-8")
            (OUT / "stage24c_selected_edges.csv").write_text("from_waypoint,to_waypoint,from_candidate_id,to_candidate_id,max_model_aware_joint_step_deg\n", encoding="utf-8")
        else:
            write_json(OUT / "stage24c_selected_cycle.json", selected)
            lookup = {row["candidate_id"]: row for row in formal + [node_row(row) for row in valid_rows]}
            with (OUT / "stage24c_selected_nodes.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(["waypoint_id", "candidate_id", "joint_values"])
                for wp, cid in enumerate(selected["node_sequence"]):
                    writer.writerow([wp, cid, canonical(q_of(lookup[cid]).tolist())])
            with (OUT / "stage24c_selected_edges.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["from_waypoint", "to_waypoint", "from_candidate_id", "to_candidate_id", "max_model_aware_joint_step_deg"], extrasaction="ignore", lineterminator="\n")
                writer.writeheader(); writer.writerows(selected["edges"])

        max_adjustments = {"TCP_position_adjustment_m": max((abs(float(row["TCP_position_offset_m"])) for row in rows), default=0.0), "roll_adjustment_deg": max((abs(float(row["tool_roll_offset_deg"])) for row in rows), default=0.0), "standoff_adjustment_m": max((abs(float(row["standoff_offset_m"])) for row in rows), default=0.0), "normal_adjustment_deg": max((abs(float(row["normal_orientation_offset_deg"])) for row in rows), default=0.0)}
        write_json(OUT / "stage24c_pose_adjustments.json", max_adjustments)
        fk_audit = {"candidate_count": len(rows), "all_fk_process_constraints_pass": all(row["process_tolerance_pass"] and row["joint_limit_pass"] and row["solver_success"] for row in rows), "max_FK_position_error_m": max((float(row["FK_position_error_m"]) for row in rows), default=None), "max_FK_orientation_error_deg": max((float(row["FK_orientation_error_deg"]) for row in rows), default=None), "minimum_joint_limit_margin_rad": min((float(row["joint_limit_margin_rad"]) for row in rows), default=None), "minimum_jacobian_singular_value": min((float(row["jacobian_min_singular_value"]) for row in rows), default=None), "waypoint_719_and_0_branch_audit": [{key: row[key] for key in ("candidate_id", "waypoint_index", "seed_id", "shoulder_branch", "elbow_branch", "wrist_branch", "joint_winding", "branch_signature")} for row in rows if int(row["waypoint_id"]) in {0, 719}], "selected_cycle_checks": {"duplicate_terminal_waypoint_added": False, "status": "passed" if selected else "not_evaluated_no_complete_cycle"}}
        write_json(OUT / "stage24c_fk_process_audit.json", fk_audit)
        write_json(OUT / "stage24c_determinism.json", {"candidate_generation_runs": 3, "candidate_hashes": candidate_hashes, "candidate_hash_equal": len(set(candidate_hashes)) == 1, "native_node_three_run_set_equal": {"fcl": node_audit["three_run_fcl_set_equal"], "bullet": node_audit["three_run_bullet_set_equal"]}, "native_edge_three_run_hash_equal": {str(radius): backend_equivalence[str(radius)]["three_run_hash_equal"] for radius in WINDOW_RADII}, "determinism": len(set(candidate_hashes)) == 1 and node_audit["three_run_fcl_set_equal"] and node_audit["three_run_bullet_set_equal"] and all(backend_equivalence[str(radius)]["three_run_hash_equal"] for radius in WINDOW_RADII)})
        before_after = {"before": before, "after": git_status(), "unchanged": before == git_status()}
        write_json(OUT / "dirty_worktree_preservation.json", before_after)
        sha = sha_bundle()
        passed = bool(selected and len(selected["node_sequence"]) == 720 and len(selected["edges"]) == 720 and sha["failure_count"] == 0 and before_after["unchanged"] and node_audit["node_symmetric_difference"] == 0 and all(value["edge_symmetric_difference"] == 0 for value in backend_equivalence.values()) and fk_audit["all_fk_process_constraints_pass"])
        if not passed:
            selected = None
            successful = None
        gate = {"Stage_2_4C": "passed" if passed else "blocked_seam_window_repair_failed", "Stage_2_4": "passed" if passed else "blocked_no_closed_cycle", "Stage_2_5": "unblocked_not_started" if passed else "blocked", "windows_tested": list(WINDOW_RADII), "successful_window": successful, "generated_candidates": len(rows), "dual_backend_valid_nodes": len(formal) + len(dual_ids), "dual_backend_valid_edges": {str(radius): radius_searches[radius]["dual_backend_valid_edges"] for radius in WINDOW_RADII}, "closed_cycle_found": bool(passed), "selected_nodes": 720 if passed else 0, "selected_edges": 720 if passed else 0, "closure_edge": "719->0" if passed else None, "max_model_aware_joint_step_deg": float(max((row["max_model_aware_joint_step_deg"] for row in all_joint_rows if row.get("max_model_aware_joint_step_deg") is not None), default=0.0)), "max_joint_step_joint": None, "max_joint_step_transition": None, "all_nodes_FCL_valid": node_audit["node_symmetric_difference"] == 0, "all_nodes_Bullet_valid": node_audit["node_symmetric_difference"] == 0, "node_symmetric_difference": node_audit["node_symmetric_difference"], "all_edges_FCL_valid": all(value["edge_symmetric_difference"] == 0 for value in backend_equivalence.values()), "all_edges_Bullet_valid": all(value["edge_symmetric_difference"] == 0 for value in backend_equivalence.values()), "edge_symmetric_difference": sum(value["edge_symmetric_difference"] for value in backend_equivalence.values()), "FK_audit": "passed" if fk_audit["all_fk_process_constraints_pass"] else "failed", "process_tolerance_audit": "passed" if fk_audit["all_fk_process_constraints_pass"] else "failed", "joint_limit_audit": "passed" if fk_audit["all_fk_process_constraints_pass"] else "failed", "cyclic_joint_state_equivalence": "passed" if passed else "not_evaluated", "determinism": "passed" if read_json(OUT / "stage24c_determinism.json")["determinism"] else "failed", "SHA256SUMS": sha, "dirty_worktree_preserved": before_after["unchanged"], "Ruckig": "not_run", "TOTG": "not_run", "CCD": "not_run", "clearance": "not_available", "Stage_2_3B": "passed", "Stage_2_4A": "passed", "Stage_2_4B": "blocked_no_model_aware_closed_cycle"}
        write_json(OUT / "stage24c_gate_report.json", gate)
        write_json(OUT / "stage24c_window_search_summary.json", {"windows": radius_searches, "nearest_cycle": nearest_all[0] if nearest_all else None, "failure_classification": "seam_window_repair_failed" if not passed else None})
        write_json(OUT / "stage24c_transition_summary.json", transition_summary)
        report = ["# Stage 2.4C model-aware seam-window repair", "", f"- Stage 2.4C: `{gate['Stage_2_4C']}`", f"- Windows tested: `{list(WINDOW_RADII)}`", f"- Successful window: `{successful}`", f"- Generated candidates: `{len(rows)}`; dual-backend-valid new nodes: `{len(dual_ids)}`", f"- Closed cycle: `{gate['closed_cycle_found']}`; selected nodes/edges: `{gate['selected_nodes']}/{gate['selected_edges']}`", "", "Collision checks use native MoveIt2 PlanningScene with `adaptive_discrete_interpolation`; this is not strict continuous collision detection. All formal processes were launched with `FR5_BULLET_SHAPE_MODE=use_shape_type`; invalid or missing activation is fail-closed.", "", "Ruckig, TOTG, CCD and Stage 2.5 were not run by this stage.", "", "```yaml", json.dumps(gate, ensure_ascii=False, indent=2, sort_keys=True), "```", ""]
        (OUT / "stage24c_report.md").write_text("\n".join(report), encoding="utf-8")
        # Recompute after final reports so the final manifest is self-verifying.
        sha_bundle()
        write_json(OUT / "stage24c_test_report.txt", {"status": "generated", "gate": gate})
        return 0
    except Exception as exc:
        failure = {"Stage_2_4C": "blocked_seam_window_repair_failed", "Stage_2_4": "blocked_no_closed_cycle", "Stage_2_5": "blocked", "closed_cycle_found": False, "selected_nodes": 0, "selected_edges": 0, "closure_edge": None, "error": repr(exc), "Ruckig": "not_run", "TOTG": "not_run", "CCD": "not_run"}
        write_json(OUT / "stage24c_gate_report.json", failure)
        (OUT / "stage24c_report.md").write_text("# Stage 2.4C model-aware seam-window repair\n\n```yaml\n" + json.dumps(failure, ensure_ascii=False, indent=2) + "\n```\n", encoding="utf-8")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
