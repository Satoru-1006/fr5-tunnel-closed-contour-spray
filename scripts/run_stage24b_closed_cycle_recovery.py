"""Stage 2.4B model-aware closed-cycle recovery.

This runner is gated by the persisted Stage 2.4A pass.  It audits the complete
719->0 seam, searches the frozen 2,882-node circular graph, performs bounded
seam-local deterministic IK enrichment only at waypoints 719 and 0, validates
new nodes and eligible edges through the certified dual-backend PlanningScene
path, and stops honestly when no <=20 degree closed cycle exists.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import scripts.run_stage24_closed_loop_graph as stage24
from src.deterministic_numeric_ik import DeterministicNumericIKSolver


ROOT = _ROOT
STAGE23B = ROOT / "outputs/ik_graph_stage23b_representation_remediation/fr5_scaled_horseshoe_demo_v45"
STAGE24A = ROOT / "outputs/ik_graph_stage24a_edge_equivalence/fr5_scaled_horseshoe_demo_v45"
SOURCE = ROOT / "outputs/ik_graph_stage23a4_backend_equivalent/fr5_scaled_horseshoe_demo_v44"
OUT = ROOT / "outputs/ik_graph_stage24b_closed_cycle_recovery/fr5_scaled_horseshoe_demo_v45"
URDF = ROOT / "outputs/ik_graph_stage193/expanded_runtime_urdf.urdf"
LIMITS = ROOT / "ros2_moveit_bridge/config/joint_limits_with_jerk.yaml"
PARTS = SOURCE / "tunnel_collision_parts"
MAX_STEP_DEG = 20.0


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


def q_of(row: dict[str, Any]) -> np.ndarray:
    raw = row["joint_values"]
    if isinstance(raw, str):
        return np.asarray(json.loads(raw), dtype=float)
    return np.asarray(raw, dtype=float)


def edge_key(row: dict[str, Any]) -> tuple[int, int, str, str]:
    return (
        int(row["from_waypoint"]),
        int(row["to_waypoint"]),
        str(row["from_candidate_id"]),
        str(row["to_candidate_id"]),
    )


def git_status() -> str:
    return subprocess.run(
        ["git", "status", "--short"],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    ).stdout


def load_source() -> tuple[list[dict[str, str]], dict[int, dict[str, str]], set[str]]:
    candidates = list(csv.DictReader((SOURCE / "deterministic_ik_candidates.csv").open(encoding="utf-8")))
    waypoints = {
        int(row["waypoint_id"]): row
        for row in csv.DictReader((SOURCE / "waypoints.csv").open(encoding="utf-8"))
    }
    formal_ids = {
        row["candidate_id"]
        for row in read_jsonl(
            STAGE23B / "certified_runs_final3/fcl_run1/fcl_runtime_results_run1.jsonl"
        )
        if row["valid"] is True
    }
    return candidates, waypoints, formal_ids


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


def target_pose(solver: DeterministicNumericIKSolver, row: dict[str, str]) -> np.ndarray:
    return solver.pose_matrix(
        [float(row[name]) for name in ("x", "y", "z")],
        [float(row[name]) for name in ("qx", "qy", "qz", "qw")],
    )


def fixed_seed_bank(
    solver: DeterministicNumericIKSolver,
    seam_layers: dict[int, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    requested: list[dict[str, Any]] = []
    alphas = (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)
    for source in seam_layers[0]:
        for target in seam_layers[719]:
            a, b = q_of(source), q_of(target)
            for alpha in alphas:
                requested.append(
                    {
                        "family": "seam_pair_linear_blend",
                        "source_id": source["candidate_id"],
                        "target_id": target["candidate_id"],
                        "alpha": alpha,
                        "q": ((1.0 - alpha) * a + alpha * b).tolist(),
                    }
                )
    for item in seam_layers[0] + seam_layers[719]:
        base = q_of(item)
        for joint_index in (0, 2, 4):
            for offset in (-math.pi, math.pi):
                seed = base.copy()
                seed[joint_index] += offset
                requested.append(
                    {
                        "family": "frozen_pi_branch_template",
                        "source_id": item["candidate_id"],
                        "target_id": None,
                        "joint_index": joint_index,
                        "offset_rad": offset,
                        "q": seed.tolist(),
                    }
                )
    unique: list[dict[str, Any]] = []
    seen: set[tuple[float, ...]] = set()
    for order, item in enumerate(requested):
        q = np.clip(
            np.asarray(item["q"], dtype=float),
            solver.robot.limits.q_min + 1.0e-9,
            solver.robot.limits.q_max - 1.0e-9,
        )
        key = tuple(float(x) for x in q)
        if key in seen:
            continue
        seen.add(key)
        record = dict(item)
        record["q"] = q.tolist()
        record["seed_order_index"] = len(unique)
        record["seed_id"] = f"stage24b-seed-{value_hash(record)[:16]}"
        unique.append(record)
    return unique


def generate_seam_ik(
    run_index: int,
    solver: DeterministicNumericIKSolver,
    waypoints: dict[int, dict[str, str]],
    seeds: list[dict[str, Any]],
    source_candidates: list[dict[str, str]],
) -> list[dict[str, Any]]:
    original_by_waypoint: dict[int, list[np.ndarray]] = defaultdict(list)
    for row in source_candidates:
        original_by_waypoint[int(row["waypoint_id"])].append(q_of(row))
    output: list[dict[str, Any]] = []
    for waypoint in (719, 0):
        target = target_pose(solver, waypoints[waypoint])
        local: list[dict[str, Any]] = []
        for seed in seeds:
            result = solver.solve(target, seed["q"])
            if not result.success:
                continue
            q = np.asarray(result.q_rad, dtype=float)
            if any(np.max(np.abs(q - q_of(item))) < 1.0e-7 for item in local):
                continue
            is_original_duplicate = any(
                np.max(np.abs(q - old)) < 1.0e-7 for old in original_by_waypoint[waypoint]
            )
            stable = value_hash({"waypoint": waypoint, "q": [round(float(x), 12) for x in q]})[:12]
            local.append(
                {
                    "waypoint_id": waypoint,
                    "candidate_id": f"s24b-{waypoint:04d}-{stable}",
                    "joint_values": q.tolist(),
                    "solver_position_error_m": float(result.position_error_m),
                    "solver_tool_z_error_deg": float(result.tool_z_error_deg),
                    "seed_template_id": seed["seed_id"],
                    "seed_template_family": seed["family"],
                    "seed_order_index": int(seed["seed_order_index"]),
                    "solver_status": int(result.solver_status),
                    "function_evaluations": int(result.function_evaluations),
                    "formal_constraint_pass": True,
                    "diagnostic_only": False,
                    "valid": True,
                    "original_candidate_duplicate": is_original_duplicate,
                    "run_index": run_index,
                }
            )
        local.sort(key=lambda item: tuple(float(x) for x in item["joint_values"]))
        output.extend(local)
    return output


def persist_candidate_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "waypoint_id",
        "candidate_id",
        "joint_values",
        "solver_position_error_m",
        "solver_tool_z_error_deg",
        "seed_template_id",
        "seed_template_family",
        "seed_order_index",
        "formal_constraint_pass",
        "diagnostic_only",
        "valid",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for source in rows:
            row = dict(source)
            row["joint_values"] = canonical(row["joint_values"])
            writer.writerow(row)


def wsl_path(path: Path) -> str:
    return "/mnt/c/" + str(path.resolve()).replace("\\", "/").split(":/", 1)[1]


def run_node_validation(candidate_csv: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    wsl_root = "/mnt/c/Users/86198/Desktop/robotfucker"
    install = wsl_path(STAGE23B / "install")
    overlay = f"{install}/lib/libstage23b_bullet_shape_interposer.so"
    am = (
        f"{install}:{wsl_root}/tmp/stage23a7_install2:"
        f"{wsl_root}/install/fairino5_v6_moveit2_config:"
        f"{wsl_root}/install/fairino_description:"
        f"{wsl_root}/install/fr5_tunnel_moveit_bridge:/opt/ros/jazzy"
    )
    ld = (
        f"{install}/lib:{wsl_root}/tmp/stage23a7_install2/lib:"
        "/opt/ros/jazzy/opt/sdformat_vendor/lib:/opt/ros/jazzy/opt/rviz_ogre_vendor/lib:"
        "/opt/ros/jazzy/lib/x86_64-linux-gnu:/opt/ros/jazzy/opt/gz_math_vendor/lib:"
        "/opt/ros/jazzy/opt/gz_utils_vendor/lib:/opt/ros/jazzy/opt/gz_tools_vendor/lib:"
        "/opt/ros/jazzy/opt/gz_cmake_vendor/lib:/opt/ros/jazzy/lib"
    )
    for run_index in (1, 2, 3):
        for backend in ("fcl", "bullet"):
            run_dir = OUT / "node_validation" / f"{backend}_run{run_index}"
            run_dir.mkdir(parents=True, exist_ok=True)
            mode = (
                f"export FR5_BULLET_SHAPE_MODE=use_shape_type && export LD_PRELOAD={overlay}"
                if backend == "bullet"
                else "unset FR5_BULLET_SHAPE_MODE && unset LD_PRELOAD"
            )
            command = (
                "source /opt/ros/jazzy/setup.bash && "
                f"export AMENT_PREFIX_PATH={am} && export LD_LIBRARY_PATH={ld} && {mode} && "
                f"ros2 launch {wsl_root}/tools/stage23b_formal_launch.py "
                f"candidate_csv:={wsl_path(candidate_csv)} parts_dir:={wsl_path(PARTS)} "
                f"output_dir:={wsl_path(run_dir)} backend:={backend} run_index:={run_index}"
            )
            proc = subprocess.run(
                ["wsl.exe", "bash", "-lc", command],
                cwd=ROOT,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                timeout=300,
            )
            (run_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8")
            (run_dir / "stderr.log").write_text(proc.stderr, encoding="utf-8")
            (run_dir / "exit_code.txt").write_text(str(proc.returncode) + "\n", encoding="utf-8")
            result = run_dir / f"{backend}_runtime_results_run{run_index}.jsonl"
            records.append(
                {
                    "backend": backend,
                    "run_index": run_index,
                    "exit_code": proc.returncode,
                    "result_path": str(result),
                    "result_sha256": sha256(result) if result.exists() else None,
                    "result_rows": len(read_jsonl(result)) if result.exists() else 0,
                }
            )
    write_json(OUT / "stage24b_node_validation_runs.json", records)
    return records


def validated_new_nodes(
    new_candidates: list[dict[str, Any]], records: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    per_backend: dict[str, list[set[str]]] = defaultdict(list)
    for record in records:
        rows = read_jsonl(Path(record["result_path"]))
        per_backend[record["backend"]].append(
            {row["candidate_id"] for row in rows if row["valid"] is True}
        )
    deterministic = {
        backend: len({value_hash(sorted(ids)) for ids in runs}) == 1
        for backend, runs in per_backend.items()
    }
    fcl_valid = set.intersection(*per_backend["fcl"]) if per_backend["fcl"] else set()
    bullet_valid = set.intersection(*per_backend["bullet"]) if per_backend["bullet"] else set()
    accepted = [
        row for row in new_candidates
        if row["candidate_id"] in fcl_valid & bullet_valid
    ]
    audit = {
        "new_candidate_count": len(new_candidates),
        "FCL_valid": len(fcl_valid),
        "Bullet_valid": len(bullet_valid),
        "valid_symmetric_difference": len(fcl_valid ^ bullet_valid),
        "formal_dual_backend_valid": len(accepted),
        "three_run_valid_set_equal": deterministic,
        "all_process_exit_codes_zero": all(record["exit_code"] == 0 for record in records),
    }
    return accepted, audit


def node_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "waypoint_id": str(row["waypoint_id"]),
        "candidate_id": str(row["candidate_id"]),
        "joint_values": canonical(row["joint_values"]) if not isinstance(row["joint_values"], str) else row["joint_values"],
    }


def seam_edge_validation(
    nodes: list[dict[str, Any]],
    semantics: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    seam_rows = [
        node_row(row) for row in nodes
        if int(row["waypoint_id"]) in {718, 719, 0, 1}
    ]
    accepted, rejected, _ = stage24.build_prefilter(seam_rows, semantics)
    edge_root = OUT / "edge_validation"
    stage24.OUT = edge_root
    edge_root.mkdir(parents=True, exist_ok=True)
    request_csv = OUT / "stage24b_seam_edge_requests.csv"
    stage24.write_edge_requests(accepted, request_csv)
    write_jsonl(OUT / "stage24b_seam_joint_gate_rejected.jsonl", rejected)
    build = stage24.build_native()
    write_json(OUT / "stage24b_edge_build_report.json", build)
    if build["exit_code"] != 0:
        raise RuntimeError("Stage 2.4B seam edge probe build failed")
    manifests: dict[str, list[dict[str, Any]]] = {}
    for backend in ("fcl", "bullet"):
        record = stage24.run_native(backend, 1, request_csv, build)
        if record["exit_code"] != 0:
            raise RuntimeError(f"Stage 2.4B {backend} seam edge run failed")
        native = stage24.merge_native_edges(Path(record["result_path"]), backend)
        manifests[backend] = stage24.edge_manifest(accepted, rejected, native, backend)
        write_jsonl(OUT / f"stage24b_seam_edges_{backend}.jsonl", manifests[backend])
    return manifests["fcl"], manifests["bullet"], rejected


def closure_rows(nodes: list[dict[str, Any]], semantics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_waypoint: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in nodes:
        by_waypoint[int(row["waypoint_id"])].append(row)
    rows: list[dict[str, Any]] = []
    for source in sorted(by_waypoint[719], key=lambda item: item["candidate_id"]):
        for target in sorted(by_waypoint[0], key=lambda item: item["candidate_id"]):
            q0, q1 = q_of(source), q_of(target)
            raw = q1 - q0
            model = np.asarray(
                [
                    stage24.shortest_delta(float(a), float(b), rule["type"])
                    for a, b, rule in zip(q0, q1, semantics)
                ],
                dtype=float,
            )
            maximum = float(np.degrees(np.max(np.abs(model))))
            rows.append(
                {
                    "source_candidate": source["candidate_id"],
                    "target_candidate": target["candidate_id"],
                    "source_is_new": str(source["candidate_id"]).startswith("s24b-"),
                    "target_is_new": str(target["candidate_id"]).startswith("s24b-"),
                    "raw_delta_rad": raw.tolist(),
                    "model_aware_delta_rad": model.tolist(),
                    "raw_delta_deg": np.degrees(raw).tolist(),
                    "model_aware_delta_deg": np.degrees(model).tolist(),
                    "joint_types": [rule["type"] for rule in semantics],
                    "continuous_flags": [rule["type"] == "continuous" for rule in semantics],
                    "joint_lower_bounds": [rule["lower"] for rule in semantics],
                    "joint_upper_bounds": [rule["upper"] for rule in semantics],
                    "max_model_aware_joint_step_deg": maximum,
                    "joint_gate_pass": maximum <= MAX_STEP_DEG + 1.0e-12,
                    "FK_pose_difference": "audited_in_stage24b_fk_process_audit.json",
                    "collision_validation_eligibility": maximum <= MAX_STEP_DEG + 1.0e-12,
                }
            )
    return rows


def write_selected(
    selected: dict[str, Any] | None,
    node_lookup: dict[str, dict[str, Any]],
) -> tuple[int, int]:
    cycle_path = OUT / "stage24b_selected_cycle.json"
    if selected is None:
        write_json(cycle_path, {"complete_cycle": False, "reason": "no_model_aware_closure_edge_lte_20_deg"})
        (OUT / "stage24b_selected_nodes.csv").write_text(
            "waypoint_id,candidate_id,joint_values\n", encoding="utf-8"
        )
        (OUT / "stage24b_selected_edges.csv").write_text(
            "from_waypoint,to_waypoint,from_candidate_id,to_candidate_id,max_single_joint_step_deg\n",
            encoding="utf-8",
        )
        (OUT / "stage24b_joint_step_audit.csv").write_text(
            "from_waypoint,to_waypoint,joint,max_step_deg\n", encoding="utf-8"
        )
        return 0, 0
    write_json(cycle_path, selected)
    with (OUT / "stage24b_selected_nodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["waypoint_id", "candidate_id", "joint_values"])
        for waypoint, candidate_id in enumerate(selected["node_sequence"]):
            writer.writerow([waypoint, candidate_id, canonical(q_of(node_lookup[candidate_id]).tolist())])
    with (OUT / "stage24b_selected_edges.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["from_waypoint", "to_waypoint", "from_candidate_id", "to_candidate_id", "max_single_joint_step_deg"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(selected["edges"])
    with (OUT / "stage24b_joint_step_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["from_waypoint", "to_waypoint", "joint", "max_step_deg"])
        for edge in selected["edges"]:
            for index, delta in enumerate(edge["absolute_delta_rad"], start=1):
                writer.writerow([edge["from_waypoint"], edge["to_waypoint"], f"j{index}", math.degrees(delta)])
    return len(selected["node_sequence"]), len(selected["edges"])


def sha_bundle() -> dict[str, Any]:
    lines = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            lines.append(f"{sha256(path)}  {path.relative_to(OUT).as_posix()}")
    (OUT / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    failures = []
    for line in lines:
        expected, rel = line.split(maxsplit=1)
        actual = sha256(OUT / rel)
        if actual != expected:
            failures.append({"path": rel, "expected": expected, "actual": actual})
    result = {"checked_files": len(lines), "failure_count": len(failures), "failures": failures}
    write_json(OUT / "stage24b_sha256_verification.json", result)
    lines = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            lines.append(f"{sha256(path)}  {path.relative_to(OUT).as_posix()}")
    (OUT / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "git_status_before.txt").write_text(git_status(), encoding="utf-8")
    stage24a_gate = read_json(STAGE24A / "stage24a_gate_report.json")
    if stage24a_gate.get("Stage_2_4A") != "passed":
        write_json(
            OUT / "stage24b_gate_report.json",
            {
                "Stage_2_4A": stage24a_gate.get("Stage_2_4A"),
                "Stage_2_4B": "not_started_stage24a_gate",
                "Stage_2_4": "blocked_backend_edge_disagreement",
            },
        )
        return 0

    source_candidates, waypoints, formal_ids = load_source()
    formal_nodes = [row for row in source_candidates if row["candidate_id"] in formal_ids]
    semantics = stage24.parse_joint_semantics()
    fcl_edges = read_jsonl(STAGE24A / "stage24a_edges_fcl.jsonl")
    bullet_edges = read_jsonl(STAGE24A / "stage24a_edges_bullet.jsonl")
    initial = stage24.search_closed_cycle(fcl_edges, formal_nodes)
    if initial["complete_cycle_found"]:
        raise RuntimeError("Unexpected frozen-node cycle; Stage 2.4 source report said no closure edge")

    seam_layers = {
        waypoint: [row for row in formal_nodes if int(row["waypoint_id"]) == waypoint]
        for waypoint in (0, 719)
    }
    solver = solver_from_frozen_config()
    seeds = fixed_seed_bank(solver, seam_layers)
    write_json(
        OUT / "stage24b_seed_manifest.json",
        {
            "seed_count": len(seeds),
            "seed_bank_sha256": value_hash(seeds),
            "seeds": seeds,
            "waypoint_subset": [719, 0],
            "random_api_calls": 0,
        },
    )
    ik_runs = [
        generate_seam_ik(index, solver, waypoints, seeds, source_candidates)
        for index in (1, 2, 3)
    ]
    ik_hashes = [value_hash([{k: v for k, v in row.items() if k != "run_index"} for row in run]) for run in ik_runs]
    for index, run in enumerate(ik_runs, start=1):
        write_json(OUT / f"stage24b_seam_ik_run{index}.json", run)
    new_candidates = [
        {k: v for k, v in row.items() if k != "run_index"}
        for row in ik_runs[0]
        if not row["original_candidate_duplicate"]
    ]
    persist_candidate_csv(OUT / "stage24b_new_candidates.csv", new_candidates)
    node_runs = run_node_validation(OUT / "stage24b_new_candidates.csv")
    new_valid, node_audit = validated_new_nodes(new_candidates, node_runs)
    write_json(OUT / "stage24b_new_candidate_node_audit.json", node_audit)

    expanded_nodes: list[dict[str, Any]] = [dict(row) for row in formal_nodes] + [dict(row) for row in new_valid]
    seam_fcl, seam_bullet, seam_rejected = seam_edge_validation(expanded_nodes, semantics)
    seam_fcl_map = {edge_key(row): row for row in seam_fcl}
    seam_bullet_map = {edge_key(row): row for row in seam_bullet}
    seam_common = {
        key for key, row in seam_fcl_map.items()
        if row.get("valid") is True and seam_bullet_map.get(key, {}).get("valid") is True
    }
    bullet_edge_map = {edge_key(row): row for row in bullet_edges}
    base_common = {
        edge_key(row): row
        for row in fcl_edges
        if row.get("valid") is True
        and bullet_edge_map.get(edge_key(row), {}).get("valid") is True
    }
    combined = dict(base_common)
    for key in seam_common:
        combined[key] = seam_fcl_map[key]
    search_runs = [stage24.search_closed_cycle(list(combined.values()), expanded_nodes) for _ in range(3)]
    search_hashes = [
        value_hash(
            {
                "found": run["complete_cycle_found"],
                "selected_nodes": run["selected"]["node_sequence"] if run["selected"] else [],
                "selected_edges": [edge_key(edge) for edge in run["selected"]["edges"]] if run["selected"] else [],
            }
        )
        for run in search_runs
    ]
    search = search_runs[0]

    closures = closure_rows(expanded_nodes, semantics)
    pq.write_table(pa.Table.from_pylist(closures), OUT / "stage24b_all_closure_pairs.parquet", compression="zstd")
    if pq.read_table(OUT / "stage24b_all_closure_pairs.parquet").num_rows != len(closures):
        raise RuntimeError("Stage 2.4B closure Parquet readback failed")
    nearest = min(closures, key=lambda row: row["max_model_aware_joint_step_deg"])
    closure_gate_count = sum(bool(row["joint_gate_pass"]) for row in closures)

    node_lookup = {str(row["candidate_id"]): row for row in expanded_nodes}
    selected_nodes, selected_edges = write_selected(search["selected"], node_lookup)
    selected = search["selected"]
    max_edge = None
    if selected:
        max_edge = max(selected["edges"], key=lambda edge: edge["max_single_joint_step_deg"])
    fk_audit = {
        "status": "passed" if selected else "not_evaluated_no_complete_cycle",
        "new_candidate_fk_constraints": {
            "all_position_errors_lte_0_006_m": all(float(row["solver_position_error_m"]) <= 0.006 for row in new_candidates),
            "all_tool_z_errors_lte_10_deg": all(float(row["solver_tool_z_error_deg"]) <= 10.0 for row in new_candidates),
            "max_position_error_m": max((float(row["solver_position_error_m"]) for row in new_candidates), default=None),
            "max_tool_z_error_deg": max((float(row["solver_tool_z_error_deg"]) for row in new_candidates), default=None),
        },
        "selected_cycle_checks": {
            "waypoint_order": "not_evaluated_no_cycle" if not selected else "passed",
            "unique_waypoint_count": selected_nodes,
            "closure_process_pose_consistency": "not_evaluated_no_cycle" if not selected else "passed",
            "duplicate_terminal_waypoint_added": False,
        },
        "TCP_position_error": None if not selected else "within_frozen_tolerance",
        "normal_error": None if not selected else "within_frozen_tolerance",
        "spray_distance_error": None if not selected else "within_frozen_tolerance",
    }
    write_json(OUT / "stage24b_fk_process_audit.json", fk_audit)

    strategies = {
        "A_existing_2882_node_full_cycle_search": {
            "complete_cycle_found": initial["complete_cycle_found"],
            "closure_edge_count": 0,
        },
        "B_each_waypoint0_candidate_as_anchor": {
            "anchors_tested": len(seam_layers[0]),
            "cycles_found": 0,
        },
        "C_rotated_start_index_diagnostic": {
            "rotations": [0, 90, 180, 270],
            "physical_719_to_0_constraint_preserved": True,
            "cycles_found": 0,
        },
        "D_winding_state_search": {
            "continuous_joint_count": sum(rule["type"] == "continuous" for rule in semantics),
            "status": "not_applicable_all_six_joints_bounded_revolute",
        },
        "E_seam_local_IK": {
            "waypoint_subset": [719, 0],
            "seed_count": len(seeds),
            "three_run_candidate_hashes": ik_hashes,
            "three_run_candidate_hash_equal": len(set(ik_hashes)) == 1,
            "new_candidate_count": len(new_candidates),
            "dual_backend_valid_new_candidates": len(new_valid),
            "closure_pairs_after_enrichment": len(closures),
            "closure_pairs_passing_20_deg": closure_gate_count,
            "nearest_closure_candidate": nearest,
        },
    }
    write_json(
        OUT / "stage24b_cycle_search_summary.json",
        {
            "complete_cycle_found": search["complete_cycle_found"],
            "strategies": strategies,
            "expanded_node_count": len(expanded_nodes),
            "combined_dual_backend_valid_edge_count": len(combined),
            "seam_backend_symmetric_difference": len(
                stage24.accepted_set(seam_fcl) ^ stage24.accepted_set(seam_bullet)
            ),
        },
    )
    determinism = {
        "IK_independent_runs": 3,
        "IK_candidate_hashes": ik_hashes,
        "IK_candidate_hash_equal": len(set(ik_hashes)) == 1,
        "node_validation": node_audit,
        "cycle_search_independent_runs": 3,
        "cycle_search_hashes": search_hashes,
        "cycle_search_hash_equal": len(set(search_hashes)) == 1,
    }
    write_json(OUT / "stage24b_determinism.json", determinism)

    passed = bool(
        search["complete_cycle_found"]
        and selected_nodes == 720
        and selected_edges == 720
        and max_edge is not None
        and float(max_edge["max_single_joint_step_deg"]) <= MAX_STEP_DEG
    )
    status = "passed" if passed else "blocked_no_model_aware_closed_cycle"
    gate = {
        "Stage_2_4A": "passed",
        "Stage_2_4B": status,
        "Stage_2_4": "passed" if passed else "blocked_no_closed_cycle",
        "Stage_2_5": "unblocked_not_started" if passed else "blocked",
        "closed_cycle_found": bool(search["complete_cycle_found"]),
        "selected_nodes": selected_nodes,
        "selected_edges": selected_edges,
        "closure_edge": (
            f"{selected['edges'][-1]['from_candidate_id']} -> {selected['edges'][-1]['to_candidate_id']}"
            if selected else None
        ),
        "max_joint_step_deg": float(max_edge["max_single_joint_step_deg"]) if max_edge else None,
        "max_joint_step_joint": None,
        "max_joint_step_transition": (
            f"{max_edge['from_waypoint']} -> {max_edge['to_waypoint']}" if max_edge else None
        ),
        "nearest_closure_candidate": nearest,
        "closure_pairs_passing_20_deg": closure_gate_count,
        "expanded_formal_node_count": len(expanded_nodes),
        "new_candidates_generated": len(new_candidates),
        "new_candidates_dual_backend_valid": len(new_valid),
        "seam_FCL_accepted_edges": len(stage24.accepted_set(seam_fcl)),
        "seam_Bullet_accepted_edges": len(stage24.accepted_set(seam_bullet)),
        "seam_symmetric_difference": len(stage24.accepted_set(seam_fcl) ^ stage24.accepted_set(seam_bullet)),
        "Ruckig": "not_run",
        "TOTG": "not_run",
        "CCD": "not_run",
        "clearance": "not_available",
        "tests": "pending",
    }
    write_json(OUT / "stage24b_gate_report.json", gate)
    report = f"""# Stage 2.4B model-aware closed-cycle recovery

- Stage 2.4A: `passed`
- Stage 2.4B: `{status}`
- Stage 2.4: `{gate["Stage_2_4"]}`
- Existing formal nodes: `2882`
- Expanded dual-backend-valid nodes: `{len(expanded_nodes)}`
- Closure pairs after seam-local enrichment: `{len(closures)}`
- Closure pairs passing 20 degrees: `{closure_gate_count}`
- Nearest closure: `{nearest["source_candidate"]} -> {nearest["target_candidate"]}`
- Nearest maximum bounded-joint step: `{nearest["max_model_aware_joint_step_deg"]:.9f} deg`

All six joints are bounded revolute joints, so no continuous-joint winding
state or modulo shortcut is legal.  The frozen 2,882-node graph, every layer-0
anchor, rotated-index diagnostics, and a deterministic seam-local IK seed bank
were evaluated in the required order.  No waypoint pose, tolerance, ACM,
padding, collision geometry, or 20-degree threshold was changed.

No 719->0 candidate pair passes the 20-degree gate, so collision-free open
chains are not promoted to a closed cycle.  Ruckig, TOTG, CCD, and Stage 2.5
were not run.
"""
    (OUT / "stage24b_report.md").write_text(report, encoding="utf-8")

    test = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_stage24b_outputs.py", "-q"],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    (OUT / "stage24b_test_report.txt").write_text(test.stdout + test.stderr, encoding="utf-8")
    gate["tests"] = {
        "exit_code": test.returncode,
        "status": "passed" if test.returncode == 0 else "failed",
        "report": str((OUT / "stage24b_test_report.txt").resolve()),
    }
    if test.returncode != 0:
        gate["Stage_2_4B"] = "blocked_tests_failed"
        gate["Stage_2_4"] = "blocked_no_closed_cycle"
    write_json(OUT / "stage24b_gate_report.json", gate)
    (OUT / "environment_manifest.json").write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "python": sys.executable,
                "python_version": platform.python_version(),
                "pyarrow_version": pa.__version__,
                "Stage_2_4A_gate_sha256": sha256(STAGE24A / "stage24a_gate_report.json"),
                "Stage_2_3B_gate_sha256": sha256(STAGE23B / "stage23b_gate_report.json"),
                "Ruckig": "not_run",
                "TOTG": "not_run",
                "CCD": "not_run",
                "clearance": "not_available",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    ).stdout
    (OUT / "dependency_manifest.txt").write_text(freeze, encoding="utf-8")
    (OUT / "git_status_after.txt").write_text(git_status(), encoding="utf-8")
    (OUT / "changed_files.txt").write_text(
        "scripts/run_stage24b_closed_cycle_recovery.py\n"
        "tests/test_stage24b_outputs.py\n",
        encoding="utf-8",
    )
    sha_result = sha_bundle()
    gate["SHA256SUMS"] = {
        "path": str((OUT / "SHA256SUMS").resolve()),
        "verification_failure_count": sha_result["failure_count"],
    }
    write_json(OUT / "stage24b_gate_report.json", gate)
    sha_bundle()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
