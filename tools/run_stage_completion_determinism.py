#!/usr/bin/env python3
"""Run the required deterministic numerical-IK repeat matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.deterministic_numeric_ik import DeterministicNumericIKSolver


TARGETS = (67, 68, 87, 94)
URDF = ROOT / "outputs/ik_graph_stage193/expanded_runtime_urdf.urdf"
LIMITS = ROOT / "ros2_moveit_bridge/config/joint_limits_with_jerk.yaml"
POSES = ROOT / "outputs/internal_wiper_moveit_inputs/open_arch_tcp_poses_base_link.csv"
SEEDS = ROOT / "outputs/internal_wiper_moveit_inputs/open_arch_seed_joints.csv"


def load_rows() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    import csv

    with POSES.open(newline="", encoding="utf-8") as handle:
        poses = list(csv.DictReader(handle))
    with SEEDS.open(newline="", encoding="utf-8") as handle:
        seeds = list(csv.DictReader(handle))
    return poses, seeds


def target_and_seed(index: int, poses: list[dict[str, str]], seeds: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    row = poses[index]
    target = np.eye(4)
    target[:3, :3] = Rotation.from_quat([float(row[k]) for k in ("qx", "qy", "qz", "qw")]).as_matrix()
    target[:3, 3] = [float(row[k]) for k in ("x", "y", "z")]
    seed = np.asarray([float(seeds[index][f"q{i}"]) for i in range(1, 7)], dtype=float)
    return target, seed


def canonical(result) -> str:
    payload = {"success": bool(result.success), "q": [float(x) for x in result.q_rad], "position_error_m": float(result.position_error_m), "tool_z_error_deg": float(result.tool_z_error_deg)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def one(index: int) -> dict[str, object]:
    poses, seeds = load_rows()
    target, seed = target_and_seed(index, poses, seeds)
    solver = DeterministicNumericIKSolver(URDF, LIMITS, continuity_weight=0.001)
    result = solver.solve(target, seed)
    return {"target_index": index, "success": bool(result.success), "solution_hash": canonical(result), "q": [float(x) for x in result.q_rad], "position_error_m": float(result.position_error_m), "tool_z_error_deg": float(result.tool_z_error_deg)}


def summarize(policy: str, index: int, records: list[dict[str, object]], repeat: int) -> dict[str, object]:
    statuses = [bool(r["success"]) for r in records]
    hashes = sorted({str(r["solution_hash"]) for r in records})
    return {"policy": policy, "target": f"waypoint_{index}", "repeat": repeat, "success_count": sum(statuses), "status_values": sorted(set(statuses)), "status_equal": len(set(statuses)) == 1, "canonical_solution_set_hashes": hashes, "canonical_solution_set_hash_equal": len(hashes) == 1}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/ik_graph_stage_completion/determinism_matrix.csv")
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--target-index", type=int)
    args = parser.parse_args()
    if args.single:
        if args.target_index is None:
            raise SystemExit("--target-index is required with --single")
        print(json.dumps(one(args.target_index), sort_keys=True))
        return 0

    rows: list[dict[str, object]] = []
    matrix: list[dict[str, object]] = []
    for index in TARGETS:
        poses, seeds = load_rows()
        target, seed = target_and_seed(index, poses, seeds)
        solver = DeterministicNumericIKSolver(URDF, LIMITS, continuity_weight=0.001)
        same = [({"success": bool((r := solver.solve(target, seed)).success), "solution_hash": canonical(r)}) for _ in range(100)]
        matrix.append(summarize("same_instance", index, same, 100))
        new = []
        for _ in range(50):
            fresh = DeterministicNumericIKSolver(URDF, LIMITS, continuity_weight=0.001)
            r = fresh.solve(target, seed)
            new.append({"success": bool(r.success), "solution_hash": canonical(r)})
        matrix.append(summarize("new_instance", index, new, 50))
        independent = []
        for _ in range(20):
            proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--single", "--target-index", str(index)], cwd=str(ROOT), capture_output=True, text=True, check=True)
            independent.append(json.loads(proc.stdout.strip()))
        matrix.append(summarize("independent_process", index, independent, 20))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["policy", "target", "repeat", "success_count", "status_values", "status_equal", "canonical_solution_set_hashes", "canonical_solution_set_hash_equal"]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in matrix:
            row = dict(row)
            row["status_values"] = json.dumps(row["status_values"], separators=(",", ":"))
            row["canonical_solution_set_hashes"] = json.dumps(row["canonical_solution_set_hashes"], separators=(",", ":"))
            writer.writerow(row)
    summary = {"schema_version": "1.0", "solver": DeterministicNumericIKSolver.solver_name, "solver_version": DeterministicNumericIKSolver.solver_version, "targets": [f"waypoint_{x}" for x in TARGETS], "hidden_random_api_calls": 0, "same_instance_status_equal": all(x["status_equal"] for x in matrix if x["policy"] == "same_instance"), "new_instance_status_equal": all(x["status_equal"] for x in matrix if x["policy"] == "new_instance"), "independent_process_status_equal": all(x["status_equal"] for x in matrix if x["policy"] == "independent_process"), "canonical_solution_set_hash_equal": all(x["canonical_solution_set_hash_equal"] for x in matrix), "rows": matrix}
    args.output.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
