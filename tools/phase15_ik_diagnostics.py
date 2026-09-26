#!/usr/bin/env python3
"""Stage 1.5 offline diagnosis for the authoritative 181-point graph.

This command reads the existing Stage 1 Parquet/JSON evidence and writes only
diagnostic artifacts.  ROS-backed local candidate enrichment is implemented in
``phase15_local_candidate_enrichment.py``; keeping the offline calculations
here makes threshold scans and root-cause tests reproducible without ROS.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.phase15_diagnostics import (
    analyze_bridge,
    candidate_count_rows,
    reachable_frontier,
    threshold_sweep,
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except Exception as exc:
        raise RuntimeError(
            "pyarrow is required for Parquet diagnostics. Run this command in the ROS/WSL Python environment."
        ) from exc
    return parquet.read_table(path).to_pylist()


def _load_pose_metrics(path: Path, left: int = 67, right: int = 68) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    a = rows[left]
    b = rows[right]
    position_a = np.asarray([float(a[key]) for key in ("x", "y", "z")])
    position_b = np.asarray([float(b[key]) for key in ("x", "y", "z")])
    normal_a = np.asarray([float(a[key]) for key in ("nx", "ny", "nz")])
    normal_b = np.asarray([float(b[key]) for key in ("nx", "ny", "nz")])
    normal_a /= max(float(np.linalg.norm(normal_a)), 1e-12)
    normal_b /= max(float(np.linalg.norm(normal_b)), 1e-12)
    qa = np.asarray([float(a[key]) for key in ("qx", "qy", "qz", "qw")])
    qb = np.asarray([float(b[key]) for key in ("qx", "qy", "qz", "qw")])
    pose_delta = Rotation.from_quat(qa).inv() * Rotation.from_quat(qb)
    return {
        "source_waypoint": left,
        "target_waypoint": right,
        "source_position_m": position_a.tolist(),
        "target_position_m": position_b.tolist(),
        "tcp_target_position_delta_m": float(np.linalg.norm(position_b - position_a)),
        "source_normal": normal_a.tolist(),
        "target_normal": normal_b.tolist(),
        "tcp_target_normal_angle_deg": float(np.rad2deg(np.arccos(np.clip(normal_a @ normal_b, -1.0, 1.0)))),
        "raw_process_target_pose_source_quaternion_xyzw": qa.tolist(),
        "raw_process_target_pose_target_quaternion_xyzw": qb.tolist(),
        "raw_process_target_pose_rotation_delta_deg": float(np.rad2deg(pose_delta.magnitude())),
        "raw_process_target_pose_rotation_delta_rotvec_rad": pose_delta.as_rotvec().tolist(),
        "input_pose_or_normal_discontinuity": bool(
            np.rad2deg(pose_delta.magnitude()) > 45.0
            or np.rad2deg(np.arccos(np.clip(normal_a @ normal_b, -1.0, 1.0))) > 45.0
        ),
        "discontinuity_threshold_deg": 45.0,
        "orientation_note": (
            "The raw pose quaternion and normal are the process target. Candidate roll is a free TCP-axis "
            "reparameterization and is not itself an input normal change."
        ),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(_json_safe(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--nodes", type=Path, default=None)
    parser.add_argument("--edges", type=Path, default=None)
    parser.add_argument("--poses", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    nodes_path = args.nodes or root / "outputs/ik_graph/nodes.parquet"
    edges_path = args.edges or root / "outputs/ik_graph/edges.parquet"
    poses_path = args.poses or root / "outputs/internal_wiper_moveit_inputs/open_arch_tcp_poses_base_link.csv"
    out_dir = args.out_dir or root / "outputs/ik_graph_diagnostics"
    nodes = _read_parquet(nodes_path)
    edges = _read_parquet(edges_path)
    frontiers = reachable_frontier(nodes, edges, 181)
    bridge = analyze_bridge(nodes, edges, 67, 68, 181)
    pose_metrics = _load_pose_metrics(poses_path)
    bridge["tcp_target_pose_metrics"] = pose_metrics
    candidate_pose_variants: dict[str, list[dict[str, Any]]] = {}
    for waypoint in (67, 68):
        raw_quaternion = np.asarray(
            pose_metrics["raw_process_target_pose_source_quaternion_xyzw"]
            if waypoint == 67
            else pose_metrics["raw_process_target_pose_target_quaternion_xyzw"],
            dtype=float,
        )
        variants: list[dict[str, Any]] = []
        for node in [row for row in nodes if int(row["waypoint"]) == waypoint]:
            stored_roll_rad = float(node.get("roll_deg") or 0.0)
            reparameterized = (
                Rotation.from_quat(raw_quaternion) * Rotation.from_euler("z", stored_roll_rad)
            ).as_quat()
            variants.append(
                {
                    "candidate_id": node["candidate_id"],
                    "raw_process_target_pose_quaternion_xyzw": raw_quaternion.tolist(),
                    "stored_roll_deg_field_value": stored_roll_rad,
                    "stored_roll_unit": "rad",
                    "ik_solver_pose_quaternion_xyzw": reparameterized.tolist(),
                    "roll_definition": "raw target rotation multiplied on the right by local TCP-Z roll",
                }
            )
        candidate_pose_variants[str(waypoint)] = variants
    bridge["candidate_pose_variants"] = candidate_pose_variants
    bridge["interpretation"]["input_path_pose_discontinuity"] = bool(
        pose_metrics["input_pose_or_normal_discontinuity"]
    )
    bridge["interpretation"]["input_path_pose_discontinuity_evidence"] = {
        "position_delta_m": pose_metrics["tcp_target_position_delta_m"],
        "normal_angle_deg": pose_metrics["tcp_target_normal_angle_deg"],
        "raw_pose_rotation_delta_deg": pose_metrics["raw_process_target_pose_rotation_delta_deg"],
        "threshold_deg": pose_metrics["discontinuity_threshold_deg"],
    }
    bridge["source_artifacts"] = {
        "nodes": str(nodes_path),
        "edges": str(edges_path),
        "poses": str(poses_path),
        "collision_check_type": "adaptive_discrete_interpolation",
        "ccd_status": "not_available",
        "clearance_status": "not_available",
    }
    _write_json(out_dir / "waypoint_67_68_bottleneck.json", bridge)
    _write_json(
        out_dir / "threshold_sweep.json",
        {
            "source_phase_id": "stage_0_1",
            "source_nodes": str(nodes_path),
            "source_edges": str(edges_path),
            "thresholds": threshold_sweep(nodes, edges, 181),
            "diagnostic_only_rule": "Every threshold above 20 degrees has diagnostic_only=true.",
            "collision_label_rule": (
                "Edges rejected before collision interpolation are not promoted by the conservative result; "
                "joint_step_only_diagnostic is provided separately and is never a formal safety path."
            ),
        },
    )
    _write_csv(out_dir / "layer_reachability.csv", candidate_count_rows(nodes, 181, frontiers))
    _write_csv(out_dir / "candidate_count_by_waypoint.csv", candidate_count_rows(nodes, 181, frontiers))
    enrichment_dir = out_dir / "local_candidate_enrichment_87_94"
    enrichment_failure_path = enrichment_dir / "waypoint_87_94_candidate_failures.json"
    enrichment_summary_path = enrichment_dir / "local_candidate_enrichment_summary.json"
    if enrichment_failure_path.exists():
        enrichment_failure = json.loads(enrichment_failure_path.read_text(encoding="utf-8"))
        _write_json(out_dir / "waypoint_87_94_candidate_failures.json", enrichment_failure)
    if enrichment_summary_path.exists():
        enrichment_summary = json.loads(enrichment_summary_path.read_text(encoding="utf-8"))
        _write_json(
            out_dir / "local_candidate_enrichment_summary.json",
            {
                "diagnostic_only": True,
                "formal_stage1_output_untouched": True,
                "focused_windows": [enrichment_summary],
                "wide_window_default": [60, 100],
                "wide_window_status": "not_completed_within_execution_budget",
                "source_detail": str(enrichment_summary_path),
            },
        )
    print(json.dumps({"out_dir": str(out_dir), "node_count": len(nodes), "edge_count": len(edges)}, indent=2))


if __name__ == "__main__":
    main()
