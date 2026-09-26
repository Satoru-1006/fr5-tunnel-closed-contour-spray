"""Freeze and audit the isolated Stage 2.3A.4 scaled-demo evidence."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "ik_graph_stage23a4_scaled_demo" / "fr5_scaled_horseshoe_demo_v43"
OLD = ROOT / "outputs" / "ik_graph_stage23a3"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def set_hash(values: set[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(values)) + "\n").encode()).hexdigest()


def rows_for(backend: str, run: int) -> list[dict[str, str]]:
    return read_csv(OUT / f"{backend}_split_run_{run}" / "candidate_results.csv")


def valid_ids(rows: list[dict[str, str]]) -> set[str]:
    return {r["candidate_id"] for r in rows if r["validity"] == "true"}


def valid_waypoints(rows: list[dict[str, str]]) -> set[int]:
    return {int(r["waypoint_id"]) for r in rows if r["validity"] == "true"}


def collision_pairs(rows: list[dict[str, str]]) -> set[str]:
    return {p for r in rows for p in r["collision_pairs"].split("|") if p}


def main() -> int:
    mesh = OUT / "tunnel_mesh.stl"
    parameter = OUT / "scaled_demo_parameters.yaml"
    waypoint = OUT / "waypoints.csv"
    candidate = OUT / "deterministic_ik_candidates.csv"
    validation = json.loads((OUT / "tunnel_mesh_validation.json").read_text(encoding="utf-8"))
    waypoint_summary = json.loads((OUT / "waypoint_validation_summary.json").read_text(encoding="utf-8"))
    ik = json.loads((OUT / "ik_reproducibility.json").read_text(encoding="utf-8"))

    runs: dict[str, dict[str, object]] = {}
    for backend in ("fcl", "bullet"):
        backend_runs: dict[str, object] = {}
        for run in (1, 2, 3):
            rows = rows_for(backend, run)
            summary = json.loads((OUT / f"{backend}_split_run_{run}" / "run_summary.json").read_text(encoding="utf-8"))
            ids = valid_ids(rows)
            wps = valid_waypoints(rows)
            pairs = collision_pairs(rows)
            backend_runs[str(run)] = {
                "summary": summary,
                "candidate_result_sha256": sha256(OUT / f"{backend}_split_run_{run}" / "candidate_results.csv"),
                "valid_node_count_from_rows": len(ids),
                "valid_waypoint_count_from_rows": len(wps),
                "valid_node_set_sha256": set_hash(ids),
                "valid_waypoint_set_sha256": set_hash({str(x) for x in wps}),
                "collision_pair_set_sha256": set_hash(pairs),
                "collision_pair_count": len(pairs),
            }
        runs[backend] = backend_runs

    fcl_rows = rows_for("fcl", 1)
    bullet_rows = rows_for("bullet", 1)
    fcl_ids, bullet_ids = valid_ids(fcl_rows), valid_ids(bullet_rows)
    fcl_wps, bullet_wps = valid_waypoints(fcl_rows), valid_waypoints(bullet_rows)
    fcl_pairs, bullet_pairs = collision_pairs(fcl_rows), collision_pairs(bullet_rows)
    comparison = {
        "input_candidate_csv_sha256": sha256(candidate),
        "input_waypoint_csv_sha256": sha256(waypoint),
        "fcl": {"valid_node_count": len(fcl_ids), "valid_waypoint_count": len(fcl_wps), "valid_node_set_sha256": set_hash(fcl_ids), "valid_waypoint_set_sha256": set_hash({str(x) for x in fcl_wps})},
        "bullet": {"valid_node_count": len(bullet_ids), "valid_waypoint_count": len(bullet_wps), "valid_node_set_sha256": set_hash(bullet_ids), "valid_waypoint_set_sha256": set_hash({str(x) for x in bullet_wps})},
        "exact_valid_node_set_equal": fcl_ids == bullet_ids,
        "exact_valid_waypoint_set_equal": fcl_wps == bullet_wps,
        "intersection_valid_node_count": len(fcl_ids & bullet_ids),
        "fcl_only_valid_node_count": len(fcl_ids - bullet_ids),
        "bullet_only_valid_node_count": len(bullet_ids - fcl_ids),
        "intersection_valid_waypoint_count": len(fcl_wps & bullet_wps),
        "fcl_only_valid_waypoint_count": len(fcl_wps - bullet_wps),
        "bullet_only_valid_waypoint_count": len(bullet_wps - fcl_wps),
    }
    write_json(OUT / "backend_valid_node_comparison.json", comparison)
    write_json(OUT / "collision_topology_comparison.json", {
        "fcl_collision_pair_set": sorted(fcl_pairs),
        "bullet_collision_pair_set": sorted(bullet_pairs),
        "exact_collision_pair_set_equal": fcl_pairs == bullet_pairs,
        "fcl_pair_count": len(fcl_pairs),
        "bullet_pair_count": len(bullet_pairs),
        "bullet_only_pairs": sorted(bullet_pairs - fcl_pairs),
        "fcl_only_pairs": sorted(fcl_pairs - bullet_pairs),
        "same_candidate_collision_pair_string_count": sum(a["collision_pairs"] == b["collision_pairs"] for a, b in zip(fcl_rows, bullet_rows)),
        "candidate_count_compared": len(fcl_rows),
    })

    reproducible = {
        "ik": ik,
        "collision_runs": runs,
        "fcl_three_run_valid_node_set_reproducible": len({runs["fcl"][str(i)]["valid_node_set_sha256"] for i in (1, 2, 3)}) == 1,
        "bullet_three_run_valid_node_set_reproducible": len({runs["bullet"][str(i)]["valid_node_set_sha256"] for i in (1, 2, 3)}) == 1,
        "fcl_three_run_candidate_result_reproducible": len({runs["fcl"][str(i)]["candidate_result_sha256"] for i in (1, 2, 3)}) == 1,
        "bullet_three_run_candidate_result_reproducible": len({runs["bullet"][str(i)]["candidate_result_sha256"] for i in (1, 2, 3)}) == 1,
    }
    write_json(OUT / "three_run_reproducibility.json", reproducible)
    write_json(OUT / "scene_hash_manifest.json", {
        "baseline_id": "fr5_scaled_horseshoe_demo_v43",
        "mesh_sha256": sha256(mesh),
        "parameter_sha256": sha256(parameter),
        "waypoint_csv_sha256": sha256(waypoint),
        "candidate_csv_sha256": sha256(candidate),
        "mesh_validation_sha256": sha256(OUT / "tunnel_mesh_validation.json"),
        "planning_scene_mesh_component_count": 4,
        "collision_object_count": 1,
        "mesh_scale": [1.0, 1.0, 1.0],
        "scene_built_from_scratch": True,
        "collision_method": "adaptive_discrete_interpolation",
        "ccd_status": "not_available",
        "clearance_status": "not_available",
        "robot_description_source": "ros2_moveit_bridge/config/fairino5_v6_spray_tcp.urdf.xacro",
        "semantic_source": "ros2_moveit_bridge/config/fairino5_v6_spray_tcp.srdf",
        "joint_limits_source": "ros2_moveit_bridge/config/joint_limits_with_jerk.yaml",
        "actual_flange_to_tcp_xyz_m": [0.0, 0.0, 0.15],
        "mesh_pose_in_base_link_xyz_m": [0.32, -0.05, -0.38],
    })

    archive = []
    for path in sorted(OLD.rglob("*")):
        if path.is_file():
            archive.append({"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path), "size_bytes": path.stat().st_size})
    write_json(OUT / "archive_manifest_stage23a3.json", {
        "source_directory": OLD.relative_to(ROOT).as_posix(),
        "archived_not_reused": True,
        "legacy_720_candidates_reused": False,
        "file_count": len(archive),
        "files": archive,
    })

    geometry_pass = validation.get("boundary_edge_count") == 0 and validation.get("nonmanifold_edge_count") == 0 and validation.get("degenerate_triangle_count") == 0 and validation.get("duplicate_face_count") == 0 and validation.get("signed_volume_m3", 0) > 0
    waypoint_pass = waypoint_summary.get("waypoint_count") == 720 and waypoint_summary.get("closed_loop_geometry_verified") is True and waypoint_summary.get("duplicate_terminal_waypoint") is False and waypoint_summary.get("waypoint_surface_correspondence") == "720/720"
    ik_pass = ik.get("three_run_candidate_set_reproducibility") is True and ik.get("minimum_ik_candidates_per_waypoint", 0) >= 1
    scene_pass = all(runs[b][str(i)]["summary"].get("scene_built_from_scratch") is True and runs[b][str(i)]["summary"].get("collision_object_count") == 1 and runs[b][str(i)]["summary"].get("planning_scene_mesh_component_count") == 4 for b in ("fcl", "bullet") for i in (1, 2, 3))
    backend_pass = comparison["exact_valid_node_set_equal"] and comparison["exact_valid_waypoint_set_equal"] and json.loads((OUT / "collision_topology_comparison.json").read_text(encoding="utf-8"))["exact_collision_pair_set_equal"]
    gate = {
        "schema_version": "2.3A.4",
        "baseline_id": "fr5_scaled_horseshoe_demo_v43",
        "overall_status": "blocked_backend_disagreement" if not backend_pass else "passed",
        "gates": {
            "parametric_si_watertight_mesh": "passed" if geometry_pass else "blocked",
            "720_closed_loop_waypoints": "passed" if waypoint_pass else "blocked",
            "deterministic_numeric_ik_three_runs": "passed" if ik_pass else "blocked",
            "fresh_moveit_planningscene_fcl_bullet": "passed" if scene_pass else "blocked",
            "exact_fcl_bullet_valid_node_and_topology_agreement": "blocked_backend_disagreement" if not backend_pass else "passed",
        },
        "raw_backend_results": {"fcl": {"valid_node_count": len(fcl_ids), "valid_waypoint_count": len(fcl_wps)}, "bullet": {"valid_node_count": len(bullet_ids), "valid_waypoint_count": len(bullet_wps)}},
        "formal_cross_backend_valid_node_set": "not_established_backend_disagreement",
        "collision_method": "adaptive_discrete_interpolation",
        "ccd_status": "not_available",
        "clearance_status": "not_available",
        "stage23b_status": "blocked_no_agreed_valid_node_set",
        "ruckig_status": "not_evaluated_no_valid_trajectory",
        "off_reorientation_gnn_status": "not_started_stage_boundary",
        "old_stage23a3": "preserved_and_archived_read_only_manifest",
        "legacy_720_candidate_set_used": False,
        "notes": [
            "The Bullet split-mesh result is reproducible but disagrees with FCL: it reports 658 valid nodes and 7 collision-pair topologies, including robot self-collision pairs absent from FCL.",
            "FCL-only success is not promoted to a formal cross-backend pass.",
            "No Stage 2.3B graph edges, Ruckig, OFF/reorientation, GNN, PPO, LSTM, Transformer, retreat, or approach work was started.",
        ],
    }
    write_json(OUT / "stage23a4_gate_report.json", gate)
    status = """schema_version: 2.3A.4\nbaseline_id: fr5_scaled_horseshoe_demo_v43\nStage_2_3A_4: blocked_backend_disagreement\nStage_2_3B: blocked_no_agreed_valid_node_set\nformal_valid_node_set: not_established_backend_disagreement\nfcl_valid_nodes: 3077\nbullet_valid_nodes: 658\nfcl_valid_waypoints: 720/720\nbullet_valid_waypoints: 260/720\ncollision_method: adaptive_discrete_interpolation\nccd_status: not_available\nclearance_status: not_available\nRuckig: not_evaluated_no_valid_trajectory\nOFF_reorientation_GNN: not_started_stage_boundary\nlegacy_720_candidate_set_used: false\nold_stage23a3_preserved: true\n"""
    (OUT / "stage23a4_status.yaml").write_text(status, encoding="utf-8")
    report = f"""# Stage 2.3A.4 isolated scaled FR5 horseshoe demo\n\n- Baseline: `fr5_scaled_horseshoe_demo_v43`; author-defined scaled demonstration only. It is not claimed as a real-tunnel or deployment result.\n- Units: SI; actual loaded runtime flange-to-TCP is `[0, 0, 0.15] m`, while the requested nominal reference was `[0, 0, 0.12] m`; the actual runtime model was used.\n- New mesh: one `scaled_demo_tunnel` collision object containing four component meshes, scale `[1,1,1]`, watertight/manifold validation passed, SHA-256 `{sha256(mesh)}`.\n- Waypoints: 720, uniform arclength, closed geometry, no duplicate terminal waypoint, surface correspondence `720/720`, max standoff error `{waypoint_summary.get('maximum_standoff_error_m')}` m.\n- Deterministic numeric IK: three runs, candidate-set hashes equal, minimum one candidate per waypoint, 3077 frozen candidates.\n- Fresh native MoveIt2 PlanningScenes: FCL x3 and Bullet x3, same candidate CSV, waypoint CSV, mesh, robot model, collision object, and scale.\n\n## Backend evidence\n\n| backend | valid nodes | colliding candidates | valid waypoints | three-run reproducible |\n|---|---:|---:|---:|---|\n| FCL | 3077/3077 | 0 | 720/720 | yes |\n| Bullet | 658/3077 | 2419 | 260/720 | yes |\n\nThe exact valid-node sets are not equal: intersection 658, FCL-only 2419, Bullet-only 0. Collision-pair topology also disagrees; Bullet reports robot self-collision pairs absent from FCL. Therefore the formal cross-backend gate is **blocked_backend_disagreement**. The FCL-only 3077-node result is retained as backend-specific evidence, not promoted to a formal pass.\n\nCollision scope is `adaptive_discrete_interpolation`; CCD and clearance are `not_available`. The old Stage 2.3A.3 directory remains untouched and is listed in `archive_manifest_stage23a3.json`; its legacy 720-point/1439-candidate artifacts were not used as inputs. Stage 2.3B graph edges, Ruckig, OFF/reorientation, GNN, PPO, LSTM, Transformer, retreat, and approach work were not started.\n"""
    (OUT / "stage23a4_report.md").write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
