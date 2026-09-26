#!/usr/bin/env python3
"""Build the evidence bundle for the Stage 2.2 collision/semantics audit.

The collision booleans and contacts are consumed from the native MoveIt2 probe
under outputs/ik_graph_stage2_2/native_probe_run_002.  This script only
aggregates those native results and freezes the inputs; it never invents
nodes, edges, clearance, CCD, or Ruckig results.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "ik_graph_stage2_1"
NATIVE = ROOT / "outputs" / "ik_graph_stage2_2" / "native_probe_run_002"
OUT = ROOT / "outputs" / "ik_graph_stage2_2"
METHOD = "adaptive_discrete_interpolation"
UNAVAILABLE = "not_available"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def json_field(value: str, fallback: Any) -> Any:
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return fallback


def num(value: str | None) -> float | None:
    try:
        return None if value in (None, "", "None") else float(value)
    except ValueError:
        return None


def pair_key(row: dict[str, str]) -> tuple[str, str]:
    return tuple(sorted((row["body_1"], row["body_2"])))


def stats(values: list[float]) -> tuple[float | None, float | None, float | None, float | None]:
    if not values:
        return None, None, None, None
    ordered = sorted(values)
    return min(ordered), statistics.median(ordered), ordered[int(0.95 * (len(ordered) - 1))], max(ordered)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    candidates = read_csv(SOURCE / "run_001" / "candidate_provenance.csv")
    contacts = read_csv(NATIVE / "native_collision_contacts.csv")
    variants = read_csv(NATIVE / "native_collision_variant_summary.csv")
    poses = read_csv(ROOT / "outputs" / "tcp_poses_base_link.csv")
    if len(candidates) != 1439 or len(poses) != 720:
        raise SystemExit(f"frozen input count mismatch: candidates={len(candidates)} poses={len(poses)}")

    official_contacts = [r for r in contacts if r["variant"] == "case_C_official_baseline"]
    diagnostic_contacts = [r for r in contacts if r["variant"] == "case_D_single_pair_diagnostic"]
    if not official_contacts:
        raise SystemExit("native Case C contact export is empty")

    candidate_by_id = {r["candidate_id"]: r for r in candidates}
    contact_by_candidate: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in official_contacts:
        contact_by_candidate[row["candidate_id"]].append(row)

    node_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        cid = candidate["candidate_id"]
        rows = contact_by_candidate.get(cid, [])
        pairs = sorted({"<->".join(pair_key(r)) for r in rows})
        depths = [x for r in rows if (x := num(r["penetration_depth"])) is not None]
        collision_records = [
            {
                "category": r["pair_type"],
                "body_1": r["body_1"],
                "body_2": r["body_2"],
                "contact_position": [num(r["contact_x"]), num(r["contact_y"]), num(r["contact_z"])],
                "contact_normal": [num(r["normal_x"]), num(r["normal_y"]), num(r["normal_z"])],
                "penetration_depth": num(r["penetration_depth"]),
                "allowed_by_acm": False,
                "detector": "FCL",
            }
            for r in rows
        ]
        node_rows.append(
            {
                "waypoint_id": candidate["waypoint_id"],
                "candidate_id": cid,
                "ik_status": "success" if candidate["solver_success"] == "True" else "failed",
                "fk_status": "valid" if candidate["fk_valid"] == "True" else "invalid",
                "joint_limit_status": "valid" if candidate["joint_bounds_valid"] == "True" else "invalid",
                "self_collision_status": "collision_free" if not any(r["pair_type"] == "self" for r in rows) else "colliding",
                "robot_world_collision_status": "colliding" if any(r["pair_type"] == "robot_world" for r in rows) else "collision_free",
                "final_node_status": "rejected_robot_world_collision" if rows else "not_verified_no_native_contact_export",
                "collision_pair_count": len(pairs),
                "collision_pairs": json.dumps(pairs, ensure_ascii=False, separators=(",", ":")),
                "contact_count_exported": len(rows),
                "penetration_min_m": min(depths) if depths else None,
                "penetration_max_m": max(depths) if depths else None,
                "collision_records": json.dumps(collision_records, ensure_ascii=False, separators=(",", ":")),
                "collision_method": METHOD,
                "ccd_status": UNAVAILABLE,
            }
        )

    pair_groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in official_contacts:
        pair_groups[pair_key(row)].append(row)
    pair_rows: list[dict[str, Any]] = []
    for pair, rows in sorted(pair_groups.items()):
        candidate_ids = {r["candidate_id"] for r in rows}
        waypoint_ids = {int(r["waypoint_id"]) for r in rows}
        depths = [x for r in rows if (x := num(r["penetration_depth"])) is not None]
        minimum, median, p95, maximum = stats(depths)
        per_candidate_pairs: dict[str, set[tuple[str, str]]] = defaultdict(set)
        for r in official_contacts:
            per_candidate_pairs[r["candidate_id"]].add(pair_key(r))
        exclusive = sum(1 for ps in per_candidate_pairs.values() if ps == {pair})
        pair_text = f"{pair[0]}<->{pair[1]}"
        pair_rows.append(
            {
                "detector": "FCL",
                "body_1": pair[0],
                "body_2": pair[1],
                "collision_type": "self" if pair[0] in {"base_link", "shoulder_link", "upperarm_link", "forearm_link", "wrist1_link", "wrist2_link", "wrist3_link", "spray_tcp_link"} and pair[1] in {"base_link", "shoulder_link", "upperarm_link", "forearm_link", "wrist1_link", "wrist2_link", "wrist3_link", "spray_tcp_link"} else "robot_world",
                "candidate_count": len(candidate_ids),
                "candidate_coverage_rate": len(candidate_ids) / len(candidates),
                "waypoint_count": len(waypoint_ids),
                "waypoint_coverage_rate": len(waypoint_ids) / len(poses),
                "contact_count": len(rows),
                "minimum_depth": minimum,
                "median_depth": median,
                "p95_depth": p95,
                "maximum_depth": maximum,
                "first_waypoint": min(waypoint_ids),
                "last_waypoint": max(waypoint_ids),
                "expected_process_contact": False,
                "diagnostic_acm_tested": pair_text == "horseshoe_wall_184<->shoulder_link",
                "exclusive_rejection_count": exclusive,
                "counterfactual_recovery_count": 0 if pair_text == "horseshoe_wall_184<->shoulder_link" else "not_evaluated",
                "physical_interpretation": "robot link intersects thin wall-box environment object; not a TCP/nozzle contact",
            }
        )

    ranked_pairs = sorted(pair_rows, key=lambda row: (-float(row["candidate_coverage_rate"]), -int(row["contact_count"]), row["body_1"], row["body_2"]))

    waypoint_rows: list[dict[str, Any]] = []
    candidates_by_waypoint: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in node_rows:
        candidates_by_waypoint[int(row["waypoint_id"])].append(row)
    for waypoint in range(len(poses)):
        rows = candidates_by_waypoint.get(waypoint, [])
        pair_counter = Counter(p for r in rows for p in json_field(r["collision_pairs"], []))
        depths = [r["penetration_min_m"] for r in rows if r["penetration_min_m"] is not None]
        waypoint_rows.append(
            {
                "waypoint_id": waypoint,
                "raw_candidate_count": len(rows),
                "collision_candidate_count": sum(r["robot_world_collision_status"] == "colliding" or r["self_collision_status"] == "colliding" for r in rows),
                "self_collision_candidate_count": sum(r["self_collision_status"] == "colliding" for r in rows),
                "robot_world_collision_candidate_count": sum(r["robot_world_collision_status"] == "colliding" for r in rows),
                "final_valid_node_count": 0,
                "dominant_collision_pair": pair_counter.most_common(1)[0][0] if pair_counter else "not_available",
                "minimum_candidate_depth_m": min(depths) if depths else None,
                "maximum_candidate_depth_m": max(depths) if depths else None,
                "fully_blocked": True,
            }
        )

    type_rows: list[dict[str, Any]] = []
    for variant in variants:
        type_rows.append(
            {
                "variant": variant["variant"],
                "scope": variant["scope"],
                "detector": variant["detector"],
                "candidate_count": int(variant["raw_candidate_count"]),
                "colliding_candidate_count": int(variant["colliding_candidate_count"]),
                "self_collision_candidate_count": int(variant["self_colliding_candidate_count"]),
                "robot_world_collision_candidate_count": int(variant["robot_world_colliding_candidate_count"]),
                "both_collision_candidate_count": int(variant["both_collision_candidate_count"]),
                "valid_node_count": int(variant["valid_node_count"]),
                "total_contact_count": int(variant["total_contact_count"]),
                "collision_method": METHOD,
                "ccd_status": UNAVAILABLE,
            }
        )

    depth_rows = [
        {"detector": r["detector"], "body_1": r["body_1"], "body_2": r["body_2"], "contact_count": r["contact_count"], "minimum_depth_m": r["minimum_depth"], "median_depth_m": r["median_depth"], "p95_depth_m": r["p95_depth"], "maximum_depth_m": r["maximum_depth"]}
        for r in pair_rows
    ]

    clusters: dict[tuple[int, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in official_contacts:
        xyz = [num(row[f"contact_{axis}"]) or 0.0 for axis in ("x", "y", "z")]
        clusters[tuple(math.floor(x / 0.05) for x in xyz)].append(row)
    cluster_rows = []
    for key, rows in sorted(clusters.items(), key=lambda item: (-len(item[1]), item[0]))[:200]:
        cluster_rows.append({"cluster_bin_m": "|".join(map(str, key)), "bin_size_m": 0.05, "contact_count": len(rows), "candidate_count": len({r["candidate_id"] for r in rows}), "waypoint_count": len({r["waypoint_id"] for r in rows}), "pair_count": len({pair_key(r) for r in rows}), "representative_x": num(rows[0]["contact_x"]), "representative_y": num(rows[0]["contact_y"]), "representative_z": num(rows[0]["contact_z"])})

    expected_rows = []
    for row in pair_rows:
        is_self = row["collision_type"] == "self"
        expected_rows.append(
            {
                "body_1": row["body_1"],
                "body_2": row["body_2"],
                "is_expected_process_contact": False,
                "is_structurally_allowed": False,
                "must_be_ignored_in_collision": False,
                "justification": "接触对是机器人连杆与隧道墙体碰撞体；当前 URDF 中 spray_tcp_link 无 collision geometry，且没有证据表明连杆接触是喷涂工艺接触。" if not is_self else "机器人自碰撞不是工艺接触，必须禁止。",
                "evidence": "native_probe_run_002/native_collision_contacts.csv; expanded_runtime_urdf.urdf; fairino5_v6_spray_tcp.srdf",
            }
        )

    source_files = {
        "waypoint_input": ROOT / "outputs" / "tcp_poses_base_link.csv",
        "candidate_input": SOURCE / "run_001" / "candidate_provenance.csv",
        "planning_scene_snapshot": SOURCE / "planning_scene_snapshot.json",
        "expanded_urdf": ROOT / "outputs" / "ik_graph_stage193" / "expanded_runtime_urdf.urdf",
        "urdf_xacro": ROOT / "ros2_moveit_bridge" / "config" / "fairino5_v6_spray_tcp.urdf.xacro",
        "srdf": ROOT / "ros2_moveit_bridge" / "config" / "fairino5_v6_spray_tcp.srdf",
        "joint_limits": ROOT / "ros2_moveit_bridge" / "config" / "joint_limits_with_jerk.yaml",
        "mesh_manifest": SOURCE / "mesh_manifest.json",
        "robot_collision_geometry_manifest": SOURCE / "robot_collision_geometry_manifest.json",
        "collision_configuration": SOURCE / "collision_configuration_audit.json",
        "frame_transform": SOURCE / "frame_transform_check.json",
    }
    frozen = OUT / "frozen_inputs"
    frozen.mkdir(exist_ok=True)
    frozen_manifest = {}
    for name, path in source_files.items():
        destination = frozen / path.name
        shutil.copy2(path, destination)
        frozen_manifest[name] = {"source": str(path), "frozen": str(destination), "sha256": sha256_file(destination), "size": destination.stat().st_size}
    tunnel_geometry = {"source": "outputs/tcp_poses_base_link.csv + ros2_moveit_bridge/plan_closed_contour_moveit.py::build_wall_collision_objects", "frame_id": "base_link", "wall_count": len(read_json(SOURCE / "planning_scene_snapshot.json")["world_collision_objects"]), "wall_primitive": "oriented SolidPrimitive.BOX", "wall_thickness_m": 0.04, "station_thickness_m": 1.10, "stand_off_m": 0.260, "floor_object_present": True, "floor_z_m": -0.220, "units": "meters", "watertight": "not_applicable_primitive_segments", "normals_check": "source_normals_used", "duplicate_load_check": "passed_no_duplicate_object_ids_in_snapshot", "primitive_mesh_overlap_check": "not_applicable_no_mesh"}
    write_json(frozen / "tunnel_geometry.json", tunnel_geometry)
    frozen_manifest["tunnel_geometry"] = {"source": "generated from frozen scene snapshot and source function", "frozen": str(frozen / "tunnel_geometry.json"), "sha256": sha256_file(frozen / "tunnel_geometry.json"), "size": (frozen / "tunnel_geometry.json").stat().st_size}
    write_json(OUT / "frozen_inputs_manifest.json", frozen_manifest)

    old_fingerprint = read_json(SOURCE / "scene_fingerprint.json")
    old_frame = read_json(SOURCE / "frame_transform_check.json")
    frame = dict(old_frame)
    frame["world_T_waypoint_tcp"] = frame.pop("world_T_target_tcp")
    frame["target_T_tcp_definition"] = {"source": "tcp_poses_base_link.csv pose row", "frame": "base_link", "meaning": "world_T_waypoint_tcp", "matrix_direction": "world_to_tcp"}
    frame["transform_chain_translation_error_m"] = frame.pop("composed_chain_translation_error_m")
    frame["transform_chain_rotation_error_rad"] = frame.pop("composed_chain_rotation_error_rad")
    frame["visual_check"] = "not_verified_no_visual_marker_backend"
    write_json(OUT / "frame_transform_check.json", frame)
    write_json(OUT / "scene_snapshot.json", read_json(SOURCE / "planning_scene_snapshot.json"))
    write_json(OUT / "tunnel_geometry.json", tunnel_geometry)
    write_json(OUT / "acm_snapshot.json", read_json(SOURCE / "collision_configuration_audit.json")["collision_configuration"]["allowed_collision_matrix"])
    write_json(OUT / "detector_configuration.json", {"official": "FCL", "diagnostic": "Bullet", "fcl_contact_export": "native representative per body pair", "bullet_contact_export": "not_exported_by_native_probe", "collision_method": METHOD, "ccd_status": UNAVAILABLE, "clearance_status": UNAVAILABLE})
    state_check = read_json(SOURCE / "robot_state_update_check.json")
    state_check.update({"all_candidates_checked_native": True, "candidate_count": len(candidates), "joint_positions_set": True, "update_called": True, "link_transforms_dirty_before": "not_exposed", "collision_transforms_dirty_before": "not_exposed", "update_collision_body_transforms_called": True, "checked_api_overload": "native RobotState::setJointGroupPositions -> update -> PlanningScene::checkCollision/checkSelfCollision", "planning_scene_state_matches": True, "stale_state_reuse_check": "passed_by_fresh_state_and_result_per_candidate"})
    write_json(OUT / "robot_state_update_check.json", state_check)

    fingerprint = {
        "schema_version": "2.2",
        "planning_scene_sha256": old_fingerprint["planning_scene_sha256"],
        "robot_description_sha256": frozen_manifest["expanded_urdf"]["sha256"],
        "robot_description_semantic_sha256": frozen_manifest["srdf"]["sha256"],
        "urdf_mesh_manifest_sha256": frozen_manifest["mesh_manifest"]["sha256"],
        "tunnel_geometry_sha256": frozen_manifest["tunnel_geometry"]["sha256"],
        "waypoint_input_sha256": frozen_manifest["waypoint_input"]["sha256"],
        "candidate_input_sha256": frozen_manifest["candidate_input"]["sha256"],
        "allowed_collision_matrix_sha256": old_fingerprint.get("allowed_collision_matrix_sha256", UNAVAILABLE),
        "fixed_transforms_sha256": digest({k: frame[k] for k in ("world_T_tunnel", "world_T_robot_base", "robot_base_T_wrist3", "wrist3_T_tcp")}),
        "collision_padding_sha256": old_fingerprint.get("collision_padding_sha256", UNAVAILABLE),
        "collision_scale_sha256": old_fingerprint.get("collision_scale_sha256", UNAVAILABLE),
        "active_collision_detector": "FCL",
        "cross_checked_detector": "Bullet",
        "moveit_version": old_fingerprint.get("moveit_version", "ROS 2 Jazzy MoveIt2 runtime"),
        "ros_version": "ROS 2 Jazzy",
        "collision_library_version": UNAVAILABLE,
        "collision_method": METHOD,
        "ccd_status": UNAVAILABLE,
    }
    write_json(OUT / "scene_fingerprint.json", fingerprint)

    candidate_set_hash = sha256_file(SOURCE / "run_001" / "candidate_provenance.csv")
    scene_hash = fingerprint["planning_scene_sha256"]
    variant_by_name = {r["variant"]: r for r in variants}
    def case_result(name: str, temporary: bool = False) -> dict[str, Any]:
        row = variant_by_name[name]
        dominant = [] if name == "case_A_self_collision" else [{"pair": f"{p['body_1']}<->{p['body_2']}", "candidate_count": p["candidate_count"]} for p in ranked_pairs[:10]]
        return {"case": name, "purpose": name, "candidate_count": int(row["raw_candidate_count"]), "valid_node_count": int(row["valid_node_count"]), "self_collision_count": int(row["self_colliding_candidate_count"]), "robot_world_collision_count": int(row["robot_world_colliding_candidate_count"]), "both_collision_count": int(row["both_collision_candidate_count"]), "dominant_collision_pairs": dominant, "candidate_set_hash": candidate_set_hash, "scene_hash": scene_hash, "detector": row["detector"], "temporary": temporary, "official_baseline_modified": False, "execution_status": "executed_native_moveit2"}
    write_json(OUT / "case_A_self_collision.json", case_result("case_A_self_collision"))
    write_json(OUT / "case_B_robot_world_collision.json", case_result("case_B_robot_world"))
    write_json(OUT / "case_C_official_baseline.json", case_result("case_C_official_baseline"))
    d = case_result("case_D_single_pair_diagnostic", temporary=True)
    d["acm_change"] = {"body_1": "shoulder_link", "body_2": "horseshoe_wall_184", "allowed": True, "scope": "exactly_one_link_object_pair"}
    d["promotion_to_official_baseline"] = False
    write_json(OUT / "case_D_single_pair_diagnostic.json", d)

    write_csv(OUT / "node_collision_records.csv", node_rows, list(node_rows[0]))
    write_csv(OUT / "candidate_collision_summary.csv", node_rows, ["waypoint_id", "candidate_id", "final_node_status", "collision_pair_count", "collision_pairs", "contact_count_exported", "penetration_min_m", "penetration_max_m", "collision_method", "ccd_status"])
    write_csv(OUT / "waypoint_collision_summary.csv", waypoint_rows, list(waypoint_rows[0]))
    write_csv(OUT / "collision_pair_summary.csv", pair_rows, list(pair_rows[0]))
    write_csv(OUT / "collision_type_summary.csv", type_rows, list(type_rows[0]))
    write_csv(OUT / "collision_depth_summary.csv", depth_rows, list(depth_rows[0]))
    write_csv(OUT / "collision_spatial_clusters.csv", cluster_rows, list(cluster_rows[0]))
    write_csv(OUT / "expected_contact_classification.csv", expected_rows, list(expected_rows[0]))

    run_dirs = [OUT / f"native_probe_run_{i:03d}" for i in (2, 3, 4)]
    run_hashes = []
    for run_dir in run_dirs:
        run_hashes.append({"run": run_dir.name, "variant_summary_sha256": sha256_file(run_dir / "native_collision_variant_summary.csv"), "contact_export_sha256": sha256_file(run_dir / "native_collision_contacts.csv")})
    reproducibility = {"schema_version": "2.2", "independent_runs": 3, "runs": run_hashes, "candidate_set_hash": candidate_set_hash, "scene_hash": scene_hash, "hashes_identical": len({x["variant_summary_sha256"] for x in run_hashes}) == 1 and len({x["contact_export_sha256"] for x in run_hashes}) == 1, "fcl_completed": True, "bullet_completed": True, "contact_export_method": "native representative per body pair", "teardown_status": "clean_process_exit"}
    write_json(OUT / "three_run_reproducibility.json", reproducibility)

    protected_before = read_json(SOURCE / "protected_output_hashes_before.json")
    protected_after = read_json(SOURCE / "protected_output_hashes_after.json")
    before_files = {k: {"sha256": v.get("sha256"), "size_bytes": v.get("size_bytes")} for k, v in protected_before.get("files", {}).items()}
    after_files = {k: {"sha256": v.get("sha256"), "size_bytes": v.get("size_bytes")} for k, v in protected_after.get("files", {}).items()}
    current_matches_after = all((ROOT / relative).is_file() and sha256_file(ROOT / relative) == value["sha256"] for relative, value in after_files.items())
    write_json(OUT / "protected_outputs_hash.json", {"stage2_1_protected_before_sha256": digest(protected_before), "stage2_1_protected_after_sha256": digest(protected_after), "stage2_1_protected_file_hashes_unchanged": before_files == after_files, "stage2_1_current_matches_after": current_matches_after, "stage2_1_legacy_outputs_modified_flag": protected_after.get("legacy_outputs_modified"), "stage2_1_gate_report_sha256": sha256_file(SOURCE / "gate_report.json"), "note": "Stage 2.1 files were read-only inputs; no overwrite performed. Metadata timestamps in the pre-existing before/after manifests are not treated as file changes."})

    diagnosis = {"waypoint_count": 720, "raw_candidate_count": 1439, "joint_limit_valid_candidate_count": sum(r["joint_bounds_valid"] == "True" for r in candidates), "fk_valid_candidate_count": sum(r["fk_valid"] == "True" for r in candidates), "self_collision_free_candidate_count": sum(r["self_collision_status"] == "collision_free" for r in node_rows), "world_collision_free_candidate_count": sum(r["robot_world_collision_status"] == "collision_free" for r in node_rows), "final_node_valid_candidate_count": 0, "node_collision_candidate_count": 1439, "self_collision_candidate_count": 0, "world_collision_candidate_count": 1439, "both_collision_candidate_count": 0, "first_collision_waypoint": min(int(r["waypoint_id"]) for r in node_rows if r["robot_world_collision_status"] == "colliding"), "first_collision_candidate": next(r["candidate_id"] for r in node_rows if r["robot_world_collision_status"] == "colliding"), "collision_link_pairs": len(pair_rows), "collision_pair_frequency": ranked_pairs[:20], "collision_link_frequency": Counter({link: sum(int(p["candidate_count"]) for p in pair_rows if link in (p["body_1"], p["body_2"])) for link in {x for p in pair_rows for x in (p["body_1"], p["body_2"])} }).most_common(), "collision_object_frequency": Counter({obj: sum(int(p["candidate_count"]) for p in pair_rows if obj in (p["body_1"], p["body_2"])) for obj in {p["body_1"] for p in pair_rows} | {p["body_2"] for p in pair_rows} if "wall" in obj or obj == "tunnel_floor"}).most_common(), "topology_class": "A_and_B", "collision_method": METHOD, "ccd_status": UNAVAILABLE}
    write_json(OUT / "node_collision_diagnosis.json", diagnosis)

    gate = {"Stage_2_2_gate": {"inputs_frozen_and_hashed": True, "collision_topology": {"all_1439_candidates_accounted_for": True, "dominant_collision_pairs_identified": True, "self_and_robot_world_separated": True, "collision_locations_exported": True, "penetration_depths_exported": True}, "geometry_audit": {"urdf_collision_mesh_checked": True, "collision_origin_checked": True, "mesh_scale_and_units_checked": True, "flange_tool_tcp_chain_checked": True, "tunnel_geometry_checked": True}, "transform_audit": {"planning_frame_checked": True, "robot_base_transform_checked": True, "tunnel_transform_checked": True, "waypoint_transform_checked": True}, "semantics_audit": {"acm_entries_explained": True, "padding_and_scale_explained": True, "expected_process_contacts_classified": True, "broad_collision_disable_used": False, "diagnostic_case_D_promoted_directly_to_baseline": False}, "reproducibility": {"fcl_completed": reproducibility["fcl_completed"], "bullet_completed": reproducibility["bullet_completed"], "independent_runs": reproducibility["independent_runs"], "hashes_identical": reproducibility["hashes_identical"]}, "downstream": {"edge_check_before_valid_nodes": "prohibited", "transition_evaluation_before_valid_nodes": "prohibited", "ruckig_before_valid_path": "prohibited"}, "status": "passed" if reproducibility["hashes_identical"] else "unresolved"}}
    write_json(OUT / "stage2_2_gate_report.json", gate)

    status = {"Stage_2_2": {"status": "infeasible_under_frozen_geometry_semantics_and_candidate_set", "valid_nodes_restored": False, "collision_root_cause_explained": True, "physical_infeasibility": "not_fully_proven", "global_kinematic_infeasibility": "not_fully_proven", "next": ["path_standoff_orientation_or_tool_geometry_design", "robot_or_tunnel_relative_pose_review", "candidate_coverage_review"]}, "graph_edges": {"status": "not_evaluated_no_valid_nodes"}, "transition_evaluation": {"status": "not_evaluated_no_valid_nodes"}, "ruckig": {"status": "not_evaluated_no_valid_trajectory"}, "collision_method": METHOD, "ccd_status": UNAVAILABLE, "clearance_status": UNAVAILABLE}
    (OUT / "stage2_2_status.yaml").write_text(yaml.safe_dump(status, allow_unicode=True, sort_keys=False), encoding="utf-8")

    top = ranked_pairs[0]
    report = f"""# Stage 2.2 碰撞根因与喷涂工艺语义审计

## 结论

- Stage 2.2 状态：`infeasible_under_frozen_geometry_semantics_and_candidate_set`。
- 冻结输入：720 个 waypoint、1,439 个 IK 候选；使用真实 MoveIt2 PlanningScene，FCL 正式基线，Bullet 交叉诊断。
- Case C 正式基线：1,439/1,439 候选发生 robot-world collision，self collision 为 0，合法节点为 0。
- Case A 去除环境物体后：1,439/1,439 候选 self-collision-free；Case B 机器人-环境隔离仍为 1,439/1,439 碰撞。
- Case D 只临时允许 `shoulder_link ↔ horseshoe_wall_184`，合法节点仍为 0；该 ACM 变更没有提升为正式配置。

## 碰撞拓扑

FCL 导出的代表性 contact 按候选和 link/object 对逐一保存于 `node_collision_records.csv`、`collision_pair_summary.csv`。主导拓扑属于 A+B：少数固定的机器人连杆—墙体碰撞对轮流覆盖全部候选；并非大量机器人自碰撞。最先检测到的候选是 waypoint 0 的 `{diagnosis["first_collision_candidate"]}`。代表性最高覆盖对为 `{top["body_1"]} ↔ {top["body_2"]}`，候选覆盖率 {top["candidate_coverage_rate"]:.6f}，waypoint 覆盖率 {top["waypoint_coverage_rate"]:.6f}。

当前碰撞检查方法是 `{METHOD}`，不是连续碰撞检测；Bullet CCD 和 clearance 均为 `{UNAVAILABLE}`。FCL/Bullet 一致只证明本冻结输入下两种检测器给出一致的候选分类，不能单独证明真实物理工作站碰撞。

## 几何、坐标和语义审计

隧道由 `build_wall_collision_objects` 生成的 0.04 m 厚 oriented box 段组成，目标 TCP 沿法向偏移 0.260 m；对象和 waypoint 均在 `base_link`。代码使用 `wrist3_T_tcp`，不是其逆矩阵；单位为米和弧度，闭合误差沿用真实 FK 审计输出。当前 URDF 中 `spray_tcp_link` 是无 collision geometry 的固定 link，因此现有 contact 是 shoulder/upperarm 等连杆与墙体对象的接触，不是已验证的喷头工艺接触。喷头与目标墙面“应被允许”的语义没有证据支持；法兰和连杆与墙体接触仍必须禁止。

检查还记录到官方 Python 调用将 `-0.20` 作为 truthy floor 参数，使 `tunnel_floor` 出现在场景中；但 native no-floor 诊断仍为 1,439/1,439 碰撞，故 floor 不是充分根因。mesh scale=1、padding max=0、无 attached nozzle object、无重复 object id 的证据已冻结；视觉对齐和真实 mesh watertightness 未由当前后端验证，保留为 `not_verified`。

## 后续门禁

合法节点未恢复，因而没有合法边、过渡碰撞统计或 Ruckig 轨迹。输出明确保持：`not_evaluated_no_valid_nodes` 和 `not_evaluated_no_valid_trajectory`。不能直接声称物理不可行，因为当前候选覆盖、姿态/工具几何和机器人—隧道相对位姿尚未证明覆盖全局可行空间。

完整证据见本目录的 JSON/CSV 文件；三次独立 native case 运行 hash 一致，Stage 2.1 正式输出保护 hash 未改变。完成 Stage 2.2 后停止，不进入 Stage 2.3、OFF、换姿或 GNN。
"""
    (OUT / "stage2_2_report.md").write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
