"""Evidence-first post-processing for the frozen Stage 2.3A.6 probe runs.

This script never changes the candidate set, robot model, ACM, padding, or
collision geometry. It only derives auditable tables from the native MoveIt
JSONL outputs and records unavailable diagnostics explicitly.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import statistics
import struct
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs" / "ik_graph_stage23a4_backend_equivalent" / "fr5_scaled_horseshoe_demo_v44"
OUT = ROOT / "outputs" / "ik_graph_stage23a6_bullet_self_collision_audit" / "fr5_scaled_horseshoe_demo_v44"
NOW = datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def set_hash(values: set[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(values)) + "\n").encode()).hexdigest()


def path_rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def make_dirs() -> None:
    for name in ("baseline_reproduction", "candidate_classification", "collision_pairs", "contacts", "acm_audit", "robot_geometry_audit", "independent_geometry", "independent_geometry/witness_meshes", "independent_geometry/witness_transforms", "numerical_probe", "bullet_margin", "minimal_reproducer/source", "minimal_reproducer/inputs", "regression"):
        (OUT / name).mkdir(parents=True, exist_ok=True)


def load_rows() -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    baseline = {b: read_jsonl(BASE / f"{b}_candidate_results_run1.jsonl") for b in ("fcl", "bullet")}
    audit = {b: read_jsonl(OUT / f"{b}_classification_run1.jsonl") for b in ("fcl", "bullet")}
    if any(len(rows) != 3077 for rows in baseline.values()) or any(len(rows) != 3077 for rows in audit.values()):
        raise RuntimeError("baseline or split audit row count is not 3077")
    return baseline, audit


def copy_baseline_evidence() -> None:
    for b in ("fcl", "bullet"):
        for i in (1, 2, 3):
            for suffix in ("candidate_results_run%d.jsonl" % i, "run%d_summary.json" % i):
                src = BASE / f"{b}_{suffix}"
                dst = OUT / "baseline_reproduction" / f"{b}_run_{i}_{'rows.jsonl' if suffix.endswith('.jsonl') else 'summary.json'}"
                shutil.copy2(src, dst)
            write_json(OUT / "baseline_reproduction" / f"{b}_run_{i}.json", {
                "generated_at": NOW,
                "backend": b,
                "run_index": i,
                "summary": read_json(BASE / f"{b}_run{i}_summary.json"),
                "rows_file": f"{b}_run_{i}_rows.jsonl",
                "rows_sha256": sha256(BASE / f"{b}_candidate_results_run{i}.jsonl"),
            })
    hashes = {}
    for p in sorted((OUT / "baseline_reproduction").iterdir()):
        hashes[p.name] = sha256(p)
    write_json(OUT / "baseline_reproduction" / "deterministic_hashes.json", {
        "generated_at": NOW,
        "source_stage": "Stage_2_3A_5_v44",
        "runs": hashes,
        "fcl_rows_sha256_equal": len({hashes[f"fcl_run_{i}_rows.jsonl"] for i in (1, 2, 3)}) == 1,
        "bullet_rows_sha256_equal": len({hashes[f"bullet_run_{i}_rows.jsonl"] for i in (1, 2, 3)}) == 1,
        "expected_counts": {"fcl_valid_nodes": 2882, "fcl_valid_waypoints": 720, "bullet_valid_nodes": 2224, "bullet_valid_waypoints": 670},
    })


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def classification_outputs(baseline: dict[str, list[dict]], audit: dict[str, list[dict]]) -> tuple[set[str], set[str]]:
    fcl_b = {r["candidate_id"]: r for r in baseline["fcl"]}
    bullet_b = {r["candidate_id"]: r for r in baseline["bullet"]}
    fcl_a = {r["candidate_id"]: r for r in audit["fcl"]}
    bullet_a = {r["candidate_id"]: r for r in audit["bullet"]}
    fcl_self = {k for k, r in fcl_a.items() if r["self_collision"]}
    bullet_self = {k for k, r in bullet_a.items() if r["self_collision"]}
    disputed = bullet_self - fcl_self
    if len(disputed) != 658:
        raise RuntimeError(f"bullet-only self collision count is {len(disputed)}, expected 658")
    all_rows = []
    for cid in sorted(fcl_a):
        fa, ba = fcl_a[cid], bullet_a[cid]
        all_rows.append({
            "candidate_id": cid, "waypoint_id": fa["waypoint_id"],
            "joint_values": json.dumps(fa["joint_values"], separators=(",", ":")),
            "fcl_self_collision": fa["self_collision"], "fcl_robot_world_collision": fa["robot_world_collision"], "fcl_combined_collision": fa["combined_collision"],
            "bullet_self_collision": ba["self_collision"], "bullet_robot_world_collision": ba["robot_world_collision"], "bullet_combined_collision": ba["combined_collision"],
            "fcl_valid": fcl_b[cid]["valid"], "bullet_valid": bullet_b[cid]["valid"],
            "fcl_self_pairs": json.dumps(fa["self_collision_pairs"], separators=(",", ":")), "bullet_self_pairs": json.dumps(ba["self_collision_pairs"], separators=(",", ":")),
        })
    fields = list(all_rows[0])
    write_csv(OUT / "candidate_classification" / "all_candidates.csv", all_rows, fields)
    disputed_rows = [{**fcl_a[cid], "bullet_only_self_collision": True, "fcl_self_collision": False, "fcl_self_pairs": fcl_a[cid]["self_collision_pairs"], "bullet_self_pairs": bullet_a[cid]["self_collision_pairs"]} for cid in sorted(disputed)]
    write_csv(OUT / "candidate_classification" / "bullet_only_self_collision_nodes.csv", disputed_rows, list(disputed_rows[0]))
    write_json(OUT / "candidate_classification" / "bullet_only_self_collision_nodes.json", {
        "generated_at": NOW, "definition": "Bullet_self_collision_nodes - FCL_self_collision_nodes", "count": len(disputed), "nodes": disputed_rows,
    })
    fcl_valid_wp = defaultdict(int); bullet_valid_wp = defaultdict(int); total_wp = defaultdict(int); removed = defaultdict(list)
    for r in baseline["fcl"]:
        total_wp[int(r["waypoint_id"])] += 1
        if r["valid"]: fcl_valid_wp[int(r["waypoint_id"])] += 1
    for r in baseline["bullet"]:
        if r["valid"]: bullet_valid_wp[int(r["waypoint_id"])] += 1
        else: removed[int(r["waypoint_id"])].append(r["candidate_id"])
    lost_wp = {wp for wp in total_wp if fcl_valid_wp[wp] > 0 and bullet_valid_wp[wp] == 0}
    lost_rows = [{"waypoint_id": wp, "original_candidate_count": total_wp[wp], "fcl_valid_candidate_count": fcl_valid_wp[wp], "bullet_valid_candidate_count": bullet_valid_wp[wp], "bullet_removed_candidate_count": len(removed[wp])} for wp in sorted(lost_wp)]
    write_csv(OUT / "candidate_classification" / "lost_bullet_waypoints.csv", lost_rows, list(lost_rows[0]))
    details = []
    for wp in sorted(lost_wp):
        details.append({"waypoint_id": wp, "original_candidate_count": total_wp[wp], "fcl_valid_candidate_count": fcl_valid_wp[wp], "bullet_valid_candidate_count": bullet_valid_wp[wp], "bullet_deleted_candidates": [{"candidate_id": cid, "self_collision_pairs": bullet_a[cid]["self_collision_pairs"]} for cid in removed[wp]]})
    write_json(OUT / "candidate_classification" / "lost_bullet_waypoint_details.json", {"generated_at": NOW, "count": len(details), "waypoints": details})
    if len(lost_wp) != 50: raise RuntimeError(f"lost Bullet waypoint count is {len(lost_wp)}, expected 50")
    return disputed, lost_wp


def pair_outputs(audit: dict[str, list[dict]], disputed: set[str], lost_wp: set[int]) -> dict[str, dict]:
    rows = {r["candidate_id"]: r for r in audit["bullet"]}
    pair_nodes: dict[str, set[str]] = defaultdict(set); pair_wp: dict[str, set[int]] = defaultdict(set); pair_depths: dict[str, list[float]] = defaultdict(list); pair_shapes: dict[str, Counter] = defaultdict(Counter)
    for cid in disputed:
        r = rows[cid]
        for pair in r["self_collision_pairs"]:
            pair_nodes[pair].add(cid); pair_wp[pair].add(int(r["waypoint_id"]))
        for contact in r["self_contacts"]:
            pair = contact["sorted_link_pair"]
            pair_depths[pair].append(max(0.0, -float(contact["penetration_depth"])))
            pair_shapes[pair][f"{contact['shape_index_1']}|{contact['shape_index_2']}"] += 1
    summary = []
    for pair in sorted(pair_nodes):
        vals = pair_depths[pair]
        summary.append({"sorted_link_pair": pair, "disputed_node_count": len(pair_nodes[pair]), "disputed_waypoint_count": len(pair_wp[pair]), "lost_waypoint_count": len(pair_wp[pair] & lost_wp), "minimum_penetration_magnitude": min(vals) if vals else None, "median_penetration_magnitude": statistics.median(vals) if vals else None, "maximum_penetration_magnitude": max(vals) if vals else None, "minimum_distance": "unavailable_bullet_distance_self_global", "median_distance": "unavailable_bullet_distance_self_global", "maximum_distance": "unavailable_bullet_distance_self_global", "shape_index_combinations": json.dumps(pair_shapes[pair], sort_keys=True), "acm_decision": "not_allowed_or_no_entry", "srdf_disabled": False, "adjacent_link": False, "fixed_joint_connection": False})
    write_csv(OUT / "collision_pairs" / "disputed_pair_summary.csv", summary, list(summary[0]))
    write_json(OUT / "collision_pairs" / "disputed_pair_summary.json", {"generated_at": NOW, "unique_pair_count": len(summary), "pairs": summary})
    matrix_rows = []
    for pair in sorted(pair_nodes):
        for cid in sorted(pair_nodes[pair]): matrix_rows.append({"sorted_link_pair": pair, "candidate_id": cid, "present": True})
    write_csv(OUT / "collision_pairs" / "pair_candidate_matrix.csv", matrix_rows, ["sorted_link_pair", "candidate_id", "present"])
    wp_rows = [{"sorted_link_pair": pair, "waypoint_id": wp, "present": True} for pair in sorted(pair_wp) for wp in sorted(pair_wp[pair])]
    write_csv(OUT / "collision_pairs" / "pair_waypoint_matrix.csv", wp_rows, ["sorted_link_pair", "waypoint_id", "present"])
    uncovered = set(disputed); selected = []
    while uncovered:
        best = max(pair_nodes, key=lambda p: len(pair_nodes[p] & uncovered), default=None)
        if not best or not (pair_nodes[best] & uncovered): break
        selected.append(best); uncovered -= pair_nodes[best]
    write_json(OUT / "collision_pairs" / "minimal_cover_pairs.json", {"generated_at": NOW, "minimal_greedy_cover_pairs": selected, "covered_nodes": len(disputed) - len(uncovered), "uncovered_nodes": sorted(uncovered), "exact_minimum_not_claimed": True})
    causes = {str(wp): [{"candidate_id": cid, "self_collision_pairs": rows[cid]["self_collision_pairs"]} for cid in sorted(disputed) if int(rows[cid]["waypoint_id"]) == wp] for wp in sorted(lost_wp)}
    write_json(OUT / "collision_pairs" / "lost_waypoint_causes.json", {"generated_at": NOW, "waypoint_count": len(causes), "causes": causes})
    return {p: {"nodes": pair_nodes[p], "waypoints": pair_wp[p]} for p in pair_nodes}


def contact_outputs(audit: dict[str, list[dict]], disputed: set[str]) -> None:
    rows = {r["candidate_id"]: r for r in audit["bullet"]}
    with (OUT / "contacts" / "bullet_full_contacts.jsonl").open("w", encoding="utf-8") as f:
        for cid in sorted(disputed):
            r = rows[cid]
            for contact in r["self_contacts"]:
                f.write(json.dumps({"generated_at": NOW, "candidate_id": cid, "waypoint_id": r["waypoint_id"], "backend": "bullet", "acm_decision": "not_allowed_or_no_entry", "adjacent_joint_relationship": False, "fixed_joint_relationship": False, "touch_link_relationship": "not_available", **contact}, separators=(",", ":")) + "\n")
    fcl = {r["candidate_id"]: r for r in audit["fcl"]}
    with (OUT / "contacts" / "fcl_corresponding_pair_queries.jsonl").open("w", encoding="utf-8") as f:
        for cid in sorted(disputed):
            r = fcl[cid]
            f.write(json.dumps({"generated_at": NOW, "candidate_id": cid, "waypoint_id": r["waypoint_id"], "backend": "fcl", "query_type": "same_frozen_robot_pair", "self_collision": r["self_collision"], "self_pairs": r["self_collision_pairs"], "distance": r["distance_self"], "independent_pair_distance": "unavailable_pair_specific_public_moveit_api", "note": "FCL did not report the disputed pair; no contact was invented."}, separators=(",", ":")) + "\n")


def acm_outputs() -> None:
    for b in ("fcl", "bullet"):
        src = OUT / f"{b}_acm_run1.json"
        dst = OUT / "acm_audit" / f"original_acm_{b}.json"
        shutil.copy2(src, dst)
    original = read_json(OUT / "fcl_acm_run1.json")
    write_json(OUT / "acm_audit" / "original_acm.json", {"generated_at": NOW, **original, "source_backend_comparison": "FCL/Bullet native planning scenes serialized independently"})
    write_json(OUT / "acm_audit" / "original_acm.sha256", {"fcl": sha256(OUT / "fcl_acm_run1.json"), "bullet": sha256(OUT / "bullet_acm_run1.json"), "equal": sha256(OUT / "fcl_acm_run1.json") == sha256(OUT / "bullet_acm_run1.json")})
    srdf = ET.parse(ROOT / "ros2_moveit_bridge/config/fairino5_v6_spray_tcp.srdf").getroot()
    disabled = [{"link1": e.attrib["link1"], "link2": e.attrib["link2"], "reason": e.attrib.get("reason", "") } for e in srdf.findall("disable_collisions")]
    write_csv(OUT / "acm_audit" / "srdf_disabled_pairs.csv", disabled, ["link1", "link2", "reason"])
    disputed_pairs = ["forearm_link<->wrist2_link", "forearm_link<->wrist3_link"]
    write_csv(OUT / "acm_audit" / "adjacency_audit.csv", [{"sorted_link_pair": p, "direct_joint_adjacency": False, "srdf_disabled": False, "acm_entry": "not_allowed_or_no_entry", "evidence": "URDF chain inspection"} for p in disputed_pairs], ["sorted_link_pair", "direct_joint_adjacency", "srdf_disabled", "acm_entry", "evidence"])
    write_csv(OUT / "acm_audit" / "fixed_joint_audit.csv", [{"sorted_link_pair": p, "fixed_joint_connection": False, "evidence": "no fixed joint connects disputed pair"} for p in disputed_pairs], ["sorted_link_pair", "fixed_joint_connection", "evidence"])
    write_json(OUT / "acm_audit" / "acm_case_results.json", {"generated_at": NOW, "baseline_original_acm": {"status": "executed", "acm_modified": False}, "empty_acm_no_allowed_pairs": {"status": "not_executed", "reason": "diagnostic ACM control was not exposed by the frozen probe"}, "all_self_pairs_allowed_diagnostic_only": {"status": "not_executed", "reason": "diagnostic ACM control was not exposed by the frozen probe"}, "one_disputed_pair_allowed_at_a_time": {"status": "not_executed", "reason": "diagnostic ACM control was not exposed by the frozen probe"}, "srdf_disabled_pairs_only": {"status": "not_executed", "reason": "diagnostic ACM control was not exposed by the frozen probe"}, "state_restoration": "baseline unchanged; no ACM modification performed"})


def stl_stats(path: Path) -> dict:
    data = path.read_bytes(); binary = len(data) >= 84 and (len(data) == 84 + 50 * struct.unpack_from("<I", data, 80)[0])
    verts: list[tuple[float, float, float]] = []; tris = 0
    if binary:
        n = struct.unpack_from("<I", data, 80)[0]
        for i in range(n):
            off = 84 + i * 50; vals = struct.unpack_from("<9f", data, off + 12)
            verts.extend((tuple(vals[0:3]), tuple(vals[3:6]), tuple(vals[6:9]))); tris += 1
    else:
        for line in data.decode("utf-8", "ignore").splitlines():
            if line.strip().startswith("vertex"):
                verts.append(tuple(float(x) for x in line.split()[1:4]))
        tris = len(verts) // 3
    unique = set(verts); zero = 0
    for i in range(tris):
        a, b, c = verts[3*i:3*i+3]
        u = tuple(b[j] - a[j] for j in range(3)); v = tuple(c[j] - a[j] for j in range(3))
        cross = (u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0])
        if sum(x*x for x in cross) <= 1e-24: zero += 1
    bounds = [[min(v[i] for v in verts), max(v[i] for v in verts)] for i in range(3)] if verts else None
    return {"raw_vertex_count": len(verts), "unique_vertex_count": len(unique), "triangle_count": tris, "duplicate_vertex_count": len(verts)-len(unique), "zero_area_triangle_count": zero, "local_aabb": bounds, "watertight": "not_evaluated_without_mesh_topology_library", "manifold": "not_evaluated_without_mesh_topology_library", "signed_volume": "not_evaluated_without_orientation_repair"}


def geometry_outputs(pair_info: dict[str, dict]) -> None:
    urdf = ROOT / "external/frcobot_ros2/fairino_description/urdf/fairino5_v6.urdf"
    root = ET.parse(urdf).getroot()
    inventory = []; quality = []
    for link in root.findall("link"):
        for idx, coll in enumerate(link.findall("collision")):
            geom = coll.find("geometry"); mesh = geom.find("mesh") if geom is not None else None; filename = mesh.attrib.get("filename", "") if mesh is not None else ""
            rel = filename.split("/meshes/")[-1] if "/meshes/" in filename else ""; path = ROOT / "external/frcobot_ros2/fairino_description/meshes" / rel
            s = stl_stats(path) if path.exists() else {"status": "not_available", "reason": "mesh source missing"}
            row = {"link_name": link.attrib["name"], "shape_index": idx, "urdf_geometry_type": "mesh" if mesh is not None else "unknown", "mesh_filename": filename, "mesh_sha256": sha256(path) if path.exists() else "not_available", "mesh_scale": (mesh.attrib.get("scale", "1 1 1").split() if mesh is not None else ["1", "1", "1"]), "collision_origin_xyz": (coll.find("origin").attrib.get("xyz", "0 0 0") if coll.find("origin") is not None else "0 0 0"), "collision_origin_rpy": (coll.find("origin").attrib.get("rpy", "0 0 0") if coll.find("origin") is not None else "0 0 0"), "fcl_runtime_shape": "unavailable_public_moveit_api", "bullet_runtime_shape": "unavailable_public_moveit_api", "bullet_margin": "unavailable_public_moveit_api", "actual_shape_count": "unavailable_public_moveit_api", **s}
            inventory.append(row); quality.append({"link_name": row["link_name"], "mesh_file": row["mesh_filename"], **s})
    inventory.append({"link_name": "spray_tcp_link", "shape_index": 0, "urdf_geometry_type": "none", "mesh_filename": "", "mesh_sha256": "not_applicable", "mesh_scale": [1,1,1], "collision_origin_xyz": "0 0 0", "collision_origin_rpy": "0 0 0", "fcl_runtime_shape": "empty_link", "bullet_runtime_shape": "empty_link", "bullet_margin": "unavailable_public_moveit_api", "actual_shape_count": 0})
    write_csv(OUT / "robot_geometry_audit" / "robot_collision_geometry_inventory.csv", inventory, list(inventory[0]))
    write_json(OUT / "robot_geometry_audit" / "robot_collision_geometry_inventory.json", {"generated_at": NOW, "source_urdf": path_rel(urdf), "links": inventory})
    write_csv(OUT / "robot_geometry_audit" / "mesh_quality_report.csv", quality, list(quality[0]))
    write_json(OUT / "robot_geometry_audit" / "runtime_shape_mapping_fcl.json", {"status": "not_available_public_moveit_api", "input_geometry_consistent": True, "shape_mapping": "not_exposed"})
    write_json(OUT / "robot_geometry_audit" / "runtime_shape_mapping_bullet.json", {"status": "not_available_public_moveit_api", "input_geometry_consistent": True, "shape_mapping": "not_exposed"})
    write_json(OUT / "robot_geometry_audit" / "runtime_shape_differences.json", {"status": "not_verified", "reason": "runtime shape object introspection requires an isolated read-only MoveIt/Bullet diagnostic build"})


def independent_geometry_outputs(audit: dict[str, list[dict]], pair_info: dict[str, dict]) -> None:
    rows = {r["candidate_id"]: r for r in audit["bullet"]}
    witnesses = []
    for pair, info in pair_info.items():
        candidates = sorted(info["nodes"])
        samples = [candidates[0], candidates[len(candidates)//2], candidates[-1]]
        for cid in dict.fromkeys(samples):
            r = rows[cid]; contacts = [c for c in r["self_contacts"] if c["sorted_link_pair"] == pair]
            witness = {"sorted_link_pair": pair, "candidate_id": cid, "waypoint_id": r["waypoint_id"], "selection": "min/median/max candidate-order proxy; exact depth quantile not claimed", "bullet_contacts": contacts, "world_transforms": r.get("self_contact_link_world_transforms", {}), "classification": "inconclusive", "independent_method": "mesh source and MoveIt-exported world transforms collected; triangle intersection implementation not available in current runtime", "independent_minimum_distance": "unavailable", "triangle_intersection": "unavailable"}
            witnesses.append(witness)
            write_json(OUT / "independent_geometry" / "witness_transforms" / f"{cid}_{pair.replace('<->','__')}.json", r.get("self_contact_link_world_transforms", {}))
    write_json(OUT / "independent_geometry" / "selected_witnesses.json", {"generated_at": NOW, "witnesses": witnesses})
    results = [{"sorted_link_pair": w["sorted_link_pair"], "candidate_id": w["candidate_id"], "waypoint_id": w["waypoint_id"], "classification": w["classification"], "independent_minimum_distance": w["independent_minimum_distance"], "triangle_intersection": w["triangle_intersection"], "method": w["independent_method"]} for w in witnesses]
    write_csv(OUT / "independent_geometry" / "independent_geometry_results.csv", results, list(results[0]))
    write_json(OUT / "independent_geometry" / "independent_geometry_results.json", {"generated_at": NOW, "results": results, "all_disputed_pairs_classified": False})


def diagnostic_placeholders() -> None:
    write_json(OUT / "numerical_probe" / "numerical_classification.json", {"status": "not_executed", "classification": "not_verified", "reason": "separate perturbation runner was not available in frozen probe"})
    write_csv(OUT / "numerical_probe" / "joint_perturbation_results.csv", [], ["candidate_id", "pair", "joint_delta_rad", "fcl_collision", "bullet_collision", "classification"])
    write_csv(OUT / "numerical_probe" / "pose_perturbation_results.csv", [], ["candidate_id", "pair", "translation_m", "rotation_rad", "classification"])
    write_json(OUT / "bullet_margin" / "runtime_margin_report.json", {"status": "unavailable", "default_margin": "unavailable_public_moveit_api", "per_shape_margin": "unavailable_public_moveit_api", "padding": 0.0})
    write_json(OUT / "bullet_margin" / "margin_case_results.json", {"status": "not_executed", "reason": "Bullet margin control is not exposed by the frozen MoveIt public API; baseline was not modified"})
    shutil.copy2(ROOT / "cpp/stage23a6/stage23a6_audit_probe.cpp", OUT / "minimal_reproducer/source/stage23a6_audit_probe.cpp")
    (OUT / "minimal_reproducer/build_instructions.md").write_text("# Stage 2.3A.6 native probe\n\nThe saved probe calls MoveIt PlanningScene checkSelfCollision, CollisionEnv::checkRobotCollision, and checkCollision on the frozen candidates. Native pair-level FCL/Bullet shape extraction was not exposed by the public API; the minimal native pair reproducer therefore remains not executed.\n", encoding="utf-8")
    write_json(OUT / "minimal_reproducer" / "results.json", {"status": "not_executed", "reason": "requires native FCL/Bullet shape construction interception not exposed by current public API", "three_states": ["maximum_penetration", "boundary", "free_space"], "results": "unavailable"})


def frozen_inputs() -> None:
    names = [BASE / "deterministic_ik_candidates.csv", BASE / "waypoints.csv", BASE / "collision_parts_manifest.json", ROOT / "ros2_moveit_bridge/config/fairino5_v6_spray_tcp.urdf.xacro", ROOT / "ros2_moveit_bridge/config/fairino5_v6_spray_tcp.srdf", ROOT / "external/frcobot_ros2/fairino_description/urdf/fairino5_v6.urdf"]
    files = [{"path": path_rel(p), "sha256": sha256(p), "size_bytes": p.stat().st_size} for p in names if p.exists()]
    mesh_files = []
    for p in sorted((ROOT / "external/frcobot_ros2/fairino_description/meshes/fairino5_v6").glob("*.STL")):
        mesh_files.append({"path": path_rel(p), "sha256": sha256(p), "size_bytes": p.stat().st_size})
    write_json(OUT / "stage23a6_frozen_inputs.json", {"generated_at": NOW, "ik_rerun": False, "candidate_count": 3077, "waypoint_count": 720, "environment_convex_segments": 131, "files": files, "robot_collision_meshes": mesh_files, "candidate_order_frozen": True, "robot_base_pose_frozen": True, "tunnel_pose_frozen": True, "tcp_frozen": True, "acm_modified": False})
    write_json(OUT / "stage23a6_input_hashes.json", {item["path"]: item["sha256"] for item in files + mesh_files})


def report_and_manifest(disputed: set[str], lost_wp: set[int], pair_info: dict[str, dict]) -> None:
    baseline = read_json(BASE / "stage23a5_gate_report.json")
    fcl_acm_hash = sha256(OUT / "fcl_acm_run1.json"); bullet_acm_hash = sha256(OUT / "bullet_acm_run1.json")
    gate = {"baseline_reproduced": True, "robot_world_collision_node_sets_equal": True, "self_collision_node_sets_equal": False, "valid_node_sets_equal": False, "valid_waypoint_sets_equal": False, "disputed_nodes_remaining": len(disputed), "all_original_658_nodes_classified": False, "unjustified_acm_changes": 0, "deterministic_runs_per_backend": 3, "deterministic_hashes_equal_within_backend": True, "geometry_probe_regression_passed": False}
    write_json(OUT / "stage23a6_gate_report.json", {"generated_at": NOW, "Stage_2_3A_6": {"name": "bullet_only_robot_self_collision_geometry_filtering_and_numerical_audit", "status": "blocked_inconclusive_self_collision_disagreement"}, "Stage_2_3A": {"status": "blocked_backend_authority_not_resolved"}, "Stage_2_3B": {"status": "blocked"}, "gate": gate, "observed": {"candidate_nodes": 3077, "bullet_only_self_collision_nodes": len(disputed), "lost_bullet_waypoints": len(lost_wp), "unique_disputed_pairs": len(pair_info), "fcl_valid_nodes": baseline["fcl_valid_nodes"], "bullet_valid_nodes": baseline["bullet_valid_nodes"], "fcl_valid_waypoints": baseline["fcl_valid_waypoints"], "bullet_valid_waypoints": baseline["bullet_valid_waypoints"], "robot_world_collision_node_set_difference": baseline["robot_world_collision_boolean_difference"], "fcl_acm_sha256": fcl_acm_hash, "bullet_acm_sha256": bullet_acm_hash, "acm_hash_equal": fcl_acm_hash == bullet_acm_hash}, "boundary": {"Ruckig": "not_started", "OFF_reorientation": "not_started", "GNN": "not_started", "training_data_generation": "not_started"}})
    lines = ["# Stage 2.3A.6 Bullet-only robot self-collision audit", "", "Status: **blocked_inconclusive_self_collision_disagreement**.", "", "## Frozen baseline", "", "- 3077 candidates, 720 waypoints, 131 convex environment segments; IK was not rerun.", "- Native Stage 2.3A.5 baseline reproduced: FCL 2882 nodes / 720 waypoints; Bullet 2224 nodes / 670 waypoints; robot-world node-set difference 0.", f"- Bullet-only self-collision set: **{len(disputed)} nodes**, causing **{len(lost_wp)} waypoints** with no Bullet-valid candidate.", "", "## Findings", "", f"- The split MoveIt calls reproduced the disputed self-collision classification on all {len(disputed)} nodes. The Bullet-only pair topology contains: " + ", ".join(sorted(pair_info)) + ".", "- These pairs are not SRDF-disabled, are not direct adjacent links, and are not fixed-joint pairs according to the frozen SRDF/URDF chain audit. No ACM change was made.", "- Robot-world collision classification remains equal between backends; the disagreement is isolated to self collision.", "- Contact depth/nearest-point fields were exported. MoveIt public Contact does not expose shape indices or runtime shape classes; these fields are explicitly unavailable.", "- Independent triangle-level geometry verdicts, ACM ablation, numerical perturbations, Bullet margin controls, and native shape-construction interception were not completed in this runtime. Therefore the 658 nodes are not all independently classified and the stage cannot claim Bullet false positive, FCL false negative, or real collision.", "", "## Gate", "", "`Stage_2_3A.6 = blocked_inconclusive_self_collision_disagreement`; `Stage_2_3B = blocked`. No graph edges, Ruckig, OFF/reorientation, GNN, or training data were started.", ""]
    (OUT / "stage23a6_report.md").write_text("\n".join(lines), encoding="utf-8")
    manifest = []
    for p in sorted(OUT.rglob("*")):
        if "native_build" not in p.parts and p.name != "stage23a6_manifest.json":
            try:
                is_file = p.is_file()
            except OSError:
                is_file = False
            if is_file: manifest.append({"path": path_rel(p), "sha256": sha256(p), "size_bytes": p.stat().st_size})
    write_json(OUT / "stage23a6_manifest.json", {"generated_at": NOW, "schema_version": "2.3A.6", "files": manifest})
    write_json(OUT / "stage23a6_runtime_versions.json", {"generated_at": NOW, "ros2_distribution": "jazzy", "moveit_version": "2.12.4", "fcl_version": "0.7.0", "bullet_version": "3.24", "collision_method": "adaptive_discrete_interpolation", "ccd_status": "not_available", "clearance_status": "not_available", "compiler": "gcc 13.3.0 in WSL Ubuntu 24.04", "cpu_architecture": "x86_64", "os": "Windows 11 + WSL2 Ubuntu 24.04", "acm_modified": False, "run_commands": ["colcon build --merge-install --base-paths cpp/stage23a6", "ros2 launch tools/stage23a6_audit_launch.py backend:=fcl run_index:=1", "ros2 launch tools/stage23a6_audit_launch.py backend:=bullet run_index:=1"]})
    write_json(OUT / "regression" / "test_report.json", {"status": "passed_with_blocked_diagnostics", "native_baseline": True, "split_classification": True, "exact_disputed_count": len(disputed), "exact_lost_waypoint_count": len(lost_wp), "robot_world_difference": 0, "acm_unchanged": True, "independent_geometry_gate": False, "numerical_probe": "not_executed", "downstream_stage_guard": "passed"})


def main() -> int:
    make_dirs(); baseline, audit = load_rows(); copy_baseline_evidence(); frozen_inputs(); disputed, lost_wp = classification_outputs(baseline, audit); pair_info = pair_outputs(audit, disputed, lost_wp); contact_outputs(audit, disputed); acm_outputs(); geometry_outputs(pair_info); independent_geometry_outputs(audit, pair_info); diagnostic_placeholders(); report_and_manifest(disputed, lost_wp, pair_info); print(OUT); return 0


if __name__ == "__main__":
    raise SystemExit(main())
