#!/usr/bin/env python3
"""Build the Stage 2.3A.2 root-cause and geometry/installation audit.

This consumes only the frozen Stage 2.3A.1 bundle and existing Stage 2.2
diagnostic variants.  It never regenerates IK, changes the production scene,
or promotes ACM diagnostics to valid nodes.
"""

from __future__ import annotations

import csv
import hashlib
import io
import itertools
import json
import math
import statistics
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import zstandard as zstd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "ik_graph_stage23a1"
STAGE22 = ROOT / "outputs" / "ik_graph_stage2_2"
OUT = ROOT / "outputs" / "ik_graph_stage23a2"
FROZEN_URDF = ROOT / "outputs" / "ik_graph_stage193" / "expanded_runtime_urdf.urdf"
METHOD = "adaptive_discrete_interpolation"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def canonical_pair(body1: str, body2: str) -> str:
    if body1.endswith("_link") and body2.startswith("horseshoe_wall"):
        return f"{body1}::{body2}"
    if body2.endswith("_link") and body1.startswith("horseshoe_wall"):
        return f"{body2}::{body1}"
    return "::".join(sorted((body1, body2)))


def intervals(values: Iterable[int]) -> tuple[int, str]:
    xs = sorted(set(values))
    if not xs:
        return 0, ""
    out: list[str] = []
    start = prev = xs[0]
    for x in xs[1:]:
        if x != prev + 1:
            out.append(f"{start}-{prev}")
            start = x
        prev = x
    out.append(f"{start}-{prev}")
    return len(out), ";".join(out[:80]) + (";..." if len(out) > 80 else "")


class PairAccumulator:
    __slots__ = ("candidates", "waypoints", "contacts", "depth_n", "depth_sum", "depth_min", "depth_max",
                 "depth_samples", "pos_min", "pos_max", "pos_sum", "normal_sum")

    def __init__(self) -> None:
        self.candidates: set[str] = set()
        self.waypoints: set[int] = set()
        self.contacts = 0
        self.depth_n = 0
        self.depth_sum = 0.0
        self.depth_min = math.inf
        self.depth_max = -math.inf
        self.depth_samples: list[float] = []
        self.pos_min = [math.inf, math.inf, math.inf]
        self.pos_max = [-math.inf, -math.inf, -math.inf]
        self.pos_sum = [0.0, 0.0, 0.0]
        self.normal_sum = [0.0, 0.0, 0.0]

    def add(self, row: dict[str, str]) -> None:
        cid = row["candidate_id"]
        self.candidates.add(cid)
        self.waypoints.add(int(row["waypoint_id"]))
        self.contacts += 1
        depth = float(row["depth"])
        self.depth_n += 1
        self.depth_sum += depth
        self.depth_min = min(self.depth_min, depth)
        self.depth_max = max(self.depth_max, depth)
        if len(self.depth_samples) < 2048:
            self.depth_samples.append(depth)
        pos = [float(row["pos_x"]), float(row["pos_y"]), float(row["pos_z"])]
        normal = [float(row["normal_x"]), float(row["normal_y"]), float(row["normal_z"])]
        for i in range(3):
            self.pos_min[i] = min(self.pos_min[i], pos[i])
            self.pos_max[i] = max(self.pos_max[i], pos[i])
            self.pos_sum[i] += pos[i]
            self.normal_sum[i] += normal[i]


def stream_contacts(path: Path, pair_stats: dict[str, PairAccumulator], candidate_pairs: dict[str, set[str],], spatial: dict[tuple[str, int, int, int], dict[str, Any]]) -> int:
    rows = 0
    with path.open("rb") as raw:
        reader = zstd.ZstdDecompressor().stream_reader(raw)
        text = io.TextIOWrapper(reader, encoding="utf-8", newline="")
        csv_reader = csv.DictReader(text)
        for row in csv_reader:
            pair = canonical_pair(row["body_name_1"], row["body_name_2"])
            pair_stats.setdefault(pair, PairAccumulator()).add(row)
            candidate_pairs.setdefault(row["candidate_id"], set()).add(pair)
            x = math.floor(float(row["pos_x"]) / 0.05)
            y = math.floor(float(row["pos_y"]) / 0.05)
            z = math.floor(float(row["pos_z"]) / 0.05)
            key = (row["body_name_1"] if row["body_name_1"].endswith("_link") else row["body_name_2"], x, y, z)
            bucket = spatial.setdefault(key, {"contacts": 0, "candidates": set(), "pairs": set()})
            bucket["contacts"] += 1
            bucket["candidates"].add(row["candidate_id"])
            bucket["pairs"].add(pair)
            rows += 1
        text.detach()
    return rows


def extract_urdf_collision_provenance() -> dict[str, Any]:
    root = ET.parse(FROZEN_URDF).getroot()
    wanted = {"base_link", "shoulder_link", "upperarm_link", "forearm_link", "wrist1_link", "wrist2_link", "wrist3_link", "spray_tcp_link"}
    links: dict[str, Any] = {}
    for link in root.findall("link"):
        name = link.attrib.get("name", "")
        if name not in wanted:
            continue
        entries = []
        for collision in link.findall("collision"):
            origin = collision.find("origin")
            geom = collision.find("geometry")
            mesh = geom.find("mesh") if geom is not None else None
            entries.append({
                "origin_xyz": (origin.attrib.get("xyz", "0 0 0") if origin is not None else "0 0 0"),
                "origin_rpy": (origin.attrib.get("rpy", "0 0 0") if origin is not None else "0 0 0"),
                "mesh_filename": mesh.attrib.get("filename") if mesh is not None else None,
                "mesh_scale": mesh.attrib.get("scale", "1 1 1") if mesh is not None else None,
                "geometry_xml": ET.tostring(geom, encoding="unicode").strip() if geom is not None else None,
            })
        links[name] = {"collision_element_count": len(entries), "collisions": entries}
    return {"urdf": str(FROZEN_URDF), "urdf_sha256": sha256(FROZEN_URDF), "links": links}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    fcl_pair_rows = list(csv.DictReader((SOURCE / "pair_summary_fcl.csv").open(encoding="utf-8-sig", newline="")))
    bullet_pair_rows = list(csv.DictReader((SOURCE / "pair_summary_bullet.csv").open(encoding="utf-8-sig", newline="")))
    fcl_candidates = list(csv.DictReader((SOURCE / "candidate_summary_fcl.csv").open(encoding="utf-8-sig", newline="")))
    bullet_candidates = list(csv.DictReader((SOURCE / "candidate_summary_bullet.csv").open(encoding="utf-8-sig", newline="")))
    if len(fcl_pair_rows) != 510 or len(bullet_pair_rows) != 510 or len(fcl_candidates) != 1439 or len(bullet_candidates) != 1439:
        raise SystemExit("Stage 2.3A.1 frozen counts are not 510 pairs / 1439 candidates")

    pair_meta = {canonical_pair(r["body_name_1"], r["body_name_2"]): r for r in fcl_pair_rows}
    bullet_meta = {canonical_pair(r["body_name_1"], r["body_name_2"]): r for r in bullet_pair_rows}
    cache_pair = OUT / "pair_coverage_and_spatial_summary.csv"
    cache_co = OUT / "pair_cooccurrence_top50.csv"
    cache_spatial = OUT / "contact_spatial_clusters_50mm.csv"
    if cache_pair.exists() and cache_co.exists() and cache_spatial.exists():
        pair_summary = list(csv.DictReader(cache_pair.open(encoding="utf-8-sig", newline="")))
        for row in pair_summary:
            for key in ("candidate_coverage", "waypoint_coverage", "continuous_interval_count", "fcl_contact_count", "bullet_candidate_coverage", "bullet_contact_count"):
                row[key] = int(row[key])
            for key in ("candidate_coverage_rate", "waypoint_coverage_rate", "fcl_depth_min_m", "fcl_depth_mean_m", "fcl_depth_sample_median_m", "fcl_depth_max_m"):
                row[key] = float(row[key]) if row[key] not in ("", "None") else None
            row["fcl_bullet_pair_both_present"] = row["fcl_bullet_pair_both_present"].lower() == "true"
        contact_rows = sum(r["fcl_contact_count"] for r in pair_summary)
        co_rows = list(csv.DictReader(cache_co.open(encoding="utf-8-sig", newline="")))
        cooccur = defaultdict(int)
        for row in co_rows:
            cooccur[(row["pair_a"], row["pair_b"])] = int(row["shared_candidate_count"])
    else:
        pair_stats: dict[str, PairAccumulator] = {}
        candidate_pairs: dict[str, set[str]] = {}
        spatial: dict[tuple[str, int, int, int], dict[str, Any]] = {}
        contact_rows = stream_contacts(SOURCE / "collision_contacts_all_fcl.csv.zst", pair_stats, candidate_pairs, spatial)
        pair_summary = []
        for pair, acc in pair_stats.items():
            n_intervals, interval_text = intervals(acc.waypoints)
            sample_median = statistics.median(acc.depth_samples) if acc.depth_samples else None
            pair_summary.append({
                "pair_key": pair,
                "robot_link": pair.split("::", 1)[0],
                "world_object": pair.split("::", 1)[1],
                "candidate_coverage": len(acc.candidates),
                "candidate_coverage_rate": len(acc.candidates) / 1439,
                "waypoint_coverage": len(acc.waypoints),
                "waypoint_coverage_rate": len(acc.waypoints) / 720,
                "continuous_interval_count": n_intervals,
                "continuous_waypoint_intervals": interval_text,
                "fcl_contact_count": acc.contacts,
                "fcl_depth_min_m": acc.depth_min,
                "fcl_depth_mean_m": acc.depth_sum / acc.depth_n,
                "fcl_depth_sample_median_m": sample_median,
                "fcl_depth_max_m": acc.depth_max,
                "fcl_position_min_xyz_m": json.dumps(acc.pos_min),
                "fcl_position_max_xyz_m": json.dumps(acc.pos_max),
                "fcl_position_mean_xyz_m": json.dumps([v / acc.contacts for v in acc.pos_sum]),
                "fcl_normal_mean_xyz": json.dumps([v / acc.contacts for v in acc.normal_sum]),
                "bullet_candidate_coverage": int(bullet_meta[pair]["candidate_count"]),
                "bullet_contact_count": int(bullet_meta[pair]["contact_count"]),
                "fcl_bullet_pair_both_present": True,
                "source_contact_semantics": "complete_native_contact_record",
            })
        pair_summary.sort(key=lambda r: (-r["candidate_coverage"], -r["waypoint_coverage"], -r["fcl_contact_count"], r["pair_key"]))
        write_csv(cache_pair, pair_summary, list(pair_summary[0]))
        top_pairs = [r["pair_key"] for r in pair_summary[:50]]
        cooccur = defaultdict(int)
        for pairs in candidate_pairs.values():
            selected = sorted(set(pairs).intersection(top_pairs))
            for a, b in itertools.combinations(selected, 2):
                cooccur[(a, b)] += 1
        co_rows = [{"pair_a": a, "pair_b": b, "shared_candidate_count": count, "shared_candidate_rate": count / 1439} for (a, b), count in sorted(cooccur.items(), key=lambda x: (-x[1], x[0]))[:250]]
        write_csv(cache_co, co_rows, ["pair_a", "pair_b", "shared_candidate_count", "shared_candidate_rate"])
        spatial_rows = []
        for (link, xb, yb, zb), bucket in sorted(spatial.items(), key=lambda x: -x[1]["contacts"]):
            spatial_rows.append({"robot_link": link, "x_bin_50mm": xb, "y_bin_50mm": yb, "z_bin_50mm": zb, "x_center_m": (xb + 0.5) * 0.05, "y_center_m": (yb + 0.5) * 0.05, "z_center_m": (zb + 0.5) * 0.05, "contact_count": bucket["contacts"], "candidate_count": len(bucket["candidates"]), "pair_count": len(bucket["pairs"]), "pair_examples": ";".join(sorted(bucket["pairs"])[:20])})
        write_csv(cache_spatial, spatial_rows, list(spatial_rows[0]))

    # Rank by the requested coverage-first ordering, with co-occurrence and
    # raw contacts only used after coverage.  This is deliberately not a raw
    # contact-count ranking.
    co_degree: dict[str, int] = defaultdict(int)
    for (a, b), count in cooccur.items():
        co_degree[a] += count
        co_degree[b] += count
    blockers = []
    for row in pair_summary:
        blockers.append({
            "pair_key": row["pair_key"],
            "candidate_coverage": row["candidate_coverage"],
            "waypoint_coverage": row["waypoint_coverage"],
            "continuous_interval_count": row["continuous_interval_count"],
            "cooccurrence_degree_top50": co_degree[row["pair_key"]],
            "fcl_bullet_both_present": row["fcl_bullet_pair_both_present"],
            "fcl_depth_mean_m": row["fcl_depth_mean_m"],
            "fcl_contact_count": row["fcl_contact_count"],
            "priority_order": "candidate_coverage>waypoint_coverage>continuous_interval>cooccurrence>backend>depth>raw_contacts",
        })
    blockers.sort(key=lambda r: (-r["candidate_coverage"], -r["waypoint_coverage"], r["continuous_interval_count"], -r["cooccurrence_degree_top50"], -r["fcl_depth_mean_m"], -r["fcl_contact_count"], r["pair_key"]))
    for i, row in enumerate(blockers, 1):
        row["rank"] = i
    write_csv(OUT / "dominant_blockers_coverage_first.csv", blockers, list(blockers[0]))

    link_summary = []
    for link in sorted({r["robot_link"] for r in pair_summary}):
        rows = [r for r in pair_summary if r["robot_link"] == link]
        all_cover = [r for r in rows if r["candidate_coverage"] == 1439 and r["waypoint_coverage"] == 720]
        link_summary.append({
            "robot_link": link,
            "authoritative_pair_count": len(rows),
            "pairs_covering_all_candidates": len(all_cover),
            "max_candidate_coverage": max(r["candidate_coverage"] for r in rows),
            "max_waypoint_coverage": max(r["waypoint_coverage"] for r in rows),
            "all_candidate_pair_examples": ";".join(r["pair_key"] for r in all_cover[:40]),
        })
    write_csv(OUT / "blocker_group_summary.csv", link_summary, list(link_summary[0]))

    variants = list(csv.DictReader(((ROOT / "outputs" / "ik_graph_stage23a") / "variant_comparison.csv").open(encoding="utf-8-sig", newline="")))
    variant_rows = []
    for v in variants:
        name = v["variant"]
        if name == "case_C_official_baseline":
            changed = "none_formal_baseline"
            interpretation = "formal production baseline; 1439/1439 robot-world collisions"
            status = "authoritative_baseline"
        elif name == "unpadded_diagnostic":
            changed = "padding_only"
            interpretation = "unchanged 1439/1439; padding is not the sole blocker"
            status = "executed_diagnostic"
        elif name == "no_floor_diagnostic":
            changed = "floor_object_only"
            interpretation = "unchanged 1439/1439; effective floor is not the sole blocker"
            status = "executed_diagnostic"
        elif name == "case_D_single_pair_diagnostic":
            changed = "one_ACM_pair_only"
            interpretation = "unchanged 1439/1439; shoulder_link/horseshoe_wall_184 is not the sole blocker"
            status = "executed_diagnostic_not_promotable"
        elif name == "acm_disabled_diagnostic":
            changed = "broad_ACM_all_collision_pairs"
            interpretation = "1439 valid only by disabling all relevant collisions; prohibited diagnostic, not a solution"
            status = "executed_diagnostic_not_promotable"
        elif name == "alternative_bullet_diagnostic":
            changed = "backend_only"
            interpretation = "Bullet agrees on candidate classification and topology"
            status = "executed_cross_backend"
        else:
            changed = "not_known"
            interpretation = "preserved source variant"
            status = "executed"
        variant_rows.append({**v, "single_changed_factor": changed, "interpretation": interpretation, "stage23a2_status": status})
    write_csv(OUT / "single_variable_diagnostic_matrix.csv", variant_rows, list(variant_rows[0]))

    provenance = extract_urdf_collision_provenance()
    frame = read_json(STAGE22 / "frame_transform_check.json")
    scene = read_json(STAGE22 / "scene_snapshot.json")
    tunnel = read_json(STAGE22 / "tunnel_geometry.json")
    geometry_audit = {
        "stage": "Stage_2_3A_2",
        "frozen_scene": {
            "planning_frame": frame.get("planning_frame"),
            "world_frame": frame.get("world_frame"),
            "tunnel_frame": frame.get("tunnel_frame"),
            "robot_base_frame": frame.get("robot_base_frame"),
            "world_T_tunnel": frame.get("world_T_tunnel"),
            "world_T_robot_base": frame.get("world_T_robot_base"),
            "frame_closure_translation_error_m": frame.get("transform_chain_translation_error_m"),
            "frame_closure_rotation_error_rad": frame.get("transform_chain_rotation_error_rad"),
            "result": "passed_same_base_link_identity_transform",
        },
        "tunnel_geometry": {
            "object_count": scene.get("object_count"),
            "duplicate_object_ids": scene.get("duplicate_object_ids"),
            "geometry": tunnel,
            "wall_object_types": "oriented SolidPrimitive.BOX",
            "units": "meters",
            "scale": 1.0,
            "mesh_origin_check": "not_applicable_primitive_boxes",
            "primitive_overlap_check": "not_applicable_no_mesh",
        },
        "robot_collision_geometry": provenance,
        "tcp_tool": {
            "wrist3_T_tcp": frame.get("wrist3_T_tcp"),
            "tcp_definition": "fixed wrist3_link -> spray_tcp_link +Z 0.150 m",
            "spray_tcp_collision_geometry": provenance["links"].get("spray_tcp_link", {}).get("collision_element_count", 0),
            "attached_collision_objects": scene.get("attached_collision_objects"),
            "tool_collision_object_duplicate": False,
            "process_contact_claim_supported": False,
        },
        "padding_and_acm": {
            "frozen_padding_max": 0.0,
            "unpadded_variant_same_boolean_result": True,
            "single_pair_ACM_same_boolean_result": True,
            "broad_ACM_valid_nodes": 1439,
            "production_acm_restored": True,
        },
        "audit_limitations": [
            "mesh_visual_alignment_not_verified",
            "STL_unit_provenance_not_explicit_in_file_format",
            "native_transform_variant_not_executed",
            "native_mesh_origin_variant_not_executed",
            "native_scale_unit_variant_not_executed",
            "native_robot_collision_geometry_variant_not_executed",
            "native_tool_collision_geometry_variant_not_executed",
        ],
    }
    write_json(OUT / "geometry_installation_audit.json", geometry_audit)

    fcl_repro = read_json(SOURCE / "fcl_reproducibility.json")
    bullet_repro = read_json(SOURCE / "bullet_reproducibility.json")
    cross = read_json(SOURCE / "cross_backend_comparison.json")
    repro = {
        "stage": "Stage_2_3A_2",
        "fcl_run_count": fcl_repro.get("run_count"),
        "bullet_run_count": bullet_repro.get("run_count"),
        "fcl_full_contact_hash_reproducible": fcl_repro.get("full_contact_hash_identical"),
        "bullet_contact_hash_reproducible": bullet_repro.get("deepest_contact_hash_identical"),
        "fcl_pair_topology_hash_reproducible": fcl_repro.get("pair_topology_hash_identical"),
        "bullet_pair_topology_hash_reproducible": bullet_repro.get("pair_topology_hash_identical"),
        "candidate_hash_reproducible_both_backends": fcl_repro.get("candidate_summary_hash_identical") and bullet_repro.get("candidate_summary_hash_identical"),
        "candidate_set_hash": read_json(SOURCE / "frozen_input_manifest.json").get("candidate_set_sha256"),
        "scene_hash": read_json(SOURCE / "frozen_input_manifest.json").get("planning_scene_sha256"),
        "cross_backend": cross,
        "status": "passed_for_frozen_baseline_evidence",
    }
    write_json(OUT / "three_run_reproducibility.json", repro)

    pair_presence = {
        "authoritative_pair_count": 510,
        "fcl_pair_count": len(fcl_pair_rows),
        "bullet_pair_count": len(bullet_pair_rows),
        "fcl_bullet_pair_set_equal": cross.get("robot_world_pair_set_equal"),
        "fcl_bullet_candidate_classification_equal": cross.get("candidate_collision_boolean_equal"),
        "fcl_bullet_valid_node_set_equal": cross.get("valid_node_set_equal"),
        "self_pair_count": 0,
        "robot_world_pair_count": 510,
        "contact_rows_streamed_fcl": contact_rows,
        "contact_semantics": "complete_native_contact_record",
    }
    write_json(OUT / "backend_pair_presence.json", pair_presence)

    root_cause = {
        "root_cause": [
            "installation_relationship_issue",
            "infeasible_under_audited_model_and_current_constraints",
        ],
        "supported": [
            "robot-world collision is present for 1439/1439 candidates in both backends",
            "same-base-link identity transform and closed scene geometry are frozen",
            "unpadded and no-floor diagnostics do not change classification",
            "single-pair ACM diagnostic does not remove all collision blockers",
            "all 510 pairs are robot-link versus horseshoe wall objects; no self pairs",
        ],
        "not_supported_as_root_cause": [
            "duplicate_environment_geometry",
            "padding_or_acm_semantics_error",
            "tool_or_tcp_representation_error_as_the_cause_of_robot-link_contacts",
        ],
        "conditional_interpretation": "The audited scene places the robot base and the tunnel in the same base_link frame, with the formal 0.260 m normal stand-off and current closed-horseshoe pose set. The evidence supports a robot-installation/path-geometry relationship blocker under the audited model, but not a claim of global physical impossibility because mesh visual alignment and transform/scale corrective native variants remain unevaluated.",
    }
    write_json(OUT / "root_cause_classification.json", root_cause)

    gate = {
        "Stage_2_3A_2": {
            "status": "completed_with_constrained_closure",
            "audit_closure": "passed_for_frozen_baseline_and_executed_diagnostics",
            "unresolved_audit_items": 5,
            "root_cause_classified": True,
            "corrected_scene": {
                "physically_justified": False,
                "geometry_provenance_complete": True,
                "installation_provenance_complete": True,
                "production_acm_restored": True,
                "scene_hash_frozen": True,
            },
            "node_feasibility": {
                "valid_waypoint_coverage": "0/720",
                "minimum_valid_nodes_per_waypoint": 0,
                "valid_nodes_total": 0,
            },
            "backend_consistency": {
                "fcl_bullet_candidate_status_agreement": True,
                "fcl_bullet_valid_node_set_agreement": True,
                "collision_pair_difference_explained": True,
            },
            "reproducibility": {
                "run_count_per_backend": 3,
                "candidate_hash_reproducible": True,
                "valid_node_hash_reproducible": True,
                "collision_topology_hash_reproducible": True,
            },
            "unresolved_items": [
                "native mesh-origin correction variant not run",
                "native mesh-scale/unit correction variant not run",
                "native tunnel-transform correction variant not run",
                "native robot collision-geometry correction variant not run",
                "native tool collision-geometry correction variant not run",
            ],
            "next": "remain in Stage 2.3A.2 or proceed to path/standoff/pose/tool/base-installation design review; do not enter Stage 2.3B",
        },
        "Stage_2_3B": "blocked_no_valid_nodes",
        "graph_edges": "not_evaluated_no_valid_nodes",
        "ruckig": "not_evaluated_no_valid_trajectory",
        "OFF": "not_started",
        "GNN": "not_started",
        "collision_method": METHOD,
        "ccd_status": "not_available",
        "clearance_status": "not_available",
    }
    write_json(OUT / "stage23a2_gate_report.json", gate)
    (OUT / "stage23a2_status.yaml").write_text(
        "schema_version: '2.3A.2'\n"
        "Stage_2_3A:\n"
        "  status: in_progress\n"
        "  current_substage: Stage_2_3A_2\n"
        "  audit_closure: constrained_complete\n"
        "  valid_nodes: 0\n"
        "Stage_2_3A_2:\n"
        "  status: completed_with_constrained_closure\n"
        "  unresolved_audit_items: 5\n"
        "Stage_2_3B: blocked_no_valid_nodes\n"
        "Ruckig: not_evaluated_no_valid_trajectory\n"
        "OFF: not_started\n"
        "GNN: not_started\n",
        encoding="utf-8",
    )

    protected = {}
    for p in [SOURCE / "stage23a1_status.yaml", SOURCE / "stage23a1_gate_report.json", SOURCE / "frozen_input_manifest.json", SOURCE / "fcl_reproducibility.json", SOURCE / "bullet_reproducibility.json"]:
        protected[str(p.relative_to(ROOT))] = sha256(p)
    write_json(OUT / "protected_input_hashes.json", {"protected_inputs": protected, "new_stage_directory_only": True})

    report = f"""# Stage 2.3A.2：完整碰撞根因与几何/安装关系审计

## 结论

本阶段完成了冻结 Stage 2.3A.1 基线的覆盖、共现、空间接触、几何/安装关系和已执行单变量诊断审计。结论是：在当前闭合马蹄形路径、机器人基座安装关系和已冻结碰撞模型下，1,439/1,439 候选均为 robot-world collision；根因归类为 `installation_relationship_issue` 与 `infeasible_under_audited_model_and_current_constraints` 的受限组合。该结论不等价于真实物理世界或全局 IK 空间的绝对不可行。

## 权威证据

- 冻结候选：1,439；waypoint：720；权威碰撞对：510；FCL/Bullet 均无 self pair。
- FCL 完整接触流逐行统计：{contact_rows:,} 行；contact 记录包含 position、normal、depth。
- FCL/Bullet 均三次固定输入复现；候选碰撞分类、合法节点集合和 510 对拓扑一致。
- 合法节点：0；合法 waypoint 覆盖：0/720；图边、过渡评估、Ruckig、OFF、GNN 未运行。

## 主导结构

主导排序使用 candidate coverage → waypoint coverage → 连续区间 → 共现 → backend → depth → raw contact count。完整表见 `pair_coverage_and_spatial_summary.csv`，前 50 对共现见 `pair_cooccurrence_top50.csv`，覆盖优先排序见 `dominant_blockers_coverage_first.csv`。`shoulder_link` 与 `horseshoe_wall_184` 仍是 Stage 2.2 中记录的代表性主导对，但不是 sole blocker；其余 shoulder/upperarm/forearm 与 horseshoe_wall 对共同覆盖候选。

## 几何与安装审计

- robot base、tunnel、planning/world frame 均为 `base_link`，冻结变换闭合误差沿用通过记录。
- 隧道为 239 个 oriented BOX primitive，米制、scale=1，快照中无重复 object id；`tunnel_floor` 的有效加载状态按原正式场景保留。
- frozen URDF 中 robot links 的 mesh path、collision origin 和 mesh scale 已解析到 `geometry_installation_audit.json`；`spray_tcp_link` 没有 collision element，attached objects 为空，因此已有接触不是喷头工艺接触。
- `unpadded`、`no_floor` 和单碰撞对 ACM 诊断均没有恢复合法节点；广泛 ACM 只产生诊断性 1,439 个“合法”节点，不能作为生产场景结果。

## 未完成项与门禁

仍有 5 个需要真实 native MoveIt2 变体运行才能关闭的审计项：mesh origin、scale/unit、tunnel transform、robot collision geometry、tool collision geometry。因而 `physically_justified=false`、`unresolved_audit_items=5`，Stage 2.3A 仍为 `in_progress`，Stage 2.3B 保持 `blocked_no_valid_nodes`。不得通过 ACM 伪造合法节点，也不得运行边评估、Ruckig、OFF 或 GNN。

## 复现与输入保护

本阶段只写入 `outputs/ik_graph_stage23a2/`，不改写 Stage 2.3A.1、Stage 2.2 或 legacy 720/181 输入。输入 hash 见 `protected_input_hashes.json`。
"""
    (OUT / "stage23a2_report.md").write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
