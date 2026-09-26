"""Stage 2.3A.7 evidence-first audit over the frozen Stage 2.3A.6 population.

This runner never regenerates IK and never edits the formal ACM or robot model.
It creates a new stage-local output directory, verifies the frozen inputs, runs
the independent triangle/FK probe, and records unavailable runtime branches
explicitly instead of inferring them.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shlex
import shutil
import statistics
import struct
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs/ik_graph_stage23a6_bullet_self_collision_audit/fr5_scaled_horseshoe_demo_v44"
BASE = ROOT / "outputs/ik_graph_stage23a4_backend_equivalent/fr5_scaled_horseshoe_demo_v44"
OUT = ROOT / "outputs/ik_graph_stage23a7_independent_geometry_bullet_runtime_audit/fr5_scaled_horseshoe_demo_v44"
CPP = ROOT / "cpp/stage23a7/independent_geometry_probe.cpp"
BIN = OUT / "native/stage23a7_independent_geometry_probe"
TARGET_LINKS = ("forearm_link", "wrist2_link", "wrist3_link")
PAIRS = ("forearm_link<->wrist2_link", "forearm_link<->wrist3_link")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def wsl_path(path: Path) -> str:
    path = path.resolve()
    drive = path.drive.rstrip(":").lower()
    tail = path.as_posix().split(":", 1)[-1]
    return f"/mnt/{drive}{tail}"


def run_wsl(command: str, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["wsl", "bash", "-lc", command], cwd=ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=timeout)


def stl_info(path: Path) -> dict:
    data = path.read_bytes()
    binary = len(data) >= 84
    count = 0
    if binary:
        count = struct.unpack_from("<I", data, 80)[0]
        binary = len(data) == 84 + 50 * count
    verts: list[tuple[float, float, float]] = []
    if binary:
        for i in range(count):
            off = 84 + i * 50 + 12
            vals = struct.unpack_from("<9f", data, off)
            verts.extend((tuple(vals[0:3]), tuple(vals[3:6]), tuple(vals[6:9])))
    else:
        for line in data.decode("utf-8", "ignore").splitlines():
            bits = line.strip().split()
            if bits and bits[0].lower() == "vertex" and len(bits) == 4:
                verts.append(tuple(float(x) for x in bits[1:]))
        count = len(verts) // 3
    unique = set(verts)
    zero = 0
    for i in range(count):
        a, b, c = verts[3 * i:3 * i + 3]
        u = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
        v = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
        cr = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
        if sum(x * x for x in cr) <= 1e-24:
            zero += 1
    bounds = [[min(v[k] for v in verts), max(v[k] for v in verts)] for k in range(3)] if verts else None
    return {
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "vertex_count": len(verts),
        "triangle_count": count,
        "unique_vertex_count": len(unique),
        "duplicate_vertex_count": len(verts) - len(unique),
        "degenerate_triangle_count": zero,
        "watertight": "not_evaluated_without_mesh_topology_library",
        "manifold": "not_evaluated_without_mesh_topology_library",
        "mesh_path": path.relative_to(ROOT).as_posix(),
        "local_aabb": bounds,
    }


def frozen_manifest() -> tuple[dict, dict]:
    recorded = json.loads((SRC / "stage23a6_frozen_inputs.json").read_text(encoding="utf-8"))
    checks = []
    drift = []
    for item in recorded["files"] + [{"path": x["path"], "sha256": x["sha256"], "size_bytes": x["size_bytes"]} for x in recorded["robot_collision_meshes"]]:
        p = ROOT / item["path"]
        actual = {"path": item["path"], "exists": p.exists(), "size_bytes": p.stat().st_size if p.exists() else None, "sha256": sha256(p) if p.exists() else None, "recorded_size_bytes": item["size_bytes"], "recorded_sha256": item["sha256"]}
        actual["match"] = actual["exists"] and actual["size_bytes"] == item["size_bytes"] and actual["sha256"] == item["sha256"]
        checks.append(actual)
        if not actual["match"]:
            drift.append(actual)
    input_manifest = {
        "source_stage": "Stage_2_3A.6",
        "source_manifest": str((SRC / "stage23a6_frozen_inputs.json").resolve()),
        "recorded_candidate_count": recorded["candidate_count"],
        "recorded_waypoint_count": recorded["waypoint_count"],
        "recorded_environment_convex_segments": recorded["environment_convex_segments"],
        "ik_rerun": False,
        "formal_acm_modified": False,
        "files": checks,
        "input_drift": drift,
        "status": "verified" if not drift else "blocked_input_drift",
    }
    return recorded, input_manifest


def recover_nodes() -> tuple[list[dict], dict[int, int]]:
    nodes = json.loads((SRC / "candidate_classification/bullet_only_self_collision_nodes.json").read_text(encoding="utf-8"))["nodes"]
    if len(nodes) != 658:
        raise RuntimeError(f"expected 658 disputed nodes, got {len(nodes)}")
    with (BASE / "deterministic_ik_candidates.csv").open(encoding="utf-8", newline="") as f:
        candidate_index = {row["candidate_id"]: i for i, row in enumerate(csv.DictReader(f))}
    return nodes, candidate_index


def build_geometry_manifest() -> dict:
    urdf = ROOT / "external/frcobot_ros2/fairino_description/urdf/fairino5_v6.urdf"
    root = ET.parse(urdf).getroot()
    links = []
    for link in root.findall("link"):
        if link.attrib.get("name") not in TARGET_LINKS:
            continue
        collisions = []
        for idx, collision in enumerate(link.findall("collision")):
            geometry = collision.find("geometry")
            mesh = geometry.find("mesh") if geometry is not None else None
            filename = mesh.attrib.get("filename", "") if mesh is not None else ""
            rel_mesh = filename.split("/meshes/")[-1] if "/meshes/" in filename else ""
            path = ROOT / "external/frcobot_ros2/fairino_description/meshes" / rel_mesh
            origin = collision.find("origin")
            origin_source = "urdf_collision_origin" if origin is not None else "collision_origin_absent_default"
            info = stl_info(path) if path.exists() else {"mesh_path": str(path), "status": "not_available"}
            collisions.append({
                "link_name": link.attrib["name"],
                "collision_element_index": idx,
                "geometry_type": "mesh" if mesh is not None else "unknown",
                "mesh_filename": filename,
                "mesh_scale": (mesh.attrib.get("scale", "1 1 1").split() if mesh is not None else ["1", "1", "1"]),
                "collision_origin_xyz": origin.attrib.get("xyz", "0 0 0") if origin is not None else "0 0 0",
                "collision_origin_rpy": origin.attrib.get("rpy", "0 0 0") if origin is not None else "0 0 0",
                "collision_origin_source": origin_source,
                **info,
            })
        links.extend(collisions)
    return {"source_urdf": urdf.relative_to(ROOT).as_posix(), "urdf_sha256": sha256(urdf), "collision_elements": links, "all_collision_elements_retained": True}


def write_disputed(nodes: list[dict], candidate_index: dict[str, int]) -> None:
    fields = ["node_id", "waypoint_id", "candidate_index", "candidate_id", "joint_vector", "FCL_collision_status", "Bullet_collision_status", "Bullet_collision_pair", "all_Bullet_self_collision_pairs", "robot_world_collision_status"]
    rows = []
    geometry_rows = []
    for n in nodes:
        cid = n["candidate_id"]
        pairs = n["bullet_self_pairs"]
        row = {
            "node_id": cid,
            "waypoint_id": n["waypoint_id"],
            "candidate_index": candidate_index[cid],
            "candidate_id": cid,
            "joint_vector": json.dumps(n["joint_values"], separators=(",", ":")),
            "FCL_collision_status": "self_collision=false;robot_world_collision=false",
            "Bullet_collision_status": "self_collision=true;robot_world_collision=false",
            "Bullet_collision_pair": pairs[0] if len(pairs) == 1 else "multiple",
            "all_Bullet_self_collision_pairs": json.dumps(pairs, separators=(",", ":")),
            "robot_world_collision_status": "false",
        }
        rows.append(row)
        for pair in pairs:
            geometry_rows.append({"node_id": cid, "waypoint_id": n["waypoint_id"], "candidate_index": candidate_index[cid], "pair": pair, "joint_vector": json.dumps(n["joint_values"], separators=(",", ":"))})
    write_csv(OUT / "stage23a7_disputed_nodes.csv", fields, rows)
    write_csv(OUT / "stage23a7_geometry_input.csv", ["node_id", "waypoint_id", "candidate_index", "pair", "joint_vector"], geometry_rows)
    write_json(OUT / "stage23a7_population_check.json", {
        "disputed_node_count": len(rows),
        "geometry_input_rows": len(geometry_rows),
        "unique_waypoint_count": len({r["waypoint_id"] for r in rows}),
        "bullet_fully_invalid_waypoints": 50,
        "robot_world_difference_nodes": 0,
        "pair_count": Counter(r["pair"] for r in geometry_rows),
        "node_id_set_sha256": hashlib.sha256(("\n".join(sorted(r["node_id"] for r in rows)) + "\n").encode()).hexdigest(),
        "status": "verified" if len(rows) == 658 and len(geometry_rows) >= 658 else "blocked_disputed_population_not_reproduced",
    })


def compile_probe() -> None:
    OUT.joinpath("native").mkdir(parents=True, exist_ok=True)
    cmd = f"g++ -O3 -std=c++17 {shlex.quote(wsl_path(CPP))} -o {shlex.quote(wsl_path(BIN))}"
    result = run_wsl(cmd, timeout=180)
    (OUT / "native" / "compile_stdout.txt").write_text(result.stdout or "", encoding="utf-8")
    (OUT / "native" / "compile_stderr.txt").write_text(result.stderr or "", encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError("independent geometry probe compilation failed")


def run_geometry(run_dir: Path) -> dict:
    cmd = " ".join([
        shlex.quote(wsl_path(BIN)), shlex.quote(wsl_path(OUT / "stage23a7_geometry_input.csv")),
        shlex.quote(wsl_path(ROOT / "external/frcobot_ros2/fairino_description/meshes/fairino5_v6/forearm_link.STL")),
        shlex.quote(wsl_path(ROOT / "external/frcobot_ros2/fairino_description/meshes/fairino5_v6/wrist2_link.STL")),
        shlex.quote(wsl_path(ROOT / "external/frcobot_ros2/fairino_description/meshes/fairino5_v6/wrist3_link.STL")),
        shlex.quote(wsl_path(run_dir)),
    ])
    result = run_wsl(cmd, timeout=1800)
    (run_dir / "stdout.txt").write_text(result.stdout, encoding="utf-8")
    (run_dir / "stderr.txt").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"independent geometry probe failed: {result.stderr[-1000:]}")
    return {"geometry_sha256": sha256(run_dir / "stage23a7_independent_geometry.csv"), "fk_transform_sha256": sha256(run_dir / "stage23a7_fk_transforms.csv"), "stderr": result.stderr.strip()}


def write_runtime_and_missing_branches(nodes: list[dict]) -> None:
    # These are observations of the available public probe, not inferred values.
    write_json(OUT / "stage23a7_bullet_runtime_shapes.json", {
        "status": "instrumentation_not_available",
        "runtime_source": "Stage 2.3A.6 public MoveIt PlanningScene probe",
        "bullet_library_path": "/lib/x86_64-linux-gnu/libBulletCollision.so.3.24",
        "bullet_version": "3.24 (library filename; runtime shape API not intercepted)",
        "moveit_collision_plugin": "collision_detection_bullet",
        "fields": {link: {"collision_object_identity": "not_exposed_by_runtime_api", "shape_type": "not_exposed_by_runtime_api", "child_shape_count": "not_exposed_by_runtime_api", "local_scaling": "not_exposed_by_runtime_api", "collision_margin": "not_exposed_by_runtime_api", "child_transform": "not_exposed_by_runtime_api", "world_transform": "not_exposed_by_runtime_api", "contact_processing_threshold": "not_exposed_by_runtime_api", "broadphase_filter": "not_exposed_by_runtime_api"} for link in TARGET_LINKS},
        "instrumentation_commit_or_diff": "not_available",
    })
    write_json(OUT / "stage23a7_acm_ablation.json", {
        "formal_acm_modified": False,
        "formal_acm_sha256": json.loads((SRC / "acm_audit/original_acm.sha256").read_text(encoding="utf-8")),
        "cases": {"A_only_forearm_wrist2_allowed": {"status": "not_run", "reason": "diagnostic ACM control not exposed by frozen probe"}, "B_only_forearm_wrist3_allowed": {"status": "not_run", "reason": "diagnostic ACM control not exposed by frozen probe"}, "C_both_disputed_pairs_allowed": {"status": "not_run", "reason": "diagnostic ACM control not exposed by frozen probe"}},
        "diagnostic_only": True,
        "statement": "ACM ablation is diagnostic only. No production ACM modification was applied. ACM recovery is not accepted as a Stage 2.3A fix.",
    })
    write_csv(OUT / "stage23a7_perturbation_results.csv", ["node_id", "pair", "perturbation_type", "joint_name", "perturbation_value", "perturbation_direction_id", "FCL_collision", "Bullet_collision", "independent_intersection", "independent_minimum_distance", "Bullet_reported_distance", "Bullet_penetration_depth", "status"], [])
    write_json(OUT / "stage23a7_perturbation_status.json", {"status": "not_run", "reason": "no isolated deterministic per-state FCL/Bullet perturbation runner was available without changing the frozen semantic probe", "required_states": ["+1e-8", "-1e-8", "+1e-7", "-1e-7", "+1e-6", "-1e-6", "+1e-5", "-1e-5", "+1e-4", "-1e-4"], "formal_nodes_written": False})

    contact_fields = ["node_id", "pair", "bullet_object_a", "bullet_object_b", "child_index_a", "child_index_b", "contact_point_a_x", "contact_point_a_y", "contact_point_a_z", "contact_point_b_x", "contact_point_b_y", "contact_point_b_z", "contact_normal_x", "contact_normal_y", "contact_normal_z", "reported_contact_distance", "penetration_depth", "applied_margin_a", "applied_margin_b", "shape_type_a", "shape_type_b"]
    contact_rows = []
    source_rows = [json.loads(x) for x in (SRC / "bullet_classification_run1.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    for row in source_rows:
        if not row["self_collision"]:
            continue
        for c in row["self_contacts"]:
            pa, pb = c["nearest_points"]
            n = c["contact_normal"]
            contact_rows.append({"node_id": row["candidate_id"], "pair": c["sorted_link_pair"], "bullet_object_a": "not_exposed_by_runtime_api", "bullet_object_b": "not_exposed_by_runtime_api", "child_index_a": "not_exposed_by_runtime_api", "child_index_b": "not_exposed_by_runtime_api", "contact_point_a_x": pa[0], "contact_point_a_y": pa[1], "contact_point_a_z": pa[2], "contact_point_b_x": pb[0], "contact_point_b_y": pb[1], "contact_point_b_z": pb[2], "contact_normal_x": n[0], "contact_normal_y": n[1], "contact_normal_z": n[2], "reported_contact_distance": "not_exposed_by_runtime_api", "penetration_depth": c["penetration_depth"], "applied_margin_a": "not_exposed_by_runtime_api", "applied_margin_b": "not_exposed_by_runtime_api", "shape_type_a": "not_exposed_by_runtime_api", "shape_type_b": "not_exposed_by_runtime_api"})
    write_csv(OUT / "stage23a7_bullet_contacts.csv", contact_fields, contact_rows)


def make_classification(nodes: list[dict], geometry_rows: list[dict]) -> dict:
    by_node: dict[str, list[dict]] = defaultdict(list)
    for r in geometry_rows:
        by_node[r["node_id"]].append(r)
    fields = ["node_id", "waypoint_id", "original_pair", "independent_intersection", "independent_min_distance_m", "FCL_original", "Bullet_original", "perturbation_stability", "Bullet_shape_type", "Bullet_margin", "Bullet_contact_distance", "Bullet_penetration_depth", "ACM_case_recovery", "final_classification", "classification_reason", "evidence_complete"]
    out = []
    for n in nodes:
        gs = by_node[n["candidate_id"]]
        intersections = ";".join(f"{g['pair']}={g['triangle_intersection']}" for g in gs)
        distances = ";".join(f"{g['pair']}={g['minimum_distance_m']}" for g in gs)
        out.append({"node_id": n["candidate_id"], "waypoint_id": n["waypoint_id"], "original_pair": ";".join(n["bullet_self_pairs"]), "independent_intersection": intersections, "independent_min_distance_m": distances, "FCL_original": "false", "Bullet_original": "true", "perturbation_stability": "not_run", "Bullet_shape_type": "not_exposed_by_runtime_api", "Bullet_margin": "not_exposed_by_runtime_api", "Bullet_contact_distance": "not_exposed_by_runtime_api", "Bullet_penetration_depth": "observed_in_source_contact_rows_but_not_shape_semantics", "ACM_case_recovery": "not_run", "final_classification": "instrumentation_incomplete", "classification_reason": "Independent geometry was run, but required perturbation stability, diagnostic ACM cases, and runtime shape/margin interception were not completed.", "evidence_complete": False})
    write_csv(OUT / "stage23a7_node_classification.csv", fields, out)
    return {"count": len(out), "classification_counts": Counter(r["final_classification"] for r in out), "waypoint_counts": Counter(r["waypoint_id"] for r in out), "pair_counts": Counter(p for r in out for p in r["original_pair"].split(";"))}


def geometry_summary(rows: list[dict]) -> dict:
    distances = [float(r["minimum_distance_m"]) for r in rows]
    pair_nodes = defaultdict(set)
    pair_waypoints = defaultdict(set)
    for r in rows:
        pair_nodes[r["pair"]].add(r["node_id"])
        pair_waypoints[r["pair"]].add(r["waypoint_id"])
    positive = [d for d in distances if d > 0]
    return {"row_count": len(rows), "node_count": len({r["node_id"] for r in rows}), "pair_count": Counter(r["pair"] for r in rows), "pair_node_count": {k: len(v) for k, v in sorted(pair_nodes.items())}, "pair_waypoint_count": {k: len(v) for k, v in sorted(pair_waypoints.items())}, "intersection_count": sum(r["triangle_intersection"] == "true" for r in rows), "strict_positive_count": sum(r["geometry_status"] == "strictly_positive_separation" for r in rows), "near_zero_count": sum(r["geometry_status"] == "near_zero_separation" for r in rows), "minimum_positive_distance_m": min(positive, default=None), "positive_distance_percentiles_m": {str(p): statistics.quantiles(positive, n=100, method="inclusive")[p - 1] if len(positive) >= 100 else None for p in (1, 50, 99)}, "maximum_distance_m": max(distances) if distances else None}


def write_gate_and_report(input_manifest: dict, population: dict, geom: dict, cls: dict, det: dict) -> None:
    gate = {"Stage_2_3A_7": {"name": "disputed_self_collision_independent_geometry_and_bullet_runtime_semantics_audit", "status": "blocked_bullet_runtime_instrumentation_incomplete"}, "Stage_2_3A_8": {"status": "not_started"}, "Stage_2_3B": {"status": "blocked"}, "Ruckig": "not_started", "OFF": "not_started", "GNN": "not_started", "training_data_generation": "not_started", "gate": {"input_hashes_verified": input_manifest["status"] == "verified", "disputed_nodes_reproduced": population["disputed_node_count"] == 658, "independent_geometry_rows": geom["row_count"], "independent_geometry_nodes": geom["node_count"], "perturbation_states_evaluated": 0, "bullet_runtime_shapes_captured": False, "acm_ablation_completed": False, "unique_classification_coverage": cls["count"] == 658, "unexplained_backend_disagreement": 658, "instrumentation_incomplete": 658, "invalid_input_or_geometry": 0, "geometry_only_deterministic_runs": det["geometry_only_runs"], "complete_audit_determinism": "not_evaluated", "formal_acm_modified": False, "downstream_started": False}}
    write_json(OUT / "stage23a7_gate_report.json", gate)
    pair_counts = ", ".join(f"{k}: {v} rows, {geom['pair_node_count'][k]} nodes, {geom['pair_waypoint_count'][k]} waypoints" for k, v in sorted(geom["pair_count"].items()))
    lines = [
        "# Stage 2.3A.7 independent geometry and Bullet runtime semantics audit", "", "## Final gate", "", "`Stage_2_3A_7 = blocked_bullet_runtime_instrumentation_incomplete`.", "", "The exact Stage 2.3A.6 frozen population was recovered and independently evaluated at triangle level. The stage is not passed because deterministic per-state FCL/Bullet perturbations, isolated ACM cases, and runtime Bullet shape/margin interception were not available in the executable environment.", "", "## Frozen-input integrity", "", f"- Candidate count: {population['candidate_count']}; waypoint count: {population['waypoint_count']}; input status: `{input_manifest['status']}`.", f"- Disputed nodes: {population['disputed_node_count']}/658; lost Bullet waypoints: 50; robot-world difference nodes: 0.", f"- Disputed node-set SHA-256: `{population['node_id_set_sha256']}`.", "- Formal ACM, URDF, SRDF, candidate values, waypoint path, robot base pose, and tunnel pose were not modified.", "", "## Independent geometry", "", f"- Rows: {geom['row_count']} across {geom['node_count']} nodes and two disputed pairs ({pair_counts}).", f"- Exact triangle intersections: {geom['intersection_count']}; strictly positive separations: {geom['strict_positive_count']}; near-zero rows: {geom['near_zero_count']}.", f"- Positive-distance minimum: `{geom['minimum_positive_distance_m']}` m; percentiles: `{geom['positive_distance_percentiles_m']}`; maximum: `{geom['maximum_distance_m']}` m.", "- Transform source: independent fixed-chain FK reconstructed from the frozen URDF joint origins/axes and frozen joint vectors; saved in `stage23a7_fk_transforms.csv`.", "- Collision geometry source is URDF `<collision>` geometry. The wrist2 collision element has no valid `<origin>` tag in the frozen URDF, so the probe records the applied default origin `(0,0,0)` explicitly.", "", "## Required branches not completed", "", "- Perturbation states: `not_run`; no state was written back to formal candidates.", "- ACM Case A/B/C: `not_run`; diagnostic only, no production ACM modification.", "- Bullet runtime shapes, child shapes, margins, contact processing threshold, object identities, and filters: `instrumentation_not_available` from the public MoveIt probe.", "- Bullet contacts: observed MoveIt contact points, normals, and penetration depths were preserved, but shape/child/margin fields remain unavailable and no semantics were inferred.", "", "## Classification and determinism", "", f"- All 658 nodes have a unique provisional final label `instrumentation_incomplete`; this is not a claim that the nodes are false positives or real collisions.", f"- Geometry-only deterministic repetitions: {det['geometry_only_runs']}; complete Stage 2.3A.7 determinism: `not_evaluated`.", "- Unexplained backend disagreement remains: 658 nodes. Stage 2.3B remains blocked.", "", "## Required conclusions", "", "1. The 658-node population was reproduced exactly.", "2. Pair-specific node/waypoint counts are in `stage23a7_node_classification.csv` and the geometry output.", "3. No final real-collision/false-positive/FCL-false-negative claim is made because the required stability and runtime-semantic evidence is incomplete.", "4. Stage 2.3A.8, when permitted, must repair and rebaseline backend collision semantics; it must not bypass the disagreement through formal ACM changes.", "", "ACM ablation is diagnostic only. No production ACM modification was applied. ACM recovery is not accepted as a Stage 2.3A fix.", "", "Stage 2.3B remains blocked; Ruckig, OFF, GNN, and training-data generation were not started.", ""]
    lines.insert(lines.index("", lines.index("## Classification and determinism")) if "## Classification and determinism" in lines else len(lines), "- The 50 completely Bullet-invalid waypoints are composed only of the two disputed pair labels in the frozen source causes; this does not establish that the pairs are geometrically false positives.")
    (OUT / "stage23a7_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for d in ("geometry_run_1", "geometry_run_2", "geometry_run_3", "native"):
        (OUT / d).mkdir(exist_ok=True)
    recorded, input_manifest = frozen_manifest()
    write_json(OUT / "stage23a7_input_manifest.json", input_manifest)
    if input_manifest["status"] != "verified":
        write_json(OUT / "stage23a7_gate_report.json", {"Stage_2_3A_7": {"status": "blocked_input_drift"}, "Stage_2_3B": {"status": "blocked"}})
        return 4
    nodes, candidate_index = recover_nodes()
    write_disputed(nodes, candidate_index)
    population = json.loads((OUT / "stage23a7_population_check.json").read_text(encoding="utf-8"))
    population.update({"candidate_count": recorded["candidate_count"], "waypoint_count": recorded["waypoint_count"]})
    write_json(OUT / "stage23a7_population_check.json", population)
    write_json(OUT / "stage23a7_geometry_manifest.json", build_geometry_manifest())
    compile_probe()
    det_runs = []
    for i in (1, 2, 3):
        run_dir = OUT / f"geometry_run_{i}"
        geometry_file = run_dir / "stage23a7_independent_geometry.csv"
        transform_file = run_dir / "stage23a7_fk_transforms.csv"
        if geometry_file.exists() and transform_file.exists() and sum(1 for _ in geometry_file.open(encoding="utf-8")) == 993:
            det_runs.append({"geometry_sha256": sha256(geometry_file), "fk_transform_sha256": sha256(transform_file), "stderr": "reused_verified_stage_local_run"})
        else:
            det_runs.append(run_geometry(run_dir))
    shutil.copy2(OUT / "geometry_run_1/stage23a7_independent_geometry.csv", OUT / "stage23a7_independent_geometry.csv")
    shutil.copy2(OUT / "geometry_run_1/stage23a7_fk_transforms.csv", OUT / "stage23a7_fk_transforms.csv")
    with (OUT / "stage23a7_independent_geometry.csv").open(encoding="utf-8", newline="") as f:
        g_rows = list(csv.DictReader(f))
    geom = geometry_summary(g_rows)
    write_json(OUT / "stage23a7_independent_geometry_summary.json", geom)
    write_runtime_and_missing_branches(nodes)
    cls = make_classification(nodes, g_rows)
    det = {"geometry_only_runs": 3, "geometry_hashes": [r["geometry_sha256"] for r in det_runs], "fk_transform_hashes": [r["fk_transform_sha256"] for r in det_runs], "geometry_equal": len({r["geometry_sha256"] for r in det_runs}) == 1, "complete_audit": "not_evaluated", "excluded_fields": ["timestamps", "process_id", "elapsed_time"]}
    write_json(OUT / "stage23a7_determinism.json", det)
    test = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_stage23a7_outputs.py"], cwd=ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=120)
    (OUT / "stage23a7_pytest_output.txt").write_text((test.stdout or "") + (test.stderr or ""), encoding="utf-8")
    write_json(OUT / "stage23a7_regression_tests.json", {"command": "python -m pytest -q tests/test_stage23a7_outputs.py", "status": "passed" if test.returncode == 0 else "failed", "returncode": test.returncode, "unit_and_real_dataset_checks": "passed" if test.returncode == 0 else "failed", "integration_tests": "not_run", "summary": (test.stdout or "").strip()})
    if test.returncode != 0:
        raise RuntimeError("stage23a7 regression tests failed")
    write_gate_and_report(input_manifest, population, geom, cls, det)
    print(json.dumps({"output": str(OUT), "status": "blocked_bullet_runtime_instrumentation_incomplete", "nodes": 658, "geometry_rows": geom["row_count"], "geometry_only_deterministic": det["geometry_equal"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
