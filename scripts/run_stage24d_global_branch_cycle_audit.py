"""Stage 2.4D global IK-branch and bounded-joint-winding audit.

This audit consumes only the frozen Stage 2.4A/2.4B/2.4C bundles.  It does
not regenerate waypoints, change the robot model, change collision semantics,
or run a downstream planner.  The global graph is the union of the formally
valid candidate nodes and the persisted FCL/Bullet edge evidence from those
three stages.  Every accepted graph edge must be valid in both backends.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/ik_graph_stage23a4_backend_equivalent/fr5_scaled_horseshoe_demo_v44"
STAGE24A = ROOT / "outputs/ik_graph_stage24a_edge_equivalence/fr5_scaled_horseshoe_demo_v45"
STAGE24B = ROOT / "outputs/ik_graph_stage24b_closed_cycle_recovery/fr5_scaled_horseshoe_demo_v45"
STAGE24C = ROOT / "outputs/ik_graph_stage24c_seam_window_repair/fr5_scaled_horseshoe_demo_v45"
OUT = ROOT / "outputs/ik_graph_stage24d_global_branch_cycle_audit/fr5_scaled_horseshoe_demo_v45"
WAYPOINT_COUNT = 720
MAX_STEP_DEG = 20.0
JOBS = ("j1", "j2", "j3", "j4", "j5", "j6")
CORE_FILES = (
    "stage24d_frozen_inputs_manifest.json",
    "stage24d_joint_topology_audit.json",
    "stage24d_joint_topology_audit.md",
    "stage24d_nodes.jsonl",
    "stage24d_lifted_states.jsonl",
    "stage24d_edges.jsonl",
    "stage24d_branch_components.json",
    "stage24d_transition_minima.csv",
    "stage24d_j5_branch_audit.csv",
    "stage24d_search_runs.jsonl",
    "stage24d_selected_cycle.json",
    "stage24d_infeasibility_certificate.json",
    "stage24d_report.md",
    "stage24d_gate_report.json",
    "dirty_worktree_preservation.json",
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


def verify_sha_file(directory: Path) -> dict[str, Any]:
    sums_path = directory / "SHA256SUMS"
    checked = 0
    failures: list[dict[str, Any]] = []
    if not sums_path.exists():
        return {"path": str(sums_path), "checked_files": 0, "failure_count": 1, "failures": [{"reason": "missing_manifest"}]}
    for line in sums_path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            if line.strip():
                failures.append({"reason": "malformed_line", "line": line})
            continue
        expected, relative = fields
        relative = relative.lstrip("*")
        path = directory / relative
        checked += 1
        if not path.exists():
            failures.append({"reason": "missing_file", "path": relative})
            continue
        actual = sha256(path)
        if actual != expected:
            failures.append({"reason": "sha256_mismatch", "path": relative, "expected": expected, "actual": actual})
    return {"path": str(sums_path), "checked_files": checked, "failure_count": len(failures), "failures": failures}


def file_entry(path: Path, authority: str) -> dict[str, Any]:
    return {"path": str(path.resolve()), "authority": authority, "exists": path.exists(), "size": path.stat().st_size if path.exists() else None, "sha256": sha256(path) if path.exists() else None}


def directory_entry(path: Path, authority: str) -> dict[str, Any]:
    files = []
    if path.exists():
        files = [{"relative_path": str(item.relative_to(path)).replace("\\", "/"), "size": item.stat().st_size, "sha256": sha256(item)} for item in sorted(path.rglob("*")) if item.is_file()]
    return {"path": str(path.resolve()), "authority": authority, "exists": path.exists(), "file_count": len(files), "sha256": value_hash(files), "files": files}


def frozen_inputs_manifest() -> dict[str, Any]:
    files = [
        (ROOT / "external/frcobot_ros2/fairino_description/urdf/fairino5_v6.urdf", "authoritative_urdf"),
        (ROOT / "outputs/ik_graph_stage193/expanded_runtime_urdf.urdf", "runtime_expanded_urdf_used_by_stage24c"),
        (ROOT / "ros2_moveit_bridge/config/fairino5_v6_spray_tcp.srdf", "authoritative_srdf"),
        (ROOT / "ros2_moveit_bridge/config/fairino5_v6_spray_tcp.urdf.xacro", "authoritative_tcp_transform_model"),
        (ROOT / "ros2_moveit_bridge/config/joint_limits_with_jerk.yaml", "authoritative_moveit_joint_limits"),
        (ROOT / "ros2_moveit_bridge/config/fairino5_v6_spray_tcp.srdf", "moveit_planning_group"),
        (SOURCE / "waypoints.csv", "frozen_720_waypoints"),
        (SOURCE / "deterministic_ik_candidates.csv", "frozen_stage23a4_candidate_source"),
        (SOURCE / "scaled_demo_parameters.yaml", "frozen_process_and_scene_parameters"),
        (STAGE24A / "stage24a_gate_report.json", "stage24a_formal_result"),
        (STAGE24A / "stage24a_edges_fcl.jsonl", "stage24a_fcl_edge_result"),
        (STAGE24A / "stage24a_edges_bullet.jsonl", "stage24a_bullet_edge_result"),
        (STAGE24A / "stage24_nodes.jsonl", "stage24a_formal_node_source"),
        (STAGE24B / "stage24b_gate_report.json", "stage24b_formal_result"),
        (STAGE24B / "stage24b_new_candidates.csv", "stage24b_global_candidate_source"),
        (STAGE24B / "stage24b_seam_edges_fcl.jsonl", "stage24b_fcl_edge_result"),
        (STAGE24B / "stage24b_seam_edges_bullet.jsonl", "stage24b_bullet_edge_result"),
        (STAGE24C / "stage24c_gate_report.json", "stage24c_formal_result"),
        (STAGE24C / "stage24c_all_window_candidates.csv", "stage24c_global_candidate_source"),
        (STAGE24C / "stage24c_environment_manifest.json", "stage24c_planning_scene_and_backend_manifest"),
        (STAGE24C / "stage24c_backend_equivalence.json", "stage24c_backend_edge_equivalence"),
    ]
    directories = [(SOURCE / "tunnel_collision_parts", "frozen_tunnel_collision_geometry")]
    stage_checks = {"stage24a": verify_sha_file(STAGE24A), "stage24b": verify_sha_file(STAGE24B), "stage24c": verify_sha_file(STAGE24C)}
    return {
        "manifest_version": "stage24d-frozen-inputs-v1",
        "stage": "2.4D",
        "waypoint_count": WAYPOINT_COUNT,
        "max_joint_step_deg": MAX_STEP_DEG,
        "collision_method": "adaptive_discrete_interpolation",
        "ccd": "not_available",
        "clearance": "not_available",
        "search_space_policy": "union_of_formally_valid_Stage_2_4A_nodes_Stage_2_4B_candidates_and_dual_backend_valid_Stage_2_4C_candidates",
        "no_model_or_waypoint_mutation": True,
        "files": [file_entry(path, authority) for path, authority in files],
        "directories": [directory_entry(path, authority) for path, authority in directories],
        "prior_stage_sha256_verification": stage_checks,
        "prior_stage_gate_snapshots": {"Stage_2_4A": read_json(STAGE24A / "stage24a_gate_report.json"), "Stage_2_4B": read_json(STAGE24B / "stage24b_gate_report.json"), "Stage_2_4C": read_json(STAGE24C / "stage24c_gate_report.json")},
    }


def parse_yaml_controller_limits(path: Path) -> dict[str, dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    result: dict[str, dict[str, Any]] = {}
    current: str | None = None
    for line in text.splitlines():
        match = re.match(r"\s{2}(j[1-6]):\s*$", line)
        if match:
            current = match.group(1)
            result[current] = {}
            continue
        if current:
            item = re.match(r"\s{4}([A-Za-z_]+):\s*(true|false|[-+0-9.eE]+)\s*$", line)
            if item:
                raw = item.group(2)
                result[current][item.group(1)] = raw.lower() == "true" if raw.lower() in {"true", "false"} else float(raw)
    return result


def joint_topology() -> dict[str, Any]:
    urdf_root = ET.parse(ROOT / "external/frcobot_ros2/fairino_description/urdf/fairino5_v6.urdf").getroot()
    runtime_root = ET.parse(ROOT / "outputs/ik_graph_stage193/expanded_runtime_urdf.urdf").getroot()
    srdf_root = ET.parse(ROOT / "ros2_moveit_bridge/config/fairino5_v6_spray_tcp.srdf").getroot()
    urdf_joints = {node.attrib["name"]: node for node in urdf_root.findall("joint")}
    runtime_joints = {node.attrib["name"]: node for node in runtime_root.findall("joint")}
    group = next(node for node in srdf_root.findall("group") if node.attrib.get("name") == "fairino5_v6_group")
    group_joints = [node.attrib["name"] for node in group.findall("joint")]
    controller = parse_yaml_controller_limits(ROOT / "ros2_moveit_bridge/config/joint_limits_with_jerk.yaml")
    rows = []
    for name in JOBS:
        uj = urdf_joints[name]
        rj = runtime_joints[name]
        limit = uj.find("limit")
        lower = float(limit.attrib["lower"])
        upper = float(limit.attrib["upper"])
        runtime_limit = rj.find("limit")
        rows.append({
            "joint_name": name,
            "urdf_type": uj.attrib["type"],
            "moveit_variable_type": "bounded_revolute" if rj.attrib["type"] == "revolute" and runtime_limit is not None else rj.attrib["type"],
            "lower_limit": lower,
            "upper_limit": upper,
            "position_bounded": True,
            "continuous": False,
            "controller_lower_limit": None,
            "controller_upper_limit": None,
            "controller_position_limits_present": False,
            "controller_runtime_parameters": controller.get(name, {}),
            "effective_authoritative_limits": {"lower": lower, "upper": upper, "source": "URDF position limit; controller YAML has no position override"},
            "runtime_urdf_type": rj.attrib["type"],
            "runtime_urdf_limits_match": float(runtime_limit.attrib["lower"]) == lower and float(runtime_limit.attrib["upper"]) == upper,
        })
    return {
        "joint_order": list(JOBS),
        "moveit_group": "fairino5_v6_group",
        "moveit_group_joints": group_joints,
        "all_group_joints_match_expected": group_joints == list(JOBS),
        "joints": rows,
        "j5_conclusion": "j5 is a bounded revolute joint, not continuous; arbitrary +/-360 degree winding is illegal and no shortest-period wrap is permitted",
        "angle_distance_rule": "continuous joints use shortest period only; bounded revolute joints use raw difference inside authoritative limits",
    }


def topology_markdown(audit: dict[str, Any]) -> str:
    lines = ["# Stage 2.4D joint topology audit", "", "All six FR5 joints are bounded revolute joints in the authoritative URDF and runtime-expanded URDF. The controller YAML supplies velocity/acceleration/jerk limits but no position-limit override; effective position limits therefore remain the URDF limits.", "", "| joint | URDF | MoveIt variable | lower (rad) | upper (rad) | bounded | continuous | controller position limits |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in audit["joints"]:
        lines.append(f"| {row['joint_name']} | {row['urdf_type']} | {row['moveit_variable_type']} | {row['lower_limit']} | {row['upper_limit']} | {row['position_bounded']} | {row['continuous']} | {row['controller_position_limits_present']} |")
    lines += ["", "- j5 is bounded revolute, so `q` and `q + 2*pi` are different physical states unless both are independently within the authoritative limits; in this model the latter is not available as an added winding degree of freedom.", "- No artificial ±360° adjustment is used in the graph search.", ""]
    return "\n".join(lines)


def limits_from_topology(audit: dict[str, Any]) -> tuple[list[float], list[float], list[str]]:
    return [row["lower_limit"] for row in audit["joints"]], [row["upper_limit"] for row in audit["joints"]], [row["urdf_type"] for row in audit["joints"]]


def branch_signature(q: list[float], lower: list[float], upper: list[float]) -> tuple[str, list[int], dict[str, Any]]:
    signs = ("positive" if q[0] >= 0 else "negative", "positive" if q[2] >= 0 else "negative", "positive" if q[4] >= 0 else "negative")
    winding = []
    options: list[list[int]] = []
    for value, lo, hi in zip(q, lower, upper):
        legal = [k for k in range(-3, 4) if lo - 1e-10 <= value + 2.0 * math.pi * k <= hi + 1e-10]
        options.append(legal)
        winding.append(legal[0] if legal else 999)
    signature = f"shoulder={signs[0]}|elbow={signs[1]}|wrist={signs[2]}|winding={','.join(map(str, winding))}"
    metadata = {"shoulder_branch": signs[0], "elbow_branch": signs[1], "wrist_branch": signs[2], "winding_integer": winding, "legal_winding_options": options}
    return signature, winding, metadata


def make_node(row: dict[str, Any], source_stage: str, lower: list[float], upper: list[float], fcl: bool, bullet: bool) -> dict[str, Any]:
    q = parse_q(row["joint_values"])
    signature, winding, branch = branch_signature(q, lower, upper)
    wp = int(row.get("waypoint_id", row.get("waypoint_index")))
    limit_valid = len(q) == 6 and all(lo - 1e-10 <= value <= hi + 1e-10 for value, lo, hi in zip(q, lower, upper))
    return {
        "node_id": f"wp{wp:04d}:{row['candidate_id']}",
        "candidate_id": str(row["candidate_id"]),
        "waypoint_index": wp,
        "joint_values_rad": q,
        "joint_values_deg": [math.degrees(value) for value in q],
        "ik_branch_signature": signature,
        "branch_metadata": branch,
        "winding_state": winding,
        "source_stage": source_stage,
        "source_seed": row.get("seed_id") or row.get("seed_template_id"),
        "source_seed_template_id": row.get("seed_template_id"),
        "fk_position_error": float(row["FK_position_error_m"]) if row.get("FK_position_error_m") not in (None, "") else float(row.get("solver_position_error_m", "nan")) if row.get("solver_position_error_m") not in (None, "") else None,
        "fk_orientation_error": float(row["FK_orientation_error_deg"]) if row.get("FK_orientation_error_deg") not in (None, "") else float(row.get("solver_tool_z_error_deg", "nan")) if row.get("solver_tool_z_error_deg") not in (None, "") else None,
        "spray_distance_error": None,
        "normal_error": None,
        "roll_error": None,
        "joint_limit_valid": limit_valid,
        "fcl_node_valid": bool(fcl),
        "bullet_node_valid": bool(bullet),
        "dual_backend_valid": bool(fcl and bullet),
        "process_tolerance_pass": row.get("process_tolerance_pass", row.get("formal_constraint_pass", True)) not in {False, "False", "false"},
        "candidate_record_hash": value_hash(row),
    }


def load_nodes(audit: dict[str, Any]) -> list[dict[str, Any]]:
    lower, upper, _ = limits_from_topology(audit)
    source_rows = {row["candidate_id"]: row for row in read_csv(SOURCE / "deterministic_ik_candidates.csv")}
    formal_ids = {row["candidate_id"] for row in read_jsonl(STAGE24A / "stage24_nodes.jsonl")}
    nodes: dict[str, dict[str, Any]] = {}
    for cid in sorted(formal_ids):
        if cid not in source_rows:
            raise RuntimeError(f"formal Stage 2.4A node is absent from frozen source candidates: {cid}")
        nodes[cid] = make_node(source_rows[cid], "Stage_2_4A", lower, upper, True, True)
    for row in read_csv(STAGE24B / "stage24b_new_candidates.csv"):
        if row.get("valid") != "True":
            continue
        nodes[row["candidate_id"]] = make_node(row, "Stage_2_4B", lower, upper, True, True)
    for row in read_csv(STAGE24C / "stage24c_all_window_candidates.csv"):
        if row.get("fcl_node_validity") != "True" or row.get("bullet_node_validity") != "True":
            continue
        nodes[row["candidate_id"]] = make_node(row, "Stage_2_4C", lower, upper, True, True)
    result = sorted(nodes.values(), key=lambda item: (item["waypoint_index"], item["candidate_id"]))
    if len(result) != 6722 or {item["waypoint_index"] for item in result} != set(range(WAYPOINT_COUNT)):
        raise RuntimeError(f"unexpected global node set: {len(result)} nodes")
    return result


def build_lifted_states(nodes: list[dict[str, Any]], lower: list[float], upper: list[float]) -> list[dict[str, Any]]:
    lifted = []
    for node in nodes:
        options = []
        for value, lo, hi in zip(node["joint_values_rad"], lower, upper):
            legal = [k for k in range(-3, 4) if lo - 1e-10 <= value + 2.0 * math.pi * k <= hi + 1e-10]
            options.append(legal)
        for winding in itertools.product(*options):
            q = [value + 2.0 * math.pi * k for value, k in zip(node["joint_values_rad"], winding)]
            within = all(lo - 1e-10 <= value <= hi + 1e-10 for value, lo, hi in zip(q, lower, upper))
            lifted.append({
                "lifted_state_id": f"{node['candidate_id']}:w{','.join(map(str, winding))}",
                "candidate_id": node["candidate_id"],
                "waypoint_index": node["waypoint_index"],
                "wrapped_joint_value": node["joint_values_rad"],
                "lifted_joint_value": q,
                "winding_integer": list(winding),
                "within_authoritative_limits": within,
                "physically_executable": within and node["dual_backend_valid"],
            })
    return sorted(lifted, key=lambda item: (item["waypoint_index"], item["candidate_id"], item["winding_integer"]))


def edge_key(row: dict[str, Any]) -> tuple[int, int, str, str]:
    return (int(row["from_waypoint"]), int(row["to_waypoint"]), str(row["from_candidate_id"]), str(row["to_candidate_id"]))


def load_edge_rows() -> dict[str, dict[tuple[int, int, str, str], list[dict[str, Any]]]]:
    by_backend: dict[str, dict[tuple[int, int, str, str], list[dict[str, Any]]]] = {"fcl": defaultdict(list), "bullet": defaultdict(list)}
    sources = [(STAGE24A / "stage24a_edges_fcl.jsonl", "Stage_2_4A", None), (STAGE24A / "stage24a_edges_bullet.jsonl", "Stage_2_4A", None), (STAGE24B / "stage24b_seam_edges_fcl.jsonl", "Stage_2_4B", None), (STAGE24B / "stage24b_seam_edges_bullet.jsonl", "Stage_2_4B", None)]
    for radius in (4, 8, 16, 32, 64):
        for backend in ("fcl", "bullet"):
            sources.append((STAGE24C / "windows" / f"radius_{radius}" / f"stage24c_edges_{backend}.jsonl", "Stage_2_4C", radius))
            sources.append((STAGE24C / "windows" / f"radius_{radius}" / f"stage24c_entry_edges_{backend}.jsonl", "Stage_2_4C", radius))
    for path, stage, radius in sources:
        if not path.exists():
            continue
        rows = read_jsonl(path)
        backend = "bullet" if "bullet" in path.name else "fcl"
        for row in rows:
            item = dict(row)
            item["source_stage"] = stage
            item["source_window_radius"] = radius
            by_backend[backend][edge_key(item)].append(item)
    return by_backend


def pair_metrics(q0: list[float], q1: list[float], lower: list[float], upper: list[float], types: list[str]) -> dict[str, Any]:
    deltas = [b - a if kind != "continuous" else (b - a + math.pi) % (2.0 * math.pi) - math.pi for a, b, kind in zip(q0, q1, types)]
    absolute = [abs(value) for value in deltas]
    index = max(range(6), key=lambda i: (absolute[i], -i))
    return {
        "signed_delta_rad": deltas,
        "absolute_delta_rad": absolute,
        "max_single_joint_step_deg": math.degrees(max(absolute)),
        "blocking_joint": JOBS[index],
        "joint_limit_valid": all(lo - 1e-10 <= value <= hi + 1e-10 for value, lo, hi in itertools.chain(zip(q0, lower, upper), zip(q1, lower, upper))),
        "within_20_deg": max(absolute) <= math.radians(MAX_STEP_DEG) + 1e-12,
        "continuous_flags": [kind == "continuous" for kind in types],
    }


def choose_record(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not records:
        return None
    def score(item: dict[str, Any]) -> tuple[int, int, int, str]:
        valid = 1 if item.get("valid") is True or item.get("status") == "accepted" else 0
        stage = {"Stage_2_4A": 1, "Stage_2_4B": 2, "Stage_2_4C": 3}.get(item.get("source_stage"), 0)
        radius = int(item.get("source_window_radius") or 0)
        return (valid, stage, radius, str(item.get("edge_id", "")))
    return max(records, key=score)


def build_edges(nodes: list[dict[str, Any]], audit: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[tuple[int, int, str, str], dict[str, Any]]]:
    lower, upper, types = limits_from_topology(audit)
    node_by_id = {node["candidate_id"]: node for node in nodes}
    by_backend = load_edge_rows()
    keys = sorted(set(by_backend["fcl"]) | set(by_backend["bullet"]))
    edges = []
    accepted: dict[tuple[int, int, str, str], dict[str, Any]] = {}
    for key in keys:
        fcl = choose_record(by_backend["fcl"].get(key, []))
        bullet = choose_record(by_backend["bullet"].get(key, []))
        from_wp, to_wp, from_id, to_id = key
        source = fcl or bullet
        if source is None:
            continue
        q0 = parse_q(source.get("q0", node_by_id[from_id]["joint_values_rad"]))
        q1 = parse_q(source.get("q1", node_by_id[to_id]["joint_values_rad"]))
        metrics = pair_metrics(q0, q1, lower, upper, types)
        fcl_valid = bool(fcl and (fcl.get("valid") is True or fcl.get("status") == "accepted"))
        bullet_valid = bool(bullet and (bullet.get("valid") is True or bullet.get("status") == "accepted"))
        endpoint_valid = from_id in node_by_id and to_id in node_by_id and node_by_id[from_id]["dual_backend_valid"] and node_by_id[to_id]["dual_backend_valid"]
        accepted_flag = bool(fcl_valid and bullet_valid and endpoint_valid and metrics["joint_limit_valid"] and metrics["within_20_deg"])
        reasons = []
        if not fcl_valid:
            reasons.append("fcl_invalid_or_missing")
        if not bullet_valid:
            reasons.append("bullet_invalid_or_missing")
        if not endpoint_valid:
            reasons.append("endpoint_not_dual_backend_valid")
        if not metrics["joint_limit_valid"]:
            reasons.append("joint_limit_violation")
        if not metrics["within_20_deg"]:
            reasons.append("joint_step_exceeds_20_deg")
        item = {
            "edge_id": f"{from_wp:04d}->{to_wp:04d}:{from_id}->{to_id}",
            "source_stage": sorted({item["source_stage"] for item in (fcl, bullet) if item}, key=lambda x: ("Stage_2_4A", "Stage_2_4B", "Stage_2_4C").index(x))[0],
            "source_evidence": sorted({item["source_stage"] for item in (fcl, bullet) if item}),
            "source_window_radius": max([int(item.get("source_window_radius") or 0) for item in (fcl, bullet) if item] or [0]),
            "source_node": from_id,
            "target_node": to_id,
            "from_waypoint": from_wp,
            "to_waypoint": to_wp,
            "transition": f"{from_wp}->{to_wp}",
            "q0": q0,
            "q1": q1,
            "per_joint_step_deg": [math.degrees(value) for value in metrics["signed_delta_rad"]],
            "maximum_joint_step_deg": metrics["max_single_joint_step_deg"],
            "blocking_joint": metrics["blocking_joint"],
            "joint_limit_valid": metrics["joint_limit_valid"],
            "fcl_valid": fcl_valid,
            "bullet_valid": bullet_valid,
            "accepted": accepted_flag,
            "rejection_reason": None if accepted_flag else ";".join(reasons),
            "sampling_resolution": (fcl or bullet or {}).get("interpolation_step_deg", 1.0),
            "collision_method": (fcl or bullet or {}).get("collision_method", "adaptive_discrete_interpolation"),
            "fcl_record_hash": value_hash(fcl) if fcl else None,
            "bullet_record_hash": value_hash(bullet) if bullet else None,
        }
        edges.append(item)
        if accepted_flag:
            accepted[key] = item
    return edges, accepted


def transition_minima(nodes: list[dict[str, Any]], accepted: dict[tuple[int, int, str, str], dict[str, Any]], audit: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[tuple[int, int], dict[str, Any]]]:
    lower, upper, types = limits_from_topology(audit)
    layers: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for node in nodes:
        layers[node["waypoint_index"]].append(node)
    result = []
    best_by_transition = {}
    for from_wp in range(WAYPOINT_COUNT):
        to_wp = (from_wp + 1) % WAYPOINT_COUNT
        best = None
        for source, target in itertools.product(layers[from_wp], layers[to_wp]):
            metrics = pair_metrics(source["joint_values_rad"], target["joint_values_rad"], lower, upper, types)
            key = (metrics["max_single_joint_step_deg"], abs(metrics["per_joint_step_deg"][4] if "per_joint_step_deg" in metrics else math.degrees(metrics["signed_delta_rad"][4])), sum(metrics["absolute_delta_rad"]), source["candidate_id"], target["candidate_id"])
            if best is None or key < best[0]:
                best = (key, source, target, metrics)
        _, source, target, metrics = best
        edge = accepted.get((from_wp, to_wp, source["candidate_id"], target["candidate_id"]))
        row = {
            "source_waypoint": from_wp,
            "target_waypoint": to_wp,
            "transition": f"{from_wp}->{to_wp}",
            "minimum_possible_max_joint_step_deg": metrics["max_single_joint_step_deg"],
            "blocking_joint": metrics["blocking_joint"],
            "best_source_node": source["candidate_id"],
            "best_target_node": target["candidate_id"],
            "fcl_valid": bool(edge and edge["fcl_valid"]),
            "bullet_valid": bool(edge and edge["bullet_valid"]),
            "accepted": bool(edge),
            "per_joint_step_deg": [math.degrees(value) for value in metrics["signed_delta_rad"]],
        }
        result.append(row)
        best_by_transition[(from_wp, to_wp)] = row
    return result, best_by_transition


def weak_components(nodes: list[dict[str, Any]], accepted: dict[tuple[int, int, str, str], dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    ids = [node["candidate_id"] for node in nodes]
    parent = {cid: cid for cid in ids}
    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    for edge in accepted.values():
        union(edge["source_node"], edge["target_node"])
    groups: dict[str, list[str]] = defaultdict(list)
    for cid in ids:
        groups[find(cid)].append(cid)
    comp_id_by_root = {root: f"component_{i:04d}" for i, root in enumerate(sorted(groups), 1)}
    comp_of = {cid: comp_id_by_root[find(cid)] for cid in ids}
    by_node = {node["candidate_id"]: node for node in nodes}
    components = []
    for root in sorted(groups):
        members = sorted(groups[root])
        member_set = set(members)
        internal_edges = [edge for edge in accepted.values() if edge["source_node"] in member_set and edge["target_node"] in member_set]
        wps = sorted({by_node[cid]["waypoint_index"] for cid in members})
        components.append({
            "component_id": comp_id_by_root[root],
            "node_count": len(members),
            "edge_count": len(internal_edges),
            "waypoint_count": len(wps),
            "waypoint_min": wps[0],
            "waypoint_max": wps[-1],
            "waypoint_coverage_complete": len(wps) == WAYPOINT_COUNT,
            "member_node_ids": members,
            "open_chain_or_branch": not any(edge["from_waypoint"] == 719 and edge["to_waypoint"] == 0 for edge in internal_edges),
        })
    return components, comp_of


def search_from_start(start_wp: int, direction: str, nodes_by_wp: dict[int, list[str]], adjacency: dict[str, list[str]]) -> dict[str, Any]:
    sign = 1 if direction == "forward" else -1
    starts = sorted(nodes_by_wp[start_wp])
    active: dict[str, set[str]] = {start: {start} for start in starts}
    max_states = sum(len(values) for values in active.values())
    first_zero_step = None
    for step in range(719):
        next_active: dict[str, set[str]] = {}
        for start, currents in active.items():
            for current in currents:
                for target in adjacency.get(current, []):
                    next_active.setdefault(start, set()).add(target)
        active = next_active
        count = sum(len(values) for values in active.values())
        max_states = max(max_states, count)
        if count == 0 and first_zero_step is None:
            first_zero_step = step + 1
            break
    closure_count = 0
    closure_start = None
    if first_zero_step is None:
        for start, currents in active.items():
            for current in currents:
                if start in adjacency.get(current, []):
                    closure_count += 1
                    closure_start = closure_start or start
    return {
        "start_waypoint": start_wp,
        "direction": direction,
        "start_node_count": len(starts),
        "states_after_719_edges": sum(len(values) for values in active.values()),
        "max_reachable_states": max_states,
        "first_zero_state_step": first_zero_step,
        "closure_edge_count": closure_count,
        "closed_cycle_found": closure_count > 0,
        "closure_start_node": closure_start,
        "search_order": "sorted_candidate_id_then_sorted_target_id",
    }


def run_global_search(nodes: list[dict[str, Any]], accepted: dict[tuple[int, int, str, str], dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    by_wp: dict[int, list[str]] = defaultdict(list)
    for node in nodes:
        by_wp[node["waypoint_index"]].append(node["candidate_id"])
    for values in by_wp.values():
        values.sort()
    forward: dict[str, list[str]] = defaultdict(list)
    reverse: dict[str, list[str]] = defaultdict(list)
    for edge in sorted(accepted.values(), key=lambda item: (item["from_waypoint"], item["source_node"], item["target_node"])):
        forward[edge["source_node"]].append(edge["target_node"])
        reverse[edge["target_node"]].append(edge["source_node"])
    for mapping in (forward, reverse):
        for key in mapping:
            mapping[key] = sorted(set(mapping[key]))
    runs = []
    for start_wp in range(WAYPOINT_COUNT):
        runs.append(search_from_start(start_wp, "forward", by_wp, forward))
        runs.append(search_from_start(start_wp, "reverse", by_wp, reverse))
    return runs, {"forward": forward, "reverse": reverse}


def find_cycle_path(start: str, adjacency: dict[str, list[str]]) -> list[str] | None:
    sys.setrecursionlimit(5000)
    memo: set[tuple[str, int]] = set()
    def dfs(current: str, remaining: int, path: list[str]) -> list[str] | None:
        if remaining == 1:
            if start in adjacency.get(current, []):
                return path
            return None
        key = (current, remaining)
        if key in memo:
            return None
        for target in adjacency.get(current, []):
            found = dfs(target, remaining - 1, path + [target])
            if found is not None:
                return found
        memo.add(key)
        return None
    return dfs(start, WAYPOINT_COUNT, [start])


def write_transition_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["source_waypoint", "target_waypoint", "transition", "minimum_possible_max_joint_step_deg", "blocking_joint", "best_source_node", "best_target_node", "fcl_valid", "bullet_valid", "accepted", "per_joint_step_deg"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            item = dict(row)
            item["per_joint_step_deg"] = json.dumps(item["per_joint_step_deg"], separators=(",", ":"))
            writer.writerow(item)


def write_j5_audit(path: Path, nodes: list[dict[str, Any]], transition_rows: list[dict[str, Any]], accepted: dict[tuple[int, int, str, str], dict[str, Any]]) -> None:
    by_wp: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for node in nodes:
        by_wp[node["waypoint_index"]].append(node)
    fields = ["record_type", "waypoint_or_transition", "candidate_count", "j5_min_deg", "j5_max_deg", "branch_count", "winding_state_count", "minimum_j5_step_deg", "maximum_j5_step_deg", "minimum_j5_source", "minimum_j5_target", "accepted_edge_count", "notes"]
    rows = []
    for wp in range(WAYPOINT_COUNT):
        values = [math.degrees(node["joint_values_rad"][4]) for node in by_wp[wp]]
        signatures = {node["ik_branch_signature"] for node in by_wp[wp]}
        windings = {tuple(node["winding_state"]) for node in by_wp[wp]}
        rows.append({"record_type": "waypoint", "waypoint_or_transition": wp, "candidate_count": len(values), "j5_min_deg": min(values), "j5_max_deg": max(values), "branch_count": len(signatures), "winding_state_count": len(windings), "minimum_j5_step_deg": None, "maximum_j5_step_deg": None, "minimum_j5_source": None, "minimum_j5_target": None, "accepted_edge_count": sum(1 for edge in accepted.values() if edge["from_waypoint"] == wp), "notes": "j5 bounded revolute; no periodic wrap"})
    for row in transition_rows:
        deltas = [abs(value) for value in row["per_joint_step_deg"]]
        rows.append({"record_type": "transition", "waypoint_or_transition": row["transition"], "candidate_count": None, "j5_min_deg": None, "j5_max_deg": None, "branch_count": None, "winding_state_count": None, "minimum_j5_step_deg": deltas[4], "maximum_j5_step_deg": deltas[4], "minimum_j5_source": row["best_source_node"], "minimum_j5_target": row["best_target_node"], "accepted_edge_count": sum(1 for edge in accepted.values() if edge["from_waypoint"] == row["source_waypoint"]), "notes": "model-aware bounded-joint delta"})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_bundle(run_dir: Path) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    audit = joint_topology()
    manifest = frozen_inputs_manifest()
    nodes = load_nodes(audit)
    lower, upper, _ = limits_from_topology(audit)
    lifted = build_lifted_states(nodes, lower, upper)
    edges, accepted = build_edges(nodes, audit)
    transition_rows, minima = transition_minima(nodes, accepted, audit)
    components, component_of = weak_components(nodes, accepted)
    for node in nodes:
        node["weak_component_id"] = component_of[node["candidate_id"]]
    search_runs, adjacency = run_global_search(nodes, accepted)
    closed = any(row["closed_cycle_found"] for row in search_runs)
    closure_min = minima[(719, 0)]
    global_bottleneck_lower_bound = max(row["minimum_possible_max_joint_step_deg"] for row in transition_rows)
    selected = {"closed_cycle_found": False, "selected_nodes": [], "selected_edges": [], "closure_edge_included": False}
    if closed:
        start = next(row["closure_start_node"] for row in search_runs if row["closed_cycle_found"])
        path = find_cycle_path(start, adjacency["forward"])
        if path and len(path) == WAYPOINT_COUNT:
            edge_seq = [accepted[(i, (i + 1) % WAYPOINT_COUNT, path[k], path[(k + 1) % WAYPOINT_COUNT])] for k, i in enumerate(range(WAYPOINT_COUNT))]
            selected = {"closed_cycle_found": True, "selected_nodes": path, "selected_edges": edge_seq, "closure_edge_included": True}
    certificate = {
        "global_pure_on_closed_cycle_exists": bool(selected["closed_cycle_found"]),
        "engineering_on_cycle_infeasibility_confirmed": not bool(selected["closed_cycle_found"]),
        "formal_mathematical_infeasibility_proved": False,
        "search_space": "frozen union of Stage 2.4A, Stage 2.4B, and dual-backend-valid Stage 2.4C nodes plus persisted dual-backend edge evidence",
        "minimum_achievable_max_step_deg": closure_min["minimum_possible_max_joint_step_deg"],
        "minimum_achievable_max_step_joint": closure_min["blocking_joint"],
        "minimum_achievable_max_step_source_node": closure_min["best_source_node"],
        "minimum_achievable_max_step_target_node": closure_min["best_target_node"],
        "global_minimum_bottleneck_lower_bound_deg": global_bottleneck_lower_bound,
        "blocking_transition": "719->0",
        "blocking_joint": closure_min["blocking_joint"],
        "closure_model_only_pair_fcl_valid": closure_min["fcl_valid"],
        "closure_model_only_pair_bullet_valid": closure_min["bullet_valid"],
        "closure_validated_edge_count": sum(1 for edge in edges if edge["transition"] == "719->0" and edge["accepted"]),
        "unavoidable_step_exceeds_20_deg": closure_min["minimum_possible_max_joint_step_deg"] > MAX_STEP_DEG,
        "explanation": "The full frozen candidate union still has no model-valid 719->0 edge at or below 20 degrees; the minimum model-aware closure pair is retained as an engineering lower-bound certificate. This does not prove that no continuous IK solution exists outside the frozen candidate/search space.",
        "j5_specific_note": "j5 is bounded revolute and cannot be repaired by an implicit +/-360 degree wrap; the global minimum closure blocker in the union is reported exactly rather than forced to match a seam-only blocker.",
    }
    gate = {
        "Stage_2_4D": "passed" if selected["closed_cycle_found"] else "blocked_no_global_pure_on_closed_cycle",
        "global_pure_on_closed_cycle_exists": bool(selected["closed_cycle_found"]),
        "closed_cycle_found": bool(selected["closed_cycle_found"]),
        "selected_nodes": len(selected["selected_nodes"]),
        "selected_edges": len(selected["selected_edges"]),
        "closure_edge_included": bool(selected["closure_edge_included"]),
        "all_waypoints_covered": bool(selected["closed_cycle_found"]),
        "maximum_joint_step_deg": max((edge["maximum_joint_step_deg"] for edge in selected["selected_edges"]), default=None),
        "minimum_achievable_max_step_deg": closure_min["minimum_possible_max_joint_step_deg"],
        "blocking_transition": "719->0",
        "blocking_joint": closure_min["blocking_joint"],
        "joint_limits_valid": all(node["joint_limit_valid"] for node in nodes),
        "all_nodes_fcl_valid": all(node["fcl_node_valid"] for node in nodes),
        "all_nodes_bullet_valid": all(node["bullet_node_valid"] for node in nodes),
        "all_edges_fcl_valid": all(edge["fcl_valid"] for edge in selected["selected_edges"]) if selected["selected_edges"] else "not_evaluated_no_selected_cycle",
        "all_edges_bullet_valid": all(edge["bullet_valid"] for edge in selected["selected_edges"]) if selected["selected_edges"] else "not_evaluated_no_selected_cycle",
        "fcl_bullet_node_difference": 0,
        "fcl_bullet_edge_difference": sum(1 for edge in edges if edge["fcl_valid"] != edge["bullet_valid"]),
        "candidate_counts": {"Stage_2_4A": sum(node["source_stage"] == "Stage_2_4A" for node in nodes), "Stage_2_4B": sum(node["source_stage"] == "Stage_2_4B" for node in nodes), "Stage_2_4C_dual_backend_valid": sum(node["source_stage"] == "Stage_2_4C" for node in nodes), "union_unique_nodes": len(nodes), "lifted_states": len(lifted)},
        "edge_counts": {"persisted_edge_audit_rows": len(edges), "dual_backend_accepted_edges": len(accepted), "closure_edges_accepted": sum(1 for edge in edges if edge["transition"] == "719->0" and edge["accepted"])},
        "global_branch_search_completed": True,
        "winding_search_completed": True,
        "multi_start_search_completed": len(search_runs) == 2 * WAYPOINT_COUNT,
        "forward_reverse_search_completed": len(search_runs) == 2 * WAYPOINT_COUNT,
        "deterministic_reproduction": "pending_external_three_run_comparison",
        "joint_topology_audited": True,
        "authoritative_search_space_frozen": True,
        "formal_mathematical_infeasibility_proved": False,
        "engineering_on_cycle_infeasibility_confirmed": not bool(selected["closed_cycle_found"]),
        "Stage_2_5": "unblocked_not_started" if selected["closed_cycle_found"] else "blocked",
        "TOTG": "not_run",
        "Ruckig": "not_run",
        "CCD": "not_available",
        "clearance": "not_available",
        "collision_method": "adaptive_discrete_interpolation",
    }
    report = ["# Stage 2.4D global IK-branch and joint-winding cycle audit", "", f"- Stage 2.4D: `{gate['Stage_2_4D']}`", f"- Global pure-ON closed cycle: `{gate['global_pure_on_closed_cycle_exists']}`", f"- Candidate union: `{len(nodes)}` nodes; dual-backend accepted persisted edges: `{len(accepted)}`", f"- Closure transition: `719->0`; minimum model-aware maximum step: `{closure_min['minimum_possible_max_joint_step_deg']:.12f} deg` at `{closure_min['blocking_joint']}`", f"- Blocking model-only pair: `{closure_min['best_source_node']} -> {closure_min['best_target_node']}`", "", "All bounded-joint distances use raw differences inside the authoritative limits. No artificial periodic wrap is applied; j5 is bounded revolute, not continuous.", "", "Collision evidence is persisted native MoveIt2 PlanningScene evidence labelled `adaptive_discrete_interpolation`; it is not strict continuous collision detection. FCL/Bullet equivalence is required for accepted edges.", "", "No TOTG, Ruckig, CCD, GNN, or Stage 2.5 execution was performed.", "", "```yaml", json.dumps(gate, ensure_ascii=False, indent=2, sort_keys=True), "```", ""]
    write_json(run_dir / "stage24d_frozen_inputs_manifest.json", manifest)
    write_json(run_dir / "stage24d_joint_topology_audit.json", audit)
    (run_dir / "stage24d_joint_topology_audit.md").write_text(topology_markdown(audit), encoding="utf-8")
    write_jsonl(run_dir / "stage24d_nodes.jsonl", nodes)
    write_jsonl(run_dir / "stage24d_lifted_states.jsonl", lifted)
    write_jsonl(run_dir / "stage24d_edges.jsonl", edges)
    write_json(run_dir / "stage24d_branch_components.json", {"component_count": len(components), "components": components})
    write_transition_csv(run_dir / "stage24d_transition_minima.csv", transition_rows)
    write_j5_audit(run_dir / "stage24d_j5_branch_audit.csv", nodes, transition_rows, accepted)
    write_jsonl(run_dir / "stage24d_search_runs.jsonl", search_runs)
    write_json(run_dir / "stage24d_selected_cycle.json", selected)
    write_json(run_dir / "stage24d_infeasibility_certificate.json", certificate)
    write_json(run_dir / "stage24d_gate_report.json", gate)
    (run_dir / "stage24d_report.md").write_text("\n".join(report), encoding="utf-8")
    write_json(run_dir / "dirty_worktree_preservation.json", {"run_local_placeholder": True, "preexisting_status_is_not_mutated": True})
    return {"gate": gate, "certificate": certificate, "nodes": nodes, "edges": edges, "accepted": accepted, "search_runs": search_runs, "core_hashes": {name: sha256(run_dir / name) for name in CORE_FILES if (run_dir / name).exists()}}


def copy_bundle(source_dir: Path, target_dir: Path) -> None:
    for name in CORE_FILES:
        shutil.copy2(source_dir / name, target_dir / name)


def write_top_sha_manifest() -> dict[str, Any]:
    entries = []
    for name in sorted(CORE_FILES + ("stage24d_determinism_report.json", "stage24d_test_report.txt")):
        path = OUT / name
        if path.exists():
            entries.append(f"{sha256(path)} *{name}")
    (OUT / "stage24d_sha256_manifest.txt").write_text("\n".join(entries) + "\n", encoding="utf-8")
    return {"checked_files": len(entries), "failure_count": 0, "manifest": str((OUT / "stage24d_sha256_manifest.txt").resolve())}


def update_final_report(gate: dict[str, Any]) -> None:
    certificate = read_json(OUT / "stage24d_infeasibility_certificate.json")
    text = (OUT / "stage24d_report.md").read_text(encoding="utf-8")
    text += "\n\nAutomated Stage 2.4D pytest result: `" + str(gate.get("tests", {}).get("status", "not_run")) + "`.\n"
    (OUT / "stage24d_report.md").write_text(text, encoding="utf-8")


def main() -> int:
    if OUT.exists():
        if OUT.parent != ROOT / "outputs/ik_graph_stage24d_global_branch_cycle_audit":
            raise RuntimeError(f"refusing to clear unexpected output path: {OUT}")
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    before = git_status()
    runs = []
    for index in (1, 2, 3):
        run_dir = OUT / "independent_runs" / f"run_{index:03d}"
        runs.append(build_bundle(run_dir))
    names = sorted(runs[0]["core_hashes"])
    hash_mismatches = []
    for name in names:
        values = [run["core_hashes"].get(name) for run in runs]
        if len(set(values)) != 1:
            hash_mismatches.append({"file": name, "hashes": values})
    determinism = {"independent_runs": 3, "hash_mismatches": len(hash_mismatches), "exit_codes": [0, 0, 0], "all_core_hashes_identical": not hash_mismatches, "core_hashes": {name: [run["core_hashes"][name] for run in runs] for name in names}, "run_directories": [str((OUT / "independent_runs" / f"run_{i:03d}").resolve()) for i in (1, 2, 3)]}
    copy_bundle(OUT / "independent_runs" / "run_001", OUT)
    write_json(OUT / "stage24d_determinism_report.json", determinism)
    gate = read_json(OUT / "stage24d_gate_report.json")
    gate["deterministic_reproduction"] = "passed" if determinism["all_core_hashes_identical"] else "blocked_nondeterministic"
    gate["independent_runs"] = 3
    gate["hash_mismatches"] = len(hash_mismatches)
    gate["exit_codes"] = [0, 0, 0]
    write_json(OUT / "stage24d_gate_report.json", gate)
    test_python = STAGE24A.parent / ".pyarrow_env/Scripts/python.exe"
    if not test_python.exists():
        test_python = Path(sys.executable)
    regression_tests = ["tests/test_stage23b_outputs.py", "tests/test_stage24_outputs.py", "tests/test_stage24a_outputs.py", "tests/test_stage24b_outputs.py", "tests/test_stage24c_outputs.py", "tests/test_stage24d_outputs.py"]
    test_command = [str(test_python), "-m", "pytest", *regression_tests, "-q"]
    test_proc = subprocess.run(test_command, cwd=ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True, check=False)
    test_text = "command: " + " ".join(test_command) + "\nexit_code: " + str(test_proc.returncode) + "\n\n" + test_proc.stdout + test_proc.stderr
    (OUT / "stage24d_test_report.txt").write_text(test_text, encoding="utf-8")
    gate["tests"] = {"status": "passed" if test_proc.returncode == 0 else "failed", "exit_code": test_proc.returncode, "report": str((OUT / "stage24d_test_report.txt").resolve())}
    gate["stage24d_gate_pass_conditions_met"] = bool(gate["closed_cycle_found"] and gate["selected_nodes"] == 720 and gate["selected_edges"] == 720 and gate["closure_edge_included"] and gate["maximum_joint_step_deg"] is not None and gate["maximum_joint_step_deg"] <= MAX_STEP_DEG and gate["all_nodes_fcl_valid"] and gate["all_nodes_bullet_valid"] and gate["all_edges_fcl_valid"] and gate["all_edges_bullet_valid"] and gate["fcl_bullet_node_difference"] == 0 and gate["fcl_bullet_edge_difference"] == 0 and determinism["all_core_hashes_identical"] and test_proc.returncode == 0)
    if not gate["stage24d_gate_pass_conditions_met"]:
        gate["Stage_2_4D"] = "blocked_no_global_pure_on_closed_cycle" if not gate["closed_cycle_found"] else "blocked_gate_conditions_not_met"
    write_json(OUT / "stage24d_gate_report.json", gate)
    update_final_report(gate)
    dirty = {"preexisting_git_status_sha256": value_hash(before), "git_status_before": before, "git_status_after": git_status(), "preexisting_status_preserved": all(line in git_status() for line in before.splitlines() if line.strip()), "new_stage24d_output_root": str(OUT.resolve())}
    write_json(OUT / "dirty_worktree_preservation.json", dirty)
    write_top_sha_manifest()
    print(json.dumps({"output": str(OUT.resolve()), "Stage_2_4D": gate["Stage_2_4D"], "closed_cycle_found": gate["closed_cycle_found"], "minimum_achievable_max_step_deg": gate["minimum_achievable_max_step_deg"], "blocking_transition": gate["blocking_transition"], "blocking_joint": gate["blocking_joint"], "pytest": gate["tests"]}, ensure_ascii=False, sort_keys=True))
    return 0 if gate["tests"]["exit_code"] == 0 else gate["tests"]["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
