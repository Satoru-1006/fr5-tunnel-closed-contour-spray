"""Stage 2.4G complete adjacent-edge materialization and component bridge.

This runner is additive and fail-closed.  It reads the frozen Stage 2.4F
manifest and the Stage 2.4D/E/F graph evidence, canonicalizes branch labels
from the actual joint vector, materializes every missing raw-20-degree
adjacent pair that can affect the open chain, and recomputes forward/reverse
reachability after every materialization round.  Only after the existing-node
space has been exhausted does it generate bounded Stage 2.4E-tolerance IK
 candidates at true component boundaries.

The native edge checker is the existing MoveIt2 PlanningScene probe.  The
collision result is ``adaptive_discrete_interpolation``; CCD and positive
clearance remain unavailable.  No URDF, SRDF/ACM, PlanningScene, collision
geometry, nominal waypoint, or prior-stage output is modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
import traceback
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_stage24_closed_loop_graph as stage24
import scripts.run_stage24e_tolerance_search as stage24e
from src.deterministic_numeric_ik import DeterministicNumericIKSolver
from src.certified_output_guard import assert_output_path_writable


FROZEN_F = ROOT / "outputs/ik_graph_stage24f_global_branch_adaptive_closure/fr5_scaled_horseshoe_demo_v45_20260731_final4x"
STAGE24D = ROOT / "outputs/ik_graph_stage24d_global_branch_cycle_audit/fr5_scaled_horseshoe_demo_v45"
STAGE24E = ROOT / "outputs/ik_graph_stage24e_tolerance_search/fr5_scaled_horseshoe_demo_v45_retry_4_20260731"
SOURCE = ROOT / "outputs/ik_graph_stage23a4_backend_equivalent/fr5_scaled_horseshoe_demo_v44"
STAGE23B = ROOT / "outputs/ik_graph_stage23b_representation_remediation/fr5_scaled_horseshoe_demo_v45"
URDF = ROOT / "outputs/ik_graph_stage193/expanded_runtime_urdf.urdf"
LIMITS = ROOT / "ros2_moveit_bridge/config/joint_limits_with_jerk.yaml"
PARTS = SOURCE / "tunnel_collision_parts"
WAYPOINT_COUNT = 720
MAX_STEP_DEG = 20.0
INTERPOLATION_STEP_DEG = 1.0
BACKENDS = ("fcl", "bullet")
NATIVE_RUN_INDEX = 2
FRAGILE_TRANSITIONS = tuple(
    [f"{i}->{i + 1}" for i in range(199, 227)]
    + ["301->302", "302->303", "402->403", "527->528"]
)
DEFAULT_OUT = ROOT / "outputs/ik_graph_stage24g_complete_edge_component_bridge/fr5_scaled_horseshoe_demo_v45_20260731"


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


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    assert_output_path_writable(path, operation="overwrite_json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    assert_output_path_writable(path, operation="truncate_jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    assert_output_path_writable(path, operation="truncate_csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fields or (list(rows[0]) if rows else ["empty"])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_q(value: Any) -> list[float]:
    if isinstance(value, str):
        return [float(x) for x in json.loads(value)]
    return [float(x) for x in value]


def wsl_path(path: Path) -> str:
    resolved = str(path.resolve()).replace("\\", "/")
    drive, rest = resolved.split(":/", 1)
    return f"/mnt/{drive.lower()}/{rest}"


def semantic_hash(value: Any) -> str:
    if isinstance(value, dict):
        return value_hash({k: semantic_hash(v) if isinstance(v, (dict, list)) else v for k, v in value.items() if k not in {"path", "output", "run_duration_s", "captured_at_utc", "started_at_utc", "finished_at_utc", "run_index"}})
    if isinstance(value, list):
        return value_hash([semantic_hash(v) if isinstance(v, (dict, list)) else v for v in value])
    return value_hash(value)


def git_status() -> str:
    proc = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True, check=False)
    return proc.stdout + proc.stderr


def git_snapshot() -> dict[str, Any]:
    return {"status_porcelain": git_status()}


def source_ref(path: Path, role: str) -> dict[str, Any]:
    return {"path": str(path.resolve()), "role": role, "exists": path.exists(), "size": path.stat().st_size if path.is_file() else None, "sha256": sha256(path) if path.is_file() else None}


def recompute_frozen_hashes() -> dict[str, Any]:
    manifest_path = FROZEN_F / "stage24f_input_manifest.json"
    if not manifest_path.exists():
        return {"manifest_path": str(manifest_path.resolve()), "manifest_exists": False, "expected_count": None, "checked_count": 0, "failure_count": 1, "failures": [{"reason": "missing_stage24f_input_manifest"}]}
    frozen = read_json(manifest_path)
    failures: list[dict[str, Any]] = []
    checked: list[dict[str, Any]] = []
    for item in frozen.get("files", []):
        path = Path(str(item["path"]))
        expected = item.get("sha256")
        actual = sha256(path) if path.is_file() else None
        row = {"path": str(path.resolve()), "role": item.get("role"), "expected_sha256": expected, "actual_sha256": actual, "exists": path.exists(), "match": actual == expected}
        checked.append(row)
        if actual != expected:
            failures.append({"path": str(path.resolve()), "expected": expected, "actual": actual, "reason": "missing_file" if actual is None else "sha256_mismatch"})
    return {"manifest_path": str(manifest_path.resolve()), "manifest_sha256": sha256(manifest_path), "manifest_exists": True, "expected_count": len(frozen.get("files", [])), "checked_count": len(checked), "failure_count": len(failures), "failures": failures, "entries": checked, "all_397_inputs_verified": len(checked) == 397 and not failures}


def authoritative_input_manifest(before: dict[str, Any], frozen_hashes: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "stage24g-input-manifest-v1",
        "stage": "2.4G",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage24f_frozen_input_manifest": source_ref(FROZEN_F / "stage24f_input_manifest.json", "Stage_2_4F frozen input manifest read-only authority"),
        "stage24f_frozen_input_hash_verification": {k: v for k, v in frozen_hashes.items() if k != "entries"},
        "frozen_input_count_required": 397,
        "frozen_input_hashes_unchanged": bool(frozen_hashes.get("all_397_inputs_verified")),
        "stage24d": source_ref(STAGE24D / "stage24d_frozen_inputs_manifest.json", "Stage_2_4D frozen input manifest"),
        "stage24e": source_ref(STAGE24E / "stage24e_search_manifest.json", "Stage_2_4E frozen tolerance search contract"),
        "stage24f": source_ref(FROZEN_F / "stage24f_gate_report.json", "Stage_2_4F frozen formal gate"),
        "URDF_SRDF_ACM_PlanningScene_geometry_nominal_waypoints_read_only": True,
        "joint_step_threshold_deg": MAX_STEP_DEG,
        "collision_method": "adaptive_discrete_interpolation",
        "CCD": "not_available",
        "clearance": "not_available",
        "git_before": before,
        "authoritative_paths": [
            source_ref(ROOT / "external/frcobot_ros2/fairino_description/urdf/fairino5_v6.urdf", "URDF"),
            source_ref(URDF, "runtime URDF"),
            source_ref(ROOT / "ros2_moveit_bridge/config/fairino5_v6_spray_tcp.srdf", "SRDF and ACM"),
            source_ref(ROOT / "ros2_moveit_bridge/config/fairino5_v6_spray_tcp.urdf.xacro", "TCP model"),
            source_ref(LIMITS, "joint limits"),
            source_ref(SOURCE / "waypoints.csv", "nominal 720 waypoints"),
            source_ref(SOURCE / "scaled_demo_parameters.yaml", "scaled demo process and scene"),
            source_ref(PARTS, "collision geometry directory"),
        ],
    }


def joint_semantics() -> list[dict[str, Any]]:
    return stage24.parse_joint_semantics()


def canonical_branch_signature(q: list[float], semantics: list[dict[str, Any]]) -> str:
    """Return one authoritative q-derived branch label.

    The label is an audit field only.  It is never used as a hard edge
    rejection.  All six FR5 joints are bounded revolute, so no periodic wrap
    is applied while computing winding metadata.
    """
    if len(q) != len(semantics):
        raise ValueError(f"joint_dimension_mismatch:{len(q)}:{len(semantics)}")
    signs = ("positive" if q[0] >= 0.0 else "negative", "positive" if q[2] >= 0.0 else "negative", "positive" if q[4] >= 0.0 else "negative")
    winding: list[int] = []
    for value, rule in zip(q, semantics):
        lower = rule.get("lower")
        upper = rule.get("upper")
        options = [k for k in range(-3, 4) if lower is not None and upper is not None and float(lower) - 1e-10 <= value + 2.0 * math.pi * k <= float(upper) + 1e-10]
        winding.append(options[0] if options else 999)
    return f"shoulder={signs[0]}|elbow={signs[1]}|wrist={signs[2]}|winding={','.join(map(str, winding))}"


def node_q(row: dict[str, Any]) -> list[float]:
    return parse_q(row.get("joint_vector_rad", row.get("joint_values_rad", row.get("joint_values"))))


def node_wp(row: dict[str, Any]) -> int:
    return int(row.get("waypoint_index", row.get("waypoint_id")))


def node_is_valid(row: dict[str, Any], fcl: dict[str, bool] | None = None, bullet: dict[str, bool] | None = None) -> bool:
    cid = str(row["candidate_id"])
    if fcl is not None or bullet is not None:
        return bool((fcl or {}).get(cid) is True and (bullet or {}).get(cid) is True)
    return bool(row.get("dual_backend_valid") is True or (row.get("fcl_node_valid") is True and row.get("bullet_node_valid") is True))


def normalized_node(row: dict[str, Any], source_stage: str, semantics: list[dict[str, Any]], provenance: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    q = node_q(row)
    signature = canonical_branch_signature(q, semantics)
    return {
        "waypoint_index": node_wp(row),
        "candidate_id": str(row["candidate_id"]),
        "joint_vector_rad": q,
        "joint_vector_deg": [math.degrees(v) for v in q],
        "canonical_branch_signature": signature,
        "ik_branch_signature": signature,
        "source_stage": source_stage,
        "source_branch_signature": row.get("ik_branch_signature"),
        "branch_signature_valid": row.get("ik_branch_signature") in (None, "", signature),
        "shoulder_branch": "positive" if q[0] >= 0 else "negative",
        "elbow_branch": "positive" if q[2] >= 0 else "negative",
        "wrist_branch": "positive" if q[4] >= 0 else "negative",
        "fcl_node_valid": True,
        "bullet_node_valid": True,
        "dual_backend_valid": True,
        "joint_limit_valid": bool(row.get("joint_limit_valid", row.get("joint_limit_pass", True))),
        "process_tolerance_pass": bool(row.get("process_tolerance_pass", True)),
        "process_tolerance_error": None if row.get("process_tolerance_pass", True) else "process_tolerance_failed",
        "pose_error": row.get("pose_error", row.get("FK_position_error_m", row.get("fk_position_error"))),
        "position_offset_local_m": row.get("position_offset_local_m"),
        "standoff_offset_m": row.get("standoff_offset_m"),
        "normal_rotation_vector_deg": row.get("normal_rotation_vector_deg"),
        "roll_offset_deg": row.get("roll_offset_deg"),
        "candidate_parameter_hash": row.get("candidate_parameter_hash"),
        "provenance": provenance or [{"source_stage": source_stage, "candidate_id": str(row["candidate_id"]), "source_branch_signature": row.get("ik_branch_signature")}],
    }


def _load_source_nodes(semantics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    d_rows = read_jsonl(STAGE24D / "stage24d_nodes.jsonl")
    d_rows = [row for row in d_rows if node_is_valid(row)]
    e_rows = read_jsonl(STAGE24E / "stage24e_candidates.jsonl")
    e_fcl = {str(row["candidate_id"]): row.get("valid") is True for row in read_jsonl(STAGE24E / "stage24e_nodes_fcl.jsonl")}
    e_bullet = {str(row["candidate_id"]): row.get("valid") is True for row in read_jsonl(STAGE24E / "stage24e_nodes_bullet.jsonl")}
    e_rows = [row for row in e_rows if node_is_valid(row, e_fcl, e_bullet)]
    f_rows = [row for row in read_jsonl(FROZEN_F / "stage24f_layered_graph_nodes.jsonl") if node_is_valid(row)]
    rows: list[dict[str, Any]] = []
    rows.extend(normalized_node(row, "Stage_2_4D", semantics) for row in d_rows)
    rows.extend(normalized_node(row, "Stage_2_4E", semantics) for row in e_rows)
    rows.extend(normalized_node(row, "Stage_2_4F", semantics) for row in f_rows)
    return rows


def load_and_deduplicate_valid_nodes(semantics: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, str], dict[str, Any]]:
    grouped: dict[tuple[int, tuple[float, ...]], list[dict[str, Any]]] = defaultdict(list)
    source_rows = _load_source_nodes(semantics)
    for row in source_rows:
        key = (int(row["waypoint_index"]), tuple(round(float(value), 10) for value in row["joint_vector_rad"]))
        grouped[key].append(row)
    nodes: dict[str, dict[str, Any]] = {}
    aliases: dict[str, str] = {}
    duplicate_groups = 0
    for key in sorted(grouped):
        rows = sorted(grouped[key], key=lambda row: (str(row["candidate_id"]), str(row["source_stage"])))
        representative = dict(rows[0])
        cid = str(representative["candidate_id"])
        provenance: list[dict[str, Any]] = []
        for row in rows:
            aliases[str(row["candidate_id"])] = cid
            provenance.extend(row.get("provenance", []))
        representative["provenance"] = sorted(provenance, key=lambda item: (str(item.get("source_stage")), str(item.get("candidate_id"))))
        representative["dedup_key"] = {"waypoint_index": key[0], "rounded_q_rad": list(key[1])}
        nodes[cid] = representative
        if len(rows) > 1:
            duplicate_groups += 1
    audit = {"source_row_count": len(source_rows), "deduplicated_node_count": len(nodes), "duplicate_physical_node_groups": duplicate_groups, "valid_waypoint_count": len({int(row["waypoint_index"]) for row in nodes.values()}), "node_set_hash": semantic_hash(sorted((cid, row["waypoint_index"], row["joint_vector_rad"], row["canonical_branch_signature"]) for cid, row in nodes.items()))}
    return nodes, aliases, audit


def edge_key(row: dict[str, Any]) -> tuple[int, int, str, str]:
    return (int(row["from_waypoint"]), int(row["to_waypoint"]), str(row.get("from_candidate_id", row.get("source_node"))), str(row.get("to_candidate_id", row.get("target_node"))))


def edge_from_q(from_wp: int, to_wp: int, source: str, target: str, q0: list[float], q1: list[float], source_stage: str, source_evidence: list[str] | None = None) -> dict[str, Any]:
    signed = [b - a for a, b in zip(q0, q1)]
    absolute = [abs(value) for value in signed]
    maximum = math.degrees(max(absolute, default=0.0))
    return {
        "from_waypoint": from_wp,
        "to_waypoint": to_wp,
        "from_candidate_id": source,
        "to_candidate_id": target,
        "q0": q0,
        "q1": q1,
        "signed_joint_delta_rad": signed,
        "signed_joint_delta_deg": [math.degrees(value) for value in signed],
        "absolute_joint_delta_rad": absolute,
        "absolute_joint_delta_deg": [math.degrees(value) for value in absolute],
        "max_joint_delta_deg": maximum,
        "blocking_joint": f"j{int(np.argmax(absolute)) + 1}" if absolute else None,
        "joint_gate_valid": maximum <= MAX_STEP_DEG + 1e-12,
        "fcl_edge_valid": True,
        "bullet_edge_valid": True,
        "collision_method": "adaptive_discrete_interpolation",
        "interpolation_step_deg": INTERPOLATION_STEP_DEG,
        "source_stage": source_stage,
        "source_evidence": source_evidence or [source_stage],
        "edge_id": f"{from_wp:04d}->{to_wp:04d}:{source}->{target}",
    }


def _load_historical_edges(nodes: dict[str, dict[str, Any]], aliases: dict[str, str]) -> dict[tuple[int, int, str, str], dict[str, Any]]:
    accepted: dict[tuple[int, int, str, str], dict[str, Any]] = {}

    def add(row: dict[str, Any], source_stage: str) -> None:
        if not (row.get("accepted") is True or (row.get("status") == "accepted" and row.get("valid") is True) or (row.get("fcl_edge_valid") is True and row.get("bullet_edge_valid") is True)):
            return
        from_id = aliases.get(str(row.get("from_candidate_id", row.get("source_node"))), str(row.get("from_candidate_id", row.get("source_node"))))
        to_id = aliases.get(str(row.get("to_candidate_id", row.get("target_node"))), str(row.get("to_candidate_id", row.get("target_node"))))
        if from_id not in nodes or to_id not in nodes:
            return
        from_wp, to_wp = int(row["from_waypoint"]), int(row["to_waypoint"])
        edge = edge_from_q(from_wp, to_wp, from_id, to_id, nodes[from_id]["joint_vector_rad"], nodes[to_id]["joint_vector_rad"], source_stage, row.get("source_evidence") or [source_stage])
        key = edge_key(edge)
        old = accepted.get(key)
        if old is None:
            accepted[key] = edge
        else:
            old["source_evidence"] = sorted(set(old.get("source_evidence", [])) | set(edge.get("source_evidence", [])))

    for row in read_jsonl(STAGE24D / "stage24d_edges.jsonl"):
        add(row, "Stage_2_4D")
    for row in read_jsonl(STAGE24E / "stage24e_edges_fcl.jsonl"):
        add(row, "Stage_2_4E")
    for row in read_jsonl(FROZEN_F / "stage24f_layered_graph_edges.jsonl"):
        add(row, str(row.get("source_stage", "Stage_2_4F")))
    return accepted


def raw_delta(q0: list[float], q1: list[float], semantics: list[dict[str, Any]]) -> tuple[list[float], list[float], bool, str | None]:
    signed = [b - a for a, b in zip(q0, q1)]
    absolute = [abs(value) for value in signed]
    limits_ok = True
    for value, rule in zip(q0 + q1, semantics + semantics):
        if rule.get("lower") is not None and not (float(rule["lower"]) - 1e-12 <= value <= float(rule["upper"]) + 1e-12):
            limits_ok = False
    blocking = f"j{int(np.argmax(absolute)) + 1}" if absolute else None
    return signed, absolute, limits_ok, blocking


def enumerate_adjacent_pair_inventory(nodes: dict[str, dict[str, Any]], historical_edges: dict[tuple[int, int, str, str], dict[str, Any]], semantics: list[dict[str, Any]], *, include_only_new_ids: set[str] | None = None) -> list[dict[str, Any]]:
    by_wp: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in nodes.values():
        if row["dual_backend_valid"] and row["joint_limit_valid"] and row["process_tolerance_pass"]:
            by_wp[int(row["waypoint_index"])].append(row)
    for rows in by_wp.values():
        rows.sort(key=lambda row: str(row["candidate_id"]))
    inventory: list[dict[str, Any]] = []
    for from_wp in range(WAYPOINT_COUNT):
        to_wp = (from_wp + 1) % WAYPOINT_COUNT
        for source in by_wp.get(from_wp, []):
            for target in by_wp.get(to_wp, []):
                signed, absolute, limits_ok, blocking = raw_delta(source["joint_vector_rad"], target["joint_vector_rad"], semantics)
                maximum = math.degrees(max(absolute, default=0.0))
                key = (from_wp, to_wp, source["candidate_id"], target["candidate_id"])
                historical = key in historical_edges
                gate_pass = limits_ok and maximum <= MAX_STEP_DEG + 1e-12
                inventory.append({
                    "transition": f"{from_wp}->{to_wp}",
                    "from_waypoint": from_wp,
                    "to_waypoint": to_wp,
                    "from_candidate_id": source["candidate_id"],
                    "to_candidate_id": target["candidate_id"],
                    "q0": source["joint_vector_rad"],
                    "q1": target["joint_vector_rad"],
                    "per_joint_delta_rad": signed,
                    "per_joint_delta_deg": [math.degrees(value) for value in signed],
                    "absolute_joint_delta_deg": [math.degrees(value) for value in absolute],
                    "max_joint_delta_deg": maximum,
                    "blocking_joint": blocking,
                    "joint_limit_valid": limits_ok,
                    "joint_gate_pass": gate_pass,
                    "historical_edge_status": "historically_accepted_dual_backend" if historical else "not_historically_accepted",
                    "native_check_required": bool(gate_pass and not historical),
                    "native_check_cache_key": value_hash({"from_waypoint": from_wp, "to_waypoint": to_wp, "from_q": [round(x, 12) for x in source["joint_vector_rad"]], "to_q": [round(x, 12) for x in target["joint_vector_rad"]], "interpolation_step_deg": INTERPOLATION_STEP_DEG}),
                    "interpolation_step_deg": INTERPOLATION_STEP_DEG,
                    "collision_method": "adaptive_discrete_interpolation",
                    "closure_transition": from_wp == 719,
                    "status": "historically_accepted_dual_backend" if historical else ("joint_gate_rejected" if not gate_pass else "pending_native"),
                    "valid": bool(historical),
                })
    return inventory


def _accepted_edge_set(historical_edges: dict[tuple[int, int, str, str], dict[str, Any]], native_evidence: list[dict[str, Any]] | None = None) -> dict[tuple[int, int, str, str], dict[str, Any]]:
    result = dict(historical_edges)
    for row in native_evidence or []:
        fcl = row.get("fcl") or {}
        bullet = row.get("bullet") or {}
        if fcl.get("valid") is True and bullet.get("valid") is True and fcl.get("status") == "accepted" and bullet.get("status") == "accepted":
            request = row["request"]
            edge = edge_from_q(int(request["from_waypoint"]), int(request["to_waypoint"]), str(request["from_candidate_id"]), str(request["to_candidate_id"]), parse_q(request["q0"]), parse_q(request["q1"]), "Stage_2_4G_native", ["Stage_2_4G_native"])
            edge["samples_checked_fcl"] = fcl.get("samples_checked")
            edge["samples_checked_bullet"] = bullet.get("samples_checked")
            edge["fcl_collision_pairs"] = fcl.get("collision_pairs", [])
            edge["bullet_collision_pairs"] = bullet.get("collision_pairs", [])
            result[edge_key(edge)] = edge
    return result


def forward_reachable_sets(nodes: dict[str, dict[str, Any]], edges: dict[tuple[int, int, str, str], dict[str, Any]]) -> dict[str, Any]:
    by_source: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for edge in edges.values():
        by_source[(int(edge["from_waypoint"]), str(edge["from_candidate_id"]))].append(edge)
    for rows in by_source.values():
        rows.sort(key=lambda row: (float(row["max_joint_delta_deg"]), str(row["to_candidate_id"])))
    current = {row["candidate_id"] for row in nodes.values() if int(row["waypoint_index"]) == 0 and row["dual_backend_valid"]}
    sets: list[dict[str, Any]] = []
    first_break = None
    for wp in range(WAYPOINT_COUNT):
        sets.append({"waypoint": wp, "valid_node_count": sum(1 for row in nodes.values() if int(row["waypoint_index"]) == wp and row["dual_backend_valid"]), "forward_reachable_count": len(current), "forward_candidate_ids": sorted(current), "forward_component_ids": sorted({row.get("component_id") for cid, row in nodes.items() if cid in current and row.get("component_id") is not None})})
        if wp == WAYPOINT_COUNT - 1:
            break
        nxt: set[str] = set()
        for cid in sorted(current):
            nxt.update(str(edge["to_candidate_id"]) for edge in by_source.get((wp, cid), []) if int(edge["to_waypoint"]) == wp + 1)
        if current and not nxt and first_break is None:
            first_break = f"{wp}->{wp + 1}"
        current = nxt
    return {"direction": "i_to_i_plus_1", "start": "all_valid_waypoint_0_nodes", "sets": sets, "first_forward_break": first_break}


def reverse_reachable_sets(nodes: dict[str, dict[str, Any]], edges: dict[tuple[int, int, str, str], dict[str, Any]]) -> dict[str, Any]:
    by_target: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for edge in edges.values():
        by_target[(int(edge["to_waypoint"]), str(edge["to_candidate_id"]))].append(edge)
    for rows in by_target.values():
        rows.sort(key=lambda row: (float(row["max_joint_delta_deg"]), str(row["from_candidate_id"])))
    current = {row["candidate_id"] for row in nodes.values() if int(row["waypoint_index"]) == 719 and row["dual_backend_valid"]}
    sets: list[dict[str, Any]] = []
    last_break = None
    for wp in range(WAYPOINT_COUNT - 1, -1, -1):
        sets.append({"waypoint": wp, "valid_node_count": sum(1 for row in nodes.values() if int(row["waypoint_index"]) == wp and row["dual_backend_valid"]), "reverse_reachable_count": len(current), "reverse_candidate_ids": sorted(current), "reverse_component_ids": sorted({row.get("component_id") for cid, row in nodes.items() if cid in current and row.get("component_id") is not None})})
        if wp == 0:
            break
        nxt: set[str] = set()
        for cid in sorted(current):
            nxt.update(str(edge["from_candidate_id"]) for edge in by_target.get((wp, cid), []) if int(edge["from_waypoint"]) == wp - 1)
        if current and not nxt and last_break is None:
            last_break = f"{wp - 1}->{wp}"
        current = nxt
    sets.sort(key=lambda row: row["waypoint"])
    return {"direction": "i_plus_1_to_i", "start": "all_valid_waypoint_719_nodes", "sets": sets, "last_reverse_break": last_break}


def weak_components(nodes: dict[str, dict[str, Any]], edges: dict[tuple[int, int, str, str], dict[str, Any]]) -> tuple[dict[str, int], list[dict[str, Any]]]:
    valid_ids = {cid for cid, row in nodes.items() if row["dual_backend_valid"]}
    adjacency: dict[str, set[str]] = defaultdict(set)
    for edge in edges.values():
        a, b = str(edge["from_candidate_id"]), str(edge["to_candidate_id"])
        if a in valid_ids and b in valid_ids:
            adjacency[a].add(b)
            adjacency[b].add(a)
    membership: dict[str, int] = {}
    components: list[dict[str, Any]] = []
    for start in sorted(valid_ids):
        if start in membership:
            continue
        queue = deque([start])
        membership[start] = len(components)
        members: list[str] = []
        while queue:
            current = queue.popleft()
            members.append(current)
            for nxt in sorted(adjacency[current]):
                if nxt not in membership:
                    membership[nxt] = membership[start]
                    queue.append(nxt)
        wps = sorted({int(nodes[cid]["waypoint_index"]) for cid in members})
        components.append({"component_id": membership[start], "node_count": len(members), "waypoint_count": len(wps), "min_waypoint": wps[0] if wps else None, "max_waypoint": wps[-1] if wps else None, "candidate_ids": sorted(members)})
    for cid, row in nodes.items():
        row["component_id"] = membership.get(cid)
    return membership, components


def find_true_component_boundaries(nodes: dict[str, dict[str, Any]], edges: dict[tuple[int, int, str, str], dict[str, Any]]) -> dict[str, Any]:
    forward = forward_reachable_sets(nodes, edges)
    reverse = reverse_reachable_sets(nodes, edges)
    f_by = {int(row["waypoint"]): set(row["forward_candidate_ids"]) for row in forward["sets"]}
    r_by = {int(row["waypoint"]): set(row["reverse_candidate_ids"]) for row in reverse["sets"]}
    boundaries: list[dict[str, Any]] = []
    for i in range(719):
        if f_by.get(i) and not f_by.get(i + 1) and r_by.get(i + 1):
            boundaries.append({"from_waypoint": i, "to_waypoint": i + 1, "forward_frontier_candidate_ids": sorted(f_by[i]), "reverse_frontier_candidate_ids": sorted(r_by[i + 1]), "minimum_frontier_joint_gap_deg": None, "blocking_joint": None})
    if not boundaries:
        # If the graph has separated forward/reverse intervals, report the
        # first nonempty forward and reverse frontier even when they are not
        # adjacent; this is diagnostic and never a pass claim.
        f_nonempty = [i for i in range(720) if f_by.get(i)]
        r_nonempty = [i for i in range(720) if r_by.get(i)]
        if f_nonempty and r_nonempty and max(f_nonempty) < min(r_nonempty):
            i = max(f_nonempty)
            j = min(r_nonempty)
            boundaries.append({"from_waypoint": i, "to_waypoint": i + 1, "forward_frontier_candidate_ids": sorted(f_by[i]), "reverse_frontier_candidate_ids": sorted(r_by[j]), "reverse_frontier_waypoint": j, "minimum_frontier_joint_gap_deg": None, "blocking_joint": None})
    return {"forward_first_break": forward.get("first_forward_break"), "reverse_last_break": reverse.get("last_reverse_break"), "boundaries": boundaries, "forward_reachable_sets": forward, "reverse_reachable_sets": reverse}


def path_cost(path_ids: tuple[str, ...], path_edges: tuple[dict[str, Any], ...], nodes: dict[str, dict[str, Any]]) -> tuple[Any, ...]:
    max_step = max((float(edge["max_joint_delta_deg"]) for edge in path_edges), default=0.0)
    total = sum(float(edge["max_joint_delta_deg"]) for edge in path_edges)
    perturb = sum(sum(abs(float(value)) for value in (nodes[cid].get("position_offset_local_m") or [])) for cid in path_ids if nodes[cid].get("source_stage") == "Stage_2_4G")
    switches = sum(nodes[path_ids[i]]["canonical_branch_signature"] != nodes[path_ids[i - 1]]["canonical_branch_signature"] for i in range(1, len(path_ids)))
    return (round(max_step, 12), round(total, 12), round(perturb, 12), int(switches), path_ids)


def search_open_chain(nodes: dict[str, dict[str, Any]], edges: dict[tuple[int, int, str, str], dict[str, Any]]) -> dict[str, Any]:
    by_source: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for edge in edges.values():
        if int(edge["from_waypoint"]) != 719:
            by_source[(int(edge["from_waypoint"]), str(edge["from_candidate_id"]))].append(edge)
    for rows in by_source.values():
        rows.sort(key=lambda row: (float(row["max_joint_delta_deg"]), str(row["to_candidate_id"])))
    layers: dict[int, set[str]] = defaultdict(set)
    for cid, row in nodes.items():
        if row["dual_backend_valid"] and row["joint_limit_valid"] and row["process_tolerance_pass"]:
            layers[int(row["waypoint_index"])].add(cid)
    labels: dict[str, tuple[tuple[str, ...], tuple[dict[str, Any], ...], tuple[Any, ...]]] = {cid: ((cid,), tuple(), (0.0, 0.0, 0.0, 0, (cid,))) for cid in sorted(layers[0])}
    counts = {0: len(labels)}
    first_break = None
    for wp in range(719):
        next_labels: dict[str, tuple[tuple[str, ...], tuple[dict[str, Any], ...], tuple[Any, ...]]] = {}
        for cid, (path_ids, path_edges, _) in sorted(labels.items()):
            for edge in by_source.get((wp, cid), []):
                target = str(edge["to_candidate_id"])
                if target not in layers[wp + 1]:
                    continue
                candidate_path = path_ids + (target,)
                candidate_edges = path_edges + (edge,)
                cost = path_cost(candidate_path, candidate_edges, nodes)
                old = next_labels.get(target)
                if old is None or cost < old[2]:
                    next_labels[target] = (candidate_path, candidate_edges, cost)
        counts[wp + 1] = len(next_labels)
        if not next_labels and first_break is None:
            first_break = f"{wp}->{wp + 1}"
        labels = next_labels
        if not labels:
            break
    if not labels:
        return {"complete_0_to_719_open_chain_exists": False, "selected_node_ids": [], "selected_edges": [], "objective_cost": None, "reachable_count_by_waypoint": counts, "first_forward_break": first_break, "frontier_candidate_ids": []}
    best = min(labels.values(), key=lambda value: value[2])
    return {"complete_0_to_719_open_chain_exists": len(best[0]) == WAYPOINT_COUNT, "selected_node_ids": list(best[0]), "selected_edges": list(best[1]), "objective_cost": [*best[2][:-1], list(best[2][-1])], "reachable_count_by_waypoint": counts, "first_forward_break": first_break, "frontier_candidate_ids": sorted(labels)}


def closure_diagnostic(nodes: dict[str, dict[str, Any]], inventory: list[dict[str, Any]], edges: dict[tuple[int, int, str, str], dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in inventory if int(row["from_waypoint"]) == 719]
    passing = [row for row in rows if row["joint_gate_pass"]]
    native = [row for row in rows if row["native_check_required"]]
    dual = [edge for edge in edges.values() if int(edge["from_waypoint"]) == 719 and int(edge["to_waypoint"]) == 0]
    best = min(rows, key=lambda row: (float(row["max_joint_delta_deg"]), str(row["from_candidate_id"]), str(row["to_candidate_id"]))) if rows else None
    return {"closure_pair_count": len(rows), "closure_pairs_passing_20deg": len(passing), "closure_pairs_native_checked": len(native), "dual_backend_valid_closure_edges": len(dual), "best_closure_candidate_ids": [best["from_candidate_id"], best["to_candidate_id"]] if best else None, "best_closure_per_joint_delta_deg": best["per_joint_delta_deg"] if best else None, "best_closure_max_joint_step_deg": best["max_joint_delta_deg"] if best else None, "best_closure_blocking_joint": best["blocking_joint"] if best else None, "complete_720_node_720_edge_cycle_found": False, "closure_gate_status": "diagnostic_only_not_a_Stage_2_4G_pass_condition"}


def write_edge_request_csv(rows: list[dict[str, Any]], path: Path) -> None:
    assert_output_path_writable(path, operation="truncate_edge_request_csv")
    fields = ["from_waypoint", "to_waypoint", "from_candidate_id", "to_candidate_id", "max_joint_delta_deg", "interpolation_step_deg"] + [f"q0_{i}" for i in range(6)] + [f"q1_{i}" for i in range(6)]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            output = {"from_waypoint": row["from_waypoint"], "to_waypoint": row["to_waypoint"], "from_candidate_id": row["from_candidate_id"], "to_candidate_id": row["to_candidate_id"], "max_joint_delta_deg": row["max_joint_delta_deg"], "interpolation_step_deg": INTERPOLATION_STEP_DEG}
            output.update({f"q0_{i}": repr(value) for i, value in enumerate(row["q0"])})
            output.update({f"q1_{i}": repr(value) for i, value in enumerate(row["q1"])})
            writer.writerow(output)


def native_edge_batch(run_dir: Path, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not rows:
        return [], {"native_checked": 0, "records": [], "build": None, "fcl_bullet_edge_difference": 0}
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "stage24g_native_edge_requests.csv"
    write_edge_request_csv(rows, csv_path)
    stage24e.OUT = run_dir
    stage24e.PARTS = PARTS
    stage24e.STAGE23B = STAGE23B
    stage24.OUT = run_dir
    build = stage24e.stage24.build_native()
    write_json(run_dir / "stage24g_edge_build_report.json", build)
    if build.get("exit_code") != 0:
        raise RuntimeError(f"native_build_failed:{build.get('exit_code')}")
    records: list[dict[str, Any]] = []
    manifests: dict[str, list[dict[str, Any]]] = {}
    for backend in BACKENDS:
        record = stage24e.run_native_edges(csv_path, backend, NATIVE_RUN_INDEX, build)
        records.append(record)
        result_path = Path(record["result_path"])
        native = stage24.merge_native_edges(result_path, backend)
        manifests[backend] = stage24e.edge_manifest(rows, native, backend)
        write_jsonl(run_dir / f"stage24g_edges_{backend}.jsonl", manifests[backend])
    fcl = {edge_key(row) for row in manifests["fcl"] if row.get("status") == "accepted" and row.get("valid") is True}
    bullet = {edge_key(row) for row in manifests["bullet"] if row.get("status") == "accepted" and row.get("valid") is True}
    evidence: list[dict[str, Any]] = []
    by_backend = {backend: {edge_key(row): row for row in manifests[backend]} for backend in BACKENDS}
    for request in rows:
        key = edge_key(request)
        evidence.append({"native_check_cache_key": request["native_check_cache_key"], "request": request, "fcl": by_backend["fcl"].get(key), "bullet": by_backend["bullet"].get(key), "dual_backend_valid": key in fcl and key in bullet, "fcl_bullet_edge_agree": (key in fcl) == (key in bullet)})
    write_jsonl(run_dir / "stage24g_native_edge_evidence.jsonl", evidence)
    return evidence, {"native_checked": len(rows), "records": records, "build": build, "fcl_valid": len(fcl), "bullet_valid": len(bullet), "dual_backend_valid": len(fcl & bullet), "fcl_bullet_edge_difference": len(fcl ^ bullet), "fcl_request_q_sequence_hash": semantic_hash([(r["from_waypoint"], r["to_waypoint"], r["from_candidate_id"], r["to_candidate_id"], r["q0"], r["q1"]) for r in rows]), "bullet_request_q_sequence_hash": semantic_hash([(r["from_waypoint"], r["to_waypoint"], r["from_candidate_id"], r["to_candidate_id"], r["q0"], r["q1"]) for r in rows])}


def edge_materialization_summary(inventory: list[dict[str, Any]], evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {edge_key(row["request"]): row for row in evidence}
    rows: list[dict[str, Any]] = []
    for transition in range(WAYPOINT_COUNT):
        values = [row for row in inventory if int(row["from_waypoint"]) == transition]
        new = [row for row in values if row["native_check_required"]]
        results = [by_key.get(edge_key(row)) for row in new]
        fcl_valid = sum(1 for result in results if result and result.get("fcl", {}).get("valid") is True)
        bullet_valid = sum(1 for result in results if result and result.get("bullet", {}).get("valid") is True)
        dual = sum(1 for result in results if result and result.get("dual_backend_valid") is True)
        rows.append({"transition": f"{transition}->{(transition + 1) % WAYPOINT_COUNT}", "from_waypoint": transition, "to_waypoint": (transition + 1) % WAYPOINT_COUNT, "total_pairs": len(values), "joint_gate_pass_pairs": sum(1 for row in values if row["joint_gate_pass"]), "historically_accepted_pairs": sum(1 for row in values if row["historical_edge_status"] == "historically_accepted_dual_backend"), "newly_native_checked_pairs": len(new), "fcl_valid_pairs": fcl_valid, "bullet_valid_pairs": bullet_valid, "dual_backend_valid_pairs": dual, "unchecked_pairs_remaining": sum(1 for row in values if row["native_check_required"] and edge_key(row) not in by_key), "forward_component_0_bridge_created": False})
    return rows


def frozen_tolerance_contract() -> dict[str, Any]:
    return read_json(STAGE24E / "stage24e_frozen_tolerance_contract.yaml.json")


def parameter_schedule(contract: dict[str, Any]) -> list[dict[str, Any]]:
    t = contract["tolerances"]
    bounds = {"tangent": float(t["tcp_position_offset_local_m"]["tangent"][1]), "lateral": float(t["tcp_position_offset_local_m"]["lateral"][1]), "normal": float(t["tcp_position_offset_local_m"]["normal"][1]), "standoff": float(t["standoff_offset_m"][1]), "rx": float(t["normal_rotation_vector_deg"]["rx"][1]), "ry": float(t["normal_rotation_vector_deg"]["ry"][1]), "roll": float(t["tool_roll_offset_deg"][1])}
    dims = tuple(bounds)
    schedule: list[dict[str, Any]] = []
    schedule.append({"level": 0, "parameter_index": 0, "parameters": {name: 0.0 for name in dims}, "description": "nominal point"})
    index = 1
    for dim in dims:
        for sign in (-1.0, 1.0):
            values = {name: 0.0 for name in dims}
            values[dim] = sign * bounds[dim]
            schedule.append({"level": 1, "parameter_index": index, "parameters": values, "description": f"axis:{dim}:{sign:+.0f}"})
            index += 1
    for fraction in (0.05, 0.50):
        for dim in dims:
            for sign in (-1.0, 1.0):
                values = {name: 0.0 for name in dims}
                values[dim] = sign * fraction * bounds[dim]
                schedule.append({"level": 2, "parameter_index": index, "parameters": values, "description": f"boundary_fraction:{fraction}:{dim}:{sign:+.0f}"})
                index += 1
    vector_rows = stage24e.parameter_vectors(contract)
    for item in vector_rows:
        schedule.append({"level": 3, "parameter_index": index, "parameters": dict(item), "description": "deterministic_level3_vector"})
        index += 1
    for item in vector_rows[:8]:
        schedule.append({"level": 4, "parameter_index": index, "parameters": {name: float(value) * 0.25 for name, value in item.items()}, "description": "deterministic_low_difference_vector"})
        index += 1
    for item in vector_rows[:8]:
        schedule.append({"level": 5, "parameter_index": index, "parameters": {name: float(value) * 0.75 for name, value in item.items()}, "description": "boundary_local_refinement_vector"})
        index += 1
    dedup: dict[str, dict[str, Any]] = {}
    for row in schedule:
        dedup.setdefault(canonical(row["parameters"]), row)
    return list(sorted(dedup.values(), key=lambda row: (int(row["level"]), int(row["parameter_index"]))))


def pose_target(waypoints: dict[int, dict[str, str]], wp: int, values: dict[str, float]) -> np.ndarray:
    nominal_position, nominal_rotation, tangent, lateral = stage24e.pose_frame(waypoints, wp)
    normal = nominal_rotation[:, 2]
    position = nominal_position + tangent * values["tangent"] + lateral * values["lateral"] + normal * (values["normal"] + values["standoff"])
    tilt = Rotation.from_rotvec(tangent * math.radians(values["rx"]) + lateral * math.radians(values["ry"])).as_matrix()
    target = np.eye(4, dtype=float)
    target[:3, :3] = tilt @ nominal_rotation @ Rotation.from_euler("z", values["roll"], degrees=True).as_matrix()
    target[:3, 3] = position
    return target


def rank_all_boundary_seeds(nodes: dict[str, dict[str, Any]], boundary: dict[str, Any], membership: dict[str, int]) -> list[dict[str, Any]]:
    from_wp = int(boundary["from_waypoint"])
    to_wp = int(boundary["to_waypoint"])
    relevant = set(boundary.get("forward_frontier_candidate_ids", [])) | set(boundary.get("reverse_frontier_candidate_ids", []))
    adjacent = {from_wp - 1, from_wp, to_wp, to_wp + 1}
    rows: list[dict[str, Any]] = []
    for cid, row in nodes.items():
        if int(row["waypoint_index"]) in adjacent or cid in relevant:
            direction_rank = 0 if cid in boundary.get("forward_frontier_candidate_ids", []) else (1 if cid in boundary.get("reverse_frontier_candidate_ids", []) else 2)
            frontier_delta = min((max(abs(a - b) for a, b in zip(row["joint_vector_rad"], nodes[other]["joint_vector_rad"])) for other in relevant if other in nodes and other != cid), default=999.0)
            rows.append({"seed_candidate_id": cid, "seed_rank_key": [direction_rank, round(frontier_delta, 12), row["canonical_branch_signature"], str(row["source_stage"]), cid], "seed_waypoint": row["waypoint_index"], "seed_direction": "forward" if cid in boundary.get("forward_frontier_candidate_ids", []) else ("reverse" if cid in boundary.get("reverse_frontier_candidate_ids", []) else "adjacent"), "seed_component_id": membership.get(cid), "seed_canonical_branch_signature": row["canonical_branch_signature"], "seed_provenance": row.get("provenance", [])})
    rows.sort(key=lambda row: tuple(row["seed_rank_key"]))
    dedup: dict[str, dict[str, Any]] = {}
    for row in rows:
        dedup.setdefault(row["seed_candidate_id"], row)
    output = list(dedup.values())
    for index, row in enumerate(output, start=1):
        row["seed_rank"] = index
    return output


def boundary_waypoints(boundaries: list[dict[str, Any]]) -> list[int]:
    selected: set[int] = set()
    for boundary in boundaries:
        i = int(boundary["from_waypoint"])
        j = int(boundary.get("to_waypoint", i + 1))
        # Stage 2.4G may introduce IK only at the endpoints of a true
        # component boundary.  Neighbouring waypoints are existing frozen
        # data and must not become an adaptive search surface.
        for wp in (i, j):
            if 0 <= wp < WAYPOINT_COUNT:
                selected.add(wp)
    return sorted(selected)


def candidate_boundary_specs(boundary: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the forward and reverse true-boundary bridge specs.

    ``find_true_component_boundaries`` reports the forward first break as an
    adjacent boundary when possible and separately reports the reverse last
    break.  Both directions are required by the Stage 2.4G objective, so the
    reverse break is promoted to its own seed/IK boundary when it is distinct.
    """
    specs = [dict(item) for item in boundary.get("boundaries", [])]
    existing = {(int(item["from_waypoint"]), int(item["to_waypoint"])) for item in specs}
    reverse_label = boundary.get("reverse_last_break")
    if reverse_label:
        from_wp, to_wp = (int(value) for value in str(reverse_label).split("->", 1))
        if (from_wp, to_wp) not in existing:
            forward_by = {int(item["waypoint"]): set(item["forward_candidate_ids"]) for item in boundary["forward_reachable_sets"]["sets"]}
            reverse_by = {int(item["waypoint"]): set(item["reverse_candidate_ids"]) for item in boundary["reverse_reachable_sets"]["sets"]}
            specs.append({
                "from_waypoint": from_wp,
                "to_waypoint": to_wp,
                "forward_frontier_candidate_ids": sorted(forward_by.get(from_wp, set())),
                "reverse_frontier_candidate_ids": sorted(reverse_by.get(to_wp, set())),
                "minimum_frontier_joint_gap_deg": None,
                "blocking_joint": None,
                "boundary_role": "reverse_last_break",
            })
    return sorted(specs, key=lambda item: (int(item["from_waypoint"]), int(item["to_waypoint"])))


def generate_bidirectional_candidates(nodes: dict[str, dict[str, Any]], boundaries: list[dict[str, Any]], membership: dict[str, int], semantics: list[dict[str, Any]], contract: dict[str, Any], run_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not boundaries:
        return [], [], []
    waypoints = {int(row["waypoint_id"]): row for row in stage24e.read_csv(SOURCE / "waypoints.csv")}
    solver = stage24e.solver_from_contract(contract)
    schedule = parameter_schedule(contract)
    generated: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    existing_q = {(int(row["waypoint_index"]), tuple(round(float(value), 10) for value in row["joint_vector_rad"])) for row in nodes.values()}
    for boundary_index, boundary in enumerate(boundaries):
        seeds = rank_all_boundary_seeds(nodes, boundary, membership)
        for seed in seeds:
            seed_rows.append({**seed, "boundary_index": boundary_index, "from_waypoint": boundary["from_waypoint"], "to_waypoint": boundary["to_waypoint"]})
            seed_node = nodes[seed["seed_candidate_id"]]
            seed_q = np.asarray(seed_node["joint_vector_rad"], dtype=float)
            for wp in boundary_waypoints([boundary]):
                candidate_added_for_seed_wp = False
                for parameter in schedule:
                    values = parameter["parameters"]
                    attempt = {"boundary_index": boundary_index, "waypoint_index": wp, "direction": seed["seed_direction"], "seed_id": seed["seed_candidate_id"], "seed_rank": seed["seed_rank"], "parameter_level": parameter["level"], "parameter_index": parameter["parameter_index"], "parameter_description": parameter["description"], "parameters": values, "status": "pending"}
                    try:
                        result = solver.solve(pose_target(waypoints, wp, values), seed_q)
                    except Exception as exc:
                        attempt.update({"status": "solver_exception", "error": repr(exc)})
                        attempts.append(attempt)
                        continue
                    if not result.success:
                        attempt.update({"status": "ik_failed", "message": str(getattr(result, "message", ""))})
                        attempts.append(attempt)
                        continue
                    q = np.asarray(result.q_rad, dtype=float)
                    fk = solver.fk_tcp(q)
                    target = pose_target(waypoints, wp, values)
                    pos_error = float(np.linalg.norm(fk[:3, 3] - target[:3, 3]))
                    actual_z = fk[:3, 2] / max(float(np.linalg.norm(fk[:3, 2])), 1e-15)
                    target_z = target[:3, 2] / max(float(np.linalg.norm(target[:3, 2])), 1e-15)
                    z_error = float(math.degrees(math.acos(float(np.clip(np.dot(actual_z, target_z), -1.0, 1.0)))))
                    limits_ok = bool(np.all(q >= solver.robot.limits.q_min - 1e-9) and np.all(q <= solver.robot.limits.q_max + 1e-9))
                    q_key = (wp, tuple(round(float(value), 10) for value in q))
                    signature = canonical_branch_signature(q.tolist(), semantics)
                    attempt.update({"joint_values_rad": q.tolist(), "FK_position_error_m": pos_error, "FK_tool_axis_error_deg": z_error, "joint_limit_valid": limits_ok, "canonical_branch_signature": signature, "branch_label_changed_from_seed": signature != seed["seed_canonical_branch_signature"], "branch_signature_valid": True, "status": "formal_candidate" if limits_ok and pos_error <= 0.006 + 1e-9 and z_error <= 10.0 + 1e-9 else "formal_rejected"})
                    attempts.append(attempt)
                    if attempt["status"] != "formal_candidate" or q_key in existing_q or candidate_added_for_seed_wp:
                        continue
                    existing_q.add(q_key)
                    candidate_added_for_seed_wp = True
                    stable = value_hash({"wp": wp, "q": [round(float(value), 12) for value in q], "parameters": values, "seed": seed["seed_candidate_id"], "boundary": boundary_index})[:16]
                    generated.append({"candidate_id": f"s24g-{wp:04d}-{stable}", "waypoint_index": wp, "waypoint_id": wp, "joint_values_rad": q.tolist(), "joint_values": q.tolist(), "joint_values_deg": [math.degrees(float(value)) for value in q], "ik_branch_signature": signature, "canonical_branch_signature": signature, "seed_id": seed["seed_candidate_id"], "seed_rank": seed["seed_rank"], "direction": seed["seed_direction"], "source_stage": "Stage_2_4G", "position_offset_local_m": [values["tangent"], values["lateral"], values["normal"]], "standoff_offset_m": values["standoff"], "normal_rotation_vector_deg": [values["rx"], values["ry"]], "roll_offset_deg": values["roll"], "FK_position_error_m": pos_error, "FK_orientation_error_deg": z_error, "joint_limit_pass": limits_ok, "process_tolerance_pass": True, "solver_success": True, "candidate_parameter_hash": value_hash(values), "branch_label_change_allowed": True, "formal_node_gate_pending": True})
    generated.sort(key=lambda row: (int(row["waypoint_index"]), str(row["candidate_id"])))
    write_jsonl(run_dir / "stage24g_seed_ranking.jsonl", seed_rows)
    write_jsonl(run_dir / "stage24g_candidate_generation.jsonl", attempts)
    write_jsonl(run_dir / "stage24g_candidate_generation_candidates.jsonl", generated)
    return generated, attempts, seed_rows


def native_node_batch(run_dir: Path, candidates: list[dict[str, Any]]) -> tuple[set[str], dict[str, Any]]:
    if not candidates:
        return set(), {"candidate_count": 0, "fcl_valid": 0, "bullet_valid": 0, "dual_backend_valid": 0, "fcl_bullet_node_difference": 0, "records": []}
    candidate_csv = run_dir / "stage24g_native_node_requests.csv"
    assert_output_path_writable(candidate_csv, operation="truncate_native_node_requests")
    with candidate_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["waypoint_id", "candidate_id", "joint_values"], lineterminator="\n")
        writer.writeheader()
        for row in candidates:
            writer.writerow({"waypoint_id": row["waypoint_id"], "candidate_id": row["candidate_id"], "joint_values": canonical(row["joint_values"])})
    stage24e.OUT = run_dir
    stage24e.PARTS = PARTS
    stage24e.STAGE23B = STAGE23B
    records = [stage24e.run_native_nodes(candidate_csv, backend, NATIVE_RUN_INDEX) for backend in BACKENDS]
    fcl_rows = read_jsonl(Path(records[0]["result_path"])) if Path(records[0]["result_path"]).exists() else []
    bullet_rows = read_jsonl(Path(records[1]["result_path"])) if Path(records[1]["result_path"]).exists() else []
    fcl = {str(row["candidate_id"]) for row in fcl_rows if row.get("valid") is True}
    bullet = {str(row["candidate_id"]) for row in bullet_rows if row.get("valid") is True}
    audit = {"candidate_count": len(candidates), "fcl_valid": len(fcl), "bullet_valid": len(bullet), "dual_backend_valid": len(fcl & bullet), "fcl_bullet_node_difference": len(fcl ^ bullet), "fcl_valid_ids": sorted(fcl), "bullet_valid_ids": sorted(bullet), "dual_backend_valid_ids": sorted(fcl & bullet), "records": records, "fcl_result_sha256": records[0].get("result_sha256"), "bullet_result_sha256": records[1].get("result_sha256")}
    write_json(run_dir / "stage24g_native_node_evidence.json", audit)
    return fcl & bullet, audit


def add_new_nodes(nodes: dict[str, dict[str, Any]], candidates: list[dict[str, Any]], valid_ids: set[str], semantics: list[dict[str, Any]]) -> None:
    for row in candidates:
        if row["candidate_id"] in valid_ids:
            nodes[row["candidate_id"]] = normalized_node(row, "Stage_2_4G", semantics)


def all_ranked_seeds_attempted(seed_rows: list[dict[str, Any]], attempts: list[dict[str, Any]]) -> bool:
    """Check seed coverage without treating a seed reused at two boundaries as a failure."""
    ranked = {str(row["seed_candidate_id"]) for row in seed_rows}
    attempted = {str(row["seed_id"]) for row in attempts}
    return bool(ranked) and ranked.issubset(attempted)


def update_inventory_status(inventory: list[dict[str, Any]], evidence: list[dict[str, Any]]) -> None:
    by_key = {edge_key(row["request"]): row for row in evidence}
    for row in inventory:
        result = by_key.get(edge_key(row))
        if result is None:
            continue
        fcl = result.get("fcl") or {}
        bullet = result.get("bullet") or {}
        row["fcl_edge_valid"] = fcl.get("valid") is True and fcl.get("status") == "accepted"
        row["bullet_edge_valid"] = bullet.get("valid") is True and bullet.get("status") == "accepted"
        row["dual_backend_valid"] = result.get("dual_backend_valid") is True
        row["fcl_samples_checked"] = fcl.get("samples_checked")
        row["bullet_samples_checked"] = bullet.get("samples_checked")
        row["fcl_first_invalid_sample"] = fcl.get("first_collision_sample")
        row["bullet_first_invalid_sample"] = bullet.get("first_collision_sample")
        row["fcl_collision_pairs"] = fcl.get("collision_pairs", [])
        row["bullet_collision_pairs"] = bullet.get("collision_pairs", [])
        row["status"] = "accepted_dual_backend" if row["dual_backend_valid"] else "native_rejected_or_disagreed"
        row["valid"] = row["dual_backend_valid"]


def build_graph_edges(historical_edges: dict[tuple[int, int, str, str], dict[str, Any]], inventory: list[dict[str, Any]]) -> dict[tuple[int, int, str, str], dict[str, Any]]:
    edges = dict(historical_edges)
    for row in inventory:
        if row.get("dual_backend_valid") is True:
            edge = edge_from_q(int(row["from_waypoint"]), int(row["to_waypoint"]), str(row["from_candidate_id"]), str(row["to_candidate_id"]), parse_q(row["q0"]), parse_q(row["q1"]), "Stage_2_4G_native", ["Stage_2_4G_native"])
            edge["samples_checked_fcl"] = row.get("fcl_samples_checked")
            edge["samples_checked_bullet"] = row.get("bullet_samples_checked")
            edge["fcl_collision_pairs"] = row.get("fcl_collision_pairs", [])
            edge["bullet_collision_pairs"] = row.get("bullet_collision_pairs", [])
            edges[edge_key(edge)] = edge
    return edges


def all_layered_nodes(nodes: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [nodes[cid] for cid in sorted(nodes, key=lambda cid: (int(nodes[cid]["waypoint_index"]), cid))]


def write_reachability_outputs(run_dir: Path, nodes: dict[str, dict[str, Any]], edges: dict[tuple[int, int, str, str], dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, int], list[dict[str, Any]]]:
    membership, components = weak_components(nodes, edges)
    forward = forward_reachable_sets(nodes, edges)
    reverse = reverse_reachable_sets(nodes, edges)
    boundary = find_true_component_boundaries(nodes, edges)
    write_json(run_dir / "stage24g_forward_reachability.json", forward)
    write_json(run_dir / "stage24g_reverse_reachability.json", reverse)
    write_json(run_dir / "stage24g_component_summary.json", {"component_count": len(components), "components": components, "waypoint_0_components": sorted({membership.get(row["candidate_id"]) for row in nodes.values() if int(row["waypoint_index"]) == 0}), "waypoint_719_components": sorted({membership.get(row["candidate_id"]) for row in nodes.values() if int(row["waypoint_index"]) == 719})})
    write_json(run_dir / "stage24g_component_boundary_analysis.json", boundary)
    write_jsonl(run_dir / "stage24g_valid_nodes.jsonl", all_layered_nodes(nodes))
    write_jsonl(run_dir / "stage24g_layered_graph_edges.jsonl", [edges[key] for key in sorted(edges)])
    return forward, reverse, boundary, membership, components


def audit_fragile_edges(inventory: list[dict[str, Any]], historical_edges: dict[tuple[int, int, str, str], dict[str, Any]], selected_chain: dict[str, Any]) -> dict[str, Any]:
    historical_preserved: list[dict[str, Any]] = []
    still_fragile: list[str] = []
    strengthened: list[str] = []
    selected_edges = {(int(row["from_waypoint"]), int(row["to_waypoint"])) for row in selected_chain.get("selected_edges", [])}
    for transition in FRAGILE_TRANSITIONS:
        from_wp, to_wp = map(int, transition.split("->"))
        old = [key for key in historical_edges if key[:2] == (from_wp, to_wp)]
        new = [row for row in inventory if int(row["from_waypoint"]) == from_wp and int(row["to_waypoint"]) == to_wp and row.get("dual_backend_valid") is True]
        row = {"transition": transition, "historical_accepted_edge_count": len(old), "materialized_dual_backend_edge_count": len(new) + len(old), "historical_fragile_edge_preserved": bool(old), "selected_in_open_chain": (from_wp, to_wp) in selected_edges}
        historical_preserved.append(row)
        if len(new) + len(old) <= 1:
            still_fragile.append(transition)
        elif len(new) + len(old) > len(old):
            strengthened.append(transition)
    return {"historical_fragile_edges_preserved": historical_preserved, "transitions_still_fragile": still_fragile, "transitions_strengthened_by_new_edges": strengthened, "historical_fragile_transition_count": len(FRAGILE_TRANSITIONS)}


def lazy_relevant_requests(inventory: list[dict[str, Any]], nodes: dict[str, dict[str, Any]], edges: dict[tuple[int, int, str, str], dict[str, Any]], checked_keys: set[tuple[int, int, str, str]], round_index: int) -> list[dict[str, Any]]:
    """Select only gate-passing pairs that can extend forward/reverse reachability.

    The complete inventory remains persisted.  A pair is marked relevant when
    it lies on a current forward break, reverse break, or an immediately
    adjacent forward/reverse frontier.  The historically disputed 34->35
    transition is forced into round one so all 607 missing gate-passing pairs
    are materialized, regardless of the currently selected path.
    """
    boundary = find_true_component_boundaries(nodes, edges)
    labels: set[str] = set()
    for label in (boundary.get("forward_first_break"), boundary.get("reverse_last_break")):
        if label:
            labels.add(str(label))
    for row in boundary.get("boundaries", []):
        labels.add(f"{int(row['from_waypoint'])}->{int(row['to_waypoint'])}")
    if round_index == 1:
        labels.add("34->35")
    forward_sets = {int(item["waypoint"]): set(item["forward_candidate_ids"]) for item in boundary["forward_reachable_sets"]["sets"]}
    reverse_sets = {int(item["waypoint"]): set(item["reverse_candidate_ids"]) for item in boundary["reverse_reachable_sets"]["sets"]}
    forward_break = boundary.get("forward_first_break")
    reverse_break = boundary.get("reverse_last_break")
    selected: list[dict[str, Any]] = []
    for row in inventory:
        key = edge_key(row)
        if not row["native_check_required"] or key in checked_keys or row["transition"] not in labels:
            continue
        full_explicit_seam_audit = round_index == 1 and row["transition"] == "34->35"
        if not full_explicit_seam_audit:
            source_frontier = forward_sets.get(int(row["from_waypoint"]), set())
            target_frontier = reverse_sets.get(int(row["to_waypoint"]), set())
            if row["transition"] == forward_break and row["from_candidate_id"] not in source_frontier:
                continue
            if row["transition"] == reverse_break and row["to_candidate_id"] not in target_frontier:
                continue
        item = dict(row)
        item["relevant_native_check"] = True
        item["relevance_reason"] = "forward_reverse_frontier" if row["transition"] != "34->35" else "explicit_34_to_35_complete_audit"
        selected.append(item)
    selected.sort(key=lambda row: (int(row["from_waypoint"]), int(row["to_waypoint"]), float(row["max_joint_delta_deg"]), str(row["from_candidate_id"]), str(row["to_candidate_id"])))
    return selected


def run_one(run_dir: Path, run_index: int, input_manifest: dict[str, Any], semantics: list[dict[str, Any]], contract: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "stage24g_input_manifest.json", {**input_manifest, "run_index": run_index})
    nodes, aliases, node_audit = load_and_deduplicate_valid_nodes(semantics)
    historical_edges = _load_historical_edges(nodes, aliases)
    inventory = enumerate_adjacent_pair_inventory(nodes, historical_edges, semantics)
    write_json(run_dir / "stage24g_node_deduplication_audit.json", {**node_audit, "historical_edge_count": len(historical_edges), "inventory_pair_count": len(inventory), "inventory_gate_passing_count": sum(1 for row in inventory if row["joint_gate_pass"]), "inventory_native_required_count": sum(1 for row in inventory if row["native_check_required"]), "transition_34_to_35": {"total_pairs": sum(1 for row in inventory if row["transition"] == "34->35"), "joint_gate_pass_pairs": sum(1 for row in inventory if row["transition"] == "34->35" and row["joint_gate_pass"]), "historically_accepted_pairs": sum(1 for row in inventory if row["transition"] == "34->35" and row["historical_edge_status"] == "historically_accepted_dual_backend"), "missing_native_pairs": sum(1 for row in inventory if row["transition"] == "34->35" and row["native_check_required"])}})
    checked_keys: set[tuple[int, int, str, str]] = set()
    native_evidence: list[dict[str, Any]] = []
    native_rounds: list[dict[str, Any]] = []
    native_meta: dict[str, Any] = {"native_checked": 0, "fcl_valid": 0, "bullet_valid": 0, "dual_backend_valid": 0, "fcl_bullet_edge_difference": 0, "rounds": []}
    # Lazy rounds are deterministic.  Each round is a separate native batch,
    # and reachability is recomputed before the next frontier is selected.
    for round_index in range(1, WAYPOINT_COUNT + 1):
        edges = build_graph_edges(historical_edges, inventory)
        forward, reverse, boundary, membership, components = write_reachability_outputs(run_dir, nodes, edges)
        open_chain = search_open_chain(nodes, edges)
        if open_chain.get("complete_0_to_719_open_chain_exists"):
            break
        requests = lazy_relevant_requests(inventory, nodes, edges, checked_keys, round_index)
        if not requests:
            break
        for row in requests:
            checked_keys.add(edge_key(row))
        batch_dir = run_dir / f"edge_materialization_round{round_index:03d}"
        evidence, meta = native_edge_batch(batch_dir, requests)
        update_inventory_status(inventory, evidence)
        native_evidence.extend(evidence)
        native_rounds.append({"round_index": round_index, "request_count": len(requests), "transition_counts": dict(sorted({label: sum(1 for row in requests if row["transition"] == label) for label in {row["transition"] for row in requests}}.items())), "meta": meta})
        native_meta["native_checked"] += int(meta.get("native_checked", 0))
        native_meta["fcl_valid"] += int(meta.get("fcl_valid", 0))
        native_meta["bullet_valid"] += int(meta.get("bullet_valid", 0))
        native_meta["dual_backend_valid"] += int(meta.get("dual_backend_valid", 0))
        native_meta["fcl_bullet_edge_difference"] += int(meta.get("fcl_bullet_edge_difference", 0))
        native_meta["rounds"] = native_rounds
    write_json(run_dir / "stage24g_edge_materialization_rounds.json", {"rounds": native_rounds, "lazy_policy": "forward_reverse_frontier_with_explicit_34_to_35_complete_audit", "relevant_native_request_count": len(checked_keys), "deferred_gate_passing_pair_count": sum(1 for row in inventory if row["native_check_required"] and edge_key(row) not in checked_keys)})
    write_jsonl(run_dir / "stage24g_pair_prefilter_inventory.jsonl", inventory)
    write_jsonl(run_dir / "stage24g_native_edge_requests.jsonl", [row for row in inventory if edge_key(row) in checked_keys])
    write_jsonl(run_dir / "stage24g_native_edge_evidence.jsonl", native_evidence)
    write_csv(run_dir / "stage24g_edge_materialization_summary.csv", edge_materialization_summary(inventory, native_evidence))
    edges = build_graph_edges(historical_edges, inventory)
    forward, reverse, boundary, membership, components = write_reachability_outputs(run_dir, nodes, edges)
    open_chain = search_open_chain(nodes, edges)
    write_json(run_dir / "stage24g_selected_open_chain.json", open_chain)
    if open_chain.get("selected_node_ids"):
        write_csv(run_dir / "stage24g_selected_open_chain.csv", [nodes[cid] for cid in open_chain["selected_node_ids"]])
    else:
        write_csv(run_dir / "stage24g_selected_open_chain.csv", [])
    candidate_attempts: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    generated: list[dict[str, Any]] = []
    candidate_levels: list[dict[str, Any]] = []
    candidate_edge_rounds: list[dict[str, Any]] = []
    node_native_meta: dict[str, Any] = {"candidate_count": 0, "dual_backend_valid": 0, "fcl_bullet_node_difference": 0, "records": []}
    if not open_chain.get("complete_0_to_719_open_chain_exists"):
        # The existing-node space is already fully materialized above.  The
        # boundary search is deliberately bounded by the frozen Stage 2.4E
        # tolerance contract and never inserts a waypoint.
        bridge_boundaries = candidate_boundary_specs(boundary)
        generated, candidate_attempts, seed_rows = generate_bidirectional_candidates(nodes, bridge_boundaries, membership, semantics, contract, run_dir)
        candidate_levels.append({"level": "all_frozen_schedule", "parameter_points": len(parameter_schedule(contract)), "waypoints": boundary_waypoints(bridge_boundaries), "boundaries": bridge_boundaries, "generated_candidates": len(generated), "status": "generated_pending_native_node_gate"})
        write_json(run_dir / "stage24g_adaptive_generation_summary.json", {"parameter_schedule": parameter_schedule(contract), "levels": candidate_levels, "candidate_generation_attempt_count": len(candidate_attempts), "all_ranked_seeds_actually_attempted": all_ranked_seeds_attempted(seed_rows, candidate_attempts), "random_api_calls": 0, "frozen_tolerance_bounds_only": True})
        valid_new_ids, node_native_meta = native_node_batch(run_dir, generated)
        add_new_nodes(nodes, generated, valid_new_ids, semantics)
        candidate_edge_checked: set[tuple[int, int, str, str]] = set()
        for candidate_round in range(1, WAYPOINT_COUNT + 1):
            new_inventory = enumerate_adjacent_pair_inventory(nodes, edges, semantics)
            candidate_relevant = lazy_relevant_requests(new_inventory, nodes, edges, candidate_edge_checked, candidate_round + 1)
            new_requests = [row for row in candidate_relevant if str(row["from_candidate_id"]) in valid_new_ids or str(row["to_candidate_id"]) in valid_new_ids]
            if not new_requests:
                break
            for row in new_requests:
                candidate_edge_checked.add(edge_key(row))
            new_evidence, new_native_meta = native_edge_batch(run_dir / f"candidate_edge_materialization_round{candidate_round:03d}", new_requests)
            update_inventory_status(new_inventory, new_evidence)
            existing_inventory_keys = {edge_key(old) for old in inventory}
            inventory.extend(row for row in new_inventory if edge_key(row) not in existing_inventory_keys)
            edges = build_graph_edges(edges, new_inventory)
            native_evidence.extend(new_evidence)
            candidate_edge_rounds.append({"round_index": candidate_round, "request_count": len(new_requests), "meta": new_native_meta})
            native_meta["fcl_bullet_edge_difference"] += int(new_native_meta.get("fcl_bullet_edge_difference", 0))
            forward, reverse, boundary, membership, components = write_reachability_outputs(run_dir, nodes, edges)
            open_chain = search_open_chain(nodes, edges)
            write_json(run_dir / "stage24g_selected_open_chain.json", open_chain)
            if open_chain.get("complete_0_to_719_open_chain_exists"):
                break
        native_meta["candidate_edge_materialization_rounds"] = candidate_edge_rounds
        write_json(run_dir / "stage24g_adaptive_generation_summary.json", {"parameter_schedule": parameter_schedule(contract), "levels": candidate_levels, "candidate_generation_attempt_count": len(candidate_attempts), "all_ranked_seeds_actually_attempted": all_ranked_seeds_attempted(seed_rows, candidate_attempts), "random_api_calls": 0, "frozen_tolerance_bounds_only": True, "new_dual_backend_valid_nodes": node_native_meta.get("dual_backend_valid", 0), "new_dual_backend_valid_edges": sum(1 for row in inventory if row.get("dual_backend_valid") is True and (str(row.get("from_candidate_id")) in valid_new_ids or str(row.get("to_candidate_id")) in valid_new_ids)), "candidate_edge_materialization_rounds": candidate_edge_rounds})
    write_jsonl(run_dir / "stage24g_native_node_evidence.jsonl", [{"candidate_id": cid, "fcl_valid": cid in set(node_native_meta.get("fcl_valid_ids", [])), "bullet_valid": cid in set(node_native_meta.get("bullet_valid_ids", [])), "dual_backend_valid": cid in set(node_native_meta.get("dual_backend_valid_ids", []))} for cid in sorted(set(row["candidate_id"] for row in generated))])
    closure = closure_diagnostic(nodes, inventory, edges)
    write_json(run_dir / "stage24g_closure_diagnostic.json", closure)
    fragile = audit_fragile_edges(inventory, historical_edges, open_chain)
    write_json(run_dir / "stage24g_fragile_transition_regression.json", fragile)
    every_waypoint = bool(open_chain.get("complete_0_to_719_open_chain_exists") and len(open_chain.get("selected_node_ids", [])) == 720 and [nodes[cid]["waypoint_index"] for cid in open_chain["selected_node_ids"]] == list(range(720)))
    status = "passed_complete_dual_backend_open_chain" if every_waypoint and native_meta.get("fcl_bullet_edge_difference", 0) == 0 else ("blocked_no_open_chain_after_complete_existing_edge_materialization" if not generated else "blocked_no_component_bridge_within_frozen_search_budget")
    if native_meta.get("fcl_bullet_edge_difference", 0) != 0:
        status = "blocked_fcl_bullet_edge_disagreement"
    if node_native_meta.get("fcl_bullet_node_difference", 0) != 0:
        status = "blocked_fcl_bullet_node_disagreement"
    candidate_edge_valid_count = sum(int(item.get("meta", {}).get("dual_backend_valid", 0)) for item in candidate_edge_rounds)
    gate = {"Stage_2_4G": status, "complete_0_to_719_open_chain_exists": every_waypoint, "open_chain_node_count": len(open_chain.get("selected_node_ids", [])), "open_chain_transition_count": len(open_chain.get("selected_edges", [])), "all_waypoints_present": every_waypoint, "every_node_fcl_valid": all(row["dual_backend_valid"] for row in nodes.values()), "every_node_bullet_valid": all(row["dual_backend_valid"] for row in nodes.values()), "every_transition_max_joint_step_deg_lte_20": all(float(edge["max_joint_delta_deg"]) <= MAX_STEP_DEG + 1e-12 for edge in open_chain.get("selected_edges", [])), "every_transition_fcl_valid": every_waypoint and all(edge.get("fcl_edge_valid") for edge in open_chain.get("selected_edges", [])), "every_transition_bullet_valid": every_waypoint and all(edge.get("bullet_edge_valid") for edge in open_chain.get("selected_edges", [])), "fcl_bullet_node_difference": node_native_meta.get("fcl_bullet_node_difference", 0), "fcl_bullet_edge_difference": native_meta.get("fcl_bullet_edge_difference", 0), "frozen_input_hashes_unchanged": bool(input_manifest.get("frozen_input_hashes_unchanged")), "deterministic_rebuild": "pending", "tests": "pending", "continuous_tolerance_space_infeasible_proven": False, "existing_valid_node_edge_space_exhausted": len([row for row in inventory if row["native_check_required"] and edge_key(row) not in checked_keys]) == 0, "all_relevant_gate_passing_pairs_native_checked": all(row.get("status") in {"accepted_dual_backend", "native_rejected_or_disagreed", "historically_accepted_dual_backend", "joint_gate_rejected"} for row in inventory if row["native_check_required"] and edge_key(row) in checked_keys), "first_true_global_break": boundary.get("forward_first_break"), "forward_frontier_candidate_ids": boundary.get("boundaries", [{}])[0].get("forward_frontier_candidate_ids", []) if boundary.get("boundaries") else [], "reverse_frontier_candidate_ids": boundary.get("boundaries", [{}])[0].get("reverse_frontier_candidate_ids", []) if boundary.get("boundaries") else [], "minimum_frontier_joint_gap_deg": None, "blocking_joint": None, "candidate_generation_attempt_count": len(candidate_attempts), "all_ranked_seeds_actually_attempted": all_ranked_seeds_attempted(seed_rows, candidate_attempts), "tolerance_parameter_points_attempted": len(parameter_schedule(contract)) if generated else 0, "new_dual_backend_valid_nodes": node_native_meta.get("dual_backend_valid", 0), "new_dual_backend_valid_edges": candidate_edge_valid_count, "next_predicted_boundary": boundary.get("boundaries", [None])[0], "graph_node_count": len(nodes), "graph_edge_count": len(edges), "graph_component_count": len(components), "historical_fragile_edges_preserved": len(fragile.get("historical_fragile_edges_preserved", [])), "transitions_still_fragile": fragile.get("transitions_still_fragile", []), "closure_diagnostic": closure, "closure_gate_passed": False, "ready_for_Stage_2_5": False, "Stage_2_5": "blocked", "Ruckig": "not_run", "TOTG": "not_run", "GNN": "not_run", "CCD": "not_available", "clearance": "not_available", "collision_method": "adaptive_discrete_interpolation", "native_edge_validation": native_meta, "native_node_validation": node_native_meta}
    write_json(run_dir / "stage24g_gate_report.json", gate)
    write_json(run_dir / "stage24g_closed_cycle_certificate.json", {"open_chain_complete": every_waypoint, "closure_edge_valid": False, "complete_cycle_found": False, "closure_diagnostic": closure})
    write_json(run_dir / "stage24g_determinism_report.json", {"run_index": run_index, "status": "pending_top_level_comparison"})
    write_json(run_dir / "stage24g_component_boundary_analysis.json", {**boundary, "ranked_seed_count": len(seed_rows), "boundary_waypoints": boundary_waypoints(boundary.get("boundaries", []))})
    report = ["# Stage 2.4G complete edge materialization and bidirectional component bridge", "", f"- Stage 2.4G: `{status}`", f"- Frozen valid physical nodes: `{len(nodes)}`; graph edges: `{len(edges)}`; components: `{len(components)}`.", f"- Existing-node native edge requests: `{sum(1 for row in inventory if row['native_check_required'])}`; dual-backend-valid new edges: `{sum(1 for row in inventory if row.get('dual_backend_valid') is True)}`.", f"- 34->35 inventory: `{sum(1 for row in inventory if row['transition'] == '34->35')}` total, `{sum(1 for row in inventory if row['transition'] == '34->35' and row['joint_gate_pass'])}` gate-passing, `{sum(1 for row in inventory if row['transition'] == '34->35' and row['native_check_required'])}` newly native checked.", f"- Open chain: `{every_waypoint}`; closure: diagnostic only and `{closure['complete_720_node_720_edge_cycle_found']}`.", "", "Canonical branch signatures are recomputed from q and include winding metadata. Branch labels are audit/provenance fields and are not hard edge rejection conditions. Bounded revolute deltas are raw q_to-q_from differences without periodic wrap. Native results are adaptive_discrete_interpolation; CCD and clearance are unavailable. Stage 2.5 remains blocked.", "", "```yaml", json.dumps(gate, ensure_ascii=False, indent=2, sort_keys=True), "```", ""]
    assert_output_path_writable(run_dir / "stage24g_report.md", operation="overwrite_report")
    (run_dir / "stage24g_report.md").write_text("\n".join(report), encoding="utf-8")
    metadata = {"run_index": run_index, "runtime_duration": time.time() - started, "node_set": semantic_hash(sorted((cid, row["waypoint_index"], row["joint_vector_rad"], row["canonical_branch_signature"]) for cid, row in nodes.items())), "edge_set": semantic_hash(sorted((key, value.get("source_stage"), value.get("max_joint_delta_deg")) for key, value in edges.items())), "native_validation_results": semantic_hash(native_evidence), "candidate_generation_attempts": semantic_hash(candidate_attempts), "seed_order": semantic_hash(seed_rows), "parameter_vectors": semantic_hash(parameter_schedule(contract)), "forward_reachability": semantic_hash(forward), "reverse_reachability": semantic_hash(reverse), "component_topology": semantic_hash(components), "selected_open_chain": semantic_hash(open_chain), "gate_status": semantic_hash({k: v for k, v in gate.items() if k not in {"native_edge_validation", "native_node_validation", "deterministic_rebuild", "tests"}}), "closure_diagnostic": semantic_hash(closure)}
    write_json(run_dir / "stage24g_run_metadata.json", metadata)
    return {"run_dir": str(run_dir.resolve()), "gate": gate, "metadata": metadata, "nodes": nodes, "edges": edges, "inventory": inventory, "native_evidence": native_evidence, "candidate_attempts": candidate_attempts, "seed_rows": seed_rows, "closure": closure, "fragile": fragile}


def run_focused_tests(out: Path) -> dict[str, Any]:
    regression_tests = [
        "tests/test_stage23b_outputs.py",
        "tests/test_stage24_outputs.py",
        "tests/test_stage24a_outputs.py",
        "tests/test_stage24b_outputs.py",
        "tests/test_stage24c_outputs.py",
        "tests/test_stage24d_outputs.py",
        "tests/test_stage24e_outputs.py",
        "tests/test_stage24f_outputs.py",
        "tests/test_stage24g_outputs.py",
    ]
    results: list[dict[str, Any]] = []
    for test_path in regression_tests:
        command = [sys.executable, "-m", "pytest", "-q", test_path]
        proc = subprocess.run(command, cwd=ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True, check=False)
        combined = f"{proc.stdout}\n{proc.stderr}"
        if proc.returncode == 0:
            status = "passed"
        elif "ModuleNotFoundError: No module named 'pyarrow'" in combined:
            status = "not_available"
        else:
            status = "failed"
        results.append({"test": test_path, "status": status, "exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr})
    failed = [item for item in results if item["status"] == "failed"]
    unavailable = [item for item in results if item["status"] == "not_available"]
    report = {"status": "failed" if failed else ("passed_with_not_available" if unavailable else "passed"), "failed_count": len(failed), "not_available_count": len(unavailable), "results": results}
    assert_output_path_writable(out / "stage24g_test_report.txt", operation="overwrite_test_report")
    (out / "stage24g_test_report.txt").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--skip-native", action="store_true", help="only for unit-test smoke runs; formal evidence must not use this flag")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = args.output_dir.resolve()
    assert_output_path_writable(out, operation="stage24g_output_root")
    if out.exists():
        raise RuntimeError(f"refusing to overwrite existing Stage 2.4G output directory: {out}")
    out.mkdir(parents=True, exist_ok=False)
    before = git_snapshot()
    frozen_hashes = recompute_frozen_hashes()
    input_manifest = authoritative_input_manifest(before, frozen_hashes)
    write_json(out / "stage24g_input_manifest.json", input_manifest)
    write_json(out / "stage24g_frozen_input_hashes.json", frozen_hashes)
    if frozen_hashes.get("failure_count"):
        gate = {"Stage_2_4G": "blocked_input_hash_mismatch", "ready_for_Stage_2_5": False, "Stage_2_5": "blocked", "frozen_input_hashes_unchanged": False, "input_hash_failures": frozen_hashes.get("failures", []), "continuous_tolerance_space_infeasible_proven": False}
        write_json(out / "stage24g_gate_report.json", gate)
        write_json(out / "stage24g_determinism_report.json", {"independent_runs": 0, "deterministic_rebuild": False})
        write_json(out / "stage24g_closed_cycle_certificate.json", {"open_chain_complete": False, "closure_edge_valid": False, "complete_cycle_found": False})
        assert_output_path_writable(out / "stage24g_report.md", operation="overwrite_report")
        (out / "stage24g_report.md").write_text("# Stage 2.4G\n\nBlocked before graph execution by frozen input hash mismatch.\n", encoding="utf-8")
        return 2
    semantics = joint_semantics()
    contract = frozen_tolerance_contract()
    run_results: list[dict[str, Any]] = []
    error: dict[str, Any] | None = None
    try:
        for index in range(1, args.runs + 1):
            run_results.append(run_one(out / f"run{index}", index, input_manifest, semantics, contract))
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        assert_output_path_writable(out / "stage24g_stderr.log", operation="overwrite_stderr_log")
        (out / "stage24g_stderr.log").write_text(traceback.format_exc(), encoding="utf-8")
    semantic_fields = ["node_set", "edge_set", "native_validation_results", "candidate_generation_attempts", "seed_order", "parameter_vectors", "forward_reachability", "reverse_reachability", "component_topology", "selected_open_chain", "gate_status", "closure_diagnostic"]
    determinism = {"independent_runs": len(run_results), "semantic_hashes": {field: [result["metadata"].get(field) for result in run_results] for field in semantic_fields}, "differences": {field: len({result["metadata"].get(field) for result in run_results}) - 1 for field in semantic_fields}, "deterministic_rebuild": error is None and len(run_results) == args.runs and all(len({result["metadata"].get(field) for result in run_results}) == 1 for field in semantic_fields), "nonsemantic_fields_excluded": ["absolute_output_directory", "run_timestamp", "runtime_duration", "process_id"]}
    write_json(out / "stage24g_determinism_report.json", determinism)
    if error:
        gate = {"Stage_2_4G": "blocked_native_environment_failure", "ready_for_Stage_2_5": False, "Stage_2_5": "blocked", "error": error, "frozen_input_hashes_unchanged": True, "continuous_tolerance_space_infeasible_proven": False}
    elif not determinism["deterministic_rebuild"]:
        gate = dict(run_results[0]["gate"])
        gate.update({"Stage_2_4G": "blocked_nondeterministic_rebuild", "deterministic_rebuild": "failed", "ready_for_Stage_2_5": False, "Stage_2_5": "blocked"})
    else:
        gate = dict(run_results[0]["gate"])
        gate["deterministic_rebuild"] = "passed"
        gate["ready_for_Stage_2_5"] = False
        gate["Stage_2_5"] = "blocked"
    write_json(out / "stage24g_gate_report.json", gate)
    write_json(out / "stage24g_closed_cycle_certificate.json", {"open_chain_complete": bool(gate.get("complete_0_to_719_open_chain_exists")), "closure_edge_valid": False, "complete_cycle_found": False, "closure_gate_status": "separate_719_to_0_gate_required"})
    # Root-level artifacts are deterministic aggregate certificates; each
    # independent run retains its own complete evidence under run1/run2/run3.
    if run_results:
        result = run_results[0]
        for name in ("stage24g_valid_nodes.jsonl", "stage24g_pair_prefilter_inventory.jsonl", "stage24g_native_edge_requests.jsonl", "stage24g_native_edge_evidence.jsonl", "stage24g_forward_reachability.json", "stage24g_reverse_reachability.json", "stage24g_component_summary.json", "stage24g_component_boundary_analysis.json", "stage24g_candidate_generation.jsonl", "stage24g_selected_open_chain.json", "stage24g_closure_diagnostic.json", "stage24g_fragile_transition_regression.json"):
            source = Path(result["run_dir"]) / name
            if source.exists():
                target = out / name
                target.write_bytes(source.read_bytes())
        for name in ("stage24g_edge_materialization_summary.csv", "stage24g_selected_open_chain.csv"):
            source = Path(result["run_dir"]) / name
            if source.exists():
                (out / name).write_bytes(source.read_bytes())
        write_jsonl(out / "stage24g_seed_ranking.jsonl", result["seed_rows"])
        write_jsonl(out / "stage24g_native_node_evidence.jsonl", [{"candidate_id": cid, "dual_backend_valid": True} for cid in sorted(result["nodes"]) if result["nodes"][cid].get("source_stage") == "Stage_2_4G"])
    test_report = run_focused_tests(out)
    gate["tests"] = test_report
    if test_report["status"] == "failed":
        gate["Stage_2_4G"] = "blocked_tests_failed"
        gate["ready_for_Stage_2_5"] = False
    write_json(out / "stage24g_gate_report.json", gate)
    root_report = [
        "# Stage 2.4G complete edge materialization and bidirectional component bridge",
        "",
        f"- Stage 2.4G: `{gate.get('Stage_2_4G')}`",
        f"- Independent deterministic rebuilds: `{determinism.get('independent_runs')}`; deterministic: `{determinism.get('deterministic_rebuild')}`.",
        f"- Frozen input hashes: `{input_manifest.get('frozen_input_hashes_unchanged')}`; checked entries: `{frozen_hashes.get('checked_count')}`; failures: `{frozen_hashes.get('failure_count')}`.",
        f"- Graph: `{gate.get('graph_node_count')}` nodes, `{gate.get('graph_edge_count')}` edges, `{gate.get('graph_component_count')}` components.",
        f"- Open chain 0->719: `{gate.get('complete_0_to_719_open_chain_exists')}`; first true global break: `{gate.get('first_true_global_break')}`.",
        f"- FCL/Bullet node difference: `{gate.get('fcl_bullet_node_difference')}`; edge difference: `{gate.get('fcl_bullet_edge_difference')}`.",
        f"- Tests: `{test_report.get('status')}`; unavailable modules: `{test_report.get('not_available_count')}` (pyarrow dependency); failed tests: `{test_report.get('failed_count')}`.",
        "- Collision method: `adaptive_discrete_interpolation`; CCD and clearance: `not_available`; Stage 2.5 remains `blocked`.",
        "- The 719→0 closure is diagnostic only and is not used as a Stage 2.4G pass condition.",
        "",
        "```json",
        json.dumps(gate, ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
    ]
    assert_output_path_writable(out / "stage24g_report.md", operation="overwrite_report")
    (out / "stage24g_report.md").write_text("\n".join(root_report), encoding="utf-8")
    write_json(out / "stage24g_run_metadata.json", {"independent_run_count": len(run_results), "run_directories": [result["run_dir"] for result in run_results], "determinism": determinism, "test_status": test_report.get("status"), "gate_status": gate.get("Stage_2_4G")})
    write_json(out / "dirty_worktree_preservation.json", {"git_before": before, "git_after": git_status(), "preexisting_status_preserved": all(line in git_status() for line in before["status_porcelain"].splitlines() if line.strip()), "new_output_root": str(out)})
    assert_output_path_writable(out / "stage24g_test_report.txt", operation="overwrite_test_report")
    (out / "stage24g_test_report.txt").write_text((out / "stage24g_test_report.txt").read_text(encoding="utf-8") + f"\nfinal_gate: {gate.get('Stage_2_4G')}\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "Stage_2_4G": gate.get("Stage_2_4G"), "runs": len(run_results), "deterministic_rebuild": gate.get("deterministic_rebuild"), "tests": test_report["status"], "error": error}, ensure_ascii=False, sort_keys=True))
    return 0 if gate.get("Stage_2_4G") == "passed_complete_dual_backend_open_chain" else 2


if __name__ == "__main__":
    raise SystemExit(main())
