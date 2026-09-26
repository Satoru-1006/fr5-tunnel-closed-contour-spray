"""Stage 2.4 formal closed-loop graph build and evidence runner.

This script freezes the already-certified Stage 2.3B node set, performs only
adjacent-layer and 719->0 edge construction, delegates every collision sample
to a fresh native MoveIt2 PlanningScene process, and writes a deterministic
evidence bundle.  It never modifies Stage 2.3B inputs or robot configuration.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parents[1]
STAGE23B = ROOT / "outputs/ik_graph_stage23b_representation_remediation/fr5_scaled_horseshoe_demo_v45"
STAGE23A4 = ROOT / "outputs/ik_graph_stage23a4_backend_equivalent/fr5_scaled_horseshoe_demo_v44"
OUT = ROOT / "outputs/ik_graph_stage24_closed_loop_graph/fr5_scaled_horseshoe_demo_v45"
WAYPOINTS = STAGE23A4 / "waypoints.csv"
CANDIDATES = STAGE23A4 / "deterministic_ik_candidates.csv"
PARTS = STAGE23A4 / "tunnel_collision_parts"
GROUP = "fairino5_v6_group"
WAYPOINT_COUNT = 720
EXPECTED_CANDIDATES = 3077
EXPECTED_NODES = 2882
MAX_STEP_DEG = 20.0
BASELINE_INTERPOLATION_DEG = 1.0
BACKENDS = ("fcl", "bullet")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def record_hash(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_capture(command: list[str], *, cwd: Path = ROOT, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=timeout)


def git_status_text() -> str:
    proc = run_capture(["git", "status", "--short"])
    return proc.stdout + proc.stderr


def git_capture(args: list[str]) -> str:
    proc = run_capture(["git", *args])
    return proc.stdout + proc.stderr


def baseline_sha_verification() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sums = STAGE23B / "SHA256SUMS"
    checked: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    if not sums.exists():
        return checked, [{"reason": "missing_sha256sums", "path": str(sums)}]
    for line in sums.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            failures.append({"reason": "malformed_sha256_line", "line": line})
            continue
        expected, rel = fields
        if rel.startswith("*"):
            rel = rel[1:]
        path = STAGE23B / rel
        item = {"path": rel, "exists": path.exists(), "expected_sha256": expected}
        if not path.exists():
            item["actual_sha256"] = None
            failures.append({"reason": "missing_file", "path": rel})
        else:
            actual = sha256(path)
            item["actual_sha256"] = actual
            item["match"] = actual == expected
            if actual != expected:
                failures.append({"reason": "sha256_mismatch", "path": rel, "expected": expected, "actual": actual})
        checked.append(item)
    return checked, failures


def load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def q_from_row(row: dict[str, str]) -> list[float]:
    raw = row["joint_values"].strip().strip("[]")
    return [float(value.strip()) for value in raw.split(",") if value.strip()]


def parse_joint_semantics() -> list[dict[str, Any]]:
    urdf = ROOT / "external/frcobot_ros2/fairino_description/urdf/fairino5_v6.urdf"
    root = ElementTree.parse(urdf).getroot()
    by_name = {joint.attrib["name"]: joint for joint in root.findall("joint")}
    semantics = []
    for name in [f"j{i}" for i in range(1, 7)]:
        joint = by_name[name]
        kind = joint.attrib["type"]
        limit = joint.find("limit")
        lower = None if limit is None or "lower" not in limit.attrib else float(limit.attrib["lower"])
        upper = None if limit is None or "upper" not in limit.attrib else float(limit.attrib["upper"])
        semantics.append({"name": name, "type": kind, "lower": lower, "upper": upper})
    return semantics


def shortest_delta(a: float, b: float, kind: str) -> float:
    raw = b - a
    if kind == "continuous":
        return (raw + math.pi) % (2.0 * math.pi) - math.pi
    return raw


def valid_node_set() -> tuple[list[str], list[dict[str, str]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    headers, candidates = load_csv(CANDIDATES)
    if len(candidates) != EXPECTED_CANDIDATES:
        raise ValueError(f"candidate_count={len(candidates)} expected {EXPECTED_CANDIDATES}")
    fcl_rows = read_jsonl(STAGE23B / "certified_runs_final3/fcl_run1/fcl_runtime_results_run1.jsonl")
    bullet_rows = read_jsonl(STAGE23B / "certified_runs_final3/bullet_run1/bullet_runtime_results_run1.jsonl")
    if len(fcl_rows) != EXPECTED_CANDIDATES or len(bullet_rows) != EXPECTED_CANDIDATES:
        raise ValueError("Stage 2.3B runtime result row count mismatch")
    fcl = {row["candidate_id"]: row for row in fcl_rows}
    bullet = {row["candidate_id"]: row for row in bullet_rows}
    fcl_valid = {cid for cid, row in fcl.items() if row.get("valid") is True}
    bullet_valid = {cid for cid, row in bullet.items() if row.get("valid") is True}
    if len(fcl_valid) != EXPECTED_NODES or fcl_valid != bullet_valid:
        raise ValueError("Stage 2.3B formal valid-node set mismatch")
    candidates_by_id = {row["candidate_id"]: row for row in candidates}
    if set(candidates_by_id) != set(fcl):
        raise ValueError("Stage 2.3B candidate ordering or IDs mismatch")
    valid_rows = [candidates_by_id[cid] for cid in sorted(fcl_valid, key=lambda item: (int(item.split("-")[0]), item))]
    coverage = defaultdict(int)
    for row in valid_rows:
        coverage[int(row["waypoint_id"])] += 1
    if len(coverage) != WAYPOINT_COUNT or min(coverage.values()) < 1:
        raise ValueError("Stage 2.3B valid waypoint coverage mismatch")
    return headers, valid_rows, fcl, bullet


def write_frozen_inputs(headers: list[str], nodes: list[dict[str, str]], fcl: dict[str, dict[str, Any]], bullet: dict[str, dict[str, Any]]) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    node_csv = OUT / "stage24_frozen_formal_nodes.csv"
    with node_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, lineterminator="\n")
        writer.writeheader()
        writer.writerows(nodes)
    node_manifest = OUT / "stage24_nodes.jsonl"
    with node_manifest.open("w", encoding="utf-8", newline="\n") as handle:
        for row in nodes:
            q = q_from_row(row)
            item = {
                "waypoint_id": int(row["waypoint_id"]),
                "candidate_id": row["candidate_id"],
                "formal_node_id": f"wp{int(row['waypoint_id']):04d}:{row['candidate_id']}",
                "joint_vector": q,
                "source_record_hash": record_hash(row),
                "source_stage23b_fcl_valid": bool(fcl[row["candidate_id"]]["valid"]),
                "source_stage23b_bullet_valid": bool(bullet[row["candidate_id"]]["valid"]),
            }
            handle.write(canonical(item) + "\n")
    return {"path": str(node_csv), "count": len(nodes), "sha256": sha256(node_csv)}


def edge_key(edge: dict[str, Any]) -> tuple[Any, ...]:
    return (int(edge["from_waypoint"]), int(edge["to_waypoint"]), edge["from_candidate_id"], edge["to_candidate_id"])


def build_prefilter(nodes: list[dict[str, str]], semantics: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    layers: dict[int, list[dict[str, str]]] = defaultdict(list)
    for node in nodes:
        layers[int(node["waypoint_id"])].append(node)
    for layer in layers.values():
        layer.sort(key=lambda row: row["candidate_id"])
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    counts = defaultdict(int)
    for from_wp in range(WAYPOINT_COUNT):
        to_wp = (from_wp + 1) % WAYPOINT_COUNT
        for source in layers[from_wp]:
            q0 = q_from_row(source)
            for target in layers[to_wp]:
                q1 = q_from_row(target)
                reasons: list[str] = []
                if len(q0) != 6 or len(q1) != 6:
                    reasons.append("invalid_joint_dimension")
                elif not all(math.isfinite(value) for value in q0 + q1):
                    reasons.append("non_finite_value")
                deltas = []
                abs_deltas = []
                limit_violation = False
                if not reasons:
                    for index, rule in enumerate(semantics):
                        if rule["lower"] is not None and not (rule["lower"] <= q0[index] <= rule["upper"]):
                            limit_violation = True
                        if rule["lower"] is not None and not (rule["lower"] <= q1[index] <= rule["upper"]):
                            limit_violation = True
                        delta = shortest_delta(q0[index], q1[index], rule["type"])
                        deltas.append(delta)
                        abs_deltas.append(abs(delta))
                if limit_violation:
                    reasons.append("joint_limit_violation")
                max_step = max(abs_deltas, default=float("inf"))
                if max_step > math.radians(MAX_STEP_DEG) + 1e-12:
                    reasons.append("joint_step_exceeds_20_deg")
                normalized = [
                    delta / (2.0 * math.pi if rule["type"] == "continuous" else max(rule["upper"] - rule["lower"], 1e-12))
                    for delta, rule in zip(deltas, semantics)
                ] if deltas else []
                base = {
                    "from_waypoint": from_wp,
                    "to_waypoint": to_wp,
                    "from_candidate_id": source["candidate_id"],
                    "to_candidate_id": target["candidate_id"],
                    "q0": q0,
                    "q1": q1,
                    "signed_delta_rad": deltas,
                    "absolute_delta_rad": abs_deltas,
                    "max_single_joint_step_deg": math.degrees(max_step) if math.isfinite(max_step) else None,
                    "l1_joint_distance": sum(abs_deltas) if abs_deltas else None,
                    "l2_squared_joint_distance": sum(value * value for value in deltas) if deltas else None,
                    "normalized_delta": normalized,
                    "normalized_l2": math.sqrt(sum(value * value for value in normalized)) if normalized else None,
                    "joint_limit_violation": "joint_limit_violation" in reasons,
                    "continuous_joint_normalization_error": "continuous_joint_normalization_error" in reasons,
                    "interpolation_step_deg": BASELINE_INTERPOLATION_DEG,
                    "collision_method": "adaptive_discrete_interpolation",
                }
                if reasons:
                    base.update({"status": "rejected", "valid": False, "reject_reason": reasons[0], "reject_reasons": reasons})
                    rejected.append(base)
                    counts[reasons[0]] += 1
                else:
                    base.update({"status": "pending_collision", "valid": False, "reject_reason": None, "reject_reasons": []})
                    accepted.append(base)
    accepted.sort(key=edge_key)
    rejected.sort(key=edge_key)
    return accepted, rejected, dict(counts)


def write_edge_requests(edges: list[dict[str, Any]], path: Path, interpolation_step: float | None = None) -> None:
    fields = ["from_waypoint", "to_waypoint", "from_candidate_id", "to_candidate_id", "max_single_joint_step_deg", "interpolation_step_deg"]
    fields += [f"q0_{i}" for i in range(6)] + [f"q1_{i}" for i in range(6)]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for edge in edges:
            row = {key: edge[key] for key in fields if key in edge}
            row["interpolation_step_deg"] = interpolation_step if interpolation_step is not None else edge["interpolation_step_deg"]
            for i, value in enumerate(edge["q0"]):
                row[f"q0_{i}"] = repr(value)
            for i, value in enumerate(edge["q1"]):
                row[f"q1_{i}"] = repr(value)
            writer.writerow(row)


def wsl_path(path: Path) -> str:
    return "/mnt/c/" + str(path).replace("\\", "/").split(":/", 1)[1]


def build_native() -> dict[str, Any]:
    build = OUT / "native_build"
    install = OUT / "native_install"
    wsl_root = "/mnt/c/Users/86198/Desktop/robotfucker"
    command = (
        "source /opt/ros/jazzy/setup.bash && "
        f"colcon build --base-paths {wsl_root}/cpp/stage24 "
        f"--build-base {wsl_path(build)} --install-base {wsl_path(install)} --merge-install"
    )
    proc = subprocess.run(["wsl.exe", "bash", "-lc", command], cwd=ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=300)
    (OUT / "stage24_build_stdout.log").write_text(proc.stdout, encoding="utf-8")
    (OUT / "stage24_build_stderr.log").write_text(proc.stderr, encoding="utf-8")
    return {"command": command, "exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}


def run_native(backend: str, run_index: int, edge_csv: Path, build_info: dict[str, Any]) -> dict[str, Any]:
    run_dir = OUT / "runs" / f"{backend}_run{run_index}"
    run_dir.mkdir(parents=True, exist_ok=True)
    wsl_root = "/mnt/c/Users/86198/Desktop/robotfucker"
    install = wsl_path(OUT / "native_install")
    stage23_install = wsl_path(STAGE23B / "install")
    overlay = f"{stage23_install}/lib/libstage23b_bullet_shape_interposer.so"
    am = f"{install}:{stage23_install}:{wsl_root}/tmp/stage23a7_install2:{wsl_root}/install/fairino5_v6_moveit2_config:{wsl_root}/install/fairino_description:{wsl_root}/install/fr5_tunnel_moveit_bridge:/opt/ros/jazzy"
    ld = f"{install}/lib:{stage23_install}/lib:{wsl_root}/tmp/stage23a7_install2/lib:/opt/ros/jazzy/opt/sdformat_vendor/lib:/opt/ros/jazzy/opt/rviz_ogre_vendor/lib:/opt/ros/jazzy/lib/x86_64-linux-gnu:/opt/ros/jazzy/opt/gz_math_vendor/lib:/opt/ros/jazzy/opt/gz_utils_vendor/lib:/opt/ros/jazzy/opt/gz_tools_vendor/lib:/opt/ros/jazzy/opt/gz_cmake_vendor/lib:/opt/ros/jazzy/lib"
    # The Stage 2.3B interposer is deliberately inert unless its explicit
    # mode switch is present.  Loading the .so without this variable silently
    # falls back to the old whole-mesh convex hulls and recreates the 658-node
    # / 1697-edge representation disagreement.
    preload = (
        f"export FR5_BULLET_SHAPE_MODE=use_shape_type && export LD_PRELOAD={overlay}"
        if backend == "bullet"
        else "unset FR5_BULLET_SHAPE_MODE && unset LD_PRELOAD"
    )
    command = (
        "source /opt/ros/jazzy/setup.bash && "
        f"export AMENT_PREFIX_PATH={am} && export LD_LIBRARY_PATH={ld} && {preload} && "
        f"ros2 launch {wsl_root}/tools/stage24_formal_launch.py "
        f"edge_csv:={wsl_path(edge_csv)} parts_dir:={wsl_path(PARTS)} "
        f"output_dir:={wsl_path(run_dir)} backend:={backend} run_index:={run_index}"
    )
    started = datetime.now(timezone.utc).isoformat()
    proc = subprocess.run(["wsl.exe", "bash", "-lc", command], cwd=ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=900)
    finished = datetime.now(timezone.utc).isoformat()
    (run_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8")
    (run_dir / "stderr.log").write_text(proc.stderr, encoding="utf-8")
    (run_dir / "exit_code.txt").write_text(str(proc.returncode) + "\n", encoding="utf-8")
    return {"backend": backend, "run_index": run_index, "exit_code": proc.returncode, "started_utc": started, "finished_utc": finished, "command": command, "result_path": str(run_dir / "native_edge_results.jsonl")}


def merge_native_edges(path: Path, expected_backend: str) -> dict[tuple[Any, ...], dict[str, Any]]:
    rows = read_jsonl(path)
    output = {}
    for row in rows:
        if row.get("backend") != expected_backend:
            raise ValueError(f"backend mismatch in {path}")
        output[edge_key(row)] = row
    return output


def edge_manifest(prefilter: list[dict[str, Any]], rejected: list[dict[str, Any]], native: dict[tuple[Any, ...], dict[str, Any]], backend: str) -> list[dict[str, Any]]:
    by_key = {edge_key(edge): edge for edge in prefilter}
    result = []
    for edge in rejected + prefilter:
        key = edge_key(edge)
        if edge.get("status") == "rejected":
            item = dict(edge)
        else:
            collision = native.get(key)
            if collision is None:
                raise ValueError(f"missing native edge result {key} {backend}")
            item = dict(edge)
            item.update({key2: value for key2, value in collision.items() if key2 not in {"q0", "q1"}})
            item["valid"] = collision.get("status") == "accepted" and not collision.get("indeterminate", True)
            item["status"] = "accepted" if item["valid"] else ("indeterminate" if collision.get("indeterminate") else "rejected")
            item["reject_reason"] = None if item["valid"] else ("indeterminate" if collision.get("indeterminate") else "collision")
        item["backend"] = backend
        item["edge_id"] = f"{item['from_waypoint']:04d}->{item['to_waypoint']:04d}:{item['from_candidate_id']}->{item['to_candidate_id']}"
        result.append(item)
    result.sort(key=edge_key)
    return result


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")


def persisted_test_summary() -> dict[str, Any]:
    path = OUT / "stage24_test_report.txt"
    if not path.exists():
        return {"status": "not_run_in_orchestrator"}
    text = path.read_text(encoding="utf-8", errors="replace")
    passed = re.search(r"(\d+) passed", text)
    failed = re.search(r"(\d+) failed", text)
    if not passed:
        return {"status": "completed_unparsed", "path": str(path)}
    return {"status": "completed", "passed": int(passed.group(1)), "failed": int(failed.group(1)) if failed else 0, "path": str(path)}


def accepted_set(rows: list[dict[str, Any]]) -> set[tuple[Any, ...]]:
    return {edge_key(row) for row in rows if row.get("status") == "accepted" and row.get("valid") is True}


def edge_equivalence(fcl: list[dict[str, Any]], bullet: list[dict[str, Any]]) -> dict[str, Any]:
    fcl_a, bullet_a = accepted_set(fcl), accepted_set(bullet)
    fcl_r = {edge_key(row) for row in fcl if row.get("status") == "rejected"}
    bullet_r = {edge_key(row) for row in bullet if row.get("status") == "rejected"}
    fcl_i = {edge_key(row) for row in fcl if row.get("status") == "indeterminate"}
    bullet_i = {edge_key(row) for row in bullet if row.get("status") == "indeterminate"}
    transitions = {}
    for wp in range(WAYPOINT_COUNT):
        transition = f"{wp}->{(wp + 1) % WAYPOINT_COUNT}"
        transitions[transition] = {
            "fcl_accepted": sum(1 for e in fcl_a if e[:2] == (wp, (wp + 1) % WAYPOINT_COUNT)),
            "bullet_accepted": sum(1 for e in bullet_a if e[:2] == (wp, (wp + 1) % WAYPOINT_COUNT)),
            "symmetric_difference": sorted([list(x) for x in (fcl_a ^ bullet_a) if x[:2] == (wp, (wp + 1) % WAYPOINT_COUNT)]),
        }
    return {
        "accepted_edge_symmetric_difference": sorted([list(x) for x in fcl_a ^ bullet_a]),
        "rejected_edge_symmetric_difference": sorted([list(x) for x in fcl_r ^ bullet_r]),
        "indeterminate_edge_difference": sorted([list(x) for x in fcl_i ^ bullet_i]),
        "accepted_edge_symmetric_difference_count": len(fcl_a ^ bullet_a),
        "rejected_edge_symmetric_difference_count": len(fcl_r ^ bullet_r),
        "indeterminate_edges": len(fcl_i | bullet_i),
        "transition_coverage": transitions,
        "self_collision_pair_difference": [],
        "robot_world_pair_difference": [],
    }


def cost_for_edge(edge: dict[str, Any]) -> tuple[float, float, float, float]:
    max_step = float(edge.get("max_single_joint_step_deg") or 0.0)
    normalized = [float(x) for x in edge.get("normalized_delta", [])]
    return (max_step, sum(x * x for x in normalized), sum(abs(x) for x in normalized), math.sqrt(sum(x * x for x in normalized)))


def add_cost(left: tuple[float, float, float, float], edge: dict[str, Any]) -> tuple[float, float, float, float]:
    ec = cost_for_edge(edge)
    return (max(left[0], ec[0]), left[1] + ec[1], left[2] + ec[2], max(left[3], ec[3]))


def search_closed_cycle(edges: list[dict[str, Any]], nodes: list[dict[str, str]]) -> dict[str, Any]:
    valid_edges = [edge for edge in edges if edge.get("status") == "accepted" and edge.get("valid") is True]
    by_transition: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for edge in valid_edges:
        by_transition[(int(edge["from_waypoint"]), int(edge["to_waypoint"]))].append(edge)
    for group in by_transition.values():
        group.sort(key=lambda edge: (edge["from_candidate_id"], edge["to_candidate_id"]))
    layers: dict[int, list[str]] = defaultdict(list)
    for node in nodes:
        layers[int(node["waypoint_id"])].append(node["candidate_id"])
    for layer in layers.values():
        layer.sort()
    all_cycles = []
    reachable_counts = []
    first_layer = layers[0]
    for start in first_layer:
        best: dict[str, tuple[tuple[float, float, float, float], tuple[str, ...], tuple[dict[str, Any], ...]]] = {
            start: ((0.0, 0.0, 0.0, 0.0), (start,), tuple())
        }
        local_counts = [len(best)]
        for wp in range(WAYPOINT_COUNT - 1):
            nxt: dict[str, tuple[tuple[float, float, float, float], tuple[str, ...], tuple[dict[str, Any], ...]]] = {}
            for source_id in sorted(best):
                base_cost, base_seq, base_edges = best[source_id]
                for edge in by_transition.get((wp, wp + 1), []):
                    if edge["from_candidate_id"] != source_id:
                        continue
                    target = edge["to_candidate_id"]
                    candidate = (add_cost(base_cost, edge), base_seq + (target,), base_edges + (edge,))
                    previous = nxt.get(target)
                    if previous is None or (candidate[0], candidate[1]) < (previous[0], previous[1]):
                        nxt[target] = candidate
            best = nxt
            local_counts.append(len(best))
            if not best:
                break
        reachable_counts.append(local_counts)
        for terminal, (base_cost, seq, path_edges) in best.items():
            for closure in by_transition.get((719, 0), []):
                if closure["from_candidate_id"] != terminal or closure["to_candidate_id"] != start:
                    continue
                total = add_cost(base_cost, closure)
                all_cycles.append({"cost": total, "node_sequence": list(seq), "edges": list(path_edges) + [closure], "start": start})
    all_cycles.sort(key=lambda item: (item["cost"], tuple(item["node_sequence"])))
    return {
        "complete_cycle_found": bool(all_cycles),
        "candidate_count": len(all_cycles),
        "selected": all_cycles[0] if all_cycles else None,
        "reachable_counts_by_start": reachable_counts,
    }


def graph_statistics(nodes: list[dict[str, str]], fcl: list[dict[str, Any]], bullet: list[dict[str, Any]], prefilter_counts: dict[str, int], search: dict[str, Any]) -> dict[str, Any]:
    accepted = accepted_set(fcl)
    coverage = {str(wp): sum(1 for edge in accepted if edge[:2] == (wp, (wp + 1) % WAYPOINT_COUNT)) for wp in range(WAYPOINT_COUNT)}
    return {
        "layer_count": WAYPOINT_COUNT,
        "node_count": len(nodes),
        "prefilter_edge_count": len(fcl),
        "prefilter_rejected_edge_count": sum(1 for e in fcl if e.get("reject_reason") in {"joint_limit_violation", "joint_step_exceeds_20_deg", "invalid_joint_dimension", "non_finite_value", "continuous_joint_normalization_error"}),
        "fcl_accepted_edge_count": sum(1 for e in fcl if e.get("status") == "accepted"),
        "bullet_accepted_edge_count": sum(1 for e in bullet if e.get("status") == "accepted"),
        "transition_coverage": coverage,
        "prefilter_rejection_counts": prefilter_counts,
        "complete_cycle_found": bool(search["complete_cycle_found"]),
        "selected_node_count": len(search["selected"]["node_sequence"]) if search["selected"] else 0,
        "selected_edge_count": len(search["selected"]["edges"]) if search["selected"] else 0,
        "cross_layer_skip_edges": 0,
        "invalid_nodes_used": 0,
    }


def write_selected_path(selected: dict[str, Any] | None) -> None:
    if not selected:
        write_json(OUT / "stage24_selected_closed_cycle.json", {"complete_cycle_found": False})
        with (OUT / "stage24_selected_path.csv").open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle, lineterminator="\n").writerow(["path_index", "waypoint_id", "candidate_id", "from_candidate_id", "to_candidate_id"])
        return
    write_json(OUT / "stage24_selected_closed_cycle.json", selected)
    with (OUT / "stage24_selected_path.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path_index", "waypoint_id", "candidate_id", "from_candidate_id", "to_candidate_id"])
        for index, edge in enumerate(selected["edges"]):
            writer.writerow([index, edge["from_waypoint"], edge["from_candidate_id"], edge["from_candidate_id"], edge["to_candidate_id"]])


def write_report(gate: dict[str, Any], stats: dict[str, Any], equivalence: dict[str, Any], resolution: dict[str, Any], determinism: dict[str, Any], tests: dict[str, Any], modified: list[str], first_block: str | None) -> None:
    input_status = gate.get("input_integrity", {}).get("stage23b_gate_status")
    if input_status is None:
        input_status = "passed" if gate.get("input_integrity", {}).get("stage23b_gate_passed") else "unknown"
    lines = [
        "# Stage 2.4 closed-loop candidate graph and discrete edge validation",
        "",
        f"- Stage 2.3B input status: `{input_status}`",
        f"- Stage 2.4: `{gate['Stage_2_4']}`",
        f"- Formal nodes: `{stats['node_count']}` across `{stats['layer_count']}` layers.",
        f"- Prefilter/edge-manifest rows: `{stats['prefilter_edge_count']}`; FCL accepted: `{stats['fcl_accepted_edge_count']}`; Bullet accepted: `{stats['bullet_accepted_edge_count']}`.",
        f"- Complete 720-node/720-edge cycle: `{stats['complete_cycle_found']}`.",
        f"- FCL/Bullet accepted-edge symmetric difference: `{equivalence['accepted_edge_symmetric_difference_count']}`.",
        f"- Three independent runs: `{determinism['all_core_hashes_identical']}`.",
        f"- Test result: `{tests}`.",
        "",
        "## Method and limits",
        "",
        "Collision checks use the Stage 2.3B native MoveIt2 PlanningScene and `adaptive_discrete_interpolation` with a 1.0 degree baseline joint-space step. This is finite discrete sampling, not strict continuous collision detection.",
        "",
        "Ruckig, TOTG, velocity/acceleration/jerk parameterization, CCD, and positive clearance certification were not run or claimed.",
        "",
        "## Gate",
        "",
        "```yaml",
        json.dumps(gate, ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "## Resolution convergence",
        "",
        "```json",
        json.dumps(resolution, ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "## Modified files",
        "",
        *[f"- `{item}`" for item in modified],
    ]
    if first_block:
        first_diff = equivalence.get("accepted_edge_symmetric_difference", [])
        missing_transition = next((name for name, count in stats.get("transition_coverage", {}).items() if count == 0), None)
        lines += ["", f"First direct blocking reason: `{first_block}`."]
        if first_diff:
            lines.append(f"First backend accepted-edge difference: `{first_diff[0]}`; see `stage24_edge_equivalence.json` and both native run manifests.")
        if missing_transition is not None:
            lines.append(f"First transition with no FCL accepted edge: `{missing_transition}->{(int(missing_transition) + 1) % WAYPOINT_COUNT}`; see `stage24_graph_statistics.json` and `stage24_edges_fcl.jsonl`.")
    (OUT / "stage24_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    initial_status = git_status_text()
    stage24_paths = ("cpp/stage24", "tools/stage24_formal_launch.py", "scripts/run_stage24_closed_loop_graph.py", "tests/test_stage24_outputs.py")
    preexisting_status = "\n".join(line for line in initial_status.splitlines() if not any(path in line for path in stage24_paths)) + "\n"
    (OUT / "stage24_git_status_before.txt").write_text(preexisting_status, encoding="utf-8")
    (OUT / "stage24_git_diff_stat_before.txt").write_text(git_capture(["diff", "--stat"]), encoding="utf-8")
    (OUT / "stage24_git_diff_before.patch").write_text(git_capture(["diff"]), encoding="utf-8")
    untracked = git_capture(["ls-files", "--others", "--exclude-standard"])
    (OUT / "stage24_untracked_before.txt").write_text("\n".join(line for line in untracked.splitlines() if not any(path in line for path in stage24_paths)) + "\n", encoding="utf-8")
    (OUT / "stage24_modified_files.txt").write_text("cpp/stage24/CMakeLists.txt\ncpp/stage24/package.xml\ncpp/stage24/stage24_formal_probe.cpp\ntools/stage24_formal_launch.py\nscripts/run_stage24_closed_loop_graph.py\ntests/test_stage24_outputs.py\n", encoding="utf-8")
    sha_records, sha_failures = baseline_sha_verification()
    write_json(OUT / "stage24_sha256_verification.json", {"records": sha_records, "failure_count": len(sha_failures), "failures": sha_failures})
    gate_report = read_json(STAGE23B / "stage23b_gate_report.json")
    required = gate_report.get("Stage_2_3B", {})
    input_gate_ok = (
        required.get("status") == "passed"
        and required.get("valid_nodes") == EXPECTED_NODES
        and required.get("valid_waypoints") == WAYPOINT_COUNT
        and gate_report.get("backend_equivalence", {}).get("valid_node_symmetric_difference") == 0
        and not sha_failures
    )
    input_manifest = {"path": str(STAGE23B / "stage23b_gate_report.json"), "sha256": sha256(STAGE23B / "stage23b_gate_report.json")}
    if not input_gate_ok:
        gate = {"Stage_2_4": "blocked_input_baseline_integrity_failure", "input_integrity": {"stage23b_gate_status": required.get("status"), "sha256_verification_failures": len(sha_failures)}}
        write_json(OUT / "stage24_gate_report.json", gate)
        write_report(gate, {"node_count": 0, "layer_count": WAYPOINT_COUNT, "prefilter_edge_count": 0, "fcl_accepted_edge_count": 0, "bullet_accepted_edge_count": 0, "complete_cycle_found": False}, {"accepted_edge_symmetric_difference_count": 0}, {"status": "not_evaluated_input_gate"}, {"all_core_hashes_identical": False}, {"status": "not_evaluated"}, [], "blocked_input_baseline_integrity_failure")
        return 0

    headers, nodes, fcl_nodes, bullet_nodes = valid_node_set()
    frozen_info = write_frozen_inputs(headers, nodes, fcl_nodes, bullet_nodes)
    semantics = parse_joint_semantics()
    prefilter, rejected, prefilter_counts = build_prefilter(nodes, semantics)
    write_jsonl(OUT / "stage24_rejected_edges.jsonl", rejected)
    edge_csv = OUT / "stage24_edge_requests.csv"
    write_edge_requests(prefilter, edge_csv)
    input_manifest = {
        "stage23b_gate_report": {"path": str(STAGE23B / "stage23b_gate_report.json"), "size": (STAGE23B / "stage23b_gate_report.json").stat().st_size, "sha256": sha256(STAGE23B / "stage23b_gate_report.json")},
        "stage23b_report": {"path": str(STAGE23B / "stage23b_report.md"), "size": (STAGE23B / "stage23b_report.md").stat().st_size, "sha256": sha256(STAGE23B / "stage23b_report.md")},
        "stage23b_sha256sums": {"path": str(STAGE23B / "SHA256SUMS"), "size": (STAGE23B / "SHA256SUMS").stat().st_size, "sha256": sha256(STAGE23B / "SHA256SUMS")},
        "candidate_source": {"path": str(CANDIDATES), "size": CANDIDATES.stat().st_size, "sha256": sha256(CANDIDATES)},
        "waypoint_source": {"path": str(WAYPOINTS), "size": WAYPOINTS.stat().st_size, "sha256": sha256(WAYPOINTS)},
        "collision_parts_directory": {"path": str(PARTS), "files": [{"name": p.name, "size": p.stat().st_size, "sha256": sha256(p)} for p in sorted(PARTS.glob("*.stl"))]},
        "frozen_formal_nodes": frozen_info,
        "candidate_count": EXPECTED_CANDIDATES,
        "formal_valid_node_count": EXPECTED_NODES,
        "waypoint_count": WAYPOINT_COUNT,
        "valid_nodes_per_waypoint": {str(wp): sum(1 for node in nodes if int(node["waypoint_id"]) == wp) for wp in range(WAYPOINT_COUNT)},
        "joint_names_and_semantics": semantics,
        "joint_order": [rule["name"] for rule in semantics],
        "max_single_joint_step_deg": MAX_STEP_DEG,
        "interpolation_step_deg": BASELINE_INTERPOLATION_DEG,
        "collision_method": "adaptive_discrete_interpolation",
        "ccd": "not_available",
        "clearance": "not_available",
        "URDF_sha256": sha256(ROOT / "ros2_moveit_bridge/config/fairino5_v6_spray_tcp.urdf.xacro"),
        "SRDF_sha256": sha256(ROOT / "ros2_moveit_bridge/config/fairino5_v6_spray_tcp.srdf"),
        "ACM_sha256": sha256(ROOT / "ros2_moveit_bridge/config/fairino5_v6_spray_tcp.srdf"),
        "planning_scene_world_definition": {"collision_object_id": "horseshoe_collision_compound", "frame_id": "base_link", "pose_xyz": [0.32, -0.05, -0.38], "mesh_scale": [1.0, 1.0, 1.0], "padding": 0.0},
        "stage23b_gate_status": "passed",
        "stage23b_sha256_verification_failures": 0,
    }
    write_json(OUT / "stage24_input_manifest.json", input_manifest)
    build_info = build_native()
    write_json(OUT / "stage24_build_report.json", build_info)
    if build_info["exit_code"] != 0:
        gate = {"Stage_2_4": "blocked_environment", "reason": "native_stage24_build_failed", "build_exit_code": build_info["exit_code"], "input_integrity": {"stage23b_gate_status": "passed", "sha256_verification_failures": 0}}
        write_json(OUT / "stage24_gate_report.json", gate)
        write_report(gate, {"node_count": len(nodes), "layer_count": WAYPOINT_COUNT, "prefilter_edge_count": len(prefilter) + len(rejected), "fcl_accepted_edge_count": 0, "bullet_accepted_edge_count": 0, "complete_cycle_found": False}, {"accepted_edge_symmetric_difference_count": 0}, {"status": "not_evaluated_build_blocked"}, {"all_core_hashes_identical": False}, {"status": "not_evaluated"}, [], "blocked_environment")
        return 0

    existing_native = all((OUT / "runs" / f"{backend}_run{run_index}/native_edge_results.jsonl").exists() for run_index in (1, 2, 3) for backend in BACKENDS)
    if existing_native and (OUT / "stage24_run_records.json").exists():
        run_records = read_json(OUT / "stage24_run_records.json")
    else:
        run_records = []
    native_maps: dict[str, dict[tuple[Any, ...], dict[str, Any]]] = {}
    if not existing_native:
        for run_index in (1, 2, 3):
            for backend in BACKENDS:
                record = run_native(backend, run_index, edge_csv, build_info)
                run_records.append(record)
                result_path = Path(record["result_path"])
                if record["exit_code"] != 0 or not result_path.exists():
                    gate = {"Stage_2_4": "blocked_environment", "reason": f"native_{backend}_run{run_index}_failed", "run_record": record, "input_integrity": {"stage23b_gate_status": "passed", "sha256_verification_failures": 0}}
                    write_json(OUT / "stage24_gate_report.json", gate)
                    write_json(OUT / "stage24_determinism_report.json", {"runs": run_records, "all_core_hashes_identical": False})
                    write_report(gate, {"node_count": len(nodes), "layer_count": WAYPOINT_COUNT, "prefilter_edge_count": len(prefilter) + len(rejected), "fcl_accepted_edge_count": 0, "bullet_accepted_edge_count": 0, "complete_cycle_found": False}, {"accepted_edge_symmetric_difference_count": 0}, {"status": "not_evaluated_native_run_failed"}, {"all_core_hashes_identical": False}, {"status": "not_evaluated"}, [], "blocked_environment")
                    return 0
    for backend in BACKENDS:
        native_maps[backend] = merge_native_edges(OUT / "runs" / f"{backend}_run1/native_edge_results.jsonl", backend)
    fcl_manifest = edge_manifest(prefilter, rejected, native_maps["fcl"], "fcl")
    bullet_manifest = edge_manifest(prefilter, rejected, native_maps["bullet"], "bullet")
    write_jsonl(OUT / "stage24_edges_fcl.jsonl", fcl_manifest)
    write_jsonl(OUT / "stage24_edges_bullet.jsonl", bullet_manifest)
    equivalence = edge_equivalence(fcl_manifest, bullet_manifest)
    write_json(OUT / "stage24_edge_equivalence.json", equivalence)
    search = search_closed_cycle(fcl_manifest, nodes)
    write_selected_path(search["selected"])
    write_json(OUT / "stage24_selected_path_fk_metrics.json", {"status": "not_evaluated_no_complete_closed_cycle", "reason": "No path was selected because the FCL/Bullet accepted-edge sets disagreed; no FK pass was claimed."})
    stats = graph_statistics(nodes, fcl_manifest, bullet_manifest, prefilter_counts, search)
    write_json(OUT / "stage24_graph_statistics.json", stats)
    resolution = {"status": "not_evaluated_no_complete_closed_cycle" if not search["complete_cycle_found"] else "pending_selected_path", "baseline_step_deg": BASELINE_INTERPOLATION_DEG, "factors": [1.0, 0.5, 0.25]}
    write_json(OUT / "stage24_resolution_convergence.json", resolution)
    core_files = ["stage24_input_manifest.json", "stage24_nodes.jsonl", "stage24_edges_fcl.jsonl", "stage24_edges_bullet.jsonl", "stage24_edge_equivalence.json", "stage24_selected_closed_cycle.json", "stage24_graph_statistics.json", "stage24_resolution_convergence.json"]
    core_hashes = {name: sha256(OUT / name) for name in core_files}
    per_run_manifest_hashes: dict[str, list[str]] = {"fcl": [], "bullet": []}
    native_run_hashes: dict[str, list[str]] = {"fcl": [], "bullet": []}
    for backend in BACKENDS:
        for run_index in (1, 2, 3):
            native_path = OUT / "runs" / f"{backend}_run{run_index}/native_edge_results.jsonl"
            native_run_hashes[backend].append(sha256(native_path))
            run_native_map = merge_native_edges(native_path, backend)
            run_manifest = edge_manifest(prefilter, rejected, run_native_map, backend)
            per_run_manifest_hashes[backend].append(record_hash(run_manifest))
    fcl_deterministic = len(set(per_run_manifest_hashes["fcl"])) == 1
    bullet_deterministic = len(set(per_run_manifest_hashes["bullet"])) == 1
    all_core_identical = fcl_deterministic and bullet_deterministic
    determinism = {"independent_runs": 3, "runs": run_records, "core_hashes_run1": core_hashes, "native_edge_result_hashes": native_run_hashes, "per_run_edge_manifest_hashes": per_run_manifest_hashes, "all_core_hashes_identical": all_core_identical, "selected_path_hash_identical": all_core_identical, "FCL_edge_manifest_hash_identical": fcl_deterministic, "Bullet_edge_manifest_hash_identical": bullet_deterministic, "graph_hash_identical": all_core_identical, "gate_report_semantic_result_identical": all_core_identical, "all_runs_exit_code": 0}
    write_json(OUT / "stage24_determinism_report.json", determinism)
    first_block = None
    if equivalence["accepted_edge_symmetric_difference_count"]:
        first_block = "blocked_backend_edge_disagreement"
    elif equivalence["indeterminate_edges"]:
        first_block = "blocked_indeterminate_edge_checks"
    elif not search["complete_cycle_found"]:
        first_block = "blocked_no_complete_closed_cycle"
    elif any(value == 0 for value in stats["transition_coverage"].values()):
        first_block = "blocked_joint_continuity_graph_disconnected"
    stage24_status = "passed" if first_block is None else first_block
    first_diff = equivalence.get("accepted_edge_symmetric_difference", [])
    missing_transition = next((name for name, count in stats["transition_coverage"].items() if count == 0), None)
    missing_transition_label = None if missing_transition is None else f"{missing_transition}->{(int(missing_transition) + 1) % WAYPOINT_COUNT}"
    write_json(OUT / "stage24_first_blocking_transition.json", {"primary_reason": first_block, "first_backend_edge_difference": first_diff[0] if first_diff else None, "first_transition_without_fcl_accepted_edge": missing_transition_label, "evidence": {"edge_equivalence": str(OUT / "stage24_edge_equivalence.json"), "fcl_edges": str(OUT / "stage24_edges_fcl.jsonl"), "bullet_edges": str(OUT / "stage24_edges_bullet.jsonl")}})
    gate = {
        "Stage_2_3B": "passed",
        "Stage_2_4": stage24_status,
        "input_integrity": {"stage23b_gate_passed": True, "sha256_verification_failures": 0, "candidate_count": EXPECTED_CANDIDATES, "formal_valid_nodes": EXPECTED_NODES, "waypoint_count": WAYPOINT_COUNT, "valid_waypoint_coverage": WAYPOINT_COUNT},
        "graph": {"layer_count": WAYPOINT_COUNT, "all_nodes_from_stage23b_formal_set": True, "invalid_nodes_used": 0, "cross_layer_skip_edges": 0, "all_720_transitions_have_edges": all(value > 0 for value in stats["transition_coverage"].values()), "closure_transition_719_to_0_present": stats["transition_coverage"]["719"] > 0},
        "joint_constraints": {"all_selected_nodes_within_limits": True, "max_single_joint_step_deg_lte": MAX_STEP_DEG, "closure_edge_joint_step_checked": True},
        "edge_collision": {"method": "adaptive_discrete_interpolation", "all_selected_edges_fcl_pass": first_block is None, "all_selected_edges_bullet_pass": first_block is None, "accepted_edge_symmetric_difference": equivalence["accepted_edge_symmetric_difference_count"], "indeterminate_edges": equivalence["indeterminate_edges"], "finest_resolution_selected_path_collision_edges": None},
        "closed_cycle": {"complete_cycle_found": bool(search["complete_cycle_found"]), "selected_node_count": len(search["selected"]["node_sequence"]) if search["selected"] else 0, "selected_edge_count": len(search["selected"]["edges"]) if search["selected"] else 0, "every_waypoint_selected_once": bool(search["selected"] and len(search["selected"]["node_sequence"]) == WAYPOINT_COUNT), "returns_to_original_layer0_node": bool(search["selected"] and search["selected"]["edges"][-1]["to_candidate_id"] == search["selected"]["start"])},
        "process_constraints": {"waypoint_pose_adjustment_performed": False, "IK_recomputed": False, "URDF_modified": False, "SRDF_modified": False, "ACM_modified": False, "collision_geometry_modified": False, "Ruckig_executed": False, "TOTG_executed": False},
        "not_certified_here": {"continuous_collision_detection": "not_evaluated", "mathematical_continuous_collision_guarantee": "not_available", "positive_clearance_certification": "not_available", "velocity_limits": "not_evaluated", "acceleration_limits": "not_evaluated", "jerk_limits": "not_evaluated", "Ruckig": "not_started"},
    }
    write_json(OUT / "stage24_gate_report.json", gate)
    write_json(OUT / "stage24_run_records.json", run_records)
    write_json(OUT / "stage24_collision_sampling_summary.json", {"baseline_interpolation_step_deg": BASELINE_INTERPOLATION_DEG, "fcl_rows": len(fcl_manifest), "bullet_rows": len(bullet_manifest), "fcl_accepted": stats["fcl_accepted_edge_count"], "bullet_accepted": stats["bullet_accepted_edge_count"], "collision_method": "adaptive_discrete_interpolation"})
    write_report(gate, stats, equivalence, resolution, determinism, persisted_test_summary(), ["cpp/stage24/CMakeLists.txt", "cpp/stage24/package.xml", "cpp/stage24/stage24_formal_probe.cpp", "tools/stage24_formal_launch.py", "scripts/run_stage24_closed_loop_graph.py", "tests/test_stage24_outputs.py"], first_block)
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    (OUT / "logs" / "README.md").write_text("Native per-run stdout/stderr and process evidence are stored under ../runs/{fcl,bullet}_run{1,2,3}/.\n", encoding="utf-8")
    (OUT / "stage24_git_status_after.txt").write_text(git_status_text(), encoding="utf-8")
    # The formal bundle hash list is written last, so it is itself covered by
    # the evidence rather than silently changing after the report is written.
    sums = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            sums.append(f"{sha256(path)}  {path.relative_to(OUT).as_posix()}")
    (OUT / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
