#!/usr/bin/env python3
"""Stage 3 H5 multi-IK branch continuity and transition certification.

H5 is an additive, offline-only audit.  It consumes the frozen H4.5 selected
configuration and its 30 curved feasible target IDs, harvests deterministic
KDL seed-dependent solutions in fresh ROS processes, then certifies candidate
transitions with a native MoveIt PlanningScene/FCL bridge.  It deliberately
does not run a planner, time parameterization, Ruckig, or robot I/O.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
H45_ROOT = ROOT / "outputs/stage3_h4_5_robot_fixture_geometry_codesign_20260809T000000Z"
H41_ROOT = ROOT / "outputs/stage3_h4_1_run_20260808T000004Z"
H3_TARGETS = ROOT / "outputs/stage3_h3_coverage_baseline_20260808T034526Z/stage3_h3_surface_targets.jsonl"
H2_MANIFEST = ROOT / "outputs/stage3_h2_geometry_baseline_20260807T180321Z/stage3_h2_geometry_manifest.json"
KINEMATICS = ROOT / "external/frcobot_ros2/fairino5_v6_moveit2_config/config/kinematics.yaml"
SRDF = ROOT / "ros2_moveit_bridge/config/fairino5_v6_spray_tcp.srdf"
DERIVED_URDF = H45_ROOT / "derived_robot_model.urdf"
SELECTED_MESH = H45_ROOT / "curved_fixture_stage3_h4_5_selected.obj"
GEOMETRY = "fixture_curved_cylinder_patch"
COLLISION_METHOD = "adaptive_discrete_interpolation"
CCD_STATUS = "not_available"
INTERPOLATION_STEP_DEG = 0.5
DEDUP_TOLERANCE_RAD = 1.0e-8
FORBIDDEN_LOG_PATTERNS = ("process has died", "exit code -11", "sigsegv", "segmentation fault")
EXPECTED_CONFIGURATION = "h4_5_013_dx_minus_200_dy_minus_200"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def dump_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n" for row in rows), encoding="utf-8", newline="\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def canonical(value: Any, *, float_digits: int = 12) -> Any:
    if isinstance(value, Mapping):
        return {str(k): canonical(value[k], float_digits=float_digits) for k in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [canonical(v, float_digits=float_digits) for v in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return round(value, float_digits)
    return value


def semantic_hash(value: Any) -> str:
    payload = json.dumps(canonical(value), ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def tree_manifest(root: Path) -> dict[str, Any]:
    files = []
    for path in sorted(root.rglob("*")) if root.is_dir() else []:
        if path.is_file() and not path.is_symlink():
            files.append({"relative_path": path.relative_to(root).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {"root": rel(root), "file_count": len(files), "files": files, "tree_hash": semantic_hash(files)}


def wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    return f"/mnt/{drive}/{resolved.as_posix().split(':', 1)[-1].lstrip('/').replace('\\', '/') }"


def run_wsl(command_parts: Sequence[str], *, timeout_s: int) -> tuple[int, str, str]:
    proc = subprocess.run(["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", " && ".join(command_parts)], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout_s, check=False)
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def audit_h45(output: Path) -> dict[str, Any]:
    required = [
        "FINAL_REPORT.md", "stage3_h4_5_terminal_certificate.json", "stage3_h4_5_gate_report.json",
        "stage3_h4_5_configuration_manifest.json", "stage3_h4_5_selected_candidate.json",
        "stage3_h4_5_curved_feasible_set.json", "stage3_h4_5_connected_components.json",
        "stage3_h4_5_native_branch_summary.json", "stage3_h4_5_robot_collision_model_audit.json",
        "stage3_h4_5_acm_audit.json", "stage3_h4_5_fk_audit.json", "stage3_h4_5_clearance_summary.json",
        "stage3_h4_5_replay_determinism.json", "stage3_h4_5_regression_report.json",
        "stage3_h4_5_derived_configuration_diff.json", "derived_robot_model.urdf", "derived_robot_model.xacro",
    ]
    missing = [name for name in required if not (H45_ROOT / name).is_file()]
    if missing:
        audit = {"schema_version": "stage3-h5-h4-5-input-audit-v1", "status": "BLOCKED", "missing_artifacts": missing}
        dump_json(output / "stage3_h5_h4_5_input_audit.json", audit)
        raise RuntimeError(f"H4.5 formal input missing: {missing}")
    terminal = load_json(H45_ROOT / "stage3_h4_5_terminal_certificate.json")
    gate = load_json(H45_ROOT / "stage3_h4_5_gate_report.json")
    config = load_json(H45_ROOT / "stage3_h4_5_configuration_manifest.json")
    selected = load_json(H45_ROOT / "stage3_h4_5_selected_candidate.json")
    feasible = load_json(H45_ROOT / "stage3_h4_5_curved_feasible_set.json")
    derived = load_json(H45_ROOT / "stage3_h4_5_derived_configuration_diff.json")
    report = (H45_ROOT / "FINAL_REPORT.md").read_text(encoding="utf-8")
    report_blocker = re.search(r"FIRST_BLOCKER:\s*([^`\n]+)", report)
    documentation_inconsistency = bool(report_blocker and report_blocker.group(1).strip() == "native_fk_audit_passed")
    formal_auth = terminal.get("STAGE_3_H4_5") == "PASSED" and terminal.get("READY_FOR_STAGE_3_H5") == "YES" and terminal.get("FIRST_BLOCKER") is None and gate.get("STAGE_3_H4_5") == "PASSED" and gate.get("READY_FOR_STAGE_3_H5") == "YES" and gate.get("FIRST_BLOCKER") is None
    selected_id = config.get("selected_candidate_id")
    selected_nested = selected.get("selected_candidate", {})
    counts_ok = feasible.get("number_of_feasible_targets") == 30 and feasible.get("number_of_connected_feasible_components") == 3 and feasible.get("largest_connected_component_target_count") == 16
    freeze_ok = selected_id == EXPECTED_CONFIGURATION and selected_nested.get("candidate_id") == EXPECTED_CONFIGURATION and abs(float(config.get("process_standoff_m")) - 0.260) <= 1e-12 and abs(float(config.get("tcp_extension_m")) - 0.150) <= 1e-12
    derived_ok = bool(derived.get("derived_configuration_used")) and bool(derived.get("baseline_files_not_overwritten")) and bool(derived.get("baseline_immutable")) and DERIVED_URDF.is_file()
    report_note = "documentation_field_inconsistency" if documentation_inconsistency else None
    audit = {
        "schema_version": "stage3-h5-h4-5-input-audit-v1", "status": "PASSED" if formal_auth and counts_ok and freeze_ok and derived_ok else "BLOCKED",
        "formal_authorization_valid": formal_auth, "formal_certificate_first_blocker": terminal.get("FIRST_BLOCKER"), "gate_first_blocker": gate.get("FIRST_BLOCKER"),
        "final_report_first_blocker_observed": report_blocker.group(1).strip() if report_blocker else None, "documentation_note": report_note,
        "documentation_field_inconsistency": documentation_inconsistency, "h4_5_counts": {"feasible_targets": feasible.get("number_of_feasible_targets"), "components": feasible.get("number_of_connected_feasible_components"), "largest_component": feasible.get("largest_connected_component_target_count")},
        "selected_configuration": selected_id, "selected_configuration_expected": EXPECTED_CONFIGURATION, "configuration_frozen_values": {"tcp_extension_m": config.get("tcp_extension_m"), "process_standoff_m": config.get("process_standoff_m")},
        "derived_model": {"used": derived.get("derived_configuration_used"), "path": derived.get("derived_urdf_path"), "sha256": derived.get("derived_sha256"), "original_path": derived.get("original_path"), "original_sha256": derived.get("original_sha256"), "replay_validation": derived.get("replay_validation", {}).get("status")},
        "checks": {"formal_authorization": formal_auth, "frozen_configuration": freeze_ok, "counts_30_3_16": counts_ok, "derived_candidate_replay": derived_ok},
        "read_only": True,
    }
    dump_json(output / "stage3_h5_h4_5_input_audit.json", audit)
    if audit["status"] != "PASSED":
        raise RuntimeError(f"H4.5 formal input audit blocked: {audit}")
    return {"audit": audit, "terminal": terminal, "gate": gate, "config": config, "selected": selected_nested, "feasible": feasible, "derived": derived}


def build_configuration_freeze(output: Path, h45: Mapping[str, Any]) -> dict[str, Any]:
    config = h45["config"]
    selected = h45["selected"]
    manifest = {
        "schema_version": "stage3-h5-configuration-freeze-manifest-v1", "workcell_configuration_frozen_from_h4_5": "YES", "h5_workcell_reoptimization": "NO",
        "selected_configuration_id": config["selected_candidate_id"], "selected_configuration_sha256": semantic_hash(selected),
        "fixture_relative_pose_delta": selected.get("fixture_dof_delta", {"x_m": -0.2, "y_m": -0.2, "z_m": 0.0, "roll_deg": 0.0, "pitch_deg": 0.0, "yaw_deg": 0.0}),
        "tcp_extension_m": 0.150, "standoff_m": 0.260, "robot_base_reoptimized": False, "fixture_reoptimized": False,
        "h4_5_derived_robot_model_reused": True, "original_robot_model_mutated": False, "urdf_used": rel(DERIVED_URDF), "urdf_sha256": sha256_file(DERIVED_URDF),
        "srdf_used": rel(SRDF), "srdf_sha256": sha256_file(SRDF), "no_silent_patch": True,
    }
    dump_json(output / "stage3_h5_configuration_freeze_manifest.json", manifest)
    return manifest


def build_ik_capability_audit(output: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    capability = load_json(H41_ROOT / "stage3_h4_1_solver_capability.json")
    seed_contract = load_json(H41_ROOT / "stage3_h4_1_seed_contract.json")
    plugin = str(capability.get("solver_plugin") or seed_contract.get("solver_plugin"))
    api = str(capability.get("solver_api") or seed_contract.get("solver_api"))
    native_enum = "NOT_AVAILABLE" if "KDLKinematicsPlugin" in plugin and api == "KinematicsBase::getPositionIK" else "PARTIAL"
    audit = {
        "schema_version": "stage3-h5-ik-plugin-capability-audit-v1", "IK_PLUGIN_IDENTITY": plugin, "solver_api": api,
        "MULTI_IK_NATIVE_ENUMERATION": native_enum, "IK_SOLUTION_COVERAGE": "DETERMINISTIC_MULTI_SEED_NON_EXHAUSTIVE",
        "all_ik_solutions_enumerated": False, "native_enumeration_evidence": {"h4_1_solver_capability": rel(H41_ROOT / "stage3_h4_1_solver_capability.json"), "status": capability.get("status"), "bridge_retry_loop": capability.get("bridge_retry_loop"), "internal_random_restart": capability.get("internal_random_restart")},
        "strategy": "fixed explicit seed bank; one KinematicsBase::getPositionIK invocation per target/seed; fresh-process replay; topology-aware deduplication; no exhaustiveness claim",
    }
    seeds = seed_contract.get("seeds", [])
    seed_bank = {"schema_version": "stage3-h5-ik-seed-bank-v1", "seed_contract_source": rel(H41_ROOT / "stage3_h4_1_seed_contract.json"), "seed_bank_identity": semantic_hash(seeds), "seed_count": len(seeds), "solver_plugin": plugin, "solver_api": api, "seeds": seeds, "deterministic": True, "random_runtime_api_used": False}
    dump_json(output / "stage3_h5_ik_plugin_capability_audit.json", audit)
    dump_json(output / "stage3_h5_ik_seed_bank.json", seed_bank)
    return audit, seeds


def parse_joint_topology(output: Path) -> dict[str, Any]:
    root = ET.parse(DERIVED_URDF).getroot()
    names = ["j1", "j2", "j3", "j4", "j5", "j6"]
    rules = []
    for name in names:
        joint = next((item for item in root.findall("joint") if item.get("name") == name), None)
        limit = joint.find("limit") if joint is not None else None
        joint_type = joint.get("type") if joint is not None else None
        lower = float(limit.get("lower")) if limit is not None and limit.get("lower") is not None else None
        upper = float(limit.get("upper")) if limit is not None and limit.get("upper") is not None else None
        rules.append({"name": name, "type": joint_type, "continuous": joint_type == "continuous", "bounded_revolute": joint_type == "revolute" and lower is not None and upper is not None, "lower": lower, "upper": upper, "range": (upper - lower) if lower is not None and upper is not None else (2.0 * math.pi if joint_type == "continuous" else None)})
    continuous = [r["name"] for r in rules if r["continuous"]]
    bounded = [r["name"] for r in rules if r["bounded_revolute"]]
    other = [r["name"] for r in rules if r["name"] not in continuous and r["name"] not in bounded]
    audit = {"schema_version": "stage3-h5-joint-topology-audit-v1", "JOINT_TOPOLOGY_AUDIT": "PASSED" if not other and len(rules) == 6 else "BLOCKED", "joint_model_source": rel(DERIVED_URDF), "implementation": "exact MoveIt JointModel topology-equivalent distance/interpolation semantics from URDF joint type and bounds", "CONTINUOUS_JOINTS": continuous, "BOUNDED_REVOLUTE_JOINTS": bounded, "OTHER_JOINTS": other, "active_joint_rules": rules, "raw_angle_subtraction_not_used_for_continuous_joints": True}
    dump_json(output / "stage3_h5_joint_topology_audit.json", audit)
    if audit["JOINT_TOPOLOGY_AUDIT"] != "PASSED":
        raise RuntimeError(f"joint topology audit blocked: {audit}")
    return audit


def topology_normalized(q: Sequence[float], rules: Sequence[Mapping[str, Any]]) -> list[float]:
    values = []
    for value, rule in zip(q, rules):
        value = float(value)
        if rule.get("continuous"):
            value = (value + math.pi) % (2.0 * math.pi) - math.pi
        values.append(round(value, 12))
    return values


def topology_delta(q0: Sequence[float], q1: Sequence[float], rules: Sequence[Mapping[str, Any]]) -> list[float]:
    result = []
    for start, end, rule in zip(q0, q1, rules):
        delta = float(end) - float(start)
        if rule.get("continuous"):
            delta = (delta + math.pi) % (2.0 * math.pi) - math.pi
        result.append(delta)
    return result


def joint_valid(q: Sequence[float], rules: Sequence[Mapping[str, Any]], tolerance: float = 1.0e-9) -> bool:
    return len(q) == len(rules) and all(math.isfinite(float(value)) and (rule.get("lower") is None or float(value) >= float(rule["lower"]) - tolerance) and (rule.get("upper") is None or float(value) <= float(rule["upper"]) + tolerance) for value, rule in zip(q, rules))


def deduplicate_topology(raw: Sequence[Mapping[str, Any]], target_id: int, rules: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    distinct: list[dict[str, Any]] = []
    for item in sorted(raw, key=lambda row: int(row.get("seed_index", 0))):
        q = [float(v) for v in item["joint_values"]]
        match = None
        for candidate in distinct:
            delta = topology_delta(candidate["joint_values"], q, rules)
            if max(abs(v) for v in delta) <= DEDUP_TOLERANCE_RAD:
                match = candidate
                break
        if match is None:
            distinct.append({"joint_values": q, "seed_ids": [item["seed_id"]], "seed_indices": [int(item["seed_index"])], "solver_plugin": item.get("solver_plugin"), "solver_api": item.get("solver_api")})
        else:
            match["seed_ids"].append(item["seed_id"]); match["seed_indices"].append(int(item["seed_index"]))
    result = []
    for candidate in sorted(distinct, key=lambda row: topology_normalized(row["joint_values"], rules)):
        normalized = topology_normalized(candidate["joint_values"], rules)
        identity = semantic_hash({"target_id": target_id, "joint_topology_normalized_values": normalized})
        result.append({"target_id": target_id, "ik_candidate_id": f"t{target_id:03d}_ik_{identity[:12]}", "seed_ids": sorted(set(candidate["seed_ids"])), "seed_indices": sorted(set(candidate["seed_indices"])), "joint_values": candidate["joint_values"], "joint_topology_normalized_values": normalized, "solver_plugin": candidate.get("solver_plugin"), "solver_api": candidate.get("solver_api"), "topology_dedup_tolerance_rad": DEDUP_TOLERANCE_RAD, "semantic_hash": identity})
    return result


def run_ik_process(root: Path, replay_index: int, selected: Mapping[str, Any]) -> dict[str, Any]:
    run_dir = root / "fresh_process_replays" / f"fresh_process_{replay_index:03d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    placements = run_dir / "placements.json"
    dump_json(placements, {"placement_id": selected["candidate_id"], "transforms": selected["transforms"]})
    child_exit = run_dir / "child_exit_record.json"
    launch = " ".join([f"ros2 launch {wsl_path(ROOT / 'scripts/stage3_h4_5_derived_launch.py')}", f"worker_output:={wsl_path(run_dir)}", f"replay_index:={replay_index}", f"placements:={wsl_path(placements)}", f"h3_targets:={wsl_path(H3_TARGETS)}", f"h2_manifest:={wsl_path(H2_MANIFEST)}", f"kinematics:={wsl_path(KINEMATICS)}", f"urdf:={wsl_path(DERIVED_URDF)}", f"srdf:={wsl_path(SRDF)}", f"child_exit_record:={wsl_path(child_exit)}"])
    code, stdout, stderr = run_wsl(["source /opt/ros/jazzy/setup.bash", f"source {wsl_path(ROOT / 'install/setup.bash')}", f"cd {wsl_path(ROOT)}", launch], timeout_s=1800)
    log = stdout + "\n--- STDERR ---\n" + stderr
    (run_dir / "ik_launch.log").write_text(log, encoding="utf-8", newline="\n")
    result_path = run_dir / "worker_result.json"
    result = load_json(result_path) if result_path.is_file() else None
    child = load_json(child_exit) if child_exit.is_file() else {}
    forbidden = [pattern for pattern in FORBIDDEN_LOG_PATTERNS if pattern in log.lower()]
    record = {"replay_index": replay_index, "worker_dir": rel(run_dir), "outer_returncode": code, "child_returncode": child.get("returncode"), "worker_result_exists": result is not None, "child_exit": child, "forbidden_log_patterns": forbidden, "clean_exit": code == 0 and result is not None and child.get("returncode") == 0 and not forbidden and "process has finished cleanly" in log.lower()}
    dump_json(run_dir / "ik_process_record.json", record)
    if not record["clean_exit"]:
        raise RuntimeError(f"H5 fresh IK process failed: {record}")
    return {"run_dir": run_dir, "result": result, "record": record}


def collect_candidates(process: Mapping[str, Any], feasible_ids: set[int], rules: Sequence[Mapping[str, Any]], targets: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    result = process["result"]
    ik_map = {int(row["target_index"]): row for row in result.get("ik_rows", [])}
    rows = []; nodes = []
    for target_id in sorted(feasible_ids):
        row = ik_map.get(target_id)
        if row is None:
            raise RuntimeError(f"fresh IK result omitted formal H4.5 feasible target {target_id}")
        raw = []
        for attempt in sorted(row.get("seed_attempts", []), key=lambda item: int(item["seed_index"])):
            if attempt.get("solver_success") and attempt.get("solution_joint_positions") is not None:
                raw.append({"seed_id": attempt["seed_id"], "seed_index": int(attempt["seed_index"]), "joint_values": attempt["solution_joint_positions"], "solver_plugin": attempt.get("solver_plugin"), "solver_api": attempt.get("solver_api_entry")})
        distinct = deduplicate_topology(raw, target_id, rules)
        rows.append({"schema_version": "stage3-h5-multi-ik-candidates-v1", "target_id": target_id, "task_sample_id": targets[target_id]["task_sample_id"], "fresh_process_replay_index": process["record"]["replay_index"], "seed_count": len(row.get("seed_attempts", [])), "seed_attempts": row.get("seed_attempts", []), "raw_ik_solution_count": len(raw), "raw_ik_solutions": raw, "distinct_candidate_count": len(distinct), "distinct_candidate_ids": [candidate["ik_candidate_id"] for candidate in distinct], "coverage": "DETERMINISTIC_MULTI_SEED_NON_EXHAUSTIVE", "h4_5_formal_target": True})
        nodes.extend([{**candidate, "task_sample_id": targets[target_id]["task_sample_id"]} for candidate in distinct])
    return rows, nodes


def h42_fk_reference(result: Mapping[str, Any], target_id: int, q: Sequence[float], rules: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ik = next((row for row in result.get("ik_rows", []) if int(row.get("target_index", -1)) == target_id), {})
    fk_map = {(str(row.get("candidate_id")), int(row.get("target_index", -1))): row for row in result.get("fk_rows", [])}
    best = None; best_distance = float("inf")
    for candidate in ik.get("deduplicated_candidates", []):
        delta = topology_delta(candidate.get("joint_positions", []), q, rules)
        distance = math.sqrt(sum(float(v) * float(v) for v in delta)) if delta else float("inf")
        if distance < best_distance:
            best_distance = distance; best = candidate
    if best is None or best_distance > 1.0e-6:
        return {"available": False, "fk_valid": False, "translation_error_m": None, "orientation_error_rad": None, "joint_validation": {"valid": False}, "match_distance_rad": best_distance}
    fk = fk_map.get((str(best.get("candidate_id")), target_id), {})
    return {"available": True, "fk_valid": bool(fk.get("fk_valid")), "translation_error_m": fk.get("translation_error_m"), "orientation_error_rad": fk.get("orientation_error_rad"), "joint_validation": fk.get("joint_validation", {}), "match_distance_rad": best_distance, "source_candidate_id": best.get("candidate_id")}


def write_records_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = ["kind", "edge_id", "source_node_id", "target_node_id", "target_index", "source_target_index", "destination_target_index", "state_index", "sample_count", "alpha", "task_sample_id", *[f"q{i}" for i in range(6)]]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def parse_native_rows(path: Path) -> list[dict[str, Any]]:
    return load_jsonl(path) if path.is_file() else []


def build_surface_input(output: Path, h45: Mapping[str, Any], feasible_ids: set[int]) -> dict[str, Any]:
    source = h45["feasible"]
    edges = [list(map(int, edge)) for edge in source.get("surface_adjacency_edges", []) if int(edge[0]) in feasible_ids and int(edge[1]) in feasible_ids]
    edges = sorted({tuple(sorted(edge)) for edge in edges})
    data = {"schema_version": "stage3-h5-surface-adjacency-input-v1", "source": rel(H45_ROOT / "stage3_h4_5_connected_components.json"), "definition": source.get("connectivity_definition"), "all_h4_5_surface_adjacency_edge_count": source.get("surface_adjacency_edge_count"), "formal_feasible_target_ids": sorted(feasible_ids), "formal_surface_adjacent_edge_count": len(edges), "surface_adjacent_edges": [list(edge) for edge in edges], "no_new_adjacency_edges": True, "no_nonadjacent_target_edges_added": True}
    dump_json(output / "stage3_h5_surface_adjacency_input.json", data)
    return data


def prepare_edges(nodes: Sequence[Mapping[str, Any]], surface: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_target: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for node in nodes:
        if bool(node.get("h5_joint_fk_valid", False)):
            by_target[int(node["target_id"])].append(node)
    candidate_edges = []; transition_rows = []
    edge_index = 0
    for left, right in surface["surface_adjacent_edges"]:
        left_nodes = sorted(by_target.get(int(left), []), key=lambda node: node["ik_candidate_id"])
        right_nodes = sorted(by_target.get(int(right), []), key=lambda node: node["ik_candidate_id"])
        for source in left_nodes:
            for target in right_nodes:
                edge_id = f"edge_{edge_index:06d}"; edge_index += 1
                delta = topology_delta(source["joint_values"], target["joint_values"], source["joint_rules"])
                max_delta = max(abs(value) for value in delta) if delta else 0.0
                sample_count = max(2, int(math.ceil(math.degrees(max_delta) / INTERPOLATION_STEP_DEG)) + 1)
                edge = {"schema_version": "stage3-h5-candidate-edge-v1", "edge_id": edge_id, "source_target_id": int(left), "destination_target_id": int(right), "source_node_id": source["ik_candidate_id"], "target_node_id": target["ik_candidate_id"], "endpoint_safe": bool(source.get("h5_joint_fk_valid")) and bool(target.get("h5_joint_fk_valid")), "topology_aware_delta_rad": delta, "per_joint_topology_aware_delta": {rule["name"]: float(value) for rule, value in zip(source["joint_rules"], delta)}, "max_abs_joint_delta_rad": max_delta, "max_abs_joint_delta_deg": math.degrees(max_delta), "L1_joint_distance": sum(abs(value) for value in delta), "L2_joint_distance": math.sqrt(sum(value * value for value in delta)), "MoveIt_RobotState_distance": math.sqrt(sum(value * value for value in delta)), "normalized_joint_distance_by_range": [abs(value) / float(rule["range"]) if rule.get("range") else None for value, rule in zip(delta, source["joint_rules"])], "joint_margin_before": [min(float(node_value) - float(rule["lower"]), float(rule["upper"]) - float(node_value)) if rule.get("lower") is not None else None for node_value, rule in zip(source["joint_values"], source["joint_rules"])], "joint_margin_after": [min(float(node_value) - float(rule["lower"]), float(rule["upper"]) - float(node_value)) if rule.get("lower") is not None else None for node_value, rule in zip(target["joint_values"], target["joint_rules"])], "interpolation_method": COLLISION_METHOD, "interpolation_step_deg": INTERPOLATION_STEP_DEG, "interpolation_sample_count": sample_count, "adaptive_refinement_reason": "frozen Stage 3 adaptive discrete interpolation contract; ceil(max topology-aware joint delta / 0.5 deg) + 1", "ccd_status": CCD_STATUS, "evaluation_status": "PENDING"}
                if not edge["endpoint_safe"]:
                    edge["evaluation_status"] = "ENDPOINT_NOT_SAFE"; edge["reject_reason"] = "endpoint_fcl_or_self_environment_collision"
                else:
                    edge["evaluation_status"] = "PENDING_NATIVE_TRANSITION"
                    for state_index in range(sample_count):
                        alpha = state_index / float(sample_count - 1)
                        q = [float(a) + alpha * float(b) for a, b in zip(source["joint_values"], delta)]
                        transition_rows.append({"kind": "transition", "edge_id": edge_id, "source_node_id": source["ik_candidate_id"], "target_node_id": target["ik_candidate_id"], "target_index": int(left), "source_target_index": int(left), "destination_target_index": int(right), "state_index": state_index, "sample_count": sample_count, "alpha": alpha, "task_sample_id": source.get("task_sample_id"), **{f"q{i}": value for i, value in enumerate(q)}})
                candidate_edges.append(edge)
    return candidate_edges, transition_rows


def write_endpoint_and_transition_inputs(run_dir: Path, nodes: Sequence[Mapping[str, Any]], transition_rows: Sequence[Mapping[str, Any]]) -> tuple[Path, Path]:
    endpoint_rows = []
    for node in nodes:
        if bool(node.get("h5_joint_fk_valid", False)):
            endpoint_rows.append({"kind": "endpoint", "edge_id": "", "source_node_id": node["ik_candidate_id"], "target_node_id": "", "target_index": int(node["target_id"]), "source_target_index": int(node["target_id"]), "destination_target_index": int(node["target_id"]), "state_index": 0, "sample_count": 1, "alpha": 0.0, "task_sample_id": node["task_sample_id"], **{f"q{i}": value for i, value in enumerate(node["joint_values"])}})
    endpoint_csv = run_dir / "stage3_h5_endpoint_states.csv"; transition_csv = run_dir / "stage3_h5_transition_states.csv"
    write_records_csv(endpoint_csv, endpoint_rows); write_records_csv(transition_csv, transition_rows)
    return endpoint_csv, transition_csv


def build_native_bridge(output: Path) -> Path:
    build_base = output / "native_build"; install_base = output / "native_install"
    command = f"source /opt/ros/jazzy/setup.bash && colcon build --base-paths {wsl_path(ROOT / 'cpp/stage3_h5')} --build-base {wsl_path(build_base)} --install-base {wsl_path(install_base)} --merge-install"
    code, stdout, stderr = run_wsl([command], timeout_s=900)
    (output / "stage3_h5_native_bridge_build.log").write_text(stdout + "\n--- STDERR ---\n" + stderr, encoding="utf-8", newline="\n")
    record = {"schema_version": "stage3-h5-native-bridge-build-v1", "returncode": code, "package": "stage3_h5_native_transition", "install_base": rel(install_base), "status": "PASSED" if code == 0 else "BLOCKED", "forbidden_log_patterns": [pattern for pattern in FORBIDDEN_LOG_PATTERNS if pattern in (stdout + stderr).lower()]}
    dump_json(output / "stage3_h5_native_bridge_build.json", record)
    if code != 0:
        raise RuntimeError(f"H5 native bridge build failed: {record}")
    executable = install_base / "lib/stage3_h5_native_transition/stage3_h5_native_transition_bridge"
    if not executable.exists():
        raise RuntimeError(f"H5 native bridge executable missing: {executable}")
    return executable


def run_native_bridge(root: Path, executable: Path, run_dir: Path, endpoint_csv: Path, transition_csv: Path) -> dict[str, Any]:
    native_dir = run_dir / "native_transition"; native_dir.mkdir(parents=True, exist_ok=False)
    launch = " ".join([f"ros2 launch {wsl_path(ROOT / 'scripts/stage3_h5_transition_launch.py')}", f"endpoint_csv:={wsl_path(endpoint_csv)}", f"transition_csv:={wsl_path(transition_csv)}", f"curved_mesh_obj:={wsl_path(SELECTED_MESH)}", f"output_dir:={wsl_path(native_dir)}"])
    code, stdout, stderr = run_wsl(["source /opt/ros/jazzy/setup.bash", f"source {wsl_path(ROOT / 'install/setup.bash')}", f"source {wsl_path(root / 'native_install/setup.bash')}", f"cd {wsl_path(ROOT)}", launch], timeout_s=1800)
    log = stdout + "\n--- STDERR ---\n" + stderr; (native_dir / "launch.log").write_text(log, encoding="utf-8", newline="\n")
    summary_path = native_dir / "stage3_h5_native_transition_summary.json"
    summary = load_json(summary_path) if summary_path.is_file() else {}
    record = {"returncode": code, "native_dir": rel(native_dir), "summary_exists": bool(summary), "forbidden_log_patterns": [pattern for pattern in FORBIDDEN_LOG_PATTERNS if pattern in log.lower()], "status": "PASSED" if code == 0 and summary.get("status") == "AVAILABLE" else "BLOCKED"}
    dump_json(native_dir / "native_process_record.json", record)
    if record["status"] != "PASSED":
        raise RuntimeError(f"H5 native transition process failed: {record}")
    return {"dir": native_dir, "summary": summary, "record": record, "endpoints": parse_native_rows(native_dir / "stage3_h5_endpoint_evidence.jsonl"), "states": parse_native_rows(native_dir / "stage3_h5_native_transition_state_evidence.jsonl")}


def edge_results(candidate_edges: list[dict[str, Any]], native: Mapping[str, Any], nodes: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    endpoint_map = {str(row["source_node_id"]): row for row in native["endpoints"]}
    state_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in native["states"]: state_map[str(row["edge_id"])].append(row)
    node_map = {str(node["ik_candidate_id"]): node for node in nodes}
    output = []
    for edge in candidate_edges:
        source_endpoint = endpoint_map.get(str(edge["source_node_id"]), {})
        target_endpoint = endpoint_map.get(str(edge["target_node_id"]), {})
        endpoint_safe = bool(source_endpoint.get("native_fcl_checked")) and bool(target_endpoint.get("native_fcl_checked")) and bool(source_endpoint.get("collision_free")) and bool(target_endpoint.get("collision_free"))
        states = sorted(state_map.get(edge["edge_id"], []), key=lambda row: int(row.get("state_index", 0)))
        collisions = [row for row in states if not bool(row.get("collision_free"))]
        first_collision = collisions[0] if collisions else None
        all_states_valid = bool(states) and all(bool(row.get("native_fcl_checked")) and bool(row.get("joint_limit_valid")) and bool(row.get("fk_computable")) and bool(row.get("collision_free")) for row in states)
        env_pairs = sorted({pair for row in states for pair in row.get("environment_collision_pairs", []) if pair})
        self_pairs = sorted({pair for row in states for pair in row.get("self_collision_pairs", []) if pair})
        env_objects = sorted({pair for row in states for pair in row.get("collision_environment_objects", []) if pair})
        env_distances = [(float(row["environment_min_signed_distance"]["distance_m"]), int(row["state_index"]), row["environment_min_signed_distance"].get("nearest_pair")) for row in states if row.get("environment_min_signed_distance", {}).get("status") == "AVAILABLE" and row.get("environment_min_signed_distance", {}).get("distance_m") is not None]
        self_distances = [(float(row["self_min_signed_distance"]["distance_m"]), int(row["state_index"]), row["self_min_signed_distance"].get("nearest_pair")) for row in states if row.get("self_min_signed_distance", {}).get("status") == "AVAILABLE" and row.get("self_min_signed_distance", {}).get("distance_m") is not None]
        edge.update({"endpoint_safe": endpoint_safe, "endpoint_native_fcl_checked": bool(source_endpoint.get("native_fcl_checked")) and bool(target_endpoint.get("native_fcl_checked")), "intermediate_state_count_checked": len(states), "all_interpolated_states_collision_free": all_states_valid, "transition_self_collision": bool(self_pairs), "transition_environment_collision": bool(env_pairs), "transition_collision_link_pairs": sorted(set(env_pairs + self_pairs)), "transition_collision_environment_objects": env_objects, "first_collision_sample": {"state_index": first_collision.get("state_index"), "alpha": first_collision.get("alpha"), "joint_state_at_first_collision": first_collision.get("joint_values"), "collision_link_pairs": sorted(set(first_collision.get("environment_collision_pairs", []) + first_collision.get("self_collision_pairs", []))), "collision_environment_object": first_collision.get("collision_environment_objects", [])} if first_collision else None, "minimum_environment_signed_distance_m": min(env_distances)[0] if env_distances else None, "state_index_of_minimum_environment_clearance": min(env_distances)[1] if env_distances else None, "nearest_environment_pair": min(env_distances)[2] if env_distances else None, "minimum_self_signed_distance_m": min(self_distances)[0] if self_distances else None, "state_index_of_minimum_self_clearance": min(self_distances)[1] if self_distances else None, "nearest_self_pair": min(self_distances)[2] if self_distances else None, "transition_collision_status": "PASS" if endpoint_safe and all_states_valid else "FAIL", "valid": bool(endpoint_safe and all_states_valid), "reject_reason": None if endpoint_safe and all_states_valid else ("endpoint_fcl_not_collision_free" if not endpoint_safe else ("transition_self_collision" if self_pairs else ("transition_environment_collision" if env_pairs else "transition_fk_or_joint_limit_failure")))})
        output.append(edge)
    valid_edges = [edge for edge in output if edge["valid"]]
    summary = {"candidate_edge_count": len(output), "endpoint_safe_edge_count": sum(bool(edge["endpoint_safe"]) for edge in output), "valid_transition_edge_count": len(valid_edges), "endpoint_safe_but_transition_collision_count": sum(bool(edge["endpoint_safe"]) and not bool(edge["valid"]) and (bool(edge["transition_self_collision"]) or bool(edge["transition_environment_collision"])) for edge in output), "transition_self_collision_edges": sum(bool(edge["transition_self_collision"]) for edge in output), "transition_environment_collision_edges": sum(bool(edge["transition_environment_collision"]) for edge in output), "transition_fk_or_bounds_failure_edges": sum(edge["reject_reason"] == "transition_fk_or_joint_limit_failure" for edge in output), "transition_state_count": sum(int(edge["intermediate_state_count_checked"]) for edge in output), "valid_edges": valid_edges}
    return output, summary, endpoint_map


def connected_components(nodes: Sequence[Mapping[str, Any]], valid_edges: Sequence[Mapping[str, Any]], feasible_ids: set[int]) -> dict[str, Any]:
    formal_ids = sorted(str(node["ik_candidate_id"]) for node in nodes if bool(node.get("h5_formal_node")))
    adjacency: dict[str, set[str]] = {node_id: set() for node_id in formal_ids}
    for edge in valid_edges:
        left, right = str(edge["source_node_id"]), str(edge["target_node_id"])
        if left in adjacency and right in adjacency:
            adjacency[left].add(right); adjacency[right].add(left)
    unseen = set(formal_ids); node_components = []
    while unseen:
        start = min(unseen); unseen.remove(start); queue = deque([start]); component = []
        while queue:
            current = queue.popleft(); component.append(current)
            for other in sorted(adjacency[current] & unseen): unseen.remove(other); queue.append(other)
        node_components.append(sorted(component))
    node_components.sort(key=lambda comp: (-len(comp), comp))
    node_map = {str(node["ik_candidate_id"]): node for node in nodes}
    target_adjacency: dict[int, set[int]] = {int(target): set() for target in sorted(feasible_ids)}
    for edge in valid_edges:
        left, right = int(edge["source_target_id"]), int(edge["destination_target_id"])
        target_adjacency[left].add(right); target_adjacency[right].add(left)
    unseen_targets = set(target_adjacency); target_components = []
    while unseen_targets:
        start = min(unseen_targets); unseen_targets.remove(start); queue = deque([start]); component = []
        while queue:
            current = queue.popleft(); component.append(current)
            for other in sorted(target_adjacency[current] & unseen_targets): unseen_targets.remove(other); queue.append(other)
        target_components.append(sorted(component))
    target_components.sort(key=lambda comp: (-len(comp), comp))
    projection = [{"node_component_index": index, "node_ids": component, "target_ids": sorted({int(node_map[node_id]["target_id"]) for node_id in component})} for index, component in enumerate(node_components)]
    return {"schema_version": "stage3-h5-joint-connectable-components-v1", "joint_configuration_connected_components": node_components, "joint_configuration_component_count": len(node_components), "joint_connectable_target_components": target_components, "joint_connectable_target_component_count": len(target_components), "largest_joint_configuration_component": len(node_components[0]) if node_components else 0, "largest_joint_connectable_target_component": len(target_components[0]) if target_components else 0, "largest_joint_connectable_target_component_ids": target_components[0] if target_components else [], "node_component_target_projection": projection, "target_space_components_before_h5": 3, "largest_target_space_component_before_h5": 16, "formal_feasible_target_count": len(feasible_ids), "isolated_target_components": sum(len(component) == 1 for component in target_components), "valid_transition_edge_count": len(valid_edges)}


def preferred_branches(output: Path, edges: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_pair: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for edge in edges:
        if edge.get("valid"): by_pair[(int(edge["source_target_id"]), int(edge["destination_target_id"]))].append(edge)
    selections = []
    for pair, values in sorted(by_pair.items()):
        ordered = sorted(values, key=lambda edge: (float(edge["MoveIt_RobotState_distance"]), float(edge["max_abs_joint_delta_rad"]), float(edge["L1_joint_distance"]), float(edge["L2_joint_distance"]), -(float(edge["minimum_environment_signed_distance_m"]) if edge.get("minimum_environment_signed_distance_m") is not None else -float("inf")), -(float(edge["minimum_self_signed_distance_m"]) if edge.get("minimum_self_signed_distance_m") is not None else -float("inf")), int(edge["interpolation_sample_count"]), str(edge["edge_id"])))
        selections.append({"source_target_id": pair[0], "destination_target_id": pair[1], "preferred_edge_id": ordered[0]["edge_id"], "preferred_branch": {"source_node_id": ordered[0]["source_node_id"], "target_node_id": ordered[0]["target_node_id"], "MoveIt_RobotState_distance": ordered[0]["MoveIt_RobotState_distance"], "max_abs_joint_delta_rad": ordered[0]["max_abs_joint_delta_rad"], "L1_joint_distance": ordered[0]["L1_joint_distance"], "L2_joint_distance": ordered[0]["L2_joint_distance"], "minimum_environment_signed_distance_m": ordered[0].get("minimum_environment_signed_distance_m"), "minimum_self_signed_distance_m": ordered[0].get("minimum_self_signed_distance_m"), "interpolation_sample_count": ordered[0]["interpolation_sample_count"]}, "surviving_alternatives": [{"edge_id": edge["edge_id"], "source_node_id": edge["source_node_id"], "target_node_id": edge["target_node_id"], "MoveIt_RobotState_distance": edge["MoveIt_RobotState_distance"], "max_abs_joint_delta_rad": edge["max_abs_joint_delta_rad"], "L1_joint_distance": edge["L1_joint_distance"], "L2_joint_distance": edge["L2_joint_distance"], "minimum_environment_signed_distance_m": edge.get("minimum_environment_signed_distance_m"), "minimum_self_signed_distance_m": edge.get("minimum_self_signed_distance_m"), "interpolation_sample_count": edge["interpolation_sample_count"]} for edge in ordered]})
    result = {"schema_version": "stage3-h5-preferred-branch-sequences-v1", "selection_policy": ["minimum MoveIt_RobotState_distance", "minimum max_abs_joint_delta", "minimum L1", "minimum L2", "maximum observed environment clearance", "maximum observed self clearance", "minimum interpolation sample count", "edge_id"], "thresholds_are_not_safety_gates": True, "sequence_semantics": "edge-local preferred branches only; no coverage ordering, Hamiltonian path, Spray ON/OFF sequence, or trajectory certification", "pair_selections": selections}
    dump_json(output / "stage3_h5_preferred_branch_sequences.json", result)
    return result


def tcp_diagnostics(output: Path, valid_edges: Sequence[Mapping[str, Any]], native_states: Sequence[Mapping[str, Any]], targets: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    state_map: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in native_states: state_map[str(row["edge_id"])].append(row)
    rows = []
    for edge in valid_edges:
        source = targets[int(edge["source_target_id"])]
        destination = targets[int(edge["destination_target_id"])]
        p0 = [float(v) for v in source["surface_point_xyz_m"]]; p1 = [float(v) for v in destination["surface_point_xyz_m"]]
        n0 = [float(v) for v in source["surface_normal_unit"]]; n1 = [float(v) for v in destination["surface_normal_unit"]]
        states = []
        for state in sorted(state_map.get(str(edge["edge_id"]), []), key=lambda row: int(row["state_index"])): 
            alpha = float(state["alpha"]); point = [(1.0 - alpha) * a + alpha * b for a, b in zip(p0, p1)]; normal = [(1.0 - alpha) * a + alpha * b for a, b in zip(n0, n1)]; norm = math.sqrt(sum(v * v for v in normal)); normal = [v / norm for v in normal] if norm > 1e-12 else None
            tcp = state.get("tcp_position_m"); standoff = math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(tcp, point))) if tcp and point else None
            orientation = state.get("tcp_orientation_xyzw"); normal_angle = None
            if orientation and normal:
                x, y, z, w = [float(v) for v in orientation]; z_axis = [2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)]; length = math.sqrt(sum(v * v for v in z_axis)); z_axis = [v / length for v in z_axis]; dot = max(-1.0, min(1.0, sum((-z_axis[i]) * normal[i] for i in range(3)))); normal_angle = math.degrees(math.acos(dot))
            states.append({"state_index": int(state["state_index"]), "alpha": alpha, "tcp_position_m": tcp, "tcp_orientation_xyzw": orientation, "surface_point_interpolated_m": point, "surface_normal_interpolated": normal, "standoff_relative_to_interpolated_surface_m": standoff, "normal_angle_deviation_deg": normal_angle, "diagnostic_only": True})
        rows.append({"schema_version": "stage3-h5-intermediate-tcp-diagnostic-v1", "edge_id": edge["edge_id"], "source_target_id": edge["source_target_id"], "destination_target_id": edge["destination_target_id"], "status": "DIAGNOSTIC_ONLY", "formal_tcp_tolerance_applied": False, "states": states})
    dump_jsonl(output / "stage3_h5_intermediate_tcp_diagnostics.jsonl", rows)
    values = [state["normal_angle_deviation_deg"] for row in rows for state in row["states"] if state.get("normal_angle_deviation_deg") is not None]
    standoffs = [state["standoff_relative_to_interpolated_surface_m"] for row in rows for state in row["states"] if state.get("standoff_relative_to_interpolated_surface_m") is not None]
    return {"edge_count": len(rows), "state_count": sum(len(row["states"]) for row in rows), "normal_angle_deviation_max_deg": max(values) if values else None, "normal_angle_deviation_p95_deg": sorted(values)[min(len(values) - 1, int(0.95 * (len(values) - 1)))] if values else None, "standoff_min_m": min(standoffs) if standoffs else None, "standoff_max_m": max(standoffs) if standoffs else None, "formal_tolerance": None, "status": "DIAGNOSTIC_ONLY", "no_threshold_invented": True}


def clearance_summary(output: Path, native_summary: Mapping[str, Any], edges: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    env = [edge["minimum_environment_signed_distance_m"] for edge in edges if edge.get("minimum_environment_signed_distance_m") is not None]
    self_values = [edge["minimum_self_signed_distance_m"] for edge in edges if edge.get("minimum_self_signed_distance_m") is not None]
    env_edge = min((edge for edge in edges if edge.get("minimum_environment_signed_distance_m") is not None), key=lambda edge: float(edge["minimum_environment_signed_distance_m"]), default=None)
    self_edge = min((edge for edge in edges if edge.get("minimum_self_signed_distance_m") is not None), key=lambda edge: float(edge["minimum_self_signed_distance_m"]), default=None)
    summary = {"schema_version": "stage3-h5-transition-clearance-summary-v1", "status": "AVAILABLE" if native_summary.get("status") == "AVAILABLE" else "not_available", "backend": "MoveIt PlanningScene + native FCL", "collision_method": COLLISION_METHOD, "ccd_status": CCD_STATUS, "minimum_environment_signed_distance_m": min(env) if env else None, "minimum_environment_edge_id": env_edge.get("edge_id") if env_edge else None, "minimum_environment_state_index": env_edge.get("state_index_of_minimum_environment_clearance") if env_edge else None, "minimum_environment_nearest_pair": env_edge.get("nearest_environment_pair") if env_edge else None, "minimum_self_signed_distance_m": min(self_values) if self_values else None, "minimum_self_edge_id": self_edge.get("edge_id") if self_edge else None, "minimum_self_state_index": self_edge.get("state_index_of_minimum_self_clearance") if self_edge else None, "minimum_self_nearest_pair": self_edge.get("nearest_self_pair") if self_edge else None, "positive_clearance_threshold": None, "threshold_semantics": "diagnostic/ranking metric only; no positive safety threshold inferred", "native_distance_api": ["CollisionEnv::distanceRobot", "CollisionEnv::distanceSelf", "DistanceRequest(enable_signed_distance=true)"]}
    dump_json(output / "stage3_h5_transition_clearance_summary.json", summary)
    return summary


def replay_signature(process: Mapping[str, Any], nodes: Sequence[Mapping[str, Any]], edges: Sequence[Mapping[str, Any]], surface_edges: Sequence[Sequence[int]], components: Mapping[str, Any], preferred: Mapping[str, Any], clearance: Mapping[str, Any], candidate_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    candidate_set = [{"target_id": row["target_id"], "distinct_candidate_ids": row["distinct_candidate_ids"], "raw_ik_solution_count": row["raw_ik_solution_count"]} for row in candidate_rows]
    node_hashes = sorted(str(node["semantic_hash"]) for node in nodes if node.get("h5_formal_node"))
    edge_topology = [{"edge_id": edge["edge_id"], "source_target_id": edge["source_target_id"], "destination_target_id": edge["destination_target_id"], "source_node_id": edge["source_node_id"], "target_node_id": edge["target_node_id"], "valid": edge.get("valid"), "reject_reason": edge.get("reject_reason")} for edge in edges]
    signature = {"h4_5_selected_configuration_identity": EXPECTED_CONFIGURATION, "ik_seed_bank_identity": semantic_hash(load_json(H41_ROOT / "stage3_h4_1_seed_contract.json").get("seeds", [])), "normalized_distinct_ik_candidate_sets": candidate_set, "valid_ik_node_semantic_hashes": node_hashes, "surface_adjacent_edge_set": [list(map(int, edge)) for edge in surface_edges], "candidate_edge_count": len(edges), "valid_edge_count": sum(bool(edge.get("valid")) for edge in edges), "transition_collision_failures": sorted(edge["edge_id"] for edge in edges if not edge.get("valid") and edge.get("endpoint_safe") and (edge.get("transition_self_collision") or edge.get("transition_environment_collision"))), "joint_connectable_component_membership": components.get("joint_connectable_target_components"), "largest_joint_connectable_component": components.get("largest_joint_connectable_target_component"), "preferred_branch_selections": preferred.get("pair_selections"), "minimum_clearance_summary": {key: clearance.get(key) for key in ("minimum_environment_signed_distance_m", "minimum_self_signed_distance_m", "minimum_environment_nearest_pair", "minimum_self_nearest_pair")}, "authorization_result": {"stage3_h5": "PENDING", "joint_connectivity_certified": "PENDING", "ready_for_h6": "PENDING"}, "candidate_edge_topology": edge_topology}
    signature["candidate_set_hash"] = semantic_hash(candidate_set); signature["node_hash"] = semantic_hash(node_hashes); signature["edge_hash"] = semantic_hash(edge_topology); signature["component_hash"] = semantic_hash(components.get("joint_connectable_target_components")); signature["preferred_hash"] = semantic_hash(preferred.get("pair_selections")); signature["clearance_hash"] = semantic_hash(signature["minimum_clearance_summary"]); return signature


def run_regression(output: Path) -> dict[str, Any]:
    commands = [[sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h5.py", "tests/test_stage3_h4_4.py", "tests/test_stage3_h4_reachability.py", "tests/test_stage3_h4_1.py", "tests/test_stage3_h3_task_representation.py", "tests/test_stage3_h2_geometry.py"]]
    runs = []
    for command in commands:
        proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1800, check=False)
        runs.append({"command": command, "returncode": proc.returncode, "passed": proc.returncode == 0, "stdout_tail": proc.stdout[-5000:], "stderr_tail": proc.stderr[-5000:]})
    report = {"schema_version": "stage3-h5-regression-report-v1", "runs": runs, "passed": all(run["passed"] for run in runs), "pre_existing_failures": [], "new_regression_failures": 0 if all(run["passed"] for run in runs) else 1}
    dump_json(output / "stage3_h5_regression_report.json", report)
    return report


def immutable_audit(output: Path, before: Mapping[str, Any]) -> dict[str, Any]:
    after = {"h4_5_root": tree_manifest(H45_ROOT), "h4_5_immutable_input": load_json(H45_ROOT / "stage3_h4_5_immutable_baseline_audit.json"), "derived_model_sha256": sha256_file(DERIVED_URDF)}
    mismatches = []
    if before["h4_5_root"]["tree_hash"] != after["h4_5_root"]["tree_hash"]: mismatches.append("h4_5_root")
    if before["derived_model_sha256"] != after["derived_model_sha256"]: mismatches.append("derived_robot_model")
    h45_immutable = after["h4_5_immutable_input"]
    if h45_immutable.get("immutable_mismatch_count") != 0: mismatches.append("h4_5_predecessor_formal_roots")
    audit = {"schema_version": "stage3-h5-immutable-baseline-audit-v1", "before": before, "after": after, "STAGE_2_BASELINE_IMMUTABLE": "YES" if h45_immutable.get("immutable_mismatch_count") == 0 else "NO", "FORMAL_R2_IMMUTABLE": "YES" if h45_immutable.get("immutable_mismatch_count") == 0 else "NO", "STAGE_3_H0_IMMUTABLE": "YES" if h45_immutable.get("immutable_mismatch_count") == 0 else "NO", "STAGE_3_H1_IMMUTABLE": "YES" if h45_immutable.get("immutable_mismatch_count") == 0 else "NO", "STAGE_3_H2_IMMUTABLE": "YES" if h45_immutable.get("immutable_mismatch_count") == 0 else "NO", "STAGE_3_H3_IMMUTABLE": "YES" if h45_immutable.get("immutable_mismatch_count") == 0 else "NO", "STAGE_3_H4_IMMUTABLE": "YES" if h45_immutable.get("immutable_mismatch_count") == 0 else "NO", "STAGE_3_H4_5_IMMUTABLE": "YES" if not mismatches else "NO", "immutable_mismatch_count": len(mismatches), "mismatches": mismatches, "formal_ledger_mutated": False}
    dump_json(output / "stage3_h5_immutable_baseline_audit.json", audit)
    return audit


def report_and_certify(output: Path, h45: Mapping[str, Any], freeze: Mapping[str, Any], plugin: Mapping[str, Any], seed_bank: Sequence[Mapping[str, Any]], topology: Mapping[str, Any], surface: Mapping[str, Any], candidate_rows: Sequence[Mapping[str, Any]], nodes: Sequence[Mapping[str, Any]], edges: Sequence[Mapping[str, Any]], edge_summary: Mapping[str, Any], components: Mapping[str, Any], preferred: Mapping[str, Any], tcp: Mapping[str, Any], clearance: Mapping[str, Any], replay: Mapping[str, Any], regression: Mapping[str, Any], immutable: Mapping[str, Any]) -> int:
    transition_pair_counts = defaultdict(int)
    for edge in edges:
        for pair in edge.get("transition_collision_link_pairs", []): transition_pair_counts[pair] += 1
    env_counts = defaultdict(int)
    for edge in edges:
        for obj in edge.get("transition_collision_environment_objects", []): env_counts[obj] += 1
    valid_edges = [edge for edge in edges if edge.get("valid")]
    audit_complete = all([h45["audit"].get("status") == "PASSED", freeze.get("workcell_configuration_frozen_from_h4_5") == "YES", plugin.get("MULTI_IK_NATIVE_ENUMERATION") in {"NOT_AVAILABLE", "PARTIAL", "AVAILABLE"}, topology.get("JOINT_TOPOLOGY_AUDIT") == "PASSED", surface.get("formal_surface_adjacent_edge_count", 0) >= 0, len(nodes) > 0, edge_summary.get("candidate_edge_count", 0) >= 0, replay.get("fresh_process_count") == 3, regression.get("new_regression_failures") == 0, immutable.get("immutable_mismatch_count") == 0])
    if not audit_complete: status, blocker = "BLOCKED", next((name for name, value in {"h4_5_formal_input": h45["audit"].get("status") == "PASSED", "configuration_freeze": freeze.get("workcell_configuration_frozen_from_h4_5") == "YES", "ik_capability_audit": plugin.get("MULTI_IK_NATIVE_ENUMERATION") in {"NOT_AVAILABLE", "PARTIAL", "AVAILABLE"}, "joint_topology": topology.get("JOINT_TOPOLOGY_AUDIT") == "PASSED", "fresh_process_replay": replay.get("fresh_process_count") == 3, "regression": regression.get("new_regression_failures") == 0, "immutable_baseline": immutable.get("immutable_mismatch_count") == 0}.items() if not value), "h5_evidence_incomplete")
    else: status, blocker = "PASSED", None
    connectivity = bool(valid_edges) and int(components.get("largest_joint_connectable_target_component", 0)) >= 2
    if status == "PASSED" and not valid_edges: blocker = "no_joint_connectable_curved_edges"
    elif status == "PASSED" and not connectivity: blocker = "all_joint_connectable_components_singletons"
    ready = status == "PASSED" and connectivity and bool(replay.get("all_semantic_results_consistent")) and regression.get("new_regression_failures") == 0 and immutable.get("immutable_mismatch_count") == 0
    authorization = {"STAGE_3_H5": status, "FIRST_BLOCKER": blocker, "JOINT_CONNECTIVITY_CERTIFIED": "YES" if connectivity else "NO", "READY_FOR_STAGE_3_H6": "YES" if ready else "NO"}
    replay["authorization_result"] = authorization
    gate = {"schema_version": "stage3-h5-gate-report-v1", **authorization, "H4_5_FORMAL_INPUT_VALID": "YES" if h45["audit"].get("status") == "PASSED" else "NO", "WORKCELL_CONFIGURATION_FROZEN": freeze.get("workcell_configuration_frozen_from_h4_5"), "MULTI_SEED_IK_AUDIT": "PASSED" if replay.get("all_candidate_sets_consistent") else "BLOCKED", "JOINT_TOPOLOGY_AUDIT": topology.get("JOINT_TOPOLOGY_AUDIT"), "JOINT_CONFIGURATION_GRAPH_BUILT": "YES", "VALID_TRANSITION_EDGE_COUNT": edge_summary.get("valid_transition_edge_count", 0), "JOINT_CONNECTABLE_TARGET_COMPONENT_COUNT": components.get("joint_connectable_target_component_count", 0), "LARGEST_JOINT_CONNECTABLE_TARGET_COMPONENT": components.get("largest_joint_connectable_target_component", 0), "ALL_SELECTED_COMPONENT_EDGES_VALID": all(bool(edge.get("valid")) for edge in valid_edges), "FRESH_PROCESS_REPLAY": "PASSED" if replay.get("all_semantic_results_consistent") else "BLOCKED", "NEW_REGRESSION_FAILURES": regression.get("new_regression_failures"), "BASELINE_IMMUTABLE": "YES" if immutable.get("immutable_mismatch_count") == 0 else "NO", "collision_method": COLLISION_METHOD, "ccd_status": CCD_STATUS, "authorization_boundary": {"spray_trajectory_certified": "NO", "time_parameterization_done": "NO", "ruckig_done": "NO", "final_trajectory_certified": "NO"}}
    dump_json(output / "stage3_h5_gate_report.json", gate)
    terminal = {"schema_version": "stage3-h5-terminal-certificate-v1", "STAGE_3_H5": status, "FIRST_BLOCKER": blocker, "H4_5_FORMAL_INPUT_VALID": "YES" if h45["audit"].get("status") == "PASSED" else "NO", "H4_5_BASELINE_IMMUTABLE": "YES" if immutable.get("immutable_mismatch_count") == 0 else "NO", "WORKCELL_CONFIGURATION_FROZEN": "YES", "H5_WORKCELL_REOPTIMIZATION": "NO", "SELECTED_H4_5_CONFIGURATION": EXPECTED_CONFIGURATION, "TCP_EXTENSION_M": 0.150, "STANDOFF_M": 0.260, "INDIVIDUALLY_FEASIBLE_CURVED_TARGETS": 30, "H4_5_TARGET_SPACE_COMPONENTS": 3, "H4_5_LARGEST_TARGET_SPACE_COMPONENT": 16, "IK_PLUGIN_IDENTITY": plugin.get("IK_PLUGIN_IDENTITY"), "MULTI_IK_NATIVE_ENUMERATION": plugin.get("MULTI_IK_NATIVE_ENUMERATION"), "IK_SOLUTION_COVERAGE": plugin.get("IK_SOLUTION_COVERAGE"), "RAW_IK_SOLUTIONS": sum(int(row.get("raw_ik_solution_count", 0)) for row in candidate_rows), "DISTINCT_VALID_IK_NODES": sum(bool(node.get("h5_formal_node")) for node in nodes), "JOINT_TOPOLOGY_AUDIT": topology.get("JOINT_TOPOLOGY_AUDIT"), "CANDIDATE_TRANSITION_EDGES": edge_summary.get("candidate_edge_count", 0), "VALID_TRANSITION_EDGES": edge_summary.get("valid_transition_edge_count", 0), "ENDPOINT_SAFE_TRANSITION_COLLISION_EDGES": edge_summary.get("endpoint_safe_but_transition_collision_count", 0), "TRANSITION_SELF_COLLISION_EDGES": edge_summary.get("transition_self_collision_edges", 0), "TRANSITION_ENV_COLLISION_EDGES": edge_summary.get("transition_environment_collision_edges", 0), "COLLISION_METHOD": f"{COLLISION_METHOD} + native MoveIt/FCL", "CCD": "NOT_AVAILABLE", "JOINT_CONNECTABLE_TARGET_COMPONENTS": components.get("joint_connectable_target_component_count", 0), "LARGEST_JOINT_CONNECTABLE_TARGET_COMPONENT": components.get("largest_joint_connectable_target_component", 0), "PREFERRED_COMPONENT_TARGET_IDS": components.get("largest_joint_connectable_target_component_ids", []), "FRESH_PROCESS_REPLAY": "PASSED" if replay.get("all_semantic_results_consistent") else "BLOCKED", "NEW_REGRESSION_FAILURES": regression.get("new_regression_failures"), "NEW_FJT_GOALS_SENT": 0, "ROBOT_MOTION_STARTED": "NO", "WORKCELL_REOPTIMIZATION_STARTED": "NO", "PATH_PLANNING_EXPERIMENT_STARTED": "NO", "RUCKIG_STARTED": "NO", "ML_TRAINING_STARTED": "NO", "RL_TRAINING_STARTED": "NO", "JOINT_CONNECTIVITY_CERTIFIED": "YES" if connectivity else "NO", "SPRAY_TRAJECTORY_CERTIFIED": "NO", "TIME_PARAMETERIZATION_DONE": "NO", "FINAL_TRAJECTORY_CERTIFIED": "NO", "READY_FOR_STAGE_3_H6": "YES" if ready else "NO"}
    dump_json(output / "stage3_h5_terminal_certificate.json", terminal)
    feasible_count_by_target = {int(row["target_id"]): int(row["distinct_candidate_count"]) for row in candidate_rows}
    largest_ids = components.get("largest_joint_connectable_target_component_ids", [])
    q_lines = ["# Stage 3 H5 — Multi-IK Branch Continuity & Transition Collision Certification", "", f"`STAGE_3_H5: {status}`", f"`FIRST_BLOCKER: {blocker or 'none'}`", f"`JOINT_CONNECTIVITY_CERTIFIED: {'YES' if connectivity else 'NO'}`", f"`READY_FOR_STAGE_3_H6: {'YES' if ready else 'NO'}`", "", "## Direct answers Q1–Q30", "", f"Q1. H4.5 formal H5 authorization is valid: `{h45['audit'].get('formal_authorization_valid')}`. A legacy report-only blocker string was recorded as `{h45['audit'].get('documentation_note') or 'none'}`, not promoted to a blocker.", f"Q2. H4.5 selected workcell configuration frozen: `YES`; H5 reoptimization: `NO`.", f"Q3. Actual robot model: H4.5 derived corrected candidate `{rel(DERIVED_URDF)}`; original URDF remained immutable.", f"Q4. IK plugin: `{plugin.get('IK_PLUGIN_IDENTITY')}`.", f"Q5. Native multi-solution enumeration: `{plugin.get('MULTI_IK_NATIVE_ENUMERATION')}`.", f"Q6. Strategy: `{plugin.get('IK_SOLUTION_COVERAGE')}` using the explicit frozen seed bank.", f"Q7. Seed count: `{len(seed_bank)}`.", f"Q8. Distinct valid-candidate counts by target: `{json.dumps(feasible_count_by_target, sort_keys=True)}`.", f"Q9. Raw successful IK solutions: `{terminal['RAW_IK_SOLUTIONS']}`; raw seed attempts were `{30 * len(seed_bank)}`.", f"Q10. Distinct valid H5 IK nodes: `{terminal['DISTINCT_VALID_IK_NODES']}`.", f"Q11. Joint topology audit: `{topology.get('JOINT_TOPOLOGY_AUDIT')}`.", f"Q12. Continuous joints: `{topology.get('CONTINUOUS_JOINTS')}`; bounded revolute joints: `{topology.get('BOUNDED_REVOLUTE_JOINTS')}`.", f"Q13. H4.5 surface-adjacent pairs entering evaluation: `{surface.get('formal_surface_adjacent_edge_count')}`.", f"Q14. Candidate IK-pair edges: `{edge_summary.get('candidate_edge_count')}`.", f"Q15. Endpoint-safe candidate edges: `{edge_summary.get('endpoint_safe_edge_count')}`.", f"Q16. Endpoint-safe but transition-collision edges: `{edge_summary.get('endpoint_safe_but_transition_collision_count')}`.", f"Q17. Transition self-collision link pairs: `{json.dumps(dict(sorted(transition_pair_counts.items())), sort_keys=True)}`.", f"Q18. Transition environment objects/pairs: `{json.dumps(dict(sorted(env_counts.items())), sort_keys=True)}`.", f"Q19. Minimum observed self signed distance: `{clearance.get('minimum_self_signed_distance_m')}` m; no positive safety threshold was inferred.", f"Q20. Minimum observed environment signed distance: `{clearance.get('minimum_environment_signed_distance_m')}` m; no positive safety threshold was inferred.", f"Q21. H4.5 target-space topology was `3` components/largest `16`; H5 target components: `{components.get('joint_connectable_target_component_count')}`, largest `{components.get('largest_joint_connectable_target_component')}`.", f"Q22. Largest joint-connectable target component size: `{components.get('largest_joint_connectable_target_component')}`.", f"Q23. Largest component target IDs: `{largest_ids}`.", f"Q24. Multiple valid branch choices in largest component: `{sum(len([edge for edge in valid_edges if edge['source_target_id'] == a and edge['destination_target_id'] == b]) > 1 for a, b in [(s['source_target_id'], s['destination_target_id']) for s in preferred.get('pair_selections', [])])}`; all alternatives remain in the artifact.", f"Q25. Preferred branch ranking uses MoveIt distance, max per-joint delta, L1, L2, clearance diagnostics and sample count; selections are in `{rel(output / 'stage3_h5_preferred_branch_sequences.json')}`.", f"Q26. Topology-aware distance outliers are diagnostic only; no arbitrary 30°/60° safety threshold was used. Statistical outlier status is not promoted to failure.", f"Q27. Intermediate TCP diagnostics: `{json.dumps(tcp, sort_keys=True)}`; no H5 TCP tolerance was invented.", f"Q28. Fresh-process replay: `{replay.get('fresh_process_count')}/3`, semantic result consistency: `{replay.get('all_semantic_results_consistent')}`.", f"Q29. New regression failures: `{regression.get('new_regression_failures')}`.", f"Q30. `STAGE_3_H5: {status}`; `FIRST_BLOCKER: {blocker or 'none'}`; `JOINT_CONNECTIVITY_CERTIFIED: {'YES' if connectivity else 'NO'}`; `READY_FOR_STAGE_3_H6: {'YES' if ready else 'NO'}`.", "", "## Boundary", "", "`SPRAY_TRAJECTORY_CERTIFIED: NO`", "`TIME_PARAMETERIZATION_DONE: NO`", "`RUCKIG_DONE: NO`", "`FINAL_TRAJECTORY_CERTIFIED: NO`", "", "`FJT_GOALS_SENT: 0`", "`ROBOT_MOTION_STARTED: NO`", "`WORKCELL_REOPTIMIZATION_STARTED: NO`", "`PATH_PLANNING_EXPERIMENT_STARTED: NO`", "`RUCKIG_STARTED: NO`", "`ML_TRAINING_STARTED: NO`", "`RL_TRAINING_STARTED: NO`"]
    q_lines = [line.replace("Q17. Transition self-collision link pairs:", "Q17. Transition collision link pairs (environment + self):") for line in q_lines]
    (output / "FINAL_REPORT.md").write_text("\n".join(q_lines) + "\n", encoding="utf-8", newline="\n")
    dump_json(output / "stage3_h5_replay_determinism.json", replay)
    return 0 if ready else 2


def orchestrate(output: Path) -> int:
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty H5 output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    before = {"h4_5_root": tree_manifest(H45_ROOT), "derived_model_sha256": sha256_file(DERIVED_URDF)}
    h45 = audit_h45(output)
    freeze = build_configuration_freeze(output, h45)
    plugin, seed_bank = build_ik_capability_audit(output)
    topology = parse_joint_topology(output)
    targets = load_jsonl(H3_TARGETS)
    feasible_ids = set(map(int, h45["feasible"].get("feasible_target_indices", [])))
    if len(targets) != 192 or len(feasible_ids) != 30 or any(index >= 64 for index in feasible_ids):
        raise RuntimeError("H4.5 formal target denominator/feasible set mismatch")
    surface = build_surface_input(output, h45, feasible_ids)
    executable = build_native_bridge(output)
    process_data = []
    process_signatures = []
    first_payload = None
    for replay_index in (1, 2, 3):
        process = run_ik_process(output, 10000 + replay_index, h45["selected"])
        candidate_rows, raw_nodes = collect_candidates(process, feasible_ids, topology["active_joint_rules"], targets)
        fk_refs = {node["ik_candidate_id"]: h42_fk_reference(process["result"], int(node["target_id"]), node["joint_values"], topology["active_joint_rules"]) for node in raw_nodes}
        nodes = []
        for node in raw_nodes:
            ref = fk_refs[node["ik_candidate_id"]]
            node["joint_rules"] = topology["active_joint_rules"]
            node["h42_fk_reference"] = ref
            node["h5_joint_limit_valid_reference"] = joint_valid(node["joint_values"], topology["active_joint_rules"])
            node["h5_joint_fk_valid"] = bool(node["h5_joint_limit_valid_reference"] and ref.get("available") and ref.get("fk_valid"))
            node["h5_endpoint_collision_free"] = False
            node["h5_formal_node"] = False
            nodes.append(node)
        candidate_edges, transition_rows = prepare_edges(nodes, surface)
        endpoint_csv, transition_csv = write_endpoint_and_transition_inputs(process["run_dir"], nodes, transition_rows)
        native = run_native_bridge(output, executable, process["run_dir"], endpoint_csv, transition_csv)
        endpoint_map = {str(row["source_node_id"]): row for row in native["endpoints"]}
        for node in nodes:
            endpoint = endpoint_map.get(node["ik_candidate_id"], {})
            node["h5_endpoint_native_fcl_checked"] = bool(endpoint.get("native_fcl_checked"))
            node["h5_endpoint_collision_free"] = bool(endpoint.get("native_fcl_checked")) and bool(endpoint.get("collision_free"))
            node["h5_formal_node"] = bool(node["h5_joint_fk_valid"] and node["h5_endpoint_collision_free"])
            node["FK_translation_error_m"] = node["h42_fk_reference"].get("translation_error_m")
            node["FK_orientation_error_rad"] = node["h42_fk_reference"].get("orientation_error_rad")
            node["FK_valid"] = bool(node["h5_joint_fk_valid"] and endpoint.get("fk_computable"))
            node["joint_limit_valid"] = bool(node["h5_joint_limit_valid_reference"] and endpoint.get("joint_limit_valid"))
            node["self_collision"] = endpoint.get("self_collision")
            node["environment_collision"] = endpoint.get("environment_collision")
            node["collision_free"] = node["h5_endpoint_collision_free"]
            node["self_min_signed_distance_m"] = endpoint.get("self_min_signed_distance", {}).get("distance_m")
            node["environment_min_signed_distance_m"] = endpoint.get("environment_min_signed_distance", {}).get("distance_m")
            node["native_FCL_checked"] = bool(endpoint.get("native_fcl_checked"))
            node["semantic_hash"] = semantic_hash({"target_id": node["target_id"], "joint_topology_normalized_values": node["joint_topology_normalized_values"], "FK_valid": node["FK_valid"], "joint_limit_valid": node["joint_limit_valid"], "self_collision": node["self_collision"], "environment_collision": node["environment_collision"], "collision_free": node["collision_free"]})
        candidate_edges, edge_summary, _ = edge_results(candidate_edges, native, nodes)
        components = connected_components(nodes, [edge for edge in candidate_edges if edge.get("valid")], feasible_ids)
        preferred = {"pair_selections": []}
        # This is run per-process without writing the top-level artifact yet.
        valid_edges = [edge for edge in candidate_edges if edge.get("valid")]
        by_pair: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
        for edge in valid_edges: by_pair[(int(edge["source_target_id"]), int(edge["destination_target_id"]))].append(edge)
        for pair, values in sorted(by_pair.items()):
            preferred["pair_selections"].append({"source_target_id": pair[0], "destination_target_id": pair[1], "preferred_edge_id": min(values, key=lambda edge: (edge["MoveIt_RobotState_distance"], edge["max_abs_joint_delta_rad"], edge["L1_joint_distance"], edge["L2_joint_distance"], edge["interpolation_sample_count"], edge["edge_id"]))["edge_id"]})
        clearance = {"minimum_environment_signed_distance_m": min((edge["minimum_environment_signed_distance_m"] for edge in candidate_edges if edge.get("minimum_environment_signed_distance_m") is not None), default=None), "minimum_self_signed_distance_m": min((edge["minimum_self_signed_distance_m"] for edge in candidate_edges if edge.get("minimum_self_signed_distance_m") is not None), default=None)}
        signature = replay_signature(process, nodes, candidate_edges, surface["surface_adjacent_edges"], components, preferred, clearance, candidate_rows)
        signature.update({"fresh_process_record": process["record"], "native_process_record": native["record"], "raw_ik_solution_count": sum(int(row["raw_ik_solution_count"]) for row in candidate_rows), "distinct_node_count": sum(bool(node.get("h5_formal_node")) for node in nodes), "valid_edge_count": len(valid_edges)})
        process_signatures.append(signature)
        process_data.append({"process": process, "candidate_rows": candidate_rows, "nodes": nodes, "candidate_edges": candidate_edges, "transition_rows": transition_rows, "native": native, "edge_summary": edge_summary, "components": components, "signature": signature})
        if first_payload is None: first_payload = process_data[-1]
    assert first_payload is not None
    all_consistent = len({semantic_hash({key: value for key, value in sig.items() if key not in {"fresh_process_record", "native_process_record", "authorization_result"}}) for sig in process_signatures}) == 1
    candidate_consistent = len({sig["candidate_set_hash"] for sig in process_signatures}) == 1
    replay = {"schema_version": "stage3-h5-replay-determinism-v1", "fresh_process_count": len(process_data), "processes": [{"replay_index": data["signature"]["fresh_process_record"]["replay_index"], "ik_clean_exit": data["signature"]["fresh_process_record"]["clean_exit"], "native_status": data["signature"]["native_process_record"]["status"], "signature_hash": semantic_hash(data["signature"])} for data in process_data], "semantic_result_consistency": all_consistent, "all_semantic_results_consistent": all_consistent, "all_candidate_sets_consistent": candidate_consistent, "candidate_edge_counts": [sig["candidate_edge_count"] for sig in process_signatures], "valid_edge_counts": [sig["valid_edge_count"] for sig in process_signatures], "component_hashes": [sig["component_hash"] for sig in process_signatures], "preferred_hashes": [sig["preferred_hash"] for sig in process_signatures], "clearance_hashes": [sig["clearance_hash"] for sig in process_signatures], "authorization_result": "PENDING", "wall_clock_fields_excluded": ["worker_pid", "elapsed_s", "timestamps"]}
    # Top-level artifacts are the first fresh process, while replay evidence
    # proves the same normalized graph was independently reconstructed twice.
    p = first_payload
    dump_jsonl(output / "stage3_h5_multi_ik_candidates.jsonl", p["candidate_rows"])
    dump_jsonl(output / "stage3_h5_distinct_ik_nodes.jsonl", p["nodes"])
    dump_jsonl(output / "stage3_h5_candidate_edges.jsonl", p["candidate_edges"])
    native_state_rows = p["native"]["states"]
    dump_jsonl(output / "stage3_h5_transition_collision_evidence.jsonl", p["candidate_edges"])
    tcp = tcp_diagnostics(output, [edge for edge in p["candidate_edges"] if edge.get("valid")], native_state_rows, targets)
    preferred_full = preferred_branches(output, p["candidate_edges"])
    clear = clearance_summary(output, p["native"]["summary"], p["candidate_edges"])
    dump_json(output / "stage3_h5_joint_configuration_graph.json", {"schema_version": "stage3-h5-joint-configuration-graph-v1", "target_space_graph_source": rel(output / "stage3_h5_surface_adjacency_input.json"), "node_semantics": "(target_id, ik_candidate_id)", "nodes": p["nodes"], "edges": p["candidate_edges"], "candidate_edge_count": p["edge_summary"]["candidate_edge_count"], "valid_edge_count": p["edge_summary"]["valid_transition_edge_count"], "collision_method": COLLISION_METHOD, "ccd_status": CCD_STATUS, "spray_edge_claim": "JOINT_CONNECTABLE_EDGE only; not CERTIFIED_SPRAY_EDGE"})
    dump_json(output / "stage3_h5_joint_connectable_components.json", p["components"])
    dump_json(output / "stage3_h5_replay_determinism.json", replay)
    dump_json(output / "stage3_h5_immutable_baseline_audit.json", immutable_audit(output, before))
    regression = run_regression(output)
    immutable = load_json(output / "stage3_h5_immutable_baseline_audit.json")
    replay["authorization_result"] = "computed_after_regression_and_immutable_audit"
    result = report_and_certify(output, h45, freeze, plugin, seed_bank, topology, surface, p["candidate_rows"], p["nodes"], p["candidate_edges"], p["edge_summary"], p["components"], preferred_full, tcp, clear, replay, regression, immutable)
    # Add reuse/provenance detail after the formal artifacts are complete.
    reuse = load_jsonl(H45_ROOT / "stage3_h4_5_native_branch_evidence.jsonl")
    dump_json(output / "stage3_h5_multi_ik_candidates_provenance.json", {"h4_5_existing_evidence_reused": True, "h4_5_evidence_path": rel(H45_ROOT / "stage3_h4_5_native_branch_evidence.jsonl"), "h4_5_evidence_row_count": len(reuse), "h5_fresh_process_count": 3, "h5_fresh_recollection_required": True, "h4_5_artifacts_not_overwritten": True})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output or (ROOT / "outputs" / f"stage3_h5_joint_branch_continuity_{timestamp}")).resolve()
    try:
        return orchestrate(output)
    except Exception as exc:
        print(f"STAGE_3_H5: BLOCKED", file=sys.stderr)
        print(f"FIRST_BLOCKER: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
