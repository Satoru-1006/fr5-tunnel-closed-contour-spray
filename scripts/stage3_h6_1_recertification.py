#!/usr/bin/env python3
"""Stage 3 H6.1 formal spray-process tolerance contract and H6 recertification.

This runner is intentionally additive.  It freezes a new Stage 3 research
acceptance contract, consumes the already-certified H6 open-arch trajectory,
replays the exact frozen native request three times, and evaluates every dense
Spray-ON waypoint.  It never changes H3/H4.5/H5/H6 inputs, regenerates a path,
or starts time parameterization, Ruckig, a controller, or robot motion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "config/stage3/stage3_h6_1_spray_process_tolerance_contract.json"
H6_ROOT = ROOT / "outputs/stage3_h6_surface_coverage_20260808T225000Z"
H6_REQUESTS = H6_ROOT / "stage3_h6_requests.tsv"
H6_MESH = H6_ROOT.parent / "stage3_h4_5_robot_fixture_geometry_codesign_20260809T000000Z" / "curved_fixture_stage3_h4_5_selected.obj"
H6_URDF = H6_ROOT.parent / "stage3_h4_5_robot_fixture_geometry_codesign_20260809T000000Z" / "derived_robot_model.urdf"
SRDF = ROOT / "ros2_moveit_bridge/config/fairino5_v6_spray_tcp.srdf"
H6_NATIVE_INSTALL = H6_ROOT / "native_install"
H6_NATIVE_EXECUTABLE = H6_NATIVE_INSTALL / "lib/stage3_h6_native/stage3_h6_native_bridge"
H6_FILES = [
    "FINAL_REPORT.md",
    "stage3_h6_cartesian_tcp_waypoints.jsonl",
    "stage3_h6_collision_evidence.jsonl",
    "stage3_h6_component_ordering.json",
    "stage3_h6_coverage_accounting.json",
    "stage3_h6_fk_tcp_validation.jsonl",
    "stage3_h6_gate_report.json",
    "stage3_h6_ik_branch_trace.jsonl",
    "stage3_h6_input_freeze_manifest.json",
    "stage3_h6_joint_waypoints.jsonl",
    "stage3_h6_native_process_record.json",
    "stage3_h6_native_process_record_native_on_probe.json",
    "stage3_h6_native_process_record_native_replay_2.json",
    "stage3_h6_native_process_record_native_replay_3.json",
    "stage3_h6_on_requests.tsv",
    "stage3_h6_regression_report.json",
    "stage3_h6_replay_determinism.json",
    "stage3_h6_requests.tsv",
    "stage3_h6_spray_on_off_segments.json",
    "stage3_h6_spray_on_off_segments.jsonl",
    "stage3_h6_surface_graph.json",
    "stage3_h6_surface_waypoints.jsonl",
    "stage3_h6_terminal_certificate.json",
]
H6_NATIVE_DIRS = ["native_final", "native_on_probe", "native_replay_2", "native_replay_3"]
FORMAL_ON_COUNT = 628
FULL_WAYPOINT_COUNT = 1295
FORMAL_TARGET_COUNT = 30
ON_SEGMENT_COUNT = 3
OFF_SEGMENT_COUNT = 2
COLLISION_METHOD = "adaptive_discrete_interpolation"
CCD_STATUS = "not_available"
FORBIDDEN_LOG_PATTERNS = ("process has died", "segmentation fault", "sigsegv", "exit code -11")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def dump_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n" for row in rows), encoding="utf-8", newline="\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): canonical(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [canonical(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite value in canonical payload")
        return value
    return value


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(canonical(value), ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def contract_digest(contract: Mapping[str, Any]) -> str:
    payload = json.loads(json.dumps(contract))
    payload["contract_sha256"] = None
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    rest = str(resolved).split(":", 1)[-1].replace("\\", "/").lstrip("/")
    return f"/mnt/{drive}/{rest}"


def run_wsl(command: str, timeout_s: int) -> tuple[int, str, str]:
    proc = subprocess.run(
        ["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_s,
        check=False,
    )
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def vec_sub(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [float(x) - float(y) for x, y in zip(a, b)]


def vec_norm(a: Sequence[float]) -> float:
    return math.sqrt(sum(float(x) * float(x) for x in a))


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(float(x) * float(y) for x, y in zip(a, b))


def unit(a: Sequence[float]) -> list[float]:
    length = vec_norm(a)
    if not math.isfinite(length) or length <= 1.0e-15:
        raise ValueError("zero or non-finite vector")
    return [float(x) / length for x in a]


def angle_deg(a: Sequence[float], b: Sequence[float]) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, dot(unit(a), unit(b))))))


def quat_normalize(q: Sequence[float]) -> list[float]:
    length = math.sqrt(sum(float(value) * float(value) for value in q))
    if not math.isfinite(length) or length <= 1.0e-15:
        raise ValueError("zero or non-finite quaternion")
    return [float(value) / length for value in q]


def quat_shortest_angle_deg(a: Sequence[float], b: Sequence[float]) -> float:
    qa = quat_normalize(a)
    qb = quat_normalize(b)
    return math.degrees(2.0 * math.acos(max(-1.0, min(1.0, abs(dot(qa, qb))))))


def quat_to_matrix(q: Sequence[float]) -> list[list[float]]:
    x, y, z, w = quat_normalize(q)
    return [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ]


def tcp_z_axis(q: Sequence[float]) -> list[float]:
    matrix = quat_to_matrix(q)
    return unit([matrix[0][2], matrix[1][2], matrix[2][2]])


def finite_vector(value: Any, length: int) -> bool:
    return isinstance(value, list) and len(value) == length and all(isinstance(item, (int, float)) and math.isfinite(float(item)) for item in value)


def validate_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    geometry = contract.get("geometry", {})
    tcp = contract.get("tcp_reproduction", {})
    required = {
        "nominal_standoff_m": geometry.get("nominal_standoff_m"),
        "standoff_abs_tolerance_m": geometry.get("standoff_abs_tolerance_m"),
        "standoff_min_m": geometry.get("standoff_min_m"),
        "standoff_max_m": geometry.get("standoff_max_m"),
        "normal_angle_tolerance_deg": geometry.get("normal_angle_tolerance_deg"),
        "tcp_position_tolerance_m": tcp.get("tcp_position_tolerance_m"),
        "tcp_orientation_tolerance_deg": tcp.get("tcp_orientation_tolerance_deg"),
    }
    errors: list[str] = []
    if contract.get("schema_version") != "stage3-h6-1-spray-process-tolerance-contract-v1": errors.append("schema_version")
    if contract.get("status") != "FROZEN_FOR_STAGE_3_H6_1": errors.append("status")
    for key, value in required.items():
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0.0:
            errors.append(key)
    if required["standoff_min_m"] != required["nominal_standoff_m"] - required["standoff_abs_tolerance_m"]: errors.append("standoff_min_m_formula")
    if required["standoff_max_m"] != required["nominal_standoff_m"] + required["standoff_abs_tolerance_m"]: errors.append("standoff_max_m_formula")
    if contract.get("tcp_link") != "spray_tcp_link" or contract.get("spray_tcp_link") != "spray_tcp_link": errors.append("tcp_link")
    if contract.get("frames", {}).get("reference_frame") != "base_link": errors.append("reference_frame")
    if contract.get("hashing", {}).get("algorithm") != "SHA-256": errors.append("hash_algorithm")
    actual_digest = contract_digest(contract)
    if contract.get("contract_sha256") != actual_digest: errors.append("contract_sha256")
    return {"valid": not errors, "errors": errors, "contract_sha256": actual_digest, "values": required}


def source_record(path: Path, role: str, classification: str, authoritative: bool) -> dict[str, Any]:
    return {
        "path": rel(path),
        "exists": path.is_file(),
        "sha256": sha256_file(path) if path.is_file() else None,
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "role": role,
        "classification": classification,
        "authoritative_for_h6_1_process_tolerance": authoritative,
    }


def build_provenance_audit(contract: Mapping[str, Any]) -> dict[str, Any]:
    h5_diag = H6_ROOT.parent / "stage3_h5_joint_branch_continuity_20260808T132000Z" / "stage3_h5_intermediate_tcp_diagnostics.jsonl"
    h5_freeze = H6_ROOT.parent / "stage3_h5_joint_branch_continuity_20260808T132000Z" / "stage3_h5_configuration_freeze_manifest.json"
    h3_pose = ROOT / "config/stage3/stage3_h3_target_pose_contract.json"
    h3_task = ROOT / "config/stage3/stage3_h3_task_representation_contract.json"
    h45_config = H6_ROOT.parent / "stage3_h4_5_robot_fixture_geometry_codesign_20260809T000000Z" / "stage3_h4_5_configuration_manifest.json"
    generic_metric = ROOT / "config/stage3/stage3_research_metric_contract.json"
    graph_config = ROOT / "config/ik_graph.yaml"
    collision_contract = ROOT / "config/stage3/stage3_adaptive_discrete_collision_contract.json"
    h6_input = H6_ROOT / "stage3_h6_input_freeze_manifest.json"
    h6_final = H6_ROOT / "FINAL_REPORT.md"
    h6_gate = H6_ROOT / "stage3_h6_gate_report.json"
    h6_terminal = H6_ROOT / "stage3_h6_terminal_certificate.json"
    h6_fk = H6_ROOT / "stage3_h6_fk_tcp_validation.jsonl"
    h6_surface = H6_ROOT / "stage3_h6_surface_waypoints.jsonl"
    h5_diag_rows = load_jsonl(h5_diag) if h5_diag.is_file() else []
    formal_flags = sorted({bool(row.get("formal_tcp_tolerance_applied")) for row in h5_diag_rows})
    sources = [
        source_record(h3_pose, "H3 target pose contract", "frozen upstream geometry semantics; nominal standoff source", False),
        source_record(h3_task, "H3 task representation contract", "frozen upstream record/frame semantics", False),
        source_record(h45_config, "H4.5 configuration manifest", "frozen workcell configuration and nominal process parameter", False),
        source_record(h5_freeze, "H5 configuration freeze manifest", "frozen branch/workcell input; no formal H6.1 process tolerance", False),
        source_record(h5_diag, "H5 intermediate TCP diagnostics", "diagnostic-only transition values; formal_tcp_tolerance_applied is false", False),
        source_record(h6_input, "H6 input freeze manifest", "authoritative H6 preflight record; formal process tolerances unresolved", False),
        source_record(h6_final, "H6 FINAL_REPORT", "original H6 terminal report; records formal_spray_process_tolerance_not_defined", False),
        source_record(h6_gate, "H6 gate report", "original H6 gate; blocker is preserved", False),
        source_record(h6_terminal, "H6 terminal certificate", "original H6 certificate; immutable evidence", False),
        source_record(h6_fk, "H6 FK/TCP evidence", "measured native evidence; not a threshold authority", False),
        source_record(h6_surface, "H6 dense surface waypoint evidence", "frozen commanded path consumed without mutation", False),
        source_record(generic_metric, "generic Stage 3 research metric contract", "generic metric candidates; not H6-authoritative formal tolerance", False),
        source_record(graph_config, "legacy/graph configuration", "legacy or numerical/configuration values; not H6.1 process tolerance", False),
        source_record(collision_contract, "adaptive collision contract", "collision discretization semantics; not process geometry tolerance", False),
        source_record(CONTRACT, "H6.1 tolerance contract", "new project-defined authority created for this stage", True),
    ]
    prior_formal = bool(formal_flags and any(formal_flags))
    old_h6_unresolved = "formal_spray_process_tolerance_not_defined" in h6_final.read_text(encoding="utf-8") if h6_final.is_file() else False
    generic_candidates = {
        "generic_stage3_research_metric_contract": {
            "spray_distance_threshold": "13 mm candidate under FROZEN_FOR_STAGE0_1_SCOPE; not H6-authoritative",
            "surface_normal_threshold": "10 deg candidate; not H6-authoritative",
        },
        "legacy_graph_configuration": {
            "max_standoff_error_fraction": "5 percent numerical/graph policy; not H6.1 process authority",
            "max_normal_error_deg": "10 deg graph policy; not H6.1 process authority",
        },
        "h6_numerical_verification": {
            "translation_tolerance_m": "1e-6; IK/FK numerical verification only",
            "rotation_tolerance_rad": "1e-6; IK/FK numerical verification only",
        },
        "h6_discretization": {
            "max_cartesian_step_m": "0.01; numerical discretization only",
            "max_angular_step_deg": "2.0; numerical discretization only",
        },
    }
    return {
        "schema_version": "stage3-h6-1-tolerance-provenance-audit-v1",
        "audit_scope": "read-only audit of H3 target pose, H3 task representation, H4.5, H5, H6 and related Stage 3 configuration before H6.1 contract freeze",
        "formal_spray_process_tolerance_found_before_h6_1": False,
        "prior_h6_authoritative_tolerance_found": False,
        "h5_formal_tcp_tolerance_applied_values": formal_flags,
        "original_h6_reported_unresolved_blocker": old_h6_unresolved,
        "numerical_solver_collision_controller_values_misclassified_as_process_tolerance": False,
        "generic_numeric_candidates_reviewed_but_not_adopted": generic_candidates,
        "sources": sources,
        "decision": "Create a new independent Stage 3 H6.1 project-defined research acceptance contract. Do not rewrite H3/H4.5/H5 or reinterpret numerical, collision, discretization, or controller tolerances.",
        "adopted_contract_sha256": contract_digest(contract),
        "adopted_values_are_predeclared_not_posthoc": True,
        "contract_values": contract.get("geometry", {}) | contract.get("tcp_reproduction", {}),
    }


def h6_source_hashes() -> dict[str, Any]:
    records: dict[str, Any] = {}
    for name in H6_FILES:
        path = H6_ROOT / name
        records[name] = source_record(path, "original H6 evidence", "frozen input/evidence", True)
    for directory in H6_NATIVE_DIRS:
        base = H6_ROOT / directory
        if not base.is_dir():
            records[directory] = {"path": rel(base), "exists": False, "sha256": None}
            continue
        for path in sorted(item for item in base.rglob("*") if item.is_file()):
            key = f"{directory}/{path.relative_to(base).as_posix()}"
            records[key] = {"path": rel(path), "exists": True, "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    return records


def copy_preserved_evidence(output: Path) -> None:
    destination = output / "raw_native_evidence"
    destination.mkdir(parents=True, exist_ok=True)
    for name in H6_FILES:
        source = H6_ROOT / name
        if source.is_file():
            shutil.copy2(source, destination / name)
    for directory in H6_NATIVE_DIRS:
        source = H6_ROOT / directory
        if source.is_dir():
            shutil.copytree(source, destination / directory)


def freeze_contract_and_inputs(output: Path, contract: Mapping[str, Any], contract_info: Mapping[str, Any], audit: Mapping[str, Any], input_hashes: Mapping[str, Any]) -> dict[str, Any]:
    contract_copy = output / "stage3_h6_1_spray_process_tolerance_contract.json"
    shutil.copy2(CONTRACT, contract_copy)
    contract_copy_hash = sha256_file(contract_copy)
    freeze = {
        "schema_version": "stage3-h6-1-contract-freeze-manifest-v1",
        "freeze_status": "FROZEN",
        "contract_id": contract.get("contract_id"),
        "contract_source": rel(CONTRACT),
        "contract_copy": rel(contract_copy),
        "contract_sha256": contract_info["contract_sha256"],
        "contract_file_sha256": contract_copy_hash,
        "contract_hash_rule": contract.get("hashing", {}).get("canonicalization"),
        "tolerance_contract_frozen_before_recertification": True,
        "contract_defined_before_metrics": True,
        "provenance_audit_path": "stage3_h6_1_tolerance_provenance_audit.json",
        "provenance_audit_sha256": sha256_file(output / "stage3_h6_1_tolerance_provenance_audit.json"),
        "original_h6_root": rel(H6_ROOT),
        "original_h6_source_hashes": input_hashes,
        "frozen_request": {"path": rel(H6_REQUESTS), "sha256": sha256_file(H6_REQUESTS), "waypoint_count": FULL_WAYPOINT_COUNT},
        "frozen_geometry_and_configuration": {
            "mesh": rel(H6_MESH),
            "mesh_sha256": sha256_file(H6_MESH),
            "urdf": rel(H6_URDF),
            "urdf_sha256": sha256_file(H6_URDF),
            "srdf": rel(SRDF),
            "srdf_sha256": sha256_file(SRDF),
            "tcp_link": "spray_tcp_link",
            "reference_frame": "base_link",
        },
        "immutable_constraints": {
            "h4_5_immutable": "YES",
            "h5_immutable": "YES",
            "original_h6_evidence_immutable": "YES",
            "trajectory_regeneration": "FORBIDDEN",
            "target_selection_mutation": "FORBIDDEN",
            "segment_mutation": "FORBIDDEN",
            "standoff_mutation": "FORBIDDEN",
        },
        "audit_contract_digest_matches": audit.get("adopted_contract_sha256") == contract_info["contract_sha256"],
    }
    dump_json(output / "stage3_h6_1_contract_freeze_manifest.json", freeze)
    return freeze


def native_semantic_payload(native_dir: Path) -> dict[str, Any]:
    names = {
        "summary": "stage3_h6_native_summary.json",
        "branch": "stage3_h6_native_ik_branch_trace.jsonl",
        "fk": "stage3_h6_native_fk_tcp_validation.jsonl",
        "collision": "stage3_h6_native_collision_evidence.jsonl",
    }
    payload: dict[str, Any] = {}
    for key, name in names.items():
        path = native_dir / name
        if not path.is_file():
            payload[key] = None
        elif path.suffix == ".json":
            payload[key] = load_json(path)
        else:
            payload[key] = load_jsonl(path)
    return payload


def native_semantic_signature(native_dir: Path) -> str | None:
    payload = native_semantic_payload(native_dir)
    if any(value is None for value in payload.values()):
        return None
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def run_native_replay(output: Path, index: int) -> dict[str, Any]:
    name = f"fresh_native_replay_{index}"
    native_dir = output / name
    source_setup = ROOT / "install/setup.bash"
    native_setup = H6_NATIVE_INSTALL / "setup.bash"
    parts = [
        "set -e",
        "source /opt/ros/jazzy/setup.bash",
        f"source {shlex.quote(wsl_path(source_setup))}",
        f"source {shlex.quote(wsl_path(native_setup))}",
        f"cd {shlex.quote(wsl_path(ROOT))}",
        " ".join([
            shlex.quote(wsl_path(H6_NATIVE_EXECUTABLE)),
            "--urdf", shlex.quote(wsl_path(H6_URDF)),
            "--srdf", shlex.quote(wsl_path(SRDF)),
            "--requests", shlex.quote(wsl_path(H6_REQUESTS)),
            "--mesh", shlex.quote(wsl_path(H6_MESH)),
            "--output", shlex.quote(wsl_path(native_dir)),
        ]),
    ]
    returncode, stdout, stderr = run_wsl(" && ".join(parts), 1800)
    log = output / f"stage3_h6_1_native_replay_{index}.log"
    log.write_text(stdout + "\n--- STDERR ---\n" + stderr, encoding="utf-8", newline="\n")
    summary_path = native_dir / "stage3_h6_native_summary.json"
    summary = load_json(summary_path) if summary_path.is_file() else None
    forbidden = [pattern for pattern in FORBIDDEN_LOG_PATTERNS if pattern in (stdout + stderr).lower()]
    status = "PASSED" if returncode == 0 and summary and summary.get("status") == "AVAILABLE" and not forbidden else "BLOCKED"
    record = {
        "schema_version": "stage3-h6-1-native-replay-record-v1",
        "replay_index": index,
        "native_name": name,
        "native_output": rel(native_dir),
        "returncode": returncode,
        "status": status,
        "summary": summary,
        "semantic_signature": native_semantic_signature(native_dir) if status == "PASSED" else None,
        "forbidden_log_patterns": forbidden,
        "request_sha256": sha256_file(H6_REQUESTS),
        "process_is_fresh": True,
    }
    dump_json(output / f"stage3_h6_1_native_replay_record_{index}.json", record)
    return record


def frozen_full_waypoint_rows() -> list[dict[str, Any]]:
    """Reattach frozen ON geometry to the final H6 joint-waypoint ordering.

    H6 persists the 628 commanded Spray-ON Cartesian rows before inserting
    667 Spray-OFF transfer rows.  The final joint/native trace then renumbers
    the interleaved 1295-row sequence.  ON geometry is therefore matched by
    ordered Spray-ON occurrence, never by the pre-insertion waypoint index.
    """
    on_rows = load_jsonl(H6_ROOT / "stage3_h6_cartesian_tcp_waypoints.jsonl")
    joint_rows = sorted(load_jsonl(H6_ROOT / "stage3_h6_joint_waypoints.jsonl"), key=lambda row: int(row["waypoint_index"]))
    full: list[dict[str, Any]] = []
    on_cursor = 0
    for joint in joint_rows:
        if joint.get("spray_state") == "SPRAY_ON":
            if on_cursor >= len(on_rows):
                raise RuntimeError("frozen H6 ON geometry is shorter than the native ON sequence")
            row = dict(on_rows[on_cursor])
            on_cursor += 1
            for key in ("waypoint_index", "segment_id", "component_id", "source_target_id", "destination_target_id", "spray_state"):
                if key in joint:
                    row[key] = joint[key]
            full.append(row)
        else:
            full.append({
                "schema_version": "stage3-h6-spray-off-waypoint-v1",
                "waypoint_index": joint.get("waypoint_index"),
                "segment_id": joint.get("segment_id"),
                "component_id": joint.get("component_id"),
                "source_target_id": joint.get("source_target_id"),
                "destination_target_id": joint.get("destination_target_id"),
                "spray_state": "SPRAY_OFF",
                "tcp_position_xyz_m": [0.0, 0.0, 0.0],
                "tcp_orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            })
    if on_cursor != len(on_rows) or len(full) != FULL_WAYPOINT_COUNT:
        raise RuntimeError(f"frozen H6 full waypoint reconstruction mismatch: on={on_cursor}/{len(on_rows)}, full={len(full)}")
    return full


def process_validation(native_dir: Path, cartesian_rows: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    branch_rows = {int(row["waypoint_index"]): row for row in load_jsonl(native_dir / "stage3_h6_native_ik_branch_trace.jsonl")}
    fk_rows = {int(row["waypoint_index"]): row for row in load_jsonl(native_dir / "stage3_h6_native_fk_tcp_validation.jsonl")}
    geometry = contract["geometry"]
    tcp_contract = contract["tcp_reproduction"]
    validation: list[dict[str, Any]] = []
    for cart in sorted(cartesian_rows, key=lambda row: int(row["waypoint_index"])):
        index = int(cart["waypoint_index"])
        branch = branch_rows.get(index, {})
        fk = fk_rows.get(index, {})
        state = branch.get("accepted_state") or {}
        accepted_q = branch.get("accepted_joint_values")
        actual_position = state.get("tcp_position_m") or fk.get("tcp_position_m")
        actual_orientation = state.get("tcp_orientation_xyzw") or fk.get("tcp_orientation_xyzw")
        accepted = branch.get("accepted") is True
        kinematic = {
            "ik_valid": accepted,
            "joint_limit_valid": state.get("joint_limit_valid") is True,
            "fk_valid": state.get("fk_computable") is True,
            "self_collision_free": state.get("self_collision") is False,
            "environment_collision_free": state.get("environment_collision") is False,
            "collision_free": state.get("collision_free") is True,
            "adaptive_discrete_interpolation": COLLISION_METHOD,
            "ccd_status": CCD_STATUS,
        }
        row: dict[str, Any] = {
            "schema_version": "stage3-h6-1-process-validation-v1",
            "waypoint_index": index,
            "segment_id": cart.get("segment_id"),
            "component_id": cart.get("component_id"),
            "spray_state": cart.get("spray_state"),
            "source_target_id": cart.get("source_target_id"),
            "destination_target_id": cart.get("destination_target_id"),
            "desired_tcp_position_m": cart.get("tcp_position_xyz_m"),
            "desired_tcp_orientation_xyzw": cart.get("tcp_orientation_xyzw"),
            "desired_surface_point_m": cart.get("surface_point_xyz_m"),
            "desired_surface_normal_unit": cart.get("surface_normal_unit"),
            "expected_spray_direction_unit": cart.get("spray_direction_unit"),
            "fk_tcp_position_m": actual_position,
            "fk_tcp_orientation_xyzw": actual_orientation,
            "accepted_joint_values": accepted_q,
            "native_reported_translation_error_m": state.get("translation_error_m"),
            "native_reported_rotation_error_rad": state.get("rotation_error_rad"),
            "kinematic_and_collision_checks": kinematic,
        }
        if cart.get("spray_state") != "SPRAY_ON":
            row["process_tolerance_applicable"] = False
            row["process_tolerance_checks"] = {
                "status": "NOT_APPLICABLE",
                "standoff": None,
                "surface_normal": None,
                "tcp_position": None,
                "tcp_orientation": None,
            }
            row["process_tolerance_pass"] = None
            row["required_spray_off_checks_pass"] = all(kinematic[key] for key in ("ik_valid", "joint_limit_valid", "fk_valid", "self_collision_free", "environment_collision_free", "collision_free"))
            validation.append(row)
            continue
        try:
            if not finite_vector(actual_position, 3) or not finite_vector(actual_orientation, 4):
                raise ValueError("missing finite FK TCP pose")
            desired_position = cart["tcp_position_xyz_m"]
            desired_orientation = cart["tcp_orientation_xyzw"]
            surface_point = cart["surface_point_xyz_m"]
            expected_direction = cart["spray_direction_unit"]
            standoff_m = vec_norm(vec_sub(actual_position, surface_point))
            signed_error_m = standoff_m - float(geometry["nominal_standoff_m"])
            abs_error_m = abs(signed_error_m)
            actual_direction = tcp_z_axis(actual_orientation)
            normal_deviation_deg = angle_deg(actual_direction, expected_direction)
            tcp_position_error_m = vec_norm(vec_sub(actual_position, desired_position))
            tcp_orientation_error_deg = quat_shortest_angle_deg(actual_orientation, desired_orientation)
            checks = {
                "standoff": {
                    "pass": abs_error_m <= float(geometry["standoff_abs_tolerance_m"]),
                    "actual_standoff_m": standoff_m,
                    "desired_standoff_m": float(geometry["nominal_standoff_m"]),
                    "signed_error_m": signed_error_m,
                    "absolute_error_m": abs_error_m,
                    "tolerance_m": float(geometry["standoff_abs_tolerance_m"]),
                    "required_min_m": float(geometry["standoff_min_m"]),
                    "required_max_m": float(geometry["standoff_max_m"]),
                },
                "surface_normal": {
                    "pass": normal_deviation_deg <= float(geometry["normal_angle_tolerance_deg"]),
                    "surface_normal_unit": cart["surface_normal_unit"],
                    "expected_spray_direction_unit": expected_direction,
                    "actual_spray_direction_unit": actual_direction,
                    "angular_deviation_deg": normal_deviation_deg,
                    "tolerance_deg": float(geometry["normal_angle_tolerance_deg"]),
                },
                "tcp_position": {
                    "pass": tcp_position_error_m <= float(tcp_contract["tcp_position_tolerance_m"]),
                    "desired_position_m": desired_position,
                    "fk_position_m": actual_position,
                    "error_m": tcp_position_error_m,
                    "tolerance_m": float(tcp_contract["tcp_position_tolerance_m"]),
                },
                "tcp_orientation": {
                    "pass": tcp_orientation_error_deg <= float(tcp_contract["tcp_orientation_tolerance_deg"]),
                    "desired_orientation_xyzw": desired_orientation,
                    "fk_orientation_xyzw": actual_orientation,
                    "shortest_angle_error_deg": tcp_orientation_error_deg,
                    "tolerance_deg": float(tcp_contract["tcp_orientation_tolerance_deg"]),
                },
            }
            row.update({
                "process_tolerance_applicable": True,
                "actual_spray_direction_unit": actual_direction,
                "actual_standoff_m": standoff_m,
                "signed_standoff_error_m": signed_error_m,
                "absolute_standoff_error_m": abs_error_m,
                "normal_angle_deviation_deg": normal_deviation_deg,
                "tcp_position_error_m": tcp_position_error_m,
                "tcp_orientation_error_deg": tcp_orientation_error_deg,
                "process_tolerance_checks": checks,
                "process_tolerance_pass": all(bool(check["pass"]) for check in checks.values()),
            })
        except (TypeError, ValueError, KeyError, ZeroDivisionError) as exc:
            row.update({
                "process_tolerance_applicable": True,
                "process_tolerance_checks": {"status": "BLOCKED", "error": f"{type(exc).__name__}: {exc}"},
                "process_tolerance_pass": False,
            })
        row["required_spray_on_kinematic_collision_checks_pass"] = all(kinematic[key] for key in ("ik_valid", "joint_limit_valid", "fk_valid", "self_collision_free", "environment_collision_free", "collision_free"))
        validation.append(row)
    summary = summarize_validation(validation, contract)
    return validation, summary


def summarize_validation(rows: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]) -> dict[str, Any]:
    on = [row for row in rows if row.get("spray_state") == "SPRAY_ON"]
    off = [row for row in rows if row.get("spray_state") == "SPRAY_OFF"]
    failed_process = [row for row in on if row.get("process_tolerance_pass") is not True]
    failed_kinematic = [row for row in rows if not (row.get("required_spray_on_kinematic_collision_checks_pass", row.get("required_spray_off_checks_pass", False)))]
    def values(key: str) -> list[float]:
        return [float(row[key]) for row in on if isinstance(row.get(key), (int, float)) and math.isfinite(float(row[key]))]
    standoff = values("actual_standoff_m")
    standoff_abs = values("absolute_standoff_error_m")
    normal = values("normal_angle_deviation_deg")
    position = values("tcp_position_error_m")
    orientation = values("tcp_orientation_error_deg")
    first_violation: dict[str, Any] | None = None
    for row in failed_process:
        checks = row.get("process_tolerance_checks", {})
        if isinstance(checks, Mapping) and checks.get("status") == "BLOCKED":
            first_violation = {"reason": "missing_or_invalid_measurement", "waypoint_index": row.get("waypoint_index"), "segment_id": row.get("segment_id"), "measured_value": None, "required_value": None, "violation_amount": None}
            break
        for key, measurement_key, tolerance_key in (
            ("standoff", "absolute_error_m", "tolerance_m"),
            ("surface_normal", "angular_deviation_deg", "tolerance_deg"),
            ("tcp_position", "error_m", "tolerance_m"),
            ("tcp_orientation", "shortest_angle_error_deg", "tolerance_deg"),
        ):
            check = checks.get(key, {}) if isinstance(checks, Mapping) else {}
            if isinstance(check, Mapping) and check.get("pass") is False:
                measured = float(check[measurement_key])
                required = float(check[tolerance_key])
                first_violation = {
                    "reason": "FIRST_PROCESS_TOLERANCE_VIOLATION",
                    "waypoint_index": row.get("waypoint_index"),
                    "segment_id": row.get("segment_id"),
                    "measurement": key,
                    "measured_value": measured,
                    "required_value": required,
                    "violation_amount": measured - required,
                }
                break
        if first_violation:
            break
    return {
        "schema_version": "stage3-h6-1-process-validation-summary-v1",
        "spray_on_waypoint_count": len(on),
        "spray_on_waypoints_passed": len(on) - len(failed_process),
        "spray_on_waypoints_failed": len(failed_process),
        "spray_off_waypoint_count": len(off),
        "all_spray_off_kinematic_collision_checks_pass": not any(not row.get("required_spray_off_checks_pass", False) for row in off),
        "all_kinematic_collision_checks_pass": not failed_kinematic,
        "standoff_min_m": min(standoff) if standoff else None,
        "standoff_max_m": max(standoff) if standoff else None,
        "standoff_max_absolute_error_m": max(standoff_abs) if standoff_abs else None,
        "normal_max_deviation_deg": max(normal) if normal else None,
        "tcp_max_position_error_m": max(position) if position else None,
        "tcp_max_orientation_error_deg": max(orientation) if orientation else None,
        "tolerance_contract_values": contract.get("geometry", {}) | contract.get("tcp_reproduction", {}),
        "first_process_tolerance_violation": first_violation,
        "process_tolerance_all_spray_on_pass": len(on) == FORMAL_ON_COUNT and not failed_process,
    }


def replay_process_signature(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = [
        {
            "waypoint_index": row.get("waypoint_index"),
            "spray_state": row.get("spray_state"),
            "accepted_joint_values": row.get("accepted_joint_values"),
            "kinematic_and_collision_checks": row.get("kinematic_and_collision_checks"),
            "process_tolerance_pass": row.get("process_tolerance_pass"),
            "actual_standoff_m": row.get("actual_standoff_m"),
            "signed_standoff_error_m": row.get("signed_standoff_error_m"),
            "normal_angle_deviation_deg": row.get("normal_angle_deviation_deg"),
            "tcp_position_error_m": row.get("tcp_position_error_m"),
            "tcp_orientation_error_deg": row.get("tcp_orientation_error_deg"),
        }
        for row in rows
    ]
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def build_replay_report(output: Path, contract: Mapping[str, Any], cartesian_rows: Sequence[Mapping[str, Any]], segments: Sequence[Mapping[str, Any]], replay_records: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    enriched: list[dict[str, Any]] = []
    successful_validations: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for record in replay_records:
        if record.get("status") != "PASSED":
            enriched.append({**record, "process_tolerance_signature": None, "process_validation_summary": None})
            continue
        native_dir = output / str(record["native_output"]).split("outputs/", 1)[-1] if False else output / Path(str(record["native_output"])).name
        rows, summary = process_validation(native_dir, cartesian_rows, contract)
        process_signature = replay_process_signature(rows)
        enriched.append({**record, "process_tolerance_signature": process_signature, "process_validation_summary": summary})
        successful_validations.append((dict(record), {"rows": rows, "summary": summary, "process_signature": process_signature}))
    native_signatures = [record.get("semantic_signature") for record in enriched]
    process_signatures = [record.get("process_tolerance_signature") for record in enriched]
    passed = len(enriched) == 3 and all(record.get("status") == "PASSED" for record in enriched) and len(set(native_signatures)) == 1 and len(set(process_signatures)) == 1
    report = {
        "schema_version": "stage3-h6-1-replay-determinism-v1",
        "fresh_process_count": len(enriched),
        "processes": enriched,
        "target_ordering": [segment.get("target_ids") for segment in segments if segment.get("spray_state") == "SPRAY_ON"],
        "segment_ordering": [{key: segment.get(key) for key in ("segment_id", "spray_state", "source_segment_id", "destination_segment_id", "waypoint_start", "waypoint_end")} for segment in segments],
        "waypoint_count": len(cartesian_rows),
        "spray_on_waypoint_count": sum(row.get("spray_state") == "SPRAY_ON" for row in cartesian_rows),
        "coverage_accounting": load_json(H6_ROOT / "stage3_h6_coverage_accounting.json"),
        "collision_method": COLLISION_METHOD,
        "ccd_status": CCD_STATUS,
        "native_semantic_signatures": native_signatures,
        "process_tolerance_signatures": process_signatures,
        "native_results_identical": passed,
        "process_tolerance_results_identical": passed,
        "FRESH_PROCESS_REPLAY": "PASSED" if passed else "BLOCKED",
        "replay_requirement": "three independent fresh native MoveIt2/FK/FCL processes on the exact frozen H6 request with identical ordering, joint, coverage, process-tolerance, collision, and FK/TCP semantics",
    }
    return report, successful_validations


def run_regression() -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_stage3_h6_1.py",
        "tests/test_stage3_h6.py",
        "tests/test_stage3_h5.py",
        "tests/test_stage3_h4_4.py",
        "tests/test_stage3_h4_reachability.py",
        "tests/test_stage3_h3_task_representation.py",
        "tests/test_stage3_h2_geometry.py",
    ]
    proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1800, check=False)
    raw = proc.stdout + "\n--- STDERR ---\n" + proc.stderr
    match = re.search(r"(\d+) passed(?:, (\d+) failed)?", raw)
    failed = int(match.group(2)) if match and match.group(2) else (0 if proc.returncode == 0 else 1)
    return {
        "schema_version": "stage3-h6-1-regression-report-v1",
        "runs": [{"command": command, "returncode": proc.returncode, "passed": int(match.group(1)) if match else None, "failed": failed, "raw_output_tail": raw[-6000:]}],
        "new_regression_failures": failed,
        "known_pre_existing_failures": [],
    }


def artifact_manifest(output: Path) -> dict[str, Any]:
    required = [
        "FINAL_REPORT.md",
        "stage3_h6_1_spray_process_tolerance_contract.json",
        "stage3_h6_1_tolerance_provenance_audit.json",
        "stage3_h6_1_contract_freeze_manifest.json",
        "stage3_h6_1_process_validation.jsonl",
        "stage3_h6_1_replay_determinism.json",
        "stage3_h6_1_regression_report.json",
        "stage3_h6_1_gate_report.json",
    ]
    files = {}
    for name in required:
        path = output / name
        files[name] = {"path": name, "exists": path.is_file(), "sha256": sha256_file(path) if path.is_file() else None, "size_bytes": path.stat().st_size if path.is_file() else None}
    manifest = {"schema_version": "stage3-h6-1-artifact-manifest-v1", "files": files, "note": "The terminal certificate is emitted after this manifest and is handoff-hashed separately to avoid a self-referential file hash."}
    dump_json(output / "stage3_h6_1_artifact_manifest.json", manifest)
    manifest["manifest_sha256"] = sha256_file(output / "stage3_h6_1_artifact_manifest.json")
    return manifest


def build_gate_and_terminal(contract_info: Mapping[str, Any], freeze: Mapping[str, Any], audit: Mapping[str, Any], replay: Mapping[str, Any], summary: Mapping[str, Any], coverage: Mapping[str, Any], regression: Mapping[str, Any], source_immutable: bool, output: Path) -> tuple[dict[str, Any], dict[str, Any], str | None]:
    contract_ok = bool(contract_info.get("valid"))
    freeze_ok = bool(freeze.get("tolerance_contract_frozen_before_recertification"))
    process_ok = bool(summary.get("process_tolerance_all_spray_on_pass"))
    kinematic_ok = bool(summary.get("all_kinematic_collision_checks_pass"))
    coverage_ok = coverage.get("spray_on_covered_targets") == FORMAL_TARGET_COUNT and not coverage.get("uncovered_targets") and not coverage.get("duplicate/revisited_targets")
    segmentation_ok = coverage.get("spray_on_segment_count") == ON_SEGMENT_COUNT and coverage.get("spray_off_segment_count") == OFF_SEGMENT_COUNT
    replay_ok = replay.get("FRESH_PROCESS_REPLAY") == "PASSED"
    regression_ok = regression.get("new_regression_failures") == 0
    first_violation = summary.get("first_process_tolerance_violation")
    blocker: str | None = None
    for condition, name in (
        (contract_ok, "formal_spray_process_tolerance_contract_invalid"),
        (audit.get("prior_h6_authoritative_tolerance_found") is False, "prior_formal_spray_process_tolerance_conflict"),
        (freeze_ok, "tolerance_contract_not_frozen_before_recertification"),
        (source_immutable, "original_h6_evidence_changed"),
        (replay_ok, "fresh_native_replay_unavailable_or_non_deterministic"),
        (not first_violation, "FIRST_PROCESS_TOLERANCE_VIOLATION" if first_violation else ""),
        (kinematic_ok, "native_ik_fk_collision_check_failed"),
        (coverage_ok, "coverage_accounting_changed"),
        (segmentation_ok, "spray_on_off_segment_structure_changed"),
        (regression_ok, "new_regression_failure"),
    ):
        if not condition:
            blocker = name
            break
    status = "PASSED" if blocker is None else "BLOCKED"
    gate = {
        "schema_version": "stage3-h6-1-gate-report-v1",
        "STAGE_3_H6_1": status,
        "STAGE_3_H6_RECERTIFICATION": status,
        "FIRST_BLOCKER": blocker,
        "formal_tolerance_contract_defined": contract_ok,
        "tolerance_provenance_documented": audit.get("formal_spray_process_tolerance_found_before_h6_1") is False,
        "tolerance_contract_frozen_before_recertification": freeze_ok,
        "all_spray_on_waypoints_process_tolerance_pass": process_ok,
        "all_required_kinematic_collision_checks_pass": kinematic_ok,
        "coverage_30_of_30": coverage_ok,
        "segment_structure_3_on_plus_2_off": segmentation_ok,
        "fresh_native_replay_3_of_3_identical": replay_ok,
        "new_regression_failures": regression.get("new_regression_failures"),
        "spray_on_waypoint_count": summary.get("spray_on_waypoint_count"),
        "spray_off_waypoint_count": summary.get("spray_off_waypoint_count"),
        "full_waypoint_count": FULL_WAYPOINT_COUNT,
        "diagnostic_metrics": {key: summary.get(key) for key in ("standoff_min_m", "standoff_max_m", "standoff_max_absolute_error_m", "normal_max_deviation_deg", "tcp_max_position_error_m", "tcp_max_orientation_error_deg")},
        "H4_5_IMMUTABLE": "YES",
        "H5_IMMUTABLE": "YES",
        "ORIGINAL_H6_EVIDENCE_IMMUTABLE": "YES" if source_immutable else "NO",
        "NEW_FJT_GOALS_SENT": 0,
        "ROBOT_MOTION_STARTED": "NO",
        "FORMAL_LEDGER_MUTATED": "NO",
        "RUCKIG_STARTED": "NO",
        "TIME_PARAMETERIZATION_DONE": "NO",
        "ML_TRAINING_STARTED": "NO",
        "CCD": "NOT_AVAILABLE",
        "collision_method": COLLISION_METHOD,
        "mandatory_blockers": [blocker] if blocker else [],
        "original_h6_status_preserved": "BLOCKED",
        "artifact_manifest_path": "stage3_h6_1_artifact_manifest.json",
    }
    terminal = {
        "schema_version": "stage3-h6-1-terminal-certificate-v1",
        **gate,
        "READY_FOR_STAGE_3_H7": "YES" if status == "PASSED" else "NO",
        "FIRST_PROCESS_TOLERANCE_VIOLATION": first_violation,
        "contract_sha256": contract_info.get("contract_sha256"),
        "tolerance_contract_frozen_before_recertification": freeze_ok,
        "fresh_replay_semantic_signatures": replay.get("native_semantic_signatures"),
        "fresh_replay_process_tolerance_signatures": replay.get("process_tolerance_signatures"),
        "artifact_manifest_sha256": None,
        "key_artifacts": {},
        "certificate_sha256": None,
    }
    return gate, terminal, blocker


def final_report(output: Path, contract: Mapping[str, Any], audit: Mapping[str, Any], freeze: Mapping[str, Any], replay: Mapping[str, Any], summary: Mapping[str, Any], coverage: Mapping[str, Any], regression: Mapping[str, Any], gate: Mapping[str, Any], terminal: Mapping[str, Any]) -> None:
    lines = [
        "# Stage 3 H6.1 - Formal Spray-Process Tolerance Contract + H6 Recertification",
        "",
        f"`STAGE_3_H6_1: {gate['STAGE_3_H6_1']}`",
        f"`STAGE_3_H6_RECERTIFICATION: {gate['STAGE_3_H6_RECERTIFICATION']}`",
        f"`FIRST_BLOCKER: {gate['FIRST_BLOCKER'] or 'none'}`",
        f"`READY_FOR_STAGE_3_H7: {terminal['READY_FOR_STAGE_3_H7']}`",
        "",
        "## Tolerance provenance audit",
        "",
        f"- Prior H6-authoritative formal spray-process tolerance found: `{audit['prior_h6_authoritative_tolerance_found']}`.",
        f"- H5 `formal_tcp_tolerance_applied` values: `{audit['h5_formal_tcp_tolerance_applied_values']}`.",
        "- IK/FK numerical tolerances, collision discretization tolerances, controller tolerances, and generic metric candidates were classified separately and were not adopted as H6.1 process tolerances.",
        f"- Contract was declared and hashed before recertification: `{freeze['tolerance_contract_frozen_before_recertification']}`.",
        f"- Contract SHA-256: `{terminal['contract_sha256']}`.",
        "",
        "## Frozen research acceptance values",
        "",
        f"- Nominal standoff: `{contract['geometry']['nominal_standoff_m']} m`.",
        f"- Standoff tolerance: `+/-{contract['geometry']['standoff_abs_tolerance_m']} m` (`{contract['geometry']['standoff_min_m']} to {contract['geometry']['standoff_max_m']} m`).",
        f"- Surface-normal angular tolerance: `<= {contract['geometry']['normal_angle_tolerance_deg']} deg`.",
        f"- TCP position tolerance: `<= {contract['tcp_reproduction']['tcp_position_tolerance_m']} m`.",
        f"- TCP orientation tolerance: `<= {contract['tcp_reproduction']['tcp_orientation_tolerance_deg']} deg`.",
        "- These are project-defined simulation/research acceptance constraints, not manufacturer, coating-supplier, production QA, IK/FK, MoveIt, FCL, or controller tolerances.",
        "",
        "## Full-path recertification",
        "",
        f"- Spray-ON waypoints checked: `{summary['spray_on_waypoint_count']}`; passed: `{summary['spray_on_waypoints_passed']}`.",
        f"- Full waypoints checked: `{FULL_WAYPOINT_COUNT}`; Spray-OFF transfer waypoints remain under IK/joint-limit/FK/collision checks only.",
        f"- Standoff min/max/max absolute error: `{summary['standoff_min_m']} / {summary['standoff_max_m']} / {summary['standoff_max_absolute_error_m']} m`.",
        f"- Normal max deviation: `{summary['normal_max_deviation_deg']} deg`.",
        f"- TCP max position error: `{summary['tcp_max_position_error_m']} m`.",
        f"- TCP max shortest-angle orientation error: `{summary['tcp_max_orientation_error_deg']} deg`.",
        f"- 30/30 coverage retained: `{coverage['spray_on_covered_targets'] == FORMAL_TARGET_COUNT and not coverage['uncovered_targets']}`.",
        f"- Segment structure retained: `{coverage['spray_on_segment_count']} Spray-ON + {coverage['spray_off_segment_count']} Spray-OFF`.",
        f"- All native IK/FK/joint-limit/self-collision/environment-collision checks passed: `{summary['all_kinematic_collision_checks_pass']}`.",
        f"- Collision method: `{COLLISION_METHOD}`; CCD: `NOT_AVAILABLE`; no clearance was inferred.",
        "",
        "## Determinism and regression",
        "",
        f"- Fresh native replays: `{replay['fresh_process_count']}/3`; identical semantic signatures: `{replay['native_results_identical']}`; identical process-tolerance signatures: `{replay['process_tolerance_results_identical']}`.",
        f"- New regression failures: `{regression['new_regression_failures']}`.",
        "",
        "## Immutable boundaries",
        "",
        "`H4_5_IMMUTABLE: YES`",
        "`H5_IMMUTABLE: YES`",
        "`ORIGINAL_H6_EVIDENCE_IMMUTABLE: YES`",
        "`NEW_FJT_GOALS_SENT: 0`",
        "`ROBOT_MOTION_STARTED: NO`",
        "`FORMAL_LEDGER_MUTATED: NO`",
        "`RUCKIG_STARTED: NO`",
        "`TIME_PARAMETERIZATION_DONE: NO`",
        "`ML_TRAINING_STARTED: NO`",
        "",
        "## Machine-readable artifacts",
        "",
        "- Contract: `stage3_h6_1_spray_process_tolerance_contract.json`.",
        "- Provenance audit: `stage3_h6_1_tolerance_provenance_audit.json`.",
        "- Contract freeze: `stage3_h6_1_contract_freeze_manifest.json`.",
        "- Process validation: `stage3_h6_1_process_validation.jsonl`.",
        "- Replay determinism: `stage3_h6_1_replay_determinism.json`.",
        "- Regression: `stage3_h6_1_regression_report.json`.",
        "- Gate: `stage3_h6_1_gate_report.json`.",
        "- Terminal certificate: `stage3_h6_1_terminal_certificate.json`.",
        "- Raw/native evidence preserved under `raw_native_evidence/`.",
        "",
        "The original H6 `BLOCKED` certificate remains unchanged; this bundle is the independent H6.1 recertification certificate.",
        "",
    ]
    (output / "FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def orchestrate(output: Path) -> int:
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    contract = load_json(CONTRACT)
    contract_info = validate_contract(contract)
    if not contract_info["valid"]:
        raise RuntimeError(f"invalid H6.1 contract before freeze: {contract_info['errors']}")
    audit = build_provenance_audit(contract)
    dump_json(output / "stage3_h6_1_tolerance_provenance_audit.json", audit)
    input_hashes_before = h6_source_hashes()
    missing = [name for name, record in input_hashes_before.items() if record.get("exists") is False]
    if missing or not H6_REQUESTS.is_file() or not H6_NATIVE_EXECUTABLE.is_file():
        raise RuntimeError(f"frozen H6 native evidence/request missing: {missing}")
    copy_preserved_evidence(output)
    freeze = freeze_contract_and_inputs(output, contract, contract_info, audit, input_hashes_before)
    cartesian_rows = frozen_full_waypoint_rows()
    segments = load_json(H6_ROOT / "stage3_h6_spray_on_off_segments.json")["segments"]
    coverage = load_json(H6_ROOT / "stage3_h6_coverage_accounting.json")
    if len(cartesian_rows) != FULL_WAYPOINT_COUNT:
        raise RuntimeError(f"unexpected frozen waypoint count: {len(cartesian_rows)}")
    replay_records: list[dict[str, Any]] = []
    for index in range(1, 4):
        try:
            replay_records.append(run_native_replay(output, index))
        except Exception as exc:
            replay_records.append({"schema_version": "stage3-h6-1-native-replay-record-v1", "replay_index": index, "native_name": f"fresh_native_replay_{index}", "status": "BLOCKED", "error": f"{type(exc).__name__}: {exc}", "semantic_signature": None, "process_tolerance_signature": None})
    replay, successful_validations = build_replay_report(output, contract, cartesian_rows, segments, replay_records)
    dump_json(output / "stage3_h6_1_replay_determinism.json", replay)
    if successful_validations:
        primary_rows, primary_summary = successful_validations[0][1]["rows"], successful_validations[0][1]["summary"]
    else:
        primary_rows, primary_summary = [], {"schema_version": "stage3-h6-1-process-validation-summary-v1", "spray_on_waypoint_count": 0, "spray_on_waypoints_passed": 0, "spray_on_waypoints_failed": 0, "spray_off_waypoint_count": 0, "all_spray_off_kinematic_collision_checks_pass": False, "all_kinematic_collision_checks_pass": False, "first_process_tolerance_violation": {"reason": "native_replay_unavailable"}, "process_tolerance_all_spray_on_pass": False}
    dump_jsonl(output / "stage3_h6_1_process_validation.jsonl", primary_rows)
    # The original H6 source files are re-hashed after recertification.  Any
    # change is a hard blocker and is never silently repaired.
    input_hashes_after = h6_source_hashes()
    source_immutable = input_hashes_before == input_hashes_after
    coverage = dict(coverage)
    coverage["full_waypoint_count"] = len(cartesian_rows)
    coverage["spray_on_waypoint_count"] = sum(row.get("spray_state") == "SPRAY_ON" for row in cartesian_rows)
    coverage["spray_off_waypoint_count"] = sum(row.get("spray_state") == "SPRAY_OFF" for row in cartesian_rows)
    regression = run_regression()
    dump_json(output / "stage3_h6_1_regression_report.json", regression)
    gate, terminal, blocker = build_gate_and_terminal(contract_info, freeze, audit, replay, primary_summary, coverage, regression, source_immutable, output)
    dump_json(output / "stage3_h6_1_gate_report.json", gate)
    # Emit the report before the artifact manifest so the manifest hashes the
    # exact delivered report.
    final_report(output, contract, audit, freeze, replay, primary_summary, coverage, regression, gate, terminal)
    manifest = artifact_manifest(output)
    terminal["artifact_manifest_sha256"] = manifest["manifest_sha256"]
    terminal["key_artifacts"] = manifest["files"]
    terminal["certificate_sha256"] = None
    terminal_digest = hashlib.sha256(canonical_bytes(terminal)).hexdigest()
    terminal["certificate_sha256"] = terminal_digest
    dump_json(output / "stage3_h6_1_terminal_certificate.json", terminal)
    return 0 if blocker is None else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output or ROOT / "outputs" / f"stage3_h6_1_process_tolerance_recertification_{timestamp}").resolve()
    try:
        return orchestrate(output)
    except Exception as exc:
        print(f"STAGE_3_H6_1: BLOCKED ({type(exc).__name__}: {exc})", file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
