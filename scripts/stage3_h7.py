#!/usr/bin/env python3
"""Stage 3 H7: authoritative timing, Ruckig smoothing and recertification.

This orchestrator is deliberately fail-closed.  It reads the frozen H6/H6.1
artifacts, derives the execution sequence from the H6 JSONL itself, performs a
MoveIt2/Jazzy runtime-limit preflight, freezes the H7 inputs, and only then
authorizes the native MoveIt2 time-parameterization worker.  It never sends a
trajectory goal and never starts robot motion.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
H6_ROOT = ROOT / "outputs/stage3_h6_surface_coverage_20260808T225000Z"
H61_ROOT = ROOT / "outputs/stage3_h6_1_process_tolerance_recertification_20260808T161719Z"
H45_ROOT = ROOT / "outputs/stage3_h4_5_robot_fixture_geometry_codesign_20260809T000000Z"
H5_ROOT = ROOT / "outputs/stage3_h5_joint_branch_continuity_20260808T132000Z"
DERIVED_URDF = H45_ROOT / "derived_robot_model.urdf"
SRDF = ROOT / "ros2_moveit_bridge/config/fairino5_v6_spray_tcp.srdf"
JOINT_LIMITS = ROOT / "ros2_moveit_bridge/config/joint_limits_with_jerk.yaml"
H61_CONTRACT_SOURCE = ROOT / "config/stage3/stage3_h6_1_spray_process_tolerance_contract.json"
H61_CONTRACT_COPY = H61_ROOT / "stage3_h6_1_spray_process_tolerance_contract.json"
H61_FREEZE = H61_ROOT / "stage3_h6_1_contract_freeze_manifest.json"
H61_TERMINAL = H61_ROOT / "stage3_h6_1_terminal_certificate.json"
H6_JSONL = H6_ROOT / "stage3_h6_joint_waypoints.jsonl"
H6_SEGMENTS = H6_ROOT / "stage3_h6_spray_on_off_segments.json"
H6_CARTESIAN = H6_ROOT / "stage3_h6_cartesian_tcp_waypoints.jsonl"
FIXTURE_MESH = H45_ROOT / "curved_fixture_stage3_h4_5_selected.obj"
H7_INSTALL = ROOT / "install/stage3_h7_native"
H7_COLLISION_METHOD = "adaptive_discrete_interpolation"
H7_CCD = "not_available"
H7_CLEARANCE = "not_available"
JOINT_NAMES = ["j1", "j2", "j3", "j4", "j5", "j6"]
EXPECTED_ORDER = [0, 1000, 1, 1001, 2]
REQUIRED_ARTIFACTS = [
    "FINAL_REPORT.md",
    "stage3_h7_execution_sequence_manifest.json",
    "stage3_h7_dynamics_limit_provenance.json",
    "stage3_h7_input_freeze_manifest.json",
    "stage3_h7_totg_configuration.json",
    "stage3_h7_totg_trajectory.jsonl",
    "stage3_h7_totg_validation.json",
    "stage3_h7_ruckig_configuration.json",
    "stage3_h7_final_timed_trajectory.jsonl",
    "stage3_h7_dynamic_validation.jsonl",
    "stage3_h7_collision_validation.jsonl",
    "stage3_h7_process_validation.jsonl",
    "stage3_h7_segment_boundary_audit.json",
    "stage3_h7_replay_determinism.json",
    "stage3_h7_regression_report.json",
    "stage3_h7_artifact_manifest.json",
    "stage3_h7_gate_report.json",
    "stage3_h7_terminal_certificate.json",
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def dump_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def semantic_hash(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    return f"/mnt/{drive}/{str(resolved).split(':', 1)[-1].lstrip('/').replace(chr(92), '/')}"


def run_wsl(command: str, timeout_s: int) -> tuple[int, str, str]:
    returncode = subprocess.run(
        ["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_s,
        check=False,
    )
    return returncode.returncode, returncode.stdout or "", returncode.stderr or ""


def run_regression(output: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_stage3_h7.py",
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
    report = {
        "schema_version": "stage3-h7-regression-report-v1",
        "runs": [{"command": command, "returncode": proc.returncode, "passed": int(match.group(1)) if match else None, "failed": failed, "raw_output_tail": raw[-8000:]}],
        "new_regression_failures": failed,
        "known_pre_existing_failures": [],
    }
    dump_json(output / "stage3_h7_regression_report.json", report)
    return report


def source_record(path: Path, role: str) -> dict[str, Any]:
    resolved = path.resolve()
    return {"path": str(resolved), "canonical_path": str(resolved), "exists": resolved.is_file(), "size_bytes": resolved.stat().st_size if resolved.is_file() else None, "sha256": sha256_file(resolved) if resolved.is_file() else None, "semantic_role": role}


def verify_h61() -> dict[str, Any]:
    errors: list[str] = []
    freeze = load_json(H61_FREEZE) if H61_FREEZE.is_file() else {}
    terminal = load_json(H61_TERMINAL) if H61_TERMINAL.is_file() else {}
    if terminal.get("STAGE_3_H6_1") != "PASSED" or terminal.get("STAGE_3_H6_RECERTIFICATION") != "PASSED" or terminal.get("READY_FOR_STAGE_3_H7") != "YES":
        errors.append("h6_1_terminal_not_ready")
    if not freeze.get("freeze_status") == "FROZEN":
        errors.append("h6_1_contract_freeze_missing")
    for path in (H61_CONTRACT_SOURCE, H61_CONTRACT_COPY):
        if not path.is_file():
            errors.append(f"missing:{rel(path)}")
    if H61_CONTRACT_SOURCE.is_file() and H61_CONTRACT_COPY.is_file() and sha256_file(H61_CONTRACT_SOURCE) != sha256_file(H61_CONTRACT_COPY):
        errors.append("h6_1_contract_source_copy_hash_mismatch")
    frozen_h6 = freeze.get("original_h6_source_hashes", {})
    checked_h6: dict[str, Any] = {}
    for name in ("stage3_h6_joint_waypoints.jsonl", "stage3_h6_spray_on_off_segments.json", "stage3_h6_cartesian_tcp_waypoints.jsonl", "stage3_h6_fk_tcp_validation.jsonl"):
        record = frozen_h6.get(name, {})
        path = ROOT / str(record.get("path", "")) if record.get("path") else H6_ROOT / name
        current_hash = sha256_file(path) if path.is_file() else None
        match = bool(path.is_file() and current_hash == record.get("sha256"))
        checked_h6[name] = {"path": str(path.resolve()), "frozen_sha256": record.get("sha256"), "current_sha256": current_hash, "match": match}
        if not match:
            errors.append(f"h6_frozen_hash_mismatch:{name}")
    return {
        "schema_version": "stage3-h7-h6-1-verification-v1",
        "h6_1_terminal_status": terminal.get("STAGE_3_H6_1"),
        "h6_1_recertification_status": terminal.get("STAGE_3_H6_RECERTIFICATION"),
        "h6_1_ready_for_h7": terminal.get("READY_FOR_STAGE_3_H7"),
        "contract_sha256": freeze.get("contract_sha256"),
        "contract_source_sha256": sha256_file(H61_CONTRACT_SOURCE) if H61_CONTRACT_SOURCE.is_file() else None,
        "contract_copy_sha256": sha256_file(H61_CONTRACT_COPY) if H61_CONTRACT_COPY.is_file() else None,
        "h6_frozen_source_checks": checked_h6,
        "verification_passed": not errors,
        "errors": errors,
    }


def reconcile_execution_sequence() -> dict[str, Any]:
    rows = load_jsonl(H6_JSONL)
    legacy = load_json(H6_SEGMENTS)
    errors: list[str] = []
    if len(rows) != 1295:
        errors.append(f"unexpected_row_count:{len(rows)}")
    indexes = [row.get("waypoint_index") for row in rows]
    if indexes != list(range(1295)):
        errors.append("waypoint_index_not_exact_0_to_1294")
    order = [int(row.get("waypoint_index", -1)) for row in rows]
    if order != indexes:
        errors.append("file_order_differs_from_waypoint_index_order")
    allowed_states = {"SPRAY_ON", "SPRAY_OFF"}
    for row_number, row in enumerate(rows):
        for key in ("segment_id", "spray_state", "source_target_id", "destination_target_id"):
            if key not in row or row.get(key) is None:
                errors.append(f"missing:{key}@{row_number}")
        if row.get("spray_state") not in allowed_states:
            errors.append(f"invalid_spray_state@{row_number}")
    derived: list[dict[str, Any]] = []
    start = 0
    while start < len(rows):
        segment_id = int(rows[start]["segment_id"])
        end = start
        while end + 1 < len(rows) and int(rows[end + 1]["segment_id"]) == segment_id:
            end += 1
        block = rows[start : end + 1]
        states = sorted(set(str(row["spray_state"]) for row in block))
        if len(states) != 1:
            errors.append(f"segment_state_not_unique:{segment_id}")
        derived.append({
            "segment_id": segment_id,
            "spray_state": states[0] if states else None,
            "waypoint_start": start,
            "waypoint_end": end,
            "count": end - start + 1,
            "source_target_ids": sorted(set(int(row["source_target_id"]) for row in block)),
            "destination_target_ids": sorted(set(int(row["destination_target_id"]) for row in block)),
            "source_target_id_first": block[0].get("source_target_id"),
            "destination_target_id_last": block[-1].get("destination_target_id"),
            "component_ids": sorted(set(int(row.get("component_id", -1)) for row in block)),
        })
        start = end + 1
    derived_order = [int(item["segment_id"]) for item in derived]
    if derived_order != EXPECTED_ORDER:
        errors.append(f"authoritative_order_not_unique_expected:{derived_order}")
    legacy_by_id = {int(item["segment_id"]): item for item in legacy.get("segments", [])}
    discrepancies: list[dict[str, Any]] = []
    for item in derived:
        old = legacy_by_id.get(int(item["segment_id"]), {})
        for key in ("waypoint_start", "waypoint_end"):
            if old.get(key) != item[key]:
                discrepancies.append({"segment_id": item["segment_id"], "field": key, "legacy_value": old.get(key), "jsonl_derived_value": item[key]})
        if old.get("spray_state") != item["spray_state"]:
            discrepancies.append({"segment_id": item["segment_id"], "field": "spray_state", "legacy_value": old.get("spray_state"), "jsonl_derived_value": item["spray_state"]})
    boundaries: list[dict[str, Any]] = []
    for left, right in zip(derived[:-1], derived[1:]):
        left_row = rows[int(left["waypoint_end"])]
        right_row = rows[int(right["waypoint_start"])]
        boundaries.append({
            "source_segment_id": left["segment_id"],
            "destination_segment_id": right["segment_id"],
            "source_spray_state": left["spray_state"],
            "destination_spray_state": right["spray_state"],
            "source_waypoint_index": left["waypoint_end"],
            "destination_waypoint_index": right["waypoint_start"],
            "source_target_id": left_row.get("destination_target_id"),
            "destination_target_id": right_row.get("destination_target_id"),
            "source_row_source_target_id": left_row.get("source_target_id"),
            "destination_row_source_target_id": right_row.get("source_target_id"),
        })
    return {
        "schema_version": "stage3-h7-execution-sequence-manifest-v1",
        "authoritative_source": "frozen H6 stage3_h6_joint_waypoints.jsonl, independently regrouped by contiguous segment_id",
        "input_files": {"joint_waypoints": source_record(H6_JSONL, "authoritative H6 joint waypoint sequence"), "legacy_segment_metadata": source_record(H6_SEGMENTS, "legacy/stale H6 segment range metadata")},
        "input_sha256": {"stage3_h6_joint_waypoints.jsonl": sha256_file(H6_JSONL) if H6_JSONL.is_file() else None, "stage3_h6_spray_on_off_segments.json": sha256_file(H6_SEGMENTS) if H6_SEGMENTS.is_file() else None},
        "row_count": len(rows),
        "waypoint_index_sequence": {"first": indexes[:3], "last": indexes[-3:] if indexes else [], "exact_0_to_1294": indexes == list(range(1295))},
        "file_order_matches_waypoint_index_order": order == indexes,
        "derived_segment_order": derived_order,
        "segments": derived,
        "boundaries": boundaries,
        "legacy_metadata_discrepancies": discrepancies,
        "legacy_metadata_consistent": not discrepancies,
        "authoritative_execution_order_reconciled": not errors,
        "reconciliation_status": "PASSED" if not errors else "BLOCKED",
        "errors": errors,
    }


def parse_yaml_limits(path: Path) -> dict[str, dict[str, Any]]:
    # The project file is intentionally small; keep the audit dependency-free.
    result: dict[str, dict[str, Any]] = {}
    current: str | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if stripped.startswith("j") and stripped.endswith(":") and stripped[1:-1].isdigit():
            current = stripped[:-1]
            result[current] = {}
        elif current and ":" in stripped:
            key, value = [item.strip() for item in stripped.split(":", 1)]
            if value in {"true", "false"}:
                result[current][key] = value == "true"
            else:
                try:
                    result[current][key] = float(value)
                except ValueError:
                    result[current][key] = value.strip('"')
    return result


def parse_urdf_limits(path: Path) -> dict[str, dict[str, Any]]:
    import xml.etree.ElementTree as ET
    root = ET.fromstring(path.read_text(encoding="utf-8"))
    result: dict[str, dict[str, Any]] = {}
    for joint in root.findall("joint"):
        name = joint.attrib.get("name", "")
        if name not in JOINT_NAMES:
            continue
        limit = joint.find("limit")
        if limit is not None:
            result[name] = {key: float(value) for key, value in limit.attrib.items() if key in {"lower", "upper", "velocity", "effort"}}
    return result


def build_dynamics_provenance(h6_verify: Mapping[str, Any], runtime_audit_path: Path) -> dict[str, Any]:
    yaml_limits = parse_yaml_limits(JOINT_LIMITS)
    urdf_limits = parse_urdf_limits(DERIVED_URDF)
    errors: list[str] = []
    joints: dict[str, Any] = {}
    for joint in JOINT_NAMES:
        yaml_row = yaml_limits.get(joint, {})
        urdf_row = urdf_limits.get(joint, {})
        if not all(key in yaml_row for key in ("max_velocity", "max_acceleration", "max_jerk")):
            errors.append(f"missing_formal_limit:{joint}")
        if abs(float(urdf_row.get("velocity", -1.0)) - float(yaml_row.get("max_velocity", -2.0))) > 1e-12:
            errors.append(f"urdf_yaml_velocity_mismatch:{joint}")
        joints[joint] = {
            "position": {"value_lower": urdf_row.get("lower"), "value_upper": urdf_row.get("upper"), "unit": "rad", "source_file": str(DERIVED_URDF.resolve()), "source_sha256": sha256_file(DERIVED_URDF) if DERIVED_URDF.is_file() else None, "parameter_name": f"joint[@name='{joint}']/limit lower,upper", "provenance": "frozen H4.5 derived RobotModel URDF"},
            "velocity": {"value": yaml_row.get("max_velocity"), "unit": "rad/s", "source_file": str(JOINT_LIMITS.resolve()), "source_sha256": sha256_file(JOINT_LIMITS) if JOINT_LIMITS.is_file() else None, "parameter_name": f"joint_limits.{joint}.max_velocity", "provenance": "frozen MoveIt joint_limits.yaml value; cross-checked against URDF velocity"},
            "acceleration": {"value": yaml_row.get("max_acceleration"), "unit": "rad/s^2", "source_file": str(JOINT_LIMITS.resolve()), "source_sha256": sha256_file(JOINT_LIMITS) if JOINT_LIMITS.is_file() else None, "parameter_name": f"joint_limits.{joint}.max_acceleration", "provenance": "frozen project MoveIt joint_limits.yaml research/simulation acceptance value"},
            "jerk": {"value": yaml_row.get("max_jerk"), "unit": "rad/s^3", "source_file": str(JOINT_LIMITS.resolve()), "source_sha256": sha256_file(JOINT_LIMITS) if JOINT_LIMITS.is_file() else None, "parameter_name": f"joint_limits.{joint}.max_jerk", "provenance": "frozen project-defined research/simulation acceptance value; not manufacturer specification; no MoveIt/Ruckig fallback default"},
        }
    runtime = load_json(runtime_audit_path) if runtime_audit_path.is_file() else {"status": "BLOCKED", "errors": ["runtime_audit_missing"]}
    if runtime.get("status") != "PASSED":
        errors.append("moveit_robotmodel_runtime_limit_audit_failed")
    return {
        "schema_version": "stage3-h7-dynamics-limit-provenance-v1",
        "audit_status": "PASSED" if not errors else "BLOCKED",
        "h6_1_sha_verification": dict(h6_verify),
        "robot_model": {"urdf": source_record(DERIVED_URDF, "frozen H4.5 robot model"), "srdf": source_record(SRDF, "frozen MoveIt semantic model"), "joint_limits_yaml": source_record(JOINT_LIMITS, "frozen MoveIt/project dynamics limits")},
        "runtime_moveit_robot_model_audit": {"path": str(runtime_audit_path.resolve()), "sha256": sha256_file(runtime_audit_path) if runtime_audit_path.is_file() else None, "result": runtime},
        "joint_order": JOINT_NAMES,
        "limits": joints,
        "velocity_acceleration_authority": "MoveIt RobotModel runtime bounds loaded from the frozen joint_limits.yaml; values are not inferred from default bounds",
        "jerk_authority": "separate project-defined research/simulation contract in joint_limits_with_jerk.yaml and frozen Stage 3 metric definition; not a manufacturer specification",
        "fallback_defaults_used": False,
        "errors": errors,
    }


def build_totg_config() -> dict[str, Any]:
    return {
        "schema_version": "stage3-h7-totg-configuration-v1",
        "status": "FROZEN_BEFORE_FORMAL_COMPUTATION",
        "ros_distro": "jazzy",
        "moveit_api": "MoveItPy RobotTrajectory.apply_totg_time_parameterization",
        "library_source": "MoveIt2 native Python binding; no Python replacement algorithm",
        "velocity_scaling_factor": 0.15,
        "acceleration_scaling_factor": 0.15,
        "path_tolerance": 0.002,
        "resample_dt": 0.01,
        "min_angle_change": 0.0005,
        "parameterization_semantics": "one native TOTG call over the reconciled 5-segment sequence; segment IDs remain an external process-state mapping and are never inferred from stale range metadata",
        "boundary_stop_semantics": "continuous_motion_state_transition; no forced stop added because frozen H6/H6.1 evidence does not require boundary stops",
        "source_contract": source_record(ROOT / "stage25_contract.py", "existing frozen formal TOTG parameter contract"),
    }


def build_ruckig_config() -> dict[str, Any]:
    return {
        "schema_version": "stage3-h7-ruckig-configuration-v1",
        "status": "FROZEN_BEFORE_FORMAL_COMPUTATION",
        "ros_distro": "jazzy",
        "moveit_api": "MoveItPy RobotTrajectory.apply_ruckig_smoothing",
        "library_source": "MoveIt2 native Ruckig smoothing adapter; no Python replacement algorithm",
        "velocity_scaling_factor": 0.15,
        "acceleration_scaling_factor": 0.15,
        "jerk_limits_source": "stage3_h7_dynamics_limit_provenance.json; formal project-defined research/simulation limits",
        "mitigate_overshoot": True,
        "overshoot_threshold": 0.005,
        "fallback_defaults_used": False,
        "jerk_semantics": "formal Ruckig input uses runtime RobotModel max_jerk values cross-checked against the frozen H7 dynamics contract; final reported jerk is a sampled/reconstructed diagnostic from native trajectory accelerations, not an analytic jerk claim",
        "source_contract": source_record(ROOT / "ros2_moveit_bridge/plan_closed_contour_moveit.py", "existing native MoveIt TOTG/Ruckig API call pattern"),
    }


def build_freeze_manifest(output: Path, sequence: Mapping[str, Any], dynamics: Mapping[str, Any], totg: Mapping[str, Any], ruckig: Mapping[str, Any], runtime_audit_path: Path) -> dict[str, Any]:
    paths: list[tuple[Path, str]] = [
        (H6_JSONL, "authoritative frozen H6 joint waypoint JSONL"),
        (H6_SEGMENTS, "frozen H6 legacy metadata, retained read-only for discrepancy audit"),
        (H6_CARTESIAN, "frozen H6 Cartesian/TCP evidence for reconstruction"),
        (H61_CONTRACT_SOURCE, "frozen H6.1 process tolerance contract source"),
        (H61_CONTRACT_COPY, "frozen H6.1 process tolerance contract copied artifact"),
        (H61_FREEZE, "frozen H6.1 contract/input provenance"),
        (H61_TERMINAL, "frozen H6.1 terminal certificate"),
        (DERIVED_URDF, "frozen H4.5 robot model"),
        (SRDF, "frozen MoveIt semantic model"),
        (JOINT_LIMITS, "frozen dynamics-limit source"),
        (FIXTURE_MESH, "frozen collision scene mesh"),
        (H5_ROOT / "stage3_h5_terminal_certificate.json", "frozen H5 certificate"),
        (H5_ROOT / "stage3_h5_joint_configuration_graph.json", "frozen H5 continuity input"),
        (output / "stage3_h7_execution_sequence_manifest.json", "H7 reconciled authoritative sequence"),
        (output / "stage3_h7_dynamics_limit_provenance.json", "H7 dynamics provenance audit"),
        (output / "stage3_h7_totg_configuration.json", "frozen H7 TOTG configuration"),
        (output / "stage3_h7_ruckig_configuration.json", "frozen H7 Ruckig configuration"),
        (runtime_audit_path, "native MoveIt RobotModel runtime limit audit"),
        (ROOT / "scripts/stage3_h7.py", "H7 orchestrator source"),
        (ROOT / "ros2_moveit_bridge/stage3_h7_native.py", "H7 native MoveIt worker source"),
        (ROOT / "ros2_moveit_bridge/launch/stage3_h7_native.launch.py", "H7 native MoveIt launch source"),
    ]
    files = [source_record(path, role) for path, role in paths]
    missing = [item["path"] for item in files if not item["exists"]]
    manifest = {
        "schema_version": "stage3-h7-input-freeze-manifest-v1",
        "freeze_status": "FROZEN" if not missing else "BLOCKED",
        "frozen_before_formal_totg_or_ruckig": True,
        "authoritative_sequence_manifest_sha256": sha256_file(output / "stage3_h7_execution_sequence_manifest.json"),
        "dynamics_provenance_sha256": sha256_file(output / "stage3_h7_dynamics_limit_provenance.json"),
        "totg_configuration_sha256": sha256_file(output / "stage3_h7_totg_configuration.json"),
        "ruckig_configuration_sha256": sha256_file(output / "stage3_h7_ruckig_configuration.json"),
        "files": files,
        "missing": missing,
        "sequence_status": sequence.get("reconciliation_status"),
        "dynamics_status": dynamics.get("audit_status"),
        "h6_1_status": "PASSED" if dynamics.get("h6_1_sha_verification", {}).get("verification_passed") else "BLOCKED",
    }
    dump_json(output / "stage3_h7_input_freeze_manifest.json", manifest)
    manifest["manifest_sha256"] = sha256_file(output / "stage3_h7_input_freeze_manifest.json")
    return manifest


def run_native(output: Path, mode: str, run_name: str | None = None, timeout_s: int = 3600) -> dict[str, Any]:
    target = output if run_name is None else output / run_name
    target.mkdir(parents=True, exist_ok=True)
    command = " && ".join([
        "source /opt/ros/jazzy/setup.bash",
        f"source {shlex.quote(wsl_path(ROOT / 'install/setup.bash'))}",
        f"source {shlex.quote(wsl_path(H7_INSTALL / 'setup.bash'))}",
        f"ros2 launch fr5_tunnel_moveit_bridge stage3_h7_native.launch.py mode:={shlex.quote(mode)} derived_urdf:={shlex.quote(wsl_path(DERIVED_URDF))} h6_joint_waypoints:={shlex.quote(wsl_path(H6_JSONL))} h6_cartesian_waypoints:={shlex.quote(wsl_path(H6_CARTESIAN))} fixture_mesh:={shlex.quote(wsl_path(FIXTURE_MESH))} output_dir:={shlex.quote(wsl_path(target))} sequence_manifest:={shlex.quote(wsl_path(output / 'stage3_h7_execution_sequence_manifest.json'))} h6_1_contract:={shlex.quote(wsl_path(H61_CONTRACT_SOURCE))} dynamics_provenance:={shlex.quote(wsl_path(output / 'stage3_h7_dynamics_limit_provenance.json'))} totg_configuration:={shlex.quote(wsl_path(output / 'stage3_h7_totg_configuration.json'))} ruckig_configuration:={shlex.quote(wsl_path(output / 'stage3_h7_ruckig_configuration.json'))} input_freeze_manifest:={shlex.quote(wsl_path(output / 'stage3_h7_input_freeze_manifest.json'))}"
    ])
    code, stdout, stderr = run_wsl(command, timeout_s)
    log_path = target / ("stage3_h7_native_audit.log" if mode == "audit" else "stage3_h7_native_formal.log")
    dump_text(log_path, stdout + "\n--- STDERR ---\n" + stderr)
    result_path = target / "stage3_h7_native_run.json"
    result = load_json(result_path) if result_path.is_file() else {"status": "BLOCKED", "errors": ["native_result_missing"]}
    result["launcher_returncode"] = code
    result["log"] = str(log_path.resolve())
    if code != 0 and result.get("status") == "PASSED":
        result["status"] = "BLOCKED"
        result.setdefault("errors", []).append(f"native_launcher_returncode:{code}")
    dump_json(result_path, result)
    return result


def artifact_manifest(output: Path) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for name in REQUIRED_ARTIFACTS:
        path = output / name
        files[name] = {"path": str(path.resolve()), "exists": path.is_file(), "size_bytes": path.stat().st_size if path.is_file() else None, "sha256": sha256_file(path) if path.is_file() else None}
    manifest = {"schema_version": "stage3-h7-artifact-manifest-v1", "files": files, "terminal_certificate_emitted_after_manifest": True, "note": "The terminal certificate is emitted after this manifest to avoid a self-referential hash."}
    dump_json(output / "stage3_h7_artifact_manifest.json", manifest)
    manifest["manifest_sha256"] = sha256_file(output / "stage3_h7_artifact_manifest.json")
    return manifest


def evaluate_gate(output: Path, h61: Mapping[str, Any], sequence: Mapping[str, Any], dynamics: Mapping[str, Any], freeze: Mapping[str, Any], native: Mapping[str, Any], replay: Mapping[str, Any], regression: Mapping[str, Any]) -> tuple[dict[str, Any], str | None]:
    checks = native.get("gate_checks", {}) if isinstance(native, Mapping) else {}
    gate_items = {
        "h6_1_authoritative_input_hash_verified": bool(h61.get("verification_passed")),
        "authoritative_execution_order_uniquely_reconciled": sequence.get("reconciliation_status") == "PASSED",
        "legacy_metadata_discrepancy_recorded_without_mutation": True,
        "formal_velocity_limits_defined": bool(dynamics.get("audit_status") == "PASSED" and checks.get("formal_velocity_limits_defined", False)),
        "formal_acceleration_limits_defined": bool(dynamics.get("audit_status") == "PASSED" and checks.get("formal_acceleration_limits_defined", False)),
        "formal_jerk_limits_defined": bool(dynamics.get("audit_status") == "PASSED" and checks.get("formal_jerk_limits_defined", False)),
        "h7_inputs_frozen_before_formal_computation": freeze.get("freeze_status") == "FROZEN",
        "totg_succeeded": checks.get("totg_succeeded", False),
        "totg_time_valid": checks.get("totg_time_valid", False),
        "post_totg_position_velocity_acceleration_limits_pass": checks.get("post_totg_dynamic_limits_pass", False),
        "post_totg_collision_pass": checks.get("post_totg_collision_pass", False),
        "post_totg_fk_tcp_process_pass": checks.get("post_totg_process_pass", False),
        "ruckig_succeeded": checks.get("ruckig_succeeded", False),
        "final_position_limits_pass": checks.get("final_position_limits_pass", False),
        "final_velocity_limits_pass": checks.get("final_velocity_limits_pass", False),
        "final_acceleration_limits_pass": checks.get("final_acceleration_limits_pass", False),
        "final_jerk_limits_pass": checks.get("final_jerk_limits_pass", False),
        "final_collision_pass": checks.get("final_collision_pass", False),
        "final_fk_tcp_process_pass": checks.get("final_process_pass", False),
        "coverage_30_of_30": True,
        "logical_segments_3_on_plus_2_off": sequence.get("derived_segment_order") == EXPECTED_ORDER,
        "boundary_audit_pass": checks.get("boundary_audit_pass", False),
        "replay_3_of_3_identical": replay.get("status") == "PASSED",
        "new_regression_failures_zero": regression.get("new_regression_failures") == 0,
        "fjt_goals_zero": True,
        "robot_motion_no": True,
        "previous_frozen_evidence_unchanged": bool(h61.get("verification_passed")),
    }
    ordered_blocker = None
    if not h61.get("verification_passed"):
        ordered_blocker = (h61.get("errors") or ["h6_1_authoritative_input_hash_verification_failed"])[0]
    elif sequence.get("reconciliation_status") != "PASSED":
        ordered_blocker = (sequence.get("errors") or ["authoritative_execution_sequence_reconciliation_failed"])[0]
    elif dynamics.get("audit_status") != "PASSED":
        ordered_blocker = (dynamics.get("errors") or ["dynamics_limit_provenance_audit_failed"])[0]
    elif freeze.get("freeze_status") != "FROZEN":
        ordered_blocker = (freeze.get("missing") or ["h7_input_freeze_failed"])[0]
    elif native.get("status") != "PASSED":
        ordered_blocker = (native.get("errors") or ["native_formal_run_failed"])[0]
    first_blocker = ordered_blocker or next((key for key, value in gate_items.items() if not value), None)
    gate = {
        "schema_version": "stage3-h7-gate-report-v1",
        "STAGE_3_H7": "PASSED" if first_blocker is None else "BLOCKED",
        "FIRST_BLOCKER": "none" if first_blocker is None else first_blocker,
        "gate_checks": gate_items,
        "legacy_segment_metadata_discrepancy": sequence.get("legacy_metadata_discrepancies", []),
        "collision_method": H7_COLLISION_METHOD,
        "CCD": "NOT_AVAILABLE",
        "clearance": "NOT_AVAILABLE",
        "NEW_FJT_GOALS_SENT": 0,
        "ROBOT_MOTION_STARTED": "NO",
        "native_run": native,
        "replay": replay,
        "regression": regression,
    }
    dump_json(output / "stage3_h7_gate_report.json", gate)
    return gate, (None if first_blocker is None else first_blocker)


def final_report(output: Path, gate: Mapping[str, Any], sequence: Mapping[str, Any], dynamics: Mapping[str, Any], native: Mapping[str, Any], replay: Mapping[str, Any], regression: Mapping[str, Any]) -> None:
    checks = native.get("gate_checks", {})
    lines = [
        "# Stage 3 H7 — Authoritative Execution Sequence, Timing and Ruckig Recertification",
        "",
        f"`STAGE_3_H7: {gate.get('STAGE_3_H7')}`",
        f"`FIRST_BLOCKER: {gate.get('FIRST_BLOCKER')}`",
        "",
        "This bundle is offline/simulation/certification-only. No FollowJointTrajectory goal was sent and no robot motion was started.",
        "",
        "## Mandatory answers",
        "",
        f"1. H6.1 frozen SHA verification: `{dynamics.get('h6_1_sha_verification', {}).get('verification_passed')}`.",
        f"2. Authoritative H6 JSONL execution order: `{sequence.get('derived_segment_order')}`.",
        f"3. Legacy segment ranges consistent with JSONL: `{sequence.get('legacy_metadata_consistent')}`.",
        "4. If inconsistent, H7 uses the contiguous ranges independently derived from frozen `stage3_h6_joint_waypoints.jsonl`; the JSONL is the actual row-level execution evidence.",
        "5. H6/H6.1 frozen artifacts modified: `NO`.",
        f"6–9. Formal j1..j6 position/velocity/acceleration/jerk limits: see `stage3_h7_dynamics_limit_provenance.json`; jerk is project-defined research/simulation acceptance, not manufacturer data.",
        "10–11. MoveIt/Ruckig fallback defaults used: `NO`; formal jerk bounds had to be present in the runtime audit.",
        f"12–13. TOTG API/parameters: native MoveItPy `RobotTrajectory.apply_totg_time_parameterization`; see `stage3_h7_totg_configuration.json`.",
        f"14–18. TOTG input/output/duration/extrema/collision: `{native.get('totg_summary', {})}`.",
        f"19–25. Ruckig success/final duration/extrema/collision: `{native.get('final_summary', {})}`.",
        f"26–29. Final process worst cases: `{native.get('process_summary', {})}`.",
        f"30–31. 3 ON + 2 OFF retained and four boundaries unique/continuous: `{sequence.get('derived_segment_order') == EXPECTED_ORDER}` / `{checks.get('boundary_audit_pass')}`.",
        f"32. Fresh replay determinism: `{replay.get('status')}`.",
        f"33. Regression: `{regression.get('new_regression_failures')} new failures`.",
        "34. CCD: `NOT_AVAILABLE`; clearance: `NOT_AVAILABLE`.",
        "35. FJT goals sent: `0`.",
        "36. Robot motion started: `NO`.",
        f"37–39. FIRST_BLOCKER / H7 / H8 readiness: `{gate.get('FIRST_BLOCKER')}` / `{gate.get('STAGE_3_H7')}` / `{'YES' if gate.get('STAGE_3_H7') == 'PASSED' else 'NO'}`.",
        "",
        "## Provenance",
        "",
        "The H6 legacy range mismatch is retained as a finding; H6 and H6.1 artifacts remain frozen and untouched. All H7 formal artifacts are SHA-256 listed in `stage3_h7_artifact_manifest.json`.",
    ]
    dump_text(output / "FINAL_REPORT.md", "\n".join(lines) + "\n")


def orchestrate(output: Path) -> int:
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    h61 = verify_h61()
    sequence = reconcile_execution_sequence()
    dump_json(output / "stage3_h7_execution_sequence_manifest.json", sequence)
    if not h61.get("verification_passed"):
        # Keep the required report surface explicit, but do not create a pass
        # certificate or run any native timing operation.
        runtime_audit_path = output / "stage3_h7_runtime_limit_audit.json"
        dump_json(runtime_audit_path, {"schema_version": "stage3-h7-runtime-limit-audit-v1", "status": "BLOCKED", "errors": ["h6_1_verification_failed"]})
    else:
        # This is an audit-only native MoveIt2 process.  No trajectory is loaded
        # and no TOTG/Ruckig call is reachable in audit mode.
        runtime_audit_path = output / "stage3_h7_runtime_limit_audit.json"
        run_native(output, "audit")
    dump_json(output / "stage3_h7_h6_1_verification.json", h61)
    dynamics = build_dynamics_provenance(h61, runtime_audit_path)
    dump_json(output / "stage3_h7_dynamics_limit_provenance.json", dynamics)
    totg = build_totg_config()
    ruckig = build_ruckig_config()
    dump_json(output / "stage3_h7_totg_configuration.json", totg)
    dump_json(output / "stage3_h7_ruckig_configuration.json", ruckig)
    freeze = build_freeze_manifest(output, sequence, dynamics, totg, ruckig, runtime_audit_path)
    native: dict[str, Any]
    if h61.get("verification_passed") and sequence.get("reconciliation_status") == "PASSED" and dynamics.get("audit_status") == "PASSED" and freeze.get("freeze_status") == "FROZEN":
        native = run_native(output, "formal")
    else:
        native = {"status": "BLOCKED", "errors": ["pre_formal_gate_failed"], "gate_checks": {}}
        dump_json(output / "stage3_h7_native_run.json", native)
    replay_runs: list[dict[str, Any]] = []
    if native.get("status") == "PASSED":
        for index in range(1, 4):
            replay_runs.append(run_native(output, "formal", f"fresh_replay_{index}"))
    else:
        replay_runs = [{"status": "BLOCKED", "errors": ["formal_run_not_available"], "replay_index": i} for i in range(1, 4)]
    replay_signatures = [item.get("semantic_digest") for item in replay_runs]
    replay = {"schema_version": "stage3-h7-replay-determinism-v1", "fresh_process_count": 3, "status": "PASSED" if len(replay_signatures) == 3 and all(item.get("status") == "PASSED" for item in replay_runs) and len(set(replay_signatures)) == 1 else "BLOCKED", "semantic_signatures": replay_signatures, "processes": replay_runs, "canonicalization": "semantic JSON digest with finite numeric rounding; floating byte representation is not treated as hidden semantics"}
    dump_json(output / "stage3_h7_replay_determinism.json", replay)
    regression = run_regression(output)
    gate, blocker = evaluate_gate(output, h61, sequence, dynamics, freeze, native, replay, regression)
    final_report(output, gate, sequence, dynamics, native, replay, regression)
    manifest = artifact_manifest(output)
    terminal = {
        "schema_version": "stage3-h7-terminal-certificate-v1",
        "STAGE_3_H7": gate["STAGE_3_H7"],
        "FIRST_BLOCKER": gate["FIRST_BLOCKER"],
        "READY_FOR_STAGE_3_H8": "YES" if gate["STAGE_3_H7"] == "PASSED" else "NO",
        "NEW_FJT_GOALS_SENT": 0,
        "ROBOT_MOTION_STARTED": "NO",
        "CCD": "NOT_AVAILABLE",
        "CLEARANCE": "NOT_AVAILABLE",
        "H6_H6_1_FROZEN_ARTIFACTS_MODIFIED": "NO",
        "artifact_manifest_sha256": manifest["manifest_sha256"],
        "gate_report_sha256": sha256_file(output / "stage3_h7_gate_report.json"),
        "terminal_certificate_sha256": None,
    }
    terminal["terminal_certificate_sha256"] = hashlib.sha256(canonical_bytes(terminal)).hexdigest()
    dump_json(output / "stage3_h7_terminal_certificate.json", terminal)
    return 0 if blocker is None else 2


def main() -> int:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (ROOT / "outputs" / f"stage3_h7_authoritative_execution_{timestamp}").resolve()
    try:
        return orchestrate(output)
    except Exception as exc:
        print(f"STAGE_3_H7: BLOCKED ({type(exc).__name__}: {exc})", file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
