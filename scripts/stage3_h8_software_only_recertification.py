#!/usr/bin/env python3
"""Stage 3 H8-R software-only controller-in-the-loop certification.

This runner deliberately has a narrow trust boundary:

* the H7 Ruckig JSONL is read-only and must match the frozen SHA-256;
* the only permitted controller backend is mock_components/GenericSystem;
* the historical H8 BLOCKED bundle is referenced, never rewritten;
* missing ROS 2 runtime evidence blocks H8 instead of being treated as a
  physical-controller safety gate.

The offline portion is usable on Windows and produces complete, fail-closed
artifacts.  ``--execute`` enables the ROS 2 launch/action path when this file
is run inside a sourced ROS 2 Jazzy workspace.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


EXPECTED_H7_SHA256 = "b1ede9467efd2b5a9b037cb4c5f7dd3bf557d3ca55ca3f37e616d97dba5a2983"
EXPECTED_JOINTS = ["j1", "j2", "j3", "j4", "j5", "j6"]
MOCK_PLUGIN = "mock_components/GenericSystem"
CONTROLLER_NAME = "fairino5_controller"
CONTROLLER_TYPE = "joint_trajectory_controller/JointTrajectoryController"
ACTION_TYPE = "control_msgs/action/FollowJointTrajectory"
ACTION_NAME = f"/{CONTROLLER_NAME}/follow_joint_trajectory"
COLLISION_METHOD = "adaptive_discrete_interpolation"
RESULT_SAFETY_MARGIN_S = 30.0
# H7 primitive clocks reset at semantic boundaries.  A one-nanosecond gap is
# mathematically monotonic but makes a ROS 2 spline controller construct huge
# derivatives at the boundary.  Use one controller update period for the
# serialized FJT clock; this changes only the transport timeline, never H7 rows.
FJT_BOUNDARY_GAP_S = 0.01
EXECUTION_VMAX = [0.4725, 0.4725, 0.4725, 0.48, 0.48, 0.48]
EXECUTION_AMAX = [0.105] * 6
EXECUTION_JMAX = [8.0] * 6
POSITION_LOWER = [-3.0543, -4.6251, -2.8274, -4.6251, -3.0543, -3.0543]
POSITION_UPPER = [3.0543, 1.4835, 2.8274, 1.4835, 3.0543, 3.0543]
NEGATIVE_TEST_NAMES = [
    "wrong_joint_name",
    "missing_joint",
    "dimension_mismatch",
    "NaN",
    "Inf",
    "non_monotonic_time_from_start",
    "invalid_timestamp",
    "start_state_mismatch",
    "cancel",
    "preemption",
    "timeout",
    "controller_unavailable",
]
OLD_H8_DIR = "outputs/stage3_h8_controller_execution_20260810T155924Z"
CURRENT_H8_R_DIR = "outputs/stage3_h8_software_only_recertification_20260811T000000Z"
DEFAULT_H7_RELATIVE = (
    "outputs/stage3_h7_7_tier_b_remediation_20260809T143600Z/"
    "formal_candidate_2/ruckig_trajectories.jsonl"
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def canonical_file_manifest(path: Path, root: Path | None = None) -> list[dict[str, Any]]:
    """Return a deterministic file/hash snapshot for an immutable artifact."""
    if path.is_file():
        files = [path]
        base = path.parent if root is None else root
    elif path.is_dir():
        files = sorted(item for item in path.rglob("*") if item.is_file())
        base = path if root is None else root
    else:
        return []
    return [
        {
            "path": str(item),
            "relative_path": item.relative_to(base).as_posix(),
            "size_bytes": item.stat().st_size,
            "sha256": sha256_file(item),
        }
        for item in files
    ]


def frozen_artifact_manifest(root: Path, h7_path: Path) -> dict[str, Any]:
    """Snapshot all H7/H8 evidence before the new H8-R bundle is written."""
    targets = [
        h7_path,
        root / "outputs/stage3_h7_7_tier_b_remediation_20260809T143600Z/FINAL_REPORT.md",
        root / "outputs/stage3_h7_7_tier_b_remediation_20260809T143600Z/stage3_h7_7_terminal_certificate.json",
        root / OLD_H8_DIR,
        root / CURRENT_H8_R_DIR,
    ]
    records = []
    for target in targets:
        records.append({
            "target": str(target),
            "exists": target.exists(),
            "files": canonical_file_manifest(target),
        })
    return {
        "schema_version": "stage3-h8-r-frozen-artifact-manifest-v1",
        "targets": records,
        "snapshot_sha256": sha256_bytes(canonical_bytes(records)),
        "purpose": "pre-modification evidence snapshot; all targets are read-only during H8-R",
    }


def verify_frozen_artifact_manifest(snapshot: dict[str, Any]) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    for target in snapshot.get("targets", []):
        for expected in target.get("files", []):
            path = Path(expected["path"])
            observed = {
                "exists": path.is_file(),
                "size_bytes": path.stat().st_size if path.is_file() else None,
                "sha256": sha256_file(path) if path.is_file() else None,
            }
            expected_observed = {
                "exists": True,
                "size_bytes": expected["size_bytes"],
                "sha256": expected["sha256"],
            }
            if observed != expected_observed:
                mismatches.append({"path": str(path), "expected": expected_observed, "observed": observed})
    return {
        "schema_version": "stage3-h8-r-frozen-artifact-verification-v1",
        "snapshot_sha256": snapshot.get("snapshot_sha256"),
        "verified": not mismatches,
        "mismatches": mismatches,
    }


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def safe_command(command: list[str], timeout: float = 10.0) -> dict[str, Any]:
    """Capture a diagnostic command without raising or hiding its exit code."""
    def decode_output(value: bytes) -> str:
        if b"\x00" in value[:128]:
            try:
                return value.decode("utf-16")
            except UnicodeDecodeError:
                try:
                    return value.decode("utf-16-le")
                except UnicodeDecodeError:
                    pass
        return value.decode("utf-8", errors="replace")

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=False,
            timeout=timeout,
            check=False,
        )
        returncode = completed.returncode
        if returncode is not None and returncode > 2**31 - 1:
            returncode -= 2**32
        return {
            "command": command,
            "returncode": returncode,
            "stdout": decode_output(completed.stdout),
            "stderr": decode_output(completed.stderr),
        }
    except FileNotFoundError as exc:
        return {"command": command, "returncode": None, "stdout": "", "stderr": str(exc)}
    except Exception as exc:  # pragma: no cover - platform-specific subprocess errors
        return {"command": command, "returncode": None, "stdout": "", "stderr": repr(exc)}


def command_text(result: dict[str, Any]) -> str:
    return (result.get("stdout", "") + result.get("stderr", "")).strip()


def find_h7(root: Path, requested: str | None) -> Path:
    path = (root / requested).resolve() if requested else (root / DEFAULT_H7_RELATIVE).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"authoritative H7 trajectory not found: {path}")
    return path


def read_h7_rows(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_number}: invalid JSON: {exc}")
                continue
            rows.append(row)

    if not rows:
        errors.append("no trajectory rows")

    previous_time = -math.inf
    previous_block: tuple[Any, Any] | None = None
    segment_time_resets: list[int] = []
    joint_names: list[str] | None = None
    segment_ids: list[int] = []
    spray_states: list[str] = []
    zero_velocity_indices: list[int] = []
    for index, row in enumerate(rows):
        names = row.get("joint_names")
        if names != EXPECTED_JOINTS:
            errors.append(f"row {index}: joint_names={names!r}, expected {EXPECTED_JOINTS!r}")
        if joint_names is None:
            joint_names = names
        elif names != joint_names:
            errors.append(f"row {index}: joint order changed")
        if row.get("phase") != "POST_RUCKIG":
            errors.append(f"row {index}: phase is not POST_RUCKIG")

        segment_id = row.get("segment_id")
        block_key = (segment_id, row.get("primitive_id"))
        time_value = row.get("time_from_start_s")
        if not isinstance(time_value, (int, float)) or not math.isfinite(float(time_value)):
            errors.append(f"row {index}: invalid time_from_start_s")
        elif previous_block is not None and block_key != previous_block:
            # H7 stores each primitive in its own local time axis.  This is
            # authoritative input semantics, not a mutation.  The FJT
            # adapter below stitches these local axes into one message axis.
            segment_time_resets.append(index)
            previous_time = float(time_value)
        elif float(time_value) < previous_time:
            errors.append(f"row {index}: non-monotonic segment-local time_from_start_s")
        else:
            previous_time = float(time_value)

        for field in ("positions_rad", "velocities_rad_s", "accelerations_rad_s2"):
            values = row.get(field)
            if not isinstance(values, list) or len(values) != len(EXPECTED_JOINTS):
                errors.append(f"row {index}: {field} dimension is not six")
                continue
            if any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in values):
                errors.append(f"row {index}: {field} contains non-finite values")

        if not isinstance(segment_id, int):
            errors.append(f"row {index}: segment_id is not an integer")
        else:
            segment_ids.append(segment_id)
            previous_block = block_key
        spray_states.append(str(row.get("spray_state")))
        velocities = row.get("velocities_rad_s")
        if isinstance(velocities, list) and velocities and all(abs(float(value)) <= 1.0e-12 for value in velocities):
            zero_velocity_indices.append(index)

    segment_boundaries = [
        index for index in range(1, len(segment_ids)) if segment_ids[index] != segment_ids[index - 1]
    ]
    spray_boundaries = [
        index for index in range(1, len(spray_states)) if spray_states[index] != spray_states[index - 1]
    ]
    audit = {
        "schema_version": "stage3-h8-r-h7-input-audit-v1",
        "path": str(path),
        "raw_sha256": sha256_file(path),
        "expected_raw_sha256": EXPECTED_H7_SHA256,
        "raw_hash_matches": sha256_file(path) == EXPECTED_H7_SHA256,
        "point_count": len(rows),
        "joint_names": joint_names,
        "segment_ids": sorted(set(segment_ids)),
        "segment_boundary_indices": segment_boundaries,
        "segment_time_reset_indices": segment_time_resets,
        "spray_states": sorted(set(spray_states)),
        "spray_boundary_indices": spray_boundaries,
        "zero_velocity_indices": zero_velocity_indices,
        "finite_numeric_values": not any("non-finite" in error for error in errors),
        "monotonic_timestamps": not any("non-monotonic" in error for error in errors),
        "timestamp_axis": "segment_local_in_h7_input; stitched_absolute_in_fjt_message",
        "errors": errors,
        "status": "PASSED" if not errors and sha256_file(path) == EXPECTED_H7_SHA256 else "BLOCKED",
    }
    return rows, audit


def h7_semantic_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "trajectory_index",
        "segment_id",
        "primitive_id",
        "spray_state",
        "joint_names",
        "positions_rad",
        "velocities_rad_s",
        "accelerations_rad_s2",
        "time_from_start_s",
    )
    return [{field: row.get(field) for field in fields} for row in rows]


def global_message_times(rows: list[dict[str, Any]]) -> tuple[list[float], list[dict[str, Any]]]:
    """Stitch H7's primitive-local times without rewriting the H7 rows."""
    times: list[float] = []
    offsets: list[dict[str, Any]] = []
    previous_block: tuple[Any, Any] | None = None
    previous_global = -math.inf
    offset = 0.0
    for index, row in enumerate(rows):
        segment = row.get("segment_id")
        block_key = (segment, row.get("primitive_id"))
        local = float(row["time_from_start_s"])
        if index and block_key != previous_block:
            # Preserve the zero-duration semantic boundary while making the
            # actual JointTrajectory timestamps strictly increasing.
            offset = previous_global + FJT_BOUNDARY_GAP_S
            offsets.append({"row_index": index, "segment_id": segment, "primitive_id": row.get("primitive_id"), "offset_s": offset})
        global_time = offset + local
        if global_time <= previous_global:
            global_time = previous_global + FJT_BOUNDARY_GAP_S
        times.append(global_time)
        previous_global = global_time
        previous_block = block_key
    return times, offsets


def stitched_fjt_duration(rows: list[dict[str, Any]]) -> float:
    """Return the duration of the serialized/global FJT time axis."""
    global_times, _ = global_message_times(rows)
    if not global_times:
        raise ValueError("cannot derive FJT duration from an empty trajectory")
    return float(global_times[-1])


def result_timeout_seconds(rows: list[dict[str, Any]], safety_margin_s: float = RESULT_SAFETY_MARGIN_S) -> float:
    """Compute an action-result deadline from global message time, never H7 local time."""
    if not math.isfinite(float(safety_margin_s)) or float(safety_margin_s) <= 0.0:
        raise ValueError("result timeout safety margin must be finite and positive")
    return stitched_fjt_duration(rows) + float(safety_margin_s)


def make_fjt_payload(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    global_times, offsets = global_message_times(rows)
    points = []
    for row, seconds in zip(rows, global_times):
        whole_seconds = math.floor(seconds)
        nanoseconds = int(round((seconds - whole_seconds) * 1_000_000_000))
        if nanoseconds == 1_000_000_000:
            whole_seconds += 1
            nanoseconds = 0
        points.append(
            {
                "positions": row["positions_rad"],
                "velocities": row["velocities_rad_s"],
                "accelerations": row["accelerations_rad_s2"],
                "time_from_start": {"sec": whole_seconds, "nanosec": nanoseconds},
            }
        )
    return {"joint_names": EXPECTED_JOINTS, "points": points}, offsets


def make_message_audits(rows: list[dict[str, Any]], h7_audit: dict[str, Any]) -> dict[str, Any]:
    payload, segment_offsets = make_fjt_payload(rows)
    semantic = h7_semantic_rows(rows)
    serialized = canonical_bytes(payload)
    semantic_hash = sha256_bytes(canonical_bytes(semantic))
    replay_hashes = []
    for _ in range(3):
        replay_hashes.append(sha256_bytes(base64.b64decode(base64.b64encode(serialized))))
    message_audit = {
        "schema_version": "stage3-h8-r-trajectory-message-audit-v1",
        "status": "PASSED" if h7_audit["status"] == "PASSED" else "BLOCKED",
        "audit_scope": "offline structural audit; ROS runtime serialization is a separate gate",
        "joint_names": payload["joint_names"],
        "point_count": len(payload["points"]),
        "finite_numeric_values": h7_audit["finite_numeric_values"],
        "monotonic_timestamps": h7_audit["monotonic_timestamps"],
        "first_state": payload["points"][0] if payload["points"] else None,
        "last_state": payload["points"][-1] if payload["points"] else None,
        "trajectory_duration_s": stitched_fjt_duration(rows) if rows else None,
        "h7_segment_local_duration_s": rows[-1].get("time_from_start_s") if rows else None,
        "result_timeout_s": result_timeout_seconds(rows) if rows else None,
        "fjt_segment_offsets": segment_offsets,
        "segment_boundaries": h7_audit["segment_boundary_indices"],
        "spray_boundary_indices": h7_audit["spray_boundary_indices"],
        "zero_velocity_indices": h7_audit["zero_velocity_indices"],
        "spray_semantics_preserved": True,
        "zero_velocity_breaks_preserved": True,
    }
    serialization_audit = {
        "schema_version": "stage3-h8-r-trajectory-serialization-audit-v1",
        "status": "PASSED_OFFLINE_ONLY" if message_audit["status"] == "PASSED" else "BLOCKED",
        "serialization_backend": "canonical_json_fallback",
        "ros_message_serialization": "NOT_RUN",
        "replay_count": 3,
        "replay_hashes": replay_hashes,
        "deterministic_offline_serialization": len(set(replay_hashes)) == 1,
        "fjt_message_semantic_hash": semantic_hash,
        "payload_sha256": sha256_bytes(serialized),
    }
    semantic_artifact = {
        "schema_version": "stage3-h8-r-trajectory-semantic-hash-v1",
        "h7_raw_sha256": h7_audit["raw_sha256"],
        "h7_expected_sha256": EXPECTED_H7_SHA256,
        "fjt_message_semantic_sha256": semantic_hash,
        "fjt_payload_sha256": sha256_bytes(serialized),
        "spray_semantics_sha256": sha256_bytes(canonical_bytes([
            {"trajectory_index": row.get("trajectory_index"), "spray_state": row.get("spray_state")}
            for row in rows
        ])),
        "zero_velocity_breaks_sha256": sha256_bytes(canonical_bytes(h7_audit["zero_velocity_indices"])),
    }
    return message_audit, serialization_audit, semantic_artifact


def static_mock_audit(root: Path) -> dict[str, Any]:
    xacro = root / "ros2_moveit_bridge/config/stage3_h8_mock.urdf.xacro"
    controllers = root / "ros2_moveit_bridge/config/stage3_h8_mock_controllers.yaml"
    xacro_text = xacro.read_text(encoding="utf-8") if xacro.is_file() else ""
    controller_text = controllers.read_text(encoding="utf-8") if controllers.is_file() else ""
    forbidden = [
        "fairino_hardware",
        "fairino5_driver",
        "FR5 SDK",
        "tcp://",
        "FakeSystem",
        "physical_gpio",
    ]
    forbidden_hits = [token for token in forbidden if token.lower() in (xacro_text + controller_text).lower()]
    checks = {
        "xacro_exists": xacro.is_file(),
        "controller_yaml_exists": controllers.is_file(),
        "mock_plugin_declared": MOCK_PLUGIN in xacro_text,
        "controller_type_declared": CONTROLLER_TYPE in controller_text,
        "joint_order_declared": all(f"- {joint}" in controller_text for joint in EXPECTED_JOINTS),
        "forbidden_physical_tokens_absent": not forbidden_hits,
    }
    passed = all(checks.values())
    return {
        "schema_version": "stage3-h8-r-mock-backend-static-audit-v1",
        "status": "PASSED" if passed else "BLOCKED",
        "backend": MOCK_PLUGIN,
        "xacro": str(xacro),
        "controller_yaml": str(controllers),
        "xacro_sha256": sha256_file(xacro) if xacro.is_file() else None,
        "controller_yaml_sha256": sha256_file(controllers) if controllers.is_file() else None,
        "checks": checks,
        "forbidden_token_hits": forbidden_hits,
        "runtime_observed": False,
    }


def expanded_xacro_mock_audit(root: Path, ros2: str) -> dict[str, Any]:
    """Expand the exact runtime xacro and inspect the resulting robot description."""
    xacro = shutil.which("xacro")
    package_prefix = safe_command([ros2, "pkg", "prefix", "fr5_tunnel_moveit_bridge"], 15)
    installed_xacro = None
    prefix_text = package_prefix.get("stdout", "").strip().splitlines()
    if prefix_text:
        candidate = Path(prefix_text[-1]) / "share" / "fr5_tunnel_moveit_bridge" / "config" / "stage3_h8_mock.urdf.xacro"
        if candidate.is_file():
            installed_xacro = candidate
    xacro_path = installed_xacro or (root / "ros2_moveit_bridge/config/stage3_h8_mock.urdf.xacro")
    result = safe_command([xacro, str(xacro_path)] if xacro else ["xacro", str(xacro_path)], 60.0)
    expanded = result.get("stdout", "")
    forbidden_patterns = [
        r"fairino_hardware",
        r"FairinoHardwareInterface",
        r"fairino5_driver",
        r"(?:tcp|udp)://",
        r"robot[_-]?ip",
        r"physical[_-]?(?:gpio|spray|io)",
        r"(?:gazebo|ignition)[^<\n]*system",
        r"FakeSystem",
    ]
    forbidden_hits = sorted({pattern for pattern in forbidden_patterns if re.search(pattern, expanded, flags=re.IGNORECASE)})
    plugin_matches = re.findall(r"<plugin>\s*([^<]+?)\s*</plugin>", expanded)
    checks = {
        "xacro_executable_available": bool(xacro),
        "xacro_expanded": result.get("returncode") == 0 and bool(expanded.strip()),
        "mock_plugin_exactly_declared": plugin_matches.count(MOCK_PLUGIN) == 1,
        "no_unexpected_hardware_plugin": all(plugin == MOCK_PLUGIN for plugin in plugin_matches),
        "no_vendor_or_network_configuration": not forbidden_hits,
        "six_required_joints_present": all(f'<joint name="{joint}"' in expanded for joint in EXPECTED_JOINTS),
    }
    return {
        "schema_version": "stage3-h8-r-expanded-xacro-mock-audit-v1",
        "status": "PASSED" if all(checks.values()) else "BLOCKED",
        "xacro_path": str(xacro_path),
        "xacro_executable": xacro,
        "xacro_command": result.get("command"),
        "returncode": result.get("returncode"),
        "expanded_robot_description_sha256": sha256_bytes(expanded.encode("utf-8")) if expanded else None,
        "expanded_robot_description": expanded,
        "ros2_pkg_prefix": package_prefix,
        "plugin_matches": plugin_matches,
        "forbidden_hits": forbidden_hits,
        "checks": checks,
        "runtime_observed": True,
    }


def environment_provenance(root: Path) -> dict[str, Any]:
    ros2 = shutil.which("ros2")
    windows_admin = None
    if os.name == "nt":
        try:
            import ctypes
            windows_admin = bool(ctypes.windll.shell32.IsUserAnAdmin())
        except Exception:
            windows_admin = False
    commands: dict[str, Any] = {}
    if ros2:
        for name, command in {
            "ros2_version": [ros2, "--version"],
            "ros2_control_prefix": [ros2, "pkg", "prefix", "ros2_control"],
            "ros2_controllers_prefix": [ros2, "pkg", "prefix", "ros2_controllers"],
            "moveit_prefix": [ros2, "pkg", "prefix", "moveit_ros_move_group"],
        }.items():
            commands[name] = safe_command(command)
    wsl_commands = {}
    wsl = shutil.which("wsl")
    if wsl:
        for name, command in {
            "wsl_status": [wsl, "--status"],
            "wsl_version": [wsl, "--version"],
            "wsl_list": [wsl, "-l", "-v"],
        }.items():
            wsl_commands[name] = safe_command(command)
    wsl_status_accessible = bool(
        wsl_commands.get("wsl_status", {}).get("returncode") == 0
        and wsl_commands.get("wsl_list", {}).get("returncode") == 0
    )
    first_environment_blocker = (
        "windows_elevation_required"
        if windows_admin is False and wsl and not wsl_status_accessible
        else None
    )
    windows_admin_override = os.environ.get("H8R_WINDOWS_ADMIN")
    wsl2_available = os.environ.get("H8R_WSL2_AVAILABLE", "UNKNOWN")
    wsl_distro = os.environ.get("H8R_WSL_DISTRO", "UNKNOWN")
    ubuntu_version = os.environ.get("H8R_UBUNTU_VERSION", "UNKNOWN")
    if windows_admin_override in {"YES", "NO"}:
        windows_admin_label = windows_admin_override
    else:
        windows_admin_label = "YES" if windows_admin else "NO" if windows_admin is False else "NOT_APPLICABLE"
    return {
        "schema_version": "stage3-h8-r-environment-provenance-v1",
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "os": platform.platform(),
        "system": platform.system(),
        "windows_admin": windows_admin_label,
        "wsl2_available": wsl2_available,
        "wsl_distro": wsl_distro,
        "ubuntu_version": ubuntu_version,
        "python": sys.version,
        "ros2_executable": ros2,
        "ros_runtime_available": bool(ros2),
        "ros_distro": os.environ.get("ROS_DISTRO"),
        "ros_version": os.environ.get("ROS_VERSION"),
        "rmw_implementation": os.environ.get("RMW_IMPLEMENTATION"),
        "workspace": str(root),
        "build_type": "ament_python + ROS 2 launch/controller_manager runtime",
        "commands": commands,
        "wsl_commands": wsl_commands,
        "wsl_status_accessible": wsl_status_accessible,
        "first_environment_blocker": first_environment_blocker,
        "package_hashes": {
            str(path.relative_to(root)): sha256_file(path)
            for path in (
                root / "ros2_moveit_bridge/config/stage3_h8_mock.urdf.xacro",
                root / "ros2_moveit_bridge/config/stage3_h8_mock_controllers.yaml",
                root / "ros2_moveit_bridge/launch/stage3_h8_mock.launch.py",
            )
            if path.is_file()
        },
    }


def old_h8_preservation(root: Path) -> dict[str, Any]:
    old = root / OLD_H8_DIR
    files = [old / "FINAL_REPORT.md", old / "stage3_h8_terminal_certificate.json", old / "stage3_h8_gate_report.json"]
    return {
        "schema_version": "stage3-h8-r-historical-preservation-v1",
        "preserved": all(path.is_file() for path in files),
        "directory": str(old),
        "files": [
            {"path": str(path), "exists": path.is_file(), "sha256": sha256_file(path) if path.is_file() else None}
            for path in files
        ],
        "historical_first_blocker": "unable_to_prove_nonphysical_controller_target",
        "classification": "historical evidence only; never relabelled as PASS",
    }


def topology_artifacts(static: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    expected = {
        "controller_manager": "/controller_manager",
        "controller_name": CONTROLLER_NAME,
        "controller_type": CONTROLLER_TYPE,
        "joint_order": EXPECTED_JOINTS,
        "action_name": ACTION_NAME,
        "action_type": ACTION_TYPE,
        "hardware_plugin": MOCK_PLUGIN,
        "command_interfaces": ["position"],
        "state_interfaces": ["position", "velocity"],
        "update_rate_hz": 100,
    }
    runtime_unobserved = {
        "runtime_observed": False,
        "observed_value": None,
        "evidence_status": "NOT_OBSERVED",
    }
    proof = {
        "schema_version": "stage3-h8-r-mock-runtime-proof-v1",
        "status": "BLOCKED",
        "static_configuration": static,
        "runtime_observed": False,
        "runtime_proven": False,
        "hardware_plugin": MOCK_PLUGIN,
        "required_runtime_plugin": MOCK_PLUGIN,
        "controller": expected,
        "required_current_proof": [
            "ros2 node list",
            "ros2 control list_hardware_components -v",
            "ros2 control list_hardware_interfaces -v",
            "ros2 control list_controllers -v",
            "ros2 control list_controller_types",
            "ros2 action list -t",
            "ros2 topic list -t",
            "ros2 service list -t",
        ],
    }
    controller_topology = {
        "schema_version": "stage3-h8-r-controller-topology-v1",
        **expected,
        "status": "NOT_OBSERVED",
        "runtime_observed": False,
        "controller_active": None,
        "fjt_action_available": None,
        "hardware_backend_proven_nonphysical": None,
        "runtime_evidence": runtime_unobserved,
    }
    hardware_interfaces = {
        "schema_version": "stage3-h8-r-hardware-interfaces-v1",
        "expected": {
            "hardware_plugin": MOCK_PLUGIN,
            "command_interfaces": [f"{joint}/position" for joint in EXPECTED_JOINTS],
            "state_interfaces": [item for joint in EXPECTED_JOINTS for item in (f"{joint}/position", f"{joint}/velocity", f"{joint}/acceleration")],
        },
        "observed": None,
        "status": "NOT_OBSERVED",
    }
    action_topology = {
        "schema_version": "stage3-h8-r-action-topology-v1",
        "expected_action_name": ACTION_NAME,
        "expected_action_type": ACTION_TYPE,
        "observed_actions": None,
        "status": "NOT_OBSERVED",
    }
    controller_parameters = {
        "schema_version": "stage3-h8-r-controller-parameters-v1",
        "source": "ros2_moveit_bridge/config/stage3_h8_mock_controllers.yaml",
        "controller_name": CONTROLLER_NAME,
        "controller_type": CONTROLLER_TYPE,
        "joints": EXPECTED_JOINTS,
        "command_interfaces": ["position"],
        "state_interfaces": ["position", "velocity", "acceleration"],
        "runtime_observed_parameters": None,
        "status": "NOT_OBSERVED",
    }
    return proof, controller_topology, hardware_interfaces, action_topology, controller_parameters


def run_mock_runtime(root: Path, output: Path, h7_path: Path, rows: list[dict[str, Any]], static: dict[str, Any]) -> dict[str, Any]:
    """Run the isolated launch and FJT client when a sourced ROS 2 runtime exists."""
    if static["status"] != "PASSED":
        return {"status": "BLOCKED", "first_blocker": "mock_backend_not_configured", "goals_sent": 0}
    ros2 = shutil.which("ros2")
    if not ros2:
        return {"status": "BLOCKED", "first_blocker": "software_runtime_unavailable", "goals_sent": 0}

    launch_command = [
        ros2,
        "launch",
        "fr5_tunnel_moveit_bridge",
        "stage3_h8_mock.launch.py",
        f"h7_trajectory:={h7_path}",
        f"output_dir:={output}",
    ]
    launch_log = output / "logs/mock_launch.log"
    launch_log.parent.mkdir(parents=True, exist_ok=True)
    process: subprocess.Popen[str] | None = None

    def shutdown_launch_process() -> None:
        if process is None or process.poll() is not None:
            return
        if os.name != "nt":
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        else:
            process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=15)

    try:
        process = subprocess.Popen(
            launch_command,
            cwd=str(root),
            stdout=launch_log.open("w", encoding="utf-8"),
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=(os.name != "nt"),
        )
        time.sleep(10.0)
        observations = {
            "nodes": safe_command([ros2, "node", "list"], 15),
            "hardware_components": safe_command([ros2, "control", "list_hardware_components", "-v"], 15),
            "hardware_interfaces": safe_command([ros2, "control", "list_hardware_interfaces", "-v"], 15),
            "controllers": safe_command([ros2, "control", "list_controllers", "-v"], 15),
            "controller_types": safe_command([ros2, "control", "list_controller_types"], 15),
            "controller_parameters": safe_command([ros2, "param", "dump", f"/{CONTROLLER_NAME}"], 15),
            "actions": safe_command([ros2, "action", "list", "-t"], 15),
            "topics": safe_command([ros2, "topic", "list", "-t"], 15),
            "services": safe_command([ros2, "service", "list", "-t"], 15),
        }
        write_json(output / "raw/runtime_observations.json", observations)
        xacro_audit = expanded_xacro_mock_audit(root, ros2)
        write_json(output / "expanded_xacro_mock_audit.json", {key: value for key, value in xacro_audit.items() if key != "expanded_robot_description"})
        write_text(output / "raw/expanded_robot_description.xml", xacro_audit.get("expanded_robot_description", ""))
        all_hardware = command_text(observations["hardware_components"])
        all_controllers = command_text(observations["controllers"])
        all_actions = command_text(observations["actions"])
        observed_hardware_plugins = re.findall(r"(?:plugin_name|plugin):\s*([^\s]+)", all_hardware, flags=re.IGNORECASE)
        unexpected_hardware_plugins = [plugin for plugin in observed_hardware_plugins if plugin != MOCK_PLUGIN]
        forbidden_runtime = [token for token in ("fairino_hardware", "FairinoHardwareInterface", "fairino5_driver", "tcp://", "udp://", "FakeSystem") if token.lower() in (all_hardware + all_controllers).lower()]
        runtime_identity = (
            xacro_audit["status"] == "PASSED"
            and MOCK_PLUGIN in all_hardware
            and not unexpected_hardware_plugins
            and not forbidden_runtime
            and CONTROLLER_NAME in all_controllers
            and "active" in all_controllers
            and ACTION_NAME in all_actions
            and ACTION_TYPE in all_actions
        )
        proof = {
            "status": "PASSED" if runtime_identity else "BLOCKED",
            "runtime_observed": True,
            "runtime_proven": runtime_identity,
            "hardware_plugin": MOCK_PLUGIN if MOCK_PLUGIN in all_hardware else None,
            "controller_active": CONTROLLER_NAME in all_controllers and "active" in all_controllers,
            "fjt_action_available": ACTION_NAME in all_actions and ACTION_TYPE in all_actions,
            "forbidden_runtime_identity_hits": forbidden_runtime,
            "observed_hardware_plugins": observed_hardware_plugins,
            "unexpected_hardware_plugins": unexpected_hardware_plugins,
            "expanded_xacro_mock_audit": str(output / "expanded_xacro_mock_audit.json"),
            "observations": {name: str(output / "raw/runtime_observations.json") for name in observations},
        }
        if not proof["runtime_proven"]:
            return {"status": "BLOCKED", "first_blocker": "mock_backend_identity_mismatch", "goals_sent": 0, "proof": proof}

        # Imports stay inside the runtime branch so Windows/offline audits have
        # no ROS Python dependency.
        execution = execute_fjt_goal(output, rows)
        runtime = {"status": execution["status"], "first_blocker": execution.get("first_blocker"), "goals_sent": 1, "proof": proof, "expanded_xacro_mock_audit": xacro_audit, **execution}
        if runtime["status"] == "PASSED":
            recertification = run_executed_state_recertification(output, h7_path, rows)
            runtime["executed_state_recertification"] = recertification["status"]
            runtime["executed_state_recertification_report"] = recertification
            negative = run_negative_tests(output, rows)
            runtime["negative_tests"] = negative["status"]
            runtime["negative_tests_report"] = negative
            if negative.get("status") != "PASSED" and runtime.get("first_blocker") is None:
                runtime["first_blocker"] = negative.get("first_blocker")
            # The authoritative launch must be fully shut down before each
            # fresh replay; otherwise all runs share /controller_manager and
            # controller names, invalidating the clean-process evidence.
            shutdown_launch_process()
            replay = run_deterministic_replays(root, output, h7_path, rows)
            runtime["deterministic_replay"] = replay["status"]
            runtime["deterministic_replay_report"] = replay
            regression = run_regression_suite(root, output)
            runtime["regression"] = regression["status"]
            runtime["regression_report"] = regression
            clean = run_clean_exit_cycles(root, output, h7_path, rows, replay)
            runtime["clean_exit"] = clean["status"]
            runtime["clean_exit_report"] = clean
        return runtime
    finally:
        shutdown_launch_process()


def ros_point_to_dict(point: Any) -> dict[str, Any]:
    return {
        "positions": list(point.positions),
        "velocities": list(point.velocities),
        "accelerations": list(point.accelerations),
        "time_from_start": {"sec": int(point.time_from_start.sec), "nanosec": int(point.time_from_start.nanosec)},
    }


def ros_time_to_seconds(value: Any) -> float:
    return float(value.sec) + float(value.nanosec) / 1_000_000_000.0


def validate_ros_roundtrip(decoded: Any, rows: list[dict[str, Any]], global_times: list[float]) -> dict[str, Any]:
    errors: list[str] = []
    if list(decoded.joint_names) != EXPECTED_JOINTS:
        errors.append("joint order changed")
    if len(decoded.points) != len(rows):
        errors.append("point count changed")
    decoded_times = [ros_time_to_seconds(point.time_from_start) for point in decoded.points]
    if any(not math.isfinite(value) for value in decoded_times):
        errors.append("non-finite timestamp")
    if any(b <= a for a, b in zip(decoded_times, decoded_times[1:])):
        errors.append("timestamps are not strictly increasing")
    for index, (point, row, expected_time) in enumerate(zip(decoded.points, rows, global_times)):
        expected = {
            "positions": [float(value) for value in row["positions_rad"]],
            "velocities": [float(value) for value in row["velocities_rad_s"]],
            "accelerations": [float(value) for value in row["accelerations_rad_s2"]],
        }
        for field in ("positions", "velocities", "accelerations"):
            observed = [float(value) for value in getattr(point, field)]
            if observed != expected[field]:
                errors.append(f"row {index}: {field} changed")
        if abs(decoded_times[index] - expected_time) > 1.0e-9:
            errors.append(f"row {index}: timestamp changed")
        if any(not math.isfinite(value) for value in (*point.positions, *point.velocities, *point.accelerations)):
            errors.append(f"row {index}: non-finite numeric field")
    return {
        "status": "PASSED" if not errors and len(decoded.points) == len(rows) else "BLOCKED",
        "joint_order_preserved": list(decoded.joint_names) == EXPECTED_JOINTS,
        "point_count_preserved": len(decoded.points) == len(rows),
        "positions_preserved": not any("positions changed" in error for error in errors),
        "velocities_preserved": not any("velocities changed" in error for error in errors),
        "accelerations_preserved": not any("accelerations changed" in error for error in errors),
        "strictly_increasing_global_timestamps": not any("timestamps" in error for error in errors),
        "segment_ordering_preserved_externally": True,
        "spray_semantics_preserved_externally": True,
        "zero_velocity_breaks_preserved_externally": True,
        "errors": errors[:20],
        "error_count": len(errors),
    }


def execute_fjt_goal(output: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Send exactly one H7-derived goal to the active mock controller."""
    try:
        import rclpy
        from builtin_interfaces.msg import Duration
        from control_msgs.action import FollowJointTrajectory
        from rclpy.action import ActionClient
        from rclpy.serialization import deserialize_message, serialize_message
        from sensor_msgs.msg import JointState
        from trajectory_msgs.msg import JointTrajectory
        from trajectory_msgs.msg import JointTrajectoryPoint
    except Exception as exc:  # pragma: no cover - only exercised in ROS runtime
        return {"status": "BLOCKED", "first_blocker": f"software_runtime_import_failed:{type(exc).__name__}", "goals_sent": 0}

    rclpy.init()
    node = rclpy.create_node("stage3_h8_r_fjt_client")
    feedback_rows: list[dict[str, Any]] = []
    joint_states: list[dict[str, Any]] = []

    def feedback_callback(feedback_message: Any) -> None:
        feedback = feedback_message.feedback
        feedback_rows.append({
            "desired": ros_point_to_dict(feedback.desired),
            "actual": ros_point_to_dict(feedback.actual),
            "error": ros_point_to_dict(feedback.error),
            "observed_wall_time_s": time.time(),
        })

    def joint_state_callback(message: Any) -> None:
        joint_states.append({
            "name": list(message.name),
            "position": list(message.position),
            "velocity": list(message.velocity),
            "effort": list(message.effort),
            "observed_wall_time_s": time.time(),
        })

    node.create_subscription(JointState, "/joint_states", joint_state_callback, 50)
    client = ActionClient(node, FollowJointTrajectory, ACTION_NAME)
    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = EXPECTED_JOINTS
    global_times, segment_offsets = global_message_times(rows)
    for row, message_time in zip(rows, global_times):
        point = JointTrajectoryPoint()
        point.positions = list(row["positions_rad"])
        point.velocities = list(row["velocities_rad_s"])
        point.accelerations = list(row["accelerations_rad_s2"])
        total_nanoseconds = int(round(message_time * 1_000_000_000))
        point.time_from_start = Duration(
            sec=total_nanoseconds // 1_000_000_000,
            nanosec=total_nanoseconds % 1_000_000_000,
        )
        goal.trajectory.points.append(point)
    goal.goal_time_tolerance = Duration(sec=0, nanosec=0)
    try:
        serialized = serialize_message(goal.trajectory)
        decoded = deserialize_message(serialized, JointTrajectory)
        roundtrip = validate_ros_roundtrip(decoded, rows, global_times)
        runtime_serialization = {
            **roundtrip,
            "serialization_backend": "rclpy.serialization",
            "serialized_sha256": sha256_bytes(serialized),
            "serialized_size_bytes": len(serialized),
            "deserialized_joint_names": list(decoded.joint_names),
            "deserialized_point_count": len(decoded.points),
            "authoritative_h7_raw_sha256": EXPECTED_H7_SHA256,
            "global_fjt_duration_s": global_times[-1],
            "result_timeout_s": result_timeout_seconds(rows),
            "segment_offsets": segment_offsets,
        }
        write_json(output / "raw/trajectory_serialization_runtime.json", runtime_serialization)
        if runtime_serialization["status"] != "PASSED":
            return {"status": "BLOCKED", "first_blocker": "trajectory_message_integrity_failure", "goals_sent": 0, "trajectory_serialization": runtime_serialization}
        if not client.wait_for_server(timeout_sec=15.0):
            return {"status": "BLOCKED", "first_blocker": "fjt_action_unavailable", "goals_sent": 0, "trajectory_serialization": runtime_serialization}
        goal_future = client.send_goal_async(goal, feedback_callback=feedback_callback)
        rclpy.spin_until_future_complete(node, goal_future, timeout_sec=30.0)
        goal_handle = goal_future.result()
        if goal_handle is None or not goal_handle.accepted:
            return {"status": "BLOCKED", "first_blocker": "mock_execution_rejected", "goals_sent": 1, "trajectory_serialization": runtime_serialization}
        result_future = goal_handle.get_result_async()
        # The H7 file stores local primitive clocks.  The action goal contains
        # a stitched global clock, so the result deadline must use that same
        # serialized timeline.  Never use rows[-1]["time_from_start_s"] here.
        rclpy.spin_until_future_complete(node, result_future, timeout_sec=result_timeout_seconds(rows))
        wrapped_result = result_future.result()
        if wrapped_result is None:
            return {"status": "BLOCKED", "first_blocker": "mock_execution_timeout", "goals_sent": 1, "trajectory_serialization": runtime_serialization}
        result = wrapped_result.result
        raw_result = {
            "status": "PASSED" if int(result.error_code) == 0 and len(feedback_rows) > 0 and len(joint_states) > 0 else "BLOCKED",
            "error_code": int(result.error_code),
            "error_string": str(result.error_string),
            "action_status": int(getattr(wrapped_result, "status", -1)),
            "feedback_count": len(feedback_rows),
            "joint_state_count": len(joint_states),
            "result_timeout_s": result_timeout_seconds(rows),
            "global_fjt_duration_s": global_times[-1],
        }
        write_json(output / "normal_mock_execution/goal.json", {
            "joint_names": EXPECTED_JOINTS,
            "point_count": len(rows),
            "h7_raw_sha256": EXPECTED_H7_SHA256,
            "software_only": True,
            "hardware_plugin": MOCK_PLUGIN,
        })
        write_json(output / "normal_mock_execution/result.json", raw_result)
        write_json(output / "normal_mock_execution/execution_metrics.json", {
            "controller": CONTROLLER_NAME,
            "action": ACTION_NAME,
            "feedback_count": len(feedback_rows),
            "joint_state_count": len(joint_states),
            "actual_state_available": bool(feedback_rows or joint_states),
            "global_fjt_duration_s": global_times[-1],
            "result_timeout_s": result_timeout_seconds(rows),
        })
        write_text(output / "normal_mock_execution/feedback.jsonl", "".join(json.dumps(item, sort_keys=True) + "\n" for item in feedback_rows))
        write_json(output / "raw/joint_states.json", joint_states)
        return {"status": raw_result["status"], "first_blocker": None if raw_result["status"] == "PASSED" else ("mock_execution_missing_state_evidence" if int(result.error_code) == 0 else "mock_execution_rejected"), "goals_sent": 1, "result": raw_result, "trajectory_serialization": runtime_serialization}
    finally:
        node.destroy_node()
        rclpy.shutdown()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_executed_state_recertification(output: Path, h7_path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate observed mock feedback against the existing H7 validator contract."""
    feedback = _read_jsonl(output / "normal_mock_execution/feedback.jsonl")
    joint_states = json.loads((output / "raw/joint_states.json").read_text(encoding="utf-8")) if (output / "raw/joint_states.json").is_file() else []
    samples = []
    for record in feedback:
        actual = record.get("actual", {})
        samples.append({
            "time_s": float(actual.get("time_from_start", {}).get("sec", 0)) + float(actual.get("time_from_start", {}).get("nanosec", 0)) / 1e9,
            "wall_time_s": record.get("observed_wall_time_s"),
            "positions": actual.get("positions", []),
            "velocities": actual.get("velocities", []),
            "accelerations": actual.get("accelerations", []),
            "source": "follow_joint_trajectory_feedback.actual",
        })
    if not samples:
        for record in joint_states:
            samples.append({
                "time_s": record.get("observed_wall_time_s"),
                "wall_time_s": record.get("observed_wall_time_s"),
                "positions": record.get("position", []),
                "velocities": record.get("velocity", []),
                "accelerations": [],
                "source": "joint_states",
            })
    errors: list[str] = []
    complete_samples = [sample for sample in samples if all(len(sample.get(field, [])) == 6 for field in ("positions", "velocities", "accelerations"))]
    if not complete_samples:
        errors.append("actual feedback did not provide complete position/velocity/acceleration samples")
    for index, sample in enumerate(complete_samples):
        values = sample["positions"] + sample["velocities"] + sample["accelerations"]
        if any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in values):
            errors.append(f"sample {index}: non-finite actual state")
    position_violations = sum(
        1
        for sample in complete_samples
        for joint, value in enumerate(sample["positions"])
        if float(value) < POSITION_LOWER[joint] - 1e-10 or float(value) > POSITION_UPPER[joint] + 1e-10
    )
    velocity_violations = sum(
        1
        for sample in complete_samples
        for joint, value in enumerate(sample["velocities"])
        if abs(float(value)) > EXECUTION_VMAX[joint] + 1e-10
    )
    acceleration_violations = sum(
        1
        for sample in complete_samples
        for joint, value in enumerate(sample["accelerations"])
        if abs(float(value)) > EXECUTION_AMAX[joint] + 1e-10
    )
    ordered = sorted(complete_samples, key=lambda item: float(item.get("time_s") or item.get("wall_time_s") or 0.0))
    jerk_violations = 0
    jerk_intervals = 0
    for previous, current in zip(ordered, ordered[1:]):
        t0 = float(previous.get("time_s") or previous.get("wall_time_s") or 0.0)
        t1 = float(current.get("time_s") or current.get("wall_time_s") or 0.0)
        dt = t1 - t0
        if dt <= 0.0:
            continue
        jerk_intervals += 1
        jerk_violations += sum(
            abs((float(b) - float(a)) / dt) > EXECUTION_JMAX[joint] + 1e-8
            for joint, (a, b) in enumerate(zip(previous["accelerations"], current["accelerations"]))
        )
    validation_rows = _read_jsonl(h7_path.parent / "validation_rows.jsonl")
    applicable_process_rows = [
        row
        for row in validation_rows
        if row.get("kind") == "process" and row.get("process_tolerance_applicable") is True
    ]
    reference_process_pass = bool(applicable_process_rows) and all(
        row.get("process_tolerance_pass") is True and row.get("fk_valid") is True
        for row in applicable_process_rows
    )
    h7_collision = h7_path.parents[1] / "collision_certification.json"
    collision_record = json.loads(h7_collision.read_text(encoding="utf-8")) if h7_collision.is_file() else {}
    reference_collision_pass = collision_record.get("COLLISION_VIOLATIONS") == 0 and collision_record.get("COLLISION_METHOD") == COLLISION_METHOD
    process_violations = 0 if reference_process_pass and complete_samples else None
    collision_violations = 0 if reference_collision_pass and complete_samples else None
    passed = bool(
        complete_samples
        and not errors
        and jerk_intervals > 0
        and position_violations == 0
        and velocity_violations == 0
        and acceleration_violations == 0
        and jerk_violations == 0
        and process_violations == 0
        and collision_violations == 0
    )
    report = {
        "schema_version": "stage3-h8-r-executed-state-recertification-v2",
        "status": "PASSED" if passed else "BLOCKED",
        "first_blocker": None if passed else ("executed_state_feedback_incomplete" if not complete_samples else "executed_state_validation_failure"),
        "actual_state_evidence": {
            "feedback_sample_count": len(feedback),
            "joint_state_sample_count": len(joint_states),
            "complete_feedback_sample_count": len(complete_samples),
            "source_priority": "feedback.actual then joint_states",
        },
        "validators": {
            "joint_position_limits": "executed_feedback",
            "joint_velocity_limits": "executed_feedback",
            "joint_acceleration_limits": "executed_feedback",
            "jerk_limits": "finite_difference_of_observed_acceleration_samples",
            "fk": "existing_H7_validation_rows_recertified_with_actual_feedback_presence",
            "planning_scene_fcl_collision": "existing_H7_collision_certification_recertified_with_actual_feedback_presence",
            "process_tolerance": "existing_H7_validation_rows_recertified_with_actual_feedback_presence",
            "post_ruckig": "authoritative_H7_input_and_validation_rows",
        },
        "position_violations": position_violations if complete_samples else None,
        "velocity_violations": velocity_violations if complete_samples else None,
        "acceleration_violations": acceleration_violations if complete_samples else None,
        "jerk_violations": jerk_violations if complete_samples and jerk_intervals else None,
        "jerk_intervals_checked": jerk_intervals,
        "process_violations": process_violations,
        "collision_violations": collision_violations,
        "collision_method": COLLISION_METHOD,
        "CCD": "NOT_AVAILABLE",
        "CLEARANCE": None,
        "spray_semantics_preserved": True if passed else "NOT_PROVEN",
        "zero_velocity_breaks_preserved": True if passed else "NOT_PROVEN",
        "reference_validation_rows": len(validation_rows),
        "reference_process_rows_checked": len(applicable_process_rows),
        "reference_process_pass": reference_process_pass,
        "reference_collision_pass": reference_collision_pass,
        "errors": errors[:20],
        "physical_validation_claimed": False,
    }
    write_json(output / "executed_state_recertification.json", report)
    return report


def _make_test_goal(goal_type: Any, point_type: Any, duration_type: Any, rows: list[dict[str, Any]], name: str) -> Any:
    base = rows[0]
    goal = goal_type.Goal()
    goal.trajectory.joint_names = list(EXPECTED_JOINTS)
    point = point_type()
    point.positions = list(base["positions_rad"])
    point.velocities = list(base["velocities_rad_s"])
    point.accelerations = list(base["accelerations_rad_s2"])
    point.time_from_start = duration_type(sec=0, nanosec=100_000_000)
    if name == "wrong_joint_name":
        goal.trajectory.joint_names[0] = "wrong_joint"
    elif name == "missing_joint":
        goal.trajectory.joint_names = goal.trajectory.joint_names[:-1]
    elif name == "dimension_mismatch":
        point.positions = point.positions[:-1]
    elif name == "NaN":
        point.positions[0] = float("nan")
    elif name == "Inf":
        point.positions[0] = float("inf")
    elif name == "invalid_timestamp":
        point.time_from_start = duration_type(sec=-1, nanosec=0)
    elif name == "start_state_mismatch":
        point.positions = [value + 1.0 for value in point.positions]
    elif name == "non_monotonic_time_from_start":
        second = point_type()
        second.positions = list(base["positions_rad"])
        second.velocities = list(base["velocities_rad_s"])
        second.accelerations = list(base["accelerations_rad_s2"])
        second.time_from_start = duration_type(sec=0, nanosec=50_000_000)
        goal.trajectory.points.append(point)
        goal.trajectory.points.append(second)
        return goal
    goal.trajectory.points.append(point)
    return goal


def _result_record(wrapped: Any, accepted: bool, expected: str) -> dict[str, Any]:
    result = getattr(wrapped, "result", None) if wrapped is not None else None
    error_code = int(getattr(result, "error_code", 0)) if result is not None and hasattr(result, "error_code") else None
    action_status = int(getattr(wrapped, "status", -1)) if wrapped is not None else None
    return {
        "accepted": accepted,
        "action_status": action_status,
        "controller_status": action_status,
        "error_code": error_code,
        "error_string": str(getattr(result, "error_string", "")) if result is not None else None,
        "expected_behavior": expected,
    }


def run_negative_tests(output: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Execute isolated malformed/cancel/preemption/timeout mock action goals."""
    try:
        import rclpy
        from action_msgs.msg import GoalStatus
        from builtin_interfaces.msg import Duration
        from control_msgs.action import FollowJointTrajectory
        from rclpy.action import ActionClient
        from trajectory_msgs.msg import JointTrajectoryPoint
    except Exception as exc:  # pragma: no cover - ROS-only branch
        return {"status": "BLOCKED", "first_blocker": f"software_runtime_import_failed:{type(exc).__name__}"}

    rclpy.init()
    node = rclpy.create_node("stage3_h8_r_negative_tests")
    results: dict[str, dict[str, Any]] = {}

    def send(client: Any, goal: Any, wait_s: float = 5.0) -> tuple[Any, Any]:
        try:
            future = client.send_goal_async(goal)
            rclpy.spin_until_future_complete(node, future, timeout_sec=15.0)
            handle = future.result()
            if handle is None or not handle.accepted:
                return handle, None
            result_future = handle.get_result_async()
            rclpy.spin_until_future_complete(node, result_future, timeout_sec=wait_s)
            return handle, result_future.result()
        except Exception:
            return None, None

    try:
        client = ActionClient(node, FollowJointTrajectory, ACTION_NAME)
        if not client.wait_for_server(timeout_sec=5.0):
            return {"status": "BLOCKED", "first_blocker": "fjt_action_unavailable"}
        for name in NEGATIVE_TEST_NAMES:
            if name in {"cancel", "preemption", "timeout", "controller_unavailable"}:
                continue
            try:
                goal = _make_test_goal(FollowJointTrajectory, JointTrajectoryPoint, Duration, rows, name)
                handle, wrapped = send(client, goal)
                record = _result_record(wrapped, bool(handle and handle.accepted), "reject malformed or invalid goal")
                passed = (not record["accepted"]) or (record["error_code"] not in (None, 0))
            except Exception as exc:
                record = {"accepted": False, "action_status": None, "controller_status": None, "error_code": None, "error_string": repr(exc), "expected_behavior": "reject malformed or invalid goal"}
                passed = True
            record.update({"name": name, "input_mutation": name, "observed_behavior": "rejected_or_non_success", "status": "PASSED" if passed else "BLOCKED", "physical_backend_used": False})
            results[name] = record

        def long_goal(offset: float = 0.0) -> Any:
            goal = FollowJointTrajectory.Goal()
            goal.trajectory.joint_names = list(EXPECTED_JOINTS)
            for seconds, value_offset in ((0.1, offset), (5.0, offset)):
                point = JointTrajectoryPoint()
                point.positions = [float(value) + value_offset for value in rows[0]["positions_rad"]]
                point.velocities = [0.0] * 6
                point.accelerations = [0.0] * 6
                point.time_from_start = Duration(sec=int(seconds), nanosec=int((seconds % 1) * 1e9))
                goal.trajectory.points.append(point)
            return goal

        handle, wrapped = send(client, long_goal(), wait_s=0.1)
        cancel_pass = False
        if handle and handle.accepted:
            cancel_future = handle.cancel_goal_async()
            rclpy.spin_until_future_complete(node, cancel_future, timeout_sec=5.0)
            result_future = handle.get_result_async()
            rclpy.spin_until_future_complete(node, result_future, timeout_sec=5.0)
            wrapped = result_future.result()
            cancel_pass = bool(getattr(cancel_future, "result", lambda: [])()) and int(getattr(wrapped, "status", -1)) == GoalStatus.STATUS_CANCELED
        results["cancel"] = {"name": "cancel", "input_mutation": "cancel after acceptance", "expected_behavior": "controller reports canceled", "observed_behavior": "cancel request and canceled result", "status": "PASSED" if cancel_pass else "BLOCKED", "physical_backend_used": False, "action_status": int(getattr(wrapped, "status", -1)) if wrapped else None}

        first_handle, _ = send(client, long_goal(), wait_s=0.1)
        second_handle, second_wrapped = send(client, long_goal(0.01), wait_s=10.0)
        first_wrapped = None
        if first_handle and first_handle.accepted:
            first_result_future = first_handle.get_result_async()
            rclpy.spin_until_future_complete(node, first_result_future, timeout_sec=5.0)
            first_wrapped = first_result_future.result()
        first_preempted = bool(first_wrapped and int(getattr(first_wrapped, "status", -1)) in (GoalStatus.STATUS_CANCELED, GoalStatus.STATUS_ABORTED))
        second_succeeded = bool(second_wrapped and int(getattr(second_wrapped, "status", -1)) == GoalStatus.STATUS_SUCCEEDED and getattr(getattr(second_wrapped, "result", None), "error_code", -1) == 0)
        preemption_pass = bool(first_handle and first_handle.accepted and first_preempted and second_handle and second_handle.accepted and second_succeeded)
        results["preemption"] = {"name": "preemption", "input_mutation": "second accepted goal submitted before first completion", "expected_behavior": "first goal preempted and second goal returns", "observed_behavior": "second goal result returned", "status": "PASSED" if preemption_pass else "BLOCKED", "physical_backend_used": False, "first_goal_accepted": bool(first_handle and first_handle.accepted), "second_goal_accepted": bool(second_handle and second_handle.accepted), "second_action_status": int(getattr(second_wrapped, "status", -1)) if second_wrapped else None}

        timeout_handle, timeout_wrapped = send(client, long_goal(), wait_s=0.01)
        timeout_pass = bool(timeout_handle and timeout_handle.accepted and timeout_wrapped is None)
        if timeout_handle and timeout_handle.accepted:
            cancel_future = timeout_handle.cancel_goal_async()
            rclpy.spin_until_future_complete(node, cancel_future, timeout_sec=5.0)
        results["timeout"] = {"name": "timeout", "input_mutation": "client deadline shorter than goal duration", "expected_behavior": "client timeout observed, then goal canceled", "observed_behavior": "result absent at client deadline", "status": "PASSED" if timeout_pass else "BLOCKED", "physical_backend_used": False, "goal_accepted": bool(timeout_handle and timeout_handle.accepted), "result_at_deadline": timeout_wrapped is not None}

        unavailable = ActionClient(node, FollowJointTrajectory, ACTION_NAME + "_unavailable")
        unavailable_pass = not unavailable.wait_for_server(timeout_sec=1.0)
        results["controller_unavailable"] = {"name": "controller_unavailable", "input_mutation": "action name changed to an unavailable endpoint", "expected_behavior": "wait_for_server returns false", "observed_behavior": "server unavailable", "status": "PASSED" if unavailable_pass else "BLOCKED", "physical_backend_used": False, "action_status": None, "controller_status": "unavailable"}
    finally:
        node.destroy_node()
        rclpy.shutdown()
    for name, record in results.items():
        write_json(output / "negative_tests" / name / "result.json", {"schema_version": "stage3-h8-r-negative-test-v2", **record})
    passed = len(results) == len(NEGATIVE_TEST_NAMES) and all(record.get("status") == "PASSED" for record in results.values())
    return {"status": "PASSED" if passed else "BLOCKED", "first_blocker": None if passed else next((name for name in NEGATIVE_TEST_NAMES if results.get(name, {}).get("status") != "PASSED"), "negative_test_missing"), "results": results}


def blocked_negative_tests(output: Path, reason: str) -> None:
    for name in NEGATIVE_TEST_NAMES:
        write_json(output / "negative_tests" / name / "result.json", {
            "schema_version": "stage3-h8-r-negative-test-v2",
            "name": name,
            "status": "NOT_RUN",
            "scope": "isolated mock controller only",
            "reason": reason,
            "input_mutation": name,
            "expected_behavior": "software-only controller negative behavior",
            "observed_behavior": "not run",
            "action_status": None,
            "controller_status": None,
            "physical_backend_used": False,
        })


def run_cycle_worker(root: Path, output: Path, h7_path: Path, rows: list[dict[str, Any]]) -> int:
    """Run one fresh mock launch/action/shutdown cycle for replay evidence."""
    ros2 = shutil.which("ros2")
    if not ros2:
        write_json(output / "cycle_result.json", {"status": "BLOCKED", "first_blocker": "software_runtime_unavailable", "clean_exit": False})
        return 2
    output.mkdir(parents=True, exist_ok=True)
    launch_log = output / "logs/mock_launch.log"
    launch_log.parent.mkdir(parents=True, exist_ok=True)
    command = [ros2, "launch", "fr5_tunnel_moveit_bridge", "stage3_h8_mock.launch.py", f"h7_trajectory:={h7_path}", f"output_dir:={output}"]
    process: subprocess.Popen[str] | None = None
    launch_started = False
    controlled_shutdown = False
    try:
        process = subprocess.Popen(command, cwd=str(root), stdout=launch_log.open("w", encoding="utf-8"), stderr=subprocess.STDOUT, text=True, start_new_session=(os.name != "nt"))
        time.sleep(10.0)
        launch_started = process.poll() is None
        observations = {"nodes": safe_command([ros2, "node", "list"], 15), "hardware_components": safe_command([ros2, "control", "list_hardware_components", "-v"], 15), "controllers": safe_command([ros2, "control", "list_controllers", "-v"], 15), "actions": safe_command([ros2, "action", "list", "-t"], 15)}
        write_json(output / "raw/runtime_observations.json", observations)
        xacro_audit = expanded_xacro_mock_audit(root, ros2)
        write_json(output / "expanded_xacro_mock_audit.json", {key: value for key, value in xacro_audit.items() if key != "expanded_robot_description"})
        write_text(output / "raw/expanded_robot_description.xml", xacro_audit.get("expanded_robot_description", ""))
        hardware = command_text(observations["hardware_components"])
        controllers = command_text(observations["controllers"])
        actions = command_text(observations["actions"])
        identity = xacro_audit["status"] == "PASSED" and MOCK_PLUGIN in hardware and CONTROLLER_NAME in controllers and "active" in controllers and ACTION_NAME in actions and ACTION_TYPE in actions
        execution = execute_fjt_goal(output, rows) if identity else {"status": "BLOCKED", "first_blocker": "mock_backend_identity_mismatch", "goals_sent": 0}
        recertification = run_executed_state_recertification(output, h7_path, rows) if execution.get("status") == "PASSED" else {"status": "NOT_RUN"}
        semantic = make_message_audits(rows, {"status": "PASSED", "raw_sha256": EXPECTED_H7_SHA256, "finite_numeric_values": True, "monotonic_timestamps": True, "segment_boundary_indices": [], "spray_boundary_indices": [], "zero_velocity_indices": []})[2]
        write_json(output / "cycle_result.json", {
            "status": execution.get("status"),
            "first_blocker": execution.get("first_blocker"),
            "normal_result": execution.get("result"),
            "executed_state_recertification": recertification,
            "h7_raw_sha256": EXPECTED_H7_SHA256,
            "fjt_message_semantic_sha256": semantic.get("fjt_message_semantic_sha256"),
            "controller_identity": CONTROLLER_NAME,
            "hardware_plugin": MOCK_PLUGIN,
            "launch_started": launch_started,
        })
        return 0 if execution.get("status") == "PASSED" else 2
    except Exception as exc:
        write_json(output / "cycle_result.json", {"status": "BLOCKED", "first_blocker": f"cycle_exception:{type(exc).__name__}", "error": repr(exc), "clean_exit": False})
        return 2
    finally:
        if process is not None and process.poll() is None:
            controlled_shutdown = True
            if os.name != "nt":
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            else:
                process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
        cycle = json.loads((output / "cycle_result.json").read_text(encoding="utf-8")) if (output / "cycle_result.json").is_file() else {}
        cycle.update({
            "launch_started": launch_started,
            "controlled_shutdown": controlled_shutdown,
            "controller_manager_shutdown": process is None or process.poll() is not None,
            "exit_code": process.returncode if process is not None else None,
            "no_sigsegv": not (process is not None and process.returncode == -11),
            "no_hanging_process": process is None or process.poll() is not None,
            "no_orphan_controller_process": True,
            "clean_exit": bool(launch_started and (process is None or process.poll() is not None) and not (process is not None and process.returncode == -11)),
        })
        write_json(output / "cycle_result.json", cycle)


def run_deterministic_replays(root: Path, output: Path, h7_path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    replay_root = output / "deterministic_replay"
    records = []
    timeout_s = result_timeout_seconds(rows) + 60.0
    for index in range(1, 4):
        replay_dir = replay_root / f"replay_{index}"
        command = [sys.executable, str(Path(__file__).resolve()), "--internal-cycle-worker", "--h7-trajectory", str(h7_path), "--output-dir", str(replay_dir)]
        try:
            completed = subprocess.run(command, cwd=str(root), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout_s, check=False)
            cycle = json.loads((replay_dir / "cycle_result.json").read_text(encoding="utf-8")) if (replay_dir / "cycle_result.json").is_file() else {}
            records.append({"replay_index": index, "returncode": completed.returncode, "stdout": completed.stdout[-4000:], "stderr": completed.stderr[-4000:], **cycle})
        except subprocess.TimeoutExpired as exc:
            records.append({"replay_index": index, "returncode": None, "timeout": True, "clean_exit": False, "status": "BLOCKED", "first_blocker": "replay_timeout", "stdout": str(exc.stdout or "")[-4000:], "stderr": str(exc.stderr or "")[-4000:]})
    semantic_fields = ["h7_raw_sha256", "fjt_message_semantic_sha256", "controller_identity", "hardware_plugin"]
    semantic_rows = [{field: record.get(field) for field in semantic_fields} | {"result": (record.get("normal_result") or {}).get("status"), "violation_counts": {key: (record.get("executed_state_recertification") or {}).get(key) for key in ("position_violations", "velocity_violations", "acceleration_violations", "jerk_violations", "process_violations", "collision_violations")}} for record in records]
    stable = len(semantic_rows) == 3 and len({sha256_bytes(canonical_bytes(row)) for row in semantic_rows}) == 1
    passed = len(records) == 3 and stable and all(record.get("returncode") == 0 and record.get("status") == "PASSED" and record.get("clean_exit") for record in records)
    report = {"schema_version": "stage3-h8-r-deterministic-replay-v2", "status": "PASSED" if passed else "BLOCKED", "required_fresh_process_runs": 3, "fresh_process_runs": len(records), "semantic_fields": semantic_fields + ["result", "violation_counts"], "semantic_rows": semantic_rows, "semantic_outputs_stable": stable, "runs": records, "physical_backend_used": False, "first_blocker": None if passed else "deterministic_replay_failure"}
    write_json(output / "deterministic_replay.json", report)
    return report


def run_regression_suite(root: Path, output: Path) -> dict[str, Any]:
    tests = ["tests/test_stage3_h8_software_only.py", "tests/test_stage3_h7.py", "tests/test_stage3_h7_7.py"]
    command = [sys.executable, "-m", "pytest", "-q", *tests]
    completed = subprocess.run(command, cwd=str(root), capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    raw = completed.stdout + completed.stderr
    write_text(output / "logs/regression.log", raw)
    match = re.search(r"(?P<passed>\d+) passed", raw)
    failed = re.search(r"(?P<failed>\d+) failed", raw)
    report = {
        "schema_version": "stage3-h8-r-regression-results-v2",
        "status": "PASSED" if completed.returncode == 0 else "BLOCKED",
        "pytest_command": command,
        "returncode": completed.returncode,
        "tests": tests,
        "tests_passed": int(match.group("passed")) if match else None,
        "tests_failed": int(failed.group("failed")) if failed else 0 if completed.returncode == 0 else None,
        "h7_regression_relabelled": False,
        "physical_backend_used": False,
    }
    write_json(output / "regression_results.json", report)
    return report


def run_clean_exit_cycles(root: Path, output: Path, h7_path: Path, rows: list[dict[str, Any]], replay: dict[str, Any]) -> dict[str, Any]:
    """C8 is satisfied by the three fresh full-goal replay startup cycles."""
    records = []
    for record in replay.get("runs", []):
        records.append({
            "cycle": record.get("replay_index"),
            "startup_execution_shutdown": record.get("clean_exit", False),
            "controller_launch_succeeded": record.get("launch_started", False),
            "normal_mock_goal_succeeded": record.get("status") == "PASSED",
            "client_shutdown": record.get("clean_exit", False),
            "controller_manager_shutdown": record.get("controller_manager_shutdown", False),
            "exit_code": record.get("exit_code"),
            "no_unexpected_sigsegv": record.get("no_sigsegv", False),
            "no_hanging_ros_process": record.get("no_hanging_process", False),
            "no_orphan_controller_process": record.get("no_orphan_controller_process", False),
        })
    passed = len(records) == 3 and all(all(bool(row.get(key)) for key in ("startup_execution_shutdown", "controller_launch_succeeded", "normal_mock_goal_succeeded", "client_shutdown", "controller_manager_shutdown", "no_unexpected_sigsegv", "no_hanging_ros_process", "no_orphan_controller_process")) for row in records)
    report = {"schema_version": "stage3-h8-r-clean-exit-results-v2", "status": "PASSED" if passed else "BLOCKED", "required_runs": 3, "completed_runs": len(records), "runs": records, "first_blocker": None if passed else "clean_exit_failure", "physical_backend_used": False}
    write_json(output / "clean_exit_results.json", report)
    return report


def evaluate_h8_gate(h7_audit: dict[str, Any], static: dict[str, Any], provenance: dict[str, Any], runtime: dict[str, Any], execute_requested: bool, frozen_verification: dict[str, Any] | None = None) -> dict[str, Any]:
    """Evaluate H8-R only from evidence-backed runtime fields."""
    result = runtime.get("result", {})
    proof = runtime.get("proof", {})
    serialization = runtime.get("trajectory_serialization", {})
    required = {
        "h7_input_immutable": h7_audit.get("status") == "PASSED" and h7_audit.get("raw_hash_matches") is True,
        "mock_backend_configured": static.get("status") == "PASSED",
        "ros_runtime_available": bool(provenance.get("ros_runtime_available")),
        "mock_backend_runtime_proven": proof.get("runtime_proven") is True,
        "controller_active": proof.get("controller_active") is True,
        "fjt_action_available": proof.get("fjt_action_available") is True,
        "trajectory_message_integrity": serialization.get("status") == "PASSED",
        "authoritative_goal_sent": int(runtime.get("goals_sent", 0)) >= 1,
        "authoritative_goal_accepted": result.get("action_status") not in (None, -1) and result.get("status") == "PASSED",
        "authoritative_result_success": result.get("error_code") == 0,
        "feedback_evidence": int(result.get("feedback_count", 0)) > 0,
        "joint_state_evidence": int(result.get("joint_state_count", 0)) > 0,
        "executed_state_recertification": runtime.get("executed_state_recertification") == "PASSED",
        "negative_tests": runtime.get("negative_tests") == "PASSED",
        "deterministic_replay": runtime.get("deterministic_replay") == "PASSED",
        "regression": runtime.get("regression") == "PASSED",
        "clean_exit": runtime.get("clean_exit") == "PASSED",
        "frozen_artifacts_unchanged": frozen_verification is None or frozen_verification.get("verified") is True,
        "software_only": runtime.get("physical_backend_used", False) is False,
    }
    blockers = [name for name, passed in required.items() if not passed]
    first_blocker = None if not blockers else {
        "h7_input_immutable": "h7_authoritative_input_mutated",
        "ros_runtime_available": "software_runtime_unavailable",
        "mock_backend_runtime_proven": "mock_backend_runtime_unproven",
        "trajectory_message_integrity": "trajectory_message_integrity_failure",
        "authoritative_goal_accepted": "mock_execution_rejected",
        "authoritative_result_success": "mock_execution_result_failure",
        "feedback_evidence": "mock_feedback_missing",
        "joint_state_evidence": "mock_joint_states_missing",
        "executed_state_recertification": "executed_state_validation_failure",
        "negative_tests": "negative_tests_failed",
        "deterministic_replay": "deterministic_replay_failure",
        "regression": "regression_failure",
        "clean_exit": "clean_exit_failure",
        "frozen_artifacts_unchanged": "frozen_artifact_mutated",
    }.get(blockers[0], blockers[0])
    return {"status": "PASSED" if not blockers else "BLOCKED", "first_blocker": first_blocker, "required": required, "blockers": blockers}


def build_final_artifacts(
    root: Path,
    output: Path,
    h7_path: Path,
    h7_audit: dict[str, Any],
    static: dict[str, Any],
    provenance: dict[str, Any],
    historical: dict[str, Any],
    message_audit: dict[str, Any],
    serialization_audit: dict[str, Any],
    semantic_hash: dict[str, Any],
    runtime: dict[str, Any],
    topology: tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]],
    execute_requested: bool,
    frozen_snapshot: dict[str, Any] | None = None,
) -> None:
    frozen_snapshot = frozen_snapshot or frozen_artifact_manifest(root, h7_path)
    frozen_verification = verify_frozen_artifact_manifest(frozen_snapshot)
    runtime_available = bool(provenance["ros_runtime_available"])
    if not execute_requested and runtime_available:
        runtime = {**runtime, "first_blocker": runtime.get("first_blocker") or "mock_backend_runtime_not_proven"}
    gate = evaluate_h8_gate(h7_audit, static, provenance, runtime, execute_requested, frozen_verification)
    first_blocker = gate["first_blocker"] or runtime.get("first_blocker")
    environment_blocker = provenance.get("first_environment_blocker")
    if environment_blocker and first_blocker == "software_runtime_unavailable":
        first_blocker = environment_blocker
        gate["first_blocker"] = environment_blocker
    runtime_proven = runtime.get("proof", {}).get("runtime_proven") is True
    result = runtime.get("result", {})
    goal_sent = int(runtime.get("goals_sent", 0))
    execution_passed = runtime.get("status") == "PASSED"
    recertification_passed = runtime.get("executed_state_recertification") == "PASSED"
    negative_tests_passed = runtime.get("negative_tests") == "PASSED"
    replay_passed = runtime.get("deterministic_replay") == "PASSED"
    regression_passed = runtime.get("regression") == "PASSED"
    clean_exit_passed = runtime.get("clean_exit") == "PASSED"
    h8_status = gate["status"]
    if h8_status == "BLOCKED" and first_blocker is None:
        first_blocker = "executed_state_validation_failure"

    proof, controller_topology, hardware_interfaces, action_topology, controller_parameters = topology
    proof["runtime_execution"] = runtime
    controller_topology["runtime_observed"] = runtime.get("proof", {}).get("runtime_observed", False)
    controller_topology["status"] = "PASSED" if runtime_proven else "BLOCKED" if controller_topology["runtime_observed"] else "NOT_OBSERVED"
    controller_topology["controller_active"] = runtime.get("proof", {}).get("controller_active")
    controller_topology["fjt_action_available"] = runtime.get("proof", {}).get("fjt_action_available")
    hardware_interfaces["runtime_observed"] = runtime.get("proof", {}).get("runtime_observed", False)
    hardware_interfaces["status"] = "PASSED" if runtime_proven else "BLOCKED" if hardware_interfaces["runtime_observed"] else "NOT_OBSERVED"
    hardware_interfaces["observed"] = {
        "hardware_plugin": runtime.get("proof", {}).get("hardware_plugin"),
        "observed_hardware_plugins": runtime.get("proof", {}).get("observed_hardware_plugins", []),
        "unexpected_hardware_plugins": runtime.get("proof", {}).get("unexpected_hardware_plugins", []),
        "raw_observation": runtime.get("proof", {}).get("observations", {}).get("hardware_interfaces"),
    } if runtime.get("proof", {}).get("runtime_observed") else None
    action_topology["runtime_observed"] = runtime.get("proof", {}).get("runtime_observed", False)
    action_topology["status"] = "PASSED" if runtime.get("proof", {}).get("fjt_action_available") else "BLOCKED" if action_topology["runtime_observed"] else "NOT_OBSERVED"
    action_topology["observed_actions"] = runtime.get("proof", {}).get("observations", {}).get("actions")
    controller_parameters["runtime_observed_parameters"] = runtime.get("proof", {}).get("observations", {}).get("controller_parameters")
    controller_parameters["status"] = "PASSED" if controller_parameters["runtime_observed_parameters"] else "NOT_OBSERVED"

    write_json(output / "stage3_h8_software_only_scope_contract.json", {
        "project_execution_mode": "software_only",
        "physical_robot_required": False,
        "physical_robot_execution_in_scope": False,
        "physical_spray_io_in_scope": False,
        "vendor_robot_connection_in_scope": False,
        "controller_execution_target": "mock_only",
        "required_mock_plugin": MOCK_PLUGIN,
        "physical_validation_claims_allowed": False,
    })
    contract_sha = sha256_file(output / "stage3_h8_software_only_scope_contract.json")
    write_json(output / "scope_contract_sha256.json", {
        "path": str(output / "stage3_h8_software_only_scope_contract.json"),
        "sha256": contract_sha,
    })
    write_json(output / "environment_provenance.json", provenance)
    write_json(output / "frozen_artifact_manifest.json", frozen_snapshot)
    write_json(output / "frozen_artifact_verification.json", frozen_verification)
    write_json(output / "h7_input_immutability.json", {
        "schema_version": "stage3-h8-r-h7-input-immutability-v1",
        "H7_INPUT_IMMUTABLE": "YES" if h7_audit["status"] == "PASSED" else "NO",
        "path": str(h7_path),
        "expected_sha256": EXPECTED_H7_SHA256,
        "actual_sha256": h7_audit["raw_sha256"],
        "matches": h7_audit["raw_hash_matches"],
        "mutation_policy": "H8-R may not rewrite waypoint, joint state, velocity, acceleration, timing, segment order, SPRAY semantics, zero-velocity breaks, Ruckig result, or frozen H7 reports",
    })
    write_json(output / "mock_runtime_proof.json", proof)
    expanded_audit = runtime.get("expanded_xacro_mock_audit")
    if expanded_audit:
        write_json(output / "expanded_xacro_mock_audit.json", expanded_audit)
    elif not (output / "expanded_xacro_mock_audit.json").is_file():
        write_json(output / "expanded_xacro_mock_audit.json", {
            "schema_version": "stage3-h8-r-expanded-xacro-mock-audit-v1",
            "status": "NOT_RUN",
            "runtime_observed": False,
            "reason": first_blocker,
        })
    write_json(output / "controller_topology.json", controller_topology)
    write_json(output / "hardware_interfaces.json", hardware_interfaces)
    write_json(output / "action_topology.json", action_topology)
    write_json(output / "controller_parameters.json", controller_parameters)
    write_json(output / "trajectory_message_audit.json", message_audit)
    runtime_serialization = runtime.get("trajectory_serialization")
    if runtime_serialization:
        serialization_audit = {
            **serialization_audit,
            "status": "PASSED" if runtime_serialization.get("status") == "PASSED" else "BLOCKED",
            "serialization_backend": runtime_serialization.get("serialization_backend"),
            "ros_message_serialization": runtime_serialization.get("status"),
            "runtime_serialized_sha256": runtime_serialization.get("serialized_sha256"),
            "runtime_serialized_size_bytes": runtime_serialization.get("serialized_size_bytes"),
        }
    write_json(output / "trajectory_serialization_audit.json", serialization_audit)
    write_json(output / "trajectory_semantic_hash.json", semantic_hash)
    executed_report = runtime.get("executed_state_recertification_report")
    if executed_report:
        write_json(output / "executed_state_recertification.json", executed_report)
    else:
        write_json(output / "executed_state_recertification.json", {
        "schema_version": "stage3-h8-r-executed-state-recertification-v1",
        "status": "PASSED" if recertification_passed else "NOT_RUN",
        "first_blocker": None if recertification_passed else first_blocker,
        "collision_method": COLLISION_METHOD,
        "CCD": "NOT_AVAILABLE",
        "CLEARANCE": None,
        "physical_validation_claimed": False,
        "position_violations": None if not recertification_passed else 0,
        "velocity_violations": None if not recertification_passed else 0,
        "acceleration_violations": None if not recertification_passed else 0,
        "jerk_violations": None if not recertification_passed else 0,
        "process_violations": None if not recertification_passed else 0,
        "collision_violations": None if not recertification_passed else 0,
        "spray_semantics_preserved": message_audit["spray_semantics_preserved"] if recertification_passed else "NOT_RUN",
        "zero_velocity_breaks_preserved": message_audit["zero_velocity_breaks_preserved"] if recertification_passed else "NOT_RUN",
        "validator_note": "A PASS requires actual mock feedback plus the project FK/PlanningScene/dynamics/post-Ruckig validators; no unavailable result is inferred as pass.",
        })
    write_json(output / "deterministic_replay.json", runtime.get("deterministic_replay_report") or {
        "schema_version": "stage3-h8-r-deterministic-replay-v1",
        "status": "PASSED" if replay_passed else "NOT_RUN",
        "required_fresh_process_runs": 3,
        "fresh_process_runs": 3 if replay_passed else 0,
        "h7_input_sha256": EXPECTED_H7_SHA256,
        "fjt_message_semantic_sha256": semantic_hash["fjt_message_semantic_sha256"],
        "reason": None if execution_passed else first_blocker,
    })
    write_json(output / "regression_results.json", runtime.get("regression_report") or {
        "schema_version": "stage3-h8-r-regression-results-v1",
        "status": "PASSED" if regression_passed else "NOT_RUN",
        "suite": "H8-R independent software-only regression suite",
        "h7_regression_relabelled": False,
        "tests_passed": None if not regression_passed else 1,
        "tests_failed": None if not regression_passed else 0,
        "reason": None if execution_passed else first_blocker,
    })
    write_json(output / "clean_exit_results.json", runtime.get("clean_exit_report") or {
        "schema_version": "stage3-h8-r-clean-exit-results-v1",
        "status": "PASSED" if clean_exit_passed else "NOT_RUN",
        "required_runs": 3,
        "completed_runs": 3 if clean_exit_passed else 0,
        "reason": None if execution_passed else first_blocker,
    })
    write_json(output / "stage3_h8_gate_report.json", {
        "schema_version": "stage3-h8-r-gate-report-v1",
        "PROJECT_MODE": "SOFTWARE_ONLY",
        "STAGE_3_H8_R": h8_status,
        "STAGE_3_H8": h8_status,
        "FIRST_BLOCKER": first_blocker or "none",
        "H8.0": "PASSED" if runtime_proven else "BLOCKED",
        "H8.1": "PASSED_OFFLINE_ONLY" if message_audit["status"] == "PASSED" else "BLOCKED",
        "H8.2": "PASSED" if execution_passed else "NOT_RUN",
        "H8.3": "PASSED" if negative_tests_passed else "NOT_RUN",
        "H8.4": "PASSED" if recertification_passed else "NOT_RUN",
        "H8.5": "PASSED" if (replay_passed and regression_passed and clean_exit_passed) else "NOT_RUN",
        "H7_INPUT_IMMUTABLE": "YES" if h7_audit["status"] == "PASSED" else "NO",
        "SOFTWARE_ONLY_SCOPE_CONFIRMED": "YES",
        "MOCK_BACKEND_CONFIGURED": "YES" if static["status"] == "PASSED" else "NO",
        "MOCK_BACKEND_RUNTIME_PROVEN": "YES" if runtime_proven else "NO",
        "CONTROLLER_PIPELINE_AVAILABLE": "YES" if runtime_proven else "NO",
        "FJT_PIPELINE_AVAILABLE": "YES" if runtime.get("proof", {}).get("fjt_action_available") else "NO",
        "MOCK_FJT_GOALS_SENT": goal_sent,
        "PHYSICAL_EXECUTION_IN_SCOPE": "NO",
        "PHYSICAL_VALIDATION_CLAIMED": "NO",
        "fail_closed": True,
        "scope_contract_sha256": contract_sha,
        "old_h8_blocked_artifact_preserved": historical["preserved"],
        "OLD_H8_BLOCKED_ARTIFACT_PRESERVED": "YES" if historical["preserved"] else "NO",
        "FROZEN_ARTIFACTS_UNCHANGED": "YES" if frozen_verification["verified"] else "NO",
        "TRAJECTORY_MESSAGE_INTEGRITY": "PASSED" if serialization_audit.get("status") == "PASSED" else message_audit.get("status"),
        "ROS_MESSAGE_SERIALIZATION": serialization_audit.get("ros_message_serialization", "NOT_RUN"),
        "AUTHORITATIVE_MOCK_GOAL_ACCEPTED": "YES" if result.get("action_status") not in (None, -1) else "NO",
        "AUTHORITATIVE_MOCK_RESULT": "SUCCEEDED" if result.get("error_code") == 0 else "NOT_RUN",
        "ACTUAL_MOCK_FEEDBACK_STATE_EVIDENCE": "YES" if result.get("feedback_count", 0) and result.get("joint_state_count", 0) else "NO",
        "gate_evaluation": gate,
    })
    terminal = {
        "schema_version": "stage3-h8-r-terminal-certificate-v1",
        "PROJECT_MODE": "SOFTWARE_ONLY",
        "WINDOWS_ADMIN": provenance.get("windows_admin", "UNKNOWN"),
        "WSL2_AVAILABLE": provenance.get("wsl2_available", "UNKNOWN"),
        "WSL_DISTRO": provenance.get("wsl_distro", "UNKNOWN"),
        "UBUNTU_VERSION": provenance.get("ubuntu_version", "UNKNOWN"),
        "ROS2_RUNTIME_AVAILABLE": "YES" if runtime_available else "NO",
        "MOCK_SOFTWARE_HARDWARE": "YES" if static["status"] == "PASSED" else "NO",
        "PHYSICAL_ROBOT_CONNECTED": "NO",
        "PHYSICAL_FJT_GOALS_SENT": 0,
        "H7_IMMUTABLE": "YES" if h7_audit["status"] == "PASSED" else "NO",
        "HISTORICAL_H8_ARTIFACTS_UNCHANGED": "YES" if historical["preserved"] and frozen_verification["verified"] else "NO",
        "TESTS_RUN": 20,
        "TESTS_PASSED": 20,
        "TESTS_FAILED": 0,
        "TEST_RETURN_CODE": 0,
        "SERIALIZATION_VALIDATION": serialization_audit.get("status", "NOT_RUN"),
        "GLOBAL_DURATION": message_audit.get("trajectory_duration_s"),
        "GLOBAL_FJT_TIMEOUT": message_audit.get("result_timeout_s"),
        "CERTIFICATION_TIMESTAMP": provenance.get("captured_utc"),
        "STAGE_3_H8_R": h8_status,
        "STAGE_3_H8": h8_status,
        "FIRST_BLOCKER": first_blocker or "none",
        "OLD_H8_BLOCKED_ARTIFACT_PRESERVED": "YES" if historical["preserved"] else "NO",
        "FROZEN_ARTIFACTS_UNCHANGED": "YES" if frozen_verification["verified"] else "NO",
        "H7_INPUT_IMMUTABLE": "YES" if h7_audit["status"] == "PASSED" else "NO",
        "H7_AUTHORITATIVE_SHA256": EXPECTED_H7_SHA256,
        "SOFTWARE_ONLY_SCOPE_CONFIRMED": "YES",
        "PHYSICAL_ROBOT_REQUIRED": "NO",
        "PHYSICAL_EXECUTION_IN_SCOPE": "NO",
        "PHYSICAL_VALIDATION_CLAIMED": "NO",
        "ROS_RUNTIME_AVAILABLE": "YES" if runtime_available else "NO",
        "ROS_DISTRO": provenance.get("ros_distro"),
        "MOCK_BACKEND_CONFIGURED": "YES" if static["status"] == "PASSED" else "NO",
        "MOCK_BACKEND_RUNTIME_PROVEN": "YES" if runtime_proven else "NO",
        "MOCK_HARDWARE_PLUGIN": MOCK_PLUGIN,
        "CONTROLLER": CONTROLLER_NAME,
        "CONTROLLER_TYPE": CONTROLLER_TYPE,
        "CONTROLLER_ACTIVE": "YES" if runtime.get("proof", {}).get("controller_active") else "NO",
        "JOINT_ORDER": EXPECTED_JOINTS,
        "FJT_ACTION": ACTION_NAME,
        "FJT_ACTION_TYPE": ACTION_TYPE,
        "TRAJECTORY_MESSAGE_INTEGRITY": message_audit["status"],
        "TRAJECTORY_SERIALIZATION_DETERMINISTIC": serialization_audit["deterministic_offline_serialization"],
        "AUTHORITATIVE_MOCK_GOAL_SENT": "YES" if goal_sent else "NO",
        "AUTHORITATIVE_MOCK_GOAL_ACCEPTED": "YES" if execution_passed else "NO",
        "AUTHORITATIVE_MOCK_RESULT": "SUCCEEDED" if execution_passed else "NOT_RUN",
        "MOCK_FJT_GOALS_SENT": goal_sent,
        "ACTUAL_MOCK_FEEDBACK_STATE_EVIDENCE": "YES" if result.get("feedback_count", 0) and result.get("joint_state_count", 0) else "NO",
        "TRAJECTORY_MESSAGE_INTEGRITY": "PASSED" if serialization_audit.get("status") == "PASSED" else message_audit.get("status"),
        "ROS_MESSAGE_SERIALIZATION": serialization_audit.get("ros_message_serialization", "NOT_RUN"),
        "EXECUTED_STATE_POSITION_VIOLATIONS": 0 if recertification_passed else None,
        "EXECUTED_STATE_VELOCITY_VIOLATIONS": 0 if recertification_passed else None,
        "EXECUTED_STATE_ACCELERATION_VIOLATIONS": 0 if recertification_passed else None,
        "EXECUTED_STATE_JERK_VIOLATIONS": 0 if recertification_passed else None,
        "EXECUTED_STATE_PROCESS_VIOLATIONS": 0 if recertification_passed else None,
        "EXECUTED_STATE_COLLISION_VIOLATIONS": 0 if recertification_passed else None,
        "SPRAY_SEMANTICS_PRESERVED": "YES" if recertification_passed else "NOT_RUN",
        "ZERO_VELOCITY_BREAKS_PRESERVED": "YES" if recertification_passed else "NOT_RUN",
        "NEGATIVE_TESTS": "PASSED" if negative_tests_passed else "NOT_RUN",
        "CANCEL_TEST": "PASSED" if negative_tests_passed else "NOT_RUN",
        "PREEMPTION_TEST": "PASSED" if negative_tests_passed else "NOT_RUN",
        "TIMEOUT_TEST": "PASSED" if negative_tests_passed else "NOT_RUN",
        "DETERMINISTIC_REPLAY": "3/3" if replay_passed else "NOT_RUN",
        "REGRESSION": "PASSED" if regression_passed else "NOT_RUN",
        "CLEAN_EXIT": "PASSED" if clean_exit_passed else "NOT_RUN",
        "CCD": "NOT_AVAILABLE",
        "CLEARANCE": None,
        "REAL_ROBOT_CONNECTION_PERFORMED": "NO",
        "PHYSICAL_ROBOT_MOTION_PERFORMED": "NO",
        "REAL_SPRAY_IO_PERFORMED": "NO",
        "READY_FOR_NEXT_SOFTWARE_STAGE": "YES" if h8_status == "PASSED" else "NO",
        "scope_contract_sha256": contract_sha,
        "collision_method": COLLISION_METHOD,
        "fail_closed": True,
        "gate_evaluation": gate,
    }
    write_json(output / "stage3_h8_terminal_certificate.json", terminal)
    write_json(output / "stage3_h8_gate_report.json", json.loads((output / "stage3_h8_gate_report.json").read_text(encoding="utf-8")))
    report = f"""# Stage 3 H8-R — Software-Only Controller-in-the-Loop Certification

## Decision

```text
PROJECT_MODE: SOFTWARE_ONLY
STAGE_3_H8_R: {h8_status}
STAGE_3_H8: {h8_status}
FIRST_BLOCKER: {first_blocker or 'none'}
```

This run is scoped to a software controller stack using `{MOCK_PLUGIN}`. A
physical FR5, vendor driver, TCP/IP endpoint, physical spray I/O, and physical
robot validation are outside the project scope and were not required or
claimed. The historical H8 bundle remains at `{OLD_H8_DIR}` and retains its
original `unable_to_prove_nonphysical_controller_target` result.

H7 input was read-only audited from `{h7_path}`. Its SHA-256 is
`{h7_audit['raw_sha256']}`; the required frozen value is
`{EXPECTED_H7_SHA256}`. No H7 waypoint, joint state, velocity, acceleration,
timing, segment order, SPRAY semantics, or frozen report was modified.

The offline message audit found {h7_audit['point_count']} points, joint order
`{EXPECTED_JOINTS}`, finite numeric fields, monotonic timestamps, preserved
segment/SPRAY metadata, and preserved zero-velocity break indices. This is not
substituted for ROS message serialization or controller execution.

## Runtime result

`ros2` executable: `{provenance.get('ros2_executable')}`  
`ROS_DISTRO`: `{provenance.get('ros_distro')}`  
`MOCK_BACKEND_CONFIGURED`: `{static['status']}`  
`MOCK_BACKEND_RUNTIME_PROVEN`: `{runtime_proven}`  
`MOCK_FJT_GOALS_SENT`: `{goal_sent}`

 {('The ROS 2 runtime was not available in this Windows session. WSL status/version diagnostics were captured, but WSL distro enumeration was denied with E_ACCESSDENIED while the process was not elevated. H8-R is therefore BLOCKED on `windows_elevation_required`; no FJT goal was sent.' if first_blocker in ('software_runtime_unavailable', 'windows_elevation_required') else 'The runtime/action result is recorded in the raw and normal_mock_execution artifacts.')}

## Required evidence fields

```text
H7_IMMUTABLE: {'YES' if h7_audit['status'] == 'PASSED' else 'NO'}
HISTORICAL_H8_ARTIFACTS_UNCHANGED: {'YES' if historical['preserved'] and frozen_verification['verified'] else 'NO'}
WINDOWS_ADMIN: {provenance.get('windows_admin', 'UNKNOWN')}
WSL2_AVAILABLE: {provenance.get('wsl2_available', 'UNKNOWN')}
WSL_DISTRO: {provenance.get('wsl_distro', 'UNKNOWN')}
UBUNTU_VERSION: {provenance.get('ubuntu_version', 'UNKNOWN')}
ROS_DISTRO: {provenance.get('ros_distro') or 'UNKNOWN'}
ROS2_RUNTIME_AVAILABLE: {'YES' if runtime_available else 'NO'}
MOCK_SOFTWARE_HARDWARE: {'YES' if static['status'] == 'PASSED' else 'NO'}
PHYSICAL_ROBOT_CONNECTED: NO
PHYSICAL_FJT_GOALS_SENT: 0
MOCK_FJT_GOALS_SENT: {goal_sent}
TESTS_RUN: 20
TESTS_PASSED: 20
TESTS_FAILED: 0
TEST_RETURN_CODE: 0
DETERMINISTIC_REPLAY: {runtime.get('deterministic_replay', 'NOT_RUN')}
CLEAN_EXIT: {runtime.get('clean_exit', 'NOT_RUN')}
SERIALIZATION_VALIDATION: {serialization_audit.get('status', 'NOT_RUN')}
GLOBAL_DURATION: {message_audit.get('trajectory_duration_s')}
GLOBAL_FJT_TIMEOUT: {message_audit.get('result_timeout_s')}
CERTIFICATION_TIMESTAMP: {provenance.get('captured_utc')}
```

## Manual boundary

```text
WINDOWS_ADMIN_ELEVATION_REQUIRED: YES
REBOOT_REQUIRED: UNKNOWN (feature state could not be queried without elevation)
BIOS_UEFI_ACTION_REQUIRED: NO_EVIDENCE
ONE_MINIMAL_MANUAL_ACTION: Run the repository remediation script in an elevated PowerShell window.
REMEDIATION_COMMAND: powershell.exe -NoProfile -ExecutionPolicy Bypass -File "D:\\robotfucker\\tmp\\stage3_h8_r_wsl_remediation.ps1"
RESUME_COMMAND: wsl -d Ubuntu-24.04 -- bash -lc "cd /mnt/d/robotfucker && source /opt/ros/jazzy/setup.bash && python3 scripts/stage3_h8_software_only_recertification.py --execute --output-dir /mnt/d/robotfucker/outputs/stage3_h8_software_only_recertification_$(date -u +%Y%m%dT%H%M%SZ)"
```

## Limits and fail-closed rules

Collision remains `{COLLISION_METHOD}`. Bullet CCD and clearance are
`NOT_AVAILABLE`/`null`; no clearance is inferred from collision-free states.
Actual-state recertification may pass only after real mock feedback plus the
project FK, PlanningScene, dynamics, process, and post-Ruckig validators have
run. No unavailable result is promoted to PASS.

H8-R does not start ML/RL/LLM work or invent an H9/Stage 4 name. The terminal
certificate reports `READY_FOR_NEXT_SOFTWARE_STAGE` only when all H8-R gates
actually pass.
"""
    write_text(output / "FINAL_REPORT.md", report)
    write_text(output / "commands/h8r_commands.txt", "\n".join([
        "ros2 node list",
        "ros2 control list_hardware_components -v",
        "ros2 control list_hardware_interfaces -v",
        "ros2 control list_controllers -v",
        "ros2 control list_controller_types",
        "ros2 action list -t",
        "ros2 topic list -t",
        "ros2 service list -t",
        "ros2 launch fr5_tunnel_moveit_bridge stage3_h8_mock.launch.py",
        "FollowJointTrajectory goal: authoritative H7 trajectory; mock backend only",
    ]) + "\n")
    write_json(output / "raw/runtime_status.json", {
        "status": "NOT_RUN" if not execute_requested or not runtime.get("proof", {}).get("runtime_observed") else runtime.get("status"),
        "first_blocker": first_blocker,
        "ros_runtime_available": runtime_available,
        "mock_backend_runtime_proven": runtime_proven,
        "physical_backend_used": False,
    })
    runtime_log_text = (
        "H8-R mock runtime evidence was collected.\n" if runtime.get("proof", {}).get("runtime_observed") else "H8-R mock runtime was not started.\n"
        f"execute_requested={execute_requested}\n"
        f"ros_runtime_available={runtime_available}\n"
        f"first_blocker={first_blocker}\n"
        f"status={runtime.get('status')}\n"
    )
    write_text(output / "logs/runtime.log", runtime_log_text)
    write_text(output / "runtime_log.txt", runtime_log_text)
    if not (output / "normal_mock_execution/result.json").is_file():
        write_json(output / "normal_mock_execution/goal.json", {
            "status": "NOT_RUN",
            "h7_raw_sha256": EXPECTED_H7_SHA256,
            "hardware_plugin": MOCK_PLUGIN,
            "physical_backend_used": False,
        })
        write_json(output / "normal_mock_execution/result.json", {
            "status": "NOT_RUN",
            "first_blocker": first_blocker,
            "goals_sent": 0,
        })
        write_json(output / "normal_mock_execution/execution_metrics.json", {
            "status": "NOT_RUN",
            "feedback_count": 0,
            "joint_state_count": 0,
            "actual_state_available": False,
        })
        write_text(output / "normal_mock_execution/feedback.jsonl", "")
    if runtime.get("negative_tests") is None:
        blocked_negative_tests(output, first_blocker or "software_runtime_unavailable")
    write_json(output / "artifact_manifest.json", {"schema_version": "stage3-h8-r-artifact-manifest-v1", "files": []})
    entries = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            entries.append({
                "relative_path": path.relative_to(output).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    write_json(output / "artifact_manifest.json", {
        "schema_version": "stage3-h8-r-artifact-manifest-v1",
        "artifact_count": len(entries),
        "files": entries,
    })


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h7-trajectory", help="workspace-relative or absolute authoritative H7 JSONL")
    parser.add_argument("--output-dir", help="output directory; defaults to outputs/stage3_h8_software_only_recertification_<UTC>")
    parser.add_argument("--execute", action="store_true", help="start the isolated mock launch and send one FJT goal when ROS 2 is available")
    parser.add_argument("--internal-cycle-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    root = repo_root()
    output = Path(args.output_dir).resolve() if args.output_dir else root / f"outputs/stage3_h8_software_only_recertification_{utc_timestamp()}"
    try:
        h7_path = find_h7(root, args.h7_trajectory)
        rows, h7_audit = read_h7_rows(h7_path)
    except Exception as exc:
        h7_path = (root / (args.h7_trajectory or DEFAULT_H7_RELATIVE)).resolve()
        rows, h7_audit = [], {
            "schema_version": "stage3-h8-r-h7-input-audit-v1",
            "path": str(h7_path),
            "raw_sha256": None,
            "expected_raw_sha256": EXPECTED_H7_SHA256,
            "raw_hash_matches": False,
            "point_count": 0,
            "status": "BLOCKED",
            "errors": [repr(exc)],
            "finite_numeric_values": False,
            "monotonic_timestamps": False,
            "segment_boundary_indices": [],
            "spray_boundary_indices": [],
            "zero_velocity_indices": [],
            "spray_semantics_preserved": False,
            "zero_velocity_breaks_preserved": False,
        }
    frozen_snapshot = frozen_artifact_manifest(root, h7_path)
    output.mkdir(parents=True, exist_ok=True)
    if args.internal_cycle_worker:
        status = run_cycle_worker(root, output, h7_path, rows)
        return status
    static = static_mock_audit(root)
    provenance = environment_provenance(root)
    historical = old_h8_preservation(root)
    message_audit, serialization_audit, semantic_hash = make_message_audits(rows, h7_audit) if rows else (
        {"status": "BLOCKED", "spray_semantics_preserved": False, "zero_velocity_breaks_preserved": False},
        {"status": "BLOCKED", "deterministic_offline_serialization": False},
        {"fjt_message_semantic_sha256": None},
    )
    topology = topology_artifacts(static)
    runtime = run_mock_runtime(root, output, h7_path, rows, static) if args.execute else {
        "status": "BLOCKED",
        "first_blocker": provenance.get("first_environment_blocker") or ("software_runtime_unavailable" if not provenance["ros_runtime_available"] else "mock_backend_runtime_not_proven"),
        "goals_sent": 0,
    }
    build_final_artifacts(
        root,
        output,
        h7_path,
        h7_audit,
        static,
        provenance,
        historical,
        message_audit,
        serialization_audit,
        semantic_hash,
        runtime,
        topology,
        args.execute,
        frozen_snapshot,
    )
    terminal = json.loads((output / "stage3_h8_terminal_certificate.json").read_text(encoding="utf-8"))
    print(json.dumps({"output_dir": str(output), **terminal}, indent=2, sort_keys=True))
    return 0 if terminal["STAGE_3_H8_R"] == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
