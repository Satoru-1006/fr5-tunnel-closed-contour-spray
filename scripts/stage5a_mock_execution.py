"""Execute the isolated Stage 5A D65 -> ROS2 mock chain.

The runner is intentionally stage-local.  It creates fresh evidence under
``outputs/STAGE5A_MOCK_EXECUTION`` and never writes D65, the canonical floor,
or protected promotion artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.stage5a_trajectory_adapter import (  # noqa: E402
    JOINT_NAMES,
    FrozenTrajectory,
    build_goal,
    load_frozen_trajectory,
    preflight,
    sha256_file,
)


AUTO = {
    "auto0": {
        "trajectory_id": "adversarial_0100",
        "path": ROOT / "outputs/D59_OFFLINE_ALGORITHM_SYSTEM_CLOSURE/timing_scale025_auto0/trajectories/adversarial_0100.csv",
        "expected_states": 7005,
        "expected_intervals": 7004,
        "profile": ROOT / "outputs/D59_OFFLINE_ALGORITHM_SYSTEM_CLOSURE/profile_j_timing025_auto0/summary.json",
    },
    "auto1": {
        "trajectory_id": "adversarial_0101",
        "path": ROOT / "outputs/D59_OFFLINE_ALGORITHM_SYSTEM_CLOSURE/timing_scale025_corrected_auto1/trajectories/adversarial_0101.csv",
        "expected_states": 7030,
        "expected_intervals": 7029,
        "profile": ROOT / "outputs/D59_OFFLINE_ALGORITHM_SYSTEM_CLOSURE/profile_j_timing025_auto1/summary.json",
    },
}
TCP_CONFIG = ROOT / "ros2_moveit_bridge/config/stage5_mock_tcp.yaml"
STAGE5_XACRO = ROOT / "ros2_moveit_bridge/config/stage5_mock.urdf.xacro"
D65_URDF = ROOT / "outputs/stage3_h4_5_robot_fixture_geometry_codesign_20260809T000000Z/derived_robot_model.urdf"
OPEN_ARCH_POSES = ROOT / "outputs/internal_wiper_moveit_inputs/open_arch_tcp_poses_base_link.csv"
OPEN_ARCH_SEEDS = ROOT / "outputs/internal_wiper_moveit_inputs/open_arch_seed_joints.csv"
PROTECTED_FILES = [
    ROOT / "outputs/D64_PROJECT_FREEZE/D64_FROZEN_ARTIFACT_MANIFEST.json",
    ROOT / "outputs/D65_FINAL_CLOSURE/D65_FINAL_CLOSURE_LEDGER.json",
    ROOT / "outputs/D47_STAGE4B_ROLLING_CHAMPION_V1/PROMOTED/B3/metrics.json",
]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def run_command(command: str, timeout: int, *, env: dict[str, str] | None = None) -> dict[str, Any]:
    started = time.monotonic()
    try:
        proc = subprocess.run(["bash", "-lc", command], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, env=env)
        return {"command": command, "exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr, "timed_out": False, "elapsed_s": round(time.monotonic() - started, 3)}
    except subprocess.TimeoutExpired as exc:
        return {"command": command, "exit_code": None, "stdout": exc.stdout or "", "stderr": exc.stderr or "", "timed_out": True, "elapsed_s": round(time.monotonic() - started, 3)}


def ros_prefix() -> str:
    return f"source /opt/ros/jazzy/setup.bash && source {ROOT / 'install/setup.bash'}"


def sha_snapshot() -> dict[str, str | None]:
    paths = [item["path"] for item in AUTO.values()] + [item["profile"] for item in AUTO.values()] + [D65_URDF, TCP_CONFIG, STAGE5_XACRO] + PROTECTED_FILES
    return {str(path.resolve()): sha256_file(path) if path.is_file() else None for path in paths}


def file_ref(path: Path, role: str) -> dict[str, Any]:
    return {"path": str(path.resolve()), "role": role, "exists": path.is_file(), "bytes": path.stat().st_size if path.is_file() else None, "sha256": sha256_file(path) if path.is_file() else None}


def load_trajectories() -> dict[str, FrozenTrajectory]:
    result = {}
    for name, spec in AUTO.items():
        result[name] = load_frozen_trajectory(spec["path"], trajectory_id=spec["trajectory_id"], expected_states=spec["expected_states"], expected_intervals=spec["expected_intervals"])
    if result["auto0"].source_sha256 == result["auto1"].source_sha256:
        raise RuntimeError("auto0_auto1_source_hash_collision")
    return result


def parse_model(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    joints: dict[str, dict[str, Any]] = {}
    for joint in root.findall("joint"):
        name = joint.attrib.get("name", "")
        if name not in JOINT_NAMES:
            continue
        axis = joint.find("axis")
        origin = joint.find("origin")
        limit = joint.find("limit")
        collision_meshes = [mesh.attrib.get("filename") for mesh in root.findall(f"link[@name='{joint.find('child').attrib['link']}']/collision/geometry/mesh")]
        joints[name] = {
            "type": joint.attrib.get("type"),
            "parent": joint.find("parent").attrib.get("link") if joint.find("parent") is not None else None,
            "child": joint.find("child").attrib.get("link") if joint.find("child") is not None else None,
            "axis": axis.attrib.get("xyz") if axis is not None else None,
            "origin_xyz": origin.attrib.get("xyz") if origin is not None else None,
            "origin_rpy": origin.attrib.get("rpy") if origin is not None else None,
            "lower": limit.attrib.get("lower") if limit is not None else None,
            "upper": limit.attrib.get("upper") if limit is not None else None,
            "velocity": limit.attrib.get("velocity") if limit is not None else None,
            "effort": limit.attrib.get("effort") if limit is not None else None,
            "collision_meshes": collision_meshes,
        }
    tcp = root.find("joint[@name='spray_tcp_fixed_joint']")
    links = [link.attrib.get("name") for link in root.findall("link")]
    return {"robot_name": root.attrib.get("name"), "joint_names": list(joints), "joints": joints, "links": links, "tcp": {"parent": tcp.find("parent").attrib.get("link") if tcp is not None else None, "child": tcp.find("child").attrib.get("link") if tcp is not None else None, "xyz": tcp.find("origin").attrib.get("xyz") if tcp is not None else None, "rpy": tcp.find("origin").attrib.get("rpy") if tcp is not None else None}}


def model_preflight(expanded: Path) -> dict[str, Any]:
    d65 = parse_model(D65_URDF)
    stage5 = parse_model(expanded)
    equivalent = True
    differences: list[dict[str, Any]] = []
    for name in JOINT_NAMES:
        for key in ("type", "parent", "child", "axis", "origin_xyz", "origin_rpy", "lower", "upper", "velocity", "effort", "collision_meshes"):
            if d65["joints"].get(name, {}).get(key) != stage5["joints"].get(name, {}).get(key):
                equivalent = False
                differences.append({"joint": name, "field": key, "d65": d65["joints"].get(name, {}).get(key), "stage5": stage5["joints"].get(name, {}).get(key)})
    for key in ("parent", "child", "xyz", "rpy"):
        if d65["tcp"].get(key) != stage5["tcp"].get(key):
            equivalent = False
            differences.append({"tcp_field": key, "d65": d65["tcp"].get(key), "stage5": stage5["tcp"].get(key)})
    raw_xacro = STAGE5_XACRO.read_text(encoding="utf-8")
    return {
        "schema_version": "stage5a-fr5-model-preflight-v1",
        "d65_model_sha256": sha256_file(D65_URDF),
        "stage5_expanded_model_sha256": sha256_file(expanded),
        "d65_model": d65,
        "stage5_model": stage5,
        "joint_mapping": {"d65_columns": JOINT_NAMES, "ros_joint_names": JOINT_NAMES, "exact_order": True},
        "structural_equivalence_ignoring_ros2_control_initial_values": equivalent,
        "differences": differences,
        "generic_system_only": "mock_components/GenericSystem" in raw_xacro and "FairinoHardwareInterface" not in raw_xacro,
        "units": "radian joint coordinates, SI metre URDF/TCP geometry",
        "passed": bool(equivalent and stage5["tcp"]["xyz"] == "0 0 0.150" and stage5["tcp"]["rpy"] == "0 0 0"),
    }


def quat_from_matrix(matrix: np.ndarray) -> list[float]:
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return [float((matrix[2, 1] - matrix[1, 2]) / s), float((matrix[0, 2] - matrix[2, 0]) / s), float((matrix[1, 0] - matrix[0, 1]) / s), float(0.25 * s)]
    index = int(np.argmax(np.diag(matrix)))
    if index == 0:
        s = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        return [float(0.25 * s), float((matrix[0, 1] + matrix[1, 0]) / s), float((matrix[0, 2] + matrix[2, 0]) / s), float((matrix[2, 1] - matrix[1, 2]) / s)]
    if index == 1:
        s = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        return [float((matrix[0, 1] + matrix[1, 0]) / s), float(0.25 * s), float((matrix[1, 2] + matrix[2, 1]) / s), float((matrix[0, 2] - matrix[2, 0]) / s)]
    s = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
    return [float((matrix[0, 2] + matrix[2, 0]) / s), float((matrix[1, 2] + matrix[2, 1]) / s), float(0.25 * s), float((matrix[1, 0] - matrix[0, 1]) / s)]


def make_environment(output: Path) -> tuple[Path, Path]:
    with OPEN_ARCH_POSES.open(encoding="utf-8", newline="") as handle:
        poses = list(csv.DictReader(handle))
    with OPEN_ARCH_SEEDS.open(encoding="utf-8", newline="") as handle:
        seeds = list(csv.DictReader(handle))
    if len(poses) != 181 or len(seeds) != 181:
        raise RuntimeError(f"stage0_1_open_arch_count_mismatch:{len(poses)}/{len(seeds)}")
    points = np.asarray([[float(row[key]) for key in ("x", "y", "z")] for row in poses], dtype=float)
    normals = np.asarray([[float(row[key]) for key in ("nx", "ny", "nz")] for row in poses], dtype=float)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-15)
    stand_off, wall_thickness, y_thickness = 0.260, 0.025, 1.10
    wall_points = points - stand_off * normals
    objects: list[dict[str, Any]] = []
    for index in range(len(points) - 1):
        segment = wall_points[index + 1] - wall_points[index]
        length = float(np.linalg.norm(segment))
        if length <= 1e-9:
            continue
        x_axis = segment / length
        y_axis = np.asarray([0.0, 1.0, 0.0])
        z_axis = np.cross(x_axis, y_axis)
        z_axis /= max(float(np.linalg.norm(z_axis)), 1e-15)
        if float(np.dot(z_axis, normals[index])) < 0.0:
            z_axis = -z_axis
        rotation = np.column_stack((x_axis, y_axis, z_axis))
        center = 0.5 * (wall_points[index] + wall_points[index + 1]) - 0.5 * wall_thickness * normals[index]
        objects.append({"id": f"stage5a_tunnel_wall_{index:03d}", "center_m": center.tolist(), "dimensions_m": [length + wall_thickness, y_thickness, wall_thickness], "quaternion_xyzw": quat_from_matrix(rotation), "source_segment": index})
    environment = {
        "schema_version": "stage5a-open-arch-tunnel-planning-scene-v1",
        "kind": "open_arch_horseshoe_tunnel_companion",
        "frame_id": "base_link",
        "base_to_tunnel_transform": {"xyz_m": [0.0, 0.0, 0.0], "rpy_rad": [0.0, 0.0, 0.0], "identity": True},
        "source": {"tcp_poses": file_ref(OPEN_ARCH_POSES, "authoritative Stage 0/1 181-point open-arch TCP pose input"), "seed_joints": file_ref(OPEN_ARCH_SEEDS, "authoritative Stage 0/1 181-point open-arch seed input"), "point_count": 181, "legacy_720_assets_used": False},
        "geometry": {"wall_sign": -1.0, "stand_off_m": stand_off, "wall_thickness_m": wall_thickness, "y_thickness_m": y_thickness, "segment_stride": 1, "open_path": True, "floor_included": False, "acm_modified": False},
        "objects": objects,
    }
    env_path = output / "stage5a_tunnel_environment.json"
    write_json(env_path, environment)
    # The existing native MoveIt/Bullet probe accepts a directory of STL mesh
    # parts and applies one historical pose (.32,-.05,-.38).  Encode the
    # inverse of that fixed pose into each local part so the resulting world
    # geometry is exactly the manifest geometry in base_link.
    parts = output / "stage5a_bullet_parts"
    parts.mkdir()
    fixed_offset = np.asarray([0.32, -0.05, -0.38], dtype=float)
    faces = ((0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6), (0, 4, 5), (0, 5, 1), (1, 5, 6), (1, 6, 2), (2, 6, 7), (2, 7, 3), (3, 7, 4), (3, 4, 0))
    for item in objects:
        qx, qy, qz, qw = item["quaternion_xyzw"]
        # Quaternion rotation matrix, with the same xyzw convention as ROS.
        rotation = np.asarray([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)], [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)], [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])
        half = 0.5 * np.asarray(item["dimensions_m"], dtype=float)
        center = np.asarray(item["center_m"], dtype=float)
        vertices = [center + rotation @ np.asarray([sx * half[0], sy * half[1], sz * half[2]]) - fixed_offset for sx, sy, sz in ((-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1))]
        path = parts / f"{item['id']}.stl"
        lines = [f"solid {item['id']}"]
        for a, b, c in faces:
            normal = np.cross(vertices[b] - vertices[a], vertices[c] - vertices[a])
            normal /= max(float(np.linalg.norm(normal)), 1e-15)
            lines.append(f"  facet normal {normal[0]:.17g} {normal[1]:.17g} {normal[2]:.17g}")
            lines.append("    outer loop")
            for vertex in (vertices[a], vertices[b], vertices[c]):
                lines.append(f"      vertex {vertex[0]:.17g} {vertex[1]:.17g} {vertex[2]:.17g}")
            lines.extend(["    endloop", "  endfacet"])
        lines.append(f"endsolid {item['id']}")
        path.write_text("\n".join(lines) + "\n", encoding="ascii")
    write_json(output / "stage5a_bullet_geometry_transform.json", {"probe_fixed_pose_xyz_m": fixed_offset.tolist(), "parts_dir": str(parts), "compensation": "mesh_vertices_are_inverse-translated; runtime CollisionObject pose restores manifest base_link geometry", "object_count": len(objects)})
    return env_path, parts


def write_intervals(path: Path, trajectory: FrozenTrajectory) -> None:
    fields = ["interval_index", "time_start_s", "time_end_s", "process_order_index", "spray_state", "process_kind", "segment_id", "transition_id", "source_boundary"] + [f"q0_{i}" for i in range(6)] + [f"q1_{i}" for i in range(6)]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for index in range(trajectory.interval_count):
            row: dict[str, Any] = {"interval_index": index, "time_start_s": float(trajectory.times[index]), "time_end_s": float(trajectory.times[index + 1]), "process_order_index": index, "spray_state": "D65_REPLAY", "process_kind": "d65_frozen_replay", "segment_id": str(index), "transition_id": "", "source_boundary": "D65_state_boundary"}
            row.update({f"q0_{j}": float(trajectory.q[index, j]) for j in range(6)})
            row.update({f"q1_{j}": float(trajectory.q[index + 1, j]) for j in range(6)})
            writer.writerow(row)


def q_cmd_metrics(run_dir: Path, expected: FrozenTrajectory) -> dict[str, Any]:
    path = run_dir / "stage5a_controller_state_raw.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []
    errors: list[np.ndarray] = []
    refs: list[float] = []
    valid = 0
    for row in rows:
        if row.get("joint_names") != JOINT_NAMES:
            continue
        ref = np.asarray(row.get("reference", {}).get("positions", []), dtype=float)
        actual = np.asarray(row.get("feedback", {}).get("positions", []), dtype=float)
        if ref.shape != (6,) or actual.shape != (6,) or not np.isfinite(ref).all() or not np.isfinite(actual).all():
            continue
        valid += 1
        errors.append(np.abs(ref - actual))
        refs.append(float(row["reference"]["time_from_start"]["seconds"]))
    if not errors:
        return {"status": "not_available", "passed": False, "controller_state_rows": len(rows), "valid_rows": 0}
    matrix = np.vstack(errors)
    per_joint_max = np.max(matrix, axis=0)
    per_joint_rms = np.sqrt(np.mean(np.square(matrix), axis=0))
    last = matrix[-1]
    return {
        "status": "measured",
        "passed": bool(valid and float(np.max(matrix)) <= 1e-3 and float(np.max(last)) <= 1e-3),
        "controller_state_rows": len(rows),
        "valid_rows": valid,
        "telemetry_missing_or_dropped_samples": None,
        "reference_time_monotonic": bool(np.all(np.diff(refs) >= -1e-6)),
        "max_error_rad": float(np.max(matrix)),
        "mean_error_rad": float(np.mean(matrix)),
        "rms_error_rad": float(np.sqrt(np.mean(np.square(matrix)))),
        "per_joint_max_error_rad": per_joint_max.tolist(),
        "per_joint_rms_error_rad": per_joint_rms.tolist(),
        "final_state_error_rad": last.tolist(),
        "expected_source_state_count": expected.state_count,
        "semantics": "q_cmd=controller_state.reference.positions; q_mock=controller_state.feedback.positions",
    }


def runtime_probe() -> dict[str, Any]:
    commands = {
        "controllers": "ros2 control list_controllers -v --spin-time 1",
        "hardware_components": "ros2 control list_hardware_components -v --spin-time 1",
        "hardware_interfaces": "ros2 control list_hardware_interfaces -v --spin-time 1",
        "action": "ros2 action info /fairino5_controller/follow_joint_trajectory",
        "controller_params": "ros2 param dump /fairino5_controller",
        "controller_state_topic": "ros2 topic info -v /fairino5_controller/controller_state",
        "joint_states_topic": "ros2 topic info -v /joint_states",
        "node_list": "ros2 node list --include-hidden-nodes",
    }
    return {name: run_command(f"{ros_prefix()} && {command}", 45) for name, command in commands.items()}


def wait_ready(timeout_s: int = 120) -> dict[str, Any]:
    started = time.monotonic()
    last: dict[str, Any] = {}
    while time.monotonic() - started < timeout_s:
        probes = runtime_probe()
        controller_text = probes["controllers"]["stdout"]
        hardware_text = probes["hardware_components"]["stdout"]
        action_text = probes["action"]["stdout"]
        jtc = "fairino5_controller" in controller_text and "active" in controller_text
        jsb = "joint_state_broadcaster" in controller_text and "active" in controller_text
        mock = "mock_components/GenericSystem" in hardware_text or "GenericSystem" in hardware_text
        action = probes["action"]["exit_code"] == 0 and "follow_joint_trajectory" in action_text
        last = {"probes": probes, "controller_manager": probes["controllers"]["exit_code"] == 0, "hardware_component_active": mock, "joint_state_broadcaster_active": jsb, "joint_trajectory_controller_active": jtc, "follow_joint_trajectory_available": action}
        if all(last[key] for key in ("controller_manager", "hardware_component_active", "joint_state_broadcaster_active", "joint_trajectory_controller_active", "follow_joint_trajectory_available")):
            return last
        time.sleep(2.0)
    return last


def start_runtime(run_dir: Path, initial: Path) -> tuple[subprocess.Popen, Any]:
    log = (run_dir / "stage5a_runtime.log").open("w", encoding="utf-8")
    command = f"{ros_prefix()} && ros2 launch fr5_tunnel_moveit_bridge stage5_mock.launch.py runtime_tag:=stage5a initial_positions_file:={initial} tool_tcp_xyz:='0 0 0.150' tool_tcp_rpy:='0 0 0'"
    process = subprocess.Popen(["bash", "-lc", command], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, text=True)
    write_json(run_dir / "stage5a_runtime_process.json", {"pid": process.pid, "command": command, "started_utc": datetime.now(timezone.utc).isoformat(), "backend": "mock_components/GenericSystem"})
    return process, log


def stop_runtime(process: subprocess.Popen | None, log: Any) -> None:
    if process is not None and process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGINT)
            process.wait(timeout=30)
        except Exception:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=15)
            except Exception:
                pass
    log.close()


def run_client(run_dir: Path, timeout_s: int) -> dict[str, Any]:
    command = f"{ros_prefix()} && python3 {ROOT / 'scripts/stage5a_runtime_client.py'} --output {run_dir}"
    result = run_command(command, timeout_s)
    (run_dir / "stage5a_client_stdout.log").write_text(result["stdout"], encoding="utf-8")
    (run_dir / "stage5a_client_stderr.log").write_text(result["stderr"], encoding="utf-8")
    path = run_dir / "stage5a_action_execution.json"
    parsed = read_json(path) if path.exists() else {"goal_sent": False, "goal_accepted": False, "execution_completed": False, "action_result": "client_no_result"}
    parsed["client_process"] = {key: result[key] for key in ("exit_code", "timed_out", "elapsed_s")}
    write_json(path, parsed)
    return parsed


def run_moveit_validation(output: Path, name: str, trajectory: FrozenTrajectory, env_path: Path, run_dir: Path, output_name: str | None = None) -> dict[str, Any]:
    validator_dir = output / (output_name or f"moveit_{name}")
    validator_dir.mkdir()
    command = f"{ros_prefix()} && ros2 launch {ROOT / 'tools/stage5a_moveit_validation_launch.py'} trajectory_csv:={trajectory.source} runtime_dir:={run_dir} environment_json:={env_path} output_dir:={validator_dir}"
    result = run_command(command, max(1800, int(trajectory.times[-1] * 4)))
    (validator_dir / "stage5a_validator_stdout.log").write_text(result["stdout"], encoding="utf-8")
    (validator_dir / "stage5a_validator_stderr.log").write_text(result["stderr"], encoding="utf-8")
    summary = read_json(validator_dir / "stage5a_moveit_validation.json") if (validator_dir / "stage5a_moveit_validation.json").exists() else {"passed": False, "error": "validator_no_result"}
    summary["process"] = {key: result[key] for key in ("exit_code", "timed_out", "elapsed_s")}
    write_json(validator_dir / "stage5a_moveit_validation.json", summary)
    return summary


def run_bullet(output: Path, name: str, trajectory: FrozenTrajectory, parts: Path, output_name: str | None = None) -> dict[str, Any]:
    bullet_dir = output / (output_name or f"bullet_{name}")
    bullet_dir.mkdir()
    interval_csv = bullet_dir / "stage5a_bullet_intervals.csv"
    write_intervals(interval_csv, trajectory)
    stage26 = ROOT / "tmp/stage26_install"
    stage23 = ROOT / "tmp/stage23a7_install2"
    interposers = sorted((ROOT / "outputs/ik_graph_stage23b_representation_remediation").rglob("libstage23b_bullet_shape_interposer.so"))
    if not interposers:
        raise RuntimeError("stage5a_bullet_interposer_missing")
    am = ":".join(str(path) for path in (stage26 / "stage26_continuous_collision", ROOT / "install", stage23, ROOT / "install/fairino5_v6_moveit2_config", ROOT / "install/fairino_description", ROOT / "install/fr5_tunnel_moveit_bridge")) + ":/opt/ros/jazzy"
    ld = ":".join([str(stage26 / "stage26_continuous_collision/lib"), str(interposers[0].parent), str(stage23 / "lib"), str(ROOT / "install/lib"), "/opt/ros/jazzy/opt/sdformat_vendor/lib", "/opt/ros/jazzy/opt/gz_math_vendor/lib", "/opt/ros/jazzy/opt/gz_utils_vendor/lib", "/opt/ros/jazzy/lib/x86_64-linux-gnu", "/opt/ros/jazzy/lib"])
    command = " && ".join([ros_prefix(), f"source {stage26 / 'setup.bash'}", f"export AMENT_PREFIX_PATH={am}", f"export LD_LIBRARY_PATH={ld}", "export FR5_BULLET_SHAPE_MODE=use_shape_type", f"export LD_PRELOAD={interposers[0]}", f"ros2 launch {ROOT / 'tools/stage5a_bullet_launch.py'} interval_csv:={interval_csv} parts_dir:={parts} output_dir:={bullet_dir} backend:=bullet run_index:=1 capability_only:=false positive_controls:=false"])
    # Native two-state CCD is compiled but can still take tens of minutes for
    # the 7k-interval D65 paths. Keep the timeout above the measured runtime
    # envelope so a valid long run is not truncated by the harness.
    result = run_command(command, max(7200, int(trajectory.times[-1] * 20)))
    (bullet_dir / "stage5a_bullet_stdout.log").write_text(result["stdout"], encoding="utf-8")
    (bullet_dir / "stage5a_bullet_stderr.log").write_text(result["stderr"], encoding="utf-8")
    raw_path = bullet_dir / "stage26_continuous_robot_world_intervals.jsonl"
    raw = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines() if line.strip()] if raw_path.exists() else []
    collisions = [row for row in raw if row.get("continuous_collision")]
    expected = trajectory.interval_count
    summary = read_json(bullet_dir / "stage26_native_summary.json") if (bullet_dir / "stage26_native_summary.json").exists() else {}
    evidence = {"schema_version": "stage5a-bullet-robot-world-ccd-v1", "backend": "native MoveIt CollisionEnvBullet", "api": "CollisionEnvBullet::checkRobotCollision(req,res,state1,state2,acm)", "trajectory_id": trajectory.trajectory_id, "expected_intervals": expected, "checked_intervals": len(raw), "missing_intervals": expected - len(raw), "continuous_collision_count": len(collisions), "first_collision": collisions[0] if collisions else None, "validation_complete": bool(result["exit_code"] == 0 and len(raw) == expected and summary.get("all_intervals_executed") is True), "strict_continuous_self_collision": "not_available", "clearance": None, "environment_parts": str(parts), "runner": {key: result[key] for key in ("exit_code", "timed_out", "elapsed_s")}}
    evidence["status"] = "measured_complete" if evidence["validation_complete"] else "not_available_or_incomplete"
    write_json(bullet_dir / "stage5a_bullet_validation.json", evidence)
    return evidence


def run_bullet_parallel(output: Path, name: str, trajectory: FrozenTrajectory, parts: Path, shard_count: int = 4, output_name: str | None = None) -> dict[str, Any]:
    """Run the same native two-state Bullet API over disjoint exact shards."""
    bullet_dir = output / (output_name or f"bullet_{name}")
    bullet_dir.mkdir()
    interval_csv = bullet_dir / "stage5a_bullet_intervals.csv"
    write_intervals(interval_csv, trajectory)
    lines = interval_csv.read_text(encoding="utf-8").splitlines()
    header, rows = lines[0], lines[1:]
    shard_count = max(1, min(int(shard_count), len(rows)))
    stage26 = ROOT / "tmp/stage26_install"
    stage23 = ROOT / "tmp/stage23a7_install2"
    interposers = sorted((ROOT / "outputs/ik_graph_stage23b_representation_remediation").rglob("libstage23b_bullet_shape_interposer.so"))
    if not interposers:
        raise RuntimeError("stage5a_bullet_interposer_missing")
    am = ":".join(str(path) for path in (stage26 / "stage26_continuous_collision", ROOT / "install", stage23, ROOT / "install/fairino5_v6_moveit2_config", ROOT / "install/fairino_description", ROOT / "install/fr5_tunnel_moveit_bridge")) + ":/opt/ros/jazzy"
    ld = ":".join([str(stage26 / "stage26_continuous_collision/lib"), str(interposers[0].parent), str(stage23 / "lib"), str(ROOT / "install/lib"), "/opt/ros/jazzy/opt/sdformat_vendor/lib", "/opt/ros/jazzy/opt/gz_math_vendor/lib", "/opt/ros/jazzy/opt/gz_utils_vendor/lib", "/opt/ros/jazzy/lib/x86_64-linux-gnu", "/opt/ros/jazzy/lib"])
    prefix = " && ".join([ros_prefix(), f"source {stage26 / 'setup.bash'}", f"export AMENT_PREFIX_PATH={am}", f"export LD_LIBRARY_PATH={ld}", "export FR5_BULLET_SHAPE_MODE=use_shape_type", f"export LD_PRELOAD={interposers[0]}"])
    shard_records: list[dict[str, Any]] = []
    processes: list[tuple[subprocess.Popen, Any, Path]] = []
    started = time.monotonic()
    for shard_index in range(shard_count):
        shard_rows = rows[shard_index::shard_count]
        shard_dir = bullet_dir / f"shard_{shard_index:02d}"
        shard_dir.mkdir()
        shard_csv = shard_dir / "intervals.csv"
        shard_csv.write_text(header + "\n" + "\n".join(shard_rows) + "\n", encoding="utf-8")
        command = f"{prefix} && ros2 launch {ROOT / 'tools/stage5a_bullet_launch.py'} interval_csv:={shard_csv} parts_dir:={parts} output_dir:={shard_dir} backend:=bullet run_index:=1 capability_only:=false positive_controls:=false"
        log = (shard_dir / "launch.log").open("w", encoding="utf-8")
        process = subprocess.Popen(["bash", "-lc", command], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, text=True)
        processes.append((process, log, shard_dir))
        shard_records.append({"shard_index": shard_index, "interval_row_count": len(shard_rows), "interval_csv": str(shard_csv.resolve()), "output_dir": str(shard_dir.resolve()), "pid": process.pid, "command": command})
    for process, log, shard_dir in processes:
        timed_out = False
        try:
            exit_code = process.wait(timeout=7200)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except Exception:
                pass
            exit_code = process.wait(timeout=30)
        finally:
            log.close()
        record = next(item for item in shard_records if item["pid"] == process.pid)
        record.update({"exit_code": exit_code, "timed_out": timed_out})
    merged_intervals: dict[int, dict[str, Any]] = {}
    merged_states: dict[int, dict[str, Any]] = {}
    for record in shard_records:
        shard_dir = Path(record["output_dir"])
        raw_path = shard_dir / "stage26_continuous_robot_world_intervals.jsonl"
        self_path = shard_dir / "stage26_self_collision_states.jsonl"
        raw_rows: list[dict[str, Any]] = []
        if raw_path.exists():
            for line in raw_path.read_text(encoding="utf-8").splitlines():
                try:
                    raw_rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        self_rows: list[dict[str, Any]] = []
        if self_path.exists():
            for line in self_path.read_text(encoding="utf-8").splitlines():
                try:
                    self_rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        record["valid_interval_rows"] = len(raw_rows)
        record["valid_state_rows"] = len(self_rows)
        for row in raw_rows:
            merged_intervals[int(row["interval_index"])] = row
        for row in self_rows:
            merged_states[int(row["state_index"])] = row
    interval_rows = [merged_intervals[index] for index in sorted(merged_intervals)]
    state_rows = [merged_states[index] for index in sorted(merged_states)]
    (bullet_dir / "stage26_continuous_robot_world_intervals.jsonl").write_text("".join(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n" for row in interval_rows), encoding="utf-8")
    (bullet_dir / "stage26_self_collision_states.jsonl").write_text("".join(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n" for row in state_rows), encoding="utf-8")
    collisions = [row for row in interval_rows if row.get("continuous_collision")]
    expected = trajectory.interval_count
    complete_indices = sorted(merged_intervals) == list(range(expected))
    all_success = all(item.get("exit_code") == 0 and not item.get("timed_out") for item in shard_records)
    summary = {"backend": "bullet", "run_index": 1, "interval_count": len(interval_rows), "intervals_with_continuous_collision": len(collisions), "states_with_endpoint_self_collision": sum(bool(row.get("collision")) for row in state_rows), "all_intervals_executed": bool(all_success and complete_indices), "skipped_intervals": expected - len(interval_rows), "unchecked_intervals": expected - len(interval_rows), "continuous_contact_fraction_available": False, "parallel_shard_count": shard_count}
    write_json(bullet_dir / "stage26_native_summary.json", summary)
    write_json(bullet_dir / "stage5a_bullet_parallel_manifest.json", {"schema_version": "stage5a-bullet-parallel-v1", "trajectory_id": trajectory.trajectory_id, "expected_interval_count": expected, "shard_count": shard_count, "shards": shard_records, "merge": {"unique_interval_count": len(interval_rows), "unique_state_count": len(state_rows), "complete_interval_index_set": complete_indices, "deduplication": "interval_index/state_index exact-key merge"}})
    combined = "\n".join((Path(item["output_dir"]) / "launch.log").read_text(encoding="utf-8", errors="replace") for item in shard_records if (Path(item["output_dir"]) / "launch.log").exists())
    (bullet_dir / "stage5a_bullet_stdout.log").write_text(combined, encoding="utf-8")
    (bullet_dir / "stage5a_bullet_stderr.log").write_text("", encoding="utf-8")
    evidence = {"schema_version": "stage5a-bullet-robot-world-ccd-v1", "backend": "native MoveIt CollisionEnvBullet", "api": "CollisionEnvBullet::checkRobotCollision(req,res,state1,state2,acm)", "trajectory_id": trajectory.trajectory_id, "expected_intervals": expected, "checked_intervals": len(interval_rows), "missing_intervals": expected - len(interval_rows), "continuous_collision_count": len(collisions), "first_collision": collisions[0] if collisions else None, "validation_complete": bool(summary["all_intervals_executed"]), "strict_continuous_self_collision": "not_available", "clearance": None, "environment_parts": str(parts), "parallel_shard_count": shard_count, "runner": {"exit_code": 0 if all_success else 1, "timed_out": any(item.get("timed_out") for item in shard_records), "elapsed_s": time.monotonic() - started}}
    evidence["status"] = "measured_complete" if evidence["validation_complete"] else "not_available_or_incomplete"
    write_json(bullet_dir / "stage5a_bullet_validation.json", evidence)
    return evidence


def execute_replay(output: Path, name: str, trajectory: FrozenTrajectory, full_goal: dict[str, Any], smoke: bool = False, run_dir_name: str | None = None) -> dict[str, Any]:
    run_dir = output / (run_dir_name or ("plumbing_smoke" if smoke else f"replay_{name}"))
    run_dir.mkdir()
    goal = dict(full_goal)
    goal["points"] = full_goal["points"][:2] if smoke else full_goal["points"]
    if smoke:
        goal["goal_identity"] = dict(full_goal["goal_identity"])
        goal["goal_identity"]["smoke_subset"] = True
        goal["goal_identity"]["state_count_sent"] = 2
    write_json(run_dir / "stage5a_follow_joint_trajectory_goal.json", goal)
    initial = run_dir / "stage5a_initial_positions.yaml"
    initial.write_text("initial_positions:\n" + "\n".join(f"  {joint}: {float(trajectory.q[0, i]):.17g}" for i, joint in enumerate(JOINT_NAMES)) + "\n", encoding="utf-8")
    process = None
    log = None
    try:
        process, log = start_runtime(run_dir, initial)
        ready = wait_ready()
        write_json(run_dir / "stage5a_runtime_ready.json", ready)
        if not all(ready.get(key, False) for key in ("controller_manager", "hardware_component_active", "joint_state_broadcaster_active", "joint_trajectory_controller_active", "follow_joint_trajectory_available")):
            return {"name": name, "smoke": smoke, "ready": ready, "action": {"execution_completed": False, "action_result": "runtime_not_ready"}}
        action = run_client(run_dir, max(300, int(float(goal["points"][-1]["time_from_start"]["seconds"]) + 180)))
        write_json(run_dir / "stage5a_q_cmd_q_mock.json", q_cmd_metrics(run_dir, trajectory))
        return {"name": name, "smoke": smoke, "ready": ready, "action": action, "run_dir": str(run_dir), "q_cmd_q_mock": read_json(run_dir / "stage5a_q_cmd_q_mock.json")}
    finally:
        stop_runtime(process, log) if log is not None else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    output = (args.output_root or ROOT / "outputs/STAGE5A_MOCK_EXECUTION" / f"stage5a_mock_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}").resolve()
    output.mkdir(parents=True, exist_ok=False)
    before = sha_snapshot()
    write_json(output / "STAGE5A_INPUT_MANIFEST.json", {"schema_version": "stage5a-input-manifest-v1", "d65_frozen": True, "replan": False, "retime": False, "reruckig": False, "resample": False, "trajectories": {name: {**{key: value for key, value in spec.items() if key not in {"path", "profile"}}, "trajectory": file_ref(spec["path"], f"D65 frozen {name}"), "profile": file_ref(spec["profile"], f"D65 profile oracle {name}")} for name, spec in AUTO.items()}, "d65_model": file_ref(D65_URDF, "D65 expanded FR5 V6 model"), "tcp_config": file_ref(TCP_CONFIG, "explicit Stage5A TCP config"), "stage0_1_environment_inputs": [file_ref(OPEN_ARCH_POSES, "181-point open-arch poses"), file_ref(OPEN_ARCH_SEEDS, "181-point open-arch seeds")], "protected_snapshot": before})
    trajectories = load_trajectories()
    limits = {"j1": (-3.0543, 3.0543), "j2": (-4.6251, 1.4835), "j3": (-2.8274, 2.8274), "j4": (-4.6251, 1.4835), "j5": (-3.0543, 3.0543), "j6": (-3.0543, 3.0543)}
    v_limits = {joint: 3.15 if joint in {"j1", "j2", "j3"} else 3.2 for joint in JOINT_NAMES}
    a_limits = {joint: 0.7 for joint in JOINT_NAMES}
    trajectory_preflight = {}
    goals = {}
    for name, trajectory in trajectories.items():
        trajectory_preflight[name] = preflight(trajectory, position_limits=limits, velocity_limits=v_limits, acceleration_limits=a_limits)
        goals[name] = build_goal(trajectory)
        write_json(output / f"stage5a_{name}_adapter_preflight.json", trajectory_preflight[name])
        write_json(output / f"stage5a_{name}_goal.json", goals[name])
    env_path, bullet_parts = make_environment(output)
    # Expand the exact model used by the runtime using the first-state YAML.
    expand_initial = output / "model_initial_positions.yaml"
    expand_initial.write_text("initial_positions:\n" + "\n".join(f"  {joint}: {float(trajectories['auto0'].q[0, i]):.17g}" for i, joint in enumerate(JOINT_NAMES)) + "\n", encoding="utf-8")
    expanded = output / "stage5a_expanded_runtime.urdf"
    expand_cmd = f"{ros_prefix()} && python3 {ROOT / 'tools/stage5a_expand_xacro.py'} --xacro {STAGE5_XACRO} --initial-positions {expand_initial} --output {expanded}"
    expand_result = run_command(expand_cmd, 120)
    (output / "stage5a_expand_stdout.log").write_text(expand_result["stdout"], encoding="utf-8")
    (output / "stage5a_expand_stderr.log").write_text(expand_result["stderr"], encoding="utf-8")
    model = model_preflight(expanded) if expanded.exists() else {"passed": False, "error": "expanded_model_missing"}
    write_json(output / "stage5a_model_preflight.json", model)
    smoke = execute_replay(output, "plumbing", trajectories["auto0"], goals["auto0"], smoke=True)
    write_json(output / "stage5a_plumbing_smoke.json", smoke)
    replays = {name: execute_replay(output, name, trajectories[name], goals[name]) for name in ("auto0", "auto1")}
    write_json(output / "stage5a_replay_summary.json", replays)
    moveit = {name: run_moveit_validation(output, name, trajectories[name], env_path, Path(replays[name]["run_dir"])) for name in ("auto0", "auto1") if replays[name].get("run_dir")}
    bullet = {name: run_bullet(output, name, trajectories[name], bullet_parts) for name in ("auto0", "auto1")}
    after = sha_snapshot()
    protected_unchanged = before == after
    regression = run_command(f"{ros_prefix()} && python3 -m py_compile src/stage5a_trajectory_adapter.py scripts/stage5a_runtime_client.py scripts/stage5a_mock_execution.py scripts/stage5a_moveit_validator.py tools/stage5a_expand_xacro.py tools/stage5a_moveit_validation_launch.py tools/stage5a_bullet_launch.py && python3 -m pytest -q tests/test_stage5a_contract.py", 900)
    (output / "stage5a_regression_stdout.log").write_text(regression["stdout"], encoding="utf-8")
    (output / "stage5a_regression_stderr.log").write_text(regression["stderr"], encoding="utf-8")
    state = {"schema_version": "stage5a-execution-state-v1", "output_root": str(output), "input_manifest": str((output / "STAGE5A_INPUT_MANIFEST.json").resolve()), "trajectory_preflight": trajectory_preflight, "model_preflight": model, "tcp": yaml.safe_load(TCP_CONFIG.read_text(encoding="utf-8")), "planning_scene_environment": str(env_path), "plumbing_smoke": smoke, "replays": replays, "moveit": moveit, "bullet": bullet, "regression": {key: regression[key] for key in ("exit_code", "timed_out", "elapsed_s")}, "protected_before": before, "protected_after": after, "protected_unchanged": protected_unchanged}
    write_json(output / "stage5a_execution_state.json", state)
    validator = run_command(f"{ros_prefix()} && python3 {ROOT / 'tools/stage5_mock_validator.py'} --state {output / 'stage5a_execution_state.json'}", 120)
    (output / "stage5a_final_validator_stdout.log").write_text(validator["stdout"], encoding="utf-8")
    (output / "stage5a_final_validator_stderr.log").write_text(validator["stderr"], encoding="utf-8")
    ledger = read_json(output / "STAGE5A_FINAL_LEDGER.json") if (output / "STAGE5A_FINAL_LEDGER.json").exists() else {"status": "FAIL_CLOSED"}
    print(json.dumps({"output_root": str(output), "protected_unchanged": protected_unchanged, "auto0_action": replays.get("auto0", {}).get("action", {}).get("action_result"), "auto1_action": replays.get("auto1", {}).get("action", {}).get("action_result"), "final_status": ledger.get("status")}, ensure_ascii=False), flush=True)
    return 0 if ledger.get("status") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
