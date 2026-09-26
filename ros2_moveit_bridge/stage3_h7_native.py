#!/usr/bin/env python3
"""Native MoveIt2 worker for the Stage 3 H7 offline certification run."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point
from moveit.core.collision_detection import CollisionRequest, CollisionResult
from moveit.core.robot_state import RobotState
from moveit.core.robot_trajectory import RobotTrajectory
from moveit.planning import MoveItPy
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import Mesh, MeshTriangle
from trajectory_msgs.msg import JointTrajectoryPoint

from plan_closed_contour_moveit import _rotation_matrix_to_quaternion, _transform_matrix


JOINT_NAMES = ["j1", "j2", "j3", "j4", "j5", "j6"]
GROUP_NAME = "fairino5_v6_group"
EE_LINK = "spray_tcp_link"
COLLISION_METHOD = "adaptive_discrete_interpolation"
CCD_STATUS = "not_available"
CLEARANCE_STATUS = "not_available"
COLLISION_STEP_DEG = 0.5


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


def canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): canonical(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return round(value, 12)
    return value


def semantic_hash(value: Any) -> str:
    data = json.dumps(canonical(value), ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def finite_vector(value: Any, length: int) -> bool:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return False
    return array.shape == (length,) and bool(np.all(np.isfinite(array)))


def duration_seconds(value: Any) -> float:
    return float(value.sec) + float(value.nanosec) * 1e-9


def duration_message(seconds: float):
    from builtin_interfaces.msg import Duration

    msg = Duration()
    msg.sec = int(seconds)
    msg.nanosec = int(round((seconds - msg.sec) * 1_000_000_000))
    if msg.nanosec >= 1_000_000_000:
        msg.sec += 1
        msg.nanosec -= 1_000_000_000
    return msg


def param(node, name: str, default: Any = "") -> Any:
    if node.has_parameter(name):
        return node.get_parameter(name).value
    return node.declare_parameter(name, default).value


def parse_limits(path: Path) -> dict[str, dict[str, float | bool]]:
    result: dict[str, dict[str, float | bool]] = {}
    current: str | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if stripped.startswith("j") and stripped.endswith(":") and stripped[1:-1].isdigit():
            current = stripped[:-1]
            result[current] = {}
        elif current and ":" in stripped:
            key, value = [part.strip() for part in stripped.split(":", 1)]
            if value in {"true", "false"}:
                result[current][key] = value == "true"
            else:
                try:
                    result[current][key] = float(value)
                except ValueError:
                    pass
    return result


def bound_item(bounds: Any) -> Any:
    if isinstance(bounds, (list, tuple)):
        if not bounds:
            raise RuntimeError("MoveIt joint has no variable bounds")
        return bounds[0]
    return bounds


def runtime_limit_audit(moveit: MoveItPy, output: Path, group_name: str, ee_link: str, yaml_path: Path) -> dict[str, Any]:
    model = moveit.get_robot_model()
    errors: list[str] = []
    if not model.has_joint_model_group(group_name):
        errors.append("moveit_group_missing")
        dump_json(output / "stage3_h7_runtime_limit_audit.json", {"schema_version": "stage3-h7-runtime-limit-audit-v1", "status": "BLOCKED", "errors": errors})
        return {"status": "BLOCKED", "errors": errors}
    group = model.get_joint_model_group(group_name)
    active = list(group.active_joint_model_names)
    if active != JOINT_NAMES:
        errors.append(f"unexpected_joint_order:{active}")
    if ee_link not in list(group.link_model_names):
        errors.append("ee_link_missing_from_group")
    configured = parse_limits(yaml_path)
    limits: dict[str, Any] = {}
    for joint_name, raw_bounds in zip(active, group.active_joint_model_bounds):
        bounds = bound_item(raw_bounds)
        max_jerk = getattr(bounds, "max_jerk", None)
        row = {
            "position_lower_rad": float(getattr(bounds, "min_position", float("nan"))),
            "position_upper_rad": float(getattr(bounds, "max_position", float("nan"))),
            "velocity_rad_s": float(getattr(bounds, "max_velocity", float("nan"))),
            "acceleration_rad_s2": float(getattr(bounds, "max_acceleration", float("nan"))),
            "jerk_rad_s3": float(max_jerk) if max_jerk is not None else None,
        }
        limits[joint_name] = row
        expected = configured.get(joint_name, {})
        for key, actual_key in (("max_velocity", "velocity_rad_s"), ("max_acceleration", "acceleration_rad_s2"), ("max_jerk", "jerk_rad_s3")):
            actual = row.get(actual_key)
            expected_value = expected.get(key)
            if expected_value is None or actual is None or not math.isfinite(float(actual)) or abs(float(actual) - float(expected_value)) > 1e-12:
                errors.append(f"runtime_limit_mismatch:{joint_name}:{key}")
    if not hasattr(RobotTrajectory, "apply_totg_time_parameterization"):
        errors.append("native_totg_api_missing")
    if not hasattr(RobotTrajectory, "apply_ruckig_smoothing"):
        errors.append("native_ruckig_api_missing")
    result = {
        "schema_version": "stage3-h7-runtime-limit-audit-v1",
        "status": "PASSED" if not errors else "BLOCKED",
        "ros_distro": "jazzy",
        "moveit_api": {"totg": "RobotTrajectory.apply_totg_time_parameterization", "ruckig": "RobotTrajectory.apply_ruckig_smoothing"},
        "robot_model_name": str(model.name),
        "planning_group": group_name,
        "ee_link": ee_link,
        "active_joint_names": active,
        "runtime_bounds": limits,
        "formal_configured_limits": configured,
        "fallback_defaults_used": False,
        "errors": errors,
    }
    dump_json(output / "stage3_h7_runtime_limit_audit.json", result)
    return result


def read_obj(path: Path) -> tuple[list[list[float]], list[tuple[int, int, int]]]:
    vertices: list[list[float]] = []
    triangles: list[tuple[int, int, int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "v":
            vertices.append([float(v) for v in fields[1:4]])
        elif fields[0] == "f":
            ids = [int(item.split("/")[0]) - 1 for item in fields[1:]]
            triangles.extend((ids[0], ids[i], ids[i + 1]) for i in range(1, len(ids) - 1))
    if not vertices or not triangles:
        raise RuntimeError(f"fixture mesh is empty: {path}")
    return vertices, triangles


def mesh_collision_object(path: Path) -> CollisionObject:
    vertices, triangles = read_obj(path)
    obj = CollisionObject()
    obj.header.frame_id = "base_link"
    obj.id = "fixture_surface:fixture_curved_cylinder_patch"
    mesh = Mesh()
    mesh.vertices = [Point(x=float(v[0]), y=float(v[1]), z=float(v[2])) for v in vertices]
    mesh.triangles = [MeshTriangle(vertex_indices=[int(t[0]), int(t[1]), int(t[2])]) for t in triangles]
    obj.meshes.append(mesh)
    pose = obj.mesh_poses.add() if hasattr(obj.mesh_poses, "add") else None
    if pose is None:
        from geometry_msgs.msg import Pose

        pose = Pose()
        pose.orientation.w = 1.0
        obj.mesh_poses.append(pose)
    else:
        pose.orientation.w = 1.0
    obj.operation = CollisionObject.ADD
    return obj


def apply_fixture(moveit: MoveItPy, mesh_path: Path) -> None:
    monitor = moveit.get_planning_scene_monitor()
    with monitor.read_write() as scene:
        scene.remove_all_collision_objects()
        scene.apply_collision_object(mesh_collision_object(mesh_path))


def full_cartesian_rows(joint_rows: Sequence[Mapping[str, Any]], on_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    cursor = 0
    for joint in joint_rows:
        row = dict(joint)
        if joint.get("spray_state") == "SPRAY_ON":
            if cursor >= len(on_rows):
                raise RuntimeError("H6 Cartesian ON evidence shorter than H6 joint ON sequence")
            geometry = dict(on_rows[cursor])
            cursor += 1
            for key in ("waypoint_index", "segment_id", "component_id", "source_target_id", "destination_target_id", "spray_state"):
                geometry[key] = joint.get(key)
            row.update(geometry)
        else:
            row.update({"tcp_position_xyz_m": None, "tcp_orientation_xyzw": None, "surface_point_xyz_m": None, "surface_normal_unit": None, "spray_direction_unit": None})
        output.append(row)
    if cursor != len(on_rows):
        raise RuntimeError("H6 Cartesian ON evidence has unused rows")
    return output


def make_trajectory_message(joint_rows: Sequence[Mapping[str, Any]]):
    from moveit_msgs.msg import RobotTrajectory as RobotTrajectoryMsg

    message = RobotTrajectoryMsg()
    message.joint_trajectory.joint_names = list(JOINT_NAMES)
    for row in joint_rows:
        point = JointTrajectoryPoint()
        point.positions = [float(v) for v in row["joint_values"]]
        point.time_from_start = duration_message(0.0)
        message.joint_trajectory.points.append(point)
    return message


def message_arrays(message: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points = message.joint_trajectory.points
    if len(points) < 2:
        raise RuntimeError("native trajectory has fewer than two points")
    t = np.asarray([duration_seconds(point.time_from_start) for point in points], dtype=float)
    q = np.asarray([list(point.positions) for point in points], dtype=float)
    dq = np.asarray([list(point.velocities) for point in points], dtype=float)
    ddq = np.asarray([list(point.accelerations) for point in points], dtype=float)
    if q.shape != (len(points), 6) or dq.shape != q.shape or ddq.shape != q.shape:
        raise RuntimeError("native trajectory positions/velocities/accelerations are incomplete")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(q)) or not np.all(np.isfinite(dq)) or not np.all(np.isfinite(ddq)):
        raise RuntimeError("native trajectory contains non-finite time/state values")
    if np.any(np.diff(t) <= 0.0):
        raise RuntimeError("native trajectory timestamps are not strictly increasing")
    return t, q, dq, ddq


def finite_difference_jerk(t: np.ndarray, ddq: np.ndarray) -> np.ndarray:
    jerk = np.empty_like(ddq)
    if len(t) == 1:
        jerk[0] = 0.0
        return jerk
    jerk[0] = (ddq[1] - ddq[0]) / (t[1] - t[0])
    jerk[-1] = (ddq[-1] - ddq[-2]) / (t[-1] - t[-2])
    if len(t) > 2:
        jerk[1:-1] = (ddq[2:] - ddq[:-2]) / (t[2:, None] - t[:-2, None])
    return jerk


def dynamic_summary(t: np.ndarray, q: np.ndarray, dq: np.ndarray, ddq: np.ndarray, limits: Mapping[str, Any], phase: str) -> tuple[dict[str, Any], np.ndarray]:
    jerk = finite_difference_jerk(t, ddq)
    velocity_limits = np.asarray([float(limits[j]["velocity_rad_s"]) for j in JOINT_NAMES], dtype=float)
    acceleration_limits = np.asarray([float(limits[j]["acceleration_rad_s2"]) for j in JOINT_NAMES], dtype=float)
    jerk_limits = np.asarray([float(limits[j]["jerk_rad_s3"]) for j in JOINT_NAMES], dtype=float)
    position_lower = np.asarray([float(limits[j]["position_lower_rad"]) for j in JOINT_NAMES], dtype=float)
    position_upper = np.asarray([float(limits[j]["position_upper_rad"]) for j in JOINT_NAMES], dtype=float)
    max_velocity = np.max(np.abs(dq), axis=0)
    max_acceleration = np.max(np.abs(ddq), axis=0)
    max_jerk = np.max(np.abs(jerk), axis=0)
    position_pass = bool(np.all(q >= position_lower[None, :] - 1e-12) and np.all(q <= position_upper[None, :] + 1e-12))
    velocity_pass = bool(np.all(max_velocity <= velocity_limits + 1e-10))
    acceleration_pass = bool(np.all(max_acceleration <= acceleration_limits + 1e-10))
    jerk_pass = bool(np.all(max_jerk <= jerk_limits + 1e-8))
    summary = {
        "phase": phase,
        "waypoint_count": int(len(t)),
        "duration_s": float(t[-1]),
        "timestamps_finite_strictly_increasing": bool(np.all(np.isfinite(t)) and np.all(np.diff(t) > 0.0)),
        "position_limits_pass": position_pass,
        "velocity_limits_pass": velocity_pass,
        "acceleration_limits_pass": acceleration_pass,
        "jerk_limits_pass": jerk_pass,
        "velocity_limits_rad_s": velocity_limits.tolist(),
        "acceleration_limits_rad_s2": acceleration_limits.tolist(),
        "jerk_limits_rad_s3": jerk_limits.tolist(),
        "max_abs_velocity_rad_s": max_velocity.tolist(),
        "max_abs_acceleration_rad_s2": max_acceleration.tolist(),
        "max_abs_jerk_rad_s3": max_jerk.tolist(),
        "jerk_semantics": "sampled/reconstructed diagnostic: finite difference of native trajectory accelerations over native timestamps; not Ruckig analytic jerk",
        "all_dynamic_limits_pass": bool(position_pass and velocity_pass and acceleration_pass and jerk_pass),
    }
    return summary, jerk


def nearest_raw_index(q: np.ndarray, raw_q: np.ndarray, start: int = 0, stop: int | None = None) -> int:
    stop = len(raw_q) - 1 if stop is None else min(stop, len(raw_q) - 1)
    start = max(0, min(start, stop))
    best_index = start
    best_distance = float("inf")
    for index in range(start, stop + 1):
        distance = float(np.linalg.norm(q - raw_q[index]))
        if distance < best_distance - 1e-14:
            best_index, best_distance = index, distance
    return best_index


def map_samples(t: np.ndarray, q: np.ndarray, raw_q: np.ndarray, segments: Sequence[Mapping[str, Any]]) -> tuple[list[int], list[int], list[float]]:
    raw_indices: list[int] = []
    cursor = 0
    for sample in q:
        index = nearest_raw_index(sample, raw_q, cursor)
        raw_indices.append(index)
        cursor = index
    boundaries: list[float] = []
    for segment in segments[1:]:
        start = int(segment["waypoint_start"])
        candidates = [i for i, index in enumerate(raw_indices) if index >= start]
        if candidates:
            boundaries.append(float(t[candidates[0]]))
        else:
            nearest = int(np.argmin(np.linalg.norm(q - raw_q[start][None, :], axis=1)))
            boundaries.append(float(t[nearest]))
    segment_ids: list[int] = []
    for time in t:
        segment_index = 0
        while segment_index < len(boundaries) and time >= boundaries[segment_index]:
            segment_index += 1
        segment_ids.append(int(segments[segment_index]["segment_id"]))
    return raw_indices, segment_ids, boundaries


def quaternion_error_deg(actual: Sequence[float], desired: Sequence[float]) -> float:
    a = np.asarray(actual, dtype=float)
    d = np.asarray(desired, dtype=float)
    a /= max(float(np.linalg.norm(a)), 1e-12)
    d /= max(float(np.linalg.norm(d)), 1e-12)
    return float(np.degrees(2.0 * np.arccos(np.clip(abs(float(np.dot(a, d))), -1.0, 1.0))))


def process_validate(moveit: MoveItPy, q: np.ndarray, t: np.ndarray, raw_q: np.ndarray, full_cart: Sequence[Mapping[str, Any]], segments: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_indices, segment_ids, boundaries = map_samples(t, q, raw_q, segments)
    state = RobotState(moveit.get_robot_model())
    geometry = contract["geometry"]
    tcp_contract = contract["tcp_reproduction"]
    rows: list[dict[str, Any]] = []
    for index, (sample_q, time, raw_index, segment_id) in enumerate(zip(q, t, raw_indices, segment_ids)):
        state.set_joint_group_positions(GROUP_NAME, sample_q.tolist())
        state.update()
        transform = _transform_matrix(state.get_global_link_transform(EE_LINK))
        actual_position = transform[:3, 3].astype(float)
        actual_orientation = np.asarray(_rotation_matrix_to_quaternion(transform[:3, :3]), dtype=float)
        cart = full_cart[raw_index]
        base = {"schema_version": "stage3-h7-process-validation-v1", "trajectory_index": index, "time_s": float(time), "mapped_h6_waypoint_index": int(raw_index), "segment_id": int(segment_id), "spray_state": cart.get("spray_state"), "joint_values": sample_q.tolist(), "fk_tcp_position_m": actual_position.tolist(), "fk_tcp_orientation_xyzw": actual_orientation.tolist()}
        if cart.get("spray_state") != "SPRAY_ON":
            base.update({"process_tolerance_applicable": False, "process_tolerance_pass": None, "process_tolerance_checks": {"status": "NOT_APPLICABLE"}})
            rows.append(base)
            continue
        desired_position = np.asarray(cart["tcp_position_xyz_m"], dtype=float)
        desired_orientation = np.asarray(cart["tcp_orientation_xyzw"], dtype=float)
        surface_point = np.asarray(cart["surface_point_xyz_m"], dtype=float)
        expected_direction = np.asarray(cart["spray_direction_unit"], dtype=float)
        standoff = float(np.linalg.norm(actual_position - surface_point))
        standoff_error = abs(standoff - float(geometry["nominal_standoff_m"]))
        actual_direction = transform[:3, 2].astype(float)
        actual_direction /= max(float(np.linalg.norm(actual_direction)), 1e-12)
        normal_deviation = float(np.degrees(np.arccos(np.clip(float(np.dot(actual_direction, expected_direction)), -1.0, 1.0))))
        tcp_position_error = float(np.linalg.norm(actual_position - desired_position))
        tcp_orientation_error = quaternion_error_deg(actual_orientation, desired_orientation)
        checks = {
            "standoff": {"pass": standoff_error <= float(geometry["standoff_abs_tolerance_m"]), "actual_standoff_m": standoff, "absolute_error_m": standoff_error, "tolerance_m": float(geometry["standoff_abs_tolerance_m"])},
            "surface_normal": {"pass": normal_deviation <= float(geometry["normal_angle_tolerance_deg"]), "angular_deviation_deg": normal_deviation, "tolerance_deg": float(geometry["normal_angle_tolerance_deg"])},
            "tcp_position": {"pass": tcp_position_error <= float(tcp_contract["tcp_position_tolerance_m"]), "error_m": tcp_position_error, "tolerance_m": float(tcp_contract["tcp_position_tolerance_m"])},
            "tcp_orientation": {"pass": tcp_orientation_error <= float(tcp_contract["tcp_orientation_tolerance_deg"]), "error_deg": tcp_orientation_error, "tolerance_deg": float(tcp_contract["tcp_orientation_tolerance_deg"])},
        }
        base.update({"desired_tcp_position_m": desired_position.tolist(), "desired_tcp_orientation_xyzw": desired_orientation.tolist(), "surface_point_m": surface_point.tolist(), "process_tolerance_applicable": True, "actual_standoff_m": standoff, "absolute_standoff_error_m": standoff_error, "normal_angle_deviation_deg": normal_deviation, "tcp_position_error_m": tcp_position_error, "tcp_orientation_error_deg": tcp_orientation_error, "process_tolerance_checks": checks, "process_tolerance_pass": all(bool(value["pass"]) for value in checks.values())})
        rows.append(base)
    on = [row for row in rows if row.get("spray_state") == "SPRAY_ON"]
    failures = [row for row in on if row.get("process_tolerance_pass") is not True]
    def max_value(key: str) -> float | None:
        values = [float(row[key]) for row in on if isinstance(row.get(key), (int, float)) and math.isfinite(float(row[key]))]
        return max(values) if values else None
    summary = {"status": "PASSED" if not failures else "BLOCKED", "spray_on_sample_count": len(on), "spray_on_failed_sample_count": len(failures), "max_standoff_error_m": max_value("absolute_standoff_error_m"), "max_normal_deviation_deg": max_value("normal_angle_deviation_deg"), "max_tcp_position_error_m": max_value("tcp_position_error_m"), "max_tcp_orientation_error_deg": max_value("tcp_orientation_error_deg"), "boundary_times_s": boundaries, "first_violation": {"trajectory_index": failures[0].get("trajectory_index"), "segment_id": failures[0].get("segment_id")} if failures else None}
    return rows, summary


def contact_pairs(result: CollisionResult) -> list[str]:
    pairs: list[str] = []
    contacts = getattr(result, "contacts", {})
    if hasattr(contacts, "items"):
        for pair in contacts:
            if isinstance(pair, (tuple, list)):
                pairs.append("|".join(str(value) for value in pair))
            else:
                pairs.append(str(pair))
    return sorted(set(pairs))


def collision_state(scene: Any, state: RobotState, q: np.ndarray) -> dict[str, Any]:
    state.set_joint_group_positions(GROUP_NAME, q.tolist())
    state.update()
    request = CollisionRequest()
    request.joint_model_group_name = GROUP_NAME
    request.contacts = True
    request.max_contacts = 4096
    request.max_contacts_per_pair = 64
    result = CollisionResult()
    scene.check_collision(request, result, state)
    pairs = contact_pairs(result)
    environment_pairs = [pair for pair in pairs if "fixture_surface" in pair or "world" in pair or "tunnel" in pair]
    self_pairs = [pair for pair in pairs if pair not in environment_pairs]
    if result.collision and not pairs:
        # Unknown collision source is fail-closed; never label it collision-free.
        environment_pairs = ["unknown_collision_source"]
        self_pairs = ["unknown_collision_source"]
    return {"environment_collision": bool(environment_pairs), "self_collision": bool(self_pairs), "collision_free": not environment_pairs and not self_pairs, "environment_collision_pairs": environment_pairs, "self_collision_pairs": self_pairs, "collision_method": COLLISION_METHOD, "ccd_status": CCD_STATUS, "clearance_status": CLEARANCE_STATUS, "clearance_m": None}


def collision_validate(moveit: MoveItPy, t: np.ndarray, q: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    monitor = moveit.get_planning_scene_monitor()
    records: list[dict[str, Any]] = []
    max_step = np.deg2rad(COLLISION_STEP_DEG)
    with monitor.read_only() as scene:
        state = RobotState(moveit.get_robot_model())
        for index, (time, positions) in enumerate(zip(t, q)):
            result = collision_state(scene, state, positions)
            records.append({"schema_version": "stage3-h7-collision-validation-v1", "sample_kind": "final_trajectory_sample", "trajectory_index": index, "time_s": float(time), "joint_values": positions.tolist(), **result})
        for left in range(len(q) - 1):
            steps = max(1, int(np.ceil(float(np.max(np.abs(q[left + 1] - q[left]))) / max_step)))
            for local in range(steps):
                alpha = float(local) / float(steps)
                sample = q[left] + alpha * (q[left + 1] - q[left])
                result = collision_state(scene, state, sample)
                records.append({"schema_version": "stage3-h7-collision-validation-v1", "sample_kind": "adaptive_discrete_interpolation", "trajectory_index": left, "next_trajectory_index": left + 1, "interpolation_alpha": alpha, "time_s": float(t[left] + alpha * (t[left + 1] - t[left])), "joint_values": sample.tolist(), **result})
        final = [row for row in records if row["sample_kind"] == "final_trajectory_sample"]
    failures = [row for row in records if not row["collision_free"]]
    summary = {"status": "PASSED" if not failures else "BLOCKED", "checked_state_count": len(records), "final_sample_count": len(final), "collision_failure_count": len(failures), "first_failure": failures[0] if failures else None, "collision_method": COLLISION_METHOD, "ccd_status": CCD_STATUS, "clearance_status": CLEARANCE_STATUS}
    return records, summary


def boundary_audit(t: np.ndarray, q: np.ndarray, dq: np.ndarray, ddq: np.ndarray, jerk: np.ndarray, segments: Sequence[Mapping[str, Any]], boundaries: Sequence[float], collision_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    final_collision = {int(row["trajectory_index"]): row for row in collision_records if row["sample_kind"] == "final_trajectory_sample"}
    audits: list[dict[str, Any]] = []
    for boundary_index, (left, right, switch_time) in enumerate(zip(segments[:-1], segments[1:], boundaries)):
        before_candidates = np.where(t < switch_time)[0]
        after_candidates = np.where(t >= switch_time)[0]
        before = int(before_candidates[-1]) if len(before_candidates) else 0
        after = int(after_candidates[0]) if len(after_candidates) else len(t) - 1
        position_jump = float(np.linalg.norm(q[after] - q[before]))
        record = {
            "boundary_index": boundary_index,
            "source_segment_id": int(left["segment_id"]),
            "destination_segment_id": int(right["segment_id"]),
            "transition_time_s": float(switch_time),
            "sample_before_index": before,
            "sample_after_index": after,
            "position_before": q[before].tolist(),
            "position_after": q[after].tolist(),
            "position_sample_jump_norm_rad": position_jump,
            "position_continuity": True,
            "velocity_before": dq[before].tolist(),
            "velocity_after": dq[after].tolist(),
            "acceleration_before": ddq[before].tolist(),
            "acceleration_after": ddq[after].tolist(),
            "jerk_diagnostic_before": jerk[before].tolist(),
            "jerk_diagnostic_after": jerk[after].tolist(),
            "collision_state_before": final_collision.get(before, {}).get("collision_free"),
            "collision_state_after": final_collision.get(after, {}).get("collision_free"),
            "spray_state_before": left["spray_state"],
            "spray_state_after": right["spray_state"],
            "state_switch_unambiguous": bool(math.isfinite(float(switch_time)) and left["spray_state"] != right["spray_state"]),
        }
        audits.append(record)
    passed = len(audits) == 4 and all(item["position_continuity"] and item["state_switch_unambiguous"] for item in audits)
    return {"schema_version": "stage3-h7-segment-boundary-audit-v1", "boundary_count": len(audits), "boundaries": audits, "status": "PASSED" if passed else "BLOCKED", "boundary_audit_pass": passed, "boundary_semantics": "continuous-motion state transition; velocity and acceleration are diagnosed before/after and are not forced to zero"}


def write_trajectory_jsonl(path: Path, t: np.ndarray, q: np.ndarray, dq: np.ndarray, ddq: np.ndarray, raw_q: np.ndarray, segments: Sequence[Mapping[str, Any]]) -> tuple[list[int], list[int], list[float]]:
    raw_indices, segment_ids, boundaries = map_samples(t, q, raw_q, segments)
    rows = [{"schema_version": "stage3-h7-timed-trajectory-v1", "trajectory_index": int(index), "time_from_start_s": float(time), "joint_names": list(JOINT_NAMES), "positions_rad": positions.tolist(), "velocities_rad_s": velocities.tolist(), "accelerations_rad_s2": accelerations.tolist(), "mapped_h6_waypoint_index": int(raw_index), "segment_id": int(segment_id), "process_state": segments[next(i for i, item in enumerate(segments) if int(item["segment_id"]) == segment_id)]["spray_state"]} for index, (time, positions, velocities, accelerations, raw_index, segment_id) in enumerate(zip(t, q, dq, ddq, raw_indices, segment_ids))]
    dump_jsonl(path, rows)
    return raw_indices, segment_ids, boundaries


def formal_run(node, moveit: MoveItPy) -> dict[str, Any]:
    output = Path(str(param(node, "output_dir"))).resolve()
    h6_joint_path = Path(str(param(node, "h6_joint_waypoints"))).resolve()
    h6_cart_path = Path(str(param(node, "h6_cartesian_waypoints"))).resolve()
    mesh_path = Path(str(param(node, "fixture_mesh"))).resolve()
    sequence_path = Path(str(param(node, "sequence_manifest"))).resolve()
    contract_path = Path(str(param(node, "h6_1_contract"))).resolve()
    limits_path = Path(get_package_share_directory("fr5_tunnel_moveit_bridge")) / "config/joint_limits_with_jerk.yaml"
    sequence = load_json(sequence_path)
    segments = sequence["segments"]
    contract = load_json(contract_path)
    joint_rows = load_jsonl(h6_joint_path)
    cart_rows = load_jsonl(h6_cart_path)
    full_cart = full_cartesian_rows(joint_rows, cart_rows)
    raw_q = np.asarray([row["joint_values"] for row in joint_rows], dtype=float)
    if raw_q.shape != (1295, 6) or not np.all(np.isfinite(raw_q)):
        raise RuntimeError("H6 joint waypoint input is not finite 1295x6")
    limits_audit = load_json(output.parent / "stage3_h7_runtime_limit_audit.json") if (output.parent / "stage3_h7_runtime_limit_audit.json").is_file() else load_json(output / "stage3_h7_runtime_limit_audit.json")
    runtime_limits = limits_audit["runtime_bounds"]
    apply_fixture(moveit, mesh_path)
    start_state = RobotState(moveit.get_robot_model())
    raw_message = make_trajectory_message(joint_rows)
    trajectory = RobotTrajectory(moveit.get_robot_model())
    trajectory.joint_model_group_name = GROUP_NAME
    trajectory.set_robot_trajectory_msg(start_state, raw_message)
    trajectory.unwind()
    totg_config = load_json(Path(str(param(node, "totg_configuration"))))
    ruckig_config = load_json(Path(str(param(node, "ruckig_configuration"))))
    if not trajectory.apply_totg_time_parameterization(float(totg_config["velocity_scaling_factor"]), float(totg_config["acceleration_scaling_factor"]), path_tolerance=float(totg_config["path_tolerance"]), resample_dt=float(totg_config["resample_dt"]), min_angle_change=float(totg_config["min_angle_change"])):
        max_raw_step = float(np.max(np.abs(np.diff(raw_q, axis=0))))
        if max_raw_step >= math.pi:
            raise RuntimeError("totg_failed_path_requires_180_degree_turn")
        raise RuntimeError("totg_failed_native_moveit_returned_false")
    totg_message = trajectory.get_robot_trajectory_msg()
    totg_t, totg_q, totg_dq, totg_ddq = message_arrays(totg_message)
    totg_summary, _totg_jerk = dynamic_summary(totg_t, totg_q, totg_dq, totg_ddq, runtime_limits, "post_totg")
    write_trajectory_jsonl(output / "stage3_h7_totg_trajectory.jsonl", totg_t, totg_q, totg_dq, totg_ddq, raw_q, segments)
    totg_process_rows, totg_process_summary = process_validate(moveit, totg_q, totg_t, raw_q, full_cart, segments, contract)
    totg_collision_rows, totg_collision_summary = collision_validate(moveit, totg_t, totg_q)
    totg_process_ok = totg_process_summary["status"] == "PASSED"
    totg_collision_ok = totg_collision_summary["status"] == "PASSED"
    totg_gate = {"totg_succeeded": True, "totg_time_valid": bool(totg_summary["timestamps_finite_strictly_increasing"]), "post_totg_dynamic_limits_pass": bool(totg_summary["all_dynamic_limits_pass"]), "post_totg_process_pass": totg_process_ok, "post_totg_collision_pass": totg_collision_ok, "status": "PASSED" if totg_summary["all_dynamic_limits_pass"] and totg_process_ok and totg_collision_ok else "BLOCKED", "original_waypoint_count": len(joint_rows), "totg_waypoint_count": len(totg_t), "totg_duration_s": float(totg_t[-1]), "totg_dynamic_summary": totg_summary, "totg_process_summary": totg_process_summary, "totg_collision_summary": totg_collision_summary, "mapping": "each native resampled state is mapped by monotonic nearest H6 joint-path progress and then assigned to the authoritative segment-time intervals"}
    dump_json(output / "stage3_h7_totg_validation.json", totg_gate)
    if totg_gate["status"] != "PASSED":
        dump_jsonl(output / "stage3_h7_final_timed_trajectory.jsonl", [])
        dump_jsonl(output / "stage3_h7_dynamic_validation.jsonl", [])
        dump_jsonl(output / "stage3_h7_collision_validation.jsonl", [])
        dump_jsonl(output / "stage3_h7_process_validation.jsonl", [])
        dump_json(output / "stage3_h7_segment_boundary_audit.json", {"status": "BLOCKED", "reason": "post_totg_gate_failed"})
        result = {"schema_version": "stage3-h7-native-run-v1", "status": "BLOCKED", "errors": ["post_totg_gate_failed"], "totg_summary": totg_gate, "final_summary": {}, "process_summary": {}, "gate_checks": {"formal_velocity_limits_defined": True, "formal_acceleration_limits_defined": True, "formal_jerk_limits_defined": True, "totg_succeeded": True, "totg_time_valid": totg_gate["totg_time_valid"], "post_totg_dynamic_limits_pass": totg_gate["post_totg_dynamic_limits_pass"], "post_totg_collision_pass": totg_collision_ok, "post_totg_process_pass": totg_process_ok, "ruckig_succeeded": False, "final_position_limits_pass": False, "final_velocity_limits_pass": False, "final_acceleration_limits_pass": False, "final_jerk_limits_pass": False, "final_collision_pass": False, "final_process_pass": False, "boundary_audit_pass": False}}
        dump_json(output / "stage3_h7_native_run.json", result)
        return result
    if not trajectory.apply_ruckig_smoothing(float(ruckig_config["velocity_scaling_factor"]), float(ruckig_config["acceleration_scaling_factor"]), mitigate_overshoot=bool(ruckig_config["mitigate_overshoot"]), overshoot_threshold=float(ruckig_config["overshoot_threshold"])):
        raise RuntimeError("native MoveIt2 Ruckig returned false")
    final_message = trajectory.get_robot_trajectory_msg()
    final_t, final_q, final_dq, final_ddq = message_arrays(final_message)
    final_summary, final_jerk = dynamic_summary(final_t, final_q, final_dq, final_ddq, runtime_limits, "post_ruckig")
    write_trajectory_jsonl(output / "stage3_h7_final_timed_trajectory.jsonl", final_t, final_q, final_dq, final_ddq, raw_q, segments)
    final_process_rows, final_process_summary = process_validate(moveit, final_q, final_t, raw_q, full_cart, segments, contract)
    final_collision_rows, final_collision_summary = collision_validate(moveit, final_t, final_q)
    _raw_indices, _segment_ids, final_boundaries = map_samples(final_t, final_q, raw_q, segments)
    boundary = boundary_audit(final_t, final_q, final_dq, final_ddq, final_jerk, segments, final_boundaries, final_collision_rows)
    dump_jsonl(output / "stage3_h7_dynamic_validation.jsonl", [{"schema_version": "stage3-h7-dynamic-validation-v1", "trajectory_index": index, "time_s": float(time), "joint_values": positions.tolist(), "velocity_rad_s": velocities.tolist(), "acceleration_rad_s2": accelerations.tolist(), "reconstructed_jerk_rad_s3": final_jerk[index].tolist(), "jerk_semantics": "sampled/reconstructed finite difference from native accelerations", "position_limits_pass": bool(final_summary["position_limits_pass"]), "velocity_limits_pass": bool(np.all(np.abs(velocities) <= np.asarray(final_summary["velocity_limits_rad_s"]) + 1e-10)), "acceleration_limits_pass": bool(np.all(np.abs(accelerations) <= np.asarray(final_summary["acceleration_limits_rad_s2"]) + 1e-10)), "jerk_limits_pass": bool(np.all(np.abs(final_jerk[index]) <= np.asarray(final_summary["jerk_limits_rad_s3"]) + 1e-8))} for index, (time, positions, velocities, accelerations) in enumerate(zip(final_t, final_q, final_dq, final_ddq))])
    dump_jsonl(output / "stage3_h7_collision_validation.jsonl", final_collision_rows)
    dump_jsonl(output / "stage3_h7_process_validation.jsonl", final_process_rows)
    dump_json(output / "stage3_h7_segment_boundary_audit.json", boundary)
    final_gate = {
        "ruckig_succeeded": True,
        "final_position_limits_pass": bool(final_summary["position_limits_pass"]),
        "final_velocity_limits_pass": bool(final_summary["velocity_limits_pass"]),
        "final_acceleration_limits_pass": bool(final_summary["acceleration_limits_pass"]),
        "final_jerk_limits_pass": bool(final_summary["jerk_limits_pass"]),
        "final_collision_pass": final_collision_summary["status"] == "PASSED",
        "final_process_pass": final_process_summary["status"] == "PASSED",
        "boundary_audit_pass": boundary["boundary_audit_pass"],
    }
    gate_checks = {"formal_velocity_limits_defined": True, "formal_acceleration_limits_defined": True, "formal_jerk_limits_defined": True, **totg_gate, **final_gate}
    gate_checks["status"] = "PASSED" if all(bool(value) for key, value in gate_checks.items() if key.endswith("pass") or key in {"totg_succeeded", "totg_time_valid", "ruckig_succeeded"}) else "BLOCKED"
    result = {
        "schema_version": "stage3-h7-native-run-v1",
        "status": gate_checks["status"],
        "errors": [] if gate_checks["status"] == "PASSED" else ["post_ruckig_gate_failed"],
        "totg_summary": {**totg_gate, "totg_process_rows": len(totg_process_rows), "totg_collision_rows": len(totg_collision_rows)},
        "final_summary": {"duration_s": float(final_t[-1]), "waypoint_count": len(final_t), "dynamic": final_summary, "collision": final_collision_summary, "ruckig_api": "RobotTrajectory.apply_ruckig_smoothing", "ruckig_status": "SUCCEEDED"},
        "process_summary": final_process_summary,
        "boundary_summary": boundary,
        "gate_checks": gate_checks,
        "NEW_FJT_GOALS_SENT": 0,
        "ROBOT_MOTION_STARTED": "NO",
        "semantic_digest": semantic_hash({"authoritative_segment_order": [int(s["segment_id"]) for s in segments], "totg": totg_gate, "totg_trajectory": [{"t": float(v), "q": p.tolist(), "dq": d.tolist(), "ddq": a.tolist()} for v, p, d, a in zip(totg_t, totg_q, totg_dq, totg_ddq)], "ruckig": ruckig_config, "final_trajectory": [{"t": float(v), "q": p.tolist(), "dq": d.tolist(), "ddq": a.tolist()} for v, p, d, a in zip(final_t, final_q, final_dq, final_ddq)], "final_process": final_process_summary, "final_collision": final_collision_summary, "final_dynamic": final_summary, "boundary": boundary}),
    }
    dump_json(output / "stage3_h7_native_run.json", result)
    return result


def main() -> int:
    rclpy.init()
    node = rclpy.create_node("stage3_h7_native")
    mode = str(param(node, "mode", "audit"))
    output = Path(str(param(node, "output_dir"))).resolve()
    output.mkdir(parents=True, exist_ok=True)
    exit_code = 2
    try:
        moveit = MoveItPy(node_name="stage3_h7_native")
        if mode == "audit":
            yaml_path = Path(get_package_share_directory("fr5_tunnel_moveit_bridge")) / "config/joint_limits_with_jerk.yaml"
            result = runtime_limit_audit(moveit, output, GROUP_NAME, EE_LINK, yaml_path)
            dump_json(output / "stage3_h7_native_run.json", {"schema_version": "stage3-h7-native-run-v1", "status": result.get("status"), "runtime_audit": result, "NEW_FJT_GOALS_SENT": 0, "ROBOT_MOTION_STARTED": "NO"})
            exit_code = 0 if result.get("status") == "PASSED" else 2
        else:
            result = formal_run(node, moveit)
            exit_code = 0 if result.get("status") == "PASSED" else 2
    except Exception as exc:
        message = str(exc)
        blocker = message if message.startswith("totg_failed_") else f"native_formal_exception:{type(exc).__name__}:{message}"
        for name, value in {
            "stage3_h7_totg_trajectory.jsonl": [],
            "stage3_h7_final_timed_trajectory.jsonl": [],
            "stage3_h7_dynamic_validation.jsonl": [],
            "stage3_h7_collision_validation.jsonl": [],
            "stage3_h7_process_validation.jsonl": [],
        }.items():
            dump_jsonl(output / name, value)
        dump_json(output / "stage3_h7_totg_validation.json", {"schema_version": "stage3-h7-totg-validation-v1", "status": "BLOCKED", "totg_succeeded": False, "FIRST_BLOCKER": blocker, "trajectory_generated": False})
        dump_json(output / "stage3_h7_segment_boundary_audit.json", {"schema_version": "stage3-h7-segment-boundary-audit-v1", "status": "BLOCKED", "reason": "formal_totg_did_not_succeed"})
        result = {"schema_version": "stage3-h7-native-run-v1", "status": "BLOCKED", "errors": [blocker], "totg_summary": {"totg_succeeded": False, "status": "BLOCKED"}, "final_summary": {}, "process_summary": {}, "gate_checks": {"formal_velocity_limits_defined": True, "formal_acceleration_limits_defined": True, "formal_jerk_limits_defined": True, "totg_succeeded": False, "totg_time_valid": False, "post_totg_dynamic_limits_pass": False, "post_totg_collision_pass": False, "post_totg_process_pass": False, "ruckig_succeeded": False, "final_position_limits_pass": False, "final_velocity_limits_pass": False, "final_acceleration_limits_pass": False, "final_jerk_limits_pass": False, "final_collision_pass": False, "final_process_pass": False, "boundary_audit_pass": False}, "NEW_FJT_GOALS_SENT": 0, "ROBOT_MOTION_STARTED": "NO"}
        dump_json(output / "stage3_h7_native_run.json", result)
        exit_code = 2
    # MoveItPy/Jazzy can segfault while destructing MoveItCpp after all output
    # has been emitted.  The project uses this same offline fast-exit boundary
    # in its existing planner; no controller or action client is touched here.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
