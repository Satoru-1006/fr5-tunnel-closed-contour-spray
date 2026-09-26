"""Stage 5B shadow experiment for the waypoint 66->67 IK branch cut.

The experiment is deliberately separate from the Stage5AC/D65 artifacts.  It
densifies only the Cartesian pose segment around the known cut and follows the
same MoveIt KDL position-IK implementation from the preceding accepted state.
It records both the raw joint coordinates and a limit-aware adjacent-step
metric; it never edits an authoritative trajectory or changes the world.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
JOINTS = [f"j{i}" for i in range(1, 7)]
GROUP = "fairino5_v6_group"
EE = "spray_tcp_link"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def matrix_of(value: Any) -> np.ndarray:
    raw = value.matrix() if hasattr(value, "matrix") else value
    return np.asarray(raw, dtype=float)


def env_objects(environment: dict[str, Any]):
    from geometry_msgs.msg import Pose
    from moveit_msgs.msg import CollisionObject
    from shape_msgs.msg import SolidPrimitive

    objects = []
    for item in environment["objects"]:
        obj = CollisionObject()
        obj.header.frame_id = str(environment["frame_id"])
        obj.id = str(item["id"])
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = [float(x) for x in item["dimensions_m"]]
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = [float(x) for x in item["center_m"]]
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = [float(x) for x in item["quaternion_xyzw"]]
        obj.primitives.append(primitive)
        obj.primitive_poses.append(pose)
        obj.operation = CollisionObject.ADD
        objects.append(obj)
    return objects


def pose_from_arrays(position: np.ndarray, quaternion_xyzw: np.ndarray):
    from geometry_msgs.msg import Pose

    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = [float(x) for x in position]
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = [float(x) for x in quaternion_xyzw]
    return pose


def pose_error(actual: np.ndarray, position: np.ndarray, quaternion_xyzw: np.ndarray) -> tuple[float, float]:
    from scipy.spatial.transform import Rotation

    observed = Rotation.from_matrix(actual[:3, :3]).as_quat()
    observed /= max(float(np.linalg.norm(observed)), 1.0e-15)
    expected = np.asarray(quaternion_xyzw, dtype=float)
    expected /= max(float(np.linalg.norm(expected)), 1.0e-15)
    angle = 2.0 * math.acos(float(np.clip(abs(np.dot(observed, expected)), -1.0, 1.0)))
    return float(np.linalg.norm(actual[:3, 3] - position)), float(angle)


def shortest_delta(target: np.ndarray, source: np.ndarray) -> np.ndarray:
    return (np.asarray(target) - np.asarray(source) + math.pi) % (2.0 * math.pi) - math.pi


def interpolate_pose(a: dict[str, str], b: dict[str, str], fraction: float) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial.transform import Rotation, Slerp

    pa = np.asarray([float(a[k]) for k in ("x", "y", "z")], dtype=float)
    pb = np.asarray([float(b[k]) for k in ("x", "y", "z")], dtype=float)
    qa = np.asarray([float(a[k]) for k in ("qx", "qy", "qz", "qw")], dtype=float)
    qb = np.asarray([float(b[k]) for k in ("qx", "qy", "qz", "qw")], dtype=float)
    rotations = Rotation.from_quat(np.stack([qa, qb], axis=0))
    quat = Slerp([0.0, 1.0], rotations)([float(fraction)]).as_quat()[0]
    return (1.0 - fraction) * pa + fraction * pb, quat


def run(node: Any) -> int:  # pragma: no cover - ROS runtime
    import rclpy
    from moveit.core.collision_detection import CollisionRequest, CollisionResult
    from moveit.core.robot_state import RobotState
    from moveit.planning import MoveItPy

    poses_path = Path(str(node.get_parameter("poses_csv").value)).resolve()
    source_path = Path(str(node.get_parameter("source_q_csv").value)).resolve()
    environment_path = Path(str(node.get_parameter("environment_json").value)).resolve()
    output = Path(str(node.get_parameter("output_dir").value)).resolve()
    start = int(node.get_parameter("start_waypoint").value)
    end = int(node.get_parameter("end_waypoint").value)
    subdivisions = int(node.get_parameter("subdivisions").value)
    if subdivisions < 1 or not (0 <= start < end):
        raise RuntimeError("stage5b_invalid_segment")

    poses = read_csv(poses_path)
    source_rows = read_csv(source_path)
    if len(poses) != len(source_rows) or len(poses) < end + 1:
        raise RuntimeError(f"stage5b_input_count:{len(poses)}/{len(source_rows)}")
    q_source = np.asarray([[float(row[f"j{i}_q"]) for i in range(1, 7)] for row in source_rows], dtype=float)
    environment = json.loads(environment_path.read_text(encoding="utf-8"))

    moveit = MoveItPy(node_name="stage5b_branch_repair_moveit")
    model = moveit.get_robot_model()
    group = model.get_joint_model_group(GROUP)
    if group is None or list(group.active_joint_model_names) != JOINTS:
        raise RuntimeError("stage5b_joint_mapping_mismatch")
    psm = moveit.get_planning_scene_monitor()
    with psm.read_write() as scene:
        scene.remove_all_collision_objects()
        for obj in env_objects(environment):
            scene.apply_collision_object(obj)
    request = CollisionRequest()
    request.joint_model_group_name = GROUP
    request.contacts = False

    # Include the actual endpoints and uniformly densify only this Cartesian
    # edge.  This is a diagnostic sequence, not a silent modification of the
    # 181-point input contract.
    target_specs: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    for k in range(subdivisions + 1):
        fraction = float(k) / float(subdivisions)
        position, quaternion = interpolate_pose(poses[start], poses[end], fraction)
        if k == 0:
            seed = q_source[start].copy()
        elif k == subdivisions:
            seed = q_source[end].copy()
        else:
            seed = q_source[start].copy()
        target_specs.append((f"{start}+{k}/{subdivisions}", position, quaternion, seed))

    state = RobotState(model)
    records: list[dict[str, Any]] = []
    previous: np.ndarray | None = None
    for index, (label, position, quaternion, initial_seed) in enumerate(target_specs):
        seed = previous.copy() if previous is not None else initial_seed.copy()
        state.set_joint_group_positions(GROUP, seed.tolist())
        state.update()
        solved = bool(state.set_from_ik(GROUP, pose_from_arrays(position, quaternion), EE, 0.0))
        record: dict[str, Any] = {
            "sample_index": index,
            "label": label,
            "fraction": float(index) / float(subdivisions),
            "solver_success": solved,
            "seed": seed.tolist(),
        }
        if solved:
            state.update()
            q = np.asarray(state.get_joint_group_positions(GROUP), dtype=float).copy()
            fk = matrix_of(state.get_global_link_transform(EE))
            position_error, orientation_error = pose_error(fk, position, quaternion)
            self_result = CollisionResult()
            with psm.read_only() as scene:
                scene.check_self_collision(request, self_result, state)
                full_result = CollisionResult()
                scene.check_collision(request, full_result, state)
            limits_ok = bool(np.all(np.asarray([b[0] if isinstance(b, (list, tuple)) else b for b in group.active_joint_model_bounds], dtype=object)))
            # Check limits directly; the object-array expression above is only
            # avoided because MoveIt Python exposes bounds in two variants.
            lows, highs = [], []
            for bound in group.active_joint_model_bounds:
                b = bound[0] if isinstance(bound, (list, tuple)) else bound
                lows.append(float(b.min_position))
                highs.append(float(b.max_position))
            limits_ok = bool(np.all(q >= np.asarray(lows)) and np.all(q <= np.asarray(highs)))
            raw_delta = None if previous is None else (q - previous)
            wrapped_delta = None if previous is None else shortest_delta(q, previous)
            record.update({
                "q": q.tolist(),
                "fk_position_error_m": position_error,
                "fk_orientation_error_rad": orientation_error,
                "self_collision": bool(self_result.collision),
                "world_collision": bool(full_result.collision and not self_result.collision),
                "joint_limits": limits_ok,
                "raw_delta_rad": None if raw_delta is None else raw_delta.tolist(),
                "wrapped_delta_rad": None if wrapped_delta is None else wrapped_delta.tolist(),
                "raw_max_step_deg": None if raw_delta is None else float(np.rad2deg(np.max(np.abs(raw_delta)))),
                "wrapped_max_step_deg": None if wrapped_delta is None else float(np.rad2deg(np.max(np.abs(wrapped_delta)))),
            })
            previous = q
        records.append(record)

    success = all(item.get("solver_success") and item.get("joint_limits") and not item.get("self_collision") and not item.get("world_collision") and item.get("fk_position_error_m", math.inf) <= 1.0e-4 and item.get("fk_orientation_error_rad", math.inf) <= 1.0e-3 for item in records)
    valid_steps = [item for item in records if item.get("wrapped_max_step_deg") is not None]
    raw_steps = [item for item in records if item.get("raw_max_step_deg") is not None]
    summary = {
        "schema_version": "stage5b-branch-densification-v1",
        "status": "PASS" if success else "UNRESOLVED",
        "source_poses": str(poses_path),
        "source_q_csv": str(source_path),
        "environment": str(environment_path),
        "segment": {"start_waypoint": start, "end_waypoint": end, "subdivisions": subdivisions, "sample_count": len(records)},
        "cartesian_interpolation": "linear_position_slerp_orientation",
        "solver": "MoveItPy KinematicsBase::getPositionIK via RobotState.set_from_ik",
        "records": records,
        "max_raw_step_deg": max((item["raw_max_step_deg"] for item in raw_steps), default=None),
        "max_wrapped_step_deg": max((item["wrapped_max_step_deg"] for item in valid_steps), default=None),
        "endpoint_source_q_start": q_source[start].tolist(),
        "endpoint_source_q_end": q_source[end].tolist(),
        "endpoint_followed_q_start": records[0].get("q"),
        "endpoint_followed_q_end": records[-1].get("q"),
        "interpretation": "diagnostic_only; no authoritative trajectory mutation; strict articulated CCD unavailable",
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "STAGE5B_BRANCH_DENSIFICATION.json", summary)
    with (output / "STAGE5B_BRANCH_DENSIFICATION.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["sample_index", "label", "fraction", "solver_success", "fk_position_error_m", "fk_orientation_error_rad", "self_collision", "world_collision", "joint_limits", "raw_max_step_deg", "wrapped_max_step_deg"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in records:
            writer.writerow({field: row.get(field) for field in fields})
    print(json.dumps({"status": summary["status"], "samples": len(records), "max_raw_step_deg": summary["max_raw_step_deg"], "max_wrapped_step_deg": summary["max_wrapped_step_deg"]}, ensure_ascii=False), flush=True)
    return 0 if success else 2


def main() -> int:  # pragma: no cover - ROS runtime
    import rclpy
    from rclpy.node import Node

    rclpy.init()
    node = Node("stage5b_branch_repair_wrapper")
    node.declare_parameter("poses_csv", "")
    node.declare_parameter("source_q_csv", "")
    node.declare_parameter("environment_json", "")
    node.declare_parameter("output_dir", "")
    node.declare_parameter("start_waypoint", 66)
    node.declare_parameter("end_waypoint", 67)
    node.declare_parameter("subdivisions", 32)
    try:
        return run(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
