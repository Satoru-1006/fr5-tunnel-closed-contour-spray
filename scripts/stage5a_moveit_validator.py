"""Independent Stage 5A MoveItPy FK, PlanningScene, and self-collision audit."""

from __future__ import annotations

import bisect
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from geometry_msgs.msg import Pose
from moveit.core.collision_detection import CollisionRequest, CollisionResult
from moveit.core.robot_state import RobotState
from moveit.planning import MoveItPy
from moveit_msgs.msg import CollisionObject
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive


JOINTS = [f"j{i}" for i in range(1, 7)]
GROUP = "fairino5_v6_group"
TCP = "spray_tcp_link"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    times = np.asarray([float(row["t"]) for row in rows], dtype=float)
    q = np.asarray([[float(row[f"j{i}_q"]) for i in range(1, 7)] for row in rows], dtype=float)
    if len(rows) < 2 or q.shape != (len(rows), 6) or not np.isfinite(times).all() or not np.isfinite(q).all() or np.any(np.diff(times) <= 0.0):
        raise RuntimeError("stage5a_invalid_trajectory")
    return times, q


def quaternion_from_matrix(matrix: np.ndarray) -> list[float]:
    trace = float(np.trace(matrix[:3, :3]))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w, x, y, z = 0.25 * s, (matrix[2, 1] - matrix[1, 2]) / s, (matrix[0, 2] - matrix[2, 0]) / s, (matrix[1, 0] - matrix[0, 1]) / s
    else:
        diagonal = np.diag(matrix[:3, :3])
        index = int(np.argmax(diagonal))
        if index == 0:
            s = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            w, x, y, z = (matrix[2, 1] - matrix[1, 2]) / s, 0.25 * s, (matrix[0, 1] + matrix[1, 0]) / s, (matrix[0, 2] + matrix[2, 0]) / s
        elif index == 1:
            s = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            w, x, y, z = (matrix[0, 2] - matrix[2, 0]) / s, (matrix[0, 1] + matrix[1, 0]) / s, 0.25 * s, (matrix[1, 2] + matrix[2, 1]) / s
        else:
            s = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            w, x, y, z = (matrix[1, 0] - matrix[0, 1]) / s, (matrix[0, 2] + matrix[2, 0]) / s, (matrix[1, 2] + matrix[2, 1]) / s, 0.25 * s
    return [float(x), float(y), float(z), float(w)]


def matrix_of(value: Any) -> np.ndarray:
    raw = value.matrix() if hasattr(value, "matrix") else value
    return np.asarray(raw, dtype=float)


def matrix_from_tf(row: dict[str, Any]) -> np.ndarray:
    q = np.asarray([float(row["rotation"][axis]) for axis in ("x", "y", "z", "w")], dtype=float)
    q /= max(float(np.linalg.norm(q)), 1e-12)
    x, y, z, w = q
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    matrix[:3, 3] = [float(row["translation"][axis]) for axis in ("x", "y", "z")]
    return matrix


def make_objects(environment: dict[str, Any]) -> list[CollisionObject]:
    objects: list[CollisionObject] = []
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


class Validator(Node):
    def __init__(self) -> None:
        super().__init__("stage5a_moveit_validator_node")
        self.declare_parameter("trajectory_csv", "")
        self.declare_parameter("runtime_dir", "")
        self.declare_parameter("environment_json", "")
        self.declare_parameter("output_dir", "")
        self.declare_parameter("collision_detector", "")
        self.declare_parameter("interpolation_step_deg", 0.0)

    def run(self) -> dict[str, Any]:
        trajectory = Path(str(self.get_parameter("trajectory_csv").value)).resolve()
        runtime_dir = Path(str(self.get_parameter("runtime_dir").value)).resolve()
        environment_path = Path(str(self.get_parameter("environment_json").value)).resolve()
        output = Path(str(self.get_parameter("output_dir").value)).resolve()
        requested_detector = str(self.get_parameter("collision_detector").value).strip()
        interpolation_step_deg = float(self.get_parameter("interpolation_step_deg").value)
        output.mkdir(parents=True, exist_ok=True)
        times, q_values = read_trajectory(trajectory)
        environment = json.loads(environment_path.read_text(encoding="utf-8"))
        moveit = MoveItPy(node_name="stage5a_moveit_validator_moveit")
        model = moveit.get_robot_model()
        group = model.get_joint_model_group(GROUP)
        if group is None or list(group.active_joint_model_names) != JOINTS:
            raise RuntimeError("stage5a_moveit_joint_mapping_mismatch")
        state = RobotState(model)
        objects = make_objects(environment)
        monitor = moveit.get_planning_scene_monitor()
        detector_info = {"requested": requested_detector or None, "status": "default_unspecified", "api": None}
        with monitor.read_write() as scene:
            scene.remove_all_collision_objects()
            for obj in objects:
                scene.apply_collision_object(obj)
            if requested_detector:
                setter = getattr(scene, "set_active_collision_detector", None)
                detector_info["api"] = "set_active_collision_detector" if setter is not None else None
                if setter is None:
                    detector_info["status"] = "requested_detector_api_unavailable"
                else:
                    try:
                        setter(requested_detector)
                        detector_info["status"] = "requested_detector_set"
                    except Exception as exc:
                        detector_info["status"] = "requested_detector_set_failed"
                        detector_info["error"] = repr(exc)
            detector_info["scene_type"] = type(scene).__name__
        write_json(output / "stage5a_planning_scene_loaded.json", {
            "status": "applied",
            "frame_id": environment["frame_id"],
            "object_apply_count": len(objects),
            "object_ids": [obj.id for obj in objects],
            "source_manifest": str(environment_path),
            "scene_api": "PlanningSceneMonitor.read_write + apply_collision_object",
            "formal_acm_modified": False,
        })
        self_contacts: list[dict[str, Any]] = []
        world_contacts: list[dict[str, Any]] = []
        interpolated_self_contacts: list[dict[str, Any]] = []
        interpolated_world_contacts: list[dict[str, Any]] = []
        fk_path = output / "stage5a_d65_fk_trace.jsonl"
        # MoveItPy Jazzy exposes link names on JointModelGroup, not on the
        # RobotModel wrapper. Include fixed links such as base_link and TCP.
        robot_link_names = set(str(name) for name in group.link_model_names) | {
            "base_link", "shoulder_link", "upperarm_link", "forearm_link",
            "wrist1_link", "wrist2_link", "wrist3_link", "spray_tcp_link",
        }
        request = CollisionRequest()
        request.joint_model_group_name = GROUP
        # Jazzy's Python binding cannot convert CollisionResult.contacts to a
        # Python dict. Use the supported self/full collision APIs and scalar
        # collision/contact-count fields; do not treat an inaccessible contact
        # map as evidence.
        request.contacts = False
        request.max_contacts = 4096
        request.max_contacts_per_pair = 256
        with fk_path.open("w", encoding="utf-8") as fk_out, (output / "stage5a_moveit_collision_states.jsonl").open("w", encoding="utf-8") as collision_out, monitor.read_only() as scene:
            for index, (time_s, q) in enumerate(zip(times, q_values)):
                state.set_joint_group_positions(GROUP, q.tolist())
                state.update()
                matrix = matrix_of(state.get_global_link_transform(TCP))
                fk_out.write(json.dumps({"state_index": index, "time_s": float(time_s), "q": q.tolist(), "translation_m": matrix[:3, 3].tolist(), "quaternion_xyzw": quaternion_from_matrix(matrix)}, separators=(",", ":")) + "\n")
                self_result = CollisionResult()
                scene.check_self_collision(request, self_result, state)
                full_result = CollisionResult()
                scene.check_collision(request, full_result, state)
                self_collision = bool(self_result.collision)
                full_collision = bool(full_result.collision)
                # A scalar full-scene query cannot attribute a simultaneous
                # self/world collision. Record only world observations that
                # are not explained by the independent self query.
                world_collision = bool(full_collision and not self_collision)
                if self_collision or world_collision:
                    record = {"state_index": index, "time_s": float(time_s), "collision": full_collision, "self_collision": self_collision, "full_scene_collision": full_collision, "self_contact_count": int(self_result.contact_count), "full_contact_count": int(full_result.contact_count), "self_pairs": [], "all_pairs": []}
                    collision_out.write(json.dumps(record, separators=(",", ":")) + "\n")
                    if self_collision:
                        self_contacts.append(record)
                if world_collision:
                    world_contacts.append({"state_index": index, "time_s": float(time_s), "full_scene_collision": full_collision, "self_collision": self_collision, "contact_count": int(full_result.contact_count)})
        interpolation_path = output / "stage5a_moveit_interpolated_collision_states.jsonl"
        interpolation_checked = 0
        if interpolation_step_deg > 0.0:
            max_step_rad = float(np.deg2rad(interpolation_step_deg))
            with interpolation_path.open("w", encoding="utf-8") as interpolation_out, monitor.read_only() as scene:
                for segment_index, (q0, q1) in enumerate(zip(q_values[:-1], q_values[1:])):
                    segment_steps = max(1, int(np.ceil(np.max(np.abs(q1 - q0)) / max_step_rad)))
                    for local_index in range(segment_steps):
                        alpha = float(local_index) / float(segment_steps)
                        positions = q0 + alpha * (q1 - q0)
                        state.set_joint_group_positions(GROUP, positions.tolist())
                        state.update()
                        self_result = CollisionResult()
                        scene.check_self_collision(request, self_result, state)
                        full_result = CollisionResult()
                        scene.check_collision(request, full_result, state)
                        self_collision = bool(self_result.collision)
                        world_collision = bool(full_result.collision and not self_collision)
                        interpolation_checked += 1
                        if self_collision or world_collision:
                            record = {"segment_index": segment_index, "local_index": local_index, "segment_steps": segment_steps, "alpha": alpha, "collision": bool(full_result.collision), "self_collision": self_collision, "full_contact_count": int(full_result.contact_count), "self_contact_count": int(self_result.contact_count)}
                            interpolation_out.write(json.dumps(record, separators=(",", ":")) + "\n")
                            if self_collision:
                                interpolated_self_contacts.append(record)
                            if world_collision:
                                interpolated_world_contacts.append(record)
                state.set_joint_group_positions(GROUP, q_values[-1].tolist())
                state.update()
                self_result = CollisionResult()
                scene.check_self_collision(request, self_result, state)
                full_result = CollisionResult()
                scene.check_collision(request, full_result, state)
                self_collision = bool(self_result.collision)
                world_collision = bool(full_result.collision and not self_collision)
                interpolation_checked += 1
                if self_collision or world_collision:
                    record = {"segment_index": len(q_values) - 2, "local_index": segment_steps if q_values.size else 0, "segment_steps": segment_steps if q_values.size else 0, "alpha": 1.0, "collision": bool(full_result.collision), "self_collision": self_collision, "full_contact_count": int(full_result.contact_count), "self_contact_count": int(self_result.contact_count)}
                    interpolation_out.write(json.dumps(record, separators=(",", ":")) + "\n")
                    if self_collision:
                        interpolated_self_contacts.append(record)
                    if world_collision:
                        interpolated_world_contacts.append(record)
        else:
            interpolation_path.write_text("", encoding="utf-8")
        fk_tf = self.runtime_tf_crosscheck(moveit, model, runtime_dir, output)
        result = {
            "schema_version": "stage5a-moveit-validation-v1",
            "trajectory_state_count": int(len(q_values)),
            "trajectory_interval_count": int(len(q_values) - 1),
            "joint_mapping": {"group": GROUP, "active_joint_names": list(group.active_joint_model_names), "expected": JOINTS, "match": True},
            "planning_scene": {"loaded": True, "object_count": len(objects), "frame_id": environment["frame_id"], "source": str(environment_path)},
            "collision_detector": detector_info,
            "d65_fk_trace": {"complete": True, "state_count": int(len(q_values)), "path": str(fk_path)},
            "moveit_discrete_self_collision": {"status": "measured", "states_checked": int(len(q_values)), "collision_state_count": len(self_contacts), "colliding_state_indices": [int(x["state_index"]) for x in self_contacts], "pairs": [], "contact_count_sum": int(sum(int(x.get("self_contact_count", 0)) for x in self_contacts)), "pair_detail_status": "unavailable_jazzy_python_contacts_binding", "evidence_path": str((output / "stage5a_moveit_collision_states.jsonl").resolve())},
            "moveit_robot_world_discrete_observation": {"states_with_world_contacts": len(world_contacts), "evidence_path": str((output / "stage5a_moveit_collision_states.jsonl").resolve())},
            "moveit_robot_world_adaptive_discrete_interpolation": {"status": "measured" if interpolation_step_deg > 0.0 else "not_requested", "interpolation_step_deg": interpolation_step_deg, "states_checked": interpolation_checked, "states_with_world_contacts": len(interpolated_world_contacts), "states_with_self_collision": len(interpolated_self_contacts), "evidence_path": str(interpolation_path.resolve())},
            "d65_fk_vs_ros_tf": fk_tf,
            "measurement_complete": bool(len(q_values) == int(environment["trajectory_state_count_expected"])) if "trajectory_state_count_expected" in environment else True,
        }
        result["passed"] = bool(result["planning_scene"]["loaded"] and result["d65_fk_trace"]["complete"] and result["d65_fk_vs_ros_tf"].get("passed"))
        write_json(output / "stage5a_moveit_validation.json", result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return result

    def runtime_tf_crosscheck(self, moveit: MoveItPy, model: Any, runtime_dir: Path, output: Path) -> dict[str, Any]:
        joint_rows = read_jsonl(runtime_dir / "stage5a_joint_states_raw.jsonl")
        tf_rows = read_jsonl(runtime_dir / "stage5a_tf_raw.jsonl")
        static_rows = read_jsonl(runtime_dir / "stage5a_tf_static_raw.jsonl")
        # /tf and /joint_states in this mock chain use different ROS clock
        # stamps (the observed offset is about 0.57 s), while both callbacks
        # are received by the same client process. Group TF transforms by
        # their exact message stamp and pair causally with the latest
        # joint-state callback already observed before that TF callback.
        tf_rows.sort(key=lambda row: float(row["capture_monotonic_s"]))
        dynamic_by_stamp: dict[tuple[int, int], list[dict[str, Any]]] = {}
        for row in tf_rows:
            stamp_key = (int(row["header_stamp"]["sec"]), int(row["header_stamp"]["nanosec"]))
            dynamic_by_stamp.setdefault(stamp_key, []).append(row)
        tf_groups = sorted(
            [(float(np.mean([float(item["capture_monotonic_s"]) for item in rows])), rows) for rows in dynamic_by_stamp.values()],
            key=lambda item: item[0],
        )
        tf_capture_times = [item[0] for item in tf_groups]
        static_edges = {(str(row["parent_frame"]), str(row["child_frame"])): matrix_from_tf(row) for row in static_rows}
        state = RobotState(model)
        position_errors: list[float] = []
        orientation_errors: list[float] = []
        pairing_gaps: list[float] = []
        pairing_window_s = 0.05
        valid_joint_rows = [row for row in joint_rows if len(row.get("name", [])) == 6 and len(set(row.get("name", []))) == 6 and set(row.get("name", [])) == set(JOINTS)]
        valid_joint_rows.sort(key=lambda item: float(item["capture_monotonic_s"]))
        joint_capture_times = [float(item["capture_monotonic_s"]) for item in valid_joint_rows]
        for tf_capture_time, tf_group_rows in tf_groups:
            if not valid_joint_rows:
                continue
            position = bisect.bisect_right(joint_capture_times, tf_capture_time) - 1
            if position < 0:
                continue
            nearest_joint = valid_joint_rows[position]
            nearest_gap = tf_capture_time - joint_capture_times[position]
            if nearest_gap > pairing_window_s:
                continue
            pairing_gaps.append(nearest_gap)
            row = nearest_joint
            names = list(row.get("name", []))
            q = [float(row["position"][names.index(joint)]) for joint in JOINTS]
            edges = {(str(item["parent_frame"]), str(item["child_frame"])): matrix_from_tf(item) for item in tf_group_rows}
            edges.update(static_edges)
            transform_chain = np.eye(4, dtype=float)
            current = "base_link"
            visited: set[str] = set()
            while current != TCP and current not in visited:
                visited.add(current)
                outgoing = [(child, edge) for (parent, child), edge in edges.items() if parent == current]
                if not outgoing:
                    break
                child, edge = outgoing[0]
                transform_chain = transform_chain @ edge
                current = child
            if current != TCP:
                continue
            state.set_joint_group_positions(GROUP, q)
            state.update()
            matrix = matrix_of(state.get_global_link_transform(TCP))
            position_errors.append(float(np.linalg.norm(matrix[:3, 3] - transform_chain[:3, 3])))
            expected = np.asarray(quaternion_from_matrix(matrix), dtype=float)
            observed = np.asarray(quaternion_from_matrix(transform_chain), dtype=float)
            if np.linalg.norm(observed) > 1e-12:
                observed /= np.linalg.norm(observed)
            expected /= max(float(np.linalg.norm(expected)), 1e-12)
            orientation_errors.append(float(2.0 * math.acos(float(np.clip(abs(np.dot(expected, observed)), -1.0, 1.0)))))
        stats = {
            "source": "runtime /joint_states + composed dynamic /tf tree + /tf_static TCP edge, independent MoveItPy FK",
            "joint_state_rows": len(joint_rows),
            "tf_rows": len(tf_rows),
            "samples_compared": len(position_errors),
            "max_position_error_m": max(position_errors) if position_errors else None,
            "mean_position_error_m": float(np.mean(position_errors)) if position_errors else None,
            "rms_position_error_m": float(np.sqrt(np.mean(np.square(position_errors)))) if position_errors else None,
            "max_orientation_error_rad": max(orientation_errors) if orientation_errors else None,
            "rms_orientation_error_rad": float(np.sqrt(np.mean(np.square(orientation_errors)))) if orientation_errors else None,
            "pairing": "latest causally prior same-process joint-state capture within 0.05 s; ROS header clock offset retained",
            "max_pairing_gap_s": max(pairing_gaps) if pairing_gaps else None,
            "tolerance_position_m": 1e-5,
            "tolerance_orientation_rad": 1e-5,
        }
        stats["passed"] = bool(position_errors and orientation_errors and stats["max_position_error_m"] <= stats["tolerance_position_m"] and stats["max_orientation_error_rad"] <= stats["tolerance_orientation_rad"])
        write_json(output / "stage5a_d65_fk_vs_ros_tf.json", stats)
        return stats


def main() -> int:
    rclpy.init()
    node = Validator()
    try:
        result = node.run()
        code = 0 if result.get("passed") else 2
    except Exception as exc:
        output = Path(str(node.get_parameter("output_dir").value)).resolve()
        write_json(output / "stage5a_moveit_validation.json", {"schema_version": "stage5a-moveit-validation-v1", "passed": False, "measurement_complete": False, "error": repr(exc)})
        print(json.dumps({"passed": False, "error": repr(exc)}), flush=True)
        code = 2
    finally:
        node.destroy_node()
        rclpy.shutdown()
    # MoveItPy/Jazzy has a known post-measurement destructor hazard in this
    # one-shot worker.  All evidence is closed before process termination.
    os._exit(code)


if __name__ == "__main__":
    main()
