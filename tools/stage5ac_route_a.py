"""Sequence-level collision-aware IK shadow for Stage 5A-C Route A.

The shadow preserves the authoritative Stage 0/1 181-point task poses and
tries explicit deterministic seed families.  It never writes D65 or changes
the protected Stage 5A-R scene.  Results are diagnostic until the candidate
is independently retimed and verified.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


JOINTS = ["j1", "j2", "j3", "j4", "j5", "j6"]
GROUP = "fairino5_v6_group"
EE = "spray_tcp_link"
ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def env_objects(environment: dict[str, Any]):
    from moveit_msgs.msg import CollisionObject
    from shape_msgs.msg import SolidPrimitive
    from geometry_msgs.msg import Pose

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


def pose_message(row: dict[str, str], slack_m: float = 0.0):
    from geometry_msgs.msg import Pose

    pose = Pose()
    pose.position.x = float(row["x"]) - float(slack_m) * float(row.get("nx", 0.0))
    pose.position.y = float(row["y"]) - float(slack_m) * float(row.get("ny", 0.0))
    pose.position.z = float(row["z"]) - float(slack_m) * float(row.get("nz", 0.0))
    pose.orientation.x = float(row["qx"])
    pose.orientation.y = float(row["qy"])
    pose.orientation.z = float(row["qz"])
    pose.orientation.w = float(row["qw"])
    return pose


def matrix_of(value: Any) -> np.ndarray:
    raw = value.matrix() if hasattr(value, "matrix") else value
    return np.asarray(raw, dtype=float)


def orientation_error(actual: np.ndarray, target: dict[str, str]) -> float:
    from scipy.spatial.transform import Rotation

    observed = Rotation.from_matrix(actual[:3, :3]).as_quat()
    expected = np.asarray([float(target[k]) for k in ("qx", "qy", "qz", "qw")], dtype=float)
    observed /= max(float(np.linalg.norm(observed)), 1e-15)
    expected /= max(float(np.linalg.norm(expected)), 1e-15)
    return float(2.0 * math.acos(float(np.clip(abs(np.dot(observed, expected)), -1.0, 1.0))))


def equivalent_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    delta = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return (delta + math.pi) % (2.0 * math.pi) - math.pi


def seed_templates(seed: np.ndarray, previous: np.ndarray | None, historical: np.ndarray | None) -> list[tuple[str, np.ndarray]]:
    """Use explicit, reproducible branch seeds; no hidden random restarts."""

    requested: list[tuple[str, np.ndarray]] = [("nominal", seed.copy())]
    if previous is not None:
        requested.append(("previous", previous.copy()))
    if historical is not None:
        requested.append(("historical_stage19", historical.copy()))
    for joint, name in ((0, "shoulder"), (2, "elbow"), (4, "wrist")):
        for sign in (-1.0, 1.0):
            candidate = seed.copy()
            candidate[joint] += sign * math.pi
            requested.append((f"{name}_{'minus' if sign < 0 else 'plus'}_pi", candidate))
    for joints, name in (((0, 2), "shoulder_elbow"), ((0, 4), "shoulder_wrist"), ((2, 4), "elbow_wrist"), ((0, 2, 4), "shoulder_elbow_wrist")):
        candidate = seed.copy()
        for joint in joints:
            candidate[joint] += math.pi
        requested.append((f"{name}_pi", candidate))
    # Cover the complete deterministic +/- pi branch set for the three
    # principal arm joints.  The historical seed is a shadow-only warm start,
    # never a replacement for the authoritative Stage 0/1 seed input.
    import itertools
    for signs in itertools.product((-1.0, 1.0), repeat=3):
        candidate = seed.copy()
        for joint, sign in zip((0, 2, 4), signs):
            candidate[joint] += sign * math.pi
        requested.append(("arm_branch_" + "_".join("m" if sign < 0 else "p" for sign in signs), candidate))
    if historical is not None:
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            candidate = historical.copy()
            for joint, sign in zip((0, 2, 4), signs):
                candidate[joint] += sign * math.pi
            requested.append(("historical_arm_branch_" + "_".join("m" if sign < 0 else "p" for sign in signs), candidate))
    result: list[tuple[str, np.ndarray]] = []
    seen: set[tuple[float, ...]] = set()
    for name, candidate in requested:
        key = tuple(float(x) for x in candidate)
        if key not in seen:
            seen.add(key)
            result.append((name, candidate))
    return result


def run(node: Any) -> int:  # pragma: no cover - ROS 2 runtime
    import rclpy
    from moveit.core.robot_state import RobotState
    from moveit.planning import MoveItPy
    from moveit.core.collision_detection import CollisionRequest, CollisionResult

    poses_path = Path(str(node.get_parameter("poses_csv").value)).resolve()
    seeds_path = Path(str(node.get_parameter("seeds_csv").value)).resolve()
    environment_path = Path(str(node.get_parameter("environment_json").value)).resolve()
    output = Path(str(node.get_parameter("output_dir").value)).resolve()
    historical_path_raw = str(node.get_parameter("historical_seed_csv").value)
    historical_rows = read_csv(Path(historical_path_raw).resolve()) if historical_path_raw else []
    task_slack_m = float(node.get_parameter("task_slack_m").value)
    task_offset_x_m = float(node.get_parameter("task_offset_x_m").value)
    task_offset_y_m = float(node.get_parameter("task_offset_y_m").value)
    task_offset_z_m = float(node.get_parameter("task_offset_z_m").value)
    task_offset_start = int(node.get_parameter("task_offset_start_waypoint").value)
    task_offset_end = int(node.get_parameter("task_offset_end_waypoint").value)
    beam_width = int(node.get_parameter("beam_width").value)
    max_step_deg = float(node.get_parameter("max_step_deg").value)
    poses = read_csv(poses_path)
    seeds_rows = read_csv(seeds_path)
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    if len(poses) != 181 or len(seeds_rows) != 181 or (historical_rows and len(historical_rows) != 181):
        raise RuntimeError(f"stage5ac_route_a_input_count:{len(poses)}/{len(seeds_rows)}")
    moveit = MoveItPy(node_name="stage5ac_route_a_moveit")
    model = moveit.get_robot_model()
    group_model = model.get_joint_model_group(GROUP)
    if group_model is None or list(group_model.active_joint_model_names) != JOINTS:
        raise RuntimeError("stage5ac_route_a_joint_mapping_mismatch")
    psm = moveit.get_planning_scene_monitor()
    with psm.read_write() as scene:
        scene.remove_all_collision_objects()
        for obj in env_objects(environment):
            scene.apply_collision_object(obj)
    request = CollisionRequest()
    request.joint_model_group_name = GROUP
    request.contacts = False
    request.max_contacts = 4096
    state = RobotState(model)
    layers: list[list[dict[str, Any]]] = []
    attempts: list[dict[str, Any]] = []
    previous = None
    for waypoint, (pose_row, seed_row) in enumerate(zip(poses, seeds_rows)):
        target_row = dict(pose_row)
        use_cartesian_offset = task_offset_start <= waypoint <= task_offset_end
        offset_x = task_offset_x_m if use_cartesian_offset else 0.0
        offset_y = task_offset_y_m if use_cartesian_offset else 0.0
        offset_z = task_offset_z_m if use_cartesian_offset else 0.0
        target_row["x"] = str(float(pose_row["x"]) + offset_x - task_slack_m * float(pose_row.get("nx", 0.0)))
        target_row["y"] = str(float(pose_row["y"]) + offset_y - task_slack_m * float(pose_row.get("ny", 0.0)))
        target_row["z"] = str(float(pose_row["z"]) + offset_z - task_slack_m * float(pose_row.get("nz", 0.0)))
        seed = np.asarray([float(seed_row[f"q{i}"]) for i in range(1, 7)], dtype=float)
        historical = None
        if historical_rows:
            historical = np.asarray([float(historical_rows[waypoint][f"j{i}_q"]) for i in range(1, 7)], dtype=float)
        layer: list[dict[str, Any]] = []
        # The historical route is an observed pose-realizing shadow seed, not
        # merely an IK initial guess.  Evaluate it directly as well: a solver
        # may reject a valid near-limit configuration even when its FK is exact.
        if historical is not None:
            state.set_joint_group_positions(GROUP, historical.tolist())
            state.update()
            fk = matrix_of(state.get_global_link_transform(EE))
            position_error = float(np.linalg.norm(fk[:3, 3] - np.asarray([float(target_row[k]) for k in ("x", "y", "z")], dtype=float)))
            rotation_error = orientation_error(fk, target_row)
            direct_self = CollisionResult()
            with psm.read_only() as scene:
                scene.check_self_collision(request, direct_self, state)
                direct_full = CollisionResult()
                scene.check_collision(request, direct_full, state)
            direct_self_collision = bool(direct_self.collision)
            direct_world_collision = bool(direct_full.collision and not direct_self_collision)
            direct_valid_fk = bool(position_error <= 1.0e-4 and rotation_error <= 1.0e-3)
            direct_valid = bool(direct_valid_fk and not direct_self_collision and not direct_world_collision and np.all(np.isfinite(historical)))
            direct_record = {"waypoint": waypoint, "seed_family": "historical_direct", "solver_success": True, "seed": historical.tolist(), "q": historical.tolist(), "position_error_m": position_error, "orientation_error_rad": rotation_error, "self_collision": direct_self_collision, "robot_world_collision": direct_world_collision, "full_collision": bool(direct_full.collision), "valid_fk": direct_valid_fk, "valid_collision_free_node": direct_valid}
            attempts.append(direct_record)
            if direct_valid:
                layer.append(direct_record)
        for family, template in seed_templates(seed, previous, historical):
            state.set_joint_group_positions(GROUP, template.tolist())
            state.update()
            solved = bool(state.set_from_ik(GROUP, pose_message(target_row, 0.0), EE, 0.0))
            record: dict[str, Any] = {"waypoint": waypoint, "seed_family": family, "solver_success": solved, "seed": template.tolist()}
            if not solved:
                attempts.append(record)
                continue
            state.update()
            q = np.asarray(state.get_joint_group_positions(GROUP), dtype=float).copy()
            fk = matrix_of(state.get_global_link_transform(EE))
            position_error = float(np.linalg.norm(fk[:3, 3] - np.asarray([float(target_row[k]) for k in ("x", "y", "z")], dtype=float)))
            rotation_error = orientation_error(fk, target_row)
            self_result = CollisionResult()
            with psm.read_only() as scene:
                scene.check_self_collision(request, self_result, state)
                full_result = CollisionResult()
                scene.check_collision(request, full_result, state)
            self_collision = bool(self_result.collision)
            world_collision = bool(full_result.collision and not self_collision)
            valid_fk = bool(position_error <= 1.0e-4 and rotation_error <= 1.0e-3)
            valid = bool(valid_fk and not self_collision and not world_collision and np.all(np.isfinite(q)))
            record.update({"q": q.tolist(), "position_error_m": position_error, "orientation_error_rad": rotation_error, "self_collision": self_collision, "robot_world_collision": world_collision, "full_collision": bool(full_result.collision), "valid_fk": valid_fk, "valid_collision_free_node": valid})
            attempts.append(record)
            if valid:
                layer.append(record)
        layers.append(layer)
        if layer:
            previous = np.asarray(min(layer, key=lambda item: float(np.linalg.norm(equivalent_delta(np.asarray(item["q"]), seed))))["q"], dtype=float)

    # Exact deterministic layered dynamic programming over collision-free
    # nodes.  The earlier beam implementation could prune a locally cheap
    # branch that was the only continuation, despite every adjacent layer
    # having a valid edge.  This O(sum(|L_i||L_{i+1}|)) shadow search retains
    # the minimum-cost path to every target node and therefore has no such
    # false negative.  Continuous robot-world validation remains a later gate.
    ordered_layers = [sorted(layer, key=lambda item: (item["seed_family"], item["q"])) for layer in layers]
    paths: list[tuple[float, list[dict[str, Any]]]] = [(0.0, [item]) for item in ordered_layers[0]] if ordered_layers and ordered_layers[0] else []
    first_unreachable = None if paths else 0
    for waypoint in range(1, len(ordered_layers)):
        targets = ordered_layers[waypoint]
        if not targets or not paths:
            first_unreachable = waypoint
            paths = []
            break
        best_for_target: list[tuple[float, list[dict[str, Any]]]] = []
        for target in targets:
            choices: list[tuple[float, list[dict[str, Any]]]] = []
            target_q = np.asarray(target["q"], dtype=float)
            for cost, path in paths:
                source_q = np.asarray(path[-1]["q"], dtype=float)
                delta = equivalent_delta(target_q, source_q)
                step_deg = float(np.rad2deg(np.max(np.abs(delta))))
                if step_deg <= max_step_deg:
                    choices.append((cost + float(np.linalg.norm(delta)), path + [target]))
            if choices:
                choices.sort(key=lambda item: (item[0], tuple((x["seed_family"], x["q"]) for x in item[1])))
                best_for_target.append(choices[0])
        if not best_for_target:
            first_unreachable = waypoint
            paths = []
            break
        paths = best_for_target
    selected = min(paths, key=lambda item: (item[0], tuple((x["seed_family"], x["q"]) for x in item[1])))[1] if paths and first_unreachable is None else []
    output.mkdir(parents=True, exist_ok=True)
    with (output / "STAGE5AC_ROUTE_A_ATTEMPTS.jsonl").open("w", encoding="utf-8") as handle:
        for record in attempts:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
    with (output / "STAGE5AC_ROUTE_A_SELECTED.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["waypoint"] + [f"j{i}_q" for i in range(1, 7)]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for record in selected:
            writer.writerow({"waypoint": record["waypoint"], **{f"j{i}_q": record["q"][i - 1] for i in range(1, 7)}})
    summary = {
        "schema_version": "stage5ac-route-a-v1",
        "route": "A_preserve_tcp_sequence_level_ik",
        "source_poses": str(poses_path),
        "source_seeds": str(seeds_path),
        "environment": str(environment_path),
        "historical_seed_csv": historical_path_raw or None,
        "task_slack_m_along_negative_normal": task_slack_m,
        "task_offset_xyz_m": [task_offset_x_m, task_offset_y_m, task_offset_z_m],
        "task_offset_waypoint_window": [task_offset_start, task_offset_end],
        "waypoint_count": 181,
        "attempt_count": len(attempts),
        "node_count": sum(len(layer) for layer in layers),
        "node_count_by_waypoint": [len(layer) for layer in layers],
        "collision_free_node_count": sum(1 for item in attempts if item.get("valid_collision_free_node")),
        "world_collision_node_count": sum(1 for item in attempts if item.get("robot_world_collision")),
        "self_collision_node_count": sum(1 for item in attempts if item.get("self_collision")),
        "fk_invalid_node_count": sum(1 for item in attempts if item.get("solver_success") and not item.get("valid_fk")),
        "beam_width": beam_width,
        "max_step_deg": max_step_deg,
        "path_search": "exact_layered_dynamic_programming",
        "selected_waypoint_count": len(selected),
        "first_unreachable_waypoint": first_unreachable,
        "full_path_found": bool(len(selected) == 181),
        "continuous_robot_world_collision": "not_run_route_a_node_screening_only",
        "exact_articulated_self_ccd": "not_available",
        "promotion": "NO_PROMOTION",
    }
    write_json(output / "STAGE5AC_ROUTE_A_SUMMARY.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


def main() -> int:
    import rclpy

    rclpy.init()
    from rclpy.node import Node

    node = Node("stage5ac_route_a_wrapper")
    node.declare_parameter("poses_csv", "")
    node.declare_parameter("seeds_csv", "")
    node.declare_parameter("environment_json", "")
    node.declare_parameter("output_dir", "")
    node.declare_parameter("historical_seed_csv", "")
    node.declare_parameter("task_slack_m", 0.0)
    node.declare_parameter("task_offset_x_m", 0.0)
    node.declare_parameter("task_offset_y_m", 0.0)
    node.declare_parameter("task_offset_z_m", 0.0)
    node.declare_parameter("task_offset_start_waypoint", -1)
    node.declare_parameter("task_offset_end_waypoint", -1)
    node.declare_parameter("beam_width", 12)
    node.declare_parameter("max_step_deg", 20.0)
    try:
        return run(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
