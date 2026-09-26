"""Stage 5B offline-only branch, timing, and JTC spline repair.

This tool deliberately writes only a shadow candidate.  It keeps the 181
authoritative Cartesian poses fixed, searches multiple IK branches at every
pose, then regenerates a mathematically self-consistent quintic JTC message.
It does not start Gazebo, alter the nominal world, or promote any result.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.controller_interpolation import audit_intervals, evaluate_coefficients, segment_coefficients


JOINT_NAMES = [f"j{i}" for i in range(1, 7)]
GROUP = "fairino5_v6_group"
EE = "spray_tcp_link"
CONTINUITY_GATE_DEG = 20.0
# Stage 0/1 authority from config/ik_graph.yaml.  The q source is allowed to
# be an approximate FK realization inside this existing task gate; using an
# invented 0.1 mm gate here would silently turn a real Stage 0/1 candidate
# into an artificial empty layer.
POSITION_TOLERANCE_M = 0.006
ORIENTATION_TOLERANCE_RAD = math.radians(10.0)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def matrix_of(value: Any) -> np.ndarray:
    raw = value.matrix() if hasattr(value, "matrix") else value
    return np.asarray(raw, dtype=float)


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


def seed_variants(seed: np.ndarray) -> list[np.ndarray]:
    """Generate deterministic nearby branch seeds without changing task poses."""

    values = [np.asarray(seed, dtype=float).copy()]
    for joint in range(6):
        for sign in (-1.0, 1.0):
            candidate = values[0].copy()
            candidate[joint] += sign * math.pi
            values.append(candidate)
    for joints in ((0, 2), (0, 4), (2, 4), (0, 2, 4), (1, 3, 5)):
        for sign in (-1.0, 1.0):
            candidate = values[0].copy()
            for joint in joints:
                candidate[joint] += sign * math.pi
            values.append(candidate)
    return values


def unique_vectors(values: list[np.ndarray], tolerance: float = 1.0e-7) -> list[np.ndarray]:
    result: list[np.ndarray] = []
    for value in values:
        value = np.asarray(value, dtype=float)
        if value.shape != (6,) or not np.isfinite(value).all():
            continue
        if not any(float(np.max(np.abs(value - old))) <= tolerance for old in result):
            result.append(value.copy())
    return result


def read_environment(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def env_objects(environment: dict[str, Any]):
    from geometry_msgs.msg import Pose
    from moveit_msgs.msg import CollisionObject
    from shape_msgs.msg import SolidPrimitive

    objects = []
    for item in environment.get("objects", []):
        obj = CollisionObject()
        obj.header.frame_id = str(environment.get("frame_id", "base_link"))
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


def q_rows(path: Path) -> np.ndarray:
    rows = read_csv(path)
    return np.asarray([[float(row[f"j{i}_q"]) for i in range(1, 7)] for row in rows], dtype=float)


def trajectory_arrays(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the persisted q/dq/ddq/t source without changing any value."""

    rows = read_csv(path)
    times = np.asarray([float(row["t"]) for row in rows], dtype=float)
    positions = np.asarray([[float(row[f"j{i}_q"]) for i in range(1, 7)] for row in rows], dtype=float)
    velocities = np.asarray([[float(row[f"j{i}_dq"]) for i in range(1, 7)] for row in rows], dtype=float)
    accelerations = np.asarray([[float(row[f"j{i}_ddq"]) for i in range(1, 7)] for row in rows], dtype=float)
    if len(times) != 181 or positions.shape != (181, 6) or velocities.shape != (181, 6) or accelerations.shape != (181, 6):
        raise RuntimeError(f"stage5b_source_trajectory_shape:{len(times)}/{positions.shape}/{velocities.shape}/{accelerations.shape}")
    if not np.isfinite(times).all() or not np.isfinite(positions).all() or not np.isfinite(velocities).all() or not np.isfinite(accelerations).all():
        raise RuntimeError("stage5b_source_trajectory_nonfinite")
    if not np.all(np.diff(times) > 0.0):
        raise RuntimeError("stage5b_source_timestamps_not_strictly_increasing")
    return times, positions, velocities, accelerations


def load_motion_limits(path: Path, position_lower: np.ndarray, position_upper: np.ndarray) -> tuple[list[dict[str, float]], dict[str, Any]]:
    """Read velocity/acceleration/jerk limits from the checked-in YAML."""

    import yaml

    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    configured = document.get("joint_limits", {})
    limits: list[dict[str, float]] = []
    for index, joint in enumerate(JOINT_NAMES):
        item = configured.get(joint, {})
        if not item.get("has_velocity_limits") or not item.get("has_acceleration_limits") or not item.get("has_jerk_limits"):
            raise RuntimeError(f"stage5b_missing_motion_limit_flags:{joint}")
        values = {"max_velocity": float(item["max_velocity"]), "max_acceleration": float(item["max_acceleration"]), "max_jerk": float(item["max_jerk"])}
        if not all(np.isfinite(value) and value > 0.0 for value in values.values()):
            raise RuntimeError(f"stage5b_invalid_motion_limits:{joint}")
        limits.append({
            "position_lower_rad": float(position_lower[index]),
            "position_upper_rad": float(position_upper[index]),
            "max_velocity_rad_s": values["max_velocity"],
            "max_acceleration_rad_s2": values["max_acceleration"],
            "max_jerk_rad_s3": values["max_jerk"],
        })
    return limits, {joint: limits[index] for index, joint in enumerate(JOINT_NAMES)}


def audit_summary(audit: dict[str, Any]) -> dict[str, Any]:
    """Drop coefficient objects from the audit while retaining decision evidence."""

    return {
        "method": audit["method"],
        "trajectory_points": audit["trajectory_points"],
        "trajectory_intervals": audit["trajectory_intervals"],
        "all_intervals_reconstructed": audit["all_intervals_reconstructed"],
        "maximums": audit["maximums"],
        "status": audit["status"],
    }


def limits_from_group(group: Any) -> tuple[np.ndarray, np.ndarray]:
    lows, highs = [], []
    for bound in group.active_joint_model_bounds:
        bound = bound[0] if isinstance(bound, (list, tuple)) else bound
        lows.append(float(bound.min_position))
        highs.append(float(bound.max_position))
    return np.asarray(lows), np.asarray(highs)


def load_pose(row: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    position = np.asarray([float(row[key]) for key in ("x", "y", "z")], dtype=float)
    quaternion = np.asarray([float(row[key]) for key in ("qx", "qy", "qz", "qw")], dtype=float)
    quaternion /= max(float(np.linalg.norm(quaternion)), 1.0e-15)
    return position, quaternion


def edge_step_deg(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.rad2deg(np.max(np.abs(shortest_delta(right, left)))))


def continuity_map(q: np.ndarray) -> dict[str, Any]:
    """Return the complete adjacent-joint continuity map for a q sequence."""

    intervals = []
    for index, (left, right) in enumerate(zip(q, q[1:])):
        raw_delta = np.asarray(right, dtype=float) - np.asarray(left, dtype=float)
        wrap_delta = shortest_delta(right, left)
        joint_index = int(np.argmax(np.abs(wrap_delta)))
        intervals.append({
            "from_waypoint": index,
            "to_waypoint": index + 1,
            "max_wrap_aware_step_deg": float(np.rad2deg(np.max(np.abs(wrap_delta)))),
            "max_raw_step_deg": float(np.rad2deg(np.max(np.abs(raw_delta)))),
            "worst_joint": JOINT_NAMES[joint_index],
            "delta_q_wrapped_rad": wrap_delta.tolist(),
            "delta_q_raw_rad": raw_delta.tolist(),
        })
    ranked = sorted(intervals, key=lambda item: item["max_wrap_aware_step_deg"], reverse=True)
    return {
        "interval_count": len(intervals),
        "over_gate_count": sum(item["max_wrap_aware_step_deg"] > CONTINUITY_GATE_DEG + 1.0e-10 for item in intervals),
        "maximum": ranked[0] if ranked else None,
        "second_maximum": ranked[1] if len(ranked) > 1 else None,
        "intervals": intervals,
    }


def make_candidate_record(q: np.ndarray, position_error: float, orientation_error: float, collision: bool, seed_index: int) -> dict[str, Any]:
    return {
        "q": [float(x) for x in q],
        "fk_position_error_m": float(position_error),
        "fk_orientation_error_rad": float(orientation_error),
        "collision": bool(collision),
        "seed_index": int(seed_index),
    }


def collision_check(scene: Any, request: Any, state: Any) -> bool:
    from moveit.core.collision_detection import CollisionResult

    result = CollisionResult()
    scene.check_collision(request, result, state)
    return bool(result.collision)


def build_layers(node: Any, poses: list[dict[str, str]], q_source: np.ndarray, environment: dict[str, Any], include_collision: bool, ik_timeout_s: float) -> tuple[Any, list[list[dict[str, Any]]], dict[str, Any]]:
    from moveit.core.collision_detection import CollisionRequest
    from moveit.core.robot_state import RobotState
    from moveit.planning import MoveItPy

    moveit = MoveItPy(node_name="stage5b_offline_repair_moveit")
    model = moveit.get_robot_model()
    group = model.get_joint_model_group(GROUP)
    if group is None or list(group.active_joint_model_names) != JOINT_NAMES:
        raise RuntimeError("stage5b_joint_mapping_mismatch")
    lows, highs = limits_from_group(group)
    psm = moveit.get_planning_scene_monitor()
    with psm.read_write() as scene:
        scene.remove_all_collision_objects()
        for obj in env_objects(environment):
            scene.apply_collision_object(obj)
    request = CollisionRequest()
    request.joint_model_group_name = GROUP
    request.contacts = False
    layers: list[list[dict[str, Any]]] = []
    layer_sizes: list[int] = []
    source_anchor_diagnostics: list[dict[str, Any]] = []
    reject_counts = {"ik_failure": 0, "nonfinite_or_limits": 0, "fk_tolerance": 0, "collision": 0}

    for index, row in enumerate(poses):
        position, quaternion = load_pose(row)
        seeds: list[np.ndarray] = []
        # The old graph omitted the local source state.  Keep it first so a
        # known valid source solution cannot disappear due to seed starvation.
        for offset in (0, -1, 1, -2, 2, -3, 3):
            neighbour = min(len(q_source) - 1, max(0, index + offset))
            seeds.extend(seed_variants(q_source[neighbour]))
        if index:
            seeds.extend(np.asarray(item["q"], dtype=float) for item in layers[-1])
        layer: list[dict[str, Any]] = []
        for seed_index, seed in enumerate(unique_vectors(seeds)):
            state = RobotState(model)
            state.set_joint_group_positions(GROUP, seed.tolist())
            state.update()
            solved = bool(state.set_from_ik(GROUP, pose_from_arrays(position, quaternion), EE, ik_timeout_s))
            if not solved:
                reject_counts["ik_failure"] += 1
                continue
            state.update()
            q = np.asarray(state.get_joint_group_positions(GROUP), dtype=float).copy()
            if not np.isfinite(q).all() or np.any(q < lows - 1.0e-10) or np.any(q > highs + 1.0e-10):
                reject_counts["nonfinite_or_limits"] += 1
                continue
            fk = matrix_of(state.get_global_link_transform(EE))
            position_error, orientation_error = pose_error(fk, position, quaternion)
            if position_error > POSITION_TOLERANCE_M or orientation_error > ORIENTATION_TOLERANCE_RAD:
                reject_counts["fk_tolerance"] += 1
                continue
            colliding = collision_check(scene, request, state) if include_collision else False
            if colliding:
                reject_counts["collision"] += 1
                continue
            if any(float(np.max(np.abs(q - np.asarray(item["q"]))) <= 1.0e-6) for item in layer):
                continue
            layer.append(make_candidate_record(q, position_error, orientation_error, colliding, seed_index))
        # Always retain the exact source state as a diagnostic anchor if it
        # passes FK/limits/collision. This is not a fake solution: it is
        # independently checked through MoveIt FK and PlanningScene.
        source_state = RobotState(model)
        source_state.set_joint_group_positions(GROUP, q_source[index].tolist())
        source_state.update()
        fk = matrix_of(source_state.get_global_link_transform(EE))
        position_error, orientation_error = pose_error(fk, position, quaternion)
        colliding = collision_check(scene, request, source_state) if include_collision else False
        source_anchor_diagnostics.append({"waypoint": index, "fk_position_error_m": position_error, "fk_orientation_error_rad": orientation_error, "collision": colliding, "accepted": bool(position_error <= POSITION_TOLERANCE_M and orientation_error <= ORIENTATION_TOLERANCE_RAD and not colliding)})
        if position_error <= POSITION_TOLERANCE_M and orientation_error <= ORIENTATION_TOLERANCE_RAD and not colliding:
            if not any(float(np.max(np.abs(q_source[index] - np.asarray(item["q"]))) <= 1.0e-6) for item in layer):
                layer.append(make_candidate_record(q_source[index], position_error, orientation_error, colliding, -1))
        layer.sort(key=lambda item: tuple(item["q"]))
        layers.append(layer)
        layer_sizes.append(len(layer))
        if not layer:
            node.get_logger().error(f"stage5b_empty_layer:{index}")
            break
    return moveit, layers, {"reject_counts": reject_counts, "joint_lower_rad": lows.tolist(), "joint_upper_rad": highs.tolist(), "layers_built": len(layers), "layer_sizes": layer_sizes, "source_anchor_diagnostics": source_anchor_diagnostics, "source_anchor_failures": [item for item in source_anchor_diagnostics if not item["accepted"]]}


def select_path(layers: list[list[dict[str, Any]]], q_source: np.ndarray, pin_endpoints: bool = True) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(layers) != len(q_source) or any(not layer for layer in layers):
        return [], {"status": "UNRESOLVED", "reason": "empty_or_incomplete_layer"}
    if pin_endpoints:
        layers = [list(layer) for layer in layers]
        for index in (0, -1):
            layers[index] = [item for item in layers[index] if float(np.max(np.abs(np.asarray(item["q"]) - q_source[index]))) <= 1.0e-6]
    if not layers[0] or not layers[-1]:
        return [], {"status": "UNRESOLVED", "reason": "endpoint_anchor_missing"}
    # State per candidate: (bottleneck, accumulated squared travel, path ids).
    states: list[tuple[float, float, list[int]]] = [(0.0, 0.0, [i]) for i in range(len(layers[0]))]
    for index in range(1, len(layers)):
        next_states: list[tuple[float, float, list[int]] | None] = [None] * len(layers[index])
        for target_index, target in enumerate(layers[index]):
            target_q = np.asarray(target["q"], dtype=float)
            for previous_index, state in enumerate(states):
                if state is None:
                    continue
                previous_q = np.asarray(layers[index - 1][previous_index]["q"], dtype=float)
                step = edge_step_deg(previous_q, target_q)
                if step > CONTINUITY_GATE_DEG + 1.0e-10:
                    continue
                candidate = (max(state[0], step), state[1] + float(np.dot(shortest_delta(target_q, previous_q), shortest_delta(target_q, previous_q))), state[2] + [target_index])
                current = next_states[target_index]
                if current is None or candidate[:2] < current[:2]:
                    next_states[target_index] = candidate
        if all(item is None for item in next_states):
            best_bridge: tuple[float, int, int, np.ndarray] | None = None
            for previous_index, state in enumerate(states):
                if state is None:
                    continue
                previous_q = np.asarray(layers[index - 1][previous_index]["q"], dtype=float)
                for target_index, target in enumerate(layers[index]):
                    target_q = np.asarray(target["q"], dtype=float)
                    delta = shortest_delta(target_q, previous_q)
                    candidate = (float(np.rad2deg(np.max(np.abs(delta)))), previous_index, target_index, delta)
                    if best_bridge is None or candidate[0] < best_bridge[0]:
                        best_bridge = candidate
            bridge_report = None
            if best_bridge is not None:
                bridge_report = {"minimum_wrap_aware_step_deg": best_bridge[0], "from_waypoint": index - 1, "to_waypoint": index, "from_candidate_index": best_bridge[1], "to_candidate_index": best_bridge[2], "delta_q_wrapped_rad": best_bridge[3].tolist()}
            return [], {"status": "UNRESOLVED", "reason": "no_path_under_continuity_gate", "first_unreachable_waypoint": index, "best_bridge_without_gate": bridge_report}
        states = [item for item in next_states if item is not None]
        # The filtering above changes indices only if candidates are removed;
        # retain a compact layer aligned with the surviving states.
        layers[index] = [layers[index][target_index] for target_index, item in enumerate(next_states) if item is not None]
    winner = min(states, key=lambda item: item[:2])
    path = [layers[index][candidate_index] for index, candidate_index in enumerate(winner[2])]
    steps = [edge_step_deg(np.asarray(path[i - 1]["q"]), np.asarray(path[i]["q"])) for i in range(1, len(path))]
    return path, {"status": "PASS", "max_wrap_aware_step_deg": max(steps, default=0.0), "sum_squared_wrap_aware_step_rad2": winner[1], "worst_interval_left": int(np.argmax(steps)) if steps else None, "worst_interval_right": int(np.argmax(steps)) + 1 if steps else None}


def rebuild_zero_knot_quintic(path: list[dict[str, Any]], limits: list[dict[str, float]], duration_scale: float = 1.10) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    q = np.asarray([item["q"] for item in path], dtype=float)
    v = np.zeros_like(q)
    a = np.zeros_like(q)
    durations = []
    raw_steps = []
    wrap_steps = []
    for left, right in zip(q, q[1:]):
        # JTC interpolates commanded revolute positions as ordinary numbers;
        # it does not apply circular wrapping to a position-only/quintic
        # segment.  Use the raw commanded delta for timing, while retaining
        # the wrap-aware value for the separate continuity gate.
        raw_delta = np.abs(right - left)
        wrap_delta = np.abs(shortest_delta(right, left))
        raw_steps.append(float(np.rad2deg(np.max(raw_delta))))
        wrap_steps.append(float(np.rad2deg(np.max(wrap_delta))))
        delta = raw_delta
        times = []
        for joint, distance in enumerate(delta):
            if distance <= 1.0e-12:
                continue
            vmax = float(limits[joint]["max_velocity_rad_s"])
            amax = float(limits[joint]["max_acceleration_rad_s2"])
            jmax = float(limits[joint]["max_jerk_rad_s3"])
            times.extend([(1.875 * distance) / vmax, math.sqrt((5.773502691896258 * distance) / amax), (60.0 * distance / jmax) ** (1.0 / 3.0)])
        durations.append(max(0.05, duration_scale * max(times, default=0.05)))
    t = np.concatenate(([0.0], np.cumsum(np.asarray(durations, dtype=float))))
    audit = audit_intervals(t, q, v, a, "quintic", limits, JOINT_NAMES)
    return q, v, a, t, {"duration_scale": duration_scale, "segment_durations_s": durations, "max_raw_knot_step_deg": max(raw_steps, default=0.0), "max_wrap_aware_knot_step_deg": max(wrap_steps, default=0.0), "jtc_analytic_audit": audit_summary(audit)}


def dense_jtc_report(t: np.ndarray, q: np.ndarray, v: np.ndarray, a: np.ndarray, limits: list[dict[str, float]], samples_per_interval: int = 101) -> dict[str, Any]:
    maxima = {name: {"value": 0.0, "joint": None, "interval": None, "global_time_s": None} for name in ("position", "velocity", "acceleration", "jerk")}
    max_endpoint_envelope = 0.0
    max_knot_position_residual = 0.0
    max_knot_velocity_residual = 0.0
    max_knot_acceleration_residual = 0.0
    sampled: dict[str, list[np.ndarray]] = {name: [] for name in ("position", "velocity", "acceleration", "jerk")}
    sampled_t: list[float] = []
    position_min = np.full(q.shape[1], np.inf, dtype=float)
    position_max = np.full(q.shape[1], -np.inf, dtype=float)
    low = np.asarray([item["position_lower_rad"] for item in limits], dtype=float)
    high = np.asarray([item["position_upper_rad"] for item in limits], dtype=float)
    for index in range(len(t) - 1):
        h = float(t[index + 1] - t[index])
        coeff = segment_coefficients(q[index], q[index + 1], v[index], v[index + 1], a[index], a[index + 1], h, "quintic")
        start = evaluate_coefficients(coeff, 0.0)
        end = evaluate_coefficients(coeff, h)
        max_knot_position_residual = max(max_knot_position_residual, float(np.max(np.abs(start["position"] - q[index]))), float(np.max(np.abs(end["position"] - q[index + 1]))))
        max_knot_velocity_residual = max(max_knot_velocity_residual, float(np.max(np.abs(start["velocity"] - v[index]))), float(np.max(np.abs(end["velocity"] - v[index + 1]))))
        max_knot_acceleration_residual = max(max_knot_acceleration_residual, float(np.max(np.abs(start["acceleration"] - a[index]))), float(np.max(np.abs(end["acceleration"] - a[index + 1]))))
        for local_index, local_time in enumerate(np.linspace(0.0, h, samples_per_interval)):
            if index and local_index == 0:
                continue
            values = evaluate_coefficients(coeff, float(local_time))
            global_time = float(t[index] + local_time)
            for name in sampled:
                sampled[name].append(values[name])
            sampled_t.append(global_time)
            position_min = np.minimum(position_min, values["position"])
            position_max = np.maximum(position_max, values["position"])
            for name in maxima:
                absolute = np.abs(values[name])
                joint = int(np.argmax(absolute))
                if float(absolute[joint]) > maxima[name]["value"]:
                    maxima[name] = {"value": float(absolute[joint]), "joint": JOINT_NAMES[joint], "interval": index, "global_time_s": global_time}
            max_endpoint_envelope = max(max_endpoint_envelope, float(np.max(np.maximum(low - values["position"], values["position"] - high).clip(min=0.0))))
    ratios = {"velocity": 0.0, "acceleration": 0.0, "jerk": 0.0}
    for name, key in (("velocity", "max_velocity_rad_s"), ("acceleration", "max_acceleration_rad_s2"), ("jerk", "max_jerk_rad_s3")):
        values = np.asarray(sampled[name], dtype=float)
        ratios[name] = float(np.max(np.abs(values) / np.asarray([item[key] for item in limits], dtype=float))) if values.size else math.inf
    finite = bool(all(np.isfinite(np.asarray(values, dtype=float)).all() for values in sampled.values()) and np.isfinite(position_min).all() and np.isfinite(position_max).all())
    dense_sample_status = bool(finite and max_endpoint_envelope <= 1.0e-12 and max_knot_position_residual <= 1.0e-9 and max_knot_velocity_residual <= 1.0e-9 and max_knot_acceleration_residual <= 1.0e-9 and ratios["velocity"] <= 1.0 + 1.0e-10 and ratios["acceleration"] <= 1.0 + 1.0e-10 and ratios["jerk"] <= 1.0 + 1.0e-10)
    maxima["position"]["min_per_joint_rad"] = position_min.tolist()
    maxima["position"]["max_per_joint_rad"] = position_max.tolist()
    return {
        "samples_per_interval": samples_per_interval,
        "sample_count": len(sampled_t),
        "sample_time_start_s": float(sampled_t[0]) if sampled_t else None,
        "sample_time_end_s": float(sampled_t[-1]) if sampled_t else None,
        "maxima_abs": maxima,
        "max_position_outside_limits_rad": max_endpoint_envelope,
        "normalized_maxima": ratios,
        "knot_reconstruction_residuals": {"position_rad": max_knot_position_residual, "velocity_rad_s": max_knot_velocity_residual, "acceleration_rad_s2": max_knot_acceleration_residual},
        "finite": finite,
        "dense_sample_status": dense_sample_status,
    }


def write_trajectory(path: Path, t: np.ndarray, q: np.ndarray, v: np.ndarray, a: np.ndarray) -> None:
    fields = ["t"] + [f"{joint}_{suffix}" for suffix in ("q", "dq", "ddq") for joint in JOINT_NAMES] + [f"{joint}_jerk" for joint in JOINT_NAMES]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for index, time_s in enumerate(t):
            row: dict[str, Any] = {"t": float(time_s)}
            row.update({f"{joint}_q": float(q[index, joint_index]) for joint_index, joint in enumerate(JOINT_NAMES)})
            row.update({f"{joint}_dq": float(v[index, joint_index]) for joint_index, joint in enumerate(JOINT_NAMES)})
            row.update({f"{joint}_ddq": float(a[index, joint_index]) for joint_index, joint in enumerate(JOINT_NAMES)})
            row.update({f"{joint}_jerk": 0.0 for joint in JOINT_NAMES})
            writer.writerow(row)


def run(node: Any) -> int:
    poses_path = Path(str(node.get_parameter("poses_csv").value)).resolve()
    source_path = Path(str(node.get_parameter("source_q_csv").value)).resolve()
    environment_path = Path(str(node.get_parameter("environment_json").value)).resolve()
    output = Path(str(node.get_parameter("output_dir").value)).resolve()
    include_collision = bool(node.get_parameter("include_collision").value)
    ik_timeout_s = float(node.get_parameter("ik_timeout_s").value)
    if not np.isfinite(ik_timeout_s) or ik_timeout_s < 0.0:
        raise RuntimeError(f"stage5b_invalid_ik_timeout:{ik_timeout_s}")
    poses = read_csv(poses_path)
    q_source = q_rows(source_path)
    environment = read_environment(environment_path)
    if len(poses) != 181 or q_source.shape != (181, 6):
        raise RuntimeError(f"stage5b_authoritative_shape:{len(poses)}/{q_source.shape}")
    moveit, layers, layer_report = build_layers(node, poses, q_source, environment, include_collision, ik_timeout_s)
    path, graph_report = select_path(layers, q_source, pin_endpoints=True)
    limits_path = ROOT / "ros2_moveit_bridge/config/joint_limits_with_jerk.yaml"
    limits, limit_report = load_motion_limits(limits_path, np.asarray(layer_report["joint_lower_rad"], dtype=float), np.asarray(layer_report["joint_upper_rad"], dtype=float))
    source_t, source_q, source_v, source_a = trajectory_arrays(source_path)
    source_continuity = continuity_map(source_q)
    source_analytic = audit_summary(audit_intervals(source_t, source_q, source_v, source_a, "quintic", limits, JOINT_NAMES))
    source_dense = dense_jtc_report(source_t, source_q, source_v, source_a, limits)
    report: dict[str, Any] = {
        "schema_version": "stage5b-offline-repair-v1",
        "status": "UNRESOLVED",
        "scope": "shadow_only_no_gazebo_no_promotion",
        "authoritative_task": {"pose_csv": str(poses_path), "source_q_csv": str(source_path), "waypoint_count": len(poses), "tcp_task_unchanged": True, "position_tolerance_m": POSITION_TOLERANCE_M, "orientation_tolerance_rad": ORIENTATION_TOLERANCE_RAD},
        "source_continuity_map": source_continuity,
        "search": {"method": "per-waypoint multi-seed MoveIt KDL IK + lexicographic minimum-bottleneck graph", "continuity_gate_deg": CONTINUITY_GATE_DEG, "collision_filter_enabled": include_collision, "ik_timeout_s_per_seed": ik_timeout_s, "environment": str(environment_path), "layers": layer_report, "graph": graph_report},
        "limits": {"source": str(limits_path), "by_joint": limit_report},
        "source_jtc_diagnostic": {"analytic": source_analytic, "dense": source_dense, "interpretation": "offline reproduction of the same endpoint-field quintic; not Gazebo feedback"},
        "controller_ground_truth": {"distribution": "ROS 2 Jazzy installed joint_trajectory_controller", "controller_config": str(ROOT / "ros2_moveit_bridge/config/stage5b_gazebo_controllers.yaml"), "command_interfaces": ["position"], "state_interfaces": ["position", "velocity"], "interpolation_method": "splines", "interpolate_from_desired_state": False, "submitted_fields": ["positions", "velocities", "accelerations", "time_from_start"], "spline_order": "quintic", "initial_point_handling": {"first_timestamp_from_start_s": float(source_t[0]), "header_stamp_zero_starts_sampling_clock": True, "before_first_point_interpolation": "not entered when the first sample is exactly at the first point", "trajectory_start_derivatives": "the submitted first-point velocity and acceleration are used"}, "source": "installed /opt/ros/jazzy trajectory.hpp + trajectory.cpp and official Jazzy ros2_controllers source"},
        "promotion": "NO_PROMOTION",
        "gazebo_replay": "PROHIBITED_IN_THIS_RUN",
    }
    report["claim_fence"] = {"collision_label": "adaptive_discrete_interpolation", "exact_articulated_self_ccd": "not_available", "clearance": "not_available", "nominal_world_feedback": "not_measured_this_run", "hardware_safety": "UNVERIFIED"}
    if not path:
        source_path_records = [{"q": row.tolist()} for row in q_source]
        source_q0, source_v0, source_a0, source_t0, source_timing = rebuild_zero_knot_quintic(source_path_records, limits)
        source_zero_dense = dense_jtc_report(source_t0, source_q0, source_v0, source_a0, limits)
        source_zero_path = output / "STAGE5B_SOURCE_ZERO_KNOT_JTC_DIAGNOSTIC.csv"
        write_trajectory(source_zero_path, source_t0, source_q0, source_v0, source_a0)
        report["source_zero_knot_jtc_diagnostic"] = {"trajectory": {"timestamps_strictly_increasing": bool(np.all(np.diff(source_t0) > 0.0)), "duration_s": float(source_t0[-1]), "max_wrap_aware_knot_step_deg": source_timing["max_wrap_aware_knot_step_deg"], "max_raw_knot_step_deg": source_timing["max_raw_knot_step_deg"], "timing": source_timing, "dense_jtc": source_zero_dense}, "interpretation": "diagnostic only: original q retained its continuity failure; zero knot derivatives and regenerated raw-command timing isolate the JTC spline cause"}
        report["source_zero_knot_jtc_diagnostic"]["trajectory"]["path"] = str(source_zero_path)
        report["offline_gates"] = {"ik_branch_continuity": False, "q_continuity_max_le_20deg": False, "timestamps": bool(np.all(np.diff(source_t0) > 0.0)), "jtc_quintic_zero_knot_diagnostic": bool(source_zero_dense["dense_sample_status"]), "gazebo_feedback_measured": False, "promotion": "NO_PROMOTION"}
        report["stage5b_decision"] = {"OFFLINE_JTC_TRAJECTORY_CAUSE_FIXED": "PASS_DIAGNOSTIC_ONLY" if source_zero_dense["dense_sample_status"] else "UNRESOLVED", "READY_FOR_GAZEBO_REPLAY": "NO", "reason": "No continuity-qualified 181-point trajectory exists under the unchanged task and 20 degree gate; the zero-knot JTC result is not a replay candidate."}
        write_json(output / "STAGE5B_OFFLINE_REPAIR_REPORT.json", report)
        print(json.dumps({"status": report["status"], "graph": graph_report}, ensure_ascii=False), flush=True)
        return 2
    q, v, a, t, timing_report = rebuild_zero_knot_quintic(path, limits)
    dense = dense_jtc_report(t, q, v, a, limits)
    analytic = timing_report["jtc_analytic_audit"]
    report["trajectory"] = {"candidate_points": len(q), "timestamps_strictly_increasing": bool(np.all(np.diff(t) > 0.0)), "duration_s": float(t[-1]), "q_dq_ddq_finite": bool(np.isfinite(q).all() and np.isfinite(v).all() and np.isfinite(a).all()), "all_knot_velocities_zero": bool(np.all(v == 0.0)), "all_knot_accelerations_zero": bool(np.all(a == 0.0)), "timing": timing_report, "analytic_jtc_audit_status": analytic["status"], "dense_jtc": dense}
    report["status"] = "READY_FOR_GAZEBO_REPLAY" if graph_report.get("status") == "PASS" and analytic["status"].get("position") == "passed" and analytic["status"].get("velocity") == "passed" and analytic["status"].get("acceleration") == "passed" and analytic["status"].get("jerk") == "passed" and dense["dense_sample_status"] else "UNRESOLVED"
    report["offline_gates"] = {"ik_branch_continuity": graph_report.get("status") == "PASS", "q_continuity_max_le_20deg": graph_report.get("max_wrap_aware_step_deg", math.inf) <= CONTINUITY_GATE_DEG, "timestamps": bool(np.all(np.diff(t) > 0.0)), "jtc_quintic_analytic": all(value == "passed" for value in analytic["status"].values()), "jtc_dense_sample": bool(dense["dense_sample_status"]), "gazebo_feedback_measured": False, "promotion": "NO_PROMOTION"}
    report["stage5b_decision"] = {"OFFLINE_JTC_TRAJECTORY_CAUSE_FIXED": "PASS" if dense["dense_sample_status"] else "UNRESOLVED", "READY_FOR_GAZEBO_REPLAY": "YES" if report["status"] == "READY_FOR_GAZEBO_REPLAY" else "NO", "reason": "candidate status follows the complete unchanged-task continuity and JTC gates"}
    output.mkdir(parents=True, exist_ok=True)
    trajectory_path = output / "STAGE5B_OFFLINE_REPAIRED_TRAJECTORY.csv"
    write_trajectory(trajectory_path, t, q, v, a)
    report["trajectory"]["candidate_path"] = str(trajectory_path)
    report["trajectory"]["candidate_source"] = "MoveIt KDL IK branch graph over unchanged 181 authoritative TCP poses"
    write_json(output / "STAGE5B_OFFLINE_REPAIR_REPORT.json", report)
    print(json.dumps({"status": report["status"], "candidate": str(trajectory_path), "duration_s": float(t[-1]), "max_step_deg": graph_report.get("max_wrap_aware_step_deg"), "jtc_dense": dense["dense_sample_status"]}, ensure_ascii=False), flush=True)
    return 0 if report["status"] == "READY_FOR_GAZEBO_REPLAY" else 2


def main() -> int:  # pragma: no cover - ROS runtime
    import rclpy
    from rclpy.node import Node

    rclpy.init()
    node = Node("stage5b_offline_repair_wrapper")
    node.declare_parameter("poses_csv", "")
    node.declare_parameter("source_q_csv", "")
    node.declare_parameter("environment_json", "")
    node.declare_parameter("output_dir", "")
    node.declare_parameter("include_collision", False)
    node.declare_parameter("ik_timeout_s", 0.02)
    try:
        return run(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
