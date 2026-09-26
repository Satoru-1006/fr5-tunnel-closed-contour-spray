"""Search a bounded task-space detour for the Stage 5B branch cut.

This is an experimental companion to ``stage5b_branch_repair.py``.  Bounded
zero-endpoint Cartesian position/orientation detours are tested against a
multi-start MoveIt KDL IK graph.  The graph uses hard URDF joint limits and
the x=-0.50 m Stage5AC scene only as a shadow collision screen.  No result is
promoted or written over any earlier artifact.
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
sys.path.insert(0, str(ROOT / "tools"))
from stage5b_branch_repair import (  # noqa: E402
    JOINTS,
    GROUP,
    EE,
    env_objects,
    interpolate_pose,
    matrix_of,
    pose_error,
    pose_from_arrays,
    read_csv,
    shortest_delta,
    write_json,
)


def unique_seeds(seeds: list[np.ndarray], tolerance: float = 1.0e-10) -> list[np.ndarray]:
    result: list[np.ndarray] = []
    for seed in seeds:
        value = np.asarray(seed, dtype=float)
        if not np.isfinite(value).all():
            continue
        if not any(float(np.max(np.abs(value - old))) <= tolerance for old in result):
            result.append(value.copy())
    return result


def seed_variants(seed: np.ndarray) -> list[np.ndarray]:
    requested = [seed.copy()]
    for joint in range(6):
        for sign in (-1.0, 1.0):
            value = seed.copy()
            value[joint] += sign * math.pi
            requested.append(value)
    for joints in ((0, 2), (0, 4), (2, 4), (0, 2, 4)):
        for sign in (-1.0, 1.0):
            value = seed.copy()
            for joint in joints:
                value[joint] += sign * math.pi
            requested.append(value)
    return unique_seeds(requested)


def orientation_bump(base_quat: np.ndarray, axis: np.ndarray, fraction: float, amplitude_rad: float, profile: str, world_left: bool) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    if profile == "sin1":
        phase = math.sin(math.pi * fraction)
    elif profile == "sin2":
        phase = math.sin(2.0 * math.pi * fraction)
    elif profile == "sin3":
        phase = math.sin(3.0 * math.pi * fraction)
    elif profile == "antisymmetric":
        phase = math.sin(math.pi * fraction) * math.cos(2.0 * math.pi * fraction)
    else:
        raise ValueError(f"unknown_profile:{profile}")
    base = Rotation.from_quat(base_quat)
    bump = Rotation.from_rotvec(np.asarray(axis, dtype=float) * amplitude_rad * phase)
    result = bump * base if world_left else base * bump
    return result.as_quat()


def bump_phase(fraction: float, profile: str) -> float:
    if profile == "sin1":
        return math.sin(math.pi * fraction)
    if profile == "sin2":
        return math.sin(2.0 * math.pi * fraction)
    if profile == "sin3":
        return math.sin(3.0 * math.pi * fraction)
    if profile == "antisymmetric":
        return math.sin(math.pi * fraction) * math.cos(2.0 * math.pi * fraction)
    raise ValueError(f"unknown_profile:{profile}")


def position_bump(base_position: np.ndarray, axis: np.ndarray, fraction: float, amplitude_m: float, profile: str) -> np.ndarray:
    return np.asarray(base_position, dtype=float) + np.asarray(axis, dtype=float) * amplitude_m * bump_phase(fraction, profile)


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
    max_step_deg = float(node.get_parameter("max_step_deg").value)
    fix_endpoints = bool(node.get_parameter("fix_endpoints").value)
    amplitudes = [float(x) for x in str(node.get_parameter("amplitudes_deg").value).split(",") if str(x).strip()]
    orientation_axes_requested = [x.strip() for x in str(node.get_parameter("orientation_axes").value).split(",") if x.strip()]
    position_axes_requested = [x.strip() for x in str(node.get_parameter("position_axes").value).split(",") if x.strip()]
    position_amplitudes = [float(x) for x in str(node.get_parameter("position_amplitudes_mm").value).split(",") if str(x).strip()]
    profiles = [x.strip() for x in str(node.get_parameter("profiles").value).split(",") if x.strip()]
    axis_map = {"x": np.array([1.0, 0.0, 0.0]), "y": np.array([0.0, 1.0, 0.0]), "z": np.array([0.0, 0.0, 1.0])}
    orientation_axes = [(name, axis_map[name]) for name in orientation_axes_requested if name in axis_map]
    position_axes = [(name, axis_map[name]) for name in position_axes_requested if name in axis_map]
    if not position_axes:
        position_axes = [("none", np.zeros(3))]
    if not (0 <= start < end) or subdivisions < 2 or not amplitudes or not profiles or not orientation_axes or not position_amplitudes:
        raise RuntimeError("stage5b_invalid_graph_parameters")
    poses = read_csv(poses_path)
    source_rows = read_csv(source_path)
    q_source = np.asarray([[float(row[f"j{i}_q"]) for i in range(1, 7)] for row in source_rows], dtype=float)
    if len(poses) != len(source_rows) or len(poses) <= end:
        raise RuntimeError("stage5b_input_shape_mismatch")
    environment = json.loads(environment_path.read_text(encoding="utf-8"))

    moveit = MoveItPy(node_name="stage5b_branch_graph_search_moveit")
    model = moveit.get_robot_model()
    group = model.get_joint_model_group(GROUP)
    if group is None or list(group.active_joint_model_names) != JOINTS:
        raise RuntimeError("stage5b_joint_mapping_mismatch")
    lows, highs = [], []
    for bound in group.active_joint_model_bounds:
        bound = bound[0] if isinstance(bound, (list, tuple)) else bound
        lows.append(float(bound.min_position))
        highs.append(float(bound.max_position))
    lows = np.asarray(lows)
    highs = np.asarray(highs)
    psm = moveit.get_planning_scene_monitor()
    with psm.read_write() as scene:
        scene.remove_all_collision_objects()
        for obj in env_objects(environment):
            scene.apply_collision_object(obj)
    request = CollisionRequest()
    request.joint_model_group_name = GROUP
    request.contacts = False

    # The two endpoint solutions plus deterministic pi-shift families cover
    # the branch labels already observed in the Stage5AC attempts.  Each
    # graph layer additionally seeds from every valid predecessor candidate.
    fixed_seeds = seed_variants(q_source[start]) + seed_variants(q_source[end])
    axes = [("x", np.array([1.0, 0.0, 0.0])), ("y", np.array([0.0, 1.0, 0.0])), ("z", np.array([0.0, 0.0, 1.0]))]
    summaries: list[dict[str, Any]] = []
    selected_result: dict[str, Any] | None = None

    for position_axis_name, position_axis in position_axes:
        for position_amplitude_mm in position_amplitudes:
            for axis_name, axis in axes:
                if axis_name not in orientation_axes_requested:
                    continue
                for sign in (1.0, -1.0):
                    for amplitude_deg in amplitudes:
                        profile = profiles[0]
                        world_left = False
                        amplitude_rad = math.radians(sign * amplitude_deg)
                        layers: list[list[dict[str, Any]]] = []
                        previous_candidates: list[np.ndarray] = []
                        for sample_index in range(subdivisions + 1):
                            fraction = float(sample_index) / float(subdivisions)
                            position, base_quat = interpolate_pose(poses[start], poses[end], fraction)
                            position = position_bump(position, position_axis, fraction, position_amplitude_mm * 1.0e-3, profile)
                            quat = orientation_bump(base_quat, axis, fraction, amplitude_rad, profile, world_left)
                            seeds = fixed_seeds + previous_candidates
                            layer: list[dict[str, Any]] = []
                            for seed_index, seed in enumerate(unique_seeds(seeds)):
                                state = RobotState(model)
                                state.set_joint_group_positions(GROUP, seed.tolist())
                                state.update()
                                solved = bool(state.set_from_ik(GROUP, pose_from_arrays(position, quat), EE, 0.0))
                                if not solved:
                                    continue
                                state.update()
                                q = np.asarray(state.get_joint_group_positions(GROUP), dtype=float).copy()
                                fk = matrix_of(state.get_global_link_transform(EE))
                                pos_error, rot_error = pose_error(fk, position, quat)
                                self_result = CollisionResult()
                                with psm.read_only() as scene:
                                    scene.check_self_collision(request, self_result, state)
                                    full_result = CollisionResult()
                                    scene.check_collision(request, full_result, state)
                                valid = bool(
                                    np.isfinite(q).all()
                                    and np.all(q >= lows)
                                    and np.all(q <= highs)
                                    and pos_error <= 1.0e-4
                                    and rot_error <= 1.0e-3
                                    and not self_result.collision
                                    and not full_result.collision
                                )
                                if not valid:
                                    continue
                                if any(float(np.max(np.abs(q - item["q"]))) <= 1.0e-5 for item in layer):
                                    continue
                                layer.append({"q": q.tolist(), "seed_index": seed_index, "position_error_m": pos_error, "orientation_error_rad": rot_error})
                            layer.sort(key=lambda item: tuple(item["q"]))
                            layers.append(layer)
                            previous_candidates = [np.asarray(item["q"], dtype=float) for item in layer]

                        if fix_endpoints and layers:
                            # The local repair must splice into the frozen
                            # 181-point path.  Do not accept a smooth path
                            # that silently starts on a different IK family.
                            start_q = q_source[start]
                            end_q = q_source[end]
                            layers[0] = [item for item in layers[0] if float(np.max(np.abs(np.asarray(item["q"]) - start_q))) <= 1.0e-4]
                            layers[-1] = [item for item in layers[-1] if float(np.max(np.abs(np.asarray(item["q"]) - end_q))) <= 1.0e-4]

                        paths: list[tuple[float, list[dict[str, Any]]]] = [(0.0, [item]) for item in layers[0]] if layers and layers[0] else []
                        first_unreachable = None if paths else 0
                        bottleneck_paths: list[tuple[float, list[dict[str, Any]]]] = paths
                        for layer_index in range(1, len(layers)):
                            targets = layers[layer_index]
                            next_paths: list[tuple[float, list[dict[str, Any]]]] = []
                            next_bottleneck: list[tuple[float, list[dict[str, Any]]]] = []
                            for target in targets:
                                q_target = np.asarray(target["q"], dtype=float)
                                choices = []
                                bottleneck_choices = []
                                for cost, path in paths:
                                    q_source_last = np.asarray(path[-1]["q"], dtype=float)
                                    delta = q_target - q_source_last
                                    step = float(np.rad2deg(np.max(np.abs(delta))))
                                    if step <= max_step_deg:
                                        choices.append((cost + float(np.linalg.norm(delta)), path + [target]))
                                for bottleneck, path in bottleneck_paths:
                                    q_source_last = np.asarray(path[-1]["q"], dtype=float)
                                    delta = q_target - q_source_last
                                    step = float(np.rad2deg(np.max(np.abs(delta))))
                                    bottleneck_choices.append((max(bottleneck, step), path + [target]))
                                if choices:
                                    next_paths.append(min(choices, key=lambda item: (item[0], tuple(tuple(x["q"]) for x in item[1]))))
                                if bottleneck_choices:
                                    next_bottleneck.append(min(bottleneck_choices, key=lambda item: (item[0], tuple(tuple(x["q"]) for x in item[1]))))
                            if not next_paths:
                                first_unreachable = layer_index
                                paths = []
                            else:
                                paths = next_paths
                            bottleneck_paths = next_bottleneck
                            if not bottleneck_paths:
                                break
                        selected = min(paths, key=lambda item: (item[0], tuple(tuple(x["q"]) for x in item[1])))[1] if paths and first_unreachable is None else []
                        best_bottleneck = min(bottleneck_paths, key=lambda item: (item[0], tuple(tuple(x["q"]) for x in item[1])))[0] if bottleneck_paths else None
                        best_bottleneck_path = min(bottleneck_paths, key=lambda item: (item[0], tuple(tuple(x["q"]) for x in item[1])))[1] if bottleneck_paths else []
                        row = {
                            "position_axis": position_axis_name,
                            "position_amplitude_mm": position_amplitude_mm,
                            "axis": axis_name,
                            "sign": sign,
                            "amplitude_deg": amplitude_deg,
                            "profile": profile,
                            "world_left": world_left,
                            "sample_count": subdivisions + 1,
                            "layer_counts": [len(layer) for layer in layers],
                            "full_path_under_step_gate": bool(selected),
                            "first_unreachable_sample": first_unreachable,
                            "minimum_bottleneck_step_deg": best_bottleneck,
                            "valid_candidate_count": sum(len(layer) for layer in layers),
                        }
                        if selected:
                            steps = [float(np.rad2deg(np.max(np.abs(np.asarray(selected[i]["q"]) - np.asarray(selected[i - 1]["q"])))) ) for i in range(1, len(selected))]
                            row["selected_max_step_deg"] = max(steps, default=0.0)
                            row["selected_path"] = selected
                        if best_bottleneck_path:
                            row["bottleneck_path"] = best_bottleneck_path
                        summaries.append(row)
                        if selected and (selected_result is None or row["selected_max_step_deg"] < selected_result["selected_max_step_deg"]):
                            selected_result = row

    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": "stage5b-branch-graph-search-v1",
        "status": "PASS" if selected_result is not None else "UNRESOLVED",
        "acceptance_step_gate_deg": max_step_deg,
        "fixed_source_endpoints": fix_endpoints,
        "segment": {"start_waypoint": start, "end_waypoint": end, "subdivisions": subdivisions},
        "position_detour": {"axes": position_axes_requested, "amplitudes_mm": position_amplitudes},
        "orientation_detour": {"axes": orientation_axes_requested, "amplitudes_deg": amplitudes, "profiles": profiles},
        "task_space": "linear_position_slerp_orientation_plus_zero_endpoint_bounded_position_and_rotation_bump",
        "solver": "MoveItPy KDL KinematicsBase::getPositionIK with hard URDF limits",
        "scene": str(environment_path),
        "experiments": summaries,
        "selected": selected_result,
        "interpretation": "shadow diagnostic; endpoint task pose is restored; any nonzero bump changes the interior task path and requires full task/collision/dynamics validation before use",
    }
    write_json(output / "STAGE5B_BRANCH_GRAPH_SEARCH.json", summary)
    if selected_result is not None:
        with (output / "STAGE5B_SELECTED_LOCAL_REPAIR.csv").open("w", encoding="utf-8", newline="") as handle:
            fields = ["sample_index", "fraction", "axis", "sign", "amplitude_deg", "profile", "world_left", "position_error_m", "orientation_error_rad"] + JOINTS
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for index, item in enumerate(selected_result["selected_path"]):
                q = item["q"]
                writer.writerow({"sample_index": index, "fraction": index / subdivisions, "axis": selected_result["axis"], "sign": selected_result["sign"], "amplitude_deg": selected_result["amplitude_deg"], "profile": selected_result["profile"], "world_left": selected_result["world_left"], "position_error_m": item["position_error_m"], "orientation_error_rad": item["orientation_error_rad"], **{JOINTS[i]: q[i] for i in range(6)}})
    print(json.dumps({"status": summary["status"], "experiment_count": len(summaries), "selected": None if selected_result is None else {k: selected_result[k] for k in ("axis", "sign", "amplitude_deg", "profile", "world_left", "selected_max_step_deg")}}, ensure_ascii=False), flush=True)
    return 0 if selected_result is not None else 2


def main() -> int:  # pragma: no cover - ROS runtime
    import rclpy
    from rclpy.node import Node

    rclpy.init()
    node = Node("stage5b_branch_graph_search_wrapper")
    node.declare_parameter("poses_csv", "")
    node.declare_parameter("source_q_csv", "")
    node.declare_parameter("environment_json", "")
    node.declare_parameter("output_dir", "")
    node.declare_parameter("start_waypoint", 66)
    node.declare_parameter("end_waypoint", 67)
    node.declare_parameter("subdivisions", 32)
    node.declare_parameter("max_step_deg", 20.0)
    node.declare_parameter("fix_endpoints", True)
    node.declare_parameter("amplitudes_deg", "5,10,15")
    node.declare_parameter("orientation_axes", "x,y,z")
    node.declare_parameter("position_axes", "none")
    node.declare_parameter("position_amplitudes_mm", "0")
    node.declare_parameter("profiles", "sin1,sin2,antisymmetric")
    try:
        return run(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
