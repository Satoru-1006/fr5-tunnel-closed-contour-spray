"""Bounded MoveIt2/FCL PlanningScene propagation smoke for FR5 P2-B3-C3."""

from __future__ import annotations

import argparse
import csv
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.physical_uncertainty import (  # noqa: E402
    BASE_FRAME,
    apply_registration_to_trajectory,
    registration_transform,
    quaternion_xyzw_to_matrix,
    six_dof_directional_smoke,
    transform_collision_object_message,
    transform_pose_array_xyzw,
    transform_surface_normals,
)


GROUP = "fairino5_v6_group"
JOINTS = [f"j{index}" for index in range(1, 7)]
TCP_LINK = "spray_tcp_link"
MAX_NATIVE_CASES = 16
COLLISION_METHOD = "adaptive_discrete_interpolation"
MAX_JOINT_STEP_RAD = math.radians(0.5)
_MOVEIT_KEEPALIVE: Any | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def target_identity(c1_result: dict[str, Any], relative_path: str) -> str | None:
    hashes = c1_result.get("input_identity_sha256", {})
    target = relative_path.replace("\\", "/")
    matches = [str(value) for key, value in hashes.items() if str(key).replace("\\", "/") == target]
    if len(set(matches)) > 1:
        raise RuntimeError("ambiguous_c1_target_identity")
    return matches[0] if matches else None


def read_c1_inputs(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, str]]:
    q_rows = load_rows(args.trajectory_csv)
    target_rows = load_rows(args.target_csv)
    fk_rows = load_rows(args.c1_fk_trace)
    c1_result = json.loads(args.c1_result.read_text(encoding="utf-8"))
    if c1_result.get("PROJECT") != "FAIRINO_FR5" or c1_result.get("STAGE") != "P2-B3-C1":
        raise RuntimeError("c1_result_project_or_stage_identity_mismatch")
    if not (len(q_rows) == len(target_rows) == len(fk_rows) == 181):
        raise RuntimeError("c1_input_waypoint_count_must_be_181")
    times = np.asarray([float(row["t"]) for row in q_rows], dtype=np.float64)
    q = np.asarray([[float(row[f"j{i}_q"]) for i in range(1, 7)] for row in q_rows], dtype=np.float64)
    target_poses = np.asarray(
        [[float(row[key]) for key in ("x", "y", "z", "qx", "qy", "qz", "qw")] for row in target_rows],
        dtype=np.float64,
    )
    normals = np.asarray([[float(row[key]) for key in ("nx", "ny", "nz")] for row in target_rows], dtype=np.float64)
    fk_times = np.asarray([float(row["t"]) for row in fk_rows], dtype=np.float64)
    fk_position = np.asarray([[float(row[key]) for key in ("actual_tcp_x", "actual_tcp_y", "actual_tcp_z")] for row in fk_rows], dtype=np.float64)
    fk_tool_z = np.asarray([[float(row[key]) for key in ("tool_z_x", "tool_z_y", "tool_z_z")] for row in fk_rows], dtype=np.float64)
    if not all(np.isfinite(value).all() for value in (times, q, target_poses, normals, fk_times, fk_position, fk_tool_z)):
        raise RuntimeError("c1_input_contains_nonfinite_values")
    if np.any(np.diff(times) <= 0.0) or not np.array_equal(times, fk_times):
        raise RuntimeError("c1_timebase_or_row_order_mismatch")
    q_sha = sha256_file(args.trajectory_csv)
    fk_sha = sha256_file(args.c1_fk_trace)
    target_sha = sha256_file(args.target_csv)
    urdf_sha = sha256_file(args.urdf)
    if q_sha != c1_result.get("C1_POST_RUCKIG_NOMINAL", {}).get("sha256"):
        raise RuntimeError("c1_post_ruckig_identity_mismatch")
    if fk_sha != c1_result.get("candidate_input_identity_sha256", {}).get("candidate_fk_trace"):
        raise RuntimeError("c1_fk_trace_identity_mismatch")
    expected_target = target_identity(c1_result, "outputs/internal_wiper_moveit_inputs/open_arch_tcp_poses_base_link.csv")
    if expected_target and expected_target != target_sha:
        raise RuntimeError("c1_target_pose_identity_mismatch")
    expected_urdf = c1_result.get("C1_DERIVED_REFERENCE_URDF_SHA256")
    if expected_urdf and expected_urdf != urdf_sha:
        raise RuntimeError("c1_urdf_identity_mismatch")
    identities = {
        "c1_post_ruckig_sha256": q_sha,
        "c1_fk_trace_sha256": fk_sha,
        "target_tcp_csv_sha256": target_sha,
        "derived_reference_urdf_sha256": urdf_sha,
        "srdf_sha256": sha256_file(args.srdf),
        "joint_limits_sha256": sha256_file(args.joint_limits),
        "c1_result_sha256": sha256_file(args.c1_result),
    }
    return times, q, target_poses, normals, np.column_stack((fk_position, fk_tool_z)), identities


def pose_messages(pose_values: np.ndarray) -> list[Any]:
    from geometry_msgs.msg import Pose

    result = []
    for row in pose_values:
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = map(float, row[:3])
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = map(float, row[3:7])
        result.append(pose)
    return result


def collision_samples(q: np.ndarray, times: np.ndarray) -> Iterable[tuple[np.ndarray, float]]:
    yield np.array(q[0], copy=True), float(times[0])
    for index, (left, right) in enumerate(zip(q[:-1], q[1:])):
        steps = max(1, int(math.ceil(float(np.max(np.abs(right - left))) / MAX_JOINT_STEP_RAD)))
        for substep in range(1, steps + 1):
            fraction = substep / steps
            yield left + fraction * (right - left), float(times[index] + fraction * (times[index + 1] - times[index]))


def apply_world(moveit: Any, objects: list[Any]) -> list[bool]:
    monitor = moveit.get_planning_scene_monitor()
    results: list[bool] = []
    with monitor.read_write() as scene:
        scene.remove_all_collision_objects()
        for obj in objects:
            applied = scene.apply_collision_object(obj)
            results.append(applied is not False)
    return results


def evaluate_scene(moveit: Any, q: np.ndarray, times: np.ndarray, *, compare_fk: tuple[np.ndarray, np.ndarray] | None = None) -> dict[str, Any]:
    from moveit.core.collision_detection import CollisionRequest, CollisionResult
    from moveit.core.robot_state import RobotState

    model = moveit.get_robot_model()
    group = model.get_joint_model_group(GROUP)
    if group is None or list(group.active_joint_model_names) != JOINTS:
        raise RuntimeError("native_fr5_joint_order_mismatch")
    state = RobotState(model)
    request = CollisionRequest()
    request.joint_model_group_name = GROUP
    request.contacts = True
    request.distance = True
    request.max_contacts = 1
    request.max_contacts_per_pair = 1
    full_collisions = 0
    self_collisions = 0
    distance_values: list[float] = []
    contact_point_total = 0
    maximum_contacts_in_one_sample = 0
    sample_count = 0
    monitor = moveit.get_planning_scene_monitor()
    with monitor.read_only() as scene:
        for q_sample, _time_s in collision_samples(q, times):
            state.set_joint_group_positions(GROUP, q_sample.tolist())
            state.update()
            full_result = CollisionResult()
            self_result = CollisionResult()
            scene.check_collision(request, full_result, state)
            scene.check_self_collision(request, self_result, state)
            full_collisions += int(bool(full_result.collision))
            self_collisions += int(bool(self_result.collision))
            contact_point_total += int(full_result.contact_count)
            maximum_contacts_in_one_sample = max(maximum_contacts_in_one_sample, int(full_result.contact_count))
            raw_distance = float(full_result.distance)
            if math.isfinite(raw_distance):
                distance_values.append(raw_distance)
            elif math.isnan(raw_distance):
                raise RuntimeError("native_fcl_distance_returned_nan")
            sample_count += 1
    if compare_fk is not None:
        positions: list[np.ndarray] = []
        tool_axes: list[np.ndarray] = []
        for state_q in q:
            state.set_joint_group_positions(GROUP, state_q.tolist())
            state.update()
            raw_transform = state.get_global_link_transform(TCP_LINK)
            transform = np.asarray(raw_transform.matrix() if hasattr(raw_transform, "matrix") else raw_transform, dtype=np.float64)
            if transform.shape != (4, 4) or not np.isfinite(transform).all():
                raise RuntimeError("native_fk_transform_invalid")
            positions.append(transform[:3, 3].copy())
            tool_axes.append(transform[:3, 2].copy())
        fk_position_array = np.asarray(positions, dtype=np.float64)
        fk_tool_z_array = np.asarray(tool_axes, dtype=np.float64)
        expected_position, expected_tool_z = compare_fk
        position_delta = np.linalg.norm(fk_position_array - expected_position, axis=1)
        tool_z_delta = np.linalg.norm(fk_tool_z_array - expected_tool_z, axis=1)
        fk_match = bool(float(np.max(position_delta)) <= 1e-6 and float(np.max(tool_z_delta)) <= 1e-6)
        fk_report: dict[str, Any] = {
            "status": "PASS" if fk_match else "FAIL",
            "waypoints_compared": len(q),
            "max_position_delta_m": float(np.max(position_delta)),
            "max_tool_z_delta": float(np.max(tool_z_delta)),
            "tolerance_position_m": 1e-6,
            "tolerance_tool_z": 1e-6,
        }
    else:
        fk_report = {"status": "NOT_RUN", "waypoints_compared": 0}
    distances_available = len(distance_values) == sample_count
    return {
        "sample_count": sample_count,
        "collision_method": COLLISION_METHOD,
        "maximum_joint_interpolation_step_rad": MAX_JOINT_STEP_RAD,
        "full_scene_collision_sample_count": full_collisions,
        "self_collision_sample_count": self_collisions,
        "contact_point_total": contact_point_total,
        "maximum_contacts_in_one_sample": maximum_contacts_in_one_sample,
        "minimum_reported_fcl_distance_m": float(min(distance_values)) if distances_available and distance_values else None,
        "fcl_distance_status": "AVAILABLE" if distances_available and distance_values else "NOT_AVAILABLE",
        "fcl_distance_interpretation": "MoveIt2 CollisionResult.distance from the active PlanningScene; sampled full-scene distance, not a continuous or physical safety clearance claim",
        "fk_regression": fk_report,
    }


def diagnose_world_collision_objects(moveit: Any, q: np.ndarray, times: np.ndarray, objects: list[Any]) -> dict[str, Any]:
    """Identify world objects that collide at three frozen nominal waypoints.

    This bounded diagnostic uses one object and one saved waypoint per query;
    it does not change the C1 trajectory or the 13-case smoke acceptance set.
    """
    indices = sorted({0, len(q) // 2, len(q) - 1})
    colliding_objects: list[dict[str, Any]] = []
    for obj in objects:
        apply_world(moveit, [obj])
        colliding_indices: list[int] = []
        for index in indices:
            metrics = evaluate_scene(moveit, q[index : index + 1], times[index : index + 1])
            if metrics["full_scene_collision_sample_count"]:
                colliding_indices.append(index)
        if colliding_indices:
            colliding_objects.append({"object_id": str(obj.id), "probe_waypoint_indices": colliding_indices})
    apply_world(moveit, objects)
    return {
        "status": "COMPLETED",
        "method": "single_world_object_vs_frozen_nominal_waypoint_probe",
        "tested_object_count": len(objects),
        "tested_waypoint_indices": indices,
        "colliding_objects": colliding_objects,
        "non_colliding_object_count": len(objects) - len(colliding_objects),
        "interpretation": "Model-only identification at three representative frozen waypoints; not physical contact localization or continuous proof.",
    }


def recover_applied_transform(nominal_object: Any, transformed_object: Any) -> np.ndarray:
    """Recover the left-applied SE(3) delta from one geometry pose pair."""
    if not nominal_object.primitive_poses or not transformed_object.primitive_poses:
        raise RuntimeError("scene_object_pose_readback_missing")
    nominal = nominal_object.primitive_poses[0]
    transformed = transformed_object.primitive_poses[0]
    p0 = np.asarray([nominal.position.x, nominal.position.y, nominal.position.z], dtype=np.float64)
    p1 = np.asarray([transformed.position.x, transformed.position.y, transformed.position.z], dtype=np.float64)
    q0 = np.asarray([nominal.orientation.x, nominal.orientation.y, nominal.orientation.z, nominal.orientation.w], dtype=np.float64)
    q1 = np.asarray([transformed.orientation.x, transformed.orientation.y, transformed.orientation.z, transformed.orientation.w], dtype=np.float64)
    rotation = quaternion_xyzw_to_matrix(q1) @ quaternion_xyzw_to_matrix(q0).T
    recovered = np.eye(4, dtype=np.float64)
    recovered[:3, :3] = rotation
    recovered[:3, 3] = p1 - rotation @ p0
    return recovered


def run(args: argparse.Namespace) -> dict[str, Any]:
    from moveit.planning import MoveItPy
    from moveit_configs_utils import MoveItConfigsBuilder
    import rclpy
    from ros2_moveit_bridge.plan_closed_contour_moveit import build_wall_collision_objects

    times, q, target_pose_values, normals, frozen_fk, identities = read_c1_inputs(args)
    nominal_pose_refs = apply_registration_to_trajectory(target_pose_values, normals, q, times, np.eye(4))
    base_objects = build_wall_collision_objects(
        target_poses=pose_messages(target_pose_values),
        target_normals=normals,
        stand_off=0.18,
        frame_id=BASE_FRAME,
        wall_thickness=0.025,
        y_thickness=0.08,
        segment_stride=4,
        include_bottom_closure=False,
        open_path=True,
        tcp_points_to_wall=False,
        include_tunnel_floor=True,
        tunnel_floor_z=-0.20,
    )
    if not base_objects or any(obj.header.frame_id != BASE_FRAME for obj in base_objects):
        raise RuntimeError("native_scene_builder_returned_no_base_frame_objects")
    identities_in_world = [str(obj.id) for obj in base_objects]
    if len(set(identities_in_world)) != len(identities_in_world):
        raise RuntimeError("native_scene_object_ids_not_unique")

    moveit_config = (
        MoveItConfigsBuilder("fairino5_v6_robot", package_name="fairino5_v6_moveit2_config")
        .robot_description(file_path=str(args.urdf))
        .robot_description_semantic(file_path=str(args.srdf))
        .joint_limits(file_path=str(args.joint_limits))
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )
    config_dict = moveit_config.to_dict()
    # MoveItCpp expects a nested pipeline map. MoveItConfigsBuilder's top-level
    # dictionary uses a list here, while the repository's native launch files
    # explicitly supply this map as a second parameter source.
    config_dict["planning_pipelines"] = {
        "pipeline_names": ["ompl"],
        "ompl": copy.deepcopy(config_dict.get("ompl", {})),
    }
    rclpy.init(args=None)
    global _MOVEIT_KEEPALIVE
    _MOVEIT_KEEPALIVE = MoveItPy(node_name="p2b3_c3_native_smoke", config_dict=config_dict)
    moveit = _MOVEIT_KEEPALIVE
    collision_object_diagnostic = diagnose_world_collision_objects(moveit, q, times, base_objects)
    cases = [{
            "case_id": "registration_zero",
            "family": "zero",
            "translation_base_m": [0.0, 0.0, 0.0],
            "rotation_vector_base_rad": [0.0, 0.0, 0.0],
            "amplitude": 0.0,
            "units": "1",
            "interpretation": "zero identity control",
        }] + six_dof_directional_smoke()
    if len(cases) > MAX_NATIVE_CASES:
        raise RuntimeError("native_smoke_case_cap_exceeded")
    case_results: list[dict[str, Any]] = []
    baseline_result: dict[str, Any] | None = None
    for case in cases:
        delta = registration_transform(case["translation_base_m"], case["rotation_vector_base_rad"])
        trajectory = apply_registration_to_trajectory(target_pose_values, normals, q, times, delta)
        if not np.array_equal(trajectory["joint_states"], q) or not np.array_equal(trajectory["timestamps_s"], times):
            raise RuntimeError("registration_mutated_frozen_robot_trajectory")
        moved_objects = [transform_collision_object_message(obj, delta) for obj in base_objects]
        realized_delta = recover_applied_transform(base_objects[0], moved_objects[0])
        if not np.allclose(realized_delta, delta, atol=1e-12, rtol=0.0):
            raise RuntimeError("realized_scene_object_transform_disagrees_with_requested_delta")
        object_pose_changes = 0
        for before, after in zip(base_objects, moved_objects):
            left = before.primitive_poses
            right = after.primitive_poses
            if len(left) != len(right):
                raise RuntimeError("registration_changed_workpiece_geometry_topology")
            for pose_before, pose_after in zip(left, right):
                before_values = [pose_before.position.x, pose_before.position.y, pose_before.position.z,
                                 pose_before.orientation.x, pose_before.orientation.y, pose_before.orientation.z, pose_before.orientation.w]
                after_values = [pose_after.position.x, pose_after.position.y, pose_after.position.z,
                                pose_after.orientation.x, pose_after.orientation.y, pose_after.orientation.z, pose_after.orientation.w]
                object_pose_changes += int(not np.array_equal(np.asarray(before_values), np.asarray(after_values)))
        applied = apply_world(moveit, moved_objects)
        if not all(applied) or len(applied) != len(base_objects):
            raise RuntimeError("planning_scene_rejected_collision_object_update")
        expected_poses = transform_pose_array_xyzw(target_pose_values, delta)
        expected_normals = (delta[:3, :3] @ normals.T).T
        target_pose_change_count = int(np.count_nonzero(np.linalg.norm(trajectory["tcp_poses_xyz_quat_xyzw"][:, :3] - target_pose_values[:, :3], axis=1) > 1e-12))
        surface_normal_change_count = int(np.count_nonzero(np.linalg.norm(trajectory["surface_normals_base"] - normals, axis=1) > 1e-12))
        target_poses_match_requested_transform = bool(np.allclose(trajectory["tcp_poses_xyz_quat_xyzw"], expected_poses, atol=1e-12, rtol=0.0))
        surface_normals_match_requested_rotation = bool(np.allclose(trajectory["surface_normals_base"], expected_normals, atol=1e-12, rtol=0.0))
        if not target_poses_match_requested_transform or not surface_normals_match_requested_rotation:
            raise RuntimeError("workpiece_reference_transform_propagation_mismatch")
        if case["family"] == "rotation" and surface_normal_change_count == 0:
            raise RuntimeError("nonzero_registration_rotation_did_not_change_surface_normals")
        if case["family"] == "translation" and surface_normal_change_count != 0:
            raise RuntimeError("pure_registration_translation_changed_surface_normals")
        scene_metrics = evaluate_scene(
            moveit,
            q,
            times,
            compare_fk=(frozen_fk[:, :3], frozen_fk[:, 3:6]) if case["case_id"] == "registration_zero" else None,
        )
        case_result = {
            **case,
            "requested_transform_base": delta.tolist(),
            "realized_transform_base": realized_delta.tolist(),
            "object_pose_count_changed": object_pose_changes,
            "target_pose_count_changed": target_pose_change_count,
            "surface_normal_count_changed": surface_normal_change_count,
            "target_poses_match_requested_transform": target_poses_match_requested_transform,
            "surface_normals_match_requested_rotation": surface_normals_match_requested_rotation,
            "world_object_count": len(moved_objects),
            "scene_apply_success_count": sum(applied),
            "joint_states_unchanged": bool(np.array_equal(trajectory["joint_states"], q)),
            "timestamps_unchanged": bool(np.array_equal(trajectory["timestamps_s"], times)),
            "surface_normal_sample_base": trajectory["surface_normals_base"][0].tolist(),
            "normal_rotation_applied": case["family"] == "rotation",
            **scene_metrics,
        }
        if case["family"] == "rotation":
            case_result["transformed_normal_sample"] = trajectory["surface_normals_base"][0].tolist()
        if case["family"] == "zero":
            baseline_result = case_result
        case_results.append(case_result)
    all_updates_realized = all(
        item["joint_states_unchanged"] and item["timestamps_unchanged"] and item["scene_apply_success_count"] == item["world_object_count"]
        and item["target_poses_match_requested_transform"] and item["surface_normals_match_requested_rotation"]
        and (item["family"] == "zero" or (item["object_pose_count_changed"] > 0 and item["target_pose_count_changed"] > 0))
        and (item["family"] != "rotation" or item["surface_normal_count_changed"] > 0)
        for item in case_results
    )
    fk_pass = bool(baseline_result and baseline_result["fk_regression"]["status"] == "PASS")
    all_collision_queries_completed = all(item["sample_count"] > 0 for item in case_results)
    status = "PASS" if all_updates_realized and fk_pass and all_collision_queries_completed else "INCOMPLETE_NATIVE_VALIDATION"
    total_scene_samples = sum(item["sample_count"] for item in case_results)
    total_scene_collisions = sum(item["full_scene_collision_sample_count"] for item in case_results)
    collision_counts = [item["full_scene_collision_sample_count"] for item in case_results]
    collision_response_status = (
        "SATURATED_ALL_CASES"
        if all(item["full_scene_collision_sample_count"] == item["sample_count"] for item in case_results)
        else "DISTINGUISHES_AT_LEAST_ONE_CASE"
    )
    return {
            "schema_version": "p2b3-c3-native-smoke-v1",
            "project": "FAIRINO_FR5",
            "stage": "P2-B3-C3",
            "status": status,
            "native_backend": "MoveIt2 Jazzy MoveItPy PlanningSceneMonitor with active FCL collision environment",
            "native_scene_propagation_tested": "YES" if all_updates_realized and all_collision_queries_completed else "NO",
            "collision_method": COLLISION_METHOD,
            "strict_continuous_ccd": "NOT_AVAILABLE",
            "hardware_validation": "NOT_RUN",
            "physical_registration_bound": "UNKNOWN",
            "frozen_trajectory": {"waypoint_count": len(q), "joint_count": q.shape[1], "timestamps_unchanged": True, "q_unchanged": True},
            "c1_input_identities": identities,
        "scene_geometry": {
                "builder": "ros2_moveit_bridge.plan_closed_contour_moveit.build_wall_collision_objects",
                "frame_id": BASE_FRAME,
                "collision_object_count": len(base_objects),
                "object_ids": identities_in_world,
                "stand_off_m": 0.18,
                "wall_thickness_m": 0.025,
                "y_thickness_m": 0.08,
                "segment_stride": 4,
                "open_path": True,
                "bottom_closure_included": False,
                "floor_included": True,
                "floor_z_m": -0.20,
            "geometry_interpretation": "Existing project thin-box tunnel approximation derived from frozen open-arch target poses and normals; not a calibrated workpiece surface reconstruction.",
        },
        "nominal_world_collision_object_diagnostic": collision_object_diagnostic,
            "registration_convention": {
                "nominal_transform": "T_base_entity_nominal maps each workpiece-attached entity into base_link",
                "perturbation_transform": "DeltaT_base = [[Exp([phi_base]x), t_base], [0,1]]",
                "composition": "T_base_entity_actual = DeltaT_base_left @ T_base_entity_nominal",
                "translation_frame": BASE_FRAME,
                "rotation_frame": BASE_FRAME,
                "rotation_origin": "base_link origin",
                "quaternion_convention": "right-handed rotation matrix; ROS quaternion order x,y,z,w",
                "parameterization": "direct base-frame translation in meters plus base-frame rotation vector in radians; not an SE(3) twist with coupled translational exponential coordinates",
            },
            "case_count": len(case_results),
            "case_cap": MAX_NATIVE_CASES,
            "waypoint_count": len(q),
            "cases": case_results,
            "nominal_zero_fk_regression": baseline_result["fk_regression"] if baseline_result else {"status": "NOT_RUN"},
            "fcl_distance_status": "AVAILABLE_BOUNDED_SMOKE" if all(item["fcl_distance_status"] == "AVAILABLE" for item in case_results) else "NOT_AVAILABLE",
            "collision_response_summary": {
                "status": collision_response_status,
                "full_scene_collision_samples": total_scene_collisions,
                "full_scene_samples": total_scene_samples,
                "self_collision_samples": sum(item["self_collision_sample_count"] for item in case_results),
                "per_case_collision_sample_counts": collision_counts,
                "interpretation": "The bounded binary collision metric is saturated for this scene/path smoke and cannot rank registration directions; retain every observed collision as model-only evidence for independent geometry review.",
            },
            "clearance_claim": "NO_PHYSICAL_CLEARANCE_OR_CONTINUOUS_CLEARANCE_CLAIM",
            "native_smoke_interpretation": "Engineering scene-propagation smoke only; nonzero amplitudes are not physical uncertainty bounds and do not form normalized margins.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-csv", type=Path, required=True)
    parser.add_argument("--target-csv", type=Path, required=True)
    parser.add_argument("--c1-fk-trace", type=Path, required=True)
    parser.add_argument("--c1-result", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--srdf", type=Path, required=True)
    parser.add_argument("--joint-limits", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    exit_code = 2
    try:
        result = run(args)
        exit_code = 0 if result["status"] == "PASS" else 2
    except Exception as exc:
        result = {
            "schema_version": "p2b3-c3-native-smoke-v1",
            "project": "FAIRINO_FR5",
            "stage": "P2-B3-C3",
            "status": "INCOMPLETE_NATIVE_VALIDATION",
            "native_scene_propagation_tested": "NO",
            "reason": f"native_exception:{type(exc).__name__}:{exc}",
            "collision_method": COLLISION_METHOD,
            "strict_continuous_ccd": "NOT_AVAILABLE",
            "hardware_validation": "NOT_RUN",
        }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: result.get(key) for key in ("status", "native_scene_propagation_tested", "case_count", "waypoint_count", "fcl_distance_status", "reason")}, sort_keys=True), flush=True)
    # This process has no controller or hardware lifecycle. MoveItPy teardown
    # is known to fault in this ROS environment; the complete JSON is closed
    # above before terminating without invoking that destructor path.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
