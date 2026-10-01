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
from p2b3_c4_scene_contract import (  # noqa: E402
    assess_zero_transform_gate,
    build_scene_manifest,
    canonical_quaternion_xyzw,
    compare_scene_manifests,
    extract_c1_scene_contract,
    fingerprint_active_scene_semantics,
    manifest_sha256,
    scene_objects_from_moveit,
    validate_phase_d_identity_gate,
    verify_frozen_c1_source_blobs,
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


def collision_samples_with_context(q: np.ndarray, times: np.ndarray) -> Iterable[tuple[np.ndarray, float, dict[str, Any]]]:
    """Return the same adaptive discrete samples with stable waypoint/segment labels."""
    sample_index = 0
    yield np.array(q[0], copy=True), float(times[0]), {
        "sample_index": sample_index, "waypoint_index": 0, "segment_start_waypoint": 0,
        "segment_end_waypoint": 0, "segment_fraction": 0.0,
    }
    sample_index += 1
    for index, (left, right) in enumerate(zip(q[:-1], q[1:])):
        steps = max(1, int(math.ceil(float(np.max(np.abs(right - left))) / MAX_JOINT_STEP_RAD)))
        for substep in range(1, steps + 1):
            fraction = substep / steps
            yield left + fraction * (right - left), float(times[index] + fraction * (times[index + 1] - times[index])), {
                "sample_index": sample_index, "waypoint_index": index + 1 if substep == steps else None,
                "segment_start_waypoint": index, "segment_end_waypoint": index + 1,
                "segment_fraction": fraction,
            }
            sample_index += 1


def _first_contact_summary(result: Any) -> dict[str, Any]:
    try:
        contacts = result.contacts
    except Exception as exc:  # MoveItPy contact conversion varies by binding build.
        return {"status": "NOT_AVAILABLE", "reason": f"contact_map_access_error:{type(exc).__name__}"}
    if not hasattr(contacts, "items"):
        return {"status": "NOT_AVAILABLE", "reason": "contacts_binding_is_not_a_mapping"}
    for pair, entries in contacts.items():
        try:
            first = entries[0] if len(entries) else None
        except Exception:
            first = None
        try:
            pair_contact_count = len(entries)
        except Exception:
            pair_contact_count = None
        fields: list[str] = []
        if first is not None:
            fields = sorted(name for name in dir(first) if not name.startswith("_"))
        pair_items = list(pair) if isinstance(pair, (tuple, list)) else [pair]
        return {
            "status": "AVAILABLE",
            "pair": [str(item) for item in pair_items],
            "contact_count_for_pair": pair_contact_count,
            "contact_record_type": type(first).__name__ if first is not None else None,
            "exposed_contact_fields": fields,
            "nearest_points": {"status": "NOT_AVAILABLE", "reason": "no verified point field semantics in active binding"},
            "penetration_depth": {"status": "NOT_AVAILABLE", "reason": "no verified signed-depth field semantics in active binding"},
        }
    return {"status": "AVAILABLE_NO_CONTACTS", "pair": None}


def apply_world(moveit: Any, objects: list[Any]) -> list[bool]:
    monitor = moveit.get_planning_scene_monitor()
    results: list[bool] = []
    with monitor.read_write() as scene:
        scene.remove_all_collision_objects()
        for obj in objects:
            applied = scene.apply_collision_object(obj)
            results.append(applied is not False)
    return results


def summarize_collision_response(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    counts = [int(item["full_scene_collision_sample_count"]) for item in case_results]
    sample_counts = [int(item["sample_count"]) for item in case_results]
    total_collisions = sum(counts)
    total_samples = sum(sample_counts)
    self_collisions = sum(int(item["self_collision_sample_count"]) for item in case_results)
    if not case_results or any(samples <= 0 for samples in sample_counts):
        status = "NOT_RUN_OR_INCOMPLETE"
        interpretation = "At least one case lacks completed collision samples; no aggregate response classification is available."
    elif all(collisions == samples for collisions, samples in zip(counts, sample_counts)):
        status = "SATURATED_ALL_CASES"
        interpretation = "Every sampled state collided in every case; the bounded binary collision metric cannot rank registration directions."
    elif all(collisions == 0 for collisions in counts):
        status = "NO_COLLISIONS_OBSERVED"
        interpretation = "No sampled collision was observed in these engineering smoke cases; this does not establish clearance, continuous safety, or a registration margin."
    elif len(set(counts)) > 1:
        status = "COLLISION_COUNTS_VARY_BY_CASE"
        interpretation = "Sampled collision counts vary across cases; retain the model-only counts without interpreting them as physical robustness or a margin."
    else:
        status = "UNIFORM_PARTIAL_COLLISIONS"
        interpretation = "Each case has the same partial sampled collision count; the bounded binary metric does not distinguish the cases."
    return {
        "status": status,
        "full_scene_collision_samples": total_collisions,
        "full_scene_samples": total_samples,
        "self_collision_samples": self_collisions,
        "per_case_collision_sample_counts": counts,
        "interpretation": interpretation,
    }


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
    request.max_contacts = 12
    request.max_contacts_per_pair = 2
    full_collisions = 0
    self_collisions = 0
    distance_values: list[float] = []
    contact_point_total = 0
    maximum_contacts_in_one_sample = 0
    sample_count = 0
    first_collision: dict[str, Any] | None = None
    scene_distance_methods: list[str] = []
    scene_semantics_methods: list[str] = []
    robot_model_semantics_methods: list[str] = []
    scene_type = "unavailable"
    result_contact_map_available: bool | None = None
    monitor = moveit.get_planning_scene_monitor()
    with monitor.read_only() as scene:
        scene_type = f"{type(scene).__module__}.{type(scene).__qualname__}"
        scene_distance_methods = sorted(
            name for name in dir(scene) if "distance" in name.lower() or "collision" in name.lower()
        )
        scene_semantics_methods = sorted(
            name for name in dir(scene)
            if any(token in name.lower() for token in ("allowed", "padding", "scale", "acm", "world", "detector"))
        )
        robot_model_semantics_methods = sorted(
            name for name in dir(model)
            if any(token in name.lower() for token in ("padding", "scale", "link_model", "collision"))
        )
        for q_sample, _time_s, context in collision_samples_with_context(q, times):
            state.set_joint_group_positions(GROUP, q_sample.tolist())
            state.update()
            full_result = CollisionResult()
            self_result = CollisionResult()
            scene.check_collision(request, full_result, state)
            scene.check_self_collision(request, self_result, state)
            full_collisions += int(bool(full_result.collision))
            if full_result.collision and first_collision is None:
                first_collision = {
                    **context,
                    "collision": True,
                    "contact_count": int(full_result.contact_count),
                    "contact_diagnostic": _first_contact_summary(full_result),
                }
            self_collisions += int(bool(self_result.collision))
            contact_point_total += int(full_result.contact_count)
            maximum_contacts_in_one_sample = max(maximum_contacts_in_one_sample, int(full_result.contact_count))
            raw_distance = float(full_result.distance)
            distance_values.append(raw_distance)
            try:
                result_contact_map_available = hasattr(full_result.contacts, "items")
            except Exception:
                result_contact_map_available = False
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
    from p2b3_c4_scene_contract import aggregate_reported_distances

    distance_summary = aggregate_reported_distances(distance_values)
    return {
        "sample_count": sample_count,
        "collision_method": COLLISION_METHOD,
        "maximum_joint_interpolation_step_rad": MAX_JOINT_STEP_RAD,
        "full_scene_collision_sample_count": full_collisions,
        "self_collision_sample_count": self_collisions,
        "contact_point_total": contact_point_total,
        "maximum_contacts_in_one_sample": maximum_contacts_in_one_sample,
        "minimum_reported_fcl_distance_m": distance_summary["minimum_reported_full_scene_distance_m"],
        "minimum_robot_world_distance_m": None,
        "fcl_distance_status": distance_summary["status"],
        "fcl_distance_diagnostics": distance_summary,
        "fcl_distance_interpretation": distance_summary["interpretation"],
        "planning_scene_binding": scene_type,
        "planning_scene_collision_distance_methods": scene_distance_methods,
        "planning_scene_acm_padding_world_methods": scene_semantics_methods,
        "robot_model_padding_collision_methods": robot_model_semantics_methods,
        "collision_request_distance_property_available": hasattr(request, "distance"),
        "collision_result_distance_property_available": hasattr(CollisionResult(), "distance"),
        "collision_contacts_mapping_available": result_contact_map_available,
        "first_collision": first_collision,
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

    c1_result = json.loads(args.c1_result.read_text(encoding="utf-8"))
    c1_runner = ROOT / "scripts" / "run_p2b3_c1_r0.py"
    planner_source = ROOT / "ros2_moveit_bridge" / "plan_closed_contour_moveit.py"
    source_blobs = verify_frozen_c1_source_blobs(ROOT, str(c1_result.get("C1_EXECUTION_CODE_COMMIT", "")))
    scene_contract = extract_c1_scene_contract(
        c1_result,
        c1_runner.read_text(encoding="utf-8"),
        planner_source.read_text(encoding="utf-8"),
    )
    times, q, target_pose_values, normals, frozen_fk, identities = read_c1_inputs(args)
    nominal_pose_refs = apply_registration_to_trajectory(target_pose_values, normals, q, times, np.eye(4))
    base_objects = build_wall_collision_objects(
        target_poses=pose_messages(target_pose_values),
        target_normals=normals,
        stand_off=scene_contract.parameters["stand_off_m"],
        frame_id=scene_contract.frame_id,
        wall_thickness=scene_contract.parameters["wall_thickness_m"],
        y_thickness=scene_contract.parameters["y_thickness_m"],
        segment_stride=scene_contract.parameters["segment_stride"],
        include_bottom_closure=scene_contract.parameters["include_bottom_closure"],
        open_path=scene_contract.parameters["open_path"],
        tcp_points_to_wall=scene_contract.parameters["tcp_points_to_wall"],
        include_tunnel_floor=scene_contract.parameters["include_tunnel_floor"],
        tunnel_floor_z=scene_contract.parameters["tunnel_floor_z_m"],
    )
    if not base_objects or any(obj.header.frame_id != BASE_FRAME for obj in base_objects):
        raise RuntimeError("native_scene_builder_returned_no_base_frame_objects")
    identities_in_world = [str(obj.id) for obj in base_objects]
    if len(set(identities_in_world)) != len(identities_in_world):
        raise RuntimeError("native_scene_object_ids_not_unique")
    source_scene_manifest = build_scene_manifest(scene_contract, scene_objects_from_moveit(base_objects))
    source_scene_manifest["source_c1"]["source_git_blobs"] = source_blobs
    source_scene_manifest["source_c1"]["target_tcp_csv_sha256"] = identities["target_tcp_csv_sha256"]
    manifest_path = getattr(args, "scene_manifest", None)
    if manifest_path is not None:
        expected_scene_manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    else:
        expected_scene_manifest = source_scene_manifest

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
    _MOVEIT_KEEPALIVE = MoveItPy(node_name="p2b3_c4_native_smoke", config_dict=config_dict)
    moveit = _MOVEIT_KEEPALIVE
    initially_applied = apply_world(moveit, base_objects)
    if not all(initially_applied) or len(initially_applied) != len(base_objects):
        raise RuntimeError("planning_scene_rejected_authoritative_base_world")
    scene_semantics = fingerprint_active_scene_semantics(moveit, identities_in_world, args.urdf)
    source_scene_manifest["acm"] = scene_semantics["acm"]
    source_scene_manifest["robot_padding_scale"] = scene_semantics["robot_padding_scale"]
    source_scene_manifest["runtime_api_surface"] = scene_semantics["api_surface"]
    scene_comparison = compare_scene_manifests(expected_scene_manifest, source_scene_manifest)
    effective_manifest_sha = manifest_sha256(expected_scene_manifest)
    if scene_comparison.get("status") != "PASS":
        raise RuntimeError("active_moveit_scene_contract_mismatch:" + json.dumps(scene_comparison, sort_keys=True))
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
    phase = getattr(args, "phase", "gated")
    identity_case, nonzero_cases = cases[0], cases[1:]
    if phase == "identity":
        cases = [identity_case]
    elif phase == "smoke":
        gate_path = getattr(args, "identity_gate_json", None)
        if gate_path is None or not Path(gate_path).is_file():
            raise RuntimeError("phase_d_requires_phase_c_identity_gate_file")
        identity_gate_record = json.loads(Path(gate_path).read_text(encoding="utf-8"))
        baseline_result = validate_phase_d_identity_gate(identity_gate_record, effective_manifest_sha)
        zero_transform_gate = identity_gate_record["zero_transform_equivalence"]
        cases = nonzero_cases
    elif phase == "gated":
        cases = [identity_case, *nonzero_cases]
    else:
        raise RuntimeError(f"unknown_native_phase:{phase}")
    case_results: list[dict[str, Any]] = []
    if phase != "smoke":
        baseline_result = None
        zero_transform_gate = None
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
        if case["family"] == "zero":
            target_pose_values_actual = trajectory["tcp_poses_xyz_quat_xyzw"]
            position_change = float(np.max(np.linalg.norm(target_pose_values_actual[:, :3] - target_pose_values[:, :3], axis=1)))
            quaternion_change = max(
                max(abs(a - b) for a, b in zip(canonical_quaternion_xyzw(actual[3:7]), canonical_quaternion_xyzw(expected[3:7])))
                for actual, expected in zip(target_pose_values_actual, target_pose_values)
            )
            target_pose_max_change = max(position_change, float(quaternion_change))
            normal_max_change = float(np.max(np.linalg.norm(trajectory["surface_normals_base"] - normals, axis=1)))
            zero_transform_gate = assess_zero_transform_gate(
                scene_contract,
                trajectory_sha256=identities["c1_post_ruckig_sha256"],
                waypoint_count=len(q),
                sampled_state_count=int(scene_metrics["sample_count"]),
                collision_count=int(scene_metrics["full_scene_collision_sample_count"]),
                fk_status=str(scene_metrics["fk_regression"]["status"]),
                q_unchanged=bool(np.array_equal(trajectory["joint_states"], q)),
                timestamps_unchanged=bool(np.array_equal(trajectory["timestamps_s"], times)),
                target_pose_max_change=target_pose_max_change,
                normal_max_change=normal_max_change,
                scene_comparison=scene_comparison,
                all_objects_applied=(sum(applied) == len(base_objects)),
            )
            gate_record = {
                "project": "FAIRINO_FR5",
                "stage": getattr(args, "stage", "P2-B3-C3"),
                "zero_transform_equivalence": zero_transform_gate,
                "scene_equivalence": scene_comparison,
                "scene_manifest_sha256": effective_manifest_sha,
                "scene_apply_success_count": sum(applied),
                "scene_object_count": len(base_objects),
                "baseline": case_result,
            }
            gate_path = getattr(args, "identity_gate_json", None)
            if gate_path is not None:
                Path(gate_path).parent.mkdir(parents=True, exist_ok=True)
                Path(gate_path).write_text(json.dumps(gate_record, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
            if zero_transform_gate["status"] != "PASS":
                return {
                    "schema_version": "p2b3-c4-native-smoke-v1",
                    "project": "FAIRINO_FR5",
                    "stage": getattr(args, "stage", "P2-B3-C3"),
                    "status": "BLOCKED_ZERO_TRANSFORM_EQUIVALENCE",
                    "native_scene_propagation_tested": "NO",
                    "collision_method": COLLISION_METHOD,
                    "strict_continuous_ccd": "NOT_AVAILABLE",
                    "hardware_validation": "NOT_RUN",
                    "physical_registration_bound": "UNKNOWN",
                    "c1_input_identities": identities,
                    "c1_scene_source_blobs": source_blobs,
                    "scene_manifest_sha256": effective_manifest_sha,
                    "scene_equivalence": scene_comparison,
                    "zero_transform_equivalence": zero_transform_gate,
                    "zero_baseline": case_result,
                    "nonzero_engineering_smoke": "NOT_RUN_ZERO_GATE_FAILED",
                    "registration_margin_campaign": "NOT_RUN",
                    "scene_geometry": {"collision_object_count": len(base_objects), "object_ids": identities_in_world, **dict(scene_contract.parameters)},
                }
            if phase == "identity":
                return {
                    "schema_version": "p2b3-c4-native-smoke-v1",
                    "project": "FAIRINO_FR5",
                    "stage": getattr(args, "stage", "P2-B3-C3"),
                    "phase": "C_IDENTITY_ONLY",
                    "status": "IDENTITY_GATE_PASS_PHASE_D_PENDING",
                    "native_scene_propagation_tested": "YES",
                    "collision_method": COLLISION_METHOD,
                    "strict_continuous_ccd": "NOT_AVAILABLE",
                    "hardware_validation": "NOT_RUN",
                    "physical_registration_bound": "UNKNOWN",
                    "c1_input_identities": identities,
                    "c1_scene_source_blobs": source_blobs,
                    "scene_manifest_sha256": effective_manifest_sha,
                    "scene_equivalence": scene_comparison,
                    "zero_transform_equivalence": zero_transform_gate,
                    "zero_baseline": case_result,
                    "nonzero_engineering_smoke": "NOT_RUN_PHASE_C_ONLY",
                    "registration_margin_campaign": "NOT_RUN",
                    "scene_geometry": {"collision_object_count": len(base_objects), "object_ids": identities_in_world, **dict(scene_contract.parameters)},
                }
    all_updates_realized = all(
        item["joint_states_unchanged"] and item["timestamps_unchanged"] and item["scene_apply_success_count"] == item["world_object_count"]
        and item["target_poses_match_requested_transform"] and item["surface_normals_match_requested_rotation"]
        and (item["family"] == "zero" or (item["object_pose_count_changed"] > 0 and item["target_pose_count_changed"] > 0))
        and (item["family"] != "rotation" or item["surface_normal_count_changed"] > 0)
        for item in case_results
    )
    fk_pass = bool(baseline_result and baseline_result["fk_regression"]["status"] == "PASS")
    all_collision_queries_completed = all(item["sample_count"] > 0 for item in case_results)
    status = "PASS" if all_updates_realized and fk_pass and all_collision_queries_completed and zero_transform_gate and zero_transform_gate["status"] == "PASS" else "INCOMPLETE_NATIVE_VALIDATION"
    collision_response_summary = summarize_collision_response(case_results)
    return {
            "schema_version": "p2b3-c4-native-smoke-v1",
            "project": "FAIRINO_FR5",
            "stage": getattr(args, "stage", "P2-B3-C3"),
            "status": status,
            "native_backend": "MoveIt2 Jazzy MoveItPy PlanningSceneMonitor with active FCL collision environment",
            "native_scene_propagation_tested": "YES" if all_updates_realized and all_collision_queries_completed else "NO",
            "collision_method": COLLISION_METHOD,
            "strict_continuous_ccd": "NOT_AVAILABLE",
            "hardware_validation": "NOT_RUN",
            "physical_registration_bound": "UNKNOWN",
            "frozen_trajectory": {"waypoint_count": len(q), "joint_count": q.shape[1], "timestamps_unchanged": True, "q_unchanged": True},
            "c1_input_identities": identities,
        "c1_scene_source_blobs": source_blobs,
            "scene_manifest_sha256": effective_manifest_sha,
            "scene_equivalence": scene_comparison,
            "zero_transform_equivalence": zero_transform_gate,
            "zero_baseline_collision_count": baseline_result["full_scene_collision_sample_count"] if baseline_result else None,
            "c1_expected_collision_count": scene_contract.expected_collision_count,
            "phase": "D_NONZERO_ENGINEERING_SMOKE" if phase == "smoke" else "GATED_ALL_CASES",
            "nonzero_engineering_smoke": "COMPLETE_12_CASES" if phase == "smoke" and len(case_results) == 12 else "NOT_RUN_OR_INCOMPLETE",
            "registration_margin_campaign": "NOT_RUN",
            "scene_geometry": {
                "builder": "ros2_moveit_bridge.plan_closed_contour_moveit.build_wall_collision_objects",
                "frame_id": scene_contract.frame_id,
                "collision_object_count": len(base_objects),
                "object_ids": identities_in_world,
                **dict(scene_contract.parameters),
                "bottom_closure_included": scene_contract.parameters["include_bottom_closure"],
                "floor_included": scene_contract.parameters["include_tunnel_floor"],
                "floor_z_m": scene_contract.parameters["tunnel_floor_z_m"],
                "geometry_interpretation": "Existing project thin-box tunnel approximation derived from frozen open-arch target poses and normals; not a calibrated workpiece surface reconstruction.",
        },
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
            "fcl_distance_status": (
                "AVAILABLE_BOUNDED_SMOKE"
                if all(item["fcl_distance_status"] == "AVAILABLE_SAMPLED_MODEL_DISTANCE" for item in case_results)
                else "PARTIAL_NOT_AVAILABLE"
                if any(item["fcl_distance_status"] == "PARTIAL_NOT_AVAILABLE" for item in case_results)
                else "NOT_AVAILABLE"
            ),
            "collision_response_summary": collision_response_summary,
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
    parser.add_argument("--scene-manifest", type=Path)
    parser.add_argument("--identity-gate-json", type=Path)
    parser.add_argument("--stage", default="P2-B3-C3")
    args = parser.parse_args()
    exit_code = 2
    try:
        result = run(args)
        exit_code = 0 if result["status"] == "PASS" else 2
    except Exception as exc:
        result = {
            "schema_version": "p2b3-c3-native-smoke-v1",
            "project": "FAIRINO_FR5",
            "stage": args.stage,
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
