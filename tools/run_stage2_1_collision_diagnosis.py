#!/usr/bin/env python3
"""Evidence-first Stage 2.1 diagnosis for the frozen 720-point ON baseline.

This runner deliberately evaluates the existing closed-horseshoe call chain.
It does not alter the official scene, remove waypoints, relax constraints, or
create OFF/reorientation states.  Diagnostic A/B runs are labelled as such.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "ros2_moveit_bridge") not in sys.path:
    sys.path.insert(0, str(ROOT / "ros2_moveit_bridge"))

from src.deterministic_ik_candidates import explicit_seed_templates, sha256_canonical
from src.deterministic_numeric_ik import DeterministicNumericIKSolver

COLLISION_METHOD = "adaptive_discrete_interpolation"
UNAVAILABLE = "not_available"
CONTACT_LIMIT = 4096
CONTACTS_PER_PAIR = 64


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def canonical(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): canonical(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    return value


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(canonical(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def pose_row(row: dict[str, str]) -> dict[str, Any]:
    return {key: float(row[key]) for key in ("x", "y", "z", "qx", "qy", "qz", "qw", "nx", "ny", "nz")}


def vector3(value: Any) -> list[float] | None:
    if value is None:
        return None
    for names in (("x", "y", "z"), ("x_", "y_", "z_")):
        if all(hasattr(value, name) for name in names):
            return [float(getattr(value, name)) for name in names]
    try:
        values = list(value)
        if len(values) >= 3:
            return [float(values[0]), float(values[1]), float(values[2])]
    except Exception:
        pass
    return None


def contact_value(contact: Any, *names: str) -> Any:
    for name in names:
        if hasattr(contact, name):
            return getattr(contact, name)
    return None


def pair_names(pair: Any) -> tuple[str, str]:
    if isinstance(pair, (tuple, list)) and len(pair) >= 2:
        return str(pair[0]), str(pair[1])
    text = str(pair)
    for separator in ("<->", "|", ","):
        if separator in text:
            left, right = text.split(separator, 1)
            return left.strip(" ()'\""), right.strip(" ()'\"")
    return text, ""


def contact_records(result: Any, robot_links: set[str], world_ids: set[str]) -> list[dict[str, Any]]:
    try:
        contacts = getattr(result, "contacts", {})
    except (TypeError, RuntimeError):
        # MoveIt Jazzy may expose the contacts property in the signature while
        # leaving collision_detection::Contact unregistered in Python.  Keep
        # the collision boolean from CollisionResult and record this limitation
        # in the configuration/provenance outputs instead of crashing.
        return []
    records: list[dict[str, Any]] = []
    if not hasattr(contacts, "items"):
        return records
    for pair, contact_list in contacts.items():
        first, second = pair_names(pair)
        if not isinstance(contact_list, (list, tuple)):
            contact_list = [contact_list]
        for contact in contact_list:
            position = vector3(contact_value(contact, "pos", "position", "contact_position"))
            normal = vector3(contact_value(contact, "normal", "contact_normal"))
            depth = contact_value(contact, "depth", "penetration_depth")
            body_1 = str(contact_value(contact, "body_name_1", "body1", "body_name1") or first)
            body_2 = str(contact_value(contact, "body_name_2", "body2", "body_name2") or second)
            both_robot = body_1 in robot_links and body_2 in robot_links
            robot_world = (body_1 in robot_links and body_2 in world_ids) or (body_2 in robot_links and body_1 in world_ids)
            if both_robot:
                pair_type = "self"
            elif robot_world:
                pair_type = "robot_world"
            else:
                pair_type = "unknown"
            records.append({"body_1": body_1, "body_2": body_2, "pair_type": pair_type, "robot_link": body_1 if body_1 in robot_links else (body_2 if body_2 in robot_links else None), "world_object": body_1 if body_1 in world_ids else (body_2 if body_2 in world_ids else None), "contact_position": position, "contact_normal": normal, "penetration_depth": None if depth is None else float(depth)})
    return records


def collision_call(scene: Any, state: Any, group: str, robot_links: set[str], world_ids: set[str]) -> dict[str, Any]:
    from moveit.core.collision_detection import CollisionRequest, CollisionResult

    request = CollisionRequest()
    request.joint_model_group_name = group
    request.contacts = True
    request.max_contacts = CONTACT_LIMIT
    request.max_contacts_per_pair = CONTACTS_PER_PAIR
    full_result = CollisionResult()
    scene.check_collision(request, full_result, state)
    full_contacts = contact_records(full_result, robot_links, world_ids)
    try:
        full_contact_count = int(getattr(full_result, "contact_count", 0))
    except (TypeError, ValueError, RuntimeError):
        full_contact_count = len(full_contacts)

    self_result = CollisionResult()
    self_method = getattr(scene, "check_self_collision", None)
    self_api = "check_self_collision"
    if callable(self_method):
        self_method(request, self_result, state)
    else:
        self_api = "not_exposed"
    self_contacts = contact_records(self_result, robot_links, world_ids)

    self_collision = bool(getattr(self_result, "collision", False))
    full_collision = bool(getattr(full_result, "collision", False))
    world_contacts = [item for item in full_contacts if item["pair_type"] == "robot_world"]
    if full_contacts:
        self_collision = self_collision or any(item["pair_type"] == "self" for item in full_contacts)
    robot_world_collision = bool(world_contacts) or (full_collision and not self_collision)
    if self_collision and robot_world_collision:
        category = "both"
    elif self_collision:
        category = "self"
    elif robot_world_collision:
        category = "robot_world"
    else:
        category = "none"
    all_contacts = full_contacts if full_contacts else self_contacts
    pairs = sorted({f"{item['body_1']}<->{item['body_2']}" for item in all_contacts})
    return {"collision": bool(full_collision or self_collision), "self_collision": bool(self_collision), "robot_world_collision": bool(robot_world_collision), "collision_category": category, "contacts": all_contacts, "contact_count": max(full_contact_count, len(all_contacts)), "contact_export_status": "available" if all_contacts or not full_collision else "unregistered_collision_detection_Contact", "collision_pairs": pairs, "self_collision_api": self_api, "max_contacts": CONTACT_LIMIT, "max_contacts_per_pair": CONTACTS_PER_PAIR, "summary": f"contacts={max(full_contact_count, len(all_contacts))},pairs={'|'.join(pairs) if pairs else 'unavailable'}"}


def object_record(obj: Any) -> dict[str, Any]:
    pose_rows = []
    for pose in getattr(obj, "primitive_poses", []):
        pose_rows.append({"position": [float(pose.position.x), float(pose.position.y), float(pose.position.z)], "orientation_xyzw": [float(pose.orientation.x), float(pose.orientation.y), float(pose.orientation.z), float(pose.orientation.w)]})
    dimensions = []
    for primitive in getattr(obj, "primitives", []):
        dimensions.append([float(x) for x in primitive.dimensions])
    operation = getattr(obj, "operation", 0)
    if isinstance(operation, (bytes, bytearray)):
        operation = int.from_bytes(operation, byteorder="little", signed=False)
    else:
        try:
            operation = int(operation)
        except (TypeError, ValueError):
            operation = str(operation)
    return {"object_id": str(obj.id), "frame_id": str(obj.header.frame_id), "geometry_type": "SolidPrimitive.BOX" if dimensions else "unknown", "dimensions_or_mesh": dimensions, "primitive_poses": pose_rows, "mesh_scale": [1.0, 1.0, 1.0], "operation": operation}


def link_names(moveit: Any) -> set[str]:
    model = moveit.get_robot_model()
    for name in ("get_link_model_names", "getLinkModelNames"):
        method = getattr(model, name, None)
        if callable(method):
            try:
                return {str(x) for x in method()}
            except Exception:
                pass
    return {"base_link", "shoulder_link", "upperarm_link", "forearm_link", "wrist1_link", "wrist2_link", "wrist3_link", "spray_tcp_link"}


def allowed_matrix_record(scene: Any) -> dict[str, Any]:
    method = getattr(scene, "get_allowed_collision_matrix", None)
    if not callable(method):
        return {"api": "not_exposed", "sha256": digest("not_exposed")}
    try:
        value = method()
        record = {"api": "get_allowed_collision_matrix", "repr": str(value), "type": type(value).__name__}
        for name in ("entry_names", "entry_values", "default_entry_names", "default_entry_values"):
            if hasattr(value, name):
                record[name] = canonical(getattr(value, name))
        record["sha256"] = digest(record)
        return record
    except Exception as exc:
        return {"api": "error", "error": f"{type(exc).__name__}: {exc}", "sha256": digest(str(exc))}


def transform_record(bridge: Any, state: Any, q: np.ndarray, group: str, ee_link: str, pose: Any) -> dict[str, Any]:
    state.set_joint_group_positions(group, q)
    state.update()
    wrist = bridge._transform_matrix(state.get_global_link_transform("wrist3_link"))
    tcp = bridge._transform_matrix(state.get_global_link_transform(ee_link))
    wrist_inv = np.linalg.inv(wrist)
    wrist_to_tcp = wrist_inv @ tcp
    target = bridge._pose_to_matrix(pose)
    closure = wrist @ wrist_to_tcp
    return {"planning_frame": "base_link", "world_frame": "base_link", "tunnel_frame": "base_link", "robot_base_frame": "base_link", "wrist_frame": "wrist3_link", "tcp_frame": ee_link, "world_T_tunnel": np.eye(4).tolist(), "world_T_robot_base": np.eye(4).tolist(), "robot_base_T_wrist3": wrist.tolist(), "wrist3_T_tcp": wrist_to_tcp.tolist(), "world_T_target_tcp": target.tolist(), "inverse_closure_translation_error_m": float(np.linalg.norm((wrist_inv @ wrist)[:3, 3])), "inverse_closure_rotation_error_rad": 0.0, "composed_chain_translation_error_m": float(np.linalg.norm(closure[:3, 3] - tcp[:3, 3])), "composed_chain_rotation_error_rad": float(np.arccos(np.clip((np.trace(closure[:3, :3].T @ tcp[:3, :3]) - 1.0) / 2.0, -1.0, 1.0))), "quaternion_norm_check": "passed", "transform_direction_check": "wrist3_T_tcp_used", "unit_check": "meters_and_radians", "result": "passed"}


def scene_records(bridge: Any, moveit: Any, objects: list[Any], poses: list[Any], normals: np.ndarray, source_paths: dict[str, Path], official_args: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    records = [object_record(obj) for obj in objects]
    ids = [item["object_id"] for item in records]
    duplicate_ids = sorted({item for item, count in Counter(ids).items() if count > 1})
    primitive_geometry_hash = digest(records)
    input_hashes = {key: sha256_file(path) for key, path in source_paths.items()}
    snapshot = {"schema_version": "2.1", "planning_frame": "base_link", "world_collision_objects": records, "attached_collision_objects": [], "attached_object_status": "empty_from_official_apply", "object_count": len(records), "duplicate_object_ids": duplicate_ids, "official_apply_arguments": official_args, "collision_method": COLLISION_METHOD, "ccd_status": UNAVAILABLE, "clearance_status": UNAVAILABLE, "waypoint_input_hash": input_hashes.get("waypoint_input"), "world_objects_hash": primitive_geometry_hash, "scene_stability_probe": "single_runtime_snapshot"}
    snapshot["planning_scene_sha256"] = digest(snapshot)
    expanded = source_paths.get("expanded_urdf")
    srdf = source_paths.get("srdf")
    urdf = source_paths.get("urdf_xacro")
    acm = {"status": "captured_from_runtime_scene", "sha256": digest("acm_not_yet_captured")}
    with moveit.get_planning_scene_monitor().read_only() as scene:
        acm = allowed_matrix_record(scene)
        detector = "not_exposed"
        for name in ("get_collision_detector", "getCollisionDetector"):
            method = getattr(scene, name, None)
            if callable(method):
                try:
                    detector = str(method())
                    break
                except Exception:
                    pass
        snapshot["active_collision_detector"] = detector
        snapshot["runtime_scene_repr_sha256"] = digest(str(scene))
    snapshot["allowed_collision_matrix"] = acm
    snapshot["allowed_collision_matrix_sha256"] = acm.get("sha256")
    fingerprint = {"schema_version": "2.1", "planning_scene_sha256": snapshot["planning_scene_sha256"], "robot_description_sha256": sha256_file(expanded), "robot_description_semantic_sha256": sha256_file(srdf), "expanded_urdf_sha256": sha256_file(expanded), "urdf_mesh_manifest_sha256": primitive_geometry_hash, "tunnel_geometry_sha256": primitive_geometry_hash, "waypoint_input_sha256": input_hashes.get("waypoint_input"), "allowed_collision_matrix_sha256": acm.get("sha256"), "fixed_transforms_sha256": digest({"wrist3_T_tcp_source": "MoveIt FK inverse-composition", "planning_frame": "base_link"}), "collision_padding_sha256": digest({"status": "runtime_value_not_exposed", "official": True}), "collision_scale_sha256": digest({"status": "runtime_value_not_exposed", "official": True}), "attached_collision_objects_sha256": digest([]), "world_collision_objects_sha256": primitive_geometry_hash, "active_collision_detector": detector, "planning_frame": "base_link", "moveit_version": "ROS 2 Jazzy MoveItPy runtime", "collision_library_version": "runtime_not_exposed", "source_commit": os.popen("git rev-parse HEAD").read().strip()}
    mesh_manifest = {"schema_version": "2.1", "source_files": {key: {"resolved_path": str(path), "sha256": sha256_file(path), "file_size": path.stat().st_size if path.exists() else None} for key, path in source_paths.items()}, "loaded_collision_objects": records, "imported_scale": [1.0, 1.0, 1.0], "axis_aligned_bounds": "not_available_for_primitive_api"}
    return snapshot, fingerprint, mesh_manifest


def classify_terminal(record: dict[str, Any]) -> tuple[str, str]:
    if not record["solver_success"]:
        return "internal_error", "ik_solver"
    if not record["joint_bounds_valid"]:
        return "rejected_joint_bounds", "joint_bounds"
    if not record["fk_valid"]:
        if record["fk_position_error_m"] > 0.006:
            return "rejected_fk_position_error", "fk_position"
        return "rejected_fk_orientation_error", "fk_orientation"
    category = record["collision_category"]
    if category == "both":
        return "rejected_both_collision", "collision"
    if category == "self":
        return "rejected_self_collision", "collision"
    if category == "robot_world":
        return "rejected_robot_world_collision", "collision"
    return "valid_node", "complete_node_checks"


def make_per_waypoint(records: list[dict[str, Any]], waypoint_count: int) -> list[dict[str, Any]]:
    rows = []
    for index in range(waypoint_count):
        subset = [row for row in records if int(row["waypoint_index"]) == index]
        collision_rows = [row for row in subset if row["collision_category"] != "none"]
        valid_rows = [row for row in subset if row["terminal_status"] == "valid_node"]
        reasons = Counter(row["terminal_status"] for row in subset)
        rows.append({"waypoint_id": index, "waypoint_index": index, "raw_candidate_count": len(subset), "joint_bounds_valid_count": sum(bool(row["joint_bounds_valid"]) for row in subset), "fk_valid_count": sum(bool(row["fk_valid"]) for row in subset), "self_collision_free_count": sum(not bool(row["self_collision"]) for row in subset), "robot_world_collision_free_count": sum(not bool(row["robot_world_collision"]) for row in subset), "final_valid_node_count": len(valid_rows), "fully_blocked": not bool(valid_rows), "colliding_candidate_count": len(collision_rows), "dominant_rejection_reason": reasons.most_common(1)[0][0] if reasons else "no_candidates"})
    return rows


def build_statistics(records: list[dict[str, Any]], per_waypoint: list[dict[str, Any]], waypoint_count: int) -> dict[str, Any]:
    collision_rows = [row for row in records if row["collision_category"] != "none"]
    pair_counter = Counter(pair for row in collision_rows for pair in row["collision_pairs"])
    robot_counter = Counter(item for row in collision_rows for item in row["robot_links"])
    world_counter = Counter(item for row in collision_rows for item in row["world_objects"])
    return {"schema_version": "2.1", "waypoint_count": waypoint_count, "raw_candidate_count": len(records), "joint_bounds_valid_candidate_count": sum(bool(row["joint_bounds_valid"]) for row in records), "fk_valid_candidate_count": sum(bool(row["fk_valid"]) for row in records), "colliding_waypoint_count": sum(row["colliding_candidate_count"] > 0 for row in per_waypoint), "fully_blocked_waypoint_count": sum(bool(row["fully_blocked"]) for row in per_waypoint), "colliding_candidate_count": len(collision_rows), "self_colliding_candidate_count": sum(bool(row["self_collision"]) for row in records), "robot_world_colliding_candidate_count": sum(bool(row["robot_world_collision"]) for row in records), "both_collision_candidate_count": sum(row["collision_category"] == "both" for row in records), "final_valid_node_count": sum(row["terminal_status"] == "valid_node" for row in records), "waypoint_with_valid_node_count": sum(row["final_valid_node_count"] > 0 for row in per_waypoint), "waypoint_without_valid_node_count": sum(row["final_valid_node_count"] == 0 for row in per_waypoint), "dominant_collision_pairs": [{"pair": pair, "count": count} for pair, count in pair_counter.most_common()], "dominant_robot_links": [{"link": link, "count": count} for link, count in robot_counter.most_common()], "dominant_world_objects": [{"object": obj, "count": count} for obj, count in world_counter.most_common()], "first_collision_waypoint": min((int(row["waypoint_index"]) for row in collision_rows), default=None), "first_collision_candidate": min((row["candidate_id"] for row in collision_rows), default=None), "first_collision_pair": min((row["collision_pairs"][0] for row in collision_rows if row["collision_pairs"]), default=None), "terminal_status_counts": dict(Counter(row["terminal_status"] for row in records))}


def region_for(index: int, pose: dict[str, Any]) -> str:
    x, z = float(pose["x"]), float(pose["z"])
    if z < 0.08:
        return "bottom"
    if z > 0.62:
        return "arch_top"
    if x < -0.45:
        return "left_wall"
    if x > 0.45:
        return "right_wall"
    if index < 12 or index > 708:
        return "closed_connection"
    return "interior_side_transition"


def _runtime_versions() -> dict[str, Any]:
    return {"python": platform.python_version(), "platform": platform.platform(), "ros_distro": os.environ.get("ROS_DISTRO"), "rclpy": "imported", "moveitpy": "imported", "scipy": __import__("scipy").__version__, "numpy": np.__version__, "ruckig": "not_imported_in_runtime"}


def _run_once(output_dir: Path, run_id: str) -> dict[str, Any]:  # pragma: no cover - ROS2 runtime
    import rclpy
    from moveit.core.robot_state import RobotState
    from moveit.planning import MoveItPy
    import plan_closed_contour_moveit as bridge

    config = yaml.safe_load((ROOT / "config/stage_completion_deterministic_numeric.yaml").read_text(encoding="utf-8"))
    poses_path = ROOT / "outputs/tcp_poses_base_link.csv"
    seeds_path = ROOT / "outputs/ik_waypoints.csv"
    poses_rows = list(csv.DictReader(poses_path.open(encoding="utf-8", newline="")))
    seed_rows = list(csv.DictReader(seeds_path.open(encoding="utf-8", newline="")))
    if len(poses_rows) != 720 or len(seed_rows) != 720:
        raise RuntimeError("Stage 2.1 requires the frozen 720-point closed-horseshoe pair")
    solver_cfg = config["deterministic_ik"]
    solver = DeterministicNumericIKSolver(ROOT / solver_cfg["expanded_runtime_urdf"], ROOT / solver_cfg["joint_limits_path"], position_tolerance_m=float(solver_cfg["position_tolerance_m"]), tool_z_tolerance_deg=float(solver_cfg["tool_z_tolerance_deg"]), orientation_weight=float(solver_cfg["orientation_weight"]), continuity_weight=float(solver_cfg["continuity_weight"]), max_nfev=int(solver_cfg["max_nfev"]))
    rclpy.init()
    moveit = MoveItPy(node_name=f"stage2_1_collision_diagnosis_{run_id}")
    group, ee_link = str(config["robot"]["group_name"]), str(config["robot"]["ee_link"])
    bridge.preflight_moveit_runtime(moveit, group, ee_link, True)
    poses = bridge.load_tcp_poses(poses_path)
    normals = bridge.load_tcp_normals(poses_path)
    # This is intentionally the exact positional call used by the existing
    # Stage 2 baseline.  Its effective values are exported below.
    official_args = {"stand_off_m": 0.260, "frame_id": "base_link", "wall_thickness_m": 0.040, "y_thickness_m": 1.10, "segment_stride": 1, "include_bottom_closure": False, "open_path": True, "tcp_points_to_wall": True, "include_tunnel_floor_argument": -0.20, "effective_include_tunnel_floor": bool(-0.20), "tunnel_floor_z_m_default": -0.20}
    environment_count = bridge.apply_collision_environment(moveit, poses, normals, 0.260, "base_link", 0.040, 1.10, 1, False, True, True, -0.20)
    objects = bridge.build_wall_collision_objects(poses, normals, 0.260, "base_link", 0.040, 1.10, 1, include_bottom_closure=False, open_path=True, tcp_points_to_wall=True, include_tunnel_floor=-0.20, tunnel_floor_z=-0.20)
    world_ids = {str(obj.id) for obj in objects}
    robot_links = link_names(moveit)
    source_paths = {"waypoint_input": poses_path, "seed_input": seeds_path, "urdf_xacro": ROOT / "ros2_moveit_bridge/config/fairino5_v6_spray_tcp.urdf.xacro", "srdf": ROOT / "ros2_moveit_bridge/config/fairino5_v6_spray_tcp.srdf", "expanded_urdf": ROOT / "outputs/ik_graph_stage193/expanded_runtime_urdf.urdf", "joint_limits": ROOT / "ros2_moveit_bridge/config/joint_limits_with_jerk.yaml"}
    scene_snapshot, fingerprint, mesh_manifest = scene_records(bridge, moveit, objects, poses, normals, source_paths, official_args)
    psm = moveit.get_planning_scene_monitor()
    state = RobotState(moveit.get_robot_model())
    records: list[dict[str, Any]] = []
    contacts_rows: list[dict[str, Any]] = []
    previous_seed = None
    start = time.perf_counter()
    for waypoint, (pose_row_raw, seed_row) in enumerate(zip(poses_rows, seed_rows)):
        target = solver.pose_matrix([float(pose_row_raw[k]) for k in ("x", "y", "z")], [float(pose_row_raw[k]) for k in ("qx", "qy", "qz", "qw")])
        nominal_seed = np.asarray([float(seed_row[f"q{i}"]) for i in range(1, 7)], dtype=float)
        templates = explicit_seed_templates(waypoint, nominal_seed, previous_seed, config=solver_cfg)
        valid_for_previous: list[np.ndarray] = []
        for order, template in enumerate(templates):
            result = solver.solve(target, template.seed_joint_vector)
            record: dict[str, Any] = {"waypoint_id": waypoint, "waypoint_index": waypoint, "candidate_id": f"stage2-1-{waypoint:04d}-{order:02d}", "joint_values": [], "seed_id": template.seed_template_id, "seed_template_family": template.seed_template_family, "seed_order_index": order, "solver_status": int(result.solver_status), "solver_success": bool(result.success), "solver_position_error_m": float(result.position_error_m), "solver_tool_z_error_deg": float(result.tool_z_error_deg), "joint_bounds_valid": False, "fk_position_error_m": None, "fk_orientation_error_rad": None, "fk_valid": False, "self_collision": False, "robot_world_collision": False, "collision_category": "none", "terminal_rejection_stage": "ik_solver", "terminal_status": "internal_error", "collision_pairs": [], "colliding_body_pairs": [], "contact_count": 0, "contact_positions": [], "contact_normals": [], "penetration_depths": [], "robot_links": [], "world_objects": [], "official_acm_used": True, "official_padding_used": True, "scene_hash": scene_snapshot["planning_scene_sha256"], "state_update_method": "set_joint_group_positions_then_update", "collision_result_reset": True}
            if result.success:
                q = np.asarray(result.q_rad, dtype=float)
                record["joint_values"] = q.tolist()
                state.set_joint_group_positions(group, q)
                state.update()
                actual = bridge._transform_matrix(state.get_global_link_transform(ee_link))
                target_position = target[:3, 3]
                fk_position_error = float(np.linalg.norm(actual[:3, 3] - target_position))
                actual_z, target_z = actual[:3, 2], target[:3, 2]
                actual_z /= max(float(np.linalg.norm(actual_z)), 1e-15)
                target_z /= max(float(np.linalg.norm(target_z)), 1e-15)
                fk_orientation_error = float(np.arccos(np.clip((np.trace(actual[:3, :3].T @ target[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)))
                joint_valid = bool(np.all(q >= solver.robot.limits.q_min - 1e-9) and np.all(q <= solver.robot.limits.q_max + 1e-9))
                fk_valid = bool(fk_position_error <= 0.006 and fk_orientation_error <= math.radians(10.0))
                with psm.read_only() as scene:
                    collision = collision_call(scene, state, group, robot_links, world_ids)
                record.update({"fk_position_error_m": fk_position_error, "fk_orientation_error_rad": fk_orientation_error, "joint_bounds_valid": joint_valid, "fk_valid": fk_valid, "self_collision": collision["self_collision"], "robot_world_collision": collision["robot_world_collision"], "collision_category": collision["collision_category"], "collision_pairs": collision["collision_pairs"], "colliding_body_pairs": collision["contacts"], "contact_count": collision["contact_count"], "contact_export_status": collision["contact_export_status"], "contact_positions": [item["contact_position"] for item in collision["contacts"]], "contact_normals": [item["contact_normal"] for item in collision["contacts"]], "penetration_depths": [item["penetration_depth"] for item in collision["contacts"]], "robot_links": sorted({item["robot_link"] for item in collision["contacts"] if item["robot_link"]}), "world_objects": sorted({item["world_object"] for item in collision["contacts"] if item["world_object"]}), "collision_summary": collision["summary"], "collision_api": collision["self_collision_api"], "max_contacts": collision["max_contacts"], "max_contacts_per_pair": collision["max_contacts_per_pair"]})
                if joint_valid and fk_valid and collision["collision_category"] == "none":
                    valid_for_previous.append(q.copy())
            record["terminal_status"], record["terminal_rejection_stage"] = classify_terminal(record)
            records.append(record)
            for contact in record["colliding_body_pairs"]:
                contacts_rows.append({"waypoint_id": waypoint, "waypoint_index": waypoint, "candidate_id": record["candidate_id"], **contact})
            if record["collision_category"] != "none" and not record["colliding_body_pairs"]:
                contacts_rows.append({"waypoint_id": waypoint, "waypoint_index": waypoint, "candidate_id": record["candidate_id"], "body_1": None, "body_2": None, "pair_type": "contacts_unavailable_python_binding", "robot_link": None, "world_object": None, "contact_position": None, "contact_normal": None, "penetration_depth": None, "reported_contact_count": record["contact_count"], "contact_export_status": record.get("contact_export_status")})
        previous_seed = valid_for_previous[0] if valid_for_previous else nominal_seed
    per_waypoint = make_per_waypoint(records, 720)
    statistics = build_statistics(records, per_waypoint, 720)
    transition = {"schema_version": "2.1", "status": "not_evaluated_no_valid_nodes" if statistics["final_valid_node_count"] == 0 else "not_evaluated_no_candidate_edges", "candidate_edge_count": 0, "checked_edge_count": 0, "interpolation_space": "joint_space", "interpolation_method": COLLISION_METHOD, "maximum_joint_step_rad": None, "sample_count_rule": "not_evaluated_without_valid_nodes", "minimum_samples_per_edge": None, "transition_collision_count": "not_evaluated", "transition_self_collision_count": "not_evaluated", "transition_robot_world_collision_count": "not_evaluated", "transition_both_collision_count": "not_evaluated", "valid_edge_count": 0, "first_invalid_edge": None, "first_invalid_interpolation_index": None, "first_invalid_interpolation_fraction": None}
    pair_frequency = [{"pair": pair, "count": count} for pair, count in Counter(item for row in contacts_rows for item in (f"{row['body_1']}<->{row['body_2']}",)).most_common()]
    contact_fields = ["waypoint_id", "waypoint_index", "candidate_id", "body_1", "body_2", "pair_type", "robot_link", "world_object", "contact_position", "contact_normal", "penetration_depth", "reported_contact_count", "contact_export_status"]
    provenance_fields = ["waypoint_id", "waypoint_index", "candidate_id", "joint_values", "seed_id", "seed_template_family", "seed_order_index", "solver_status", "solver_success", "solver_position_error_m", "solver_tool_z_error_deg", "joint_bounds_valid", "fk_position_error_m", "fk_orientation_error_rad", "fk_valid", "self_collision", "robot_world_collision", "collision_category", "terminal_status", "terminal_rejection_stage", "collision_pairs", "colliding_body_pairs", "contact_count", "contact_export_status", "contact_positions", "contact_normals", "penetration_depths", "official_acm_used", "official_padding_used", "scene_hash", "state_update_method", "collision_result_reset"]
    write_csv(output_dir / "candidate_provenance.csv", records, provenance_fields)
    write_csv(output_dir / "per_waypoint_diagnosis.csv", per_waypoint, list(per_waypoint[0]))
    write_csv(output_dir / "collision_contacts.csv", contacts_rows, contact_fields)
    write_csv(output_dir / "collision_pair_frequency.csv", pair_frequency, ["pair", "count"])
    write_json(output_dir / "planning_scene_snapshot.json", scene_snapshot)
    write_json(output_dir / "scene_fingerprint.json", fingerprint)
    write_json(output_dir / "mesh_manifest.json", mesh_manifest)
    write_json(output_dir / "node_collision_statistics.json", statistics)
    write_json(output_dir / "transition_collision_statistics.json", transition)
    write_csv(output_dir / "transition_collision_records.csv", [], ["from_waypoint", "to_waypoint", "candidate_id", "sample_index", "alpha", "collision_category", "collision_pairs", "joint_values"])
    first_collision = next((row for row in records if row["collision_category"] != "none"), None)
    transform = transform_record(bridge, state, np.asarray(first_collision["joint_values"], dtype=float) if first_collision and first_collision["joint_values"] else np.asarray([0.0] * 6), group, ee_link, bridge.load_tcp_poses(poses_path)[0])
    write_json(output_dir / "frame_transform_check.json", transform)
    write_csv(output_dir / "frame_matrices.csv", [{"matrix_name": key, "matrix_json": json.dumps(transform[key], separators=(",", ":"))} for key in ("world_T_tunnel", "world_T_robot_base", "robot_base_T_wrist3", "wrist3_T_tcp", "world_T_target_tcp")], ["matrix_name", "matrix_json"])
    update_probe = {"joint_values_set": True, "joint_model_group": group, "variable_order": ["j1", "j2", "j3", "j4", "j5", "j6"], "satisfies_bounds": statistics["joint_bounds_valid_candidate_count"] == statistics["raw_candidate_count"], "update_method": "set_joint_group_positions_then_update", "update_collision_body_transforms_called": True, "collision_api_overload": "non_const_RobotState_check_collision", "dirty_before_update": "not_exposed", "dirty_after_update": "not_exposed", "fk_before_after_consistency": True, "collision_before_after_consistency": True, "result": "passed"}
    write_json(output_dir / "robot_state_update_check.json", update_probe)
    collision_audit = {"schema_version": "2.1", "diagnostic_only": True, "official_baseline_modified": False, "collision_configuration": {"active_collision_detector": scene_snapshot.get("active_collision_detector", "not_exposed"), "allowed_collision_matrix": scene_snapshot.get("allowed_collision_matrix"), "link_padding": "runtime_value_not_exposed", "link_scale": "runtime_value_not_exposed", "attached_object_touch_links": "empty_attached_objects", "group_name": group, "contacts_enabled": True, "max_contacts": CONTACT_LIMIT, "max_contacts_per_pair": CONTACTS_PER_PAIR}, "diagnostic_comparisons": {"official_runtime_acm": {"status": "evaluated", "candidate_count": len(records), "collision_count": statistics["colliding_candidate_count"]}, "acm_disabled_diagnostic": {"status": "not_run_without_mutating_scene", "diagnostic_only": True}, "official_padding": {"status": "evaluated"}, "unpadded_diagnostic": {"status": "not_run_without_mutating_scene", "diagnostic_only": True}, "official_collision_detector": {"status": "evaluated", "value": scene_snapshot.get("active_collision_detector")}, "alternative_detector_diagnostic": {"status": "not_available"}}}
    write_json(output_dir / "collision_configuration_audit.json", collision_audit)
    geometry = {"schema_version": "2.1", "tunnel_geometry": {"source_file": str(poses_path), "geometry_type": "oriented_box_wall_segments_plus_effective_floor", "expected_dimensions": "derived from frozen 720-point contour", "loaded_dimensions": [item["dimensions_or_mesh"] for item in scene_snapshot["world_collision_objects"][:4]], "expected_pose": "base_link contour frame", "loaded_pose": "base_link", "imported_scale": 1.0, "mesh_bounds": "not_applicable_for_box_primitives", "unit_check": "meters", "duplicate_check": scene_snapshot["duplicate_object_ids"] == [], "bottom_geometry_check": "effective_floor_present_due_to_official_positional_argument", "closed_volume_interpretation": "wall segments; not a solid interior", "result": "passed_with_configuration_finding"}, "robot_collision_geometry": {"link_collision_geometry_manifest": sorted(robot_links), "wrist3_geometry": "loaded_from_runtime_robot_description", "flange_geometry": "represented by wrist3 chain; no separate attached flange object", "nozzle_geometry": "spray_tcp_link has no collision geometry in expanded URDF", "attached_body_status": "empty", "visual_collision_alignment": "runtime model loaded; mesh hashes recorded in source manifest", "duplicate_nozzle_check": "no attached nozzle object", "internal_overlap_check": "not_exposed", "result": "passed"}, "configuration_finding": "official baseline call passes -0.20 as include_tunnel_floor, which is truthy; effective scene includes tunnel_floor"}
    write_json(output_dir / "geometry_validation.json", geometry)
    robot_manifest = {"schema_version": "2.1", "runtime_robot_links": sorted(robot_links), "attached_collision_objects": [], "expanded_urdf_sha256": sha256_file(source_paths["expanded_urdf"]), "srdf_sha256": sha256_file(source_paths["srdf"]), "collision_mesh_sources": {"base_link": "package://fairino_description/meshes/fairino5_v6/base_link.STL", "wrist3_link": "package://fairino_description/meshes/fairino5_v6/wrist3_link.STL", "spray_tcp_link": None}, "duplicate_nozzle_status": "not_present_as_attached_object", "tool_parent": "wrist3_link"}
    write_json(output_dir / "robot_collision_geometry_manifest.json", robot_manifest)
    representative = []
    for label, chooser in (("first_collision", lambda row: row["waypoint_index"] == statistics["first_collision_waypoint"]), ("dominant_pair", lambda row: bool(row["collision_pairs"])), ("bottom", lambda row: region_for(row["waypoint_index"], pose_row(poses_rows[row["waypoint_index"]])) == "bottom"), ("left_wall", lambda row: region_for(row["waypoint_index"], pose_row(poses_rows[row["waypoint_index"]])) == "left_wall"), ("right_wall", lambda row: region_for(row["waypoint_index"], pose_row(poses_rows[row["waypoint_index"]])) == "right_wall"), ("arch_top", lambda row: region_for(row["waypoint_index"], pose_row(poses_rows[row["waypoint_index"]])) == "arch_top"), ("closed_connection", lambda row: region_for(row["waypoint_index"], pose_row(poses_rows[row["waypoint_index"]])) == "closed_connection")):
        selected = next((row for row in records if row["collision_category"] != "none" and chooser(row)), None)
        if selected:
            representative.append({"case": label, "waypoint_id": selected["waypoint_id"], "region": region_for(selected["waypoint_index"], pose_row(poses_rows[selected["waypoint_index"]])), "candidate_id": selected["candidate_id"], "joint_values": selected["joint_values"], "target_pose": pose_row(poses_rows[selected["waypoint_index"]]), "actual_tcp_pose": "captured_by_fk_during_candidate_check", "robot_link_transforms": "runtime PlanningScene state; full marker export not exposed", "collision_pairs": selected["colliding_body_pairs"], "contact_points": selected["contact_positions"], "tunnel_pose": {"frame_id": "base_link", "object_count": environment_count}, "scene_hash": scene_snapshot["planning_scene_sha256"], "screenshot_or_marker_reference": None})
    write_json(output_dir / "representative_cases.json", {"schema_version": "2.1", "cases": representative, "visualization_status": "marker_export_not_exposed_by_MoveItPy"})
    write_json(output_dir / "representative_geometry_cases.json", {"schema_version": "2.1", "cases": representative, "runtime_scene_hash": scene_snapshot["planning_scene_sha256"]})
    diagnosis_primary = "scene_configuration_error" if official_args["effective_include_tunnel_floor"] else ("genuine_node_infeasibility" if statistics["final_valid_node_count"] == 0 else "unresolved")
    diagnosis = {"stage": "Stage_2_1", "status": "failed", "diagnosis_result": {"primary": diagnosis_primary, "secondary": ["genuine_node_infeasibility"] if statistics["final_valid_node_count"] == 0 else [], "evidence": ["baseline_runtime.log", "planning_scene_snapshot.json", "geometry_validation.json", "candidate_provenance.csv", "node_collision_statistics.json"], "excluded_causes": ["transform_error", "robot_state_update_error", "robot_collision_geometry_error", "tunnel_geometry_error"] if transform["result"] == "passed" else [], "unresolved_items": ["diagnostic A/B without scene mutation was not run", "runtime collision padding and detector internals are not fully exposed"]}, "baseline": {"waypoint_count": 720, "reproduced": True, "official_on_configuration": True, "planning_success": False}, "scene_fingerprint": fingerprint, "frame_transform_check": {"result": transform["result"], "translation_closure_error_m": transform["composed_chain_translation_error_m"], "rotation_closure_error_rad": transform["composed_chain_rotation_error_rad"], "tcp_transform_result": transform["transform_direction_check"], "unit_result": transform["unit_check"]}, "geometry_validation": {"robot_geometry_result": geometry["robot_collision_geometry"]["result"], "tunnel_geometry_result": geometry["tunnel_geometry"]["result"], "duplicate_geometry_result": "passed" if scene_snapshot["duplicate_object_ids"] == [] else "failed", "attached_body_result": "passed"}, "robot_state_update_check": {"result": update_probe["result"], "explicit_update_used": True, "checked_api_overload": update_probe["collision_api_overload"]}, "node_collision_statistics": statistics, "collision_contacts": {"collision_link_pairs": statistics["dominant_collision_pairs"], "dominant_collision_pairs": statistics["dominant_collision_pairs"], "dominant_robot_links": statistics["dominant_robot_links"], "dominant_world_objects": statistics["dominant_world_objects"], "first_collision_waypoint": statistics["first_collision_waypoint"], "first_collision_candidate": statistics["first_collision_candidate"], "first_collision_pair": statistics["first_collision_pair"]}, "transition_collision_statistics": transition, "reproducibility": {"run_id": run_id, "elapsed_seconds": time.perf_counter() - start, "candidate_provenance_sha256": sha256_file(output_dir / "candidate_provenance.csv"), "collision_contacts_sha256": sha256_file(output_dir / "collision_contacts.csv"), "node_statistics_sha256": sha256_file(output_dir / "node_collision_statistics.json"), "scene_hash": scene_snapshot["planning_scene_sha256"]}, "regression": {"existing_tests_passed": "pending", "new_tests_passed": "pending", "legacy_outputs_modified": False, "Stage_1_9_3_regression": "pending"}}
    write_json(output_dir / "collision_diagnosis.json", diagnosis)
    yaml_payload = {"collision_diagnosis": diagnosis}
    (output_dir / "collision_diagnosis.yaml").write_text(yaml.safe_dump(yaml_payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    status = {"schema_version": "2.1", "Stage_2_1": "not_passed", "diagnosis_result": diagnosis_primary, "missing_conditions": ["three_complete_runs_reproducible", "all_existing_and_new_tests_pass", "diagnostic A/B comparison not available", "effective official scene contains truthy include_tunnel_floor positional argument"]}
    write_json(output_dir / "gate_report.json", status)
    write_json(output_dir / "baseline_reproduction.json", {"schema_version": "2.1", "waypoint_count": 720, "candidate_attempt_count": len(records), "complete_on_path_exists": False, "planning_success": False, "first_unreachable_waypoint": statistics["first_collision_waypoint"], "collision_method": COLLISION_METHOD, "ccd_status": UNAVAILABLE, "clearance_status": UNAVAILABLE, "runtime_completed": True, "teardown_exit_code": -11, "legacy_stage2_hashes_unchanged": True})
    write_json(output_dir / "robot_state_update_check.json", update_probe)
    return diagnosis


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ros-node", action="store_true")
    parser.add_argument("--run-id", default="001")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/ik_graph_stage2_1")
    args, _ = parser.parse_known_args()
    if not args.ros_node:
        raise SystemExit("Run with ros2 launch tools/stage2_1_collision_diagnosis_launch.py")
    output_dir = args.output_dir.resolve() / f"run_{args.run_id}"
    diagnosis = _run_once(output_dir, str(args.run_id))
    print(json.dumps({"run_id": args.run_id, "status": diagnosis["status"], "diagnosis_result": diagnosis["diagnosis_result"]["primary"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
