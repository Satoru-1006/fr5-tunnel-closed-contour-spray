"""Deterministic, audit-oriented PlanningScene snapshot helpers.

The snapshot records the scene description that was actually submitted to
MoveIt.  It deliberately does not claim CCD or clearance when the backend
does not expose those values.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from src.deterministic_ik_candidates import canonical_json, sha256_canonical


def file_sha256(path: str | Path) -> str | None:
    path = Path(path)
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pose_message(pose: Any) -> dict[str, Any]:
    return {
        "position": {"x": float(pose.position.x), "y": float(pose.position.y), "z": float(pose.position.z)},
        "orientation": {"x": float(pose.orientation.x), "y": float(pose.orientation.y), "z": float(pose.orientation.z), "w": float(pose.orientation.w)},
    }


def collision_object_record(obj: Any) -> dict[str, Any]:
    primitives = []
    for primitive, pose in zip(getattr(obj, "primitives", []), getattr(obj, "primitive_poses", [])):
        primitives.append({"shape_type": int(primitive.type), "dimensions": [float(x) for x in primitive.dimensions], "pose": _pose_message(pose)})
    return {"id": str(obj.id), "frame": str(obj.header.frame_id), "primitives": primitives}


def build_snapshot(*, robot_model_hash: str | None, urdf_hash: str | None, srdf_hash: str | None, kinematics_hash: str | None, joint_limits_hash: str | None, collision_detector_type: str, planning_frame: str, group_name: str, objects: Iterable[Any], allowed_collision_matrix: Any, robot_state: Any, interpolation_step_deg: float) -> dict[str, Any]:
    world = [collision_object_record(obj) for obj in objects]
    world.sort(key=lambda item: item["id"])
    state = {"joint_positions": [float(x) for x in getattr(robot_state, "joint_positions", [])]} if isinstance(robot_state, dict) else {"repr": str(robot_state)}
    snapshot: dict[str, Any] = {
        "schema_version": "1.0",
        "robot_model_hash": robot_model_hash,
        "urdf_hash": urdf_hash,
        "srdf_hash": srdf_hash,
        "kinematics_hash": kinematics_hash,
        "joint_limits_hash": joint_limits_hash,
        "collision_detector_type": collision_detector_type,
        "planning_frame": planning_frame,
        "group_name": group_name,
        "world_collision_objects": world,
        "attached_collision_objects": [],
        "allowed_collision_matrix": allowed_collision_matrix,
        "link_padding": {},
        "link_scale": {},
        "bottom_collision_enabled": True,
        "bottom_closure_enabled": True,
        "collision_object_count": len(world),
        "robot_state": state,
        "interpolation_step_deg": float(interpolation_step_deg),
        "collision_method": "adaptive_discrete_interpolation",
        "ccd_status": "not_available",
        "clearance_status": "not_available",
    }
    snapshot["planning_scene_semantic_hash"] = sha256_canonical(snapshot)
    return snapshot


def write_snapshot(path: str | Path, snapshot: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(snapshot, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
