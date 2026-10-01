"""Materialize the frozen C1 nominal collision world without running a replay."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from p2b3_c4_scene_contract import (  # noqa: E402
    build_scene_manifest,
    extract_c1_scene_contract,
    fingerprint_active_scene_semantics,
    manifest_sha256,
    scene_objects_from_moveit,
    verify_frozen_c1_source_blobs,
)

_MOVEIT_KEEPALIVE: Any | None = None


def build(args: argparse.Namespace) -> dict[str, Any]:
    from geometry_msgs.msg import Pose
    from moveit.planning import MoveItPy
    from moveit_configs_utils import MoveItConfigsBuilder
    import rclpy
    from ros2_moveit_bridge.plan_closed_contour_moveit import build_wall_collision_objects

    c1_result = json.loads(args.c1_result.read_text(encoding="utf-8"))
    source_blobs = verify_frozen_c1_source_blobs(ROOT, str(c1_result.get("C1_EXECUTION_CODE_COMMIT", "")))
    contract = extract_c1_scene_contract(
        c1_result,
        (ROOT / "scripts" / "run_p2b3_c1_r0.py").read_text(encoding="utf-8"),
        (ROOT / "ros2_moveit_bridge" / "plan_closed_contour_moveit.py").read_text(encoding="utf-8"),
    )
    with args.target_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != contract.waypoint_count:
        raise RuntimeError("C1_target_waypoint_count_mismatch")
    expected_target_sha = c1_result.get("input_identity_sha256", {}).get(
        "outputs\\internal_wiper_moveit_inputs\\open_arch_tcp_poses_base_link.csv"
    )
    actual_target_sha = hashlib.sha256(args.target_csv.read_bytes()).hexdigest()
    if expected_target_sha and expected_target_sha != actual_target_sha:
        raise RuntimeError("C1_target_csv_identity_mismatch")
    poses: list[Any] = []
    normals: list[list[float]] = []
    for row in rows:
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = (float(row[key]) for key in ("x", "y", "z"))
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = (
            float(row[key]) for key in ("qx", "qy", "qz", "qw")
        )
        poses.append(pose)
        normals.append([float(row[key]) for key in ("nx", "ny", "nz")])
    normal_array = np.asarray(normals, dtype=np.float64)
    parameters = contract.parameters
    objects = build_wall_collision_objects(
        target_poses=poses,
        target_normals=normal_array,
        stand_off=parameters["stand_off_m"],
        frame_id=contract.frame_id,
        wall_thickness=parameters["wall_thickness_m"],
        y_thickness=parameters["y_thickness_m"],
        segment_stride=parameters["segment_stride"],
        include_bottom_closure=parameters["include_bottom_closure"],
        open_path=parameters["open_path"],
        tcp_points_to_wall=parameters["tcp_points_to_wall"],
        include_tunnel_floor=parameters["include_tunnel_floor"],
        tunnel_floor_z=parameters["tunnel_floor_z_m"],
    )
    moveit_config = (
        MoveItConfigsBuilder("fairino5_v6_robot", package_name="fairino5_v6_moveit2_config")
        .robot_description(file_path=str(args.urdf))
        .robot_description_semantic(file_path=str(args.srdf))
        .joint_limits(file_path=str(args.joint_limits))
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )
    config_dict = moveit_config.to_dict()
    config_dict["planning_pipelines"] = {
        "pipeline_names": ["ompl"],
        "ompl": copy.deepcopy(config_dict.get("ompl", {})),
    }
    rclpy.init(args=None)
    global _MOVEIT_KEEPALIVE
    _MOVEIT_KEEPALIVE = MoveItPy(node_name="p2b3_c4_scene_manifest", config_dict=config_dict)
    moveit = _MOVEIT_KEEPALIVE
    monitor = moveit.get_planning_scene_monitor()
    with monitor.read_write() as scene:
        scene.remove_all_collision_objects()
        for obj in objects:
            applied = scene.apply_collision_object(obj)
            if applied is False:
                raise RuntimeError(f"planning_scene_rejected_manifest_object:{obj.id}")
    scene_semantics = fingerprint_active_scene_semantics(moveit, [str(obj.id) for obj in objects], args.urdf)
    manifest = build_scene_manifest(
        contract,
        scene_objects_from_moveit(objects),
        acm=scene_semantics["acm"],
        padding=scene_semantics["robot_padding_scale"],
    )
    manifest["runtime_api_surface"] = scene_semantics["api_surface"]
    manifest["source_c1"]["source_git_blobs"] = source_blobs
    manifest["source_c1"]["target_tcp_csv_sha256"] = actual_target_sha
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return {
        "status": "PASS",
        "project": manifest["project"],
        "scene_object_count": manifest["collision_objects"]["count"],
        "waypoint_count": contract.waypoint_count,
        "manifest_sha256": manifest_sha256(manifest),
        "output": str(args.output),
        "source_git_blobs": source_blobs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c1-result", type=Path, default=ROOT / "outputs" / "p2b3_c1_result.json")
    parser.add_argument("--target-csv", type=Path, default=ROOT / "outputs" / "internal_wiper_moveit_inputs" / "open_arch_tcp_poses_base_link.csv")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, default=ROOT / "outputs" / "p2b2_inputs" / "derived_reference_robot_model.urdf")
    parser.add_argument("--srdf", type=Path, default=ROOT / "ros2_moveit_bridge" / "config" / "fairino5_v6_spray_tcp.srdf")
    parser.add_argument("--joint-limits", type=Path, default=ROOT / "ros2_moveit_bridge" / "config" / "joint_limits_with_jerk.yaml")
    args = parser.parse_args()
    try:
        result = build(args)
    except Exception as exc:
        result = {"status": "FAIL", "reason": f"{type(exc).__name__}:{exc}"}
        print(json.dumps(result, sort_keys=True), flush=True)
        return 2
    print(json.dumps(result, sort_keys=True), flush=True)
    if result.get("status") == "PASS":
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
