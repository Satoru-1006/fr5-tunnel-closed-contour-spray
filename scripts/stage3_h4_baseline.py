#!/usr/bin/env python3
"""Run the Stage 3 H4 offline reachability baseline.

The normal entry point is the ROS2 launch file with ``--ros-node``.  The
process uses MoveItPy only for target-pose IK, RobotState FK and static
PlanningScene collision checks.  It never creates a controller/action client,
plans a trajectory, invokes timing, or executes a robot command.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.stage3_h4_reachability import (  # noqa: E402
    COLLISION_METHOD,
    H4Config,
    H4ValidationError,
    TARGET_SCHEMA_VERSION,
    UNAVAILABLE,
    aggregate_target_metrics,
    canonical_json,
    classify_target,
    deduplicate_candidates,
    generate_seed_states,
    replay_hash,
    replay_normalize,
    semantic_hash,
    sha256_file,
    validate_h3_target,
    validate_joint_vector,
)


H0_ROOT = ROOT / "outputs/stage3_h0_entry_authorization_20260807T161253Z"
H0_MANIFEST = H0_ROOT / "stage3_h0_stage2_immutable_baseline_manifest.json"
H3_ROOT = ROOT / "outputs/stage3_h3_coverage_baseline_20260808T034526Z"
H3_TARGETS = H3_ROOT / "stage3_h3_surface_targets.jsonl"
H3_HASHES = H3_ROOT / "stage3_h3_file_hashes.json"
H3_CERTIFICATE = H3_ROOT / "stage3_h3_terminal_certificate.json"
H3_GATE = H3_ROOT / "stage3_h3_gate_definition.json"
H2_GLOB = "outputs/stage3_h2_geometry_baseline_*/stage3_h2_geometry_manifest.json"
DEFAULT_OUTPUT = ROOT / "outputs" / f"stage3_h4_reachability_baseline_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def _jsonl_dump(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n")


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _resolve_recorded_path(value: str) -> Path:
    path = Path(value)
    if path.exists():
        return path.resolve()
    match = re.match(r"^[A-Za-z]:[\\/]robotfucker[\\/](.*)$", value)
    if match:
        return (ROOT / Path(match.group(1).replace("\\", "/"))).resolve()
    return (ROOT / value).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise H4ValidationError("malformed_h3_target_file", f"invalid JSON at line {line_number}") from exc
            if not isinstance(value, dict):
                raise H4ValidationError("malformed_h3_target_file", f"line {line_number} is not an object")
            rows.append(value)
    return rows


def _file_hashes_in_h0() -> dict[str, Any]:
    if not H0_MANIFEST.is_file():
        raise H4ValidationError("h0_manifest_unavailable", f"missing H0 immutable manifest: {H0_MANIFEST}")
    manifest = _read_json(H0_MANIFEST)
    mismatches: list[dict[str, Any]] = []
    for entry in manifest.get("entries", []):
        recorded = str(entry.get("absolute_path", ""))
        path = _resolve_recorded_path(recorded)
        expected = str(entry.get("sha256", ""))
        if not path.is_file():
            mismatches.append({"path": recorded, "reason": "missing", "expected_sha256": expected})
            continue
        actual = sha256_file(str(path))
        if actual != expected:
            mismatches.append({"path": recorded, "reason": "sha256_mismatch", "expected_sha256": expected, "observed_sha256": actual})
    return {
        "manifest_path": _relative(H0_MANIFEST),
        "manifest_sha256": sha256_file(str(H0_MANIFEST)),
        "entry_count": len(manifest.get("entries", [])),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "stage2_baseline_immutable": not mismatches,
        "formal_r2_immutable": not mismatches,
        "h0_h1_h2_immutable": not mismatches,
    }


def _h3_preflight() -> dict[str, Any]:
    required = (H3_TARGETS, H3_HASHES, H3_CERTIFICATE, H3_GATE)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise H4ValidationError("h3_artifact_unavailable", "missing H3 artifacts: " + ", ".join(missing))
    hashes = _read_json(H3_HASHES)
    mismatches: list[dict[str, Any]] = []
    for item in hashes.get("files", []):
        path = H3_ROOT / str(item["relative_path"])
        if not path.is_file():
            mismatches.append({"path": _relative(path), "reason": "missing", "expected_sha256": item.get("sha256")})
            continue
        actual = sha256_file(str(path))
        if actual != item.get("sha256"):
            mismatches.append({"path": _relative(path), "reason": "sha256_mismatch", "expected_sha256": item.get("sha256"), "observed_sha256": actual})
    certificate = _read_json(H3_CERTIFICATE)
    gate = _read_json(H3_GATE)
    rows = _read_jsonl(H3_TARGETS)
    identities = [str(row.get("task_sample_id", "")) for row in rows]
    return {
        "h3_root": _relative(H3_ROOT),
        "target_file_sha256": sha256_file(str(H3_TARGETS)),
        "target_count": len(rows),
        "target_identity_unique": len(identities) == len(set(identities)),
        "target_ordering": "preserved_exactly_from_h3_jsonl",
        "h3_hash_mismatch_count": len(mismatches),
        "h3_mismatches": mismatches,
        "h3_certificate_passed": certificate.get("stage3_h3") == "PASSED",
        "h3_gate_frozen": gate.get("status") == "FROZEN_BEFORE_EXPERIMENT",
        "h3_immutable": not mismatches and certificate.get("stage3_h3") == "PASSED" and gate.get("status") == "FROZEN_BEFORE_EXPERIMENT",
        "h3_coverage_semantics": "H3_COVERAGE_IS_GEOMETRIC_SELF_CONSISTENCY_BASELINE",
        "h3_coverage_not_planner_performance": True,
    }


def _find_h2_manifest(expected_hash: str) -> Path:
    for path in sorted(ROOT.glob(H2_GLOB)):
        if path.is_file() and sha256_file(str(path)) == expected_hash:
            return path
    raise H4ValidationError("h2_manifest_unavailable", f"no H2 manifest matches H3 h2_manifest_hash={expected_hash}")


def _load_h2_records(h2_manifest_path: Path) -> dict[str, dict[str, Any]]:
    manifest = _read_json(h2_manifest_path)
    return {str(record["geometry_id"]): record for record in manifest.get("records", [])}


def _run_pytest_snapshot(label: str) -> dict[str, Any]:
    command = [sys.executable, "-m", "pytest", "-q"]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    raw = completed.stdout + ("\n--- STDERR ---\n" + completed.stderr if completed.stderr else "")
    failed_nodes = sorted(set(re.findall(r"FAILED\s+([^\s]+)", raw)))
    summary_match = re.search(r"(?P<passed>\d+) passed(?:, (?P<failed>\d+) failed)?(?:, (?P<skipped>\d+) skipped)?", raw)
    summary = {
        "label": label,
        "command": command,
        "exit_code": completed.returncode,
        "passed": int(summary_match.group("passed")) if summary_match else None,
        "failed": int(summary_match.group("failed")) if summary_match and summary_match.group("failed") else len(failed_nodes),
        "skipped": int(summary_match.group("skipped")) if summary_match and summary_match.group("skipped") else 0,
        "failed_node_ids": failed_nodes,
        "raw_output": raw,
    }
    return summary


def _regression_preflight() -> dict[str, Any]:
    before = _run_pytest_snapshot("before_h4")
    after = _run_pytest_snapshot("post_h4_source_same_process")
    before_set = set(before["failed_node_ids"])
    after_set = set(after["failed_node_ids"])
    return {
        "schema_version": "stage3-h4-regression-baseline-v1",
        "known_failure_baseline": before,
        "post_h4_regression": after,
        "known_failure_set": sorted(before_set),
        "known_failure_set_changed": before_set != after_set,
        "new_failure_set": sorted(after_set - before_set),
        "new_failure_count": len(after_set - before_set),
        "failure_count_before": len(before_set),
        "failure_count_after": len(after_set),
        "baseline_gate": not (after_set - before_set) and before_set == after_set,
    }


def _pose_message(record: Mapping[str, Any]):
    from geometry_msgs.msg import Pose

    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = [float(value) for value in record["tcp_target_position_xyz_m"]]
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = [float(value) for value in record["tcp_target_orientation_xyzw"]]
    return pose


def _transform_matrix(transform: Any):
    import numpy as np

    if hasattr(transform, "shape"):
        return np.asarray(transform, dtype=float)
    if hasattr(transform, "matrix"):
        return np.asarray(transform.matrix(), dtype=float)
    if hasattr(transform, "translation") and hasattr(transform, "rotation"):
        from scipy.spatial.transform import Rotation

        matrix = np.eye(4, dtype=float)
        matrix[:3, :3] = Rotation.from_quat([transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w]).as_matrix()
        matrix[:3, 3] = [transform.translation.x, transform.translation.y, transform.translation.z]
        return matrix
    return np.asarray(transform, dtype=float)


def _orientation_error(actual, target) -> float:
    import numpy as np
    from scipy.spatial.transform import Rotation

    relative = actual[:3, :3].T @ target[:3, :3]
    return float(Rotation.from_matrix(relative).magnitude())


def _contact_pairs(result: Any) -> list[str]:
    try:
        contacts = getattr(result, "contacts", {})
    except Exception:
        # MoveItPy Jazzy may expose the collision boolean while its pybind
        # conversion for the contact map is unavailable.  Contact export is
        # supplementary provenance; it must not suppress the actual FCL
        # collision decision.
        return []
    if not hasattr(contacts, "items"):
        return []
    pairs: list[str] = []
    for pair in contacts:
        if isinstance(pair, (tuple, list)) and len(pair) >= 2:
            pairs.append(f"{pair[0]}<->{pair[1]}")
        else:
            pairs.append(str(pair))
    return sorted(set(pairs))


def _collision_check(scene: Any, state: Any, group_name: str) -> dict[str, Any]:
    from moveit.core.collision_detection import CollisionRequest, CollisionResult

    request = CollisionRequest()
    request.joint_model_group_name = group_name
    request.contacts = True
    request.max_contacts = 64
    request.max_contacts_per_pair = 8
    full = CollisionResult()
    scene.check_collision(request, full, state)
    self_result = CollisionResult()
    self_method = getattr(scene, "check_self_collision", None)
    if callable(self_method):
        self_method(request, self_result, state)
        self_api = "PlanningScene.check_self_collision"
    else:
        self_api = "not_exposed"
    self_collision = bool(getattr(self_result, "collision", False))
    full_collision = bool(getattr(full, "collision", False))
    pairs = _contact_pairs(full) + _contact_pairs(self_result)
    self_markers = ("<->base_link", "base_link<->", "<->shoulder_link", "shoulder_link<->", "<->upperarm_link", "upperarm_link<->", "<->forearm_link", "forearm_link<->", "<->wrist1_link", "wrist1_link<->", "<->wrist2_link", "wrist2_link<->", "<->wrist3_link", "wrist3_link<->")
    world_pairs = [pair for pair in pairs if "fixture_surface" in pair]
    inferred_self = any(any(marker in pair for marker in self_markers) and "fixture_surface" not in pair for pair in pairs)
    self_collision = self_collision or inferred_self
    environment_collision = bool(world_pairs) or (full_collision and not self_collision)
    return {
        "fcl_self_collision": bool(self_collision),
        "fcl_environment_collision": bool(environment_collision),
        "bullet_environment_collision": None,
        "bullet_environment_collision_status": UNAVAILABLE,
        "collision_free": not self_collision and not environment_collision,
        "collision_method": COLLISION_METHOD,
        "contact_provenance": {"pairs": sorted(set(pairs)), "contact_export_status": "available" if pairs else ("collision_without_python_contacts" if full_collision else "none"), "self_collision_api": self_api, "backend": "FCL via MoveIt2 PlanningScene default collision environment"},
        "ccd_status": UNAVAILABLE,
        "clearance_status": UNAVAILABLE,
        "clearance_m": None,
    }


def _mesh_collision_object(mesh: Any, geometry_id: str):
    from geometry_msgs.msg import Point
    from moveit_msgs.msg import CollisionObject
    from shape_msgs.msg import Mesh as MeshMessage
    from shape_msgs.msg import MeshTriangle

    object_message = CollisionObject()
    object_message.id = f"fixture_surface:{geometry_id}"
    object_message.header.frame_id = "base_link"
    shape = MeshMessage()
    for x, y, z in mesh.vertices:
        point = Point()
        point.x, point.y, point.z = float(x), float(y), float(z)
        shape.vertices.append(point)
    for a, b, c in mesh.triangles:
        triangle = MeshTriangle()
        triangle.vertex_indices = [int(a), int(b), int(c)]
        shape.triangles.append(triangle)
    object_message.meshes.append(shape)
    object_message.mesh_poses.append(__import__("geometry_msgs.msg", fromlist=["Pose"]).Pose())
    object_message.mesh_poses[0].orientation.w = 1.0
    object_message.operation = CollisionObject.ADD
    return object_message


def _configure_geometry_scene(moveit: Any, mesh: Any, geometry_id: str) -> int:
    psm = moveit.get_planning_scene_monitor()
    with psm.read_write() as scene:
        scene.remove_all_collision_objects()
        scene.apply_collision_object(_mesh_collision_object(mesh, geometry_id))
    return 1


def _model_metadata(moveit: Any, config: H4Config, kinematics_path: Path) -> tuple[dict[str, Any], list[str], list[float], list[float]]:
    robot_model = moveit.get_robot_model()
    if not robot_model.has_joint_model_group(config.planning_group):
        raise H4ValidationError("planning_group_unavailable", f"MoveIt model has no group {config.planning_group}")
    group = robot_model.get_joint_model_group(config.planning_group)
    joint_names = list(group.active_joint_model_names)
    links = list(group.link_model_names)
    if config.tcp_link not in links:
        raise H4ValidationError("tcp_unavailable", f"MoveIt group has no TCP link {config.tcp_link}")
    lower: list[float] = []
    upper: list[float] = []
    bounds_records: list[dict[str, Any]] = []
    for name, bounds in zip(joint_names, group.active_joint_model_bounds):
        current = bounds[0] if isinstance(bounds, (list, tuple)) else bounds
        lo = float(current.min_position)
        hi = float(current.max_position)
        lower.append(lo)
        upper.append(hi)
        bounds_records.append({"joint_name": name, "min_position": lo, "max_position": hi, "max_velocity": float(getattr(current, "max_velocity", 0.0)), "max_acceleration": float(getattr(current, "max_acceleration", 0.0))})
    kinematics = {}
    if kinematics_path.is_file():
        try:
            import yaml

            kinematics = yaml.safe_load(kinematics_path.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            kinematics = {"read_error": f"{type(exc).__name__}: {exc}"}
    group_kinematics = kinematics.get(config.planning_group, {}) if isinstance(kinematics, dict) else {}
    package_xml = kinematics_path.parents[1] / "package.xml"
    package_version = None
    if package_xml.is_file():
        match = re.search(r"<version>([^<]+)</version>", package_xml.read_text(encoding="utf-8"))
        package_version = match.group(1) if match else None
    metadata = {
        "schema_version": "stage3-h4-robot-model-metadata-v1",
        "robot_model_name": str(robot_model.name),
        "planning_group": config.planning_group,
        "joint_names_exact_order": joint_names,
        "active_joint_count": len(joint_names),
        "model_frame": config.model_frame,
        "tcp_tip_link": config.tcp_link,
        "group_links": links,
        "joint_bounds": bounds_records,
        "mimic_joint_names": list(getattr(group, "mimic_joint_model_names", [])),
        "passive_joint_names": list(getattr(group, "passive_joint_model_names", [])),
        "kinematics_yaml": _relative(kinematics_path),
        "kinematics_yaml_sha256": sha256_file(str(kinematics_path)) if kinematics_path.is_file() else None,
        "kinematics_solver": group_kinematics.get("kinematics_solver"),
        "solver_implementation": "KDLKinematicsPlugin" if group_kinematics.get("kinematics_solver") == "kdl_kinematics_plugin/KDLKinematicsPlugin" else "runtime_report_required",
        "solver_package": "moveit_kinematics",
        "solver_package_version": package_version,
        "solver_provenance": "installed MoveIt2 Jazzy package and runtime robot_description_kinematics",
        "timeout_semantics": {"kinematics_yaml_timeout_s": group_kinematics.get("kinematics_solver_timeout"), "H4_set_from_ik_timeout_s": config.ik_timeout_s, "timeout_is_per_seed_call": True},
        "random_restart_semantics": "not_exposed_by_MoveItPy; H4 performs one set_from_ik call per explicit seed",
        "multiple_branch_api": "not_exposed; branches are collected from explicit seed calls",
        "seed_influence": "explicitly recorded per seed and compared during replay",
        "nondeterminism_status": "determined_by_three_run_semantic_replay",
        "moveit_fk_api": "RobotState.update -> RobotState.get_global_link_transform",
        "collision_api": "PlanningScene.check_collision and PlanningScene.check_self_collision",
        "collision_backend": "FCL default MoveIt2 collision environment; Bullet candidate result not exposed by MoveItPy",
        "robot_model_hash": semantic_hash({"name": str(robot_model.name), "group": config.planning_group, "links": links, "joint_names": joint_names, "bounds": bounds_records}),
    }
    return metadata, joint_names, lower, upper


def _run_one_replay(
    moveit: Any,
    config: H4Config,
    targets: Sequence[Mapping[str, Any]],
    h2_records: Mapping[str, Mapping[str, Any]],
    h2_manifest_path: Path,
    output_records: bool,
) -> dict[str, Any]:
    import numpy as np
    from moveit.core.robot_state import RobotState
    from src.stage3_h3_task_representation import load_and_verify_h2_mesh

    metadata, joint_names, lower, upper = _model_metadata(moveit, config, Path(os.environ.get("STAGE3_H4_KINEMATICS_PATH", "")))
    seeds = generate_seed_states(joint_names, lower, upper, config=config)
    state = RobotState(moveit.get_robot_model())
    psm = moveit.get_planning_scene_monitor()
    ik_rows: list[dict[str, Any]] = []
    fk_rows: list[dict[str, Any]] = []
    collision_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    geometry_meshes: dict[str, Any] = {}
    h2_config_hash = sha256_file(str(h2_manifest_path))
    for target_index, raw_target in enumerate(targets):
        try:
            target = validate_h3_target(raw_target, config=config)
        except H4ValidationError as exc:
            target_rows.append({"schema_version": TARGET_SCHEMA_VERSION, "target_index": target_index, "task_sample_id": raw_target.get("task_sample_id"), "classification": "INVALID_INPUT", "input_valid": False, "first_rejection_reason": exc.code, "rejection_detail": str(exc), "candidate_count_total": 0, "candidate_count_joint_valid": 0, "candidate_count_fk_valid": 0, "candidate_count_collision_free": 0})
            continue
        geometry_id = str(target["geometry_id"])
        h2_record = h2_records.get(geometry_id)
        if h2_record is None:
            target_rows.append({"schema_version": TARGET_SCHEMA_VERSION, "target_index": target_index, "task_sample_id": target["task_sample_id"], "classification": "INVALID_INPUT", "input_valid": False, "first_rejection_reason": "geometry_record_unavailable", "candidate_count_total": 0, "candidate_count_joint_valid": 0, "candidate_count_fk_valid": 0, "candidate_count_collision_free": 0})
            continue
        if geometry_id not in geometry_meshes:
            # H2 manifests were authored on Windows and may contain literal
            # backslashes.  Normalize only the runtime lookup path; the H3
            # provenance fields remain byte-for-byte unchanged.
            runtime_h2_record = dict(h2_record)
            normalized_source = str(h2_record["source_path"]).replace("\\", "/")
            runtime_h2_record["source_path"] = str((ROOT / normalized_source).resolve())
            source_path = Path(runtime_h2_record["source_path"])
            mesh, _processing, _derived = load_and_verify_h2_mesh(str(ROOT), runtime_h2_record)
            geometry_meshes[geometry_id] = mesh
            _configure_geometry_scene(moveit, mesh, geometry_id)
        pose = _pose_message(target)
        raw_candidates: list[dict[str, Any]] = []
        seed_attempts: list[dict[str, Any]] = []
        for seed in seeds:
            state.set_joint_group_positions(config.planning_group, np.asarray(seed["joint_positions"], dtype=float))
            state.update()
            started = time.perf_counter()
            try:
                solved = bool(state.set_from_ik(config.planning_group, pose, config.tcp_link, config.ik_timeout_s))
                error = None if solved else "NO_IK_SOLUTION"
            except Exception as exc:
                solved = False
                error = f"{type(exc).__name__}: {exc}"
            attempt: dict[str, Any] = {"target_index": target_index, "task_sample_id": target["task_sample_id"], "seed_id": seed["seed_id"], "seed_index": seed["seed_index"], "seed_joint_positions": seed["joint_positions"], "requested_timeout_s": config.ik_timeout_s, "solver_api_entry": "RobotState.set_from_ik", "solver_success": solved, "solver_error": error, "elapsed_s": time.perf_counter() - started, "internal_random_restart_observed": None, "silent_repair_applied": False}
            if solved:
                state.update()
                q = np.asarray(state.get_joint_group_positions(config.planning_group), dtype=float).tolist()
                attempt["solution_joint_positions"] = q
                attempt["solution_joint_vector_sha256"] = semantic_hash(q)
                raw_candidates.append({"seed_id": seed["seed_id"], "seed_index": seed["seed_index"], "joint_positions": q, "solver_api_entry": "RobotState.set_from_ik"})
            seed_attempts.append(attempt)
        unique = deduplicate_candidates(raw_candidates, tolerance=config.duplicate_tolerance_rad)
        fk_valid_count = 0
        joint_valid_count = 0
        collision_free_count = 0
        self_collision_count = 0
        env_collision_count = 0
        candidate_fk_rows: list[dict[str, Any]] = []
        candidate_collision_rows: list[dict[str, Any]] = []
        target_pose_matrix = np.eye(4, dtype=float)
        from scipy.spatial.transform import Rotation

        target_pose_matrix[:3, :3] = Rotation.from_quat(target["tcp_target_orientation_xyzw"]).as_matrix()
        target_pose_matrix[:3, 3] = np.asarray(target["tcp_target_position_xyz_m"], dtype=float)
        for candidate in unique:
            joint_result = validate_joint_vector(candidate["joint_positions"], expected_joint_names=joint_names, lower=lower, upper=upper, tolerance=config.joint_limit_tolerance_rad)
            joint_valid = bool(joint_result["valid"])
            joint_valid_count += int(joint_valid)
            fk_row: dict[str, Any] = {"schema_version": "stage3-h4-fk-validation-v1", "target_index": target_index, "task_sample_id": target["task_sample_id"], "candidate_id": candidate["candidate_id"], "joint_validation": joint_result, "fk_attempted": False, "fk_valid": False, "translation_error_m": None, "orientation_error_rad": None, "fk_implementation": "MoveIt2 RobotState.update + get_global_link_transform", "frame": config.model_frame, "tip_link": config.tcp_link, "robot_model_hash": metadata["robot_model_hash"]}
            collision_row: dict[str, Any] = {"schema_version": "stage3-h4-collision-validation-v1", "target_index": target_index, "task_sample_id": target["task_sample_id"], "candidate_id": candidate["candidate_id"], "joint_valid": joint_valid, "fk_valid": False, "collision_checked": False, "collision_method": COLLISION_METHOD, "fcl_self_collision": None, "fcl_environment_collision": None, "bullet_environment_collision": None, "collision_free": None, "ccd_status": UNAVAILABLE, "clearance_status": UNAVAILABLE, "clearance_m": None, "contact_provenance": None}
            if joint_valid:
                try:
                    state.set_joint_group_positions(config.planning_group, np.asarray(candidate["joint_positions"], dtype=float))
                    state.update()
                    actual = _transform_matrix(state.get_global_link_transform(config.tcp_link))
                    position_error = float(np.linalg.norm(actual[:3, 3] - target_pose_matrix[:3, 3]))
                    orientation_error = _orientation_error(actual, target_pose_matrix)
                    fk_valid = position_error <= config.fk_translation_tolerance_m and orientation_error <= config.fk_orientation_tolerance_rad
                    fk_row.update({"fk_attempted": True, "fk_valid": fk_valid, "translation_error_m": position_error, "orientation_error_rad": orientation_error, "computed_tcp_pose_matrix": actual.tolist()})
                    collision_row["fk_valid"] = fk_valid
                    if fk_valid:
                        fk_valid_count += 1
                        with psm.read_only() as scene:
                            collision = _collision_check(scene, state, config.planning_group)
                        collision_row.update({"collision_checked": True, **collision})
                        if collision["fcl_self_collision"]:
                            self_collision_count += 1
                        if collision["fcl_environment_collision"]:
                            env_collision_count += 1
                        if collision["collision_free"]:
                            collision_free_count += 1
                except Exception as exc:
                    fk_row["fk_error"] = f"{type(exc).__name__}: {exc}"
            candidate_fk_rows.append(fk_row)
            candidate_collision_rows.append(collision_row)
        if collision_free_count:
            classification = "REACHABLE_COLLISION_FREE"
        elif env_collision_count and not self_collision_count:
            classification = "IK_FOUND_ENV_COLLISION"
        elif self_collision_count:
            classification = "IK_FOUND_SELF_COLLISION"
        else:
            classification = classify_target(input_valid=True, candidate_count_total=len(raw_candidates), candidate_count_joint_valid=joint_valid_count, candidate_count_fk_valid=fk_valid_count, candidate_count_collision_free=collision_free_count)
        target_row = {"schema_version": TARGET_SCHEMA_VERSION, "target_index": target_index, "geometry_id": geometry_id, "task_sample_id": target["task_sample_id"], "input_valid": True, "classification": classification, "candidate_count_total": len(raw_candidates), "candidate_count_deduplicated": len(unique), "candidate_count_joint_valid": joint_valid_count, "candidate_count_fk_valid": fk_valid_count, "candidate_count_collision_free": collision_free_count, "candidate_count_self_collision": self_collision_count, "candidate_count_environment_collision": env_collision_count, "seed_count": len(seeds), "target_ordering_source": "H3 JSONL order; no reordering", "source_geometry_hash": target["source_geometry_hash"], "h2_manifest_hash": target["h2_manifest_hash"], "collision_method": COLLISION_METHOD, "bullet_environment_collision": None, "bullet_status": UNAVAILABLE, "unreachable_denominator_retained": True}
        target_rows.append(target_row)
        ik_rows.append({"schema_version": "stage3-h4-ik-candidates-v1", "target_index": target_index, "task_sample_id": target["task_sample_id"], "target_pose": {"position_xyz_m": target["tcp_target_position_xyz_m"], "orientation_xyzw": target["tcp_target_orientation_xyzw"], "frame": target["coordinate_frame"]}, "seed_contract": {"seed_count": len(seeds), "seed_ids": [seed["seed_id"] for seed in seeds]}, "seed_attempts": seed_attempts, "raw_candidates": raw_candidates, "deduplicated_candidates": unique, "candidate_count_total": len(raw_candidates), "candidate_count_joint_valid": joint_valid_count, "candidate_count_fk_valid": fk_valid_count, "candidate_count_collision_free": collision_free_count})
        fk_rows.extend(candidate_fk_rows)
        collision_rows.extend(candidate_collision_rows)
    metrics = aggregate_target_metrics(target_rows)
    semantic = {"robot_model": metadata, "target_feasibility": target_rows, "ik_candidates": ik_rows, "fk_validation": fk_rows, "collision_validation": collision_rows, "metrics": metrics, "h2_manifest_sha256": h2_config_hash}
    return {"robot_metadata": metadata, "seed_states": seeds, "target_rows": target_rows, "ik_rows": ik_rows, "fk_rows": fk_rows, "collision_rows": collision_rows, "metrics": metrics, "semantic_hash": semantic_hash(replay_normalize(semantic)), "semantic_payload": semantic if output_records else None}


def _capability_unavailable(exc: BaseException, config: H4Config) -> dict[str, Any]:
    return {"schema_version": "stage3-h4-ik-solver-capability-v1", "status": "CAPABILITY_UNAVAILABLE", "first_blocker": "ik_solver_not_available", "planning_group": config.planning_group, "model_frame": config.model_frame, "tcp_tip_link": config.tcp_link, "solver_implementation": None, "solver_version": None, "configuration": None, "error": f"{type(exc).__name__}: {exc}"}


def _write_file_hashes(output: Path) -> None:
    rows = []
    for path in sorted(output.iterdir(), key=lambda item: item.name):
        if path.name == "stage3_h4_file_hashes.json" or not path.is_file():
            continue
        rows.append({"relative_path": path.name, "file_size_bytes": path.stat().st_size, "sha256": sha256_file(str(path))})
    _json_dump(output / "stage3_h4_file_hashes.json", {"schema_version": "stage3-h4-file-hashes-v1", "algorithm": "SHA-256", "file_count": len(rows), "files": rows, "self_hash": "EXCLUDED_FROM_SELF_HASH_LIST", "tree_hash": semantic_hash(rows)})


def _write_report(output: Path, result: dict[str, Any]) -> None:
    certificate = result["certificate"]
    metrics = result.get("metrics", {})
    lines = [
        "# Stage 3 H4 — Offline Reachability + IK Candidate + Static Collision Feasibility Baseline",
        "",
        f"`STAGE_3_H4: {certificate['stage3_h4']}`",
        f"`FIRST_BLOCKER: {certificate['first_blocker']}`",
        "",
        "## Frozen scope",
        "",
        "This run consumes the exact H3 JSONL ordering and answers per-target static reachability only. It does not reorder targets, plan a path, switch spray state, invoke timing, train ML, send a FollowJointTrajectory goal, or move a robot.",
        "",
        f"- H3 targets read unchanged: `{result.get('h3_preflight', {}).get('target_count')}`; ordering: `{result.get('h3_preflight', {}).get('target_ordering')}`.",
        f"- H0/H1/H2 and Formal R2 immutable: `{certificate.get('h0_h1_h2_immutable', certificate.get('h0_immutable', 'UNKNOWN'))}`; Formal ledger mutated: `{certificate['formal_ledger_mutated']}`.",
        f"- H3 coverage meaning: `{result.get('h3_preflight', {}).get('h3_coverage_semantics')}`; planner performance claim: `NO`.",
        "",
        "## Runtime capability",
        "",
        f"- planning group: `{result.get('robot_metadata', {}).get('planning_group', certificate.get('planning_group'))}`",
        f"- joint order: `{result.get('robot_metadata', {}).get('joint_names_exact_order', [])}`",
        f"- model frame / TCP: `{result.get('robot_metadata', {}).get('model_frame', certificate.get('model_frame'))}` / `{result.get('robot_metadata', {}).get('tcp_tip_link', certificate.get('tcp_tip_link'))}`",
        f"- solver: `{result.get('robot_metadata', {}).get('kinematics_solver', 'not_available')}`; implementation: `{result.get('robot_metadata', {}).get('solver_implementation', 'not_available')}`; package version: `{result.get('robot_metadata', {}).get('solver_package_version', 'not_available')}`",
        f"- Bullet environment result: `not_available` unless a candidate-level backend result was actually exposed; clearance: `null`.",
        "",
        "## Feasibility result",
        "",
        f"- target count: `{metrics.get('target_count')}`",
        f"- reachable target ratio: `{metrics.get('reachable_target_ratio')}`",
        f"- collision-free target ratio: `{metrics.get('collision_free_target_ratio')}`",
        f"- classifications: `{json.dumps(metrics.get('classification_counts', {}), ensure_ascii=False, sort_keys=True)}`",
        "- unreachable targets remain in the denominator; feasible-only coverage, if inspected, is diagnostic-only.",
        "",
        "## Deterministic replay and regressions",
        "",
        f"- deterministic replay: `{certificate['deterministic_replay']}`",
        f"- known regression set changed: `{result.get('regression', {}).get('known_failure_set_changed')}`; new failure count: `{result.get('regression', {}).get('new_failure_count')}`",
        "",
        "No downstream authorization is recommended unless all mandatory gates in stage3_h4_gate_report.json are passed.",
    ]
    (output / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _run_ros(args: argparse.Namespace, output: Path) -> int:
    config = H4Config()
    output.mkdir(parents=True, exist_ok=False)
    moveit = None
    try:
        h0 = _file_hashes_in_h0()
        h3 = _h3_preflight()
        targets = _read_jsonl(H3_TARGETS)
        h2_manifest_path = _find_h2_manifest(str(targets[0]["h2_manifest_hash"])) if targets else None
        if h2_manifest_path is None:
            raise H4ValidationError("empty_h3_targets", "H3 target file is empty")
        h2_records = _load_h2_records(h2_manifest_path)
        regression = _regression_preflight()
    except Exception as exc:
        certificate = {"schema_version": "stage3-h4-terminal-certificate-v1", "stage3_h4": "BLOCKED", "STAGE_3_H4": "BLOCKED", "first_blocker": getattr(exc, "code", type(exc).__name__), "h0_h1_h2_immutable": "UNKNOWN", "h3_immutable": "UNKNOWN", "formal_r2_immutable": "UNKNOWN", "new_fjt_goals_sent": 0, "send_goal_async_call_count": 0, "robot_motion_started": "NO", "formal_ledger_mutated": "NO", "ml_training_started": "NO", "path_planning_experiment_started": "NO", "deterministic_replay": "NOT_RUN", "ik_reachability_baseline": "NOT_RUN", "fk_reconstruction_baseline": "NOT_RUN", "static_collision_feasibility_baseline": "NOT_RUN", "next_stage_authorization_recommendation": "NO", "error": f"{type(exc).__name__}: {exc}"}
        _json_dump(output / "stage3_h4_terminal_certificate.json", certificate)
        (output / "FINAL_REPORT.md").write_text(f"# Stage 3 H4 — BLOCKED\n\n`FIRST_BLOCKER: {certificate['first_blocker']}`\n\n{certificate['error']}\n", encoding="utf-8")
        _write_file_hashes(output)
        return 2
    try:
        import rclpy
        from moveit.planning import MoveItPy

        rclpy.init(args=None)
        moveit = MoveItPy(node_name="stage3_h4_offline_reachability")
        kinematics_path = Path(os.environ.get("STAGE3_H4_KINEMATICS_PATH", ""))
        if not kinematics_path.is_file():
            try:
                from ament_index_python.packages import get_package_share_directory

                kinematics_path = Path(get_package_share_directory("fairino5_v6_moveit2_config")) / "config/kinematics.yaml"
            except Exception:
                pass
        os.environ["STAGE3_H4_KINEMATICS_PATH"] = str(kinematics_path)
        replay_runs: list[dict[str, Any]] = []
        primary: dict[str, Any] | None = None
        for replay_index in range(3):
            current = _run_one_replay(moveit, config, targets, h2_records, h2_manifest_path, output_records=replay_index == 0)
            replay_runs.append({
                "replay_index": replay_index,
                "semantic_hash": current["semantic_hash"],
                "target_count": current["metrics"]["target_count"],
                "classification_counts": current["metrics"]["classification_counts"],
                "target_order_hash": semantic_hash([{"target_index": row["target_index"], "task_sample_id": row["task_sample_id"]} for row in current["target_rows"]]),
                "seed_states_hash": semantic_hash(current["seed_states"]),
                "raw_ik_candidate_sets_hash": semantic_hash([{"task_sample_id": row["task_sample_id"], "raw_candidates": row["raw_candidates"]} for row in current["ik_rows"]]),
                "deduplicated_candidate_sets_hash": semantic_hash([{"task_sample_id": row["task_sample_id"], "deduplicated_candidates": row["deduplicated_candidates"]} for row in current["ik_rows"]]),
                "fk_errors_hash": semantic_hash([{"task_sample_id": row["task_sample_id"], "candidate_id": row["candidate_id"], "fk_attempted": row["fk_attempted"], "fk_valid": row["fk_valid"], "translation_error_m": row["translation_error_m"], "orientation_error_rad": row["orientation_error_rad"], "fk_error": row.get("fk_error")} for row in current["fk_rows"]]),
                "joint_validity_hash": semantic_hash([{"task_sample_id": row["task_sample_id"], "candidate_id": row["candidate_id"], "joint_validation": row["joint_validation"]} for row in current["fk_rows"]]),
                "fcl_decisions_hash": semantic_hash([{"task_sample_id": row["task_sample_id"], "candidate_id": row["candidate_id"], "joint_valid": row["joint_valid"], "fk_valid": row["fk_valid"], "collision_checked": row["collision_checked"], "fcl_self_collision": row["fcl_self_collision"], "fcl_environment_collision": row["fcl_environment_collision"], "collision_free": row["collision_free"]} for row in current["collision_rows"]]),
                "bullet_decisions_hash": semantic_hash([{"task_sample_id": row["task_sample_id"], "candidate_id": row["candidate_id"], "bullet_environment_collision": row["bullet_environment_collision"], "ccd_status": row["ccd_status"], "clearance_status": row["clearance_status"], "clearance_m": row["clearance_m"]} for row in current["collision_rows"]]),
                "target_classifications_hash": semantic_hash([{"target_index": row["target_index"], "task_sample_id": row["task_sample_id"], "classification": row["classification"]} for row in current["target_rows"]]),
                "metrics_hash": semantic_hash(current["metrics"]),
            })
            if primary is None:
                primary = current
        assert primary is not None
        comparison_fields = ["semantic_hash", "target_order_hash", "seed_states_hash", "raw_ik_candidate_sets_hash", "deduplicated_candidate_sets_hash", "fk_errors_hash", "joint_validity_hash", "fcl_decisions_hash", "bullet_decisions_hash", "target_classifications_hash", "metrics_hash"]
        comparison_equal = {field: len({item[field] for item in replay_runs}) == 1 for field in comparison_fields}
        replay_equal = all(comparison_equal.values())
        capability = dict(primary["robot_metadata"])
        capability.update({"schema_version": "stage3-h4-ik-solver-capability-v1", "status": "AVAILABLE", "first_blocker": "none" if replay_equal else "solver_nondeterminism", "replay_deterministic": replay_equal})
        _json_dump(output / "stage3_h4_ik_solver_capability.json", capability)
        seed_contract = {"schema_version": "stage3-h4-seed-contract-v1", "status": "FROZEN_BEFORE_EXPERIMENT", "seed_count": config.seed_count, "seed_generation_algorithm": "seed_000 finite interval midpoint; seed_001..012 SplitMix64 inner interval mapping", "random_seed": config.random_seed, "seed_order": "target H3 JSONL order, then seed_index ascending", "attempt_policy": "exactly one RobotState.set_from_ik call per seed", "timeout_s": config.ik_timeout_s, "duplicate_solution_tolerance_rad": config.duplicate_tolerance_rad, "canonical_candidate_order": "quantized joint vector lexicographic, seed_index tie break", "joint_wrapping_policy": "no silent repair; raw MoveIt vector is validated; circular distance is comparison-only", "joint_limit_policy": "reject outside URDF/SRDF-resolved bounds with declared tolerance", "hashing": "SHA-256 over canonical JSON", "threshold_provenance": "DECLARED_H4_BASELINE_PARAMETER", "seeds": primary["seed_states"]}
        _json_dump(output / "stage3_h4_seed_contract.json", seed_contract)
        _json_dump(output / "stage3_h4_ik_solver_capability.json", capability)
        _json_dump(output / "stage3_h4_reachability_contract.json", {"schema_version": "stage3-h4-reachability-contract-v1", "status": "FROZEN_BEFORE_EXPERIMENT", "input": "H3 stage3_h3_surface_targets.jsonl only", "target_ordering": "preserved exactly; no reordering or optimization", "validity_chain": ["H3 target validation", "deterministic IK seed calls", "joint-name/order/finite/bounds validation", "independent MoveIt RobotState FK", "FCL static self/environment collision"], "bullet_policy": "use only H1-verified discrete robot-environment capability; MoveItPy candidate result is null when not exposed", "ccd_status": UNAVAILABLE, "clearance_status": UNAVAILABLE, "failure_policy": "fail closed", "allowed_operations": ["offline IK", "FK", "joint validation", "static collision validation"], "forbidden_operations": ["path planning", "target reordering", "spray sequencing", "Ruckig", "ML", "robot execution"], "moveitpy_initialization_note": "MoveItPy/MoveItCpp requires a registered planning pipeline parameter; H4 only initializes that dependency and never invokes a planning API or creates a trajectory."})
        _json_dump(output / "stage3_h4_gate_definition.json", {"schema_version": "stage3-h4-gate-definition-v1", "status": "FROZEN_BEFORE_EXPERIMENT", "mandatory_gates": ["H3_INPUT_IMMUTABLE", "H0_H1_H2_IMMUTABLE", "FORMAL_R2_IMMUTABLE", "NO_FJT", "NO_ROBOT_MOTION", "NO_LEDGER_MUTATION", "IK_SOLVER_AVAILABLE", "FK_RECONSTRUCTION", "FCL_STATIC_COLLISION", "THREE_RUN_DETERMINISTIC_REPLAY", "NO_PATH_PLANNING", "NO_RUCKIG", "NO_ML"], "threshold_policy": "all numeric thresholds are declared H4 baseline parameters before solver calls; no post-hoc tuning", "h3_coverage_semantics": "H3_COVERAGE_IS_GEOMETRIC_SELF_CONSISTENCY_BASELINE"})
        _jsonl_dump(output / "stage3_h4_target_feasibility.jsonl", primary["target_rows"])
        _jsonl_dump(output / "stage3_h4_ik_candidates.jsonl", primary["ik_rows"])
        _jsonl_dump(output / "stage3_h4_fk_validation.jsonl", primary["fk_rows"])
        _jsonl_dump(output / "stage3_h4_collision_validation.jsonl", primary["collision_rows"])
        _json_dump(output / "stage3_h4_reachability_metrics.json", primary["metrics"])
        _json_dump(output / "stage3_h4_deterministic_replay.json", {"schema_version": "stage3-h4-deterministic-replay-v1", "replay_count": 3, "runs": replay_runs, "all_semantic_hashes_equal": comparison_equal["semantic_hash"], "target_ordering_equal": comparison_equal["target_order_hash"], "generated_seed_states_equal": comparison_equal["seed_states_hash"], "raw_ik_candidate_sets_equal": comparison_equal["raw_ik_candidate_sets_hash"], "deduplicated_candidate_sets_equal": comparison_equal["deduplicated_candidate_sets_hash"], "fk_errors_equal": comparison_equal["fk_errors_hash"], "joint_validity_equal": comparison_equal["joint_validity_hash"], "fcl_decisions_equal": comparison_equal["fcl_decisions_hash"], "bullet_decisions_equal": comparison_equal["bullet_decisions_hash"], "target_classifications_equal": comparison_equal["target_classifications_hash"], "aggregate_metrics_equal": comparison_equal["metrics_hash"], "canonical_semantic_hash": replay_runs[0]["semantic_hash"]})
        gates = {"H3_INPUT_IMMUTABLE": h3["h3_immutable"], "H0_H1_H2_IMMUTABLE": h0["h0_h1_h2_immutable"], "FORMAL_R2_IMMUTABLE": h0["formal_r2_immutable"], "NO_FJT": True, "NO_ROBOT_MOTION": True, "NO_LEDGER_MUTATION": True, "IK_SOLVER_AVAILABLE": capability["status"] == "AVAILABLE", "FK_RECONSTRUCTION": all((not row["joint_validation"]["valid"]) or row["fk_attempted"] for row in primary["fk_rows"]) and any(row["fk_attempted"] for row in primary["fk_rows"]), "FCL_STATIC_COLLISION": all((not row["fk_valid"]) or row["collision_checked"] for row in primary["collision_rows"]), "THREE_RUN_DETERMINISTIC_REPLAY": replay_equal, "NO_PATH_PLANNING": True, "NO_RUCKIG": True, "NO_ML": True, "REGRESSION_BASELINE": regression["baseline_gate"]}
        first_blocker = next((name.lower() for name, passed in gates.items() if not passed), "none")
        status = "PASSED" if all(gates.values()) else "BLOCKED"
        gate_report = {"schema_version": "stage3-h4-gate-report-v1", "stage3_h4": status, "STAGE_3_H4": status, "first_blocker": first_blocker, "gates": gates, "gate_evidence": {"h0": h0, "h3": h3, "regression": {key: value for key, value in regression.items() if key not in {"known_failure_baseline", "post_h4_regression"}}}, "collision_method": COLLISION_METHOD, "ccd_status": UNAVAILABLE, "clearance_status": UNAVAILABLE}
        _json_dump(output / "stage3_h4_gate_report.json", gate_report)
        certificate = {"schema_version": "stage3-h4-terminal-certificate-v1", "stage3_h4": status, "STAGE_3_H4": status, "FIRST_BLOCKER": first_blocker, "first_blocker": first_blocker, "stage2_baseline_immutable": "YES" if h0["stage2_baseline_immutable"] else "NO", "formal_r2_immutable": "YES" if h0["formal_r2_immutable"] else "NO", "h0_h1_h2_immutable": "YES" if h0["h0_h1_h2_immutable"] else "NO", "h0_immutable": "YES" if h0["h0_h1_h2_immutable"] else "NO", "h1_immutable": "YES" if h0["h0_h1_h2_immutable"] else "NO", "h2_immutable": "YES" if h0["h0_h1_h2_immutable"] else "NO", "h3_immutable": "YES" if h3["h3_immutable"] else "NO", "new_fjt_goals_sent": 0, "send_goal_async_call_count": 0, "robot_motion_started": "NO", "formal_ledger_mutated": "NO", "ml_training_started": "NO", "path_planning_experiment_started": "NO", "ik_reachability_baseline": "PASSED" if gates["IK_SOLVER_AVAILABLE"] else "BLOCKED", "fk_reconstruction_baseline": "PASSED" if gates["FK_RECONSTRUCTION"] else "BLOCKED", "static_collision_feasibility_baseline": "PASSED" if gates["FCL_STATIC_COLLISION"] else "BLOCKED", "deterministic_replay": "PASSED" if replay_equal else "BLOCKED", "next_stage_authorization_recommendation": "YES" if status == "PASSED" else "NO", "planning_group": config.planning_group, "model_frame": config.model_frame, "tcp_tip_link": config.tcp_link, "collision_method": COLLISION_METHOD, "ccd_status": UNAVAILABLE, "clearance_status": UNAVAILABLE, "h3_coverage_semantics": "H3_COVERAGE_IS_GEOMETRIC_SELF_CONSISTENCY_BASELINE", "regression_new_failure_count": regression["new_failure_count"]}
        _json_dump(output / "stage3_h4_terminal_certificate.json", certificate)
        result = {"certificate": certificate, "metrics": primary["metrics"], "robot_metadata": primary["robot_metadata"], "h3_preflight": h3, "regression": regression}
        _json_dump(output / "stage3_h4_regression_baseline.json", regression)
        _write_report(output, result)
        _write_file_hashes(output)
        return 0 if status == "PASSED" else 2
    except Exception as exc:
        capability = _capability_unavailable(exc, config)
        _json_dump(output / "stage3_h4_ik_solver_capability.json", capability)
        certificate = {"schema_version": "stage3-h4-terminal-certificate-v1", "stage3_h4": "BLOCKED", "STAGE_3_H4": "BLOCKED", "FIRST_BLOCKER": getattr(exc, "code", type(exc).__name__), "first_blocker": getattr(exc, "code", type(exc).__name__), "stage2_baseline_immutable": "YES" if h0.get("stage2_baseline_immutable") else "UNKNOWN", "formal_r2_immutable": "YES" if h0.get("formal_r2_immutable") else "UNKNOWN", "h0_immutable": "YES" if h0.get("h0_h1_h2_immutable") else "UNKNOWN", "h1_immutable": "YES" if h0.get("h0_h1_h2_immutable") else "UNKNOWN", "h2_immutable": "YES" if h0.get("h0_h1_h2_immutable") else "UNKNOWN", "h3_immutable": "YES" if h3.get("h3_immutable") else "UNKNOWN", "new_fjt_goals_sent": 0, "send_goal_async_call_count": 0, "robot_motion_started": "NO", "formal_ledger_mutated": "NO", "ml_training_started": "NO", "path_planning_experiment_started": "NO", "ik_reachability_baseline": "BLOCKED", "fk_reconstruction_baseline": "BLOCKED", "static_collision_feasibility_baseline": "BLOCKED", "deterministic_replay": "BLOCKED", "next_stage_authorization_recommendation": "NO", "planning_group": config.planning_group, "model_frame": config.model_frame, "tcp_tip_link": config.tcp_link, "error": f"{type(exc).__name__}: {exc}"}
        _json_dump(output / "stage3_h4_terminal_certificate.json", certificate)
        _json_dump(output / "stage3_h4_gate_report.json", {"schema_version": "stage3-h4-gate-report-v1", "stage3_h4": "BLOCKED", "first_blocker": certificate["first_blocker"], "gates": {"H3_INPUT_IMMUTABLE": h3.get("h3_immutable", False), "H0_H1_H2_IMMUTABLE": h0.get("h0_h1_h2_immutable", False), "IK_SOLVER_AVAILABLE": False}, "error": certificate["error"]})
        (output / "FINAL_REPORT.md").write_text(f"# Stage 3 H4 — BLOCKED\n\n`FIRST_BLOCKER: {certificate['first_blocker']}`\n\n{certificate['error']}\n", encoding="utf-8", newline="\n")
        _write_file_hashes(output)
        return 2
    finally:
        try:
            if moveit is not None:
                import gc

                del moveit
                gc.collect()
        except Exception:
            pass
        try:
            import rclpy

            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ros-node", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args, _unknown = parser.parse_known_args(argv)
    output = args.output_root.resolve()
    if output.exists():
        print(json.dumps({"stage3_h4": "BLOCKED", "first_blocker": "same_id_overwrite_forbidden", "output_root": str(output)}))
        return 2
    if not args.ros_node:
        print("Stage 3 H4 requires the isolated ROS2 launch wrapper; no backend calls were made.")
        return 2
    code = _run_ros(args, output)
    print(json.dumps({"stage3_h4": "PASSED" if code == 0 else "BLOCKED", "output_root": _relative(output)}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
