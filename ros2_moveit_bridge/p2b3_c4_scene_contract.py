"""Authoritative C1 scene contract extraction and deterministic comparison."""

from __future__ import annotations

import ast
import hashlib
import itertools
import json
import math
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT = "FAIRINO_FR5"
STAGE = "P2-B3-C4"
SCHEMA_VERSION = "p2b3-c4-authoritative-scene-v1"
POSE_TOLERANCE = 1.0e-10
DIMENSION_TOLERANCE = 1.0e-12
DISTANCE_SENTINEL = -1.0


def _git_output(repo_root: Path, *arguments: str) -> str:
    try:
        return subprocess.run(
            ["git", *arguments], cwd=repo_root, check=True, capture_output=True, text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as first_error:
        try:
            win_root = subprocess.run(
                ["wslpath", "-w", str(repo_root)], check=True, capture_output=True, text=True,
            ).stdout.strip()
            return subprocess.run(
                ["git.exe", "-C", win_root, *arguments], check=True, capture_output=True, text=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError) as second_error:
            raise ValueError("git_source_identity_unavailable_in_current_worktree_runtime") from second_error


@dataclass(frozen=True)
class C1SceneContract:
    project: str
    source_stage: str
    source_execution_commit: str
    nominal_trajectory_sha256: str
    waypoint_count: int
    expected_sample_count: int
    expected_collision_count: int
    frame_id: str
    parameters: Mapping[str, Any]
    sampling: Mapping[str, Any]
    builder: str = "ros2_moveit_bridge.plan_closed_contour_moveit.build_wall_collision_objects"


_EXPORTED_SCENE_VALUES = {
    "STAND_OFF": "stand_off_m",
    "TCP_POINTS_TO_WALL": "tcp_points_to_wall",
    "TUNNEL_WALL_THICKNESS": "wall_thickness_m",
    "TUNNEL_Y_THICKNESS": "y_thickness_m",
    "COLLISION_SEGMENT_STRIDE": "segment_stride",
    "OPEN_PATH": "open_path",
    "INCLUDE_BOTTOM_CLOSURE_COLLISION": "include_bottom_closure",
    "INCLUDE_TUNNEL_FLOOR_COLLISION": "include_tunnel_floor",
    "TUNNEL_FLOOR_Z": "tunnel_floor_z_m",
    "COLLISION_CHECK_STRIDE": "collision_check_stride",
    "COLLISION_INTERPOLATION_STEP_DEG": "collision_interpolation_step_deg",
}


def _literal_exports(source: str) -> dict[str, Any]:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(target, ast.Name) and target.id == "exports" for target in targets):
            continue
        value = node.value
        if not isinstance(value, ast.Dict):
            raise ValueError("C1_exports_must_be_a_literal_dictionary")
        result: dict[str, Any] = {}
        for key_node, value_node in zip(value.keys, value.values):
            if not isinstance(key_node, ast.Constant) or not isinstance(key_node.value, str):
                continue
            key = key_node.value
            if key in _EXPORTED_SCENE_VALUES or key in {
                "SAMPLES_PER_LOOP", "VALIDATE_COLLISION", "COLLISION_METHOD"
            }:
                try:
                    result[key] = ast.literal_eval(value_node)
                except (ValueError, TypeError):
                    continue
        return result
    raise ValueError("C1_execution_script_exports_not_found")


def _strict_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise ValueError(f"C1_scene_boolean_invalid:{name}")


def _constant_string(source: str, name: str) -> str:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            if isinstance(value, str):
                return value
    raise ValueError(f"C1_source_constant_not_found:{name}")


def _node_parameter_default(source: str, parameter_name: str) -> str:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "declare_parameter" or len(node.args) < 2:
            continue
        if not isinstance(node.args[0], ast.Constant) or node.args[0].value != parameter_name:
            continue
        if isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str):
            return node.args[1].value
    raise ValueError(f"C1_parameter_default_not_found:{parameter_name}")


def extract_c1_scene_contract(
    c1_result: Mapping[str, Any],
    c1_runner_source: str,
    planner_source: str,
) -> C1SceneContract:
    """Read effective scene and sampling values from the frozen C1 run path."""
    if c1_result.get("PROJECT") != PROJECT or c1_result.get("STAGE") != "P2-B3-C1":
        raise ValueError("foreign_or_non_C1_project_result_rejected")
    execution_commit = str(c1_result.get("C1_EXECUTION_CODE_COMMIT", ""))
    nominal = c1_result.get("C1_POST_RUCKIG_NOMINAL", {})
    trajectory_sha = str(nominal.get("sha256", "")) if isinstance(nominal, Mapping) else ""
    if len(execution_commit) != 40 or len(trajectory_sha) != 64:
        raise ValueError("C1_authority_identity_missing_or_malformed")
    c1_measurement = c1_result.get("c1_measurement", {})
    summary = c1_measurement.get("summary_metrics", {}) if isinstance(c1_measurement, Mapping) else {}
    try:
        expected_samples = int(float(summary["collision_checked_state_count"]))
        expected_collisions = int(float(summary["collision_count"]))
    except (KeyError, TypeError, ValueError):
        raise ValueError("C1_collision_baseline_missing") from None
    exports = _literal_exports(c1_runner_source)
    params: dict[str, Any] = {}
    for source_name, target_name in _EXPORTED_SCENE_VALUES.items():
        if source_name not in exports:
            raise ValueError(f"C1_scene_parameter_missing:{source_name}")
        value = exports[source_name]
        if target_name in {
            "tcp_points_to_wall", "open_path", "include_bottom_closure", "include_tunnel_floor"
        }:
            params[target_name] = _strict_bool(value, source_name)
        elif target_name == "segment_stride" or target_name == "collision_check_stride":
            params[target_name] = int(value)
        else:
            params[target_name] = float(value)
    samples_per_loop = int(exports.get("SAMPLES_PER_LOOP", 0))
    if samples_per_loop <= 0:
        raise ValueError("C1_waypoint_count_missing_from_execution_script")
    if c1_measurement.get("collision_method") != "adaptive_discrete_interpolation":
        raise ValueError("C1_collision_method_mismatch")
    params["collision_method"] = "adaptive_discrete_interpolation"
    interpolation = params.pop("collision_interpolation_step_deg")
    sampling = {
        "policy": "adaptive_discrete_interpolation",
        "maximum_joint_interpolation_step_deg": interpolation,
        "collision_check_stride": params["collision_check_stride"],
        "waypoint_count": samples_per_loop,
    }
    frame_id = _node_parameter_default(planner_source, "base_frame")
    if frame_id != "base_link":
        raise ValueError("C1_scene_frame_not_base_link")
    return C1SceneContract(
        project=PROJECT,
        source_stage="P2-B3-C1",
        source_execution_commit=execution_commit,
        nominal_trajectory_sha256=trajectory_sha,
        waypoint_count=samples_per_loop,
        expected_sample_count=expected_samples,
        expected_collision_count=expected_collisions,
        frame_id=frame_id,
        parameters=params,
        sampling=sampling,
    )


def verify_frozen_c1_source_blobs(repo_root: Path, source_commit: str) -> dict[str, str]:
    """Require the C1 runner and shared builder files to match C1's execution commit."""
    files = (
        "scripts/run_p2b3_c1_r0.py",
        "ros2_moveit_bridge/plan_closed_contour_moveit.py",
    )
    identities: dict[str, str] = {}
    for relative in files:
        try:
            authority = _git_output(repo_root, "rev-parse", f"{source_commit}:{relative}").strip()
            current = _git_output(repo_root, "rev-parse", f"HEAD:{relative}").strip()
        except ValueError as exc:
            raise ValueError(f"C1_scene_source_blob_unavailable:{relative}") from exc
        if authority != current:
            raise ValueError(f"frozen_C1_scene_source_blob_mismatch:{relative}")
        identities[relative] = authority
    return identities


def extract_legacy_c3_scene_contract(repo_root: Path, c3_result: Mapping[str, Any]) -> dict[str, Any]:
    """Read the pre-repair C3 builder arguments from its recorded execution commit."""
    if c3_result.get("PROJECT") != PROJECT or c3_result.get("STAGE") != "P2-B3-C3":
        raise ValueError("foreign_or_non_C3_project_result_rejected")
    commit = str(c3_result.get("EXECUTION_CODE_COMMIT", ""))
    if len(commit) != 40:
        raise ValueError("C3_execution_commit_missing_or_malformed")
    relative = "ros2_moveit_bridge/p2b3_c3_native_smoke.py"
    try:
        source = _git_output(repo_root, "show", f"{commit}:{relative}")
        planner_source = _git_output(repo_root, "show", f"{commit}:ros2_moveit_bridge/plan_closed_contour_moveit.py")
    except ValueError as exc:
        raise ValueError("C3_historical_execution_source_unavailable") from exc
    tree = ast.parse(source)
    base_frame = _node_parameter_default(planner_source, "base_frame")
    selected: dict[str, Any] | None = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
        if name != "build_wall_collision_objects":
            continue
        values: dict[str, Any] = {}
        for keyword in node.keywords:
            try:
                values[keyword.arg] = ast.literal_eval(keyword.value)
            except (ValueError, TypeError):
                if keyword.arg == "frame_id" and isinstance(keyword.value, ast.Name) and keyword.value.id == "BASE_FRAME":
                    values[keyword.arg] = base_frame
        if "stand_off" in values and "segment_stride" in values:
            selected = values
            break
    if selected is None:
        raise ValueError("C3_historical_scene_builder_call_unavailable")
    required = (
        "stand_off", "frame_id", "wall_thickness", "y_thickness", "segment_stride",
        "include_bottom_closure", "open_path", "tcp_points_to_wall", "include_tunnel_floor", "tunnel_floor_z",
    )
    missing = [key for key in required if key not in selected]
    if missing:
        raise ValueError(f"C3_historical_scene_parameters_missing:{','.join(missing)}")
    return {key: selected[key] for key in required} | {"execution_code_commit": commit, "source_path": relative}


def compare_c3_to_c1_parameters(legacy: Mapping[str, Any], c1: C1SceneContract) -> dict[str, Any]:
    pairs = {
        "stand_off_m": ("stand_off", "stand_off_m"),
        "tcp_points_to_wall": ("tcp_points_to_wall", "tcp_points_to_wall"),
        "wall_thickness_m": ("wall_thickness", "wall_thickness_m"),
        "y_thickness_m": ("y_thickness", "y_thickness_m"),
        "segment_stride": ("segment_stride", "segment_stride"),
        "open_path": ("open_path", "open_path"),
        "include_bottom_closure": ("include_bottom_closure", "include_bottom_closure"),
        "include_tunnel_floor": ("include_tunnel_floor", "include_tunnel_floor"),
        "tunnel_floor_z_m": ("tunnel_floor_z", "tunnel_floor_z_m"),
    }
    differences = []
    for name, (old_key, new_key) in pairs.items():
        old_value, new_value = legacy.get(old_key), c1.parameters.get(new_key)
        if old_value != new_value:
            differences.append({"parameter": name, "c3_observed": old_value, "c1_authoritative": new_value})
    return {"status": "DRIFT_FOUND" if differences else "NO_DRIFT", "differences": differences}


def validate_phase_d_identity_gate(record: Mapping[str, Any], expected_manifest_sha256: str) -> Mapping[str, Any]:
    if record.get("project") != PROJECT or record.get("stage") != STAGE:
        raise ValueError("phase_d_foreign_project_gate_rejected")
    if record.get("zero_transform_equivalence", {}).get("status") != "PASS":
        raise ValueError("phase_d_rejected_without_zero_transform_pass")
    if record.get("scene_manifest_sha256") != expected_manifest_sha256:
        raise ValueError("phase_d_scene_manifest_differs_from_phase_c")
    baseline = record.get("baseline")
    if not isinstance(baseline, Mapping) or int(baseline.get("sample_count", 0)) <= 0:
        raise ValueError("phase_d_identity_gate_baseline_missing")
    return baseline


def fingerprint_active_scene_semantics(
    moveit: Any,
    world_object_names: Sequence[str],
    urdf_path: Path,
) -> dict[str, Any]:
    """Fingerprint the active ACM through the observed MoveItPy getter surface."""
    try:
        robot_names = sorted({str(node.get("name")) for node in ET.parse(urdf_path).getroot().findall("link") if node.get("name")})
    except (OSError, ET.ParseError) as exc:
        return {
            "acm": {"status": "NOT_AVAILABLE", "reason": f"robot_link_names_unavailable:{type(exc).__name__}"},
            "robot_padding_scale": {"status": "NOT_AVAILABLE", "reason": "padding_binding_not_inspected"},
            "api_surface": {},
        }
    names = sorted(set(robot_names) | {str(name) for name in world_object_names})
    monitor = moveit.get_planning_scene_monitor()
    with monitor.read_only() as scene:
        scene_methods = sorted(
            name for name in dir(scene)
            if any(token in name.lower() for token in ("allowed", "padding", "scale", "acm", "world", "detector"))
        )
        model = moveit.get_robot_model()
        model_methods = sorted(
            name for name in dir(model)
            if any(token in name.lower() for token in ("padding", "scale", "link_model", "collision"))
        )
        api_surface = {
            "planning_scene_type": f"{type(scene).__module__}.{type(scene).__qualname__}",
            "planning_scene_acm_padding_world_methods": scene_methods,
            "robot_model_padding_collision_methods": model_methods,
        }
        try:
            acm = scene.allowed_collision_matrix
        except Exception as exc:
            acm = None
            acm_error = f"allowed_collision_matrix_property_error:{type(exc).__name__}"
        else:
            acm_error = "allowed_collision_matrix_property_missing"
        getter = getattr(acm, "get_entry", None) if acm is not None else None
        if not callable(getter):
            acm_record = {"status": "NOT_AVAILABLE", "reason": acm_error if acm is None else "ACM_get_entry_method_missing"}
        else:
            pair_records: list[list[Any]] = []
            allowed_pair_count = 0
            try:
                for left, right in itertools.product(names, repeat=2):
                    entry = getter(left, right)
                    if not isinstance(entry, (tuple, list)) or len(entry) != 2:
                        raise TypeError("AllowedCollisionMatrix.get_entry_did_not_return_bool_and_type")
                    allowed, entry_type = bool(entry[0]), str(entry[1])
                    allowed_pair_count += int(allowed)
                    pair_records.append([left, right, allowed, entry_type])
            except Exception as exc:
                acm_record = {
                    "status": "NOT_AVAILABLE",
                    "reason": f"AllowedCollisionMatrix.get_entry_query_failed:{type(exc).__name__}:{exc}",
                    "queried_pair_count": len(pair_records),
                    "universe_name_count": len(names),
                }
            else:
                fingerprint = hashlib.sha256(canonical_json_bytes(pair_records)).hexdigest()
                acm_record = {
                    "status": "AVAILABLE",
                    "api": "PlanningScene.allowed_collision_matrix -> AllowedCollisionMatrix.get_entry(name1,name2)",
                    "scope": "all ordered pairs among URDF robot links and current authoritative world-object IDs",
                    "universe_name_count": len(names),
                    "queried_pair_count": len(pair_records),
                    "allowed_pair_count": allowed_pair_count,
                    "sha256": fingerprint,
                }
        if any("padding" in name.lower() or "scale" in name.lower() for name in scene_methods + model_methods):
            padding_record = {
                "status": "NOT_AVAILABLE",
                "reason": "padding_or_scale_name_exposed_but_no_semantics_verified",
                "exposed_methods": sorted(name for name in scene_methods + model_methods if "padding" in name.lower() or "scale" in name.lower()),
            }
        else:
            padding_record = {
                "status": "NOT_AVAILABLE",
                "reason": "PlanningScene and RobotModel Python bindings expose no padding or scale accessor",
            }
    return {"acm": acm_record, "robot_padding_scale": padding_record, "api_surface": api_surface}


def _finite_float(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"nonfinite_scene_value:{label}")
    return number


def _stable_float(value: Any, label: str) -> float:
    number = _finite_float(value, label)
    rounded = float(format(number, ".12g"))
    return 0.0 if rounded == 0.0 else rounded


def canonical_quaternion_xyzw(values: Sequence[Any]) -> list[float]:
    if len(values) != 4:
        raise ValueError("scene_quaternion_requires_xyzw")
    q = [_finite_float(value, "quaternion") for value in values]
    norm = math.sqrt(sum(value * value for value in q))
    if norm <= 1.0e-15:
        raise ValueError("scene_quaternion_norm_is_zero")
    q = [value / norm for value in q]
    first_nonzero = next((value for value in q if abs(value) > 1.0e-15), 1.0)
    if first_nonzero < 0.0:
        q = [-value for value in q]
    return [_stable_float(value, "quaternion") for value in q]


def canonical_scene_objects(objects: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in objects:
        object_id = str(raw.get("id", ""))
        if not object_id or object_id in seen:
            raise ValueError("scene_object_id_missing_or_duplicate")
        seen.add(object_id)
        frame = str(raw.get("frame_id", ""))
        primitives: list[dict[str, Any]] = []
        for primitive in raw.get("primitives", []):
            pose = primitive.get("pose", {})
            position = pose.get("position_xyz", [])
            orientation = pose.get("orientation_xyzw", [])
            if len(position) != 3:
                raise ValueError(f"scene_position_requires_xyz:{object_id}")
            dimensions = [_stable_float(value, "dimension") for value in primitive.get("dimensions", [])]
            primitives.append({
                "type": str(primitive.get("type", "")),
                "dimensions": dimensions,
                "pose": {
                    "position_xyz": [_stable_float(value, "position") for value in position],
                    "orientation_xyzw": canonical_quaternion_xyzw(orientation),
                },
            })
        result.append({"id": object_id, "frame_id": frame, "primitives": primitives})
    return sorted(result, key=lambda item: item["id"])


def scene_objects_from_moveit(objects: Sequence[Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for obj in objects:
        primitives: list[dict[str, Any]] = []
        for primitive, pose in zip(obj.primitives, obj.primitive_poses):
            primitive_type = int(primitive.type)
            primitives.append({
                "type": "BOX" if primitive_type == 1 else f"PRIMITIVE_{primitive_type}",
                "dimensions": [float(value) for value in primitive.dimensions],
                "pose": {
                    "position_xyz": [float(pose.position.x), float(pose.position.y), float(pose.position.z)],
                    "orientation_xyzw": [float(pose.orientation.x), float(pose.orientation.y), float(pose.orientation.z), float(pose.orientation.w)],
                },
            })
        records.append({"id": str(obj.id), "frame_id": str(obj.header.frame_id), "primitives": primitives})
    return canonical_scene_objects(records)


def build_scene_manifest(
    contract: C1SceneContract,
    objects: Sequence[Mapping[str, Any]],
    *,
    acm: Mapping[str, Any] | None = None,
    padding: Mapping[str, Any] | None = None,
    backend: str = "MoveIt2 Jazzy MoveItPy PlanningSceneMonitor / FCL",
) -> dict[str, Any]:
    scene_objects = canonical_scene_objects(objects)
    return {
        "schema_version": SCHEMA_VERSION,
        "project": PROJECT,
        "stage": STAGE,
        "source_c1": {
            "stage": contract.source_stage,
            "execution_code_commit": contract.source_execution_commit,
            "nominal_trajectory_sha256": contract.nominal_trajectory_sha256,
        },
        "frame_id": contract.frame_id,
        "builder": contract.builder,
        "effective_parameters": dict(contract.parameters),
        "semantics": {
            "open_path": contract.parameters["open_path"],
            "bottom_closure_included": contract.parameters["include_bottom_closure"],
            "floor_included": contract.parameters["include_tunnel_floor"],
            "floor_z_m": contract.parameters["tunnel_floor_z_m"],
            "tcp_points_to_wall": contract.parameters["tcp_points_to_wall"],
        },
        "collision_backend": backend,
        "collision_objects": {"count": len(scene_objects), "ordered": scene_objects},
        "acm": dict(acm or {"status": "NOT_AVAILABLE", "reason": "not_read_from_active_MoveIt_binding"}),
        "robot_padding_scale": dict(padding or {"status": "NOT_AVAILABLE", "reason": "not_read_from_active_MoveIt_binding"}),
        "sampling": dict(contract.sampling),
        "expected_c1_baseline": {
            "waypoint_count": contract.waypoint_count,
            "sampled_state_count": contract.expected_sample_count,
            "collision_count": contract.expected_collision_count,
        },
        "canonicalization": {
            "float_encoding": "finite IEEE-754 values rounded to 12 significant decimal digits",
            "quaternion": "normalized xyzw with first nonzero component positive; q and -q canonicalize identically",
            "object_order": "ascending object id; primitive order retained",
        },
    }


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def manifest_sha256(manifest: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()


def compare_scene_manifests(
    expected: Mapping[str, Any], observed: Mapping[str, Any],
    *, pose_tolerance: float = POSE_TOLERANCE, dimension_tolerance: float = DIMENSION_TOLERANCE,
) -> dict[str, Any]:
    differences: list[dict[str, Any]] = []

    def mismatch(path: str, left: Any, right: Any) -> None:
        differences.append({"path": path, "expected": left, "observed": right})

    for field in ("project", "frame_id", "builder"):
        if expected.get(field) != observed.get(field):
            mismatch(field, expected.get(field), observed.get(field))
    if expected.get("effective_parameters") != observed.get("effective_parameters"):
        left = expected.get("effective_parameters", {})
        right = observed.get("effective_parameters", {})
        for key in sorted(set(left) | set(right)):
            if left.get(key) != right.get(key):
                mismatch(f"effective_parameters.{key}", left.get(key), right.get(key))

    for field in ("sampling", "acm", "robot_padding_scale", "runtime_api_surface"):
        if expected.get(field) != observed.get(field):
            mismatch(field, expected.get(field), observed.get(field))

    left_objects = {item["id"]: item for item in expected.get("collision_objects", {}).get("ordered", [])}
    right_objects = {item["id"]: item for item in observed.get("collision_objects", {}).get("ordered", [])}
    for object_id in sorted(set(left_objects) - set(right_objects)):
        mismatch(f"collision_objects.{object_id}", left_objects[object_id], None)
    for object_id in sorted(set(right_objects) - set(left_objects)):
        mismatch(f"collision_objects.{object_id}", None, right_objects[object_id])
    for object_id in sorted(set(left_objects) & set(right_objects)):
        left, right = left_objects[object_id], right_objects[object_id]
        if left.get("frame_id") != right.get("frame_id"):
            mismatch(f"collision_objects.{object_id}.frame_id", left.get("frame_id"), right.get("frame_id"))
        left_primitives, right_primitives = left.get("primitives", []), right.get("primitives", [])
        if len(left_primitives) != len(right_primitives):
            mismatch(f"collision_objects.{object_id}.primitive_count", len(left_primitives), len(right_primitives))
            continue
        for index, (lp, rp) in enumerate(zip(left_primitives, right_primitives)):
            prefix = f"collision_objects.{object_id}.primitives[{index}]"
            if lp.get("type") != rp.get("type"):
                mismatch(f"{prefix}.type", lp.get("type"), rp.get("type"))
            ld, rd = lp.get("dimensions", []), rp.get("dimensions", [])
            if len(ld) != len(rd) or any(abs(float(a) - float(b)) > dimension_tolerance for a, b in zip(ld, rd)):
                mismatch(f"{prefix}.dimensions", ld, rd)
            lpose, rpose = lp.get("pose", {}), rp.get("pose", {})
            for coord_field in ("position_xyz", "orientation_xyzw"):
                a, b = lpose.get(coord_field, []), rpose.get(coord_field, [])
                if len(a) != len(b) or any(abs(float(x) - float(y)) > pose_tolerance for x, y in zip(a, b)):
                    mismatch(f"{prefix}.pose.{coord_field}", a, b)
    return {"status": "PASS" if not differences else "FAIL", "difference_count": len(differences), "differences": differences}


def classify_reported_distance(value: Any) -> dict[str, Any]:
    if value is None:
        return {"status": "NOT_AVAILABLE", "value_m": None, "reason": "distance_value_missing"}
    try:
        number = float(value)
    except (TypeError, ValueError):
        return {"status": "NOT_AVAILABLE", "value_m": None, "reason": "distance_value_not_numeric"}
    if not math.isfinite(number):
        return {"status": "NOT_AVAILABLE", "value_m": None, "reason": "distance_value_nonfinite"}
    if number == DISTANCE_SENTINEL:
        return {"status": "NOT_AVAILABLE", "value_m": None, "reason": "MoveIt_CollisionResult_distance_sentinel_minus_one"}
    if number < 0.0:
        return {"status": "NOT_AVAILABLE", "value_m": None, "reason": "negative_distance_semantics_unverified"}
    return {"status": "AVAILABLE_SAMPLED_MODEL_DISTANCE", "value_m": number, "reason": None}


def aggregate_reported_distances(values: Sequence[Any]) -> dict[str, Any]:
    classified = [classify_reported_distance(value) for value in values]
    available = [item["value_m"] for item in classified if item["status"] == "AVAILABLE_SAMPLED_MODEL_DISTANCE"]
    unavailable = len(classified) - len(available)
    if classified and unavailable == 0:
        status = "AVAILABLE_SAMPLED_MODEL_DISTANCE"
        minimum = min(available) if available else None
    elif available:
        status = "PARTIAL_NOT_AVAILABLE"
        minimum = None
    else:
        status = "NOT_AVAILABLE"
        minimum = None
    return {
        "status": status,
        "minimum_robot_world_distance_m": None,
        "minimum_reported_full_scene_distance_m": minimum,
        "sample_count": len(classified),
        "available_count": len(available),
        "unavailable_count": unavailable,
        "sentinel_count": sum(item.get("reason") == "MoveIt_CollisionResult_distance_sentinel_minus_one" for item in classified),
        "interpretation": "full-scene sampled model distance only; not isolated to robot-world pairs and not a continuous or physical clearance guarantee",
    }


def assess_zero_transform_gate(
    contract: C1SceneContract,
    *,
    trajectory_sha256: str,
    waypoint_count: int,
    sampled_state_count: int,
    collision_count: int,
    fk_status: str,
    q_unchanged: bool,
    timestamps_unchanged: bool,
    target_pose_max_change: float,
    normal_max_change: float,
    scene_comparison: Mapping[str, Any],
    all_objects_applied: bool = True,
) -> dict[str, Any]:
    checks = {
        "trajectory_sha256": trajectory_sha256 == contract.nominal_trajectory_sha256,
        "waypoint_count": waypoint_count == contract.waypoint_count,
        "q_identity": bool(q_unchanged),
        "timestamp_identity": bool(timestamps_unchanged),
        "fk_regression": fk_status == "PASS",
        "target_pose_identity": math.isfinite(target_pose_max_change) and target_pose_max_change <= POSE_TOLERANCE,
        "surface_normal_identity": math.isfinite(normal_max_change) and normal_max_change <= POSE_TOLERANCE,
        "scene_equivalence": scene_comparison.get("status") == "PASS",
        "all_scene_objects_applied": bool(all_objects_applied),
        "sampled_state_count": sampled_state_count == contract.expected_sample_count,
        "collision_count": collision_count == contract.expected_collision_count,
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "failed_checks": [key for key, value in checks.items() if not value]}
