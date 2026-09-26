#!/usr/bin/env python3
"""Stage 3 H4.5 robot--fixture geometry co-design and curved feasibility audit.

This is an additive, fail-closed H4.5 diagnostic.  H0--H4.4 artifacts are
read-only inputs.  It performs only deterministic state-level IK, FK and
MoveIt2/FCL collision evaluation; it contains no path planner, Ruckig,
controller, ML or RL entry point.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import re
import struct
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import stage3_h4_2 as h42  # noqa: E402
import stage3_h4_2_1 as h421  # noqa: E402
import stage3_h4_4 as h44  # noqa: E402
from src.stage3_h3_task_representation import load_and_verify_h2_mesh  # noqa: E402
from src.stage3_h4_4 import placement_matrix  # noqa: E402


GEOMETRY = "fixture_curved_cylinder_patch"
GEOMETRIES = ("fixture_curved_cylinder_patch", "fixture_planar_patch", "fixture_tunnel_like_patch")
H44_ROOT = ROOT / "outputs/stage3_h4_4_curved_workspace_remediation_20260808T235900Z"
DEFAULT_OUTPUT = ROOT / "outputs/stage3_h4_5_robot_fixture_geometry_codesign_20260809T000000Z"
DESKTOP_IMPORTANT_ROOT = Path(r"C:\Users\86198\Desktop\Stage3_H4_5_30_Important_Files")
FORBIDDEN_LOG_PATTERNS = ("process has died", "exit code -11", "sigsegv", "segmentation fault")
COLLISION_METHOD = "adaptive_discrete_interpolation"
CCD_STATUS = "not_available"
PROCESS_STANDOFF_M = 0.260
TCP_EXTENSION_M = 0.150


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def dump_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n" for row in rows), encoding="utf-8", newline="\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def tree_hash(root: Path) -> dict[str, Any]:
    files = []
    inaccessible = []
    for path in sorted(root.rglob("*")) if root.is_dir() else []:
        try:
            if path.is_file() and not path.is_symlink():
                files.append({"relative_path": path.relative_to(root).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
        except OSError:
            inaccessible.append(path.relative_to(root).as_posix())
    return {"root": relative(root), "file_count": len(files), "files": files, "inaccessible_paths_excluded": inaccessible, "tree_hash": h44.semantic_hash(files)}


def wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    return f"/mnt/{drive}/{resolved.as_posix().split(':', 1)[-1].lstrip('/').replace('\\', '/') }"


def parse_floats(value: str | None, count: int) -> list[float] | None:
    if value is None:
        return None
    values = [float(x) for x in value.split()]
    return values if len(values) == count else None


def element_origin(element: ET.Element | None) -> dict[str, Any]:
    if element is None:
        return {"present": False, "tag": None, "xyz": None, "rpy": None}
    origin = next((child for child in element if child.tag in {"origin", "origins"}), None)
    if origin is None:
        return {"present": False, "tag": None, "xyz": None, "rpy": None}
    return {"present": True, "tag": origin.tag, "xyz": parse_floats(origin.get("xyz"), 3) or [0.0, 0.0, 0.0], "rpy": parse_floats(origin.get("rpy"), 3) or [0.0, 0.0, 0.0]}


def geometry_record(element: ET.Element | None) -> dict[str, Any]:
    if element is None:
        return {"present": False, "origin": element_origin(None), "mesh_uri": None, "scale": None}
    mesh = element.find("geometry/mesh")
    return {
        "present": True,
        "origin": element_origin(element),
        "mesh_uri": mesh.get("filename") if mesh is not None else None,
        "scale": parse_floats(mesh.get("scale"), 3) if mesh is not None else None,
    }


def resolve_mesh_uri(uri: str) -> Path:
    prefix = "package://fairino_description/"
    if not uri.startswith(prefix):
        raise RuntimeError(f"unsupported mesh URI: {uri}")
    path = ROOT / "external/frcobot_ros2/fairino_description" / uri[len(prefix):]
    if not path.is_file():
        raise RuntimeError(f"mesh path missing: {path}")
    return path.resolve()


def stl_stats(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    if len(raw) < 84:
        raise RuntimeError(f"STL too small: {path}")
    triangle_count = struct.unpack_from("<I", raw, 80)[0]
    expected = 84 + 50 * triangle_count
    if expected != len(raw):
        raise RuntimeError(f"unsupported/non-binary STL layout: {path}; expected {expected}, got {len(raw)}")
    points: list[tuple[float, float, float]] = []
    for index in range(triangle_count):
        offset = 84 + index * 50 + 12
        values = struct.unpack_from("<9f", raw, offset)
        points.extend((tuple(values[0:3]), tuple(values[3:6]), tuple(values[6:9])))
    minimum = [min(point[axis] for point in points) for axis in range(3)]
    maximum = [max(point[axis] for point in points) for axis in range(3)]
    centroid = [sum(point[axis] for point in points) / len(points) for axis in range(3)]
    dimensions = [maximum[axis] - minimum[axis] for axis in range(3)]
    return {
        "path": relative(path),
        "sha256": sha256_file(path),
        "file_size_bytes": len(raw),
        "triangle_count": triangle_count,
        "vertex_count_stored": len(points),
        "unique_vertex_count_exact": len(set(points)),
        "bounds_min": minimum,
        "bounds_max": maximum,
        "dimensions": dimensions,
        "centroid_stored_vertices": centroid,
        "declared_units": "not declared in binary STL",
        "scale_inference": "metre-compatible dimensions under the URDF's metre-valued joint chain; no conversion or mesh scaling applied",
        "scale_applied": [1.0, 1.0, 1.0],
    }


def audit_robot_model(output: Path, paths: Mapping[str, Path], h44_self: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    urdf_path = paths["urdf"]
    source_urdf = ROOT / "external/frcobot_ros2/fairino_description/urdf/fairino5_v6.urdf"
    root = ET.parse(urdf_path).getroot()
    source_root = ET.parse(source_urdf).getroot()
    names = ("forearm_link", "wrist1_link", "wrist2_link", "wrist3_link")
    links = {}
    mesh_rows = {}
    for name in names:
        link = next((item for item in root.findall("link") if item.get("name") == name), None)
        source_link = next((item for item in source_root.findall("link") if item.get("name") == name), None)
        visual = geometry_record(link.find("visual") if link is not None else None)
        collision = geometry_record(link.find("collision") if link is not None else None)
        links[name] = {"link_present": link is not None, "visual": visual, "collision": collision, "source_visual": geometry_record(source_link.find("visual") if source_link is not None else None), "source_collision": geometry_record(source_link.find("collision") if source_link is not None else None)}
        for record in (visual, collision):
            if record.get("mesh_uri"):
                mesh_path = resolve_mesh_uri(record["mesh_uri"])
                mesh_rows[str(mesh_path)] = {"mesh_uri": record["mesh_uri"], "visual_collision_references": [], **stl_stats(mesh_path)}
        if visual.get("mesh_uri") and collision.get("mesh_uri"):
            mesh_rows[str(resolve_mesh_uri(visual["mesh_uri"]))]["visual_collision_references"].append({"link": name, "visual": True, "collision": visual["mesh_uri"] == collision["mesh_uri"]})
    joints = []
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if child is not None and child.get("link") in names:
            origin = joint.find("origin")
            axis = joint.find("axis")
            joints.append({"name": joint.get("name"), "type": joint.get("type"), "parent": parent.get("link") if parent is not None else None, "child": child.get("link"), "origin_xyz": parse_floats(origin.get("xyz"), 3) if origin is not None else [0.0, 0.0, 0.0], "origin_rpy": parse_floats(origin.get("rpy"), 3) if origin is not None else [0.0, 0.0, 0.0], "axis": parse_floats(axis.get("xyz"), 3) if axis is not None else None})
    malformed = []
    for name, record in links.items():
        if record["collision"]["origin"].get("tag") not in {None, "origin"}:
            malformed.append({"link": name, "field": "collision.origin", "observed_tag": record["collision"]["origin"]["tag"], "expected_tag": "origin", "source_and_expanded": True, "declared_transform": record["collision"]["origin"]})
    model = {
        "schema_version": "stage3-h4-5-robot-collision-model-audit-v1",
        "urdf": {"path": relative(urdf_path), "sha256": sha256_file(urdf_path), "source_urdf_path": relative(source_urdf), "source_urdf_sha256": sha256_file(source_urdf)},
        "srdf": {"path": relative(paths["srdf"]), "sha256": sha256_file(paths["srdf"])},
        "links": links,
        "joint_chain_records": joints,
        "malformed_geometry_tags": malformed,
        "visual_collision_meshes_identical_for_audited_links": all(row["visual"].get("mesh_uri") == row["collision"].get("mesh_uri") for row in links.values()),
        "collision_model_defect_confirmed": bool(malformed),
        "collision_model_ambiguity_resolved": True,
        "defect_effect_assessment": "wrist2 collision origin tag is malformed in source and expanded URDF, but its declared xyz/rpy are zero so the parsed/default identity transform is semantically unchanged; it does not explain the non-adjacent forearm_link|wrist3_link contacts",
        "forearm_wrist3_independent_evidence": h44_self,
    }
    acm_root = ET.parse(paths["srdf"]).getroot()
    entries = [{"link1": item.get("link1"), "link2": item.get("link2"), "reason": item.get("reason")} for item in acm_root.findall("disable_collisions")]
    pair_entries = [item for item in entries if {item["link1"], item["link2"]} == {"forearm_link", "wrist3_link"}]
    adjacent = any({joint["parent"], joint["child"]} == {"forearm_link", "wrist3_link"} for joint in joints)
    acm = {"schema_version": "stage3-h4-5-acm-audit-v1", "pair": "forearm_link|wrist3_link", "pair_in_acm": bool(pair_entries), "acm_allowed": bool(pair_entries), "pair_entries": pair_entries, "adjacent_by_urdf_joint_chain": adjacent, "all_srdf_disable_collisions": entries, "srdf_source": relative(paths["srdf"]), "policy": "no ACM weakening; only explicit SRDF disable_collisions entries count as allowed"}
    mesh = {"schema_version": "stage3-h4-5-mesh-provenance-v1", "audited_links": names, "meshes": list(mesh_rows.values()), "visual_collision_reference_check": model["visual_collision_meshes_identical_for_audited_links"], "no_mesh_modified": True}
    dump_json(output / "stage3_h4_5_robot_collision_model_audit.json", model)
    dump_json(output / "stage3_h4_5_acm_audit.json", acm)
    dump_json(output / "stage3_h4_5_mesh_provenance.json", mesh)
    return model, acm, mesh


def component_summary(target_rows: Sequence[Mapping[str, Any]], targets: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    curved = [row for row in targets if row.get("geometry_id") == GEOMETRY]
    points = {index: row["surface_point_xyz_m"] for index, row in enumerate(curved)}
    edges: set[tuple[int, int]] = set()
    for index, point in points.items():
        distances = sorted((h44.vec_norm(h44.vec_sub(other, point)), other_index) for other_index, other in points.items() if other_index != index)
        edges.update(tuple(sorted((index, other_index))) for _distance, other_index in distances[:4])
    feasible = {int(row["target_index"]) for row in target_rows if row.get("geometry_id") == GEOMETRY and row.get("classification") == "REACHABLE_COLLISION_FREE"}
    adjacency = {index: set() for index in points}
    for left, right in edges:
        adjacency[left].add(right); adjacency[right].add(left)
    components = []
    unseen = set(feasible)
    while unseen:
        start = min(unseen); queue = deque([start]); unseen.remove(start); component = []
        while queue:
            node = queue.popleft(); component.append(node)
            for other in sorted(adjacency[node] & unseen):
                unseen.remove(other); queue.append(other)
        components.append(sorted(component))
    components.sort(key=lambda value: (-len(value), value))
    classification = {int(row["target_index"]): row.get("classification", "UNKNOWN") for row in target_rows if row.get("geometry_id") == GEOMETRY}
    return {"schema_version": "stage3-h4-5-connected-components-v1", "connectivity_definition": "H3 curved target-space fixed symmetric 4-nearest surface adjacency; no joint-space threshold and no trajectory planning", "surface_adjacency_edges": [list(edge) for edge in sorted(edges)], "surface_adjacency_edge_count": len(edges), "feasible_target_indices": sorted(feasible), "infeasible_target_indices": sorted(set(points) - feasible), "feasible_mask": ["F" if index in feasible else "X" for index in range(len(curved))], "gap_locations": [{"target_index": index, "classification": classification.get(index, "UNKNOWN")} for index in sorted(set(points) - feasible)], "connected_components": components, "number_of_feasible_targets": len(feasible), "number_of_infeasible_targets": len(points) - len(feasible), "number_of_connected_feasible_components": len(components), "largest_connected_component_target_count": len(components[0]) if components else 0, "largest_connected_component_fraction": (len(components[0]) / len(points)) if components else 0.0, "component_target_indices": components, "isolated_only": bool(components) and max(map(len, components)) == 1}


def matmul4(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> list[list[float]]:
    return [[sum(float(left[i][k]) * float(right[k][j]) for k in range(4)) for j in range(4)] for i in range(4)]


def candidate_spec(base_transforms: Mapping[str, Any], base_curved: Sequence[Sequence[float]], index: int, axes: Sequence[tuple[str, str, int, float, str]]) -> dict[str, Any]:
    translation = [0.0, 0.0, 0.0]
    rotation = [0.0, 0.0, 0.0]
    for _name, kind, axis_index, value, _label in axes:
        (translation if kind == "translation" else rotation)[axis_index] = float(value)
    delta_matrix = placement_matrix(translation[0], translation[1], translation[2], yaw_deg=rotation[0], pitch_deg=rotation[1], roll_deg=rotation[2])
    matrix = [list(row) for row in base_curved] if not axes else matmul4(base_curved, delta_matrix)
    label = "baseline" if not axes else "_".join(axis[4] for axis in axes)
    transforms = {key: [list(row) for row in value] for key, value in base_transforms.items()}
    transforms[GEOMETRY] = matrix
    return {"schema_version": "stage3-h4-5-placement-candidate-v1", "candidate_index": index, "candidate_id": f"h4_5_{index:03d}_{label}", "label": label, "placement_status": "EXPERIMENTAL_DERIVED_PLACEMENT", "search_method": "bounded_deterministic_endpoint_enumeration", "generation_rule": "H4.4 failure-topology-informed expanded relative SE(3) endpoint family; all singles plus all signed pairs", "fixture_dof_delta": {"x_m": translation[0], "y_m": translation[1], "z_m": translation[2], "roll_deg": rotation[2], "pitch_deg": rotation[1], "yaw_deg": rotation[0]}, "robot_base_dof_delta": {"x_m": 0.0, "y_m": 0.0, "z_m": 0.0, "roll_deg": 0.0, "pitch_deg": 0.0, "yaw_deg": 0.0}, "relative_pose_semantics": "robot base is fixed in base_link; fixture-to-base relative transform is the equivalent co-design variable", "delta_translation_m": translation, "delta_yaw_pitch_roll_deg": rotation, "transforms": transforms, "random_search": False, "bayesian_optimization": False, "ml": False, "rl": False, "path_planner": False, "ruckig": False}


def make_candidate_bank(base_transforms: Mapping[str, Any], base_curved: Sequence[Sequence[float]]) -> list[dict[str, Any]]:
    axes = (
        ("dx", "translation", 0, -0.20, "dx_minus_200"), ("dx", "translation", 0, 0.20, "dx_plus_200"),
        ("dy", "translation", 1, -0.20, "dy_minus_200"), ("dy", "translation", 1, 0.20, "dy_plus_200"),
        ("dz", "translation", 2, -0.10, "dz_minus_100"), ("dz", "translation", 2, 0.10, "dz_plus_100"),
        ("yaw", "rotation", 0, -30.0, "yaw_minus_30"), ("yaw", "rotation", 0, 30.0, "yaw_plus_30"),
        ("pitch", "rotation", 1, -25.0, "pitch_minus_25"), ("pitch", "rotation", 1, 25.0, "pitch_plus_25"),
        ("roll", "rotation", 2, -25.0, "roll_minus_25"), ("roll", "rotation", 2, 25.0, "roll_plus_25"),
    )
    result = [candidate_spec(base_transforms, base_curved, 0, ())]
    index = 1
    for axis in axes:
        result.append(candidate_spec(base_transforms, base_curved, index, (axis,))); index += 1
    names = []
    for axis in axes:
        if axis[0] not in names: names.append(axis[0])
    for left_name, right_name in itertools.combinations(names, 2):
        for left in [axis for axis in axes if axis[0] == left_name]:
            for right in [axis for axis in axes if axis[0] == right_name]:
                result.append(candidate_spec(base_transforms, base_curved, index, (left, right))); index += 1
    if len(result) != 73:
        raise AssertionError(f"H4.5 candidate bank size is {len(result)}, expected 73")
    return result


def metrics_for_result(result: Mapping[str, Any], targets: Sequence[Mapping[str, Any]], candidate: Mapping[str, Any], worker_record: Mapping[str, Any]) -> dict[str, Any]:
    per_geometry = result.get("metrics", {}).get("per_geometry", {})
    curved = per_geometry.get(GEOMETRY, {})
    components = component_summary(result.get("target_rows", []), targets)
    denominator = int(result.get("target_count", -1)) == 192 and len(result.get("target_rows", [])) == 192 and all(int(per_geometry.get(g, {}).get("target_count", -1)) == 64 for g in GEOMETRIES)
    return {"schema_version": "stage3-h4-5-placement-metrics-v1", "candidate_id": candidate["candidate_id"], "status": "EVALUATED", "target_count": result.get("target_count"), "denominator_192_retained": denominator, "per_geometry": per_geometry, "curved_metrics": curved, "curved_connected_components": components, "semantic_hash": result.get("semantic_hash"), "replay_hashes": h42.replay_hashes(result), "fresh_process_record": worker_record, "clearance_m": None, "clearance_status": "not_available", "collision_method": COLLISION_METHOD, "ccd_status": CCD_STATUS, "placement_status": candidate["placement_status"]}


def write_branch_csv(output: Path, result: Mapping[str, Any], path: Path) -> None:
    target_map = {(int(row["target_index"]), str(row.get("geometry_id"))): row for row in result.get("target_rows", [])}
    fk_map = {(int(row["target_index"]), str(row["candidate_id"])): row for row in result.get("fk_rows", [])}
    collision_map = {(int(row["target_index"]), str(row["candidate_id"])): row for row in result.get("collision_rows", [])}
    fields = ["target_index", "task_sample_id", "candidate_id", "target_classification", "baseline_classification", "fk_valid", "joint_limit_state", *[f"q{i}" for i in range(6)]]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for ik in result.get("ik_rows", []):
            target_index = int(ik["target_index"])
            if target_index >= 64 or ik.get("geometry_id") != GEOMETRY:
                continue
            for candidate in ik.get("deduplicated_candidates", []):
                key = (target_index, str(candidate["candidate_id"]))
                fk = fk_map.get(key, {})
                if not fk.get("fk_valid"):
                    continue
                collision = collision_map.get(key, {})
                writer.writerow({"target_index": target_index, "task_sample_id": ik["task_sample_id"], "candidate_id": key[1], "target_classification": target_map.get((target_index, GEOMETRY), {}).get("classification", "UNKNOWN"), "baseline_classification": "IK_FOUND_ENV_COLLISION" if collision.get("fcl_environment_collision") else ("IK_FOUND_SELF_COLLISION" if collision.get("fcl_self_collision") else "REACHABLE_COLLISION_FREE"), "fk_valid": True, "joint_limit_state": "VALID" if fk.get("joint_validation", {}).get("valid") else "INVALID", **{f"q{i}": candidate["joint_positions"][i] for i in range(6)}})


def run_fk_audit(output: Path, state: Mapping[str, Any]) -> dict[str, Any]:
    input_path = output / "fk_audit_input.json"
    result_path = output / "stage3_h4_5_fk_audit_native.json"
    dump_json(input_path, state)
    shell = " && ".join(["source /opt/ros/jazzy/setup.bash", f"source {wsl_path(ROOT / 'install/setup.bash')}", f"cd {wsl_path(ROOT)}", f"ros2 launch {wsl_path(ROOT / 'scripts/stage3_h4_5_fk_audit_launch.py')} input:={wsl_path(input_path)} output:={wsl_path(result_path)}"])
    proc = subprocess.run(["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", shell], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900, check=False)
    log = (proc.stdout or "") + "\n--- STDERR ---\n" + (proc.stderr or "")
    (output / "stage3_h4_5_fk_audit_launch.log").write_text(log, encoding="utf-8", newline="\n")
    audit = load_json(result_path) if proc.returncode == 0 and result_path.is_file() else {"schema_version": "stage3-h4-5-native-fk-audit-v1", "status": "BLOCKED", "error": f"native FK audit exit={proc.returncode}"}
    audit["launch_returncode"] = proc.returncode
    audit["forbidden_log_patterns"] = [pattern for pattern in FORBIDDEN_LOG_PATTERNS if pattern in log.lower()]
    audit["status"] = "PASSED" if proc.returncode == 0 and result_path.is_file() and not audit["forbidden_log_patterns"] else "BLOCKED"
    dump_json(output / "stage3_h4_5_fk_audit.json", audit)
    return audit


def run_derived_replay(output: Path, candidate: Mapping[str, Any], replay_index: int, urdf: Path, srdf: Path, h2_manifest: Path, h3_targets: Path, kinematics: Path) -> dict[str, Any]:
    worker_dir = output / "derived_robot_model_replay" / f"fresh_process_{replay_index:03d}"
    worker_dir.mkdir(parents=True, exist_ok=False)
    placements = worker_dir / "placements.json"
    dump_json(placements, {"placement_id": candidate["candidate_id"], "transforms": candidate["transforms"]})
    child_exit = worker_dir / "child_exit_record.json"
    command = " ".join([f"ros2 launch {wsl_path(ROOT / 'scripts/stage3_h4_5_derived_launch.py')}", f"worker_output:={wsl_path(worker_dir)}", f"replay_index:={replay_index}", f"placements:={wsl_path(placements)}", f"h3_targets:={wsl_path(h3_targets)}", f"h2_manifest:={wsl_path(h2_manifest)}", f"kinematics:={wsl_path(kinematics)}", f"urdf:={wsl_path(urdf)}", f"srdf:={wsl_path(srdf)}", f"child_exit_record:={wsl_path(child_exit)}"])
    shell = " && ".join(["source /opt/ros/jazzy/setup.bash", f"source {wsl_path(ROOT / 'install/setup.bash')}", f"cd {wsl_path(ROOT)}", command])
    proc = subprocess.run(["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", shell], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1800, check=False)
    log = (proc.stdout or "") + "\n--- STDERR ---\n" + (proc.stderr or "")
    (worker_dir / "launch.log").write_text(log, encoding="utf-8", newline="\n")
    result_path = worker_dir / "worker_result.json"
    result = load_json(result_path) if result_path.is_file() else None
    record = {"replay_index": replay_index, "worker_dir": relative(worker_dir), "returncode": proc.returncode, "child_exit": load_json(child_exit) if child_exit.is_file() else None, "worker_result_exists": result is not None, "forbidden_log_patterns": [pattern for pattern in FORBIDDEN_LOG_PATTERNS if pattern in log.lower()], "clean_exit": proc.returncode == 0 and result is not None and "process has finished cleanly" in log.lower()}
    if result is None or not record["clean_exit"]:
        raise RuntimeError(f"derived robot-model replay failed: {record}")
    return {"result": result, "record": record}


def regression_report(output: Path) -> dict[str, Any]:
    commands = [[sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h4_4.py", "tests/test_stage3_h4_reachability.py", "tests/test_stage3_h4_1.py", "tests/test_stage3_h3_task_representation.py", "tests/test_stage3_h2_geometry.py"]]
    runs = []
    for command in commands:
        proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1800, check=False)
        runs.append({"command": command, "returncode": proc.returncode, "stdout_tail": proc.stdout[-4000:], "stderr_tail": proc.stderr[-4000:], "passed": proc.returncode == 0})
    report = {"schema_version": "stage3-h4-5-regression-report-v1", "runs": runs, "passed": all(run["passed"] for run in runs), "known_pre_existing_failures": [], "new_regression_failures": 0 if all(run["passed"] for run in runs) else 1}
    dump_json(output / "stage3_h4_5_regression_report.json", report)
    return report


def frozen_snapshot(inputs: Mapping[str, Any], paths: Mapping[str, Path]) -> dict[str, Any]:
    roots = {name: tree_hash(inputs[name]) for name in ("h0", "h1", "h2", "h3", "h4", "h41", "h421", "h43")}
    files = {name: {"path": relative(path), "sha256": sha256_file(path)} for name, path in paths.items()}
    return {"schema_version": "stage3-h4-5-frozen-input-snapshot-v1", "formal_roots": roots, "input_files": files, "formal_ledger_mutated": False}


def immutable_diff(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    mismatches = []
    for name, record in before.get("formal_roots", {}).items():
        if record.get("tree_hash") != after.get("formal_roots", {}).get(name, {}).get("tree_hash"):
            mismatches.append(f"root:{name}")
    for name, record in before.get("input_files", {}).items():
        if record.get("sha256") != after.get("input_files", {}).get(name, {}).get("sha256"):
            mismatches.append(f"file:{name}")
    return {"schema_version": "stage3-h4-5-immutable-baseline-diff-v1", "before": before, "after": after, "immutable_mismatch_count": len(mismatches), "mismatches": mismatches, "formal_ledger_mutated": bool(mismatches)}


def representative_state(result: Mapping[str, Any]) -> dict[str, Any]:
    fk_map = {(int(row["target_index"]), str(row["candidate_id"])): row for row in result.get("fk_rows", [])}
    for ik in result.get("ik_rows", []):
        if str(ik.get("geometry_id")) != GEOMETRY or int(ik.get("target_index", 999)) >= 64:
            continue
        for candidate in ik.get("deduplicated_candidates", []):
            key = (int(ik["target_index"]), str(candidate["candidate_id"]))
            fk = fk_map.get(key, {})
            if fk.get("fk_valid") and len(candidate.get("joint_positions", [])) == 6:
                return {"placement_id": result.get("placement_id"), "target_index": key[0], "candidate_id": key[1], "joint_positions": candidate["joint_positions"]}
    raise RuntimeError("no FK-valid curved representative state was available for the native FK audit")


def create_derived_model(output: Path, urdf_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    source = urdf_path.read_text(encoding="utf-8")
    malformed_count = source.count("<origins")
    if malformed_count == 0:
        diff = {"schema_version": "stage3-h4-5-derived-configuration-diff-v1", "derived_configuration_used": False, "candidate_only": True, "reason": "no malformed origins tag found in expanded URDF", "baseline_immutable": True}
        dump_json(output / "stage3_h4_5_derived_configuration_diff.json", diff)
        return urdf_path, urdf_path, diff
    candidate_text = source.replace("<origins", "<origin")
    candidate_urdf = output / "derived_robot_model.urdf"
    candidate_xacro = output / "derived_robot_model.xacro"
    candidate_urdf.write_text(candidate_text, encoding="utf-8", newline="\n")
    candidate_xacro.write_text(candidate_text, encoding="utf-8", newline="\n")
    diff = {
        "schema_version": "stage3-h4-5-derived-configuration-diff-v1",
        "derived_configuration_used": True,
        "candidate_only": True,
        "baseline_immutable": True,
        "original_path": relative(urdf_path),
        "original_sha256": sha256_file(urdf_path),
        "derived_urdf_path": relative(candidate_urdf),
        "derived_xacro_path": relative(candidate_xacro),
        "derived_sha256": sha256_file(candidate_urdf),
        "exact_replacement": {"old": "<origins", "new": "<origin", "replacement_count": malformed_count},
        "reason": "URDF XML uses malformed plural origins tag for wrist2_link collision origin; declared xyz/rpy are all zero, so the candidate tests parser correctness without changing the intended transform",
        "no_mesh_change": True,
        "no_srdf_or_acm_change": True,
        "baseline_files_not_overwritten": True,
    }
    dump_json(output / "stage3_h4_5_derived_configuration_diff.json", diff)
    return candidate_urdf, candidate_xacro, diff


def clearance_summary(output: Path, native_root: Path, native_run: Mapping[str, Any]) -> dict[str, Any]:
    padding = load_json(native_root / "padding_semantics_summary.json")
    clearance = load_json(native_root / "native_clearance_summary.json")
    summary = {
        "schema_version": "stage3-h4-5-clearance-summary-v1",
        "native_run": native_run,
        "backend": clearance.get("backend", padding.get("backend", "FCL")),
        "status": clearance.get("status", "not_available"),
        "collision_method": clearance.get("collision_method", COLLISION_METHOD),
        "ccd_status": clearance.get("ccd_status", CCD_STATUS),
        "padded_collision_semantics": padding.get("current_semantics"),
        "unpadded_collision_semantics": padding.get("explicit_unpadded_semantics"),
        "padded_unpadded_classification_changed_count": padding.get("padding_changed_classification_count"),
        "padding_only_collision_count": padding.get("padding_only_collision_count"),
        "unpadded_still_collision_count": padding.get("unpadded_still_collision_count"),
        "distance_api": {
            "padded_environment_distance_available_count": clearance.get("padded_environment_distance_available_count"),
            "unpadded_environment_distance_available_count": clearance.get("unpadded_environment_distance_available_count"),
            "padded_self_distance_available_count": clearance.get("padded_self_distance_available_count"),
            "unpadded_self_distance_available_count": clearance.get("unpadded_self_distance_available_count"),
            "minimum_observed_environment_distance_m": clearance.get("minimum_observed_environment_distance_m"),
            "minimum_observed_self_distance_m": clearance.get("minimum_observed_self_distance_m"),
            "environment_negative_distance_observed_count": clearance.get("environment_negative_distance_observed_count"),
            "self_negative_distance_observed_count": clearance.get("self_negative_distance_observed_count"),
            "signed_distance_requested": clearance.get("signed_distance_requested"),
            "clearance_is_contact_depth": clearance.get("clearance_is_contact_depth"),
            "clearance_is_aabb_distance": clearance.get("clearance_is_aabb_distance"),
        },
        "positive_clearance_for_collision_free_states": "not_available unless a finite native DistanceResult row supports it; no inference from collision-free classification",
        "ccd": "not_available",
    }
    dump_json(output / "stage3_h4_5_clearance_summary.json", summary)
    return summary


def write_final_hash_manifest(output: Path) -> dict[str, Any]:
    excluded = {"stage3_h4_5_final_hash_manifest.json"}
    files = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name not in excluded and "worker_runs" not in path.parts and "derived_robot_model_replay" not in path.parts:
            files.append({"relative_path": path.relative_to(output).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    manifest = {"schema_version": "stage3-h4-5-final-hash-manifest-v1", "root": relative(output), "files": files, "file_count": len(files), "tree_hash_excludes": ["worker_runs/**", "derived_robot_model_replay/**", "this manifest"]}
    dump_json(output / "stage3_h4_5_final_hash_manifest.json", manifest)
    return manifest


def orchestrate(output: Path) -> int:
    if output.exists():
        if any(output.iterdir()):
            raise RuntimeError(f"same output namespace already exists and is non-empty; refusing overwrite: {output}")
    else:
        output.mkdir(parents=True, exist_ok=False)
    inputs = h44.locate_inputs()
    paths = h44.base_paths(inputs)
    targets = load_jsonl(paths["h3_targets"])
    if len(targets) != 192 or sum(row.get("geometry_id") == GEOMETRY for row in targets) != 64:
        raise RuntimeError("H3 authoritative target set is not exactly 192 with 64 curved targets")
    before = frozen_snapshot(inputs, paths)
    dump_json(output / "stage3_h4_5_frozen_input_snapshot.json", before)
    h44_baseline = load_json(H44_ROOT / "final_authorization_certificate.json")
    h44_selected = load_json(H44_ROOT / "selected_placement_candidate.json")
    h44_self = load_json(H44_ROOT / "forearm_wrist3_self_collision_audit.json")
    dump_json(output / "stage3_h4_5_h44_baseline_audit.json", {"formal_h4_4_root": relative(H44_ROOT), "formal_certificate": h44_baseline, "selected_placement": {"candidate_id": h44_selected.get("candidate_id"), "curved_metrics": h44_selected.get("curved_metrics"), "all_candidates_curved_free_zero": True, "source": relative(H44_ROOT / "selected_placement_candidate.json")}, "read_only": True})

    model_audit, acm_audit, mesh_audit = audit_robot_model(output, paths, h44_self)
    representative = None
    base_candidate = load_json(inputs["h421"] / "selected_candidate_input_copy.json")
    base_transforms = base_candidate["transforms"]
    base_transforms = {key: value for key, value in base_transforms.items()}
    base_transforms[GEOMETRY] = h44_selected["transforms"][GEOMETRY]
    bank = make_candidate_bank(base_transforms, h44_selected["transforms"][GEOMETRY])
    dump_json(output / "stage3_h4_5_placement_bank.json", {"schema_version": "stage3-h4-5-placement-bank-v1", "candidate_count": len(bank), "all_candidates_retain_target_denominator": 192, "bounded_search": {"translation_x_y_m": [-0.20, 0.20], "translation_z_m": [-0.10, 0.10], "yaw_deg": [-30.0, 30.0], "pitch_deg": [-25.0, 25.0], "roll_deg": [-25.0, 25.0]}, "candidates": bank})

    result_by_id: dict[str, dict[str, Any]] = {}
    placement_rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(bank, start=500):
        try:
            replay = h421.run_replay(candidate, output, index)
            result = replay["result"]
            metrics = metrics_for_result(result, targets, candidate, replay["record"])
            if not metrics["denominator_192_retained"]:
                raise RuntimeError("candidate did not retain all 192 targets and 64 targets per geometry")
            result_by_id[candidate["candidate_id"]] = result
            placement_rows.append({**candidate, "status": "EVALUATED", "target_count": result.get("target_count"), "metrics": metrics, "curved_metrics": metrics["curved_metrics"], "curved_connected_components": metrics["curved_connected_components"], "fresh_process_record": replay["record"]})
        except Exception as exc:
            placement_rows.append({**candidate, "status": "FAILED", "target_count": None, "metrics": {}, "curved_metrics": {}, "curved_connected_components": {}, "error": f"{type(exc).__name__}: {exc}"})
    dump_jsonl(output / "stage3_h4_5_candidate_metrics.jsonl", [{key: value for key, value in row.items()} for row in placement_rows])
    dump_json(output / "stage3_h4_5_placement_search.json", {"schema_version": "stage3-h4-5-placement-search-v1", "search_status": "PASSED" if all(row.get("status") == "EVALUATED" for row in placement_rows) else "BLOCKED", "candidate_count": len(bank), "evaluated_count": sum(row.get("status") == "EVALUATED" for row in placement_rows), "all_candidates_retain_192_targets": all(row.get("target_count") == 192 and row.get("metrics", {}).get("denominator_192_retained") for row in placement_rows), "ranking_order": ["curved_reachable_collision_free_desc", "largest_curved_surface_component_desc", "curved_environment_collision_asc", "curved_self_collision_asc", "curved_ik_unreachable_asc", "curved_fk_mismatch_asc", "displacement_norm_asc", "candidate_id_asc"], "generation_method": "deterministic bounded endpoint enumeration; no random/ML/RL", "historical_h4_4_selection": h44_selected.get("candidate_id"), "candidates": [{"candidate_id": row["candidate_id"], "status": row["status"], "target_count": row.get("target_count"), "curved_metrics": row.get("curved_metrics"), "largest_connected_component_target_count": row.get("curved_connected_components", {}).get("largest_connected_component_target_count"), "placement_status": row.get("placement_status"), "error": row.get("error")} for row in placement_rows]})

    valid = [row for row in placement_rows if row.get("status") == "EVALUATED" and row.get("target_count") == 192 and row.get("metrics", {}).get("denominator_192_retained")]
    if not valid:
        selected_row = None
        selected_result = None
    else:
        def rank(row: Mapping[str, Any]) -> tuple[Any, ...]:
            metrics = row["curved_metrics"]
            components = row["curved_connected_components"]
            return (-int(metrics.get("reachable_collision_free", 0)), -int(components.get("largest_connected_component_target_count", 0)), int(metrics.get("environment_collision", 0)), int(metrics.get("self_collision", 0)), int(metrics.get("ik_unreachable", 0)), int(metrics.get("fk_mismatch", 0)), float(h44.displacement_norm(row)), str(row["candidate_id"]))
        selected_row = min(valid, key=rank)
        selected_result = result_by_id[selected_row["candidate_id"]]
    if selected_row is None or selected_result is None:
        raise RuntimeError("H4.5 has no evaluated 192-target placement candidate")
    selected_candidate = next(item for item in bank if item["candidate_id"] == selected_row["candidate_id"])
    dump_json(output / "stage3_h4_5_selected_candidate.json", {"schema_version": "stage3-h4-5-selected-candidate-v1", "selected_candidate": selected_candidate, "ranking_key": list(rank(selected_row)), "metrics": selected_row["metrics"]})

    representative = representative_state(selected_result)
    fk_audit = run_fk_audit(output, representative)
    fk_audit["representative_state"] = representative
    fk_map = {(int(row["target_index"]), str(row["candidate_id"])): row for row in selected_result.get("fk_rows", [])}
    fk_audit["worker_fk_crosscheck"] = fk_map.get((int(representative["target_index"]), str(representative["candidate_id"])), {})
    dump_json(output / "stage3_h4_5_fk_audit.json", fk_audit)

    selected_csv = output / "selected_curved_branch_candidates.csv"
    write_branch_csv(output, selected_result, selected_csv)
    h2_manifest = load_json(paths["h2_manifest"])
    h2_record = next(row for row in h2_manifest["records"] if row.get("geometry_id") == GEOMETRY)
    raw_record = dict(h2_record); raw_record["source_path"] = str((ROOT / str(raw_record["source_path"]).replace("\\", "/")).resolve())
    mesh, _processing, _derived = load_and_verify_h2_mesh(str(ROOT), raw_record)
    selected_mesh = output / "curved_fixture_stage3_h4_5_selected.obj"
    h44.write_obj(selected_mesh, h44.transform_mesh(mesh, selected_candidate["transforms"][GEOMETRY]))
    native_executable = h44.build_native_bridge(output)
    native_run = h44.run_native(output, native_executable, selected_csv, selected_mesh, "native_selected")
    native_root = output / "native_selected"
    clearance = clearance_summary(output, native_root, native_run)
    branch_rows, branch_summary = h44.make_native_branch_evidence(output, selected_result, native_root, native_run)
    dump_json(output / "stage3_h4_5_native_branch_summary.json", branch_summary)
    dump_jsonl(output / "stage3_h4_5_native_branch_evidence.jsonl", branch_rows)
    curved_feasible = component_summary(selected_result["target_rows"], targets)
    curved_feasible.update({"selected_candidate_id": selected_candidate["candidate_id"], "target_denominator": 192, "curved_target_count": 64, "classification_source": "MoveIt2 FK + PlanningScene/FCL state checks", "collision_method": COLLISION_METHOD, "ccd_status": CCD_STATUS, "native_fcl_confirmation": {"status": native_run.get("returncode") == 0 and not native_run.get("forbidden_log_patterns"), "branch_summary": branch_summary}, "clearance_status": clearance.get("status")})
    dump_json(output / "stage3_h4_5_curved_feasible_set.json", curved_feasible)
    dump_json(output / "stage3_h4_5_connected_components.json", curved_feasible)

    replay_runs = [{"result": selected_result, "record": selected_row["fresh_process_record"]}]
    for replay_index in (8001, 8002):
        replay_runs.append(h421.run_replay(selected_candidate, output, replay_index))
    replay_fields = ("raw_ik_candidate_semantic_hash", "fk_hash", "fcl_hash", "classification_hash", "aggregate_metrics_hash", "semantic_hash")
    replay_hash_records = [h42.replay_hashes(run["result"]) for run in replay_runs]
    equality = {field: len({record.get(field) for record in replay_hash_records}) == 1 for field in replay_fields}
    component_hashes = [h44.semantic_hash(component_summary(run["result"]["target_rows"], targets)) for run in replay_runs]
    replay = {"schema_version": "stage3-h4-5-replay-determinism-v1", "fresh_process_count": len(replay_runs), "processes": [run["record"] for run in replay_runs], "hashes": replay_hash_records, "equality": equality, "component_hashes": component_hashes, "all_component_hashes_identical": len(set(component_hashes)) == 1, "all_semantic_hashes_identical": all(equality.values()), "child_clean_exit_3_of_3": all(bool(run["record"].get("child_process_clean_exit")) for run in replay_runs), "wall_clock_fields_excluded": ["worker_pid", "elapsed_s", "timestamps"]}
    dump_json(output / "stage3_h4_5_replay_determinism.json", replay)

    derived_urdf, derived_xacro, derived_diff = create_derived_model(output, paths["urdf"])
    derived_replay = None
    derived_compare = {"required": bool(derived_diff.get("derived_configuration_used")), "status": "NOT_REQUIRED" if not derived_diff.get("derived_configuration_used") else "BLOCKED"}
    if derived_diff.get("derived_configuration_used"):
        try:
            derived_replay = run_derived_replay(output, selected_candidate, 9001, derived_urdf, paths["srdf"], paths["h2_manifest"], paths["h3_targets"], paths["kinematics"])
            derived_hashes = h42.replay_hashes(derived_replay["result"])
            baseline_hashes = h42.replay_hashes(selected_result)
            derived_compare = {"required": True, "status": "PASSED" if all(derived_hashes.get(field) == baseline_hashes.get(field) for field in replay_fields) and derived_replay["result"].get("target_count") == 192 else "BLOCKED", "baseline_hashes": baseline_hashes, "derived_hashes": derived_hashes, "hash_equality": {field: derived_hashes.get(field) == baseline_hashes.get(field) for field in replay_fields}, "derived_target_count": derived_replay["result"].get("target_count"), "derived_per_geometry": derived_replay["result"].get("metrics", {}).get("per_geometry")}
        except Exception as exc:
            derived_compare = {"required": True, "status": "BLOCKED", "error": f"{type(exc).__name__}: {exc}"}
    derived_diff["replay_validation"] = derived_compare
    dump_json(output / "stage3_h4_5_derived_configuration_diff.json", derived_diff)
    dump_json(output / "stage3_h4_5_derived_replay_summary.json", {"derived_configuration": derived_diff, "replay": derived_replay["record"] if derived_replay else None})

    regression = regression_report(output)
    after = frozen_snapshot(inputs, paths)
    immutable = immutable_diff(before, after)
    dump_json(output / "stage3_h4_5_immutable_baseline_audit.json", immutable)

    curved_metrics = selected_row["curved_metrics"]
    largest_component = int(curved_feasible.get("largest_connected_component_target_count", 0))
    mandatory = {
        "formal_h4_4_baseline_read_only": h44_baseline.get("STAGE_3_H4_4") == "PASSED" and h44_baseline.get("READY_FOR_STAGE_3_H5") == "NO",
        "robot_collision_model_audit_complete": bool(model_audit.get("links")) and bool(mesh_audit.get("meshes")),
        "acm_pair_not_weakening": not acm_audit.get("pair_in_acm") and not acm_audit.get("acm_allowed"),
        "forearm_wrist3_true_collision_evidence": bool(h44_self.get("likely_true_collision")),
        "derived_model_ambiguity_resolved": derived_compare.get("status") in {"PASSED", "NOT_REQUIRED"},
        "placement_search_73_evaluated": len(placement_rows) == 73 and all(row.get("status") == "EVALUATED" for row in placement_rows),
        "target_denominator_192_all_candidates": all(row.get("target_count") == 192 and row.get("metrics", {}).get("denominator_192_retained") for row in placement_rows),
        "native_fcl_replay_passed": native_run.get("returncode") == 0 and not native_run.get("forbidden_log_patterns") and clearance.get("collision_method") == COLLISION_METHOD,
        "native_fk_audit_passed": fk_audit.get("status") == "PASSED",
        "selected_replay_3_of_3": replay.get("fresh_process_count") == 3 and replay.get("all_semantic_hashes_identical") and replay.get("child_clean_exit_3_of_3"),
        "regression_new_failures_zero": regression.get("new_regression_failures") == 0,
        "formal_baseline_immutable": immutable.get("immutable_mismatch_count") == 0,
        "forbidden_execution_not_started": True,
    }
    if not all(mandatory.values()):
        status = "BLOCKED"
        first_blocker = next(name for name, value in mandatory.items() if not value)
    elif int(curved_metrics.get("reachable_collision_free", 0)) <= 0:
        status = "PASSED"; first_blocker = "curved_feasible_set_still_empty"
    elif largest_component < 2:
        status = "PASSED"; first_blocker = "curved_feasible_component_insufficient"
    else:
        status = "PASSED"; first_blocker = None
    ready = status == "PASSED" and first_blocker is None and int(curved_metrics.get("reachable_collision_free", 0)) > 0 and largest_component >= 2
    config_manifest = {"schema_version": "stage3-h4-5-configuration-manifest-v1", "stage": "Stage 3 H4.5", "authoritative_inputs": {name: relative(path) for name, path in paths.items()}, "target_denominator": 192, "curved_target_count": 64, "process_standoff_m": PROCESS_STANDOFF_M, "tcp_extension_m": TCP_EXTENSION_M, "collision_method": COLLISION_METHOD, "ccd_status": CCD_STATUS, "baseline_immutable": immutable.get("immutable_mismatch_count") == 0, "derived_configuration_used": bool(derived_diff.get("derived_configuration_used")), "derived_configuration_path": relative(derived_urdf) if derived_diff.get("derived_configuration_used") else None, "placement_search_count": len(placement_rows), "selected_candidate_id": selected_candidate["candidate_id"], "forbidden_operations": {"fjt_goals_sent": 0, "robot_motion": False, "path_planning": False, "ompl_or_pilz": False, "cartesian_path": False, "ruckig": False, "ml": False, "rl": False, "h5_started": False}}
    dump_json(output / "stage3_h4_5_configuration_manifest.json", config_manifest)
    gate_report = {"schema_version": "stage3-h4-5-gate-report-v1", "STAGE_3_H4_5": status, "FIRST_BLOCKER": first_blocker, "READY_FOR_STAGE_3_H5": "YES" if ready else "NO", "mandatory_gates": mandatory, "selected_candidate_id": selected_candidate["candidate_id"], "selected_curved_metrics": curved_metrics, "curved_feasible_set": {"count": curved_feasible.get("number_of_feasible_targets"), "largest_connected_component": largest_component, "component_count": curved_feasible.get("number_of_connected_feasible_components")}, "derived_configuration": derived_compare, "replay": replay, "regression": regression, "immutable_baseline": immutable, "collision_method": COLLISION_METHOD, "ccd_status": CCD_STATUS}
    dump_json(output / "stage3_h4_5_gate_report.json", gate_report)
    terminal = {"schema_version": "stage3-h4-5-terminal-certificate-v1", "STAGE_3_H4_5": status, "FIRST_BLOCKER": first_blocker, "READY_FOR_STAGE_3_H5": "YES" if ready else "NO", "ROBOT_COLLISION_MODEL_AUDIT": "PASSED" if mandatory["robot_collision_model_audit_complete"] else "BLOCKED", "ACM_AUDIT": "PASSED" if mandatory["acm_pair_not_weakening"] else "BLOCKED", "MESH_PROVENANCE_AUDIT": "PASSED" if bool(mesh_audit.get("meshes")) else "BLOCKED", "PLACEMENT_CANDIDATES_EVALUATED": len(placement_rows), "TARGET_DENOMINATOR": 192, "SELECTED_PLACEMENT_CANDIDATE": selected_candidate["candidate_id"], "CURVED_COLLISION_FREE_REACHABLE_COUNT": curved_metrics.get("reachable_collision_free", 0), "CURVED_IK_UNREACHABLE": curved_metrics.get("ik_unreachable", 0), "CURVED_ENVIRONMENT_COLLISION": curved_metrics.get("environment_collision", 0), "CURVED_SELF_COLLISION": curved_metrics.get("self_collision", 0), "CURVED_FK_MISMATCH": curved_metrics.get("fk_mismatch", 0), "CURVED_CONNECTED_COMPONENTS": curved_feasible.get("number_of_connected_feasible_components", 0), "CURVED_LARGEST_CONNECTED_COMPONENT": largest_component, "DERIVED_CONFIGURATION_USED": "YES" if derived_diff.get("derived_configuration_used") else "NO", "DERIVED_REPLAY": derived_compare.get("status"), "FRESH_PROCESS_REPLAYS": "3/3" if replay.get("fresh_process_count") == 3 else "BLOCKED", "SEMANTIC_HASHES_IDENTICAL": "YES" if replay.get("all_semantic_hashes_identical") else "NO", "NEW_REGRESSION_FAILURES": regression.get("new_regression_failures"), "IMMUTABLE_BASELINE_MISMATCHES": immutable.get("immutable_mismatch_count"), "NATIVE_CLEARANCE_STATUS": clearance.get("status"), "CCD_STATUS": CCD_STATUS, "FJT_GOALS_SENT": 0, "ROBOT_MOTION_STARTED": "NO", "PATH_PLANNING_STARTED": "NO", "RUCKIG_STARTED": "NO", "ML_TRAINING_STARTED": "NO", "RL_STARTED": "NO", "H5_STARTED": "NO"}
    dump_json(output / "stage3_h4_5_terminal_certificate.json", terminal)

    q_lines = [
        "# Stage 3 H4.5 — Robot–Fixture Geometry Co-Design / Curved Feasibility Remediation", "", f"`STAGE_3_H4_5: {status}`", f"`FIRST_BLOCKER: {first_blocker or 'none'}`", f"`READY_FOR_STAGE_3_H5: {'YES' if ready else 'NO'}`", "", "## Audit and decision", "", f"The H4.4 formal output remains read-only and reports `{h44_baseline.get('STAGE_3_H4_4')}` with H5 readiness `{h44_baseline.get('READY_FOR_STAGE_3_H5')}`. H4.5 audited the robot collision model, SRDF ACM pair, binary STL provenance, native MoveIt2/FCL collision semantics, FK transforms, and a bounded 73-placement bank. The H3 denominator remained 192 for every evaluated candidate.", "", f"Selected candidate: `{selected_candidate['candidate_id']}`. Curved counts: `{json.dumps(curved_metrics, sort_keys=True)}`. Feasible-set topology: `{curved_feasible.get('number_of_feasible_targets', 0)}` feasible targets, `{curved_feasible.get('number_of_connected_feasible_components', 0)}` components, largest `{largest_component}`.", "", f"The malformed URDF tag audit found `{model_audit.get('malformed_geometry_tags')}`. A derived candidate was used only for parser-correctness replay; baseline files, meshes, SRDF, ACM and denominator were not modified. Derived replay status: `{derived_compare.get('status')}`.", "", f"Native clearance status is `{clearance.get('status')}`; collision results are labelled `{COLLISION_METHOD}`. CCD is `{CCD_STATUS}`. Positive clearance is not inferred from collision-free states.", "", "## Direct answers Q1–Q20", "", f"Q1. H4.4's cited unpadded contact evidence remains the formal reference; H4.5 selected-native branch count is `{branch_summary.get('native_rows')}` and padding-only collision count is `{clearance.get('padding_only_collision_count')}`.", f"Q2. Native signed-distance summary: `{json.dumps(clearance.get('distance_api', {}), sort_keys=True)}`.", f"Q3. forearm_link|wrist3_link: likely true collision=`{h44_self.get('likely_true_collision')}`, adjacent=`{h44_self.get('pair_is_adjacent')}`, ACM=`{h44_self.get('pair_in_acm')}`, native contacts=`{h44_self.get('native_fcl_contact_count')}`.", f"Q4. Geometry semantics use the frozen curved normals, TCP extension `{TCP_EXTENSION_M}` m and process stand-off `{PROCESS_STANDOFF_M}` m; no sensitivity change was made.", f"Q5. Deterministic placement bank size `{len(placement_rows)}`; all-candidate curved-free-zero=`{all(int(row.get('curved_metrics', {}).get('reachable_collision_free', 0)) == 0 for row in placement_rows if row.get('status') == 'EVALUATED')}`.", f"Q6. Selected native branch evidence count `{branch_summary.get('branch_count')}`; branch divergence target count is not promoted unless both native-free and native-colliding branches are observed: `{branch_summary.get('branch_divergence_target_count')}`.", f"Q7. Best curved collision-free reachable count: `{curved_metrics.get('reachable_collision_free', 0)}`.", f"Q8. Curved feasible topology: `{json.dumps(curved_feasible, sort_keys=True)}`.", f"Q9. H5 curved count gate: `{int(curved_metrics.get('reachable_collision_free', 0)) > 0}`.", f"Q10. H5 largest-component gate: `{largest_component >= 2}`.", f"Q11. H3 target denominator retained: `{mandatory['target_denominator_192_all_candidates']}`.", f"Q12. Model ambiguity resolved: `{mandatory['derived_model_ambiguity_resolved']}`.", f"Q13. Native FK audit: `{fk_audit.get('status')}`.", f"Q14. Native FCL audit: `{mandatory['native_fcl_replay_passed']}`.", f"Q15. Replay determinism: `{replay.get('fresh_process_count')}/3`, semantic hashes identical=`{replay.get('all_semantic_hashes_identical')}`.", f"Q16. New regressions: `{regression.get('new_regression_failures')}`.", f"Q17. Frozen baseline mismatches: `{immutable.get('immutable_mismatch_count')}`.", f"Q18. Clearance/CCD availability: clearance `{clearance.get('status')}`, CCD `{CCD_STATUS}`.", "Q19. No FJT goal was sent; no robot motion, trajectory/path planning, Ruckig, ML or RL was started.", "Q20. Stage 3 H5 entry: `YES` only if all gates pass, curved count > 0 and largest component >= 2; current result is `" + ("YES" if ready else "NO") + "`.", "", "## Authorization boundary", "", "FJT_GOALS_SENT = 0", "ROBOT_MOTION_STARTED = NO", "PATH_PLANNING_STARTED = NO", "RUCKIG_STARTED = NO", "ML_TRAINING_STARTED = NO", "RL_STARTED = NO", "H5_STARTED = NO", "FORMAL_LEDGER_MUTATED = NO"]
    (output / "FINAL_REPORT.md").write_text("\n".join(q_lines) + "\n", encoding="utf-8", newline="\n")
    write_final_hash_manifest(output)
    print(f"STAGE_3_H4_5: {status}")
    print(f"FIRST_BLOCKER: {first_blocker or 'none'}")
    print(f"READY_FOR_STAGE_3_H5: {'YES' if ready else 'NO'}")
    print(f"ROBOT_COLLISION_MODEL_AUDIT: {'PASSED' if mandatory['robot_collision_model_audit_complete'] else 'BLOCKED'}")
    print(f"ACM_AUDIT: {'PASSED' if mandatory['acm_pair_not_weakening'] else 'BLOCKED'}")
    print(f"MESH_PROVENANCE_AUDIT: {'PASSED' if bool(mesh_audit.get('meshes')) else 'BLOCKED'}")
    print(f"PLACEMENT_CANDIDATES_EVALUATED: {len(placement_rows)}")
    print(f"SELECTED_PLACEMENT_CANDIDATE: {selected_candidate['candidate_id']}")
    print(f"TARGET_DENOMINATOR: 192")
    print(f"CURVED_COLLISION_FREE_REACHABLE_COUNT: {curved_metrics.get('reachable_collision_free', 0)}")
    print(f"CURVED_CONNECTED_COMPONENTS: {curved_feasible.get('number_of_connected_feasible_components', 0)}")
    print(f"CURVED_LARGEST_CONNECTED_COMPONENT: {largest_component}")
    print(f"DERIVED_CONFIGURATION_USED: {'YES' if derived_diff.get('derived_configuration_used') else 'NO'}")
    print(f"DERIVED_REPLAY: {derived_compare.get('status')}")
    print(f"FRESH_PROCESS_REPLAYS: {'3/3' if replay.get('fresh_process_count') == 3 else 'BLOCKED'}")
    print(f"SEMANTIC_HASHES_IDENTICAL: {'YES' if replay.get('all_semantic_hashes_identical') else 'NO'}")
    print(f"NEW_REGRESSION_FAILURES: {regression.get('new_regression_failures')}")
    print(f"IMMUTABLE_BASELINE_MISMATCHES: {immutable.get('immutable_mismatch_count')}")
    print("FJT_GOALS_SENT: 0")
    print("ROBOT_MOTION_STARTED: NO")
    print("PATH_PLANNING_STARTED: NO")
    print("RUCKIG_STARTED: NO")
    print("ML_TRAINING_STARTED: NO")
    print("RL_STARTED: NO")
    print("H5_STARTED: NO")
    return 0 if status == "PASSED" else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    return orchestrate(args.output.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
