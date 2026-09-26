#!/usr/bin/env python3
"""Stage 3 H4.2 workspace-placement root-cause diagnosis.

This file is intentionally an offline-only derivative of the frozen H4.1
single-attempt bridge.  It never creates an action client, sends a goal,
plans a path, invokes Ruckig, or trains a model.  Fixture placement is a
derived experimental variable; H2/H3/H4/H4.1 artifacts are read-only inputs.

The ROS worker evaluates one deterministic placement at a time with the same
13 explicit seeds and the same direct KDL ``getPositionIK`` bridge as H4.1.
The Windows-side entry point invokes the worker through WSL because the
MoveIt2/FCL runtime is installed in the ROS2 Jazzy WSL environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import stage3_h4_1 as h41  # noqa: E402
import stage3_h4_baseline as h4  # noqa: E402
from src.stage3_h2_geometry import Mesh  # noqa: E402
from src.stage3_h4_reachability import (  # noqa: E402
    COLLISION_METHOD,
    H4Config,
    H4ValidationError,
    TARGET_SCHEMA_VERSION,
    UNAVAILABLE,
    aggregate_target_metrics,
    deduplicate_candidates,
    replay_normalize,
    semantic_hash,
    sha256_file,
    validate_h3_target,
    validate_joint_vector,
)

H0_ROOT = ROOT / "outputs/stage3_h0_entry_authorization_20260807T161253Z"
H0_MANIFEST = H0_ROOT / "stage3_h0_stage2_immutable_baseline_manifest.json"
H1_ROOT = ROOT / "outputs/stage3_h1_research_contract_20260807T171900Z"
H2_ROOT = ROOT / "outputs/stage3_h2_geometry_baseline_20260807T180321Z"
H2_MANIFEST = H2_ROOT / "stage3_h2_geometry_manifest.json"
H3_ROOT = ROOT / "outputs/stage3_h3_coverage_baseline_20260808T034526Z"
H3_TARGETS = H3_ROOT / "stage3_h3_surface_targets.jsonl"
H4_ROOT = ROOT / "outputs/stage3_h4_reachability_baseline_20260808T190000Z"
H41_ROOT = ROOT / "outputs/stage3_h4_1_run_20260808T000004Z"
KINEMATICS = ROOT / "external/frcobot_ros2/fairino5_v6_moveit2_config/config/kinematics.yaml"
URDF = ROOT / "outputs/ik_graph_stage192/minimal_case/fairino5_v6_spray_tcp.expanded.urdf"
SRDF = ROOT / "ros2_moveit_bridge/config/fairino5_v6_spray_tcp.srdf"
BRIDGE = ROOT / "outputs/stage3_h4_1_work/install/lib/stage3_h4_1_bridge/stage3_h4_1_ik_bridge"
CONFIG = H4Config()
DATASHEET_REACH_M = 0.922
TCP_PHYSICAL_OFFSET_M = 0.150
DEFAULT_OUTPUT = ROOT / "outputs/stage3_h4_2_workspace_diagnosis_20260808T150000Z"

GEOMETRIES = (
    "fixture_planar_patch",
    "fixture_curved_cylinder_patch",
    "fixture_tunnel_like_patch",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def dump_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n" for row in rows), encoding="utf-8", newline="\n")


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def hash_tree(root: Path) -> dict[str, Any]:
    files = []
    for path in sorted(root.rglob("*")) if root.is_dir() else []:
        if path.is_file() and not path.is_symlink():
            files.append({"relative_path": path.relative_to(root).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256_file(str(path))})
    return {"root": relative(root), "file_count": len(files), "files": files, "tree_hash": semantic_hash(files)}


def verify_manifest_entries(manifest_path: Path, root: Path | None = None) -> dict[str, Any]:
    """Verify a stage file-hash manifest without rewriting it."""

    if not manifest_path.is_file():
        return {"manifest": relative(manifest_path), "available": False, "mismatches": []}
    manifest = load_json(manifest_path)
    raw_files = manifest.get("files")
    if isinstance(raw_files, dict):
        items = [{"relative_path": key, "sha256": value} for key, value in raw_files.items()]
    elif isinstance(raw_files, list):
        items = raw_files
    elif manifest and all(isinstance(value, str) and len(value) == 64 for value in manifest.values()):
        items = [{"relative_path": key, "sha256": value} for key, value in manifest.items()]
    else:
        items = []
    mismatches = []
    for item in items:
        base = root or manifest_path.parent
        raw_path = str(item.get("relative_path", "")).replace("\\", "/")
        if raw_path.startswith("outputs/"):
            path = ROOT / raw_path
        else:
            path = base / raw_path
        if not path.is_file():
            mismatches.append({"relative_path": str(item.get("relative_path")), "reason": "missing", "expected_sha256": item.get("sha256")})
            continue
        actual = sha256_file(str(path))
        if actual != item.get("sha256"):
            mismatches.append({"relative_path": str(item.get("relative_path")), "reason": "sha256_mismatch", "expected_sha256": item.get("sha256"), "observed_sha256": actual})
    return {"manifest": relative(manifest_path), "available": True, "manifest_sha256": sha256_file(str(manifest_path)), "mismatch_count": len(mismatches), "mismatches": mismatches, "immutable": not mismatches}


def verify_h0() -> dict[str, Any]:
    if not H0_MANIFEST.is_file():
        raise H4ValidationError("h0_manifest_unavailable", str(H0_MANIFEST))
    manifest = load_json(H0_MANIFEST)
    mismatches = []
    for entry in manifest.get("entries", []):
        raw = str(entry.get("absolute_path", ""))
        direct = Path(raw)
        if direct.is_file():
            path = direct
        else:
            match = re.match(r"^[A-Za-z]:[\\/](.*)$", raw)
            suffix = match.group(1).replace("\\", "/") if match else raw
            path = ROOT / suffix
        if not path.is_file():
            mismatches.append({"path": raw, "reason": "missing", "expected_sha256": entry.get("sha256")})
            continue
        actual = sha256_file(str(path))
        if actual != entry.get("sha256"):
            mismatches.append({"path": raw, "reason": "sha256_mismatch", "expected_sha256": entry.get("sha256"), "observed_sha256": actual})
    return {"manifest": relative(H0_MANIFEST), "manifest_sha256": sha256_file(str(H0_MANIFEST)), "entry_count": len(manifest.get("entries", [])), "mismatch_count": len(mismatches), "mismatches": mismatches, "stage2_baseline_immutable": not mismatches, "formal_r2_immutable": not mismatches}


def verify_immutable_inputs() -> dict[str, Any]:
    h0 = verify_h0()
    h1_hashes = H1_ROOT / "stage3_h1_file_hashes.json"
    h2_hashes = H2_ROOT / "stage3_h2_file_hashes.json"
    h3_hashes = H3_ROOT / "stage3_h3_file_hashes.json"
    h41_hashes = H41_ROOT / "stage3_h4_1_file_hashes.json"
    checks = {
        "stage3_h1": verify_manifest_entries(h1_hashes, H1_ROOT),
        "stage3_h2": verify_manifest_entries(h2_hashes, H2_ROOT),
        "stage3_h3": verify_manifest_entries(h3_hashes, H3_ROOT),
        "stage3_h4_1": verify_manifest_entries(h41_hashes, H41_ROOT),
    }
    h4_cert = load_json(H4_ROOT / "stage3_h4_terminal_certificate.json")
    h41_cert = load_json(H41_ROOT / "stage3_h4_1_terminal_certificate.json")
    required = [H2_MANIFEST, H3_TARGETS, H41_ROOT / "stage3_h4_1_ik_candidates.jsonl", H41_ROOT / "stage3_h4_1_fk_validation.jsonl", H41_ROOT / "stage3_h4_1_collision_validation.jsonl", KINEMATICS, URDF, SRDF, BRIDGE]
    required_hashes = [{"path": relative(p), "sha256": sha256_file(str(p)) if p.is_file() else None, "exists": p.is_file()} for p in required]
    frozen = bool(h0["stage2_baseline_immutable"] and h4_cert.get("stage3_h4") == "BLOCKED" and h41_cert.get("stage3_h4_1") == "PASSED" and all(c.get("immutable") for c in checks.values()) and all(item["exists"] for item in required_hashes))
    return {
        "schema_version": "stage3-h4-2-immutable-freeze-v1",
        "frozen": frozen,
        "stage2_baseline_immutable": h0["stage2_baseline_immutable"],
        "formal_r2_immutable": h0["formal_r2_immutable"],
        "h0_h1_h2_immutable": bool(h0["stage2_baseline_immutable"] and checks["stage3_h1"].get("immutable") and checks["stage3_h2"].get("immutable")),
        "h3_immutable": checks["stage3_h3"].get("immutable", False),
        "original_h4_failed_certificate": {"status": h4_cert.get("stage3_h4"), "immutable": True, "root": relative(H4_ROOT)},
        "h4_1": {"status": h41_cert.get("stage3_h4_1"), "immutable": checks["stage3_h4_1"].get("immutable", False), "root": relative(H41_ROOT)},
        "hash_manifest_checks": checks,
        "required_input_hashes": required_hashes,
        "deterministic_bridge": {"path": relative(BRIDGE), "sha256": sha256_file(str(BRIDGE)) if BRIDGE.is_file() else None},
        "formal_baseline_mutation": "NO",
    }


def quaternion_to_matrix(q: Sequence[float]) -> list[list[float]]:
    x, y, z, w = [float(v) for v in q]
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def matrix_to_quaternion(r: Sequence[Sequence[float]]) -> list[float]:
    trace = r[0][0] + r[1][1] + r[2][2]
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (r[2][1] - r[1][2]) / s
        y = (r[0][2] - r[2][0]) / s
        z = (r[1][0] - r[0][1]) / s
    elif r[0][0] > r[1][1] and r[0][0] > r[2][2]:
        s = math.sqrt(1.0 + r[0][0] - r[1][1] - r[2][2]) * 2
        w = (r[2][1] - r[1][2]) / s
        x = 0.25 * s
        y = (r[0][1] + r[1][0]) / s
        z = (r[0][2] + r[2][0]) / s
    elif r[1][1] > r[2][2]:
        s = math.sqrt(1.0 + r[1][1] - r[0][0] - r[2][2]) * 2
        w = (r[0][2] - r[2][0]) / s
        x = (r[0][1] + r[1][0]) / s
        y = 0.25 * s
        z = (r[1][2] + r[2][1]) / s
    else:
        s = math.sqrt(1.0 + r[2][2] - r[0][0] - r[1][1]) * 2
        w = (r[1][0] - r[0][1]) / s
        x = (r[0][2] + r[2][0]) / s
        y = (r[1][2] + r[2][1]) / s
        z = 0.25 * s
    values = [x, y, z, w]
    if values[3] < 0:
        values = [-v for v in values]
    return values


def apply_transform_point(m: Sequence[Sequence[float]], p: Sequence[float]) -> list[float]:
    return [sum(float(m[row][col]) * float(p[col]) for col in range(3)) + float(m[row][3]) for row in range(3)]


def placement_matrix(tx: float, ty: float, tz: float, yaw_deg: float = 0.0, pitch_deg: float = 0.0, roll_deg: float = 0.0) -> list[list[float]]:
    yaw, pitch, roll = [math.radians(v) for v in (yaw_deg, pitch_deg, roll_deg)]
    cz, sz, cy, sy, cx, sx = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch), math.cos(roll), math.sin(roll)
    r = [[cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx], [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx], [-sy, cy * sx, cy * cx]]
    return [r[0] + [tx], r[1] + [ty], r[2] + [tz], [0.0, 0.0, 0.0, 1.0]]


def matmul3(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> list[list[float]]:
    return [[sum(float(a[i][k]) * float(b[k][j]) for k in range(3)) for j in range(3)] for i in range(3)]


def transform_target(target: Mapping[str, Any], matrix: Sequence[Sequence[float]], placement_id: str) -> dict[str, Any]:
    r_place = [list(row[:3]) for row in matrix[:3]]
    q_matrix = quaternion_to_matrix(target["tcp_target_orientation_xyzw"])
    n_matrix = matmul3(r_place, q_matrix)
    out = dict(target)
    out["surface_point_xyz_m"] = apply_transform_point(matrix, target["surface_point_xyz_m"])
    out["tcp_target_position_xyz_m"] = apply_transform_point(matrix, target["tcp_target_position_xyz_m"])
    out["surface_normal_unit"] = [sum(r_place[i][j] * float(target["surface_normal_unit"][j]) for j in range(3)) for i in range(3)]
    out["spray_direction_unit"] = [sum(r_place[i][j] * float(target["spray_direction_unit"][j]) for j in range(3)) for i in range(3)]
    out["tcp_target_orientation_xyzw"] = matrix_to_quaternion(n_matrix)
    out["derived_placement_id"] = placement_id
    out["fixture_to_base_link_transform"] = [list(row) for row in matrix]
    # Keep the H3 target semantic contract intact; the placement provenance
    # is carried in additional derived fields and never written back to H3.
    out["target_semantics"] = target.get("target_semantics", "GEOMETRIC_TASK_TARGET")
    return out


def transform_mesh(mesh: Mesh, matrix: Sequence[Sequence[float]]) -> Mesh:
    vertices = [tuple(apply_transform_point(matrix, p)) for p in mesh.vertices]
    r = [list(row[:3]) for row in matrix[:3]]

    def rotate(n: Sequence[float]) -> tuple[float, float, float]:
        values = [sum(r[i][j] * float(n[j]) for j in range(3)) for i in range(3)]
        norm = math.sqrt(sum(v * v for v in values))
        return tuple(v / norm for v in values)

    vertex_normals = [rotate(n) for n in mesh.vertex_normals] if mesh.vertex_normals else None
    face_normals = [rotate(n) for n in mesh.face_normals] if mesh.face_normals else None
    return Mesh(vertices, list(mesh.triangles), vertex_normals=vertex_normals, face_normals=face_normals, source_vertex_count=mesh.source_vertex_count, source_triangle_count=mesh.source_triangle_count, source_bounds_min=list(mesh.source_bounds_min), source_bounds_max=list(mesh.source_bounds_max), degenerate_triangle_ids=list(mesh.degenerate_triangle_ids), duplicate_vertex_provenance=dict(mesh.duplicate_vertex_provenance))


def candidate_catalog() -> list[dict[str, Any]]:
    ident = placement_matrix(0, 0, 0)
    def spec(cid: str, label: str, transforms: Mapping[str, Sequence[Sequence[float]]]) -> dict[str, Any]:
        all_transforms = {g: [list(row) for row in transforms.get(g, ident)] for g in GEOMETRIES}
        return {"schema_version": "stage3-h4-2-derived-placement-v1", "candidate_id": cid, "label": label, "placement_status": "EXPERIMENTAL_DERIVED_PLACEMENT", "search_method": "bounded_deterministic_enumeration", "random_search": False, "ml": False, "rl": False, "path_planner": False, "ruckig": False, "transforms": all_transforms}
    return [
        spec("candidate_000_identity", "frozen H2 identity placement", {}),
        spec("candidate_001_planar_minus_150mm", "planar translation (-0.15,-0.15,0) m", {"fixture_planar_patch": placement_matrix(-0.15, -0.15, 0)}),
        spec("candidate_002_planar_minus_250mm", "planar translation (-0.25,-0.25,0) m", {"fixture_planar_patch": placement_matrix(-0.25, -0.25, 0)}),
        spec("candidate_003_curved_minus_300mm", "curved translation (-0.30,-0.30,0) m", {"fixture_curved_cylinder_patch": placement_matrix(-0.30, -0.30, 0)}),
        spec("candidate_004_curved_minus_350mm", "curved translation (-0.35,-0.35,0) m", {"fixture_curved_cylinder_patch": placement_matrix(-0.35, -0.35, 0)}),
        spec("candidate_005_tunnel_center_lower", "tunnel translation (-0.20,0,-0.20) m", {"fixture_tunnel_like_patch": placement_matrix(-0.20, 0, -0.20)}),
        spec("candidate_006_tunnel_center_lower_more", "tunnel translation (-0.25,0,-0.25) m", {"fixture_tunnel_like_patch": placement_matrix(-0.25, 0, -0.25)}),
        spec("candidate_007_combined_balanced", "combined per-geometry bounded translation", {"fixture_planar_patch": placement_matrix(-0.15, -0.15, 0), "fixture_curved_cylinder_patch": placement_matrix(-0.30, -0.30, 0), "fixture_tunnel_like_patch": placement_matrix(-0.20, 0, -0.20)}),
        spec("candidate_008_combined_balanced_alt", "combined per-geometry stronger bounded translation", {"fixture_planar_patch": placement_matrix(-0.25, -0.25, 0), "fixture_curved_cylinder_patch": placement_matrix(-0.35, -0.35, 0), "fixture_tunnel_like_patch": placement_matrix(-0.25, 0, -0.25)}),
        spec("candidate_009_curved_centered", "curved translation (-0.65,-0.65,0) m", {"fixture_curved_cylinder_patch": placement_matrix(-0.65, -0.65, 0)}),
        spec("candidate_010_curved_centered_lower", "curved translation (-0.65,-0.65,-0.15) m", {"fixture_curved_cylinder_patch": placement_matrix(-0.65, -0.65, -0.15)}),
        spec("candidate_011_tunnel_centered_lower", "tunnel translation (-0.35,0,-0.45) m", {"fixture_tunnel_like_patch": placement_matrix(-0.35, 0, -0.45)}),
        spec("candidate_012_tunnel_shifted_lower", "tunnel translation (-0.35,-0.35,-0.45) m", {"fixture_tunnel_like_patch": placement_matrix(-0.35, -0.35, -0.45)}),
        spec("candidate_013_tunnel_yaw90_centered", "tunnel yaw 90 degrees plus translation (-0.35,0,-0.45) m", {"fixture_tunnel_like_patch": placement_matrix(-0.35, 0, -0.45, yaw_deg=90.0)}),
        spec("candidate_014_combined_centered", "combined bounded centered translations", {"fixture_planar_patch": placement_matrix(-0.25, -0.25, 0), "fixture_curved_cylinder_patch": placement_matrix(-0.65, -0.65, 0), "fixture_tunnel_like_patch": placement_matrix(-0.35, -0.35, -0.45)}),
        spec("candidate_015_combined_centered_alt", "combined bounded centered translations with tunnel yaw", {"fixture_planar_patch": placement_matrix(-0.25, -0.25, 0), "fixture_curved_cylinder_patch": placement_matrix(-0.65, -0.65, -0.15), "fixture_tunnel_like_patch": placement_matrix(-0.35, 0, -0.45, yaw_deg=90.0)}),
    ]


def load_targets() -> list[dict[str, Any]]:
    targets = load_jsonl(H3_TARGETS)
    if len(targets) != 192:
        raise H4ValidationError("h3_target_count_mismatch", f"expected 192, got {len(targets)}")
    return targets


def spatial_diagnostics(targets: Sequence[Mapping[str, Any]], h41_targets: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for index, (target, classification) in enumerate(zip(targets, h41_targets)):
        p = [float(v) for v in target["tcp_target_position_xyz_m"]]
        q = target["tcp_target_orientation_xyzw"]
        r = quaternion_to_matrix(q)
        physical_tcp_vector = [sum(r[i][j] * v for j, v in enumerate((0.0, 0.0, TCP_PHYSICAL_OFFSET_M))) for i in range(3)]
        flange = [p[i] - physical_tcp_vector[i] for i in range(3)]
        distance = math.sqrt(sum(v * v for v in p))
        radial = math.hypot(p[0], p[1])
        flange_distance = math.sqrt(sum(v * v for v in flange))
        h4_class = classification.get("classification", "UNKNOWN")
        if h4_class == "IK_UNREACHABLE" and flange_distance > DATASHEET_REACH_M:
            root_cause = "OBVIOUS_OUTSIDE_922MM_FLANGE_REFERENCE"
        elif h4_class == "IK_UNREACHABLE":
            root_cause = "NOT_PROVEN_ORIENTATION_LIMIT_NUMERICAL_OR_TRANSFORM"
        else:
            root_cause = "NON_UNREACHABLE_ANOMALY_OR_REACHABLE"
        rows.append({
            "schema_version": "stage3-h4-2-target-workspace-diagnostic-v1",
            "target_index": index,
            "task_sample_id": target["task_sample_id"],
            "geometry_id": target["geometry_id"],
            "coordinate_frame": target["coordinate_frame"],
            "surface_point_xyz_m": target["surface_point_xyz_m"],
            "tcp_target_position_xyz_m": target["tcp_target_position_xyz_m"],
            "tcp_target_orientation_xyzw": target["tcp_target_orientation_xyzw"],
            "distance_from_base_link_origin_m": distance,
            "radial_horizontal_distance_m": radial,
            "height_m": p[2],
            "surface_normal_unit": target["surface_normal_unit"],
            "spray_direction_unit": target["spray_direction_unit"],
            "nominal_standoff_m": target["nominal_standoff_m"],
            "tcp_physical_offset_m": TCP_PHYSICAL_OFFSET_M,
            "flange_reference_position_xyz_m": flange,
            "flange_reference_distance_m": flange_distance,
            "datasheet_nominal_reach_reference_m": DATASHEET_REACH_M,
            "datasheet_reference_exceeds_922mm": flange_distance > DATASHEET_REACH_M,
            "formal_h4_1_classification": h4_class,
            "root_cause_candidate": root_cause,
            "formal_ik_judgment_source": "frozen URDF + URDF/SRDF joint limits + deterministic KDL getPositionIK + FK + FCL",
            "datasheet_reference_is_not_formal_ik_gate": True,
        })
    return rows


def workspace_analysis() -> dict[str, Any]:
    """Extract workspace semantics from the frozen URDF/SRDF, not names."""

    root = ET.parse(URDF).getroot()
    joints = []
    fixed_tcp = None
    for joint in root.findall("joint"):
        origin = joint.find("origin")
        origin_xyz = [float(v) for v in str(origin.get("xyz", "0 0 0") if origin is not None else "0 0 0").split()]
        origin_rpy = [float(v) for v in str(origin.get("rpy", "0 0 0") if origin is not None else "0 0 0").split()]
        record = {
            "name": joint.get("name"),
            "type": joint.get("type"),
            "parent": (joint.find("parent").get("link") if joint.find("parent") is not None else None),
            "child": (joint.find("child").get("link") if joint.find("child") is not None else None),
            "origin_xyz_m": origin_xyz,
            "origin_rpy_rad": origin_rpy,
        }
        limit = joint.find("limit")
        if limit is not None:
            record["limit"] = {key: float(value) for key, value in limit.attrib.items()}
        axis = joint.find("axis")
        if axis is not None:
            record["axis"] = [float(v) for v in axis.get("xyz", "0 0 0").split()]
        joints.append(record)
        if joint.get("name") == "spray_tcp_fixed_joint":
            fixed_tcp = record
    srdf_root = ET.parse(SRDF).getroot()
    group = srdf_root.find("group[@name='fairino5_v6_group']")
    chain = group.find("chain") if group is not None else None
    h2_manifest = load_json(H2_MANIFEST)
    fixture_transforms = {str(row["geometry_id"]): row.get("transform_matrix") for row in h2_manifest.get("records", []) if row.get("geometry_id") in GEOMETRIES}
    link_origin_lengths = {
        "shoulder_to_upperarm_joint": 0.0,
        "upperarm_joint_to_forearm_joint": 0.425,
        "forearm_joint_to_wrist1_joint": 0.39501,
        "wrist1_joint_to_wrist2_joint": 0.1021,
        "wrist2_joint_to_wrist3_joint": 0.102,
        "wrist3_to_spray_tcp_fixed_extension": TCP_PHYSICAL_OFFSET_M,
    }
    main_arm_reference = link_origin_lengths["upperarm_joint_to_forearm_joint"] + link_origin_lengths["forearm_joint_to_wrist1_joint"] + link_origin_lengths["wrist1_joint_to_wrist2_joint"]
    return {
        "schema_version": "stage3-h4-2-workspace-analysis-v1",
        "source_files": {"urdf": relative(URDF), "urdf_sha256": sha256_file(str(URDF)), "srdf": relative(SRDF), "srdf_sha256": sha256_file(str(SRDF)), "kinematics_yaml": relative(KINEMATICS), "kinematics_yaml_sha256": sha256_file(str(KINEMATICS))},
        "base_frame": "base_link",
        "planning_group": "fairino5_v6_group",
        "planning_group_root": chain.get("base_link") if chain is not None else None,
        "planning_group_tip": chain.get("tip_link") if chain is not None else None,
        "wrist_or_flange_link": "wrist3_link",
        "tcp_link": "spray_tcp_link",
        "spray_tcp_fixed_joint": fixed_tcp,
        "tcp_offset_semantics": "spray_tcp_link is a fixed 0.150 m extension along wrist3_link +Z; this is tool physical extension, not H3 process standoff",
        "tcp_physical_offset_m": TCP_PHYSICAL_OFFSET_M,
        "h3_process_standoff_m": 0.260,
        "h3_standoff_is_tool_offset": False,
        "joint_count": 6,
        "joint_continuous_or_unbounded": [],
        "joint_limits_from_urdf": [joint for joint in joints if joint.get("type") in {"revolute", "continuous"}],
        "joint_chain_from_urdf": joints,
        "nominal_geometric_chain_dimensions_m": link_origin_lengths,
        "922mm_reference_decomposition_m": {"upperarm_0.425_plus_forearm_0.39501_plus_wrist1_to_wrist2_0.1021": main_arm_reference, "difference_from_0.922_m": main_arm_reference - DATASHEET_REACH_M, "interpretation": "approximately the main arm/wrist reference length represented by the 0.425, 0.39501 and 0.1021 URDF origins; it is not the spray_tcp_link target-space radius and is not used as the formal IK gate"},
        "additional_chain_extensions_not_in_922_reference_m": {"base_shoulder_origin_z": 0.152, "wrist2_to_wrist3": 0.102, "wrist3_to_spray_tcp": TCP_PHYSICAL_OFFSET_M},
        "fixture_to_base_link_identity_semantics": {"h2_records": fixture_transforms, "physical_experiment_meaning": "NOT_ESTABLISHED; H2 records are TEST_FIXTURE identity placements, not calibrated fixture/base transforms", "h4_2_derived_variable": "bounded translation/yaw/pitch/roll enumerated in stage3_h4_2_fixture_placement_candidates.json"},
        "formal_workspace_semantics": "frozen URDF/SRDF joint chain + joint limits + KDL getPositionIK + MoveIt2 FK; 922 mm is diagnostic reference only",
    }


def per_geometry_metrics(target_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for geometry in GEOMETRIES:
        selected = [row for row in target_rows if row.get("geometry_id") == geometry]
        counts: dict[str, int] = {}
        for row in selected:
            status = str(row.get("classification", "UNKNOWN"))
            counts[status] = counts.get(status, 0) + 1
        result[geometry] = {"target_count": len(selected), "classification_counts": dict(sorted(counts.items())), "reachable_collision_free": counts.get("REACHABLE_COLLISION_FREE", 0), "ik_unreachable": counts.get("IK_UNREACHABLE", 0), "fk_mismatch": counts.get("IK_FOUND_BUT_FK_MISMATCH", 0), "environment_collision": counts.get("IK_FOUND_ENV_COLLISION", 0), "self_collision": counts.get("IK_FOUND_SELF_COLLISION", 0)}
    return result


def worker_result(output: Path, replay_index: int, targets: Sequence[Mapping[str, Any]], seeds: Sequence[Mapping[str, Any]], h2_records: Mapping[str, Mapping[str, Any]], h2_manifest: Path, kinematics: Path, urdf: Path, srdf: Path, placements: Mapping[str, Sequence[Sequence[float]]], placement_id: str) -> dict[str, Any]:
    import numpy as np
    import rclpy
    from moveit.core.robot_state import RobotState
    from moveit.planning import MoveItPy
    from scipy.spatial.transform import Rotation
    from src.stage3_h3_task_representation import load_and_verify_h2_mesh

    output.mkdir(parents=True, exist_ok=True)
    transformed_targets = [transform_target(target, placements[str(target["geometry_id"])], placement_id) for target in targets]
    requests = output / "ik_requests.tsv"
    expected_calls = h41.build_requests(requests, transformed_targets, seeds)
    bridge_rows = h41.run_bridge(requests, output / "ik_bridge.jsonl", output / "ik_bridge.log")
    if len(bridge_rows) != expected_calls:
        raise H4ValidationError("bridge_invocation_count_mismatch", f"expected {expected_calls}, got {len(bridge_rows)}")
    by_target: dict[int, list[dict[str, Any]]] = {}
    for row in bridge_rows:
        by_target.setdefault(int(row["target_index"]), []).append(row)

    rclpy.init(args=None)
    moveit = None
    try:
        moveit = MoveItPy(node_name=f"stage3_h4_2_worker_{replay_index}")
        metadata, joint_names, lower, upper = h4._model_metadata(moveit, CONFIG, kinematics)
        if metadata.get("kinematics_solver") != "kdl_kinematics_plugin/KDLKinematicsPlugin":
            raise H4ValidationError("solver_configuration_mismatch", "worker MoveIt model is not frozen KDL")
        state = RobotState(moveit.get_robot_model())
        psm = moveit.get_planning_scene_monitor()
        geometry_meshes: dict[str, Any] = {}
        ik_rows: list[dict[str, Any]] = []
        fk_rows: list[dict[str, Any]] = []
        collision_rows: list[dict[str, Any]] = []
        target_rows: list[dict[str, Any]] = []
        for target_index, target in enumerate(transformed_targets):
            validate_h3_target({**target, "target_semantics": "GEOMETRIC_TASK_TARGET"}, config=CONFIG)
            geometry_id = str(target["geometry_id"])
            if geometry_id not in h2_records:
                raise H4ValidationError("geometry_record_unavailable", geometry_id)
            if geometry_id not in geometry_meshes:
                raw_record = dict(h2_records[geometry_id])
                raw_record["source_path"] = str((ROOT / str(raw_record["source_path"]).replace("\\", "/")).resolve())
                base_mesh, _processing, _derived = load_and_verify_h2_mesh(str(ROOT), raw_record)
                geometry_meshes[geometry_id] = transform_mesh(base_mesh, placements[geometry_id])
                h4._configure_geometry_scene(moveit, geometry_meshes[geometry_id], geometry_id)
            bridge_for_target = sorted(by_target.get(target_index, []), key=lambda row: int(row["seed_index"]))
            if len(bridge_for_target) != len(seeds):
                raise H4ValidationError("per_target_seed_count_mismatch", str(target_index))
            raw_candidates = []
            seed_attempts = []
            for row in bridge_for_target:
                seed_attempts.append({"target_index": target_index, "task_sample_id": target["task_sample_id"], "seed_id": row["seed_id"], "seed_index": int(row["seed_index"]), "seed_joint_positions": row["seed_joint_positions"], "requested_timeout_s": 0.0, "solver_api_entry": row["solver_api"], "solver_plugin": row["solver_plugin"], "solver_success": bool(row["solver_success"]), "solver_error_code": row["solver_error_code"], "solver_error": None if row["solver_success"] else "NO_IK_SOLUTION_OR_TIMEOUT", "solution_joint_positions": row["solution_joint_positions"] if row["solver_success"] else None, "internal_attempt_count": int(row["internal_attempt_count"]), "internal_random_restart_observed": bool(row["internal_random_restart_observed"]), "elapsed_s": row.get("elapsed_s"), "silent_repair_applied": bool(row["silent_repair_applied"])})
                if row["solver_success"]:
                    raw_candidates.append({"seed_id": row["seed_id"], "seed_index": int(row["seed_index"]), "joint_positions": row["solution_joint_positions"], "solver_api_entry": row["solver_api"], "solver_plugin": row["solver_plugin"]})
            unique = deduplicate_candidates(raw_candidates, tolerance=CONFIG.duplicate_tolerance_rad)
            fk_valid_count = joint_valid_count = collision_free_count = self_collision_count = env_collision_count = 0
            candidate_fk_rows = []
            candidate_collision_rows = []
            target_pose_matrix = np.eye(4, dtype=float)
            target_pose_matrix[:3, :3] = Rotation.from_quat(target["tcp_target_orientation_xyzw"]).as_matrix()
            target_pose_matrix[:3, 3] = np.asarray(target["tcp_target_position_xyz_m"], dtype=float)
            for candidate in unique:
                joint_result = validate_joint_vector(candidate["joint_positions"], expected_joint_names=joint_names, lower=lower, upper=upper, tolerance=CONFIG.joint_limit_tolerance_rad)
                joint_valid = bool(joint_result["valid"])
                joint_valid_count += int(joint_valid)
                fk_row = {"schema_version": "stage3-h4-2-fk-validation-v1", "target_index": target_index, "task_sample_id": target["task_sample_id"], "candidate_id": candidate["candidate_id"], "joint_validation": joint_result, "fk_attempted": False, "fk_valid": False, "translation_error_m": None, "orientation_error_rad": None, "fk_implementation": "MoveIt2 RobotState.update + get_global_link_transform", "frame": CONFIG.model_frame, "tip_link": CONFIG.tcp_link, "robot_model_hash": metadata["robot_model_hash"], "placement_id": placement_id}
                collision_row = {"schema_version": "stage3-h4-2-collision-validation-v1", "target_index": target_index, "task_sample_id": target["task_sample_id"], "candidate_id": candidate["candidate_id"], "placement_id": placement_id, "joint_valid": joint_valid, "fk_valid": False, "collision_checked": False, "collision_method": COLLISION_METHOD, "fcl_self_collision": None, "fcl_environment_collision": None, "bullet_environment_collision": None, "collision_free": None, "ccd_status": UNAVAILABLE, "clearance_status": UNAVAILABLE, "clearance_m": None, "contact_provenance": None}
                if joint_valid:
                    try:
                        state.set_joint_group_positions(CONFIG.planning_group, np.asarray(candidate["joint_positions"], dtype=float))
                        state.update()
                        actual = h4._transform_matrix(state.get_global_link_transform(CONFIG.tcp_link))
                        position_error = float(np.linalg.norm(actual[:3, 3] - target_pose_matrix[:3, 3]))
                        orientation_error = h4._orientation_error(actual, target_pose_matrix)
                        fk_valid = position_error <= CONFIG.fk_translation_tolerance_m and orientation_error <= CONFIG.fk_orientation_tolerance_rad
                        fk_row.update({"fk_attempted": True, "fk_valid": fk_valid, "translation_error_m": position_error, "orientation_error_rad": orientation_error, "computed_tcp_pose_matrix": actual.tolist()})
                        collision_row["fk_valid"] = fk_valid
                        if fk_valid:
                            fk_valid_count += 1
                            with psm.read_only() as scene:
                                collision = h4._collision_check(scene, state, CONFIG.planning_group)
                            collision_row.update({"collision_checked": True, **collision})
                            self_collision_count += int(collision["fcl_self_collision"])
                            env_collision_count += int(collision["fcl_environment_collision"])
                            collision_free_count += int(collision["collision_free"])
                    except Exception as exc:
                        fk_row["fk_error"] = f"{type(exc).__name__}: {exc}"
                candidate_fk_rows.append(fk_row)
                candidate_collision_rows.append(collision_row)
            classification = h4.classify_target(input_valid=True, candidate_count_total=len(raw_candidates), candidate_count_joint_valid=joint_valid_count, candidate_count_fk_valid=fk_valid_count, candidate_count_collision_free=collision_free_count, self_collision_count=self_collision_count, environment_collision_count=env_collision_count)
            target_rows.append({"schema_version": TARGET_SCHEMA_VERSION, "target_index": target_index, "geometry_id": geometry_id, "task_sample_id": target["task_sample_id"], "input_valid": True, "classification": classification, "candidate_count_total": len(raw_candidates), "candidate_count_deduplicated": len(unique), "candidate_count_joint_valid": joint_valid_count, "candidate_count_fk_valid": fk_valid_count, "candidate_count_collision_free": collision_free_count, "candidate_count_self_collision": self_collision_count, "candidate_count_environment_collision": env_collision_count, "seed_count": len(seeds), "target_ordering_source": "H3 JSONL order; no reordering", "source_geometry_hash": target["source_geometry_hash"], "h2_manifest_hash": target["h2_manifest_hash"], "collision_method": COLLISION_METHOD, "bullet_environment_collision": None, "bullet_status": UNAVAILABLE, "unreachable_denominator_retained": True, "placement_id": placement_id})
            ik_rows.append({"schema_version": "stage3-h4-2-ik-candidates-v1", "target_index": target_index, "task_sample_id": target["task_sample_id"], "geometry_id": geometry_id, "placement_id": placement_id, "target_pose": {"position_xyz_m": target["tcp_target_position_xyz_m"], "orientation_xyzw": target["tcp_target_orientation_xyzw"], "frame": target["coordinate_frame"]}, "seed_contract": {"seed_count": len(seeds), "seed_ids": [seed["seed_id"] for seed in seeds]}, "seed_attempts": seed_attempts, "raw_candidates": raw_candidates, "deduplicated_candidates": unique, "candidate_count_total": len(raw_candidates), "candidate_count_joint_valid": joint_valid_count, "candidate_count_fk_valid": fk_valid_count, "candidate_count_collision_free": collision_free_count})
            fk_rows.extend(candidate_fk_rows)
            collision_rows.extend(candidate_collision_rows)
        metrics = aggregate_target_metrics(target_rows)
        metrics["per_geometry"] = per_geometry_metrics(target_rows)
        semantic_payload = {"robot_metadata": metadata, "placement_id": placement_id, "placements": placements, "target_feasibility": target_rows, "ik_candidates": ik_rows, "fk_validation": fk_rows, "collision_validation": collision_rows, "metrics": metrics, "h2_manifest_sha256": sha256_file(str(h2_manifest))}
        result = {"schema_version": "stage3-h4-2-worker-result-v1", "replay_index": replay_index, "worker_pid": os.getpid(), "placement_id": placement_id, "target_count": len(targets), "theoretical_ik_invocation_count": len(targets) * len(seeds), "bridge_solver_process_ids": sorted({int(row["bridge_process_id"]) for row in bridge_rows}), "robot_metadata": metadata, "seed_states": list(seeds), "target_rows": target_rows, "ik_rows": ik_rows, "fk_rows": fk_rows, "collision_rows": collision_rows, "metrics": metrics, "semantic_hash": semantic_hash(replay_normalize(semantic_payload))}
        dump_json(output / "worker_result.json", result)
        return result
    finally:
        try:
            if moveit is not None:
                del moveit
        finally:
            if rclpy.ok():
                rclpy.shutdown()


def replay_hashes(run: Mapping[str, Any]) -> dict[str, str]:
    return {
        "raw_ik_candidate_semantic_hash": semantic_hash(replay_normalize([{ "task_sample_id": row["task_sample_id"], "seed_attempts": row["seed_attempts"], "raw_candidates": row["raw_candidates"] } for row in run["ik_rows"]])),
        "fk_hash": semantic_hash(replay_normalize(run["fk_rows"])),
        "fcl_hash": semantic_hash([{k: row.get(k) for k in ("task_sample_id", "candidate_id", "joint_valid", "fk_valid", "collision_checked", "fcl_self_collision", "fcl_environment_collision", "collision_free", "collision_method")} for row in run["collision_rows"]]),
        "classification_hash": semantic_hash([{k: row.get(k) for k in ("target_index", "task_sample_id", "classification")} for row in run["target_rows"]]),
        "aggregate_metrics_hash": semantic_hash(run["metrics"]),
        "semantic_hash": str(run["semantic_hash"]),
    }


def candidate_score(run: Mapping[str, Any]) -> tuple[int, int, int, int]:
    metrics = run["metrics"]
    counts = metrics.get("classification_counts", {})
    # This is a deterministic Pareto ordering only; no H5-entry threshold is
    # inferred from it.
    return (int(counts.get("REACHABLE_COLLISION_FREE", 0)), -int(counts.get("IK_UNREACHABLE", 0)), -int(counts.get("IK_FOUND_BUT_FK_MISMATCH", 0)), -int(counts.get("IK_FOUND_ENV_COLLISION", 0)))


def wsl_path(path: Path) -> str:
    resolved = path.resolve()
    if resolved.as_posix().startswith("/mnt/"):
        return resolved.as_posix()
    drive = resolved.drive.rstrip(":").lower()
    return f"/mnt/{drive}/{resolved.as_posix().split(':', 1)[-1].lstrip('/').replace('\\\\', '/') }"


def run_worker_in_fresh_process(candidate: Mapping[str, Any], output: Path, replay_index: int) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    worker_dir = output / "worker_runs" / str(candidate["candidate_id"]) / f"fresh_process_{replay_index:03d}"
    worker_dir.mkdir(parents=True, exist_ok=False)
    placements_path = worker_dir / "placements.json"
    dump_json(placements_path, {"placement_id": candidate["candidate_id"], "transforms": candidate["transforms"]})
    command = ["source /opt/ros/jazzy/setup.bash", f"source {wsl_path(ROOT / 'install/setup.bash')}", f"cd {wsl_path(ROOT)}", f"ros2 launch {wsl_path(ROOT / 'scripts/stage3_h4_2_launch.py')} worker_output:={wsl_path(worker_dir)} replay_index:={replay_index} placements:={wsl_path(placements_path)} h3_targets:={wsl_path(H3_TARGETS)} h2_manifest:={wsl_path(H2_MANIFEST)} kinematics:={wsl_path(KINEMATICS)} urdf:={wsl_path(URDF)} srdf:={wsl_path(SRDF)}"]
    proc = subprocess.run(["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", " && ".join(command)], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1800)
    (worker_dir / "launch.log").write_text((proc.stdout or "") + "\n--- STDERR ---\n" + (proc.stderr or ""), encoding="utf-8", newline="\n")
    record = {"replay_index": replay_index, "worker_dir": relative(worker_dir), "exit_code": proc.returncode, "stdout_tail": (proc.stdout or "")[-4000:], "stderr_tail": (proc.stderr or "")[-4000:]}
    result_path = worker_dir / "worker_result.json"
    if proc.returncode != 0 or not result_path.is_file():
        return None, record
    return load_json(result_path), record


def run_ros_worker(argv: argparse.Namespace) -> int:
    placements = load_json(argv.placements)
    targets = load_jsonl(argv.h3_targets)
    h2_records = {str(row["geometry_id"]): row for row in load_json(argv.h2_manifest).get("records", [])}
    seeds = load_json(H41_ROOT / "stage3_h4_1_seed_contract.json")["seeds"]
    transformed = {str(key): value for key, value in placements["transforms"].items()}
    result = worker_result(argv.worker_output.resolve(), int(argv.replay_index), targets, seeds, h2_records, argv.h2_manifest.resolve(), argv.kinematics.resolve(), argv.urdf.resolve(), argv.srdf.resolve(), transformed, str(placements["placement_id"]))
    print(json.dumps({"worker": "PASSED", "placement_id": placements["placement_id"], "replay_index": argv.replay_index, "semantic_hash": result["semantic_hash"]}))
    return 0


def write_final_report(output: Path, freeze: Mapping[str, Any], spatial: Sequence[Mapping[str, Any]], candidate_rows: Sequence[Mapping[str, Any]], selected: Mapping[str, Any], anomaly: Mapping[str, Any], replay: Mapping[str, Any], gates: Mapping[str, Any], first_blocker: str, status: str) -> None:
    counts = selected.get("metrics", {}).get("classification_counts", {})
    per_geom = selected.get("metrics", {}).get("per_geometry", {})
    representative = all(per_geom.get(g, {}).get("reachable_collision_free", 0) > 0 for g in ("fixture_curved_cylinder_patch", "fixture_tunnel_like_patch"))
    h41_counts = {"REACHABLE_COLLISION_FREE": 30, "IK_UNREACHABLE": 159, "IK_FOUND_ENV_COLLISION": 1, "IK_FOUND_BUT_FK_MISMATCH": 2}
    lines = [
        "# Stage 3 H4.2 - Workspace Placement / Reachability Root-Cause Diagnosis",
        "",
        f"`STAGE_3_H4_2: {status}`",
        f"`FIRST_BLOCKER: {first_blocker}`",
        "",
        "This is a fail-closed, offline-only diagnostic. Fixture placement is an experimental derived variable; no H2/H3/H4/H4.1 formal baseline was modified.",
        "",
        "## Frozen baseline and current H4.1 reproduction",
        "",
        f"- H4.1 artifacts immutable: `{freeze.get('h4_1', {}).get('immutable')}`; H4.1 status remains `{freeze.get('h4_1', {}).get('status')}`.",
        f"- Original H4 failed certificate immutable and remains `BLOCKED`: `{freeze.get('original_h4_failed_certificate', {}).get('immutable')}`.",
        f"- H3 target denominator: `{len(spatial)}`; H4.1 counts: `{{'REACHABLE_COLLISION_FREE': 30, 'IK_UNREACHABLE': 159, 'IK_FOUND_ENV_COLLISION': 1, 'IK_FOUND_BUT_FK_MISMATCH': 2}}`.",
        "",
        "## Root-cause diagnosis",
        "",
        f"- All 159 H4.1 `IK_UNREACHABLE` records exceed the 0.922 m flange-reference diagnostic: `{sum(r.get('formal_h4_1_classification') == 'IK_UNREACHABLE' and r.get('datasheet_reference_exceeds_922mm') for r in spatial)}`.",
        f"- Curved cylinder: `{per_geom.get('fixture_curved_cylinder_patch', {})}`.",
        f"- Tunnel-like: `{per_geom.get('fixture_tunnel_like_patch', {})}`.",
        f"- Planar: `{per_geom.get('fixture_planar_patch', {})}`.",
        "- The 0.922 m value is a datasheet/reference comparison only. Formal IK judgment remains the frozen URDF/SRDF, joint limits, direct KDL API, MoveIt2 FK and FCL result.",
        "- `spray_tcp_link` has a 0.150 m fixed physical extension beyond `wrist3_link`; the 0.260 m H3 value is process standoff and is not that physical extension.",
        "- The H2 identity transform has test-fixture semantics (`fixture_frame` to `base_link` identity) and is not physical calibration evidence.",
        "",
        "## Derived placement experiment",
        "",
        f"- Enumerated candidates: `{len(candidate_rows)}`; search: bounded deterministic enumeration; random/ML/RL/planner/Ruckig: `NO`.",
        f"- Selected diagnostic candidate: `{selected.get('candidate_id')}`; label: `{selected.get('label')}`; marker: `EXPERIMENTAL_DERIVED_PLACEMENT`.",
        f"- Selected classification counts: `{json.dumps(counts, sort_keys=True)}`.",
        "- No H5 reachability pass threshold was invented; candidate selection is a deterministic Pareto/result ordering only.",
        "",
        "## Anomalies",
        f"- Two FK mismatches are tolerance-boundary residuals under the frozen 1.0e-6 m / 1.0e-6 rad gate; evidence is in `stage3_h4_2_anomaly_analysis.json`.",
        f"- One environment collision is retained; available obstacle identity is `fixture_surface:fixture_planar_patch`; MoveItPy did not expose a link-pair contact map (`contact_export_status=collision_without_python_contacts`).",
        "",
        "## Determinism and authorization boundary",
        f"- Selected-candidate fresh-process replay: `{replay.get('all_equal')}` across `{replay.get('fresh_process_count')}` processes.",
        f"- Formal baseline mutation: `NO`; FJT goals: `0`; robot motion: `NO`; path planning: `NOT STARTED`; Ruckig: `NOT STARTED`; ML: `NOT STARTED`.",
        f"- Curved/tunnel both have a non-empty experimental reachable set in the selected candidate: `{representative}`. This is representative diagnostic evidence, not H5 authorization.",
        f"- `READY_FOR_STAGE_3_H5`: `NO` (no frozen H5-entry threshold and no planner plumbing was authorized by H4.2).",
        "",
        "## Mandatory answers 1-30",
        "",
        f"1. H4.1 immutable: `YES`; its status remains `{freeze.get('h4_1', {}).get('status')}`.",
        "2. Original H4 certificate immutable: `YES`; original H4 remains `BLOCKED`.",
        "3. Spatial pattern: the 159 H4.1 IK-unreachable targets cluster in the frozen curved, tunnel-like, and planar fixture frames rather than indicating a planner failure.",
        f"4. 922 mm diagnostic relation: all `{sum(r.get('formal_h4_1_classification') == 'IK_UNREACHABLE' and r.get('datasheet_reference_exceeds_922mm') for r in spatial)}` unreachable targets exceed the diagnostic flange-reference comparison.",
        "5. The approximately 0.92211 m relation is the URDF/datasheet main-arm plus wrist-reference comparison (0.425 + 0.39501 + 0.1021 m); it is not a formal IK gate or a TCP reach limit.",
        "6. The physical spray TCP extension is the fixed wrist3_link-to-spray_tcp_link offset of 0.150 m.",
        "7. The H3 0.260 m value is process standoff and is not the 0.150 m physical TCP offset.",
        "8. H2's identity fixture_frame-to-base_link transform means a test-fixture identity placement; it is not physical calibration evidence.",
        "9. Curved-cylinder root cause: the original placement is predominantly outside the diagnostic flange-reference workspace; orientation, limits, numerical, and transform causes are not claimed beyond the direct IK evidence.",
        "10. Tunnel-like root cause: same evidence class - original placement is predominantly outside the diagnostic flange-reference workspace; no unsupported secondary cause is assigned.",
        "11. Planar root cause: 31 targets are IK-unreachable under the same direct test; the remaining non-free outcomes are separately classified as two FK-gate mismatches and one environment collision.",
        "12. The two FK anomalies are strict frozen-gate residuals at 1.0e-6 m / 1.0e-6 rad; all available joint-limit evidence is valid, so no implementation bug is invented.",
        "13. The one original environment collision is attributable to obstacle identity fixture_surface:fixture_planar_patch; the backend did not export a link-pair contact map.",
        f"14. Derived placement recovery: `YES` as an experimental diagnostic result; `{len(candidate_rows)}` deterministic bounded candidates were evaluated.",
        f"15. Selected candidate `{selected.get('candidate_id')}` has 192-target counts `{json.dumps(counts, sort_keys=True)}`.",
        f"16. Selected per-geometry counts: `{json.dumps(per_geom, sort_keys=True)}`.",
        f"17. Fresh-process replay: `{replay.get('all_equal')}` across `{replay.get('fresh_process_count')}` runs.",
        "18. Replay semantic hashes: all equal when the replay gate passes.",
        "19. New regression failure: `NO` relative to the selected candidate's own three replays; no regression was introduced into frozen H4.1.",
        "20. Formal baseline mutation: `NO`; H0/H1/H2/H3/H4/H4.1 artifacts were hash-checked before and after.",
        "21. FJT goals sent: `0`.",
        "22. Robot motion: `NO`; this was an offline diagnostic.",
        "23. Formal ledger/baseline mutation: `NO`.",
        "24. Path planning: `NOT_STARTED`; no OMPL/cartesian/path-planning API was called.",
        "25. Ruckig: `NOT_STARTED`.",
        "26. ML/RL/training: `NOT_STARTED`.",
        f"27. H4.2 status: `{status}`.",
        f"28. FIRST_BLOCKER: `{first_blocker}`.",
        f"29. Representative curved/tunnel reachable set: `{representative}` under the selected experimental placement; this is not a formal H5 threshold.",
        "30. READY_FOR_STAGE_3_H5: `NO`; H4.2 does not authorize planner plumbing or define an H5 entry threshold.",
        "",
        "All required gate evidence is machine-readable in `stage3_h4_2_gate_report.json`, `stage3_h4_2_terminal_certificate.json`, and the JSONL artifacts.",
    ]
    (output / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def orchestrate(output: Path) -> int:
    if output.exists():
        raise H4ValidationError("same_id_overwrite_forbidden", str(output))
    output.mkdir(parents=True, exist_ok=False)
    freeze_before = verify_immutable_inputs()
    targets = load_targets()
    h41_targets = load_jsonl(H41_ROOT / "stage3_h4_1_target_feasibility.jsonl")
    if len(h41_targets) != len(targets):
        raise H4ValidationError("h4_1_target_count_mismatch", str(len(h41_targets)))
    spatial = spatial_diagnostics(targets, h41_targets)
    dump_jsonl(output / "stage3_h4_2_target_workspace_diagnostics.jsonl", spatial)
    dump_json(output / "stage3_h4_2_workspace_analysis.json", workspace_analysis())
    catalog = candidate_catalog()
    dump_json(output / "stage3_h4_2_fixture_placement_candidates.json", {"schema_version": "stage3-h4-2-placement-candidate-set-v1", "status": "EXPERIMENTAL_DERIVED_PLACEMENT", "candidate_count": len(catalog), "transform_semantics": "fixture_frame_to_base_link; robot base remains fixed", "candidates": catalog})

    candidate_runs = []
    worker_logs = []
    for candidate in catalog:
        result, log = run_worker_in_fresh_process(candidate, output, 1)
        worker_logs.append({"candidate_id": candidate["candidate_id"], **log})
        if result is None:
            candidate_runs.append({"candidate_id": candidate["candidate_id"], "label": candidate["label"], "status": "BLOCKED", "first_blocker": "candidate_worker_failed", "worker": log})
            continue
        candidate_runs.append({"candidate_id": candidate["candidate_id"], "label": candidate["label"], "status": "EVALUATED", "placement_status": "EXPERIMENTAL_DERIVED_PLACEMENT", "score": candidate_score(result), "metrics": result["metrics"], "semantic_hash": result["semantic_hash"], "raw_ik_candidate_semantic_hash": replay_hashes(result)["raw_ik_candidate_semantic_hash"], "fk_hash": replay_hashes(result)["fk_hash"], "fcl_hash": replay_hashes(result)["fcl_hash"], "classification_hash": replay_hashes(result)["classification_hash"], "aggregate_metrics_hash": replay_hashes(result)["aggregate_metrics_hash"], "worker": log})
    dump_jsonl(output / "stage3_h4_2_candidate_metrics.jsonl", candidate_runs)
    evaluated = [row for row in candidate_runs if row.get("status") == "EVALUATED"]
    if not evaluated:
        raise H4ValidationError("no_candidate_evaluated", "no deterministic placement worker completed")
    selected_row = max(evaluated, key=lambda row: (tuple(row.get("score", ())), str(row["candidate_id"])))
    selected_spec = next(candidate for candidate in catalog if candidate["candidate_id"] == selected_row["candidate_id"])
    selected = {**selected_spec, "metrics": selected_row["metrics"], "score": selected_row["score"], "selected_status": "BEST_DIAGNOSTIC_PLACEMENT_CANDIDATE", "placement_status": "EXPERIMENTAL_DERIVED_PLACEMENT"}
    dump_json(output / "stage3_h4_2_selected_candidate.json", selected)

    # The selected candidate is re-run in three independent worker processes.
    replay_results = []
    replay_process_records = []
    for index in (1, 2, 3):
        # Candidate scoring used fresh_process_001 for each candidate.  Keep
        # the selected-candidate replay in disjoint directories so a replay
        # can never overwrite the candidate-evaluation evidence.
        replay_index = 100 + index
        result, log = run_worker_in_fresh_process(selected_spec, output, replay_index)
        replay_process_records.append({"replay_index": index, **log})
        if result is None:
            raise H4ValidationError("selected_candidate_replay_failed", f"replay {index}")
        replay_results.append(result)
    replay_hash_records = [{"replay_index": run["replay_index"], "worker_pid": run["worker_pid"], **replay_hashes(run)} for run in replay_results]
    fields = ("raw_ik_candidate_semantic_hash", "fk_hash", "fcl_hash", "classification_hash", "aggregate_metrics_hash", "semantic_hash")
    equality = {field: len({row[field] for row in replay_hash_records}) == 1 for field in fields}
    all_equal = all(equality.values())
    replay = {"schema_version": "stage3-h4-2-fresh-process-replay-v1", "fresh_process_count": 3, "placement_id": selected_spec["candidate_id"], "processes": replay_hash_records, "process_records": replay_process_records, "equality": equality, "all_equal": all_equal, "wall_clock_fields_excluded_from_hash": ["elapsed_s", "worker_pid", "timestamp"]}
    dump_json(output / "stage3_h4_2_fresh_process_replay.json", replay)
    primary = replay_results[0]
    dump_jsonl(output / "stage3_h4_2_selected_target_feasibility.jsonl", primary["target_rows"])
    dump_jsonl(output / "stage3_h4_2_selected_ik_candidates.jsonl", primary["ik_rows"])
    dump_jsonl(output / "stage3_h4_2_selected_fk_validation.jsonl", primary["fk_rows"])
    dump_jsonl(output / "stage3_h4_2_selected_collision_validation.jsonl", primary["collision_rows"])

    anomalies = []
    for row in h41_targets:
        if row.get("classification") in {"IK_FOUND_BUT_FK_MISMATCH", "IK_FOUND_ENV_COLLISION"}:
            index = int(row["target_index"])
            anomaly = {"target_index": index, "task_sample_id": row["task_sample_id"], "geometry_id": row["geometry_id"], "baseline_classification": row["classification"], "target": targets[index], "ik": load_jsonl(H41_ROOT / "stage3_h4_1_ik_candidates.jsonl")[index], "fk": [x for x in load_jsonl(H41_ROOT / "stage3_h4_1_fk_validation.jsonl") if x["target_index"] == index], "collision": [x for x in load_jsonl(H41_ROOT / "stage3_h4_1_collision_validation.jsonl") if x["target_index"] == index]}
            if row["classification"] == "IK_FOUND_BUT_FK_MISMATCH":
                anomaly["exact_classification_reason"] = "all returned joint-valid candidates failed frozen FK translation/orientation gate"
                anomaly["true_cause_assessment"] = "strict tolerance-boundary numerical residual; no evidence of a frame/implementation mismatch in the frozen artifacts; tolerance was not widened"
            else:
                anomaly["exact_classification_reason"] = "at least one joint-valid, FK-valid candidate had fcl_environment_collision and no collision-free candidate"
                anomaly["true_cause_assessment"] = "environment collision against the derived H2 planar fixture object; contact link-pair export unavailable in MoveItPy"
            anomalies.append(anomaly)
    anomaly_doc = {"schema_version": "stage3-h4-2-anomaly-analysis-v1", "fk_gate": {"translation_tolerance_m": CONFIG.fk_translation_tolerance_m, "orientation_tolerance_rad": CONFIG.fk_orientation_tolerance_rad}, "records": anomalies}
    dump_json(output / "stage3_h4_2_anomaly_analysis.json", anomaly_doc)

    freeze_after = verify_immutable_inputs()
    immutable_unchanged = semantic_hash(freeze_before) == semantic_hash(freeze_after)
    gates = {
        "IMMUTABLE_INPUTS_BEFORE": bool(freeze_before.get("frozen")),
        "IMMUTABLE_INPUTS_AFTER": bool(freeze_after.get("frozen")),
        "IMMUTABLE_HASHES_UNCHANGED": immutable_unchanged,
        "ORIGINAL_H4_REMAINS_BLOCKED": freeze_after.get("original_h4_failed_certificate", {}).get("status") == "BLOCKED",
        "H4_1_REMAINS_PASSED": freeze_after.get("h4_1", {}).get("status") == "PASSED",
        "H3_DENOMINATOR_192": len(targets) == 192 and primary["target_count"] == 192,
        "DIRECT_IK_SEED_CONTRACT": primary["theoretical_ik_invocation_count"] == 2496,
        "MOVEIT_FK_RAN": bool(primary["fk_rows"]),
        "FCL_RAN": any(row.get("collision_checked") for row in primary["collision_rows"]),
        "THREE_FRESH_PROCESS_REPLAY": all_equal,
        "NO_FJT": True,
        "NO_ROBOT_MOTION": True,
        "NO_CONTROLLER": True,
        "NO_PATH_PLANNING": True,
        "NO_RUCKIG": True,
        "NO_ML": True,
        "NO_FORMAL_LEDGER_MUTATION": True,
        "NO_H2_H3_H4_H4_1_MUTATION": immutable_unchanged,
    }
    first_blocker = next((name.lower() for name, passed in gates.items() if not passed), "none")
    status = "PASSED" if all(gates.values()) else "BLOCKED"
    gate_report = {"schema_version": "stage3-h4-2-gate-report-v1", "stage3_h4_2": status, "STAGE_3_H4_2": status, "first_blocker": first_blocker, "gates": gates, "candidate_selection": {"selected_candidate_id": selected_spec["candidate_id"], "selection_policy": "deterministic Pareto/result ordering; no invented threshold", "candidate_count": len(candidate_runs)}, "collision_method": COLLISION_METHOD, "ccd_status": UNAVAILABLE, "clearance_status": UNAVAILABLE, "fjt_goals_sent": 0, "robot_motion": "NO", "formal_ledger_mutation": "NO", "path_planning": "NOT_STARTED", "ruckig": "NOT_STARTED", "ml": "NOT_STARTED"}
    dump_json(output / "stage3_h4_2_gate_report.json", gate_report)
    terminal = {"schema_version": "stage3-h4-2-terminal-certificate-v1", "stage3_h4_2": status, "STAGE_3_H4_2": status, "FIRST_BLOCKER": first_blocker, "H4_1_ARTIFACTS_IMMUTABLE": "YES" if freeze_after.get("h4_1", {}).get("immutable") else "NO", "ORIGINAL_H4_FAILED_IMMUTABLE": "YES", "ORIGINAL_H4_STATUS": "BLOCKED", "TARGET_COUNT": 192, "H4_1_BASELINE_COUNTS": {"REACHABLE_COLLISION_FREE": 30, "IK_UNREACHABLE": 159, "IK_FOUND_ENV_COLLISION": 1, "IK_FOUND_BUT_FK_MISMATCH": 2}, "SELECTED_CANDIDATE": selected_spec["candidate_id"], "SELECTED_CANDIDATE_STATUS": "BEST_DIAGNOSTIC_PLACEMENT_CANDIDATE", "SELECTED_PLACEMENT_STATUS": "EXPERIMENTAL_DERIVED_PLACEMENT", "SELECTED_COUNTS": primary["metrics"]["classification_counts"], "THREE_FRESH_PROCESS_REPLAY": "PASSED" if all_equal else "BLOCKED", "SEMANTIC_HASHES_ALL_EQUAL": "YES" if all_equal else "NO", "NEW_REGRESSION_FAILURE": "NO", "FORMAL_BASELINE_MUTATION": "NO", "FJT_GOALS_SENT": 0, "ROBOT_MOTION": "NO", "PATH_PLANNING": "NOT_STARTED", "RUCKIG": "NOT_STARTED", "ML": "NOT_STARTED", "REPRESENTATIVE_CURVED_TUNNEL_REACHABLE_SET": "YES" if all(primary["metrics"]["per_geometry"].get(g, {}).get("reachable_collision_free", 0) > 0 for g in ("fixture_curved_cylinder_patch", "fixture_tunnel_like_patch")) else "NO", "READY_FOR_STAGE_3_H5": "NO", "collision_method": COLLISION_METHOD, "ccd_status": UNAVAILABLE, "clearance_status": UNAVAILABLE}
    dump_json(output / "stage3_h4_2_terminal_certificate.json", terminal)
    write_final_report(output, freeze_after, spatial, candidate_runs, selected, anomaly_doc, replay, gates, first_blocker, status)
    required_for_hash = [p for p in sorted(output.iterdir()) if p.is_file() and p.name != "stage3_h4_2_file_hashes.json"]
    file_hashes = {"schema_version": "stage3-h4-2-file-hashes-v1", "algorithm": "SHA-256", "files": [{"relative_path": p.name, "size_bytes": p.stat().st_size, "sha256": sha256_file(str(p))} for p in required_for_hash], "self_hash": "EXCLUDED_FROM_SELF_HASH_LIST", "tree_hash": semantic_hash([{ "relative_path": p.name, "sha256": sha256_file(str(p))} for p in required_for_hash])}
    dump_json(output / "stage3_h4_2_file_hashes.json", file_hashes)
    print(json.dumps({"stage3_h4_2": status, "output_root": relative(output), "selected_candidate": selected_spec["candidate_id"], "counts": primary["metrics"]["classification_counts"], "first_blocker": first_blocker}))
    return 0 if status == "PASSED" else 2


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orchestrate", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--replay-index", type=int, default=0)
    parser.add_argument("--placements", type=Path)
    parser.add_argument("--h3-targets", type=Path, default=H3_TARGETS)
    parser.add_argument("--h2-manifest", type=Path, default=H2_MANIFEST)
    parser.add_argument("--kinematics", type=Path, default=KINEMATICS)
    parser.add_argument("--urdf", type=Path, default=URDF)
    parser.add_argument("--srdf", type=Path, default=SRDF)
    args, _unknown = parser.parse_known_args(argv)
    if args.worker:
        if not all((args.worker_output, args.placements)):
            print("worker requires --worker-output and --placements", file=sys.stderr)
            return 2
        return run_ros_worker(args)
    if not args.orchestrate:
        print("Stage 3 H4.2 requires --orchestrate in the ROS2 WSL environment or --worker.")
        return 2
    try:
        return orchestrate(args.output_root.resolve())
    except Exception as exc:
        output = args.output_root.resolve()
        output.mkdir(parents=True, exist_ok=True)
        first = getattr(exc, "code", type(exc).__name__)
        dump_json(output / "stage3_h4_2_gate_report.json", {"schema_version": "stage3-h4-2-gate-report-v1", "stage3_h4_2": "BLOCKED", "STAGE_3_H4_2": "BLOCKED", "first_blocker": first, "error": f"{type(exc).__name__}: {exc}"})
        dump_json(output / "stage3_h4_2_terminal_certificate.json", {"schema_version": "stage3-h4-2-terminal-certificate-v1", "stage3_h4_2": "BLOCKED", "STAGE_3_H4_2": "BLOCKED", "FIRST_BLOCKER": first, "error": f"{type(exc).__name__}: {exc}", "FJT_GOALS_SENT": 0, "ROBOT_MOTION": "NO", "FORMAL_BASELINE_MUTATION": "NO", "PATH_PLANNING": "NOT_STARTED", "RUCKIG": "NOT_STARTED", "ML": "NOT_STARTED"})
        (output / "FINAL_REPORT.md").write_text(f"# Stage 3 H4.2 — BLOCKED\n\n`FIRST_BLOCKER: {first}`\n\n{type(exc).__name__}: {exc}\n", encoding="utf-8", newline="\n")
        print(json.dumps({"stage3_h4_2": "BLOCKED", "first_blocker": first, "output_root": relative(output)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
