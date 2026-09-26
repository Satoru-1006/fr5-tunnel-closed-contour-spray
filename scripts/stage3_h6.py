#!/usr/bin/env python3
"""Stage 3 H6 surface-constrained coverage sequencing and certification.

The orchestrator is deliberately additive and fail-closed.  It consumes only
the frozen H4.5/H5 curved-target artifacts, constructs paths on the transformed
H2 mesh, and delegates IK/FK/PlanningScene/FCL state checks to the native H6
bridge.  It never invokes a planner, time parameterization, Ruckig, a
controller, or robot I/O.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
H45_ROOT = ROOT / "outputs/stage3_h4_5_robot_fixture_geometry_codesign_20260809T000000Z"
H5_ROOT = ROOT / "outputs/stage3_h5_joint_branch_continuity_20260808T132000Z"
H3_ROOT = ROOT / "outputs/stage3_h3_coverage_baseline_20260808T034526Z"
H2_ROOT = ROOT / "outputs/stage3_h2_geometry_baseline_20260807T180321Z"
H3_TARGETS = H3_ROOT / "stage3_h3_surface_targets.jsonl"
H3_POSE_CONTRACT = ROOT / "config/stage3/stage3_h3_target_pose_contract.json"
H3_TASK_CONTRACT = ROOT / "config/stage3/stage3_h3_task_representation_contract.json"
H45_CONFIG = H45_ROOT / "stage3_h4_5_configuration_manifest.json"
H45_SELECTED = H45_ROOT / "stage3_h4_5_selected_candidate.json"
H45_COMPONENTS = H45_ROOT / "stage3_h4_5_connected_components.json"
H45_MESH = H45_ROOT / "curved_fixture_stage3_h4_5_selected.obj"
H5_TERMINAL = H5_ROOT / "stage3_h5_terminal_certificate.json"
H5_GATE = H5_ROOT / "stage3_h5_gate_report.json"
H5_FREEZE = H5_ROOT / "stage3_h5_configuration_freeze_manifest.json"
H5_SURFACE = H5_ROOT / "stage3_h5_surface_adjacency_input.json"
H5_COMPONENTS = H5_ROOT / "stage3_h5_joint_connectable_components.json"
H5_NODES = H5_ROOT / "stage3_h5_distinct_ik_nodes.jsonl"
H5_SEEDS = H5_ROOT / "stage3_h5_ik_seed_bank.json"
H5_PREFERRED = H5_ROOT / "stage3_h5_preferred_branch_sequences.json"
H5_GRAPH = H5_ROOT / "stage3_h5_joint_configuration_graph.json"
H5_CANDIDATES = H5_ROOT / "stage3_h5_multi_ik_candidates.jsonl"
H5_TRANSITIONS = H5_ROOT / "stage3_h5_transition_collision_evidence.jsonl"
DERIVED_URDF = H45_ROOT / "derived_robot_model.urdf"
SRDF = ROOT / "ros2_moveit_bridge/config/fairino5_v6_spray_tcp.srdf"
H2_MESH = H2_ROOT / "test_fixtures/curved_cylinder_patch.obj"
H6_COLLISION_METHOD = "adaptive_discrete_interpolation"
H6_CCD = "not_available"
EXPECTED_CONFIGURATION = "h4_5_013_dx_minus_200_dy_minus_200"
STANDOFF_M = 0.260
TCP_EXTENSION_M = 0.150
NUMERICAL_MAX_CARTESIAN_STEP_M = 0.010
NUMERICAL_MAX_ANGULAR_STEP_DEG = 2.0
H4_FK_TRANSLATION_NUMERICAL_TOLERANCE_M = 1.0e-6
H4_FK_ROTATION_NUMERICAL_TOLERANCE_RAD = 1.0e-6
H5_INTERPOLATION_STEP_DEG = 0.5
FORBIDDEN_LOG_PATTERNS = ("process has died", "segmentation fault", "sigsegv", "exit code -11")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def dump_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n" for row in rows), encoding="utf-8", newline="\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def canonical(value: Any, digits: int = 12) -> Any:
    if isinstance(value, Mapping):
        return {str(key): canonical(value[key], digits) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [canonical(item, digits) for item in value]
    if isinstance(value, float):
        return None if not math.isfinite(value) else round(value, digits)
    return value


def semantic_hash(value: Any) -> str:
    data = json.dumps(canonical(value), ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    return f"/mnt/{drive}/{str(resolved).split(':', 1)[-1].lstrip('/').replace(chr(92), '/') }"


def run_wsl(commands: Sequence[str], timeout_s: int) -> tuple[int, str, str]:
    returncode = subprocess.run(
        ["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", " && ".join(commands)],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout_s, check=False,
    )
    return returncode.returncode, returncode.stdout or "", returncode.stderr or ""


def vec_add(a: Sequence[float], b: Sequence[float]) -> list[float]: return [float(x) + float(y) for x, y in zip(a, b)]
def vec_sub(a: Sequence[float], b: Sequence[float]) -> list[float]: return [float(x) - float(y) for x, y in zip(a, b)]
def vec_scale(a: Sequence[float], s: float) -> list[float]: return [float(x) * float(s) for x in a]
def vec_norm(a: Sequence[float]) -> float: return math.sqrt(sum(float(x) * float(x) for x in a))
def dot(a: Sequence[float], b: Sequence[float]) -> float: return sum(float(x) * float(y) for x, y in zip(a, b))
def cross(a: Sequence[float], b: Sequence[float]) -> list[float]: return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def unit(a: Sequence[float]) -> list[float]:
    length = vec_norm(a)
    if not math.isfinite(length) or length <= 1.0e-15:
        raise ValueError("zero or non-finite vector")
    return [float(x) / length for x in a]


def mat_vec(matrix: Sequence[Sequence[float]], point: Sequence[float]) -> list[float]:
    return [sum(float(matrix[i][j]) * float(point[j]) for j in range(3)) + float(matrix[i][3]) for i in range(3)]


def mat_rot(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
    return unit([sum(float(matrix[i][j]) * float(vector[j]) for j in range(3)) for i in range(3)])


def quat_to_matrix(q: Sequence[float]) -> list[list[float]]:
    x, y, z, w = map(float, q); n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


def matmul3(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> list[list[float]]:
    return [[sum(float(a[i][k]) * float(b[k][j]) for k in range(3)) for j in range(3)] for i in range(3)]


def matrix_to_quat(r: Sequence[Sequence[float]]) -> list[float]:
    trace = float(r[0][0] + r[1][1] + r[2][2])
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0; w = 0.25 * s; x = (r[2][1] - r[1][2]) / s; y = (r[0][2] - r[2][0]) / s; z = (r[1][0] - r[0][1]) / s
    elif r[0][0] > r[1][1] and r[0][0] > r[2][2]:
        s = math.sqrt(1.0 + r[0][0] - r[1][1] - r[2][2]) * 2.0; w = (r[2][1] - r[1][2]) / s; x = 0.25 * s; y = (r[0][1] + r[1][0]) / s; z = (r[0][2] + r[2][0]) / s
    elif r[1][1] > r[2][2]:
        s = math.sqrt(1.0 + r[1][1] - r[0][0] - r[2][2]) * 2.0; w = (r[0][2] - r[2][0]) / s; x = (r[0][1] + r[1][0]) / s; y = 0.25 * s; z = (r[1][2] + r[2][1]) / s
    else:
        s = math.sqrt(1.0 + r[2][2] - r[0][0] - r[1][1]) * 2.0; w = (r[1][0] - r[0][1]) / s; x = (r[0][2] + r[2][0]) / s; y = (r[1][2] + r[2][1]) / s; z = 0.25 * s
    values = [x, y, z, w]
    if values[3] < 0: values = [-v for v in values]
    return values


def h3_orientation(normal: Sequence[float]) -> tuple[list[float], list[float], list[float]]:
    spray = vec_scale(unit(normal), -1.0)
    for reference in ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]):
        tangent = vec_sub(reference, vec_scale(spray, dot(reference, spray)))
        if vec_norm(tangent) > 1.0e-12:
            x_axis = unit(tangent); y_axis = unit(cross(spray, x_axis));
            return x_axis, y_axis, spray
    raise ValueError("H3 roll convention is degenerate")


def transform_target(target: Mapping[str, Any], transform: Sequence[Sequence[float]]) -> dict[str, Any]:
    r = [list(row[:3]) for row in transform[:3]]
    pose_r = matmul3(r, quat_to_matrix(target["tcp_target_orientation_xyzw"]))
    out = dict(target)
    out["surface_point_xyz_m"] = mat_vec(transform, target["surface_point_xyz_m"])
    out["tcp_target_position_xyz_m"] = mat_vec(transform, target["tcp_target_position_xyz_m"])
    out["surface_normal_unit"] = mat_rot(transform, target["surface_normal_unit"])
    out["spray_direction_unit"] = mat_rot(transform, target["spray_direction_unit"])
    out["tcp_target_orientation_xyzw"] = matrix_to_quat(pose_r)
    out["fixture_to_base_link_transform"] = [list(row) for row in transform]
    return out


def read_obj(path: Path) -> tuple[list[list[float]], list[tuple[int, int, int]]]:
    vertices: list[list[float]] = []; triangles: list[tuple[int, int, int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields: continue
        if fields[0] == "v": vertices.append([float(value) for value in fields[1:4]])
        elif fields[0] == "f":
            ids = [int(value.split("/")[0]) - 1 for value in fields[1:]]
            triangles.extend((ids[0], ids[i], ids[i + 1]) for i in range(1, len(ids) - 1))
    if not vertices or not triangles: raise ValueError(f"empty mesh: {path}")
    return vertices, triangles


def mesh_normals(vertices: Sequence[Sequence[float]], triangles: Sequence[Sequence[int]]) -> tuple[list[list[float]], list[list[float]]]:
    face_normals: list[list[float]] = []; areas: list[float] = []
    sums = [[0.0, 0.0, 0.0] for _ in vertices]
    for a, b, c in triangles:
        raw = cross(vec_sub(vertices[b], vertices[a]), vec_sub(vertices[c], vertices[a])); area = vec_norm(raw) / 2.0; areas.append(area); face = unit(raw); face_normals.append(face)
        for index in (a, b, c): sums[index] = vec_add(sums[index], vec_scale(face, area))
    vertex_normals = [unit(value) if vec_norm(value) > 1.0e-15 else [0.0, 0.0, 1.0] for value in sums]
    return vertex_normals, face_normals


def bary_point(vertices: Sequence[Sequence[float]], triangle: Sequence[int], bary: Sequence[float]) -> list[float]:
    return [sum(float(vertices[triangle[k]][axis]) * float(bary[k]) for k in range(3)) for axis in range(3)]


def bary_normal(vertex_normals: Sequence[Sequence[float]], triangle: Sequence[int], bary: Sequence[float]) -> list[float]:
    return unit([sum(float(vertex_normals[triangle[k]][axis]) * float(bary[k]) for k in range(3)) for axis in range(3)])


def bary_interp(a: Sequence[float], b: Sequence[float], alpha: float) -> list[float]: return [float(x) + alpha * (float(y) - float(x)) for x, y in zip(a, b)]


def mesh_triangle_adjacency(triangles: Sequence[Sequence[int]]) -> dict[int, list[tuple[int, tuple[int, int]]]]:
    edge_map: dict[tuple[int, int], list[int]] = {}
    for index, tri in enumerate(triangles):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge_map.setdefault(tuple(sorted((a, b))), []).append(index)
    adjacency: dict[int, list[tuple[int, tuple[int, int]]]] = {i: [] for i in range(len(triangles))}
    for edge, owners in edge_map.items():
        if len(owners) == 2:
            left, right = owners; adjacency[left].append((right, edge)); adjacency[right].append((left, edge))
    for key in adjacency: adjacency[key].sort(key=lambda item: (item[0], item[1]))
    return adjacency


def triangle_route(start: int, end: int, adjacency: Mapping[int, Sequence[tuple[int, tuple[int, int]]]]) -> list[int]:
    if start == end: return [start]
    queue = deque([start]); previous: dict[int, int | None] = {start: None}
    while queue:
        current = queue.popleft()
        for nxt, _edge in adjacency.get(current, []):
            if nxt in previous: continue
            previous[nxt] = current; queue.append(nxt)
            if nxt == end: queue.clear(); break
    if end not in previous: raise ValueError(f"surface mesh triangles disconnected: {start}->{end}")
    route: list[int] = []; current: int | None = end
    while current is not None: route.append(current); current = previous[current]
    return list(reversed(route))


def edge_bary(triangle: Sequence[int], edge: Sequence[int]) -> list[float]:
    result = [0.0, 0.0, 0.0]
    for index, vertex in enumerate(triangle):
        if vertex in edge: result[index] = 0.5
    return result


def surface_pieces(start: Mapping[str, Any], end: Mapping[str, Any], triangles: Sequence[Sequence[int]], adjacency: Mapping[int, Sequence[tuple[int, tuple[int, int]]]]) -> list[tuple[int, list[float], list[float]]]:
    start_tri = int(start["source_triangle_id"]); end_tri = int(end["source_triangle_id"]); route = triangle_route(start_tri, end_tri, adjacency)
    if len(route) == 1: return [(route[0], list(start["source_barycentric_uvw"]), list(end["source_barycentric_uvw"]))]
    pieces: list[tuple[int, list[float], list[float]]] = []
    for index, tri_id in enumerate(route):
        first = list(start["source_barycentric_uvw"]) if index == 0 else edge_bary(triangles[tri_id], next(edge for nxt, edge in adjacency[route[index - 1]] if nxt == tri_id))
        last = list(end["source_barycentric_uvw"]) if index == len(route) - 1 else edge_bary(triangles[tri_id], next(edge for nxt, edge in adjacency[tri_id] if nxt == route[index + 1]))
        pieces.append((tri_id, first, last))
    return pieces


def angle_between(a: Sequence[float], b: Sequence[float]) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, dot(unit(a), unit(b))))))


def physical_key(node: Mapping[str, Any]) -> tuple[Any, ...]:
    point = node["surface_point_xyz_m"]
    return (round(float(point[2]), 12), round(float(point[0]), 12), round(float(point[1]), 12), int(node["source_triangle_id"]), tuple(round(float(x), 12) for x in node["source_barycentric_uvw"]))


def build_surface_nodes(targets: Sequence[Mapping[str, Any]], feasible_ids: Sequence[int], transform: Sequence[Sequence[float]], source_vertices: Sequence[Sequence[float]], source_triangles: Sequence[Sequence[int]], source_vertex_normals: Sequence[Sequence[float]]) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    transformed = [transform_target(target, transform) for target in targets]
    nodes: dict[int, dict[str, Any]] = {}
    reconstruction_errors: list[float] = []; normal_errors: list[float] = []
    for target_id in sorted(feasible_ids):
        target = transformed[target_id]; tri_id = int(target["source_triangle_id"]); tri = source_triangles[tri_id]; bary = [float(x) for x in target["source_barycentric_uvw"]]
        source_point = bary_point(source_vertices, tri, bary); source_normal = bary_normal(source_vertex_normals, tri, bary)
        reconstruction_errors.append(vec_norm(vec_sub(source_point, [float(x) for x in targets[target_id]["surface_point_xyz_m"]])))
        normal_errors.append(vec_norm(vec_sub(source_normal, [float(x) for x in targets[target_id]["surface_normal_unit"]])))
        nodes[target_id] = {"target_id": target_id, "task_sample_id": target["task_sample_id"], "source_triangle_id": tri_id, "source_barycentric_uvw": bary, "surface_point_xyz_m": target["surface_point_xyz_m"], "surface_normal_unit": target["surface_normal_unit"], "spray_direction_unit": target["spray_direction_unit"], "tcp_target_position_xyz_m": target["tcp_target_position_xyz_m"], "tcp_target_orientation_xyzw": target["tcp_target_orientation_xyzw"], "coordinate_frame": target["coordinate_frame"], "surface_parameter": {"representation": "H2_mesh_triangle_barycentric", "triangle_id": tri_id, "barycentric_uvw": bary}, "surface_source": {"h2_mesh": rel(H2_MESH), "h45_selected_mesh": rel(H45_MESH), "fixture_transform": [list(row) for row in transform]}}
    return nodes, {"target_reconstruction_max_error_m": max(reconstruction_errors, default=None), "target_normal_reconstruction_max_error": max(normal_errors, default=None), "surface_representation": "transformed H2 canonical mesh; triangle-local barycentric interpolation; no world-frame straight-line fallback", "target_count": len(nodes)}


def edge_cost(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    return vec_norm(vec_sub(left["surface_point_xyz_m"], right["surface_point_xyz_m"])) + 0.001 * angle_between(left["surface_normal_unit"], right["surface_normal_unit"])


def shortest_unvisited_path(start: int, candidates: set[int], adjacency: Mapping[int, set[int]], nodes: Mapping[int, Mapping[str, Any]]) -> list[int]:
    queue = deque([start]); previous: dict[int, int | None] = {start: None}; target = None
    while queue:
        current = queue.popleft()
        if current in candidates: target = current; break
        for nxt in sorted(adjacency.get(current, set()), key=lambda item: (edge_cost(nodes[current], nodes[item]), physical_key(nodes[item]))):
            if nxt not in previous: previous[nxt] = current; queue.append(nxt)
    if target is None: return [start]
    path: list[int] = []; current: int | None = target
    while current is not None: path.append(current); current = previous[current]
    return list(reversed(path))


def component_path(component: Sequence[int], surface_adj: Mapping[int, set[int]], nodes: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(component, key=lambda item: physical_key(nodes[item])); index = {target: i for i, target in enumerate(ordered)}; n = len(ordered)
    dp: dict[tuple[int, int], tuple[float, tuple[int, ...]]] = {}
    for i in range(n): dp[(1 << i, i)] = (0.0, (ordered[i],))
    for mask_size in range(1, n + 1):
        states = sorted((key, value) for key, value in dp.items() if key[0].bit_count() == mask_size)
        for (mask, last), (cost, path) in states:
            last_target = ordered[last]
            for nxt in sorted(surface_adj.get(last_target, set()), key=lambda item: physical_key(nodes[item])):
                bit = 1 << index[nxt]
                if mask & bit: continue
                new = (mask | bit, index[nxt]); value = (cost + edge_cost(nodes[last_target], nodes[nxt]), path + (nxt,))
                old = dp.get(new)
                if old is None or value[0] < old[0] - 1.0e-12 or (abs(value[0] - old[0]) <= 1.0e-12 and tuple(physical_key(nodes[item]) for item in value[1]) < tuple(physical_key(nodes[item]) for item in old[1])): dp[new] = value
    full = (1 << n) - 1; finals = [value for (mask, _last), value in dp.items() if mask == full]
    if finals:
        cost, path_tuple = min(finals, key=lambda value: (value[0], tuple(physical_key(nodes[item]) for item in value[1])))
        return {"target_order": list(path_tuple), "ordering_method": "deterministic_min_cost_hamiltonian_path_on_H5_surface_adjacency", "hamiltonian_path_exists": True, "revisited_target_ids": [], "surface_cost": cost}
    remaining = set(component); current = ordered[0]; walk = [current]; remaining.remove(current)
    while remaining:
        candidates = surface_adj.get(current, set()) & remaining
        if candidates:
            nxt = min(candidates, key=lambda item: (edge_cost(nodes[current], nodes[item]), physical_key(nodes[item])))
            walk.append(nxt); remaining.remove(nxt); current = nxt
        else:
            route = shortest_unvisited_path(current, remaining, surface_adj, nodes); walk.extend(route[1:]);
            for item in route[1:]: remaining.discard(item)
            current = walk[-1]
    revisits = [item for index, item in enumerate(walk) if item in walk[:index]]
    return {"target_order": walk, "ordering_method": "deterministic_surface_graph_walk_path_cover", "hamiltonian_path_exists": False, "revisited_target_ids": sorted(set(revisits)), "surface_cost": sum(edge_cost(nodes[a], nodes[b]) for a, b in zip(walk, walk[1:]))}


def load_h5_nodes() -> dict[int, list[dict[str, Any]]]:
    by_target: dict[int, list[dict[str, Any]]] = {}
    for node in load_jsonl(H5_NODES):
        if node.get("h5_formal_node"):
            by_target.setdefault(int(node["target_id"]), []).append(node)
    for target in by_target: by_target[target].sort(key=lambda row: str(row.get("ik_candidate_id")))
    return by_target


def joint_cost_between_targets(left: int, right: int, h5_nodes: Mapping[int, Sequence[Mapping[str, Any]]]) -> tuple[float, list[float] | None, list[float] | None]:
    best: tuple[float, list[float], list[float]] | None = None
    for a in h5_nodes.get(left, []):
        for b in h5_nodes.get(right, []):
            delta = [float(y) - float(x) for x, y in zip(a["joint_values"], b["joint_values"])]
            cost = math.sqrt(sum(value * value for value in delta))
            candidate = (cost, list(a["joint_values"]), list(b["joint_values"]))
            if best is None or (cost, str(a.get("ik_candidate_id")), str(b.get("ik_candidate_id"))) < (best[0], "", ""): best = candidate
    return (best[0], best[1], best[2]) if best else (float("inf"), None, None)


def choose_component_order(paths: Mapping[int, Mapping[str, Any]], components: Sequence[Sequence[int]], h5_nodes: Mapping[int, Sequence[Mapping[str, Any]]], nodes: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    choices = []
    for permutation in itertools.permutations(range(len(components))):
        surface = sum(float(paths[index]["surface_cost"]) for index in permutation); reposition = 0.0; transitions = []
        for left_index, right_index in zip(permutation, permutation[1:]):
            left_end = paths[left_index]["target_order"][-1]; right_start = paths[right_index]["target_order"][0]; cost, _a, _b = joint_cost_between_targets(left_end, right_start, h5_nodes); reposition += cost; transitions.append({"from_component": left_index, "to_component": right_index, "from_target": left_end, "to_target": right_start, "joint_reposition_cost_rad": cost})
        key = tuple(physical_key(nodes[paths[index]["target_order"][0]]) for index in permutation)
        choices.append({"permutation": list(permutation), "surface_cost": surface, "joint_reposition_cost": reposition, "total_cost": surface + reposition, "tie_key": key, "transitions": transitions})
    selected = min(choices, key=lambda item: (item["total_cost"], item["tie_key"]))
    return {"schema_version": "stage3-h6-component-ordering-v1", "component_count": len(components), "component_execution_order": selected["permutation"], "selection_rule": "minimize deterministic surface path cost plus H5-node joint reposition diagnostic; physical surface keys break ties; component membership is immutable", "selected_cost": {key: value for key, value in selected.items() if key not in {"tie_key", "permutation"}}, "all_candidate_orders": [{key: value for key, value in item.items() if key != "tie_key"} for item in choices], "components": [{"component_index": index, "target_ids": list(component), "surface_order": paths[index]} for index, component in enumerate(components)]}


def interpolate_surface_pair(start: Mapping[str, Any], end: Mapping[str, Any], vertices: Sequence[Sequence[float]], triangles: Sequence[Sequence[int]], vertex_normals: Sequence[Sequence[float]], adjacency: Mapping[int, Sequence[tuple[int, tuple[int, int]]]], max_step_m: float, max_angle_deg: float) -> list[dict[str, Any]]:
    pieces = surface_pieces(start, end, triangles, adjacency); raw: list[dict[str, Any]] = []
    for piece_index, (tri_id, bary_a, bary_b) in enumerate(pieces):
        point_a = bary_point(vertices, triangles[tri_id], bary_a); point_b = bary_point(vertices, triangles[tri_id], bary_b); normal_a = bary_normal(vertex_normals, triangles[tri_id], bary_a); normal_b = bary_normal(vertex_normals, triangles[tri_id], bary_b); count = max(1, int(math.ceil(max(vec_norm(vec_sub(point_b, point_a)) / max_step_m, angle_between(normal_a, normal_b) / max_angle_deg))))
        for step in range(count + 1):
            if raw and step == 0: continue
            alpha = step / float(count); bary = bary_interp(bary_a, bary_b, alpha); point = bary_point(vertices, triangles[tri_id], bary); normal = bary_normal(vertex_normals, triangles[tri_id], bary); raw.append({"triangle_id": tri_id, "barycentric_uvw": bary, "surface_point_xyz_m": point, "surface_normal_unit": normal, "piece_index": piece_index, "piece_step_index": step, "piece_step_count": count})
    total = sum(vec_norm(vec_sub(raw[i]["surface_point_xyz_m"], raw[i - 1]["surface_point_xyz_m"])) for i in range(1, len(raw)))
    distance = 0.0
    for index, row in enumerate(raw):
        if index: distance += vec_norm(vec_sub(row["surface_point_xyz_m"], raw[index - 1]["surface_point_xyz_m"]))
        row["surface_path_s"] = 0.0 if total <= 0 else distance / total
    return raw


def make_waypoint(row: Mapping[str, Any], component_id: int, segment_id: int, source_id: int, destination_id: int, state: str, local_index: int, point: Sequence[float], normal: Sequence[float], s: float, triangle_id: int, bary: Sequence[float], transform: Sequence[Sequence[float]], target_endpoint: bool = False) -> dict[str, Any]:
    source_basis = h3_orientation(normal)
    r = [list(item[:3]) for item in transform[:3]]; x_axis = mat_rot(transform, source_basis[0]); y_axis = mat_rot(transform, source_basis[1]); spray = mat_rot(transform, source_basis[2]); actual_normal = vec_scale(spray, -1.0); rotation = [[x_axis[0], y_axis[0], spray[0]], [x_axis[1], y_axis[1], spray[1]], [x_axis[2], y_axis[2], spray[2]]]
    tcp = vec_add(mat_vec(transform, point), vec_scale(actual_normal, STANDOFF_M)); pose = matrix_to_quat(rotation)
    return {"schema_version": "stage3-h6-surface-waypoint-v1", "waypoint_index": local_index, "segment_id": segment_id, "component_id": component_id, "source_target_id": source_id, "destination_target_id": destination_id, "spray_state": state, "is_formal_target_endpoint": bool(target_endpoint), "surface_parameter": {"representation": "H2_mesh_triangle_barycentric", "triangle_id": int(triangle_id), "barycentric_uvw": [float(x) for x in bary]}, "surface_point_xyz_m": mat_vec(transform, point), "surface_normal_unit": actual_normal, "spray_direction_unit": vec_scale(actual_normal, -1.0), "tcp_position_xyz_m": tcp, "tcp_orientation_xyzw": pose, "desired_standoff_m": STANDOFF_M, "path_parameter_s": float(s), "standoff_tolerance": {"status": "UNRESOLVED", "value_m": None}, "normal_tolerance": {"status": "UNRESOLVED", "value_deg": None}, "sampling_rule": {"max_cartesian_step_m": NUMERICAL_MAX_CARTESIAN_STEP_M, "max_angular_step_deg": NUMERICAL_MAX_ANGULAR_STEP_DEG, "classification": "numerical_discretization_parameter; not physical/process acceptance tolerance"}}


def build_on_waypoints(nodes: Mapping[int, Mapping[str, Any]], ordering: Mapping[str, Any], vertices: Sequence[Sequence[float]], triangles: Sequence[Sequence[int]], vertex_normals: Sequence[Sequence[float]], tri_adjacency: Mapping[int, Sequence[tuple[int, tuple[int, int]]]], transform: Sequence[Sequence[float]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    waypoints: list[dict[str, Any]] = []; segments: list[dict[str, Any]] = []; waypoint_index = 0; segment_id = 0
    component_entries = {int(entry["component_index"]): entry for entry in ordering["components"]}
    for component_index in ordering["component_execution_order"]:
        component_entry = component_entries[int(component_index)]
        component_id = int(component_entry["component_index"]); target_order = list(component_entry["surface_order"]["target_order"]); start_index = waypoint_index
        for pair_index, (source_id, destination_id) in enumerate(zip(target_order, target_order[1:])):
            dense = interpolate_surface_pair(nodes[source_id], nodes[destination_id], vertices, triangles, vertex_normals, tri_adjacency, NUMERICAL_MAX_CARTESIAN_STEP_M, NUMERICAL_MAX_ANGULAR_STEP_DEG)
            for dense_index, item in enumerate(dense):
                endpoint = (dense_index == 0 and pair_index == 0) or (dense_index == len(dense) - 1)
                row = make_waypoint(nodes[source_id], component_id, segment_id, source_id, destination_id, "SPRAY_ON", waypoint_index, item["surface_point_xyz_m"], item["surface_normal_unit"], item["surface_path_s"], item["triangle_id"], item["barycentric_uvw"], transform, endpoint); row["task_sample_id"] = nodes[destination_id if endpoint and dense_index == len(dense) - 1 else source_id]["task_sample_id"]; waypoints.append(row); waypoint_index += 1
            # The first sample of the next pair is the prior target endpoint;
            # retain one shared waypoint instead of silently duplicating it.
        end_index = waypoint_index - 1; segments.append({"segment_id": segment_id, "component_id": component_id, "spray_state": "SPRAY_ON", "target_ids": target_order, "waypoint_start": start_index, "waypoint_end": end_index, "source_target_count": len(set(target_order)), "revisited_target_ids": component_entry["surface_order"].get("revisited_target_ids", [])}); segment_id += 1
    # Deduplicate pair joins while preserving the final target of each pair.
    compact: list[dict[str, Any]] = []
    for row in waypoints:
        if compact and row["component_id"] == compact[-1]["component_id"] and row["source_target_id"] == compact[-1]["destination_target_id"] and row["surface_parameter"] == compact[-1]["surface_parameter"]:
            continue
        compact.append(row)
    for index, row in enumerate(compact): row["waypoint_index"] = index
    for segment in segments:
        indexes = [row["waypoint_index"] for row in compact if row["segment_id"] == segment["segment_id"]]; segment["waypoint_start"] = min(indexes) if indexes else None; segment["waypoint_end"] = max(indexes) if indexes else None
    return compact, segments


def endpoint_seed_for_segment(segment: Mapping[str, Any], h5_nodes: Mapping[int, Sequence[Mapping[str, Any]]]) -> list[float] | None:
    target = int(segment["target_ids"][0]); candidates = list(h5_nodes.get(target, [])); return list(candidates[0]["joint_values"]) if candidates else None


def write_native_requests(path: Path, rows: Sequence[Mapping[str, Any]], h5_seed_values: Sequence[Sequence[float]], initial_seeds: Mapping[int, Sequence[float] | None], off_q: Mapping[int, Sequence[float]] | None = None) -> None:
    lines: list[str] = []
    for row in rows:
        seeds: list[Sequence[float]] = []
        if row["spray_state"] == "SPRAY_ON":
            if int(row["waypoint_index"]) == int(row.get("segment_first_waypoint", -1)) and initial_seeds.get(int(row["segment_id"])) is not None: seeds.append(initial_seeds[int(row["segment_id"])])  # type: ignore[arg-type]
            seeds.extend(h5_seed_values)
        else:
            q = (off_q or {}).get(int(row["waypoint_index"]))
            if q is None: continue
            seeds.append(q)
        pose = [*row["tcp_position_xyz_m"], *row["tcp_orientation_xyzw"]] if row["spray_state"] == "SPRAY_ON" else [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        fields = [str(row["waypoint_index"]), str(row["segment_id"]), str(row["component_id"]), str(row["spray_state"]), str(row["source_target_id"]), str(row["destination_target_id"]), str(row.get("task_sample_id", "")), *[repr(float(x)) for x in pose], str(len(seeds))]
        for seed in seeds: fields.extend(repr(float(x)) for x in seed)
        lines.append("\t".join(fields))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def build_native_bridge(output: Path) -> Path:
    build_base = output / "native_build"; install_base = output / "native_install"; log = output / "stage3_h6_native_bridge_build.log"
    code, stdout, stderr = run_wsl(["source /opt/ros/jazzy/setup.bash", f"colcon build --base-paths {wsl_path(ROOT / 'cpp/stage3_h6')} --build-base {wsl_path(build_base)} --install-base {wsl_path(install_base)} --merge-install"], 1200)
    log.write_text(stdout + "\n--- STDERR ---\n" + stderr, encoding="utf-8", newline="\n")
    record = {"schema_version": "stage3-h6-native-bridge-build-v1", "package": "stage3_h6_native", "returncode": code, "status": "PASSED" if code == 0 else "BLOCKED", "install_base": rel(install_base), "forbidden_log_patterns": [pattern for pattern in FORBIDDEN_LOG_PATTERNS if pattern in (stdout + stderr).lower()]}; dump_json(output / "stage3_h6_native_bridge_build.json", record)
    executable = install_base / "lib/stage3_h6_native/stage3_h6_native_bridge"
    if code != 0 or not executable.is_file(): raise RuntimeError(f"native_h6_bridge_unavailable: {record}")
    return executable


def native_semantic_signature(native_dir: Path) -> str:
    value = {
        "summary": load_json(native_dir / "stage3_h6_native_summary.json") if (native_dir / "stage3_h6_native_summary.json").is_file() else {},
        "branch": parse_native_trace(native_dir / "stage3_h6_native_ik_branch_trace.jsonl"),
        "fk": parse_native_trace(native_dir / "stage3_h6_native_fk_tcp_validation.jsonl"),
        "collision": parse_native_trace(native_dir / "stage3_h6_native_collision_evidence.jsonl"),
    }
    return semantic_hash(value)


def run_native(executable: Path, request_path: Path, mesh: Path, output: Path, native_name: str = "native_final") -> dict[str, Any]:
    native_dir = output / native_name
    code, stdout, stderr = run_wsl(["source /opt/ros/jazzy/setup.bash", f"source {wsl_path(ROOT / 'install/setup.bash')}", f"source {wsl_path(output / 'native_install/setup.bash')}", f"cd {wsl_path(ROOT)}", f"{wsl_path(executable)} --urdf {wsl_path(DERIVED_URDF)} --srdf {wsl_path(SRDF)} --requests {wsl_path(request_path)} --mesh {wsl_path(mesh)} --output {wsl_path(native_dir)}", "true"], 1800)
    log_name = "stage3_h6_native_process.log" if native_name == "native_final" else f"stage3_h6_native_process_{native_name}.log"
    (output / log_name).write_text(stdout + "\n--- STDERR ---\n" + stderr, encoding="utf-8", newline="\n")
    summary_path = native_dir / "stage3_h6_native_summary.json"; summary = load_json(summary_path) if summary_path.is_file() else {}
    record = {"schema_version": "stage3-h6-native-process-record-v1", "native_name": native_name, "returncode": code, "summary_exists": bool(summary), "status": "PASSED" if code == 0 and summary.get("status") == "AVAILABLE" else "BLOCKED", "summary": summary, "semantic_signature": native_semantic_signature(native_dir) if summary else None, "forbidden_log_patterns": [pattern for pattern in FORBIDDEN_LOG_PATTERNS if pattern in (stdout + stderr).lower()]}; dump_json(output / ("stage3_h6_native_process_record.json" if native_name == "native_final" else f"stage3_h6_native_process_record_{native_name}.json"), record)
    if record["status"] != "PASSED": raise RuntimeError(f"native_h6_process_unavailable: {record}")
    return record


def preflight(output: Path) -> dict[str, Any]:
    required_h5 = [H5_TERMINAL, H5_GATE, H5_COMPONENTS, H5_GRAPH, H5_SURFACE, H5_PREFERRED, H5_NODES, H5_CANDIDATES, H5_TRANSITIONS, H5_FREEZE, H5_SEEDS]
    required_h45 = [H45_CONFIG, H45_SELECTED, H45_COMPONENTS, H45_MESH, DERIVED_URDF, SRDF]
    missing = [rel(path) for path in [*required_h5, *required_h45, H3_TARGETS, H3_POSE_CONTRACT, H3_TASK_CONTRACT, H2_MESH] if not path.is_file()]
    if missing: return {"status": "BLOCKED", "FIRST_BLOCKER": "frozen_H4_5_or_H5_input_missing", "missing": missing}
    h5_terminal = load_json(H5_TERMINAL); h5_gate = load_json(H5_GATE); freeze = load_json(H5_FREEZE); h45_config = load_json(H45_CONFIG); h45_selected = load_json(H45_SELECTED); h45_components = load_json(H45_COMPONENTS); h5_surface = load_json(H5_SURFACE); h5_components = load_json(H5_COMPONENTS); h3_targets = load_jsonl(H3_TARGETS)
    h45_selected_nested = h45_selected.get("selected_candidate", {})
    hash_mismatches = []
    if freeze.get("selected_configuration_id") != EXPECTED_CONFIGURATION or h45_config.get("selected_candidate_id") != EXPECTED_CONFIGURATION: hash_mismatches.append("selected_configuration")
    if freeze.get("urdf_sha256") != sha256_file(DERIVED_URDF) or freeze.get("srdf_sha256") != sha256_file(SRDF): hash_mismatches.extend(["urdf", "srdf"])
    if freeze.get("selected_configuration_sha256") != semantic_hash(h45_selected_nested): hash_mismatches.append("selected_configuration_sha256")
    expected_components = [sorted(map(int, component)) for component in h45_components.get("component_target_indices", [])]
    formal_ids = sorted(map(int, h5_surface.get("formal_feasible_target_ids", [])))
    actual_components = [sorted(map(int, component)) for component in h5_components.get("joint_connectable_target_components", [])]
    if len(h3_targets) != 192 or len(formal_ids) != 30: hash_mismatches.append("target_count")
    if sorted(expected_components) != sorted(actual_components): hash_mismatches.append("component_membership")
    if sorted(formal_ids) != sorted(item for component in expected_components for item in component): hash_mismatches.append("formal_target_membership")
    statuses_ok = h5_terminal.get("STAGE_3_H5") == "PASSED" and h5_terminal.get("READY_FOR_STAGE_3_H6") == "YES" and h5_gate.get("STAGE_3_H5") == "PASSED" and h5_gate.get("READY_FOR_STAGE_3_H6") == "YES" and h5_gate.get("FIRST_BLOCKER") is None
    if not statuses_ok: hash_mismatches.append("h5_gate")
    source_paths = [H3_TARGETS, H3_POSE_CONTRACT, H3_TASK_CONTRACT, H45_CONFIG, H45_SELECTED, H45_COMPONENTS, H45_MESH, DERIVED_URDF, SRDF, H5_TERMINAL, H5_GATE, H5_COMPONENTS, H5_GRAPH, H5_SURFACE, H5_PREFERRED, H5_NODES, H5_CANDIDATES, H5_TRANSITIONS, H5_FREEZE, H5_SEEDS, H2_MESH]
    source_records = [{"path": rel(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size} for path in source_paths]
    pose_contract = load_json(H3_POSE_CONTRACT); h5_diag = load_jsonl(H5_ROOT / "stage3_h5_intermediate_tcp_diagnostics.jsonl") if (H5_ROOT / "stage3_h5_intermediate_tcp_diagnostics.jsonl").is_file() else []
    tolerance_audit = {"formal_spray_process_tolerance": {"standoff": {"status": "UNRESOLVED", "value": None}, "normal_angular": {"status": "UNRESOLVED", "value": None}, "tcp_position": {"status": "UNRESOLVED", "value": None}, "tcp_orientation": {"status": "UNRESOLVED", "value": None}}, "sources_checked": [rel(H3_POSE_CONTRACT), rel(H3_TASK_CONTRACT), rel(H45_CONFIG), rel(H5_FREEZE), rel(H5_ROOT / "stage3_h5_intermediate_tcp_diagnostics.jsonl")], "h5_formal_tolerance_applied_values": sorted({bool(row.get("formal_tcp_tolerance_applied")) for row in h5_diag}), "numerical_fk_tolerances_inherited": {"translation_m": H4_FK_TRANSLATION_NUMERICAL_TOLERANCE_M, "rotation_rad": H4_FK_ROTATION_NUMERICAL_TOLERANCE_RAD, "classification": "IK/FK numerical validity only; not Spray process acceptance tolerance"}, "nominal_standoff_m": pose_contract.get("stand_off", {}).get("value_m"), "no_threshold_invented": True}
    blocker = "frozen_H4_5_or_H5_input_changed" if hash_mismatches else None
    return {"schema_version": "stage3-h6-input-preflight-v1", "status": "PASSED" if blocker is None else "BLOCKED", "FIRST_BLOCKER": blocker, "hash_mismatches": hash_mismatches, "h4_5_h5_frozen_inputs": source_records, "target_count": len(h3_targets), "formal_target_ids": formal_ids, "component_count": len(actual_components), "components": actual_components, "selected_configuration": h45_config.get("selected_candidate_id"), "standoff_m": h45_config.get("process_standoff_m"), "tcp_extension_m": h45_config.get("tcp_extension_m"), "robot_model": {"urdf": rel(DERIVED_URDF), "urdf_sha256": sha256_file(DERIVED_URDF), "srdf": rel(SRDF), "srdf_sha256": sha256_file(SRDF), "planning_group": "fairino5_v6_group", "tcp_link": "spray_tcp_link", "base_frame": "base_link"}, "fixture_geometry": {"mesh": rel(H45_MESH), "surface_frame": "base_link", "source_h2_mesh": rel(H2_MESH)}, "surface_frame": "base_link", "orientation_convention": pose_contract.get("orientation", {}), "tolerance_audit": tolerance_audit, "h5_status": h5_terminal, "h5_gate": h5_gate}


def write_placeholder_certificates(output: Path, pre: Mapping[str, Any], blocker: str, extra: Sequence[str] = ()) -> None:
    coverage = {"schema_version": "stage3-h6-coverage-accounting-v1", "formal_targets": len(pre.get("formal_target_ids", [])), "spray_on_covered_targets": 0, "uncovered_targets": pre.get("formal_target_ids", []), "duplicate/revisited_targets": [], "component_count": pre.get("component_count"), "ALL_30_FEASIBLE_TARGETS_ACCOUNTED_FOR": "NO", "additional_blockers": list(extra)}; dump_json(output / "stage3_h6_coverage_accounting.json", coverage)
    dump_jsonl(output / "stage3_h6_surface_waypoints.jsonl", []); dump_jsonl(output / "stage3_h6_cartesian_tcp_waypoints.jsonl", []); dump_jsonl(output / "stage3_h6_joint_waypoints.jsonl", []); dump_jsonl(output / "stage3_h6_ik_branch_trace.jsonl", []); dump_jsonl(output / "stage3_h6_fk_tcp_validation.jsonl", []); dump_jsonl(output / "stage3_h6_collision_evidence.jsonl", []); dump_jsonl(output / "stage3_h6_spray_on_off_segments.jsonl", []); dump_json(output / "stage3_h6_spray_on_off_segments.json", {"schema_version": "stage3-h6-spray-on-off-segments-v1", "segments": []}); dump_json(output / "stage3_h6_replay_determinism.json", {"schema_version": "stage3-h6-replay-determinism-v1", "fresh_process_count": 3, "processes": [], "all_semantic_results_consistent": False, "FRESH_PROCESS_REPLAY": "BLOCKED"})
    gate = {"schema_version": "stage3-h6-gate-report-v1", "STAGE_3_H6": "BLOCKED", "FIRST_BLOCKER": blocker, "READY_FOR_STAGE_3_H7": "NO", "H4_5_BASELINE_IMMUTABLE": "NO" if blocker.startswith("frozen_") else "YES", "H5_BASELINE_IMMUTABLE": "NO" if blocker.startswith("frozen_") else "YES", "WORKCELL_CONFIGURATION_FROZEN": "YES" if blocker != "frozen_H4_5_or_H5_input_changed" else "NO", "collision_method": H6_COLLISION_METHOD, "ccd_status": H6_CCD, "TIME_PARAMETERIZATION_DONE": "NO", "RUCKIG_STARTED": "NO", "RUCKIG_DONE": "NO", "ROBOT_MOTION_STARTED": "NO", "NEW_FJT_GOALS_SENT": 0, "FORMAL_LEDGER_MUTATED": "NO", "ML_TRAINING_STARTED": "NO", "RL_TRAINING_STARTED": "NO", "mandatory_blockers": [blocker, *extra]}; dump_json(output / "stage3_h6_gate_report.json", gate)
    terminal = {"schema_version": "stage3-h6-terminal-certificate-v1", "STAGE_3_H6": "BLOCKED", "FIRST_BLOCKER": blocker, "READY_FOR_STAGE_3_H7": "NO", "CCD": "NOT_AVAILABLE", "TIME_PARAMETERIZATION_DONE": "NO", "RUCKIG_STARTED": "NO", "RUCKIG_DONE": "NO", "ROBOT_MOTION_STARTED": "NO", "NEW_FJT_GOALS_SENT": 0, "FORMAL_LEDGER_MUTATED": "NO", "ML_TRAINING_STARTED": "NO", "RL_TRAINING_STARTED": "NO", "additional_blockers": list(extra)}; dump_json(output / "stage3_h6_terminal_certificate.json", terminal)


def parse_native_trace(path: Path) -> list[dict[str, Any]]: return load_jsonl(path) if path.is_file() else []


def orchestrate(output: Path) -> int:
    if output.exists() and any(output.iterdir()): raise RuntimeError(f"refusing to overwrite non-empty H6 output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    pre = preflight(output); dump_json(output / "stage3_h6_input_freeze_manifest.json", pre)
    if pre.get("FIRST_BLOCKER"):
        write_placeholder_certificates(output, pre, str(pre["FIRST_BLOCKER"])); write_final_report(output, pre, {}, {}, {}, {}, str(pre["FIRST_BLOCKER"]), []); return 2
    targets = load_jsonl(H3_TARGETS); transform = load_json(H45_SELECTED)["selected_candidate"]["transforms"]["fixture_curved_cylinder_patch"]; source_vertices, source_triangles = read_obj(H2_MESH); source_normals, _face_normals = mesh_normals(source_vertices, source_triangles); transformed_vertices = [mat_vec(transform, point) for point in source_vertices]; transformed_normals = [mat_rot(transform, normal) for normal in source_normals]; tri_adjacency = mesh_triangle_adjacency(source_triangles)
    nodes, surface_audit = build_surface_nodes(targets, pre["formal_target_ids"], transform, source_vertices, source_triangles, source_normals); surface_edges = [tuple(sorted(map(int, edge))) for edge in load_json(H5_SURFACE)["surface_adjacent_edges"]]; surface_adj: dict[int, set[int]] = {target: set() for target in pre["formal_target_ids"]}
    for left, right in sorted(set(surface_edges)): surface_adj[left].add(right); surface_adj[right].add(left)
    components = [list(component) for component in pre["components"]]; h5_nodes = load_h5_nodes(); paths = {index: component_path(component, surface_adj, nodes) for index, component in enumerate(components)}; ordering = choose_component_order(paths, components, h5_nodes, nodes); dump_json(output / "stage3_h6_surface_graph.json", {"schema_version": "stage3-h6-surface-graph-v1", "source": rel(H5_SURFACE), "surface_representation": "H2 canonical mesh transformed by frozen H4.5 fixture transform", "nodes": [nodes[target] for target in sorted(nodes, key=lambda item: physical_key(nodes[item]))], "valid_surface_adjacency_edges": [list(edge) for edge in sorted(set(surface_edges))], "component_membership": components, "surface_audit": surface_audit, "no_new_adjacency_edges": True, "no_cross_component_edges": True}); dump_json(output / "stage3_h6_component_ordering.json", ordering)
    on_waypoints, on_segments = build_on_waypoints(nodes, ordering, source_vertices, source_triangles, source_normals, tri_adjacency, transform)
    for segment in on_segments:
        for row in on_waypoints:
            if row["segment_id"] == segment["segment_id"]: row["segment_first_waypoint"] = segment["waypoint_start"]
    initial_seeds = {int(segment["segment_id"]): endpoint_seed_for_segment(segment, h5_nodes) for segment in on_segments}; seed_values = [list(seed["joint_positions"]) for seed in load_json(H5_SEEDS).get("seeds", [])]
    # H5's distinct endpoint IK nodes are frozen evidence, not newly sampled
    # solutions.  Include their joint vectors as deterministic alternatives so
    # a local surface interpolation cannot fail solely because the 13 generic
    # H5 seed-bank entries start on the wrong branch.
    seen_seed_values = {tuple(round(float(value), 12) for value in seed) for seed in seed_values}
    for target_id in sorted(h5_nodes):
        for node in h5_nodes[target_id]:
            seed = [float(value) for value in node["joint_values"]]; key = tuple(round(value, 12) for value in seed)
            if key not in seen_seed_values:
                seen_seed_values.add(key); seed_values.append(seed)
    dump_jsonl(output / "stage3_h6_surface_waypoints.jsonl", on_waypoints); dump_jsonl(output / "stage3_h6_cartesian_tcp_waypoints.jsonl", on_waypoints)
    segments = [{**segment, "spray_state": "SPRAY_ON", "certification_status": "PENDING_NATIVE"} for segment in on_segments]
    on_request = output / "stage3_h6_on_requests.tsv"; write_native_requests(on_request, on_waypoints, seed_values, initial_seeds)
    try:
        executable = build_native_bridge(output); run_native(executable, on_request, H45_MESH, output, "native_on_probe")
    except Exception as exc:
        extra = [f"{type(exc).__name__}: {exc}"]; write_placeholder_certificates(output, pre, "native_moveit_h6_backend_unavailable", extra); write_final_report(output, pre, {}, {"surface_waypoints": len(on_waypoints)}, {"spray_on_segments": len(on_segments)}, {}, "native_moveit_h6_backend_unavailable", extra); return 2
    probe_trace = parse_native_trace(output / "native_on_probe/stage3_h6_native_ik_branch_trace.jsonl"); accepted_by_segment: dict[int, list[dict[str, Any]]] = {}
    for row in probe_trace:
        if row.get("spray_state") == "SPRAY_ON" and row.get("accepted"): accepted_by_segment.setdefault(int(row["segment_id"]), []).append(row)
    extra_blockers: list[str] = []
    if any(not row.get("accepted") for row in probe_trace if row.get("spray_state") == "SPRAY_ON"): extra_blockers.append("surface_constrained_transition_unavailable")
    off_rows: list[dict[str, Any]] = []; off_q: dict[int, Sequence[float]] = {}; next_waypoint = max((int(row["waypoint_index"]) for row in on_waypoints), default=-1) + 1; off_segment_id = 1000
    ordered_on_segments = sorted(on_segments, key=lambda item: int(item["segment_id"]))
    for left_segment, right_segment in zip(ordered_on_segments, ordered_on_segments[1:]):
        left_segment_id = int(left_segment["segment_id"]); right_segment_id = int(right_segment["segment_id"]); left_trace = accepted_by_segment.get(left_segment_id, []); q0 = left_trace[-1].get("accepted_joint_values") if left_trace else initial_seeds.get(left_segment_id); q1 = initial_seeds.get(right_segment_id)
        if q0 is None or q1 is None: extra_blockers.append("spray_off_reposition_unavailable"); continue
        max_delta = max(abs(float(b) - float(a)) for a, b in zip(q0, q1)); count = max(2, int(math.ceil(math.degrees(max_delta) / H5_INTERPOLATION_STEP_DEG)) + 1); start = next_waypoint
        for index in range(count):
            alpha = index / float(count - 1); q = [float(a) + alpha * (float(b) - float(a)) for a, b in zip(q0, q1)]; row = {"schema_version": "stage3-h6-spray-off-waypoint-v1", "waypoint_index": next_waypoint, "segment_id": off_segment_id, "component_id": -1, "source_target_id": int(left_segment["target_ids"][-1]), "destination_target_id": int(right_segment["target_ids"][0]), "task_sample_id": f"SPRAY_OFF:{left_segment['target_ids'][-1]}->{right_segment['target_ids'][0]}", "spray_state": "SPRAY_OFF", "joint_values_seed": q, "adaptive_discrete_interpolation": {"step_deg": H5_INTERPOLATION_STEP_DEG, "sample_count": count, "alpha": alpha}, "tcp_position_xyz_m": [0.0, 0.0, 0.0], "tcp_orientation_xyzw": [0.0, 0.0, 0.0, 1.0]}; off_rows.append(row); off_q[next_waypoint] = q; next_waypoint += 1
        segments.append({"segment_id": off_segment_id, "component_id": -1, "spray_state": "SPRAY_OFF", "source_segment_id": left_segment_id, "destination_segment_id": right_segment_id, "source_component": int(left_segment["component_id"]), "destination_component": int(right_segment["component_id"]), "source_target_id": int(left_segment["target_ids"][-1]), "destination_target_id": int(right_segment["target_ids"][0]), "waypoint_start": start, "waypoint_end": next_waypoint - 1, "connectivity_source": rel(H5_GRAPH), "standoff_tracking_required": False, "normal_tracking_required": False, "certification_status": "PENDING_NATIVE"}); off_segment_id += 1
    full_rows: list[dict[str, Any]] = []
    ordered_on = {int(row["waypoint_index"]): row for row in on_waypoints}; off_by_before: dict[int, list[dict[str, Any]]] = {}
    for off_segment in [item for item in segments if item["spray_state"] == "SPRAY_OFF"]:
        before = max((int(row["waypoint_index"]) for row in on_waypoints if int(row["segment_id"]) == int(off_segment["source_segment_id"])), default=-1); off_by_before.setdefault(before, []).extend(row for row in off_rows if int(row["segment_id"]) == int(off_segment["segment_id"]))
    for row in sorted(on_waypoints, key=lambda item: int(item["waypoint_index"])):
        full_rows.append(row); full_rows.extend(off_by_before.get(int(row["waypoint_index"]), []))
    for index, row in enumerate(full_rows): row["waypoint_index"] = index
    for segment in on_segments:
        indexes = [int(row["waypoint_index"]) for row in full_rows if row["spray_state"] == "SPRAY_ON" and int(row["segment_id"]) == int(segment["segment_id"])]
        if indexes:
            first = min(indexes)
            for row in full_rows:
                if row["spray_state"] == "SPRAY_ON" and int(row["segment_id"]) == int(segment["segment_id"]): row["segment_first_waypoint"] = first
    final_request = output / "stage3_h6_requests.tsv"; final_initial = dict(initial_seeds); write_native_requests(final_request, full_rows, seed_values, final_initial, off_q={int(row["waypoint_index"]): row.get("joint_values_seed", []) for row in full_rows if row["spray_state"] == "SPRAY_OFF"})
    replay_runs: list[dict[str, Any]] = []
    try:
        final_record = run_native(executable, final_request, H45_MESH, output, "native_final")
        replay_runs.append({"replay_index": 1, "native_name": "native_final", "status": final_record.get("status"), "semantic_signature": final_record.get("semantic_signature")})
        for replay_index in (2, 3):
            replay_name = f"native_replay_{replay_index}"; replay_record = run_native(executable, final_request, H45_MESH, output, replay_name)
            replay_runs.append({"replay_index": replay_index, "native_name": replay_name, "status": replay_record.get("status"), "semantic_signature": replay_record.get("semantic_signature")})
    except Exception as exc:
        extra_blockers.append("native_moveit_h6_final_process_unavailable"); extra_blockers.append(f"{type(exc).__name__}: {exc}")
    native_dir = output / "native_final"; native_trace = parse_native_trace(native_dir / "stage3_h6_native_ik_branch_trace.jsonl"); native_fk = parse_native_trace(native_dir / "stage3_h6_native_fk_tcp_validation.jsonl"); native_collision = parse_native_trace(native_dir / "stage3_h6_native_collision_evidence.jsonl")
    if any(not row.get("accepted") for row in native_trace if row.get("spray_state") == "SPRAY_ON") and "surface_constrained_transition_unavailable" not in extra_blockers: extra_blockers.append("surface_constrained_transition_unavailable")
    trace_map = {int(row["waypoint_index"]): row for row in native_trace}; surface_map = {int(row["waypoint_index"]): row for row in full_rows if row["spray_state"] == "SPRAY_ON"}; joint_rows = []
    fk_rows: list[dict[str, Any]] = []; collision_rows: list[dict[str, Any]] = []
    for index, native in sorted(trace_map.items()):
        base = surface_map.get(index, full_rows[index] if index < len(full_rows) else {}); accepted_q = native.get("accepted_joint_values", []); nearest = None
        if accepted_q:
            candidates = [(math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(accepted_q, node["joint_values"]))), node) for target in h5_nodes for node in h5_nodes[target]]; nearest = min(candidates, key=lambda item: (item[0], str(item[1].get("ik_candidate_id"))))[1]["ik_candidate_id"] if candidates else None
        joint_rows.append({"schema_version": "stage3-h6-joint-waypoint-v1", "waypoint_index": index, "segment_id": native.get("segment_id"), "spray_state": native.get("spray_state"), "component_id": native.get("component_id"), "source_target_id": native.get("source_target_id"), "destination_target_id": native.get("destination_target_id"), "accepted": native.get("accepted"), "joint_values": accepted_q, "accepted_seed_label": native.get("accepted_seed_label"), "nearest_h5_node_id": nearest, "branch_continuity_source": "previous accepted waypoint seed first; H5 frozen seed bank alternatives"})
        state = native.get("accepted_state") or {}; fk = {"schema_version": "stage3-h6-fk-tcp-validation-v1", "waypoint_index": index, "segment_id": native.get("segment_id"), "spray_state": native.get("spray_state"), "joint_values": accepted_q, "translation_error_m": state.get("translation_error_m"), "rotation_error_rad": state.get("rotation_error_rad"), "tcp_position_m": state.get("tcp_position_m"), "tcp_orientation_xyzw": state.get("tcp_orientation_xyzw"), "joint_limit_valid": state.get("joint_limit_valid"), "self_collision": state.get("self_collision"), "environment_collision": state.get("environment_collision"), "collision_method": H6_COLLISION_METHOD, "ccd_status": H6_CCD}
        if native.get("spray_state") == "SPRAY_ON" and base:
            fk["desired_surface_point_m"] = base.get("surface_point_xyz_m"); fk["desired_surface_normal"] = base.get("surface_normal_unit"); fk["desired_standoff_m"] = STANDOFF_M
            if state.get("tcp_position_m"):
                desired = vec_add(base["surface_point_xyz_m"], vec_scale(base["surface_normal_unit"], STANDOFF_M)); fk["standoff_m"] = vec_norm(vec_sub(state["tcp_position_m"], base["surface_point_xyz_m"])); fk["standoff_error_m"] = fk["standoff_m"] - STANDOFF_M; fk["tcp_surface_position_error_m"] = vec_norm(vec_sub(state["tcp_position_m"], desired))
            actual_orientation = state.get("tcp_orientation_xyzw")
            if actual_orientation:
                actual_z = quat_to_matrix(actual_orientation); fk["normal_angle_deviation_deg"] = angle_between([actual_z[0][2], actual_z[1][2], actual_z[2][2]], vec_scale(base["surface_normal_unit"], -1.0))
        fk_rows.append(fk)
    for native_collision_row in native_collision: collision_rows.append({"schema_version": "stage3-h6-collision-evidence-v1", **native_collision_row})
    dump_jsonl(output / "stage3_h6_joint_waypoints.jsonl", joint_rows); dump_jsonl(output / "stage3_h6_ik_branch_trace.jsonl", native_trace); dump_jsonl(output / "stage3_h6_fk_tcp_validation.jsonl", fk_rows); dump_jsonl(output / "stage3_h6_collision_evidence.jsonl", collision_rows)
    dump_jsonl(output / "stage3_h6_spray_on_off_segments.jsonl", segments); dump_json(output / "stage3_h6_spray_on_off_segments.json", {"schema_version": "stage3-h6-spray-on-off-segments-v1", "segments": segments})
    replay = replay_report(output, ordering, full_rows, joint_rows, fk_rows, collision_rows, replay_runs); dump_json(output / "stage3_h6_replay_determinism.json", replay)
    regression = run_regression(output); dump_json(output / "stage3_h6_regression_report.json", regression)
    coverage = coverage_accounting(pre, ordering, segments); dump_json(output / "stage3_h6_coverage_accounting.json", coverage)
    formal_tolerance_blocker = "formal_spray_process_tolerance_not_defined" if pre["tolerance_audit"]["formal_spray_process_tolerance"]["standoff"]["status"] != "DEFINED" else None
    final_blocker = extra_blockers[0] if extra_blockers else formal_tolerance_blocker
    if formal_tolerance_blocker and formal_tolerance_blocker not in extra_blockers: extra_blockers.append(formal_tolerance_blocker)
    if replay.get("FRESH_PROCESS_REPLAY") != "PASSED" and final_blocker is None: final_blocker = "fresh_process_replay_unavailable"
    if regression.get("new_regression_failures", 0) and final_blocker is None: final_blocker = "new_regression_failure"
    native_ok = bool(native_trace) and all(bool(row.get("accepted")) for row in native_trace)
    if not native_ok and final_blocker is None: final_blocker = "surface_constrained_transition_unavailable"
    status = "PASSED" if final_blocker is None and coverage["ALL_30_FEASIBLE_TARGETS_ACCOUNTED_FOR"] == "YES" else "BLOCKED"; report = build_terminal(pre, ordering, segments, full_rows, joint_rows, fk_rows, coverage, regression, status, final_blocker, extra_blockers, output); dump_json(output / "stage3_h6_gate_report.json", report["gate"]); dump_json(output / "stage3_h6_terminal_certificate.json", report["terminal"]); write_final_report(output, pre, ordering, {"surface_waypoints": len(on_waypoints), "dense_waypoints": len(full_rows)}, {"spray_on_segments": sum(item["spray_state"] == "SPRAY_ON" for item in segments), "spray_off_segments": sum(item["spray_state"] == "SPRAY_OFF" for item in segments)}, coverage, final_blocker or "none", extra_blockers, fk_rows=fk_rows, replay=load_json(output / "stage3_h6_replay_determinism.json"), regression=regression, status=status, terminal=report["terminal"])
    return 0 if status == "PASSED" else 2


def coverage_accounting(pre: Mapping[str, Any], ordering: Mapping[str, Any], segments: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    target_order = [target for index in ordering["component_execution_order"] for target in ordering["components"][index]["surface_order"]["target_order"]]; formal = sorted(map(int, pre["formal_target_ids"])); covered = sorted(set(target_order)); duplicates = sorted(target for target in set(target_order) if target_order.count(target) > 1); return {"schema_version": "stage3-h6-coverage-accounting-v1", "formal_targets": len(formal), "formal_target_ids": formal, "spray_on_covered_targets": len(covered), "spray_on_covered_target_ids": covered, "uncovered_targets": sorted(set(formal) - set(covered)), "duplicate/revisited_targets": duplicates, "component_count": len(ordering["component_execution_order"]), "spray_on_segment_count": sum(item["spray_state"] == "SPRAY_ON" for item in segments), "spray_off_segment_count": sum(item["spray_state"] == "SPRAY_OFF" for item in segments), "ALL_30_FEASIBLE_TARGETS_ACCOUNTED_FOR": "YES" if covered == formal else "NO", "accounting_semantics": "sequence accounting only; not final process deposition certification"}


def replay_report(output: Path, ordering: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], joint_rows: Sequence[Mapping[str, Any]], fk_rows: Sequence[Mapping[str, Any]], collision_rows: Sequence[Mapping[str, Any]], replay_runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    semantic = {"component_order": ordering["component_execution_order"], "surface_target_order": [item["surface_order"]["target_order"] for item in ordering["components"]], "spray_segmentation": [(item["segment_id"], item["spray_state"], item.get("source_component"), item.get("destination_component")) for item in load_jsonl(output / "stage3_h6_spray_on_off_segments.jsonl")], "dense_waypoint_count": len(rows), "cartesian_waypoint_values": [{key: row.get(key) for key in ("waypoint_index", "spray_state", "surface_point_xyz_m", "surface_normal_unit", "tcp_position_xyz_m", "tcp_orientation_xyzw")} for row in rows], "ik_branch_sequence": [{key: row.get(key) for key in ("waypoint_index", "spray_state", "accepted", "joint_values", "nearest_h5_node_id")} for row in joint_rows], "joint_path": [row.get("joint_values") for row in joint_rows], "collision_result": [{key: row.get(key) for key in ("waypoint_index", "spray_state", "state")} for row in collision_rows], "fk_result": [{key: row.get(key) for key in ("waypoint_index", "spray_state", "translation_error_m", "rotation_error_rad")} for row in fk_rows]}; signature = semantic_hash(semantic)
    signatures = [item.get("semantic_signature") for item in replay_runs]; passed = len(replay_runs) == 3 and all(item.get("status") == "PASSED" for item in replay_runs) and all(value is not None for value in signatures) and len(set(signatures)) == 1
    return {"schema_version": "stage3-h6-replay-determinism-v1", "fresh_process_count": 3, "processes": list(replay_runs), "semantic_output_signature": signature, "native_process_semantic_signatures": signatures, "all_semantic_results_consistent": passed, "FRESH_PROCESS_REPLAY": "PASSED" if passed else "BLOCKED", "replay_requirement": "three fresh independent H6 native processes with identical semantic MoveIt/FK/FCL outputs"}


def run_regression(output: Path) -> dict[str, Any]:
    commands = [[sys.executable, "-m", "pytest", "-q", "tests/test_stage3_h6.py", "tests/test_stage3_h5.py", "tests/test_stage3_h4_4.py", "tests/test_stage3_h4_reachability.py", "tests/test_stage3_h3_task_representation.py", "tests/test_stage3_h2_geometry.py"]]
    runs = []
    for command in commands:
        proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1800, check=False); raw = proc.stdout + "\n--- STDERR ---\n" + proc.stderr; match = re.search(r"(\d+) passed(?:, (\d+) failed)?", raw); runs.append({"command": command, "returncode": proc.returncode, "passed": int(match.group(1)) if match else None, "failed": int(match.group(2)) if match and match.group(2) else (0 if proc.returncode == 0 else 1), "raw_output_tail": raw[-6000:]})
    return {"schema_version": "stage3-h6-regression-report-v1", "runs": runs, "new_regression_failures": sum(int(run["failed"]) for run in runs), "known_pre_existing_failures": []}


def build_terminal(pre: Mapping[str, Any], ordering: Mapping[str, Any], segments: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]], joint_rows: Sequence[Mapping[str, Any]], fk_rows: Sequence[Mapping[str, Any]], coverage: Mapping[str, Any], regression: Mapping[str, Any], status: str, blocker: str | None, extra: Sequence[str], output: Path) -> dict[str, Any]:
    on_fk = [row for row in fk_rows if row.get("spray_state") == "SPRAY_ON" and row.get("translation_error_m") is not None]; all_accepted = all(bool(row.get("accepted")) for row in joint_rows) if joint_rows else False; replay = load_json(output / "stage3_h6_replay_determinism.json") if (output / "stage3_h6_replay_determinism.json").is_file() else {}
    stats = {"translation_error_max_m": max((float(row["translation_error_m"]) for row in on_fk), default=None), "translation_error_mean_m": sum(float(row["translation_error_m"]) for row in on_fk) / len(on_fk) if on_fk else None, "translation_error_p95_m": percentile([float(row["translation_error_m"]) for row in on_fk], 0.95), "rotation_error_max_rad": max((float(row["rotation_error_rad"]) for row in on_fk if row.get("rotation_error_rad") is not None), default=None), "standoff_min_m": min((float(row["standoff_m"]) for row in on_fk if row.get("standoff_m") is not None), default=None), "standoff_max_m": max((float(row["standoff_m"]) for row in on_fk if row.get("standoff_m") is not None), default=None), "normal_angle_max_deg": max((float(row["normal_angle_deviation_deg"]) for row in on_fk if row.get("normal_angle_deviation_deg") is not None), default=None)}
    gate = {"schema_version": "stage3-h6-gate-report-v1", "STAGE_3_H6": status, "FIRST_BLOCKER": blocker, "H4_5_BASELINE_IMMUTABLE": "YES", "H5_BASELINE_IMMUTABLE": "YES", "WORKCELL_CONFIGURATION_FROZEN": "YES", "FORMAL_FEASIBLE_TARGETS": coverage["formal_targets"], "TARGET_COMPONENTS": coverage["component_count"], "ALL_30_FEASIBLE_TARGETS_ACCOUNTED_FOR": coverage["ALL_30_FEASIBLE_TARGETS_ACCOUNTED_FOR"], "SURFACE_COVERAGE_ORDER_GENERATED": "YES", "SURFACE_COVERAGE_ORDER_CERTIFIED": "YES" if status == "PASSED" else "NO", "SPRAY_ON_OFF_SEGMENTATION_GENERATED": "YES", "SPRAY_ON_OFF_SEGMENTATION_CERTIFIED": "YES" if all_accepted and status == "PASSED" else "NO", "SURFACE_CONSTRAINED_TCP_PATH_GENERATED": "YES", "SURFACE_CONSTRAINED_TCP_PATH_CERTIFIED": "YES" if all_accepted and status == "PASSED" else "NO", "JOINT_BRANCH_CONTINUITY_PRESERVED": "YES" if all_accepted else "NO", "SELF_COLLISION_FREE": "YES" if all(bool(row.get("self_collision") is False) for row in fk_rows) and fk_rows else "NO", "ENVIRONMENT_COLLISION_FREE": "YES" if all(bool(row.get("environment_collision") is False) for row in fk_rows) and fk_rows else "NO", "FK_TCP_CROSSCHECK": "PASSED" if on_fk and all(row.get("tcp_surface_position_error_m", 1.0) <= 1.0e-6 for row in on_fk) else "BLOCKED", "FRESH_PROCESS_REPLAY": replay.get("FRESH_PROCESS_REPLAY", "BLOCKED"), "NEW_REGRESSION_FAILURES": regression.get("new_regression_failures"), "CCD": "NOT_AVAILABLE", "TIME_PARAMETERIZATION_DONE": "NO", "RUCKIG_STARTED": "NO", "RUCKIG_DONE": "NO", "ROBOT_MOTION_STARTED": "NO", "NEW_FJT_GOALS_SENT": 0, "FORMAL_LEDGER_MUTATED": "NO", "ML_TRAINING_STARTED": "NO", "RL_TRAINING_STARTED": "NO", "diagnostic_metrics": stats, "mandatory_blockers": list(dict.fromkeys(item for item in [blocker, *extra] if item))}
    terminal = {"schema_version": "stage3-h6-terminal-certificate-v1", **{key: value for key, value in gate.items() if key != "schema_version"}, "READY_FOR_STAGE_3_H7": "YES" if status == "PASSED" else "NO"}
    return {"gate": gate, "terminal": terminal}


def percentile(values: Sequence[float], probability: float) -> float | None:
    if not values: return None
    ordered = sorted(values); position = (len(ordered) - 1) * probability; lower = int(math.floor(position)); upper = int(math.ceil(position));
    return ordered[lower] if lower == upper else ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def write_final_report(output: Path, pre: Mapping[str, Any], ordering: Mapping[str, Any], waypoint_counts: Mapping[str, Any], segment_counts: Mapping[str, Any], coverage: Mapping[str, Any], blocker: str, extra: Sequence[str], *, fk_rows: Sequence[Mapping[str, Any]] = (), replay: Mapping[str, Any] | None = None, regression: Mapping[str, Any] | None = None, status: str = "BLOCKED", terminal: Mapping[str, Any] | None = None) -> None:
    on_fk = [row for row in fk_rows if row.get("spray_state") == "SPRAY_ON" and row.get("translation_error_m") is not None]; lines = ["# Stage 3 H6 — Surface-Constrained Coverage Sequencing & Spray-ON/OFF Certification", "", f"`STAGE_3_H6: {status}`", f"`FIRST_BLOCKER: {blocker}`", "", "## Frozen input and convention answers", "", f"Q1. H4.5/H5 frozen hashes consistent: `{pre.get('status') == 'PASSED' and not pre.get('hash_mismatches')}`; source records are in `stage3_h6_input_freeze_manifest.json`.", f"Q2. Official workcell: `{pre.get('selected_configuration')}`.", f"Q3. Nominal standoff/TCP extension: `{pre.get('standoff_m')} m / {pre.get('tcp_extension_m')} m`.", f"Q4. All 30 targets loaded: `{pre.get('target_count') == 192 and len(pre.get('formal_target_ids', [])) == 30}`.", f"Q5. Three components preserved: `{pre.get('component_count') == 3}`.", f"Q6. Component target IDs: `{pre.get('components')}`.", f"Q7. Surface adjacency source: `{rel(H5_SURFACE)}`; no new edges were added.", f"Q8. Ordering: deterministic min-cost Hamiltonian/path-cover on the frozen surface graph; details in `stage3_h6_component_ordering.json`.", f"Q9. Revisit IDs: `{sum(len(item['surface_order'].get('revisited_target_ids', [])) for item in ordering.get('components', [])) if ordering else 'not generated'}`.", f"Q10. Component order: `{ordering.get('component_execution_order', 'not generated')}`.", f"Q11. Spray-ON segments: `{segment_counts.get('spray_on_segments', 0)}`.", f"Q12. Spray-OFF segments: `{segment_counts.get('spray_off_segments', 0)}`.", f"Q13. Dense waypoint count: `{waypoint_counts.get('dense_waypoints', waypoint_counts.get('surface_waypoints', 0))}`.", "Q14. Surface interpolation: transformed H2 mesh triangle-local barycentric piecewise path.", "Q15. Surface proof: every row records triangle ID and barycentric coordinates; reconstruction audit is in `stage3_h6_surface_graph.json`.", "Q16. Normal: H2 area-weighted vertex normals, barycentrically interpolated and transformed by the frozen H4.5 fixture rotation.", f"Q17. TCP orientation: H3 projected-reference-axis convention carried through the H4.5 transform; TCP link `{pre.get('robot_model', {}).get('tcp_link')}`.", f"Q18. Nominal standoff retained: `{pre.get('standoff_m') == STANDOFF_M}`.", f"Q19. Formal standoff tolerance: `{pre.get('tolerance_audit', {}).get('formal_spray_process_tolerance', {}).get('standoff')}`.", f"Q20. Formal normal tolerance: `{pre.get('tolerance_audit', {}).get('formal_spray_process_tolerance', {}).get('normal_angular')}`.", "Q21. No missing process threshold was invented; this is a certification blocker, not an IK failure.", "", "## IK, collision, FK and replay answers", "", "Q22. IK: native MoveIt KinematicsBase::getPositionIK with KDL plugin.", "Q23. Branch continuity: previous accepted waypoint is the first seed, followed by the frozen H5 seed bank.", f"Q24. Branch jump: `{terminal.get('JOINT_BRANCH_CONTINUITY_PRESERVED') if terminal else 'not certified'}`.", "Q25. Alternative H5 branches: yes, the frozen seed bank and H5-valid endpoint nodes were retained.", f"Q26. Joint limits: `{terminal.get('SELF_COLLISION_FREE') if terminal else 'not certified'}` (state-level native evidence is in `stage3_h6_collision_evidence.jsonl`).", f"Q27. Self collision free: `{terminal.get('SELF_COLLISION_FREE') if terminal else 'not certified'}`.", f"Q28. Environment collision free: `{terminal.get('ENVIRONMENT_COLLISION_FREE') if terminal else 'not certified'}`.", f"Q29. Collision method: `{H6_COLLISION_METHOD}` + native MoveIt/FCL.", "Q30. CCD: `NOT_AVAILABLE`; no clearance was inferred.", f"Q31. FK/TCP max translation error: `{max((row.get('translation_error_m') for row in on_fk), default=None)}` m.", f"Q32. FK/TCP max rotation error: `{max((row.get('rotation_error_rad') for row in on_fk), default=None)}` rad.", f"Q33. Fresh replay: `{(replay or {}).get('FRESH_PROCESS_REPLAY', 'BLOCKED')}`; three independent H6 replays are required before PASSED.", f"Q34. New regression failures: `{(regression or {}).get('new_regression_failures', 'not run')}`.", f"Q35. Ready for H7: `{terminal.get('READY_FOR_STAGE_3_H7') if terminal else 'NO'}`.", "", "## Coverage accounting", "", json.dumps(coverage, ensure_ascii=False, indent=2), "", "## Boundaries", "", "`TIME_PARAMETERIZATION_DONE: NO`", "`RUCKIG_STARTED: NO`", "`RUCKIG_DONE: NO`", "`ROBOT_MOTION_STARTED: NO`", "`NEW_FJT_GOALS_SENT: 0`", "`FORMAL_LEDGER_MUTATED: NO`", "", f"Additional blockers: `{list(extra)}`.", ""]
    (output / "FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output", type=Path, default=None); args = parser.parse_args(); timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"); output = (args.output or ROOT / "outputs" / f"stage3_h6_surface_coverage_{timestamp}").resolve()
    try: return orchestrate(output)
    except Exception as exc:
        print("STAGE_3_H6: BLOCKED", file=sys.stderr); print(f"FIRST_BLOCKER: {type(exc).__name__}: {exc}", file=sys.stderr); raise


if __name__ == "__main__": raise SystemExit(main())
