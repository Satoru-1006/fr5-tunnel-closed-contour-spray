"""Pure, deterministic contracts for the Stage 3 H4.4 audit.

The ROS runner lives in ``scripts/stage3_h4_4.py``.  This module deliberately
contains only numerical and manifest logic so it can be tested without ROS.
It does not change any H0--H4.3 artifact and it does not contain a motion or
planning entry point.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from itertools import combinations, product
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "stage3-h4-4-contract-v1"
COLLISION_METHOD = "adaptive_discrete_interpolation"
UNAVAILABLE = "not_available"
TRANSLATION_LIMIT_M = 0.05
ROTATION_LIMITS_DEG = {"yaw": 15.0, "pitch": 10.0, "roll": 10.0}

AXES = (
    ("dx", "translation", 0, -TRANSLATION_LIMIT_M, "dx_minus_050"),
    ("dx", "translation", 0, TRANSLATION_LIMIT_M, "dx_plus_050"),
    ("dy", "translation", 1, -TRANSLATION_LIMIT_M, "dy_minus_050"),
    ("dy", "translation", 1, TRANSLATION_LIMIT_M, "dy_plus_050"),
    ("dz", "translation", 2, -TRANSLATION_LIMIT_M, "dz_minus_050"),
    ("dz", "translation", 2, TRANSLATION_LIMIT_M, "dz_plus_050"),
    ("yaw", "rotation", 0, -ROTATION_LIMITS_DEG["yaw"], "yaw_minus_15"),
    ("yaw", "rotation", 0, ROTATION_LIMITS_DEG["yaw"], "yaw_plus_15"),
    ("pitch", "rotation", 1, -ROTATION_LIMITS_DEG["pitch"], "pitch_minus_10"),
    ("pitch", "rotation", 1, ROTATION_LIMITS_DEG["pitch"], "pitch_plus_10"),
    ("roll", "rotation", 2, -ROTATION_LIMITS_DEG["roll"], "roll_minus_10"),
    ("roll", "rotation", 2, ROTATION_LIMITS_DEG["roll"], "roll_plus_10"),
)


def canonicalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): canonicalize(value[k]) for k in sorted(value, key=lambda x: str(x))}
    if isinstance(value, (list, tuple)):
        return [canonicalize(x) for x in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite value cannot be canonicalized")
        return 0.0 if value == 0.0 else value
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(canonicalize(value), ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)


def semantic_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def vec_sub(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [float(a[i]) - float(b[i]) for i in range(3)]


def vec_norm(a: Sequence[float]) -> float:
    return math.sqrt(sum(float(x) * float(x) for x in a))


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(float(a[i]) * float(b[i]) for i in range(3))


def apply_transform(matrix: Sequence[Sequence[float]], point: Sequence[float]) -> list[float]:
    return [sum(float(matrix[row][col]) * float(point[col]) for col in range(3)) + float(matrix[row][3]) for row in range(3)]


def rotate_vector(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
    return [sum(float(matrix[row][col]) * float(vector[col]) for col in range(3)) for row in range(3)]


def placement_matrix(tx: float, ty: float, tz: float, yaw_deg: float = 0.0, pitch_deg: float = 0.0, roll_deg: float = 0.0) -> list[list[float]]:
    yaw, pitch, roll = [math.radians(v) for v in (yaw_deg, pitch_deg, roll_deg)]
    cz, sz = math.cos(yaw), math.sin(yaw)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cx, sx = math.cos(roll), math.sin(roll)
    r = [
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
        [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
        [-sy, cy * sx, cy * cx],
    ]
    return [r[0] + [float(tx)], r[1] + [float(ty)], r[2] + [float(tz)], [0.0, 0.0, 0.0, 1.0]]


def _zero_delta() -> dict[str, Any]:
    return {"translation_m": [0.0, 0.0, 0.0], "yaw_pitch_roll_deg": [0.0, 0.0, 0.0]}


def _apply_axis(delta: dict[str, Any], axis: tuple[str, str, int, float, str]) -> None:
    _name, kind, index, value, _label = axis
    if kind == "translation":
        delta["translation_m"][index] = float(value)
    else:
        delta["yaw_pitch_roll_deg"][index] = float(value)


def _label(axes: Sequence[tuple[str, str, int, float, str]]) -> str:
    return "_".join(axis[4] for axis in axes)


def _spec(base: Sequence[Sequence[float]], axes: Sequence[tuple[str, str, int, float, str]], index: int) -> dict[str, Any]:
    delta = _zero_delta()
    for axis in axes:
        _apply_axis(delta, axis)
    t = delta["translation_m"]
    r = delta["yaw_pitch_roll_deg"]
    matrix = placement_matrix(float(base[0][3]) + t[0], float(base[1][3]) + t[1], float(base[2][3]) + t[2], yaw_deg=r[0], pitch_deg=r[1], roll_deg=r[2])
    label = "baseline" if not axes else _label(axes)
    return {
        "schema_version": "stage3-h4-4-placement-candidate-v1",
        "candidate_index": index,
        "candidate_id": f"h4_4_{index:03d}_{label}",
        "label": label,
        "delta_translation_m": list(t),
        "delta_yaw_pitch_roll_deg": list(r),
        "transforms": {"fixture_curved_cylinder_patch": matrix},
        "placement_status": "EXPERIMENTAL_DERIVED_PLACEMENT",
        "search_method": "bounded_deterministic_endpoint_enumeration",
        "generation_rule": "baseline + all single-axis endpoints + all two-axis endpoint combinations",
        "random_search": False,
        "bayesian_optimization": False,
        "ml": False,
        "rl": False,
        "path_planner": False,
        "ruckig": False,
    }


def placement_bank(base_transform: Sequence[Sequence[float]]) -> list[dict[str, Any]]:
    """Return the frozen first-layer bank: 1 + 12 + 60 = 73 candidates."""

    result = [_spec(base_transform, (), 0)]
    index = 1
    for axis in AXES:
        result.append(_spec(base_transform, (axis,), index))
        index += 1
    axis_names = []
    for axis in AXES:
        if axis[0] not in axis_names:
            axis_names.append(axis[0])
    for left_name, right_name in combinations(axis_names, 2):
        # AXES contains two signed endpoints per named degree of freedom. The
        # nested loops intentionally include all four sign pairs.
        left_signs = [axis for axis in AXES if axis[0] == left_name]
        right_signs = [axis for axis in AXES if axis[0] == right_name]
        for left_axis in left_signs:
            for right_axis in right_signs:
                result.append(_spec(base_transform, (left_axis, right_axis), index))
                index += 1
    if len(result) != 73:
        raise AssertionError(f"unexpected H4.4 bank size: {len(result)}")
    return result


def displacement_norm(candidate: Mapping[str, Any]) -> float:
    t = candidate["delta_translation_m"]
    r = candidate["delta_yaw_pitch_roll_deg"]
    return math.sqrt(sum(float(x) ** 2 for x in t) + sum(math.radians(float(x)) ** 2 for x in r))


def geometry_counts(target_rows: Iterable[Mapping[str, Any]], geometry_id: str) -> dict[str, Any]:
    selected = [row for row in target_rows if row.get("geometry_id") == geometry_id]
    counts: dict[str, int] = {}
    for row in selected:
        classification = str(row.get("classification", "UNKNOWN"))
        counts[classification] = counts.get(classification, 0) + 1
    return {
        "target_count": len(selected),
        "classification_counts": dict(sorted(counts.items())),
        "reachable_collision_free": counts.get("REACHABLE_COLLISION_FREE", 0),
        "ik_unreachable": counts.get("IK_UNREACHABLE", 0),
        "fk_mismatch": counts.get("IK_FOUND_BUT_FK_MISMATCH", 0),
        "environment_collision": counts.get("IK_FOUND_ENV_COLLISION", 0),
        "self_collision": counts.get("IK_FOUND_SELF_COLLISION", 0),
    }


def _nearest_neighbors(points: Mapping[int, Sequence[float]], k: int = 4) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for index, point in points.items():
        distances = sorted((vec_norm(vec_sub(other, point)), other_index) for other_index, other in points.items() if other_index != index)
        for _distance, other_index in distances[:k]:
            edges.add(tuple(sorted((int(index), int(other_index)))))
    return edges


def continuity_graph(
    target_rows: Sequence[Mapping[str, Any]],
    ik_rows: Sequence[Mapping[str, Any]],
    fk_rows: Sequence[Mapping[str, Any]],
    native_rows: Mapping[tuple[int, str], Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]],
    geometry_id: str = "fixture_curved_cylinder_patch",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a conservative target/branch adjacency graph.

    H3 uses deterministic area-weighted samples rather than a rectilinear
    grid.  Consequently the graph records a fixed symmetric 4-nearest
    surface adjacency and reports raw joint/TCP deltas; it applies no
    invented joint-space threshold.
    """

    target_map = {i: row for i, row in enumerate(targets) if row.get("geometry_id") == geometry_id}
    points = {i: row["surface_point_xyz_m"] for i, row in target_map.items()}
    surface_edges = _nearest_neighbors(points, k=4)
    target_lookup = {int(row["target_index"]): row for row in target_rows if row.get("geometry_id") == geometry_id}
    ik_lookup = {int(row["target_index"]): row for row in ik_rows if row.get("geometry_id") == geometry_id}
    fk_lookup = {(int(row["target_index"]), str(row["candidate_id"])): row for row in fk_rows if row.get("target_index") is not None}
    nodes: list[dict[str, Any]] = []
    node_ids: set[tuple[int, str]] = set()
    for target_index, target in sorted(target_lookup.items()):
        if target.get("classification") != "REACHABLE_COLLISION_FREE":
            continue
        for candidate in ik_lookup.get(target_index, {}).get("deduplicated_candidates", []):
            cid = str(candidate["candidate_id"])
            fk = fk_lookup.get((target_index, cid), {})
            native = native_rows.get((target_index, cid), {})
            if not fk.get("fk_valid") or native.get("collision_free") is not True:
                continue
            node = {"target_id": target_index, "solution_id": cid, "joint_values": candidate["joint_positions"], "fk_valid": True, "joint_limit_valid": bool(fk.get("joint_validation", {}).get("valid", True)), "native_fcl_collision_free": True}
            nodes.append(node)
            node_ids.add((target_index, cid))
    edges: list[dict[str, Any]] = []
    adjacency: dict[tuple[int, str], set[tuple[int, str]]] = {node: set() for node in node_ids}
    for left_target, right_target in sorted(surface_edges):
        if left_target not in target_lookup or right_target not in target_lookup:
            continue
        left_nodes = [node for node in nodes if node["target_id"] == left_target]
        right_nodes = [node for node in nodes if node["target_id"] == right_target]
        for left in left_nodes:
            for right in right_nodes:
                joint_delta = [float(right["joint_values"][j]) - float(left["joint_values"][j]) for j in range(6)]
                tcp_left = target_map[left_target]["tcp_target_position_xyz_m"]
                tcp_right = target_map[right_target]["tcp_target_position_xyz_m"]
                edge = {
                    "source": {"target_id": left_target, "solution_id": left["solution_id"]},
                    "target": {"target_id": right_target, "solution_id": right["solution_id"]},
                    "surface_adjacency": True,
                    "joint_delta_rad": joint_delta,
                    "joint_delta_norm_rad": vec_norm(joint_delta),
                    "tcp_delta_m": vec_sub(tcp_right, tcp_left),
                    "tcp_delta_norm_m": vec_norm(vec_sub(tcp_right, tcp_left)),
                    "continuity_threshold_status": "NOT_DECLARED_RAW_STATISTICS_ONLY",
                    "continuity_classification": "SURFACE_ADJACENT_COLLISION_FREE_BRANCH_PAIR",
                }
                edges.append(edge)
                a, b = (left_target, left["solution_id"]), (right_target, right["solution_id"])
                adjacency[a].add(b)
                adjacency[b].add(a)
    components: list[set[tuple[int, str]]] = []
    unseen = set(node_ids)
    while unseen:
        root = min(unseen)
        unseen.remove(root)
        component = {root}
        queue: deque[tuple[int, str]] = deque([root])
        while queue:
            current = queue.popleft()
            for neighbor in sorted(adjacency[current]):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)
        components.append(component)
    largest = max(components, key=lambda component: (len({x[0] for x in component}), len(component), sorted(component))) if components else set()
    graph = {
        "schema_version": "stage3-h4-4-curved-continuity-graph-v1",
        "geometry_id": geometry_id,
        "nodes": nodes,
        "edges": edges,
        "surface_adjacency_policy": "symmetric 4-nearest-neighbor graph over frozen H3 area-weighted targets; deterministic distance ordering",
        "joint_continuity_threshold": None,
        "joint_continuity_threshold_status": "NOT_DECLARED_RAW_STATISTICS_ONLY",
        "components": [[{"target_id": target, "solution_id": solution} for target, solution in sorted(component)] for component in sorted(components, key=lambda x: (len(x), sorted(x)), reverse=True)],
    }
    summary = {
        "schema_version": "stage3-h4-4-curved-continuity-summary-v1",
        "collision_free_target_count": len({node["target_id"] for node in nodes}),
        "collision_free_branch_count": len(nodes),
        "connected_component_count": len(components),
        "largest_connected_component_target_count": len({x[0] for x in largest}),
        "largest_connected_component_branch_count": len(largest),
        "adjacent_collision_free_edge_count": len(edges),
        "surface_adjacency_edge_count": len(surface_edges),
        "continuity_evidence_status": "PRESENT_RAW_DELTAS_ONLY" if nodes else "NOT_PRESENT",
        "isolated_only": bool(nodes) and not edges,
        "h5_continuity_gate_satisfied": bool(edges and len({x[0] for x in largest}) >= 2),
        "no_formal_joint_threshold_invented": True,
    }
    return graph, summary


def ranking_key(candidate: Mapping[str, Any], metrics: Mapping[str, Any]) -> tuple[Any, ...]:
    curved = metrics.get("curved_metrics", {})
    continuity = metrics.get("continuity_summary", {})
    clearance = metrics.get("minimum_clearance_m")
    clearance_key = -float(clearance) if isinstance(clearance, (int, float)) and math.isfinite(float(clearance)) else float("inf")
    return (
        -int(curved.get("reachable_collision_free", 0)),
        -int(continuity.get("largest_connected_component_target_count", 0)),
        int(curved.get("environment_collision", 0)),
        int(curved.get("self_collision", 0)),
        int(curved.get("ik_unreachable", 0)),
        int(curved.get("fk_mismatch", 0)),
        clearance_key,
        float(metrics.get("displacement_norm", displacement_norm(candidate))),
        str(candidate.get("candidate_id", "")),
    )


def immutable_mismatch_count(before: Mapping[str, Any], after: Mapping[str, Any]) -> int:
    mismatches = 0
    before_items = before.get("artifacts", {})
    after_items = after.get("artifacts", {})
    for key in sorted(set(before_items) | set(after_items)):
        if before_items.get(key) != after_items.get(key):
            mismatches += 1
    return mismatches


__all__ = [
    "AXES",
    "COLLISION_METHOD",
    "ROTATION_LIMITS_DEG",
    "SCHEMA_VERSION",
    "TRANSLATION_LIMIT_M",
    "UNAVAILABLE",
    "apply_transform",
    "canonical_json",
    "continuity_graph",
    "displacement_norm",
    "dot",
    "geometry_counts",
    "immutable_mismatch_count",
    "placement_bank",
    "placement_matrix",
    "ranking_key",
    "rotate_vector",
    "semantic_hash",
    "vec_norm",
    "vec_sub",
]
