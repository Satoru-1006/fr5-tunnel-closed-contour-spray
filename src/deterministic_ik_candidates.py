"""Deterministic Stage 1.9 candidate generation primitives.

The module is deliberately independent of ROS2.  The ROS runner supplies one
public ``set_from_ik`` invocation for each (task pose, explicit seed) pair;
this module owns the stable identities, periodic-joint handling, provenance,
deduplication, sorting, edge records, and exact DP tie-break semantics.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from src.ik_candidate_graph import circular_joint_delta, nearest_equivalent_joint_positions


COLLISION_METHOD = "adaptive_discrete_interpolation"
UNAVAILABLE = "not_available"


def _finite(value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite numeric value: {value!r}")
    return result


def quantize(value: float, step: float) -> float:
    """Quantize only for identity/hash/sort keys, never for physical values."""

    return float(round(_finite(value) / float(step)) * float(step))


def canonical_json(value: Any) -> str:
    """Stable UTF-8 JSON representation with no NaN/Infinity."""

    def clean(item: Any) -> Any:
        if isinstance(item, np.ndarray):
            return clean(item.tolist())
        if isinstance(item, (np.integer, np.floating)):
            return clean(item.item())
        if isinstance(item, Mapping):
            return {str(k): clean(item[k]) for k in sorted(item, key=lambda k: str(k))}
        if isinstance(item, (list, tuple)):
            return [clean(x) for x in item]
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError("NaN/Infinity is forbidden in canonical JSON")
            return item
        return item

    return json.dumps(clean(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_canonical(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class SeedTemplate:
    seed_template_id: str
    seed_template_family: str
    seed_template_parameters: tuple[tuple[str, float], ...]
    seed_joint_vector: tuple[float, ...]
    seed_order_index: int

    def to_record(self) -> dict[str, Any]:
        return {
            "seed_template_id": self.seed_template_id,
            "seed_template_family": self.seed_template_family,
            "seed_template_parameters": {k: v for k, v in self.seed_template_parameters},
            "seed_joint_vector": list(self.seed_joint_vector),
            "seed_order_index": self.seed_order_index,
        }


def _template_id(family: str, parameters: Mapping[str, float]) -> str:
    encoded = canonical_json({"family": family, "parameters": parameters})
    return f"seed:{family}:{hashlib.sha256(encoded.encode('utf-8')).hexdigest()[:12]}"


def explicit_seed_templates(
    waypoint_id: int,
    nominal_seed: Iterable[float],
    previous_seed: Iterable[float] | None,
    *,
    config: Mapping[str, Any],
) -> list[SeedTemplate]:
    """Create the configured seed list in explicit, process-independent order."""

    base = np.asarray(list(nominal_seed), dtype=float).reshape(-1)
    previous = None if previous_seed is None else np.asarray(list(previous_seed), dtype=float).reshape(-1)
    if previous is not None and previous.shape != base.shape:
        raise ValueError("previous seed shape mismatch")
    families = config.get("seed_templates", {})
    offsets = config.get("seed_offsets_rad", {})
    requested: list[tuple[str, np.ndarray, dict[str, float]]] = []

    if bool(families.get("include_nominal_seed", True)):
        requested.append(("nominal", base.copy(), {"waypoint_id": float(waypoint_id)}))
    if previous is not None and bool(families.get("include_previous_waypoint_seeds", True)):
        requested.append(("previous_waypoint", previous.copy(), {"source_waypoint_id": float(waypoint_id - 1)}))

    family_specs = (
        ("fixed_shoulder", 0, "include_fixed_shoulder_templates"),
        ("fixed_elbow", 2, "include_fixed_elbow_templates"),
        ("fixed_wrist", 4, "include_fixed_wrist_templates"),
    )
    for family, joint, enabled_key in family_specs:
        if not bool(families.get(enabled_key, True)):
            continue
        values = offsets.get(family, [-math.pi, math.pi])
        for offset in values:
            params = {"joint_index": float(joint), "offset_rad": float(offset)}
            candidate = base.copy()
            candidate[joint] += float(offset)
            requested.append((family, candidate, params))

    if bool(families.get("include_fixed_combination_templates", True)):
        for item in offsets.get("fixed_combination", [{"joint_indices": [0, 2, 4], "offset_rad": math.pi}]):
            indices = tuple(int(i) for i in item.get("joint_indices", []))
            offset = float(item.get("offset_rad", math.pi))
            candidate = base.copy()
            for index in indices:
                candidate[index] += offset
            requested.append(("fixed_combination", candidate, {"joint_indices": float(sum(indices)), "offset_rad": offset}))

    templates: list[SeedTemplate] = []
    seen: set[tuple[float, ...]] = set()
    for family, vector, parameters in requested:
        key = tuple(float(x) for x in vector)
        if key in seen:
            continue
        seen.add(key)
        normalized_parameters = tuple(sorted((str(k), float(v)) for k, v in parameters.items()))
        templates.append(SeedTemplate(_template_id(family, dict(normalized_parameters)), family, normalized_parameters, key, len(templates)))
    return templates


def stable_task_pose_identity(record: Mapping[str, Any], *, offset_step_mm: float = 1.0e-4, angle_step_deg: float = 1.0e-6) -> tuple[str, str]:
    """Return an identity independent of generation/discovery order."""

    stable_key = {
        "waypoint_id": int(record["waypoint_id"]),
        "quantized_tangential_offset": quantize(float(record.get("tangential_offset_mm", 0.0)), offset_step_mm),
        "quantized_longitudinal_offset": quantize(float(record.get("longitudinal_offset_mm", 0.0)), offset_step_mm),
        "quantized_standoff_offset": quantize(float(record.get("standoff_offset_mm", 0.0)), offset_step_mm),
        "quantized_roll_offset": quantize(float(record.get("roll_offset_deg", 0.0)), angle_step_deg),
        "repair_generation_type": str(record.get("generation_strategy", record.get("generation_type", "unknown"))),
        "formal_or_diagnostic": "formal" if bool(record.get("formal_constraint_pass", True)) and not bool(record.get("diagnostic_only", False)) else "diagnostic",
    }
    encoded = canonical_json(stable_key)
    return encoded, f"tp:{hashlib.sha256(encoded.encode('utf-8')).hexdigest()[:16]}"


def stable_node_identity(waypoint_id: int, task_pose_stable_id: str, q_unwrapped_rad: Iterable[float], *, step: float = 1.0e-7) -> tuple[str, str]:
    q = [quantize(float(x), step) for x in q_unwrapped_rad]
    stable_key = {"waypoint_id": int(waypoint_id), "task_pose_stable_id": str(task_pose_stable_id), "quantized_equivalent_joint_vector": q}
    encoded = canonical_json(stable_key)
    return encoded, f"node:{hashlib.sha256(encoded.encode('utf-8')).hexdigest()[:20]}"


def normalize_solution(solution: Iterable[float], reference: Iterable[float], *, lower: Iterable[float] | None = None, upper: Iterable[float] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Preserve full-precision physical q and return nearest-equivalent q."""

    raw = np.asarray(list(solution), dtype=float).reshape(-1)
    ref = np.asarray(list(reference), dtype=float).reshape(-1)
    if raw.shape != ref.shape:
        raise ValueError("solution/reference shape mismatch")
    q = nearest_equivalent_joint_positions(raw, ref)
    if lower is not None and upper is not None:
        lo = np.asarray(list(lower), dtype=float)
        hi = np.asarray(list(upper), dtype=float)
        q = np.minimum(np.maximum(q, lo), hi)
    return raw, q


def deduplicate_ik_records(records: Sequence[Mapping[str, Any]], tolerance_rad: float) -> list[dict[str, Any]]:
    """Deduplicate physical IK solutions while retaining complete provenance."""

    ordered = sorted((dict(record) for record in records), key=lambda r: (str(r["task_pose_stable_id"]), tuple(float(x) for x in r["q_unwrapped_rad"]), int(r.get("seed_order_index", 0))))
    result: list[dict[str, Any]] = []
    for record in ordered:
        q = np.asarray(record["q_unwrapped_rad"], dtype=float)
        duplicate = next((item for item in result if np.max(np.abs(circular_joint_delta(q - np.asarray(item["q_unwrapped_rad"], dtype=float)))) < tolerance_rad and item["waypoint_id"] == record["waypoint_id"] and item["task_pose_stable_id"] == record["task_pose_stable_id"]), None)
        if duplicate is None:
            record["source_seed_template_ids"] = [str(record["seed_template_id"])]
            record["source_seed_count"] = 1
            record["first_seed_template_id"] = str(record["seed_template_id"])
            record["all_seed_provenance"] = [dict(record.get("seed_provenance", {}))]
            result.append(record)
        else:
            duplicate["source_seed_template_ids"] = sorted(set(duplicate["source_seed_template_ids"] + [str(record["seed_template_id"])]))
            duplicate["source_seed_count"] = len(duplicate["source_seed_template_ids"])
            duplicate["all_seed_provenance"].append(dict(record.get("seed_provenance", {})))
    for record in result:
        _, stable_id = stable_node_identity(int(record["waypoint_id"]), str(record["task_pose_stable_id"]), record["q_unwrapped_rad"])
        record["stable_node_id"] = stable_id
        record["ik_candidate_id"] = stable_id
        record["candidate_id"] = stable_id
        record["all_seed_provenance"] = sorted(record["all_seed_provenance"], key=lambda x: (int(x.get("seed_order_index", 0)), str(x.get("seed_template_id", ""))))
    return sorted(result, key=ik_record_sort_key)


def ik_record_sort_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    q = tuple(quantize(float(x), 1.0e-7) for x in record.get("q_unwrapped_rad", []))
    return (str(record["task_pose_stable_id"]), *q, str(record["stable_node_id"]))


def task_pose_sort_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return (str(record["task_pose_stable_id"]),)


def transition_sort_key(record: Mapping[str, Any]) -> tuple[str, str]:
    return (str(record["source_stable_node_id"]), str(record["target_stable_node_id"]))


def deterministic_path_key(path: Sequence[Mapping[str, Any]], edges: Sequence[Mapping[str, Any]]) -> tuple[Any, ...]:
    max_step = max((float(e.get("max_joint_step_deg", 0.0)) for e in edges), default=0.0)
    motion = sum(float(e.get("total_joint_motion_rad", e.get("total_joint_change_rad", 0.0))) for e in edges)
    modified = sum(not bool(n.get("is_nominal", True)) for n in path)
    tcp_offset = sum(float(n.get("actual_position_offset_mm", n.get("tcp_offset_mm", 0.0))) for n in path)
    roll_standoff = sum(abs(float(n.get("roll_offset_deg", 0.0))) + abs(float(n.get("standoff_offset_mm", 0.0))) for n in path)
    return (sum(float(e.get("cost", 0.0)) for e in edges) + sum(float(n.get("node_cost", 0.0)) for n in path), max_step, motion, modified, tcp_offset, roll_standoff, tuple(str(n.get("task_pose_stable_id", "")) for n in path), tuple(str(n.get("stable_node_id", n.get("ik_candidate_id", ""))) for n in path))


def deterministic_dp(layers: Sequence[Sequence[Mapping[str, Any]]], edge_map: Mapping[tuple[str, str], Mapping[str, Any]]) -> dict[str, Any]:
    """Exact layered DP with the Stage 1.9 lexicographic tie-break."""

    if not layers:
        return {"found": False, "candidate_ids": [], "first_unreachable_waypoint": 0}
    states: list[tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]] = [([node], []) for node in sorted(layers[0], key=ik_record_sort_key) if bool(node.get("valid", node.get("formal_constraint_pass", False)))]
    if not states:
        return {"found": False, "candidate_ids": [], "first_unreachable_waypoint": 0}
    for waypoint in range(1, len(layers)):
        next_states: list[tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]] = []
        for path, edges in states:
            source = path[-1]
            for target in sorted(layers[waypoint], key=ik_record_sort_key):
                if not bool(target.get("valid", target.get("formal_constraint_pass", False))):
                    continue
                edge = edge_map.get((str(source.get("stable_node_id", source.get("ik_candidate_id"))), str(target.get("stable_node_id", target.get("ik_candidate_id")))))
                if edge is None or not bool(edge.get("valid", False)):
                    continue
                next_states.append((path + [target], edges + [edge]))
        if not next_states:
            return {"found": False, "candidate_ids": [], "first_unreachable_waypoint": waypoint}
        # Keep all equal-cost states; this is intentionally an exact baseline.
        states = sorted(next_states, key=lambda item: deterministic_path_key(*item))
        best_by_terminal: dict[str, tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]] = {}
        for state in states:
            terminal = str(state[0][-1].get("stable_node_id", state[0][-1].get("ik_candidate_id")))
            current = best_by_terminal.get(terminal)
            if current is None or deterministic_path_key(*state) < deterministic_path_key(*current):
                best_by_terminal[terminal] = state
        states = list(best_by_terminal.values())
    path, edges = min(states, key=lambda item: deterministic_path_key(*item))
    return {"found": True, "candidate_ids": [str(n.get("stable_node_id", n.get("ik_candidate_id"))) for n in path], "task_pose_candidate_ids": [str(n["task_pose_stable_id"]) for n in path], "selected_nodes": [dict(n) for n in path], "selected_edges": [dict(e) for e in edges], "total_cost": deterministic_path_key(path, edges)[0], "first_unreachable_waypoint": None}
