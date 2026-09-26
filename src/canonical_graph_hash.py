"""Canonical artifact and semantic hashes for Stage 1.9."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from src.deterministic_ik_candidates import canonical_json, quantize, sha256_canonical


NOISY_FIELDS = {"timestamp", "timestamp_utc", "created_time_utc", "elapsed_time", "elapsed_seconds", "pid", "log_id", "discovery_order", "seed_elapsed_time", "stdout", "stderr", "parquet_metadata", "moveitpy_exit_status"}


def artifact_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clean(value: Any, *, numeric_step: float = 1.0e-9, key: str = "") -> Any:
    if key in NOISY_FIELDS:
        return None
    if isinstance(value, np.ndarray):
        return [_clean(x, numeric_step=numeric_step) for x in value.tolist()]
    if isinstance(value, Mapping):
        return {str(k): _clean(v, numeric_step=numeric_step, key=str(k)) for k, v in sorted(value.items(), key=lambda item: str(item[0])) if str(k) not in NOISY_FIELDS}
    if isinstance(value, (list, tuple)):
        return [_clean(x, numeric_step=numeric_step) for x in value]
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)) and not isinstance(value, bool):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("NaN/Infinity is forbidden in semantic hash")
        return quantize(result, numeric_step)
    return value


def canonical_records(records: Iterable[Mapping[str, Any]], *, sort_fields: Sequence[str], numeric_step: float = 1.0e-9) -> list[dict[str, Any]]:
    cleaned = [_clean(dict(record), numeric_step=numeric_step) for record in records]
    return sorted(cleaned, key=lambda record: tuple(str(record.get(field, "")) for field in sort_fields) + (canonical_json(record),))


def semantic_hash(records: Iterable[Mapping[str, Any]], *, sort_fields: Sequence[str], numeric_step: float = 1.0e-9) -> str:
    return sha256_canonical(canonical_records(records, sort_fields=sort_fields, numeric_step=numeric_step))


def graph_hashes(*, task_pose_records: Sequence[Mapping[str, Any]], node_records: Sequence[Mapping[str, Any]], edge_records: Sequence[Mapping[str, Any]], selected_task_pose_records: Sequence[Mapping[str, Any]], selected_node_records: Sequence[Mapping[str, Any]], selected_joint_records: Sequence[Mapping[str, Any]], repair_waypoint_records: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    task_hash = semantic_hash(task_pose_records, sort_fields=("waypoint_id", "task_pose_stable_id"))
    node_hash = semantic_hash(node_records, sort_fields=("waypoint_id", "task_pose_stable_id", "stable_node_id"))
    edge_hash = semantic_hash(edge_records, sort_fields=("source_stable_node_id", "target_stable_node_id"))
    return {
        "task_pose_candidate_semantic_hash": task_hash,
        "ik_node_semantic_hash": node_hash,
        "transition_edge_semantic_hash": edge_hash,
        "candidate_graph_semantic_hash": sha256_canonical({"task_pose": task_hash, "nodes": node_hash, "edges": edge_hash}),
        "selected_task_pose_sequence_hash": semantic_hash(selected_task_pose_records, sort_fields=("waypoint_id", "task_pose_stable_id")),
        "selected_ik_sequence_hash": semantic_hash(selected_node_records, sort_fields=("waypoint_id", "stable_node_id")),
        "repair_waypoint_set_hash": semantic_hash(repair_waypoint_records, sort_fields=("waypoint_id",)),
        "selected_joint_path_semantic_hash": semantic_hash(selected_joint_records, sort_fields=("waypoint_id", "stable_node_id")),
    }


def artifact_hash_map(paths: Iterable[str | Path], *, root: str | Path) -> dict[str, str]:
    root_path = Path(root)
    return {str(Path(path).relative_to(root_path).as_posix()): artifact_sha256(path) for path in sorted((Path(path) for path in paths), key=lambda p: str(p))}
