"""Streaming H10 loader used only by H12-R runtime.

It preserves the H11 dataset contract while using the installed ujson parser
when available.  No values or fields are regenerated; the H10 JSONL remains
the sole source of sample data.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    import ujson as _json
except Exception:  # pragma: no cover
    import json as _json

from src.stage3_h11_dataset import H11ContractError, H11Dataset, Segment, JOINTS, load_json


def _six(row: dict[str, Any], field: str) -> list[float]:
    value = row.get(field)
    if not isinstance(value, list) or len(value) != JOINTS:
        raise H11ContractError(f"invalid_h10_vector:{field}")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise H11ContractError(f"nonfinite_h10_vector:{field}")
    return result


def load_h10_segments_fast(h10_root: Path) -> H11Dataset:
    split_manifest = load_json(h10_root / "dataset_split_manifest.json")
    split_map = {str(key): str(value) for key, value in split_manifest.get("group_to_role", {}).items()}
    if set(split_map.values()) != {"TRAIN", "VALIDATION", "TEST", "GENERALIZATION"}:
        raise H11ContractError("frozen_split_roles_incomplete")
    groups: dict[tuple[str, str, int, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    source_count = 0
    with (h10_root / "trajectory_samples.jsonl").open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            row = _json.loads(raw)
            source_count += 1
            family = str(row.get("trajectory_family_id"))
            role = split_map.get(family)
            if role is None or str(row.get("split_role")) != role:
                raise H11ContractError(f"dataset_split_contamination:line_{line_number}")
            if row.get("actual_tcp_position") is not None or row.get("actual_tcp_orientation") is not None:
                raise H11ContractError("unavailable_tcp_signal_not_null")
            if row.get("collision_method") != "adaptive_discrete_interpolation":
                raise H11ContractError("collision_method_mismatch")
            key = (family, str(row.get("trajectory_id")), int(row.get("segment_id")), int(row.get("segment_order")), str(row.get("spray_state")), str(row.get("primitive_id")))
            groups[key].append(row)
    segments: list[Segment] = []
    for key, rows in groups.items():
        rows.sort(key=lambda item: (int(item["sample_index"]), float(item["trajectory_time"])))
        times = np.asarray([float(row["trajectory_time"]) for row in rows], dtype=np.float64)
        if len(times) < 2 or np.any(np.diff(times) <= 0.0):
            raise H11ContractError(f"segment_time_not_strictly_monotonic:{key}")
        segments.append(Segment(
            family_id=key[0], trajectory_id=key[1], segment_id=key[2], segment_order=key[3], spray_state=key[4], primitive_id=key[5],
            times=times,
            positions=np.asarray([_six(row, "planned_joint_position") for row in rows], dtype=np.float64),
            velocities=np.asarray([_six(row, "planned_joint_velocity") for row in rows], dtype=np.float64),
            accelerations=np.asarray([_six(row, "planned_joint_acceleration") for row in rows], dtype=np.float64),
            sample_indices=np.asarray([int(row["sample_index"]) for row in rows], dtype=np.int64),
            controlled_stop_count=max(int(row.get("controlled_stop_count", 0) or 0) for row in rows),
        ))
    segments.sort(key=lambda item: (item.family_id, item.segment_order, item.segment_id))
    return H11Dataset(h10_root, split_map, segments, source_count, len(split_map), len(segments))

