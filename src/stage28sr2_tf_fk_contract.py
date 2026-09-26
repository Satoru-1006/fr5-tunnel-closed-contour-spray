"""Pure preflight checks for the formal TF/FK timestamp contract.

The runtime validator remains the authority for actual ``tf2_ros.Buffer`` and
MoveIt native FK. These helpers make the pairing rules regression-testable
without starting ROS or movement.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable


def stamp_key(row: dict[str, Any]) -> tuple[int, int]:
    stamp = row["header_stamp"]
    return int(stamp["sec"]), int(stamp["nanosec"])


def exact_tf_jointstate_pairing(
    tf_rows: Iterable[dict[str, Any]],
    joint_rows: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Pair only on exact ROS header sec/nanosec, never capture time/index."""

    joint_by_stamp: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in joint_rows:
        joint_by_stamp[stamp_key(row)].append(row)
    matches = 0
    unmatched = 0
    ambiguous = 0
    for row in tf_rows:
        candidates = joint_by_stamp.get(stamp_key(row), [])
        if len(candidates) == 1:
            matches += 1
        elif not candidates:
            unmatched += 1
        else:
            ambiguous += 1
    total = matches + unmatched + ambiguous
    return {
        "key": "exact ROS header timestamp (sec,nanosec)",
        "capture_time_nearest": False,
        "array_index": False,
        "zip_arrays": False,
        "matches": matches,
        "unmatched": unmatched,
        "ambiguous": ambiguous,
        "total": total,
        "passed": bool(total and unmatched == 0 and ambiguous == 0),
    }


def quaternion_distance_deg(q1: Iterable[float], q2: Iterable[float]) -> float:
    """Return the q/-q-safe angular distance in degrees."""

    a = [float(value) for value in q1]
    b = [float(value) for value in q2]
    norm_a = math.sqrt(sum(value * value for value in a))
    norm_b = math.sqrt(sum(value * value for value in b))
    if norm_a <= 0.0 or norm_b <= 0.0:
        return math.inf
    dot = abs(sum(x * y for x, y in zip(a, b)) / (norm_a * norm_b))
    return math.degrees(2.0 * math.acos(min(1.0, max(0.0, dot))))


def tf_fk_threshold_contract() -> dict[str, Any]:
    return {
        "pairing": {
            "key": "exact ROS header timestamp",
            "unmatched_required": 0,
            "ambiguous_required": 0,
            "zip_arrays_forbidden": True,
            "capture_time_nearest_forbidden": True,
            "array_index_forbidden": True,
        },
        "thresholds": {
            "wrist3_rotation_limit_deg": 0.1,
            "spray_tcp_translation_limit_mm": 0.001,
            "spray_tcp_rotation_limit_deg": 0.1,
        },
        "runtime_authorities": ["tf2_ros.Buffer", "explicit requested timestamp", "MoveIt native FK", "q/-q safe quaternion metric"],
    }
