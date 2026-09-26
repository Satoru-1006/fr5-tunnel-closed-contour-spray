"""Pure helpers for localized post-TOTG TCP-position reprojection."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


TCP_POSITION_LIMIT_M = 0.001
ELIGIBLE_SEGMENTS = frozenset({2, 3, 4})


def repair_eligible(*, spray_state: str, segment_id: int, tcp_position_error_m: float) -> bool:
    """Return the frozen H12-R7 failed-samples-only repair predicate."""
    return bool(
        spray_state == "SPRAY_ON"
        and int(segment_id) in ELIGIBLE_SEGMENTS
        and float(tcp_position_error_m) > TCP_POSITION_LIMIT_M
    )


def bounded_seeds(positions: np.ndarray, index: int) -> list[tuple[str, np.ndarray]]:
    """Current, previous-biased and next-biased deterministic local IK seeds."""
    q = np.asarray(positions, dtype=float)
    current = q[index].copy()
    seeds: list[tuple[str, np.ndarray]] = [("current", current)]
    if index > 0:
        seeds.append(("previous_neighbor_biased", 0.75 * current + 0.25 * q[index - 1]))
    if index + 1 < len(q):
        seeds.append(("next_neighbor_biased", 0.75 * current + 0.25 * q[index + 1]))
    return seeds


def candidate_score(q_before: Sequence[float], q_after: Sequence[float], q_previous: Sequence[float] | None, q_next: Sequence[float] | None) -> tuple[float, float, tuple[float, ...]]:
    """Rank valid solutions by correction first and local continuity second."""
    before = np.asarray(q_before, dtype=float)
    after = np.asarray(q_after, dtype=float)
    correction = float(np.linalg.norm(after - before))
    continuity = 0.0
    if q_previous is not None:
        continuity += float(np.linalg.norm(after - np.asarray(q_previous, dtype=float)))
    if q_next is not None:
        continuity += float(np.linalg.norm(np.asarray(q_next, dtype=float) - after))
    return correction, continuity, tuple(float(value) for value in np.round(after, 15))


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def immutable_sample_hash(positions: np.ndarray, repaired_indices: Iterable[int]) -> str:
    repaired = {int(value) for value in repaired_indices}
    rows = [(index, np.asarray(q, dtype=float).tolist()) for index, q in enumerate(np.asarray(positions)) if index not in repaired]
    return canonical_hash(rows)

