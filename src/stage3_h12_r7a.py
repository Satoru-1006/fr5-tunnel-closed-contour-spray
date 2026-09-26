"""Deterministic helpers for the H12-R7A spray-process audit."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any, Iterable, Mapping

import numpy as np


CONSTRAINTS = (
    ("STANDOFF", "standoff_error_m", "standoff_limit_m"),
    ("SURFACE_NORMAL", "normal_deviation_deg", "normal_limit_deg"),
    ("TCP_POSITION", "tcp_position_error_m", "tcp_position_limit_m"),
    ("TCP_ORIENTATION", "tcp_orientation_error_deg", "tcp_orientation_limit_deg"),
)


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def failed_constraints(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    result = []
    for name, value_key, limit_key in CONSTRAINTS:
        value, limit = float(row[value_key]), float(row[limit_key])
        if value > limit:
            result.append({"constraint_name": name, "observed_value": value, "limit_value": limit, "absolute_exceedance": value - limit, "relative_exceedance": (value - limit) / limit})
    return result


def classify_sample(row: Mapping[str, Any]) -> str:
    failed = failed_constraints(row)
    if not failed:
        raise ValueError("row is not a process violation")
    if len(failed) != 1:
        return "COMPOUND:" + "+".join(item["constraint_name"] for item in failed)
    return failed[0]["constraint_name"]


def percentile(values: Iterable[float], quantile: float) -> float | None:
    array = np.asarray(list(values), dtype=float)
    return None if not len(array) else float(np.percentile(array, quantile, method="linear"))


def severity(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, float | None]]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        for failure in failed_constraints(row):
            grouped.setdefault(failure["constraint_name"], []).append(float(failure["absolute_exceedance"]))
    return {name: {"MAX_EXCEEDANCE": max(values), "P99_EXCEEDANCE": percentile(values, 99), "P95_EXCEEDANCE": percentile(values, 95), "MEDIAN_EXCEEDANCE": percentile(values, 50), "MIN_EXCEEDANCE": min(values)} for name, values in sorted(grouped.items())}


def distribution(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    records = list(rows)
    classes = Counter(classify_sample(row) for row in records)
    return {
        "RAW_VIOLATION_RECORDS": len(records),
        "SPRAY_CONSTRAINT_CLASSES": dict(sorted(classes.items())),
        "class_count_sum": sum(classes.values()),
        "severity": severity(records),
    }


def exact_reproduction(observed: int, expected: int = 2304) -> bool:
    return int(observed) == int(expected)


def lineage_breakdown(pre_rows: Iterable[Mapping[str, Any]], post_rows: Iterable[Mapping[str, Any]], *, tolerance: float = 1e-12) -> dict[str, int]:
    """Match post-TOTG violations to exact pre-TOTG joint states per unit."""
    pre_by_unit: dict[str, list[np.ndarray]] = {}
    for row in pre_rows:
        pre_by_unit.setdefault(str(row["unit_id"]), []).append(np.asarray(row["joint_values"], dtype=float))
    post = list(post_rows)
    persistent = 0
    for row in post:
        q = np.asarray(row["joint_values"], dtype=float)
        if any(float(np.max(np.abs(q - candidate))) <= tolerance for candidate in pre_by_unit.get(str(row["unit_id"]), [])):
            persistent += 1
    return {"PRE_EXISTING_PERSISTING": persistent, "TOTG_INDUCED": len(post) - persistent, "POST_TOTG_TOTAL": len(post)}


def first_blocker(*, reproduced: bool, inventory_complete: bool, lineage_complete: bool, immutable: bool, replay: bool, tests: bool, native_regression: bool) -> str:
    checks = (
        (immutable, "upstream_immutability_violation"),
        (reproduced, "h12_r6_spray_violation_reproduction_mismatch"),
        (inventory_complete, "spray_violation_inventory_incomplete"),
        (lineage_complete, "stage_lineage_incomplete"),
        (replay, "nondeterministic_replay"),
        (tests, "tests_failed"),
        (native_regression, "native_validation_regression"),
    )
    return next((reason for passed, reason in checks if not passed), "none")
