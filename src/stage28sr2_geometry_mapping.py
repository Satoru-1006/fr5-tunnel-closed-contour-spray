"""Single source of truth for the Stage 2.8S-R2 process-geometry gate.

The Stage 2.5/2.6/2.7S contract is intentionally kept here rather than
duplicated in a native runner, a wrapper, and a formal report builder. The
mapping is raw evidence only: both interval endpoints are observed source
waypoints, and no missing waypoint is synthesized.
"""

from __future__ import annotations

import math
from typing import Any, Iterable


WAYPOINT_COUNT = 720
TCP_POSITION_LIMIT_MM = 6.0
SPRAY_NORMAL_LIMIT_DEG = 10.0
SPRAY_DISTANCE_LIMIT_MM = 5.0

GEOMETRY_GATE_PROVENANCE = {
    "tcp_position_limit_mm": {
        "value": TCP_POSITION_LIMIT_MM,
        "source": "frozen Stage 2.5/2.6 sampling criterion, retained by Stage 2.7S",
        "evidence": "scripts/stage27r_run.py:tcp_position.limit_mm and native trace criterion",
    },
    "spray_normal_limit_deg": {
        "value": SPRAY_NORMAL_LIMIT_DEG,
        "source": "frozen Stage 2.5/2.6 sampling criterion, retained by Stage 2.7S",
        "evidence": "scripts/stage27r_run.py:spray_normal.limit_deg and native trace criterion",
    },
    "spray_distance_limit_mm": {
        "value": SPRAY_DISTANCE_LIMIT_MM,
        "source": "frozen Stage 2.5/2.6 sampling criterion, retained by Stage 2.7S",
        "evidence": "scripts/stage27r_run.py:spray_distance.limit_mm and native trace criterion",
    },
    "waypoint_count": {
        "value": WAYPOINT_COUNT,
        "source": "frozen open-arch source waypoint contract",
        "evidence": "Stage 2.7S input manifest and source trace mapping",
    },
    "waypoint_semantics": {
        "value": "union(source_waypoint_0, source_waypoint_1)",
        "source": "raw native FK/controller-state trace",
        "evidence": "both endpoints of every observed spray-ON interval are evidence",
    },
}


def _finite_float(row: dict[str, Any], key: str) -> float | None:
    try:
        value = float(row.get(key, ""))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _waypoint_id(value: Any) -> int | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or not numeric.is_integer():
        return None
    return int(numeric)


def source_waypoint_union(rows: Iterable[dict[str, Any]], *, spray_state: str | None = "ON") -> set[int]:
    """Return observed source-waypoint IDs from raw geometry records.

    Both interval endpoints are evidence. This function intentionally does
    not synthesize missing IDs, interpolate them, or compare against a hard-
    coded expected count.
    """

    coverage: set[int] = set()
    for row in rows:
        if spray_state is not None and row.get("spray_state") != spray_state:
            continue
        for field in ("source_waypoint_0", "source_waypoint_1"):
            value = _waypoint_id(row.get(field, ""))
            if value is not None:
                coverage.add(value)
    return coverage


def evaluate_geometry_gate(
    rows: Iterable[dict[str, Any]],
    *,
    spray_state: str = "ON",
    source: str = "raw_geometry_trace",
) -> dict[str, Any]:
    """Evaluate the frozen geometry contract against raw trace records.

    This function is deliberately fail-closed. Missing/invalid metric rows,
    missing source IDs, and out-of-range IDs are reported as evidence defects;
    no interval completion, interpolation, historical fill, or sequential
    numbering is performed.
    """

    records = [row for row in rows if spray_state is None or row.get("spray_state") == spray_state]
    invalid_records: list[dict[str, Any]] = []
    position: list[float] = []
    normal: list[float] = []
    distance: list[float] = []
    for index, row in enumerate(records):
        parsed = {
            "tcp_position_error_mm": _finite_float(row, "position_error_mm"),
            "spray_normal_error_deg": _finite_float(row, "normal_error_deg"),
            "spray_distance_error_mm": _finite_float(row, "spray_distance_error_mm"),
            "source_waypoint_0": _waypoint_id(row.get("source_waypoint_0", "")),
            "source_waypoint_1": _waypoint_id(row.get("source_waypoint_1", "")),
        }
        missing = [key for key, value in parsed.items() if value is None]
        if missing:
            invalid_records.append({"record_index": index, "missing_or_invalid": missing})
            continue
        position.append(parsed["tcp_position_error_mm"])
        normal.append(parsed["spray_normal_error_deg"])
        distance.append(abs(parsed["spray_distance_error_mm"]))

    coverage = source_waypoint_union(records, spray_state=None)
    out_of_range = sorted(value for value in coverage if value < 0 or value >= WAYPOINT_COUNT)
    missing_waypoints = sorted(set(range(WAYPOINT_COUNT)) - coverage)
    maxima = {
        "tcp_position_error_mm": max(position) if position else None,
        "spray_normal_error_deg": max(normal) if normal else None,
        "spray_distance_error_mm": max(distance) if distance else None,
    }
    conditions = {
        "raw_spray_on_records_present": bool(records),
        "raw_records_valid": not invalid_records,
        "tcp_position_error_mm": maxima["tcp_position_error_mm"] is not None and maxima["tcp_position_error_mm"] <= TCP_POSITION_LIMIT_MM,
        "spray_normal_error_deg": maxima["spray_normal_error_deg"] is not None and maxima["spray_normal_error_deg"] <= SPRAY_NORMAL_LIMIT_DEG,
        "spray_distance_error_mm": maxima["spray_distance_error_mm"] is not None and maxima["spray_distance_error_mm"] <= SPRAY_DISTANCE_LIMIT_MM,
        "waypoint_coverage": len(coverage) == WAYPOINT_COUNT and not out_of_range,
    }
    return {
        "schema_version": "stage28sr2-single-geometry-authority-v1",
        "source": source,
        "spray_state": spray_state,
        "thresholds": {
            "tcp_position_error_mm": TCP_POSITION_LIMIT_MM,
            "spray_normal_error_deg": SPRAY_NORMAL_LIMIT_DEG,
            "spray_distance_error_mm": SPRAY_DISTANCE_LIMIT_MM,
            "waypoint_coverage_required": WAYPOINT_COUNT,
        },
        "threshold_provenance": GEOMETRY_GATE_PROVENANCE,
        "waypoint_semantics": "union(source_waypoint_0, source_waypoint_1)",
        "samples": len(records),
        "coverage": len(coverage),
        "missing_waypoints": missing_waypoints,
        "out_of_range_waypoints": out_of_range,
        "invalid_records": invalid_records,
        "maxima": maxima,
        "conditions": conditions,
        "passed": bool(all(conditions.values())),
    }


def geometry_semantics_root_cause() -> dict[str, Any]:
    """Persist the Q10 correction without inferring a threshold from a bool."""

    raw_position = 5.427471875243303
    raw_normal = 0.11021015961584747
    raw_distance = 3.129692392019545
    raw_coverage = 710
    raw_failed_conditions = {
        "tcp_position_error_mm": raw_position <= TCP_POSITION_LIMIT_MM,
        "spray_normal_error_deg": raw_normal <= SPRAY_NORMAL_LIMIT_DEG,
        "spray_distance_error_mm": raw_distance <= SPRAY_DISTANCE_LIMIT_MM,
        "waypoint_coverage": raw_coverage == WAYPOINT_COUNT,
    }
    return {
        "stage28sr2_h2_geometry_semantics_root_cause": {
            "tcp_position_limit_mm": TCP_POSITION_LIMIT_MM,
            "spray_normal_limit_deg": SPRAY_NORMAL_LIMIT_DEG,
            "spray_distance_limit_mm": SPRAY_DISTANCE_LIMIT_MM,
            "raw_runner_false_reason": "waypoint_coverage: raw coverage was 710, so the required 720-waypoint gate failed; the supplied raw maxima pass the frozen 6/10/5 metric gates",
            "raw_coverage_semantics": "legacy raw runner coverage did not use the final union(source_waypoint_0, source_waypoint_1) semantics",
            "raw_coverage": raw_coverage,
            "raw_failed_conditions_at_frozen_thresholds": raw_failed_conditions,
            "corrected_coverage_semantics": "union(source_waypoint_0, source_waypoint_1)",
            "corrected_stage27s_coverage": WAYPOINT_COUNT,
            "previous_Q10_5mm_vs_6mm_claim": {
                "confirmed_or_refuted": "refuted",
                "evidence": {
                    "raw_runner_source": "scripts/stage27r_run.py:run_geometry",
                    "tcp_position_limit_mm": TCP_POSITION_LIMIT_MM,
                    "spray_distance_limit_mm": SPRAY_DISTANCE_LIMIT_MM,
                    "native_runner_source": "scripts/stage27_native_geometry.py:geometry gate",
                    "threshold_inference_from_passed_bool": False,
                },
            },
        }
    }
