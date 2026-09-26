"""Independent freshness checks for FAIRINO ``ROBOT_STATE_PKG`` records.

The H1 gate is deliberately recomputed from the raw CSV after the probe has
closed RPC.  A successful SDK call is not itself evidence of a fresh frame:
the raw frame counter must progress modulo 256 and the host monotonic clock
must advance strictly.  Joint position constancy is not used as a freshness
signal because a stationary robot can still publish fresh telemetry.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


FRAME_HEAD = 0x5A5A
JOINT_KEYS = ("j1_deg", "j2_deg", "j3_deg", "j4_deg", "j5_deg", "j6_deg")
RAW_REQUIRED_FIELDS = (
    "sample_index",
    "host_monotonic_ns",
    "host_wall_time",
    "return_code",
    "frame_head",
    "frame_cnt",
    "program_state",
    "robot_state",
    "robot_mode",
    "main_code",
    "sub_code",
    *JOINT_KEYS,
)
_RAW_INT_FIELDS = {
    "sample_index",
    "host_monotonic_ns",
    "return_code",
    "frame_head",
    "frame_cnt",
    "program_state",
    "robot_state",
    "robot_mode",
    "main_code",
    "sub_code",
}


def _present(value: Any) -> bool:
    return value is not None and not (isinstance(value, str) and not value.strip())


def _finite(value: Any) -> bool:
    if isinstance(value, bool) or not _present(value):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _integer(value: Any) -> int | None:
    """Parse an integer without truncating floats or accepting booleans."""

    if isinstance(value, bool) or not _present(value):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip()
        try:
            return int(text, 10)
        except ValueError:
            return None
    return None


def _uint8(value: Any) -> int | None:
    parsed = _integer(value)
    return parsed if parsed is not None and 0 <= parsed <= 255 else None


def verify_frame_counter(
    frame_counts: Sequence[Any],
    *,
    sample_count_min: int = 100,
) -> dict[str, Any]:
    """Verify uint8 frame progression without mistaking 255 -> 0 for rollback.

    A modulo delta of 1..127 is accepted as forward progress.  A delta above
    1 records a dropped-frame observation.  A delta above 127 is rejected,
    except for the protocol's high-to-low uint8 rollover (for example
    255 -> 0).  Missing or malformed counters are hard failures.
    """

    raw_counts = list(frame_counts)
    parsed = [_uint8(value) for value in raw_counts]
    errors: list[dict[str, Any]] = []
    for index, (raw, value) in enumerate(zip(raw_counts, parsed)):
        if value is None:
            errors.append(
                {
                    "index": index,
                    "kind": "frame_cnt_missing_or_invalid_uint8",
                    "value": raw,
                }
            )

    dropped_frames = False
    for index, (previous, current) in enumerate(zip(parsed, parsed[1:]), start=1):
        if previous is None or current is None:
            continue
        delta = (current - previous) % 256
        if delta == 0:
            errors.append(
                {
                    "index": index,
                    "kind": "frame_counter_constant",
                    "previous": previous,
                    "current": current,
                }
            )
        elif delta > 127 and not (previous >= 240 and current <= 15):
            errors.append(
                {
                    "index": index,
                    "kind": "frame_counter_reverse_or_unexplained_jump",
                    "previous": previous,
                    "current": current,
                    "modulo_delta": delta,
                }
            )
        elif delta > 1:
            dropped_frames = True

    all_valid = bool(raw_counts) and all(value is not None for value in parsed)
    sequence_valid = all_valid and not errors
    return {
        "frame_cnt_present": bool(raw_counts) and all(value is not None for value in parsed),
        "frame_cnt_uint8_valid": all_valid,
        "frame_cnt_sequence_valid_mod_256": sequence_valid,
        "fresh": sequence_valid,
        "fresh_samples_proven": sequence_valid and len(raw_counts) >= sample_count_min,
        "sample_count": len(raw_counts),
        "sample_count_min": sample_count_min,
        "sample_count_gate": len(raw_counts) >= sample_count_min,
        "dropped_frames_detected": dropped_frames,
        "anomalies": errors,
        "modulo": 256,
        "wrap_rule": "expected_next=(previous+1)%256; high-to-low uint8 wrap is valid",
    }


def _parsed_return_codes(rows: Sequence[Mapping[str, Any]]) -> tuple[bool, bool]:
    values = [_integer(row.get("return_code")) for row in rows]
    present = bool(rows) and all(value is not None for value in values)
    return present, present and all(value == 0 for value in values)


def _parsed_frame_heads(rows: Sequence[Mapping[str, Any]]) -> tuple[bool, bool]:
    values = [_integer(row.get("frame_head")) for row in rows]
    present = bool(rows) and all(value is not None for value in values)
    return present, present and all(value == FRAME_HEAD for value in values)


def _parsed_timestamps(rows: Sequence[Mapping[str, Any]]) -> tuple[bool, bool]:
    values = [_integer(row.get("host_monotonic_ns")) for row in rows]
    present = bool(rows) and all(value is not None and value >= 0 for value in values)
    increasing = present and all(current > previous for previous, current in zip(values, values[1:]))
    return present, increasing


def verify_state_samples(
    samples: Iterable[Mapping[str, Any]],
    *,
    sample_count_min: int = 100,
) -> dict[str, Any]:
    """Verify normalized state rows using all H1 formal freshness gates."""

    rows = list(samples)
    return_codes_present, return_codes_zero = _parsed_return_codes(rows)
    frame_head_present, frame_head_valid = _parsed_frame_heads(rows)
    timestamps_present, timestamps_monotonic = _parsed_timestamps(rows)
    frame_counter = verify_frame_counter(
        [row.get("frame_cnt") for row in rows],
        sample_count_min=sample_count_min,
    )
    finite_joints = bool(rows) and all(
        _finite(row.get(key)) for row in rows for key in JOINT_KEYS
    )
    freshness_anomalies = list(frame_counter["anomalies"])
    if not return_codes_present:
        freshness_anomalies.append({"kind": "return_code_missing_or_invalid"})
    elif not return_codes_zero:
        freshness_anomalies.append({"kind": "GetRobotRealTimeState_return_code_nonzero"})
    if not frame_head_present:
        freshness_anomalies.append({"kind": "frame_head_missing_or_invalid"})
    if not timestamps_present:
        freshness_anomalies.append({"kind": "host_monotonic_timestamp_missing_or_invalid"})
    elif not timestamps_monotonic:
        freshness_anomalies.append({"kind": "host_monotonic_timestamp_nonmonotonic"})
    if not finite_joints:
        freshness_anomalies.append({"kind": "joint_value_missing_or_nonfinite"})

    return {
        "sample_count": len(rows),
        "sample_count_min": sample_count_min,
        "sample_count_gate": len(rows) >= sample_count_min,
        "all_GetRobotRealTimeState_return_codes_zero": return_codes_zero,
        "return_code_present_all_samples": return_codes_present,
        "frame_head_expected": FRAME_HEAD,
        "frame_head_present_all_samples": frame_head_present,
        "frame_head_valid": frame_head_valid,
        "frame_cnt_present": frame_counter["frame_cnt_present"],
        "frame_cnt_uint8_valid": frame_counter["frame_cnt_uint8_valid"],
        "frame_cnt_sequence_valid_mod_256": frame_counter["frame_cnt_sequence_valid_mod_256"],
        "dropped_frames_detected": frame_counter["dropped_frames_detected"],
        "host_monotonic_timestamp_present": timestamps_present,
        "timestamps_host_monotonic": timestamps_monotonic,
        "all_6_joint_values_present": finite_joints,
        "joint_values_finite": finite_joints,
        "fresh_samples_proven": (
            len(rows) >= sample_count_min
            and return_codes_zero
            and frame_head_valid
            and frame_counter["frame_cnt_sequence_valid_mod_256"]
            and finite_joints
            and timestamps_monotonic
        ),
        "freshness_anomalies": freshness_anomalies,
        "position_constancy_is_not_stale": True,
        "counter_detail": frame_counter,
    }


def _parse_raw_row(row_number: int, row: Mapping[str, str]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    normalized: dict[str, Any] = {}
    for field in RAW_REQUIRED_FIELDS:
        value = row.get(field)
        if not _present(value):
            return None, {"row": row_number, "kind": "missing_field", "field": field}
        if field in _RAW_INT_FIELDS:
            parsed = _integer(value)
            if parsed is None:
                return None, {"row": row_number, "kind": "invalid_integer", "field": field, "value": value}
            normalized[field] = parsed
        elif field == "host_wall_time":
            normalized[field] = str(value).strip()
        else:
            try:
                parsed_float = float(value)
            except (TypeError, ValueError, OverflowError):
                return None, {"row": row_number, "kind": "invalid_float", "field": field, "value": value}
            normalized[field] = parsed_float
    return normalized, None


def verify_raw_csv(
    path: str | Path,
    *,
    sample_count_min: int = 100,
) -> dict[str, Any]:
    """Recompute H1 freshness from the immutable raw CSV only."""

    csv_path = Path(path)
    base: dict[str, Any] = {
        "schema_version": "stage28ah1-live-state-freshness-v2",
        "raw_csv_path": str(csv_path),
        "sample_count_requested": sample_count_min,
        "sample_count_min": sample_count_min,
        "csv_header_valid": False,
        "raw_rows_parseable": False,
        "csv_parse_errors": [],
    }
    if not csv_path.is_file():
        base["csv_parse_errors"] = [{"kind": "raw_csv_missing"}]
        base.update(verify_state_samples([], sample_count_min=sample_count_min))
        base["fresh_samples_proven"] = False
        return base

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing_fields = [field for field in RAW_REQUIRED_FIELDS if field not in fieldnames]
        base["csv_header"] = fieldnames
        base["missing_required_fields"] = missing_fields
        base["csv_header_valid"] = not missing_fields
        if missing_fields:
            base["csv_parse_errors"] = [{"kind": "missing_required_header", "fields": missing_fields}]
            base.update(verify_state_samples([], sample_count_min=sample_count_min))
            base["fresh_samples_proven"] = False
            return base

        rows: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        for row_number, row in enumerate(reader, start=2):
            normalized, error = _parse_raw_row(row_number, row)
            if error is not None:
                errors.append(error)
            elif normalized is not None:
                rows.append(normalized)

    base["csv_parse_errors"] = errors
    base["raw_rows_parseable"] = not errors
    base.update(verify_state_samples(rows, sample_count_min=sample_count_min))
    base["fresh_samples_proven"] = bool(base["raw_rows_parseable"] and base["fresh_samples_proven"])
    return base
