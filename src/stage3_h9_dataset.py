"""Stage 3 H9 dataset contract, deterministic extraction, and evaluator.

This module intentionally has no ROS dependency.  H9 consumes the immutable
structured H8-R feedback and the certified H7 trajectory/validation records.
It never writes to an upstream evidence directory and never treats a replay
as an independent demonstration.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


STAGE_ID = "STAGE_3_H9"
DATASET_SCHEMA_VERSION = "stage3_h9_dataset_schema_v1"
JOINT_ORDER = ["j1", "j2", "j3", "j4", "j5", "j6"]
COLLISION_METHOD = "adaptive_discrete_interpolation"
EXPECTED_MAIN_FEEDBACK_COUNT = 4172
H7_SHA256 = "b1ede9467efd2b5a9b037cb4c5f7dd3bf557d3ca55ca3f37e616d97dba5a2983"
FJT_BOUNDARY_GAP_S = 0.01

# These are the execution limits used by the H8-R actual-state validator.
# They are deliberately kept separate from the broader H7 planning limits.
EXECUTION_POSITION_LOWER = [-3.0543, -4.6251, -2.8274, -4.6251, -3.0543, -3.0543]
EXECUTION_POSITION_UPPER = [3.0543, 1.4835, 2.8274, 1.4835, 3.0543, 3.0543]
EXECUTION_VMAX = [0.4725, 0.4725, 0.4725, 0.48, 0.48, 0.48]
EXECUTION_AMAX = [0.105] * 6
EXECUTION_JMAX = [8.0] * 6


class DatasetValidationError(ValueError):
    """Raised when an authoritative input cannot satisfy the H9 contract."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical_value(value[key]) for key in sorted(value, key=lambda item: str(item))}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DatasetValidationError("non-finite value is not canonicalizable")
        # Normalize negative zero and normalize insignificant binary display
        # noise without changing the source values used by evaluation.
        return 0.0 if value == 0.0 else float(format(value, ".17g"))
    return value


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(_canonical_value(value), ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def semantic_sha256(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(canonical_bytes(dict(row)).decode("utf-8"))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def relpath(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def duration_to_seconds(value: Mapping[str, Any] | None) -> float | None:
    if not isinstance(value, Mapping):
        return None
    sec = value.get("sec")
    nanosec = value.get("nanosec")
    if not isinstance(sec, (int, float)) or not isinstance(nanosec, (int, float)):
        return None
    result = float(sec) + float(nanosec) / 1_000_000_000.0
    return result if math.isfinite(result) else None


def _finite_vector(value: Any, length: int = 6) -> bool:
    return (
        isinstance(value, list)
        and len(value) == length
        and all(isinstance(item, (int, float)) and math.isfinite(float(item)) for item in value)
    )


def global_message_times(rows: Sequence[Mapping[str, Any]]) -> tuple[list[float], list[dict[str, Any]]]:
    """Reuse the H8-R H7-local-clock to FJT-global-clock adapter exactly."""
    times: list[float] = []
    offsets: list[dict[str, Any]] = []
    previous_block: tuple[Any, Any] | None = None
    previous_global = -math.inf
    offset = 0.0
    for index, row in enumerate(rows):
        block = (row.get("segment_id"), row.get("primitive_id"))
        local = float(row["time_from_start_s"])
        if index and block != previous_block:
            offset = previous_global + FJT_BOUNDARY_GAP_S
            offsets.append(
                {
                    "row_index": index,
                    "segment_id": row.get("segment_id"),
                    "primitive_id": row.get("primitive_id"),
                    "offset_s": offset,
                }
            )
        current = offset + local
        if current <= previous_global:
            current = previous_global + FJT_BOUNDARY_GAP_S
        times.append(current)
        previous_global = current
        previous_block = block
    return times, offsets


def build_segments(h7_rows: Sequence[Mapping[str, Any]], global_times: Sequence[float]) -> list[dict[str, Any]]:
    """Build logical segment/primitive records without renaming H7 semantics."""
    segments: list[dict[str, Any]] = []
    start = 0
    previous_block: tuple[Any, Any] | None = None
    for index, row in enumerate(list(h7_rows) + [{"segment_id": None, "primitive_id": None, "spray_state": None}]):
        block = (row.get("segment_id"), row.get("primitive_id"))
        if index == len(h7_rows) or block != previous_block:
            if index > start:
                previous = h7_rows[start]
                end = index - 1
                state = previous.get("spray_state")
                previous_state = segments[-1]["spray_state"] if segments else None
                segments.append(
                    {
                        "trajectory_id": "h7_certified_post_ruckig_trajectory",
                        "segment_id": previous.get("segment_id"),
                        "primitive_id": previous.get("primitive_id"),
                        "segment_block_id": f"{previous.get('segment_id')}:{previous.get('primitive_id')}",
                        "segment_order": len(segments),
                        "segment_type": "spray_on" if state == "SPRAY_ON" else "spray_off_transfer",
                        "spray_state": state,
                        "start_sample": start,
                        "end_sample": end,
                        "start_time": float(global_times[start]),
                        "end_time": float(global_times[end]),
                        "duration": float(global_times[end]) - float(global_times[start]),
                        "source_target_ids": None,
                        "process_break_reason": "authoritative SPRAY_OFF transfer segment" if state == "SPRAY_OFF" else None,
                        "transition_type": "initial" if previous_state is None else ("spray_state_change" if previous_state != state else "primitive_boundary"),
                    }
                )
            start = index
        previous_block = block
    return segments


def validate_h7_rows(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    errors: list[str] = []
    if not rows:
        return ["no H7 rows"]
    previous_block: tuple[Any, Any] | None = None
    previous_local = -math.inf
    for index, row in enumerate(rows):
        if row.get("joint_names") != JOINT_ORDER:
            errors.append(f"row {index}: authoritative joint order changed")
        if row.get("phase") != "POST_RUCKIG":
            errors.append(f"row {index}: phase is not POST_RUCKIG")
        block = (row.get("segment_id"), row.get("primitive_id"))
        time_value = row.get("time_from_start_s")
        if not isinstance(time_value, (int, float)) or not math.isfinite(float(time_value)):
            errors.append(f"row {index}: invalid segment-local time")
        elif block != previous_block:
            previous_local = float(time_value)
        elif float(time_value) < previous_local:
            errors.append(f"row {index}: non-monotonic segment-local time")
        else:
            previous_local = float(time_value)
        for field in ("positions_rad", "velocities_rad_s", "accelerations_rad_s2"):
            if not _finite_vector(row.get(field)):
                errors.append(f"row {index}: {field} is not a finite six-vector")
        previous_block = block
    global_times, _ = global_message_times(rows)
    if any(b <= a for a, b in zip(global_times, global_times[1:])):
        errors.append("stitched H7/FJT time is not strictly increasing")
    return errors


def validate_feedback_rows(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    errors: list[str] = []
    if not rows:
        return ["no H8-R feedback rows"]
    previous_desired = -math.inf
    for index, row in enumerate(rows):
        for state_name in ("desired", "actual", "error"):
            state = row.get(state_name)
            if not isinstance(state, Mapping):
                errors.append(f"feedback {index}: missing {state_name}")
                continue
            for field in ("positions", "velocities", "accelerations"):
                if not _finite_vector(state.get(field)):
                    errors.append(f"feedback {index}: {state_name}.{field} is not a finite six-vector")
        actual_time = duration_to_seconds((row.get("actual") or {}).get("time_from_start"))
        desired_time = duration_to_seconds((row.get("desired") or {}).get("time_from_start"))
        if actual_time is None or desired_time is None or actual_time < 0.0 or desired_time < 0.0:
            errors.append(f"feedback {index}: desired/actual trajectory clock unavailable")
        elif desired_time < previous_desired:
            errors.append(f"feedback {index}: desired trajectory time is decreasing")
        else:
            # The mock controller's actual field can repeat or locally reset
            # at a boundary. H8-R sorts that field before jerk validation; it
            # is preserved verbatim and is not mistaken for the dataset clock.
            previous_desired = desired_time
    return errors


def compute_forward_jerk(accelerations: Sequence[Sequence[float]], times: Sequence[float]) -> tuple[list[list[float] | None], dict[str, float | None]]:
    """Compute the H8-R-compatible interval jerk with variable dt.

    The first sample is a boundary and receives null.  Every following sample
    stores the forward difference ending at that sample.  This matches H8-R's
    executed-state validator, which checks each adjacent acceleration interval.
    """
    if len(accelerations) != len(times):
        raise DatasetValidationError("acceleration/time length mismatch")
    jerks: list[list[float] | None] = [None]
    dts: list[float] = []
    for previous, current, t0, t1 in zip(accelerations, accelerations[1:], times, times[1:]):
        dt = float(t1) - float(t0)
        if dt < 0.0 or not math.isfinite(dt):
            raise DatasetValidationError("jerk requires non-decreasing finite timestamps")
        if dt == 0.0:
            # H8-R skips non-positive intervals when validating jerk. Keep the
            # sample, but do not fabricate an infinite derivative.
            jerks.append(None)
            continue
        dts.append(dt)
        jerks.append([(float(b) - float(a)) / dt for a, b in zip(previous, current)])
    return jerks, {
        "DT_MIN": min(dts) if dts else None,
        "DT_MAX": max(dts) if dts else None,
        "DT_MEAN": sum(dts) / len(dts) if dts else None,
    }


def _nearest_index(sorted_values: Sequence[float], target: float) -> int:
    index = bisect.bisect_left(sorted_values, target)
    if index <= 0:
        return 0
    if index >= len(sorted_values):
        return len(sorted_values) - 1
    return index if abs(sorted_values[index] - target) < abs(target - sorted_values[index - 1]) else index - 1


def _constraint_bool(value: Any) -> bool:
    return value is True


def _sample_joint_limit_labels(q: Sequence[float], dq: Sequence[float], ddq: Sequence[float], jerk: Sequence[float] | None) -> tuple[dict[str, bool], list[float]]:
    position = all(lo - 1e-10 <= float(value) <= hi + 1e-10 for value, lo, hi in zip(q, EXECUTION_POSITION_LOWER, EXECUTION_POSITION_UPPER))
    velocity = all(abs(float(value)) <= limit + 1e-10 for value, limit in zip(dq, EXECUTION_VMAX))
    acceleration = all(abs(float(value)) <= limit + 1e-10 for value, limit in zip(ddq, EXECUTION_AMAX))
    jerk_valid = True if jerk is None else all(abs(float(value)) <= limit + 1e-8 for value, limit in zip(jerk, EXECUTION_JMAX))
    margin = [min(float(value) - lo, hi - float(value)) for value, lo, hi in zip(q, EXECUTION_POSITION_LOWER, EXECUTION_POSITION_UPPER)]
    return {
        "position_limit_valid": position,
        "velocity_limit_valid": velocity,
        "acceleration_limit_valid": acceleration,
        "jerk_limit_valid": jerk_valid,
    }, margin


def build_schema() -> dict[str, Any]:
    """Return the fully documented machine-readable H9 schema."""
    fields: list[dict[str, Any]] = []

    def add(name: str, dtype: str, shape: Any, units: str | None, frame: str | None, provenance: str, nullable: bool, description: str, level: str = "trajectory_samples") -> None:
        fields.append(
            {
                "name": name,
                "level": level,
                "dtype": dtype,
                "shape": shape,
                "units": units,
                "coordinate_frame": frame,
                "source_provenance": provenance,
                "nullable": nullable,
                "semantic_description": description,
            }
        )

    add("run_id", "string", [], None, None, "H9 source-run identity", False, "Stable identity for one source execution.", "run_metadata_and_trajectory_samples")
    for name, dtype, units, provenance, nullable, description in (
        ("source_stage", "string", None, "H9 release identity", False, "Source stage identity."),
        ("source_artifact_hash", "string", "sha256", "H8-R terminal certificate", False, "Hash of the authoritative source certificate."),
        ("replay_index", "int64|null", "ordinal", "H8-R source execution", True, "Replay index; null for the main certified execution."),
        ("ros_distro", "string", None, "H8-R terminal certificate", False, "ROS distribution used by the source runtime."),
        ("software_runtime", "string", None, "H8-R environment provenance", False, "Software runtime identity."),
        ("mock_only", "bool", None, "H8-R software-only proof", False, "True for this software/mock dataset."),
        ("robot_model", "string", None, "H8-R/URDF configuration", False, "Robot model identity."),
        ("planning_frame", "string", None, "H6.1/H7 frame contract", False, "Planning/reference frame."),
        ("tcp_frame", "string", None, "H6.1/H7 frame contract", False, "TCP/tool frame."),
        ("dataset_schema_version", "string", None, "H9 schema", False, "Versioned dataset schema identifier."),
        ("timestamp", "string", "UTC ISO-8601", "H9 release metadata", False, "Release timestamp excluded from semantic hashing."),
        ("semantic_hash", "string|null", "sha256", "H9 canonical hashing", True, "Semantic run/dataset hash."),
    ):
        add(name, dtype, [], units, None, provenance, nullable, description, "run_metadata")
    add("joint_order", "string[6]", [6], None, None, "authoritative robot/controller ordering", False, "Canonical six-joint order; never inferred by alphabetical sorting.")
    for name, dtype, units, description in (
        ("segment_block_id", "string", "identifier", "Stable H9 identifier combining authoritative segment_id and primitive_id."),
        ("segment_order", "int64", "ordinal", "Authoritative segment/primitive block order."),
        ("segment_type", "enum", None, "Derived classification: spray_on or authoritative spray_off_transfer."),
        ("start_sample", "int64", "index", "Inclusive H7 planned start row."),
        ("end_sample", "int64", "index", "Inclusive H7 planned end row."),
        ("start_time", "float64", "s", "Stitched H8-R/FJT global start time."),
        ("end_time", "float64", "s", "Stitched H8-R/FJT global end time."),
        ("duration", "float64", "s", "Segment duration on the stitched time axis."),
        ("source_target_ids", "array|null", "identifier", "Null because H7/H8 did not persist target IDs."),
        ("process_break_reason", "string|null", None, "Authoritative process/SPRAY semantic note."),
        ("transition_type", "enum", None, "Initial, spray_state_change, or primitive_boundary."),
    ):
        add(name, dtype, [], units, None, "H7 certified trajectory / H9 deterministic segment reconstruction", True if name in {"source_target_ids", "process_break_reason"} else False, description, "trajectory_segments")
    add("trajectory_id", "string", [], None, None, "H7 certified trajectory identity", False, "Identity of the frozen trajectory family member.")
    add("trajectory_family_id", "string", [], None, None, "H9 grouping identity", False, "Leakage-prevention group; deterministic replays share this value.")
    add("replay_id", "string", [], None, None, "H8-R source execution", False, "Execution/replay label; replay rows are not independent demonstrations.")
    add("replay_group_id", "string", [], None, None, "H9 grouping identity", False, "Group key preventing replay leakage.")
    add("sample_index", "int64", [], "count", None, "H8-R feedback ordering", False, "Zero-based row index in the source feedback stream.")
    add("timestamp", "string|null", [], "UTC ISO-8601 or null", None, "H8-R persisted fields", True, "No ROS header timestamp was persisted in authoritative feedback; null is intentional.")
    add("trajectory_time", "float64", [], "s", None, "H8-R feedback.actual.time_from_start", False, "Software execution trajectory clock.")
    add("actual_feedback_time", "float64", [], "s", None, "H8-R feedback.actual.time_from_start", False, "Raw mock actual clock; may repeat/reset at controller boundaries and is preserved separately from the monotonic desired clock.")
    add("desired_trajectory_time", "float64", [], "s", None, "H8-R feedback.desired.time_from_start", False, "Controller desired trajectory clock.")
    add("segment_id", "int64", [], "identifier", None, "H7 certified trajectory", False, "Authoritative H7 segment ID; primitive_id disambiguates repeated IDs.")
    add("primitive_id", "string|int64", [], "identifier", None, "H7 certified trajectory", False, "Authoritative H7 primitive ID.")
    add("segment_order", "int64", [], "ordinal", None, "H9 deterministic reconstruction", False, "Order of the authoritative segment/primitive block.")
    add("spray_state", "enum", [], None, None, "H7 certified SPRAY semantics", False, "SPRAY_ON or SPRAY_OFF carried without reinterpretation.")
    add("target_id", "string|null", [], None, "base_link", "H7/H8 source metadata", True, "Surface target ID was not persisted in H7/H8 feedback.")
    add("planned_source_index", "int64", [], "index", None, "H9 nearest-time alignment to H7", False, "H7 row used for planned/process metadata alignment.")
    add("planned_process_metadata_available", "enum", [], None, None, "H7 certified process validation mapping", False, "Whether the mapped planned state has certified FK/process metadata.")
    for name, source in (
        ("planned_joint_position", "H7 post-Ruckig positions"),
        ("planned_joint_velocity", "H7 post-Ruckig velocities"),
        ("planned_joint_acceleration", "H7 post-Ruckig accelerations"),
        ("desired_joint_position", "H8-R feedback.desired"),
        ("desired_joint_velocity", "H8-R feedback.desired"),
        ("desired_joint_acceleration", "H8-R feedback.desired"),
        ("actual_joint_position", "H8-R feedback.actual; mock runtime"),
        ("actual_joint_velocity", "H8-R feedback.actual; mock runtime"),
        ("actual_joint_acceleration", "H8-R feedback.actual; mock runtime"),
        ("joint_position_error", "H8-R feedback.error"),
        ("joint_velocity_error", "H8-R feedback.error"),
        ("joint_acceleration_error", "H8-R feedback.error"),
        ("derived_joint_jerk", "H9 derived from actual acceleration and actual trajectory time"),
        ("joint_limit_margin", "H9 derived from actual position and H8-R execution bounds"),
    ):
        add(name, "float64[6] or null", [6], "rad, rad/s, rad/s², rad/s³, or rad", None, source, True, "Canonical six-joint vector in j1,j2,j3,j4,j5,j6 order.")
    for name, source, frame in (
        ("desired_tcp_position", "H8-R feedback did not persist TCP pose", "base_link"),
        ("actual_tcp_position", "H8-R feedback did not persist TCP pose; no H9 replay", "base_link"),
        ("tcp_position_error", "H8-R feedback did not persist TCP pose", "base_link"),
    ):
        add(name, "float64[3] or null", [3], "m", frame, source, True, "Unavailable unless an authoritative TCP pose is persisted.")
    for name, source in (("desired_tcp_orientation", "H8-R feedback did not persist TCP pose"), ("actual_tcp_orientation", "H8-R feedback did not persist TCP pose")):
        add(name, "float64[4] or null", [4], "quaternion xyzw", "base_link", source, True, "Unavailable unless an authoritative TCP pose is persisted.")
    add("tcp_orientation_error", "float64|null", [], "deg", "base_link", "H8-R feedback did not persist TCP pose", True, "Unavailable unless an authoritative TCP pose is persisted.")
    add("standoff", "float64|null", [], "m", "base_link", "H7 certified post-Ruckig process validation mapped to planned state", True, "Reconstructed planned-state FK standoff; never a mock sensor measurement.")
    add("standoff_error", "float64|null", [], "m", "base_link", "H7 certified post-Ruckig process validation", True, "Absolute reconstructed planned-state standoff error.")
    add("normal_angle_error", "float64|null", [], "deg", "base_link", "H7 certified post-Ruckig process validation", True, "Reconstructed planned-state normal deviation.")
    add("PROCESS_ASSOCIATION_AVAILABLE", "enum", [], None, "base_link", "H8-R/H7 provenance audit", False, "NO for actual feedback because no per-sample actual FK/surface target association was persisted.")
    add("collision_state", "enum", [], None, None, "H8-R executed-state recertification", False, "Collision-free boolean gate reconstructed from H8-R adaptive discrete validation.")
    add("collision_method", "string", [], None, None, "H8-R/H7 collision contract", False, "Must remain adaptive_discrete_interpolation.")
    add("position_limit_valid", "bool", [], None, None, "H9 evaluation of mock actual state", False, "Actual mock feedback position limit label.")
    add("velocity_limit_valid", "bool", [], None, None, "H9 evaluation of mock actual state", False, "Actual mock feedback velocity limit label.")
    add("acceleration_limit_valid", "bool", [], None, None, "H9 evaluation of mock actual state", False, "Actual mock feedback acceleration limit label.")
    add("jerk_limit_valid", "bool", [], None, None, "H9 evaluation of derived jerk", False, "Actual mock feedback jerk label; first boundary is applicable and valid by definition.")
    add("collision_free", "bool", [], None, None, "H8-R executed-state recertification", False, "Authoritative H8-R collision gate, not inferred clearance.")
    add("process_tolerance_valid", "bool", [], None, None, "H8-R recertified H7 process gate", False, "Authoritative process gate; detailed actual TCP process features remain unavailable.")
    add("time_monotonic", "bool", [], None, None, "H9 validation", False, "Desired trajectory time is non-decreasing; raw mock actual clock behavior is preserved separately.")
    add("spray_semantics_valid", "bool", [], None, None, "H7/H8 spray semantic preservation", False, "Mapped SPRAY_ON/OFF state matches the frozen H7 source.")
    add("hard_constraint_valid", "bool", [], None, None, "H9 logical AND of applicable hard gates", False, "True only when all applicable authoritative gates for this sample are valid.")
    add("provenance", "object", [], None, None, "H9 field provenance map", False, "Explicit planned, desired, mock-actual, derived, and reconstructed labels.")
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "stage": STAGE_ID,
        "dataset_levels": ["run_metadata", "trajectory_segments", "trajectory_samples"],
        "canonical_joint_order": JOINT_ORDER,
        "canonicalization": {
            "algorithm": "SHA-256",
            "representation": "UTF-8 JSON with recursively sorted object keys and compact separators",
            "float_normalization": "finite IEEE-754 values normalized with 17 significant digits; negative zero becomes 0.0",
            "excluded_from_semantic_hash": ["filesystem timestamps", "absolute paths", "generated_at", "runtime wall-clock feedback timestamps"],
        },
        "frames": {"planning_frame": "base_link", "tcp_frame": "spray_tcp_link", "quaternion_order": "xyzw"},
        "fields": fields,
    }


def validate_trajectory_samples(samples: Sequence[Mapping[str, Any]], joint_order: Sequence[str] = JOINT_ORDER) -> list[str]:
    errors: list[str] = []
    previous_time = -math.inf
    for index, sample in enumerate(samples):
        if sample.get("joint_order", list(joint_order)) != list(joint_order):
            errors.append(f"sample {index}: joint order mismatch")
        for field in ("planned_joint_position", "desired_joint_position", "actual_joint_position", "joint_position_error"):
            if not _finite_vector(sample.get(field)):
                errors.append(f"sample {index}: {field} must be a finite six-vector")
        time_value = sample.get("trajectory_time")
        if not isinstance(time_value, (int, float)) or not math.isfinite(float(time_value)) or float(time_value) < previous_time:
            errors.append(f"sample {index}: trajectory_time is decreasing or invalid")
        else:
            previous_time = float(time_value)
        if sample.get("collision_method") != COLLISION_METHOD:
            errors.append(f"sample {index}: collision method is not {COLLISION_METHOD}")
        if not isinstance(sample.get("hard_constraint_valid"), bool):
            errors.append(f"sample {index}: hard_constraint_valid is not boolean")
    return errors


def _rmse(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
    values: list[float] = []
    for row in rows:
        vector = row.get(field)
        if _finite_vector(vector):
            values.extend(float(item) for item in vector)
    return math.sqrt(sum(value * value for value in values) / len(values)) if values else None


def _max_abs(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
    values: list[float] = []
    for row in rows:
        vector = row.get(field)
        if _finite_vector(vector):
            values.extend(abs(float(item)) for item in vector)
    return max(values) if values else None


def _peak_abs(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
    return _max_abs(rows, field)


def _integrated_squared(rows: Sequence[Mapping[str, Any]], field: str, time_field: str = "trajectory_time") -> float | None:
    total = 0.0
    found = False
    for previous, current in zip(rows, rows[1:]):
        vector = current.get(field)
        t0, t1 = previous.get(time_field), current.get(time_field)
        if _finite_vector(vector) and isinstance(t0, (int, float)) and isinstance(t1, (int, float)):
            dt = float(t1) - float(t0)
            if dt > 0.0:
                total += sum(float(item) ** 2 for item in vector) * dt
                found = True
    return total if found else None


def _joint_path_length(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
    total = 0.0
    found = False
    for previous, current in zip(rows, rows[1:]):
        a, b = previous.get(field), current.get(field)
        if _finite_vector(a) and _finite_vector(b):
            total += math.sqrt(sum((float(y) - float(x)) ** 2 for x, y in zip(a, b)))
            found = True
    return total if found else None


def evaluate_trajectory(
    samples: Sequence[Mapping[str, Any]],
    segments: Sequence[Mapping[str, Any]],
    source_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate one run without inventing a scalar reward or clearance."""
    source_metadata = source_metadata or {}
    errors = validate_trajectory_samples(samples)
    if not samples:
        errors.append("trajectory has no samples")
    hard_fields = (
        "position_limit_valid",
        "velocity_limit_valid",
        "acceleration_limit_valid",
        "jerk_limit_valid",
        "collision_free",
        "process_tolerance_valid",
        "time_monotonic",
        "spray_semantics_valid",
    )
    violation_count = sum(1 for row in samples for field in hard_fields if row.get(field) is False)
    hard_valid = not errors and violation_count == 0 and all(row.get("hard_constraint_valid") is True for row in samples)
    times = [float(row["trajectory_time"]) for row in samples if isinstance(row.get("trajectory_time"), (int, float))]
    metrics: dict[str, Any] = {
        "trajectory_duration": times[-1] - times[0] if len(times) >= 2 else None,
        "joint_path_length": _joint_path_length(samples, "actual_joint_position"),
        "planned_joint_path_length": _joint_path_length(samples, "planned_joint_position"),
        "tcp_path_length": None,
        "peak_joint_velocity": _peak_abs(samples, "actual_joint_velocity"),
        "peak_joint_acceleration": _peak_abs(samples, "actual_joint_acceleration"),
        "peak_joint_jerk": _peak_abs(samples, "derived_joint_jerk"),
        "integrated_squared_velocity": _integrated_squared(samples, "actual_joint_velocity"),
        "integrated_squared_acceleration": _integrated_squared(samples, "actual_joint_acceleration"),
        "integrated_squared_jerk": _integrated_squared(samples, "derived_joint_jerk"),
        "tracking_position_rmse": _rmse(samples, "joint_position_error"),
        "tracking_position_max": _max_abs(samples, "joint_position_error"),
        "tracking_velocity_rmse": _rmse(samples, "joint_velocity_error"),
        "tracking_acceleration_rmse": _rmse(samples, "joint_acceleration_error"),
        "tcp_tracking_position_rmse": None,
        "tcp_tracking_orientation_rmse": None,
        "max_standoff_error": max((float(row["standoff_error"]) for row in samples if isinstance(row.get("standoff_error"), (int, float))), default=None),
        "max_normal_error": max((float(row["normal_angle_error"]) for row in samples if isinstance(row.get("normal_angle_error"), (int, float))), default=None),
        "spray_on_duration": sum(float(segment.get("duration", 0.0)) for segment in segments if segment.get("spray_state") == "SPRAY_ON"),
        "spray_off_duration": sum(float(segment.get("duration", 0.0)) for segment in segments if segment.get("spray_state") == "SPRAY_OFF"),
        "reposition_duration": sum(float(segment.get("duration", 0.0)) for segment in segments if segment.get("segment_type") == "spray_off_transfer"),
        "segment_count": len(segments),
        "stop_count": source_metadata.get("stop_count"),
        "constraint_violation_count": violation_count,
    }
    return {
        "schema_version": "stage3_h9_evaluation_metrics_v1",
        "hard_gate_result": {
            "valid": hard_valid,
            "blockers": errors + (["hard_constraint_violation"] if violation_count else []),
            "constraint_violation_count": violation_count,
        },
        "metrics": metrics,
        "metric_units": {
            "trajectory_duration": "s",
            "joint_path_length": "rad",
            "planned_joint_path_length": "rad",
            "tcp_path_length": "m",
            "peak_joint_velocity": "rad/s",
            "peak_joint_acceleration": "rad/s^2",
            "peak_joint_jerk": "rad/s^3",
            "integrated_squared_velocity": "rad^2/s",
            "integrated_squared_acceleration": "rad^2/s^3",
            "integrated_squared_jerk": "rad^2/s^5",
            "tracking_position_rmse": "rad",
            "tracking_position_max": "rad",
            "tracking_velocity_rmse": "rad/s",
            "tracking_acceleration_rmse": "rad/s^2",
            "max_standoff_error": "m",
            "max_normal_error": "deg",
            "spray_on_duration": "s",
            "spray_off_duration": "s",
            "reposition_duration": "s",
            "stop_count": "count",
            "constraint_violation_count": "count",
        },
        "metric_provenance": {
            "actual_joint_*": "H8-R software_execution_feedback",
            "planned_joint_*": "H7 certified post-Ruckig trajectory mapped to H8 desired clock",
            "standoff_and_normal": "H7 certified FK/process metadata reconstructed for mapped planned state; not actual sensor data",
            "tcp_metrics": "unavailable because H8-R feedback did not persist TCP pose and H9 did not replay",
        },
        "ENERGY_METRIC_AVAILABLE": "NO",
        "SCALAR_REWARD_DEFINED": "NO",
    }


def validate_software_only(proof: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if proof.get("MOCK_ONLY_RUNTIME") is not True:
        errors.append("mock_only_runtime_not_proven")
    if proof.get("PHYSICAL_DRIVER_LOADED") is not False:
        errors.append("physical_driver_loaded_or_unknown")
    if proof.get("PHYSICAL_ROBOT_CONNECTED") is not False:
        errors.append("physical_robot_connected_or_unknown")
    if proof.get("PHYSICAL_FJT_GOALS_SENT") != 0:
        errors.append("physical_fjt_goal_count_not_zero")
    return errors


def validate_no_group_leakage(records: Sequence[Mapping[str, Any]], role_field: str = "split_role", group_field: str = "trajectory_family_id") -> list[str]:
    groups: dict[Any, set[Any]] = defaultdict(set)
    for record in records:
        groups[record.get(group_field)].add(record.get(role_field))
    return [f"group {group!r} occurs in roles {sorted(str(role) for role in roles)}" for group, roles in groups.items() if len(roles) > 1]


def assign_group_split(records: Sequence[Mapping[str, Any]], role: str = "generalization") -> list[dict[str, Any]]:
    """Policy-only split helper: assign whole groups, never adjacent samples."""
    return [dict(record, split_role=role, split_group_key=record.get("trajectory_family_id")) for record in records]


def semantic_dataset_payload(run_metadata: Mapping[str, Any], segments: Sequence[Mapping[str, Any]], samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    stable_run = {key: value for key, value in run_metadata.items() if key not in {"timestamp", "semantic_hash", "output_dir"}}
    stable_samples: list[dict[str, Any]] = []
    excluded = {"timestamp", "observed_wall_time_s"}
    for sample in samples:
        stable_samples.append({key: value for key, value in sample.items() if key not in excluded})
    return {
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "run_metadata": stable_run,
        "trajectory_segments": list(segments),
        "trajectory_samples": stable_samples,
    }


def build_split_policy() -> dict[str, Any]:
    return {
        "schema_version": "stage3_h9_split_policy_v1",
        "split_unit": "trajectory_family_id / scenario_id",
        "adjacent_sample_split": "forbidden",
        "replay_split": "forbidden; all deterministic replays remain in one replay_group_id",
        "current_policy_result": "ML_SPLIT_READY: NO",
        "reason": "insufficient_independent_trajectory_groups",
        "DATA_PIPELINE_READY": "YES",
        "MODEL_TRAINING_DATA_SUFFICIENT": "NO",
        "future_roles": ["training", "validation", "test", "generalization"],
    }


def source_hash_manifest(root: Path, targets: Sequence[tuple[str, Path, str]]) -> dict[str, Any]:
    records = []
    missing = []
    for group, path, status in targets:
        if not path.is_file():
            missing.append({"group": group, "path": relpath(root, path), "status": status})
            continue
        records.append(
            {
                "group": group,
                "path": relpath(root, path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "immutable": True,
                "status": status,
            }
        )
    return {
        "schema_version": "stage3_h9_h8_r_frozen_input_manifest_v1",
        "algorithm": "SHA-256",
        "records": records,
        "missing": missing,
        "H7_IMMUTABLE": "YES" if not any(item["group"].startswith("H7") for item in missing) else "NO",
        "H8_R_IMMUTABLE": "YES" if not any(item["group"].startswith("H8_R") for item in missing) else "NO",
        "HISTORICAL_H8_IMMUTABLE": "YES" if not any(item["group"].startswith("HISTORICAL_H8") for item in missing) else "NO",
    }


def verify_source_hash_manifest(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    checks = []
    for record in manifest.get("records", []):
        path = root / record["path"] if not os.path.isabs(record["path"]) else Path(record["path"])
        exists = path.is_file()
        after_hash = sha256_file(path) if exists else None
        checks.append(
            {
                "group": record.get("group"),
                "path": record.get("path"),
                "before_sha256": record.get("sha256"),
                "after_sha256": after_hash,
                "before_size_bytes": record.get("size_bytes"),
                "after_size_bytes": path.stat().st_size if exists else None,
                "unchanged": bool(exists and after_hash == record.get("sha256") and path.stat().st_size == record.get("size_bytes")),
            }
        )
    groups = {}
    for group in ("H7", "H8_R", "HISTORICAL_H8"):
        group_checks = [item for item in checks if item["group"].startswith(group)]
        groups[group] = all(item["unchanged"] for item in group_checks) and bool(group_checks)
    return {
        "schema_version": "stage3_h9_frozen_artifact_verification_v1",
        "checks": checks,
        "H7_IMMUTABLE": "YES" if groups.get("H7") else "NO",
        "H8_R_IMMUTABLE": "YES" if groups.get("H8_R") else "NO",
        "HISTORICAL_H8_IMMUTABLE": "YES" if groups.get("HISTORICAL_H8") else "NO",
        "all_unchanged": all(item["unchanged"] for item in checks),
    }
