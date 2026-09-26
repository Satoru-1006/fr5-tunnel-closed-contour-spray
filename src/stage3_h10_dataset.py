"""Pure-Python contracts used by the Stage 3 H10 dataset builder.

The native MoveIt2 worker owns all claims about FK, PlanningScene/FCL,
TOTG/Ruckig, and process validity.  This module only performs deterministic
dataset assembly, family identity, diversity, split, and fail-closed audit
operations on those native results.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence


STAGE_ID = "STAGE_3_H10"
DATASET_SCHEMA_VERSION = "stage3_h10_dataset_schema_v1"
COLLISION_METHOD = "adaptive_discrete_interpolation"
JOINT_ORDER = ["j1", "j2", "j3", "j4", "j5", "j6"]
MINIMUM_ACCEPTED_TRAJECTORY_FAMILIES = 30
SIMILARITY_POSITION_TOLERANCE_RAD = 5.0e-5
SIMILARITY_RMS_TOLERANCE_RAD = 1.0e-5


class H10ValidationError(ValueError):
    """Raised when an H10 invariant cannot be satisfied."""


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise H10ValidationError("non_finite_value_in_semantic_payload")
        if value == 0.0:
            return 0.0
        return float(format(value, ".17g"))
    return value


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(_canonical(value), ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def semantic_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _vec(row: Mapping[str, Any], field: str = "positions_rad") -> list[float]:
    value = row.get(field)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 6:
        raise H10ValidationError(f"invalid_six_vector:{field}")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise H10ValidationError(f"non_finite_six_vector:{field}")
    return result


def _resample_vectors(rows: Sequence[Mapping[str, Any]], count: int = 64) -> list[list[float]]:
    if len(rows) < 2:
        raise H10ValidationError("family_path_has_fewer_than_two_states")
    q = [_vec(row) for row in rows]
    result: list[list[float]] = []
    for sample in range(count):
        position = sample * (len(q) - 1) / float(count - 1)
        left = min(len(q) - 2, int(math.floor(position)))
        alpha = position - left
        result.append([(1.0 - alpha) * q[left][joint] + alpha * q[left + 1][joint] for joint in range(6)])
    return result


def geometric_path_payload(segments: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return a timing-independent representation of a complete joint path."""
    canonical_segments: list[dict[str, Any]] = []
    for segment in sorted(segments, key=lambda item: int(item.get("segment_order", 0))):
        rows = segment.get("rows") or []
        canonical_segments.append({
            "segment_id": segment.get("segment_id"),
            "primitive_id": segment.get("primitive_id"),
            "spray_state": segment.get("spray_state"),
            "positions_rad_resampled": _resample_vectors(rows),
        })
    return {"joint_order": list(JOINT_ORDER), "segments": canonical_segments}


def geometric_path_signature(segments: Sequence[Mapping[str, Any]]) -> str:
    return semantic_sha256(geometric_path_payload(segments))


def compare_geometric_paths(left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    left_sorted = sorted(left, key=lambda item: int(item.get("segment_order", 0)))
    right_sorted = sorted(right, key=lambda item: int(item.get("segment_order", 0)))
    if len(left_sorted) != len(right_sorted):
        return {"comparable": False, "reason": "segment_count_mismatch", "max_position_delta_rad": None, "rms_position_delta_rad": None}
    deltas: list[float] = []
    segment_deltas: list[dict[str, Any]] = []
    for a, b in zip(left_sorted, right_sorted):
        if (a.get("segment_id"), a.get("primitive_id"), a.get("spray_state")) != (b.get("segment_id"), b.get("primitive_id"), b.get("spray_state")):
            return {"comparable": False, "reason": "segment_identity_mismatch", "max_position_delta_rad": None, "rms_position_delta_rad": None}
        av = _resample_vectors(a.get("rows") or [])
        bv = _resample_vectors(b.get("rows") or [])
        local = [math.sqrt(sum((x - y) ** 2 for x, y in zip(xv, yv))) for xv, yv in zip(av, bv)]
        deltas.extend(local)
        segment_deltas.append({"segment_id": a.get("segment_id"), "max_position_delta_rad": max(local), "rms_position_delta_rad": math.sqrt(sum(value * value for value in local) / len(local))})
    maximum = max(deltas) if deltas else 0.0
    rms = math.sqrt(sum(value * value for value in deltas) / len(deltas)) if deltas else 0.0
    return {"comparable": True, "reason": None, "max_position_delta_rad": maximum, "rms_position_delta_rad": rms, "segment_deltas": segment_deltas, "trivial_duplicate": maximum <= SIMILARITY_POSITION_TOLERANCE_RAD or rms <= SIMILARITY_RMS_TOLERANCE_RAD}


def validate_family_identity(family: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if not family.get("trajectory_family_id"):
        errors.append("missing_trajectory_family_id")
    if not family.get("trajectory_id"):
        errors.append("missing_trajectory_id")
    if not isinstance(family.get("generation_seed"), int):
        errors.append("generation_seed_not_integer")
    if not family.get("geometric_path_signature"):
        errors.append("missing_geometric_path_signature")
    if not family.get("generation_method"):
        errors.append("missing_generation_method")
    return errors


def validate_sample_rows(rows: Sequence[Mapping[str, Any]], expected_family_id: str | None = None) -> list[str]:
    errors: list[str] = []
    previous_time = -math.inf
    for index, row in enumerate(rows):
        if row.get("joint_order") != list(JOINT_ORDER):
            errors.append(f"sample_{index}:joint_order_mismatch")
        if expected_family_id is not None and row.get("trajectory_family_id") != expected_family_id:
            errors.append(f"sample_{index}:family_id_mismatch")
        for field in ("planned_joint_position", "planned_joint_velocity", "planned_joint_acceleration", "desired_joint_position", "actual_joint_position", "actual_joint_velocity", "actual_joint_acceleration"):
            try:
                _vec(row, field)
            except H10ValidationError as exc:
                errors.append(f"sample_{index}:{exc}")
        time_value = row.get("trajectory_time")
        if not isinstance(time_value, (int, float)) or not math.isfinite(float(time_value)) or float(time_value) < previous_time:
            errors.append(f"sample_{index}:non_monotonic_trajectory_time")
        else:
            previous_time = float(time_value)
        if row.get("collision_method") != COLLISION_METHOD:
            errors.append(f"sample_{index}:collision_method_mismatch")
        if row.get("spray_state") not in {"SPRAY_ON", "SPRAY_OFF"}:
            errors.append(f"sample_{index}:invalid_spray_state")
        if row.get("hard_constraint_valid") is not True:
            errors.append(f"sample_{index}:hard_constraint_invalid")
        if row.get("actual_tcp_position") is not None or row.get("actual_tcp_orientation") is not None:
            errors.append(f"sample_{index}:unavailable_tcp_signal_not_null")
    return errors


def aggregate_hard_constraints(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fields = ("position_limit_valid", "velocity_limit_valid", "acceleration_limit_valid", "jerk_limit_valid", "collision_free", "process_tolerance_valid", "time_monotonic", "spray_semantics_valid")
    counts = {field: sum(row.get(field) is False for row in rows) for field in fields}
    hard_invalid = sum(row.get("hard_constraint_valid") is not True for row in rows)
    return {"sample_count": len(rows), "violation_counts": counts, "hard_constraint_invalid_sample_count": hard_invalid, "HARD_CONSTRAINT_VIOLATIONS": sum(counts.values()) + hard_invalid}


def _range(rows: Sequence[Mapping[str, Any]], field: str) -> list[float] | None:
    vectors = [_vec(row, field) for row in rows]
    if not vectors:
        return None
    return [max(vector[j] for vector in vectors) - min(vector[j] for vector in vectors) for j in range(6)]


def _peak(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
    values = [abs(value) for row in rows for value in _vec(row, field)]
    return max(values) if values else None


def family_metrics(samples: Sequence[Mapping[str, Any]], segments: Sequence[Mapping[str, Any]], geometric_signature: str) -> dict[str, Any]:
    if not samples:
        raise H10ValidationError("family_has_no_samples")
    duration = float(samples[-1]["trajectory_time"]) - float(samples[0]["trajectory_time"])
    path_length = sum(math.sqrt(sum((b[j] - a[j]) ** 2 for j in range(6))) for a, b in zip([_vec(row, "planned_joint_position") for row in samples], [_vec(row, "planned_joint_position") for row in samples][1:]))
    return {
        "trajectory_family_id": samples[0].get("trajectory_family_id"),
        "geometric_joint_path_signature": geometric_signature,
        "joint_path_length_rad": path_length,
        "trajectory_duration_s": duration,
        "per_joint_position_range_rad": _range(samples, "planned_joint_position"),
        "per_joint_velocity_range_rad_s": _range(samples, "planned_joint_velocity"),
        "per_joint_acceleration_range_rad_s2": _range(samples, "planned_joint_acceleration"),
        "peak_jerk_rad_s3": _peak([row for row in samples if row.get("derived_joint_jerk") is not None], "derived_joint_jerk") if any(row.get("derived_joint_jerk") is not None for row in samples) else None,
        "segment_count": len(segments),
        "spray_on_duration_s": sum(float(item.get("duration", 0.0)) for item in segments if item.get("spray_state") == "SPRAY_ON"),
        "spray_off_duration_s": sum(float(item.get("duration", 0.0)) for item in segments if item.get("spray_state") == "SPRAY_OFF"),
        "reposition_duration_s": sum(float(item.get("duration", 0.0)) for item in segments if item.get("segment_type") == "spray_off_transfer"),
        "stop_count": sum(int(item.get("controlled_stop_count", 0)) for item in segments),
        "start_configuration_rad": _vec(samples[0], "planned_joint_position"),
        "end_configuration_rad": _vec(samples[-1], "planned_joint_position"),
        "tcp_metrics": "unavailable_not_persisted_in_H8_R_feedback",
        "standoff_metrics": {"available": any(isinstance(row.get("standoff"), (int, float)) for row in samples)},
        "normal_error_metrics": {"available": any(isinstance(row.get("normal_angle_error"), (int, float)) for row in samples)},
    }


def deterministic_group_split(families: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Assign complete trajectory families to deterministic benchmark roles."""
    ids = sorted({str(item["trajectory_family_id"]) for item in families}, key=lambda value: semantic_sha256({"family": value}))
    if not ids:
        return {}
    n = len(ids)
    generalization = max(0, min(6, n // 10)) if n >= 40 else 0
    remaining = n - generalization
    train = max(1, round(remaining * 0.70))
    validation = max(1, round(remaining * 0.15))
    test = remaining - train - validation
    if test < 1:
        test = 1
        train = max(1, remaining - validation - test)
    roles: dict[str, str] = {}
    for index, family_id in enumerate(ids):
        if index < train:
            roles[family_id] = "TRAIN"
        elif index < train + validation:
            roles[family_id] = "VALIDATION"
        elif index < remaining:
            roles[family_id] = "TEST"
        else:
            roles[family_id] = "GENERALIZATION"
    return roles


def validate_group_split(records: Sequence[Mapping[str, Any]], role_field: str = "split_role", group_field: str = "trajectory_family_id") -> list[str]:
    roles: dict[Any, set[Any]] = defaultdict(set)
    for record in records:
        roles[record.get(group_field)].add(record.get(role_field))
    return [f"{group_field}={group!r}:roles={sorted(str(role) for role in values)}" for group, values in roles.items() if len(values) > 1]


def build_schema() -> dict[str, Any]:
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "stage": STAGE_ID,
        "units": {"joint_position": "rad", "joint_velocity": "rad/s", "joint_acceleration": "rad/s^2", "joint_jerk": "rad/s^3", "time": "s", "standoff": "m", "normal_error": "deg"},
        "frames": {"planning_frame": "base_link", "tcp_frame": "spray_tcp_link", "quaternion_order": "xyzw"},
        "canonical_joint_order": list(JOINT_ORDER),
        "collision_method": COLLISION_METHOD,
        "null_semantics": "null means unavailable/not persisted; unavailable values are never replaced with numeric zero",
        "split_policy": {"unit": "trajectory_family_id", "adjacent_sample_split": "forbidden", "replay_split": "forbidden", "timing_variant_split": "forbidden"},
        "fields": [
            {"name": name, "level": level, "dtype": dtype, "nullable": nullable, "provenance": provenance}
            for name, level, dtype, nullable, provenance in [
                ("run_id", "sample", "string", False, "H10 builder"), ("scenario_id", "sample", "string", False, "frozen task contract"), ("trajectory_id", "sample", "string", False, "H10 native candidate"), ("trajectory_family_id", "sample", "string", False, "geometric path identity"), ("trajectory_variant_id", "sample", "string", False, "H10 candidate variant"), ("generation_seed", "sample", "int64", False, "deterministic generator"), ("generation_method", "sample", "string", False, "deterministic generator"), ("parent_source", "sample", "string", False, "H9/H7 provenance"), ("segment_id", "sample", "int64", False, "native candidate"), ("primitive_id", "sample", "string|int64", False, "native candidate"), ("spray_state", "sample", "enum", False, "frozen H7 semantics"), ("sample_index", "sample", "int64", False, "H10 dataset"), ("trajectory_time", "sample", "float64", False, "native post-Ruckig"), ("joint_order", "sample", "string[6]", False, "frozen controller order"), ("planned_joint_position", "sample", "float64[6]", False, "native post-Ruckig"), ("planned_joint_velocity", "sample", "float64[6]", False, "native post-Ruckig"), ("planned_joint_acceleration", "sample", "float64[6]", False, "native post-Ruckig"), ("desired_joint_position", "sample", "float64[6]", False, "software mock desired"), ("actual_joint_position", "sample", "float64[6]", False, "software mock actual"), ("actual_tcp_position", "sample", "float64[3]", True, "unavailable in H8-R"), ("derived_joint_jerk", "sample", "float64[6]", True, "finite difference diagnostic"), ("hard_constraint_valid", "sample", "bool", False, "logical AND"), ("collision_method", "sample", "string", False, "native H7/H10"), ("process_tolerance_valid", "sample", "bool", False, "native FK/process or not-applicable OFF"), ("provenance", "sample", "object", False, "field-level provenance"),
                ("segment_order", "segment", "int64", False, "native candidate"), ("segment_type", "segment", "enum", False, "H7 SPRAY semantics"), ("duration", "segment", "float64", False, "native post-Ruckig"),
            ]
        ],
        "semantic_hash_exclusions": ["generated_at", "output_dir", "wall_clock_timestamp", "absolute_paths"],
    }


def validate_software_only(proof: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    expected = {"MOCK_ONLY_DATA": "YES", "PHYSICAL_ROBOT_CONNECTED": "NO", "PHYSICAL_DRIVER_LOADED": "NO", "PHYSICAL_FJT_GOALS_SENT": 0, "ROBOT_MOTION_STARTED": "NO"}
    for key, value in expected.items():
        if proof.get(key) != value:
            errors.append(f"software_only_boundary_failed:{key}")
    return errors


def summarize_distribution(samples: Sequence[Mapping[str, Any]], families: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {"spray_state_samples": dict(Counter(str(row.get("spray_state")) for row in samples)), "generation_methods": dict(Counter(str(row.get("generation_method")) for row in samples)), "family_generation_methods": dict(Counter(str(row.get("generation_method")) for row in families)), "segment_types": dict(Counter(str(row.get("segment_type")) for row in families))}
