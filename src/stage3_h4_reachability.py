"""Portable contracts and validation helpers for the Stage 3 H4 baseline.

The ROS2 runner is deliberately kept separate from this module.  This file
contains the frozen target, seed, FK, candidate, collision, classification and
replay semantics so focused tests can exercise them without a ROS installation.
It never repairs malformed inputs and it never turns a missing backend into a
pass.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "stage3-h4-reachability-v1"
TARGET_SCHEMA_VERSION = "stage3-h4-target-feasibility-v1"
COLLISION_METHOD = "adaptive_discrete_interpolation"
UNAVAILABLE = "not_available"
FORBIDDEN_EXECUTION_TOKENS = (
    "FollowJointTrajectory",
    "send_goal_async",
    "execute()",
    "apply_ruckig",
    "Ruckig",
    "OMPL",
    "TrajOpt",
    "Descartes",
    "RRT",
    "TSP",
    "genetic algorithm",
)


class H4ValidationError(ValueError):
    """Fail-closed H4 input or semantic validation error."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class H4Config:
    """Parameters frozen before any IK call.

    These are declared baseline parameters, not tuned or optimized from the
    observed result.  The KDL timeout and resolution mirror the installed
    MoveIt kinematics.yaml; the seed policy is H4-specific and deterministic.
    """

    planning_group: str = "fairino5_v6_group"
    model_frame: str = "base_link"
    tcp_link: str = "spray_tcp_link"
    ik_timeout_s: float = 0.005
    kinematics_search_resolution_rad: float = 0.005
    seed_count: int = 13
    random_seed: int = 17
    duplicate_tolerance_rad: float = 1.0e-8
    fk_translation_tolerance_m: float = 1.0e-6
    fk_orientation_tolerance_rad: float = 1.0e-6
    quaternion_unit_tolerance: float = 1.0e-8
    joint_limit_tolerance_rad: float = 1.0e-9
    seed_span_fraction: float = 0.45

    def __post_init__(self) -> None:
        if not self.planning_group or not self.model_frame or not self.tcp_link:
            raise H4ValidationError("missing_robot_identity", "planning group, model frame and TCP link are required")
        if not math.isfinite(self.ik_timeout_s) or self.ik_timeout_s <= 0.0:
            raise H4ValidationError("invalid_ik_timeout", "ik_timeout_s must be finite and positive")
        if not math.isfinite(self.kinematics_search_resolution_rad) or self.kinematics_search_resolution_rad <= 0.0:
            raise H4ValidationError("invalid_search_resolution", "kinematics search resolution must be finite and positive")
        if self.seed_count != 13:
            raise H4ValidationError("invalid_seed_count", "the frozen H4 baseline uses exactly 13 seeds")
        if self.random_seed < 0:
            raise H4ValidationError("invalid_random_seed", "random_seed must be non-negative")
        for name in ("duplicate_tolerance_rad", "fk_translation_tolerance_m", "fk_orientation_tolerance_rad", "quaternion_unit_tolerance", "joint_limit_tolerance_rad"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise H4ValidationError("invalid_tolerance", f"{name} must be finite and positive")
        if not 0.0 < self.seed_span_fraction < 0.5:
            raise H4ValidationError("invalid_seed_span", "seed_span_fraction must be in (0, 0.5)")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "planning_group": self.planning_group,
            "model_frame": self.model_frame,
            "tcp_link": self.tcp_link,
            "ik_timeout_s": self.ik_timeout_s,
            "kinematics_search_resolution_rad": self.kinematics_search_resolution_rad,
            "seed_count": self.seed_count,
            "random_seed": self.random_seed,
            "duplicate_tolerance_rad": self.duplicate_tolerance_rad,
            "fk_translation_tolerance_m": self.fk_translation_tolerance_m,
            "fk_orientation_tolerance_rad": self.fk_orientation_tolerance_rad,
            "quaternion_unit_tolerance": self.quaternion_unit_tolerance,
            "joint_limit_tolerance_rad": self.joint_limit_tolerance_rad,
            "seed_span_fraction": self.seed_span_fraction,
            "threshold_provenance": "DECLARED_H4_BASELINE_PARAMETER",
            "post_hoc_tuning": False,
        }


def canonicalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): canonicalize(value[key]) for key in sorted(value, key=lambda item: str(item))}
    if isinstance(value, (list, tuple)):
        return [canonicalize(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str) or isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise H4ValidationError("nonfinite_value", "non-finite value cannot be canonicalized")
        return 0.0 if value == 0.0 else value
    return str(value)


def canonical_json(value: Any) -> str:
    return json.dumps(canonicalize(value), ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)


def semantic_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _vector(value: Any, length: int, field: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != length:
        raise H4ValidationError("invalid_vector", f"{field} must contain exactly {length} values")
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise H4ValidationError("invalid_vector", f"{field} is not numeric") from exc
    if not all(math.isfinite(item) for item in result):
        raise H4ValidationError("nonfinite_value", f"{field} contains NaN or Inf")
    return result


def _unit(value: Any, field: str, tolerance: float = 1.0e-9) -> list[float]:
    result = _vector(value, 3, field)
    norm = math.sqrt(sum(item * item for item in result))
    if norm <= tolerance or not math.isfinite(norm):
        raise H4ValidationError("invalid_unit_vector", f"{field} has zero or non-finite norm")
    if abs(norm - 1.0) > 1.0e-6:
        raise H4ValidationError("non_unit_vector", f"{field} is not unit length")
    return result


def validate_quaternion(value: Any, *, tolerance: float = 1.0e-8) -> list[float]:
    result = _vector(value, 4, "tcp_target_orientation_xyzw")
    norm = math.sqrt(sum(item * item for item in result))
    if abs(norm - 1.0) > tolerance:
        raise H4ValidationError("invalid_quaternion", "TCP quaternion is not unit length")
    return result


def validate_h3_target(target: Mapping[str, Any], *, config: H4Config) -> dict[str, Any]:
    """Validate and copy one H3 target without repairing it."""

    required = (
        "geometry_id", "task_sample_id", "surface_point_xyz_m", "surface_normal_unit",
        "source_triangle_id", "source_barycentric_uvw", "coordinate_frame", "nominal_standoff_m",
        "spray_direction_unit", "tcp_target_position_xyz_m", "tcp_target_orientation_xyzw",
        "source_geometry_hash", "h2_manifest_hash", "configuration_hash", "random_seed",
    )
    missing = [name for name in required if name not in target]
    if missing:
        raise H4ValidationError("malformed_h3_target", "missing H3 fields: " + ", ".join(missing))
    if target.get("target_semantics") != "GEOMETRIC_TASK_TARGET":
        raise H4ValidationError("invalid_target_semantics", "H4 accepts only H3 GEOMETRIC_TASK_TARGET records")
    if target.get("coordinate_frame") != config.model_frame:
        raise H4ValidationError("invalid_frame", "H3 target frame does not match the frozen MoveIt model frame")
    for name in ("source_geometry_hash", "h2_manifest_hash", "configuration_hash"):
        value = str(target.get(name) or "")
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower()):
            raise H4ValidationError("invalid_provenance_hash", f"{name} is not a SHA-256 hex digest")
    _vector(target["surface_point_xyz_m"], 3, "surface_point_xyz_m")
    _unit(target["surface_normal_unit"], "surface_normal_unit")
    _unit(target["spray_direction_unit"], "spray_direction_unit")
    _vector(target["tcp_target_position_xyz_m"], 3, "tcp_target_position_xyz_m")
    validate_quaternion(target["tcp_target_orientation_xyzw"], tolerance=config.quaternion_unit_tolerance)
    barycentric = _vector(target["source_barycentric_uvw"], 3, "source_barycentric_uvw")
    if abs(sum(barycentric) - 1.0) > 1.0e-6 or any(item < -1.0e-9 for item in barycentric):
        raise H4ValidationError("invalid_barycentric", "source barycentric coordinates are invalid")
    standoff = float(target["nominal_standoff_m"])
    if not math.isfinite(standoff) or standoff <= 0.0:
        raise H4ValidationError("invalid_standoff", "nominal standoff must be finite and positive")
    if not isinstance(target["source_triangle_id"], int) or target["source_triangle_id"] < 0:
        raise H4ValidationError("invalid_triangle_provenance", "source triangle id must be a non-negative integer")
    return dict(target)


def _splitmix64(state: int) -> tuple[int, int]:
    state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    value = state
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return state, (value ^ (value >> 31)) & 0xFFFFFFFFFFFFFFFF


def generate_seed_states(
    joint_names: Sequence[str],
    lower: Sequence[float],
    upper: Sequence[float],
    *,
    config: H4Config,
) -> list[dict[str, Any]]:
    """Generate the frozen 13-state seed set in exact order.

    Seed 0 is the finite joint-limit midpoint.  Seeds 1..12 use a fixed
    SplitMix64 stream seeded from H4Config.random_seed, mapped to the inner
    ``seed_span_fraction`` of each finite interval.  There is no runtime
    random API, time dependency, or filesystem-order dependency.
    """

    names = [str(name) for name in joint_names]
    low = _vector(lower, len(names), "joint_lower_bounds")
    high = _vector(upper, len(names), "joint_upper_bounds")
    if len(names) == 0 or any(lo >= hi for lo, hi in zip(low, high)):
        raise H4ValidationError("invalid_joint_bounds", "joint bounds must be finite and strictly ordered")
    center = [(lo + hi) * 0.5 for lo, hi in zip(low, high)]
    seeds: list[dict[str, Any]] = [{"seed_id": "seed_000", "seed_index": 0, "joint_names": names, "joint_positions": center, "generator": "finite_interval_midpoint_v1"}]
    state = int(config.random_seed) & 0xFFFFFFFFFFFFFFFF
    for index in range(1, config.seed_count):
        values: list[float] = []
        raw_words: list[str] = []
        for lo, hi in zip(low, high):
            state, word = _splitmix64(state)
            raw_words.append(f"0x{word:016x}")
            unit = ((word >> 11) + 0.5) / float(1 << 53)
            signed = 2.0 * unit - 1.0
            values.append((lo + hi) * 0.5 + signed * (hi - lo) * config.seed_span_fraction)
        seeds.append({"seed_id": f"seed_{index:03d}", "seed_index": index, "joint_names": names, "joint_positions": values, "generator": "splitmix64_inner_interval_v1", "generator_seed": config.random_seed, "raw_uint64_words": raw_words})
    return seeds


def validate_joint_vector(
    values: Any,
    *,
    expected_joint_names: Sequence[str],
    lower: Sequence[float],
    upper: Sequence[float],
    tolerance: float,
) -> dict[str, Any]:
    try:
        q = _vector(values, len(expected_joint_names), "joint_positions")
    except H4ValidationError as exc:
        return {"valid": False, "rejection_reason": exc.code, "joint_positions": None}
    low = _vector(lower, len(expected_joint_names), "joint_lower_bounds")
    high = _vector(upper, len(expected_joint_names), "joint_upper_bounds")
    violations = [
        {"joint_name": name, "value": value, "lower": lo, "upper": hi}
        for name, value, lo, hi in zip(expected_joint_names, q, low, high)
        if value < lo - tolerance or value > hi + tolerance
    ]
    return {
        "valid": not violations,
        "rejection_reason": None if not violations else "joint_limit_violation",
        "joint_positions": q,
        "joint_names": list(expected_joint_names),
        "joint_limit_violations": violations,
        "finite": True,
        "dimension": len(q),
        "exact_joint_name_order": True,
        "silent_repair_applied": False,
    }


def wrap_delta(values: Sequence[float]) -> list[float]:
    return [((float(value) + math.pi) % (2.0 * math.pi)) - math.pi for value in values]


def circular_distance(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        return math.inf
    return max(abs(value) for value in wrap_delta([x - y for x, y in zip(a, b)]))


def deduplicate_candidates(candidates: Iterable[Mapping[str, Any]], *, tolerance: float) -> list[dict[str, Any]]:
    """Deduplicate returned IK vectors without altering any vector."""

    ordered = sorted((dict(candidate) for candidate in candidates), key=lambda item: (tuple(round(float(value), 12) for value in item["joint_positions"]), int(item.get("seed_index", 0))))
    unique: list[dict[str, Any]] = []
    for candidate in ordered:
        if any(circular_distance(candidate["joint_positions"], existing["joint_positions"]) <= tolerance for existing in unique):
            continue
        unique.append(candidate)
    for index, candidate in enumerate(unique):
        candidate["canonical_candidate_index"] = index
        candidate["candidate_id"] = f"candidate_{index:03d}"
    return unique


def classify_target(
    *,
    input_valid: bool,
    candidate_count_total: int,
    candidate_count_joint_valid: int,
    candidate_count_fk_valid: int,
    candidate_count_collision_free: int,
    self_collision_count: int = 0,
    environment_collision_count: int = 0,
    collision_backend_disagreement: bool = False,
    capability_available: bool = True,
) -> str:
    if not input_valid:
        return "INVALID_INPUT"
    if not capability_available:
        return "CAPABILITY_UNAVAILABLE"
    if collision_backend_disagreement:
        return "COLLISION_BACKEND_DISAGREEMENT"
    if candidate_count_collision_free:
        return "REACHABLE_COLLISION_FREE"
    if environment_collision_count and not self_collision_count:
        return "IK_FOUND_ENV_COLLISION"
    if self_collision_count:
        return "IK_FOUND_SELF_COLLISION"
    if candidate_count_fk_valid:
        return "IK_FOUND_BUT_FK_MISMATCH"
    if candidate_count_joint_valid:
        return "IK_FOUND_BUT_FK_MISMATCH"
    if candidate_count_total:
        return "IK_FOUND_BUT_JOINT_INVALID"
    return "IK_UNREACHABLE"


def aggregate_target_metrics(target_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(target_records)
    by_status: dict[str, int] = {}
    for record in target_records:
        status = str(record.get("classification", "INVALID_INPUT"))
        by_status[status] = by_status.get(status, 0) + 1
    reachable = sum(status == "REACHABLE_COLLISION_FREE" for status in (record.get("classification") for record in target_records))
    collision_free = reachable
    return {
        "schema_version": "stage3-h4-reachability-metrics-v1",
        "target_count": total,
        "reachable_target_count": int(reachable),
        "collision_free_target_count": int(collision_free),
        "reachable_target_ratio": (reachable / total) if total else None,
        "collision_free_target_ratio": (collision_free / total) if total else None,
        "classification_counts": dict(sorted(by_status.items())),
        "denominator_policy": "all H3 targets retained; unreachable targets are never removed",
        "h3_coverage_semantics": "H3_COVERAGE_IS_GEOMETRIC_SELF_CONSISTENCY_BASELINE",
        "planner_performance_claim": False,
        "diagnostic_feasible_target_only_coverage": "DIAGNOSTIC_ONLY",
    }


def replay_semantic_payload(run_records: Sequence[Mapping[str, Any]], metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {"records": list(run_records), "metrics": dict(metrics), "collision_method": COLLISION_METHOD}


def replay_hash(run_records: Sequence[Mapping[str, Any]], metrics: Mapping[str, Any]) -> str:
    return semantic_hash(replay_semantic_payload(run_records, metrics))


def replay_normalize(value: Any) -> Any:
    """Remove wall-clock observations from replay identity material.

    Solver outputs, FK/collision decisions and all declared configuration are
    semantic evidence.  Per-call elapsed time is retained in the raw artifact
    for diagnostics, but it must not make an otherwise identical replay fail
    the deterministic comparison.
    """

    if isinstance(value, Mapping):
        return {str(key): replay_normalize(item) for key, item in value.items() if str(key) not in {"elapsed_s", "elapsed_time_s"}}
    if isinstance(value, list):
        return [replay_normalize(item) for item in value]
    if isinstance(value, tuple):
        return [replay_normalize(item) for item in value]
    return value


def detect_forbidden_execution_source(source: str) -> list[str]:
    return [token for token in FORBIDDEN_EXECUTION_TOKENS if token in source]
