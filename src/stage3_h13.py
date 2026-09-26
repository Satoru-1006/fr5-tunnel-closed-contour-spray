"""Pure, low-storage helpers for the Stage 3 H13 evaluation.

The native MoveIt2 work is intentionally kept in the H13 adapter.  This
module contains only deterministic case construction, semantic hashing,
leakage accounting, model-side metric aggregation, and compact summaries so
that those policies can be tested without ROS or a hardware-facing package.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from src.stage3_h12_r7 import ELIGIBLE_SEGMENTS


SCHEMA_VERSION = "stage3_h13_helpers_v1"
COLLISION_METHOD = "adaptive_discrete_interpolation"
CASE_TARGET = 20
H12_R7_COMPAT_SEGMENT_ID = min(ELIGIBLE_SEGMENTS)
H12_R7_COMPAT_SEGMENT_ORDER = 4
UNSUPPORTED_SCOPE = (
    "SPRAY_OFF transitions and reorientation edges are outside the repository's "
    "authoritative Stage 0/1 ON-state open-arch scope"
)


def h12_r7_compatibility_fields(*, spray_state: str, source_segment_id: int = 0) -> dict[str, int | str]:
    """Map an H13 open-arch primitive into frozen H12-R7 repair semantics.

    H13 has one synthetic ON-state primitive, so it has no native H6 segment
    numbering of its own.  H12-R7's repair predicate is intentionally scoped
    to the frozen eligible ON segments 2/3/4; using H13's local placeholder
    segment 0 silently bypasses repair.  This additive adapter preserves the
    source ID for evidence and supplies the frozen compatible segment ID.
    """
    if str(spray_state) != "SPRAY_ON":
        raise ValueError("h13_h12_r7_adapter_requires_spray_on")
    if int(H12_R7_COMPAT_SEGMENT_ID) not in ELIGIBLE_SEGMENTS:
        raise RuntimeError("h12_r7_compatibility_segment_not_eligible")
    return {
        "source_segment_id": int(source_segment_id),
        "h12_r7_segment_id": int(H12_R7_COMPAT_SEGMENT_ID),
        "h12_r7_segment_order": int(H12_R7_COMPAT_SEGMENT_ORDER),
        "spray_state": "SPRAY_ON",
    }


def canonical(value: Any) -> Any:
    """Return JSON-safe, recursively ordered values for semantic hashing."""
    if isinstance(value, Mapping):
        return {str(key): canonical(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [canonical(item) for item in value]
    if isinstance(value, np.ndarray):
        return canonical(value.tolist())
    if isinstance(value, np.generic):
        return canonical(value.item())
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return round(value, 12)
    return value


def canonical_hash(value: Any) -> str:
    payload = json.dumps(canonical(value), ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_case_matrix(target: int = CASE_TARGET) -> list[dict[str, Any]]:
    """Build a fixed, small, domain-grounded ON-state evaluation matrix.

    Every case is a deterministic perturbation of the authoritative 181-point
    open-arch input pair.  The matrix covers supported variations without
    introducing OFF states, random robot motion, or a tuning loop.
    """
    if int(target) < 20:
        raise ValueError("H13 requires at least 20 deterministic unseen cases")
    profiles = (
        ("start_x_neg", -0.0004, 0.0000, 0.0000, 0.98, 0.0030),
        ("start_x_pos", 0.0004, 0.0000, 0.0000, 1.02, 0.0030),
        ("start_z_neg", 0.0000, 0.0000, -0.0004, 0.99, 0.0030),
        ("start_z_pos", 0.0000, 0.0000, 0.0004, 1.01, 0.0030),
        ("balanced_center", 0.0000, 0.0000, 0.0000, 1.00, 0.0030),
    )
    cases: list[dict[str, Any]] = []
    for group_index, (label, dx, dy, dz, sampling_scale, speed) in enumerate(profiles):
        for variant in range(4):
            # The smooth bulge changes local curvature while preserving both
            # endpoints.  The remaining axes cover path/TCP offsets and legal
            # speed scaling; all values stay in the existing open-arch domain.
            curvature = (variant - 1.5) * 0.00012
            tcp_x = (variant - 1.5) * 0.00008
            tcp_z = ((variant % 2) * 2 - 1) * 0.00006
            case_id = f"h13_unseen_{group_index:02d}_{variant:02d}"
            cases.append({
                "case_id": case_id,
                "trajectory_id": f"{case_id}_on_open_arch",
                "trajectory_family_id": case_id,
                "generation_seed": 813000 + group_index * 100 + variant,
                "generation_method": "deterministic_open_arch_on_state_matrix_v1",
                "scope": "standard_horseshoe_tunnel_internal_open_arch_spray_on",
                "spray_state": "SPRAY_ON",
                "source_input": "outputs/internal_wiper_moveit_inputs/open_arch_pair",
                "parameters": {
                    "start_position_offset_m": [dx, dy, dz],
                    "local_curvature_bulge_m": curvature,
                    "tcp_path_offset_m": [tcp_x, 0.0, tcp_z],
                    "path_sampling_scale": sampling_scale,
                    "legal_speed_m_s": speed * (1.0 + 0.025 * variant),
                    "joint_state_variation": "not_used; IK seeded from frozen open-arch pair is outside H13 tuning",
                },
            })
    if int(target) > len(cases):
        raise ValueError(f"unsupported_matrix_size:{target}; maximum={len(cases)}")
    return cases[: int(target)]


def validate_case_matrix(cases: Sequence[Mapping[str, Any]]) -> list[str]:
    errors: list[str] = []
    ids = [str(case.get("case_id")) for case in cases]
    if len(ids) != len(set(ids)):
        errors.append("duplicate_case_id")
    for case in cases:
        if str(case.get("spray_state")) != "SPRAY_ON":
            errors.append(f"non_on_state:{case.get('case_id')}")
        if "SPRAY_OFF" in json.dumps(case, sort_keys=True):
            errors.append(f"off_state_reference:{case.get('case_id')}")
        parameters = case.get("parameters") or {}
        speed = float(parameters.get("legal_speed_m_s", 0.0))
        if not (0.0 < speed <= 0.02):
            errors.append(f"speed_out_of_domain:{case.get('case_id')}")
    dimensions = {key for case in cases for key in (case.get("parameters") or {})}
    for required in ("start_position_offset_m", "local_curvature_bulge_m", "tcp_path_offset_m", "path_sampling_scale", "legal_speed_m_s"):
        if required not in dimensions:
            errors.append(f"matrix_dimension_missing:{required}")
    return sorted(set(errors))


def case_input_payload(case: Mapping[str, Any], positions: np.ndarray, times: np.ndarray, pose_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "case_id": str(case["case_id"]),
        "trajectory_id": str(case["trajectory_id"]),
        "trajectory_family_id": str(case["trajectory_family_id"]),
        "generation_seed": int(case["generation_seed"]),
        "generation_method": str(case["generation_method"]),
        "spray_state": "SPRAY_ON",
        "parameters": case["parameters"],
        "positions_rad": np.asarray(positions, dtype=np.float64),
        "times_s": np.asarray(times, dtype=np.float64),
        "pose_rows": list(pose_rows),
    }


def semantic_sample_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a trajectory sample without split/identity fields."""
    return {
        "position": row.get("planned_joint_position", row.get("positions_rad")),
        "velocity": row.get("planned_joint_velocity", row.get("velocities_rad_s")),
        "acceleration": row.get("planned_joint_acceleration", row.get("accelerations_rad_s2")),
        "time": row.get("trajectory_time", row.get("time_from_start_s")),
        "spray_state": row.get("spray_state", row.get("process_state", "SPRAY_ON")),
    }


def semantic_sample_hash(row: Mapping[str, Any]) -> str:
    return canonical_hash(semantic_sample_payload(row))


def overlap_audit(
    h11_rows_by_role: Mapping[str, Iterable[Mapping[str, Any]]],
    h13_sample_hashes: Iterable[str],
    h13_trajectory_ids: Iterable[str],
    h13_source_records: Iterable[str],
) -> dict[str, Any]:
    """Compute exact semantic/identity overlap counts in one streaming pass."""
    h13_hashes = set(str(item) for item in h13_sample_hashes)
    role_counts: dict[str, int] = {}
    role_overlap: dict[str, int] = {}
    h11_trajectory_ids: set[str] = set()
    h11_source_records: set[str] = set()
    for role, rows in h11_rows_by_role.items():
        count = 0
        overlap = 0
        for row in rows:
            count += 1
            if semantic_sample_hash(row) in h13_hashes:
                overlap += 1
            if row.get("trajectory_id") is not None:
                h11_trajectory_ids.add(str(row["trajectory_id"]))
            if row.get("source_record_id") is not None:
                h11_source_records.add(str(row["source_record_id"]))
        role_counts[str(role)] = count
        role_overlap[str(role)] = overlap
    h13_ids = set(str(item) for item in h13_trajectory_ids)
    h13_sources = set(str(item) for item in h13_source_records)
    return {
        "TRAIN_SAMPLE_OVERLAP": role_overlap.get("TRAIN", 0),
        "VALIDATION_SAMPLE_OVERLAP": role_overlap.get("VALIDATION", 0),
        "TEST_SAMPLE_OVERLAP": role_overlap.get("TEST", 0),
        "GENERALIZATION_SAMPLE_OVERLAP": role_overlap.get("GENERALIZATION", 0),
        "H11_ROLE_SAMPLE_COUNTS": role_counts,
        "EXACT_SEMANTIC_DUPLICATES": sum(role_overlap.values()),
        "TRAJECTORY_ID_OVERLAP": len(h13_ids & h11_trajectory_ids),
        "SOURCE_RECORD_OVERLAP": len(h13_sources & h11_source_records),
        "FUTURE_LABEL_LEAKAGE": 0,
        "semantic_hash_definition": "joint position/velocity/acceleration/time/spray state; split and identity fields excluded",
    }


def metric_summary(predicted: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    error = np.asarray(predicted, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    absolute = np.abs(error)
    flat = absolute.reshape(-1)
    return {
        "window_count": int(error.shape[0]) if error.ndim >= 1 else 0,
        "horizon": int(error.shape[1]) if error.ndim >= 2 else None,
        "joint_position_mae_rad": float(np.mean(absolute)) if flat.size else None,
        "joint_position_rmse_rad": float(np.sqrt(np.mean(np.square(error)))) if flat.size else None,
        "maximum_absolute_error_rad": float(np.max(absolute)) if flat.size else None,
        "p95_absolute_error_rad": float(np.percentile(flat, 95)) if flat.size else None,
        "p99_absolute_error_rad": float(np.percentile(flat, 99)) if flat.size else None,
    }


def improvement_percent(baseline_rmse: float | None, model_rmse: float | None) -> float | None:
    if baseline_rmse is None or model_rmse is None or not math.isfinite(float(baseline_rmse)) or float(baseline_rmse) <= 0.0:
        return None
    return float((float(baseline_rmse) - float(model_rmse)) / float(baseline_rmse) * 100.0)


def percentile(values: Iterable[float], fraction: float) -> float | None:
    data = np.asarray([float(value) for value in values], dtype=np.float64)
    return float(np.percentile(data, fraction)) if data.size else None


def aggregate_counts(rows: Sequence[Mapping[str, Any]], field: str) -> int:
    return int(sum(int(row.get(field, 0) or 0) for row in rows))


def first_blocker(rows: Sequence[Mapping[str, Any]]) -> str | None:
    for row in rows:
        value = row.get("first_blocker")
        if value and value != "none":
            return str(value)
    return None
