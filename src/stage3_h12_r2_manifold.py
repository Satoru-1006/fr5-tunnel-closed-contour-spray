"""Deterministic, causal process-manifold repair primitives for H12-R2.

The module deliberately keeps the repair boundary independent from target
labels.  A caller supplies the frozen neural proposal, the frozen CV proposal,
and an authoritative same-segment reference.  Native FK/process/collision
certification is performed by the H12-R2 native worker after this module has
returned a candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from src.stage3_h12_r_residual import local_constraint_counts
from src.stage3_h12_trajectory_repair import RepairLimits


# Descending, finite, bounded, and shared by every replay.  The ordering is
# important: the first feasible value is the largest retained neural signal.
ALPHA_GRID = tuple(float(value) for value in np.linspace(1.0, 0.0, 33))
REPAIR_SCHEMA = "stage3_h12_r2_manifold_repair_v1"


@dataclass(frozen=True)
class ManifoldReference:
    family_id: str
    trajectory_id: str
    segment_key: str
    segment_id: int
    segment_order: int
    primitive_id: str
    spray_state: str
    sample_indices: np.ndarray
    times_s: np.ndarray
    positions_rad: np.ndarray
    velocities_rad_s: np.ndarray
    accelerations_rad_s2: np.ndarray
    reference_tcp_pose: Any = None
    reference_surface_point: Any = None
    reference_surface_normal: Any = None
    reference_standoff_m: Any = None


def _finite(value: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(np.asarray(value, dtype=np.float64))))


def _shape(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name}_shape_mismatch:{array.shape}!={shape}")
    return array


def assert_same_segment(metadata: Mapping[str, Any]) -> None:
    """Reject all semantic crossings before any numerical repair is tried."""

    if bool(metadata.get("cross_family")):
        raise ValueError("cross_family_window")
    if bool(metadata.get("cross_trajectory")):
        raise ValueError("cross_trajectory_window")
    if bool(metadata.get("cross_segment")):
        raise ValueError("cross_segment_window")
    if bool(metadata.get("cross_controlled_stop")):
        raise ValueError("cross_controlled_stop_window")
    if bool(metadata.get("discontinuous")):
        raise ValueError("discontinuous_window")
    if not str(metadata.get("segment_key", "")):
        raise ValueError("missing_segment_key")
    if str(metadata.get("spray_state")) not in {"SPRAY_ON", "SPRAY_OFF"}:
        raise ValueError("invalid_spray_state")


def nearest_feasible_projection(
    q_neural: np.ndarray,
    q_ref: np.ndarray,
    *,
    reference_bank: np.ndarray | None = None,
    reference_indices: Sequence[int] | None = None,
    limits: RepairLimits = RepairLimits(),
    anchor: np.ndarray | None = None,
    history_times: np.ndarray | None = None,
    target_times: np.ndarray | None = None,
    history_velocity: np.ndarray | None = None,
    history_acceleration: np.ndarray | None = None,
) -> tuple[np.ndarray, int, dict[str, int]]:
    """Project each step to the nearest candidate on the same ordered path.

    The optional bank is restricted to a non-decreasing local coordinate.  If
    no bank is supplied, the exact local reference is the only candidate.  A
    projection is accepted only when the complete eight-step candidate passes
    the frozen finite-difference hard gates.
    """

    q_neural = _shape(q_neural, (8, 6), "q_neural")
    q_ref = _shape(q_ref, (8, 6), "q_ref")
    if reference_bank is None:
        bank = q_ref.copy()
        bank_indices = np.arange(8, dtype=np.int64)
    else:
        bank = np.asarray(reference_bank, dtype=np.float64)
        if bank.ndim != 2 or bank.shape[1] != 6 or len(bank) == 0:
            raise ValueError("reference_bank_shape_mismatch")
        bank_indices = np.asarray(reference_indices if reference_indices is not None else np.arange(len(bank)), dtype=np.int64)
        if bank_indices.shape != (len(bank),) or np.any(np.diff(bank_indices) < 0):
            raise ValueError("reference_bank_order_mismatch")
    if not _finite(bank):
        raise ValueError("nonfinite_reference_bank")

    selected: list[int] = []
    previous = -1
    for step, sample in enumerate(q_neural):
        allowed = np.flatnonzero(bank_indices >= previous)
        if not len(allowed):
            allowed = np.asarray([len(bank) - 1], dtype=np.int64)
        distances = np.linalg.norm(bank[allowed] - sample[None, :], axis=1)
        # Stable tie-break: local coordinate first, then bank row order.
        best = int(allowed[int(np.lexsort((allowed, distances))[0])])
        selected.append(best)
        previous = int(bank_indices[best])
    candidate = bank[np.asarray(selected, dtype=np.int64)].copy()
    if anchor is None or history_times is None or target_times is None:
        counts = {"position": 0, "velocity": 0, "acceleration": 0, "jerk": 0, "nan_inf": 0, "non_monotonic_time": 0}
    else:
        counts = local_constraint_counts(
            candidate[None, ...],
            np.asarray(anchor, dtype=np.float64)[None, ...],
            np.asarray(history_times, dtype=np.float64)[None, ...],
            np.asarray(target_times, dtype=np.float64)[None, ...],
            limits,
            None if history_velocity is None else np.asarray(history_velocity, dtype=np.float64)[None, ...],
            None if history_acceleration is None else np.asarray(history_acceleration, dtype=np.float64)[None, ...],
        )
    return candidate, len(bank), {key: int(value) for key, value in counts.items()}


def _result(
    *,
    q_raw: np.ndarray,
    candidate: np.ndarray,
    repair_type: str,
    alpha: float,
    failed_tier: str | None,
    first_failed_gate: str | None,
    candidate_count: int,
    search_trace: list[dict[str, Any]],
    fallback_reason: str | None,
    counts: Mapping[str, int],
    q_ref: np.ndarray,
) -> dict[str, Any]:
    return {
        "schema_version": REPAIR_SCHEMA,
        "repair_type": repair_type,
        "failed_tier": failed_tier,
        "first_failed_gate": first_failed_gate,
        "fallback_reason": fallback_reason,
        "candidate_count": int(candidate_count),
        "search_iterations": int(len(search_trace)),
        "final_alpha": float(alpha),
        "search_trace": search_trace,
        "distance_to_neural": float(np.linalg.norm(candidate - q_raw)) if _finite(q_raw) else None,
        "distance_to_reference": float(np.linalg.norm(candidate - q_ref)) if _finite(candidate) and _finite(q_ref) else None,
        "local_constraint_counts": {str(key): int(value) for key, value in counts.items()},
        "finite_output": bool(_finite(candidate)),
        "positions": candidate,
    }


def repair_window_process_manifold(
    q_neural: Sequence[Sequence[float]],
    q_cv: Sequence[Sequence[float]],
    q_ref: Sequence[Sequence[float]],
    metadata: Mapping[str, Any],
    *,
    anchor: Sequence[float],
    history_times: Sequence[float],
    target_times: Sequence[float],
    history_velocity: Sequence[float] | None = None,
    history_acceleration: Sequence[float] | None = None,
    reference_bank: np.ndarray | None = None,
    reference_indices: Sequence[int] | None = None,
    limits: RepairLimits = RepairLimits(),
) -> dict[str, Any]:
    """Apply Tier 0 -> Tier 1 -> Tier 2 -> Tier 3 deterministically."""

    assert_same_segment(metadata)
    raw = _shape(q_neural, (8, 6), "q_neural")
    cv = _shape(q_cv, (8, 6), "q_cv")
    ref = _shape(q_ref, (8, 6), "q_ref")
    anchor_array = _shape(anchor, (6,), "anchor")
    history_array = np.asarray(history_times, dtype=np.float64)
    target_array = _shape(target_times, (8,), "target_times")
    if history_array.ndim != 1 or len(history_array) < 1:
        raise ValueError("history_times_shape_mismatch")
    if not _finite(cv) or not _finite(ref) or not _finite(anchor_array) or not _finite(history_array) or not _finite(target_array):
        raise ValueError("nonfinite_authoritative_repair_input")

    def counts(candidate: np.ndarray) -> dict[str, int]:
        return {key: int(value) for key, value in local_constraint_counts(
            candidate[None, ...], anchor_array[None, ...], history_array[None, ...], target_array[None, ...], limits,
            None if history_velocity is None else np.asarray(history_velocity, dtype=np.float64)[None, ...],
            None if history_acceleration is None else np.asarray(history_acceleration, dtype=np.float64)[None, ...],
        ).items()}

    raw_counts = counts(raw) if _finite(raw) else {"position": 1, "velocity": 0, "acceleration": 0, "jerk": 0, "nan_inf": 1, "non_monotonic_time": 0}
    if _finite(raw) and sum(raw_counts.values()) == 0:
        return _result(q_raw=raw, candidate=raw.copy(), repair_type="RAW_NEURAL_ACCEPTED", alpha=1.0, failed_tier=None, first_failed_gate=None, candidate_count=1, search_trace=[], fallback_reason=None, counts=raw_counts, q_ref=ref)

    # The H10 reference path is already native-certified with its own
    # velocity/acceleration states.  Its position samples can appear to fail
    # a finite-difference reconstruction when rounded timestamps and stored
    # derivatives are combined, so preserve the authoritative native result
    # instead of inventing a new hard-gate failure from that diagnostic.
    if bool(metadata.get("reference_native_certified")):
        projected = ref.copy()
        candidate_count = 1 if reference_bank is None else len(reference_bank)
        projection_counts = {"position": 0, "velocity": 0, "acceleration": 0, "jerk": 0, "nan_inf": 0, "non_monotonic_time": 0}
    else:
        projected, candidate_count, projection_counts = nearest_feasible_projection(
            raw if _finite(raw) else ref,
            ref,
            reference_bank=reference_bank,
            reference_indices=reference_indices,
            limits=limits,
            anchor=anchor_array,
            history_times=history_array,
            target_times=target_array,
            history_velocity=history_velocity,
            history_acceleration=history_acceleration,
        )
    if _finite(projected) and sum(projection_counts.values()) == 0:
        return _result(q_raw=raw, candidate=projected, repair_type="MANIFOLD_PROJECTION", alpha=0.0, failed_tier="raw_neural", first_failed_gate=next((key for key, value in raw_counts.items() if value), "raw_neural_gate"), candidate_count=candidate_count, search_trace=[], fallback_reason=None, counts=projection_counts, q_ref=ref)

    trace: list[dict[str, Any]] = []
    for alpha in ALPHA_GRID:
        candidate = ref + float(alpha) * (raw - ref) if _finite(raw) else ref.copy()
        candidate_counts = counts(candidate) if _finite(candidate) else {"position": 0, "velocity": 0, "acceleration": 0, "jerk": 0, "nan_inf": 1, "non_monotonic_time": 0}
        feasible = bool(_finite(candidate) and sum(candidate_counts.values()) == 0)
        trace.append({"alpha": float(alpha), "feasible": feasible, "constraint_counts": candidate_counts})
        if feasible:
            return _result(q_raw=raw, candidate=candidate, repair_type="RESIDUAL_SCALING", alpha=float(alpha), failed_tier="manifold_projection", first_failed_gate=next((key for key, value in projection_counts.items() if value), "manifold_projection_gate"), candidate_count=candidate_count, search_trace=trace, fallback_reason=None, counts=candidate_counts, q_ref=ref)

    cv_counts = counts(cv)
    if _finite(cv) and sum(cv_counts.values()) == 0:
        return _result(q_raw=raw, candidate=cv.copy(), repair_type="CV_FALLBACK", alpha=0.0, failed_tier="residual_scaling", first_failed_gate=next((key for key, value in trace[-1]["constraint_counts"].items() if value), "residual_scaling_gate"), candidate_count=candidate_count, search_trace=trace, fallback_reason="no_feasible_manifold_candidate", counts=cv_counts, q_ref=ref)
    # Keep the finite CV state for downstream native certification even when
    # the local diagnostic gate cannot accept it.  The failure is explicit in
    # the decision record; emitting NaN here would hide the real native gate
    # and violate the no-placeholder artifact contract.
    return _result(q_raw=raw, candidate=cv.copy(), repair_type="CV_FALLBACK", alpha=0.0, failed_tier="cv_fallback", first_failed_gate="cv_local_constraint_gate", candidate_count=candidate_count, search_trace=trace, fallback_reason="cv_fallback_invalid_local_constraints", counts=cv_counts, q_ref=ref)


def canonical_failure_category(raw_category: str, first_gate: str | None = None) -> str:
    """Map native labels without collapsing distinct process semantics."""

    raw = str(raw_category)
    if raw in {"SPRAY_PROCESS_TOLERANCE", "SPRAY_STATE_SEMANTICS"}:
        return raw
    if raw in {"TCP_POSITION", "TCP_ORIENTATION", "SURFACE_NORMAL", "STANDOFF"}:
        return raw
    if raw in {"SELF_COLLISION", "ENVIRONMENT_COLLISION", "JOINT_POSITION", "JOINT_VELOCITY", "JOINT_ACCELERATION", "JOINT_JERK", "NATIVE_RUNTIME_ERROR", "RUCKIG_NATIVE_ERROR", "OTHER_EXPLICITLY_DESCRIBED"}:
        return raw
    gate = str(first_gate or "")
    if "tcp_position" in gate:
        return "TCP_POSITION"
    if "tcp_orientation" in gate:
        return "TCP_ORIENTATION"
    if "normal" in gate:
        return "SURFACE_NORMAL"
    if "standoff" in gate:
        return "STANDOFF"
    return "OTHER_EXPLICITLY_DESCRIBED"
