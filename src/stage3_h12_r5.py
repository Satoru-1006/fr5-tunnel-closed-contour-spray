"""Pure, fail-closed helpers for Stage 3 H12-R5 evidence analysis."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence


ACCEPTABLE_RUCKIG_RESULTS = {"Working", "Finished"}
BOUNDARY_RECONDITION_FACTORS = (0.999, 0.995, 0.99, 0.95, 0.9, 0.5, 0.0)


def semantic_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sign(value: float, tolerance: float = 1.0e-15) -> int:
    return 1 if value > tolerance else (-1 if value < -tolerance else 0)


def analytic_piece_position_extrema(
    position: float, velocity: float, acceleration: float, jerk: float, duration: float
) -> tuple[float, float]:
    """Exact position range for one constant-jerk native Ruckig phase."""
    if duration < 0.0 or not all(math.isfinite(value) for value in (position, velocity, acceleration, jerk, duration)):
        raise ValueError("invalid constant-jerk phase")
    candidates = [0.0, duration]
    if abs(jerk) > 1.0e-14:
        discriminant = acceleration * acceleration - 2.0 * jerk * velocity
        if discriminant >= 0.0:
            root = math.sqrt(discriminant)
            candidates.extend([(-acceleration - root) / jerk, (-acceleration + root) / jerk])
    elif abs(acceleration) > 1.0e-14:
        candidates.append(-velocity / acceleration)
    values = [
        position + t * (velocity + t * (acceleration / 2.0 + t * jerk / 6.0))
        for t in candidates if 0.0 <= t <= duration
    ]
    return min(values), max(values)


def select_boundary_velocity_recondition(
    velocity: float, max_velocity: float, validator: Any
) -> tuple[float, float] | None:
    """Mirror the deterministic Tier-A candidate ordering used natively."""
    if max_velocity <= 0.0 or abs(velocity) < 0.9 * max_velocity:
        return None
    for factor in BOUNDARY_RECONDITION_FACTORS:
        candidate = velocity * factor
        if validator(candidate):
            return factor, candidate
    return None


def native_signed_overshoot(
    current_position: float,
    target_position: float,
    minimum: float,
    maximum: float,
) -> tuple[float, float, float]:
    """Return signed overshoot, absolute overshoot, and responsible extremum.

    MoveIt calls motion beyond the target an overshoot only on the opposite
    side of the target from the current state. Native Ruckig extrema make this
    calculation exact rather than dependent on a 10 ms sampling grid.
    """

    delta = current_position - target_position
    if delta > 0.0:
        extremum = minimum
        signed = min(0.0, extremum - target_position)
    elif delta < 0.0:
        extremum = maximum
        signed = max(0.0, extremum - target_position)
    else:
        lower = minimum - target_position
        upper = maximum - target_position
        signed = lower if abs(lower) >= abs(upper) else upper
        extremum = minimum if signed == lower else maximum
    return signed, abs(signed), extremum


def completion_gate(record: Mapping[str, Any]) -> dict[str, Any]:
    result_name = str(((record.get("ruckig_result") or {}).get("result_name")) or "")
    checks = {
        "native_result_acceptable": result_name in ACCEPTABLE_RUCKIG_RESULTS,
        "wrapper_true": record.get("native_smoothing_returned_success") is True,
        "smoothing_complete": record.get("smoothing_complete") is True,
        "duration_ceiling_clear": record.get("duration_ceiling_hit") is False,
        "overshoot_gate_passed": record.get("overshoot_gate_passed") is True,
        "output_finite": record.get("output_finite") is True,
        "input_validation_passed": record.get("input_validation_passed") is True,
        "native_error_clear": not record.get("native_error"),
    }
    return {**checks, "raw_working": result_name == "Working", "raw_finished": result_name == "Finished", "accepted": all(checks.values())}


def classify_boundary(event: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    displacement = float(event["next_segment_delta_position"])
    current_velocity = float(event["current_velocity"])
    target_velocity = float(event["target_velocity"])
    current_acceleration = float(event["current_acceleration"])
    target_acceleration = float(event["target_acceleration"])
    retries = list(event.get("retry_absolute_overshoots") or [])
    invariant = len(retries) > 1 and max(retries) - min(retries) <= max(1.0e-12, 1.0e-9 * max(retries))
    evidence = {
        "absolute_displacement_rad": abs(displacement),
        "absolute_current_velocity_rad_s": abs(current_velocity),
        "velocity_sign_reversal": sign(current_velocity) * sign(target_velocity) < 0,
        "acceleration_sign_reversal": sign(current_acceleration) * sign(target_acceleration) < 0,
        "duration_extension_invariant": invariant,
    }
    if abs(displacement) <= 1.0e-3 and abs(current_velocity) > 1.0e-6:
        return "NEAR_ZERO_DISPLACEMENT_NONZERO_VELOCITY", evidence
    if evidence["velocity_sign_reversal"]:
        return "VELOCITY_SIGN_REVERSAL", evidence
    if evidence["acceleration_sign_reversal"]:
        return "ACCELERATION_SIGN_REVERSAL", evidence
    if invariant:
        return "DURATION_EXTENSION_INVARIANT_OVERSHOOT", evidence
    return "OTHER", evidence


def threshold_sensitivity(events: Sequence[Mapping[str, Any]], thresholds: Sequence[float]) -> list[dict[str, Any]]:
    rows = []
    for threshold in thresholds:
        selected = [event for event in events if float(event["absolute_overshoot"]) > float(threshold)]
        values = sorted(float(event["absolute_overshoot"]) for event in selected)
        primitive_ids = {str(event["primitive_id"]) for event in selected}

        def percentile(fraction: float) -> float | None:
            if not values:
                return None
            index = min(len(values) - 1, max(0, math.ceil(fraction * len(values)) - 1))
            return values[index]

        rows.append({
            "threshold_rad": float(threshold),
            "overshoot_event_count": len(selected),
            "affected_primitive_count": len(primitive_ids),
            "max_absolute_overshoot_rad": max(values) if values else None,
            "p50_absolute_overshoot_rad": percentile(0.50),
            "p95_absolute_overshoot_rad": percentile(0.95),
            "p99_absolute_overshoot_rad": percentile(0.99),
        })
    return rows


def aggregate_events(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_joint: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_waypoint: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_retry: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    causes = Counter()
    for event in events:
        by_joint[str(event["joint_name"])].append(event)
        by_waypoint[str(event["waypoint_idx"])].append(event)
        by_retry[str(event["retry_index"])].append(event)
        causes[str(event["root_cause_class"])] += 1

    def summarize(groups: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
        return {
            key: {
                "event_count": len(rows),
                "affected_primitive_count": len({str(row["primitive_id"]) for row in rows}),
                "max_absolute_overshoot_rad": max(float(row["absolute_overshoot"]) for row in rows),
            }
            for key, rows in sorted(groups.items())
        }

    return {
        "by_joint": summarize(by_joint),
        "by_waypoint": summarize(by_waypoint),
        "by_retry": summarize(by_retry),
        "root_cause_counts": dict(sorted(causes.items(), key=lambda item: (-item[1], item[0]))),
    }


def final_gate(certificate: Mapping[str, Any]) -> tuple[bool, str]:
    ordered = [
        (certificate.get("H12_R4_REPRODUCED") == "YES", "h12_r4_reproduction_mismatch"),
        (certificate.get("PRIMITIVES") == 480, "primitive_coverage_failure"),
        (certificate.get("TOTG_FINAL_PASS") == 480, "totg_regression"),
        (certificate.get("RUCKIG_SMOOTHING_SUCCESS") == 480, "ruckig_smoothing_incomplete"),
        (certificate.get("RUCKIG_DURATION_CEILING_HIT") == 0, "ruckig_duration_ceiling"),
        (certificate.get("RUCKIG_NATIVE_ERRORS") == 0, "ruckig_native_error"),
        (certificate.get("POST_RUCKIG_VALIDATED") == 189216, "incomplete_post_ruckig_validation"),
        (certificate.get("POSITION_VIOLATIONS") == 0, "position_violation"),
        (certificate.get("VELOCITY_VIOLATIONS") == 0, "velocity_violation"),
        (certificate.get("ACCELERATION_VIOLATIONS") == 0, "acceleration_violation"),
        (certificate.get("JERK_VIOLATIONS") == 0, "jerk_violation"),
        (certificate.get("SPRAY_PROCESS_VIOLATIONS") == 0, "spray_process_violation"),
        (certificate.get("COLLISION_VIOLATIONS") == 0, "collision_violation"),
        (certificate.get("LEAKAGE") == 0, "future_label_leakage"),
        (certificate.get("REPLAY") == "3/3", "replay_inconsistency"),
        (certificate.get("FOCUSED_TESTS_PASS") is True, "focused_test_failure"),
        (certificate.get("REGRESSION_TESTS_PASS") is True, "regression_test_failure"),
        (certificate.get("UPSTREAM_IMMUTABLE") == "YES", "upstream_mutation"),
    ]
    for passed, blocker in ordered:
        if not passed:
            return False, blocker
    return True, "none"


def exact_coverage(rows: Iterable[Mapping[str, Any]], expected_ids: Iterable[str]) -> bool:
    observed = [str(row.get("unit_id")) for row in rows]
    expected = [str(value) for value in expected_ids]
    return len(observed) == len(expected) and len(set(observed)) == len(observed) and sorted(observed) == sorted(expected)
