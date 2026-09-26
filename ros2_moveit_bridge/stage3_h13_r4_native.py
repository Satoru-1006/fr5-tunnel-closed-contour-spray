#!/usr/bin/env python3
"""Additive H13-R4 native adapter.

This process reuses the authoritative H13/R4 MoveIt2, FK, TOTG, Ruckig,
PlanningScene, and spray-process implementations.  It replaces only the
H12-R7 reprojection callback with a bounded deterministic ladder and adds
compact root-cause evidence.  The H12-R7 and H13-R3 files are not modified.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

import stage3_h12_r4_native as r4
import stage3_h13_native as h13
import stage3_h7_native as collision_base
from src.stage3_h13_r4_reprojection import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_MAX_SEGMENT_WAYPOINTS,
    DEFAULT_RADIUS_LADDER_RAD,
    TIER0,
    TIER1,
    TIER2,
    TIER3,
    classify_failure,
    contiguous_ranges,
    correction_summary,
    expand_range,
    no_label_repair_audit,
    radius_ladder,
    signed_limit_correction,
)


_ORIGINAL_LOCAL_REPROJECTION = r4.localized_tcp_reprojection
_ORIGINAL_RAW_AUDIT = h13.raw_audit
_ORIGINAL_COMPACT_RECORD = h13.compact_record


def _safe_collision_state(scene: Any, state: Any, q: np.ndarray) -> dict[str, Any]:
    """Preserve fail-closed collision semantics around a MoveIt Python bug.

    Some installed MoveIt Python bindings throw while converting the optional
    ``contacts`` property even when ``result.collision`` is false.  The
    collision decision remains the native result: false is collision-free;
    true with unavailable contacts is an unknown collision source and fails
    closed.  No CCD or clearance is introduced.
    """

    state.set_joint_group_positions(collision_base.GROUP_NAME, np.asarray(q, dtype=float).tolist())
    state.update()
    request = collision_base.CollisionRequest()
    request.joint_model_group_name = collision_base.GROUP_NAME
    request.contacts = True
    request.max_contacts = 4096
    request.max_contacts_per_pair = 64
    result = collision_base.CollisionResult()
    scene.check_collision(request, result, state)
    try:
        pairs = collision_base.contact_pairs(result)
        contacts_available = True
    except Exception:
        pairs = []
        contacts_available = False
    collision = bool(result.collision)
    if collision and not pairs:
        pairs = ["unknown_collision_source"]
    environment_pairs = [pair for pair in pairs if "fixture_surface" in pair or "world" in pair or "tunnel" in pair]
    self_pairs = [pair for pair in pairs if pair not in environment_pairs]
    if not contacts_available and not collision:
        environment_pairs = []
        self_pairs = []
    return {
        "environment_collision": bool(environment_pairs),
        "self_collision": bool(self_pairs),
        "collision_free": not environment_pairs and not self_pairs,
        "environment_collision_pairs": environment_pairs,
        "self_collision_pairs": self_pairs,
        "collision_method": h13.COLLISION_METHOD,
        "ccd_status": collision_base.CCD_STATUS,
        "clearance_status": collision_base.CLEARANCE_STATUS,
        "clearance_m": None,
        "contacts_conversion_available": contacts_available,
    }


def _candidate_seeds(working: np.ndarray, index: int, left: int, right: int, tier: str) -> list[tuple[str, np.ndarray]]:
    values: list[tuple[str, np.ndarray]] = [("current", working[index].copy())]
    if index > 0:
        values.append(("previous_neighbor", working[index - 1].copy()))
    if index + 1 < len(working):
        values.append(("next_neighbor", working[index + 1].copy()))
    if tier == TIER3:
        anchor_left = working[left - 1] if left > 0 else working[index]
        anchor_right = working[right + 1] if right + 1 < len(working) else working[index]
        alpha = (index - left + 1) / float(max(1, right - left + 2))
        values.append(("bounded_anchor_interpolation", (1.0 - alpha) * anchor_left + alpha * anchor_right))
    return values


def _try_ik_candidate(
    moveit: Any,
    primitive: Mapping[str, Any],
    item: Mapping[str, Any],
    working: np.ndarray,
    index: int,
    left: int,
    right: int,
    times: np.ndarray,
    bounds: Mapping[str, Any],
    contract: Mapping[str, Any],
    radius_rad: float,
    tier: str,
) -> tuple[tuple[Any, ...], np.ndarray, dict[str, Any]] | None:
    failure_rows, _ = h13.base.process_validation(moveit, primitive, working[index:index + 1], np.asarray([times[index]]), contract)
    reference = failure_rows[0]
    target_position = np.asarray(reference.get("reference_tcp_position_xyz_m"), dtype=float)
    target_orientation = np.asarray(reference.get("reference_tcp_orientation_xyzw"), dtype=float)
    state = r4.RobotState(moveit.get_robot_model())
    lower = np.asarray([bounds[name]["position_lower_rad"] for name in r4.JOINT_NAMES], dtype=float)
    upper = np.asarray([bounds[name]["position_upper_rad"] for name in r4.JOINT_NAMES], dtype=float)
    current = working[index].copy()
    candidates: list[tuple[tuple[Any, ...], np.ndarray, dict[str, Any]]] = []
    for seed_order, (seed_name, seed) in enumerate(_candidate_seeds(working, index, left, right, tier)):
        state.set_joint_group_positions(seed.tolist())
        state.update()
        try:
            solved = bool(state.set_from_ik(r4.base.GROUP_NAME, r4._pose(target_position, target_orientation), r4.base.EE_LINK, 0.02))
        except Exception as exc:
            solved = False
            error = f"{type(exc).__name__}:{exc}"
        else:
            error = None
        if not solved:
            continue
        state.update()
        candidate = np.asarray(state.get_joint_group_positions(r4.base.GROUP_NAME), dtype=float)
        delta = float(np.linalg.norm(candidate - current))
        joint_valid = bool(np.all(np.isfinite(candidate)) and np.all(candidate >= lower - 1.0e-10) and np.all(candidate <= upper + 1.0e-10))
        if not joint_valid or delta > radius_rad + 1.0e-10:
            continue
        process_rows, _ = h13.base.process_validation(moveit, primitive, candidate[None, :], np.asarray([times[index]]), contract)
        process = process_rows[0]
        collision_rows, collision_summary = h13.base.collision_validate(moveit, np.asarray([times[index]]), candidate[None, :])
        collision_free = bool(collision_summary.get("status") == "PASSED")
        if process.get("process_tolerance_pass") is not True or not collision_free:
            continue
        continuity = 0.0
        if index > 0:
            continuity += float(np.linalg.norm(candidate - working[index - 1]))
        if index + 1 < len(working):
            continuity += float(np.linalg.norm(working[index + 1] - candidate))
        score = (delta, continuity, seed_order, tuple(float(value) for value in candidate))
        candidates.append((score, candidate, {"seed": seed_name, "delta_rad": delta, "process": process, "collision_free": collision_free, "ik_error": error}))
    if not candidates:
        return None
    return min(candidates, key=lambda row: row[0])


def _whole_trajectory_valid(moveit: Any, primitive: Mapping[str, Any], q: np.ndarray, times: np.ndarray, bounds: Mapping[str, Any], contract: Mapping[str, Any]) -> tuple[bool, dict[str, Any], dict[str, Any]]:
    lower = np.asarray([bounds[name]["position_lower_rad"] for name in r4.JOINT_NAMES], dtype=float)
    upper = np.asarray([bounds[name]["position_upper_rad"] for name in r4.JOINT_NAMES], dtype=float)
    position_ok = bool(np.all(q >= lower[None, :] - 1.0e-10) and np.all(q <= upper[None, :] + 1.0e-10))
    process_rows, process = h13.base.process_validation(moveit, primitive, q, times, contract)
    collision_rows, collision = h13.base.collision_validate(moveit, times, q)
    return bool(position_ok and process.get("status") == "PASSED" and collision.get("status") == "PASSED"), process, collision


def _repair_segment(
    moveit: Any,
    primitive: Mapping[str, Any],
    item: Mapping[str, Any],
    working: np.ndarray,
    segment: tuple[int, int],
    times: np.ndarray,
    bounds: Mapping[str, Any],
    contract: Mapping[str, Any],
    tier: str,
    radius_rad: float,
    context: int,
) -> tuple[np.ndarray, list[dict[str, Any]]] | None:
    left, right = segment
    if right - left + 1 > DEFAULT_MAX_SEGMENT_WAYPOINTS:
        return None
    context_left, context_right = expand_range(segment, len(working), context)
    candidate = working.copy()
    events: list[dict[str, Any]] = []
    for index in range(left, right + 1):
        solved = _try_ik_candidate(moveit, primitive, item, candidate, index, context_left, context_right, times, bounds, contract, radius_rad, tier)
        if solved is None:
            return None
        _, q_after, details = solved
        q_before = candidate[index].copy()
        candidate[index] = q_after
        events.append({"trajectory_index": index, "q_before": q_before.tolist(), "q_after": q_after.tolist(), "selected_seed": details["seed"], "joint_delta_norm": details["delta_rad"], "tier": tier, "radius_rad": radius_rad})
    valid, process, collision = _whole_trajectory_valid(moveit, primitive, candidate, times, bounds, contract)
    if not valid:
        return None
    return candidate, events


def expanded_local_tcp_reprojection(
    moveit: Any,
    primitive: Mapping[str, Any],
    item: Mapping[str, Any],
    positions: np.ndarray,
    times: np.ndarray,
    process_rows: list[Mapping[str, Any]],
    bounds: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    """Run Tier 0 then the bounded Tier 1/2/3 ladder."""

    original = np.asarray(positions, dtype=float)
    working, tier0, tier0_events = _ORIGINAL_LOCAL_REPROJECTION(moveit, primitive, item, original, times, process_rows, bounds, contract)
    failures = [int(row.get("trajectory_index")) for row in tier0.get("repair_failures", [])]
    all_events = list(tier0_events)
    tier_attempts: list[dict[str, Any]] = [{"tier": TIER0, "accepted": not failures, "eligible_count": int(tier0.get("eligible_count", 0)), "repaired_count": int(tier0.get("repaired_count", 0))}]
    selected_tiers: list[str] = [TIER0] if not failures and tier0.get("eligible_count", 0) else []
    if failures:
        segments = contiguous_ranges(failures)
        for segment in segments:
            accepted = False
            if segment[1] - segment[0] + 1 > DEFAULT_MAX_SEGMENT_WAYPOINTS:
                for skipped_tier in (TIER1, TIER2, TIER3):
                    tier_attempts.append({"tier": skipped_tier, "segment": list(segment), "accepted": False, "reason": "bounded_segment_domain_exceeded"})
                continue
            for radius in radius_ladder(DEFAULT_RADIUS_LADDER_RAD[0], maximum_rad=DEFAULT_RADIUS_LADDER_RAD[-1]):
                result = _repair_segment(moveit, primitive, item, working, segment, times, bounds, contract, TIER1, radius, DEFAULT_CONTEXT_WINDOW)
                tier_attempts.append({"tier": TIER1, "segment": list(segment), "context_window": DEFAULT_CONTEXT_WINDOW, "radius_rad": radius, "accepted": result is not None})
                if result is not None:
                    working, events = result
                    all_events.extend(events)
                    selected_tiers.append(TIER1)
                    accepted = True
                    break
            if accepted:
                continue
            for radius in radius_ladder(DEFAULT_RADIUS_LADDER_RAD[0], maximum_rad=DEFAULT_RADIUS_LADDER_RAD[-1]):
                result = _repair_segment(moveit, primitive, item, working, segment, times, bounds, contract, TIER2, radius, DEFAULT_CONTEXT_WINDOW)
                tier_attempts.append({"tier": TIER2, "segment": list(segment), "context_window": DEFAULT_CONTEXT_WINDOW, "radius_rad": radius, "accepted": result is not None})
                if result is not None:
                    working, events = result
                    all_events.extend(events)
                    selected_tiers.append(TIER2)
                    accepted = True
                    break
            if accepted:
                continue
            for radius in radius_ladder(DEFAULT_RADIUS_LADDER_RAD[0], maximum_rad=DEFAULT_RADIUS_LADDER_RAD[-1]):
                result = _repair_segment(moveit, primitive, item, working, segment, times, bounds, contract, TIER3, radius, 0)
                tier_attempts.append({"tier": TIER3, "segment": list(segment), "context_window": 0, "radius_rad": radius, "accepted": result is not None})
                if result is not None:
                    working, events = result
                    all_events.extend(events)
                    selected_tiers.append(TIER3)
                    accepted = True
                    break
            if not accepted:
                tier_attempts.append({"tier": "REJECTED", "segment": list(segment), "accepted": False, "reason": "bounded_repair_domain_exceeded"})
    indices = [int(row["trajectory_index"]) for row in all_events]
    correction = correction_summary(original, working)
    lower = np.asarray([bounds[name]["position_lower_rad"] for name in r4.JOINT_NAMES], dtype=float)
    upper = np.asarray([bounds[name]["position_upper_rad"] for name in r4.JOINT_NAMES], dtype=float)
    post_position_violations = int(np.count_nonzero((working < lower[None, :] - 1.0e-10) | (working > upper[None, :] + 1.0e-10)))
    _, post_process, post_collision = _whole_trajectory_valid(moveit, primitive, working, times, bounds, contract)
    immutable_before = r4.immutable_sample_hash(original, indices)
    immutable_after = r4.immutable_sample_hash(working, indices)
    failures_remaining = [index for index in failures if index not in indices]
    summary = {
        "schema_version": "stage3_h13_r4_reprojection_summary_v1",
        "repair_layer": "POST_TOTG_PRE_RUCKIG",
        "eligible_count": int(tier0.get("eligible_count", 0)),
        "repaired_count": len(all_events),
        "repair_failure_count": len(failures_remaining),
        "repair_failures": [{"trajectory_index": index, "attempts": []} for index in failures_remaining],
        "max_tcp_error_before_m": float(tier0.get("max_tcp_error_before_m", 0.0)),
        "max_joint_correction_rad": correction["max_joint_correction_rad"],
        "rms_joint_correction_rad": correction["rms_joint_correction_rad"],
        "affected_waypoint_count": correction["affected_waypoint_count"],
        "affected_trajectory_fraction": correction["affected_trajectory_fraction"],
        "post_reprojection_position_violations": post_position_violations,
        "post_reprojection_collision_violations": int(post_collision.get("collision_failure_count", 0) or 0),
        "post_reprojection_spray_process_violations": int(post_process.get("failed_spray_on_sample_count", 0) or 0),
        "immutable_sample_hash_before": immutable_before,
        "immutable_sample_hash_after": immutable_after,
        "immutable_samples_unchanged": immutable_before == immutable_after,
        "tier_attempts": tier_attempts,
        "selected_tiers": selected_tiers,
        "tier0_accepted": not failures,
        "no_label_repair_audit": no_label_repair_audit(),
    }
    return working, summary, all_events


def _root_cause(raw: Mapping[str, Any], q: np.ndarray, bounds: Mapping[str, Any], process_rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    lower = [bounds[name]["position_lower_rad"] for name in h13.JOINT_NAMES]
    upper = [bounds[name]["position_upper_rad"] for name in h13.JOINT_NAMES]
    norm_profile, signed = signed_limit_correction(q, lower, upper)
    position_indices = np.flatnonzero(norm_profile > 0.0).astype(int).tolist()
    spray_rows = [row for row in process_rows if row.get("process_tolerance_pass") is False]
    spray_indices = [int(row["trajectory_index"]) for row in spray_rows]
    union = sorted(set(position_indices) | set(spray_indices))
    counts = {"standoff": 0, "normal_angle": 0, "tcp_position": 0, "tcp_orientation": 0}
    max_ratio = {key: 0.0 for key in counts}
    for row in spray_rows:
        checks = row.get("process_tolerance_checks") or {}
        if not checks:
            checks = {
                "standoff": {"pass": float(row.get("standoff_error_m", 0.0)) <= float(row.get("standoff_limit_m", 0.0))},
                "normal_angle": {"pass": float(row.get("normal_deviation_deg", 0.0)) <= float(row.get("normal_limit_deg", 0.0))},
                "tcp_position": {"pass": float(row.get("tcp_position_error_m", 0.0)) <= float(row.get("tcp_position_limit_m", 0.0))},
                "tcp_orientation": {"pass": float(row.get("tcp_orientation_error_deg", 0.0)) <= float(row.get("tcp_orientation_limit_deg", 0.0))},
            }
        aliases = {"surface_normal": "normal_angle", "normal_angle": "normal_angle", "tcp_position": "tcp_position", "tcp_orientation": "tcp_orientation", "standoff": "standoff"}
        metric_fields = {
            "standoff": ("standoff_error_m", "absolute_standoff_error_m", "standoff_limit_m", "tolerance_m"),
            "normal_angle": ("normal_deviation_deg", "normal_angle_deviation_deg", "normal_limit_deg", "tolerance_deg"),
            "tcp_position": ("tcp_position_error_m", "tcp_position_error_m", "tcp_position_limit_m", "tolerance_m"),
            "tcp_orientation": ("tcp_orientation_error_deg", "tcp_orientation_error_deg", "tcp_orientation_limit_deg", "tolerance_deg"),
        }
        for key, check in checks.items():
            canonical_key = aliases.get(key)
            if canonical_key not in metric_fields:
                continue
            value_field, alternate_value_field, alternate_limit_field, check_limit_field = metric_fields[canonical_key]
            value = float(check.get(value_field, check.get(alternate_value_field, row.get(value_field, row.get(alternate_value_field, 0.0)))))
            limit = float(check.get(check_limit_field, row.get(alternate_limit_field, 1.0)))
            if check.get("pass") is not True:
                counts[canonical_key] += 1
            max_ratio[canonical_key] = max(max_ratio[canonical_key], value / max(limit, 1.0e-12))
    signed_totals = np.sum(signed, axis=0)
    abs_total = float(np.sum(np.abs(signed_totals)))
    dominant = float(np.max(np.abs(signed_totals)) / abs_total) if abs_total else 0.0
    global_bias = bool(dominant >= 0.8 and np.count_nonzero(signed_totals) == 1)
    profile = {
        "position_violating_waypoint_count": len(position_indices),
        "spray_process_violating_waypoint_count": len(spray_indices),
        "contiguous_violating_segments": [list(item) for item in contiguous_ranges(union)],
        "maximum_joint_space_displacement_required_rad": float(np.max(norm_profile)) if norm_profile.size else 0.0,
        "rms_joint_space_displacement_required_rad": float(np.sqrt(np.mean(np.square(norm_profile[norm_profile > 0.0])))) if np.any(norm_profile > 0.0) else 0.0,
        "maximum_position_limit_violation_rad": float(np.max(np.abs(signed))) if signed.size else 0.0,
        "spray_subconstraint_violation_counts": counts,
        "spray_subconstraint_max_normalized_magnitude": max_ratio,
        "position_spray_spatial_overlap_count": len(set(position_indices) & set(spray_indices)),
        "position_spray_spatial_jaccard": float(len(set(position_indices) & set(spray_indices)) / max(1, len(set(position_indices) | set(spray_indices)))),
        "global_bias_evidence": global_bias,
        "correction_structure": "global_bias" if global_bias else ("full_trajectory_shape_deformation" if len(union) >= max(1, int(0.8 * len(q))) else ("local_or_segment_shape" if union else "none")),
        "tunnel_transition_cluster": "not_applicable_single_on_open_arch_primitive",
        "raw_trajectory_collision_free": int((raw.get("collision") or {}).get("collision_failure_count", 0) or 0) == 0,
    }
    profile["classification"] = classify_failure(profile)
    return {"schema_version": "stage3_h13_r4_root_cause_case_v1", **profile}


def raw_audit_with_root_cause(moveit: Any, primitive: Mapping[str, Any], q: np.ndarray, t: np.ndarray, baseline_q: np.ndarray, limits: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    raw = _ORIGINAL_RAW_AUDIT(moveit, primitive, q, t, baseline_q, limits, contract)
    process_rows, _ = h13.base.process_validation(moveit, primitive, q, t - t[0], contract)
    raw["root_cause"] = _root_cause(raw, q, limits, process_rows)
    return raw


def compact_record_with_r4(
    unit: Mapping[str, Any],
    record: Mapping[str, Any],
    raw: Mapping[str, Any],
    corrections: Mapping[str, Any],
) -> dict[str, Any]:
    compact = _ORIGINAL_COMPACT_RECORD(unit, record, raw, corrections)
    repair = record.get("h12_r7_reprojection") or {}
    compact["repair"].update({
        "RMS_JOINT_CORRECTION_RAD": repair.get("rms_joint_correction_rad"),
        "AFFECTED_WAYPOINT_COUNT": repair.get("affected_waypoint_count"),
        "AFFECTED_TRAJECTORY_FRACTION": repair.get("affected_trajectory_fraction"),
        "POST_REPROJECTION_POSITION_VIOLATIONS": repair.get("post_reprojection_position_violations"),
        "POST_REPROJECTION_COLLISION_VIOLATIONS": repair.get("post_reprojection_collision_violations"),
        "POST_REPROJECTION_SPRAY_PROCESS_VIOLATIONS": repair.get("post_reprojection_spray_process_violations"),
    })
    compact["reprojection"].update({
        "tier_attempts": repair.get("tier_attempts", []),
        "selected_tiers": repair.get("selected_tiers", []),
        "tier0_accepted": repair.get("tier0_accepted"),
        "UNSEEN_LABEL_USED_FOR_REPAIR": "NO",
        "FUTURE_LABEL_LEAKAGE": 0,
    })
    compact["root_cause"] = raw.get("root_cause")
    return compact


# Patch only the imported module objects used by h13.run; the historical
# modules remain byte-for-byte unchanged on disk.
r4.localized_tcp_reprojection = expanded_local_tcp_reprojection
collision_base.collision_state = _safe_collision_state
h13.raw_audit = raw_audit_with_root_cause
h13.compact_record = compact_record_with_r4


if __name__ == "__main__":
    raise SystemExit(h13.main())
