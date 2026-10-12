"""Bounded, tri-state, non-monotonic FR5 registration-ray boundary search.

Search semantics are finite-domain and sampling-based. No global-optimum,
continuous-collision, coating-quality, or physical safety guarantee is made.
A callback MUST supply genuine evaluator observations, not guessed statuses.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Mapping, Any

PREDICATES = (
    "robot_world_collision",
    "self_collision",
    "position_diagnostic",
    "normal_diagnostic",
)
STATES = {"PASS", "FAIL", "UNKNOWN"}
AXES = ("tx", "ty", "tz", "rx", "ry", "rz")
POSITION_GATE_M = 0.006            # engineering diagnostic, NOT coating acceptance
NORMAL_GATE_DEG = 10.0            # engineering diagnostic, NOT coating acceptance
DEFAULT_LIMITS = {
    "pilot": {"translation_m": 0.001, "rotation_rad": math.radians(0.1)},
    "all": {"translation_m": 0.020, "rotation_rad": math.radians(2.0)},
}


def rays() -> list[tuple[str, int]]:
    return [(axis, sign) for axis in AXES for sign in (-1, 1)]


def ray_transform(axis: str, sign: int, magnitude: float) -> dict[str, Any]:
    if axis not in AXES or sign not in (-1, 1):
        raise ValueError("unsupported_ray")
    if not math.isfinite(magnitude) or magnitude < 0:
        raise ValueError("invalid_magnitude")
    index = "xyz".index(axis[1])
    translation = [0.0, 0.0, 0.0]
    rotation = [0.0, 0.0, 0.0]
    (translation if axis[0] == "t" else rotation)[index] = sign * magnitude
    return {
        "translation_base_m": translation,
        "rotation_vector_base_rad": rotation,
        "registration_convention": "BASE_LEFT",
        "rotation_origin": "base_link origin",
    }


def classify(payload: Mapping[str, Any]) -> dict[str, str]:
    """Require explicit coverage. Individual unavailable predicates -> UNKNOWN.

    Native collision counts override distance >0; distance <=0 means sampled
    model collision even when a count is omitted. Contradictory telemetry is
    UNKNOWN rather than silently accepting whichever is convenient.
    """
    statuses: dict[str, str] = {}
    for kind, distance_name, count_name in (
        ("robot_world_collision", "robot_world_distance_m", "robot_world_collision_samples"),
        ("self_collision", "self_distance_m", "self_collision_samples"),
    ):
        dist, count = payload.get(distance_name), payload.get(count_name)
        coverage = payload.get(kind + "_coverage")
        if coverage is not True or isinstance(count, bool) or not isinstance(count, int) or count < 0:
            statuses[kind] = "UNKNOWN"
            continue
        if not isinstance(dist, (int, float)) or isinstance(dist, bool) or not math.isfinite(dist):
            statuses[kind] = "UNKNOWN"
        elif (count > 0) != (dist <= 0):
            statuses[kind] = "UNKNOWN"
        else:
            statuses[kind] = "FAIL" if count > 0 else "PASS"
    for kind, key, gate in (
        ("position_diagnostic", "max_position_error_m", POSITION_GATE_M),
        ("normal_diagnostic", "max_normal_error_deg", NORMAL_GATE_DEG),
    ):
        value = payload.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
            statuses[kind] = "UNKNOWN"
        else:
            statuses[kind] = "FAIL" if value > gate else "PASS"
    return statuses


def _observation(
    callback: Callable[[str, int, float], Mapping[str, Any]],
    axis: str,
    sign: int,
    magnitude: float,
) -> dict[str, Any]:
    try:
        raw = callback(axis, sign, magnitude)
        if not isinstance(raw, Mapping):
            raise ValueError("evaluator_non_mapping")
        states = classify(raw)
        return {
            "magnitude": magnitude,
            "statuses": states,
            "critical_waypoint": raw.get("critical_waypoint"),
            "critical_segment": raw.get("critical_segment"),
            "failure_mode": raw.get("failure_mode"),
            "metrics": {k: raw.get(k) for k in (
                "robot_world_distance_m", "self_distance_m",
                "max_position_error_m", "max_normal_error_deg"
            )},
            "backend_status": raw.get("backend_status", "UNSPECIFIED"),
            "error": raw.get("error"),
        }
    except Exception as exc:
        return {
            "magnitude": magnitude,
            "statuses": {name: "UNKNOWN" for name in PREDICATES},
            "critical_waypoint": None,
            "critical_segment": None,
            "failure_mode": None,
            "metrics": {},
            "backend_status": "ERROR",
            "error": f"{type(exc).__name__}: {exc}"[:400],
        }


def scan_ray(
    callback: Callable[[str, int, float], Mapping[str, Any]],
    *,
    axis: str,
    sign: int,
    upper: float,
    coarse_steps: int = 8,
    tolerance: float = 1e-5,
    max_evaluations: int = 100,
) -> dict[str, Any]:
    """Refine EVERY adjacent PASS<->FAIL bracket per predicate, not just the first.

    Unknown edges never bridge. "NO_FAILURE_OBSERVED" is a finite sampled
    observation, not proof that unobserved interiors have no failures.
    """
    ray_transform(axis, sign, 0.0)
    if not math.isfinite(upper) or upper <= 0:
        raise ValueError("positive_finite_upper_required")
    if not isinstance(coarse_steps, int) or coarse_steps < 2 or not isinstance(max_evaluations, int):
        raise ValueError("invalid_search_budget")
    if max_evaluations < coarse_steps + 1 or tolerance <= 0 or not math.isfinite(tolerance):
        raise ValueError("insufficient_budget_or_invalid_tolerance")
    cache: dict[float, dict[str, Any]] = {}

    def at(value: float) -> dict[str, Any] | None:
        key = round(value, 15)
        if key not in cache:
            if len(cache) >= max_evaluations:
                return None
            cache[key] = _observation(callback, axis, sign, key)
        return cache[key]

    coarse = [at(upper * i / coarse_steps) for i in range(coarse_steps + 1)]
    assert all(o is not None for o in coarse)
    summaries = {}
    for predicate in PREDICATES:
        transitions = []
        unresolved = []
        statuses = [o["statuses"][predicate] for o in coarse]
        for i in range(coarse_steps):
            lo, hi = coarse[i], coarse[i+1]
            low_status, high_status = statuses[i], statuses[i+1]
            if low_status == "UNKNOWN" or high_status == "UNKNOWN":
                unresolved.append({"a": lo["magnitude"], "b": hi["magnitude"], "reason": "UNKNOWN_ENDPOINT"})
                continue
            if low_status == high_status:
                continue
            # For a non-monotonic ray both PASS->FAIL and FAIL->PASS are retained.
            a, b, sa, sb = lo["magnitude"], hi["magnitude"], low_status, high_status
            while b-a > tolerance:
                middle = at((a+b)/2)
                if middle is None:
                    unresolved.append({"a": a, "b": b, "reason": "EVALUATION_BUDGET_EXHAUSTED"})
                    break
                sm = middle["statuses"][predicate]
                if sm == "UNKNOWN":
                    unresolved.append({"a": a, "b": b, "reason": "UNKNOWN_INTERIOR"})
                    break
                if sm == sa:
                    a = middle["magnitude"]
                else:
                    b = middle["magnitude"]
            else:
                transitions.append({
                    "from": low_status, "to": high_status,
                    "a": a, "b": b, "width": b-a,
                    "critical_waypoint": cache[round(b if high_status == "FAIL" else a, 15)]["critical_waypoint"],
                    "critical_segment": cache[round(b if high_status == "FAIL" else a, 15)]["critical_segment"],
                    "failure_mode": cache[round(b if high_status == "FAIL" else a, 15)]["failure_mode"],
                })
                continue
            # Do not promote an unfinished bracket to a measured boundary.
        baseline = statuses[0]
        fail_to = [x for x in transitions if x["from"] == "PASS" and x["to"] == "FAIL"]
        any_fail = "FAIL" in statuses or bool(fail_to)
        if baseline == "UNKNOWN":
            verdict = "UNKNOWN_BASELINE"
        elif baseline == "FAIL":
            verdict = "BASELINE_FAIL"
        elif unresolved:
            verdict = "INDETERMINATE_WITHIN_DOMAIN"
        elif fail_to:
            verdict = "OBSERVED_FAILURE_BRACKETS"
        elif any_fail:
            verdict = "FAIL_OBSERVED_UNBRACKETED"
        else:
            verdict = "NO_FAILURE_OBSERVED_WITHIN_DOMAIN"
        summaries[predicate] = {
            "status": verdict,
            "transitions": transitions,
            "unresolved_intervals": unresolved,
            "first_observed_entry": min(fail_to, key=lambda x: x["b"]) if fail_to else None,
            "claim": "sampled finite-domain evidence; no global minimum or continuous guarantee",
        }
    return {
        "axis": axis, "sign": sign, "unit": "m" if axis[0] == "t" else "rad",
        "domain": [0.0, upper],
        "coarse_steps": coarse_steps,
        "refinement_tolerance": tolerance,
        "evaluation_count": len(cache),
        "coarse_observations": coarse,
        "predicates": summaries,
        "evidence_type": "DETERMINISTIC_SAMPLED_MODEL_OR_FIXTURE",
    }


def scan_all(
    callback: Callable[[str, int, float], Mapping[str, Any]],
    *,
    limits: Mapping[str, float],
    coarse_steps: int = 8,
    translation_tolerance_m: float = 1e-5,
    rotation_tolerance_rad: float = math.radians(0.001),
    max_evaluations_per_ray: int = 100,
    selection: tuple[tuple[str, int], ...] | None = None,
) -> dict[str, Any]:
    directions = selection if selection is not None else tuple(rays())
    records = [scan_ray(callback, axis=axis, sign=sign,
                        upper=limits["translation_m" if axis[0] == "t" else "rotation_rad"],
                        coarse_steps=coarse_steps,
                        tolerance=translation_tolerance_m if axis[0] == "t" else rotation_tolerance_rad,
                        max_evaluations=max_evaluations_per_ray)
               for axis, sign in directions]
    return {"ray_count": len(records), "rays": records,
            "search_semantics": "coarse sampled + all adjacent state transitions; unknown fail-closed",
            "physical_registration_margin": "NOT_AVAILABLE",
            "strict_self_ccd": "NOT_AVAILABLE"}
