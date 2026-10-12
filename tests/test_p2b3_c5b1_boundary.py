"""Synthetic known-answer and adversarial tests for bounded C5B1 ray search.

Every callback is a fixture, not FR5 FCL evidence.
"""
import math
import pytest

from src.p2b3_c5b1_boundary import (
    PREDICATES, classify, ray_transform, rays, scan_ray, scan_all, DEFAULT_LIMITS,
)
from scripts.run_p2b3_c5b1_boundary import fixture_evaluator


def known(m, collision=False):
    d = -0.001 if collision else 0.08
    return dict(
        robot_world_collision_coverage=True, self_collision_coverage=True,
        robot_world_distance_m=d, robot_world_collision_samples=int(collision),
        self_distance_m=0.016, self_collision_samples=0,
        max_position_error_m=m, max_normal_error_deg=1.0,
        critical_waypoint=86, critical_segment="85->86",
    )


def test_twelve_signed_base_left_rays_and_units():
    assert len(rays()) == 12
    assert len(set(rays())) == 12
    for axis, sign in rays():
        result = ray_transform(axis, sign, 0.01)
        assert result["registration_convention"] == "BASE_LEFT"
        vector = (result["translation_base_m"] if axis.startswith("t")
                  else result["rotation_vector_base_rad"])
        assert sum(v != 0 for v in vector) == 1
        assert math.isclose(sum(abs(v) for v in vector), 0.01)


def test_invalid_rays_fail_closed():
    for args in [("j6", 1, 0.01), ("tx", 0, 0.01), ("rx", 1, float("nan")),
                 ("ty", -1, -0.1)]:
        with pytest.raises(ValueError):
            ray_transform(*args)


def test_distance_status_without_full_coverage_is_unknown():
    p = known(0.0)
    p["robot_world_collision_coverage"] = False
    assert classify(p)["robot_world_collision"] == "UNKNOWN"


def test_distance_collision_count_inconsistent_is_unknown():
    p = known(0.0)
    p["robot_world_collision_samples"] = 1
    assert classify(p)["robot_world_collision"] == "UNKNOWN"


def test_process_engineering_thresholds_separate_from_collision():
    p = known(0.006)
    p["max_normal_error_deg"] = 10.0
    x = classify(p)
    assert all(x[k] == "PASS" for k in PREDICATES)
    p["max_position_error_m"] = 0.00601
    p["max_normal_error_deg"] = 10.01
    x = classify(p)
    assert x["position_diagnostic"] == "FAIL"
    assert x["normal_diagnostic"] == "FAIL"
    assert x["robot_world_collision"] == "PASS"


def test_first_failure_boundary_bracket_refined():
    def callback(axis, sign, magnitude):
        return known(magnitude, collision=magnitude >= 0.0032)
    r = scan_ray(callback, axis="tx", sign=1, upper=0.01, coarse_steps=10, tolerance=1e-6)
    p = r["predicates"]["robot_world_collision"]
    assert p["status"] == "OBSERVED_FAILURE_BRACKETS"
    assert p["first_observed_entry"]["a"] <= 0.0032 <= p["first_observed_entry"]["b"]
    assert p["first_observed_entry"]["width"] <= 1e-6
    assert r["evaluation_count"] <= 100


def test_nonmonotonic_all_entry_exit_intervals_refined():
    def callback(axis, sign, magnitude):
        collision = 0.002 <= magnitude < 0.004 or 0.006 <= magnitude < 0.008
        return known(0.0, collision=collision)
    r = scan_ray(callback, axis="ty", sign=-1, upper=0.010, coarse_steps=20, tolerance=1e-6)
    t = r["predicates"]["robot_world_collision"]["transitions"]
    assert [x["from"] + "->" + x["to"] for x in t] == [
        "PASS->FAIL", "FAIL->PASS", "PASS->FAIL", "FAIL->PASS"
    ]


def test_unknown_endpoint_never_bridges_and_is_not_success():
    def callback(axis, sign, magnitude):
        if 0.004 <= magnitude <= 0.006:
            raise RuntimeError("native timeout")
        return known(0.0, collision=magnitude >= 0.007)
    r = scan_ray(callback, axis="tx", sign=1, upper=0.01, coarse_steps=10)
    p = r["predicates"]["robot_world_collision"]
    assert p["status"] == "INDETERMINATE_WITHIN_DOMAIN"
    assert p["unresolved_intervals"]


def test_unknown_refinement_midpoint_blocks_false_boundary():
    def callback(axis, sign, magnitude):
        if magnitude == 0.005:
            raise RuntimeError("partial native failure")
        return known(0.0, collision=magnitude > 0.005)
    r = scan_ray(callback, axis="tx", sign=1, upper=0.01, coarse_steps=3)
    p = r["predicates"]["robot_world_collision"]
    assert p["status"] == "INDETERMINATE_WITHIN_DOMAIN"
    assert not p["transitions"]


def test_no_failure_is_bounded_not_infinite_margin():
    r = scan_ray(lambda a, s, m: known(0.0), axis="rz", sign=1, upper=0.2)
    p = r["predicates"]["robot_world_collision"]
    assert p["status"] == "NO_FAILURE_OBSERVED_WITHIN_DOMAIN"
    assert p["first_observed_entry"] is None
    assert r["domain"] == [0.0, 0.2]


def test_initial_failure_not_relabelled_boundary():
    r = scan_ray(lambda a, s, m: known(0.0, collision=True), axis="tx", sign=-1, upper=0.01)
    p = r["predicates"]["robot_world_collision"]
    assert p["status"] == "BASELINE_FAIL"


def test_budget_exhaustion_remains_unresolved():
    def callback(a, s, m):
        return known(0.0, collision=m >= 0.005)
    r = scan_ray(callback, axis="tx", sign=1, upper=0.01,
                 coarse_steps=3, tolerance=1e-12, max_evaluations=4)
    p = r["predicates"]["robot_world_collision"]
    assert p["status"] == "INDETERMINATE_WITHIN_DOMAIN"
    assert p["unresolved_intervals"]


def test_pilot_synthetic_four_rays_then_all_twelve():
    pilot = scan_all(fixture_evaluator, limits=DEFAULT_LIMITS["pilot"], coarse_steps=8,
                     selection=(("tx", -1), ("tx", 1), ("rx", -1), ("rx", 1)))
    all_axes = scan_all(fixture_evaluator, limits=DEFAULT_LIMITS["all"], coarse_steps=8)
    assert pilot["ray_count"] == 4
    assert all_axes["ray_count"] == 12
    assert all(set(ray["predicates"]) == set(PREDICATES) for ray in all_axes["rays"])
    assert all(ray["evaluation_count"] <= 100 for ray in all_axes["rays"])
    assert all("infinite" not in str(ray).lower() for ray in all_axes["rays"])


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.001, 0.0])
def test_upper_bounds_checked(bad):
    with pytest.raises(ValueError):
        scan_ray(lambda a, s, m: known(0.0), axis="tx", sign=1, upper=bad)
