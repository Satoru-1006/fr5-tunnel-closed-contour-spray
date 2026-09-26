import numpy as np
import pytest

from src.stage3_h13_r4_reprojection import (
    TIER0,
    TIER1,
    affected_fraction,
    choose_tier,
    classify_failure,
    contiguous_ranges,
    correction_summary,
    expand_range,
    no_label_repair_audit,
    radius_ladder,
    signed_limit_correction,
    violation_profile,
)


def test_contiguous_violation_segments_and_context_are_deterministic():
    assert contiguous_ranges([5, 2, 3, 9, 10, 12]) == [(2, 3), (5, 5), (9, 10), (12, 12)]
    assert expand_range((2, 3), 8, context=2) == (0, 5)
    assert expand_range((2, 3), 8, context=20) == (0, 7)


def test_adaptive_radius_ladder_is_smallest_first_and_bounded():
    assert radius_ladder(0.02, maximum_rad=0.08) == [0.02, 0.03, 0.04, 0.06, 0.08]
    assert radius_ladder(0.02, factors=(1.0, 1.0, 2.0), maximum_rad=0.05) == [0.02, 0.04, 0.05]


def test_violation_profile_and_failure_classification_are_compact():
    q = np.asarray([[0.0, 0.0], [1.01, 0.0], [1.02, 0.0], [0.0, 0.0]])
    profile = violation_profile(q, [-1.0, -1.0], [1.0, 1.0])
    assert profile["violating_waypoint_count"] == 2
    assert profile["contiguous_segments"] == [[1, 2]]
    combined = {"position_violating_waypoint_count": 2, "spray_process_violating_waypoint_count": 2, **profile}
    assert classify_failure(combined) == "F.mixed_position_spray_violation"


def test_correction_summary_preserves_unaffected_regions():
    before = np.zeros((4, 2))
    after = before.copy()
    after[1] = [0.01, 0.02]
    result = correction_summary(before, after)
    assert result["affected_waypoint_indices"] == [1]
    assert result["affected_trajectory_fraction"] == pytest.approx(0.25)
    assert np.array_equal(before[[0, 2, 3]], after[[0, 2, 3]])


def test_tier_selection_is_lexicographic_and_label_free():
    attempts = [{"tier": TIER1, "accepted": True}, {"tier": TIER0, "accepted": True}]
    assert choose_tier(attempts) == TIER0
    assert no_label_repair_audit() == {"UNSEEN_LABEL_USED_FOR_REPAIR": "NO", "FUTURE_LABEL_LEAKAGE": 0}


def test_hard_limit_correction_is_measured_without_clipping():
    q = np.asarray([[0.0, 1.2], [0.0, 0.0]])
    norms, signed = signed_limit_correction(q, [-1.0, -1.0], [1.0, 1.0])
    assert norms.tolist() == pytest.approx([0.2, 0.0])
    assert signed[0].tolist() == pytest.approx([0.0, -0.2])
