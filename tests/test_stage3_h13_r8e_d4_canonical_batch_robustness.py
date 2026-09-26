from __future__ import annotations

import math

import pytest

from src.stage3_h13_r8e_d4 import (
    CHAMPION_MEDIAN_H32_LIMIT,
    FROZEN_SEEDS,
    H1_THRESHOLD,
    H32_THRESHOLD,
    assert_frozen20_excluded,
    build_canonical_schedule,
    champion_replacement_pass,
    h1_pass,
    h32_pass,
    metric_spread,
    schedule_indices,
    validate_frozen_seeds,
)


def test_canonical_schedule_generation_and_seed_independence() -> None:
    first = build_canonical_schedule(range(11), batch_size=4, update_count=7)
    second = build_canonical_schedule(range(11), batch_size=4, update_count=7)
    assert first == second
    assert first["number_of_batches"] == 7
    assert all(len(batch["window_ids"]) == 4 for batch in first["ordered_batches"])
    assert first["prospective_seed_consumed"] == "NO"
    assert first["canonical_schedule_seed_independent"] == "YES"
    # Changing a prospective seed is not an input to schedule generation.
    assert schedule_indices(first, list(range(11))).shape == (7, 4)


def test_schedule_identity_is_deterministic_and_tail_crosses_cycles() -> None:
    manifest = build_canonical_schedule(range(5), batch_size=4, update_count=3)
    assert manifest["cycle_count_used"] == 3
    assert manifest["ordered_batches"][1]["source_cycle_spans"]
    assert manifest["ordered_batches"][1]["source_cycle_spans"][0]["count"] == 1
    assert manifest["ordered_batches"][1]["source_cycle_spans"][1]["count"] == 3


def test_frozen_seed_set_and_threshold_comparisons_are_exact() -> None:
    assert validate_frozen_seeds(FROZEN_SEEDS)["unique_seed_count"] == 5
    assert h1_pass(H1_THRESHOLD)
    assert not h1_pass(math.nextafter(H1_THRESHOLD, math.inf))
    assert h32_pass(H32_THRESHOLD)
    assert not h32_pass(math.nextafter(H32_THRESHOLD, math.inf))
    with pytest.raises(ValueError):
        validate_frozen_seeds((FROZEN_SEEDS[0], *FROZEN_SEEDS[:-1]))


def test_cross_seed_robustness_calculation_and_nonfinite_fail_closed() -> None:
    result = metric_spread([1.0, 1.0, 1.0, 1.0, 1.0])
    assert result["pass"] is True
    assert result["spread"] == 0.0
    assert metric_spread([1.0, 1.0, 1.0, 1.0, math.inf])["pass"] is False


def test_champion_replacement_requires_every_gate() -> None:
    kwargs = dict(
        completed_count=5,
        valid_count=5,
        h1_pass_count=5,
        h32_pass_count=5,
        h16_all_finite=True,
        h1_spread_pass=True,
        h16_spread_pass=True,
        h32_spread_pass=True,
        median_h32=CHAMPION_MEDIAN_H32_LIMIT,
        integrity_pass=True,
        leakage_pass=True,
        schedule_pass=True,
        checkpoint_pass=True,
        no_seed_replacement=True,
        no_post_hoc_change=True,
        frozen20_excluded=True,
    )
    assert champion_replacement_pass(**kwargs)
    assert not champion_replacement_pass(**{**kwargs, "h1_pass_count": 4})
    assert not champion_replacement_pass(**{**kwargs, "median_h32": math.nextafter(CHAMPION_MEDIAN_H32_LIMIT, math.inf)})


def test_missing_run_and_frozen20_access_fail_closed() -> None:
    kwargs = dict(
        completed_count=4,
        valid_count=4,
        h1_pass_count=4,
        h32_pass_count=4,
        h16_all_finite=True,
        h1_spread_pass=True,
        h16_spread_pass=True,
        h32_spread_pass=True,
        median_h32=0.01,
        integrity_pass=True,
        leakage_pass=True,
        schedule_pass=True,
        checkpoint_pass=True,
        no_seed_replacement=True,
        no_post_hoc_change=True,
        frozen20_excluded=True,
    )
    assert champion_replacement_pass(**kwargs) is False
    with pytest.raises(ValueError):
        assert_frozen20_excluded(["outputs/stage3_h13_r3_frozen20_native_certification"])
    assert_frozen20_excluded(["outputs/stage3_h13_r8e_d4_canonical_batch_robustness"])
