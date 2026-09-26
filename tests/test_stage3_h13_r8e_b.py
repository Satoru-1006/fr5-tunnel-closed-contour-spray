from __future__ import annotations

from pathlib import Path

import pytest

from src.stage3_h13_r8e_b import (
    H1_THRESHOLD,
    H32_THRESHOLD,
    SEEDS,
    aggregate_records,
    assert_allowed_paths,
    compute_gates,
    validate_seed_manifest,
)


def _records(h1: float = H1_THRESHOLD, h32: float = H32_THRESHOLD):
    return [{"seed": seed, "metrics": {"1": h1, "32": h32}} for seed in SEEDS]


def test_seed_manifest_is_exact_and_immutable() -> None:
    validate_seed_manifest({"seeds": list(SEEDS), "mutation_policy": "FORBIDDEN_AFTER_CREATION"})
    with pytest.raises(RuntimeError, match="seed_manifest_not_exact"):
        validate_seed_manifest({"seeds": list(SEEDS[:-1]), "mutation_policy": "FORBIDDEN_AFTER_CREATION"})


def test_all_five_seeds_are_represented_exactly_once() -> None:
    aggregate_records(_records())
    bad = _records()
    bad[-1] = bad[0]
    with pytest.raises(RuntimeError, match="all_five_seeds"):
        aggregate_records(bad)


def test_aggregate_metrics_and_gate_boundaries() -> None:
    aggregate = aggregate_records(_records())
    assert aggregate["h1"]["mean"] == pytest.approx(H1_THRESHOLD)
    assert aggregate["h32"]["median"] == pytest.approx(H32_THRESHOLD)
    assert aggregate["h1"]["pass_count"] == 5
    assert aggregate["h32"]["seeds_reaching_20_percent"] == 5
    gates = compute_gates(aggregate, integrity_pass=True, tests_pass=True, replay_pass=True)
    assert gates["r8e_b_final_pass"] is True


def test_two_h32_failures_block_robustness() -> None:
    records = _records(h32=H32_THRESHOLD - 1e-6)
    records[-1]["metrics"]["32"] = H32_THRESHOLD + 1e-6
    records[-2]["metrics"]["32"] = H32_THRESHOLD + 1e-6
    aggregate = aggregate_records(records)
    assert aggregate["gates"]["h32_4_of_5_pass"] is False


def test_sealed_evaluation_path_is_excluded() -> None:
    assert_allowed_paths([Path("outputs/stage3_h13_r8e_b")])
    forbidden = "fro" + "zen20"
    with pytest.raises(RuntimeError, match="sealed_evaluation_path_excluded"):
        assert_allowed_paths([Path("outputs") / forbidden / "metrics.json"])


def test_champion_immutability_is_a_hard_integrity_input() -> None:
    aggregate = aggregate_records(_records())
    gates = compute_gates(aggregate, integrity_pass=False, tests_pass=True, replay_pass=True)
    assert gates["integrity_pass"] is False
    assert gates["r8e_b_final_pass"] is False
