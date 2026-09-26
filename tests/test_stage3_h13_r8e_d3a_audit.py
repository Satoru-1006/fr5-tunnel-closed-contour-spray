"""Focused, low-cost checks for the completed D3A evidence bundle."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts.stage3_h13_r8e_d3a_seed_218309645_model_selection_failure_audit import (
    EXPECTED_218309645_CANONICAL_SHA256,
    EXPECTED_218309645_RAW_SHA256,
    EXPECTED_13086_CANONICAL_SHA256,
    EXPECTED_13086_RAW_SHA256,
    HORIZONS,
    SEED_13086,
    SEED_218309645,
    selection_predicate,
)


def output_dir() -> Path:
    value = os.environ.get("D3A_OUTPUT")
    if not value:
        pytest.skip("D3A_OUTPUT is not set")
    path = Path(value)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def read(name: str):
    return json.loads((output_dir() / name).read_text(encoding="utf-8"))


def test_frozen_threshold_and_comparison_operator_are_reported() -> None:
    contract = read("model_selection_contract.json")
    assert contract["comparison_operator"] == "<="
    assert contract["threshold_type"] == "upper guardrail"
    threshold = contract["threshold_value"]
    assert selection_predicate(updates=512, phase_stability=[True, True, True, True], metrics={"1": threshold}, threshold=threshold)
    assert not selection_predicate(updates=512, phase_stability=[True, True, True, True], metrics={"1": threshold + 1.0e-12}, threshold=threshold)


def test_validation_split_and_checkpoint_hashes_are_authoritative() -> None:
    contract = read("model_selection_contract.json")
    assert contract["data_split"]["selection_split"] == "VALIDATION"
    assert contract["data_split"]["window_count"] == 25991
    assert contract["data_split"]["d2_h11_reference_window_count"] == 27594
    comparison = read("seed_selection_metric_comparison.json")
    assert comparison["model_hashes"]["prospective_seed_13086"]["raw_sha256"] == EXPECTED_13086_RAW_SHA256
    assert comparison["model_hashes"]["prospective_seed_13086"]["canonical_sha256"] == EXPECTED_13086_CANONICAL_SHA256
    assert comparison["model_hashes"]["prospective_seed_218309645"]["raw_sha256"] == EXPECTED_218309645_RAW_SHA256
    assert comparison["model_hashes"]["prospective_seed_218309645"]["canonical_sha256"] == EXPECTED_218309645_CANONICAL_SHA256


def test_replay_and_independent_recomputation_agree() -> None:
    replay = read("seed_218309645_replay_results.json")
    independent = read("independent_metric_recomputation.json")
    assert replay["replay_semantic_match"]
    assert replay["replay_bitwise_metric_match"]
    assert replay["decision_stable"]
    assert independent["semantic_match"] == "YES"
    assert independent["absolute_difference"] <= 1.0e-15
    assert set(independent["authoritative_evaluator_metrics"]) == set(str(item) for item in HORIZONS)


def test_frozen20_and_training_paths_remained_inaccessible() -> None:
    authority = read("authority_manifest.json")
    assert authority["frozen20_opened"] == "NO"
    assert authority["frozen20_used"] == "NO"
    assert authority["future_label_leakage"] == 0
    assert authority["d3a_training_or_tuning_performed"] == "NO"
    source = Path(__file__).parents[1] / "scripts/stage3_h13_r8e_d3a_seed_218309645_model_selection_failure_audit.py"
    text = source.read_text(encoding="utf-8")
    assert "torch.optim" not in text
    assert "optimizer.step" not in text
    assert "train_candidate" not in text
    assert "torch.save" not in text
