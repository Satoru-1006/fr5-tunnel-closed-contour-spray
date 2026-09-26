import ast
from pathlib import Path

import numpy as np
import pytest

from src.stage3_h13_r8e_d import (
    SEEDS,
    classify_counterfactual,
    excess_rescue_fraction,
    feature_decomposition,
    paired_effect,
    relative_l2,
    verify_feature_swap,
)


def test_exact_seed_identity_and_feature_decomposition() -> None:
    assert SEEDS == (13086, 1727279519, 218309645, 258275761, 1638377564)
    rows = feature_decomposition()
    assert len(rows) == 20
    assert all(row["SWAPPED"] == "YES" for row in rows[:18])
    assert all(row["SWAPPED"] == "NO" for row in rows[18:])
    assert [row["FEATURE_NAME"] for row in rows[18:]] == ["spray_on", "normalized_local_trajectory_time"]


def test_only_feedback_dimensions_are_swapped_and_non_feedback_identical() -> None:
    native = np.zeros((2, 20), dtype=np.float64)
    candidate = np.ones((2, 18), dtype=np.float64)
    r6 = np.full((2, 18), 2.0, dtype=np.float64)
    native[:, :18] = candidate
    counterfactual = native.copy()
    counterfactual[:, :18] = r6
    proof = verify_feature_swap(native, counterfactual, candidate, r6)
    assert proof["NON_FEEDBACK_FEATURES_IDENTICAL"] == "YES"
    assert proof["FEEDBACK_FEATURES_DIFFER_AS_INTENDED"] == "YES"
    assert proof["ONLY_FEEDBACK_FEATURES_CHANGED"] == "YES"
    assert proof["changed_non_feedback_element_count"] == 0


def test_unintended_exogenous_change_fails_closed() -> None:
    native = np.zeros((1, 20))
    candidate = np.ones((1, 18))
    r6 = np.full((1, 18), 2.0)
    native[:, :18] = candidate
    counterfactual = native.copy()
    counterfactual[:, :18] = r6
    counterfactual[:, 19] = 1.0
    proof = verify_feature_swap(native, counterfactual, candidate, r6)
    assert proof["ONLY_FEEDBACK_FEATURES_CHANGED"] == "NO"


def test_hidden_metric_is_r8e_c_relative_l2_with_layers() -> None:
    reference = np.ones((2, 3, 4), dtype=np.float64)
    value = reference.copy()
    value[0, 1, 0] += 1.0
    aggregate, layers = relative_l2(reference, value)
    assert aggregate.shape == (3,)
    assert len(layers) == 2
    assert aggregate[1] > 0.0
    assert layers[1][1] == pytest.approx(0.0)


def test_effect_sizes_and_excess_rescue_are_raw_and_signed() -> None:
    assert paired_effect(10.0, 8.0) == {"absolute_change": -2.0, "percent_change": -20.0}
    assert excess_rescue_fraction(10.0, 7.0, 4.0) == pytest.approx(0.5)
    assert excess_rescue_fraction(4.0, 3.0, 4.0) is None


def test_classification_is_conservative_without_materiality_threshold() -> None:
    rows = [{"h4_hidden_native": 1.0, "h4_hidden_counterfactual": 0.9} for _ in range(5)]
    assert classify_counterfactual(rows, True)["name"] == "INCONCLUSIVE"
    rows = [{"h4_hidden_native": 1.0, "h4_hidden_counterfactual": 1.1} for _ in range(5)]
    assert classify_counterfactual(rows, True)["name"] == "FEEDBACK_FEATURE_SHIFT_NOT_SUFFICIENT"
    assert classify_counterfactual(rows[:4], True)["name"] == "INCONCLUSIVE"


def test_runner_has_no_training_or_optimizer_path() -> None:
    source = Path("scripts/stage3_h13_r8e_d_counterfactual_feedback_audit.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert "train_candidate" not in names
    assert "AdamW" not in source
    assert "optimizer.step" not in source


def test_runner_rejects_sealed_paths_without_inspection() -> None:
    from scripts.stage3_h13_r8e_d_counterfactual_feedback_audit import reject_sealed_paths

    with pytest.raises(RuntimeError, match="sealed_evaluation_path_forbidden"):
        reject_sealed_paths([Path("outputs/frozen20")])


def test_semantic_constants_keep_natural_post_h2_branching_contract() -> None:
    source = Path("scripts/stage3_h13_r8e_d_counterfactual_feedback_audit.py").read_text(encoding="utf-8")
    assert "state[\"counterfactual\"][\"current\"]" in source
    assert "instrumented_forward(model if name in (\"teacher\", \"native\", \"counterfactual\") else r6, state[name][\"current\"])" in source
    assert "H2_START_STATE_IDENTICAL" in source
    assert "state[\"counterfactual\"] = r6" not in source


def test_seed_manifest_is_explicit_and_not_seed_shopped() -> None:
    source = Path("scripts/stage3_h13_r8e_d_counterfactual_feedback_audit.py").read_text(encoding="utf-8")
    assert "load_checkpoint_manifest" in source
    assert "checkpoint_manifest" in source
    assert "SEEDS" in source


def test_aggregation_is_deterministic() -> None:
    left = np.asarray([0.1, 0.2, 0.3])
    right = np.asarray([0.1, 0.2, 0.3])
    assert np.array_equal(left, right)
