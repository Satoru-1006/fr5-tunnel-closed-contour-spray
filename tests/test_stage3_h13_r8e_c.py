import numpy as np
import pytest

from src.stage3_h13_r8e_c import (
    FAIL_SEEDS,
    H1_GUARDRAIL,
    PASS_SEEDS,
    SEEDS,
    aggregate_h1,
    assert_validation_only_paths,
    classify_root_cause,
    guardrail_pass,
    seed_group,
)


def test_seed_grouping_is_exact() -> None:
    assert SEEDS == (13086, 218309645, 258275761, 1638377564, 1727279519)
    assert PASS_SEEDS == (13086, 1727279519)
    assert FAIL_SEEDS == (218309645, 258275761, 1638377564)
    assert [seed_group(seed) for seed in SEEDS] == ["PASS", "FAIL", "FAIL", "FAIL", "PASS"]
    with pytest.raises(ValueError, match="unexpected_seed"):
        seed_group(1)


def test_guardrail_semantics_are_inclusive_and_exact() -> None:
    assert guardrail_pass(H1_GUARDRAIL)
    assert not guardrail_pass(np.nextafter(H1_GUARDRAIL, np.inf))
    assert not guardrail_pass(float("nan"))


def test_per_case_and_component_aggregation_are_exact() -> None:
    errors = np.asarray([[1.0, -2.0], [3.0, -4.0]])
    result = aggregate_h1(errors)
    assert result["rmse"] == pytest.approx(np.sqrt(7.5))
    assert result["signed_mean_error"] == pytest.approx(-0.5)
    assert result["per_case_squared_error"].tolist() == [5.0, 25.0]
    assert result["per_case_relative_contribution"].tolist() == pytest.approx([1 / 6, 5 / 6])
    assert result["per_dimension_squared_error_share"].tolist() == pytest.approx([1 / 3, 2 / 3])


def test_classification_is_conservative_and_precedence_is_fixed() -> None:
    assert classify_root_cause({}) == {"code": "G", "name": "INCONCLUSIVE"}
    assert classify_root_cause({"case_concentrated": True, "common_h1_h4_mechanism": True, "temporal_chain_supported": True})["code"] == "E"
    assert classify_root_cause({"feature_fail_consistently_higher": True, "feature_effect": 0.049})["code"] == "G"
    assert classify_root_cause({"feature_fail_consistently_higher": True, "feature_effect": 0.05})["code"] == "A"
    assert classify_root_cause({"early_hidden_fail_consistently_higher": True, "hidden_effect": 0.05})["code"] == "B"
    assert classify_root_cause({"input_hidden_similar": True, "output_local_difference_strong": True})["code"] == "C"


def test_hidden_and_feature_metric_semantics_and_sealed_exclusion() -> None:
    # Relative L2 is ||free-teacher|| / ||teacher||, while OOD is a separate
    # feature-domain event and must not be substituted for hidden divergence.
    teacher = np.asarray([[3.0, 4.0]])
    free = np.asarray([[6.0, 8.0]])
    relative_l2 = np.linalg.norm(free - teacher, axis=1) / np.maximum(np.linalg.norm(teacher, axis=1), 1e-12)
    assert relative_l2.tolist() == pytest.approx([1.0])
    feature = np.asarray([[0.0, 3.1]])
    assert bool(np.any(np.abs(feature) > 3.0, axis=1)[0])
    assert_validation_only_paths(["outputs/h10", "outputs/h11", "outputs/r8e_b"])
    with pytest.raises(ValueError, match="sealed_evaluation_path_forbidden"):
        assert_validation_only_paths(["outputs/frozen20"])
