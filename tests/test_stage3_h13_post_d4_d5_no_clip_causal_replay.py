from __future__ import annotations

from scripts.stage3_h13_post_d4_d5_no_clip_causal_replay import (
    BATCH_SIZE,
    EXPERIMENT_ID,
    H32_THRESHOLD,
    D4_H1_THRESHOLD,
    TOTAL_UPDATES,
    build_recipe_diff,
    classify_divergence,
    raw_gradient_norm,
)
import torch


def test_d5_recipe_diff_contains_exactly_one_substantive_change() -> None:
    diff = build_recipe_diff("schedule", "storage")
    assert diff["changed_fields"] == ["gradient_clipping"]
    assert diff["substantive_training_change_count"] == 1
    assert diff["single_factor_change_confirmed"] == "YES"
    assert diff["D4_recipe"]["gradient_clipping"]["enabled"] == "YES"
    assert diff["D5_recipe"]["gradient_clipping"]["enabled"] == "NO"
    assert diff["D5_recipe"]["gradient_clipping"]["max_norm"] is None


def test_d5_protocol_constants_and_no_clip_raw_norm_are_exact() -> None:
    assert EXPERIMENT_ID == "STAGE_3_H13_POST_D4_D5_NO_CLIP_CAUSAL_REPLAY"
    assert (BATCH_SIZE, TOTAL_UPDATES) == (256, 512)
    assert D4_H1_THRESHOLD == 8.59791431235184e-05
    assert H32_THRESHOLD == 0.01856902565856056
    parameter = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    parameter.grad = torch.tensor([3.0, 4.0])
    before = parameter.grad.clone()
    assert raw_gradient_norm([("p", parameter)]) == 5.0
    assert torch.equal(parameter.grad, before)


def test_d5_divergence_classification_is_fail_closed_before_terminal() -> None:
    assert classify_divergence({"first_post_warmup_h1_degradation_update": None}, completed=False) == "NOT_RESOLVED"
