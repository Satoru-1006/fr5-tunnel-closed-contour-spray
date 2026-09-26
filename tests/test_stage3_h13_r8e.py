from __future__ import annotations

import inspect

import pytest

from src.stage3_h11_dataset import FEATURE_NAMES
from src.stage3_h11_r2_model import ResidualCausalGRUTrajectoryPredictor
from src.stage3_h13_r8e import (
    GRADUATION_THRESHOLD,
    causal_paired_rollout,
    derive_h1_guardrail,
    graduation_passes,
    early_hidden_drift_reduced,
    h4_amplification_reduced_or_delayed,
    hidden_alignment_weights,
    normalized_hidden_smooth_l1,
    phase_for_update,
    select_candidate,
)


EXPECTED_THRESHOLD = 0.01856902565856056


def _batch(torch, batch: int = 3):
    inputs = torch.randn(batch, 16, len(FEATURE_NAMES)) * 0.01
    positions = torch.randn(batch, 16, 6) * 0.01
    times = torch.arange(16, dtype=torch.float32)[None, :].repeat(batch, 1)
    target_times = torch.arange(16, 20, dtype=torch.float32)[None, :].repeat(batch, 1)
    starts = torch.zeros(batch)
    ends = torch.full((batch,), 40.0)
    targets_a = torch.randn(batch, 4, 6) * 0.1 + 0.5
    targets_b = targets_a + 7.0
    channels = {name: {"mean": 0.0, "scale": 1.0} for name in FEATURE_NAMES if name != "spray_on"}
    return inputs, positions, times, target_times, starts, ends, targets_a, targets_b, channels


def test_r6_checkpoint_immutability_helper_is_load_only() -> None:
    source = inspect.getsource(causal_paired_rollout)
    assert "torch.save" not in source
    assert "load_state_dict" not in source


def test_free_rollout_is_causal_and_future_label_independent() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(8)
    model = ResidualCausalGRUTrajectoryPredictor(hidden_size=16, num_layers=2)
    args = _batch(torch)
    common = args[:6]
    channels = args[8]
    first = causal_paired_rollout(model, *common, channels, horizon=4, target_positions_for_teacher=args[6], hidden_consistency_enabled=False)
    second = causal_paired_rollout(model, *common, channels, horizon=4, target_positions_for_teacher=args[7], hidden_consistency_enabled=False)
    assert torch.equal(first["predictions"], second["predictions"])
    assert first["free_branch_reads_target_positions"] is False
    paired_first = causal_paired_rollout(model, *common, channels, horizon=4, target_positions_for_teacher=args[6], hidden_consistency_enabled=True)
    paired_second = causal_paired_rollout(model, *common, channels, horizon=4, target_positions_for_teacher=args[7], hidden_consistency_enabled=True)
    assert torch.equal(paired_first["predictions"], paired_second["predictions"])


def test_teacher_and_free_branches_share_parameters() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(9)
    model = ResidualCausalGRUTrajectoryPredictor(hidden_size=16, num_layers=2)
    args = _batch(torch)
    parameter_ids = tuple(id(parameter) for parameter in model.parameters())
    result = causal_paired_rollout(model, *args[:6], args[8], horizon=4, target_positions_for_teacher=args[6], hidden_consistency_enabled=True)
    (result["predictions"].mean() + result["hidden_loss"]).backward()
    assert tuple(id(parameter) for parameter in model.parameters()) == parameter_ids
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_teacher_hidden_reference_has_stop_gradient() -> None:
    torch = pytest.importorskip("torch")
    free = torch.randn(2, 3, 5, requires_grad=True)
    teacher = torch.randn(2, 3, 5, requires_grad=True)
    loss, _ = normalized_hidden_smooth_l1(free, teacher)
    loss.backward()
    assert free.grad is not None
    assert teacher.grad is None


def test_a0_hidden_loss_is_exactly_disabled_and_a1_enabled() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(10)
    model = ResidualCausalGRUTrajectoryPredictor(hidden_size=16, num_layers=2)
    args = _batch(torch)
    a0 = causal_paired_rollout(model, *args[:6], args[8], horizon=4, target_positions_for_teacher=None, hidden_consistency_enabled=False)
    a1 = causal_paired_rollout(model, *args[:6], args[8], horizon=4, target_positions_for_teacher=args[6], hidden_consistency_enabled=True)
    assert a0["hidden_loss"].item() == 0.0
    assert a0["teacher_branch_executed"] is False
    assert a1["hidden_loss"].item() > 0.0
    assert a1["teacher_branch_executed"] is True
    assert hidden_alignment_weights(32).tolist()[0] == 0.0
    assert hidden_alignment_weights(32).tolist()[16:] == [0.0] * 16


def test_curriculum_transitions_are_predeclared() -> None:
    assert [phase_for_update(item, 8) for item in (0, 7, 8, 15, 16, 23, 24, 999)] == [4, 4, 8, 8, 16, 16, 32, 32]


def test_guardrail_is_frozen_before_selection() -> None:
    guardrail = derive_h1_guardrail(8.188013630811277e-05, 8.188013636181157e-05)
    assert guardrail.frozen_before_candidate_selection is True
    assert guardrail.threshold == pytest.approx(8.597914317990215e-05, abs=1e-18)


def test_selection_is_validation_only_and_has_no_external_path_argument() -> None:
    guardrail = derive_h1_guardrail(8.188013630811277e-05, 8.188013636181157e-05)
    assert tuple(inspect.signature(select_candidate).parameters) == ("candidates", "guardrail", "tie_tolerance")
    candidates = [
        {"name": "A0", "h1_rmse": guardrail.reference, "h32_rmse": 0.02, "integrity_pass": True, "future_label_leakage": 0, "early_hidden_divergence": 0.4},
        {"name": "A1", "h1_rmse": guardrail.reference, "h32_rmse": 0.019, "integrity_pass": True, "future_label_leakage": 0, "early_hidden_divergence": 0.3},
    ]
    assert select_candidate(candidates, guardrail)["name"] == "A1"


def test_exact_graduation_threshold_and_hard_gates() -> None:
    assert GRADUATION_THRESHOLD == EXPECTED_THRESHOLD
    guardrail = derive_h1_guardrail(8.188013630811277e-05, 8.188013636181157e-05)
    base = {"h1_rmse": guardrail.reference, "h32_rmse": EXPECTED_THRESHOLD, "integrity_pass": True, "future_label_leakage": 0}
    assert graduation_passes(base, guardrail)
    assert not graduation_passes({**base, "h32_rmse": EXPECTED_THRESHOLD + 1e-15}, guardrail)
    assert not graduation_passes({**base, "h1_rmse": guardrail.threshold + 1e-15}, guardrail)


def test_mechanism_classification_uses_reported_direction_without_posthoc_cutoff() -> None:
    r6 = {"2": 0.3140690139101206, "4": 0.3727912893284838, "8": 0.34650557605671095}
    a1 = {"2": 0.2767824099409424, "4": 0.37789546905271126, "8": 0.33784951207442293}
    assert early_hidden_drift_reduced(r6, a1)
    assert h4_amplification_reduced_or_delayed(0.00045622708501127457, 0.0004539211738746857, 4, 4)
