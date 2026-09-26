from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.stage3_h13_r8d import BoundedPositionResidualCorrector, CORRECTOR_INPUT_SIZE


def test_r8d_corrector_is_small_and_bounded() -> None:
    torch = pytest.importorskip("torch")
    bounds = np.asarray([0.01] * 6, dtype=np.float32)
    model = BoundedPositionResidualCorrector(bounds, hidden_size=32)
    values = model(torch.full((4, CORRECTOR_INPUT_SIZE), 100.0))
    assert values.shape == (4, 6)
    assert torch.all(torch.abs(values) <= torch.from_numpy(bounds) + 1.0e-7)
    assert sum(parameter.numel() for parameter in model.parameters()) < 3000


def test_r8d_corrector_starts_as_zero_residual() -> None:
    torch = pytest.importorskip("torch")
    model = BoundedPositionResidualCorrector([0.01] * 6, hidden_size=32)
    output = model(torch.randn(3, CORRECTOR_INPUT_SIZE))
    assert torch.equal(output, torch.zeros_like(output))


def test_r8d_source_has_no_scheduled_teacher_forcing_training_path() -> None:
    source = (Path(__file__).resolve().parents[1] / "scripts" / "stage3_h13_r8d_residual_rollout_correction.py").read_text(encoding="utf-8")
    assert "teacher_forcing_ratio=0.0" in source
    assert "tf0.75" not in source
    assert "tf0.50" not in source
    assert "tf0.25" not in source
    assert "run_native" not in source


def test_r8d_checkpoint_path_is_additive() -> None:
    source = (Path(__file__).resolve().parents[1] / "scripts" / "stage3_h13_r8d_residual_rollout_correction.py").read_text(encoding="utf-8")
    assert "R6_CHECKPOINT" in source
    assert "r8d_best_corrector_checkpoint.pt" in source
    assert "R6_CHECKPOINT.unlink" not in source
