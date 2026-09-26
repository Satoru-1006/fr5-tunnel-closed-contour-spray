from __future__ import annotations

import numpy as np
import pytest

from src.p2b0_margin_attribution import (
    capability_status,
    nullspace_j6_retreat,
    process_task_jacobian,
    shadow_buffer_projection,
)


def _full_rank_process_case():
    # For normal +Z, the process task is XYZ translation plus angular X/Y.
    return process_task_jacobian(np.eye(6), [0.0, 0.0, 1.0])


def test_process_jacobian_and_null_projector_are_consistent() -> None:
    task = _full_rank_process_case()
    projector = task.null_projector
    assert task.raw.shape == (5, 6)
    assert task.scaled.shape == (5, 6)
    assert task.effective_rank == 5
    assert task.null_basis.shape == (6, 1)
    assert np.allclose(projector, projector.T, atol=1e-12)
    assert np.allclose(projector @ projector, projector, atol=1e-12)
    assert np.linalg.norm(task.scaled @ task.null_basis) < 1e-12


def test_nullspace_perturbation_preserves_primary_task_to_first_order() -> None:
    task = _full_rank_process_case()
    delta = nullspace_j6_retreat(task, 0.007)
    assert np.linalg.norm(task.scaled @ delta) < 1e-12
    assert delta[5] == pytest.approx(-0.007)


def test_j6_retreat_objective_increases_upper_limit_margin() -> None:
    task = _full_rank_process_case()
    upper = 3.0543
    q_before = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, upper - 1.0e-4])
    q_after = q_before + nullspace_j6_retreat(task, 0.010)
    assert upper - q_after[5] > upper - q_before[5]
    assert upper - q_after[5] == pytest.approx(0.0101)


def test_shadow_buffer_projection_matches_four_known_boundaries() -> None:
    low = np.full(6, -3.0543)
    high = np.full(6, 3.0543)
    proposal = np.zeros(6)
    proposal[5] = high[5] + 0.2
    for buffer in (1.0e-4, 1.0e-3, 5.0e-3, 1.0e-2):
        projected = shadow_buffer_projection(proposal, low, high, buffer)
        assert projected[5] == pytest.approx(high[5] - buffer)
        assert high[5] - projected[5] == pytest.approx(buffer)


def test_invalid_shadow_buffer_fails_closed() -> None:
    with pytest.raises(ValueError):
        shadow_buffer_projection(np.zeros(2), [-1.0, -1.0], [1.0, 1.0], -1e-4)
    with pytest.raises(ValueError):
        shadow_buffer_projection(np.zeros(2), [-1.0, -1.0], [1.0, 1.0], 1.0)


def test_unavailable_robot_measurement_is_inconclusive_not_pass() -> None:
    result = capability_status(False, "MoveIt2 native backend did not return a Jacobian")
    assert result["status"] == "INCONCLUSIVE"
    assert "did not return" in result["reason"]


def test_degenerate_or_nonfinite_process_jacobian_is_rejected() -> None:
    with pytest.raises(ValueError):
        process_task_jacobian(np.full((6, 6), np.nan), [0.0, 0.0, 1.0])
    with pytest.raises(ValueError):
        process_task_jacobian(np.eye(6), [0.0, 0.0, 0.0])
