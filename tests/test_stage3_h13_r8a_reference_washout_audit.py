from __future__ import annotations

import numpy as np

from scripts.stage3_h13_r8a_reference_washout_audit import (
    INPUT_HISTORY,
    MATERIAL_GATE,
    history_washout,
    reference_numpy,
)


def test_reference_history_accounting_is_exact() -> None:
    audit = history_washout()
    assert audit["FULL_REFERENCE_CONTEXT_FIRST_HORIZON"] == 16
    assert audit["rows"]["8"]["observed_history_fraction"] == 0.5
    assert audit["rows"]["15"]["reference_history_fraction"] == 15.0 / INPUT_HISTORY
    assert audit["rows"]["16"]["observed_history_fraction"] == 0.0
    assert audit["rows"]["17"]["reference_history_fraction"] == 1.0


def test_constant_velocity_reference_uses_only_history() -> None:
    positions = np.asarray([[[0.0] * 6, [1.0] * 6, [2.0] * 6]], dtype=np.float64)
    times = np.asarray([[0.0, 1.0, 2.0]], dtype=np.float64)
    targets = np.asarray([[3.0, 4.0]], dtype=np.float64)
    result = reference_numpy("cv", positions, times, targets)
    np.testing.assert_allclose(result, np.asarray([[[3.0] * 6, [4.0] * 6]]))


def test_material_gate_is_strictly_below_r6() -> None:
    assert MATERIAL_GATE < 0.0232112820732007
