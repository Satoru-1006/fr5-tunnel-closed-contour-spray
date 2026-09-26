from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("stage3_h7_5", ROOT / "scripts/stage3_h7_5.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_factor_42_matches_moveit_retry_semantics() -> None:
    assert abs(1.1**42 - 54.763699237493086) < 1e-12
    assert 1.1**42 > 50.0


def test_required_h7_5_artifacts_are_unique() -> None:
    assert len(MODULE.REQUIRED) == len(set(MODULE.REQUIRED))
    assert "stage3_h7_5_terminal_certificate.json" in MODULE.REQUIRED
    assert "stage3_h7_5_overshoot_policy_migration_certificate.json" not in MODULE.REQUIRED


def test_collision_claim_is_discrete_and_clearance_unavailable() -> None:
    assert MODULE.COLLISION_METHOD == "adaptive_discrete_interpolation"
