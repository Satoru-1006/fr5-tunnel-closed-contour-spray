"""Focused non-scientific D25 design-review tests."""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import stage3_h13_post_d24_d25_trust_region_design_review as d25


def test_static_validation_passes_without_scientific_execution() -> None:
    result = d25.run_static_tests()
    assert result["pass"] is True
    assert result["scientific_training_executed"] is False
    assert result["scientific_backward_pass_executed"] is False
    assert result["scientific_optimizer_step_executed"] is False


def test_frozen_alpha_ladder_and_exact_h32_contract() -> None:
    assert d25.ALPHA_LADDER == (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625)
    assert d25.H32_LIMIT == 0.01856902565856056
    assert math.isclose(d25.H32_LIMIT - d25.START_H32, d25.H32_MARGIN_AT_START, rel_tol=0.0, abs_tol=1e-18)
    assert d25.H32_MARGIN_AT_START == 7.5024675936338e-05


def test_p4_definition_is_authenticated_d24_definition() -> None:
    assert d25.P4_SPEC == {
        "name": "P4",
        "family": "C5_PLUS_POSTCLIP_H1_BIAS",
        "rule": "structure_plus_postclip_h1_bias",
        "attenuation": 0.5,
        "postclip_h1_coefficient": 0.35,
        "attenuated_region": list(range(18, 32)),
    }


def test_candidate_schema_covers_required_transition_identity() -> None:
    required = {
        "parent_update", "candidate_update", "proposal", "alpha", "canonical_lr", "effective_lr",
        "parent_H1", "candidate_H1", "parent_H32", "candidate_H32", "H32_limit",
        "finite_state_pass", "H32_preservation_pass", "H1_improvement_pass", "candidate_valid",
        "rejection_reason", "selected", "model_checkpoint_sha256",
        "optimizer_state_sha256_or_equivalent", "batch_identity", "source_revision",
    }
    assert required.issubset(d25.CANDIDATE_FIELDS)
