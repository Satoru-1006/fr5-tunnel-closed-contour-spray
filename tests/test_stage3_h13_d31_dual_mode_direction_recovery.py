import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import stage3_h13_post_d30_d31_dual_mode_direction_recovery as d31


def test_mode_a_alpha_neighborhood_is_bounded_and_deterministic():
    assert d31.mode_a_alphas(0.25) == [0.125, 0.25, 0.5]
    assert d31.mode_a_alphas(1.0 / 256.0) == [1.0 / 256.0, 1.0 / 128.0]


def test_hard_gate_requires_replay_optimizer_rng_h1_and_h32():
    base = {
        "finite_state_pass": True,
        "H32_preservation_pass": True,
        "H1_improvement_pass": True,
        "optimizer_consistency_pass": True,
        "rng_identity_pass": True,
        "deterministic_replay_consistency_pass": True,
    }
    assert d31.candidate_is_legal(base)
    for key in tuple(base):
        broken = dict(base)
        broken[key] = False
        assert not d31.candidate_is_legal(broken)


def test_checkpoint_rule_declares_modified_gradient_and_preserved_history():
    assert d31.MODE_A_COEFFICIENT == 0.35
    assert d31.MODE_B_COEFFICIENTS == (0.70, 1.40, 2.80)
    assert d31.LAST_UPDATE == 428
