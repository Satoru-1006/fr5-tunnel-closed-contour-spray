import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import stage3_h13_d33_inheritance_preserving_completion as d33


def record(h1, margin, delta_h32=0.0, valid=True, replay=False):
    return {
        "candidate_H1": h1,
        "candidate_H32": d33.H32_LIMIT - margin,
        "delta_H1": h1 - 1.0,
        "delta_H32": delta_h32,
        "H32_margin": margin,
        "effective_update_norm": 1.0,
        "spec_id": str(h1),
        "candidate_valid_before_replay": valid,
        "finite_state_pass": valid,
        "H32_preservation_pass": valid,
        "H1_improvement_pass": valid,
        "optimizer_consistency_pass": valid,
        "rng_identity_pass": valid,
        "deterministic_replay_consistency_pass": replay,
    }


def test_reserve_policy_adapts_across_green_yellow_red():
    assert d33.reserve_policy(4.8e-4)["regime"] == "GREEN"
    assert d33.reserve_policy(3.0e-4)["regime"] == "YELLOW"
    assert d33.reserve_policy(2.0e-4)["regime"] == "RED"
    assert d33.reserve_policy(4.8e-4)["reserve"] == 2.5e-4
    relaxed = d33.reserve_policy(2.53e-4, relaxed=True)
    assert relaxed["regime"] == "GREEN"
    assert relaxed["reserve"] == 1.25 * 8.30074e-5


def test_h1_first_rank_uses_h1_before_extra_margin():
    better_h1 = record(0.8, 2.0e-4)
    more_inward = record(0.9, 8.0e-4, delta_h32=-1.0e-3)
    assert d33.h1_first_rank(better_h1) < d33.h1_first_rank(more_inward)


def test_safety_reserve_is_a_gate_not_a_ranking_objective():
    policy = {"regime": "GREEN", "reserve": 2.5e-4, "outward_cap": 8.0e-5}
    passing = d33.annotate_safety(record(0.8, 2.6e-4), policy)
    failing = d33.annotate_safety(record(0.7, 2.4e-4), policy)
    assert d33.safely_legal(passing)
    assert not d33.safely_legal(failing)


def test_yellow_lattice_does_not_stop_without_cap_passing_comparator():
    policy = {"regime": "YELLOW", "reserve": 2.5e-4, "outward_cap": 8.0e-5}
    aggressive = [
        d33.annotate_safety(record(0.8 + i * 0.01, 3.0e-4, delta_h32=1.0e-4), policy)
        for i in range(5)
    ]
    assert not d33.yellow_lattice_has_enough_evidence(aggressive)
    conservative = d33.annotate_safety(record(0.9, 3.0e-4, delta_h32=4.0e-5), policy)
    assert d33.yellow_lattice_has_enough_evidence(aggressive + [conservative])


def test_most_h32_conservative_comparator_can_preserve_an_h1_leader():
    h1_leader = record(0.7, 3.0e-4, delta_h32=5.0e-5)
    inward = record(0.8, 4.0e-4, delta_h32=-2.0e-5)
    less_inward = record(0.75, 4.0e-4, delta_h32=-1.0e-5)
    assert d33.most_h32_conservative([h1_leader, inward, less_inward]) is inward


def test_smallest_scale_comparator_is_independent_of_one_step_h32_direction():
    large_inward = {**record(0.7, 3.0e-4, delta_h32=-2.0e-5), "alpha": 0.015625}
    small_outward = {**record(0.8, 3.0e-4, delta_h32=2.0e-6), "alpha": 0.0078125}
    assert d33.smallest_scale_comparator([large_inward, small_outward]) is small_outward


def test_retained_horizon_map_requires_same_parent_update_and_reserve():
    common = {
        "candidate_update": 460,
        "parent_update": 459,
        "screening_stage": "D33_SERIOUS_DETERMINISTIC_REPLAY",
        "horizon_status": "PASS",
        "horizon_length": 3,
        "horizon_criterion_version": d33.HORIZON_CRITERION_VERSION,
        "safety_reserve": 1.0e-4,
    }
    matching = {**common, "spec_id": "matching"}
    wrong_parent = {**common, "spec_id": "wrong_parent", "parent_update": 458}
    wrong_reserve = {**common, "spec_id": "wrong_reserve", "safety_reserve": 2.0e-4}
    retained = d33.retained_horizon_map([matching, wrong_parent, wrong_reserve], 460, 459, 1.0e-4)
    assert set(retained) == {"matching"}
    assert retained["matching"]["horizon_status"] == "PASS"


def test_retained_horizon_map_rejects_obsolete_criterion():
    obsolete = {
        "candidate_update": 460,
        "parent_update": 459,
        "screening_stage": "D33_SERIOUS_DETERMINISTIC_REPLAY",
        "horizon_status": "FAIL",
        "horizon_criterion_version": "OLD",
        "safety_reserve": 1.0e-4,
        "spec_id": "obsolete",
    }
    assert d33.retained_horizon_map([obsolete], 460, 459, 1.0e-4) == {}


def test_outward_cap_is_advisory_when_terminal_reserve_is_intact():
    policy = {"regime": "GREEN", "reserve": 2.5e-4, "outward_cap": 8.0e-5}
    candidate = d33.annotate_safety(record(0.8, 2.6e-4, delta_h32=1.0e-4), policy)
    assert not candidate["H32_outward_cap_pass"]
    assert d33.safely_legal(candidate)


def test_inherited_baseline_is_always_in_primary_lattice():
    for regime in ("GREEN", "YELLOW", "RED"):
        baseline = d33.inherited_baseline(0.03125, 0.875)
        identities = {d33.d32.spec_id(spec) for spec in d33.primary_specs(0.03125, 0.875, regime, False)}
        assert d33.d32.spec_id(baseline) in identities


def test_yellow_primary_lattice_keeps_quarter_scale_stability_candidate():
    specs = d33.primary_specs(0.015625, 0.4375, "YELLOW", False)
    assert any(
        spec.get("base") == "P4"
        and spec.get("p4_beta") == 0.4375
        and spec.get("lambda") == 0.0
        and spec.get("alpha") == 0.00390625
        for spec in specs
    )


def test_replay_gate_requires_deterministic_replay_when_requested():
    candidate = record(0.8, 3.0e-4, replay=False)
    candidate = d33.annotate_safety(candidate, {"regime": "GREEN", "reserve": 2.5e-4, "outward_cap": 8.0e-5})
    assert d33.safely_legal(candidate)
    assert not d33.safely_legal(candidate, replay_required=True)
    candidate["deterministic_replay_consistency_pass"] = True
    assert d33.safely_legal(candidate, replay_required=True)


def test_horizon_allows_nonmonotonic_intermediate_h1_when_terminal_improves():
    steps = []
    for h1 in (0.9, 1.05, 0.8):
        step = record(h1, 3.0e-4, valid=True)
        step["safety_reserve_pass"] = True
        steps.append(step)
    steps[1]["H1_improvement_pass"] = False
    steps[1]["candidate_valid_before_replay"] = False
    assert d33.horizon_step_safety_pass(steps[1])
    assert d33.horizon_chain_pass(steps, 3, parent_h1=1.0)


def test_horizon_rejects_terminal_h1_regression_even_when_every_step_is_safe():
    steps = []
    for h1 in (0.9, 0.95, 1.01):
        step = record(h1, 3.0e-4, valid=True)
        step["safety_reserve_pass"] = True
        steps.append(step)
    assert not d33.horizon_chain_pass(steps, 3, parent_h1=1.0)


def test_close_finalist_policy_can_promote_single_horizon_safe_candidate():
    common = {"H32_margin": 1.0, "effective_update_norm": 1.0}
    immediate = {**common, "spec_id": "immediate", "horizon_status": "FAIL", "candidate_H1": 0.8}
    safer = {**common, "spec_id": "safer", "horizon_status": "PASS", "candidate_H1": 0.81, "horizon_terminal_H1": 0.7}
    immediate_item = (immediate, None, None, {})
    safer_item = (safer, None, None, {})
    selected, basis = d33.select_replayed_candidate([immediate_item, safer_item], close=True)
    assert selected is safer_item
    assert basis == "CLOSE_FINALIST_SHORT_HORIZON_THEN_H1"


def test_outward_cap_exception_requires_horizon_pass_before_promotion():
    candidate = record(0.8, 3.0e-4, delta_h32=1.0e-4, valid=True, replay=True)
    candidate = d33.annotate_safety(candidate, {"regime": "GREEN", "reserve": 2.5e-4, "outward_cap": 8.0e-5})
    policy = {"regime": "GREEN"}
    assert not candidate["H32_outward_cap_pass"]
    assert not d33.promotion_safety_pass(candidate, policy)
    candidate["horizon_status"] = "PASS"
    assert d33.promotion_safety_pass(candidate, policy)


def test_explicit_horizon_failure_blocks_even_cap_passing_candidate():
    policy = {"regime": "GREEN", "reserve": 2.5e-4, "outward_cap": 8.0e-5}
    candidate = d33.annotate_safety(record(0.8, 3.0e-4, delta_h32=4.0e-5, valid=True, replay=True), policy)
    assert candidate["H32_outward_cap_pass"]
    candidate["horizon_status"] = "FAIL"
    assert not d33.promotion_safety_pass(candidate, policy)


def test_terminal_h1_only_horizon_failure_can_compete_when_cap_passes():
    policy = {"regime": "GREEN", "reserve": 2.5e-4, "outward_cap": 8.0e-5}
    candidate = d33.annotate_safety(record(0.8, 3.0e-4, delta_h32=4.0e-5, valid=True, replay=True), policy)
    candidate.update(
        {
            "horizon_status": "FAIL",
            "horizon_safety_status": "PASS",
            "horizon_terminal_H1_pass": False,
        }
    )
    assert candidate["H32_outward_cap_pass"]
    assert d33.promotion_safety_pass(candidate, policy)


def test_terminal_h1_only_failure_does_not_waive_outward_cap():
    policy = {"regime": "GREEN", "reserve": 2.5e-4, "outward_cap": 8.0e-5}
    candidate = d33.annotate_safety(record(0.8, 3.0e-4, delta_h32=1.0e-4, valid=True, replay=True), policy)
    candidate.update({"horizon_status": "FAIL", "horizon_safety_status": "PASS"})
    assert not candidate["H32_outward_cap_pass"]
    assert not d33.promotion_safety_pass(candidate, policy)


def test_cap_passing_candidate_without_requested_horizon_remains_eligible():
    policy = {"regime": "GREEN", "reserve": 2.5e-4, "outward_cap": 8.0e-5}
    candidate = d33.annotate_safety(record(0.8, 3.0e-4, delta_h32=4.0e-5, valid=True, replay=True), policy)
    assert candidate["H32_outward_cap_pass"]
    assert "horizon_status" not in candidate
    assert d33.promotion_safety_pass(candidate, policy)
