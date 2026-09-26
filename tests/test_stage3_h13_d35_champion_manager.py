from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tools.stage3_h13_d35_champion_manager import (
    Candidate, Champion, ChampionManager, D35InvariantError, H32_LIMIT,
    candidate_rejection, run_persistent_outer_loop,
)


def write_checkpoint(path: Path, value: bytes = b"checkpoint") -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return hashlib.sha256(value).hexdigest()


def make_manager(tmp_path: Path, h1: float = 7.0e-5) -> ChampionManager:
    checkpoint = tmp_path / "start.pt"
    digest = write_checkpoint(checkpoint)
    manager = ChampionManager(tmp_path / "d35")
    manager.initialize(Champion(469, 0, h1, 0.01848, str(checkpoint), digest))
    return manager


def make_candidate(manager: ChampionManager, h1: float = 6.9e-5, **changes):
    parent = manager.champion()
    prepared = manager.root / "prepared.pt"
    write_checkpoint(prepared, b"prepared")
    values = dict(
        candidate_id="candidate", parent_update=parent.update,
        parent_generation=parent.generation, parent_h1=parent.h1,
        parent_checkpoint=parent.checkpoint,
        parent_checkpoint_sha256=parent.checkpoint_sha256,
        update=parent.update + 1, h1=h1, h32=0.01849,
        method="PCGRAD", trajectory="ONE_STEP", checkpoint=str(prepared),
        replay_pass=True, optimizer_consistency_pass=True, rng_lineage_pass=True,
        scheduler_continuity_pass=True, checkpoint_reproducibility_pass=True,
    )
    values.update(changes)
    return Candidate(**values)


def verifier(path: Path, candidate: Candidate) -> bool:
    return path.is_file()


def test_a_target_false_no_blocker_must_continue(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    assert manager.must_continue()
    assert not manager.normal_report_allowed()


def test_b_promotion_starts_another_cycle(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    calls = []
    def wave(champion, state):
        calls.append(champion.update)
        if len(calls) == 1:
            return make_candidate(manager)
        state = manager.mark_external_interruption("RESUME_TEST")
        return None
    assert run_persistent_outer_loop(manager, wave, verifier) == "EXTERNAL_RUNTIME_INTERRUPTION"
    assert calls == [469, 470]


def test_c_dynamic_champion_ids(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    candidate = make_candidate(manager, update=477)
    assert manager.promote(candidate, verifier).update == 477


def test_d_canonical_monotonicity(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    assert candidate_rejection(make_candidate(manager, h1=manager.champion().h1), manager.champion()) == "CANONICAL_MONOTONICITY_REJECTED"


def test_e_shadow_regression_does_not_change_champion(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    assert candidate_rejection(make_candidate(manager, h1=8.0e-5), manager.champion())
    assert manager.champion().h1 == 7.0e-5


def test_f_stale_agent_cannot_overwrite_new_champion(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    stale = make_candidate(manager, h1=6.0e-5)
    manager.promote(make_candidate(manager, h1=6.9e-5), verifier)
    assert candidate_rejection(stale, manager.champion()) == "STALE_SHADOW_EVIDENCE"


def test_g_failed_wave_diversifies_instead_of_exit(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    state = manager.record_failed_wave()
    assert state["diversification_level"] == 1
    assert manager.must_continue(state)


def test_h_interruption_is_resumable_not_pass(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    state = manager.mark_external_interruption("NEXT")
    assert state["interrupted_resumable"] and not state["target_reached"]
    assert manager.normal_report_allowed(state)


def test_i_only_valid_target_candidate_can_succeed(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    bad = make_candidate(manager, h1=4.9e-5, replay_pass=False)
    with pytest.raises(D35InvariantError, match="REPLAY_FAILED"):
        manager.promote(bad, verifier)
    good = make_candidate(manager, h1=4.9e-5)
    manager.promote(good, verifier)
    assert manager.load_state()["target_reached"]


def test_j_report_gate_and_h32_gate(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    assert not manager.normal_report_allowed()
    bad = make_candidate(manager, h32=H32_LIMIT + 1e-9)
    assert candidate_rejection(bad, manager.champion()) == "H32_LIMIT_EXCEEDED"

