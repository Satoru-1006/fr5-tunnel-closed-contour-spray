"""Persistent, update-agnostic D35 Champion state machine.

This module is deliberately independent from the expensive Stage-3 runtime.  It
owns control-flow invariants and durable metadata; the scientific runner supplies
replay-validated candidate checkpoints.  A promotion is a transaction and a
promotion never terminates the outer loop.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

TARGET_H1 = 5.0e-5
H32_LIMIT = 0.01856902565856056
MIN_CHAMPION_H32_RESERVE = 7.5e-5


class D35InvariantError(RuntimeError):
    pass


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Champion:
    update: int
    generation: int
    h1: float
    h32: float
    checkpoint: str
    checkpoint_sha256: str

    @property
    def h32_margin(self) -> float:
        return H32_LIMIT - self.h32

    @property
    def identity(self) -> str:
        return f"update-{self.update}:{self.checkpoint_sha256}:generation-{self.generation}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "update": self.update,
            "generation": self.generation,
            "H1": self.h1,
            "H32": self.h32,
            "H32_margin": self.h32_margin,
            "checkpoint": self.checkpoint,
            "checkpoint_sha256": self.checkpoint_sha256,
            "identity": self.identity,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Champion":
        return cls(
            update=int(value["update"]), generation=int(value["generation"]),
            h1=float(value["H1"]), h32=float(value["H32"]),
            checkpoint=str(value["checkpoint"]),
            checkpoint_sha256=str(value["checkpoint_sha256"]),
        )


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    parent_update: int
    parent_generation: int
    parent_h1: float
    parent_checkpoint: str
    parent_checkpoint_sha256: str
    update: int
    h1: float
    h32: float
    method: str
    trajectory: str
    checkpoint: str
    replay_pass: bool
    optimizer_consistency_pass: bool
    rng_lineage_pass: bool
    scheduler_continuity_pass: bool
    checkpoint_reproducibility_pass: bool

    @property
    def h32_margin(self) -> float:
        return H32_LIMIT - self.h32


def candidate_rejection(candidate: Candidate, champion: Champion) -> str | None:
    if (
        candidate.parent_update != champion.update
        or candidate.parent_generation != champion.generation
        or candidate.parent_checkpoint_sha256 != champion.checkpoint_sha256
        or Path(candidate.parent_checkpoint).resolve() != Path(champion.checkpoint).resolve()
        or candidate.parent_h1 != champion.h1
    ):
        return "STALE_SHADOW_EVIDENCE"
    if candidate.update <= champion.update:
        return "NONFORWARD_UPDATE"
    if not math.isfinite(candidate.h1) or not math.isfinite(candidate.h32):
        return "NONFINITE_ENDPOINT"
    if candidate.h1 >= champion.h1:
        return "CANONICAL_MONOTONICITY_REJECTED"
    if candidate.h32 > H32_LIMIT:
        return "H32_LIMIT_EXCEEDED"
    if candidate.h32_margin < MIN_CHAMPION_H32_RESERVE:
        return "ADAPTIVE_H32_RESERVE_INSUFFICIENT"
    gates = {
        "REPLAY_FAILED": candidate.replay_pass,
        "OPTIMIZER_CONSISTENCY_FAILED": candidate.optimizer_consistency_pass,
        "RNG_LINEAGE_FAILED": candidate.rng_lineage_pass,
        "SCHEDULER_CONTINUITY_FAILED": candidate.scheduler_continuity_pass,
        "CHECKPOINT_REPRODUCIBILITY_FAILED": candidate.checkpoint_reproducibility_pass,
    }
    for reason, passed in gates.items():
        if not passed:
            return reason
    path = Path(candidate.checkpoint)
    if not path.is_file():
        return "PREPARED_CHECKPOINT_MISSING"
    return None


class ChampionManager:
    STATE_NAME = "D35_EXECUTION_STATE.json"
    HISTORY_NAME = "D35_CHAMPION_HISTORY.csv"
    LEDGER_NAME = "D35_CANDIDATE_LEDGER.json"
    REGISTRY_NAME = "D35_METHOD_REGISTRY.json"
    TRANSACTION_NAME = "D35_PROMOTION_TRANSACTION.json"
    EVENT_LEDGER_NAME = "D35_EVENT_LEDGER.jsonl"

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.state_path = self.root / self.STATE_NAME
        self.history_path = self.root / self.HISTORY_NAME
        self.ledger_path = self.root / self.LEDGER_NAME
        self.registry_path = self.root / self.REGISTRY_NAME
        self.transaction_path = self.root / self.TRANSACTION_NAME
        self.event_ledger_path = self.root / self.EVENT_LEDGER_NAME

    def initialize(self, champion: Champion) -> dict[str, Any]:
        self.root.mkdir(parents=True, exist_ok=True)
        if self.state_path.exists():
            return self.load_state()
        checkpoint = Path(champion.checkpoint)
        if not checkpoint.is_file() or sha256_file(checkpoint) != champion.checkpoint_sha256:
            raise D35InvariantError("START_CHAMPION_AUTHENTICATION_FAILED")
        if champion.h32 > H32_LIMIT or champion.h1 <= 0.0:
            raise D35InvariantError("START_CHAMPION_METRICS_INVALID")
        atomic_json(self.ledger_path, [])
        atomic_json(self.registry_path, {})
        self._append_event("INITIALIZE", {"champion": champion.as_dict()})
        self._append_history({
            "champion_update": champion.update, "parent_update": "", "generation": champion.generation,
            "H1": champion.h1, "delta_H1": 0.0, "H32": champion.h32,
            "H32_margin": champion.h32_margin, "method": "D34_VERIFIED_START",
            "trajectory": "D34_RECOVER_SMALL_THEN_PCGRAD_5_128", "meaningful_shadows": 0,
            "replay": "PASS", "checkpoint": champion.checkpoint,
            "checkpoint_sha256": champion.checkpoint_sha256,
            "reason_promoted": "AUTHENTICATED_D35_INITIAL_FLOOR", "snowball_cycle": 0,
        })
        state = self._state_for(champion)
        atomic_json(self.state_path, state)
        return state

    def _state_for(self, champion: Champion) -> dict[str, Any]:
        return {
            "schema_version": "d35_execution_state_v1",
            "current_champion": champion.as_dict(),
            "current_champion_update": champion.update,
            "current_champion_h1": champion.h1,
            "current_champion_h32": champion.h32,
            "current_h32_margin": champion.h32_margin,
            "current_checkpoint": champion.checkpoint,
            "target_h1": TARGET_H1,
            "remaining_gap": max(0.0, champion.h1 - TARGET_H1),
            "target_reached": champion.h1 <= TARGET_H1,
            "true_blocker": False,
            "true_blocker_status": "NONE_ESTABLISHED",
            "external_interruption": False,
            "snowball_cycle": champion.generation,
            "exploration_wave": 0,
            "diversification_level": 0,
            "active_methods": [], "promising_methods": [], "breakthrough_methods": [],
            "plateaued_methods": [], "failed_current_geometry_methods": [], "toolbox_methods": [],
            "stale_agent_results": 0,
            "champion_history_path": str(self.history_path),
            "candidate_ledger_path": str(self.ledger_path),
            "method_registry_path": str(self.registry_path),
            "last_completed_action": "AUTHENTICATE_CHAMPION",
            "next_intended_action": "ANALYZE_AND_LAUNCH_EXPLORATION_WAVE",
            "interrupted_resumable": False,
        }

    def load_state(self) -> dict[str, Any]:
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        champion = Champion.from_mapping(state["current_champion"])
        if sha256_file(Path(champion.checkpoint)) != champion.checkpoint_sha256:
            raise D35InvariantError("CURRENT_CHAMPION_SHA256_MISMATCH")
        return state

    def champion(self) -> Champion:
        return Champion.from_mapping(self.load_state()["current_champion"])

    def must_continue(self, state: Mapping[str, Any] | None = None) -> bool:
        current = dict(state or self.load_state())
        return not (
            bool(current.get("target_reached"))
            or bool(current.get("true_blocker"))
            or bool(current.get("external_interruption"))
        )

    def normal_report_allowed(self, state: Mapping[str, Any] | None = None) -> bool:
        return not self.must_continue(state)

    def record_candidate(self, row: Mapping[str, Any]) -> None:
        ledger = json.loads(self.ledger_path.read_text(encoding="utf-8"))
        ledger.append(dict(row))
        atomic_json(self.ledger_path, ledger)
        self._append_event("CANDIDATE", dict(row))

    def record_failed_wave(self) -> dict[str, Any]:
        state = self.load_state()
        if state["target_reached"] or state["true_blocker"]:
            raise D35InvariantError("FAILED_WAVE_AFTER_TERMINAL_STATE")
        state["exploration_wave"] = int(state["exploration_wave"]) + 1
        state["diversification_level"] = int(state["diversification_level"]) + 1
        state["last_completed_action"] = "WAVE_WITHOUT_PROMOTION"
        state["next_intended_action"] = "DIVERSIFY_AND_LAUNCH_NEXT_WAVE"
        atomic_json(self.state_path, state)
        self._append_event("FAILED_WAVE_DIVERSIFY", {
            "exploration_wave": state["exploration_wave"],
            "diversification_level": state["diversification_level"],
        })
        return state

    def promote(self, candidate: Candidate, checkpoint_verifier: Callable[[Path, Candidate], bool]) -> Champion:
        champion = self.champion()
        if (
            champion.update == candidate.update and champion.h1 == candidate.h1
            and champion.h32 == candidate.h32
        ):
            return champion
        reason = candidate_rejection(candidate, champion)
        if reason is not None:
            if reason == "STALE_SHADOW_EVIDENCE":
                state = self.load_state()
                state["stale_agent_results"] = int(state["stale_agent_results"]) + 1
                atomic_json(self.state_path, state)
            raise D35InvariantError(reason)
        prepared = Path(candidate.checkpoint).resolve()
        if not checkpoint_verifier(prepared, candidate):
            raise D35InvariantError("PREPARED_CHECKPOINT_VERIFICATION_FAILED")
        destination = self.root / "checkpoints" / f"committed_update_{candidate.update}.pt"
        transaction = {
            "status": "PREPARED", "parent": champion.as_dict(),
            "candidate": candidate.__dict__, "prepared_checkpoint_sha256": sha256_file(prepared),
            "destination": str(destination),
        }
        atomic_json(self.transaction_path, transaction)
        self._append_event("PROMOTION_PREPARED", transaction)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise D35InvariantError("CHAMPION_DESTINATION_ALREADY_EXISTS")
        commit_temporary = destination.with_suffix(destination.suffix + ".commit.tmp")
        shutil.copyfile(prepared, commit_temporary)
        with commit_temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(commit_temporary, destination)
        digest = sha256_file(destination)
        if digest != transaction["prepared_checkpoint_sha256"]:
            raise D35InvariantError("COMMITTED_CHECKPOINT_COPY_DIGEST_MISMATCH")
        atomic_json(self.transaction_path, {**transaction, "status": "CHECKPOINT_COMMITTED", "checkpoint_sha256": digest})
        promoted = Champion(
            update=candidate.update, generation=champion.generation + 1,
            h1=candidate.h1, h32=candidate.h32,
            checkpoint=str(destination), checkpoint_sha256=digest,
        )
        if not checkpoint_verifier(destination, candidate):
            raise D35InvariantError("COMMITTED_CHECKPOINT_RELOAD_VERIFICATION_FAILED")
        self._append_history({
            "champion_update": promoted.update, "parent_update": champion.update,
            "generation": promoted.generation, "H1": promoted.h1,
            "delta_H1": promoted.h1 - champion.h1, "H32": promoted.h32,
            "H32_margin": promoted.h32_margin, "method": candidate.method,
            "trajectory": candidate.trajectory, "meaningful_shadows": "",
            "replay": "PASS", "checkpoint": promoted.checkpoint,
            "checkpoint_sha256": promoted.checkpoint_sha256,
            "reason_promoted": "STRICTLY_BETTER_REPLAY_VALIDATED_ENDPOINT",
            "snowball_cycle": promoted.generation,
        })
        old_state = self.load_state()
        state = self._state_for(promoted)
        state["exploration_wave"] = int(old_state["exploration_wave"])
        state["diversification_level"] = 0
        state["toolbox_methods"] = sorted(set(old_state.get("toolbox_methods", [])) | {candidate.method})
        state["last_completed_action"] = "ATOMIC_CHAMPION_PROMOTION"
        state["next_intended_action"] = (
            "FINAL_VALIDATION" if promoted.h1 <= TARGET_H1
            else "RECOMPUTE_GEOMETRY_AND_RESTART_METHOD_COMPETITION"
        )
        atomic_json(self.state_path, state)
        atomic_json(self.transaction_path, {**transaction, "status": "COMMITTED", "champion": promoted.as_dict()})
        self._append_event("PROMOTION_COMMITTED", {"parent": champion.as_dict(), "champion": promoted.as_dict()})
        return promoted

    def mark_external_interruption(self, next_action: str) -> dict[str, Any]:
        state = self.load_state()
        state["external_interruption"] = True
        state["interrupted_resumable"] = True
        state["last_completed_action"] = "SAVE_RESUME_STATE"
        state["next_intended_action"] = next_action
        atomic_json(self.state_path, state)
        self._append_event("EXTERNAL_INTERRUPTION", {"next_intended_action": next_action})
        return state

    def mark_true_blocker(self, obstruction: str, falsification_attempts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        attempts = [dict(item) for item in falsification_attempts]
        if len(attempts) < 2 or any(not item.get("independent") for item in attempts):
            raise D35InvariantError("TRUE_BLOCKER_REQUIRES_INDEPENDENT_FALSIFICATION")
        state = self.load_state()
        if state["target_reached"]:
            raise D35InvariantError("BLOCKER_AFTER_TARGET_REACHED")
        state["true_blocker"] = True
        state["true_blocker_status"] = "ESTABLISHED_AFTER_FALSIFICATION"
        state["true_blocker_evidence"] = {"obstruction": obstruction, "falsification_attempts": attempts}
        state["last_completed_action"] = "BLOCKER_FALSIFICATION_PHASE"
        state["next_intended_action"] = "TRUE_BLOCKER_REPORT"
        atomic_json(self.state_path, state)
        self._append_event("TRUE_BLOCKER_ESTABLISHED", state["true_blocker_evidence"])
        return state

    def _append_event(self, kind: str, payload: Mapping[str, Any]) -> None:
        self.event_ledger_path.parent.mkdir(parents=True, exist_ok=True)
        event = {"kind": kind, "payload": dict(payload)}
        with self.event_ledger_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _append_history(self, row: Mapping[str, Any]) -> None:
        fields = [
            "champion_update", "parent_update", "generation", "H1", "delta_H1", "H32",
            "H32_margin", "method", "trajectory", "meaningful_shadows", "replay",
            "checkpoint", "checkpoint_sha256", "reason_promoted", "snowball_cycle",
        ]
        exists = self.history_path.exists()
        with self.history_path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            if not exists:
                writer.writeheader()
            writer.writerow({name: row.get(name, "") for name in fields})


def run_persistent_outer_loop(
    manager: ChampionManager,
    run_wave: Callable[[Champion, Mapping[str, Any]], Candidate | None],
    checkpoint_verifier: Callable[[Path, Candidate], bool],
) -> str:
    """Hard outer loop. It returns only for one of the three D35 terminal classes."""
    while True:
        state = manager.load_state()
        if state["target_reached"]:
            return "TARGET_REACHED"
        if state["true_blocker"]:
            return "TRUE_BLOCKER"
        if state["external_interruption"]:
            return "EXTERNAL_RUNTIME_INTERRUPTION"
        candidate = run_wave(manager.champion(), state)
        if candidate is None:
            refreshed = manager.load_state()
            if not manager.must_continue(refreshed):
                continue
            manager.record_failed_wave()
            continue
        manager.promote(candidate, checkpoint_verifier)
        # Intentionally continue: promotion is a new snowball cycle, never completion.
