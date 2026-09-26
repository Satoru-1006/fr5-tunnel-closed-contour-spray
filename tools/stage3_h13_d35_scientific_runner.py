"""D35 inherited-AdamW scientific runner and persistent snowball controller.

The default command is intentionally persistent.  ``--one-wave`` is a resumable
operator boundary for testing/monitoring and is never reported as scientific
success.
"""

from __future__ import annotations

import argparse
import copy
import json
import signal
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import stage3_h13_d34_champion_breakthrough_search as d34
from tools.stage3_h13_d35_champion_manager import (
    Candidate, Champion, ChampionManager, D35InvariantError, H32_LIMIT,
    MIN_CHAMPION_H32_RESERVE, TARGET_H1,
    atomic_json, run_persistent_outer_loop, sha256_file,
)

d32 = d34.d32
d33 = d34.d33

START_UPDATE = 469
START_H1 = 7.173430599053105e-05
START_H32 = 0.01848121489533577
START_SHA256 = "f2c905335b3ea8e03611aa5fed45dad8e606949de2f60274811119bb9064ff88"
START_CHECKPOINT = (
    ROOT / "outputs" / "stage3_h13_d34_champion_breakthrough_search_20260825T120000+0800"
    / "checkpoints" / "committed_update_469.pt"
)
DEFAULT_OUTPUT = ROOT / "outputs" / "stage3_h13_d35_permanent_champion"


class ExternalStop:
    requested = False

    def __call__(self, signum: int, frame: Any) -> None:
        del signum, frame
        self.requested = True


def authenticate_start() -> Champion:
    if sha256_file(START_CHECKPOINT) != START_SHA256:
        raise D35InvariantError("UPDATE_469_CHECKPOINT_SHA256_MISMATCH")
    payload = torch.load(START_CHECKPOINT, map_location="cpu", weights_only=False)
    d33.verify_loaded_payload(
        payload, START_UPDATE, START_H1, START_H32,
        int(payload["parent_update"]), str(payload["parent_checkpoint_sha256"]),
    )
    if not bool(payload.get("replay_consistency", {}).get("pass")):
        raise D35InvariantError("UPDATE_469_REPLAY_AUTHENTICATION_FAILED")
    return Champion(START_UPDATE, 0, START_H1, START_H32, str(START_CHECKPOINT.resolve()), START_SHA256)


def load_branch(runtime: Mapping[str, Any], champion: Champion):
    path = Path(champion.checkpoint)
    if sha256_file(path) != champion.checkpoint_sha256:
        raise D35InvariantError("D35_PARENT_SHA256_MISMATCH")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    d33.verify_loaded_payload(
        payload, champion.update, champion.h1, champion.h32,
        int(payload["parent_update"]), str(payload["parent_checkpoint_sha256"]),
    )
    model, optimizer, rng, canonical = d33.instantiate_state(runtime, payload)
    return model, optimizer, rng, canonical, payload


def recovery(alpha: float = 0.03125, inward: float = 0.005) -> dict[str, Any]:
    return {
        "family": "D34_REALIZED_ACTIVE_SET_PCGRAD", "base": "H1", "alpha": alpha,
        "inward_cosine": inward, "lambda": inward,
    }


def pcgrad(alpha: float) -> dict[str, Any]:
    return {"family": "D34_BLOCKWISE_PCGRAD", "base": "H1", "alpha": alpha}


def trajectory_profiles(diversification: int, generation: int = 0) -> dict[str, list[dict[str, Any]]]:
    pair = [recovery(), pcgrad(0.0390625)]
    if generation >= 3 and diversification == 0:
        return {
            "EXPLOIT_ACTIVE_SETUP": [recovery(0.0390625, 0.002)],
            "EXPLOIT_PCGRAD_1_128": [pcgrad(0.0078125)],
            "EXPLOIT_DESCENT_RECOVER_X2": [pcgrad(0.015625), recovery(), recovery()],
            "EXPLOIT_DESCENT_RECOVER_X3": [pcgrad(0.015625), recovery(), recovery(), recovery()],
            "EXPLOIT_PAIR_X1": pair,
            "EXPLOIT_PAIR_X2": pair * 2,
            "EXPLOIT_PAIR_X3": pair * 3,
        }
    profiles = {
        "H32_RECHARGE_1_64_IN_005": [recovery(0.015625, 0.005)],
        "H32_RECHARGE_1_32_IN_005": [recovery(0.03125, 0.005)],
        "H32_RECHARGE_3_64_IN_005": [recovery(0.046875, 0.005)],
        "H32_RECHARGE_1_16_IN_005": [recovery(0.0625, 0.005)],
        "H32_RECHARGE_1_32_IN_010": [recovery(0.03125, 0.010)],
        "DIRECT_PCGRAD_1_64": [pcgrad(0.015625)],
        "DIRECT_PCGRAD_1_128": [pcgrad(0.0078125)],
        "DIRECT_PCGRAD_3_256": [pcgrad(0.01171875)],
        "DIRECT_ACTIVE_PCGRAD_5_128_IN_002": [recovery(0.0390625, 0.002)],
        "PCGRAD_1_64_THEN_RECOVER": [pcgrad(0.015625), recovery(0.03125, 0.005)],
        "PCGRAD_1_128_THEN_RECOVER": [pcgrad(0.0078125), recovery(0.03125, 0.005)],
        "PCGRAD_1_64_THEN_RECOVER_X2": [pcgrad(0.015625), recovery(0.03125, 0.005), recovery(0.03125, 0.005)],
        "RECOVER_THEN_PCGRAD": pair,
        "RECOVER_PCGRAD_X2": pair * 2,
        "RECOVER_PCGRAD_X4": pair * 4,
        "RECOVER_PCGRAD_X8": pair * 8,
        "DEEP_RECHARGE_THEN_DESCENT": [recovery(0.0625, 0.005), pcgrad(0.0625)],
    }
    if diversification >= 1:
        profiles.update({
            "ACTIVE_FINE_A0078_IN0001": [recovery(0.0078125, 0.0001)],
            "ACTIVE_FINE_A0078_IN0005": [recovery(0.0078125, 0.0005)],
            "ACTIVE_FINE_A0078_IN0010": [recovery(0.0078125, 0.0010)],
            "ACTIVE_FINE_A0156_IN0005": [recovery(0.015625, 0.0005)],
            "ACTIVE_FINE_A0156_IN0010": [recovery(0.015625, 0.0010)],
            "BLOCKWISE_TANGENT_MICRO": [{"family": "D34_BLOCKWISE_TANGENT", "base": "H1", "alpha": 0.015625}],
            "CAGRAD_H32_CORRECTION": [{"family": "D31_BASE_PLUS_H32_CORRECTION", "base": "CAGRAD", "alpha": 0.03125, "lambda": 0.4}],
            "REALIZED_H32_CORRECTION": [{"family": "REALIZED_ADAMW_H32_CORRECTION", "base": "H1", "alpha": 0.03125, "kappa": 0.02}],
            "P4_H32_CORRECTION_010": [{"family": "D31_BASE_PLUS_H32_CORRECTION", "base": "P4", "alpha": 0.015625, "lambda": 0.10}],
            "P4_H32_CORRECTION_020": [{"family": "D31_BASE_PLUS_H32_CORRECTION", "base": "P4", "alpha": 0.015625, "lambda": 0.20}],
            "H1_H32_PRESENTED_100": [{"family": "H1_PLUS_H32_PRESENTED", "base": "H1", "alpha": 0.0078125, "lambda": 1.0}],
            "H1_H32_PRESENTED_200": [{"family": "H1_PLUS_H32_PRESENTED", "base": "H1", "alpha": 0.0078125, "lambda": 2.0}],
        })
    if diversification >= 2:
        for inward in (0.001, 0.003, 0.008, 0.016):
            profiles[f"ACTIVE_SET_INWARD_{inward:g}"] = [recovery(0.03125, inward)]
        for scale in (0.0078125, 0.0234375, 0.046875):
            profiles[f"PCGRAD_SCALE_{scale:g}"] = [pcgrad(scale)]
    return profiles


def run_path(runtime: Mapping[str, Any], champion: Champion, specs: Sequence[Mapping[str, Any]]):
    model, optimizer, rng, canonical, _ = load_branch(runtime, champion)
    update, h1, h32 = champion.update, champion.h1, champion.h32
    records: list[dict[str, Any]] = []
    for spec in specs:
        child = update + 1
        if child > len(runtime["authority"]["schedule_rows"]):
            raise D35InvariantError("AUTHENTICATED_SCHEDULE_EXHAUSTED")
        identity = d32.semantic_parent_identity(model, optimizer, rng)
        record, model, optimizer, rng = d32.run_trial(
            runtime, model, optimizer, rng, canonical, identity,
            update, child, h1, h32, spec,
        )
        record["d35_spec"] = dict(spec)
        record["parent_champion_update"] = champion.update
        record["parent_champion_generation"] = champion.generation
        record["parent_champion_checkpoint"] = champion.checkpoint
        record["parent_champion_sha256"] = champion.checkpoint_sha256
        records.append(record)
        update, h1, h32 = child, float(record["candidate_H1"]), float(record["candidate_H32"])
    return records, model, optimizer, rng, canonical


def path_safety(records: Sequence[Mapping[str, Any]]) -> bool:
    return all(
        bool(row.get("finite_state_pass"))
        and bool(row.get("optimizer_consistency_pass"))
        and bool(row.get("rng_identity_pass"))
        and float(row["candidate_H32"]) <= H32_LIMIT
        for row in records
    )


def screen_wave(runtime: Mapping[str, Any], manager: ChampionManager, champion: Champion, state: Mapping[str, Any]):
    wave = int(state["exploration_wave"]) + 1
    profiles = trajectory_profiles(int(state["diversification_level"]), champion.generation)
    path = manager.root / "shadows" / f"wave_{wave:04d}_generation_{champion.generation:04d}.json"
    results: list[dict[str, Any]] = []
    prior_ledger = json.loads(manager.ledger_path.read_text(encoding="utf-8"))
    completed = {
        str(item.get("trajectory_id"))
        for item in prior_ledger
        if int(item.get("parent_champion_generation", -1)) == champion.generation
        and item.get("replay_status") in ("NOT_RUN_SCREEN_ONLY", "SHADOW_FAILURE", "PASS")
    }
    original = d32.build_presented_gradient
    d32.build_presented_gradient = d34.build_d34_presented_gradient
    try:
        for trajectory_id, specs in profiles.items():
            if trajectory_id in completed:
                continue
            try:
                records, _, _, _, _ = run_path(runtime, champion, specs)
                final = records[-1]
                item = {
                    "candidate_id": f"g{champion.generation}-w{wave}-{trajectory_id}",
                    "parent_champion_update": champion.update,
                    "parent_champion_generation": champion.generation,
                    "parent_champion_checkpoint": champion.checkpoint,
                    "parent_champion_sha256": champion.checkpoint_sha256,
                    "trajectory_id": trajectory_id, "specs": specs, "steps": records,
                    "endpoint_update": int(final["candidate_update"]),
                    "endpoint_H1": float(final["candidate_H1"]),
                    "endpoint_H32": float(final["candidate_H32"]),
                    "endpoint_H32_margin": H32_LIMIT - float(final["candidate_H32"]),
                    "endpoint_delta_H1": float(final["candidate_H1"]) - champion.h1,
                    "all_intermediate_hard_safety_pass": path_safety(records),
                    "endpoint_strictly_better": float(final["candidate_H1"]) < champion.h1,
                    "endpoint_adaptive_reserve_pass": H32_LIMIT - float(final["candidate_H32"]) >= MIN_CHAMPION_H32_RESERVE,
                    "replay_status": "NOT_RUN_SCREEN_ONLY", "promotion": False,
                }
            except Exception as exc:
                item = {
                    "candidate_id": f"g{champion.generation}-w{wave}-{trajectory_id}",
                    "parent_champion_update": champion.update,
                    "parent_champion_generation": champion.generation,
                    "parent_champion_checkpoint": champion.checkpoint,
                    "parent_champion_sha256": champion.checkpoint_sha256,
                    "trajectory_id": trajectory_id, "specs": specs,
                    "replay_status": "SHADOW_FAILURE", "promotion": False,
                    "rejection_reason": f"{type(exc).__name__}:{exc}",
                }
            results.append(item)
            manager.record_candidate(item)
            atomic_json(path, results)
            print(
                f"D35_SHADOW generation={champion.generation} wave={wave} trajectory={trajectory_id} "
                f"H1={item.get('endpoint_H1')} H32={item.get('endpoint_H32')} "
                f"safe={item.get('all_intermediate_hard_safety_pass')}", flush=True,
            )
    finally:
        d32.build_presented_gradient = original
    serious = [
        item for item in results
        if item.get("all_intermediate_hard_safety_pass") and item.get("endpoint_strictly_better")
        and item.get("endpoint_adaptive_reserve_pass")
    ]
    serious.sort(key=lambda item: float(item["endpoint_H1"]))
    registry = json.loads(manager.registry_path.read_text(encoding="utf-8"))
    serious_ids = {str(item["trajectory_id"]) for item in serious}
    for item in results:
        trajectory_id = str(item["trajectory_id"])
        previous = dict(registry.get(trajectory_id, {}))
        delta = item.get("endpoint_delta_H1")
        best_delta = previous.get("best_delta_H1")
        if delta is not None and (best_delta is None or float(delta) < float(best_delta)):
            best_delta = float(delta)
        status = (
            "PROMISING" if trajectory_id in serious_ids else
            "FAILED_CURRENT_GEOMETRY" if item.get("replay_status") == "SHADOW_FAILURE" else
            "RESEARCHED_SHADOW_TESTED"
        )
        registry[trajectory_id] = {
            **previous,
            "method": trajectory_id,
            "source": "D34_PRIOR_AND_D35_ADAPTATION",
            "method_family": sorted({str(spec["family"]) for spec in item.get("specs", [])}),
            "champion_tested_from": champion.update,
            "champion_generation": champion.generation,
            "best_delta_H1": best_delta,
            "endpoint_H32": item.get("endpoint_H32"),
            "H32_margin": item.get("endpoint_H32_margin"),
            "replay_result": item.get("replay_status"),
            "status": status,
            "failure_mode": item.get("rejection_reason", "NONE"),
            "retest_recommendation": "RETEST_AFTER_CHAMPION_CHANGE" if status == "FAILED_CURRENT_GEOMETRY" else "ACTIVE",
        }
    atomic_json(manager.registry_path, registry)
    state = manager.load_state()
    state["exploration_wave"] = wave
    state["active_methods"] = sorted({str(spec["family"]) for specs in profiles.values() for spec in specs})
    state["promising_methods"] = sorted({str(item["trajectory_id"]) for item in serious[:3]})
    state["failed_current_geometry_methods"] = sorted(
        key for key, value in registry.items()
        if value.get("champion_generation") == champion.generation and value.get("status") == "FAILED_CURRENT_GEOMETRY"
    )
    state["last_completed_action"] = "BROAD_SHADOW_EXPLORATION"
    state["next_intended_action"] = "INDEPENDENT_REPLAY" if serious else "DIVERSIFY_AND_LAUNCH_NEXT_WAVE"
    atomic_json(manager.state_path, state)
    return serious[0] if serious else None


def prepare_replayed_candidate(runtime: Mapping[str, Any], manager: ChampionManager, champion: Champion, screened: Mapping[str, Any]) -> Candidate:
    specs = list(screened["specs"])
    original = d32.build_presented_gradient
    d32.build_presented_gradient = d34.build_d34_presented_gradient
    try:
        first, model, optimizer, rng, canonical = run_path(runtime, champion, specs)
        replay, replay_model, replay_optimizer, replay_rng, _ = run_path(runtime, champion, specs)
    finally:
        d32.build_presented_gradient = original
    replay_check = d32.d27.replay_compare(
        first[-1], replay[-1], model, replay_model,
        optimizer, replay_optimizer, rng, replay_rng,
    )
    final = dict(first[-1])
    final.update({
        "deterministic_replay_consistency_pass": bool(replay_check["pass"]),
        "deterministic_replay_status": "PASS" if replay_check["pass"] else "FAIL",
        "deterministic_replay_candidate_record_digest": replay_check["candidate_record_digest"],
        "deterministic_replay_record_digest": replay_check["replay_record_digest"],
        "candidate_valid": bool(replay_check["pass"] and path_safety(first) and float(first[-1]["candidate_H1"]) < champion.h1),
        "selection_status": "D35_PREPARED_FOR_PRIMARY_PROMOTION",
        "trial_state": "PREPARED_FORWARD_STATE",
    })
    if not final["candidate_valid"]:
        raise D35InvariantError("D35_SERIOUS_CANDIDATE_REPLAY_OR_SAFETY_FAILED")
    payload = d32.checkpoint_payload(model, optimizer, rng, final, champion.checkpoint_sha256, canonical)
    payload.update({
        "schema_version": "stage3_h13_d35_resumable_champion_checkpoint_v1",
        "experiment_id": "D35_PERMANENT_CHAMPION_SNOWBALL",
        "mode": "D35_REPLAY_VALIDATED_TRAJECTORY_ENDPOINT",
        "parent_update": champion.update,
        "parent_checkpoint_sha256": champion.checkpoint_sha256,
        "champion_parent_generation": champion.generation,
        "champion_trajectory_updates": [int(row["candidate_update"]) for row in first],
        "champion_trajectory_length": len(first),
        "champion_trajectory_id": str(screened["trajectory_id"]),
        "trajectory_evidence": copy.deepcopy(first),
        "replay_consistency": replay_check,
        "scientific_state": "COMMITTED_FORWARD_STATE",
        "resumable": True,
    })
    prepared = manager.root / "prepared" / f"candidate_update_{int(final['candidate_update'])}.pt.tmp"
    prepared.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, prepared)
    candidate = Candidate(
        candidate_id=str(screened["candidate_id"]), parent_update=champion.update,
        parent_generation=champion.generation, parent_h1=champion.h1,
        parent_checkpoint=champion.checkpoint, parent_checkpoint_sha256=champion.checkpoint_sha256,
        update=int(final["candidate_update"]), h1=float(final["candidate_H1"]), h32=float(final["candidate_H32"]),
        method=str(final["candidate_family"]), trajectory=str(screened["trajectory_id"]),
        checkpoint=str(prepared), replay_pass=bool(replay_check["pass"]),
        optimizer_consistency_pass=all(bool(row["optimizer_consistency_pass"]) for row in first),
        rng_lineage_pass=all(bool(row["rng_identity_pass"]) for row in first),
        scheduler_continuity_pass=int(final["candidate_update"]) == champion.update + len(first),
        checkpoint_reproducibility_pass=True,
    )
    manager.record_candidate({
        **screened, "replay_status": "PASS", "prepared_checkpoint": str(prepared),
        "replay": replay_check, "endpoint_H1": candidate.h1, "endpoint_H32": candidate.h32,
    })
    return candidate


def verify_checkpoint(path: Path, candidate: Candidate) -> bool:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        states = payload["optimizer_state_dict"]["state"].values()
        return bool(
            int(payload["completed_optimizer_step"]) == candidate.update
            and int(payload["next_schedule_index"]) == candidate.update
            and int(payload["parent_update"]) == candidate.parent_update
            and str(payload["parent_checkpoint_sha256"]) == candidate.parent_checkpoint_sha256
            and float(payload["H1"]) == candidate.h1
            and float(payload["H32"]) == candidate.h32
            and payload.get("scientific_state") == "COMMITTED_FORWARD_STATE"
            and payload.get("resumable") is True
            and payload.get("replay_consistency", {}).get("pass") is True
            and bool(states)
            and all(int(state.get("step", -1)) == candidate.update for state in states)
        )
    except Exception:
        return False


class ScientificWave:
    def __init__(self, runtime: Mapping[str, Any], manager: ChampionManager, stop: ExternalStop, one_wave: bool):
        self.runtime, self.manager, self.stop, self.one_wave = runtime, manager, stop, one_wave
        self.calls = 0

    def __call__(self, champion: Champion, state: Mapping[str, Any]) -> Candidate | None:
        if self.stop.requested:
            self.manager.mark_external_interruption("RESUME_CURRENT_SNOWBALL_CYCLE")
            return None
        screened = screen_wave(self.runtime, self.manager, champion, state)
        self.calls += 1
        if screened is None:
            if self.one_wave:
                self.manager.mark_external_interruption("DIVERSIFY_AND_LAUNCH_NEXT_WAVE")
            return None
        candidate = prepare_replayed_candidate(self.runtime, self.manager, champion, screened)
        if self.one_wave:
            # The promotion remains valid; the controller continues once more and
            # then records the explicit operator boundary as resumable interruption.
            self.stop.requested = True
        return candidate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--one-wave", action="store_true")
    args = parser.parse_args()
    manager = ChampionManager(args.output)
    manager.initialize(authenticate_start())
    state = manager.load_state()
    if state.get("external_interruption"):
        state["external_interruption"] = False
        state["interrupted_resumable"] = False
        atomic_json(manager.state_path, state)
    runtime = d32.d24.d17.preflight_runtime()
    stop = ExternalStop()
    signal.signal(signal.SIGINT, stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, stop)
    wave = ScientificWave(runtime, manager, stop, args.one_wave)
    reason = run_persistent_outer_loop(manager, wave, verify_checkpoint)
    state = manager.load_state()
    print(json.dumps({
        "TASK_STATUS": "PASS" if reason == "TARGET_REACHED" else ("BLOCKED" if reason == "TRUE_BLOCKER" else "INTERRUPTED_RESUMABLE"),
        "TERMINATION_REASON": reason,
        "SCIENTIFIC_EXECUTION_COMPLETE": reason in ("TARGET_REACHED", "TRUE_BLOCKER"),
        "CURRENT_CHAMPION": state["current_champion"],
        "TARGET_H1": TARGET_H1,
        "REMAINING_GAP": state["remaining_gap"],
    }, sort_keys=True), flush=True)
    return 0 if reason == "TARGET_REACHED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
