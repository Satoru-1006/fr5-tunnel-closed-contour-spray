"""One-time repair: demote the reserve-insufficient update-470 D35 branch to shadow evidence."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.stage3_h13_d35_champion_manager import Champion, ChampionManager, atomic_json, sha256_file
from tools.stage3_h13_d35_scientific_runner import authenticate_start


def main() -> int:
    manager = ChampionManager(Path("outputs/stage3_h13_d35_permanent_champion"))
    state = manager.load_state()
    current = Champion.from_mapping(state["current_champion"])
    if current.update == 469:
        return 0
    if current.update != 470 or current.checkpoint_sha256 != "446dc23bf51897cfc2e9dd829bc8e79e9d8b48e98d838bf2ff8a9d43514240a7":
        raise RuntimeError("UNEXPECTED_D35_STATE_REFUSING_REPAIR")
    source = Path(current.checkpoint).resolve()
    expected_root = manager.root / "checkpoints"
    if source.parent != expected_root or sha256_file(source) != current.checkpoint_sha256:
        raise RuntimeError("UPDATE_470_REPAIR_SOURCE_AUTHENTICATION_FAILED")
    rejected = manager.root / "rejected_checkpoints" / "update_470_reserve_insufficient.pt"
    rejected.parent.mkdir(parents=True, exist_ok=True)
    if rejected.exists():
        raise RuntimeError("REJECTED_UPDATE_470_DESTINATION_ALREADY_EXISTS")
    os.replace(source, rejected)
    start = authenticate_start()
    repaired = manager._state_for(start)
    repaired["exploration_wave"] = int(state.get("exploration_wave", 0))
    repaired["diversification_level"] = max(1, int(state.get("diversification_level", 0)))
    repaired["last_completed_action"] = "INVALID_PROMOTION_RECLASSIFIED_AS_SHADOW"
    repaired["next_intended_action"] = "RESERVE_AWARE_EXPLORATION_FROM_CHAMPION_469"
    repaired["reserve_gate_correction"] = {
        "rejected_update": 470,
        "reason": "ADAPTIVE_H32_RESERVE_INSUFFICIENT",
        "required_margin": 7.5e-5,
        "actual_margin": current.h32_margin,
        "preserved_shadow_checkpoint": str(rejected),
        "checkpoint_sha256": current.checkpoint_sha256,
    }
    atomic_json(manager.state_path, repaired)
    rows = list(csv.DictReader(manager.history_path.open("r", newline="", encoding="utf-8")))
    rows = [row for row in rows if int(row["champion_update"]) != 470]
    fields = list(rows[0].keys())
    temporary = manager.history_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, manager.history_path)
    transaction = json.loads(manager.transaction_path.read_text(encoding="utf-8"))
    transaction.update({
        "status": "REJECTED_AFTER_RESERVE_GATE_AUDIT",
        "rejection_reason": "ADAPTIVE_H32_RESERVE_INSUFFICIENT",
        "preserved_shadow_checkpoint": str(rejected),
    })
    atomic_json(manager.transaction_path, transaction)
    manager._append_event("INVALID_PROMOTION_RECLASSIFIED_AS_SHADOW", repaired["reserve_gate_correction"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
