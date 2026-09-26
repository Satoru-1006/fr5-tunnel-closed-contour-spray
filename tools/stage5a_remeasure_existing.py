"""Recompute only repaired Stage 5A measurement domains for an existing run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import stage5a_mock_execution as stage5


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--skip-replay", action="store_true", help="reuse already completed TF-repaired full replays")
    args = parser.parse_args()
    state_path = args.state.resolve()
    state = json.loads(state_path.read_text(encoding="utf-8"))
    root = Path(state["output_root"]).resolve()
    trajectories = stage5.load_trajectories()
    environment = Path(state["planning_scene_environment"]).resolve()
    parts = root / "stage5a_bullet_parts"
    repaired_replays: dict[str, object] = {}
    for name in ("auto0", "auto1"):
        replay_dir = root / f"replay_{name}_tf_repaired"
        if args.skip_replay:
            action_path = replay_dir / "stage5a_action_execution.json"
            ready_path = replay_dir / "stage5a_runtime_ready.json"
            qmetrics_path = replay_dir / "stage5a_q_cmd_q_mock.json"
            repaired_replays[name] = {
                "name": name,
                "smoke": False,
                "ready": json.loads(ready_path.read_text(encoding="utf-8")),
                "action": json.loads(action_path.read_text(encoding="utf-8")),
                "run_dir": str(replay_dir.resolve()),
                "q_cmd_q_mock": json.loads(qmetrics_path.read_text(encoding="utf-8")),
            }
        else:
            repaired_replays[name] = stage5.execute_replay(
                root, name, trajectories[name],
                json.loads((root / f"stage5a_{name}_goal.json").read_text(encoding="utf-8")),
                run_dir_name=f"replay_{name}_tf_repaired",
            )
    state["replays"] = repaired_replays
    repaired_moveit: dict[str, object] = {}
    repaired_bullet: dict[str, object] = {}
    for name in ("auto0", "auto1"):
        replay_dir = Path(state["replays"][name]["run_dir"]).resolve()
        repaired_moveit[name] = stage5.run_moveit_validation(
            root, name, trajectories[name], environment, replay_dir,
            output_name=f"moveit_{name}_parallel",
        )
        repaired_bullet[name] = stage5.run_bullet_parallel(
            root, name, trajectories[name], parts,
            output_name=f"bullet_{name}_parallel",
        )
    state["moveit"] = repaired_moveit
    state["bullet"] = repaired_bullet
    stage5.write_json(state_path, state)
    validator = stage5.run_command(
        f"{stage5.ros_prefix()} && python3 {stage5.ROOT / 'tools/stage5_mock_validator.py'} --state {state_path}",
        120,
    )
    (root / "stage5a_final_validator_stdout.log").write_text(validator["stdout"], encoding="utf-8")
    (root / "stage5a_final_validator_stderr.log").write_text(validator["stderr"], encoding="utf-8")
    ledger = json.loads((root / "STAGE5A_FINAL_LEDGER.json").read_text(encoding="utf-8")) if (root / "STAGE5A_FINAL_LEDGER.json").exists() else {"status": "FAIL_CLOSED"}
    print(json.dumps({"status": ledger["status"], "output_root": str(root)}, ensure_ascii=False))
    return 0 if ledger["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
