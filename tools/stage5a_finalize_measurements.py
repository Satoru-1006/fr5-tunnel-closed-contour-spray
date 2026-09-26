"""Finalize Stage 5A after repaired full replay and native Bullet measurements."""

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
    args = parser.parse_args()
    state_path = args.state.resolve()
    state = json.loads(state_path.read_text(encoding="utf-8"))
    root = Path(state["output_root"]).resolve()
    trajectories = stage5.load_trajectories()
    environment = Path(state["planning_scene_environment"]).resolve()
    moveit: dict[str, object] = {}
    for name in ("auto0", "auto1"):
        replay_dir = Path(state["replays"][name]["run_dir"]).resolve()
        moveit[name] = stage5.run_moveit_validation(
            root, name, trajectories[name], environment, replay_dir,
            output_name=f"moveit_{name}_final_causal",
        )
    state["moveit"] = moveit
    regression = stage5.run_command(
        f"{stage5.ros_prefix()} && python3 -m py_compile "
        f"{ROOT / 'src/stage5a_trajectory_adapter.py'} "
        f"{ROOT / 'scripts/stage5a_runtime_client.py'} "
        f"{ROOT / 'scripts/stage5a_mock_execution.py'} "
        f"{ROOT / 'scripts/stage5a_moveit_validator.py'} "
        f"{ROOT / 'tools/stage5a_expand_xacro.py'} "
        f"{ROOT / 'tools/stage5a_moveit_validation_launch.py'} "
        f"{ROOT / 'tools/stage5a_bullet_launch.py'} "
        f"{ROOT / 'tools/stage5a_remeasure_existing.py'} "
        f"{ROOT / 'tools/stage5a_replay_tf_only.py'} "
        f"{ROOT / 'tools/stage5a_finalize_measurements.py'} && python3 -m pytest -q tests/test_stage5a_contract.py",
        900,
    )
    state["regression"] = {key: regression[key] for key in ("exit_code", "timed_out", "elapsed_s")}
    stage5.write_json(state_path, state)
    validator = stage5.run_command(
        f"{stage5.ros_prefix()} && python3 {ROOT / 'tools/stage5_mock_validator.py'} --state {state_path}",
        120,
    )
    (root / "stage5a_final_validator_stdout.log").write_text(validator["stdout"], encoding="utf-8")
    (root / "stage5a_final_validator_stderr.log").write_text(validator["stderr"], encoding="utf-8")
    ledger = json.loads((root / "STAGE5A_FINAL_LEDGER.json").read_text(encoding="utf-8")) if (root / "STAGE5A_FINAL_LEDGER.json").exists() else {"status": "FAIL_CLOSED"}
    print(json.dumps({"status": ledger["status"], "output_root": str(root)}, ensure_ascii=False))
    return 0 if ledger["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
