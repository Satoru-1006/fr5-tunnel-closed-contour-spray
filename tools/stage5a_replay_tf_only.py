"""Re-run exact D65 goals only to repair synchronized ROS TF capture."""

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
    replays: dict[str, object] = {}
    for name in ("auto0", "auto1"):
        replays[name] = stage5.execute_replay(
            root,
            name,
            trajectories[name],
            json.loads((root / f"stage5a_{name}_goal.json").read_text(encoding="utf-8")),
            run_dir_name=f"replay_{name}_tf100",
        )
    state["replays"] = replays
    stage5.write_json(state_path, state)
    print(json.dumps({"output_root": str(root), "auto0": replays["auto0"]["action"], "auto1": replays["auto1"]["action"]}, ensure_ascii=False))
    return 0 if all(item.get("action", {}).get("goal_accepted") and item.get("action", {}).get("execution_completed") for item in replays.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
