"""Repair only the persisted entry-transition summary indices."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_stage24c_seam_window_repair as c


def main() -> int:
    summary_path = c.OUT / "stage24c_transition_summary.json"
    summary = c.read_json(summary_path)
    for radius in c.WINDOW_RADII:
        radius_key = str(radius)
        transitions = summary[radius_key]["transitions"]
        entry_from = 720 - radius - 1
        entry_to = 720 - radius
        for key in (f"{719 - radius - 1}->{720 - radius - 1}", f"{entry_from}->{entry_from}"):
            transitions.pop(key, None)
        window_dir = c.OUT / "windows" / f"radius_{radius}"
        fcl = c.read_jsonl(window_dir / "stage24c_entry_edges_fcl.jsonl")
        bullet = c.read_jsonl(window_dir / "stage24c_entry_edges_bullet.jsonl")
        transitions[f"{entry_from}->{entry_to}"] = {
            "fcl_accepted": sum(1 for row in fcl if row.get("status") == "accepted" and row.get("valid") is True),
            "bullet_accepted": sum(1 for row in bullet if row.get("status") == "accepted" and row.get("valid") is True),
            "minimum_model_aware_step_deg": min([float(row["max_model_aware_joint_step_deg"]) for row in fcl if row.get("max_model_aware_joint_step_deg") is not None] or [None]),
        }
    c.write_json(summary_path, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
