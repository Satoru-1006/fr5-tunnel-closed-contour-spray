"""Create compact callback-audit candidates from frozen D65 q states.

This only projects existing trajectory rows into the callback probe's input
format.  It does not alter, resample, or write a replacement trajectory.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


JOINTS = [f"j{i}" for i in range(1, 7)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.trajectory.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("empty trajectory")
    indices = sorted({0, 1, 2, len(rows) // 4, len(rows) // 2, (3 * len(rows)) // 4, len(rows) - 2, len(rows) - 1})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["candidate_id", "waypoint_id", "joint_values"])
        for index in indices:
            q = [float(rows[index][f"{joint}_q"]) for joint in JOINTS]
            writer.writerow([f"d65_state_{index:05d}", index, json.dumps(q, separators=(",", ":"))])
    print(json.dumps({"trajectory": str(args.trajectory.resolve()), "state_count": len(rows), "selected_indices": indices, "output": str(args.output.resolve())}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
