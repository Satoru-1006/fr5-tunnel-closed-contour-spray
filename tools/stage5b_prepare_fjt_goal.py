"""Prepare a Stage 5B shadow FollowJointTrajectory goal from a CSV trajectory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


JOINTS = [f"j{i}" for i in range(1, 7)]


def duration_fields(seconds: float) -> dict[str, int]:
    sec = math.floor(seconds)
    nanosec = int(round((seconds - sec) * 1_000_000_000))
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000
    return {"sec": int(sec), "nanosec": int(nanosec), "seconds": float(seconds)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    source = args.trajectory_csv.resolve()
    rows = list(csv.DictReader(source.open(newline="", encoding="utf-8")))
    if not rows:
        raise SystemExit("trajectory CSV is empty")
    points = []
    previous_t = -1.0
    for index, row in enumerate(rows):
        t = float(row["t"])
        if t < previous_t:
            raise SystemExit(f"non-monotonic timestamp at row {index}")
        previous_t = t
        point = {
            "positions": [float(row[f"j{i}_q"]) for i in range(1, 7)],
            "velocities": [float(row[f"j{i}_dq"]) for i in range(1, 7)],
            "accelerations": [float(row[f"j{i}_ddq"]) for i in range(1, 7)],
            "time_from_start": duration_fields(t),
        }
        points.append(point)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "stage5b-fjt-goal-v1",
        "joint_names": JOINTS,
        "points": points,
        "goal_identity": {
            "trajectory_id": "STAGE5AC_ROUTE_E_X_MINUS_050_SHADOW",
            "source_path": str(source),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "source_row_count": len(rows),
            "timestamps_preserved": True,
            "q_dq_ddq_preserved": True,
        },
    }
    output = args.output_dir / "stage5a_follow_joint_trajectory_goal.json"
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PREPARED_STAGE5B_SHADOW_GOAL", "output": str(output), "points": len(points), "duration_s": points[-1]["time_from_start"]["seconds"], "source_sha256": payload["goal_identity"]["source_sha256"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
