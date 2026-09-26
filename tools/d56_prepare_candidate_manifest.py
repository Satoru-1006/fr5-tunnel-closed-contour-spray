"""Write a D56 shadow manifest for the fixed 12-case Stage 4B suite."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.d52_motion_repair import JOINTS, read_native


CASES = (
    ("regression_0000", "REGRESSION"), ("regression_0001", "REGRESSION"),
    ("normal_0000", "NORMAL"), ("normal_0100", "NORMAL"),
    ("boundary_0000", "BOUNDARY"), ("boundary_0100", "BOUNDARY"),
    ("collision_sensitive_0000", "COLLISION_SENSITIVE"),
    ("collision_sensitive_0051", "COLLISION_SENSITIVE"),
    ("adversarial_0100", "ADVERSARIAL"), ("adversarial_0101", "ADVERSARIAL"),
    ("perturbation_0000", "PERTURBATION"), ("perturbation_0100", "PERTURBATION"),
)


def wsl_path(path: Path) -> str:
    value = str(path.resolve()).replace("\\", "/")
    if len(value) >= 3 and value[1:3] == ":/":
        return f"/mnt/{value[0].lower()}/{value[3:]}"
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--q-only-dir", type=Path, default=None)
    parser.add_argument("--case-ids", default=None)
    args = parser.parse_args()
    selected = {value for value in args.case_ids.split(",") if value} if args.case_ids else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("case_id", "trajectory_csv", "family"))
        for case_id, family in CASES:
            if selected is not None and case_id not in selected:
                continue
            trajectory = args.trajectory_dir.resolve() / f"{case_id}.csv"
            if not trajectory.is_file():
                raise FileNotFoundError(trajectory)
            manifest_trajectory = trajectory
            if args.q_only_dir is not None:
                native = read_native(trajectory)
                manifest_trajectory = args.q_only_dir.resolve() / f"{case_id}.csv"
                manifest_trajectory.parent.mkdir(parents=True, exist_ok=True)
                with manifest_trajectory.open("w", encoding="utf-8", newline="") as q_stream:
                    q_writer = csv.writer(q_stream, lineterminator="\n")
                    q_writer.writerow(("waypoint", *(f"{joint}_q" for joint in JOINTS)))
                    for index, row in enumerate(native.rows):
                        q_writer.writerow((index, *(row[f"{joint}_q"] for joint in JOINTS)))
            writer.writerow((case_id, wsl_path(manifest_trajectory), family))
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
