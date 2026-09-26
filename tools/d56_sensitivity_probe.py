"""Native FK/FCL hotspot sensitivity probe for D56.

It creates only tiny, explicitly labelled shadow trajectories around the D54
adversarial clearance and singularity hotspots.  The native D41 executable is
run separately by the calling command; this script only materializes inputs
and a manifest under the D56 controlled measurement sandbox.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.d52_motion_repair import read_native


JOINTS = tuple(f"j{i}" for i in range(1, 7))


def wsl_path(path: Path) -> str:
    value = str(path.resolve()).replace("\\", "/")
    if len(value) >= 3 and value[1:3] == ":/":
        return f"/mnt/{value[0].lower()}/{value[3:]}"
    return value


def write_q(path: Path, states: list[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["waypoint", *[f"{joint}_q" for joint in JOINTS]])
        for index, state in enumerate(states):
            writer.writerow([index, *[float(value) for value in state]])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--waypoint", type=int, required=True)
    parser.add_argument("--delta", type=float, default=1.0e-3)
    args = parser.parse_args()

    source = read_native(args.source.resolve())
    if args.waypoint <= 0 or args.waypoint >= len(source.rows):
        raise ValueError(f"waypoint_out_of_range:{args.waypoint}:{len(source.rows)}")
    center = source.q[args.waypoint].copy()
    output = args.output.resolve()
    cases = []
    for joint in range(6):
        for sign in (-1, 1):
            case_id = f"adversarial_hotspot_j{joint + 1}_{'minus' if sign < 0 else 'plus'}"
            path = output / "inputs" / f"{case_id}.csv"
            shifted = center.copy()
            shifted[joint] += sign * args.delta
            write_q(path, [center, shifted, center])
            cases.append({
                "case_id": case_id,
                "family": "D56_SENSITIVITY",
                "trajectory_csv": wsl_path(path),
                "joint_index_zero_based": joint,
                "delta_rad": sign * args.delta,
            })

    manifest = output / "sensitivity.manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "trajectory_csv", "family"], lineterminator="\n")
        writer.writeheader()
        for row in cases:
            writer.writerow({key: row[key] for key in ("case_id", "trajectory_csv", "family")})
    (output / "sensitivity_definition.json").write_text(
        json.dumps({
            "schema_version": "d56-native-hotspot-sensitivity-v1",
            "source_trajectory": str(args.source.resolve()),
            "center_waypoint": args.waypoint,
            "center_q_rad": center.tolist(),
            "delta_rad": args.delta,
            "method": "native_MoveIt2_FK_and_FCL_distance_at_center_plus_minus_joint_perturbation",
            "collision_method": "adaptive_discrete_interpolation",
            "cases": cases,
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", "case_count": len(cases), "manifest": str(manifest)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
