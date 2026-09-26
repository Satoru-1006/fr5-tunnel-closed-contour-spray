"""Apply one evidence-driven smooth local joint-path deformation.

This is a controlled-box helper for D56 experiments.  It preserves the source
time grid and all other cases, and writes a separate candidate tree.  The
deformation is C4 at its compact-support boundary so that the subsequent
native Ruckig pass remains the authority for the final state derivatives.
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

from tools.d52_motion_repair import CASES, JOINTS, NativeTrajectory, read_native, write_native  # noqa: E402


def wsl_path(path: Path) -> str:
    value = str(path.resolve()).replace("\\", "/")
    if len(value) >= 3 and value[1:3] == ":/":
        return f"/mnt/{value[0].lower()}/{value[3:]}"
    return value


def c4_bump(time: np.ndarray, center: float, half_width: float, amplitude: float):
    u = (time - center) / half_width
    mask = np.abs(u) < 1.0
    polynomial = np.poly1d([-1.0, 0.0, 1.0]) ** 4
    derivatives = [polynomial]
    derivatives.extend(np.polyder(polynomial, order) for order in (1, 2, 3))
    values = []
    for order, derivative in enumerate(derivatives):
        value = np.zeros_like(time)
        value[mask] = amplitude * derivative(u[mask]) / (half_width**order)
        values.append(value)
    return tuple(values)


def case_family(case_id: str) -> str:
    if case_id.startswith("collision_sensitive"):
        return "COLLISION_SENSITIVE"
    return case_id.split("_", 1)[0].upper()


def build_candidate(source_dir: Path, output_dir: Path, candidate: str, case_id: str,
                    joint_index: int, center: float, half_width: float, amplitude: float) -> dict:
    if case_id not in CASES:
        raise ValueError(f"unknown_case:{case_id}")
    if not 0 <= joint_index < len(JOINTS):
        raise ValueError(f"joint_index_out_of_range:{joint_index + 1}")
    if half_width <= 0.0 or not np.isfinite(half_width):
        raise ValueError("half_width_must_be_positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for current_case in CASES:
        source = source_dir / f"{current_case}.csv"
        trajectory = read_native(source)
        rows = [dict(row) for row in trajectory.rows]
        applied = None
        if current_case == case_id:
            time = trajectory.time
            bump_q, bump_dq, bump_ddq, bump_jerk = c4_bump(time, center, half_width, amplitude)
            joint = JOINTS[joint_index]
            for index, row in enumerate(rows):
                row[f"{joint}_q"] += float(bump_q[index])
                row[f"{joint}_dq"] += float(bump_dq[index])
                row[f"{joint}_ddq"] += float(bump_ddq[index])
                if f"{joint}_jerk" in row:
                    row[f"{joint}_jerk"] += float(bump_jerk[index])
            applied = {
                "joint": joint,
                "center_s": center,
                "half_width_s": half_width,
                "amplitude_rad": amplitude,
                "basis": "(1-u^2)^4",
                "max_basis_jerk_abs": 31.620710374489313,
                "peak_added_jerk_abs_rad_s3": 31.620710374489313 * abs(amplitude) / (half_width**3),
            }
        destination = output_dir / f"{current_case}.csv"
        write_native(destination, NativeTrajectory(trajectory.fields, tuple(rows)))
        records.append({
            "case_id": current_case,
            "source": str(source.resolve()),
            "destination": str(destination.resolve()),
            "applied_bump": applied,
            "untouched": applied is None,
        })
    manifest = output_dir.parent / f"{candidate}.manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["case_id", "trajectory_csv", "family"])
        for record in records:
            writer.writerow([record["case_id"], wsl_path(Path(record["destination"])), case_family(record["case_id"])])
    return {
        "schema_version": "d56-targeted-c4-geometry-bump-v1",
        "candidate": candidate,
        "source_dir": str(source_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "manifest": str(manifest.resolve()),
        "method": "single_native_sensitivity_guided_C4_joint_path_bump",
        "collision_method": "adaptive_discrete_interpolation",
        "target_case": case_id,
        "target_joint": JOINTS[joint_index],
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--case", default="adversarial_0101")
    parser.add_argument("--joint", type=int, default=4, help="one-based active joint index")
    parser.add_argument("--center", type=float, required=True)
    parser.add_argument("--half-width", type=float, default=1.5)
    parser.add_argument("--amplitude", type=float, required=True)
    args = parser.parse_args()
    payload = build_candidate(args.source_dir.resolve(), args.output_dir.resolve(), args.candidate, args.case,
                              args.joint - 1, args.center, args.half_width, args.amplitude)
    definition = args.output_dir.resolve().parent / f"{args.candidate}.definition.json"
    definition.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "candidate": args.candidate, "manifest": payload["manifest"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
