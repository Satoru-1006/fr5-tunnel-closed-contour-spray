"""Generate evidence-driven D56 geometry shadows from native trajectories.

The directions are taken from native MoveIt2 FK/FCL and Jacobian sensitivity
probes, not guessed joint perturbations.  Every displacement is a compact C4
polynomial with zero displacement, velocity, acceleration and jerk at its
support boundary; endpoint poses and the protected source files are untouched.
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

from tools.d52_motion_repair import CASES, JOINTS, read_native  # noqa: E402


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
        value[mask] = amplitude * derivative(u[mask]) / (half_width ** order)
        values.append(value)
    return tuple(values)


def write_native(path: Path, fields: tuple[str, ...], time, q, dq, ddq, jerk) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(fields)
        for index in range(len(time)):
            writer.writerow([float(time[index]), *q[index], *dq[index], *ddq[index], *jerk[index]])


def build_candidate(source_dir: Path, output_dir: Path, name: str, amplitude_scale: float = 1.0) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = {
        # Native hotspot sensitivity: j4 negative increased endpoint
        # forearm|wrist2 distance, while j5 positive gave a smaller gain.
        "adversarial_0100": [(3, 37.037636992, 1.50, -0.005), (4, 37.037636992, 1.50, 0.005)],
        # Native Jacobian sensitivity: j2 positive lifted the controlling
        # sigma_min at perturbation_0100 waypoint 4266.
        "perturbation_0100": [(1, 43.697636992, 1.50, 0.001)],
    }
    records = []
    for case_id in CASES:
        source = source_dir / f"{case_id}.csv"
        trajectory = read_native(source)
        time = trajectory.time
        q = trajectory.q.copy()
        dq = trajectory.dq.copy()
        ddq = trajectory.ddq.copy()
        jerk = np.asarray([[row.get(f"j{joint}_jerk", 0.0) for joint in range(1, 7)] for row in trajectory.rows], dtype=float)
        if not np.isfinite(jerk).all():
            jerk = np.gradient(ddq, time, axis=0, edge_order=1)
        applied = []
        for joint, center, width, amplitude in specs.get(case_id, []):
            amplitude *= float(amplitude_scale)
            bump_q, bump_dq, bump_ddq, bump_jerk = c4_bump(time, center, width, amplitude)
            q[:, joint] += bump_q
            dq[:, joint] += bump_dq
            ddq[:, joint] += bump_ddq
            jerk[:, joint] += bump_jerk
            applied.append({"joint": f"j{joint + 1}", "center_s": center, "half_width_s": width, "amplitude_rad": amplitude,
                            "basis": "(1-u^2)^4", "max_basis_jerk_abs": 31.620710374489313,
                            "amplitude_scale": float(amplitude_scale)})
        destination = output_dir / f"{case_id}.csv"
        write_native(destination, trajectory.fields, time, q, dq, ddq, jerk)
        records.append({
            "case_id": case_id,
            "source": str(source.resolve()),
            "destination": str(destination.resolve()),
            "applied_repairs": applied,
            "endpoint_q_preserved": bool(np.array_equal(q[[0, -1]], trajectory.q[[0, -1]])),
            "endpoint_dq_preserved": bool(np.array_equal(dq[[0, -1]], trajectory.dq[[0, -1]])),
            "endpoint_ddq_preserved": bool(np.array_equal(ddq[[0, -1]], trajectory.ddq[[0, -1]])),
        })
    manifest = output_dir.parent / f"{name}.manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["case_id", "trajectory_csv", "family"])
        for row in records:
            family = case_id_family(row["case_id"])
            writer.writerow([row["case_id"], wsl_path(Path(row["destination"])), family])
    return {"schema_version": "d56-c4-geometry-shadow-v1", "candidate": name, "records": records,
            "manifest": str(manifest.resolve()), "method": "native_sensitivity_guided_C4_local_joint_path_reshaping",
            "collision_method": "adaptive_discrete_interpolation"}


def case_id_family(case_id: str) -> str:
    if case_id.startswith("collision_sensitive"):
        return "COLLISION_SENSITIVE"
    return case_id.split("_")[0].upper()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate", default="c4_sensitivity_hybrid")
    parser.add_argument("--amplitude-scale", type=float, default=1.0)
    args = parser.parse_args()
    if args.amplitude_scale <= 0.0:
        raise ValueError("amplitude_scale_must_be_positive")
    payload = build_candidate(args.source_dir.resolve(), args.output_dir.resolve(), args.candidate, args.amplitude_scale)
    (args.output_dir.resolve().parent / f"{args.candidate}.definition.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", "candidate": args.candidate, "manifest": payload["manifest"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
