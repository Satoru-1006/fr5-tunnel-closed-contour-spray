"""Localized analytic time-dilation experiment for D52.

This is an experimental candidate generator, not a silent replacement for the
native MoveIt2/Ruckig path.  It preserves the repaired native q path and
transforms qdot/qddot using the exact chain rule for a smooth local time map.
The source native post-Ruckig result remains recorded, while downstream
geometry, clearance, and Pinocchio audits decide whether the experiment is
useful.  All outputs are confined to the D52 shadow tree.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.d52_motion_repair import CASES, FAMILIES, JOINTS, NativeTrajectory, read_native, wsl_path, write_native


DEFAULT_HOTSPOTS = ROOT / "outputs" / "D51_STAGE4_SHADOW" / "torque_causality" / "torque_slew_causality.json"


def _smooth_bump(time: np.ndarray, center: float, half_width: float) -> np.ndarray:
    if half_width <= 0.0:
        raise ValueError("half_width_must_be_positive")
    normalized = np.abs((time - float(center)) / float(half_width))
    bump = np.zeros_like(time)
    inside = normalized <= 1.0
    bump[inside] = 0.5 * (1.0 + np.cos(np.pi * normalized[inside]))
    return bump


def load_hotspots(path: Path, case_id: str, limit: int = 3) -> list[float]:
    source = json.loads(path.read_text(encoding="utf-8"))
    for item in source.get("cases", []):
        if str(item.get("case_id")) != case_id:
            continue
        candidate = item.get("candidate", {})
        regions = candidate.get("top_torque_slew_regions", [])
        centers = [float(region["time_start_s"]) for region in regions[:limit] if "time_start_s" in region]
        if centers:
            return centers
    raise ValueError(f"missing_hotspots:{case_id}")


def retime_trajectory(source: Path, destination: Path, centers: list[float], gain: float, half_width: float) -> dict[str, Any]:
    trajectory = read_native(source)
    old_time = trajectory.time
    old_q = trajectory.q
    old_dq = trajectory.dq
    old_ddq = trajectory.ddq
    bump = np.zeros_like(old_time)
    for center in centers:
        bump = np.maximum(bump, _smooth_bump(old_time, center, half_width))
    speed_map = 1.0 + float(gain) * bump
    speed_map_prime = np.gradient(speed_map, old_time, edge_order=1)
    new_time = np.zeros_like(old_time)
    new_time[1:] = np.cumsum(0.5 * (speed_map[:-1] + speed_map[1:]) * np.diff(old_time))
    new_dq = old_dq / speed_map[:, None]
    new_ddq = old_ddq / (speed_map[:, None] ** 2) - old_dq * speed_map_prime[:, None] / (speed_map[:, None] ** 3)
    rows = []
    mutable_fields = set(trajectory.fields)
    diagnostic_jerk = np.gradient(new_ddq, new_time, axis=0, edge_order=1) if len(new_time) >= 3 else np.zeros_like(new_ddq)
    for index, original in enumerate(trajectory.rows):
        row = dict(original)
        row["t"] = float(new_time[index])
        for joint_index, joint in enumerate(JOINTS):
            row[f"{joint}_q"] = float(old_q[index, joint_index])
            row[f"{joint}_dq"] = float(new_dq[index, joint_index])
            row[f"{joint}_ddq"] = float(new_ddq[index, joint_index])
            if f"{joint}_jerk" in mutable_fields:
                row[f"{joint}_jerk"] = float(diagnostic_jerk[index, joint_index])
        rows.append(row)
    result = NativeTrajectory(trajectory.fields, tuple(rows))
    if np.any(np.diff(result.time) <= 0.0) or not all(np.isfinite(array).all() for array in (result.time, result.q, result.dq, result.ddq)):
        raise ValueError(f"invalid_local_retime:{source}")
    write_native(destination, result)
    return {
        "status": "PASS",
        "source": str(source.resolve()),
        "destination": str(destination.resolve()),
        "state_count": len(result.rows),
        "source_duration_s": float(old_time[-1]),
        "retimed_duration_s": float(result.time[-1]),
        "duration_added_s": float(result.time[-1] - old_time[-1]),
        "hotspot_centers_s": centers,
        "gain": float(gain),
        "half_width_s": float(half_width),
        "maximum_local_time_scale": float(np.max(speed_map)),
        "q_path_max_abs_delta_rad": float(np.max(np.abs(result.q - old_q))),
        "retiming_equations": {
            "dt_new_dt_old": "s(tau)=1+gain*max_smooth_bump(tau)",
            "qdot_new": "qdot_old/s",
            "qddot_new": "qddot_old/s^2-qdot_old*ds_dtau/s^3",
        },
        "native_post_ruckig_source_preserved": True,
        "direct_native_post_ruckig_for_transformed_state": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hotspots", type=Path, default=DEFAULT_HOTSPOTS)
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--half-width", type=float, default=0.5)
    parser.add_argument("--hotspot-count", type=int, default=3)
    parser.add_argument("--case-ids", default=",".join(CASES))
    args = parser.parse_args()
    if args.gain < 0.0:
        raise ValueError("gain_must_be_nonnegative")
    case_ids = tuple(value for value in args.case_ids.split(",") if value)
    records = []
    for case_id in case_ids:
        source = args.source_dir.resolve() / f"{case_id}.csv"
        destination = args.output_dir.resolve() / f"{case_id}.csv"
        centers = load_hotspots(args.hotspots.resolve(), case_id, limit=max(1, args.hotspot_count))
        family = next(name for name, members in FAMILIES.items() if case_id in members)
        records.append({"case_id": case_id, "family": family, "evidence": retime_trajectory(source, destination, centers, args.gain, args.half_width)})
    summary = {
        "schema_version": "d52-local-analytic-time-dilation-v1",
        "status": "PASS" if len(records) == len(case_ids) else "BLOCKED",
        "method": "smooth_local_time_dilation_with_chain_rule_state_transform",
        "collision_method": "adaptive_discrete_interpolation",
        "candidate_direct_native_post_ruckig": False,
        "source_native_post_ruckig_preserved": True,
        "case_count": len(records),
        "cases": records,
    }
    args.output_dir.resolve().mkdir(parents=True, exist_ok=True)
    (args.output_dir.resolve() / "local_retime_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "case_count": len(records), "output_dir": str(args.output_dir.resolve())}, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
