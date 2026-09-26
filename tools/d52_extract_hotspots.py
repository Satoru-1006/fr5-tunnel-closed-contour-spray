"""Extract per-case torque-slew hotspots from a D52 Pinocchio state audit."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top", type=int, default=3)
    args = parser.parse_args()
    grouped = defaultdict(list)
    with args.states.resolve().open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                grouped[str(row["case_id"])].append(row)
    records = []
    for case_id in sorted(grouped):
        rows = grouped[case_id]
        time = np.asarray([row["t_s"] for row in rows], dtype=float)
        tau = np.asarray([row["tau_Nm"] for row in rows], dtype=float)
        gravity = np.asarray([row["gravity_tau_Nm"] for row in rows], dtype=float)
        dynamic = tau - gravity
        dt = np.diff(time)
        slew = np.diff(tau, axis=0) / dt[:, None]
        order = np.argsort(-np.max(np.abs(slew), axis=1))
        regions = []
        used = set()
        for interval_index in order:
            interval_index = int(interval_index)
            if any(abs(interval_index - value) <= 1 for value in used):
                continue
            joint_index = int(np.argmax(np.abs(slew[interval_index])))
            dynamic_slew = np.diff(dynamic, axis=0)[interval_index, joint_index] / dt[interval_index]
            gravity_slew = np.diff(gravity, axis=0)[interval_index, joint_index] / dt[interval_index]
            regions.append({
                "interval_index": interval_index,
                "joint": f"j{joint_index + 1}",
                "time_start_s": float(time[interval_index]),
                "time_end_s": float(time[interval_index + 1]),
                "dt_s": float(dt[interval_index]),
                "abs_torque_slew_Nm_s": float(abs(slew[interval_index, joint_index])),
                "abs_dynamic_residual_slew_Nm_s": float(abs(dynamic_slew)),
                "abs_gravity_slew_Nm_s": float(abs(gravity_slew)),
                "qddot_left_rad_s2": float(rows[interval_index]["qddot"][joint_index]),
                "qddot_right_rad_s2": float(rows[interval_index + 1]["qddot"][joint_index]),
                "qdot_left_rad_s": float(rows[interval_index]["qdot"][joint_index]),
                "qdot_right_rad_s": float(rows[interval_index + 1]["qdot"][joint_index]),
            })
            used.add(interval_index)
            if len(regions) >= max(1, args.top):
                break
        records.append({"case_id": case_id, "candidate": {"top_torque_slew_regions": regions}})
    result = {
        "schema_version": "d52-native-state-hotspots-v1",
        "status": "PASS" if records else "BLOCKED",
        "source": str(args.states.resolve()),
        "case_count": len(records),
        "cases": records,
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "case_count": len(records), "output": str(args.output.resolve())}, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
