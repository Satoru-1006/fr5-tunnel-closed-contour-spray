"""Validate Stage5AC waypoint FK against explicit local task-slack semantics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def quat_error(observed: list[float], expected: list[float]) -> float:
    a = np.asarray(observed, dtype=float)
    b = np.asarray(expected, dtype=float)
    a /= max(float(np.linalg.norm(a)), 1e-15)
    b /= max(float(np.linalg.norm(b)), 1e-15)
    return float(2.0 * math.acos(float(np.clip(abs(float(np.dot(a, b))), -1.0, 1.0))))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("poses_csv", type=Path)
    parser.add_argument("fk_trace", type=Path)
    parser.add_argument("selected_path_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--offset-y", type=float, default=-0.0025)
    parser.add_argument("--start", type=int, default=87)
    parser.add_argument("--end", type=int, default=94)
    args = parser.parse_args()
    poses = list(csv.DictReader(args.poses_csv.open(encoding="utf-8", newline="")))
    traces = [json.loads(line) for line in args.fk_trace.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = list(csv.DictReader(args.selected_path_csv.open(encoding="utf-8", newline="")))
    if len(poses) != 181 or len(selected) != 181 or len(traces) < 181:
        raise RuntimeError(f"stage5ac_task_validator_shape:{len(poses)}/{len(selected)}/{len(traces)}")
    trace_q = np.asarray([row["q"] for row in traces], dtype=float)
    selected_q = np.asarray([[float(row[f"j{i}_q"]) for i in range(1, 7)] for row in selected], dtype=float)
    rows = []
    cursor = 0
    for waypoint, pose in enumerate(poses):
        target_q = selected_q[waypoint]
        distances = np.linalg.norm(trace_q[cursor:] - target_q[None, :], axis=1)
        state_index = cursor + int(np.argmin(distances))
        cursor = state_index
        observed = traces[state_index]
        effective_target = np.asarray([float(pose["x"]), float(pose["y"]) + (args.offset_y if args.start <= waypoint <= args.end else 0.0), float(pose["z"])], dtype=float)
        actual = np.asarray(observed["translation_m"], dtype=float)
        expected_quat = [float(pose[k]) for k in ("qx", "qy", "qz", "qw")]
        position_error = float(np.linalg.norm(actual - effective_target))
        orientation_error = quat_error(observed["quaternion_xyzw"], expected_quat)
        rows.append({"waypoint": waypoint, "candidate_state_index": int(state_index), "candidate_time_s": float(observed["time_s"]), "q_match_distance_rad": float(np.linalg.norm(trace_q[state_index] - target_q)), "target_effective_position_m": effective_target.tolist(), "actual_fk_position_m": actual.tolist(), "position_error_m": position_error, "orientation_error_rad": orientation_error, "local_task_slack_applied": bool(args.start <= waypoint <= args.end), "original_target_y_m": float(pose["y"]), "effective_target_y_m": float(effective_target[1])})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "STAGE5AC_TASK_VALIDATION.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    position_tol = 1.0e-4
    orientation_tol = 1.0e-3
    summary = {"schema_version": "stage5ac-task-validation-v1", "waypoint_count": len(rows), "task_slack": {"offset_y_m": args.offset_y, "waypoint_window": [args.start, args.end], "semantic": "explicit_local_task_slack_not_original_pose_exactness"}, "max_position_error_m": max(r["position_error_m"] for r in rows), "max_orientation_error_rad": max(r["orientation_error_rad"] for r in rows), "position_tolerance_m": position_tol, "orientation_tolerance_rad": orientation_tol, "position_pass": all(r["position_error_m"] <= position_tol for r in rows), "orientation_pass": all(r["orientation_error_rad"] <= orientation_tol for r in rows), "original_pose_exact_pass": all(abs(r["actual_fk_position_m"][1] - r["original_target_y_m"]) <= position_tol for r in rows), "mapped_state_count": len({r["candidate_state_index"] for r in rows}), "evidence_csv": str((args.output_dir / "STAGE5AC_TASK_VALIDATION.csv").resolve()), "promotion": "NO_PROMOTION"}
    (args.output_dir / "STAGE5AC_TASK_VALIDATION.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["position_pass"] and summary["orientation_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
