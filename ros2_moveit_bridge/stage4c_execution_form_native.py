#!/usr/bin/env python3
"""Native MoveIt2 execution-form worker for the D48 shadow campaign.

The worker takes a geometric joint path, runs the same MoveIt2 trajectory
chain used by the project (unwind, TOTG, then native Ruckig smoothing), and
persists the actual returned time/position/velocity/acceleration samples.
It never writes to the protected D47 champion or any Stage-3 release path.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

import rclpy
from moveit.core.robot_state import RobotState
from moveit.core.robot_trajectory import RobotTrajectory
from moveit.planning import MoveItPy

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage3_h7_native import GROUP_NAME, JOINT_NAMES, make_trajectory_message, runtime_limit_audit
from stage3_h7_2_native import message_arrays


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError("empty_execution_form_manifest")
    required = {"case_id", "trajectory_csv", "family"}
    if not required.issubset(rows[0]):
        raise RuntimeError(f"execution_form_manifest_columns_missing:{sorted(required - set(rows[0]))}")
    return [{key: str(value) for key, value in row.items()} for row in rows]


def read_q(path: Path) -> np.ndarray:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < 2:
        raise RuntimeError(f"trajectory_too_short:{path}")
    values: list[list[float]] = []
    for row in rows[1:]:
        if not row:
            continue
        try:
            values.append([float(row[f"j{i}_q"]) for i in range(1, 7)])
        except ValueError as exc:
            raise RuntimeError(f"invalid_joint_row:{path}:{list(row.items())[:2]}") from exc
        except KeyError as exc:
            raise RuntimeError(f"missing_joint_column:{path}:{exc}") from exc
    q = np.asarray(values, dtype=float)
    if q.ndim != 2 or q.shape[1] != 6 or len(q) < 2 or not np.isfinite(q).all():
        raise RuntimeError(f"invalid_joint_path:{path}:{q.shape}")
    return q


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_trajectory(path: Path, t: np.ndarray, q: np.ndarray, dq: np.ndarray, ddq: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not all(array.shape == q.shape for array in (dq, ddq)):
        raise RuntimeError("native_output_array_shape_mismatch")
    if not all(np.isfinite(array).all() for array in (t, q, dq, ddq)) or np.any(np.diff(t) <= 0.0):
        raise RuntimeError("native_output_nonfinite_or_nonmonotonic")
    # Ruckig's native output exposes q, qdot and qddot.  Jerk is deliberately
    # derived only for audit display and is labelled as such by the manifest.
    jerk = np.gradient(ddq, t, axis=0, edge_order=1)
    with path.open("w", encoding="utf-8", newline="") as stream:
        fields = ["t"] + [f"j{i}_q" for i in range(1, 7)] + [f"j{i}_dq" for i in range(1, 7)] + [f"j{i}_ddq" for i in range(1, 7)] + [f"j{i}_jerk" for i in range(1, 7)]
        writer = csv.writer(stream)
        writer.writerow(fields)
        for index in range(len(t)):
            writer.writerow([float(t[index]), *q[index], *dq[index], *ddq[index], *jerk[index]])


def write_ruckig_oracle_input(path: Path, t: np.ndarray, q: np.ndarray, dq: np.ndarray, ddq: np.ndarray) -> None:
    """Persist the exact pre-Ruckig knot state for an isolated D50 oracle.

    MoveIt's Python wrapper does not expose the internal Ruckig profiles.  The
    D50 shadow uses this optional dump to replay the exact TOTG knot boundary
    states through a standalone native Ruckig profile inspector.  This path is
    never used by the normal D48/D49 worker outputs.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        fields = ["primitive", "local", "time"] + [f"j{i}_q" for i in range(1, 7)] + [f"j{i}_dq" for i in range(1, 7)] + [f"j{i}_ddq" for i in range(1, 7)]
        writer = csv.writer(stream)
        writer.writerow(fields)
        for index in range(len(t)):
            writer.writerow([0, index, float(t[index]), *q[index], *dq[index], *ddq[index]])


def process_case(moveit: MoveItPy, item: dict[str, str], out_root: Path, velocity_scaling: float, acceleration_scaling: float, overshoot_threshold: float, dump_totg_dir: Path | None = None) -> dict[str, Any]:
    case_id = item["case_id"]
    source = Path(item["trajectory_csv"]).resolve()
    record: dict[str, Any] = {
        "case_id": case_id,
        "family": item["family"],
        "source_trajectory": str(source),
        "status": "BLOCKED",
        "post_processing": "MoveIt2 RobotTrajectory.unwind -> apply_totg_time_parameterization -> apply_ruckig_smoothing",
        "native_post_ruckig": False,
        "jerk_method": "DERIVED_FINITE_DIFFERENCE_FROM_NATIVE_POST_RUCKIG_ACCELERATION",
    }
    try:
        q_input = read_q(source)
        start_state = RobotState(moveit.get_robot_model())
        trajectory = RobotTrajectory(moveit.get_robot_model())
        trajectory.joint_model_group_name = GROUP_NAME
        trajectory.set_robot_trajectory_msg(
            start_state,
            make_trajectory_message([{"joint_values": row.tolist()} for row in q_input]),
        )
        trajectory.unwind()
        q_unwound = np.asarray([list(point.positions) for point in trajectory.get_robot_trajectory_msg().joint_trajectory.points], dtype=float)
        if q_unwound.shape != q_input.shape or not np.isfinite(q_unwound).all():
            raise RuntimeError("native_unwind_output_invalid")
        totg_ok = bool(trajectory.apply_totg_time_parameterization(
            velocity_scaling,
            acceleration_scaling,
            path_tolerance=0.01,
            resample_dt=0.01,
            min_angle_change=0.0005,
        ))
        record["totg_returned_success"] = totg_ok
        if not totg_ok:
            raise RuntimeError("native_totg_returned_false")
        totg_t, totg_q, totg_dq, totg_ddq = message_arrays(trajectory.get_robot_trajectory_msg())
        if dump_totg_dir is not None:
            oracle_path = dump_totg_dir / f"{case_id}.csv"
            write_ruckig_oracle_input(oracle_path, totg_t, totg_q, totg_dq, totg_ddq)
            record["totg_oracle_input"] = str(oracle_path.resolve())
        ruckig_ok = bool(trajectory.apply_ruckig_smoothing(
            velocity_scaling,
            acceleration_scaling,
            mitigate_overshoot=True,
            overshoot_threshold=overshoot_threshold,
        ))
        record["ruckig_returned_success"] = ruckig_ok
        if not ruckig_ok:
            raise RuntimeError("native_ruckig_returned_false")
        t, q, dq, ddq = message_arrays(trajectory.get_robot_trajectory_msg())
        output_path = out_root / "trajectories" / f"{case_id}.csv"
        write_trajectory(output_path, t, q, dq, ddq)
        record.update({
            "status": "PASS",
            "native_post_ruckig": True,
            "input_state_count": int(len(q_input)),
            "totg_state_count": int(len(totg_t)),
            "post_ruckig_state_count": int(len(t)),
            "duration_s": float(t[-1]),
            "endpoint_start_error_rad": float(np.max(np.abs(q[0] - q_input[0]))),
            "endpoint_end_error_rad": float(np.max(np.abs(q[-1] - q_input[-1]))),
            "max_abs_velocity_rad_s": float(np.max(np.abs(dq))),
            "max_abs_acceleration_rad_s2": float(np.max(np.abs(ddq))),
            "trajectory_csv": str(output_path.resolve()),
        })
    except Exception as exc:
        record.update({
            "error": f"{type(exc).__name__}:{exc}",
            "traceback_tail": traceback.format_exc().splitlines()[-8:],
        })
    return record


def run(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    items = read_manifest(Path(args.manifest).resolve())
    rclpy.init()
    moveit: MoveItPy | None = None
    try:
        moveit = MoveItPy(node_name="stage4c_execution_form_native")
        limit_audit = runtime_limit_audit(moveit, output, GROUP_NAME, "spray_tcp_link", Path(args.joint_limits).resolve())
        if limit_audit.get("status") != "PASSED":
            dump_json(output / "runtime_limit_audit.json", limit_audit)
            dump_json(output / "execution_form_summary.json", {"status": "BLOCKED", "first_blocker": "runtime_joint_limit_audit_failed", "case_count": len(items), "cases": []})
            return 2
        dump_totg_dir = Path(args.dump_totg_dir).resolve() if args.dump_totg_dir else None
        records = [process_case(moveit, item, output, float(args.velocity_scaling), float(args.acceleration_scaling), float(args.overshoot_threshold), dump_totg_dir) for item in items]
        dump_json(output / "runtime_limit_audit.json", limit_audit)
        dump_json(output / "execution_form_summary.json", {
            "schema_version": "d48-c1-native-execution-form-v1",
            "status": "PASS" if all(item.get("status") == "PASS" for item in records) else "BLOCKED",
            "case_count": len(records),
            "native_post_ruckig_case_count": sum(item.get("native_post_ruckig") is True for item in records),
            "failed_case_ids": [item["case_id"] for item in records if item.get("status") != "PASS"],
            "velocity_scaling": float(args.velocity_scaling),
            "acceleration_scaling": float(args.acceleration_scaling),
            "overshoot_threshold": float(args.overshoot_threshold),
            "cases": records,
        })
        return 0 if all(item.get("status") == "PASS" for item in records) else 2
    finally:
        if moveit is not None:
            moveit.shutdown()
        rclpy.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--joint-limits", required=True)
    parser.add_argument("--velocity-scaling", type=float, default=0.15)
    parser.add_argument("--acceleration-scaling", type=float, default=0.15)
    parser.add_argument("--overshoot-threshold", type=float, default=0.005)
    parser.add_argument("--dump-totg-dir", default=None)
    # ros2 launch appends --ros-args and parameter-file arguments to every
    # node.  They are consumed by rclpy/MoveIt and are intentionally outside
    # this worker's application-level argument contract.
    args, _ = parser.parse_known_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
