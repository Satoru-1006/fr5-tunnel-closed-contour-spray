"""Generate a D52 native candidate from stateful locally retimed samples."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
import traceback
from pathlib import Path

import numpy as np
from moveit.core.robot_state import RobotState
from moveit.core.robot_trajectory import RobotTrajectory
from moveit.planning import MoveItPy
from moveit_msgs.msg import RobotTrajectory as RobotTrajectoryMsg
from trajectory_msgs.msg import JointTrajectoryPoint

ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "ros2_moveit_bridge"
if str(BRIDGE) not in sys.path:
    sys.path.insert(0, str(BRIDGE))

from stage3_h7_native import GROUP_NAME, JOINT_NAMES, duration_message  # noqa: E402
from stage3_h7_2_native import message_arrays  # noqa: E402

CASES = (
    "regression_0000", "regression_0001", "normal_0000", "normal_0100",
    "boundary_0000", "boundary_0100", "collision_sensitive_0000", "collision_sensitive_0051",
    "adversarial_0100", "adversarial_0101", "perturbation_0000", "perturbation_0100",
)
FAMILIES = {
    "REGRESSION": {"regression_0000", "regression_0001"},
    "NORMAL": {"normal_0000", "normal_0100"},
    "BOUNDARY": {"boundary_0000", "boundary_0100"},
    "COLLISION_SENSITIVE": {"collision_sensitive_0000", "collision_sensitive_0051"},
    "ADVERSARIAL": {"adversarial_0100", "adversarial_0101"},
    "PERTURBATION": {"perturbation_0000", "perturbation_0100"},
}

BASE_VELOCITY_LIMITS = np.asarray([3.15, 3.15, 3.15, 3.2, 3.2, 3.2], dtype=float)
BASE_ACCELERATION_LIMITS = np.full(6, 0.7, dtype=float)
JERK_LIMITS = np.full(6, 8.0, dtype=float)


def family(case_id: str) -> str:
    return next(name for name, members in FAMILIES.items() if case_id in members)


def read_rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < 2:
        raise RuntimeError(f"trajectory_too_short:{path}")
    return [
        (
            float(row["t"]),
            np.asarray([float(row[f"j{i}_q"]) for i in range(1, 7)], dtype=float),
            np.asarray([float(row[f"j{i}_dq"]) for i in range(1, 7)], dtype=float),
            np.asarray([float(row[f"j{i}_ddq"]) for i in range(1, 7)], dtype=float),
        )
        for row in rows
    ]


def stateful_message(rows):
    message = RobotTrajectoryMsg()
    message.joint_trajectory.joint_names = list(JOINT_NAMES)
    for time, q, dq, ddq in rows:
        point = JointTrajectoryPoint()
        point.positions = q.tolist()
        point.velocities = dq.tolist()
        point.accelerations = ddq.tolist()
        point.time_from_start = duration_message(time)
        message.joint_trajectory.points.append(point)
    return message


def project_jerk_reachable_boundaries(rows, velocity_scaling: float, acceleration_scaling: float):
    """Project only impossible Ruckig boundary derivatives, with evidence.

    Ruckig rejects a target state when its velocity/acceleration pair would
    cross a velocity limit before the acceleration can reach zero under the
    declared jerk limit.  MoveIt's duration extension cannot repair that
    state because the endpoint derivatives stay unchanged.  This explicit,
    bounded projection keeps q and timestamps intact and changes only the
    smallest derivative component needed to make each endpoint reachable.
    """
    max_velocity = BASE_VELOCITY_LIMITS * float(velocity_scaling)
    max_acceleration = BASE_ACCELERATION_LIMITS * float(acceleration_scaling)
    projected = []
    changed = 0
    max_delta_v = 0.0
    max_delta_a = 0.0
    for time, q, velocity, acceleration in rows:
        v = np.clip(np.asarray(velocity, dtype=float), -max_velocity, max_velocity)
        a = np.clip(np.asarray(acceleration, dtype=float), -max_acceleration, max_acceleration)
        original_v = v.copy()
        original_a = a.copy()
        for joint in range(6):
            if a[joint] > 0.0 and v[joint] - (a[joint] * a[joint]) / (2.0 * JERK_LIMITS[joint]) < -max_velocity[joint]:
                reachable = math.sqrt(max(0.0, 2.0 * JERK_LIMITS[joint] * (v[joint] + max_velocity[joint])))
                a[joint] = min(a[joint], reachable)
            elif a[joint] < 0.0 and v[joint] + (a[joint] * a[joint]) / (2.0 * JERK_LIMITS[joint]) > max_velocity[joint]:
                reachable = math.sqrt(max(0.0, 2.0 * JERK_LIMITS[joint] * (max_velocity[joint] - v[joint])))
                a[joint] = max(a[joint], -reachable)
        delta_v = float(np.max(np.abs(v - original_v)))
        delta_a = float(np.max(np.abs(a - original_a)))
        changed += int(delta_v > 0.0 or delta_a > 0.0)
        max_delta_v = max(max_delta_v, delta_v)
        max_delta_a = max(max_delta_a, delta_a)
        projected.append((time, np.asarray(q, dtype=float), v, a))
    return projected, {
        "enabled": True,
        "projected_state_count": changed,
        "max_abs_velocity_delta_rad_s": max_delta_v,
        "max_abs_acceleration_delta_rad_s2": max_delta_a,
        "limits": {
            "velocity_rad_s": max_velocity.tolist(),
            "acceleration_rad_s2": max_acceleration.tolist(),
            "jerk_rad_s3": JERK_LIMITS.tolist(),
        },
    }


def repair_consistent_time_parameterization(rows, velocity_scaling: float, acceleration_scaling: float, time_scale_multiplier: float = 1.0):
    """Rebuild derivatives from the frozen q(t) path, then dilate time uniformly.

    The former D52 input contract preserved q and timestamps but carried
    derivatives from a different time parameterization.  That made the
    serialized state disagree with the geometric path and with the active
    MoveIt limits.  This repair leaves q exactly unchanged, derives dq/ddq
    from q(t), and applies one conservative time scale to v/a/jerk together.
    It is intentionally an explicit shadow transformation before native
    MoveIt2/Ruckig; no protected Stage 3 artifact is modified.
    """
    time = np.asarray([item[0] for item in rows], dtype=float)
    q = np.asarray([item[1] for item in rows], dtype=float)
    if len(time) < 3 or np.any(np.diff(time) <= 0.0):
        raise RuntimeError("invalid_time_grid_for_consistent_retime")
    dq_old = np.gradient(q, time, axis=0, edge_order=1)
    ddq_old = np.gradient(dq_old, time, axis=0, edge_order=1)
    jerk_old = np.gradient(ddq_old, time, axis=0, edge_order=1)
    max_velocity = BASE_VELOCITY_LIMITS * float(velocity_scaling)
    max_acceleration = BASE_ACCELERATION_LIMITS * float(acceleration_scaling)
    velocity_ratio = float(np.max(np.abs(dq_old) / max_velocity))
    acceleration_ratio = float(np.max(np.abs(ddq_old) / max_acceleration))
    jerk_ratio = float(np.max(np.abs(jerk_old) / JERK_LIMITS))
    safety_margin = 1.05
    scale = max(1.0, safety_margin * velocity_ratio,
                safety_margin * math.sqrt(max(0.0, acceleration_ratio)),
                safety_margin * (max(0.0, jerk_ratio) ** (1.0 / 3.0)))
    scale *= float(time_scale_multiplier)
    new_time = time * scale
    new_dq = dq_old / scale
    new_ddq = ddq_old / (scale * scale)
    repaired = [(float(new_time[index]), q[index].copy(), new_dq[index].copy(), new_ddq[index].copy())
                for index in range(len(rows))]
    return repaired, {
        "enabled": True,
        "method": "q_path_finite_difference_plus_uniform_time_dilation",
        "q_path_max_abs_delta_rad": 0.0,
        "source_duration_s": float(time[-1]),
        "retimed_duration_s": float(new_time[-1]),
        "time_scale": float(scale),
        "time_scale_multiplier": float(time_scale_multiplier),
        "safety_margin": safety_margin,
        "source_path_velocity_ratio": velocity_ratio,
        "source_path_acceleration_ratio": acceleration_ratio,
        "source_path_jerk_ratio": jerk_ratio,
        "active_limits": {
            "velocity_rad_s": max_velocity.tolist(),
            "acceleration_rad_s2": max_acceleration.tolist(),
            "jerk_rad_s3": JERK_LIMITS.tolist(),
        },
        "derived_max_velocity_rad_s": float(np.max(np.abs(new_dq))),
        "derived_max_acceleration_rad_s2": float(np.max(np.abs(new_ddq))),
        "derived_max_jerk_rad_s3": float(np.max(np.abs(jerk_old / (scale ** 3)))),
    }


def write_native(path: Path, arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> None:
    time, q, dq, ddq = arrays
    jerk = np.gradient(ddq, time, axis=0, edge_order=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["t"] + [f"j{i}_q" for i in range(1, 7)] + [f"j{i}_dq" for i in range(1, 7)] + [f"j{i}_ddq" for i in range(1, 7)] + [f"j{i}_jerk" for i in range(1, 7)]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(fields)
        for index in range(len(time)):
            writer.writerow([float(time[index]), *q[index], *dq[index], *ddq[index], *jerk[index]])


def process_case(moveit: MoveItPy, source: Path, destination: Path, velocity_scaling: float, acceleration_scaling: float, repair_jerk_boundaries: bool = False, repair_consistent_time: bool = False, time_scale_multiplier: float = 1.0) -> dict:
    case_id = source.stem
    record = {
        "case_id": case_id,
        "family": family(case_id),
        "source_trajectory": str(source.resolve()),
        "status": "BLOCKED",
        "native_post_ruckig": False,
        "post_processing": "MoveIt2 RobotTrajectory.set_robot_trajectory_msg(stateful q/dq/ddq/t) -> apply_ruckig_smoothing",
        "jerk_method": "DERIVED_FINITE_DIFFERENCE_FROM_NATIVE_POST_RUCKIG_ACCELERATION",
    }
    try:
        rows = read_rows(source)
        repair_evidence = {"enabled": False, "projected_state_count": 0}
        time_repair_evidence = {"enabled": False}
        if repair_consistent_time:
            rows, time_repair_evidence = repair_consistent_time_parameterization(rows, velocity_scaling, acceleration_scaling, time_scale_multiplier)
        if repair_jerk_boundaries:
            rows, repair_evidence = project_jerk_reachable_boundaries(rows, velocity_scaling, acceleration_scaling)
        trajectory = RobotTrajectory(moveit.get_robot_model())
        trajectory.joint_model_group_name = GROUP_NAME
        start = RobotState(moveit.get_robot_model())
        trajectory.set_robot_trajectory_msg(start, stateful_message(rows))
        before = message_arrays(trajectory.get_robot_trajectory_msg())
        ruckig_ok = bool(trajectory.apply_ruckig_smoothing(
            float(velocity_scaling),
            float(acceleration_scaling),
            mitigate_overshoot=True,
            overshoot_threshold=0.005,
        ))
        after = message_arrays(trajectory.get_robot_trajectory_msg())
        if not ruckig_ok:
            raise RuntimeError("stateful_ruckig_returned_false")
        if not all(np.isfinite(array).all() for array in after) or np.any(np.diff(after[0]) <= 0.0):
            raise RuntimeError("stateful_ruckig_output_invalid")
        derived_jerk = np.gradient(after[3], after[0], axis=0, edge_order=1)
        if not np.isfinite(derived_jerk).all():
            raise RuntimeError("stateful_ruckig_derived_jerk_invalid")
        max_abs_jerk = float(np.max(np.abs(derived_jerk)))
        if max_abs_jerk > float(np.max(JERK_LIMITS)) + 1.0e-9:
            raise RuntimeError(f"stateful_ruckig_derived_jerk_limit_exceeded:{max_abs_jerk}")
        write_native(destination, after)
        record.update({
            "status": "PASS",
            "native_post_ruckig": True,
            "ruckig_returned_success": True,
            "input_state_count": len(before[0]),
            "post_ruckig_state_count": len(after[0]),
            "source_duration_s": float(before[0][-1]),
            "duration_s": float(after[0][-1]),
            "duration_delta_s": float(after[0][-1] - before[0][-1]),
            "max_abs_velocity_rad_s": float(np.max(np.abs(after[2]))),
            "max_abs_acceleration_rad_s2": float(np.max(np.abs(after[3]))),
            "max_abs_jerk_rad_s3": max_abs_jerk,
            "jerk_limit_rad_s3": float(np.max(JERK_LIMITS)),
            "jerk_check": {
                "status": "PASS",
                "method": "independent_finite_difference_of_native_post_ruckig_acceleration",
            },
            "trajectory_csv": str(destination.resolve()),
            "boundary_repair": repair_evidence,
            "consistent_time_repair": time_repair_evidence,
        })
    except Exception as exc:
        record.update({"error": f"{type(exc).__name__}:{exc}", "traceback_tail": traceback.format_exc().splitlines()[-8:]})
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--velocity-scaling", type=float, default=0.075)
    parser.add_argument("--acceleration-scaling", type=float, default=0.075)
    parser.add_argument("--repair-jerk-boundaries", action="store_true")
    parser.add_argument("--repair-consistent-time", action="store_true")
    parser.add_argument("--time-scale-multiplier", type=float, default=1.0)
    args, _ = parser.parse_known_args()
    if args.time_scale_multiplier <= 0.0:
        raise ValueError("time_scale_multiplier_must_be_positive")
    args.output_dir.resolve().mkdir(parents=True, exist_ok=True)
    moveit = MoveItPy(node_name="d52_stateful_ruckig_candidate")
    try:
        records = [process_case(moveit, args.input_dir.resolve() / f"{case_id}.csv", args.output_dir.resolve() / "trajectories" / f"{case_id}.csv", args.velocity_scaling, args.acceleration_scaling, args.repair_jerk_boundaries, args.repair_consistent_time, args.time_scale_multiplier) for case_id in CASES]
        summary = {
            "schema_version": "d52-stateful-native-ruckig-candidate-v1",
            "status": "PASS" if all(item["status"] == "PASS" for item in records) else "BLOCKED",
            "case_count": len(records),
            "native_post_ruckig_case_count": sum(item["native_post_ruckig"] for item in records),
            "velocity_scaling": float(args.velocity_scaling),
            "acceleration_scaling": float(args.acceleration_scaling),
            "repair_jerk_boundaries": bool(args.repair_jerk_boundaries),
            "repair_consistent_time": bool(args.repair_consistent_time),
            "time_scale_multiplier": float(args.time_scale_multiplier),
            "method": "MoveIt2 stateful RobotTrajectory q/dq/ddq/t local retime followed by native apply_ruckig_smoothing",
            "cases": records,
        }
        (args.output_dir.resolve() / "execution_form_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"status": summary["status"], "case_count": len(records), "native_post_ruckig_case_count": summary["native_post_ruckig_case_count"]}, sort_keys=True))
        return 0 if summary["status"] == "PASS" else 2
    finally:
        # MoveItPy owns the rclcpp context in this process.  Do not call
        # rclpy.init()/rclpy.shutdown() around it: initializing a second
        # Python-side owner leaves the C++ custom deleter with an invalid
        # context during MoveItCpp/CallbackGroup teardown.
        moveit = None
        gc.collect()


if __name__ == "__main__":
    raise SystemExit(main())
