#!/usr/bin/env python3
"""Replay the frozen DP path through the real MoveIt Ruckig adapter."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "ros2_moveit_bridge") not in sys.path:
    sys.path.insert(0, str(ROOT / "ros2_moveit_bridge"))

from src.frozen_replay_diagnostics import OUTPUT_ROOT, load_yaml, raw_hash, read_parquet, write_json, write_parquet


def read_samples(path: Path) -> list[dict[str, float | int | None]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = []
        for row in csv.DictReader(handle):
            item: dict[str, float | int | None] = {"sample_index": int(row["sample_index"]), "t": float(row["t"])}
            for joint in range(1, 7):
                for prefix in ("q", "v", "a"):
                    value = row.get(f"{prefix}_j{joint}", "")
                    item[f"{prefix}_j{joint}"] = None if value in ("", None) else float(value)
            rows.append(item)
    return rows


def run_ros(config_path: Path) -> dict[str, object]:  # pragma: no cover - ROS2 only
    import rclpy
    from moveit.core.robot_state import RobotState
    from moveit.planning import MoveItPy
    import plan_closed_contour_moveit as bridge
    from tools.run_stage17_pose_repair import _write_ruckig_joint_samples

    raw = load_yaml(config_path)
    path_rows = read_parquet(OUTPUT_ROOT / "frozen_selected_joint_path.parquet")
    q_path = np.asarray([row["q_unwrapped_rad"] for row in sorted(path_rows, key=lambda row: int(row["waypoint_id"]))], dtype=float)
    if q_path.shape != (181, 6):
        raise RuntimeError(f"frozen selected path must be 181x6, got {q_path.shape}")
    input_payload = {"q_path": q_path.tolist(), "start_velocity": [0.0] * 6, "start_acceleration": [0.0] * 6, "velocity_limits": [1.0] * 6, "acceleration_limits": [2.0] * 6, "jerk_limits": [5.0] * 6, "control_period_s": 0.01, "smoothing": "moveit_ruckig_filter"}
    input_hash = raw_hash(input_payload)
    rclpy.init()
    moveit = MoveItPy(node_name="stage191_ruckig_replay")
    group, ee_link = str(raw["robot"]["group_name"]), str(raw["robot"]["ee_link"])
    records = []
    first_samples: list[dict[str, object]] = []
    for iteration in range(int(raw["ruckig_replay"]["repeat_count"])):
        temp_csv = OUTPUT_ROOT / "ruckig_replay" / f"trajectory_{iteration:03d}.csv"
        try:
            message = bridge.build_seed_joint_trajectory(q_path, [f"j{i}" for i in range(1, 7)])
            start_state = RobotState(moveit.get_robot_model())
            trajectory = bridge.build_and_smooth_moveit_trajectory(moveit, group, ee_link, start_state, message, 0.15, 0.15, True, time_parameterization="tcp_arclength", target_tcp_speed=0.003, zero_boundary_state=True)
            count = _write_ruckig_joint_samples(temp_csv, trajectory)
            samples = read_samples(temp_csv)
            if not first_samples:
                first_samples = samples
            records.append({"replay_id": f"ruckig:{iteration:03d}", "iteration": iteration, "ruckig_input_semantic_hash": input_hash, "ruckig_status": "pass", "sample_count": count, "ruckig_trajectory_semantic_hash": raw_hash(samples), "first_invalid_segment": None, "validation_failure_reason": None})
        except Exception as exc:
            records.append({"replay_id": f"ruckig:{iteration:03d}", "iteration": iteration, "ruckig_input_semantic_hash": input_hash, "ruckig_status": "fail", "sample_count": 0, "ruckig_trajectory_semantic_hash": None, "first_invalid_segment": 0, "input_q": q_path[0].tolist(), "target_q": q_path[-1].tolist(), "velocity_limits": [1.0] * 6, "acceleration_limits": [2.0] * 6, "jerk_limits": [5.0] * 6, "validation_failure_reason": f"{type(exc).__name__}: {exc}"})
    write_parquet(OUTPUT_ROOT / "ruckig_replay" / "replay_records.parquet", records)
    write_parquet(OUTPUT_ROOT / "frozen_ruckig_samples.parquet", first_samples)
    comparison = {"repeat_count": len(records), "success_count": sum(row["ruckig_status"] == "pass" for row in records), "input_hashes": sorted(set(row["ruckig_input_semantic_hash"] for row in records)), "trajectory_hashes": sorted(set(row["ruckig_trajectory_semantic_hash"] for row in records if row["ruckig_trajectory_semantic_hash"])), "all_results_identical": len({row["ruckig_trajectory_semantic_hash"] for row in records}) == 1 if records else False, "failures": [row for row in records if row["ruckig_status"] != "pass"]}
    write_json(OUTPUT_ROOT / "ruckig_replay" / "comparison.json", comparison)
    rclpy.shutdown()
    return comparison


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ros-node", action="store_true")
    parser.add_argument("--config", type=Path, default=ROOT / "config/stage191_determinism_diagnostics.yaml")
    args, _ = parser.parse_known_args()
    if not args.ros_node:
        return 2
    print(run_ros(args.config.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
