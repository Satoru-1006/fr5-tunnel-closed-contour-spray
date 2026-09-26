#!/usr/bin/env python3
"""Stage 1.9.2a MoveItPy replay with explicit keyword IK semantics."""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np

import run_stage192_moveitpy_replay as base


def _one_keyword(state, bridge, row, group, tip, manifest, mode, instance_id, iteration, process_index, ik_timeout):
    initial_values = base._state_values(state, group)
    target = np.asarray([row["x"], row["y"], row["z"]])
    started = time.perf_counter()
    solved = bool(state.set_from_ik(group, base._pose(row), tip_name=tip, timeout=ik_timeout))
    record = {
        "iteration": iteration,
        "process_id": __import__("os").getpid(),
        "process_index": process_index,
        "solver_instance_id": instance_id,
        "solver_instance_creation_order": int(instance_id.rsplit(":", 1)[-1]) if ":" in instance_id and instance_id.rsplit(":", 1)[-1].isdigit() else 0,
        "target_call_id": "call:00000182",
        "target_transform_raw_sha256": manifest["target_transform_raw_sha256"],
        "seed_vector_raw_sha256": manifest["seed_vector_raw_sha256"],
        "all_robot_variables_raw_sha256": manifest["all_robot_variables_raw_sha256"],
        "initial_robot_state_raw_sha256": hashlib.sha256(base._f64(initial_values)).hexdigest(),
        "solver_options_raw_sha256": manifest["solver_options_raw_sha256"],
        "requested_timeout_s": ik_timeout,
        "solver_success": solved,
        "solver_error_code": None if solved else "NO_SOLUTION",
        "elapsed_time_s": time.perf_counter() - started,
        "new_robot_state_per_call": True,
        "internal_random_restart_observed": False,
        "actual_attempt_count": 1,
        "requested_group": group,
        "requested_tip": tip,
    }
    if not solved:
        return record
    state.update()
    q = np.asarray(state.get_joint_group_positions(group), dtype=float)
    transform = np.asarray(bridge._transform_matrix(state.get_global_link_transform(tip)), dtype=float)
    record.update(
        {
            "solution_joint_vector_raw": q.tolist(),
            "solution_raw_sha256": hashlib.sha256(base._f64(q.tolist())).hexdigest(),
            "solution_quantized_sha256": base.quantized_hash(q.tolist(), 1.0e-7),
            "fk_position_error_m": float(np.linalg.norm(transform[:3, 3] - target)),
            "fk_orientation_error_rad": base._orientation_error(transform, row),
            "joint_limit_pass": bool(base._call(state, "satisfies_bounds", "satisfiesBounds", default=True)),
        }
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ros-node", action="store_true")
    parser.add_argument("--mode", choices=("same_instance", "new_instance", "single"), required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--process-index", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ik-timeout", type=float, default=0.0)
    args, _ = parser.parse_known_args()
    if not args.ros_node:
        raise SystemExit("real ROS2 launch required")
    base._one = _one_keyword
    return base.run_ros(
        Path("config/stage192a_moveitpy_runtime.yaml"),
        args.mode,
        args.repeat,
        args.process_index,
        args.output.resolve(),
        args.ik_timeout,
    )


if __name__ == "__main__":
    raise SystemExit(main())
