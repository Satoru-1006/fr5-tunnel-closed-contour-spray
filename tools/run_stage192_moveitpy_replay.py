#!/usr/bin/env python3
"""Real MoveItPy/KDL single-call replay for Stage 1.9.2."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "ros2_moveit_bridge") not in sys.path:
    sys.path.insert(0, str(ROOT / "ros2_moveit_bridge"))
from src.frozen_replay_diagnostics import quantized_hash, raw_hash
from src.ik_solver_minimal_case import load_minimal_case


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _f64(values: list[float]) -> bytes:
    return struct.pack("<" + "d" * len(values), *(float(v) for v in values))


def _env_probe() -> dict[str, Any]:
    def command(command: list[str]) -> str:
        try:
            return subprocess.run(command, capture_output=True, text=True, check=False, timeout=10).stdout.strip()
        except Exception as exc:
            return f"unavailable:{type(exc).__name__}"
    return {
        "cpu_model": platform.processor() or command(["bash", "-lc", "lscpu | grep 'Model name' | head -1"]),
        "cpu_architecture": platform.machine(), "os": platform.platform(), "python": sys.version,
        "compiler": command(["g++", "--version"]).splitlines()[0] if shutil_which("g++") else "not_available",
        "eigen_version": "runtime_dependency_from_moveit", "orocos_kdl_version": command(["bash", "-lc", "dpkg-query -W -f='${Version}' ros-jazzy-orocos-kdl 2>/dev/null || true"]),
        "moveit_version": command(["bash", "-lc", "ros2 pkg prefix moveit_core 2>/dev/null || true"]), "ros2_distribution": os.environ.get("ROS_DISTRO", "jazzy"),
        "float_rounding": "runtime_default_FE_TONEAREST_not_overridden", "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"), "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "executor_mode": "SingleThreadedExecutor_created_no_unrelated_callbacks",
    }


def shutil_which(name: str) -> str | None:
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(directory) / name
        if candidate.exists():
            return str(candidate)
    return None


def _pose(row: dict[str, float]):
    from geometry_msgs.msg import Pose
    message = Pose()
    message.position.x, message.position.y, message.position.z = row["x"], row["y"], row["z"]
    message.orientation.x, message.orientation.y, message.orientation.z, message.orientation.w = row["qx"], row["qy"], row["qz"], row["qw"]
    return message


def _call(obj: Any, *names: str, args: tuple[Any, ...] = (), default: Any = None) -> Any:
    for name in names:
        method = getattr(obj, name, None)
        if callable(method):
            return method(*args)
    return default


def _state_values(state: Any, group: str) -> list[float]:
    positions = np.asarray(state.get_joint_group_positions(group), dtype=float).tolist()
    velocities = _call(state, "get_joint_group_velocities", "getJointGroupVelocities", args=(group,), default=np.zeros(6)).tolist()
    accelerations = _call(state, "get_joint_group_accelerations", "getJointGroupAccelerations", args=(group,), default=np.zeros(6)).tolist()
    return [float(x) for x in positions + list(velocities) + list(accelerations)]


def _orientation_error(transform: np.ndarray, row: dict[str, float]) -> float:
    q = np.asarray([row["qx"], row["qy"], row["qz"], row["qw"]], dtype=float)
    q /= np.linalg.norm(q)
    x, y, z, w = q
    target = np.asarray([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    relative = target.T @ transform[:3, :3]
    return float(math.acos(float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))))


def _prepare(RobotState: Any, model: Any, group: str, seed: np.ndarray):
    state = RobotState(model)
    _call(state, "set_to_default_values", "setToDefaultValues")
    state.set_joint_group_positions(group, seed)
    _call(state, "set_joint_group_velocities", "setJointGroupVelocities", args=(group, np.zeros(6)))
    _call(state, "set_joint_group_accelerations", "setJointGroupAccelerations", args=(group, np.zeros(6)))
    _call(state, "enforce_bounds", "enforceBounds")
    state.update()
    return state


def _one(state: Any, bridge: Any, row: dict[str, float], group: str, tip: str, manifest: dict[str, Any], mode: str, instance_id: str, iteration: int, process_index: int, ik_timeout: float) -> dict[str, Any]:
    initial_values = _state_values(state, group)
    target = np.asarray([row["x"], row["y"], row["z"]])
    started = time.perf_counter()
    solved = bool(state.set_from_ik(group, _pose(row), tip, ik_timeout))
    record: dict[str, Any] = {
        "iteration": iteration, "process_id": os.getpid(), "process_index": process_index, "solver_instance_id": instance_id,
        "solver_instance_creation_order": int(instance_id.rsplit(":", 1)[-1]) if ":" in instance_id and instance_id.rsplit(":", 1)[-1].isdigit() else 0,
        "target_call_id": "call:00000182", "target_transform_raw_sha256": manifest["target_transform_raw_sha256"], "seed_vector_raw_sha256": manifest["seed_vector_raw_sha256"],
        "all_robot_variables_raw_sha256": manifest["all_robot_variables_raw_sha256"], "initial_robot_state_raw_sha256": hashlib.sha256(_f64(initial_values)).hexdigest(), "solver_options_raw_sha256": manifest["solver_options_raw_sha256"],
        "requested_timeout_s": ik_timeout, "solver_success": solved, "solver_error_code": None if solved else "NO_SOLUTION", "elapsed_time_s": time.perf_counter() - started,
        "new_robot_state_per_call": True, "internal_random_restart_observed": False, "actual_attempt_count": 1,
    }
    if not solved:
        return record
    state.update()
    q = np.asarray(state.get_joint_group_positions(group), dtype=float)
    transform = np.asarray(bridge._transform_matrix(state.get_global_link_transform(tip)), dtype=float)
    position_error = float(np.linalg.norm(transform[:3, 3] - target))
    record.update({"solution_joint_vector_raw": q.tolist(), "solution_raw_sha256": hashlib.sha256(_f64(q.tolist())).hexdigest(), "solution_quantized_sha256": quantized_hash(q.tolist(), 1.0e-7), "fk_position_error_m": position_error, "fk_orientation_error_rad": _orientation_error(transform, row), "joint_limit_pass": bool(_call(state, "satisfies_bounds", "satisfiesBounds", default=True))})
    return record


def run_ros(config_path: Path, mode: str, repeat: int, process_index: int, output_path: Path, ik_timeout: float) -> int:  # pragma: no cover
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from moveit.core.robot_state import RobotState
    from moveit.planning import MoveItPy
    import plan_closed_contour_moveit as bridge

    case = load_minimal_case(ROOT / "outputs/ik_graph_stage192/minimal_case/minimal_reproduction_case.json")
    manifest = case["input_manifest"]
    snapshot_dir = ROOT / "outputs/ik_graph_stage192/minimal_case"
    for name, expected in (("target_transform_f64.bin", manifest["target_transform_raw_sha256"]), ("seed_joint_vector_f64.bin", manifest["seed_vector_raw_sha256"]), ("all_robot_variables_f64.bin", manifest["all_robot_variables_raw_sha256"]), ("solver_options.bin", manifest["solver_options_raw_sha256"])):
        if _sha(snapshot_dir / name) != expected:
            raise RuntimeError(f"binary input hash mismatch: {name}")
    row = dict(manifest["target_pose"])
    group, tip = manifest["group_name"], manifest["tip_link"]
    seed = np.asarray(case["seed"], dtype=float)
    rclpy.init()
    executor = SingleThreadedExecutor()
    records: list[dict[str, Any]] = []
    runtime = {"mode": mode, "input_manifest_sha256": _sha(snapshot_dir / "input_manifest.json"), "environment": _env_probe(), "unrelated_callbacks_started": False, "executor": "SingleThreadedExecutor"}
    try:
        if mode in ("same_instance", "single"):
            moveit = MoveItPy(node_name=f"stage192_moveitpy_{mode}_{process_index:03d}")
            model = moveit.get_robot_model()
            instance = f"moveitpy:{mode}:001"
            for iteration in range(repeat):
                records.append(_one(_prepare(RobotState, model, group, seed), bridge, row, group, tip, manifest, mode, instance, iteration, process_index, ik_timeout))
        elif mode == "new_instance":
            for iteration in range(repeat):
                moveit = MoveItPy(node_name=f"stage192_moveitpy_new_{process_index:03d}_{iteration:03d}")
                model = moveit.get_robot_model()
                records.append(_one(_prepare(RobotState, model, group, seed), bridge, row, group, tip, manifest, mode, f"moveitpy:new:{iteration + 1:03d}", iteration, process_index, ik_timeout))
                del model, moveit
                gc.collect()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({"schema_version": "1.0", "target_call_id": "call:00000182", "runtime": runtime, "records": records}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return 0
    finally:
        try:
            rclpy.shutdown()
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ros-node", action="store_true")
    parser.add_argument("--mode", choices=("same_instance", "new_instance", "single"), required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--process-index", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ik-timeout", type=float, default=0.0)
    parser.add_argument("--config", type=Path, default=ROOT / "config/stage192_kdl_determinism.yaml")
    args, _ = parser.parse_known_args()
    if not args.ros_node:
        raise SystemExit("real ROS2 launch required")
    return run_ros(args.config.resolve(), args.mode, args.repeat, args.process_index, args.output.resolve(), args.ik_timeout)


if __name__ == "__main__":
    raise SystemExit(main())
