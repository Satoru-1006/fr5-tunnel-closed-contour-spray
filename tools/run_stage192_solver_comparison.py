#!/usr/bin/env python3
"""Run all Stage 1.9.2 minimum-case replays in isolated ROS2/KDL processes."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/ik_graph_stage192"
WSL_ROOT = "/mnt/c/Users/86198/Desktop/robotfucker"
DISTRO = "Ubuntu-24.04-D"
LAUNCH = f"{WSL_ROOT}/tools/stage192_moveitpy_launch.py"
BIN_DIR = f"{WSL_ROOT}/outputs/ik_graph_stage192/cpp_build"
SNAP = f"{WSL_ROOT}/outputs/ik_graph_stage192/minimal_case"
URDF = f"{SNAP}/fairino5_v6_spray_tcp.expanded.urdf"
SRDF = f"{WSL_ROOT}/ros2_moveit_bridge/config/fairino5_v6_spray_tcp.srdf"


def run_wsl(script: str, log: Path, timeout: int = 1800) -> tuple[int, str]:
    command = ["wsl.exe", "-d", DISTRO, "--", "bash", "-lc", script]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    text = result.stdout + "\n" + result.stderr
    log.parent.mkdir(parents=True, exist_ok=True); log.write_text(text, encoding="utf-8")
    matches = re.findall(r"STAGE192_RETURN_CODE=(-?\d+)", text)
    return (int(matches[-1]) if matches else result.returncode), text


def ros_launch(mode: str, repeat: int, output: Path, log: Path, index: int, timeout: float = 0.0) -> tuple[int, str]:
    output_wsl = output.as_posix().replace("C:/", "/mnt/c/").replace("\\", "/")
    script = f"set +e; source /opt/ros/jazzy/setup.bash; source {WSL_ROOT}/install/setup.bash; export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; ros2 launch {LAUNCH} stage192_mode:={mode} stage192_repeat:={repeat} stage192_process_index:={index} stage192_ik_timeout:={timeout} stage192_output:={output_wsl}; rc=$?; echo STAGE192_RETURN_CODE=$rc; exit 0"
    return run_wsl(script, log)


def cpp_run(binary: str, backend_dir: str, mode: str, repeat: int, output: Path, log: Path) -> tuple[int, str]:
    output_wsl = output.as_posix().replace("C:/", "/mnt/c/").replace("\\", "/")
    root_wsl = f"{WSL_ROOT}/{backend_dir}"
    script = f"set +e; source /opt/ros/jazzy/setup.bash; source {WSL_ROOT}/install/setup.bash; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; {BIN_DIR}/{binary} --urdf {URDF} --target {SNAP}/target_transform_f64.bin --seed {SNAP}/seed_joint_vector_f64.bin --limits {SNAP}/joint_limits_f64.bin --mode {mode} --repeat {repeat}"
    if binary == "stage192_moveit_kdl_minimal": script += f" --srdf {SRDF}"
    script += f" --output {output_wsl}; rc=$?; echo STAGE192_RETURN_CODE=$rc; exit 0"
    result = run_wsl(script, log)
    if output.exists():
        data = json.loads(output.read_text(encoding="utf-8"))
        manifest = json.loads((OUT / "minimal_case/input_manifest.json").read_text(encoding="utf-8"))
        for record in data.get("records", []):
            record.update({"target_call_id": "call:00000182", "target_transform_raw_sha256": manifest["target_transform_raw_sha256"], "seed_vector_raw_sha256": manifest["seed_vector_raw_sha256"], "all_robot_variables_raw_sha256": manifest["all_robot_variables_raw_sha256"], "solver_options_raw_sha256": manifest["solver_options_raw_sha256"]})
        output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return result


def aggregate_independent(kind: str, runner, count: int, directory: Path) -> list[dict]:
    records = []
    for index in range(1, count + 1):
        path = directory / f"process_{index:03d}.json"
        runner(index, path, directory / f"process_{index:03d}.log")
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8")); records.extend(data.get("records", []))
    aggregate = directory / "comparison.json"
    aggregate.write_text(json.dumps({"schema_version": "1.0", "count": len(records), "records": records}, indent=2) + "\n", encoding="utf-8")
    return records


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if "--aggregate-only" in sys.argv:
        manifest = json.loads((OUT / "minimal_case/input_manifest.json").read_text(encoding="utf-8"))
        for directory in (OUT / "cpp_moveit_plugin_independent_processes", OUT / "direct_orocos_kdl/independent_processes"):
            rows = []
            for p in sorted(directory.glob("process_*.json")):
                data = json.loads(p.read_text(encoding="utf-8"))
                for record in data.get("records", []):
                    record.update({"target_call_id": "call:00000182", "target_transform_raw_sha256": manifest["target_transform_raw_sha256"], "seed_vector_raw_sha256": manifest["seed_vector_raw_sha256"], "all_robot_variables_raw_sha256": manifest["all_robot_variables_raw_sha256"], "solver_options_raw_sha256": manifest["solver_options_raw_sha256"]})
                p.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
                rows.extend(data.get("records", []))
            (directory / "comparison.json").write_text(json.dumps({"schema_version": "1.0", "count": len(rows), "records": rows}, indent=2) + "\n", encoding="utf-8")
        for p in (OUT / "cpp_moveit_plugin_same_instance/replay.json", OUT / "cpp_moveit_plugin_new_instance/replay.json", OUT / "direct_orocos_kdl/same_instance.json", OUT / "direct_orocos_kdl/new_instance.json"):
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                for record in data.get("records", []):
                    record.update({"target_call_id": "call:00000182", "target_transform_raw_sha256": manifest["target_transform_raw_sha256"], "seed_vector_raw_sha256": manifest["seed_vector_raw_sha256"], "all_robot_variables_raw_sha256": manifest["all_robot_variables_raw_sha256"], "solver_options_raw_sha256": manifest["solver_options_raw_sha256"]})
                p.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        return 0
    # The formal zero-timeout replay is intentionally first and immutable in meaning.
    ros_launch("same_instance", 100, OUT / "moveitpy_same_instance/replay.json", OUT / "moveitpy_same_instance/replay.log", 1, 0.0)
    ros_launch("new_instance", 50, OUT / "moveitpy_new_instance/replay.json", OUT / "moveitpy_new_instance/replay.log", 1, 0.0)
    if not (OUT / "moveitpy_new_instance/replay.json").exists():
        # A MoveItPy destructor can terminate the parent before the next instance.
        # Each child still creates a fresh solver and is recorded as an independent new-instance observation.
        def new_runner(index, path, log): return ros_launch("single", 1, path, log, index, 0.0)
        aggregate_independent("moveitpy_new_instance_child", new_runner, 50, OUT / "moveitpy_new_instance/child_processes")
    def independent_runner(index, path, log): return ros_launch("single", 1, path, log, index, 0.0)
    aggregate_independent("moveitpy_independent", independent_runner, 20, OUT / "moveitpy_independent_processes")
    cpp_run("stage192_moveit_kdl_minimal", "cpp_moveit_plugin_same_instance", "same_instance", 100, OUT / "cpp_moveit_plugin_same_instance/replay.json", OUT / "cpp_moveit_plugin_same_instance/replay.log")
    cpp_run("stage192_moveit_kdl_minimal", "cpp_moveit_plugin_new_instance", "new_instance", 50, OUT / "cpp_moveit_plugin_new_instance/replay.json", OUT / "cpp_moveit_plugin_new_instance/replay.log")
    def cpp_new(index, path, log): return cpp_run("stage192_moveit_kdl_minimal", "cpp_moveit_plugin_independent_processes", "new_instance", 1, path, log)
    aggregate_independent("cpp_moveit_plugin_independent", cpp_new, 20, OUT / "cpp_moveit_plugin_independent_processes")
    cpp_run("stage192_direct_orocos_kdl", "direct_orocos_kdl", "same_instance", 100, OUT / "direct_orocos_kdl/same_instance.json", OUT / "direct_orocos_kdl/same_instance.log")
    cpp_run("stage192_direct_orocos_kdl", "direct_orocos_kdl", "new_instance", 50, OUT / "direct_orocos_kdl/new_instance.json", OUT / "direct_orocos_kdl/new_instance.log")
    def direct_new(index, path, log): return cpp_run("stage192_direct_orocos_kdl", "direct_orocos_kdl", "new_instance", 1, path, log)
    aggregate_independent("direct_orocos_kdl_independent", direct_new, 20, OUT / "direct_orocos_kdl/independent_processes")
    print("stage192 comparison executions complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
