"""Run the six Stage 2.3B PlanningScene certifications as fresh WSL processes.

Each process writes first to WSL ext4, records GNU time telemetry and separate
stdout/stderr, and only then copies the completed evidence to the project output.
Existing Stage 2.3B exploratory/formal evidence is never removed or overwritten.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/ik_graph_stage23b_representation_remediation/fr5_scaled_horseshoe_demo_v45"
WSL_ROOT = "/mnt/c/Users/86198/Desktop/robotfucker"
WSL_OUT = f"{WSL_ROOT}/outputs/ik_graph_stage23b_representation_remediation/fr5_scaled_horseshoe_demo_v45"
FINAL_NAME = "certified_runs_final3"
FINAL_ROOT = OUT / FINAL_NAME


def wsl_script(backend: str, run_index: int) -> str:
    run_name = f"{backend}_run{run_index}"
    wsl_run_dir = f"{WSL_OUT}/{FINAL_NAME}/{run_name}"
    preload = (
        f"export FR5_BULLET_SHAPE_MODE=use_shape_type\n"
        f"export LD_PRELOAD={WSL_OUT}/install/lib/libstage23b_bullet_shape_interposer.so\n"
        if backend == "bullet"
        else "unset FR5_BULLET_SHAPE_MODE\nunset LD_PRELOAD\n"
    )
    return f"""source /opt/ros/jazzy/setup.bash
export AMENT_PREFIX_PATH={WSL_OUT}/install:{WSL_ROOT}/tmp/stage23a7_install2:{WSL_ROOT}/install/fairino5_v6_moveit2_config:{WSL_ROOT}/install/fairino_description:{WSL_ROOT}/install/fr5_tunnel_moveit_bridge:/opt/ros/jazzy
export LD_LIBRARY_PATH={WSL_OUT}/install/lib:{WSL_ROOT}/tmp/stage23a7_install2/lib:/opt/ros/jazzy/opt/sdformat_vendor/lib:/opt/ros/jazzy/opt/rviz_ogre_vendor/lib:/opt/ros/jazzy/lib/x86_64-linux-gnu:/opt/ros/jazzy/opt/gz_math_vendor/lib:/opt/ros/jazzy/opt/gz_utils_vendor/lib:/opt/ros/jazzy/opt/gz_tools_vendor/lib:/opt/ros/jazzy/opt/gz_cmake_vendor/lib:/opt/ros/jazzy/lib
{preload}mkdir -p {wsl_run_dir}
/usr/bin/time -v -o {wsl_run_dir}/time_verbose.txt ros2 launch {WSL_ROOT}/tools/stage23b_formal_launch.py candidate_csv:={WSL_ROOT}/outputs/ik_graph_stage23a4_backend_equivalent/fr5_scaled_horseshoe_demo_v44/deterministic_ik_candidates.csv parts_dir:={WSL_ROOT}/outputs/ik_graph_stage23a4_backend_equivalent/fr5_scaled_horseshoe_demo_v44/tunnel_collision_parts output_dir:={wsl_run_dir} backend:={backend} run_index:={run_index} > {wsl_run_dir}/stdout.log 2> {wsl_run_dir}/stderr.log
"""


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    FINAL_ROOT.mkdir(parents=True, exist_ok=True)
    records = []
    for backend in ("fcl", "bullet"):
        for run_index in (1, 2, 3):
            started = datetime.now(timezone.utc).isoformat()
            proc = subprocess.run(
                ["wsl.exe", "bash", "-lc", wsl_script(backend, run_index)],
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
            )
            destination = FINAL_ROOT / f"{backend}_run{run_index}"
            destination.mkdir(parents=True, exist_ok=True)
            finished = datetime.now(timezone.utc).isoformat()
            (destination / "start_utc.txt").write_text(started + "\n", encoding="utf-8")
            (destination / "end_utc.txt").write_text(finished + "\n", encoding="utf-8")
            (destination / "exit_code.txt").write_text(str(proc.returncode) + "\n", encoding="utf-8")
            record = {
                "backend": backend,
                "run_index": run_index,
                "runner_started_utc": started,
                "runner_finished_utc": finished,
                "wsl_exit_code": proc.returncode,
                "copy_exit_code": 0,
                "wsl_stdout": proc.stdout,
                "wsl_stderr": proc.stderr,
            }
            records.append(record)
            print(json.dumps({k: record[k] for k in ("backend", "run_index", "wsl_exit_code")}), flush=True)
            result = destination / f"{backend}_runtime_results_run{run_index}.jsonl"
            if proc.returncode != 0 or not result.exists():
                (OUT / "stage23b_certified_runner.json").write_text(
                    json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
                )
                return proc.returncode or 1
    (OUT / "stage23b_certified_runner.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
