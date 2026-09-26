"""Run the inherited FK-aware articulated certificate on D58 shadows."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CERTIFIER = ROOT / "tmp" / "d54_install" / "lib" / "stage4f_articulated_certificate" / "stage4f_articulated_certificate"
URDF = ROOT / "outputs" / "stage3_h4_5_robot_fixture_geometry_codesign_20260809T000000Z" / "derived_robot_model.urdf"
SRDF = ROOT / "ros2_moveit_bridge" / "config" / "fairino5_v6_spray_tcp.srdf"


def wsl_path(path: Path) -> str:
    value = str(path.resolve()).replace("\\", "/")
    if len(value) >= 3 and value[1:3] == ":/":
        return f"/mnt/{value[0].lower()}/{value[3:]}"
    return value


def write_manifest(native_dir: Path, output: Path) -> Path:
    summary = json.loads((native_dir / "execution_form_summary.json").read_text(encoding="utf-8"))
    manifest = output / "certificate_manifest.csv"
    output.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("case_id", "trajectory_csv", "family"))
        for item in summary.get("cases", []):
            if item.get("status") == "PASS":
                writer.writerow((item["case_id"], item["trajectory_csv"], item["family"]))
    return manifest


def run(native_dir: Path, output: Path, initial_stride: int = 16, max_depth: int = 12) -> int:
    manifest = write_manifest(native_dir, output)
    args = [
        wsl_path(CERTIFIER),
        "--manifest", wsl_path(manifest),
        "--urdf", wsl_path(URDF),
        "--srdf", wsl_path(SRDF),
        "--output", wsl_path(output),
        "--initial-stride", str(initial_stride),
        "--max-depth", str(max_depth),
        "--jerk-bound", "8.0",
        "--threshold", "0.0",
    ]
    command = "; ".join([
        "set -e",
        "source /opt/ros/jazzy/setup.bash",
        "source /mnt/d/robotfucker/install/setup.bash",
        "source /mnt/d/robotfucker/tmp/d54_install/setup.bash",
        "export LD_LIBRARY_PATH=/mnt/d/robotfucker/tmp/d54_install/lib:/mnt/d/robotfucker/tmp/d41_install/lib:/mnt/d/robotfucker/install/lib:/opt/ros/jazzy/lib:/opt/ros/jazzy/lib/x86_64-linux-gnu:/opt/ros/jazzy/opt/sdformat_vendor/lib:/opt/ros/jazzy/opt/gz_math_vendor/lib:/opt/ros/jazzy/opt/gz_utils_vendor/lib:/opt/ros/jazzy/opt/gz_tools_vendor/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}",
        "exec " + " ".join(shlex.quote(item) for item in args),
    ])
    with (output / "certificate_launch.log").open("w", encoding="utf-8") as stream:
        completed = subprocess.run(["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", command], stdout=stream, stderr=subprocess.STDOUT, check=False, timeout=6 * 60 * 60)
    return int(completed.returncode)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initial-stride", type=int, default=16)
    parser.add_argument("--max-depth", type=int, default=12)
    args = parser.parse_args()
    return run(args.native_dir.resolve(), args.output_dir.resolve(), args.initial_stride, args.max_depth)


if __name__ == "__main__":
    raise SystemExit(main())
