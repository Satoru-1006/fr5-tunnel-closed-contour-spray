"""Run the D53 shadow collision routes against the frozen D52 finalist.

All routes write below ``outputs/D53_STAGE4_OFFLINE_CERTIFICATION_SHADOW``.
The launcher uses the D52 q/dq/ddq/jerk CSV directly for FCL and the existing
q-only adapters only for the legacy MoveIt2 Bullet executable, whose input
format is explicitly q-only.  No Stage 3 or D52 final artifact is modified.
"""

from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SHADOW = ROOT / "outputs" / "D53_STAGE4_OFFLINE_CERTIFICATION_SHADOW"
D52 = ROOT / "outputs" / "D52_STAGE4B_SHADOW"
MANIFEST = D52 / "manifests" / "stateful_local_g1_w05_0025.csv"
POST_MANIFEST = D52 / "evaluation" / "stateful_local_g1_w05_0025" / "post_manifest.csv"
POSES = ROOT / "outputs" / "internal_wiper_moveit_inputs" / "open_arch_tcp_poses_base_link.csv"
URDF = ROOT / "outputs" / "stage3_h4_5_robot_fixture_geometry_codesign_20260809T000000Z" / "derived_robot_model.urdf"
SRDF = ROOT / "ros2_moveit_bridge" / "config" / "fairino5_v6_spray_tcp.srdf"
FCL = ROOT / "tmp" / "stage4d_install" / "lib" / "stage4d_continuous_self_collision" / "stage4d_continuous_self_collision"
CLEARANCE = ROOT / "tmp" / "stage4e_install" / "stage4e_continuous_clearance" / "lib" / "stage4e_continuous_clearance" / "stage4e_continuous_clearance"
BULLET = ROOT / "tmp" / "d41_install" / "lib" / "stage3_h13_d41_native" / "stage3_h13_d41_native"


def wsl(path: Path) -> str:
    resolved = path.resolve()
    return f"/mnt/{resolved.drive.rstrip(':').lower()}{resolved.as_posix()[2:]}"


def cases() -> list[str]:
    with MANIFEST.open(newline="", encoding="utf-8") as stream:
        return [row["case_id"] for row in csv.DictReader(stream)]


def env() -> str:
    return (
        "source /opt/ros/jazzy/setup.bash; "
        f"source {shlex.quote(wsl(ROOT / 'install' / 'setup.bash'))}; "
        "export LD_LIBRARY_PATH="
        f"{wsl(ROOT / 'tmp' / 'stage4d_install' / 'lib')}:{wsl(ROOT / 'tmp' / 'd41_install' / 'lib')}:"
        f"{wsl(ROOT / 'install' / 'lib')}:/opt/ros/jazzy/lib:/opt/ros/jazzy/lib/x86_64-linux-gnu:"
        "/opt/ros/jazzy/opt/sdformat_vendor/lib:/opt/ros/jazzy/opt/gz_math_vendor/lib:"
        "/opt/ros/jazzy/opt/gz_utils_vendor/lib:/opt/ros/jazzy/opt/gz_tools_vendor/lib:/usr/lib/x86_64-linux-gnu; "
    )


def run_fcl(case_id: str, timeout_s: int) -> int:
    output = SHADOW / "routeA_d52_full" / case_id
    output.mkdir(parents=True, exist_ok=True)
    command = env() + "exec " + " ".join(shlex.quote(item) for item in (
        wsl(FCL), "--manifest", wsl(MANIFEST), "--urdf", wsl(URDF), "--srdf", wsl(SRDF),
        "--output", wsl(output), "--case", case_id,
    ))
    log = output / "launch.log"
    with log.open("w", encoding="utf-8", errors="replace") as stream:
        try:
            result = subprocess.run(["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", command], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout_s)
            return result.returncode
        except subprocess.TimeoutExpired:
            stream.write("\nD53_ROUTE_A_TIMEOUT\n")
            return 124


def run_clearance(timeout_s: int) -> int:
    output = SHADOW / "routeB_d52_full_endpoint_jerk_cone"
    output.mkdir(parents=True, exist_ok=True)
    command = env() + "exec " + " ".join(shlex.quote(item) for item in (
        wsl(CLEARANCE), "--poses", wsl(POSES), "--manifest", wsl(MANIFEST), "--urdf", wsl(URDF), "--srdf", wsl(SRDF),
        "--output", wsl(output), "--max-depth", "12", "--stride", "1", "--threshold", "0", "--motion-model", "endpoint_jerk_cone",
    ))
    log = output / "launch.log"
    with log.open("w", encoding="utf-8", errors="replace") as stream:
        try:
            result = subprocess.run(["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", command], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout_s)
            return result.returncode
        except subprocess.TimeoutExpired:
            stream.write("\nD53_ROUTE_B_TIMEOUT\n")
            return 124


def run_bullet(timeout_s: int) -> int:
    output = SHADOW / "routeC_moveit2_bullet_full"
    output.mkdir(parents=True, exist_ok=True)
    command = env() + "export D41_SKIP_CONTROLS=1; exec " + " ".join(shlex.quote(item) for item in (
        wsl(BULLET), "--poses", wsl(POSES), "--cases", wsl(POST_MANIFEST), "--urdf", wsl(URDF), "--srdf", wsl(SRDF), "--output", wsl(output),
    ))
    log = output / "launch.log"
    with log.open("w", encoding="utf-8", errors="replace") as stream:
        try:
            result = subprocess.run(["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", command], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout_s)
            return result.returncode
        except subprocess.TimeoutExpired:
            stream.write("\nD53_ROUTE_C_TIMEOUT\n")
            return 124


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--routes", default="a,b,c", help="comma-separated subset of a,b,c")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--timeout-s", type=int, default=8 * 60 * 60)
    args = parser.parse_args(argv)
    required = [MANIFEST, POST_MANIFEST, POSES, URDF, SRDF]
    selected = {item.strip().lower() for item in args.routes.split(",") if item.strip()}
    if not selected.issubset({"a", "b", "c"}) or not selected:
        raise ValueError("--routes must be a non-empty subset of a,b,c")
    if "a" in selected:
        with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as pool:
            futures = {pool.submit(run_fcl, case_id, args.timeout_s): case_id for case_id in cases()}
            for future in as_completed(futures):
                print(f"routeA {futures[future]} return_code={future.result()}", flush=True)
    if "b" in selected:
        print(f"routeB return_code={run_clearance(args.timeout_s)}", flush=True)
    if "c" in selected:
        print(f"routeC return_code={run_bullet(args.timeout_s)}", flush=True)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        print("missing required input: " + "; ".join(missing), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
