"""D54 controlled-box execution driver.

This module owns only the D54 shadow output directory.  It repairs the one
reproducible exporter timestamp bridge in a copied trajectory representation,
launches the new FK-aware self-collision certificate, and reruns the existing
native MoveIt2 Bullet robot-world cross-check.  It never writes D52 or Stage 3
scientific state.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
from pathlib import Path
from typing import Iterable

import numpy as np

from tools.d52_motion_repair import (
    CASES,
    JOINTS,
    NativeTrajectory,
    _bridge_state_residual,
    _small_step_statistics,
    case_family,
    read_native,
    write_native,
)


ROOT = Path(__file__).resolve().parents[1]
D52 = ROOT / "outputs" / "D52_STAGE4B_SHADOW"
D54 = ROOT / "outputs" / "D54_STAGE4_OFFLINE_TRAJECTORY_CERTIFICATION"
SOURCE_DIR = D52 / "candidates" / "stateful_local_g1_w05_0025" / "trajectories"
REPAIRED_DIR = D54 / "measurement_repaired_trajectories"
REPAIRED_MANIFEST = D54 / "D54_REPAIRED_MANIFEST.csv"
Q_ONLY_DIR = D54 / "native_bullet_q_only"
BULLET_MANIFEST = D54 / "D54_BULLET_Q_ONLY_MANIFEST.csv"
URDF = ROOT / "outputs" / "stage3_h4_5_robot_fixture_geometry_codesign_20260809T000000Z" / "derived_robot_model.urdf"
SRDF = ROOT / "ros2_moveit_bridge" / "config" / "fairino5_v6_spray_tcp.srdf"
POSES = ROOT / "outputs" / "internal_wiper_moveit_inputs" / "open_arch_tcp_poses_base_link.csv"
SELF_CERTIFIER = ROOT / "tmp" / "d54_install" / "lib" / "stage4f_articulated_certificate" / "stage4f_articulated_certificate"
BULLET = ROOT / "tmp" / "d41_install" / "lib" / "stage3_h13_d41_native" / "stage3_h13_d41_native"


def wsl_path(path: Path) -> str:
    value = str(path.resolve()).replace("\\", "/")
    if len(value) >= 3 and value[1:3] == ":/":
        return f"/mnt/{value[0].lower()}/{value[3:]}"
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _repair_one(source: Path, destination: Path, *, expected_dt: float = 0.01) -> dict[str, object]:
    """Repair only the proven bridge while retaining native variable dt later.

    The D52 trajectory exporter contains a legitimate native retiming tail, so
    D52's older repair helper is too strict when it demands every post-repair
    sample interval be <= 1.5*median.  D54 validates monotonicity and removes
    only the one large bridge; it does not resample or alter q.
    """
    original = read_native(source)
    time = original.time
    nominal, _, dt = _small_step_statistics(time)
    if abs(nominal - expected_dt) > max(1.0e-6, 0.02 * expected_dt):
        raise ValueError(f"unexpected_nominal_dt:{source}:{nominal}")
    large_indices = np.flatnonzero(dt > max(10.0 * nominal, nominal + 0.05))
    if len(large_indices) == 0:
        write_native(destination, original)
        return {
            "status": "NO_REPAIR_NEEDED",
            "source": str(source.resolve()),
            "destination": str(destination.resolve()),
            "original_state_count": len(original.rows),
            "repaired_state_count": len(original.rows),
            "nominal_dt_s": nominal,
            "large_gap_count": 0,
            "q_max_abs_delta_rad": 0.0,
        }
    if len(large_indices) != 1:
        raise ValueError(f"ambiguous_large_gap_count:{source}:{len(large_indices)}")
    bridge_index = int(large_indices[0] + 1)
    evidence = _bridge_state_residual(original, bridge_index, nominal)
    if abs(evidence["bridge_to_right_dt_s"] - nominal) > max(1.0e-6, 0.02 * nominal):
        raise ValueError(f"bridge_following_step_not_regular:{source}:{evidence}")
    if max(evidence["bridge_q_from_left_position_residual_rad"], evidence["bridge_q_from_right_position_residual_rad"]) > 5.0e-6:
        raise ValueError(f"bridge_position_not_continuous:{source}:{evidence}")
    gap = float(dt[bridge_index - 1])
    excess = gap - nominal
    rows: list[dict[str, float]] = []
    for index, row in enumerate(original.rows):
        updated = dict(row)
        if index == bridge_index:
            updated["t"] = float(original.time[index - 1] + nominal)
            for joint_index, joint in enumerate(JOINTS):
                left_q = original.q[index - 1, joint_index]
                right_q = original.q[index + 1, joint_index]
                updated[f"{joint}_dq"] = float((right_q - left_q) / (2.0 * nominal))
                left_v = original.dq[index - 1, joint_index]
                right_v = original.dq[index + 1, joint_index]
                updated[f"{joint}_ddq"] = float((right_v - left_v) / (2.0 * nominal))
        if index > bridge_index:
            updated["t"] -= excess
        rows.append(updated)
    repaired = NativeTrajectory(original.fields, tuple(rows))
    # Recompute only the diagnostic jerk field after the timebase repair.  The
    # native post-Ruckig q/dq/ddq fields remain the certificate inputs.
    if all(f"{joint}_jerk" in repaired.fields for joint in JOINTS):
        diagnostic = np.gradient(repaired.ddq, repaired.time, axis=0, edge_order=1)
        mutable = [dict(row) for row in repaired.rows]
        for index, row in enumerate(mutable):
            for joint_index, joint in enumerate(JOINTS):
                row[f"{joint}_jerk"] = float(diagnostic[index, joint_index])
        repaired = NativeTrajectory(repaired.fields, tuple(mutable))
    repaired_dt = np.diff(repaired.time)
    if np.any(repaired_dt <= 0.0) or not np.isfinite(repaired_dt).all():
        raise ValueError(f"repair_left_nonmonotone_time:{source}")
    q_delta = float(np.max(np.abs(repaired.q - original.q)))
    write_native(destination, repaired)
    return {
        "status": "REPAIRED",
        "source": str(source.resolve()),
        "destination": str(destination.resolve()),
        "original_state_count": len(original.rows),
        "repaired_state_count": len(repaired.rows),
        "nominal_dt_s": nominal,
        "large_gap_count": 1,
        "repaired_bridge_index_zero_based": bridge_index,
        "repaired_bridge_gap_s": gap,
        "timestamp_shift_after_bridge_s": excess,
        "bridge_evidence": evidence,
        "repaired_max_dt_s": float(np.max(repaired_dt)),
        "repaired_min_dt_s": float(np.min(repaired_dt)),
        "q_max_abs_delta_rad": q_delta,
        "q_path_preserved": q_delta == 0.0,
        "bridge_q_preserved_derivatives_reconstructed": True,
        "native_variable_dt_tail_retained": True,
    }


def repair_baseline(case_ids: Iterable[str] = CASES) -> dict[str, object]:
    selected = tuple(case_ids)
    records = []
    for case_id in selected:
        records.append(_repair_one(SOURCE_DIR / f"{case_id}.csv", REPAIRED_DIR / f"{case_id}.csv"))
    summary: dict[str, object] = {
        "schema_version": "d54-measurement-timebase-repair-v1",
        "status": "PASS",
        "classification": "TYPE_A_MEASUREMENT_INFRASTRUCTURE_DEFECT",
        "method": "remove_one_proven_exporter_timestamp_bridge_and_shift_following_native_states",
        "scientific_state_mutated": False,
        "collision_method": "adaptive_discrete_interpolation",
        "case_count": len(records),
        "repaired_case_count": sum(record["status"] == "REPAIRED" for record in records),
        "cases": records,
    }
    _write_json(D54 / "D54_TIMEBASE_REPAIR_SUMMARY.json", summary)
    write_manifest(REPAIRED_MANIFEST, REPAIRED_DIR, selected)
    for case_id in selected:
        write_q_only(REPAIRED_DIR / f"{case_id}.csv", Q_ONLY_DIR / f"{case_id}.csv")
    write_manifest(BULLET_MANIFEST, Q_ONLY_DIR, selected)
    return summary


def write_manifest(path: Path, trajectory_dir: Path, case_ids: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("case_id", "trajectory_csv", "family"))
        for case_id in case_ids:
            trajectory = trajectory_dir / f"{case_id}.csv"
            if not trajectory.is_file():
                raise FileNotFoundError(trajectory)
            writer.writerow((case_id, wsl_path(trajectory), case_family(case_id)))


def write_q_only(source: Path, destination: Path) -> None:
    native = read_native(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("waypoint", *(f"{joint}_q" for joint in JOINTS)))
        for index, row in enumerate(native.rows):
            writer.writerow((index, *(row[f"{joint}_q"] for joint in JOINTS)))


def _ros_env(*, bullet: bool = False) -> str:
    prefixes = [
        "/mnt/d/robotfucker/tmp/d54_install/lib",
        "/mnt/d/robotfucker/tmp/d41_install/lib",
        "/mnt/d/robotfucker/install/lib",
        "/opt/ros/jazzy/lib",
        "/opt/ros/jazzy/lib/x86_64-linux-gnu",
        "/opt/ros/jazzy/opt/sdformat_vendor/lib",
        "/opt/ros/jazzy/opt/gz_math_vendor/lib",
        "/opt/ros/jazzy/opt/gz_utils_vendor/lib",
        "/opt/ros/jazzy/opt/gz_tools_vendor/lib",
        "/usr/lib/x86_64-linux-gnu",
    ]
    lines = [
        "set -e",
        "source /opt/ros/jazzy/setup.bash",
        "source /mnt/d/robotfucker/install/setup.bash",
        "source /mnt/d/robotfucker/tmp/d54_install/setup.bash",
        f"export LD_LIBRARY_PATH={':'.join(prefixes)}:${{LD_LIBRARY_PATH:-}}",
    ]
    if bullet:
        lines.append("export D41_SKIP_CONTROLS=1")
    return "; ".join(lines)


def run_self_certificate(output: Path, *, case_id: str | None = None, initial_stride: int = 16, max_depth: int = 12) -> int:
    output.mkdir(parents=True, exist_ok=True)
    args = [
        wsl_path(SELF_CERTIFIER),
        "--manifest", wsl_path(REPAIRED_MANIFEST),
        "--urdf", wsl_path(URDF),
        "--srdf", wsl_path(SRDF),
        "--output", wsl_path(output),
        "--initial-stride", str(initial_stride),
        "--max-depth", str(max_depth),
        "--jerk-bound", "8.0",
        "--threshold", "0.0",
    ]
    if case_id:
        args += ["--case", case_id]
    command = _ros_env() + "; exec " + " ".join(shlex.quote(item) for item in args)
    log_path = output / "launch.log"
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            ["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", command],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=6 * 60 * 60,
        )
    return completed.returncode


def run_bullet(output: Path) -> int:
    output.mkdir(parents=True, exist_ok=True)
    args = [
        wsl_path(BULLET),
        "--poses", wsl_path(POSES),
        "--cases", wsl_path(BULLET_MANIFEST),
        "--urdf", wsl_path(URDF),
        "--srdf", wsl_path(SRDF),
        "--output", wsl_path(output),
    ]
    command = _ros_env(bullet=True) + "; exec " + " ".join(shlex.quote(item) for item in args)
    log_path = output / "launch.log"
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            ["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", command],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=6 * 60 * 60,
        )
    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repair", action="store_true")
    parser.add_argument("--certify", action="store_true")
    parser.add_argument("--bullet", action="store_true")
    parser.add_argument("--case", default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if not any((args.repair, args.certify, args.bullet)):
        parser.error("select --repair, --certify, or --bullet")
    if args.repair:
        summary = repair_baseline((args.case,) if args.case else CASES)
        print(json.dumps({"status": summary["status"], "case_count": summary["case_count"], "repaired_case_count": summary["repaired_case_count"]}, sort_keys=True))
    if args.certify:
        output = args.output or (D54 / "self_collision_repaired")
        return_code = run_self_certificate(output, case_id=args.case)
        if return_code:
            return return_code
    if args.bullet:
        output = args.output or (D54 / "native_environment_ccd_repaired")
        return_code = run_bullet(output)
        if return_code:
            return return_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
