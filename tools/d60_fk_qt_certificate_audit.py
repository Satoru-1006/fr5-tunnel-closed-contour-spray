"""Run and independently audit the strongest available FK(q(t)) certificate.

The existing D54/Stage4F executable is retained as the MoveIt2/FCL narrow
phase authority.  D60 runs it on the correctly routed D59 candidates with
initial_stride=1 and a deeper refinement budget, then independently audits the
exported q,dq,ddq,jerk contract before accepting the result as certificate
evidence.  All output is written below D60_SYSTEM_LEVEL_CLOSURE and no
protected D56--D59 artifact is modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "D60_SYSTEM_LEVEL_CLOSURE" / "fk_qt_certificate"
URDF = ROOT / "outputs" / "stage3_h4_5_robot_fixture_geometry_codesign_20260809T000000Z" / "derived_robot_model.urdf"
SRDF = ROOT / "ros2_moveit_bridge" / "config" / "fairino5_v6_spray_tcp.srdf"
CERTIFIER = ROOT / "tmp" / "d54_install" / "lib" / "stage4f_articulated_certificate" / "stage4f_articulated_certificate"
TRAJECTORIES = {
    "d59_auto0": ROOT / "outputs" / "D59_OFFLINE_ALGORITHM_SYSTEM_CLOSURE" / "timing_scale025_auto0" / "trajectories" / "adversarial_0100.csv",
    "d59_auto1_corrected": ROOT / "outputs" / "D59_OFFLINE_ALGORITHM_SYSTEM_CLOSURE" / "timing_scale025_corrected_auto1" / "trajectories" / "adversarial_0101.csv",
}


def wsl_path(path: Path) -> str:
    value = str(path.resolve()).replace("\\", "/")
    if len(value) >= 3 and value[1:3] == ":/":
        return f"/mnt/{value[0].lower()}/{value[3:]}"
    return value


def write_manifest(output: Path) -> Path:
    manifest = output / "certificate_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("case_id", "trajectory_csv", "family"))
        for case_id, trajectory in TRAJECTORIES.items():
            writer.writerow((case_id, wsl_path(trajectory), "D59_CORRECTLY_ROUTED_FINALIST"))
    return manifest


def run_certifier(output: Path, manifest: Path, initial_stride: int, max_depth: int, rerun: bool) -> dict[str, Any]:
    existing = output / "D54_CONTINUOUS_SELF_COLLISION_CERTIFICATION.json"
    if existing.exists() and not rerun:
        result = json.loads(existing.read_text(encoding="utf-8"))
        return {
            "status": "PASS" if result.get("status") == "PASS" else "BLOCKED",
            "returncode": 0 if result.get("status") == "PASS" else 2,
            "certificate_report": str(existing),
            "certificate": result,
            "parameters": {"initial_stride": initial_stride, "max_depth": max_depth, "jerk_bound_rad_s3": 8.0, "threshold_m": 0.0},
            "execution": "REUSED_EXISTING_ISOLATED_D60_CERTIFICATE",
        }
    if not CERTIFIER.exists():
        return {"status": "NOT_AVAILABLE", "reason": "D54_certifier_binary_missing", "binary": str(CERTIFIER)}
    command = "; ".join(
        [
            "set -e",
            "source /opt/ros/jazzy/setup.bash",
            "source /mnt/d/robotfucker/install/setup.bash",
            "source /mnt/d/robotfucker/tmp/d54_install/setup.bash",
            "export LD_LIBRARY_PATH=/mnt/d/robotfucker/tmp/d54_install/lib:/mnt/d/robotfucker/tmp/d41_install/lib:/mnt/d/robotfucker/install/lib:/opt/ros/jazzy/lib:/opt/ros/jazzy/lib/x86_64-linux-gnu:/opt/ros/jazzy/opt/sdformat_vendor/lib:/opt/ros/jazzy/opt/gz_math_vendor/lib:/opt/ros/jazzy/opt/gz_utils_vendor/lib:/opt/ros/jazzy/opt/gz_tools_vendor/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}",
            "exec "
            + " ".join(
                shlex.quote(value)
                for value in (
                    wsl_path(CERTIFIER),
                    "--manifest",
                    wsl_path(manifest),
                    "--urdf",
                    wsl_path(URDF),
                    "--srdf",
                    wsl_path(SRDF),
                    "--output",
                    wsl_path(output),
                    "--initial-stride",
                    str(initial_stride),
                    "--max-depth",
                    str(max_depth),
                    "--jerk-bound",
                    "8.0",
                    "--threshold",
                    "0.0",
                )
            ),
        ]
    )
    completed = subprocess.run(
        ["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", command],
        check=False,
        capture_output=True,
        timeout=6 * 60 * 60,
    )
    stdout = completed.stdout.decode("utf-8", errors="replace")
    stderr = completed.stderr.decode("utf-8", errors="replace")
    (output / "certificate_launch.log").write_text(stdout + "\n--- STDERR ---\n" + stderr, encoding="utf-8")
    result_path = output / "D54_CONTINUOUS_SELF_COLLISION_CERTIFICATION.json"
    if not result_path.exists():
        return {"status": "BLOCKED", "returncode": completed.returncode, "reason": "certificate_report_missing"}
    result = json.loads(result_path.read_text(encoding="utf-8"))
    return {
        "status": "PASS" if completed.returncode == 0 and result.get("status") == "PASS" else "BLOCKED",
        "returncode": completed.returncode,
        "certificate_report": str(result_path),
        "certificate": result,
        "parameters": {"initial_stride": initial_stride, "max_depth": max_depth, "jerk_bound_rad_s3": 8.0, "threshold_m": 0.0},
    }


def audit_trajectory(path: Path, jerk_bound: float = 8.0) -> dict[str, Any]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    times = [float(row["t"]) for row in rows]
    q = [[float(row[f"j{joint}_q"]) for joint in range(1, 7)] for row in rows]
    velocity = [[float(row[f"j{joint}_dq"]) for joint in range(1, 7)] for row in rows]
    acceleration = [[float(row[f"j{joint}_ddq"]) for joint in range(1, 7)] for row in rows]
    jerk = [[float(row[f"j{joint}_jerk"]) for joint in range(1, 7)] for row in rows]
    def finite_matrix(matrix: list[list[float]]) -> bool:
        return all(value == value and abs(value) != float("inf") for row in matrix for value in row)

    finite = all(value == value and abs(value) != float("inf") for value in times) and all(
        finite_matrix(matrix) for matrix in (q, velocity, acceleration, jerk)
    )
    positive_dt = all(b > a for a, b in zip(times, times[1:]))
    jerk_max = max(max(abs(value) for value in row) for row in jerk)
    max_q_envelope_excess = 0.0
    max_velocity_envelope_excess = 0.0
    max_acceleration_envelope_excess = 0.0
    for index, (t0, t1) in enumerate(zip(times, times[1:])):
        dt = t1 - t0
        for joint in range(6):
            q_delta = abs(q[index + 1][joint] - q[index][joint])
            q_bound = abs(velocity[index][joint]) * dt + 0.5 * abs(acceleration[index][joint]) * dt * dt + jerk_bound * dt**3 / 6.0
            max_q_envelope_excess = max(max_q_envelope_excess, q_delta - q_bound)
            v_delta = abs(velocity[index + 1][joint] - velocity[index][joint])
            v_bound = abs(acceleration[index][joint]) * dt + 0.5 * jerk_bound * dt * dt
            max_velocity_envelope_excess = max(max_velocity_envelope_excess, v_delta - v_bound)
            a_delta = abs(acceleration[index + 1][joint] - acceleration[index][joint])
            max_acceleration_envelope_excess = max(max_acceleration_envelope_excess, a_delta - jerk_bound * dt)
    tolerance = 1.0e-9
    return {
        "trajectory": str(path),
        "state_count": len(rows),
        "interval_count": max(0, len(rows) - 1),
        "duration_s": times[-1] - times[0],
        "finite": finite,
        "strictly_increasing_time": positive_dt,
        "max_abs_exported_jerk_rad_s3": jerk_max,
        "jerk_bound_rad_s3": jerk_bound,
        "exported_jerk_within_bound": jerk_max <= jerk_bound + tolerance,
        "max_q_taylor_envelope_excess_rad": max_q_envelope_excess,
        "max_velocity_taylor_envelope_excess_rad_s": max_velocity_envelope_excess,
        "max_acceleration_taylor_envelope_excess_rad_s2": max_acceleration_envelope_excess,
        "state_transition_envelope_contract_pass": max(
            max_q_envelope_excess,
            max_velocity_envelope_excess,
            max_acceleration_envelope_excess,
        ) <= tolerance,
        "joint_order": [f"j{i}" for i in range(1, 7)],
        "measurement_semantics": "independent finite-state and bounded-jerk transition audit; not a geometry certificate",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--initial-stride", type=int, default=1)
    parser.add_argument("--max-depth", type=int, default=20)
    parser.add_argument("--rerun", action="store_true", help="rerun the isolated MoveIt2/FCL certificate instead of reusing its report")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = write_manifest(output)
    certifier = run_certifier(output, manifest, args.initial_stride, args.max_depth, args.rerun)
    audits = {case_id: audit_trajectory(path) for case_id, path in TRAJECTORIES.items()}
    certificate = certifier.get("certificate", {})
    status = "PASS" if certifier.get("status") == "PASS" and all(
        item["finite"] and item["strictly_increasing_time"] and item["exported_jerk_within_bound"] and item["state_transition_envelope_contract_pass"]
        for item in audits.values()
    ) else "BLOCKED"
    payload = {
        "schema_version": "d60-fk-qt-certificate-audit-v1",
        "status": status,
        "scope": "D59 correctly routed finalists; shadow output only",
        "certificate_backend": "MoveIt2 RobotState FK plus FCL signed distance with adaptive pairwise interval refinement",
        "certificate_semantics": "conservative lower-bound model for actual articulated FK(q(t)); not theorem-level exact external CCD",
        "certificate_run": certifier,
        "independent_trajectory_contract_audits": audits,
        "hard_gate_interpretation": {
            "collision_region_count": certificate.get("collision_region_count"),
            "unresolved_region_count": certificate.get("unresolved_region_count"),
            "required_pair_coverage_complete": certificate.get("required_pair_coverage_complete"),
            "unresolved_is_not_pass": True,
        },
    }
    (output / "D60_FK_QT_CERTIFICATE_AUDIT.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "output": str(output / "D60_FK_QT_CERTIFICATE_AUDIT.json")}, sort_keys=True))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
