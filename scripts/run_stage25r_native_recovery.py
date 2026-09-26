"""Run the isolated Stage 2.5R native Ruckig recovery three times.

This script deliberately launches only the MoveIt timing runner.  It does not
invoke Stage 2.4, Stage 2.6, controllers, or JTC.  Each rebuild has its own
interposer binary and evidence directory, and all frozen Stage 2.5 inputs are
hashed before and after the run.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
RECOVERY_ROOT = OUTPUTS / "stage25r_native_ruckig_recovery"
FROZEN = OUTPUTS / "ik_graph_stage25_timed_certification" / "fr5_scaled_horseshoe_demo_v45_20260803_formal_000002_Stage25"
WAYPOINTS = OUTPUTS / "ik_graph_stage23a4_backend_equivalent" / "fr5_scaled_horseshoe_demo_v44" / "waypoints.csv"
INTERPOSER = ROOT / "tools" / "stage25r_ruckig_interposer.cpp"
LAUNCH = ROOT / "tools" / "stage25_moveit_launch.py"


def wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    return f"/mnt/{drive}{resolved.as_posix()[2:]}"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def frozen_manifest() -> dict[str, object]:
    if not FROZEN.is_dir():
        raise FileNotFoundError(FROZEN)
    names = [
        "SHA256SUMS",
        "stage25_pre_timing_joint_trajectory.csv",
        "stage25_totg_trajectory.csv",
        "stage25_ruckig_trajectory.csv",
        "stage25_path_reference.json",
        "stage25_input_manifest.json",
        "stage25_gate_report.json",
        "stage25_ruckig_audit.json",
        "stage25_totg_audit.json",
        "stage25_joint_limits.json",
    ]
    entries: dict[str, object] = {}
    for name in names:
        path = FROZEN / name
        if path.exists():
            entries[str(path)] = {"sha256": sha256(path), "size": path.stat().st_size}
    stage24_candidates = sorted(OUTPUTS.rglob("*Stage24T"))
    for directory in stage24_candidates:
        if not directory.is_dir():
            continue
        for name in ("SHA256SUMS", "stage24t_gate_report.json", "stage24t_trajectory.csv", "stage24t_final_trajectory.csv"):
            path = directory / name
            if path.exists():
                entries[str(path)] = {"sha256": sha256(path), "size": path.stat().st_size}
    return {"schema_version": "stage25r-frozen-hashes-v1", "files": entries}


def run_wsl(command: str, log_path: Path) -> int:
    completed = subprocess.run(
        ["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", command],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=1800,
        check=False,
    )
    log_path.write_bytes(completed.stdout)
    return int(completed.returncode)


def capture_wsl(command: str) -> tuple[int, str]:
    completed = subprocess.run(
        ["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", command],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60,
        check=False,
    )
    return int(completed.returncode), completed.stdout.decode("utf-8", errors="replace")


def package_version(package_xml: str) -> str | None:
    match = re.search(r"<version>\s*([^<]+?)\s*</version>", package_xml)
    return match.group(1).strip() if match else None


def runtime_fingerprint() -> dict[str, object]:
    commands = {
        "moveit_core_package_xml": "cat /opt/ros/jazzy/share/moveit_core/package.xml",
        "ruckig_package_xml": "cat /opt/ros/jazzy/share/ruckig/package.xml",
        "trajectory_processing_sha256": "sha256sum /opt/ros/jazzy/lib/libmoveit_trajectory_processing.so",
        "compiler_version": "g++ --version",
        "trajectory_processing_build_id": "readelf -n /opt/ros/jazzy/lib/libmoveit_trajectory_processing.so",
    }
    captured: dict[str, object] = {}
    for name, command in commands.items():
        code, output = capture_wsl(command)
        captured[name] = {"exit_code": code, "stdout": output}
    moveit_xml = str(captured["moveit_core_package_xml"]["stdout"])
    ruckig_xml = str(captured["ruckig_package_xml"]["stdout"])
    return {
        "schema_version": "stage25r-runtime-fingerprint-v1",
        "ros_distribution": "jazzy",
        "moveit_core_version": package_version(moveit_xml),
        "ruckig_version": package_version(ruckig_xml),
        "moveit_core_package_xml_sha256": hashlib.sha256(moveit_xml.encode("utf-8")).hexdigest(),
        "ruckig_package_xml_sha256": hashlib.sha256(ruckig_xml.encode("utf-8")).hexdigest(),
        "trajectory_processing_binary": captured["trajectory_processing_sha256"],
        "compiler": captured["compiler_version"],
        "trajectory_processing_build_id": captured["trajectory_processing_build_id"],
        "formal_run_binary_fingerprint_persisted": False,
        "note": "This fingerprint records the isolated recovery runtime and is not written into frozen Stage 2.5 artifacts.",
    }


def compile_interposer(output: Path) -> tuple[int, str]:
    so = wsl_path(output / "libstage25r_ruckig_interposer.so")
    source = wsl_path(INTERPOSER)
    command = (
        "CXXFLAGS=-I/opt/ros/jazzy/include; "
        "for d in /opt/ros/jazzy/include/*; do CXXFLAGS=\"$CXXFLAGS -I$d\"; done; "
        "g++ $CXXFLAGS -I/usr/include/eigen3 -std=c++17 -O2 "
        "-fPIC -shared -Wl,-Bsymbolic "
        f"-o {so} {source} "
        "-L/opt/ros/jazzy/lib -L/opt/ros/jazzy/lib/x86_64-linux-gnu "
        "-Wl,-rpath,/opt/ros/jazzy/lib -Wl,-rpath,/opt/ros/jazzy/lib/x86_64-linux-gnu "
        "-Wl,--no-as-needed -lmoveit_robot_trajectory -lmoveit_robot_state "
        "-lmoveit_robot_model -lmoveit_utils -lruckig -ldl"
    )
    compile_script = output / "compile.sh"
    compile_script.write_bytes(b"#!/usr/bin/env bash\nset -e\n" + command.encode("utf-8") + b"\n")
    returncode = run_wsl(f"bash {wsl_path(compile_script)}", output / "compile.log")
    (output / "compile_command.txt").write_text(command + "\n", encoding="utf-8")
    return returncode, so


def launch_rebuild(rebuild: Path, so: str) -> dict[str, object]:
    moveit = rebuild / "moveit"
    native = rebuild / "native"
    moveit.mkdir()
    native.mkdir()
    command = (
        "source /opt/ros/jazzy/setup.bash; "
        "source /mnt/d/robotfucker/install/setup.bash; "
        "export RCUTILS_COLORIZED_OUTPUT=0; "
        f"export LD_PRELOAD={so}; "
        f"export STAGE25R_NATIVE_DIR={wsl_path(native)}; "
        f"ros2 launch {wsl_path(LAUNCH)} "
        f"input_csv:={wsl_path(FROZEN / 'stage25_pre_timing_joint_trajectory.csv')} "
        f"source_reference_json:={wsl_path(FROZEN / 'stage25_path_reference.json')} "
        f"waypoints_csv:={wsl_path(WAYPOINTS)} "
        f"output_dir:={wsl_path(moveit)} "
        "velocity_scaling:=0.15 acceleration_scaling:=0.15 "
        "path_tolerance:=0.002 resample_dt:=0.01 min_angle_change:=0.0005 "
        "ruckig_mitigate_overshoot:=true ruckig_overshoot_threshold:=0.000001 "
        "stage25r_zero_target_acceleration:=true stage25r_historical_replay:=false"
    )
    launch_script = rebuild / "launch.sh"
    launch_script.write_bytes(b"#!/usr/bin/env bash\nset -e\n" + command.encode("utf-8") + b"\n")
    (rebuild / "launch_command.txt").write_text(command + "\n", encoding="utf-8")
    returncode = run_wsl(f"bash {wsl_path(launch_script)}", rebuild / "launch.log")
    summary_path = native / "stage25r_native_run_summary.json"
    summary: dict[str, object] = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    required = [
        native / "stage25r_ruckig_calls.jsonl",
        native / "stage25r_ruckig_call_resolutions.jsonl",
        native / "stage25r_native_samples.jsonl",
        moveit / "stage25_ruckig_trajectory.csv",
    ]
    artifacts_present = all(p.exists() and p.stat().st_size > 0 for p in required)
    return {
        "launcher_exit_code": returncode,
        "artifacts_present": artifacts_present,
        "summary": summary,
        "effective_completed": summary.get("status") == "completed" and artifacts_present,
    }


def main() -> int:
    if not FROZEN.exists() or not WAYPOINTS.exists():
        raise FileNotFoundError("formal Stage 2.5 frozen input or Stage 2.3A.4 waypoints missing")
    run_id = "stage25r_formal_native_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = RECOVERY_ROOT / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    before = frozen_manifest()
    write_json(run_root / "frozen_artifact_hashes_before.json", before)
    write_json(run_root / "runtime_fingerprint.json", runtime_fingerprint())
    write_json(
        run_root / "run_manifest.json",
        {
            "schema_version": "stage25r-run-manifest-v1",
            "run_id": run_id,
            "recovery_unit": "one_ruckig_calculate_call",
            "formal_source": str(FROZEN),
            "formal_input_csv": str(FROZEN / "stage25_pre_timing_joint_trajectory.csv"),
            "formal_frozen_trajectory_csv": str(FROZEN / "stage25_ruckig_trajectory.csv"),
            "path_tolerance": 0.002,
            "ruckig_overshoot_threshold": 0.000001,
            "stage25r_zero_target_acceleration": True,
            "stage25r_historical_replay": False,
            "moveit_version_target": "2.12.4",
            "ruckig_version_target": "0.9.2",
            "runtime_fingerprint": "runtime_fingerprint.json",
            "stage27_started": False,
            "stage26_started": False,
        },
    )
    rebuild_results: list[dict[str, object]] = []
    for index in range(1, 4):
        rebuild = run_root / f"rebuild_{index:02d}"
        (rebuild / "instrumentation").mkdir(parents=True)
        code, so = compile_interposer(rebuild / "instrumentation")
        result: dict[str, object] = {"rebuild": index, "compile_exit_code": code}
        if code == 0 and Path(so.replace("/mnt/d", "D:").replace("/", "\\")).exists():
            result.update(launch_rebuild(rebuild, so))
            so_win = Path(so.replace("/mnt/d", "D:").replace("/", "\\"))
            result["interposer_sha256"] = sha256(so_win)
        else:
            result["effective_completed"] = False
        rebuild_results.append(result)
        write_json(rebuild / "rebuild_result.json", result)
    after = frozen_manifest()
    write_json(run_root / "frozen_artifact_hashes_after.json", after)
    unchanged = before == after
    write_json(
        run_root / "rebuild_results.json",
        {"rebuilds": rebuild_results, "frozen_artifacts_unchanged": unchanged},
    )
    print(run_root)
    print(json.dumps({"frozen_artifacts_unchanged": unchanged, "rebuilds": rebuild_results}, indent=2))
    return 0 if unchanged and all(r.get("effective_completed") for r in rebuild_results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
