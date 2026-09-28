#!/usr/bin/env python3
"""Build the bridge in external scratch and run the targeted C1 R0 replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
SOURCE_BASE = "9adfe22d0fa53e3f15052659d5e08af06fbd8733"
BRANCH = "codex/fr5-p2b3-c1-lineage-order-20260928"
FAIRINO_SOURCE_COMMIT = "60755d44d521a5ad6bee8494cc19522f8801aa20"
FAIRINO_REMOTE_MARKER = "FAIR-INNOVATION/frcobot_ros2"
FAIRINO_URDF_SHA256 = "923a4d2f754162dadc9a6b0bcbe5b4caf9a32ccd31c479e72ed690febe211b4e"
FAIRINO_CONTROL_SHA256 = "9296111525fa892d63c700b37d58bd0e9b4a2f7d9713ff3d00bd95b56b5d08b5"
JOINTS = tuple(f"j{index}" for index in range(1, 7))


def wsl_path(path: Path) -> str:
    value = str(path.resolve()).replace("\\", "/")
    drive, tail = value.split(":", 1)
    return f"/mnt/{drive.lower()}/{tail.lstrip('/')}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_execution_tree() -> tuple[str, str]:
    def git(*args: str) -> str:
        result = subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True)
        return result.stdout.strip()

    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    parent = git("rev-parse", "HEAD^")
    dirty = git("status", "--porcelain")
    if branch != BRANCH or parent != SOURCE_BASE or dirty:
        raise RuntimeError(f"execution_tree_identity_mismatch:branch={branch}:head={head}:parent={parent}:dirty={bool(dirty)}")
    return branch, head


def check_fairino_source(source: Path) -> dict[str, str]:
    def git(*args: str) -> str:
        result = subprocess.run(["git", *args], cwd=source, check=True, capture_output=True, text=True)
        return result.stdout.strip()

    remote = git("remote", "get-url", "origin")
    commit = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain")
    if FAIRINO_REMOTE_MARKER not in remote or commit != FAIRINO_SOURCE_COMMIT or dirty:
        raise RuntimeError(f"FAIRINO_source_identity_mismatch:remote={remote}:commit={commit}:dirty={bool(dirty)}")
    urdf = source / "fairino_description/urdf/fairino5_v6.urdf"
    control = source / "fairino5_v6_moveit2_config/config/fairino5_v6_robot.ros2_control.xacro"
    if sha256(urdf) != FAIRINO_URDF_SHA256 or sha256(control) != FAIRINO_CONTROL_SHA256:
        raise RuntimeError("FAIRINO_source_model_asset_identity_mismatch")
    return {"remote": remote, "commit": commit, "working_tree_clean": "true",
            "urdf_sha256": FAIRINO_URDF_SHA256, "moveit_control_xacro_sha256": FAIRINO_CONTROL_SHA256}


def limits_from_urdf(path: Path) -> tuple[list[float], list[float]]:
    values: dict[str, tuple[float, float]] = {}
    for joint in ET.parse(path).getroot().findall("joint"):
        name, limit = joint.get("name", ""), joint.find("limit")
        if name in JOINTS and limit is not None:
            values[name] = (float(limit.attrib["lower"]), float(limit.attrib["upper"]))
    if set(values) != set(JOINTS):
        raise RuntimeError("authoritative_urdf_joint_limits_missing")
    return ([values[name][0] for name in JOINTS], [values[name][1] for name in JOINTS])


def run_wsl(script: str, log_path: Path, distro: str, timeout_s: int) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["wsl.exe", "-d", distro, "--", "bash", "-lc", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_s,
    )
    log_path.write_text((result.stdout or "") + "\n--- STDERR ---\n" + (result.stderr or ""), encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"WSL_execution_failed:{result.returncode}:see {log_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True, help="New external scratch output directory; must not exist")
    parser.add_argument("--fairino-source", type=Path, required=True, help="Clean FAIRINO ROS source checkout at the recorded release commit")
    parser.add_argument("--distro", default="Ubuntu-24.04-D")
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    args = parser.parse_args()

    output = args.output.resolve()
    fairino_source = args.fairino_source.resolve()
    if output.exists():
        raise RuntimeError(f"scratch_output_must_not_exist:{output}")
    if not (fairino_source / ".git").exists():
        raise FileNotFoundError(fairino_source / ".git")
    branch, head = check_execution_tree()
    fairino_identity = check_fairino_source(fairino_source)

    inputs = {
        "target_tcp_csv": ROOT / "outputs/internal_wiper_moveit_inputs/open_arch_tcp_poses_base_link.csv",
        "d39_observation_seed_csv": ROOT / "outputs/p2b2_inputs/stable_velocity_residual_update.csv",
        "reference_post_ruckig_csv": ROOT / "outputs/p2b2_inputs/p2b1_b0_reference/strict_replay/moveit_smoothed_joint_trajectory.csv",
        "derived_reference_urdf": ROOT / "outputs/p2b2_inputs/derived_reference_robot_model.urdf",
    }
    for input_path in inputs.values():
        if not input_path.is_file():
            raise FileNotFoundError(input_path)

    output.mkdir(parents=True)
    install = output / "bridge_install"
    build = output / "bridge_build"
    install_fairino = output / "fairino_install"
    fairino_build = output / "fairino_build"
    log = output / "strict_replay"
    lower, upper = limits_from_urdf(inputs["derived_reference_urdf"])
    path = lambda item: shlex.quote(wsl_path(item))

    fairino_script = "\n".join((
        "set -eo pipefail",
        "source /opt/ros/jazzy/setup.bash",
        f"colcon --log-base {path(output / 'fairino_colcon_log')} build --base-paths {path(fairino_source / 'fairino_description')} {path(fairino_source / 'fairino5_v6_moveit2_config')} --build-base {path(fairino_build)} --install-base {path(install_fairino)} --packages-select fairino_description fairino5_v6_moveit2_config --event-handlers console_direct+ --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3 -DPYTHON_EXECUTABLE=/usr/bin/python3 -DBUILD_TESTING=OFF",
    ))
    run_wsl(fairino_script, output / "fairino_colcon_build.log", args.distro, args.timeout_seconds)
    installed_urdf = install_fairino / "fairino_description/share/fairino_description/urdf/fairino5_v6.urdf"
    installed_control = install_fairino / "fairino5_v6_moveit2_config/share/fairino5_v6_moveit2_config/config/fairino5_v6_robot.ros2_control.xacro"
    if sha256(installed_urdf) != FAIRINO_URDF_SHA256 or sha256(installed_control) != FAIRINO_CONTROL_SHA256:
        raise RuntimeError("FAIRINO_installed_model_asset_identity_mismatch")

    build_script = "\n".join((
        "set -eo pipefail",
        "source /opt/ros/jazzy/setup.bash",
        f"source {path(install_fairino / 'setup.bash')}",
        f"colcon --log-base {path(output / 'bridge_colcon_log')} build --base-paths {path(ROOT / 'ros2_moveit_bridge')} --build-base {path(build)} --install-base {path(install)} --packages-select fr5_tunnel_moveit_bridge --event-handlers console_direct+",
    ))
    run_wsl(build_script, output / "colcon_build.log", args.distro, args.timeout_seconds)

    exports = {
        "REPO_ROOT": wsl_path(ROOT), "ROS_WS": wsl_path(install), "FINAL_OUT_DIR": wsl_path(log),
        "TCP_POSES_CSV": wsl_path(inputs["target_tcp_csv"]),
        "SEED_JOINT_CSV": wsl_path(inputs["d39_observation_seed_csv"]),
        "SAMPLES_PER_LOOP": "181", "OPEN_PATH": "true", "TCP_POINTS_TO_WALL": "true",
        "TOOL_TCP_XYZ": "0.000 0.000 0.150", "TOOL_TCP_RPY": "0 0 0",
        "TOOL_TCP_SOURCE": "assumed_150mm_placeholder", "TOOL_TCP_MEASURED_BY": "simulation",
        "TOOL_TCP_MEASURED_DATE": "not_applicable_offline", "TOOL_TCP_CALIBRATION_METHOD": "virtual_design_parameter",
        "STAND_OFF": "0.260", "WAYPOINT_STRIDE": "1", "PLANNING_MODE": "normal_dls_waypoints",
        "VALIDATION_STRIDE": "1", "VELOCITY_SCALING": "0.15", "ACCELERATION_SCALING": "0.15",
        "TIME_PARAMETERIZATION": "tcp_arclength", "TARGET_TCP_SPEED": "0.003", "ZERO_BOUNDARY_STATE": "true",
        "MAX_PATH_DEVIATION": "0.006", "MAX_NORMAL_ERROR_DEG": "10.0", "MAX_STANDOFF_FRACTION": "0.05",
        "MAX_SPEED_FLUCTUATION": "0.05", "VALIDATE_COLLISION": "true", "COLLISION_CHECK_STRIDE": "1",
        "COLLISION_SEGMENT_STRIDE": "1", "COLLISION_INTERPOLATION_STEP_DEG": "0.5",
        "INCLUDE_TUNNEL_FLOOR_COLLISION": "true", "TUNNEL_FLOOR_Z": "-0.20",
        "TUNNEL_WALL_THICKNESS": "0.04", "TUNNEL_Y_THICKNESS": "1.10",
        "INCLUDE_BOTTOM_CLOSURE_COLLISION": "false", "FAST_EXIT_AFTER_REPORTS": "true",
        "WRITE_FINAL_VISUALS": "false", "WRITE_FINAL_ANIMATION": "false", "SEGMENTED_EXECUTION": "true",
        "D41_DLS_LIMIT_AWARE": "true", "D41_DLS_LOWER_LIMITS": " ".join(map(str, lower)),
        "D41_DLS_UPPER_LIMITS": " ".join(map(str, upper)), "D41_DLS_LIMIT_BUFFER_RAD": "1.0e-4",
        "D41_DLS_CASE_ID": "P2B3_C1_R0", "D41_DLS_REPO_ROOT": wsl_path(ROOT),
        "D41_DLS_DIAGNOSTICS_CSV": wsl_path(log / "D41_dls_diagnostics.csv"),
        "P2B1_DLS_VARIANT": "B0", "P2B1_SOLVER_PRESSURE_CSV": wsl_path(output / "solver_pressure.csv"),
        "P2B1_FROZEN_D41_Q_CSV": wsl_path(inputs["reference_post_ruckig_csv"]),
        "P2B1_FD_CONSISTENCY_JSON": wsl_path(output / "process_jacobian_fd.json"),
        "P2B2_VARIANT": "R0", "P2B2_WARM_START_ROWS": "16", "P2B2_SECONDARY_OBJECTIVE": "none",
        "P2B3_C1_INITIALIZATION_ONLY": "true",
        "QUALITY_REPORT": wsl_path(log / "moveit_quality_report.csv"),
        "DYNAMICS_REPORT": wsl_path(log / "moveit_joint_dynamics_report.csv"),
        "COLLISION_REPORT": wsl_path(log / "moveit_collision_report.csv"),
        "FK_TRACE": wsl_path(log / "moveit_fk_tcp_trace.csv"),
        "TRAJECTORY_CSV": wsl_path(log / "moveit_smoothed_joint_trajectory.csv"),
        "SEGMENTED_TRAJECTORY_CSV": wsl_path(log / "moveit_executed_segmented_joint_trajectory.csv"),
        "WAYPOINT_TRAJECTORY_CSV": wsl_path(log / "moveit_waypoint_joint_trajectory.csv"),
        "STRICT_JSON": wsl_path(log / "audit_goal_requirements_strict.json"),
        "PRODUCTION_READINESS_JSON": wsl_path(log / "production_readiness_check.json"),
        "RUNTIME_LOG": wsl_path(log / "moveit_runtime.log"),
    }
    run_script = "\n".join((
        "set -eo pipefail",
        "source /opt/ros/jazzy/setup.bash",
        f"source {path(install_fairino / 'setup.bash')}",
        f"source {path(install / 'setup.bash')}",
        *(f"export {key}={shlex.quote(value)}" for key, value in exports.items()),
        f"bash {path(ROOT / 'scripts/run_moveit_strict_validation.sh')}",
    ))
    run_wsl(run_script, output / "strict_execution.log", args.distro, args.timeout_seconds)

    manifest = {
        "schema": "p2b3-c1-r0-execution-manifest-v1",
        "project": "FAIRINO_FR5",
        "stage": "P2-B3-C1",
        "branch": branch,
        "source_base_commit": SOURCE_BASE,
        "execution_code_commit": head,
        "execution_tree_clean_at_start": True,
        "runtime": "WSL2 Ubuntu-24.04-D / ROS Jazzy / MoveItPy / source-built bridge overlay",
        "scope": "181 point ON-state open-arch planned trajectory; no hardware execution",
        "solver": {"P2B2_VARIANT": "R0", "P2B1_DLS_VARIANT": "B0", "warm_start_rows": 16,
                   "secondary_objective": "none", "initialization_only": True},
        "collision_method": "adaptive_discrete_interpolation",
        "strict_self_ccd": "NOT_AVAILABLE",
        "hardware_validation": "NOT_RUN",
        "spray_command_state": "NOT_REPRESENTED",
        "fairino_description_source": fairino_identity,
        "ros_underlay": "/opt/ros/jazzy",
        "fairino_install": str(install_fairino),
        "overlay_install": str(install),
        "output_directory": str(output),
        "inputs": {name: {"path": str(file.relative_to(ROOT)), "sha256": sha256(file)} for name, file in inputs.items()},
        "ros_output_files": sorted(file.name for file in log.glob("*") if file.is_file()),
    }
    (output / "execution_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "R0_REPLAY_FINISHED", "output": str(output), "strict_report": str(log / "final_acceptance_summary.json")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
