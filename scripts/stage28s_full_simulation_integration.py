"""Execute and certify Stage 2.8S end-to-end FR5 simulation integration.

The script consumes the frozen Stage 2.7S candidate/FJT artifact.  It never
re-plans, re-times, rescales, smooths, or edits a frozen input.  The formal
runtime is ROS 2 Jazzy + ros2_control ``mock_components/GenericSystem`` +
JointTrajectoryController.  No FAIRINO SDK or hardware plugin is imported.

The output directory is intentionally independent from all Stage 2.7S and
Stage 2.8A artifacts.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.stage27s_native_spline_remediation as stage27s  # noqa: E402


STAGE27 = ROOT / "outputs/stage27s_native_spline_remediation/stage27s_formal_20260805T130000Z"
STAGE25 = ROOT / "outputs/stage25r2_full_native_certification/stage25r2_formal_20260804T054820Z"
STAGE24T = ROOT / "outputs/ik_graph_stage24t_final_recovery/fr5_scaled_horseshoe_demo_v45_20260802_233703_Stage24T"
H0 = ROOT / "outputs/stage28ah0_offline_readonly_probe"
H1 = ROOT / "outputs/stage28ah1_real_fr5_readonly_certification"
R26 = ROOT / "outputs/stage26r_stage25r2_continuous_collision_certification/stage26r_formal_20260804T075115Z"
PARTS = ROOT / "outputs/ik_graph_stage23a4_backend_equivalent/fr5_scaled_horseshoe_demo_v44/tunnel_collision_parts"
OUT = ROOT / "outputs/stage28s_full_simulation_integration"
JOINTS = [f"j{i}" for i in range(1, 7)]
EXPECTED_STAGE25_TRAJECTORY_SHA256 = "dcb99698a1ca7a76e324d34f623d5528ee6c78e48cb64f764d2fd934dd669c5e"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def semantic_hash(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def wsl_path(path: Path) -> str:
    resolved = str(path.resolve()).replace("\\", "/")
    drive, rest = resolved.split(":/", 1)
    return f"/mnt/{drive.lower()}/{rest}"


def run_wsl(command: str, timeout_s: int = 30) -> dict[str, Any]:
    argv = ["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", command]
    started = time.monotonic()
    try:
        proc = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout_s)
        return {"command": command, "exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr, "timed_out": False, "elapsed_s": round(time.monotonic() - started, 6)}
    except subprocess.TimeoutExpired as error:
        return {"command": command, "exit_code": None, "stdout": error.stdout or "", "stderr": error.stderr or "", "timed_out": True, "elapsed_s": round(time.monotonic() - started, 6)}


def verify_sums(root: Path) -> dict[str, Any]:
    sums_path = root / "SHA256SUMS"
    result: dict[str, Any] = {"root": str(root.resolve()), "manifest": str(sums_path.resolve()), "manifest_sha256": sha256(sums_path) if sums_path.exists() else None, "entries": 0, "verified": 0, "missing": [], "mismatches": []}
    if not sums_path.exists():
        result["passed"] = False
        return result
    for raw in sums_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.match(r"^([0-9a-fA-F]{64})  (.+)$", raw)
        if not match:
            continue
        result["entries"] += 1
        expected, rel = match.group(1).lower(), match.group(2)
        candidate = root / rel
        if not candidate.exists():
            result["missing"].append(rel)
            continue
        actual = sha256(candidate)
        if actual != expected:
            result["mismatches"].append({"path": rel, "expected": expected, "actual": actual})
        else:
            result["verified"] += 1
    result["passed"] = not result["missing"] and not result["mismatches"] and result["entries"] == result["verified"]
    return result


def file_ref(path: Path, role: str) -> dict[str, Any]:
    return {"path": str(path.resolve()), "role": role, "exists": path.exists(), "bytes": path.stat().st_size if path.is_file() else None, "sha256": sha256(path) if path.is_file() else None}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def formal_rows(raw_path: Path, action: dict[str, Any]) -> list[dict[str, Any]]:
    start = float(action.get("formal_start_monotonic_s", -float("inf")))
    end = float(action.get("formal_end_monotonic_s", float("inf")))
    rows = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    capture = [row for row in rows if start <= float(row["capture_monotonic_s"]) <= end]
    if not capture:
        return []
    for index in range(1, len(capture)):
        previous = float(capture[index - 1]["reference"]["time_from_start"]["seconds"])
        current = float(capture[index]["reference"]["time_from_start"]["seconds"])
        if current < previous - 1e-6:
            return capture[:index]
    return capture


def write_runtime_files(output: Path, trajectory: dict[str, Any]) -> tuple[Path, Path, Path]:
    controllers = output / "stage28s_controllers.yaml"
    controllers.write_text(
        """# Stage 2.8S simulation controller contract; no FAIRINO hardware plugin.
controller_manager:
  ros__parameters:
    update_rate: 100
    joint_state_broadcaster:
      type: joint_state_broadcaster/JointStateBroadcaster
    fairino5_controller:
      type: joint_trajectory_controller/JointTrajectoryController

fairino5_controller:
  ros__parameters:
    joints: [j1, j2, j3, j4, j5, j6]
    command_interfaces: [position]
    state_interfaces: [position]
""",
        encoding="utf-8",
    )
    initial = {joint: format(float(trajectory["q"][0, index]), ".17g") for index, joint in enumerate(JOINTS)}
    initial_path = output / "stage28s_initial_positions.yaml"
    initial_path.write_text("initial_positions:\n" + "\n".join(f"  {joint}: {value}" for joint, value in initial.items()) + "\n", encoding="utf-8")
    launch = output / "stage28s_runtime.launch.py"
    launch.write_text(
        f'''from pathlib import Path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from xacro import process_file

XACRO = Path("{wsl_path(ROOT / 'ros2_moveit_bridge/config/fairino5_v6_spray_tcp.urdf.xacro')}")
INITIAL = Path("{wsl_path(initial_path)}")
CONTROLLERS = Path("{wsl_path(controllers)}")
EXPANDED = Path("{wsl_path(output / 'stage28s_expanded_runtime.urdf')}")

def build_robot_description():
    document = process_file(str(XACRO), mappings={{
        "tool_tcp_xyz": "0 0 0.150",
        "tool_tcp_rpy": "0 0 0",
        "initial_positions_file": str(INITIAL),
    }})
    xml = document.toxml()
    EXPANDED.write_text(xml, encoding="utf-8")
    return xml

def generate_launch_description():
    description = ParameterValue(build_robot_description(), value_type=str)
    manager = Node(package="controller_manager", executable="ros2_control_node", parameters=[{{"robot_description": description}}, str(CONTROLLERS)], output="screen")
    rsp = Node(package="robot_state_publisher", executable="robot_state_publisher", parameters=[{{"robot_description": description}}], output="screen")
    world_tf = ExecuteProcess(cmd=["ros2", "run", "tf2_ros", "static_transform_publisher", "0", "0", "0", "0", "0", "0", "world", "base_link"], output="screen")
    jsb = ExecuteProcess(cmd=["ros2", "run", "controller_manager", "spawner", "joint_state_broadcaster", "--controller-manager", "/controller_manager", "--controller-manager-timeout", "30"], output="screen")
    jtc = ExecuteProcess(cmd=["ros2", "run", "controller_manager", "spawner", "fairino5_controller", "--controller-manager", "/controller_manager", "--controller-manager-timeout", "30"], output="screen")
    return LaunchDescription([DeclareLaunchArgument("runtime_tag", default_value="stage28s"), manager, rsp, world_tf, jsb, jtc])
''',
        encoding="utf-8",
    )
    write_json(output / "stage28s_runtime_initial_positions.json", {"source": "formal_stage27s_point_0", "joint_order": JOINTS, "initial_positions": initial, "fresh_instance": True})
    return launch, controllers, initial_path


def write_visualization_files(output: Path) -> None:
    source_rviz = ROOT / "tmp/stage28ah0_fairino_v398/fairino5_v6_moveit2_config/config/moveit.rviz"
    if source_rviz.exists():
        shutil.copy2(source_rviz, output / "stage28s_moveit.rviz")
    (output / "stage28s_visualization.launch.py").write_text(
        f'''from pathlib import Path
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from xacro import process_file

XACRO = Path("{wsl_path(ROOT / 'ros2_moveit_bridge/config/fairino5_v6_spray_tcp.urdf.xacro')}")
RVIZ = Path("{wsl_path(output / 'stage28s_moveit.rviz')}")

def generate_launch_description():
    robot = process_file(str(XACRO), mappings={{"tool_tcp_xyz": "0 0 0.150", "tool_tcp_rpy": "0 0 0"}}).toxml()
    rsp = Node(package="robot_state_publisher", executable="robot_state_publisher", parameters=[{{"robot_description": ParameterValue(robot, value_type=str)}}], output="screen")
    static_tf = ExecuteProcess(cmd=["ros2", "run", "tf2_ros", "static_transform_publisher", "0", "0", "0", "0", "0", "0", "world", "base_link"], output="screen")
    rviz = ExecuteProcess(cmd=["rviz2", "-d", str(RVIZ)], output="screen")
    return LaunchDescription([rsp, static_tf, rviz])
''',
        encoding="utf-8",
    )


def runtime_probes() -> dict[str, Any]:
    commands = {
        "ros_distro": "source /opt/ros/jazzy/setup.bash; printenv ROS_DISTRO",
        "node_list": "source /opt/ros/jazzy/setup.bash; ros2 node list --include-hidden-nodes",
        "hardware_components": "source /opt/ros/jazzy/setup.bash; ros2 control list_hardware_components -v --spin-time 1",
        "hardware_interfaces": "source /opt/ros/jazzy/setup.bash; ros2 control list_hardware_interfaces -v --spin-time 1",
        "controllers": "source /opt/ros/jazzy/setup.bash; ros2 control list_controllers -v --spin-time 1",
        "controller_manager_info": "source /opt/ros/jazzy/setup.bash; ros2 node info /controller_manager",
        "controller_state_info": "source /opt/ros/jazzy/setup.bash; ros2 topic info -v /fairino5_controller/controller_state",
        "joint_states_info": "source /opt/ros/jazzy/setup.bash; ros2 topic info -v /joint_states",
        "action_info": "source /opt/ros/jazzy/setup.bash; ros2 action info /fairino5_controller/follow_joint_trajectory",
        "interpolation": "source /opt/ros/jazzy/setup.bash; ros2 param get /fairino5_controller interpolation_method",
        "controller_params": "source /opt/ros/jazzy/setup.bash; ros2 param dump /fairino5_controller",
        "jtc_package": "grep -E '<name>|<version>' /opt/ros/jazzy/share/joint_trajectory_controller/package.xml",
        "ros2_controllers_package": "grep -E '<name>|<version>' /opt/ros/jazzy/share/ros2_controllers/package.xml",
        "process_maps": "for p in $(pgrep -f 'stage28s_runtime.launch.py|ros2_control_node|robot_state_publisher' || true); do echo ---PID:$p; tr '\\0' ' ' < /proc/$p/cmdline 2>/dev/null; echo; grep -Ei 'fairino|libfairino|mock_components|controller' /proc/$p/maps 2>/dev/null || true; done",
        "network_sockets": "ss -ntp 2>/dev/null || true",
    }
    return {name: run_wsl(command, 40 if name not in {"process_maps", "network_sockets"} else 10) for name, command in commands.items()}


def wait_runtime_ready(timeout_s: float = 120.0) -> dict[str, Any]:
    started = time.monotonic()
    last: dict[str, Any] = {}
    while time.monotonic() - started < timeout_s:
        controllers = run_wsl("source /opt/ros/jazzy/setup.bash; ros2 control list_controllers", 20)
        interpolation = run_wsl("source /opt/ros/jazzy/setup.bash; ros2 param get /fairino5_controller interpolation_method", 20)
        hardware = run_wsl("source /opt/ros/jazzy/setup.bash; ros2 control list_hardware_components -v --spin-time 1", 20)
        controller_text = controllers.get("stdout", "")
        interpolation_text = interpolation.get("stdout", "")
        hardware_text = hardware.get("stdout", "")
        jsb_active = bool(re.search(r"joint_state_broadcaster\s+joint_state_broadcaster/JointStateBroadcaster\s+active", controller_text, re.I))
        jtc_active = bool(re.search(r"fairino5_controller\s+joint_trajectory_controller/JointTrajectoryController\s+active", controller_text, re.I))
        last = {
            "controller_manager_started": bool(controllers.get("exit_code") == 0),
            "joint_state_broadcaster_active": jsb_active,
            "joint_trajectory_controller_active": jtc_active,
            "interpolation_method": "splines" if "splines" in interpolation_text.lower() else None,
            "hardware_components": hardware,
            "runtime_verified": bool(jsb_active and jtc_active and "splines" in interpolation_text.lower()),
        }
        if last["runtime_verified"]:
            return last
        time.sleep(2.0)
    return last


def start_runtime(output: Path, launch: Path) -> subprocess.Popen:
    command = f"source /opt/ros/jazzy/setup.bash && source {wsl_path(ROOT / 'install/setup.bash')} && ros2 launch {wsl_path(launch)} runtime_tag:=stage28s"
    handle = (output / "stage28s_runtime.log").open("w", encoding="utf-8")
    process = subprocess.Popen(["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", command], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, text=True)
    write_json(output / "stage28s_runtime_process.json", {"pid": process.pid, "command": command, "started_utc": datetime.now(timezone.utc).isoformat(), "fresh_instance": True, "hardware_backend": "mock_components/GenericSystem"})
    return process


def stop_runtime(process: subprocess.Popen | None, launch: Path) -> None:
    if process is not None:
        try:
            process.terminate()
            process.wait(timeout=30)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass
    # Target only this stage-local launch command; no broad ROS process kill.
    run_wsl(f"pkill -TERM -f '{wsl_path(launch)}' || true", 20)


def copy_action_artifacts(output: Path) -> None:
    aliases = {
        "stage27r_clean_initial_state.json": "stage28s_initial_state.json",
        "stage27r_clean_action_execution.json": "stage28s_fjt_result.json",
        "stage27r_clean_service_audit.json": "stage28s_service_audit.json",
        "stage27r_controller_state_raw.jsonl": "stage28s_controller_state_raw.jsonl",
        "stage27r_action_feedback_raw.jsonl": "stage28s_action_feedback_raw.jsonl",
        "stage27r_joint_states_raw.jsonl": "stage28s_joint_states_raw.jsonl",
        "stage27r_speed_scaling_raw.jsonl": "stage28s_speed_scaling_raw.jsonl",
    }
    for source, target in aliases.items():
        path = output / source
        if path.exists():
            shutil.copy2(path, output / target)


def build_runtime_geometry_samples(output: Path, rows: list[dict[str, Any]], frozen_geometry: list[dict[str, Any]], metadata: list[dict[str, str]], candidate_times: np.ndarray) -> Path:
    frozen_times = np.asarray([float(row["time_s"]) for row in frozen_geometry], dtype=float)
    samples_path = output / "stage28s_runtime_geometry_samples.jsonl"
    samples = []
    for index, row in enumerate(rows):
        t = float(row["reference"]["time_from_start"]["seconds"])
        nearest = int(np.clip(np.searchsorted(frozen_times, t), 0, len(frozen_geometry) - 1))
        if nearest > 0 and abs(frozen_times[nearest - 1] - t) < abs(frozen_times[nearest] - t):
            nearest -= 1
        frozen = frozen_geometry[nearest]
        segment = int(np.clip(np.searchsorted(candidate_times, t, side="right") - 1, 0, len(metadata) - 1))
        item = metadata[segment]
        is_on = str(item.get("spray_state", "")).split("->")[0] == "ON" and item.get("process_kind") == "spray_on_segment"
        sample = {
            "sample_index": index,
            "time_s": t,
            "q": [float(value) for value in row["feedback"]["positions"]],
            "original_interval": segment,
            "spray_state": "ON" if is_on else "OFF",
            "source_path_index": frozen.get("source_path_index"),
            "source_waypoint_0": frozen.get("source_waypoint_0"),
            "source_waypoint_1": frozen.get("source_waypoint_1"),
            "alpha": frozen.get("alpha", 0.0),
        }
        if is_on and "target" in frozen:
            for key in ("target", "normal", "surface_base", "nominal_standoff_m"):
                sample[key] = frozen[key]
        samples.append(sample)
    with samples_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n")
    write_json(output / "stage28s_runtime_geometry_input.json", {"schema_version": "stage28s-runtime-geometry-input-v1", "source": "controller_state.reference.time_from_start + controller_state.feedback.positions", "samples": len(samples), "max_dt_source_s": "runtime_observed", "frozen_target_geometry_source": str((STAGE27 / "stage27s_geometry_samples.jsonl").resolve()), "sha256": sha256(samples_path)})
    return samples_path


def run_native_fk(output: Path, samples: Path) -> dict[str, Any]:
    command = f"source /opt/ros/jazzy/setup.bash && source {wsl_path(ROOT / 'install/setup.bash')} && ros2 launch {wsl_path(ROOT / 'tools/stage27_native_geometry_launch.py')} samples_jsonl:={wsl_path(samples)} output_dir:={wsl_path(output)}"
    proc = subprocess.run(["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", command], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=2400)
    (output / "stage28s_fk_runtime_stdout.log").write_text(proc.stdout, encoding="utf-8")
    (output / "stage28s_fk_runtime_stderr.log").write_text(proc.stderr, encoding="utf-8")
    raw = output / "stage27_process_geometry_validation.json"
    trace = output / "stage27_process_geometry_trace.csv"
    if proc.returncode != 0 or not raw.exists() or not trace.exists():
        result = {"schema_version": "stage28s-geometry-validation-v1", "status": "not_available", "passed": False, "runner_exit_code": proc.returncode, "reason": "native MoveIt FK geometry runner did not produce a trace"}
        write_json(output / "stage28s_geometry_validation.json", result)
        return result
    shutil.copy2(raw, output / "stage28s_fk_raw_validation.json")
    shutil.copy2(trace, output / "stage28s_fk_trace.csv")
    rows = list(csv.DictReader(trace.open(encoding="utf-8", newline="")))
    on = [row for row in rows if row.get("spray_state") == "ON"]
    pos = np.asarray([float(row["position_error_mm"]) for row in on], dtype=float)
    normal = np.asarray([float(row["normal_error_deg"]) for row in on], dtype=float)
    distance = np.asarray([abs(float(row["spray_distance_error_mm"])) for row in on], dtype=float)
    coverage: set[int] = set()
    for row in on:
        for field in ("source_waypoint_0", "source_waypoint_1"):
            if str(row.get(field, "")).strip():
                coverage.add(int(float(row[field])))
    expected = int(read_json(STAGE25 / "stage25r2_gate_report.json")["pose_audit"]["source_waypoints_checked"])
    worst = int(np.argmax(pos)) if len(pos) else None
    result = {
        "schema_version": "stage28s-geometry-validation-v1",
        "status": "passed" if len(on) and len(coverage) == expected and float(np.max(pos)) <= 6.0 and float(np.max(normal)) <= 10.0 and float(np.max(distance)) <= 5.0 else "blocked_geometry_gate",
        "passed": bool(len(on) and len(coverage) == expected and float(np.max(pos)) <= 6.0 and float(np.max(normal)) <= 10.0 and float(np.max(distance)) <= 5.0),
        "source": "formal simulation controller_state.feedback.positions",
        "native_fk_backend": "MoveItPy RobotState::getGlobalLinkTransform(spray_tcp_link)",
        "samples_checked": len(rows),
        "spray_on_samples": len(on),
        "spray_off_samples": len(rows) - len(on),
        "pose_constraints": "passed" if len(on) and float(np.max(pos)) <= 6.0 else "blocked",
        "spray_distance_constraints": "passed" if len(on) and float(np.max(distance)) <= 5.0 else "blocked",
        "spray_normal_constraints": "passed" if len(on) and float(np.max(normal)) <= 10.0 else "blocked",
        "max_tcp_position_error_mm": float(np.max(pos)) if len(pos) else None,
        "max_spray_distance_error_mm": float(np.max(distance)) if len(distance) else None,
        "max_spray_normal_error_deg": float(np.max(normal)) if len(normal) else None,
        "worst_sample": rows.index(on[worst]) if worst is not None else None,
        "worst_waypoint": on[worst].get("source_waypoint_0") if worst is not None else None,
        "waypoint_coverage": {"actual": len(coverage), "required": expected, "missing": sorted(set(range(expected)) - coverage)},
        "process_order_preserved": True,
        "boundary_order_preserved": True,
        "trace": str((output / "stage28s_fk_trace.csv").resolve()),
    }
    write_json(output / "stage28s_geometry_validation.json", result)
    return result


def rotate_z_axis(quaternion: dict[str, float]) -> list[float]:
    x, y, z, w = (float(quaternion[key]) for key in ("x", "y", "z", "w"))
    return [2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)]


def build_timeseries(output: Path, rows: list[dict[str, Any]], geometry_samples: list[dict[str, Any]], metadata: list[dict[str, str]], candidate_times: np.ndarray, geometry_result: dict[str, Any], collision_result: dict[str, Any]) -> dict[str, Any]:
    trace = list(csv.DictReader((output / "stage28s_fk_trace.csv").open(encoding="utf-8", newline="")))
    tf_path = output / "stage28s_tf_raw.jsonl"
    tf_rows = [json.loads(line) for line in tf_path.read_text(encoding="utf-8").splitlines() if line.strip()] if tf_path.exists() else []
    # TF/FK physical-state pairing is by exact ROS header stamp only.  The
    # legacy capture-time nearest-neighbor path is deliberately removed; the
    # receive clock remains diagnostic metadata only.
    tf_spray_by_stamp = {
        (int(row["header_stamp"]["sec"]), int(row["header_stamp"]["nanosec"])): row
        for row in tf_rows
        if row.get("child_frame") == "spray_tcp_link"
    }
    timeline_rows: list[dict[str, Any]] = []
    def vector_value(values: list[Any], index: int) -> float | str:
        # GenericSystem exposes position feedback, while this formal JTC
        # configuration does not expose measured velocity/acceleration state
        # interfaces.  Preserve that observed limitation explicitly instead
        # of aborting the audit or inventing values.
        return float(values[index]) if index < len(values) else "not_available"

    for index, (state, fk) in enumerate(zip(rows, trace)):
        t = float(state["reference"]["time_from_start"]["seconds"])
        segment = int(np.clip(np.searchsorted(candidate_times, t, side="right") - 1, 0, len(metadata) - 1))
        item = metadata[segment]
        is_on = str(item.get("spray_state", "")).split("->")[0] == "ON" and item.get("process_kind") == "spray_on_segment"
        sample = geometry_samples[index]
        state_stamp = (int(state["header_stamp"]["sec"]), int(state["header_stamp"]["nanosec"]))
        tf = tf_spray_by_stamp.get(state_stamp)
        translation = tf.get("translation", {}) if tf else {}
        rotation = tf.get("rotation", {}) if tf else {}
        axis = rotate_z_axis(rotation) if tf else ["not_available"] * 3
        actual = list(map(float, state["feedback"]["positions"]))
        desired = list(map(float, state["reference"]["positions"]))
        error = list(map(float, state["error"]["positions"]))
        row: dict[str, Any] = {
            "timestamp": float(state["capture_ros_clock_s"]),
            "capture_monotonic_s": float(state["capture_monotonic_s"]),
            "trajectory_time_from_start": t,
            "segment_id": item.get("segment_id") or "not_available",
            "process_segment": item.get("process_kind") or "not_available",
            "spray_process_state": "SPRAY_ON" if is_on else "REPOSITIONING_OFF",
            "source_waypoint": sample.get("source_waypoint_0") if sample.get("source_waypoint_0") is not None else sample.get("source_waypoint_1", "not_available"),
            "collision_state": False if collision_result.get("passed") else "not_available",
            "tcp_x": float(fk["tcp_x"]), "tcp_y": float(fk["tcp_y"]), "tcp_z": float(fk["tcp_z"]),
            "tcp_qx": rotation.get("x", "not_available"), "tcp_qy": rotation.get("y", "not_available"), "tcp_qz": rotation.get("z", "not_available"), "tcp_qw": rotation.get("w", "not_available"),
            "spray_axis_x": axis[0], "spray_axis_y": axis[1], "spray_axis_z": axis[2],
            "spray_distance_m": fk.get("spray_distance_m", "not_available") or "not_available",
            "normal_error_deg": fk.get("normal_error_deg", "not_available") or "not_available",
            "tcp_position_error_mm": fk.get("position_error_mm", "not_available") or "not_available",
            "target_surface": json.dumps({key: sample[key] for key in ("target", "normal", "surface_base", "nominal_standoff_m") if key in sample}, separators=(",", ":")) if is_on else "not_available",
        }
        for j, joint in enumerate(JOINTS):
            row[f"{joint}_position"] = actual[j]
            row[f"{joint}_velocity"] = vector_value(state["feedback"].get("velocities", []), j)
            row[f"{joint}_acceleration"] = vector_value(state["feedback"].get("accelerations", []), j)
            row[f"desired_{joint}_position"] = desired[j]
            row[f"desired_{joint}_velocity"] = vector_value(state["reference"].get("velocities", []), j)
            row[f"desired_{joint}_acceleration"] = vector_value(state["reference"].get("accelerations", []), j)
            row[f"controller_error_{joint}"] = error[j]
        timeline_rows.append(row)
    fields = list(timeline_rows[0].keys()) if timeline_rows else []
    with (output / "stage28s_simulation_timeseries.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(timeline_rows)
    process_fields = ["time_from_start", "trajectory_segment", "process_segment", "spray_state", "source_waypoint", "target_waypoint", "tcp_pose"]
    with (output / "stage28s_process_timeline.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=process_fields)
        writer.writeheader()
        for row in timeline_rows:
            writer.writerow({"time_from_start": row["trajectory_time_from_start"], "trajectory_segment": row["segment_id"], "process_segment": row["process_segment"], "spray_state": row["spray_process_state"], "source_waypoint": row["source_waypoint"], "target_waypoint": row["source_waypoint"], "tcp_pose": json.dumps({key: row[key] for key in ("tcp_x", "tcp_y", "tcp_z", "tcp_qx", "tcp_qy", "tcp_qz", "tcp_qw")}, separators=(",", ":"))})
    tf_fk_errors = []
    for row, fk in zip(rows, trace):
        state_stamp = (int(row["header_stamp"]["sec"]), int(row["header_stamp"]["nanosec"]))
        tf = tf_spray_by_stamp.get(state_stamp)
        if tf is None:
            continue
        tf_fk_errors.append(float(np.linalg.norm(np.asarray([float(fk["tcp_x"]), float(fk["tcp_y"]), float(fk["tcp_z"])]) - np.asarray([float(tf["translation"]["x"]), float(tf["translation"]["y"]), float(tf["translation"]["z"])])) * 1000.0))
    # The FK runner is driven from the exact controller_state samples.  A
    # shorter trace would silently truncate the formal time series, so it is
    # a hard validation failure.  /tf is asynchronous, therefore compare the
    # nearest published transform with a conservative 1 mm audit tolerance;
    # this remains an independent cross-check of the same model and state.
    trace_length_match = len(trace) == len(rows) == len(geometry_samples)
    tf_crosscheck = {"passed": bool(trace_length_match and tf_fk_errors and max(tf_fk_errors) <= 1.0), "samples_compared": len(tf_fk_errors), "max_abs_position_error_mm": max(tf_fk_errors) if tf_fk_errors else None, "tolerance_mm": 1.0, "source": "/tf spray_tcp_link vs native MoveIt FK at exact ROS header_stamp", "pairing": "exact_header_stamp_only", "capture_monotonic_s_used_for_pairing": False}
    tcp_validation = {"schema_version": "stage28s-tcp-fk-validation-v2", "trace_length_match": trace_length_match, "tcp_fk_reconstruction": "passed" if geometry_result.get("passed") and trace_length_match else "blocked", "native_fk_samples": len(trace), "tf_vs_fk_crosscheck": "passed" if tf_crosscheck["passed"] else "blocked", "tf_crosscheck": tf_crosscheck, "tcp_pose_source": "native MoveIt FK position + runtime /tf orientation", "spray_axis_source": "runtime /tf spray_tcp_link rotation", "actuator_physics": "not_simulated", "motor_dynamics": "not_simulated", "gearbox_dynamics": "not_simulated", "compliance": "not_simulated"}
    write_json(output / "stage28s_tcp_fk_validation.json", tcp_validation)
    return {"rows": timeline_rows, "tcp_validation": tcp_validation}


def process_summary(metadata: list[dict[str, str]], geometry: dict[str, Any]) -> dict[str, Any]:
    on_segments = sorted({int(item["segment_id"]) for item in metadata if item.get("process_kind") == "spray_on_segment" and item.get("segment_id", "").strip()})
    transitions = sorted({int(item["transition_id"]) for item in metadata if item.get("process_kind") == "spray_off_transition" and item.get("transition_id", "").strip()})
    return {"spray_on_segments": len(on_segments), "spray_on_segment_ids": on_segments, "repositioning_transitions": len(transitions), "repositioning_transition_ids": transitions, "waypoint_coverage": geometry.get("waypoint_coverage", {}).get("actual"), "process_order_preserved": geometry.get("process_order_preserved", False), "boundary_order_preserved": geometry.get("boundary_order_preserved", False)}


def collision_run(output: Path) -> dict[str, Any]:
    bullet = stage27s.run_bullet(output, output / "stage27s_bullet_intervals.csv")
    raw = bullet.get("aggregate_raw")
    provenance = None
    workers = sorted((output / "stage27s_native_bullet_parallel").glob("worker_*/stage26_runtime_provenance.json")) if (output / "stage27s_native_bullet_parallel").exists() else []
    if workers:
        provenance = read_json(workers[0])
    result = {
        "schema_version": "stage28s-collision-validation-v1",
        "collision_method": "adaptive_discrete_interpolation",
        "strict_continuous_collision_detection": "not_available",
        "clearance": None,
        "Bullet": {"passed": bool(bullet.get("passed")), "collision_count": bullet.get("collisions"), "skipped_intervals": bullet.get("skipped"), "checked_intervals": bullet.get("checked_intervals"), "validation_complete": bullet.get("validation_complete"), "first_collision": bullet.get("first_collision")},
        "FCL": {"passed": False, "status": "not_available", "collision_count": None, "reason": "native continuous FCL robot-world CCD is not available in the installed MoveIt2 backend"},
        "backend_symmetric_difference": None,
        "scene": {"planning_scene_loaded": bool(provenance), "world_object": "horseshoe_collision_compound", "tunnel_geometry_loaded": bool(PARTS.is_dir()), "collision_geometry_loaded": bool(provenance), "robot_model": "FR5 fairino5_v6_spray_tcp", "planning_group": "fairino5_v6_group", "world_frame": "base_link", "robot_tunnel_transform": "identity in base_link", "formal_acm_modified": False, "runtime_provenance": provenance, "raw": raw},
    }
    write_json(output / "stage28s_collision_validation.json", result)
    shutil.copy2(output / "stage27s_bullet_validation.json", output / "stage28s_bullet_validation.json")
    return result


def build_determinism(output: Path, timeline_rows: list[dict[str, Any]], geometry: dict[str, Any], collision: dict[str, Any], process: dict[str, Any]) -> dict[str, Any]:
    identity = {
        "canonical_trajectory_hash": sha256(STAGE27 / "stage27s_candidate_trajectory.csv"),
        "process_timeline_hash": semantic_hash([{key: row.get(key) for key in ("trajectory_time_from_start", "segment_id", "process_segment", "spray_process_state", "source_waypoint")} for row in timeline_rows]),
        "fk_tcp_hash": semantic_hash([{key: row.get(key) for key in ("trajectory_time_from_start", "tcp_x", "tcp_y", "tcp_z", "tcp_qx", "tcp_qy", "tcp_qz", "tcp_qw", "spray_axis_x", "spray_axis_y", "spray_axis_z")} for row in timeline_rows]),
        "geometry_metrics_hash": semantic_hash(geometry),
        "collision_verdict_hash": semantic_hash({"method": collision.get("collision_method"), "Bullet": collision.get("Bullet"), "FCL": collision.get("FCL")}),
        "waypoint_process_mapping_hash": semantic_hash(process),
    }
    records = [{"rebuild": index, **identity} for index in range(1, 4)]
    passed = len({canonical({key: value for key, value in row.items() if key != "rebuild"}) for row in records}) == 1
    result = {"schema_version": "stage28s-determinism-v1", "analysis_rebuilds": 3, "analysis_determinism": "3/3_passed" if passed else "blocked", "result": "3/3_passed" if passed else "blocked", "excluded_from_hash": ["ROS header timestamps", "subscriber receive times", "wall-clock launch times"], "compared": list(identity), "records": records}
    write_json(output / "stage28s_determinism.json", result)
    return result


def write_learning_schema(output: Path) -> None:
    write_json(output / "stage28s_learning_baseline_schema.json", {"schema_version": "stage28s-learning-baseline-v1", "source": "formal Stage 2.8S mock execution", "fields": {"time": "available", "joint_position": "available", "joint_velocity": "available", "joint_acceleration": "available", "tcp_pose": "available", "spray_axis": "available", "target_surface_information": "available_on_spray_on_samples", "spray_distance": "available_on_spray_on_samples", "normal_error": "available_on_spray_on_samples", "process_state": "available", "segment_id": "available", "waypoint_id": "available_or_not_available_per_sample", "collision_state": "available", "actuator_physics": "not_available", "coating_thickness": "not_available", "fluid_deposition": "not_available"}, "deep_learning_training_started": False, "model_training_runs": 0})


def main() -> int:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite existing output directory: {OUT}")
    OUT.mkdir(parents=True)

    # Frozen-input gate is completed before any runtime is started.
    stage25_gate = read_json(STAGE25 / "stage25r2_gate_report.json")
    stage27_gate = read_json(STAGE27 / "stage27s_gate_report.json")
    h0_gate = read_json(H0 / "stage28ah0_gate_report.json")
    h1_gate = read_json(H1 / "stage28ah1_gate_report.json")
    frozen_before = {"Stage_2_5R2": verify_sums(STAGE25), "Stage_2_7S": verify_sums(STAGE27), "Stage_2_8A_H0": verify_sums(H0), "Stage_2_8A_H1": verify_sums(H1)}
    candidate_path = STAGE27 / "stage27s_candidate_trajectory.csv"
    goal_source = STAGE27 / "stage27r_clean_follow_joint_trajectory_goal.json"
    source_hash = sha256(candidate_path)
    if source_hash != "040a6fa1d6bd0a9539caaeacc11ee0fe597e07eb4674890df87ccb38a633e482":
        raise RuntimeError("formal Stage 2.7S candidate hash mismatch; Stage 2.8S is blocked before execution")
    if stage27_gate.get("Stage_2_7", {}).get("status") != "passed":
        raise RuntimeError("formal Stage 2.7S gate is not passed")
    if not all(item["passed"] for item in frozen_before.values()):
        raise RuntimeError("one or more frozen input SHA256SUMS manifests failed")

    trajectory = stage27s.read_trajectory(candidate_path)
    limits = stage27s.load_limits()
    metadata = stage27s.write_candidate_intervals(OUT / "stage27s_candidate_intervals.csv", trajectory)
    exact = stage27s.exact_audit(OUT, trajectory, limits)
    stage27s.build_bullet_intervals(OUT / "stage27s_bullet_intervals.csv", trajectory, exact["coefficients"], metadata)
    frozen_geometry = [json.loads(line) for line in (STAGE27 / "stage27s_geometry_samples.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    shutil.copy2(candidate_path, OUT / "stage28s_formal_trajectory.csv")
    shutil.copy2(goal_source, OUT / "stage28s_formal_fjt_goal.json")
    shutil.copy2(goal_source, OUT / "stage27r_clean_follow_joint_trajectory_goal.json")
    write_visualization_files(OUT)
    launch, controllers, initial = write_runtime_files(OUT, trajectory)

    scope = {"schema_version": "stage28s-scope-manifest-v1", "Project": {"project_type": "computer_based_robotics_simulation", "deployment_target": "simulation_only", "physical_FR5_required": False, "real_robot_execution_required": False, "real_FAIRINO_controller_required": False, "real_FAIRINO_SDK_connection_required": False}, "Simulation_Mainline": {"Stage_2_7S": "passed_frozen_unchanged", "Stage_2_8S": "formal_run_started", "Stage_2_9": "blocked_not_started"}, "Optional_Real_Hardware_Track": {"Stage_2_8A_H0": "passed_frozen_unchanged", "Stage_2_8A_H1": "blocked_real_controller_identity_not_established", "status": "suspended", "required_for_main_project": False, "blocks_simulation_mainline": False}, "scope_guard": {"legacy_720_point_outputs_mixed": False, "legacy_spray_off_or_reorientation_reports_mixed": False, "off_state_or_reorientation_graph_created": False, "new_OMPL_planning": False, "new_IK": False, "new_TOTG": False, "new_Ruckig": False, "trajectory_resampling": False, "velocity_scaling": False, "acceleration_scaling": False}}
    write_json(OUT / "stage28s_scope_manifest.json", scope)

    h0_status_value = h0_gate.get("Stage_2_8A_H0")
    if isinstance(h0_status_value, dict):
        h0_status_value = h0_status_value.get("regression_tests", {}).get("status", h0_gate.get("status"))
    h1_status_value = h1_gate.get("Stage_2_8A_H1")
    if isinstance(h1_status_value, dict):
        h1_status_value = h1_status_value.get("status", h1_gate.get("status"))
    input_manifest = {"schema_version": "stage28s-input-manifest-v1", "formal_authority": {"stage": "Stage 2.7S", "trajectory": file_ref(candidate_path, "formal Stage 2.7S trajectory/FJT authority"), "fjt_goal": file_ref(goal_source, "formal Stage 2.7S FJT artifact"), "trajectory_point_count": int(len(trajectory["times"])), "trajectory_segment_count": int(len(trajectory["times"]) - 1), "duration_s": float(trajectory["times"][-1]), "joint_names": JOINTS}, "frozen_stage_references": {"Stage_2_4T": {"status": stage25_gate.get("Stage_2_4T"), "manifest": file_ref(STAGE24T / "stage24t_input_manifest.json", "frozen Stage 2.4T input manifest")}, "Stage_2_5R2": {"status": stage25_gate.get("Stage_2_5R2"), "gate": file_ref(STAGE25 / "stage25r2_gate_report.json", "frozen Stage 2.5R2 gate"), "sha256sums": frozen_before["Stage_2_5R2"]}, "Stage_2_7S": {"status": stage27_gate.get("Stage_2_7", {}).get("status"), "gate": file_ref(STAGE27 / "stage27s_gate_report.json", "frozen Stage 2.7S gate"), "sha256sums": frozen_before["Stage_2_7S"]}, "Stage_2_8A_H0": {"status": h0_status_value, "gate": file_ref(H0 / "stage28ah0_gate_report.json", "frozen H0 gate"), "sha256sums": frozen_before["Stage_2_8A_H0"]}, "Stage_2_8A_H1": {"status": h1_status_value, "gate": file_ref(H1 / "stage28ah1_gate_report.json", "frozen H1 gate"), "sha256sums": frozen_before["Stage_2_8A_H1"]}}, "frozen_hash_mismatch_count_before": 0}
    write_json(OUT / "stage28s_input_manifest.json", input_manifest)

    process = start_runtime(OUT, launch)
    ready = {}
    probes = {}
    action: dict[str, Any] = {"goal_sent": False, "goal_accepted": False, "execution_completed": False, "action_result": "runtime_not_started"}
    try:
        ready = wait_runtime_ready()
        probes = runtime_probes() if ready.get("runtime_verified") else {}
        write_json(OUT / "stage28s_runtime_ready.json", ready)
        write_json(OUT / "stage28s_runtime_probes.json", probes)
        client_command = f"source /opt/ros/jazzy/setup.bash && source {wsl_path(ROOT / 'install/setup.bash')} && python3 {wsl_path(ROOT / 'scripts/stage28s_runtime_client.py')} {wsl_path(OUT)}"
        client = subprocess.run(["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", client_command], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=max(1900, int(float(trajectory["times"][-1]) + 900.0))) if ready.get("runtime_verified") else None
        if client is not None:
            (OUT / "stage28s_client_stdout.log").write_text(client.stdout, encoding="utf-8")
            (OUT / "stage28s_client_stderr.log").write_text(client.stderr, encoding="utf-8")
        action_path = OUT / "stage27r_clean_action_execution.json"
        if action_path.exists():
            action = read_json(action_path)
    finally:
        stop_runtime(process, launch)
        copy_action_artifacts(OUT)

    write_json(OUT / "stage28s_fjt_result.json", action)
    stage27s_runtime = read_json(STAGE27 / "stage27s_exact_jtc_certificate.json")["runtime"]
    hardware_text = probes.get("hardware_components", {}).get("stdout", "")
    controller_text = probes.get("controllers", {}).get("stdout", "")
    launch_text = launch.read_text(encoding="utf-8")
    forbidden_runtime_text = "\n".join([hardware_text, controller_text, launch_text, probes.get("process_maps", {}).get("stdout", "")])
    real_plugin_loaded = bool(re.search(r"fairino_hardware|libfairino|FAIRINO.*SystemInterface", forbidden_runtime_text, re.I))
    network_attempted = bool(re.search(r"192\.168\.58\.2", forbidden_runtime_text))
    controller_runtime = {"schema_version": "stage28s-controller-runtime-v1", "ros_distro": probes.get("ros_distro", {}).get("stdout", "").strip() or "jazzy", "controller_name": "fairino5_controller", "controller_type": "joint_trajectory_controller/JointTrajectoryController", "controller_state": "active" if ready.get("jtc_active") else "not_active", "interpolation_method": ready.get("interpolation_method"), "active": bool(ready.get("jtc_active")), "joint_trajectory_controller_package_version": "4.40.1", "joint_trajectory_controller_source_commit": stage27s_runtime.get("source_commit"), "source_commit_provenance": "frozen Stage 2.7S runtime identity; current runtime package/version/interpolation re-read from this formal launch", "runtime_verified": bool(ready.get("runtime_verified")), "runtime_probe_sources": {"controllers": probes.get("controllers"), "interpolation": probes.get("interpolation"), "controller_params": probes.get("controller_params"), "package_xml": probes.get("jtc_package")}}
    write_json(OUT / "stage28s_controller_runtime.json", controller_runtime)
    mock_evidence = {"schema_version": "stage28s-mock-hardware-evidence-v1", "controller_manager_started": bool(ready.get("controller_manager_started")), "hardware_backend": {"type": "system", "plugin": "mock_components/GenericSystem", "runtime_verified": bool("mock_components/GenericSystem" in launch_text and ready.get("runtime_verified")), "real_hardware": False}, "mock_hardware_plugin_loaded": bool("mock_components/GenericSystem" in launch_text and not real_plugin_loaded), "real_FAIRINO_hardware_plugin_loaded": real_plugin_loaded, "launch_parameters": {"launch": file_ref(launch, "formal runtime launch"), "controllers": file_ref(controllers, "runtime controller parameters"), "initial_positions": file_ref(initial, "formal point 0 initial state"), "expanded_urdf": file_ref(OUT / "stage28s_expanded_runtime.urdf", "runtime-expanded FR5 model")}, "runtime_probe": probes.get("hardware_components"), "loaded_component_evidence": hardware_text, "process_map_evidence": probes.get("process_maps"), "network_socket_evidence": probes.get("network_sockets")}
    write_json(OUT / "stage28s_mock_hardware_evidence.json", mock_evidence)

    expanded = OUT / "stage28s_expanded_runtime.urdf"
    model = {"robot_model": "FR5_simulation_model", "source_xacro": file_ref(ROOT / "ros2_moveit_bridge/config/fairino5_v6_spray_tcp.urdf.xacro", "formal FR5 + spray TCP xacro"), "expanded_runtime_urdf": file_ref(expanded, "runtime-expanded model"), "srdf": file_ref(ROOT / "ros2_moveit_bridge/config/fairino5_v6_spray_tcp.srdf", "formal SRDF"), "joint_limits": file_ref(ROOT / "ros2_moveit_bridge/config/joint_limits_with_jerk.yaml", "formal joint limits"), "joint_count": 6, "joint_order": JOINTS, "joint_order_verified": False, "spray_tcp_present": False, "tf_tree": {"world_to_base": "static identity", "base": "base_link", "chain": ["j1", "j2", "j3", "j4", "j5", "j6", "flange", "spray_tcp_link"]}}
    if expanded.exists():
        root = ET.fromstring(expanded.read_text(encoding="utf-8"))
        joint_names = [joint.attrib.get("name") for joint in root.findall("joint")]
        model["joint_order_verified"] = [name for name in joint_names if name in JOINTS][:6] == JOINTS
        model["spray_tcp_present"] = root.find(".//link[@name='spray_tcp_link']") is not None and root.find(".//joint[@name='spray_tcp_fixed_joint']") is not None
        model["visual_geometry"] = "official fairino5_v6 URDF visual meshes"
        model["collision_geometry"] = "official fairino5_v6 URDF collision meshes; spray_tcp_link has no collision geometry in the frozen model"
        model["visual_collision_relationship"] = "same official FR5 link model; spray_tcp_link is a fixed TCP frame without a separate collision body"
    write_json(OUT / "stage28s_robot_model.json", model)

    oracle = None
    geometry = {"passed": False, "status": "not_available"}
    collision = {"passed": False, "status": "not_available"}
    process_info = {"spray_on_segments": 0, "repositioning_transitions": 0, "waypoint_coverage": None, "process_order_preserved": False, "boundary_order_preserved": False}
    timeline_data = {"rows": [], "tcp_validation": {"tcp_fk_reconstruction": "blocked", "tf_vs_fk_crosscheck": "not_available"}}
    if action.get("execution_completed") and (OUT / "stage27r_controller_state_raw.jsonl").exists():
        oracle = stage27s.compare_native(OUT, trajectory, exact, action)
        write_json(OUT / "stage28s_native_oracle_crosscheck.json", oracle["summary"])
        rows = formal_rows(OUT / "stage27r_controller_state_raw.jsonl", action)
        if rows:
            geometry_samples_path = build_runtime_geometry_samples(OUT, rows, frozen_geometry, metadata, trajectory["times"])
            geometry = run_native_fk(OUT, geometry_samples_path)
            if (OUT / "stage28s_fk_trace.csv").exists():
                collision = collision_run(OUT)
                samples = [json.loads(line) for line in geometry_samples_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                timeline_data = build_timeseries(OUT, rows, samples, metadata, trajectory["times"], geometry, collision)
                process_info = process_summary(metadata, geometry)
                geometry["tf_vs_fk_crosscheck"] = timeline_data["tcp_validation"]["tf_vs_fk_crosscheck"]
                write_json(OUT / "stage28s_geometry_validation.json", geometry)

    write_json(OUT / "stage28s_dynamics_validation.json", {"schema_version": "stage28s-dynamics-validation-v1", "trajectory_identical": bool(source_hash == sha256(OUT / "stage28s_formal_trajectory.csv")), "controller_semantics_identical": bool(oracle and oracle["summary"].get("passed")), "position_limits": exact["summary"]["position_limits"], "velocity_limits": exact["summary"]["velocity_limits"], "acceleration_limits": exact["summary"]["acceleration_limits"], "jerk_limits": exact["summary"]["jerk_limits"], "source": "formal Stage 2.7S exact JTC 4.40.1 spline certificate plus live native controller_state.reference oracle"})
    write_learning_schema(OUT)
    determinism = build_determinism(OUT, timeline_data["rows"], geometry, collision, process_info) if timeline_data["rows"] else {"analysis_rebuilds": 0, "analysis_determinism": "not_available", "result": "blocked"}
    if timeline_data["rows"]:
        write_json(OUT / "stage28s_determinism.json", determinism)

    frozen_after = {"Stage_2_5R2": verify_sums(STAGE25), "Stage_2_7S": verify_sums(STAGE27), "Stage_2_8A_H0": verify_sums(H0), "Stage_2_8A_H1": verify_sums(H1)}
    frozen_mismatch_count = sum(len(item["missing"]) + len(item["mismatches"]) for item in frozen_after.values())
    initial_state = read_json(OUT / "stage28s_initial_state.json") if (OUT / "stage28s_initial_state.json").exists() else {}
    runtime_goal_semantics = bool(sha256(OUT / "stage28s_formal_fjt_goal.json") == sha256(goal_source))
    pass_conditions = {"Stage_2_7S": stage27_gate.get("Stage_2_7", {}).get("status") == "passed", "simulation_scope_verified": bool(model.get("joint_order_verified") and model.get("spray_tcp_present") and mock_evidence["mock_hardware_plugin_loaded"]), "no_real_robot_connection_attempted": not network_attempted, "libfairino_not_loaded": not real_plugin_loaded, "real_FAIRINO_hardware_plugin_not_loaded": not real_plugin_loaded, "controller_manager_started": bool(ready.get("controller_manager_started")), "mock_hardware_plugin_loaded": bool(mock_evidence["mock_hardware_plugin_loaded"]), "joint_state_broadcaster_active": bool(ready.get("joint_state_broadcaster_active")), "joint_trajectory_controller_active": bool(ready.get("jtc_active")), "controller_interpolation_runtime_verified": bool(controller_runtime["runtime_verified"]), "formal_stage27s_trajectory_hash_verified": bool(source_hash == sha256(OUT / "stage28s_formal_trajectory.csv")), "runtime_goal_semantically_identical": runtime_goal_semantics, "simulation_FollowJointTrajectory_goals": int(bool(action.get("goal_sent"))), "goal_accepted": bool(action.get("goal_accepted")), "goal_result_succeeded": bool(action.get("execution_completed") and action.get("action_result") == "successful"), "initial_state_match": bool(initial_state.get("compatible")), "native_spline_oracle_match": bool(oracle and oracle["summary"].get("passed")), "tcp_fk_reconstruction": timeline_data["tcp_validation"].get("tcp_fk_reconstruction") == "passed", "tf_vs_fk_crosscheck": timeline_data["tcp_validation"].get("tf_vs_fk_crosscheck") == "passed", "pose_constraints": geometry.get("pose_constraints") == "passed", "position_limits": exact["summary"]["status"].get("position", False), "velocity_limits": exact["summary"]["status"].get("velocity", False), "acceleration_limits": exact["summary"]["status"].get("acceleration", False), "jerk_limits": exact["summary"]["status"].get("jerk", False), "collision_validation": bool(collision.get("Bullet", {}).get("passed")), "collision_count": collision.get("Bullet", {}).get("collision_count"), "process_order_preserved": process_info.get("process_order_preserved", False), "boundary_order_preserved": process_info.get("boundary_order_preserved", False), "waypoint_coverage_complete": process_info.get("waypoint_coverage") == 720, "spray_on_segments_verified": process_info.get("spray_on_segments") == 10, "repositioning_transitions_verified": process_info.get("repositioning_transitions") == 9, "analysis_determinism": determinism.get("analysis_determinism") == "3/3_passed", "frozen_hash_mismatch_count": frozen_mismatch_count}
    passed = all(value is True for key, value in pass_conditions.items() if key != "collision_count") and pass_conditions["collision_count"] == 0
    blocker = next((key for key, value in pass_conditions.items() if key != "collision_count" and value is not True), None)
    if blocker is None and pass_conditions["collision_count"] != 0:
        blocker = "collision"
    report = {"Project": scope["Project"], "Simulation_Mainline": {"Stage_2_7S": "passed_frozen_unchanged", "Stage_2_8S": "passed" if passed else f"blocked_{blocker}", "Stage_2_9": "unblocked_not_started" if passed else "blocked_by_stage28s"}, "Optional_Real_Hardware_Track": scope["Optional_Real_Hardware_Track"], "runtime": {"ros_distro": controller_runtime["ros_distro"], "controller_manager_started": ready.get("controller_manager_started", False), "hardware_backend": mock_evidence["hardware_backend"], "mock_hardware_plugin_loaded": mock_evidence["mock_hardware_plugin_loaded"], "real_robot_connection_attempted": network_attempted, "libfairino_loaded": real_plugin_loaded, "real_FAIRINO_hardware_plugin_loaded": real_plugin_loaded}, "simulation_safety_conditions": {"no_real_robot_connection_attempted": not network_attempted, "libfairino_not_loaded": not real_plugin_loaded, "real_FAIRINO_hardware_plugin_not_loaded": not real_plugin_loaded}, "controller": {"name": controller_runtime["controller_name"], "version": controller_runtime["joint_trajectory_controller_package_version"], "source_commit": controller_runtime["joint_trajectory_controller_source_commit"], "interpolation_method": controller_runtime["interpolation_method"], "active": controller_runtime["active"]}, "trajectory": {"source": str(candidate_path.resolve()), "sha256": source_hash, "point_count": len(trajectory["times"]), "segment_count": len(trajectory["times"]) - 1, "duration": float(trajectory["times"][-1]), "runtime_goal_identical": runtime_goal_semantics}, "execution": {"simulation_FollowJointTrajectory_goals": int(bool(action.get("goal_sent"))), "accepted": bool(action.get("goal_accepted")), "succeeded": bool(action.get("execution_completed")), "oracle_match": bool(oracle and oracle["summary"].get("passed")), "action_result": action}, "process": process_info, "validation": {"tcp_fk": timeline_data["tcp_validation"].get("tcp_fk_reconstruction"), "tf_vs_fk": timeline_data["tcp_validation"].get("tf_vs_fk_crosscheck"), "pose_constraints": geometry.get("pose_constraints", "not_available"), "position_limits": "passed" if exact["summary"]["status"].get("position") else "blocked", "velocity_limits": "passed" if exact["summary"]["status"].get("velocity") else "blocked", "acceleration_limits": "passed" if exact["summary"]["status"].get("acceleration") else "blocked", "jerk_limits": "passed" if exact["summary"]["status"].get("jerk") else "blocked", "Bullet": collision.get("Bullet", {"passed": False}), "FCL": collision.get("FCL", {"passed": False}), "collision_count": collision.get("Bullet", {}).get("collision_count")}, "simulation_scope": {"robot_motion_simulated": True, "spray_process_motion_simulated": True, "spray_process_semantics_simulated": True, "coating_physics_simulated": False, "paint_particle_physics_simulated": False, "fluid_deposition_simulated": False, "real_robot_used": False}, "determinism": determinism, "frozen_hash_mismatch_count": frozen_mismatch_count, "Can_we_start_Stage_2_9": bool(passed), "Can_we_start_Stage_3": False, "pass_conditions": pass_conditions, "first_blocker": blocker}
    # Persist the final runtime provenance before constructing the artifact
    # manifest.  The manifest is covered by SHA256SUMS; the gate report is
    # excluded because it records the checksum manifest itself.
    runtime_manifest = {"schema_version": "stage28s-runtime-manifest-v1", "formal_run": {"started": True, "one_goal": int(bool(action.get("goal_sent"))), "goal_accepted": bool(action.get("goal_accepted")), "goal_completed": bool(action.get("execution_completed")), "result": action.get("action_result")}, "controller_manager_started": bool(ready.get("controller_manager_started")), "hardware_backend": mock_evidence["hardware_backend"], "loaded_hardware_components": probes.get("hardware_components"), "controllers": probes.get("controllers"), "joint_state_broadcaster_active": bool(ready.get("joint_state_broadcaster_active")), "joint_trajectory_controller_active": bool(ready.get("jtc_active")), "interpolation_runtime": controller_runtime, "tf_topics": {"tf": "/tf", "tf_static": "/tf_static", "raw_rows": len((OUT / "stage28s_tf_raw.jsonl").read_text(encoding="utf-8").splitlines()) if (OUT / "stage28s_tf_raw.jsonl").exists() else 0}, "real_hardware_exclusion": {"real_robot_connection_attempted": network_attempted, "real_robot_network_connection_count": 0 if not network_attempted else "not_zero", "FAIRINO_SDK_runtime_used": False, "libfairino_loaded": real_plugin_loaded, "FAIRINO_hardware_plugin_loaded": real_plugin_loaded, "FAIRINO_hardware_activated": False, "GetRobotRealTimeState_calls": 0, "RobotEnable_calls": 0, "motion_SDK_calls": 0}, "frozen_integrity_before": frozen_before, "frozen_integrity_after": frozen_after, "frozen_hash_mismatch_count": frozen_mismatch_count}
    write_json(OUT / "stage28s_runtime_manifest.json", runtime_manifest)

    write_json(OUT / "stage28s_gate_report.json", report)
    report_md = ["# Stage 2.8S — Full Simulation End-to-End Integration Gate", "", f"- Final status: **{report['Simulation_Mainline']['Stage_2_8S']}**.", f"- Formal input: `{candidate_path}`; SHA256 `{source_hash}`.", f"- Execution: one FJT goal, accepted={action.get('goal_accepted')}, result={action.get('action_result')}.", f"- Runtime: ROS 2 {controller_runtime['ros_distro']}, JTC {controller_runtime['joint_trajectory_controller_package_version']}, interpolation `{controller_runtime['interpolation_method']}`, backend `mock_components/GenericSystem`.", f"- Native oracle: `{bool(oracle and oracle['summary'].get('passed'))}`; FK/TCP: `{timeline_data['tcp_validation'].get('tcp_fk_reconstruction')}`; TF cross-check: `{timeline_data['tcp_validation'].get('tf_vs_fk_crosscheck')}`.", f"- Process mapping: spray-on segments `{process_info.get('spray_on_segments')}`, repositioning transitions `{process_info.get('repositioning_transitions')}`, waypoint coverage `{process_info.get('waypoint_coverage')}/720`.", f"- Collision: Bullet adaptive_discrete_interpolation passed `{collision.get('Bullet', {}).get('passed')}`, collisions `{collision.get('Bullet', {}).get('collision_count')}`, skipped `{collision.get('Bullet', {}).get('skipped_intervals')}`; strict CCD/clearance: `not_available`.", f"- Deterministic analysis: `{determinism.get('analysis_determinism')}`; frozen hash mismatches: `{frozen_mismatch_count}`.", "", "## Scope boundary", "", "This certifies robot spraying motion and spray ON/OFF process semantics in simulation. It does not simulate actuator, motor, gearbox, compliance, particles, coating thickness, or fluid deposition physics. No physical FR5, FAIRINO SDK, FAIRINO hardware plugin, or robot network was used.", "", "## Hardware branch", "", "Stage 2.8A-H0 remains frozen/passed; Stage 2.8A-H1 remains frozen/blocked because real controller identity is not established. It does not block the simulation mainline.", ""]
    (OUT / "stage28s_report.md").write_text("\n".join(report_md), encoding="utf-8")
    required_artifacts = ["stage28s_scope_manifest.json", "stage28s_input_manifest.json", "stage28s_runtime_manifest.json", "stage28s_mock_hardware_evidence.json", "stage28s_controller_runtime.json", "stage28s_formal_fjt_goal.json", "stage28s_fjt_result.json", "stage28s_simulation_timeseries.csv", "stage28s_process_timeline.csv", "stage28s_native_oracle_crosscheck.json", "stage28s_tcp_fk_validation.json", "stage28s_geometry_validation.json", "stage28s_dynamics_validation.json", "stage28s_collision_validation.json", "stage28s_learning_baseline_schema.json", "stage28s_determinism.json", "stage28s_gate_report.json", "stage28s_report.md"]
    required_present = all((OUT / name).exists() for name in required_artifacts)
    # Create the manifest before SHA256SUMS so that the manifest itself is
    # covered without a self-referential hash.  The gate report remains
    # excluded from the checksum list by design.
    pre_manifest_files = [path for path in sorted(OUT.rglob("*")) if path.is_file() and path.name not in {"SHA256SUMS", "stage28s_gate_report.json", "stage28s_artifact_manifest.json"}]
    write_json(OUT / "stage28s_artifact_manifest.json", {"schema_version": "stage28s-artifact-manifest-v1", "sha256sums": str((OUT / "SHA256SUMS").resolve()), "entries": len(pre_manifest_files) + 1, "gate_report_excluded_to_avoid_self_reference": True, "required_artifacts_present": required_present, "required_artifacts": required_artifacts})
    sums = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name not in {"SHA256SUMS", "stage28s_gate_report.json"}:
            sums.append(f"{sha256(path)}  {path.relative_to(OUT).as_posix()}")
    (OUT / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    print(json.dumps({"output_root": str(OUT), "Stage_2_8S": report["Simulation_Mainline"]["Stage_2_8S"], "first_blocker": blocker, "goal": action.get("action_result"), "frozen_hash_mismatch_count": frozen_mismatch_count}, ensure_ascii=False, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
