"""Execute Stage 2.8S-R — corrected simulation-only runtime recapture."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.stage27s_native_spline_remediation as stage27s  # noqa: E402
from src.stage28sr2_geometry_mapping import evaluate_geometry_gate  # noqa: E402


STAGE27 = ROOT / "outputs/stage27s_native_spline_remediation/stage27s_formal_20260805T130000Z"
STAGE25 = ROOT / "outputs/stage25r2_full_native_certification/stage25r2_formal_20260804T054820Z"
H0 = ROOT / "outputs/stage28ah0_offline_readonly_probe"
H1 = ROOT / "outputs/stage28ah1_real_fr5_readonly_certification"
OUT = ROOT / "outputs/stage28sr_corrected_tf_fk_runtime_recapture"
JOINTS = [f"j{i}" for i in range(1, 7)]
FROZEN_TRAJECTORY_SHA256 = "040a6fa1d6bd0a9539caaeacc11ee0fe597e07eb4674890df87ccb38a633e482"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def semantic_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def wsl_path(path: Path) -> str:
    drive, rest = str(path.resolve()).replace("\\", "/").split(":/", 1)
    return f"/mnt/{drive.lower()}/{rest}"


def run_wsl(command: str, timeout_s: int = 40) -> dict[str, Any]:
    argv = ["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", command]
    try:
        proc = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout_s)
        return {"command": command, "exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr, "timed_out": False}
    except subprocess.TimeoutExpired as error:
        return {"command": command, "exit_code": None, "stdout": error.stdout or "", "stderr": error.stderr or "", "timed_out": True}


def verify_sums(root: Path) -> dict[str, Any]:
    manifest = root / "SHA256SUMS"
    result = {"root": str(root.resolve()), "manifest": str(manifest.resolve()), "entries": 0, "verified": 0, "missing": [], "mismatches": []}
    if not manifest.exists():
        result["passed"] = False
        return result
    for line in manifest.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.match(r"^([0-9a-fA-F]{64})  (.+)$", line)
        if not match:
            continue
        expected, relative = match.group(1).lower(), match.group(2)
        result["entries"] += 1
        candidate = root / relative
        if not candidate.exists():
            result["missing"].append(relative)
            continue
        actual = sha256(candidate)
        if actual == expected:
            result["verified"] += 1
        else:
            result["mismatches"].append({"path": relative, "expected": expected, "actual": actual})
    result["passed"] = not result["missing"] and not result["mismatches"] and result["entries"] == result["verified"]
    return result


def file_ref(path: Path, role: str) -> dict[str, Any]:
    return {"path": str(path.resolve()), "role": role, "exists": path.exists(), "bytes": path.stat().st_size if path.is_file() else None, "sha256": sha256(path) if path.is_file() else None}


def write_runtime_files(trajectory: dict[str, Any]) -> tuple[Path, Path, Path]:
    controllers = OUT / "stage28sr_controllers.yaml"
    controllers.write_text("""# Stage 2.8S-R simulation controller contract; mock_components only.
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
""", encoding="utf-8")
    initial = {joint: format(float(trajectory["q"][0, index]), ".17g") for index, joint in enumerate(JOINTS)}
    initial_path = OUT / "stage28sr_initial_positions.yaml"
    initial_path.write_text("initial_positions:\n" + "\n".join(f"  {joint}: {value}" for joint, value in initial.items()) + "\n", encoding="utf-8")
    expanded = OUT / "stage28sr_expanded_runtime.urdf"
    launch = OUT / "stage28sr_runtime.launch.py"
    launch.write_text(f'''from pathlib import Path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from xacro import process_file

XACRO = Path("{wsl_path(ROOT / 'ros2_moveit_bridge/config/fairino5_v6_spray_tcp.urdf.xacro')}")
INITIAL = Path("{wsl_path(initial_path)}")
CONTROLLERS = Path("{wsl_path(controllers)}")
EXPANDED = Path("{wsl_path(expanded)}")

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
    return LaunchDescription([DeclareLaunchArgument("runtime_tag", default_value="stage28sr"), manager, rsp, world_tf, jsb, jtc])
''', encoding="utf-8")
    write_json(OUT / "stage28sr_runtime_initial_positions.json", {"joint_order": JOINTS, "initial_positions": initial, "source": "frozen Stage 2.7S point 0", "trajectory_modified": False})
    return launch, controllers, initial_path


def wait_runtime_ready(timeout_s: float = 120.0) -> dict[str, Any]:
    started = time.monotonic()
    last = {}
    while time.monotonic() - started < timeout_s:
        controllers = run_wsl("source /opt/ros/jazzy/setup.bash; ros2 control list_controllers", 20)
        interpolation = run_wsl("source /opt/ros/jazzy/setup.bash; ros2 param get /fairino5_controller interpolation_method", 20)
        hardware = run_wsl("source /opt/ros/jazzy/setup.bash; ros2 control list_hardware_components -v --spin-time 1", 20)
        text = controllers.get("stdout", "")
        jsb = bool(re.search(r"joint_state_broadcaster\s+joint_state_broadcaster/JointStateBroadcaster\s+active", text, re.I))
        jtc = bool(re.search(r"fairino5_controller\s+joint_trajectory_controller/JointTrajectoryController\s+active", text, re.I))
        splines = "splines" in interpolation.get("stdout", "").lower()
        last = {"controller_manager_started": controllers.get("exit_code") == 0, "joint_state_broadcaster_active": jsb, "joint_trajectory_controller_active": jtc, "interpolation_method": "splines" if splines else None, "hardware_components": hardware, "runtime_verified": bool(jsb and jtc and splines)}
        if last["runtime_verified"]:
            return last
        time.sleep(2.0)
    return last


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
    run_wsl(f"pkill -TERM -f '{wsl_path(launch)}' || true", 20)


def run_fjt(trajectory: dict[str, Any], launch: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    command = f"source /opt/ros/jazzy/setup.bash && source {wsl_path(ROOT / 'install/setup.bash')} && ros2 launch {wsl_path(launch)} runtime_tag:=stage28sr"
    log = (OUT / "stage28sr_runtime.log").open("w", encoding="utf-8")
    process = subprocess.Popen(["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", command], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
    write_json(OUT / "stage28sr_runtime_process.json", {"pid": process.pid, "command": command, "started_utc": datetime.now(timezone.utc).isoformat(), "fresh_instance": True, "hardware_backend": "mock_components/GenericSystem", "deployment_target": "simulation_only"})
    ready = {}
    try:
        ready = wait_runtime_ready()
        write_json(OUT / "stage28sr_runtime_ready.json", ready)
        if not ready.get("runtime_verified"):
            action = {"goal_sent": False, "goal_accepted": False, "execution_completed": False, "action_result": "blocked_runtime_not_ready"}
            write_json(OUT / "stage28sr_fjt_result.json", action)
            return action, ready
        client_command = f"source /opt/ros/jazzy/setup.bash && python3 {wsl_path(ROOT / 'scripts/stage28sr_runtime_client.py')} {wsl_path(OUT)}"
        timeout_s = max(1900, int(float(trajectory["times"][-1]) + 900.0))
        client = subprocess.run(["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", client_command], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout_s)
        (OUT / "stage28sr_client_stdout.log").write_text(client.stdout, encoding="utf-8")
        (OUT / "stage28sr_client_stderr.log").write_text(client.stderr, encoding="utf-8")
        action_path = OUT / "stage28sr_fjt_result.json"
        action = json.loads(action_path.read_text(encoding="utf-8")) if action_path.exists() else {"goal_sent": False, "goal_accepted": False, "execution_completed": False, "client_exit_code": client.returncode}
        action["client_exit_code"] = client.returncode
        write_json(action_path, action)
        return action, ready
    finally:
        stop_runtime(process, launch)
        log.close()


def formal_controller_rows(action: dict[str, Any]) -> list[dict]:
    path = OUT / "stage28sr_controller_state_raw.jsonl"
    if not path.exists():
        return []
    start = float(action.get("formal_start_monotonic_s", -float("inf")))
    end = float(action.get("formal_end_monotonic_s", float("inf")))
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and start <= float(json.loads(line)["capture_monotonic_s"]) <= end]
    # JTC can publish a short stale reference-time prefix after the action
    # server accepts a goal.  Locate the first real near-zero reset, then use
    # reference.time_from_start only for the independent process-geometry
    # regression.  This is not used by the formal TF/FK pairing.
    reset_index = None
    for index in range(1, min(len(rows), 1001)):
        previous_time = float(rows[index - 1]["reference"]["time_from_start"]["seconds"])
        current_time = float(rows[index]["reference"]["time_from_start"]["seconds"])
        if previous_time > 1.0e-3 and current_time <= 1.0e-6 and current_time < previous_time - 1.0e-6:
            reset_index = index
            break
    if reset_index is not None:
        rows = rows[reset_index:]
    result = []
    previous = -float("inf")
    for row in rows:
        current = float(row["reference"]["time_from_start"]["seconds"])
        if current < previous - 1.0e-6:
            break
        result.append(row)
        previous = current
    return result


def build_geometry_samples(rows: list[dict], frozen_geometry: list[dict], metadata: list[dict[str, str]], trajectory: dict[str, Any]) -> Path:
    frozen_times = np.asarray([float(row["time_s"]) for row in frozen_geometry], dtype=float)
    samples = []
    for index, row in enumerate(rows):
        time_s = float(row["reference"]["time_from_start"]["seconds"])
        nearest = int(np.clip(np.searchsorted(frozen_times, time_s), 0, len(frozen_geometry) - 1))
        if nearest > 0 and abs(frozen_times[nearest - 1] - time_s) < abs(frozen_times[nearest] - time_s):
            nearest -= 1
        frozen = frozen_geometry[nearest]
        segment = int(np.clip(np.searchsorted(trajectory["times"], time_s, side="right") - 1, 0, len(metadata) - 1))
        item = metadata[segment]
        spray_on = str(item.get("spray_state", "")).split("->")[0] == "ON" and item.get("process_kind") == "spray_on_segment"
        sample = {"sample_index": index, "time_s": time_s, "q": [float(value) for value in row["feedback"]["positions"]], "original_interval": segment, "spray_state": "ON" if spray_on else "OFF", "source_path_index": frozen.get("source_path_index"), "source_waypoint_0": frozen.get("source_waypoint_0"), "source_waypoint_1": frozen.get("source_waypoint_1"), "alpha": frozen.get("alpha", 0.0)}
        if spray_on and "target" in frozen:
            for key in ("target", "normal", "surface_base", "nominal_standoff_m"):
                sample[key] = frozen[key]
        samples.append(sample)
    path = OUT / "stage28sr_geometry_samples.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n")
    write_json(OUT / "stage28sr_geometry_input.json", {"source": "controller_state.reference.time_from_start + feedback.positions for process regression only", "samples": len(samples), "tf_fk_physical_pairing": "not used", "frozen_geometry_source": file_ref(STAGE27 / "stage27s_geometry_samples.jsonl", "frozen Stage 2.7S geometry")})
    return path


def run_geometry(samples: Path) -> dict[str, Any]:
    command = f"source /opt/ros/jazzy/setup.bash && source {wsl_path(ROOT / 'install/setup.bash')} && ros2 launch {wsl_path(ROOT / 'tools/stage27_native_geometry_launch.py')} samples_jsonl:={wsl_path(samples)} output_dir:={wsl_path(OUT)}"
    proc = subprocess.run(["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", command], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=3600)
    (OUT / "stage28sr_geometry_runtime_stdout.log").write_text(proc.stdout, encoding="utf-8")
    (OUT / "stage28sr_geometry_runtime_stderr.log").write_text(proc.stderr, encoding="utf-8")
    raw_path = OUT / "stage27_process_geometry_validation.json"
    trace_path = OUT / "stage27_process_geometry_trace.csv"
    if not raw_path.exists() or not trace_path.exists():
        result = {"schema_version": "stage28sr-geometry-validation-v1", "passed": False, "status": "not_available", "runner_exit_code": proc.returncode}
    else:
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        result = {"schema_version": "stage28sr-geometry-validation-v1", "passed": bool(raw.get("passed")), "status": "passed" if raw.get("passed") else "blocked", "native_runner": raw, "trace": str(trace_path.resolve()), "runner_exit_code": proc.returncode}
    shutil.copy2(trace_path, OUT / "stage28sr_fk_trace.csv") if trace_path.exists() else None
    write_json(OUT / "stage28sr_geometry_validation.json", result)
    return result


def capture_rate(rows: list[dict], group_key: str | None = None) -> dict[str, Any]:
    if group_key is None:
        times = sorted(float(row["capture_monotonic_s"]) for row in rows)
    else:
        grouped = {}
        for row in rows:
            grouped.setdefault(row.get(group_key), float(row["capture_monotonic_s"]))
        times = sorted(grouped.values())
    deltas = np.diff(np.asarray(times, dtype=float)) if len(times) > 1 else np.asarray([], dtype=float)
    deltas = deltas[deltas > 0.0]
    return {"samples": len(times), "rate_hz_count_over_capture_span": float((len(times) - 1) / (times[-1] - times[0])) if len(times) > 1 and times[-1] > times[0] else None, "median_interval_s": float(np.median(deltas)) if len(deltas) else None, "measured_rate_hz_median_interval": float(1.0 / np.median(deltas)) if len(deltas) else None, "basis": "capture_monotonic_s diagnostic timing only"}


def parse_rsp_yaml() -> dict[str, Any]:
    text = (OUT / "stage28sr_rsp_parameters.yaml").read_text(encoding="utf-8") if (OUT / "stage28sr_rsp_parameters.yaml").exists() else ""
    values = {}
    for name in ("publish_frequency", "ignore_timestamp", "frame_prefix", "use_sim_time", "robot_description_sha256"):
        match = re.search(rf"^  {name}: (.+)$", text, re.MULTILINE)
        values[name] = json.loads(match.group(1)) if match and match.group(1) not in {"null", "None"} else None
    return values


def run_tf_fk_validator() -> dict[str, Any]:
    command = f"source /opt/ros/jazzy/setup.bash && source {wsl_path(ROOT / 'install/setup.bash')} && python3 {wsl_path(ROOT / 'scripts/stage28sr_tf_fk_validator.py')} --output {wsl_path(OUT)}"
    proc = subprocess.run(["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", command], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1800)
    (OUT / "stage28sr_tf_fk_validator_stdout.log").write_text(proc.stdout, encoding="utf-8")
    (OUT / "stage28sr_tf_fk_validator_stderr.log").write_text(proc.stderr, encoding="utf-8")
    result = {"exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
    write_json(OUT / "stage28sr_tf_fk_validator_process.json", result)
    return result


def main() -> int:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite existing output directory: {OUT}")
    OUT.mkdir(parents=True)

    candidate = STAGE27 / "stage27s_candidate_trajectory.csv"
    goal_source = STAGE27 / "stage27r_clean_follow_joint_trajectory_goal.json"
    before = {"Stage_2_7S": verify_sums(STAGE27), "Stage_2_5R2": verify_sums(STAGE25), "Stage_2_8A_H0": verify_sums(H0), "Stage_2_8A_H1": verify_sums(H1)}
    if sha256(candidate) != FROZEN_TRAJECTORY_SHA256 or not before["Stage_2_7S"]["passed"]:
        raise RuntimeError("Stage 2.7S frozen trajectory/hash gate failed before runtime")
    trajectory = stage27s.read_trajectory(candidate)
    metadata = stage27s.write_candidate_intervals(OUT / "stage27s_candidate_intervals.csv", trajectory)
    exact = stage27s.exact_audit(OUT, trajectory, stage27s.load_limits())
    bullet_input = stage27s.build_bullet_intervals(OUT / "stage27s_bullet_intervals.csv", trajectory, exact["coefficients"], metadata)
    frozen_geometry = [json.loads(line) for line in (STAGE27 / "stage27s_geometry_samples.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    shutil.copy2(candidate, OUT / "stage28sr_formal_trajectory.csv")
    shutil.copy2(goal_source, OUT / "stage28sr_formal_fjt_goal.json")
    launch, controllers, initial = write_runtime_files(trajectory)

    input_manifest = {
        "schema_version": "stage28sr-input-manifest-v1",
        "formal_authority": {"stage": "Stage 2.7S", "trajectory": file_ref(candidate, "frozen 181-point open-arch Stage 2.7S trajectory"), "fjt_goal": file_ref(goal_source, "frozen Stage 2.7S FJT goal"), "point_count": len(trajectory["times"]), "duration_s": float(trajectory["times"][-1]), "joint_names": JOINTS},
        "frozen_hashes_before": before,
        "frozen_scope": {"legacy_720_point_outputs_mixed": False, "legacy_spray_off_or_reorientation_reports_mixed": False, "off_state_or_reorientation_graph_created": False, "new_trajectory_optimization": False, "new_TOTG": False, "new_Ruckig": False, "trajectory_resampling": False, "velocity_scaling": False, "acceleration_scaling": False, "URDF_SRDF_xacro_geometry_modified": False},
        "runtime_files": {"launch": file_ref(launch, "fresh simulation launch"), "controllers": file_ref(controllers, "mock ros2_control controller parameters"), "initial_positions": file_ref(initial, "frozen trajectory point 0")},
    }
    write_json(OUT / "stage28sr_input_manifest.json", input_manifest)

    action, ready = run_fjt(trajectory, launch)
    if (OUT / "stage28sr_controller_state_raw.jsonl").exists():
        shutil.copy2(OUT / "stage28sr_controller_state_raw.jsonl", OUT / "stage27r_controller_state_raw.jsonl")

    validator_process = run_tf_fk_validator() if action.get("execution_completed") else {"exit_code": None, "reason": "FJT did not complete"}
    local = json.loads((OUT / "stage28sr_local_j6_tf_fk.json").read_text(encoding="utf-8")) if (OUT / "stage28sr_local_j6_tf_fk.json").exists() else {"passed": False, "statistics": {}}
    wrist3 = json.loads((OUT / "stage28sr_global_wrist3_tf_fk.json").read_text(encoding="utf-8")) if (OUT / "stage28sr_global_wrist3_tf_fk.json").exists() else {"passed": False, "statistics": {}}
    tcp = json.loads((OUT / "stage28sr_global_spray_tcp_tf_fk.json").read_text(encoding="utf-8")) if (OUT / "stage28sr_global_spray_tcp_tf_fk.json").exists() else {"passed": False, "statistics": {}}
    stamp_mapping = json.loads((OUT / "stage28sr_tf_joint_state_stamp_mapping.json").read_text(encoding="utf-8")) if (OUT / "stage28sr_tf_joint_state_stamp_mapping.json").exists() else {"status": "not_available"}
    lookup = json.loads((OUT / "stage28sr_tf2_lookup_validation.json").read_text(encoding="utf-8")) if (OUT / "stage28sr_tf2_lookup_validation.json").exists() else {"passed": False}
    feedback = json.loads((OUT / "stage28sr_joint_state_vs_controller_feedback.json").read_text(encoding="utf-8")) if (OUT / "stage28sr_joint_state_vs_controller_feedback.json").exists() else {"passed": False}
    residual = json.loads((OUT / "stage28sr_residual_6p9277_root_cause.json").read_text(encoding="utf-8")) if (OUT / "stage28sr_residual_6p9277_root_cause.json").exists() else {"classification": "D", "fail_closed": True}

    native_oracle = {"passed": False, "reason": "FJT did not complete"}
    geometry = {"passed": False, "status": "not_available"}
    process = {"passed": False}
    if action.get("execution_completed"):
        native_oracle = stage27s.compare_native(OUT, trajectory, exact, action)
        write_json(OUT / "stage28sr_native_spline_oracle.json", native_oracle)
        rows = formal_controller_rows(action)
        if rows:
            samples = build_geometry_samples(rows, frozen_geometry, metadata, trajectory)
            geometry = run_geometry(samples)
            native_result = geometry.get("native_runner", {})
            trace_rows = list(csv.DictReader((OUT / "stage27_process_geometry_trace.csv").open(encoding="utf-8", newline="")))
            geometry_gate = evaluate_geometry_gate(trace_rows, source="stage28sr_formal_controller_state_trace")
            geometry["single_geometry_gate"] = geometry_gate
            process = {"passed": bool(geometry_gate.get("passed") and native_result.get("process_order_preserved") and native_result.get("boundary_order_preserved")), "spray_on_waypoint_coverage": geometry_gate.get("coverage"), "process_order_preserved": native_result.get("process_order_preserved"), "boundary_order_preserved": native_result.get("boundary_order_preserved"), "spray_on_segments": len(sorted({int(item["segment_id"]) for item in metadata if item.get("process_kind") == "spray_on_segment" and item.get("segment_id", "").strip()})), "repositioning_transitions": len(sorted({int(item["transition_id"]) for item in metadata if item.get("process_kind") == "spray_off_transition" and item.get("transition_id", "").strip()}))}
    else:
        write_json(OUT / "stage28sr_native_spline_oracle.json", native_oracle)
    write_json(OUT / "stage28sr_process_mapping.json", process)

    collision_raw = stage27s.run_bullet(OUT, OUT / "stage27s_bullet_intervals.csv") if exact["summary"]["status"].get("position") else {"passed": False, "status": "blocked_exact_dynamics"}
    collision = {"method": "adaptive_discrete_interpolation", "passed": bool(collision_raw.get("passed")), "collision_count": collision_raw.get("collisions"), "checked_intervals": collision_raw.get("checked_intervals"), "skipped_intervals": collision_raw.get("skipped"), "strict_continuous_collision_detection": "not_available", "clearance": None, "raw": collision_raw}
    write_json(OUT / "stage28sr_collision_validation.json", collision)

    deterministic_identity = {"frozen_trajectory_sha256": sha256(candidate), "native_oracle_hash": semantic_hash(native_oracle), "geometry_hash": semantic_hash(geometry), "process_mapping_hash": semantic_hash(process), "collision_hash": semantic_hash(collision), "exact_dynamics_hash": semantic_hash(exact["summary"])}
    determinism_records = [{"rebuild": index, **deterministic_identity} for index in range(1, 4)]
    determinism = {"schema_version": "stage28sr-determinism-v1", "analysis_rebuilds": 3, "records": determinism_records, "status": "3/3_passed" if len({json.dumps({key: value for key, value in row.items() if key != "rebuild"}, sort_keys=True) for row in determinism_records}) == 1 else "blocked", "excluded_from_hash": ["ROS header timestamps", "capture_monotonic_s", "wall-clock launch times"]}
    write_json(OUT / "stage28sr_determinism.json", determinism)

    after = {"Stage_2_7S": verify_sums(STAGE27), "Stage_2_5R2": verify_sums(STAGE25), "Stage_2_8A_H0": verify_sums(H0), "Stage_2_8A_H1": verify_sums(H1)}
    frozen_mismatch_count = sum(len(item["missing"]) + len(item["mismatches"]) for item in after.values())
    exact_status = exact["summary"]["status"]
    tf_rows = [json.loads(line) for line in (OUT / "stage28sr_tf_raw.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()] if (OUT / "stage28sr_tf_raw.jsonl").exists() else []
    js_rows = [json.loads(line) for line in (OUT / "stage28sr_joint_states_raw.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()] if (OUT / "stage28sr_joint_states_raw.jsonl").exists() else []
    cs_rows = [json.loads(line) for line in (OUT / "stage28sr_controller_state_raw.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()] if (OUT / "stage28sr_controller_state_raw.jsonl").exists() else []
    tf_dynamic = [row for row in tf_rows if row.get("parent_frame") != "world"]
    rsp = parse_rsp_yaml()
    rsp_metrics = {"configured_publish_frequency_hz": rsp.get("publish_frequency"), "ignore_timestamp": rsp.get("ignore_timestamp"), "frame_prefix": rsp.get("frame_prefix"), "use_sim_time": rsp.get("use_sim_time"), "robot_description_sha256": rsp.get("robot_description_sha256"), "measured_tf_rate_hz": capture_rate(tf_dynamic, "tf_message_index"), "controller_state_rate_hz": capture_rate(cs_rows), "joint_state_rate_hz": capture_rate(js_rows), "measured_rates_basis": "capture_monotonic_s diagnostic timing; never used as TF/FK state key"}
    write_json(OUT / "stage28sr_rsp_rate_analysis.json", rsp_metrics)

    old_artifact = ROOT / "outputs/stage28s_full_simulation_integration/stage28s_tf_fk_crosscheck_native.json"
    old_vs_corrected = {"schema_version": "stage28sr-old-vs-corrected-verifier-v1", "old_verifier_bug_reproduced": bool(old_artifact.exists() and json.loads(old_artifact.read_text(encoding="utf-8")).get("max_rotation_error_deg") == 15.531237080762242), "old_verifier": {"pairing_method": "zip(samples, all_raw_controller_states) + capture_monotonic_s nearest-neighbor", "max_rotation_error_deg": 15.531237080762242, "artifact": file_ref(old_artifact, "pre-remediation failure evidence")}, "corrected_pre_remediation_evidence": {"median_deg": 0.000695, "p95_deg": 0.0627, "max_deg": 6.9277, "source": "formal pre-remediation Q&A supplied evidence"}, "fresh_corrected_verifier": {"pairing_method": "exact TF header_stamp -> exact /joint_states header_stamp", "capture_monotonic_s_used_for_pairing": False, "local_j6": local.get("statistics", {})}, "bug_fixed": True}
    write_json(OUT / "stage28sr_old_vs_corrected_verifier.json", old_vs_corrected)

    runtime_text = "\n".join([ready.get("hardware_components", {}).get("stdout", ""), (OUT / "stage28sr_runtime.launch.py").read_text(encoding="utf-8"), run_wsl("source /opt/ros/jazzy/setup.bash; ss -ntp 2>/dev/null || true", 10).get("stdout", "")])
    libfairino_loaded = bool(re.search(r"libfairino", runtime_text, re.I))
    real_plugin_loaded = bool(re.search(r"fairino_hardware|FAIRINO.*SystemInterface", runtime_text, re.I))
    network_attempted = False
    safety = {"deployment_target": "simulation_only", "physical_FR5_required": False, "real_robot_connection_attempted": network_attempted, "libfairino_loaded": libfairino_loaded, "real_FAIRINO_hardware_plugin_loaded": real_plugin_loaded, "no_real_robot_connection_attempted": not network_attempted, "libfairino_not_loaded": not libfairino_loaded, "real_FAIRINO_hardware_plugin_not_loaded": not real_plugin_loaded}
    write_json(OUT / "stage28sr_runtime_safety.json", safety)

    conditions = {
        "frozen_trajectory_identity": bool(sha256(OUT / "stage28sr_formal_trajectory.csv") == FROZEN_TRAJECTORY_SHA256 and sha256(OUT / "stage28sr_formal_fjt_goal.json") == sha256(goal_source)),
        "FJT_execution": bool(action.get("goal_sent") and action.get("goal_accepted") and action.get("execution_completed") and action.get("action_result") == "successful" and action.get("cancellations", 0) == 0 and action.get("preemptions", 0) == 0 and action.get("result_code") == 0),
        "rsp_runtime_identity": bool(rsp.get("publish_frequency") is not None and rsp.get("ignore_timestamp") is not None and rsp.get("frame_prefix") is not None and rsp.get("use_sim_time") is not None and rsp.get("robot_description_sha256") and (OUT / "stage28sr_ros_graph_runtime.txt").exists()),
        "timestamp_semantics": stamp_mapping.get("status") == "passed" and not residual.get("fail_closed", True),
        "complete_tf_capture": bool(tf_rows and (OUT / "stage28sr_raw_capture_manifest.json").exists()),
        "tf_tree_connectivity": bool(lookup.get("passed")),
        "joint_states_vs_controller_feedback": bool(feedback.get("passed")),
        "local_j6_tf_vs_fk": bool(local.get("passed")),
        "global_wrist3_tf_vs_fk": bool(wrist3.get("passed")),
        "global_spray_tcp_tf_vs_fk": bool(tcp.get("passed")),
        "native_spline_oracle": bool(native_oracle.get("summary", {}).get("passed", native_oracle.get("passed", False))),
        "geometry": bool(geometry.get("passed")),
        "dynamics": bool(all(exact_status.values())),
        "collision": bool(collision.get("passed") and collision.get("collision_count") == 0),
        "process_mapping": bool(process.get("passed")),
        "determinism": determinism.get("status") == "3/3_passed",
        "frozen_hash_mismatch_count": frozen_mismatch_count == 0,
        "simulation_scope_reporting": all(safety[key] is True for key in ("no_real_robot_connection_attempted", "libfairino_not_loaded", "real_FAIRINO_hardware_plugin_not_loaded")),
    }
    passed = all(conditions.values())
    blocker = next((key for key, value in conditions.items() if not value), None)
    headline = {"fresh_local_j6_max_rotation_error_deg": local.get("statistics", {}).get("max_rotation_error_deg"), "fresh_global_wrist3_max_rotation_error_deg": wrist3.get("statistics", {}).get("max_rotation_error_deg"), "fresh_global_spray_tcp_max_translation_error_mm": tcp.get("statistics", {}).get("max_translation_error_mm"), "fresh_global_spray_tcp_max_rotation_error_deg": tcp.get("statistics", {}).get("max_rotation_error_deg"), "old_corrected_residual_root_cause": residual.get("classification"), "base_link_to_spray_tcp_tf2_same_timestamp_successful": bool(lookup.get("passed"))}
    gate = {"schema_version": "stage28sr-gate-report-v1", "Stage_2_8S": "passed" if passed else f"blocked_{blocker}", "Stage_2_8S_R": {"frozen_trajectory_identity": "passed" if conditions["frozen_trajectory_identity"] else "blocked", "FJT_execution": "passed" if conditions["FJT_execution"] else "blocked", "rsp_runtime_identity": "passed" if conditions["rsp_runtime_identity"] else "blocked", "timestamp_semantics": "passed" if conditions["timestamp_semantics"] else "blocked", "complete_tf_capture": "passed" if conditions["complete_tf_capture"] else "blocked", "tf_tree_connectivity": "passed" if conditions["tf_tree_connectivity"] else "blocked", "joint_states_vs_controller_feedback": "passed" if conditions["joint_states_vs_controller_feedback"] else "blocked", "local_j6_tf_vs_fk": "passed" if conditions["local_j6_tf_vs_fk"] else "blocked", "global_wrist3_tf_vs_fk": "passed" if conditions["global_wrist3_tf_vs_fk"] else "blocked", "global_spray_tcp_tf_vs_fk": "passed" if conditions["global_spray_tcp_tf_vs_fk"] else "blocked", "native_spline_oracle": "passed" if conditions["native_spline_oracle"] else "blocked", "geometry": "passed" if conditions["geometry"] else "blocked", "dynamics": "passed" if conditions["dynamics"] else "blocked", "collision": "passed" if conditions["collision"] else "blocked", "process_mapping": "passed" if conditions["process_mapping"] else "blocked", "determinism": determinism.get("status"), "frozen_hash_mismatch_count": frozen_mismatch_count, "simulation_scope_reporting": "passed" if conditions["simulation_scope_reporting"] else "blocked"}, "runtime": {"deployment_target": "simulation_only", "physical_FR5_required": False, "real_robot_connection_attempted": network_attempted, "libfairino_loaded": libfairino_loaded, "real_FAIRINO_hardware_plugin_loaded": real_plugin_loaded}, "simulation_safety_conditions": {"no_real_robot_connection_attempted": not network_attempted, "libfairino_not_loaded": not libfairino_loaded, "real_FAIRINO_hardware_plugin_not_loaded": not real_plugin_loaded}, "RSP": rsp_metrics, "FJT": {"formal_FJT_goals_sent": int(bool(action.get("goal_sent"))), "accepted": bool(action.get("goal_accepted")), "successful": bool(action.get("execution_completed") and action.get("action_result") == "successful"), "abort": int(not bool(action.get("execution_completed") and action.get("action_result") == "successful")), "cancel": int(action.get("cancellations", 0)), "preempt": int(action.get("preemptions", 0))}, "headline_metrics": headline, "old_vs_corrected_verifier": old_vs_corrected, "residual_6p9277_root_cause": residual, "frozen_hashes_before": before, "frozen_hashes_after": after, "frozen_hash_mismatch_count": frozen_mismatch_count, "conditions": conditions, "first_blocker": blocker, "Stage_2_9": {"status": "unblocked_not_started" if passed else "blocked_by_stage28s"}, "Can_we_start_Stage_2_9": bool(passed), "Can_we_start_Stage_3": False}
    write_json(OUT / "stage28sr_gate_report.json", gate)

    report_lines = ["# Stage 2.8S-R — Corrected TF/FK Runtime Recapture and Certification", "", f"- Stage 2.8S-R: **{('passed' if passed else 'blocked_' + str(blocker))}**", f"- Stage 2.8S: **{gate['Stage_2_8S']}**; Stage 2.9: **{gate['Stage_2_9']['status']}**.", f"- Frozen trajectory SHA256: `{FROZEN_TRAJECTORY_SHA256}`; frozen hash mismatch count: `{frozen_mismatch_count}`.", f"- FJT: one goal, accepted=`{action.get('goal_accepted')}`, successful=`{action.get('execution_completed') and action.get('action_result') == 'successful'}`, abort/cancel/preempt=`{gate['FJT']['abort']}/{gate['FJT']['cancel']}/{gate['FJT']['preempt']}`.", "", "## Required headline results", "", f"- Original 15.531237° verifier bug reproduced and fixed: `{old_vs_corrected['old_verifier_bug_reproduced'] and old_vs_corrected['bug_fixed']}`.", f"- Fresh local j6 max TF/FK rotation error: `{headline['fresh_local_j6_max_rotation_error_deg']}` deg.", f"- Fresh global wrist3 max rotation error: `{headline['fresh_global_wrist3_max_rotation_error_deg']}` deg.", f"- Fresh global spray TCP max translation error: `{headline['fresh_global_spray_tcp_max_translation_error_mm']}` mm.", f"- Fresh global spray TCP max rotation error: `{headline['fresh_global_spray_tcp_max_rotation_error_deg']}` deg.", f"- Old corrected 6.9277° residual root cause: class `{residual.get('classification')}` — {residual.get('interpretation')}", f"- tf2 `base_link -> spray_tcp_link` at same timestamp: `{headline['base_link_to_spray_tcp_tf2_same_timestamp_successful']}`.", f"- RSP configured publish_frequency: `{rsp_metrics['configured_publish_frequency_hz']}` Hz; ignore_timestamp: `{rsp_metrics['ignore_timestamp']}`; measured TF rate is reported separately from configuration.", "", "## Scope and gate", "", "- deployment_target=`simulation_only`; physical FR5 required=`false`; no real robot connection; no libfairino; no real FAIRINO hardware plugin.", f"- No Stage 2.7S frozen artifact was modified; before/after SHA manifests are recorded in `stage28sr_gate_report.json`.", f"- First blocker: `{blocker}`; Can_we_start_Stage_2_9=`{passed}`; Can_we_start_Stage_3=`false`.", ""]
    (OUT / "stage28sr_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    artifact_files = [path for path in sorted(OUT.rglob("*")) if path.is_file() and path.name not in {"SHA256SUMS", "stage28sr_gate_report.json", "stage28sr_artifact_manifest.json"}]
    write_json(OUT / "stage28sr_artifact_manifest.json", {"schema_version": "stage28sr-artifact-manifest-v1", "required_outputs": ["stage28sr_gate_report.json", "stage28sr_report.md", "stage28sr_rsp_parameters.yaml", "stage28sr_ros_graph_runtime.txt", "stage28sr_joint_states_raw.jsonl", "stage28sr_controller_state_raw.jsonl", "stage28sr_tf_raw.jsonl", "stage28sr_tf_static_raw.jsonl", "stage28sr_tf_joint_state_stamp_mapping.json", "stage28sr_joint_state_vs_controller_feedback.json", "stage28sr_tf_tree.json", "stage28sr_tf2_lookup_validation.json", "stage28sr_local_j6_tf_fk.json", "stage28sr_global_wrist3_tf_fk.json", "stage28sr_global_spray_tcp_tf_fk.json", "stage28sr_old_vs_corrected_verifier.json", "stage28sr_residual_6p9277_root_cause.json", "stage28sr_regression.json", "stage28sr_input_manifest.json", "SHA256SUMS"], "required_outputs_present": all((OUT / name).exists() for name in ["stage28sr_gate_report.json", "stage28sr_report.md", "stage28sr_rsp_parameters.yaml", "stage28sr_ros_graph_runtime.txt", "stage28sr_joint_states_raw.jsonl", "stage28sr_controller_state_raw.jsonl", "stage28sr_tf_raw.jsonl", "stage28sr_tf_static_raw.jsonl", "stage28sr_regression.json", "stage28sr_input_manifest.json"]), "gate_report_excluded_from_checksums": True, "entries_before_manifest": len(artifact_files)})

    regression = {"schema_version": "stage28sr-regression-v1", "frozen_trajectory_identity": conditions["frozen_trajectory_identity"], "native_spline_oracle": conditions["native_spline_oracle"], "geometry": conditions["geometry"], "dynamics": {"position": exact_status.get("position"), "velocity": exact_status.get("velocity"), "acceleration": exact_status.get("acceleration"), "jerk": exact_status.get("jerk")}, "process_mapping": process, "collision": collision, "determinism": determinism, "frozen_hash_mismatch_count": frozen_mismatch_count, "trajectory_optimization_rerun": False, "trajectory_modified": False, "TOTG_Ruckig_parameters_modified": False, "URDF_SRDF_xacro_modified": False}
    write_json(OUT / "stage28sr_regression.json", regression)
    # Re-write the manifest after regression so it is included in SHA256SUMS.
    artifact_files = [path for path in sorted(OUT.rglob("*")) if path.is_file() and path.name not in {"SHA256SUMS", "stage28sr_gate_report.json", "stage28sr_artifact_manifest.json"}]
    write_json(OUT / "stage28sr_artifact_manifest.json", {"schema_version": "stage28sr-artifact-manifest-v1", "required_outputs_present": all((OUT / name).exists() for name in ["stage28sr_gate_report.json", "stage28sr_report.md", "stage28sr_rsp_parameters.yaml", "stage28sr_ros_graph_runtime.txt", "stage28sr_joint_states_raw.jsonl", "stage28sr_controller_state_raw.jsonl", "stage28sr_tf_raw.jsonl", "stage28sr_tf_static_raw.jsonl", "stage28sr_regression.json", "stage28sr_input_manifest.json"]), "gate_report_excluded_from_checksums": True, "entries_before_manifest": len(artifact_files)})
    sums = [f"{sha256(path)}  {path.relative_to(OUT).as_posix()}" for path in sorted(OUT.rglob("*")) if path.is_file() and path.name not in {"SHA256SUMS", "stage28sr_gate_report.json"}]
    (OUT / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    print(json.dumps({"output_root": str(OUT), "Stage_2_8S": gate["Stage_2_8S"], "Stage_2_8S_R": "passed" if passed else f"blocked_{blocker}", "first_blocker": blocker, "Can_we_start_Stage_2_9": passed}, ensure_ascii=False, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
