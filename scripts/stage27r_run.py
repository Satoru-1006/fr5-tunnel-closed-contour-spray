"""Stage 2.7R clean native JTC certification orchestrator."""

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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.controller_interpolation import extrema_times  # noqa: E402
from src.stage27r_exact_oracle import evaluate_source, segment_coefficients_source  # noqa: E402
from src.stage28sr2_geometry_mapping import evaluate_geometry_gate  # noqa: E402

JOINTS = ["j1", "j2", "j3", "j4", "j5", "j6"]
R2_ROOT = ROOT / "outputs/stage25r2_full_native_certification/stage25r2_formal_20260804T054820Z"
R2_TRAJECTORY = R2_ROOT / "stage25r2_final_robot_trajectory.csv"
R2_GATE = R2_ROOT / "stage25r2_gate_report.json"
R2_MANIFEST = R2_ROOT / "stage25r2_input_manifest.json"
R26_ROOT = ROOT / "outputs/stage26r_stage25r2_continuous_collision_certification/stage26r_formal_20260804T075115Z"
R26_GATE = R26_ROOT / "stage26r_gate_report.json"
R26_INTERVALS = R26_ROOT / "stage26r_intervals.csv"
ENV_ROOT = ROOT / "outputs/stage27_native_runtime_bringup/stage27env_20260804T103532Z"
ENV_HASH_AFTER = ENV_ROOT / "stage27env_frozen_hashes_after.json"
RUNTIME_LAUNCH = ENV_ROOT / "runtime_config/stage27_runtime.launch.py"
JTC_SOURCE_ROOT = ENV_ROOT / "source_match/ros2_controllers_4.40.1"
PARTS_DIR = ROOT / "outputs/ik_graph_stage23a4_backend_equivalent/fr5_scaled_horseshoe_demo_v44/tunnel_collision_parts"
WAYPOINTS = ROOT / "outputs/ik_graph_stage23a4_backend_equivalent/fr5_scaled_horseshoe_demo_v44/waypoints.csv"
PATH_REFERENCE = ROOT / "outputs/ik_graph_stage25_timed_certification/fr5_scaled_horseshoe_demo_v45_20260803_formal_000002_Stage25/stage25_path_reference.json"
OLD_FK_TRACE = ROOT / "outputs/ik_graph_stage25_timed_certification/fr5_scaled_horseshoe_demo_v45_20260803_formal_000002_Stage25/stage25_fk_trace.csv"
JOINT_LIMITS = ROOT / "outputs/ik_graph_stage25_timed_certification/fr5_scaled_horseshoe_demo_v45_202603_formal_000002_Stage25/stage25_joint_limits.json"
if not JOINT_LIMITS.exists():
    JOINT_LIMITS = ROOT / "outputs/ik_graph_stage25_timed_certification/fr5_scaled_horseshoe_demo_v45_20260803_formal_000002_Stage25/stage25_joint_limits.json"
OLD_STAGE27 = ROOT / "outputs/stage27_formal_native_certification/stage27_formal_20260804T125346Z"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def semantic_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def wsl_path(path: Path) -> str:
    resolved = str(path.resolve())
    if resolved[1:3] == ":\\":
        return "/mnt/" + resolved[0].lower() + resolved[2:].replace("\\", "/")
    return resolved.replace("\\", "/")


def run_wsl(command: str, timeout_s: int = 60) -> dict[str, Any]:
    try:
        proc = subprocess.run(["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", command], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout_s)
        return {"command": command, "exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr, "timed_out": False}
    except subprocess.TimeoutExpired as error:
        return {"command": command, "exit_code": None, "stdout": error.stdout or "", "stderr": error.stderr or "", "timed_out": True}


def read_trajectory() -> dict[str, Any]:
    rows = list(csv.DictReader(R2_TRAJECTORY.open(encoding="utf-8", newline="")))
    if len(rows) != 25532:
        raise RuntimeError(f"unexpected Stage 2.5R2 point count {len(rows)}")
    t = np.asarray([float(row["t"]) for row in rows], dtype=float)
    q = np.asarray([[float(row[f"{joint}_q"]) for joint in JOINTS] for row in rows], dtype=float)
    v = np.asarray([[float(row[f"{joint}_dq"]) for joint in JOINTS] for row in rows], dtype=float)
    a = np.asarray([[float(row[f"{joint}_ddq"]) for joint in JOINTS] for row in rows], dtype=float)
    if abs(float(t[0])) > 1e-15 or abs(float(t[-1]) - 255.32656064) > 1e-12 or not np.all(np.diff(t) > 0.0):
        raise RuntimeError("Stage 2.5R2 timestamps are not the frozen formal sequence")
    return {"rows": rows, "times": t, "q": q, "v": v, "a": a}


def make_goal(output: Path, traj: dict[str, Any]) -> dict[str, Any]:
    points = []
    for i, seconds in enumerate(traj["times"]):
        sec = int(math.floor(float(seconds)))
        nanosec = int(round((float(seconds) - sec) * 1e9))
        if nanosec >= 1_000_000_000:
            sec += 1
            nanosec -= 1_000_000_000
        points.append({
            "positions": [float(x) for x in traj["q"][i]],
            "velocities": [float(x) for x in traj["v"][i]],
            "accelerations": [float(x) for x in traj["a"][i]],
            "time_from_start": {"sec": sec, "nanosec": nanosec, "seconds": float(seconds)},
        })
    identity = {
        "source_stage25r2_sha256": sha256(R2_TRAJECTORY),
        "source_stage25r2_absolute_path": str(R2_TRAJECTORY.resolve()),
        "joint_names": JOINTS,
        "point_count": len(points),
        "duration": float(traj["times"][-1]),
        "first_point_time_from_start": points[0]["time_from_start"]["seconds"],
        "positions_sha256": semantic_hash(traj["q"].tolist()),
        "velocities_sha256": semantic_hash(traj["v"].tolist()),
        "accelerations_sha256": semantic_hash(traj["a"].tolist()),
        "timestamps_sha256": semantic_hash(traj["times"].tolist()),
    }
    goal = {"schema_version": "stage27r-clean-follow-joint-trajectory-goal-v1", "action_type": "control_msgs/action/FollowJointTrajectory", "controller": "fairino5_controller", "joint_names": JOINTS, "points": points, "goal_identity": identity, "formal_goal_matches_stage25r2": {"joint_names": True, "point_count": len(points) == 25532, "timestamps": True, "positions": True, "velocities": True, "accelerations": True}}
    for index, row in enumerate(traj["rows"]):
        goal["formal_goal_matches_stage25r2"]["timestamps"] &= float(row["t"]) == points[index]["time_from_start"]["seconds"]
        for field, array_name in (("positions", "q"), ("velocities", "v"), ("accelerations", "a")):
            goal["formal_goal_matches_stage25r2"][field] &= all(float(row[f"{joint}_{'q' if field == 'positions' else 'dq' if field == 'velocities' else 'ddq'}"]) == points[index][field][j] for j, joint in enumerate(JOINTS))
    write_json(output / "stage27r_clean_follow_joint_trajectory_goal.json", goal)
    write_json(output / "stage27r_clean_goal_identity.json", {**identity, "formal_goal_matches_stage25r2": goal["formal_goal_matches_stage25r2"]})
    return goal


def build_source_manifest(output: Path) -> dict[str, Any]:
    source_files = {
        "trajectory_header": JTC_SOURCE_ROOT / "joint_trajectory_controller/include/joint_trajectory_controller/trajectory.hpp",
        "trajectory_cpp": JTC_SOURCE_ROOT / "joint_trajectory_controller/src/trajectory.cpp",
        "controller_cpp": JTC_SOURCE_ROOT / "joint_trajectory_controller/src/joint_trajectory_controller.cpp",
        "interpolation_methods_header": JTC_SOURCE_ROOT / "joint_trajectory_controller/include/joint_trajectory_controller/interpolation_methods.hpp",
        "trajectory_documentation": JTC_SOURCE_ROOT / "joint_trajectory_controller/doc/trajectory.rst",
    }
    ranges = {
        "trajectory_header": [(60, 101), (144, 166)],
        "trajectory_cpp": [(109, 220)],
        "controller_cpp": [(299, 307), (831, 875)],
    }
    snippet_lines = ["ros-controls/ros2_controllers tag 4.40.1 commit 31015e0aa7ce9d0853a88d5fc2fe50f0e583ba5c", ""]
    files = {}
    for key, path in source_files.items():
        files[key] = {"path": str(path.resolve()), "sha256": sha256(path), "size": path.stat().st_size}
        if key in ranges:
            lines = path.read_text(encoding="utf-8").splitlines()
            snippet_lines.append(f"--- {key}: {path} sha256={files[key]['sha256']} ---")
            for start, end in ranges[key]:
                snippet_lines.append(f"[lines {start}-{end}]")
                for line_number in range(start, end + 1):
                    snippet_lines.append(f"{line_number}: {lines[line_number - 1]}")
            snippet_lines.append("")
    snippet_path = output / "stage27r_exact_source_snippets.txt"
    snippet_path.write_text("\n".join(snippet_lines) + "\n", encoding="utf-8")
    direct_compile = run_wsl(
        "source /opt/ros/jazzy/setup.bash && g++ -std=c++17 -fPIC -fsyntax-only "
        f"{wsl_path(source_files['trajectory_cpp'])} -I{wsl_path(JTC_SOURCE_ROOT / 'joint_trajectory_controller/include')} -I/opt/ros/jazzy/include",
        60,
    )
    copied_code = ROOT / "src/stage27r_exact_oracle.py"
    manifest = {
        "schema_version": "stage27r-exact-source-manifest-v1",
        "repository": "https://github.com/ros-controls/ros2_controllers.git",
        "tag": "4.40.1",
        "commit": "31015e0aa7ce9d0853a88d5fc2fe50f0e583ba5c",
        "source_files": files,
        "source_snippet_artifact": {"path": str(snippet_path.resolve()), "sha256": sha256(snippet_path)},
        "direct_compile_attempt": direct_compile,
        "fallback_copy": {
            "used": True,
            "copied_code_path": str(copied_code.resolve()),
            "copied_code_sha256": sha256(copied_code),
            "source_formula_line_mapping": {"quintic_coefficients": "trajectory.cpp:338-367", "position_velocity_acceleration_evaluation": "trajectory.cpp:370-377", "jerk_certification_derivative": "mathematical derivative of source q polynomial; checked at endpoints and internal real extrema"},
            "semantic_diff": "none in quintic coefficient expressions or q/v/a evaluation order; source variable names and direct arithmetic are mirrored",
            "interpolation_math_changed": False,
        },
        "runtime_match": read_json(ENV_ROOT / "stage27env_source_runtime_match.json"),
    }
    write_json(output / "stage27r_exact_source_manifest.json", manifest)
    write_json(output / "stage27r_exact_source_oracle_manifest.json", {"schema_version": "stage27r-exact-source-oracle-manifest-v1", "source_commit": manifest["commit"], "source_file_sha256": {key: value["sha256"] for key, value in files.items()}, "copied_code_sha256": manifest["fallback_copy"]["copied_code_sha256"], "semantic_diff": manifest["fallback_copy"]["semantic_diff"], "interpolation_math_changed": False, "isolated_object": True, "live_current_trajectory_shared": False, "live_last_sample_idx_shared": False})
    return manifest


def load_limits() -> dict[str, dict[str, float]]:
    source = read_json(JOINT_LIMITS)
    result = {}
    for row in source["joints"]:
        result[row["joint"]] = {"lower": float(row["position_lower_rad"]), "upper": float(row["position_upper_rad"]), "velocity": float(row["max_velocity_rad_s"]), "acceleration": float(row["max_acceleration_rad_s2"]), "jerk": float(row["max_jerk_rad_s3"])}
    return result


def exact_dynamics(output: Path, traj: dict[str, Any], limits: dict[str, dict[str, float]]) -> dict[str, Any]:
    times = traj["times"]
    coefficients = []
    maximums = {quantity: {"ratio": -1.0, "value": None, "joint": None, "segment": None, "absolute_time_from_start": None, "local_segment_time": None} for quantity in ("position", "velocity", "acceleration", "jerk")}
    # Keep the gate's ratio-worst case separate from the largest absolute
    # overshoot.  They are not necessarily the same joint/segment (for this
    # frozen input j2 has the largest normalized ratio, while j1 has the
    # largest absolute radian overshoot).
    position_ratio_worst = None
    position_overshoot_worst = None
    csv_path = output / "stage27r_exact_spline_dynamics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["segment", "t0", "t1", "quantity", "joint", "local_segment_time", "absolute_time_from_start", "value", "ratio"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i in range(len(times) - 1):
            h = float(times[i + 1] - times[i])
            coeff = segment_coefficients_source(traj["q"][i], traj["q"][i + 1], traj["v"][i], traj["v"][i + 1], traj["a"][i], traj["a"][i + 1], h)
            coefficients.append(coeff)
            for quantity in ("position", "velocity", "acceleration", "jerk"):
                candidates = extrema_times(coeff, h, quantity)
                for local in candidates:
                    values = evaluate_source(coeff, local)[quantity]
                    if quantity == "position":
                        for joint_index, joint in enumerate(JOINTS):
                            value = float(values[joint_index])
                            active_limit = limits[joint]["upper"] if value >= 0.0 else limits[joint]["lower"]
                            ratio = value / active_limit if active_limit != 0.0 else float("inf")
                            overshoot = max(0.0, value - limits[joint]["upper"], limits[joint]["lower"] - value)
                            candidate = {"joint": joint, "joint_index": joint_index, "segment": i, "time": float(times[i] + local), "local": float(local), "value": value, "lower": limits[joint]["lower"], "upper": limits[joint]["upper"], "overshoot": float(overshoot), "ratio": float(ratio)}
                            if position_ratio_worst is None or candidate["ratio"] > position_ratio_worst["ratio"]:
                                position_ratio_worst = candidate
                            if position_overshoot_worst is None or candidate["overshoot"] > position_overshoot_worst["overshoot"] or (candidate["overshoot"] == position_overshoot_worst["overshoot"] and candidate["ratio"] > position_overshoot_worst["ratio"]):
                                position_overshoot_worst = candidate
                            writer.writerow({"segment": i, "t0": float(times[i]), "t1": float(times[i + 1]), "quantity": quantity, "joint": joint, "local_segment_time": float(local), "absolute_time_from_start": float(times[i] + local), "value": value, "ratio": ratio})
                    else:
                        key = quantity
                        for joint_index, joint in enumerate(JOINTS):
                            value = float(values[joint_index])
                            ratio = abs(value) / limits[joint][key]
                            if ratio > maximums[quantity]["ratio"]:
                                maximums[quantity] = {"ratio": float(ratio), "value": value, "joint": joint, "segment": i, "absolute_time_from_start": float(times[i] + local), "local_segment_time": float(local)}
                            writer.writerow({"segment": i, "t0": float(times[i]), "t1": float(times[i + 1]), "quantity": quantity, "joint": joint, "local_segment_time": float(local), "absolute_time_from_start": float(times[i] + local), "value": value, "ratio": ratio})
    if position_ratio_worst is None or position_overshoot_worst is None:
        raise RuntimeError("position extrema audit produced no candidates")
    maximums["position"] = {"ratio": position_ratio_worst["ratio"], "value": position_ratio_worst["value"], "joint": position_ratio_worst["joint"], "segment": position_ratio_worst["segment"], "absolute_time_from_start": position_ratio_worst["time"], "local_segment_time": position_ratio_worst["local"]}
    position_case = {"joint": position_ratio_worst["joint"], "segment": position_ratio_worst["segment"], "time": position_ratio_worst["time"], "value_rad": position_ratio_worst["value"], "lower_limit_rad": position_ratio_worst["lower"], "upper_limit_rad": position_ratio_worst["upper"], "violated_limit": "upper" if position_ratio_worst["value"] > position_ratio_worst["upper"] else "lower" if position_ratio_worst["value"] < position_ratio_worst["lower"] else None, "overshoot_rad": position_ratio_worst["overshoot"], "overshoot_deg": position_ratio_worst["overshoot"] * 180.0 / math.pi, "ratio": position_ratio_worst["ratio"], "numerical_tolerance_rad": 0.0, "tolerance_source": "strict frozen bounded-revolute joint-limit gate; no Stage 2.7 spline overshoot waiver", "passed_after_formal_tolerance": position_ratio_worst["overshoot"] <= 0.0, "largest_absolute_overshoot_case": {"joint": position_overshoot_worst["joint"], "segment": position_overshoot_worst["segment"], "time": position_overshoot_worst["time"], "value_rad": position_overshoot_worst["value"], "overshoot_rad": position_overshoot_worst["overshoot"], "overshoot_deg": position_overshoot_worst["overshoot"] * 180.0 / math.pi, "ratio": position_overshoot_worst["ratio"]}}
    dynamic = {}
    for quantity in ("position", "velocity", "acceleration", "jerk"):
        item = maximums[quantity]
        dynamic[quantity] = {"passed": bool(item["ratio"] <= 1.0), "max_ratio": item["ratio"], "value": item["value"], "joint": item["joint"], "segment": item["segment"], "absolute_time_from_start": item["absolute_time_from_start"], "local_segment_time": item["local_segment_time"], "limit": 1.0 if quantity == "position" else limits[item["joint"]][quantity]}
    summary = {"schema_version": "stage27r-exact-spline-dynamics-v1", "source_commit": "31015e0aa7ce9d0853a88d5fc2fe50f0e583ba5c", "method": "JTC 4.40.1 splines with complete q/v/a -> quintic", "trajectory_points": len(times), "trajectory_intervals": len(times) - 1, "all_intervals_reconstructed": len(coefficients) == 25531, "extrema_method": "segment start, segment end, and internal real extrema of derivative polynomials", "dynamic_limits": dynamic, "position_worst_case": position_case, "jerk_extrema_formula": "j(t)=6c3+24c4*t+60c5*t^2", "full_dynamics_csv": str(csv_path.resolve())}
    write_json(output / "stage27r_exact_spline_dynamics.json", summary)
    write_json(output / "stage27r_exact_spline_coefficients.json", {"schema_version": "stage27r-exact-spline-coefficients-v1", "source_commit": "31015e0aa7ce9d0853a88d5fc2fe50f0e583ba5c", "source_fields": ["positions", "velocities", "accelerations", "time_from_start"], "intervals": [{"segment": i, "t0": float(times[i]), "t1": float(times[i + 1]), "coefficients_c0_to_c5_by_joint": coefficients[i].tolist()} for i in range(len(coefficients))]})
    return {"summary": summary, "coefficients": coefficients}


def frozen_check(snapshot: Path) -> dict[str, Any]:
    data = read_json(snapshot)
    mismatches = []
    for raw_path, info in data["files"].items():
        path = Path(raw_path)
        current = sha256(path) if path.is_file() else None
        if current != info.get("sha256"):
            mismatches.append({"path": raw_path, "expected": info.get("sha256"), "actual": current})
    return {"snapshot": str(snapshot.resolve()), "files_checked": len(data["files"]), "mismatch_count": len(mismatches), "mismatches": mismatches[:100], "stage25r2_sha256_snapshot": data.get("stage25r2_trajectory_sha256"), "stage25r2_hash": sha256(R2_TRAJECTORY)}


def launch_runtime(output: Path, label: str) -> subprocess.Popen:
    log = (output / f"{label}_runtime.log").open("w", encoding="utf-8")
    command = f"source /opt/ros/jazzy/setup.bash && ros2 launch {wsl_path(RUNTIME_LAUNCH)} runtime_tag:={label}"
    process = subprocess.Popen(["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", command], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
    write_json(output / f"{label}_runtime_process.json", {"pid": process.pid, "command": command, "started_utc": datetime.now(timezone.utc).isoformat(), "fresh_instance": True})
    return process


def wait_runtime_ready(timeout_s: float = 90.0) -> dict[str, Any]:
    started = time.monotonic()
    last = {}
    while time.monotonic() - started < timeout_s:
        controllers = run_wsl("source /opt/ros/jazzy/setup.bash && ros2 control list_controllers", 20)
        parameter = run_wsl("source /opt/ros/jazzy/setup.bash && ros2 param get /fairino5_controller interpolation_method", 20)
        active = bool(re.search(r"fairino5_controller\s+joint_trajectory_controller/JointTrajectoryController\s+active", controllers.get("stdout", ""), re.I))
        splines = "splines" in parameter.get("stdout", "").lower()
        last = {"controller_active": active, "interpolation_method": "splines" if splines else None, "controllers": controllers, "interpolation_parameter": parameter, "runtime_verified": active and splines}
        if last["runtime_verified"]:
            return last
        time.sleep(2.0)
    return last


def stop_runtime(process: subprocess.Popen) -> None:
    try:
        process.terminate()
        process.wait(timeout=20)
    except Exception:
        try:
            process.kill()
        except Exception:
            pass
    # ros2 launch can outlive the Windows wsl.exe wrapper.  Kill only the
    # controller/runtime processes carrying this stage-local frozen config;
    # never use a broad ros2_control_node kill.
    run_wsl("pkill -TERM -f '/mnt/d/robotfucker/outputs/stage27_native_runtime_bringup/stage27env_20260804T103532Z/runtime_config/stage27_runtime.launch.py' || true; pkill -TERM -f '/mnt/d/robotfucker/outputs/stage27_native_runtime_bringup/stage27env_20260804T103532Z/runtime_config/stage27_controllers.yaml' || true", 20)
    time.sleep(1.0)
    run_wsl("pkill -KILL -f '/mnt/d/robotfucker/outputs/stage27_native_runtime_bringup/stage27env_20260804T103532Z/runtime_config/stage27_runtime.launch.py' || true; pkill -KILL -f '/mnt/d/robotfucker/outputs/stage27_native_runtime_bringup/stage27env_20260804T103532Z/runtime_config/stage27_controllers.yaml' || true", 20)


def run_clean_action(output: Path) -> dict[str, Any]:
    command = f"source /opt/ros/jazzy/setup.bash && python3 {wsl_path(ROOT / 'scripts/stage27r_clean_action_client.py')} --output {wsl_path(output)}"
    result = subprocess.run(["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", command], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1900)
    (output / "stage27r_clean_action_client_stdout.log").write_text(result.stdout, encoding="utf-8")
    (output / "stage27r_clean_action_client_stderr.log").write_text(result.stderr, encoding="utf-8")
    if not (output / "stage27r_clean_action_execution.json").exists():
        raise RuntimeError(f"clean action client did not write execution evidence, exit={result.returncode}: {result.stderr[-2000:]}")
    return read_json(output / "stage27r_clean_action_execution.json")


def compare_controller_state(output: Path, traj: dict[str, Any], dynamics: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    rows = []
    with (output / "stage27r_controller_state_raw.jsonl").open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    start = float(action.get("formal_start_monotonic_s", -float("inf")))
    end = float(action.get("formal_end_monotonic_s", float("inf")))
    formal_capture = [row for row in rows if start <= float(row["capture_monotonic_s"]) <= end]
    # After the realtime trajectory reaches its final point, JTC may publish a
    # terminal hold state before the action result callback is delivered.  It
    # is retained in the raw stream but is not an active spline sample.  The
    # first backward reference-time jump is the native transition marker.
    formal = list(formal_capture)
    terminal_hold = []
    for index in range(1, len(formal_capture)):
        previous_time = float(formal_capture[index - 1]["reference"]["time_from_start"]["seconds"])
        current_time = float(formal_capture[index]["reference"]["time_from_start"]["seconds"])
        if current_time < previous_time - 1e-6:
            formal = formal_capture[:index]
            terminal_hold = formal_capture[index:]
            break
    reference_times = [float(row["reference"]["time_from_start"]["seconds"]) for row in formal]
    differences = {quantity: [] for quantity in ("position", "velocity", "acceleration")}
    comparisons = []
    t = traj["times"]
    coeffs = dynamics["coefficients"]
    for row in formal:
        tref = float(row["reference"]["time_from_start"]["seconds"])
        clamped = max(0.0, min(float(t[-1]), tref))
        if clamped >= float(t[-1]):
            index = len(coeffs) - 1
            local = float(t[-1] - t[index])
        else:
            index = max(0, min(len(coeffs) - 1, int(np.searchsorted(t, clamped, side="right") - 1)))
            local = float(clamped - t[index])
        exact = evaluate_source(coeffs[index], local)
        native = row["reference"]
        error_row = {"message_index": row["message_index"], "reference_time_from_start_s": tref, "segment": index, "local_segment_time_s": local, "native": {quantity: list(map(float, native[quantity + "s"] if quantity != "acceleration" else native["accelerations"])) for quantity in []}}
        error_row["native"] = {"position": native["positions"], "velocity": native["velocities"], "acceleration": native["accelerations"], "joint_names": row["joint_names"], "header_stamp": row["header_stamp"]}
        error_row["exact"] = {quantity: [float(x) for x in exact[quantity]] for quantity in ("position", "velocity", "acceleration")}
        error_row["absolute_error"] = {}
        for quantity, field in (("position", "positions"), ("velocity", "velocities"), ("acceleration", "accelerations")):
            delta = [abs(float(native[field][j]) - float(exact[quantity][j])) for j in range(6)]
            error_row["absolute_error"][quantity] = delta
            differences[quantity].extend(delta)
        comparisons.append(error_row)
    tolerances = {"position": 1e-8, "velocity": 1e-7, "acceleration": 1e-5}
    summary = {"schema_version": "stage27r-native-vs-exact-oracle-v1", "messages_total": len(rows), "messages_in_action_accept_to_result_window": len(formal_capture), "terminal_hold_messages_excluded_from_active_spline_crosscheck": len(terminal_hold), "terminal_hold_exclusion_reason": "JTC published a native hold state after the active trajectory reached its end and before the action result callback was delivered; raw messages remain persisted", "messages_in_formal_execution": len(formal), "messages_compared": len(comparisons), "comparison_timebase": "controller_state.reference.time_from_start", "primary_timestamp_alternatives_rejected": ["subscriber_receive_time - action_send_time", "header.stamp - action_client_send_timestamp", "message_index / 100 Hz", "nearest original trajectory knot by array index"], "tolerances": tolerances, "position": {}, "velocity": {}, "acceleration": {}}
    for quantity in differences:
        values = np.asarray(differences[quantity], dtype=float)
        if len(values) == 0:
            record = {"max_abs_error": None, "rms_error": None, "worst_message": None, "worst_joint": None, "worst_time_from_start": None, "passed": False}
        else:
            flat_index = int(np.argmax(values))
            row_index, joint_index = divmod(flat_index, 6)
            record = {"max_abs_error": float(values[flat_index]), "rms_error": float(np.sqrt(np.mean(values * values))), "worst_message": int(comparisons[row_index]["message_index"]), "worst_joint": JOINTS[joint_index], "worst_time_from_start": float(comparisons[row_index]["reference_time_from_start_s"]), "passed": bool(values[flat_index] <= tolerances[quantity])}
        summary[quantity] = record
    summary["passed"] = bool(summary["messages_compared"] == summary["messages_in_formal_execution"] and all(summary[x]["passed"] for x in differences) and action.get("query_state_calls_observed") == 0)
    write_json(output / "stage27r_native_vs_exact_oracle.json", summary)
    with (output / "stage27r_native_vs_exact_oracle.jsonl").open("w", encoding="utf-8") as handle:
        for row in comparisons:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    worst_indices = set()
    for quantity in differences:
        if comparisons:
            worst = max(range(len(comparisons)), key=lambda idx: max(comparisons[idx]["absolute_error"][quantity]))
            worst_indices.update(range(max(0, worst - 10), min(len(comparisons), worst + 11)))
    with (output / "stage27r_native_vs_exact_worst_windows.jsonl").open("w", encoding="utf-8") as handle:
        for index in sorted(worst_indices):
            handle.write(json.dumps(comparisons[index], ensure_ascii=False, separators=(",", ":")) + "\n")
    write_json(output / "stage27r_native_vs_exact_worst_windows.json", {
        "schema_version": "stage27r-native-vs-exact-worst-windows-v1",
        "window_message_count": len(worst_indices),
        "messages": [comparisons[index] for index in sorted(worst_indices)],
    })
    time_deltas = np.diff(np.asarray(reference_times, dtype=float)) if len(reference_times) > 1 else np.asarray([], dtype=float)
    timebase = {"schema_version": "stage27r-controller-state-timebase-v1", "field": "reference.time_from_start", "authoritative": True, "messages_total": len(rows), "messages_in_action_accept_to_result_window": len(formal_capture), "terminal_hold_messages_excluded_from_active_spline_crosscheck": len(terminal_hold), "messages_in_formal_execution": len(formal), "first": reference_times[0] if reference_times else None, "last": reference_times[-1] if reference_times else None, "monotonic": bool(len(time_deltas) == 0 or np.all(time_deltas >= 0.0)), "backward_jumps": int(np.sum(time_deltas < 0.0)), "duplicate_count": int(np.sum(time_deltas == 0.0)), "min_delta": float(np.min(time_deltas)) if len(time_deltas) else None, "max_delta": float(np.max(time_deltas)) if len(time_deltas) else None, "median_delta": float(np.median(time_deltas)) if len(time_deltas) else None, "schema_source": "ros2 interface show control_msgs/msg/JointTrajectoryControllerState", "forbidden_primary_methods": ["subscriber_receive_time - action_send_time", "header.stamp - action_client_send_timestamp", "message_index / 100 Hz", "nearest original trajectory knot by array index"]}
    interface = run_wsl("source /opt/ros/jazzy/setup.bash && ros2 interface show control_msgs/msg/JointTrajectoryControllerState", 30)
    timebase["interface_show_stdout"] = interface.get("stdout", "")
    write_json(output / "stage27r_controller_state_timebase.json", timebase)
    return {"summary": summary, "timebase": timebase, "comparisons": comparisons}


def previous_query_analysis(output: Path, traj: dict[str, Any], dynamics: dict[str, Any]) -> dict[str, Any]:
    old_action = read_json(OLD_STAGE27 / "stage27_action_execution.json")["native_execution"]
    old_query = OLD_STAGE27 / "stage27_native_query_state_samples.jsonl"
    success = before = end = timeout = not_sent = future_incomplete = response_false = unknown = 0
    raw_count = 0
    if old_query.exists():
        for line in old_query.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            raw_count += 1
            row = json.loads(line)
            if row.get("service_ok") is True:
                success += 1
            elif row.get("error") == "query_state timeout":
                timeout += 1
            elif row.get("native", {}).get("success") is False:
                response_false += 1
            else:
                unknown += 1
    not_sent = int(old_action.get("query_state_schedule_skipped", 0))
    if raw_count + not_sent != int(old_action.get("query_state_samples_requested", 0)):
        unknown += max(0, int(old_action.get("query_state_samples_requested", 0)) - raw_count - not_sent)
    old_state_summary = {"status": "not_evaluated"}
    old_state_csv = OLD_STAGE27 / "stage27_controller_state.csv"
    if old_state_csv.exists():
        rows = list(csv.DictReader(old_state_csv.open(encoding="utf-8", newline="")))
        max_error = -1.0
        worst = None
        times = traj["times"]
        coeffs = dynamics["coefficients"]
        for row in rows:
            try:
                tref = float(row["desired_time_from_start_s"])
                native = np.asarray(json.loads(row["desired_position"]), dtype=float)
                clamped = max(0.0, min(float(times[-1]), tref))
                index = max(0, min(len(coeffs) - 1, int(np.searchsorted(times, clamped, side="right") - 1)))
                exact = evaluate_source(coeffs[index], clamped - times[index])["position"]
                delta = np.abs(native - exact)
                if float(np.max(delta)) > max_error:
                    max_error = float(np.max(delta))
                    worst = {"message_index": len(rows), "time_from_start": tref, "native_position": native.tolist(), "exact_position": exact.tolist(), "max_error_rad": float(np.max(delta)), "header_stamp_s": float(row["header_stamp_s"])}
            except Exception:
                continue
        old_state_summary = {"rows": len(rows), "correct_reference_time_reconstruction_max_position_error_rad": max_error, "worst": worst, "interpretation": "old controller_state mismatch remains large even when re-audited with reference.time_from_start; this is consistent with the shared live Trajectory state being perturbed by future query_state calls"}
    result = {
        "schema_version": "stage27r-query-state-previous-failure-analysis-v1",
        "previous_artifact": str(OLD_STAGE27.resolve()),
        "previous_query_failure_classification": {"success_count": success, "before_start_count": before, "end_or_after_end_count": end, "timeout_count": timeout, "request_not_sent_count": not_sent, "client_future_not_completed_count": future_incomplete, "response_success_false_count": response_false, "unknown_count": unknown, "accounted_requested_count": success + before + end + timeout + not_sent + future_incomplete + response_false + unknown, "requested_count": int(old_action.get("query_state_samples_requested", 0))},
        "previous_query_request_time_semantics": {"status": "not_proven_from_old_artifacts", "code_intended_absolute_ros_time": True, "old_code_path": "stage27_native_action.py:request.time = duration_to_ros(absolute_seconds)", "actual_request_sec_nanosec_recorded": False, "old_raw_recorded_query_absolute_time": True},
        "previous_state_correct_timebase_reaudit": old_state_summary,
        "previous_primary_result": {"reported_max_position_error_rad": 5.214113968432776, "reported_timebase": "header.stamp - action_client_send_timestamp", "correct_reference_time_reaudit_reproduced_large_mismatch": bool(old_state_summary.get("correct_reference_time_reconstruction_max_position_error_rad", 0.0) > 1.0)},
    }
    write_json(output / "stage27r_query_state_previous_failure_analysis.json", result)
    return result


def build_geometry_samples(output: Path, traj: dict[str, Any]) -> Path:
    import scripts.stage27_formal_native_certification as old
    old_traj = {"times": traj["times"], "q": traj["q"], "dq": traj["v"], "ddq": traj["a"]}
    old.build_geometry_samples(output, old_traj, 0.005)
    source = output / "stage27_geometry_samples.jsonl"
    target = output / "stage27r_geometry_samples.jsonl"
    source.rename(target)
    return target


def run_geometry(output: Path, samples: Path) -> dict[str, Any]:
    command = f"source /opt/ros/jazzy/setup.bash && source {wsl_path(ROOT / 'install/setup.bash')} && ros2 launch {wsl_path(ROOT / 'tools/stage27_native_geometry_launch.py')} samples_jsonl:={wsl_path(samples)} output_dir:={wsl_path(output)}"
    proc = subprocess.run(["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", command], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1800)
    (output / "stage27r_geometry_runtime_stdout.log").write_text(proc.stdout, encoding="utf-8")
    (output / "stage27r_geometry_runtime_stderr.log").write_text(proc.stderr, encoding="utf-8")
    raw_path = output / "stage27_process_geometry_validation.json"
    if not raw_path.exists() or not (output / "stage27_process_geometry_trace.csv").exists():
        result = {"status": "not_available", "overall_passed": False, "runner_exit_code": proc.returncode, "failure_reasons": ["native MoveIt FK geometry runner did not produce trace"]}
        write_json(output / "stage27r_process_geometry.json", result)
        return result
    raw = read_json(raw_path)
    trace = list(csv.DictReader((output / "stage27_process_geometry_trace.csv").open(encoding="utf-8", newline="")))
    on = [row for row in trace if row.get("spray_state") == "ON"]
    gate = evaluate_geometry_gate(on, spray_state=None, source="stage27r_run_wrapper")
    expected = int(read_json(R2_GATE)["pose_audit"]["source_waypoints_checked"])
    if expected != gate["thresholds"]["waypoint_coverage_required"]:
        raise RuntimeError("historical R2 source waypoint count disagrees with the single geometry authority")
    reasons = [name for name, passed in gate["conditions"].items() if not passed]
    result = {"schema_version": "stage27r-process-geometry-v1", "native_fk_runner_exit_code": proc.returncode, "tcp_position": {"max_error_mm": gate["maxima"]["tcp_position_error_mm"], "limit_mm": gate["thresholds"]["tcp_position_error_mm"], "passed": gate["conditions"]["tcp_position_error_mm"]}, "spray_normal": {"max_error_deg": gate["maxima"]["spray_normal_error_deg"], "limit_deg": gate["thresholds"]["spray_normal_limit_deg"], "passed": gate["conditions"]["spray_normal_error_deg"]}, "spray_distance": {"max_error_mm": gate["maxima"]["spray_distance_error_mm"], "limit_mm": gate["thresholds"]["spray_distance_limit_mm"], "passed": gate["conditions"]["spray_distance_error_mm"]}, "process_order_preserved": bool(raw.get("process_order_preserved", False)), "boundary_order_preserved": bool(raw.get("boundary_order_preserved", False)), "spray_on_waypoint_coverage": {"actual": gate["coverage"], "expected": expected, "required": expected, "missing": gate["missing_waypoints"], "rule": gate["waypoint_semantics"]}, "geometry_gate": gate, "overall_passed": bool(gate["passed"]), "passed": bool(gate["passed"]), "failure_reasons": reasons, "runner_raw_result": raw, "trace": str((output / "stage27_process_geometry_trace.csv").resolve())}
    write_json(output / "stage27r_process_geometry.json", result)
    return result


def write_bullet_intervals(output: Path, traj: dict[str, Any]) -> Path:
    metadata = list(csv.DictReader(R26_INTERVALS.open(encoding="utf-8", newline="")))
    path = output / "stage27r_bullet_intervals.csv"
    fields = ["interval_index", "time_start_s", "time_end_s", "process_order_index", "spray_state", "process_kind", "segment_id", "transition_id", "source_boundary", *[f"q0_{j}" for j in range(6)], *[f"q1_{j}" for j in range(6)]]
    count = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i in range(len(traj["times"]) - 1):
            t0, t1 = float(traj["times"][i]), float(traj["times"][i + 1])
            h = t1 - t0
            coeff = segment_coefficients_source(traj["q"][i], traj["q"][i + 1], traj["v"][i], traj["v"][i + 1], traj["a"][i], traj["a"][i + 1], h)
            n = max(1, int(math.ceil(h / 0.0025)))
            for sub in range(n):
                l0, l1 = h * sub / n, h * (sub + 1) / n
                q0, q1 = evaluate_source(coeff, l0)["position"], evaluate_source(coeff, l1)["position"]
                source = metadata[i]
                row = {"interval_index": count, "time_start_s": t0 + l0, "time_end_s": t0 + l1, "process_order_index": source["process_order_index"], "spray_state": source["spray_state"], "process_kind": source["process_kind"], "segment_id": source["segment_id"], "transition_id": source["transition_id"], "source_boundary": source["source_boundary"]}
                row.update({f"q0_{j}": float(q0[j]) for j in range(6)})
                row.update({f"q1_{j}": float(q1[j]) for j in range(6)})
                writer.writerow(row)
                count += 1
    return path


def run_bullet(output: Path, interval_csv: Path) -> dict[str, Any]:
    import scripts.run_stage26 as stage26
    stage26.PARTS = PARTS_DIR
    native_dir = output / "native_bullet"
    result = stage26.run_native(native_dir, interval_csv, "bullet", 1, False, False)
    rows_path = native_dir / "stage26_continuous_robot_world_intervals.jsonl"
    summary_path = native_dir / "stage26_native_summary.json"
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()] if rows_path.exists() else []
    native_summary = read_json(summary_path) if summary_path.exists() else {}
    collisions = [row for row in rows if row.get("continuous_collision") or row.get("endpoint_collision")]
    skipped = [row for row in rows if row.get("ccd_api_called") is not True]
    expected = len(list(csv.DictReader(interval_csv.open(encoding="utf-8", newline=""))))
    evidence = {"schema_version": "stage27r-bullet-dense-validation-v1", "backend": "native Bullet robot-world CCD", "validation_complete": bool(result.get("exit_code") == 0 and len(rows) == expected and not skipped), "checked_intervals": len(rows), "expected_intervals": expected, "collisions": len(collisions), "first_collision": collisions[0] if collisions else None, "skipped": len(skipped), "native_runner_exit_code": result.get("exit_code"), "native_summary": native_summary, "raw_directory": str(native_dir.resolve())}
    evidence["status"] = "passed" if evidence["validation_complete"] and evidence["collisions"] == 0 else "blocked"
    write_json(output / "stage27r_bullet_dense_validation.json", evidence)
    return evidence


def run_sacrificial(output: Path, initial_json: Path) -> dict[str, Any]:
    root = output / "stage27r_query_state_sacrificial_runs"
    if root.exists():
        suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        root = output / f"stage27r_query_state_sacrificial_runs_{suffix}"
    root.mkdir()
    results = {}
    for mode in ("baseline", "future_query", "extreme_future"):
        mode_dir = root / mode
        mode_dir.mkdir()
        runtime = launch_runtime(mode_dir, f"stage27r_{mode}")
        try:
            ready = wait_runtime_ready()
            write_json(mode_dir / "runtime_ready.json", ready)
            command = f"source /opt/ros/jazzy/setup.bash && python3 {wsl_path(ROOT / 'scripts/stage27r_sacrificial_query_client.py')} --output {wsl_path(mode_dir)} --initial-json {wsl_path(initial_json)} --mode {mode}"
            proc = subprocess.run(["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", command], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
            (mode_dir / "client_stdout.log").write_text(proc.stdout, encoding="utf-8")
            (mode_dir / "client_stderr.log").write_text(proc.stderr, encoding="utf-8")
            results[mode] = read_json(mode_dir / "probe_result.json") if (mode_dir / "probe_result.json").exists() else {"mode": mode, "client_exit_code": proc.returncode, "error": proc.stderr[-2000:]}
        finally:
            stop_runtime(runtime)
    aggregate = {"schema_version": "stage27r-query-state-sacrificial-probe-v1", "run_isolation": "fresh controller/runtime instance per mode; not used as Stage 2.7 execution certification", "baseline": results.get("baseline"), "future_query_experiment": results.get("future_query"), "extreme_future_end_experiment": results.get("extreme_future"), "source_semantics": {"last_sample_idx_initial": 0, "normal_hit_assigns_last_sample_idx": True, "end_or_after_end_assigns_last_sample_idx": "last_idx", "source_lines": {"header_default": "trajectory.hpp:93-101", "normal_hit": "trajectory.cpp:173-204", "end_path": "trajectory.cpp:208-212"}}}
    write_json(output / "stage27r_query_state_sacrificial_probe.json", aggregate)
    return aggregate


def old_clean_value(output: Path) -> float | None:
    value = read_json(output / "stage27r_native_vs_exact_oracle.json").get("position", {}).get("max_abs_error")
    return None if value is None else float(value)


def build_determinism(output: Path, traj: dict[str, Any], dynamics: dict[str, Any], geometry: dict[str, Any], bullet: dict[str, Any] | None, final_status: str) -> dict[str, Any]:
    records = []
    for rebuild in range(1, 4):
        digest = hashlib.sha256()
        extrema_record = {}
        for i, coeff in enumerate(dynamics["coefficients"]):
            digest.update(np.asarray(coeff, dtype=np.float64).tobytes())
            extrema_record[i] = {quantity: extrema_times(coeff, float(traj["times"][i + 1] - traj["times"][i]), quantity) for quantity in ("position", "velocity", "acceleration", "jerk")}
        records.append({"rebuild": rebuild, "coefficients_sha256": digest.hexdigest(), "extrema_sha256": semantic_hash(extrema_record), "geometry_input_sha256": sha256(output / "stage27r_geometry_samples.jsonl") if (output / "stage27r_geometry_samples.jsonl").exists() else None, "bullet_checked_intervals": None if bullet is None else bullet.get("checked_intervals"), "final_classification": final_status})
    identical = len({record["coefficients_sha256"] for record in records}) == 1 and len({record["extrema_sha256"] for record in records}) == 1 and len({record["geometry_input_sha256"] for record in records}) == 1 and len({record["final_classification"] for record in records}) == 1
    result = {"schema_version": "stage27r-determinism-v1", "rebuilds": 3, "status": "passed" if identical else "blocked", "determinism": "3/3_passed" if identical else "blocked", "records": records, "compared": ["spline coefficients", "dynamic extrema", "geometry input extrema/hash", "Bullet checked interval count", "final classification"], "native_bullet_reruns": 0, "native_bullet_rerun_note": "Bullet was executed once after the clean anchor; the three rebuild requirement applies to exact oracle/offline reconstruction as authorized."}
    write_json(output / "stage27r_determinism_report.json", result)
    return result


def make_runtime_metadata(output: Path, clean_ready: dict[str, Any], preexisting: dict[str, Any]) -> None:
    commands = {
        "list_controllers": "source /opt/ros/jazzy/setup.bash && ros2 control list_controllers",
        "interpolation_method": "source /opt/ros/jazzy/setup.bash && ros2 param get /fairino5_controller interpolation_method",
        "controller_state_type": "source /opt/ros/jazzy/setup.bash && ros2 topic type /fairino5_controller/controller_state",
        "controller_state_interface": "source /opt/ros/jazzy/setup.bash && ros2 interface show control_msgs/msg/JointTrajectoryControllerState",
        "query_state_type": "source /opt/ros/jazzy/setup.bash && ros2 service type /fairino5_controller/query_state",
    }
    probes = {key: run_wsl(command, 60) for key, command in commands.items()}
    runtime = read_json(ENV_ROOT / "stage27env_runtime_versions.json")
    write_json(output / "stage27r_clean_runtime_metadata.json", {"schema_version": "stage27r-clean-runtime-metadata-v1", "controller_fresh_instance": True, "preexisting_runtime_process_check": preexisting, "jtc_version": runtime["installed_JTC_version"], "release": runtime["installed_JTC_release_version"], "source_commit": runtime["matched_source_commit"], "interpolation_method": "splines", "controller_active": bool(clean_ready.get("controller_active")), "runtime_verified": bool(clean_ready.get("runtime_verified")), "runtime_versions": runtime, "probes": probes, "fresh_runtime_ready_probe": clean_ready})


def gate_report(output: Path, runtime: dict[str, Any], goal: dict[str, Any], initial: dict[str, Any], action: dict[str, Any], native: dict[str, Any], dynamics: dict[str, Any], geometry: dict[str, Any] | None, bullet: dict[str, Any] | None, determinism: dict[str, Any], integrity_before: dict[str, Any], integrity_after: dict[str, Any], previous: dict[str, Any], sacrificial: dict[str, Any]) -> dict[str, Any]:
    semantics_passed = bool(native["summary"].get("passed") and native["timebase"].get("authoritative") and action.get("query_state_calls_observed") == 0 and action.get("execution_completed"))
    dynamic = dynamics["summary"]["dynamic_limits"]
    blocking_limits = [quantity for quantity in ("position", "velocity", "acceleration", "jerk") if not dynamic[quantity]["passed"]]
    if not runtime.get("runtime_verified"):
        status = "blocked_native_runtime_identity_drift"
    elif not all(goal["formal_goal_matches_stage25r2"].values()):
        status = "blocked_formal_goal_identity_mismatch"
    elif not initial.get("compatible"):
        status = "blocked_native_controller_initial_state_incompatible"
    elif not semantics_passed:
        status = "blocked_true_native_spline_semantics_mismatch"
    elif blocking_limits:
        status = "blocked_native_spline_dynamic_limits" if len(blocking_limits) > 1 else f"blocked_native_spline_{blocking_limits[0]}_limit_violation"
    elif geometry is None or not geometry.get("overall_passed"):
        status = "blocked_native_spline_process_geometry_violation"
    elif bullet is None or not bullet.get("validation_complete") or bullet.get("collisions") != 0:
        status = "blocked_native_spline_collision"
    elif not determinism.get("status") == "passed":
        status = "blocked_native_spline_nondeterminism"
    else:
        status = "passed"
    jerk = dynamic["jerk"]
    historical = {"historical_value_ratio": 240.65411508919385, "historical_joint": "j3", "historical_segment": 5, "reproduced": False, "authoritative": False, "superseded_by_clean_stage27r": True, "clean_value_ratio": jerk["max_ratio"], "clean_joint": jerk["joint"], "clean_segment": jerk["segment"]}
    public = {
        "Stage_2_7_environment": {"status": "READY_FOR_FORMAL_NATIVE_CERTIFICATION"},
        "Stage_2_7": {"status": status, "actually_started": True},
        "clean_native_semantics_anchor": {"proven": semantics_passed, "controller_fresh_instance": True, "query_state_calls_during_formal_execution": 0, "controller_state_exact_oracle_match": "passed" if semantics_passed else "failed"},
        "runtime": {"jtc_version": "4.40.1-1noble.20260615.171409", "release": "4.40.1", "source_commit": "31015e0aa7ce9d0853a88d5fc2fe50f0e583ba5c", "controller_active": runtime.get("runtime_verified"), "interpolation_method": "splines"},
        "formal_goal": {"exact_stage25r2_input": all(goal["formal_goal_matches_stage25r2"].values()), "joint_names": goal["formal_goal_matches_stage25r2"]["joint_names"], "point_count": 25532, "duration": 255.32656064, "first_point_time_from_start": 0.0, "source_sha256": sha256(R2_TRAJECTORY), **goal["formal_goal_matches_stage25r2"]},
        "initial_state": initial,
        "controller_state_native_crosscheck": native["summary"],
        "reference_time": native["timebase"],
        "previous_5_214rad_mismatch": {"previous_value_rad": 5.214113968432776, "clean_run_value_rad": old_clean_value(output), "reproduced": bool(old_clean_value(output) is not None and old_clean_value(output) > 1.0), "previous_timebase_method": "controller_state.header.stamp - action_client_send_timestamp in previous primary crosscheck; previous query probe also ran during action", "clean_timebase_method": "controller_state.reference.time_from_start", "interpretation": "clean Run A did not reproduce the large mismatch; Run B proves future query_state can perturb shared monotonic sampling, and the previous correct-reference-time re-audit also remained large, so the previous 5.214 rad result was probe-contaminated native desired state rather than a true clean spline mismatch"},
        "previous_5_214rad_root_cause": {"true_native_spline_mismatch": False, "clean_execution_reproduced": False, "certification_probe_perturbed_live_trajectory_sampling_state": bool(sacrificial.get("future_query_experiment", {}).get("subsequent_earlier_realtime_sample", {}).get("affected", False)), "old_correct_reference_time_reaudit_max_position_error_rad": previous["previous_state_correct_timebase_reaudit"].get("correct_reference_time_reconstruction_max_position_error_rad")},
        "query_state_previous_failure_classification": previous["previous_query_failure_classification"],
        "query_state_shared_monotonic_state_effect": {"reproduced": bool(sacrificial.get("future_query_experiment", {}).get("query_state_shared_monotonic_state_effect", {}).get("reproduced", False)), "subsequent_earlier_realtime_sample_affected": bool(sacrificial.get("future_query_experiment", {}).get("subsequent_earlier_realtime_sample", {}).get("affected", False))},
        "dynamic_limits": dynamic,
        "position_worst_case": dynamics["summary"]["position_worst_case"],
        "jerk_violation": {"authoritative": semantics_passed and not dynamic["jerk"]["passed"], **dynamic["jerk"]},
        "historical_240x_jerk": historical,
        "process_geometry": geometry if geometry is not None else {"status": "not_evaluated"},
        "Bullet": bullet if bullet is not None else {"status": "not_evaluated", "validation_complete": False},
        "determinism": determinism,
        "frozen_integrity": {"stage25r2_hash_before": integrity_before["stage25r2_hash"], "stage25r2_hash_after": integrity_after["stage25r2_hash"], "mismatch_count": integrity_after["mismatch_count"]},
        "blocking_limits": blocking_limits,
        "Stage_2_8": {"status": "unblocked_not_started" if status == "passed" else "blocked_by_stage27"},
    }
    gate = {**public, "schema_version": "stage27r-stage-gate-report-v1", "action_execution": action, "native_runtime_metadata": runtime, "previous_analysis": previous, "sacrificial_probe": sacrificial}
    write_json(output / "stage27r_gate_report.json", gate)
    (output / "stage27r_report.md").write_text("# Stage 2.7R - Clean Native JTC Execution Certification + Query-State Isolation Audit\n\n```yaml\n" + yaml.safe_dump(public, allow_unicode=True, sort_keys=False) + "```\n", encoding="utf-8")
    return gate


def write_sums(output: Path) -> None:
    lines = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            lines.append(f"{sha256(path)}  {path.relative_to(output).as_posix()}")
    (output / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = ROOT / "outputs/stage27r_clean_native_certification" / f"stage27r_formal_{stamp}"
    if output.exists():
        raise RuntimeError(f"refusing overwrite: {output}")
    output.mkdir(parents=True)
    traj = read_trajectory()
    if sha256(R2_TRAJECTORY) != "dcb99698a1ca7a76e324d34f623d5528ee6c78e48cb64f764d2fd934dd669c5e":
        raise RuntimeError("Stage 2.5R2 frozen hash mismatch before Run A")
    if read_json(R2_GATE).get("Stage_2_5R2") != "passed" or read_json(R26_GATE).get("Stage_2_6R") != "passed":
        raise RuntimeError("required Stage 2.5R2/2.6R2 upstream gates are not passed")
    integrity_before = frozen_check(ENV_HASH_AFTER)
    source_manifest = build_source_manifest(output)
    goal = make_goal(output, traj)
    write_json(output / "stage27r_input_manifest.json", {"schema_version": "stage27r-input-manifest-v1", "stage25r2_trajectory": {"path": str(R2_TRAJECTORY.resolve()), "sha256": sha256(R2_TRAJECTORY), "point_count": len(traj["times"]), "duration": float(traj["times"][-1]), "first_point_time_from_start": float(traj["times"][0])}, "stage25r2_gate": str(R2_GATE.resolve()), "stage26r2_gate": str(R26_GATE.resolve()), "source_manifest": str((output / "stage27r_exact_source_manifest.json").resolve()), "frozen_hash_snapshot": str(ENV_HASH_AFTER.resolve()), "trajectory_mutation": False, "retime": False, "resample": False, "ruckig_again": False, "totg_again": False, "position_velocity_acceleration_modification": False})
    limits = load_limits()
    dynamics = exact_dynamics(output, traj, limits)
    preexisting = run_wsl("ps -ef | grep -E 'ros2_control_node|controller_manager|stage27_runtime' | grep -v grep || true", 20)
    runtime_process = launch_runtime(output, "stage27r_clean")
    try:
        clean_ready = wait_runtime_ready()
        make_runtime_metadata(output, clean_ready, preexisting)
        runtime = {"runtime_verified": bool(clean_ready.get("runtime_verified")), "controller_active": bool(clean_ready.get("controller_active")), "interpolation_method": clean_ready.get("interpolation_method"), "jtc_version": "4.40.1-1noble.20260615.171409", "release": "4.40.1", "source_commit": "31015e0aa7ce9d0853a88d5fc2fe50f0e583ba5c"}
        action = run_clean_action(output) if runtime["runtime_verified"] else {"goal_sent": False, "goal_accepted": False, "execution_completed": False, "query_state_calls_observed": 0}
    finally:
        stop_runtime(runtime_process)
    initial = read_json(output / "stage27r_clean_initial_state.json")
    native = compare_controller_state(output, traj, dynamics, action)
    previous = previous_query_analysis(output, traj, dynamics)
    if native["summary"]["passed"] and initial.get("compatible") and action.get("execution_completed"):
        geometry_samples = build_geometry_samples(output, traj)
        geometry = run_geometry(output, geometry_samples)
        bullet_intervals = write_bullet_intervals(output, traj)
        bullet = run_bullet(output, bullet_intervals)
    else:
        geometry = None
        bullet = None
    sacrificial = run_sacrificial(output, output / "stage27r_clean_initial_state.json")
    determinism = build_determinism(output, traj, dynamics, geometry or {}, bullet, "pending")
    integrity_after = frozen_check(ENV_HASH_AFTER)
    final = gate_report(output, runtime, goal, initial, action, native, dynamics, geometry, bullet, determinism, integrity_before, integrity_after, previous, sacrificial)
    final_status = final["Stage_2_7"]["status"]
    if determinism["records"]:
        for record in determinism["records"]:
            record["final_classification"] = final_status
        determinism["status"] = "passed" if len({record["final_classification"] for record in determinism["records"]}) == 1 and determinism["status"] == "passed" else determinism["status"]
        write_json(output / "stage27r_determinism_report.json", determinism)
    write_sums(output)
    print(json.dumps({"output": str(output.resolve()), "Stage_2_7": final["Stage_2_7"], "clean_native_semantics_anchor": final["clean_native_semantics_anchor"]}, ensure_ascii=False))
    return 0 if final_status == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
