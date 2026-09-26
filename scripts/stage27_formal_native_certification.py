"""Stage 2.7 formal native controller certification orchestrator.

This script is deliberately stage-local.  It consumes the exact Stage 2.5R2
CSV selected by the Stage 2.5R2 and Stage 2.6R manifests, creates a complete
FJT goal without changing the source, runs the native ROS action client,
reconstructs the JTC 4.40.1 spline analytically, calls the existing native
Bullet probe on the reconstructed path, and records a fail-closed gate.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
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

from src.controller_interpolation import (  # noqa: E402
    audit_intervals,
    evaluate_coefficients,
    extrema_times,
    segment_coefficients,
)

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
ENV_HASH_BEFORE = ENV_ROOT / "stage27env_frozen_hashes_before.json"
RUNTIME_LAUNCH = ENV_ROOT / "runtime_config/stage27_runtime.launch.py"
PARTS_DIR = ROOT / "outputs/ik_graph_stage23a4_backend_equivalent/fr5_scaled_horseshoe_demo_v44/tunnel_collision_parts"
WAYPOINTS = ROOT / "outputs/ik_graph_stage23a4_backend_equivalent/fr5_scaled_horseshoe_demo_v44/waypoints.csv"
PATH_REFERENCE = ROOT / "outputs/ik_graph_stage25_timed_certification/fr5_scaled_horseshoe_demo_v45_20260803_formal_000002_Stage25/stage25_path_reference.json"
OLD_FK_TRACE = ROOT / "outputs/ik_graph_stage25_timed_certification/fr5_scaled_horseshoe_demo_v45_20260803_formal_000002_Stage25/stage25_fk_trace.csv"
PROCESS_MANIFEST = ROOT / "outputs/ik_graph_stage25_timed_certification/fr5_scaled_horseshoe_demo_v45_20260803_formal_000002_Stage25/stage25_process_manifest.json"
JOINT_LIMITS = ROOT / "outputs/ik_graph_stage25_timed_certification/fr5_scaled_horseshoe_demo_v45_20260803_formal_000002_Stage25/stage25_joint_limits.json"
CONTROLLER_CONFIG = ROOT / "external/frcobot_ros2/fairino5_v6_moveit2_config/config/ros2_controllers.yaml"
URDF = ROOT / "external/frcobot_ros2/fairino_description/urdf/fairino5_v6.urdf"
MOVEIT_LIMITS = ROOT / "ros2_moveit_bridge/config/joint_limits_with_jerk.yaml"
SRDF = ROOT / "ros2_moveit_bridge/config/fairino5_v6_spray_tcp.srdf"
TCP_XACRO = ROOT / "ros2_moveit_bridge/config/fairino5_v6_spray_tcp.urdf.xacro"
JTC_SOURCE_ROOT = ENV_ROOT / "source_match/ros2_controllers_4.40.1"


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def semantic_sha(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def run_wsl(command: str, timeout_s: int = 60) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            ["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", command],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
        )
        return {"command": command, "exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr, "timed_out": False}
    except subprocess.TimeoutExpired as error:
        return {"command": command, "exit_code": None, "stdout": error.stdout or "", "stderr": error.stderr or "", "timed_out": True}


def read_trajectory() -> dict[str, Any]:
    rows = list(csv.DictReader(R2_TRAJECTORY.open(encoding="utf-8", newline="")))
    if len(rows) != 25532:
        raise RuntimeError(f"Stage 2.5R2 formal trajectory point count is {len(rows)}, expected 25532")
    times = np.asarray([float(row["t"]) for row in rows], dtype=float)
    q = np.asarray([[float(row[f"{j}_q"]) for j in JOINTS] for row in rows], dtype=float)
    dq = np.asarray([[float(row[f"{j}_dq"]) for j in JOINTS] for row in rows], dtype=float)
    ddq = np.asarray([[float(row[f"{j}_ddq"]) for j in JOINTS] for row in rows], dtype=float)
    for key, array in (("t", times), ("q", q), ("dq", dq), ("ddq", ddq)):
        if not np.all(np.isfinite(array)):
            raise RuntimeError(f"Formal trajectory contains non-finite {key}")
    if abs(times[0]) > 1e-15 or not np.all(np.diff(times) > 0.0):
        raise RuntimeError("Formal trajectory timestamps are not strictly increasing from 0")
    return {"rows": rows, "times": times, "q": q, "dq": dq, "ddq": ddq}


def field_hash(values: Any) -> str:
    return semantic_sha(values)


def build_goal(output: Path, traj: dict[str, Any]) -> dict[str, Any]:
    q, dq, ddq, times = traj["q"], traj["dq"], traj["ddq"], traj["times"]
    points = []
    for i, t in enumerate(times):
        sec = int(np.floor(float(t)))
        nanosec = int(round((float(t) - sec) * 1e9))
        if nanosec >= 1_000_000_000:
            sec += 1
            nanosec -= 1_000_000_000
        points.append({
            "positions": [float(x) for x in q[i]],
            "velocities": [float(x) for x in dq[i]],
            "accelerations": [float(x) for x in ddq[i]],
            "time_from_start": {"sec": sec, "nanosec": nanosec, "seconds": float(t)},
        })
    identity = {
        "source_stage25r2_sha256": sha256(R2_TRAJECTORY),
        "source_stage25r2_absolute_path": str(R2_TRAJECTORY.resolve()),
        "point_count": len(points),
        "joint_count": len(JOINTS),
        "duration": float(times[-1]),
        "positions_sha256": field_hash(q.tolist()),
        "velocities_sha256": field_hash(dq.tolist()),
        "accelerations_sha256": field_hash(ddq.tolist()),
        "timestamps_sha256": field_hash([float(x) for x in times]),
    }
    goal = {
        "schema_version": "stage27-follow-joint-trajectory-goal-v1",
        "action_type": "control_msgs/action/FollowJointTrajectory",
        "controller": "fairino5_controller",
        "joint_names": list(JOINTS),
        "points": points,
        "goal_identity": identity,
        "formal_goal_matches_stage25r2": {
            "joint_names": JOINTS == [R2_TRAJECTORY.open(encoding="utf-8").readline().split(f"_{s}")[0] for s in []],
            "point_count": len(points) == 25532,
            "timestamps": True,
            "positions": True,
            "velocities": True,
            "accelerations": True,
        },
    }
    # The source CSV header is the authoritative field-order check.
    expected_header = ["t"] + [f"{j}_{s}" for s in ("q", "dq", "ddq") for j in JOINTS] + [f"{j}_jerk" for j in JOINTS]
    actual_header = list(csv.reader(R2_TRAJECTORY.open(encoding="utf-8", newline=""))).__getitem__(0)
    goal["formal_goal_matches_stage25r2"]["joint_names"] = actual_header[:1] == ["t"] and all(f"{j}_q" in actual_header for j in JOINTS) and all(f"{j}_dq" in actual_header for j in JOINTS) and all(f"{j}_ddq" in actual_header for j in JOINTS)
    source_rows = traj["rows"]
    point_match = True
    for i, row in enumerate(source_rows):
        point_match &= all(float(row[f"{j}_q"]) == points[i]["positions"][k] for k, j in enumerate(JOINTS))
        point_match &= all(float(row[f"{j}_dq"]) == points[i]["velocities"][k] for k, j in enumerate(JOINTS))
        point_match &= all(float(row[f"{j}_ddq"]) == points[i]["accelerations"][k] for k, j in enumerate(JOINTS))
        point_match &= abs(float(row["t"]) - points[i]["time_from_start"]["seconds"]) == 0.0
    goal["formal_goal_matches_stage25r2"].update({"timestamps": point_match, "positions": point_match, "velocities": point_match, "accelerations": point_match})
    write_json(output / "stage27_follow_joint_trajectory_goal.json", goal)
    write_json(output / "stage27_follow_joint_trajectory_goal_hashes.json", {
        "goal_json_sha256": sha256(output / "stage27_follow_joint_trajectory_goal.json"),
        "goal_identity": identity,
        "source_header": actual_header,
        "source_header_expected": expected_header,
    })
    return goal


def load_limits() -> dict[str, Any]:
    report = read_json(JOINT_LIMITS)
    joints = report["joints"]
    if len(joints) != 6 or any(float(x["max_jerk_rad_s3"]) != 8.0 for x in joints):
        raise RuntimeError("Formal Stage 2.5R2 dynamic-limit provenance is incomplete or unexpected")
    result = {
        "schema_version": "stage27-effective-dynamic-limits-v1",
        "source": {
            "stage25_joint_limits": str(JOINT_LIMITS.resolve()),
            "stage25_gate": str(R2_GATE.resolve()),
            "stage25r2_gate": str(R2_ROOT.joinpath("stage25r2_gate_report.json").resolve()),
            "position": "URDF bounded revolute limits as captured by formal Stage 2.5 joint-limits evidence",
            "velocity_acceleration": "joint_limits_with_jerk.yaml MoveIt override as captured by formal Stage 2.5 joint-limits evidence",
            "jerk": "explicit formal Ruckig max_jerk contract, 8.0 rad/s^3 per joint",
        },
        "per_joint": {},
    }
    for item in joints:
        result["per_joint"][item["joint"]] = {
            "position_min": float(item["position_lower_rad"]),
            "position_max": float(item["position_upper_rad"]),
            "max_velocity": float(item["max_velocity_rad_s"]),
            "max_acceleration": float(item["max_acceleration_rad_s2"]),
            "max_jerk": float(item["max_jerk_rad_s3"]),
            "provenance": item["sources"],
        }
    return result


def build_reconstruction(output: Path, traj: dict[str, Any], limits_record: dict[str, Any]) -> dict[str, Any]:
    t, q, dq, ddq = traj["times"], traj["q"], traj["dq"], traj["ddq"]
    limits = [
        {
            "position_lower_rad": limits_record["per_joint"][j]["position_min"],
            "position_upper_rad": limits_record["per_joint"][j]["position_max"],
            "max_velocity_rad_s": limits_record["per_joint"][j]["max_velocity"],
            "max_acceleration_rad_s2": limits_record["per_joint"][j]["max_acceleration"],
            "max_jerk_rad_s3": limits_record["per_joint"][j]["max_jerk"],
        }
        for j in JOINTS
    ]
    audit = audit_intervals(t, q, dq, ddq, "quintic", limits, JOINTS)
    coefficients = []
    with (output / "stage27_exact_spline_dynamics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["interval", "t0", "t1"]
        for quantity in ("position", "velocity", "acceleration", "jerk"):
            fields += [f"{quantity}_ratio", f"{quantity}_value", f"{quantity}_joint", f"{quantity}_time"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i in range(len(t) - 1):
            coeff = segment_coefficients(q[i], q[i + 1], dq[i], dq[i + 1], ddq[i], ddq[i + 1], float(t[i + 1] - t[i]), "quintic")
            coefficients.append({"interval": i, "t0": float(t[i]), "t1": float(t[i + 1]), "coefficients_c0_to_c5_by_joint": coeff.tolist()})
            row = {"interval": i, "t0": float(t[i]), "t1": float(t[i + 1])}
            for quantity in ("position", "velocity", "acceleration", "jerk"):
                item = audit["interval_audits"][i].extrema[quantity]
                row.update({f"{quantity}_ratio": item["ratio"], f"{quantity}_value": item["max_abs_value"], f"{quantity}_joint": item["joint"], f"{quantity}_time": item["global_time_s"]})
            writer.writerow(row)
    write_json(output / "stage27_spline_coefficients.json", {
        "schema_version": "stage27-jtc-4.40.1-spline-coefficients-v1",
        "method": "quintic",
        "source_commit": "31015e0aa7ce9d0853a88d5fc2fe50f0e583ba5c",
        "source_fields": ["positions", "velocities", "accelerations", "time_from_start"],
        "physical_time_polynomial": "q=c0+c1*t+c2*t^2+c3*t^3+c4*t^4+c5*t^5; derivatives analytic",
        "coefficients": coefficients,
    })
    summary = {
        "schema_version": "stage27-exact-spline-dynamics-v1",
        "method": "quintic",
        "trajectory_points": 25532,
        "trajectory_intervals": 25531,
        "all_intervals_reconstructed": audit["all_intervals_reconstructed"],
        "extrema_method": "segment endpoints plus real roots of derivative polynomials inside each physical-time interval",
        "dynamic_validation": {},
        "full_interval_csv": str((output / "stage27_exact_spline_dynamics.csv").resolve()),
    }
    for quantity in ("position", "velocity", "acceleration", "jerk"):
        item = audit["maximums"][quantity]
        summary["dynamic_validation"][quantity] = {
            "passed": audit["status"][quantity] == "passed",
            "max_ratio": item["ratio"],
            "joint": item["worst_joint"],
            "segment": item["worst_interval"],
            "time": item["worst_time"],
            "value": item["value"],
            "limit": (1.0 if quantity == "position" else limits_record["per_joint"][item["worst_joint"]][{"velocity":"max_velocity","acceleration":"max_acceleration","jerk":"max_jerk"}[quantity]]),
        }
    summary["knot_level"] = {
        "position_limits": bool(np.all(q >= np.asarray([x["position_lower_rad"] for x in limits]) - 1e-12) and np.all(q <= np.asarray([x["position_upper_rad"] for x in limits]) + 1e-12)),
        "velocity_limits": bool(np.all(np.abs(dq) <= np.asarray([x["max_velocity_rad_s"] for x in limits]) + 1e-12)),
        "acceleration_limits": bool(np.all(np.abs(ddq) <= np.asarray([x["max_acceleration_rad_s2"] for x in limits]) + 1e-12)),
        "jerk_limits_if_defined": "CSV jerk field is diagnostic; JTC segment jerk is certified below",
    }
    summary["native_spline_interval_level"] = {x: summary["dynamic_validation"][x]["passed"] for x in ("position", "velocity", "acceleration", "jerk")}
    write_json(output / "stage27_exact_spline_dynamics.json", summary)
    return {"audit": audit, "summary": summary, "coefficients": coefficients}


def build_query_schedule(output: Path, traj: dict[str, Any], reconstruction: dict[str, Any]) -> list[float]:
    t = traj["times"]
    schedule = {float(x) for x in t}
    schedule.update(float((t[i] + t[i + 1]) * 0.5) for i in range(len(t) - 1))
    for quantity in ("position", "velocity", "acceleration", "jerk"):
        item = reconstruction["summary"]["dynamic_validation"][quantity]
        if item["time"] is not None:
            schedule.add(float(item["time"]))
    # Required historical risk anchors.
    schedule.add(float((t[5] + t[6]) * 0.5))
    schedule.add(float(t[5]))
    schedule.add(float(t[6]))
    schedule.add(float(t[9819]))
    schedule.add(float(t[9820]))
    values = sorted(schedule)
    write_json(output / "stage27_query_schedule.json", {
        "schema_version": "stage27-query-schedule-v1",
        "coverage": ["all trajectory knots", "all segment midpoints", "all formal global dynamic extrema", "historical interval 5/j3 anchors", "historical interval 9819/j2 position anchor"],
        "count": len(values),
        "times": values,
    })
    return values


def build_geometry_samples(output: Path, traj: dict[str, Any], max_dt: float = 0.005) -> dict[str, Any]:
    metadata = list(csv.DictReader(R26_INTERVALS.open(encoding="utf-8", newline="")))
    if len(metadata) != len(traj["times"]) - 1:
        raise RuntimeError("Stage 2.6 interval metadata is not aligned with Stage 2.5R2 trajectory")
    path_ref = read_json(PATH_REFERENCE)
    source_q = np.asarray(path_ref["source_q"], dtype=float)
    path_samples = path_ref["path_samples"]
    old_fk = list(csv.DictReader(OLD_FK_TRACE.open(encoding="utf-8", newline="")))
    if len(old_fk) != len(traj["times"]):
        raise RuntimeError("Frozen FK source mapping is not row-aligned with Stage 2.5R2 q path")
    waypoints = {}
    for row in csv.DictReader(WAYPOINTS.open(encoding="utf-8", newline="")):
        waypoints[int(row["waypoint_id"])] = row
    base = np.asarray(path_ref["robot_base_xyz_m"], dtype=float)
    rows = []
    sample_index = 0
    for i in range(len(traj["times"]) - 1):
        t0, t1 = float(traj["times"][i]), float(traj["times"][i + 1])
        n = max(1, int(np.ceil((t1 - t0) / max_dt)))
        coeff = segment_coefficients(traj["q"][i], traj["q"][i + 1], traj["dq"][i], traj["dq"][i + 1], traj["ddq"][i], traj["ddq"][i + 1], t1 - t0, "quintic")
        for sub in range(n):
            local = (t1 - t0) * sub / n
            when = t0 + local
            q = evaluate_coefficients(coeff, local)["position"]
            source_idx = int(old_fk[i]["source_path_index"])
            source_next = int(old_fk[min(i + 1, len(old_fk) - 1)]["source_path_index"])
            source_idx = min(max(source_idx, 0), len(path_samples) - 1)
            next_idx = min(max(source_idx + (1 if source_next >= source_idx else 0), 0), len(path_samples) - 1)
            meta = metadata[i]
            is_on = str(meta["spray_state"]).split("->")[0] == "ON" and meta["process_kind"] == "spray_on_segment"
            wp0 = (path_samples[source_idx].get("source_waypoints") or [path_samples[source_idx].get("source_waypoint")])[0]
            wp1 = (path_samples[next_idx].get("source_waypoints") or [path_samples[next_idx].get("source_waypoint")])[0]
            wp0 = int(wp0) if wp0 is not None else None
            wp1 = int(wp1) if wp1 is not None else wp0
            alpha = 0.0
            if is_on and next_idx > source_idx and wp0 is not None and wp1 is not None:
                direction = source_q[next_idx] - source_q[source_idx]
                alpha = float(np.clip(np.dot(q - source_q[source_idx], direction) / max(float(np.dot(direction, direction)), 1e-15), 0.0, 1.0))
            record = {"sample_index": sample_index, "time_s": when, "q": q.tolist(), "original_interval": i, "spray_state": "ON" if is_on else "OFF", "source_path_index": source_idx, "source_waypoint_0": wp0, "source_waypoint_1": wp1, "alpha": alpha}
            if is_on and wp0 is not None:
                a, b = waypoints[wp0], waypoints[wp1]
                target = (1.0 - alpha) * np.asarray([float(a["x"]), float(a["y"]), float(a["z"])]) + alpha * np.asarray([float(b["x"]), float(b["y"]), float(b["z"])])
                normal = (1.0 - alpha) * np.asarray([float(a["nx"]), float(a["ny"]), float(a["nz"])]) + alpha * np.asarray([float(b["nx"]), float(b["ny"]), float(b["nz"])])
                normal /= max(float(np.linalg.norm(normal)), 1e-15)
                surface = (1.0 - alpha) * (np.asarray([float(a["surface_x_m"]), float(a["surface_y_m"]), float(a["surface_z_m"])]) - base) + alpha * (np.asarray([float(b["surface_x_m"]), float(b["surface_y_m"]), float(b["surface_z_m"])]) - base)
                record.update({"target": target.tolist(), "normal": normal.tolist(), "surface_base": surface.tolist(), "nominal_standoff_m": float(path_ref["standoff_nominal_m"])})
            rows.append(record)
            sample_index += 1
    # Include final endpoint exactly once.
    q = traj["q"][-1]
    rows.append({"sample_index": sample_index, "time_s": float(traj["times"][-1]), "q": q.tolist(), "original_interval": len(traj["times"]) - 2, "spray_state": "OFF", "source_path_index": int(old_fk[-1]["source_path_index"]), "source_waypoint_0": None, "source_waypoint_1": None, "alpha": 0.0})
    path = output / "stage27_geometry_samples.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    return {"path": str(path.resolve()), "sha256": sha256(path), "sample_count": len(rows), "max_dt_s": max_dt, "source_intervals": len(metadata), "on_samples": sum(row["spray_state"] == "ON" for row in rows), "off_samples": sum(row["spray_state"] == "OFF" for row in rows)}


def source_identity() -> dict[str, Any]:
    candidates = [
        JTC_SOURCE_ROOT / "joint_trajectory_controller/include/joint_trajectory_controller/interpolation_methods.hpp",
        JTC_SOURCE_ROOT / "joint_trajectory_controller/src/joint_trajectory_controller.cpp",
        JTC_SOURCE_ROOT / "joint_trajectory_controller/src/trajectory.cpp",
        JTC_SOURCE_ROOT / "joint_trajectory_controller/include/joint_trajectory_controller/trajectory.hpp",
        JTC_SOURCE_ROOT / "joint_trajectory_controller/doc/trajectory.rst",
    ]
    files = []
    for path in candidates:
        if path.exists():
            files.append({"path": str(path.resolve()), "sha256": sha256(path), "size": path.stat().st_size})
    return {"repository": "https://github.com/ros-controls/ros2_controllers.git", "tag": "4.40.1", "commit": "31015e0aa7ce9d0853a88d5fc2fe50f0e583ba5c", "source_files": files, "source_runtime_match": read_json(ENV_ROOT / "stage27env_source_runtime_match.json")}


def frozen_hash_check(snapshot: Path) -> dict[str, Any]:
    baseline = read_json(snapshot)
    mismatches = []
    checked = 0
    for raw_path, info in baseline["files"].items():
        path = Path(raw_path)
        checked += 1
        current = sha256(path) if path.is_file() else None
        if current != info.get("sha256"):
            mismatches.append({"path": raw_path, "expected": info.get("sha256"), "actual": current, "exists": path.exists()})
    return {"snapshot": str(snapshot.resolve()), "files_checked": checked, "mismatch_count": len(mismatches), "mismatches": mismatches[:100], "snapshot_reference_mismatch_count": baseline.get("reference_mismatch_count", 0), "stage25r2_sha256_snapshot": baseline.get("stage25r2_trajectory_sha256")}


def runtime_probe() -> dict[str, Any]:
    commands = {
        "list_controllers": "source /opt/ros/jazzy/setup.bash && ros2 control list_controllers",
        "interpolation_method": "source /opt/ros/jazzy/setup.bash && ros2 param get /fairino5_controller interpolation_method",
        "action_list_types": "source /opt/ros/jazzy/setup.bash && ros2 action list -t",
        "topic_list_types": "source /opt/ros/jazzy/setup.bash && ros2 topic list -t",
        "service_list_types": "source /opt/ros/jazzy/setup.bash && ros2 service list -t",
        "controller_state_info": "source /opt/ros/jazzy/setup.bash && ros2 topic info -v /fairino5_controller/controller_state",
        "query_state_type": "source /opt/ros/jazzy/setup.bash && ros2 service type /fairino5_controller/query_state",
    }
    raw = {name: run_wsl(command, 45) for name, command in commands.items()}
    active = bool(re.search(r"fairino5_controller\s+joint_trajectory_controller/JointTrajectoryController\s+active", raw["list_controllers"]["stdout"], re.I))
    method = "splines" in raw["interpolation_method"]["stdout"].lower()
    return {"ros_distro": "jazzy", "jtc_version": "4.40.1-1noble.20260615.171409", "release": "4.40.1", "source_tag": "4.40.1", "source_commit": "31015e0aa7ce9d0853a88d5fc2fe50f0e583ba5c", "controller": "fairino5_controller", "controller_active": active, "interpolation_method": "splines" if method else None, "runtime_verified": active and method, "action_endpoint": "/fairino5_controller/follow_joint_trajectory", "state_topic": "/fairino5_controller/controller_state", "query_state_service": "/fairino5_controller/query_state", "raw_probes": raw}


def launch_runtime(output: Path) -> subprocess.Popen[Any]:
    log = (output / "stage27_full_runtime_logs.txt").open("w", encoding="utf-8")
    command = f"source /opt/ros/jazzy/setup.bash && ros2 launch /mnt/d/robotfucker/outputs/stage27_native_runtime_bringup/stage27env_20260804T103532Z/runtime_config/stage27_runtime.launch.py"
    process = subprocess.Popen(["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", command], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
    write_json(output / "stage27_runtime_process.json", {"pid": process.pid, "command": command, "started_utc": datetime.now(timezone.utc).isoformat()})
    return process


def wait_until_runtime_ready(timeout_s: float = 90.0) -> dict[str, Any]:
    started = time.time()
    last = {}
    while time.time() - started < timeout_s:
        last = runtime_probe()
        if last["runtime_verified"]:
            return last
        time.sleep(2.0)
    return last


def run_native_action(output: Path) -> dict[str, Any]:
    command = "source /opt/ros/jazzy/setup.bash && python3 /mnt/d/robotfucker/scripts/stage27_native_action.py --output /mnt/d/robotfucker/" + str(output.relative_to(ROOT)).replace("\\", "/")
    proc = subprocess.run(["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", command], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1800)
    (output / "stage27_action_client_stdout.log").write_text(proc.stdout, encoding="utf-8")
    (output / "stage27_action_client_stderr.log").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"Native action client failed with exit {proc.returncode}: {proc.stderr[-2000:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def run_geometry(output: Path) -> dict[str, Any]:
    command = "source /opt/ros/jazzy/setup.bash && source /mnt/d/robotfucker/install/setup.bash && ros2 launch /mnt/d/robotfucker/tools/stage27_native_geometry_launch.py samples_jsonl:=" + f"/mnt/d/robotfucker/{output.relative_to(ROOT).as_posix()}" + "/stage27_geometry_samples.jsonl output_dir:=" + f"/mnt/d/robotfucker/{output.relative_to(ROOT).as_posix()}"
    proc = subprocess.run(["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", command], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1800)
    (output / "stage27_geometry_runtime_stdout.log").write_text(proc.stdout, encoding="utf-8")
    (output / "stage27_geometry_runtime_stderr.log").write_text(proc.stderr, encoding="utf-8")
    summary_path = output / "stage27_process_geometry_validation.json"
    if proc.returncode != 0 or not summary_path.exists():
        return {"status": "blocked_native_spline_process_geometry_violation", "passed": False, "reason": "MoveIt FK geometry runner failed", "exit_code": proc.returncode}
    return read_json(summary_path)


def run_bullet(output: Path) -> dict[str, Any]:
    import scripts.run_stage26 as stage26_runner
    stage26_runner.PARTS = PARTS_DIR
    native_dir = output / "native_bullet"
    result = stage26_runner.run_native(native_dir, output / "stage27_bullet_intervals.csv", "bullet", 1, False, False)
    rows_path = native_dir / "stage26_continuous_robot_world_intervals.jsonl"
    summary_path = native_dir / "stage26_native_summary.json"
    rows = []
    if rows_path.exists():
        with rows_path.open(encoding="utf-8", errors="replace") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    summary = read_json(summary_path) if summary_path.exists() else {}
    collisions = [x for x in rows if x.get("continuous_collision")]
    skipped = [x for x in rows if x.get("ccd_api_called") is not True]
    evidence = {"schema_version": "stage27-bullet-dense-validation-v1", "status": "passed" if result.get("exit_code") == 0 and len(rows) == len(read_csv_rows(output / "stage27_bullet_intervals.csv")) and not collisions and not skipped else "blocked", "validation_complete": bool(summary.get("all_intervals_executed") and len(rows) == int(summary.get("interval_count", -1)) and not skipped), "backend": "native Bullet robot-world CCD", "checked_intervals": len(rows), "expected_intervals": int(summary.get("interval_count", -1)), "collisions": len(collisions), "skipped": len(skipped), "first_collision": collisions[0] if collisions else None, "runner": {"exit_code": result.get("exit_code"), "stdout_tail": result.get("stdout", "")[-1000:], "stderr_tail": result.get("stderr", "")[-1000:]}, "raw_summary": summary, "raw_directory": str(native_dir.resolve())}
    write_json(output / "stage27_bullet_dense_validation.json", evidence)
    return evidence


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_bullet_intervals(output: Path, traj: dict[str, Any], max_dt: float = 0.0025) -> dict[str, Any]:
    metadata = read_csv_rows(R26_INTERVALS)
    fields = ["interval_index", "time_start_s", "time_end_s", "process_order_index", "spray_state", "process_kind", "segment_id", "transition_id", "source_boundary", *[f"q0_{j}" for j in range(6)], *[f"q1_{j}" for j in range(6)]]
    count = 0
    with (output / "stage27_bullet_intervals.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i in range(len(traj["times"]) - 1):
            t0, t1 = float(traj["times"][i]), float(traj["times"][i + 1])
            n = max(1, int(np.ceil((t1 - t0) / max_dt)))
            coeff = segment_coefficients(traj["q"][i], traj["q"][i + 1], traj["dq"][i], traj["dq"][i + 1], traj["ddq"][i], traj["ddq"][i + 1], t1 - t0, "quintic")
            for sub in range(n):
                l0, l1 = (t1 - t0) * sub / n, (t1 - t0) * (sub + 1) / n
                q0 = evaluate_coefficients(coeff, l0)["position"]
                q1 = evaluate_coefficients(coeff, l1)["position"]
                source = metadata[i]
                row = {"interval_index": count, "time_start_s": t0 + l0, "time_end_s": t0 + l1, "process_order_index": source["process_order_index"], "spray_state": source["spray_state"], "process_kind": source["process_kind"], "segment_id": source["segment_id"], "transition_id": source["transition_id"], "source_boundary": source["source_boundary"]}
                row.update({f"q0_{j}": float(q0[j]) for j in range(6)})
                row.update({f"q1_{j}": float(q1[j]) for j in range(6)})
                writer.writerow(row)
                count += 1
    return {"path": str((output / "stage27_bullet_intervals.csv").resolve()), "sha256": sha256(output / "stage27_bullet_intervals.csv"), "source_intervals": len(metadata), "checked_intervals_expected": count, "max_subdivision_dt_s": max_dt}


def source_integrity_after(output: Path, before: dict[str, Any]) -> dict[str, Any]:
    after = frozen_hash_check(ENV_HASH_AFTER)
    after["stage25r2_hash_before"] = before.get("stage25r2_sha256_snapshot")
    after["stage25r2_hash_after"] = sha256(R2_TRAJECTORY)
    after["stage25r2_hash_equal"] = after["stage25r2_hash_before"] == after["stage25r2_hash_after"]
    write_json(output / "stage27_frozen_integrity.json", after)
    return after


def build_determinism(output: Path, traj: dict[str, Any], limits: dict[str, Any]) -> dict[str, Any]:
    records = []
    for rebuild in range(1, 4):
        digest = hashlib.sha256()
        maxima = {}
        for i in range(len(traj["times"]) - 1):
            coeff = segment_coefficients(traj["q"][i], traj["q"][i + 1], traj["dq"][i], traj["dq"][i + 1], traj["ddq"][i], traj["ddq"][i + 1], float(traj["times"][i + 1] - traj["times"][i]), "quintic")
            digest.update(np.asarray(coeff, dtype=np.float64).tobytes())
            for quantity in ("position", "velocity", "acceleration", "jerk"):
                candidates = extrema_times(coeff, float(traj["times"][i + 1] - traj["times"][i]), quantity)
                values = [evaluate_coefficients(coeff, x)[quantity] for x in candidates]
                if values:
                    matrix = np.asarray(values)
                    idx = np.unravel_index(np.argmax(np.abs(matrix)), matrix.shape)
                    maxima.setdefault(quantity, []).append((float(np.abs(matrix[idx])), i, float(traj["times"][i] + candidates[idx[0]]), int(idx[1])))
        records.append({"rebuild": rebuild, "coefficients_sha256": digest.hexdigest(), "maxima_sha256": semantic_sha(maxima)})
    identical = len({x["coefficients_sha256"] for x in records}) == 1 and len({x["maxima_sha256"] for x in records}) == 1
    result = {"schema_version": "stage27-determinism-v1", "rebuilds": 3, "identical": identical, "status": "passed" if identical else "blocked", "records": records, "compared": ["spline coefficients", "extrema candidates", "worst locations"]}
    write_json(output / "stage27_determinism_report.json", result)
    return result


def final_gate(output: Path, runtime: dict[str, Any], goal: dict[str, Any], action: dict[str, Any], reconstruction: dict[str, Any], geometry: dict[str, Any], bullet: dict[str, Any], determinism: dict[str, Any], integrity: dict[str, Any], trajectory: dict[str, Any]) -> dict[str, Any]:
    dyn = reconstruction["summary"]["dynamic_validation"]
    execution = action.get("native_execution", action)
    goal_exact = all(goal["formal_goal_matches_stage25r2"].values())
    query_complete = (
        int(action.get("query_state_samples_received", 0)) == int(action.get("query_state_samples_requested", 0))
        and int(action.get("query_state_schedule_skipped", 0)) == 0
    )
    semantic = bool(action.get("native_query_state_vs_reconstruction", {}).get("reconstruction_matches_native_runtime")) and query_complete
    pre_goal = read_json(output / "stage27_pre_goal_initial_state.json") if (output / "stage27_pre_goal_initial_state.json").exists() else {}
    first_blocker = None
    status = "passed"
    if not runtime.get("runtime_verified"):
        status, first_blocker = "blocked_native_runtime_identity_drift", {"reason": "formal runtime identity was not verified"}
    elif not goal_exact:
        status, first_blocker = "blocked_formal_goal_identity_mismatch", {"reason": "formal goal differs from Stage 2.5R2"}
    elif not pre_goal.get("compatible", False):
        status, first_blocker = "blocked_native_controller_initial_state_incompatible", pre_goal
    elif not action.get("goal_accepted", False):
        status, first_blocker = "blocked_native_controller_execution_failure", {"reason": "FollowJointTrajectory goal was not accepted"}
    elif not semantic:
        status = "blocked_native_spline_semantics_mismatch"
        first_blocker = {
            "reason": "native query_state coverage/reconstruction and/or controller_state crosscheck failed",
            "query_state_samples_received": int(action.get("query_state_samples_received", 0)),
            "query_state_samples_requested": int(action.get("query_state_samples_requested", 0)),
            "query_state_schedule_skipped": int(action.get("query_state_schedule_skipped", 0)),
            "native_query_state_vs_reconstruction": action.get("native_query_state_vs_reconstruction", {}),
            "controller_state_vs_reconstruction": action.get("controller_state_vs_reconstruction", {}),
        }
    else:
        for quantity in ("position", "velocity", "acceleration", "jerk"):
            if not dyn[quantity]["passed"]:
                status = f"blocked_native_spline_{quantity}_limit_violation"
                first_blocker = {"quantity": quantity, **dyn[quantity]}
                break
        if first_blocker is None and not geometry.get("passed", geometry.get("status") == "passed"):
            status, first_blocker = "blocked_native_spline_process_geometry_violation", geometry
        elif first_blocker is None and not bullet.get("validation_complete"):
            status, first_blocker = "blocked_native_spline_collision", bullet
        elif first_blocker is None and bullet.get("collisions", 0) != 0:
            status, first_blocker = "blocked_native_spline_collision", bullet
        elif first_blocker is None and not execution.get("execution_completed", False):
            status, first_blocker = "blocked_native_controller_execution_failure", execution
        elif first_blocker is None and not determinism.get("identical"):
            status, first_blocker = "blocked_native_spline_nondeterminism", determinism
        elif first_blocker is None and integrity.get("mismatch_count") != 0:
            status, first_blocker = "blocked_frozen_input_integrity_mismatch", integrity
    historical = {"historical_max_jerk_ratio": 240.65411508919385, "formal_max_jerk_ratio": dyn["jerk"]["max_ratio"], "same_joint": dyn["jerk"]["joint"] == "j3", "same_interval": dyn["jerk"]["segment"] == 5, "interpretation": "formal JTC quintic reconstruction is native-query anchored; historical value is warning only"}
    gate = {"schema_version": "stage27-formal-native-gate-v1", "Stage_2_7_environment": {"status": "READY_FOR_FORMAL_NATIVE_CERTIFICATION"}, "Stage_2_7": {"status": status, "actually_started": True}, "runtime_identity": runtime, "formal_goal": {"exact_stage25r2_trajectory": goal_exact, "goal_sent": action.get("goal_sent", False), "goal_accepted": action.get("goal_accepted", False), "point_count": len(trajectory["times"]), "duration": float(trajectory["times"][-1])}, "native_semantics": action.get("native_query_state_vs_reconstruction", {}), "dynamic_limits": dyn, "process_geometry": geometry, "Bullet": bullet, "execution": execution, "determinism": determinism, "frozen_integrity": integrity, "historical_240x_jerk": historical, "primary_blocker": first_blocker, "Stage_2_8": {"status": "unblocked_not_started" if status == "passed" else "blocked_by_stage27"}, "formal_native_JTC_execution": "passed" if status == "passed" else status}
    write_json(output / "stage27_gate_report.json", gate)
    report = "# Stage 2.7 Formal Native Controller Execution Certification\n\n"
    report += f"- Stage 2.7: **{status}**\n- Formal input: `{R2_TRAJECTORY}`\n- Native runtime: JTC 4.40.1 / commit `31015e0aa7ce9d0853a88d5fc2fe50f0e583ba5c` / `splines`\n- Goal: sent=`{action.get('goal_sent')}`, accepted=`{action.get('goal_accepted')}`, completed=`{execution.get('execution_completed')}`\n- Query-state reconstruction: `{semantic}`\n- Bullet: `{bullet.get('checked_intervals')}/{bullet.get('expected_intervals')}`, collisions=`{bullet.get('collisions')}`\n\n"
    report += "The source Stage 2.5R2 trajectory was not retimed, resampled, smoothed, or edited.\n\n"
    report += "Primary blocker:\n\n```json\n" + json.dumps(first_blocker, ensure_ascii=False, indent=2) + "\n```\n"
    (output / "stage27_report.md").write_text(report, encoding="utf-8")
    return gate


def write_sums(output: Path) -> None:
    rows = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name not in {"SHA256SUMS"}:
            rows.append(f"{sha256(path)}  {path.relative_to(output).as_posix()}")
    (output / "SHA256SUMS").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> int:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = ROOT / "outputs/stage27_formal_native_certification" / f"stage27_formal_{stamp}"
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite formal output: {output}")
    output.mkdir(parents=True)
    traj = read_trajectory()
    if read_json(R2_GATE).get("Stage_2_5R2") != "passed" or read_json(R26_GATE).get("Stage_2_6") != "passed":
        raise RuntimeError("Frozen upstream gates are not passed")
    before_integrity = frozen_hash_check(ENV_HASH_AFTER)
    write_json(output / "stage27_frozen_integrity_before.json", before_integrity)
    write_json(output / "stage27_runtime_identity.json", {"bringup": read_json(ENV_ROOT / "stage27env_runtime_versions.json"), "source_identity": source_identity()})
    write_json(output / "stage27_input_manifest.json", {"schema_version": "stage27-formal-input-manifest-v1", "stage25r2_gate": str(R2_GATE.resolve()), "stage25r2_manifest": str(R2_MANIFEST.resolve()), "stage26_gate": str(R26_GATE.resolve()), "stage25r2_trajectory": {"path": str(R2_TRAJECTORY.resolve()), "sha256": sha256(R2_TRAJECTORY), "point_count": len(traj["times"]), "first_timestamp": float(traj["times"][0]), "last_timestamp": float(traj["times"][-1]), "duration": float(traj["times"][-1]), "contains_positions": True, "contains_velocities": True, "contains_accelerations": True}, "frozen_snapshot": str(ENV_HASH_AFTER.resolve()), "modify_frozen_upstream_inputs": False})
    goal = build_goal(output, traj)
    limits_record = load_limits()
    write_json(output / "stage27_effective_dynamic_limits.json", limits_record)
    reconstruction = build_reconstruction(output, traj, limits_record)
    build_query_schedule(output, traj, reconstruction)
    geometry_info = build_geometry_samples(output, traj)
    write_json(output / "stage27_process_geometry_input.json", geometry_info)
    bullet_info = write_bullet_intervals(output, traj)
    runtime_proc = launch_runtime(output)
    runtime = wait_until_runtime_ready()
    write_json(output / "stage27_runtime_identity.json", {"bringup": read_json(ENV_ROOT / "stage27env_runtime_versions.json"), "source_identity": source_identity(), "formal_probe": runtime})
    action = None
    try:
        if not runtime.get("runtime_verified"):
            action = {"goal_sent": False, "goal_accepted": False, "execution_completed": False, "primary_blocker": "blocked_native_runtime_identity_drift"}
        elif not all(goal["formal_goal_matches_stage25r2"].values()):
            action = {"goal_sent": False, "goal_accepted": False, "execution_completed": False, "primary_blocker": "blocked_formal_goal_identity_mismatch"}
        else:
            try:
                action = run_native_action(output)
            except Exception as error:
                action = {"goal_sent": True, "goal_accepted": None, "execution_started": None, "execution_completed": False, "action_result": "tooling_or_runtime_failure", "error": repr(error), "primary_blocker": "blocked_native_controller_execution_failure"}
                write_json(output / "stage27_action_execution.json", {"native_execution": action})
        if action.get("goal_accepted"):
            geometry = run_geometry(output)
            bullet = run_bullet(output)
        else:
            geometry = {"status": "not_evaluated", "passed": False}
            bullet = {"status": "not_evaluated", "validation_complete": False, "collisions": None, "checked_intervals": 0, "expected_intervals": bullet_info["checked_intervals_expected"]}
    finally:
        # Leave the native runtime evidence in place, then terminate only the
        # process started by this run.  The WSL distro itself is not shut down.
        try:
            runtime_proc.terminate()
        except Exception:
            pass
    determinism = build_determinism(output, traj, limits_record)
    integrity = source_integrity_after(output, before_integrity)
    gate = final_gate(output, runtime, goal, action, reconstruction, geometry, bullet, determinism, integrity, traj)
    write_sums(output)
    print(json.dumps({"output": str(output.resolve()), "Stage_2_7": gate["Stage_2_7"], "primary_blocker": gate["primary_blocker"]}, ensure_ascii=False))
    return 0 if gate["Stage_2_7"]["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
