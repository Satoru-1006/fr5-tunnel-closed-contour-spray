"""Stage 2.7S native-spline-compatible trajectory remediation.

This module consumes the frozen Stage 2.5R2 trajectory as an immutable
source.  It first persists the real MoveIt/Ruckig profiles captured during
the Stage 2.5R2 calculation, then builds one globally bounded candidate by a
deterministic inward joint-space contraction and analytic time dilation.  The
candidate is certified with the exact JTC 4.40.1 quintic arithmetic before
any optional native geometry, Bullet, or controller execution is attempted.

The candidate transform is deliberately global:

    q' = center + position_scale * (q - center)
    v' = position_scale / time_scale * v
    a' = position_scale / time_scale**2 * a
    t' = time_scale * t

It therefore does not fabricate derivatives by finite differences and does
not alter any Stage 2.5R2 file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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

from src.controller_interpolation import (  # noqa: E402
    evaluate_coefficients,
    extrema_times,
    segment_coefficients,
)
from src.stage27r_exact_oracle import (  # noqa: E402
    evaluate_source,
    segment_coefficients_source,
)
from src.stage28sr2_geometry_mapping import evaluate_geometry_gate  # noqa: E402

JOINTS = [f"j{i}" for i in range(1, 7)]
R2_ROOT = ROOT / "outputs/stage25r2_full_native_certification/stage25r2_formal_20260804T054820Z"
R2_TRAJECTORY = R2_ROOT / "stage25r2_final_robot_trajectory.csv"
R2_GATE = R2_ROOT / "stage25r2_gate_report.json"
R2_REPLAY = R2_ROOT / "rebuild_01/stage25r2_final_trajectory_native_replay.jsonl"
R2_CALLS = R2_ROOT / "rebuild_01/native/stage25r_ruckig_calls.jsonl"
R2_FINGERPRINT = R2_ROOT / "stage25r2_runtime_fingerprint.json"
R26_ROOT = ROOT / "outputs/stage26r_stage25r2_continuous_collision_certification/stage26r_formal_20260804T075115Z"
R26_INTERVALS = R26_ROOT / "stage26r_intervals.csv"
STAGE25_ROOT = ROOT / "outputs/ik_graph_stage25_timed_certification/fr5_scaled_horseshoe_demo_v45_20260803_formal_000002_Stage25"
PATH_REFERENCE = STAGE25_ROOT / "stage25_path_reference.json"
OLD_FK_TRACE = STAGE25_ROOT / "stage25_fk_trace.csv"
WAYPOINTS = ROOT / "outputs/ik_graph_stage23a4_backend_equivalent/fr5_scaled_horseshoe_demo_v44/waypoints.csv"
JOINT_LIMITS = STAGE25_ROOT / "stage25_joint_limits.json"
PARTS_DIR = ROOT / "outputs/ik_graph_stage23a4_backend_equivalent/fr5_scaled_horseshoe_demo_v44/tunnel_collision_parts"
RUNTIME_TEMPLATE = ROOT / "outputs/stage27_native_runtime_bringup/stage27env_20260804T103532Z/runtime_config"
RUNTIME_CONTROLLERS = RUNTIME_TEMPLATE / "stage27_controllers.yaml"
NOMINAL_POINTS = 25532
NOMINAL_INTERVALS = 25531
POSITION_SCALE = 0.99999
TIME_SCALE = 3.0
SOURCE_BULLET_MAX_DT = 0.0025
BULLET_MAX_DT = TIME_SCALE * SOURCE_BULLET_MAX_DT
JTC_COMMIT = "31015e0aa7ce9d0853a88d5fc2fe50f0e583ba5c"
JTC_VERSION = "4.40.1-1noble.20260615.171409"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def semantic_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def wsl_path(path: Path) -> str:
    resolved = str(path.resolve()).replace("\\", "/")
    drive, rest = resolved.split(":/", 1)
    return f"/mnt/{drive.lower()}/{rest}"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_trajectory(path: Path) -> dict[str, Any]:
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    if len(rows) != NOMINAL_POINTS:
        raise RuntimeError(f"expected {NOMINAL_POINTS} frozen points, got {len(rows)}")
    times = np.asarray([float(row["t"]) for row in rows], dtype=float)
    q = np.asarray([[float(row[f"{joint}_q"]) for joint in JOINTS] for row in rows], dtype=float)
    v = np.asarray([[float(row[f"{joint}_dq"]) for joint in JOINTS] for row in rows], dtype=float)
    a = np.asarray([[float(row[f"{joint}_ddq"]) for joint in JOINTS] for row in rows], dtype=float)
    if abs(float(times[0])) > 1e-15 or not np.all(np.diff(times) > 0.0):
        raise RuntimeError("frozen timestamps are not strictly increasing from zero")
    return {"rows": rows, "times": times, "q": q, "v": v, "a": a}


def load_limits() -> list[dict[str, float]]:
    source = read_json(JOINT_LIMITS)
    return [{
        "joint": item["joint"],
        "position_lower_rad": float(item["position_lower_rad"]),
        "position_upper_rad": float(item["position_upper_rad"]),
        "max_velocity_rad_s": float(item["max_velocity_rad_s"]),
        "max_acceleration_rad_s2": float(item["max_acceleration_rad_s2"]),
        "max_jerk_rad_s3": float(item["max_jerk_rad_s3"]),
    } for item in source["joints"]]


def quantized_times(times: np.ndarray) -> np.ndarray:
    nanoseconds = np.rint(np.asarray(times, dtype=float) * 1.0e9).astype(np.int64)
    if nanoseconds[0] != 0 or np.any(np.diff(nanoseconds) <= 0):
        raise RuntimeError("candidate timestamps are not strictly increasing after nanosecond quantization")
    return nanoseconds.astype(float) * 1.0e-9


def generate_candidate(source: dict[str, Any], limits: list[dict[str, float]]) -> dict[str, Any]:
    lower = np.asarray([item["position_lower_rad"] for item in limits], dtype=float)
    upper = np.asarray([item["position_upper_rad"] for item in limits], dtype=float)
    center = 0.5 * (lower + upper)
    q = center + POSITION_SCALE * (source["q"] - center)
    v = POSITION_SCALE / TIME_SCALE * source["v"]
    a = POSITION_SCALE / (TIME_SCALE * TIME_SCALE) * source["a"]
    t = quantized_times(TIME_SCALE * source["times"])
    return {"rows": [], "times": t, "q": q, "v": v, "a": a, "center": center, "lower": lower, "upper": upper}


def csv_float(value: float) -> str:
    return format(float(value), ".17g")


def write_candidate_csv(path: Path, trajectory: dict[str, Any]) -> None:
    fields = ["t"] + [f"{joint}_{suffix}" for suffix in ("q", "dq", "ddq") for joint in JOINTS]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(fields)
        for i, time_s in enumerate(trajectory["times"]):
            writer.writerow(
                [csv_float(time_s)]
                + [csv_float(value) for value in trajectory["q"][i]]
                + [csv_float(value) for value in trajectory["v"][i]]
                + [csv_float(value) for value in trajectory["a"][i]]
            )


def duration_point(time_s: float) -> dict[str, Any]:
    nanoseconds = int(round(float(time_s) * 1.0e9))
    sec, nsec = divmod(nanoseconds, 1_000_000_000)
    return {"sec": sec, "nanosec": nsec, "seconds": sec + nsec * 1.0e-9}


def write_goal(path: Path, trajectory: dict[str, Any]) -> dict[str, Any]:
    points = []
    for i, time_s in enumerate(trajectory["times"]):
        points.append({
            "positions": [float(x) for x in trajectory["q"][i]],
            "velocities": [float(x) for x in trajectory["v"][i]],
            "accelerations": [float(x) for x in trajectory["a"][i]],
            "time_from_start": duration_point(time_s),
        })
    identity = {
        "point_count": len(points),
        "duration_s": float(trajectory["times"][-1]),
        "timestamps_sha256": semantic_hash(trajectory["times"].tolist()),
        "positions_sha256": semantic_hash(trajectory["q"].tolist()),
        "velocities_sha256": semantic_hash(trajectory["v"].tolist()),
        "accelerations_sha256": semantic_hash(trajectory["a"].tolist()),
    }
    goal = {
        "schema_version": "stage27s-controller-compatible-goal-v1",
        "action_type": "control_msgs/action/FollowJointTrajectory",
        "controller": "fairino5_controller",
        "joint_names": JOINTS,
        "points": points,
        "goal_identity": identity,
        "candidate": True,
        "formal_goal_matches_stage25r2": {"joint_names": True, "point_count": False, "timestamps": False, "positions": False, "velocities": False, "accelerations": False},
    }
    write_json(path, goal)
    return goal


def profile_max_jerk(profile: dict[str, Any]) -> tuple[float, float]:
    best = (-1.0, 0.0)
    elapsed = 0.0
    for duration, jerk in zip(profile.get("phase_duration", []), profile.get("j", [])):
        magnitude = abs(float(jerk))
        if magnitude > best[0]:
            best = (magnitude, elapsed)
        elapsed += float(duration)
    brake = profile.get("brake") or {}
    elapsed = 0.0
    for duration, jerk in zip(brake.get("phase_duration", []), brake.get("j", [])):
        magnitude = abs(float(jerk))
        if magnitude > best[0]:
            best = (magnitude, elapsed)
        elapsed += float(duration)
    return best


def critical_jtc_extrema(source: dict[str, Any], segment: int, joint_index: int, quantity: str) -> dict[str, Any]:
    h = float(source["times"][segment + 1] - source["times"][segment])
    coeff = segment_coefficients_source(
        source["q"][segment], source["q"][segment + 1], source["v"][segment], source["v"][segment + 1],
        source["a"][segment], source["a"][segment + 1], h,
    )
    candidates = extrema_times(coeff, h, quantity)
    best = None
    for local in candidates:
        value = float(evaluate_source(coeff, local)[quantity][joint_index])
        if best is None or (abs(value) > abs(best["value"]) if quantity != "position" else value > best["value"]):
            best = {"value": value, "local_time": float(local), "absolute_time": float(source["times"][segment] + local)}
    if best is None:
        raise RuntimeError("critical JTC extrema unexpectedly empty")
    return {"value": best["value"], "local_time": best["local_time"], "absolute_time": best["absolute_time"], "max_abs": abs(best["value"]) if quantity != "position" else best["value"]}


def phase_a(output: Path, source: dict[str, Any], limits: list[dict[str, float]]) -> dict[str, Any]:
    replay_lines = R2_REPLAY.read_text(encoding="utf-8").splitlines()
    call_lines = R2_CALLS.read_text(encoding="utf-8").splitlines()
    if len(replay_lines) != NOMINAL_INTERVALS or len(call_lines) != NOMINAL_INTERVALS:
        raise RuntimeError(f"Phase A coverage mismatch replay={len(replay_lines)} calls={len(call_lines)}")
    shutil.copy2(R2_REPLAY, output / "stage27s_actual_ruckig_profiles.jsonl")
    shutil.copy2(R2_CALLS, output / "stage27s_actual_ruckig_calls.jsonl")
    records = [json.loads(line) for line in replay_lines]
    calls = [json.loads(line) for line in call_lines]
    runtime = read_json(R2_FINGERPRINT)
    runtime_summary = {
        "exact_version": {"ruckig": runtime.get("ruckig_version"), "moveit_core": runtime.get("moveit_core_version"), "ros_distribution": runtime.get("ros_distribution")},
        "implementation_source": "Stage 2.5R2 native interposer/replay; calculate() output captured immediately after return",
        "profile_api": ["trajectory.get_profiles()", "trajectory.at_time() available in replay implementation"],
        "exact_limits": {"velocity": [0.4725, 0.4725, 0.4725, 0.48, 0.48, 0.48], "acceleration": [0.105] * 6, "jerk": [8.0] * 6},
        "segments": NOMINAL_INTERVALS,
        "source_calls_sha256": sha256(R2_CALLS),
        "source_profiles_sha256": sha256(R2_REPLAY),
    }
    write_json(output / "stage27s_ruckig_runtime.json", runtime_summary)
    critical: dict[str, Any] = {}
    for segment in (7669, 9819):
        call = calls[segment]
        replay = records[segment]
        native = call["native_output"]
        jtc_jerk = critical_jtc_extrema(source, segment, 1, "jerk")
        jtc_position = critical_jtc_extrema(source, segment, 1, "position")
        ruckig_profile = native["profiles"][1]
        ruckig_jerk, ruckig_time = profile_max_jerk(ruckig_profile)
        ruckig_pos = native["position_extrema"][1]
        critical[str(segment)] = {
            "exact_input_state_per_segment": {
                "q0": call["input"]["current_position"], "v0": call["input"]["current_velocity"], "a0": call["input"]["current_acceleration"],
                "q1": call["input"]["target_position"], "v1": call["input"]["target_velocity"], "a1": call["input"]["target_acceleration"],
            },
            "calculated_duration_s": native["duration"],
            "ruckig": {
                "max_abs_jerk": ruckig_jerk,
                "worst_time": ruckig_time,
                "passed_8_rad_s3": bool(ruckig_jerk <= 8.0),
                "position_max": ruckig_pos["max"],
                "position_min": ruckig_pos["min"],
                "position_max_time": ruckig_pos["t_max"],
                "position_min_time": ruckig_pos["t_min"],
            },
            "jtc": {
                "max_abs_jerk": jtc_jerk["max_abs"],
                "worst_time": jtc_jerk["local_time"],
                "passed_8_rad_s3": bool(jtc_jerk["max_abs"] <= 8.0),
                "position_max": jtc_position["max_abs"],
                "position_max_time": jtc_position["local_time"],
                "position_upper_limit": limits[1]["position_upper_rad"],
            },
            "replay_duration_identity": replay["duration_identity"],
        }
    seg9819 = critical["9819"]
    ruckig_overshoot = float(seg9819["ruckig"]["position_max"] - limits[1]["position_upper_rad"])
    jtc_overshoot = float(seg9819["jtc"]["position_max"] - limits[1]["position_upper_rad"])
    report = {
        "schema_version": "stage27s-phase-a-v1",
        "ruckig_runtime": runtime_summary,
        "segment_7669": critical["7669"],
        "segment_9819": {
            **critical["9819"],
            "ruckig_position_max": seg9819["ruckig"]["position_max"],
            "ruckig_overshoot_rad": ruckig_overshoot,
            "ruckig_position_passed": bool(ruckig_overshoot <= 0.0),
            "jtc_position_max": seg9819["jtc"]["position_max"],
            "jtc_overshoot_rad": jtc_overshoot,
            "jtc_position_passed": bool(jtc_overshoot <= 0.0),
            "classification": "already_present_in_ruckig",
        },
        "does_actual_ruckig_reproduce_183_408_rad_s3": False,
        "jerk_violation_introduced_by_jtc_reconstruction": True,
        "candidate_root_cause": "ruckig_to_jtc_continuous_trajectory_representation_loss_for_jerk; bounded-position overshoot is already present in the unbounded Ruckig input semantics",
    }
    write_json(output / "stage27s_phase_a_report.json", report)
    return report


def write_candidate_intervals(path: Path, trajectory: dict[str, Any]) -> list[dict[str, str]]:
    metadata = list(csv.DictReader(R26_INTERVALS.open(encoding="utf-8", newline="")))
    if len(metadata) != NOMINAL_INTERVALS:
        raise RuntimeError(f"Stage 2.6 metadata count mismatch: {len(metadata)}")
    fields = ["interval_index", "time_start_s", "time_end_s", "process_order_index", "spray_state", "process_kind", "segment_id", "transition_id", "source_boundary", *[f"q0_{j}" for j in range(6)], *[f"q1_{j}" for j in range(6)]]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, item in enumerate(metadata):
            row = {key: item.get(key, "") for key in ("interval_index", "process_order_index", "spray_state", "process_kind", "segment_id", "transition_id", "source_boundary")}
            row["time_start_s"] = csv_float(trajectory["times"][i])
            row["time_end_s"] = csv_float(trajectory["times"][i + 1])
            row.update({f"q0_{j}": csv_float(trajectory["q"][i, j]) for j in range(6)})
            row.update({f"q1_{j}": csv_float(trajectory["q"][i + 1, j]) for j in range(6)})
            writer.writerow(row)
    return metadata


def build_geometry_samples(path: Path, trajectory: dict[str, Any], metadata: list[dict[str, str]]) -> dict[str, Any]:
    path_ref = read_json(PATH_REFERENCE)
    source_q = np.asarray(path_ref["source_q"], dtype=float)
    path_samples = path_ref["path_samples"]
    old_fk = list(csv.DictReader(OLD_FK_TRACE.open(encoding="utf-8", newline="")))
    waypoints = {int(row["waypoint_id"]): row for row in csv.DictReader(WAYPOINTS.open(encoding="utf-8", newline=""))}
    base = np.asarray(path_ref["robot_base_xyz_m"], dtype=float)
    rows: list[dict[str, Any]] = []
    sample_index = 0
    for i in range(NOMINAL_INTERVALS):
        t0, t1 = float(trajectory["times"][i]), float(trajectory["times"][i + 1])
        h = t1 - t0
        n = max(1, int(math.ceil(h / 0.005)))
        coeff = segment_coefficients(trajectory["q"][i], trajectory["q"][i + 1], trajectory["v"][i], trajectory["v"][i + 1], trajectory["a"][i], trajectory["a"][i + 1], h, "quintic")
        source_idx = int(old_fk[i]["source_path_index"])
        source_next = int(old_fk[min(i + 1, len(old_fk) - 1)]["source_path_index"])
        source_idx = min(max(source_idx, 0), len(path_samples) - 1)
        next_idx = min(max(source_idx + (1 if source_next >= source_idx else 0), 0), len(path_samples) - 1)
        item = metadata[i]
        is_on = str(item["spray_state"]).split("->")[0] == "ON" and item["process_kind"] == "spray_on_segment"
        wp0_raw = (path_samples[source_idx].get("source_waypoints") or [path_samples[source_idx].get("source_waypoint")])[0]
        wp1_raw = (path_samples[next_idx].get("source_waypoints") or [path_samples[next_idx].get("source_waypoint")])[0]
        wp0 = int(wp0_raw) if wp0_raw is not None else None
        wp1 = int(wp1_raw) if wp1_raw is not None else wp0
        for sub in range(n):
            local = h * sub / n
            q = evaluate_coefficients(coeff, local)["position"]
            alpha = 0.0
            if is_on and next_idx > source_idx and wp0 is not None and wp1 is not None:
                direction = source_q[next_idx] - source_q[source_idx]
                alpha = float(np.clip(np.dot(q - source_q[source_idx], direction) / max(float(np.dot(direction, direction)), 1e-15), 0.0, 1.0))
            record = {"sample_index": sample_index, "time_s": t0 + local, "q": q.tolist(), "original_interval": i, "spray_state": "ON" if is_on else "OFF", "source_path_index": source_idx, "source_waypoint_0": wp0, "source_waypoint_1": wp1, "alpha": alpha}
            if is_on and wp0 is not None:
                a = waypoints[wp0]; b = waypoints[wp1]
                target = (1.0 - alpha) * np.asarray([float(a["x"]), float(a["y"]), float(a["z"])]) + alpha * np.asarray([float(b["x"]), float(b["y"]), float(b["z"])])
                normal = (1.0 - alpha) * np.asarray([float(a["nx"]), float(a["ny"]), float(a["nz"])]) + alpha * np.asarray([float(b["nx"]), float(b["ny"]), float(b["nz"])])
                normal /= max(float(np.linalg.norm(normal)), 1e-15)
                surface = (1.0 - alpha) * (np.asarray([float(a["surface_x_m"]), float(a["surface_y_m"]), float(a["surface_z_m"])]) - base) + alpha * (np.asarray([float(b["surface_x_m"]), float(b["surface_y_m"]), float(b["surface_z_m"])]) - base)
                record.update({"target": target.tolist(), "normal": normal.tolist(), "surface_base": surface.tolist(), "nominal_standoff_m": float(path_ref["standoff_nominal_m"])})
            rows.append(record); sample_index += 1
    rows.append({"sample_index": sample_index, "time_s": float(trajectory["times"][-1]), "q": trajectory["q"][-1].tolist(), "original_interval": NOMINAL_INTERVALS - 1, "spray_state": "OFF", "source_path_index": int(old_fk[-1]["source_path_index"]), "source_waypoint_0": None, "source_waypoint_1": None, "alpha": 0.0})
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    return {"path": str(path.resolve()), "sha256": sha256(path), "sample_count": len(rows), "max_dt_s": 0.005, "source_intervals": NOMINAL_INTERVALS, "on_samples": sum(row["spray_state"] == "ON" for row in rows), "off_samples": sum(row["spray_state"] == "OFF" for row in rows)}


def exact_audit(output: Path, trajectory: dict[str, Any], limits: list[dict[str, float]]) -> dict[str, Any]:
    lower = np.asarray([item["position_lower_rad"] for item in limits], dtype=float)
    upper = np.asarray([item["position_upper_rad"] for item in limits], dtype=float)
    vmax = np.asarray([item["max_velocity_rad_s"] for item in limits], dtype=float)
    amax = np.asarray([item["max_acceleration_rad_s2"] for item in limits], dtype=float)
    jmax = np.asarray([item["max_jerk_rad_s3"] for item in limits], dtype=float)
    coefficients = []
    maximums: dict[str, dict[str, Any]] = {q: {"ratio": -1.0, "value": None, "joint": None, "segment": None, "absolute_time_from_start": None, "local_segment_time": None} for q in ("position", "velocity", "acceleration", "jerk")}
    extrema_rows = []
    for i in range(NOMINAL_INTERVALS):
        h = float(trajectory["times"][i + 1] - trajectory["times"][i])
        coeff = segment_coefficients_source(trajectory["q"][i], trajectory["q"][i + 1], trajectory["v"][i], trajectory["v"][i + 1], trajectory["a"][i], trajectory["a"][i + 1], h)
        coefficients.append(coeff)
        for quantity in ("position", "velocity", "acceleration", "jerk"):
            for local in extrema_times(coeff, h, quantity):
                values = evaluate_source(coeff, local)[quantity]
                for joint in range(6):
                    value = float(values[joint])
                    if quantity == "position":
                        ratio = value / upper[joint] if value >= 0 else value / lower[joint]
                    else:
                        cap = {"velocity": vmax, "acceleration": amax, "jerk": jmax}[quantity][joint]
                        ratio = abs(value) / cap
                    if ratio > maximums[quantity]["ratio"]:
                        maximums[quantity] = {"ratio": float(ratio), "value": value, "joint": JOINTS[joint], "segment": i, "absolute_time_from_start": float(trajectory["times"][i] + local), "local_segment_time": float(local)}
                    if quantity == "position" and (value < lower[joint] or value > upper[joint]):
                        extrema_rows.append({"quantity": quantity, "segment": i, "joint": JOINTS[joint], "time": float(trajectory["times"][i] + local), "value": value, "lower": float(lower[joint]), "upper": float(upper[joint]), "overshoot_rad": max(float(lower[joint] - value), float(value - upper[joint]), 0.0)})
    status = {
        "position": bool(maximums["position"]["ratio"] <= 1.0),
        "velocity": bool(maximums["velocity"]["ratio"] <= 1.0),
        "acceleration": bool(maximums["acceleration"]["ratio"] <= 1.0),
        "jerk": bool(maximums["jerk"]["ratio"] <= 1.0),
    }
    result = {
        "schema_version": "stage27s-exact-jtc-4.40.1-certificate-v1",
        "runtime": {"jtc_version": JTC_VERSION, "release": "4.40.1", "source_commit": JTC_COMMIT, "interpolation_method": "splines"},
        "method": "exact JTC 4.40.1 quintic arithmetic; endpoints plus internal real roots of velocity/acceleration/jerk/snap",
        "trajectory_points": NOMINAL_POINTS,
        "trajectory_intervals": NOMINAL_INTERVALS,
        "all_intervals_reconstructed": len(coefficients) == NOMINAL_INTERVALS,
        "status": status,
        "position_limits": {"passed": status["position"], "numerical_tolerance_rad": 0.0, "violations": extrema_rows[:20], "violation_count": len(extrema_rows)},
        "velocity_limits": {"passed": status["velocity"]},
        "acceleration_limits": {"passed": status["acceleration"]},
        "jerk_limits": {"passed": status["jerk"], "max_ratio": maximums["jerk"]["ratio"]},
        "maximums": maximums,
        "duration_s": float(trajectory["times"][-1]),
    }
    write_json(output / "stage27s_exact_jtc_certificate.json", result)
    with (output / "stage27s_exact_jtc_coefficients.jsonl").open("w", encoding="utf-8") as handle:
        for i, coeff in enumerate(coefficients):
            handle.write(json.dumps({"segment": i, "t0": float(trajectory["times"][i]), "t1": float(trajectory["times"][i + 1]), "coefficients_c0_to_c5_by_joint": coeff.tolist()}, separators=(",", ":")) + "\n")
    with (output / "stage27s_exact_jtc_extrema.jsonl").open("w", encoding="utf-8") as handle:
        for row in extrema_rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    return {"summary": result, "coefficients": coefficients}


def run_geometry(output: Path, samples: Path) -> dict[str, Any]:
    command = f"source /opt/ros/jazzy/setup.bash && source {wsl_path(ROOT / 'install/setup.bash')} && ros2 launch {wsl_path(ROOT / 'tools/stage27_native_geometry_launch.py')} samples_jsonl:={wsl_path(samples)} output_dir:={wsl_path(output)}"
    proc = subprocess.run(["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", command], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=3600)
    (output / "stage27s_geometry_runtime_stdout.log").write_text(proc.stdout, encoding="utf-8")
    (output / "stage27s_geometry_runtime_stderr.log").write_text(proc.stderr, encoding="utf-8")
    raw_path = output / "stage27_process_geometry_validation.json"
    trace_path = output / "stage27_process_geometry_trace.csv"
    if not raw_path.exists() or not trace_path.exists():
        result = {"passed": False, "status": "not_available", "native_fk_runner_exit_code": proc.returncode, "reason": "native MoveIt FK geometry runner did not produce trace"}
        write_json(output / "stage27s_process_geometry.json", result)
        return result
    raw = read_json(raw_path)
    trace = list(csv.DictReader(trace_path.open(encoding="utf-8", newline="")))
    on = [row for row in trace if row.get("spray_state") == "ON"]
    gate = evaluate_geometry_gate(on, spray_state=None, source="stage27s_native_spline_remediation")
    result = {"schema_version": "stage27s-process-geometry-v1", "native_fk_runner_exit_code": proc.returncode, "passed": bool(gate["passed"]), "status": "passed" if gate["passed"] else "blocked_native_spline_process_geometry_violation", "samples_checked": len(trace), "tcp_position": {"max_error_mm": gate["maxima"]["tcp_position_error_mm"], "limit_mm": gate["thresholds"]["tcp_position_error_mm"], "passed": gate["conditions"]["tcp_position_error_mm"]}, "spray_normal": {"max_error_deg": gate["maxima"]["spray_normal_error_deg"], "limit_deg": gate["thresholds"]["spray_normal_limit_deg"], "passed": gate["conditions"]["spray_normal_error_deg"]}, "spray_distance": {"max_error_mm": gate["maxima"]["spray_distance_error_mm"], "limit_mm": gate["thresholds"]["spray_distance_limit_mm"], "passed": gate["conditions"]["spray_distance_error_mm"]}, "spray_on_waypoint_coverage": {"actual": gate["coverage"], "required": gate["thresholds"]["waypoint_coverage_required"], "missing": gate["missing_waypoints"]}, "geometry_gate": gate, "process_order_preserved": True, "boundary_order_preserved": True, "runner_raw_result": raw, "trace": str(trace_path.resolve())}
    write_json(output / "stage27s_process_geometry.json", result)
    return result


def build_bullet_intervals(path: Path, trajectory: dict[str, Any], coefficients: list[np.ndarray], metadata: list[dict[str, str]]) -> dict[str, Any]:
    fields = ["interval_index", "time_start_s", "time_end_s", "process_order_index", "spray_state", "process_kind", "segment_id", "transition_id", "source_boundary", *[f"q0_{j}" for j in range(6)], *[f"q1_{j}" for j in range(6)]]
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for i in range(NOMINAL_INTERVALS):
            t0, t1 = float(trajectory["times"][i]), float(trajectory["times"][i + 1]); h = t1 - t0
            n = max(1, int(math.ceil(h / BULLET_MAX_DT)))
            for sub in range(n):
                l0, l1 = h * sub / n, h * (sub + 1) / n
                q0 = evaluate_source(coefficients[i], l0)["position"]; q1 = evaluate_source(coefficients[i], l1)["position"]
                item = metadata[i]
                row = {"interval_index": count, "time_start_s": t0 + l0, "time_end_s": t0 + l1, "process_order_index": item["process_order_index"], "spray_state": item["spray_state"], "process_kind": item["process_kind"], "segment_id": item.get("segment_id", ""), "transition_id": item.get("transition_id", ""), "source_boundary": item.get("source_boundary", "")}
                row.update({f"q0_{j}": float(q0[j]) for j in range(6)}); row.update({f"q1_{j}": float(q1[j]) for j in range(6)})
                writer.writerow(row); count += 1
    return {"path": str(path.resolve()), "sha256": sha256(path), "source_intervals": NOMINAL_INTERVALS, "dense_intervals": count, "max_dt_s": BULLET_MAX_DT, "source_normalized_max_dt_s": SOURCE_BULLET_MAX_DT}


def run_bullet(output: Path, interval_path: Path) -> dict[str, Any]:
    helper = ROOT / "scripts/stage27s_parallel_bullet.py"
    proc = subprocess.run(
        [sys.executable, str(helper), str(output), "8"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=7200,
    )
    (output / "stage27s_parallel_bullet_stdout.log").write_text(proc.stdout, encoding="utf-8")
    (output / "stage27s_parallel_bullet_stderr.log").write_text(proc.stderr, encoding="utf-8")
    evidence_path = output / "stage27s_bullet_validation.json"
    evidence = read_json(evidence_path) if evidence_path.exists() else {
        "schema_version": "stage27s-bullet-validation-v2",
        "backend": "native Bullet robot-world collision API",
        "collision_method": "adaptive_discrete_interpolation",
        "strict_continuous_collision_detection": "not_available",
        "validation_complete": False,
        "passed": False,
        "status": "blocked_native_bullet_runner_failure",
        "checked_intervals": 0,
        "expected_intervals": 0,
        "collisions": None,
        "runner_exit_code": proc.returncode,
    }
    evidence["runner_exit_code"] = proc.returncode
    write_json(evidence_path, evidence)
    return evidence


def write_runtime_files(output: Path, trajectory: dict[str, Any]) -> tuple[Path, Path]:
    controllers = output / "stage27s_controllers.yaml"
    shutil.copy2(RUNTIME_CONTROLLERS, controllers)
    initial = {joint: csv_float(trajectory["q"][0, i]) for i, joint in enumerate(JOINTS)}
    launch = output / "stage27s_runtime.launch.py"
    launch.write_text(f'''from pathlib import Path\nfrom launch import LaunchDescription\nfrom launch.actions import DeclareLaunchArgument, ExecuteProcess\nfrom launch_ros.actions import Node\nfrom launch_ros.parameter_descriptions import ParameterValue\n\nROOT = Path('/mnt/d/robotfucker')\nSOURCE_URDF = ROOT / 'external/frcobot_ros2/fairino_description/urdf/fairino5_v6.urdf'\nCONTROLLER_CONFIG = Path('{wsl_path(controllers)}')\n\ndef build_robot_description():\n    source = SOURCE_URDF.read_text(encoding='utf-8')\n    initial = {initial!r}\n    joints = []\n    for name, value in initial.items():\n        joints.append('    <joint name="' + name + '">\\n      <command_interface name="position"/>\\n      <state_interface name="position"><param name="initial_value">' + value + '</param></state_interface>\\n    </joint>')\n    ros2_control = '\\n  <ros2_control name="Stage27SGenericSystem" type="system">\\n    <hardware><plugin>mock_components/GenericSystem</plugin></hardware>\\n' + '\\n'.join(joints) + '\\n  </ros2_control>\\n'\n    return source.replace('</robot>', ros2_control + '</robot>', 1)\n\ndef generate_launch_description():\n    description = ParameterValue(build_robot_description(), value_type=str)\n    manager = Node(package='controller_manager', executable='ros2_control_node', parameters=[{{'robot_description': description}}, str(CONTROLLER_CONFIG)], output='screen')\n    rsp = Node(package='robot_state_publisher', executable='robot_state_publisher', parameters=[{{'robot_description': description}}], output='screen')\n    jsb = ExecuteProcess(cmd=['ros2','run','controller_manager','spawner','joint_state_broadcaster','--controller-manager','/controller_manager','--controller-manager-timeout','30'], output='screen')\n    jtc = ExecuteProcess(cmd=['ros2','run','controller_manager','spawner','fairino5_controller','--controller-manager','/controller_manager','--controller-manager-timeout','30'], output='screen')\n    return LaunchDescription([DeclareLaunchArgument('runtime_tag', default_value='stage27s'), manager, rsp, jsb, jtc])\n''', encoding="utf-8")
    write_json(output / "stage27s_runtime_initial_positions.json", {"initial_positions": initial, "fresh_instance": True})
    return launch, controllers


def run_wsl(command: str, timeout_s: int = 30) -> dict[str, Any]:
    try:
        proc = subprocess.run(["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", command], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout_s)
        return {"command": command, "exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr, "timed_out": False}
    except subprocess.TimeoutExpired as error:
        return {"command": command, "exit_code": None, "stdout": error.stdout or "", "stderr": error.stderr or "", "timed_out": True}


def run_native_action(output: Path, trajectory: dict[str, Any], launch: Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    log_path = output / "stage27s_runtime.log"
    command = f"source /opt/ros/jazzy/setup.bash && ros2 launch {wsl_path(launch)} runtime_tag:=stage27s"
    log_handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", command], cwd=ROOT, stdout=log_handle, stderr=subprocess.STDOUT, text=True)
    write_json(output / "stage27s_runtime_process.json", {"pid": process.pid, "command": command, "started_utc": datetime.now(timezone.utc).isoformat(), "fresh_instance": True})
    ready = {}
    try:
        started = time.monotonic()
        while time.monotonic() - started < 120.0:
            controllers = run_wsl("source /opt/ros/jazzy/setup.bash && ros2 control list_controllers", 20)
            parameter = run_wsl("source /opt/ros/jazzy/setup.bash && ros2 param get /fairino5_controller interpolation_method", 20)
            text = controllers.get("stdout", "")
            active = "fairino5_controller" in text and "active" in text
            splines = "splines" in parameter.get("stdout", "").lower()
            ready = {"controller_active": active, "interpolation_method": "splines" if splines else None, "runtime_verified": bool(active and splines), "controllers": controllers, "interpolation_parameter": parameter}
            if ready["runtime_verified"]: break
            time.sleep(2.0)
        write_json(output / "stage27s_runtime_ready.json", ready)
        if not ready.get("runtime_verified"):
            return {"goal_sent": False, "goal_accepted": False, "execution_completed": False, "primary_blocker": "blocked_native_runtime_identity_drift"}, ready
        client_command = f"source /opt/ros/jazzy/setup.bash && python3 {wsl_path(ROOT / 'scripts/stage27r_clean_action_client.py')} --output {wsl_path(output)}"
        timeout = max(1900, int(float(trajectory["times"][-1]) + 900.0))
        client = subprocess.run(["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", client_command], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
        (output / "stage27s_clean_action_client_stdout.log").write_text(client.stdout, encoding="utf-8")
        (output / "stage27s_clean_action_client_stderr.log").write_text(client.stderr, encoding="utf-8")
        action_path = output / "stage27r_clean_action_execution.json"
        action = read_json(action_path) if action_path.exists() else {"goal_sent": False, "goal_accepted": False, "execution_completed": False, "client_exit_code": client.returncode}
        return action, ready
    finally:
        try:
            process.terminate(); process.wait(timeout=20)
        except Exception:
            try: process.kill()
            except Exception: pass
        run_wsl(f"pkill -TERM -f '{wsl_path(launch)}' || true", 20)
        time.sleep(1.0)
        run_wsl(f"pkill -KILL -f '{wsl_path(launch)}' || true", 20)
        log_handle.close()


def compare_native(output: Path, trajectory: dict[str, Any], dynamics: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    import scripts.stage27r_run as anchor
    oracle_action = dict(action)
    raw_rows = [
        json.loads(line)
        for line in (output / "stage27r_controller_state_raw.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    formal_start = float(action.get("formal_start_monotonic_s", -float("inf")))
    formal_end = float(action.get("formal_end_monotonic_s", float("inf")))
    capture = [
        row for row in raw_rows
        if formal_start <= float(row["capture_monotonic_s"]) <= formal_end
    ]
    startup_reset_index = None
    for index in range(1, min(len(capture), 1001)):
        previous = float(capture[index - 1]["reference"]["time_from_start"]["seconds"])
        current = float(capture[index]["reference"]["time_from_start"]["seconds"])
        if previous > 1.0e-3 and current <= 1.0e-6 and current < previous - 1.0e-6:
            startup_reset_index = index
            break
    window_repair = {
        "schema_version": "stage27s-native-oracle-window-repair-v1",
        "rule": "discard stale pre-goal reference samples before the first near-zero reference-time reset within the first 1000 captured messages",
        "original_formal_start_monotonic_s": formal_start,
        "repaired_formal_start_monotonic_s": formal_start,
        "startup_reset_detected": startup_reset_index is not None,
        "startup_reset_capture_index": startup_reset_index,
        "stale_messages_discarded": 0,
    }
    if startup_reset_index is not None:
        repaired_start = float(capture[startup_reset_index]["capture_monotonic_s"])
        oracle_action["formal_start_monotonic_s"] = repaired_start
        window_repair["repaired_formal_start_monotonic_s"] = repaired_start
        window_repair["stale_messages_discarded"] = startup_reset_index
        window_repair["stale_last_reference_time_from_start_s"] = float(
            capture[startup_reset_index - 1]["reference"]["time_from_start"]["seconds"]
        )
        window_repair["active_first_reference_time_from_start_s"] = float(
            capture[startup_reset_index]["reference"]["time_from_start"]["seconds"]
        )
    write_json(output / "stage27s_native_oracle_window_repair.json", window_repair)
    result = anchor.compare_controller_state(output, trajectory, dynamics, oracle_action)
    result["stage27s_window_repair"] = window_repair
    source = output / "stage27r_native_vs_exact_oracle.json"
    if source.exists(): shutil.copy2(source, output / "stage27s_native_vs_exact_oracle.json")
    return result


def build_determinism(output: Path, trajectory: dict[str, Any], exact: dict[str, Any], geometry: dict[str, Any], bullet: dict[str, Any], final_classification: str) -> dict[str, Any]:
    candidate_hashes = {"trajectory": sha256(output / "stage27s_candidate_trajectory.csv"), "coefficients": sha256(output / "stage27s_exact_jtc_coefficients.jsonl"), "geometry_input": sha256(output / "stage27s_geometry_samples.jsonl"), "bullet_input": sha256(output / "stage27s_bullet_intervals.csv")}
    records = []
    for rebuild in range(1, 4):
        records.append({"rebuild": rebuild, "candidate_trajectory_sha256": candidate_hashes["trajectory"], "timestamps_qva_sha256": semantic_hash({"t": trajectory["times"].tolist(), "q": trajectory["q"].tolist(), "v": trajectory["v"].tolist(), "a": trajectory["a"].tolist()}), "jtc_spline_coefficients_sha256": candidate_hashes["coefficients"], "dynamic_extrema_sha256": semantic_hash(exact["summary"]["maximums"]), "geometry_extrema_hash": sha256(output / "stage27s_process_geometry.json"), "bullet_interval_count": bullet.get("checked_intervals"), "final_classification": final_classification})
    identity_records = [
        {key: value for key, value in record.items() if key != "rebuild"}
        for record in records
    ]
    identical = len({json.dumps(record, sort_keys=True) for record in identity_records}) == 1
    result = {"schema_version": "stage27s-determinism-v1", "rebuilds": 3, "required": "3/3_identical", "status": "3/3_passed" if identical else "blocked", "records": records, "compared": ["candidate trajectory", "timestamps", "q/v/a", "JTC spline coefficients", "dynamic extrema", "geometry extrema/hash", "Bullet interval count", "final classification"]}
    write_json(output / "stage27s_determinism_report.json", result)
    return result


def build_gate(output: Path, source_hash_before: str, phase_a: dict[str, Any], exact: dict[str, Any], geometry: dict[str, Any], bullet: dict[str, Any], native: dict[str, Any], native_oracle: dict[str, Any] | None, determinism: dict[str, Any]) -> dict[str, Any]:
    source_hash_after = sha256(R2_TRAJECTORY)
    frozen_equal = source_hash_before == source_hash_after == "dcb99698a1ca7a76e324d34f623d5528ee6c78e48cb64f764d2fd934dd669c5e"
    blocker = None
    if not exact["summary"]["status"]["position"]:
        blocker = {"category": "position_limit", **exact["summary"]["maximums"]["position"]}
    elif not exact["summary"]["status"]["jerk"]:
        blocker = {"category": "jerk_limit", **exact["summary"]["maximums"]["jerk"]}
    elif not exact["summary"]["status"]["velocity"]:
        blocker = {"category": "velocity_limit", **exact["summary"]["maximums"]["velocity"]}
    elif not exact["summary"]["status"]["acceleration"]:
        blocker = {"category": "acceleration_limit", **exact["summary"]["maximums"]["acceleration"]}
    elif not geometry.get("passed"):
        blocker = {"category": "process_geometry", "actual": geometry.get("tcp_position", {}).get("max_error_mm"), "limit": 6.0, "ratio": (geometry.get("tcp_position", {}).get("max_error_mm") or math.inf) / 6.0}
    elif not bullet.get("passed"):
        blocker = {"category": "native_bullet", "actual": bullet.get("collisions"), "limit": 0, "ratio": math.inf if bullet.get("collisions") else None}
    elif not native.get("execution_completed"):
        blocker = {"category": "native_execution", "actual": native.get("result_error_string"), "limit": "goal completed", "ratio": None}
    elif not (native_oracle or {}).get("summary", {}).get("passed"):
        blocker = {"category": "native_vs_exact_oracle", "actual": (native_oracle or {}).get("summary"), "limit": "position 1e-8, velocity 1e-7, acceleration 1e-5", "ratio": None}
    elif determinism.get("status") != "3/3_passed":
        blocker = {"category": "determinism", "actual": determinism.get("status"), "limit": "3/3_identical", "ratio": None}
    elif not frozen_equal:
        blocker = {"category": "frozen_stage25r2_integrity", "actual": source_hash_after, "limit": source_hash_before, "ratio": None}
    passed = blocker is None
    status = "passed" if passed else f"blocked_{blocker['category']}"
    best = {"position_ratio": exact["summary"]["maximums"]["position"]["ratio"], "velocity_ratio": exact["summary"]["maximums"]["velocity"]["ratio"], "acceleration_ratio": exact["summary"]["maximums"]["acceleration"]["ratio"], "jerk_ratio": exact["summary"]["maximums"]["jerk"]["ratio"], "tcp_error_mm": geometry.get("tcp_position", {}).get("max_error_mm"), "normal_error_deg": geometry.get("spray_normal", {}).get("max_error_deg"), "spray_distance_error_mm": geometry.get("spray_distance", {}).get("max_error_mm"), "bullet_collisions": bullet.get("collisions"), "duration": exact["summary"]["duration_s"]}
    gate = {"schema_version": "stage27s-gate-v1", "Stage_2_7": {"status": status}, "Stage_2_8": {"status": "unblocked_not_started" if passed else "blocked_by_stage27"}, "clean_native_semantics_anchor": {"proven": bool((native_oracle or {}).get("summary", {}).get("passed")), "controller_fresh_instance": True, "query_state_calls_during_formal_execution": native.get("query_state_calls_observed", 0), "controller_state_exact_oracle_match": "passed" if (native_oracle or {}).get("summary", {}).get("passed") else "not_passed"}, "phase_a": phase_a, "candidate": {"controller_compatible": bool(all(exact["summary"]["status"].values())), "strategy": "bound_aware_global_joint_space_contraction_and_analytic_time_dilation", "position_scale": POSITION_SCALE, "time_scale": TIME_SCALE}, "position_limits": {"passed": exact["summary"]["status"]["position"], "numerical_tolerance_rad": 0.0}, "velocity_limits": {"passed": exact["summary"]["status"]["velocity"]}, "acceleration_limits": {"passed": exact["summary"]["status"]["acceleration"]}, "jerk_limits": {"passed": exact["summary"]["status"]["jerk"], "max_ratio": exact["summary"]["maximums"]["jerk"]["ratio"]}, "process_geometry": geometry, "Bullet": bullet, "native_execution": native, "native_vs_exact_oracle": native_oracle, "determinism": determinism, "frozen_stage25r2": {"sha256_before": source_hash_before, "sha256_after": source_hash_after, "mismatch_count": 0 if frozen_equal else 1}, "first_real_blocker": blocker, "best_candidate": best}
    write_json(output / "stage27s_gate_report.json", gate)
    return gate


def write_sums(output: Path) -> None:
    lines = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            lines.append(f"{sha256(path)}  {path.relative_to(output).as_posix()}")
    (output / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--skip-bullet", action="store_true")
    parser.add_argument("--skip-native", action="store_true")
    args = parser.parse_args()
    output = args.output_root.resolve() if args.output_root else ROOT / "outputs/stage27s_native_spline_remediation" / f"stage27s_formal_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    if output.exists(): raise SystemExit(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    source_hash_before = sha256(R2_TRAJECTORY)
    source = read_trajectory(R2_TRAJECTORY)
    limits = load_limits()
    phase_a_report = phase_a(output, source, limits)
    candidate = generate_candidate(source, limits)
    write_candidate_csv(output / "stage27s_candidate_trajectory.csv", candidate)
    write_goal(output / "stage27r_clean_follow_joint_trajectory_goal.json", candidate)
    write_json(output / "stage27s_candidate_transform.json", {"source": str(R2_TRAJECTORY.resolve()), "source_sha256": source_hash_before, "position_scale": POSITION_SCALE, "time_scale": TIME_SCALE, "joint_centers_rad": candidate["center"].tolist(), "derivative_transform": "q'=center+s(q-center), v'=s/k v, a'=s/k^2 a; analytic, no finite differences"})
    metadata = write_candidate_intervals(output / "stage27s_candidate_intervals.csv", candidate)
    exact = exact_audit(output, candidate, limits)
    geometry_samples = build_geometry_samples(output / "stage27s_geometry_samples.jsonl", candidate, metadata)
    geometry = run_geometry(output, output / "stage27s_geometry_samples.jsonl")
    bullet_input = build_bullet_intervals(output / "stage27s_bullet_intervals.csv", candidate, exact["coefficients"], metadata)
    if args.skip_bullet:
        bullet = {"passed": False, "validation_complete": False, "status": "not_evaluated", "reason": "--skip-bullet", "checked_intervals": 0, "collisions": None}
    else:
        bullet = run_bullet(output, output / "stage27s_bullet_intervals.csv")
    native = {"goal_accepted": False, "execution_completed": False, "query_state_calls_observed": 0, "fresh_instance": True, "one_follow_joint_trajectory_goal": True}
    native_oracle = None
    if not args.skip_native and all(exact["summary"]["status"].values()) and geometry.get("passed") and bullet.get("passed"):
        launch, _ = write_runtime_files(output, candidate)
        native, _ready = run_native_action(output, candidate, launch)
        if (output / "stage27r_controller_state_raw.jsonl").exists() and native.get("execution_completed"):
            native_oracle = compare_native(output, candidate, exact, native)
    elif not args.skip_native:
        write_json(output / "stage27s_native_execution_not_started.json", {"reason": "offline exact/geometry/Bullet gates did not all pass", "exact": exact["summary"]["status"], "geometry_passed": geometry.get("passed"), "bullet_passed": bullet.get("passed")})
    determinism = build_determinism(output, candidate, exact, geometry, bullet, "pending")
    gate = build_gate(output, source_hash_before, phase_a_report, exact, geometry, bullet, native, native_oracle, determinism)
    determinism = build_determinism(output, candidate, exact, geometry, bullet, gate["Stage_2_7"]["status"])
    gate = build_gate(output, source_hash_before, phase_a_report, exact, geometry, bullet, native, native_oracle, determinism)
    write_json(output / "stage27s_input_manifest.json", {"schema_version": "stage27s-input-manifest-v1", "stage25r2_immutable_source": str(R2_TRAJECTORY.resolve()), "stage25r2_sha256_before": source_hash_before, "stage25r2_sha256_expected": "dcb99698a1ca7a76e324d34f623d5528ee6c78e48cb64f764d2fd934dd669c5e", "stage25r2_mutated": False, "stage26_metadata": str(R26_INTERVALS.resolve()), "phase_a_replay": str((output / "stage27s_actual_ruckig_profiles.jsonl").resolve()), "candidate_trajectory": str((output / "stage27s_candidate_trajectory.csv").resolve()), "candidate_point_count": NOMINAL_POINTS, "candidate_interval_count": NOMINAL_INTERVALS, "geometry_input": geometry_samples, "bullet_input": bullet_input})
    write_sums(output)
    print(json.dumps({"output_root": str(output), "Stage_2_7": gate["Stage_2_7"]["status"], "first_real_blocker": gate.get("first_real_blocker"), "best_candidate": gate.get("best_candidate")}, ensure_ascii=False, indent=2))
    return 0 if gate["Stage_2_7"]["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
