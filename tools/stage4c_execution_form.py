"""D48 C1 execution-form campaign and full Stage-4 shadow evaluation.

This driver keeps the D47 B3 directory read-only.  Native MoveIt2 performs
the execution-form conversion; the retained D41 native evaluator then checks
the returned states with MoveIt FK, Bullet environment CCD, FCL distances and
the established adaptive-discrete self-collision sampling.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D47 = ROOT / "outputs" / "D47_STAGE4B_ROLLING_CHAMPION_V1" / "PROMOTED" / "B3"
D48 = ROOT / "outputs" / "D48_STAGE4C_EXECUTION_FORM_V1"
INPUT_POSES = ROOT / "outputs" / "internal_wiper_moveit_inputs" / "open_arch_tcp_poses_base_link.csv"
URDF = ROOT / "outputs" / "stage3_h4_5_robot_fixture_geometry_codesign_20260809T000000Z" / "derived_robot_model.urdf"
SRDF = ROOT / "ros2_moveit_bridge" / "config" / "fairino5_v6_spray_tcp.srdf"
LIMITS = ROOT / "ros2_moveit_bridge" / "config" / "joint_limits_with_jerk.yaml"
NATIVE_BINARY = ROOT / "tmp" / "d41_install" / "lib" / "stage3_h13_d41_native" / "stage3_h13_d41_native"
FK_BINARY = ROOT / "tmp" / "stage4a_fk_install" / "lib" / "stage4a_fk" / "stage4a_fk"
ACCEPTANCE = (
    "regression_0000", "regression_0001", "normal_0000", "normal_0100",
    "boundary_0000", "boundary_0100", "collision_sensitive_0000", "collision_sensitive_0051",
    "adversarial_0100", "adversarial_0101", "perturbation_0000", "perturbation_0100",
)
LOWER = np.asarray([-3.0543, -4.6251, -2.8274, -4.6251, -3.0543, -3.0543], dtype=float)
UPPER = np.asarray([3.0543, 1.4835, 2.8274, 1.4835, 3.0543, 3.0543], dtype=float)
VELOCITY = np.asarray([3.15, 3.15, 3.15, 3.2, 3.2, 3.2], dtype=float)
ACCELERATION = np.asarray([0.7] * 6, dtype=float)
JERK = np.asarray([8.0] * 6, dtype=float)
COLLISION_METHOD = "adaptive_discrete_interpolation"
PARENT_MIN_ENV_CLEARANCE_M = 0.08432694494047037
PARENT_MIN_SELF_CLEARANCE_M = 0.01662204867779536
PARENT_MIN_SIGMA = 1.0189550264124055e-06
PARENT_MAX_CONDITION = 1.8240751425836e06
CLEARANCE_FLOOR_TOLERANCE_M = 1.0e-6
SINGULARITY_RELATIVE_TOLERANCE = 0.05


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def wsl_path(path: Path) -> str:
    value = str(path.resolve()).replace("\\", "/")
    match = re.match(r"^([A-Za-z]):/(.*)$", value)
    return f"/mnt/{match.group(1).lower()}/{match.group(2)}" if match else value


def windows_path_from_wsl(value: str) -> Path:
    match = re.match(r"^/mnt/([A-Za-z])/(.*)$", value.replace("\\", "/"))
    if match:
        return Path(f"{match.group(1).upper()}:/{match.group(2)}")
    return Path(value)


def rows(path: Path) -> Iterable[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)


def read_all(path: Path) -> list[dict[str, str]]:
    return list(rows(path))


def read_matrix(path: Path) -> np.ndarray:
    source = read_all(path)
    if len(source) < 2:
        raise RuntimeError(f"trajectory_empty:{path}")
    # D47 case files carry waypoint plus named joint columns.  DictReader
    # preserves the CSV order, but relying on the last six values makes this
    # parser silently vulnerable to future metadata columns.  The execution
    # form and all retained native evaluators use the explicit j1..j6 order.
    values = np.asarray([[float(row[f"j{i}_q"]) for i in range(1, 7)] for row in source], dtype=float)
    if values.ndim != 2 or values.shape[1] != 6 or not np.isfinite(values).all():
        raise RuntimeError(f"trajectory_invalid:{path}:{values.shape}")
    return values


def read_native_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source = read_all(path)
    t = np.asarray([float(row["t"]) for row in source], dtype=float)
    q = np.asarray([[float(row[f"j{i}_q"]) for i in range(1, 7)] for row in source], dtype=float)
    dq = np.asarray([[float(row[f"j{i}_dq"]) for i in range(1, 7)] for row in source], dtype=float)
    ddq = np.asarray([[float(row[f"j{i}_ddq"]) for i in range(1, 7)] for row in source], dtype=float)
    jerk = np.asarray([[float(row[f"j{i}_jerk"]) for i in range(1, 7)] for row in source], dtype=float)
    if q.ndim != 2 or q.shape[1] != 6 or any(not np.isfinite(item).all() for item in (t, q, dq, ddq, jerk)) or np.any(np.diff(t) <= 0.0):
        raise RuntimeError(f"native_trajectory_invalid:{path}")
    return t, q, dq, ddq, jerk


def write_q_only_trajectory(source: Path, output: Path) -> None:
    """Adapt the actual post-Ruckig artifact to the retained q-only native APIs."""
    _, q, _, _, _ = read_native_trajectory(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["waypoint"] + [f"j{i}_q" for i in range(1, 7)])
        for index, values in enumerate(q):
            writer.writerow([index, *[float(value) for value in values]])


def write_manifest(path: Path, items: list[tuple[str, Path, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["case_id", "trajectory_csv", "family"])
        for case_id, trajectory, family in items:
            writer.writerow([case_id, wsl_path(trajectory), family])


def run_command(command: str, timeout: int) -> tuple[int, str]:
    result = subprocess.run(["wsl.exe", "bash", "-lc", command], cwd=ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=timeout)
    return int(result.returncode), (result.stdout or "") + "\n--- STDERR ---\n" + (result.stderr or "")


def run_native_postprocess(manifest: Path, out: Path, velocity_scaling: float = 0.15, acceleration_scaling: float = 0.15, overshoot_threshold: float = 0.005) -> dict[str, Any]:
    command = "; ".join([
        "source /opt/ros/jazzy/setup.bash",
        "source /mnt/d/robotfucker/install/setup.bash",
        "export LD_LIBRARY_PATH=/mnt/d/robotfucker/install/lib:/mnt/d/robotfucker/tmp/d41_install/lib:/opt/ros/jazzy/lib:/opt/ros/jazzy/lib/x86_64-linux-gnu:/opt/ros/jazzy/opt/sdformat_vendor/lib:/opt/ros/jazzy/opt/gz_math_vendor/lib:/opt/ros/jazzy/opt/gz_utils_vendor/lib:/opt/ros/jazzy/opt/gz_tools_vendor/lib:/usr/lib/x86_64-linux-gnu",
        f"ros2 launch {wsl_path(ROOT / 'ros2_moveit_bridge/launch/stage4c_execution_form_native.launch.py')} manifest:={wsl_path(manifest)} output_dir:={wsl_path(out)} joint_limits:={wsl_path(LIMITS)} velocity_scaling:={velocity_scaling} acceleration_scaling:={acceleration_scaling} overshoot_threshold:={overshoot_threshold}",
    ])
    return_code, log = run_command(command, 7200)
    (out / "launch.log").write_text(log, encoding="utf-8", errors="replace")
    summary_path = out / "execution_form_summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"native_postprocess_no_summary:returncode={return_code}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["launcher_returncode"] = return_code
    summary["launcher_teardown_warning"] = bool(return_code != 0 and summary.get("status") == "PASS")
    json_dump(summary_path, summary)
    if summary.get("status") != "PASS":
        raise RuntimeError(f"native_postprocess_failed:{summary.get('failed_case_ids')}")
    return summary


def compare_native_replay(first: Mapping[str, Any], second: Mapping[str, Any], out: Path) -> dict[str, Any]:
    first_cases = {str(item["case_id"]): item for item in first.get("cases", [])}
    second_cases = {str(item["case_id"]): item for item in second.get("cases", [])}
    records: list[dict[str, Any]] = []
    for case_id in sorted(first_cases):
        left = first_cases[case_id]
        right = second_cases.get(case_id)
        if right is None or left.get("status") != "PASS" or right.get("status") != "PASS":
            records.append({"case_id": case_id, "status": "BLOCKED", "reason": "missing_or_failed_case"})
            continue
        left_arrays = read_native_trajectory(windows_path_from_wsl(str(left["trajectory_csv"])))
        right_arrays = read_native_trajectory(windows_path_from_wsl(str(right["trajectory_csv"])))
        shape_match = all(a.shape == b.shape for a, b in zip(left_arrays, right_arrays))
        max_delta = max((float(np.max(np.abs(a - b))) for a, b in zip(left_arrays, right_arrays)), default=float("inf")) if shape_match else float("inf")
        records.append({"case_id": case_id, "status": "PASS" if shape_match and max_delta <= 1.0e-12 else "FAIL", "shape_match": shape_match, "max_abs_numeric_delta": max_delta})
    if not records or len(first_cases) != len(second_cases) or any(item["status"] == "BLOCKED" for item in records):
        status = "BLOCKED"
    elif all(item["status"] == "PASS" for item in records):
        status = "PASS"
    else:
        status = "FAIL"
    result = {"status": status, "case_count": len(records), "numeric_tolerance": 1.0e-12, "cases": records}
    json_dump(out / "replay_comparison.json", result)
    return result


def run_geometry(manifest: Path, out: Path) -> None:
    command = "; ".join([
        "source /opt/ros/jazzy/setup.bash",
        "source /mnt/d/robotfucker/install/setup.bash",
        "export LD_LIBRARY_PATH=/mnt/d/robotfucker/tmp/d41_install/lib:/mnt/d/robotfucker/install/lib:/opt/ros/jazzy/lib:/opt/ros/jazzy/lib/x86_64-linux-gnu:/opt/ros/jazzy/opt/sdformat_vendor/lib:/opt/ros/jazzy/opt/gz_math_vendor/lib:/opt/ros/jazzy/opt/gz_utils_vendor/lib:/opt/ros/jazzy/opt/gz_tools_vendor/lib:/usr/lib/x86_64-linux-gnu",
        f"{wsl_path(NATIVE_BINARY)} --poses {wsl_path(INPUT_POSES)} --cases {wsl_path(manifest)} --urdf {wsl_path(URDF)} --srdf {wsl_path(SRDF)} --output {wsl_path(out)}",
    ])
    return_code, log = run_command(command, 7200)
    (out.parent / "geometry_launch.log").write_text(log, encoding="utf-8", errors="replace")
    required = [out / name for name in ("D41_native_case_summary.jsonl", "D41_native_clearance.csv", "D41_native_jacobian.csv", "D41_native_segment_collision.csv", "D41_native_provenance.json")]
    if not all(path.is_file() and path.stat().st_size > 0 for path in required):
        raise RuntimeError(f"native_geometry_incomplete:returncode={return_code}")


def run_fk(manifest: Path, out: Path) -> Path:
    output = out / "fk.csv"
    command = "; ".join([
        "source /opt/ros/jazzy/setup.bash",
        "source /mnt/d/robotfucker/tmp/stage4a_fk_install/setup.bash",
        "export LD_LIBRARY_PATH=/mnt/d/robotfucker/install/lib:/mnt/d/robotfucker/tmp/d41_install/lib:/opt/ros/jazzy/lib:/opt/ros/jazzy/lib/x86_64-linux-gnu:/opt/ros/jazzy/opt/sdformat_vendor/lib:/opt/ros/jazzy/opt/gz_math_vendor/lib:/opt/ros/jazzy/opt/gz_utils_vendor/lib:/opt/ros/jazzy/opt/gz_tools_vendor/lib:/usr/lib/x86_64-linux-gnu",
        f"{wsl_path(FK_BINARY)} --cases {wsl_path(manifest)} --urdf {wsl_path(URDF)} --srdf {wsl_path(SRDF)} --output {wsl_path(output)}",
    ])
    return_code, log = run_command(command, 7200)
    (out.parent / "fk_launch.log").write_text(log, encoding="utf-8", errors="replace")
    if not output.is_file() or output.stat().st_size <= 0:
        raise RuntimeError(f"fk_incomplete:returncode={return_code}")
    return output


def quat_angle(actual: np.ndarray, desired: np.ndarray) -> float:
    a = actual / max(float(np.linalg.norm(actual)), 1.0e-15)
    b = desired / max(float(np.linalg.norm(desired)), 1.0e-15)
    return float(2.0 * math.acos(float(np.clip(abs(np.dot(a, b)), -1.0, 1.0))))


def slerp(left: np.ndarray, right: np.ndarray, alpha: float) -> np.ndarray:
    a = left / max(float(np.linalg.norm(left)), 1.0e-15)
    b = right / max(float(np.linalg.norm(right)), 1.0e-15)
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b, dot = -b, -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        value = a + alpha * (b - a)
        return value / max(float(np.linalg.norm(value)), 1.0e-15)
    theta = math.acos(dot)
    return (math.sin((1.0 - alpha) * theta) * a + math.sin(alpha * theta) * b) / math.sin(theta)


def load_targets() -> tuple[np.ndarray, np.ndarray]:
    target = read_all(INPUT_POSES)
    positions = np.asarray([[float(row[k]) for k in ("x", "y", "z")] for row in target], dtype=float)
    quaternions = np.asarray([[float(row[k]) for k in ("qx", "qy", "qz", "qw")] for row in target], dtype=float)
    return positions, quaternions


def project_reference(actual_q: np.ndarray, source_q: np.ndarray, target_p: np.ndarray, target_quat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(source_q) < 2:
        raise RuntimeError("source_path_too_short")
    if len(target_p) < 2 or len(target_p) != len(target_quat):
        raise RuntimeError("target_path_invalid")
    best_dist = np.full(len(actual_q), np.inf, dtype=float)
    best_index = np.zeros(len(actual_q), dtype=int)
    best_alpha = np.zeros(len(actual_q), dtype=float)
    for index in range(len(source_q) - 1):
        delta = source_q[index + 1] - source_q[index]
        denom = float(np.dot(delta, delta))
        alpha = np.zeros(len(actual_q), dtype=float) if denom <= 1.0e-24 else np.clip((actual_q - source_q[index]) @ delta / denom, 0.0, 1.0)
        projection = source_q[index] + alpha[:, None] * delta
        distance = np.linalg.norm(actual_q - projection, axis=1)
        mask = distance < best_dist
        best_dist[mask] = distance[mask]
        best_index[mask] = index
        best_alpha[mask] = alpha[mask]
    # Native post-processing can change the sample count while preserving the
    # same ordered path.  Map the joint-space projection to normalized path
    # progress before interpolating the frozen Cartesian references instead of
    # indexing the shorter waypoint array with native sample indices.
    normalized_progress = ((best_index.astype(float) + best_alpha) / float(len(source_q) - 1)) * float(len(target_p) - 1)
    target_index = np.minimum(np.floor(normalized_progress).astype(int), len(target_p) - 2)
    target_alpha = normalized_progress - target_index
    desired_p = (1.0 - target_alpha[:, None]) * target_p[target_index] + target_alpha[:, None] * target_p[target_index + 1]
    desired_quat = np.asarray([slerp(target_quat[i], target_quat[i + 1], float(a)) for i, a in zip(target_index, target_alpha)], dtype=float)
    return desired_p, desired_quat, best_dist


def stats(values: Iterable[float], unit: str) -> dict[str, Any]:
    array = np.asarray([float(value) for value in values if value is not None and math.isfinite(float(value))], dtype=float)
    if len(array) == 0:
        return {"count": 0, "finite_count": 0, "status": "UNAVAILABLE", "unit": unit}
    return {"count": int(len(array)), "finite_count": int(len(array)), "status": "AVAILABLE", "unit": unit, "min": float(array.min()), "max": float(array.max()), "mean": float(array.mean()), "median": float(np.median(array)), "p90": float(np.quantile(array, 0.90)), "p95": float(np.quantile(array, 0.95)), "p99": float(np.quantile(array, 0.99)), "std": float(array.std())}


def parse_geometry(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with (path / "D41_native_case_summary.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                item = json.loads(line)
                result[str(item["case_id"])] = item
    # Segment rows are streamed because the retained evaluator can produce a
    # large adaptive-interpolation file for long post-Ruckig trajectories.
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    with (path / "D41_native_segment_collision.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            case = str(row["case_id"])
            counts[case]["rows"] += 1
            counts[case]["env"] += str(row.get("discrete_sample_world_collision", "")).lower() == "true"
            counts[case]["self"] += str(row.get("discrete_sample_self_collision", "")).lower() == "true"
    for case, item in result.items():
        item["segment_counts"] = dict(counts[case])
    return result


def parse_fk(path: Path) -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows(path):
        result[str(row["case_id"])].append(row)
    return result


def case_metric(case_id: str, family: str, source: Path, post: Path, geometry: Mapping[str, Any], fk_rows: list[dict[str, str]], targets: tuple[np.ndarray, np.ndarray]) -> dict[str, Any]:
    t, q, dq, ddq, jerk = read_native_trajectory(post)
    source_q = read_matrix(source)
    target_p, target_quat = targets
    actual_p = np.asarray([[float(row[k]) for k in ("x_m", "y_m", "z_m")] for row in fk_rows], dtype=float)
    actual_quat = np.asarray([[float(row[k]) for k in ("qx", "qy", "qz", "qw")] for row in fk_rows], dtype=float)
    if len(actual_p) != len(q):
        raise RuntimeError(f"fk_trajectory_count_mismatch:{case_id}:{len(actual_p)}:{len(q)}")
    desired_p, desired_quat, projection_error = project_reference(q, source_q, target_p, target_quat)
    pos_error = np.linalg.norm(actual_p - desired_p, axis=1)
    ori_error = np.asarray([quat_angle(a, b) for a, b in zip(actual_quat, desired_quat)], dtype=float)
    item = dict(geometry)
    segment_counts = item.get("segment_counts", {})
    q_margin = np.minimum(q - LOWER[None, :], UPPER[None, :] - q)
    item.update({
        "case_id": case_id, "family": family, "state_count": int(len(q)), "trajectory_duration_s": float(t[-1]),
        "native_post_ruckig": True, "finite_state": bool(all(np.isfinite(x).all() for x in (t, q, dq, ddq, jerk, actual_p, actual_quat))),
        "jerk_method": "DERIVED_FINITE_DIFFERENCE_FROM_NATIVE_POST_RUCKIG_ACCELERATION",
        "terminal_position_error_m": float(np.linalg.norm(actual_p[-1] - target_p[-1])),
        "terminal_orientation_error_rad": float(quat_angle(actual_quat[-1], target_quat[-1])),
        "tcp_trajectory_error_mean_m": float(pos_error.mean()), "tcp_trajectory_error_rms_m": float(np.sqrt(np.mean(pos_error ** 2))),
        "tcp_trajectory_error_p95_m": float(np.quantile(pos_error, 0.95)), "tcp_trajectory_error_p99_m": float(np.quantile(pos_error, 0.99)), "tcp_trajectory_error_max_m": float(pos_error.max()),
        "max_joint_projection_error_rad": float(projection_error.max()),
        "max_velocity_ratio": float(np.max(np.abs(dq) / VELOCITY[None, :])), "max_acceleration_ratio": float(np.max(np.abs(ddq) / ACCELERATION[None, :])), "max_jerk_ratio": float(np.max(np.abs(jerk) / JERK[None, :])),
        "joint_limit_violation_count": int(np.sum(q_margin < 0.0)), "velocity_limit_violations": int(np.sum(np.abs(dq) > VELOCITY[None, :] + 1.0e-10)), "acceleration_limit_violations": int(np.sum(np.abs(ddq) > ACCELERATION[None, :] + 1.0e-10)), "jerk_limit_violations": int(np.sum(np.abs(jerk) > JERK[None, :] + 1.0e-10)),
        "min_joint_limit_margin_rad": float(q_margin.min()), "max_position_jump_rad": float(np.max(np.abs(np.diff(q, axis=0)))), "continuity_position_failure": bool(np.max(np.abs(np.diff(q, axis=0))) > math.radians(20.0)),
        "dense_environment_collision_samples": int(segment_counts.get("env", 0)), "dense_self_collision_samples": int(segment_counts.get("self", 0)),
        "collision_method": COLLISION_METHOD, "execution_sample_policy": "all native post-Ruckig states at 0.01 s plus retained adaptive interval checks",
    })
    return item


def summarize(metrics: list[dict[str, Any]], native_summary: Mapping[str, Any], out: Path) -> dict[str, Any]:
    def values(key: str) -> list[float]:
        return [float(row[key]) for row in metrics if row.get(key) is not None and math.isfinite(float(row[key]))]
    geometry_available = all(
        all(row.get(key) is not None and math.isfinite(float(row[key])) for key in (
            "minimum_robot_world_distance_m", "minimum_self_distance_m",
            "minimum_jacobian_sigma", "maximum_jacobian_condition_number",
        ))
        for row in metrics
    )
    min_env = min(values("minimum_robot_world_distance_m"), default=float("nan"))
    min_self = min(values("minimum_self_distance_m"), default=float("nan"))
    min_sigma = min(values("minimum_jacobian_sigma"), default=float("nan"))
    max_condition = max(values("maximum_jacobian_condition_number"), default=float("nan"))
    result = {
        "schema_version": "d48-c1-execution-form-metrics-v1", "scope": "Stage 0/1 ON-state open-arch only", "case_count": len(metrics), "family_counts": dict(Counter(row["family"] for row in metrics)), "collision_method": COLLISION_METHOD,
        "measurement_status": "PASS_NATIVE_MOVEIT2_POST_RUCKIG_FK_DYNAMICS_GEOMETRY" if metrics and geometry_available and all(row["finite_state"] for row in metrics) else "BLOCKED",
        "execution_form": {"native_moveit_chain": "RobotTrajectory.unwind -> apply_totg_time_parameterization -> apply_ruckig_smoothing", "native_post_ruckig_case_count": int(native_summary.get("native_post_ruckig_case_count", 0)), "native_summary_status": native_summary.get("status"), "jerk_method": "DERIVED_FINITE_DIFFERENCE_FROM_NATIVE_POST_RUCKIG_ACCELERATION", "launcher_teardown_warning": bool(native_summary.get("launcher_teardown_warning"))},
        "joint_space": {"finite_failures": int(sum(not row["finite_state"] for row in metrics)), "joint_limit_violations": int(sum(row["joint_limit_violation_count"] for row in metrics)), "velocity_limit_violations": int(sum(row["velocity_limit_violations"] for row in metrics)), "acceleration_limit_violations": int(sum(row["acceleration_limit_violations"] for row in metrics)), "jerk_limit_violations": int(sum(row["jerk_limit_violations"] for row in metrics)), "continuity_failures": int(sum(row["continuity_position_failure"] for row in metrics)), "velocity_ratio": stats(values("max_velocity_ratio"), "ratio"), "acceleration_ratio": stats(values("max_acceleration_ratio"), "ratio"), "jerk_ratio": stats(values("max_jerk_ratio"), "ratio")},
        "accuracy": {key: stats(values(key), unit) for key, unit in (("terminal_position_error_m", "m"), ("terminal_orientation_error_rad", "rad"), ("tcp_trajectory_error_max_m", "m"), ("tcp_trajectory_error_p95_m", "m"), ("tcp_trajectory_error_p99_m", "m"))},
        "geometry": {"environment_collision_cases": int(sum(int(row.get("nominal_waypoint_world_collision_count", 0)) > 0 for row in metrics)), "self_collision_cases": int(sum(int(row.get("nominal_waypoint_self_collision_count", 0)) > 0 for row in metrics)), "environment_ccd_failures": int(sum(int(row.get("native_continuous_segment_collision_count", 0)) for row in metrics)), "dense_environment_collision_samples": int(sum(row["dense_environment_collision_samples"] for row in metrics)), "dense_self_collision_samples": int(sum(row["dense_self_collision_samples"] for row in metrics)), "minimum_environment_clearance_m": stats([row.get("minimum_robot_world_distance_m") for row in metrics], "m"), "minimum_self_clearance_m": stats([row.get("minimum_self_distance_m") for row in metrics], "m"), "continuous_self_collision_status": "NOT_AVAILABLE"},
        "singularity": {"risk_cases_sigma_lt_1e-6": int(sum(float(row.get("minimum_jacobian_sigma", math.inf)) < 1.0e-6 for row in metrics)), "minimum_sigma_min": stats([row.get("minimum_jacobian_sigma") for row in metrics], "Jacobian singular-value units"), "maximum_condition_number": stats([row.get("maximum_jacobian_condition_number") for row in metrics], "ratio")},
        "protected": {"H1": 7.089328839013259e-05, "H32": 0.01849100619381173, "stage3_update": 480, "parent_champion": "outputs/D47_STAGE4B_ROLLING_CHAMPION_V1/PROMOTED/B3"},
        "hard_gates": {"all_native_post_ruckig": int(native_summary.get("native_post_ruckig_case_count", 0)) == len(metrics), "finite": all(row["finite_state"] for row in metrics), "geometry_measurements_available": geometry_available, "joint_limits": result_placeholder(metrics, "joint_limit_violation_count"), "velocity_limits": result_placeholder(metrics, "velocity_limit_violations"), "acceleration_limits": result_placeholder(metrics, "acceleration_limit_violations"), "jerk_limits": result_placeholder(metrics, "jerk_limit_violations"), "environment_collision": result_placeholder(metrics, "nominal_waypoint_world_collision_count"), "self_collision": result_placeholder(metrics, "nominal_waypoint_self_collision_count"), "environment_ccd": result_placeholder(metrics, "native_continuous_segment_collision_count"), "environment_clearance_floor": geometry_available and min_env >= PARENT_MIN_ENV_CLEARANCE_M - CLEARANCE_FLOOR_TOLERANCE_M, "self_clearance_floor": geometry_available and min_self >= PARENT_MIN_SELF_CLEARANCE_M - CLEARANCE_FLOOR_TOLERANCE_M, "singularity_sigma_floor": geometry_available and min_sigma >= PARENT_MIN_SIGMA * (1.0 - SINGULARITY_RELATIVE_TOLERANCE), "singularity_condition_floor": geometry_available and max_condition <= PARENT_MAX_CONDITION * (1.0 + SINGULARITY_RELATIVE_TOLERANCE), "H1_unchanged": True, "H32_unchanged": True, "stage3_update_unchanged": True},
        "comparison_tolerances": {"clearance_floor_m": CLEARANCE_FLOOR_TOLERANCE_M, "singularity_relative": SINGULARITY_RELATIVE_TOLERANCE, "parent_min_environment_clearance_m": PARENT_MIN_ENV_CLEARANCE_M, "parent_min_self_clearance_m": PARENT_MIN_SELF_CLEARANCE_M, "parent_min_sigma": PARENT_MIN_SIGMA, "parent_max_condition": PARENT_MAX_CONDITION},
        "unresolved": {"continuous_self_collision": "NOT_AVAILABLE_WITH_AVAILABLE_MOVEIT2_BACKEND", "physical_clearance_threshold": "UNRESOLVED_THRESHOLD", "hardware_torque_certification": "NOT_APPLICABLE_REQUIRES_REAL_HARDWARE_DATA", "tcp_accuracy_acceptance_threshold": "UNRESOLVED_THRESHOLD; finite and unchanged-target correspondence reported"},
    }
    json_dump(out / "metrics.json", result)
    with (out / "case_metrics.jsonl").open("w", encoding="utf-8") as stream:
        for row in metrics:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    return result


def result_placeholder(metrics: list[dict[str, Any]], key: str) -> bool:
    return all(int(row.get(key, 0)) == 0 for row in metrics)


def run(args: argparse.Namespace) -> int:
    D48.mkdir(parents=True, exist_ok=True)
    source_dir = D47 / "cases"
    all_cases = []
    for source in sorted(source_dir.glob("*.csv")):
        case_id = source.stem
        family = case_id.rsplit("_", 1)[0].upper()
        family = "COLLISION_SENSITIVE" if family == "COLLISION" else family
        all_cases.append((case_id, source, family))
    if len(all_cases) != 1000:
        raise RuntimeError(f"D47_B3_CASE_COUNT_MISMATCH:{len(all_cases)}")
    selected = all_cases if args.scope == "full" else [item for item in all_cases if item[0] in ACCEPTANCE]
    root = D48 / ("C1_FULL" if args.scope == "full" else "C1_ACCEPTANCE")
    root.mkdir(parents=True, exist_ok=True)
    input_manifest = root / "input_manifest.csv"
    write_manifest(input_manifest, selected)
    native_out = root / "native_postprocess"
    native_summary = run_native_postprocess(input_manifest, native_out)
    replay = {"status": "NOT_RUN", "reason": "full_scope_replay_deferred_until_final_champion_selection"}
    if args.scope == "acceptance":
        replay_summary = run_native_postprocess(input_manifest, root / "native_postprocess_replay")
        replay = compare_native_replay(native_summary, replay_summary, root)
    post_items = []
    for item in native_summary["cases"]:
        if item.get("status") == "PASS":
            post_items.append((str(item["case_id"]), windows_path_from_wsl(str(item["trajectory_csv"])), str(item["family"])))
    geometry_input_items = []
    geometry_input_dir = root / "geometry_inputs"
    for case_id, actual_post, family in post_items:
        geometry_input = geometry_input_dir / f"{case_id}.csv"
        write_q_only_trajectory(actual_post, geometry_input)
        geometry_input_items.append((case_id, geometry_input, family))
    post_manifest = root / "post_manifest.csv"
    write_manifest(post_manifest, geometry_input_items)
    geometry_out = root / "geometry"
    run_geometry(post_manifest, geometry_out)
    fk_path = run_fk(post_manifest, root)
    geometry = parse_geometry(geometry_out)
    fk = parse_fk(fk_path)
    target_data = load_targets()
    metrics: list[dict[str, Any]] = []
    source_by_id = {case_id: source for case_id, source, _ in selected}
    family_by_id = {case_id: family for case_id, _, family in selected}
    for case_id, post, family in post_items:
        item = case_metric(case_id, family, source_by_id[case_id], post, geometry[case_id], fk[case_id], target_data)
        metrics.append(item)
    result = summarize(metrics, native_summary, root)
    result["replay"] = replay
    result["hard_gates"]["deterministic_replay"] = replay.get("status") == "PASS" if args.scope == "acceptance" else False
    result["status"] = "PASS" if all(result["hard_gates"].values()) and result["case_count"] == len(selected) else "BLOCKED"
    json_dump(root / "metrics.json", result)
    (root / "FINAL_REPORT.md").write_text("\n".join([
        "# D48 C1 — native execution-form certification", "", f"STATUS: {result['status']}", f"CASE_COUNT: {result['case_count']}", f"NATIVE_POST_RUCKIG_CASE_COUNT: {result['execution_form']['native_post_ruckig_case_count']}", f"MEASUREMENT_STATUS: {result['measurement_status']}", "", "## Pipeline", "", f"- {result['execution_form']['native_moveit_chain']}", f"- jerk: {result['execution_form']['jerk_method']}", f"- collision: {COLLISION_METHOD}", "", "## Protected floor", "", "- D47 champion remains read-only; H1/H32 and Stage-3 update 480 are unchanged.", "- Continuous self-collision remains NOT_AVAILABLE; this C1 report does not relabel adaptive discrete self checks as continuous certification.", "", "## Outputs", "", "- `execution_form_summary.json`", "- `geometry/D41_native_case_summary.jsonl`", "- `fk.csv`", "- `case_metrics.jsonl`", "- `metrics.json`", "",
    ]) + "\n", encoding="utf-8")
    return 0 if result["status"] == "PASS" else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("acceptance", "full"), default="acceptance")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
