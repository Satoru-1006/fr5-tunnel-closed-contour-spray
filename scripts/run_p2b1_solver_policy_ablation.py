#!/usr/bin/env python3
"""Run the frozen-input P2-B1 DLS/Ruckig variants and the original P2-A J6+ scan.

The source checkout must be clean and committed. Builds and all generated
evidence belong in an external scratch directory supplied on the command line.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys
import xml.etree.ElementTree as ET
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def ensure_project_root_importable() -> None:
    root = str(ROOT)
    if root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)


ensure_project_root_importable()
SOURCE_BASE = "ce18bfda76931385e2fbb1f923d8f980b5c487a4"
BRANCH = "codex/fr5-p2b1-solver-causal-ablation-20260926"
POSES = ROOT / "outputs/internal_wiper_moveit_inputs/open_arch_tcp_poses_base_link.csv"
SEEDS = ROOT / "outputs/internal_wiper_moveit_inputs/open_arch_seed_joints.csv"
D39_SEED = ROOT / "outputs/stage3_h13_d39_causal_task_space_recovery/shadow_candidates/stable_velocity_residual_update.csv"
D41_RUN = ROOT / "outputs/stage3_h13_d41_offline_robot_certification/run_20260827T152326Z"
D41_PRE = D41_RUN / "strict_replay/moveit_waypoint_joint_trajectory.csv"
D41_POST = D41_RUN / "strict_replay/moveit_smoothed_joint_trajectory.csv"
D41_FK = D41_RUN / "strict_replay/moveit_fk_tcp_trace.csv"
P2A_RESULT = ROOT / "outputs/p2a_axiswise_robustness_margin.json"
P2A_MODULE = ROOT / "src/p2a_axiswise_robustness.py"
POSE_COUNT = 181
JOINTS = tuple(f"j{i}" for i in range(1, 7))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wsl_path(path: Path) -> str:
    value = str(path.resolve()).replace("\\", "/")
    drive, tail = value.split(":", 1)
    return f"/mnt/{drive.lower()}/{tail.lstrip('/')}"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def read_q(path: Path) -> np.ndarray:
    rows = read_csv(path)
    return np.asarray([[float(row[f"j{i}_q"]) for i in range(1, 7)] for row in rows], dtype=np.float64)


def read_metric_report(path: Path, report_name: str) -> dict[str, Any]:
    rows = read_csv(path)
    if not rows:
        return {"status": "NOT_AVAILABLE", "rows": []}
    if {"metric", "value"}.issubset(rows[0]):
        return {
            "status": "AVAILABLE",
            "metrics": {row["metric"]: row["value"] for row in rows},
        }
    if report_name == "dynamics" and "joint" in rows[0]:
        return {
            "status": "AVAILABLE",
            "joints": {row["joint"]: {key: value for key, value in row.items() if key != "joint"} for row in rows},
        }
    return {"status": "UNRESOLVED_SCHEMA", "fieldnames": list(rows[0]), "rows": rows}


def limits_from_urdf(path: Path) -> tuple[np.ndarray, np.ndarray]:
    bounds: dict[str, tuple[float, float]] = {}
    for joint in ET.parse(path).getroot().findall("joint"):
        name = joint.get("name", "")
        limit = joint.find("limit")
        if name in JOINTS and limit is not None:
            bounds[name] = (float(limit.attrib["lower"]), float(limit.attrib["upper"]))
    if set(bounds) != set(JOINTS):
        raise RuntimeError(f"FR5_URDF_POSITION_LIMITS_MISSING:{sorted(set(JOINTS) - set(bounds))}")
    return (
        np.asarray([bounds[name][0] for name in JOINTS], dtype=np.float64),
        np.asarray([bounds[name][1] for name in JOINTS], dtype=np.float64),
    )


def check_clean_execution_tree() -> tuple[str, str]:
    def git(*args: str) -> str:
        run = subprocess.run(["git", *args], cwd=ROOT, text=True, encoding="utf-8", capture_output=True)
        if run.returncode:
            raise RuntimeError(f"git_{args[0]}_failed:{run.stderr.strip()}")
        return run.stdout.strip()

    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    parent = git("rev-parse", "HEAD^")
    dirty = git("status", "--porcelain")
    if branch != BRANCH or parent != SOURCE_BASE or dirty:
        raise RuntimeError(f"execution_tree_identity_mismatch:branch={branch}:head={head}:parent={parent}:dirty={bool(dirty)}")
    return branch, head


def run_wsl(script: str, log_path: Path, distro: str, timeout_s: int = 7200) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    run = subprocess.run(
        ["wsl.exe", "-d", distro, "--", "bash", "-lc", script],
        cwd=ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=timeout_s,
    )
    log_path.write_text((run.stdout or "") + "\n--- STDERR ---\n" + (run.stderr or ""), encoding="utf-8", errors="replace")
    return int(run.returncode)


def source_ros(underlay: Path, overlay: Path) -> str:
    return "\n".join((
        "source /opt/ros/jazzy/setup.bash",
        f"source {shlex.quote(wsl_path(underlay / 'setup.bash'))}",
        f"source {shlex.quote(wsl_path(overlay / 'setup.bash'))}",
    ))


def runtime_provenance(underlay: Path, overlay: Path, distro: str, out_root: Path) -> str:
    command = "\n".join((
        source_ros(underlay, overlay),
        "printf 'ROS_DISTRO=%s\\n' \"$ROS_DISTRO\"",
        "python3 --version",
        "dpkg-query -W ros-jazzy-moveit-py ros-jazzy-moveit-core ros-jazzy-moveit-plugins ros-jazzy-ruckig 2>/dev/null || true",
        "ros2 pkg prefix moveit_py",
    ))
    log_path = out_root / "runtime_provenance.txt"
    code = run_wsl(command, log_path, distro)
    if code:
        raise RuntimeError("MoveIt_Ruckig_runtime_provenance_query_failed")
    return log_path.read_text(encoding="utf-8", errors="replace").strip()


def regenerate_d41_urdf(
    underlay: Path, overlay: Path, fairino_source: Path, expected: Path,
    out_root: Path, distro: str,
) -> dict[str, Any]:
    source_urdf = fairino_source / "fairino_description/urdf/fairino5_v6.urdf"
    installed_urdf = underlay / "fairino_description/share/fairino_description/urdf/fairino5_v6.urdf"
    source_control = fairino_source / "fairino5_v6_moveit2_config/config/fairino5_v6_robot.ros2_control.xacro"
    installed_control = underlay / "fairino5_v6_moveit2_config/share/fairino5_v6_moveit2_config/config/fairino5_v6_robot.ros2_control.xacro"
    for source in (source_urdf, source_control):
        if not source.is_file():
            raise RuntimeError(f"FAIRINO_source_asset_missing:{source.name}")
    raw = out_root / "expanded_fairino_xacro_raw.urdf"
    reproduced = out_root / "derived_robot_model.urdf"
    xacro = ROOT / "ros2_moveit_bridge/config/fairino5_v6_spray_tcp.urdf.xacro"
    inline = (
        "from pathlib import Path; "
        f"p=Path({str(wsl_path(raw))!r}); t=p.read_text(); n=t.count('<origins'); "
        "p_out=Path(" + repr(wsl_path(reproduced)) + "); "
        "p_out.write_text(t.replace('<origins','<origin'),newline='\\n'); "
        "print('malformed_origins_replaced='+str(n))"
    )
    command = "\n".join((
        source_ros(underlay, overlay),
        f"xacro {shlex.quote(wsl_path(xacro))} tool_tcp_xyz:=\"0.000 0.000 0.150\" tool_tcp_rpy:=\"0 0 0\" > {shlex.quote(wsl_path(raw))}",
        f"python3 -c {shlex.quote(inline)}",
        "sha256sum " + " ".join(shlex.quote(wsl_path(path)) for path in (source_urdf, installed_urdf, source_control, installed_control)),
        f"cmp -s {shlex.quote(wsl_path(reproduced))} {shlex.quote(wsl_path(expected))}",
    ))
    log_path = out_root / "model_regeneration.log"
    code = run_wsl(command, log_path, distro)
    if code:
        raise RuntimeError(f"D41_frozen_model_xacro_regeneration_mismatch:{code}")
    hashes: dict[str, str] = {}
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) == 2 and len(parts[0]) == 64:
            hashes[parts[1]] = parts[0]
    installed_urdf_hash = hashes.get(wsl_path(installed_urdf))
    installed_control_hash = hashes.get(wsl_path(installed_control))
    if installed_urdf_hash != sha256(source_urdf) or installed_control_hash != sha256(source_control):
        raise RuntimeError("FAIRINO_runtime_underlay_source_mismatch")
    return {
        "status": "BYTE_MATCH", "xacro_source": str(xacro.relative_to(ROOT)),
        "model_sha256": sha256(reproduced), "expected_model_sha256": sha256(expected),
        "source_urdf_sha256": sha256(source_urdf), "installed_urdf_sha256": installed_urdf_hash,
        "source_moveit_control_xacro_sha256": sha256(source_control), "installed_moveit_control_xacro_sha256": installed_control_hash,
        "malformed_tag_rule": "single source-preserving <origins to <origin replacement per D41 H4.5",
        "regenerated_model": str(reproduced), "reference_model": str(expected),
    }


def strict_environment(
    name: str, solver: str, buffer_rad: float, gain: float | None,
    output: Path, repo: Path, overlay: Path, urdf: Path,
    lower: np.ndarray, upper: np.ndarray,
) -> dict[str, str]:
    strict = output / "strict_replay"
    dls_diag = strict / "D41_dls_diagnostics.csv"
    env = {
        "REPO_ROOT": wsl_path(repo), "ROS_WS": wsl_path(overlay), "FINAL_OUT_DIR": wsl_path(strict),
        "TCP_POSES_CSV": wsl_path(POSES), "SAMPLES_PER_LOOP": str(POSE_COUNT),
        "OPEN_PATH": "true", "TCP_POINTS_TO_WALL": "true",
        "TOOL_TCP_XYZ": "0.000 0.000 0.150", "TOOL_TCP_RPY": "0 0 0",
        "TOOL_TCP_SOURCE": "assumed_150mm_placeholder", "TOOL_TCP_MEASURED_BY": "simulation",
        "TOOL_TCP_MEASURED_DATE": "not_applicable_offline", "TOOL_TCP_CALIBRATION_METHOD": "virtual_design_parameter",
        "STAND_OFF": "0.260", "WAYPOINT_STRIDE": "1", "PLANNING_MODE": "normal_dls_waypoints",
        "SEED_JOINT_CSV": wsl_path(D39_SEED), "VALIDATION_STRIDE": "1",
        "VELOCITY_SCALING": "0.15", "ACCELERATION_SCALING": "0.15",
        "TIME_PARAMETERIZATION": "tcp_arclength", "TARGET_TCP_SPEED": "0.003", "ZERO_BOUNDARY_STATE": "true",
        "MAX_PATH_DEVIATION": "0.006", "MAX_NORMAL_ERROR_DEG": "10.0",
        "MAX_STANDOFF_FRACTION": "0.05", "MAX_SPEED_FLUCTUATION": "0.05",
        "VALIDATE_COLLISION": "true", "COLLISION_CHECK_STRIDE": "1", "COLLISION_SEGMENT_STRIDE": "1",
        "COLLISION_INTERPOLATION_STEP_DEG": "0.5", "INCLUDE_TUNNEL_FLOOR_COLLISION": "true",
        "TUNNEL_FLOOR_Z": "-0.20", "TUNNEL_WALL_THICKNESS": "0.04", "TUNNEL_Y_THICKNESS": "1.10",
        "INCLUDE_BOTTOM_CLOSURE_COLLISION": "false", "FAST_EXIT_AFTER_REPORTS": "true",
        "WRITE_FINAL_VISUALS": "false", "WRITE_FINAL_ANIMATION": "false", "SEGMENTED_EXECUTION": "true",
        "D41_DLS_LIMIT_AWARE": "true",
        "D41_DLS_LOWER_LIMITS": " ".join(format(float(x), ".17g") for x in lower),
        "D41_DLS_UPPER_LIMITS": " ".join(format(float(x), ".17g") for x in upper),
        "D41_DLS_LIMIT_BUFFER_RAD": format(buffer_rad, ".17g"), "D41_DLS_CASE_ID": name,
        "D41_DLS_REPO_ROOT": wsl_path(repo), "D41_DLS_DIAGNOSTICS_CSV": wsl_path(dls_diag),
        "P2B1_DLS_VARIANT": solver, "P2B1_SOLVER_PRESSURE_CSV": wsl_path(output / "solver_pressure.csv"),
        "P2B1_FROZEN_D41_Q_CSV": wsl_path(D41_POST),
        "P2B1_FD_CONSISTENCY_JSON": wsl_path(output / "process_jacobian_fd.json"),
        "QUALITY_REPORT": wsl_path(strict / "moveit_quality_report.csv"),
        "DYNAMICS_REPORT": wsl_path(strict / "moveit_joint_dynamics_report.csv"),
        "COLLISION_REPORT": wsl_path(strict / "moveit_collision_report.csv"),
        "FK_TRACE": wsl_path(strict / "moveit_fk_tcp_trace.csv"),
        "TRAJECTORY_CSV": wsl_path(strict / "moveit_smoothed_joint_trajectory.csv"),
        "SEGMENTED_TRAJECTORY_CSV": wsl_path(strict / "moveit_executed_segmented_joint_trajectory.csv"),
        "WAYPOINT_TRAJECTORY_CSV": wsl_path(strict / "moveit_waypoint_joint_trajectory.csv"),
        "STRICT_JSON": wsl_path(strict / "audit_goal_requirements_strict.json"),
        "PRODUCTION_READINESS_JSON": wsl_path(strict / "production_readiness_check.json"),
        "RUNTIME_LOG": wsl_path(strict / "moveit_runtime.log"),
    }
    if gain is not None:
        env["P2B1_SECONDARY_GAIN"] = format(gain, ".17g")
    return env


def run_strict_variant(
    name: str, solver: str, buffer_rad: float, gain: float | None,
    root_out: Path, overlay: Path, underlay: Path, urdf: Path,
    lower: np.ndarray, upper: np.ndarray, distro: str,
) -> dict[str, Any]:
    output = root_out / "variants" / name
    output.mkdir(parents=True, exist_ok=False)
    env = strict_environment(name, solver, buffer_rad, gain, output, ROOT, overlay, urdf, lower, upper)
    assignments = [f"export {key}={shlex.quote(value)}" for key, value in env.items()]
    script = "\n".join((source_ros(underlay, overlay), *assignments, f"bash {shlex.quote(wsl_path(ROOT / 'scripts/run_moveit_strict_validation.sh'))}"))
    code = run_wsl(script, output / "strict_execution.log", distro)
    strict = output / "strict_replay"
    status_path = strict / "audit_goal_requirements_strict.json"
    summary_path = strict / "final_acceptance_summary.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    result: dict[str, Any] = {
        "variant_id": name, "solver": solver, "buffer_rad": buffer_rad, "secondary_gain": gain,
        "strict_runner_exit_code": code,
        "strict_audit_status": str(status.get("overall_status", "NOT_RUN")).upper(),
        "post_ruckig_validation_status": str(summary.get("overall_status", status.get("overall_status", "NOT_RUN"))).upper(),
        "output_directory": str(output),
        "pre_ruckig_path": str(strict / "moveit_waypoint_joint_trajectory.csv") if (strict / "moveit_waypoint_joint_trajectory.csv").is_file() else None,
        "post_ruckig_path": str(strict / "moveit_smoothed_joint_trajectory.csv") if (strict / "moveit_smoothed_joint_trajectory.csv").is_file() else None,
        "solver_pressure_path": str(output / "solver_pressure.csv") if (output / "solver_pressure.csv").is_file() else None,
        "fd_consistency_path": str(output / "process_jacobian_fd.json") if (output / "process_jacobian_fd.json").is_file() else None,
    }
    if summary:
        metrics = summary.get("metrics", {})
        result["strict_metrics"] = {
            "max_fk_path_error_mm": _number(metrics.get("fk_path_deviation_max_mm")),
            "max_normal_error_deg": _number(metrics.get("fk_normal_error_max_deg")),
            "max_joint_step_deg": _number(metrics.get("max_joint_step_deg")),
            "collision_count": _number(metrics.get("collision_count")),
            "collision_checked_state_count": _number(metrics.get("collision_checked_state_count")),
            "ruckig_used": str(metrics.get("moveit_ruckig_smoothing_used", "")).lower() == "true",
            "time_parameterization": metrics.get("moveit_time_parameterization"),
            "audit_status": metrics.get("strict_audit_status"),
        }
    for report_name, report_file in (
        ("quality", "moveit_quality_report.csv"), ("dynamics", "moveit_joint_dynamics_report.csv"),
        ("collision", "moveit_collision_report.csv"),
    ):
        path = strict / report_file
        if path.is_file():
            result[f"{report_name}_metrics"] = read_metric_report(path, report_name)
    pre_path, post_path = strict / "moveit_waypoint_joint_trajectory.csv", strict / "moveit_smoothed_joint_trajectory.csv"
    if pre_path.is_file() and post_path.is_file():
        pre_q = read_q(pre_path)
        post_rows = read_csv(post_path)
        post_q = np.asarray([[float(row[f"j{i}_q"]) for i in range(1, 7)] for row in post_rows], dtype=np.float64)
        if pre_q.shape == (POSE_COUNT, 6) and post_q.shape == (POSE_COUNT, 6) and np.isfinite(np.r_[pre_q.ravel(), post_q.ravel()]).all():
            joint_margin = np.minimum(pre_q - lower[None, :], upper[None, :] - pre_q)
            j6_slack = upper[5] - pre_q[:, 5]
            result["trajectory_metrics"] = {
                "waypoint_count": int(len(pre_q)),
                "pre_ruckig_min_joint_margin_rad": float(np.min(joint_margin)),
                "pre_ruckig_min_joint_margin_waypoint": int(np.unravel_index(np.argmin(joint_margin), joint_margin.shape)[0]),
                "pre_ruckig_min_j6_upper_slack_rad": float(np.min(j6_slack)),
                "pre_ruckig_min_j6_upper_slack_waypoint": int(np.argmin(j6_slack)),
                "post_ruckig_min_joint_margin_rad": float(np.min(np.minimum(post_q - lower[None, :], upper[None, :] - post_q))),
                "post_ruckig_min_j6_upper_slack_rad": float(np.min(upper[5] - post_q[:, 5])),
            }
    pressure_path = output / "solver_pressure.csv"
    if pressure_path.is_file():
        rows = [row for row in read_csv(pressure_path) if row.get("waypoint") and 80 <= int(row["waypoint"]) <= 95]
        if rows:
            clipped = []
            for row in rows:
                try:
                    projected = json.loads(row.get("projected_joints") or "[]")
                except json.JSONDecodeError:
                    projected = []
                if 5 in projected:
                    clipped.append(row)
            def max_field(key: str) -> tuple[float | None, int | None]:
                values = [(_number(row.get(key)), int(row["waypoint"])) for row in rows]
                values = [(value, index) for value, index in values if value is not None]
                return max(values, default=(None, None), key=lambda item: float(item[0]) if item[0] is not None else -math.inf)
            max_hard_excess, hard_wp = max_field("max_j6_upper_bound_excess")
            max_buffer_excess, buffer_wp = max_field("max_j6_buffer_boundary_excess")
            secondary = [float(row["secondary_objective_contribution_norm"]) for row in rows if _number(row.get("secondary_objective_contribution_norm")) is not None]
            primary = [float(row["primary_task_contribution_norm"]) for row in rows if _number(row.get("primary_task_contribution_norm")) is not None]
            wp86_87 = [row for row in rows if int(row["waypoint"]) in {86, 87}]
            secondary_accepted = [
                row for row in rows
                if _number(row.get("secondary_objective_contribution_norm")) is not None
                and float(row["secondary_objective_contribution_norm"]) > 0.0
                and row.get("line_search_accepted", "").strip().lower() in {"true", "1"}
            ]
            result["solver_pressure_wp80_95"] = {
                "record_count": len(rows), "joint6_clip_activations": len(clipped),
                "wp86_87_record_count": len(wp86_87),
                "wp86_87_joint6_clip_activations": sum(row in clipped for row in wp86_87),
                "max_preprojection_j6_hard_upper_excess_rad": max_hard_excess,
                "waypoint_of_max_hard_upper_excess": hard_wp,
                "max_preprojection_j6_buffer_boundary_excess_rad": max_buffer_excess,
                "waypoint_of_max_buffer_boundary_excess": buffer_wp,
                "max_secondary_objective_contribution_norm": max(secondary, default=None),
                "max_primary_task_contribution_norm": max(primary, default=None),
                "accepted_secondary_step_count": len(secondary_accepted),
            }
    return result


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def compare_a0(result: dict[str, Any]) -> dict[str, Any]:
    pre_path, post_path = D41_PRE, D41_POST
    candidate_pre = Path(str(result.get("pre_ruckig_path") or ""))
    candidate_post = Path(str(result.get("post_ruckig_path") or ""))
    if not all(path.is_file() for path in (pre_path, post_path, candidate_pre, candidate_post)):
        return {"status": "FAIL", "reason": "baseline_or_A0_trajectory_missing"}
    old_pre, new_pre = read_q(pre_path), read_q(candidate_pre)
    old_post = np.asarray([[float(row[f"j{i}_q"]) for i in range(1, 7)] for row in read_csv(post_path)], dtype=np.float64)
    new_post_rows = read_csv(candidate_post)
    new_post = np.asarray([[float(row[f"j{i}_q"]) for i in range(1, 7)] for row in new_post_rows], dtype=np.float64)
    old_t = np.asarray([float(row["t"]) for row in read_csv(post_path)])
    new_t = np.asarray([float(row["t"]) for row in new_post_rows])
    pre_diff = float(np.max(np.abs(old_pre - new_pre))) if old_pre.shape == new_pre.shape else math.inf
    post_diff = float(np.max(np.abs(old_post - new_post))) if old_post.shape == new_post.shape else math.inf
    time_diff = float(np.max(np.abs(old_t - new_t))) if old_t.shape == new_t.shape else math.inf
    historical_metrics = json.loads((D41_RUN / "strict_replay/final_acceptance_summary.json").read_text(encoding="utf-8")).get("metrics", {})
    candidate_metrics = result.get("strict_metrics", {})
    path_candidate = _number(candidate_metrics.get("max_fk_path_error_mm"))
    path_reference = _number(historical_metrics.get("fk_path_deviation_max_mm"))
    normal_candidate = _number(candidate_metrics.get("max_normal_error_deg"))
    normal_reference = _number(historical_metrics.get("fk_normal_error_max_deg"))
    path_diff = abs(path_candidate - path_reference) if path_candidate is not None and path_reference is not None else math.inf
    normal_diff = abs(normal_candidate - normal_reference) if normal_candidate is not None and normal_reference is not None else math.inf
    passed = (
        pre_diff <= 1e-6 and post_diff <= 1e-6 and time_diff <= 1e-6
        and result.get("strict_audit_status") == "PASS"
        and candidate_metrics.get("ruckig_used") is True
        and path_diff <= 1e-3 and normal_diff <= 1e-3
    )
    return {
        "status": "PASS" if passed else "FAIL", "pre_ruckig_max_abs_joint_delta_rad": pre_diff,
        "post_ruckig_max_abs_joint_delta_rad": post_diff, "post_ruckig_max_abs_time_delta_s": time_diff,
        "max_fk_path_error_abs_difference_mm": path_diff,
        "max_normal_error_abs_difference_deg": normal_diff,
        "tolerances": {"joint_rad": 1e-6, "time_s": 1e-6},
    }


def frozen_p2a_j6_margin() -> float | None:
    result = json.loads(P2A_RESULT.read_text(encoding="utf-8"))
    axis = next((row for row in result.get("axis_results", []) if row.get("specification", {}).get("axis_id") == "joint:j6:positive"), None)
    return _number(((axis or {}).get("estimated_raw_axis_margin") or {}).get("value"))


def _run_native_with_fresh_build(
    run_dir: Path, cases: list[dict[str, Any]], native_name: str, d46: Any,
    native_binary: Path, overlay: Path, underlay: Path, urdf: Path, distro: str,
) -> Path:
    native = run_dir / native_name
    native.mkdir(parents=True, exist_ok=False)
    manifest = native / "cases.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("case_id", "trajectory_csv", "family"), lineterminator="\n")
        writer.writeheader()
        for row in cases:
            writer.writerow({
                "case_id": row["case_id"], "trajectory_csv": d46.wsl_path(Path(str(row["trajectory_path"]))),
                "family": row["family"],
            })
    out_cmd = f"{shlex.quote(d46.wsl_path(native_binary))} --poses {shlex.quote(d46.wsl_path(POSES))} --cases {shlex.quote(d46.wsl_path(manifest))} --urdf {shlex.quote(d46.wsl_path(urdf))} --srdf {shlex.quote(d46.wsl_path(d46.SRDF))} --output {shlex.quote(d46.wsl_path(native))}"
    script = "\n".join((source_ros(underlay, overlay), "export D41_SKIP_CONTROLS=1", out_cmd))
    code = run_wsl(script, run_dir / "native_execution.log", distro)
    if code:
        raise RuntimeError(f"fresh_D41_native_failure:{code}:{run_dir}")
    return native


def _run_fresh_fk(
    batch_root: Path, native_root: Path, d46: Any, fk_binary: Path,
    overlay: Path, underlay: Path, urdf: Path, distro: str,
) -> Path:
    trace = batch_root / "fk" / "STAGE4A_FK_TRACE.csv"
    trace.parent.mkdir(parents=True, exist_ok=True)
    command = f"{shlex.quote(d46.wsl_path(fk_binary))} --cases {shlex.quote(d46.wsl_path(native_root / 'cases.csv'))} --urdf {shlex.quote(d46.wsl_path(urdf))} --srdf {shlex.quote(d46.wsl_path(d46.SRDF))} --output {shlex.quote(d46.wsl_path(trace))}"
    script = source_ros(underlay, overlay) + "\n" + command
    code = run_wsl(script, batch_root / "fk_execution.log", distro)
    if code or not trace.is_file() or trace.stat().st_size == 0:
        raise RuntimeError(f"fresh_MoveIt_FK_failure:{code}:{batch_root}")
    return trace


def remeasure_j6_margin(
    variant: dict[str, Any], scratch: Path, native_binary: Path, fk_binary: Path,
    overlay: Path, underlay: Path, urdf: Path, distro: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import tools.stage4a_system_benchmark as d46
    from src.process_aware_stress import ProcessTrajectory
    import src.p2a_axiswise_robustness as p2a

    d46.URDF = urdf
    lower, upper, dynamic_limits = d46.load_limits()
    times, q, _velocity, _acceleration, _jerk = d46.load_post_ruckig(Path(str(variant["post_ruckig_path"])))
    fk_root = scratch / "p2a" / str(variant["variant_id"])
    fk_root.mkdir(parents=True, exist_ok=False)
    fk_dir = fk_root / "nominal_fk"
    fk_dir.mkdir()
    p2a._run_existing_fk = lambda batch_root, native_root, module: _run_fresh_fk(  # type: ignore[assignment]
        batch_root, native_root, module, fk_binary, overlay, underlay, urdf, distro,
    )
    d46.run_native = lambda run_dir, cases, native_name="native": _run_native_with_fresh_build(  # type: ignore[assignment]
        run_dir, cases, native_name, d46, native_binary, overlay, underlay, urdf, distro,
    )
    _trace, q_by_case = p2a._run_fk_only_cases(fk_dir, [(f"{variant['variant_id']}_nominal", q)], d46)
    fk_cases = p2a._read_fk_trace(fk_dir / "fk" / "STAGE4A_FK_TRACE.csv")
    case_id = f"{variant['variant_id']}_nominal"
    if case_id not in fk_cases or case_id not in q_by_case:
        raise RuntimeError("P2A_fresh_nominal_FK_missing")
    fk = fk_cases[case_id]
    target_poses = np.column_stack((fk["position"], fk["quaternion"]))
    d41_reference = p2a._read_d41_process_reference(D41_FK)
    nominal = ProcessTrajectory(
        tcp_poses=target_poses, joint_states=q, timestamps_s=times,
        surface_normals=d41_reference["surface_normals"], joint_lower_rad=lower, joint_upper_rad=upper,
        metadata={"robot": "FAIRINO_FR5", "scope": "181-point ON-state open-arch only", "source": "fresh MoveIt2 FK of candidate post-Ruckig q"},
    )
    thresholds = d46.read_json(d46.OUT / "STAGE4_SYSTEM_BENCHMARK_V1.json")["diagnostic_thresholds"]
    import tools.audit_stage17_reproducibility as d17
    spec = next(row for row in p2a._build_axis_specs(q, lower, upper) if row.axis_id == "joint:j6:positive")
    evaluator = p2a._D41BatchEvaluator(
        nominal, times, lower, upper, dynamic_limits,
        float(thresholds["tcp_trajectory_error_m"]["value"]),
        float(thresholds["terminal_position_error_m"]["value"]), math.radians(float(d17.FORMAL_NORMAL_DEG)),
        float(thresholds["joint_step_rad"]["value"]), fk_root / "scan", d46,
    )
    (fk_root / "scan").mkdir()
    result = p2a.scan_axes_batched(nominal, [spec], evaluator.evaluate_batch)[0]
    observations = []
    for phase_name in ("coarse_observations", "refinement_observations"):
        for record in result.get(phase_name, []):
            observations.append({
                "variant_id": variant["variant_id"], "phase": record.get("phase"),
                "magnitude_rad": record.get("magnitude"), "status": record.get("status"),
                "failure_modes": ";".join(record.get("failure_modes", [])),
                "critical_waypoint": record.get("critical_waypoint"),
                "critical_segment": record.get("critical_segment"),
                "joint_limit_margin_min_rad": (record.get("evidence") or {}).get("joint_limit_margin_min_rad"),
            })
    summary = {
        "axis_id": result["specification"]["axis_id"], "axis_status": result["axis_status"],
        "search_completeness": result["search_completeness"],
        "estimated_raw_axis_margin": result["estimated_raw_axis_margin"],
        "refined_boundary_estimate": result["refined_boundary_estimate"],
        "last_passing_perturbation": result["last_passing_perturbation"],
        "first_failing_perturbation": result["first_failing_perturbation"],
        "failure_intervals": result.get("failure_intervals", []),
        "monotonicity_observation": result["monotonicity_observation"],
        "scan_case_count": len(observations), "native_batches": evaluator._batch_number,
        "method_source": "src/p2a_axiswise_robustness.py:scan_axes_batched + _D41BatchEvaluator, fresh source-built MoveIt2 D41/FK",
        "collision_method": "adaptive_discrete_interpolation", "strict_self_ccd": "NOT_AVAILABLE",
    }
    return summary, observations


def variants() -> list[tuple[str, str, float, float | None]]:
    rows = [("A0", "A0", 1e-4, None)]
    rows.extend((f"A1_buffer_{buffer:g}", "A1", buffer, None) for buffer in (1e-4, 1e-3, 5e-3, 1e-2))
    rows.append(("B0_process_5d", "B0", 1e-4, None))
    rows.extend((f"B1_gain_{gain:g}", "B1", 1e-4, gain) for gain in (2.5e-4, 5e-4, 1e-3))
    return rows


def _p2a_value(row: Mapping[str, Any]) -> float | None:
    margin = (row.get("p2a_j6_positive") or {}).get("estimated_raw_axis_margin") or {}
    return _number(margin.get("value"))


def classify_cause(measured: list[dict[str, Any]], baseline_reproduction: Mapping[str, Any]) -> str:
    if baseline_reproduction.get("status") != "PASS":
        return "INCONCLUSIVE"
    by_id = {row["variant_id"]: row for row in measured}
    a0 = by_id.get("A0")
    baseline_margin = _p2a_value(a0 or {})
    if baseline_margin is None:
        return "INCONCLUSIVE"
    # The effect floor is 10% of the reproduced baseline margin or four P2-A
    # refinement quanta, whichever is larger; this separates mechanism changes
    # from the published 1e-6 rad boundary resolution without demanding a target gain.
    effect_floor = max(0.10 * baseline_margin, 4e-6)
    effects: set[str] = set()
    a1_rows = sorted(
        (row for row in measured if row["variant_id"].startswith("A1_buffer_")),
        key=lambda row: float(row["buffer_rad"]),
    )
    a1_values = [_p2a_value(row) for row in a1_rows]
    b0_row = by_id.get("B0_process_5d", {})
    b0_margin = _p2a_value(b0_row)
    b1_rows = [row for row in measured if row["variant_id"].startswith("B1_gain_")]
    b1_values = [_p2a_value(row) for row in b1_rows]
    if len(a1_values) != 4 or any(value is None for value in a1_values) or b0_margin is None or len(b1_values) != 3 or any(value is None for value in b1_values):
        return "INCONCLUSIVE"
    if any(float(a1_values[i + 1]) + 1e-6 < float(a1_values[i]) for i in range(len(a1_values) - 1)):
        return "INCONCLUSIVE"
    if any(float(value) - baseline_margin >= effect_floor for value in a1_values):
        effects.add("buffer")
    if b0_margin - baseline_margin >= effect_floor:
        effects.add("task")
    if any(float(value) - b0_margin >= effect_floor for value in b1_values):
        effects.add("redundancy")
    if len(effects) > 1:
        return "MULTIFACTOR"
    if effects == {"buffer"}:
        return "BUFFER_POLICY_DOMINATED"
    if effects == {"task"}:
        return "PROCESS_TASK_FORMULATION_DOMINATED"
    if effects == {"redundancy"}:
        return "REDUNDANCY_ALLOCATION_DOMINATED"
    # A geometry-limited result needs the same boundary to persist after a
    # verified B1 task solve and a recorded accepted null-space contribution.
    b1_passes = [row for row in b1_rows if row.get("strict_audit_status") == "PASS"]
    nominal_slack = (a0.get("trajectory_metrics") or {}).get("post_ruckig_min_j6_upper_slack_rad")
    b1_nominal_slack = [
        (row.get("trajectory_metrics") or {}).get("post_ruckig_min_j6_upper_slack_rad") for row in b1_rows
    ]
    other_failure = any(
        any(mode in {"ENVIRONMENT_COLLISION", "SELF_COLLISION", "TCP_PATH_DEVIATION", "SPRAY_AXIS_NORMAL_ERROR", "TERMINAL_POSITION_ERROR"}
            for mode in ((row.get("p2a_j6_positive") or {}).get("first_failing_perturbation") or {}).get("failure_modes", []))
        for row in b1_rows
    )
    if b1_passes and _number(nominal_slack) is not None and all(_number(value) is not None and float(value) - float(nominal_slack) >= effect_floor for value in b1_nominal_slack) and all(float(value) <= baseline_margin + effect_floor for value in b1_values) and other_failure:
        if any(int((row.get("solver_pressure_wp80_95") or {}).get("accepted_secondary_step_count") or 0) > 0 for row in b1_passes):
            return "TASK_GEOMETRY_LIMITED"
    return "INCONCLUSIVE"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True, help="Byte-verified regenerated D41 derived robot model URDF")
    parser.add_argument("--underlay-install", type=Path, required=True, help="ROS workspace install providing FAIRINO description/config packages")
    parser.add_argument("--fairino-source-repository", type=Path, required=True, help="Versioned FAIRINO ROS description source used by the runtime underlay")
    parser.add_argument("--distro", default="Ubuntu-24.04-D")
    parser.add_argument("--native-build-install", type=Path, required=True, help="This execution's source-built ROS overlay")
    args = parser.parse_args()

    scratch, urdf, underlay, overlay, fairino_source = (path.resolve() for path in (args.scratch, args.urdf, args.underlay_install, args.native_build_install, args.fairino_source_repository))
    if not all(path.exists() for path in (scratch, urdf, underlay / "setup.bash", overlay / "setup.bash", fairino_source / ".git")):
        raise RuntimeError("P2B1_scratch_or_runtime_dependency_missing")
    branch, head = check_clean_execution_tree()
    for path in (POSES, SEEDS, D39_SEED, D41_PRE, D41_POST, D41_FK, P2A_RESULT, urdf,
                 ROOT / "ros2_moveit_bridge/config/fairino5_v6_spray_tcp.srdf",
                 ROOT / "ros2_moveit_bridge/config/fairino5_v6_spray_tcp.urdf.xacro",
                 ROOT / "ros2_moveit_bridge/config/joint_limits_with_jerk.yaml"):
        if not path.is_file():
            raise FileNotFoundError(path)
    lower, upper = limits_from_urdf(urdf)
    native_binary = overlay / "lib/stage3_h13_d41_native/stage3_h13_d41_native"
    fk_binary = overlay / "lib/stage4a_fk/stage4a_fk"
    if not native_binary.is_file() or not fk_binary.is_file():
        raise RuntimeError("fresh_source_build_D41_native_or_stage4a_fk_missing")
    out_root = scratch / "p2b1_formal_execution"
    out_root.mkdir(parents=True, exist_ok=False)
    execution = {
        "schema": "p2b1-execution-manifest-v1", "project": "FAIRINO_FR5", "stage": "P2-B1",
        "branch": branch, "source_base_commit": SOURCE_BASE, "execution_code_commit": head,
        "execution_tree_clean_at_start": True, "operating_system_runner": "Windows Python -> WSL2 Ubuntu-24.04-D",
        "ros_distro": "Jazzy", "moveit_runtime_package": "MoveItPy; source-built fr5_tunnel_moveit_bridge overlay",
        "underlay_install": str(underlay), "overlay_install": str(overlay), "derived_robot_model_urdf": str(urdf),
        "identity_sha256": {
            str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else path.name: sha256(path)
            for path in (POSES, SEEDS, D39_SEED, D41_PRE, D41_POST, D41_FK, P2A_RESULT, urdf,
                         ROOT / "ros2_moveit_bridge/config/fairino5_v6_spray_tcp.srdf",
                         ROOT / "ros2_moveit_bridge/config/fairino5_v6_spray_tcp.urdf.xacro",
                         ROOT / "ros2_moveit_bridge/config/joint_limits_with_jerk.yaml")
        },
        "native_source_build_commit": head,
        "native_binary_sha256": {str(native_binary.name): sha256(native_binary), str(fk_binary.name): sha256(fk_binary)},
        "collision_method": "adaptive_discrete_interpolation", "strict_self_ccd": "NOT_AVAILABLE",
        "hardware_validation": "NOT_RUN", "hardware_safety_certified": "NO",
        "variant_environment": {
            "tcp_input": str(POSES.relative_to(ROOT)), "seed_input": str(D39_SEED.relative_to(ROOT)),
            "points": POSE_COUNT, "joint_order": list(JOINTS), "task_scope": "ON-state open-arch only",
            "waypoint_stride": 1, "stand_off_m": 0.260, "tcp_offset_m": [0.0, 0.0, 0.15],
            "time_parameterization": "tcp_arclength", "moveit_ruckig": True,
            "collision_interpolation_deg": 0.5, "collision_stride": 1,
        },
    }
    write_json(out_root / "execution_manifest.json", execution)
    fairino_remote = subprocess.run(["git", "remote", "get-url", "origin"], cwd=fairino_source, text=True, encoding="utf-8", capture_output=True)
    fairino_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=fairino_source, text=True, encoding="utf-8", capture_output=True)
    fairino_status = subprocess.run(["git", "status", "--porcelain"], cwd=fairino_source, text=True, encoding="utf-8", capture_output=True)
    if fairino_remote.returncode or fairino_head.returncode or fairino_status.returncode or "FAIR-INNOVATION/frcobot_ros2" not in fairino_remote.stdout or fairino_status.stdout.strip():
        raise RuntimeError("FAIRINO_description_source_identity_invalid")
    execution["fairino_description_source"] = {
        "remote": fairino_remote.stdout.strip(), "commit": fairino_head.stdout.strip(),
        "working_tree_clean": not fairino_status.stdout.strip(),
    }
    execution["runtime_versions"] = runtime_provenance(underlay, overlay, args.distro, out_root)
    execution["derived_model_regeneration"] = regenerate_d41_urdf(underlay, overlay, fairino_source, urdf, out_root, args.distro)
    write_json(out_root / "execution_manifest.json", execution)

    measured: list[dict[str, Any]] = []
    margin_observations: list[dict[str, Any]] = []
    for name, solver, buffer_rad, gain in variants():
        variant = run_strict_variant(name, solver, buffer_rad, gain, out_root, overlay, underlay, urdf, lower, upper, args.distro)
        has_complete_trajectory = variant.get("post_ruckig_path") is not None
        strict_pass = variant.get("strict_audit_status") == "PASS"
        if has_complete_trajectory and strict_pass:
            try:
                margin, observations = remeasure_j6_margin(
                    variant, out_root, native_binary, fk_binary, overlay, underlay, urdf, args.distro,
                )
                variant["p2a_j6_positive"] = margin
                margin_observations.extend(observations)
            except Exception as exc:
                variant["p2a_j6_positive"] = {"status": "BLOCKED", "error": f"{type(exc).__name__}: {exc}"}
        else:
            variant["p2a_j6_positive"] = {"status": "NOT_RUN_POST_RUCKIG_STRICT_GATE_NOT_PASS"}
        if name == "A0":
            variant["baseline_reproduction"] = compare_a0(variant)
        measured.append(variant)
        write_json(out_root / "partial_result.json", {"execution": execution, "completed_variants": measured})

    result: dict[str, Any] = {
        "schema": "p2b1-solver-policy-causal-ablation-v1", "project": "FAIRINO_FR5", "stage": "P2-B1",
        "source_base_commit": SOURCE_BASE, "execution_code_commit": head,
        "execution_tree_clean_at_start": True, "execution_manifest": str(out_root / "execution_manifest.json"),
        "variants": measured, "causal_classification": "INCONCLUSIVE",
        "measurement_pipeline_status": "COMPLETE" if all(row.get("output_directory") for row in measured) else "INCOMPLETE",
        "frozen_robot_baseline_performance_status": "MEASURED_SEPARATELY_PER_VARIANT",
        "limitations": [
            "P2-A robustness uses the original joint:j6:positive axis definition on candidate-specific post-Ruckig timestamps.",
            "Collision outcomes use adaptive discrete interpolation; strict self collision CCD is unavailable.",
            "Configured dynamics bounds are offline project limits, not vendor-certified limits.",
            "No hardware execution or safety certification was run.",
        ],
    }
    a0 = next(row for row in measured if row["variant_id"] == "A0")
    result["baseline_reproduction"] = a0.get("baseline_reproduction", {"status": "FAIL"})
    a0_margin = (a0.get("p2a_j6_positive") or {}).get("estimated_raw_axis_margin", {})
    baseline_margin = _number(a0_margin.get("value"))
    frozen_margin = frozen_p2a_j6_margin()
    p2a_difference = abs(baseline_margin - frozen_margin) if baseline_margin is not None and frozen_margin is not None else None
    if p2a_difference is None or p2a_difference > 1e-6:
        result["baseline_reproduction"]["status"] = "FAIL"
    result["baseline_reproduction"]["frozen_P2A_J6_positive_margin_rad"] = frozen_margin
    result["baseline_reproduction"]["A0_P2A_J6_positive_margin_rad"] = baseline_margin
    result["baseline_reproduction"]["P2A_margin_abs_difference_rad"] = p2a_difference
    a0["baseline_reproduction"] = result["baseline_reproduction"]
    for row in measured:
        margin_record = (row.get("p2a_j6_positive") or {}).get("estimated_raw_axis_margin", {})
        margin = _number(margin_record.get("value"))
        row["j6_margin_improvement_ratio_vs_A0"] = margin / baseline_margin if margin is not None and baseline_margin and baseline_margin > 0 else None
    result["causal_classification"] = classify_cause(measured, result["baseline_reproduction"])
    complete_runs = all(row.get("post_ruckig_path") is not None for row in measured)
    result["P2B1_STATUS"] = "COMPLETE" if complete_runs and result["baseline_reproduction"].get("status") == "PASS" else "INCOMPLETE"
    result["P2B1_BASELINE_REPRODUCTION"] = result["baseline_reproduction"].get("status", "FAIL")
    result["A1_BUFFER_ABLATION"] = "PASS" if all(
        next(row for row in measured if row["variant_id"] == f"A1_buffer_{buffer:g}").get("strict_audit_status") == "PASS"
        and (next(row for row in measured if row["variant_id"] == f"A1_buffer_{buffer:g}").get("p2a_j6_positive") or {}).get("estimated_raw_axis_margin") is not None
        for buffer in (1e-4, 1e-3, 5e-3, 1e-2)
    ) else "FAIL" if complete_runs else "INCONCLUSIVE"
    b0 = next(row for row in measured if row["variant_id"] == "B0_process_5d")
    b0_fd = json.loads(Path(str(b0["fd_consistency_path"])).read_text(encoding="utf-8")) if b0.get("fd_consistency_path") and Path(str(b0["fd_consistency_path"])).is_file() else {}
    result["B0_PROCESS_5DOF"] = "PASS" if b0.get("strict_audit_status") == "PASS" and b0_fd.get("status") == "PASS" and _p2a_value(b0) is not None else "FAIL" if b0.get("post_ruckig_path") else "INCONCLUSIVE"
    b1_rows = [row for row in measured if row["variant_id"].startswith("B1_gain_")]
    result["B1_NULLSPACE_AVOIDANCE"] = "PASS" if all(
        row.get("strict_audit_status") == "PASS" and row.get("fd_consistency_path") and Path(str(row["fd_consistency_path"])).is_file() and json.loads(Path(str(row["fd_consistency_path"])).read_text(encoding="utf-8")).get("status") == "PASS" and _p2a_value(row) is not None
        for row in b1_rows
    ) else "FAIL" if all(row.get("post_ruckig_path") for row in b1_rows) else "INCONCLUSIVE"
    result["A0_MIN_J6_UPPER_SLACK_RAD"] = (a0.get("trajectory_metrics") or {}).get("post_ruckig_min_j6_upper_slack_rad")
    result["A0_P2A_J6_POSITIVE_MARGIN_RAD"] = baseline_margin
    result["B0_MIN_J6_UPPER_SLACK_RAD"] = (b0.get("trajectory_metrics") or {}).get("post_ruckig_min_j6_upper_slack_rad")
    result["B0_P2A_J6_POSITIVE_MARGIN_RAD"] = _p2a_value(b0)
    result["B1_GAIN_SENSITIVITY"] = [
        {"gain": row.get("secondary_gain"), "minimum_j6_upper_slack_rad": (row.get("trajectory_metrics") or {}).get("post_ruckig_min_j6_upper_slack_rad"), "p2a_j6_positive_margin_rad": _p2a_value(row), "improvement_ratio_vs_A0": row.get("j6_margin_improvement_ratio_vs_A0"), "validation_status": row.get("strict_audit_status")}
        for row in b1_rows
    ]
    best = max(
        (row for row in measured if _p2a_value(row) is not None),
        key=lambda row: float(_p2a_value(row)), default=None,
    )
    result["BEST_VARIANT"] = best["variant_id"] if best else None
    result["BEST_IMPROVEMENT_RATIO_VS_A0"] = best.get("j6_margin_improvement_ratio_vs_A0") if best else None
    result["POST_RUCKIG_VALIDATION"] = "PASS" if all(row.get("strict_audit_status") == "PASS" for row in measured) else "FAIL_OR_INCOMPLETE"
    result["COLLISION_METHOD"] = "adaptive_discrete_interpolation"
    result["STRICT_SELF_CCD"] = "NOT_AVAILABLE"
    result["HARDWARE_VALIDATION"] = "NOT_RUN"
    result["HARDWARE_SAFETY_CERTIFIED"] = "NO"
    result["MEASUREMENT_OUTPUT_DIRECTORY"] = str(out_root)
    observations_path = out_root / "p2b1_j6_margin_observations.csv"
    result["P2A_MARGIN_OBSERVATIONS_CSV"] = str(observations_path) if margin_observations else None
    if margin_observations:
        with observations_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(margin_observations[0]), lineterminator="\n")
            writer.writeheader(); writer.writerows(margin_observations)
    write_json(out_root / "p2b1_result.json", result)
    print(json.dumps({"P2B1_RESULT": str(out_root / "p2b1_result.json"), "variant_count": len(measured), "completed": sum(bool(row.get("post_ruckig_path")) for row in measured), "baseline_reproduction": result["baseline_reproduction"], "causal_classification": result["causal_classification"]}, sort_keys=True))
    return 0 if len(measured) == len(variants()) else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"P2B1_STATUS": "BLOCKED_OR_FAILED", "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
        raise
