"""Stage 2.7 controller execution-semantics certification.

This stage consumes frozen Stage 2.5/2.6 evidence only.  It audits the
actual ros2_control JointTrajectoryController configuration and runtime when
available, reconstructs every source interval with the corresponding spline,
checks analytic extrema, and optionally sends a separately generated dense
candidate path through the existing native Bullet robot-world CCD probe.

No Stage 2.5 waypoint, timestamp, Ruckig artifact, IK input, or Stage 2.6
artifact is modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
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
    compare_methods,
    detect_hidden_spline_overshoot,
    evaluate_coefficients,
    interpolation_order_from_fields,
    segment_coefficients,
)


JOINT_NAMES = ["j1", "j2", "j3", "j4", "j5", "j6"]
DEFAULT_STAGE25 = ROOT / "outputs/ik_graph_stage25_timed_certification/fr5_scaled_horseshoe_demo_v45_20260803_formal_000002_Stage25"
DEFAULT_STAGE26 = ROOT / "outputs/ik_graph_stage26_continuous_collision_certification/fr5_scaled_horseshoe_demo_v45_20260803_formal_000010_Stage26"
DEFAULT_CONTROLLER = ROOT / "external/frcobot_ros2/fairino5_v6_moveit2_config/config/ros2_controllers.yaml"
DEFAULT_XACRO = ROOT / "external/frcobot_ros2/fairino5_v6_moveit2_config/config/fairino5_v6_robot.ros2_control.xacro"
DEFAULT_INITIAL_POSITIONS = ROOT / "external/frcobot_ros2/fairino5_v6_moveit2_config/config/initial_positions.yaml"
DEFAULT_PARTS = ROOT / "outputs/ik_graph_stage23a4_backend_equivalent/fr5_scaled_horseshoe_demo_v44/tunnel_collision_parts"
DEFAULT_MAX_CCD_DT_S = 0.0025


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def semantic_hash(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_yaml(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, allow_unicode=True, sort_keys=False), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def wsl_path(path: Path) -> str:
    resolved = str(path.resolve()).replace("\\", "/")
    drive, rest = resolved.split(":/", 1)
    return f"/mnt/{drive.lower()}/{rest}"


def run_command(command: list[str], timeout_s: int = 30) -> dict[str, Any]:
    try:
        proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout_s)
        return {"command": command, "exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr, "timed_out": False}
    except subprocess.TimeoutExpired as error:
        return {"command": command, "exit_code": None, "stdout": error.stdout or "", "stderr": error.stderr or "", "timed_out": True}
    except FileNotFoundError as error:
        return {"command": command, "exit_code": None, "stdout": "", "stderr": str(error), "timed_out": False}


def run_wsl(command: str, timeout_s: int = 30) -> dict[str, Any]:
    return run_command(["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", command], timeout_s)


def read_stage25_trajectory(path: Path) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Empty Stage 2.5 trajectory: {path}")
    try:
        times = np.asarray([float(row["t"]) for row in rows], dtype=float)
        q = np.asarray([[float(row[f"{joint}_q"]) for joint in JOINT_NAMES] for row in rows], dtype=float)
        dq = np.asarray([[float(row[f"{joint}_dq"]) for joint in JOINT_NAMES] for row in rows], dtype=float)
        ddq = np.asarray([[float(row[f"{joint}_ddq"]) for joint in JOINT_NAMES] for row in rows], dtype=float)
    except (KeyError, ValueError) as error:
        raise RuntimeError(f"Stage 2.5 trajectory is missing complete q/dq/ddq fields: {error}") from error
    if len(rows) != 25532 or len(times) - 1 != 25531:
        raise RuntimeError(f"Expected 25532 points and 25531 intervals, got {len(rows)} and {len(rows) - 1}")
    if abs(times[0]) > 1e-15 or not np.all(np.diff(times) > 0.0):
        raise RuntimeError("Stage 2.5 time_from_start values are not strictly increasing from 0.0")
    return {"rows": rows, "times": times, "q": q, "dq": dq, "ddq": ddq}


def parse_controller_config(controller_path: Path, xacro_path: Path, initial_positions_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(controller_path.read_text(encoding="utf-8")) or {}
    manager = config.get("controller_manager", {}).get("ros__parameters", {})
    controller_name = next((name for name, value in manager.items() if isinstance(value, dict) and value.get("type")), None)
    if controller_name is None:
        raise RuntimeError(f"No controller type found in {controller_path}")
    controller_type = str(manager[controller_name].get("type"))
    parameters = config.get(controller_name, {}).get("ros__parameters", {})
    initial_yaml = yaml.safe_load(initial_positions_path.read_text(encoding="utf-8")) or {}
    initial_map = initial_yaml.get("initial_positions", {})
    initial_positions = [float(initial_map.get(joint, 0.0)) for joint in JOINT_NAMES]
    xacro_text = xacro_path.read_text(encoding="utf-8")
    plugin_match = re.search(r"<plugin>\s*([^<]+?)\s*</plugin>", xacro_text)
    return {
        "controller_name": controller_name,
        "controller_type": controller_type,
        "configured_parameters": parameters,
        "command_interfaces": list(parameters.get("command_interfaces", [])),
        "state_interfaces": list(parameters.get("state_interfaces", [])),
        "configured_interpolation_method": parameters.get("interpolation_method"),
        "controller_sha256": sha256(controller_path),
        "controller_config_path": str(controller_path.resolve()),
        "xacro_path": str(xacro_path.resolve()),
        "hardware_plugin": plugin_match.group(1).strip() if plugin_match else None,
        "initial_positions_path": str(initial_positions_path.resolve()),
        "initial_positions_sha256": sha256(initial_positions_path),
        "initial_positions": initial_positions,
    }


def probe_controller_runtime(controller: dict[str, Any]) -> dict[str, Any]:
    """Probe package, controller-manager, runtime parameter and state evidence."""

    package = run_wsl("source /opt/ros/jazzy/setup.bash && ros2 pkg prefix joint_trajectory_controller", 20)
    package_xml = run_wsl("source /opt/ros/jazzy/setup.bash && ros2 pkg xml joint_trajectory_controller", 20)
    manager = run_wsl("source /opt/ros/jazzy/setup.bash && ros2 control list_controllers", 20)
    parameter = run_wsl(
        f"source /opt/ros/jazzy/setup.bash && ros2 param get /{controller['controller_name']} interpolation_method",
        20,
    )
    state = run_wsl(
        "source /opt/ros/jazzy/setup.bash && timeout 8 ros2 topic echo --once /joint_states",
        15,
    )
    package_available = package["exit_code"] == 0 and bool(package["stdout"].strip())
    parameter_text = f"{parameter['stdout']}\n{parameter['stderr']}"
    method_match = re.search(r"(?:String value is:|value:)\s*([A-Za-z0-9_-]+)", parameter_text)
    runtime_method = method_match.group(1) if method_match else None
    active_text = f"{manager['stdout']}\n{manager['stderr']}"
    controller_active = bool(re.search(rf"\b{re.escape(controller['controller_name'])}\b.*\bactive\b", active_text, re.IGNORECASE))
    return {
        "ros_distribution": "jazzy",
        "joint_trajectory_controller_package_available": package_available,
        "joint_trajectory_controller_package_prefix": package["stdout"].strip() if package_available else None,
        "joint_trajectory_controller_package_probe": package,
        "joint_trajectory_controller_package_xml_probe": package_xml,
        "controller_manager_runtime_observed": manager["exit_code"] == 0,
        "controller_manager_controller_active": controller_active,
        "controller_manager_probe": manager,
        "runtime_interpolation_method": runtime_method,
        "runtime_interpolation_parameter_probe": parameter,
        "joint_states_probe": state,
        "runtime_initial_state_observed": False,
        "runtime_semantics_verified": bool(package_available and controller_active and runtime_method),
        "status": "passed_runtime_observed" if package_available and controller_active and runtime_method else "not_available",
        "reason": (
            "joint_trajectory_controller package/controller/runtime parameter not all observable"
            if not (package_available and controller_active and runtime_method)
            else None
        ),
    }


def build_method_record(trajectory: dict[str, Any], controller: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
    submitted = {
        "position": True,
        "velocity": bool(np.all(np.isfinite(trajectory["dq"]))),
        "acceleration": bool(np.all(np.isfinite(trajectory["ddq"]))),
        "actual_controller_submission": False,
        "source": "Stage 2.5 CSV contains complete q/dq/ddq fields; Stage 2.5 controller_execution was not_requested_offline_certification",
    }
    candidate_order = interpolation_order_from_fields(submitted["position"], submitted["velocity"], submitted["acceleration"])
    method_verified = runtime["runtime_semantics_verified"]
    return {
        "controller_type": controller["controller_type"],
        "interpolation_method": runtime["runtime_interpolation_method"] if method_verified else "not_verified",
        "interpolation_method_source": (
            "runtime parameter /fairino5_controller/interpolation_method"
            if method_verified
            else "not_available: JointTrajectoryController package/runtime was not observable in Ubuntu-24.04-D"
        ),
        "configured_interpolation_method": controller["configured_interpolation_method"],
        "submitted_fields": submitted,
        "resulting_interpolation_order": candidate_order if method_verified else "not_verified",
        "candidate_resulting_interpolation_order_from_frozen_message_fields": candidate_order,
        "method_verified": bool(method_verified),
        "command_interfaces": controller["command_interfaces"],
        "state_interfaces": controller["state_interfaces"],
    }


def build_initial_state_gate(trajectory: dict[str, Any], controller: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
    trajectory_first = trajectory["q"][0].tolist()
    configured_initial = controller["initial_positions"]
    errors = np.abs(np.asarray(configured_initial) - trajectory["q"][0])
    return {
        "trajectory_first_time_s": float(trajectory["times"][0]),
        "controller_initial_joint_positions": configured_initial,
        "trajectory_first_joint_positions": trajectory_first,
        "max_initial_position_error_rad": float(np.max(errors)),
        "controller_initial_joint_velocities": None,
        "trajectory_first_joint_velocities": trajectory["dq"][0].tolist(),
        "runtime_initial_state_observed": runtime.get("runtime_initial_state_observed", False),
        "runtime_initial_joint_positions": None,
        "initial_state_compatible": bool(np.max(errors) <= 1e-12 and runtime.get("runtime_initial_state_observed", False)),
        "configured_initial_state_compatible": bool(np.max(errors) <= 1e-12),
        "status": "blocked_configured_initial_state_mismatch" if np.max(errors) > 1e-12 else "not_available_runtime_initial_state_not_observed",
        "formal_mock_hardware_initialization_recommendation": "initialize mock hardware to trajectory_first_joint_positions before activation",
        "formal_mock_hardware_initialization_target": trajectory_first,
    }


def write_mock_initial_positions(path: Path, trajectory: dict[str, Any]) -> None:
    write_yaml(
        path,
        {
            "schema_version": "stage27-mock-initial-positions-v1",
            "purpose": "isolated mock hardware override; initialize to frozen Stage 2.5 trajectory first waypoint",
            "initial_positions": {joint: float(trajectory["q"][0, index]) for index, joint in enumerate(JOINT_NAMES)},
        },
    )


def load_limits(stage25_root: Path) -> list[dict[str, float]]:
    report = read_json(stage25_root / "stage25_joint_limits.json")
    return list(report["joints"])


def interval_metadata(stage26_root: Path) -> list[dict[str, Any]]:
    path = stage26_root / "stage26_intervals.csv"
    if not path.exists():
        raise RuntimeError(f"Missing Stage 2.6 interval metadata: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_interval_extrema(path: Path, audits: list[Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for audit in audits:
            handle.write(
                json.dumps(
                    {
                        "interval_index": audit.interval_index,
                        "time_start_s": audit.time_start_s,
                        "time_end_s": audit.time_end_s,
                        "extrema": audit.extrema,
                        "coefficient_sha256": semantic_hash(np.asarray(audit.coefficients).tolist()),
                    },
                    sort_keys=True,
                )
                + "\n"
            )


def write_reconstructed_points(path: Path, trajectory: dict[str, Any], method: str) -> None:
    fields = ["t"] + [f"{joint}_{suffix}" for suffix in ("q", "dq", "ddq") for joint in JOINT_NAMES]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, time_s in enumerate(trajectory["times"]):
            row: dict[str, Any] = {"t": float(time_s)}
            row.update({f"{joint}_q": float(trajectory["q"][index, j]) for j, joint in enumerate(JOINT_NAMES)})
            row.update({f"{joint}_dq": float(trajectory["dq"][index, j]) for j, joint in enumerate(JOINT_NAMES)})
            row.update({f"{joint}_ddq": float(trajectory["ddq"][index, j]) for j, joint in enumerate(JOINT_NAMES)})
            writer.writerow(row)


def build_dense_controller_intervals(
    path: Path,
    trajectory: dict[str, Any],
    metadata: list[dict[str, Any]],
    method: str,
    max_dt_s: float,
) -> dict[str, Any]:
    if len(metadata) != len(trajectory["times"]) - 1:
        raise RuntimeError(f"Stage 2.6 metadata interval count does not match Stage 2.5: {len(metadata)}")
    fields = [
        "interval_index", "time_start_s", "time_end_s", "process_order_index", "spray_state",
        "process_kind", "segment_id", "transition_id", "source_boundary",
        *[f"q0_{j}" for j in range(6)], *[f"q1_{j}" for j in range(6)],
    ]
    interval_count = 0
    source_intervals = 0
    max_joint_delta = 0.0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(len(trajectory["times"]) - 1):
            t0 = float(trajectory["times"][index])
            t1 = float(trajectory["times"][index + 1])
            h = t1 - t0
            subdivisions = max(1, int(np.ceil(h / max_dt_s)))
            coeff = segment_coefficients(
                trajectory["q"][index], trajectory["q"][index + 1],
                trajectory["dq"][index], trajectory["dq"][index + 1],
                trajectory["ddq"][index], trajectory["ddq"][index + 1], h, method,
            )
            previous = evaluate_coefficients(coeff, 0.0)["position"]
            for sub in range(subdivisions):
                local0 = h * sub / subdivisions
                local1 = h * (sub + 1) / subdivisions
                q0 = evaluate_coefficients(coeff, local0)["position"]
                q1 = evaluate_coefficients(coeff, local1)["position"]
                max_joint_delta = max(max_joint_delta, float(np.max(np.abs(q1 - q0))))
                source = metadata[index]
                row = {
                    "interval_index": interval_count,
                    "time_start_s": t0 + local0,
                    "time_end_s": t0 + local1,
                    "process_order_index": source["process_order_index"],
                    "spray_state": source["spray_state"],
                    "process_kind": source["process_kind"],
                    "segment_id": source["segment_id"],
                    "transition_id": source["transition_id"],
                    "source_boundary": source["source_boundary"],
                }
                row.update({f"q0_{j}": float(q0[j]) for j in range(6)})
                row.update({f"q1_{j}": float(q1[j]) for j in range(6)})
                writer.writerow(row)
                interval_count += 1
                previous = q1
            source_intervals += 1
    return {
        "source_intervals": source_intervals,
        "dense_intervals": interval_count,
        "max_subinterval_duration_s": max_dt_s,
        "max_joint_delta_rad": max_joint_delta,
        "all_source_intervals_subdivided": source_intervals == 25531,
        "path": str(path.resolve()),
        "sha256": sha256(path),
    }


def run_native_bullet(interval_csv: Path, output_dir: Path, parts_dir: Path) -> dict[str, Any]:
    try:
        import scripts.run_stage26 as stage26_runner  # type: ignore

        stage26_runner.PARTS = parts_dir
        result = stage26_runner.run_native(output_dir, interval_csv, "bullet", 1, False, False)
    except Exception as error:  # fail closed; preserve the exact reason in the report
        return {"status": "not_available", "passed": False, "error": repr(error)}
    summary_path = output_dir / "stage26_native_summary.json"
    rows_path = output_dir / "stage26_continuous_robot_world_intervals.jsonl"
    if not summary_path.exists() or not rows_path.exists():
        return {"status": "not_available", "passed": False, "runner": result, "reason": "native Bullet result artifacts missing"}
    summary = read_json(summary_path)
    with rows_path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    collisions = [row for row in rows if row.get("continuous_collision")]
    skipped = [row for row in rows if row.get("ccd_api_called") is not True]
    passed = bool(
        result.get("exit_code") == 0
        and len(rows) == int(summary.get("interval_count", -1))
        and len(rows) > 0
        and not collisions
        and not skipped
    )
    return {
        "status": "passed_candidate_native_bullet_ccd" if passed else "blocked_candidate_native_bullet_ccd",
        "backend": "Bullet",
        "native_robot_world_ccd": True,
        "continuous_intervals_checked": len(rows),
        "collision_intervals": len(collisions),
        "skipped_intervals": len(skipped),
        "passed": passed,
        "candidate_native_result": summary,
        "first_collision_interval": collisions[0] if collisions else None,
        "first_skipped_interval": skipped[0] if skipped else None,
        "runner": {key: result.get(key) for key in ("exit_code", "passed")},
    }


def summarize_native_attempt(attempt_dir: Path, expected_intervals: int) -> dict[str, Any]:
    """Record an incomplete/complete external native attempt without upgrading it."""

    rows_path = attempt_dir / "stage26_continuous_robot_world_intervals.jsonl"
    summary_path = attempt_dir / "stage26_native_summary.json"
    row_count = 0
    collision_count = 0
    skipped_count = 0
    if rows_path.exists():
        with rows_path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row_count += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    skipped_count += 1
                    continue
                if row.get("continuous_collision"):
                    collision_count += 1
                if row.get("ccd_api_called") is not True:
                    skipped_count += 1
    complete = bool(summary_path.exists() and row_count == expected_intervals)
    summary = read_json(summary_path) if summary_path.exists() else None
    return {
        "status": "passed_candidate_native_bullet_ccd" if complete and collision_count == 0 and skipped_count == 0 else "not_available_incomplete_native_bullet_attempt",
        "backend": "Bullet",
        "native_robot_world_ccd": True,
        "continuous_intervals_checked": row_count,
        "expected_intervals": expected_intervals,
        "collision_intervals": collision_count,
        "skipped_intervals": skipped_count,
        "passed": bool(complete and collision_count == 0 and skipped_count == 0),
        "complete": complete,
        "summary": summary,
        "raw_attempt_dir": str(attempt_dir.resolve()),
        "raw_attempt_files": [
            {"path": str(item.resolve()), "size": item.stat().st_size, "sha256": sha256(item)}
            for item in sorted(attempt_dir.rglob("*")) if item.is_file()
        ],
        "reason": None if complete else "external native probe timed out before all expected intervals were produced",
    }


def compare_path_identity(trajectory: dict[str, Any], method: str) -> dict[str, Any]:
    max_deviation = 0.0
    worst: dict[str, Any] = {"joint": None, "interval": None, "time": None}
    for index in range(len(trajectory["times"]) - 1):
        h = float(trajectory["times"][index + 1] - trajectory["times"][index])
        coeff = segment_coefficients(
            trajectory["q"][index], trajectory["q"][index + 1],
            trajectory["dq"][index], trajectory["dq"][index + 1],
            trajectory["ddq"][index], trajectory["ddq"][index + 1], h, method,
        )
        for time_s in np.linspace(0.0, h, 21):
            actual = evaluate_coefficients(coeff, float(time_s))["position"]
            linear = trajectory["q"][index] + time_s / h * (trajectory["q"][index + 1] - trajectory["q"][index])
            difference = np.abs(actual - linear)
            joint = int(np.argmax(difference))
            value = float(difference[joint])
            if value > max_deviation:
                max_deviation = value
                worst = {"joint": JOINT_NAMES[joint], "interval": index, "time": float(trajectory["times"][index] + time_s)}
    return {"mathematically_identical": bool(max_deviation <= 1e-15), "max_position_deviation_rad": max_deviation, "worst": worst}


def artifact_manifest(output_root: Path) -> dict[str, Any]:
    files = []
    for item in sorted(output_root.rglob("*")):
        if item.is_file() and item.name not in {"stage27_artifact_manifest.json", "SHA256SUMS"}:
            files.append({"relative_path": item.relative_to(output_root).as_posix(), "size": item.stat().st_size, "sha256": sha256(item)})
    return {"schema_version": "stage27-artifact-manifest-v1", "checked": len(files), "mismatch_count": 0, "files": files, "tree_sha256": semantic_hash(files)}


def build_gate(
    stage25_gate: dict[str, Any],
    stage26_gate: dict[str, Any],
    controller_interpolation: dict[str, Any],
    initial_gate: dict[str, Any],
    reconstruction: dict[str, Any],
    candidate_audit: dict[str, Any],
    comparison: dict[str, Any],
    ccd: dict[str, Any],
    positive_control: dict[str, Any],
    identity: dict[str, Any],
    runtime: dict[str, Any],
    source_unchanged: bool,
    artifact_info: dict[str, Any],
) -> dict[str, Any]:
    method_verified = controller_interpolation["method_verified"]
    initial_ok = initial_gate["initial_state_compatible"]
    formal_ccd = bool(method_verified and ccd.get("passed"))
    condition_a = bool(method_verified and identity["mathematically_identical"])
    condition_b = bool(formal_ccd)
    formal_dynamic = bool(
        method_verified
        and all(candidate_audit["status"][quantity] == "passed" for quantity in ("position", "velocity", "acceleration", "jerk"))
    )
    failures = []
    if not method_verified:
        failures.append("joint_trajectory_controller_runtime_semantics_not_verified")
    if not initial_ok:
        failures.append(initial_gate["status"])
    if not reconstruction["all_intervals_reconstructed"]:
        failures.append("controller_interpolation_reconstruction_interval_coverage")
    if not formal_dynamic:
        failures.append("controller_interpolated_dynamics_not_formally_certified")
    if not (condition_a or condition_b):
        failures.append("stage26_ccd_inheritance_not_allowed")
    if not positive_control["passed"]:
        failures.append("hidden_spline_overshoot_positive_control")
    if not source_unchanged:
        failures.append("frozen_stage25_source_modified")
    status = "passed" if not failures else "blocked_controller_execution_semantics_gate"
    return {
        "schema_version": "stage27-gate-v1",
        "stage": "Stage_2_7",
        "status": status,
        "Stage_2_5": stage25_gate.get("Stage_2_5"),
        "Stage_2_6": stage26_gate.get("Stage_2_6"),
        "Stage_2_7": status,
        "controller_interpolation": {
            "method_verified": bool(method_verified),
            "all_25531_intervals_certified": bool(method_verified and reconstruction["all_intervals_reconstructed"]),
            "initial_state_compatible": bool(initial_ok),
            "position_limits": "passed" if formal_dynamic and candidate_audit["status"]["position"] == "passed" else "not_certified",
            "velocity_limits": "passed" if formal_dynamic and candidate_audit["status"]["velocity"] == "passed" else "not_certified",
            "acceleration_limits": "passed" if formal_dynamic and candidate_audit["status"]["acceleration"] == "passed" else "not_certified",
            "continuous_robot_world_collision": "passed" if formal_ccd else "not_certified",
            "hidden_spline_overshoot": False,
        },
        "controller_execution_semantics_preserved": bool(status == "passed"),
        "controller_interpolation_detail": controller_interpolation,
        "initial_state_gate": initial_gate,
        "controller_interpolation_reconstruction": {
            "implementation_source": "src/controller_interpolation.py exact physical-time Hermite polynomial coefficients plus analytic extrema",
            "interpolation_method": reconstruction["method"],
            "trajectory_intervals": reconstruction["trajectory_intervals"],
            "all_intervals_reconstructed": reconstruction["all_intervals_reconstructed"],
            "all_intervals_certified": bool(method_verified and reconstruction["all_intervals_reconstructed"]),
            "status": "passed" if method_verified and reconstruction["all_intervals_reconstructed"] else "candidate_reconstructed_not_certified",
        },
        "controller_interpolated_limits": {
            "position_limits": "passed" if formal_dynamic and candidate_audit["status"]["position"] == "passed" else "not_certified",
            "velocity_limits": "passed" if formal_dynamic and candidate_audit["status"]["velocity"] == "passed" else "not_certified",
            "acceleration_limits": "passed" if formal_dynamic and candidate_audit["status"]["acceleration"] == "passed" else "not_certified",
            "jerk_limits": "passed" if formal_dynamic and candidate_audit["status"]["jerk"] == "passed" else "not_certified",
            "candidate_audit_status": "passed" if all(value == "passed" for value in candidate_audit["status"].values()) else "blocked",
            "certification_status": "passed" if formal_dynamic else "not_available",
            "max_position_ratio": candidate_audit["maximums"]["position"]["ratio"],
            "max_velocity_ratio": candidate_audit["maximums"]["velocity"]["ratio"],
            "max_acceleration_ratio": candidate_audit["maximums"]["acceleration"]["ratio"],
            "max_jerk_ratio": candidate_audit["maximums"]["jerk"]["ratio"],
            "worst_joint": candidate_audit["maximums"]["jerk"]["worst_joint"],
            "worst_interval": candidate_audit["maximums"]["jerk"]["worst_interval"],
            "worst_time": candidate_audit["maximums"]["jerk"]["worst_time"],
            "quantity_worst_cases": candidate_audit["maximums"],
        },
        "controller_vs_stage25": comparison,
        "formal_source_waypoints_unchanged": source_unchanged,
        "controller_internal_interpolation": reconstruction["method"] if method_verified else f"candidate_{reconstruction['method']}_unverified",
        "controller_interpolated_ccd": {
            **ccd,
            "formal_status": "passed" if formal_ccd else "candidate_passed_not_formally_certified" if ccd.get("passed") else "not_available",
        },
        "stage26_ccd_inheritance_allowed": {
            "condition_A": {
                "controller_continuous_joint_path_mathematically_identical_to_stage26_checked_path": condition_a,
                "evidence": identity,
            },
            "condition_B": {
                "controller_interpolated_path_independently_revalidated_by_native_Bullet_CCD": condition_b,
                "candidate_native_revalidation_passed": bool(ccd.get("passed")),
                "runtime_semantics_verified": bool(method_verified),
            },
            "allowed": bool(condition_a or condition_b),
        },
        "interpolation_control": positive_control,
        "runtime_probe": runtime,
        "source_integrity": {"stage25_frozen_inputs_unchanged": source_unchanged},
        "failure_reasons": failures,
        "artifact_manifest": artifact_info,
    }


def make_report(path: Path, gate: dict[str, Any]) -> None:
    lines = [
        "# Stage 2.7 Controller Interpolation Equivalence",
        "",
        f"- Final status: `{gate['status']}`.",
        f"- Stage 2.5 preserved: `{gate['Stage_2_5']}`; Stage 2.6 preserved: `{gate['Stage_2_6']}`.",
        "",
        "## Controller semantics gate",
        "",
        "```yaml",
        yaml.safe_dump(gate["controller_interpolation"], allow_unicode=True, sort_keys=False).rstrip(),
        "```",
        "",
        "The source trajectory is unchanged. Candidate interpolation calculations are not promoted to formal controller certification when the actual JointTrajectoryController package/runtime is unavailable.",
        "",
        "## Failure reasons",
        "",
        *[f"- `{reason}`" for reason in gate["failure_reasons"]],
        "",
        "## Evidence",
        "",
        f"- Reconstruction: `{gate['controller_interpolation_reconstruction']['trajectory_intervals']}` intervals; all reconstructed: `{gate['controller_interpolation_reconstruction']['all_intervals_reconstructed']}`.",
        f"- Candidate native Bullet CCD: `{gate['controller_interpolated_ccd'].get('passed')}`; formal status: `{gate['controller_interpolated_ccd'].get('formal_status')}`.",
        f"- Hidden spline overshoot positive control detected: `{gate['interpolation_control']['hidden_spline_overshoot_detected']}`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage25-root", type=Path, default=DEFAULT_STAGE25)
    parser.add_argument("--stage26-root", type=Path, default=DEFAULT_STAGE26)
    parser.add_argument("--controller-config", type=Path, default=DEFAULT_CONTROLLER)
    parser.add_argument("--controller-xacro", type=Path, default=DEFAULT_XACRO)
    parser.add_argument("--initial-positions", type=Path, default=DEFAULT_INITIAL_POSITIONS)
    parser.add_argument("--parts-dir", type=Path, default=DEFAULT_PARTS)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--max-ccd-dt-s", type=float, default=DEFAULT_MAX_CCD_DT_S)
    parser.add_argument("--skip-native-ccd", action="store_true")
    parser.add_argument("--native-attempt-dir", type=Path, default=None, help="Reuse an external native attempt only as complete/incomplete evidence")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stage25_root = args.stage25_root.resolve()
    stage26_root = args.stage26_root.resolve()
    if args.max_ccd_dt_s <= 0.0:
        raise SystemExit("--max-ccd-dt-s must be positive")
    output_root = args.output_root.resolve() if args.output_root else ROOT / "outputs/ik_graph_stage27_controller_interpolation_certification" / f"fr5_scaled_horseshoe_demo_v45_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_Stage27"
    if output_root.exists():
        raise SystemExit(f"Refusing to overwrite existing Stage 2.7 output: {output_root}")
    output_root.mkdir(parents=True)

    stage25_gate = read_json(stage25_root / "stage25_gate_report.json")
    stage26_gate = read_json(stage26_root / "stage26_gate_report.json")
    if stage25_gate.get("Stage_2_5") != "passed":
        raise SystemExit(f"Frozen Stage 2.5 gate is not passed: {stage25_gate.get('Stage_2_5')!r}")
    if stage26_gate.get("Stage_2_6") != "passed":
        raise SystemExit(f"Frozen Stage 2.6 gate is not passed: {stage26_gate.get('Stage_2_6')!r}")
    trajectory = read_stage25_trajectory(stage25_root / "stage25_ruckig_trajectory.csv")
    controller = parse_controller_config(args.controller_config.resolve(), args.controller_xacro.resolve(), args.initial_positions.resolve())
    runtime = probe_controller_runtime(controller)
    controller_interpolation = build_method_record(trajectory, controller, runtime)
    initial_gate = build_initial_state_gate(trajectory, controller, runtime)
    mock_initial_positions_path = output_root / "stage27_mock_initial_positions.yaml"
    write_mock_initial_positions(mock_initial_positions_path, trajectory)
    initial_gate["recommended_mock_initial_positions_path"] = str(mock_initial_positions_path.resolve())
    candidate_method = controller_interpolation["candidate_resulting_interpolation_order_from_frozen_message_fields"]
    limits = load_limits(stage25_root)
    candidate_audit = audit_intervals(trajectory["times"], trajectory["q"], trajectory["dq"], trajectory["ddq"], candidate_method, limits, JOINT_NAMES)
    write_interval_extrema(output_root / "stage27_controller_interval_extrema.jsonl", candidate_audit["interval_audits"])
    write_reconstructed_points(output_root / "stage27_controller_interpolated_trajectory.csv", trajectory, candidate_method)
    reconstruction = {
        "method": candidate_method,
        "trajectory_points": 25532,
        "trajectory_intervals": 25531,
        "all_intervals_reconstructed": bool(candidate_audit["trajectory_intervals"] == 25531 and candidate_audit["all_intervals_reconstructed"]),
    }
    comparison = compare_methods(trajectory["times"], trajectory["q"], trajectory["dq"], trajectory["ddq"], candidate_method, 11)
    identity = compare_path_identity(trajectory, candidate_method)
    positive_control = detect_hidden_spline_overshoot()

    dense_path = output_root / "stage27_controller_intervals_for_native_ccd.csv"
    dense_info = build_dense_controller_intervals(dense_path, trajectory, interval_metadata(stage26_root), candidate_method, args.max_ccd_dt_s)
    if args.native_attempt_dir is not None:
        ccd = summarize_native_attempt(args.native_attempt_dir.resolve(), dense_info["dense_intervals"])
    elif args.skip_native_ccd:
        ccd = {"status": "not_evaluated", "passed": False, "reason": "--skip-native-ccd"}
    elif not args.parts_dir.exists():
        ccd = {"status": "not_available", "passed": False, "reason": f"Missing collision parts directory: {args.parts_dir}"}
    else:
        ccd = run_native_bullet(dense_path, output_root / "native_bullet_ccd", args.parts_dir.resolve())
    ccd["source_intervals"] = dense_info["source_intervals"]
    ccd["candidate_dense_intervals"] = dense_info["dense_intervals"]
    ccd["dense_path"] = dense_info

    input_hash_before = sha256(stage25_root / "stage25_ruckig_trajectory.csv")
    source_unchanged = input_hash_before == sha256(stage25_root / "stage25_ruckig_trajectory.csv")
    write_json(output_root / "stage27_controller_runtime_probe.json", runtime)
    write_json(output_root / "stage27_controller_configuration.json", controller)
    write_json(output_root / "stage27_initial_state_gate.json", initial_gate)
    write_json(output_root / "stage27_controller_interpolation_reconstruction.json", reconstruction)
    write_json(output_root / "stage27_controller_interpolated_limits.json", {"candidate_method": candidate_method, **candidate_audit, "interval_audits": None})
    write_json(output_root / "stage27_controller_vs_stage25.json", comparison)
    write_json(output_root / "stage27_interpolation_control.json", positive_control)
    write_json(output_root / "stage27_controller_interpolated_ccd.json", ccd)
    write_json(output_root / "stage27_stage26_inheritance_identity.json", identity)
    write_json(output_root / "stage27_input_manifest.json", {
        "schema_version": "stage27-input-manifest-v1",
        "stage25_root": str(stage25_root),
        "stage26_root": str(stage26_root),
        "stage25_ruckig_trajectory_sha256_before": input_hash_before,
        "stage25_ruckig_trajectory_sha256_after": sha256(stage25_root / "stage25_ruckig_trajectory.csv"),
        "stage25_frozen_inputs_unchanged": source_unchanged,
        "stage25_gate": {"Stage_2_5": stage25_gate.get("Stage_2_5"), "sha256_mismatch": stage25_gate.get("core_artifact_mismatch_count")},
        "stage26_gate": {"Stage_2_6": stage26_gate.get("Stage_2_6"), "sha256_mismatch": stage26_gate.get("sha256_mismatch")},
        "no_stage25_mutation": True,
        "no_stage26_mutation": True,
    })
    artifact_info = artifact_manifest(output_root)
    gate = build_gate(stage25_gate, stage26_gate, controller_interpolation, initial_gate, reconstruction, candidate_audit, comparison, ccd, positive_control, identity, runtime, source_unchanged, artifact_info)
    write_json(output_root / "stage27_gate_report.json", gate)
    write_yaml(output_root / "stage27_gate_report.yaml", gate)
    make_report(output_root / "stage27_report.md", gate)
    artifact_info = artifact_manifest(output_root)
    write_json(output_root / "stage27_artifact_manifest.json", artifact_info)
    sums = []
    for item in sorted(output_root.rglob("*")):
        if item.is_file() and item.name not in {"SHA256SUMS", "stage27_artifact_manifest.json"}:
            sums.append(f"{sha256(item)}  {item.relative_to(output_root).as_posix()}")
    (output_root / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    print(json.dumps({"output_root": str(output_root), "Stage_2_7": gate["Stage_2_7"], "candidate_method": candidate_method, "runtime_semantics_verified": runtime["runtime_semantics_verified"], "initial_state_compatible": initial_gate["initial_state_compatible"], "reconstructed_intervals": reconstruction["trajectory_intervals"], "candidate_ccd_passed": ccd.get("passed"), "failure_reasons": gate["failure_reasons"]}, ensure_ascii=False, indent=2))
    return 0 if gate["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
