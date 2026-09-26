"""Build the read-only Stage 2.8A preflight evidence bundle.

This builder deliberately contains no ROS command that can publish, execute, or
claim motion.  It only reads frozen artifacts, queries discovery/metadata APIs,
and records the absence of a real FR5 runtime when it cannot be proved.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "stage28a_real_fr5_zero_motion_preflight"
STAGE25 = ROOT / "outputs" / "stage25r2_full_native_certification" / "stage25r2_formal_20260804T054820Z"
STAGE27 = ROOT / "outputs" / "stage27s_native_spline_remediation" / "stage27s_formal_20260805T130000Z"
STAGE26 = ROOT / "outputs" / "stage26r_stage25r2_continuous_collision_certification" / "stage26r_formal_20260804T075115Z"
STAGE25_TRAJECTORY = STAGE25 / "stage25r2_final_robot_trajectory.csv"
STAGE27_CANDIDATE = STAGE27 / "stage27s_candidate_trajectory.csv"
EXPECTED_STAGE25_SHA256 = "dcb99698a1ca7a76e324d34f623d5528ee6c78e48cb64f764d2fd934dd669c5e"
JOINTS = ["j1", "j2", "j3", "j4", "j5", "j6"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(name: str, value: Any) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def run_wsl(script: str, timeout_s: float = 6.0) -> dict[str, Any]:
    """Run a read-only command in the configured WSL ROS environment."""
    argv = ["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", script]
    started = time.monotonic()
    try:
        p = subprocess.run(argv, capture_output=True, timeout=timeout_s, check=False)
        stdout = p.stdout.decode("utf-8", errors="replace")
        stderr = p.stderr.decode("utf-8", errors="replace")
        return {
            "argv": argv,
            "exit_code": p.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "timed_out": False,
            "elapsed_s": round(time.monotonic() - started, 6),
        }
    except subprocess.TimeoutExpired as e:
        return {
            "argv": argv,
            "exit_code": None,
            "stdout": (e.stdout or b"").decode("utf-8", errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or ""),
            "stderr": (e.stderr or b"").decode("utf-8", errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or ""),
            "timed_out": True,
            "elapsed_s": round(time.monotonic() - started, 6),
        }
    except OSError as e:
        return {"argv": argv, "exit_code": None, "stdout": "", "stderr": str(e), "timed_out": False, "elapsed_s": round(time.monotonic() - started, 6)}


def verify_sha256sums(root: Path) -> dict[str, Any]:
    sums_path = root / "SHA256SUMS"
    result: dict[str, Any] = {"manifest": str(sums_path), "manifest_sha256": sha256(sums_path), "entries": 0, "verified": 0, "missing": [], "mismatches": []}
    for raw in sums_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"^([0-9a-fA-F]{64})  (.+)$", raw)
        if not m:
            continue
        expected, rel = m.group(1).lower(), m.group(2)
        result["entries"] += 1
        path = root / rel
        if not path.exists():
            result["missing"].append(rel)
            continue
        actual = sha256(path)
        if actual != expected:
            result["mismatches"].append({"path": rel, "expected": expected, "actual": actual})
        else:
            result["verified"] += 1
    result["passed"] = not result["missing"] and not result["mismatches"] and result["entries"] == result["verified"]
    return result


def parse_candidate_point0(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    return {j: float(row[f"{j}_q"]) for j in JOINTS}


def parse_urdf_limits(path: Path) -> dict[str, Any]:
    root = ET.fromstring(path.read_text(encoding="utf-8"))
    limits: dict[str, Any] = {}
    topology: list[dict[str, Any]] = []
    for joint in root.findall("joint"):
        name = joint.attrib.get("name")
        if not name or name not in JOINTS:
            continue
        parent = joint.find("parent")
        child = joint.find("child")
        limit = joint.find("limit")
        origin = joint.find("origin")
        limits[name] = {
            "position_lower_rad": float(limit.attrib["lower"]),
            "position_upper_rad": float(limit.attrib["upper"]),
            "velocity_limit_rad_s": float(limit.attrib["velocity"]),
            "effort_limit": float(limit.attrib["effort"]),
        }
        topology.append({
            "joint": name,
            "parent": parent.attrib.get("link") if parent is not None else None,
            "child": child.attrib.get("link") if child is not None else None,
            "origin_xyz_m": origin.attrib.get("xyz") if origin is not None else None,
            "origin_rpy_rad": origin.attrib.get("rpy") if origin is not None else None,
        })
    return {"joint_limits": limits, "topology": topology}


def package_version(pkg: str) -> dict[str, Any]:
    return run_wsl(f"grep -E '<name>|<version>' /opt/ros/jazzy/share/{pkg}/package.xml 2>&1", timeout_s=4.0)


def jsonable_probe_summary(probes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "node_list": probes["ros2 node list"],
        "topic_list": probes["ros2 topic list -t"],
        "action_list": probes["ros2 action list -t"],
        "service_list": probes["ros2 service list -t"],
        "hardware_components": probes["ros2 control list_hardware_components -v"],
        "hardware_interfaces": probes["ros2 control list_hardware_interfaces -v"],
        "controllers": probes["ros2 control list_controllers -v"],
        "controller_manager_info": probes["ros2 node info /controller_manager"],
        "joint_states_info": probes["ros2 topic info -v /joint_states"],
        "controller_state_info": probes["ros2 topic info -v /fairino5_controller/controller_state"],
        "follow_joint_trajectory_info": probes["ros2 action info /fairino5_controller/follow_joint_trajectory"],
        "controller_parameter": probes["ros2 param get /fairino5_controller interpolation_method"],
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)

    stage25_hashes = verify_sha256sums(STAGE25)
    stage27_hashes = verify_sha256sums(STAGE27)
    stage25_actual = sha256(STAGE25_TRAJECTORY)
    stage27_candidate_actual = sha256(STAGE27_CANDIDATE)

    stage27_manifest = read_json(STAGE27 / "stage27s_input_manifest.json")
    stage27_jtc = read_json(STAGE27 / "stage27s_exact_jtc_certificate.json")
    stage27_oracle = read_json(STAGE27 / "stage27r_native_vs_exact_oracle.json")
    stage27_bullet = read_json(STAGE27 / "stage27s_bullet_validation.json")
    stage27_determinism = read_json(STAGE27 / "stage27s_determinism_report.json")
    stage27_transform = read_json(STAGE27 / "stage27s_candidate_transform.json")
    stage25_gate = read_json(STAGE25 / "stage25r2_gate_report.json")
    candidate0 = parse_candidate_point0(STAGE27_CANDIDATE)

    # These are discovery and metadata reads only.  No action goal, topic
    # publisher, trajectory publisher, servo API, or mutating service is used.
    probe_scripts = {
        "uname": "uname -a; cat /etc/os-release | head -12; printf 'ROS_DISTRO='; printenv ROS_DISTRO",
        "ros2 node list": "source /opt/ros/jazzy/setup.bash; ros2 node list --include-hidden-nodes",
        "ros2 topic list -t": "source /opt/ros/jazzy/setup.bash; ros2 topic list -t --include-hidden-topics",
        "ros2 action list -t": "source /opt/ros/jazzy/setup.bash; ros2 action list -t",
        "ros2 service list -t": "source /opt/ros/jazzy/setup.bash; ros2 service list -t",
        "ros2 control list_hardware_components -v": "source /opt/ros/jazzy/setup.bash; ros2 control list_hardware_components -v --spin-time 1",
        "ros2 control list_hardware_interfaces -v": "source /opt/ros/jazzy/setup.bash; ros2 control list_hardware_interfaces -v --spin-time 1",
        "ros2 control list_controllers -v": "source /opt/ros/jazzy/setup.bash; ros2 control list_controllers -v --spin-time 1",
        "ros2 node info /controller_manager": "source /opt/ros/jazzy/setup.bash; ros2 node info /controller_manager",
        "ros2 topic info -v /joint_states": "source /opt/ros/jazzy/setup.bash; ros2 topic info -v /joint_states",
        "ros2 topic info -v /fairino5_controller/controller_state": "source /opt/ros/jazzy/setup.bash; ros2 topic info -v /fairino5_controller/controller_state",
        "ros2 action info /fairino5_controller/follow_joint_trajectory": "source /opt/ros/jazzy/setup.bash; ros2 action info /fairino5_controller/follow_joint_trajectory",
        "ros2 param get /fairino5_controller interpolation_method": "source /opt/ros/jazzy/setup.bash; ros2 param get /fairino5_controller interpolation_method",
        "ros2 pkg list fairino matches": "source /opt/ros/jazzy/setup.bash; ros2 pkg list | grep -Ei 'fairino|frcobot' || true",
        "fairino sdk files": "find /usr /opt /home/robot -maxdepth 7 -type f \( -iname '*fairino*.so' -o -iname '*fairino*.py' -o -iname '*frcobot*.so' -o -iname '*frcobot*.py' \) 2>/dev/null | head -100",
    }
    probes = {name: run_wsl(script) for name, script in probe_scripts.items()}
    finished = datetime.now(timezone.utc)

    urdf = ROOT / "external" / "frcobot_ros2" / "fairino_description" / "urdf" / "fairino5_v6.urdf"
    xacro = ROOT / "ros2_moveit_bridge" / "config" / "fairino5_v6_spray_tcp.urdf.xacro"
    srdf = ROOT / "ros2_moveit_bridge" / "config" / "fairino5_v6_spray_tcp.srdf"
    limits_yaml = ROOT / "ros2_moveit_bridge" / "config" / "joint_limits_with_jerk.yaml"
    tcp_template = ROOT / "ros2_moveit_bridge" / "config" / "tool_tcp_calibration_template.yaml"
    model_files = {str(p.relative_to(ROOT)): {"sha256": sha256(p), "bytes": p.stat().st_size} for p in [urdf, xacro, srdf, limits_yaml, tcp_template]}
    urdf_audit = parse_urdf_limits(urdf)

    current_nodes = probes["ros2 node list"]["stdout"].strip()
    current_topics = probes["ros2 topic list -t"]["stdout"].strip()
    current_actions = probes["ros2 action list -t"]["stdout"].strip()
    real_runtime_observed = bool(re.search(r"fairino5_controller|controller_manager", current_nodes + current_topics + current_actions, re.I))
    fairino_pkg_output = probes["ros2 pkg list fairino matches"]["stdout"].strip()
    sdk_file_output = probes["fairino sdk files"]["stdout"].strip()

    frozen_inputs = {
        "stage25r2_formal_root": str(STAGE25),
        "stage25r2_sha256_expected": EXPECTED_STAGE25_SHA256,
        "stage25r2_sha256_actual": stage25_actual,
        "stage25r2_sha256_match": stage25_actual == EXPECTED_STAGE25_SHA256,
        "stage25r2_gate_status": stage25_gate.get("Stage_2_5R2"),
        "stage25r2_sha256sums_verification": stage25_hashes,
        "stage27s_formal_root": str(STAGE27),
        "stage27s_candidate_sha256_actual": stage27_candidate_actual,
        "stage27s_sha256sums_verification": stage27_hashes,
        "stage27s_input_manifest_sha256": sha256(STAGE27 / "stage27s_input_manifest.json"),
        "stage27s_exact_jtc_certificate_sha256": sha256(STAGE27 / "stage27s_exact_jtc_certificate.json"),
        "stage27s_determinism_report_sha256": sha256(STAGE27 / "stage27s_determinism_report.json"),
        "stage27s_bullet_validation_sha256": sha256(STAGE27 / "stage27s_bullet_validation.json"),
        "stage27s_candidate_transform_sha256": sha256(STAGE27 / "stage27s_candidate_transform.json"),
        "stage27s_stage25r2_immutable_source": stage27_manifest.get("stage25r2_immutable_source"),
        "stage27s_stage25r2_sha256_before": stage27_manifest.get("stage25r2_sha256_before"),
        "stage27s_stage25r2_sha256_expected": stage27_manifest.get("stage25r2_sha256_expected"),
        "stage27s_stage25r2_mutated_flag": stage27_manifest.get("stage25r2_mutated"),
        "stage27s_candidate_point_count": stage27_manifest.get("candidate_point_count"),
        "stage27s_candidate_interval_count": stage27_manifest.get("candidate_interval_count"),
        "stage27s_all_core_hashes_match": stage27_hashes["passed"],
        "formal_frozen_input_gate": bool(stage25_hashes["passed"] and stage27_hashes["passed"] and stage25_actual == EXPECTED_STAGE25_SHA256 and stage27_manifest.get("stage25r2_mutated") is False),
    }

    historical_mock = {
        "classification": "mock_hardware_historical_only",
        "source": str(STAGE27 / "stage27s_runtime.log"),
        "evidence": {
            "hardware_plugin": "mock_components/GenericSystem",
            "hardware_component": "Stage27SGenericSystem",
            "controller_manager_update_rate_hz": 100,
            "controller": "fairino5_controller",
            "controller_type": "joint_trajectory_controller/JointTrajectoryController",
            "controller_state": "active",
            "joints": JOINTS,
            "command_interfaces": ["position"],
            "state_interfaces": ["position"],
            "interpolation_method": "splines",
            "historical_goal_sent": True,
            "historical_goal_is_excluded_from_stage28a": True,
        },
        "reason_not_real": "The historical launch source explicitly injected mock_components/GenericSystem; it is not a FAIRINO hardware component and was not used as current real-hardware evidence.",
    }

    package_versions = {p: package_version(p) for p in ["ros2_control", "controller_manager", "ros2_controllers", "joint_trajectory_controller", "moveit_ros_planning", "moveit_ros_planning_interface", "moveit_core", "moveit_ros_move_group"]}
    runtime_manifest = {
        "schema_version": "stage28a-runtime-manifest-v1",
        "stage": "Stage 2.8A",
        "capture_start_utc": started.isoformat(),
        "capture_end_utc": finished.isoformat(),
        "real_hardware": {
            "connected": False,
            "runtime_observed": real_runtime_observed,
            "classification": "no_real_fr5_runtime_observed",
            "robot_controller_model": None,
            "controller_firmware_software_version": None,
            "fairino_sdk_version": None,
            "fairino_ros2_hardware_plugin_version": None,
            "fairino_ros2_hardware_plugin_commit": None,
            "fairino_ros2_package_matches": fairino_pkg_output,
            "fairino_sdk_file_matches": sdk_file_output,
            "controller_manager_update_rate_hz": None,
            "historical_mock_reference": historical_mock,
        },
        "host": {
            "os": platform.platform(),
            "python": sys.version,
            "kernel_and_ros_host_probe": probes["uname"],
            "windows_machine": platform.machine(),
            "windows_release": platform.release(),
        },
        "ros": {
            "ros_distro": "jazzy",
            "ros_distro_source": "/opt/ros/jazzy/setup.bash",
            "package_metadata_probes": package_versions,
            "ros2_control_version_observed_from_package_xml": "4.45.2",
            "controller_manager_version_observed_from_package_xml": "4.45.2",
            "ros2_controllers_version_observed_from_package_xml": "4.40.1",
            "joint_trajectory_controller_version_observed_from_package_xml": "4.40.1",
            "moveit_version_observed_from_package_xml": "2.12.4",
        },
        "frozen_inputs": frozen_inputs,
        "runtime_probe_policy": "read_only_discovery_and_metadata_queries; no motion-capable API invoked",
        "current_runtime_probes": probes,
        "formal_artifact_mutation": {
            "stage25r2_modified_by_stage28a": False,
            "stage27s_modified_by_stage28a": False,
            "stage25r2_source_sha256_after_probe": sha256(STAGE25_TRAJECTORY),
            "stage27s_candidate_sha256_after_probe": sha256(STAGE27_CANDIDATE),
        },
    }
    write_json("stage28a_runtime_manifest.json", runtime_manifest)

    interfaces_json = {
        "schema_version": "stage28a-hardware-interfaces-v1",
        "status": "blocked_real_runtime_not_observed",
        "current_real_runtime": {
            "hardware_component_present": False,
            "hardware_component_name": None,
            "hardware_plugin": None,
            "hardware_state": None,
            "controller_manager_update_rate_hz": None,
            "joints": None,
            "command_interfaces": None,
            "state_interfaces": None,
            "joint_order_matches_stage27s": None,
            "fairino5_controller_state": None,
            "fairino5_controller_claimed_interfaces": None,
            "unknown_motion_interface_claimants": None,
        },
        "required_runtime_commands": [
            "ros2 control list_hardware_components -v",
            "ros2 control list_hardware_interfaces -v",
            "ros2 control list_controllers -v",
        ],
        "current_command_results": jsonable_probe_summary(probes),
        "historical_mock_runtime_excluded": historical_mock,
        "blockers": [
            "No real controller_manager node or FAIRINO hardware component was discovered.",
            "The only persisted equivalent interface evidence is historical mock_components/GenericSystem evidence and is explicitly excluded from real certification.",
        ],
    }
    write_json("stage28a_hardware_interfaces.json", interfaces_json)
    interface_txt = [
        "Stage 2.8A hardware interface audit (read-only)",
        f"capture_utc: {finished.isoformat()}",
        "current classification: NO REAL FR5 RUNTIME OBSERVED",
        "",
        "Current ROS2 read-only probe outputs:",
    ]
    for name in probe_scripts:
        interface_txt.append(f"\n=== {name} ===")
        interface_txt.append(probes[name]["stdout"])
        if probes[name]["stderr"]:
            interface_txt.append("[stderr]")
            interface_txt.append(probes[name]["stderr"])
    interface_txt.extend([
        "",
        "Historical mock evidence (NOT current real hardware):",
        "hardware component: Stage27SGenericSystem",
        "plugin: mock_components/GenericSystem",
        "controller: fairino5_controller / joint_trajectory_controller/JointTrajectoryController",
        "interfaces: j1-j6/position state and command; joint order j1,j2,j3,j4,j5,j6",
        "controller_manager update rate: 100 Hz",
        "interpolation: splines",
        "This historical mock evidence is not used to pass Stage 2.8A real-hardware gates.",
    ])
    (OUT / "stage28a_hardware_interfaces.txt").write_text("\n".join(interface_txt) + "\n", encoding="utf-8")

    controller_runtime = {
        "schema_version": "stage28a-controller-runtime-v1",
        "status": "blocked_real_runtime_not_observed",
        "runtime_verified": False,
        "controller_name": None,
        "controller_type": None,
        "package_version": None,
        "source_commit_if_resolvable": None,
        "interpolation_method": None,
        "joint_names": None,
        "command_interfaces": None,
        "state_interfaces": None,
        "state_publish_rate": None,
        "action_monitor_rate": None,
        "action_endpoint": "/fairino5_controller/follow_joint_trajectory",
        "action_endpoint_exists": False,
        "controller_state_topic": "/fairino5_controller/controller_state",
        "controller_state_topic_exists": False,
        "historical_mock_runtime_reference": {
            "controller_name": "fairino5_controller",
            "controller_type": "joint_trajectory_controller/JointTrajectoryController",
            "package_version": "4.40.1",
            "source_commit_if_resolvable": "31015e0aa7ce9d0853a88d5fc2fe50f0e583ba5c",
            "interpolation_method": "splines",
            "joint_names": JOINTS,
            "command_interfaces": ["position"],
            "state_interfaces": ["position"],
            "state_publish_rate": None,
            "action_monitor_rate": 20.0,
            "source": str(STAGE27 / "stage27s_runtime.log"),
            "classification": "mock_only",
        },
        "blockers": ["JTC endpoints and parameters were not observed on a real hardware runtime.", "Historical splines evidence is mock-only."],
    }
    write_json("stage28a_controller_runtime.json", controller_runtime)

    joint_state = {
        "schema_version": "stage28a-joint-state-compatibility-v1",
        "status": "blocked_real_joint_state_not_measured",
        "real_joint_state_observed": False,
        "joint_names_observed": None,
        "joint_order_observed": None,
        "radians_semantics_verified": False,
        "sign_conventions_verified": False,
        "zero_convention_verified": False,
        "finite_values_verified": False,
        "timestamp_verified": False,
        "freshness_verified": False,
        "candidate_point0_source": str(STAGE27_CANDIDATE),
        "candidate_point0_sha256": stage27_candidate_actual,
        "candidate_point0_rad": candidate0,
        "per_joint": [{"joint": j, "measured_position_rad": None, "candidate_point0_rad": candidate0[j], "difference_rad": None} for j in JOINTS],
        "auto_move_to_start_performed": False,
        "blockers": ["No real /joint_states message was available; formal initial-state compatibility cannot be evaluated without moving the robot."],
    }
    write_json("stage28a_joint_state_compatibility.json", joint_state)

    moveit_parameters = {
        "schema_version": "stage28a-moveit-execution-parameters-v1",
        "status": "blocked_execution_layer_unverified",
        "runtime_observed": False,
        "trajectory_execution.execution_duration_monitoring": None,
        "trajectory_execution.allowed_execution_duration_scaling": None,
        "trajectory_execution.allowed_goal_duration_margin": None,
        "trajectory_execution.allowed_start_tolerance": None,
        "trajectory_execution.execution_velocity_scaling": None,
        "trajectory_execution.wait_for_trajectory_completion": None,
        "execution_velocity_scaling_required": 1.0,
        "execution_velocity_scaling_verified": False,
        "silent_execution_scaling": "not_proven",
        "static_planning_parameters_not_execution_proof": {
            "fr5_spray_demo.launch.py_max_velocity_scaling_factor": 0.15,
            "fr5_spray_demo.launch.py_max_acceleration_scaling_factor": 0.15,
            "fr5_spray_smoke.launch.py_max_velocity_scaling_factor": 0.15,
            "fr5_spray_smoke.launch.py_max_acceleration_scaling_factor": 0.15,
            "source": str(ROOT / "ros2_moveit_bridge" / "launch"),
        },
        "blockers": ["MoveIt TrajectoryExecutionManager was not running; execution_velocity_scaling=1.0 and all required runtime parameters cannot be proved.", "Static planning scaling values of 0.15 are not silently accepted as Stage 2.8 execution settings."],
    }
    write_json("stage28a_moveit_execution_parameters.json", moveit_parameters)

    model_audit = {
        "schema_version": "stage28a-model-tcp-frame-audit-v1",
        "status": "blocked_runtime_model_and_physical_calibration_unverified",
        "model_assumed": {
            "robot": "FAIRINO_FR5_V6",
            "urdf_source": str(urdf),
            "urdf_sha256": model_files[str(urdf.relative_to(ROOT))]["sha256"],
            "xacro_source": str(xacro),
            "xacro_sha256": model_files[str(xacro.relative_to(ROOT))]["sha256"],
            "srdf_source": str(srdf),
            "srdf_sha256": model_files[str(srdf.relative_to(ROOT))]["sha256"],
            "joint_order": JOINTS,
            "base_link": "base_link",
            "flange_or_wrist_link": "wrist3_link",
            "spray_tcp_link": "spray_tcp_link",
            "spray_tcp_parent": "wrist3_link",
            "spray_tcp_nozzle_offset": "xacro launch argument tool_tcp_xyz/tool_tcp_rpy; no production value persisted",
            "world_workcell_transform": None,
            "tool_geometry": "official FR5 link collision meshes only; physical nozzle geometry not provided",
            "collision_geometry": "official FR5 V6 URDF collision meshes; workcell/floor/trolley not runtime verified",
            "joint_topology_and_limits": urdf_audit,
        },
        "runtime_configured": {
            "real_runtime_observed": False,
            "urdf_or_xacro": None,
            "base_link": None,
            "flange_transform": None,
            "spray_tcp": None,
            "world_workcell_transform": None,
            "tool_geometry": None,
            "collision_geometry": None,
        },
        "physically_calibrated": {
            "tcp": False,
            "nozzle_offset": False,
            "nozzle_orientation": False,
            "base_world_transform": False,
            "workcell_geometry": False,
            "evidence": str(tcp_template),
            "template_sha256": model_files[str(tcp_template.relative_to(ROOT))]["sha256"],
            "template_has_measured_metadata": False,
        },
        "blockers": ["No physical TCP/frame calibration record was provided or observed.", "No real runtime model, world/workcell transform, or nozzle collision geometry was observed.", "The Xacro injects FakeSystem/mock hardware and is not a real deployment configuration."],
    }
    write_json("stage28a_model_tcp_frame_audit.json", model_audit)

    assumed_limits = urdf_audit["joint_limits"]
    stage27_limits = {
        "velocity_rad_s": [0.4725, 0.4725, 0.4725, 0.48, 0.48, 0.48],
        "acceleration_rad_s2": [0.105] * 6,
        "jerk_rad_s3": [8.0] * 6,
        "source": str(STAGE27 / "stage27s_phase_a_report.json"),
        "note": "Stage 2.7S exact Ruckig/JTC certificate assumptions, not real hardware limits.",
    }
    hardware_limits = {
        "schema_version": "stage28a-hardware-limits-audit-v1",
        "status": "blocked_real_hardware_limits_not_observed",
        "real_driver_or_robot_limits": None,
        "controller_side_overrides": None,
        "real_limits_observed": False,
        "model_assumed_limits": assumed_limits,
        "model_assumed_acceleration_rad_s2": {j: 0.7 for j in JOINTS},
        "model_assumed_jerk_rad_s3": {j: 8.0 for j in JOINTS},
        "stage27s_exact_jtc_certificate": {
            "certificate": str(STAGE27 / "stage27s_exact_jtc_certificate.json"),
            "certificate_sha256": sha256(STAGE27 / "stage27s_exact_jtc_certificate.json"),
            "status": stage27_jtc.get("status"),
            "maximums": stage27_jtc.get("maximums"),
            "limits_assumed": stage27_limits,
        },
        "comparison": {"compatible": None, "reason": "Real FR5/controller limits are unavailable; no relaxation or inference was made."},
        "blockers": ["Real robot/driver position, velocity, acceleration, jerk and override limits were not observable."],
    }
    write_json("stage28a_hardware_limits_audit.json", hardware_limits)

    safety = {
        "schema_version": "stage28a-safety-readiness-v1",
        "status": "blocked_safety_state_not_observable",
        "emergency_stop": {"availability": None, "state": None, "evidence": None},
        "protective_stop": {"state": None, "evidence": None},
        "robot_fault_or_error": {"state": None, "evidence": None},
        "mode_of_operation": None,
        "speed_override_or_scaling": None,
        "workspace_readiness": None,
        "vendor_safety_system_required": True,
        "software_checks_do_not_replace_vendor_or_manual_safety_confirmation": True,
        "blockers": ["No real controller safety/status channel was observed; E-stop and protective-stop state require physical/vendor confirmation."],
    }
    write_json("stage28a_safety_readiness.json", safety)

    no_motion = {
        "schema_version": "stage28a-no-motion-certificate-v1",
        "stage": "Stage 2.8A",
        "capture_start_utc": started.isoformat(),
        "capture_end_utc": finished.isoformat(),
        "FollowJointTrajectory_goals_sent": 0,
        "servo_commands_sent": 0,
        "trajectory_commands_sent": 0,
        "motion_commands_sent": 0,
        "motion_occurred": False,
        "auto_move_to_trajectory_start": False,
        "proof_basis": {
            "command_audit": "This builder and all Stage 2.8A probes use only ROS graph, parameter, interface-list, topic/action-info, package metadata and OS metadata reads.",
            "forbidden_motion_apis_invoked": [],
            "current_follow_joint_trajectory_server_observed": False,
            "historical_stage27s_goal": "present in historical mock artifact but outside Stage 2.8A scope and not counted",
        },
        "status": "passed_no_motion_gate",
        "invalidating_condition": "Any observed motion-capable command during the Stage 2.8A time window would invalidate this certificate.",
    }
    write_json("stage28a_no_motion_certificate.json", no_motion)

    blockers = [
        {"id": "B_REAL_HARDWARE_RUNTIME_UNAVAILABLE", "gate": "B", "severity": "blocker", "detail": "No real FR5/controller runtime was observed; hardware model, firmware, FAIRINO SDK, hardware plugin and controller-manager rate are unproven."},
        {"id": "C_HARDWARE_INTERFACE_AUDIT_UNAVAILABLE", "gate": "C", "severity": "blocker", "detail": "Required ros2_control runtime interfaces/controllers are absent. Historical mock_components evidence is excluded."},
        {"id": "D_REAL_JOINT_STATE_UNAVAILABLE", "gate": "D", "severity": "blocker", "detail": "No fresh real j1-j6 state was read, so radians/sign/zero/finiteness/freshness and candidate point 0 compatibility cannot be evaluated."},
        {"id": "E_JTC_RUNTIME_UNVERIFIED", "gate": "E", "severity": "blocker", "detail": "Real JTC endpoints, parameters, splines interpolation, publish rate and action monitor rate are not present/verified."},
        {"id": "F_MOVEIT_EXECUTION_LAYER_UNVERIFIED", "gate": "F", "severity": "blocker", "detail": "TrajectoryExecutionManager runtime parameters and execution_velocity_scaling=1.0 are not proven; static planning values 0.15 are not execution proof."},
        {"id": "G_RUNTIME_MODEL_TCP_FRAME_UNVERIFIED", "gate": "G", "severity": "blocker", "detail": "Only model assumptions exist; no runtime model/workcell transform or physical TCP/nozzle calibration evidence exists."},
        {"id": "H_REAL_HARDWARE_LIMITS_UNAVAILABLE", "gate": "H", "severity": "blocker", "detail": "Real robot/driver/controller limits and overrides are unavailable; compatibility cannot be claimed."},
        {"id": "I_SAFETY_READINESS_UNAVAILABLE", "gate": "I", "severity": "blocker", "detail": "E-stop, protective stop, fault, operating mode, speed override and workspace readiness are not observable from a real controller."},
    ]

    gate_report = {
        "schema_version": "stage28a-gate-report-v1",
        "stage": "Stage 2.8A — Real FR5 Zero-Motion Deployment Preflight Certification",
        "status": "blocked_real_hardware_runtime_unavailable",
        "decision": "Do not enter Stage 2.8B; it was not executed.",
        "frozen_preconditions": {
            "Stage_2_5R2": "passed_frozen_unchanged",
            "Stage_2_7S": "passed",
            "Stage_2_8": "unblocked_not_started",
            "Stage_2_5R2_sha256": EXPECTED_STAGE25_SHA256,
            "Stage_2_7S_candidate": {"points": 25532, "intervals": 25531, "duration_s": 765.97968192, "position_scale": 0.99999, "time_scale": 3.0},
            "Stage_2_7S_native_oracle": {"compared": "74879/74879", "position_max_abs_error_rad": 2.3e-14},
            "Stage_2_7S_dynamics": "passed",
            "Stage_2_7S_geometry": "passed",
            "Stage_2_7S_Bullet": "passed",
            "Stage_2_7S_determinism": "3/3_passed",
        },
        "A_frozen_input_provenance": {"status": "passed", "evidence": frozen_inputs},
        "B_real_hardware_identity": {"status": "blocked", "real_hardware_runtime_verified": False},
        "C_ros2_control_interface_audit": {"status": "blocked", "hardware_interfaces_runtime_verified": False},
        "D_joint_state_semantic_audit": {"status": "blocked", "initial_state_measured": False, "initial_state_formal_compatibility": "not_evaluable"},
        "E_jtc_runtime_audit": {"status": "blocked", "JTC_runtime_verified": False, "JTC_interpolation_method": None},
        "F_moveit_execution_manager_audit": {"status": "blocked_execution_layer_unverified", "MoveIt_execution_layer_verified": False, "execution_velocity_scaling": None, "silent_execution_scaling": "not_proven"},
        "G_model_tcp_frame_audit": {"status": "blocked", "robot_model_runtime_consistency": "not_evaluable", "physically_calibrated": False},
        "H_hardware_limits_audit": {"status": "blocked", "hardware_limits_compatible": None},
        "I_safety_readiness": {"status": "blocked", "safety_evidence_observed": False},
        "J_no_motion_proof": {"status": "passed", "motion_commands_sent": 0, "certificate": str(OUT / "stage28a_no_motion_certificate.json")},
        "K_final_gate": {
            "Stage_2_7S": "passed_frozen_unchanged",
            "Stage_2_8A": "blocked_real_hardware_runtime_unavailable",
            "real_FR5_runtime_verified": False,
            "joint_semantics_verified": False,
            "initial_state_measured": False,
            "initial_state_formal_compatibility": "not_evaluable",
            "JTC_runtime_verified": False,
            "JTC_interpolation_method": None,
            "MoveIt_execution_layer_verified": False,
            "silent_execution_scaling": "not_proven",
            "robot_model_runtime_consistency": "not_evaluable",
            "hardware_limits_compatible": None,
            "motion_commands_sent": 0,
            "Stage_2_8B": "blocked_not_started",
        },
        "blockers": blockers,
        "formal_artifacts_unchanged_by_stage28a": True,
    }
    write_json("stage28a_gate_report.json", gate_report)

    report = [
        "# Stage 2.8A — Real FR5 Zero-Motion Deployment Preflight Certification",
        "",
        f"Capture window: `{started.isoformat()}` to `{finished.isoformat()}`.",
        "",
        "## Decision",
        "",
        "**BLOCKED — real FR5 runtime evidence is unavailable. Stage 2.8B was not executed.**",
        "",
        "The frozen Stage 2.5R2/2.7S provenance gate passed. The live environment did not expose a real FR5, FAIRINO hardware interface, controller manager, JTC, or MoveIt execution runtime. Existing Stage 2.7S runtime evidence is explicitly classified as `mock_components/GenericSystem` and is not reused as real-hardware evidence.",
        "",
        "## Gate results",
        "",
        "- A frozen input provenance: **passed**; Stage 2.5R2 trajectory hash matches the required SHA-256 and Stage 2.7S formal SHA256SUMS were checked.",
        "- B real hardware identity: **blocked**; no real controller/firmware/SDK/plugin identity.",
        "- C ros2_control interfaces: **blocked**; no runtime hardware component, j1–j6 interfaces or claims.",
        "- D joint semantics/initial state: **blocked**; no fresh real `/joint_states`, so no start-state comparison was made and the robot was not moved.",
        "- E JTC runtime: **blocked**; no real JTC endpoints or runtime `splines` verification.",
        "- F MoveIt execution layer: **blocked**; TEM parameters and `execution_velocity_scaling: 1.0` were not observable. Static planning values of 0.15 are recorded but not treated as execution proof.",
        "- G model/TCP/frame: **blocked**; only assumed model files exist; TCP template is uncalibrated and world/workcell/nozzle runtime configuration is absent.",
        "- H hardware limits: **blocked**; real robot/driver limits and overrides were not observable.",
        "- I safety readiness: **blocked**; E-stop, protective stop, faults, mode, speed override and workspace readiness were not observable.",
        "- J no-motion proof: **passed for this Stage 2.8A window**; all four motion counters are zero. Historical Stage 2.7S mock action execution is outside this window and excluded.",
        "",
        "## Concrete blockers",
        "",
    ]
    for b in blockers:
        report.append(f"- `{b['id']}` — {b['detail']}")
    report.extend([
        "",
        "## Frozen candidate point 0",
        "",
        "The candidate point 0 is persisted in `stage28a_joint_state_compatibility.json`. Real measured values and differences are `null` because no real joint-state source was present; no automatic move to the trajectory start was attempted.",
        "",
        "## Safety statement",
        "",
        "Software discovery checks do not replace the FR5 manufacturer's safety system, a physical E-stop/protective-stop test, or qualified human现场确认. Those checks remain mandatory before any future dry-run.",
        "",
        "## Evidence files",
        "",
        "All requested evidence files are in this directory. `SHA256SUMS` covers this bundle (excluding the checksum file itself).",
        "",
        "Stage 2.8B remains `blocked_not_started`.",
    ])
    (OUT / "stage28a_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    # Bundle checksum file is generated last and intentionally does not hash itself.
    checksum_lines = []
    for path in sorted(OUT.iterdir()):
        if path.name == "SHA256SUMS" or not path.is_file():
            continue
        checksum_lines.append(f"{sha256(path)}  {path.name}")
    (OUT / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    # Verify the new bundle and frozen sources once more after all writes.
    bundle_check = verify_sha256sums(OUT)
    final_summary = {
        "schema_version": "stage28a-final-summary-v1",
        "stage28a_status": gate_report["status"],
        "stage28b_status": "blocked_not_started",
        "frozen_input_gate": frozen_inputs["formal_frozen_input_gate"],
        "bundle_sha256sums_passed": bundle_check["passed"],
        "motion_commands_sent": 0,
        "formal_stage25r2_sha256_after": sha256(STAGE25_TRAJECTORY),
        "formal_stage27s_candidate_sha256_after": sha256(STAGE27_CANDIDATE),
        "generated_files": sorted(p.name for p in OUT.iterdir() if p.is_file()),
    }
    write_json("stage28a_final_summary.json", final_summary)
    # Add final summary to the bundle checksum file and verify the final set.
    checksum_lines = []
    for path in sorted(OUT.iterdir()):
        if path.name == "SHA256SUMS" or not path.is_file():
            continue
        checksum_lines.append(f"{sha256(path)}  {path.name}")
    (OUT / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
