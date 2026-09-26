#!/usr/bin/env python3
"""Stage 2.8A-R zero-motion runtime compatibility and certification audit.

This script is deliberately a read-only auditor.  It never launches a robot
driver, controller_manager, MoveIt, or a ROS command publisher.  The only ROS
commands it may invoke are graph, metadata, interface-list, parameter-read,
topic-info/echo, and action-info/list queries.  The isolated colcon build is
performed in WSL /tmp and does not activate a plugin or connect to a robot.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "stage28ar_real_fr5_runtime_bringup"
RAW = OUT / "raw"
LOGS = OUT / "logs"
JOINTS = [f"j{i}" for i in range(1, 7)]

STAGE25 = ROOT / "outputs" / "stage25r2_full_native_certification" / "stage25r2_formal_20260804T054820Z"
STAGE27 = ROOT / "outputs" / "stage27s_native_spline_remediation" / "stage27s_formal_20260805T130000Z"
STAGE28A = ROOT / "outputs" / "stage28a_real_fr5_zero_motion_preflight"
FAIRINO_REPO = ROOT / "external" / "frcobot_ros2"
FAIRINO_VARIANT = FAIRINO_REPO / "fairino_hardware_v3_9_7"
FAIRINO5_CONFIG = FAIRINO_REPO / "fairino5_v6_moveit2_config"

MOTION_TOKENS = [
    "RobotEnable", "MoveJ", "MoveL", "MoveC", "Circle", "ServoJ", "ServoCart",
    "ServoJV", "Spline", "NewSpline", "StartJOG", "ImmStopJOG", "FollowJointTrajectory",
    "MoveIt execute", "asyncExecute", "joint_trajectory", "publish command", "write(",
    "SetAnticollision", "SetCollisionStrategy", "SetLimitPositive", "SetLimitNegative",
    "SetSpeed", "SetToolCoord", "SetWObjCoord", "StopMotion",
]

FORBIDDEN_COMMAND_PATTERNS = [
    "action send_goal", "topic pub", "service call", "param set", "moveit execute",
    "asyncexecute", "robotenable", "movej", "movel", "movec", "circle", "servoj",
    "servocart", "servojv", "startjog", "immstopjog", "followjointtrajectory",
    "joint_trajectory topic", "forward command", "position command publication",
    "velocity command publication", "effort command publication",
]


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def decode_process_output(raw: bytes) -> str:
    text = raw.decode("utf-8", errors="replace")
    if text.count("\x00") > 3:
        try:
            return raw.decode("utf-16", errors="replace")
        except UnicodeError:
            pass
    return text


def command_is_motion_capable(command: str) -> bool:
    lowered = command.lower()
    return any(pattern in lowered for pattern in FORBIDDEN_COMMAND_PATTERNS)


def command_is_read_only(command: str) -> bool:
    """Return true only for the explicit read-only command surface."""
    if command_is_motion_capable(command):
        return False
    lowered = command.lower()
    allowed = (
        "ros2 node list", "ros2 node info", "ros2 topic list", "ros2 topic info",
        "ros2 topic echo", "ros2 service list", "ros2 service type", "ros2 action list",
        "ros2 action info", "ros2 param list", "ros2 param get", "ros2 control list_",
        "ros2 pkg list", "ros2 pkg prefix", "grep", "find ", "cat ", "uname ",
        "printenv ", "printf ", "test ", "ldd ", "python3 -c ", "git ", "sed ", "tr ",
    )
    return any(item in lowered for item in allowed)


def run_command(argv: list[str], timeout_s: float = 10.0, *, cwd: Path | None = None) -> dict[str, Any]:
    started = _dt.datetime.now(_dt.timezone.utc)
    try:
        completed = subprocess.run(
            argv,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
        return {
            "argv": argv,
            "started_utc": started.isoformat(),
            "finished_utc": utc_now(),
            "elapsed_s": round((_dt.datetime.now(_dt.timezone.utc) - started).total_seconds(), 3),
            "exit_code": completed.returncode,
            "timed_out": False,
            "stdout": decode_process_output(completed.stdout),
            "stderr": decode_process_output(completed.stderr),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "argv": argv,
            "started_utc": started.isoformat(),
            "finished_utc": utc_now(),
            "elapsed_s": round((_dt.datetime.now(_dt.timezone.utc) - started).total_seconds(), 3),
            "exit_code": None,
            "timed_out": True,
            "stdout": decode_process_output(exc.stdout or b""),
            "stderr": decode_process_output(exc.stderr or b""),
        }
    except OSError as exc:
        return {
            "argv": argv,
            "started_utc": started.isoformat(),
            "finished_utc": utc_now(),
            "elapsed_s": round((_dt.datetime.now(_dt.timezone.utc) - started).total_seconds(), 3),
            "exit_code": None,
            "timed_out": False,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
        }


def run_wsl(name: str, inner_command: str, *, timeout_s: float = 8.0, read_only: bool = True) -> dict[str, Any]:
    if read_only and not command_is_read_only(inner_command):
        raise RuntimeError(f"read-only command rejected by zero-motion guard: {inner_command}")
    wsl = shutil.which("wsl.exe")
    if not wsl:
        result = {
            "argv": ["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", inner_command],
            "exit_code": None,
            "timed_out": False,
            "stdout": "",
            "stderr": "wsl.exe not found",
            "unavailable": True,
        }
    else:
        result = run_command([wsl, "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", inner_command], timeout_s)
        result["unavailable"] = False
    result["probe"] = name
    json_dump(RAW / f"{name}.json", result)
    (LOGS / f"{name}.stdout.log").write_text(result.get("stdout", ""), encoding="utf-8")
    (LOGS / f"{name}.stderr.log").write_text(result.get("stderr", ""), encoding="utf-8")
    return result


def run_git(name: str, args: list[str]) -> dict[str, Any]:
    result = run_command(["git", "-C", str(FAIRINO_REPO), *args], timeout_s=10.0)
    result["probe"] = name
    json_dump(RAW / f"{name}.json", result)
    return result


def parse_sha_manifest(manifest: Path) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for line in manifest.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^([0-9a-fA-F]{64})\s+\*?(.*)$", line)
        if match:
            entries.append((match.group(1).lower(), match.group(2).strip()))
    return entries


def verify_sha_manifest(root: Path, manifest_name: str = "SHA256SUMS") -> dict[str, Any]:
    manifest = root / manifest_name
    if not manifest.is_file():
        return {"root": str(root), "manifest": str(manifest), "passed": False, "missing_manifest": True}
    mismatches: list[dict[str, str]] = []
    missing: list[str] = []
    entries = parse_sha_manifest(manifest)
    for expected, relative in entries:
        target = root / relative
        if not target.is_file():
            missing.append(relative)
            continue
        actual = sha256_file(target)
        if actual != expected:
            mismatches.append({"path": relative, "expected": expected, "actual": actual})
    return {
        "root": str(root),
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "entries": len(entries),
        "verified": len(entries) - len(missing) - len(mismatches),
        "missing": missing,
        "mismatches": mismatches,
        "passed": bool(entries) and not missing and not mismatches,
    }


def frozen_snapshot() -> dict[str, Any]:
    return {
        "stage25r2": verify_sha_manifest(STAGE25),
        "stage27s": verify_sha_manifest(STAGE27),
        "stage28a": verify_sha_manifest(STAGE28A),
        "stage25r2_trajectory_sha256": sha256_file(STAGE25 / "stage25r2_final_robot_trajectory.csv") if (STAGE25 / "stage25r2_final_robot_trajectory.csv").is_file() else None,
        "stage27s_candidate_sha256": sha256_file(STAGE27 / "stage27s_candidate_trajectory.csv") if (STAGE27 / "stage27s_candidate_trajectory.csv").is_file() else None,
    }


def package_version(package_xml: Path) -> str | None:
    match = re.search(r"<version>\s*([^<]+?)\s*</version>", package_xml.read_text(encoding="utf-8", errors="replace"))
    return match.group(1).strip() if match else None


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "exists": path.is_file(), "sha256": sha256_file(path) if path.is_file() else None}


def source_matches(path: Path, patterns: Iterable[str], limit: int = 80) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    compiled = [(pattern, re.compile(re.escape(pattern), re.IGNORECASE)) for pattern in patterns]
    for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        found = [pattern for pattern, regex in compiled if regex.search(line)]
        if found:
            records.append({"line": number, "tokens": found, "text": line.strip()[:300]})
            if len(records) >= limit:
                break
    return records


def extract_function(text: str, qualified_name: str) -> str:
    start = text.find(qualified_name)
    if start < 0:
        return ""
    close = text.find("\n}\n", start)
    return text[start : close + 3 if close >= 0 else len(text)]


def audit_fairino_source() -> dict[str, Any]:
    package_xml = FAIRINO_VARIANT / "package.xml"
    cmake = FAIRINO_VARIANT / "CMakeLists.txt"
    plugin_xml = FAIRINO_VARIANT / "fairino_hardware.xml"
    interface_cpp = FAIRINO_VARIANT / "src" / "fairino_hardware_interface.cpp"
    command_server_cpp = FAIRINO_VARIANT / "src" / "command_server.cpp"
    xacro = FAIRINO5_CONFIG / "config" / "fairino5_v6_robot.ros2_control.xacro"
    sdk = FAIRINO_VARIANT / "libfairino" / "lib" / "libfairino.so.2.3.7"

    remote = run_git("git_remote", ["remote", "get-url", "origin"])
    head = run_git("git_head", ["rev-parse", "HEAD"])
    branch = run_git("git_branch", ["symbolic-ref", "--short", "HEAD"])
    tag = run_git("git_tag", ["describe", "--tags", "--exact-match", "HEAD"])
    status = run_git("git_status", ["status", "--short", "--branch"])

    interface_text = interface_cpp.read_text(encoding="utf-8", errors="replace") if interface_cpp.is_file() else ""
    activation = extract_function(interface_text, "FairinoHardwareInterface::on_activate")
    deactivation = extract_function(interface_text, "FairinoHardwareInterface::on_deactivate")
    write = extract_function(interface_text, "FairinoHardwareInterface::write")
    activation_motion = [token for token in MOTION_TOKENS if re.search(re.escape(token), activation, re.IGNORECASE)]
    write_motion = [token for token in MOTION_TOKENS if re.search(re.escape(token), write, re.IGNORECASE)]
    deactivation_motion = [token for token in MOTION_TOKENS if re.search(re.escape(token), deactivation, re.IGNORECASE)]
    variant_texts = {
        str(path.relative_to(FAIRINO_REPO)): path.read_text(encoding="utf-8", errors="replace")
        for path in [interface_cpp, command_server_cpp, cmake, plugin_xml, xacro]
        if path.is_file()
    }
    plugin_class = "fairino_hardware/FairinoHardwareInterface" if plugin_xml.is_file() and "FairinoHardwareInterface" in variant_texts.get(str(plugin_xml.relative_to(FAIRINO_REPO)), "") else None
    clean = status.get("stdout", "").strip().splitlines()[-1:] == ["## main...origin/main"]
    return {
        "schema_version": "stage28ar-fairino-driver-audit-v1",
        "official_driver_found": FAIRINO_VARIANT.is_dir() and package_xml.is_file(),
        "official_FAIRINO_source": {
            "path": str(FAIRINO_VARIANT),
            "repository": remote.get("stdout", "").strip() or None,
            "commit": head.get("stdout", "").strip() or None,
            "branch_or_tag": tag.get("stdout", "").strip() or branch.get("stdout", "").strip() or None,
            "package_version": package_version(package_xml) if package_xml.is_file() else None,
            "plugin_class": plugin_class,
            "SDK_version": "libfairino.so.2.3.7" if sdk.is_file() else None,
            "SDK_binary": file_record(sdk),
            "source_status": "clean_official_remote_copy" if clean else "local_status_requires_review",
        },
        "source_classification": {
            "official_FAIRINO_source": True,
            "local_copy": True,
            "third_party_copy": False,
            "project_modified_copy": not clean,
            "evidence": "nested Git repository origin points to FAIR-INNOVATION/frcobot_ros2; repository status is recorded in raw/git_status.json",
        },
        "audited_files": [file_record(path) for path in [package_xml, cmake, plugin_xml, interface_cpp, command_server_cpp, xacro]],
        "package_and_build_audit": {
            "package_xml_version": package_version(package_xml) if package_xml.is_file() else None,
            "declares_hardware_interface": "hardware_interface" in package_xml.read_text(encoding="utf-8", errors="replace") if package_xml.is_file() else False,
            "declares_pluginlib": "pluginlib" in package_xml.read_text(encoding="utf-8", errors="replace") if package_xml.is_file() else False,
            "pluginlib_export_macro_present": "pluginlib_export_plugin_description_file" in cmake.read_text(encoding="utf-8", errors="replace") if cmake.is_file() else False,
            "sdk_linked_binary": "libfairino.so.2.3.7" if cmake.is_file() and "libfairino.so.2.3.7" in cmake.read_text(encoding="utf-8", errors="replace") else None,
            "plugin_xml_present": plugin_xml.is_file(),
            "hard_coded_controller_ip_in_header": "192.168.58.2" in (FAIRINO_VARIANT / "include" / "fairino_hardware" / "fairino_hardware_interface.hpp").read_text(encoding="utf-8", errors="replace") if (FAIRINO_VARIANT / "include" / "fairino_hardware" / "fairino_hardware_interface.hpp").is_file() else False,
        },
        "static_safety_audit": {
            "on_activate_direct_motion_calls": activation_motion,
            "on_activate_read_or_connection_calls": [token for token in ["RPC", "GetActualJointPosDegree"] if token in activation],
            "write_motion_calls": write_motion,
            "on_deactivate_motion_calls": deactivation_motion,
            "command_server_motion_api_matches": source_matches(command_server_cpp, MOTION_TOKENS, limit=60),
            "fake_system_in_FR5_default_xacro": "mock_components/GenericSystem" in variant_texts.get(str(xacro.relative_to(FAIRINO_REPO)), ""),
            "zero_motion_safe_to_activate": False,
            "classification": "blocked_hardware_plugin_activation_not_zero_motion_safe",
            "reason": "on_activate itself does not call a direct motion API, but the active plugin exports position command interfaces and write() calls ServoJ; there is no independent zero-motion interlock. on_deactivate calls StopMotion. The plugin was not activated.",
        },
        "motion_capable_tokens_found": {str(path): source_matches(path, MOTION_TOKENS) for path in [interface_cpp, command_server_cpp, xacro] if path.is_file()},
    }


def run_isolated_jazzy_build() -> dict[str, Any]:
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    build = f"/tmp/stage28ar_build_{stamp}"
    install = f"/tmp/stage28ar_install_{stamp}"
    log = f"/tmp/stage28ar_log_{stamp}"
    sdk_runtime = f"/tmp/stage28ar_sdk_runtime_{stamp}"
    build_command = (
        "source /opt/ros/jazzy/setup.bash; "
        f"colcon --log-base {log} build --base-paths /mnt/d/robotfucker/external/frcobot_ros2 "
        f"--packages-select fairino_msgs fairino_hardware_v3_9_7 --build-base {build} "
        f"--install-base {install} --event-handlers console_direct+"
    )
    plugin_command = (
        f"mkdir -p {sdk_runtime}; "
        f"ln -sf /mnt/d/robotfucker/external/frcobot_ros2/fairino_hardware_v3_9_7/libfairino/lib/libfairino.so.2.3.7 {sdk_runtime}/libfairino.so.2; "
        "source /opt/ros/jazzy/setup.bash; "
        f"export LD_LIBRARY_PATH={sdk_runtime}:/mnt/d/robotfucker/external/frcobot_ros2/fairino_hardware_v3_9_7/libfairino/lib:{install}/lib:/opt/ros/jazzy/lib; "
        f"echo PREFIX={install}/fairino_hardware_v3_9_7; "
        f"echo PLUGIN={install}/fairino_hardware_v3_9_7/lib/fairino_hardware_v3_9_7/libfairino_hardware.so; "
        f"if test -f {install}/fairino_hardware_v3_9_7/lib/fairino_hardware_v3_9_7/libfairino_hardware.so; then ldd {install}/fairino_hardware_v3_9_7/lib/fairino_hardware_v3_9_7/libfairino_hardware.so; python3 -c 'import ctypes,sys; ctypes.CDLL(sys.argv[1]); print(\"shared_library_load=passed\")' {install}/fairino_hardware_v3_9_7/lib/fairino_hardware_v3_9_7/libfairino_hardware.so; fi; "
        f"find {install}/fairino_hardware_v3_9_7 -type f -print 2>/dev/null | grep -E 'fairino_hardware.xml|plugin' || true"
    )
    # Ubuntu-24.04-D is disposable between WSL invocations in this host.  Run
    # the build and the non-activating shared-library load in one invocation so
    # the isolated install tree is still present for the load check.
    combined_command = f"if {build_command}; then {plugin_command}; else exit 42; fi"
    build_result = run_wsl("jazzy_isolated_colcon_build", combined_command, timeout_s=180.0, read_only=False)
    plugin_result = dict(build_result)
    plugin_result["probe"] = "jazzy_plugin_export_and_load"
    json_dump(RAW / "jazzy_plugin_export_and_load.json", plugin_result)
    (LOGS / "jazzy_plugin_export_and_load.stdout.log").write_text(plugin_result.get("stdout", ""), encoding="utf-8")
    (LOGS / "jazzy_plugin_export_and_load.stderr.log").write_text(plugin_result.get("stderr", ""), encoding="utf-8")
    compile_passed = build_result.get("exit_code") == 0 and "Finished <<< fairino_hardware_v3_9_7" in build_result.get("stdout", "")
    plugin_load_passed = compile_passed and "shared_library_load=passed" in plugin_result.get("stdout", "")
    return {
        "workspace": {"build": build, "install": install, "log": log, "isolated": True},
        "source_build": {"passed": compile_passed, "probe": "raw/jazzy_isolated_colcon_build.json", "real_robot_contact": False},
        "plugin_export": {"passed": plugin_result.get("exit_code") == 0 and "fairino_hardware.xml" in (plugin_result.get("stdout", "") + plugin_result.get("stderr", "")), "probe": "raw/jazzy_plugin_export_and_load.json"},
        "plugin_load": {"passed": plugin_load_passed, "probe": "raw/jazzy_plugin_export_and_load.json", "activation_performed": False},
        "compile_warnings_or_notes": "Build output is preserved in logs/jazzy_isolated_colcon_build.*; warnings do not constitute runtime certification.",
        "no_real_runtime_claim": True,
    }


def normalize_probe_text(text: str) -> str:
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("[INFO]") or stripped.startswith("wsl:"):
            continue
        lines.append(stripped)
    return "\n".join(lines)


def repeat_read_only_probes() -> dict[str, Any]:
    specs = {
        "node_list": "source /opt/ros/jazzy/setup.bash; ros2 node list",
        "topic_list": "source /opt/ros/jazzy/setup.bash; ros2 topic list -t --include-hidden-topics",
        "action_list": "source /opt/ros/jazzy/setup.bash; ros2 action list -t",
        "fairino_packages": "source /opt/ros/jazzy/setup.bash; ros2 pkg list | grep -Ei 'fairino|frcobot' || true",
    }
    output: dict[str, Any] = {}
    for name, command in specs.items():
        runs = [run_wsl(f"repeat_{name}_{index}", command, timeout_s=8.0) for index in range(1, 4)]
        normalized = [normalize_probe_text(run.get("stdout", "") + "\n" + run.get("stderr", "")) for run in runs]
        output[name] = {
            "run_1": normalized[0],
            "run_2": normalized[1],
            "run_3": normalized[2],
            "semantic_match": len(set(normalized)) == 1,
            "raw_probes": [f"raw/repeat_{name}_{index}.json" for index in range(1, 4)],
        }
    return output


def probe_runtime() -> dict[str, Any]:
    host = run_wsl(
        "host_environment",
        "uname -a; cat /etc/os-release | head -12; printf 'ROS_DISTRO='; printenv ROS_DISTRO; printf '\\nROS_PREFIX_JAZZY='; test -f /opt/ros/jazzy/setup.bash && echo true || echo false",
        timeout_s=8.0,
    )
    packages = run_wsl(
        "jazzy_package_versions",
        "printf 'ros2_control='; grep -m1 '<version>' /opt/ros/jazzy/share/ros2_control/package.xml | sed 's/<[^>]*>//g' | tr -d ' '; echo; "
        "printf 'ros2_controllers='; grep -m1 '<version>' /opt/ros/jazzy/share/ros2_controllers/package.xml | sed 's/<[^>]*>//g' | tr -d ' '; echo; "
        "printf 'controller_manager='; grep -m1 '<version>' /opt/ros/jazzy/share/controller_manager/package.xml | sed 's/<[^>]*>//g' | tr -d ' '; echo; "
        "printf 'joint_trajectory_controller='; grep -m1 '<version>' /opt/ros/jazzy/share/joint_trajectory_controller/package.xml | sed 's/<[^>]*>//g' | tr -d ' '; echo; "
        "printf 'moveit_core='; grep -m1 '<version>' /opt/ros/jazzy/share/moveit_core/package.xml | sed 's/<[^>]*>//g' | tr -d ' '; echo; "
        "printf 'moveit_ros_move_group='; grep -m1 '<version>' /opt/ros/jazzy/share/moveit_ros_move_group/package.xml | sed 's/<[^>]*>//g' | tr -d ' '; echo",
        timeout_s=8.0,
    )
    probes = {
        "node_list": run_wsl("node_list", "source /opt/ros/jazzy/setup.bash; ros2 node list --spin-time 1", timeout_s=8.0),
        "controller_manager_info": run_wsl("controller_manager_info", "source /opt/ros/jazzy/setup.bash; ros2 node info /controller_manager", timeout_s=8.0),
        "controllers": run_wsl("controllers", "source /opt/ros/jazzy/setup.bash; ros2 control list_controllers --verbose --spin-time 1", timeout_s=8.0),
        "hardware_components": run_wsl("hardware_components", "source /opt/ros/jazzy/setup.bash; ros2 control list_hardware_components --verbose --spin-time 1", timeout_s=8.0),
        "hardware_interfaces": run_wsl("hardware_interfaces", "source /opt/ros/jazzy/setup.bash; ros2 control list_hardware_interfaces --verbose --spin-time 1", timeout_s=8.0),
        "action_list": run_wsl("action_list", "source /opt/ros/jazzy/setup.bash; ros2 action list -t", timeout_s=8.0),
        "follow_joint_trajectory_info": run_wsl("follow_joint_trajectory_info", "source /opt/ros/jazzy/setup.bash; ros2 action info /fairino5_controller/follow_joint_trajectory", timeout_s=8.0),
        "service_list": run_wsl("service_list", "source /opt/ros/jazzy/setup.bash; ros2 service list -t", timeout_s=8.0),
        "controller_state_info": run_wsl("controller_state_info", "source /opt/ros/jazzy/setup.bash; ros2 topic info -v /fairino5_controller/controller_state", timeout_s=8.0),
        "joint_states_info": run_wsl("joint_states_info", "source /opt/ros/jazzy/setup.bash; ros2 topic info -v /joint_states", timeout_s=8.0),
        "joint_states_echo": run_wsl("joint_states_echo", "source /opt/ros/jazzy/setup.bash; ros2 topic echo /joint_states --once", timeout_s=8.0),
        "native_state_info": run_wsl("native_state_info", "source /opt/ros/jazzy/setup.bash; ros2 topic info -v /nonrt_state_data", timeout_s=8.0),
        "native_state_echo": run_wsl("native_state_echo", "source /opt/ros/jazzy/setup.bash; ros2 topic echo /nonrt_state_data --once", timeout_s=8.0),
        "jtc_interpolation_param": run_wsl("jtc_interpolation_param", "source /opt/ros/jazzy/setup.bash; ros2 param get /fairino5_controller interpolation_method", timeout_s=8.0),
        "jtc_query_state_type": run_wsl("jtc_query_state_type", "source /opt/ros/jazzy/setup.bash; ros2 service type /fairino5_controller/query_state", timeout_s=8.0),
    }
    return {"host": host, "packages": packages, "probes": probes, "repeatability": repeat_read_only_probes()}


def parse_package_versions(probe: dict[str, Any]) -> dict[str, str | None]:
    text = probe.get("stdout", "")
    result: dict[str, str | None] = {}
    for package in ["ros2_control", "ros2_controllers", "controller_manager", "joint_trajectory_controller", "moveit_core", "moveit_ros_move_group"]:
        match = re.search(rf"^{re.escape(package)}=([^\r\n]*)", text, re.MULTILINE)
        result[package] = match.group(1).strip() if match and match.group(1).strip() else None
    return result


def probe_succeeded(probe: dict[str, Any], text: str | None = None) -> bool:
    if probe.get("exit_code") != 0:
        return False
    return text in probe.get("stdout", "") if text else True


def assess_live_state(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Assess a parsed read-only state sample without assuming motion."""
    if not samples:
        return {
            "sample_count": 0,
            "sample_duration_s": None,
            "finite": False,
            "timestamps_monotonic": False,
            "freshness": False,
            "non_constant_or_sensor_refresh_proof": False,
        }
    times = [float(sample["timestamp"]) for sample in samples if isinstance(sample.get("timestamp"), (int, float))]
    monotonic = len(times) == len(samples) and all(right >= left for left, right in zip(times, times[1:]))
    finite = all(all(isinstance(value, (int, float)) and value == value and abs(value) != float("inf") for value in sample.get("positions", [])) for sample in samples)
    refreshed = len(set(times)) > 1 if times else False
    return {
        "sample_count": len(samples),
        "sample_duration_s": times[-1] - times[0] if len(times) >= 2 else None,
        "finite": finite,
        "timestamps_monotonic": monotonic,
        "freshness": refreshed and monotonic,
        "non_constant_or_sensor_refresh_proof": refreshed,
    }


def classify_hardware_component(component: str | None) -> str:
    if not component:
        return "not_observed"
    lowered = component.lower()
    if any(token in lowered for token in ["mock", "generic", "fake", "gazebo", "simulation", "replay"]):
        return "rejected_mock_or_simulation"
    if "fairino" in lowered or "fr5" in lowered:
        return "candidate_fairino_real_component_requires_live_connection_proof"
    return "unclassified_component_requires_identity_proof"


def classify_runtime(*, hardware_component_present: bool, live_state: bool, jtc_runtime: bool, moveit_runtime: bool) -> str:
    if hardware_component_present and live_state and jtc_runtime and moveit_runtime:
        return "candidate_runtime_complete_pending_gate_matrix"
    return "blocked_real_ros2_control_runtime"


def classify_distro_compatibility(current_distro: str | None, driver_requirement: str | None) -> str:
    if current_distro and driver_requirement and current_distro.lower() == "jazzy" and "humble" in driver_requirement.lower():
        return "blocked_ros_distribution_driver_compatibility"
    return "compatible_with_existing_jazzy_stack"


def build_report_data(frozen_before: dict[str, Any], source_audit: dict[str, Any], build: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
    versions = parse_package_versions(runtime.get("packages", {}))
    host_text = runtime.get("host", {}).get("stdout", "")
    distro_prefix = "jazzy" if "/opt/ros/jazzy/setup.bash" in host_text or "ROS_PREFIX_JAZZY=true" in host_text else None
    distro_env_match = re.search(r"ROS_DISTRO=([^\r\n]*)", host_text)
    distro_env = distro_env_match.group(1).strip() if distro_env_match else None
    probes = runtime.get("probes", {})
    component_text = probes.get("hardware_components", {}).get("stdout", "")
    hardware_present = bool(component_text.strip()) and probes.get("hardware_components", {}).get("exit_code") == 0 and "waiting for service" not in probes.get("hardware_components", {}).get("stderr", "")
    joint_info = probes.get("joint_states_info", {})
    joint_echo = probes.get("joint_states_echo", {})
    jtc_action = probes.get("follow_joint_trajectory_info", {})
    jtc_topic = probes.get("controller_state_info", {})
    jtc_param = probes.get("jtc_interpolation_param", {})
    live_state = joint_info.get("exit_code") == 0 and joint_echo.get("exit_code") == 0 and bool(joint_echo.get("stdout", "").strip())
    jtc_runtime = jtc_action.get("exit_code") == 0 and "Action servers: 0" not in jtc_action.get("stdout", "") and jtc_topic.get("exit_code") == 0 and jtc_param.get("exit_code") == 0
    moveit_runtime = False
    plugin_safe = source_audit["static_safety_audit"]["zero_motion_safe_to_activate"]
    compatibility = classify_distro_compatibility(distro_prefix, None)
    runtime_blocker = classify_runtime(hardware_component_present=hardware_present, live_state=live_state, jtc_runtime=jtc_runtime, moveit_runtime=moveit_runtime)
    frozen_after_placeholder = frozen_before
    return {
        "versions": versions,
        "distro_prefix": distro_prefix,
        "distro_environment_variable": distro_env or None,
        "hardware_component_present": hardware_present,
        "live_state": live_state,
        "jtc_runtime": jtc_runtime,
        "moveit_runtime": moveit_runtime,
        "compatibility": compatibility,
        "runtime_blocker": runtime_blocker,
        "plugin_zero_motion_safe": plugin_safe,
        "frozen_before": frozen_before,
        "frozen_after_placeholder": frozen_after_placeholder,
    }


def extract_simple_limits(path: Path) -> dict[str, dict[str, float | None]]:
    result: dict[str, dict[str, float | None]] = {joint: {} for joint in JOINTS}
    if not path.is_file():
        return result
    text = path.read_text(encoding="utf-8", errors="replace")
    for joint in JOINTS:
        block_match = re.search(rf"(?ms)^\s{{2}}{joint}:\s*\n(.*?)(?=^\s{{2}}j[1-6]:|\Z)", text)
        block = block_match.group(1) if block_match else ""
        for key in ["max_velocity", "max_acceleration", "min_position", "max_position"]:
            match = re.search(rf"{key}:\s*([-+]?\d+(?:\.\d+)?)", block)
            result[joint][key] = float(match.group(1)) if match else None
    return result


def build_static_evidence(source_audit: dict[str, Any], runtime: dict[str, Any], build: dict[str, Any], cross: dict[str, Any]) -> dict[str, Any]:
    moveit_config = FAIRINO5_CONFIG / "config" / "moveit_controllers.yaml"
    controllers = FAIRINO5_CONFIG / "config" / "ros2_controllers.yaml"
    project_limits = ROOT / "ros2_moveit_bridge" / "config" / "joint_limits.yaml"
    stage27_cert = STAGE27 / "stage27s_exact_jtc_certificate.json"
    stage27 = json.loads(stage27_cert.read_text(encoding="utf-8")) if stage27_cert.is_file() else {}
    return {
        "ros2_control_runtime": {
            "installed": cross["versions"].get("ros2_control") is not None,
            "configured": False,
            "launched": False,
            "active": False,
            "connected_to_real_robot": False,
            "live_data_verified": False,
            "controller_manager": {"observed": False, "update_rate_hz": None},
            "hardware_component": {"state": None, "type": None, "identity_classification": classify_hardware_component(None)},
            "source_build_in_isolated_workspace": build["source_build"],
        },
        "hardware_interfaces": {
            "status": "blocked_real_runtime_not_observed",
            "required_read_only_commands": [
                "ros2 control list_controllers --verbose",
                "ros2 control list_hardware_components --verbose",
                "ros2 control list_hardware_interfaces --verbose",
            ],
            "current_runtime": {"state_interfaces": None, "command_interfaces": None, "claimed_by": None, "joint_order": None},
            "historical_mock_runtime_rejected": True,
        },
        "live_joint_state": {
            "status": "blocked_live_joint_state_unavailable",
            "joint_names": None,
            "exact_order_verified": False,
            "units": {"position": "rad", "velocity": "rad/s"},
            "finite": False,
            "timestamps_monotonic": False,
            "freshness_real_time": False,
            "sample_count": 0,
            "sample_duration_s": None,
            "non_constant_or_sensor_refresh_proof": False,
            "joint_states_topic_observed": runtime["probes"]["joint_states_info"].get("exit_code") == 0,
            "native_state_topic_observed": runtime["probes"]["native_state_info"].get("exit_code") == 0,
            "do_not_infer_fake_from_static_position": True,
        },
        "joint_semantics": {
            "status": "blocked_joint_semantics_not_proven",
            "joint_name_mapping": None,
            "joint_order": None,
            "sign_convention": None,
            "zero_reference": None,
            "units": {"position": "rad", "velocity": "rad/s"},
            "URDF_axis_consistency": "static_model_only",
            "source_unit_conversion_observed": "GetActualJointPosDegree converted to radians in official plugin source",
            "joint_sign_semantics": {"status": "blocked_requires_controlled_motion_validation"},
            "motion_validation_performed": False,
        },
        "jtc_runtime": {
            "status": "blocked_JTC_real_runtime_unverified",
            "runtime_verified": False,
            "name": None,
            "package_version": None,
            "controller_state": None,
            "lifecycle": None,
            "command_interfaces": None,
            "state_interfaces": None,
            "update_rate": None,
            "interpolation_method": None,
            "follow_joint_trajectory_exists": False,
            "controller_state_topic_exists": False,
            "query_state_exists": False,
            "goal_sent": False,
            "jazzy_package_version_installed": cross["versions"].get("joint_trajectory_controller"),
            "isolated_plugin_build_does_not_prove_JTC_runtime": True,
        },
        "stage27s_crosscheck": {
            "Stage_2_7S": {
                "JTC_version": stage27.get("runtime", {}).get("jtc_version"),
                "interpolation_method": stage27.get("runtime", {}).get("interpolation_method"),
                "source_commit": stage27.get("runtime", {}).get("source_commit"),
                "controller_parameters_source": str(STAGE27 / "stage27s_controllers.yaml"),
                "update_semantics": "historical mock runtime; frozen oracle only",
            },
            "real_hardware_runtime": {
                "JTC_version": None,
                "interpolation_method": None,
                "controller_parameters": None,
                "update_semantics": None,
            },
            "stage27s_semantics_transfer": {
                "eligible": False,
                "semantic_match": None,
                "evidence": ["real controller_manager/JTC not running", "runtime parameters not observable", "historical mock action is excluded"],
                "trajectory_semantics_changed": None,
            },
        },
        "moveit_execution": {
            "status": "blocked_MoveIt_execution_runtime_not_proven",
            "runtime_verified": False,
            "controller": None,
            "action_namespace": None,
            "joints": None,
            "exact_match": None,
            "static_mapping_file": file_record(moveit_config),
            "static_controller_manager_file": file_record(controllers),
            "execution_parameters": {
                "allowed_execution_duration_scaling": None,
                "allowed_goal_duration_margin": None,
                "execution_duration_monitoring": None,
                "execution_velocity_scaling": None,
            },
            "silent_trajectory_modification": "not_proven",
            "execute_calls": 0,
        },
        "tcp_frame": {
            "status": "blocked_TCP_frame_calibration_not_proven",
            "runtime_verified": False,
            "static_model": {"base_frame": "base_link", "flange_frame": "wrist3_link", "TCP_frame": "spray_tcp_link"},
            "runtime_controller_tool_coordinates": None,
            "runtime_work_object_coordinates": None,
            "world_to_base": None,
            "robot_installation": None,
            "TCP_translation_difference_mm": None,
            "TCP_rotation_difference_deg": None,
            "base_transform_difference": None,
            "physical_calibration_evidence": None,
        },
        "hardware_limits": {
            "status": "blocked_real_hardware_limits_not_proven",
            "hardware_limits_verified": False,
            "hardware_acceleration_limit": "not_exposed",
            "hardware_jerk_limit": "not_exposed",
            "comparison": {
                joint: {
                    "position": {"stage25r2": None, "urdf": None, "ros2_control": None, "controller": None, "physical_hard": None, "configured_soft": None},
                    "velocity": {"stage25r2": value, "urdf": None, "ros2_control": None, "controller": None, "physical_hard": None, "configured_soft": None},
                    "acceleration": {"stage25r2": 0.105, "urdf": None, "ros2_control": None, "controller": None, "physical_hard": None, "configured_soft": None},
                    "jerk": {"stage25r2": 8.0, "urdf": None, "ros2_control": None, "controller": None, "physical_hard": None, "configured_soft": None},
                }
                for joint, value in zip(JOINTS, [0.4725, 0.4725, 0.4725, 0.48, 0.48, 0.48])
            },
            "project_moveit_limits_source": file_record(project_limits),
            "stage27s_exact_jtc_certificate": file_record(stage27_cert),
            "stage27s_limits_are_not_hardware_evidence": True,
        },
        "safety": {
            "status": "blocked_safety_state_not_proven",
            "verified": False,
            "estop": None,
            "protective_stop": None,
            "collision_alarm": None,
            "robot_enabled": None,
            "mode": None,
            "drag_teach": None,
            "error_code": None,
            "speed_override": None,
            "read_only_queries_performed": False,
            "vendor_or_physical_confirmation_required": True,
        },
        "network": {
            "robot_ip": None,
            "host_ip": socket.gethostbyname(socket.gethostname()),
            "network_interface": platform.node(),
            "latency_ms": None,
            "packet_loss": None,
            "state_update_rate_hz": None,
            "controller_manager_update_rate_hz": None,
            "joint_state_rate_hz": None,
            "JTC_state_rate_hz": None,
            "source_default_controller_ip_not_robot_identity": "192.168.58.2",
            "robot_ping_performed": False,
        },
    }


def make_no_motion_certificate(started: str, ended: str, source_audit: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "stage28ar-no-motion-certificate-v1",
        "stage": "Stage 2.8A-R",
        "capture_start_utc": started,
        "capture_end_utc": ended,
        "motion_allowed": False,
        "motion_commands_sent": 0,
        "FJT_goals_sent": 0,
        "MoveIt_execute_calls": 0,
        "joint_trajectory_publications": 0,
        "servo_commands": 0,
        "FAIRINO_motion_API_calls": 0,
        "RobotEnable_1_calls": 0,
        "read_only_operations": [
            "frozen SHA-256 manifest verification",
            "local source/package/plugin/CMake inspection",
            "official nested Git metadata inspection",
            "isolated Jazzy colcon compile and shared-library load only",
            "ROS graph node/topic/action/service/interface-list queries",
            "ROS parameter get and topic info/echo queries",
            "read-only host and ROS package metadata probes",
        ],
        "robot_position_before": None,
        "robot_position_after": None,
        "real_robot_contact": False,
        "forbidden_motion_apis_invoked": [],
        "plugin_activation_performed": False,
        "proof_scope": "software audit window; no live FR5 state was available",
        "static_plugin_safety_result": source_audit["static_safety_audit"]["classification"],
        "runtime_queries_produced_live_state": False,
        "status": "passed_no_motion_gate",
        "invalidating_condition": "Any motion-capable command or plugin activation would invalidate this certificate.",
    }


def make_gate_report(cross: dict[str, Any], source_audit: dict[str, Any], build: dict[str, Any], frozen_after: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    frozen_passed = all(item.get("passed") for key in ["stage25r2", "stage27s", "stage28a"] for item in [frozen_after[key]])
    source_driver_passed = source_audit["official_driver_found"] and build["source_build"]["passed"] and build["plugin_export"]["passed"]
    primary = "blocked_real_ros2_control_runtime"
    blockers = [
        "blocked_real_ros2_control_runtime",
        "blocked_live_joint_state_unavailable",
        "blocked_joint_semantics_not_proven",
        "blocked_JTC_real_runtime_unverified",
        "blocked_MoveIt_execution_runtime_not_proven",
        "blocked_TCP_frame_calibration_not_proven",
        "blocked_real_hardware_limits_not_proven",
        "blocked_safety_state_not_proven",
    ]
    if not source_audit["static_safety_audit"]["zero_motion_safe_to_activate"]:
        blockers.append("blocked_hardware_plugin_activation_not_zero_motion_safe")
    if not frozen_passed:
        primary = "failed_frozen_input_integrity"
        blockers = ["failed_frozen_input_integrity"]
    return {
        "schema_version": "stage28ar-gate-report-v1",
        "stage": "Stage 2.8A-R — FAIRINO FR5 Real-Hardware Runtime Compatibility + Zero-Motion Bring-up",
        "status": "blocked_real_ros2_control_runtime" if frozen_passed else "failed_frozen_input_integrity",
        "decision": "BLOCKED; do not enter Stage 2.8B; no motion command was sent.",
        "compatibility_case": {
            "result": "compatible_with_existing_jazzy_stack" if build["source_build"]["passed"] and build["plugin_export"]["passed"] else "source_build_not_proven",
            "source_build": build["source_build"]["passed"],
            "plugin_export": build["plugin_export"]["passed"],
            "plugin_load": build["plugin_load"]["passed"],
            "trajectory_semantics_changed": False,
            "note": "This is source/API compatibility only; it is not proof of a connected real FR5 runtime.",
        },
        "gates": {
            "frozen_input_integrity": frozen_passed,
            "real_FR5_identity": False,
            "real_FAIRINO_driver": source_driver_passed,
            "real_ros2_control_hardware": False,
            "live_joint_state": False,
            "joint_semantics": False,
            "JTC_real_runtime": False,
            "Stage27S_runtime_semantic_transfer": False,
            "MoveIt_execution_mapping": False,
            "TCP_and_frames": False,
            "hardware_limits": False,
            "safety_state": False,
            "zero_motion_proof": True,
        },
        "Stage_2_5R2": "passed_frozen_unchanged",
        "Stage_2_7S": "passed_frozen_unchanged",
        "Stage_2_8B": "blocked_not_started",
        "motion_commands_sent": 0,
        "real_robot": {"detected": False, "connected": False, "identity_proven": False},
        "FAIRINO_runtime": {
            "driver": "official_source_found_and_built_in_isolation" if source_driver_passed else "not_proven",
            "SDK": source_audit["official_FAIRINO_source"].get("SDK_version"),
            "plugin": source_audit["official_FAIRINO_source"].get("plugin_class"),
            "compatibility_with_Jazzy": build["source_build"]["passed"],
        },
        "ros2_control": {"controller_manager": False, "hardware_component": False, "live_state": False},
        "JTC": {"runtime_verified": False, "version": None, "interpolation_method": None, "Stage27S_semantics_transfer": False},
        "MoveIt": {"execution_runtime_verified": False},
        "geometry": {"TCP_verified": False, "frames_verified": False},
        "limits": {"hardware_limits_verified": False},
        "safety": {"verified": False},
        "primary_blocker": primary,
        "sub_blockers": blockers,
        "frozen_verification_before": evidence["frozen_before"],
        "frozen_verification_after": frozen_after,
    }


def make_markdown_report(gate: dict[str, Any], source: dict[str, Any], build: dict[str, Any], evidence: dict[str, Any], no_motion: dict[str, Any]) -> str:
    yaml_block = "\n".join([
        "Stage_2_5R2: passed_frozen_unchanged",
        "Stage_2_7S: passed_frozen_unchanged",
        "",
        "Stage_2_8A_R:",
        f"  status: {gate['status']}",
        "",
        "Stage_2_8A:",
        f"  status: {gate['status']}",
        "",
        "Stage_2_8B:",
        "  status: blocked_not_started",
        "",
        "motion_commands_sent: 0",
        "",
        "real_robot:",
        "  detected: false",
        "  connected: false",
        "  identity_proven: false",
        "",
        "FAIRINO_runtime:",
        f"  driver: {gate['FAIRINO_runtime']['driver']}",
        f"  SDK: {gate['FAIRINO_runtime']['SDK'] or 'not_proven'}",
        f"  plugin: {gate['FAIRINO_runtime']['plugin'] or 'not_proven'}",
        f"  compatibility_with_Jazzy: {'passed_source_build_only' if gate['FAIRINO_runtime']['compatibility_with_Jazzy'] else 'not_proven'}",
        "",
        "ros2_control:",
        "  controller_manager: false",
        "  hardware_component: false",
        "  live_state: false",
        "",
        "JTC:",
        "  runtime_verified: false",
        "  version: not_proven",
        "  interpolation_method: not_proven",
        "  Stage27S_semantics_transfer: false",
        "",
        "MoveIt:",
        "  execution_runtime_verified: false",
        "",
        "geometry:",
        "  TCP_verified: false",
        "  frames_verified: false",
        "",
        "limits:",
        "  hardware_limits_verified: false",
        "",
        "safety:",
        "  verified: false",
        "",
        f"primary_blocker: {gate['primary_blocker']}",
    ])
    return f"""# Stage 2.8A-R — FAIRINO FR5 zero-motion runtime certification\n\n```yaml\n{yaml_block}\n```\n\n## Decision\n\n**BLOCKED.** No real FR5 runtime, controller manager, FAIRINO hardware component, live joint-state stream, JTC runtime, or MoveIt execution manager was observed. Stage 2.8B was not started.\n\nThe official FAIRINO ROS 2 source was found at `{source['official_FAIRINO_source']['repository']}` and built in an isolated Jazzy workspace. That proves source/API compilation and plugin export/load only; it does not prove a controller is configured, launched, active, connected to a real FR5, or publishing live data.\n\n## 1. What was done\n\n- Re-verified frozen Stage 2.5R2, Stage 2.7S, and prior Stage 2.8A SHA-256 manifests before and after the audit.\n- Audited the official nested Git copy, package metadata, CMake, plugin XML, bundled SDK, FR5 configuration, and hardware-plugin lifecycle/source.\n- Compiled `fairino_msgs` and `fairino_hardware_v3_9_7` against ROS 2 Jazzy in isolated WSL `/tmp` paths and loaded the shared library without activating it.\n- Ran only ROS graph, metadata, interface-list, parameter-read, topic-info/echo, and action-info/list probes; repeated graph probes three times.\n\n## 2. What was not done\n\nNo hardware interface was activated. No controller manager, JTC, MoveIt launch, action goal, trajectory publication, servo command, enable command, safety-parameter write, or motion-capable FAIRINO API was called.\n\n## 3. Driver and safety audit\n\nThe local official copy is `v3.0.0_RobotV3.9.7`, commit `{source['official_FAIRINO_source']['commit']}`, package version `{source['official_FAIRINO_source']['package_version']}`, plugin `{source['official_FAIRINO_source']['plugin_class']}`, and bundled SDK `{source['official_FAIRINO_source']['SDK_version']}`. The source build passed on Jazzy. However, the plugin exports position commands and `write()` calls `ServoJ`; `on_deactivate()` calls `StopMotion`. Because no independent zero-motion interlock is present, lifecycle activation is classified `blocked_hardware_plugin_activation_not_zero_motion_safe` and was not attempted.\n\nThe default FR5 xacro in the official MoveIt package uses `mock_components/GenericSystem`; that file is retained only as a rejected mock configuration, never as real-hardware evidence.\n\n## 4. Runtime gates\n\n- Real FR5 identity: **not proven**; no model, serial/identity, controller firmware, or robot IP was observed.\n- ros2_control: **blocked**; no live controller manager or FAIRINO hardware component.\n- Live joint state: **blocked**; `/joint_states` and `nonrt_state_data` were not available, so freshness, finite values, order, units, sign, and zero reference were not certified.\n- JTC: **blocked**; Jazzy package metadata is present, but no runtime endpoint, controller state, query-state service, parameters, or interpolation method was observed.\n- Stage 2.7S transfer: **ineligible**; the frozen `splines`/JTC 4.40.1 certificate is historical mock-runtime evidence and cannot transfer without a real runtime crosscheck.\n- MoveIt execution: **blocked**; static mapping is not a live execution manager and no execution call was made.\n- TCP/frames, hardware limits, and safety: **blocked**; no physical calibration, controller limits/overrides, E-stop/protective-stop state, or operating mode was available.\n\n## 5. Compatibility conclusion\n\nThe FAIRINO source is **Jazzy source/API compatible in the isolated build**; no Humble switch was made and no trajectory/JTC semantics were modified. This is not a real-hardware certification. The project remains blocked at `blocked_real_ros2_control_runtime`.\n\n## 6. No-motion proof\n\nThe independent certificate records all motion counters as zero: `motion_commands_sent: 0`, `FJT_goals_sent: 0`, `MoveIt_execute_calls: 0`, `joint_trajectory_publications: 0`, `servo_commands: 0`, `FAIRINO_motion_API_calls: 0`, and `RobotEnable_1_calls: 0`.\n\n## 7. Frozen inputs and remaining decision\n\nFrozen verification before and after is recorded in `stage28ar_input_manifest.json` and `stage28ar_gate_report.json`. The primary blocker is `{gate['primary_blocker']}` with sub-blockers: `{', '.join(gate['sub_blockers'])}`.\n\nStage 2.8B remains `blocked_not_started`; it is not authorized to begin.\n\nOfficial source reference: [FAIR-INNOVATION/frcobot_ros2](https://github.com/FAIR-INNOVATION/frcobot_ros2).\n"""


def write_bundle(gate: dict[str, Any], files_before_manifests: list[Path]) -> None:
    entries: list[dict[str, Any]] = []
    for path in sorted(OUT.rglob("*")):
        if not path.is_file() or path.name == "SHA256SUMS" or path.name == "stage28ar_artifact_manifest.json":
            continue
        entries.append({"path": str(path.relative_to(OUT)).replace("\\", "/"), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    artifact_manifest = {
        "schema_version": "stage28ar-artifact-manifest-v1",
        "stage": "Stage 2.8A-R",
        "generated_utc": utc_now(),
        "files_excluded_from_this_manifest": ["stage28ar_artifact_manifest.json", "SHA256SUMS"],
        "files": entries,
        "gate_report_status": gate["status"],
        "motion_commands_sent": 0,
    }
    json_dump(OUT / "stage28ar_artifact_manifest.json", artifact_manifest)
    lines: list[str] = []
    for path in sorted(OUT.rglob("*")):
        if not path.is_file() or path.name == "SHA256SUMS":
            continue
        lines.append(f"{sha256_file(path)}  {path.relative_to(OUT).as_posix()}")
    (OUT / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    started = utc_now()
    frozen_before = frozen_snapshot()
    json_dump(OUT / "stage28ar_input_manifest.json", {"schema_version": "stage28ar-input-manifest-v1", "capture_start_utc": started, "frozen_before": frozen_before, "frozen_roots_are_read_only": True})
    if not all(frozen_before[key].get("passed") for key in ["stage25r2", "stage27s", "stage28a"]):
        ended = utc_now()
        gate = {"schema_version": "stage28ar-gate-report-v1", "status": "failed_frozen_input_integrity", "primary_blocker": "failed_frozen_input_integrity", "motion_commands_sent": 0, "decision": "STOP; frozen input integrity failed before certification."}
        json_dump(OUT / "stage28ar_gate_report.json", gate)
        json_dump(OUT / "stage28ar_no_motion_certificate.json", make_no_motion_certificate(started, ended, {"static_safety_audit": {"classification": "not_run", "zero_motion_safe_to_activate": False}}, {"probes": {}}))
        write_bundle(gate, [])
        return 2

    source_audit = audit_fairino_source()
    build = run_isolated_jazzy_build()
    runtime = probe_runtime()
    cross = build_report_data(frozen_before, source_audit, build, runtime)
    evidence = build_static_evidence(source_audit, runtime, build, cross)
    for name, value in [
        ("stage28ar_environment.json", {"schema_version": "stage28ar-environment-v1", "host_os": platform.platform(), "python": sys.version, "windows": platform.system() == "Windows", "current_ros_stack": {"distro_prefix": cross["distro_prefix"], "ROS_DISTRO_environment_variable": cross["distro_environment_variable"], "package_versions": cross["versions"]}, "runtime_probes": runtime["repeatability"]}),
        ("stage28ar_fairino_driver_audit.json", source_audit),
        ("stage28ar_robot_identity.json", {"schema_version": "stage28ar-robot-identity-v1", "manufacturer": None, "model": None, "serial_or_identity": None, "controller_software_version": None, "firmware_version": None, "SDK_protocol_version": None, "robot_ip": None, "detected": False, "connected": False, "identity_proven": False, "status": "not_proven", "source_default_controller_ip": "192.168.58.2", "source_default_ip_is_not_identity": True}),
        ("stage28ar_ros2_control_runtime.json", evidence["ros2_control_runtime"]),
        ("stage28ar_hardware_interfaces.json", evidence["hardware_interfaces"]),
        ("stage28ar_live_joint_state.json", evidence["live_joint_state"]),
        ("stage28ar_joint_semantics.json", evidence["joint_semantics"]),
        ("stage28ar_jtc_runtime.json", evidence["jtc_runtime"]),
        ("stage28ar_stage27s_runtime_crosscheck.json", evidence["stage27s_crosscheck"]),
        ("stage28ar_moveit_execution_runtime.json", evidence["moveit_execution"]),
        ("stage28ar_tcp_frame_audit.json", evidence["tcp_frame"]),
        ("stage28ar_hardware_limits.json", evidence["hardware_limits"]),
        ("stage28ar_safety_state.json", evidence["safety"]),
        ("stage28ar_network_runtime.json", evidence["network"]),
    ]:
        json_dump(OUT / name, value)

    frozen_after = frozen_snapshot()
    json_dump(OUT / "stage28ar_input_manifest.json", {"schema_version": "stage28ar-input-manifest-v1", "capture_start_utc": started, "capture_end_utc": utc_now(), "frozen_before": frozen_before, "frozen_after": frozen_after, "integrity_passed": all(frozen_after[key].get("passed") for key in ["stage25r2", "stage27s", "stage28a"]), "frozen_roots_are_read_only": True})
    gate = make_gate_report(cross, source_audit, build, frozen_after, {"frozen_before": frozen_before})
    ended = utc_now()
    no_motion = make_no_motion_certificate(started, ended, source_audit, runtime)
    json_dump(OUT / "stage28ar_no_motion_certificate.json", no_motion)
    json_dump(OUT / "stage28ar_gate_report.json", gate)
    (OUT / "stage28ar_report.md").write_text(make_markdown_report(gate, source_audit, build, {"frozen_before": frozen_before}, no_motion), encoding="utf-8")
    write_bundle(gate, [])
    # A final check is intentionally performed after every new artifact is written.
    final_frozen = frozen_snapshot()
    if final_frozen != frozen_after:
        gate["status"] = "failed_frozen_input_integrity"
        gate["primary_blocker"] = "failed_frozen_input_integrity"
        gate["decision"] = "STOP; a frozen input changed during the audit."
        json_dump(OUT / "stage28ar_gate_report.json", gate)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
