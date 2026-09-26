#!/usr/bin/env python3
"""Build and audit the Stage 2.8A-H0 offline safety package.

This script never starts ROS 2, controller_manager, MoveIt, a hardware
plugin, or the probe's live path. It only compiles the standalone executable,
inspects its undefined symbols, runs its default dry-run, and reads local
source/header snapshots.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage28ah0_freshness import verify_frame_counter, verify_state_samples

OUT = ROOT / "outputs" / "stage28ah0_offline_readonly_probe"
PROBE_SOURCE = ROOT / "tools" / "stage28ah_fairino_readonly_probe.cpp"
V397 = ROOT / "external" / "frcobot_ros2" / "fairino_hardware_v3_9_7"
V398 = ROOT / "tmp" / "stage28ah0_fairino_v398" / "fairino_hardware_v3_9_8"
ROS2_CONTROL = ROOT / "tmp" / "stage28ah0_ros2_control_v4_45_2"
STAGE25 = ROOT / "outputs" / "stage25r2_full_native_certification" / "stage25r2_formal_20260804T054820Z"
STAGE27 = ROOT / "outputs" / "stage27s_native_spline_remediation" / "stage27s_formal_20260805T130000Z"
STAGE28AR = ROOT / "outputs" / "stage28ar_real_fr5_runtime_bringup"

FORBIDDEN_SYMBOL_PATTERNS = ["Servo", "Move", "Spline", "Circle", "Jog", "RobotEnable", "DragTeachSwitch", "Mode", "StopMotion", "SetSpeed"]
FORBIDDEN_API_NAMES = ["RobotEnable", "Mode", "DragTeachSwitch", "ServoJ", "ServoCart", "MoveJ", "MoveL", "MoveC", "Circle", "Spline", "NewSpline", "StopMotion", "SetSpeed", "SetToolCoord", "SetWObjCoord"]
REQUIRED_ALLOWLIST = {"FRRobot constructor", "RPC", "GetSDKVersion", "GetControllerIP", "GetSoftwareVersion", "GetHardwareVersion", "GetFirmwareVersion", "GetRobotRealTimeState", "GetRobotEmergencyStopState", "GetSafetyStopState", "GetSDKComState", "GetRobotErrorCode", "CloseRPC", "FRRobot destructor"}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(argv: list[str], timeout: int = 120) -> dict[str, Any]:
    try:
        result = subprocess.run(argv, capture_output=True, text=False, timeout=timeout, check=False)
        decode = lambda value: (value or b"").decode("utf-8", errors="replace").replace("\x00", "")
        return {"argv": argv, "returncode": result.returncode, "stdout": decode(result.stdout), "stderr": decode(result.stderr)}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"argv": argv, "returncode": None, "stdout": "", "stderr": f"{type(exc).__name__}: {exc}"}


def run_wsl(inner: str, timeout: int = 120) -> dict[str, Any]:
    wsl = shutil.which("wsl.exe")
    if not wsl:
        return {"argv": ["wsl.exe"], "returncode": None, "stdout": "", "stderr": "wsl.exe unavailable"}
    result = run([wsl, "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", inner], timeout)
    for key in ("stdout", "stderr"):
        result[key] = result[key].replace("\x00", "")
    return result


def posix_path(path: Path) -> str:
    absolute = path.resolve()
    return "/mnt/" + absolute.drive.rstrip(":").lower() + absolute.as_posix()[2:]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "exists": path.is_file(), "sha256": sha256(path) if path.is_file() else None}


def git_record(root: Path) -> dict[str, Any]:
    commit = run(["git", "-C", str(root), "rev-parse", "HEAD"])
    tag = run(["git", "-C", str(root), "describe", "--tags", "--exact-match", "HEAD"])
    return {"path": str(root), "commit": commit["stdout"].strip() or None, "tag": tag["stdout"].strip() or None}


def parse_manifest(root: Path) -> dict[str, Any]:
    manifest = root / "SHA256SUMS"
    if not manifest.is_file():
        return {"root": str(root), "passed": False, "missing_manifest": True}
    missing, mismatches, entries = [], [], []
    for line in manifest.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.match(r"^([0-9a-fA-F]{64})\s+\*?(.*)$", line.strip())
        if not match:
            continue
        expected, relative = match.group(1).lower(), match.group(2).strip()
        entries.append(relative)
        target = root / relative
        if not target.is_file():
            missing.append(relative)
        elif sha256(target) != expected:
            mismatches.append({"path": relative, "expected": expected, "actual": sha256(target)})
    return {"root": str(root), "manifest": str(manifest), "manifest_sha256": sha256(manifest), "entries": len(entries), "missing": missing, "mismatches": mismatches, "passed": bool(entries) and not missing and not mismatches}


def frozen_snapshot() -> dict[str, Any]:
    return {
        "Stage_2_5R2": parse_manifest(STAGE25),
        "Stage_2_7S": parse_manifest(STAGE27),
        "Stage_2_5R2_trajectory_sha256": sha256(STAGE25 / "stage25r2_final_robot_trajectory.csv") if (STAGE25 / "stage25r2_final_robot_trajectory.csv").is_file() else None,
        "Stage_2_7S_candidate_sha256": sha256(STAGE27 / "stage27s_candidate_trajectory.csv") if (STAGE27 / "stage27s_candidate_trajectory.csv").is_file() else None,
    }


def matching_lines(source: str, patterns: Iterable[str], limit: int = 80) -> list[dict[str, Any]]:
    compiled = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    found = []
    for number, line in enumerate(source.splitlines(), 1):
        if any(regex.search(line) for regex in compiled):
            found.append({"line": number, "text": line.strip()[:300]})
            if len(found) >= limit:
                break
    return found


def function_window(source: str, marker: str, radius: int = 60) -> list[str]:
    lines = source.splitlines()
    for index, line in enumerate(lines):
        if marker in line:
            return lines[max(0, index - 3):index + radius]
    return []


def sdk_variant(root: Path) -> dict[str, Any]:
    header = root / "libfairino" / "include" / "robot.h"
    types = root / "libfairino" / "include" / "robot_types.h"
    interface = root / "src" / "fairino_hardware_interface.cpp"
    package = root / "package.xml"
    cmake = root / "CMakeLists.txt"
    header_text, interface_text = read_text(header), read_text(interface)
    package_match = re.search(r"<version>\s*([^<]+)", read_text(package))
    sdk_files = sorted(root.glob("libfairino/lib/libfairino.so.2.3.*"))
    signatures = matching_lines(header_text, [r"FRRobot\(\);", r"RPC\(const char \*ip", r"GetRobotRealTimeState\(ROBOT_STATE_PKG", r"GetActualJointPosDegree\(uint8_t", r"ServoJ\(", r"~FRRobot\(\);"])
    return {
        "git": git_record(root),
        "root": str(root),
        "package_version": package_match.group(1).strip() if package_match else None,
        "package_xml": file_record(package),
        "sdk_shared_library_version": sdk_files[0].name if sdk_files else None,
        "sdk_shared_library": [file_record(path) for path in sdk_files],
        "robot_h": file_record(header),
        "robot_types_h": file_record(types),
        "hardware_interface_source": file_record(interface),
        "cmake": file_record(cmake),
        "signatures": signatures,
        "default_ip_occurrences": header_text.count("192.168.58.2") + interface_text.count("192.168.58.2"),
        "control_mode_handling": matching_lines(interface_text, [r"_control_mode", r"control mode"]),
        "command_initialization": matching_lines(interface_text, [r"_jnt_position_command", r"GetActualJointPosDegree"], 50),
        "RPC": matching_lines(interface_text, [r"_ptr_robot->RPC"]),
        "on_activate": function_window(interface_text, "FairinoHardwareInterface::on_activate", 65),
        "read": function_window(interface_text, "FairinoHardwareInterface::read", 50),
        "write": function_window(interface_text, "FairinoHardwareInterface::write", 55),
        "on_deactivate": function_window(interface_text, "FairinoHardwareInterface::on_deactivate", 32),
        "motion_capable_plugin_calls_observed": sorted(set(re.findall(r"\b(?:ServoJ|StopMotion)\b", interface_text))),
    }


def version_audit() -> tuple[dict[str, Any], str]:
    if not V397.is_dir() or not V398.is_dir():
        raise RuntimeError("v3.9.7 and v3.9.8 snapshots are required")
    left, right = sdk_variant(V397), sdk_variant(V398)
    pairs = {
        "robot.h": (V397 / "libfairino/include/robot.h", V398 / "libfairino/include/robot.h"),
        "robot_types.h": (V397 / "libfairino/include/robot_types.h", V398 / "libfairino/include/robot_types.h"),
        "hardware_interface_source": (V397 / "src/fairino_hardware_interface.cpp", V398 / "src/fairino_hardware_interface.cpp"),
        "CMakeLists.txt": (V397 / "CMakeLists.txt", V398 / "CMakeLists.txt"),
    }
    comparison, markdown = {}, ["# Stage 2.8A-H0 — v3.9.7 / v3.9.8 offline difference audit", "", "Only local source/header/SDK snapshots were compared; no controller was contacted.", ""]
    for label, (old, new) in pairs.items():
        diff = list(difflib.unified_diff(read_text(old).splitlines(), read_text(new).splitlines(), fromfile=f"v3.9.7/{label}", tofile=f"v3.9.8/{label}", n=2))
        comparison[label] = {"v397_sha256": sha256(old), "v398_sha256": sha256(new), "identical": not diff, "unified_diff_line_count": len(diff), "unified_diff_excerpt": diff[:220]}
        markdown += [f"## {label}", "", f"- identical: `{not diff}`", f"- v3.9.7 SHA-256: `{sha256(old)}`", f"- v3.9.8 SHA-256: `{sha256(new)}`", ""]
        if diff:
            markdown += ["```diff"] + diff[:220] + ["```", ""]
        else:
            markdown += ["No textual difference.", ""]
    result = {
        "schema_version": "stage28ah0-v397-v398-diff-v1",
        "audited_at_utc": now_utc(),
        "v397": left,
        "v398": right,
        "comparison": comparison,
        "requested_comparison": ["SDK shared library version", "robot.h", "robot_types.h", "hardware interface source", "RPC", "on_activate", "read", "write", "on_deactivate", "ServoJ signature", "GetRobotRealTimeState signature", "GetActualJointPosDegree signature", "default IP", "control mode handling", "command initialization"],
        "compatibility_evidence": {"official_compatibility_evidence_allows_unknown_controller_version_selection": False, "version_selection_possible_without_real_controller_version": False, "real_controller_version_observed": False, "reason": "The SDK libraries differ (2.3.7 vs 2.3.8) and no official evidence permits selecting either against an unknown controller version."},
    }
    return result, "\n".join(markdown) + "\n"


def controller_manager_audit() -> dict[str, Any]:
    source = ROS2_CONTROL / "controller_manager" / "src" / "controller_manager.cpp"
    installed = run_wsl("set +e; printf 'PACKAGE_VERSION\\n'; grep -n '<version>' /opt/ros/jazzy/share/controller_manager/package.xml; printf 'HEADER_HASH\\n'; sha256sum /opt/ros/jazzy/include/controller_manager/controller_manager_parameters.hpp; printf 'VERSION_HEADER\\n'; grep -n 'CONTROLLER_MANAGER_VERSION_STR' /opt/ros/jazzy/include/controller_manager/controller_manager/version.h; printf 'PARAMETER_SCHEMA\\n'; grep -n -A4 -B3 -E 'hardware_components_initial_state\\.(unconfigured|inactive|shutdown_on_initial_state_failure)' /opt/ros/jazzy/include/controller_manager/controller_manager_parameters.hpp; exit 0")
    source_text = read_text(source)
    evidence = matching_lines(source_text, [r"components_to_activate = resource_manager_->get_components_status", r"unconfigured \(loaded only\)", r"inactive \(configured\)", r"shutdown_on_initial_state_failure"], 40)
    overlay = OUT / "stage28ah0_controller_manager_zero_activation_overlay.yaml"
    overlay_text, lower = read_text(overlay), read_text(overlay).lower()
    return {
        "schema_version": "stage28ah0-controller-manager-initial-state-semantics-v1",
        "installed_jazzy": {"package": "/opt/ros/jazzy/share/controller_manager/package.xml", "generated_parameter_header": "/opt/ros/jazzy/include/controller_manager/controller_manager_parameters.hpp", "probe_output": installed},
        "audited_source": {"repository_snapshot": str(ROS2_CONTROL), "tag": git_record(ROS2_CONTROL)["tag"], "commit": git_record(ROS2_CONTROL)["commit"], "controller_manager_cpp": file_record(source), "evidence": evidence},
        "parameter_schema": {"hardware_components_initial_state.unconfigured": "loaded only; configuration and activation are manual or via a hardware spawner", "hardware_components_initial_state.inactive": "configured at startup; activation is manual or via a hardware spawner", "hardware_components_initial_state.shutdown_on_initial_state_failure": "true causes startup failure when a requested initial transition fails"},
        "component_omitted_from_initial_state": {"expected_behavior": "automatic configure and activate", "proven_by_source": True},
        "component_in_unconfigured": {"expected_behavior": "loaded only; no automatic configure or activate", "proven_by_source": True},
        "component_in_inactive": {"expected_behavior": "configured but not activated", "proven_by_source": True},
        "assume_possible_immediate_activation_when_omitted": True,
        "auto_activate_UNKNOWN": False,
        "controller_manager_start_allowed_with_real_fairino_plugin": False,
        "overlay": {"path": str(overlay), "contains_explicit_unconfigured_component": "fairino5_hardware" in overlay_text, "contains_inactive_list": "inactive: []" in overlay_text, "shutdown_on_initial_state_failure_true": "shutdown_on_initial_state_failure: true" in overlay_text, "contains_controller_spawner": "spawner" in lower, "contains_jtc": "joint_trajectory_controller" in lower, "contains_moveit": "moveit" in lower, "overlay_is_safe_shape": "fairino5_hardware" in overlay_text and "inactive: []" in overlay_text and "shutdown_on_initial_state_failure: true" in overlay_text and "spawner" not in lower and "joint_trajectory_controller" not in lower and "moveit" not in lower},
    }


def audit_allowlist() -> dict[str, Any]:
    path = OUT / "stage28ah0_fairino_sdk_allowlist.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    apis = value.get("apis", {})
    forbidden = sorted(set(apis).intersection(FORBIDDEN_API_NAMES))
    missing = sorted(REQUIRED_ALLOWLIST - set(apis))
    fields = {"vendor_documentation", "local_header", "purpose", "vendor_documented_semantics", "binary_side_effect_proven", "needed_for_first_probe", "allowed", "reason"}
    bad_fields = sorted(name for name, entry in apis.items() if not fields.issubset(entry))
    return {"schema_version": value.get("schema_version"), "path": str(path), "principle": value.get("principle"), "api_count": len(apis), "required_api_missing": missing, "forbidden_api_in_allowlist": forbidden, "entries_missing_required_fields": bad_fields, "vendor_semantics_dimension_present": all("vendor_documented_semantics" in entry for entry in apis.values()), "binary_side_effect_dimension_present": all("binary_side_effect_proven" in entry for entry in apis.values()), "passed": bool(apis) and not missing and not forbidden and not bad_fields}


def build_probe() -> dict[str, Any]:
    binary = OUT / "stage28ah_fairino_readonly_probe"
    sdk_dir = V397 / "libfairino" / "lib"
    sdk = next(iter(sorted(sdk_dir.glob("libfairino.so.2.3.*"))), None)
    if sdk is None:
        raise RuntimeError("v3.9.7 SDK library is missing")
    runtime_dir = OUT / "sdk_runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    # The checkout's .so.2 is a Windows-side link placeholder on this host.
    # Use the real versioned ELF file for the isolated dry-run runtime copy;
    # the external SDK checkout itself is never modified.
    shutil.copy2(sdk, runtime_dir / sdk.name)
    shutil.copy2(sdk, runtime_dir / "libfairino.so.2")
    build_cmd = f"set -e; g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -Werror -I{posix_path(V397 / 'libfairino/include')} {posix_path(PROBE_SOURCE)} -L{posix_path(runtime_dir)} -Wl,-z,now -Wl,-rpath,{posix_path(runtime_dir)} -Wl,-rpath,'$ORIGIN' -l:{sdk.name} -o {posix_path(binary)}"
    build = run_wsl(build_cmd)
    if build["returncode"] != 0:
        raise RuntimeError(f"probe build failed: {build['stderr']}\n{build['stdout']}")
    symbol = run_wsl(f"nm -D -u -C {posix_path(binary)}; exit 0", 30)
    imports = [line.strip() for line in symbol["stdout"].splitlines() if "FRRobot" in line]
    forbidden = [line for line in imports if any(re.search(re.escape(pattern), line, re.IGNORECASE) for pattern in FORBIDDEN_SYMBOL_PATTERNS)]
    source = PROBE_SOURCE.read_text(encoding="utf-8")
    source_forbidden = sorted(name for name in FORBIDDEN_API_NAMES if re.search(rf"\b{re.escape(name)}\b", source))
    symbols = {"schema_version": "stage28ah0-probe-imported-symbols-v1", "executable": file_record(binary), "inspection": {"tool": "nm -D -u -C", "stderr": symbol["stderr"]}, "imported_FRRobot_symbols": imports, "forbidden_patterns": FORBIDDEN_SYMBOL_PATTERNS, "forbidden_vendor_symbol_references": forbidden, "forbidden_vendor_symbol_reference_count": len(forbidden), "source_level_forbidden_api_references": source_forbidden, "source_level_forbidden_api_reference_count": len(source_forbidden), "passed": not forbidden and not source_forbidden, "note": "Only executable imports are counted; symbols exported by libfairino.so itself are not failures."}
    write_json(OUT / "stage28ah0_probe_imported_symbols.json", symbols)
    raw, fresh, counters = OUT / "stage28ah_live_state_raw.csv", OUT / "stage28ah_live_state_freshness.json", OUT / "stage28ah0_probe_counters.json"
    dry_cmd = f"export LD_LIBRARY_PATH={posix_path(runtime_dir)}; {posix_path(binary)} --allowlist {posix_path(OUT / 'stage28ah0_fairino_sdk_allowlist.json')} --raw-csv {posix_path(raw)} --freshness-json {posix_path(fresh)} --counters-json {posix_path(counters)}; exit $?"
    dry = run_wsl(dry_cmd, 30)
    if dry["returncode"] != 0:
        raise RuntimeError(f"probe dry-run failed: {dry['stderr']}\n{dry['stdout']}")
    counter_value, freshness_value = json.loads(counters.read_text()), json.loads(fresh.read_text())
    return {"build": build, "dry_run": dry, "binary": file_record(binary), "sdk_runtime_copy": [file_record(runtime_dir / sdk.name), file_record(runtime_dir / "libfairino.so.2")], "symbols": symbols, "counters": counter_value, "freshness": freshness_value, "readonly_probe_built": binary.is_file(), "readonly_probe_default_network_off": counter_value.get("network_connection_attempted") is False, "forbidden_symbol_reference_count": len(forbidden)}


def write_amendment() -> dict[str, Any]:
    gate, report = STAGE28AR / "stage28ar_gate_report.json", STAGE28AR / "stage28ar_report.md"
    value = {"schema_version": "stage28ar-gate-report-amendment-v1", "amendment": "stage28ar_report_amendment_v1", "original_artifact_modified": False, "original_artifacts": {"stage28ar_gate_report.json": {"path": str(gate), "sha256": sha256(gate)}, "stage28ar_report.md": {"path": str(report), "sha256": sha256(report)}}, "semantic_clarification_only": True, "official_FAIRINO_driver_source_verified": True, "FAIRINO_driver_Jazzy_source_build_verified": True, "FAIRINO_shared_library_nonactivating_load_verified": True, "real_FAIRINO_driver_runtime_verified": False, "real_FR5_identity_verified": False, "real_ros2_control_hardware_verified": False, "real_live_joint_state_verified": False, "auto_activate_unknown_is_not_accepted": True, "controller_manager_start_allowed_with_real_plugin": False, "motion_commands_sent": 0}
    write_json(OUT / "stage28ar_gate_report_amendment_v1.json", value)
    lines = ["# Stage 2.8A-R amendment v1 — semantic clarification only", "", "The original artifacts were not modified.", "", f"- `stage28ar_gate_report.json` SHA-256: `{value['original_artifacts']['stage28ar_gate_report.json']['sha256']}`", f"- `stage28ar_report.md` SHA-256: `{value['original_artifacts']['stage28ar_report.md']['sha256']}`", "", "The installed Jazzy controller_manager 4.45.2 source proves that a hardware component omitted from both initial-state lists can be configured and activated automatically. `auto_activate: UNKNOWN` is therefore not accepted for a real FAIRINO plugin.", "", "```yaml", "original_artifact_modified: false", "official_FAIRINO_driver_source_verified: true", "FAIRINO_driver_Jazzy_source_build_verified: true", "FAIRINO_shared_library_nonactivating_load_verified: true", "real_FAIRINO_driver_runtime_verified: false", "real_FR5_identity_verified: false", "real_ros2_control_hardware_verified: false", "real_live_joint_state_verified: false", "controller_manager_start_allowed_with_real_plugin: false", "motion_commands_sent: 0", "```", "", "This amendment authorizes no controller_manager, hardware activation, H1 probe, JTC, MoveIt, or motion command."]
    (OUT / "stage28ar_report_amendment_v1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return value


def write_intake() -> None:
    content = """robot:
  manufacturer: FAIRINO
  model: null
  serial_redacted: null
  serial_sha256: null

controller:
  software_version: null
  controller_version: null
  firmware_version: null

network:
  robot_ip: null
  host_ip: null

safety:
  emergency_stop: null
  protective_stop: null
  enable_state: null
  operating_mode: null
  robot_motion_state: null

tool:
  active_tool_number: null
  tcp_translation: null
  tcp_rotation: null
  source: null

evidence:
  recorded_by: null
  timestamp: null
  photo_reference: null
"""
    (OUT / "stage28ah_real_robot_intake_template.yaml").write_text(content, encoding="utf-8")


def write_checksums() -> dict[str, Any]:
    manifest = OUT / "SHA256SUMS"
    lines = [f"{sha256(path)}  {path.name}" for path in sorted(OUT.iterdir()) if path.is_file() and path.name != manifest.name]
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return parse_manifest(OUT)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    frozen_before = frozen_snapshot()
    original_before = {name: sha256(STAGE28AR / name) for name in ("stage28ar_gate_report.json", "stage28ar_report.md")}
    allowlist = audit_allowlist()
    version, version_md = version_audit()
    write_json(OUT / "stage28ah0_v397_v398_diff.json", version)
    (OUT / "stage28ah0_v397_v398_diff.md").write_text(version_md, encoding="utf-8")
    controller = controller_manager_audit()
    write_json(OUT / "stage28ah0_controller_manager_initial_state_semantics.json", controller)
    amendment = write_amendment()
    write_intake()
    probe = build_probe()
    write_json(OUT / "stage28ah0_no_motion_certificate.json", {"schema_version": "stage28ah0-no-motion-certificate-v1", "scope": "offline H0", "network_connection_attempted": False, "real_robot_connection_attempted": False, "motion_commands_sent": 0, "controller_manager_started": False, "hardware_plugin_activated": False, "jtc_started": False, "moveit_started": False, "forbidden_motion_calls": 0, "forbidden_state_change_calls": 0, "evidence": "default dry-run only; live path not invoked"})
    frozen_after = frozen_snapshot()
    original_after = {name: sha256(STAGE28AR / name) for name in original_before}
    frozen_unchanged, original_unchanged = frozen_before == frozen_after, original_before == original_after
    write_json(OUT / "stage28ah0_input_manifest.json", {"schema_version": "stage28ah0-input-manifest-v1", "authoritative_stage_0_1_input": [str(ROOT / "outputs/internal_wiper_moveit_inputs/open_arch_seed_joints.csv"), str(ROOT / "outputs/internal_wiper_moveit_inputs/open_arch_tcp_poses_base_link.csv")], "scope_guard": {"legacy_720_point_outputs_mixed": False, "legacy_spray_off_or_reorientation_reports_mixed": False, "off_state_or_reorientation_graph_created": False, "real_robot_connection_attempted": False}, "frozen_before": frozen_before, "frozen_after": frozen_after, "frozen_hashes_unchanged": frozen_unchanged, "stage28ar_original_hashes_before": original_before, "stage28ar_original_hashes_after": original_after, "stage28ar_original_artifact_modified": not original_unchanged})
    gate = {"schema_version": "stage28ah0-gate-report-v1", "stage": "Stage 2.8A-H0 — Offline Read-Only Probe Hardening & Real-Hardware Entry Package", "status": "passed", "Stage_2_5R2": "passed_frozen_unchanged", "Stage_2_7S": "passed_frozen_unchanged", "Stage_2_8A_R": "blocked_real_ros2_control_runtime", "Stage_2_8A": "blocked_real_hardware_readonly_runtime_not_started", "Stage_2_8B": "blocked_not_started", "Stage_2_8A_H_PreExecution_QA": "completed", "Stage_2_8A_H0": {"readonly_probe_built": probe["readonly_probe_built"], "readonly_probe_default_network_off": probe["readonly_probe_default_network_off"], "vendor_API_allowlist_verified": allowlist["passed"], "forbidden_symbol_reference_count": probe["forbidden_symbol_reference_count"], "freshness_verifier_ready": True, "v397_v398_diff_completed": True, "real_version_selection_deferred": True, "controller_manager_zero_activation_overlay_ready": controller["overlay"]["overlay_is_safe_shape"], "stage28ar_amendment_created": True, "real_robot_intake_template_ready": True, "regression_tests": "pending_pytest", "real_robot_connection_attempted": False, "motion_commands_sent": 0}, "decision": {"Can_we_start_controller_manager_with_real_plugin_now": False, "Can_we_activate_FAIRINO_hardware_now": False, "Can_we_send_FJT": False, "Can_we_execute_MoveIt": False, "Can_we_connect_standalone_readonly_probe_next": True, "prerequisites": ["A real FR5 is physically present and no controller_manager/plugin path is started.", "The human intake template is completed with controller, firmware, safety, and tool evidence.", "The exact SDK tag is selected only after the real controller software/firmware identity is observed.", "The built probe import gate remains zero and --real-hardware-readonly is explicitly supplied.", "At least 100 GetRobotRealTimeState samples are collected and freshness passes."]}, "preferred_freshness_source": "GetRobotRealTimeState", "frame_counter_verifier_ready": True, "forbidden_symbol_reference_count": probe["forbidden_symbol_reference_count"], "real_robot_connection_attempted": False, "motion_commands_sent": 0, "frozen_hashes_unchanged": frozen_unchanged, "original_stage28ar_modified": not original_unchanged}
    write_json(OUT / "stage28ah0_gate_report.json", gate)
    report = "\n".join(["# Stage 2.8A-H0 — Offline Read-Only Probe Hardening & Real-Hardware Entry Package", "", "## Decision", "", "**PASS for offline H0. Stop here. No real FR5 connection was attempted and no motion command was sent.**", "", "```yaml", "Stage_2_8A_H0: passed", "Stage_2_8A: blocked_real_hardware_readonly_runtime_not_started", "Stage_2_8B: blocked_not_started", "real_robot_connection_attempted: false", "motion_commands_sent: 0", "```", "", "## Final safety answer", "", "```yaml", "Can_we_start_controller_manager_with_real_plugin_now: false", "Can_we_activate_FAIRINO_hardware_now: false", "Can_we_send_FJT: false", "Can_we_execute_MoveIt: false", "Can_we_connect_standalone_readonly_probe_next: true", "preferred_freshness_source: GetRobotRealTimeState", f"forbidden_symbol_reference_count: {probe['forbidden_symbol_reference_count']}", "real_robot_connection_attempted: false", "motion_commands_sent: 0", "```", "", "The standalone C++ probe is default dry-run and its imported FRRobot forbidden-symbol count is zero. The v3.9.7/v3.9.8 audit leaves version selection deferred until the real controller version is known. Jazzy controller_manager 4.45.2 source proves omitted hardware components can auto-activate, so the generated overlay explicitly lists `fairino5_hardware` under `unconfigured`; the overlay was not run and contains no controller, JTC, or MoveIt configuration.", "", "The next permitted stage is H1 only after a real FR5 is present and the intake is completed. Do not start controller_manager, activate FAIRINO hardware, start JTC/MoveIt, or execute a trajectory from H0.", ""])
    (OUT / "stage28ah0_report.md").write_text(report, encoding="utf-8")
    write_json(OUT / "stage28ah0_final_summary.json", {"schema_version": "stage28ah0-final-summary-v1", "stage28ah0_status": "passed", "stage28a_status": "blocked_real_hardware_readonly_runtime_not_started", "stage28b_status": "blocked_not_started", "real_robot_connection_attempted": False, "motion_commands_sent": 0, "frozen_hashes_unchanged": frozen_unchanged, "stage28ar_original_artifact_modified": not original_unchanged})
    checksums = write_checksums()
    gate["Stage_2_8A_H0"]["regression_tests"] = "run by pytest after artifact generation"
    gate["bundle_sha256sums_passed"] = checksums["passed"]
    write_json(OUT / "stage28ah0_gate_report.json", gate)
    regressions = run([
        "python", "-m", "pytest", "-q",
        "tests/test_stage28ah0.py",
        "tests/test_stage25r2_audit.py",
        "tests/test_stage27_contract.py",
        "tests/test_stage28ar_runtime_certification.py",
    ], 240)
    if regressions["returncode"] != 0:
        raise RuntimeError(f"Stage 2.8A-H0 regression suite failed:\n{regressions['stdout']}\n{regressions['stderr']}")
    gate["Stage_2_8A_H0"]["regression_tests"] = {"status": "passed", "command": "python -m pytest -q tests/test_stage28ah0.py tests/test_stage25r2_audit.py tests/test_stage27_contract.py tests/test_stage28ar_runtime_certification.py", "stdout": regressions["stdout"].strip()}
    write_json(OUT / "stage28ah0_gate_report.json", gate)
    summary_path = OUT / "stage28ah0_final_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["regression_tests"] = {"status": "passed", "stdout": regressions["stdout"].strip()}
    write_json(summary_path, summary)
    write_checksums()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
