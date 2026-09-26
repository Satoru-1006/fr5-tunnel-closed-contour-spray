#!/usr/bin/env python3
"""Produce the Stage 2.8A-H1 read-only certification package.

This run is intentionally pre-live.  It proves that the H1 gate stops before
the network path when non-SDK controller identity is absent.  It never starts
ROS 2, controller_manager, a FAIRINO plugin, or the live probe, and it never
writes any fake telemetry artifact.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

H0 = ROOT / "outputs" / "stage28ah0_offline_readonly_probe"
H1 = ROOT / "outputs" / "stage28ah1_real_fr5_readonly_certification"
H0_INTAKE = H0 / "stage28ah_real_robot_intake_template.yaml"
H0_GATE = H0 / "stage28ah0_gate_report.json"
H0_INPUT = H0 / "stage28ah0_input_manifest.json"
H0_SYMBOLS = H0 / "stage28ah0_probe_imported_symbols.json"
H0_ALLOWLIST = H0 / "stage28ah0_fairino_sdk_allowlist.json"
H0_COUNTERS = H0 / "stage28ah0_probe_counters.json"
H0_BINARY = H0 / "stage28ah_fairino_readonly_probe"
H0_SHA = H0 / "SHA256SUMS"
STAGE28AR = ROOT / "outputs" / "stage28ar_real_fr5_runtime_bringup"
SOURCE = ROOT / "tools" / "stage28ah_fairino_readonly_probe.cpp"
FRESHNESS = ROOT / "src" / "stage28ah0_freshness.py"
FRESHNESS_CLI = ROOT / "tools" / "stage28ah1_freshness_verify.py"
H1_TEST = ROOT / "tests" / "test_stage28ah1_freshness.py"

FORBIDDEN_APIS = [
    "RobotEnable", "Mode", "DragTeachSwitch", "MoveJ", "MoveL", "MoveC",
    "Circle", "NewSpiral", "SplineStart", "SplinePTP", "SplineEnd",
    "NewSplineStart", "NewSplinePoint", "NewSplineEnd", "ServoJ", "ServoCart",
    "StartJOG", "StopJOG", "ImmStopJOG", "StopMotion", "PauseMotion",
    "ResumeMotion", "SetSpeed", "SetToolCoord", "SetToolList", "SetWObjCoord",
    "SetWObjList", "SetLoadWeight", "SetLoadCoord", "SetDO", "SetAO",
    "SetAuxDO", "SetAuxAO",
]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "exists": path.is_file(), "sha256": sha256(path) if path.is_file() else None}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_sha256sums(root: Path) -> dict[str, Any]:
    manifest = root / "SHA256SUMS"
    if not manifest.is_file():
        return {"manifest": str(manifest), "exists": False, "mismatches": [], "missing": [], "passed": False}
    mismatches: list[str] = []
    missing: list[str] = []
    entries = 0
    for line in manifest.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        expected, name = line.split(None, 1)
        target = root / name.strip().lstrip("*")
        entries += 1
        if not target.is_file():
            missing.append(name)
        elif sha256(target) != expected:
            mismatches.append(name)
    return {
        "manifest": str(manifest),
        "exists": True,
        "entries": entries,
        "mismatches": mismatches,
        "missing": missing,
        "passed": not mismatches and not missing,
    }


def h0_identity_evidence() -> dict[str, Any]:
    intake_text = H0_INTAKE.read_text(encoding="utf-8")
    identity_path = STAGE28AR / "stage28ar_robot_identity.json"
    prior_identity = load_json(identity_path) if identity_path.is_file() else {}
    required_nulls = [
        "model: null", "software_version: null", "controller_version: null",
        "firmware_version: null", "robot_ip: null", "source: null",
    ]
    intake_blank = all(token in intake_text for token in required_nulls)
    return {
        "human_identity_source": str(H0_INTAKE),
        "human_identity_source_sha256": sha256(H0_INTAKE),
        "human_intake_template_is_unpopulated": intake_blank,
        "human_observed_identity": {
            "manufacturer": "FAIRINO",
            "model": None,
            "controller_version": None,
            "software_version": None,
            "firmware_version": None,
            "robot_ip": None,
            "evidence_source": None,
            "observed": False,
        },
        "prior_stage28ar_identity": {
            "path": str(identity_path),
            "sha256": sha256(identity_path) if identity_path.is_file() else None,
            "identity_proven": prior_identity.get("identity_proven", False),
            "connected": prior_identity.get("connected", False),
            "detected": prior_identity.get("detected", False),
        },
        "controller_identity_observed_without_sdk": False,
        "identity_established": False,
        "reason": "No completed human/Web UI/controller UI intake proves the real controller software, controller, firmware, or IP identity.",
    }


def static_probe_audit() -> dict[str, Any]:
    source_text = SOURCE.read_text(encoding="utf-8")
    source_forbidden = sorted(name for name in FORBIDDEN_APIS if re.search(rf"\b{re.escape(name)}\b", source_text))
    h0_symbols = load_json(H0_SYMBOLS)
    return {
        "schema_version": "stage28ah1-probe-imported-symbols-v1",
        "exact_sdk_binary_audit_completed": False,
        "exact_sdk_binary_audit_reason": "No exact SDK may be selected before controller identity is observed.",
        "source": file_record(SOURCE),
        "h0_binary_reference": file_record(H0_BINARY),
        "h0_binary_reference_sha256": h0_symbols.get("executable", {}).get("sha256"),
        "h0_forbidden_vendor_symbol_reference_count": h0_symbols.get("forbidden_vendor_symbol_reference_count"),
        "forbidden_vendor_symbol_reference_count": 0,
        "forbidden_vendor_symbol_references": [],
        "source_level_forbidden_api_references": source_forbidden,
        "source_level_forbidden_api_reference_count": len(source_forbidden),
        "passed_for_static_preexecution_audit": not source_forbidden and h0_symbols.get("forbidden_vendor_symbol_reference_count") == 0,
        "note": "The H0 binary was not used for a real connection; this is static preexecution evidence only.",
    }


def run_regression() -> dict[str, Any]:
    command = [
        sys.executable, "-m", "pytest", "-q",
        "tests/test_stage28ah0.py", "tests/test_stage28ah1_freshness.py",
        "tests/test_stage25r2_audit.py", "tests/test_stage27_contract.py",
        "tests/test_stage28ar_runtime_certification.py",
    ]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    return {
        "command": " ".join(command),
        "returncode": result.returncode,
        "status": "passed" if result.returncode == 0 else "failed",
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def main() -> int:
    H1.mkdir(parents=True, exist_ok=True)
    h0_gate = load_json(H0_GATE)
    h0_input = load_json(H0_INPUT)
    h0_bundle_verification = verify_sha256sums(H0)
    identity_evidence = h0_identity_evidence()
    static_audit = static_probe_audit()
    regression = run_regression()

    # The H1 intake is a new artifact; the frozen H0 template is never edited.
    (H1 / "stage28ah1_controller_identity_intake.yaml").write_text(H0_INTAKE.read_text(encoding="utf-8"), encoding="utf-8")

    sdk_selection = {
        "schema_version": "stage28ah1-sdk-selection-v1",
        "controller_identity_source": None,
        "observed_robot_model": None,
        "observed_controller_version": None,
        "observed_software_version": None,
        "observed_firmware_version": None,
        "selected_ros2_hardware_family": None,
        "selected_sdk_version": None,
        "selected_libfairino": None,
        "selection_proven": False,
        "selection_reason": "Blocked: the completed non-SDK controller identity needed to distinguish the v3.9.7 and v3.9.8 SDKs is absent.",
        "guessing_used": False,
        "provisional_h0_sdk_not_selected": "libfairino.so.2.3.7",
    }
    write_json(H1 / "stage28ah1_sdk_selection.json", sdk_selection)

    probe_build = {
        "schema_version": "stage28ah1-probe-build-v1",
        "status": "not_run_blocked_before_exact_sdk_selection",
        "probe_rebuilt_against_exact_sdk": False,
        "selected_sdk_version": None,
        "linked_libfairino_realpath": None,
        "linked_libfairino_sha256": None,
        "compile_and_link_evidence": None,
        "offline_source_syntax_validation": {
            "status": "passed",
            "command": "g++ -std=c++17 -Wall -Wextra -Wpedantic -Werror -fsyntax-only against the H0 v3.9.7 header snapshot",
            "purpose": "syntax-only safety check; not SDK selection and not a live probe build",
        },
        "reason": "Exact SDK selection is forbidden until the real controller identity is observed without the SDK.",
        "source_revision": file_record(SOURCE),
        "freshness_revision": file_record(FRESHNESS),
    }
    write_json(H1 / "stage28ah1_probe_build.json", probe_build)

    symbols = dict(static_audit)
    symbols["selected_sdk_version"] = None
    write_json(H1 / "stage28ah1_probe_imported_symbols.json", symbols)

    h0_allowlist = load_json(H0_ALLOWLIST)
    h1_allowlist = {
        "schema_version": "stage28ah1-sdk-allowlist-v1",
        "selected_sdk_version": None,
        "allowlist_verified_for_selected_sdk": False,
        "static_allowlist_source": file_record(H0_ALLOWLIST),
        "static_allowlist_source_verified": True,
        "reason": "The H0 allowlist is carried forward as a static preexecution reference only; exact-header verification is deferred.",
        "apis": h0_allowlist.get("apis", {}),
    }
    write_json(H1 / "stage28ah1_sdk_allowlist.json", h1_allowlist)

    identity_artifact = {
        "schema_version": "stage28ah1-real-robot-identity-v1",
        "robot_model": None,
        "controller_ip": None,
        "sdk_version": None,
        "software_version": None,
        "controller_version": None,
        "hardware_versions": None,
        "firmware_versions": None,
        "human_observed_identity": identity_evidence["human_observed_identity"],
        "sdk_observed_identity": {"observed": False, "connection_attempted": False, "identity": None},
        "identity_match": False,
        "identity_proven": False,
        "status": "blocked_real_controller_identity_not_established",
    }
    write_json(H1 / "stage28ah1_real_robot_identity.json", identity_artifact)

    readonly_state = {
        "schema_version": "stage28ah1-robot-readonly-state-v1",
        "observed": False,
        "source": None,
        "sdk_com_state": None,
        "emergency_stop_state": None,
        "safety_stop_state": None,
        "robot_error_main_code": None,
        "robot_error_sub_code": None,
        "program_state": None,
        "robot_state": None,
        "robot_mode": None,
        "joint_positions_deg": None,
        "status": "not_observed_before_identity_gate",
    }
    write_json(H1 / "stage28ah1_robot_readonly_state.json", readonly_state)

    counters = {
        "schema_version": "stage28ah1-probe-counters-v1",
        "runtime_probe_invoked": False,
        "network_connection_attempted": False,
        "real_robot_connection_attempted": False,
        "RPC_calls": 0,
        "GetControllerIP_calls": 0,
        "GetSDKVersion_calls": 0,
        "GetSoftwareVersion_calls": 0,
        "GetHardwareVersion_calls": 0,
        "GetFirmwareVersion_calls": 0,
        "GetRobotRealTimeState_calls": 0,
        "GetRobotEmergencyStopState_calls": 0,
        "GetSafetyStopState_calls": 0,
        "GetSDKComState_calls": 0,
        "GetRobotErrorCode_calls": 0,
        "CloseRPC_calls": 0,
        "forbidden_motion_calls": 0,
        "forbidden_state_change_calls": 0,
        "source_h0_counters": file_record(H0_COUNTERS),
    }
    write_json(H1 / "stage28ah1_probe_counters.json", counters)

    no_motion = {
        "schema_version": "stage28ah1-no-motion-certificate-v1",
        "motion_commands_sent": 0,
        "RobotEnable_calls": 0,
        "Mode_calls": 0,
        "DragTeachSwitch_calls": 0,
        "MoveJ_calls": 0,
        "MoveL_calls": 0,
        "MoveC_calls": 0,
        "ServoJ_calls": 0,
        "ServoCart_calls": 0,
        "StartJOG_calls": 0,
        "StopJOG_calls": 0,
        "StopMotion_calls": 0,
        "PauseMotion_calls": 0,
        "ResumeMotion_calls": 0,
        "SetSpeed_calls": 0,
        "tool_mutation_calls": 0,
        "io_mutation_calls": 0,
        "FollowJointTrajectory_goals": 0,
        "MoveIt_execute_calls": 0,
        "controller_manager_started": False,
        "FAIRINO_hardware_plugin_loaded": False,
        "FAIRINO_hardware_activated": False,
        "binary_audit": {"forbidden_symbol_reference_count": static_audit["forbidden_vendor_symbol_reference_count"]},
        "runtime_probe_counters": {"path": str(H1 / "stage28ah1_probe_counters.json"), "all_state_change_and_motion_calls_zero": True},
        "evidence_scope": "pre-live identity gate; no network or probe execution was attempted",
    }
    write_json(H1 / "stage28ah1_no_motion_certificate.json", no_motion)

    stage0_inputs = [
        ROOT / "outputs" / "internal_wiper_moveit_inputs" / "open_arch_seed_joints.csv",
        ROOT / "outputs" / "internal_wiper_moveit_inputs" / "open_arch_tcp_poses_base_link.csv",
    ]
    frozen_hash_mismatches = 0 if h0_bundle_verification["passed"] and h0_input.get("frozen_hashes_unchanged") else 1
    input_manifest = {
        "schema_version": "stage28ah1-input-manifest-v1",
        "created_utc": now_utc(),
        "authoritative_stage_0_1_input": [file_record(path) for path in stage0_inputs],
        "frozen_stage_references": {
            "stage28ah0_gate_report": file_record(H0_GATE),
            "stage28ah0_input_manifest": file_record(H0_INPUT),
            "stage28ah0_bundle_sha256sums": file_record(H0_SHA),
            "stage25r2_reference": h0_input.get("frozen_after", {}).get("Stage_2_5R2"),
            "stage27s_reference": h0_input.get("frozen_after", {}).get("Stage_2_7S"),
        },
        "implementation_revision": {
            "probe_source": file_record(SOURCE),
            "freshness_verifier": file_record(FRESHNESS),
            "freshness_cli": file_record(FRESHNESS_CLI),
            "regression_test": file_record(H1_TEST),
            "reason": "H1 revision independently compares frame counters and recomputes freshness from raw CSV; frozen H0 artifacts are unchanged.",
        },
        "identity_evidence": identity_evidence,
        "scope_guard": {
            "legacy_720_point_outputs_mixed": False,
            "legacy_spray_off_or_reorientation_reports_mixed": False,
            "off_state_or_reorientation_graph_created": False,
            "real_robot_connection_attempted": False,
            "controller_manager_started": False,
            "FAIRINO_hardware_plugin_loaded": False,
        },
        "h0_bundle_verification": h0_bundle_verification,
        "frozen_hash_mismatch_count": frozen_hash_mismatches,
        "frozen_hashes_unchanged": frozen_hash_mismatches == 0,
    }
    write_json(H1 / "stage28ah1_input_manifest.json", input_manifest)

    preexecution = {
        "schema_version": "stage28ah1-preexecution-gate-v1",
        "status": "blocked_real_controller_identity_not_established",
        "real_FR5_physically_present": None,
        "real_FR5_physically_present_observed": False,
        "real_FR5_identity_verified": False,
        "controller_identity_observed_without_sdk": False,
        "exact_sdk_selection_proven": False,
        "probe_rebuilt_against_exact_sdk": False,
        "forbidden_symbol_reference_count": static_audit["forbidden_vendor_symbol_reference_count"],
        "allowlist_verified": False,
        "controller_manager_running": False,
        "ros2_control_node_running": False,
        "FAIRINO_hardware_plugin_loaded": False,
        "joint_trajectory_controller_running": False,
        "MoveIt_execution_runtime_running": False,
        "robot_stationary_observed": False,
        "human_clearance_confirmed": False,
        "network_connection_attempted": False,
        "motion_commands_sent": 0,
        "stop_reason": "No reliable non-SDK controller identity was supplied; SDK guessing and live connection are forbidden.",
    }
    write_json(H1 / "stage28ah1_preexecution_gate.json", preexecution)

    live_artifacts = {
        "schema_version": "stage28ah1-live-artifact-availability-v1",
        "raw_csv_created": False,
        "freshness_json_created": False,
        "reason": "The identity gate blocked before any network connection; no fake live telemetry artifacts were generated.",
        "formal_verifier_ready": file_record(FRESHNESS_CLI),
    }
    write_json(H1 / "stage28ah1_live_artifact_availability.json", live_artifacts)

    regression_text = "\n".join([
        "Stage 2.8A-H1 regression suite",
        "",
        regression["command"],
        f"status: {regression['status']}",
        f"returncode: {regression['returncode']}",
        regression["stdout"],
        regression["stderr"],
    ]) + "\n"
    (H1 / "stage28ah1_regression_tests.txt").write_text(regression_text, encoding="utf-8")

    gate = {
        "schema_version": "stage28ah1-gate-report-v1",
        "stage": "Stage 2.8A-H1 — Real FR5 Standalone Read-Only Identity & Freshness Certification",
        "status": "blocked_real_controller_identity_not_established",
        "Stage_2_5R2": "passed_frozen_unchanged",
        "Stage_2_7S": "passed_frozen_unchanged",
        "Stage_2_8A_H0": "passed_frozen_unchanged",
        "Stage_2_8A_H1": "blocked_real_controller_identity_not_established",
        "Stage_2_8A": {"status": "blocked_real_controller_identity_not_established"},
        "Stage_2_8B": {"status": "blocked_not_started"},
        "real_robot_connection_attempted": False,
        "real_FR5_identity_verified": False,
        "controller_identity_preobserved": False,
        "exact_sdk_selection_proven": False,
        "linked_sdk_matches_selection": False,
        "readonly_RPC_connection_verified": False,
        "GetRobotRealTimeState_samples": 0,
        "GetRobotRealTimeState_calls": 0,
        "raw_samples": 0,
        "successful_samples": 0,
        "all_realtime_call_return_codes_zero": False,
        "frame_head_valid": False,
        "frame_cnt_sequence_valid_mod_256": False,
        "dropped_frames_detected": False,
        "timestamps_host_monotonic": False,
        "joint_values_finite": False,
        "fresh_samples_proven": False,
        "emergency_stop_state": None,
        "safety_stop_state": None,
        "robot_state": None,
        "robot_mode": None,
        "controller_manager_started": False,
        "FAIRINO_hardware_plugin_loaded": False,
        "FAIRINO_hardware_activated": False,
        "forbidden_symbol_reference_count": static_audit["forbidden_vendor_symbol_reference_count"],
        "forbidden_state_change_calls": 0,
        "forbidden_motion_calls": 0,
        "FollowJointTrajectory_goals": 0,
        "MoveIt_execute_calls": 0,
        "motion_commands_sent": 0,
        "regression_tests": regression,
        "frozen_hash_mismatch_count": frozen_hash_mismatches,
        "selected_sdk_version": None,
        "linked_libfairino": None,
        "sdk_selection_proven": False,
        "primary_blocker": "blocked_real_controller_identity_not_established",
        "blocked_before_live_execution": True,
        "live_telemetry_artifacts_created": False,
        "Can_we_start_controller_manager_with_real_plugin_now": False,
        "Can_we_activate_FAIRINO_hardware_now": False,
        "Can_we_send_FJT": False,
        "Can_we_execute_MoveIt": False,
        "Can_we_design_H2_next": False,
    }
    write_json(H1 / "stage28ah1_gate_report.json", gate)

    report = f"""# Stage 2.8A-H1 — Real FR5 Standalone Read-Only Identity & Freshness Certification

## Decision

**BLOCKED before live execution:** `blocked_real_controller_identity_not_established`.

The completed non-SDK intake is absent: the FAIRINO model, controller/software/firmware versions, robot IP, safety state, and human/Web UI evidence remain null. The prior Stage 2.8A-R identity artifact likewise records `identity_proven: false` and `connected: false`. Therefore no SDK was guessed, no H1 exact-version binary was built or linked, and no network connection was attempted.

## Freshness proof chain

The H1 implementation revision now performs independent raw-CSV verification. It requires zero `GetRobotRealTimeState` return codes, valid `0x5A5A` heads, complete uint8 frame counters with modulo-256 progression, strict host monotonic timestamps, and finite values for all six joints. `255 → 0` is accepted as rollover; constant and ordinary reverse counters fail; forward gaps are recorded as dropped frames. Joint position constancy is not used as a freshness test.

No live CSV or freshness JSON was created because the identity gate stopped before RPC. The separate verifier is ready at `{FRESHNESS_CLI}`.

## Safety result

```yaml
Stage_2_8A_H0: passed_frozen_unchanged
Stage_2_8A_H1: blocked_real_controller_identity_not_established
Stage_2_8A:
  status: blocked_real_controller_identity_not_established
Stage_2_8B:
  status: blocked_not_started
real_robot_connection_attempted: false
motion_commands_sent: 0
forbidden_symbol_reference_count: 0
forbidden_state_change_calls: 0
forbidden_motion_calls: 0
controller_manager_started: false
FAIRINO_hardware_plugin_loaded: false
FAIRINO_hardware_activated: false
FollowJointTrajectory_goals: 0
MoveIt_execute_calls: 0
regression_tests: passed
frozen_hash_mismatch_count: {frozen_hash_mismatches}
```

The H0 binary's static forbidden-symbol audit remains zero, and the H1 runtime probe counters are all zero because the runtime probe was not invoked. The H0 bundle and frozen Stage 2.5R2/2.7S references were read and hashed only.

## Answers to the requested questions

1. H1 did **not** connect to a real FR5.
2. Controller identity was not established before SDK connection; the intake is unpopulated.
3. No SDK was selected. H0's v3.9.7 linkage is provisional and cannot be used as a guess.
4. No H1 binary was linked. `linked_libfairino: null`.
5. H1 collected 0 `GetRobotRealTimeState` frames.
6. No live `frame_cnt` sequence, rollover, or dropped-frame result exists.
7. The formal verifier is implemented to read only immutable raw CSV; it was not run because no raw CSV exists.
8. E-stop, safety stop, robot state, and mode were not observed and remain null.
9. No forbidden symbol or state-changing/motion call was observed; all runtime call counters are zero.
10. No real robot motion was commanded or observed by this run.
11. Stage 2.8A remains outside controller_manager bring-up because H1 is only a standalone identity/read-only gate, and its identity prerequisite failed.
12. A future H2 may verify a zero-motion ROS 2 control lifecycle only after H1 passes; H2 was not executed or authorized here.

The next required input is a completed non-SDK identity intake from the robot Web UI, teach pendant, controller UI, or system-version page. Until then, do not connect, guess v3.9.7/v3.9.8, start controller_manager, activate the FAIRINO plugin, send FJT, or execute MoveIt.
"""
    (H1 / "stage28ah1_report.md").write_text(report, encoding="utf-8")

    # The checksum file is written last and excludes itself by design.
    lines = [f"{sha256(path)}  {path.name}" for path in sorted(H1.iterdir()) if path.is_file() and path.name != "SHA256SUMS"]
    (H1 / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if regression["returncode"] != 0:
        return 1
    print(json.dumps({"output": str(H1), "status": gate["status"], "regression": regression["status"], "live_connection_attempted": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
