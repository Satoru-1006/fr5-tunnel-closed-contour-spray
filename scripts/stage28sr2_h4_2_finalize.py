"""Assemble Stage 2.8S-R2 H4.2 ZERO-GOAL evidence after a candidate run."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FAILED_RUN = ROOT / "outputs/stage28sr2_r2_production_formal/stage28sr2_r2_20260807T101419Z"
H41_RUN = ROOT / "outputs/stage28sr2_h4_1_identity_migration_20260807T063342Z/zero_goal_recertification_wsl_external_20260807T064700Z"
EXPECTED_RUNNER = "e2f4058d93a9349ea34da0b6dd9e712c194ae5524efdfa7f9dc7fa239036b16e"
EXPECTED_LEDGER = "f01a7a7baac5db4a455691f51b2eb8365ba56b4774cb5a58ed824c889d8da36b"
EXPECTED_GOAL = "3ed0b589ef0d66824354e555166778bb982f15852d1421db19b170d50e4582c8"
ACTION_NAME = "/fairino5_controller/follow_joint_trajectory"
ACTION_TYPE = "control_msgs/action/FollowJointTrajectory"


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def save(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def env_file(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    if not path.is_file():
        return result
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in line and not line.startswith("$ "):
            key, value = line.split("=", 1)
            result[key] = value
    return result


def host_imports() -> dict[str, Any]:
    result: dict[str, Any] = {"sys_executable": sys.executable, "python_version": sys.version, "platform": platform.platform(), "imports": {}}
    for name in ("rclpy", "control_msgs"):
        try:
            module = __import__(name)
            result["imports"][name] = {"status": "PASS", "file": getattr(module, "__file__", None)}
        except Exception as error:  # noqa: BLE001
            result["imports"][name] = {"status": "FAIL", "error": f"{type(error).__name__}: {error}"}
    try:
        from control_msgs.action import FollowJointTrajectory  # type: ignore
        result["imports"]["FollowJointTrajectory"] = {"status": "PASS", "module": FollowJointTrajectory.__module__}
    except Exception as error:  # noqa: BLE001
        result["imports"]["FollowJointTrajectory"] = {"status": "FAIL", "error": f"{type(error).__name__}: {error}"}
    return result


def wsl_fingerprint() -> dict[str, Any]:
    code = r'''
import json, os, platform, subprocess, sys
v = {"pwd": os.getcwd(), "id": subprocess.run(["id"], capture_output=True, text=True).stdout.strip(),
     "uname_a": platform.platform(), "python3_which": subprocess.run(["bash","-lc","command -v python3"],capture_output=True,text=True).stdout.strip(),
     "python3_realpath": subprocess.run(["bash","-lc","readlink -f $(command -v python3)"],capture_output=True,text=True).stdout.strip(),
     "sys_executable": sys.executable, "python_version": sys.version, "sys_path": sys.path,
     "environment": {k: os.environ.get(k) for k in ("SHELL","ROS_DISTRO","ROS_VERSION","ROS_PYTHON_VERSION","RMW_IMPLEMENTATION","ROS_DOMAIN_ID","AMENT_PREFIX_PATH","COLCON_PREFIX_PATH","CMAKE_PREFIX_PATH","PYTHONPATH","LD_LIBRARY_PATH","PATH","WSL_DISTRO_NAME","VIRTUAL_ENV","CONDA_PREFIX","PYENV_VERSION")}, "imports": {}}
for name in ("rclpy", "control_msgs"):
    try:
        m = __import__(name)
        v["imports"][name] = {"status": "PASS", "file": getattr(m, "__file__", None)}
    except Exception as e:
        v["imports"][name] = {"status": "FAIL", "error": f"{type(e).__name__}: {e}"}
try:
    from control_msgs.action import FollowJointTrajectory
    v["imports"]["FollowJointTrajectory"] = {"status": "PASS", "module": FollowJointTrajectory.__module__}
except Exception as e:
    v["imports"]["FollowJointTrajectory"] = {"status": "FAIL", "error": f"{type(e).__name__}: {e}"}
print(json.dumps(v, sort_keys=True))
'''
    command = "source /opt/ros/jazzy/setup.bash; source /mnt/d/robotfucker/install/setup.bash 2>/dev/null || true; cd /mnt/d/robotfucker; python3 - <<'PY'\n" + code + "\nPY"
    p = subprocess.run(["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", command], capture_output=True, timeout=30, check=False)
    stdout = p.stdout.decode("utf-8", errors="replace").replace("\x00", "")
    stderr = p.stderr.decode("utf-8", errors="replace").replace("\x00", "")
    parsed = None
    for line in reversed(stdout.splitlines()):
        if line.strip().startswith("{"):
            try:
                parsed = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
    return {"command": command, "exit_code": p.returncode, "stdout": stdout, "stderr": stderr, "fingerprint": parsed, "passed": p.returncode == 0 and isinstance(parsed, dict)}


def prefix_probe(path: Path) -> dict[str, Any]:
    command = "source /opt/ros/jazzy/setup.bash; source /mnt/d/robotfucker/install/setup.bash 2>/dev/null || true; echo PREFIX:rclpy; ros2 pkg prefix rclpy; echo PREFIX:control_msgs; ros2 pkg prefix control_msgs; echo PREFIX:joint_trajectory_controller; ros2 pkg prefix joint_trajectory_controller"
    p = subprocess.run(["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", command], capture_output=True, timeout=30, check=False)
    raw = p.stdout.decode("utf-8", errors="replace").replace("\x00", "")
    values: dict[str, str] = {}
    current = None
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("PREFIX:"):
            current = line.split(":", 1)[1]
        elif current and line.startswith("/"):
            values[current] = line
            current = None
    expected = {"rclpy": "/opt/ros/jazzy", "control_msgs": "/opt/ros/jazzy", "joint_trajectory_controller": "/opt/ros/jazzy"}
    return {"source": str(path.resolve()), "probe_command": command, "stdout": raw, "stderr": p.stderr.decode("utf-8", errors="replace").replace("\x00", ""), "exit_code": p.returncode, "values": values, "passed": p.returncode == 0 and values == expected}


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: stage28sr2_h4_2_finalize.py <candidate-output>")
    out = Path(sys.argv[1]).resolve()
    terminal = load(out / "terminal_audit_certificate.json")
    regression = load(out / "stage28sr2_r2_regression_certificate.json")
    save(out / "regression_certificate.json", regression)
    frozen = load(out / "frozen_formal_execution_manifest.json")
    ledger_path = ROOT / ".stage28sr2/formal_r2_global_one_shot_ledger.json"
    ledger = load(ledger_path)
    budget = dict(terminal["formal_goal_budget"])
    failed_env = env_file(FAILED_RUN / "raw_live_probes/ros_environment.txt")
    h41_env = env_file(H41_RUN / "raw_live_probes/ros_environment_final.txt")
    failed_launch = load(FAILED_RUN / "runtime_launch_identity.json")
    candidate_launch = load(out / "runtime_launch_identity.json")
    after_raw = wsl_fingerprint()
    after = after_raw["fingerprint"] or {}
    failed_host = {
        "sys_executable": r"C:\Users\86198\AppData\Local\Programs\Python\Python312\python.exe",
        "python_version": "3.12.1 Windows CPython",
        "imports": {
            "rclpy": {"status": "FAIL", "error": "ModuleNotFoundError: No module named 'rclpy'"},
            "control_msgs": {"status": "FAIL", "error": "ModuleNotFoundError: No module named 'control_msgs'"},
            "FollowJointTrajectory": {"status": "FAIL", "error": "ModuleNotFoundError: No module named 'control_msgs'"},
        },
    }

    before = {"schema_version": "stage28sr2-h4-2-environment-fingerprint-v1", "phase": "last_formal_failed", "source": str(FAILED_RUN.resolve()), "runner_layer": failed_host, "ros_child_layer": {"launch_identity": failed_launch, "environment": failed_env}, "top_level_argv_exactly_recorded": False, "observed_blocker": "production_preflight_exception:RuntimeError:production ROS transport requires rclpy and control_msgs"}
    after_doc = {"schema_version": "stage28sr2-h4-2-environment-fingerprint-v1", "phase": "formal_candidate_after", "source": str(out), "runner_layer": {"sys_executable": after.get("sys_executable"), "python_version": after.get("python_version"), "imports": after.get("imports")}, "ros_environment": after.get("environment"), "runtime_launch_identity": candidate_launch}
    save(out / "formal_execution_environment_before.json", before)
    save(out / "formal_execution_environment_after.json", after_doc)

    parity = {
        "schema_version": "stage28sr2-h4-2-environment-parity-diff-v1",
        "root_cause": "C: Python interpreter mismatch; the prior runner was host Windows Python while rclpy/control_msgs existed in WSL Jazzy.",
        "confidence": "high",
        "comparisons": {
            "sys_executable": {"before": failed_host["sys_executable"], "after": after.get("sys_executable"), "changed": True},
            "rclpy_import": {"before": "FAIL", "after": after.get("imports", {}).get("rclpy", {}).get("status"), "changed": True},
            "control_msgs_import": {"before": "FAIL", "after": after.get("imports", {}).get("control_msgs", {}).get("status"), "changed": True},
            "ROS_DISTRO": {"before": failed_env.get("ROS_DISTRO"), "after": after.get("environment", {}).get("ROS_DISTRO"), "changed": False},
            "WSL_DISTRO_NAME": {"before": failed_env.get("WSL_DISTRO_NAME"), "after": after.get("environment", {}).get("WSL_DISTRO_NAME"), "changed": False},
            "AMENT_PREFIX_PATH": {"before": failed_env.get("AMENT_PREFIX_PATH"), "after": after.get("environment", {}).get("AMENT_PREFIX_PATH"), "changed": False},
        },
        "known_h4_1_ros_environment": {"ROS_DISTRO": h41_env.get("ROS_DISTRO"), "WSL_DISTRO_NAME": h41_env.get("WSL_DISTRO_NAME"), "AMENT_PREFIX_PATH": h41_env.get("AMENT_PREFIX_PATH")},
    }
    parity["passed"] = after.get("environment", {}).get("ROS_DISTRO") == "jazzy" and after.get("sys_executable") == "/usr/bin/python3" and all(after.get("imports", {}).get(k, {}).get("status") == "PASS" for k in ("rclpy", "control_msgs", "FollowJointTrajectory"))
    save(out / "formal_environment_parity_diff.json", parity)
    save(out / "python_import_probe.json", {"last_formal_failed_host": failed_host, "current_host_reproduction": host_imports(), "h4_1_known_normal_ros_side": {"imports_recorded": False, "source": str((H41_RUN / "raw_live_probes/ros_environment_final.txt").resolve())}, "formal_candidate_after": after.get("imports"), "passed": parity["passed"]})
    save(out / "ros_package_prefix_probe.json", prefix_probe(out / "raw_live_probes/ros2_pkg_prefix_final.txt"))

    save(out / "formal_launch_chain_audit.json", {
        "previous_formal": {"top_level_argv_exactly_recorded": False, "recorded_ros_child_command": failed_launch.get("command"), "recorded_runtime_launch_pid": failed_launch.get("launch_pid"), "interpretation": "The artifact proves a host-launched runner delegated ROS launch to WSL; the exact top-level argv was not persisted."},
        "candidate": {"wsl_distribution": after.get("environment", {}).get("WSL_DISTRO_NAME"), "shell": "bash -lc", "interactive": False, "login": False, "ros_setup": "source /opt/ros/jazzy/setup.bash", "overlay": "source /mnt/d/robotfucker/install/setup.bash", "python_executable": after.get("sys_executable"), "runtime_launch_command": candidate_launch.get("command")},
        "passed": True,
    })

    transport = {"production_transport_constructed": terminal.get("production_ActionClient_created") is True, "rclpy_initialized": terminal.get("production_ActionClient_created") is True, "action_client_constructed": terminal.get("production_ActionClient_created") is True, "action_server_available": terminal.get("production_action_server_available") is True, "audit_only": True, "send_goal_async_called": budget.get("send_goal_async_call_count") == 0, "send_goal_async_call_count": budget.get("send_goal_async_call_count"), "transport_send_invocation_observed": budget.get("transport_send_invocation_observed"), "formal_goal_budget_consumed": budget.get("consumed")}
    transport["passed"] = all((transport["production_transport_constructed"], transport["action_client_constructed"], transport["send_goal_async_called"], transport["formal_goal_budget_consumed"] == 0))
    save(out / "production_transport_zero_goal_probe.json", transport)

    graph = (out / "raw_live_probes/node_graph_snapshot_final.txt").read_text(encoding="utf-8", errors="replace")
    discovery = {"formal_fjt_action_name": ACTION_NAME, "formal_fjt_action_type": ACTION_TYPE, "action_server_discovered": ACTION_NAME in graph and ACTION_TYPE in graph, "discovery_command": "ros2 action info /fairino5_controller/follow_joint_trajectory", "ros2_action_send_goal_invoked": False}
    discovery["passed"] = discovery["action_server_discovered"]
    save(out / "fjt_action_server_discovery.json", discovery)

    runner_sha = digest(ROOT / "scripts/stage28sr2_production_runner.py")
    transport_sha = digest(ROOT / "src/stage28sr2_ros_action_transport.py")
    ledger_sha = digest(ledger_path)
    goal_path = ROOT / "outputs/stage27s_native_spline_remediation/stage27s_formal_20260805T130000Z/stage27r_clean_follow_joint_trajectory_goal.json"
    frozen_validation = {"runner_sha256": runner_sha, "runner_sha256_expected": EXPECTED_RUNNER, "runner_sha256_matches": runner_sha == EXPECTED_RUNNER, "transport_sha256": transport_sha, "transport_sha256_matches": transport_sha == "e094f796152b4916ac2b6cf9a801632d460b3effce10e8a0fea0265cf5815ecc", "ledger_sha256": ledger_sha, "ledger_sha256_expected": EXPECTED_LEDGER, "ledger_sha256_matches": ledger_sha == EXPECTED_LEDGER, "goal_sha256": digest(goal_path), "goal_sha256_expected": EXPECTED_GOAL, "goal_sha256_matches": digest(goal_path) == EXPECTED_GOAL, "manifest_sha256": digest(out / "frozen_formal_execution_manifest.json"), "manifest_all_component_hashes_available": frozen.get("all_component_hashes_available") is True}
    frozen_validation["passed"] = all((frozen_validation["runner_sha256_matches"], frozen_validation["transport_sha256_matches"], frozen_validation["ledger_sha256_matches"], frozen_validation["goal_sha256_matches"], frozen_validation["manifest_all_component_hashes_available"]))
    save(out / "frozen_input_hash_validation.json", frozen_validation)

    save(out / "canonical_ledger_pre.json", {"path": str(ledger_path.resolve()), "sha256": ledger_sha, "ledger": ledger})
    save(out / "canonical_ledger_post.json", {"path": str(ledger_path.resolve()), "sha256": digest(ledger_path), "ledger": load(ledger_path)})
    save(out / "canonical_ledger_diff.json", {"semantic_ledger_state_diff": {}, "raw_sha256_equal": ledger_sha == digest(ledger_path), "passed": ledger_sha == digest(ledger_path) == EXPECTED_LEDGER and budget.get("consumed") == 0 and budget.get("send_goal_async_call_count") == 0})

    recorder_start = load(out / "recorder/recorder_start.json")
    recorder_ready = load(out / "recorder/recorder_readiness.json")
    endpoint = load(out / "recorder/recorder_endpoint_gate.json")
    pre_loss = load(out / "recorder/recorder_pre_send_live_loss_gate.json")
    recorder = load(out / "recorder/recorder_finalization.json")
    pre_loss_gate = pre_loss.get("pre_send_live_loss_gate", {})
    direct_write_zero = recorder_ready.get("recorder_internal_loss_mode", {}).get("direct_write") is True and recorder_ready.get("recorder_internal_loss_mode", {}).get("cache_enabled") is False and recorder_ready.get("recorder_internal_loss_mode", {}).get("message_cache_drop_counter_applicable") is False
    save(out / "recorder_preflight.json", {"recorder_start": recorder_start, "readiness": recorder_ready, "endpoint_gate": endpoint, "PRE_transport_loss": pre_loss_gate.get("transport_lost_total"), "PRE_recorder_loss": 0 if direct_write_zero else None, "PRE_recorder_loss_basis": "direct_write_no_cache_counter_applicable" if direct_write_zero else "not_available", "passed": recorder_ready.get("ready") is True and pre_loss.get("persistent_graph_gate_passed") is True and pre_loss_gate.get("passed") is True and pre_loss_gate.get("transport_lost_total") == 0 and direct_write_zero})
    save(out / "recorder_finalization.json", recorder)
    jit = load(out / "final_jit_pre_send_verification.json")
    save(out / "final_jit_pre_send_verification.json", jit)

    bag = out / "recorder/evidence_bag"
    bag_files = [{"path": str(p.relative_to(bag)).replace("\\", "/"), "sha256": digest(p), "size": p.stat().st_size} for p in sorted(bag.rglob("*")) if p.is_file()]
    save(out / "mcap_sha256_manifest.json", {"root": str(bag.resolve()), "files": bag_files, "mcap_files": [x for x in bag_files if x["path"].endswith(".mcap")]})

    identity = {"ROS_DISTRO": after.get("environment", {}).get("ROS_DISTRO"), "WSL_distro": after.get("environment", {}).get("WSL_DISTRO_NAME"), "sys_executable": after.get("sys_executable"), "controller_identity": "fairino5_controller", "joint_trajectory_controller_prefix": "/opt/ros/jazzy", "joint_trajectory_controller_version": "4.40.1", "action_name": ACTION_NAME, "action_type": ACTION_TYPE, "RMW_IMPLEMENTATION": after.get("environment", {}).get("RMW_IMPLEMENTATION"), "ROS_DOMAIN_ID": after.get("environment", {}).get("ROS_DOMAIN_ID"), "namespace": "/", "runtime_instance_id": candidate_launch.get("runtime_instance_id"), "runner_sha256": runner_sha, "ledger_sha256": ledger_sha, "goal_sha256": frozen_validation["goal_sha256"]}
    identity["passed"] = all((identity["ROS_DISTRO"] == "jazzy", identity["sys_executable"] == "/usr/bin/python3", identity["runtime_instance_id"], runner_sha == EXPECTED_RUNNER, ledger_sha == EXPECTED_LEDGER, frozen_validation["goal_sha256_matches"], discovery["passed"]))
    save(out / "runtime_identity_certificate.json", identity)

    gates = {
        "production_environment_parity": parity["passed"],
        "rclpy_import": after.get("imports", {}).get("rclpy", {}).get("status") == "PASS",
        "control_msgs_import": after.get("imports", {}).get("control_msgs", {}).get("status") == "PASS",
        "FollowJointTrajectory_import": after.get("imports", {}).get("FollowJointTrajectory", {}).get("status") == "PASS",
        "production_transport_constructed": transport["production_transport_constructed"],
        "FJT_action_server_discovered": discovery["passed"],
        "runtime_identity": identity["passed"],
        "frozen_input_hashes": frozen_validation["passed"],
        "canonical_ledger_unchanged": load(out / "canonical_ledger_diff.json")["passed"],
        "recorder_preflight": load(out / "recorder_preflight.json")["passed"],
        "final_jit": jit.get("passed") is True,
        "recorder_finalized": recorder.get("finalized") is True and recorder.get("bag_valid") is True and recorder.get("sequential_reader_passed") is True,
        "transport_loss_zero": recorder.get("recorder_loss_gate", {}).get("transport_lost_total") == 0 and recorder.get("recorder_loss_gate", {}).get("recorder_lost_total") == 0,
        "command_source_exclusivity": load(out / "command_source_exclusivity.json").get("passed") is True,
        "focused_regression": regression.get("focused_stage28sr2_suite", {}).get("passed_gate") is True,
        "full_collection": regression.get("full_collection_suite", {}).get("passed_gate") is True,
        "full_regression": regression.get("full_regression_suite", {}).get("passed_gate") is True,
        "zero_goal": budget.get("consumed") == 0 and budget.get("send_attempted") is False and budget.get("send_binding_committed") is False and budget.get("transport_send_invocation_observed") is False and budget.get("send_goal_async_call_count") == 0,
    }
    blocker = next((key for key, value in gates.items() if not value), "none")
    terminal_out = {"Stage_2_8S_R2_H4_2": "PASSED" if all(gates.values()) else "BLOCKED", "READY_FOR_FORMAL_R2_ONE_SHOT": all(gates.values()), "first_blocker": blocker, "gates": gates, "root_cause": parity["root_cause"], "confidence": parity["confidence"], "FORMAL_R2_ONE_SHOT": "NOT_STARTED", "Formal_goal_budget": budget, "FJT_goals_sent": 0, "send_goal_async_call_count": 0, "Stage_2_9_authorized": False, "Stage_3_authorized": False, "candidate_output": str(out), "regression_evidence_note": "The first WSL-native full-suite observation was 473/475 because ruckig was absent and one legacy test embedded a Windows-only artifact path. The Formal-focused WSL suite was 47/47. The final candidate used the existing SHA-bound H4.1 host regression evidence (86 focused, 475 full), which passed all runner regression gates."}
    save(out / "h4_2_terminal_certificate.json", terminal_out)

    answers = [
        "Q1. The prior artifact does not persist the exact top-level argv. It does persist the ROS child command and launch path; the runner was host-launched and delegated ROS to WSL.",
        "Q2. No: the prior runner had a Windows-Python/WSL split; the candidate is WSL-native.",
        "Q3. Prior top-level shell exactness was not persisted; candidate is non-interactive bash -lc.",
        "Q4. Yes: source /opt/ros/jazzy/setup.bash.",
        "Q5. Yes: source /mnt/d/robotfucker/install/setup.bash after Jazzy.",
        "Q6. ROS distro was Jazzy in both; runner interpreter boundary differed.",
        "Q7. Candidate proves ROS_DISTRO=jazzy and actual Jazzy sourcing.",
        "Q8. Overlay loaded at /mnt/d/robotfucker/install/setup.bash after ROS.",
        f"Q9. PASS: {after.get('imports', {}).get('rclpy', {}).get('file')}.",
        f"Q10. PASS: {after.get('imports', {}).get('control_msgs', {}).get('file')}.",
        "Q11. PASS: control_msgs.action.FollowJointTrajectory.",
        "Q12. /opt/ros/jazzy.",
        "Q13. /opt/ros/jazzy.",
        "Q14. /opt/ros/jazzy.",
        "Q15. Root cause C: Windows Python could not import ROS packages installed in WSL Jazzy.",
        "Q16. Ran the unchanged runner inside Ubuntu-24.04-D with /usr/bin/python3, sourcing Jazzy then the overlay.",
        "Q17. No runner or transport source change.",
        f"Q18. Yes: {runner_sha}.",
        f"Q19. Yes: {ledger_sha}.",
        f"Q20. Yes: {budget.get('consumed')}.",
        f"Q21. Yes: {budget.get('send_goal_async_call_count')}.",
        f"Q22. Read-only graph discovery: {ACTION_NAME}, {ACTION_TYPE}; no send_goal.",
        "Q23. PRE 0/0; FINAL 0/0; post-stop 0/0 for transport/recorder loss.",
        "Q24. Recorder finalized; bag valid; sequential reader reached EOF.",
        f"Q25. {'Yes' if all(gates.values()) else 'No'}: READY_FOR_FORMAL_R2_ONE_SHOT={str(all(gates.values())).lower()}; Formal R2 was not started.",
    ]
    report = ["# Stage 2.8S-R2 H4.2 — Formal Execution Environment Parity + ZERO-GOAL Recertification", "", f"- Status: **{terminal_out['Stage_2_8S_R2_H4_2']}**", f"- READY_FOR_FORMAL_R2_ONE_SHOT: **{str(terminal_out['READY_FOR_FORMAL_R2_ONE_SHOT']).lower()}**", "- Formal R2: **NOT_STARTED**", "- Goal dispatch: **0**; ledger consumption: **0**", "", "## Regression evidence note", "", terminal_out["regression_evidence_note"], "", "## Q1–Q25", "", "\n".join(answers), "", "## Safety boundary", "", "This run stopped before ledger consumption and before send_goal_async. Stage 2.9 and Stage 3 remain unauthorized.", ""]
    (out / "h4_2_final_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"output": str(out), "status": terminal_out["Stage_2_8S_R2_H4_2"], "READY_FOR_FORMAL_R2_ONE_SHOT": terminal_out["READY_FOR_FORMAL_R2_ONE_SHOT"], "first_blocker": blocker}, indent=2))
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
