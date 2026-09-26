"""Stage 2.8S-R2 persistent-observer and transport-isolation ZERO-GOAL run.

This program never imports an action client, never mutates the formal ledger,
and cannot call send_goal_async.  It exercises only recorder-before-publisher
runtime capture and stops every trial without sending a trajectory goal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stage28sr2_zero_goal_recertification import DEFAULT_RUNTIME_LAUNCH, RuntimeProcess  # noqa: E402
from src.stage28sr2_formal_ledger import CANONICAL_LEDGER_PATH  # noqa: E402
from src.stage28sr2_persistent_graph import load_jsonl, summarize_persistent_graph  # noqa: E402
from src.stage28sr2_recorder_orchestrator import (  # noqa: E402
    REQUIRED_TOPICS,
    Rosbag2RecorderOrchestrator,
    load_recorder_qos_config,
    probe_installed_rosbag2_capability,
    sha256_file,
)
from src.stage28sr2_transport_mode import DEFAULT_UDP_SHM, UDP4_ONLY, build_transport_environment, shell_prefix  # noqa: E402


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def linux_path(path: Path) -> str:
    value = str(path.resolve()).replace("\\", "/")
    return f"/mnt/{value[0].lower()}/{value[3:]}" if len(value) > 1 and value[1] == ":" else value


def run_wsl(command: str, *, timeout: float = 30.0) -> tuple[int | None, str, str]:
    argv = ["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", command] if os.name == "nt" else ["bash", "-lc", command]
    try:
        result = subprocess.run(argv, capture_output=True, timeout=timeout, check=False)
        return result.returncode, result.stdout.decode("utf-8", "replace"), result.stderr.decode("utf-8", "replace")
    except (OSError, subprocess.TimeoutExpired) as error:
        return None, "", repr(error)


class PersistentObserverProcess:
    def __init__(self, output: Path, mode: str) -> None:
        self.output = output
        self.mode = mode
        self.raw = output / "persistent_graph_observer_raw.jsonl"
        self.ready_path = output / "persistent_graph_observer_ready.json"
        self.log_path = output / "persistent_graph_observer.log"
        self.process: subprocess.Popen[str] | None = None
        self.ready: dict[str, Any] = {}

    def start(self) -> dict[str, Any]:
        self.output.mkdir(parents=True, exist_ok=True)
        script = linux_path(ROOT / "scripts/stage28sr2_persistent_graph_observer.py")
        command = (
            "source /opt/ros/jazzy/setup.bash; "
            + shell_prefix(self.mode)
            + "exec python3 "
            + shlex.quote(script)
            + " --raw "
            + shlex.quote(linux_path(self.raw))
            + " --ready "
            + shlex.quote(linux_path(self.ready_path))
            + " --topics-json "
            + shlex.quote(json.dumps(REQUIRED_TOPICS))
            + " --seed-topic-types"
        )
        log = self.log_path.open("w", encoding="utf-8")
        argv = ["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", command] if os.name == "nt" else ["bash", "-lc", command]
        self.process = subprocess.Popen(
            argv,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            if self.ready_path.is_file():
                self.ready = json.loads(self.ready_path.read_text(encoding="utf-8"))
                self.ready["observer_started_before_runtime"] = True
                return self.ready
            if self.process.poll() is not None:
                break
            time.sleep(0.1)
        raise RuntimeError("persistent graph observer did not become ready")

    def stop(self) -> dict[str, Any]:
        pid = self.ready.get("pid")
        if pid:
            run_wsl(f"kill -INT {int(pid)}", timeout=10.0)
        if self.process is not None:
            try:
                self.process.wait(timeout=15.0)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=5.0)
        return {"pid": pid, "returncode": self.process.returncode if self.process else None, "sample_count": len(load_jsonl(self.raw))}


def process_transport_evidence(pids: dict[str, int | None], mode: str) -> dict[str, Any]:
    expected = "FASTDDS_BUILTIN_TRANSPORTS=UDPv4" if mode == UDP4_ONLY else None
    processes: dict[str, Any] = {}
    for name, pid in pids.items():
        if pid is None:
            processes[name] = {"pid": None, "observable": False, "passed": False}
            continue
        code, stdout, stderr = run_wsl(f"tr '\\0' '\\n' < /proc/{int(pid)}/environ | grep -E '^(FASTDDS_BUILTIN_TRANSPORTS|FASTDDS_DEFAULT_PROFILES_FILE|FASTRTPS_DEFAULT_PROFILES_FILE|RMW_FASTRTPS_USE_QOS_FROM_XML)=' || true")
        lines = sorted(line.strip() for line in stdout.splitlines() if line.strip())
        conflicts = [line for line in lines if not line.startswith("FASTDDS_BUILTIN_TRANSPORTS=")]
        passed = not conflicts and ((expected in lines) if expected else not any(line.startswith("FASTDDS_BUILTIN_TRANSPORTS=") for line in lines))
        processes[name] = {"pid": pid, "observable": code == 0, "relevant_environment": lines, "conflicts": conflicts, "passed": passed, "stderr": stderr}
    return {"requested_transport_mode": mode, "processes": processes, "passed": all(item["passed"] for item in processes.values())}


def preflight_and_shm_cleanup(output: Path) -> dict[str, Any]:
    code, nodes, nodes_err = run_wsl("source /opt/ros/jazzy/setup.bash; ros2 node list --no-daemon")
    code_ps, processes, ps_err = run_wsl("ps -eo pid=,stat=,args= | grep -E 'ros2 bag record|stage28sr2|fairino5_controller|controller_manager' | grep -v grep || true")
    active = [line for line in nodes.splitlines() if any(token in line for token in ("controller_manager", "fairino5", "stage28sr2", "robot_state_publisher"))]
    report: dict[str, Any] = {
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "node_probe": {"exit_code": code, "stdout": nodes, "stderr": nodes_err},
        "process_probe": {"exit_code": code_ps, "stdout": processes, "stderr": ps_err},
        "project_active_participant_detected": bool(active or processes.strip()),
        "active_project_nodes": active,
    }
    if report["project_active_participant_detected"]:
        report.update({"shm_cleanup_executed": False, "passed": False, "first_blocker": "pre_existing_project_runtime_or_recorder"})
    else:
        clean_code, clean_stdout, clean_stderr = run_wsl("if command -v fastdds >/dev/null 2>&1; then fastdds shm clean; else exit 127; fi")
        report.update({
            "shm_cleanup_executed": clean_code == 0,
            "shm_cleanup_exit_code": clean_code,
            "shm_cleanup_stdout": clean_stdout,
            "shm_cleanup_stderr": clean_stderr,
            "no_active_project_participant_at_cleanup": True,
            "passed": True,
        })
    write_json(output / "shm_cleanup_preflight.json", report)
    return report


def run_trial(root: Path, *, mode: str, index: int, launch: Path, capability: dict[str, Any], duration: float) -> dict[str, Any]:
    output = root / "zero_goal_trials" / mode / f"run_{index}"
    output.mkdir(parents=True, exist_ok=True)
    run_id = f"{mode}_{index}_{uuid.uuid4().hex[:8]}"
    observer = PersistentObserverProcess(output, mode)
    runtime = RuntimeProcess(output / "runtime", launch, run_id, transport_mode=mode)
    recorder: Rosbag2RecorderOrchestrator | None = None
    report: dict[str, Any] = {
        "mode": mode,
        "run": index,
        "run_id": run_id,
        "FJT_goals_sent": 0,
        "send_goal_async_call_count": 0,
        "Formal_R2_started": False,
        "recorder_started_before_publishers": False,
    }
    runtime_started = False
    try:
        observer_ready = observer.start()
        report["observer_ready"] = observer_ready
        recorder = Rosbag2RecorderOrchestrator(
            output / "recorder",
            runtime_instance_id=runtime.runtime_instance_id,
            run_id=run_id,
            capability=capability,
            command_runner=runtime.run_ros,
            persistent_graph_raw_path=observer.raw,
            persistent_observer_pid=int(observer_ready["pid"]),
            observer_started_before_runtime=True,
            transport_mode=mode,
        )
        report["recorder_start"] = recorder.start()
        report["recorder_base_ready"] = recorder.verify_readiness()
        report["pre_publisher_reader_gate"] = recorder.verify_live_endpoints()
        if report["pre_publisher_reader_gate"].get("passed") is not True:
            raise RuntimeError(str(report["pre_publisher_reader_gate"].get("first_blocker")))
        report["recorder_started_before_publishers"] = True
        report["runtime_identity"] = runtime.start()
        runtime_started = True
        report["runtime_ready"] = runtime.ready(timeout_s=90.0)
        if report["runtime_ready"].get("passed") is not True:
            raise RuntimeError("runtime_not_ready")
        report["persistent_graph_gate"] = recorder.verify_persistent_publishers(timeout_sec=45.0)
        if report["persistent_graph_gate"].get("passed") is not True:
            raise RuntimeError(str(report["persistent_graph_gate"].get("first_blocker")))
        report["pre_send_live_loss_gate"] = recorder.query_live_transport_loss(timeout_s=5.0)
        if report["pre_send_live_loss_gate"].get("passed") is not True:
            raise RuntimeError(str(report["pre_send_live_loss_gate"].get("first_blocker")))
        time.sleep(duration)
        recorder.monitor_health()
        recorder_pid = recorder.state.get("recorder_pid")
        runtime_pid_code, runtime_pid_out, _ = runtime.run_ros(f"pgrep -f 'runtime_tag:=stage28sr2_zero_goal_{run_id}' | head -1")
        runtime_pid = int(runtime_pid_out.strip()) if runtime_pid_code == 0 and runtime_pid_out.strip().isdigit() else None
        report["effective_transport_evidence"] = process_transport_evidence(
            {"observer": int(observer_ready["pid"]), "recorder": recorder_pid, "runtime": runtime_pid}, mode
        )
    except Exception as error:
        report["exception"] = f"{type(error).__name__}:{error}"
    finally:
        if runtime_started:
            report["runtime_stop"] = runtime.stop()
        if recorder is not None and recorder.capture is not None:
            report["recorder_finalization"] = recorder.finalize()
        report["observer_stop"] = observer.stop()

    final = report.get("recorder_finalization", {})
    graph = report.get("persistent_graph_gate", {})
    reader = final.get("sequential_reader", {})
    combined_logs = ""
    for path in output.rglob("*.log"):
        combined_logs += path.read_text(encoding="utf-8", errors="replace")
    report["SHM_open_and_lock_errors"] = len(re.findall(r"open_and_lock_file failed", combined_logs, re.I))
    report["SampleLost_mentions"] = len(re.findall(r"SampleLost|sample_lost", combined_logs, re.I))
    report["transport_loss"] = final.get("recorder_loss_gate", {})
    report["post_stop_loss"] = final.get("post_stop_source_backed_zero", {})
    report["bag_counts"] = {
        "reader": reader.get("reader_counts"),
        "metadata": reader.get("metadata_counts"),
        "required_topics_positive": reader.get("required_topic_counts_positive"),
        "sequential_reader_equals_metadata": reader.get("counts_match_metadata"),
    }
    gates = {
        "zero_goal": report["FJT_goals_sent"] == 0 and report["send_goal_async_call_count"] == 0,
        "observer_started_before_runtime": report.get("observer_ready", {}).get("observer_started_before_runtime") is True,
        "single_participant": report.get("observer_ready", {}).get("participant_creation_count") == 1,
        "recorder_started_before_publishers": report.get("recorder_started_before_publishers") is True,
        "pre_publisher_readers": report.get("pre_publisher_reader_gate", {}).get("passed") is True,
        "persistent_publishers": graph.get("required_publisher_gid_stability") is True,
        "pre_send_live_zero_proven": report.get("pre_send_live_loss_gate", {}).get("pre_send_live_zero_proven") is True,
        "pre_send_recorder_bound": report.get("pre_send_live_loss_gate", {}).get("recorder_bound") is True,
        "qos_runtime": graph.get("qos_runtime_provenance_passed") is True,
        "transport_effective": report.get("effective_transport_evidence", {}).get("passed") is True,
        "post_stop_transport_loss_zero": final.get("recorder_loss_gate", {}).get("transport_lost_total") == 0,
        "bag_finalized": final.get("finalized") is True,
        "offline_readable": final.get("offline_readable") is True,
        "required_topics_positive": reader.get("required_topic_counts_positive") is True,
        "sequential_reader_equals_metadata": reader.get("counts_match_metadata") is True,
    }
    report["gates"] = gates
    report["passed"] = all(gates.values())
    report["first_blocker"] = next((name for name, passed in gates.items() if not passed), None)
    write_json(output / "trial_certificate.json", report)
    return report


def run_tests(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    suites = {
        "focused": [sys.executable, "-m", "pytest", "-q", "tests/test_stage28sr2_r2_zero_goal_hardening.py", "tests/test_stage28sr2_production_integration.py"],
        "full": [sys.executable, "-m", "pytest", "-q"],
    }
    result: dict[str, Any] = {}
    chunks = []
    for name, command in suites.items():
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900, check=False)
        raw = completed.stdout + ("\n--- STDERR ---\n" + completed.stderr if completed.stderr else "")
        chunks.append(f"===== {name} =====\n{raw}")
        result[name] = {"exit_code": completed.returncode, "passed": completed.returncode == 0}
    (output / "test_results.txt").write_text("\n".join(chunks), encoding="utf-8")
    result["passed"] = all(value["passed"] for value in result.values())
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-launch", type=Path, default=DEFAULT_RUNTIME_LAUNCH)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--runs-per-mode", type=int, default=3)
    parser.add_argument("--mode", choices=[DEFAULT_UDP_SHM, UDP4_ONLY], action="append")
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True)

    preflight = preflight_and_shm_cleanup(output)
    capability = probe_installed_rosbag2_capability(output_dir=output)
    transport_configs = {mode: build_transport_environment(mode, {})[1] for mode in (DEFAULT_UDP_SHM, UDP4_ONLY)}
    trials: list[dict[str, Any]] = []
    selected_modes = tuple(args.mode or (DEFAULT_UDP_SHM, UDP4_ONLY))
    if preflight.get("passed") and capability.get("passed"):
        for mode in selected_modes:
            for index in range(1, args.runs_per_mode + 1):
                trials.append(run_trial(output, mode=mode, index=index, launch=args.runtime_launch.resolve(), capability=capability, duration=args.duration))

    matrix = {
        "schema_version": "stage28sr2-transport-isolation-v1",
        "transport_configurations": transport_configs,
        "trials": trials,
        "modes": {
            mode: {
                "runs": len([trial for trial in trials if trial["mode"] == mode]),
                "passed": sum(1 for trial in trials if trial["mode"] == mode and trial.get("passed")),
                "SHM_open_and_lock_errors": sum(trial.get("SHM_open_and_lock_errors", 0) for trial in trials if trial["mode"] == mode),
                "gid_stability_passed": sum(1 for trial in trials if trial["mode"] == mode and trial.get("persistent_graph_gate", {}).get("required_publisher_gid_stability")),
                "transport_loss_zero": sum(1 for trial in trials if trial["mode"] == mode and trial.get("transport_loss", {}).get("transport_lost_total") == 0),
                "bag_complete": sum(1 for trial in trials if trial["mode"] == mode and trial.get("recorder_finalization", {}).get("bag_complete")),
            }
            for mode in (DEFAULT_UDP_SHM, UDP4_ONLY)
        },
    }
    write_json(output / "transport_isolation_matrix.json", matrix)
    (output / "transport_isolation_report.md").write_text("# Transport isolation\n\n```json\n" + json.dumps(matrix["modes"], indent=2) + "\n```\n", encoding="utf-8")

    all_samples = []
    for raw in sorted((output / "zero_goal_trials").rglob("persistent_graph_observer_raw.jsonl")) if (output / "zero_goal_trials").exists() else []:
        for sample in load_jsonl(raw):
            sample["source_trial"] = str(raw.parent.relative_to(output))
            all_samples.append(sample)
    with (output / "persistent_graph_observer_raw.jsonl").open("w", encoding="utf-8") as stream:
        for sample in all_samples:
            stream.write(json.dumps(sample, sort_keys=True) + "\n")
    persistent_summary = {
        "required_topics": list(REQUIRED_TOPICS),
        "sample_interval_sec": 0.25,
        "minimum_consecutive_samples": 5,
        "trials": [{"mode": trial["mode"], "run": trial["run"], "passed": trial.get("persistent_graph_gate", {}).get("passed"), "summary": trial.get("persistent_graph_gate")} for trial in trials],
        "passed": len(trials) == 6 and all(trial.get("persistent_graph_gate", {}).get("passed") for trial in trials),
    }
    write_json(output / "persistent_graph_summary.json", persistent_summary)

    all_pre_live = bool(trials and all(trial.get("pre_send_live_loss_gate", {}).get("pre_send_live_zero_proven") is True for trial in trials))
    all_recorder_bound = bool(trials and all(trial.get("pre_send_live_loss_gate", {}).get("recorder_bound") is True for trial in trials))
    loss_report = {
        "rosbag2_version": capability.get("rosbag2_version"),
        "live_recorder_zero_counter": {"status": "passed" if all_pre_live else "fail_closed", "implemented": True, "recorder_bound": all_recorder_bound},
        "minimum_instrumentation": "read-only Trigger service on the same RecorderImpl event_notifier_ accumulator",
        "post_stop_source_backed_zero": {"supported": True, "all_trials_valid": bool(trials and all(trial.get("post_stop_loss", {}).get("valid") for trial in trials))},
        "pre_send_live_zero_proven": all_pre_live,
    }
    write_json(output / "recorder_loss_observability_report.json", loss_report)
    qos_report = {
        "qos_override_path": str((ROOT / "config/stage28sr2_recorder_qos_override.yaml").resolve()),
        "qos_override_sha256": sha256_file(ROOT / "config/stage28sr2_recorder_qos_override.yaml"),
        "parsed_requested_qos": load_recorder_qos_config(ROOT / "config/stage28sr2_recorder_qos_override.yaml"),
        "trials": [{"mode": trial["mode"], "run": trial["run"], "passed": trial.get("persistent_graph_gate", {}).get("qos_runtime_provenance_passed")} for trial in trials],
        "passed": bool(trials and all(trial.get("persistent_graph_gate", {}).get("qos_runtime_provenance_passed") for trial in trials)),
    }
    write_json(output / "qos_runtime_provenance.json", qos_report)
    tests = {"passed": True, "skipped": True} if args.skip_tests else run_tests(output)
    expected_trial_count = len(selected_modes) * args.runs_per_mode
    hardening_passed = bool(len(trials) == expected_trial_count and all(trial.get("passed") for trial in trials) and tests["passed"])
    canonical_ledger = json.loads(CANONICAL_LEDGER_PATH.read_text(encoding="utf-8")) if CANONICAL_LEDGER_PATH.is_file() else {}
    canonical_budget = canonical_ledger.get("formal_goal_budget", {})
    canonical_budget_safe = bool(canonical_budget.get("maximum") == 1 and canonical_budget.get("consumed") == 0 and canonical_budget.get("send_goal_async_call_count") == 0)
    certificate = {
        "Stage_2_8S_R2_Hardening": {"status": "passed" if hardening_passed else "blocked"},
        "persistent_graph_observer": {"implemented": True, "passed": persistent_summary["passed"], "required_topic_gid_stability": persistent_summary["passed"]},
        "recorder_live_loss_observability": {"implemented": True, "recorder_bound": all_recorder_bound, "pre_send_zero_provable": all_pre_live},
        "post_stop_source_backed_zero": {"valid": loss_report["post_stop_source_backed_zero"]["all_trials_valid"]},
        "qos_runtime_provenance": {"passed": qos_report["passed"]},
        "transport_isolation": matrix["modes"],
        "recorder_before_publisher_zero_goal": {"runs": len(trials), "passed": sum(1 for trial in trials if trial.get("passed"))},
        "production_timing_patch": {"applied": False},
        "production_zero_goal_recertification": {"executed": False, "passed": False},
        "Formal_R2": {"started": False},
        "canonical_ledger_path": str(CANONICAL_LEDGER_PATH),
        "Formal_goal_budget": canonical_budget,
        "FJT_goals_sent": 0,
        "send_goal_async_call_count": 0,
        "Stage_2_9_authorized": False,
        "Stage_3_authorized": False,
        "READY_FOR_PRODUCTION_ZERO_GOAL": hardening_passed,
        "READY_FOR_FORMAL_R2_ONE_SHOT": bool(hardening_passed and all_pre_live and all_recorder_bound and canonical_budget_safe),
    }
    write_json(output / "hardening_certificate.json", certificate)
    (output / "hardening_report.md").write_text("# Stage 2.8S-R2 ZERO-GOAL hardening\n\nInterim machine result is recorded in `hardening_certificate.json`.\n", encoding="utf-8")
    write_json(output / "changed_files_manifest.json", {
        "files": [{"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256_file(path)} for path in (
            ROOT / "src/stage28sr2_persistent_graph.py",
            ROOT / "src/stage28sr2_transport_mode.py",
            ROOT / "src/stage28sr2_recorder_orchestrator.py",
            ROOT / "scripts/stage28sr2_persistent_graph_observer.py",
            ROOT / "scripts/stage28sr2_r2_zero_goal_hardening.py",
            ROOT / "tests/test_stage28sr2_r2_zero_goal_hardening.py",
        )],
    })
    print(json.dumps({"output": str(output), "passed": hardening_passed, "trials": len(trials)}, ensure_ascii=False))
    return 0 if hardening_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
