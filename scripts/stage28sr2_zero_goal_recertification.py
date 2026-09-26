"""Fresh zero-goal Stage 2.8S-R2 recorder remediation and recertification.

The runner starts only the simulation runtime and independent rosbag2 recorder
processes.  It does not import a ROS action client, build an FJT goal, consume
the formal ledger, or call ``send_goal_async``.  A readiness result is an
evidence conclusion only; this command always stops before the formal shot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import shutil
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

from src.stage28sr2_formal_ledger import CANONICAL_LEDGER_PATH, FormalGoalLedger  # noqa: E402
from src.stage28sr2_recorder_orchestrator import (  # noqa: E402
    REQUIRED_TOPICS,
    Rosbag2RecorderOrchestrator,
    probe_installed_rosbag2_capability,
    sha256_file,
)
from src.stage28sr2_transport_mode import DEFAULT_UDP_SHM, shell_prefix  # noqa: E402


DEFAULT_RUNTIME_LAUNCH = ROOT / "tmp/stage28s_preflight_20260805/stage28s_runtime.launch.py"
REQUIRED_SOURCE_FILES = (
    "scripts/stage28sr2_production_runner.py",
    "scripts/stage28sr2_formal_runner.py",
    "src/stage28sr2_formal_ledger.py",
    "src/stage28sr2_recorder_orchestrator.py",
    "src/stage28sr2_ros_action_transport.py",
    "src/stage28sr2_ros_goal.py",
    "src/stage28sr2_live_runtime_evidence.py",
    "src/stage28sr2_command_source_collector.py",
    "tests/test_stage28sr2_production_integration.py",
    "tests/test_stage28sr2_r2_formal_runner.py",
    "config/stage28sr2_recorder_qos_override.yaml",
)


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def manifest_source_paths() -> list[Path]:
    paths = {ROOT / relative for relative in REQUIRED_SOURCE_FILES}
    for pattern in ("scripts/stage28sr2_*.py", "src/stage28sr2_*.py", "tests/test_stage28sr2_*.py"):
        paths.update(ROOT.glob(pattern))
    paths.add(ROOT / "pytest.ini")
    missing = [path for path in sorted(paths) if not path.is_file()]
    if missing:
        raise FileNotFoundError("source manifest missing: " + ", ".join(str(path) for path in missing))
    return sorted(paths)


def build_source_manifest() -> dict[str, Any]:
    files = []
    for path in manifest_source_paths():
        files.append(
            {
                "relative_path": path.relative_to(ROOT).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {"schema_version": "stage28sr2-source-freeze-v1", "root": str(ROOT), "file_count": len(files), "files": files}


def freeze_sources(output: Path) -> tuple[dict[str, Any], Path]:
    manifest = build_source_manifest()
    path = output / "source_manifest_pre_test.json"
    write_json(path, manifest)
    return manifest, path


def copy_manifest(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def compare_manifest_bytes(source: Path, checkpoint_paths: list[Path]) -> dict[str, Any]:
    source_bytes = source.read_bytes()
    mismatches = []
    checks = {}
    for path in checkpoint_paths:
        identical = path.is_file() and path.read_bytes() == source_bytes
        checks[path.name] = identical
        if not identical:
            mismatches.append(str(path.resolve()))
    return {"passed": not mismatches, "manifest_file_count": len(json.loads(source_bytes.decode("utf-8")).get("files", [])), "checks": checks, "mismatches": mismatches}


def archive_source_tree(output: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    archive_root = output / "source_archive"
    archive_root.mkdir(parents=True, exist_ok=True)
    copied = []
    mismatches = []
    for entry in manifest["files"]:
        source = ROOT / entry["relative_path"]
        destination = archive_root / entry["relative_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        observed = {"relative_path": entry["relative_path"], "size": destination.stat().st_size, "sha256": sha256_file(destination)}
        copied.append(observed)
        if observed != entry:
            mismatches.append(entry["relative_path"])
    return {"archive_root": str(archive_root.resolve()), "file_count": len(copied), "mismatches": mismatches, "passed": not mismatches}


def pytest_counts(raw: str, *, collection_only: bool = False) -> dict[str, int | None]:
    def number(pattern: str) -> int:
        match = re.search(pattern, raw, re.I)
        return int(match.group(1)) if match else 0

    collected_match = re.search(r"(?P<count>\d+)\s+tests?\s+collected\b", raw, re.I)
    passed = number(r"\b(\d+)\s+passed\b")
    failed = number(r"\b(\d+)\s+failed\b")
    errors = number(r"\b(\d+)\s+errors?\b")
    skipped = number(r"\b(\d+)\s+skipped\b")
    collected = int(collected_match.group("count")) if collected_match else None
    if collected is None and not collection_only and (passed or failed or errors or skipped):
        collected = passed + failed + errors + skipped
    return {"collected": collected, "passed": passed, "failed": failed, "errors": errors, "skipped": skipped}


def run_pytest(output: Path, name: str, args: list[str], *, collection_only: bool = False) -> dict[str, Any]:
    completed = subprocess.run([sys.executable, "-m", "pytest", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900, check=False)
    raw = completed.stdout + ("\n--- STDERR ---\n" + completed.stderr if completed.stderr else "")
    log = output / f"{name}.log"
    log.write_text(raw, encoding="utf-8")
    result = {
        **pytest_counts(raw, collection_only=collection_only),
        "executed": True,
        "exit_code": completed.returncode,
        "log_path": str(log.resolve()),
        "log_sha256": sha256_file(log),
    }
    result["passed_gate"] = bool(result["exit_code"] == 0 and result["collected"] is not None and result["failed"] == 0 and result["errors"] == 0)
    if not result["passed_gate"]:
        result["first_blocker"] = f"pytest_{name}_failed"
    return result


class RuntimeProcess:
    def __init__(self, output: Path, launch: Path, run_id: str, *, transport_mode: str = DEFAULT_UDP_SHM) -> None:
        self.output = output
        self.launch = launch.resolve()
        self.run_id = run_id
        self.process: subprocess.Popen[str] | None = None
        self.runtime_instance_id = str(uuid.uuid4())
        self.command: list[str] = []
        self.transport_mode = transport_mode

    @staticmethod
    def linux_path(path: Path) -> str:
        value = str(path.resolve()).replace("\\", "/")
        return f"/mnt/{value[0].lower()}/{value[3:]}" if len(value) > 1 and value[1] == ":" else value

    def run_ros(self, command: str, timeout_s: float = 30.0) -> tuple[int | None, str, str]:
        argv = [
            "wsl.exe",
            "-d",
            "Ubuntu-24.04-D",
            "--",
            "bash",
            "-lc",
            f"source /opt/ros/jazzy/setup.bash; source /mnt/d/robotfucker/install/setup.bash 2>/dev/null || true; {shell_prefix(self.transport_mode)}{command}",
        ]
        try:
            completed = subprocess.run(argv, capture_output=True, timeout=timeout_s, check=False)
            return completed.returncode, completed.stdout.decode("utf-8", errors="replace"), completed.stderr.decode("utf-8", errors="replace")
        except (OSError, subprocess.TimeoutExpired) as error:
            return None, "", repr(error)

    def start(self) -> dict[str, Any]:
        if not self.launch.is_file():
            raise FileNotFoundError(self.launch)
        self.output.mkdir(parents=True, exist_ok=True)
        launch_linux = self.linux_path(self.launch)
        command = f"source /opt/ros/jazzy/setup.bash; source /mnt/d/robotfucker/install/setup.bash 2>/dev/null || true; {shell_prefix(self.transport_mode)}exec ros2 launch {launch_linux} runtime_tag:=stage28sr2_zero_goal_{self.run_id}"
        self.command = ["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", command]
        log = (self.output / "runtime.log").open("w", encoding="utf-8")
        self.process = subprocess.Popen(self.command, stdout=log, stderr=subprocess.STDOUT, text=True)
        identity = {
            "schema_version": "stage28sr2-zero-goal-runtime-identity-v1",
            "runtime_instance_id": self.runtime_instance_id,
            "host_launcher_pid": self.process.pid,
            "run_id": self.run_id,
            "command": command,
            "launch_path": str(self.launch),
            "restart_count": 0,
            "controller_restart_count": 0,
            "controller_reactivation_count": 0,
        }
        write_json(self.output / "runtime_identity.json", identity)
        return identity

    def ready(self, timeout_s: float = 90.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        last: dict[str, Any] = {}
        while time.monotonic() < deadline:
            code, stdout, stderr = self.run_ros("ros2 control list_controllers; ros2 topic list", timeout_s=20.0)
            last = {"exit_code": code, "stdout": stdout, "stderr": stderr}
            if code == 0 and "fairino5_controller" in stdout and "active" in stdout and all(topic in stdout for topic in REQUIRED_TOPICS):
                last["passed"] = True
                last["observed_from_live_graph"] = True
                write_json(self.output / "runtime_ready.json", last)
                return last
            if self.process is not None and self.process.poll() is not None:
                break
            time.sleep(1.0)
        last["passed"] = False
        last["first_blocker"] = "zero_goal_runtime_not_ready"
        write_json(self.output / "runtime_ready.json", last)
        return last

    def stop(self) -> dict[str, Any]:
        if self.process is None:
            return {"started": False}
        try:
            self.process.send_signal(signal.SIGINT)
            self.process.wait(timeout=45)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
        result = {"started": True, "pid": self.process.pid, "returncode": self.process.returncode}
        write_json(self.output / "runtime_stop.json", result)
        return result


def run_one_recorder(output: Path, runtime: RuntimeProcess, capability: dict[str, Any], run_number: int, stability_s: float) -> dict[str, Any]:
    run_id = f"{runtime.run_id}_run_{run_number}"
    recorder = Rosbag2RecorderOrchestrator(
        output,
        runtime_instance_id=runtime.runtime_instance_id,
        run_id=run_id,
        capability=capability,
        command_runner=runtime.run_ros,
    )
    report: dict[str, Any] = {
        "run_number": run_number,
        "run_id": run_id,
        "FJT_goals_sent": 0,
        "send_goal_async_call_count": 0,
        "formal_ledger_consumed": 0,
        "max_cache_size": 0,
        "active_recording_filesystem": "linux_native",
    }
    try:
        started = recorder.start()
        readiness = recorder.verify_readiness()
        report["recorder_start"] = started
        report["recorder_pre_endpoint_readiness"] = readiness
        if readiness.get("ready") is not True:
            report["first_blocker"] = "recorder_base_readiness_failed"
            return report
        endpoint_gate = recorder.verify_live_endpoints()
        report["endpoint_gate"] = endpoint_gate
        if endpoint_gate.get("passed") is not True:
            report["first_blocker"] = endpoint_gate.get("first_blocker") or "live_recorder_endpoint_gate_failed"
            return report
        stability_start = time.monotonic()
        health_samples = []
        while time.monotonic() - stability_start < stability_s:
            health_samples.append(recorder.monitor_health())
            if health_samples[-1].get("process_alive") is not True:
                report["first_blocker"] = "recorder_died_during_stability_window"
                return report
            time.sleep(min(1.0, max(0.05, stability_s - (time.monotonic() - stability_start))))
        report["stability_window"] = {"duration_s": time.monotonic() - stability_start, "health_samples": health_samples, "passed": True}
        report["pre_send_live_loss_gate"] = recorder.query_live_transport_loss(timeout_s=5.0)
        final = recorder.finalize()
        report["recorder_finalization"] = final
        loss_gate = final.get("recorder_loss_gate", {})
        report["transport_lost_total"] = loss_gate.get("transport_lost_total")
        report["recorder_lost_total"] = loss_gate.get("recorder_lost_total")
        report["sequential_reader"] = final.get("sequential_reader", {})
        report["mcap_counts"] = {
            "reader": final.get("sequential_reader", {}).get("reader_counts"),
            "metadata": final.get("sequential_reader", {}).get("metadata_counts"),
            "reader_total": final.get("sequential_reader", {}).get("reader_total_count"),
            "metadata_total": final.get("sequential_reader", {}).get("metadata_total_count"),
        }
        report["source_hashes"] = {
            "recorder_start_sha256": sha256_file(output / "recorder_start.json"),
            "endpoint_gate_sha256": sha256_file(output / "recorder_endpoint_gate.json"),
            "reader_result_sha256": sha256_file(output / "recorder_finalization.json"),
            "copy_sha256": sha256_file(output / "bag_copy_sha256.json"),
        }
        run_gates = {
            "pre_send_live_zero_proven": report["pre_send_live_loss_gate"].get("pre_send_live_zero_proven") is True,
            "pre_send_recorder_bound": report["pre_send_live_loss_gate"].get("recorder_bound") is True,
            "transport_loss_zero": loss_gate.get("transport_lost_total") == 0,
            "loss_gate_passed": loss_gate.get("passed") is True,
            "sequential_reader_passed": final.get("sequential_reader_passed") is True,
            "all_required_topics_positive": final.get("sequential_reader", {}).get("required_topic_counts_positive") is True,
            "endpoint_gate_passed": final.get("endpoint_gate_passed") is True,
            "direct_write_verified": final.get("direct_write_verified") is True,
            "linux_native_filesystem_verified": final.get("native_filesystem_verified") is True,
            "byte_identical_evidence_copy": final.get("evidence_bag_byte_identical") is True,
            "no_goal_activity": report["FJT_goals_sent"] == 0 and report["send_goal_async_call_count"] == 0 and report["formal_ledger_consumed"] == 0,
        }
        report["gates"] = run_gates
        report["passed"] = all(run_gates.values())
        if not report["passed"]:
            report["first_blocker"] = next((name for name, passed in run_gates.items() if not passed), "zero_goal_run_failed")
    except Exception as error:
        report["passed"] = False
        report["first_blocker"] = f"zero_goal_run_exception:{type(error).__name__}:{error}"
    finally:
        if recorder.capture is not None and recorder.state.get("finalized") is not True:
            final = recorder.finalize()
            report.setdefault("recorder_finalization", final)
            report.setdefault("transport_lost_total", final.get("recorder_loss_gate", {}).get("transport_lost_total"))
            report.setdefault("sequential_reader", final.get("sequential_reader", {}))
        write_json(output / "zero_goal_run_report.json", report)
    return report


def build_certificate(output: Path, runtime: RuntimeProcess, capability: dict[str, Any], runs: list[dict[str, Any]], regression: dict[str, Any], source_binding: dict[str, Any], budget: dict[str, Any], runtime_identity_gate: dict[str, Any]) -> dict[str, Any]:
    run_passed = len(runs) == 3 and all(run.get("passed") is True for run in runs)
    regression_passed = regression.get("passed") is True
    budget_safe = bool(budget.get("maximum") == 1 and budget.get("consumed") == 0 and budget.get("send_attempted") is False and budget.get("send_goal_async_call_count") == 0)
    pre_send_live_zero_proven = bool(runs and all(run.get("pre_send_live_loss_gate", {}).get("pre_send_live_zero_proven") is True and run.get("pre_send_live_loss_gate", {}).get("recorder_bound") is True for run in runs))
    all_gates = {
        "focused_tests": bool(regression.get("focused", {}).get("passed_gate")),
        "full_regression": bool(regression.get("full", {}).get("passed_gate")),
        "source_hash_binding": source_binding.get("passed") is True,
        "runtime_identity": runtime_identity_gate.get("passed") is True,
        "recorder_capability": capability.get("passed") is True,
        "zero_goal_runs": run_passed,
        "pre_send_live_zero_proven": pre_send_live_zero_proven,
        "formal_goal_budget_zero": budget_safe,
    }
    ready = bool(all(all_gates.values()) and regression_passed and budget_safe)
    first_blocker = next((name for name, passed in all_gates.items() if not passed), "none")
    certificate = {
        "schema_version": "stage28sr2-zero-goal-recertification-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "Stage_2_8S_R2_zero_goal_recertification": {"status": "passed" if ready else f"blocked_{first_blocker}"},
        "rosbag2": {
            "version": capability.get("rosbag2_version"),
            "max_cache_size": 0,
            "cache_enabled": False,
            "direct_write_verified": capability.get("max_cache_size_zero_direct_write_supported") is True,
            "stock_recorder_side_machine_readable_zero_counter": capability.get("stock_recorder_side_machine_readable_zero_counter"),
        },
        "zero_goal_runs": {
            f"run_{index}": {
                "transport_lost_total": run.get("transport_lost_total"),
                "sequential_reader": run.get("sequential_reader", {}).get("passed") is True,
                "endpoint_gate": run.get("gates", {}).get("endpoint_gate_passed") is True,
                "status": "passed" if run.get("passed") is True else "blocked",
                "first_blocker": run.get("first_blocker"),
            }
            for index, run in enumerate(runs, 1)
        },
        "source_hash_binding": source_binding,
        "recorder_runtime_identity": runtime_identity_gate,
        "formal_goal_budget": budget,
        "pre_send_live_zero_proven": pre_send_live_zero_proven,
        "focused_tests": regression.get("focused"),
        "full_regression": regression.get("full"),
        "READY_FOR_FORMAL_R2_ONE_SHOT": ready,
        "Can_we_send_the_formal_R2_goal_in_this_run": False,
        "Stage_2_9_authorized": False,
        "Stage_3_authorized": False,
        "formal_goal_send_attempted": False,
        "send_goal_async_call_count": 0,
        "first_blocker": first_blocker,
        "runtime_stop_recorded": (output / "runtime_stop.json").is_file(),
    }
    write_json(output / "stage28sr2_zero_goal_recertification_certificate.json", certificate)
    return certificate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 2.8S-R2 fresh zero-goal recorder recertification")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--runtime-launch", type=Path, default=DEFAULT_RUNTIME_LAUNCH)
    parser.add_argument("--stability-seconds", type=float, default=5.0)
    args = parser.parse_args(argv)
    if args.stability_seconds <= 0:
        raise SystemExit("--stability-seconds must be positive")
    output = (args.output or ROOT / "outputs/stage28sr2_r2_zero_goal_recertification" / f"stage28sr2_r2_{utc_stamp()}").resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest, pre_manifest_path = freeze_sources(output)
    checkpoint_paths: list[Path] = []
    ledger = FormalGoalLedger(CANONICAL_LEDGER_PATH)
    budget = ledger.snapshot()["formal_goal_budget"]
    regression: dict[str, Any] = {}
    runs: list[dict[str, Any]] = []
    runtime: RuntimeProcess | None = None
    capability: dict[str, Any] = {}
    runtime_identity_gate: dict[str, Any] = {"passed": False, "first_blocker": "runtime_not_started"}
    try:
        focused = run_pytest(output, "focused_pytest", ["-q", "tests/test_stage28sr2_r2_formal_runner.py", "tests/test_stage28sr2_production_integration.py"])
        post_test = output / "source_manifest_post_test.json"
        copy_manifest(pre_manifest_path, post_test)
        checkpoint_paths.append(post_test)
        collection = run_pytest(output, "full_collection_pytest", ["--collect-only", "-q"], collection_only=True)
        full = run_pytest(output, "full_pytest", ["-q"])
        post_regression = output / "source_manifest_post_regression.json"
        copy_manifest(pre_manifest_path, post_regression)
        checkpoint_paths.append(post_regression)
        regression = {"focused": focused, "full_collection": collection, "full": full, "passed": bool(focused.get("passed_gate") and collection.get("passed_gate") and full.get("passed_gate"))}
        write_json(output / "regression_certificate.json", regression)
        if not regression["passed"]:
            raise RuntimeError("focused or full regression prerequisite failed")
        archive = archive_source_tree(output, manifest)
        write_json(output / "source_archive_gate.json", archive)
        archive_manifest = output / "source_manifest_archive.json"
        copy_manifest(pre_manifest_path, archive_manifest)
        checkpoint_paths.append(archive_manifest)
        pre_runtime_manifest = output / "source_manifest_pre_runtime.json"
        copy_manifest(pre_manifest_path, pre_runtime_manifest)
        checkpoint_paths.append(pre_runtime_manifest)
        capability = probe_installed_rosbag2_capability(output_dir=output)
        if capability.get("passed") is not True:
            raise RuntimeError("installed rosbag2 capability gate failed")
        runtime = RuntimeProcess(output, args.runtime_launch, f"{utc_stamp()}_{uuid.uuid4().hex[:8]}")
        code, nodes_stdout, nodes_stderr = runtime.run_ros("ros2 node list --no-daemon")
        if code == 0 and any(name in nodes_stdout for name in ("/controller_manager", "/robot_state_publisher", "/fairino5_controller")):
            raise RuntimeError("pre-existing Stage 2.8 runtime nodes detected")
        runtime_identity = runtime.start()
        ready = runtime.ready()
        runtime_identity_gate = {
            **runtime_identity,
            "ready_probe": ready,
            "live_graph_observed": ready.get("observed_from_live_graph") is True,
            "same_runtime_instance": True,
            "passed": ready.get("passed") is True and runtime.process is not None and runtime.process.poll() is None,
        }
        write_json(output / "runtime_identity_gate.json", runtime_identity_gate)
        if not runtime_identity_gate["passed"]:
            raise RuntimeError("zero-goal runtime identity/readiness gate failed")
        for run_number in range(1, 4):
            run_output = output / f"run_{run_number}"
            report = run_one_recorder(run_output, runtime, capability, run_number, args.stability_seconds)
            runs.append(report)
        write_json(
            output / "zero_goal_rehearsal_summary.json",
            {
                "required_run_count": 3,
                "actual_run_count": len(runs),
                "all_runs_passed": len(runs) == 3 and all(run.get("passed") is True for run in runs),
                "runs": [{"run_number": run.get("run_number"), "passed": run.get("passed"), "first_blocker": run.get("first_blocker")} for run in runs],
            },
        )
    except Exception as error:
        write_json(output / "zero_goal_runtime_error.json", {"error": f"{type(error).__name__}:{error}"})
    finally:
        if runtime is not None:
            runtime.stop()
        if not capability:
            capability = {"passed": False, "first_blocker": "capability_test_not_completed", "rosbag2_version": None}
        if not regression:
            regression = {"focused": {}, "full_collection": {}, "full": {}, "passed": False}
        post_runtime_manifest = output / "source_manifest_post_runtime.json"
        if not post_runtime_manifest.is_file():
            copy_manifest(pre_manifest_path, post_runtime_manifest)
        if post_runtime_manifest not in checkpoint_paths:
            checkpoint_paths.append(post_runtime_manifest)
        final_checkpoint = output / "source_manifest_final_certificate.json"
        copy_manifest(pre_manifest_path, final_checkpoint)
        checkpoint_paths.append(final_checkpoint)
        source_binding = compare_manifest_bytes(pre_manifest_path, checkpoint_paths)
        source_binding["archive_gate_passed"] = (output / "source_archive_gate.json").is_file() and json.loads((output / "source_archive_gate.json").read_text(encoding="utf-8")).get("passed") is True
        source_binding["passed"] = bool(source_binding["passed"] and source_binding["archive_gate_passed"])
        certificate = build_certificate(output, runtime or RuntimeProcess(output, args.runtime_launch, "not_started"), capability, runs, regression, source_binding, budget, runtime_identity_gate)
    print(json.dumps({
        "output": str(output),
        "status": certificate["Stage_2_8S_R2_zero_goal_recertification"]["status"],
        "READY_FOR_FORMAL_R2_ONE_SHOT": certificate["READY_FOR_FORMAL_R2_ONE_SHOT"],
        "formal_goal_budget": certificate["formal_goal_budget"],
        "first_blocker": certificate["first_blocker"],
        "Can_we_send_the_formal_R2_goal_in_this_run": False,
        "Stage_2_9_authorized": False,
        "Stage_3_authorized": False,
    }, ensure_ascii=False, indent=2))
    return 0 if certificate["READY_FOR_FORMAL_R2_ONE_SHOT"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
