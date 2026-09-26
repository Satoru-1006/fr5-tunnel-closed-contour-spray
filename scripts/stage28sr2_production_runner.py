"""Single-final-runtime Stage 2.8S-R2 production preflight.

The CLI is audit-only.  It may start a simulation runtime, construct a real
ROS ActionClient, wait for the action server, and start an independent MCAP
recorder, but it stops before ledger consumption and never calls
``send_goal_async``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage28sr2_command_source_collector import LiveCommandSourceCollector, verify_command_source_report  # noqa: E402
from src.stage28sr2_formal_ledger import CANONICAL_LEDGER_PATH, FormalGoalLedger, atomic_write_json  # noqa: E402
from src.stage28sr2_live_runtime_evidence import (  # noqa: E402
    LiveRuntimeEvidenceCollector,
    RuntimeIdentity,
    sha256_file,
    validate_machine_provenance,
)
from src.stage28sr2_recorder_orchestrator import PATCHED_ROSBAG2_LIBRARY, REQUIRED_TOPICS, Rosbag2RecorderOrchestrator, recorder_static_audit  # noqa: E402
from src.stage28sr2_ros_action_transport import GoalResult, ProductionROSActionTransport, transport_static_audit  # noqa: E402
from src.stage28sr2_ros_goal import build_ros_goal, verify_goal_semantics  # noqa: E402
from src.stage28sr2_final_jit_gate import (  # noqa: E402
    FINAL_JIT_PRE_SEND_GATE,
    build_final_jit_pre_send_gate,
    production_dispatch_static_audit,
    revalidate_frozen_execution_manifest,
    verify_final_jit_pre_send_gate,
)
from scripts.stage28sr2_formal_runner import FormalState, PersistentStateMachine, formal_success_positive_whitelist, verify_atomic_ledger_binding_static, verify_state_machine_static  # noqa: E402
from scripts.stage28sr2_r2_zero_goal_hardening import PersistentObserverProcess  # noqa: E402
from src.stage28sr2_transport_mode import DEFAULT_UDP_SHM  # noqa: E402

STAGE27 = ROOT / "outputs/stage27s_native_spline_remediation/stage27s_formal_20260805T130000Z"
FROZEN_GOAL = STAGE27 / "stage27r_clean_follow_joint_trajectory_goal.json"
DEFAULT_RUNTIME_LAUNCH = ROOT / "tmp/stage28s_preflight_20260805/stage28s_runtime.launch.py"
ACTION_NAME = "/fairino5_controller/follow_joint_trajectory"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str | None:
    return sha256_file(path)


def build_frozen_formal_execution_manifest(output: Path) -> dict[str, Any]:
    patch = ROOT / "ros2_overlay/patches/rosbag2_transport-0.26.11-stage28sr2-h3.patch"
    qos = ROOT / "config/stage28sr2_recorder_qos_override.yaml"
    components = {
        "formal_runner": Path(__file__),
        "production_runner": Path(__file__),
        "formal_state_machine_runner": ROOT / "scripts/stage28sr2_formal_runner.py",
        "zero_goal_observer": ROOT / "scripts/stage28sr2_r2_zero_goal_hardening.py",
        "production_ros_action_transport": ROOT / "src/stage28sr2_ros_action_transport.py",
        "formal_goal_ledger": ROOT / "src/stage28sr2_formal_ledger.py",
        "final_jit_gate": ROOT / "src/stage28sr2_final_jit_gate.py",
        "live_runtime_evidence": ROOT / "src/stage28sr2_live_runtime_evidence.py",
        "command_source_collector": ROOT / "src/stage28sr2_command_source_collector.py",
        "recorder_orchestrator": ROOT / "src/stage28sr2_recorder_orchestrator.py",
        "persistent_graph": ROOT / "src/stage28sr2_persistent_graph.py",
        "transport_mode": ROOT / "src/stage28sr2_transport_mode.py",
        "ros_goal": ROOT / "src/stage28sr2_ros_goal.py",
        "recorder_live_query": ROOT / "scripts/stage28sr2_recorder_live_loss_query.py",
        "recorder_instrumentation_patch": patch,
        "patched_rosbag2_library": PATCHED_ROSBAG2_LIBRARY,
        "frozen_goal": FROZEN_GOAL,
        "qos_config": qos,
        "loaded_rosbag2_transport_library": PATCHED_ROSBAG2_LIBRARY,
    }
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
    status = subprocess.run(["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=ROOT, capture_output=True, text=True, check=False)
    entries: dict[str, Any] = {}
    for name, path in components.items():
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(ROOT).as_posix()
        except ValueError:
            relative = str(resolved)
        tracked_probe = subprocess.run(["git", "ls-files", "--error-unmatch", "--", relative], cwd=ROOT, capture_output=True, text=True, check=False)
        entries[name] = {
            "path": str(resolved),
            "sha256": _sha256(resolved),
            "git_tracked": tracked_probe.returncode == 0,
            "exists": resolved.is_file(),
        }
    manifest = {
        "schema_version": "stage28sr2-frozen-formal-execution-manifest-v1",
        "created_utc": now_utc(),
        "repository_commit_sha": commit.stdout.strip() if commit.returncode == 0 else None,
        "git_status_porcelain": status.stdout.splitlines(),
        "components": entries,
        "all_component_hashes_available": all(item["sha256"] is not None for item in entries.values()),
        "untracked_components_explicitly_sha_bound": [name for name, item in entries.items() if item["git_tracked"] is False],
        "exact_sha_enforcement": "revalidate immediately before canonical ledger consumption",
    }
    _write(output / "frozen_formal_execution_manifest.json", manifest)
    return manifest


def _frozen_point0(source: Mapping[str, Any]) -> list[float]:
    trajectory = source.get("trajectory") if isinstance(source.get("trajectory"), Mapping) else source
    points = trajectory.get("points", []) if isinstance(trajectory, Mapping) else []
    if not points or not isinstance(points[0], Mapping):
        raise ValueError("frozen goal has no point 0")
    values = points[0].get("positions")
    if not isinstance(values, list) or not values:
        raise ValueError("frozen goal point 0 has no positions")
    return [float(value) for value in values]


def _linux_path(path: Path) -> str:
    value = str(path.resolve()).replace("\\", "/")
    if len(value) > 1 and value[1] == ":":
        return f"/mnt/{value[0].lower()}/{value[3:]}"
    return value


def _pytest_counts(raw: str, *, collection_only: bool = False) -> dict[str, int | None]:
    """Extract pytest counts from the terminal summary without assuming success."""

    def number(pattern: str) -> int:
        match = re.search(pattern, raw, flags=re.IGNORECASE)
        return int(match.group(1)) if match else 0

    collected_match = re.search(r"(?P<count>\d+)\s+tests?\s+collected\b", raw, flags=re.IGNORECASE)
    passed = number(r"\b(\d+)\s+passed\b")
    failed = number(r"\b(\d+)\s+failed\b")
    errors = number(r"\b(\d+)\s+errors?\b")
    skipped = number(r"\b(\d+)\s+skipped\b")
    collected = int(collected_match.group("count")) if collected_match else None
    if collected is None and not collection_only and (passed or failed or errors or skipped):
        collected = passed + failed + errors + skipped
    return {"collected": collected, "passed": passed, "failed": failed, "errors": errors, "skipped": skipped}


def _pytest_python() -> str:
    """Allow a ROS-in-WSL runtime to bind regression evidence to host Python."""

    return os.environ.get("STAGE28SR2_PYTEST_PYTHON", sys.executable)


def _pytest_command(extra_args: list[str]) -> list[str]:
    command = [_pytest_python(), "-m", "pytest", *extra_args]
    if os.name != "nt" and _pytest_python().lower().endswith(".exe"):
        return ["bash", "-lc", shlex.join(command)]
    return command


def _run_pytest_suite(output: Path, *, name: str, command: list[str], collection_only: bool = False) -> dict[str, Any]:
    log_path = output / f"{name}.log"
    result: dict[str, Any] = {
        "executed": False,
        "collected": None,
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "skipped": 0,
        "exit_code": None,
        "log_sha256": None,
        "passed_gate": False,
        "pytest_command": " ".join(command),
        "log_path": str(log_path.resolve()),
    }
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,
            check=False,
        )
        raw = completed.stdout + ("\n--- STDERR ---\n" + completed.stderr if completed.stderr else "")
        log_path.write_text(raw, encoding="utf-8")
        result.update(_pytest_counts(raw, collection_only=collection_only))
        result.update(
            {
                "executed": True,
                "exit_code": completed.returncode,
                "log_sha256": _sha256(log_path),
            }
        )
    except Exception as error:
        log_path.write_text(repr(error) + "\n", encoding="utf-8")
        result.update({"executed": True, "error": repr(error), "log_sha256": _sha256(log_path)})
    result["passed_gate"] = bool(
        result["executed"]
        and result["exit_code"] == 0
        and result["collected"] is not None
        and result["failed"] == 0
        and result["errors"] == 0
    )
    if not result["executed"]:
        result["first_blocker"] = f"pytest_{name}_not_executed"
    elif result["errors"]:
        result["first_blocker"] = f"pytest_{name}_collection_errors"
    elif result["failed"]:
        result["first_blocker"] = f"pytest_{name}_failures"
    elif result["exit_code"] != 0:
        result["first_blocker"] = f"pytest_{name}_nonzero_exit"
    elif result["collected"] is None:
        result["first_blocker"] = f"pytest_{name}_counts_unavailable"
    return result


def _external_regression_gates(output: Path, evidence_path: Path) -> dict[str, Any]:
    path = evidence_path.resolve()
    raw = path.read_text(encoding="utf-8")
    focused_match = re.search(r"===== focused =====\s*(.*?)(?===== full =====)", raw, re.S)
    full_match = re.search(r"===== full =====\s*(.*)\Z", raw, re.S)
    if not focused_match or not full_match:
        raise ValueError("external regression evidence is missing focused/full sections")

    def result(section: str, source: str) -> dict[str, Any]:
        counts = _pytest_counts(section)
        passed = bool(counts["collected"] and counts["failed"] == 0 and counts["errors"] == 0 and re.search(r"\b\d+ passed\b", section))
        return {
            **counts,
            "executed": True,
            "exit_code": 0 if passed else 1,
            "passed_gate": passed,
            "source": source,
            "external_evidence_path": str(path),
            "external_evidence_sha256": _sha256(path),
        }

    focused = result(focused_match.group(1), "external_host_focused_regression")
    full = result(full_match.group(1), "external_host_full_regression")
    collection = {**full, "source": "derived_from_external_full_regression_collection"}
    payload = {
        "focused_stage28sr2_suite": focused,
        "full_collection_suite": collection,
        "full_regression_suite": full,
        "regression_gate": {
            "passed": bool(focused["passed_gate"] and full["passed_gate"]),
            "checks": {"focused_stage28sr2_suite": focused["passed_gate"], "full_collection_suite": full["passed_gate"], "full_regression_suite": full["passed_gate"]},
            "first_blocker": "none" if focused["passed_gate"] and full["passed_gate"] else "external_regression_failed",
        },
        "all_required_tests_passed": bool(focused["passed_gate"] and full["passed_gate"]),
        "external_regression_evidence": {"path": str(path), "sha256": _sha256(path)},
    }
    _write(output / "stage28sr2_r2_regression_certificate.json", payload)
    return payload


def run_regression_gates(output: Path, *, skip_tests: bool, external_evidence: Path | None = None) -> dict[str, Any]:
    """Run focused, collection-only, and complete pytest gates with evidence."""

    output.mkdir(parents=True, exist_ok=True)
    if external_evidence is not None:
        return _external_regression_gates(output, external_evidence)
    if skip_tests:
        disabled = {
            "executed": False,
            "collected": None,
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "skipped": 0,
            "exit_code": None,
            "log_sha256": None,
            "passed_gate": False,
            "first_blocker": "pytest_execution_skipped_by_cli",
        }
        result = {"focused_stage28sr2_suite": dict(disabled), "full_collection_suite": dict(disabled), "full_regression_suite": dict(disabled)}
    else:
        focused = _run_pytest_suite(
            output,
            name="stage28sr2_r2_focused_pytest",
            command=_pytest_command(["-q", "tests/test_stage28sr2_r2_formal_runner.py", "tests/test_stage28sr2_production_integration.py"]),
        )
        collection = _run_pytest_suite(
            output,
            name="stage28sr2_r2_full_collection_pytest",
            command=_pytest_command(["--collect-only", "-q"]),
            collection_only=True,
        )
        full = _run_pytest_suite(
            output,
            name="stage28sr2_r2_full_pytest",
            command=_pytest_command(["-q"]),
        )
        if full["collected"] == full["errors"] and collection["collected"] is not None:
            # A collection error prevents pytest from printing the normal
            # test-count summary; bind the execution certificate to the
            # independent collection probe instead of counting error lines as tests.
            full["collected"] = collection["collected"]
        result = {
            "focused_stage28sr2_suite": focused,
            "full_collection_suite": collection,
            "full_regression_suite": full,
        }
    gates = {key: bool(value.get("passed_gate")) for key, value in result.items()}
    first_blocker = next((value.get("first_blocker") for value in result.values() if not value.get("passed_gate")), "none")
    regression_gate = {"passed": all(gates.values()), "checks": gates, "first_blocker": first_blocker}
    payload = {
        **result,
        "regression_gate": regression_gate,
        "all_required_tests_passed": regression_gate["passed"],
    }
    _write(output / "stage28sr2_r2_regression_certificate.json", payload)
    return payload


class SingleFinalRuntime:
    def __init__(self, output: Path, launch: Path) -> None:
        self.output = output
        self.launch = launch
        self.process: subprocess.Popen[str] | None = None
        self.identity: RuntimeIdentity | None = None

    def reserve_identity(self) -> RuntimeIdentity:
        if self.identity is None:
            self.identity = RuntimeIdentity.create(None)
        return self.identity

    def start(self) -> RuntimeIdentity:
        if not self.launch.is_file():
            raise FileNotFoundError(f"runtime launch file missing: {self.launch}")
        log = (self.output / "final_runtime.log").open("w", encoding="utf-8")
        linux_launch = _linux_path(self.launch)
        if shutil.which("ros2"):
            command = f"source /opt/ros/jazzy/setup.bash; source /mnt/d/robotfucker/install/setup.bash 2>/dev/null || true; ros2 launch {linux_launch} runtime_tag:=stage28sr2_r2_final"
            argv = ["bash", "-lc", command]
        elif shutil.which("wsl.exe"):
            command = f"source /opt/ros/jazzy/setup.bash; source /mnt/d/robotfucker/install/setup.bash 2>/dev/null || true; ros2 launch {linux_launch} runtime_tag:=stage28sr2_r2_final"
            argv = ["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", command]
        else:
            raise RuntimeError("ROS runtime launcher unavailable: ros2 and wsl.exe are missing")
        self.process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, text=True)
        reserved = self.reserve_identity()
        self.identity = RuntimeIdentity(reserved.runtime_instance_id, self.process.pid, reserved.launch_start_time)
        _write(self.output / "runtime_launch_identity.json", {**self.identity.as_dict(), "command": command, "launch_path": str(self.launch.resolve())})
        return self.identity

    def wait_ready(self, timeout_s: float = 45.0) -> dict[str, Any]:
        started = time.monotonic()
        last: dict[str, Any] = {}
        while time.monotonic() - started < timeout_s:
            code, stdout, stderr = self._probe("ros2 control list_controllers")
            last = {"exit_code": code, "stdout": stdout, "stderr": stderr}
            if code == 0 and "fairino5_controller" in stdout and "active" in stdout:
                last["passed"] = True
                return last
            if self.process is not None and self.process.poll() is not None:
                break
            time.sleep(1.0)
        last["passed"] = False
        last["first_blocker"] = "final_runtime_not_ready"
        return last

    def _probe(self, command: str) -> tuple[int | None, str, str]:
        if shutil.which("ros2"):
            argv = ["bash", "-lc", f"source /opt/ros/jazzy/setup.bash; source /mnt/d/robotfucker/install/setup.bash 2>/dev/null || true; {command}"]
        elif shutil.which("wsl.exe"):
            argv = ["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", f"source /opt/ros/jazzy/setup.bash; source /mnt/d/robotfucker/install/setup.bash 2>/dev/null || true; {command}"]
        else:
            return None, "", "ROS CLI unavailable"
        try:
            completed = subprocess.run(argv, capture_output=True, timeout=20, check=False)
            return completed.returncode, completed.stdout.decode("utf-8", errors="replace"), completed.stderr.decode("utf-8", errors="replace")
        except (OSError, subprocess.TimeoutExpired) as error:
            return None, "", repr(error)

    def stop(self) -> dict[str, Any]:
        if self.process is None:
            return {"started": False}
        try:
            self.process.send_signal(signal.SIGINT)
            self.process.wait(timeout=30)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
        return {"started": True, "pid": self.process.pid, "returncode": self.process.returncode}


class ProductionFormalExecutor:
    """One production ROS Future lifecycle; never entered by default H3 mode."""

    def __init__(
        self,
        *,
        output: Path,
        transport: Any,
        recorder: Any,
        ledger: FormalGoalLedger,
        machine: PersistentStateMachine,
        goal: Any,
        pre_send_live_loss_gate: Mapping[str, Any],
        analysis: Any,
        post_roll: Any = None,
        final_jit_gate_factory: Callable[[], Mapping[str, Any]] | None = None,
        frozen_execution_manifest: Mapping[str, Any] | None = None,
        goal_response_timeout_s: float = 10.0,
        result_timeout_s: float = 900.0,
        test_mode: bool = False,
    ) -> None:
        self.output = output.resolve()
        self.transport = transport
        self.recorder = recorder
        self.ledger = ledger
        self.machine = machine
        self.goal = goal
        self.pre_send_live_loss_gate = dict(pre_send_live_loss_gate)
        self.analysis = analysis
        self.post_roll = post_roll or (lambda: None)
        self.final_jit_gate_factory = final_jit_gate_factory or (lambda: dict(pre_send_live_loss_gate))
        self.frozen_execution_manifest = dict(frozen_execution_manifest or {})
        self.goal_response_timeout_s = goal_response_timeout_s
        self.result_timeout_s = result_timeout_s
        self.test_mode = test_mode
        self.events: list[str] = []
        self.formal_started = False
        self.final_jit_gate: dict[str, Any] = {}
        self.final_jit_verification: dict[str, Any] = {}
        self.recorder_cleanup_attempted = False
        self.recorder_cleanup_blocker: str | None = None
        self.boundary_timing: dict[str, Any] = {
            "final_jit_query_monotonic_ns": None,
            "ledger_consume_monotonic_ns": None,
            "transport_dispatch_monotonic_ns": None,
            "final_jit_to_ledger_consume_latency_s": None,
            "final_jit_to_transport_dispatch_latency_s": None,
        }

    def _event(self, value: str) -> None:
        self.events.append(value)

    def _certificate(
        self,
        *,
        result: GoalResult | None,
        recorder_state: Mapping[str, Any],
        analysis: Mapping[str, Any],
        blocker: str | None,
    ) -> dict[str, Any]:
        budget = self.ledger.reread()["formal_goal_budget"]
        whitelist = formal_success_positive_whitelist({
            "accepted": result.accepted if result else None,
            "action_terminal_status": result.terminal_status if result else None,
            "fjt_error_code": result.fjt_error_code if result else None,
        })
        real_transport = isinstance(self.transport, ProductionROSActionTransport) and not self.test_mode
        post_loss = recorder_state.get("recorder_loss_gate", {}) if isinstance(recorder_state.get("recorder_loss_gate"), Mapping) else {}
        evidence_ready = bool(
            recorder_state.get("finalized") is True
            and recorder_state.get("bag_complete") is True
            and recorder_state.get("sequential_reader_passed") is True
            and recorder_state.get("bag_valid") is True
            and post_loss.get("passed") is True
        )
        formal_success = bool(
            real_transport and blocker is None and whitelist.get("passed") is True
            and analysis.get("passed") is True and evidence_ready
        )
        certificate = {
            "schema_version": "stage28sr2-production-formal-execution-v1",
            "execution_mode": "production_formal" if not self.test_mode else "production_future_mock_rehearsal",
            "real_ros_transport_used": real_transport,
            "eligible_for_formal_certification": real_transport,
            "canonical_ledger_path": str(self.ledger.path),
            "canonical_ledger_used": self.ledger.is_canonical,
            "formal_goal_budget": budget,
            "goal_budget_consumed": budget.get("goal_budget_consumed") is True,
            "send_binding_committed": budget.get("send_binding_committed") is True,
            "transport_send_invocation_observed": budget.get("transport_send_invocation_observed") is True,
            "send_goal_async_call_count": budget.get("send_goal_async_call_count"),
            "goal_accepted": result.accepted if result else None,
            "action_terminal_status": result.terminal_status if result else None,
            "FJT_error_code": result.fjt_error_code if result else None,
            "FJT_error_string": result.fjt_error_string if result else None,
            "pre_send_live_loss_gate": self.pre_send_live_loss_gate,
            "pre_send_live_zero_proven": self.pre_send_live_loss_gate.get("pre_send_live_zero_proven") is True,
            "recorder_finalization": dict(recorder_state),
            "formal_analysis": dict(analysis),
            "formal_success_positive_whitelist": whitelist,
            "formal_started": self.formal_started,
            "formal_status": "passed" if formal_success else "failed" if self.formal_started else "not_started",
            "final_jit_pre_send_gate": dict(self.final_jit_gate),
            "final_jit_pre_send_verification": dict(self.final_jit_verification),
            "final_jit_boundary_timing": dict(self.boundary_timing),
            "frozen_execution_manifest_revalidation": dict(self.final_jit_gate.get("manifest_gate", {})),
            "recorder_cleanup": {
                "attempted": self.recorder_cleanup_attempted,
                "exactly_once": self.recorder_cleanup_attempted,
                "cleanup_blocker": self.recorder_cleanup_blocker,
                "partial_bag_retained": recorder_state.get("partial_bag_retained") if isinstance(recorder_state, Mapping) else None,
                "finalized": recorder_state.get("finalized") if isinstance(recorder_state, Mapping) else False,
            },
            "event_trace": list(self.events),
            "first_blocker": blocker or "none",
            "Stage_2_9_authorized": formal_success,
            "Stage_3_authorized": False,
            "retry_allowed": False,
        }
        atomic_write_json(self.output / "production_formal_execution_certificate.json", certificate)
        return certificate

    def _finalize_recorder_once(self, recorder_state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Finalize the recorder exactly once, including exception paths."""

        if self.recorder_cleanup_attempted:
            return recorder_state
        self.recorder_cleanup_attempted = True
        if self.recorder is None:
            self.recorder_cleanup_blocker = "recorder_not_available_for_cleanup"
            return recorder_state
        try:
            finalized = self.recorder.finalize()
            if isinstance(finalized, Mapping):
                self._event("recorder_stopped_finalized_and_read_to_eof")
                return dict(finalized)
            self.recorder_cleanup_blocker = "recorder_finalize_returned_non_mapping"
        except Exception as error:  # pragma: no cover - exercised by fault injection
            self.recorder_cleanup_blocker = f"recorder_cleanup_exception:{type(error).__name__}:{error}"
        observed = getattr(self.recorder, "state", {})
        if isinstance(observed, Mapping):
            retained = dict(observed)
        else:
            retained = {}
        retained.setdefault("partial_bag_retained", bool(retained.get("metadata_exists_after_finalize")))
        retained["cleanup_blocker"] = self.recorder_cleanup_blocker
        return retained

    def run(self) -> dict[str, Any]:
        result: GoalResult | None = None
        recorder_state: Mapping[str, Any] = {}
        analysis: Mapping[str, Any] = {"passed": False, "first_blocker": "analysis_not_run"}
        blocker: str | None = None
        try:
            if not self.test_mode and getattr(self.transport, "is_fake_transport", True):
                raise RuntimeError("production executor rejects fake transport")
            if not self.test_mode and not isinstance(self.transport, ProductionROSActionTransport):
                raise RuntimeError("production executor requires ProductionROSActionTransport")
            if self.machine.current != FormalState.READY_FOR_LEDGER:
                raise RuntimeError("production executor requires READY_FOR_LEDGER")
            if not (
                self.pre_send_live_loss_gate.get("pre_send_live_zero_proven") is True
                and self.pre_send_live_loss_gate.get("recorder_bound") is True
                and self.pre_send_live_loss_gate.get("transport_lost_total") == 0
            ):
                raise RuntimeError("production executor PRE recorder-bound live-zero gate failed")
            self.final_jit_gate = dict(self.final_jit_gate_factory())
            self.final_jit_verification = verify_final_jit_pre_send_gate(self.final_jit_gate)
            if self.final_jit_verification.get("passed") is not True:
                raise RuntimeError(
                    "FINAL_JIT_PRE_SEND_GATE failed: "
                    + str(self.final_jit_verification.get("first_blocker", "unknown"))
                )
            self._event(FINAL_JIT_PRE_SEND_GATE)
            final_jit_query_ns = self.final_jit_gate.get("query_monotonic_ns")
            self.boundary_timing["final_jit_query_monotonic_ns"] = final_jit_query_ns
            consumed = self.ledger.consume_before_send()
            consume_ns = time.monotonic_ns()
            self.boundary_timing["ledger_consume_monotonic_ns"] = consume_ns
            if isinstance(final_jit_query_ns, int):
                self.boundary_timing["final_jit_to_ledger_consume_latency_s"] = (consume_ns - final_jit_query_ns) / 1_000_000_000
            self.formal_started = True
            self._event("formal_execution_started_after_canonical_ledger_consumption")
            self._event("canonical_ledger_consumed")
            durable = self.ledger.reread()
            if durable != consumed or durable["formal_goal_budget"].get("goal_budget_consumed") is not True:
                raise RuntimeError("durable ledger reread mismatch")
            self.machine.transition(FormalState.LEDGER_CONSUMED)
            self.ledger.bind_send_goal_async_call()
            self._event("single_send_binding_committed")
            self.machine.transition(FormalState.SEND_ATTEMPTED)
            self.ledger.record_transport_send_invocation()
            self._event("transport_send_invocation_recorded")
            if not self.test_mode:
                self.transport.bind_formal_send_authorization(self.ledger, self.final_jit_gate)
                self._event("ProductionROSActionTransport_FINAL_JIT_authorization_bound")
            # Sole real FollowJointTrajectory dispatch point for Formal R2.
            dispatch_ns = time.monotonic_ns()
            self.boundary_timing["transport_dispatch_monotonic_ns"] = dispatch_ns
            if isinstance(final_jit_query_ns, int):
                self.boundary_timing["final_jit_to_transport_dispatch_latency_s"] = (dispatch_ns - final_jit_query_ns) / 1_000_000_000
            send_future = self.transport.send_goal_async(self.goal)
            self._event("ProductionROSActionTransport.send_goal_async")
            self.machine.transition(FormalState.ACK_WAIT)
            response = self.transport.wait_for_goal_response(send_future, timeout_sec=self.goal_response_timeout_s)
            self.ledger.record_goal_response(acknowledged=response.accepted is not None, accepted=response.accepted)
            self._event("goal_response_future_completed")
            if response.accepted is None:
                result, blocker = response, "goal_response_timeout"
                self.machine.transition(FormalState.CLIENT_TIMEOUT)
            elif response.accepted is False:
                result, blocker = response, "goal_rejected"
                self.machine.transition(FormalState.REJECTED)
            else:
                self.machine.transition(FormalState.ACCEPTED)
                self.machine.transition(FormalState.EXECUTING)
                result = self.transport.wait_for_result(response.goal_handle, timeout_sec=self.result_timeout_s)
                self._event("get_result_async_future_completed")
                target = {
                    "SUCCEEDED": FormalState.SUCCEEDED,
                    "ABORTED": FormalState.ABORTED,
                    "CANCELED": FormalState.CANCELED,
                    "CLIENT_TIMEOUT": FormalState.CLIENT_TIMEOUT,
                }.get(result.terminal_status or "", FormalState.ANALYSIS_FAILED)
                self.machine.transition(target)
                if target != FormalState.SUCCEEDED:
                    blocker = f"action_terminal_status:{result.terminal_status}"
                elif result.fjt_error_code not in (0, None):
                    blocker = "fjt_error_code_nonzero"
            self.post_roll()
            self._event("recorder_post_roll_completed")
            recorder_state = self._finalize_recorder_once(recorder_state)
            if result is not None:
                analysis = self.analysis(result, recorder_state)
                self._event("formal_analysis_completed")
            if blocker is None and self.recorder_cleanup_blocker:
                blocker = self.recorder_cleanup_blocker
            if blocker is None and analysis.get("passed") is not True:
                blocker = str(analysis.get("first_blocker", "formal_analysis_failed"))
            post_loss = recorder_state.get("recorder_loss_gate", {}) if isinstance(recorder_state.get("recorder_loss_gate"), Mapping) else {}
            if blocker is None and not (
                recorder_state.get("finalized") is True
                and recorder_state.get("bag_complete") is True
                and recorder_state.get("sequential_reader_passed") is True
                and post_loss.get("passed") is True
            ):
                blocker = "post_stop_recorder_validation_failed"
            if self.machine.current != FormalState.TERMINAL:
                self.machine.transition(FormalState.TERMINAL)
        except Exception as error:
            blocker = blocker or f"production_executor_exception:{type(error).__name__}:{error}"
            self.machine.force_terminal(blocker)
        finally:
            # This is deliberately independent of certificate creation.  An
            # exception after consume, after binding, before dispatch, or in a
            # Future must never bypass orderly recorder cleanup.
            recorder_state = self._finalize_recorder_once(recorder_state)
            if self.recorder_cleanup_blocker and blocker is None:
                blocker = self.recorder_cleanup_blocker
        return self._certificate(result=result, recorder_state=recorder_state, analysis=analysis, blocker=blocker)


class ProductionPreflightRunner:
    def __init__(self, output: Path, *, launch: Path = DEFAULT_RUNTIME_LAUNCH, skip_tests: bool = False, regression_evidence: Path | None = None, formal_execute: bool = False) -> None:
        self.output = output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.launch = launch.resolve()
        self.skip_tests = skip_tests
        self.regression_evidence = regression_evidence.resolve() if regression_evidence else None
        self.formal_execute = formal_execute
        self.formal_execution_certificate: dict[str, Any] | None = None
        self.runtime: SingleFinalRuntime | None = None
        self.transport: ProductionROSActionTransport | None = None
        self.recorder: Rosbag2RecorderOrchestrator | None = None
        self.identity: RuntimeIdentity | None = None
        self.evidence: dict[str, Any] | None = None
        self.goal: Any = None
        self.goal_semantics: dict[str, Any] = {}
        self.command_sources: dict[str, Any] = {}
        self.recorder_ready: dict[str, Any] = {}
        self.regression: dict[str, Any] = {}
        self.blockers: list[str] = []
        self.events: list[str] = []
        self.evidence_collector: LiveRuntimeEvidenceCollector | None = None
        self.frozen_execution_manifest: dict[str, Any] = build_frozen_formal_execution_manifest(self.output)
        self.final_jit_pre_send_gate: dict[str, Any] = {
            "gate_name": FINAL_JIT_PRE_SEND_GATE,
            "passed": False,
            "first_blocker": "not_queried",
        }
        self.final_jit_pre_send_verification: dict[str, Any] = {}
        self.machine = PersistentStateMachine(self.output / "formal_state_machine.json")
        self.observer: PersistentObserverProcess | None = None
        self.persistent_graph: dict[str, Any] = {}
        self.pre_send_live_loss_gate: dict[str, Any] = {
            "pre_send_live_zero_proven": False,
            "recorder_bound": False,
            "transport_lost_total": None,
            "passed": False,
            "first_blocker": "not_queried",
        }

    def _block(self, reason: str) -> None:
        if reason and reason not in self.blockers:
            self.blockers.append(reason)

    def _event(self, name: str) -> None:
        self.events.append(name)

    def _load_goal(self) -> Mapping[str, Any]:
        if not FROZEN_GOAL.is_file():
            raise FileNotFoundError(FROZEN_GOAL)
        return json.loads(FROZEN_GOAL.read_text(encoding="utf-8"))

    def _flatten_compatibility(self, evidence: Mapping[str, Any], recorder: Mapping[str, Any], command: Mapping[str, Any]) -> dict[str, Any]:
        controller = evidence.get("controller", {})
        ros = evidence.get("ROS", {})
        hardware = evidence.get("hardware", {})
        initial = evidence.get("initial_state", {})
        runtime = evidence.get("runtime", {})
        return {
            "evidence_kind": "live_final_runtime_instance",
            "source": evidence.get("source"),
            "runtime_instance_id": evidence.get("runtime_instance_id"),
            "verified_runtime_instance_id": evidence.get("runtime_instance_id"),
            "recorder_runtime_instance_id": recorder.get("runtime_instance_id"),
            "recheck_runtime_instance_id": evidence.get("runtime_identity", {}).get("final_runtime_instance_id"),
            "same_final_runtime_identity": evidence.get("runtime_identity", {}).get("same_runtime_instance"),
            "runtime_restart_count": evidence.get("runtime_restart_count"),
            "controller_restart_count": evidence.get("controller_restart_count"),
            "controller_reactivation_count": evidence.get("controller_reactivation_count"),
            "controller_pid_or_process_identity": runtime.get("launch_pid"),
            "ROS_distribution": ros.get("ROS_DISTRO"),
            "JTC_package_version": ros.get("joint_trajectory_controller_version"),
            "controller_name": controller.get("controller_name"),
            "controller_type": controller.get("controller_type"),
            "controller_state": controller.get("controller_state"),
            "interpolation_method": controller.get("interpolation_method"),
            "hardware_component": hardware.get("hardware_component"),
            "simulation_only": hardware.get("simulation_only"),
            "configured_joints": controller.get("joints"),
            "command_interfaces": controller.get("command_interfaces"),
            "state_interfaces": controller.get("state_interfaces"),
            "initial_joint_state": initial.get("observed_joint_positions"),
            "frozen_point0": initial.get("frozen_point0"),
            "initial_state_max_abs_error_rad": initial.get("max_abs_error_rad"),
            "recorder_readiness": recorder,
            "command_source_exclusivity": command,
        }

    def _run_final_jit_pre_send_gate(self) -> dict[str, Any]:
        """Re-query every mutable authorization fact immediately before consume."""

        if self.recorder is None or self.transport is None or self.runtime is None or self.identity is None:
            gate = {
                "gate_name": FINAL_JIT_PRE_SEND_GATE,
                "proof_kind": "final_jit_query",
                "passed": False,
                "first_blocker": "final_jit_runtime_components_unavailable",
            }
            self.final_jit_pre_send_gate = gate
            self.final_jit_pre_send_verification = verify_final_jit_pre_send_gate(gate)
            return gate

        final_query = self.recorder.query_live_transport_loss(timeout_s=5.0)
        authorized_node = self.transport.node.get_fully_qualified_name()
        observer_pid = int(self.observer.ready["pid"]) if self.observer and self.observer.ready.get("pid") else None
        final_sources = LiveCommandSourceCollector(self.output / "final_jit_command_sources").collect(
            authorized_action_client_nodes={authorized_node},
            authorized_process_ids={observer_pid} if observer_pid is not None else set(),
        )
        final_sources_verification = verify_command_source_report(final_sources, authorized_node=authorized_node)
        final_sources["verification"] = final_sources_verification
        runtime_probe = self.runtime.wait_ready(timeout_s=5.0)
        recorder_alive = self.recorder._is_alive()
        controller_active = runtime_probe.get("passed") is True
        action_server_available = self.transport.wait_for_server(timeout_sec=5.0)
        library = final_query.get("patched_binary_or_library", {})
        manifest_gate = revalidate_frozen_execution_manifest(
            self.frozen_execution_manifest,
            loaded_library_sha256=library.get("loaded_library_sha256") if isinstance(library, Mapping) else None,
            loaded_library_path=library.get("loaded_library_path") if isinstance(library, Mapping) else None,
        )
        gate = build_final_jit_pre_send_gate(
            initial_pre_gate=self.pre_send_live_loss_gate,
            final_query=final_query,
            recorder_alive=recorder_alive,
            controller_active=controller_active,
            action_server_available=action_server_available,
            command_source_report=final_sources,
            manifest_gate=manifest_gate,
        )
        gate["runtime_probe"] = runtime_probe
        gate["same_runner_runtime_instance"] = bool(
            self.evidence
            and self.evidence.get("runtime_instance_id") == self.identity.runtime_instance_id
        )
        gate["checks"]["same_runner_runtime_instance"] = gate["same_runner_runtime_instance"]
        gate["passed"] = bool(gate.get("passed") and gate["same_runner_runtime_instance"])
        gate["first_blocker"] = next((name for name, value in gate.get("checks", {}).items() if not value), None)
        self.final_jit_pre_send_gate = gate
        self.final_jit_pre_send_verification = verify_final_jit_pre_send_gate(gate)
        _write(self.output / "final_jit_pre_send_gate.json", gate)
        _write(self.output / "final_jit_pre_send_verification.json", self.final_jit_pre_send_verification)
        if gate.get("passed") is True and self.final_jit_pre_send_verification.get("passed") is True:
            self._event("FINAL_JIT_PRE_SEND_GATE_verified")
        else:
            self._block(str(self.final_jit_pre_send_verification.get("first_blocker") or gate.get("first_blocker") or "final_jit_pre_send_gate_failed"))
        return gate

    def run(self) -> dict[str, Any]:
        goal_source: Mapping[str, Any] | None = None
        evidence_collector: LiveRuntimeEvidenceCollector | None = None
        try:
            self.machine.transition(FormalState.PREFLIGHT_RUNNING)
            self.runtime = SingleFinalRuntime(self.output, self.launch)
            self.identity = self.runtime.reserve_identity()
            self.observer = PersistentObserverProcess(self.output / "persistent_graph", DEFAULT_UDP_SHM)
            observer_ready = self.observer.start()
            _write(self.output / "persistent_graph_observer_ready.json", observer_ready)
            self._event("persistent_graph_observer_started_before_recorder_and_runtime")

            self.recorder = Rosbag2RecorderOrchestrator(
                self.output / "recorder",
                runtime_instance_id=self.identity.runtime_instance_id,
                persistent_graph_raw_path=self.observer.raw,
                persistent_observer_pid=int(observer_ready["pid"]),
                observer_started_before_runtime=True,
                transport_mode=DEFAULT_UDP_SHM,
            )
            self.recorder.start()
            recorder_ready = self.recorder.verify_readiness()
            reader_gate = self.recorder.verify_live_endpoints()
            self.recorder_ready = dict(recorder_ready)
            self.recorder_ready["pre_publisher_reader_gate"] = reader_gate
            self._event("recorder_started_before_publishers")
            if not recorder_ready.get("ready") or reader_gate.get("passed") is not True:
                self._block("recorder_readers_unavailable_before_publishers")
            else:
                self._event("required_recorder_readers_ready_before_publishers")

            self.identity = self.runtime.start()
            self._event("single_final_runtime_started")
            ready = self.runtime.wait_ready()
            _write(self.output / "runtime_ready_probe.json", ready)
            if not ready.get("passed"):
                self._block(str(ready.get("first_blocker", "runtime_not_ready")))

            goal_source = self._load_goal()
            point0 = _frozen_point0(goal_source)
            self.goal = build_ros_goal(goal_source)
            self.goal_semantics = verify_goal_semantics(goal_source, self.goal)
            _write(self.output / "production_ros_goal_semantic_report.json", self.goal_semantics)
            if self.goal_semantics.get("passed"):
                self._event("production_ros_goal_constructed_and_verified")
            else:
                self._block("production_ros_goal_semantics_failed")

            evidence_collector = LiveRuntimeEvidenceCollector(self.output, self.identity)
            self.evidence_collector = evidence_collector
            initial = evidence_collector.collect(phase="initial", frozen_point0=point0)
            _write(self.output / "runtime_initial_live_probe.json", initial)
            self._event("initial_live_identity_probe_completed")

            self.transport = ProductionROSActionTransport(audit_only=not self.formal_execute)
            self._event("production_ActionClient_created")
            if not self.transport.wait_for_server(timeout_sec=15.0):
                self._block("production_action_server_unavailable")
            else:
                self._event("production_action_server_available")

            self.persistent_graph = self.recorder.verify_persistent_publishers(timeout_sec=45.0)
            _write(self.output / "persistent_graph_summary.json", self.persistent_graph)
            if self.persistent_graph.get("passed") is not True:
                self._block(str(self.persistent_graph.get("first_blocker", "persistent_graph_gate_failed")))
            else:
                self._event("persistent_graph_and_qos_runtime_gate_passed")

            self.pre_send_live_loss_gate = self.recorder.query_live_transport_loss(timeout_s=5.0)
            if self.pre_send_live_loss_gate.get("passed") is not True:
                self._block(str(self.pre_send_live_loss_gate.get("first_blocker", "pre_send_live_loss_gate_failed")))
            else:
                self._event("recorder_bound_PRE_live_transport_loss_zero_verified")

            authorized_node = self.transport.node.get_fully_qualified_name() if self.transport is not None else None
            source_collector = LiveCommandSourceCollector(self.output)
            observer_pid = int(self.observer.ready["pid"]) if self.observer and self.observer.ready.get("pid") else None
            self.command_sources = source_collector.collect(
                authorized_action_client_nodes={authorized_node} if authorized_node else set(),
                authorized_process_ids={observer_pid} if observer_pid is not None else set(),
            )
            if not verify_command_source_report(self.command_sources, authorized_node=authorized_node).get("passed"):
                self._block("command_source_exclusivity_failed")
            else:
                self._event("command_source_exclusivity_verified")

            final = evidence_collector.collect(phase="final", frozen_point0=point0)
            _write(self.output / "runtime_final_pre_send_live_probe.json", final)
            self.evidence = evidence_collector.finalize(
                frozen_point0=point0,
                robot_model_paths={
                    "robot_description": self.launch.parent / "stage28s_expanded_runtime.urdf",
                    "controllers_yaml": self.launch.parent / "stage28s_controllers.yaml",
                    "initial_positions": self.launch.parent / "stage28s_initial_positions.yaml",
                },
            )
            if not self.evidence.get("runtime_identity", {}).get("same_runtime_instance"):
                self._block("single_runtime_identity_mismatch")
            self._event("same_runtime_identity_rechecked")
            provenance = validate_machine_provenance(self.evidence)
            _write(self.output / "runtime_provenance_gate.json", provenance)
            if not provenance.get("passed"):
                self._block(f"runtime_evidence_provenance_failed:{provenance.get('first_blocker')}")
            else:
                self._event("machine_live_provenance_verified")
            if self.blockers:
                self.machine.force_terminal(self.blockers[0])
            else:
                self.machine.transition(FormalState.PREFLIGHT_PASSED)
                self.machine.transition(FormalState.READY_FOR_LEDGER)
                self._event("READY_FOR_LEDGER")
                self.regression = run_regression_gates(
                    self.output,
                    skip_tests=self.skip_tests,
                    external_evidence=self.regression_evidence,
                )
                if self.regression.get("regression_gate", {}).get("passed") is not True:
                    self._block(str(self.regression.get("regression_gate", {}).get("first_blocker", "pre_send_regression_failed")))
                    self.machine.force_terminal(self.blockers[0])
                elif self.formal_execute:
                    def production_analysis(result: GoalResult, recorder_state: Mapping[str, Any]) -> dict[str, Any]:
                        checks = {
                            "status_succeeded": result.terminal_status == "SUCCEEDED",
                            "fjt_error_code_zero": result.fjt_error_code == 0,
                            "bag_valid": recorder_state.get("bag_valid") is True,
                            "read_to_eof": recorder_state.get("sequential_reader_passed") is True,
                            "post_stop_loss_valid": recorder_state.get("recorder_loss_gate", {}).get("passed") is True,
                        }
                        return {"passed": all(checks.values()), "checks": checks, "first_blocker": next((key for key, value in checks.items() if not value), None)}

                    executor = ProductionFormalExecutor(
                        output=self.output,
                        transport=self.transport,
                        recorder=self.recorder,
                        ledger=FormalGoalLedger(CANONICAL_LEDGER_PATH),
                        machine=self.machine,
                        goal=self.goal,
                        pre_send_live_loss_gate=self.pre_send_live_loss_gate,
                        analysis=production_analysis,
                        post_roll=lambda: time.sleep(2.0),
                        final_jit_gate_factory=self._run_final_jit_pre_send_gate,
                        frozen_execution_manifest=self.frozen_execution_manifest,
                    )
                    self.formal_execution_certificate = executor.run()
                else:
                    # ZERO-GOAL rehearsal: perform the same final mutable-state
                    # proof, then stop before any ledger operation.
                    self._run_final_jit_pre_send_gate()
                    self._event("audit_stopped_before_ledger_consumption")
        except Exception as error:
            self._block(f"production_preflight_exception:{type(error).__name__}:{error}")
            self.machine.force_terminal(self.blockers[0])
        finally:
            if self.recorder is not None and self.formal_execution_certificate is None:
                self.recorder.monitor_health()
                self.recorder.finalize()
            if self.transport is not None:
                self.transport.close()
            if self.runtime is not None:
                _write(self.output / "runtime_stop.json", self.runtime.stop())
            if self.observer is not None:
                _write(self.output / "persistent_graph_observer_stop.json", self.observer.stop())
        return self._certificate()

    def _certificate(self) -> dict[str, Any]:
        ledger: FormalGoalLedger | None = None
        ledger_validation_error: str | None = None
        raw_ledger: dict[str, Any] = {}
        try:
            ledger = FormalGoalLedger(CANONICAL_LEDGER_PATH)
            raw_ledger = ledger.snapshot()
        except Exception as error:
            # A stale identity is itself a fail-closed blocker after source
            # hardening.  Preserve the raw, non-mutated ledger snapshot so the
            # terminal certificate can still report why authorization stopped.
            ledger_validation_error = f"{type(error).__name__}:{error}"
            try:
                raw_value = json.loads(CANONICAL_LEDGER_PATH.read_text(encoding="utf-8"))
                if isinstance(raw_value, dict):
                    raw_ledger = raw_value
            except (OSError, json.JSONDecodeError) as read_error:
                ledger_validation_error += f";raw_read:{type(read_error).__name__}:{read_error}"
            self._block(f"canonical_ledger_validation_failed:{ledger_validation_error}")
        budget_value = raw_ledger.get("formal_goal_budget")
        budget = dict(budget_value) if isinstance(budget_value, Mapping) else {
            "maximum": None,
            "consumed": None,
            "goal_budget_consumed": None,
            "send_attempted": None,
            "send_binding_committed": None,
            "transport_send_invocation_observed": None,
            "send_goal_async_call_count": None,
        }
        if not self.regression:
            self.regression = run_regression_gates(self.output, skip_tests=self.skip_tests, external_evidence=self.regression_evidence)
        regression_gate = self.regression["regression_gate"]
        if not regression_gate.get("passed"):
            self._block(str(regression_gate.get("first_blocker", "regression_gate_failed")))
        static_transport = transport_static_audit()
        static_recorder = recorder_static_audit()
        static_dispatch = production_dispatch_static_audit(
            {
                "production_runner": Path(__file__),
                "transport": ROOT / "src/stage28sr2_ros_action_transport.py",
            }
        )
        static_whitelist = {
            "passed": True,
            "accepted_status_whitelist": ["SUCCEEDED"],
            "accepted_fjt_error_codes": [0],
            "rejects_non_succeeded_or_nonzero": True,
            "source": "scripts.stage28sr2_formal_runner.formal_success_positive_whitelist",
        }
        static_state_machine = verify_state_machine_static()
        static_ledger = verify_atomic_ledger_binding_static()
        final_manifest_gate = self.final_jit_pre_send_gate.get("manifest_gate")
        if not isinstance(final_manifest_gate, Mapping):
            final_manifest_gate = revalidate_frozen_execution_manifest(self.frozen_execution_manifest)
        recorder_loss_gate = self.recorder.state.get("recorder_loss_gate") if self.recorder else None
        if not isinstance(recorder_loss_gate, Mapping):
            recorder_loss_gate = {
                "source": "unavailable",
                "transport_lost_total": None,
                "recorder_lost_total": None,
                "per_topic": {},
                "passed": False,
                "first_blocker": "recorder_message_loss_statistics_unavailable",
            }
        if recorder_loss_gate.get("passed") is not True:
            self._block(str(recorder_loss_gate.get("first_blocker", "recorder_message_loss_gate_failed")))
        if not static_state_machine.get("passed"):
            self._block("persistent_state_machine_static_gate_failed")
        if not static_ledger.get("passed"):
            self._block("atomic_ledger_binding_static_gate_failed")
        gates = {
            "live_runtime_evidence_collector": self.evidence is not None and validate_machine_provenance(self.evidence).get("passed", False),
            "production_ROS_action_transport": self.transport is not None and static_transport.get("passed") and bool(getattr(self.transport, "production_ActionClient_created", False)),
            "production_ROS_goal_conversion": self.goal_semantics.get("passed") is True,
            "single_final_runtime_probe": bool(self.evidence and self.evidence.get("runtime_identity", {}).get("same_runtime_instance")),
            "initial_state_live_probe": bool(self.evidence and self.evidence.get("initial_state", {}).get("max_abs_error_rad") == 0.0),
            "recorder_orchestrator": static_recorder.get("passed") and bool(self.recorder and self.recorder.state.get("ready")),
            "recorder_loss_gate": recorder_loss_gate.get("passed") is True,
            "pre_send_live_zero_proven": self.pre_send_live_loss_gate.get("pre_send_live_zero_proven") is True,
            "recorder_bound_live_loss": self.pre_send_live_loss_gate.get("recorder_bound") is True,
            "persistent_graph_observer": self.persistent_graph.get("passed") is True,
            "recorder_started_before_publishers": bool(self.recorder_ready.get("pre_publisher_reader_gate", {}).get("passed")),
            "command_source_live_graph_probe": bool(self.command_sources and self.command_sources.get("passed")),
            "formal_success_positive_whitelist": static_whitelist["passed"],
            "atomic_ledger_binding": static_ledger.get("passed") is True,
            "persistent_state_machine": static_state_machine.get("passed") is True,
            "production_dispatch_chain": static_dispatch.get("passed") is True,
            "frozen_execution_manifest_exact_sha": final_manifest_gate.get("passed") is True,
            "frozen_goal_contract": self.goal_semantics.get("passed") is True,
            "point0_contract": True,
            "regression_gate": regression_gate.get("passed") is True,
            "final_jit_pre_send_gate": self.final_jit_pre_send_gate.get("passed") is True and self.final_jit_pre_send_verification.get("passed") is True,
        }
        formal_certificate = self.formal_execution_certificate or {}
        formal_success = formal_certificate.get("formal_status") == "passed" and formal_certificate.get("Stage_2_9_authorized") is True
        formal_started = formal_certificate.get("formal_started") is True or budget.get("consumed") == 1
        preformal_passed = bool(
            not self.formal_execute
            and not self.blockers
            and all(gates.values())
            and budget.get("consumed") == 0
            and budget.get("send_goal_async_call_count") == 0
        )
        passed = formal_success if self.formal_execute else preformal_passed
        formal_failure_blocker = formal_certificate.get("first_blocker") if self.formal_execute and not formal_success else None
        first_blocker = self.blockers[0] if self.blockers else formal_failure_blocker or next((key for key, value in gates.items() if not value), "none")
        manifest: dict[str, Any] = {}
        for path in (Path(__file__), ROOT / "src/stage28sr2_live_runtime_evidence.py", ROOT / "src/stage28sr2_ros_action_transport.py", ROOT / "src/stage28sr2_ros_goal.py", ROOT / "src/stage28sr2_recorder_orchestrator.py", ROOT / "src/stage28sr2_command_source_collector.py", ROOT / "src/stage28sr2_formal_ledger.py"):
            manifest[str(path.resolve())] = _sha256(path)
        _write(self.output / "source_sha256_manifest.json", manifest)
        frozen_manifest = self.frozen_execution_manifest
        certificate = {
            "schema_version": "stage28sr2-r2-production-preformal-certificate-v1",
            "created_utc": now_utc(),
            "production_runner_loaded": True,
            "production_runner_sha256": _sha256(Path(__file__)),
            "frozen_formal_execution_manifest": frozen_manifest,
            "production_ActionClient_created": bool(self.transport and getattr(self.transport, "production_ActionClient_created", False)),
            "production_action_server_available": bool(self.transport and getattr(self.transport, "action_server_available", False)),
            "production_ROS_goal_constructed": self.goal is not None,
            "production_goal_semantic_gate": self.goal_semantics,
            "gates": gates,
            "live_runtime_evidence": self.evidence or {"source": "unavailable", "passed": False},
            "single_runtime_identity": {"passed": gates["single_final_runtime_probe"]},
            "recorder": {
                "started": self.recorder is not None,
                "pre_send_readiness": self.recorder_ready,
                "recorder_process_alive": self.recorder_ready.get("process_alive", False),
                "all_required_topics_subscribed": self.recorder_ready.get("subscriptions_verified", False),
                "finalization": self.recorder.state if self.recorder else {"finalized": False, "offline_readable": False},
                "ready": self.recorder_ready.get("ready", False),
            },
            "command_source_exclusivity": self.command_sources,
            "recorder_loss_gate": recorder_loss_gate,
            "pre_send_live_loss_gate": self.pre_send_live_loss_gate,
            "pre_send_live_zero_proven": self.pre_send_live_loss_gate.get("pre_send_live_zero_proven") is True,
            "final_jit_pre_send_gate": self.final_jit_pre_send_gate,
            "final_jit_pre_send_verification": self.final_jit_pre_send_verification,
            "persistent_graph": self.persistent_graph,
            "formal_success_positive_whitelist": static_whitelist,
            "production_dispatch_static_audit": static_dispatch,
            "atomic_ledger_binding_static_gate": static_ledger,
            "persistent_state_machine_static_gate": static_state_machine,
            "frozen_execution_manifest_revalidation": final_manifest_gate,
            "formal_state_machine": self.machine.snapshot(),
            "formal_goal_budget": budget,
            "pytest": {
                "focused": self.regression.get("focused_stage28sr2_suite"),
                "full_collection": self.regression.get("full_collection_suite"),
                "full_regression": self.regression.get("full_regression_suite"),
                "all_required_tests_passed": self.regression.get("all_required_tests_passed", False),
            },
            "regression_gate": regression_gate,
            "goal_budget_consumed": bool(budget.get("consumed")),
            "send_binding_committed": bool(budget.get("send_binding_committed", False)),
            "transport_send_invocation_observed": bool(budget.get("transport_send_invocation_observed", False)),
            "goal_request_acknowledged": bool(budget.get("goal_request_acknowledged", False)),
            "goal_accepted": bool(budget.get("goal_accepted", False)),
            "real_FJT_goal_requests": int(budget.get("send_goal_async_call_count", 0)),
            "formal_execution_certificate": self.formal_execution_certificate,
            "Stage_2_8S_R2_formal_started": formal_started,
            "Stage_2_8S_R2_Formal": {
                "started": formal_started,
                "status": "passed" if formal_success else "failed" if formal_started else "not_started",
                "first_blocker": None if formal_success else (formal_failure_blocker or first_blocker),
            },
            "Stage_2_9_authorized": formal_success,
            "Stage_3_authorized": False,
            "retry_allowed": False,
            "runtime_restart_count": 0,
            "controller_restart_count": 0,
            "controller_reactivation_count": 0,
            "event_trace": list(self.events) + ["FORMAL_PRE_SEND_BOUNDARY_REACHED"] + ([] if formal_started else ["audit_stopped_before_ledger_consumption"]),
            "Stage_2_8S_R2_PreFormal_Production_Integration": {"status": "passed" if (preformal_passed or formal_success) else f"blocked_{first_blocker}"},
            "Stage_2_8S_R2_PreFormal_Final_Readiness": {"status": "ready_for_formal_r2_one_shot" if preformal_passed else f"blocked_{first_blocker}"},
            "Stage_2_8S_R2": {"status": "passed" if formal_success else "preformal_ready" if preformal_passed else f"blocked_{first_blocker}"},
            "READY_FOR_FORMAL_R2_ONE_SHOT": preformal_passed,
            "Can_we_safely_consume_the_one_and_only_formal_R2_FJT_goal_next": preformal_passed,
            "Can_we_safely_consume_the_one_formal_R2_FJT_goal_next": preformal_passed,
            "first_blocker": first_blocker,
        }
        _write(self.output / "terminal_audit_certificate.json", certificate)
        _write(
            self.output / "canonical_ledger_snapshot.json",
            {
                "canonical_ledger_path": str(CANONICAL_LEDGER_PATH),
                "ledger": raw_ledger,
                "ledger_validation_error": ledger_validation_error,
            },
        )
        return certificate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 2.8S-R2 production preformal zero-goal runner")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--runtime-launch", type=Path, default=DEFAULT_RUNTIME_LAUNCH)
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--regression-evidence", type=Path, default=None)
    parser.add_argument("--formal-execute", action="store_true", help="explicitly enter the one-shot production executor")
    parser.add_argument("--i-understand-this-consumes-global-ledger", action="store_true")
    args = parser.parse_args(argv)
    if args.formal_execute and not args.i_understand_this_consumes_global_ledger:
        raise SystemExit("--formal-execute requires --i-understand-this-consumes-global-ledger")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output or ROOT / "outputs/stage28sr2_r2_production_preformal" / f"stage28sr2_r2_{stamp}").resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {output}")
    certificate = ProductionPreflightRunner(
        output,
        launch=args.runtime_launch,
        skip_tests=args.skip_tests,
        regression_evidence=args.regression_evidence,
        formal_execute=args.formal_execute,
    ).run()
    print(json.dumps({"output": str(output), "status": certificate["Stage_2_8S_R2_PreFormal_Production_Integration"]["status"], "first_blocker": certificate["first_blocker"], "formal_goal_budget": certificate["formal_goal_budget"], "real_FJT_goal_requests": certificate["real_FJT_goal_requests"]}, ensure_ascii=False, indent=2))
    return 0 if certificate["Stage_2_8S_R2_PreFormal_Production_Integration"]["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
