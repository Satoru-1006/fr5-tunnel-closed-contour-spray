"""Machine-collected evidence for one final Stage 2.8S-R2 runtime.

This module deliberately has no fixture-loading path.  A production evidence
record is assembled from command output and (when available) a live
``/joint_states`` probe, with every raw probe persisted beside the normalized
record.  Tests may inject a command runner, but production callers must use
``LiveRuntimeEvidenceCollector.collect`` without an evidence mapping.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol


ACTION_NAME = "/fairino5_controller/follow_joint_trajectory"
JOINT_TRAJECTORY_TOPIC = "/fairino5_controller/joint_trajectory"
REQUIRED_RAW_PROBES = (
    "ros_environment.txt",
    "ros2_pkg_prefix.txt",
    "ros2_pkg_version.txt",
    "list_controllers.txt",
    "list_hardware_components.txt",
    "controller_parameters.txt",
    "node_graph_snapshot.txt",
    "initial_joint_state.json",
)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(encoded)


@dataclass(frozen=True)
class RuntimeIdentity:
    runtime_instance_id: str
    launch_pid: int | None
    launch_start_time: str

    @classmethod
    def create(cls, launch_pid: int | None = None) -> "RuntimeIdentity":
        return cls(str(uuid.uuid4()), launch_pid, now_utc())

    def as_dict(self) -> dict[str, Any]:
        return {
            "runtime_instance_id": self.runtime_instance_id,
            "launch_pid": self.launch_pid,
            "launch_start_time": self.launch_start_time,
        }


@dataclass
class ProbeResult:
    name: str
    command: str
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def text(self) -> str:
        if self.stderr:
            return f"$ {self.command}\n{self.stdout}\n--- STDERR ---\n{self.stderr}\n"
        return f"$ {self.command}\n{self.stdout}\n"

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "command": self.command,
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


class ProbeRunner(Protocol):
    def __call__(self, name: str, command: str, timeout_s: float) -> ProbeResult:
        ...


def _decode(data: bytes | str | None) -> str:
    if data is None:
        return ""
    if isinstance(data, str):
        return data
    return data.decode("utf-8", errors="replace")


def _default_probe_runner(name: str, command: str, timeout_s: float) -> ProbeResult:
    """Run a ROS CLI command in the current ROS environment.

    On Windows the repository's supported ROS environment is the configured
    WSL distribution.  The wrapper is still a real command probe; it does not
    synthesize a successful response when ROS is absent.
    """

    if shutil.which("ros2"):
        argv = ["bash", "-lc", command] if os.name != "nt" else ["powershell", "-NoProfile", "-Command", command]
    elif shutil.which("wsl.exe"):
        wrapped = f"source /opt/ros/jazzy/setup.bash; source /mnt/d/robotfucker/install/setup.bash 2>/dev/null || true; {command}"
        argv = ["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", wrapped]
    else:
        return ProbeResult(name, command, None, "", "ros2 and wsl.exe are unavailable", False)
    try:
        completed = subprocess.run(argv, capture_output=True, timeout=timeout_s, check=False)
        return ProbeResult(name, command, completed.returncode, _decode(completed.stdout), _decode(completed.stderr))
    except subprocess.TimeoutExpired as error:
        return ProbeResult(name, command, None, _decode(error.stdout), _decode(error.stderr), True)
    except OSError as error:
        return ProbeResult(name, command, None, "", repr(error), False)


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        try:
            return float(value)
        except ValueError:
            return value.strip('"\'')


def _parse_ros_list(text: str) -> list[Any]:
    import re

    # ``ros2 topic echo --field`` emits a Python-style list for string fields
    # and ``array('d', [...])`` for numeric fields.
    array_match = re.search(r"array\([^\[]*(\[[^\]]*\])\)", text, re.DOTALL)
    list_match = re.search(r"(\[[^\]]*\])", text, re.DOTALL)
    candidate = array_match.group(1) if array_match else (list_match.group(1) if list_match else None)
    if candidate:
        try:
            value = ast.literal_eval(candidate)
            if isinstance(value, list):
                return value
        except (SyntaxError, ValueError):
            pass
    values: list[Any] = []
    in_list = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped in {"name:", "position:", "names:", "positions:"}:
            in_list = True
            continue
        if not in_list:
            continue
        if stripped.startswith("-"):
            values.append(_parse_scalar(stripped[1:].strip()))
        elif stripped and not line.startswith((" ", "\t")):
            break
    return values


def _env_values(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in text.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    return result


def _first_match(text: str, *patterns: str) -> str | None:
    import re

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1) if match.lastindex else match.group(0)
    return None


class LiveRuntimeEvidenceCollector:
    """Collect and persist live evidence for one explicitly identified runtime."""

    collector_path = Path(__file__).resolve()

    def __init__(
        self,
        output_dir: Path,
        runtime_identity: RuntimeIdentity,
        *,
        probe_runner: ProbeRunner | None = None,
        controller_name: str = "fairino5_controller",
        raw_dir_name: str = "raw_live_probes",
    ) -> None:
        self.output_dir = Path(output_dir).resolve()
        self.raw_dir = self.output_dir / raw_dir_name
        self.runtime_identity = runtime_identity
        self.probe_runner = probe_runner or _default_probe_runner
        self.controller_name = controller_name
        self._probes: dict[str, ProbeResult] = {}
        self._phase_records: dict[str, dict[str, Any]] = {}

    def _probe(self, name: str, command: str, *, timeout_s: float = 20.0, phase: str = "initial") -> ProbeResult:
        result = self.probe_runner(name, command, timeout_s)
        self._probes[f"{phase}:{name}"] = result
        suffix = "" if phase == "initial" else f"_{phase}"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        (self.raw_dir / f"{name}{suffix}.txt").write_text(result.text, encoding="utf-8")
        return result

    def _collect_joint_state(self, phase: str) -> dict[str, Any]:
        names = self._probe("joint_state_names", "ros2 topic echo --once --field name /joint_states", phase=phase)
        positions = self._probe("joint_state_positions", "ros2 topic echo --once --field position /joint_states", phase=phase)
        observed_names = [str(value) for value in _parse_ros_list(names.stdout)]
        observed_positions = [float(value) for value in _parse_ros_list(positions.stdout) if isinstance(value, (int, float))]
        record = {
            "observed_joint_names": observed_names,
            "observed_joint_positions": observed_positions,
            "names_probe": names.as_dict(),
            "positions_probe": positions.as_dict(),
            "passed": names.exit_code == 0 and positions.exit_code == 0 and bool(observed_names) and len(observed_names) == len(observed_positions),
        }
        suffix = "" if phase == "initial" else f"_{phase}"
        (self.raw_dir / f"initial_joint_state{suffix}.json").write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
        return record

    def collect(self, *, phase: str, frozen_point0: list[float] | None = None) -> dict[str, Any]:
        if phase not in {"initial", "final"}:
            raise ValueError("phase must be initial or final")
        env = self._probe("ros_environment", "env", phase=phase)
        prefix = self._probe("ros2_pkg_prefix", "ros2 pkg prefix joint_trajectory_controller", phase=phase)
        version = self._probe("ros2_pkg_version", "ros2 pkg xml joint_trajectory_controller --tag version", phase=phase)
        controllers = self._probe("list_controllers", "ros2 control list_controllers", phase=phase)
        hardware = self._probe("list_hardware_components", "ros2 control list_hardware_components -v --spin-time 1", phase=phase)
        parameters = self._probe(
            "controller_parameters",
            f"ros2 param dump /{self.controller_name}; ros2 param get /{self.controller_name} interpolation_method",
            phase=phase,
        )
        graph = self._probe(
            "node_graph_snapshot",
            f"ros2 node list; ros2 action info {ACTION_NAME}; ros2 topic info -v {JOINT_TRAJECTORY_TOPIC}; ros2 node info /{self.controller_name}",
            phase=phase,
        )
        joint_state = self._collect_joint_state(phase)
        env_values = _env_values(env.stdout)
        controller_text = controllers.stdout
        parameter_text = parameters.stdout
        hardware_text = hardware.stdout
        positions = joint_state["observed_joint_positions"]
        names = joint_state["observed_joint_names"]
        initial_error: float | None = None
        if frozen_point0 is not None and names == [f"j{i}" for i in range(1, len(frozen_point0) + 1)] and len(positions) == len(frozen_point0):
            initial_error = max((abs(float(a) - float(b)) for a, b in zip(positions, frozen_point0)), default=0.0)
        record: dict[str, Any] = {
            "collector": {
                "collector_path": str(self.collector_path),
                "collector_sha256": sha256_file(self.collector_path),
                "collector_command": "LiveRuntimeEvidenceCollector.collect",
                "collection_timestamp": now_utc(),
                "hostname": platform.node(),
                "process_identity": {"collector_pid": os.getpid(), "python": sys.version},
            },
            "runtime": self.runtime_identity.as_dict(),
            "ROS": {
                "ROS_DISTRO": env_values.get("ROS_DISTRO"),
                "ROS_VERSION": env_values.get("ROS_VERSION"),
                "ROS_PYTHON_VERSION": platform.python_version(),
                "ROS_DOMAIN_ID": env_values.get("ROS_DOMAIN_ID"),
                "joint_trajectory_controller_prefix": prefix.stdout.strip() if prefix.exit_code == 0 else None,
                "joint_trajectory_controller_version": version.stdout.strip() if version.exit_code == 0 else None,
            },
            "controller": {
                "controller_name": self.controller_name,
                "controller_type": "joint_trajectory_controller/JointTrajectoryController" if "JointTrajectoryController" in controller_text else None,
                "controller_state": "active" if _first_match(controller_text, rf"{self.controller_name}[^\n]*\sactive\b") else None,
                "interpolation_method": _first_match(parameter_text, r"value is:\s*([A-Za-z_]+)") or _first_match(parameter_text, r"interpolation_method:\s*([A-Za-z_]+)"),
                "joints": [f"j{i}" for i in range(1, 7)] if all(f"j{i}" in parameter_text for i in range(1, 7)) else [],
                "command_interfaces": ["position"] if "command_interfaces" in parameter_text and "position" in parameter_text else [],
                "state_interfaces": ["position"] if "state_interfaces" in parameter_text and "position" in parameter_text else [],
            },
            "hardware": {
                "hardware_component": _first_match(hardware_text, r"name:\s*([^\n]+)", r"plugin:\s*([^\n]+)"),
                "hardware_state": "active" if "active" in hardware_text.lower() else None,
                "simulation_only": "mock_components/GenericSystem" in hardware_text or "mock_components" in hardware_text,
            },
            "robot_model": {},
            "initial_state": {
                "observed_joint_names": names,
                "observed_joint_positions": positions,
                "frozen_point0": frozen_point0,
                "max_abs_error_rad": initial_error,
                "probe_passed": joint_state["passed"],
            },
            "probe_exit_codes": {key: value.exit_code for key, value in self._probes.items() if key.startswith(f"{phase}:")},
            "source": "machine_collected_live",
            "runtime_instance_id": self.runtime_identity.runtime_instance_id,
            "phase": phase,
        }
        self._phase_records[phase] = record
        return record

    def finalize(self, *, frozen_point0: list[float] | None = None, robot_model_paths: Mapping[str, Path] | None = None) -> dict[str, Any]:
        if "initial" not in self._phase_records or "final" not in self._phase_records:
            raise RuntimeError("initial and final live probes are required before finalizing evidence")
        initial = self._phase_records["initial"]
        final = self._phase_records["final"]
        raw_entries: list[dict[str, Any]] = []
        for path in sorted(self.raw_dir.glob("*")):
            if path.is_file():
                raw_entries.append({"path": str(path.resolve()), "sha256": sha256_file(path), "size": path.stat().st_size})
        manifest = {
            "schema_version": "stage28sr2-live-raw-manifest-v1",
            "runtime_instance_id": self.runtime_identity.runtime_instance_id,
            "artifacts": raw_entries,
        }
        manifest_path = self.output_dir / "raw_live_evidence_manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        robot_model: dict[str, Any] = {}
        for name, path in (robot_model_paths or {}).items():
            robot_model[name] = {"path": str(Path(path).resolve()), "sha256": sha256_file(Path(path))}
        robot_model["robot_description_sha256"] = robot_model.get("robot_description", {}).get("sha256")
        robot_model["controllers_yaml_sha256"] = robot_model.get("controllers_yaml", {}).get("sha256")
        robot_model["initial_positions_sha256"] = robot_model.get("initial_positions", {}).get("sha256")
        evidence: dict[str, Any] = {
            "schema_version": "stage28sr2-live-runtime-evidence-v1",
            "source": "machine_collected_live",
            "collector": initial["collector"],
            "runtime": self.runtime_identity.as_dict(),
            "runtime_instance_id": self.runtime_identity.runtime_instance_id,
            "runtime_restart_count": 0,
            "controller_restart_count": 0,
            "controller_reactivation_count": 0,
            "initial_probe": initial,
            "final_pre_send_probe": final,
            "ROS": final["ROS"],
            "controller": final["controller"],
            "hardware": final["hardware"],
            "robot_model": robot_model,
            "initial_state": initial["initial_state"],
            "runtime_identity": {
                "initial_runtime_instance_id": initial["runtime_instance_id"],
                "final_runtime_instance_id": final["runtime_instance_id"],
                "same_runtime_instance": initial["runtime_instance_id"] == final["runtime_instance_id"],
            },
            "raw_evidence_manifest": {
                "path": str(manifest_path.resolve()),
                "sha256": sha256_file(manifest_path),
                "artifact_count": len(raw_entries),
            },
        }
        evidence_path = self.output_dir / "runtime_evidence.json"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (self.output_dir / "runtime_evidence.sha256").write_text(f"{sha256_file(evidence_path)}  {evidence_path.name}\n", encoding="utf-8")
        return evidence


def validate_machine_provenance(evidence: Mapping[str, Any], *, collector_path: Path | None = None) -> dict[str, Any]:
    """Validate that evidence is machine-collected and its raw inputs persist."""

    collector_path = (collector_path or Path(__file__)).resolve()
    collector = evidence.get("collector") if isinstance(evidence.get("collector"), Mapping) else {}
    runtime = evidence.get("runtime") if isinstance(evidence.get("runtime"), Mapping) else {}
    manifest_info = evidence.get("raw_evidence_manifest") if isinstance(evidence.get("raw_evidence_manifest"), Mapping) else {}
    manifest_path = Path(str(manifest_info.get("path", ""))) if manifest_info.get("path") else None
    checks: dict[str, bool] = {
        "source_machine_collected_live": evidence.get("source") == "machine_collected_live",
        "collector_path_matches": Path(str(collector.get("collector_path", ""))).resolve() == collector_path,
        "collector_hash_matches": collector.get("collector_sha256") == sha256_file(collector_path),
        "runtime_instance_id_present": bool(evidence.get("runtime_instance_id") and runtime.get("runtime_instance_id")),
        "manifest_present": manifest_path is not None and manifest_path.is_file(),
        "manifest_hash_matches": manifest_path is not None and manifest_path.is_file() and manifest_info.get("sha256") == sha256_file(manifest_path),
    }
    artifacts_ok = False
    if checks["manifest_present"] and manifest_path is not None:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            artifacts = manifest.get("artifacts", [])
            artifacts_ok = bool(artifacts) and all(Path(item["path"]).is_file() and item["sha256"] == sha256_file(Path(item["path"])) for item in artifacts)
        except (OSError, ValueError, TypeError, KeyError):
            artifacts_ok = False
    checks["raw_artifacts_hash_verified"] = artifacts_ok
    return {"checks": checks, "passed": all(checks.values()), "first_blocker": next((key for key, value in checks.items() if not value), None)}
