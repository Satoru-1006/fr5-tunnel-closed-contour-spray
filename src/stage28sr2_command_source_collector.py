"""Live ROS graph probes for command-source exclusivity."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping


ACTION_NAME = "/fairino5_controller/follow_joint_trajectory"
COMMAND_TOPIC = "/fairino5_controller/joint_trajectory"


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _default_runner(command: str) -> tuple[int | None, str, str]:
    if os.name == "nt":
        argv = ["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", f"source /opt/ros/jazzy/setup.bash; source /mnt/d/robotfucker/install/setup.bash 2>/dev/null || true; {command}"]
    else:
        argv = ["bash", "-lc", command]
    try:
        proc = subprocess.run(argv, capture_output=True, timeout=20, check=False)
        return proc.returncode, proc.stdout.decode("utf-8", errors="replace"), proc.stderr.decode("utf-8", errors="replace")
    except (OSError, subprocess.TimeoutExpired) as error:
        return None, "", repr(error)


class LiveCommandSourceCollector:
    """Collect machine graph snapshots and calculate authorized-source counts."""

    collector_path = Path(__file__).resolve()

    def __init__(self, output_dir: Path, *, runner: Callable[[str], tuple[int | None, str, str]] | None = None) -> None:
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir = self.output_dir / "raw_live_probes"
        self.runner = runner or _default_runner

    def _probe(self, name: str, command: str) -> dict[str, Any]:
        code, stdout, stderr = self.runner(command)
        record = {"command": command, "exit_code": code, "stdout": stdout, "stderr": stderr}
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        (self.raw_dir / name).write_text(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return record

    @staticmethod
    def _section_lines(text: str, heading: str) -> list[str]:
        lines = text.splitlines()
        start = next((i for i, line in enumerate(lines) if line.strip().lower().startswith(heading.lower())), None)
        if start is None:
            return []
        result: list[str] = []
        for line in lines[start + 1:]:
            if re.match(r"^(Action clients|Action servers|Publisher count|Subscription count|Type|Topic type|Node name):", line.strip(), re.I):
                break
            result.append(line.strip())
        return result

    @staticmethod
    def _node_names(text: str) -> list[str]:
        return [line.strip() for line in text.splitlines() if line.strip().startswith("/")]

    def collect(
        self,
        *,
        authorized_action_client_nodes: set[str] | None = None,
        authorized_process_ids: set[int] | None = None,
    ) -> dict[str, Any]:
        authorized = {name if name.startswith("/") else f"/{name}" for name in (authorized_action_client_nodes or set())}
        allowed_pids = {int(pid) for pid in (authorized_process_ids or set())}
        graph = self._probe("command_source_node_graph.json", f"ros2 node list; ros2 action info {ACTION_NAME}; ros2 topic info -v {COMMAND_TOPIC}")
        process = self._probe("command_source_process_snapshot.json", "ps -ef")
        graph_text = graph["stdout"]
        nodes = self._node_names(graph_text)
        action_lines = self._section_lines(graph_text, "Action clients")
        server_lines = self._section_lines(graph_text, "Action servers")
        publisher_lines = self._section_lines(graph_text, "Publisher count")
        action_clients = [line.split()[0] for line in action_lines if line.strip().startswith("/")]
        action_servers = [line.split()[0] for line in server_lines if line.strip().startswith("/")]
        publisher_nodes = [line.split()[0] for line in publisher_lines if line.strip().startswith("/")]
        # ros2 action info commonly prints one node per line without a stable
        # heading indentation.  Keep a raw-derived fallback that only counts
        # explicit node-looking lines, never a hard-coded zero.
        if not action_clients:
            action_clients = [line.strip() for line in graph_text.splitlines() if "client" in line.lower() and line.strip().startswith("/")]
        if not publisher_nodes and re.search(r"Publisher count:\s*[1-9]", graph_text, re.I):
            publisher_nodes = [
                match.group(1).strip()
                for match in re.finditer(r"Node name:\s*([^\n]+).*?Endpoint type:\s*PUBLISHER", graph_text, re.I | re.S)
            ]
        known_old = [name for name in nodes if re.search(r"stage27|stage28(?!sr2)|runtime_client|rqt", name, re.I)]
        moveit = [name for name in nodes if re.search(r"move_group|moveit", name, re.I)]
        conflicting_processes = [
            line for line in process["stdout"].splitlines()
            if re.search(r"stage27|stage28|move_group|moveit|rqt.*trajectory", line, re.I)
            and not (
                len(line.split()) >= 2
                and line.split()[1].isdigit()
                and int(line.split()[1]) in allowed_pids
            )
            and not (str(os.getpid()) in line and "stage28sr2_production_runner.py" in line)
            and not re.search(r"stage28s_runtime\.launch\.py|controller_manager/ros2_control_node|robot_state_publisher|static_transform_publisher|ros2\s+bag\s+record", line, re.I)
        ]
        unauthorized_action_clients = sorted(set(action_clients) - authorized)
        unauthorized_publishers = sorted(set(publisher_nodes))
        report = {
            "source": "machine_collected_live",
            "collector_path": str(self.collector_path),
            "collector_sha256": sha256_file(self.collector_path),
            "action_interface": ACTION_NAME,
            "topic_interface": COMMAND_TOPIC,
            "action_servers": action_servers,
            "action_clients": action_clients,
            "authorized_action_client_nodes": sorted(authorized),
            "authorized_process_ids": sorted(allowed_pids),
            "unauthorized_action_clients": unauthorized_action_clients,
            "publisher_nodes": publisher_nodes,
            "publisher_gids_qos_raw": graph_text,
            "unauthorized_publishers": unauthorized_publishers,
            "moveit_execution_nodes": moveit,
            "old_runtime_clients": known_old,
            "conflicting_project_processes": conflicting_processes,
            "observed_from_runtime_graph": graph["exit_code"] == 0,
            "checks": {
                "exactly_authorized_action_clients": not unauthorized_action_clients,
                "no_unauthorized_joint_trajectory_publishers": not unauthorized_publishers,
                "no_moveit_execution_nodes": not moveit,
                "no_old_runtime_clients": not known_old,
                "no_conflicting_project_processes": not conflicting_processes,
            },
        }
        report["passed"] = bool(report["observed_from_runtime_graph"] and all(report["checks"].values()))
        report["first_blocker"] = next((name for name, passed in report["checks"].items() if not passed), None)
        path = self.output_dir / "command_source_exclusivity.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        report["artifact_path"] = str(path.resolve())
        report["artifact_sha256"] = sha256_file(path)
        return report


def verify_command_source_report(report: Mapping[str, Any], *, authorized_node: str | None = None) -> dict[str, Any]:
    """Pure verifier used by unit tests and the production gate."""

    unauthorized_clients = list(report.get("unauthorized_action_clients", []))
    unauthorized_publishers = list(report.get("unauthorized_publishers", []))
    if authorized_node is not None:
        clients = set(report.get("action_clients", []))
        expected = authorized_node if authorized_node.startswith("/") else f"/{authorized_node}"
        client_check = clients <= {expected}
    else:
        client_check = not unauthorized_clients
    checks = {
        "live_graph_observed": report.get("observed_from_runtime_graph") is True,
        "authorized_action_clients_only": client_check,
        "unauthorized_publishers_empty": not unauthorized_publishers,
        "moveit_execution_nodes_empty": not report.get("moveit_execution_nodes"),
        "old_runtime_clients_empty": not report.get("old_runtime_clients"),
        "conflicting_processes_empty": not report.get("conflicting_project_processes"),
    }
    return {"checks": checks, "passed": all(checks.values()), "first_blocker": next((key for key, value in checks.items() if not value), None)}
