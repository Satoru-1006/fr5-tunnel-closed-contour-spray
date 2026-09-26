"""Stage 2.8S-R simulation-only FJT recorder.

The inherited clean action client sends exactly one frozen FJT goal.  This
remediation recorder keeps every received transform in ``/tf`` and
``/tf_static``; no child-frame filtering or capture-time state pairing is
performed here.  Capture monotonic time is diagnostic metadata only.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from tf2_msgs.msg import TFMessage

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stage27r_clean_action_client import CleanClient as Stage27CleanClient  # noqa: E402


def stamp_dict(value) -> dict:
    return {
        "sec": int(value.sec),
        "nanosec": int(value.nanosec),
        "seconds": float(value.sec) + float(value.nanosec) * 1.0e-9,
    }


def transform_dict(value: TransformStamped, *, message_index: int, transform_index: int, capture_s: float) -> dict:
    stamp = stamp_dict(value.header.stamp)
    return {
        "tf_message_index": int(message_index),
        "transform_index": int(transform_index),
        "capture_monotonic_s": float(capture_s),
        "ros_message_header_stamp": dict(stamp),
        "header_stamp": dict(stamp),
        "parent_frame": str(value.header.frame_id),
        "child_frame": str(value.child_frame_id),
        "translation": {
            "x": float(value.transform.translation.x),
            "y": float(value.transform.translation.y),
            "z": float(value.transform.translation.z),
        },
        "quaternion_xyzw": {
            "x": float(value.transform.rotation.x),
            "y": float(value.transform.rotation.y),
            "z": float(value.transform.rotation.z),
            "w": float(value.transform.rotation.w),
        },
    }


def run_command(arguments: list[str], timeout_s: int = 40) -> dict:
    try:
        proc = subprocess.run(arguments, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout_s)
        return {"command": arguments, "exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
    except Exception as exc:  # pragma: no cover - runtime diagnostic fallback
        return {"command": arguments, "exit_code": None, "stdout": "", "stderr": repr(exc)}


def parse_param_value(text: str):
    value = text.strip()
    match = re.search(r"value is:\s*(.*)$", value, re.IGNORECASE)
    value = match.group(1).strip() if match else value
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if not value:
        return ""
    try:
        return float(value)
    except ValueError:
        return value


def indent_block(text: str, spaces: int = 6) -> str:
    prefix = " " * spaces
    return "\n".join(prefix + line for line in text.rstrip("\n").splitlines())


class Stage28SRClient(Stage27CleanClient):
    """One-goal action client with complete raw-message capture."""

    def __init__(self, output: Path, goal_data: dict) -> None:
        super().__init__(output, goal_data)
        self.tf_rows: list[dict] = []
        self.tf_static_rows: list[dict] = []
        self.tf_message_count = 0
        self.tf_static_message_count = 0
        tf_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=100,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        static_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=100,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.tf_sub = self.create_subscription(TFMessage, "/tf", self._tf_callback, tf_qos)
        self.tf_static_sub = self.create_subscription(TFMessage, "/tf_static", self._tf_static_callback, static_qos)

    def _tf_callback(self, message: TFMessage) -> None:
        message_index = self.tf_message_count
        self.tf_message_count += 1
        capture_s = time.monotonic()
        for transform_index, transform in enumerate(message.transforms):
            self.tf_rows.append(transform_dict(transform, message_index=message_index, transform_index=transform_index, capture_s=capture_s))

    def _tf_static_callback(self, message: TFMessage) -> None:
        message_index = self.tf_static_message_count
        self.tf_static_message_count += 1
        capture_s = time.monotonic()
        for transform_index, transform in enumerate(message.transforms):
            self.tf_static_rows.append(transform_dict(transform, message_index=message_index, transform_index=transform_index, capture_s=capture_s))

    def _joint_callback(self, msg) -> None:
        stamp = stamp_dict(msg.header.stamp)
        self.joint_state_rows.append({
            "message_index": len(self.joint_state_rows),
            "capture_monotonic_s": time.monotonic(),
            "ros_message_header_stamp": dict(stamp),
            "header_stamp": dict(stamp),
            "names": [str(x) for x in msg.name],
            "name": [str(x) for x in msg.name],
            "positions": [float(x) for x in msg.position],
            "position": [float(x) for x in msg.position],
            "velocities": [float(x) for x in msg.velocity],
            "velocity": [float(x) for x in msg.velocity],
            "effort": [float(x) for x in msg.effort],
        })

    def _write_jsonl(self, name: str, rows: list[dict]) -> None:
        with (self.output / name).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=True) + "\n")

    def capture_runtime_identity(self) -> None:
        dump = run_command(["ros2", "param", "dump", "/robot_state_publisher"], timeout_s=60)
        params = {}
        for name in ("publish_frequency", "ignore_timestamp", "frame_prefix", "use_sim_time"):
            result = run_command(["ros2", "param", "get", "/robot_state_publisher", name], timeout_s=20)
            params[name] = parse_param_value(result.get("stdout", "")) if result.get("exit_code") == 0 else None
            params[f"{name}_probe"] = result

        urdf = self.output / "stage28sr_expanded_runtime.urdf"
        robot_description_sha256 = hashlib.sha256(urdf.read_bytes()).hexdigest() if urdf.exists() else None
        rsp = {
            "schema_version": "stage28sr-rsp-parameters-v1",
            "robot_state_publisher": {
                "publish_frequency": params.get("publish_frequency"),
                "ignore_timestamp": params.get("ignore_timestamp"),
                "frame_prefix": params.get("frame_prefix"),
                "use_sim_time": params.get("use_sim_time"),
                "robot_description_sha256": robot_description_sha256,
            },
            "ros2_param_dump_command": dump["command"],
            "ros2_param_dump_exit_code": dump["exit_code"],
            "ros2_param_dump_raw": dump.get("stdout", ""),
            "parameter_probes": params,
            "captured_before_FJT_goal": True,
        }
        lines = [
            "schema_version: stage28sr-rsp-parameters-v1",
            "robot_state_publisher:",
            f"  publish_frequency: {json.dumps(rsp['robot_state_publisher']['publish_frequency'])}",
            f"  ignore_timestamp: {json.dumps(rsp['robot_state_publisher']['ignore_timestamp'])}",
            f"  frame_prefix: {json.dumps(rsp['robot_state_publisher']['frame_prefix'], ensure_ascii=False)}",
            f"  use_sim_time: {json.dumps(rsp['robot_state_publisher']['use_sim_time'])}",
            f"  robot_description_sha256: {json.dumps(robot_description_sha256)}",
            "captured_before_FJT_goal: true",
            "ros2_param_dump:",
            f"  command: {json.dumps(' '.join(dump['command']))}",
            f"  exit_code: {dump['exit_code']}",
            "  raw: |",
            indent_block(dump.get("stdout", ""), 4),
        ]
        (self.output / "stage28sr_rsp_parameters.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")

        graph_commands = [
            ["ros2", "node", "info", "/robot_state_publisher"],
            ["ros2", "topic", "info", "-v", "/joint_states"],
            ["ros2", "topic", "info", "-v", "/tf"],
            ["ros2", "topic", "info", "-v", "/tf_static"],
        ]
        graph_lines = ["Stage 2.8S-R ROS graph runtime identity", "phase: before FJT goal, recorder subscriptions active", ""]
        for command in graph_commands:
            result = run_command(command, timeout_s=40)
            graph_lines.extend([f"$ {' '.join(command)}", f"exit_code: {result['exit_code']}", result.get("stdout", "").rstrip(), "stderr:", result.get("stderr", "").rstrip(), "", "---", ""])
        (self.output / "stage28sr_ros_graph_runtime.txt").write_text("\n".join(graph_lines), encoding="utf-8")

    def wait_ready(self) -> None:
        super().wait_ready()
        # This occurs after subscriptions are connected and before make_goal()
        # is called by CleanClient.run().
        self.capture_runtime_identity()

    def _write_streams(self, result: dict) -> None:
        super()._write_streams(result)
        self._write_jsonl("stage28sr_joint_states_raw.jsonl", self.joint_state_rows)
        self._write_jsonl("stage28sr_controller_state_raw.jsonl", self.state_rows)
        self._write_jsonl("stage28sr_tf_raw.jsonl", self.tf_rows)
        self._write_jsonl("stage28sr_tf_static_raw.jsonl", self.tf_static_rows)
        (self.output / "stage28sr_fjt_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (self.output / "stage28sr_raw_capture_manifest.json").write_text(
            json.dumps({
                "schema_version": "stage28sr-raw-capture-v1",
                "topics": ["/joint_states", "/tf", "/tf_static", "/fairino5_controller/controller_state"],
                "joint_states_rows": len(self.joint_state_rows),
                "controller_state_rows": len(self.state_rows),
                "tf_message_count": self.tf_message_count,
                "tf_transform_records": len(self.tf_rows),
                "tf_static_message_count": self.tf_static_message_count,
                "tf_static_transform_records": len(self.tf_static_rows),
                "tf_capture_filter": "none; every TFMessage.transforms[] record retained",
                "physical_pairing_key": "exact ROS header_stamp; capture_monotonic_s is diagnostic only",
            }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: stage28sr_runtime_client.py OUTPUT_DIR")
    output = Path(sys.argv[1]).resolve()
    goal_data = json.loads((output / "stage28sr_formal_fjt_goal.json").read_text(encoding="utf-8"))
    rclpy.init()
    node = Stage28SRClient(output, goal_data)
    try:
        result = node.run()
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result.get("goal_accepted") and result.get("execution_completed") else 2
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
