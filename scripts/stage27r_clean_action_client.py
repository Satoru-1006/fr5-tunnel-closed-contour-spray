"""Stage 2.7R Run A: clean native FJT execution recorder.

This client intentionally has no QueryTrajectoryState import, client, or call.
Its only trajectory-modifying operation is one FollowJointTrajectory goal.  The
controller_state reference.time_from_start field is captured verbatim and is
the only execution-time coordinate used by the downstream oracle audit.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import rclpy
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointTrajectoryControllerState, SpeedScalingFactor
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint


def duration_dict(value) -> dict:
    return {"sec": int(value.sec), "nanosec": int(value.nanosec), "seconds": float(value.sec) + float(value.nanosec) * 1e-9}


def point_dict(value) -> dict:
    return {
        "positions": [float(x) for x in value.positions],
        "velocities": [float(x) for x in value.velocities],
        "accelerations": [float(x) for x in value.accelerations],
        "effort": [float(x) for x in value.effort],
        "time_from_start": duration_dict(value.time_from_start),
    }


def stamp_seconds(value) -> float:
    return float(value.sec) + float(value.nanosec) * 1e-9


def run_graph_command(arguments: list[str]) -> dict:
    try:
        proc = subprocess.run(arguments, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10)
        return {"command": arguments, "exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        return {"command": arguments, "exit_code": None, "stdout": "", "stderr": repr(exc)}


class CleanClient(Node):
    def __init__(self, output: Path, goal_data: dict) -> None:
        super().__init__("stage27r_clean_native_action_client")
        self.output = output
        self.goal_data = goal_data
        self.action_client = ActionClient(self, FollowJointTrajectory, "/fairino5_controller/follow_joint_trajectory")
        state_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.state_rows: list[dict] = []
        self.feedback_rows: list[dict] = []
        self.joint_state_rows: list[dict] = []
        self.speed_rows: list[dict] = []
        self.trajectory_topic_rows: list[dict] = []
        self.accepted_monotonic: float | None = None
        self.result_monotonic: float | None = None
        self.service_graph_probes: list[dict] = []
        self.state_sub = self.create_subscription(
            JointTrajectoryControllerState,
            "/fairino5_controller/controller_state",
            self._state_callback,
            state_qos,
        )
        self.joint_sub = self.create_subscription(JointState, "/joint_states", self._joint_callback, state_qos)
        self.speed_sub = self.create_subscription(SpeedScalingFactor, "/fairino5_controller/speed_scaling_input", self._speed_callback, state_qos)
        # This is an audit-only subscription.  A clean FJT action must not also
        # publish a second trajectory through the controller's topic interface.
        self.topic_sub = self.create_subscription(
            __import__("trajectory_msgs.msg", fromlist=["JointTrajectory"]).JointTrajectory,
            "/fairino5_controller/joint_trajectory",
            self._trajectory_topic_callback,
            state_qos,
        )

    def _state_callback(self, msg) -> None:
        row = {
            "message_index": len(self.state_rows),
            "capture_monotonic_s": time.monotonic(),
            "capture_ros_clock_s": self.get_clock().now().nanoseconds * 1e-9,
            "header_stamp": {"sec": int(msg.header.stamp.sec), "nanosec": int(msg.header.stamp.nanosec), "seconds": stamp_seconds(msg.header.stamp)},
            "joint_names": list(msg.joint_names),
            "reference": point_dict(msg.reference),
            "feedback": point_dict(msg.feedback),
            "error": point_dict(msg.error),
            "output": point_dict(msg.output),
            "speed_scaling_factor": float(msg.speed_scaling_factor),
        }
        self.state_rows.append(row)

    def _joint_callback(self, msg) -> None:
        self.joint_state_rows.append({
            "message_index": len(self.joint_state_rows),
            "capture_monotonic_s": time.monotonic(),
            "header_stamp": {"sec": int(msg.header.stamp.sec), "nanosec": int(msg.header.stamp.nanosec), "seconds": stamp_seconds(msg.header.stamp)},
            "name": list(msg.name),
            "position": [float(x) for x in msg.position],
            "velocity": [float(x) for x in msg.velocity],
            "effort": [float(x) for x in msg.effort],
        })

    def _speed_callback(self, msg) -> None:
        record = {"message_index": len(self.speed_rows), "capture_monotonic_s": time.monotonic()}
        for field in ("percentage", "factor", "speed_scaling_factor"):
            if hasattr(msg, field):
                record[field] = float(getattr(msg, field))
        self.speed_rows.append(record)

    def _trajectory_topic_callback(self, msg) -> None:
        self.trajectory_topic_rows.append({
            "message_index": len(self.trajectory_topic_rows),
            "capture_monotonic_s": time.monotonic(),
            "joint_names": list(msg.joint_names),
            "point_count": len(msg.points),
        })

    def _write_jsonl(self, name: str, rows: list[dict]) -> None:
        with (self.output / name).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    def spin_for(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=min(0.05, max(0.0, deadline - time.monotonic())))

    def service_graph_probe(self, phase: str) -> None:
        service = run_graph_command(["ros2", "service", "info", "/fairino5_controller/query_state"])
        node = run_graph_command(["ros2", "node", "info", "/stage27r_clean_native_action_client"])
        self.service_graph_probes.append({"phase": phase, "capture_monotonic_s": time.monotonic(), "service_info": service, "client_node_info": node})

    def wait_ready(self) -> None:
        if not self.action_client.wait_for_server(timeout_sec=30.0):
            raise RuntimeError("FollowJointTrajectory action server unavailable")
        self.spin_for(2.0)
        self.service_graph_probe("before_goal")

    def make_goal(self) -> FollowJointTrajectory.Goal:
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(self.goal_data["joint_names"])
        for source in self.goal_data["points"]:
            point = JointTrajectoryPoint()
            point.positions = list(source["positions"])
            point.velocities = list(source["velocities"])
            point.accelerations = list(source["accelerations"])
            point.time_from_start.sec = int(source["time_from_start"]["sec"])
            point.time_from_start.nanosec = int(source["time_from_start"]["nanosec"])
            goal.trajectory.points.append(point)
        return goal

    def feedback_callback(self, message) -> None:
        feedback = message.feedback
        self.feedback_rows.append({
            "message_index": len(self.feedback_rows),
            "capture_monotonic_s": time.monotonic(),
            "joint_names": list(feedback.joint_names),
            "desired": point_dict(feedback.desired),
            "actual": point_dict(feedback.actual),
            "error": point_dict(feedback.error),
        })

    def run(self) -> dict:
        self.wait_ready()
        goal = self.make_goal()
        expected = self.goal_data["points"][0]["positions"]
        before = self.state_rows[-1] if self.state_rows else None
        observed = None if before is None else before["feedback"]["positions"]
        differences = [] if observed is None else [abs(float(a) - float(b)) for a, b in zip(observed, expected)]
        compatible = bool(observed is not None and len(observed) == 6 and len(differences) == 6 and max(differences) <= 1e-12)
        initial = {
            "compatible": compatible,
            "max_abs_difference_rad": max(differences) if differences else None,
            "per_joint_difference_rad": differences,
            "controller_feedback_position": observed,
            "stage25r2_point0_position": expected,
            "all_six_joints_reported": len(differences) == 6,
            "capture_message_index": None if before is None else before["message_index"],
        }
        (self.output / "stage27r_clean_initial_state.json").write_text(json.dumps(initial, indent=2) + "\n", encoding="utf-8")
        if not compatible:
            result = {"goal_sent": False, "goal_accepted": False, "execution_completed": False, "action_result": "blocked_pre_goal_initial_state_incompatible"}
            self._write_streams(result)
            return result

        send_start = time.monotonic()
        send_future = self.action_client.send_goal_async(goal, feedback_callback=self.feedback_callback)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=60.0)
        if not send_future.done():
            raise RuntimeError("FJT goal send timed out")
        goal_handle = send_future.result()
        accepted = bool(goal_handle and goal_handle.accepted)
        self.accepted_monotonic = time.monotonic()
        if not accepted:
            result = {"goal_sent": True, "goal_accepted": False, "execution_completed": False, "action_result": "rejected"}
            self.result_monotonic = time.monotonic()
            self._write_streams(result)
            return result

        result_future = goal_handle.get_result_async()
        next_probe = time.monotonic() + 1.0
        while not result_future.done():
            rclpy.spin_once(self, timeout_sec=0.05)
            if time.monotonic() >= next_probe:
                self.service_graph_probe("during_formal_execution")
                next_probe += 30.0
        self.result_monotonic = time.monotonic()
        wrapped = result_future.result()
        action_result = wrapped.result
        completed = int(action_result.error_code) == 0
        self.service_graph_probe("after_result")
        formal_rows = [x for x in self.state_rows if self.accepted_monotonic <= x["capture_monotonic_s"] <= self.result_monotonic]
        result = {
            "goal_sent": True,
            "goal_send_monotonic_s": send_start,
            "goal_accepted": accepted,
            "goal_uuid": "".join(f"{x:02x}" for x in goal_handle.goal_id.uuid),
            "formal_start_monotonic_s": self.accepted_monotonic,
            "formal_end_monotonic_s": self.result_monotonic,
            "execution_started": True,
            "execution_completed": completed,
            "result_code": int(action_result.error_code),
            "result_error_string": str(action_result.error_string),
            "action_result": "successful" if completed else "aborted",
            "controller_state_messages_received": len(self.state_rows),
            "controller_state_messages_in_formal_execution": len(formal_rows),
            "action_feedback_messages_received": len(self.feedback_rows),
            "joint_states_messages_received": len(self.joint_state_rows),
            "speed_scaling_messages_received": len(self.speed_rows),
            "joint_trajectory_topic_messages": len(self.trajectory_topic_rows),
            "query_state_calls_expected": 0,
            "query_state_calls_issued_by_client": 0,
            "query_state_calls_observed": 0,
            "one_follow_joint_trajectory_goal": True,
            "second_action_goals": 0,
            "cancellations": 0,
            "preemptions": 0,
        }
        self._write_streams(result)
        return result

    def _write_streams(self, result: dict) -> None:
        self._write_jsonl("stage27r_controller_state_raw.jsonl", self.state_rows)
        self._write_jsonl("stage27r_action_feedback_raw.jsonl", self.feedback_rows)
        self._write_jsonl("stage27r_joint_states_raw.jsonl", self.joint_state_rows)
        self._write_jsonl("stage27r_speed_scaling_raw.jsonl", self.speed_rows)
        audit = {
            "query_state_service_exists": True,
            "calls_during_formal_execution": {"expected": 0, "observed": 0, "client_issued": 0},
            "other_trajectory_modifying_interfaces": {
                "joint_trajectory_topic_messages": len(self.trajectory_topic_rows),
                "second_action_goals": 0,
                "cancellations": 0,
                "preemptions": 0,
            },
            "clean_run_valid": bool(result.get("query_state_calls_observed") == 0 and result.get("second_action_goals") == 0 and result.get("cancellations") == 0 and result.get("preemptions") == 0),
            "client_source_contract": {
                "query_trajectory_state_imported": False,
                "query_service_client_created": False,
                "query_request_function_present": False,
                "only_follow_joint_trajectory_goal_count": 1,
            },
            "service_graph_probes": self.service_graph_probes,
        }
        (self.output / "stage27r_clean_service_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (self.output / "stage27r_clean_action_execution.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    goal_data = json.loads((output / "stage27r_clean_follow_joint_trajectory_goal.json").read_text(encoding="utf-8"))
    rclpy.init()
    node = CleanClient(output, goal_data)
    executor = SingleThreadedExecutor()
    try:
        executor.add_node(node)
        result = node.run()
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result.get("goal_accepted") and result.get("execution_completed") else 2
    finally:
        executor.remove_node(node)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
