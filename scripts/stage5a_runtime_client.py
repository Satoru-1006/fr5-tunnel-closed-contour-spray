"""Stage 5A one-goal FJT recorder for the isolated GenericSystem chain."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import rclpy
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointTrajectoryControllerState
from geometry_msgs.msg import TransformStamped
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from tf2_msgs.msg import TFMessage
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


JOINTS = [f"j{i}" for i in range(1, 7)]


def stamp(value) -> dict:
    return {"sec": int(value.sec), "nanosec": int(value.nanosec), "seconds": float(value.sec) + float(value.nanosec) * 1e-9}


def point(value) -> dict:
    return {
        "positions": [float(x) for x in value.positions],
        "velocities": [float(x) for x in value.velocities],
        "accelerations": [float(x) for x in value.accelerations],
        "effort": [float(x) for x in value.effort],
        "time_from_start": stamp(value.time_from_start),
    }


def transform(value: TransformStamped) -> dict:
    return {
        "parent_frame": str(value.header.frame_id),
        "child_frame": str(value.child_frame_id),
        "header_stamp": stamp(value.header.stamp),
        "translation": {"x": float(value.transform.translation.x), "y": float(value.transform.translation.y), "z": float(value.transform.translation.z)},
        "rotation": {"x": float(value.transform.rotation.x), "y": float(value.transform.rotation.y), "z": float(value.transform.rotation.z), "w": float(value.transform.rotation.w)},
    }


class Stage5AClient(Node):
    def __init__(self, output: Path, goal: dict, initial_tolerance: float = 1.0e-9) -> None:
        super().__init__("stage5a_runtime_client")
        self.output = output
        self.goal = goal
        self.initial_tolerance = float(initial_tolerance)
        self.action = ActionClient(self, FollowJointTrajectory, "/fairino5_controller/follow_joint_trajectory")
        reliable = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=100, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE)
        sensor = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=100, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
        static = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=100, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.controller_rows: list[dict] = []
        self.joint_rows: list[dict] = []
        self.feedback_rows: list[dict] = []
        self.tf_rows: list[dict] = []
        self.tf_static_rows: list[dict] = []
        self.state_sub = self.create_subscription(JointTrajectoryControllerState, "/fairino5_controller/controller_state", self._controller, reliable)
        # joint_state_broadcaster uses a reliable transient-local publisher in
        # the Jazzy shadow launch; matching durability avoids a discovery race
        # when the recorder starts after the controller has activated.
        self.joint_sub = self.create_subscription(JointState, "/joint_states", self._joint, static)
        self.tf_sub = self.create_subscription(TFMessage, "/tf", self._tf, sensor)
        self.tf_static_sub = self.create_subscription(TFMessage, "/tf_static", self._tf_static, static)
        self.accepted_at: float | None = None
        self.finished_at: float | None = None

    def _controller(self, message) -> None:
        self.controller_rows.append({
            "message_index": len(self.controller_rows),
            "capture_monotonic_s": time.monotonic(),
            "capture_ros_clock_s": self.get_clock().now().nanoseconds * 1e-9,
            "header_stamp": stamp(message.header.stamp),
            "joint_names": list(message.joint_names),
            "reference": point(message.reference),
            "feedback": point(message.feedback),
            "error": point(message.error),
            "output": point(message.output),
            "speed_scaling_factor": float(message.speed_scaling_factor),
        })

    def _joint(self, message) -> None:
        self.joint_rows.append({
            "message_index": len(self.joint_rows),
            "capture_monotonic_s": time.monotonic(),
            "header_stamp": stamp(message.header.stamp),
            "name": list(message.name),
            "position": [float(x) for x in message.position],
            "velocity": [float(x) for x in message.velocity],
            "effort": [float(x) for x in message.effort],
        })

    def _tf(self, message: TFMessage) -> None:
        for item in message.transforms:
            # Keep the complete dynamic TF tree. The fixed spray_tcp joint is
            # published on /tf_static, so a TCP cross-check must compose the
            # base_link->...->wrist3_link dynamic chain with that static edge.
            row = transform(item)
            row["message_index"] = len(self.tf_rows)
            row["capture_monotonic_s"] = time.monotonic()
            self.tf_rows.append(row)

    def _tf_static(self, message: TFMessage) -> None:
        for item in message.transforms:
            if item.child_frame_id in {"wrist3_link", "spray_tcp_link"}:
                row = transform(item)
                row["message_index"] = len(self.tf_static_rows)
                row["capture_monotonic_s"] = time.monotonic()
                self.tf_static_rows.append(row)

    def spin_for(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=min(0.05, max(0.0, deadline - time.monotonic())))

    def make_goal(self) -> FollowJointTrajectory.Goal:
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(self.goal["joint_names"])
        for source in self.goal["points"]:
            item = JointTrajectoryPoint()
            item.positions = list(source["positions"])
            item.velocities = list(source["velocities"])
            item.accelerations = list(source["accelerations"])
            item.time_from_start.sec = int(source["time_from_start"]["sec"])
            item.time_from_start.nanosec = int(source["time_from_start"]["nanosec"])
            goal.trajectory.points.append(item)
        return goal

    def run(self) -> dict:
        if not self.action.wait_for_server(timeout_sec=60.0):
            result = {"goal_sent": False, "goal_accepted": False, "execution_completed": False, "action_result": "follow_joint_trajectory_unavailable"}
            self._write(result)
            return result
        self.spin_for(2.0)
        expected = list(self.goal["points"][0]["positions"])
        observed = self.controller_rows[-1]["feedback"]["positions"] if self.controller_rows else (self.joint_rows[-1]["position"] if self.joint_rows else None)
        initial_error = [abs(float(a) - float(b)) for a, b in zip(observed or [], expected)]
        initial = {"observed": observed, "expected": expected, "per_joint_abs_error_rad": initial_error, "max_abs_error_rad": max(initial_error) if initial_error else None, "compatibility_tolerance_rad": self.initial_tolerance, "compatible": bool(len(initial_error) == 6 and max(initial_error) <= self.initial_tolerance)}
        (self.output / "stage5a_initial_state.json").write_text(json.dumps(initial, indent=2) + "\n", encoding="utf-8")
        if not initial["compatible"]:
            result = {"goal_sent": False, "goal_accepted": False, "execution_completed": False, "action_result": "initial_state_incompatible"}
            self._write(result)
            return result
        started = time.monotonic()
        future = self.action.send_goal_async(self.make_goal(), feedback_callback=self._feedback)
        rclpy.spin_until_future_complete(self, future, timeout_sec=60.0)
        if not future.done() or future.result() is None:
            result = {"goal_sent": True, "goal_accepted": False, "execution_completed": False, "action_result": "goal_send_timeout"}
            self._write(result)
            return result
        handle = future.result()
        accepted = bool(handle.accepted)
        self.accepted_at = time.monotonic()
        if not accepted:
            result = {"goal_sent": True, "goal_accepted": False, "execution_completed": False, "action_result": "rejected"}
            self.finished_at = time.monotonic()
            self._write(result)
            return result
        result_future = handle.get_result_async()
        next_progress = time.monotonic() + 30.0
        while not result_future.done():
            rclpy.spin_once(self, timeout_sec=0.05)
            if time.monotonic() >= next_progress:
                print(json.dumps({"stage5a_progress": True, "controller_messages": len(self.controller_rows), "joint_state_messages": len(self.joint_rows)}, separators=(",", ":")), flush=True)
                next_progress += 30.0
        self.finished_at = time.monotonic()
        wrapped = result_future.result()
        action_result = wrapped.result
        completed = int(action_result.error_code) == 0
        self.spin_for(1.0)
        formal_controller = [row for row in self.controller_rows if self.accepted_at <= row["capture_monotonic_s"] <= self.finished_at]
        result = {
            "goal_sent": True,
            "goal_accepted": accepted,
            "goal_uuid": "".join(f"{x:02x}" for x in handle.goal_id.uuid),
            "execution_completed": completed,
            "result_code": int(action_result.error_code),
            "result_error_string": str(action_result.error_string),
            "action_result": "successful" if completed else "aborted",
            "goal_point_count": len(self.goal["points"]),
            "trajectory_id": self.goal.get("goal_identity", {}).get("trajectory_id"),
            "source_sha256": self.goal.get("goal_identity", {}).get("source_sha256"),
            "formal_start_monotonic_s": self.accepted_at,
            "formal_end_monotonic_s": self.finished_at,
            "measured_execution_duration_s": float(self.finished_at - self.accepted_at),
            "controller_state_messages_received": len(self.controller_rows),
            "controller_state_messages_in_formal_execution": len(formal_controller),
            "joint_states_messages_received": len(self.joint_rows),
            "action_feedback_messages_received": len(self.feedback_rows),
            "tf_spray_tcp_messages_received": sum(row.get("child_frame") == "spray_tcp_link" for row in self.tf_rows),
            "tf_dynamic_rows_received": len(self.tf_rows),
            "tf_static_messages_received": len(self.tf_static_rows),
            "one_follow_joint_trajectory_goal": True,
            "second_action_goals": 0,
            "cancellations": 0,
            "preemptions": 0,
        }
        self._write(result)
        return result

    def _feedback(self, message) -> None:
        feedback = message.feedback
        self.feedback_rows.append({"message_index": len(self.feedback_rows), "capture_monotonic_s": time.monotonic(), "joint_names": list(feedback.joint_names), "desired": point(feedback.desired), "actual": point(feedback.actual), "error": point(feedback.error)})

    def _jsonl(self, name: str, rows: list[dict]) -> None:
        with (self.output / name).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")

    def _write(self, result: dict) -> None:
        self._jsonl("stage5a_controller_state_raw.jsonl", self.controller_rows)
        self._jsonl("stage5a_joint_states_raw.jsonl", self.joint_rows)
        self._jsonl("stage5a_action_feedback_raw.jsonl", self.feedback_rows)
        self._jsonl("stage5a_tf_raw.jsonl", self.tf_rows)
        self._jsonl("stage5a_tf_static_raw.jsonl", self.tf_static_rows)
        (self.output / "stage5a_action_execution.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (self.output / "stage5a_tf_capture_manifest.json").write_text(json.dumps({"schema_version": "stage5a-tf-capture-v1", "topics": ["/tf", "/tf_static"], "dynamic_tf_tree": True, "spray_tcp_child_frame": "spray_tcp_link", "tf_rows": len(self.tf_rows), "tf_static_rows": len(self.tf_static_rows)}, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial-tolerance", type=float, default=1.0e-9)
    args = parser.parse_args()
    output = args.output.resolve()
    goal = json.loads((output / "stage5a_follow_joint_trajectory_goal.json").read_text(encoding="utf-8"))
    rclpy.init()
    node = Stage5AClient(output, goal, initial_tolerance=args.initial_tolerance)
    executor = SingleThreadedExecutor()
    try:
        executor.add_node(node)
        result = node.run()
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 0 if result.get("goal_accepted") and result.get("execution_completed") else 2
    finally:
        executor.remove_node(node)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
