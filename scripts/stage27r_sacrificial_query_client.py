"""Stage 2.7R Run B: sacrificial live query_state perturbation experiment."""

from __future__ import annotations

import argparse
import json
import time
from bisect import bisect_right
from pathlib import Path

import rclpy
from builtin_interfaces.msg import Time as RosTime
from control_msgs.action import FollowJointTrajectory
from control_msgs.srv import QueryTrajectoryState
from control_msgs.msg import JointTrajectoryControllerState
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from trajectory_msgs.msg import JointTrajectoryPoint


def stamp_seconds(value) -> float:
    return float(value.sec) + float(value.nanosec) * 1e-9


def duration_seconds(value) -> float:
    return float(value.sec) + float(value.nanosec) * 1e-9


def point_dict(value) -> dict:
    return {
        "positions": [float(x) for x in value.positions],
        "velocities": [float(x) for x in value.velocities],
        "accelerations": [float(x) for x in value.accelerations],
        "time_from_start_s": duration_seconds(value.time_from_start),
    }


def ros_time(seconds: float) -> RosTime:
    msg = RosTime()
    msg.sec = int(seconds)
    msg.nanosec = int(round((seconds - int(seconds)) * 1e9))
    if msg.nanosec >= 1_000_000_000:
        msg.sec += 1
        msg.nanosec -= 1_000_000_000
    return msg


class SacrificialClient(Node):
    def __init__(self, output: Path, initial_q: list[float], mode: str) -> None:
        super().__init__(f"stage27r_sacrificial_{mode}")
        self.output = output
        self.initial_q = [float(x) for x in initial_q]
        self.mode = mode
        self.action_client = ActionClient(self, FollowJointTrajectory, "/fairino5_controller/follow_joint_trajectory")
        self.query_client = self.create_client(QueryTrajectoryState, "/fairino5_controller/query_state")
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=20, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.states: list[dict] = []
        self.query_call_time: float | None = None
        self.state_sub = self.create_subscription(JointTrajectoryControllerState, "/fairino5_controller/controller_state", self._state_callback, qos)

    def _state_callback(self, msg) -> None:
        self.states.append({
            "message_index": len(self.states),
            "capture_monotonic_s": time.monotonic(),
            "header_stamp_s": stamp_seconds(msg.header.stamp),
            "reference": point_dict(msg.reference),
            "feedback": point_dict(msg.feedback),
            "speed_scaling_factor": float(msg.speed_scaling_factor),
        })

    def spin_for(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=min(0.02, max(0.0, deadline - time.monotonic())))

    def fixture_goal(self) -> tuple[FollowJointTrajectory.Goal, list[float], list[float]]:
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ["j1", "j2", "j3", "j4", "j5", "j6"]
        duration = 2.0
        times = [0.1 * i for i in range(21)]
        final_q = list(self.initial_q)
        final_q[0] += 0.1
        for index, value in enumerate(times):
            point = JointTrajectoryPoint()
            alpha = value / duration
            point.positions = [float((1.0 - alpha) * self.initial_q[j] + alpha * final_q[j]) for j in range(6)]
            point.velocities = [0.0] * 6
            point.accelerations = [0.0] * 6
            point.time_from_start.sec = int(value)
            point.time_from_start.nanosec = int(round((value - int(value)) * 1e9))
            goal.trajectory.points.append(point)
        return goal, times, final_q

    def wait_until_reference_time(self, value: float, timeout_s: float = 2.0, *, after_monotonic: float | None = None) -> dict:
        end = time.monotonic() + timeout_s
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.02)
            candidates = [x for x in self.states if x["reference"]["time_from_start_s"] >= value and (after_monotonic is None or x["capture_monotonic_s"] >= after_monotonic)]
            if candidates:
                return candidates[-1]
        raise RuntimeError(f"did not observe controller reference time {value}")

    def query(self, start_abs: float, requested_rel: float) -> dict:
        request = QueryTrajectoryState.Request()
        absolute = start_abs + requested_rel
        request.time = ros_time(absolute)
        self.query_call_time = time.monotonic()
        future = self.query_client.call_async(request)
        end = time.monotonic() + 5.0
        while not future.done() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.01)
        if not future.done():
            return {"completed": False, "timeout": True, "request": {"sec": request.time.sec, "nanosec": request.time.nanosec, "absolute_ros_time_s": absolute, "requested_time_from_start_s": requested_rel}}
        response = future.result()
        return {
            "completed": True,
            "timeout": False,
            "request": {"sec": request.time.sec, "nanosec": request.time.nanosec, "absolute_ros_time_s": absolute, "requested_time_from_start_s": requested_rel},
            "response": {
                "success": bool(response.success),
                "message": str(response.message),
                "name": list(response.name),
                "position": [float(x) for x in response.position],
                "velocity": [float(x) for x in response.velocity],
                "acceleration": [float(x) for x in response.acceleration],
            },
        }

    def run(self) -> dict:
        if not self.action_client.wait_for_server(timeout_sec=30.0):
            raise RuntimeError("action server unavailable")
        if not self.query_client.wait_for_service(timeout_sec=30.0):
            raise RuntimeError("query_state service unavailable")
        self.spin_for(0.5)
        goal, times, final_q = self.fixture_goal()
        send = self.action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send, timeout_sec=10.0)
        if not send.done() or send.result() is None or not send.result().accepted:
            result = {"mode": self.mode, "goal_accepted": False, "action_result": "rejected"}
            self._write(result)
            return result
        handle = send.result()
        accepted = time.monotonic()
        current = self.wait_until_reference_time(0.10, after_monotonic=accepted)
        start_abs = current["header_stamp_s"] - current["reference"]["time_from_start_s"]
        current_rel = current["reference"]["time_from_start_s"]
        inferred_before = max(0, min(len(times) - 2, bisect_right(times, current_rel) - 1))
        record: dict = {
            "mode": self.mode,
            "fixture": {"segment_count": 20, "point_count": 21, "times_from_start_s": times, "initial_q": self.initial_q, "final_q": final_q},
            "controller_current_time": {"header_stamp_s": current["header_stamp_s"], "reference_time_from_start_s": current_rel, "derived_trajectory_start_absolute_ros_time_s": start_abs},
            "last_sample_idx_before": inferred_before,
            "query_state_shared_monotonic_state_effect": {"reproduced": False},
        }
        if self.mode == "baseline":
            self.spin_for(0.4)
            record["baseline_controller_desired_state"] = self.states[-1] if self.states else None
        else:
            requested_rel = 0.8 if self.mode == "future_query" else 2.5
            query_record = self.query(start_abs, requested_rel)
            record["query_absolute_time"] = query_record["request"]["absolute_ros_time_s"]
            record["derived_query_time_from_start"] = requested_rel
            record["query_request_sec"] = query_record["request"]["sec"]
            record["query_request_nanosec"] = query_record["request"]["nanosec"]
            record["query_response"] = query_record
            record["last_sample_idx_after"] = (len(times) - 1) if requested_rel >= times[-1] else max(0, min(len(times) - 2, bisect_right(times, requested_rel) - 1))
            self.spin_for(0.35)
            post = [x for x in self.states if x["capture_monotonic_s"] > (self.query_call_time or accepted)]
            affected = False
            first_affected = None
            if self.mode == "future_query":
                for row in post:
                    ref_time = row["reference"]["time_from_start_s"]
                    if ref_time < requested_rel - 0.1 and max(abs(row["reference"]["positions"][j] - final_q[j]) for j in range(6)) < 1e-7:
                        affected = True
                        first_affected = row
                        break
            else:
                for row in post:
                    ref_time = row["reference"]["time_from_start_s"]
                    if ref_time < times[-1] - 0.1 and max(abs(row["reference"]["positions"][j] - final_q[j]) for j in range(6)) < 1e-7:
                        affected = True
                        first_affected = row
                        break
            record["subsequent_earlier_realtime_sample"] = {"affected": affected, "first_observation": first_affected}
            record["query_state_shared_monotonic_state_effect"] = {"reproduced": bool(record["last_sample_idx_after"] > record["last_sample_idx_before"]), "evidence": "source last_sample_index mapping plus post-query native controller_state"}
        result_future = handle.get_result_async()
        while not result_future.done():
            rclpy.spin_once(self, timeout_sec=0.02)
        result = result_future.result().result
        record["goal_accepted"] = True
        record["action_result"] = {"error_code": int(result.error_code), "error_string": str(result.error_string)}
        record["controller_state_messages"] = len(self.states)
        self._write(record)
        return record

    def _write(self, result: dict) -> None:
        with (self.output / "controller_state_raw.jsonl").open("w", encoding="utf-8") as handle:
            for row in self.states:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        (self.output / "probe_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--initial-json", required=True, type=Path)
    parser.add_argument("--mode", choices=["baseline", "future_query", "extreme_future"], required=True)
    args = parser.parse_args()
    initial = json.loads(args.initial_json.read_text(encoding="utf-8"))["stage25r2_point0_position"]
    rclpy.init()
    node = SacrificialClient(args.output.resolve(), initial, args.mode)
    try:
        result = node.run()
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result.get("goal_accepted") else 2
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
