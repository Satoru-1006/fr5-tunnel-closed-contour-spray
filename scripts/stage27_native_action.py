"""ROS 2 Jazzy native FJT client and controller/query_state recorder."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from builtin_interfaces.msg import Time as RosTime
from control_msgs.action import FollowJointTrajectory
from control_msgs.srv import QueryTrajectoryState
from control_msgs.msg import JointTrajectoryControllerState
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.controller_interpolation import evaluate_coefficients, segment_coefficients  # noqa: E402

JOINTS = ["j1", "j2", "j3", "j4", "j5", "j6"]


def read_goal(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def duration_to_ros(seconds: float) -> RosTime:
    sec = int(np.floor(seconds))
    nanosec = int(round((seconds - sec) * 1e9))
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000
    msg = RosTime()
    msg.sec = sec
    msg.nanosec = nanosec
    return msg


class FormalClient(Node):
    def __init__(self, output: Path, goal_data: dict, schedule: list[float]) -> None:
        super().__init__("stage27_formal_native_action_client")
        self.output = output
        self.goal_data = goal_data
        self.schedule = schedule
        self.action_client = ActionClient(self, FollowJointTrajectory, "/fairino5_controller/follow_joint_trajectory")
        self.query_client = self.create_client(QueryTrajectoryState, "/fairino5_controller/query_state")
        # JTC 4.40.1 publishes controller_state with reliable + transient-local QoS.
        # A default volatile subscription is incompatible and silently receives zero samples.
        state_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.state_sub = self.create_subscription(
            JointTrajectoryControllerState,
            "/fairino5_controller/controller_state",
            self._state_callback,
            state_qos,
        )
        self.controller_state_rows = []
        self.first_state = None
        self.last_state = None

    def _state_callback(self, msg) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        try:
            # Jazzy control_msgs uses reference/feedback (not desired/actual) fields.
            desired = msg.reference
            actual = msg.feedback
            error = msg.error
            row = {
                "wall_time_s": now,
                "header_stamp_s": float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9,
                "joint_names": list(msg.joint_names),
                "desired_time_from_start_s": float(desired.time_from_start.sec)
                + float(desired.time_from_start.nanosec) * 1e-9,
                "actual_time_from_start_s": float(actual.time_from_start.sec)
                + float(actual.time_from_start.nanosec) * 1e-9,
                "desired_position": list(desired.positions),
                "desired_velocity": list(desired.velocities),
                "desired_acceleration": list(desired.accelerations),
                "actual_position": list(actual.positions),
                "actual_velocity": list(actual.velocities),
                "actual_acceleration": list(actual.accelerations),
                "error_position": list(error.positions),
                "error_velocity": list(getattr(error, "velocities", [])),
                "error_acceleration": list(getattr(error, "accelerations", [])),
            }
            self.controller_state_rows.append(row)
            self.last_state = row
            if self.first_state is None:
                self.first_state = row
        except Exception:
            pass

    def spin_for(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=min(0.1, max(0.0, end - time.monotonic())))

    def wait_ready(self) -> None:
        if not self.action_client.wait_for_server(timeout_sec=30.0):
            raise RuntimeError("FollowJointTrajectory action server unavailable")
        end = time.monotonic() + 30.0
        while not self.query_client.wait_for_service(timeout_sec=0.5):
            if time.monotonic() > end:
                raise RuntimeError("query_state service unavailable")
        self.spin_for(2.0)

    def make_goal(self) -> FollowJointTrajectory.Goal:
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(self.goal_data["joint_names"])
        for source in self.goal_data["points"]:
            point = __import__("trajectory_msgs.msg", fromlist=["JointTrajectoryPoint"]).JointTrajectoryPoint()
            point.positions = list(source["positions"])
            point.velocities = list(source["velocities"])
            point.accelerations = list(source["accelerations"])
            point.time_from_start.sec = int(source["time_from_start"]["sec"])
            point.time_from_start.nanosec = int(source["time_from_start"]["nanosec"])
            goal.trajectory.points.append(point)
        return goal

    def query_one(
        self,
        requested_seconds: float,
        actual_seconds: float,
        absolute_seconds: float,
        coeffs: list[np.ndarray],
        times: np.ndarray,
    ) -> dict:
        request = QueryTrajectoryState.Request()
        request.time = duration_to_ros(absolute_seconds)
        future = self.query_client.call_async(request)
        end = time.monotonic() + 3.0
        while not future.done() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.01)
        if not future.done():
            return {"query_time": requested_seconds, "native_sample_time": actual_seconds, "query_absolute_time": absolute_seconds, "service_ok": False, "error": "query_state timeout"}
        response = future.result()
        actual_position = response.position
        actual_velocity = response.velocity
        actual_acceleration = response.acceleration
        if not response.success or min(len(actual_position), len(actual_velocity), len(actual_acceleration)) < 6:
            return {"query_time": requested_seconds, "native_sample_time": actual_seconds, "query_absolute_time": absolute_seconds, "service_ok": False, "error": str(response.message), "native": {"position": list(actual_position), "velocity": list(actual_velocity), "acceleration": list(actual_acceleration), "joint_names": list(response.name), "success": bool(response.success), "message": str(response.message)}}
        seconds = actual_seconds
        i = int(np.searchsorted(times, seconds, side="right") - 1)
        i = max(0, min(i, len(coeffs) - 1))
        local = float(seconds - times[i])
        reconstructed = evaluate_coefficients(coeffs[i], local)
        native = {"position": list(actual_position), "velocity": list(actual_velocity), "acceleration": list(actual_acceleration), "joint_names": list(response.name), "success": bool(response.success), "message": str(response.message), "error_code": 0 if response.success else 1}
        errors = {quantity: [abs(float(native[quantity][j]) - float(reconstructed[quantity][j])) for j in range(6)] for quantity in ("position", "velocity", "acceleration")}
        return {"query_time": requested_seconds, "native_sample_time": actual_seconds, "query_absolute_time": absolute_seconds, "service_ok": True, "native": native, "reconstructed": {k: list(map(float, reconstructed[k])) for k in ("position", "velocity", "acceleration")}, "absolute_error": errors, "segment": i, "local_time_s": local}

    def run(self) -> dict:
        self.wait_ready()
        goal = self.make_goal()
        state_before = self.last_state or self.first_state
        # Formal execution is not allowed to start without a live controller-state
        # observation matching the frozen Stage 2.5R2 first point.
        expected_initial = np.asarray(self.goal_data["points"][0]["positions"], dtype=float)
        observed_initial = None if state_before is None else state_before.get("actual_position")
        initial_compatible = (
            observed_initial is not None
            and len(observed_initial) == len(expected_initial)
            and max(abs(float(a) - float(b)) for a, b in zip(observed_initial, expected_initial)) <= 1e-9
        )
        if not initial_compatible:
            execution = {
                "goal_sent": False,
                "goal_accepted": False,
                "execution_started": False,
                "execution_completed": False,
                "action_result": "blocked_pre_goal_initial_state_incompatible",
                "primary_blocker": "blocked_native_controller_initial_state_incompatible",
            }
            self._write_evidence(execution, state_before)
            return execution
        times = np.asarray([float(x["time_from_start"]["seconds"]) for x in self.goal_data["points"]], dtype=float)
        coeff_path = self.output / "stage27_spline_coefficients.json"
        if coeff_path.exists():
            coeff_payload = json.loads(coeff_path.read_text(encoding="utf-8"))
            coeffs = [
                np.asarray(item["coefficients_c0_to_c5_by_joint"], dtype=float)
                for item in coeff_payload["coefficients"]
            ]
        else:
            coeffs = [segment_coefficients(np.asarray(self.goal_data["points"][i]["positions"]), np.asarray(self.goal_data["points"][i + 1]["positions"]), np.asarray(self.goal_data["points"][i]["velocities"]), np.asarray(self.goal_data["points"][i + 1]["velocities"]), np.asarray(self.goal_data["points"][i]["accelerations"]), np.asarray(self.goal_data["points"][i + 1]["accelerations"]), float(times[i + 1] - times[i]), "quintic") for i in range(len(times) - 1)]
        duration = float(times[-1])
        goal_sent_wall = self.get_clock().now().nanoseconds * 1e-9
        send_future = self.action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=60.0)
        if not send_future.done():
            raise RuntimeError("FJT goal send timed out")
        handle = send_future.result()
        accepted = bool(handle and handle.accepted)
        if not accepted:
            result = {"goal_sent": True, "goal_accepted": False, "execution_completed": False, "action_result": "rejected"}
            self._write_evidence(result, state_before)
            return result
        uuid = "".join(f"{x:02x}" for x in handle.goal_id.uuid)
        accepted_state_stamp = None
        accepted_clock = self.get_clock().now().nanoseconds * 1e-9
        end = time.monotonic() + 0.03
        while time.monotonic() < end:
            candidates = [
                row
                for row in self.controller_state_rows
                if row["header_stamp_s"] >= accepted_clock - 0.02
                and row["desired_time_from_start_s"] <= 0.02
            ]
            if candidates:
                accepted_state_stamp = candidates[0]["header_stamp_s"] - candidates[0]["desired_time_from_start_s"]
                break
            rclpy.spin_once(self, timeout_sec=0.01)
        if accepted_state_stamp is None:
            # The goal acceptance clock is the only safe fallback; the scheduler
            # below will clamp the first query to the live controller time.
            accepted_state_stamp = accepted_clock
        trajectory_start_abs = float(accepted_state_stamp)
        result_future = handle.get_result_async()
        # query_state() calls the same Trajectory::sample() object used by the
        # controller.  In 4.40.1 that sampler keeps a monotone last_sample_idx_,
        # so querying future timestamps in a burst would alter live execution.
        # Query in real controller-time order, only one small margin ahead, and
        # reconstruct at the actual requested native timestamp.
        # One native query per spline segment midpoint keeps the request rate at
        # the 100 Hz JTC update rate.  The controller_state stream independently
        # captures the 100 Hz knot/reference samples.
        online_schedule = [float((times[i] + times[i + 1]) * 0.5) for i in range(len(times) - 1)]
        query_dispatch_lead = 0.010
        previous_actual = -1.0
        queries = []
        skipped_schedule = 0
        query_path = self.output / "stage27_native_query_state_samples.jsonl"
        with query_path.open("w", encoding="utf-8") as handle_out:
            for index, requested in enumerate(online_schedule):
                requested = float(requested)
                actual = requested
                actual = max(previous_actual, actual)
                previous_actual = actual
                target_clock = trajectory_start_abs + actual
                dispatch_clock = target_clock - query_dispatch_lead
                while True:
                    now_clock = self.get_clock().now().nanoseconds * 1e-9
                    remaining = dispatch_clock - now_clock
                    if remaining <= 0.0:
                        break
                    rclpy.spin_once(self, timeout_sec=min(0.02, remaining))
                live_relative = max(0.0, self.get_clock().now().nanoseconds * 1e-9 - trajectory_start_abs)
                if actual < live_relative - 0.001:
                    skipped_schedule += 1
                    continue
                record = self.query_one(requested, actual, trajectory_start_abs + actual, coeffs, times)
                handle_out.write(json.dumps(record, separators=(",", ":")) + "\n")
                queries.append(record)
                if index % 2000 == 0:
                    self.get_logger().info(f"query_state samples {index + 1}/{len(self.schedule)}")
        while not result_future.done():
            rclpy.spin_once(self, timeout_sec=0.1)
        wrapped = result_future.result()
        native_result = wrapped.result
        completed = int(native_result.error_code) == 0
        query_ok = [x for x in queries if x.get("service_ok")]
        max_errors = {quantity: max((max(x["absolute_error"][quantity]) for x in query_ok), default=None) for quantity in ("position", "velocity", "acceleration")}
        # These are strict floating-point crosscheck limits for the native C++
        # polynomial evaluator versus the Python audit evaluator.  They are
        # orders of magnitude below trajectory/process tolerances and cover
        # double-rounding in acceleration evaluation at large coefficients.
        reconstruction_tolerances = {"position": 1e-8, "velocity": 1e-7, "acceleration": 2e-6}
        state_checks = []
        for row in self.controller_state_rows:
            relative = float(row["header_stamp_s"] - trajectory_start_abs)
            if relative < -0.02 or relative > duration + 0.02:
                continue
            i = int(np.searchsorted(times, max(0.0, min(duration, relative)), side="right") - 1)
            i = max(0, min(i, len(coeffs) - 1))
            local = float(max(0.0, min(duration, relative)) - times[i])
            expected = evaluate_coefficients(coeffs[i], local)
            for quantity, field in (("position", "desired_position"), ("velocity", "desired_velocity"), ("acceleration", "desired_acceleration")):
                values = row.get(field) or []
                if len(values) == 6:
                    state_checks.append((quantity, max(abs(float(values[j]) - float(expected[quantity][j])) for j in range(6)), relative))
        state_max_errors = {quantity: max((x[1] for x in state_checks if x[0] == quantity), default=None) for quantity in ("position", "velocity", "acceleration")}
        worst = {}
        for quantity in max_errors:
            if max_errors[quantity] is None:
                worst[quantity] = None
            else:
                item = max(query_ok, key=lambda x: max(x["absolute_error"][quantity]))
                worst[quantity] = {"query_time": item["query_time"], "segment": item.get("segment"), "error": max_errors[quantity]}
        execution = {
            "goal_sent": True,
            "goal_send_timestamp": goal_sent_wall,
            "goal_accepted": accepted,
            "goal_uuid": uuid,
            "trajectory_start_absolute_time_s": trajectory_start_abs,
            "execution_started": True,
            "execution_completed": completed,
            "result_code": int(native_result.error_code),
            "result_error_code": int(native_result.error_code),
            "result_error_string": str(native_result.error_string),
            "action_result": "successful" if completed else "aborted",
            "controller_state_messages_received": len(self.controller_state_rows),
            "desired_state_captured": bool(self.controller_state_rows and any(x["desired_position"] for x in self.controller_state_rows)),
            "actual_state_captured": bool(self.controller_state_rows and any(x["actual_position"] for x in self.controller_state_rows)),
            "query_state_samples_requested": len(online_schedule),
            "query_state_samples_attempted": len(queries),
            "query_state_schedule_skipped": skipped_schedule,
            "query_state_schedule_mode": "native_query_state_segment_midpoints; controller_state_knots_captured_separately",
            "query_state_schedule_online_count": len(online_schedule),
            "query_state_schedule_total_requested": len(self.schedule),
            "query_state_samples_received": len(query_ok),
            "native_query_state_vs_reconstruction": {
                "position_match": max_errors["position"] is not None and max_errors["position"] <= reconstruction_tolerances["position"],
                "velocity_match": max_errors["velocity"] is not None and max_errors["velocity"] <= reconstruction_tolerances["velocity"],
                "acceleration_match": max_errors["acceleration"] is not None and max_errors["acceleration"] <= reconstruction_tolerances["acceleration"],
                "reconstruction_matches_native_runtime": all(max_errors[x] is not None and max_errors[x] <= reconstruction_tolerances[x] for x in ("position", "velocity", "acceleration")),
                "max_abs_position_difference": max_errors["position"],
                "max_abs_velocity_difference": max_errors["velocity"],
                "max_abs_acceleration_difference": max_errors["acceleration"],
                "tolerances": reconstruction_tolerances,
                "worst": worst,
            },
            "controller_state_vs_reconstruction": {
                "samples_compared": len(state_checks) // 3,
                "max_abs_position_difference": state_max_errors["position"],
                "max_abs_velocity_difference": state_max_errors["velocity"],
                "max_abs_acceleration_difference": state_max_errors["acceleration"],
                "position_match": state_max_errors["position"] is not None and state_max_errors["position"] <= reconstruction_tolerances["position"],
                "velocity_match": state_max_errors["velocity"] is not None and state_max_errors["velocity"] <= reconstruction_tolerances["velocity"],
                "acceleration_match": state_max_errors["acceleration"] is not None and state_max_errors["acceleration"] <= reconstruction_tolerances["acceleration"],
            },
        }
        self._write_evidence(execution, state_before)
        return execution

    def _write_evidence(self, execution: dict, state_before: dict | None) -> None:
        fieldnames = ["wall_time_s", "header_stamp_s", "joint_names", "desired_time_from_start_s", "actual_time_from_start_s", "desired_position", "desired_velocity", "desired_acceleration", "actual_position", "actual_velocity", "actual_acceleration", "error_position", "error_velocity", "error_acceleration"]
        with (self.output / "stage27_controller_state.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.controller_state_rows:
                writer.writerow({key: json.dumps(row[key], separators=(",", ":")) if isinstance(row[key], list) else row[key] for key in fieldnames})
        (self.output / "stage27_controller_state_metadata.json").write_text(json.dumps({"topic": "/fairino5_controller/controller_state", "message_type": "control_msgs/msg/JointTrajectoryControllerState", "messages": len(self.controller_state_rows), "desired_state_captured": bool(self.controller_state_rows and any(x["desired_position"] for x in self.controller_state_rows)), "actual_state_captured": bool(self.controller_state_rows and any(x["actual_position"] for x in self.controller_state_rows))}, indent=2) + "\n", encoding="utf-8")
        pre = {"controller_position": (state_before or {}).get("actual_position"), "controller_velocity": (state_before or {}).get("actual_velocity"), "stage25r2_point0": self.goal_data["points"][0]["positions"], "per_joint_difference": None, "max_abs_difference": None, "first_point_time_from_start": self.goal_data["points"][0]["time_from_start"]["seconds"]}
        if pre["controller_position"] is not None:
            diffs = [abs(float(a) - float(b)) for a, b in zip(pre["controller_position"], pre["stage25r2_point0"])]
            pre["per_joint_difference"] = diffs
            pre["max_abs_difference"] = max(diffs)
            pre["compatible"] = max(diffs) <= 1e-9
        else:
            pre["compatible"] = False
        (self.output / "stage27_pre_goal_initial_state.json").write_text(json.dumps(pre, indent=2) + "\n", encoding="utf-8")
        (self.output / "stage27_action_execution.json").write_text(json.dumps({"native_execution": execution}, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    goal_data = read_goal(output / "stage27_follow_joint_trajectory_goal.json")
    schedule = json.loads((output / "stage27_query_schedule.json").read_text(encoding="utf-8"))["times"]
    rclpy.init()
    node = FormalClient(output, goal_data, schedule)
    try:
        result = node.run()
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result.get("goal_accepted") and result.get("execution_completed") else 2
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
