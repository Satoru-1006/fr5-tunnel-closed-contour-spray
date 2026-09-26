"""Stage 2.8S formal mock-controller telemetry client.

This client is deliberately limited to one FollowJointTrajectory action goal.
It does not import QueryTrajectoryState, create a query client, publish a
trajectory topic, or load any FAIRINO SDK/hardware library.  It extends the
clean Stage 2.7 action recorder only to retain TF evidence for the simulation
time series.
"""

from __future__ import annotations

import json
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


def transform_dict(value: TransformStamped) -> dict:
    return {
        "parent_frame": str(value.header.frame_id),
        "child_frame": str(value.child_frame_id),
        "translation": {
            "x": float(value.transform.translation.x),
            "y": float(value.transform.translation.y),
            "z": float(value.transform.translation.z),
        },
        "rotation": {
            "x": float(value.transform.rotation.x),
            "y": float(value.transform.rotation.y),
            "z": float(value.transform.rotation.z),
            "w": float(value.transform.rotation.w),
        },
    }


class Stage28Client(Stage27CleanClient):
    """Clean FJT client with selected TF telemetry."""

    def __init__(self, output: Path, goal_data: dict) -> None:
        super().__init__(output, goal_data)
        self.tf_rows: list[dict] = []
        self.tf_static_rows: list[dict] = []
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

    @staticmethod
    def _selected(message: TransformStamped) -> bool:
        return message.child_frame_id in {"base_link", "wrist3_link", "spray_tcp_link"}

    def _tf_callback(self, message: TFMessage) -> None:
        for transform in message.transforms:
            if not self._selected(transform):
                continue
            self.tf_rows.append({
                "message_index": len(self.tf_rows),
                "capture_monotonic_s": time.monotonic(),
                "header_stamp": {
                    "sec": int(transform.header.stamp.sec),
                    "nanosec": int(transform.header.stamp.nanosec),
                    "seconds": float(transform.header.stamp.sec) + float(transform.header.stamp.nanosec) * 1e-9,
                },
                **transform_dict(transform),
            })

    def _tf_static_callback(self, message: TFMessage) -> None:
        for transform in message.transforms:
            if not self._selected(transform):
                continue
            self.tf_static_rows.append({
                "message_index": len(self.tf_static_rows),
                "capture_monotonic_s": time.monotonic(),
                "header_stamp": {
                    "sec": int(transform.header.stamp.sec),
                    "nanosec": int(transform.header.stamp.nanosec),
                    "seconds": float(transform.header.stamp.sec) + float(transform.header.stamp.nanosec) * 1e-9,
                },
                **transform_dict(transform),
            })

    def _write_jsonl(self, name: str, rows: list[dict]) -> None:
        with (self.output / name).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    def _write_streams(self, result: dict) -> None:
        super()._write_streams(result)
        self._write_jsonl("stage28s_tf_raw.jsonl", self.tf_rows)
        self._write_jsonl("stage28s_tf_static_raw.jsonl", self.tf_static_rows)
        (self.output / "stage28s_tf_capture_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": "stage28s-tf-capture-v1",
                    "topics": ["/tf", "/tf_static"],
                    "selected_child_frames": ["base_link", "wrist3_link", "spray_tcp_link"],
                    "tf_rows": len(self.tf_rows),
                    "tf_static_rows": len(self.tf_static_rows),
                    "formal_run_only": True,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: stage28s_runtime_client.py OUTPUT_DIR")
    output = Path(sys.argv[1]).resolve()
    goal_path = output / "stage27r_clean_follow_joint_trajectory_goal.json"
    goal_data = json.loads(goal_path.read_text(encoding="utf-8"))
    rclpy.init()
    node = Stage28Client(output, goal_data)
    try:
        result = node.run()
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result.get("goal_accepted") and result.get("execution_completed") else 2
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
