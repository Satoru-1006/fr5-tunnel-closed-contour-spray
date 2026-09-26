"""One-participant persistent ROS graph witness for Stage 2.8S-R2."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage28sr2_persistent_graph import SAMPLE_INTERVAL_SEC  # noqa: E402


def _gid(value: Any) -> str | None:
    try:
        raw = bytes(value)
    except (TypeError, ValueError):
        raw = b""
    return raw.hex() if raw and any(raw) else None


def _policy(value: Any) -> tuple[str | None, bool]:
    name = getattr(value, "name", None)
    if isinstance(name, str) and name.upper() not in {"UNKNOWN", "SYSTEM_DEFAULT"}:
        return name.lower(), True
    return None, False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--topics-json", required=True)
    parser.add_argument("--interval", type=float, default=SAMPLE_INTERVAL_SEC)
    parser.add_argument("--seed-topic-types", action="store_true")
    args = parser.parse_args()
    topics = tuple(json.loads(args.topics_json))
    if args.interval != SAMPLE_INTERVAL_SEC:
        raise SystemExit(f"interval is frozen at {SAMPLE_INTERVAL_SEC}")

    import rclpy

    rclpy.init(args=None)
    node = rclpy.create_node("stage28sr2_persistent_graph_observer")
    type_seed_subscriptions = []
    if args.seed_topic_types:
        from control_msgs.msg import JointTrajectoryControllerState
        from sensor_msgs.msg import JointState
        from tf2_msgs.msg import TFMessage
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

        def qos(durability: Any) -> Any:
            return QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=durability,
            )

        volatile = DurabilityPolicy.VOLATILE
        transient = DurabilityPolicy.TRANSIENT_LOCAL
        seeds = {
            "/joint_states": (JointState, qos(volatile)),
            "/fairino5_controller/controller_state": (JointTrajectoryControllerState, qos(volatile)),
            "/tf": (TFMessage, qos(volatile)),
            "/tf_static": (TFMessage, qos(transient)),
        }
        for topic, (message_type, profile) in seeds.items():
            if topic in topics:
                type_seed_subscriptions.append(node.create_subscription(message_type, topic, lambda _message: None, profile))
    running = True

    def stop(_signum: int, _frame: Any) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    args.raw.parent.mkdir(parents=True, exist_ok=True)
    args.ready.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    args.ready.write_text(json.dumps({
        "pid": os.getpid(),
        "node_fqn": node.get_fully_qualified_name(),
        "participant_creation_count": 1,
        "type_seed_subscription_count": len(type_seed_subscriptions),
        "type_seed_publisher_count": 0,
        "timestamp_monotonic": started,
        "sample_interval_sec": SAMPLE_INTERVAL_SEC,
    }, sort_keys=True) + "\n", encoding="utf-8")
    with args.raw.open("a", encoding="utf-8", buffering=1) as stream:
        index = 0
        while running and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)
            endpoints: list[dict[str, Any]] = []
            for topic in topics:
                for endpoint_type, getter in (
                    ("publisher", node.get_publishers_info_by_topic),
                    ("subscription", node.get_subscriptions_info_by_topic),
                ):
                    try:
                        infos = getter(topic, no_mangle=False)
                    except TypeError:
                        infos = getter(topic)
                    for info in infos:
                        reliability, reliability_observable = _policy(info.qos_profile.reliability)
                        durability, durability_observable = _policy(info.qos_profile.durability)
                        endpoints.append({
                            "timestamp_monotonic": time.monotonic(),
                            "topic_name": topic,
                            "endpoint_type": endpoint_type,
                            "node_name": info.node_name,
                            "node_namespace": info.node_namespace,
                            "gid": _gid(info.endpoint_gid),
                            "reliability": reliability,
                            "durability": durability,
                            "runtime_observable": {
                                "gid": _gid(info.endpoint_gid) is not None,
                                "reliability": reliability_observable,
                                "durability": durability_observable,
                                "history": False,
                                "depth": False,
                            },
                        })
            stream.write(json.dumps({
                "schema_version": "stage28sr2-persistent-graph-sample-v1",
                "sample_index": index,
                "timestamp_monotonic": time.monotonic(),
                "observer_pid": os.getpid(),
                "observer_node_fqn": node.get_fully_qualified_name(),
                "participant_creation_count": 1,
                "endpoints": endpoints,
            }, sort_keys=True) + "\n")
            index += 1
            time.sleep(SAMPLE_INTERVAL_SEC)
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
