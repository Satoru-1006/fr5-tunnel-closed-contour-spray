"""Query the read-only counter exported by the exact rosbag2 recorder process."""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--service", required=True)
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args(argv)
    try:
        import rclpy
        from std_srvs.srv import Trigger
    except ImportError as error:
        print(json.dumps({"passed": False, "first_blocker": f"query_import_failed:{error}"}))
        return 2
    rclpy.init(args=None)
    node = rclpy.create_node("stage28sr2_recorder_live_loss_query")
    result: dict[str, object]
    try:
        client = node.create_client(Trigger, args.service)
        if not client.wait_for_service(timeout_sec=args.timeout):
            result = {"passed": False, "first_blocker": "recorder_live_loss_service_timeout"}
        else:
            future = client.call_async(Trigger.Request())
            rclpy.spin_until_future_complete(node, future, timeout_sec=args.timeout)
            if not future.done():
                result = {"passed": False, "first_blocker": "recorder_live_loss_query_timeout"}
            elif future.exception() is not None:
                result = {"passed": False, "first_blocker": f"recorder_live_loss_query_exception:{future.exception()}"}
            else:
                response = future.result()
                try:
                    payload = json.loads(response.message)
                except (TypeError, json.JSONDecodeError) as error:
                    result = {"passed": False, "first_blocker": f"recorder_live_loss_malformed:{error}"}
                else:
                    result = {"passed": bool(response.success), "service_success": bool(response.success), "payload": payload}
                    if not response.success:
                        result["first_blocker"] = "recorder_live_loss_service_reported_failure"
    finally:
        node.destroy_node()
        rclpy.shutdown()
    print("STAGE28SR2_JSON_START")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    print("STAGE28SR2_JSON_END")
    return 0 if result.get("passed") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
