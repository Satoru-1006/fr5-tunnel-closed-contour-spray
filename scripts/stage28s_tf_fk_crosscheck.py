"""Compare recorded dynamic TF with native MoveIt FK link-to-link transforms.

This verifier intentionally uses ROS header stamps as the physical-state key.
Recorder receive time is retained only as diagnostic metadata and is never used
to pair a TF sample with a joint state.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import rclpy
from moveit.core.robot_state import RobotState
from moveit.planning import MoveItPy
from rclpy.node import Node


def quat_to_matrix(q: dict[str, float]) -> np.ndarray:
    x, y, z, w = (float(q[k]) for k in ("x", "y", "z", "w"))
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def matrix_from_raw(value) -> np.ndarray:
    return np.asarray(value.matrix() if hasattr(value, "matrix") else value, dtype=float)


class CrosscheckNode(Node):
    def __init__(self) -> None:
        super().__init__("stage28s_tf_fk_crosscheck")
        self.declare_parameter("joint_states_jsonl", "")
        self.declare_parameter("tf_jsonl", "")
        self.declare_parameter("tf_static_jsonl", "")
        self.declare_parameter("output_json", "")

    def run(self) -> dict:
        joint_rows = [json.loads(line) for line in Path(str(self.get_parameter("joint_states_jsonl").value)).read_text(encoding="utf-8").splitlines() if line.strip()]
        tf_rows = [json.loads(line) for line in Path(str(self.get_parameter("tf_jsonl").value)).read_text(encoding="utf-8").splitlines() if line.strip() and json.loads(line).get("child_frame") == "wrist3_link"]
        joint_by_stamp = {
            (int(row["header_stamp"]["sec"]), int(row["header_stamp"]["nanosec"])): row
            for row in joint_rows
        }
        static_rows = [json.loads(line) for line in Path(str(self.get_parameter("tf_static_jsonl").value)).read_text(encoding="utf-8").splitlines() if line.strip()]
        static_tcp = next((row for row in static_rows if row.get("child_frame") == "spray_tcp_link" and row.get("parent_frame") == "wrist3_link"), None)

        moveit = MoveItPy(node_name="stage28s_tf_fk_crosscheck_moveit")
        state = RobotState(moveit.get_robot_model())
        errors_mm: list[float] = []
        angles_deg: list[float] = []
        unmatched = []
        for tf in tf_rows:
            key = (int(tf["header_stamp"]["sec"]), int(tf["header_stamp"]["nanosec"]))
            recorded_state = joint_by_stamp.get(key)
            if recorded_state is None:
                unmatched.append(tf)
                continue
            names = list(recorded_state.get("name", recorded_state.get("names", [])))
            positions = list(recorded_state.get("position", recorded_state.get("positions", [])))
            by_name = {str(name): float(value) for name, value in zip(names, positions)}
            if any(joint not in by_name for joint in ("j1", "j2", "j3", "j4", "j5", "j6")):
                unmatched.append(tf)
                continue
            q = np.asarray([by_name[joint] for joint in ("j1", "j2", "j3", "j4", "j5", "j6")], dtype=float)
            state.set_joint_group_positions("fairino5_v6_group", q)
            state.update()
            wrist2 = matrix_from_raw(state.get_global_link_transform("wrist2_link"))
            wrist3 = matrix_from_raw(state.get_global_link_transform("wrist3_link"))
            relative = np.linalg.inv(wrist2) @ wrist3
            tf_translation = np.asarray([float(tf["translation"][key]) for key in ("x", "y", "z")], dtype=float)
            errors_mm.append(float(np.linalg.norm(relative[:3, 3] - tf_translation) * 1000.0))
            tf_rotation = quat_to_matrix(tf["rotation"])
            rotation_delta = relative[:3, :3] @ tf_rotation.T
            cosine = float(np.clip((np.trace(rotation_delta) - 1.0) * 0.5, -1.0, 1.0))
            angles_deg.append(math.degrees(math.acos(cosine)))

        static_translation_error = None
        static_rotation_error_deg = None
        if static_tcp is not None:
            static_translation_error = float(np.linalg.norm(np.asarray([float(static_tcp["translation"][key]) for key in ("x", "y", "z")]) - np.asarray([0.0, 0.0, 0.15]))) * 1000.0
            static_rotation_error_deg = 0.0 if np.allclose(quat_to_matrix(static_tcp["rotation"]), np.eye(3), atol=1e-12) else None

        result = {
            "schema_version": "stage28s-tf-fk-crosscheck-v2-strict-header-stamp",
            "passed": bool(tf_rows and not unmatched and errors_mm and max(errors_mm) <= 1e-6 and angles_deg and max(angles_deg) <= 1e-6 and static_translation_error is not None and static_translation_error <= 1e-6 and static_rotation_error_deg == 0.0),
            "source": "recorded /tf wrist2_link->wrist3_link + /joint_states exact header_stamp + /tf_static",
            "physical_state_pairing": "exact ROS header stamp only",
            "capture_monotonic_s_used_for_pairing": False,
            "native_fk_backend": "MoveItPy RobotState::getGlobalLinkTransform for wrist2_link and wrist3_link",
            "samples_compared": len(errors_mm),
            "tf_rows_available": len(tf_rows),
            "exact_stamp_match_count": len(errors_mm),
            "unmatched_count": len(unmatched),
            "max_translation_error_mm": max(errors_mm) if errors_mm else None,
            "max_rotation_error_deg": max(angles_deg) if angles_deg else None,
            "dynamic_tf_tolerance_mm": 1e-6,
            "dynamic_tf_tolerance_deg": 1e-6,
            "static_spray_tcp_translation_error_mm": static_translation_error,
            "static_spray_tcp_rotation_error_deg": static_rotation_error_deg,
            "static_tcp_row_available": static_tcp is not None,
        }
        output = Path(str(self.get_parameter("output_json").value))
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return result


def main() -> int:
    rclpy.init()
    node = CrosscheckNode()
    try:
        result = node.run()
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result["passed"] else 2
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
