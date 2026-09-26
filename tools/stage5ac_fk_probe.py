"""Read-only FK probe for the Stage5AC shadow task-boundary diagnosis."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


GROUP = "fairino5_v6_group"
EE = "spray_tcp_link"


def main() -> int:  # pragma: no cover - ROS 2 runtime
    import rclpy
    from moveit.planning import MoveItPy
    from rclpy.node import Node

    rclpy.init()
    node = Node("stage5ac_fk_probe")
    node.declare_parameter("poses_csv", "")
    node.declare_parameter("historical_seed_csv", "")
    node.declare_parameter("output_json", "")
    poses_path = Path(str(node.get_parameter("poses_csv").value)).resolve()
    history_path = Path(str(node.get_parameter("historical_seed_csv").value)).resolve()
    output_path = Path(str(node.get_parameter("output_json").value)).resolve()
    poses = list(csv.DictReader(poses_path.open(encoding="utf-8", newline="")))
    history = list(csv.DictReader(history_path.open(encoding="utf-8", newline="")))
    moveit = MoveItPy(node_name="stage5ac_fk_probe_moveit")
    model = moveit.get_robot_model()
    from moveit.core.robot_state import RobotState

    state = RobotState(model)
    rows = []
    for i in range(181):
        q = np.asarray([float(history[i][f"j{j}_q"]) for j in range(1, 7)], dtype=float)
        state.set_joint_group_positions(GROUP, q.tolist())
        state.update()
        transform = state.get_global_link_transform(EE)
        matrix = np.asarray(transform.matrix() if hasattr(transform, "matrix") else transform, dtype=float)
        target = np.asarray([float(poses[i][k]) for k in ("x", "y", "z")], dtype=float)
        actual = matrix[:3, 3]
        rows.append({"waypoint": i, "target_position_m": target.tolist(), "fk_position_m": actual.tolist(), "position_delta_fk_minus_target_m": (actual - target).tolist(), "position_error_m": float(np.linalg.norm(actual - target))})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"schema_version": "stage5ac-fk-probe-v1", "rows": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "selected": [rows[i] for i in range(84, 96)]}, ensure_ascii=False), flush=True)
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
