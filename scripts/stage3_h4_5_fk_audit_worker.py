"""MoveIt2 FK-only link-transform audit worker for H4.5."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> int:  # pragma: no cover - requires ROS 2 MoveItPy
    import numpy as np
    import rclpy
    from moveit.core.robot_state import RobotState
    from moveit.planning import MoveItPy

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args, _ros_args = parser.parse_known_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    rclpy.init(args=None)
    moveit = MoveItPy(node_name="stage3_h4_5_fk_audit")
    state = RobotState(moveit.get_robot_model())
    state.set_joint_group_positions("fairino5_v6_group", np.asarray(payload["joint_positions"], dtype=float))
    state.update()
    links = {}
    for name in ("forearm_link", "wrist1_link", "wrist2_link", "wrist3_link", "spray_tcp_link"):
        links[name] = state.get_global_link_transform(name).tolist()
    args.output.write_text(json.dumps({
        "schema_version": "stage3-h4-5-native-fk-audit-v1",
        "implementation": "MoveIt2 RobotState.update + get_global_link_transform",
        "frame": "base_link",
        "joint_names": ["j1", "j2", "j3", "j4", "j5", "j6"],
        "joint_positions": payload["joint_positions"],
        "target_index": payload.get("target_index"),
        "candidate_id": payload.get("candidate_id"),
        "placement_id": payload.get("placement_id"),
        "links": links,
    }, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    # MoveItCpp destruction is known to fault in this environment after all
    # requested FK data is written.  Exit only after the artifact is closed;
    # this preserves the numerical FK evidence and prevents a destructor
    # crash from invalidating the child lifecycle certificate.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
