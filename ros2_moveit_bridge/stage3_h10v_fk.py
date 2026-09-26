#!/usr/bin/env python3
"""Stage 3 H10-V native MoveIt2 FK and PlanningScene audit worker.

This worker is deliberately read-only and software-only.  It consumes a
materialized copy of one already-certified H10 family, evaluates native
RobotState FK for every sample, and rechecks the family with the same
PlanningScene/FCL adaptive discrete interpolation convention used upstream.
It does not import a controller, action client, or physical driver.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from moveit.core.robot_state import RobotState
from moveit.planning import MoveItPy

import stage3_h7_2_native as h72
import stage3_h7_native as h7


LINK_ORDER = [
    "base_link",
    "shoulder_link",
    "upperarm_link",
    "forearm_link",
    "wrist1_link",
    "wrist2_link",
    "wrist3_link",
    "spray_tcp_link",
]


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n")


def main() -> int:  # pragma: no cover - requires a ROS 2 MoveIt runtime
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--model-urdf", type=Path, required=True)
    args, _ros_args = parser.parse_known_args()

    result_path = args.output / "native_fk_result.json"
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        samples = payload["samples"]
        if not samples:
            raise RuntimeError("selected_family_has_no_samples")

        rclpy.init(args=None)
        moveit = MoveItPy(node_name="stage3_h10v_fk")
        model = moveit.get_robot_model()
        state = RobotState(model)
        fk_rows: list[dict[str, Any]] = []

        for sample in samples:
            q = np.asarray(sample["joint_positions"], dtype=float)
            if q.shape != (6,) or not np.all(np.isfinite(q)):
                raise RuntimeError(f"invalid_joint_state:{sample.get('sample_index')}")
            state.set_joint_group_positions(h72.GROUP_NAME, q.tolist())
            state.update()
            links: dict[str, list[list[float]]] = {}
            for link_name in LINK_ORDER:
                transform = np.asarray(state.get_global_link_transform(link_name), dtype=float)
                if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
                    raise RuntimeError(f"invalid_fk_transform:{link_name}:{sample.get('sample_index')}")
                links[link_name] = transform.tolist()
            fk_rows.append(
                {
                    "sample_index": int(sample["sample_index"]),
                    "trajectory_time": float(sample["trajectory_time"]),
                    "segment_order": int(sample["segment_order"]),
                    "segment_id": int(sample["segment_id"]),
                    "spray_state": str(sample["spray_state"]),
                    "joint_positions": q.tolist(),
                    "links": links,
                }
            )

        # Reuse the existing H7 native PlanningScene/FCL implementation.  It
        # checks every state and adaptively interpolates every consecutive pair;
        # no continuous-CCD or clearance claim is made here.
        h7.apply_fixture(moveit, args.fixture.resolve())
        t = np.asarray([float(item["trajectory_time"]) for item in samples], dtype=float)
        q = np.asarray([item["joint_positions"] for item in samples], dtype=float)
        collision_rows, collision_summary = h7.collision_validate(moveit, t, q)

        write_jsonl(args.output / "native_fk_states.jsonl", fk_rows)
        native_result = {
            "schema_version": "stage3-h10v-native-fk-result-v1",
            "status": "PASSED",
            "sample_count": len(fk_rows),
            "fk_executed": True,
            "planning_scene_executed": True,
            "collision_recheck_executed": True,
            "collision_method": "adaptive_discrete_interpolation",
            "ccd_status": "not_available",
            "clearance_m": None,
            "collision_summary": collision_summary,
            "robot_model_name": str(model.name),
            "group_name": h72.GROUP_NAME,
            "joint_names": list(h72.JOINT_NAMES),
            "tip_link": "spray_tcp_link",
            "link_order": list(LINK_ORDER),
            "frame": "base_link",
            "implementation": "MoveIt2 RobotState.set_joint_group_positions + update + get_global_link_transform",
            "robot_model_urdf": str(args.model_urdf.resolve()),
            "robot_model_urdf_sha256": sha256_file(args.model_urdf.resolve()),
            "fixture_mesh": str(args.fixture.resolve()),
            "fixture_mesh_sha256": sha256_file(args.fixture.resolve()),
            "PHYSICAL_ROBOT_CONNECTED": "NO",
            "PHYSICAL_DRIVER_LOADED": "NO",
            "PHYSICAL_FJT_GOALS_SENT": 0,
            "ROBOT_MOTION_STARTED": "NO",
        }
        dump_json(result_path, native_result)

        # MoveItPy teardown has produced environment-specific destructor faults
        # in older project workers.  All evidence is closed before the process
        # exits; this worker has no controller or hardware lifecycle to close.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    except Exception as exc:
        dump_json(
            result_path,
            {
                "schema_version": "stage3-h10v-native-fk-result-v1",
                "status": "BLOCKED",
                "first_blocker": f"native_exception:{type(exc).__name__}:{exc}",
                "fk_executed": False,
                "planning_scene_executed": False,
                "collision_recheck_executed": False,
                "collision_method": "adaptive_discrete_interpolation",
                "ccd_status": "not_available",
                "clearance_m": None,
                "PHYSICAL_ROBOT_CONNECTED": "NO",
                "PHYSICAL_DRIVER_LOADED": "NO",
                "PHYSICAL_FJT_GOALS_SENT": 0,
                "ROBOT_MOTION_STARTED": "NO",
            },
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
