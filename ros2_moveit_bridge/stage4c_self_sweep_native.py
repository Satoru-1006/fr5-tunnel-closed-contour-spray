#!/usr/bin/env python3
"""Independent MoveIt2 adaptive self-collision sweep for D48 C2.

MoveIt2/Bullet in this workspace does not expose a continuous swept
robot-link/self-link API.  This worker therefore reports the strongest
available independent adaptive-discrete result and explicitly leaves
continuous self-collision certification unavailable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

import rclpy
from moveit.core.collision_detection import CollisionRequest, CollisionResult
from moveit.core.robot_state import RobotState
from moveit.planning import MoveItPy

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage3_h7_native import GROUP_NAME, JOINT_NAMES, contact_pairs
from tools.stage4c_self_sweep import adaptive_interpolation_fractions


MAX_STEP_RAD = math.radians(0.5)


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def read_native(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < 2 or "t" not in rows[0]:
        raise RuntimeError(f"native_trajectory_missing:{path}")
    t = np.asarray([float(row["t"]) for row in rows], dtype=float)
    q = np.asarray([[float(row[f"j{i}_q"]) for i in range(1, 7)] for row in rows], dtype=float)
    if q.ndim != 2 or q.shape[1] != 6 or not np.isfinite(t).all() or not np.isfinite(q).all() or np.any(np.diff(t) <= 0.0):
        raise RuntimeError(f"native_trajectory_invalid:{path}")
    return t, q


def self_pairs(scene: Any, state: RobotState, q: np.ndarray) -> list[str]:
    state.set_joint_group_positions(GROUP_NAME, q.tolist())
    state.update()
    request = CollisionRequest()
    request.joint_model_group_name = GROUP_NAME
    request.contacts = True
    request.max_contacts = 4096
    request.max_contacts_per_pair = 64
    result = CollisionResult()
    scene.check_collision(request, result, state)
    pairs = contact_pairs(result)
    environment_tokens = ("fixture_surface", "world", "tunnel")
    filtered = [pair for pair in pairs if not any(token in pair for token in environment_tokens)]
    if result.collision and not pairs:
        return ["unknown_collision_source"]
    return sorted(set(filtered))


def process_case(moveit: MoveItPy, case_id: str, family: str, source: Path, output: Path) -> dict[str, Any]:
    t, q = read_native(source)
    monitor = moveit.get_planning_scene_monitor()
    state = RobotState(moveit.get_robot_model())
    contacts: list[dict[str, Any]] = []
    intervals = 0
    samples = 0
    pair_counts: Counter[str] = Counter()
    with monitor.read_only() as scene:
        for index, (time, positions) in enumerate(zip(t, q)):
            pairs = self_pairs(scene, state, positions)
            samples += 1
            if pairs:
                for pair in pairs:
                    pair_counts[pair] += 1
                contacts.append({"case_id": case_id, "family": family, "sample_kind": "execution_state", "trajectory_index": index, "time_s": float(time), "pairs": pairs})
        for index in range(len(q) - 1):
            intervals += 1
            fractions = adaptive_interpolation_fractions(q[index], q[index + 1], MAX_STEP_RAD)
            for fraction in fractions[1:]:
                sample = q[index] + float(fraction) * (q[index + 1] - q[index])
                pairs = self_pairs(scene, state, sample)
                samples += 1
                if pairs:
                    for pair in pairs:
                        pair_counts[pair] += 1
                    contacts.append({"case_id": case_id, "family": family, "sample_kind": "adaptive_discrete_interpolation", "trajectory_index": index, "next_trajectory_index": index + 1, "interpolation_alpha": float(fraction), "time_s": float(t[index] + fraction * (t[index + 1] - t[index])), "pairs": pairs})
    with (output / "self_sweep_contacts.jsonl").open("a", encoding="utf-8") as stream:
        for row in contacts:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    return {
        "case_id": case_id, "family": family, "state_count": int(len(q)), "swept_interval_count": intervals,
        "adaptive_sample_count": samples, "adaptive_self_contact_count": len(contacts),
        "self_contact_pairs": dict(pair_counts), "adaptive_self_sweep_status": "PASS" if not contacts else "BLOCKED",
        "continuous_self_collision_status": "NOT_AVAILABLE_NO_CONTINUOUS_SELF_BACKEND",
        "collision_method": "adaptive_discrete_interpolation", "max_step_rad": MAX_STEP_RAD,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args, _ = parser.parse_known_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with Path(args.manifest).resolve().open(encoding="utf-8-sig", newline="") as stream:
        manifest = list(csv.DictReader(stream))
    if not manifest:
        raise RuntimeError("empty_self_sweep_manifest")
    rclpy.init()
    moveit: MoveItPy | None = None
    try:
        moveit = MoveItPy(node_name="stage4c_self_sweep_native")
        model = moveit.get_robot_model()
        group = model.get_joint_model_group(GROUP_NAME)
        mapping = {"active_joint_names": list(group.active_joint_model_names), "group_link_names": list(group.link_model_names), "expected_joint_order": list(JOINT_NAMES), "joint_order_match": list(group.active_joint_model_names) == list(JOINT_NAMES)}
        if not mapping["joint_order_match"]:
            dump_json(output / "self_sweep_summary.json", {"status": "BLOCKED", "reason": "joint_order_mismatch", "mapping": mapping})
            return 2
        (output / "self_sweep_contacts.jsonl").write_text("", encoding="utf-8")
        cases = [process_case(moveit, str(item["case_id"]), str(item["family"]), Path(str(item["trajectory_csv"])).resolve(), output) for item in manifest]
        summary = {
            "schema_version": "d48-c2-adaptive-self-sweep-v1", "status": "MEASUREMENT_COMPLETE", "continuous_self_collision_status": "NOT_AVAILABLE_NO_CONTINUOUS_SELF_BACKEND", "continuous_backend_search": {"preferred": "Tesseract BulletCastBVHManager", "available": False, "reason": "no Tesseract collision package/backend installed in the available ROS/apt environment"},
            "verifier": "MoveIt2 PlanningScene adaptive discrete self sweep", "collision_method": "adaptive_discrete_interpolation", "geometry_source": "same MoveIt2 robot model/SRDF used by native C1 chain", "mapping": mapping, "case_count": len(cases), "adaptive_self_sweep_contact_cases": sum(item["adaptive_self_sweep_status"] != "PASS" for item in cases), "cases": cases,
        }
        dump_json(output / "self_sweep_summary.json", summary)
        # MoveItPy/Jazzy can double-destroy MoveItCpp during interpreter
        # teardown after a collision-only PlanningScene read.  All measurement
        # files are closed before this point, so terminate the one-shot worker
        # with the measured status and let the OS reclaim the already-finished
        # ROS objects.  This repairs the launcher -11 infrastructure defect
        # without skipping or altering a self-collision sample.
        os._exit(0)
    finally:
        if moveit is not None:
            moveit.shutdown()
            # Avoid a second MoveItCpp teardown from the Python wrapper after
            # the explicit shutdown.  The native C1 worker has shown that this
            # ordering is safe; keeping the reference alive here can otherwise
            # turn a completed measurement into a launcher -11 warning.
            moveit = None
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
