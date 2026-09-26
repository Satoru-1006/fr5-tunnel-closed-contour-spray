"""Stage 5A-R exact-state TF/FK witness.

This is deliberately separate from the original Stage 5A validator.  The
original validator paired TF batches with the latest causally prior callback
arrival.  This witness uses the JointState header stamp as the state identity,
then compares MoveIt FK, each published dynamic edge, and the composed TCP
transform for that exact state.  It never edits a trajectory or a Stage 5A
artifact.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from moveit.core.robot_state import RobotState
from moveit.planning import MoveItPy
from rclpy.node import Node


JOINTS = [f"j{i}" for i in range(1, 7)]
GROUP = "fairino5_v6_group"
TCP = "spray_tcp_link"
EDGES = (
    ("base_link", "shoulder_link"),
    ("shoulder_link", "upperarm_link"),
    ("upperarm_link", "forearm_link"),
    ("forearm_link", "wrist1_link"),
    ("wrist1_link", "wrist2_link"),
    ("wrist2_link", "wrist3_link"),
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stamp_key(row: dict[str, Any]) -> tuple[int, int]:
    stamp = row["header_stamp"]
    return int(stamp["sec"]), int(stamp["nanosec"])


def stamp_seconds(key: tuple[int, int]) -> float:
    return float(key[0]) + float(key[1]) * 1e-9


def matrix_from_tf(row: dict[str, Any]) -> np.ndarray:
    q = np.asarray([float(row["rotation"][axis]) for axis in ("x", "y", "z", "w")], dtype=float)
    q /= max(float(np.linalg.norm(q)), 1e-15)
    x, y, z, w = q
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    matrix[:3, 3] = [float(row["translation"][axis]) for axis in ("x", "y", "z")]
    return matrix


def matrix_of(value: Any) -> np.ndarray:
    raw = value.matrix() if hasattr(value, "matrix") else value
    return np.asarray(raw, dtype=float)


def rotation_error(a: np.ndarray, b: np.ndarray) -> float:
    relative = a[:3, :3].T @ b[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(math.acos(cosine))


def quaternion_from_matrix(matrix: np.ndarray) -> list[float]:
    trace = float(np.trace(matrix[:3, :3]))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w, x, y, z = 0.25 * s, (matrix[2, 1] - matrix[1, 2]) / s, (matrix[0, 2] - matrix[2, 0]) / s, (matrix[1, 0] - matrix[0, 1]) / s
    else:
        index = int(np.argmax(np.diag(matrix[:3, :3])))
        if index == 0:
            s = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            w, x, y, z = (matrix[2, 1] - matrix[1, 2]) / s, 0.25 * s, (matrix[0, 1] + matrix[1, 0]) / s, (matrix[0, 2] + matrix[2, 0]) / s
        elif index == 1:
            s = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            w, x, y, z = (matrix[0, 2] - matrix[2, 0]) / s, (matrix[0, 1] + matrix[1, 0]) / s, 0.25 * s, (matrix[1, 2] + matrix[2, 1]) / s
        else:
            s = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            w, x, y, z = (matrix[1, 0] - matrix[0, 1]) / s, (matrix[0, 2] + matrix[2, 0]) / s, (matrix[1, 2] + matrix[2, 1]) / s, 0.25 * s
    return [float(x), float(y), float(z), float(w)]


def read_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    times = np.asarray([float(row["t"]) for row in rows], dtype=float)
    q = np.asarray([[float(row[f"j{i}_q"]) for i in range(1, 7)] for row in rows], dtype=float)
    if q.shape != (len(rows), 6) or len(rows) < 2 or not np.isfinite(times).all() or not np.isfinite(q).all() or np.any(np.diff(times) <= 0):
        raise RuntimeError("invalid_frozen_trajectory")
    return times, q


def compose(edges: dict[tuple[str, str], np.ndarray]) -> np.ndarray | None:
    current = "base_link"
    result = np.eye(4, dtype=float)
    visited: set[str] = set()
    while current != TCP and current not in visited:
        visited.add(current)
        outgoing = [(child, matrix) for (parent, child), matrix in edges.items() if parent == current]
        if len(outgoing) != 1:
            return None
        child, matrix = outgoing[0]
        result = result @ matrix
        current = child
    return result if current == TCP else None


def percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(np.asarray(values, dtype=float), q)) if values else None


class Witness(Node):
    def __init__(self, runtime_dir: Path, trajectory: Path, output: Path) -> None:
        super().__init__("stage5ar_exact_tf_fk_witness")
        self.runtime_dir = runtime_dir
        self.trajectory = trajectory
        self.output = output

    def run(self) -> dict[str, Any]:
        joint_rows = read_jsonl(self.runtime_dir / "stage5a_joint_states_raw.jsonl")
        tf_rows = read_jsonl(self.runtime_dir / "stage5a_tf_raw.jsonl")
        static_rows = read_jsonl(self.runtime_dir / "stage5a_tf_static_raw.jsonl")
        times, d65_q = read_trajectory(self.trajectory)
        moveit = MoveItPy(node_name="stage5ar_exact_tf_fk_witness_moveit")
        model = moveit.get_robot_model()
        group = model.get_joint_model_group(GROUP)
        if group is None or list(group.active_joint_model_names) != JOINTS:
            raise RuntimeError("joint_mapping_mismatch")
        joint_by_stamp: dict[tuple[int, int], list[dict[str, Any]]] = {}
        for row in joint_rows:
            if len(row.get("name", [])) == 6 and set(row.get("name", [])) == set(JOINTS):
                joint_by_stamp.setdefault(stamp_key(row), []).append(row)
        tf_by_stamp: dict[tuple[int, int], list[dict[str, Any]]] = {}
        for row in tf_rows:
            tf_by_stamp.setdefault(stamp_key(row), []).append(row)
        static_edges = {(str(row["parent_frame"]), str(row["child_frame"])): matrix_from_tf(row) for row in static_rows}
        common = sorted(set(joint_by_stamp) & set(tf_by_stamp))
        missing_joint_stamp = sorted(set(tf_by_stamp) - set(joint_by_stamp))
        state = RobotState(model)
        position_errors: list[float] = []
        orientation_errors: list[float] = []
        edge_position_errors: list[float] = []
        edge_orientation_errors: list[float] = []
        witnesses: list[dict[str, Any]] = []
        compared_rows = 0
        malformed_batches = 0
        for key in common:
            joint_batch = joint_by_stamp[key]
            tf_batch = tf_by_stamp[key]
            if len(joint_batch) != 1:
                malformed_batches += 1
                continue
            dynamic = {(str(row["parent_frame"]), str(row["child_frame"])): matrix_from_tf(row) for row in tf_batch}
            if any(edge not in dynamic for edge in EDGES):
                malformed_batches += 1
                continue
            row = joint_batch[0]
            names = list(row["name"])
            q = np.asarray([float(row["position"][names.index(joint)]) for joint in JOINTS], dtype=float)
            state.set_joint_group_positions(GROUP, q.tolist())
            state.update()
            all_edges = dict(dynamic)
            all_edges.update(static_edges)
            observed_tcp = compose(all_edges)
            if observed_tcp is None:
                malformed_batches += 1
                continue
            expected_global: dict[str, np.ndarray] = {"base_link": np.eye(4, dtype=float)}
            for parent, child in EDGES:
                # MoveItPy Jazzy accepts the link name directly; RobotModel's
                # Python binding does not expose get_link_model().
                expected_global[child] = matrix_of(state.get_global_link_transform(child))
            expected_tcp = matrix_of(state.get_global_link_transform(TCP))
            dynamic_edge_error = []
            for parent, child in EDGES:
                expected_edge = np.linalg.inv(expected_global[parent]) @ expected_global[child]
                observed_edge = dynamic[(parent, child)]
                p_error = float(np.linalg.norm(expected_edge[:3, 3] - observed_edge[:3, 3]))
                r_error = rotation_error(expected_edge, observed_edge)
                edge_position_errors.append(p_error)
                edge_orientation_errors.append(r_error)
                dynamic_edge_error.append({"edge": f"{parent}->{child}", "position_error_m": p_error, "orientation_error_rad": r_error})
            p_error = float(np.linalg.norm(expected_tcp[:3, 3] - observed_tcp[:3, 3]))
            r_error = rotation_error(expected_tcp, observed_tcp)
            position_errors.append(p_error)
            orientation_errors.append(r_error)
            compared_rows += 1
            d65_index = int(np.argmin(np.linalg.norm(d65_q - q[None, :], axis=1)))
            witnesses.append({
                "state_sequence_id": {"header_stamp": {"sec": key[0], "nanosec": key[1]}, "joint_message_index": int(row["message_index"]), "tf_message_indices": [int(item["message_index"]) for item in tf_batch]},
                "source_timestamp_s": stamp_seconds(key),
                "joint_capture_monotonic_s": float(row["capture_monotonic_s"]),
                "tf_capture_monotonic_s": [float(item["capture_monotonic_s"]) for item in tf_batch],
                "q_reported": q.tolist(),
                "nearest_d65_state_index": d65_index,
                "nearest_d65_q_l2_rad": float(np.linalg.norm(d65_q[d65_index] - q)),
                "tf_frame_ids": [f"{parent}->{child}" for parent, child in EDGES] + ["wrist3_link->spray_tcp_link"],
                "tcp_expected_moveit_fk": {"translation_m": expected_tcp[:3, 3].tolist(), "quaternion_xyzw": quaternion_from_matrix(expected_tcp)},
                "tcp_observed_tf_composed": {"translation_m": observed_tcp[:3, 3].tolist(), "quaternion_xyzw": quaternion_from_matrix(observed_tcp)},
                "tcp_position_error_m": p_error,
                "tcp_orientation_error_rad": r_error,
                "dynamic_edge_errors": dynamic_edge_error,
            })
        worst = sorted(witnesses, key=lambda item: (item["tcp_position_error_m"], item["tcp_orientation_error_rad"]), reverse=True)[:5]
        stats = {
            "schema_version": "stage5ar-exact-same-state-tf-fk-witness-v1",
            "status": "MEASURED_COMPLETE" if compared_rows else "BLOCKED_NO_COMMON_STATES",
            "runtime_dir": str(self.runtime_dir.resolve()),
            "trajectory": {"path": str(self.trajectory.resolve()), "sha256": sha256(self.trajectory), "state_count": int(len(times)), "duration_s": float(times[-1])},
            "model": {"moveit_group": GROUP, "moveit_tcp": TCP, "dynamic_edges": [f"{a}->{b}" for a, b in EDGES], "static_edges": [f"{a}->{b}" for a, b in static_edges]},
            "pairing": {"identity": "exact JointState.header.stamp == TFMessage TransformStamped.header.stamp", "causal_arrival_pairing_used": False, "joint_rows": len(joint_rows), "tf_rows": len(tf_rows), "joint_stamp_groups": len(joint_by_stamp), "tf_stamp_groups": len(tf_by_stamp), "common_stamp_groups": len(common), "missing_joint_stamp_group_count": len(missing_joint_stamp), "malformed_common_batch_count": malformed_batches},
            "samples_compared": compared_rows,
            "tcp": {"max_position_error_m": max(position_errors) if position_errors else None, "mean_position_error_m": float(np.mean(position_errors)) if position_errors else None, "rms_position_error_m": float(np.sqrt(np.mean(np.square(position_errors)))) if position_errors else None, "p50_position_error_m": percentile(position_errors, 50), "p95_position_error_m": percentile(position_errors, 95), "p99_position_error_m": percentile(position_errors, 99), "max_orientation_error_rad": max(orientation_errors) if orientation_errors else None, "mean_orientation_error_rad": float(np.mean(orientation_errors)) if orientation_errors else None, "rms_orientation_error_rad": float(np.sqrt(np.mean(np.square(orientation_errors)))) if orientation_errors else None, "p95_orientation_error_rad": percentile(orientation_errors, 95), "p99_orientation_error_rad": percentile(orientation_errors, 99)},
            "individual_dynamic_edges": {"max_position_error_m": max(edge_position_errors) if edge_position_errors else None, "max_orientation_error_rad": max(edge_orientation_errors) if edge_orientation_errors else None},
            "strict_gate": {"position_m": 1e-5, "orientation_rad": 1e-5, "passed": bool(position_errors and orientation_errors and max(position_errors) <= 1e-5 and max(orientation_errors) <= 1e-5)},
            "worst_state_witnesses": worst,
            "interpretation": "same-state MoveIt FK versus published TF composition; mock/runtime observation evidence only, not physical tracking or hardware safety",
        }
        write_json(self.output / "STAGE5AR_TF_FK_SAME_STATE_SUMMARY.json", stats)
        with (self.output / "STAGE5AR_TF_FK_SAME_STATE_WITNESSES.jsonl").open("w", encoding="utf-8") as handle:
            for item in witnesses:
                handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
        return stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args, ros_args = parser.parse_known_args()
    rclpy.init(args=ros_args)
    node = Witness(args.runtime_dir.resolve(), args.trajectory.resolve(), args.output.resolve())
    try:
        result = node.run()
        print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
        return 0 if result["status"] == "MEASURED_COMPLETE" and result["strict_gate"]["passed"] else 3
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
