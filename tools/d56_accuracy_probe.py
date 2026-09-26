"""Measure TCP/path error for a D56 native candidate against the frozen path."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.stage4c_execution_form import project_reference, quat_angle
from tools.d56_prepare_candidate_manifest import CASES


def read_q(path: Path) -> np.ndarray:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return np.asarray([[float(row[f"j{i}_q"]) for i in range(1, 7)] for row in rows], dtype=float)


def read_targets(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    p = np.asarray([[float(row[k]) for k in ("x", "y", "z")] for row in rows], dtype=float)
    quat = np.asarray([[float(row[k]) for k in ("qx", "qy", "qz", "qw")] for row in rows], dtype=float)
    normal = np.asarray([[float(row[k]) for k in ("nx", "ny", "nz")] for row in rows], dtype=float)
    if len(p) < 2 or quat.shape != (len(p), 4) or normal.shape != (len(p), 3):
        raise RuntimeError(f"target_shape_invalid:{path}")
    normal_norm = np.linalg.norm(normal, axis=1)
    if not np.isfinite(normal).all() or np.any(normal_norm <= 1.0e-12):
        raise RuntimeError(f"target_normal_invalid:{path}")
    return p, quat, normal / normal_norm[:, None]


def read_fk(path: Path) -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            result[str(row["case_id"])].append(row)
    return result


def quat_z_axis(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=float)
    norm = float(np.linalg.norm(quat))
    if norm <= 1.0e-15:
        raise RuntimeError("zero_quaternion")
    x, y, z, w = quat / norm
    axis = np.asarray((2.0 * (x * z + y * w), 2.0 * (y * z - x * w), 1.0 - 2.0 * (x * x + y * y)), dtype=float)
    return axis / max(float(np.linalg.norm(axis)), 1.0e-15)


def project_task_path(actual_p: np.ndarray, target_p: np.ndarray, target_normal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project TCP samples to the authoritative task-space polyline.

    Native post-Ruckig trajectories are densely resampled and may have a
    different time allocation than the 181-point task input.  A nearest-point
    projection measures geometric path fidelity without treating a timing
    change as a Cartesian tracking error.  The target normal is interpolated
    at the same projected segment.
    """
    best_distance = np.full(len(actual_p), np.inf, dtype=float)
    best_position = np.zeros_like(actual_p)
    best_normal = np.zeros_like(actual_p)
    for index in range(len(target_p) - 1):
        delta = target_p[index + 1] - target_p[index]
        denom = float(np.dot(delta, delta))
        alpha = np.zeros(len(actual_p), dtype=float) if denom <= 1.0e-24 else np.clip((actual_p - target_p[index]) @ delta / denom, 0.0, 1.0)
        position = target_p[index] + alpha[:, None] * delta
        normal = target_normal[index] + alpha[:, None] * (target_normal[index + 1] - target_normal[index])
        normal /= np.maximum(np.linalg.norm(normal, axis=1), 1.0e-15)[:, None]
        distance = np.linalg.norm(actual_p - position, axis=1)
        mask = distance < best_distance
        best_distance[mask] = distance[mask]
        best_position[mask] = position[mask]
        best_normal[mask] = normal[mask]
    return best_position, best_normal, best_distance


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--q-baseline-dir", type=Path, default=None)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--fk", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    target_p, target_quat, target_normal = read_targets(args.targets)
    fk = read_fk(args.fk)
    cases = []
    for case_id, family in CASES:
        if case_id not in fk:
            continue
        source_q = read_q(args.source_dir / f"{case_id}.csv")
        candidate_q = read_q(args.candidate_dir / f"{case_id}.csv")
        actual = fk[case_id]
        actual_p = np.asarray([[float(row[k]) for k in ("x_m", "y_m", "z_m")] for row in actual], dtype=float)
        actual_quat = np.asarray([[float(row[k]) for k in ("qx", "qy", "qz", "qw")] for row in actual], dtype=float)
        if len(candidate_q) != len(actual_p):
            raise RuntimeError(f"fk_count_mismatch:{case_id}:{len(candidate_q)}:{len(actual_p)}")
        _, _, projection = project_reference(candidate_q, source_q, target_p, target_quat)
        desired_p, desired_normal, _ = project_task_path(actual_p, target_p, target_normal)
        position_error = np.linalg.norm(actual_p - desired_p, axis=1)
        actual_normal = np.asarray([quat_z_axis(item) for item in actual_quat], dtype=float)
        normal_error = np.asarray([
            math.acos(float(np.clip(np.dot(actual, desired), -1.0, 1.0)))
            for actual, desired in zip(actual_normal, desired_normal)
        ], dtype=float)
        q_baseline = read_q(args.q_baseline_dir / f"{case_id}.csv") if args.q_baseline_dir else source_q
        q_delta = float(np.max(np.abs(candidate_q - q_baseline))) if candidate_q.shape == q_baseline.shape else None
        cases.append({
            "case_id": case_id,
            "family": family,
            "q_path_max_abs_delta_rad_vs_source": q_delta,
            "max_joint_projection_error_rad": float(np.max(projection)),
            "tcp_trajectory_error_p95_m": float(np.quantile(position_error, 0.95)),
            "tcp_trajectory_error_max_m": float(np.max(position_error)),
            "terminal_position_error_m": float(np.linalg.norm(actual_p[-1] - target_p[-1])),
            "terminal_orientation_error_rad": None,
            "terminal_normal_error_rad": float(math.acos(float(np.clip(np.dot(actual_normal[-1], target_normal[-1]), -1.0, 1.0)))),
            "tcp_normal_error_p95_rad": float(np.quantile(normal_error, 0.95)),
            "tcp_normal_error_max_rad": float(np.max(normal_error)),
            "finite": bool(np.isfinite(candidate_q).all() and np.isfinite(actual_p).all() and np.isfinite(actual_quat).all()),
        })
    report = {
        "schema_version": "d56-accuracy-probe-v1",
        "status": "PASS" if all(row["finite"] for row in cases) else "BLOCKED",
        "scope": "Stage 0/1 ON-state open-arch only",
        "cases": cases,
        "max_tcp_trajectory_error_m": max(row["tcp_trajectory_error_max_m"] for row in cases),
        "max_tcp_trajectory_error_p95_m": max(row["tcp_trajectory_error_p95_m"] for row in cases),
        "max_joint_projection_error_rad": max(row["max_joint_projection_error_rad"] for row in cases),
        "q_path_max_abs_delta_rad": max((row["q_path_max_abs_delta_rad_vs_source"] for row in cases if row["q_path_max_abs_delta_rad_vs_source"] is not None), default=None),
        "interpretation": "TCP path error is nearest-point deviation to the authoritative 181-point task polyline; TCP orientation is evaluated through the authoritative target normal, because target quaternion roll is not the established D41 acceptance observable; no task-accuracy threshold is invented",
        "task_accuracy_method": "nearest_point_authoritative_task_polyline_plus_tcp_z_axis_normal_error",
        "full_quaternion_roll_error": "NOT_USED_TARGET_ROLL_NOT_AUTHORITATIVE",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(args.output.resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
