#!/usr/bin/env python3
"""Validate generated TCP poses before moving into a ROS2 workspace."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def _load_pose_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points: list[list[float]] = []
    quaternions: list[list[float]] = []
    normals: list[list[float]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"x", "y", "z", "qx", "qy", "qz", "qw", "nx", "ny", "nz"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        for row in reader:
            points.append([float(row["x"]), float(row["y"]), float(row["z"])])
            quaternions.append([float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"])])
            normal = np.array([float(row["nx"]), float(row["ny"]), float(row["nz"])], dtype=float)
            normal /= max(float(np.linalg.norm(normal)), 1e-12)
            normals.append(normal.tolist())
    if not points:
        raise ValueError(f"{path} contains no TCP poses.")
    return np.asarray(points), np.asarray(quaternions), np.asarray(normals)


def _tool_z_from_quaternion(quaternion: np.ndarray) -> np.ndarray:
    q = quaternion / np.maximum(np.linalg.norm(quaternion, axis=1, keepdims=True), 1e-12)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.column_stack(
        [
            2.0 * (x * z + y * w),
            2.0 * (y * z - x * w),
            1.0 - 2.0 * (x * x + y * y),
        ]
    )


def _closure_error(points: np.ndarray, samples_per_loop: int) -> float:
    if samples_per_loop <= 0 or len(points) < samples_per_loop + 1:
        return float("nan")
    return float(np.linalg.norm(points[0] - points[samples_per_loop]))


def validate_tcp_pose_csv(
    path: Path,
    samples_per_loop: int,
    max_normal_error_deg: float = 0.1,
    max_station_std: float = 1e-9,
) -> dict[str, float | str]:
    points, quaternions, normals = _load_pose_csv(path)
    tool_z = _tool_z_from_quaternion(quaternions)
    dots = np.clip(np.sum(tool_z * normals, axis=1), -1.0, 1.0)
    normal_error = np.rad2deg(np.arccos(dots))
    q_norm = np.linalg.norm(quaternions, axis=1)
    metrics: dict[str, float | str] = {
        "pose_count": float(len(points)),
        "quaternion_norm_min": float(np.min(q_norm)),
        "quaternion_norm_max": float(np.max(q_norm)),
        "normal_angle_error_max_deg": float(np.max(normal_error)),
        "normal_angle_error_mean_deg": float(np.mean(normal_error)),
        "station_y_std": float(np.std(points[:, 1])),
        "close_position_error": _closure_error(points, samples_per_loop),
    }
    failures: list[str] = []
    if float(metrics["normal_angle_error_max_deg"]) > max_normal_error_deg:
        failures.append("quaternion/normal alignment")
    if abs(float(metrics["quaternion_norm_min"]) - 1.0) > 1e-6 or abs(float(metrics["quaternion_norm_max"]) - 1.0) > 1e-6:
        failures.append("quaternion normalization")
    if float(metrics["station_y_std"]) > max_station_std:
        failures.append("fixed station y")
    if samples_per_loop > 0 and float(metrics["close_position_error"]) > 1e-6:
        failures.append("closed contour repeat")
    metrics["status"] = "pass" if not failures else "fail:" + ",".join(failures)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate FR5 MoveIt bridge TCP pose CSV.")
    parser.add_argument("--tcp-path-csv", type=Path, default=Path("outputs/tcp_poses.csv"))
    parser.add_argument("--samples-per-loop", type=int, default=240)
    parser.add_argument("--report-csv", type=Path, default=Path("outputs/bridge_input_validation.csv"))
    args = parser.parse_args()

    metrics = validate_tcp_pose_csv(args.tcp_path_csv, args.samples_per_loop)
    args.report_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.report_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows(metrics.items())
    for key, value in metrics.items():
        print(f"{key}: {value}")
    if metrics["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
