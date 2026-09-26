from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]


def read_xyz(path: Path, start: int, end: int) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return np.asarray([[float(row[key]) for key in ("x", "y", "z")] for row in rows[start : end + 1]])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build base_link TCP poses and seeds for the open arch MoveIt validation.")
    parser.add_argument("--surface-path", type=Path, default=ROOT / "outputs" / "surface_path.csv")
    parser.add_argument("--tcp-path", type=Path, default=ROOT / "outputs" / "tcp_path.csv")
    parser.add_argument("--ik-csv", type=Path, default=ROOT / "outputs" / "animation_open_arch_lshape_ik.csv")
    parser.add_argument("--out-poses", type=Path, default=ROOT / "outputs" / "open_arch_tcp_poses_base_link.csv")
    parser.add_argument("--out-seeds", type=Path, default=ROOT / "outputs" / "open_arch_seed_joints.csv")
    parser.add_argument("--phase-start", type=int, default=30)
    parser.add_argument("--phase-end", type=int, default=210)
    parser.add_argument("--base-y", type=float, default=0.45)
    parser.add_argument("--base-z", type=float, default=0.20)
    parser.add_argument("--work-plane-y", type=float, default=0.55)
    args = parser.parse_args()

    surface = read_xyz(args.surface_path, args.phase_start, args.phase_end)
    tcp = read_xyz(args.tcp_path, args.phase_start, args.phase_end)
    tcp[:, 1] = args.work_plane_y
    surface[:, 1] = args.work_plane_y
    with args.ik_csv.open(newline="", encoding="utf-8") as f:
        seeds = list(csv.DictReader(f))
    if len(seeds) != len(tcp):
        raise ValueError(f"Seed count {len(seeds)} does not match open-arch pose count {len(tcp)}.")

    # World-to-base transform for the relocated robot base: yaw = pi.
    base_rotation = Rotation.from_euler("z", np.pi).as_matrix()
    world_to_base_rotation = base_rotation.T
    base_origin = np.asarray([0.0, args.base_y, args.base_z])
    pose_rows: list[dict[str, float]] = []
    for position, surface_point in zip(tcp, surface):
        tool_z_world = surface_point - position
        tool_z_world /= np.linalg.norm(tool_z_world)
        x_world = np.cross(np.asarray([0.0, 1.0, 0.0]), tool_z_world)
        x_world /= np.linalg.norm(x_world)
        y_world = np.cross(tool_z_world, x_world)
        rotation_world = np.column_stack([x_world, y_world, tool_z_world])
        position_base = world_to_base_rotation @ (position - base_origin)
        normal_base = world_to_base_rotation @ tool_z_world
        rotation_base = world_to_base_rotation @ rotation_world
        qx, qy, qz, qw = Rotation.from_matrix(rotation_base).as_quat()
        pose_rows.append(
            {
                "x": float(position_base[0]),
                "y": float(position_base[1]),
                "z": float(position_base[2]),
                "qx": float(qx),
                "qy": float(qy),
                "qz": float(qz),
                "qw": float(qw),
                "nx": float(normal_base[0]),
                "ny": float(normal_base[1]),
                "nz": float(normal_base[2]),
            }
        )
    seed_rows = [
        {f"q{i}": float(row[f"j{i}_q"]) for i in range(1, 7)}
        for row in seeds
    ]
    args.out_poses.parent.mkdir(parents=True, exist_ok=True)
    with args.out_poses.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["x", "y", "z", "qx", "qy", "qz", "qw", "nx", "ny", "nz"])
        writer.writeheader()
        writer.writerows(pose_rows)
    with args.out_seeds.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[f"q{i}" for i in range(1, 7)])
        writer.writeheader()
        writer.writerows(seed_rows)
    print(f"Wrote {args.out_poses} and {args.out_seeds} ({len(pose_rows)} open-arch points).")


if __name__ == "__main__":
    main()
