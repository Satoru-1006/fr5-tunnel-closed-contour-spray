"""Compute a conservative continuous-time clearance lower bound.

The bound is intentionally independent of FCL's unavailable continuous
distance output.  It uses exact collision-mesh vertex radii from the supplied
URDF, fixed-chain translation lengths, and the verified native jerk bound to
bound each joint's displacement inside every persisted sample interval.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def mesh_radius(path: Path) -> float:
    data = path.read_bytes()
    if len(data) < 84:
        raise ValueError(f"mesh_too_short:{path}")
    count = struct.unpack_from("<I", data, 80)[0]
    expected = 84 + 50 * count
    if expected > len(data):
        raise ValueError(f"mesh_not_binary_stl:{path}")
    maximum = 0.0
    offset = 84
    for _ in range(count):
        for vertex in range(3):
            values = struct.unpack_from("<3f", data, offset + 12 + vertex * 12)
            maximum = max(maximum, math.sqrt(sum(float(value) ** 2 for value in values)))
        offset += 50
    return maximum


def xyz_norm(element: ET.Element | None) -> float:
    if element is None:
        return 0.0
    return float(np.linalg.norm(np.asarray([float(value) for value in element.attrib.get("xyz", "0 0 0").split()], dtype=float)))


def collision_radii(urdf: Path, mesh_root: Path) -> tuple[dict[str, float], dict[str, list[tuple[str, float]]]]:
    root = ET.parse(urdf).getroot()
    parent_of: dict[str, tuple[str, float]] = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is not None and child is not None:
            parent_of[child.attrib["link"]] = (parent.attrib["link"], xyz_norm(joint.find("origin")))
    link_shapes: dict[str, list[tuple[str, float]]] = {}
    for link in root.findall("link"):
        shapes: list[tuple[str, float]] = []
        for collision in link.findall("collision"):
            mesh = collision.find("geometry/mesh")
            if mesh is None:
                raise ValueError(f"unsupported_non_mesh_collision:{link.attrib['name']}")
            filename = mesh.attrib["filename"]
            if not filename.startswith("package://fairino_description/"):
                raise ValueError(f"unsupported_mesh_package:{filename}")
            relative = filename.split("package://fairino_description/", 1)[1]
            mesh_path = mesh_root / relative
            shapes.append((str(mesh_path), xyz_norm(collision.find("origin")) + mesh_radius(mesh_path)))
        if shapes:
            link_shapes[link.attrib["name"]] = shapes

    def ancestors(link: str) -> list[tuple[str, float]]:
        chain: list[tuple[str, float]] = []
        current = link
        while current in parent_of:
            parent, translation = parent_of[current]
            chain.append((current, translation))
            current = parent
        chain.reverse()
        return chain

    joint_radii: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for link, shapes in link_shapes.items():
        chain = ancestors(link)
        for joint_index, (joint_link, _) in enumerate(chain):
            downstream_translation = sum(item[1] for item in chain[joint_index + 1 :])
            shape_radius = max(radius for _, radius in shapes)
            joint_radii[f"j{joint_index + 1}"].append((link, downstream_translation + shape_radius))
    per_joint_link: dict[str, dict[str, float]] = {}
    for joint, values in joint_radii.items():
        per_joint_link[joint] = {link: radius for link, radius in values}
    return {link: max(radius for _, radius in shapes) for link, shapes in link_shapes.items()}, per_joint_link


def read_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    t = np.asarray([float(row["t"]) for row in rows], dtype=float)
    q = np.asarray([[float(row[f"j{i}_q"]) for i in range(1, 7)] for row in rows], dtype=float)
    v = np.asarray([[float(row[f"j{i}_dq"]) for i in range(1, 7)] for row in rows], dtype=float)
    a = np.asarray([[float(row[f"j{i}_ddq"]) for i in range(1, 7)] for row in rows], dtype=float)
    if len(t) < 2 or any(not np.isfinite(item).all() for item in (t, q, v, a)) or np.any(np.diff(t) <= 0.0):
        raise ValueError(f"invalid_trajectory:{path}")
    return t, q, v, a


def read_clearance(path: Path) -> dict[str, dict[str, np.ndarray]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            case = row["case_id"]
            grouped[case]["world"].append(float(row["robot_world_distance_m"]))
            grouped[case]["self"].append(float(row["self_distance_m"]))
    return {case: {name: np.asarray(values, dtype=float) for name, values in data.items()} for case, data in grouped.items()}


def bound_coefficients(link_radii: dict[str, float], per_joint_link: dict[str, dict[str, float]]) -> tuple[np.ndarray, np.ndarray]:
    world = np.zeros(6, dtype=float)
    self_bound = np.zeros(6, dtype=float)
    links = sorted(link_radii)
    for index in range(6):
        values = per_joint_link.get(f"j{index + 1}", {})
        world[index] = max(values.values(), default=0.0)
        self_bound[index] = max((values.get(first, 0.0) + values.get(second, 0.0) for first in links for second in links if first != second), default=0.0)
    return world, self_bound


def one_case(case_id: str, trajectory: Path, clearance: dict[str, np.ndarray], world_coeff: np.ndarray, self_coeff: np.ndarray, jerk: float) -> dict[str, Any]:
    t, q, v, a = read_trajectory(trajectory)
    if any(len(clearance[name]) != len(t) for name in ("world", "self")):
        raise ValueError(f"clearance_state_count_mismatch:{case_id}")
    world_lbs: list[float] = []
    self_lbs: list[float] = []
    for index, dt in enumerate(np.diff(t)):
        # Taylor remainder with |jerk| <= jerk bounds the displacement from
        # either endpoint, even when the sampled q path is not linear.
        start_motion = np.abs(v[index]) * dt + 0.5 * np.abs(a[index]) * dt * dt + jerk * dt**3 / 6.0
        end_motion = np.abs(v[index + 1]) * dt + 0.5 * np.abs(a[index + 1]) * dt * dt + jerk * dt**3 / 6.0
        world_lbs.append(max(float(clearance["world"][index] - np.dot(world_coeff, start_motion)), float(clearance["world"][index + 1] - np.dot(world_coeff, end_motion))))
        self_lbs.append(max(float(clearance["self"][index] - np.dot(self_coeff, start_motion)), float(clearance["self"][index + 1] - np.dot(self_coeff, end_motion))))
    return {
        "case_id": case_id,
        "state_count": int(len(t)),
        "interval_count": int(len(t) - 1),
        "sampled_min_world_clearance_m": float(np.min(clearance["world"])),
        "sampled_min_self_clearance_m": float(np.min(clearance["self"])),
        "conservative_continuous_lower_bound_world_m": float(min(world_lbs)),
        "conservative_continuous_lower_bound_self_m": float(min(self_lbs)),
        "positive_world_lower_bound": bool(min(world_lbs) > 0.0),
        "positive_self_lower_bound": bool(min(self_lbs) > 0.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--mesh-root", type=Path, required=True)
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument("--clearance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jerk-bound", type=float, default=8.0)
    args = parser.parse_args()
    link_radii, per_joint_link = collision_radii(args.urdf, args.mesh_root)
    world_coeff, self_coeff = bound_coefficients(link_radii, per_joint_link)
    clearance = read_clearance(args.clearance)
    cases = []
    for trajectory in sorted((args.trajectory_root / "trajectories").glob("*.csv")):
        cases.append(one_case(trajectory.stem, trajectory, clearance[trajectory.stem], world_coeff, self_coeff, args.jerk_bound))
    payload = {
        "schema_version": "d50-conservative-clearance-lower-bound-v1",
        "method": "URDF_collision_mesh_vertex_radius_plus_fixed_chain_translation_Lipschitz_bound",
        "distance_semantics": "continuous_lower_bound_from_native_sampled_distance",
        "jerk_bound_rad_s3": args.jerk_bound,
        "mesh_count": len(link_radii),
        "link_collision_radii_m": link_radii,
        "world_displacement_coefficients_m_per_rad": world_coeff.tolist(),
        "self_displacement_coefficients_m_per_rad": self_coeff.tolist(),
        "case_count": len(cases),
        "cases": cases,
        "positive_lower_bound_all_cases": bool(cases and all(item["positive_world_lower_bound"] and item["positive_self_lower_bound"] for item in cases)),
        "status": "PASS_CONSERVATIVE_POSITIVE_LOWER_BOUND" if cases and all(item["positive_world_lower_bound"] and item["positive_self_lower_bound"] for item in cases) else "UNRESOLVED_NONPOSITIVE_OR_MISSING_BOUND",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if payload["status"].startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
