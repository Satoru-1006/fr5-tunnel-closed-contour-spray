#!/usr/bin/env python3
"""Stage D5 bounded-limit matrix and legal spray-task freedom shadow.

This tool deliberately keeps the exact 181-point target set immutable.  The
task-freedom cases are separate shadow targets derived only from the frozen
Stage 3 process contract: standoff 0.25--0.27 m and normal/incidence
deviation <= 5 deg.  The project-defined deterministic tool roll is not
treated as a free degree of freedom.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class Joint:
    name: str
    xyz: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float


def rpy(rpy_value: np.ndarray) -> np.ndarray:
    return Rotation.from_euler("xyz", rpy_value).as_matrix()


def parse_chain(path: Path) -> tuple[list[Joint], np.ndarray]:
    root = ET.parse(path).getroot()
    elements = {item.attrib["name"]: item for item in root.findall("./joint")}
    chain: list[Joint] = []
    parent = "base_link"
    for index in range(1, 7):
        elem = elements[f"joint_{index}"]
        origin = elem.find("origin")
        axis = elem.find("axis")
        limit = elem.find("limit")
        vector = np.fromstring(axis.attrib["xyz"], sep=" ")
        vector = vector / np.linalg.norm(vector)
        chain.append(Joint(
            elem.attrib["name"],
            np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" "),
            np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" "),
            vector,
            float(limit.attrib["lower"]),
            float(limit.attrib["upper"]),
        ))
        if elem.find("parent").attrib["link"] != parent:
            raise RuntimeError(f"unexpected chain parent at joint_{index}")
        parent = elem.find("child").attrib["link"]
    tool = np.eye(4)
    while parent != "sames_asb_spray_tcp":
        fixed = [item for item in root.findall("./joint") if item.find("parent").attrib["link"] == parent]
        if len(fixed) != 1 or fixed[0].attrib.get("type") != "fixed":
            raise RuntimeError(f"cannot resolve fixed chain after {parent}")
        origin = fixed[0].find("origin")
        transform = np.eye(4)
        transform[:3, :3] = rpy(np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" "))
        transform[:3, 3] = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ")
        tool = tool @ transform
        parent = fixed[0].find("child").attrib["link"]
    return chain, tool


def fk(q: np.ndarray, chain: list[Joint], tool: np.ndarray) -> np.ndarray:
    result = np.eye(4)
    for value, joint in zip(q, chain):
        origin = np.eye(4)
        origin[:3, :3] = rpy(joint.rpy)
        origin[:3, 3] = joint.xyz
        result = result @ origin
        result[:3, :3] = result[:3, :3] @ Rotation.from_rotvec(joint.axis * float(value)).as_matrix()
    return result @ tool


def pose_error(q: np.ndarray, target: np.ndarray, chain: list[Joint], tool: np.ndarray) -> np.ndarray:
    actual = fk(q, chain, tool)
    orientation = Rotation.from_matrix(actual[:3, :3].T @ target[:3, :3]).as_rotvec()
    return np.r_[actual[:3, 3] - target[:3, 3], orientation]


def target_pose(row: dict[str, str]) -> np.ndarray:
    result = np.eye(4)
    result[:3, 3] = [float(row[key]) for key in ("x", "y", "z")]
    quaternion = np.asarray([float(row[key]) for key in ("qx", "qy", "qz", "qw")])
    quaternion /= np.linalg.norm(quaternion)
    result[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    return result


def load_target_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 181:
        raise RuntimeError(f"expected 181 authoritative target rows, got {len(rows)}")
    return rows


def q_to_row(q: np.ndarray) -> dict[str, str]:
    return {f"q{i + 1}_rad": f"{value:.17g}" for i, value in enumerate(q)}


def solve_bounded(target: np.ndarray, seed: np.ndarray, chain: list[Joint], tool: np.ndarray, max_nfev: int) -> tuple[np.ndarray, float, float, int]:
    lower = np.asarray([joint.lower for joint in chain]) + 1e-10
    upper = np.asarray([joint.upper for joint in chain]) - 1e-10
    result = least_squares(
        lambda q: pose_error(q, target, chain, tool),
        np.clip(seed, lower, upper),
        bounds=(lower, upper),
        max_nfev=max_nfev,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        x_scale="jac",
    )
    residual = pose_error(result.x, target, chain, tool)
    return result.x, float(np.linalg.norm(residual[:3])), float(np.linalg.norm(residual[3:])), int(result.nfev)


def halton(index: int, base: int) -> float:
    value = 0.0
    fraction = 1.0 / base
    while index:
        value += (index % base) * fraction
        index //= base
        fraction /= base
    return value


def task_starts(chain: list[Joint], extra: list[np.ndarray], include_halton: bool = True) -> list[np.ndarray]:
    limits = [(joint.lower, joint.upper) for joint in chain]
    lower = np.asarray([item[0] for item in limits]) + 1e-9
    upper = np.asarray([item[1] for item in limits]) - 1e-9
    starts = [np.asarray(extra_item, dtype=float) for extra_item in extra]
    starts.append((lower + upper) / 2.0)
    starts.append(np.zeros(6))
    if include_halton:
        for index in range(1, 9):
            starts.append(np.asarray([lo + halton(index, base) * (hi - lo) for (lo, hi), base in zip(limits, (2, 3, 5, 7, 11, 13))]))
    unique: list[np.ndarray] = []
    for value in starts:
        clipped = np.clip(value, lower, upper)
        if not any(np.max(np.abs(clipped - old)) < 1e-8 for old in unique):
            unique.append(clipped)
    return unique


def max_violation(q: np.ndarray, chain: list[Joint]) -> tuple[float, int, float, int]:
    violations = []
    for index, joint in enumerate(chain):
        if q[index] < joint.lower:
            signed = q[index] - joint.lower
        elif q[index] > joint.upper:
            signed = q[index] - joint.upper
        else:
            signed = 0.0
        violations.append(signed)
    absolute = np.abs(violations)
    index = int(np.argmax(absolute))
    return float(violations[index]), index, float(absolute[index]), int(np.count_nonzero(absolute > 1e-9))


def margin(q: np.ndarray, chain: list[Joint]) -> float:
    return float(min(min((value - joint.lower) / (joint.upper - joint.lower), (joint.upper - value) / (joint.upper - joint.lower)) for value, joint in zip(q, chain)))


def variant_pose(exact: np.ndarray, standoff_m: float, tilt_axis: str, angle_deg: float) -> np.ndarray:
    original_direction = exact[:3, 2]
    surface = exact[:3, 3] + 0.26 * original_direction
    result = exact.copy()
    if tilt_axis != "none" and abs(angle_deg) > 0.0:
        axis = {"local_x": np.array([1.0, 0.0, 0.0]), "local_y": np.array([0.0, 1.0, 0.0])}[tilt_axis]
        result[:3, :3] = exact[:3, :3] @ Rotation.from_rotvec(axis * math.radians(angle_deg)).as_matrix()
    # H3 semantics: TCP position is surface point minus the declared
    # standoff along the *current* spray direction (the target local +Z),
    # not along the pre-tilt normal.  The former implementation placed an
    # incidence-tilted orientation at the old-normal standoff, which made
    # those shadow targets semantically inconsistent.
    result[:3, 3] = surface - standoff_m * result[:3, 2]
    return result


def pose_csv_row(pose: np.ndarray) -> dict[str, str]:
    quaternion = Rotation.from_matrix(pose[:3, :3]).as_quat()
    return {
        "x": f"{pose[0, 3]:.17g}", "y": f"{pose[1, 3]:.17g}", "z": f"{pose[2, 3]:.17g}",
        "qx": f"{quaternion[0]:.17g}", "qy": f"{quaternion[1]:.17g}", "qz": f"{quaternion[2]:.17g}", "qw": f"{quaternion[3]:.17g}",
    }


def write_targets(path: Path, rows: list[dict[str, str]], poses: list[np.ndarray]) -> None:
    fields = ["x", "y", "z", "qx", "qy", "qz", "qw"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for pose in poses:
            writer.writerow(pose_csv_row(pose))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--unbounded", type=Path, required=True)
    parser.add_argument("--d4-waypoints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-nfev", type=int, default=220)
    parser.add_argument("--reduced-starts", action="store_true", help="post-audit speed mode: use root, prior best, center, and zero only")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    chain, tool = parse_chain(args.urdf)
    rows = load_target_rows(args.targets)
    exact_poses = [target_pose(row) for row in rows]

    with args.unbounded.open(newline="", encoding="utf-8") as stream:
        roots = {int(row["point_id"]): row for row in csv.DictReader(stream)}
    with args.d4_waypoints.open(newline="", encoding="utf-8") as stream:
        d4_best = {int(row["point_id"]): np.asarray([float(row[f"best_q{i}_rad"]) for i in range(1, 7)]) for row in csv.DictReader(stream)}

    conflict_rows = []
    violations = []
    for point in range(1, 57):
        root = roots[point]
        q = np.asarray([float(root[f"root_q{i}_rad"]) for i in range(1, 7)])
        signed, joint_index, absolute, number = max_violation(q, chain)
        residual = pose_error(q, exact_poses[point - 1], chain, tool)
        violations.append(absolute)
        conflict_rows.append({
            "waypoint_id": point,
            "closest_unbounded_root": f"D4_UNBOUNDED_WP{point:03d}",
            "violating_joint": f"J{joint_index + 1}",
            "official_lower_limit_rad": f"{chain[joint_index].lower:.17g}",
            "official_upper_limit_rad": f"{chain[joint_index].upper:.17g}",
            "official_lower_limit_deg": f"{math.degrees(chain[joint_index].lower):.12g}",
            "official_upper_limit_deg": f"{math.degrees(chain[joint_index].upper):.12g}",
            "root_joint_value_rad": f"{q[joint_index]:.17g}",
            "root_joint_value_deg": f"{math.degrees(q[joint_index]):.12g}",
            "signed_violation_rad": f"{signed:.17g}",
            "absolute_violation_deg": f"{math.degrees(absolute):.12g}",
            "number_of_violating_joints": number,
            "fk_position_error_m": f"{np.linalg.norm(residual[:3]):.17g}",
            "fk_orientation_error_rad": f"{np.linalg.norm(residual[3:]):.17g}",
            "all_root_q_rad": ";".join(f"{value:.17g}" for value in q),
        })
    q33, q67 = np.quantile(np.asarray(violations), [1 / 3, 2 / 3])
    for row, value in zip(conflict_rows, violations):
        row["conflict_class"] = "NEAR_LIMIT_CONFLICT" if value <= q33 else "MODERATE_LIMIT_CONFLICT" if value <= q67 else "LARGE_LIMIT_CONFLICT"
    fields = list(conflict_rows[0])
    with (args.output / "WP1_56_JOINT_LIMIT_CONFLICT.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(conflict_rows)

    # The contract has a legal standoff interval and normal/incidence tolerance.
    # The deterministic roll policy is fixed and therefore excluded from this grid.
    variants = [("exact", 0.26, "none", 0.0)]
    for standoff in (0.25, 0.255, 0.265, 0.27):
        variants.append((f"standoff_{standoff:.3f}", standoff, "none", 0.0))
    for axis in ("local_x", "local_y"):
        for angle in (-5.0, -2.5, 2.5, 5.0):
            variants.append((f"incidence_{axis}_{angle:+.1f}", 0.26, axis, angle))
            for standoff in (0.25, 0.27):
                variants.append((f"incidence_{axis}_{angle:+.1f}_standoff_{standoff:.3f}", standoff, axis, angle))

    freedom_rows = []
    freedom_candidates = []
    selected_by_variant: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    for point in range(1, 57):
        root_q = np.asarray([float(roots[point][f"root_q{i}_rad"]) for i in range(1, 7)])
        starts = task_starts(chain, [root_q, d4_best.get(point, root_q)], include_halton=not args.reduced_starts)
        candidates = []
        for variant_key, standoff, axis, angle in variants:
            target = variant_pose(exact_poses[point - 1], standoff, axis, angle)
            best = None
            for seed_index, seed in enumerate(starts):
                q, position, orientation, nfev = solve_bounded(target, seed, chain, tool, args.max_nfev)
                score = (max(position, orientation), position, orientation, -margin(q, chain))
                if best is None or score < best[0]:
                    best = (score, q, position, orientation, nfev)
                if position <= 1e-7 and orientation <= 1e-7:
                    candidates.append((variant_key, standoff, axis, angle, q, position, orientation, nfev, seed_index))
                    break
        if candidates:
            candidates.sort(key=lambda item: (0 if item[0] == "exact" else 1, abs(item[3]), abs(item[1] - 0.26), -margin(item[4], chain)))
            chosen = candidates[0]
            selected_by_variant.setdefault(chosen[0], {})[point] = (chosen[4], variant_pose(exact_poses[point - 1], chosen[1], chosen[2], chosen[3]))
            freedom_class = "EXACT_POSE_BOUNDED_IK_VALID" if chosen[0] == "exact" else "TASK_FREEDOM_RECOVERABLE"
            chosen_key, chosen_standoff, chosen_axis, chosen_angle, chosen_q, pos, ori, nfev, seed_index = chosen
            freedom_rows.append({
                "waypoint_id": point, "classification": freedom_class, "selected_variant": chosen_key,
                "standoff_m": f"{chosen_standoff:.6g}", "incidence_axis": chosen_axis, "incidence_deg": f"{chosen_angle:.6g}",
                "q1_rad": f"{chosen_q[0]:.17g}", "q2_rad": f"{chosen_q[1]:.17g}", "q3_rad": f"{chosen_q[2]:.17g}", "q4_rad": f"{chosen_q[3]:.17g}", "q5_rad": f"{chosen_q[4]:.17g}", "q6_rad": f"{chosen_q[5]:.17g}",
                "position_error_m": f"{pos:.17g}", "orientation_error_rad": f"{ori:.17g}", "joint_margin_fraction": f"{margin(chosen_q, chain):.17g}",
                "native_validation": "REQUIRED", "native_collision_status": "REQUIRED",
            })
            for candidate in candidates:
                freedom_candidates.append({
                    "waypoint_id": point, "variant": candidate[0], "standoff_m": candidate[1], "incidence_axis": candidate[2], "incidence_deg": candidate[3],
                    **q_to_row(candidate[4]), "position_error_m": candidate[5], "orientation_error_rad": candidate[6], "nfev": candidate[7],
                    "joint_margin_fraction": margin(candidate[4], chain), "native_validation": "REQUIRED",
                })
        else:
            freedom_rows.append({"waypoint_id": point, "classification": "UNRESOLVED", "selected_variant": "", "standoff_m": "", "incidence_axis": "", "incidence_deg": "", "native_validation": "NOT_RUN", "native_collision_status": "NOT_RUN"})

    fields = sorted({key for row in freedom_rows for key in row})
    with (args.output / "TASK_FREEDOM_RECOVERY.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(freedom_rows)
    candidate_fields = sorted({key for row in freedom_candidates for key in row}) if freedom_candidates else ["waypoint_id"]
    with (args.output / "TASK_FREEDOM_CANDIDATES.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=candidate_fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(freedom_candidates)

    native_dir = args.output / "native_cases"
    native_dir.mkdir(exist_ok=True)
    original_targets = exact_poses.copy()
    seed_fields = ["point_id", "seed_id", "seed_source", *[f"q{i}_rad" for i in range(1, 7)]]
    for variant_key, selected in selected_by_variant.items():
        poses = original_targets.copy()
        seed_rows = []
        for point, (q, pose) in selected.items():
            poses[point - 1] = pose
            seed_rows.append({"point_id": point, "seed_id": f"D5_{variant_key}_WP{point:03d}", "seed_source": "D5_legal_task_freedom_shadow", **q_to_row(q)})
        write_targets(native_dir / f"targets_{variant_key}.csv", rows, poses)
        with (native_dir / f"seeds_{variant_key}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=seed_fields); writer.writeheader(); writer.writerows(seed_rows)

    summary = {
        "schema_version": "ier8_stage_d5_limit_task_freedom_v1",
        "active_robot": "ESTUN_iER8_720_MI",
        "target_source": str(args.targets.resolve()),
        "formal_limits_source": str(args.urdf.resolve()),
        "task_contract": {"standoff_interval_m": [0.25, 0.27], "normal_incidence_tolerance_deg": 5.0, "roll": "fixed_project_defined_policy; not treated as free", "incidence_position_semantics": "surface_minus_current_tilted_spray_direction_times_standoff"},
        "wp1_56": {"count": 56, "violation_min_deg": math.degrees(min(violations)), "violation_max_deg": math.degrees(max(violations)), "adaptive_threshold_q33_deg": math.degrees(q33), "adaptive_threshold_q67_deg": math.degrees(q67), "class_counts": {name: sum(row["conflict_class"] == name for row in conflict_rows) for name in ("NEAR_LIMIT_CONFLICT", "MODERATE_LIMIT_CONFLICT", "LARGE_LIMIT_CONFLICT")}},
        "task_freedom_geometry": {"exact_or_task_freedom_candidates": len(freedom_candidates), "points_with_candidate": sum(row["classification"] != "UNRESOLVED" for row in freedom_rows), "native_validation": "REQUIRED", "selected_variant_count": len(selected_by_variant)},
        "variant_grid": [{"key": key, "standoff_m": standoff, "incidence_axis": axis, "incidence_deg": angle} for key, standoff, axis, angle in variants],
        "collision_method": "adaptive_discrete_interpolation",
        "ccd_status": "not_available",
        "clearance_status": "not_available",
    }
    (args.output / "D5_LIMIT_TASK_FREEDOM_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    try:
        import matplotlib.pyplot as plt
        points = np.arange(1, 57)
        values = np.degrees(np.asarray(violations))
        colors = ["#2ca02c" if value <= math.degrees(q33) else "#ff7f0e" if value <= math.degrees(q67) else "#d62728" for value in values]
        fig, ax = plt.subplots(figsize=(10, 4.8), dpi=160)
        ax.bar(points, values, color=colors, width=0.85)
        ax.axhline(math.degrees(q33), color="#ff7f0e", linestyle="--", linewidth=1, label=f"adaptive Q33={math.degrees(q33):.2f}°")
        ax.axhline(math.degrees(q67), color="#d62728", linestyle="--", linewidth=1, label=f"adaptive Q67={math.degrees(q67):.2f}°")
        ax.set(xlabel="Waypoint", ylabel="Maximum joint-limit violation (deg)", title="D5 WP1–56 unbounded-root limit conflict")
        ax.legend(loc="upper left", fontsize=8); ax.grid(axis="y", alpha=0.25); fig.tight_layout()
        fig.savefig(args.output / "FIG1_WP1_56_LIMIT_VIOLATION.png"); plt.close(fig)
    except Exception as exc:
        (args.output / "FIGURE_BACKEND_UNAVAILABLE.txt").write_text(f"matplotlib figure generation unavailable: {type(exc).__name__}: {exc}\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
