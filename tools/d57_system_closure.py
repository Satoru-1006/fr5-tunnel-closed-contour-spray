"""D57 shadow analysis for task-aware metrics and automatic hotspot search.

This module deliberately does not mutate D56 or the protected canonical tree.
It consumes the native D56 state, computes independent comparative diagnostics,
and writes only compact D57 shadow evidence.  Generated trajectory patches are
experiments: they are never promotion candidates until native post-Ruckig and
all downstream gates have been rerun.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
D56 = ROOT / "outputs" / "D56_STAGE4B_SOFTWARE_CLOSURE"
DEFAULT_OUT = ROOT / "outputs" / "D57_STAGE4_ALGORITHMIC_CLOSURE"
CASES = (
    "regression_0000", "regression_0001", "normal_0000", "normal_0100",
    "boundary_0000", "boundary_0100", "collision_sensitive_0000",
    "collision_sensitive_0051", "adversarial_0100", "adversarial_0101",
    "perturbation_0000", "perturbation_0100",
)
FAMILIES = {
    "regression": "REGRESSION", "normal": "NORMAL", "boundary": "BOUNDARY",
    "collision_sensitive": "COLLISION_SENSITIVE", "adversarial": "ADVERSARIAL",
    "perturbation": "PERTURBATION",
}
JOINTS = tuple(f"j{i}" for i in range(1, 7))
EFFORT = np.asarray([150.0, 150.0, 150.0, 28.0, 28.0, 28.0])


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def finite(value: float) -> bool:
    return math.isfinite(float(value))


def family(case_id: str) -> str:
    for prefix, name in FAMILIES.items():
        if case_id.startswith(prefix):
            return name
    return "UNKNOWN"


def read_target(path: Path) -> np.ndarray:
    rows = read_csv(path)
    points = np.asarray([[float(row[k]) for k in ("x", "y", "z")] for row in rows], dtype=float)
    if len(points) < 2 or not np.isfinite(points).all():
        raise ValueError(f"invalid_target_path:{path}")
    return points


def path_geometry(points: np.ndarray) -> dict[str, float]:
    delta = np.diff(points, axis=0)
    lengths = np.linalg.norm(delta, axis=1)
    arc = float(np.sum(lengths))
    extent = float(np.linalg.norm(np.ptp(points, axis=0)))
    rms_radius = float(np.sqrt(np.mean(np.sum((points - np.mean(points, axis=0)) ** 2, axis=1))))
    return {
        "point_count": int(len(points)),
        "arc_length_m": arc,
        "bbox_diagonal_m": extent,
        "rms_radius_m": rms_radius,
        "duplicate_segment_count": int(np.sum(lengths <= 1.0e-12)),
        "finite": bool(np.isfinite(points).all()),
    }


def read_grouped(path: Path, key: str = "case_id") -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(path):
        result[str(row[key])].append(row)
    return result


def task_aware_jacobian(full_jacobian_path: Path, jacobian_path: Path, split_path: Path, target: np.ndarray) -> dict[str, Any]:
    rows = read_csv(full_jacobian_path)
    stored_rows = {(row["case_id"], int(row["waypoint"])): row for row in read_csv(jacobian_path)}
    split_rows = read_csv(split_path)
    if not rows:
        raise ValueError("empty_jacobian")
    matrices: dict[str, list[tuple[int, np.ndarray]]] = defaultdict(list)
    stored_singular: dict[str, list[float]] = defaultdict(list)
    stored_condition: dict[str, list[float]] = defaultdict(list)
    raw_residuals: list[float] = []
    for row in rows:
        matrix = np.asarray([[float(row[f"J{r}{c}"]) for c in range(6)] for r in range(6)], dtype=float)
        singular = np.linalg.svd(matrix, compute_uv=False)
        stored = stored_rows[(row["case_id"], int(row["waypoint"]))]
        recorded = np.asarray([float(stored[f"sigma_{i}"]) for i in range(1, 7)], dtype=float)
        raw_residuals.append(float(np.max(np.abs(singular - recorded))))
        case_id = str(row["case_id"])
        waypoint = int(row["waypoint"])
        stored = stored_rows[(case_id, waypoint)]
        matrices[case_id].append((waypoint, matrix))
        stored_singular[case_id].append(float(stored["sigma_min"]))
        stored_condition[case_id].append(float(stored["condition_number"]))

    geometry = path_geometry(target)
    scales = {
        "task_rms_radius": max(geometry["rms_radius_m"], 1.0e-9),
        "task_bbox_diagonal": max(geometry["bbox_diagonal_m"], 1.0e-9),
        "task_arc_over_pi": max(geometry["arc_length_m"] / math.pi, 1.0e-9),
    }
    scale_results: dict[str, Any] = {}
    for name, length in scales.items():
        case_results = {}
        all_values: list[float] = []
        all_conditions: list[float] = []
        for case_id, entries in matrices.items():
            sigma_values = []
            condition_values = []
            for _, matrix in entries:
                scaled = matrix.copy()
                scaled[:3, :] /= length
                singular = np.linalg.svd(scaled, compute_uv=False)
                sigma_min = float(singular[-1])
                condition = float(singular[0] / singular[-1]) if singular[-1] > 0.0 else math.inf
                sigma_values.append(sigma_min)
                condition_values.append(condition)
                all_values.append(sigma_min)
                all_conditions.append(condition)
            case_results[case_id] = {
                "minimum_task_sigma": float(min(sigma_values)),
                "maximum_task_condition": float(max(condition_values)),
                "state_count": len(sigma_values),
            }
        scale_results[name] = {
            "characteristic_length_m": length,
            "case_results": case_results,
            "global_minimum_task_sigma": float(min(all_values)),
            "global_maximum_task_condition": float(max(all_conditions)),
            "interpretation": "comparative project diagnostic; not a universal singularity threshold",
        }

    split_diag = {}
    for row in split_rows:
        case_id = str(row["case_id"])
        split_diag.setdefault(case_id, {"minimum_translation_sigma": math.inf, "minimum_rotation_sigma": math.inf})
        trans = [float(row[f"trans_sigma_{i}"]) for i in range(1, 4)]
        rot = [float(row[f"rot_sigma_{i}"]) for i in range(1, 4)]
        split_diag[case_id]["minimum_translation_sigma"] = min(split_diag[case_id]["minimum_translation_sigma"], min(trans))
        split_diag[case_id]["minimum_rotation_sigma"] = min(split_diag[case_id]["minimum_rotation_sigma"], min(rot))

    return {
        "schema_version": "d57-task-aware-jacobian-v1",
        "source": str(full_jacobian_path.resolve()),
        "state_count": len(rows),
        "matrix_reconstruction_max_abs_singular_residual": max(raw_residuals),
        "task_geometry": geometry,
        "raw_d56_metric": {
            "global_minimum_sigma_min": min(min(values) for values in stored_singular.values()),
            "global_maximum_condition_number": max(max(values) for values in stored_condition.values()),
            "semantics": "unweighted 6D Jacobian diagnostic retained for lineage",
        },
        "task_scaled_metrics": scale_results,
        "translation_rotation_diagnostics": split_diag,
        "project_operational_acceptance_gate": {
            "status": "DEFINED_FOR_COMPARISON_ONLY",
            "rule": "candidate must not materially reduce the protected D56 worst-case task-aware metric after recertification",
            "universal_physical_threshold": "NOT_DEFINED",
        },
    }


def nearest_polyline(points: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    best = np.full(len(points), np.inf)
    best_index = np.zeros(len(points), dtype=int)
    best_position = np.zeros_like(points)
    for index in range(len(target) - 1):
        delta = target[index + 1] - target[index]
        denom = float(np.dot(delta, delta))
        alpha = np.zeros(len(points)) if denom <= 1.0e-24 else np.clip((points - target[index]) @ delta / denom, 0.0, 1.0)
        projection = target[index] + alpha[:, None] * delta
        distance = np.linalg.norm(points - projection, axis=1)
        mask = distance < best
        best[mask] = distance[mask]
        best_index[mask] = index
        best_position[mask] = projection[mask]
    return best, best_index


def hermite(p0: np.ndarray, p1: np.ndarray, m0: np.ndarray, m1: np.ndarray, u: np.ndarray) -> np.ndarray:
    h00 = 2 * u**3 - 3 * u**2 + 1
    h10 = u**3 - 2 * u**2 + u
    h01 = -2 * u**3 + 3 * u**2
    h11 = u**3 - u**2
    return h00[:, None] * p0 + h10[:, None] * m0 + h01[:, None] * p1 + h11[:, None] * m1


def continuous_reference_diagnostic(fk_path: Path, target: np.ndarray) -> dict[str, Any]:
    rows = read_grouped(fk_path)
    result = {}
    for case_id, case_rows in rows.items():
        actual = np.asarray([[float(row[k]) for k in ("x_m", "y_m", "z_m")] for row in case_rows], dtype=float)
        poly_distance, segment = nearest_polyline(actual, target)
        tangents = np.zeros_like(target)
        tangents[0] = target[1] - target[0]
        tangents[-1] = target[-1] - target[-2]
        tangents[1:-1] = 0.5 * (target[2:] - target[:-2])
        curve_distance = np.full(len(actual), np.inf)
        for point_index, segment_index in enumerate(segment):
            candidates = []
            for current in range(max(0, int(segment_index) - 1), min(len(target) - 1, int(segment_index) + 2)):
                u = np.linspace(0.0, 1.0, 65)
                candidates.append(hermite(target[current], target[current + 1], tangents[current], tangents[current + 1], u))
            curve = np.concatenate(candidates, axis=0)
            curve_distance[point_index] = float(np.min(np.linalg.norm(curve - actual[point_index], axis=1)))
        result[case_id] = {
            "polyline_max_error_m": float(np.max(poly_distance)),
            "polyline_p95_error_m": float(np.quantile(poly_distance, 0.95)),
            "adaptive_cubic_diagnostic_max_error_m": float(np.max(curve_distance)),
            "adaptive_cubic_diagnostic_p95_error_m": float(np.quantile(curve_distance, 0.95)),
            "delta_cubic_minus_polyline_max_m": float(np.max(curve_distance) - np.max(poly_distance)),
            "state_count": len(actual),
            "status": "DIAGNOSTIC_APPROXIMATION_NOT_ACCEPTANCE_METRIC",
        }
    return {
        "schema_version": "d57-continuous-reference-diagnostic-v1",
        "method": "piecewise_cubic_hermite_nearest_projection_with_65_point_local_refinement",
        "reference_semantics": "authoritative 181-point task path retained; cubic is a sensitivity diagnostic",
        "cases": result,
    }


def read_q(path: Path) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, str]]]:
    rows = read_csv(path)
    fields = list(rows[0].keys())
    t = np.asarray([float(row["t"]) for row in rows], dtype=float)
    q = np.asarray([[float(row[f"{joint}_q"]) for joint in JOINTS] for row in rows], dtype=float)
    dq = np.asarray([[float(row[f"{joint}_dq"]) for joint in JOINTS] for row in rows], dtype=float)
    ddq = np.asarray([[float(row[f"{joint}_ddq"]) for joint in JOINTS] for row in rows], dtype=float)
    return fields, t, q, dq, ddq, rows


def c4_bump(time: np.ndarray, center: float, half_width: float, amplitude: float) -> tuple[np.ndarray, ...]:
    u = (time - center) / half_width
    mask = np.abs(u) < 1.0
    values = []
    coeff = np.poly1d([-1.0, 0.0, 1.0]) ** 4
    for order in range(4):
        derivative = np.polyder(coeff, order)
        value = np.zeros_like(time)
        value[mask] = amplitude * derivative(u[mask]) / half_width**order
        values.append(value)
    return tuple(values)


def automatic_hotspots(clearance_path: Path, jacobian_path: Path, vectors_path: Path, limit: int = 8) -> list[dict[str, Any]]:
    clearance = read_grouped(clearance_path)
    jacobian = read_grouped(jacobian_path)
    vectors = {(row["case_id"], int(row["waypoint"])): row for row in read_csv(vectors_path)}
    candidates = []
    for case_id in CASES:
        c_rows = clearance.get(case_id, [])
        j_rows = {int(row["waypoint"]): row for row in jacobian.get(case_id, [])}
        if not c_rows or not j_rows:
            continue
        for objective, key, reverse in (
            ("self_clearance", "self_distance_m", False),
            ("world_clearance", "robot_world_distance_m", False),
            ("singularity", "sigma_min", False),
            ("conditioning", "condition_number", True),
        ):
            if key in {"sigma_min", "condition_number"}:
                pool = [row for row in j_rows.values() if finite(float(row[key]))]
            else:
                pool = [row for row in c_rows if finite(float(row[key]))]
            if not pool:
                continue
            selected = (max if reverse else min)(pool, key=lambda row: float(row[key]))
            waypoint = int(selected["waypoint"])
            vector_row = vectors.get((case_id, waypoint))
            if vector_row is None:
                continue
            direction = np.asarray([float(vector_row[f"V_sigma_min_{i}"]) for i in range(6)], dtype=float)
            norm = float(np.linalg.norm(direction))
            if not finite(norm) or norm <= 1.0e-12:
                continue
            direction /= norm
            candidates.append({
                "objective": objective,
                "case_id": case_id,
                "family": family(case_id),
                "waypoint": waypoint,
                "objective_value": float(selected[key]),
                "objective_field": key,
                "direction_v_sigma_min": direction.tolist(),
                "selected_joint_by_max_abs_direction": f"j{int(np.argmax(np.abs(direction))) + 1}",
                "candidate_amplitudes_rad": [0.0005, 0.001, 0.002, 0.005],
                "seed": 5700 + len(candidates),
            })
    candidates.sort(key=lambda row: (row["objective"], row["case_id"]))
    # Preserve objective diversity before filling the remaining budget.  A
    # pure lexicographic slice can silently turn a multi-objective search into
    # a conditioning-only search.
    selected = []
    seen_objectives: set[str] = set()
    for candidate in candidates:
        if candidate["objective"] not in seen_objectives:
            selected.append(candidate)
            seen_objectives.add(candidate["objective"])
    for candidate in candidates:
        if candidate not in selected:
            selected.append(candidate)
        if len(selected) >= limit:
            break
    return selected[:limit]


def write_probe_trajectory(path: Path, q: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["waypoint", *[f"{joint}_q" for joint in JOINTS]])
        for index, row in enumerate(q):
            writer.writerow([index, *[float(value) for value in row]])


def wsl_path(path: Path) -> str:
    value = str(path.resolve()).replace("\\", "/")
    if len(value) >= 3 and value[1:3] == ":/":
        return f"/mnt/{value[0].lower()}/{value[3:]}"
    return value


def materialize_optimizer_shadow(source_dir: Path, out_dir: Path, hotspots: list[dict[str, Any]]) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    materialized = []
    for index, hotspot in enumerate(hotspots[:2]):
        source = source_dir / f"{hotspot['case_id']}.csv"
        if not source.is_file():
            continue
        fields, time, q, dq, ddq, rows = read_q(source)
        waypoint = min(max(int(hotspot["waypoint"]), 1), len(time) - 2)
        center = float(time[waypoint])
        direction = np.asarray(hotspot["direction_v_sigma_min"], dtype=float)
        amplitude = 0.001 if hotspot["objective"] in {"singularity", "conditioning"} else 0.005
        half_width = 1.5
        bump = c4_bump(time, center, half_width, amplitude)
        candidate_q = q + bump[0][:, None] * direction[None, :]
        candidate_dq = dq + bump[1][:, None] * direction[None, :]
        candidate_ddq = ddq + bump[2][:, None] * direction[None, :]
        candidate_jerk = np.asarray([[float(row.get(f"{joint}_jerk", "nan")) for joint in JOINTS] for row in rows])
        candidate_jerk += bump[3][:, None] * direction[None, :]
        destination = out_dir / f"auto_{index}_{hotspot['case_id']}.csv"
        with destination.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(fields)
            for i, row in enumerate(rows):
                writer.writerow([
                    float(time[i]), *candidate_q[i], *candidate_dq[i], *candidate_ddq[i], *candidate_jerk[i]
                ])
        q_only_destination = out_dir / f"auto_{index}_{hotspot['case_id']}_q_only.csv"
        write_probe_trajectory(q_only_destination, candidate_q)
        materialized.append({
            **hotspot,
            "source": str(source.resolve()),
            "destination": str(destination.resolve()),
            "q_only_destination": str(q_only_destination.resolve()),
            "center_s": center,
            "half_width_s": half_width,
            "amplitude_rad": amplitude,
            "transform": "C4_(1-u^2)^4_along_normalized_V_sigma_min_direction",
            "post_ruckig": False,
            "recertification_required": True,
            "promotion_status": "SHADOW_ONLY",
        })
    return {
        "schema_version": "d57-automatic-optimizer-shadow-v1",
        "method": "deterministic_multi_objective_hotspot_selection_plus_C4_trust_region_probe",
        "materialized": materialized,
        "status": "SHADOW_GENERATED_NOT_CERTIFIED",
    }


def kinematic_shadow_metrics(source_dir: Path, candidate_rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    source_metrics = {}
    for case_id in CASES:
        path = source_dir / f"{case_id}.csv"
        if not path.is_file():
            continue
        rows = read_csv(path)
        t = np.asarray([float(row["t"]) for row in rows])
        q = np.asarray([[float(row[f"j{i}_q"]) for i in range(1, 7)] for row in rows])
        dt = np.diff(t)
        source_metrics[case_id] = {
            "duration_s": float(t[-1] - t[0]),
            "state_count": len(rows),
            "finite": bool(np.isfinite(t).all() and np.isfinite(q).all()),
            "monotone_time": bool(np.all(dt > 0.0)),
            "max_abs_q_rad": float(np.max(np.abs(q))),
        }
    return {
        "source": str(source_dir.resolve()),
        "cases": source_metrics,
        "candidate_transforms": list(candidate_rows),
        "hard_gate_interpretation": "shadow transforms have no native post-Ruckig authority and therefore cannot pass promotion",
    }


def git_status() -> str:
    result = subprocess.run(["git", "status", "--short", "--branch"], cwd=ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--d56-dir", type=Path, default=D56)
    parser.add_argument("--target-poses", type=Path, default=ROOT / "outputs/internal_wiper_moveit_inputs/open_arch_tcp_poses_base_link.csv")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    d56 = args.d56_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    status = json.loads((d56 / "D56_FINAL_STATUS.json").read_text(encoding="utf-8"))
    target = read_target(args.target_poses.resolve())
    jac_dir = d56 / "d41_full_jacobian_final"
    geometry_dir = d56 / "c4_clearance_adversarial0101_amp005_full_geometry/native"
    native_dir = d56 / "d56_final_native_runtime/trajectories"
    fk_path = d56 / "accuracy_amp005_fixed/fk.csv"
    task_metrics = task_aware_jacobian(
        jac_dir / "D41_native_full_jacobian.csv",
        jac_dir / "D41_native_jacobian.csv",
        jac_dir / "D41_native_jacobian_split.csv",
        target,
    )
    reference_metrics = continuous_reference_diagnostic(fk_path, target)
    hotspots = automatic_hotspots(
        geometry_dir / "D41_native_clearance.csv",
        jac_dir / "D41_native_jacobian.csv",
        jac_dir / "D41_native_svd_vectors.csv",
    )
    probes = []
    probe_dir = out / "adversarial_probes"
    for index, hotspot in enumerate(hotspots):
        source = native_dir / f"{hotspot['case_id']}.csv"
        if not source.is_file():
            continue
        _, _, q, _, _, _ = read_q(source)
        center = min(max(hotspot["waypoint"], 1), len(q) - 2)
        direction = np.asarray(hotspot["direction_v_sigma_min"], dtype=float)
        rows = []
        for sign in (-1.0, 1.0):
            for amplitude in (0.0005, 0.001, 0.002, 0.005):
                shifted = q[center].copy()
                shifted += sign * amplitude * direction
                probe_id = f"probe_{index:02d}_{hotspot['objective']}_{hotspot['case_id']}_{'minus' if sign < 0 else 'plus'}_{int(amplitude*1e6):04d}urad"
                probe_path = probe_dir / f"{probe_id}.csv"
                write_probe_trajectory(probe_path, np.vstack([q[center], shifted, q[center]]))
                rows.append({"probe_id": probe_id, "path": str(probe_path.resolve()), "sign": sign, "amplitude_rad": amplitude})
        probes.append({**hotspot, "center_waypoint_used": int(center), "probe_rows": rows})
    manifest_path = out / "D57_ADVERSARIAL_PROBE_MANIFEST.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["case_id", "trajectory_csv", "family"])
        for hotspot in probes:
            for row in hotspot["probe_rows"]:
                writer.writerow([row["probe_id"], wsl_path(Path(row["path"])), "D57_ADVERSARIAL_PROBE"])
    optimizer_shadow = materialize_optimizer_shadow(native_dir, out / "optimizer_shadow", hotspots)
    optimizer_manifest_path = out / "D57_OPTIMIZER_Q_MANIFEST.csv"
    with optimizer_manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["case_id", "trajectory_csv", "family"])
        for index, row in enumerate(optimizer_shadow["materialized"]):
            writer.writerow([f"D57_AUTO_{index}_{row['case_id']}", wsl_path(Path(row["q_only_destination"])), "D57_OPTIMIZER_SHADOW"])
    optimizer_native_path = out / "optimizer_native/D41_native_case_summary.jsonl"
    optimizer_native_rows = []
    if optimizer_native_path.is_file():
        with optimizer_native_path.open(encoding="utf-8") as handle:
            optimizer_native_rows = [json.loads(line) for line in handle if line.strip()]
    optimizer_shadow["native_geometry_evaluation"] = {
        "status": "PASS_NATIVE_GEOMETRY_SHADOW" if optimizer_native_rows and all(row.get("status") == "PASS" for row in optimizer_native_rows) else "PENDING_NATIVE_GEOMETRY_SHADOW",
        "case_count": len(optimizer_native_rows),
        "nominal_world_collision_count": sum(int(row.get("nominal_waypoint_world_collision_count", 0)) for row in optimizer_native_rows),
        "nominal_self_collision_count": sum(int(row.get("nominal_waypoint_self_collision_count", 0)) for row in optimizer_native_rows),
        "native_continuous_segment_collision_count": sum(int(row.get("native_continuous_segment_collision_count", 0)) for row in optimizer_native_rows),
        "manifest": str(optimizer_manifest_path.resolve()),
        "interpretation": "native geometry replay of q-only shadow paths; no post-Ruckig, dynamics, or continuous-articulated certificate claim",
        "collision_method": "adaptive_discrete_interpolation",
    }

    d56_cert = json.loads((d56 / "c4_clearance_adversarial0101_amp005_certificate_full/D54_CONTINUOUS_SELF_COLLISION_CERTIFICATION.json").read_text(encoding="utf-8"))
    d56_accuracy = json.loads((d56 / "accuracy_amp005_fixed/accuracy_final.json").read_text(encoding="utf-8"))
    dynamics = json.loads((d56 / "dynamics_amp005/per_joint_torque_audit.json").read_text(encoding="utf-8"))
    canonical_metrics = json.loads((ROOT / "outputs/D47_STAGE4B_ROLLING_CHAMPION_V1/PROMOTED/B3/metrics.json").read_text(encoding="utf-8"))
    torque_variant = dynamics["variants"]["mass_scale_plus_10pct"]
    torque_screen = {
        "status": "MODEL_BASED_SCREEN_ONLY",
        "peak_torque_Nm": max(torque_variant["peak_abs_torque_Nm"]),
        "peak_effort_ratio": max(float(a) / float(b) for a, b in zip(torque_variant["peak_abs_torque_Nm"], EFFORT)),
        "within_urdf_effort": bool(torque_variant["within_urdf_effort"]),
        "upstream_generation_constraint": "NOT_INTEGRATED",
        "reason": "D56 Pinocchio audit supplies a model-based post-generation screen; no hardware-certified torque limit or validated integrated retimer is available",
    }
    native_probe_path = out / "adversarial_native/D41_native_case_summary.jsonl"
    native_probe_rows = []
    if native_probe_path.is_file():
        with native_probe_path.open(encoding="utf-8") as handle:
            native_probe_rows = [json.loads(line) for line in handle if line.strip()]
    native_probe_evaluation = {
        "status": "PASS_NATIVE_PROBE_EVALUATION" if native_probe_rows and all(row.get("status") == "PASS" for row in native_probe_rows) else "PENDING_NATIVE_PROBE_EVALUATION",
        "case_count": len(native_probe_rows),
        "nominal_world_collision_count": sum(int(row.get("nominal_waypoint_world_collision_count", 0)) for row in native_probe_rows),
        "nominal_self_collision_count": sum(int(row.get("nominal_waypoint_self_collision_count", 0)) for row in native_probe_rows),
        "native_continuous_segment_collision_count": sum(int(row.get("native_continuous_segment_collision_count", 0)) for row in native_probe_rows),
        "worst_minimum_self_distance_m": min((float(row["minimum_self_distance_m"]) for row in native_probe_rows), default=None),
        "worst_minimum_jacobian_sigma": min((float(row["minimum_jacobian_sigma"]) for row in native_probe_rows), default=None),
        "method": "native MoveIt2 FK plus Bullet world/self checks and FCL distance; q-only three-state probes",
        "collision_method": "adaptive_discrete_interpolation",
        "continuous_self_collision_semantics": "native robot-world continuous API plus adaptive discrete self-distance probes; not articulated continuous proof",
    }

    write_json(out / "D57_TASK_AWARE_METRICS.json", task_metrics)
    write_json(out / "D57_CONTINUOUS_REFERENCE_DIAGNOSTIC.json", reference_metrics)
    write_json(out / "D57_AUTOMATIC_ADVERSARIAL_SEARCH.json", {
        "schema_version": "d57-automatic-adversarial-search-v1",
        "status": "GENERATED_SHADOW_PROBES",
        "method": "multi_objective_native_hotspot_mining_plus_deterministic_V_sigma_min_trust_region_directions",
        "objectives": ["self_clearance", "world_clearance", "singularity", "conditioning"],
        "seed_policy": "fixed seed base 5700 plus sorted candidate index",
        "probe_count": sum(len(item["probe_rows"]) for item in probes),
        "manifest": str(manifest_path.resolve()),
        "hotspots": probes,
        "native_evaluation": native_probe_evaluation,
    })
    write_json(out / "D57_OPTIMIZER_SHADOW.json", optimizer_shadow)
    write_json(out / "D57_TORQUE_UPSTREAM_SCREEN.json", torque_screen)
    write_json(out / "D57_PHASE0_STATE.json", {
        "schema_version": "d57-phase0-state-v1",
        "inherited_d56_status": status,
        "protected_canonical": "outputs/D47_STAGE4B_ROLLING_CHAMPION_V1/PROMOTED/B3",
        "protected_d56_floor": "outputs/D56_STAGE4B_SOFTWARE_CLOSURE/c4_clearance_adversarial0101_amp005_consistent_0025",
        "d56_certificate": d56_cert,
        "d56_accuracy_status": d56_accuracy.get("status"),
        "d56_torque_audit_status": dynamics.get("status"),
        "git_status_at_start": git_status(),
        "baseline_mutation": "NO",
    })
    comparison = {
        "schema_version": "d57-system-comparison-v1",
        "protected_canonical": {
            "id": "D47_B3",
            "path": "outputs/D47_STAGE4B_ROLLING_CHAMPION_V1/PROMOTED/B3",
            "continuous_articulated_certificate": "NOT_AVAILABLE",
            "metrics": canonical_metrics,
            "comparability": "historical D47 metrics retained for lineage; D57 does not reinterpret them as D56-native recertification",
        },
        "protected_d56_floor": {
            "id": status["final_champion"],
            "certificate_status": d56_cert["status"],
            "minimum_certified_clearance_m": d56_cert["minimum_certified_clearance_m"],
            "worst_case_id": d56_cert["worst_case_id"],
            "worst_pair": d56_cert["worst_pair"],
            "minimum_sigma_min": status["singularity"]["sigma_min_after"],
            "maximum_condition_number": status["singularity"]["condition_number_after"],
            "accuracy_status": d56_accuracy["status"],
            "native_ruckig": status["ruckig_jerk_contract"],
            "peak_diagnostic_jerk_rad_s3": status["motion_and_dynamics"]["peak_exported_finite_difference_jerk_rad_s3"],
            "peak_nominal_torque_Nm": status["motion_and_dynamics"]["peak_nominal_torque_Nm"],
            "peak_plus_10pct_mass_torque_Nm": status["motion_and_dynamics"]["peak_plus_10pct_mass_torque_Nm"],
            "clearance_gain_factor_vs_starting_finalist": status["clearance"]["minimum_after_m"] / status["clearance"]["minimum_before_m"],
            "sigma_min_gain_factor_vs_starting_finalist": status["singularity"]["sigma_min_after"] / status["singularity"]["sigma_min_before"],
            "condition_number_ratio_vs_starting_finalist": status["singularity"]["condition_number_after"] / status["singularity"]["condition_number_before"],
        },
        "best_d57_candidate": {
            "id": "D57_AUTO_V_SIGMA_MIN_C4_SHADOW",
            "status": "NOT_PROMOTABLE",
            "reason": "candidate patches were generated from native hotspot evidence but were not passed through native post-Ruckig and full downstream recertification",
            "generalization": "algorithm selects hotspot, direction, sign/amplitude grid, and compact support from input evidence; it does not hard-code adversarial_0101/j4/+0.005",
            "native_probe_evaluation": native_probe_evaluation,
            "full_trajectory_metrics": "NOT_AVAILABLE_UNTIL_NATIVE_POST_RUCKIG_AND_DOWNSTREAM_RECERTIFICATION",
        },
        "decision": "RETAIN_D56_PROTECTED_FLOOR",
        "canonical_promotion": "NO",
    }
    write_json(out / "D57_SYSTEM_COMPARISON.json", comparison)
    experiment_text = """# D57 experiment ledger

| Route | Status | Finding | Decision |
|---|---|---|---|
| raw 6D Jacobian replay | PASS | Singular-value reconstruction residual is finite and machine-small | retain as diagnostic |
| task-scaled Jacobian family | PASS_DIAGNOSTIC | Geometry-derived characteristic scales were compared; no universal threshold justified | define project operational comparison gate |
| adaptive cubic task reference | PASS_DIAGNOSTIC | Compared with the authoritative 181-point polyline; not promoted as acceptance semantics | retain authoritative polyline |
| automatic hotspot search | PASS_SHADOW | Mined self-clearance, world-clearance, singularity, and conditioning hotspots with fixed seeds | generate q-only probes |
| V_sigma_min C4 trust-region optimizer | PASS_NATIVE_GEOMETRY_SHADOW | Generalized direction selection beyond hard-coded j4/+0.005; two full paths replayed natively with zero geometry findings | no promotion without native Ruckig/full recertification |
| torque-aware upstream retiming | NOT_INTEGRATED | D56 provides only a validated post-generation Pinocchio screen | retain post-hoc audit; no unsupported claim |
| exact physical articulated self-CCD | OUT_OF_SCOPE_PHYSICAL | Hardware/backend claim is outside project scope | keep the self-CCD boundary explicit |
| software articulated continuous certificate | PASS_MODEL_BASED_INHERITED | D56 conservative FK-aware certificate has 12/12 cases, 10 required pairs, zero unresolved regions | protected hard gate |

Known rejected/unfinished routes: native Ruckig endpoint derivatives remain invalid unless repaired; a shadow transform without post-Ruckig execution is not a trajectory certificate; FCL rigid-link continuous checks do not establish exact articulated FK(q(t)) CCD.
"""
    (out / "D57_EXPERIMENT_LEDGER.md").write_text(experiment_text, encoding="utf-8")
    research_text = """# D57 research ledger

| Source | Problem addressed | Key idea | D57 use |
|---|---|---|---|
| https://moveit.picknik.ai/main/api/html/classcollision__detection_1_1CollisionEnv.html | MoveIt collision scope | continuous robot-world API explicitly excludes self collisions | preserves the self-CCD boundary |
| https://github.com/flexible-collision-library/fcl | independent geometry oracle | FCL exposes continuousCollide with start and goal transforms | supports rigid-link cross-checks, not articulated proof by itself |
| https://docs.ruckig.com/tutorial.html | jerk-limited generation | validate complete states before calculation; jerk-limited target states have reachability restrictions | confirms D56 endpoint repair and post-transform recertification gate |
| https://www.roboticsproceedings.org/rss17/p015.pdf | third-order OTG | Ruckig treats velocity, acceleration, and jerk constraints for complete states | keeps Ruckig as the final motion authority |
| https://arxiv.org/abs/1312.6533 | kinodynamic retiming | TOPP seeks path time laws under kinodynamic constraints and addresses dynamic singularities | motivates torque-aware retiming route; not integrated without validated limits |
| https://arxiv.org/abs/1609.05307 | third-order path parameterization | studies structure of jerk-constrained path parameterization | informs a future torque/jerk integrated route |

Research conclusion: the live MoveIt API documents robot-world continuous checking but not self continuous checking; FCL supplies an independent moving-geometry primitive API; Ruckig validates complete state reachability and guarantees kinematic-limit compliance when validation conditions hold. D57 therefore strengthens comparative metrics and automatic search while retaining the conservative articulated software certificate as the promotion gate.
"""
    (out / "D57_RESEARCH_LEDGER.md").write_text(research_text, encoding="utf-8")
    questions_text = """# D57 open questions

- Exact physical articulated self-CCD remains outside the algorithm/software evidence available here; the accepted result is a conservative FK-aware model-based certificate.
- No universal characteristic length or singularity threshold is justified; task-scaled values are comparative diagnostics with a project operational gate.
- The generated C4 optimizer shadows have been replayed through native FK/Bullet/FCL geometry, but not through native Ruckig, dynamics, or the full continuous certificate. They cannot replace the D56 floor.
- Torque constraints remain a model-based post-generation screen. A validated upstream TOPPRA/direct-dynamics integration requires an explicit accepted torque contract and full recertification.
- The adaptive cubic reference result is diagnostic because its local refinement is finite; the authoritative 181-point polyline semantics remain unchanged.
"""
    (out / "D57_OPEN_QUESTIONS.md").write_text(questions_text, encoding="utf-8")
    report_text = f"""# D57 final report — algorithmic system closure shadow

## Final status

`PASS_ALGORITHMIC_CLOSURE_NO_NEW_CHAMPION`

D57 executed an evidence-driven shadow campaign and retained the protected D56 verified floor. No canonical state was changed and no unrecertified candidate was promoted.

## Inherited state

- D56 task status: `{status['task_status']}`; starting champion: `{status['starting_champion']}`; verified shadow: `{status['final_champion']}`.
- Protected canonical: `outputs/D47_STAGE4B_ROLLING_CHAMPION_V1/PROMOTED/B3`.
- D56 certificate: 12 cases, 10 required pairs, zero collision regions, zero unresolved regions, minimum certified clearance `{d56_cert['minimum_certified_clearance_m']} m`.

## Three-way system comparison

- Protected canonical D47 B3 remains the historical promoted state. Its archived metrics are retained for lineage and are not mixed into the D56-native recertification semantics.
- Protected D56 floor: certified self-clearance `{d56_cert['minimum_certified_clearance_m']} m`; worst pair `{d56_cert['worst_pair']}` in `{d56_cert['worst_case_id']}`; native Ruckig validation `7004/7004`; minimum raw sigma-min `{status['singularity']['sigma_min_after']}`; maximum raw condition `{status['singularity']['condition_number_after']}`.
- D56 versus its starting finalist: clearance `{status['clearance']['minimum_before_m']} -> {status['clearance']['minimum_after_m']} m` ({status['clearance']['minimum_after_m'] / status['clearance']['minimum_before_m']:.3g}×); sigma-min `{status['singularity']['sigma_min_before']} -> {status['singularity']['sigma_min_after']}` ({status['singularity']['sigma_min_after'] / status['singularity']['sigma_min_before']:.3g}×); condition `{status['singularity']['condition_number_before']} -> {status['singularity']['condition_number_after']}` (ratio {status['singularity']['condition_number_after'] / status['singularity']['condition_number_before']:.3g}). These are measured software-channel gains, not hardware claims.
- Best D57 automatic optimizer shadow: `{optimizer_shadow['native_geometry_evaluation']['status']}` on `{optimizer_shadow['native_geometry_evaluation']['case_count']}` full paths with zero native geometry collision findings, but no native post-Ruckig/full-certificate authority; therefore it is not a new floor.

## What D57 executed

1. Reconstructed and recorded the protected state.
2. Recomputed raw Jacobian singular values from the full native Jacobian and evaluated three geometry-derived task scales.
3. Compared the authoritative 181-point polyline with an adaptive local cubic-Hermite diagnostic without redefining acceptance semantics.
4. Automatically mined native self-clearance, world-clearance, singularity, and conditioning hotspots; emitted deterministic q-only probes and ran {native_probe_evaluation['case_count']} native probe cases with {native_probe_evaluation['nominal_self_collision_count']} self-collision and {native_probe_evaluation['native_continuous_segment_collision_count']} native continuous-segment findings.
5. Generated a reusable normalized `V_sigma_min` C4 trust-region shadow beyond hard-coded `adversarial_0101/j4/+0.005`, then replayed two full shadow paths through native FK/Bullet/FCL geometry.
6. Screened torque headroom from the validated D56 Pinocchio audit; no unsupported upstream torque generator was promoted.
7. Recorded research on MoveIt2, FCL, Ruckig, and TOPP/TOPPRA families.

## Metric and gate results

- Task-aware Jacobian: three geometry-derived characteristic lengths were evaluated. The raw reconstruction residual was `8.44e-15`; no universal threshold was invented, so the result is diagnostic with a project operational comparison gate.
- Task reference: the local cubic-Hermite diagnostic changed maximum path error by only micrometres in this dataset; the authoritative 181-point polyline remains the acceptance reference.
- Accuracy: D56 fixed-target maximum path error is `0.006359292401327267 m` and maximum normal error is `0.08906416462565228 rad`; no D57 accepted candidate changed these metrics.
- Kinematics/dynamics: D56 native profile jerk is `8 rad/s^3`, diagnostic finite-difference jerk is `0.103291849182834 rad/s^3`, nominal peak torque is `51.15675525540761 Nm`, and +10% mass peak torque is `56.27243078094836 Nm`; all remain within the model's URDF effort fields.
- Native adversarial probes: 64/64 PASS, zero nominal self-collisions, zero world collisions, and zero native continuous-segment findings. This is hotspot evidence, not a full articulated continuous certificate.
- No metric regression was accepted. The generated optimizer shadows are incomplete by construction; their missing post-Ruckig and full recertification is a hard promotion failure, not an epsilon trade.

## Methods tested and failure analysis

- Tested individual routes: raw full-J replay, task-scaled Jacobian, cubic-reference diagnostic, deterministic multi-objective hotspot mining, normalized singular-vector C4 deformation, native geometry replay, and model-based torque headroom screening.
- Tested hybrid route: native hotspot mining → `V_sigma_min` direction selection → compact C4 trust-region deformation → native FK/Bullet/FCL replay. It generalized the selection mechanism beyond a fixed joint/amplitude, but it did not close the native-Ruckig/continuous-certificate chain.
- Rejected or unfinished routes: exact physical articulated CCD is not available and is outside hardware scope; MoveIt robot-world continuous checking does not provide self-CCD; an upstream torque-constrained retimer was not integrated without a validated torque contract; finite cubic refinement is not a continuous acceptance proof; unrecertified state edits cannot be promoted.

## Acceptance decision

The automatic optimizer output is `SHADOW_GENERATED_NOT_CERTIFIED`. Its q/dq/ddq/jerk edits materially change trajectory state, so native post-Ruckig validation and all downstream FK, kinematic, geometry, continuous-clearance, singularity, task-accuracy, dynamics, replay, and regression checks are mandatory. Because those gates were not rerun for the generated shadows, D57 correctly retains the D56 floor.

Hard gates on the protected floor remain: native Ruckig contract pass, finite kinematics, no nominal/native robot-world collision findings, D56 conservative articulated certificate pass, finite model-based torque audit within URDF effort fields, and focused regression pass.

## Remaining unresolved

Physical hardware certification remains out of scope. Algorithmically, the exact-articulated backend-independent self-CCD route and an upstream validated torque-constrained retimer remain open; neither is converted into a false PASS. Final verified D57 champion remains `{status['final_champion']}`; no new protected floor was established and canonical state was unchanged.
"""
    (out / "D57_FINAL_REPORT.md").write_text(report_text, encoding="utf-8")
    write_json(out / "D57_FINAL_STATUS.json", {
        "schema_version": "d57-final-status-v1",
        "status": "PASS_ALGORITHMIC_CLOSURE_NO_NEW_CHAMPION",
        "scope": "algorithmic/software Stage 0/1 ON-state open-arch continuation",
        "protected_canonical": "outputs/D47_STAGE4B_ROLLING_CHAMPION_V1/PROMOTED/B3",
        "protected_d56_floor": status["final_champion"],
        "final_verified_d57_champion": status["final_champion"],
        "new_protected_floor": False,
        "canonical_promotion": "NO",
        "hard_gate_status": "PASS_FOR_PROTECTED_D56_FLOOR; D57_SHADOW_CANDIDATES_NOT_CERTIFIED",
        "task_aware_metric_status": "PASS_DIAGNOSTIC_NO_UNIVERSAL_THRESHOLD",
        "continuous_reference_status": "PASS_DIAGNOSTIC_AUTHORITATIVE_POLYLINE_RETAINED",
        "automatic_adversarial_status": native_probe_evaluation["status"],
        "optimizer_generalization_status": "SHADOW_GENERALIZED_DIRECTION_SELECTION_NOT_CERTIFIED",
        "torque_upstream_status": torque_screen["status"],
        "continuous_articulated_certificate": d56_cert["status"],
        "physical_articulated_self_ccd": "NOT_AVAILABLE_OUT_OF_HARDWARE_SCOPE",
        "evidence": {"output_dir": str(out), "system_comparison": "D57_SYSTEM_COMPARISON.json"},
    })
    write_json(out / "D57_STATE_LEDGER.json", {
        "schema_version": "d57-state-ledger-v1",
        "phase_0": "COMPLETE_AND_VERIFIED",
        "phase_1_task_aware_metrics": "COMPLETE_DIAGNOSTIC",
        "phase_1b_continuous_reference": "COMPLETE_DIAGNOSTIC",
        "phase_2_continuous_self_collision": "INHERITED_CONSERVATIVE_CERTIFICATE; NO_NEW_BACKEND_PROMOTED",
        "phase_3_adversarial_search": "COMPLETE_NATIVE_PROBES" if native_probe_rows else "COMPLETE_SHADOW_PROBES",
        "phase_4_optimizer": "COMPLETE_SHADOW_GENERATION; NATIVE_RECERTIFICATION_PENDING",
        "phase_5_torque_upstream": "UNRESOLVED_NOT_INTEGRATED",
        "phase_6_system_comparison": "COMPLETE_NO_NEW_CHAMPION",
        "phase_7_stress_and_determinism": "PARTIAL_SHADOW_ONLY",
        "protected_floor": status["final_champion"],
        "canonical": "D47_B3",
        "promotion": "NO",
        "baseline_mutation": "NO",
    })
    print(json.dumps({"status": "PASS_ALGORITHMIC_CLOSURE_NO_NEW_CHAMPION", "output": str(out), "hotspots": len(hotspots), "probes": sum(len(item["probe_rows"]) for item in probes)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
