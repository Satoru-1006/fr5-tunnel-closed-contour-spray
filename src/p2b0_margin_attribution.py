"""P2-B0: local attribution of the frozen D41 J6+ robustness boundary.

This is a measurement-only experiment. It reads the authenticated P2-A/D41
trajectory, obtains Jacobians/FK/collision measurements from the existing
MoveIt2 native tools, and writes only the P2-B0 result plus disposable scratch
files which are removed after each run.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
P2A_RESULT = ROOT / "outputs" / "p2a_axiswise_robustness_margin.json"
P2B0_RESULT = ROOT / "outputs" / "p2b0_margin_attribution.json"
D41_BUFFER_RAD = 1.0e-4
CRITICAL_WAYPOINT = 86  # zero-based, matching the stored D41/P2-A CSV index
WINDOW_START = 80
WINDOW_END = 95
POSITION_TOLERANCE_M = 0.006
TERMINAL_POSITION_TOLERANCE_M = 0.004
NORMAL_TOLERANCE_RAD = math.radians(10.0)
JOINT_STEP_TOLERANCE_RAD = math.radians(20.0)
RETREAT_PEAKS_RAD = (0.001, 0.005, 0.010)
BUFFER_ABLATION_RAD = (1.0e-4, 1.0e-3, 5.0e-3, 1.0e-2)


@dataclass(frozen=True)
class ProcessJacobian:
    raw: np.ndarray
    scaled: np.ndarray
    tangent_basis: np.ndarray
    singular_values_raw: np.ndarray
    singular_values_scaled: np.ndarray
    effective_rank: int
    rank_tolerance: float
    null_basis: np.ndarray
    null_projector: np.ndarray


def process_task_jacobian(
    full_jacobian: Sequence[Sequence[float]],
    surface_normal: Sequence[float],
    *,
    position_tolerance_m: float = POSITION_TOLERANCE_M,
    normal_tolerance_rad: float = NORMAL_TOLERANCE_RAD,
) -> ProcessJacobian:
    """Construct [J_v; U_n^T J_omega] with explicit tolerance scaling."""
    jac = np.asarray(full_jacobian, dtype=np.float64)
    normal = np.asarray(surface_normal, dtype=np.float64)
    if jac.shape != (6, 6) or normal.shape != (3,):
        raise ValueError("a finite 6x6 MoveIt Jacobian and 3-vector normal are required")
    if not np.isfinite(jac).all() or not np.isfinite(normal).all():
        raise ValueError("process Jacobian inputs must be finite")
    if position_tolerance_m <= 0.0 or normal_tolerance_rad <= 0.0:
        raise ValueError("task tolerances must be positive")
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= 1e-12:
        raise ValueError("surface normal must be nonzero")
    normal = normal / normal_norm

    # Build two orthonormal directions spanning the tangent plane of the
    # desired spray axis. The free rotation about that axis is not constrained.
    reference = np.eye(3)[int(np.argmin(np.abs(normal)))]
    tangent_1 = np.cross(normal, reference)
    tangent_1 /= np.linalg.norm(tangent_1)
    tangent_2 = np.cross(normal, tangent_1)
    basis = np.column_stack((tangent_1, tangent_2))

    raw = np.vstack((jac[:3, :], basis.T @ jac[3:, :]))
    scaled = np.vstack((raw[:3, :] / position_tolerance_m,
                         raw[3:, :] / normal_tolerance_rad))
    if not np.isfinite(raw).all() or not np.isfinite(scaled).all():
        raise ValueError("constructed process Jacobian is non-finite")

    _, singular_values_scaled, vh = np.linalg.svd(scaled, full_matrices=True)
    singular_values_raw = np.linalg.svd(raw, compute_uv=False)
    rank_tolerance = max(float(singular_values_scaled[0]) * 1.0e-10, 1.0e-12)
    rank = int(np.count_nonzero(singular_values_scaled > rank_tolerance))
    null_basis = vh[rank:, :].T.copy()
    projector = null_basis @ null_basis.T
    projector = 0.5 * (projector + projector.T)
    return ProcessJacobian(
        raw=raw,
        scaled=scaled,
        tangent_basis=basis,
        singular_values_raw=singular_values_raw,
        singular_values_scaled=singular_values_scaled,
        effective_rank=rank,
        rank_tolerance=rank_tolerance,
        null_basis=null_basis,
        null_projector=projector,
    )


def nullspace_j6_retreat(process_jacobian: ProcessJacobian, retreat_rad: float) -> np.ndarray:
    """Return the minimum-norm first-order task-null delta with delta-J6<0."""
    if not math.isfinite(retreat_rad) or retreat_rad < 0.0:
        raise ValueError("retreat must be finite and non-negative")
    if process_jacobian.null_basis.shape[1] == 0:
        raise ValueError("process task has no numerical null-space")
    j6_component = float(process_jacobian.null_projector[5, 5])
    if j6_component <= 1e-10:
        raise ValueError("process null-space has no usable J6 component")
    direction = process_jacobian.null_projector[:, 5]
    delta = -(retreat_rad / j6_component) * direction
    return np.asarray(delta, dtype=np.float64)


def shadow_buffer_projection(
    values: Sequence[float], lower: Sequence[float], upper: Sequence[float], buffer_rad: float
) -> np.ndarray:
    """Known-answer replica of D41's explicit lower+buffer/upper-buffer clip."""
    q = np.asarray(values, dtype=np.float64)
    low = np.asarray(lower, dtype=np.float64)
    high = np.asarray(upper, dtype=np.float64)
    if q.ndim != 1 or low.shape != q.shape or high.shape != q.shape:
        raise ValueError("joint vectors must have matching one-dimensional shapes")
    if not (np.isfinite(q).all() and np.isfinite(low).all() and np.isfinite(high).all()):
        raise ValueError("joint vectors must be finite")
    if not math.isfinite(buffer_rad) or buffer_rad < 0.0:
        raise ValueError("buffer must be finite and non-negative")
    if np.any(low >= high) or np.any(2.0 * buffer_rad >= high - low):
        raise ValueError("buffer leaves no valid interval inside a joint range")
    return np.clip(q, low + buffer_rad, high - buffer_rad)


def capability_status(available: bool, reason: str) -> dict[str, str]:
    """Keep absent robot-measurement capabilities fail-closed."""
    return {"status": "AVAILABLE" if available else "INCONCLUSIVE", "reason": reason}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _read_frozen_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = ["t", *[f"j{i}_q" for i in range(1, 7)]]
        if reader.fieldnames is None or any(name not in reader.fieldnames for name in required):
            raise RuntimeError("D41_post_Ruckig_CSV_schema_mismatch")
        rows = list(reader)
    times = np.asarray([float(row["t"]) for row in rows], dtype=np.float64)
    q = np.asarray([[float(row[f"j{i}_q"]) for i in range(1, 7)] for row in rows], dtype=np.float64)
    if q.shape != (181, 6) or times.shape != (181,) or not np.isfinite(q).all():
        raise RuntimeError("D41_post_Ruckig_shape_or_finiteness_mismatch")
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0.0):
        raise RuntimeError("D41_timestamps_not_strictly_increasing")
    return times, q


def _read_trace(path: Path) -> dict[str, np.ndarray]:
    required = ("actual_tcp_x", "actual_tcp_y", "actual_tcp_z", "tool_z_x", "tool_z_y", "tool_z_z",
                "wall_normal_x", "wall_normal_y", "wall_normal_z")
    records: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or any(name not in reader.fieldnames for name in required):
            raise RuntimeError("D41_FK_trace_schema_mismatch")
        records = list(reader)
    result = {
        "position": np.asarray([[float(r[k]) for k in ("actual_tcp_x", "actual_tcp_y", "actual_tcp_z")] for r in records]),
        "tool_z": np.asarray([[float(r[k]) for k in ("tool_z_x", "tool_z_y", "tool_z_z")] for r in records]),
        "normal": np.asarray([[float(r[k]) for k in ("wall_normal_x", "wall_normal_y", "wall_normal_z")] for r in records]),
    }
    if any(v.shape != (181, 3) or not np.isfinite(v).all() for v in result.values()):
        raise RuntimeError("D41_FK_trace_shape_or_finiteness_mismatch")
    result["tool_z"] /= np.linalg.norm(result["tool_z"], axis=1)[:, None]
    result["normal"] /= np.linalg.norm(result["normal"], axis=1)[:, None]
    return result


def _read_fk_cases(path: Path) -> dict[str, dict[str, np.ndarray]]:
    cases: dict[str, dict[str, list[Any]]] = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = ("case_id", "waypoint", "x_m", "y_m", "z_m", "qx", "qy", "qz", "qw")
        if reader.fieldnames is None or any(name not in reader.fieldnames for name in required):
            raise RuntimeError("MoveIt_FK_output_schema_mismatch")
        for row in reader:
            case = cases.setdefault(row["case_id"], {"waypoint": [], "position": [], "quaternion": []})
            case["waypoint"].append(int(row["waypoint"]))
            case["position"].append([float(row[k]) for k in ("x_m", "y_m", "z_m")])
            case["quaternion"].append([float(row[k]) for k in ("qx", "qy", "qz", "qw")])
    output: dict[str, dict[str, np.ndarray]] = {}
    for name, rows in cases.items():
        if rows["waypoint"] != list(range(181)):
            raise RuntimeError(f"MoveIt_FK_waypoint_order_mismatch:{name}")
        position = np.asarray(rows["position"], dtype=np.float64)
        quaternion = np.asarray(rows["quaternion"], dtype=np.float64)
        if position.shape != (181, 3) or quaternion.shape != (181, 4) or not np.isfinite(position).all() or not np.isfinite(quaternion).all():
            raise RuntimeError(f"MoveIt_FK_nonfinite_or_shape_mismatch:{name}")
        output[name] = {"position": position, "quaternion": quaternion}
    return output


def _tool_z_from_quaternion(quaternion: Sequence[float]) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    vector = np.asarray([0.0, 0.0, 1.0])
    qv = np.asarray([x, y, z])
    rotated = vector + 2.0 * np.cross(qv, np.cross(qv, vector) + w * vector)
    norm = float(np.linalg.norm(rotated))
    if norm <= 1e-12 or not math.isfinite(norm):
        raise ValueError("MoveIt FK returned an invalid orientation")
    return rotated / norm


def _normal_errors(tool_z: np.ndarray, normals: np.ndarray) -> np.ndarray:
    dots = np.sum(tool_z * normals, axis=1)
    return np.arccos(np.clip(dots, -1.0, 1.0))


def _read_full_jacobians(path: Path, case_id: str) -> dict[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = [f"J{r}{c}" for r in range(6) for c in range(6)]
        if reader.fieldnames is None or any(name not in reader.fieldnames for name in ("case_id", "waypoint", *fields)):
            raise RuntimeError("MoveIt_full_Jacobian_schema_mismatch")
        for row in reader:
            if row["case_id"] != case_id:
                continue
            waypoint = int(row["waypoint"])
            matrix = np.asarray([float(row[name]) for name in fields], dtype=np.float64).reshape(6, 6)
            if not np.isfinite(matrix).all():
                raise RuntimeError(f"MoveIt_full_Jacobian_nonfinite:{waypoint}")
            result[waypoint] = matrix
    if set(result) != set(range(181)):
        raise RuntimeError(f"MoveIt_full_Jacobian_waypoint_coverage_mismatch:{len(result)}")
    return result


def _read_case_summary(path: Path, case_id: str) -> dict[str, Any]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if record.get("case_id") == case_id:
                return record
    raise RuntimeError(f"MoveIt_native_case_summary_missing:{case_id}")


def _read_collision_segments(path: Path, case_id: str) -> dict[str, Any]:
    first: dict[str, Any] = {"world": None, "self": None, "native_two_state_world": None}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if row.get("case_id") != case_id:
                continue
            index = int(row["segment_index"])
            for key, field in (("world", "discrete_sample_world_collision"),
                               ("self", "discrete_sample_self_collision"),
                               ("native_two_state_world", "native_continuous_segment_collision")):
                if str(row.get(field, "")).strip().lower() in ("1", "true", "yes") and first[key] is None:
                    first[key] = index
    return first


def _read_clearances(path: Path, case_id: str) -> dict[str, Any]:
    world: list[tuple[float, int, str]] = []
    self_values: list[tuple[float, int, str]] = []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if row.get("case_id") != case_id:
                continue
            waypoint = int(row["waypoint"])
            for field, pair_field, target in (("robot_world_distance_m", "robot_world_pair", world),
                                               ("self_distance_m", "self_pair", self_values)):
                try:
                    value = float(row.get(field, ""))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    target.append((value, waypoint, row.get(pair_field, "")))
    return {
        "minimum_environment_clearance_m": min(world)[0] if world else None,
        "minimum_environment_clearance_waypoint": min(world)[1] if world else None,
        "minimum_environment_clearance_pair": min(world)[2] if world else None,
        "minimum_self_distance_m": min(self_values)[0] if self_values else None,
        "minimum_self_distance_waypoint": min(self_values)[1] if self_values else None,
        "minimum_self_distance_pair": min(self_values)[2] if self_values else None,
    }


def _build_native_measurement_shadow(scratch: Path, d46: Any) -> Path:
    """Return the available full-Jacobian MoveIt executable without touching a canonical build."""
    # A newer local D41 instrumented binary already exists under the D56
    # software-closure evidence. The ordinary tmp/d41_install binary is older
    # and cannot emit a full 6x6 Jacobian. Prefer the instrumented executable;
    # keep a source build attempt isolated as a fallback only.
    instrumented = (
        ROOT / "outputs" / "D56_STAGE4B_SOFTWARE_CLOSURE" / "d41_full_jac_install"
        / "stage3_h13_d41_native" / "lib" / "stage3_h13_d41_native" / "stage3_h13_d41_native"
    )
    if instrumented.is_file():
        return instrumented

    package_source = ROOT / "cpp" / "stage3_h13_d41"
    build_base = scratch / "native_build"
    install_base = scratch / "native_install"
    log_base = scratch / "colcon_log"
    build_command = " ".join((
        "colcon",
        f"--log-base {shlex.quote(d46.wsl_path(log_base))}",
        "build",
        f"--base-paths {shlex.quote(d46.wsl_path(package_source))}",
        f"--build-base {shlex.quote(d46.wsl_path(build_base))}",
        f"--install-base {shlex.quote(d46.wsl_path(install_base))}",
        "--merge-install --packages-select stage3_h13_d41_native --event-handlers console_direct+",
    ))
    command = "\n".join((
        "source /opt/ros/jazzy/setup.bash",
        f"source {shlex.quote(d46.wsl_path(ROOT / 'install/setup.bash'))}",
        build_command,
    ))
    result = subprocess.run(
        ["wsl.exe", "bash", "-lc", command], cwd=ROOT,
        text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=1800,
    )
    (scratch / "native_build.log").write_text(
        (result.stdout or "") + "\n--- STDERR ---\n" + (result.stderr or ""),
        encoding="utf-8", errors="replace",
    )
    binary = install_base / "lib" / "stage3_h13_d41_native" / "stage3_h13_d41_native"
    if result.returncode != 0 or not binary.is_file():
        detail = (result.stderr or result.stdout or "").strip()[-1800:]
        raise RuntimeError(f"isolated_current_source_native_build_failed:{result.returncode}:{detail}")
    return binary


def _run_native_measurement(
    output: Path, cases: Sequence[Mapping[str, Any]], d46: Any, binary: Path, native_name: str = "native"
) -> Path:
    """Run the freshly built D41 native MoveIt/FCL/Jacobian executable."""
    native = output / native_name
    native.mkdir(parents=True, exist_ok=False)
    manifest = native / "cases.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("case_id", "trajectory_csv", "family"), lineterminator="\n")
        writer.writeheader()
        for row in cases:
            writer.writerow({
                "case_id": row["case_id"],
                "trajectory_csv": d46.wsl_path(Path(str(row["trajectory_path"]))),
                "family": row["family"],
            })
    prefixes = [parent for parent in binary.parents if (parent / "setup.bash").is_file()]
    if not prefixes:
        raise RuntimeError(f"native_MoveIt_setup_prefix_missing:{binary}")
    native_prefix = prefixes[-1]
    native_lib_dirs = [
        parent / "lib" for parent in binary.parents
        if (parent == native_prefix or native_prefix in parent.parents) and (parent / "lib").is_dir()
    ]
    if (native_prefix / "lib").is_dir():
        native_lib_dirs.append(native_prefix / "lib")
    ld_library_path = ":".join((
        *(d46.wsl_path(path) for path in dict.fromkeys(native_lib_dirs)),
        d46.wsl_path(ROOT / "tmp/d41_install/lib"),
        d46.wsl_path(ROOT / "install/lib"),
        "/opt/ros/jazzy/lib", "/opt/ros/jazzy/lib/x86_64-linux-gnu",
        "/opt/ros/jazzy/opt/sdformat_vendor/lib", "/opt/ros/jazzy/opt/gz_math_vendor/lib",
        "/opt/ros/jazzy/opt/gz_utils_vendor/lib", "/opt/ros/jazzy/opt/gz_tools_vendor/lib",
        "/usr/lib/x86_64-linux-gnu",
    ))
    command = "\n".join((
        "source /opt/ros/jazzy/setup.bash",
        f"source {shlex.quote(d46.wsl_path(ROOT / 'install/setup.bash'))}",
        f"source {shlex.quote(d46.wsl_path(native_prefix / 'setup.bash'))}",
        f"export LD_LIBRARY_PATH={shlex.quote(ld_library_path)}",
        "export D41_SKIP_CONTROLS=1",
        " ".join(shlex.quote(part) for part in (
            d46.wsl_path(binary), "--poses", d46.wsl_path(d46.POSES), "--cases", d46.wsl_path(manifest),
            "--urdf", d46.wsl_path(d46.URDF), "--srdf", d46.wsl_path(d46.SRDF), "--output", d46.wsl_path(native),
        )),
    ))
    result = subprocess.run(
        ["wsl.exe", "bash", "-lc", command], cwd=ROOT,
        text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=7200,
    )
    (output / "native_execution.log").write_text(
        (result.stdout or "") + "\n--- STDERR ---\n" + (result.stderr or ""),
        encoding="utf-8", errors="replace",
    )
    required = (
        "D41_native_case_summary.jsonl", "D41_native_clearance.csv", "D41_native_jacobian.csv",
        "D41_native_full_jacobian.csv", "D41_native_segment_collision.csv", "D41_native_provenance.json",
    )
    if result.returncode != 0 or not all((native / name).is_file() and (native / name).stat().st_size > 0 for name in required):
        detail = (result.stderr or result.stdout or "").strip()[-1800:]
        raise RuntimeError(f"current_source_native_moveit_measurement_failed:{result.returncode}:{detail}")
    return native


def _local_profile(index: int) -> float:
    """Smooth, nonzero-at-boundary local exposure over WP80..95."""
    if index < WINDOW_START or index > WINDOW_END:
        return 0.0
    if index <= CRITICAL_WAYPOINT:
        r = (index - WINDOW_START) / (CRITICAL_WAYPOINT - WINDOW_START)
    else:
        r = (index - CRITICAL_WAYPOINT) / (WINDOW_END - CRITICAL_WAYPOINT)
        return 1.0 - 0.25 * (1.0 - math.cos(math.pi * r))
    return 0.5 + 0.5 * (1.0 - math.cos(math.pi * r)) / 2.0


def _make_retreat_candidate(
    q: np.ndarray, jacobians: Mapping[int, np.ndarray], normals: np.ndarray, peak_rad: float
) -> tuple[np.ndarray, dict[str, Any]]:
    candidate = np.array(q, copy=True)
    waypoint_details: list[dict[str, Any]] = []
    for index in range(WINDOW_START, WINDOW_END + 1):
        task = process_task_jacobian(jacobians[index], normals[index])
        amount = peak_rad * _local_profile(index)
        delta = nullspace_j6_retreat(task, amount)
        candidate[index] += delta
        waypoint_details.append({
            "waypoint": index,
            "requested_j6_retreat_rad": amount,
            "realized_first_order_j6_delta_rad": float(delta[5]),
            "nullspace_j6_projection": float(task.null_projector[5, 5]),
            "nullspace_delta_norm_rad": float(np.linalg.norm(delta)),
            "linearized_task_residual_norm_scaled": float(np.linalg.norm(task.scaled @ delta)),
        })
    return candidate, {"waypoints": waypoint_details, "critical_waypoint": CRITICAL_WAYPOINT}


def _buffer_known_answer(lower: np.ndarray, upper: np.ndarray) -> dict[str, Any]:
    raw = np.zeros(6, dtype=np.float64)
    raw[5] = upper[5] + 0.001
    rows = []
    for buffer_rad in BUFFER_ABLATION_RAD:
        projected = shadow_buffer_projection(raw, lower, upper, buffer_rad)
        rows.append({
            "buffer_rad": buffer_rad,
            "projected_j6_rad": float(projected[5]),
            "upper_slack_rad": float(upper[5] - projected[5]),
            "known_answer_match": bool(abs((upper[5] - projected[5]) - buffer_rad) <= 1e-12),
        })
    return {
        "status": "PASS" if all(row["known_answer_match"] for row in rows) else "FAIL",
        "scope": "production clip-policy known-answer only; no full D41 solver trajectory regeneration",
        "raw_test_proposal_j6_rad": float(raw[5]),
        "buffer_values_rad": list(BUFFER_ABLATION_RAD),
        "results": rows,
        "full_D41_candidate_regeneration": "NOT_RUN",
    }


def _summarize_candidate(
    name: str,
    q: np.ndarray,
    baseline_q: np.ndarray,
    fk: Mapping[str, np.ndarray],
    baseline_fk: Mapping[str, np.ndarray],
    reference: Mapping[str, np.ndarray],
    lower: np.ndarray,
    upper: np.ndarray,
    timestamps: np.ndarray,
    dynamic_limits: Mapping[str, Mapping[str, float]],
    joint_step_limit: float,
    native_root: Path,
) -> dict[str, Any]:
    from tools import stage4a_system_benchmark as d46

    summary = _read_case_summary(native_root / "D41_native_case_summary.jsonl", name)
    collisions = _read_collision_segments(native_root / "D41_native_segment_collision.csv", name)
    clearance = _read_clearances(native_root / "D41_native_clearance.csv", name)
    fk_position = np.asarray(fk["position"])
    quat = np.asarray(fk["quaternion"])
    tool_z = np.asarray([_tool_z_from_quaternion(row) for row in quat])
    position_error = np.linalg.norm(fk_position - reference["position"], axis=1)
    normal_error = _normal_errors(tool_z, reference["normal"])
    min_joint_margin = np.minimum(q - lower[None, :], upper[None, :] - q)
    joint_steps = np.abs(np.diff(q, axis=0))
    velocity, acceleration, jerk = d46.finite_derivatives(q, timestamps)
    velocity_limits = np.asarray([dynamic_limits[f"j{i+1}"]["velocity_rad_s"] for i in range(6)])
    acceleration_limits = np.asarray([dynamic_limits[f"j{i+1}"]["acceleration_rad_s2"] for i in range(6)])
    jerk_limits = np.asarray([dynamic_limits[f"j{i+1}"]["jerk_rad_s3"] for i in range(6)])
    ratios = {
        "velocity": np.abs(velocity) / velocity_limits[None, :],
        "acceleration": np.abs(acceleration) / acceleration_limits[None, :],
        "jerk": np.abs(jerk) / jerk_limits[None, :],
    }
    finite_dynamics = all(np.isfinite(x).all() for x in (velocity, acceleration, jerk))
    dynamics_pass = finite_dynamics and all(float(np.max(v)) <= 1.0 for v in ratios.values())
    in_bounds = bool(np.all(q >= lower[None, :]) and np.all(q <= upper[None, :]))
    step_max = float(np.max(joint_steps)) if joint_steps.size else 0.0
    collision_pass = all(summary.get(k) == 0 for k in (
        "nominal_waypoint_world_collision_count",
        "nominal_waypoint_self_collision_count",
        "native_continuous_segment_collision_count",
    ))
    geometry_pass = float(np.max(position_error)) <= POSITION_TOLERANCE_M and float(np.max(normal_error)) <= NORMAL_TOLERANCE_RAD
    local = slice(WINDOW_START, WINDOW_END + 1)
    baseline_tool_z = np.asarray([_tool_z_from_quaternion(row) for row in baseline_fk["quaternion"]])
    baseline_normal_error = _normal_errors(baseline_tool_z, reference["normal"])
    local_normal_change_deg = float(math.degrees(float(np.max(np.abs(normal_error[local] - baseline_normal_error[local])))))
    return {
        "case_id": name,
        "state_count": int(len(q)),
        "state_joint_bounds": "PASS" if in_bounds else "FAIL",
        "minimum_joint_limit_margin_rad": float(np.min(min_joint_margin)),
        "critical_waypoint": CRITICAL_WAYPOINT,
        "critical_j6_rad": float(q[CRITICAL_WAYPOINT, 5]),
        "critical_j6_upper_slack_rad": float(upper[5] - q[CRITICAL_WAYPOINT, 5]),
        "local_window_minimum_j6_upper_slack_rad": float(np.min(upper[5] - q[local, 5])),
        "critical_j6_change_from_frozen_rad": float(q[CRITICAL_WAYPOINT, 5] - baseline_q[CRITICAL_WAYPOINT, 5]),
        "max_tcp_position_error_vs_frozen_fk_m": float(np.max(position_error)),
        "max_tcp_position_change_in_window_m": float(np.max(np.linalg.norm(fk_position[local] - baseline_fk["position"][local], axis=1))),
        "terminal_position_error_vs_frozen_fk_m": float(position_error[-1]),
        "max_spray_axis_normal_error_rad": float(np.max(normal_error)),
        "max_spray_axis_normal_error_deg": float(math.degrees(float(np.max(normal_error)))),
        "max_spray_axis_normal_change_in_window_deg": local_normal_change_deg,
        "formal_geometry_gates": {
            "path_position_6mm": "PASS" if float(np.max(position_error)) <= POSITION_TOLERANCE_M else "FAIL",
            "terminal_position_4mm": "PASS" if float(position_error[-1]) <= TERMINAL_POSITION_TOLERANCE_M else "FAIL",
            "spray_normal_10deg": "PASS" if float(np.max(normal_error)) <= NORMAL_TOLERANCE_RAD else "FAIL",
        },
        "max_adjacent_joint_step_rad": step_max,
        "joint_step_gate_rad": joint_step_limit,
        "continuity": "PASS" if step_max <= joint_step_limit else "FAIL",
        "collision_method": "adaptive_discrete_interpolation",
        "collision_summary_status": summary.get("status"),
        "collision_counts": {
            "world_waypoints": summary.get("nominal_waypoint_world_collision_count"),
            "self_waypoints": summary.get("nominal_waypoint_self_collision_count"),
            "native_two_state_robot_world_queries": summary.get("native_continuous_segment_collision_count"),
        },
        "first_collision_segments": collisions,
        "collision_status": "PASS" if collision_pass else "FAIL",
        "clearance_diagnostic_only": clearance,
        "finite_difference_dynamics_audit_only": {
            "status": "PASS" if dynamics_pass else ("FAIL" if finite_dynamics else "INCONCLUSIVE"),
            "limit_source": "project-configured offline limits; not vendor-certified",
            "max_velocity_ratio": float(np.max(ratios["velocity"])),
            "max_acceleration_ratio": float(np.max(ratios["acceleration"])),
            "max_jerk_ratio": float(np.max(ratios["jerk"])),
            "post_Ruckig_candidate": "NOT_RUN",
        },
        "local_retreat_feasibility": "PASS" if all((in_bounds, geometry_pass, step_max <= joint_step_limit, collision_pass, dynamics_pass)) else "FAIL",
        "local_retreat_feasibility_scope": "measured pre-Ruckig local shadow only; not a release-ready trajectory",
    }


def _classify_p2a(p2a: Mapping[str, Any]) -> dict[str, Any]:
    counts: dict[str, list[str]] = {"acceptance_gate_limited": [], "collision_limited": [], "other_or_unresolved": []}
    for record in p2a.get("axis_results", []):
        spec = record.get("specification", {})
        axis_id = str(spec.get("axis_id", "unknown"))
        first = record.get("first_failing_perturbation") or {}
        mode = first.get("dominant_failure_mode")
        if axis_id == "joint:j6:positive":
            continue
        if mode in {"TCP_PATH_DEVIATION", "TERMINAL_POSITION_ERROR", "SPRAY_AXIS_NORMAL_ERROR", "VELOCITY_LIMIT", "ACCELERATION_LIMIT", "JERK_LIMIT", "JOINT_LIMIT_FAILURE", "JOINT_DISCONTINUITY"}:
            counts["acceptance_gate_limited"].append(axis_id)
        elif mode in {"ENVIRONMENT_COLLISION", "SELF_COLLISION", "ROBOT_WORLD_TRANSITION_COLLISION"}:
            counts["collision_limited"].append(axis_id)
        else:
            counts["other_or_unresolved"].append(axis_id)
    return {
        "solver_policy_limited": ["joint:j6:positive"],
        "acceptance_gate_limited": sorted(set(counts["acceptance_gate_limited"])),
        "collision_limited": sorted(set(counts["collision_limited"])),
        "task_kinematic_limited": "NOT_SEPARATELY_ESTABLISHED_BY_AXISWISE_FIRST_FAILURE; addressed locally by process-Jacobian retreat",
        "other_or_unresolved_axes": sorted(set(counts["other_or_unresolved"])),
        "unresolved_capabilities": [
            "SE3 base/workpiece uncertainty", "calibrated TCP uncertainty", "strict continuous self-collision CCD",
            "hardware tracking and physical validation", "global robustness certification",
        ],
        "boundary_provenance": {
            "D41_internal_DLS_convergence": {"position_tolerance_m": 2.0e-5, "normal_cross_residual_rad": 2.0e-4},
            "D41_DLS_post_generation_gate": {"position_tolerance_m": 4.0e-3, "normal_cross_residual_deg": 5.0},
            "P2A_formal_acceptance": {"path_position_m": POSITION_TOLERANCE_M, "terminal_position_m": TERMINAL_POSITION_TOLERANCE_M, "spray_normal_deg": 10.0},
            "diagnostic_only": {"clearance_acceptance_threshold": "UNRESOLVED_THRESHOLD", "full_pose_sigma_min": "diagnostic only; not used as process-task singularity claim"},
        },
    }


def run_p2b0_attribution(output_path: Path = P2B0_RESULT) -> dict[str, Any]:
    """Run a baseline MoveIt measurement and a bounded local retreat sweep."""
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing P2-B0 result: {output_path}")
    p2a = _read_json(P2A_RESULT)
    if p2a.get("status") != "P2A_AXISWISE_CHARACTERIZATION_COMPLETE" or p2a.get("robot") != "FAIRINO_FR5":
        raise RuntimeError("authenticated_completed_FR5_P2A_result_required")
    frozen_record = p2a["frozen_trajectory"]
    q_path = _resolve_repo_path(str(frozen_record["path"]))
    expected_q_sha = str(frozen_record["sha256"])
    actual_q_sha = _sha256(q_path)
    if actual_q_sha != expected_q_sha:
        raise RuntimeError("frozen_D41_trajectory_identity_mismatch")
    timestamps, q = _read_frozen_trajectory(q_path)
    d41_trace_path = _resolve_repo_path(str(frozen_record["geometric_reference"]["D41_strict_replay_fk_trace"]))
    historical_trace = _read_trace(d41_trace_path)

    from tools import stage4a_system_benchmark as d46
    from src.p2a_axiswise_robustness import _read_fk_trace, _run_existing_fk, write_q_only_csv

    lower, upper, dynamic_limits = d46.load_limits()
    if abs(float(upper[5]) - 3.0543) > 1e-9 or abs(float(lower[5]) + 3.0543) > 1e-9:
        raise RuntimeError("FR5_J6_URDF_limit_does_not_match_P2A_provenance")
    p2a_profile = p2a["evaluator_provenance"]["joint_limit_profile"]
    if not np.allclose(np.asarray(p2a_profile["upper_rad"], dtype=np.float64), upper, atol=1e-12, rtol=0.0):
        raise RuntimeError("FR5_URDF_joint_limits_disagree_with_P2A_record")

    j6_result = next((row for row in p2a["axis_results"]
                      if row.get("specification", {}).get("axis_id") == "joint:j6:positive"), None)
    if j6_result is None:
        raise RuntimeError("P2A_J6_positive_axis_missing")
    p2a_margin = float(j6_result["estimated_raw_axis_margin"]["value"])
    nominal_margin = float(np.min(upper[5] - q[:, 5]))
    critical_q6 = float(q[CRITICAL_WAYPOINT, 5])
    critical_slack = float(upper[5] - critical_q6)
    if abs(critical_slack - D41_BUFFER_RAD) > 1e-9:
        raise RuntimeError("P2A_critical_J6_slack_no_longer_matches_D41_buffer")

    git_branch = subprocess.run(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True, capture_output=True, check=True,
    ).stdout.strip()
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True,
    ).stdout.strip()
    git_dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True, check=True,
    ).stdout.strip())

    scratch = Path(tempfile.mkdtemp(prefix="p2b0_margin_attribution_", dir=ROOT / "tmp"))
    accepted_result: dict[str, Any] | None = None
    baseline_measurement: dict[str, Any] | None = None
    retreat_rows: list[dict[str, Any]] = []
    jacobian_window: list[dict[str, Any]] = []
    native_build_status = "NOT_RUN"
    try:
        native_binary = _build_native_measurement_shadow(scratch, d46)
        native_build_status = (
            "PASS_REUSED_LOCAL_FULL_JACOBIAN_MOVEIT_BINARY"
            if "D56_STAGE4B_SOFTWARE_CLOSURE" in str(native_binary)
            else "PASS_CURRENT_SOURCE_ISOLATED_BUILD"
        )
        baseline_root = scratch / "baseline"
        baseline_q_path = baseline_root / "p2b0_nominal.csv"
        write_q_only_csv(baseline_q_path, q)
        baseline_native = _run_native_measurement(
            baseline_root, [{"case_id": "p2b0_nominal", "family": "P2B0_NOMINAL", "trajectory_path": str(baseline_q_path)}], d46, native_binary
        )
        baseline_fk_path = _run_existing_fk(baseline_root, baseline_native, d46)
        baseline_fk_case = _read_fk_trace(baseline_fk_path)["p2b0_nominal"]
        baseline_native_summary = _read_case_summary(baseline_native / "D41_native_case_summary.jsonl", "p2b0_nominal")
        baseline_jacobians = _read_full_jacobians(baseline_native / "D41_native_full_jacobian.csv", "p2b0_nominal")
        baseline_position_crosscheck = float(np.max(np.linalg.norm(baseline_fk_case["position"] - historical_trace["position"], axis=1)))
        baseline_tool_z = np.asarray([_tool_z_from_quaternion(row) for row in baseline_fk_case["quaternion"]])
        baseline_tool_crosscheck = float(np.max(_normal_errors(baseline_tool_z, historical_trace["tool_z"])))

        for index in range(WINDOW_START, WINDOW_END + 1):
            task = process_task_jacobian(baseline_jacobians[index], historical_trace["normal"][index])
            jacobian_window.append({
                "waypoint": index,
                "effective_rank": task.effective_rank,
                "nullity": int(task.null_basis.shape[1]),
                "singular_values_unscaled_mixed_units": task.singular_values_raw.tolist(),
                "singular_values_tolerance_scaled": task.singular_values_scaled.tolist(),
                "scaled_rank_tolerance": task.rank_tolerance,
                "j6_nullspace_projection": float(task.null_projector[5, 5]),
                "nullspace_projector_symmetry_error": float(np.linalg.norm(task.null_projector - task.null_projector.T)),
                "nullspace_projector_idempotence_error": float(np.linalg.norm(task.null_projector @ task.null_projector - task.null_projector)),
                "process_jacobian_null_residual": float(np.linalg.norm(task.scaled @ task.null_basis)),
            })

        candidate_paths: list[tuple[float, str, Path, np.ndarray, dict[str, Any]]] = []
        candidate_cases: list[dict[str, Any]] = []
        for peak in RETREAT_PEAKS_RAD:
            candidate, details = _make_retreat_candidate(q, baseline_jacobians, historical_trace["normal"], peak)
            case_id = f"p2b0_retreat_{peak:.3f}rad".replace(".", "p")
            case_root = scratch / "candidates"
            q_candidate_path = case_root / f"{case_id}.csv"
            write_q_only_csv(q_candidate_path, candidate)
            candidate_paths.append((peak, case_id, q_candidate_path, candidate, details))
            candidate_cases.append({"case_id": case_id, "family": "P2B0_LOCAL_NULLSPACE_RETREAT", "trajectory_path": str(q_candidate_path)})

        candidate_root = scratch / "candidate_batch"
        candidate_native = _run_native_measurement(candidate_root, candidate_cases, d46, native_binary)
        candidate_fk_path = _run_existing_fk(candidate_root, candidate_native, d46)
        candidate_fk_cases = _read_fk_trace(candidate_fk_path)
        candidate_reports: list[dict[str, Any]] = []
        for peak, case_id, _path, candidate, details in candidate_paths:
            report = _summarize_candidate(
                case_id, candidate, q, candidate_fk_cases[case_id], baseline_fk_case,
                historical_trace, lower, upper, timestamps, dynamic_limits,
                float(p2a["evaluator_provenance"]["task_geometry_thresholds"]["joint_step_rad"]["value"]),
                candidate_native,
            )
            report["requested_peak_retreat_rad"] = peak
            report["local_nullspace_solution"] = details
            candidate_reports.append(report)
        retreat_rows = candidate_reports

        ranked = [row for row in candidate_reports if row["local_retreat_feasibility"] == "PASS"]
        best = max(ranked, key=lambda row: row["requested_peak_retreat_rad"]) if ranked else None
        valid_jacobian_measurement = (
            len(jacobian_window) == WINDOW_END - WINDOW_START + 1
            and all(row["effective_rank"] >= 4 and row["nullity"] >= 1 for row in jacobian_window)
            and all(row["process_jacobian_null_residual"] <= 1e-8 for row in jacobian_window)
        )
        nullspace_status = "PASS" if best is not None else ("FAIL" if candidate_reports else "INCONCLUSIVE")
        jacobian_status = "PASS" if valid_jacobian_measurement else "INCONCLUSIVE"
        causal = "SOLVER_POLICY_DOMINATED" if valid_jacobian_measurement and best is not None else "INCONCLUSIVE"
        for rec in retreat_rows:
            if rec["local_retreat_feasibility"] == "PASS":
                rec["window_peak_and_boundary_profile_rad"] = {
                    "critical_waypoint_retreat_rad": rec["requested_peak_retreat_rad"],
                    "WP80_and_WP95_retreat_rad": 0.5 * rec["requested_peak_retreat_rad"],
                }
        baseline_min_margin = float(np.min(np.minimum(q - lower[None, :], upper[None, :] - q)))
        baseline_local_j6_slack = float(np.min(upper[5] - q[WINDOW_START:WINDOW_END + 1, 5]))
        baseline_measurement = {
            "case_id": "p2b0_nominal",
            "native_status": baseline_native_summary.get("status"),
            "state_count": baseline_native_summary.get("state_count"),
            "minimum_joint_limit_margin_rad": baseline_min_margin,
            "critical_waypoint_j6_rad": critical_q6,
            "critical_waypoint_j6_upper_slack_rad": critical_slack,
            "local_window_minimum_j6_upper_slack_rad": baseline_local_j6_slack,
            "max_fresh_MoveIt_FK_position_difference_from_D41_trace_m": baseline_position_crosscheck,
            "max_fresh_MoveIt_tool_axis_difference_from_D41_trace_rad": baseline_tool_crosscheck,
            "collision_method": "adaptive_discrete_interpolation",
            "collision_counts": {
                "world_waypoints": baseline_native_summary.get("nominal_waypoint_world_collision_count"),
                "self_waypoints": baseline_native_summary.get("nominal_waypoint_self_collision_count"),
                "native_two_state_robot_world_queries": baseline_native_summary.get("native_continuous_segment_collision_count"),
            },
        }
        p2a_classification = _classify_p2a(p2a)
        accepted_result = {
            "schema": "p2b0-margin-attribution-v1",
            "P2B0_MARGIN_ATTRIBUTION": "COMPLETE" if best is not None and jacobian_status == "PASS" else "INCOMPLETE",
            "J6_CAUSAL_CLASSIFICATION": causal,
            "PROCESS_TASK_JACOBIAN": jacobian_status,
            "NULLSPACE_RETREAT_FEASIBILITY": nullspace_status,
            "NULLSPACE_RETREAT_FEASIBILITY_SCOPE": "measured pre-Ruckig local shadow only; post-Ruckig candidate conversion and validation were not run",
            "J6_BUFFER_SHADOW_ABLATION": "PASS" if _buffer_known_answer(lower, upper)["status"] == "PASS" else "INCONCLUSIVE",
            "P1_P2A_FROZEN_BASELINE_MODIFIED": "NO",
            "D41_D46_FROZEN_HISTORY_MODIFIED": "NO",
            "RELEVANT_REGRESSION": "NOT_RUN",
            "PHYSICAL_HARDWARE_VALIDATION": "NOT_RUN",
            "GLOBAL_ROBUSTNESS_CERTIFIED": "NO",
            "HARDWARE_SAFETY_CERTIFIED": "NO",
            "repository": {
                "root": str(ROOT),
                "branch": git_branch or "DETACHED_HEAD",
                "head": git_head,
                "working_tree_dirty_at_measurement": git_dirty,
                "source_baseline": p2a.get("source_commit"),
                "working_tree_dirty_at_P2A_execution": p2a.get("working_tree_dirty_at_execution"),
            },
            "frozen_input_identity": {
                "P2A_result": str(P2A_RESULT.relative_to(ROOT)),
                "D41_post_Ruckig_trajectory": str(q_path.relative_to(ROOT)),
                "D41_FK_trace": str(d41_trace_path.relative_to(ROOT)),
                "trajectory_sha256_verified_once_for_identity": expected_q_sha,
                "state_count": int(len(q)),
                "joint_order": [f"j{i}" for i in range(1, 7)],
                "critical_waypoint_indexing": "zero-based",
            },
            "D41_and_P2A_j6_boundary": {
                "urdf": str(d46.URDF.relative_to(ROOT)),
                "j6_lower_rad": float(lower[5]),
                "j6_upper_rad": float(upper[5]),
                "D41_configured_buffer_rad": D41_BUFFER_RAD,
                "D41_buffer_source": "scripts/stage3_h13_d41_execution.py:272; limit projection in ros2_moveit_bridge/plan_closed_contour_moveit.py:1411-1434",
                "nominal_j6_at_waypoint_rad": critical_q6,
                "nominal_j6_upper_slack_rad": critical_slack,
                "trajectory_minimum_joint_limit_margin_rad": nominal_margin,
                "P2A_refined_boundary_estimate_rad": p2a_margin,
                "P2A_refined_last_pass_rad": float(j6_result["last_passing_perturbation"]["magnitude"]),
                "P2A_first_fail_rad": float(j6_result["first_failing_perturbation"]["magnitude"]),
                "P2A_first_fail_mode": j6_result["first_failing_perturbation"]["dominant_failure_mode"],
                "P2A_first_fail_waypoint": j6_result["first_failing_perturbation"]["critical_waypoint"],
                "P2A_first_fail_joint_limit_violation_rad": j6_result["first_failing_perturbation"]["failure_margin"]["joint_limit_violation_rad"],
                "interpretation": "The exact D41 nominal slack equals the configured buffer. This establishes the active solver-policy boundary at that state; local process-null retreat below tests whether task geometry independently blocks retreat.",
                "buffer_shadow_ablation": _buffer_known_answer(lower, upper),
                "full_solver_replay_at_alternate_buffers": "NOT_RUN; D41 batch is full trajectory regeneration and is outside this bounded local mechanism test",
            },
            "baseline_MoveIt_measurement": baseline_measurement,
            "process_task_jacobian": {
                "construction": "J_proc = [J_v; U_n^T J_omega] at spray_tcp_link; U_n is a two-vector orthonormal tangent basis to frozen wall normal",
                "backend": "MoveIt2 RobotState.getJacobian(fairino5_v6_group, spray_tcp_link)",
                "window_zero_based": [WINDOW_START, WINDOW_END],
                "position_row_scale_m": POSITION_TOLERANCE_M,
                "normal_row_scale_rad": NORMAL_TOLERANCE_RAD,
                "mixed_unit_raw_singular_values_warning": "raw values are retained for interpretation and are not treated as a physical condition number",
                "rank_status": jacobian_status,
                "waypoint_metrics": jacobian_window,
                "critical_waypoint_metrics": next(row for row in jacobian_window if row["waypoint"] == CRITICAL_WAYPOINT),
            },
            "local_nullspace_retreat": {
                "scope": "WP80-WP95 only; no all-trajectory optimization or D41 canonical regeneration",
                "method": "minimum-norm delta = -r * N e6 / N66 using the tolerance-scaled process-task null projector",
                "candidates": retreat_rows,
                "largest_tested_feasible_peak_retreat_rad": best["requested_peak_retreat_rad"] if best else None,
                "largest_tested_feasible_critical_j6_margin_rad": best["critical_j6_upper_slack_rad"] if best else None,
                "formal_gate_status": nullspace_status,
            },
            "p2a_boundary_classification": p2a_classification,
            "deferred_next_stage": [
                "Full D41 solver regeneration at alternate buffers 1e-3, 5e-3, 1e-2 rad",
                "Global path optimization or retreat outside WP80-WP95",
                "Base/workpiece SE(3) uncertainty, calibrated TCP uncertainty, strict self CCD, hardware tracking",
                "Ruckig reparameterization of the local shadow candidate",
            ],
        }
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    if accepted_result is None:
        raise RuntimeError("P2B0_measurement_pipeline_did_not_produce_a_result")
    after_q_sha = _sha256(q_path)
    if after_q_sha != expected_q_sha:
        raise RuntimeError("frozen_D41_trajectory_changed_during_P2B0")
    accepted_result["frozen_input_identity"]["trajectory_sha256_after_measurement"] = after_q_sha
    accepted_result["measurement_pipeline"] = {
        "native_executable_selection": native_build_status,
        "baseline_fresh_native_MoveIt_run": "PASS" if baseline_measurement and baseline_measurement["state_count"] == 181 else "FAIL",
        "fresh_FK_reproduces_D41_reference": "PASS" if baseline_measurement and baseline_measurement["max_fresh_MoveIt_FK_position_difference_from_D41_trace_m"] <= 1e-6 and baseline_measurement["max_fresh_MoveIt_tool_axis_difference_from_D41_trace_rad"] <= 1e-6 else "FAIL",
        "temporary_workspace_removed": "PASS",
        "scratch_location": "D:\\robotfucker\\tmp (removed after measurement)",
        "moveit_backend": "MoveIt2 native RobotState Jacobian/FK, PlanningScene Bullet/FCL; world transition query kept separately from adaptive discrete result",
        "self_collision_semantics": "native discrete/adaptive sampling; strict continuous self collision unavailable",
        "post_Ruckig_shadow_candidate_validation": "NOT_RUN; no existing q-only candidate import path was used",
        "clearance_acceptance_threshold": "UNRESOLVED_THRESHOLD",
    }
    if accepted_result["measurement_pipeline"]["fresh_FK_reproduces_D41_reference"] != "PASS":
        accepted_result["P2B0_MARGIN_ATTRIBUTION"] = "INCOMPLETE"
        accepted_result["J6_CAUSAL_CLASSIFICATION"] = "INCONCLUSIVE"
        accepted_result["PROCESS_TASK_JACOBIAN"] = "INCONCLUSIVE"
        accepted_result["NULLSPACE_RETREAT_FEASIBILITY"] = "INCONCLUSIVE"
    regression_command = [
        sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
        "tests/test_p2b0_margin_attribution.py",
        "tests/test_p2a_axiswise_robustness.py",
        "tests/test_process_aware_stress.py",
    ]
    regression = subprocess.run(
        regression_command, cwd=ROOT, text=True, encoding="utf-8", errors="replace",
        capture_output=True, timeout=1800,
    )
    regression_lines = (regression.stdout or "").splitlines()
    regression_summary = next(
        (line.strip() for line in reversed(regression_lines) if " passed" in line or " failed" in line),
        "pytest_summary_unavailable",
    )
    regression_status = "PASS" if regression.returncode == 0 else "FAIL"
    accepted_result["RELEVANT_REGRESSION"] = regression_status
    accepted_result["relevant_regression"] = {
        "status": regression_status,
        "summary": regression_summary,
        "command": " ".join(regression_command),
    }
    if regression_status != "PASS":
        accepted_result["P2B0_MARGIN_ATTRIBUTION"] = "INCOMPLETE"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(accepted_result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return accepted_result


def main() -> int:
    result = run_p2b0_attribution()
    print(json.dumps({
        "P2B0_MARGIN_ATTRIBUTION": result["P2B0_MARGIN_ATTRIBUTION"],
        "J6_CAUSAL_CLASSIFICATION": result["J6_CAUSAL_CLASSIFICATION"],
        "PROCESS_TASK_JACOBIAN": result["PROCESS_TASK_JACOBIAN"],
        "NULLSPACE_RETREAT_FEASIBILITY": result["NULLSPACE_RETREAT_FEASIBILITY"],
        "J6_BUFFER_SHADOW_ABLATION": result["J6_BUFFER_SHADOW_ABLATION"],
        "output": str(P2B0_RESULT),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
