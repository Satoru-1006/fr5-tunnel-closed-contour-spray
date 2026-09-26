"""D58 isolated certification driver for realized D57 optimizer shadows.

The driver consumes only the D57 shadow outputs and writes under D58.  It
reuses the established native MoveIt2 geometry/FK metric functions, but keeps
candidate identity distinct from the protected floor and canonical tree.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.stage4c_execution_form import (
    ACCELERATION,
    JERK,
    LOWER,
    UPPER,
    VELOCITY,
    parse_fk,
    parse_geometry,
    project_reference,
    read_matrix,
    read_native_trajectory,
    run_fk,
    run_geometry,
    summarize,
    windows_path_from_wsl,
    write_manifest,
    write_q_only_trajectory,
)
from tools.d56_accuracy_probe import project_task_path, quat_z_axis, read_targets


DEFAULT_NATIVE = ROOT / "outputs" / "D58_STAGE4_FULL_SYSTEM_CLOSURE" / "c4_shadows_native_post_ruckig"
DEFAULT_OUTPUT = ROOT / "outputs" / "D58_STAGE4_FULL_SYSTEM_CLOSURE" / "c4_shadows_full_native_metrics"


def d58_case_metric(case_id: str, family: str, source: Path, post: Path, geometry: dict, fk_rows: list[dict[str, str]], targets: tuple) -> dict:
    """Use D56's established Cartesian accuracy observable for this campaign."""
    t, q, dq, ddq, jerk = read_native_trajectory(post)
    source_q = read_matrix(source)
    target_p, target_quat, target_normal = targets
    actual_p = np.asarray([[float(row[k]) for k in ("x_m", "y_m", "z_m")] for row in fk_rows], dtype=float)
    actual_quat = np.asarray([[float(row[k]) for k in ("qx", "qy", "qz", "qw")] for row in fk_rows], dtype=float)
    if len(actual_p) != len(q):
        raise RuntimeError(f"fk_trajectory_count_mismatch:{case_id}:{len(actual_p)}:{len(q)}")
    _, _, projection_error = project_reference(q, source_q, target_p, target_quat)
    desired_p, desired_normal, _ = project_task_path(actual_p, target_p, target_normal)
    pos_error = np.linalg.norm(actual_p - desired_p, axis=1)
    actual_normal = np.asarray([quat_z_axis(item) for item in actual_quat], dtype=float)
    normal_error = np.asarray([
        math.acos(float(np.clip(np.dot(actual, desired), -1.0, 1.0)))
        for actual, desired in zip(actual_normal, desired_normal)
    ], dtype=float)
    segment_counts = geometry.get("segment_counts", {})
    q_margin = np.minimum(q - LOWER[None, :], UPPER[None, :] - q)
    item = dict(geometry)
    item.update({
        "case_id": case_id,
        "family": family,
        "state_count": int(len(q)),
        "trajectory_duration_s": float(t[-1]),
        "native_post_ruckig": True,
        "finite_state": bool(all(np.isfinite(x).all() for x in (t, q, dq, ddq, jerk, actual_p, actual_quat))),
        "jerk_method": "DERIVED_FINITE_DIFFERENCE_FROM_NATIVE_POST_RUCKIG_ACCELERATION",
        "terminal_position_error_m": float(np.linalg.norm(actual_p[-1] - target_p[-1])),
        "terminal_orientation_error_rad": None,
        "terminal_normal_error_rad": float(normal_error[-1]),
        "tcp_trajectory_error_mean_m": float(pos_error.mean()),
        "tcp_trajectory_error_rms_m": float(np.sqrt(np.mean(pos_error ** 2))),
        "tcp_trajectory_error_p95_m": float(np.quantile(pos_error, 0.95)),
        "tcp_trajectory_error_p99_m": float(np.quantile(pos_error, 0.99)),
        "tcp_trajectory_error_max_m": float(pos_error.max()),
        "tcp_normal_error_p95_rad": float(np.quantile(normal_error, 0.95)),
        "tcp_normal_error_max_rad": float(normal_error.max()),
        "max_joint_projection_error_rad": float(projection_error.max()),
        "max_velocity_ratio": float(np.max(np.abs(dq) / VELOCITY[None, :])),
        "max_acceleration_ratio": float(np.max(np.abs(ddq) / ACCELERATION[None, :])),
        "max_jerk_ratio": float(np.max(np.abs(jerk) / JERK[None, :])),
        "joint_limit_violation_count": int(np.sum(q_margin < 0.0)),
        "velocity_limit_violations": int(np.sum(np.abs(dq) > VELOCITY[None, :] + 1.0e-10)),
        "acceleration_limit_violations": int(np.sum(np.abs(ddq) > ACCELERATION[None, :] + 1.0e-10)),
        "jerk_limit_violations": int(np.sum(np.abs(jerk) > JERK[None, :] + 1.0e-10)),
        "min_joint_limit_margin_rad": float(q_margin.min()),
        "max_position_jump_rad": float(np.max(np.abs(np.diff(q, axis=0)))),
        "continuity_position_failure": bool(np.max(np.abs(np.diff(q, axis=0))) > math.radians(20.0)),
        "dense_environment_collision_samples": int(segment_counts.get("env", 0)),
        "dense_self_collision_samples": int(segment_counts.get("self", 0)),
        "collision_method": "adaptive_discrete_interpolation",
        "execution_sample_policy": "all native post-Ruckig states at 0.01 s plus retained adaptive interval checks",
    })
    return item


def read_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run(native_dir: Path, output: Path, reuse_measurements: bool = False) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    summary = read_summary(native_dir / "execution_form_summary.json")
    actual_items = [
        (
            str(item["case_id"]),
            windows_path_from_wsl(str(item["trajectory_csv"])),
            str(item["family"]),
            windows_path_from_wsl(str(item["source_trajectory"])),
        )
        for item in summary.get("cases", [])
        if item.get("status") == "PASS"
    ]
    if not actual_items:
        raise RuntimeError("d58_native_summary_has_no_passing_cases")

    geometry_items = []
    for case_id, actual, family, _ in actual_items:
        q_only = output / "geometry_inputs" / f"{case_id}.csv"
        write_q_only_trajectory(actual, q_only)
        geometry_items.append((case_id, q_only, family))
    manifest = output / "post_manifest.csv"
    write_manifest(manifest, geometry_items)
    geometry_path = output / "geometry"
    fk_path = output / "fk.csv"
    required_geometry = [geometry_path / name for name in ("D41_native_case_summary.jsonl", "D41_native_clearance.csv", "D41_native_jacobian.csv", "D41_native_segment_collision.csv", "D41_native_provenance.json")]
    if not reuse_measurements or not (fk_path.is_file() and all(path.is_file() and path.stat().st_size > 0 for path in required_geometry)):
        run_geometry(manifest, geometry_path)
        fk_path = run_fk(manifest, output)
    geometry = parse_geometry(geometry_path)
    fk = parse_fk(fk_path)
    targets = read_targets(ROOT / "outputs" / "internal_wiper_moveit_inputs" / "open_arch_tcp_poses_base_link.csv")
    metrics = [
        d58_case_metric(case_id, family, source, actual, geometry[case_id], fk[case_id], targets)
        for case_id, actual, family, source in actual_items
    ]
    result = summarize(metrics, summary, output)
    result["schema_version"] = "d58-c4-shadow-full-native-metrics-v1"
    result["candidate_safety_status"] = "PASS" if all(result["hard_gates"].values()) else "BLOCKED"
    result["promotion_status"] = "PENDING_CERTIFICATE_DYNAMICS_REPLAY_REGRESSION"
    result["source_contract"] = {
        "native_summary": str((native_dir / "execution_form_summary.json").resolve()),
        "candidate_input_contract": "D57 full q/dq/ddq/jerk shadow; native worker consumes explicit j1..j6 q path",
        "task_reference_source": "each native summary source_trajectory (the exact D57 shadow q path)",
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    with (output / "case_metrics.jsonl").open("w", encoding="utf-8") as stream:
        for row in metrics:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-dir", type=Path, default=DEFAULT_NATIVE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reuse-measurements", action="store_true")
    args = parser.parse_args()
    result = run(args.native_dir.resolve(), args.output_dir.resolve(), reuse_measurements=args.reuse_measurements)
    print(json.dumps({"status": result["candidate_safety_status"], "case_count": result["case_count"], "hard_gates": result["hard_gates"]}, sort_keys=True))
    return 0 if result["candidate_safety_status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
