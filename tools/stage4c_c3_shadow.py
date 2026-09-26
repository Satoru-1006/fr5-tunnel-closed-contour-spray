"""Run genuinely different timing shadows and report non-promotion evidence."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.stage4c_execution_form import (
    ACCEPTANCE,
    D47,
    INPUT_POSES,
    LIMITS,
    case_metric,
    load_targets,
    parse_fk,
    read_matrix,
    read_native_trajectory,
    run_fk,
    run_native_postprocess,
    windows_path_from_wsl,
    write_manifest,
    write_q_only_trajectory,
)


PARENT = ROOT / "outputs" / "D48_STAGE4C_EXECUTION_FORM_V1" / "C1_ACCEPTANCE"
OUT = ROOT / "outputs" / "D48_STAGE4C_EXECUTION_FORM_V1" / "C3_TIMING_SHADOW"


def json_dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def family_for(case_id: str) -> str:
    prefix = case_id.rsplit("_", 1)[0].upper()
    return "COLLISION_SENSITIVE" if prefix == "COLLISION" else prefix


def make_input_manifest(path: Path) -> list[tuple[str, Path, str]]:
    items = []
    for case_id in ACCEPTANCE:
        source = D47 / "cases" / f"{case_id}.csv"
        items.append((case_id, source, family_for(case_id)))
    write_manifest(path, items)
    return items


def smoothness(post: Path) -> dict[str, float]:
    t, _, _, _, jerk = read_native_trajectory(post)
    absolute = np.abs(jerk)
    return {"max_abs_jerk_rad_s3": float(np.max(absolute)), "p95_abs_jerk_rad_s3": float(np.quantile(absolute, 0.95)), "rms_jerk_rad_s3": float(np.sqrt(np.mean(jerk ** 2))), "integrated_abs_jerk_rad_s2": float(np.trapezoid(np.sum(absolute, axis=1), t))}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    input_manifest = OUT / "input_manifest.csv"
    items = make_input_manifest(input_manifest)
    shadows = {"slow_010": (0.10, 0.10), "fast_020": (0.20, 0.20)}
    summaries = {}
    for name, (velocity, acceleration) in shadows.items():
        native_out = OUT / name / "native_postprocess"
        summary = run_native_postprocess(input_manifest, native_out, velocity, acceleration)
        summaries[name] = {"native_summary": summary, "case_metrics": []}
        q_items = []
        for item in summary["cases"]:
            actual = windows_path_from_wsl(str(item["trajectory_csv"]))
            q_path = OUT / name / "geometry_inputs" / f"{item['case_id']}.csv"
            write_q_only_trajectory(actual, q_path)
            q_items.append((str(item["case_id"]), q_path, str(item["family"])))
        q_manifest = OUT / name / "q_manifest.csv"
        write_manifest(q_manifest, q_items)
        fk_path = run_fk(q_manifest, OUT / name)
        fk = parse_fk(fk_path)
        targets = load_targets()
        for case_id, actual, family in [(str(item["case_id"]), windows_path_from_wsl(str(item["trajectory_csv"])), str(item["family"])) for item in summary["cases"]]:
            t, q, _, _, _ = read_native_trajectory(actual)
            source = D47 / "cases" / f"{case_id}.csv"
            # The geometry object is intentionally omitted: this shadow is
            # not promoted on native post-processing metrics alone.
            source_q = read_matrix(source)
            target_p, target_quat = targets
            from tools.stage4c_execution_form import project_reference, quat_angle
            desired_p, desired_quat, projection = project_reference(q, source_q, target_p, target_quat)
            actual_p = np.asarray([[float(row[k]) for k in ("x_m", "y_m", "z_m")] for row in fk[case_id]], dtype=float)
            actual_quat = np.asarray([[float(row[k]) for k in ("qx", "qy", "qz", "qw")] for row in fk[case_id]], dtype=float)
            position_error = np.linalg.norm(actual_p - desired_p, axis=1)
            orientation_error = np.asarray([quat_angle(a, b) for a, b in zip(actual_quat, desired_quat)])
            summaries[name]["case_metrics"].append({"case_id": case_id, "family": family, "duration_s": float(t[-1]), "trajectory_error_max_m": float(position_error.max()), "trajectory_error_p95_m": float(np.quantile(position_error, 0.95)), "terminal_position_error_m": float(np.linalg.norm(actual_p[-1] - target_p[-1])), "terminal_orientation_error_rad": float(orientation_error[-1]), "max_joint_projection_error_rad": float(projection.max()), "smoothness": smoothness(actual)})
    comparisons = []
    base = {row["case_id"]: row for row in summaries["slow_010"]["case_metrics"]}
    for row in summaries["fast_020"]["case_metrics"]:
        left = base[row["case_id"]]
        comparisons.append({"case_id": row["case_id"], "family": row["family"], "slow_010_duration_s": left["duration_s"], "fast_020_duration_s": row["duration_s"], "slow_010_p95_error_m": left["trajectory_error_p95_m"], "fast_020_p95_error_m": row["trajectory_error_p95_m"], "slow_010_max_jerk": left["smoothness"]["max_abs_jerk_rad_s3"], "fast_020_max_jerk": row["smoothness"]["max_abs_jerk_rad_s3"], "slow_010_integrated_abs_jerk": left["smoothness"]["integrated_abs_jerk_rad_s2"], "fast_020_integrated_abs_jerk": row["smoothness"]["integrated_abs_jerk_rad_s2"]})
    report = {"schema_version": "d48-c3-timing-shadow-v1", "status": "SHADOW_COMPLETE_NOT_PROMOTED", "routes": {"slow_010": "native MoveIt2 post-processing at velocity/acceleration scaling 0.10", "fast_020": "native MoveIt2 post-processing at velocity/acceleration scaling 0.20"}, "case_count": len(items), "candidate_promotion": "REJECTED_PENDING_FULL_C1_GEOMETRY_AND_C2_REVALIDATION", "reason": "Two genuinely different timing shadows were measured with actual post-Ruckig states and FK accuracy; neither is promoted from timing/accuracy alone, and the protected parent remains unchanged.", "summaries": summaries, "comparisons": comparisons}
    json_dump(OUT / "c3_shadow_report.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
