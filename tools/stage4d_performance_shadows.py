"""Run distinct D49 execution-form timing shadows after C2 closure.

All candidates remain under D49 shadow output.  Native MoveIt2 post-Ruckig is
the only source of candidate trajectories; this module never time-warps a
post-Ruckig artifact and never changes D47/D48 protected state.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from tools.stage4c_execution_form import (
    ACCEPTANCE,
    D47,
    read_matrix,
    read_native_trajectory,
    run_native_postprocess,
    write_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "D49_STAGE4D_SHADOW" / "performance"


def family_for(case_id: str) -> str:
    prefix = case_id.rsplit("_", 1)[0].upper()
    return "COLLISION_SENSITIVE" if prefix == "COLLISION" else prefix


def write_q_path(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["waypoint"] + [f"j{i}_q" for i in range(1, 7)])
        for index, row in enumerate(values):
            writer.writerow([index, *[float(value) for value in row]])


def make_manifest(path: Path, candidate: str) -> None:
    items: list[tuple[str, Path, str]] = []
    for case_id in ACCEPTANCE:
        source = D47 / "cases" / f"{case_id}.csv"
        if candidate != "local_0075_midpoints":
            trajectory = source
        else:
            q = read_matrix(source)
            curvature = np.linalg.norm(np.diff(q, n=2, axis=0), axis=1)
            selected = sorted(np.argsort(curvature)[-8:].tolist())
            inserts = set(selected)
            expanded: list[np.ndarray] = []
            for index in range(len(q) - 1):
                expanded.append(q[index])
                if index in inserts:
                    expanded.append(0.5 * (q[index] + q[index + 1]))
            expanded.append(q[-1])
            trajectory = OUT / candidate / "input_paths" / f"{case_id}.csv"
            write_q_path(trajectory, np.asarray(expanded, dtype=float))
        items.append((case_id, trajectory, family_for(case_id)))
    write_manifest(path, items)


def screen_trajectory(path: Path) -> dict[str, Any]:
    t, _, _, _, jerk = read_native_trajectory(path)
    absolute = np.abs(jerk)
    return {
        "duration_s": float(t[-1]),
        "max_abs_jerk_rad_s3": float(np.max(absolute)),
        "max_jerk_ratio": float(np.max(absolute) / 8.0),
        "jerk_limit_violations": int(np.sum(absolute > 8.0 + 1.0e-10)),
        "integrated_abs_jerk_rad_s2": float(np.trapezoid(np.sum(absolute, axis=1), t)),
    }


def run_candidate(candidate: str, velocity: float, acceleration: float, output_root: Path) -> dict[str, Any]:
    candidate_root = output_root / candidate
    manifest = candidate_root / "input_manifest.csv"
    make_manifest(manifest, candidate)
    record: dict[str, Any] = {
        "candidate_id": candidate,
        "family": {
            "global_0075": "B_RUCKIG_AWARE_GLOBAL_SCALING",
            "asymmetric_005v_010a": "B_RUCKIG_AWARE_ASYMMETRIC_LIMITS",
            "local_0075_midpoints": "A_LOCAL_SEGMENT_WAYPOINT_REFINEMENT",
        }[candidate],
        "velocity_scaling": velocity,
        "acceleration_scaling": acceleration,
        "input_manifest": str(manifest),
        "promotion_status": "SHADOW_ONLY",
    }
    try:
        native = run_native_postprocess(manifest, candidate_root / "native_postprocess", velocity, acceleration)
        cases: list[dict[str, Any]] = []
        for native_case in native.get("cases", []):
            trajectory = Path(str(native_case["trajectory_csv"]).replace("/mnt/d/", "D:/").replace("/", "\\"))
            item = {"case_id": str(native_case["case_id"]), **screen_trajectory(trajectory)}
            cases.append(item)
        record["native_summary_status"] = native.get("status")
        record["case_count"] = len(cases)
        record["cases"] = cases
        record["duration_max_s"] = max((item["duration_s"] for item in cases), default=None)
        record["duration_mean_s"] = float(np.mean([item["duration_s"] for item in cases])) if cases else None
        record["max_jerk_ratio"] = max((item["max_jerk_ratio"] for item in cases), default=None)
        record["jerk_limit_violations"] = sum(item["jerk_limit_violations"] for item in cases)
        record["screen_status"] = "PASS" if cases and record["jerk_limit_violations"] == 0 else "REJECT_JERK"
        record["promotion_status"] = "REQUIRES_FULL_C1_C2_REVALIDATION" if record["screen_status"] == "PASS" else "SHADOW_REJECTED"
    except Exception as error:  # candidate failure is evidence, not a campaign failure
        record.update({"screen_status": "SHADOW_FAILURE", "promotion_status": "SHADOW_REJECTED", "error": f"{type(error).__name__}:{error}"})
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUT)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    candidates = [
        ("global_0075", 0.075, 0.075),
        ("asymmetric_005v_010a", 0.05, 0.10),
        ("local_0075_midpoints", 0.075, 0.075),
    ]
    results = [run_candidate(name, velocity, acceleration, args.output_root) for name, velocity, acceleration in candidates]
    payload = {
        "schema_version": "d49-performance-shadow-v1",
        "status": "SHADOW_COMPLETE_NOT_PROMOTED",
        "parent_execution_form": "outputs/D48_STAGE4C_EXECUTION_FORM_V1/C3_ULTRA_SLOW_005",
        "candidate_count": len(results),
        "candidates": results,
        "promotion_rule": "safety and protected gates first; no candidate may promote without complete native C1, continuous C2, replay, and regression evidence",
    }
    (args.output_root / "performance_shadow_report.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
