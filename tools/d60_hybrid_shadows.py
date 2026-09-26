"""Bounded D60 hybrid shadow campaign.

The campaign composes two route families already present in the repository:
protected q-path versus local curvature midpoint insertion, followed by the
native MoveIt2/Ruckig execution-form conversion.  It is intentionally small,
keeps every result in the D60 shadow directory, and rejects jerk failures
before any downstream promotion consideration.
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
DEFAULT_OUT = ROOT / "outputs" / "D60_SYSTEM_LEVEL_CLOSURE" / "hybrid_shadows_v2"


def family_for(case_id: str) -> str:
    prefix = case_id.rsplit("_", 1)[0].upper()
    return "COLLISION_SENSITIVE" if prefix == "COLLISION" else prefix


def write_q_path(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["waypoint"] + [f"j{i}_q" for i in range(1, 7)])
        for index, row in enumerate(values):
            writer.writerow([index, *[float(value) for value in row]])


def make_input(source: Path, candidate: str) -> Path:
    if candidate.startswith("global") or candidate.startswith("asymmetric"):
        return source
    q = read_matrix(source)
    curvature = np.linalg.norm(np.diff(q, n=2, axis=0), axis=1)
    selected = sorted(np.argsort(curvature)[-8:].tolist())
    expanded: list[np.ndarray] = []
    for index in range(len(q) - 1):
        expanded.append(q[index])
        if index in selected:
            expanded.append(0.5 * (q[index] + q[index + 1]))
    expanded.append(q[-1])
    path = DEFAULT_OUT / candidate / "input_paths" / source.name
    write_q_path(path, np.asarray(expanded, dtype=float))
    return path


def screen(path: Path) -> dict[str, Any]:
    t, _, _, _, jerk = read_native_trajectory(path)
    absolute = np.abs(jerk)
    return {
        "duration_s": float(t[-1]),
        "max_abs_jerk_rad_s3": float(np.max(absolute)),
        "max_jerk_ratio": float(np.max(absolute) / 8.0),
        "jerk_limit_violations": int(np.sum(absolute > 8.0 + 1.0e-10)),
        "integrated_abs_jerk_rad_s2": float(np.trapezoid(np.sum(absolute, axis=1), t)),
    }


def run_candidate(output_root: Path, candidate: str, velocity: float, acceleration: float) -> dict[str, Any]:
    candidate_root = output_root / candidate
    items = []
    for case_id in ACCEPTANCE:
        source = D47 / "cases" / f"{case_id}.csv"
        items.append((case_id, make_input(source, candidate), family_for(case_id)))
    manifest = candidate_root / "input_manifest.csv"
    write_manifest(manifest, items)
    result: dict[str, Any] = {
        "candidate_id": candidate,
        "architecture": "geometry_path_stage -> native_MoveIt2_Ruckig_execution_form",
        "geometry_stage": "protected_q_path" if candidate.startswith(("global", "asymmetric")) else "local_curvature_midpoint_insertion",
        "velocity_scaling": velocity,
        "acceleration_scaling": acceleration,
        "input_manifest": str(manifest),
        "promotion_status": "SHADOW_ONLY",
    }
    try:
        native = run_native_postprocess(manifest, candidate_root / "native_postprocess", velocity, acceleration)
        cases = []
        for native_case in native.get("cases", []):
            trajectory = Path(str(native_case["trajectory_csv"]).replace("/mnt/d/", "D:/").replace("/", "\\"))
            cases.append({"case_id": str(native_case["case_id"]), **screen(trajectory)})
        result.update(
            {
                "native_summary_status": native.get("status"),
                "case_count": len(cases),
                "cases": cases,
                "duration_max_s": max((item["duration_s"] for item in cases), default=None),
                "duration_mean_s": float(np.mean([item["duration_s"] for item in cases])) if cases else None,
                "max_jerk_ratio": max((item["max_jerk_ratio"] for item in cases), default=None),
                "jerk_limit_violations": sum(item["jerk_limit_violations"] for item in cases),
            }
        )
        result["screen_status"] = "PASS" if cases and result["jerk_limit_violations"] == 0 else "REJECT_JERK"
        result["promotion_status"] = "REQUIRES_FULL_DOWNSTREAM_VALIDATION" if result["screen_status"] == "PASS" else "SHADOW_REJECTED"
    except Exception as error:  # shadow candidate failure is evidence
        result.update({"screen_status": "SHADOW_FAILURE", "promotion_status": "SHADOW_REJECTED", "error": f"{type(error).__name__}:{error}"})
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    candidates = [
        ("global_015", 0.15, 0.15),
        ("asymmetric_010v_015a", 0.10, 0.15),
        ("local_015_midpoints", 0.15, 0.15),
    ]
    results = [run_candidate(output_root, name, velocity, acceleration) for name, velocity, acceleration in candidates]
    payload = {
        "schema_version": "d60-hybrid-shadow-v1",
        "status": "SHADOW_COMPLETE_NOT_PROMOTED",
        "scope": "D47 protected q paths; native execution-form shadow only",
        "candidate_count": len(results),
        "candidates": results,
        "promotion_rule": "a screen pass still requires full native, FK, geometry, conservative articulated certificate, adversarial replay, and regression evidence",
    }
    (output_root / "D60_HYBRID_SHADOWS.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": str(output_root / "D60_HYBRID_SHADOWS.json")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
