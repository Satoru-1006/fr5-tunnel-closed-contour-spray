"""Aggregate the isolated D50 native-profile jerk truth campaign.

This report keeps the sampled acceleration derivative visible for diagnosis,
but treats Ruckig Profile.j as the authoritative jerk oracle.  It reads only
shadow outputs and never alters a protected Stage 3/D47/D48/D49 artifact.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def finite_difference_metrics(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    t = np.asarray([float(row["t"]) for row in rows], dtype=float)
    ddq = np.asarray([[float(row[f"j{i}_ddq"]) for i in range(1, 7)] for row in rows], dtype=float)
    stored = np.asarray([[float(row[f"j{i}_jerk"]) for i in range(1, 7)] for row in rows], dtype=float)
    recomputed = np.gradient(ddq, t, axis=0, edge_order=1)
    absolute = np.abs(stored)
    return {
        "state_count": int(len(t)),
        "max_abs_finite_difference_jerk_rad_s3": float(np.max(absolute)),
        "finite_difference_jerk_ratio": float(np.max(absolute) / 8.0),
        "finite_difference_limit_violations": int(np.sum(absolute > 8.0 + 1.0e-10)),
        "stored_vs_recomputed_max_abs_delta": float(np.max(np.abs(stored - recomputed))),
    }


def audit_candidate(root: Path) -> dict[str, Any]:
    oracle_paths = sorted((root / "oracle_summaries").glob("*.json"))
    case_metrics: list[dict[str, Any]] = []
    for oracle_path in oracle_paths:
        summary = load_json(oracle_path)
        case_id = oracle_path.stem
        post = root / "native_postprocess" / "trajectories" / f"{case_id}.csv"
        fd = finite_difference_metrics(post)
        case_metrics.append({
            "case_id": case_id,
            "analytic": summary,
            "finite_difference": fd,
        })
    analytic_segments = sum(int(item["analytic"]["segments"]) for item in case_metrics)
    analytic_profiles = sum(int(item["analytic"]["successful_native_profiles"]) for item in case_metrics)
    analytic_invalid = sum(int(item["analytic"]["error_invalid_input_count"]) for item in case_metrics)
    analytic_errors = sum(int(item["analytic"]["other_error_count"]) for item in case_metrics)
    fd_violations = sum(int(item["finite_difference"]["finite_difference_limit_violations"]) for item in case_metrics)
    return {
        "case_count": len(case_metrics),
        "analytic_segment_count": analytic_segments,
        "analytic_successful_profile_count": analytic_profiles,
        "analytic_invalid_input_count": analytic_invalid,
        "analytic_other_error_count": analytic_errors,
        "analytic_max_abs_jerk_rad_s3": max((float(item["analytic"]["max_abs_analytic_jerk_rad_s3"]) for item in case_metrics), default=None),
        "analytic_max_jerk_ratio": max((float(item["analytic"]["max_analytic_jerk_ratio"]) for item in case_metrics), default=None),
        "finite_difference_max_abs_jerk_rad_s3": max((float(item["finite_difference"]["max_abs_finite_difference_jerk_rad_s3"]) for item in case_metrics), default=None),
        "finite_difference_max_jerk_ratio": max((float(item["finite_difference"]["finite_difference_jerk_ratio"]) for item in case_metrics), default=None),
        "finite_difference_limit_violations": fd_violations,
        "analytic_oracle_pass": bool(case_metrics and analytic_profiles == analytic_segments and analytic_invalid == 0 and analytic_errors == 0 and all(item["analytic"]["passed"] for item in case_metrics)),
        "cases": sorted(case_metrics, key=lambda item: item["case_id"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidates = {path.name: audit_candidate(path) for path in sorted(args.root.iterdir()) if path.is_dir() and (path / "oracle_summaries").is_dir()}
    payload = {
        "schema_version": "d50-jerk-truth-resolution-v1",
        "scope": "Stage 0/1 ON-state open-arch execution-form shadows only",
        "authoritative_jerk_oracle": "NATIVE_RUCKIG_PROFILE_J",
        "non_authoritative_diagnostic": "DERIVED_FINITE_DIFFERENCE_FROM_NATIVE_POST_RUCKIG_ACCELERATION",
        "resolution": "FINITE_DIFFERENCE_NOT_AUTHORITATIVE_NATIVE_PROFILE_J_PASS",
        "candidates": candidates,
        "all_analytic_profiles_pass": bool(candidates) and all(item["analytic_oracle_pass"] for item in candidates.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if payload["all_analytic_profiles_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
