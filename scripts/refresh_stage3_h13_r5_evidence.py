#!/usr/bin/env python3
"""Refresh derived R5 summaries after a compact-schema correction.

This only rewrites the additive R5 directory; it never touches upstream
authoritative evidence.  No native computation or model evaluation is run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import stage3_h13_r5_bounded_full_trajectory_reconstruction as r5  # noqa: E402
from src.stage3_h13 import canonical  # noqa: E402
from src.stage3_h13_r5_reconstruction import classify_rollout_drift  # noqa: E402


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(canonical(value), ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(canonical(row), ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n" for row in rows), encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    case_rows = load_jsonl(output / "reconstruction_case_summary.jsonl")
    drift_rows = []
    for row in case_rows:
        drift = dict(row.get("rollout_drift") or {})
        root = (((row.get("raw_autoregressive") or {}).get("native") or {}).get("root_cause") or {})
        if root.get("maximum_joint_space_displacement_required_rad") is not None:
            drift["required_joint_correction_rad"] = float(root["maximum_joint_space_displacement_required_rad"])
        drift_rows.append({"case_id": row.get("case_id"), **drift})
    drift_summary = classify_rollout_drift(drift_rows)
    write_jsonl(output / "rollout_drift_case_summary.jsonl", drift_rows)
    write_json(output / "rollout_drift_root_cause_summary.json", drift_summary)

    native = load_json(output / "native_validation_summary.json")
    native["model_prior_reconstruction"].update({
        key: None for key in ("FINAL_POSITION_VIOLATIONS", "FINAL_VELOCITY_VIOLATIONS", "FINAL_ACCELERATION_VIOLATIONS", "FINAL_JERK_VIOLATIONS", "FINAL_COLLISION_VIOLATIONS", "FINAL_SPRAY_PROCESS_VIOLATIONS")
    })
    write_json(output / "native_validation_summary.json", native)
    reconstruction = load_json(output / "reconstruction_summary.json")
    reconstruction["model_prior_native"].update(native["model_prior_reconstruction"])
    write_json(output / "reconstruction_summary.json", reconstruction)

    certificate = load_json(output / "stage3_h13_r5_terminal_certificate.json")
    certificate["ROOT_CAUSE_CLASS"] = drift_summary["ROOT_CAUSE_CLASS"]
    certificate["FULL_TRAJECTORY_DEFORMATION_CASES"] = drift_summary["FULL_TRAJECTORY_DEFORMATION_CASES"]
    certificate["MEDIAN_FIRST_SIGNIFICANT_DIVERGENCE"] = drift_summary["MEDIAN_FIRST_SIGNIFICANT_DIVERGENCE"]
    certificate["MAX_REQUIRED_RAW_JOINT_CORRECTION_RAD"] = drift_summary["MAX_REQUIRED_RAW_JOINT_CORRECTION_RAD"]
    certificate["MAX_RAW_MODEL_TO_CAUSAL_JOINT_DISPLACEMENT_RAD"] = drift_summary["MAX_RAW_MODEL_TO_CAUSAL_JOINT_DISPLACEMENT_RAD"]
    certificate["MODEL_PRIOR_MEAN_RECONSTRUCTION_MAGNITUDE"] = reconstruction.get("model_prior_mean_reconstruction_magnitude")
    certificate["MODEL_PRIOR_MEAN_OBJECTIVE"] = reconstruction.get("model_prior_mean_reconstruction_magnitude")
    if reconstruction.get("accepted", 0) == 0:
        certificate["PRE_NATIVE_POSITION_VIOLATIONS"] = None
        certificate["PRE_NATIVE_COLLISION_VIOLATIONS"] = None
        certificate["PRE_NATIVE_SPRAY_PROCESS_VIOLATIONS"] = None
    for key in ("POST_RUCKIG_POSITION_VIOLATIONS", "POST_RUCKIG_VELOCITY_VIOLATIONS", "POST_RUCKIG_ACCELERATION_VIOLATIONS", "POST_RUCKIG_JERK_VIOLATIONS", "POST_RUCKIG_COLLISION_VIOLATIONS", "POST_RUCKIG_SPRAY_PROCESS_VIOLATIONS"):
        certificate[key] = None
    write_json(output / "stage3_h13_r5_terminal_certificate.json", certificate)
    replay = r5.run_replays(output, {"case_rows": case_rows, "drift_summary": drift_summary, "reconstruction_summary": reconstruction})
    write_json(output / "replay_summary.json", replay)
    certificate["AUTHORITATIVE_OUTPUT_SIZE_MB"] = round(r5.size_bytes(output) / (1024.0 * 1024.0), 3)
    write_json(output / "stage3_h13_r5_terminal_certificate.json", certificate)
    (output / "FINAL_REPORT.md").write_text(r5.final_report(certificate, output), encoding="utf-8", newline="\n")
    print("refreshed", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
