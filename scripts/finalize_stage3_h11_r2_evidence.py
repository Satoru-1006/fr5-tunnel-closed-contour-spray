#!/usr/bin/env python3
"""Finalize already-produced H11-R2 evidence without retraining or reevaluation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.stage3_h11_r2_low_storage_unseen_generalization import final_report, output_size_mb  # noqa: E402


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: finalize_stage3_h11_r2_evidence.py OUTPUT_DIR")
    output = Path(sys.argv[1]).resolve()
    cert_path = output / "stage3_h11_r2_terminal_certificate.json"
    native_path = output / "native_prediction_validation.json"
    cert = load(cert_path)
    native = load(native_path)
    summaries = native.get("summaries", []) if isinstance(native.get("summaries", []), list) else []
    dynamic_violations = sum(int((row.get("check") or {}).get("execution_limits", {}).get("velocity_violation_count", 0) or 0) + int((row.get("check") or {}).get("execution_limits", {}).get("acceleration_violation_count", 0) or 0) for row in summaries)
    executed = bool(native.get("status") == "PASSED" and native.get("native_backend_executed") and native.get("planning_scene_executed") and native.get("fk_executed") and native.get("dynamics_executed") and native.get("post_ruckig_executed"))
    cert["NATIVE_VALIDATION_EXECUTED"] = "YES" if executed else "NO"
    cert["NATIVE_SAFETY_CHECK_STATUS"] = "PASS" if executed and all(str((row.get("check") or {}).get("status")) == "PASSED" for row in summaries) else "BLOCKED_RAW_DYNAMIC_LIMIT_VIOLATIONS" if executed else "NOT_AVAILABLE"
    cert["NATIVE_RAW_DYNAMIC_LIMIT_VIOLATIONS"] = dynamic_violations
    cert["NATIVE_RAW_COLLISION_VIOLATIONS"] = native.get("RAW_MODEL_COLLISION_VIOLATIONS")
    cert["NATIVE_RAW_PROCESS_VIOLATIONS"] = native.get("RAW_MODEL_PROCESS_TOLERANCE_VIOLATIONS")
    cert["CCD_AVAILABLE"] = native.get("CCD_AVAILABLE")
    cert["CLEARANCE_AVAILABLE"] = native.get("CLEARANCE_AVAILABLE")
    cert["COLLISION_METHOD"] = native.get("COLLISION_METHOD")
    cert["AUTHORITATIVE_OUTPUT_SIZE_MB"] = round(output_size_mb(output), 3)
    write(cert_path, cert)

    direct = load(output / "generalization_metrics.json")
    baseline = load(output / "causal_baseline_metrics.json")["GENERALIZATION"]
    rollout = load(output / "rollout_error_summary.json")
    ablation = load(output / "ablation_summary.json")
    summary = {
        "schema_version": "stage3_h11_r2_generalization_summary_v1",
        "terminal_status": cert.get("STAGE_3_H11_R2"),
        "first_blocker": cert.get("FIRST_BLOCKER"),
        "selected_candidate": ablation.get("selected_candidate"),
        "selection_split": ablation.get("selection_split"),
        "selection_metric": ablation.get("selection_metric"),
        "direct": direct,
        "causal_constant_velocity_baseline": baseline,
        "rollout": {"generalization": rollout.get("roles", {}).get("GENERALIZATION"), "horizon_16": rollout.get("horizon_16")},
        "comparison": {
            "original_direct_generalization_rmse": cert.get("ORIGINAL_MODEL_RMSE"),
            "h11_r2_direct_generalization_rmse": cert.get("H11_R2_MODEL_RMSE"),
            "causal_baseline_generalization_rmse": cert.get("CAUSAL_BASELINE_RMSE"),
            "original_autoregressive_generalization_rmse": cert.get("ORIGINAL_AUTOREGRESSIVE_RMSE"),
            "h11_r2_autoregressive_generalization_rmse": cert.get("H11_R2_AUTOREGRESSIVE_RMSE"),
            "direct_improvement_vs_original_percent": cert.get("IMPROVEMENT_VS_ORIGINAL_PERCENT"),
            "direct_improvement_vs_baseline_percent": cert.get("IMPROVEMENT_VS_BASELINE_PERCENT"),
            "rollout_improvement_vs_original_percent": cert.get("ROLLOUT_IMPROVEMENT_PERCENT"),
        },
        "leakage": {"TRAIN_SAMPLE_LEAKAGE": cert.get("TRAIN_SAMPLE_LEAKAGE"), "VALIDATION_SAMPLE_LEAKAGE": cert.get("VALIDATION_SAMPLE_LEAKAGE"), "FUTURE_LABEL_LEAKAGE": cert.get("FUTURE_LABEL_LEAKAGE"), "TEST_USED_FOR_SELECTION": cert.get("TEST_USED_FOR_SELECTION"), "GENERALIZATION_USED_FOR_SELECTION": cert.get("GENERALIZATION_USED_FOR_SELECTION"), "H13_UNSEEN_USED_FOR_TRAINING_OR_SELECTION": cert.get("H13_UNSEEN_USED_FOR_TRAINING_OR_SELECTION")},
        "native": {"executed": cert.get("NATIVE_VALIDATION_EXECUTED"), "safety_check_status": cert.get("NATIVE_SAFETY_CHECK_STATUS"), "raw_dynamic_limit_violations": dynamic_violations, "raw_collision_violations": native.get("RAW_MODEL_COLLISION_VIOLATIONS"), "raw_process_violations": native.get("RAW_MODEL_PROCESS_TOLERANCE_VIOLATIONS"), "collision_method": native.get("COLLISION_METHOD"), "ccd_available": native.get("CCD_AVAILABLE"), "clearance_available": native.get("CLEARANCE_AVAILABLE")},
    }
    write(output / "generalization_summary.json", summary)
    (output / "FINAL_REPORT.md").write_text(final_report(cert, output), encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
