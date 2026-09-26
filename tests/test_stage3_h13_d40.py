import csv
import json
from pathlib import Path

from tools import stage3_h13_d40_task_space_geometry_locked_causal_cartesian_recovery_execution as d40


def test_d40_manifest_is_authoritative_181_point_causal_scope():
    manifest = json.loads((d40.OUTPUT / "D40_SEED_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["scope"] == "Stage 0/1 ON-state open-arch 181-point pair"
    assert manifest["future_joint_reads"] == 0
    assert set(manifest["variants"]) == {
        "routeB_ik_initial_anchor",
        "routeB_ik_raw_initial_anchor",
        "routeB_ik_warm_anchor",
    }
    assert all(item["rows"] == 181 for item in manifest["variants"].values())
    assert all(item["future_target_joint_rows_used"] == 0 for item in manifest["variants"].values())
    assert "internal_wiper_moveit_inputs" in manifest["authoritative_tcp_poses"]
    assert "stage3_h13_d39" in manifest["d39_stable_prior"]
    assert "stage3_h13_d39" in manifest["d39_raw_prior"]


def test_d40_native_winner_is_a_real_full_pipeline_pass():
    summary = json.loads((d40.OUTPUT / "D40_SUMMARY.json").read_text(encoding="utf-8"))
    quality_rows = list(csv.DictReader((d40.NATIVE_WINNER / "moveit_quality_report.csv").open(encoding="utf-8-sig", newline="")))
    quality = {row["metric"]: row["value"] for row in quality_rows}
    acceptance = json.loads((d40.NATIVE_WINNER / "final_acceptance_summary.json").read_text(encoding="utf-8"))
    assert summary["TASK_STATUS"] == "PASS"
    assert summary["D39_BASELINE_RETENTION_STATUS"] == "LOCKED"
    assert summary["D40_RETENTION_BASELINE_2"]["status"] == "LOCKED"
    assert quality["status"] == "pass"
    assert float(quality["fk_path_deviation_p95_mm"]) < 6.0
    assert float(quality["fk_path_deviation_max_mm"]) < 6.0
    assert float(quality["fk_standoff_error_max_abs_mm"]) < 13.0
    assert float(quality["fk_normal_error_max_deg"]) < 10.0
    assert acceptance["overall_status"] == "pass"


def test_d39_retention_floor_is_unchanged_and_locked():
    retention = json.loads(
        (Path(d40.ROOT) / "outputs" / "stage3_h13_d39_causal_task_space_recovery" / "D39_RETENTION_BASELINES.json").read_text(
            encoding="utf-8"
        )
    )
    assert retention["baseline_1"]["status"] == "LOCKED"
    assert retention["baseline_1"]["candidate_id"] == "stable_velocity_residual_update"
    assert retention["baseline_2"]["status"] == "NONE"
