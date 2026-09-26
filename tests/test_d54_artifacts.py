from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
D54 = ROOT / "outputs" / "D54_STAGE4_OFFLINE_TRAJECTORY_CERTIFICATION"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_d54_final_artifact_contract() -> None:
    final = load(D54 / "D54_FINAL_CERTIFICATION.json")
    self_cert = load(D54 / "D54_CONTINUOUS_SELF_COLLISION_CERTIFICATION.json")
    clearance = load(D54 / "D54_CONTINUOUS_CLEARANCE_CERTIFICATION.json")
    metrics = load(D54 / "D54_KINEMATIC_DYNAMICS_METRICS.json")
    ledger = load(D54 / "D54_EXPERIMENT_LEDGER.json")

    assert final["task_status"] == "D54_PASS"
    assert final["canonical_promotion"] == "NO"
    assert final["measurement_pipeline_status"] == "PASS"
    assert self_cert["case_count"] == 12
    assert self_cert["passing_case_count"] == 12
    assert self_cert["pair_universe_count"] == 21
    assert self_cert["required_pair_count"] == 10
    assert self_cert["required_pair_coverage_complete"] is True
    assert self_cert["collision_region_count"] == 0
    assert self_cert["unresolved_region_count"] == 0
    assert clearance["status"] == "PASS"
    assert clearance["hardware_clearance"] == "not_available"
    assert clearance["unresolved_is_not_pass"] is True
    assert metrics["continuity"]["status"] == "PASS"
    assert metrics["joint_limit_checks"]["status"] == "PASS"
    assert metrics["singularity"]["threshold_status"] == "UNRESOLVED_THRESHOLD"
    assert ledger["deterministic_replay"]["status"] == "PASS"


def test_d54_repaired_case_summaries_are_pair_complete_and_tri_state_safe() -> None:
    path = D54 / "self_collision_repaired_final" / "D54_CONTINUOUS_SELF_COLLISION_CASES.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 12
    for row in rows:
        assert row["required_pair_count"] == 10
        assert row["status"] == "PASS"
        assert row["strict_articulated_continuous_self_collision"] == "PASS"
        assert row["continuous_self_clearance_status"] == "PASS"
        assert row["unresolved_region_count"] == 0
        assert all(run["status_runs"] and all(item["status"] != "UNRESOLVED" for item in run["status_runs"])
                   for run in row["pair_interval_runs"])


def test_d54_bullet_manifest_is_explicitly_q_only() -> None:
    with (D54 / "D54_BULLET_Q_ONLY_MANIFEST.csv").open(encoding="utf-8", newline="") as stream:
        header = next(csv.reader(stream))
        first_case = next(csv.DictReader(stream, fieldnames=header))
    assert header == ["case_id", "trajectory_csv", "family"]
    q_only_path = Path(first_case["trajectory_csv"].replace("/mnt/d/", "D:/").replace("/", "\\"))
    with q_only_path.open(encoding="utf-8", newline="") as stream:
        row = next(csv.reader(stream))
    assert row == ["waypoint", "j1_q", "j2_q", "j3_q", "j4_q", "j5_q", "j6_q"]
