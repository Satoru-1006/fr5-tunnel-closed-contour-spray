from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.stage3_h12_r2_process_manifold_post_ruckig import aggregate_native, verify_manifest


def unit(unit_id: str, count: int = 3) -> dict:
    return {
        "unit_id": unit_id,
        "window_count": count,
        "trajectory_family_id": "family_a",
        "segment_id": 0,
        "segment_order": 0,
        "spray_state": "SPRAY_ON",
    }


def result(unit_id: str, *, executed: bool, status: str, post_status: str, ruckig_native=None, check=None, blocker="test") -> dict:
    return {
        "unit_id": unit_id,
        "status": status,
        "pre_ruckig_executed": True,
        "pre_ruckig_status": "BLOCKED",
        "post_ruckig_executed": executed,
        "post_ruckig_status": post_status,
        "first_blocker": blocker,
        "ruckig_native": ruckig_native,
        "post_ruckig_check": check,
        "pre_ruckig_check": check,
    }


def test_not_run_cannot_be_counted_as_validated(tmp_path):
    units = [unit("u0")]
    (tmp_path / "stage3_h12_r2_post_ruckig_results.jsonl").write_text(json.dumps(result("u0", executed=False, status="BLOCKED", post_status="NOT_RUN")) + "\n", encoding="utf-8")
    (tmp_path / "stage3_h12_r2_native_failures.jsonl").write_text("", encoding="utf-8")
    aggregate, _ = aggregate_native(tmp_path, units, {"returncode": 0})
    assert aggregate["POST_RUCKIG_ATTEMPTED_WINDOWS"] == 0
    assert aggregate["POST_RUCKIG_VALIDATED_WINDOWS"] == 0
    assert aggregate["POST_RUCKIG_FAILED_WINDOWS"] == 0
    assert aggregate["POST_RUCKIG_UNVALIDATED_WINDOWS"] == 3
    assert aggregate["status"] == "BLOCKED"


def test_ruckig_error_and_post_failure_are_fail_closed(tmp_path):
    units = [unit("u0")]
    native = {"status": "REPORTED", "RUCKIG_ERROR_INVALID_INPUT": True, "RUCKIG_OTHER_ERRORS": False, "smoothing_complete": False}
    check = {"execution_limits": {"acceleration_violation_count": 1}}
    (tmp_path / "stage3_h12_r2_post_ruckig_results.jsonl").write_text(json.dumps(result("u0", executed=True, status="BLOCKED", post_status="FAILED", ruckig_native=native, check=check)) + "\n", encoding="utf-8")
    (tmp_path / "stage3_h12_r2_native_failures.jsonl").write_text("", encoding="utf-8")
    aggregate, _ = aggregate_native(tmp_path, units, {"returncode": 0})
    assert aggregate["POST_RUCKIG_ATTEMPTED_WINDOWS"] == 3
    assert aggregate["POST_RUCKIG_FAILED_WINDOWS"] == 3
    assert aggregate["POST_RUCKIG_UNVALIDATED_WINDOWS"] == 0
    assert aggregate["POST_RUCKIG_VALIDATED_WINDOWS"] == 0
    assert aggregate["failure_counts"]["RUCKIG_NATIVE_ERROR"] == 3
    assert aggregate["status"] == "BLOCKED"


def test_post_process_and_collision_revalidation_are_counted(tmp_path):
    units = [unit("u0")]
    check = {
        "process": {"failed_spray_on_sample_count": 1, "first_failure": {"tcp_position_error_m": 0.01}},
        "collision": {"self_collision_failure_count": 1, "environment_collision_failure_count": 0},
    }
    (tmp_path / "stage3_h12_r2_post_ruckig_results.jsonl").write_text(json.dumps(result("u0", executed=True, status="BLOCKED", post_status="FAILED", check=check)) + "\n", encoding="utf-8")
    (tmp_path / "stage3_h12_r2_native_failures.jsonl").write_text("", encoding="utf-8")
    aggregate, _ = aggregate_native(tmp_path, units, {"returncode": 0})
    assert aggregate["failure_counts"]["TCP_POSITION"] == 3
    assert aggregate["final_violations"]["SELF_COLLISION_FAILURE_COUNT"] == 1
    assert aggregate["POST_RUCKIG_UNVALIDATED_WINDOWS"] == 0


def test_frozen_manifest_hash_audit_is_fail_closed(tmp_path):
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("immutable", encoding="utf-8")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = tmp_path / "frozen_artifact_manifest.json"
    manifest.write_text(json.dumps({"records": [{"path": artifact.name, "sha256": digest}]}), encoding="utf-8")
    assert verify_manifest(manifest)["status"] == "PASSED"
    artifact.write_text("changed", encoding="utf-8")
    assert verify_manifest(manifest)["status"] == "BLOCKED"


def test_native_and_replay_contracts_are_explicit():
    native_source = Path("ros2_moveit_bridge/stage3_h12_r2_native.py").read_text(encoding="utf-8")
    runner_source = Path("scripts/stage3_h12_r2_process_manifold_post_ruckig.py").read_text(encoding="utf-8")
    assert '"post_ruckig_executed"' in native_source
    assert '"post_ruckig_status"' in native_source
    assert "smoothing_complete" in native_source
    assert "duration_ceiling_hit" in native_source
    assert "subprocess.run" in runner_source
    assert '"fresh_processes": True' in runner_source
