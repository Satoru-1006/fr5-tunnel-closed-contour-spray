from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.audit_stage25r2 import audit_case, final_replay_audit, generation_audit, profile_stats
from scripts.run_stage25r2 import safe_copy, sha256


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _call(index: int, waypoint: int, generation: int, accepted: bool, retry: bool) -> tuple[dict, dict]:
    call = {
        "global_calculate_call_index": index,
        "waypoint_idx": waypoint,
        "attempt_index": 0,
        "trajectory_generation": generation,
        "duration_extension_factor": 1.0,
        "result": {"numeric_result": 0, "result_name": "Working"},
        "input": {"max_jerk": [8.0] * 6},
        "native_output": {
            "duration": 0.1,
            "profile_available": True,
            "profiles": [{"j": [8.0], "brake": {"j": []}} for _ in range(6)],
        },
    }
    resolution = {
        "global_calculate_call_index": index,
        "waypoint_idx": waypoint,
        "attempt_index": 0,
        "trajectory_generation": generation,
        "accepted_segment_call": accepted,
        "retry_call": retry,
        "overshoot": retry,
        "duration_extension_factor": 1.1 if retry else 1.0,
        "robot_trajectory_segment_duration_after": 0.11 if retry else 0.1,
    }
    return call, resolution


def test_historical_early_return_is_not_strict_completion(tmp_path: Path) -> None:
    native = tmp_path / "native"
    native.mkdir()
    call, resolution = _call(0, 0, 0, True, False)
    _write_jsonl(native / "stage25r_ruckig_calls.jsonl", [call])
    _write_jsonl(native / "stage25r_ruckig_call_resolutions.jsonl", [resolution])
    (native / "stage25r_native_run_summary.json").write_text(
        json.dumps({"moveit_native_return_value": True, "moveit_smoothing_complete": False, "trajectory_generation_final": 0}),
        encoding="utf-8",
    )
    audit = generation_audit(native, [resolution], 2)
    assert audit["all_nominal_segments_reached"] is False
    assert audit["missing_segments"] == [1]


def test_all_call_jerk_audit_includes_retry_profiles(tmp_path: Path) -> None:
    calls = []
    resolutions = []
    for index, (accepted, retry) in enumerate(((True, False), (False, True))):
        call, resolution = _call(index, index, 0, accepted, retry)
        calls.append(call)
        resolutions.append(resolution)
    path = tmp_path / "calls.jsonl"
    _write_jsonl(path, calls)
    stats = profile_stats(path, resolutions)
    assert stats["accepted_native_profiles"]["profile_count"] == 1
    assert stats["all_successful_calculate_profiles"]["profile_count"] == 2
    assert stats["all_calculate_profiles"]["passed_J8_for_every_available_profile"] is True


def test_generation_reset_invalidates_accepted_evidence(tmp_path: Path) -> None:
    native = tmp_path / "native"
    native.mkdir()
    first_call, first_resolution = _call(0, 0, 0, True, False)
    second_call, second_resolution = _call(1, 1, 1, True, False)
    _write_jsonl(native / "stage25r_ruckig_call_resolutions.jsonl", [first_resolution, second_resolution])
    _write_jsonl(
        native / "stage25r2_trajectory_generation_events.jsonl",
        [
            {"event": "initial_generation", "trajectory_generation": 0},
            {"event": "trajectory_reset", "from_generation": 0, "to_generation": 1, "invalidated_accepted_call_indices": [0]},
        ],
    )
    (native / "stage25r_native_run_summary.json").write_text(
        json.dumps({"moveit_native_return_value": True, "moveit_smoothing_complete": True, "trajectory_generation_final": 1, "trajectory_generation_reset_count": 1}),
        encoding="utf-8",
    )
    audit = generation_audit(native, [first_resolution, second_resolution], 2)
    assert audit["stale_accepted_calls"] == [0]
    assert audit["unique_waypoints_accepted_in_final_generation"] == [1]
    assert audit["all_nominal_segments_reached"] is False


def test_final_replay_rejects_stale_failed_segment(tmp_path: Path) -> None:
    path = tmp_path / "replay.jsonl"
    _write_jsonl(
        path,
        [
            {"segment_index": 0, "ruckig_result": {"result_name": "Working"}, "native_profiles_available": True, "max_abs_jerk": 8.0, "overshoot_at_selected_semantics": {"passed": True}},
            {"segment_index": 1, "ruckig_result": {"result_name": "ErrorInvalidInput"}, "native_profiles_available": False, "max_abs_jerk": None, "overshoot_at_selected_semantics": {"passed": False}},
        ],
    )
    audit = final_replay_audit(path, 2, True)
    assert audit["segments_checked"] == 2
    assert audit["successful_native_results"] == 1
    assert audit["failed_segments"] == [1]


def test_full_nominal_segment_coverage_requires_every_index(tmp_path: Path) -> None:
    native = tmp_path / "native"
    native.mkdir()
    (native / "stage25r_native_run_summary.json").write_text(
        json.dumps({"moveit_smoothing_complete": True, "trajectory_generation_final": 0}),
        encoding="utf-8",
    )
    resolutions = [_call(index, index, 0, True, False)[1] for index in (0, 1, 2)]
    audit = generation_audit(native, resolutions, 3)
    assert audit["first_segment"] == 0
    assert audit["last_segment"] == 2
    assert audit["missing_segments"] == []
    assert audit["all_nominal_segments_reached"] is True


def test_overshoot_bool_is_preserved_by_selected_semantics(tmp_path: Path) -> None:
    path = tmp_path / "replay.jsonl"
    _write_jsonl(
        path,
        [{
            "segment_index": 0,
            "ruckig_result": {"result_name": "Working"},
            "native_profiles_available": True,
            "max_abs_jerk": 8.0,
            "overshoot_at_selected_semantics": {"passed": False},
        }],
    )
    audit = final_replay_audit(path, 1, True)
    assert audit["overshoot_check_applicable"] is True
    assert audit["overshoot_pass"] == "0/1"


def test_frozen_artifact_copy_refuses_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    destination = tmp_path / "destination.bin"
    source.write_bytes(b"frozen-source")
    destination.write_bytes(b"user-owned-destination")
    before_hash = sha256(destination)
    with pytest.raises(FileExistsError):
        safe_copy(source, destination)
    assert sha256(destination) == before_hash


def test_three_rebuild_determinism_requires_all_hashes_to_match() -> None:
    matching = [{"final_replay_hash": "same", "trajectory_hash": "same"} for _ in range(3)]
    mismatching = matching[:2] + [{"final_replay_hash": "different", "trajectory_hash": "same"}]
    fields = ("final_replay_hash", "trajectory_hash")
    assert all(all(item[field] == matching[0][field] for item in matching) for field in fields)
    assert not all(all(item[field] == mismatching[0][field] for item in mismatching) for field in fields)
