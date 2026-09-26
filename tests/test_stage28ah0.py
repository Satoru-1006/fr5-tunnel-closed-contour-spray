"""Offline-only guards for Stage 2.8A-H0."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage28ah0_freshness import verify_frame_counter, verify_state_samples


OUT = ROOT / "outputs" / "stage28ah0_offline_readonly_probe"


def test_default_probe_artifact_is_network_off_and_counters_zero() -> None:
    counters = json.loads((OUT / "stage28ah0_probe_counters.json").read_text(encoding="utf-8"))
    assert counters["network_connection_attempted"] is False
    assert counters["real_FAIRINO_robot_connection"] is False
    for key, value in counters.items():
        if key.endswith("_calls"):
            assert value == 0, key
    assert counters["forbidden_motion_calls"] == 0
    assert counters["forbidden_state_change_calls"] == 0


def test_live_flag_is_explicit_and_probe_source_has_no_forbidden_api_names() -> None:
    source = (ROOT / "tools" / "stage28ah_fairino_readonly_probe.cpp").read_text(encoding="utf-8")
    assert "--real-hardware-readonly" in source
    for name in ["RobotEnable", "ServoJ", "ServoCart", "MoveJ", "MoveL", "MoveC", "Circle", "Spline", "NewSpline", "StopMotion", "SetSpeed", "SetToolCoord", "SetWObjCoord"]:
        assert name not in source


def test_allowlist_excludes_forbidden_vendor_api() -> None:
    value = json.loads((OUT / "stage28ah0_fairino_sdk_allowlist.json").read_text(encoding="utf-8"))
    apis = value["apis"]
    for name in ["RobotEnable", "Mode", "DragTeachSwitch", "ServoJ", "MoveJ", "MoveL", "MoveC", "Circle", "Spline", "StopMotion", "SetSpeed"]:
        assert name not in apis
    assert all("vendor_documented_semantics" in entry and "binary_side_effect_proven" in entry for entry in apis.values())


def test_forbidden_symbols_are_zero() -> None:
    value = json.loads((OUT / "stage28ah0_probe_imported_symbols.json").read_text(encoding="utf-8"))
    assert value["forbidden_vendor_symbol_reference_count"] == 0
    assert value["source_level_forbidden_api_reference_count"] == 0


def test_synthetic_frame_counter_pass_cases() -> None:
    for counts in ([0, 1, 2, 3, 4], [253, 254, 255, 0, 1], [1, 3, 4, 7]):
        result = verify_frame_counter(counts)
        assert result["fresh"] is True
        assert result["frame_cnt_sequence_valid_mod_256"] is True
    assert verify_frame_counter([1, 3, 4, 7])["dropped_frames_detected"] is True


def test_constant_frame_counter_fails() -> None:
    result = verify_frame_counter([10, 10, 10, 10])
    assert result["fresh"] is False
    assert result["frame_cnt_sequence_valid_mod_256"] is False


def test_constant_joint_values_with_progressing_counter_are_fresh() -> None:
    samples = [
        {"return_code": 0, "frame_head": 0x5A5A, "frame_cnt": index % 256, "host_monotonic_ns": index + 1, **{f"j{joint}_deg": 0.0 for joint in range(1, 7)}}
        for index in range(100)
    ]
    result = verify_state_samples(samples)
    assert result["all_GetRobotRealTimeState_return_codes_zero"] is True
    assert result["fresh_samples_proven"] is True
    assert result["position_constancy_is_not_stale"] is True


def test_unexplained_reverse_is_not_a_wrap() -> None:
    result = verify_frame_counter([10, 9, 10])
    assert result["fresh"] is False
    assert result["anomalies"]


def test_amendment_does_not_modify_original_stage28ar() -> None:
    amendment = json.loads((OUT / "stage28ar_gate_report_amendment_v1.json").read_text(encoding="utf-8"))
    for name, record in amendment["original_artifacts"].items():
        path = Path(record["path"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
    assert amendment["original_artifact_modified"] is False


def test_version_selection_is_deferred_without_real_controller_version() -> None:
    value = json.loads((OUT / "stage28ah0_v397_v398_diff.json").read_text(encoding="utf-8"))
    assert value["compatibility_evidence"]["version_selection_possible_without_real_controller_version"] is False


def test_controller_manager_overlay_is_zero_activation_shape() -> None:
    overlay = (OUT / "stage28ah0_controller_manager_zero_activation_overlay.yaml").read_text(encoding="utf-8").lower()
    assert "hardware_components_initial_state" in overlay
    assert "unconfigured:" in overlay
    assert "fairino5_hardware" in overlay
    assert "inactive: []" in overlay
    assert "shutdown_on_initial_state_failure: true" in overlay
    assert "spawner" not in overlay
    assert "joint_trajectory_controller" not in overlay
    assert "moveit" not in overlay


def test_frozen_hashes_and_final_safety_gate() -> None:
    value = json.loads((OUT / "stage28ah0_gate_report.json").read_text(encoding="utf-8"))
    assert value["frozen_hashes_unchanged"] is True
    assert value["original_stage28ar_modified"] is False
    assert value["real_robot_connection_attempted"] is False
    assert value["motion_commands_sent"] == 0
