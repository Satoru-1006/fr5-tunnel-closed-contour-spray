import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/stage3_h8_software_only_recertification.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("stage3_h8_software_only_recertification", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_h7_authoritative_input_is_immutable_and_expected_hash():
    runner = load_runner()
    path = ROOT / runner.DEFAULT_H7_RELATIVE
    assert path.is_file()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == runner.EXPECTED_H7_SHA256
    rows, audit = runner.read_h7_rows(path)
    assert audit["status"] == "PASSED"
    assert len(rows) == 20822
    assert all(row["joint_names"] == runner.EXPECTED_JOINTS for row in rows)


def test_static_h8_backend_is_mock_only():
    runner = load_runner()
    audit = runner.static_mock_audit(ROOT)
    assert audit["status"] == "PASSED"
    assert audit["backend"] == "mock_components/GenericSystem"
    assert not audit["forbidden_token_hits"]


def test_historical_h8_blocker_is_preserved():
    runner = load_runner()
    historical = runner.old_h8_preservation(ROOT)
    assert historical["preserved"] is True
    old_terminal = json.loads((ROOT / runner.OLD_H8_DIR / "stage3_h8_terminal_certificate.json").read_text())
    assert old_terminal["FIRST_BLOCKER"] == "unable_to_prove_nonphysical_controller_target"


def test_stitched_global_timestamps_are_strictly_monotonic():
    runner = load_runner()
    rows, _ = runner.read_h7_rows(ROOT / runner.DEFAULT_H7_RELATIVE)
    times, offsets = runner.global_message_times(rows)
    assert len(times) == len(rows) == 20822
    assert all(later > earlier for earlier, later in zip(times, times[1:]))
    assert offsets


def test_fjt_duration_and_result_timeout_use_final_global_time():
    runner = load_runner()
    rows, _ = runner.read_h7_rows(ROOT / runner.DEFAULT_H7_RELATIVE)
    times, _ = runner.global_message_times(rows)
    duration = runner.stitched_fjt_duration(rows)
    timeout = runner.result_timeout_seconds(rows)
    assert duration == times[-1]
    assert timeout > duration
    assert timeout > float(rows[-1]["time_from_start_s"]) + runner.RESULT_SAFETY_MARGIN_S


def test_segment_local_timestamp_resets_cannot_shorten_result_deadline():
    runner = load_runner()
    rows, _ = runner.read_h7_rows(ROOT / runner.DEFAULT_H7_RELATIVE)
    local_duration = float(rows[-1]["time_from_start_s"])
    global_duration = runner.stitched_fjt_duration(rows)
    assert global_duration > local_duration
    assert runner.result_timeout_seconds(rows) > local_duration + runner.RESULT_SAFETY_MARGIN_S


def test_h7_raw_bytes_remain_unchanged_during_timeline_audit():
    runner = load_runner()
    path = ROOT / runner.DEFAULT_H7_RELATIVE
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    rows, _ = runner.read_h7_rows(path)
    runner.make_fjt_payload(rows)
    runner.make_message_audits(rows, runner.read_h7_rows(path)[1])
    after = hashlib.sha256(path.read_bytes()).hexdigest()
    assert before == after == runner.EXPECTED_H7_SHA256


def _complete_runtime_fixture(runner):
    return {
        "status": "PASSED",
        "proof": {"runtime_proven": True, "controller_active": True, "fjt_action_available": True},
        "goals_sent": 1,
        "result": {"status": "PASSED", "action_status": 4, "error_code": 0, "feedback_count": 3, "joint_state_count": 3},
        "trajectory_serialization": {"status": "PASSED"},
        "executed_state_recertification": "PASSED",
        "negative_tests": "PASSED",
        "deterministic_replay": "PASSED",
        "regression": "PASSED",
        "clean_exit": "PASSED",
        "physical_backend_used": False,
    }


def test_complete_runtime_evidence_can_produce_h8_pass():
    runner = load_runner()
    rows, audit = runner.read_h7_rows(ROOT / runner.DEFAULT_H7_RELATIVE)
    static = runner.static_mock_audit(ROOT)
    provenance = {"ros_runtime_available": True}
    gate = runner.evaluate_h8_gate(audit, static, provenance, _complete_runtime_fixture(runner), True, {"verified": True})
    assert gate["status"] == "PASSED"
    assert gate["first_blocker"] is None


def test_omitting_any_required_runtime_gate_remains_blocked():
    runner = load_runner()
    _, audit = runner.read_h7_rows(ROOT / runner.DEFAULT_H7_RELATIVE)
    static = runner.static_mock_audit(ROOT)
    provenance = {"ros_runtime_available": True}
    fixture = _complete_runtime_fixture(runner)
    for key in ("executed_state_recertification", "negative_tests", "deterministic_replay", "regression", "clean_exit"):
        omitted = dict(fixture)
        omitted.pop(key)
        gate = runner.evaluate_h8_gate(audit, static, provenance, omitted, True, {"verified": True})
        assert gate["status"] == "BLOCKED", key
