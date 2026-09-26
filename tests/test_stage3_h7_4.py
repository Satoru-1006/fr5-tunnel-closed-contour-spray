from pathlib import Path

from scripts import stage3_h7_4 as h74


def test_h7_4_freezes_authoritative_h7_3_and_8_rad_jerk():
    assert h74.H73.name == "stage3_h7_3_target55_controlled_stop_20260809T170900Z"
    params = h74.h72.load_json(h74.H73 / "stage3_h7_3_totg_parameters.json")
    assert params["velocity_scaling_factor"] == 0.15
    assert params["acceleration_scaling_factor"] == 0.15
    assert params["path_tolerance"] == 0.00025
    assert params["resample_dt"] == 0.01
    assert params["min_angle_change"] == 0.0005


def test_native_worker_has_no_execution_api():
    source = (h74.ROOT / "ros2_moveit_bridge/stage3_h7_4_native.py").read_text(encoding="utf-8")
    forbidden = ("send_goal_async(", "FollowJointTrajectory", "move_group.execute(", "control_msgs")
    assert all(token not in source for token in forbidden)


def test_interposer_records_native_validation_and_strict_completion():
    source = h74.INTERPOSER.read_text(encoding="utf-8")
    assert "validate_input(input_before, true, false)" in source
    assert "validate_input(input_before, false, true)" in source
    assert "validate_input(input_before, true, true)" in source
    assert "stage25r_native_run_summaries.jsonl" in source
    assert "successful(last_result) && smoothing_complete && !duration_ceiling_hit" in source


def test_strict_terminal_contract_is_fail_closed():
    source = Path(h74.__file__).read_text(encoding="utf-8")
    assert 'true_complete == 10' in source
    assert 'log_count == 0' in source
    assert 'result_errors == 0' in source
    assert 'READY_FOR_STAGE_3_H8="YES"' in source
