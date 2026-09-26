from pathlib import Path

from scripts import stage3_h7_2 as h72


def test_authoritative_nine_segment_order_and_counts():
    audit = h72.authoritative_segments()
    assert audit["status"] == "PASSED"
    assert audit["authoritative_segment_order"] == [0, 1000, 1, 1001, 2, 1002, 3, 1003, 4]
    assert [row["input_waypoint_count"] for row in audit["segments"]] == [221, 12, 40, 426, 4, 241, 298, 361, 25]
    assert sum(row["input_waypoint_count"] for row in audit["segments"]) == 1628


def test_h6_4_is_formally_ready():
    audit = h72.audit_h6_4()
    assert audit["status"] == "PASSED"
    assert audit["terminal"]["STAGE_3_H6_4"] == "PASSED"
    assert audit["terminal"]["DETERMINISTIC_REPLAY"] == "3/3"


def test_jerk_limits_are_predeclared_and_not_moveit_defaults():
    audit = h72.jerk_limits_audit()
    assert audit["ruckig_authorized"] is True
    assert audit["jerk_limits_rad_s3"] == [8.0] * 6
    assert audit["moveit_default_jerk_accepted"] is False
    assert "not claimed as FAIRINO manufacturer" in audit["provenance_classification"]


def test_runner_has_no_robot_execution_calls():
    source = Path(h72.__file__).read_text(encoding="utf-8")
    native = (h72.ROOT / "ros2_moveit_bridge/stage3_h7_2_native.py").read_text(encoding="utf-8")
    forbidden = (".send_goal_async(", "from control_msgs", "import control_msgs", "move_group.execute(")
    for token in forbidden:
        assert token not in source + native


def test_totg_parameters_are_per_segment_native_moveit():
    parameters = h72.totg_parameters()
    assert "TimeOptimalTrajectoryGeneration" in parameters["implementation"]
    assert parameters["parameterization_scope"].startswith("nine independent RobotTrajectory")
    assert parameters["path_tolerance"] == 0.00025
    assert parameters["resample_dt"] == 0.01
