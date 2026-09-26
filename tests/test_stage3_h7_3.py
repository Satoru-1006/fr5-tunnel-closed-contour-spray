from pathlib import Path

from scripts import stage3_h7_3 as h73


def test_target55_filter_exposes_the_authoritative_reversal_cusp():
    audit = h73.target55_cusp_audit()
    assert audit["status"] == "PASSED"
    assert audit["classification"] == "TARGET_55_FILTER_EXPOSED_REVERSAL_CUSP"
    assert audit["local_194_filtered"] is True
    assert audit["filtered_segment_local_triple"] == [192, 193, 195]
    assert audit["h6_4_storage_triple"] == [457, 458, 460]
    assert audit["process_order_triple"] == [1136, 1137, 1139]
    assert abs(audit["local_193_to_194_joint_space_displacement_rad"] - 9.25217933937491e-7) < 1e-15
    assert abs(audit["local_193_to_194_max_single_joint_change_rad"] - 6.834498267327405e-7) < 1e-15
    assert audit["cosine"] <= -1.0 + h73.ANGLE_TOLERANCE
    assert abs(audit["turn_angle_deg"] - 180.0) < 1e-12


def test_h7_3_derives_ten_primitives_without_changing_waypoints():
    order = h73.authoritative_primitives()
    assert order["status"] == "PASSED"
    assert order["h6_4_process_segment_count"] == 9
    assert order["h7_3_time_parameterization_primitive_count"] == 10
    assert order["authoritative_primitive_order"] == [0, 1000, 1, 1001, 2, 1002, "3A", "3B", 1003, 4]
    assert [item["input_waypoint_count"] for item in order["primitives"]] == h73.PRIMITIVE_COUNTS
    assert sum(item["input_waypoint_count"] for item in order["primitives"]) == 1628
    assert order["does_not_rewrite_h6_4_segmentation"] is True


def test_filtered_path_audit_has_no_remaining_180_degree_turn():
    audit = h73.pre_totg_filtered_path_audit(h73.authoritative_primitives())
    assert audit["status"] == "PASSED"
    assert audit["moveit_filtered_180_turn_count"] == 0
    assert len(audit["primitive_audits"]) == 10


def test_h7_2_parameters_are_reused_without_relaxation():
    parameters = h73.totg_parameters()
    assert parameters["status"] == "FROZEN_FROM_AUTHORITATIVE_H7_2"
    assert parameters["path_tolerance"] == 0.00025
    assert parameters["resample_dt"] == 0.01
    assert parameters["min_angle_change"] == 0.0005
    assert parameters["velocity_scaling_factor"] == 0.15
    assert parameters["acceleration_scaling_factor"] == 0.15
    assert parameters["h7_2_parameter_mismatches"] == {}


def test_authoritative_h7_2_blocked_input_is_recognized():
    audit = h73.audit_h7_2()
    assert audit["status"] == "PASSED"
    assert audit["terminal"]["STAGE_3_H7_2"] == "BLOCKED"
    assert audit["terminal"]["FIRST_BLOCKER"] == "segment_3:native_totg_path_requires_180_degree_turn"


def test_h7_3_runner_has_no_robot_execution_calls():
    source = Path(h73.__file__).read_text(encoding="utf-8")
    native = (h73.ROOT / "ros2_moveit_bridge/stage3_h7_3_native.py").read_text(encoding="utf-8")
    forbidden = (".send_goal_async(", "from control_msgs", "import control_msgs", "move_group.execute(")
    for token in forbidden:
        assert token not in source + native


def test_ruckig_no_solution_log_is_fail_closed_even_when_api_returns_true(tmp_path):
    primitives = [
        {
            "primitive_id": primitive_id,
            "spray_state": "SPRAY_ON",
            "post_totg_status": "PASSED",
            "post_totg_process": {"status": "PASSED"},
            "post_totg_collision": {"status": "PASSED"},
            "ruckig_status": "PASSED",
            "status": "PASSED",
        }
        for primitive_id in h73.PRIMITIVE_ORDER
    ]
    native = {"primitive_results": primitives, "ruckig_native_no_solution_log_count": 10}
    _totg, _post_totg, ruckig, post_ruckig = h73.aggregate_native(tmp_path, native, {})
    assert ruckig["native_api_returned_true_primitives"] == 10
    assert ruckig["native_smoothing_completion_status"] == "BLOCKED"
    assert ruckig["certification_status"] == "BLOCKED"
    assert post_ruckig["status"] == "BLOCKED"
