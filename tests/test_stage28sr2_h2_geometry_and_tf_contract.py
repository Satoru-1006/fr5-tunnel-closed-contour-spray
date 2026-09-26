"""Stage 2.8S-R2-H2 offline contract regressions."""

from __future__ import annotations

from src.stage28sr2_geometry_mapping import (
    SPRAY_DISTANCE_LIMIT_MM,
    SPRAY_NORMAL_LIMIT_DEG,
    TCP_POSITION_LIMIT_MM,
    WAYPOINT_COUNT,
    evaluate_geometry_gate,
    geometry_semantics_root_cause,
    source_waypoint_union,
)
from src.stage28sr2_tf_fk_contract import exact_tf_jointstate_pairing, quaternion_distance_deg


def _geometry_row(index: int, end: int) -> dict[str, str]:
    return {
        "spray_state": "ON",
        "position_error_mm": "1.0",
        "normal_error_deg": "1.0",
        "spray_distance_error_mm": "1.0",
        "source_waypoint_0": str(index),
        "source_waypoint_1": str(end),
    }


def test_positive_stage27s_control_uses_frozen_gate() -> None:
    rows = [_geometry_row(index, index + 1) for index in range(WAYPOINT_COUNT - 1)]
    result = evaluate_geometry_gate(rows, spray_state="ON", source="positive_control")
    assert result["passed"]
    assert result["coverage"] == WAYPOINT_COUNT
    assert result["missing_waypoints"] == []
    assert result["thresholds"] == {
        "tcp_position_error_mm": TCP_POSITION_LIMIT_MM,
        "spray_normal_error_deg": SPRAY_NORMAL_LIMIT_DEG,
        "spray_distance_error_mm": SPRAY_DISTANCE_LIMIT_MM,
        "waypoint_coverage_required": WAYPOINT_COUNT,
    }


def test_historical_negative_control_does_not_get_filled() -> None:
    rows = [_geometry_row(index, index + 1) for index in range(WAYPOINT_COUNT - 1)]
    missing = {91, 92, 93, 634}
    rows = [row for row in rows if int(row["source_waypoint_0"]) not in missing and int(row["source_waypoint_1"]) not in missing]
    result = evaluate_geometry_gate(rows, spray_state="ON", source="historical_stage28sr")
    assert not result["passed"]
    assert result["coverage"] == 716
    assert result["missing_waypoints"] == [91, 92, 93, 634]


def test_synthetic_endpoints_are_union_only() -> None:
    rows = [_geometry_row(90, 91), _geometry_row(92, 93), _geometry_row(633, 634)]
    assert source_waypoint_union(rows) == {90, 91, 92, 93, 633, 634}


def test_q10_root_cause_does_not_infer_tcp_limit_from_false() -> None:
    root = geometry_semantics_root_cause()["stage28sr2_h2_geometry_semantics_root_cause"]
    assert root["tcp_position_limit_mm"] == 6.0
    assert root["spray_distance_limit_mm"] == 5.0
    assert root["raw_failed_conditions_at_frozen_thresholds"]["tcp_position_error_mm"]
    assert not root["raw_failed_conditions_at_frozen_thresholds"]["waypoint_coverage"]


def test_tf_pairing_is_exact_and_q_negation_safe() -> None:
    stamp = {"sec": 7, "nanosec": 11}
    tf = [{"header_stamp": stamp}]
    joint = [{"header_stamp": dict(stamp)}]
    result = exact_tf_jointstate_pairing(tf, joint)
    assert result["passed"]
    assert result["key"] == "exact ROS header timestamp (sec,nanosec)"
    assert quaternion_distance_deg([0, 0, 0, 1], [0, 0, 0, -1]) == 0.0


def test_tf_pairing_fails_closed_for_unmatched_and_ambiguous_stamps() -> None:
    stamp = {"sec": 7, "nanosec": 11}
    result = exact_tf_jointstate_pairing(
        [{"header_stamp": stamp}, {"header_stamp": {"sec": 7, "nanosec": 12}}],
        [{"header_stamp": dict(stamp)}, {"header_stamp": dict(stamp)}],
    )
    assert not result["passed"]
    assert result["ambiguous"] == 1
    assert result["unmatched"] == 1
