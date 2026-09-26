import pytest

from src.stage3_h12_r7a import classify_sample, distribution, exact_reproduction, first_blocker, lineage_breakdown


def _row(position=0.0012, standoff=0.0002):
    return {
        "standoff_error_m": standoff, "standoff_limit_m": 0.01,
        "normal_deviation_deg": 0.1, "normal_limit_deg": 5.0,
        "tcp_position_error_m": position, "tcp_position_limit_m": 0.001,
        "tcp_orientation_error_deg": 0.1, "tcp_orientation_limit_deg": 1.0,
    }


def test_exact_reproduction_is_strict():
    assert exact_reproduction(2304)
    assert not exact_reproduction(2303)


def test_constraint_classification_is_complete_and_deterministic():
    rows = [_row(), _row(position=0.0015)]
    assert classify_sample(rows[0]) == "TCP_POSITION"
    first = distribution(rows)
    assert first == distribution(rows)
    assert first["SPRAY_CONSTRAINT_CLASSES"] == {"TCP_POSITION": 2}
    assert first["class_count_sum"] == first["RAW_VIOLATION_RECORDS"] == 2


def test_compound_failure_is_not_silently_dropped():
    assert classify_sample(_row(standoff=0.02)).startswith("COMPOUND:")
    with pytest.raises(ValueError):
        classify_sample(_row(position=0.0005))


def test_fail_closed_blocker_order():
    args = dict(reproduced=True, inventory_complete=True, lineage_complete=True, immutable=True, replay=True, tests=True, native_regression=True)
    assert first_blocker(**args) == "none"
    args["reproduced"] = False
    assert first_blocker(**args) == "h12_r6_spray_violation_reproduction_mismatch"


def test_mixed_lineage_breakdown_matches_exact_joint_state_per_unit():
    pre = [{"unit_id": "a", "joint_values": [0.0] * 6}, {"unit_id": "a", "joint_values": [1.0] * 6}]
    post = [{"unit_id": "a", "joint_values": [0.0] * 6}, {"unit_id": "a", "joint_values": [0.5] * 6}, {"unit_id": "b", "joint_values": [0.0] * 6}]
    assert lineage_breakdown(pre, post) == {"PRE_EXISTING_PERSISTING": 1, "TOTG_INDUCED": 2, "POST_TOTG_TOTAL": 3}
