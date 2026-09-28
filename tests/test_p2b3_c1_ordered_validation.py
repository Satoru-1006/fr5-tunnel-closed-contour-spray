from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ros2_moveit_bridge"))

from p2b3_c1_ordered_validation import (  # noqa: E402
    local_segment_timing,
    project_open_polyline,
    station_deltas,
    waypoint_semantics,
)


def test_projection_reports_real_backstep_without_monotone_repair() -> None:
    path = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]
    projections = [
        project_open_polyline([0.9, 0.8, 0.0], path),
        project_open_polyline([0.8, 0.1, 0.0], path),
    ]
    assert [row.segment_index for row in projections] == [1, 0]
    assert station_deltas(projections)[0] < 0.0


def test_local_timing_keeps_single_slow_segment_visible() -> None:
    report = local_segment_timing(
        [[0.0, 0.0, 0.0], [0.003, 0.0, 0.0], [0.06753777184104506, 0.0, 0.0], [0.07053777184104506, 0.0, 0.0]],
        [0.0, 1.0, 29.633258105, 30.633258105],
    )
    assert len(report) == 3
    assert report[1].speed_m_s == pytest.approx(0.0022539444)
    assert report[1].speed_band_status == "BELOW_CONFIGURED_BAND"
    assert not report[1].dwell
    assert report[0].speed_band_status == "WITHIN_CONFIGURED_BAND"
    assert report[2].speed_band_status == "WITHIN_CONFIGURED_BAND"


@pytest.mark.parametrize(
    "index,solved,retained,role",
    [(0, False, True, "RETAINED_OBSERVED_WARM_START"), (14, False, True, "RETAINED_OBSERVED_WARM_START"),
     (15, False, True, "RETAINED_OBSERVED_WARM_START"), (16, True, False, "DLS_TARGET_SOLVED"),
     (17, True, False, "DLS_TARGET_SOLVED"), (180, True, False, "DLS_TARGET_SOLVED")],
)
def test_legacy_lineage_distinguishes_observed_prefix_from_solved_target(
    index: int, solved: bool, retained: bool, role: str,
) -> None:
    row = waypoint_semantics(index, corrected_candidate=False)
    assert row["target_solved"] is solved
    assert row["retained_seed"] is retained
    assert row["solver_role"] == role
    assert row["expected_target_index"] == index


def test_corrected_lineage_marks_every_target_solved_and_only_row_zero_seeded() -> None:
    first = waypoint_semantics(0, corrected_candidate=True)
    second = waypoint_semantics(1, corrected_candidate=True)
    assert first["target_solved"] and second["target_solved"]
    assert first["upstream_source_row"] == 0
    assert second["upstream_source_row"] is None
    assert not first["retained_seed"] and not second["retained_seed"]


def test_bad_path_and_time_inputs_fail_closed() -> None:
    with pytest.raises(ValueError, match="at_least_two_vertices"):
        project_open_polyline([0, 0, 0], [[0, 0, 0]])
    with pytest.raises(ValueError, match="timestamps_must_increase"):
        local_segment_timing([[0, 0, 0], [1, 0, 0]], [1, 1])
