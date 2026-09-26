"""Stage 2.8S-R2-H1 geometry accounting regression tests."""

from __future__ import annotations

import csv
from pathlib import Path

from src.stage28sr2_geometry_mapping import source_waypoint_union


ROOT = Path(__file__).resolve().parents[1]
STAGE27_TRACE = ROOT / "outputs/stage27s_native_spline_remediation/stage27s_formal_20260805T130000Z/stage27_process_geometry_trace.csv"
HISTORICAL_STAGE28_TRACE = ROOT / "outputs/stage28sr_corrected_tf_fk_runtime_recapture/stage27_process_geometry_trace.csv"


def load_trace(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_stage27s_formal_trace_unions_both_interval_endpoints() -> None:
    rows = load_trace(STAGE27_TRACE)
    coverage = source_waypoint_union(rows)
    assert len(coverage) == 720
    assert sorted(set(range(720)) - coverage) == []


def test_historical_stage28s_r_trace_preserves_real_capture_gap() -> None:
    rows = load_trace(HISTORICAL_STAGE28_TRACE)
    coverage = source_waypoint_union(rows)
    assert len(coverage) == 716
    assert sorted(set(range(720)) - coverage) == [91, 92, 93, 634]


def test_four_waypoint_union_is_raw_state_only() -> None:
    rows = [
        {"spray_state": "ON", "source_waypoint_0": "90", "source_waypoint_1": "91"},
        {"spray_state": "ON", "source_waypoint_0": "92", "source_waypoint_1": "93"},
        {"spray_state": "ON", "source_waypoint_0": "633", "source_waypoint_1": "634"},
    ]
    assert source_waypoint_union(rows) == {90, 91, 92, 93, 633, 634}
    assert source_waypoint_union([], spray_state=None) == set()
