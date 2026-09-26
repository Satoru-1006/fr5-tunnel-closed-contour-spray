"""Regression tests for the H1 raw-CSV freshness proof chain."""

from __future__ import annotations

import csv
from pathlib import Path

from src.stage28ah0_freshness import RAW_REQUIRED_FIELDS, verify_raw_csv, verify_state_samples


def _row(index: int, *, frame_cnt: int | str | None = None, return_code: int = 0) -> dict[str, object]:
    value = index % 256 if frame_cnt is None else frame_cnt
    row: dict[str, object] = {
        "sample_index": index,
        "host_monotonic_ns": 1_000 + index,
        "host_wall_time": f"2026-08-05T00:00:{index:02d}Z",
        "return_code": return_code,
        "frame_head": 0x5A5A,
        "frame_cnt": value,
        "program_state": 0,
        "robot_state": 0,
        "robot_mode": 0,
        "main_code": 0,
        "sub_code": 0,
    }
    row.update({key: 0.0 for key in ("j1_deg", "j2_deg", "j3_deg", "j4_deg", "j5_deg", "j6_deg")})
    return row


def _write_csv(path: Path, rows: list[dict[str, object]], fields: tuple[str, ...] = RAW_REQUIRED_FIELDS) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_raw_csv_positive_progress_and_rollover(tmp_path: Path) -> None:
    rows = [_row(index, frame_cnt=(253 + index) % 256) for index in range(100)]
    path = tmp_path / "raw.csv"
    _write_csv(path, rows)

    result = verify_raw_csv(path)

    assert result["csv_header_valid"] is True
    assert result["raw_rows_parseable"] is True
    assert result["all_GetRobotRealTimeState_return_codes_zero"] is True
    assert result["frame_cnt_sequence_valid_mod_256"] is True
    assert result["fresh_samples_proven"] is True


def test_raw_csv_forward_gap_is_fresh_but_records_dropped_frame(tmp_path: Path) -> None:
    counts = [1, 2, 4, 5, 8] + list(range(9, 104))
    path = tmp_path / "raw.csv"
    _write_csv(path, [_row(index, frame_cnt=count) for index, count in enumerate(counts)])

    result = verify_raw_csv(path)

    assert result["frame_cnt_sequence_valid_mod_256"] is True
    assert result["dropped_frames_detected"] is True
    assert result["fresh_samples_proven"] is True


def test_raw_csv_constant_counter_fails(tmp_path: Path) -> None:
    path = tmp_path / "raw.csv"
    _write_csv(path, [_row(index, frame_cnt=7) for index in range(100)])

    result = verify_raw_csv(path)

    assert result["frame_cnt_sequence_valid_mod_256"] is False
    assert result["fresh_samples_proven"] is False
    assert any(item["kind"] == "frame_counter_constant" for item in result["freshness_anomalies"])


def test_raw_csv_reverse_counter_fails_without_wrap(tmp_path: Path) -> None:
    counts = [10, 9] + list(range(10, 108))
    path = tmp_path / "raw.csv"
    _write_csv(path, [_row(index, frame_cnt=count) for index, count in enumerate(counts)])

    result = verify_raw_csv(path)

    assert result["frame_cnt_sequence_valid_mod_256"] is False
    assert result["fresh_samples_proven"] is False


def test_raw_csv_nonzero_return_code_is_explicit_failure(tmp_path: Path) -> None:
    rows = [_row(index) for index in range(100)]
    rows[37]["return_code"] = 17
    path = tmp_path / "raw.csv"
    _write_csv(path, rows)

    result = verify_raw_csv(path)

    assert result["all_GetRobotRealTimeState_return_codes_zero"] is False
    assert result["sample_count"] == 100
    assert result["fresh_samples_proven"] is False
    assert any(item["kind"] == "GetRobotRealTimeState_return_code_nonzero" for item in result["freshness_anomalies"])


def test_raw_csv_invalid_head_timestamp_and_nonfinite_joint_fail(tmp_path: Path) -> None:
    rows = [_row(index) for index in range(100)]
    rows[2]["frame_head"] = 0x1234
    rows[4]["host_monotonic_ns"] = rows[3]["host_monotonic_ns"]
    rows[6]["j4_deg"] = "nan"
    path = tmp_path / "raw.csv"
    _write_csv(path, rows)

    result = verify_raw_csv(path)

    assert result["frame_head_valid"] is False
    assert result["timestamps_host_monotonic"] is False
    assert result["joint_values_finite"] is False
    assert result["fresh_samples_proven"] is False


def test_state_verifier_requires_return_codes_and_all_frame_counters() -> None:
    samples = [
        {
            "return_code": 0,
            "frame_head": 0x5A5A,
            "frame_cnt": index,
            "host_monotonic_ns": index + 1,
            **{f"j{joint}_deg": 0.0 for joint in range(1, 7)},
        }
        for index in range(100)
    ]
    samples[50].pop("frame_cnt")

    result = verify_state_samples(samples)

    assert result["frame_cnt_present"] is False
    assert result["fresh_samples_proven"] is False

