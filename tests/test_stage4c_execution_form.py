"""Focused regression tests for the D48 execution-form measurement driver."""

from pathlib import Path

import numpy as np

from tools.stage4c_execution_form import compare_native_replay, project_reference, read_matrix, read_native_trajectory, stats


def test_read_matrix_uses_named_joint_order(tmp_path: Path) -> None:
    path = tmp_path / "case.csv"
    path.write_text(
        "waypoint,j1_q,j2_q,j3_q,j4_q,j5_q,j6_q,metadata\n"
        "0,1,2,3,4,5,6,first\n"
        "1,7,8,9,10,11,12,second\n",
        encoding="utf-8",
    )

    result = read_matrix(path)

    np.testing.assert_allclose(result, [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])


def test_project_reference_preserves_endpoints_and_interpolates_targets() -> None:
    source_q = np.asarray([[0.0] * 6, [1.0] * 6])
    target_p = np.asarray([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    target_quat = np.asarray([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]])

    desired_p, desired_quat, projection_error = project_reference(
        np.asarray([[0.0] * 6, [0.5] * 6, [1.0] * 6]),
        source_q,
        target_p,
        target_quat,
    )

    np.testing.assert_allclose(desired_p[[0, 2]], target_p)
    np.testing.assert_allclose(projection_error, 0.0)
    assert np.isclose(np.linalg.norm(desired_quat[1]), 1.0)


def test_project_reference_maps_native_samples_to_shorter_waypoint_path() -> None:
    source_q = np.asarray([[0.0] * 6, [0.5] * 6, [1.0] * 6, [1.5] * 6])
    actual_q = np.asarray([[0.0] * 6, [0.5] * 6, [1.0] * 6, [1.5] * 6])
    target_p = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    target_quat = np.asarray([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]])

    desired_p, desired_quat, projection_error = project_reference(actual_q, source_q, target_p, target_quat)

    np.testing.assert_allclose(desired_p[:, 0], [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
    np.testing.assert_allclose(desired_p[:, 1:], 0.0)
    np.testing.assert_allclose(projection_error, 0.0)
    np.testing.assert_allclose(desired_quat, target_quat[[0, 0, 0, 0]])


def test_stats_reports_unavailable_without_fabricating_values() -> None:
    result = stats([None, float("nan")], "m")

    assert result == {"count": 0, "finite_count": 0, "status": "UNAVAILABLE", "unit": "m"}


def test_native_trajectory_parser_requires_strict_time_order(tmp_path: Path) -> None:
    path = tmp_path / "native.csv"
    fields = ["t"] + [f"j{i}_{suffix}" for suffix in ("q", "dq", "ddq", "jerk") for i in range(1, 7)]
    path.write_text(
        ",".join(fields) + "\n"
        + "0," + ",".join(["0"] * 24) + "\n"
        + "0," + ",".join(["0"] * 24) + "\n",
        encoding="utf-8",
    )

    try:
        read_native_trajectory(path)
    except RuntimeError as exc:
        assert "native_trajectory_invalid" in str(exc)
    else:
        raise AssertionError("non-monotonic trajectory was accepted")


def test_replay_comparison_detects_numeric_change(tmp_path: Path) -> None:
    def make(path: Path, offset: float) -> None:
        fields = ["t"] + [f"j{i}_{suffix}" for suffix in ("q", "dq", "ddq", "jerk") for i in range(1, 7)]
        rows = [
            [0.0] + [offset] + [0.0] * 23,
            [1.0] + [offset] + [0.0] * 23,
        ]
        with path.open("w", encoding="utf-8", newline="") as stream:
            stream.write(",".join(fields) + "\n")
            for row in rows:
                stream.write(",".join(str(value) for value in row) + "\n")

    first_path = tmp_path / "first.csv"
    second_path = tmp_path / "second.csv"
    make(first_path, 0.0)
    make(second_path, 1.0e-6)
    first = {"cases": [{"case_id": "c", "status": "PASS", "trajectory_csv": str(first_path)}]}
    second = {"cases": [{"case_id": "c", "status": "PASS", "trajectory_csv": str(second_path)}]}

    result = compare_native_replay(first, second, tmp_path)

    assert result["status"] == "FAIL"
