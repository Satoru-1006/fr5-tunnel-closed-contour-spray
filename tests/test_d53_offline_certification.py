from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from tools.d53_offline_certification import MANIFEST, SRDF, URDF, native_trajectory, relative_quaternion_log, srdf_acm_audit, wsl_to_windows


def test_authoritative_manifest_is_the_181_point_d52_finalist() -> None:
    rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8")))
    assert len(rows) == 12
    assert {row["family"] for row in rows} == {
        "REGRESSION", "NORMAL", "BOUNDARY", "COLLISION_SENSITIVE", "ADVERSARIAL", "PERTURBATION"
    }


def test_native_reader_requires_named_state_columns() -> None:
    trajectory = native_trajectory(wsl_to_windows(rows_for_test()[0]["trajectory_csv"]))
    assert trajectory["q"].shape[1] == 6
    assert trajectory["dq"].shape == trajectory["q"].shape
    assert trajectory["ddq"].shape == trajectory["q"].shape
    assert trajectory["jerk"].shape == trajectory["q"].shape


def rows_for_test() -> list[dict[str, str]]:
    result = list(csv.DictReader(MANIFEST.open(encoding="utf-8")))
    value = result[0]["trajectory_csv"]
    if value.startswith("/mnt/d/"):
        value = "D:/" + value[6:].replace("/", "\\")
    result[0]["trajectory_csv"] = value
    return result


def test_acm_audits_full_robot_collision_pair_universe() -> None:
    audit = srdf_acm_audit()
    assert audit["collision_link_count"] == 7
    assert audit["pair_universe_count"] == 21
    assert audit["allowed_pair_count"] == 11
    assert audit["srdf_disabled_pair_count"] == 12
    assert audit["srdf_disabled_pair_universe_count"] == 11
    assert audit["srdf_disabled_outside_collision_universe"] == ["spray_tcp_link|wrist3_link"]
    assert audit["checked_pair_count"] == 10
    assert audit["status"] == "PASS_COMPLETE_PAIR_UNIVERSE_AUDITED"


def test_authoritative_geometry_inputs_exist() -> None:
    assert URDF.is_file()
    assert SRDF.is_file()


def test_relative_quaternion_log_has_no_absolute_log_wrap_spike() -> None:
    time = np.linspace(0.0, 1.0, 101)
    angle = 1.25 * time
    quaternion = np.column_stack((np.zeros_like(time), np.zeros_like(time), np.sin(angle / 2.0), np.cos(angle / 2.0)))
    angular_velocity = relative_quaternion_log(quaternion, time)
    assert float(np.max(np.abs(angular_velocity[:, 2] - 1.25))) < 1.0e-10
    assert float(np.max(np.abs(angular_velocity[:, :2]))) < 1.0e-12
