from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from tools.d57_system_closure import automatic_hotspots, c4_bump


def test_c4_trust_region_has_zero_boundary_state_derivatives() -> None:
    time = np.asarray((0.0, 1.0, 2.0), dtype=float)
    q, dq, ddq, jerk = c4_bump(time, center=1.0, half_width=1.0, amplitude=0.005)
    assert q[0] == 0.0 and dq[0] == 0.0 and ddq[0] == 0.0 and jerk[0] == 0.0
    assert q[-1] == 0.0 and dq[-1] == 0.0 and ddq[-1] == 0.0 and jerk[-1] == 0.0
    assert q[1] > 0.0


def test_automatic_hotspot_selection_preserves_objective_diversity(tmp_path: Path) -> None:
    clearance = tmp_path / "clearance.csv"
    jacobian = tmp_path / "jacobian.csv"
    vectors = tmp_path / "vectors.csv"
    with clearance.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case_id", "waypoint", "robot_world_distance_m", "self_distance_m"])
        writer.writerow(["adversarial_0100", 0, 0.2, 0.01])
    with jacobian.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case_id", "waypoint", "sigma_min", "condition_number"])
        writer.writerow(["adversarial_0100", 0, 0.001, 1000.0])
    with vectors.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case_id", "waypoint", *[f"V_sigma_min_{i}" for i in range(6)]])
        writer.writerow(["adversarial_0100", 0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    selected = automatic_hotspots(clearance, jacobian, vectors, limit=4)

    assert {row["objective"] for row in selected} == {
        "self_clearance", "world_clearance", "singularity", "conditioning"
    }
    assert all(row["selected_joint_by_max_abs_direction"] == "j1" for row in selected)
