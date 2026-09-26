from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.stage4b_rolling_champion import branch_window, gate


def _stage(*, env_collisions: int = 0, self_collisions: int = 0, ccd: int = 0, env_clearance: float = 0.05, self_clearance: float = 0.02) -> dict:
    stats = lambda value: {"min": value, "p90": value, "p95": value, "p99": value, "max": value}
    return {
        "finite_failures": 0,
        "geometry": {
            "environment_collision_cases": env_collisions,
            "self_collision_cases": self_collisions,
            "environment_ccd_failures": ccd,
            "minimum_environment_clearance_m": stats(env_clearance),
            "minimum_self_clearance_m": stats(self_clearance),
        },
        "b1": {
            "singularity_risk_cases_sigma_lt_1e-6": 0,
            "minimum_sigma_min": stats(1.0e-4),
            "maximum_condition_number": stats(100.0),
        },
        "accuracy": {key: stats(0.0) for key in ("terminal_position_error_m", "terminal_orientation_error_rad", "tcp_trajectory_error_max_m", "tcp_trajectory_error_p95_m")},
        "motion_quality": {
            "joint_limit_violations": 0,
            "velocity_limit_violations": 0,
            "acceleration_limit_violations": 0,
            "jerk_limit_violations": 0,
            "jerk_ratio": stats(0.1),
        },
        "collision_failure_case_count": int(env_collisions or self_collisions or ccd),
    }


def test_branch_window_is_local_and_c1_at_endpoints() -> None:
    weights = branch_window()
    assert weights.shape == (181,)
    assert np.all(weights[:75] == 0.0)
    assert np.all(weights[111:] == 0.0)
    assert np.isclose(weights[75], 0.0)
    assert np.isclose(weights[110], 0.0)
    assert np.max(weights) > 0.99


def test_b3_accepts_environment_only_margin_improvement_without_self_regression() -> None:
    parent = _stage(env_clearance=0.01, self_clearance=0.02)
    candidate = _stage(env_clearance=0.08, self_clearance=0.02)
    decision = gate(parent, candidate, "B3", [])
    assert decision["status"] == "PROMOTE"
    assert decision["checks"]["b3_environment_clearance_improves"] is True
    assert decision["checks"]["b3_self_clearance_improves"] is False
    assert decision["checks"]["b3_clearance_floor_non_regressive"] is True


def test_gate_rejects_a_real_collision_regression() -> None:
    parent = _stage()
    candidate = _stage(env_collisions=1)
    decision = gate(parent, candidate, "B2", [])
    assert decision["status"] == "REJECT"
    assert "no_environment_collision_regression" in decision["reasons"]
