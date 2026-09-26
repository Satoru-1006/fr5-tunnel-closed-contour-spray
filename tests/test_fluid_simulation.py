from __future__ import annotations

from pathlib import Path
import csv

import numpy as np

from fluid_simulation.particle_spray import (
    MaterialParameters,
    NozzleParameters,
    SimulationParameters,
    rebound_probability,
    simulate_from_trace,
)


def test_rebound_probability_keeps_nonzero_normal_incidence_rebound() -> None:
    normal = float(rebound_probability(90.0))
    oblique = float(rebound_probability(30.0))
    assert 0.08 <= normal <= 0.10
    assert oblique > normal


def test_particle_outcomes_and_mass_are_conserved(tmp_path: Path) -> None:
    trace = tmp_path / "trace.csv"
    fields = [
        "t", "actual_tcp_x", "actual_tcp_y", "actual_tcp_z",
        "tool_z_x", "tool_z_y", "tool_z_z",
        "wall_normal_x", "wall_normal_y", "wall_normal_z",
        "standoff_error_mm",
    ]
    rows = [
        [0.0, 0.0, 0.0, 0.26, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        [1.0, 0.01, 0.0, 0.26, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
    ]
    with trace.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(rows)

    result = simulate_from_trace(
        trace,
        MaterialParameters(2400.0, 40.58, 4.24, 0.0677, "paper", "test"),
        NozzleParameters(25.0, 0.032, 35.0, 0.04, "test"),
        SimulationParameters(superparticles_per_interval=2000, random_seed=7),
    )
    assert result.emitted_particle_count == (
        result.deposited_particle_count
        + result.rebound_particle_count
        + result.domain_escaped_particle_count
    )
    assert np.isclose(result.emitted_mass_kg, 0.04)
    assert abs(result.mass_balance_error_kg) <= 1e-12
    assert 0.05 < result.rebound_fraction < 0.20
    assert result.deposited_fraction < 0.95
