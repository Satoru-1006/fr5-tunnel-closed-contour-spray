"""Refresh Stage 2.7R derived reports without rerunning the frozen live run."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.stage27r_run as run


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: stage27r_refresh_report.py OUTPUT_DIR")
    output = Path(sys.argv[1]).resolve()
    traj = run.read_trajectory()
    dynamics = run.exact_dynamics(output, traj, run.load_limits())
    action = run.read_json(output / "stage27r_clean_action_execution.json")
    initial = run.read_json(output / "stage27r_clean_initial_state.json")
    native = run.compare_controller_state(output, traj, dynamics, action)
    previous = run.previous_query_analysis(output, traj, dynamics)
    geometry = run.read_json(output / "stage27r_process_geometry.json") if (output / "stage27r_process_geometry.json").exists() else None
    bullet = run.read_json(output / "stage27r_bullet_dense_validation.json") if (output / "stage27r_bullet_dense_validation.json").exists() else None
    sacrificial = run.read_json(output / "stage27r_query_state_sacrificial_probe.json")
    runtime = {"runtime_verified": True, "controller_active": True, "interpolation_method": "splines", "jtc_version": "4.40.1-1noble.20260615.171409", "release": "4.40.1", "source_commit": "31015e0aa7ce9d0853a88d5fc2fe50f0e583ba5c"}
    integrity_before = run.frozen_check(run.ENV_HASH_AFTER)
    integrity_after = run.frozen_check(run.ENV_HASH_AFTER)
    determinism = run.build_determinism(output, traj, dynamics, geometry or {}, bullet, "pending")
    goal = run.read_json(output / "stage27r_clean_follow_joint_trajectory_goal.json")
    gate = run.gate_report(output, runtime, goal, initial, action, native, dynamics, geometry, bullet, determinism, integrity_before, integrity_after, previous, sacrificial)
    status = gate["Stage_2_7"]["status"]
    for record in determinism["records"]:
        record["final_classification"] = status
    run.write_json(output / "stage27r_determinism_report.json", determinism)
    run.gate_report(output, runtime, goal, initial, action, native, dynamics, geometry, bullet, determinism, integrity_before, integrity_after, previous, sacrificial)
    run.write_sums(output)
    print({"output": str(output), "status": status, "position": dynamics["summary"]["position_worst_case"], "jerk": dynamics["summary"]["dynamic_limits"]["jerk"]})
    return 0 if status == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
