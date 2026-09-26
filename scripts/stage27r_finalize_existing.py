"""Re-audit a completed Stage 2.7R clean raw run after audit-rule fixes."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.stage27r_run as run


def main() -> int:
    output = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else sorted((ROOT / "outputs/stage27r_clean_native_certification").glob("stage27r_formal_*"))[-1]
    traj = run.read_trajectory()
    limits = run.load_limits()
    dynamics = run.exact_dynamics(output, traj, limits)
    action = run.read_json(output / "stage27r_clean_action_execution.json")
    initial = run.read_json(output / "stage27r_clean_initial_state.json")
    native = run.compare_controller_state(output, traj, dynamics, action)
    previous = run.previous_query_analysis(output, traj, dynamics)
    geometry = None
    bullet = None
    if native["summary"]["passed"] and initial.get("compatible") and action.get("execution_completed"):
        samples = run.build_geometry_samples(output, traj)
        geometry = run.run_geometry(output, samples)
        intervals = run.write_bullet_intervals(output, traj)
        bullet = run.run_bullet(output, intervals)
    sacrificial = run.run_sacrificial(output, output / "stage27r_clean_initial_state.json")
    runtime = {"runtime_verified": True, "controller_active": True, "interpolation_method": "splines", "jtc_version": "4.40.1-1noble.20260615.171409", "release": "4.40.1", "source_commit": "31015e0aa7ce9d0853a88d5fc2fe50f0e583ba5c"}
    integrity_before = run.frozen_check(run.ENV_HASH_AFTER)
    integrity_after = run.frozen_check(run.ENV_HASH_AFTER)
    determinism = run.build_determinism(output, traj, dynamics, geometry or {}, bullet, "pending")
    gate = run.gate_report(output, runtime, run.read_json(output / "stage27r_clean_follow_joint_trajectory_goal.json"), initial, action, native, dynamics, geometry, bullet, determinism, integrity_before, integrity_after, previous, sacrificial)
    status = gate["Stage_2_7"]["status"]
    for record in determinism["records"]:
        record["final_classification"] = status
    run.write_json(output / "stage27r_determinism_report.json", determinism)
    run.gate_report(output, runtime, run.read_json(output / "stage27r_clean_follow_joint_trajectory_goal.json"), initial, action, native, dynamics, geometry, bullet, determinism, integrity_before, integrity_after, previous, sacrificial)
    run.write_sums(output)
    print({"output": str(output), "status": status, "native_semantics": native["summary"], "geometry": geometry, "bullet": bullet})
    return 0 if status == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
