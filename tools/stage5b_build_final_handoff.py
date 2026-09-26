"""Build the concise Stage 5B shadow handoff from recorded evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/STAGE5B_SHADOW")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    audit = load(out / "STAGE5B_EXECUTION_AUDIT.json")
    blocker = load(out / "STAGE5B_NOMINAL_REPLAY_BLOCKER.json")
    authority = load(out / "STAGE5B_NOMINAL_WORLD_AUTHORITY.json")
    dense = load(out / "branch_graph_search_dense_large_detours/STAGE5B_BRANCH_GRAPH_SEARCH.json")
    campaigns = []
    for path in sorted(out.glob("branch_graph_search*/STAGE5B_BRANCH_GRAPH_SEARCH.json")):
        item = load(path)
        bottlenecks = [float(x["minimum_bottleneck_step_deg"]) for x in item.get("experiments", []) if x.get("minimum_bottleneck_step_deg") is not None]
        fixed = item.get("fixed_source_endpoints") is True
        campaigns.append({"path": str(path.relative_to(ROOT)), "status": item.get("status") if fixed else "DIAGNOSTIC_NOT_ADMISSIBLE", "experiments": len(item.get("experiments", [])), "min_bottleneck_deg": min(bottlenecks) if bottlenecks else None, "fixed_source_endpoints": item.get("fixed_source_endpoints"), "admissible_for_promotion": bool(fixed and item.get("status") == "PASS")})
    report = {
        "schema_version": "stage5b-final-shadow-handoff-v1",
        "status": "BLOCKED_EXACT_CONTINUITY_AND_NOMINAL_PHYSICS",
        "software_control_loop": {
            "gazebo_gz_ros2_control": "configured_and_activated",
            "follow_joint_trajectory": audit["execution"]["action_result"],
            "x_minus_050_full_replay": "completed_with_result_code_0",
        },
        "continuity_gate": {
            "status": "UNRESOLVED",
            "gate_max_step_deg": audit["trajectory"]["continuity_gate_max_step_deg"],
            "source_max_raw_step_deg": audit["trajectory"]["max_raw_step_deg"],
            "source_max_wrap_aware_step_deg": audit["trajectory"]["max_wrap_aware_step_deg"],
            "intervals_over_gate": audit["trajectory"]["intervals_over_wrap_aware_gate"],
            "blocking_interval": audit["trajectory"]["wp66_to_wp67"],
            "dense_fixed_endpoint_campaign": {
                "status": dense["status"],
                "experiments": len(dense["experiments"]),
                "subdivisions": dense["segment"]["subdivisions"],
                "minimum_bottleneck_step_deg": min(float(x["minimum_bottleneck_step_deg"]) for x in dense["experiments"] if x.get("minimum_bottleneck_step_deg") is not None),
            },
        },
        "x_minus_050_shadow_execution": {
            "status": "COMPLETED_BUT_NOT_PROMOTABLE",
            "action_result": audit["execution"]["action_result"],
            "formal_controller_rows": audit["execution"]["formal_controller_rows"],
            "formal_joint_state_rows": audit["execution"]["formal_joint_state_rows"],
            "measured_execution_duration_s": 705.650151562,
            "controller_error_max_abs_rad": max(audit["execution"]["controller_error_max_abs_rad_by_joint"]),
            "final_endpoint_error_max_abs_rad": audit["execution"]["final_endpoint_error_max_abs_rad"],
            "interpretation": "JTC spline reference overshoot exposes the candidate q/dq/ddq/timestamp incompatibility after the branch cut; action success is not a trajectory-quality PASS.",
        },
        "nominal_world": {
            "authority_status": authority["status"],
            "base_to_tunnel_transform": authority["nominal_simulation_world"]["base_to_tunnel_transform"],
            "full_replay_status": blocker["status"],
            "blocker": blocker["decision"],
            "initial_drift_max_rad": blocker["initial_state"]["measured_max_pre_goal_drift_rad"],
            "clock_probe_s": blocker["execution_observation"]["clock_sample_s"],
        },
        "claim_fence": authority["claim_fence"],
        "campaigns": campaigns,
        "protected_state": {
            "canonical_or_protected_baseline_modified": False,
            "all_stage5b_outputs_are_shadow_or_diagnostic": True,
            "promotion": "NO_PROMOTION",
        },
        "next_stage4b_targets": [
            "Find a fixed-endpoint IK/task-space route through wp66->wp67 with every adjacent joint step <=20 deg.",
            "Regenerate q/dq/ddq/timestamps jointly so the native JTC spline does not overshoot or drive into joint limits.",
            "Resolve nominal installation transform and run collision/clearance with a real backend before any physical claim.",
        ],
    }
    json_path = out / "STAGE5B_FINAL_HANDOFF.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md = f"""# FR5 Stage 5B final shadow handoff

## Final decision

`STAGE5B_STATUS = {report['status']}`

The ROS 2 Jazzy + Gazebo Sim + `gz_ros2_control` software loop is operational. The `x=-0.50 m` shadow completed a single 181-point `FollowJointTrajectory` goal with result code 0. This is execution evidence only; it is not a continuity, collision, clearance, physical-installation, or hardware-safety PASS.

The exact continuity gate remains unresolved. The source has one wrap-aware interval over the 20 degree gate: wp66 -> wp67 reaches {report['continuity_gate']['blocking_interval']['wrapped_max_abs_deg']:.6f} degrees (raw j6 delta {report['continuity_gate']['blocking_interval']['raw_delta_deg'][5]:.6f} degrees). A dense 128-subdivision, fixed-endpoint campaign with {report['continuity_gate']['dense_fixed_endpoint_campaign']['experiments']} experiments also returned `UNRESOLVED`; its best bottleneck was {report['continuity_gate']['dense_fixed_endpoint_campaign']['minimum_bottleneck_step_deg']:.6f} degrees.

The full shadow replay exposed a second defect: the controller accepted the goal and reached the endpoint, but native JTC spline interpolation of the submitted q/dq/ddq/timestamp sequence produced a maximum reference-to-feedback error of {report['x_minus_050_shadow_execution']['controller_error_max_abs_rad']:.6f} rad. The final endpoint error was {report['x_minus_050_shadow_execution']['final_endpoint_error_max_abs_rad']:.6f} rad. This candidate is therefore rejected, not promoted.

The nominal identity world remains authoritative for software simulation only. Its full physics replay was safely aborted after a measured pre-goal drift of {report['nominal_world']['initial_drift_max_rad']:.6f} rad and a 12-second clock probe that advanced only {report['nominal_world']['clock_probe_s']:.2f} s, consistent with a real wall-contact/physics slowdown. The nominal world was not altered to hide that finding.

## Claim fence

- Collision: `adaptive_discrete_interpolation`; not strict continuous CCD.
- Exact articulated self-CCD: `not_available`.
- Clearance: `not_available`.
- Physical installation alignment: `UNVERIFIED`.
- Hardware safety: `UNVERIFIED`.
- Promotion: `NO_PROMOTION`.

## Evidence

- `outputs/STAGE5B_SHADOW/STAGE5B_NOMINAL_WORLD_AUTHORITY.json` / `.md`
- `outputs/STAGE5B_SHADOW/STAGE5B_EXECUTION_AUDIT.json`
- `outputs/STAGE5B_SHADOW/runtime_x_minus_050/stage5a_action_execution.json`
- `outputs/STAGE5B_SHADOW/runtime_x_minus_050/stage5a_controller_state_raw.jsonl`
- `outputs/STAGE5B_SHADOW/runtime_x_minus_050/stage5a_joint_states_raw.jsonl`
- `outputs/STAGE5B_SHADOW/STAGE5B_NOMINAL_REPLAY_BLOCKER.json`
- `outputs/STAGE5B_SHADOW/branch_graph_search_dense_large_detours/STAGE5B_BRANCH_GRAPH_SEARCH.json`

## Stage 4B targets

1. Fixed-endpoint task-space/IK route through wp66 -> wp67 with adjacent joint step <=20 degrees.
2. Jointly regenerated q/dq/ddq/timestamps that remain compatible with native JTC interpolation.
3. Authoritative installation transform plus real collision/clearance backend before physical claims.
"""
    (out / "STAGE5B_FINAL_HANDOFF.md").write_text(md, encoding="utf-8")
    print(json.dumps({"status": report["status"], "json": str(json_path), "markdown": str(out / "STAGE5B_FINAL_HANDOFF.md"), "campaigns": len(campaigns)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
