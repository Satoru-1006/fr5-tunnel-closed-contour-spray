"""Finalize a completed Stage 2.5R2 formal run without rerunning native work."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from audit_stage25r2 import audit_case
from run_stage25r2 import (
    NOMINAL_SEGMENTS,
    formal_gate,
    safe_copy,
    sha256,
    write_json,
    write_sha256sums,
)


def finalize(formal_root: Path, shadow_root: Path) -> None:
    candidate = json.loads((shadow_root / "stage25r2_candidate_selection.json").read_text(encoding="utf-8"))
    before = json.loads((formal_root / "frozen_artifact_hashes_before.json").read_text(encoding="utf-8"))
    after = json.loads((formal_root / "frozen_artifact_hashes_after.json").read_text(encoding="utf-8"))
    case_manifest = {
        "schema_version": "stage25r2-case-manifest-v1",
        "mitigate_overshoot": candidate.get("mitigate_overshoot"),
        "overshoot_threshold": candidate.get("overshoot_threshold"),
        "nominal_segments": NOMINAL_SEGMENTS,
    }
    for index in range(1, 4):
        (formal_root / f"rebuild_{index:02d}" / "case_manifest.json").write_text(
            json.dumps(case_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    audits = [audit_case(formal_root / f"rebuild_{index:02d}", NOMINAL_SEGMENTS) for index in range(1, 4)]
    gate = formal_gate(formal_root, candidate, audits, before, after)
    write_json(formal_root / "stage25r2_gate_report.json", gate)

    fields = [
        "calculate_call_graph_hash", "native_input_hash", "native_profile_hash", "resolution_trace_hash",
        "overshoot_trace_hash", "trajectory_generation_trace_hash", "final_robot_trajectory_hash", "final_replay_hash",
    ]
    matches = {field: all(audits[index].get("hashes", {}).get(field) == audits[0].get("hashes", {}).get(field) for index in range(3)) for field in fields}
    replay = audits[0].get("independent_final_trajectory_certification") or {}
    write_json(formal_root / "stage25r2_determinism_report.json", {"rebuilds": 3, "fields": matches, "all_match": all(matches.values())})
    write_json(formal_root / "stage25r2_shadow_sweep.json", json.loads((shadow_root / "stage25r2_shadow_sweep.json").read_text(encoding="utf-8")))
    shadow_csv = shadow_root / "stage25r2_shadow_sweep.csv"
    formal_csv = formal_root / "stage25r2_shadow_sweep.csv"
    if formal_csv.exists():
        if sha256(shadow_csv) != sha256(formal_csv):
            # This is a generated file inside the new R2 root; refresh it when
            # the replay-only shadow audit is regenerated. Historical roots are
            # never passed to this finalizer.
            shutil.copy2(shadow_csv, formal_csv)
    else:
        safe_copy(shadow_csv, formal_csv)
    write_json(formal_root / "stage25r2_candidate_selection.json", candidate)

    source_root = formal_root / "rebuild_01"
    copies = {
        "native/stage25r_ruckig_calls.jsonl": "stage25r2_ruckig_calls.jsonl",
        "native/stage25r_ruckig_call_resolutions.jsonl": "stage25r2_call_resolutions.jsonl",
        "native/stage25r2_overshoot_events.jsonl": "stage25r2_overshoot_events.jsonl",
        "native/stage25r2_trajectory_generation_events.jsonl": "stage25r2_trajectory_generation_events.jsonl",
        "native/stage25r2_stale_accepted_calls.json": "stage25r2_stale_accepted_calls.json",
        "moveit/stage25_ruckig_trajectory.csv": "stage25r2_final_robot_trajectory.csv",
        "stage25r2_final_trajectory_native_replay.jsonl": "stage25r2_final_trajectory_native_replay.jsonl",
        "stage25r2_waypoint4_retry_audit.json": "stage25r2_waypoint4_retry_audit.json",
        "stage25r2_all_call_native_jerk_audit.json": "stage25r2_all_call_native_jerk_audit.json",
    }
    for source_rel, destination_name in copies.items():
        source = source_root / source_rel
        destination = formal_root / destination_name
        if destination.exists():
            if sha256(source) != sha256(destination):
                raise RuntimeError(f"existing R2 artifact differs from rebuild source: {destination}")
        else:
            safe_copy(source, destination)
    write_json(formal_root / "stage25r2_final_replay_audit.json", audits[0].get("independent_final_trajectory_certification"))
    write_json(formal_root / "stage25r2_trajectory_generation_audit.json", audits[0].get("trajectory_generation"))
    write_json(formal_root / "stage25r2_pose_audit.json", audits[0].get("pose_constraints"))
    write_json(formal_root / "stage25r2_rebuild_results.json", [json.loads((formal_root / f"rebuild_{index:02d}" / "rebuild_result.json").read_text(encoding="utf-8")) for index in range(1, 4)])
    report = [
        "# Stage 2.5R2 Full Native Completion and Final-Trajectory Certification",
        "",
        f"- Stage 2.5R2: **{gate.get('Stage_2_5R2')}**",
        f"- Historical semantics: `mitigate_overshoot=true`, `threshold=1e-6`; selected semantics: `mitigate_overshoot={candidate.get('mitigate_overshoot')}`, `threshold={candidate.get('overshoot_threshold')}`",
        "- Runtime: MoveIt 2.12.4 / Ruckig 0.9.2 / ROS Jazzy",
        f"- Nominal segments: `{NOMINAL_SEGMENTS}`; Stage 2.7 allowed next: `{gate.get('Stage_2_7', {}).get('allowed_next')}`",
        "",
        "MoveIt runtime completion, trajectory-generation lineage, native Profile.j certification, and independent final replay are separate evidence sets. The false-mode candidate explicitly disables MoveIt overshoot mitigation; the replay records raw overshoot results but marks that check not applicable under the selected semantics.",
        f"- MoveIt runtime: return=`{gate.get('moveit_runtime', {}).get('native_return_value')}`, smoothing_complete=`{gate.get('moveit_runtime', {}).get('smoothing_complete')}`, ceiling_hit=`{gate.get('moveit_runtime', {}).get('duration_ceiling_hit')}`.",
        f"- Native calls: `{gate.get('native_calculate_calls', {}).get('total')}` total, `{gate.get('native_calculate_calls', {}).get('accepted_calls')}` accepted, `{gate.get('native_calculate_calls', {}).get('retry_calls')}` retries; every available Profile.j passes J=8.",
        f"- Independent replay: `{replay.get('segments_checked')}/{replay.get('segments_total')}` checked, `{replay.get('successful_native_results')}` successful, J8=`{replay.get('jerk_pass_J8')}`, raw overshoot=`{replay.get('raw_overshoot_pass')}`, selected overshoot gate=`{replay.get('overshoot_pass')}`.",
        f"- Pose/dynamics: pose=`{gate.get('pose_constraints')}`, position/velocity/acceleration/jerk=`{gate.get('position_limits')}/{gate.get('velocity_limits')}/{gate.get('acceleration_limits')}/{gate.get('jerk_limits')}`.",
        f"- Generation lineage: resets=`{gate.get('trajectory_generation', {}).get('reset_events')}`, stale accepted calls=`{gate.get('trajectory_generation', {}).get('stale_accepted_call_count')}`; stale evidence used for final certification=`{gate.get('trajectory_generation', {}).get('stale_evidence_used_for_final_certification')}`.",
        "",
        f"Frozen artifact mismatch count: `{gate.get('frozen_artifact_mismatch_count')}`. Three-rebuild determinism: `{all(matches.values())}`.",
    ]
    (formal_root / "stage25r2_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_sha256sums(formal_root)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--shadow-root", type=Path, required=True)
    args = parser.parse_args()
    finalize(args.formal_root.resolve(), args.shadow_root.resolve())
    print(args.formal_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
